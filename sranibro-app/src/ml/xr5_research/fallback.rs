//! One-way GPU -> CPU recovery, without overlapping child processes or retry loops.
use super::{Bridge, Process, Settings, Snapshot};
use std::{io, sync::{Arc, Weak}, time::{Duration, Instant}};

pub(crate) trait Worker: Sized {
    type Drain;
    fn start(settings: &Settings) -> io::Result<Self>;
    fn tick(&mut self, state: &mut Snapshot);
    fn failed(&self) -> bool;
    fn available(&self) -> bool;
    fn submit(&mut self, left: &[u8], right: &[u8], at: Instant, set_open: bool) -> bool;
    fn stop(self) -> Self::Drain;
    fn drained(drain: &Self::Drain) -> bool;
}
impl Worker for Bridge {
    type Drain = Weak<Process>;
    fn start(settings: &Settings) -> io::Result<Self> { Bridge::start(settings) }
    fn tick(&mut self, state: &mut Snapshot) { Bridge::tick(self, state); }
    fn failed(&self) -> bool { self.failed }
    fn available(&self) -> bool { Bridge::available(self) }
    fn submit(&mut self, left: &[u8], right: &[u8], at: Instant, set_open: bool) -> bool {
        Bridge::submit(self, left, right, at, set_open)
    }
    fn stop(self) -> Self::Drain {
        let drain = Arc::downgrade(&self.process);
        drop(self); // kills the child; the pipe owner releases the lease after reaping it
        drain
    }
    fn drained(drain: &Self::Drain) -> bool {
        drain.upgrade().is_none() && !super::OCCUPIED.load(std::sync::atomic::Ordering::Acquire)
    }
}

pub(crate) type AutoFallbackBridge = ManagedBridge<Bridge>;
pub(crate) struct ManagedBridge<W: Worker> {
    worker: Option<W>,
    cpu_fallback: Option<Settings>,
    draining: Option<(Instant, W::Drain, Settings)>,
}
impl<W: Worker> ManagedBridge<W> {
    pub fn start(settings: &Settings) -> io::Result<Self> {
        let mut cpu = settings.clone();
        cpu.gpu = false;
        match W::start(settings) {
            Ok(worker) => Ok(Self {
                worker: Some(worker),
                cpu_fallback: settings.gpu.then_some(cpu),
                draining: None,
            }),
            Err(error) if settings.gpu => {
                eprintln!("[xr5-model] GPU startup failed; trying CPU: {error}");
                Ok(Self { worker: Some(W::start(&cpu)?), cpu_fallback: None, draining: None })
            }
            Err(error) => Err(error),
        }
    }
    pub fn tick(&mut self, state: &mut Snapshot) { self.tick_at(state, Instant::now()); }
    fn tick_at(&mut self, state: &mut Snapshot, now: Instant) {
        if let Some(worker) = self.worker.as_mut() {
            worker.tick(state);
            if worker.failed() {
                if let Some(cpu) = self.cpu_fallback.take() {
                    eprintln!("[xr5-model] GPU failed; switching to CPU: {}", state.message);
                    let drain = self.worker.take().unwrap().stop();
                    self.draining = Some((now, drain, cpu));
                    state.captured_at = None;
                    state.message = "Starting CPU fallback".into();
                }
            }
        }
        let Some((started, drain, _)) = self.draining.as_ref() else { return; };
        if !W::drained(drain) {
            if now.saturating_duration_since(*started) >= Duration::from_secs(3) {
                self.draining = None;
                state.captured_at = None;
                state.message = "GPU worker did not stop. Reload to retry.".into();
            }
            return;
        }
        let (_, _, cpu) = self.draining.take().unwrap();
        match W::start(&cpu) {
            Ok(worker) => self.worker = Some(worker),
            Err(error) => {
                state.captured_at = None;
                state.message = format!("CPU fallback failed: {error}");
            }
        }
    }
    pub fn available(&self) -> bool { self.worker.as_ref().is_some_and(Worker::available) }
    pub fn submit(&mut self, left: &[u8], right: &[u8], at: Instant, set_open: bool) -> bool {
        self.worker.as_mut().is_some_and(|w| w.submit(left, right, at, set_open))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    #[derive(Default)]
    struct Scenario { starts: Vec<bool>, fail_start: bool, fail_live: bool, drained: bool }
    thread_local! { static PLAN: RefCell<Scenario> = RefCell::default(); }
    struct Fake(bool, bool);
    impl Worker for Fake {
        type Drain = ();
        fn start(settings: &Settings) -> io::Result<Self> {
            PLAN.with(|p| {
                let mut p = p.borrow_mut(); p.starts.push(settings.gpu);
                if p.fail_start { p.fail_start = false; return Err(io::Error::other("startup failure")); }
                Ok(Self(settings.gpu, false))
            })
        }
        fn tick(&mut self, state: &mut Snapshot) {
            self.1 |= PLAN.with(|p| p.borrow().fail_live);
            if self.1 { state.captured_at = None; state.message = "worker failure".into(); }
            else { state.backend = if self.0 { "GPU" } else { "CPU" }; state.count += 1; }
        }
        fn failed(&self) -> bool { self.1 }
        fn available(&self) -> bool { !self.1 }
        fn submit(&mut self, _: &[u8], _: &[u8], _: Instant, _: bool) -> bool { !self.1 }
        fn stop(self) {}
        fn drained(_: &()) -> bool { PLAN.with(|p| p.borrow().drained) }
    }
    fn reset() { PLAN.with(|p| *p.borrow_mut() = Scenario { drained: true, ..Default::default() }); }
    fn settings(gpu: bool) -> Settings { Settings { gpu, ..Default::default() } }
    fn starts() -> Vec<bool> { PLAN.with(|p| p.borrow().starts.clone()) }
    #[test]
    fn gpu_start_failure_uses_cpu_once() {
        reset(); PLAN.with(|p| p.borrow_mut().fail_start = true);
        let mut b = ManagedBridge::<Fake>::start(&settings(true)).unwrap();
        let mut state = Snapshot::default(); b.tick(&mut state);
        assert_eq!(starts(), [true, false]); assert_eq!(state.backend, "CPU");
        PLAN.with(|p| p.borrow_mut().fail_live = true);
        for _ in 0..100 { b.tick(&mut state); }
        assert_eq!(starts(), [true, false]); assert!(!b.available());
    }
    #[test]
    fn runtime_failure_waits_for_shutdown_then_recovers_on_cpu() {
        reset(); let mut b = ManagedBridge::<Fake>::start(&settings(true)).unwrap();
        let mut state = Snapshot::default(); b.tick(&mut state);
        PLAN.with(|p| { let mut p=p.borrow_mut(); p.fail_live=true; p.drained=false; });
        b.tick(&mut state); assert_eq!(starts(), [true]); assert!(!b.available());
        assert!(state.captured_at.is_none());
        for _ in 0..10 { b.tick(&mut state); }
        assert_eq!(starts(), [true]);
        PLAN.with(|p| { let mut p=p.borrow_mut(); p.fail_live=false; p.drained=true; });
        b.tick(&mut state); b.tick(&mut state);
        assert_eq!(starts(), [true,false]); assert!(b.available()); assert_eq!(state.backend,"CPU");
        PLAN.with(|p| p.borrow_mut().fail_live=true);
        for _ in 0..100 { b.tick(&mut state); }
        assert_eq!(starts(), [true,false]); assert!(!b.available());
    }
    #[test]
    fn unresponsive_shutdown_never_starts_overlapping_cpu() {
        reset(); let mut b = ManagedBridge::<Fake>::start(&settings(true)).unwrap();
        let mut state = Snapshot::default(); let now=Instant::now();
        PLAN.with(|p| { let mut p=p.borrow_mut(); p.fail_live=true; p.drained=false; });
        b.tick_at(&mut state,now); b.tick_at(&mut state,now+Duration::from_secs(4));
        PLAN.with(|p| p.borrow_mut().drained=true);
        b.tick_at(&mut state,now+Duration::from_secs(5));
        assert_eq!(starts(),[true]); assert!(!b.available()); assert!(state.message.contains("did not stop"));
    }
    #[test]
    fn explicit_cpu_never_attempts_gpu_or_restarts_after_failure() {
        reset(); let mut b=ManagedBridge::<Fake>::start(&settings(false)).unwrap();
        PLAN.with(|p| p.borrow_mut().fail_live=true);
        for _ in 0..100 { b.tick(&mut Snapshot::default()); }
        assert_eq!(starts(),[false]); assert!(!b.available());
    }
    #[test]
    fn healthy_gpu_does_not_start_cpu() {
        reset(); let mut b=ManagedBridge::<Fake>::start(&settings(true)).unwrap();
        let mut state=Snapshot::default();
        for _ in 0..100 { b.tick(&mut state); }
        assert_eq!(starts(),[true]); assert_eq!(state.backend,"GPU"); assert!(b.available());
    }

    #[test]
    fn failed_cpu_start_is_terminal() {
        reset(); let mut b=ManagedBridge::<Fake>::start(&settings(true)).unwrap();
        PLAN.with(|p| { let mut p=p.borrow_mut(); p.fail_live=true; p.fail_start=true; });
        let mut state=Snapshot::default();
        for _ in 0..100 { b.tick(&mut state); }
        assert_eq!(starts(),[true,false]); assert!(!b.available());
        assert!(state.message.contains("CPU fallback failed"));
    }

    #[cfg(all(windows, feature = "external-eyelid-model"))]
    #[test]
    #[ignore = "Requires built app + native model + DirectML; no camera or app UI is opened"]
    fn real_native_gpu_exit_recovers_on_cpu_without_reloading_tracking() {
        let model=std::env::var("XR5_NATIVE_TEST_MODEL").expect("XR5_NATIVE_TEST_MODEL");
        assert!(std::env::var_os("XR5_NATIVE_TEST_EXE").is_some());
        let mut b=AutoFallbackBridge::start(&Settings { model_file:model, gpu:true, enabled:true, avatar:true, ..Default::default() }).unwrap();
        let pixels:Vec<u8>=(0..40000).map(|i| ((i*31+i/200*53)%256) as u8).collect();
        let mut state=Snapshot::default();
        let mut until=Instant::now()+Duration::from_secs(40);
        while state.count<1 {
            b.tick(&mut state);
            if b.available() { b.submit(&pixels,&pixels,Instant::now(),false); }
            assert!(Instant::now()<until,"GPU startup: {}",state.message);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(state.backend,"GPU","test needs actual GPU execution, not fallback");
        let before=state.count;
        let gpu=Arc::downgrade(&b.worker.as_ref().unwrap().process);
        b.worker.as_ref().unwrap().process.kill();
        until=Instant::now()+Duration::from_secs(40);
        while state.backend!="CPU" || state.count<=before {
            b.tick(&mut state);
            if b.available() { b.submit(&pixels,&pixels,Instant::now(),false); }
            assert!(Instant::now()<until,"CPU recovery: {}",state.message);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert!(gpu.upgrade().is_none(),"old GPU process owner must be gone before CPU runs");
        assert!(b.cpu_fallback.is_none());
        b.worker.as_ref().unwrap().process.kill();
        until=Instant::now()+Duration::from_secs(6);
        while !b.worker.as_ref().unwrap().failed {
            b.tick(&mut state);
            if b.available() { b.submit(&pixels,&pixels,Instant::now(),false); }
            assert!(Instant::now()<until,"CPU failure was not detected");
            std::thread::sleep(Duration::from_millis(2));
        }
        for _ in 0..50 { b.tick(&mut state); }
        assert!(!b.available()); assert!(b.draining.is_none());
        drop(b);
        until=Instant::now()+Duration::from_secs(5);
        while super::super::OCCUPIED.load(std::sync::atomic::Ordering::Acquire) {
            assert!(Instant::now()<until,"worker lease leaked");
            std::thread::sleep(Duration::from_millis(2));
        }
        eprintln!("Actual native GPU reply -> killed child -> CPU reply -> terminal CPU failure; no UI/HMD opened");
    }
}
