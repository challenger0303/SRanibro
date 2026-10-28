//! Opt-in research gaze worker. SRanibro alone owns the cameras.
use crate::{core::types::EyeResult, pipeline::Telemetry};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    io::{self, Read, Write},
    path::Path,
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

const AGE: Duration = Duration::from_millis(50);
#[derive(Clone, Default)]
pub struct Snapshot {
    pub message: String,
    pub backend: String,
    pub count: u64,
    pub states: [u8; 2],
    pub directions: [[f32; 3]; 2],
    pub captured: Option<Instant>,
}
impl Snapshot {
    pub fn apply(&self, eyes: &mut [EyeResult; 2], now: Instant, allow_partner: bool) {
        let fresh = self
            .captured
            .is_some_and(|t| now.checked_duration_since(t).is_some_and(|age| age <= AGE));
        for (i, e) in eyes.iter_mut().enumerate() {
            e.gaze_valid = fresh && (self.states[i] == 1 || (allow_partner && self.states[i] == 2));
            e.gaze_yoked = e.gaze_valid && self.states[i] == 2;
            if e.gaze_valid {
                // Worker OpenVR: +X right,+Y up,-Z forward. Existing sinks negate X.
                let [x, y, z] = self.directions[i];
                e.gaze = [-x, y, -z];
            } else {
                e.gaze = [0., 0., 1.];
            }
        }
    }
    pub fn raw_sample(&self, eyes: &[EyeResult; 2]) -> crate::core::types::GazeSample {
        use crate::core::types::{EyeSample, GazeSample};
        let eye = |e: &EyeResult| EyeSample {
            gaze: e.gaze,
            gaze_valid: e.gaze_valid,
            gaze_reported: true,
            ..Default::default()
        };
        GazeSample {
            left: eye(&eyes[0]),
            right: eye(&eyes[1]),
            ..Default::default()
        }
    }
}
#[derive(Deserialize)]
struct Package {
    schema: String,
    python: String,
    worker: String,
    worker_sha256: String,
    device_serial: String,
}
fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.into())
}
struct Process {
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    job: super::gpu_worker::WorkerJob,
}
impl Process {
    fn kill(&self) {
        #[cfg(windows)]
        self.job.terminate_tree();
        if let Ok(mut slot) = self.child.lock() {
            if let Some(c) = slot.as_mut() {
                let _ = c.kill();
            }
        }
    }
}
struct Job {
    seq: u64,
    gens: [u64; 2],
    times: [f64; 2],
    pixels: Vec<u8>,
}
enum Reply {
    Ready,
    Value(u64, u8, [u8; 2], [[f32; 3]; 2]),
    Error(String),
}
fn read_reply(r: &mut impl Read) -> io::Result<Reply> {
    let mut b = [0; 39];
    r.read_exact(&mut b)?;
    if &b[..4] != b"XGZ1" || b[12] > 1 {
        return Err(invalid("Invalid gaze reply"));
    }
    let mut states = [0; 2];
    let mut directions = [[0.; 3]; 2];
    for i in 0..2 {
        let at = 13 + i * 13;
        states[i] = b[at];
        for j in 0..3 {
            directions[i][j] =
                f32::from_le_bytes(b[at + 1 + j * 4..at + 5 + j * 4].try_into().unwrap());
        }
        let d = directions[i];
        let norm = d[0].hypot(d[1]).hypot(d[2]);
        if states[i] > 2
            || !d.iter().all(|v| v.is_finite())
            || (states[i] > 0 && (!(0.99..=1.01).contains(&norm) || d[2] >= 0.))
            || (states[i] == 0 && d != [0.; 3])
        {
            return Err(invalid("Invalid gaze direction"));
        }
    }
    if states == [2, 2] || (states[0] == 2 && states[1] != 1) || (states[1] == 2 && states[0] != 1)
    {
        return Err(invalid("Unsupported binocular estimate"));
    }
    Ok(Reply::Value(
        u64::from_le_bytes(b[4..12].try_into().unwrap()),
        b[12],
        states,
        directions,
    ))
}
pub struct Worker {
    stop: Arc<AtomicBool>,
    process: Arc<Process>,
    threads: Vec<thread::JoinHandle<()>>,
}
impl Worker {
    pub fn start(path: &Path, gpu: bool, tele: Arc<Telemetry>) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() > 65536 {
            return Err(invalid("Gaze package too large"));
        }
        let p: Package = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        if p.schema != "sranibro-xr5-gaze-local-v1" {
            return Err(invalid("Unsupported gaze package"));
        }
        let root = path
            .parent()
            .ok_or_else(|| invalid("Missing package directory"))?;
        let script = root.join(&p.worker);
        if Path::new(&p.worker).is_absolute()
            || script.canonicalize()?.parent() != Some(root.canonicalize()?.as_path())
        {
            return Err(invalid("Worker must be inside gaze package"));
        }
        if format!("{:x}", Sha256::digest(std::fs::read(&script)?)) != p.worker_sha256 {
            return Err(invalid("Gaze worker hash mismatch"));
        }
        #[cfg(windows)]
        if crate::device::usb::peek_serial().as_deref() != Some(p.device_serial.as_str()) {
            return Err(invalid("Gaze mapping belongs to a different XR5 device"));
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(crate::config::base_dir().join("xr5-gaze-worker.log"))?;
        let mut command = Command::new(&p.python);
        command
            .args(["-I", "-S", "-u"])
            .arg(script)
            .arg(path)
            .arg(if gpu { "auto" } else { "cpu" })
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        Self::supervise(command.spawn()?, tele)
    }
    fn supervise(mut child: Child, tele: Arc<Telemetry>) -> io::Result<Self> {
        #[cfg(windows)]
        let job = match super::gpu_worker::WorkerJob::attach(&child) {
            Ok(j) => j,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let (Some(mut stdin), Some(mut stdout)) = (child.stdin.take(), child.stdout.take()) else {
            let _ = child.kill();
            let _ = child.wait();
            return Err(invalid("Missing gaze worker pipes"));
        };
        let process = Arc::new(Process {
            child: Mutex::new(Some(child)),
            #[cfg(windows)]
            job,
        });
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::sync_channel::<Job>(1);
        let (replies, receiver) = mpsc::sync_channel(1);
        let owner = process.clone();
        let pipe = match thread::Builder::new()
            .name("xr5-gaze-pipe".into())
            .spawn(move || {
                let mut run = || -> io::Result<()> {
                    let mut magic = [0; 8];
                    stdout.read_exact(&mut magic)?;
                    if &magic != b"XRGZ0001" {
                        return Err(invalid("Gaze startup failed"));
                    }
                    replies.send(Reply::Ready).map_err(|_| invalid("Stopped"))?;
                    while let Ok(j) = rx.recv() {
                        stdin.write_all(&j.seq.to_le_bytes())?;
                        for n in j.gens {
                            stdin.write_all(&n.to_le_bytes())?;
                        }
                        for t in j.times {
                            stdin.write_all(&t.to_le_bytes())?;
                        }
                        stdin.write_all(&j.pixels)?;
                        stdin.flush()?;
                        let value = read_reply(&mut stdout)?;
                        replies.send(value).map_err(|_| invalid("Stopped"))?;
                    }
                    Ok(())
                };
                if let Err(e) = run() {
                    let _ = replies.try_send(Reply::Error(e.to_string()));
                }
                owner.kill();
                // Never wait for process/driver cleanup while holding the kill lock.
                let child = owner.child.lock().ok().and_then(|mut slot| slot.take());
                if let Some(mut c) = child {
                    let _ = c.wait();
                }
            }) {
            Ok(t) => t,
            Err(e) => {
                process.kill();
                return Err(e);
            }
        };
        tele.xr5_gaze.lock().unwrap().message = "Loading model gaze…".into();
        let stopped = stop.clone();
        let proc = process.clone();
        let supervisor = match thread::Builder::new()
            .name("xr5-gaze".into())
            .spawn(move || {
                let start = Instant::now();
                let mut ready = false;
                let mut pending: Option<(u64, Instant, Instant)> = None;
                let mut seq = 0u64;
                let mut gens = [0; 2];
                let mut failure = None;
                while !stopped.load(Ordering::Relaxed) {
                    match receiver.try_recv() {
                        Ok(Reply::Ready) => ready = true,
                        Ok(Reply::Error(e)) => {
                            failure = Some(e);
                            break;
                        }
                        Ok(Reply::Value(id, backend, states, directions)) => {
                            if let Some((expected, _, captured)) = pending.take() {
                                if id != expected {
                                    failure = Some("Mismatched gaze reply".into());
                                    break;
                                }
                                let mut s = tele.xr5_gaze.lock().unwrap();
                                s.captured = Some(captured);
                                s.states = states;
                                s.directions = directions;
                                s.count += 1;
                                tele.c_gaze.fetch_add(1, Ordering::Relaxed);
                                s.backend = if backend == 1 { "GPU" } else { "CPU" }.into();
                                s.message = "Model gaze · experimental".into();
                            } else {
                                failure = Some("Unexpected gaze reply".into());
                                break;
                            }
                        }
                        Err(mpsc::TryRecvError::Disconnected) => {
                            failure = Some("Gaze worker stopped".into());
                            break;
                        }
                        Err(mpsc::TryRecvError::Empty) => {}
                    }
                    if (!ready && start.elapsed() > Duration::from_secs(60))
                        || pending.is_some_and(|(_, at, _)| at.elapsed() > Duration::from_secs(2))
                    {
                        failure = Some("Gaze worker timeout — reload to retry".into());
                        break;
                    }
                    if ready && pending.is_none() {
                        let pair = tele.xr5_raw_frames.lock().unwrap().clone();
                        if let [Some(l), Some(r)] = pair {
                            let now = Instant::now();
                            let captured = l.at.min(r.at);
                            let skew = l.at.max(r.at).duration_since(captured);
                            if l.generation > gens[0]
                                && r.generation > gens[1]
                                && now.duration_since(captured) <= AGE
                                && skew <= Duration::from_millis(20)
                                && l.pixels.len() == 40000
                                && r.pixels.len() == 40000
                            {
                                let mut pixels = Vec::with_capacity(80000);
                                pixels.extend_from_slice(&l.pixels);
                                pixels.extend_from_slice(&r.pixels);
                                seq += 1;
                                let next = [l.generation, r.generation];
                                let times = [
                                    l.at.saturating_duration_since(start).as_secs_f64(),
                                    r.at.saturating_duration_since(start).as_secs_f64(),
                                ];
                                if tx
                                    .try_send(Job {
                                        seq,
                                        gens: next,
                                        times,
                                        pixels,
                                    })
                                    .is_err()
                                {
                                    failure = Some("Gaze worker queue unavailable".into());
                                    break;
                                }
                                gens = next;
                                pending = Some((seq, now, captured));
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(1));
                }
                {
                    let mut s = tele.xr5_gaze.lock().unwrap();
                    s.captured = None;
                    s.states = [0; 2];
                    s.message = failure.unwrap_or_else(|| "Model gaze stopped".into());
                }
                proc.kill();
            }) {
            Ok(t) => t,
            Err(e) => {
                process.kill();
                let _ = pipe.join();
                return Err(e);
            }
        };
        Ok(Self {
            stop,
            process,
            threads: vec![supervisor, pipe],
        })
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.process.kill();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn packet(states: [u8; 2], dirs: [[f32; 3]; 2]) -> Vec<u8> {
        let mut b = b"XGZ1".to_vec();
        b.extend(1u64.to_le_bytes());
        b.push(0);
        for i in 0..2 {
            b.push(states[i]);
            for f in dirs[i] {
                b.extend(f.to_le_bytes());
            }
        }
        b
    }
    #[test]
    fn accepts_independent() {
        assert!(read_reply(&mut &packet([1, 1], [[0., 0., -1.]; 2])[..]).is_ok());
    }
    #[test]
    fn rejects_invalid_numbers() {
        for bad in [f32::NAN, f32::INFINITY, 2.] {
            assert!(read_reply(&mut &packet([1, 1], [[bad, 0., -1.], [0., 0., -1.]])[..]).is_err());
        }
    }
    #[test]
    fn rejects_unsupported_estimates() {
        for s in [[2, 2], [0, 2], [2, 0]] {
            assert!(read_reply(&mut &packet(s, [[0., 0., -1.]; 2])[..]).is_err());
        }
    }
    #[test]
    fn rejects_truncated() {
        assert!(read_reply(&mut &b"XGZ1"[..]).is_err());
    }
    #[test]
    fn preserves_expressions_and_axes() {
        use crate::core::types::Eye;
        let now = Instant::now();
        let mut eyes = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        eyes[0].openness = 0.2;
        eyes[0].wide = 0.7;
        eyes[0].blink = true;
        eyes[0].squeeze = 0.3;
        let s = Snapshot {
            states: [1, 2],
            directions: [[0.6, 0., -0.8], [-0.6, 0., -0.8]],
            captured: Some(now),
            ..Default::default()
        };
        s.apply(&mut eyes, now, true);
        assert_eq!(eyes[0].gaze, [-0.6, 0., 0.8]);
        assert_eq!(eyes[1].gaze, [0.6, 0., 0.8]);
        assert!(eyes[0].blink && !eyes[0].gaze_yoked && eyes[1].gaze_yoked);
        assert_eq!(
            (eyes[0].openness, eyes[0].wide, eyes[0].squeeze),
            (0.2, 0.7, 0.3)
        );
        s.apply(&mut eyes, now, false);
        assert!(!eyes[1].gaze_valid && eyes[0].gaze_valid);
    }
    #[test]
    fn stale_future_and_missing_never_emit_tobii() {
        use crate::core::types::Eye;
        let now = Instant::now();
        for captured in [
            None,
            Some(now - Duration::from_millis(51)),
            Some(now + Duration::from_millis(1)),
        ] {
            let mut eyes = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
            for e in &mut eyes {
                e.gaze_valid = true;
                e.gaze = [0.6, 0., 0.8];
            }
            Snapshot {
                states: [1, 1],
                directions: [[0., 0., -1.]; 2],
                captured,
                ..Default::default()
            }
            .apply(&mut eyes, now, true);
            assert!(eyes.iter().all(|e| !e.gaze_valid && e.gaze == [0., 0., 1.]));
        }
    }
    fn fixture(code: &str) -> (Worker, Arc<Telemetry>) {
        let python =
            std::env::var("SRANIBRO_TEST_PYTHON").expect("explicit local test Python required");
        let mut cmd = Command::new(python);
        cmd.args(["-I", "-S", "-u", "-c", code])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let tele = Telemetry::new(false, false, false, &Default::default());
        (
            Worker::supervise(cmd.spawn().unwrap(), tele.clone()).unwrap(),
            tele,
        )
    }
    fn wait(mut condition: impl FnMut() -> bool) {
        let start = Instant::now();
        while !condition() {
            assert!(start.elapsed() < Duration::from_secs(5));
            thread::sleep(Duration::from_millis(2));
        }
    }
    fn frames(tele: &Telemetry, n: u64) {
        let at = Instant::now();
        let frame = crate::ml::xr5_research::RawFrame {
            generation: n,
            at,
            pixels: vec![0; 40000].into(),
        };
        *tele.xr5_raw_frames.lock().unwrap() = [Some(frame.clone()), Some(frame)];
    }
    #[test]
    #[ignore = "requires explicit SRANIBRO_TEST_PYTHON; no hardware"]
    fn lifecycle_repeated_drop_while_startup_blocked() {
        for _ in 0..3 {
            let (worker, _) = fixture("import time;time.sleep(60)");
            let start = Instant::now();
            drop(worker);
            assert!(start.elapsed() < Duration::from_secs(2));
        }
    }
    #[test]
    #[ignore = "requires explicit SRANIBRO_TEST_PYTHON; no hardware"]
    fn lifecycle_bad_startup_fails_closed() {
        let (worker, tele) =
            fixture("import sys;sys.stdout.buffer.write(b'BADSTART');sys.stdout.flush()");
        wait(|| {
            tele.xr5_gaze
                .lock()
                .unwrap()
                .message
                .contains("startup failed")
        });
        assert!(tele.xr5_gaze.lock().unwrap().captured.is_none());
        drop(worker);
    }
    #[test]
    #[ignore = "requires explicit SRANIBRO_TEST_PYTHON; no hardware"]
    fn lifecycle_frame_reply_then_hang_times_out() {
        let code="import sys,struct,time\nr=sys.stdin.buffer;w=sys.stdout.buffer\nw.write(b'XRGZ0001');w.flush()\nh=r.read(40);r.read(80000)\nn=struct.unpack('<Q',h[:8])[0]\nw.write(struct.pack('<4sQB',b'XGZ1',n,0)+struct.pack('<Bfff',1,0.,0.,-1.)*2);w.flush()\nr.read(40);r.read(80000);time.sleep(60)";
        let (worker, tele) = fixture(code);
        // Keep only newest test frame fresh while the child starts.
        let mut n = 0;
        wait(|| {
            n += 1;
            frames(&tele, n);
            tele.xr5_gaze.lock().unwrap().count == 1
        });
        frames(&tele, n + 1);
        wait(|| tele.xr5_gaze.lock().unwrap().message.contains("timeout"));
        assert!(tele.xr5_gaze.lock().unwrap().captured.is_none());
        assert_eq!(tele.xr5_gaze.lock().unwrap().count, 1);
        drop(worker);
    }

    #[test]
    #[ignore = "explicit local package and capture required; no camera access"]
    fn real_worker_recorded_input_and_teardown() {
        use crate::core::types::Eye;
        let path = std::env::var("SRANIBRO_TEST_GAZE_PACKAGE").expect("local gaze package");
        let capture = std::path::PathBuf::from(
            std::env::var("SRANIBRO_TEST_CAPTURE").expect("offline capture"),
        );
        let p: Package = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let mut images = Vec::new();
        for line in std::fs::read_to_string(capture.join("events.jsonl"))
            .unwrap()
            .lines()
        {
            let r: serde_json::Value = serde_json::from_str(line).unwrap();
            if r["kind"] == "frame" {
                let bytes = std::fs::read(capture.join(r["image"].as_str().unwrap())).unwrap();
                assert_eq!(&bytes[..15], b"P5\n200 200\n255\n");
                assert_eq!(bytes.len(), 40015);
                images.push(Arc::<[u8]>::from(bytes[15..].to_vec()));
                if images.len() == 2 {
                    break;
                }
            }
        }
        assert_eq!(images.len(), 2);
        let mut cmd = Command::new(p.python);
        cmd.args(["-I", "-S", "-u"])
            .arg(Path::new(&path).parent().unwrap().join(p.worker))
            .arg(path)
            .arg("auto")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000);
        }
        let tele = Telemetry::new(false, false, false, &Default::default());
        let worker = Worker::supervise(cmd.spawn().unwrap(), tele.clone()).unwrap();
        let mut n = 0;
        wait(|| {
            n += 1;
            let at = Instant::now();
            *tele.xr5_raw_frames.lock().unwrap() = std::array::from_fn(|i| {
                Some(crate::ml::xr5_research::RawFrame {
                    generation: n,
                    at,
                    pixels: images[i].clone(),
                })
            });
            tele.xr5_gaze.lock().unwrap().count >= 4
        });
        assert_eq!(tele.xr5_gaze.lock().unwrap().backend, "GPU");
        assert_eq!(tele.xr5_gaze.lock().unwrap().states, [1, 1]);
        *tele.xr5_raw_frames.lock().unwrap() = [None, None];
        thread::sleep(Duration::from_millis(75));
        let mut eyes = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        tele.xr5_gaze
            .lock()
            .unwrap()
            .apply(&mut eyes, Instant::now(), true);
        assert!(eyes.iter().all(|e| !e.gaze_valid));
        let start = Instant::now();
        drop(worker);
        assert!(start.elapsed() < Duration::from_secs(2));
        assert!(tele.xr5_gaze.lock().unwrap().captured.is_none());
    }
}
