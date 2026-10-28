//! Read-only PSVR2Toolkit acquisition, isolated from the UI and inference process.
//! Native initialization, reads and destruction may block or crash. A supervised
//! child owns that entire lifetime; only copied, validated data crosses the pipe.
//! Camera A/B and native gaze mapping remain the recorded PSVR2 hardware mapping.
#![cfg(any(windows, target_os = "linux"))]

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use super::psvr2_capi::{GazeStatus, GazeVec3, WearableEye, CAMERA_HEIGHT, CAMERA_WIDTH};
use super::psvr2_worker::{StreamKind, WorkerClient, WorkerEvent};
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye, EyeSample, GazeSample};

const START_TIMEOUT: Duration = Duration::from_secs(20);
const SESSION_TIMEOUT: Duration = Duration::from_secs(8);
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(3);
const FRESH_FOR: Duration = Duration::from_millis(1500);
const STALE_RESTART: Duration = Duration::from_secs(5);
const INACTIVE_GRACE: Duration = Duration::from_secs(1);
const STABLE_RESET: Duration = Duration::from_secs(30);
const STOP_TIMEOUT: Duration = Duration::from_millis(1500);
const POLL_INTERVAL: Duration = Duration::from_millis(20);
const MAX_SESSIONS: u32 = 4;
const MAX_STREAM_ERRORS: u32 = 30;

static SUPERVISOR_OCCUPIED: OnceLock<Arc<AtomicBool>> = OnceLock::new();
fn supervisor_slot() -> &'static Arc<AtomicBool> {
    SUPERVISOR_OCCUPIED.get_or_init(|| Arc::new(AtomicBool::new(false)))
}

/// Also excludes replacement if an application callback, rather than native
/// code, gets stuck. Detaching a timed-out thread must never create two owners.
struct SupervisorLease(Arc<AtomicBool>);
impl SupervisorLease {
    fn acquire(slot: &Arc<AtomicBool>) -> io::Result<Self> {
        slot.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                io::Error::new(io::ErrorKind::WouldBlock,
                "The previous PSVR2 session is still stopping; another session cannot start yet.")
            })?;
        Ok(Self(slot.clone()))
    }
}
impl Drop for SupervisorLease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

fn join_with_budget(handle: JoinHandle<()>, budget: Duration) -> bool {
    let deadline = Instant::now() + budget;
    while !handle.is_finished() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return false;
        }
        thread::sleep(remaining.min(Duration::from_millis(5)));
    }
    let _ = handle.join();
    true
}

fn publish_if_running(stop: &AtomicBool, publish: impl FnOnce()) -> bool {
    if stop.load(Ordering::Acquire) {
        return false;
    }
    publish();
    true
}

fn set_status(status: &Arc<Mutex<String>>, message: impl Into<String>) {
    let message = message.into();
    if let Ok(mut value) = status.lock() {
        if *value != message {
            *value = message;
        }
    }
}

pub struct Psvr2Adapter {
    profile: DeviceProfile,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}
impl Default for Psvr2Adapter {
    fn default() -> Self {
        Self::new()
    }
}
impl Psvr2Adapter {
    pub fn new() -> Self {
        Self {
            profile: DeviceProfile {
                name: "PlayStation VR2".into(),
                ml_device: "vr4".into(),
                image_w: CAMERA_WIDTH as u32,
                image_h: CAMERA_HEIGHT as u32,
                slot_a_eye: Eye::Left,
                transport: "PSVR2Toolkit CAPI".into(),
                streams: "native gaze + paired 400x200 eye-camera preview".into(),
                gaze_src: "PSVR2 gaze · pupil · blink".into(),
            },
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }
}

/// A cached first frame is not proof that the producer is alive. Require two
/// distinct native updates, and age each stream independently after that.
struct StreamEvidence<K> {
    key: Option<K>,
    last_fresh: Option<Instant>,
    confirmations: u8,
    errors: u32,
}
impl<K: PartialEq + Copy> StreamEvidence<K> {
    fn new() -> Self {
        Self {
            key: None,
            last_fresh: None,
            confirmations: 0,
            errors: 0,
        }
    }
    fn observe(&mut self, key: K, now: Instant) -> bool {
        if self.key == Some(key) {
            return false;
        }
        self.key = Some(key);
        self.last_fresh = Some(now);
        self.confirmations = self.confirmations.saturating_add(1).min(2);
        self.errors = 0;
        self.confirmations >= 2
    }
    fn invalidate(&mut self) {
        // Retain the old sequence key: replaying a cached sample after a driver
        // transition must not count as a new update.
        self.last_fresh = None;
        self.confirmations = 0;
    }
    fn fresh(&self, now: Instant) -> bool {
        self.confirmations >= 2
            && self
                .last_fresh
                .is_some_and(|t| now.duration_since(t) <= FRESH_FOR)
    }
    fn failed(&mut self) {
        self.errors = self.errors.saturating_add(1);
    }
}

struct SessionHealth {
    started: Instant,
    initialized: Option<Instant>,
    last_driver_heartbeat: Option<Instant>,
    active: bool,
    active_since: Option<Instant>,
    inactive_since: Option<Instant>,
    gaze: StreamEvidence<(i64, u32)>,
    image: StreamEvidence<u32>,
    ever_streaming: bool,
    stable_since: Option<Instant>,
}
impl SessionHealth {
    fn new(now: Instant) -> Self {
        Self {
            started: now,
            initialized: None,
            last_driver_heartbeat: None,
            active: false,
            active_since: None,
            inactive_since: None,
            gaze: StreamEvidence::new(),
            image: StreamEvidence::new(),
            ever_streaming: false,
            stable_since: None,
        }
    }
    fn driver(&mut self, active: bool, now: Instant) {
        self.last_driver_heartbeat = Some(now);
        if !active {
            if self.active {
                self.gaze.invalidate();
                self.image.invalidate();
            }
            self.inactive_since.get_or_insert(now);
            self.stable_since = None;
        } else {
            self.inactive_since = None;
            if !self.active {
                self.active_since = Some(now);
            }
        }
        self.active = active;
    }
    fn streams(&self, now: Instant) -> (bool, bool) {
        (
            self.active && self.gaze.fresh(now),
            self.active && self.image.fresh(now),
        )
    }
    fn update_stability(&mut self, now: Instant) -> bool {
        if self.streams(now) == (true, true) {
            self.ever_streaming = true;
            let since = self.stable_since.get_or_insert(now);
            now.duration_since(*since) >= STABLE_RESET
        } else {
            self.stable_since = None;
            false
        }
    }
    fn failure(&self, now: Instant) -> Option<&'static str> {
        let Some(initialized) = self.initialized else {
            return (now.duration_since(self.started) >= SESSION_TIMEOUT)
                .then_some("PSVR2Toolkit initialization did not respond");
        };
        if now.duration_since(self.last_driver_heartbeat.unwrap_or(initialized))
            >= HEARTBEAT_TIMEOUT
        {
            return Some("PSVR2Toolkit driver heartbeat stopped responding");
        }
        if self
            .inactive_since
            .is_some_and(|t| now.duration_since(t) >= INACTIVE_GRACE)
        {
            return Some("PSVR2Toolkit SteamVR driver is inactive");
        }
        if self.gaze.errors >= MAX_STREAM_ERRORS {
            return Some("PSVR2Toolkit repeatedly returned invalid gaze data");
        }
        if self.image.errors >= MAX_STREAM_ERRORS {
            return Some("PSVR2Toolkit repeatedly returned invalid eye-camera data");
        }
        if !self.ever_streaming {
            return (now.duration_since(initialized) >= SESSION_TIMEOUT)
                .then_some(match self.streams(now) {
                (true, false) => "No fresh PSVR2 eye images; check Toolkit camera access",
                (false, true) => "No fresh PSVR2 gaze data",
                _ => "No fresh PSVR2 gaze and eye images; check SteamVR and Toolkit camera access",
            });
        }
        if self.active
            && self
                .gaze
                .last_fresh
                .or(self.active_since)
                .is_some_and(|t| now.duration_since(t) >= STALE_RESTART)
        {
            return Some("PSVR2 gaze updates stopped");
        }
        if self.active
            && self
                .image
                .last_fresh
                .or(self.active_since)
                .is_some_and(|t| now.duration_since(t) >= STALE_RESTART)
        {
            return Some("PSVR2 eye-camera updates stopped");
        }
        None
    }
    fn description(&self, now: Instant) -> &'static str {
        if !self.active {
            return "waiting for PSVR2Toolkit SteamVR driver";
        }
        match self.streams(now) {
            (true, true) => "streaming gaze + eye images at native rate",
            (true, false) => "gaze live · waiting for fresh eye images",
            (false, true) => "eye images live · waiting for fresh gaze",
            (false, false) => "waiting for fresh gaze and eye images",
        }
    }
}

fn finite_vec3(value: GazeVec3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn eye_sample(native: WearableEye) -> EyeSample {
    let blink_reported = native.blink_valid();
    let blink = native.blink != 0;
    EyeSample {
        gaze: [
            native.gaze_dir_norm.x,
            native.gaze_dir_norm.y,
            native.gaze_dir_norm.z,
        ],
        gaze_valid: native.gaze_valid(),
        // The status block always contains an explicit per-eye gaze validity bit.
        gaze_reported: true,
        origin_mm: [
            native.gaze_origin_mm.x,
            native.gaze_origin_mm.y,
            native.gaze_origin_mm.z,
        ],
        origin_valid: native.is_gaze_origin_valid != 0 && finite_vec3(native.gaze_origin_mm),
        pupil_mm: native.pupil_dia_mm,
        pupil_valid: native.pupil_valid(),
        pupil_pos: [
            native.pupil_pos_in_sensor_area.x,
            native.pupil_pos_in_sensor_area.y,
        ],
        pupil_pos_valid: native.pupil_position_valid(),
        // Invalidity is explicit and must clear a stale pupil position.
        pupil_pos_reported: true,
        // PSVR2 exposes a binary native blink classifier, not continuous
        // openness. Feed it only as the absolute open/closed classifier used by
        // SRanibro's blink latch. Camera ML remains the continuous eyelid signal.
        openness: if blink { 0.0 } else { 1.0 },
        openness_valid: blink_reported && !blink,
        openness_reported: blink_reported,
    }
}

fn gaze_sample(status: GazeStatus) -> GazeSample {
    GazeSample {
        timestamp_us: status.wearable.timestamp.max(0) as u64,
        left: eye_sample(status.wearable.left),
        right: eye_sample(status.wearable.right),
    }
}

fn report_failure(
    ready: &mut Option<mpsc::SyncSender<io::Result<()>>>,
    status: &Arc<Mutex<String>>,
    reason: &str,
    can_reload: bool,
) {
    let action = if can_reload {
        " Press Reload to try again."
    } else {
        ""
    };
    let message = format!("{reason}; automatic retries paused.{action}");
    eprintln!("[psvr2] {message}");
    set_status(status, &message);
    if let Some(tx) = ready.take() {
        let _ = tx.send(Err(io::Error::new(
            io::ErrorKind::ConnectionAborted,
            message,
        )));
    }
}

fn sleep_until_or_stopped(stop: &AtomicBool, duration: Duration) {
    let deadline = Instant::now() + duration;
    while !stop.load(Ordering::Acquire) {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        thread::sleep(remaining.min(POLL_INTERVAL));
    }
}

fn supervise(
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<String>>,
    ready_tx: mpsc::SyncSender<io::Result<()>>,
    mut on_frame: FrameFn,
    mut on_gaze: GazeFn,
) {
    let mut ready = Some(ready_tx);
    let mut attempts = 0u32;
    while !stop.load(Ordering::Acquire) {
        attempts += 1;
        set_status(
            &status,
            format!("connecting to PSVR2Toolkit (attempt {attempts}/{MAX_SESSIONS})"),
        );
        let result = WorkerClient::spawn();
        let reason = match result {
            Err(error) => format!("could not start PSVR2 acquisition: {error}"),
            Ok(mut worker) => {
                let mut health = SessionHealth::new(Instant::now());
                let failure = loop {
                    if stop.load(Ordering::Acquire) {
                        break None;
                    }
                    let event = match worker.recv_timeout(POLL_INTERVAL) {
                        Ok(event) => event,
                        Err(error) => {
                            break Some(format!("PSVR2 acquisition process ended: {error}"))
                        }
                    };
                    if stop.load(Ordering::Acquire) {
                        break None;
                    }
                    let now = Instant::now();
                    if let Some(event) = event {
                        match event {
                            WorkerEvent::Loading { path } => {
                                eprintln!("[psvr2] acquisition runtime: {path}");
                                set_status(&status, "loading PSVR2Toolkit");
                            }
                            WorkerEvent::Initialized { path } => {
                                health.initialized.get_or_insert(now);
                                eprintln!("[psvr2] isolated CAPI initialized: {path}");
                            }
                            WorkerEvent::Driver(active) => health.driver(active, now),
                            WorkerEvent::Gaze(native) if health.active => {
                                if health.gaze.observe(
                                    (native.wearable.timestamp, native.wearable.frame_counter),
                                    now,
                                ) {
                                    if !publish_if_running(&stop, || on_gaze(gaze_sample(native))) {
                                        break None;
                                    }
                                }
                            }
                            WorkerEvent::Image(frame) if health.active => {
                                if health.image.observe(frame.timestamp, now) {
                                    let [left, right] = frame.split_camera_planes();
                                    if !publish_if_running(&stop, || {
                                        on_frame(
                                            Eye::Left,
                                            CAMERA_WIDTH as u32,
                                            CAMERA_HEIGHT as u32,
                                            &left,
                                        )
                                    }) {
                                        break None;
                                    }
                                    if !publish_if_running(&stop, || {
                                        on_frame(
                                            Eye::Right,
                                            CAMERA_WIDTH as u32,
                                            CAMERA_HEIGHT as u32,
                                            &right,
                                        )
                                    }) {
                                        break None;
                                    }
                                }
                            }
                            WorkerEvent::StreamError { stream, message } => {
                                let evidence_errors = match stream {
                                    StreamKind::Gaze => {
                                        health.gaze.failed();
                                        health.gaze.errors
                                    }
                                    StreamKind::Image => {
                                        health.image.failed();
                                        health.image.errors
                                    }
                                };
                                if evidence_errors == 1 || evidence_errors == MAX_STREAM_ERRORS {
                                    eprintln!(
                                        "[psvr2] stream validation ({evidence_errors}): {message}"
                                    );
                                }
                            }
                            WorkerEvent::Fatal(message) => break Some(message),
                            _ => {}
                        }
                    }
                    if health.update_stability(now) {
                        // A single successful frame never resets a crash-loop budget.
                        attempts = 0;
                    }
                    if let Some(reason) = health.failure(now) {
                        break Some(reason.to_owned());
                    }
                    if health.initialized.is_some() {
                        set_status(&status, health.description(now));
                    }
                    if health.streams(now) == (true, true) {
                        if let Some(tx) = ready.take() {
                            if tx.send(Ok(())).is_err() {
                                break None;
                            }
                        }
                    }
                };
                // Native deinit can hang too. The child owns it; bounded stop
                // kills/reaps that child before any replacement is considered.
                if let Err(error) = worker.stop() {
                    report_failure(
                        &mut ready,
                        &status,
                        &format!("PSVR2 acquisition could not be stopped safely: {error}"),
                        false,
                    );
                    return;
                }
                let Some(failure) = failure else {
                    set_status(&status, "stopped");
                    return;
                };
                failure
            }
        };
        if stop.load(Ordering::Acquire) {
            break;
        }
        if attempts >= MAX_SESSIONS {
            report_failure(&mut ready, &status, &reason, true);
            return;
        }
        let delay = Duration::from_secs(1u64 << attempts.saturating_sub(1).min(2));
        eprintln!("[psvr2] {reason}; retrying in {}s", delay.as_secs());
        set_status(
            &status,
            format!("{reason} · reconnecting in {}s", delay.as_secs()),
        );
        sleep_until_or_stopped(&stop, delay);
    }
    set_status(&status, "stopped");
}

impl HmdAdapter for Psvr2Adapter {
    fn name(&self) -> &'static str {
        "psvr2"
    }
    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }
    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> io::Result<()> {
        if self.thread.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "PSVR2 adapter is already running",
            ));
        }
        // Do not revive a detached thread by clearing its stop flag.
        let lease = SupervisorLease::acquire(supervisor_slot())?;
        self.stop.store(false, Ordering::Release);
        let stop = self.stop.clone();
        let status = self.status.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        self.thread = Some(
            thread::Builder::new()
                .name("psvr2-supervisor".into())
                .spawn(move || {
                    let _lease = lease;
                    supervise(stop, status, ready_tx, on_frame, on_gaze);
                })?,
        );
        match ready_rx.recv_timeout(START_TIMEOUT) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.stop();
                Err(error)
            }
            Err(error) => {
                let detail = self.status.lock().map(|s| s.clone()).unwrap_or_default();
                self.stop();
                let message = format!("PSVR2 startup did not provide fresh gaze and both eye images: {detail} ({error}). Check SteamVR, Toolkit and camera access, then press Reload.");
                set_status(&self.status, &message);
                Err(io::Error::new(io::ErrorKind::TimedOut, message))
            }
        }
    }
    fn stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.thread.take() {
            if !join_with_budget(handle, STOP_TIMEOUT) {
                set_status(
                    &self.status,
                    "PSVR2 is still stopping; reconnect is blocked until cleanup finishes",
                );
                eprintln!(
                    "[psvr2] supervisor exceeded shutdown budget; replacement remains blocked"
                );
            }
        }
    }
    fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }
}
impl Drop for Psvr2Adapter {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::super::psvr2_capi::GazeVec2;
    use super::*;

    #[test]
    fn cached_sample_and_duplicates_cannot_prove_or_extend_liveness() {
        let now = Instant::now();
        let mut stream = StreamEvidence::new();
        assert!(!stream.observe(7, now));
        assert!(!stream.observe(7, now + Duration::from_secs(1)));
        assert!(!stream.fresh(now));
        assert!(stream.observe(8, now + Duration::from_secs(1)));
        assert!(stream.fresh(now + Duration::from_secs(2)));
        assert!(!stream.observe(8, now + Duration::from_secs(3)));
        assert!(!stream.fresh(now + Duration::from_secs(3)));
    }

    #[test]
    fn image_and_gaze_errors_and_freshness_are_independent() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        health.initialized = Some(now);
        health.driver(true, now);
        health.gaze.observe((1, 1), now);
        health.image.observe(1, now);
        health.gaze.observe((2, 2), now);
        health.image.observe(2, now);
        health.update_stability(now);
        for i in 0..MAX_STREAM_ERRORS {
            health.image.failed();
            health.gaze.observe((i as i64 + 3, i + 3), now);
        }
        assert_eq!(health.image.errors, MAX_STREAM_ERRORS);
        assert_eq!(health.gaze.errors, 0);
        assert!(health.failure(now).unwrap().contains("eye-camera"));
        health.image.errors = 0;
        let later = now + STALE_RESTART;
        health.last_driver_heartbeat = Some(later);
        health.gaze.observe((100, 100), later);
        assert!(health
            .failure(later)
            .unwrap()
            .contains("eye-camera updates stopped"));
    }

    #[test]
    fn transient_driver_loss_requires_new_samples_and_has_a_grace_period() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        health.initialized = Some(now);
        health.driver(true, now);
        health.gaze.observe((1, 1), now);
        health.gaze.observe((2, 2), now);
        health.image.observe(1, now);
        health.image.observe(2, now);
        assert_eq!(health.streams(now), (true, true));
        health.driver(false, now);
        assert_eq!(health.streams(now), (false, false));
        assert!(health.failure(now + Duration::from_millis(500)).is_none());
        assert!(health
            .failure(now + INACTIVE_GRACE)
            .unwrap()
            .contains("inactive"));
        health.driver(true, now + Duration::from_millis(500));
        assert!(!health.gaze.observe((2, 2), now));
        assert!(!health.gaze.observe((3, 3), now));
        assert!(health.gaze.observe((4, 4), now));
        assert_eq!(health.streams(now), (true, false));
    }

    #[test]
    fn native_init_and_read_stalls_have_independent_deadlines() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        assert!(health
            .failure(now + SESSION_TIMEOUT)
            .unwrap()
            .contains("initialization"));
        health.initialized = Some(now);
        health.driver(true, now);
        assert!(health
            .failure(now + HEARTBEAT_TIMEOUT)
            .unwrap()
            .contains("responding"));
        health.last_driver_heartbeat = Some(now + SESSION_TIMEOUT);
        assert!(health
            .failure(now + SESSION_TIMEOUT)
            .unwrap()
            .contains("No fresh"));
    }

    #[test]
    fn retry_budget_reset_requires_continuously_fresh_both_streams() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        health.driver(true, now);
        health.gaze.observe((0, 0), now);
        health.image.observe(0, now);
        for n in 1..=32 {
            let at = now + Duration::from_secs(n);
            health.gaze.observe((n as i64, n as u32), at);
            health.image.observe(n as u32, at);
            assert_eq!(health.update_stability(at), n >= 31);
        }
        assert!(!health.update_stability(now + Duration::from_secs(34)));
        assert!(health.stable_since.is_none());
    }

    #[test]
    fn image_timestamp_wrap_is_new_data() {
        let now = Instant::now();
        let mut stream = StreamEvidence::new();
        assert!(!stream.observe(u32::MAX, now));
        assert!(stream.observe(0, now));
    }

    #[test]
    fn fresh_data_cannot_mask_a_missing_driver_heartbeat() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        health.initialized = Some(now);
        health.driver(true, now);
        health.gaze.observe((1, 1), now);
        health.image.observe(1, now);
        let later = now + HEARTBEAT_TIMEOUT;
        health.gaze.observe((2, 2), later);
        health.image.observe(2, later);
        assert_eq!(health.streams(later), (true, true));
        assert!(health.failure(later).unwrap().contains("heartbeat"));
    }

    #[test]
    fn late_driver_interruption_gets_a_new_bounded_data_grace() {
        let now = Instant::now();
        let mut health = SessionHealth::new(now);
        health.initialized = Some(now);
        health.driver(true, now);
        health.gaze.observe((1, 1), now);
        health.image.observe(1, now);
        health.gaze.observe((2, 2), now);
        health.image.observe(2, now);
        health.update_stability(now);
        let interruption = now + Duration::from_secs(60);
        health.driver(false, interruption);
        let resume = interruption + Duration::from_millis(100);
        health.driver(true, resume);
        assert!(health.failure(resume).is_none());
        let later = resume + STALE_RESTART - Duration::from_millis(1);
        health.driver(true, later);
        assert!(health.failure(later).is_none());
        assert!(health
            .failure(later + Duration::from_millis(1))
            .unwrap()
            .contains("updates stopped"));
    }

    #[test]
    fn stalled_callback_has_bounded_stop_and_retains_exclusive_lease() {
        let slot = Arc::new(AtomicBool::new(false));
        let lease = SupervisorLease::acquire(&slot).unwrap();
        let (release_tx, release_rx) = mpsc::channel();
        let worker = thread::spawn(move || {
            let _lease = lease;
            let _ = release_rx.recv_timeout(Duration::from_secs(2));
        });
        assert!(!join_with_budget(worker, Duration::from_millis(20)));
        assert!(SupervisorLease::acquire(&slot).is_err());
        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        while slot.load(Ordering::Acquire) {
            assert!(Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        }
        assert!(SupervisorLease::acquire(&slot).is_ok());
    }

    #[test]
    fn stop_between_stereo_callbacks_suppresses_the_second_callback() {
        let stop = AtomicBool::new(false);
        assert!(publish_if_running(&stop, || stop.store(true, Ordering::Release)));
        let mut called = false;
        assert!(!publish_if_running(&stop, || called = true));
        assert!(!called);
    }

    fn tracked_eye(gaze_x: f32, blink: bool) -> WearableEye {
        WearableEye {
            is_gaze_origin_valid: 1,
            gaze_origin_mm: GazeVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            is_gaze_dir_valid: 1,
            gaze_dir_norm: GazeVec3 {
                x: gaze_x,
                y: -0.2,
                z: 0.97,
            },
            is_pupil_dia_valid: 1,
            pupil_dia_mm: 4.2,
            is_pupil_pos_in_sensor_area_valid: 1,
            pupil_pos_in_sensor_area: GazeVec2 { x: 0.4, y: 0.6 },
            is_blink_valid: 1,
            blink: blink as u32,
            ..WearableEye::default()
        }
    }

    #[test]
    fn hardware_mapping_keeps_raw_gaze_and_routes_a_b_as_left_right() {
        let mut status = GazeStatus::default();
        status.wearable.timestamp = 123_456;
        status.wearable.left = tracked_eye(0.5, false);
        status.wearable.right = tracked_eye(-0.4, false);
        let sample = gaze_sample(status);

        assert_eq!(sample.timestamp_us, 123_456);
        assert_eq!(sample.left.gaze[0], 0.5);
        assert_eq!(sample.right.gaze[0], -0.4);
        assert!(sample.left.gaze_valid && sample.right.gaze_valid);
        assert_eq!(Psvr2Adapter::new().profile().slot_a_eye, Eye::Left);
    }

    #[test]
    fn native_blink_is_absolute_closed_evidence_not_analog_openness() {
        let open = eye_sample(tracked_eye(0.0, false));
        assert!(open.openness_reported);
        assert!(open.openness_valid);
        assert_eq!(open.openness, 1.0);

        let closed = eye_sample(tracked_eye(0.0, true));
        assert!(closed.openness_reported);
        assert!(!closed.openness_valid);
        assert_eq!(closed.openness, 0.0);
    }

    #[test]
    fn unreported_blink_does_not_fabricate_native_openness() {
        let mut eye = tracked_eye(0.0, false);
        eye.is_blink_valid = 0;
        let sample = eye_sample(eye);
        assert!(!sample.openness_reported);
        assert!(!sample.openness_valid);
    }
}
