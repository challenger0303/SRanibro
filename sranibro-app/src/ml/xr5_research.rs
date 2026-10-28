//! Optional raw-200x200 research model. Bounded child IPC, no UI-thread inference.
use crate::core::types::EyeResult;
use serde::{Deserialize, Serialize};
mod fallback;
pub(crate) use fallback::AutoFallbackBridge;
#[cfg(feature = "xr5-research-model")]
use std::path::PathBuf;
use std::{
    io::{self, Read, Write},
    process::{Child, Command, Stdio},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Mutex,
    },
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub response: ResponseSettings,
    /// Data-only model file for normal builds; legacy research folders stay separate.
    pub model_file: String,
    pub enabled: bool,
    pub avatar: bool,
    pub bundle: String,
    pub python: String,
    pub python_path: String,
    /// Worker/diagnostic preference; the live app overrides this legacy field
    /// with the shared ui.eyelid_inference_backend setting.
    pub gpu: bool,
}
impl Default for Settings {
    fn default() -> Self {
        Self {
            response: ResponseSettings::default(),
            model_file: String::new(),
            enabled: cfg!(all(windows, feature = "external-eyelid-model", not(feature = "xr5-only"))),
            avatar: cfg!(all(windows, feature = "external-eyelid-model", not(feature = "xr5-only"))),
            bundle: "model".into(),
            python: "python.exe".into(),
            python_path: String::new(),
            gpu: false,
        }
    }
}

impl Settings {
    /// An unset selection uses the distributed model, but never replaces an
    /// explicit model selection. Explicit backend selections survive deserialization.
    pub fn model_file(&self) -> &str {
        #[cfg(all(feature = "external-eyelid-model", not(feature = "xr5-only")))]
        if self.model_file.is_empty() {
            return "models/XR5-Eyelid-Balanced-20260928.sranimodel";
        }
        &self.model_file
    }
}

/// Coordinates are measured aperture / relaxed-open reference, not EyeNet units.
/// Keep them separate from SRanipal endpoints and adaptive Wearing Memory.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResponseSettings {
    /// Opt-in research readout; legacy aperture remains the default.
    pub independent_wide: bool,
    pub wide_reference: [f32; 2],
    pub independent_start: [f32; 2],
    pub independent_full: [f32; 2],
    pub closed: [f32; 2],
    pub open: [f32; 2],
    pub wide_start: [f32; 2],
    pub wide_full: [f32; 2],
}
impl Default for ResponseSettings {
    fn default() -> Self {
        Self { independent_wide: false, wide_reference: [0.0; 2], independent_start: [1.1; 2], independent_full: [1.5; 2], closed: [0.0; 2], open: [1.0; 2], wide_start: [1.10; 2], wide_full: [1.50; 2] }
    }
}
impl ResponseSettings {
    pub fn sanitized(mut self) -> Self {
        let defaults = Self::default();
        let safe = |x: f32, fallback: f32| if x.is_finite() { x } else { fallback };
        for e in 0..2 {
            self.wide_reference[e] = safe(self.wide_reference[e], 0.0).clamp(0.0, 300.0);
            self.independent_start[e] = safe(self.independent_start[e], 1.1).clamp(1.0, 7.95);
            self.independent_full[e] = safe(self.independent_full[e], 1.5).clamp(self.independent_start[e] + 0.05, 8.0);
            self.closed[e] = safe(self.closed[e], defaults.closed[e]).clamp(0.0, 7.85);
            self.open[e] = safe(self.open[e], defaults.open[e]).clamp(self.closed[e] + 0.05, 7.90);
            self.wide_start[e] = safe(self.wide_start[e], defaults.wide_start[e]).clamp(self.open[e], 7.95);
            self.wide_full[e] = safe(self.wide_full[e], defaults.wide_full[e]).clamp(self.wide_start[e] + 0.05, 8.0);
        }
        self
    }
    pub fn map(self, eye: usize, opening: f32) -> (f32, f32) {
        if !opening.is_finite() { return (1.0, 0.0); }
        let p = self.sanitized();
        let openness = ((opening - p.closed[eye]) / (p.open[eye] - p.closed[eye])).clamp(0.0, 1.0);
        let wide = ((opening - p.wide_start[eye]) / (p.wide_full[eye] - p.wide_start[eye])).clamp(0.0, 1.0);
        (openness, wide)
    }
    pub fn wide_ratio(self, eye: usize, height: Option<f32>) -> Option<f32> {
        let reference = self.wide_reference[eye];
        if !reference.is_finite() || !(3.0..=300.0).contains(&reference) { return None; }
        height.filter(|h| h.is_finite() && (0.0..=300.0).contains(h))
            .map(|h| h / reference).filter(|r| r.is_finite() && (0.0..=8.0).contains(r))
    }
    pub fn map_independent_wide(self, eye: usize, height: Option<f32>) -> f32 {
        let p = self.sanitized();
        p.wide_ratio(eye, height).map(|v|
            ((v - p.independent_start[eye]) / (p.independent_full[eye] - p.independent_start[eye])).clamp(0.0, 1.0)
        ).unwrap_or(0.0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EyePrediction {
    pub wide_height: Option<f32>,
    pub status: u8,
    pub opening: Option<f32>,
    pub closed_score: f32,
    pub height: f32,
}
impl EyePrediction {
    pub fn label(&self) -> &'static str {
        match self.status {
            0 => "Tracking",
            1 => "Closed",
            2 => "No region",
            3 => "Uncertain",
            4 => "Narrow / uncertain",
            _ => "Waiting",
        }
    }
}
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub enabled: bool,
    pub avatar: bool,
    pub message: String,
    pub eyes: [EyePrediction; 2],
    pub captured_at: Option<Instant>,
    pub count: u64,
    pub backend: &'static str,
}
#[derive(Clone)]
pub struct RawFrame {
    pub generation: u64,
    pub at: Instant,
    pub pixels: Arc<[u8]>,
}
pub fn coherent_pair(left: &RawFrame, right: &RawFrame, now: Instant) -> bool {
    now.saturating_duration_since(left.at) <= Duration::from_millis(100)
        && now.saturating_duration_since(right.at) <= Duration::from_millis(100)
        && left.at.max(right.at).duration_since(left.at.min(right.at)) <= Duration::from_millis(40)
}
impl Default for Snapshot {
    fn default() -> Self {
        Self {
            enabled: false,
            avatar: false,
            message: "Not selected".into(),
            eyes: [EyePrediction::default(); 2],
            captured_at: None,
            count: 0,
            backend: "CPU",
        }
    }
}
impl Snapshot {
    /// Adapter into the app's existing scalar-model domain. This affine change
    /// of units is not a new classifier: neutral1 ->0.5; closed0 ->0.1.
    pub fn common_frame(&self, now: Instant) -> [[f32; 5]; 2] {
        std::array::from_fn(|i| {
            let p = self.eyes[i];
            let value = self.fresh(now).then_some(p.opening).flatten()
                .filter(|v| p.status < 2 && v.is_finite() && (0.0..=8.0).contains(v));
            match value {
                Some(v) => [1.0, 0.1 + 0.4 * if p.status == 1 { 0.0 } else { v }, 0.0, 0.0, 0.0],
                None => [0.0, 0.5, 0.0, 0.0, 0.0],
            }
        })
    }
    /// Only the processing copy gets model closure evidence. Native gaze/pupil
    /// values and their validity remain untouched, including in dedicated OSC.
    pub fn common_lid_sample(&self, mut gaze: crate::core::types::GazeSample, now: Instant) -> crate::core::types::GazeSample {
        let frame = self.common_frame(now);
        for (i, eye) in [&mut gaze.left, &mut gaze.right].into_iter().enumerate() {
            eye.openness_reported = frame[i][0] > 0.0;
            eye.openness_valid = eye.openness_reported && self.eyes[i].status == 0;
        }
        gaze
    }
    pub fn guard_common_output(&self, results: &mut [EyeResult; 2], now: Instant) {
        let frame = self.common_frame(now);
        for (i, result) in results.iter_mut().enumerate() {
            result.squeeze = 0.0; result.frown = 0.0;
            if frame[i][0] == 0.0 {
                result.openness = 1.0; result.openness_valid = false; result.blink = false; result.wide = 0.0;
            }
        }
    }
    pub fn fresh(&self, now: Instant) -> bool {
        self.captured_at
            .is_some_and(|t| now.saturating_duration_since(t) <= Duration::from_millis(200))
    }
    pub fn apply(&self, results: &mut [EyeResult; 2], now: Instant) {
        self.apply_response(results, now, ResponseSettings::default());
    }
    pub fn apply_response(&self, results: &mut [EyeResult; 2], now: Instant, response: ResponseSettings) -> bool {
        if !self.enabled || !self.avatar {
            return false;
        }
        for (i, result) in results.iter_mut().enumerate() {
            let prediction = self.eyes[i];
            let value = self.fresh(now).then_some(prediction.opening).flatten()
                .filter(|v| prediction.status < 2 && v.is_finite() && (0.0..=8.0).contains(v));
            // Unknown is neutral-open + INVALID, never synthetic closure or stale hold.
            let (openness, wide) = value.map(|v| {
                if prediction.status == 1 { (0.0, 0.0) } else { response.map(i, v) }
            }).unwrap_or((1.0, 0.0));
            result.openness = openness;
            result.openness_valid = value.is_some();
            result.blink = value.is_some() && openness == 0.0;
            // Wide is aperture-derived, not a separate expression classifier.
            result.wide = if response.independent_wide {
                if value.is_some() && prediction.status == 0 && openness > 0.0 {
                    response.map_independent_wide(i, prediction.wide_height)
                } else { 0.0 }
            } else { wide };
            result.squeeze = 0.0;
            result.frown = 0.0;
            // Independent eyebrow output is owned by the eyebrow worker.
        }
        true
    }
}

/// Presentation continuity only. This never manufactures model evidence or feeds
/// a held/fallback value into calibration. Transport validity must remain true:
/// VRCFT otherwise ignores the fallback and leaves the avatar frozen indefinitely.
#[derive(Default)]
pub struct OutputRecovery {
    eyes: [RecoveryEye; 2],
}
#[derive(Default)]
struct RecoveryEye {
    output: Option<[f32; 2]>,
    uncertain_hold: bool,
    mode: u8,
    last_tick: Option<Instant>,
    lost_at: Option<Instant>,
    good_since: Option<Instant>,
    last_good_count: Option<u64>,
    good_samples: u8,
    recovering_until: Option<Instant>,
    loss_origin: [f32; 2],
}
impl OutputRecovery {
    /// Diagnostic codes: 0 tracking, 1 fresh uncertain hold, 2 transport/sensor
    /// fallback, 3 reentry. Presentation continuity is not measured evidence.
    pub fn modes(&self) -> [u8; 2] { self.eyes.each_ref().map(|eye| eye.mode) }

    pub fn apply(&mut self, snapshot: &Snapshot, results: &mut [EyeResult; 2], now: Instant) {
        let frame = snapshot.common_frame(now);
        for i in 0..2 {
            let s = &mut self.eyes[i];
            let r = &mut results[i];
            let dt = s.last_tick.map(|t| now.saturating_duration_since(t).as_secs_f32()).unwrap_or(0.0);
            s.last_tick = Some(now);
            let measured = frame[i][0] > 0.0 && r.openness.is_finite() && r.wide.is_finite();
            let p = snapshot.eyes[i];
            // Fresh segmentation uncertainty is NOT evidence of opening and is
            // NOT a transport outage. Preserve the last delivered pose, including
            // a partly completed wink. Never synthesize full closure from Unknown.
            // An actual outage must first recover with valid measurements; fresh
            // but unclassifiable replies alone cannot re-authorize a stale pose.
            let hold_uncertain = !measured && snapshot.fresh(now)
                && (2..=4).contains(&p.status)
                && p.closed_score.is_finite() && (0.0..=1.0).contains(&p.closed_score)
                && p.height.is_finite() && (0.0..=200.0).contains(&p.height)
                && s.output.is_some() && s.lost_at.is_none();
            if hold_uncertain {
                let previous = s.output.unwrap();
                r.openness = previous[0];
                r.wide = 0.0;
                r.squeeze = 0.0;
                r.frown = 0.0;
                r.openness_valid = true; // Presentation only; common_frame stays invalid.
                r.blink = previous[0] < 0.08;
                s.output = Some([previous[0], 0.0]);
                s.uncertain_hold = true;
                s.mode = 1;
                s.lost_at = None;
                s.good_since = None;
                s.last_good_count = None;
                s.good_samples = 0;
                s.recovering_until = None;
                continue;
            }
            if s.uncertain_hold {
                s.uncertain_hold = false;
                if measured { s.recovering_until = Some(now + Duration::from_millis(60)); }
            }
            if !measured {
                if s.lost_at.is_none() {
                    s.lost_at = Some(now);
                    s.loss_origin = s.output.unwrap_or([1.0, 0.0]);
                }
                s.good_since = None;
                s.last_good_count = None;
                s.good_samples = 0;
                s.recovering_until = None;
            } else if s.lost_at.is_some() {
                s.good_since.get_or_insert(now);
                if s.last_good_count != Some(snapshot.count) {
                    s.last_good_count = Some(snapshot.count);
                    s.good_samples = s.good_samples.saturating_add(1);
                }
                if s.good_samples >= 2 && now.duration_since(s.good_since.unwrap()) >= Duration::from_millis(25) {
                    s.lost_at = None;
                    s.recovering_until = Some(now + Duration::from_millis(60));
                }
            }
            let continuity = s.lost_at.is_some() || s.recovering_until.is_some();
            s.mode = if s.lost_at.is_some() { 2 } else if s.recovering_until.is_some() { 3 } else { 0 };
            let output = if let Some(t) = s.lost_at {
                let a = ((now.saturating_duration_since(t).as_secs_f32() - 0.060) / 0.180).clamp(0.0, 1.0);
                [s.loss_origin[0] + a * (1.0 - s.loss_origin[0]), s.loss_origin[1] * (1.0 - a)]
            } else if let Some(until) = s.recovering_until {
                let previous = s.output.unwrap_or([1.0, 0.0]);
                let target = [r.openness.clamp(0.0, 1.0), r.wide.clamp(0.0, 1.0)];
                let step = (dt / 0.060).clamp(0.0, 1.0);
                let next = std::array::from_fn(|e| previous[e] + (target[e] - previous[e]).clamp(-step, step));
                if now >= until && next == target { s.recovering_until = None; }
                next
            } else { [r.openness.clamp(0.0, 1.0), r.wide.clamp(0.0, 1.0)] };
            s.output = Some(output);
            r.openness = output[0]; r.wide = output[1]; r.squeeze = 0.0; r.frown = 0.0;
            r.openness_valid = true;
            if continuity { r.blink = output[0] < 0.08; }
        }
    }
}

/// UI-owned, bounded capture. Never changes the eyelid worker's reference.
pub struct WideReferenceCapture {
    start: Instant,
    last_count: Option<u64>,
    samples: [Vec<f32>; 2],
}
impl WideReferenceCapture {
    pub fn new(now: Instant) -> Self {
        Self { start: now, last_count: None, samples: Default::default() }
    }
    pub fn tick(&mut self, snapshot: &Snapshot, now: Instant) -> Option<Result<[f32; 2], &'static str>> {
        let elapsed = now.saturating_duration_since(self.start);
        if elapsed < Duration::from_secs(2) { return None; }
        if elapsed >= Duration::from_secs(6) { return Some(Err("Not enough steady frames. Try again.")); }
        if snapshot.enabled && snapshot.fresh(now) && self.last_count != Some(snapshot.count) {
            self.last_count = Some(snapshot.count);
            for e in 0..2 {
                let p = snapshot.eyes[e];
                if p.status == 0 && p.closed_score < 0.5 && self.samples[e].len() < 240 {
                    if let Some(h) = p.wide_height.filter(|h| h.is_finite() && (3.0..=300.0).contains(h)) {
                        self.samples[e].push(h);
                    }
                }
            }
        }
        if elapsed < Duration::from_millis(3500) || self.samples.iter().any(|s| s.len() < 30) { return None; }
        let mut result = [0.0; 2];
        for (e, s) in self.samples.iter_mut().enumerate() {
            s.sort_by(f32::total_cmp);
            result[e] = (s[(s.len()-1)/2] + s[s.len()/2]) * 0.5;
            if (s[(s.len()-1)*9/10] - s[(s.len()-1)/10]) / result[e] > 0.15 {
                return Some(Err("Eyes moved too much. Keep them relaxed and try again."));
            }
        }
        Some(Ok(result))
    }
    pub fn preparing(&self, now: Instant) -> bool { now.saturating_duration_since(self.start) < Duration::from_secs(2) }
}

static OCCUPIED: AtomicBool = AtomicBool::new(false);
struct Lease;
impl Drop for Lease {
    fn drop(&mut self) {
        OCCUPIED.store(false, Ordering::Release);
    }
}
pub(crate) struct Process {
    child: Mutex<Option<Child>>,
    #[cfg(windows)]
    _job: super::gpu_worker::WorkerJob,
    _lease: Lease,
}
impl Process {
    fn kill(&self) {
        #[cfg(windows)]
        self._job.terminate_tree();
        if let Ok(mut c) = self.child.lock() {
            if let Some(c) = c.as_mut() {
                let _ = c.kill();
            }
        }
    }
    fn reap(&self) {
        self.kill();
        let child = self.child.lock().ok().and_then(|mut c| c.take());
        if let Some(mut c) = child {
            let _ = c.wait();
        }
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        self.kill();
    }
}
struct Job {
    seq: u64,
    pixels: Vec<u8>,
    set_open: bool,
}
enum Reply {
    Ready,
    Value(u64, [EyePrediction; 2], &'static str, u8),
    Error(String),
}
pub struct Bridge {
    process: Arc<Process>,
    sender: mpsc::SyncSender<Job>,
    receiver: mpsc::Receiver<Reply>,
    started: Instant,
    pending: Option<(u64, Instant, Instant)>,
    ready: bool,
    failed: bool,
    next: u64,
}
fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
#[cfg(feature = "xr5-research-model")]
fn resolve(path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|parent| parent.join(path)))
            .unwrap_or(p)
    }
}

fn read_reply(r: &mut impl Read) -> io::Result<(u64, [EyePrediction; 2], &'static str, u8)> {
    let mut bytes = [0u8; 48];
    r.read_exact(&mut bytes[..4])?;
    let stride = match &bytes[..4] { b"X5R1" => 13, b"X5R2" => 17, _ => return Err(invalid("wrong model reply")) };
    r.read_exact(&mut bytes[4..14 + 2 * stride])?;
    let seq = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    let backend = match bytes[12] {
        0 => "CPU",
        1 => "GPU",
        _ => return Err(invalid("invalid backend")),
    };
    let reference_status = bytes[13];
    if reference_status > 3 {
        return Err(invalid("invalid reference status"));
    }
    let mut eyes = [EyePrediction::default(); 2];
    for (i, e) in eyes.iter_mut().enumerate() {
        let offset = 14 + i * stride;
        let s = bytes[offset];
        let values: [f32; 3] = std::array::from_fn(|n| {
            f32::from_le_bytes(
                bytes[offset + 1 + n * 4..offset + 5 + n * 4]
                    .try_into()
                    .unwrap(),
            )
        });
        if s > 5
            || !values.iter().all(|v| v.is_finite())
            || !(0.0..=8.0).contains(&values[0])
            || !(0.0..=1.0).contains(&values[1])
            || !(0.0..=200.0).contains(&values[2])
            || (s == 1 && values[0] != 0.0)
            || (s > 1 && values[0] != 0.0)
        {
            return Err(invalid("invalid model values"));
        }
        *e = EyePrediction {
            wide_height: if stride == 17 {
                let h = f32::from_le_bytes(bytes[offset + 13..offset + 17].try_into().unwrap());
                if !h.is_finite() || !(0.0..=300.0).contains(&h) { return Err(invalid("invalid Wide measurement")); }
                (s == 0).then_some(h)
            } else { None },
            status: s,
            opening: (s < 2).then_some(values[0]),
            closed_score: values[1],
            height: values[2],
        };
    }
    Ok((seq, eyes, backend, reference_status))
}
impl Bridge {
    pub fn start(settings: &Settings) -> io::Result<Self> {
        if OCCUPIED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(invalid("Previous XR5 worker is still stopping"));
        }
        let lease = Lease;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(crate::config::base_dir().join("xr5-model-worker.log"))?;
        let mut command = model_command(settings)?;
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(log);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000);
        }
        let mut child = command.spawn()?;
        #[cfg(windows)]
        let job = match super::gpu_worker::WorkerJob::attach(&child) {
            Ok(j) => j,
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(e);
            }
        };
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| invalid("no model input pipe"))?;
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| invalid("no model output pipe"))?;
        let process = Arc::new(Process {
            child: Mutex::new(Some(child)),
            #[cfg(windows)]
            _job: job,
            _lease: lease,
        });
        let (sender, jobs) = mpsc::sync_channel::<Job>(1);
        let (tx, receiver) = mpsc::sync_channel(1);
        let owner = process.clone();
        std::thread::Builder::new()
            .name("xr5-model-pipe".into())
            .spawn(move || {
                let mut run = || -> io::Result<()> {
                    let mut magic = [0; 8];
                    stdout.read_exact(&mut magic)?;
                    if &magic != b"XR5LIV17" {
                        return Err(invalid("model startup failed"));
                    }
                    tx.send(Reply::Ready)
                        .map_err(|_| invalid("model stopped"))?;
                    while let Ok(j) = jobs.recv() {
                        stdin.write_all(&j.seq.to_le_bytes())?;
                        stdin.write_all(&[u8::from(j.set_open)])?;
                        stdin.write_all(&j.pixels)?;
                        stdin.flush()?;
                        let (seq, eyes, backend, reference) = read_reply(&mut stdout)?;
                        tx.send(Reply::Value(seq, eyes, backend, reference))
                            .map_err(|_| invalid("model stopped"))?;
                    }
                    Ok(())
                };
                if let Err(e) = run() {
                    let _ = tx.send(Reply::Error(e.to_string()));
                }
                owner.reap();
            })?;
        Ok(Self {
            process,
            sender,
            receiver,
            started: Instant::now(),
            pending: None,
            ready: false,
            failed: false,
            next: 0,
        })
    }
}

fn model_command(settings: &Settings) -> io::Result<Command> {
    #[cfg(all(windows, feature = "external-eyelid-model"))]
    if !settings.model_file().is_empty() {
        let path = super::external_eyelid::resolve(settings.model_file());
        super::external_eyelid::Package::read(&path)?;
        let executable = std::env::current_exe()?;
        // The ignored no-camera integration test runs the built app as its child,
        // rather than treating the Rust test harness as a model worker.
        #[cfg(test)]
        let executable = std::env::var_os("XR5_NATIVE_TEST_EXE").map(std::path::PathBuf::from).unwrap_or(executable);
        let mut command = Command::new(executable);
        command
            .arg("--xr5-model-worker")
            .arg(path)
            .arg(if settings.gpu { "gpu" } else { "cpu" });
        return Ok(command);
    }
    #[cfg(not(feature = "xr5-research-model"))]
    {
        Err(invalid(
            "Choose an XR5 eyelid model file in Tracking settings",
        ))
    }
    #[cfg(feature = "xr5-research-model")]
    {
        let bundle = resolve(&settings.bundle);
        for file in [
            "live_worker.py",
            "manifest.json",
            "xr5-joint-stereo.onnx",
            "recording-profile.json",
        ] {
            if !bundle.join(file).is_file() {
                return Err(invalid(&format!("XR5 model bundle missing {file}")));
            }
        }
        let mut command = Command::new(&settings.python);
        command
            .args(["-s", "-u"])
            .arg(bundle.join("live_worker.py"))
            .arg("--bundle")
            .arg(&bundle)
            .arg("--backend")
            .arg(if settings.gpu { "directml" } else { "cpu" })
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONNOUSERSITE", "1");
        if !settings.python_path.is_empty() {
            command.env("PYTHONPATH", &settings.python_path);
        }
        Ok(command)
    }
}
impl Bridge {
    pub fn tick(&mut self, state: &mut Snapshot) {
        if self.failed {
            return;
        }
        while let Ok(reply) = self.receiver.try_recv() {
            match reply {
                Reply::Ready => {
                    self.ready = true;
                    state.message = "Waiting for eye cameras".into();
                }
                Reply::Value(seq, eyes, backend, reference) => {
                    if let Some((expected, t, captured)) = self.pending.take() {
                        if seq != expected || t.elapsed() > Duration::from_secs(3) {
                            self.fail(state, "Late or mismatched model reply");
                            return;
                        }
                        state.eyes = eyes;
                        state.captured_at = Some(captured);
                        state.backend = backend;
                        state.count += 1;
                        state.message = match reference {
                            1 => "Keep both eyes comfortably open…",
                            2 => "Open reference updated · this session",
                            3 => "Reference unchanged — open both eyes and try again",
                            _ => "Live — raw camera input",
                        }
                        .into();
                    } else {
                        self.fail(state, "Unexpected model reply");
                        return;
                    }
                }
                Reply::Error(e) => {
                    self.fail(
                        state,
                        &format!("Model stopped: {e}. See xr5-model-worker.log"),
                    );
                    return;
                }
            }
        }
        // Output is already invalid after 200ms. The outer manager replaces a
        // failed GPU worker with CPU only after its process/pipe owner has drained.
        if (!self.ready && self.started.elapsed() > Duration::from_secs(35))
            || self
                .pending
                .is_some_and(|(_, t, _)| t.elapsed() > Duration::from_secs(3))
        {
            self.fail(state, "Model timeout — reload to retry");
        }
    }
    pub fn available(&self) -> bool {
        self.ready && !self.failed && self.pending.is_none()
    }
    pub fn submit(&mut self, left: &[u8], right: &[u8], captured: Instant, set_open: bool) -> bool {
        if !self.available() || left.len() != 40000 || right.len() != 40000 {
            return false;
        }
        let mut pixels = Vec::with_capacity(80000);
        pixels.extend_from_slice(left);
        pixels.extend_from_slice(right);
        self.next += 1;
        let at = Instant::now();
        if self
            .sender
            .try_send(Job {
                seq: self.next,
                pixels,
                set_open,
            })
            .is_ok()
        {
            self.pending = Some((self.next, at, captured));
            true
        } else {
            false
        }
    }
    fn fail(&mut self, state: &mut Snapshot, message: &str) {
        self.failed = true;
        self.process.kill();
        state.captured_at = None;
        state.message = message.into();
    }
}
impl Drop for Bridge {
    fn drop(&mut self) {
        self.process.kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Eye;
    #[cfg(all(feature = "external-eyelid-model", not(feature = "xr5-only")))]
    #[test]
    fn unset_model_uses_bundle_without_enabling_tracking_or_changing_custom_path() {
        let mut s: Settings = toml::from_str("model_file = ''\nenabled = false\navatar = false").unwrap();
        assert_eq!(s.model_file(), "models/XR5-Eyelid-Balanced-20260928.sranimodel");
        assert!(!s.enabled && !s.avatar);
        assert!(s.model_file.is_empty());
        s.model_file = "D:/custom/missing.sranimodel".into();
        assert_eq!(s.model_file(), "D:/custom/missing.sranimodel");
        let restored: Settings = toml::from_str(&toml::to_string(&s).unwrap()).unwrap();
        assert_eq!(restored.model_file(), s.model_file());
    }
    #[cfg(all(windows, feature = "external-eyelid-model", not(feature = "xr5-only")))]
    #[test]
    fn native_default_preserves_explicit_sranipal_fallback() {
        for text in ["", "model_file = ''"] {
            let settings: Settings = toml::from_str(text).unwrap();
            assert!(settings.enabled && settings.avatar);
        }
        for text in ["", "[hmd]\ndevice = 'pimax_xr5'"] {
            let config: crate::config::Config = toml::from_str(text).unwrap();
            let (_, init) = crate::engine::pipeline_start_settings(&config, "pimax_xr5");
            let selected = init.xr5_research.unwrap();
            assert!(selected.enabled && selected.avatar);
            assert_eq!(selected.model_file(), "models/XR5-Eyelid-Balanced-20260928.sranimodel");
        }
        let fallback: Settings = toml::from_str("enabled = false\navatar = false").unwrap();
        assert!(!fallback.enabled && !fallback.avatar);
        let restored: Settings = toml::from_str(&toml::to_string(&fallback).unwrap()).unwrap();
        assert!(!restored.enabled && !restored.avatar);
    }
    #[test]
    fn openness_and_wide_share_a_monotonic_nonoverlapping_axis() {
        let p = ResponseSettings::default();
        assert_eq!(p.map(0, 0.0), (0.0, 0.0));
        assert_eq!(p.map(0, 0.5), (0.5, 0.0));
        assert_eq!(p.map(0, 1.0), (1.0, 0.0));
        assert_eq!(p.map(0, 1.5), (1.0, 1.0));
        let mut previous = (0.0, 0.0);
        for n in 0..=2000 {
            let next = p.map(0, n as f32 / 1000.0);
            assert!(next.0 >= previous.0 && next.1 >= previous.1);
            assert!(next.1 == 0.0 || next.0 == 1.0);
            assert!(next.0 - previous.0 < 0.002 && next.1 - previous.1 < 0.003);
            previous = next;
        }
    }
    #[test]
    fn response_sanitizes_nan_reversed_and_zero_width_ranges() {
        let p = ResponseSettings { closed: [f32::NAN, 9.0], open: [-1.0, 0.0],
            wide_start: [f32::INFINITY, -2.0], wide_full: [0.0, f32::NAN], ..Default::default() }.sanitized();
        for e in 0..2 {
            assert!(p.closed[e] < p.open[e] && p.open[e] <= p.wide_start[e] && p.wide_start[e] < p.wide_full[e]);
            let (open, wide) = p.map(e, 2.0);
            assert!(open.is_finite() && wide.is_finite());
        }
        assert_eq!(p.map(0, f32::NAN), (1.0, 0.0));
    }
    #[test]
    fn response_is_saved_independently_and_old_config_gets_defaults() {
        let old: Settings = toml::from_str("enabled = true\navatar = true").unwrap();
        assert_eq!(old.response, ResponseSettings::default());
        let mut edited = old;
        edited.response.closed = [0.12, 0.18];
        edited.response.wide_full = [1.4, 1.7];
        let next: Settings = toml::from_str(&toml::to_string(&edited).unwrap()).unwrap();
        assert_eq!(edited.response, next.response);
    }
    #[test]
    fn native_range_changes_are_live_per_eye_and_preserve_gaze_brow() {
        let now = Instant::now();
        let mut s = Snapshot { enabled:true, avatar:true, captured_at:Some(now),
            eyes:[EyePrediction { status:0, opening:Some(0.2), ..Default::default() };2], ..Default::default() };
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        out[0].gaze=[0.2, 0.1, 1.0]; out[0].brow=0.4; out[0].brow_valid=true;
        let mut p=ResponseSettings::default(); p.closed[0]=0.2;
        assert!(s.apply_response(&mut out, now, p));
        assert_eq!(out[0].openness,0.0); assert!(out[0].blink);
        assert_eq!(out[1].openness,0.2); assert!(!out[1].blink);
        assert_eq!(out[0].gaze,[0.2,0.1,1.0]); assert_eq!(out[0].brow,0.4); assert!(out[0].brow_valid);
        s.eyes[0].opening=Some(1.5); s.apply_response(&mut out,now,p);
        assert_eq!(out[0].wide,1.0); assert_eq!(out[0].openness,1.0); assert!(!out[0].blink);
        s.eyes[0].status=1; s.eyes[0].opening=Some(0.0); s.apply_response(&mut out,now,p);
        assert_eq!(out[0].wide,0.0); assert!(out[0].blink); assert_eq!(out[0].squeeze,0.0);
    }
    #[test]
    fn uncertain_narrow_and_stale_never_become_a_fake_closed_or_wide() {
        let now=Instant::now();
        for status in [2,3,4,5] {
            let s=Snapshot {enabled:true,avatar:true,captured_at:Some(now),
                eyes:[EyePrediction {status,opening:Some(2.0),..Default::default()};2],..Default::default()};
            let mut out=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
            s.apply(&mut out,now);
            assert!(out.iter().all(|r| !r.openness_valid && !r.blink && r.wide==0.0));
        }
        let s=Snapshot {enabled:true,avatar:true,captured_at:Some(now),
            eyes:[EyePrediction {status:0,opening:Some(2.0),..Default::default()};2],..Default::default()};
        let mut out=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
        s.apply(&mut out,now+Duration::from_millis(201));
        assert!(out.iter().all(|r| !r.openness_valid && r.wide==0.0));
    }
    #[test]
    fn unknown_and_stale_never_force_closed_or_change_gaze() {
        let mut s = Snapshot {
            enabled: true,
            avatar: true,
            captured_at: Some(Instant::now()),
            ..Default::default()
        };
        s.eyes[0] = EyePrediction {
            status: 0,
            opening: Some(0.4),
            ..Default::default()
        };
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        out[0].gaze = [0.2, 0.3, 1.];
        s.apply(&mut out, Instant::now());
        assert_eq!(out[0].openness, 0.4);
        assert!(out[0].openness_valid);
        assert!(!out[1].openness_valid);
        assert_eq!(out[1].openness, 1.);
        assert_eq!(out[0].gaze, [0.2, 0.3, 1.]);
        s.apply(&mut out, Instant::now() + Duration::from_secs(1));
        assert!(!out[0].openness_valid);
        assert!(!out[0].blink);
    }
    #[test]
    fn preview_does_not_override_avatar() {
        let s = Snapshot {
            enabled: true,
            avatar: false,
            ..Default::default()
        };
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        out[0].openness = 0.3;
        s.apply(&mut out, Instant::now());
        assert_eq!(out[0].openness, 0.3);
    }
    #[test]
    fn missing_native_gaze_does_not_disable_image_eyelids() {
        let now = Instant::now();
        let s = Snapshot {
            enabled: true,
            avatar: true,
            captured_at: Some(now),
            eyes: [EyePrediction {
                status: 0,
                opening: Some(0.35),
                ..Default::default()
            }; 2],
            ..Default::default()
        };
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        // EyeResult starts without valid gaze. The independent image model owns
        // only openness validity; losing Tobii must not suppress a valid image.
        s.apply(&mut out, now);
        assert!(out.iter().all(|e| e.openness_valid && e.openness == 0.35));
    }
    #[test]
    fn parser_rejects_nan_status_and_truncation() {
        let mut b = vec![0; 40];
        b[..4].copy_from_slice(b"X5R1");
        assert!(read_reply(&mut b.as_slice()).is_ok());
        b[14] = 9;
        assert!(read_reply(&mut b.as_slice()).is_err());
        b[14] = 0;
        b[15..19].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(read_reply(&mut b.as_slice()).is_err());
        assert!(read_reply(&mut &b[..20]).is_err());
    }
    #[test]
    fn common_adapter_keeps_native_gaze_but_uses_model_closure_evidence() {
        use crate::core::types::GazeSample;
        let now=Instant::now();
        let mut s=Snapshot {enabled:true,avatar:true,captured_at:Some(now),
            eyes:[EyePrediction{status:0,opening:Some(1.0),..Default::default()};2],..Default::default()};
        let mut g=GazeSample::default();g.left.gaze=[0.2,0.3,1.];g.left.pupil_mm=4.0;
        g.left.openness_reported=true;g.left.openness_valid=false;
        let copy=s.common_lid_sample(g,now);
        assert_eq!(copy.left.gaze,g.left.gaze);assert_eq!(copy.left.gaze_valid,g.left.gaze_valid);
        assert_eq!(copy.left.pupil_mm,4.0);assert!(copy.left.openness_valid);
        assert_eq!(s.common_frame(now)[0],[1.0,0.5,0.0,0.0,0.0]);
        s.eyes[1].status=1;s.eyes[1].opening=Some(0.0);
        assert_eq!(s.common_frame(now)[1],[1.0,0.1,0.0,0.0,0.0]);
        assert!(!s.common_lid_sample(g,now).right.openness_valid);
        assert!(s.common_lid_sample(g,now).right.openness_reported);
        assert!(!s.common_lid_sample(g,now+Duration::from_secs(1)).left.openness_reported);
        let mut out=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
        out[1].openness=0.37; out[1].blink=false;
        s.guard_common_output(&mut out,now);assert_eq!(out[1].openness,0.37);assert!(!out[1].blink);
        s.guard_common_output(&mut out,now+Duration::from_secs(1));
        assert!(out.iter().all(|e| e.openness==1.0 && !e.openness_valid && e.wide==0.0 && !e.blink));
    }
    fn recovery_snapshot(now: Instant, count: u64, status: u8) -> Snapshot {
        Snapshot {enabled:true,avatar:true,count,captured_at:Some(now),
            eyes:[EyePrediction{status,opening:Some(if status==1 {0.0} else {1.0}),..Default::default()};2],
            ..Default::default()}
    }
    fn recovery_results(open: f32) -> [EyeResult;2] {
        let mut r=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
        for e in &mut r {e.openness=open;e.openness_valid=true;e.blink=open<0.08;}
        r
    }
    #[test]
    fn recovery_short_loss_holds_long_loss_releases_and_other_eye_tracks() {
        let t=Instant::now();let mut recovery=OutputRecovery::default();
        let mut r=recovery_results(0.25);recovery.apply(&recovery_snapshot(t,1,0),&mut r,t);
        for ms in [10,50,130,250,2000] {
            let now=t+Duration::from_millis(ms);let mut s=recovery_snapshot(now,ms,0);
            s.eyes[0].status=5;s.eyes[0].opening=None;
            let mut r=recovery_results(0.7);s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
            let a=((ms as f32-10.0-60.0)/180.0).clamp(0.0,1.0);
            assert!((r[0].openness-(0.25+0.75*a)).abs()<1e-6);
            assert!(r[0].openness_valid,"VRCFT must receive the bounded fallback");
            assert_eq!(r[1].openness,0.7);assert_eq!(r[0].wide,0.0);
        }
    }
    #[test]
    fn recovery_requires_distinct_predictions_and_bounds_reentry() {
        let t=Instant::now();let mut recovery=OutputRecovery::default();
        let mut r=recovery_results(1.0);recovery.apply(&recovery_snapshot(t,1,3),&mut r,t);
        for ms in [300,330,340] {
            let now=t+Duration::from_millis(ms);let mut r=recovery_results(0.0);
            recovery.apply(&recovery_snapshot(now,2,1),&mut r,now);
            assert_eq!(r[0].openness,1.0,"repeating one reply cannot confirm recovery");
        }
        let mut last=1.0;
        for ms in (350..=430).step_by(10) {
            let now=t+Duration::from_millis(ms);let mut r=recovery_results(0.0);
            recovery.apply(&recovery_snapshot(now,ms,1),&mut r,now);
            assert!((last-r[0].openness).abs()<=1.0/6.0+1e-5);
            last=r[0].openness;
        }
        assert_eq!(last,0.0);
    }
    #[test]
    fn recovery_does_not_smooth_normal_blinks_or_override_core_closing_slew() {
        let t=Instant::now();let mut recovery=OutputRecovery::default();
        for (i,value) in [1.0,0.7,0.2,0.0,0.4,1.0].into_iter().enumerate() {
            let now=t+Duration::from_millis(i as u64*8);
            let mut r=recovery_results(value);let s=recovery_snapshot(now,i as u64,if value<=0.2 {1} else {0});
            s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
            assert_eq!(r[0].openness,value);
        }
    }
    #[test]
    fn recovery_handles_stale_startup_and_flapping_without_restarting_hold() {
        let t=Instant::now();let mut recovery=OutputRecovery::default();
        let s=recovery_snapshot(t,1,0);let mut r=recovery_results(0.0);
        recovery.apply(&s,&mut r,t+Duration::from_secs(1));assert_eq!(r[0].openness,1.0);
        for i in 1..100 {
            let now=t+Duration::from_secs(1)+Duration::from_millis(i*10);
            let mut r=recovery_results(0.0);
            recovery.apply(&recovery_snapshot(now,i,if i%2==0 {0} else {4}),&mut r,now);
            assert_eq!(r[0].openness,1.0);
        }
    }
    #[test]
    fn recovery_missing_evidence_cannot_learn_endpoints_or_close_partner() {
        use crate::core::{eye_state::SRanipalState,types::{GazeSample,GazeEyelidProfile,WinkProfile,BlinkTimingProfile}};
        let mut state=SRanipalState::new();let mut g=GazeSample::default();
        g.left.openness_reported=true;g.left.openness_valid=true;
        g.right.openness_reported=true;g.right.openness_valid=true;
        let full=[[1.,0.5,0.,0.,0.];2];let p=GazeEyelidProfile::default();let w=WinkProfile::default();let b=BlinkTimingProfile::default();
        for _ in 0..400 {state.process_frame_with_model_evidence(full,&g,true,&p,&w,&b);}
        let before=toml::to_string(&state.snapshot_all().left).unwrap();
        let mut missing=full;missing[0]=[0.,0.5,0.,0.,0.];g.left.openness_reported=false;
        for _ in 0..600 {
            let r=state.process_frame_with_model_evidence(missing,&g,true,&p,&w,&b);
            assert!(r[1].openness>0.9,"lost left eye must not close right eye");
        }
        assert_eq!(before,toml::to_string(&state.snapshot_all().left).unwrap());
        assert_eq!(state.eyelid_live_diag()[0].reason,crate::core::eye_state::ClosureReason::TrackingLost);
        missing[1][1]=0.1;g.right.openness_valid=false;
        let mut last=1.;for _ in 0..100 {last=state.process_frame_with_model_evidence(missing,&g,true,&p,&w,&b)[1].openness;}
        assert!(last<0.05,"tracked eye must still blink independently");
    }
    #[test]
    fn recovery_closed_conflict_preserves_a_measured_wink_and_bounds_reentry() {
        for eye in 0..2 {
            let t=Instant::now();let mut recovery=OutputRecovery::default();
            let mut r=recovery_results(0.0);
            recovery.apply(&recovery_snapshot(t,1,1),&mut r,t);
            for ms in (8..=5000).step_by(8) {
                let now=t+Duration::from_millis(ms);
                let mut s=recovery_snapshot(now,ms,0);
                s.eyes[eye]=EyePrediction{status:3,opening:None,closed_score:0.95,..Default::default()};
                assert_eq!(s.common_frame(now)[eye][0],0.0,"not calibration evidence");
                let mut r=recovery_results(0.63);
                s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
                assert_eq!(r[eye].openness,0.0);
                assert_eq!(r[eye].wide,0.0);
                assert_eq!(r[1-eye].openness,0.63);
            }
            let mut last=0.0;
            for ms in (5008..=5072).step_by(8) {
                let now=t+Duration::from_millis(ms);
                let mut r=recovery_results(0.37);
                recovery.apply(&recovery_snapshot(now,ms,0),&mut r,now);
                assert!(r[eye].openness>=last && r[eye].openness-last<=8.0/60.0+1e-5);
                last=r[eye].openness;
            }
            assert_eq!(last,0.37);
        }
    }
    #[test]
    fn recovery_closed_conflict_never_initiates_closure_of_a_narrow_open_eye() {
        for initial in [0.1,0.4,1.0] {
            let t=Instant::now();let mut recovery=OutputRecovery::default();
            let mut r=recovery_results(initial);
            recovery.apply(&recovery_snapshot(t,1,0),&mut r,t);
            for ms in (8..=1000).step_by(8) {
                let now=t+Duration::from_millis(ms);
                let mut s=recovery_snapshot(now,ms,3);
                for p in &mut s.eyes {p.closed_score=0.99;p.opening=None;}
                let mut r=recovery_results(0.0);
                s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
                assert_eq!(r[0].openness,initial,"Unknown cannot close or open the last valid pose");
            }
        }
    }
    #[test]
    fn recovery_closed_conflict_expires_on_stale_transport_or_unsupported_evidence() {
        for (status,score,stale) in [(3,0.99,true),
            (5,0.99,false),(3,f32::NAN,false),(9,0.89,false),(3,1.1,false)] {
            let t=Instant::now();let mut recovery=OutputRecovery::default();
            let mut r=recovery_results(0.0);recovery.apply(&recovery_snapshot(t,1,1),&mut r,t);
            let mut last=0.0;
            for ms in (8..=1000).step_by(8) {
                let now=t+Duration::from_millis(ms);
                let mut s=recovery_snapshot(if stale {t} else {now},ms,status);
                for p in &mut s.eyes {p.closed_score=score;p.opening=None;}
                let mut r=recovery_results(0.0);
                s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
                assert!(r[0].openness>=last);last=r[0].openness;
            }
            assert_eq!(last,1.0,"status={status}, score={score}, stale={stale}");
        }
    }
    #[test]
    fn recovery_fresh_uncertainty_is_not_a_timeout_to_open() {
        for gap in [32,320] {
            let t=Instant::now();let mut recovery=OutputRecovery::default();
            let mut r=recovery_results(0.0);recovery.apply(&recovery_snapshot(t,1,1),&mut r,t);
            for ms in (8..=gap+320).step_by(8) {
                let now=t+Duration::from_millis(ms);
                let mut s=recovery_snapshot(now,ms,if ms<=gap {4} else {3});
                for p in &mut s.eyes {p.closed_score=if ms<=gap {0.5} else {0.99};p.opening=None;}
                let mut r=recovery_results(0.0);
                s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
                assert_eq!(r[0].openness,0.0,"fresh uncertain images do not mean opening");
            }
        }
    }
    #[test]
    fn recovery_recorded_partial_winks_do_not_reopen_during_fresh_uncertainty() {
        // Delivered values just before the 2026-09-29 recording lost per-eye
        // model evidence. Neither wink reached the former 8% hold gate.
        let traces: [&[f32];2] = [
            &[1.0,0.9489,0.8676,0.7777,0.6804,0.5622,0.4399,0.2397,0.1897],
            &[1.0,0.9882,0.9522,0.8964,0.8128,0.7401,0.6487,0.5393,0.4276,0.2936,0.1366],
        ];
        for eye in 0..2 { for status in 2..=4 {
            let t=Instant::now();let mut recovery=OutputRecovery::default();
            for (n,&v) in traces[eye].iter().enumerate() {
                let now=t+Duration::from_millis(n as u64*9);
                let mut r=recovery_results(1.0);r[eye].openness=v;
                recovery.apply(&recovery_snapshot(now,n as u64+1,0),&mut r,now);
            }
            let start=traces[eye].len() as u64*9;
            let expected=*traces[eye].last().unwrap();
            for ms in (start..start+5000).step_by(9) {
                let now=t+Duration::from_millis(ms);
                let mut s=recovery_snapshot(now,ms+1,0);
                s.eyes[eye]=EyePrediction{status,opening:None,closed_score:if status==3 {0.95} else {0.2},..Default::default()};
                let mut r=recovery_results(0.91);
                s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
                assert_eq!(r[eye].openness,expected);
                assert_eq!(r[1-eye].openness,0.91);
                assert_eq!(recovery.modes()[eye],1);
                assert_eq!(s.common_frame(now)[eye][0],0.0,"held output is never measured evidence");
            }
        }}
    }
    #[test]
    fn recovery_fresh_uncertainty_releases_when_images_actually_stop() {
        let t=Instant::now();let mut recovery=OutputRecovery::default();
        let mut r=recovery_results(0.19);recovery.apply(&recovery_snapshot(t,1,0),&mut r,t);
        let mut s=recovery_snapshot(t+Duration::from_millis(8),2,4);
        for p in &mut s.eyes {p.opening=None;}
        for ms in (8..=1000).step_by(8) {
            let now=t+Duration::from_millis(ms);let mut r=recovery_results(0.0);
            s.guard_common_output(&mut r,now);recovery.apply(&s,&mut r,now);
            if ms<=208 {assert_eq!(r[0].openness,0.19);}
            if ms>=464 {assert_eq!(r[0].openness,1.0);assert_eq!(recovery.modes()[0],2);}
        }
        // New uncertain frames cannot revive the pre-outage held pose.
        s.captured_at=Some(t+Duration::from_millis(1008));s.count=3;
        let mut r=recovery_results(0.0);
        recovery.apply(&s,&mut r,t+Duration::from_millis(1008));
        assert_eq!(r[0].openness,1.0);
    }
    #[test]
    fn native_scalar_uses_common_manual_endpoints_curve_and_wide_extension() {
        use crate::core::{eye_state::SRanipalState,types::GazeSample};
        let now=Instant::now();
        let mut s=Snapshot{enabled:true,avatar:true,captured_at:Some(now),
            eyes:[EyePrediction{status:0,opening:Some(1.0),..Default::default()};2],..Default::default()};
        let mut state=SRanipalState::new();
        state.set_eyelid_response_tuning(true,[0.03;2],[0.4;2],[1.0;2],[0.5;2],0.0,1.0,false);
        let mut g=GazeSample::default(); g.left.gaze_valid=true;g.right.gaze_valid=true;
        for _ in 0..400 {state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true);}
        s.eyes[0].opening=Some(0.6);s.eyes[1].opening=Some(0.6);
        let mut before=0.0;
        for _ in 0..100 {before=state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true)[0].openness;}
        state.set_eyelid_response_tuning(true,[0.03;2],[0.2;2],[1.0;2],[0.5;2],0.0,1.0,false);
        let mut after=0.0;
        for _ in 0..100 {after=state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true)[0].openness;}
        assert!(after<before-0.1,"manual endpoints must affect native output: {before} -> {after}");
        state.set_eyelid_response_tuning(true,[0.03;2],[0.4;2],[1.0;2],[0.2;2],0.0,1.0,false);
        let mut curve_low=0.0;
        for _ in 0..100 {curve_low=state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true)[0].openness;}
        state.set_eyelid_response_tuning(true,[0.03;2],[0.4;2],[1.0;2],[0.8;2],0.0,1.0,false);
        let mut curve_high=0.0;
        for _ in 0..100 {curve_high=state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true)[0].openness;}
        assert!(curve_high>curve_low+0.1,"response curve must remain active");
        s.eyes[0].opening=Some(1.7);s.eyes[1].opening=Some(1.7);
        let mut wide=0.0;
        for _ in 0..100 {wide=state.process_frame(s.common_frame(now),&s.common_lid_sample(g,now),true)[0].wide;}
        assert!(wide>0.0,"same scalar must extend into Wide");
    }
    #[test]
    fn independent_wide_never_changes_openness_or_uses_missing_reference() {
        let now = Instant::now();
        let mut s = Snapshot { enabled: true, avatar: true, captured_at: Some(now),
            eyes: [EyePrediction { status:0, opening:Some(0.6), wide_height:Some(60.0), ..Default::default() };2], ..Default::default() };
        let mut p = ResponseSettings { independent_wide:true, ..Default::default() };
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        s.apply_response(&mut out, now, p);
        assert_eq!(out[0].wide, 0.0);
        p.wide_reference = [40.0;2];
        s.apply_response(&mut out, now, p);
        assert_eq!(out[0].openness, 0.6); assert_eq!(out[0].wide, 1.0);
        for status in 1..=5 {
            s.eyes[0].status = status;
            s.apply_response(&mut out, now, p); assert_eq!(out[0].wide, 0.0);
        }
        s.eyes[0].status = 0; s.eyes[0].opening = Some(0.0);
        s.apply_response(&mut out, now, p); assert_eq!(out[0].wide, 0.0);
        s.eyes[0].opening = Some(1.0);
        s.apply_response(&mut out, now+Duration::from_secs(1), p); assert_eq!(out[0].wide, 0.0);
        for h in [None, Some(f32::NAN), Some(f32::INFINITY), Some(-1.0), Some(301.0)] {
            assert_eq!(p.map_independent_wide(0,h),0.0);
        }
        p.independent_wide = false;
        s.apply_response(&mut out, now, p); assert_eq!(out[0].wide, 0.0);
    }
    #[test]
    fn wide_reference_capture_counts_fresh_unique_frames_and_preserves_on_failure() {
        let start=Instant::now();
        let mut capture=WideReferenceCapture::new(start);
        let mut s=Snapshot { enabled:true, eyes:[EyePrediction {status:0,wide_height:Some(40.0),..Default::default()};2],..Default::default() };
        for i in 0..40 {
            let now=start+Duration::from_millis(2000+i*40);
            s.captured_at=Some(now); s.count=i;
            let r=capture.tick(&s,now);
            if i<38 { assert!(r.is_none()); } else { assert_eq!(r,Some(Ok([40.0;2]))); break; }
        }
        let mut stalled=WideReferenceCapture::new(start);
        for i in 0..100 {
            let now=start+Duration::from_millis(2000+i*35); s.captured_at=Some(now);
            assert!(stalled.tick(&s,now).is_none()); // same count cannot create samples
        }
        assert!(stalled.tick(&s,start+Duration::from_secs(6)).unwrap().is_err());
        assert!(stalled.samples.iter().all(|v| v.len()==1));
    }
    #[test]
    fn wide_reference_rejects_unstable_or_unknown_eyes() {
        let start=Instant::now(); let mut c=WideReferenceCapture::new(start);
        let mut s=Snapshot {enabled:true,..Default::default()};
        for i in 0..39 {
            let now=start+Duration::from_millis(2000+i*40);s.count=i;s.captured_at=Some(now);
            s.eyes=[EyePrediction{status:0,wide_height:Some(if i%2==0 {30.0} else {50.0}),..Default::default()};2];
            let r=c.tick(&s,now); if i==38 {assert!(r.unwrap().is_err());}
        }
    }
    #[test]
    fn v2_reply_preserves_old_fields_and_rejects_bad_wide() {
        let mut b=[0u8;48]; b[..4].copy_from_slice(b"X5R2");
        b[27..31].copy_from_slice(&41.5f32.to_le_bytes());
        b[44..48].copy_from_slice(&32.0f32.to_le_bytes());
        let (_,eyes,_,_)=read_reply(&mut b.as_slice()).unwrap();
        assert_eq!(eyes[0].wide_height,Some(41.5)); assert_eq!(eyes[1].wide_height,Some(32.0));
        assert!(read_reply(&mut &b[..47]).is_err());
        b[27..31].copy_from_slice(&f32::NAN.to_le_bytes());assert!(read_reply(&mut b.as_slice()).is_err());
        let mut old=[0u8;40];old[..4].copy_from_slice(b"X5R1");
        assert_eq!(read_reply(&mut old.as_slice()).unwrap().1[0].wide_height,None);
    }
    #[test]
    fn stale_or_skewed_camera_pair_is_rejected() {
        let now = Instant::now();
        let l = RawFrame {
            generation: 1,
            at: now,
            pixels: Arc::from(vec![0; 40000]),
        };
        let mut r = l.clone();
        assert!(coherent_pair(&l, &r, now));
        r.at = now - Duration::from_millis(60);
        assert!(!coherent_pair(&l, &r, now));
        assert!(!coherent_pair(&l, &l, now + Duration::from_millis(101)));
    }

    #[test]
    #[ignore = "Requires the private local model and Python environment; no camera is opened"]
    fn real_worker_roundtrip_cpu_gpu_and_shutdown() {
        let env = |key| std::env::var(key).expect(key);
        let mut images = Vec::new();
        for key in ["XR5_TEST_LEFT", "XR5_TEST_RIGHT"] {
            let mut reader = png::Decoder::new(std::fs::File::open(env(key)).unwrap())
                .read_info()
                .unwrap();
            let mut pixels = vec![0; reader.output_buffer_size()];
            let info = reader.next_frame(&mut pixels).unwrap();
            assert_eq!(
                (info.width, info.height, info.color_type),
                (200, 200, png::ColorType::Grayscale)
            );
            pixels.truncate(info.buffer_size());
            images.push(pixels);
        }
        for gpu in [false, true] {
            let settings = Settings {
                enabled: true,
                avatar: true,
                gpu,
                bundle: env("XR5_TEST_BUNDLE"),
                python: env("XR5_TEST_PYTHON"),
                python_path: env("PYTHONPATH"),
                ..Default::default()
            };
            let mut bridge = Bridge::start(&settings).unwrap();
            let owner = Arc::downgrade(&bridge.process);
            let mut state = Snapshot {
                enabled: true,
                avatar: true,
                ..Default::default()
            };
            let start = Instant::now();
            while !bridge.available() {
                bridge.tick(&mut state);
                assert!(!bridge.failed, "{}", state.message);
                assert!(start.elapsed() < Duration::from_secs(40));
                std::thread::sleep(Duration::from_millis(5));
            }
            for frame in 0..3 {
                let blank = vec![0; 40000];
                let (left, right) = if frame == 2 {
                    (&blank, &blank)
                } else {
                    (&images[0], &images[1])
                };
                assert!(bridge.submit(left, right, Instant::now(), false));
                let start = Instant::now();
                while state.count < frame + 1 {
                    bridge.tick(&mut state);
                    assert!(!bridge.failed, "{}", state.message);
                    assert!(start.elapsed() < Duration::from_secs(5));
                    std::thread::sleep(Duration::from_millis(2));
                }
                if frame == 2 {
                    assert!(state.eyes.iter().all(|e| e.opening.is_none()));
                } else {
                    assert!(state.eyes.iter().all(|e| e.opening.is_some()));
                }
            }
            assert_eq!(state.backend, if gpu { "GPU" } else { "CPU" });
            // Kill while a request is outstanding: descendants must release pipes.
            assert!(bridge.submit(&images[0], &images[1], Instant::now(), false));
            drop(bridge);
            let stopped = Instant::now();
            while owner.upgrade().is_some() {
                assert!(
                    stopped.elapsed() < Duration::from_secs(5),
                    "worker tree leaked"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(!OCCUPIED.load(Ordering::Acquire));
        }
    }
}
