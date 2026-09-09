//! Immutable, per-headset wearing-position reference and live reseat guidance.
//!
//! This is deliberately not another calibration loop. A reference changes only
//! after the user explicitly confirms that the current fit is good. Live matching
//! never edits geometry, photometric correction, eyelid endpoints, or baselines.
//!
//! Matching uses a compact census descriptor over temporally stable eye-camera
//! structure. Census comparisons are insensitive to monotonic brightness/contrast
//! changes; very dark pupil pixels, saturated IR glints, borders, and unstable
//! capture pixels receive no weight. Both eyes must independently agree before a
//! physical direction is shown.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::config::{canonical_device_key, EyeMapping};
use crate::core::types::MlGeometry;
use crate::geometry_calib::GazeTarget;
use crate::pipeline::EyeFrame;

const MAGIC: &[u8; 8] = b"SRSEAT01";
const SCHEMA_VERSION: u16 = 2;
pub const TEMPLATE_SIDE: usize = 48;
const TEMPLATE_PIXELS: usize = TEMPLATE_SIDE * TEMPLATE_SIDE;
const CAPTURE_SECONDS: f32 = 7.5;
/// A brief evidence-driven tail prevents a busy UI frame from discarding an
/// otherwise complete reference. The user simply keeps looking straight ahead.
const CAPTURE_GRACE_SECONDS: f32 = 3.0;
const CAPTURE_SETTLE_END: f32 = 0.8;
const CAPTURE_FIRST_CENTRE_END: f32 = 2.0;
const CAPTURE_SWEEP_END: f32 = 4.8;
const CAPTURE_BLINK_END: f32 = 6.3;
// The controller is polled by a nominal 60 Hz UI. 35 ms aliases to every third
// repaint (about 20 Hz), while 30 ms accepts every second repaint (about 30 Hz)
// without cloning anything close to the 120 Hz source rate.
const CAPTURE_INTERVAL: Duration = Duration::from_millis(30);
const MIN_CAPTURE_FRAMES: usize = 120;
const MIN_CENTRE_FRAMES: usize = 45;
const MAX_CAPTURE_FRAMES: usize = 210;
const ANALYZE_INTERVAL: Duration = Duration::from_millis(180);
const MIN_WEIGHTED_PIXELS: usize = 180;
const MIN_EYE_SCORE: f32 = 0.66;
const MIN_EYE_MARGIN: f32 = 0.012;
const MIN_FEATURE_COVERAGE: f32 = 0.72;
const SEARCH_SHIFT_LIMIT: f32 = 7.0;
const SEARCH_ROTATION_LIMIT: f32 = 6.0;
const SEARCH_SCALE_MIN: f32 = 0.855;
const SEARCH_SCALE_MAX: f32 = 1.145;
const MAX_STEREO_SHIFT_DISAGREEMENT: f32 = 3.0;
const MAX_STEREO_SCALE_DISAGREEMENT: f32 = 0.065;
const MAX_STEREO_ROTATION_DISAGREEMENT_DEG: f32 = 5.0;
const ALIGNED_SHIFT_PX: f32 = 1.0;
const ALIGNED_SCALE_ERROR: f32 = 0.028;
const ALIGNED_ROTATION_DEG: f32 = 1.5;
const ALIGNED_HOLD: Duration = Duration::from_secs(1);

/// A wearing-position reference is a user-owned local artifact, not a tracking
/// calibration parameter. The assist reads frames and writes only its own file.
pub const CHANGE_DOMAINS: crate::calib_session::DomainSet = crate::calib_session::DomainSet::NONE;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceContext {
    pub device_key: String,
    pub unit_id: String,
    pub image_fingerprint: u64,
}

impl ReferenceContext {
    pub fn new(device_key: &str, unit_id: String, image_fingerprint: u64) -> Self {
        Self {
            device_key: canonical_device_key(device_key),
            unit_id,
            image_fingerprint,
        }
    }
}

/// Fingerprint only mapping settings applied to `Telemetry.frames`. ML crop,
/// rotation, per-eye mirror, brightness and photometry happen later in the pipeline
/// and therefore must not invalidate this raw-camera wearing-position reference.
pub fn image_fingerprint(mapping: EyeMapping, _geometry: [MlGeometry; 2]) -> u64 {
    let mut hash = Fnv64::new();
    hash.bytes(b"reseat-raw-image-context-v2");
    hash.byte(mapping.swap_eyes as u8);
    hash.byte(mapping.flip_image as u8);
    hash.finish()
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct EyeAxes {
    upright_deg: f32,
    handedness: f32,
    scale_y: f32,
}

fn device_axes(device_key: &str) -> [EyeAxes; 2] {
    crate::config::default_ml_geometry(device_key).map(|geometry| EyeAxes {
        upright_deg: geometry.rotate_deg,
        handedness: if geometry.mirror_h.unwrap_or(false) {
            -1.0
        } else {
            1.0
        },
        scale_y: geometry.scale_y.abs().max(0.25),
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GazePoint {
    Direction([f32; 3]),
}

impl GazePoint {
    fn finite(self) -> bool {
        match self {
            Self::Direction(value) => value.iter().all(|value| value.is_finite()),
        }
    }
}

#[derive(Clone, Debug)]
struct EyeTemplate {
    source_width: u32,
    source_height: u32,
    median: Vec<u8>,
    weights: Vec<u8>,
    census: Vec<u8>,
    feature_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Reference {
    pub context: ReferenceContext,
    pub captured_unix: u64,
    eyes: [EyeTemplate; 2],
}

impl Reference {
    pub fn source_dimensions(&self) -> [[u32; 2]; 2] {
        self.eyes
            .each_ref()
            .map(|eye| [eye.source_width, eye.source_height])
    }

    pub fn save(&self) -> Result<PathBuf, String> {
        let path = reference_path(&self.context);
        save_to(&path, self)?;
        Ok(path)
    }

    pub fn load(context: &ReferenceContext) -> Result<Option<Self>, String> {
        let path = reference_path(context);
        let Some(reference) = load_with_backup(&path)? else {
            return Ok(None);
        };
        if reference.context.device_key != context.device_key
            || reference.context.unit_id != context.unit_id
        {
            return Err("saved wearing-position reference belongs to another headset".into());
        }
        if reference.context.image_fingerprint != context.image_fingerprint {
            return Err(
                "image mapping or geometry changed after the wearing-position reference was saved"
                    .into(),
            );
        }
        Ok(Some(reference))
    }

    pub fn remove(context: &ReferenceContext) -> Result<bool, String> {
        let path = reference_path(context);
        let mut removed = false;
        for candidate in [
            path.clone(),
            path.with_extension("bak"),
            path.with_extension("partial"),
        ] {
            if candidate.exists() {
                fs::remove_file(&candidate).map_err(|error| {
                    format!("could not remove {}: {error}", candidate.display())
                })?;
                removed = true;
            }
        }
        Ok(removed)
    }
}

#[derive(Clone, Debug, Default)]
pub enum Guidance {
    #[default]
    WaitingForFrames,
    CapturingReference {
        progress: f32,
        frames: usize,
        step: CaptureStep,
    },
    ReferenceReady {
        weighted_pixels: [usize; 2],
    },
    ReferenceSaved {
        path: PathBuf,
    },
    EyesClosed,
    LowConfidence {
        detail: String,
    },
    Adjust(Estimate),
    Aligned(Estimate),
    Error(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureStep {
    Settling,
    CentreBefore,
    EyeSweep,
    BlinkTwice,
    CentreAfter,
}

impl CaptureStep {
    pub fn instruction(self) -> &'static str {
        match self {
            Self::Settling => "Look straight ahead and let the image settle. Keep the HMD still.",
            Self::CentreBefore => {
                "Look straight ahead with both eyes comfortably open. Keep the HMD still."
            }
            Self::EyeSweep => {
                "Keep your head and HMD still. Slowly look left, right, up and down with your eyes."
            }
            Self::BlinkTwice => {
                "Keep looking straight ahead and blink normally twice. Do not squeeze your eyes."
            }
            Self::CentreAfter => {
                "Look straight ahead again with both eyes comfortably open. Keep still."
            }
        }
    }

    /// Target shown by the SteamVR capture guide. The eye-sweep target moves
    /// through all four directions while the head and HMD remain stationary.
    pub fn target(self, overall_progress: f32) -> Option<GazeTarget> {
        match self {
            Self::Settling | Self::CentreBefore | Self::BlinkTwice | Self::CentreAfter => {
                Some(GazeTarget::Center)
            }
            Self::EyeSweep => {
                let elapsed = overall_progress.clamp(0.0, 1.0) * CAPTURE_SECONDS;
                let sweep_progress = ((elapsed - CAPTURE_FIRST_CENTRE_END)
                    / (CAPTURE_SWEEP_END - CAPTURE_FIRST_CENTRE_END))
                    .clamp(0.0, 0.999);
                Some(match (sweep_progress * 4.0) as usize {
                    0 => GazeTarget::Left,
                    1 => GazeTarget::Right,
                    2 => GazeTarget::Up,
                    _ => GazeTarget::Down,
                })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct EyeMatch {
    pub shift_px: [f32; 2],
    pub scale: f32,
    pub rotation_deg: f32,
    pub score: f32,
    pub margin: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Estimate {
    /// Physical correction in a normalized, frontal image plane. Positive X/Y means
    /// move the HMD right/down. Units are 48x48 template pixels.
    pub correction_px: [f32; 2],
    /// A value above 1.0 means the current eye image is larger than the saved fit
    /// (move the HMD slightly farther away); below 1.0 means move it closer.
    pub scale: f32,
    /// Positive means rotate the HMD clockwise as seen by the wearer.
    pub rotation_deg: f32,
    pub confidence: f32,
    pub eyes: [EyeMatch; 2],
}

impl Estimate {
    pub fn is_aligned(self) -> bool {
        self.correction_px[0].abs() <= ALIGNED_SHIFT_PX
            && self.correction_px[1].abs() <= ALIGNED_SHIFT_PX
            && (self.scale - 1.0).abs() <= ALIGNED_SCALE_ERROR
            && self.rotation_deg.abs() <= ALIGNED_ROTATION_DEG
            && self.confidence >= 0.55
    }

    pub fn instruction(self) -> String {
        // Depth and roll alter the apparent x/y displacement, so guide one dominant
        // physical adjustment at a time in that order. This is much easier to follow
        // inside VR and avoids rapidly alternating two instructions near a boundary.
        if (self.scale - 1.0).abs() > ALIGNED_SCALE_ERROR {
            return if self.scale > 1.0 {
                "move it slightly farther from your eyes".into()
            } else {
                "move it slightly closer to your eyes".into()
            };
        }
        if self.rotation_deg.abs() > ALIGNED_ROTATION_DEG {
            return if self.rotation_deg > 0.0 {
                "rotate it clockwise"
            } else {
                "rotate it counter-clockwise"
            }
            .into();
        }
        if self.correction_px[1].abs() > ALIGNED_SHIFT_PX {
            return if self.correction_px[1] > 0.0 {
                "move the HMD down"
            } else {
                "move the HMD up"
            }
            .into();
        }
        if self.correction_px[0].abs() > ALIGNED_SHIFT_PX {
            if self.correction_px[0] > 0.0 {
                "move the HMD right".into()
            } else {
                "move the HMD left".into()
            }
        } else {
            "Hold this position".into()
        }
    }
}

struct ReferenceCapture {
    context: ReferenceContext,
    geometry: [MlGeometry; 2],
    started: Instant,
    last_sample: Option<Instant>,
    last_generation: [u64; 2],
    source_dimensions: Option<[[u32; 2]; 2]>,
    centre_frames: [Vec<Vec<u8>>; 2],
    all_frames: [Vec<Vec<u8>>; 2],
    centre_gaze: [Vec<GazePoint>; 2],
    sweep_gaze: [Vec<GazePoint>; 2],
    blink_count: usize,
    was_blinking: bool,
    blink_signal_seen: bool,
}

impl ReferenceCapture {
    fn new(context: ReferenceContext, geometry: [MlGeometry; 2], now: Instant) -> Self {
        Self {
            context,
            geometry,
            started: now,
            last_sample: None,
            last_generation: [0; 2],
            source_dimensions: None,
            centre_frames: std::array::from_fn(|_| Vec::new()),
            all_frames: std::array::from_fn(|_| Vec::new()),
            centre_gaze: std::array::from_fn(|_| Vec::new()),
            sweep_gaze: std::array::from_fn(|_| Vec::new()),
            blink_count: 0,
            was_blinking: false,
            blink_signal_seen: false,
        }
    }

    fn step(&self, now: Instant) -> CaptureStep {
        let elapsed = now.duration_since(self.started).as_secs_f32();
        if elapsed < CAPTURE_SETTLE_END {
            CaptureStep::Settling
        } else if elapsed < CAPTURE_FIRST_CENTRE_END {
            CaptureStep::CentreBefore
        } else if elapsed < CAPTURE_SWEEP_END {
            CaptureStep::EyeSweep
        } else if elapsed < CAPTURE_BLINK_END {
            CaptureStep::BlinkTwice
        } else {
            CaptureStep::CentreAfter
        }
    }

    fn progress(&self, now: Instant) -> f32 {
        (now.duration_since(self.started).as_secs_f32() / CAPTURE_SECONDS).clamp(0.0, 1.0)
    }

    fn ingest(
        &mut self,
        frames: &[Option<EyeFrame>; 2],
        gaze: [Option<GazePoint>; 2],
        blinking: Option<bool>,
        now: Instant,
    ) -> Result<Option<Reference>, String> {
        // A bounded pipeline history may still contain pairs published just before
        // the user pressed Start. Advance the cursor, but never label them as part
        // of this wearing-position reference.
        if now < self.started {
            if let [Some(left), Some(right)] = frames {
                self.last_generation = [left.generation, right.generation];
            }
            return Ok(None);
        }
        if self
            .last_sample
            .is_some_and(|last| now.duration_since(last) < CAPTURE_INTERVAL)
        {
            return Ok(None);
        }
        let [Some(left), Some(right)] = frames else {
            return self.finish_if_due(now);
        };
        let pair = [left, right];
        let generations = [left.generation, right.generation];
        if generations == self.last_generation {
            return self.finish_if_due(now);
        }
        let dimensions = [[left.width, left.height], [right.width, right.height]];
        if self
            .source_dimensions
            .is_some_and(|expected| expected != dimensions)
        {
            return Err("eye-camera dimensions changed while saving the reference".into());
        }
        self.source_dimensions = Some(dimensions);
        let step = self.step(now);
        if step == CaptureStep::BlinkTwice {
            if let Some(blinking) = blinking {
                self.blink_signal_seen = true;
                if blinking && !self.was_blinking {
                    self.blink_count += 1;
                }
                self.was_blinking = blinking;
            }
        } else {
            self.was_blinking = false;
        }
        for eye in 0..2 {
            let expected = pair[eye].width as usize * pair[eye].height as usize;
            if pair[eye].width == 0 || pair[eye].height == 0 || pair[eye].pixels.len() < expected {
                return Err("an eye-camera frame was incomplete".into());
            }
            if step != CaptureStep::Settling && self.all_frames[eye].len() < MAX_CAPTURE_FRAMES {
                let sampled = downsample(
                    pair[eye].pixels.as_ref(),
                    pair[eye].width as usize,
                    pair[eye].height as usize,
                );
                self.all_frames[eye].push(sampled.clone());
                if step != CaptureStep::EyeSweep {
                    self.centre_frames[eye].push(sampled);
                }
                if let Some(point) = gaze[eye].filter(|point| point.finite()) {
                    if step == CaptureStep::EyeSweep {
                        self.sweep_gaze[eye].push(point);
                    } else {
                        self.centre_gaze[eye].push(point);
                    }
                }
            }
        }
        self.last_generation = generations;
        self.last_sample = Some(now);
        self.finish_if_due(now)
    }

    fn finish_if_due(&mut self, now: Instant) -> Result<Option<Reference>, String> {
        let elapsed = now.duration_since(self.started).as_secs_f32();
        if elapsed < CAPTURE_SECONDS {
            return Ok(None);
        }
        let missing_capture = self
            .all_frames
            .iter()
            .any(|frames| frames.len() < MIN_CAPTURE_FRAMES);
        let missing_centre = self
            .centre_frames
            .iter()
            .any(|frames| frames.len() < MIN_CENTRE_FRAMES);
        if (missing_capture || missing_centre) && elapsed < CAPTURE_SECONDS + CAPTURE_GRACE_SECONDS
        {
            return Ok(None);
        }
        if missing_capture {
            return Err(format!(
                "only {}/{} fresh stereo frames were available; keep the cameras running and try again",
                self.all_frames.iter().map(Vec::len).min().unwrap_or(0),
                MIN_CAPTURE_FRAMES
            ));
        }
        if missing_centre {
            return Err(format!(
                "only {}/{} straight-ahead frames were captured; keep the HMD still and try again",
                self.centre_frames.iter().map(Vec::len).min().unwrap_or(0),
                MIN_CENTRE_FRAMES
            ));
        }
        validate_gaze_sweep(&self.centre_gaze, &self.sweep_gaze)?;
        if self.blink_signal_seen && self.blink_count < 2 {
            return Err(format!(
                "only {} of 2 normal blinks were observed; blink twice without squeezing and record again",
                self.blink_count
            ));
        }
        let dimensions = self
            .source_dimensions
            .ok_or_else(|| "no stereo eye-camera frames were received".to_string())?;
        let eyes: [Result<EyeTemplate, String>; 2] = std::array::from_fn(|eye| {
            build_template(
                dimensions[eye][0],
                dimensions[eye][1],
                &self.centre_frames[eye],
                &self.all_frames[eye],
                &self.context.device_key,
                eye,
                self.geometry[eye],
            )
        });
        let eyes = [eyes[0].clone()?, eyes[1].clone()?];
        let reference = Reference {
            context: self.context.clone(),
            captured_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            eyes,
        };
        Ok(Some(reference))
    }
}

/// UI-side controller. Work is bounded to one compact stereo registration every
/// 180 ms; the 120 Hz camera and ML threads are never blocked.
pub struct Assist {
    context: ReferenceContext,
    axes: [EyeAxes; 2],
    reference: Option<Reference>,
    pending_reference: Option<Reference>,
    capture: Option<ReferenceCapture>,
    active: bool,
    guidance: Guidance,
    load_warning: Option<String>,
    last_analysis: Option<Instant>,
    last_generation: [u64; 2],
    smoothed: Option<Estimate>,
    aligned_since: Option<Instant>,
    last_valid_estimate: Option<Instant>,
}

impl Assist {
    pub fn load(context: ReferenceContext) -> Self {
        let (reference, load_warning) = match Reference::load(&context) {
            Ok(reference) => (reference, None),
            Err(error) => (None, Some(error)),
        };
        Self {
            axes: device_axes(&context.device_key),
            context,
            reference,
            pending_reference: None,
            capture: None,
            active: false,
            guidance: Guidance::WaitingForFrames,
            load_warning,
            last_analysis: None,
            last_generation: [0; 2],
            smoothed: None,
            aligned_since: None,
            last_valid_estimate: None,
        }
    }

    pub fn context(&self) -> &ReferenceContext {
        &self.context
    }

    pub fn has_reference(&self) -> bool {
        self.reference.is_some()
    }

    pub fn has_pending_reference(&self) -> bool {
        self.pending_reference.is_some()
    }

    pub fn is_active(&self) -> bool {
        self.active || self.capture.is_some()
    }

    pub fn is_capturing(&self) -> bool {
        self.capture.is_some()
    }

    pub fn last_generation(&self) -> [u64; 2] {
        self.capture
            .as_ref()
            .map(|capture| capture.last_generation)
            .unwrap_or(self.last_generation)
    }

    pub fn discard_through(&mut self, generation: [u64; 2]) {
        if let Some(capture) = self.capture.as_mut() {
            capture.last_generation[0] = capture.last_generation[0].max(generation[0]);
            capture.last_generation[1] = capture.last_generation[1].max(generation[1]);
        }
        self.last_generation[0] = self.last_generation[0].max(generation[0]);
        self.last_generation[1] = self.last_generation[1].max(generation[1]);
    }

    /// Remove a native UI/event-loop hiatus from capture and guidance clocks.
    /// Frames generated while the instructions were frozen are intentionally skipped.
    pub fn suspend_for(&mut self, duration: Duration, generation: [u64; 2]) {
        if !self.is_active() || duration.is_zero() {
            return;
        }
        let now = Instant::now();
        if let Some(capture) = self.capture.as_mut() {
            capture.started = capture.started.checked_add(duration).unwrap_or(now);
            capture.last_sample = capture.last_sample.and_then(|at| at.checked_add(duration));
            capture.last_generation = generation;
        }
        self.last_analysis = self.last_analysis.and_then(|at| at.checked_add(duration));
        self.aligned_since = self.aligned_since.and_then(|at| at.checked_add(duration));
        self.last_valid_estimate = self
            .last_valid_estimate
            .and_then(|at| at.checked_add(duration));
        self.last_generation = generation;
    }

    pub fn guidance(&self) -> &Guidance {
        &self.guidance
    }

    pub fn load_warning(&self) -> Option<&str> {
        self.load_warning.as_deref()
    }

    pub fn sync_context(&mut self, context: ReferenceContext) {
        if self.context == context {
            return;
        }
        *self = Self::load(context);
    }

    pub fn begin_reference_capture(&mut self, now: Instant) {
        self.active = false;
        self.pending_reference = None;
        self.capture = Some(ReferenceCapture::new(
            self.context.clone(),
            crate::config::default_ml_geometry(&self.context.device_key),
            now,
        ));
        self.guidance = Guidance::CapturingReference {
            progress: 0.0,
            frames: 0,
            step: CaptureStep::Settling,
        };
        self.smoothed = None;
        self.aligned_since = None;
        self.last_valid_estimate = None;
    }

    pub fn begin_assist(&mut self) -> Result<(), String> {
        if self.reference.is_none() {
            return Err("save a known-good wearing position first".into());
        }
        self.capture = None;
        self.pending_reference = None;
        self.active = true;
        self.guidance = Guidance::WaitingForFrames;
        self.last_analysis = None;
        self.last_generation = [0; 2];
        self.smoothed = None;
        self.aligned_since = None;
        self.last_valid_estimate = None;
        Ok(())
    }

    pub fn stop(&mut self) {
        self.capture = None;
        self.pending_reference = None;
        self.active = false;
        self.smoothed = None;
        self.aligned_since = None;
        self.last_valid_estimate = None;
        self.guidance = Guidance::WaitingForFrames;
    }

    pub fn remove_reference(&mut self) -> Result<bool, String> {
        self.stop();
        let removed = Reference::remove(&self.context)?;
        self.reference = None;
        self.load_warning = None;
        Ok(removed)
    }

    pub fn confirm_reference(&mut self) -> Result<PathBuf, String> {
        let reference = self
            .pending_reference
            .take()
            .ok_or_else(|| "there is no captured wearing-position reference to save".to_string())?;
        match reference.save() {
            Ok(path) => {
                self.reference = Some(reference);
                self.guidance = Guidance::ReferenceSaved { path: path.clone() };
                self.load_warning = None;
                Ok(path)
            }
            Err(error) => {
                self.pending_reference = Some(reference);
                Err(error)
            }
        }
    }

    pub fn discard_pending_reference(&mut self) {
        self.pending_reference = None;
        self.guidance = Guidance::WaitingForFrames;
    }

    pub fn update(
        &mut self,
        frames: &[Option<EyeFrame>; 2],
        gaze: [Option<GazePoint>; 2],
        blinking: Option<bool>,
        now: Instant,
    ) {
        if let Some(capture) = &mut self.capture {
            match capture.ingest(frames, gaze, blinking, now) {
                Ok(Some(reference)) => {
                    let weighted_pixels = reference
                        .eyes
                        .each_ref()
                        .map(|eye| eye.weights.iter().filter(|weight| **weight > 0).count());
                    self.capture = None;
                    self.pending_reference = Some(reference);
                    self.guidance = Guidance::ReferenceReady { weighted_pixels };
                }
                Ok(None) => {
                    self.guidance = Guidance::CapturingReference {
                        progress: capture.progress(now),
                        frames: capture.all_frames.iter().map(Vec::len).min().unwrap_or(0),
                        step: capture.step(now),
                    };
                }
                Err(error) => {
                    self.capture = None;
                    self.guidance = Guidance::Error(error);
                }
            }
            return;
        }
        if !self.active {
            return;
        }
        if blinking == Some(true) {
            self.guidance = Guidance::EyesClosed;
            self.aligned_since = None;
            self.reset_stale_estimate(now);
            return;
        }
        if self
            .last_analysis
            .is_some_and(|last| now.duration_since(last) < ANALYZE_INTERVAL)
        {
            return;
        }
        let [Some(left), Some(right)] = frames else {
            self.guidance = Guidance::WaitingForFrames;
            self.aligned_since = None;
            self.reset_stale_estimate(now);
            return;
        };
        let generation = [left.generation, right.generation];
        if generation == self.last_generation {
            return;
        }
        self.last_generation = generation;
        self.last_analysis = Some(now);
        let Some(reference) = &self.reference else {
            self.guidance = Guidance::Error("wearing-position reference is missing".into());
            self.active = false;
            return;
        };
        if reference.source_dimensions() != [[left.width, left.height], [right.width, right.height]]
        {
            self.guidance = Guidance::Error(
                "eye-camera resolution differs from the saved reference; save it again".into(),
            );
            self.aligned_since = None;
            return;
        }
        let complete = [left, right].into_iter().all(|frame| {
            frame.width > 0
                && frame.height > 0
                && frame.pixels.len() >= frame.width as usize * frame.height as usize
        });
        if !complete {
            self.guidance = Guidance::LowConfidence {
                detail: "an eye-camera frame was incomplete; waiting for the next stereo frame"
                    .into(),
            };
            self.aligned_since = None;
            self.reset_stale_estimate(now);
            return;
        }
        let current = [
            downsample(
                left.pixels.as_ref(),
                left.width as usize,
                left.height as usize,
            ),
            downsample(
                right.pixels.as_ref(),
                right.width as usize,
                right.height as usize,
            ),
        ];
        match estimate(reference, &current, self.axes) {
            Ok(next) => {
                let smoothed = smooth_estimate(self.smoothed, next, 0.35);
                self.smoothed = Some(smoothed);
                self.last_valid_estimate = Some(now);
                if smoothed.is_aligned() {
                    let since = self.aligned_since.get_or_insert(now);
                    if now.duration_since(*since) >= ALIGNED_HOLD {
                        self.guidance = Guidance::Aligned(smoothed);
                    } else {
                        self.guidance = Guidance::Adjust(smoothed);
                    }
                } else {
                    self.aligned_since = None;
                    self.guidance = Guidance::Adjust(smoothed);
                }
            }
            Err(detail) => {
                self.guidance = Guidance::LowConfidence { detail };
                self.aligned_since = None;
                self.reset_stale_estimate(now);
            }
        }
    }

    fn reset_stale_estimate(&mut self, now: Instant) {
        if self
            .last_valid_estimate
            .is_some_and(|last| now.saturating_duration_since(last) > Duration::from_millis(700))
        {
            self.smoothed = None;
            self.last_valid_estimate = None;
        }
    }
}

fn estimate(
    reference: &Reference,
    current: &[Vec<u8>; 2],
    axes: [EyeAxes; 2],
) -> Result<Estimate, String> {
    let matches = [
        register_eye(&reference.eyes[0], &current[0])?,
        register_eye(&reference.eyes[1], &current[1])?,
    ];
    // Convert both raw camera coordinate systems into the same anatomical image
    // plane before stereo agreement. XR5's right ML channel is mirrored; omitting
    // that handedness makes real horizontal displacement and roll cancel between
    // the eyes and can produce an authoritative-looking but wrong arrow.
    let canonical: [([f32; 2], f32); 2] =
        std::array::from_fn(|eye| canonicalize_match(matches[eye], axes[eye]));
    let face_shift = [canonical[0].0, canonical[1].0];
    let face_rotation = [canonical[0].1, canonical[1].1];
    let disagreement = ((face_shift[0][0] - face_shift[1][0]).powi(2)
        + (face_shift[0][1] - face_shift[1][1]).powi(2))
    .sqrt();
    if disagreement > MAX_STEREO_SHIFT_DISAGREEMENT
        || (matches[0].scale - matches[1].scale).abs() > MAX_STEREO_SCALE_DISAGREEMENT
        || (face_rotation[0] - face_rotation[1]).abs() > MAX_STEREO_ROTATION_DISAGREEMENT_DEG
    {
        return Err(
            "left/right structure disagrees; look at the centre target with both eyes comfortably open"
                .into(),
        );
    }
    let confidence = eye_confidence(matches[0]).min(eye_confidence(matches[1]))
        * (1.0 - disagreement / MAX_STEREO_SHIFT_DISAGREEMENT).clamp(0.25, 1.0);
    if confidence < 0.35 {
        return Err(
            "the saved skin/lid structure could not be matched confidently; hold still and look straight ahead"
                .into(),
        );
    }
    Ok(Estimate {
        correction_px: [
            (face_shift[0][0] + face_shift[1][0]) * 0.5,
            (face_shift[0][1] + face_shift[1][1]) * 0.5,
        ],
        scale: (matches[0].scale + matches[1].scale) * 0.5,
        rotation_deg: (face_rotation[0] + face_rotation[1]) * 0.5,
        confidence,
        eyes: matches,
    })
}

fn register_eye(reference: &EyeTemplate, current: &[u8]) -> Result<EyeMatch, String> {
    if current.len() != TEMPLATE_PIXELS {
        return Err("current eye descriptor has the wrong dimensions".into());
    }
    let current_census = census(current);
    let mut best = Candidate::default();
    let mut candidates = Vec::with_capacity(400);
    // Staged coordinate descent keeps the dashboard thread light: translation
    // first, then roll, depth, and one joint local refinement. The previous full
    // Cartesian grid evaluated more than four times as many transforms.
    for dy in (-6..=6).step_by(2) {
        for dx in (-6..=6).step_by(2) {
            consider_candidate(
                &mut best,
                &mut candidates,
                score_transform(reference, &current_census, dx as f32, dy as f32, 1.0, 0.0),
            );
        }
    }
    let translated = best;
    for &rotation in &[-6.0f32, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0] {
        for dy_delta in -1..=1 {
            for dx_delta in -1..=1 {
                consider_candidate(
                    &mut best,
                    &mut candidates,
                    score_transform(
                        reference,
                        &current_census,
                        translated.dx + dx_delta as f32,
                        translated.dy + dy_delta as f32,
                        1.0,
                        rotation,
                    ),
                );
            }
        }
    }
    let rotated = best;
    for &scale in &[0.88f32, 0.94, 1.0, 1.06, 1.12] {
        for dy_delta in -1..=1 {
            for dx_delta in -1..=1 {
                consider_candidate(
                    &mut best,
                    &mut candidates,
                    score_transform(
                        reference,
                        &current_census,
                        rotated.dx + dx_delta as f32,
                        rotated.dy + dy_delta as f32,
                        scale,
                        rotated.rotation,
                    ),
                );
            }
        }
    }
    let coarse = best;
    for scale_delta in [-0.025f32, 0.0, 0.025] {
        for rotation_delta in [-1.0f32, 0.0, 1.0] {
            for dy_delta in -1..=1 {
                for dx_delta in -1..=1 {
                    consider_candidate(
                        &mut best,
                        &mut candidates,
                        score_transform(
                            reference,
                            &current_census,
                            coarse.dx + dx_delta as f32,
                            coarse.dy + dy_delta as f32,
                            (coarse.scale + scale_delta).clamp(0.82, 1.18),
                            coarse.rotation + rotation_delta,
                        ),
                    );
                }
            }
        }
    }
    // Recompute ambiguity against the final transform. Carrying a `second`
    // candidate through coordinate-descent stages can leave a near-duplicate of
    // the final answer or a coarse predecessor, neither of which represents a
    // genuinely competing registration mode.
    let margin = ambiguity_margin(best, &candidates);
    if !best.score.is_finite() || best.score < MIN_EYE_SCORE || margin < MIN_EYE_MARGIN {
        return Err(format!(
            "eye structure match was ambiguous (score {:.2}, margin {:.3})",
            best.score, margin
        ));
    }
    if best.dx.abs() >= 6.9
        || best.dy.abs() >= 6.9
        || best.rotation.abs() >= 5.9
        || best.scale <= 0.856
        || best.scale >= 1.144
    {
        return Err(
            "the wearing-position difference is outside the safe search range; adjust the HMD roughly toward the saved fit, then continue"
                .into(),
        );
    }
    Ok(EyeMatch {
        shift_px: [best.dx, best.dy],
        scale: best.scale,
        rotation_deg: best.rotation,
        score: best.score,
        margin,
    })
}

fn ambiguity_margin(best: Candidate, candidates: &[Candidate]) -> f32 {
    let second_score = candidates
        .iter()
        .copied()
        .filter(|candidate| distinct_transform(best, *candidate))
        .map(|candidate| candidate.score)
        .fold(f32::NEG_INFINITY, f32::max);
    if second_score.is_finite() {
        (best.score - second_score).max(0.0)
    } else {
        1.0
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    dx: f32,
    dy: f32,
    scale: f32,
    rotation: f32,
    score: f32,
}

impl Default for Candidate {
    fn default() -> Self {
        Self {
            dx: 0.0,
            dy: 0.0,
            scale: 1.0,
            rotation: 0.0,
            score: f32::NEG_INFINITY,
        }
    }
}

fn consider_candidate(best: &mut Candidate, candidates: &mut Vec<Candidate>, candidate: Candidate) {
    if !candidate.score.is_finite() {
        return;
    }
    if candidate.score > best.score {
        *best = candidate;
    }
    candidates.push(candidate);
}

fn distinct_transform(a: Candidate, b: Candidate) -> bool {
    (a.dx - b.dx).abs() >= 2.0
        || (a.dy - b.dy).abs() >= 2.0
        || (a.scale - b.scale).abs() >= 0.04
        || (a.rotation - b.rotation).abs() >= 3.0
}

fn score_transform(
    reference: &EyeTemplate,
    current_census: &[u8],
    dx: f32,
    dy: f32,
    scale: f32,
    rotation_deg: f32,
) -> Candidate {
    if dx.abs() > SEARCH_SHIFT_LIMIT
        || dy.abs() > SEARCH_SHIFT_LIMIT
        || rotation_deg.abs() > SEARCH_ROTATION_LIMIT
        || !(SEARCH_SCALE_MIN..=SEARCH_SCALE_MAX).contains(&scale)
    {
        return Candidate {
            dx,
            dy,
            scale,
            rotation: rotation_deg,
            score: f32::NEG_INFINITY,
        };
    }
    let centre = (TEMPLATE_SIDE as f32 - 1.0) * 0.5;
    let (sin, cos) = rotation_deg.to_radians().sin_cos();
    let mut distance = 0.0f32;
    let mut weight_sum = 0.0f32;
    let mut covered_weight = 0.0f32;
    for &index in &reference.feature_indices {
        let x = index % TEMPLATE_SIDE;
        let y = index / TEMPLATE_SIDE;
        let weight = reference.weights[index] as f32 / 255.0;
        let rx = x as f32 - centre;
        let ry = y as f32 - centre;
        let cx = centre + scale * (rx * cos - ry * sin) + dx;
        let cy = centre + scale * (rx * sin + ry * cos) + dy;
        let ix = cx.round() as isize;
        let iy = cy.round() as isize;
        let weighted_bits = 8.0 * weight;
        weight_sum += weighted_bits;
        if ix < 2 || iy < 2 || ix >= TEMPLATE_SIDE as isize - 2 || iy >= TEMPLATE_SIDE as isize - 2
        {
            // Missing reference structure is a maximal mismatch. Omitting it
            // would make large shifts/scales look artificially good by scoring
            // only the easiest surviving subset.
            distance += weighted_bits;
            continue;
        }
        let current = current_census[iy as usize * TEMPLATE_SIDE + ix as usize];
        distance += (reference.census[index] ^ current).count_ones() as f32 * weight;
        covered_weight += weighted_bits;
    }
    let coverage = covered_weight / weight_sum.max(1.0);
    Candidate {
        dx,
        dy,
        scale,
        rotation: rotation_deg,
        score: if weight_sum > 1.0 && coverage >= MIN_FEATURE_COVERAGE {
            1.0 - distance / weight_sum
        } else {
            f32::NEG_INFINITY
        },
    }
}

fn eye_confidence(value: EyeMatch) -> f32 {
    let score = ((value.score - MIN_EYE_SCORE) / (0.88 - MIN_EYE_SCORE)).clamp(0.0, 1.0);
    let margin = ((value.margin - MIN_EYE_MARGIN) / 0.055).clamp(0.0, 1.0);
    (0.65 * score + 0.35 * margin).clamp(0.0, 1.0)
}

fn smooth_estimate(previous: Option<Estimate>, next: Estimate, alpha: f32) -> Estimate {
    let Some(previous) = previous else {
        return next;
    };
    let mix = |a: f32, b: f32| a + (b - a) * alpha;
    Estimate {
        correction_px: [
            mix(previous.correction_px[0], next.correction_px[0]),
            mix(previous.correction_px[1], next.correction_px[1]),
        ],
        scale: mix(previous.scale, next.scale),
        rotation_deg: mix(previous.rotation_deg, next.rotation_deg),
        confidence: mix(previous.confidence, next.confidence),
        eyes: next.eyes,
    }
}

fn rotate_shift(value: [f32; 2], degrees: f32) -> [f32; 2] {
    let (sin, cos) = degrees.to_radians().sin_cos();
    [
        value[0] * cos - value[1] * sin,
        value[0] * sin + value[1] * cos,
    ]
}

fn canonicalize_match(value: EyeMatch, axes: EyeAxes) -> ([f32; 2], f32) {
    let mut shift = rotate_shift(value.shift_px, axes.upright_deg);
    // `warp_into` applies geometry rotation and then mirrors the destination ML
    // channel, so the same order is required for a raw-frame displacement.
    shift[0] *= axes.handedness;
    shift[1] /= axes.scale_y;
    (shift, value.rotation_deg * axes.handedness)
}

fn validate_gaze_sweep(
    centre: &[Vec<GazePoint>; 2],
    sweep: &[Vec<GazePoint>; 2],
) -> Result<(), String> {
    let mut measured = false;
    let mut widest_yaw = 0.0f32;
    let mut widest_pitch = 0.0f32;
    for eye in 0..2 {
        let directions = centre[eye]
            .iter()
            .chain(&sweep[eye])
            .filter_map(|sample| match sample {
                GazePoint::Direction(value) => {
                    let radius = value[0].hypot(value[2]);
                    (radius > 1.0e-6).then_some([
                        value[0].atan2(value[2].abs()).to_degrees(),
                        value[1].atan2(radius).to_degrees(),
                    ])
                }
            })
            .collect::<Vec<_>>();
        if directions.len() >= 5 {
            measured = true;
            widest_yaw = widest_yaw.max(range_f32(directions.iter().map(|angles| angles[0])));
            widest_pitch = widest_pitch.max(range_f32(directions.iter().map(|angles| angles[1])));
        }
    }
    // Devices without usable native gaze remain supported; their image variability
    // mask and stereo abstention are the fallback safeguards.
    if measured && (widest_yaw < 8.0 || widest_pitch < 6.0) {
        return Err(format!(
            "the eye-only look-around step was too small ({widest_yaw:.1} deg horizontal, {widest_pitch:.1} deg vertical); keep the HMD still and look farther left, right, up and down"
        ));
    }
    Ok(())
}

fn range_f32(values: impl Iterator<Item = f32>) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values.filter(|value| value.is_finite()) {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        max - min
    } else {
        0.0
    }
}

fn build_template(
    source_width: u32,
    source_height: u32,
    centre_frames: &[Vec<u8>],
    all_frames: &[Vec<u8>],
    device_key: &str,
    eye: usize,
    geometry: MlGeometry,
) -> Result<EyeTemplate, String> {
    let mut median = vec![0u8; TEMPLATE_PIXELS];
    let mut variability = vec![0u8; TEMPLATE_PIXELS];
    let mut glare = vec![false; TEMPLATE_PIXELS];
    for pixel in 0..TEMPLATE_PIXELS {
        let mut values = centre_frames
            .iter()
            .map(|frame| frame[pixel])
            .collect::<Vec<_>>();
        values.sort_unstable();
        median[pixel] = values[values.len() / 2];
        let mut all_values = all_frames
            .iter()
            .map(|frame| frame[pixel])
            .collect::<Vec<_>>();
        all_values.sort_unstable();
        let p10 = all_values[all_values.len() / 10];
        let p90 = all_values[all_values.len() * 9 / 10];
        variability[pixel] = p90.saturating_sub(p10);
        glare[pixel] = p90 >= 238;
    }
    // Saturated glints and emitter blooms have useful-looking edges but are
    // camera/illumination structure, not a wearing-position landmark. Dilate their
    // mask so the bright-to-dark halo cannot dominate census registration.
    let mut glare_dilated = vec![false; TEMPLATE_PIXELS];
    let mut nuisance_dilated = vec![false; TEMPLATE_PIXELS];
    for y in 0..TEMPLATE_SIDE {
        for x in 0..TEMPLATE_SIDE {
            let y0 = y.saturating_sub(2);
            let y1 = (y + 2).min(TEMPLATE_SIDE - 1);
            let x0 = x.saturating_sub(2);
            let x1 = (x + 2).min(TEMPLATE_SIDE - 1);
            glare_dilated[y * TEMPLATE_SIDE + x] =
                (y0..=y1).any(|yy| (x0..=x1).any(|xx| glare[yy * TEMPLATE_SIDE + xx]));
            nuisance_dilated[y * TEMPLATE_SIDE + x] =
                (y0..=y1).any(|yy| (x0..=x1).any(|xx| variability[yy * TEMPLATE_SIDE + xx] > 18));
        }
    }
    let mut weights = vec![0u8; TEMPLATE_PIXELS];
    let mut weighted = 0usize;
    for y in 3..TEMPLATE_SIDE - 3 {
        for x in 3..TEMPLATE_SIDE - 3 {
            let index = y * TEMPLATE_SIDE + x;
            let value = median[index];
            if !xr5_safe_pixel(device_key, eye, geometry, x, y)
                || glare_dilated[index]
                || nuisance_dilated[index]
                || !(32..=224).contains(&value)
                || (value > 200 && variability[index] < 4)
            {
                continue;
            }
            let gx = median[index + 1].abs_diff(median[index - 1]) as u16;
            let gy = median[index + TEMPLATE_SIDE].abs_diff(median[index - TEMPLATE_SIDE]) as u16;
            let edge = gx + gy;
            if edge < 10 {
                continue;
            }
            let stable = 255u16.saturating_sub(variability[index] as u16 * 8);
            let strength = (edge * 4).min(255);
            weights[index] = stable.min(strength) as u8;
            weighted += 1;
        }
    }
    if weighted < MIN_WEIGHTED_PIXELS {
        return Err(format!(
            "only {weighted} stable skin/lid feature pixels were found (need {MIN_WEIGHTED_PIXELS}); keep both eyes open and hold the HMD still"
        ));
    }
    let census = census(&median);
    let feature_indices = weights
        .iter()
        .enumerate()
        .filter_map(|(index, weight)| (*weight > 0).then_some(index))
        .collect();
    Ok(EyeTemplate {
        source_width,
        source_height,
        median,
        weights,
        census,
        feature_indices,
    })
}

fn xr5_safe_pixel(
    device_key: &str,
    _eye: usize,
    geometry: MlGeometry,
    template_x: usize,
    template_y: usize,
) -> bool {
    if canonical_device_key(device_key) != "pimax_xr5" {
        return true;
    }
    let normalized_x = (template_x as f32 + 0.5) / TEMPLATE_SIDE as f32;
    let normalized_y = (template_y as f32 + 0.5) / TEMPLATE_SIDE as f32;
    if !(34.0 / 200.0..=166.0 / 200.0).contains(&normalized_y) {
        return false;
    }
    // The inner LED/lens stack is fixed to the camera and must never vote as
    // "stable face structure". The geometry passed here is the frozen shipped
    // optical preset, not a live user-editable ML geometry.
    if geometry.crop_right >= geometry.crop_left {
        normalized_x <= 108.0 / 200.0
    } else {
        normalized_x >= 91.0 / 200.0
    }
}

fn downsample(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; TEMPLATE_PIXELS];
    if width == 0 || height == 0 {
        return output;
    }
    for y in 0..TEMPLATE_SIDE {
        let y0 = y * height / TEMPLATE_SIDE;
        let y1 = (((y + 1) * height).div_ceil(TEMPLATE_SIDE))
            .min(height)
            .max(y0 + 1);
        for x in 0..TEMPLATE_SIDE {
            let x0 = x * width / TEMPLATE_SIDE;
            let x1 = (((x + 1) * width).div_ceil(TEMPLATE_SIDE))
                .min(width)
                .max(x0 + 1);
            let mut sum = 0u64;
            let mut count = 0u64;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += pixels[yy * width + xx] as u64;
                    count += 1;
                }
            }
            output[y * TEMPLATE_SIDE + x] = (sum / count.max(1)) as u8;
        }
    }
    output
}

fn census(image: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; TEMPLATE_PIXELS];
    let neighbours = [
        (-1isize, -1isize),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    for y in 1..TEMPLATE_SIDE - 1 {
        for x in 1..TEMPLATE_SIDE - 1 {
            let centre = image[y * TEMPLATE_SIDE + x];
            let mut code = 0u8;
            for (bit, (dx, dy)) in neighbours.iter().enumerate() {
                let xx = (x as isize + dx) as usize;
                let yy = (y as isize + dy) as usize;
                if image[yy * TEMPLATE_SIDE + xx] > centre.saturating_add(2) {
                    code |= 1 << bit;
                }
            }
            output[y * TEMPLATE_SIDE + x] = code;
        }
    }
    output
}

fn reference_path(context: &ReferenceContext) -> PathBuf {
    crate::config::base_dir()
        .join("reseat-references")
        .join(format!(
            "{}_{}.bin",
            safe_component(&context.device_key),
            safe_component(&context.unit_id)
        ))
}

fn safe_component(value: &str) -> String {
    let safe = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    if safe.is_empty() {
        "unknown".into()
    } else {
        safe
    }
}

fn save_to(path: &Path, reference: &Reference) -> Result<(), String> {
    let mut bytes = Vec::with_capacity(2 * TEMPLATE_PIXELS * 2 + 256);
    bytes.extend_from_slice(MAGIC);
    push_u16(&mut bytes, SCHEMA_VERSION);
    push_u16(&mut bytes, TEMPLATE_SIDE as u16);
    push_string(&mut bytes, &reference.context.device_key)?;
    push_string(&mut bytes, &reference.context.unit_id)?;
    push_u64(&mut bytes, reference.context.image_fingerprint);
    push_u64(&mut bytes, reference.captured_unix);
    for eye in 0..2 {
        push_u32(&mut bytes, reference.eyes[eye].source_width);
        push_u32(&mut bytes, reference.eyes[eye].source_height);
        bytes.extend_from_slice(&reference.eyes[eye].median);
        bytes.extend_from_slice(&reference.eyes[eye].weights);
    }
    let crc = crate::diagnostics::crc32_fingerprint(&bytes);
    push_u32(&mut bytes, crc);
    let parent = path
        .parent()
        .ok_or_else(|| "wearing-position reference path has no parent".to_string())?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    let partial = path.with_extension("partial");
    let backup = path.with_extension("bak");
    let mut file = fs::File::create(&partial)
        .map_err(|error| format!("could not create {}: {error}", partial.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("could not write {}: {error}", partial.display()))?;
    file.sync_all()
        .map_err(|error| format!("could not flush {}: {error}", partial.display()))?;
    drop(file);
    if path.exists() {
        let _ = fs::remove_file(&backup);
        fs::rename(path, &backup)
            .map_err(|error| format!("could not stage old reference: {error}"))?;
    }
    if let Err(error) = fs::rename(&partial, path) {
        if backup.exists() {
            let _ = fs::rename(&backup, path);
        }
        return Err(format!("could not publish {}: {error}", path.display()));
    }
    let _ = fs::remove_file(backup);
    Ok(())
}

/// Load the published reference, falling back to the previous confirmed file if
/// the process or machine stopped between `path -> .bak` and `.partial -> path`.
/// A missing primary is restored best-effort; even if the rename is denied, the
/// validated backup remains usable in memory and is never silently discarded.
fn load_with_backup(path: &Path) -> Result<Option<Reference>, String> {
    let backup = path.with_extension("bak");
    if path.is_file() {
        return match load_from(path) {
            Ok(reference) => Ok(Some(reference)),
            Err(primary_error) if backup.is_file() => load_from(&backup)
                .map(Some)
                .map_err(|backup_error| {
                    format!(
                        "published wearing-position reference is invalid ({primary_error}); backup is also invalid ({backup_error})"
                    )
                }),
            Err(error) => Err(error),
        };
    }
    if !backup.is_file() {
        return Ok(None);
    }
    let reference = load_from(&backup)?;
    let _ = fs::rename(&backup, path);
    Ok(Some(reference))
}

fn load_from(path: &Path) -> Result<Reference, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("could not read {}: {error}", path.display()))?;
    if bytes.len() < MAGIC.len() + 4 + 4 {
        return Err("wearing-position reference is truncated".into());
    }
    let payload_len = bytes.len() - 4;
    let expected_crc = u32::from_le_bytes(
        bytes[payload_len..]
            .try_into()
            .map_err(|_| "reference CRC is missing")?,
    );
    if crate::diagnostics::crc32_fingerprint(&bytes[..payload_len]) != expected_crc {
        return Err("wearing-position reference failed its integrity check".into());
    }
    let mut reader = Reader::new(&bytes[..payload_len]);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err("not an SRanibro wearing-position reference".into());
    }
    let version = reader.u16()?;
    if version != SCHEMA_VERSION {
        return Err(format!(
            "wearing-position reference schema {version} is not supported"
        ));
    }
    if reader.u16()? as usize != TEMPLATE_SIDE {
        return Err("wearing-position reference uses another template size".into());
    }
    let context = ReferenceContext {
        device_key: reader.string()?,
        unit_id: reader.string()?,
        image_fingerprint: reader.u64()?,
    };
    let captured_unix = reader.u64()?;
    let mut eyes = Vec::with_capacity(2);
    for _eye in 0..2 {
        let source_width = reader.u32()?;
        let source_height = reader.u32()?;
        let median = reader.take(TEMPLATE_PIXELS)?.to_vec();
        let weights = reader.take(TEMPLATE_PIXELS)?.to_vec();
        if weights.iter().filter(|weight| **weight > 0).count() < MIN_WEIGHTED_PIXELS {
            return Err("wearing-position reference has too little stable structure".into());
        }
        let census = census(&median);
        let feature_indices = weights
            .iter()
            .enumerate()
            .filter_map(|(index, weight)| (*weight > 0).then_some(index))
            .collect();
        eyes.push(EyeTemplate {
            source_width,
            source_height,
            median,
            weights,
            census,
            feature_indices,
        });
    }
    if !reader.remaining().is_empty() {
        return Err("wearing-position reference has unexpected trailing data".into());
    }
    Ok(Reference {
        context,
        captured_unix,
        eyes: [eyes.remove(0), eyes.remove(0)],
    })
}

fn push_string(output: &mut Vec<u8>, value: &str) -> Result<(), String> {
    let bytes = value.as_bytes();
    let len = u16::try_from(bytes.len()).map_err(|_| "reference text is too long")?;
    push_u16(output, len);
    output.extend_from_slice(bytes);
    Ok(())
}

fn push_u16(output: &mut Vec<u8>, value: u16) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u32(output: &mut Vec<u8>, value: u32) {
    output.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(output: &mut Vec<u8>, value: u64) {
    output.extend_from_slice(&value.to_le_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .filter(|end| *end <= self.bytes.len())
            .ok_or_else(|| "wearing-position reference is truncated".to_string())?;
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().map_err(|_| "missing u16")?,
        ))
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| "missing u32")?,
        ))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| "missing u64")?,
        ))
    }

    fn string(&mut self) -> Result<String, String> {
        let len = self.u16()? as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| "wearing-position reference text is not UTF-8".into())
    }

    fn remaining(&self) -> &[u8] {
        &self.bytes[self.offset..]
    }
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn byte(&mut self, byte: u8) {
        self.0 ^= byte as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.byte(*byte);
        }
    }

    fn finish(self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(
        shift: [f32; 2],
        scale: f32,
        rotation_deg: f32,
        iris_shift: [f32; 2],
        brightness: f32,
        offset: f32,
    ) -> Vec<u8> {
        let centre = (TEMPLATE_SIDE as f32 - 1.0) * 0.5;
        let (sin, cos) = rotation_deg.to_radians().sin_cos();
        let mut image = vec![0u8; TEMPLATE_PIXELS];
        for y in 0..TEMPLATE_SIDE {
            for x in 0..TEMPLATE_SIDE {
                let ox = (x as f32 - centre - shift[0]) / scale;
                let oy = (y as f32 - centre - shift[1]) / scale;
                let sx = ox * cos + oy * sin + centre;
                let sy = -ox * sin + oy * cos + centre;
                let ring = (((sx - 21.0 - iris_shift[0]).powi(2)
                    + (sy - 25.0 - iris_shift[1]).powi(2))
                .sqrt()
                    - 8.0)
                    .abs()
                    < 1.8;
                let lid = (sy - (15.0 + 0.08 * (sx - 24.0).powi(2))).abs() < 1.5;
                let skin = 82.0
                    + sx * 1.4
                    + sy * 0.7
                    + (sx * 0.72).sin() * 17.0
                    + (sy * 0.57).cos() * 14.0;
                let value = if ring {
                    35.0
                } else if lid {
                    205.0
                } else {
                    skin
                };
                image[y * TEMPLATE_SIDE + x] =
                    (value * brightness + offset).clamp(0.0, 255.0) as u8;
            }
        }
        image
    }

    fn pattern(shift: [f32; 2], scale: f32, brightness: f32, offset: f32) -> Vec<u8> {
        scene(shift, scale, 0.0, [0.0, 0.0], brightness, offset)
    }

    fn template() -> EyeTemplate {
        let frames = (0..24)
            .map(|index| pattern([0.0, 0.0], 1.0, 1.0, (index % 3) as f32))
            .collect::<Vec<_>>();
        build_template(
            200,
            200,
            &frames,
            &frames,
            "pimax_vr4",
            0,
            MlGeometry::default(),
        )
        .unwrap()
    }

    #[test]
    fn census_registration_recovers_shift_and_ignores_affine_brightness() {
        let reference = template();
        let current = pattern([4.0, -2.0], 1.0, 0.72, 31.0);
        let result = register_eye(&reference, &current).unwrap();
        assert!((result.shift_px[0] - 4.0).abs() <= 1.0, "{result:?}");
        assert!((result.shift_px[1] + 2.0).abs() <= 1.0, "{result:?}");
        assert!(result.score > MIN_EYE_SCORE, "{result:?}");
    }

    #[test]
    fn census_registration_recovers_scale() {
        let reference = template();
        let current = pattern([0.0, 0.0], 1.06, 1.0, 0.0);
        let result = register_eye(&reference, &current).unwrap();
        assert!((result.scale - 1.06).abs() <= 0.03, "{result:?}");
    }

    #[test]
    fn out_of_bounds_features_are_penalized_instead_of_dropped() {
        let reference = template();
        let current = pattern([0.0, 0.0], 1.0, 1.0, 0.0);
        let current_census = census(&current);
        let centred = score_transform(&reference, &current_census, 0.0, 0.0, 1.0, 0.0);
        let edge = score_transform(&reference, &current_census, 7.0, 7.0, 1.0, 0.0);
        assert!(centred.score > edge.score, "{centred:?} versus {edge:?}");
    }

    #[test]
    fn ambiguity_is_recomputed_against_the_final_transform() {
        let best = Candidate {
            dx: 3.0,
            dy: 0.0,
            scale: 1.0,
            rotation: 0.0,
            score: 0.90,
        };
        let candidates = [
            // This coarse predecessor is close to the final transform and must
            // not masquerade as an alternative registration mode.
            Candidate {
                dx: 2.0,
                score: 0.895,
                ..best
            },
            Candidate {
                dx: -3.0,
                score: 0.82,
                ..best
            },
            best,
        ];
        assert!((ambiguity_margin(best, &candidates) - 0.08).abs() < 1.0e-6);
    }

    #[test]
    fn registration_recovers_a_joint_similarity_transform() {
        let reference = template();
        let current = scene([3.0, -2.0], 1.06, 3.0, [0.0, 0.0], 0.82, 21.0);
        let result = register_eye(&reference, &current).unwrap();
        assert!((result.shift_px[0] - 3.0).abs() <= 1.0, "{result:?}");
        assert!((result.shift_px[1] + 2.0).abs() <= 1.0, "{result:?}");
        assert!((result.scale - 1.06).abs() <= 0.04, "{result:?}");
        assert!((result.rotation_deg - 3.0).abs() <= 1.5, "{result:?}");
    }

    #[test]
    fn gaze_only_iris_motion_is_removed_from_the_reference_weights() {
        let centre = (0..30)
            .map(|index| pattern([0.0, 0.0], 1.0, 1.0, (index % 3) as f32))
            .collect::<Vec<_>>();
        let sweep = [[-6.0, 0.0], [6.0, 0.0], [0.0, -5.0], [0.0, 5.0]];
        let all = (0..80)
            .map(|index| {
                scene(
                    [0.0, 0.0],
                    1.0,
                    0.0,
                    sweep[index % sweep.len()],
                    1.0,
                    (index % 3) as f32,
                )
            })
            .collect::<Vec<_>>();
        let reference = build_template(
            200,
            200,
            &centre,
            &all,
            "pimax_vr4",
            0,
            MlGeometry::default(),
        )
        .unwrap();
        let current = scene([0.0, 0.0], 1.0, 0.0, [5.0, -3.0], 0.75, 25.0);
        let result = register_eye(&reference, &current).unwrap();
        assert!(result.shift_px[0].abs() <= 1.0, "{result:?}");
        assert!(result.shift_px[1].abs() <= 1.0, "{result:?}");
    }

    #[test]
    fn image_context_changes_with_mapping_or_geometry() {
        let mapping = EyeMapping::default();
        let geometry = [MlGeometry::default(); 2];
        let base = image_fingerprint(mapping, geometry);
        let mut changed_mapping = mapping;
        changed_mapping.swap_eyes = true;
        assert_ne!(base, image_fingerprint(changed_mapping, geometry));
        let mut changed_geometry = geometry;
        changed_geometry[0].rotate_deg = 1.0;
        assert_eq!(
            base,
            image_fingerprint(mapping, changed_geometry),
            "ML-only geometry does not alter the raw camera reference"
        );
        let mut changed_mapping = mapping;
        changed_mapping.flip_image = true;
        assert_ne!(base, image_fingerprint(changed_mapping, geometry));
    }

    #[test]
    fn capture_guide_visits_all_four_eye_sweep_targets() {
        let progress = |elapsed: f32| elapsed / CAPTURE_SECONDS;
        assert_eq!(
            CaptureStep::EyeSweep.target(progress(2.2)),
            Some(GazeTarget::Left)
        );
        assert_eq!(
            CaptureStep::EyeSweep.target(progress(3.0)),
            Some(GazeTarget::Right)
        );
        assert_eq!(
            CaptureStep::EyeSweep.target(progress(3.7)),
            Some(GazeTarget::Up)
        );
        assert_eq!(
            CaptureStep::EyeSweep.target(progress(4.5)),
            Some(GazeTarget::Down)
        );
        assert_eq!(
            CaptureStep::CentreAfter.target(1.0),
            Some(GazeTarget::Center)
        );
    }

    #[test]
    fn capture_rate_does_not_alias_a_sixty_hz_ui_to_twenty_hz() {
        assert!(CAPTURE_INTERVAL <= Duration::from_millis(30));
        assert!(CAPTURE_SECONDS * (1000.0 / CAPTURE_INTERVAL.as_millis() as f32) > 200.0);
    }

    #[test]
    fn frame_quota_gets_a_bounded_straight_ahead_grace_period() {
        let started = Instant::now();
        let mut capture = ReferenceCapture::new(
            ReferenceContext::new("pimax_vr4", "grace-test".into(), 1),
            [MlGeometry::default(); 2],
            started,
        );
        let nominal_end = started + Duration::from_secs_f32(CAPTURE_SECONDS + 0.1);
        assert!(capture.finish_if_due(nominal_end).unwrap().is_none());
        let hard_end =
            started + Duration::from_secs_f32(CAPTURE_SECONDS + CAPTURE_GRACE_SECONDS + 0.1);
        assert!(capture
            .finish_if_due(hard_end)
            .unwrap_err()
            .contains("fresh stereo frames"));
    }

    #[test]
    fn mirrored_right_eye_canonicalizes_shift_and_roll_before_stereo_average() {
        let left = EyeMatch {
            shift_px: [3.0, -1.0],
            rotation_deg: 2.0,
            ..EyeMatch::default()
        };
        // A horizontally mirrored camera channel observes opposite X and roll.
        let right = EyeMatch {
            shift_px: [-3.0, -1.0],
            rotation_deg: -2.0,
            ..EyeMatch::default()
        };
        let normal = EyeAxes {
            upright_deg: 0.0,
            handedness: 1.0,
            scale_y: 1.0,
        };
        let mirrored = EyeAxes {
            handedness: -1.0,
            ..normal
        };
        let (left_shift, left_roll) = canonicalize_match(left, normal);
        let (right_shift, right_roll) = canonicalize_match(right, mirrored);
        assert_eq!(left_shift, right_shift);
        assert_eq!(left_roll, right_roll);
    }

    #[test]
    fn device_axes_are_frozen_from_the_shipped_optical_preset() {
        let xr5 = device_axes("pimax_xr5");
        assert_eq!(xr5[0].upright_deg, -30.0);
        assert_eq!(xr5[0].handedness, 1.0);
        assert_eq!(xr5[0].scale_y, 1.20);
        assert_eq!(xr5[1].upright_deg, 30.0);
        assert_eq!(xr5[1].handedness, -1.0);
        assert_eq!(xr5[1].scale_y, 1.20);
        assert_eq!(
            device_axes("pimax_vr4"),
            [EyeAxes {
                upright_deg: 0.0,
                handedness: 1.0,
                scale_y: 1.0,
            }; 2]
        );
    }

    #[test]
    fn xr5_inner_led_region_is_excluded_for_both_eye_geometries() {
        let left = MlGeometry {
            crop_right: 0.40,
            ..MlGeometry::default()
        };
        let right = MlGeometry {
            crop_left: 0.40,
            ..MlGeometry::default()
        };
        assert!(xr5_safe_pixel("pimax_xr5", 0, left, 2, 24));
        assert!(!xr5_safe_pixel("pimax_xr5", 0, left, 46, 24));
        assert!(!xr5_safe_pixel("pimax_xr5", 1, right, 1, 24));
        assert!(xr5_safe_pixel("pimax_xr5", 1, right, 45, 24));
        assert!(!xr5_safe_pixel("pimax_xr5", 0, left, 10, 2));
        assert!(xr5_safe_pixel("pimax_vr4", 0, left, 46, 2));
    }

    #[test]
    fn native_gaze_sweep_rejects_a_motionless_reference_step() {
        let centre = std::array::from_fn(|_| vec![GazePoint::Direction([0.0, 0.0, -1.0]); 8]);
        let still = std::array::from_fn(|_| vec![GazePoint::Direction([0.01, 0.0, -1.0]); 8]);
        assert!(validate_gaze_sweep(&centre, &still).is_err());
        let moved = std::array::from_fn(|_| {
            vec![
                GazePoint::Direction([-0.18, 0.0, 1.0]),
                GazePoint::Direction([0.18, 0.0, 1.0]),
                GazePoint::Direction([0.0, -0.14, 1.0]),
                GazePoint::Direction([0.0, 0.14, 1.0]),
                GazePoint::Direction([-0.18, 0.0, 1.0]),
                GazePoint::Direction([0.18, 0.0, 1.0]),
                GazePoint::Direction([0.0, -0.14, 1.0]),
                GazePoint::Direction([0.0, 0.14, 1.0]),
            ]
        });
        assert!(validate_gaze_sweep(&centre, &moved).is_ok());
        let unavailable: [Vec<GazePoint>; 2] = std::array::from_fn(|_| Vec::new());
        assert!(validate_gaze_sweep(&unavailable, &unavailable).is_ok());
    }

    #[test]
    fn live_assist_rejects_an_incomplete_frame_without_panicking() {
        let context = ReferenceContext::new(
            "pimax_vr4",
            format!(
                "incomplete-frame-test-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            42,
        );
        let eye = template();
        let mut assist = Assist::load(context.clone());
        assist.reference = Some(Reference {
            context,
            captured_unix: 123,
            eyes: [eye.clone(), eye],
        });
        assist.active = true;
        let bad = EyeFrame {
            generation: 1,
            width: 200,
            height: 200,
            pixels: std::sync::Arc::from(vec![0u8; 8]),
        };
        assist.update(
            &[Some(bad.clone()), Some(bad)],
            [None, None],
            Some(false),
            Instant::now(),
        );
        assert!(matches!(
            assist.guidance(),
            Guidance::LowConfidence { detail } if detail.contains("incomplete")
        ));
    }

    #[test]
    fn binary_roundtrip_and_crc_rejection() {
        let temp = std::env::temp_dir().join(format!(
            "sranibro_reseat_test_{}_{}.bin",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let eye = template();
        let reference = Reference {
            context: ReferenceContext::new("pimax_xr5", "unit-test".into(), 42),
            captured_unix: 123,
            eyes: [eye.clone(), eye],
        };
        save_to(&temp, &reference).unwrap();
        let loaded = load_from(&temp).unwrap();
        assert_eq!(loaded.context, reference.context);
        assert_eq!(loaded.captured_unix, 123);
        let mut bytes = fs::read(&temp).unwrap();
        bytes[32] ^= 0x80;
        fs::write(&temp, bytes).unwrap();
        assert!(load_from(&temp).unwrap_err().contains("integrity"));
        let _ = fs::remove_file(temp);
    }

    #[test]
    fn interrupted_publish_recovers_the_last_confirmed_backup() {
        let temp = std::env::temp_dir().join(format!(
            "sranibro_reseat_backup_test_{}_{}.bin",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let eye = template();
        let reference = Reference {
            context: ReferenceContext::new("pimax_vr4", "backup-test".into(), 7),
            captured_unix: 456,
            eyes: [eye.clone(), eye],
        };
        save_to(&temp, &reference).unwrap();
        let backup = temp.with_extension("bak");
        fs::rename(&temp, &backup).unwrap();
        let loaded = load_with_backup(&temp).unwrap().unwrap();
        assert_eq!(loaded.captured_unix, 456);
        assert!(temp.is_file() || backup.is_file());
        let _ = fs::remove_file(temp);
        let _ = fs::remove_file(backup);
    }
}
