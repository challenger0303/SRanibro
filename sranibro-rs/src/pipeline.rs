//! Orchestration: adapter -> ML -> SRanipalState -> output sinks, with live
//! telemetry + controls for the UI.
//!
//! Thread layout mirrors the Python `core/pipeline.py`:
//!   adapter --frames--> latest[L/R] --(60Hz)--> EyeNet --> openness raw[L/R]
//!   adapter --gaze----> GazeSample
//!                            |
//!                       (120Hz emit)
//!         SRanipalState::process_frame -> [EyeResult;2] -> Telemetry + sinks

use std::collections::VecDeque;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::blink_timing_fit::ApplyRequest as BlinkTimingApplyRequest;
use crate::config::{EyelidResponseProfile, GazeCorrection, WideSource};
use crate::core::brow_state::BrowState;
use crate::core::eye_state::{CalibStore, EyelidLiveDiag, SRanipalState, Tuning};
use crate::core::types::{
    BlinkTimingProfile, BrightnessNorm, DespeckleParams, DeviceProfile, Eye, EyeResult, EyeSample,
    FlattenParams, GazeEyelidProfile, GazeSample, MlGeometry, PhotometricCorrection, WinkProfile,
};
use crate::core::wide_state::WideState;
use crate::device::HmdAdapter;
use crate::endpoint_fit::EndpointApplyRequest;
use crate::eye_image_http::EyeImageHttpServer;
use crate::gaze_eyelid_fit::ApplyRequest as GazeEyelidApplyRequest;
use crate::ml::brow_net::BrowNet;
use crate::ml::eye_net::EyeNet;
#[cfg(windows)]
use crate::ml::eyelid_model::EyelidGpuContext;
use crate::ml::eyelid_model::{
    build_eyelid_model, CanonicalStereoInput, EyelidBackendReport, LegacyPublishFrame,
    EYELID_INPUT_LEN,
};
use crate::ml::preprocess;
use crate::ml::wide_net::WideNet;
use crate::output::OutputSink;
use crate::wink_fit::ApplyRequest as WinkApplyRequest;

/// One atomically published camera frame. The generation belongs to these exact
/// pixels; consumers must not pair a separately loaded counter with an image.
#[derive(Clone)]
pub struct EyeFrame {
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<[u8]>,
}

impl EyeFrame {
    pub fn view(&self) -> (u32, u32, &[u8]) {
        (self.width, self.height, self.pixels.as_ref())
    }
}

/// One coherent calibration source sample published by the 60 Hz EyeNet worker.
///
/// The UI may present at 24/30 Hz or temporarily stop receiving callbacks.
/// Calibration must therefore consume camera generations,
/// not UI repaints.  Keeping a short, bounded history of Arc-backed frames lets the
/// UI drain every unprocessed sample without copying image payloads here.
#[derive(Clone)]
pub struct CalibrationFrameSnapshot {
    pub captured_at: Instant,
    pub source_generation: [u64; 2],
    pub affine: [[f32; 2]; 2],
    pub frames: [EyeFrame; 2],
    pub gaze: GazeSample,
}

impl CalibrationFrameSnapshot {
    pub fn stereo_frames(&self) -> [Option<EyeFrame>; 2] {
        [Some(self.frames[0].clone()), Some(self.frames[1].clone())]
    }
}

/// A little over two seconds at the EyeNet worker's nominal 60 Hz.  A normal
/// 24 Hz UI drains this every ~42 ms; the extra headroom is for compositor jitter.
/// Long event-loop gaps are detected separately and intentionally discarded.
const CALIBRATION_HISTORY_CAPACITY: usize = 128;
/// Prevent high-resolution Varjo/PSVR2 frames from turning the time-oriented
/// history into a large idle-memory tax. Always retain the newest sample even
/// when one unusually large pair exceeds this budget by itself.
const CALIBRATION_HISTORY_MAX_BYTES: usize = 32 * 1024 * 1024;

fn trim_calibration_history(
    history: &mut VecDeque<CalibrationFrameSnapshot>,
    max_frames: usize,
    max_bytes: usize,
) {
    while history.len() > 1
        && (history.len() > max_frames
            || history
                .iter()
                .map(|sample| sample.frames[0].pixels.len() + sample.frames[1].pixels.len())
                .sum::<usize>()
                > max_bytes)
    {
        history.pop_front();
    }
}

/// The fixed manual-brightness transform and the exact stereo camera pair to which it was
/// applied. Keeping the frame handles in the same mutex snapshot
/// avoids a 120 Hz camera / 60 Hz ML race: a capture consumer can persist the
/// precise pixels described by `affine` instead of accidentally pairing it with
/// newer camera pixels. Cloning this value only clones the two `Arc` handles.
#[derive(Clone)]
pub struct BrightAffineSnapshot {
    pub source_generation: [u64; 2],
    pub affine: [[f32; 2]; 2],
    source_frames: [Option<EyeFrame>; 2],
}

impl BrightAffineSnapshot {
    fn initial() -> Self {
        Self {
            source_generation: [0; 2],
            affine: [[1.0, 0.0]; 2],
            source_frames: [None, None],
        }
    }

    /// Return the exact raw stereo pair used to produce this affine snapshot.
    pub fn stereo_frames(&self) -> [Option<EyeFrame>; 2] {
        self.source_frames.clone()
    }
}

/// Selected expression strength immediately before and after the user range map.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpressionResponseLive {
    pub wide_input: f32,
    pub wide_output: f32,
    pub squeeze_input: f32,
    pub squeeze_output: f32,
}

/// Runtime-only calibration recalled from a user-confirmed eye-image appearance.
/// The emit thread owns all smoothing; selecting a profile never rewrites the
/// per-HMD TOML calibration or the user's base response sliders.
#[derive(Clone, Copy, Debug)]
pub struct WearingCalibrationTarget {
    pub baseline: [f32; 2],
    pub wide_baseline: [f32; 2],
    pub wide_entry_ref: [f32; 2],
    /// Only the image-dependent endpoint coordinates are recalled. Blink timing,
    /// response curve, Squeeze and expression ranges remain the user's current
    /// global settings and cannot change when an appearance match changes.
    pub manual_range: bool,
    pub open_point_offset: [f32; 2],
    pub closed_point_depth: [f32; 2],
    pub profile_id: u64,
    pub confidence: f32,
}

/// Diagnostic-only result after horizontally mirroring both canonical eye inputs and
/// swapping their model channels. Values remain indexed by PHYSICAL eye, so index 1
/// means "the right-eye image evaluated by the model's left-eye output head".
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EyeHeadComparison {
    pub normal_openness: [f32; 2],
    pub normal_squeeze: [f32; 2],
    pub openness: [f32; 2],
    pub squeeze: [f32; 2],
}

/// Live snapshot of the running pipeline, shared with the UI (read-only there).
pub struct Telemetry {
    /// Newest eye frame per eye as `(width, height, bytes)` — dimensions travel with
    /// the frame so resolution is per-HMD (VR4/StarVR 200x200, Varjo higher).
    pub frames: Mutex<[Option<EyeFrame>; 2]>,
    /// Bounded camera-generation clock for calibration consumers. Payloads are Arc
    /// handles, but retaining a frame also retains its pixels, so this must stay capped.
    calibration_history: Mutex<VecDeque<CalibrationFrameSnapshot>>,
    /// Nominal eye-camera resolution from the device profile — the UI's fallback
    /// label before any frame arrives (the live frame's own dims win once streaming).
    pub eye_w: u32,
    pub eye_h: u32,
    pub gaze: Mutex<GazeSample>,
    /// Wall-clock time of the newest VALID gaze report for each eye. Tobii 1289
    /// interleaves short invalid-status packets with valid samples; validity is
    /// therefore freshness-based instead of being cleared by one transient packet.
    gaze_last_valid: Mutex<[Option<Instant>; 2]>,
    /// Newest packet that explicitly reported wearable pupil position/openness. These
    /// streams are sample-and-hold in the merged gaze state, so research capture must
    /// age them out rather than repeating stale native evidence indefinitely.
    pupil_pos_last_reported: Mutex<[Option<Instant>; 2]>,
    openness_last_reported: Mutex<[Option<Instant>; 2]>,
    native_timestamp_last_reported: Mutex<Option<Instant>>,
    pub ml_raw: Mutex<[f32; 2]>,
    /// Legacy per-eye post-processor input layout reconstructed from the stereo
    /// EyeNet output. Only presence, openness, and squeeze are populated in Phase 1;
    /// structural channels 2 and 4 remain zero exactly as in the previous path.
    pub ml5: Mutex<[[f32; 5]; 2]>,
    /// Shadow inference mapped back to physical L/R eyes. Besides the optional A/B
    /// diagnostic, the emit thread uses the right-head values as the SRanipal EyeWide
    /// source while the production mirrored-right route is active. Eyelid openness and
    /// blink output remain on the left head.
    pub eye_head_comparison: Mutex<Option<EyeHeadComparison>>,
    /// Native pupil diameter (mm, valid) per eye [L, R], from stream 1285 when available.
    pub pupil: Mutex<[(f32, bool); 2]>,
    /// RAW brow CNN output per eye [L, R] (one per inference, NOT smoothed/clamped).
    /// The emit thread does the time-based EMA, neutral baseline, and blink hold.
    /// `c_brow` is the inference generation, so the EMA advances once per new
    /// stereo camera pair rather than once per emit tick.
    pub brow_raw: Mutex<[f32; 2]>,
    pub c_brow: AtomicU64,
    /// Whether an eyebrow model is currently loaded. An `AtomicBool` (not a fixed flag) so
    /// [`Pipeline::set_brow`] can hot-load a freshly trained model into the LIVE pipeline
    /// without a device reconnect — the emit thread and the UI both observe the flip.
    pub brow_loaded: AtomicBool,
    /// Custom XR5 image-based EyeWide: raw model score, post-processed comparison,
    /// and the legacy SRanipal-derived value from the same emit frame.
    pub wide_raw: Mutex<[f32; 2]>,
    pub wide_custom: Mutex<[f32; 2]>,
    pub wide_sranipal: Mutex<[f32; 2]>,
    pub c_wide: AtomicU64,
    pub wide_loaded: AtomicBool,
    /// True only while the custom result is fresh, calibrated, and selected for output.
    pub wide_custom_active: AtomicBool,
    /// Custom model neutral-bootstrap visibility for the XR5 card.
    pub wide_ready: Mutex<[bool; 2]>,
    pub wide_bootstrap_seen: Mutex<[u32; 2]>,
    pub results: Mutex<[EyeResult; 2]>,
    pub baselines: Mutex<[f32; 2]>,
    /// Calibrated baseline of the independent ch2 EyeWide state.
    pub wide_baselines: Mutex<[f32; 2]>,
    /// Exact live eyelid-response stages and the reason for any forced close.
    /// The Open / closed editor uses this to distinguish endpoint mapping from a
    /// native/fast-blink snap instead of guessing from the final value.
    pub eyelid_live: Mutex<[EyelidLiveDiag; 2]>,
    /// Final Wide/Squeeze provider output before and after the visual response range.
    pub expression_live: Mutex<[ExpressionResponseLive; 2]>,
    /// Full post-processor calibration snapshot.  The guided onboarding flow
    /// reads this to preserve learned curve anchors when replacing baseline and
    /// blink depth.
    pub calibration: Mutex<Option<CalibStore>>,
    /// The EXACT per-eye image last fed to the eye net (100x100 u8, all filters +
    /// geometry applied), un-mirrored back to natural orientation — the dashboard's
    /// "NET" view shows what the model actually sees.
    pub ml_input: Mutex<[Option<Vec<u8>>; 2]>,
    pub ml_loaded: bool,
    pub device_name: String,
    /// Device-supplied descriptors for the dashboard's pipeline-node detail cards, so the
    /// UI reflects the ACTIVE adapter (Varjo/StarVR/VR4) instead of hardcoding Pimax.
    pub transport: String,
    pub streams: String,
    pub gaze_src: String,
    // Monotonic counters; the UI derives rates from deltas over time.
    pub c_frame_l: AtomicU64,
    pub c_frame_r: AtomicU64,
    pub c_gaze: AtomicU64,
    pub c_ml: AtomicU64,
    pub c_emit: AtomicU64,
    /// Emit cycles that overran the 120Hz period (a real dropped-frame count).
    pub c_drop: AtomicU64,
}

impl Telemetry {
    pub(crate) fn new(
        ml_loaded: bool,
        brow_loaded: bool,
        wide_loaded: bool,
        profile: &DeviceProfile,
    ) -> Arc<Self> {
        Arc::new(Telemetry {
            device_name: profile.name.clone(),
            transport: profile.transport.clone(),
            streams: profile.streams.clone(),
            gaze_src: profile.gaze_src.clone(),
            eye_w: profile.image_w,
            eye_h: profile.image_h,
            brow_raw: Mutex::new([0.0, 0.0]),
            c_brow: AtomicU64::new(0),
            brow_loaded: AtomicBool::new(brow_loaded),
            wide_raw: Mutex::new([0.0, 0.0]),
            wide_custom: Mutex::new([0.0, 0.0]),
            wide_sranipal: Mutex::new([0.0, 0.0]),
            c_wide: AtomicU64::new(0),
            wide_loaded: AtomicBool::new(wide_loaded),
            wide_custom_active: AtomicBool::new(false),
            wide_ready: Mutex::new([false; 2]),
            wide_bootstrap_seen: Mutex::new([0; 2]),
            frames: Mutex::new([None, None]),
            calibration_history: Mutex::new(VecDeque::with_capacity(CALIBRATION_HISTORY_CAPACITY)),
            gaze: Mutex::new(GazeSample::default()),
            gaze_last_valid: Mutex::new([None, None]),
            pupil_pos_last_reported: Mutex::new([None, None]),
            openness_last_reported: Mutex::new([None, None]),
            native_timestamp_last_reported: Mutex::new(None),
            ml_raw: Mutex::new([0.5, 0.5]),
            // Neutral seed: ch0=1.0 (eye present, so the gate doesn't force-close
            // before the first inference / when ML is absent), ch1=0.5 (neutral
            // openness), matching the `ml_raw` [0.5,0.5] no-ML path.
            ml5: Mutex::new([[1.0, 0.5, 0.0, 0.0, 0.0]; 2]),
            eye_head_comparison: Mutex::new(None),
            pupil: Mutex::new([(0.0, false), (0.0, false)]),
            results: Mutex::new([EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)]),
            baselines: Mutex::new([0.6, 0.6]),
            wide_baselines: Mutex::new([0.6, 0.6]),
            eyelid_live: Mutex::new([EyelidLiveDiag::default(); 2]),
            expression_live: Mutex::new([ExpressionResponseLive::default(); 2]),
            calibration: Mutex::new(None),
            ml_input: Mutex::new([None, None]),
            ml_loaded,
            c_frame_l: AtomicU64::new(0),
            c_frame_r: AtomicU64::new(0),
            c_gaze: AtomicU64::new(0),
            c_ml: AtomicU64::new(0),
            c_emit: AtomicU64::new(0),
            c_drop: AtomicU64::new(0),
        })
    }

    /// Latest gaze with the same freshness rule used by the 120 Hz emit thread.
    /// The UI uses this for XR5 centre capture so a stale-but-sticky coordinate cannot
    /// be mistaken for a full second of valid calibration samples.
    pub fn fresh_gaze(&self) -> GazeSample {
        let mut gaze = *self.gaze.lock().unwrap();
        if let Ok(last) = self.gaze_last_valid.lock() {
            let now = Instant::now();
            gaze.left.gaze_valid &= gaze_is_fresh(last[0], now);
            gaze.right.gaze_valid &= gaze_is_fresh(last[1], now);
        }
        gaze
    }

    /// Runtime sample with independent freshness for gaze and native openness.
    /// The eyelid post-processor uses this so a stalled 1285 stream cannot leave a
    /// sticky open-state sample authorizing session-only reseat recovery.
    pub fn fresh_runtime_sample(&self) -> GazeSample {
        let mut gaze = self.fresh_gaze();
        let now = Instant::now();
        if let Ok(last) = self.openness_last_reported.lock() {
            for (eye, timestamp) in [&mut gaze.left, &mut gaze.right].into_iter().zip(*last) {
                if !gaze_is_fresh(timestamp, now) {
                    eye.openness_valid = false;
                    eye.openness_reported = false;
                }
            }
        }
        gaze
    }

    /// Latest native sample with independent freshness for 1289 gaze and 1285
    /// pupil/openness fields. Intended for timestamp-adjacent diagnostic capture;
    /// stale sample-and-hold values become explicitly absent.
    pub fn fresh_capture_sample(&self) -> GazeSample {
        let mut gaze = self.fresh_runtime_sample();
        let now = Instant::now();
        if let Ok(last) = self.pupil_pos_last_reported.lock() {
            for (eye, timestamp) in [&mut gaze.left, &mut gaze.right].into_iter().zip(*last) {
                if !gaze_is_fresh(timestamp, now) {
                    eye.pupil_pos_valid = false;
                    eye.pupil_pos_reported = false;
                }
            }
        }
        if let Ok(last) = self.native_timestamp_last_reported.lock() {
            if !gaze_is_fresh(*last, now) {
                gaze.timestamp_us = 0;
            }
        }
        gaze
    }

    /// Copy both frame handles under one lock. Cloning an `EyeFrame` only clones its
    /// Arc, so capture and UI readers do not copy the camera payload.
    pub fn stereo_frames(&self) -> [Option<EyeFrame>; 2] {
        self.frames.lock().unwrap().clone()
    }

    pub fn frame_generations(&self) -> [u64; 2] {
        let frames = self.frames.lock().unwrap();
        std::array::from_fn(|eye| {
            frames[eye]
                .as_ref()
                .map(|frame| frame.generation)
                .unwrap_or(0)
        })
    }

    fn publish_calibration_frame(&self, sample: CalibrationFrameSnapshot) {
        let mut history = self
            .calibration_history
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if history
            .back()
            .is_some_and(|last| last.source_generation == sample.source_generation)
        {
            return;
        }
        history.push_back(sample);
        trim_calibration_history(
            &mut history,
            CALIBRATION_HISTORY_CAPACITY,
            CALIBRATION_HISTORY_MAX_BYTES,
        );
    }

    /// Return coherent samples for which both eye generations are newer than the
    /// consumer cursor.  The returned values only clone Arc handles.
    pub fn calibration_frames_after(&self, generation: [u64; 2]) -> Vec<CalibrationFrameSnapshot> {
        self.calibration_history
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|sample| {
                sample.source_generation[0] > generation[0]
                    && sample.source_generation[1] > generation[1]
            })
            .cloned()
            .collect()
    }
}

/// Per-unit camera, auxiliary-data, and gaze mapping (handles hardware variants).
#[derive(Clone, Copy, Default)]
pub struct DeviceMap {
    /// Swap the complete physical left/right eye streams.
    pub swap_eyes: bool,
    /// Horizontally mirror each eye image.
    pub flip_image: bool,
    /// Negate the gaze X (left/right) sign at the output.
    pub flip_gaze_x: bool,
}

/// Complete live state installed before the adapter and worker threads start. Keeping
/// this separate from [`DeviceMap`] avoids a one-frame identity/default window at startup
/// and keeps the product UI and diagnostic `run` entrypoint on the same initialization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EyelidModelIdentity {
    pub crc32: u32,
    pub bytes: u64,
}

#[derive(Clone, Copy)]
pub struct PipelineInit {
    pub eyebrow_enabled: bool,
    /// Global EyeWide output master. This does not stop either Wide provider or its
    /// diagnostics; it only controls the final values and Custom Wide pose assist.
    pub eye_wide_enabled: bool,
    pub right_eye_left_head: bool,
    pub eyelid_inference_backend: crate::config::EyelidInferenceBackend,
    pub ml_mirror: [bool; 2],
    pub tuning: Tuning,
    pub geometry: [MlGeometry; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    pub brightness: BrightnessNorm,
    pub photometric_correction: PhotometricCorrection,
    pub gaze_correction: GazeCorrection,
    pub gaze_eyelid_profile: GazeEyelidProfile,
    pub wink_profile: WinkProfile,
    pub blink_timing_profile: BlinkTimingProfile,
    pub eyelid_response_profile: EyelidResponseProfile,
    pub wide_enabled: [bool; 2],
    pub wide_source: WideSource,
    /// Fingerprint of the exact file bytes parsed into the live EyeNet.
    pub eyelid_model_identity: Option<EyelidModelIdentity>,
}

impl Default for PipelineInit {
    fn default() -> Self {
        Self {
            eyebrow_enabled: true,
            eye_wide_enabled: true,
            right_eye_left_head: true,
            eyelid_inference_backend: crate::config::EyelidInferenceBackend::Auto,
            ml_mirror: [false; 2],
            tuning: Tuning::default(),
            geometry: [MlGeometry::default(); 2],
            despeckle: DespeckleParams::default(),
            flatten: FlattenParams::default(),
            brightness: BrightnessNorm::default(),
            photometric_correction: PhotometricCorrection::default(),
            gaze_correction: GazeCorrection::default(),
            gaze_eyelid_profile: GazeEyelidProfile::default(),
            wink_profile: WinkProfile::default(),
            blink_timing_profile: BlinkTimingProfile::default(),
            eyelid_response_profile: EyelidResponseProfile::default(),
            wide_enabled: [true; 2],
            wide_source: WideSource::Sranipal,
            eyelid_model_identity: None,
        }
    }
}

/// On-demand occlusion-heatmap request/result, shared with the ML thread (which owns the
/// net). The UI sets `req` + `mode`; the ML thread computes both eyes on its next tick — a
/// On-demand blocking diagnostic — and publishes `result`, toggling `computing` around it.
/// High-resolution centred sensitivity modes take longer than the one-pass probes.
pub struct HeatState {
    pub req: AtomicBool,
    // 0 = occlusion, 1 = glint, 2 = local brightness, 3 = local contrast.
    pub mode: std::sync::atomic::AtomicU8,
    pub computing: AtomicBool,
    pub result: Mutex<Option<crate::ml::heatmap::HeatResult>>,
}

impl HeatState {
    fn new() -> Self {
        Self {
            req: AtomicBool::new(false),
            mode: std::sync::atomic::AtomicU8::new(0),
            computing: AtomicBool::new(false),
            result: Mutex::new(None),
        }
    }
}

pub struct Pipeline {
    adapter: Option<Box<dyn HmdAdapter>>,
    /// Canonical key for the adapter that is actually running. Unlike `[hmd].device`,
    /// this is resolved after `auto` sniffing and therefore selects the correct per-HMD
    /// geometry/mapping bucket in the live UI.
    pub device_key: String,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    /// Optional loopback HTTP/MJPEG server. It is stopped before an engine reload so
    /// the replacement engine can immediately bind the same port.
    pub(crate) eye_image_http: Option<EyeImageHttpServer>,
    pub tele: Arc<Telemetry>,
    /// Exact input-byte identity captured when the currently loaded EyeNet was parsed.
    pub eyelid_model_identity: Option<EyelidModelIdentity>,
    /// Startup validation/benchmark result for the selected eyelid inference backend.
    pub(crate) eyelid_backend_report: EyelidBackendReport,
    /// When true, the emit thread keeps telemetry live but stops feeding sinks.
    pub paused: Arc<AtomicBool>,
    /// Set to request a one-shot calibration recenter (cleared by the engine).
    pub recenter: Arc<AtomicBool>,
    /// One-shot calibrated state installed by the guided Dream Air flow.  The
    /// emit thread owns `SRanipalState`, so UI code hands the value across here
    /// rather than mutating processor internals concurrently.
    pub guided_calibration: Arc<Mutex<Option<CalibStore>>>,
    /// One-shot explicit endpoint apply/removal request. The emit thread is the
    /// sole owner of SRanipalState and changes only the requested eye(s).
    pub endpoint_apply: Arc<Mutex<Option<EndpointApplyRequest>>>,
    /// One-shot installation of a holdout-validated gaze-dependent eyelid profile.
    /// The emit thread owns both the active profile and its smoothing state.
    pub gaze_eyelid_apply: Arc<Mutex<Option<GazeEyelidApplyRequest>>>,
    /// One-shot installation of a holdout-validated per-eye wink response.
    pub wink_apply: Arc<Mutex<Option<WinkApplyRequest>>>,
    /// One-shot installation of natural-blink visible-bottom timing.
    pub blink_timing_apply: Arc<Mutex<Option<BlinkTimingApplyRequest>>>,
    /// Diagnostic CSV recorder toggle (dashboard REC button): while true, the
    /// emit thread writes one row per frame — raw ml values next to every
    /// post-processing internal — to `sranibro_diag_<unix>.csv` in the app dir.
    pub diag_rec: Arc<AtomicBool>,
    /// On-demand diagnostic: compare each physical eye through the opposite model head
    /// using a mirrored+swapped canonical tensor. Never persisted or used for output.
    pub eye_head_comparison_enabled: Arc<AtomicBool>,
    /// Live production route: replace only the physical right eye's raw openness and
    /// squeeze with the mirrored right image evaluated by EyeNet's left output head.
    pub right_eye_left_head: Arc<AtomicBool>,
    /// One-shot per-eye relaxed-baseline recenter request. Bit 0 = left, bit 1 = right.
    recenter_eye_mask: Arc<AtomicU8>,
    /// Live whole-stream eye identity mapping.
    pub swap_eyes: Arc<AtomicBool>,
    pub flip_image: Arc<AtomicBool>,
    /// Per-eye horizontal mirror applied ONLY to the ML input (A/B experiment for
    /// matching the model's expected eye handedness; see preprocess::vr4_to_input_stereo_flip).
    pub ml_mirror_l: Arc<AtomicBool>,
    pub ml_mirror_r: Arc<AtomicBool>,
    /// Negate gaze X (left/right) at the output — per-device gaze handedness fix.
    pub flip_gaze_x: Arc<AtomicBool>,
    /// Live per-device Pimax post-calibration gaze trim.
    pub gaze_correction: Arc<Mutex<GazeCorrection>>,
    /// Per-eye EyeWide capability gate. Weak source signals are reported and
    /// disabled instead of being amplified into false expressions.
    pub wide_enabled: Arc<Mutex<[bool; 2]>>,
    /// Live XR5 EyeWide provider. Changing it never tears down the camera adapter.
    pub wide_source: Arc<Mutex<WideSource>>,
    /// Global live EyeWide output master. Raw SRanipal/Custom inference and telemetry
    /// continue while false; only final output and Custom Wide pose assist are disabled.
    pub eye_wide_enabled: Arc<AtomicBool>,
    /// Live master switch for eyebrow inference/output. The model remains loaded while
    /// disabled so tracking can resume without rebuilding the pipeline.
    pub eyebrow_enabled: Arc<AtomicBool>,
    /// Live calibration parameters (UI exposes these as sliders).
    pub tuning: Arc<Mutex<Tuning>>,
    /// Per-HMD response trim. UI drags write only this live preview; persistence
    /// happens explicitly when the user presses Apply.
    pub eyelid_response: Arc<Mutex<EyelidResponseProfile>>,
    /// Appearance-conditioned target selected by the UI-side matcher. `None`
    /// fades transient offsets back to the normal per-HMD calibration.
    pub wearing_calibration_target: Arc<Mutex<Option<WearingCalibrationTarget>>>,
    /// Manual endpoint edits must take effect on the very next emit frame. This
    /// clears the emit-thread's private blend state without changing saved memory.
    wearing_response_reset: Arc<AtomicBool>,
    manual_endpoint_edit: Arc<Mutex<Option<([f32; 2], EyelidResponseProfile)>>>,
    /// Live PER-EYE ML-input geometry `[left, right]` (crop/stretch/rotation), read by
    /// the ML thread each frame and edited live in the gear modal. Identity = legacy
    /// resize for that eye.
    pub geometry: Arc<Mutex<[crate::core::types::MlGeometry; 2]>>,
    /// Live per-device specular-dot suppression applied to the ML input (before geometry).
    pub despeckle: Arc<Mutex<crate::core::types::DespeckleParams>>,
    /// Live per-device illumination flatten (close-up shadow removal), after despeckle.
    pub flatten: Arc<Mutex<crate::core::types::FlattenParams>>,
    /// Live per-device manual eye-image brightness. Legacy adaptive fields remain in the
    /// persisted type for compatibility but are deliberately ignored by the ML path.
    pub brightness: Arc<Mutex<crate::core::types::BrightnessNorm>>,
    /// Fixed, fitted correction after manual brightness and before geometry.
    pub photometric_correction: Arc<Mutex<PhotometricCorrection>>,
    /// The per-eye brightness affine (a,b) the ML thread last applied — for the UI preview.
    /// Source generations and exact source frame handles are published under
    /// the same mutex as the affine. Recording paths must use that bundled pair.
    pub bright_affine: Arc<Mutex<BrightAffineSnapshot>>,
    /// On-demand ML occlusion heatmap (diagnostic), computed by the ML thread.
    pub heatmap: Arc<HeatState>,
    /// Live device status string (for the UI's diagnostic line).
    pub device_status: Arc<Mutex<String>>,
    /// The live eyebrow model, shared with the brow worker so a freshly trained model can be
    /// hot-swapped in ([`Pipeline::set_brow`]) with no device reconnect. `None` = no brow.
    brow: Arc<Mutex<Option<BrowNet>>>,
    /// Optional custom XR5 EyeWide model; shared so a fitted model can be hot-loaded later.
    wide: Arc<Mutex<Option<WideNet>>>,
    /// Dedicated reset for custom Wide normalization when its model is hot-swapped.
    wide_recenter: Arc<AtomicBool>,
}

/// Horizontally mirror a `w`x`h` grayscale image (row-major).
fn mirror_h(px: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            out[row + (w - 1 - x)] = px[row + x];
        }
    }
    out
}

/// Consume a stereo brow pair only after both eyes have advanced. Alternating
/// L/R callbacks therefore cannot re-infer the unchanged eye image.
fn stereo_pair_advanced(generation: [u64; 2], previous: [u64; 2]) -> bool {
    generation[0] != 0
        && generation[1] != 0
        && generation[0] != previous[0]
        && generation[1] != previous[1]
}

fn merge_eye_sample(dst: &mut EyeSample, src: EyeSample) {
    if src.gaze_reported {
        if src.gaze_valid {
            dst.gaze = src.gaze;
            dst.gaze_valid = true;
        }
        dst.gaze_reported = true;
    }
    if src.origin_valid {
        dst.origin_mm = src.origin_mm;
        dst.origin_valid = true;
    }
    if src.pupil_valid {
        dst.pupil_mm = src.pupil_mm;
        dst.pupil_valid = true;
    }
    if src.pupil_pos_reported {
        if src.pupil_pos_valid {
            dst.pupil_pos = src.pupil_pos;
        }
        dst.pupil_pos_valid = src.pupil_pos_valid;
        dst.pupil_pos_reported = true;
    }
    if src.openness_reported {
        dst.openness = src.openness;
        dst.openness_valid = src.openness_valid;
        dst.openness_reported = true;
    }
}

fn merge_gaze_sample(dst: &mut GazeSample, src: GazeSample) {
    if src.timestamp_us != 0 {
        dst.timestamp_us = src.timestamp_us;
    }
    merge_eye_sample(&mut dst.left, src.left);
    merge_eye_sample(&mut dst.right, src.right);
}

/// Correct a physically reversed unit by exchanging the complete eye streams.
/// Keeping every eye-scoped field together prevents a normal conjugate look from
/// hiding an inverted vergence channel order.
fn route_eye_sample(mut sample: GazeSample, swap_eye_streams: bool) -> GazeSample {
    if swap_eye_streams {
        std::mem::swap(&mut sample.left, &mut sample.right);
    }
    sample
}

fn apply_gaze_x_handedness(results: &mut [EyeResult; 2], flip: bool) {
    if flip {
        for result in results {
            result.gaze[0] = -result.gaze[0];
        }
    }
}

/// Return a continuous confidence that the legacy lid already represents an open eye.
/// Custom Wide may strengthen an open lid, but it must not reinterpret a narrowed or
/// closing lid as Wide. A smooth ramp avoids a one-frame mode switch at either boundary.
fn custom_wide_pose_eligibility(openness: f32) -> f32 {
    let t = ((openness.clamp(0.0, 1.0) - 0.35) / (0.75 - 0.35)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Blend the legacy openness/squeeze pose toward fully open in proportion to Custom Wide.
/// The Wide envelope has a short release tail, so eligibility is derived continuously from
/// the current lid pose rather than from WideState's per-eye boolean `active` flag. When a
/// bilateral gesture is credible, the more-open eye is the reference; a closing eye must
/// never drag its open partner downward.
fn apply_custom_wide_pose(
    results: &mut [EyeResult; 2],
    use_custom: bool,
    wide_requires_both: bool,
) {
    if !use_custom {
        return;
    }

    let mut eligibility = [0.0; 2];
    for i in 0..2 {
        if results[i].blink || !results[i].openness_valid || results[i].wide <= 0.002 {
            continue;
        }
        let natural = results[i].openness.clamp(0.0, 1.0);
        eligibility[i] = custom_wide_pose_eligibility(natural);
        if eligibility[i] <= f32::EPSILON {
            continue;
        }
        let assist = results[i].wide.clamp(0.0, 1.0) * eligibility[i];
        results[i].openness = natural + assist * (1.0 - natural);
        results[i].squeeze *= 1.0 - assist;
    }

    let bilateral = wide_requires_both
        && !results[0].blink
        && !results[1].blink
        && results[0].openness_valid
        && results[1].openness_valid
        && results[0].wide > 0.002
        && results[1].wide > 0.002;
    if bilateral {
        let wide = results[0].wide.min(results[1].wide).clamp(0.0, 1.0);
        let t = ((wide - 0.02) / (0.20 - 0.02)).clamp(0.0, 1.0);
        let gesture = t * t * (3.0 - 2.0 * t);
        let pair = eligibility[0].min(eligibility[1]) * gesture;
        let shared = results[0].openness.max(results[1].openness);
        for result in results {
            result.openness += pair * (shared - result.openness);
        }
    }
}

/// Apply the final EyeWide output policy after provider selection. Per-eye capability is
/// intentionally independent from the global master: guided XR5 setup may update the
/// former, but must never re-enable the user's avatar-level preference. Returning whether
/// Custom Wide is actually exposed keeps telemetry aligned with the values sent to sinks.
fn apply_eye_wide_output_policy(
    results: &mut [EyeResult; 2],
    master_enabled: bool,
    capability: [bool; 2],
    use_custom: bool,
    wide_requires_both: bool,
) -> bool {
    if master_enabled {
        for i in 0..2 {
            if !capability[i] {
                results[i].wide = 0.0;
            }
        }
    } else {
        // Emit explicit zeroes rather than relying on missing/invalid semantics: every
        // sink should actively release an avatar's previously latched EyeWide pose.
        results[0].wide = 0.0;
        results[1].wide = 0.0;
    }

    let custom_output_active =
        master_enabled && use_custom && capability.iter().copied().any(|enabled| enabled);
    apply_custom_wide_pose(results, custom_output_active, wide_requires_both);
    custom_output_active
}

fn map_expression_range(value: f32, start: f32, full: f32) -> f32 {
    if !value.is_finite() {
        return 0.0;
    }
    let start = if start.is_finite() {
        start.clamp(0.0, 0.95)
    } else {
        0.0
    };
    let full = if full.is_finite() {
        full.clamp(start + EyelidResponseProfile::MIN_EXPRESSION_RANGE, 1.0)
    } else {
        1.0
    };
    ((value - start) / (full - start).max(EyelidResponseProfile::MIN_EXPRESSION_RANGE))
        .clamp(0.0, 1.0)
}

fn blend_response_profile(
    current: EyelidResponseProfile,
    target: EyelidResponseProfile,
    alpha: f32,
) -> EyelidResponseProfile {
    let alpha = alpha.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * alpha;
    let mut next = target;
    for eye in 0..2 {
        next.open_point_offset[eye] = lerp(
            current.open_point_offset[eye],
            target.open_point_offset[eye],
        );
        next.closed_point_depth[eye] = lerp(
            current.closed_point_depth[eye],
            target.closed_point_depth[eye],
        );
        next.wide_start[eye] = lerp(current.wide_start[eye], target.wide_start[eye]);
        next.wide_full[eye] = lerp(current.wide_full[eye], target.wide_full[eye]);
        next.squeeze_start[eye] = lerp(current.squeeze_start[eye], target.squeeze_start[eye]);
        next.squeeze_full[eye] = lerp(current.squeeze_full[eye], target.squeeze_full[eye]);
        next.close_depth_scale[eye] = lerp(
            current.close_depth_scale[eye],
            target.close_depth_scale[eye],
        );
        next.curve_mid_output[eye] =
            lerp(current.curve_mid_output[eye], target.curve_mid_output[eye]);
    }
    next.blink_close_ms = lerp(current.blink_close_ms, target.blink_close_ms);
    next.snap_gate_open = lerp(current.snap_gate_open, target.snap_gate_open);
    next.sanitized()
}

fn response_profiles_near(a: EyelidResponseProfile, b: EyelidResponseProfile) -> bool {
    let mut max_delta = (a.blink_close_ms - b.blink_close_ms).abs();
    max_delta = max_delta.max((a.snap_gate_open - b.snap_gate_open).abs());
    for eye in 0..2 {
        max_delta = max_delta.max((a.open_point_offset[eye] - b.open_point_offset[eye]).abs());
        max_delta = max_delta.max((a.closed_point_depth[eye] - b.closed_point_depth[eye]).abs());
        max_delta = max_delta.max((a.wide_start[eye] - b.wide_start[eye]).abs());
        max_delta = max_delta.max((a.wide_full[eye] - b.wide_full[eye]).abs());
        max_delta = max_delta.max((a.squeeze_start[eye] - b.squeeze_start[eye]).abs());
        max_delta = max_delta.max((a.squeeze_full[eye] - b.squeeze_full[eye]).abs());
        max_delta = max_delta.max((a.close_depth_scale[eye] - b.close_depth_scale[eye]).abs());
        max_delta = max_delta.max((a.curve_mid_output[eye] - b.curve_mid_output[eye]).abs());
    }
    max_delta <= 0.0005 && a.manual_range == b.manual_range && a.auto_reseat == b.auto_reseat
}

/// Apply the user-facing expression ranges only after the Wide provider and capability
/// policy are settled. Internal blink/wink recognition therefore continues to see the
/// model's established signals, while every output sink sees the same final mapping.
fn apply_expression_response_ranges(
    results: &mut [EyeResult; 2],
    profile: EyelidResponseProfile,
    exclusive: bool,
) -> [ExpressionResponseLive; 2] {
    let mut live = [ExpressionResponseLive::default(); 2];
    for eye in 0..2 {
        let wide_input = results[eye].wide.clamp(0.0, 1.0);
        let squeeze_input = results[eye].squeeze.clamp(0.0, 1.0);
        results[eye].wide =
            map_expression_range(wide_input, profile.wide_start[eye], profile.wide_full[eye]);
        results[eye].squeeze = map_expression_range(
            squeeze_input,
            profile.squeeze_start[eye],
            profile.squeeze_full[eye],
        );
        live[eye].wide_input = wide_input;
        live[eye].squeeze_input = squeeze_input;
    }
    if exclusive {
        for result in results.iter_mut() {
            let (wide, squeeze) = (result.wide, result.squeeze);
            result.wide = wide * (1.0 - squeeze).clamp(0.0, 1.0);
            result.squeeze = squeeze * (1.0 - wide).clamp(0.0, 1.0);
        }
    }
    for eye in 0..2 {
        live[eye].wide_output = results[eye].wide;
        live[eye].squeeze_output = results[eye].squeeze;
    }
    live
}

/// Optionally collapse a valid stereo eyebrow pair to one shared signed value.
/// Averaging happens after each eye's own baseline, deadzone and smoothing, so the
/// switch removes residual L/R imbalance without changing either model input.
fn apply_brow_lr_sync(results: &mut [EyeResult; 2], enabled: bool) {
    if !enabled || !results[0].brow_valid || !results[1].brow_valid {
        return;
    }
    let shared = (0.5 * (results[0].brow + results[1].brow)).clamp(-1.0, 1.0);
    results[0].brow = shared;
    results[1].brow = shared;
}

/// Convert canonical `[L, R]` model input into `[mirror(R), mirror(L)]`. This preserves
/// the paired stereo context while evaluating each physical eye through the opposite
/// handedness/output head. The destination is caller-owned so the diagnostic can reuse
/// one buffer instead of allocating on every probe.
fn mirror_swap_canonical_stereo(src: &[f32], dst: &mut [f32]) -> bool {
    let side = preprocess::DST;
    let plane = side * side;
    if src.len() != 2 * plane || dst.len() != 2 * plane {
        return false;
    }
    for dst_eye in 0..2 {
        let src_eye = 1 - dst_eye;
        for y in 0..side {
            for x in 0..side {
                dst[dst_eye * plane + y * side + x] =
                    src[src_eye * plane + y * side + (side - 1 - x)];
            }
        }
    }
    true
}

fn physical_eye_head_comparison(
    normal_ml5: [[f32; 5]; 2],
    shadow_ml5: [[f32; 5]; 2],
) -> EyeHeadComparison {
    // shadow ch0/out-left contains physical RIGHT; shadow ch1/out-right contains
    // physical LEFT. Map back to the UI's ordinary [physical L, physical R] order.
    EyeHeadComparison {
        normal_openness: [normal_ml5[0][1], normal_ml5[1][1]],
        normal_squeeze: [normal_ml5[0][3], normal_ml5[1][3]],
        openness: [shadow_ml5[1][1], shadow_ml5[0][1]],
        squeeze: [shadow_ml5[1][3], shadow_ml5[0][3]],
    }
}

/// Replace only physical RIGHT eyelid capabilities. Presence remains the normal
/// stereo inference's value, and structural channels stay untouched.
fn apply_right_left_head(frame: &mut LegacyPublishFrame, comparison: EyeHeadComparison) {
    frame.ml_raw[1] = comparison.openness[1];
    frame.ml5[1][1] = comparison.openness[1];
    frame.ml5[1][3] = comparison.squeeze[1];
}

/// Build the per-physical-eye right-head frame used only for EyeWide detection.
///
/// Normal inference is `[L, R]`, so its ch2/ch4 belong to physical RIGHT. Shadow
/// inference is `[mirror(R), mirror(L)]`, so its ch2/ch4 belong to physical LEFT.
/// Mapping those two values back to `[L, R]` lets both eyes use the same right EyeNet
/// head, exactly mirroring the production eyelid route where both eyes use ch1/ch3.
fn ch2_wide_frame(production_ml5: [[f32; 5]; 2], comparison: EyeHeadComparison) -> [[f32; 5]; 2] {
    [
        [
            production_ml5[0][0],
            comparison.openness[0],
            0.0,
            comparison.squeeze[0],
            0.0,
        ],
        [
            production_ml5[1][0],
            comparison.normal_openness[1],
            0.0,
            comparison.normal_squeeze[1],
            0.0,
        ],
    ]
}

/// Open the sole Wide-to-non-Wide interaction while output is disabled, without
/// mutating the user's saved tuning. Otherwise a hidden Wide value could attenuate
/// squeeze before the final output policy zeroes EyeWide.
fn apply_eye_wide_master_to_tuning(tuning: &mut Tuning, master_enabled: bool) {
    if !master_enabled {
        tuning.wide_squeeze_exclusive = false;
    }
}

/// Drain one pending explicit-endpoint apply/removal into the live post-processor.
///
/// Keeping the drain and application together lets the emit loop use the exact
/// same operation during normal frames and once more during shutdown. The latter
/// closes the small window where the UI can publish a validated endpoint just
/// before `Pipeline::stop` makes the emit loop exit.
fn drain_endpoint_apply(
    pending: &Mutex<Option<EndpointApplyRequest>>,
    state: &mut SRanipalState,
) -> bool {
    let request = pending
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let Some(request) = request else {
        return false;
    };

    let mut applied = false;
    for (index, eye) in [Eye::Left, Eye::Right].into_iter().enumerate() {
        if request.reset_to_adaptive[index] {
            state.remove_explicit_endpoint(eye);
            applied = true;
        } else if let Some(endpoint) = request.eyes[index].as_ref() {
            state.apply_explicit_endpoint(
                eye,
                endpoint.baseline,
                endpoint.blink_depth,
                endpoint.calibrated_unix,
            );
            applied = true;
        }
    }
    applied
}

#[cfg(test)]
mod merge_tests {
    use super::*;
    use crate::endpoint_fit::EndpointApply;

    #[test]
    fn reload_transfers_shutdown_without_waiting_on_the_ui_owner() {
        struct TestAdapter {
            profile: crate::core::types::DeviceProfile,
            stopped_on: Arc<Mutex<Vec<std::thread::ThreadId>>>,
        }
        impl HmdAdapter for TestAdapter {
            fn name(&self) -> &'static str {
                "reload-test"
            }
            fn profile(&self) -> &crate::core::types::DeviceProfile {
                &self.profile
            }
            fn start(
                &mut self,
                _: crate::device::FrameFn,
                _: crate::device::GazeFn,
            ) -> io::Result<()> {
                Ok(())
            }
            fn stop(&mut self) {
                self.stopped_on.lock().unwrap().push(thread::current().id());
            }
        }
        fn assert_send<T: Send>() {}
        assert_send::<crate::engine::Engine>();
        let stopped_on = Arc::new(Mutex::new(Vec::new()));
        let mut pipeline = Pipeline::run(
            Box::new(TestAdapter {
                profile: Default::default(),
                stopped_on: stopped_on.clone(),
            }),
            None,
            None,
            None,
            Vec::new(),
            DeviceMap::default(),
            Arc::new(Mutex::new(String::new())),
            "reload-unit-test".into(),
            PipelineInit::default(),
        )
        .unwrap();
        let shutdown = pipeline.take_shutdown();
        assert!(stopped_on.lock().unwrap().is_empty());
        let owner = thread::current().id();
        thread::spawn(shutdown).join().unwrap();
        pipeline.stop(); // Old UI handle is safe to stop again after ownership moved.
        let calls = stopped_on.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_ne!(calls[0], owner);
    }

    #[test]
    fn remembered_response_enters_and_leaves_without_a_jump() {
        let base = EyelidResponseProfile::default();
        let mut remembered = base;
        remembered.open_point_offset = [0.12, 0.14];
        remembered.closed_point_depth = [0.22, 0.26];

        let first_enter = blend_response_profile(base, remembered, 0.012);
        assert!(first_enter.open_point_offset[0] > base.open_point_offset[0]);
        assert!(first_enter.open_point_offset[0] < remembered.open_point_offset[0]);

        let first_leave = blend_response_profile(remembered, base, 0.012);
        assert!(first_leave.open_point_offset[0] < remembered.open_point_offset[0]);
        assert!(first_leave.open_point_offset[0] > base.open_point_offset[0]);

        let mut faded = remembered;
        for _ in 0..1000 {
            faded = blend_response_profile(faded, base, 0.012);
        }
        assert!(response_profiles_near(faded, base));
    }

    #[test]
    fn eye_head_ab_mirrors_and_swaps_without_mutating_normal_input() {
        let side = preprocess::DST;
        let plane = side * side;
        let src: Vec<f32> = (0..2 * plane).map(|index| index as f32).collect();
        let original = src.clone();
        let mut shadow = vec![0.0; src.len()];
        assert!(mirror_swap_canonical_stereo(&src, &mut shadow));

        for eye in 0..2 {
            for &(x, y) in &[(0, 0), (3, 7), (side - 1, side - 1)] {
                let got = shadow[eye * plane + y * side + x];
                let want = src[(1 - eye) * plane + y * side + (side - 1 - x)];
                assert_eq!(got.to_bits(), want.to_bits());
            }
        }
        assert_eq!(src, original, "shadow transform changed production input");

        let mut roundtrip = vec![0.0; src.len()];
        assert!(mirror_swap_canonical_stereo(&shadow, &mut roundtrip));
        assert_eq!(roundtrip, src, "mirror+swap must be self-inverse");
    }

    #[test]
    fn eye_head_ab_maps_shadow_outputs_back_to_physical_eyes() {
        let normal = [[0.9, 0.81, 0.0, 0.11, 0.0], [0.9, 0.42, 0.0, 0.52, 0.0]];
        let shadow = [
            [0.8, 0.21, 0.0, 0.31, 0.0], // physical RIGHT through LEFT head
            [0.8, 0.72, 0.0, 0.62, 0.0], // physical LEFT through RIGHT head
        ];
        let comparison = physical_eye_head_comparison(normal, shadow);
        assert_eq!(comparison.normal_openness, [0.81, 0.42]);
        assert_eq!(comparison.normal_squeeze, [0.11, 0.52]);
        assert_eq!(comparison.openness, [0.72, 0.21]);
        assert_eq!(comparison.squeeze, [0.62, 0.31]);
    }

    #[test]
    fn right_left_head_route_changes_only_right_openness_and_squeeze() {
        let mut frame = LegacyPublishFrame {
            ml_raw: [0.81, 0.42],
            ml5: [[0.9, 0.81, 0.0, 0.11, 0.0], [0.9, 0.42, 0.0, 0.52, 0.0]],
        };
        let before = frame;
        apply_right_left_head(
            &mut frame,
            EyeHeadComparison {
                normal_openness: before.ml_raw,
                normal_squeeze: [before.ml5[0][3], before.ml5[1][3]],
                openness: [0.72, 0.21],
                squeeze: [0.62, 0.31],
            },
        );

        assert_eq!(frame.ml_raw[0].to_bits(), before.ml_raw[0].to_bits());
        assert_eq!(frame.ml5[0], before.ml5[0]);
        assert_eq!(frame.ml_raw[1], 0.21);
        assert_eq!(frame.ml5[1][1], 0.21);
        assert_eq!(frame.ml5[1][3], 0.31);
        assert_eq!(frame.ml5[1][0], before.ml5[1][0], "presence unchanged");
        assert_eq!(frame.ml5[1][2], 0.0, "structural channel unchanged");
        assert_eq!(frame.ml5[1][4], 0.0, "structural channel unchanged");
    }

    #[test]
    fn ch2_wide_frame_routes_both_physical_eyes_through_right_head() {
        // Production eyelids already use normal ch1 for LEFT and shadow ch1 for RIGHT.
        let production = [[0.91, 0.81, 0.0, 0.11, 0.0], [0.91, 0.21, 0.0, 0.31, 0.0]];
        let comparison = EyeHeadComparison {
            // Normal ch2/ch4 are physical RIGHT through the right head.
            normal_openness: [0.81, 0.92],
            normal_squeeze: [0.11, 0.12],
            // Shadow values are mapped to physical [L, R], so index 0 is
            // mirror(LEFT) through the right head.
            openness: [0.88, 0.21],
            squeeze: [0.18, 0.31],
        };

        let wide = ch2_wide_frame(production, comparison);
        assert_eq!(wide[0], [0.91, 0.88, 0.0, 0.18, 0.0]);
        assert_eq!(wide[1], [0.91, 0.92, 0.0, 0.12, 0.0]);
    }

    fn asymmetric_routing_sample() -> GazeSample {
        GazeSample {
            timestamp_us: 77,
            left: EyeSample {
                gaze: [0.40, -0.10, 0.91],
                gaze_valid: true,
                gaze_reported: true,
                origin_mm: [1.0, 2.0, 3.0],
                origin_valid: true,
                pupil_mm: 3.1,
                pupil_valid: true,
                pupil_pos: [0.2, 0.3],
                pupil_pos_valid: true,
                pupil_pos_reported: true,
                openness: 0.25,
                openness_valid: true,
                openness_reported: true,
            },
            right: EyeSample {
                gaze: [-0.55, 0.20, 0.81],
                gaze_valid: false,
                gaze_reported: false,
                origin_mm: [4.0, 5.0, 6.0],
                origin_valid: false,
                pupil_mm: 4.2,
                pupil_valid: false,
                pupil_pos: [0.7, 0.8],
                pupil_pos_valid: false,
                pupil_pos_reported: true,
                openness: 0.85,
                openness_valid: false,
                openness_reported: true,
            },
        }
    }

    fn history_sample(generation: u64) -> CalibrationFrameSnapshot {
        let frame = |offset| EyeFrame {
            generation: generation + offset,
            width: 1,
            height: 1,
            pixels: Arc::from([generation as u8]),
        };
        CalibrationFrameSnapshot {
            captured_at: Instant::now(),
            source_generation: [generation, generation + 1000],
            affine: [[1.0, 0.0]; 2],
            frames: [frame(0), frame(1000)],
            gaze: GazeSample {
                timestamp_us: generation,
                ..Default::default()
            },
        }
    }

    #[test]
    fn calibration_history_is_bounded_ordered_and_generation_addressable() {
        let tele = Telemetry::new(false, false, false, &DeviceProfile::default());
        for generation in 1..=(CALIBRATION_HISTORY_CAPACITY as u64 + 9) {
            tele.publish_calibration_frame(history_sample(generation));
        }
        let all = tele.calibration_frames_after([0, 0]);
        assert_eq!(all.len(), CALIBRATION_HISTORY_CAPACITY);
        assert_eq!(
            all.first().unwrap().source_generation[0],
            10,
            "old Arc-backed images must be released at the fixed capacity"
        );
        assert_eq!(
            all.last().unwrap().source_generation[0],
            CALIBRATION_HISTORY_CAPACITY as u64 + 9
        );

        let tail = tele.calibration_frames_after([
            CALIBRATION_HISTORY_CAPACITY as u64 + 5,
            CALIBRATION_HISTORY_CAPACITY as u64 + 1005,
        ]);
        assert_eq!(tail.len(), 4);
        assert_eq!(
            tail[0].source_generation[0],
            CALIBRATION_HISTORY_CAPACITY as u64 + 6
        );
    }

    #[test]
    fn calibration_history_has_a_byte_budget_for_large_camera_frames() {
        let mut history = VecDeque::new();
        for generation in 1..=4 {
            let frame = |offset| EyeFrame {
                generation: generation + offset,
                width: 20,
                height: 20,
                pixels: Arc::from(vec![generation as u8; 400]),
            };
            history.push_back(CalibrationFrameSnapshot {
                captured_at: Instant::now(),
                source_generation: [generation, generation + 1000],
                affine: [[1.0, 0.0]; 2],
                frames: [frame(0), frame(1000)],
                gaze: GazeSample::default(),
            });
            trim_calibration_history(&mut history, 128, 1_600);
        }
        assert_eq!(history.len(), 2);
        assert_eq!(history.front().unwrap().source_generation[0], 3);
        assert_eq!(history.back().unwrap().source_generation[0], 4);
    }

    #[test]
    fn runtime_sample_expires_native_openness_independently_from_live_gaze() {
        let tele = Telemetry::new(false, false, false, &DeviceProfile::default());
        let eye = EyeSample {
            gaze: [0.0, 0.0, -1.0],
            gaze_valid: true,
            gaze_reported: true,
            openness: 1.0,
            openness_valid: true,
            openness_reported: true,
            ..Default::default()
        };
        *tele.gaze.lock().unwrap() = GazeSample {
            left: eye,
            right: eye,
            ..Default::default()
        };

        let now = Instant::now();
        *tele.gaze_last_valid.lock().unwrap() = [Some(now); 2];
        *tele.openness_last_reported.lock().unwrap() =
            [Some(now - GAZE_VALID_GRACE - Duration::from_millis(1)); 2];

        let stale = tele.fresh_runtime_sample();
        assert!(stale.left.gaze_valid && stale.right.gaze_valid);
        assert!(!stale.left.openness_reported && !stale.right.openness_reported);
        assert!(!stale.left.openness_valid && !stale.right.openness_valid);

        let refreshed = Instant::now();
        *tele.gaze_last_valid.lock().unwrap() = [Some(refreshed); 2];
        *tele.openness_last_reported.lock().unwrap() = [Some(refreshed); 2];
        let fresh = tele.fresh_runtime_sample();
        assert!(fresh.left.gaze_valid && fresh.right.gaze_valid);
        assert!(fresh.left.openness_reported && fresh.right.openness_reported);
        assert!(fresh.left.openness_valid && fresh.right.openness_valid);
    }

    #[test]
    fn whole_stream_swap_keeps_every_eye_scoped_field_together() {
        let source = asymmetric_routing_sample();
        let routed = route_eye_sample(source, true);

        assert_eq!(routed.timestamp_us, source.timestamp_us);
        assert_eq!(routed.left.gaze, source.right.gaze);
        assert_eq!(routed.right.gaze, source.left.gaze);
        assert_eq!(routed.left.gaze_valid, source.right.gaze_valid);
        assert_eq!(routed.right.gaze_reported, source.left.gaze_reported);
        assert_eq!(routed.left.origin_mm, source.right.origin_mm);
        assert_eq!(routed.right.origin_valid, source.left.origin_valid);
        assert_eq!(routed.left.pupil_mm, source.right.pupil_mm);
        assert_eq!(routed.right.pupil_pos, source.left.pupil_pos);
        assert_eq!(routed.left.openness, source.right.openness);
        assert_eq!(routed.right.openness_valid, source.left.openness_valid);
    }

    #[test]
    fn whole_stream_swap_off_preserves_eye_identity() {
        let source = asymmetric_routing_sample();
        let routed = route_eye_sample(source, false);

        assert_eq!(routed.timestamp_us, source.timestamp_us);
        assert_eq!(
            routed.left.gaze.map(f32::to_bits),
            source.left.gaze.map(f32::to_bits)
        );
        assert_eq!(
            routed.right.gaze.map(f32::to_bits),
            source.right.gaze.map(f32::to_bits)
        );
        assert_eq!(routed.left.origin_mm, source.left.origin_mm);
        assert_eq!(routed.right.pupil_mm, source.right.pupil_mm);
        assert_eq!(routed.left.openness, source.left.openness);
        assert_eq!(routed.right.openness, source.right.openness);
    }

    #[test]
    fn swapped_pimax_stream_plus_x_flip_repairs_vergence_without_changing_conjugate_motion() {
        let convergence = GazeSample {
            left: EyeSample {
                gaze: [0.39, 0.0, 0.92],
                ..Default::default()
            },
            right: EyeSample {
                gaze: [-0.45, 0.0, 0.89],
                ..Default::default()
            },
            ..Default::default()
        };
        let routed = route_eye_sample(convergence, true);
        let corrected_x = [-routed.left.gaze[0], -routed.right.gaze[0]];
        assert!(corrected_x[0] > 0.0 && corrected_x[1] < 0.0);

        let conjugate = GazeSample {
            left: EyeSample {
                gaze: [0.25, -0.05, 0.96],
                ..Default::default()
            },
            right: EyeSample {
                gaze: [0.25, -0.05, 0.96],
                ..Default::default()
            },
            ..Default::default()
        };
        let swapped = route_eye_sample(conjugate, true);
        let normal = route_eye_sample(conjugate, false);
        assert_eq!(swapped.left.gaze, normal.left.gaze);
        assert_eq!(swapped.right.gaze, normal.right.gaze);
    }
    #[test]
    fn horizontal_flip_changes_only_gaze_x() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].gaze = [0.25, -0.30, 0.92];
        results[1].gaze = [-0.40, 0.20, 0.88];
        results[0].gaze_valid = true;
        results[1].gaze_valid = false;

        apply_gaze_x_handedness(&mut results, true);

        assert_eq!(results[0].gaze, [-0.25, -0.30, 0.92]);
        assert_eq!(results[1].gaze, [0.40, 0.20, 0.88]);
        assert!(results[0].gaze_valid);
        assert!(!results[1].gaze_valid);
    }

    #[test]
    fn brow_pair_requires_both_eye_generations_to_advance() {
        assert!(!stereo_pair_advanced([0, 0], [0, 0]));
        assert!(stereo_pair_advanced([1, 1], [0, 0]));
        assert!(!stereo_pair_advanced([2, 1], [1, 1]));
        assert!(!stereo_pair_advanced([1, 2], [1, 1]));
        assert!(stereo_pair_advanced([2, 2], [1, 1]));
        assert!(stereo_pair_advanced([8, 12], [5, 9]));
    }

    #[test]
    fn brightness_snapshot_keeps_affine_and_exact_frame_generations_together() {
        let left = EyeFrame {
            generation: 17,
            width: 2,
            height: 1,
            pixels: Arc::from([1_u8, 2]),
        };
        let right = EyeFrame {
            generation: 23,
            width: 2,
            height: 1,
            pixels: Arc::from([3_u8, 4]),
        };
        let snapshot = BrightAffineSnapshot {
            source_generation: [left.generation, right.generation],
            affine: [[1.2, -3.0], [0.8, 4.0]],
            source_frames: [Some(left.clone()), Some(right.clone())],
        };

        let frames = snapshot.stereo_frames();
        assert_eq!(snapshot.source_generation, [17, 23]);
        assert_eq!(snapshot.affine, [[1.2, -3.0], [0.8, 4.0]]);
        assert_eq!(frames[0].as_ref().map(|frame| frame.generation), Some(17));
        assert_eq!(frames[1].as_ref().map(|frame| frame.generation), Some(23));
        assert!(Arc::ptr_eq(
            &frames[0].as_ref().unwrap().pixels,
            &left.pixels
        ));
        assert!(Arc::ptr_eq(
            &frames[1].as_ref().unwrap().pixels,
            &right.pixels
        ));
    }

    #[test]
    fn native_openness_disable_survives_mixed_stream_merge() {
        let mut dst = EyeSample::default();
        merge_eye_sample(
            &mut dst,
            EyeSample {
                openness: 0.8,
                openness_valid: true,
                openness_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.openness_valid);

        // A gaze-only packet must not erase the last wearable state.
        merge_eye_sample(&mut dst, EyeSample::default());
        assert!(dst.openness_valid && dst.openness_reported);

        // A wearable packet carrying Disable must replace the previous Enable.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                openness: 0.0,
                openness_valid: false,
                openness_reported: true,
                ..Default::default()
            },
        );
        assert!(!dst.openness_valid && dst.openness_reported);
    }

    #[test]
    fn transient_invalid_gaze_holds_last_valid_but_aux_packet_does_not_disturb_it() {
        let mut dst = EyeSample::default();
        merge_eye_sample(
            &mut dst,
            EyeSample {
                gaze: [0.2, -0.1, 0.97],
                gaze_valid: true,
                gaze_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid && dst.gaze_reported);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);

        // Wearable 1285 contributes pupil/openness only and must not disturb
        // the canonical 1289 gaze state.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                pupil_mm: 3.4,
                pupil_valid: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);

        // A single 1289 invalid status is only a transient. Freshness aging in
        // the emit thread clears it if no later valid sample arrives.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                gaze_valid: false,
                gaze_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);
    }

    #[test]
    fn reported_invalid_pupil_position_clears_stale_validity() {
        let mut dst = EyeSample::default();
        merge_eye_sample(
            &mut dst,
            EyeSample {
                pupil_pos: [0.25, 0.75],
                pupil_pos_valid: true,
                pupil_pos_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.pupil_pos_valid && dst.pupil_pos_reported);
        assert_eq!(dst.pupil_pos, [0.25, 0.75]);

        merge_eye_sample(
            &mut dst,
            EyeSample {
                pupil_pos: [0.9, 0.1],
                pupil_pos_valid: false,
                pupil_pos_reported: true,
                ..Default::default()
            },
        );
        assert!(!dst.pupil_pos_valid && dst.pupil_pos_reported);
        assert_eq!(
            dst.pupil_pos,
            [0.25, 0.75],
            "reported-invalid data must not replace the last valid coordinates"
        );
    }

    #[test]
    fn gaze_only_packet_does_not_clear_pupil_position() {
        let mut dst = EyeSample {
            pupil_pos: [0.4, 0.6],
            pupil_pos_valid: true,
            pupil_pos_reported: true,
            ..Default::default()
        };
        merge_eye_sample(
            &mut dst,
            EyeSample {
                gaze: [0.1, 0.0, 0.99],
                gaze_valid: true,
                gaze_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.pupil_pos_valid && dst.pupil_pos_reported);
        assert_eq!(dst.pupil_pos, [0.4, 0.6]);
    }

    #[test]
    fn gaze_validity_uses_a_short_freshness_grace() {
        let now = Instant::now();
        assert!(!gaze_is_fresh(None, now));
        assert!(gaze_is_fresh(Some(now), now));
        let recent = now.checked_sub(Duration::from_millis(100)).unwrap();
        let stale = now.checked_sub(Duration::from_millis(200)).unwrap();
        assert!(gaze_is_fresh(Some(recent), now));
        assert!(!gaze_is_fresh(Some(stale), now));
    }

    #[test]
    fn gaze_correction_centres_and_scales_in_angle_space() {
        let mut gaze = [10.0f32.to_radians().sin(), 0.0, 10.0f32.to_radians().cos()];
        apply_gaze_correction(
            &mut gaze,
            Eye::Left,
            GazeCorrection {
                enabled: true,
                offset_x_deg: [-20.0, 0.0],
                scale_x: [2.0, 1.0],
                ..GazeCorrection::default()
            },
        );
        let [yaw, pitch] = gaze_angles_deg(gaze).unwrap();
        assert!(yaw.abs() < 1.0e-3, "yaw={yaw}");
        assert!(pitch.abs() < 1.0e-3, "pitch={pitch}");
    }

    #[test]
    fn gaze_range_expands_the_same_combined_vector_for_both_eyes() {
        let yaw = 8.0f32.to_radians();
        let pitch = 4.0f32.to_radians();
        let source = [
            yaw.sin() * pitch.cos(),
            pitch.sin(),
            yaw.cos() * pitch.cos(),
        ];
        let correction = GazeCorrection {
            enabled: true,
            scale_x: [2.5; 2],
            scale_y: [2.5; 2],
            ..GazeCorrection::default()
        };
        let mut left = source;
        let mut right = source;

        apply_gaze_correction(&mut left, Eye::Left, correction);
        apply_gaze_correction(&mut right, Eye::Right, correction);

        for gaze in [left, right] {
            let [expanded_yaw, expanded_pitch] = gaze_angles_deg(gaze).unwrap();
            assert!((expanded_yaw - 20.0).abs() < 1.0e-3);
            assert!((expanded_pitch - 10.0).abs() < 1.0e-3);
        }
    }

    #[test]
    fn maximum_gaze_adjustment_saturates_forward_without_reversing() {
        let correction = GazeCorrection {
            enabled: true,
            offset_x_deg: [15.0; 2],
            offset_y_deg: [15.0; 2],
            scale_x: [2.5; 2],
            scale_y: [2.5; 2],
            vergence_deg: 10.0,
        };

        for eye in Eye::ALL {
            let mut previous = [f32::NEG_INFINITY; 2];
            for source_angle in -80..=80 {
                let yaw = (source_angle as f32).to_radians();
                let pitch = (source_angle as f32).to_radians();
                let cp = pitch.cos();
                let mut gaze = [yaw.sin() * cp, pitch.sin(), yaw.cos() * cp];
                apply_gaze_correction(&mut gaze, eye, correction);

                assert!(gaze.iter().all(|value| value.is_finite()));
                let norm = gaze.iter().map(|value| value * value).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1.0e-5, "norm={norm} gaze={gaze:?}");
                assert!(
                    gaze[2] > 0.0,
                    "corrected gaze left forward hemisphere: {gaze:?}"
                );

                let angles = gaze_angles_deg(gaze).unwrap();
                for axis in 0..2 {
                    assert!(
                        angles[axis].abs() <= GAZE_OUTPUT_ANGLE_LIMIT_DEG + 1.0e-3,
                        "axis={axis} angle={} gaze={gaze:?}",
                        angles[axis]
                    );
                    assert!(
                        angles[axis] + 1.0e-3 >= previous[axis],
                        "axis={axis} reversed: {} -> {}",
                        previous[axis],
                        angles[axis]
                    );
                }
                previous = angles;
            }
        }
    }

    #[test]
    fn disabled_gaze_adjustment_is_bit_identical() {
        let original = [0.12, -0.07, 0.98];
        let mut gaze = original;
        apply_gaze_correction(
            &mut gaze,
            Eye::Left,
            GazeCorrection {
                enabled: false,
                offset_x_deg: [15.0; 2],
                offset_y_deg: [-15.0; 2],
                scale_x: [2.5; 2],
                scale_y: [2.5; 2],
                vergence_deg: 10.0,
            },
        );
        assert_eq!(gaze.map(f32::to_bits), original.map(f32::to_bits));
    }

    #[test]
    fn gaze_vergence_moves_eyes_in_opposite_directions() {
        let correction = GazeCorrection {
            enabled: true,
            vergence_deg: 4.0,
            ..Default::default()
        };
        let mut left = [0.0, 0.0, 1.0];
        let mut right = [0.0, 0.0, 1.0];
        apply_gaze_correction(&mut left, Eye::Left, correction);
        apply_gaze_correction(&mut right, Eye::Right, correction);
        assert!(gaze_angles_deg(left).unwrap()[0] < 0.0);
        assert!(gaze_angles_deg(right).unwrap()[0] > 0.0);
    }

    #[test]
    fn custom_wide_overrides_gaze_dependent_openness_dips() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.80;
        results[1].openness = 0.91;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].squeeze = 0.25;
        results[1].squeeze = 0.10;
        results[0].wide = 0.7;
        results[1].wide = 0.7;

        apply_custom_wide_pose(&mut results, true, true);

        assert!((results[0].openness - 0.973).abs() < 1.0e-6);
        assert!((results[1].openness - 0.973).abs() < 1.0e-6);
        assert!((results[0].squeeze - 0.075).abs() < 1.0e-6);
        assert!((results[1].squeeze - 0.03).abs() < 1.0e-6);
    }

    #[test]
    fn custom_wide_release_tail_does_not_pull_narrowed_lids_open() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.30;
        results[1].openness = 0.40;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.55;
        results[1].wide = 0.55;

        apply_custom_wide_pose(&mut results, true, true);

        assert!((results[0].openness - 0.30).abs() < 1.0e-6);
        assert!(results[1].openness > 0.40);
        assert!(results[1].openness < 0.42);
    }

    #[test]
    fn wide_tail_does_not_open_partner_during_a_blink() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.0;
        results[0].blink = true;
        results[1].openness = 0.35;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.20;
        results[1].wide = 0.20;

        apply_custom_wide_pose(&mut results, true, true);

        assert_eq!(results[0].openness, 0.0);
        assert_eq!(results[1].openness, 0.35);
    }

    #[test]
    fn custom_wide_still_keeps_open_partner_open_during_a_blink() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.0;
        results[0].blink = true;
        results[1].openness = 0.82;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.0;
        results[1].wide = 0.70;

        apply_custom_wide_pose(&mut results, true, true);

        assert_eq!(results[0].openness, 0.0);
        assert!((results[1].openness - 0.946).abs() < 1.0e-6);
    }

    #[test]
    fn bilateral_wide_never_pulls_open_partner_down() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.20;
        results[1].openness = 0.90;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.0677;
        results[1].wide = 0.0677;

        apply_custom_wide_pose(&mut results, true, true);

        assert!((results[0].openness - 0.20).abs() < 1.0e-6);
        assert!(results[1].openness > 0.90);
    }

    #[test]
    fn blink_reopen_with_held_wide_does_not_drag_open_partner_down() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.088;
        results[1].openness = 1.0;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.5542;
        results[1].wide = 0.5542;

        apply_custom_wide_pose(&mut results, true, true);

        assert!((results[0].openness - 0.088).abs() < 1.0e-6);
        assert_eq!(results[1].openness, 1.0);
    }

    #[test]
    fn independent_wide_never_accidentally_cross_couples_equal_values() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.80;
        results[1].openness = 0.91;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.70;
        results[1].wide = 0.70;

        apply_custom_wide_pose(&mut results, true, false);

        assert!((results[0].openness - 0.94).abs() < 1.0e-6);
        assert!((results[1].openness - 0.973).abs() < 1.0e-6);
    }

    #[test]
    fn custom_wide_does_not_fabricate_valid_openness() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].wide = 0.70;
        results[1].wide = 0.70;

        apply_custom_wide_pose(&mut results, true, true);

        assert!(!results[0].openness_valid);
        assert!(!results[1].openness_valid);
        assert_eq!(results[0].openness, 1.0);
        assert_eq!(results[1].openness, 1.0);
    }

    #[test]
    fn custom_wide_eligibility_has_no_threshold_jump() {
        let below = custom_wide_pose_eligibility(0.3499);
        let edge = custom_wide_pose_eligibility(0.35);
        let above = custom_wide_pose_eligibility(0.3501);

        assert_eq!(below, 0.0);
        assert_eq!(edge, 0.0);
        assert!(above > 0.0);
        assert!(above < 1.0e-5);
    }

    #[test]
    fn eye_wide_master_emits_explicit_zero_and_disables_custom_pose_assist() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.80;
        results[1].openness = 0.91;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].squeeze = 0.25;
        results[1].squeeze = 0.10;
        results[0].wide = 0.70;
        results[1].wide = 0.60;

        let active = apply_eye_wide_output_policy(&mut results, false, [true; 2], true, true);

        assert!(!active);
        assert_eq!([results[0].wide, results[1].wide], [0.0, 0.0]);
        assert_eq!([results[0].openness, results[1].openness], [0.80, 0.91]);
        assert_eq!([results[0].squeeze, results[1].squeeze], [0.25, 0.10]);
    }

    #[test]
    fn eye_wide_master_off_opens_only_the_wide_squeeze_chain() {
        let mut tuning = Tuning {
            wide_squeeze_exclusive: true,
            alpha_open: 0.23,
            squeeze_gain: 0.71,
            ..Tuning::default()
        };
        apply_eye_wide_master_to_tuning(&mut tuning, false);
        assert!(!tuning.wide_squeeze_exclusive);
        assert_eq!(tuning.alpha_open.to_bits(), 0.23_f32.to_bits());
        assert_eq!(tuning.squeeze_gain.to_bits(), 0.71_f32.to_bits());

        tuning.wide_squeeze_exclusive = true;
        apply_eye_wide_master_to_tuning(&mut tuning, true);
        assert!(tuning.wide_squeeze_exclusive);
    }

    #[test]
    fn visual_expression_ranges_map_endpoints_after_provider_selection() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].wide = 0.20;
        results[1].wide = 0.65;
        results[0].squeeze = 0.30;
        results[1].squeeze = 0.80;
        let profile = EyelidResponseProfile {
            wide_start: [0.20, 0.15],
            wide_full: [0.60, 0.65],
            squeeze_start: [0.10, 0.30],
            squeeze_full: [0.50, 0.80],
            ..Default::default()
        };

        let live = apply_expression_response_ranges(&mut results, profile, false);
        assert_eq!(results[0].wide, 0.0);
        assert_eq!(results[1].wide, 1.0);
        assert!((results[0].squeeze - 0.5).abs() < 1e-6);
        assert_eq!(results[1].squeeze, 1.0);
        assert_eq!(live[0].wide_input, 0.20);
        assert_eq!(live[1].squeeze_input, 0.80);
        assert_eq!(live[1].wide_output, 1.0);
    }

    #[test]
    fn brow_lr_sync_averages_only_a_complete_valid_pair() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].brow = 0.8;
        results[1].brow = 0.2;
        results[0].brow_valid = true;
        results[1].brow_valid = true;

        apply_brow_lr_sync(&mut results, true);
        assert!((results[0].brow - 0.5).abs() < 1.0e-6);
        assert!((results[1].brow - 0.5).abs() < 1.0e-6);

        results[0].brow = -0.4;
        results[1].brow = 0.7;
        results[1].brow_valid = false;
        apply_brow_lr_sync(&mut results, true);
        assert_eq!(results[0].brow, -0.4);
        assert_eq!(results[1].brow, 0.7);
    }

    #[test]
    fn expression_exclusivity_uses_range_mapped_values_once() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].wide = 0.50;
        results[0].squeeze = 0.25;
        results[1].wide = 0.25;
        results[1].squeeze = 0.50;

        let live =
            apply_expression_response_ranges(&mut results, EyelidResponseProfile::default(), true);
        assert!((results[0].wide - 0.375).abs() < 1e-6);
        assert!((results[0].squeeze - 0.125).abs() < 1e-6);
        assert!((results[1].wide - 0.125).abs() < 1e-6);
        assert!((results[1].squeeze - 0.375).abs() < 1e-6);
        assert_eq!(live[0].wide_output, results[0].wide);
    }

    #[test]
    fn eye_wide_capability_remains_per_eye_beneath_global_master() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.80;
        results[1].openness = 0.91;
        results[0].openness_valid = true;
        results[1].openness_valid = true;
        results[0].wide = 0.70;
        results[1].wide = 0.60;

        let active = apply_eye_wide_output_policy(&mut results, true, [true, false], true, false);

        assert!(active);
        assert_eq!(results[0].wide, 0.70);
        assert_eq!(results[1].wide, 0.0);
        assert!(
            results[0].openness > 0.80,
            "left custom pose assist remains active"
        );
        assert_eq!(
            results[1].openness, 0.91,
            "disabled eye is not pose-assisted"
        );
    }

    #[test]
    fn endpoint_drain_applies_both_eyes_and_consumes_the_request() {
        let pending = Mutex::new(Some(EndpointApplyRequest {
            eyes: [
                Some(EndpointApply {
                    baseline: 0.73,
                    blink_depth: 0.24,
                    calibrated_unix: 101,
                }),
                Some(EndpointApply {
                    baseline: 0.66,
                    blink_depth: 0.17,
                    calibrated_unix: 202,
                }),
            ],
            reset_to_adaptive: [false; 2],
        }));
        let mut state = SRanipalState::new();

        assert!(drain_endpoint_apply(&pending, &mut state));
        assert!(pending.lock().unwrap().is_none());

        let store = state.snapshot_all();
        assert_eq!(store.left.baseline.to_bits(), 0.73_f32.to_bits());
        assert_eq!(store.left.blink_depth.to_bits(), 0.24_f32.to_bits());
        assert_eq!(store.left.endpoint_calibrated_unix, 101);
        assert!(store.left.endpoint_locked);
        assert_eq!(store.right.baseline.to_bits(), 0.66_f32.to_bits());
        assert_eq!(store.right.blink_depth.to_bits(), 0.17_f32.to_bits());
        assert_eq!(store.right.endpoint_calibrated_unix, 202);
        assert!(store.right.endpoint_locked);
    }

    #[test]
    fn endpoint_drain_is_one_shot_and_preserves_an_unselected_eye() {
        let pending = Mutex::new(Some(EndpointApplyRequest {
            eyes: [
                Some(EndpointApply {
                    baseline: 0.69,
                    blink_depth: 0.21,
                    calibrated_unix: 303,
                }),
                None,
            ],
            reset_to_adaptive: [false; 2],
        }));
        let mut state = SRanipalState::new();
        let before = state.snapshot_all();

        assert!(drain_endpoint_apply(&pending, &mut state));
        let after = state.snapshot_all();
        assert_eq!(after.left.baseline.to_bits(), 0.69_f32.to_bits());
        assert_eq!(after.left.blink_depth.to_bits(), 0.21_f32.to_bits());
        assert_eq!(after.left.endpoint_calibrated_unix, 303);
        assert_eq!(
            after.right.baseline.to_bits(),
            before.right.baseline.to_bits()
        );
        assert_eq!(
            after.right.blink_depth.to_bits(),
            before.right.blink_depth.to_bits()
        );
        assert!(!drain_endpoint_apply(&pending, &mut state));
        let second = state.snapshot_all();
        assert_eq!(
            second.left.baseline.to_bits(),
            after.left.baseline.to_bits()
        );
        assert_eq!(
            second.right.baseline.to_bits(),
            after.right.baseline.to_bits()
        );
    }

    #[test]
    fn endpoint_drain_reset_wins_and_only_resets_the_selected_eye() {
        let mut state = SRanipalState::new();
        state.apply_explicit_endpoint(Eye::Left, 0.71, 0.12, 101);
        state.apply_explicit_endpoint(Eye::Right, 0.58, 0.17, 202);
        let pending = Mutex::new(Some(EndpointApplyRequest {
            eyes: [
                Some(EndpointApply {
                    baseline: 0.80,
                    blink_depth: 0.30,
                    calibrated_unix: 303,
                }),
                None,
            ],
            reset_to_adaptive: [true, false],
        }));

        assert!(drain_endpoint_apply(&pending, &mut state));
        let store = state.snapshot_all();
        assert!(!store.left.endpoint_locked);
        assert_eq!(store.left.endpoint_calibrated_unix, 0);
        assert_eq!(store.left.baseline.to_bits(), 0.60_f32.to_bits());
        assert_eq!(store.left.blink_depth.to_bits(), 0.20_f32.to_bits());
        assert!(store.right.endpoint_locked);
        assert_eq!(store.right.endpoint_calibrated_unix, 202);
    }
}

const GAZE_VALID_GRACE: Duration = Duration::from_millis(150);
/// Keep corrected gaze in the forward hemisphere.  The UI intentionally permits a
/// fairly large range multiplier for users whose native XR5 calibration only spans a
/// few degrees, but multiplying an already-large native angle must saturate rather
/// than cross 90 degrees and make the avatar eye reverse direction.
const GAZE_OUTPUT_ANGLE_LIMIT_DEG: f32 = 85.0;

fn gaze_is_fresh(last_valid: Option<Instant>, now: Instant) -> bool {
    last_valid.is_some_and(|t| now.saturating_duration_since(t) <= GAZE_VALID_GRACE)
}

/// Convert a direction vector to yaw/pitch degrees. `None` rejects the zero/non-finite
/// sentinel so correction can never turn missing gaze into a valid-looking direction.
pub(crate) fn gaze_angles_deg(gaze: [f32; 3]) -> Option<[f32; 2]> {
    GazeEyelidProfile::gaze_angles_deg(gaze)
}

/// Apply angular centre/range/vergence correction while preserving a normalized 3-D
/// direction. This operates after per-device handedness mapping and before every sink.
pub(crate) fn apply_gaze_correction(gaze: &mut [f32; 3], eye: Eye, correction: GazeCorrection) {
    if !correction.enabled {
        return;
    }
    let Some([yaw_deg, pitch_deg]) = gaze_angles_deg(*gaze) else {
        return;
    };
    let i = eye.idx();
    let sx = if correction.scale_x[i].is_finite() {
        correction.scale_x[i].clamp(0.25, 2.5)
    } else {
        1.0
    };
    let sy = if correction.scale_y[i].is_finite() {
        correction.scale_y[i].clamp(0.25, 2.5)
    } else {
        1.0
    };
    let ox = if correction.offset_x_deg[i].is_finite() {
        correction.offset_x_deg[i].clamp(-30.0, 30.0)
    } else {
        0.0
    };
    let oy = if correction.offset_y_deg[i].is_finite() {
        correction.offset_y_deg[i].clamp(-30.0, 30.0)
    } else {
        0.0
    };
    let vergence = if correction.vergence_deg.is_finite() {
        correction.vergence_deg.clamp(-20.0, 20.0)
    } else {
        0.0
    };
    let eye_sign = if eye == Eye::Left { -0.5 } else { 0.5 };
    let yaw = (yaw_deg * sx + ox + vergence * eye_sign)
        .clamp(-GAZE_OUTPUT_ANGLE_LIMIT_DEG, GAZE_OUTPUT_ANGLE_LIMIT_DEG)
        .to_radians();
    let pitch = (pitch_deg * sy + oy)
        .clamp(-GAZE_OUTPUT_ANGLE_LIMIT_DEG, GAZE_OUTPUT_ANGLE_LIMIT_DEG)
        .to_radians();
    let cp = pitch.cos();
    *gaze = [yaw.sin() * cp, pitch.sin(), yaw.cos() * cp];
}

impl Pipeline {
    /// Start the adapter and the ML + emit threads. `net` is `None` to run
    /// without ML (gaze still flows; openness held at the neutral 0.5).
    pub fn run(
        adapter: Box<dyn HmdAdapter>,
        net: Option<EyeNet>,
        brow: Option<BrowNet>,
        wide: Option<WideNet>,
        sinks: Vec<Box<dyn OutputSink>>,
        map: DeviceMap,
        device_status: Arc<Mutex<String>>,
        device_key: String,
        init: PipelineInit,
    ) -> io::Result<Pipeline> {
        #[cfg(windows)]
        {
            Self::run_inner(
                adapter,
                net,
                brow,
                wide,
                sinks,
                map,
                device_status,
                device_key,
                init,
                None,
            )
        }
        #[cfg(not(windows))]
        {
            Self::run_inner(
                adapter,
                net,
                brow,
                wide,
                sinks,
                map,
                device_status,
                device_key,
                init,
            )
        }
    }

    #[cfg(windows)]
    pub(crate) fn run_with_gpu_context(
        adapter: Box<dyn HmdAdapter>,
        net: Option<EyeNet>,
        brow: Option<BrowNet>,
        wide: Option<WideNet>,
        sinks: Vec<Box<dyn OutputSink>>,
        map: DeviceMap,
        device_status: Arc<Mutex<String>>,
        device_key: String,
        init: PipelineInit,
        gpu_context: Option<EyelidGpuContext>,
    ) -> io::Result<Pipeline> {
        Self::run_inner(
            adapter,
            net,
            brow,
            wide,
            sinks,
            map,
            device_status,
            device_key,
            init,
            gpu_context,
        )
    }

    fn run_inner(
        mut adapter: Box<dyn HmdAdapter>,
        net: Option<EyeNet>,
        brow: Option<BrowNet>,
        mut wide: Option<WideNet>,
        mut sinks: Vec<Box<dyn OutputSink>>,
        map: DeviceMap,
        device_status: Arc<Mutex<String>>,
        device_key: String,
        mut init: PipelineInit,
        #[cfg(windows)] gpu_context: Option<EyelidGpuContext>,
    ) -> io::Result<Pipeline> {
        // Defense in depth: Pipeline is public and the diagnostic CLI calls it directly.
        // Even a stale/global config or future caller must never run XR5 image-Wide on
        // another HMD, nor a Pimax gaze trim on StarVR, Varjo, or VPE.
        let is_xr5 = crate::config::canonical_device_key(&device_key) == "pimax_xr5";
        if !is_xr5 {
            if wide.is_some() {
                eprintln!(
                    "[wide] ignored XR5 custom EyeWide model for non-XR5 device {device_key}"
                );
            }
            wide = None;
            init.wide_source = WideSource::Sranipal;
            init.wide_enabled = [true; 2];
        }
        if !crate::config::supports_gaze_correction(&device_key) {
            init.gaze_correction = GazeCorrection::default();
        }
        if !crate::config::supports_photometric_fit(&device_key) {
            init.photometric_correction = PhotometricCorrection::default();
        }

        eprintln!(
            "[mapping] device={device_key} eye_stream_swap={} camera_flip={} gaze_x_flip={}",
            map.swap_eyes, map.flip_image, map.flip_gaze_x
        );
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let recenter = Arc::new(AtomicBool::new(false));
        let wide_recenter = Arc::new(AtomicBool::new(false));
        let guided_calibration = Arc::new(Mutex::new(None));
        let endpoint_apply = Arc::new(Mutex::new(None::<EndpointApplyRequest>));
        let gaze_eyelid_apply = Arc::new(Mutex::new(None::<GazeEyelidApplyRequest>));
        let wink_apply = Arc::new(Mutex::new(None::<WinkApplyRequest>));
        let blink_timing_apply = Arc::new(Mutex::new(None::<BlinkTimingApplyRequest>));
        let diag_rec = Arc::new(AtomicBool::new(false));
        let eye_head_comparison_enabled = Arc::new(AtomicBool::new(false));
        let right_eye_left_head = Arc::new(AtomicBool::new(init.right_eye_left_head));
        let recenter_eye_mask = Arc::new(AtomicU8::new(0));
        let swap_eyes = Arc::new(AtomicBool::new(map.swap_eyes));
        let flip_image = Arc::new(AtomicBool::new(map.flip_image));
        let ml_mirror_l = Arc::new(AtomicBool::new(init.ml_mirror[0]));
        let ml_mirror_r = Arc::new(AtomicBool::new(init.ml_mirror[1]));
        let flip_gaze_x = Arc::new(AtomicBool::new(map.flip_gaze_x));
        let gaze_correction = Arc::new(Mutex::new(init.gaze_correction));
        let wide_enabled = Arc::new(Mutex::new(init.wide_enabled));
        let wide_source = Arc::new(Mutex::new(init.wide_source));
        let eye_wide_enabled = Arc::new(AtomicBool::new(init.eye_wide_enabled));
        let eyebrow_enabled = Arc::new(AtomicBool::new(init.eyebrow_enabled));
        let tuning = Arc::new(Mutex::new(init.tuning));
        let eyelid_response = Arc::new(Mutex::new(init.eyelid_response_profile.sanitized()));
        let wearing_calibration_target = Arc::new(Mutex::new(None::<WearingCalibrationTarget>));
        let wearing_response_reset = Arc::new(AtomicBool::new(false));
        let manual_endpoint_edit = Arc::new(Mutex::new(None::<([f32; 2], EyelidResponseProfile)>));
        let geometry = Arc::new(Mutex::new(init.geometry));
        let despeckle = Arc::new(Mutex::new(init.despeckle));
        let flatten = Arc::new(Mutex::new(init.flatten));
        let brightness = Arc::new(Mutex::new(init.brightness));
        let photometric_correction = Arc::new(Mutex::new(init.photometric_correction));
        let bright_affine = Arc::new(Mutex::new(BrightAffineSnapshot::initial()));
        let heatmap = Arc::new(HeatState::new());
        let ml_loaded = net.is_some();
        let eyelid_model_identity = net
            .is_some()
            .then_some(init.eyelid_model_identity)
            .flatten();
        let (net, eyelid_backend_report) = match net {
            Some(net) => {
                #[cfg(windows)]
                let (model, report) =
                    build_eyelid_model(net, init.eyelid_inference_backend, gpu_context);
                #[cfg(not(windows))]
                let (model, report) = build_eyelid_model(net, init.eyelid_inference_backend);
                eprintln!(
                    "[ml] backend={} adapter={} cpu_pair_ms={} gpu_pair_ms={} ({})",
                    report.active,
                    report.adapter.as_deref().unwrap_or("n/a"),
                    report
                        .cpu_pair_ms
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "n/a".into()),
                    report
                        .gpu_pair_ms
                        .map(|value| format!("{value:.3}"))
                        .unwrap_or_else(|| "n/a".into()),
                    report.note
                );
                (Some(model), report)
            }
            None => (
                None,
                EyelidBackendReport {
                    active: "Unavailable",
                    runtime_gpu_active: None,
                    adapter: None,
                    cpu_pair_ms: None,
                    gpu_pair_ms: None,
                    note: "No eyelid model is loaded".into(),
                },
            ),
        };
        let brow_loaded = brow.is_some();
        let wide_loaded = wide.is_some();
        // Brow lives behind a shared handle so a freshly trained model can be hot-swapped
        // into the event-driven brow worker (see `set_brow`) without a device reconnect.
        let brow = Arc::new(Mutex::new(brow));
        let wide = Arc::new(Mutex::new(wide));
        let tele = Telemetry::new(ml_loaded, brow_loaded, wide_loaded, adapter.profile());

        // A one-slot notification coalesces camera callbacks while the brow worker is
        // busy. The worker snapshots the newest stereo pair and additionally checks the
        // generations, so duplicate callbacks never re-run inference on identical pixels.
        let (brow_frame_tx, brow_frame_rx) = mpsc::sync_channel::<()>(1);

        // Adapter callbacks: apply per-unit mapping, then stash newest frame per eye.
        let t_cb = tele.clone();
        let (cb_swap, cb_flip) = (swap_eyes.clone(), flip_image.clone());
        let on_frame = Box::new(move |eye: Eye, w: u32, h: u32, px: &[u8]| {
            let eye = if cb_swap.load(Ordering::Relaxed) {
                eye.opposite()
            } else {
                eye
            };
            let sz = (w as usize) * (h as usize);
            // Mirror per the ACTUAL frame width (not a hardcoded 200) so the flip is
            // correct at any resolution.
            let stored = if cb_flip.load(Ordering::Relaxed) && px.len() >= sz {
                mirror_h(&px[..sz], w as usize, h as usize)
            } else {
                px.to_vec()
            };
            let counter = match eye {
                Eye::Left => &t_cb.c_frame_l,
                Eye::Right => &t_cb.c_frame_r,
            };
            let mut published = false;
            if let Ok(mut frames) = t_cb.frames.lock() {
                // Publish the exact bytes and their generation together. The atomics
                // remain the public rate counters, but capture code reads this field.
                let generation = counter.fetch_add(1, Ordering::Relaxed) + 1;
                frames[eye.idx()] = Some(EyeFrame {
                    generation,
                    width: w,
                    height: h,
                    pixels: Arc::from(stored),
                });
                published = true;
            }
            if published {
                // Never block the camera callback. A full queue already means the worker
                // has been told that newer pixels exist and will snapshot the latest pair.
                let _ = brow_frame_tx.try_send(());
            }
        });
        let t_g = tele.clone();
        let eye_stream_swap = swap_eyes.clone();
        let on_gaze = Box::new(move |s: GazeSample| {
            // A reversed Pimax unit swaps the complete L/R streams. Route image,
            // auxiliary data and gaze identity from one source of truth.
            let s = route_eye_sample(s, eye_stream_swap.load(Ordering::Relaxed));
            let valid = [
                s.left.gaze_reported && s.left.gaze_valid,
                s.right.gaze_reported && s.right.gaze_valid,
            ];
            let pupil_pos_reported = [s.left.pupil_pos_reported, s.right.pupil_pos_reported];
            let openness_reported = [s.left.openness_reported, s.right.openness_reported];
            let native_timestamp_reported = s.timestamp_us != 0;
            if let Ok(mut g) = t_g.gaze.lock() {
                merge_gaze_sample(&mut g, s);
                if valid[0] || valid[1] {
                    if let Ok(mut last) = t_g.gaze_last_valid.lock() {
                        let now = Instant::now();
                        for i in 0..2 {
                            if valid[i] {
                                last[i] = Some(now);
                            }
                        }
                    }
                }
                if pupil_pos_reported.iter().any(|reported| *reported) {
                    if let Ok(mut last) = t_g.pupil_pos_last_reported.lock() {
                        let now = Instant::now();
                        for eye in 0..2 {
                            if pupil_pos_reported[eye] {
                                last[eye] = Some(now);
                            }
                        }
                    }
                }
                if openness_reported.iter().any(|reported| *reported) {
                    if let Ok(mut last) = t_g.openness_last_reported.lock() {
                        let now = Instant::now();
                        for eye in 0..2 {
                            if openness_reported[eye] {
                                last[eye] = Some(now);
                            }
                        }
                    }
                }
                if native_timestamp_reported {
                    if let Ok(mut last) = t_g.native_timestamp_last_reported.lock() {
                        *last = Some(Instant::now());
                    }
                }
                if let Ok(mut pu) = t_g.pupil.lock() {
                    *pu = [
                        (g.left.pupil_mm, g.left.pupil_valid),
                        (g.right.pupil_mm, g.right.pupil_valid),
                    ];
                }
            }
            t_g.c_gaze.fetch_add(1, Ordering::Relaxed);
        });
        adapter.start(on_frame, on_gaze)?;

        let mut threads = Vec::new();

        // Eyebrow worker: exactly one inference per newly advanced stereo camera pair.
        //
        // This deliberately does not share the eyelid thread's 16 ms polling cadence.
        // The old standalone tracker only inferred when its frame object changed; doing
        // the same here avoids duplicate inference and lets a 120 Hz camera drive brow at
        // its natural rate (subject to inference cost). The bounded notification channel
        // coalesces backlog instead of accumulating frames or memory.
        {
            let t_brow = tele.clone();
            let brow_stop = stop.clone();
            let brow_on = eyebrow_enabled.clone();
            let brow = brow.clone();
            threads.push(thread::spawn(move || {
                const BN: usize = preprocess::BROW_SIDE * preprocess::BROW_SIDE;
                let mut input = [0.0f32; BN];
                let mut last_generation = [0u64; 2];

                while !brow_stop.load(Ordering::Relaxed) {
                    match brow_frame_rx.recv_timeout(Duration::from_millis(100)) {
                        Ok(()) => {}
                        Err(mpsc::RecvTimeoutError::Timeout) => continue,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    if !brow_on.load(Ordering::Relaxed)
                        || !t_brow.brow_loaded.load(Ordering::Relaxed)
                    {
                        continue;
                    }

                    let (left, right) = {
                        let frames = t_brow.frames.lock().unwrap();
                        (frames[0].clone(), frames[1].clone())
                    };
                    let (Some(left), Some(right)) = (left, right) else {
                        continue;
                    };
                    let generation = [left.generation, right.generation];
                    if !stereo_pair_advanced(generation, last_generation) {
                        continue;
                    }

                    let (lw, lh, lp) = left.view();
                    let (rw, rh, rp) = right.view();
                    if lp.len() < (lw as usize) * (lh as usize)
                        || rp.len() < (rw as usize) * (rh as usize)
                    {
                        last_generation = generation;
                        continue;
                    }

                    preprocess::brow_input(lp, lw as usize, lh as usize, false, &mut input);
                    let left_input = input;
                    preprocess::brow_input(rp, rw as usize, rh as usize, true, &mut input);

                    // Keep the model lock only around the two forward calls. If no model
                    // is currently loaded, leave the generations unconsumed so a hot-load
                    // can use the next camera notification immediately.
                    let prediction = if let Ok(mut guard) = brow.lock() {
                        guard.as_mut().map(|model| {
                            [
                                model.forward_one(&left_input)[0],
                                model.forward_one(&input)[0],
                            ]
                        })
                    } else {
                        None
                    };
                    let Some([bl, br]) = prediction else {
                        continue;
                    };
                    last_generation = generation;
                    if bl.is_finite() && br.is_finite() {
                        if let Ok(mut raw) = t_brow.brow_raw.lock() {
                            *raw = [bl, br];
                        }
                        // One increment now represents one genuinely new stereo pair,
                        // not another pass over the same camera image.
                        t_brow.c_brow.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }));
        }

        // ML thread @ ~60Hz: newest stereo pair -> eye openness/squeeze + optional Wide.
        // ALWAYS spawned so `set_wide` can hot-load a freshly trained model into a running
        // pipeline with no reconnect.
        {
            let (t_ml, ms) = (tele.clone(), stop.clone());
            let (mml, mmr) = (ml_mirror_l.clone(), ml_mirror_r.clone());
            let head_ab_enabled = eye_head_comparison_enabled.clone();
            let right_left_head = right_eye_left_head.clone();
            let mut net = net;
            let wide = wide.clone();
            let mgeo = geometry.clone();
            let mdsp = despeckle.clone();
            let mflt = flatten.clone();
            let mbn = brightness.clone();
            let mphoto = photometric_correction.clone();
            let maff = bright_affine.clone();
            let hm = heatmap.clone();
            threads.push(thread::spawn(move || {
                const WN: usize = preprocess::WIDE_SIDE * preprocess::WIDE_SIDE;
                let period = Duration::from_millis(16);
                let mut win = [0.0f32; WN];
                let mut model_error_reported = false;
                let mut last_calibration_generation = [0u64; 2];
                let mut head_ab_input = vec![0.0f32; EYELID_INPUT_LEN];
                let mut last_head_ab = Instant::now() - Duration::from_secs(1);
                let mut shadow_was_enabled = false;
                while !ms.load(Ordering::Relaxed) {
                    let (l, r) = {
                        let f = t_ml.frames.lock().unwrap();
                        (f[0].clone(), f[1].clone())
                    };
                    if let (Some(l), Some(r)) = (l, r) {
                        let source_generation = [l.generation, r.generation];
                        let source_frames = [Some(l.clone()), Some(r.clone())];
                        let (lw, lh, l) = l.view();
                        let (rw, rh, r) = r.view();
                        if l.len() >= (lw as usize) * (lh as usize)
                            && r.len() >= (rw as usize) * (rh as usize)
                        {
                            // Eye net: ONE pass over both eyes (L in ch0, R in ch1), each
                            // resized to 100x100. Outputs (RE'd 2026-06-26): s0=presence,
                            // s1=L openness, s2=R openness, s3=L squeeze, s4=R squeeze.
                            // Keep camera-only tools (reseat and eyebrow collection)
                            // functional even when no eyelid weights are configured.
                            // Without EyeNet there is no adaptive affine to describe,
                            // so the coherent raw pair is published with identity.
                            if net.is_none()
                                && source_generation[0] > last_calibration_generation[0]
                                && source_generation[1] > last_calibration_generation[1]
                            {
                                t_ml.publish_calibration_frame(CalibrationFrameSnapshot {
                                    captured_at: Instant::now(),
                                    source_generation,
                                    affine: [[1.0, 0.0]; 2],
                                    frames: [
                                        source_frames[0]
                                            .as_ref()
                                            .expect("stereo source contains left eye")
                                            .clone(),
                                        source_frames[1]
                                            .as_ref()
                                            .expect("stereo source contains right eye")
                                            .clone(),
                                    ],
                                    gaze: t_ml.fresh_capture_sample(),
                                });
                                last_calibration_generation = source_generation;
                            }
                            if let Some(net) = net.as_mut() {
                                let geom = *mgeo.lock().unwrap();
                                // Suppress specular spots (glasses / IR glints) BEFORE the
                                // model — the heatmaps showed the net reads brightness as
                                // "more open", so a reflection inflates/destabilizes openness.
                                let dsp = *mdsp.lock().unwrap();
                                let lf = preprocess::despeckle(&l, lw as usize, lh as usize, &dsp);
                                let rf = preprocess::despeckle(&r, rw as usize, rh as usize, &dsp);
                                // Illumination flatten (close-up shadow removal), after despeckle.
                                let flt = *mflt.lock().unwrap();
                                let lf = preprocess::flatten(&lf, lw as usize, lh as usize, &flt);
                                let rf = preprocess::flatten(&rf, rw as usize, rh as usize, &flt);
                                // Fixed manual brightness only. The old adaptive normalizer
                                // intentionally stays out of the live path: it could chase a
                                // changed wearing position and move eyelid behaviour over time.
                                let aff = {
                                    let norm = mbn.lock().unwrap();
                                    let mut affine = [(1.0, 0.0); 2];
                                    crate::ml::brightness::compose_manual_gain(
                                        &mut affine,
                                        norm.manual_gain,
                                    );
                                    affine
                                };
                                if let Ok(mut a) = maff.lock() {
                                    *a = BrightAffineSnapshot {
                                        source_generation,
                                        affine: [[aff[0].0, aff[0].1], [aff[1].0, aff[1].1]],
                                        source_frames: source_frames.clone(),
                                    };
                                }
                                // Calibration consumers run on the UI thread, whose
                                // presentation rate may be below 60 Hz. Publish every
                                // coherent EyeNet source pair to a bounded Arc-backed
                                // history before doing the more expensive forward pass.
                                if source_generation[0] > last_calibration_generation[0]
                                    && source_generation[1] > last_calibration_generation[1]
                                {
                                    t_ml.publish_calibration_frame(CalibrationFrameSnapshot {
                                        captured_at: Instant::now(),
                                        source_generation,
                                        affine: [
                                            [aff[0].0, aff[0].1],
                                            [aff[1].0, aff[1].1],
                                        ],
                                        frames: [
                                            source_frames[0]
                                                .as_ref()
                                                .expect("stereo source contains left eye")
                                                .clone(),
                                            source_frames[1]
                                                .as_ref()
                                                .expect("stereo source contains right eye")
                                                .clone(),
                                        ],
                                        gaze: t_ml.fresh_capture_sample(),
                                    });
                                    last_calibration_generation = source_generation;
                                }
                                let nlf = crate::ml::brightness::apply(&lf, aff[0].0, aff[0].1);
                                let nrf = crate::ml::brightness::apply(&rf, aff[1].0, aff[1].1);
                                // Fixed fit discovered from labelled train evidence and
                                // accepted only on untouched holdout. Geometry remains
                                // unchanged for frontal VR4/Varjo cameras.
                                let photo = *mphoto.lock().unwrap();
                                let (nlf, nrf) = if photo.enabled {
                                    (
                                        preprocess::fitted_photometric(
                                            &nlf,
                                            lw as usize,
                                            lh as usize,
                                            &geom[0],
                                            &photo,
                                            0,
                                        ),
                                        preprocess::fitted_photometric(
                                            &nrf,
                                            rw as usize,
                                            rh as usize,
                                            &geom[1],
                                            &photo,
                                            1,
                                        ),
                                    )
                                } else {
                                    // Preserve the old path and allocation count exactly
                                    // until a holdout-validated correction is enabled.
                                    (nlf, nrf)
                                };
                                let mirr =
                                    [mml.load(Ordering::Relaxed), mmr.load(Ordering::Relaxed)];
                                let input = preprocess::to_input_stereo_geom(
                                    &nlf, lw, lh, &nrf, rw, rh, mirr[0], mirr[1], &geom[0],
                                    &geom[1],
                                );
                                // Publish the exact net input for the dashboard's NET
                                // view, un-mirrored back to natural orientation.
                                if let Ok(mut mi) = t_ml.ml_input.lock() {
                                    let n = preprocess::DST;
                                    for e in 0..2 {
                                        let sl = &input[e * n * n..(e + 1) * n * n];
                                        let mut px = vec![0u8; n * n];
                                        for y in 0..n {
                                            for x in 0..n {
                                                let sx = if mirr[e] { n - 1 - x } else { x };
                                                px[y * n + x] = (sl[y * n + sx] * 255.0)
                                                    .clamp(0.0, 255.0)
                                                    as u8;
                                            }
                                        }
                                        mi[e] = Some(px);
                                    }
                                }
                                // Preprocessing is defined to produce exactly CHW [2, 100, 100].
                                // The old EyeNet path also relied on this invariant and would fail
                                // rather than silently retaining a stale sample if it were broken.
                                debug_assert_eq!(input.len(), EYELID_INPUT_LEN);
                                let head_ab_now = head_ab_enabled.load(Ordering::Relaxed);
                                let production_right_left =
                                    right_left_head.load(Ordering::Relaxed);
                                let shadow_enabled = head_ab_now || production_right_left;
                                let shadow_due = production_right_left
                                    || last_head_ab.elapsed() >= Duration::from_millis(50);
                                let mut publish = None;
                                let mut current_comparison = None;
                                let mut inference_pending = false;
                                let pair_ready = shadow_enabled
                                    && shadow_due
                                    && mirror_swap_canonical_stereo(&input, &mut head_ab_input);
                                if pair_ready {
                                    last_head_ab = Instant::now();
                                    let canonical = CanonicalStereoInput::try_from(input.as_slice())
                                        .expect(
                                            "eyelid preprocessing must produce CHW [2, 100, 100]",
                                        );
                                    let shadow = CanonicalStereoInput::try_from(
                                        head_ab_input.as_slice(),
                                    )
                                    .expect("head A/B input must preserve CHW [2, 100, 100]");
                                    match net.infer_pair_live(canonical, shadow) {
                                        Ok(Some([normal_prediction, shadow_prediction])) => {
                                            publish =
                                                normal_prediction.require_legacy_frame().ok();
                                            current_comparison = match (
                                                publish,
                                                shadow_prediction.require_legacy_frame().ok(),
                                            ) {
                                                (Some(normal), Some(shadow)) => Some(
                                                    physical_eye_head_comparison(
                                                        normal.ml5, shadow.ml5,
                                                    ),
                                                ),
                                                _ => None,
                                            };
                                        }
                                        Ok(None) => inference_pending = true,
                                        Err(_) => {}
                                    }
                                    if let Ok(mut slot) = t_ml.eye_head_comparison.lock() {
                                        *slot = current_comparison;
                                    }
                                } else {
                                    let canonical = CanonicalStereoInput::try_from(input.as_slice())
                                        .expect(
                                            "eyelid preprocessing must produce CHW [2, 100, 100]",
                                        );
                                    publish = match net.infer_live(canonical) {
                                        Ok(Some(prediction)) => {
                                            prediction.require_legacy_frame().ok()
                                        }
                                        Ok(None) => {
                                            inference_pending = true;
                                            None
                                        }
                                        Err(_) => None,
                                    };
                                    if !shadow_enabled && shadow_was_enabled {
                                        if let Ok(mut slot) = t_ml.eye_head_comparison.lock() {
                                            *slot = None;
                                        }
                                    }
                                }
                                shadow_was_enabled = shadow_enabled;
                                if production_right_left {
                                    if let (Some(frame), Some(comparison)) =
                                        (publish.as_mut(), current_comparison)
                                    {
                                        apply_right_left_head(frame, comparison);
                                    }
                                }
                                if let Some(frame) = publish {
                                    if let Ok(mut o) = t_ml.ml_raw.lock() {
                                        *o = frame.ml_raw;
                                    }
                                    if let Ok(mut o5) = t_ml.ml5.lock() {
                                        *o5 = frame.ml5;
                                    }
                                    t_ml.c_ml.fetch_add(1, Ordering::Relaxed);
                                } else if !inference_pending && !model_error_reported {
                                    // Unreachable for the only Phase 1 backend. Do not invent
                                    // zeroes for a missing capability or failed inference.
                                    eprintln!(
                                        "[ml] eyelid backend did not provide the required legacy frame; retaining the previous eyelid sample"
                                    );
                                    model_error_reported = true;
                                }
                                // On-demand occlusion heatmap: reuse this frame's input +
                                // net. Blocks the ML loop ~2s (openness freezes) — a manual
                                // one-shot diagnostic, so that's acceptable.
                                if hm.req.swap(false, Ordering::Relaxed) {
                                    hm.computing.store(true, Ordering::Relaxed);
                                    let mode = crate::ml::heatmap::HeatMode::from_u8(
                                        hm.mode.load(Ordering::Relaxed),
                                    );
                                    let res = crate::ml::heatmap::compute_model(
                                        net.as_mut(),
                                        &input,
                                        mode,
                                    );
                                    if let (Some(res), Ok(mut r)) = (res, hm.result.lock()) {
                                        *r = Some(res);
                                    }
                                    hm.computing.store(false, Ordering::Relaxed);
                                }
                            }
                            // Custom Dream Air/XR5 EyeWide model. It shares the TinyEyeNet
                            // runtime with brow but uses a full-eye crop and its own task-tagged
                            // weights. The legacy EyeWide path continues in parallel for A/B.
                            if let Ok(mut guard) = wide.lock() {
                                if let Some(wide) = guard.as_mut() {
                                    preprocess::wide_input(
                                        &l,
                                        lw as usize,
                                        lh as usize,
                                        false,
                                        &mut win,
                                    );
                                    let wl = wide.forward_one(&win);
                                    preprocess::wide_input(
                                        &r,
                                        rw as usize,
                                        rh as usize,
                                        true,
                                        &mut win,
                                    );
                                    let wr = wide.forward_one(&win);
                                    if wl.is_finite() && wr.is_finite() {
                                        if let Ok(mut raw) = t_ml.wide_raw.lock() {
                                            *raw = [wl, wr];
                                        }
                                        t_ml.c_wide.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                    thread::sleep(period);
                }
            }));
        }

        // Emit thread @ ~120Hz: post-process -> telemetry + sinks.
        let calib_path = crate::config::calib_path_for(&device_key)
            .to_string_lossy()
            .into_owned();
        let (t_em, es, ep, er) = (tele.clone(), stop.clone(), paused.clone(), recenter.clone());
        let eye_recenter = recenter_eye_mask.clone();
        let edr = diag_rec.clone();
        let t_tune = tuning.clone();
        let response_em = eyelid_response.clone();
        let wearing_target_em = wearing_calibration_target.clone();
        let wearing_response_reset_em = wearing_response_reset.clone();
        let manual_endpoint_edit_em = manual_endpoint_edit.clone();
        let fgx = flip_gaze_x.clone();
        let gaze_trim = gaze_correction.clone();
        let guided_seed = guided_calibration.clone();
        let endpoint_seed = endpoint_apply.clone();
        let gaze_eyelid_seed = gaze_eyelid_apply.clone();
        let wink_seed = wink_apply.clone();
        let blink_timing_seed = blink_timing_apply.clone();
        let wide_on = wide_enabled.clone();
        let wide_master = eye_wide_enabled.clone();
        let wide_reset = wide_recenter.clone();
        let brow_on = eyebrow_enabled.clone();
        let live_wide_source = wide_source.clone();
        threads.push(thread::spawn(move || {
            let mut state = SRanipalState::new();
            // EyeWide uses the otherwise-discarded right EyeNet head for BOTH physical
            // eyes. Keep its baseline/state independent from the ch1 eyelid state: the
            // two heads have visibly different scales and only ch1 remains authoritative
            // for openness, blinks and calibration persistence.
            let mut ch2_wide_state = SRanipalState::new();
            let mut active_gaze_eyelid = init.gaze_eyelid_profile;
            let mut active_wink = init.wink_profile;
            let mut active_blink_timing = init.blink_timing_profile;
            let mut recalled_response: Option<EyelidResponseProfile> = None;
            let mut brow_state = BrowState::default();
            let mut custom_wide_state = WideState::default();
            let mut last_brow_gen = 0u64;
            let mut last_brow_infer = Instant::now();
            let mut last_wide_gen = 0u64;
            let mut last_wide_infer = Instant::now();
            let mut wide_fresh_at = Instant::now() - Duration::from_secs(10);
            let period = Duration::from_micros(8333);
            let mut last = std::time::Instant::now();
            // Persisted PER-HMD calibration: load this camera/model pair's baseline,
            // blink depth, and mid-close anchor, then re-save periodically and on stop.
            // Never import the ambiguous legacy shared file into a new device bucket.
            // A disconnected/gaze-only recovery pipeline has no real openness evidence;
            // it must never overwrite a valid calibration with the neutral placeholder.
            let persist_calibration = t_em.ml_loaded;
            if persist_calibration {
                if let Some(store) = crate::core::eye_state::load_calib(&calib_path) {
                    state.restore_all(&store);
                }
            }
            let mut since_save = 0u32;
            // Diagnostic CSV recorder state (REC button; see Pipeline::diag_rec).
            let mut diag_file: Option<std::io::BufWriter<std::fs::File>> = None;
            let mut diag_t0 = std::time::Instant::now();
            while !es.load(Ordering::Relaxed) {
                // A cycle that took >=2x the target period means we missed one or
                // more 120Hz slots — count the shortfall as real dropped frames.
                let elapsed = last.elapsed();
                last = std::time::Instant::now();
                let missed = (elapsed.as_micros() / 8333).saturating_sub(1);
                if missed > 0 {
                    t_em.c_drop.fetch_add(missed as u64, Ordering::Relaxed);
                }
                // Snapshot the master once so SRanipal post-processing and the final
                // output policy agree for this whole frame.  With EyeWide output off,
                // Wide must not attenuate squeeze through the optional chain before we
                // zero the final Wide value.
                let wide_master_enabled = wide_master.load(Ordering::Relaxed);
                state.tuning = *t_tune.lock().unwrap();
                if let Some((baseline, profile)) = manual_endpoint_edit_em.lock().unwrap().take() {
                    state.adopt_appearance_baseline(baseline);
                    ch2_wide_state.adopt_current_appearance();
                    *response_em.lock().unwrap() = profile;
                    *wearing_target_em.lock().unwrap() = None;
                    recalled_response = None;
                }
                let base_response = response_em
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .sanitized();
                let clear_wearing_correction =
                    wearing_response_reset_em.swap(false, Ordering::AcqRel);
                if clear_wearing_correction {
                    recalled_response = None;
                }
                let wearing_target = *wearing_target_em
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let mut response = if let Some(target) = wearing_target {
                    let current = recalled_response.unwrap_or(base_response);
                    let mut endpoint_target = base_response;
                    endpoint_target.manual_range = target.manual_range;
                    endpoint_target.open_point_offset = target.open_point_offset;
                    endpoint_target.closed_point_depth = target.closed_point_depth;
                    let blended =
                        blend_response_profile(current, endpoint_target.sanitized(), 0.012);
                    recalled_response = Some(blended);
                    blended
                } else if let Some(current) = recalled_response {
                    // Losing or replacing a visual match must not make the avatar's
                    // response jump. Return to the user's base sliders with the same
                    // gentle slew used when a remembered profile is selected.
                    let blended = blend_response_profile(current, base_response, 0.012);
                    if response_profiles_near(blended, base_response) {
                        recalled_response = None;
                        base_response
                    } else {
                        recalled_response = Some(blended);
                        blended
                    }
                } else {
                    base_response
                };
                // Only recovered endpoints slew. Unrelated controls remain live.
                response.wide_start = base_response.wide_start;
                response.wide_full = base_response.wide_full;
                response.squeeze_start = base_response.squeeze_start;
                response.squeeze_full = base_response.squeeze_full;
                response.curve_mid_output = base_response.curve_mid_output;
                response.close_depth_scale = base_response.close_depth_scale;
                response.blink_close_ms = base_response.blink_close_ms;
                response.snap_gate_open = base_response.snap_gate_open;
                if clear_wearing_correction {
                    // Captures and direct slider edits must observe the uncorrected
                    // coordinate system, not a slowly fading remembered state.
                    state.clear_appearance_profile();
                    ch2_wide_state.clear_appearance_profile();
                } else {
                    state.update_appearance_profile(
                        wearing_target.map(|target| target.baseline),
                        None,
                    );
                    ch2_wide_state.update_appearance_profile(
                        wearing_target.map(|target| target.wide_baseline),
                        wearing_target.map(|target| target.wide_entry_ref),
                    );
                }
                state.set_eyelid_response_tuning(
                    response.manual_range,
                    response.open_point_offset,
                    response.closed_point_depth,
                    response.close_depth_scale,
                    response.curve_mid_output,
                    response.blink_close_ms,
                    response.snap_gate_open,
                    // Wearing Memory supersedes the older baseline-only auto-reseat.
                    // Keeping both active would let two recovery systems fight over
                    // the same relaxed-open coordinate.
                    false,
                );
                apply_eye_wide_master_to_tuning(&mut state.tuning, wide_master_enabled);
                // Apply exclusivity once, after the active Wide provider and the new
                // visual ranges are resolved. Direct SRanipalState users retain the
                // legacy in-core behaviour; the app avoids double attenuation here.
                // The former dashboard WIDE/SQUEEZE switch was an experimental
                // mutual attenuator, not an output master. Native squeeze is now
                // closed-eye gated, so the two expressions are already structurally
                // exclusive. Ignore legacy persisted `true` values in the app path.
                let expression_exclusive = false;
                state.tuning.wide_squeeze_exclusive = false;
                ch2_wide_state.tuning = state.tuning;
                if let Ok(mut seed) = guided_seed.lock() {
                    if let Some(store) = seed.take() {
                        state.restore_all(&store);
                        if persist_calibration {
                            crate::core::eye_state::save_calib(&calib_path, &store);
                        }
                        since_save = 0;
                    }
                }
                if drain_endpoint_apply(endpoint_seed.as_ref(), &mut state) {
                    if persist_calibration {
                        let store = state.snapshot_all();
                        crate::core::eye_state::save_calib(&calib_path, &store);
                    }
                    since_save = 0;
                }
                if let Ok(mut seed) = gaze_eyelid_seed.lock() {
                    if let Some(request) = seed.take() {
                        active_gaze_eyelid = request.profile;
                    }
                }
                if let Ok(mut seed) = wink_seed.lock() {
                    if let Some(request) = seed.take() {
                        active_wink = request.profile;
                    }
                }
                if let Ok(mut seed) = blink_timing_seed.lock() {
                    if let Some(request) = seed.take() {
                        active_blink_timing = request.profile;
                    }
                }
                let did_recenter = er.swap(false, Ordering::Relaxed);
                let eye_recenter_mask = eye_recenter.swap(0, Ordering::Relaxed);
                if did_recenter {
                    state.recenter();
                    ch2_wide_state.recenter();
                    brow_state.recenter(); // re-baseline the brow neutral too
                } else {
                    if eye_recenter_mask & 0b01 != 0 {
                        state.recenter_eye(Eye::Left);
                        ch2_wide_state.recenter_eye(Eye::Left);
                    }
                    if eye_recenter_mask & 0b10 != 0 {
                        state.recenter_eye(Eye::Right);
                        ch2_wide_state.recenter_eye(Eye::Right);
                    }
                }
                let did_wide_reset = wide_reset.swap(false, Ordering::Relaxed);
                if did_wide_reset {
                    ch2_wide_state.recenter();
                }
                if did_recenter || did_wide_reset {
                    custom_wide_state.recenter();
                    *t_em.wide_ready.lock().unwrap() = [false; 2];
                    *t_em.wide_bootstrap_seen.lock().unwrap() = [0; 2];
                }
                let g = t_em.fresh_runtime_sample();
                let m5 = *t_em.ml5.lock().unwrap();
                let mut results = state.process_frame_with_all_profiles(
                    m5,
                    &g,
                    t_em.ml_loaded,
                    &active_gaze_eyelid,
                    &active_wink,
                    &active_blink_timing,
                );
                // The left EyeNet head (ch1/ch3) remains authoritative for BOTH
                // physical eyelids. EyeWide deliberately uses the right head for
                // BOTH eyes: shadow ch2/ch4 for LEFT, normal ch2/ch4 for RIGHT.
                // The heads get independent baselines because their numerical scales
                // differ substantially on real SRanipal weights.
                let mut sranipal_wide = [results[0].wide, results[1].wide];
                let mut sranipal_wide_diag = None;
                let comparison = *t_em.eye_head_comparison.lock().unwrap();
                if let Some(comparison) = comparison {
                    let wide_ml5 = ch2_wide_frame(m5, comparison);
                    let wide_results = ch2_wide_state.process_frame(wide_ml5, &g, t_em.ml_loaded);
                    sranipal_wide = [wide_results[0].wide, wide_results[1].wide];
                    sranipal_wide_diag = Some(ch2_wide_state.eyelid_live_diag());

                    // ch1 owns closure truth. Never let a large or noisy ch2 value
                    // leak EyeWide through a blink, wink, lost eye, or untrusted lid.
                    if results
                        .iter()
                        .any(|result| result.blink || !result.openness_valid)
                    {
                        sranipal_wide = [0.0; 2];
                    }
                    results[0].wide = sranipal_wide[0];
                    results[1].wide = sranipal_wide[1];
                }
                if let Ok(mut value) = t_em.wide_sranipal.lock() {
                    *value = sranipal_wide;
                }

                // Always calculate Custom Wide in parallel when a model is loaded, even
                // while SRanipal remains selected. This gives us same-frame A/B telemetry
                // before the user entrusts VRCFT output to the new model.
                let mut custom = [None, None];
                let mut custom_fresh = false;
                if t_em.wide_loaded.load(Ordering::Relaxed) {
                    let raw = *t_em.wide_raw.lock().unwrap();
                    let generation = t_em.c_wide.load(Ordering::Relaxed);
                    let is_new = generation != last_wide_gen;
                    let now = Instant::now();
                    let infer_dt = if is_new {
                        let dt = now.duration_since(last_wide_infer).as_secs_f32();
                        last_wide_infer = now;
                        wide_fresh_at = now;
                        last_wide_gen = generation;
                        dt
                    } else {
                        0.0
                    };
                    custom = custom_wide_state.process_pair(
                        raw,
                        is_new,
                        [results[0].blink, results[1].blink],
                        infer_dt,
                        elapsed.as_secs_f32(),
                        state.tuning.wide_requires_both,
                    );
                    let wide_diag = custom_wide_state.diag();
                    *t_em.wide_ready.lock().unwrap() = wide_diag.ready;
                    *t_em.wide_bootstrap_seen.lock().unwrap() = wide_diag.bootstrap_seen;
                    custom_fresh = now.duration_since(wide_fresh_at) <= Duration::from_millis(500)
                        && custom[0].is_some()
                        && custom[1].is_some();
                    if let Ok(mut value) = t_em.wide_custom.lock() {
                        *value = [custom[0].unwrap_or(0.0), custom[1].unwrap_or(0.0)];
                    }
                }

                let selected_wide_source = *live_wide_source.lock().unwrap();
                let use_custom = match selected_wide_source {
                    WideSource::Sranipal => false,
                    WideSource::Auto | WideSource::Custom => custom_fresh,
                };
                if use_custom {
                    results[0].wide = custom[0].unwrap_or(0.0);
                    results[1].wide = custom[1].unwrap_or(0.0);
                } else if selected_wide_source == WideSource::Custom {
                    // Strict selection never silently falls back to SRanipal. During the
                    // short neutral bootstrap (or a stale model) it emits a safe zero.
                    results[0].wide = 0.0;
                    results[1].wide = 0.0;
                }
                // Final output policy: the guided XR5 result is a per-eye capability,
                // while the global master is the user's avatar-level preference on every
                // HMD. Neither stops raw SRanipal/Custom calculation or A/B diagnostics.
                let capability = *wide_on.lock().unwrap();
                let custom_output_active = apply_eye_wide_output_policy(
                    &mut results,
                    wide_master_enabled,
                    capability,
                    use_custom,
                    state.tuning.wide_requires_both,
                );
                t_em.wide_custom_active
                    .store(custom_output_active, Ordering::Relaxed);
                let expression_live =
                    apply_expression_response_ranges(&mut results, response, expression_exclusive);
                if let Ok(mut live) = t_em.expression_live.lock() {
                    *live = expression_live;
                }
                // Brow: time-based EMA + baseline of the brow worker's raw output, per eye.
                // `is_new` advances the EMA once per inference (not per 120Hz tick); the
                // result is None until the first open inference, so we never emit a
                // baseline derived from the zero placeholder.
                if brow_on.load(Ordering::Relaxed) && t_em.brow_loaded.load(Ordering::Relaxed) {
                    let raw = *t_em.brow_raw.lock().unwrap();
                    let gen = t_em.c_brow.load(Ordering::Relaxed);
                    let is_new = gen != last_brow_gen;
                    let infer_dt = if is_new {
                        let now = Instant::now();
                        let dt = now.duration_since(last_brow_infer).as_secs_f32();
                        last_brow_infer = now;
                        last_brow_gen = gen;
                        dt
                    } else {
                        0.0
                    };
                    for (i, r) in results.iter_mut().enumerate() {
                        if let Some(b) =
                            brow_state.process(i, raw[i], is_new, r.blink, did_recenter, infer_dt)
                        {
                            r.brow = b;
                            r.brow_valid = true;
                        }
                    }
                    apply_brow_lr_sync(&mut results, state.tuning.brow_lr_sync);
                }
                // Per-device gaze handedness: negate X so left/right isn't mirrored.
                // Applied here so telemetry AND all sinks see the same corrected gaze.
                apply_gaze_x_handedness(&mut results, fgx.load(Ordering::Relaxed));
                let correction = *gaze_trim.lock().unwrap();
                apply_gaze_correction(&mut results[0].gaze, Eye::Left, correction);
                apply_gaze_correction(&mut results[1].gaze, Eye::Right, correction);
                if let Ok(mut pu) = t_em.pupil.lock() {
                    *pu = [
                        (results[0].pupil_mm, results[0].pupil_valid),
                        (results[1].pupil_mm, results[1].pupil_valid),
                    ];
                }
                if let Ok(mut slot) = t_em.results.lock() {
                    *slot = results;
                }
                if let Ok(mut b) = t_em.baselines.lock() {
                    *b = [state.baseline(Eye::Left), state.baseline(Eye::Right)];
                }
                if let Ok(mut b) = t_em.wide_baselines.lock() {
                    *b = [
                        ch2_wide_state.baseline(Eye::Left),
                        ch2_wide_state.baseline(Eye::Right),
                    ];
                }
                if let Ok(mut live) = t_em.eyelid_live.lock() {
                    let mut published = state.eyelid_live_diag();
                    // Openness remains ch1-owned, but the combined visual rail's
                    // Wide boundary must come from the independent ch2 state that
                    // actually drives SRanipal EyeWide. This also makes a freshly
                    // learned Set Wide neutral threshold visible immediately.
                    if let Some(wide_diag) = sranipal_wide_diag {
                        for eye in 0..2 {
                            published[eye].wide_entry_ref = wide_diag[eye].wide_entry_ref;
                            published[eye].wide_full_ref = wide_diag[eye].wide_full_ref;
                        }
                    }
                    *live = published;
                }
                if let Ok(mut calibration) = t_em.calibration.lock() {
                    *calibration = Some(state.snapshot_all());
                }
                // Diagnostic recorder: raw ml values next to every post-processing
                // internal, one CSV row per emit frame, so a mis-correction can be
                // diagnosed offline from a short recorded session.
                if edr.load(Ordering::Relaxed) {
                    use std::io::Write;
                    if diag_file.is_none() {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let path =
                            crate::config::base_dir().join(format!("sranibro_diag_{ts}.csv"));
                        match std::fs::File::create(&path) {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = writeln!(
                                    w,
                                    "t_ms,raw_l,raw_r,ch0_l,ch0_r,sq3_l,sq3_r,\
                                     x_l,x_r,anchor_l,anchor_r,staged_l,staged_r,\
                                     open_l,open_r,wide_l,wide_r,squeeze_l,squeeze_r,\
                                     blink_l,blink_r,baseline_l,baseline_r,\
                                     closed_ref_l,closed_ref_r,is_wide_l,is_wide_r,\
                                     latched_l,latched_r,fall_run_l,fall_run_r,\
                                     since_down_l,since_down_r,blink_len_l,blink_len_r,\
                                     ep_exit_l,ep_exit_r,pend_w,\
                                     gin_lx,gin_ly,gin_rx,gin_ry,gv_l,gv_r,\
                                     gout_lx,gout_ly,gout_rx,gout_ry,\
                                     yoked_l,yoked_r,yhold_l,yhold_r,\
                                     ch2_l,ch2_r,ch4_l,ch4_r,\
                                     native_open_l,native_open_r,\
                                     native_open_valid_l,native_open_valid_r,\
                                     native_open_reported_l,native_open_reported_r,\
                                     wide_raw_l,wide_raw_r,wide_custom_l,wide_custom_r,\
                                     wide_sranipal_l,wide_sranipal_r,wide_custom_active"
                                );
                                println!("[diag] recording to {}", path.display());
                                diag_t0 = std::time::Instant::now();
                                diag_file = Some(w);
                            }
                            Err(e) => {
                                eprintln!("[diag] could not create the recording file: {e}");
                                edr.store(false, Ordering::Relaxed);
                            }
                        }
                    }
                    if let Some(w) = diag_file.as_mut() {
                        let d = state.diag();
                        let wide_raw_diag = *t_em.wide_raw.lock().unwrap();
                        let wide_custom_diag = *t_em.wide_custom.lock().unwrap();
                        let _ = writeln!(
                            w,
                            "{},{:.4},{:.4},{:.3},{:.3},{:.4},{:.4},\
                             {:.4},{:.4},{:.4},{:.4},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
                             {},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{},{},{},{},{},{},{:.2},\
                             {:.4},{:.4},{:.4},{:.4},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                            diag_t0.elapsed().as_millis(),
                            m5[0][1],
                            m5[1][1],
                            m5[0][0],
                            m5[1][0],
                            m5[0][3],
                            m5[1][3],
                            d.ramp_pre[0],
                            d.ramp_pre[1],
                            d.mid_anchor[0],
                            d.mid_anchor[1],
                            d.staged[0] as u8,
                            d.staged[1] as u8,
                            results[0].openness,
                            results[1].openness,
                            results[0].wide,
                            results[1].wide,
                            results[0].squeeze,
                            results[1].squeeze,
                            results[0].blink as u8,
                            results[1].blink as u8,
                            d.baseline[0],
                            d.baseline[1],
                            d.closed_ref[0],
                            d.closed_ref[1],
                            d.is_wide[0] as u8,
                            d.is_wide[1] as u8,
                            d.latched[0] as u8,
                            d.latched[1] as u8,
                            d.fall_run[0],
                            d.fall_run[1],
                            d.since_down[0].min(9999),
                            d.since_down[1].min(9999),
                            d.blink_len[0],
                            d.blink_len[1],
                            d.ep_exit[0] as u8,
                            d.ep_exit[1] as u8,
                            d.pend_w,
                            // Gaze diagnostics: device-in vs emitted-out + yoke state
                            // (the emitted gaze has flip_gaze_x already applied above,
                            // matching exactly what the sinks send).
                            g.left.gaze[0],
                            g.left.gaze[1],
                            g.right.gaze[0],
                            g.right.gaze[1],
                            g.left.gaze_valid as u8,
                            g.right.gaze_valid as u8,
                            results[0].gaze[0],
                            results[0].gaze[1],
                            results[1].gaze[0],
                            results[1].gaze[1],
                            results[0].gaze_yoked as u8,
                            results[1].gaze_yoked as u8,
                            d.yoke_hold[0] as u8,
                            d.yoke_hold[1] as u8,
                            m5[0][2],
                            m5[1][2],
                            m5[0][4],
                            m5[1][4],
                            g.left.openness,
                            g.right.openness,
                            g.left.openness_valid as u8,
                            g.right.openness_valid as u8,
                            g.left.openness_reported as u8,
                            g.right.openness_reported as u8,
                            wide_raw_diag[0],
                            wide_raw_diag[1],
                            wide_custom_diag[0],
                            wide_custom_diag[1],
                            sranipal_wide[0],
                            sranipal_wide[1],
                            use_custom as u8,
                        );
                    }
                } else if let Some(mut w) = diag_file.take() {
                    use std::io::Write;
                    let _ = w.flush();
                    println!("[diag] recording stopped");
                }
                t_em.c_emit.fetch_add(1, Ordering::Relaxed);
                if !ep.load(Ordering::Relaxed) {
                    for s in sinks.iter_mut() {
                        s.on_frame(&results);
                    }
                }
                since_save += 1;
                if persist_calibration && since_save >= 2400 {
                    since_save = 0;
                    crate::core::eye_state::save_calib(&calib_path, &state.snapshot_all());
                }
                thread::sleep(period);
            }
            // The UI may publish a validated endpoint immediately before stop.
            // Drain once after leaving the emit loop so the final checkpoint cannot
            // overwrite that update with the pre-apply state.
            drain_endpoint_apply(endpoint_seed.as_ref(), &mut state);
            if persist_calibration {
                crate::core::eye_state::save_calib(&calib_path, &state.snapshot_all());
            }
        }));

        Ok(Pipeline {
            adapter: Some(adapter),
            device_key,
            stop,
            threads,
            eye_image_http: None,
            tele,
            eyelid_model_identity,
            eyelid_backend_report,
            paused,
            recenter,
            guided_calibration,
            endpoint_apply,
            gaze_eyelid_apply,
            wink_apply,
            blink_timing_apply,
            diag_rec,
            eye_head_comparison_enabled,
            right_eye_left_head,
            recenter_eye_mask,
            swap_eyes,
            flip_image,
            ml_mirror_l,
            ml_mirror_r,
            flip_gaze_x,
            gaze_correction,
            eyebrow_enabled,
            eye_wide_enabled,
            tuning,
            eyelid_response,
            wearing_calibration_target,
            wearing_response_reset,
            manual_endpoint_edit,
            geometry,
            despeckle,
            flatten,
            brightness,
            photometric_correction,
            bright_affine,
            heatmap,
            device_status,
            brow,
            wide,
            wide_recenter,
            wide_enabled,
            wide_source,
        })
    }

    /// Hot-swap the eyebrow model into the LIVE pipeline with no device reconnect. Pass
    /// `Some(net)` to load a freshly trained model (e.g. right after B-2 train+bake) or
    /// `None` to drop brow output. The event-driven brow worker picks it up on the next
    /// camera notification, and `brow_loaded` flips so the emit thread + UI react.
    ///
    /// Note: brow is only *useful* when the eye (SRanipal) model is also loaded — the emit
    /// thread blink-gates brow on the eye net's openness. Loading brow without the eye net
    /// still stores it, but no brow is emitted until the eye net is present.
    pub fn set_brow(&self, net: Option<BrowNet>) {
        let loaded = net.is_some();
        if let Ok(mut g) = self.brow.lock() {
            *g = net;
        }
        self.tele.brow_loaded.store(loaded, Ordering::Relaxed);
    }

    /// Switch the physical right eyelid to the mirrored LEFT EyeNet head without
    /// rebuilding the camera pipeline. A changed raw coordinate invalidates only the
    /// right relaxed-open baseline, so request a one-eye recenter on the emit thread.
    pub fn set_right_eye_left_head(&self, enabled: bool) -> bool {
        let previous = self.right_eye_left_head.swap(enabled, Ordering::Relaxed);
        if previous != enabled {
            self.recenter_eye_mask.fetch_or(0b10, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    pub fn uses_right_eye_left_head(&self) -> bool {
        self.right_eye_left_head.load(Ordering::Relaxed)
    }

    /// Relearn only EyeWide's relaxed neutral coordinate. Unlike the main Recenter,
    /// this leaves eyelid openness, blink/wink endpoints, brow neutral and gaze state
    /// untouched. Both the SRanipal right-head route and an optional XR5 custom Wide
    /// provider consume the request on the next emit frame.
    pub fn recenter_wide_neutral(&self) {
        self.wide_recenter.store(true, Ordering::Relaxed);
    }

    pub fn set_wearing_calibration_target(&self, target: Option<WearingCalibrationTarget>) {
        *self
            .wearing_calibration_target
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = target;
    }

    /// Release a recalled response immediately so direct manipulation of the visual
    /// endpoint controls cannot be overwritten by an earlier appearance match.
    pub fn clear_wearing_calibration_target_immediately(&self) {
        self.set_wearing_calibration_target(None);
        self.wearing_response_reset.store(true, Ordering::Release);
    }

    pub fn edit_manual_endpoints(&self, baseline: [f32; 2], profile: EyelidResponseProfile) {
        *self.manual_endpoint_edit.lock().unwrap() = Some((baseline, profile.sanitized()));
    }

    pub fn set_live_response(&self, profile: EyelidResponseProfile) {
        let mut pending = self.manual_endpoint_edit.lock().unwrap();
        if let Some((_, queued)) = pending.as_mut() {
            *queued = profile.sanitized();
        } else {
            *self.eyelid_response.lock().unwrap() = profile;
        }
    }

    pub fn wearing_calibration_target(&self) -> Option<WearingCalibrationTarget> {
        *self
            .wearing_calibration_target
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Select the XR5 EyeWide provider without restarting the camera or output pipeline.
    /// Auto/Custom require a loaded model; callers get a recoverable error instead of
    /// leaving an unstartable setting behind.
    pub fn set_wide_source(&self, requested: WideSource) -> Result<(), &'static str> {
        let source = if crate::config::canonical_device_key(&self.device_key) == "pimax_xr5" {
            requested
        } else {
            WideSource::Sranipal
        };
        if source != WideSource::Sranipal && !self.tele.wide_loaded.load(Ordering::Relaxed) {
            return Err("Auto and Custom require a fitted XR5 EyeWide model");
        }
        *self.wide_source.lock().unwrap() = source;
        self.tele.wide_custom_active.store(false, Ordering::Relaxed);
        self.wide_recenter.store(true, Ordering::Relaxed);
        Ok(())
    }

    pub fn selected_wide_source(&self) -> WideSource {
        *self.wide_source.lock().unwrap()
    }

    /// Hot-swap the custom XR5 EyeWide model. The next ML iteration starts producing
    /// A/B telemetry; source selection still follows `[hmd].wide_source` after reload.
    pub fn set_wide(&self, net: Option<WideNet>) {
        let net = if crate::config::canonical_device_key(&self.device_key) == "pimax_xr5" {
            net
        } else {
            if net.is_some() {
                eprintln!(
                    "[wide] refused XR5 custom EyeWide hot-load for non-XR5 device {}",
                    self.device_key
                );
            }
            None
        };
        let loaded = net.is_some();
        if let Ok(mut guard) = self.wide.lock() {
            *guard = net;
        }
        self.tele.wide_loaded.store(loaded, Ordering::Relaxed);
        self.tele.wide_custom_active.store(false, Ordering::Relaxed);
        *self.tele.wide_ready.lock().unwrap() = [false; 2];
        *self.tele.wide_bootstrap_seen.lock().unwrap() = [0; 2];
        self.wide_recenter.store(true, Ordering::Relaxed);
    }

    /// Stop all threads and the adapter.
    pub fn stop(&mut self) {
        self.take_shutdown()();
    }

    /// Transfer only lifecycle ownership. Shared UI handles stay usable, but the
    /// old pipeline cannot be restarted. Join/device waits run on the caller's worker.
    pub fn take_shutdown(&mut self) -> impl FnOnce() + Send + 'static {
        self.stop.store(true, Ordering::Relaxed);
        let http = self.eye_image_http.take();
        let threads = std::mem::take(&mut self.threads);
        let adapter = self.adapter.take();
        move || {
            drop(http);
            for t in threads {
                let _ = t.join();
            }
            if let Some(mut adapter) = adapter {
                adapter.stop();
            }
        }
    }
}
