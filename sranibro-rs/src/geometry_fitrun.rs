//! Background, pure-Rust search for a safer per-user XR5 EyeNet input geometry.
//!
//! The fixed SRanipal network is never trained here.  We search a deliberately small
//! neighbourhood around the currently active XR5 reconstruction, score scale-free
//! response shape on labelled capture frames, and accept a candidate only when it also
//! beats the original geometry on a separate holdout tail.

use std::cmp::Ordering as CmpOrdering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::core::types::{DespeckleParams, FlattenParams, MlGeometry, PhotometricCorrection};
use crate::geometry_calib::{
    GeometryDataset, GeometrySample, SampleFamily, SampleKind, SharedEvidence,
};
use crate::geometry_discovery::{
    estimate_appearance_geometry, estimate_motion_geometry, AppearanceGeometryEstimate,
    MotionFrame, MotionGeometryEstimate,
};
use crate::ml::{brightness, eye_net::EyeNet, preprocess, tvm_params};

const LOG_CAP: usize = 80;
const STAGE1_CANDIDATES: usize = 48;
const STAGE1_FRAMES: usize = 112;
const STAGE2_CANDIDATES: usize = 10;
const STAGE2_FRAMES: usize = 280;
const STAGE3_FRAMES: usize = 420;
const HOLDOUT_FRAMES: usize = 420;
const XR5_MIN_INNER_CROP: f32 = 0.40;
const AUDIT_FOLDS: usize = 5;
const AUDIT_BLOCK_FRAMES: usize = 12;
const AUDIT_GUARD_FRAMES: usize = 1;
// Every fold must contain the same amount of train-only evidence.  Ten is the
// guaranteed minimum after block guards in the real capture protocol; asking for
// more made fold zero larger than the remaining folds and biased the audit mean.
const AUDIT_FRAMES_PER_FAMILY_FOLD: usize = 10;
const STATIC_PHASE_TRIM_S: f32 = 0.70;
const STATIC_TAIL_S: f32 = 1.0;
const STATIC_END_GUARD_FRAMES: usize = 2;
const ANCHOR_SIM_MULT: f32 = 2.0;
const MIN_STATIC_STABLE_S: f32 = 0.40;
const GAZE_PHASE_TRIM_S: f32 = 0.50;
const MIN_STATIC_STABLE_FRAMES: usize = 5;
const MIN_GAZE_STABLE_FRAMES: usize = 5;

#[derive(Clone, Debug)]
pub struct FitInputs {
    pub model_path: PathBuf,
    /// Exact bytes verified against the live model before the capture is consumed.
    /// Workers parse this immutable snapshot rather than re-reading a mutable path.
    pub model_bytes: Arc<[u8]>,
    pub expected_model_crc32: u32,
    pub expected_model_bytes: u64,
    pub dataset: SharedEvidence,
    /// Geometry active at capture start. It is the immutable fallback and search centre.
    pub baseline: [MlGeometry; 2],
    /// Effective live mirror flags. Mirror is hardware handedness, never a search variable.
    pub mirrors: [bool; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
}

#[derive(Debug)]
pub struct StartError {
    pub message: String,
    pub inputs: FitInputs,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeometryMetrics {
    pub evidence_valid: bool,
    pub score: f32,
    pub separation: [f32; 2],
    pub open_ref: [f32; 2],
    pub closed_ref: [f32; 2],
    pub monotonicity: [f32; 2],
    pub slow_close_std: [f32; 2],
    pub blink_response: f32,
    pub stability: f32,
    pub presence_rate: f32,
    pub finite_rate: f32,
    pub image_information: f32,
    pub image_std: f32,
    pub saturation_rate: f32,
    pub motion_energy: f32,
    pub neutral_noise: f32,
    pub gaze_noise: f32,
    pub neutral_noise_per_eye: [f32; 2],
    pub gaze_noise_per_eye: [f32; 2],
    /// Fraction of the open-to-closed eyelid span retained while looking around.
    /// Values are per eye; 1.0 means gaze direction did not falsely close the lid.
    pub gaze_retention: [f32; 2],
    /// Spurious squeeze produced during the gaze sweep, relative to relaxed neutral.
    pub gaze_squeeze_fp: [f32; 2],
    /// Absolute difference between left and right gaze retention.
    pub gaze_asymmetry: f32,
    /// Fraction of gaze-sweep frames with valid native Tobii vectors for both eyes.
    pub gaze_evidence_rate: f32,
    pub blink_events: [usize; 2],
}

#[derive(Clone, Debug)]
pub struct GeometryFitResult {
    pub baseline: [MlGeometry; 2],
    pub candidate: [MlGeometry; 2],
    pub baseline_train: GeometryMetrics,
    pub candidate_train: GeometryMetrics,
    pub baseline_holdout: GeometryMetrics,
    pub candidate_holdout: GeometryMetrics,
    pub holdout_improvement: f32,
    /// EyeNet-independent absolute crop/rotation initialization derived from the
    /// training blink motion. It is reported even when safety gates keep it out of the
    /// search, and it never sees the untouched holdout.
    pub motion_seed: Option<MotionGeometryEstimate>,
    pub candidate_from_motion_seed: bool,
    /// EyeNet-independent absolute seed derived from repeated relaxed-neutral pupil
    /// centres and aperture axes. Like the motion seed, it never sees the holdout.
    pub appearance_seed: Option<AppearanceGeometryEstimate>,
    pub candidate_from_appearance_seed: bool,
    pub invalid_static_phases: usize,
    pub degraded_static_phases: usize,
    pub valid_closed_phases: [usize; 2],
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct PhotometricFitInputs {
    pub model_path: PathBuf,
    pub model_bytes: Arc<[u8]>,
    pub expected_model_crc32: u32,
    pub expected_model_bytes: u64,
    pub dataset: SharedEvidence,
    /// Frozen frontal geometry. It is evaluated but never searched or changed.
    pub geometry: [MlGeometry; 2],
    pub mirrors: [bool; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    /// Correction active at capture start and the immutable fallback.
    pub baseline: PhotometricCorrection,
}

#[derive(Debug)]
pub struct PhotometricStartError {
    pub message: String,
    pub inputs: PhotometricFitInputs,
}

#[derive(Clone, Debug)]
pub struct PhotometricFitResult {
    pub baseline: PhotometricCorrection,
    pub candidate: PhotometricCorrection,
    pub baseline_train: GeometryMetrics,
    pub candidate_train: GeometryMetrics,
    pub baseline_holdout: GeometryMetrics,
    pub candidate_holdout: GeometryMetrics,
    pub holdout_improvement: f32,
    pub invalid_static_phases: usize,
    pub degraded_static_phases: usize,
    pub valid_closed_phases: [usize; 2],
    pub accepted: bool,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum PhotometricStatus {
    Idle,
    Running {
        stage: String,
        completed: usize,
        total: usize,
        log: Vec<String>,
    },
    Done {
        result: PhotometricFitResult,
        log: Vec<String>,
    },
    Failed {
        message: String,
        log: Vec<String>,
    },
    Cancelled {
        log: Vec<String>,
    },
}

impl PhotometricStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

/// Mean and between-fold spread for one geometry-audit signal.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuditStat {
    pub mean: f32,
    pub stddev: f32,
}

/// One deterministic probe around the active, user-validated geometry.
#[derive(Clone, Debug)]
pub struct GeometryAuditCase {
    pub name: String,
    pub geometry: [MlGeometry; 2],
    pub current_score: AuditStat,
    pub legacy_score: AuditStat,
    pub absolute_span: AuditStat,
    pub half_position: [AuditStat; 2],
    pub half_error: AuditStat,
    pub bimodality: AuditStat,
    pub reproducibility: f32,
    pub confident_wrong: bool,
}

#[derive(Clone, Debug, Default)]
pub struct HalfQuality {
    pub position: [f32; 2],
    pub normalized_stddev: [f32; 2],
    pub block_disagreement: [f32; 2],
    pub native_coverage: [f32; 2],
    pub warnings: Vec<String>,
}

/// Diagnostic-only comparison of the in-app objective against the criteria that found
/// the original XR5 preset. It never changes or previews live geometry.
#[derive(Clone, Debug)]
pub struct GeometryAuditResult {
    pub cases: Vec<GeometryAuditCase>,
    pub current_best: usize,
    pub legacy_best: usize,
    pub confident_wrong_count: usize,
    pub edge_drift_axes: Vec<String>,
    pub half_quality: HalfQuality,
    pub evidence_ready: bool,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Running {
        stage: String,
        completed: usize,
        total: usize,
        log: Vec<String>,
    },
    Done {
        result: GeometryFitResult,
        log: Vec<String>,
    },
    AuditDone {
        result: GeometryAuditResult,
        log: Vec<String>,
    },
    Failed {
        message: String,
        log: Vec<String>,
    },
    Cancelled {
        log: Vec<String>,
    },
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

struct Shared {
    status: Status,
    log: Vec<String>,
    stage: String,
    completed: usize,
    total: usize,
}

impl Shared {
    fn push(&mut self, line: impl Into<String>) {
        if self.log.len() >= LOG_CAP {
            self.log.drain(0..self.log.len() - LOG_CAP + 1);
        }
        self.log.push(line.into());
        if matches!(self.status, Status::Running { .. }) {
            self.status = Status::Running {
                stage: self.stage.clone(),
                completed: self.completed,
                total: self.total,
                log: self.log.clone(),
            };
        }
    }

    fn progress(&mut self, stage: &str, completed: usize, total: usize) {
        self.stage = stage.into();
        self.completed = completed.min(total);
        self.total = total;
        self.status = Status::Running {
            stage: self.stage.clone(),
            completed: self.completed,
            total,
            log: self.log.clone(),
        };
    }
}

pub struct GeometryFitter {
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy)]
enum JobKind {
    Fit,
    Audit,
}

impl Default for GeometryFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryFitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
                log: Vec::new(),
                stage: String::new(),
                completed: 0,
                total: 0,
            })),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }

    pub fn status(&self) -> Status {
        lock_shared(&self.shared).status.clone()
    }

    pub fn is_running(&self) -> bool {
        self.status().is_running()
    }

    pub fn start(&mut self, inputs: FitInputs) -> Result<(), StartError> {
        self.start_job(inputs, JobKind::Fit)
    }

    pub fn start_audit(&mut self, inputs: FitInputs) -> Result<(), StartError> {
        self.start_job(inputs, JobKind::Audit)
    }

    fn start_job(&mut self, inputs: FitInputs, job: JobKind) -> Result<(), StartError> {
        if self.is_running() {
            return Err(StartError {
                message: "an XR5 geometry job is already running".into(),
                inputs,
            });
        }
        if inputs.model_bytes.is_empty()
            || inputs.model_bytes.len() as u64 != inputs.expected_model_bytes
            || crate::diagnostics::crc32_fingerprint(&inputs.model_bytes)
                != inputs.expected_model_crc32
        {
            return Err(StartError {
                message: "EyePrediction model snapshot does not match the live load-time identity"
                    .into(),
                inputs,
            });
        }
        if let Err(message) = validate_dataset_shape(inputs.dataset.as_dataset()) {
            return Err(StartError { message, inputs });
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.cancel.store(false, Ordering::Relaxed);
        {
            let mut state = lock_shared(&self.shared);
            state.log.clear();
            state.stage = "starting".into();
            state.completed = 0;
            state.total = 1;
            state.status = Status::Running {
                stage: state.stage.clone(),
                completed: 0,
                total: 1,
                log: Vec::new(),
            };
        }
        let shared = self.shared.clone();
        let panic_shared = shared.clone();
        let cancel = self.cancel.clone();
        // Keep a second Arc to the not-yet-consumed input. If OS thread creation
        // fails, the UI can restore the completed capture without cloning the large
        // raw-frame dataset. A successfully started worker takes it exactly once.
        let pending = Arc::new(Mutex::new(Some(inputs)));
        let worker_pending = pending.clone();
        match std::thread::Builder::new()
            .name(match job {
                JobKind::Fit => "xr5-geometry-fitter".into(),
                JobKind::Audit => "xr5-geometry-audit".into(),
            })
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    // Candidate replay is sustained bulk CPU work. Keep live camera,
                    // ML/output and the Windows compositor ahead of it under contention.
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let inputs = worker_pending
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take()
                        .expect("geometry fitter input must be present at worker start");
                    match job {
                        JobKind::Fit => run(shared, cancel, inputs),
                        JobKind::Audit => run_audit(shared, cancel, inputs),
                    }
                }));
                if outcome.is_err() {
                    fail(
                        &panic_shared,
                        "geometry worker hit an unexpected internal panic; current geometry was not changed"
                            .into(),
                    );
                }
            }) {
            Ok(handle) => {
                self.handle = Some(handle);
                Ok(())
            }
            Err(error) => {
                let inputs = pending
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .expect("failed thread spawn must leave geometry fitter input available");
                lock_shared(&self.shared).status = Status::Idle;
                Err(StartError {
                    message: format!("could not spawn XR5 geometry worker: {error}"),
                    inputs,
                })
            }
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Drop a completed/cancelled result before a new capture starts. A running job is
    /// never cleared: its worker must be cancelled and finish first.
    pub fn clear_finished(&mut self) -> bool {
        if self.is_running() {
            return false;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let mut state = lock_shared(&self.shared);
        state.status = Status::Idle;
        state.log.clear();
        state.stage.clear();
        state.completed = 0;
        state.total = 0;
        true
    }
}

impl Drop for GeometryFitter {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

struct PhotometricShared {
    status: PhotometricStatus,
    log: Vec<String>,
    stage: String,
    completed: usize,
    total: usize,
}

impl PhotometricShared {
    fn push(&mut self, line: impl Into<String>) {
        if self.log.len() >= LOG_CAP {
            self.log.drain(0..self.log.len() - LOG_CAP + 1);
        }
        self.log.push(line.into());
        if matches!(self.status, PhotometricStatus::Running { .. }) {
            self.publish_running();
        }
    }

    fn progress(&mut self, stage: &str, completed: usize, total: usize) {
        self.stage = stage.into();
        self.completed = completed.min(total);
        self.total = total;
        self.publish_running();
    }

    fn publish_running(&mut self) {
        self.status = PhotometricStatus::Running {
            stage: self.stage.clone(),
            completed: self.completed,
            total: self.total,
            log: self.log.clone(),
        };
    }
}

pub struct PhotometricFitter {
    shared: Arc<Mutex<PhotometricShared>>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Default for PhotometricFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl PhotometricFitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(PhotometricShared {
                status: PhotometricStatus::Idle,
                log: Vec::new(),
                stage: String::new(),
                completed: 0,
                total: 0,
            })),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }

    pub fn status(&self) -> PhotometricStatus {
        lock_photometric(&self.shared).status.clone()
    }

    pub fn is_running(&self) -> bool {
        self.status().is_running()
    }

    pub fn start(&mut self, inputs: PhotometricFitInputs) -> Result<(), PhotometricStartError> {
        if self.is_running() {
            return Err(PhotometricStartError {
                message: "a photometric fit is already running".into(),
                inputs,
            });
        }
        if inputs.model_bytes.is_empty()
            || inputs.model_bytes.len() as u64 != inputs.expected_model_bytes
            || crate::diagnostics::crc32_fingerprint(&inputs.model_bytes)
                != inputs.expected_model_crc32
        {
            return Err(PhotometricStartError {
                message: "EyePrediction model snapshot does not match the live load-time identity"
                    .into(),
                inputs,
            });
        }
        if let Err(message) = validate_dataset_shape(inputs.dataset.as_dataset()) {
            return Err(PhotometricStartError { message, inputs });
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.cancel.store(false, Ordering::Relaxed);
        {
            let mut state = lock_photometric(&self.shared);
            state.log.clear();
            state.stage = "starting".into();
            state.completed = 0;
            state.total = 1;
            state.publish_running();
        }
        let shared = self.shared.clone();
        let panic_shared = shared.clone();
        let cancel = self.cancel.clone();
        let pending = Arc::new(Mutex::new(Some(inputs)));
        let worker_pending = pending.clone();
        match std::thread::Builder::new()
            .name("photometric-fitter".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let inputs = worker_pending
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .take()
                        .expect("photometric fitter input must be present");
                    run_photometric(shared, cancel, inputs);
                }));
                if outcome.is_err() {
                    photometric_fail(
                        &panic_shared,
                        "photometric worker hit an unexpected internal panic; current correction was not changed"
                            .into(),
                    );
                }
            }) {
            Ok(handle) => {
                self.handle = Some(handle);
                Ok(())
            }
            Err(error) => {
                let inputs = pending
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                    .expect("failed spawn must leave photometric input available");
                lock_photometric(&self.shared).status = PhotometricStatus::Idle;
                Err(PhotometricStartError {
                    message: format!("could not spawn photometric worker: {error}"),
                    inputs,
                })
            }
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn clear_finished(&mut self) -> bool {
        if self.is_running() {
            return false;
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let mut state = lock_photometric(&self.shared);
        state.status = PhotometricStatus::Idle;
        state.log.clear();
        state.stage.clear();
        state.completed = 0;
        state.total = 0;
        true
    }
}

impl Drop for PhotometricFitter {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            drop(handle);
        }
    }
}

#[derive(Clone)]
struct PreparedSample {
    kind: SampleKind,
    expected_open: Option<f32>,
    phase_index: usize,
    native_open: [Option<f32>; 2],
    native_gaze_deg: [Option<[f32; 2]>; 2],
    /// Candidate-independent capture stability derived once from the raw eye images.
    stable: bool,
    left: Vec<u8>,
    right: Vec<u8>,
    left_size: (u32, u32),
    right_size: (u32, u32),
}

/// Unilateral wink poses are semantic evidence for `wink_fit`, not bilateral
/// closed-eye evidence for image alignment or photometric fitting. `SampleKind`
/// intentionally maps them to `SampleFamily::Closed` for legacy exhaustiveness,
/// so every geometry-scoring boundary must exclude them by exact kind.
fn is_geometry_scoring_kind(kind: SampleKind) -> bool {
    !matches!(
        kind,
        SampleKind::LeftWink
            | SampleKind::RightWink
            | SampleKind::HoldoutLeftWink
            | SampleKind::HoldoutRightWink
    )
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SearchParams([f32; 5]);

impl SearchParams {
    fn clamped(mut self) -> Self {
        for value in &mut self.0 {
            *value = value.clamp(-1.0, 1.0);
        }
        self
    }

    fn distance(self, other: Self) -> f32 {
        self.0
            .iter()
            .zip(other.0)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[derive(Clone)]
struct Scored {
    params: SearchParams,
    geometry: [MlGeometry; 2],
    metrics: GeometryMetrics,
    motion_seed: bool,
    appearance_seed: bool,
}

#[derive(Clone, Copy)]
struct Observation {
    kind: SampleKind,
    expected_open: Option<f32>,
    phase_index: usize,
    native_open: [Option<f32>; 2],
    native_gaze_deg: [Option<[f32; 2]>; 2],
    stable: bool,
    presence: f32,
    open: [f32; 2],
    squeeze: [f32; 2],
}

#[derive(Default)]
struct ImageAccum {
    spatial_std_sum: f64,
    saturation_sum: f64,
    pixels: usize,
    frames: usize,
    motion_sum: f64,
    motion_pixels: usize,
}

#[derive(Clone, Debug, Default)]
pub struct StabilityReport {
    pub flags: Vec<bool>,
    pub invalid_static_phases: usize,
    pub degraded_static_phases: usize,
    pub valid_closed_phases: [usize; 2],
    pub notes: Vec<String>,
}

#[derive(Clone)]
struct StereoAnchor {
    left: Vec<u8>,
    right: Vec<u8>,
}

struct StaticPhaseEvidence {
    phase_index: usize,
    family: SampleFamily,
    split: usize,
    trimmed: Vec<usize>,
    tail: Vec<usize>,
    anchor: StereoAnchor,
    similarity: f32,
    invalid: bool,
}

/// Select the requested static pose from the *end* of each phase. A participant can
/// remain perfectly still in the previous pose for several seconds, so first-quiet-
/// streak detection is fundamentally ambiguous. End anchoring uses only raw pixels,
/// labels and the fixed fallback crop and is therefore identical for every candidate.
fn capture_stability_flags(
    samples: &[GeometrySample],
    baseline: [MlGeometry; 2],
) -> StabilityReport {
    let mut report = StabilityReport {
        flags: vec![false; samples.len()],
        ..StabilityReport::default()
    };
    for (index, sample) in samples.iter().enumerate() {
        if !is_geometry_scoring_kind(sample.kind) {
            continue;
        }
        if matches!(
            sample.kind.family(),
            SampleFamily::SlowClose | SampleFamily::NaturalBlinks
        ) {
            report.flags[index] = true;
        }
    }

    let mut grouped = BTreeMap::<usize, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        if is_geometry_scoring_kind(sample.kind)
            && matches!(
                sample.kind.family(),
                SampleFamily::Neutral | SampleFamily::Closed | SampleFamily::HalfOpen
            )
        {
            grouped.entry(sample.phase_index).or_default().push(index);
        }
    }

    let mut phases = Vec::new();
    for (phase_index, indices) in grouped {
        let mut trimmed = indices
            .into_iter()
            .filter(|index| samples[*index].phase_time_s >= STATIC_PHASE_TRIM_S)
            .collect::<Vec<_>>();
        if trimmed.len() <= STATIC_END_GUARD_FRAMES {
            report.invalid_static_phases += 1;
            report.notes.push(format!(
                "phase {phase_index}: fewer than three guarded static frames"
            ));
            continue;
        }
        trimmed.truncate(trimmed.len() - STATIC_END_GUARD_FRAMES);
        let guarded_end_s = samples[*trimmed.last().unwrap()].phase_time_s;
        let tail = trimmed
            .iter()
            .copied()
            .filter(|index| samples[*index].phase_time_s >= guarded_end_s - STATIC_TAIL_S - 1.0e-4)
            .collect::<Vec<_>>();
        if tail.len() < 3 {
            report.invalid_static_phases += 1;
            report.notes.push(format!(
                "phase {phase_index}: static tail has fewer than 3 frames"
            ));
            continue;
        }
        let Some(anchor) = median_stereo_anchor(samples, &tail, baseline) else {
            report.invalid_static_phases += 1;
            report.notes.push(format!(
                "phase {phase_index}: camera dimensions changed inside the static tail"
            ));
            continue;
        };
        let sample = &samples[tail[0]];
        phases.push(StaticPhaseEvidence {
            phase_index,
            family: sample.kind.family(),
            split: sample.kind.is_holdout() as usize,
            trimmed,
            tail,
            anchor,
            similarity: 0.0,
            invalid: false,
        });
    }

    let mut motion_threshold = [0.01; 2];
    for (split, threshold) in motion_threshold.iter_mut().enumerate() {
        let motions = phases
            .iter()
            .filter(|phase| phase.split == split)
            .flat_map(|phase| phase.tail.windows(2))
            .filter_map(|pair| raw_stereo_l1(&samples[pair[0]], &samples[pair[1]], baseline))
            .collect::<Vec<_>>();
        *threshold = (percentile(&motions, 0.50).unwrap_or(0.005) * 2.0).clamp(0.002, 0.05);
    }

    for phase in &mut phases {
        let threshold = motion_threshold[phase.split];
        let tail_distances = phase
            .tail
            .iter()
            .filter_map(|index| anchor_distance(&samples[*index], &phase.anchor, baseline))
            .collect::<Vec<_>>();
        phase.similarity = (ANCHOR_SIM_MULT
            * percentile(&tail_distances, 0.50).unwrap_or(threshold))
        .max(threshold)
        .clamp(0.002, 0.05);
    }

    // A Closed phase whose final raw appearance is indistinguishable from the nearest
    // Neutral phase is non-compliant evidence, not a valid zero-span calibration point.
    for closed_index in 0..phases.len() {
        if phases[closed_index].family != SampleFamily::Closed {
            continue;
        }
        let split = phases[closed_index].split;
        let phase_index = phases[closed_index].phase_index;
        let nearest = phases
            .iter()
            .enumerate()
            .filter(|(_, phase)| phase.split == split && phase.family == SampleFamily::Neutral)
            .min_by_key(|(_, phase)| phase.phase_index.abs_diff(phase_index));
        if let Some((_, neutral)) = nearest {
            let distance = anchor_l1(&phases[closed_index].anchor, &neutral.anchor);
            // Compare two median pose anchors against their own within-pose noise
            // envelopes. Using twice the frame-to-frame motion threshold here made the
            // limit four times the camera noise and rejected real VR4 closures: the
            // eyelid changes only a small part of the full 200x200 image, while sensor
            // noise is present everywhere.
            let indistinguishable_limit = phases[closed_index]
                .similarity
                .max(neutral.similarity)
                .max(0.004);
            match distance {
                Some(distance) if distance <= indistinguishable_limit => {
                    phases[closed_index].invalid = true;
                    report.invalid_static_phases += 1;
                    report.notes.push(format!(
                        "phase {phase_index}: CLOSED tail is visually indistinguishable from \
                         NEUTRAL (distance {distance:.4}, noise limit \
                         {indistinguishable_limit:.4})"
                    ));
                }
                _ => {}
            }
        }
    }

    for phase in &phases {
        if phase.invalid {
            continue;
        }
        let threshold = motion_threshold[phase.split];
        let last_move_position =
            phase
                .trimmed
                .windows(2)
                .enumerate()
                .rev()
                .find_map(|(position, pair)| {
                    raw_stereo_l1(&samples[pair[0]], &samples[pair[1]], baseline)
                        .is_some_and(|motion| motion > threshold)
                        .then_some(position + 1)
                });
        let after_last_move = last_move_position.unwrap_or(0);
        let mut start = phase.trimmed.len();
        for position in (after_last_move..phase.trimmed.len()).rev() {
            let index = phase.trimmed[position];
            if anchor_distance(&samples[index], &phase.anchor, baseline)
                .is_some_and(|distance| distance <= phase.similarity)
            {
                start = position;
            } else {
                break;
            }
        }
        let stable = &phase.trimmed[start..];
        let stable_span = stable
            .first()
            .zip(stable.last())
            .map(|(first, last)| samples[*last].phase_time_s - samples[*first].phase_time_s)
            .unwrap_or(0.0);
        let selected =
            if stable.len() >= MIN_STATIC_STABLE_FRAMES && stable_span >= MIN_STATIC_STABLE_S {
                stable
            } else {
                report.degraded_static_phases += 1;
                report.notes.push(format!(
                    "phase {}: no long stable suffix; guarded final {:.1}s tail used",
                    phase.phase_index, STATIC_TAIL_S
                ));
                phase.tail.as_slice()
            };
        for &index in selected {
            report.flags[index] = true;
        }
        if phase.family == SampleFamily::Closed {
            report.valid_closed_phases[phase.split] += 1;
        }
    }

    // Gaze begins only after the raw eye image has actually departed from the preceding
    // static pose for two consecutive frames. This removes delayed reaction without
    // consulting EyeNet output or a candidate geometry.
    let mut gaze_groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        if sample.kind.family() == SampleFamily::GazeSweep {
            gaze_groups
                .entry(sample.phase_index)
                .or_default()
                .push(index);
        }
    }
    for (phase_index, indices) in gaze_groups {
        let split = samples[indices[0]].kind.is_holdout() as usize;
        let previous = phases
            .iter()
            .filter(|phase| {
                !phase.invalid
                    && phase.split == split
                    && phase.family == SampleFamily::Neutral
                    && phase.phase_index < phase_index
            })
            .max_by_key(|phase| phase.phase_index);
        // Without a preceding valid Neutral anchor there is no candidate-independent
        // way to distinguish an actual gaze departure from carry-over motion. Exclude
        // that phase instead of treating every post-trim frame as valid gaze evidence.
        let mut departed = false;
        let mut departure_streak = 0usize;
        for index in indices {
            if samples[index].phase_time_s < GAZE_PHASE_TRIM_S {
                continue;
            }
            if let Some(previous) = previous {
                let outside = anchor_distance(&samples[index], &previous.anchor, baseline)
                    .is_some_and(|distance| distance > previous.similarity);
                departure_streak = if outside { departure_streak + 1 } else { 0 };
                if departure_streak >= 2 {
                    departed = true;
                }
            }
            report.flags[index] = departed;
        }
    }
    report
}

fn crop_stereo_pixels(sample: &GeometrySample, baseline: [MlGeometry; 2]) -> Option<StereoAnchor> {
    Some(StereoAnchor {
        left: crop_pixels(&sample.left, sample.left_size, baseline[0])?,
        right: crop_pixels(&sample.right, sample.right_size, baseline[1])?,
    })
}

fn crop_pixels(frame: &[u8], size: (u32, u32), geometry: MlGeometry) -> Option<Vec<u8>> {
    let (width, height) = (size.0 as usize, size.1 as usize);
    if width == 0 || height == 0 || frame.len() < width.saturating_mul(height) {
        return None;
    }
    let x0 = (geometry.crop_left.clamp(0.0, 0.95) * width as f32).floor() as usize;
    let x1 = ((1.0 - geometry.crop_right.clamp(0.0, 0.95)) * width as f32).ceil() as usize;
    let y0 = (geometry.crop_top.clamp(0.0, 0.95) * height as f32).floor() as usize;
    let y1 = ((1.0 - geometry.crop_bottom.clamp(0.0, 0.95)) * height as f32).ceil() as usize;
    let (x1, y1) = (x1.clamp(x0 + 1, width), y1.clamp(y0 + 1, height));
    let mut pixels = Vec::with_capacity((x1 - x0) * (y1 - y0));
    for y in y0..y1 {
        pixels.extend_from_slice(&frame[y * width + x0..y * width + x1]);
    }
    Some(pixels)
}

fn median_stereo_anchor(
    samples: &[GeometrySample],
    indices: &[usize],
    baseline: [MlGeometry; 2],
) -> Option<StereoAnchor> {
    let frames = indices
        .iter()
        .map(|index| crop_stereo_pixels(&samples[*index], baseline))
        .collect::<Option<Vec<_>>>()?;
    let left_len = frames.first()?.left.len();
    let right_len = frames.first()?.right.len();
    if frames
        .iter()
        .any(|frame| frame.left.len() != left_len || frame.right.len() != right_len)
    {
        return None;
    }
    Some(StereoAnchor {
        left: median_anchor_channel(&frames, false, left_len),
        right: median_anchor_channel(&frames, true, right_len),
    })
}

fn median_anchor_channel(frames: &[StereoAnchor], right: bool, len: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(len);
    let mut values = Vec::with_capacity(frames.len());
    for pixel in 0..len {
        values.clear();
        values.extend(frames.iter().map(|frame| {
            if right {
                frame.right[pixel]
            } else {
                frame.left[pixel]
            }
        }));
        let middle = values.len() / 2;
        values.select_nth_unstable(middle);
        result.push(values[middle]);
    }
    result
}

fn anchor_distance(
    sample: &GeometrySample,
    anchor: &StereoAnchor,
    baseline: [MlGeometry; 2],
) -> Option<f32> {
    let pixels = crop_stereo_pixels(sample, baseline)?;
    anchor_l1(&pixels, anchor)
}

fn anchor_l1(a: &StereoAnchor, b: &StereoAnchor) -> Option<f32> {
    if a.left.len() != b.left.len() || a.right.len() != b.right.len() {
        return None;
    }
    let pixels = a.left.len() + a.right.len();
    if pixels == 0 {
        return None;
    }
    let total = a
        .left
        .iter()
        .zip(&b.left)
        .chain(a.right.iter().zip(&b.right))
        .map(|(left, right)| left.abs_diff(*right) as f64)
        .sum::<f64>();
    Some((total / (pixels as f64 * 255.0)) as f32)
}

fn raw_stereo_l1(a: &GeometrySample, b: &GeometrySample, baseline: [MlGeometry; 2]) -> Option<f32> {
    let a = crop_stereo_pixels(a, baseline)?;
    let b = crop_stereo_pixels(b, baseline)?;
    anchor_l1(&a, &b)
}

fn run(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: FitInputs) {
    log(&shared, format!("[load] {}", inputs.model_path.display()));
    let map = match tvm_params::parse_map_bytes(&inputs.model_bytes) {
        Ok(map) => map,
        Err(error) => {
            return fail(
                &shared,
                format!("EyePrediction model parse failed: {error}"),
            )
        }
    };
    let mut net = match EyeNet::new(map) {
        Ok(net) => net,
        Err(error) => {
            return fail(
                &shared,
                format!("EyePrediction model is incompatible: {error}"),
            )
        }
    };
    if cancelled(&shared, &cancel) {
        return;
    }

    // Derive the absolute seed before per-frame adaptive brightness. Geometry must
    // follow the spatial eyelid motion, not a user's changing photometric affine.
    let motion_seed = match estimate_dataset_motion_seed(
        inputs.dataset.as_dataset(),
        inputs.baseline,
        inputs.mirrors,
    ) {
        Ok(estimate) => {
            log(&shared, format!("[motion seed] {}", estimate.reason));
            for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                let value = &estimate.eyes[eye];
                let g = value.geometry;
                log(
                    &shared,
                    format!(
                        "[motion seed {name}] crop {:.3}/{:.3}/{:.3}/{:.3} rot {:+.1} error {:.4}",
                        g.crop_left,
                        g.crop_right,
                        g.crop_top,
                        g.crop_bottom,
                        g.rotate_deg,
                        value.fit_error
                    ),
                );
            }
            Some(estimate)
        }
        Err(message) => {
            log(
                &shared,
                format!("[motion seed skipped] {message}; local ML search remains available"),
            );
            None
        }
    };

    let appearance_seed = match estimate_dataset_appearance_seed(
        inputs.dataset.as_dataset(),
        inputs.baseline,
    ) {
        Ok(estimate) => {
            log(&shared, format!("[appearance seed] {}", estimate.reason));
            for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                let value = &estimate.eyes[eye];
                let descriptor = value.descriptor;
                let g = value.geometry;
                log(
                        &shared,
                        format!(
                            "[appearance seed {name}] pupil {:.1}/{:.1} contrast {:.1} axis {:+.1} spread {:.1}px/{:.1}deg{} crop {:.3}/{:.3}/{:.3}/{:.3} rot {:+.1}",
                            descriptor.pupil_center_px[0],
                            descriptor.pupil_center_px[1],
                            descriptor.pupil_contrast,
                            descriptor.aperture_angle_deg,
                            descriptor.block_center_spread_px,
                            descriptor.block_angle_spread_deg,
                            if descriptor.stereo_recovered {
                                " stereo-recovered"
                            } else {
                                ""
                            },
                            g.crop_left,
                            g.crop_right,
                            g.crop_top,
                            g.crop_bottom,
                            g.rotate_deg,
                        ),
                    );
            }
            Some(estimate)
        }
        Err(message) => {
            log(
                &shared,
                format!("[appearance seed skipped] {message}; local ML search remains available"),
            );
            None
        }
    };

    log(
        &shared,
        "[prepare] applying the live reflection/brightness preprocessing",
    );
    let prepare_total = inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_geometry_scoring_kind(sample.kind))
        .count();
    progress(&shared, "preparing captured frames", 0, prepare_total);
    let mut prepared = Vec::with_capacity(prepare_total);
    let stability = capture_stability_flags(inputs.dataset.samples(), inputs.baseline);
    for note in &stability.notes {
        log(&shared, format!("[capture evidence] {note}"));
    }
    if stability.invalid_static_phases >= 2
        || stability
            .valid_closed_phases
            .iter()
            .any(|count| *count == 0)
    {
        return fail(
            &shared,
            "capture contains invalid static/closed evidence; follow the final pose in each prompt and record again"
                .into(),
        );
    }
    for (index, sample) in inputs.dataset.samples().iter().enumerate() {
        if !is_geometry_scoring_kind(sample.kind) {
            continue;
        }
        if cancel.load(Ordering::Relaxed) {
            cancelled(&shared, &cancel);
            return;
        }
        let (lw, lh) = sample.left_size;
        let (rw, rh) = sample.right_size;
        let left = preprocess::despeckle(&sample.left, lw as usize, lh as usize, &inputs.despeckle);
        let right =
            preprocess::despeckle(&sample.right, rw as usize, rh as usize, &inputs.despeckle);
        let left = preprocess::flatten(&left, lw as usize, lh as usize, &inputs.flatten);
        let right = preprocess::flatten(&right, rw as usize, rh as usize, &inputs.flatten);
        let left = brightness::apply(
            &left,
            sample.brightness_affine[0][0],
            sample.brightness_affine[0][1],
        );
        let right = brightness::apply(
            &right,
            sample.brightness_affine[1][0],
            sample.brightness_affine[1][1],
        );
        prepared.push(PreparedSample {
            kind: sample.kind,
            expected_open: sample.expected_open,
            phase_index: sample.phase_index,
            native_open: sample.native_open,
            native_gaze_deg: sample
                .native_gaze
                .map(|gaze| gaze.and_then(crate::pipeline::gaze_angles_deg)),
            stable: stability.flags[index],
            left,
            right,
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
        let prepared_count = prepared.len();
        if prepared_count % 20 == 0 || prepared_count == prepare_total {
            progress(
                &shared,
                "preparing captured frames",
                prepared_count,
                prepare_total,
            );
        }
    }

    let train1 = stratified_indices(&prepared, false, STAGE1_FRAMES);
    let train2 = stratified_indices(&prepared, false, STAGE2_FRAMES);
    let train3 = stratified_indices(&prepared, false, STAGE3_FRAMES);
    let holdout = stratified_indices(&prepared, true, HOLDOUT_FRAMES);
    if train1.is_empty() || holdout.is_empty() {
        return fail(
            &shared,
            "capture contains no usable train or holdout frames".into(),
        );
    }

    let mut work_total = STAGE1_CANDIDATES * train1.len()
        + STAGE2_CANDIDATES * train2.len()
        + 12 * train3.len()
        + 2 * holdout.len();
    let mut work_done = prepare_total;

    log(&shared, "[search 1/3] 48 bounded quasi-random candidates");
    let mut stage1_params = Vec::with_capacity(STAGE1_CANDIDATES);
    stage1_params.push(SearchParams::default());
    for index in 1..STAGE1_CANDIDATES {
        stage1_params.push(halton_params(index));
    }
    let mut stage1 = evaluate_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train1,
        inputs.baseline,
        inputs.mirrors,
        &stage1_params,
        "coarse search",
        &mut work_done,
        work_total,
    );
    if cancelled(&shared, &cancel) {
        return;
    }
    if stage1.is_empty() {
        return fail(&shared, "coarse search produced no finite candidate".into());
    }
    let Some(baseline1) = stage1
        .iter()
        .find(|entry| entry.params == SearchParams::default())
        .map(|entry| entry.metrics.clone())
    else {
        return fail(
            &shared,
            "the fallback geometry produced a non-finite score; capture was not evaluated".into(),
        );
    };
    if let Some(issue) = capture_quality_issue(&baseline1) {
        return fail(
            &shared,
            format!("capture quality check failed: {issue}; record the guided sequence again"),
        );
    }
    stage1.retain(|entry| admissible(&entry.metrics, &baseline1));
    sort_scored(&mut stage1);
    if stage1.is_empty() {
        return fail(
            &shared,
            "every coarse candidate violated a safety guard".into(),
        );
    }

    log(
        &shared,
        "[search 2/3] top candidates on a larger stratified set",
    );
    let mut stage2_params = vec![SearchParams::default()];
    for entry in &stage1 {
        push_unique(&mut stage2_params, entry.params);
        if stage2_params.len() == STAGE2_CANDIDATES {
            break;
        }
    }
    work_total =
        work_total.saturating_sub((STAGE2_CANDIDATES - stage2_params.len()) * train2.len());
    let mut stage2 = evaluate_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train2,
        inputs.baseline,
        inputs.mirrors,
        &stage2_params,
        "successive halving",
        &mut work_done,
        work_total,
    );
    if cancelled(&shared, &cancel) {
        return;
    }
    let baseline2 = stage2
        .iter()
        .find(|entry| entry.params == SearchParams::default())
        .map(|entry| entry.metrics.clone())
        .unwrap_or_else(|| baseline1.clone());
    stage2.retain(|entry| admissible(&entry.metrics, &baseline2));
    sort_scored(&mut stage2);
    let Some(stage2_best) = stage2.first().cloned() else {
        return fail(
            &shared,
            "no safe candidate survived successive halving".into(),
        );
    };

    log(&shared, "[search 3/3] local coordinate refinement");
    let mut stage3_params = vec![stage2_best.params, SearchParams::default()];
    for axis in 0..5 {
        for direction in [-1.0f32, 1.0] {
            let mut candidate = stage2_best.params;
            candidate.0[axis] += direction * 0.22;
            push_unique(&mut stage3_params, candidate.clamped());
        }
    }
    stage3_params.truncate(12);
    work_total = work_total.saturating_sub((12 - stage3_params.len()) * train3.len());
    let motion_seed_geometry = motion_seed
        .as_ref()
        .filter(|estimate| estimate.search_eligible && estimate.geometry != inputs.baseline)
        .map(|estimate| estimate.geometry);
    if motion_seed_geometry.is_some() {
        work_total += train3.len();
    }
    let appearance_seed_geometry = appearance_seed
        .as_ref()
        .filter(|estimate| estimate.search_eligible && estimate.geometry != inputs.baseline)
        .map(|estimate| estimate.geometry);
    if appearance_seed_geometry.is_some() {
        work_total += train3.len();
    }
    let mut stage3 = evaluate_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train3,
        inputs.baseline,
        inputs.mirrors,
        &stage3_params,
        "local refinement",
        &mut work_done,
        work_total,
    );
    if let Some(geometry) = motion_seed_geometry {
        log(
            &shared,
            "[search 3/3] evaluating the independent motion-derived seed",
        );
        let metrics = evaluate_candidate(
            &mut net,
            &prepared,
            &train3,
            geometry,
            inputs.mirrors,
            &cancel,
        );
        work_done += train3.len();
        progress(
            &shared,
            "motion-seed validation",
            work_done.min(work_total),
            work_total,
        );
        if metrics.score.is_finite() {
            stage3.push(Scored {
                params: SearchParams::default(),
                geometry,
                metrics,
                motion_seed: true,
                appearance_seed: false,
            });
        }
    }
    if let Some(geometry) = appearance_seed_geometry {
        log(
            &shared,
            "[search 3/3] evaluating the independent neutral-appearance seed",
        );
        let metrics = evaluate_candidate(
            &mut net,
            &prepared,
            &train3,
            geometry,
            inputs.mirrors,
            &cancel,
        );
        work_done += train3.len();
        progress(
            &shared,
            "appearance-seed validation",
            work_done.min(work_total),
            work_total,
        );
        if metrics.score.is_finite() {
            stage3.push(Scored {
                params: SearchParams::default(),
                geometry,
                metrics,
                motion_seed: false,
                appearance_seed: true,
            });
        }
    }
    if cancelled(&shared, &cancel) {
        return;
    }
    let baseline_train = stage3
        .iter()
        .find(|entry| {
            entry.params == SearchParams::default() && !entry.motion_seed && !entry.appearance_seed
        })
        .map(|entry| entry.metrics.clone())
        .unwrap_or_else(|| baseline2.clone());
    stage3.retain(|entry| admissible(&entry.metrics, &baseline_train));
    sort_scored(&mut stage3);
    let Some(best) = stage3.first().cloned() else {
        return fail(&shared, "local search produced no safe candidate".into());
    };

    let flat_objective = stage3.get(1).is_some_and(|runner_up| {
        (best.metrics.score - runner_up.metrics.score).abs() < 0.012
            && scored_distance(&best, runner_up) > 0.75
    });

    log(
        &shared,
        "[holdout] comparing winner against the untouched fallback",
    );
    let baseline_holdout = evaluate_candidate(
        &mut net,
        &prepared,
        &holdout,
        inputs.baseline,
        inputs.mirrors,
        &cancel,
    );
    work_done += holdout.len();
    progress(&shared, "holdout validation", work_done, work_total);
    let candidate_holdout = evaluate_candidate(
        &mut net,
        &prepared,
        &holdout,
        best.geometry,
        inputs.mirrors,
        &cancel,
    );
    work_done += holdout.len();
    progress(
        &shared,
        "holdout validation",
        work_done.min(work_total),
        work_total,
    );
    if cancelled(&shared, &cancel) {
        return;
    }

    let (accepted, reason) = acceptance(
        &baseline_train,
        &best.metrics,
        &baseline_holdout,
        &candidate_holdout,
        best.params == SearchParams::default() && !best.motion_seed && !best.appearance_seed,
        flat_objective,
    );
    let improvement = candidate_holdout.score - baseline_holdout.score;
    let result = GeometryFitResult {
        baseline: inputs.baseline,
        candidate: best.geometry,
        baseline_train,
        candidate_train: best.metrics,
        baseline_holdout,
        candidate_holdout,
        holdout_improvement: improvement,
        motion_seed,
        candidate_from_motion_seed: best.motion_seed,
        appearance_seed,
        candidate_from_appearance_seed: best.appearance_seed,
        invalid_static_phases: stability.invalid_static_phases,
        degraded_static_phases: stability.degraded_static_phases,
        valid_closed_phases: stability.valid_closed_phases,
        accepted,
        reason,
    };
    let mut state = lock_shared(&shared);
    state.push(format!(
        "[done] holdout {:.3} -> {:.3}; {}",
        result.baseline_holdout.score,
        result.candidate_holdout.score,
        if result.accepted {
            "candidate accepted"
        } else {
            "fallback retained"
        }
    ));
    let log = state.log.clone();
    state.status = Status::Done { result, log };
}

#[derive(Clone)]
struct AuditCaseSpec {
    name: String,
    geometry: [MlGeometry; 2],
    axis: Option<usize>,
    offset: f32,
}

#[derive(Clone, Copy, Default)]
struct LegacyFoldMetrics {
    valid: bool,
    score: f32,
    span: f32,
    half_position: [f32; 2],
    half_error: f32,
}

struct AuditFoldSignals {
    current_score: f32,
    legacy: LegacyFoldMetrics,
    bimodality: f32,
    slow_curve: [[f32; 10]; 2],
}

/// Compare the current in-app objective with the absolute-span/half-position criteria
/// that originally found the XR5 preset. Exploratory folds use train phases only;
/// untouched holdout remains reserved for the normal frozen candidate evaluation.
/// This is diagnostic only: no geometry is accepted, previewed, or persisted.
fn run_audit(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: FitInputs) {
    log(
        &shared,
        format!("[audit load] {}", inputs.model_path.display()),
    );
    let map = match tvm_params::parse_map_bytes(&inputs.model_bytes) {
        Ok(map) => map,
        Err(error) => {
            return fail(
                &shared,
                format!("EyePrediction model parse failed: {error}"),
            )
        }
    };
    let mut net = match EyeNet::new(map) {
        Ok(net) => net,
        Err(error) => {
            return fail(
                &shared,
                format!("EyePrediction model is incompatible: {error}"),
            )
        }
    };
    if cancelled(&shared, &cancel) {
        return;
    }

    log(
        &shared,
        "[audit prepare] applying the captured deterministic preprocessing",
    );
    let prepare_total = inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_geometry_scoring_kind(sample.kind))
        .count();
    progress(&shared, "preparing audit frames", 0, prepare_total);
    let mut prepared = Vec::with_capacity(prepare_total);
    let stability = capture_stability_flags(inputs.dataset.samples(), inputs.baseline);
    for note in &stability.notes {
        log(&shared, format!("[audit evidence] {note}"));
    }
    if stability.invalid_static_phases >= 2
        || stability
            .valid_closed_phases
            .iter()
            .any(|count| *count == 0)
    {
        return fail(
            &shared,
            "capture contains invalid static/closed evidence; follow the final pose in each prompt and record again"
                .into(),
        );
    }
    for (index, sample) in inputs.dataset.samples().iter().enumerate() {
        if !is_geometry_scoring_kind(sample.kind) {
            continue;
        }
        if cancel.load(Ordering::Relaxed) {
            cancelled(&shared, &cancel);
            return;
        }
        let (lw, lh) = sample.left_size;
        let (rw, rh) = sample.right_size;
        let left = preprocess::despeckle(&sample.left, lw as usize, lh as usize, &inputs.despeckle);
        let right =
            preprocess::despeckle(&sample.right, rw as usize, rh as usize, &inputs.despeckle);
        let left = preprocess::flatten(&left, lw as usize, lh as usize, &inputs.flatten);
        let right = preprocess::flatten(&right, rw as usize, rh as usize, &inputs.flatten);
        let left = brightness::apply(
            &left,
            sample.brightness_affine[0][0],
            sample.brightness_affine[0][1],
        );
        let right = brightness::apply(
            &right,
            sample.brightness_affine[1][0],
            sample.brightness_affine[1][1],
        );
        prepared.push(PreparedSample {
            kind: sample.kind,
            expected_open: sample.expected_open,
            phase_index: sample.phase_index,
            native_open: sample.native_open,
            native_gaze_deg: sample
                .native_gaze
                .map(|gaze| gaze.and_then(crate::pipeline::gaze_angles_deg)),
            stable: stability.flags[index],
            left,
            right,
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
        let prepared_count = prepared.len();
        if prepared_count % 20 == 0 || prepared_count == prepare_total {
            progress(
                &shared,
                "preparing audit frames",
                prepared_count,
                prepare_total,
            );
        }
    }

    let folds = match audit_fold_indices(&prepared) {
        Ok(folds) => folds,
        Err(message) => return fail(&shared, message),
    };
    let specs = audit_case_specs(inputs.baseline);
    let evaluations_per_case = folds.iter().map(Vec::len).sum::<usize>();
    let reference_indices: Vec<_> = prepared
        .iter()
        .enumerate()
        .filter(|(_, sample)| is_geometry_scoring_kind(sample.kind) && !sample.kind.is_holdout())
        .map(|(index, _)| index)
        .collect();
    let work_total = prepare_total + reference_indices.len() + specs.len() * evaluations_per_case;
    let mut work_done = prepare_total;
    progress(
        &shared,
        "validating held-half evidence",
        work_done,
        work_total,
    );
    let (_, reference_observations) = evaluate_candidate_detailed(
        &mut net,
        &prepared,
        &reference_indices,
        inputs.baseline,
        inputs.mirrors,
        &cancel,
    );
    if cancel.load(Ordering::Relaxed) {
        cancelled(&shared, &cancel);
        return;
    }
    work_done += reference_indices.len();
    let half_quality = match half_quality(&reference_observations) {
        Ok(quality) => quality,
        Err(message) => return fail(&shared, message),
    };
    for warning in &half_quality.warnings {
        log(&shared, format!("[audit warning] {warning}"));
    }
    log(
        &shared,
        format!(
            "[half] position L/R {:.3}/{:.3}  spread {:.3}/{:.3}  block delta {:.3}/{:.3}",
            half_quality.position[0],
            half_quality.position[1],
            half_quality.normalized_stddev[0],
            half_quality.normalized_stddev[1],
            half_quality.block_disagreement[0],
            half_quality.block_disagreement[1],
        ),
    );
    progress(&shared, "objective landscape audit", work_done, work_total);
    log(
        &shared,
        format!(
            "[audit] {} geometries x {} blocked folds ({} frame evaluations)",
            specs.len(),
            folds.len(),
            specs.len() * evaluations_per_case
        ),
    );

    let mut cases = Vec::with_capacity(specs.len());
    for (case_index, spec) in specs.iter().enumerate() {
        let mut fold_signals = Vec::with_capacity(folds.len());
        for indices in &folds {
            let (current, observations) = evaluate_candidate_detailed(
                &mut net,
                &prepared,
                indices,
                spec.geometry,
                inputs.mirrors,
                &cancel,
            );
            if cancel.load(Ordering::Relaxed) {
                cancelled(&shared, &cancel);
                return;
            }
            fold_signals.push(AuditFoldSignals {
                current_score: if current.evidence_valid {
                    current.score
                } else {
                    f32::NAN
                },
                legacy: legacy_fold_metrics(&observations),
                bimodality: bimodality_score(&observations),
                slow_curve: slow_close_curve(&observations),
            });
            work_done += indices.len();
            progress(
                &shared,
                "objective landscape audit",
                work_done.min(work_total),
                work_total,
            );
        }
        let current_values: Vec<_> = fold_signals.iter().map(|fold| fold.current_score).collect();
        let legacy_values: Vec<_> = fold_signals
            .iter()
            .map(|fold| {
                if fold.legacy.valid {
                    fold.legacy.score
                } else {
                    f32::NAN
                }
            })
            .collect();
        let span_values: Vec<_> = fold_signals
            .iter()
            .map(|fold| {
                if fold.legacy.valid {
                    fold.legacy.span
                } else {
                    f32::NAN
                }
            })
            .collect();
        let half_values: Vec<_> = fold_signals
            .iter()
            .map(|fold| {
                if fold.legacy.valid {
                    fold.legacy.half_error
                } else {
                    f32::NAN
                }
            })
            .collect();
        let half_position = std::array::from_fn(|eye| {
            let values: Vec<_> = fold_signals
                .iter()
                .map(|fold| {
                    if fold.legacy.valid {
                        fold.legacy.half_position[eye]
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            audit_stat(&values)
        });
        let bimodal_values: Vec<_> = fold_signals.iter().map(|fold| fold.bimodality).collect();
        cases.push(GeometryAuditCase {
            name: spec.name.clone(),
            geometry: spec.geometry,
            current_score: audit_stat(&current_values),
            legacy_score: audit_stat(&legacy_values),
            absolute_span: audit_stat(&span_values),
            half_position,
            half_error: audit_stat(&half_values),
            bimodality: audit_stat(&bimodal_values),
            reproducibility: slow_curve_reproducibility(&fold_signals),
            confident_wrong: false,
        });
        log(
            &shared,
            format!(
                "[audit] {}/{} {}  current {:.3}  legacy {:.3}",
                case_index + 1,
                specs.len(),
                spec.name,
                cases
                    .last()
                    .map_or(f32::NAN, |case| case.current_score.mean),
                cases.last().map_or(f32::NAN, |case| case.legacy_score.mean),
            ),
        );
    }

    if cases.is_empty()
        || !cases[0].current_score.mean.is_finite()
        || !cases[0].legacy_score.mean.is_finite()
    {
        return fail(
            &shared,
            "objective audit could not obtain valid reference evidence in every required phase"
                .into(),
        );
    }
    let reference = cases[0].clone();
    let current_band = 2.0 * reference.current_score.stddev.max(0.005);
    let legacy_band = 2.0 * reference.legacy_score.stddev.max(0.02);
    let bimodal_band = 2.0 * reference.bimodality.stddev.max(0.01);
    for case in cases.iter_mut().skip(1) {
        let current_gain = case.current_score.mean - reference.current_score.mean;
        let legacy_loss = reference.legacy_score.mean - case.legacy_score.mean;
        let bimodal_loss = reference.bimodality.mean - case.bimodality.mean;
        let reproducibility_loss = reference.reproducibility.is_finite()
            && case.reproducibility.is_finite()
            && reference.reproducibility - case.reproducibility > 0.08;
        case.confident_wrong = current_gain > current_band
            && (legacy_loss > legacy_band || bimodal_loss > bimodal_band || reproducibility_loss);
    }

    let current_best = best_audit_case(&cases, |case| case.current_score.mean);
    let legacy_best = best_audit_case(&cases, |case| case.legacy_score.mean);
    let confident_wrong_count = cases.iter().filter(|case| case.confident_wrong).count();
    let axis_names = ["inward", "vertical", "size", "scaleY", "rotation"];
    let mut edge_drift_axes = Vec::new();
    for (axis, name) in axis_names.iter().enumerate() {
        let mut best_index = 0usize;
        for (index, spec) in specs.iter().enumerate().skip(1) {
            if spec.axis == Some(axis)
                && cases[index].current_score.mean > cases[best_index].current_score.mean
            {
                best_index = index;
            }
        }
        if specs[best_index].axis == Some(axis)
            && specs[best_index].offset.abs() >= 0.99
            && cases[best_index].current_score.mean - reference.current_score.mean > current_band
        {
            edge_drift_axes.push((*name).to_string());
        }
    }

    let evidence_ready = reference.half_error.mean <= 0.12
        && reference.half_error.stddev <= 0.08
        && reference.reproducibility >= 0.80
        && reference
            .half_position
            .iter()
            .all(|position| (0.30..=0.70).contains(&position.mean) && position.stddev <= 0.06)
        && half_quality
            .block_disagreement
            .iter()
            .all(|difference| *difference <= 0.15);
    let reason = if !evidence_ready {
        "EVIDENCE WEAK: explicit HALF was valid enough to score, but its fold or block repeatability did not meet the v2 decision threshold. Repeat the recording before judging the objective.".into()
    } else if confident_wrong_count > 0 {
        format!(
            "NO-GO: {confident_wrong_count} geometry probe(s) beat the active geometry on the in-app objective beyond its fold noise, while legacy or unsupervised evidence regressed."
        )
    } else if !edge_drift_axes.is_empty() {
        format!(
            "NO-GO: the in-app objective is still improving at the audit boundary on {}.",
            edge_drift_axes.join(", ")
        )
    } else if current_best != legacy_best {
        "INCONCLUSIVE: the in-app and legacy objectives prefer different local probes, but no confident-wrong case cleared the fold-noise guard.".into()
    } else {
        "CONSISTENT IN THIS CAPTURE: no confident local objective mismatch was found. This audit alone does not validate automatic fitting for other users.".into()
    };
    let result = GeometryAuditResult {
        cases,
        current_best,
        legacy_best,
        confident_wrong_count,
        edge_drift_axes,
        half_quality,
        evidence_ready,
        reason,
    };
    let mut state = lock_shared(&shared);
    state.push(format!(
        "[audit done] current best={}  legacy best={}  confident-wrong={}",
        result.cases[result.current_best].name,
        result.cases[result.legacy_best].name,
        result.confident_wrong_count
    ));
    let log = state.log.clone();
    state.status = Status::AuditDone { result, log };
}

fn audit_case_specs(baseline: [MlGeometry; 2]) -> Vec<AuditCaseSpec> {
    let axis_names = ["inward", "vertical", "size", "scaleY", "rotation"];
    let mut cases = vec![AuditCaseSpec {
        name: "active reference".into(),
        geometry: baseline,
        axis: None,
        offset: 0.0,
    }];
    for (axis, name) in axis_names.iter().enumerate() {
        for offset in [-1.0f32, -0.5, 0.5, 1.0] {
            let mut params = [0.0; 5];
            params[axis] = offset;
            push_audit_case(
                &mut cases,
                AuditCaseSpec {
                    name: format!("{name} {offset:+.1}"),
                    geometry: geometry_from_params(baseline, SearchParams(params)),
                    axis: Some(axis),
                    offset,
                },
            );
        }
    }

    for (axis, name) in [(0usize, "inward"), (4usize, "rotation")] {
        let mut params = [0.0; 5];
        params[axis] = 0.5;
        let symmetric = geometry_from_params(baseline, SearchParams(params));
        for (eye, eye_name) in [(0usize, "L"), (1usize, "R")] {
            let mut geometry = baseline;
            geometry[eye] = symmetric[eye];
            push_audit_case(
                &mut cases,
                AuditCaseSpec {
                    name: format!("{eye_name}-only {name} +0.5"),
                    geometry,
                    axis: None,
                    offset: 0.5,
                },
            );
        }
    }

    for (name, inner, vertical) in [
        ("legacy neighbour inner .45", Some(0.45), None),
        ("legacy neighbour vertical .10", None, Some(0.10)),
    ] {
        let mut geometry = baseline;
        if let Some(inner) = inner {
            geometry[0].crop_right = inner;
            geometry[1].crop_left = inner;
        }
        if let Some(vertical) = vertical {
            for eye in &mut geometry {
                eye.crop_top = vertical;
                eye.crop_bottom = vertical;
            }
        }
        push_audit_case(
            &mut cases,
            AuditCaseSpec {
                name: name.into(),
                geometry,
                axis: None,
                offset: 0.0,
            },
        );
    }
    cases
}

fn push_audit_case(cases: &mut Vec<AuditCaseSpec>, candidate: AuditCaseSpec) {
    if candidate.geometry[0].crop_right + 1e-6 < XR5_MIN_INNER_CROP
        || candidate.geometry[1].crop_left + 1e-6 < XR5_MIN_INNER_CROP
    {
        return;
    }
    if !cases
        .iter()
        .any(|case| geometry_nearly_equal(case.geometry, candidate.geometry))
    {
        cases.push(candidate);
    }
}

fn geometry_nearly_equal(left: [MlGeometry; 2], right: [MlGeometry; 2]) -> bool {
    left.iter().zip(right).all(|(left, right)| {
        (left.crop_left - right.crop_left).abs() <= 1e-6
            && (left.crop_right - right.crop_right).abs() <= 1e-6
            && (left.crop_top - right.crop_top).abs() <= 1e-6
            && (left.crop_bottom - right.crop_bottom).abs() <= 1e-6
            && (left.scale_x - right.scale_x).abs() <= 1e-6
            && (left.scale_y - right.scale_y).abs() <= 1e-6
            && (left.rotate_deg - right.rotate_deg).abs() <= 1e-6
            && left.mirror_h == right.mirror_h
    })
}

fn audit_fold_indices(samples: &[PreparedSample]) -> Result<Vec<Vec<usize>>, String> {
    let families = [
        SampleFamily::Neutral,
        SampleFamily::HalfOpen,
        SampleFamily::GazeSweep,
        SampleFamily::SlowClose,
        SampleFamily::NaturalBlinks,
        SampleFamily::Closed,
    ];
    let mut pools = vec![vec![Vec::<usize>::new(); AUDIT_FOLDS]; families.len()];
    for (family_index, family) in families.iter().copied().enumerate() {
        let mut block_number = 0usize;
        let mut slow_band_blocks = [0usize; 3];
        let mut cursor = 0usize;
        while cursor < samples.len() {
            if !is_geometry_scoring_kind(samples[cursor].kind)
                || samples[cursor].kind.is_holdout()
                || samples[cursor].kind.family() != family
            {
                cursor += 1;
                continue;
            }
            let kind = samples[cursor].kind;
            let run_start = cursor;
            while cursor < samples.len() && samples[cursor].kind == kind {
                cursor += 1;
            }
            let run_end = cursor;
            let mut block_start = run_start;
            while block_start < run_end {
                let block_end = (block_start + AUDIT_BLOCK_FRAMES).min(run_end);
                if block_end.saturating_sub(block_start) > 2 * AUDIT_GUARD_FRAMES {
                    let fold = if family == SampleFamily::SlowClose {
                        let (sum, count) = samples[block_start..block_end]
                            .iter()
                            .filter_map(|sample| sample.expected_open)
                            .fold((0.0f32, 0usize), |(sum, count), value| {
                                (sum + value, count + 1)
                            });
                        let mean = if count == 0 { 0.5 } else { sum / count as f32 };
                        let band = if mean < 0.33 {
                            0
                        } else if mean > 0.67 {
                            2
                        } else {
                            1
                        };
                        let fold = slow_band_blocks[band] % AUDIT_FOLDS;
                        slow_band_blocks[band] += 1;
                        fold
                    } else {
                        let fold = block_number % AUDIT_FOLDS;
                        block_number += 1;
                        fold
                    };
                    pools[family_index][fold].extend(
                        (block_start + AUDIT_GUARD_FRAMES)..(block_end - AUDIT_GUARD_FRAMES),
                    );
                }
                block_start = block_end;
            }
        }
    }

    let mut folds = vec![Vec::new(); AUDIT_FOLDS];
    for fold in 0..AUDIT_FOLDS {
        for (family_index, family) in families.iter().enumerate() {
            let pool = &pools[family_index][fold];
            if pool.len() < 10 {
                return Err(format!(
                    "objective audit fold {} has only {} usable {:?} frames; record again",
                    fold + 1,
                    pool.len(),
                    family
                ));
            }
            let take = AUDIT_FRAMES_PER_FAMILY_FOLD.min(pool.len());
            for position in 0..take {
                folds[fold].push(pool[position * pool.len() / take]);
            }
        }
        folds[fold].sort_unstable();
        folds[fold].dedup();
    }
    Ok(folds)
}

fn legacy_fold_metrics(observations: &[Observation]) -> LegacyFoldMetrics {
    let mut levels = [[0.0f32; 3]; 2];
    let mut spreads = [[0.0f32; 3]; 2];
    for eye in 0..2 {
        let groups = [
            values(observations, eye, |observation| {
                (observation.kind.family() == SampleFamily::Closed && observation.stable)
                    || (observation.kind.family() == SampleFamily::SlowClose
                        && observation
                            .expected_open
                            .is_some_and(|target| target <= 0.20))
            }),
            values(observations, eye, |observation| {
                observation.kind.family() == SampleFamily::HalfOpen && observation.stable
            }),
            values(observations, eye, |observation| {
                (observation.kind.family() == SampleFamily::Neutral && observation.stable)
                    || (observation.kind.family() == SampleFamily::SlowClose
                        && observation
                            .expected_open
                            .is_some_and(|target| target >= 0.80))
            }),
        ];
        if groups.iter().any(|group| group.len() < 3) {
            return LegacyFoldMetrics::default();
        }
        for level in 0..3 {
            let (mean, variance) = mean_variance(&groups[level]);
            levels[eye][level] = mean;
            spreads[eye][level] = variance.sqrt();
        }
    }
    let spans = [levels[0][2] - levels[0][0], levels[1][2] - levels[1][0]];
    if spans.iter().any(|span| !span.is_finite() || *span <= 0.001) {
        return LegacyFoldMetrics::default();
    }
    let span = average(spans);
    let half_position = std::array::from_fn(|eye| (levels[eye][1] - levels[eye][0]) / spans[eye]);
    let half_error = half_position
        .iter()
        .map(|position| (*position - 0.5).abs())
        .sum::<f32>()
        * 0.5;
    let ordering = (0..2)
        .map(|eye| {
            let ordered = (levels[eye][0] < levels[eye][1]) as u8 as f32
                + (levels[eye][1] < levels[eye][2]) as u8 as f32
                + (levels[eye][0] < levels[eye][2]) as u8 as f32;
            ordered / 3.0
        })
        .sum::<f32>()
        * 0.5;
    let jitter = (0..2)
        .map(|eye| spreads[eye].iter().sum::<f32>() / (3.0 * spans[eye]))
        .sum::<f32>()
        * 0.5;
    let lr_error = (0..3)
        .map(|level| (levels[0][level] - levels[1][level]).abs())
        .sum::<f32>()
        / (3.0 * span.max(0.001));
    let score = 8.0 * span + 0.8 * ordering
        - 0.8 * half_error.min(3.0)
        - 0.35 * jitter.min(3.0)
        - 0.25 * lr_error.min(3.0);
    LegacyFoldMetrics {
        valid: score.is_finite(),
        score,
        span,
        half_position,
        half_error,
    }
}

fn half_quality(observations: &[Observation]) -> Result<HalfQuality, String> {
    let mut quality = HalfQuality::default();
    for eye in 0..2 {
        let open = values(observations, eye, |observation| {
            observation.kind == SampleKind::Neutral && observation.stable
        });
        let closed = values(observations, eye, |observation| {
            observation.kind == SampleKind::Closed && observation.stable
        });
        let half = values(observations, eye, |observation| {
            observation.kind == SampleKind::HalfOpen && observation.stable
        });
        if open.len() < 5 || closed.len() < 5 || half.len() < 5 {
            return Err(format!(
                "HALF evidence is incomplete for eye {} (open={}, half={}, closed={}); record again",
                if eye == 0 { "L" } else { "R" },
                open.len(),
                half.len(),
                closed.len()
            ));
        }
        let open_mean = mean_variance(&open).0;
        let closed_mean = mean_variance(&closed).0;
        let (half_mean, half_variance) = mean_variance(&half);
        let span = open_mean - closed_mean;
        if !span.is_finite() || span < 0.05 {
            return Err(format!(
                "HALF evidence has too little open/closed model span for eye {} ({span:.3}, need >=0.050); check the image alignment and record again",
                if eye == 0 { "L" } else { "R" }
            ));
        }
        let position = (half_mean - closed_mean) / span;
        let normalized_stddev = half_variance.sqrt() / span;
        let mut block_values = BTreeMap::<usize, Vec<f32>>::new();
        for observation in observations {
            if observation.kind == SampleKind::HalfOpen
                && observation.stable
                && observation.open[eye].is_finite()
            {
                block_values
                    .entry(observation.phase_index)
                    .or_default()
                    .push(observation.open[eye]);
            }
        }
        let block_positions: Vec<_> = block_values
            .values()
            .filter(|block| block.len() >= 3)
            .map(|block| (mean_variance(block).0 - closed_mean) / span)
            .collect();
        if block_positions.len() < 2 {
            return Err(format!(
                "HALF evidence did not retain two independent blocks for eye {}; record again",
                if eye == 0 { "L" } else { "R" }
            ));
        }
        let block_disagreement = block_positions
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or(position)
            - block_positions
                .iter()
                .copied()
                .reduce(f32::min)
                .unwrap_or(position);
        quality.position[eye] = position;
        quality.normalized_stddev[eye] = normalized_stddev;
        quality.block_disagreement[eye] = block_disagreement;
        if !(0.25..=0.75).contains(&position) {
            return Err(format!(
                "HALF pose for eye {} landed at {position:.3} of the open/closed span (need 0.25..0.75); hold a clearer halfway pose and record again",
                if eye == 0 { "L" } else { "R" }
            ));
        }
        if normalized_stddev >= 0.35 {
            return Err(format!(
                "HALF pose for eye {} was not steady (normalized spread {normalized_stddev:.3}, need <0.350); record again",
                if eye == 0 { "L" } else { "R" }
            ));
        }
        if block_disagreement > 0.20 {
            return Err(format!(
                "the two HALF poses disagreed for eye {} by {block_disagreement:.3} of the open/closed span (need <=0.200); record again",
                if eye == 0 { "L" } else { "R" }
            ));
        }

        let half_observations: Vec<_> = observations
            .iter()
            .filter(|observation| observation.kind == SampleKind::HalfOpen && observation.stable)
            .collect();
        let native_half: Vec<_> = half_observations
            .iter()
            .filter_map(|observation| observation.native_open[eye])
            .filter(|value| value.is_finite())
            .collect();
        let native_coverage = native_half.len() as f32 / half_observations.len().max(1) as f32;
        quality.native_coverage[eye] = native_coverage;
        if native_coverage >= 0.80 {
            let native_values = |kind: SampleKind| {
                observations
                    .iter()
                    .filter(|observation| observation.kind == kind && observation.stable)
                    .filter_map(|observation| observation.native_open[eye])
                    .filter(|value| value.is_finite())
                    .collect::<Vec<_>>()
            };
            let native_open = native_values(SampleKind::Neutral);
            let native_closed = native_values(SampleKind::Closed);
            if native_open.len() >= 3 && native_closed.len() >= 3 && native_half.len() >= 3 {
                let native_open_mean = mean_variance(&native_open).0;
                let native_closed_mean = mean_variance(&native_closed).0;
                let native_span = native_open_mean - native_closed_mean;
                if native_span > 0.05 {
                    let native_position =
                        (mean_variance(&native_half).0 - native_closed_mean) / native_span;
                    if !(0.0..=1.0).contains(&native_position)
                        || (native_position - position).abs() > 0.30
                    {
                        quality.warnings.push(format!(
                            "native Tobii openness and EyeNet disagree on eye {} HALF position ({native_position:.3} vs {position:.3}); result remains diagnostic only",
                            if eye == 0 { "L" } else { "R" }
                        ));
                    }
                }
            }
        }
    }
    Ok(quality)
}

fn bimodality_score(observations: &[Observation]) -> f32 {
    let mut score = 0.0;
    for eye in 0..2 {
        let values: Vec<_> = observations
            .iter()
            .filter(|observation| {
                matches!(
                    observation.kind.family(),
                    SampleFamily::Neutral | SampleFamily::SlowClose | SampleFamily::Closed
                ) && observation.open[eye].is_finite()
            })
            .map(|observation| observation.open[eye])
            .collect();
        let Some(value) = one_dimensional_silhouette(&values) else {
            return f32::NAN;
        };
        score += value;
    }
    score * 0.5
}

fn one_dimensional_silhouette(values: &[f32]) -> Option<f32> {
    if values.len() < 10 {
        return None;
    }
    let mut low = values.iter().copied().reduce(f32::min)?;
    let mut high = values.iter().copied().reduce(f32::max)?;
    if !low.is_finite() || !high.is_finite() || high - low <= 1e-5 {
        return None;
    }
    let mut assignment = vec![false; values.len()];
    for _ in 0..16 {
        for (index, value) in values.iter().enumerate() {
            assignment[index] = (*value - high).abs() < (*value - low).abs();
        }
        let mut sums = [0.0f32; 2];
        let mut counts = [0usize; 2];
        for (value, upper) in values.iter().zip(&assignment) {
            let cluster = usize::from(*upper);
            sums[cluster] += *value;
            counts[cluster] += 1;
        }
        if counts.iter().any(|count| *count < 2) {
            return None;
        }
        low = sums[0] / counts[0] as f32;
        high = sums[1] / counts[1] as f32;
    }
    let mut total = 0.0;
    for (index, value) in values.iter().enumerate() {
        let own = assignment[index];
        let mut own_sum = 0.0;
        let mut own_count = 0usize;
        let mut other_sum = 0.0;
        let mut other_count = 0usize;
        for (other_index, other) in values.iter().enumerate() {
            if other_index == index {
                continue;
            }
            if assignment[other_index] == own {
                own_sum += (*value - *other).abs();
                own_count += 1;
            } else {
                other_sum += (*value - *other).abs();
                other_count += 1;
            }
        }
        if own_count == 0 || other_count == 0 {
            return None;
        }
        let within = own_sum / own_count as f32;
        let between = other_sum / other_count as f32;
        total += (between - within) / within.max(between).max(1e-6);
    }
    Some((total / values.len() as f32).clamp(-1.0, 1.0))
}

fn slow_close_curve(observations: &[Observation]) -> [[f32; 10]; 2] {
    let mut curve = [[f32::NAN; 10]; 2];
    for (eye, eye_curve) in curve.iter_mut().enumerate() {
        let mut sums = [0.0f32; 10];
        let mut counts = [0usize; 10];
        for observation in observations {
            if observation.kind.family() != SampleFamily::SlowClose {
                continue;
            }
            let (Some(target), value) = (observation.expected_open, observation.open[eye]) else {
                continue;
            };
            if !value.is_finite() {
                continue;
            }
            let bin = (target.clamp(0.0, 0.999_999) * 10.0) as usize;
            sums[bin] += value;
            counts[bin] += 1;
        }
        for bin in 0..10 {
            if counts[bin] > 0 {
                eye_curve[bin] = sums[bin] / counts[bin] as f32;
            }
        }
    }
    curve
}

fn slow_curve_reproducibility(folds: &[AuditFoldSignals]) -> f32 {
    let mut correlations = Vec::new();
    for eye in 0..2 {
        for left in 0..folds.len() {
            for right in left + 1..folds.len() {
                let mut a = Vec::new();
                let mut b = Vec::new();
                for bin in 0..10 {
                    let av = folds[left].slow_curve[eye][bin];
                    let bv = folds[right].slow_curve[eye][bin];
                    if av.is_finite() && bv.is_finite() {
                        a.push(av);
                        b.push(bv);
                    }
                }
                if a.len() >= 4 {
                    correlations.push(pearson(&a, &b).clamp(-1.0, 1.0));
                }
            }
        }
    }
    if correlations.is_empty() {
        f32::NAN
    } else {
        correlations.iter().sum::<f32>() / correlations.len() as f32
    }
}

fn audit_stat(values: &[f32]) -> AuditStat {
    let finite: Vec<_> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return AuditStat {
            mean: f32::NAN,
            stddev: f32::NAN,
        };
    }
    let (mean, variance) = mean_variance(&finite);
    AuditStat {
        mean,
        stddev: variance.sqrt(),
    }
}

fn best_audit_case(
    cases: &[GeometryAuditCase],
    value: impl Fn(&GeometryAuditCase) -> f32,
) -> usize {
    cases
        .iter()
        .enumerate()
        .filter(|(_, case)| value(case).is_finite())
        .max_by(|(_, left), (_, right)| {
            value(left)
                .partial_cmp(&value(right))
                .unwrap_or(CmpOrdering::Equal)
        })
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn estimate_dataset_appearance_seed(
    dataset: &GeometryDataset,
    baseline: [MlGeometry; 2],
) -> Result<AppearanceGeometryEstimate, String> {
    // Keep the spatial detector independent from adaptive brightness and user filter
    // tuning. The normal ML evaluation below still uses the captured live pipeline.
    let despeckle = DespeckleParams::default();
    let flatten = FlattenParams::default();
    let selected = dataset
        .samples
        .iter()
        .filter(|sample| !sample.kind.is_holdout() && sample.kind == SampleKind::Neutral)
        .collect::<Vec<_>>();
    let left_owned = selected
        .iter()
        .map(|sample| {
            let (width, height) = sample.left_size;
            let pixels =
                preprocess::despeckle(&sample.left, width as usize, height as usize, &despeckle);
            let pixels = preprocess::flatten(&pixels, width as usize, height as usize, &flatten);
            (sample.phase_index, width, height, pixels)
        })
        .collect::<Vec<_>>();
    let right_owned = selected
        .iter()
        .map(|sample| {
            let (width, height) = sample.right_size;
            let pixels =
                preprocess::despeckle(&sample.right, width as usize, height as usize, &despeckle);
            let pixels = preprocess::flatten(&pixels, width as usize, height as usize, &flatten);
            (sample.phase_index, width, height, pixels)
        })
        .collect::<Vec<_>>();
    let left = left_owned
        .iter()
        .map(|(group, width, height, pixels)| MotionFrame {
            group: *group,
            width: *width,
            height: *height,
            pixels,
        })
        .collect::<Vec<_>>();
    let right = right_owned
        .iter()
        .map(|(group, width, height, pixels)| MotionFrame {
            group: *group,
            width: *width,
            height: *height,
            pixels,
        })
        .collect::<Vec<_>>();
    estimate_appearance_geometry(&left, &right, baseline)
}

fn estimate_dataset_motion_seed(
    dataset: &GeometryDataset,
    baseline: [MlGeometry; 2],
    mirrors: [bool; 2],
) -> Result<MotionGeometryEstimate, String> {
    // The canonical descriptor was defined under these fixed defaults. Do not let a
    // user's active brightness/flatten tuning move the coordinate system we are trying
    // to recover; the ordinary ML search still evaluates with the captured live filters.
    let despeckle = DespeckleParams::default();
    let flatten = FlattenParams::default();
    let selected = dataset
        .samples
        .iter()
        .filter(|sample| {
            !sample.kind.is_holdout() && sample.kind.family() == SampleFamily::NaturalBlinks
        })
        .collect::<Vec<_>>();
    let left_owned = selected
        .iter()
        .map(|sample| {
            let (width, height) = sample.left_size;
            let pixels =
                preprocess::despeckle(&sample.left, width as usize, height as usize, &despeckle);
            let pixels = preprocess::flatten(&pixels, width as usize, height as usize, &flatten);
            (sample.phase_index, width, height, pixels)
        })
        .collect::<Vec<_>>();
    let right_owned = selected
        .iter()
        .map(|sample| {
            let (width, height) = sample.right_size;
            let pixels =
                preprocess::despeckle(&sample.right, width as usize, height as usize, &despeckle);
            let pixels = preprocess::flatten(&pixels, width as usize, height as usize, &flatten);
            (sample.phase_index, width, height, pixels)
        })
        .collect::<Vec<_>>();
    let left = left_owned
        .iter()
        .map(|(group, width, height, pixels)| MotionFrame {
            group: *group,
            width: *width,
            height: *height,
            pixels,
        })
        .collect::<Vec<_>>();
    let right = right_owned
        .iter()
        .map(|(group, width, height, pixels)| MotionFrame {
            group: *group,
            width: *width,
            height: *height,
            pixels,
        })
        .collect::<Vec<_>>();
    estimate_motion_geometry(&left, &right, baseline, mirrors)
}

fn scored_distance(left: &Scored, right: &Scored) -> f32 {
    if !left.motion_seed && !right.motion_seed && !left.appearance_seed && !right.appearance_seed {
        return left.params.distance(right.params);
    }
    let mut squared = 0.0f32;
    for eye in 0..2 {
        let a = left.geometry[eye];
        let b = right.geometry[eye];
        let aw = 1.0 - a.crop_left - a.crop_right;
        let bw = 1.0 - b.crop_left - b.crop_right;
        let ah = 1.0 - a.crop_top - a.crop_bottom;
        let bh = 1.0 - b.crop_top - b.crop_bottom;
        let acx = a.crop_left + aw * 0.5;
        let bcx = b.crop_left + bw * 0.5;
        let acy = a.crop_top + ah * 0.5;
        let bcy = b.crop_top + bh * 0.5;
        squared += ((acx - bcx) / 0.08).powi(2);
        squared += ((acy - bcy) / 0.08).powi(2);
        squared += ((aw - bw) / 0.15).powi(2);
        squared += ((ah - bh) / 0.15).powi(2);
        squared += ((a.rotate_deg - b.rotate_deg) / 8.0).powi(2);
    }
    (squared / 2.0).sqrt()
}

#[allow(clippy::too_many_arguments)]
fn evaluate_set(
    shared: &Arc<Mutex<Shared>>,
    cancel: &Arc<AtomicBool>,
    net: &mut EyeNet,
    samples: &[PreparedSample],
    indices: &[usize],
    baseline: [MlGeometry; 2],
    mirrors: [bool; 2],
    candidates: &[SearchParams],
    stage: &str,
    work_done: &mut usize,
    work_total: usize,
) -> Vec<Scored> {
    let mut scored = Vec::with_capacity(candidates.len());
    for (index, params) in candidates.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let geometry = geometry_from_params(baseline, params);
        let metrics = evaluate_candidate(net, samples, indices, geometry, mirrors, cancel);
        *work_done += indices.len();
        progress(shared, stage, (*work_done).min(work_total), work_total);
        if index % 8 == 0 || index + 1 == candidates.len() {
            log(
                shared,
                format!(
                    "[{stage}] {}/{} score {:.3}",
                    index + 1,
                    candidates.len(),
                    metrics.score
                ),
            );
        }
        if metrics.score.is_finite() {
            scored.push(Scored {
                params,
                geometry,
                metrics,
                motion_seed: false,
                appearance_seed: false,
            });
        }
    }
    scored
}

fn evaluate_candidate(
    net: &mut EyeNet,
    samples: &[PreparedSample],
    indices: &[usize],
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    cancel: &AtomicBool,
) -> GeometryMetrics {
    evaluate_candidate_detailed(net, samples, indices, geometry, mirrors, cancel).0
}

/// Research-only deterministic replay seam for comparing photometric hypotheses on a
/// completed XR5 recording. It deliberately reuses the production stability labels,
/// preprocessing, EyeNet inference and metrics so an offline result cannot be caused by
/// a second scorer. The extra affine is applied after the affine captured by the app.
#[cfg(feature = "research-synthetic-eye-lab")]
#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialGainField {
    pub horizontal: f32,
    pub vertical: f32,
    pub horizontal_curve: f32,
    pub vertical_curve: f32,
}

/// Bounded, low-dimensional full-frame coordinate warp for offline XR5 research.
///
/// Coordinates are normalized to `[-1, 1]` over the complete captured eye frame,
/// before any candidate geometry is applied. `vertical_bow` is the zero-mean
/// quadratic vertical displacement along the horizontal axis; `radial_k1` is the
/// first radial distortion coefficient. Translation, rotation and linear scale stay
/// in [`MlGeometry`] instead of being duplicated here.
#[cfg(feature = "research-synthetic-eye-lab")]
#[derive(Clone, Copy, Debug, Default)]
pub struct ResearchCoordinateWarp {
    /// Vertical inverse-sampling displacement `b * (x^2 - 1/3)` in normalized
    /// full-frame coordinates.
    pub vertical_bow: f32,
    /// First inverse radial coefficient in `source = destination * (1 + k1*r^2)`.
    pub radial_k1: f32,
}

#[cfg(feature = "research-synthetic-eye-lab")]
impl ResearchCoordinateWarp {
    /// Maximum absolute normalized quadratic bow coefficient used by replay.
    pub const MAX_VERTICAL_BOW: f32 = 0.12;
    /// Maximum absolute first-order radial coefficient used by replay.
    pub const MAX_RADIAL_K1: f32 = 0.10;
}

/// Per-frame output from the exact production preprocessing and EyeNet path.
///
/// This is exposed only to offline research tools.  `sample_index` always refers to
/// the original [`GeometryDataset`] order, so callers can join candidate-independent
/// raw-image measurements without relying on timing or row position.
#[cfg(feature = "research-synthetic-eye-lab")]
#[derive(Clone, Copy, Debug)]
pub struct ResearchObservation {
    pub sample_index: usize,
    pub kind: SampleKind,
    pub expected_open: Option<f32>,
    pub phase_index: usize,
    pub stable: bool,
    pub native_open: [Option<f32>; 2],
    pub native_gaze_deg: [Option<[f32; 2]>; 2],
    pub presence: f32,
    pub open: [f32; 2],
    pub squeeze: [f32; 2],
}

/// Detailed research replay result. Metrics and observations come from the same
/// inference pass, and stability is computed once from raw pixels and the immutable
/// capture geometry. Optional photometric interventions occur only after the captured
/// adaptive-brightness affine; they describe a proposed research seam, not the current
/// production preprocessing order.
#[cfg(feature = "research-synthetic-eye-lab")]
#[derive(Clone, Debug)]
pub struct ResearchReplay {
    pub train: GeometryMetrics,
    pub holdout: GeometryMetrics,
    pub stability: StabilityReport,
    pub observations: Vec<ResearchObservation>,
}

#[cfg(feature = "research-synthetic-eye-lab")]
pub fn research_stability_report(
    dataset: &GeometryDataset,
    stability_baseline: [MlGeometry; 2],
) -> StabilityReport {
    capture_stability_flags(&dataset.samples, stability_baseline)
}

#[cfg(feature = "research-synthetic-eye-lab")]
pub fn research_evaluate_photometric(
    net: &mut EyeNet,
    dataset: &GeometryDataset,
    stability_baseline: [MlGeometry; 2],
    stability: &StabilityReport,
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    despeckle: DespeckleParams,
    captured_flatten: FlattenParams,
    post_normalization_flatten: FlattenParams,
    extra_affine: [[f32; 2]; 2],
    gain_field: [Option<SpatialGainField>; 2],
    coordinate_warp: [Option<ResearchCoordinateWarp>; 2],
) -> (GeometryMetrics, GeometryMetrics) {
    let prepared = research_prepare_samples(
        dataset,
        stability_baseline,
        stability,
        despeckle,
        captured_flatten,
        post_normalization_flatten,
        extra_affine,
        gain_field,
        coordinate_warp,
    );
    let train = prepared
        .iter()
        .enumerate()
        .filter(|(_, sample)| !sample.kind.is_holdout())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let holdout = prepared
        .iter()
        .enumerate()
        .filter(|(_, sample)| is_geometry_scoring_kind(sample.kind) && sample.kind.is_holdout())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    let metrics = (
        evaluate_candidate(net, &prepared, &train, geometry, mirrors, &cancel),
        evaluate_candidate(net, &prepared, &holdout, geometry, mirrors, &cancel),
    );
    metrics
}

/// Detailed variant of [`research_evaluate_photometric`].  It is intentionally kept
/// behind the research feature so production code cannot make runtime decisions from
/// uncalibrated landmark correlations.
#[cfg(feature = "research-synthetic-eye-lab")]
#[allow(clippy::too_many_arguments)]
pub fn research_replay_detailed(
    net: &mut EyeNet,
    dataset: &GeometryDataset,
    stability_baseline: [MlGeometry; 2],
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    despeckle: DespeckleParams,
    captured_flatten: FlattenParams,
    post_normalization_flatten: FlattenParams,
    extra_affine: [[f32; 2]; 2],
    gain_field: [Option<SpatialGainField>; 2],
    coordinate_warp: [Option<ResearchCoordinateWarp>; 2],
) -> ResearchReplay {
    let stability = capture_stability_flags(&dataset.samples, stability_baseline);
    let prepared = research_prepare_samples(
        dataset,
        stability_baseline,
        &stability,
        despeckle,
        captured_flatten,
        post_normalization_flatten,
        extra_affine,
        gain_field,
        coordinate_warp,
    );
    let train_indices = prepared
        .iter()
        .enumerate()
        .filter(|(_, sample)| !sample.kind.is_holdout())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let holdout_indices = prepared
        .iter()
        .enumerate()
        .filter(|(_, sample)| sample.kind.is_holdout())
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let cancel = AtomicBool::new(false);
    let (train, train_observations) =
        evaluate_candidate_detailed(net, &prepared, &train_indices, geometry, mirrors, &cancel);
    let (holdout, holdout_observations) =
        evaluate_candidate_detailed(net, &prepared, &holdout_indices, geometry, mirrors, &cancel);
    let mut observations = Vec::with_capacity(dataset.samples.len());
    observations.extend(
        train_indices
            .iter()
            .copied()
            .zip(train_observations)
            .map(|(sample_index, observation)| research_observation(sample_index, observation)),
    );
    observations.extend(
        holdout_indices
            .iter()
            .copied()
            .zip(holdout_observations)
            .map(|(sample_index, observation)| research_observation(sample_index, observation)),
    );
    observations.sort_by_key(|observation| observation.sample_index);
    ResearchReplay {
        train,
        holdout,
        stability,
        observations,
    }
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn research_observation(sample_index: usize, observation: Observation) -> ResearchObservation {
    ResearchObservation {
        sample_index,
        kind: observation.kind,
        expected_open: observation.expected_open,
        phase_index: observation.phase_index,
        stable: observation.stable,
        native_open: observation.native_open,
        native_gaze_deg: observation.native_gaze_deg,
        presence: observation.presence,
        open: observation.open,
        squeeze: observation.squeeze,
    }
}

#[cfg(feature = "research-synthetic-eye-lab")]
#[allow(clippy::too_many_arguments)]
fn research_prepare_samples(
    dataset: &GeometryDataset,
    stability_baseline: [MlGeometry; 2],
    stability: &StabilityReport,
    despeckle: DespeckleParams,
    captured_flatten: FlattenParams,
    post_normalization_flatten: FlattenParams,
    extra_affine: [[f32; 2]; 2],
    gain_field: [Option<SpatialGainField>; 2],
    coordinate_warp: [Option<ResearchCoordinateWarp>; 2],
) -> Vec<PreparedSample> {
    dataset
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let (lw, lh) = sample.left_size;
            let (rw, rh) = sample.right_size;
            // Reconstruct the exact deterministic capture path first.  The stored
            // adaptive affine was learned from these captured pixels, so applying a
            // candidate before it would not replay the stateful production controller.
            let left = preprocess::despeckle(&sample.left, lw as usize, lh as usize, &despeckle);
            let right = preprocess::despeckle(&sample.right, rw as usize, rh as usize, &despeckle);
            let left = preprocess::flatten(&left, lw as usize, lh as usize, &captured_flatten);
            let right = preprocess::flatten(&right, rw as usize, rh as usize, &captured_flatten);
            let left = brightness::apply(
                &left,
                sample.brightness_affine[0][0],
                sample.brightness_affine[0][1],
            );
            let right = brightness::apply(
                &right,
                sample.brightness_affine[1][0],
                sample.brightness_affine[1][1],
            );

            // Research interventions deliberately live after the captured adaptive
            // brightness result and before geometry.  This makes every probe a valid
            // counterfactual without pretending that BrightState can be reconstructed
            // from a recording that did not save its complete initial history.
            let left =
                preprocess::flatten(&left, lw as usize, lh as usize, &post_normalization_flatten);
            let right = preprocess::flatten(
                &right,
                rw as usize,
                rh as usize,
                &post_normalization_flatten,
            );
            let left = brightness::apply(&left, extra_affine[0][0], extra_affine[0][1]);
            let right = brightness::apply(&right, extra_affine[1][0], extra_affine[1][1]);
            let left = research_apply_spatial_gain(
                &left,
                sample.left_size,
                stability_baseline[0],
                gain_field[0],
            );
            let right = research_apply_spatial_gain(
                &right,
                sample.right_size,
                stability_baseline[1],
                gain_field[1],
            );
            // Coordinate hypotheses use fixed full-frame coordinates and remain
            // independent of the geometry candidate being scored.
            let left = research_apply_coordinate_warp(&left, sample.left_size, coordinate_warp[0]);
            let right =
                research_apply_coordinate_warp(&right, sample.right_size, coordinate_warp[1]);
            PreparedSample {
                kind: sample.kind,
                expected_open: sample.expected_open,
                phase_index: sample.phase_index,
                native_open: sample.native_open,
                native_gaze_deg: sample
                    .native_gaze
                    .map(|gaze| gaze.and_then(crate::pipeline::gaze_angles_deg)),
                stable: stability.flags.get(index).copied().unwrap_or(false),
                left,
                right,
                left_size: sample.left_size,
                right_size: sample.right_size,
            }
        })
        .collect()
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn research_apply_spatial_gain(
    frame: &[u8],
    size: (u32, u32),
    geometry: MlGeometry,
    field: Option<SpatialGainField>,
) -> Vec<u8> {
    let Some(field) = field else {
        return frame.to_vec();
    };
    let (width, height) = (size.0 as usize, size.1 as usize);
    if width == 0 || height == 0 || frame.len() < width.saturating_mul(height) {
        return frame.to_vec();
    }
    let x0 = (geometry.crop_left.clamp(0.0, 0.95) * width as f32).floor() as usize;
    let x1 = ((1.0 - geometry.crop_right.clamp(0.0, 0.95)) * width as f32).ceil() as usize;
    let y0 = (geometry.crop_top.clamp(0.0, 0.95) * height as f32).floor() as usize;
    let y1 = ((1.0 - geometry.crop_bottom.clamp(0.0, 0.95)) * height as f32).ceil() as usize;
    let (x1, y1) = (x1.clamp(x0 + 1, width), y1.clamp(y0 + 1, height));
    let mut result = frame.to_vec();
    for y in 0..height {
        let yn = if y1 - y0 <= 1 {
            0.0
        } else {
            2.0 * (y as f32 - y0 as f32) / (y1 - y0 - 1) as f32 - 1.0
        };
        for x in 0..width {
            let xn = if x1 - x0 <= 1 {
                0.0
            } else {
                2.0 * (x as f32 - x0 as f32) / (x1 - x0 - 1) as f32 - 1.0
            };
            let gain = (1.0
                + field.horizontal.clamp(-0.12, 0.12) * xn
                + field.vertical.clamp(-0.12, 0.12) * yn
                + field.horizontal_curve.clamp(-0.08, 0.08) * (xn * xn - 1.0 / 3.0)
                + field.vertical_curve.clamp(-0.08, 0.08) * (yn * yn - 1.0 / 3.0))
                .clamp(0.70, 1.30);
            let index = y * width + x;
            result[index] = (frame[index] as f32 * gain).clamp(0.0, 255.0) as u8;
        }
    }
    result
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn research_apply_coordinate_warp(
    frame: &[u8],
    size: (u32, u32),
    warp: Option<ResearchCoordinateWarp>,
) -> Vec<u8> {
    let Some(warp) = warp else {
        return frame.to_vec();
    };
    if !warp.vertical_bow.is_finite() || !warp.radial_k1.is_finite() {
        return frame.to_vec();
    }
    let vertical_bow = warp.vertical_bow.clamp(
        -ResearchCoordinateWarp::MAX_VERTICAL_BOW,
        ResearchCoordinateWarp::MAX_VERTICAL_BOW,
    );
    let radial_k1 = warp.radial_k1.clamp(
        -ResearchCoordinateWarp::MAX_RADIAL_K1,
        ResearchCoordinateWarp::MAX_RADIAL_K1,
    );
    if vertical_bow == 0.0 && radial_k1 == 0.0 {
        return frame.to_vec();
    }

    let (width, height) = (size.0 as usize, size.1 as usize);
    let Some(pixel_count) = width.checked_mul(height) else {
        return frame.to_vec();
    };
    if width == 0 || height == 0 || frame.len() != pixel_count {
        return frame.to_vec();
    }

    let mut result = Vec::with_capacity(pixel_count);
    for y in 0..height {
        let yn = pixel_to_normalized(y, height);
        for x in 0..width {
            let xn = pixel_to_normalized(x, width);
            let radius_squared = xn * xn + yn * yn;
            let radial_scale = 1.0 + radial_k1 * radius_squared;
            let source_xn = xn * radial_scale;
            let source_yn = yn * radial_scale + vertical_bow * (xn * xn - 1.0 / 3.0);
            let source_x = normalized_to_pixel(source_xn, width);
            let source_y = normalized_to_pixel(source_yn, height);
            result.push(bilinear_sample_clamped(
                frame, width, height, source_x, source_y,
            ));
        }
    }
    result
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn pixel_to_normalized(position: usize, extent: usize) -> f32 {
    if extent <= 1 {
        0.0
    } else {
        2.0 * position as f32 / (extent - 1) as f32 - 1.0
    }
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn normalized_to_pixel(position: f32, extent: usize) -> f32 {
    if extent <= 1 {
        0.0
    } else {
        (position + 1.0) * 0.5 * (extent - 1) as f32
    }
}

#[cfg(feature = "research-synthetic-eye-lab")]
fn bilinear_sample_clamped(frame: &[u8], width: usize, height: usize, x: f32, y: f32) -> u8 {
    let x = x.clamp(0.0, (width - 1) as f32);
    let y = y.clamp(0.0, (height - 1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let top = frame[y0 * width + x0] as f32 * (1.0 - tx) + frame[y0 * width + x1] as f32 * tx;
    let bottom = frame[y1 * width + x0] as f32 * (1.0 - tx) + frame[y1 * width + x1] as f32 * tx;
    (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8
}

#[cfg(feature = "research-synthetic-eye-lab")]
pub fn research_candidate_admissible(
    candidate: &GeometryMetrics,
    baseline: &GeometryMetrics,
) -> bool {
    admissible(candidate, baseline)
}

fn evaluate_candidate_detailed(
    net: &mut EyeNet,
    samples: &[PreparedSample],
    indices: &[usize],
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    cancel: &AtomicBool,
) -> (GeometryMetrics, Vec<Observation>) {
    let mut observations = Vec::with_capacity(indices.len());
    let mut image = ImageAccum::default();
    let mut previous: Option<(SampleKind, Vec<f32>)> = None;
    for &index in indices {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let sample = &samples[index];
        let input = preprocess::to_input_stereo_geom(
            &sample.left,
            sample.left_size.0,
            sample.left_size.1,
            &sample.right,
            sample.right_size.0,
            sample.right_size.1,
            mirrors[0],
            mirrors[1],
            &geometry[0],
            &geometry[1],
        );
        let contributes_to_fit =
            is_geometry_scoring_kind(sample.kind) && sample.kind.family() != SampleFamily::HalfOpen;
        if contributes_to_fit {
            accumulate_image_stats(&mut image, &input);
            if let Some((previous_kind, previous_input)) = previous.as_ref() {
                if previous_kind.family() == SampleFamily::SlowClose
                    && sample.kind.family() == SampleFamily::SlowClose
                    && previous_kind == &sample.kind
                {
                    image.motion_sum += previous_input
                        .iter()
                        .zip(&input)
                        .map(|(a, b)| (a - b).abs() as f64)
                        .sum::<f64>();
                    image.motion_pixels += input.len();
                }
            }
        }
        let output = net.forward_one(&input);
        observations.push(Observation {
            kind: sample.kind,
            expected_open: sample.expected_open,
            phase_index: sample.phase_index,
            native_open: sample.native_open,
            native_gaze_deg: sample.native_gaze_deg,
            stable: sample.stable,
            presence: output[0],
            open: [output[1], output[2]],
            squeeze: [output[3], output[4]],
        });
        previous = contributes_to_fit.then_some((sample.kind, input));
    }
    let metrics = metrics_from_observations(&observations, &image);
    (metrics, observations)
}

fn accumulate_image_stats(accum: &mut ImageAccum, input: &[f32]) {
    if input.is_empty() {
        return;
    }
    let mean = input.iter().map(|value| *value as f64).sum::<f64>() / input.len() as f64;
    let variance = input
        .iter()
        .map(|value| (*value as f64 - mean).powi(2))
        .sum::<f64>()
        / input.len() as f64;
    accum.spatial_std_sum += variance.sqrt();
    accum.saturation_sum += input
        .iter()
        .filter(|value| **value <= 0.01 || **value >= 0.99)
        .count() as f64;
    accum.pixels += input.len();
    accum.frames += 1;
}

fn metrics_from_observations(
    all_observations: &[Observation],
    image: &ImageAccum,
) -> GeometryMetrics {
    let filtered = all_observations
        .iter()
        .copied()
        .filter(|observation| {
            is_geometry_scoring_kind(observation.kind)
                && observation.kind.family() != SampleFamily::HalfOpen
        })
        .collect::<Vec<_>>();
    let observations = filtered.as_slice();
    if observations.is_empty() {
        return GeometryMetrics::default();
    }
    let finite = observations
        .iter()
        .filter(|observation| {
            observation.presence.is_finite()
                && observation.open.iter().all(|value| value.is_finite())
                && observation.squeeze.iter().all(|value| value.is_finite())
        })
        .count();
    let finite_rate = finite as f32 / observations.len() as f32;
    let presence_rate = observations
        .iter()
        .filter(|observation| observation.presence.is_finite() && observation.presence >= 0.02)
        .count() as f32
        / observations.len() as f32;

    let mut separation = [0.0; 2];
    let mut open_reference = [0.0; 2];
    let mut closed_reference = [0.0; 2];
    let mut monotonicity = [0.0; 2];
    let mut slow_close_std = [0.0; 2];
    let mut neutral_ratio = [1.0; 2];
    let mut gaze_ratio = [1.0; 2];
    let mut blink_depth = [0.0; 2];
    let mut blink_events = [0usize; 2];
    let mut gaze_retention = [1.0; 2];
    let mut gaze_squeeze_fp = [0.0; 2];
    let gaze_observations = observations
        .iter()
        .filter(|observation| {
            observation.kind.family() == SampleFamily::GazeSweep && observation.stable
        })
        .count();
    let gaze_evidence_rate = if gaze_observations == 0 {
        0.0
    } else {
        observations
            .iter()
            .filter(|observation| {
                observation.kind.family() == SampleFamily::GazeSweep
                    && observation.stable
                    && observation.native_gaze_deg.iter().all(Option::is_some)
            })
            .count() as f32
            / gaze_observations as f32
    };
    if gaze_observations < MIN_GAZE_STABLE_FRAMES {
        return GeometryMetrics {
            evidence_valid: false,
            finite_rate,
            presence_rate,
            gaze_evidence_rate,
            ..GeometryMetrics::default()
        };
    }
    for eye in 0..2 {
        let open = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Neutral && observation.stable
        });
        let closed = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Closed && observation.stable
        });
        if open.len() < 5 || closed.len() < 5 {
            return GeometryMetrics {
                evidence_valid: false,
                finite_rate,
                presence_rate,
                ..GeometryMetrics::default()
            };
        }
        let (open_mean, open_var) = mean_variance(&open);
        let (closed_mean, closed_var) = mean_variance(&closed);
        let open_ref = percentile(&open, 0.50).unwrap_or(open_mean);
        let closed_ref = percentile(&closed, 0.50).unwrap_or(closed_mean);
        open_reference[eye] = open_ref;
        closed_reference[eye] = closed_ref;
        let span = (open_ref - closed_ref).max(0.001);
        separation[eye] = ((open_mean - closed_mean)
            / ((0.5 * (open_var + closed_var) + 1e-4).sqrt()))
        .clamp(-4.0, 8.0);

        let mut target = Vec::new();
        let mut response = Vec::new();
        for observation in observations {
            if observation.kind.family() == SampleFamily::SlowClose {
                if let Some(expected) = observation.expected_open {
                    if observation.open[eye].is_finite() {
                        target.push(expected);
                        response.push(observation.open[eye]);
                    }
                }
            }
        }
        if target.len() < 5 {
            return GeometryMetrics {
                evidence_valid: false,
                finite_rate,
                presence_rate,
                ..GeometryMetrics::default()
            };
        }
        monotonicity[eye] = pearson(&target, &response).clamp(-1.0, 1.0);
        slow_close_std[eye] = mean_variance(&response).1.sqrt();

        let neutral = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Neutral && observation.stable
        });
        let gaze = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::GazeSweep && observation.stable
        });
        neutral_ratio[eye] = mean_variance(&neutral).1.sqrt() / span;
        gaze_ratio[eye] = mean_variance(&gaze).1.sqrt() / span;
        gaze_retention[eye] = gaze_retention_for_eye(
            observations,
            eye,
            closed_ref,
            span,
            gaze_evidence_rate >= 0.60,
        );
        let neutral_squeeze = squeeze_values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Neutral && observation.stable
        });
        let gaze_squeeze = squeeze_values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::GazeSweep && observation.stable
        });
        let neutral_squeeze_ref = percentile(&neutral_squeeze, 0.50).unwrap_or(0.0);
        gaze_squeeze_fp[eye] = (percentile(&gaze_squeeze, 0.90).unwrap_or(neutral_squeeze_ref)
            - neutral_squeeze_ref)
            .max(0.0);

        let blink = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::NaturalBlinks
        });
        let blink_low = percentile(&blink, 0.10).unwrap_or(open_ref);
        blink_depth[eye] = ((open_ref - blink_low) / span).clamp(0.0, 1.5) / 1.5;
        let threshold = (open_ref + closed_ref) * 0.5;
        blink_events[eye] = count_blink_events(&blink, threshold, span * 0.05);
    }

    let blink_left = values(observations, 0, |observation| {
        observation.kind.family() == SampleFamily::NaturalBlinks
    });
    let blink_right = values(observations, 1, |observation| {
        observation.kind.family() == SampleFamily::NaturalBlinks
    });
    let blink_stereo = pearson(&blink_left, &blink_right).clamp(0.0, 1.0);
    let expected_blinks = if observations
        .iter()
        .any(|observation| observation.kind == SampleKind::HoldoutNaturalBlinks)
    {
        3.0
    } else {
        5.0
    };
    let count_score = blink_events
        .iter()
        .map(|count| {
            (1.0 - (*count as f32 - expected_blinks).abs() / expected_blinks).clamp(0.0, 1.0)
        })
        .sum::<f32>()
        * 0.5;
    let blink_response =
        (0.45 * (blink_depth[0] + blink_depth[1]) * 0.5 + 0.35 * blink_stereo + 0.20 * count_score)
            .clamp(0.0, 1.0);

    let neutral_noise = (neutral_ratio[0] + neutral_ratio[1]) * 0.5;
    let gaze_noise = (gaze_ratio[0] + gaze_ratio[1]) * 0.5;
    let stability = (1.0
        - 0.5 * (neutral_noise / 0.15).clamp(0.0, 1.0)
        - 0.5 * (gaze_noise / 0.25).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);

    let image_std = if image.frames == 0 {
        0.0
    } else {
        (image.spatial_std_sum / image.frames as f64) as f32
    };
    let saturation_rate = if image.pixels == 0 {
        1.0
    } else {
        (image.saturation_sum / image.pixels as f64) as f32
    };
    let motion_energy = if image.motion_pixels == 0 {
        0.0
    } else {
        (image.motion_sum / image.motion_pixels as f64) as f32
    };
    let image_information = (0.50 * (image_std / 0.10).clamp(0.0, 1.0)
        + 0.20 * (1.0 - saturation_rate / 0.35).clamp(0.0, 1.0)
        + 0.30 * (motion_energy / 0.025).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    let separation_score = separation
        .iter()
        .map(|value| (value / 3.0).clamp(0.0, 1.0))
        .sum::<f32>()
        * 0.5;
    let monotonicity_score = monotonicity
        .iter()
        .map(|value| value.clamp(0.0, 1.0))
        .sum::<f32>()
        * 0.5;
    let gaze_asymmetry = (gaze_retention[0] - gaze_retention[1]).abs();
    let gaze_quality = (0.60 * gaze_retention[0].min(gaze_retention[1]).clamp(0.0, 1.0)
        + 0.40 * (1.0 - gaze_squeeze_fp[0].max(gaze_squeeze_fp[1]) / 0.30).clamp(0.0, 1.0))
    .clamp(0.0, 1.0);
    let score = (0.28 * separation_score
        + 0.22 * monotonicity_score
        + 0.13 * blink_response
        + 0.13 * stability
        + 0.07 * presence_rate
        + 0.05 * image_information)
        + 0.12 * gaze_quality;
    let score = score * finite_rate;
    GeometryMetrics {
        evidence_valid: true,
        score,
        separation,
        open_ref: open_reference,
        closed_ref: closed_reference,
        monotonicity,
        slow_close_std,
        blink_response,
        stability,
        presence_rate,
        finite_rate,
        image_information,
        image_std,
        saturation_rate,
        motion_energy,
        neutral_noise,
        gaze_noise,
        neutral_noise_per_eye: neutral_ratio,
        gaze_noise_per_eye: gaze_ratio,
        gaze_retention,
        gaze_squeeze_fp,
        gaze_asymmetry,
        gaze_evidence_rate,
        blink_events,
    }
}

fn admissible(candidate: &GeometryMetrics, baseline: &GeometryMetrics) -> bool {
    candidate.evidence_valid
        && baseline.evidence_valid
        && candidate.finite_rate >= 0.99
        && candidate.presence_rate + 0.02 >= baseline.presence_rate
        && candidate.image_std >= baseline.image_std * 0.65
        && candidate.motion_energy + 0.001 >= baseline.motion_energy * 0.55
        && candidate.saturation_rate <= baseline.saturation_rate + 0.15
        && candidate.gaze_asymmetry <= baseline.gaze_asymmetry + 0.05
        && (0..2).all(|eye| {
            candidate.separation[eye] + 0.15 >= baseline.separation[eye] * 0.85
                && candidate.monotonicity[eye] + 0.10 >= baseline.monotonicity[eye]
                && candidate.gaze_retention[eye] + 0.05 >= baseline.gaze_retention[eye]
                && candidate.gaze_squeeze_fp[eye] <= baseline.gaze_squeeze_fp[eye] + 0.05
        })
}

fn capture_quality_issue(baseline: &GeometryMetrics) -> Option<&'static str> {
    if !baseline.evidence_valid {
        Some("one or more required open/closed/slow-close/stable-gaze evidence classes are missing")
    } else if baseline.finite_rate < 0.99 {
        Some("the fixed network produced non-finite outputs")
    } else if baseline.presence_rate < 0.50 {
        Some("the eyelid network was absent on more than half of the frames")
    } else if average(baseline.separation) < 0.15 {
        Some("the recorded open and gently-closed phases are not distinguishable")
    } else if average(baseline.monotonicity) < 0.05 {
        Some("the slow-close recording did not follow the on-screen guide")
    } else if baseline.image_std < 0.02 || baseline.image_information < 0.05 {
        Some("the current crop contains too little eye-image information to search safely")
    } else {
        None
    }
}

fn acceptance(
    baseline_train: &GeometryMetrics,
    candidate_train: &GeometryMetrics,
    baseline_holdout: &GeometryMetrics,
    candidate_holdout: &GeometryMetrics,
    candidate_is_baseline: bool,
    flat_objective: bool,
) -> (bool, String) {
    if candidate_is_baseline {
        return (
            false,
            "The current geometry already scored best; nothing was changed.".into(),
        );
    }
    if flat_objective {
        return (
            false,
            "Several distant geometries scored the same; the fit is uncertain, so the current geometry was kept."
                .into(),
        );
    }
    if !admissible(candidate_holdout, baseline_holdout) {
        return (
            false,
            "The candidate violated a holdout safety guard; the current geometry was kept.".into(),
        );
    }
    if candidate_train.score <= baseline_train.score + 0.01 {
        return (
            false,
            "The search did not materially improve its training frames; the current geometry was kept."
                .into(),
        );
    }
    let improvement = candidate_holdout.score - baseline_holdout.score;
    let relative = improvement / baseline_holdout.score.max(0.05);
    if improvement < 0.03 || relative < 0.08 {
        return (
            false,
            format!(
                "Holdout improvement was only {:+.3} ({:+.1}%); at least +0.030 and +8% are required.",
                improvement,
                relative * 100.0
            ),
        );
    }
    let per_eye_regression = (0..2).any(|eye| {
        candidate_holdout.separation[eye] + 0.05 < baseline_holdout.separation[eye] * 0.98
            || candidate_holdout.monotonicity[eye] + 0.03 < baseline_holdout.monotonicity[eye]
            || candidate_holdout.gaze_noise_per_eye[eye]
                > baseline_holdout.gaze_noise_per_eye[eye] * 1.05 + 0.02
            || candidate_holdout.neutral_noise_per_eye[eye]
                > baseline_holdout.neutral_noise_per_eye[eye] * 1.10 + 0.01
            || candidate_holdout.gaze_retention[eye] + 0.03 < baseline_holdout.gaze_retention[eye]
            || candidate_holdout.gaze_squeeze_fp[eye] > baseline_holdout.gaze_squeeze_fp[eye] + 0.03
    });
    if per_eye_regression {
        return (
            false,
            "The total score improved, but an essential eyelid or gaze-stability metric regressed."
                .into(),
        );
    }
    (
        true,
        format!(
            "Accepted on untouched holdout frames: {:+.3} ({:+.1}%).",
            improvement,
            relative * 100.0
        ),
    )
}

#[derive(Clone)]
struct ScoredPhotometric {
    correction: PhotometricCorrection,
    metrics: GeometryMetrics,
}

fn lock_photometric(shared: &Arc<Mutex<PhotometricShared>>) -> MutexGuard<'_, PhotometricShared> {
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn photometric_log(shared: &Arc<Mutex<PhotometricShared>>, line: impl Into<String>) {
    lock_photometric(shared).push(line);
}

fn photometric_progress(
    shared: &Arc<Mutex<PhotometricShared>>,
    stage: &str,
    completed: usize,
    total: usize,
) {
    lock_photometric(shared).progress(stage, completed, total);
}

fn photometric_fail(shared: &Arc<Mutex<PhotometricShared>>, message: String) {
    let mut state = lock_photometric(shared);
    let log = state.log.clone();
    state.status = PhotometricStatus::Failed { message, log };
}

fn photometric_cancelled(shared: &Arc<Mutex<PhotometricShared>>, cancel: &AtomicBool) -> bool {
    if !cancel.load(Ordering::Relaxed) {
        return false;
    }
    let mut state = lock_photometric(shared);
    let log = state.log.clone();
    state.status = PhotometricStatus::Cancelled { log };
    true
}

fn run_photometric(
    shared: Arc<Mutex<PhotometricShared>>,
    cancel: Arc<AtomicBool>,
    inputs: PhotometricFitInputs,
) {
    photometric_log(&shared, format!("[load] {}", inputs.model_path.display()));
    let map = match tvm_params::parse_map_bytes(&inputs.model_bytes) {
        Ok(map) => map,
        Err(error) => {
            return photometric_fail(
                &shared,
                format!("EyePrediction model parse failed: {error}"),
            )
        }
    };
    let mut net = match EyeNet::new(map) {
        Ok(net) => net,
        Err(error) => {
            return photometric_fail(
                &shared,
                format!("EyePrediction model is incompatible: {error}"),
            )
        }
    };

    photometric_log(
        &shared,
        "[prepare] replaying captured filters and adaptive-brightness affine",
    );
    let stability = capture_stability_flags(inputs.dataset.samples(), inputs.geometry);
    for note in &stability.notes {
        photometric_log(&shared, format!("[capture evidence] {note}"));
    }
    if stability.invalid_static_phases >= 2
        || stability
            .valid_closed_phases
            .iter()
            .any(|count| *count == 0)
    {
        return photometric_fail(
            &shared,
            "capture contains invalid static/closed evidence; follow the final pose in each prompt and record again"
                .into(),
        );
    }

    let prepare_total = inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_geometry_scoring_kind(sample.kind))
        .count();
    photometric_progress(&shared, "preparing captured frames", 0, prepare_total);
    let mut prepared = Vec::with_capacity(prepare_total);
    for (index, sample) in inputs.dataset.samples().iter().enumerate() {
        if !is_geometry_scoring_kind(sample.kind) {
            continue;
        }
        if photometric_cancelled(&shared, &cancel) {
            return;
        }
        let (lw, lh) = sample.left_size;
        let (rw, rh) = sample.right_size;
        let left = preprocess::despeckle(&sample.left, lw as usize, lh as usize, &inputs.despeckle);
        let right =
            preprocess::despeckle(&sample.right, rw as usize, rh as usize, &inputs.despeckle);
        let left = preprocess::flatten(&left, lw as usize, lh as usize, &inputs.flatten);
        let right = preprocess::flatten(&right, rw as usize, rh as usize, &inputs.flatten);
        let left = brightness::apply(
            &left,
            sample.brightness_affine[0][0],
            sample.brightness_affine[0][1],
        );
        let right = brightness::apply(
            &right,
            sample.brightness_affine[1][0],
            sample.brightness_affine[1][1],
        );
        prepared.push(PreparedSample {
            kind: sample.kind,
            expected_open: sample.expected_open,
            phase_index: sample.phase_index,
            native_open: sample.native_open,
            native_gaze_deg: sample
                .native_gaze
                .map(|gaze| gaze.and_then(crate::pipeline::gaze_angles_deg)),
            stable: stability.flags[index],
            left,
            right,
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
        let prepared_count = prepared.len();
        if prepared_count % 20 == 0 || prepared_count == prepare_total {
            photometric_progress(
                &shared,
                "preparing captured frames",
                prepared_count,
                prepare_total,
            );
        }
    }

    let train_coarse = stratified_indices(&prepared, false, 140);
    let train_large = stratified_indices(&prepared, false, 320);
    let train_final = stratified_indices(&prepared, false, 420);
    let holdout = stratified_indices(&prepared, true, 420);
    if train_coarse.is_empty() || train_final.is_empty() || holdout.is_empty() {
        return photometric_fail(
            &shared,
            "capture contains no usable train or untouched holdout frames".into(),
        );
    }

    let mut work_done = 0usize;
    let mut work_total = prepare_total;
    let baseline = inputs.baseline;

    let coarse = coarse_photometric_candidates(baseline);
    photometric_log(
        &shared,
        format!(
            "[search 1/3] {} shared/per-eye brightness and contrast probes",
            coarse.len()
        ),
    );
    let mut coarse_scored = evaluate_photometric_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train_coarse,
        inputs.geometry,
        inputs.mirrors,
        &coarse,
        "coarse affine search",
        &mut work_done,
        &mut work_total,
    );
    if photometric_cancelled(&shared, &cancel) {
        return;
    }
    let Some(coarse_baseline) = coarse_scored
        .iter()
        .find(|entry| entry.correction == baseline)
        .map(|entry| entry.metrics.clone())
    else {
        return photometric_fail(
            &shared,
            "the current photometric path produced no finite baseline".into(),
        );
    };
    if let Some(issue) = capture_quality_issue(&coarse_baseline) {
        return photometric_fail(
            &shared,
            format!("capture quality check failed: {issue}; record the sequence again"),
        );
    }
    coarse_scored.retain(|entry| {
        photometric_admissible(&entry.metrics, &coarse_baseline)
            && photometric_always_open_guards(&entry.metrics, &coarse_baseline)
    });
    sort_photometric(&mut coarse_scored);
    if coarse_scored.is_empty() {
        return photometric_fail(
            &shared,
            "every coarse photometric candidate violated a safety guard".into(),
        );
    }

    let mut finalists = vec![baseline];
    for entry in coarse_scored.iter().take(7) {
        push_unique_photometric(&mut finalists, entry.correction);
    }
    photometric_log(
        &shared,
        "[search 2/3] successive halving plus weak local flatten refinement",
    );
    let mut halved = evaluate_photometric_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train_large,
        inputs.geometry,
        inputs.mirrors,
        &finalists,
        "successive halving",
        &mut work_done,
        &mut work_total,
    );
    let halved_baseline = halved
        .iter()
        .find(|entry| entry.correction == baseline)
        .map(|entry| entry.metrics.clone())
        .unwrap_or_else(|| coarse_baseline.clone());
    halved.retain(|entry| {
        photometric_admissible(&entry.metrics, &halved_baseline)
            && photometric_always_open_guards(&entry.metrics, &halved_baseline)
    });
    sort_photometric(&mut halved);
    let Some(affine_best) = halved.first().map(|entry| entry.correction) else {
        return photometric_fail(
            &shared,
            "no safe affine candidate survived successive halving".into(),
        );
    };

    let refined = refinement_photometric_candidates(baseline, affine_best);
    let mut refined_scored = evaluate_photometric_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train_large,
        inputs.geometry,
        inputs.mirrors,
        &refined,
        "affine and flatten refinement",
        &mut work_done,
        &mut work_total,
    );
    let refined_baseline = refined_scored
        .iter()
        .find(|entry| entry.correction == baseline)
        .map(|entry| entry.metrics.clone())
        .unwrap_or_else(|| halved_baseline.clone());
    refined_scored.retain(|entry| {
        photometric_admissible(&entry.metrics, &refined_baseline)
            && photometric_always_open_guards(&entry.metrics, &refined_baseline)
    });
    sort_photometric(&mut refined_scored);
    let Some(refined_best) = refined_scored.first().map(|entry| entry.correction) else {
        return photometric_fail(
            &shared,
            "photometric refinement produced no safe result".into(),
        );
    };

    photometric_log(
        &shared,
        "[search 3/3] bounded low-frequency illumination fields; geometry remains frozen",
    );
    let field_candidates = field_photometric_candidates(baseline, refined_best);
    let mut field_scored = evaluate_photometric_set(
        &shared,
        &cancel,
        &mut net,
        &prepared,
        &train_large,
        inputs.geometry,
        inputs.mirrors,
        &field_candidates,
        "local illumination search",
        &mut work_done,
        &mut work_total,
    );
    let field_baseline = field_scored
        .iter()
        .find(|entry| entry.correction == baseline)
        .map(|entry| entry.metrics.clone())
        .unwrap_or_else(|| refined_baseline.clone());
    field_scored.retain(|entry| {
        photometric_admissible(&entry.metrics, &field_baseline)
            && photometric_always_open_guards(&entry.metrics, &field_baseline)
    });
    sort_photometric(&mut field_scored);
    let Some(best) = field_scored.first().cloned() else {
        return photometric_fail(
            &shared,
            "low-frequency illumination search produced no safe result".into(),
        );
    };
    let flat_objective = field_scored.get(1).is_some_and(|runner_up| {
        (best.metrics.score - runner_up.metrics.score).abs() < 0.012
            && photometric_distance(best.correction, runner_up.correction) > 0.75
    });

    photometric_log(
        &shared,
        "[verify train] replaying current and selected correction on the full train set",
    );
    work_total += 2 * train_final.len() + 2 * holdout.len();
    let baseline_train = evaluate_photometric_candidate(
        &mut net,
        &prepared,
        &train_final,
        inputs.geometry,
        inputs.mirrors,
        baseline,
        &cancel,
    );
    work_done += train_final.len();
    photometric_progress(&shared, "full train verification", work_done, work_total);
    let candidate_train = evaluate_photometric_candidate(
        &mut net,
        &prepared,
        &train_final,
        inputs.geometry,
        inputs.mirrors,
        best.correction,
        &cancel,
    );
    work_done += train_final.len();
    photometric_progress(&shared, "full train verification", work_done, work_total);
    if photometric_cancelled(&shared, &cancel) {
        return;
    }

    photometric_log(
        &shared,
        "[holdout] comparing only the train-selected winner against the frozen current path",
    );
    let baseline_holdout = evaluate_photometric_candidate(
        &mut net,
        &prepared,
        &holdout,
        inputs.geometry,
        inputs.mirrors,
        baseline,
        &cancel,
    );
    work_done += holdout.len();
    photometric_progress(
        &shared,
        "untouched holdout validation",
        work_done,
        work_total,
    );
    let candidate_holdout = evaluate_photometric_candidate(
        &mut net,
        &prepared,
        &holdout,
        inputs.geometry,
        inputs.mirrors,
        best.correction,
        &cancel,
    );
    work_done += holdout.len();
    photometric_progress(
        &shared,
        "untouched holdout validation",
        work_done,
        work_total,
    );
    if photometric_cancelled(&shared, &cancel) {
        return;
    }

    let safety_ok = photometric_admissible(&candidate_train, &baseline_train)
        && photometric_always_open_guards(&candidate_train, &baseline_train)
        && photometric_admissible(&candidate_holdout, &baseline_holdout)
        && photometric_always_open_guards(&candidate_holdout, &baseline_holdout);
    let (accepted, reason) = if !safety_ok {
        (
            false,
            "The candidate violated a train or holdout photometric safety guard; the current correction was kept."
                .into(),
        )
    } else {
        acceptance(
            &baseline_train,
            &candidate_train,
            &baseline_holdout,
            &candidate_holdout,
            best.correction == baseline,
            flat_objective,
        )
    };
    let result = PhotometricFitResult {
        baseline,
        candidate: best.correction,
        holdout_improvement: candidate_holdout.score - baseline_holdout.score,
        baseline_train,
        candidate_train,
        baseline_holdout,
        candidate_holdout,
        invalid_static_phases: stability.invalid_static_phases,
        degraded_static_phases: stability.degraded_static_phases,
        valid_closed_phases: stability.valid_closed_phases,
        accepted,
        reason,
    };
    let mut state = lock_photometric(&shared);
    state.push(format!(
        "[done] holdout {:.3} -> {:.3}; {}",
        result.baseline_holdout.score,
        result.candidate_holdout.score,
        if result.accepted {
            "candidate accepted"
        } else {
            "fallback retained"
        }
    ));
    let log = state.log.clone();
    state.status = PhotometricStatus::Done { result, log };
}

#[allow(clippy::too_many_arguments)]
fn evaluate_photometric_set(
    shared: &Arc<Mutex<PhotometricShared>>,
    cancel: &AtomicBool,
    net: &mut EyeNet,
    prepared: &[PreparedSample],
    indices: &[usize],
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    candidates: &[PhotometricCorrection],
    stage: &str,
    work_done: &mut usize,
    work_total: &mut usize,
) -> Vec<ScoredPhotometric> {
    *work_total += candidates.len() * indices.len();
    let mut scored = Vec::with_capacity(candidates.len());
    for (candidate_index, correction) in candidates.iter().copied().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let metrics = evaluate_photometric_candidate(
            net, prepared, indices, geometry, mirrors, correction, cancel,
        );
        *work_done += indices.len();
        photometric_progress(shared, stage, *work_done, *work_total);
        if metrics.score.is_finite() {
            scored.push(ScoredPhotometric {
                correction,
                metrics,
            });
        }
        if candidate_index % 8 == 0 {
            photometric_log(
                shared,
                format!(
                    "[{stage}] {}/{} candidates",
                    candidate_index + 1,
                    candidates.len()
                ),
            );
        }
    }
    scored
}

fn evaluate_photometric_candidate(
    net: &mut EyeNet,
    prepared: &[PreparedSample],
    indices: &[usize],
    geometry: [MlGeometry; 2],
    mirrors: [bool; 2],
    correction: PhotometricCorrection,
    cancel: &AtomicBool,
) -> GeometryMetrics {
    let mut transformed = Vec::with_capacity(indices.len());
    for &index in indices {
        if cancel.load(Ordering::Relaxed) {
            return GeometryMetrics::default();
        }
        let sample = &prepared[index];
        transformed.push(PreparedSample {
            kind: sample.kind,
            expected_open: sample.expected_open,
            phase_index: sample.phase_index,
            native_open: sample.native_open,
            native_gaze_deg: sample.native_gaze_deg,
            stable: sample.stable,
            left: preprocess::fitted_photometric(
                &sample.left,
                sample.left_size.0 as usize,
                sample.left_size.1 as usize,
                &geometry[0],
                &correction,
                0,
            ),
            right: preprocess::fitted_photometric(
                &sample.right,
                sample.right_size.0 as usize,
                sample.right_size.1 as usize,
                &geometry[1],
                &correction,
                1,
            ),
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
    }
    let local_indices = (0..transformed.len()).collect::<Vec<_>>();
    evaluate_candidate(net, &transformed, &local_indices, geometry, mirrors, cancel)
}

fn effective_photometric(mut correction: PhotometricCorrection) -> PhotometricCorrection {
    if !correction.enabled {
        correction = PhotometricCorrection::default();
    }
    correction.enabled = true;
    correction
}

fn compose_affine_delta(
    baseline: PhotometricCorrection,
    targets: [bool; 2],
    gain: f32,
    bias: f32,
) -> PhotometricCorrection {
    let mut candidate = effective_photometric(baseline);
    for eye in 0..2 {
        if targets[eye] {
            let [base_gain, base_bias] = candidate.affine[eye];
            candidate.affine[eye] = [
                (gain * base_gain).clamp(0.70, 1.30),
                (gain * base_bias + bias).clamp(-30.0, 30.0),
            ];
        }
    }
    candidate
}

fn coarse_photometric_candidates(baseline: PhotometricCorrection) -> Vec<PhotometricCorrection> {
    let mut candidates = vec![baseline];
    for gain in [0.85, 0.93, 1.07, 1.15] {
        push_unique_photometric(
            &mut candidates,
            compose_affine_delta(baseline, [true, true], gain, 0.0),
        );
    }
    for bias in [-16.0, -8.0, 8.0, 16.0] {
        push_unique_photometric(
            &mut candidates,
            compose_affine_delta(baseline, [true, true], 1.0, bias),
        );
    }
    for (gain, bias) in [(0.90, -10.0), (0.90, 10.0), (1.10, -10.0), (1.10, 10.0)] {
        push_unique_photometric(
            &mut candidates,
            compose_affine_delta(baseline, [true, true], gain, bias),
        );
    }
    for eye in 0..2 {
        let targets = [eye == 0, eye == 1];
        for gain in [0.88, 0.95, 1.05, 1.12] {
            push_unique_photometric(
                &mut candidates,
                compose_affine_delta(baseline, targets, gain, 0.0),
            );
        }
        for bias in [-12.0, -6.0, 6.0, 12.0] {
            push_unique_photometric(
                &mut candidates,
                compose_affine_delta(baseline, targets, 1.0, bias),
            );
        }
    }
    candidates
}

fn refinement_photometric_candidates(
    baseline: PhotometricCorrection,
    centre: PhotometricCorrection,
) -> Vec<PhotometricCorrection> {
    let mut candidates = vec![baseline, centre];
    for targets in [[true, true], [true, false], [false, true]] {
        for gain in [0.96, 1.04] {
            push_unique_photometric(
                &mut candidates,
                compose_affine_delta(centre, targets, gain, 0.0),
            );
        }
        for bias in [-4.0, 4.0] {
            push_unique_photometric(
                &mut candidates,
                compose_affine_delta(centre, targets, 1.0, bias),
            );
        }
    }
    for radius in [0.22, 0.33, 0.45] {
        for strength in [0.25, 0.45, 0.65] {
            let mut candidate = effective_photometric(centre);
            candidate.flatten = FlattenParams {
                enabled: true,
                strength,
                radius,
            };
            push_unique_photometric(&mut candidates, candidate);
        }
    }
    candidates
}

fn field_photometric_candidates(
    baseline: PhotometricCorrection,
    centre: PhotometricCorrection,
) -> Vec<PhotometricCorrection> {
    let mut candidates = vec![baseline, centre];
    for targets in [[true, true], [true, false], [false, true]] {
        for (axis, values) in [
            (0usize, [-0.12, -0.06, 0.06, 0.12]),
            (1usize, [-0.12, -0.06, 0.06, 0.12]),
            (2usize, [-0.08, -0.04, 0.04, 0.08]),
            (3usize, [-0.08, -0.04, 0.04, 0.08]),
        ] {
            for value in values {
                let mut candidate = effective_photometric(centre);
                for eye in 0..2 {
                    if !targets[eye] {
                        continue;
                    }
                    match axis {
                        0 => {
                            candidate.field[eye].horizontal =
                                (candidate.field[eye].horizontal + value).clamp(-0.12, 0.12)
                        }
                        1 => {
                            candidate.field[eye].vertical =
                                (candidate.field[eye].vertical + value).clamp(-0.12, 0.12)
                        }
                        2 => {
                            candidate.field[eye].horizontal_curve =
                                (candidate.field[eye].horizontal_curve + value).clamp(-0.08, 0.08)
                        }
                        _ => {
                            candidate.field[eye].vertical_curve =
                                (candidate.field[eye].vertical_curve + value).clamp(-0.08, 0.08)
                        }
                    }
                }
                push_unique_photometric(&mut candidates, candidate);
            }
        }
    }
    candidates
}

fn push_unique_photometric(
    candidates: &mut Vec<PhotometricCorrection>,
    candidate: PhotometricCorrection,
) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn sort_photometric(scored: &mut [ScoredPhotometric]) {
    scored.sort_by(|left, right| right.metrics.score.total_cmp(&left.metrics.score));
}

fn photometric_admissible(candidate: &GeometryMetrics, baseline: &GeometryMetrics) -> bool {
    admissible(candidate, baseline)
}

fn photometric_always_open_guards(candidate: &GeometryMetrics, baseline: &GeometryMetrics) -> bool {
    candidate.saturation_rate <= baseline.saturation_rate + 0.10
        && (0..2).all(|eye| {
            let span = (baseline.open_ref[eye] - baseline.closed_ref[eye]).max(0.001);
            candidate.closed_ref[eye] <= baseline.closed_ref[eye] + 0.05 * span
                && candidate.slow_close_std[eye] >= baseline.slow_close_std[eye] * 0.60
                && candidate.monotonicity[eye] + 0.05 >= baseline.monotonicity[eye]
        })
}

fn photometric_distance(left: PhotometricCorrection, right: PhotometricCorrection) -> f32 {
    let mut sum = 0.0;
    for eye in 0..2 {
        sum += ((left.affine[eye][0] - right.affine[eye][0]) / 0.15).powi(2);
        sum += ((left.affine[eye][1] - right.affine[eye][1]) / 15.0).powi(2);
        sum += ((left.field[eye].horizontal - right.field[eye].horizontal) / 0.12).powi(2);
        sum += ((left.field[eye].vertical - right.field[eye].vertical) / 0.12).powi(2);
        sum +=
            ((left.field[eye].horizontal_curve - right.field[eye].horizontal_curve) / 0.08).powi(2);
        sum += ((left.field[eye].vertical_curve - right.field[eye].vertical_curve) / 0.08).powi(2);
    }
    sum += ((left.flatten.strength - right.flatten.strength) / 0.70).powi(2);
    sum.sqrt()
}

fn geometry_from_params(baseline: [MlGeometry; 2], params: SearchParams) -> [MlGeometry; 2] {
    // Preserve the fallback byte-for-byte. Besides making rollback exact, this avoids
    // tiny float reconstruction differences becoming a false candidate in reports.
    if params == SearchParams::default() {
        return baseline;
    }
    let [inward, vertical, size, stretch_y, rotation] = params.clamped().0;
    std::array::from_fn(|eye| {
        let base = baseline[eye];
        let width = (1.0 - base.crop_left - base.crop_right).clamp(0.20, 1.0);
        let height = (1.0 - base.crop_top - base.crop_bottom).clamp(0.20, 1.0);
        let base_cx = base.crop_left + width * 0.5;
        let base_cy = base.crop_top + height * 0.5;
        let inward_sign = if eye == 0 { 1.0 } else { -1.0 };
        let scale = 1.0 + size * 0.15;
        // A wider window cannot coexist with the fixed inner LED exclusion while
        // remaining inside the physical frame. Limit horizontal size accordingly.
        let width = (width * scale).clamp(0.20, 1.0 - XR5_MIN_INNER_CROP);
        let height = (height * scale).clamp(0.20, 0.90);
        let requested_cx = base_cx + inward_sign * inward * 0.08;
        let requested_cy = base_cy + vertical * 0.08;
        // Intersect the physical frame bounds with the promised local neighbourhood.
        // Independent edge clamps would silently change window size near an edge.
        let hardware_min_cx = if eye == 0 {
            width * 0.5
        } else {
            width * 0.5 + XR5_MIN_INNER_CROP
        };
        let hardware_max_cx = if eye == 0 {
            1.0 - width * 0.5 - XR5_MIN_INNER_CROP
        } else {
            1.0 - width * 0.5
        };
        let min_cx = (width * 0.5).max(base_cx - 0.08).max(hardware_min_cx);
        let max_cx = (1.0 - width * 0.5).min(base_cx + 0.08).min(hardware_max_cx);
        let min_cy = (height * 0.5).max(base_cy - 0.08);
        let max_cy = (1.0 - height * 0.5).min(base_cy + 0.08);
        let cx = if min_cx <= max_cx {
            requested_cx.clamp(min_cx, max_cx)
        } else {
            base_cx.clamp(width * 0.5, 1.0 - width * 0.5)
        };
        let cy = if min_cy <= max_cy {
            requested_cy.clamp(min_cy, max_cy)
        } else {
            base_cy.clamp(height * 0.5, 1.0 - height * 0.5)
        };
        let angle_sign = if base.rotate_deg.abs() < 1.0 {
            if eye == 0 {
                -1.0
            } else {
                1.0
            }
        } else {
            base.rotate_deg.signum()
        };
        MlGeometry {
            crop_left: (cx - width * 0.5).max(0.0),
            crop_right: (1.0 - cx - width * 0.5).max(0.0),
            crop_top: (cy - height * 0.5).max(0.0),
            crop_bottom: (1.0 - cy - height * 0.5).max(0.0),
            scale_x: base.scale_x,
            scale_y: (base.scale_y + stretch_y * 0.10).clamp(0.70, 1.60),
            rotate_deg: (base.rotate_deg + angle_sign * rotation * 8.0).clamp(-45.0, 45.0),
            mirror_h: base.mirror_h,
        }
    })
}

fn halton_params(index: usize) -> SearchParams {
    let bases = [2usize, 3, 5, 7, 11];
    SearchParams(std::array::from_fn(|axis| {
        halton(index + 1, bases[axis]) * 2.0 - 1.0
    }))
}

fn halton(mut index: usize, base: usize) -> f32 {
    let mut factor = 1.0f32;
    let mut result = 0.0f32;
    while index > 0 {
        factor /= base as f32;
        result += factor * (index % base) as f32;
        index /= base;
    }
    result
}

fn stratified_indices(samples: &[PreparedSample], holdout: bool, limit: usize) -> Vec<usize> {
    let families = if holdout {
        vec![
            SampleFamily::Neutral,
            SampleFamily::GazeSweep,
            SampleFamily::SlowClose,
            SampleFamily::NaturalBlinks,
            SampleFamily::Closed,
        ]
    } else {
        vec![
            SampleFamily::Neutral,
            SampleFamily::GazeSweep,
            SampleFamily::SlowClose,
            SampleFamily::NaturalBlinks,
            SampleFamily::Closed,
        ]
    };
    let eligible: Vec<usize> = samples
        .iter()
        .enumerate()
        .filter(|(_, sample)| {
            is_geometry_scoring_kind(sample.kind) && sample.kind.is_holdout() == holdout
        })
        .map(|(index, _)| index)
        .collect();
    if eligible.len() <= limit {
        return eligible;
    }
    let quota = (limit / families.len().max(1)).max(1);
    let mut chosen = BTreeSet::new();
    for family in families {
        let group: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|index| samples[*index].kind.family() == family)
            .collect();
        let take = quota.min(group.len());
        for position in 0..take {
            chosen.insert(group[position * group.len() / take]);
        }
    }
    let remaining = limit.saturating_sub(chosen.len());
    if remaining > 0 {
        let rest: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|index| !chosen.contains(index))
            .collect();
        let take = remaining.min(rest.len());
        for position in 0..take {
            chosen.insert(rest[position * rest.len() / take]);
        }
    }
    chosen.into_iter().take(limit).collect()
}

fn validate_dataset_shape(dataset: &GeometryDataset) -> Result<(), String> {
    let train = dataset
        .samples
        .iter()
        .filter(|sample| is_geometry_scoring_kind(sample.kind) && !sample.kind.is_holdout())
        .count();
    let holdout = dataset
        .samples
        .iter()
        .filter(|sample| is_geometry_scoring_kind(sample.kind) && sample.kind.is_holdout())
        .count();
    if train < 200 || holdout < 80 {
        return Err(format!(
            "capture is incomplete: train={train}, holdout={holdout} (need at least 200/80)"
        ));
    }
    for family in [
        SampleFamily::Neutral,
        SampleFamily::GazeSweep,
        SampleFamily::SlowClose,
        SampleFamily::NaturalBlinks,
        SampleFamily::Closed,
        SampleFamily::HalfOpen,
    ] {
        let train_count = dataset
            .samples
            .iter()
            .filter(|sample| {
                is_geometry_scoring_kind(sample.kind)
                    && !sample.kind.is_holdout()
                    && sample.kind.family() == family
            })
            .count();
        let holdout_count = dataset
            .samples
            .iter()
            .filter(|sample| {
                is_geometry_scoring_kind(sample.kind)
                    && sample.kind.is_holdout()
                    && sample.kind.family() == family
            })
            .count();
        if train_count < 20 || holdout_count < 20 {
            return Err(format!(
                "capture phase {family:?} is incomplete: train={train_count}, holdout={holdout_count}"
            ));
        }
    }
    Ok(())
}

fn values(
    observations: &[Observation],
    eye: usize,
    predicate: impl Fn(&Observation) -> bool,
) -> Vec<f32> {
    observations
        .iter()
        .filter(|observation| predicate(observation) && observation.open[eye].is_finite())
        .map(|observation| observation.open[eye])
        .collect()
}

fn squeeze_values(
    observations: &[Observation],
    eye: usize,
    predicate: impl Fn(&Observation) -> bool,
) -> Vec<f32> {
    observations
        .iter()
        .filter(|observation| predicate(observation) && observation.squeeze[eye].is_finite())
        .map(|observation| observation.squeeze[eye])
        .collect()
}

fn gaze_retention_for_eye(
    observations: &[Observation],
    eye: usize,
    closed_mean: f32,
    span: f32,
    use_direction_bins: bool,
) -> f32 {
    let gaze = values(observations, eye, |observation| {
        observation.kind.family() == SampleFamily::GazeSweep && observation.stable
    });
    let filtered_gaze = temporal_median(&gaze, 5);
    let pooled = ((percentile(&filtered_gaze, 0.10).unwrap_or(closed_mean + span) - closed_mean)
        / span)
        .clamp(0.0, 1.2);
    if !use_direction_bins {
        return pooled;
    }

    let directional: Vec<_> = observations
        .iter()
        .filter_map(|observation| {
            (observation.kind.family() == SampleFamily::GazeSweep
                && observation.stable
                && observation.open[eye].is_finite())
            .then_some(observation.native_gaze_deg[eye].map(|gaze| (gaze, observation.open[eye])))
            .flatten()
        })
        .collect();
    if directional.len() < 5 {
        return pooled;
    }
    let yaw: Vec<_> = directional.iter().map(|(gaze, _)| gaze[0]).collect();
    let pitch: Vec<_> = directional.iter().map(|(gaze, _)| gaze[1]).collect();
    let center_yaw = percentile(&yaw, 0.50).unwrap_or(0.0);
    let center_pitch = percentile(&pitch, 0.50).unwrap_or(0.0);
    let mut worst = pooled;
    for direction in 0..4 {
        let bin: Vec<_> = directional
            .iter()
            .filter(|(gaze, _)| match direction {
                0 => gaze[0] <= center_yaw - 7.0,
                1 => gaze[0] >= center_yaw + 7.0,
                2 => gaze[1] <= center_pitch - 7.0,
                _ => gaze[1] >= center_pitch + 7.0,
            })
            .map(|(_, openness)| *openness)
            .collect();
        if bin.len() >= 3 {
            let retention = ((percentile(&bin, 0.50).unwrap_or(closed_mean) - closed_mean) / span)
                .clamp(0.0, 1.2);
            worst = worst.min(retention);
        }
    }
    worst
}

fn temporal_median(values: &[f32], window: usize) -> Vec<f32> {
    if values.is_empty() || window <= 1 {
        return values.to_vec();
    }
    let radius = window / 2;
    (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(values.len());
            percentile(&values[start..end], 0.50).unwrap_or(values[index])
        })
        .collect()
}

fn mean_variance(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / values.len() as f32;
    (mean, variance)
}

fn pearson(left: &[f32], right: &[f32]) -> f32 {
    let len = left.len().min(right.len());
    if len < 3 {
        return 0.0;
    }
    let (left_mean, _) = mean_variance(&left[..len]);
    let (right_mean, _) = mean_variance(&right[..len]);
    let mut numerator = 0.0;
    let mut left_energy = 0.0;
    let mut right_energy = 0.0;
    for index in 0..len {
        let l = left[index] - left_mean;
        let r = right[index] - right_mean;
        numerator += l * r;
        left_energy += l * l;
        right_energy += r * r;
    }
    numerator / (left_energy * right_energy).sqrt().max(1e-6)
}

fn percentile(values: &[f32], quantile: f32) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let index = ((sorted.len() - 1) as f32 * quantile.clamp(0.0, 1.0)).round() as usize;
    sorted.get(index).copied()
}

fn count_blink_events(values: &[f32], threshold: f32, hysteresis: f32) -> usize {
    let mut below = false;
    let mut count = 0usize;
    let enter = threshold - hysteresis.abs();
    let exit = threshold + hysteresis.abs();
    for value in values {
        if !below && *value < enter {
            below = true;
            count += 1;
        } else if below && *value > exit {
            below = false;
        }
    }
    count
}

fn average(values: [f32; 2]) -> f32 {
    (values[0] + values[1]) * 0.5
}

fn sort_scored(scored: &mut [Scored]) {
    scored.sort_by(|left, right| {
        right
            .metrics
            .score
            .partial_cmp(&left.metrics.score)
            .unwrap_or(CmpOrdering::Equal)
    });
}

fn push_unique(candidates: &mut Vec<SearchParams>, candidate: SearchParams) {
    if !candidates.contains(&candidate) {
        candidates.push(candidate);
    }
}

fn lock_shared(shared: &Arc<Mutex<Shared>>) -> MutexGuard<'_, Shared> {
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn progress(shared: &Arc<Mutex<Shared>>, stage: &str, completed: usize, total: usize) {
    lock_shared(shared).progress(stage, completed, total);
}

fn log(shared: &Arc<Mutex<Shared>>, line: impl Into<String>) {
    lock_shared(shared).push(line);
}

fn fail(shared: &Arc<Mutex<Shared>>, message: String) {
    let mut state = lock_shared(shared);
    state.push(format!("[error] {message}"));
    let log = state.log.clone();
    state.status = Status::Failed { message, log };
}

fn cancelled(shared: &Arc<Mutex<Shared>>, cancel: &AtomicBool) -> bool {
    if !cancel.load(Ordering::Relaxed) {
        return false;
    }
    let mut state = lock_shared(shared);
    state.push("[cancelled] current geometry remains unchanged");
    let log = state.log.clone();
    state.status = Status::Cancelled { log };
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::default_ml_geometry;

    fn observation(
        kind: SampleKind,
        expected_open: Option<f32>,
        open: [f32; 2],
        _time: f32,
    ) -> Observation {
        Observation {
            kind,
            expected_open,
            phase_index: kind as usize,
            native_open: [None; 2],
            native_gaze_deg: [None; 2],
            stable: true,
            presence: 0.10,
            open,
            squeeze: [0.0; 2],
        }
    }

    fn raw_sample(kind: SampleKind, phase_index: usize, time: f32, value: u8) -> GeometrySample {
        GeometrySample {
            kind,
            expected_open: None,
            phase_time_s: time,
            left: vec![value; 16],
            right: vec![value; 16],
            left_size: (4, 4),
            right_size: (4, 4),
            brightness_affine: [[1.0, 0.0]; 2],
            native_open: [None; 2],
            native_gaze: [None; 2],
            commanded_target: None,
            native_pupil_pos: [None; 2],
            frame_generation: [0; 2],
            native_timestamp_us: None,
            phase_index,
        }
    }

    fn localized_closed_sample(
        kind: SampleKind,
        phase_index: usize,
        time: f32,
        sensor_level: u8,
    ) -> GeometrySample {
        let mut sample = raw_sample(kind, phase_index, time, sensor_level);
        sample.left_size = (20, 20);
        sample.right_size = (20, 20);
        sample.left = vec![sensor_level; 400];
        sample.right = vec![sensor_level; 400];
        if kind == SampleKind::Closed {
            for pixel in 0..48 {
                sample.left[pixel] = sensor_level.saturating_add(50);
                sample.right[pixel] = sensor_level.saturating_add(50);
            }
        }
        sample
    }

    #[test]
    fn photometric_coarse_search_keeps_exact_fallback_and_independent_eyes() {
        let baseline = PhotometricCorrection::default();
        let candidates = coarse_photometric_candidates(baseline);
        assert_eq!(
            candidates
                .iter()
                .filter(|candidate| **candidate == baseline)
                .count(),
            1
        );
        assert!(candidates.iter().any(|candidate| {
            candidate.affine[0] != [1.0, 0.0] && candidate.affine[1] == [1.0, 0.0]
        }));
        assert!(candidates.iter().any(|candidate| {
            candidate.affine[1] != [1.0, 0.0] && candidate.affine[0] == [1.0, 0.0]
        }));
    }

    #[test]
    fn photometric_candidate_generation_stays_inside_production_bounds() {
        let baseline = PhotometricCorrection::default();
        let centre = compose_affine_delta(baseline, [true, false], 1.30, 30.0);
        let mut candidates = refinement_photometric_candidates(baseline, centre);
        candidates.extend(field_photometric_candidates(baseline, centre));
        for candidate in candidates {
            for affine in candidate.affine {
                assert!((0.70..=1.30).contains(&affine[0]));
                assert!((-30.0..=30.0).contains(&affine[1]));
            }
            assert!((0.0..=0.75).contains(&candidate.flatten.strength));
            for field in candidate.field {
                assert!((-0.12..=0.12).contains(&field.horizontal));
                assert!((-0.12..=0.12).contains(&field.vertical));
                assert!((-0.08..=0.08).contains(&field.horizontal_curve));
                assert!((-0.08..=0.08).contains(&field.vertical_curve));
            }
        }
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_identity_replay_is_byte_exact_to_captured_preprocessing() {
        let mut sample = raw_sample(SampleKind::Neutral, 0, 1.0, 0);
        sample.left = vec![
            4, 18, 36, 250, 8, 27, 70, 100, 14, 49, 120, 180, 25, 80, 160, 230,
        ];
        sample.right = sample.left.iter().copied().rev().collect();
        sample.brightness_affine = [[1.13, -9.0], [0.87, 12.0]];
        let dataset = GeometryDataset {
            samples: vec![sample.clone()],
        };
        let despeckle = DespeckleParams::default();
        let captured_flatten = FlattenParams {
            enabled: true,
            strength: 0.65,
            radius: 0.33,
        };
        let stability = StabilityReport {
            flags: vec![true],
            ..StabilityReport::default()
        };
        let prepared = research_prepare_samples(
            &dataset,
            default_ml_geometry("pimax_xr5"),
            &stability,
            despeckle,
            captured_flatten,
            FlattenParams::default(),
            [[1.0, 0.0]; 2],
            [None; 2],
            [None; 2],
        );

        let expected = [&sample.left, &sample.right]
            .into_iter()
            .enumerate()
            .map(|(eye, pixels)| {
                let pixels = preprocess::despeckle(pixels, 4, 4, &despeckle);
                let pixels = preprocess::flatten(&pixels, 4, 4, &captured_flatten);
                brightness::apply(
                    &pixels,
                    sample.brightness_affine[eye][0],
                    sample.brightness_affine[eye][1],
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(prepared[0].left, expected[0]);
        assert_eq!(prepared[0].right, expected[1]);
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_coordinate_warp_identity_is_byte_exact() {
        let frame = vec![0, 4, 17, 99, 255, 3, 70, 121, 8, 33, 64, 192];
        assert_eq!(research_apply_coordinate_warp(&frame, (4, 3), None), frame);
        assert_eq!(
            research_apply_coordinate_warp(&frame, (4, 3), Some(ResearchCoordinateWarp::default()),),
            frame
        );
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_coordinate_warp_preserves_a_constant_frame() {
        let frame = vec![137; 13 * 11];
        let warped = research_apply_coordinate_warp(
            &frame,
            (13, 11),
            Some(ResearchCoordinateWarp {
                vertical_bow: ResearchCoordinateWarp::MAX_VERTICAL_BOW,
                radial_k1: -ResearchCoordinateWarp::MAX_RADIAL_K1,
            }),
        );
        assert_eq!(warped, frame);
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_coordinate_warp_nonzero_parameters_move_coordinates() {
        let frame = (0..9)
            .flat_map(|y| std::iter::repeat_n(y * 25, 9))
            .collect::<Vec<u8>>();
        let bowed = research_apply_coordinate_warp(
            &frame,
            (9, 9),
            Some(ResearchCoordinateWarp {
                vertical_bow: 0.10,
                radial_k1: 0.0,
            }),
        );
        let radial = research_apply_coordinate_warp(
            &frame,
            (9, 9),
            Some(ResearchCoordinateWarp {
                vertical_bow: 0.0,
                radial_k1: 0.08,
            }),
        );
        assert_ne!(bowed, frame);
        assert_ne!(bowed[4 * 9], frame[4 * 9]);
        assert_ne!(radial, frame);
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_coordinate_warp_is_horizontally_mirror_equivariant() {
        let (width, height) = (9usize, 7usize);
        let frame = (0..height)
            .flat_map(|y| (0..width).map(move |x| (x * 7 + y * 23) as u8))
            .collect::<Vec<_>>();
        let mirror = |pixels: &[u8]| {
            pixels
                .chunks_exact(width)
                .flat_map(|row| row.iter().rev().copied())
                .collect::<Vec<_>>()
        };
        let warp = Some(ResearchCoordinateWarp {
            vertical_bow: -0.08,
            radial_k1: 0.07,
        });
        let warped = research_apply_coordinate_warp(&frame, (width as u32, height as u32), warp);
        let mirrored_then_warped =
            research_apply_coordinate_warp(&mirror(&frame), (width as u32, height as u32), warp);
        assert_eq!(mirrored_then_warped, mirror(&warped));
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn research_coordinate_warp_rejects_nonfinite_or_invalid_frames() {
        let frame = vec![1, 2, 3, 4, 5];
        assert_eq!(
            research_apply_coordinate_warp(
                &frame,
                (5, 1),
                Some(ResearchCoordinateWarp {
                    vertical_bow: f32::NAN,
                    radial_k1: 0.1,
                }),
            ),
            frame
        );
        assert_eq!(
            research_apply_coordinate_warp(
                &frame,
                (3, 2),
                Some(ResearchCoordinateWarp {
                    vertical_bow: 0.1,
                    radial_k1: 0.1,
                }),
            ),
            frame
        );

        let valid_frame = (0..35).map(|value| value * 7).collect::<Vec<u8>>();
        let bounded = research_apply_coordinate_warp(
            &valid_frame,
            (7, 5),
            Some(ResearchCoordinateWarp {
                vertical_bow: ResearchCoordinateWarp::MAX_VERTICAL_BOW,
                radial_k1: -ResearchCoordinateWarp::MAX_RADIAL_K1,
            }),
        );
        let excessive = research_apply_coordinate_warp(
            &valid_frame,
            (7, 5),
            Some(ResearchCoordinateWarp {
                vertical_bow: 1000.0,
                radial_k1: -1000.0,
            }),
        );
        assert_eq!(excessive, bounded);
    }

    #[test]
    fn clear_finished_removes_stale_result_state() {
        let mut fitter = GeometryFitter::new();
        {
            let mut state = lock_shared(&fitter.shared);
            state.log.push("old result".into());
            state.stage = "done".into();
            state.completed = 10;
            state.total = 10;
            state.status = Status::Cancelled {
                log: state.log.clone(),
            };
        }

        assert!(fitter.clear_finished());
        let state = lock_shared(&fitter.shared);
        assert!(matches!(state.status, Status::Idle));
        assert!(state.log.is_empty());
        assert!(state.stage.is_empty());
        assert_eq!(state.completed, 0);
        assert_eq!(state.total, 0);
    }

    #[test]
    fn clear_finished_never_hides_a_running_job() {
        let mut fitter = GeometryFitter::new();
        {
            let mut state = lock_shared(&fitter.shared);
            state.status = Status::Running {
                stage: "search".into(),
                completed: 1,
                total: 2,
                log: Vec::new(),
            };
        }

        assert!(!fitter.clear_finished());
        assert!(matches!(fitter.status(), Status::Running { .. }));
    }

    #[test]
    fn static_stability_rejects_still_previous_pose_before_late_transition() {
        let mut samples = Vec::new();
        for index in 0..=60 {
            let time = index as f32 * 0.1;
            let value = if time < 2.5 {
                0
            } else if time < 2.8 {
                ((time - 2.5) / 0.3 * 200.0) as u8
            } else {
                200
            };
            samples.push(raw_sample(SampleKind::Closed, 7, time, value));
        }
        let report = capture_stability_flags(&samples, default_ml_geometry("pimax_xr5"));
        assert!(samples
            .iter()
            .zip(&report.flags)
            .filter(|(sample, _)| sample.phase_time_s < 2.8)
            .all(|(_, flag)| !flag));
        assert!(samples
            .iter()
            .zip(&report.flags)
            .filter(|(sample, _)| (2.8..5.8).contains(&sample.phase_time_s))
            .all(|(_, flag)| *flag));
        assert!(!report.flags[59]);
        assert!(!report.flags[60]);
    }

    #[test]
    fn fallback_is_the_tail_window_not_all_trimmed_frames() {
        let mut samples = (0..=20)
            .map(|index| {
                raw_sample(
                    SampleKind::Closed,
                    4,
                    index as f32 * 0.1,
                    if index % 2 == 0 { 0 } else { 200 },
                )
            })
            .collect::<Vec<_>>();
        samples.push(raw_sample(SampleKind::SlowClose, 6, 0.0, 0));
        samples.push(raw_sample(SampleKind::NaturalBlinks, 7, 0.0, 0));
        let report = capture_stability_flags(&samples, default_ml_geometry("pimax_xr5"));
        assert!(samples[..21]
            .iter()
            .zip(&report.flags[..21])
            .filter(|(sample, _)| sample.phase_time_s < 0.8)
            .all(|(_, flag)| !flag));
        assert!(samples[..21]
            .iter()
            .zip(&report.flags[..21])
            .filter(|(sample, _)| (0.8..1.9).contains(&sample.phase_time_s))
            .all(|(_, flag)| *flag));
        assert!(report.degraded_static_phases >= 1);
        assert!(report.flags[21]);
        assert!(report.flags[22]);
    }

    #[test]
    fn gaze_departure_uses_neutral_not_the_intervening_closed_pose() {
        let mut samples = Vec::new();
        for index in 0..=20 {
            let time = index as f32 * 0.1;
            samples.push(raw_sample(SampleKind::Neutral, 1, time, 100));
            samples.push(raw_sample(SampleKind::Closed, 2, time, 0));
        }
        samples.push(raw_sample(SampleKind::GazeSweep, 3, 0.2, 100));
        samples.push(raw_sample(SampleKind::GazeSweep, 3, 0.6, 100));
        samples.push(raw_sample(SampleKind::GazeSweep, 3, 0.7, 200));
        samples.push(raw_sample(SampleKind::GazeSweep, 3, 0.8, 200));
        let report = capture_stability_flags(&samples, default_ml_geometry("pimax_xr5"));
        assert!(!report.flags[42]);
        assert!(!report.flags[43]);
        assert!(!report.flags[44]);
        assert!(report.flags[45]);
    }

    #[test]
    fn gaze_without_a_preceding_valid_neutral_is_excluded() {
        let samples = (0..=20)
            .map(|index| {
                raw_sample(
                    SampleKind::GazeSweep,
                    3,
                    index as f32 * 0.1,
                    100 + index as u8,
                )
            })
            .collect::<Vec<_>>();
        let report = capture_stability_flags(&samples, default_ml_geometry("pimax_xr5"));
        assert!(report.flags.iter().all(|flag| !flag));
    }

    #[test]
    fn identical_closed_and_neutral_tail_is_invalidated() {
        let mut samples = Vec::new();
        for index in 0..=20 {
            let time = index as f32 * 0.1;
            samples.push(raw_sample(SampleKind::Neutral, 1, time, 80));
            samples.push(raw_sample(SampleKind::Closed, 2, time, 80));
        }
        let report = capture_stability_flags(&samples, default_ml_geometry("pimax_xr5"));
        assert_eq!(report.invalid_static_phases, 1);
        assert_eq!(report.valid_closed_phases[0], 0);
        assert!(samples
            .iter()
            .zip(&report.flags)
            .filter(|(sample, _)| sample.kind == SampleKind::Closed)
            .all(|(_, flag)| !flag));
    }

    #[test]
    fn localized_closed_pose_is_not_hidden_by_full_frame_sensor_noise() {
        let mut samples = Vec::new();
        for index in 0..=20 {
            let time = index as f32 * 0.1;
            let sensor_level = 80 + (index % 2) as u8 * 2;
            samples.push(localized_closed_sample(
                SampleKind::Neutral,
                1,
                time,
                sensor_level,
            ));
            samples.push(localized_closed_sample(
                SampleKind::Closed,
                2,
                time,
                sensor_level,
            ));
        }

        let report = capture_stability_flags(&samples, [MlGeometry::default(); 2]);
        assert_eq!(report.invalid_static_phases, 0);
        assert_eq!(report.valid_closed_phases[0], 1);
        assert!(samples
            .iter()
            .zip(&report.flags)
            .filter(|(sample, _)| sample.kind == SampleKind::Closed)
            .any(|(_, flag)| *flag));
    }

    #[cfg(feature = "research-synthetic-eye-lab")]
    #[test]
    fn spatial_gain_identity_is_exact_and_extends_smoothly_past_the_fixed_crop() {
        let frame = vec![100u8; 16];
        let geometry = MlGeometry {
            crop_left: 0.25,
            crop_right: 0.25,
            crop_top: 0.25,
            crop_bottom: 0.25,
            ..MlGeometry::default()
        };
        let identity = research_apply_spatial_gain(
            &frame,
            (4, 4),
            geometry,
            Some(SpatialGainField::default()),
        );
        assert_eq!(identity, frame);

        let changed = research_apply_spatial_gain(
            &frame,
            (4, 4),
            geometry,
            Some(SpatialGainField {
                vertical: 0.12,
                ..SpatialGainField::default()
            }),
        );
        for y in 0..4 {
            for x in 0..4 {
                let value = changed[y * 4 + x];
                if (1..3).contains(&x) && y == 1 {
                    assert_eq!(value, 88);
                } else if (1..3).contains(&x) && y == 2 {
                    assert_eq!(value, 112);
                } else if y == 0 {
                    assert_eq!(value, 70);
                } else if y == 3 {
                    assert_eq!(value, 130);
                } else {
                    assert!(matches!(value, 88 | 112));
                }
            }
        }
    }

    #[test]
    fn zero_params_are_byte_for_byte_the_fallback() {
        let baseline = default_ml_geometry("pimax_xr5");
        assert_eq!(
            geometry_from_params(baseline, SearchParams::default()),
            baseline
        );
    }

    #[test]
    fn search_never_changes_mirror_or_scale_x_and_stays_bounded() {
        let baseline = default_ml_geometry("pimax_xr5");
        let geometry = geometry_from_params(baseline, SearchParams([1.0, -1.0, 1.0, 1.0, 1.0]));
        for eye in 0..2 {
            assert_eq!(geometry[eye].mirror_h, baseline[eye].mirror_h);
            assert_eq!(geometry[eye].scale_x, baseline[eye].scale_x);
            assert!(geometry[eye].crop_left >= 0.0);
            assert!(geometry[eye].crop_right >= 0.0);
            assert!(geometry[eye].crop_top >= 0.0);
            assert!(geometry[eye].crop_bottom >= 0.0);
            assert!((geometry[eye].rotate_deg - baseline[eye].rotate_deg).abs() <= 8.01);
        }
        assert!(geometry[0].crop_right + 1e-6 >= XR5_MIN_INNER_CROP);
        assert!(geometry[1].crop_left + 1e-6 >= XR5_MIN_INNER_CROP);
    }

    #[test]
    fn labelled_motion_beats_constant_confident_output() {
        let mut good = Vec::new();
        let mut bad = Vec::new();
        for index in 0..20 {
            let jitter = (index % 3) as f32 * 0.002;
            good.push(observation(
                SampleKind::Neutral,
                None,
                [0.80 + jitter; 2],
                index as f32,
            ));
            bad.push(observation(
                SampleKind::Neutral,
                None,
                [0.80; 2],
                index as f32,
            ));
            good.push(observation(
                SampleKind::GazeSweep,
                None,
                [0.79 + jitter; 2],
                index as f32,
            ));
            bad.push(observation(
                SampleKind::GazeSweep,
                None,
                [0.80; 2],
                index as f32,
            ));
            let target = if index < 10 {
                1.0 - index as f32 / 9.0
            } else {
                (index - 10) as f32 / 9.0
            };
            good.push(observation(
                SampleKind::SlowClose,
                Some(target),
                [0.20 + 0.60 * target; 2],
                index as f32,
            ));
            bad.push(observation(
                SampleKind::SlowClose,
                Some(target),
                [0.80; 2],
                index as f32,
            ));
            let blink = if matches!(index, 2 | 6 | 10 | 14 | 18) {
                0.20
            } else {
                0.80
            };
            good.push(observation(
                SampleKind::NaturalBlinks,
                None,
                [blink; 2],
                index as f32,
            ));
            bad.push(observation(
                SampleKind::NaturalBlinks,
                None,
                [0.80; 2],
                index as f32,
            ));
            good.push(observation(
                SampleKind::Closed,
                None,
                [0.20; 2],
                index as f32,
            ));
            bad.push(observation(
                SampleKind::Closed,
                None,
                [0.80; 2],
                index as f32,
            ));
        }
        let image = ImageAccum {
            spatial_std_sum: 10.0,
            saturation_sum: 0.0,
            pixels: 100_000,
            frames: 100,
            motion_sum: 2_500.0,
            motion_pixels: 100_000,
        };
        let good = metrics_from_observations(&good, &image);
        let bad = metrics_from_observations(&bad, &image);
        assert!(good.score > bad.score + 0.25, "good={good:?} bad={bad:?}");
        assert!(average(good.monotonicity) > 0.95);
        assert!(average(bad.separation) < 0.1);
        assert!(capture_quality_issue(&good).is_none());
        assert!(capture_quality_issue(&bad).is_some());
    }

    #[test]
    fn gaze_failure_fixture_exposes_worst_eye_and_false_squeeze() {
        let mut observations = Vec::new();
        for index in 0..20 {
            observations.push(observation(
                SampleKind::Neutral,
                None,
                [0.56, 0.56],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::Closed,
                None,
                [0.05, 0.05],
                index as f32,
            ));
            let target = index as f32 / 19.0;
            observations.push(observation(
                SampleKind::SlowClose,
                Some(target),
                [0.05 + 0.51 * target; 2],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::NaturalBlinks,
                None,
                [if index % 4 == 0 { 0.05 } else { 0.56 }; 2],
                index as f32,
            ));
            let mut gaze = observation(SampleKind::GazeSweep, None, [0.52, 0.39], index as f32);
            gaze.squeeze = [0.01, 0.295];
            observations.push(gaze);
        }
        let metrics = metrics_from_observations(&observations, &ImageAccum::default());
        assert!(metrics.evidence_valid);
        assert!((metrics.gaze_retention[0] - 0.922).abs() < 0.01);
        assert!((metrics.gaze_retention[1] - 0.667).abs() < 0.01);
        assert!((metrics.gaze_squeeze_fp[1] - 0.295).abs() < 0.01);
        assert!(metrics.gaze_asymmetry > 0.24);

        let mut safe_baseline = metrics.clone();
        safe_baseline.gaze_retention = [1.0; 2];
        safe_baseline.gaze_squeeze_fp = [0.0; 2];
        safe_baseline.gaze_asymmetry = 0.0;
        assert!(!admissible(&metrics, &safe_baseline));
    }

    #[test]
    fn retention_p10_is_robust_to_two_frame_blinks() {
        let mut clean = Vec::new();
        let mut blinked = Vec::new();
        for index in 0..20 {
            for target in [&mut clean, &mut blinked] {
                target.push(observation(
                    SampleKind::Neutral,
                    None,
                    [0.8; 2],
                    index as f32,
                ));
                target.push(observation(
                    SampleKind::Closed,
                    None,
                    [0.2; 2],
                    index as f32,
                ));
                let expected = index as f32 / 19.0;
                target.push(observation(
                    SampleKind::SlowClose,
                    Some(expected),
                    [0.2 + 0.6 * expected; 2],
                    index as f32,
                ));
                target.push(observation(
                    SampleKind::NaturalBlinks,
                    None,
                    [0.8; 2],
                    index as f32,
                ));
            }
            clean.push(observation(
                SampleKind::GazeSweep,
                None,
                [0.76; 2],
                index as f32,
            ));
            blinked.push(observation(
                SampleKind::GazeSweep,
                None,
                [if matches!(index, 5 | 6 | 14 | 15) {
                    0.2
                } else {
                    0.76
                }; 2],
                index as f32,
            ));
        }
        let clean = metrics_from_observations(&clean, &ImageAccum::default());
        let blinked = metrics_from_observations(&blinked, &ImageAccum::default());
        assert!((clean.gaze_retention[0] - blinked.gaze_retention[0]).abs() < 0.01);
    }

    #[test]
    fn median_open_closed_references_resist_minority_contamination() {
        let mut clean = Vec::new();
        let mut contaminated = Vec::new();
        for index in 0..20 {
            for target in [&mut clean, &mut contaminated] {
                target.push(observation(
                    SampleKind::Neutral,
                    None,
                    [0.8; 2],
                    index as f32,
                ));
                target.push(observation(
                    SampleKind::GazeSweep,
                    None,
                    [0.7; 2],
                    index as f32,
                ));
                let expected = index as f32 / 19.0;
                target.push(observation(
                    SampleKind::SlowClose,
                    Some(expected),
                    [0.2 + 0.6 * expected; 2],
                    index as f32,
                ));
                target.push(observation(
                    SampleKind::NaturalBlinks,
                    None,
                    [0.8; 2],
                    index as f32,
                ));
            }
            clean.push(observation(
                SampleKind::Closed,
                None,
                [0.2; 2],
                index as f32,
            ));
            contaminated.push(observation(
                SampleKind::Closed,
                None,
                [if index < 4 { 0.8 } else { 0.2 }; 2],
                index as f32,
            ));
        }
        let clean = metrics_from_observations(&clean, &ImageAccum::default());
        let contaminated = metrics_from_observations(&contaminated, &ImageAccum::default());
        assert_eq!(clean.closed_ref, [0.2; 2]);
        assert_eq!(contaminated.closed_ref, [0.2; 2]);
        assert!((clean.gaze_retention[0] - contaminated.gaze_retention[0]).abs() < 1e-5);
    }

    #[test]
    fn half_quality_ignores_unstable_reaction_frames() {
        let mut observations = Vec::new();
        for index in 0..10 {
            observations.push(observation(
                SampleKind::Neutral,
                None,
                [0.8; 2],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::Closed,
                None,
                [0.2; 2],
                index as f32,
            ));
        }
        for block in [2usize, 4usize] {
            for index in 0..10 {
                let mut value = observation(
                    SampleKind::HalfOpen,
                    Some(0.5),
                    [if index < 5 { 0.8 } else { 0.5 }; 2],
                    index as f32,
                );
                value.phase_index = block;
                value.stable = index >= 5;
                observations.push(value);
            }
        }
        let quality = half_quality(&observations).unwrap();
        assert!((quality.position[0] - 0.5).abs() < 1e-5);
        assert!(quality.block_disagreement[0] < 1e-5);
    }

    #[test]
    fn legacy_capture_without_native_gaze_still_has_finite_gaze_metrics() {
        let mut observations = Vec::new();
        for index in 0..12 {
            for (kind, expected, open) in [
                (SampleKind::Neutral, None, 0.8),
                (SampleKind::Closed, None, 0.2),
                (SampleKind::GazeSweep, None, 0.78),
                (
                    SampleKind::SlowClose,
                    Some(index as f32 / 11.0),
                    0.2 + 0.6 * index as f32 / 11.0,
                ),
                (SampleKind::NaturalBlinks, None, 0.8),
            ] {
                observations.push(observation(kind, expected, [open; 2], index as f32));
            }
        }
        let metrics = metrics_from_observations(&observations, &ImageAccum::default());
        assert!(metrics.evidence_valid);
        assert_eq!(metrics.gaze_evidence_rate, 0.0);
        assert!(metrics.gaze_retention.iter().all(|value| value.is_finite()));
        assert!(metrics
            .gaze_squeeze_fp
            .iter()
            .all(|value| value.is_finite()));
        assert!(metrics.score.is_finite());
    }

    #[test]
    fn zero_stable_gaze_frames_invalidate_the_objective() {
        let mut observations = Vec::new();
        for index in 0..12 {
            observations.push(observation(
                SampleKind::Neutral,
                None,
                [0.8; 2],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::Closed,
                None,
                [0.2; 2],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::SlowClose,
                Some(index as f32 / 11.0),
                [0.2 + 0.6 * index as f32 / 11.0; 2],
                index as f32,
            ));
            let mut gaze = observation(SampleKind::GazeSweep, None, [0.78; 2], index as f32);
            gaze.stable = false;
            observations.push(gaze);
        }
        let metrics = metrics_from_observations(&observations, &ImageAccum::default());
        assert!(!metrics.evidence_valid);
        assert_eq!(metrics.gaze_evidence_rate, 0.0);
        assert_eq!(metrics.score, 0.0);
    }

    #[test]
    fn explicit_half_evidence_does_not_change_the_normal_fit_objective() {
        let mut observations = Vec::new();
        for index in 0..24 {
            let target = if index < 12 {
                1.0 - index as f32 / 11.0
            } else {
                (index - 12) as f32 / 11.0
            };
            for (kind, expected, open) in [
                (SampleKind::Neutral, None, 0.80),
                (
                    SampleKind::GazeSweep,
                    None,
                    0.78 + (index % 3) as f32 * 0.002,
                ),
                (SampleKind::SlowClose, Some(target), 0.20 + 0.60 * target),
                (
                    SampleKind::NaturalBlinks,
                    None,
                    if index % 5 == 0 { 0.20 } else { 0.80 },
                ),
                (SampleKind::Closed, None, 0.20),
            ] {
                observations.push(observation(kind, expected, [open; 2], index as f32));
            }
        }
        let image = ImageAccum {
            spatial_std_sum: 12.0,
            saturation_sum: 5.0,
            pixels: 120_000,
            frames: 120,
            motion_sum: 2_400.0,
            motion_pixels: 120_000,
        };
        let without_half = metrics_from_observations(&observations, &image);
        for index in 0..40 {
            observations.push(observation(
                SampleKind::HalfOpen,
                Some(0.5),
                [0.05 + index as f32 * 0.02, 0.95 - index as f32 * 0.02],
                index as f32,
            ));
        }
        let with_half = metrics_from_observations(&observations, &image);
        assert_eq!(with_half, without_half);
    }

    #[test]
    fn acceptance_needs_real_holdout_gain() {
        let baseline = GeometryMetrics {
            evidence_valid: true,
            score: 0.70,
            separation: [2.0; 2],
            monotonicity: [0.8; 2],
            presence_rate: 1.0,
            finite_rate: 1.0,
            image_std: 0.10,
            motion_energy: 0.03,
            ..GeometryMetrics::default()
        };
        let mut training_winner = baseline.clone();
        training_winner.score = 0.82;
        let mut tiny_holdout_gain = baseline.clone();
        tiny_holdout_gain.score = 0.72;
        let (accepted, _) = acceptance(
            &baseline,
            &training_winner,
            &baseline,
            &tiny_holdout_gain,
            false,
            false,
        );
        assert!(!accepted);
    }

    #[test]
    fn acceptance_has_a_reachable_positive_path() {
        let baseline = GeometryMetrics {
            evidence_valid: true,
            score: 0.70,
            separation: [2.0; 2],
            monotonicity: [0.80; 2],
            presence_rate: 1.0,
            finite_rate: 1.0,
            image_information: 0.8,
            image_std: 0.10,
            motion_energy: 0.03,
            stability: 0.8,
            neutral_noise_per_eye: [0.04; 2],
            gaze_noise_per_eye: [0.08; 2],
            ..GeometryMetrics::default()
        };
        let mut candidate_train = baseline.clone();
        candidate_train.score = 0.84;
        candidate_train.separation = [2.3; 2];
        candidate_train.monotonicity = [0.88; 2];
        let mut candidate_holdout = baseline.clone();
        candidate_holdout.score = 0.80;
        candidate_holdout.separation = [2.2; 2];
        candidate_holdout.monotonicity = [0.86; 2];
        let (accepted, reason) = acceptance(
            &baseline,
            &candidate_train,
            &baseline,
            &candidate_holdout,
            false,
            false,
        );
        assert!(accepted, "{reason}");
    }

    #[test]
    fn missing_closed_evidence_is_invalid_not_a_perfect_separation() {
        let observations: Vec<_> = (0..20)
            .flat_map(|index| {
                [
                    observation(SampleKind::Neutral, None, [0.8; 2], index as f32),
                    observation(SampleKind::SlowClose, Some(0.9), [0.75; 2], index as f32),
                ]
            })
            .collect();
        let metrics = metrics_from_observations(&observations, &ImageAccum::default());
        assert!(!metrics.evidence_valid);
        assert_eq!(metrics.score, 0.0);
        assert_eq!(metrics.separation, [0.0; 2]);
    }

    fn prepared(kind: SampleKind) -> PreparedSample {
        PreparedSample {
            kind,
            expected_open: None,
            phase_index: kind as usize,
            native_open: [None; 2],
            native_gaze_deg: [None; 2],
            stable: true,
            left: vec![0; 4],
            right: vec![0; 4],
            left_size: (2, 2),
            right_size: (2, 2),
        }
    }

    #[test]
    fn stratified_selection_never_crosses_the_holdout_boundary() {
        let kinds = [
            SampleKind::Neutral,
            SampleKind::GazeSweep,
            SampleKind::SlowClose,
            SampleKind::NaturalBlinks,
            SampleKind::Closed,
            SampleKind::HoldoutNeutral,
            SampleKind::HoldoutGazeSweep,
            SampleKind::HoldoutSlowClose,
            SampleKind::HoldoutNaturalBlinks,
            SampleKind::HoldoutClosed,
        ];
        let mut samples = Vec::new();
        for kind in kinds {
            for _ in 0..25 {
                samples.push(prepared(kind));
            }
        }
        let train = stratified_indices(&samples, false, 80);
        let holdout = stratified_indices(&samples, true, 80);
        assert!(train.iter().all(|index| !samples[*index].kind.is_holdout()));
        assert!(holdout
            .iter()
            .all(|index| samples[*index].kind.is_holdout()));
        assert!(holdout
            .iter()
            .any(|index| samples[*index].kind.family() == SampleFamily::Closed));
    }

    #[test]
    fn unilateral_winks_never_become_closed_geometry_or_photometric_evidence() {
        for kind in [
            SampleKind::LeftWink,
            SampleKind::RightWink,
            SampleKind::HoldoutLeftWink,
            SampleKind::HoldoutRightWink,
        ] {
            assert!(!is_geometry_scoring_kind(kind));
        }

        let wink_samples = (0..40)
            .flat_map(|index| {
                let time = index as f32 * 0.05;
                [
                    raw_sample(SampleKind::LeftWink, 91, time, 40),
                    raw_sample(SampleKind::RightWink, 92, time, 210),
                    raw_sample(SampleKind::HoldoutLeftWink, 93, time, 45),
                    raw_sample(SampleKind::HoldoutRightWink, 94, time, 205),
                ]
            })
            .collect::<Vec<_>>();
        let stability = capture_stability_flags(&wink_samples, default_ml_geometry("pimax_xr5"));
        assert!(stability.flags.iter().all(|flag| !flag));
        assert_eq!(stability.valid_closed_phases, [0, 0]);
        assert_eq!(stability.invalid_static_phases, 0);

        let prepared_winks = [
            SampleKind::LeftWink,
            SampleKind::RightWink,
            SampleKind::HoldoutLeftWink,
            SampleKind::HoldoutRightWink,
        ]
        .into_iter()
        .map(prepared)
        .collect::<Vec<_>>();
        assert!(stratified_indices(&prepared_winks, false, usize::MAX).is_empty());
        assert!(stratified_indices(&prepared_winks, true, usize::MAX).is_empty());

        let mut baseline = Vec::new();
        for index in 0..30 {
            let t = index as f32 / 29.0;
            baseline.push(observation(
                SampleKind::Neutral,
                None,
                [0.80; 2],
                index as f32,
            ));
            baseline.push(observation(
                SampleKind::Closed,
                None,
                [0.20; 2],
                index as f32,
            ));
            baseline.push(observation(
                SampleKind::SlowClose,
                Some(t),
                [0.20 + 0.60 * t; 2],
                index as f32,
            ));
            baseline.push(observation(
                SampleKind::GazeSweep,
                None,
                [0.78; 2],
                index as f32,
            ));
            baseline.push(observation(
                SampleKind::NaturalBlinks,
                None,
                if index % 3 == 0 { [0.20; 2] } else { [0.80; 2] },
                index as f32,
            ));
        }
        let expected = metrics_from_observations(&baseline, &ImageAccum::default());
        let mut contaminated = baseline;
        for index in 0..200 {
            contaminated.push(observation(
                SampleKind::LeftWink,
                None,
                [0.20, 0.80],
                index as f32,
            ));
            contaminated.push(observation(
                SampleKind::RightWink,
                None,
                [0.80, 0.20],
                index as f32,
            ));
        }
        assert_eq!(
            metrics_from_observations(&contaminated, &ImageAccum::default()),
            expected
        );
    }

    #[test]
    fn audit_matrix_keeps_the_active_geometry_as_its_reference() {
        let baseline = default_ml_geometry("pimax_xr5");
        let cases = audit_case_specs(baseline);
        assert_eq!(cases[0].name, "active reference");
        assert_eq!(cases[0].geometry, baseline);
        // The preset already touches the outer frame edge and the fixed inner LED crop;
        // probes that collapse onto the reference are intentionally deduplicated.
        // Horizontal inward motion has no legal room at the maximum 60%-wide preset;
        // it becomes available only together with a smaller-window candidate.
        for axis in 1..5 {
            assert!(
                cases.iter().any(|case| case.axis == Some(axis)),
                "missing audit axis {axis}"
            );
        }
        assert!(cases.iter().all(|case| {
            case.geometry[0].crop_right + 1e-6 >= XR5_MIN_INNER_CROP
                && case.geometry[1].crop_left + 1e-6 >= XR5_MIN_INNER_CROP
        }));
        assert!(
            cases.len() * AUDIT_FOLDS * 6 * AUDIT_FRAMES_PER_FAMILY_FOLD <= 13_000,
            "audit must stay near one normal-fit evaluation budget"
        );
    }

    #[test]
    fn audit_folds_are_balanced_blocked_and_do_not_share_adjacent_frames() {
        let kinds = [
            SampleKind::Neutral,
            SampleKind::HalfOpen,
            SampleKind::GazeSweep,
            SampleKind::SlowClose,
            SampleKind::NaturalBlinks,
            SampleKind::Closed,
            SampleKind::HoldoutNeutral,
            SampleKind::HoldoutHalfOpen,
            SampleKind::HoldoutGazeSweep,
            SampleKind::HoldoutSlowClose,
            SampleKind::HoldoutNaturalBlinks,
            SampleKind::HoldoutClosed,
        ];
        let mut samples = Vec::new();
        for kind in kinds {
            for _ in 0..72 {
                samples.push(prepared(kind));
            }
        }
        let folds = audit_fold_indices(&samples).expect("balanced fixture should form folds");
        assert_eq!(folds.len(), AUDIT_FOLDS);
        let mut owner = vec![None; samples.len()];
        for (fold, indices) in folds.iter().enumerate() {
            assert_eq!(indices.len(), 6 * AUDIT_FRAMES_PER_FAMILY_FOLD);
            assert!(indices
                .iter()
                .all(|index| !samples[*index].kind.is_holdout()));
            for family in [
                SampleFamily::Neutral,
                SampleFamily::HalfOpen,
                SampleFamily::GazeSweep,
                SampleFamily::SlowClose,
                SampleFamily::NaturalBlinks,
                SampleFamily::Closed,
            ] {
                assert_eq!(
                    indices
                        .iter()
                        .filter(|index| samples[**index].kind.family() == family)
                        .count(),
                    AUDIT_FRAMES_PER_FAMILY_FOLD
                );
            }
            for index in indices {
                assert!(owner[*index].replace(fold).is_none());
            }
        }
        for (index, fold) in owner.iter().enumerate() {
            let Some(fold) = fold else { continue };
            for neighbour in index.saturating_sub(2)..=(index + 2).min(samples.len() - 1) {
                if samples[neighbour].kind == samples[index].kind {
                    assert!(owner[neighbour].is_none_or(|other| other == *fold));
                }
            }
        }
    }

    #[test]
    fn real_protocol_folds_retain_open_half_and_closed_legacy_evidence() {
        let phases = [
            (SampleKind::Neutral, 80usize, 0u32),
            (SampleKind::HalfOpen, 80, 0),
            (SampleKind::Closed, 80, 0),
            (SampleKind::GazeSweep, 160, 0),
            (SampleKind::SlowClose, 200, 3),
            (SampleKind::NaturalBlinks, 140, 0),
            (SampleKind::Closed, 80, 0),
            (SampleKind::HalfOpen, 80, 0),
            (SampleKind::Neutral, 80, 0),
            (SampleKind::HoldoutNeutral, 60, 0),
            (SampleKind::HoldoutHalfOpen, 60, 0),
            (SampleKind::HoldoutClosed, 60, 0),
            (SampleKind::HoldoutGazeSweep, 100, 0),
            (SampleKind::HoldoutSlowClose, 100, 1),
            (SampleKind::HoldoutNaturalBlinks, 120, 0),
        ];
        let mut samples = Vec::new();
        for (phase_index, (kind, count, cycles)) in phases.into_iter().enumerate() {
            for index in 0..count {
                let mut sample = prepared(kind);
                sample.phase_index = phase_index;
                if kind.family() == SampleFamily::HalfOpen {
                    sample.expected_open = Some(0.5);
                }
                if cycles > 0 {
                    let phase = (index as f32 / count as f32 * cycles as f32).fract();
                    sample.expected_open = Some(if phase < 0.5 {
                        1.0 - 2.0 * phase
                    } else {
                        2.0 * (phase - 0.5)
                    });
                }
                samples.push(sample);
            }
        }
        let folds = audit_fold_indices(&samples).expect("real protocol should form folds");
        for indices in folds {
            let observations: Vec<_> = indices
                .iter()
                .map(|index| {
                    let sample = &samples[*index];
                    let open = match sample.kind.family() {
                        SampleFamily::Closed => 0.20,
                        SampleFamily::HalfOpen => 0.50,
                        SampleFamily::SlowClose => {
                            0.20 + 0.60 * sample.expected_open.unwrap_or(1.0)
                        }
                        _ => 0.80,
                    };
                    let mut observation =
                        observation(sample.kind, sample.expected_open, [open; 2], *index as f32);
                    observation.phase_index = sample.phase_index;
                    observation
                })
                .collect();
            let legacy = legacy_fold_metrics(&observations);
            assert!(legacy.valid, "fold lost a required open/half/closed bin");
            assert!(legacy.half_error < 0.08, "legacy={}", legacy.half_error);
        }
    }

    #[test]
    fn legacy_audit_rewards_absolute_span_and_a_centered_half() {
        let mut good = Vec::new();
        let mut weak = Vec::new();
        for index in 0..30 {
            good.push(observation(
                SampleKind::Neutral,
                None,
                [0.80; 2],
                index as f32,
            ));
            weak.push(observation(
                SampleKind::Neutral,
                None,
                [0.65; 2],
                index as f32,
            ));
            good.push(observation(
                SampleKind::Closed,
                None,
                [0.20; 2],
                index as f32,
            ));
            weak.push(observation(
                SampleKind::Closed,
                None,
                [0.35; 2],
                index as f32,
            ));
            good.push(observation(
                SampleKind::HalfOpen,
                Some(0.5),
                [0.50; 2],
                index as f32,
            ));
            weak.push(observation(
                SampleKind::HalfOpen,
                Some(0.5),
                [0.62; 2],
                index as f32,
            ));
            let target = index as f32 / 29.0;
            good.push(observation(
                SampleKind::SlowClose,
                Some(target),
                [0.20 + 0.60 * target; 2],
                index as f32,
            ));
            weak.push(observation(
                SampleKind::SlowClose,
                Some(target),
                [0.35 + 0.30 * target.powf(0.25); 2],
                index as f32,
            ));
        }
        let good = legacy_fold_metrics(&good);
        let weak = legacy_fold_metrics(&weak);
        assert!(good.valid && weak.valid);
        assert!(good.span > weak.span + 0.20);
        assert!(good.half_error < weak.half_error);
        assert!(good.score > weak.score + 1.0);
    }

    fn half_quality_observations(half_a: f32, half_b: f32) -> Vec<Observation> {
        let mut observations = Vec::new();
        for index in 0..20 {
            observations.push(observation(
                SampleKind::Neutral,
                None,
                [0.80; 2],
                index as f32,
            ));
            observations.push(observation(
                SampleKind::Closed,
                None,
                [0.20; 2],
                index as f32,
            ));
            let mut first = observation(SampleKind::HalfOpen, Some(0.5), [half_a; 2], index as f32);
            first.phase_index = 10;
            observations.push(first);
            let mut second =
                observation(SampleKind::HalfOpen, Some(0.5), [half_b; 2], index as f32);
            second.phase_index = 20;
            observations.push(second);
        }
        observations
    }

    #[test]
    fn half_quality_accepts_repeatable_centered_blocks() {
        let quality = half_quality(&half_quality_observations(0.49, 0.51))
            .expect("repeatable centered HALF blocks should pass");
        assert!((quality.position[0] - 0.5).abs() < 0.01);
        assert!(quality.normalized_stddev[0] < 0.02);
        assert!(quality.block_disagreement[0] < 0.04);
    }

    #[test]
    fn half_quality_rejects_a_pose_near_open() {
        let error = half_quality(&half_quality_observations(0.74, 0.76))
            .expect_err("an almost-open HALF pose must not become audit evidence");
        assert!(error.contains("landed at"), "{error}");
    }

    #[test]
    fn half_quality_rejects_disagreeing_blocks() {
        let error = half_quality(&half_quality_observations(0.40, 0.60))
            .expect_err("two materially different HALF blocks must be repeated");
        assert!(error.contains("disagreed"), "{error}");
    }

    #[test]
    fn native_openness_disagreement_is_warning_only() {
        let mut observations = half_quality_observations(0.49, 0.51);
        for observation in &mut observations {
            observation.native_open = [Some(match observation.kind {
                SampleKind::Neutral => 0.90,
                SampleKind::Closed => 0.10,
                SampleKind::HalfOpen => 0.90,
                _ => 0.50,
            }); 2];
        }
        let quality = half_quality(&observations)
            .expect("native openness must never gate otherwise good EyeNet evidence");
        assert_eq!(quality.native_coverage, [1.0; 2]);
        assert_eq!(quality.warnings.len(), 2);
    }
}
