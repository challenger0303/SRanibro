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

use crate::core::types::{DespeckleParams, FlattenParams, MlGeometry};
use crate::geometry_calib::{GeometryDataset, SampleFamily, SampleKind};
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
const XR5_MIN_INNER_CROP: f32 = 0.35;
const AUDIT_FOLDS: usize = 5;
const AUDIT_BLOCK_FRAMES: usize = 12;
const AUDIT_GUARD_FRAMES: usize = 1;
const AUDIT_FRAMES_PER_FAMILY_FOLD: usize = 16;

#[derive(Clone, Debug)]
pub struct FitInputs {
    pub model_path: PathBuf,
    pub dataset: GeometryDataset,
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
    pub monotonicity: [f32; 2],
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
    pub accepted: bool,
    pub reason: String,
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
        if !inputs.model_path.is_file() {
            return Err(StartError {
                message: format!(
                    "SRanipal EyePrediction model not found: {}",
                    inputs.model_path.display()
                ),
                inputs,
            });
        }
        if let Err(message) = validate_dataset_shape(&inputs.dataset) {
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
}

impl Drop for GeometryFitter {
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
    left: Vec<u8>,
    right: Vec<u8>,
    left_size: (u32, u32),
    right_size: (u32, u32),
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
    presence: f32,
    open: [f32; 2],
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

fn run(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: FitInputs) {
    log(&shared, format!("[load] {}", inputs.model_path.display()));
    let map = match tvm_params::parse_map(&inputs.model_path.to_string_lossy()) {
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
    let motion_seed =
        match estimate_dataset_motion_seed(&inputs.dataset, inputs.baseline, inputs.mirrors) {
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

    let appearance_seed = match estimate_dataset_appearance_seed(&inputs.dataset, inputs.baseline) {
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
    let prepare_total = inputs.dataset.samples.len();
    progress(&shared, "preparing captured frames", 0, prepare_total);
    let mut prepared = Vec::with_capacity(prepare_total);
    for (index, sample) in inputs.dataset.samples.into_iter().enumerate() {
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
            left,
            right,
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
        if index % 20 == 0 || index + 1 == prepare_total {
            progress(
                &shared,
                "preparing captured frames",
                index + 1,
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
    let mut work_done = 0usize;

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
/// that originally found the XR5 preset. This is diagnostic only: no geometry is
/// accepted, previewed, or persisted from this path.
fn run_audit(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: FitInputs) {
    log(
        &shared,
        format!("[audit load] {}", inputs.model_path.display()),
    );
    let map = match tvm_params::parse_map(&inputs.model_path.to_string_lossy()) {
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
    let prepare_total = inputs.dataset.samples.len();
    progress(&shared, "preparing audit frames", 0, prepare_total);
    let mut prepared = Vec::with_capacity(prepare_total);
    for (index, sample) in inputs.dataset.samples.into_iter().enumerate() {
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
            left,
            right,
            left_size: sample.left_size,
            right_size: sample.right_size,
        });
        if index % 20 == 0 || index + 1 == prepare_total {
            progress(&shared, "preparing audit frames", index + 1, prepare_total);
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
        .filter(|(_, sample)| !sample.kind.is_holdout())
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
        ("legacy neighbour inner .35", Some(0.35), None),
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
            if samples[cursor].kind.family() != family {
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
                observation.kind.family() == SampleFamily::Closed
                    || (observation.kind.family() == SampleFamily::SlowClose
                        && observation
                            .expected_open
                            .is_some_and(|target| target <= 0.20))
            }),
            values(observations, eye, |observation| {
                observation.kind.family() == SampleFamily::HalfOpen
            }),
            values(observations, eye, |observation| {
                observation.kind.family() == SampleFamily::Neutral
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
            observation.kind == SampleKind::Neutral
        });
        let closed = values(observations, eye, |observation| {
            observation.kind == SampleKind::Closed
        });
        let half = values(observations, eye, |observation| {
            observation.kind == SampleKind::HalfOpen
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
            if observation.kind == SampleKind::HalfOpen && observation.open[eye].is_finite() {
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
            .filter(|observation| observation.kind == SampleKind::HalfOpen)
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
                    .filter(|observation| observation.kind == kind)
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
        let contributes_to_fit = sample.kind.family() != SampleFamily::HalfOpen;
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
            presence: output[0],
            open: [output[1], output[2]],
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
    let filtered;
    let observations = if all_observations
        .iter()
        .any(|observation| observation.kind.family() == SampleFamily::HalfOpen)
    {
        filtered = all_observations
            .iter()
            .copied()
            .filter(|observation| observation.kind.family() != SampleFamily::HalfOpen)
            .collect::<Vec<_>>();
        filtered.as_slice()
    } else {
        all_observations
    };
    if observations.is_empty() {
        return GeometryMetrics::default();
    }
    let finite = observations
        .iter()
        .filter(|observation| {
            observation.presence.is_finite()
                && observation.open.iter().all(|value| value.is_finite())
        })
        .count();
    let finite_rate = finite as f32 / observations.len() as f32;
    let presence_rate = observations
        .iter()
        .filter(|observation| observation.presence.is_finite() && observation.presence >= 0.02)
        .count() as f32
        / observations.len() as f32;

    let mut separation = [0.0; 2];
    let mut monotonicity = [0.0; 2];
    let mut neutral_ratio = [1.0; 2];
    let mut gaze_ratio = [1.0; 2];
    let mut blink_depth = [0.0; 2];
    let mut blink_events = [0usize; 2];
    for eye in 0..2 {
        let open = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Neutral
                || (observation.kind.family() == SampleFamily::SlowClose
                    && observation
                        .expected_open
                        .is_some_and(|target| target >= 0.80))
        });
        let closed = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Closed
                || (observation.kind.family() == SampleFamily::SlowClose
                    && observation
                        .expected_open
                        .is_some_and(|target| target <= 0.20))
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
        let span = (open_mean - closed_mean).max(0.001);
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

        let neutral = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::Neutral
        });
        let gaze = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::GazeSweep
        });
        neutral_ratio[eye] = mean_variance(&neutral).1.sqrt() / span;
        gaze_ratio[eye] = mean_variance(&gaze).1.sqrt() / span;

        let blink = values(observations, eye, |observation| {
            observation.kind.family() == SampleFamily::NaturalBlinks
        });
        let blink_low = percentile(&blink, 0.10).unwrap_or(open_mean);
        blink_depth[eye] = ((open_mean - blink_low) / span).clamp(0.0, 1.5) / 1.5;
        let threshold = (open_mean + closed_mean) * 0.5;
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
    let score = (0.32 * separation_score
        + 0.25 * monotonicity_score
        + 0.15 * blink_response
        + 0.15 * stability
        + 0.08 * presence_rate
        + 0.05 * image_information)
        * finite_rate;
    GeometryMetrics {
        evidence_valid: true,
        score,
        separation,
        monotonicity,
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
        && (0..2).all(|eye| {
            candidate.separation[eye] + 0.15 >= baseline.separation[eye] * 0.85
                && candidate.monotonicity[eye] + 0.10 >= baseline.monotonicity[eye]
        })
}

fn capture_quality_issue(baseline: &GeometryMetrics) -> Option<&'static str> {
    if !baseline.evidence_valid {
        Some("one or more required open/closed/slow-close evidence classes are missing")
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
        .filter(|(_, sample)| sample.kind.is_holdout() == holdout)
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
    let train = dataset.train_len();
    let holdout = dataset.holdout_len();
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
            .filter(|sample| !sample.kind.is_holdout() && sample.kind.family() == family)
            .count();
        let holdout_count = dataset
            .samples
            .iter()
            .filter(|sample| sample.kind.is_holdout() && sample.kind.family() == family)
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
            presence: 0.10,
            open,
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
    fn audit_matrix_keeps_the_active_geometry_as_its_reference() {
        let baseline = default_ml_geometry("pimax_xr5");
        let cases = audit_case_specs(baseline);
        assert_eq!(cases[0].name, "active reference");
        assert_eq!(cases[0].geometry, baseline);
        // The preset already touches the outer frame edge, so both negative-inward
        // probes clamp back to the reference and are intentionally deduplicated.
        assert_eq!(cases.len(), 26);
        assert_eq!(cases.iter().filter(|case| case.axis.is_some()).count(), 18);
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
