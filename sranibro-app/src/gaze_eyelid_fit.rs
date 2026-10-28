//! Holdout-validated gaze-direction compensation for raw eyelid openness.
//!
//! Categorical targets are labels only. EyeNet is replayed to measure openness,
//! while the HMD's recorded native gaze supplies the continuous runtime coordinate.
//! The fitted model can add a bounded lift to a repeatable relaxed-open droop; it
//! cannot lower openness, alter gaze output, or change image preprocessing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::calib_session::{CommitPermit, GazeEyelidChange};
use crate::core::eye_state::{CalibSnapshot, CalibStore};
use crate::core::types::{
    DespeckleParams, FlattenParams, GazeEyelidEyeProfile, GazeEyelidProfile, MlGeometry,
    PhotometricCorrection,
};
use crate::geometry_calib::{
    EvidenceAction, EvidenceLabel, EvidenceSplit, GazeTarget, SampleFamily, SharedEvidence,
};
use crate::ml::{brightness, eye_net::EyeNet, preprocess, tvm_params};

const PRESENCE_MIN: f32 = 0.05;
const MIN_BLOCK_FRAMES: usize = 12;
const MIN_GAZE_COVERAGE: f32 = 0.70;
const MAX_RAW_REPEAT_DELTA: f32 = 0.050;
const MAX_GAZE_REPEAT_DEG: f32 = 6.0;
const MAX_LIFT: f32 = 0.12;
const MIN_ABS_IMPROVEMENT: f32 = 0.010;
const MIN_REL_IMPROVEMENT: f32 = 0.25;

#[derive(Debug)]
pub struct FitInputs {
    pub model_bytes: Arc<[u8]>,
    pub expected_model_crc32: u32,
    pub expected_model_bytes: u64,
    pub dataset: SharedEvidence,
    pub geometry: [MlGeometry; 2],
    pub mirrors: [bool; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    pub photometric: PhotometricCorrection,
    pub current_endpoints: CalibStore,
}

#[derive(Debug)]
pub struct StartError {
    pub message: String,
    pub inputs: FitInputs,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub target: Option<GazeTarget>,
    pub phase_index: usize,
    pub family: SampleFamily,
    pub holdout: bool,
    pub raw: [Option<f32>; 2],
    pub presence: [f32; 2],
    pub gaze_deg: [Option<[f32; 2]>; 2],
    pub native_open: [Option<f32>; 2],
}

#[derive(Clone, Debug)]
pub struct EyeFit {
    pub accepted: bool,
    pub proposed: GazeEyelidEyeProfile,
    pub rejections: Vec<String>,
    pub notes: Vec<String>,
    pub max_train_droop: f32,
    pub holdout_before: f32,
    pub holdout_after: f32,
    pub safety_frames: usize,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub profile: GazeEyelidProfile,
    pub eyes: [EyeFit; 2],
}

impl FitResult {
    pub fn accepted_eyes(&self) -> [bool; 2] {
        [self.eyes[0].accepted, self.eyes[1].accepted]
    }

    pub fn any_accepted(&self) -> bool {
        self.eyes.iter().any(|eye| eye.accepted)
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Running { completed: usize, total: usize },
    Done { result: FitResult },
    Failed { message: String },
    Cancelled,
}

impl Status {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ApplyRequest {
    pub profile: GazeEyelidProfile,
}

pub fn commit_request(
    result: &FitResult,
    apply_eyes: [bool; 2],
    current: GazeEyelidProfile,
    _permit: CommitPermit<GazeEyelidChange>,
) -> Option<ApplyRequest> {
    let mut profile = current;
    profile.schema_version = 2;
    profile.calibrated_unix = result.profile.calibrated_unix;
    profile.angle_scale_deg = result.profile.angle_scale_deg;
    let mut changed = false;
    for (eye, (apply, fit)) in apply_eyes.iter().zip(result.eyes.iter()).enumerate() {
        if *apply && fit.accepted {
            profile.eyes[eye] = result.profile.eyes[eye];
            changed = true;
        }
    }
    changed.then_some(ApplyRequest { profile })
}

struct Shared {
    status: Status,
}

pub struct Fitter {
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Default for Fitter {
    fn default() -> Self {
        Self::new()
    }
}

impl Fitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: Status::Idle,
            })),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }

    pub fn status(&self) -> Status {
        lock(&self.shared).status.clone()
    }

    pub fn is_running(&self) -> bool {
        self.status().is_running()
    }

    pub fn clear_finished(&mut self) {
        if !self.is_running() {
            lock(&self.shared).status = Status::Idle;
            self.handle.take();
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    #[allow(clippy::result_large_err)]
    pub fn start(&mut self, inputs: FitInputs) -> Result<(), StartError> {
        if self.is_running() {
            return Err(StartError {
                message: "a gaze-direction eyelid fit is already running".into(),
                inputs,
            });
        }
        if inputs.model_bytes.is_empty()
            || inputs.model_bytes.len() as u64 != inputs.expected_model_bytes
            || crate::diagnostics::crc32_fingerprint(&inputs.model_bytes)
                != inputs.expected_model_crc32
        {
            return Err(StartError {
                message: "EyePrediction model snapshot no longer matches the live model".into(),
                inputs,
            });
        }
        let total = gaze_fit_sample_count(&inputs.dataset);
        self.cancel.store(false, Ordering::Relaxed);
        lock(&self.shared).status = Status::Running {
            completed: 0,
            total,
        };
        let slot = Arc::new(Mutex::new(Some(inputs)));
        let worker_slot = slot.clone();
        let shared = self.shared.clone();
        let cancel = self.cancel.clone();
        match std::thread::Builder::new()
            .name("gaze-eyelid-fitter".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    // Replaying thousands of EyeNet frames is bulk calibration work.
                    // Keep live camera/inference and the compositor ahead of it.
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let Some(inputs) = worker_slot.lock().ok().and_then(|mut slot| slot.take()) else {
                    lock(&shared).status = Status::Failed {
                        message: "gaze-eyelid worker lost its immutable input snapshot".into(),
                    };
                    return;
                };
                run(shared, cancel, inputs);
            }) {
            Ok(handle) => {
                self.handle = Some(handle);
                Ok(())
            }
            Err(error) => {
                let inputs = slot
                    .lock()
                    .ok()
                    .and_then(|mut slot| slot.take())
                    .expect("failed spawn must leave fit inputs available");
                lock(&self.shared).status = Status::Idle;
                Err(StartError {
                    message: format!("could not spawn gaze-eyelid worker: {error}"),
                    inputs,
                })
            }
        }
    }
}

impl Drop for Fitter {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
        self.handle.take();
    }
}

fn lock(shared: &Arc<Mutex<Shared>>) -> MutexGuard<'_, Shared> {
    shared
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The gaze surface is trained and validated only from instructed open-gaze
/// blocks. Close frames are replayed solely as untouched validation safety
/// evidence; endpoint, wink, half-open, and fit-split close frames belong to
/// their dedicated fitters and must not consume EyeNet work here.
fn is_gaze_fit_evidence(label: EvidenceLabel) -> bool {
    label.action == EvidenceAction::GazeOpen
        || (label.split == EvidenceSplit::Validation
            && matches!(
                label.action,
                EvidenceAction::SlowCloseOpen | EvidenceAction::NaturalBlink
            ))
}

fn gaze_fit_sample_count(dataset: &SharedEvidence) -> usize {
    dataset
        .samples()
        .iter()
        .filter(|sample| is_gaze_fit_evidence(sample.evidence_label()))
        .count()
}

fn run(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: FitInputs) {
    let map = match tvm_params::parse_map_bytes(&inputs.model_bytes) {
        Ok(map) => map,
        Err(error) => {
            lock(&shared).status = Status::Failed {
                message: format!("EyePrediction model parse failed: {error}"),
            };
            return;
        }
    };
    let mut net = match EyeNet::new(map) {
        Ok(net) => net,
        Err(error) => {
            lock(&shared).status = Status::Failed {
                message: format!("EyePrediction model is incompatible: {error}"),
            };
            return;
        }
    };
    let total = gaze_fit_sample_count(&inputs.dataset);
    let mut observations = Vec::with_capacity(total);
    for sample in inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_gaze_fit_evidence(sample.evidence_label()))
    {
        if cancel.load(Ordering::Relaxed) {
            lock(&shared).status = Status::Cancelled;
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
        let left = preprocess::fitted_photometric(
            &left,
            lw as usize,
            lh as usize,
            &inputs.geometry[0],
            &inputs.photometric,
            0,
        );
        let right = preprocess::fitted_photometric(
            &right,
            rw as usize,
            rh as usize,
            &inputs.geometry[1],
            &inputs.photometric,
            1,
        );
        let input = preprocess::to_input_stereo_geom(
            &left,
            lw,
            lh,
            &right,
            rw,
            rh,
            inputs.mirrors[0],
            inputs.mirrors[1],
            &inputs.geometry[0],
            &inputs.geometry[1],
        );
        let output = net.forward_one(&input);
        observations.push(Observation {
            target: sample.commanded_target,
            phase_index: sample.phase_index,
            family: sample.kind.family(),
            holdout: sample.kind.is_holdout(),
            raw: [finite(output[1]), finite(output[2])],
            presence: [output[0], output[0]],
            gaze_deg: sample
                .native_gaze
                .map(|gaze| gaze.and_then(GazeEyelidProfile::gaze_angles_deg)),
            native_open: sample.native_open,
        });
        lock(&shared).status = Status::Running {
            completed: observations.len(),
            total,
        };
    }
    let result = fit_observations(&observations, inputs.current_endpoints);
    lock(&shared).status = Status::Done { result };
}

fn finite(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

#[derive(Clone, Copy, Debug, Default)]
struct TargetSummary {
    raw: f32,
    gaze: [f32; 2],
    frames: usize,
    blocks: usize,
    raw_repeat_delta: f32,
    gaze_repeat_delta: f32,
}

pub fn fit_observations(observations: &[Observation], current: CalibStore) -> FitResult {
    let calibrated_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let endpoints = [current.left, current.right];
    let eyes = std::array::from_fn(|eye| fit_eye(observations, eye, endpoints[eye]));
    let mut profile = GazeEyelidProfile {
        calibrated_unix,
        ..GazeEyelidProfile::default()
    };
    for (profile_eye, fit) in profile.eyes.iter_mut().zip(eyes.iter()) {
        *profile_eye = fit.proposed;
        profile_eye.enabled = fit.accepted;
    }
    FitResult { profile, eyes }
}

fn fit_eye(observations: &[Observation], eye: usize, endpoint: CalibSnapshot) -> EyeFit {
    let mut rejections = Vec::new();
    let mut notes = Vec::new();
    let train = GazeTarget::ALL.map(|target| summarize(observations, eye, target, false));
    let holdout = GazeTarget::ALL.map(|target| summarize(observations, eye, target, true));

    for (split, summaries) in [("training", &train), ("holdout", &holdout)] {
        for (index, summary) in summaries.iter().enumerate() {
            let target = GazeTarget::ALL[index];
            if summary.blocks < 2 || summary.frames < 2 * MIN_BLOCK_FRAMES {
                rejections.push(format!(
                    "{split} {} had {}/{} usable blocks and {} frames",
                    target.as_str(),
                    summary.blocks,
                    2,
                    summary.frames
                ));
            }
            if summary.raw_repeat_delta > MAX_RAW_REPEAT_DELTA {
                rejections.push(format!(
                    "{split} {} raw repetitions differed by {:.3} (need <= {:.3})",
                    target.as_str(),
                    summary.raw_repeat_delta,
                    MAX_RAW_REPEAT_DELTA
                ));
            }
            if summary.gaze_repeat_delta > MAX_GAZE_REPEAT_DEG {
                rejections.push(format!(
                    "{split} {} gaze repetitions differed by {:.1}deg (need <= {:.1}deg)",
                    target.as_str(),
                    summary.gaze_repeat_delta,
                    MAX_GAZE_REPEAT_DEG
                ));
            }
        }
    }

    let open_samples = observations
        .iter()
        .filter(|observation| {
            matches!(
                observation.family,
                SampleFamily::Neutral | SampleFamily::GazeSweep
            ) && observation.target.is_some()
                && observation.raw[eye].is_some()
                && observation.presence[eye] > PRESENCE_MIN
        })
        .count();
    let gaze_samples = observations
        .iter()
        .filter(|observation| {
            matches!(
                observation.family,
                SampleFamily::Neutral | SampleFamily::GazeSweep
            ) && observation.target.is_some()
                && observation.raw[eye].is_some()
                && observation.presence[eye] > PRESENCE_MIN
                && observation.gaze_deg[eye].is_some()
        })
        .count();
    let gaze_coverage = gaze_samples as f32 / open_samples.max(1) as f32;
    if gaze_coverage < MIN_GAZE_COVERAGE {
        rejections.push(format!(
            "native per-eye gaze coverage was {:.0}% (need >= {:.0}%)",
            gaze_coverage * 100.0,
            MIN_GAZE_COVERAGE * 100.0
        ));
    }

    let center = train[target_index(GazeTarget::Center)];
    let x_values = train.map(|summary| summary.gaze[0]);
    let y_values = train.map(|summary| summary.gaze[1]);
    let span_x = finite_span(&x_values);
    let span_y = finite_span(&y_values);
    if span_x < 12.0 || span_y < 8.0 {
        rejections.push(format!(
            "native gaze covered only {:.1}deg horizontal / {:.1}deg vertical",
            span_x, span_y
        ));
    }

    let mut rows = Vec::with_capacity(9);
    let mut max_train_droop = 0.0f32;
    for summary in train {
        let x = (summary.gaze[0] - center.gaze[0]) / 20.0;
        let y = (summary.gaze[1] - center.gaze[1]) / 20.0;
        let lift = (center.raw - summary.raw).max(0.0);
        max_train_droop = max_train_droop.max(lift);
        rows.push(([x, y, x * y, x * x, y * y], lift));
    }
    if max_train_droop > MAX_LIFT {
        rejections.push(format!(
            "required lift {:.3} exceeds the safe {:.3} raw limit; image alignment or photometric fitting is required",
            max_train_droop, MAX_LIFT
        ));
    }
    let coefficients = ridge_fit(&rows, 0.02).unwrap_or([0.0; 5]);
    if !coefficients.iter().all(|value| value.is_finite()) {
        rejections.push("gaze correction surface was numerically invalid".into());
    }
    let max_lift = (max_train_droop + 0.01).clamp(0.0, MAX_LIFT);
    let mut proposed = GazeEyelidEyeProfile {
        enabled: false,
        center_deg: center.gaze,
        coefficients,
        max_lift,
        ..GazeEyelidEyeProfile::default()
    };

    let train_error = directional_error(&train, center.raw, &proposed);
    let holdout_center = holdout[target_index(GazeTarget::Center)].raw;
    let (holdout_before, holdout_after, worst_worsening) =
        holdout_errors(&holdout, holdout_center, &proposed);
    proposed.train_error = train_error;
    proposed.holdout_error_before = holdout_before;
    proposed.holdout_error_after = holdout_after;

    if holdout_before < 0.015 {
        rejections.push(format!(
            "holdout found no material gaze-direction eyelid droop ({holdout_before:.3})"
        ));
    } else {
        let improvement = holdout_before - holdout_after;
        let relative = improvement / holdout_before.max(1e-6);
        if improvement < MIN_ABS_IMPROVEMENT || relative < MIN_REL_IMPROVEMENT {
            rejections.push(format!(
                "holdout improvement was {improvement:+.3} ({:.0}%); need +{:.3} and {:.0}%",
                relative * 100.0,
                MIN_ABS_IMPROVEMENT,
                MIN_REL_IMPROVEMENT * 100.0
            ));
        }
        if worst_worsening > 0.010 {
            rejections.push(format!(
                "one holdout direction worsened by {worst_worsening:.3} (need <= 0.010)"
            ));
        }
    }

    let closed_ref = endpoint.baseline - endpoint.blink_depth;
    let mut safety_frames = 0usize;
    let mut max_closed_lift = 0.0f32;
    for observation in observations.iter().filter(|observation| {
        observation.holdout
            && matches!(
                observation.family,
                SampleFamily::SlowClose | SampleFamily::NaturalBlinks
            )
    }) {
        let (Some(raw), Some(gaze)) = (observation.raw[eye], observation.gaze_deg[eye]) else {
            continue;
        };
        if observation.presence[eye] <= PRESENCE_MIN || raw > closed_ref + 0.04 {
            continue;
        }
        safety_frames += 1;
        let predicted = predict_at(&proposed, gaze);
        let corrected = safety_corrected_raw(
            raw,
            predicted,
            endpoint.baseline,
            closed_ref,
            observation.native_open[eye],
        );
        max_closed_lift = max_closed_lift.max(corrected - raw);
        if corrected > closed_ref + 0.025 {
            rejections.push(format!(
                "closed-eye safety failed: a holdout close would rise to {corrected:.3}"
            ));
            break;
        }
    }
    if safety_frames < 3 {
        rejections.push(format!(
            "only {safety_frames}/3 untouched close-safety frames reached the calibrated floor"
        ));
    } else {
        notes.push(format!(
            "{safety_frames} untouched close frames stayed protected (max lift {max_closed_lift:.3})"
        ));
    }

    let accepted = rejections.is_empty();
    proposed.enabled = accepted;
    EyeFit {
        accepted,
        proposed,
        rejections,
        notes,
        max_train_droop,
        holdout_before,
        holdout_after,
        safety_frames,
    }
}

fn summarize(
    observations: &[Observation],
    eye: usize,
    target: GazeTarget,
    holdout: bool,
) -> TargetSummary {
    let mut phase_ids = observations
        .iter()
        .filter(|observation| {
            observation.holdout == holdout
                && observation.target == Some(target)
                && matches!(
                    observation.family,
                    SampleFamily::Neutral | SampleFamily::GazeSweep
                )
        })
        .map(|observation| observation.phase_index)
        .collect::<Vec<_>>();
    phase_ids.sort_unstable();
    phase_ids.dedup();
    let mut block_raw = Vec::new();
    let mut block_gaze = Vec::new();
    let mut frames = 0usize;
    for phase in phase_ids {
        let points = observations
            .iter()
            .filter_map(|observation| {
                (observation.phase_index == phase
                    && observation.holdout == holdout
                    && observation.target == Some(target)
                    && matches!(
                        observation.family,
                        SampleFamily::Neutral | SampleFamily::GazeSweep
                    )
                    && observation.presence[eye] > PRESENCE_MIN)
                    .then_some((observation.raw[eye]?, observation.gaze_deg[eye]?))
            })
            .collect::<Vec<_>>();
        if points.len() < MIN_BLOCK_FRAMES {
            continue;
        }
        frames += points.len();
        block_raw.push(percentile(
            &points.iter().map(|point| point.0).collect::<Vec<_>>(),
            0.5,
        ));
        block_gaze.push([
            percentile(
                &points.iter().map(|point| point.1[0]).collect::<Vec<_>>(),
                0.5,
            ),
            percentile(
                &points.iter().map(|point| point.1[1]).collect::<Vec<_>>(),
                0.5,
            ),
        ]);
    }
    let raw = percentile(&block_raw, 0.5);
    let gaze = [
        percentile(
            &block_gaze.iter().map(|point| point[0]).collect::<Vec<_>>(),
            0.5,
        ),
        percentile(
            &block_gaze.iter().map(|point| point[1]).collect::<Vec<_>>(),
            0.5,
        ),
    ];
    TargetSummary {
        raw,
        gaze,
        frames,
        blocks: block_raw.len(),
        raw_repeat_delta: span(&block_raw),
        gaze_repeat_delta: block_gaze
            .iter()
            .flat_map(|a| {
                block_gaze
                    .iter()
                    .map(move |b| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt())
            })
            .fold(0.0, f32::max),
    }
}

fn directional_error(
    summaries: &[TargetSummary; 9],
    center_raw: f32,
    profile: &GazeEyelidEyeProfile,
) -> f32 {
    let mut errors = Vec::new();
    for (index, summary) in summaries.iter().enumerate() {
        if GazeTarget::ALL[index] == GazeTarget::Center {
            continue;
        }
        let corrected = (summary.raw + predict_at(profile, summary.gaze)).min(center_raw);
        errors.push((center_raw - corrected).max(0.0));
    }
    mean(&errors)
}

fn holdout_errors(
    summaries: &[TargetSummary; 9],
    center_raw: f32,
    profile: &GazeEyelidEyeProfile,
) -> (f32, f32, f32) {
    let mut before = Vec::new();
    let mut after = Vec::new();
    let mut worst_worsening = 0.0f32;
    for (index, summary) in summaries.iter().enumerate() {
        if GazeTarget::ALL[index] == GazeTarget::Center {
            continue;
        }
        let b = (center_raw - summary.raw).max(0.0);
        let corrected = (summary.raw + predict_at(profile, summary.gaze)).min(center_raw);
        let a = (center_raw - corrected).max(0.0);
        worst_worsening = worst_worsening.max(a - b);
        before.push(b);
        after.push(a);
    }
    (mean(&before), mean(&after), worst_worsening)
}

fn predict_at(profile: &GazeEyelidEyeProfile, gaze_deg: [f32; 2]) -> f32 {
    let x = (gaze_deg[0] - profile.center_deg[0]) / 20.0;
    let y = (gaze_deg[1] - profile.center_deg[1]) / 20.0;
    let basis = [x, y, x * y, x * x, y * y];
    profile
        .coefficients
        .iter()
        .zip(basis)
        .map(|(coefficient, value)| coefficient * value)
        .sum::<f32>()
        .clamp(0.0, profile.max_lift)
}

fn safety_corrected_raw(
    raw: f32,
    lift: f32,
    baseline: f32,
    closed_ref: f32,
    native_open: Option<f32>,
) -> f32 {
    if native_open.is_some_and(|value| value <= 0.0) {
        return raw;
    }
    let span = (baseline - closed_ref).max(0.05);
    let normalized = (raw - closed_ref) / span;
    let x = ((normalized - 0.08) / 0.32).clamp(0.0, 1.0);
    let gate = x * x * (3.0 - 2.0 * x);
    if raw < baseline {
        raw + (lift * gate).min(baseline - raw)
    } else {
        raw
    }
}

fn ridge_fit(rows: &[([f32; 5], f32)], lambda: f32) -> Option<[f32; 5]> {
    let mut a = [[0.0f32; 6]; 5];
    for (x, y) in rows {
        for row in 0..5 {
            for column in 0..5 {
                a[row][column] += x[row] * x[column];
            }
            a[row][5] += x[row] * y;
        }
    }
    for (index, row) in a.iter_mut().enumerate() {
        row[index] += lambda;
    }
    for pivot in 0..5 {
        let best = (pivot..5)
            .max_by(|left, right| a[*left][pivot].abs().total_cmp(&a[*right][pivot].abs()))?;
        a.swap(pivot, best);
        let divisor = a[pivot][pivot];
        if !divisor.is_finite() || divisor.abs() < 1e-8 {
            return None;
        }
        for column in pivot..6 {
            a[pivot][column] /= divisor;
        }
        for row in 0..5 {
            if row == pivot {
                continue;
            }
            let factor = a[row][pivot];
            for column in pivot..6 {
                a[row][column] -= factor * a[pivot][column];
            }
        }
    }
    Some(std::array::from_fn(|index| a[index][5]))
}

fn target_index(target: GazeTarget) -> usize {
    GazeTarget::ALL
        .iter()
        .position(|candidate| *candidate == target)
        .expect("target belongs to ALL")
}

fn percentile(values: &[f32], q: f32) -> f32 {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let index = ((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    values[index]
}

fn span(values: &[f32]) -> f32 {
    finite_span(values)
}

fn finite_span(values: &[f32]) -> f32 {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        min = min.min(value);
        max = max.max(value);
    }
    if min.is_finite() && max.is_finite() {
        max - min
    } else {
        f32::INFINITY
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint() -> CalibSnapshot {
        CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        }
    }

    fn synthetic(droop_right: [f32; 2], shallow_close: bool) -> Vec<Observation> {
        let mut observations = Vec::new();
        let offsets: [[f32; 2]; 9] = [
            [0.0, 0.0],
            [-18.0, 0.0],
            [18.0, 0.0],
            [0.0, -12.0],
            [0.0, 12.0],
            [-18.0, -12.0],
            [18.0, -12.0],
            [-18.0, 12.0],
            [18.0, 12.0],
        ];
        for holdout in [false, true] {
            for repetition in 0..2 {
                for (target_index, target) in GazeTarget::ALL.iter().copied().enumerate() {
                    for frame in 0..16 {
                        let side = offsets[target_index][0].abs() / 18.0;
                        let raw = [
                            0.60 - 0.07 * side,
                            0.60 - droop_right[holdout as usize] * side,
                        ];
                        observations.push(Observation {
                            target: Some(target),
                            phase_index: holdout as usize * 100 + repetition * 10 + target_index,
                            family: if target == GazeTarget::Center {
                                SampleFamily::Neutral
                            } else {
                                SampleFamily::GazeSweep
                            },
                            holdout,
                            raw: raw.map(|value| Some(value + frame as f32 * 0.00005)),
                            presence: [1.0; 2],
                            gaze_deg: [Some(offsets[target_index]), Some(offsets[target_index])],
                            native_open: [None; 2],
                        });
                    }
                }
            }
        }
        for _frame in 0..8 {
            let raw = if shallow_close { 0.385 } else { 0.35 };
            observations.push(Observation {
                target: Some(GazeTarget::Center),
                phase_index: 999,
                family: SampleFamily::SlowClose,
                holdout: true,
                raw: [Some(raw); 2],
                presence: [1.0; 2],
                gaze_deg: [Some([18.0, 0.0]); 2],
                native_open: [Some(0.0); 2],
            });
        }
        observations
    }

    fn sample(
        kind: crate::geometry_calib::SampleKind,
        target: Option<GazeTarget>,
        phase_index: usize,
    ) -> crate::geometry_calib::GeometrySample {
        crate::geometry_calib::GeometrySample {
            kind,
            commanded_target: target,
            expected_open: None,
            phase_time_s: 0.0,
            left: Vec::new(),
            right: Vec::new(),
            left_size: (0, 0),
            right_size: (0, 0),
            brightness_affine: [[1.0, 0.0]; 2],
            native_open: [None; 2],
            native_gaze: [None; 2],
            native_pupil_pos: [None; 2],
            frame_generation: [0; 2],
            native_timestamp_us: None,
            phase_index,
        }
    }

    #[test]
    fn semantic_replay_filter_preserves_fit_parity_and_reports_exact_count() {
        use crate::geometry_calib::{GeometryDataset, SampleKind};

        let dataset = SharedEvidence::new(GeometryDataset {
            samples: vec![
                sample(SampleKind::Neutral, None, 0),
                sample(SampleKind::Neutral, Some(GazeTarget::Center), 1),
                sample(SampleKind::GazeSweep, Some(GazeTarget::Left), 2),
                sample(SampleKind::HoldoutGazeSweep, Some(GazeTarget::Right), 3),
                sample(SampleKind::SlowClose, Some(GazeTarget::Center), 4),
                sample(SampleKind::HoldoutSlowClose, Some(GazeTarget::Center), 5),
                sample(SampleKind::NaturalBlinks, Some(GazeTarget::Center), 6),
                sample(
                    SampleKind::HoldoutNaturalBlinks,
                    Some(GazeTarget::Center),
                    7,
                ),
                sample(SampleKind::HalfOpen, Some(GazeTarget::Center), 8),
                sample(SampleKind::HoldoutClosed, Some(GazeTarget::Center), 9),
                sample(SampleKind::LeftWink, Some(GazeTarget::Center), 10),
                sample(SampleKind::HoldoutRightWink, Some(GazeTarget::Center), 11),
            ],
        });
        assert_eq!(gaze_fit_sample_count(&dataset), 5);

        let relevant = synthetic([0.08, 0.08], false);
        let expected = fit_observations(
            &relevant,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
        );
        let mut contaminated = relevant;
        for family in [
            SampleFamily::HalfOpen,
            SampleFamily::Closed,
            SampleFamily::SlowClose,
            SampleFamily::NaturalBlinks,
        ] {
            contaminated.push(Observation {
                target: Some(GazeTarget::Left),
                // Deliberately collide with a real gaze block. The fitter's
                // semantic guards must make this observation invisible.
                phase_index: 1,
                family,
                holdout: false,
                raw: [Some(-10.0), Some(10.0)],
                presence: [1.0; 2],
                gaze_deg: [Some([90.0, 90.0]); 2],
                native_open: [Some(0.0); 2],
            });
        }
        let actual = fit_observations(
            &contaminated,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
        );
        assert_eq!(actual.accepted_eyes(), expected.accepted_eyes());
        for eye in 0..2 {
            assert_eq!(actual.eyes[eye].proposed, expected.eyes[eye].proposed);
            assert_eq!(actual.eyes[eye].rejections, expected.eyes[eye].rejections);
            assert_eq!(actual.eyes[eye].notes, expected.eyes[eye].notes);
            assert_eq!(
                actual.eyes[eye].safety_frames,
                expected.eyes[eye].safety_frames
            );
        }
    }

    #[test]
    fn repeatable_side_droop_passes_independently() {
        let observations = synthetic([0.08, 0.08], false);
        let result = fit_observations(
            &observations,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
        );
        assert!(result.eyes[0].accepted, "{:?}", result.eyes[0].rejections);
        assert!(result.eyes[1].accepted, "{:?}", result.eyes[1].rejections);
        assert!(result.eyes[0].holdout_after < result.eyes[0].holdout_before);
    }

    #[test]
    fn holdout_disagreement_rejects_only_bad_eye() {
        let observations = synthetic([0.08, 0.015], false);
        let result = fit_observations(
            &observations,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
        );
        assert!(result.eyes[0].accepted, "{:?}", result.eyes[0].rejections);
        assert!(!result.eyes[1].accepted);
    }

    #[test]
    fn closed_gate_never_lifts_the_endpoint() {
        let corrected = safety_corrected_raw(0.35, 0.12, 0.60, 0.35, None);
        assert_eq!(corrected, 0.35);
        let disabled = safety_corrected_raw(0.39, 0.12, 0.60, 0.35, Some(0.0));
        assert_eq!(disabled, 0.39);
    }

    #[test]
    fn profile_prediction_is_positive_bounded_and_center_zero() {
        let mut profile = GazeEyelidProfile::default();
        profile.eyes[0] = GazeEyelidEyeProfile {
            enabled: true,
            coefficients: [0.02, 0.0, 0.0, 0.08, 0.0],
            max_lift: 0.09,
            ..GazeEyelidEyeProfile::default()
        };
        assert_eq!(profile.predicted_lift(0, [0.0, 0.0, 1.0]), Some(0.0));
        assert_eq!(profile.predicted_lift(0, [0.0, 0.0, -1.0]), None);
        assert_eq!(profile.predicted_lift(0, [0.0; 3]), None);
        let gaze = [30f32.to_radians().sin(), 0.0, 30f32.to_radians().cos()];
        assert!(profile.predicted_lift(0, gaze).unwrap() <= 0.09);
    }

    #[test]
    fn profile_rejects_non_finite_persisted_values() {
        let mut profile = GazeEyelidProfile::default();
        profile.eyes[0].enabled = true;
        profile.eyes[0].max_lift = f32::NAN;
        assert!(!profile.is_compatible());
        assert_eq!(profile.predicted_lift(0, [0.0, 0.0, 1.0]), None);
    }
}
