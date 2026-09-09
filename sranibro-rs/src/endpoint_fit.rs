//! Explicit per-eye open/closed endpoint fitting.
//!
//! The pose instruction is the label. EyeNet is replayed only to measure the raw
//! coordinate produced for that labelled pose; its output never chooses or changes
//! a label. Train blocks propose endpoints and untouched holdout blocks decide
//! whether each eye may be applied.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::calib_session::{CommitPermit, EndpointChange};
use crate::core::eye_state::{CalibSnapshot, CalibStore};
use crate::core::types::{DespeckleParams, FlattenParams, MlGeometry, PhotometricCorrection};
use crate::geometry_calib::{
    EvidenceAction, EvidenceLabel, EvidenceSplit, SampleFamily, SharedEvidence,
};
use crate::ml::{brightness, eye_net::EyeNet, preprocess, tvm_params};

const PRESENCE_MIN: f32 = 0.05;
const TRAIN_MIN: usize = 15;
const HOLDOUT_MIN: usize = 15;
const MIN_CLOSE_SPAN: f32 = 0.05;
const MIN_FIT_BLOCKS: usize = 2;
const MIN_VALIDATION_BLOCKS: usize = 1;
// The master recorder holds fit poses for 3.6 s and validation poses for 2.6 s.
// Score the final stable second exactly as the UI promises. The former 3.0/2.0
// cutoffs left only 0.6 s (about 12--14 frames at 20 Hz) while requiring 15,
// causing otherwise perfect captures to fail on an impossible quota.
/// Score the settled tail relative to each block's actual last captured timestamp.
/// Capture producers may use different countdown/settle layouts, so an absolute
/// "seconds since phase start" threshold is not a portable protocol boundary.
const STATIC_TAIL_S: f32 = 1.25;

#[derive(Debug)]
pub struct EndpointFitInputs {
    pub model_path: PathBuf,
    pub model_bytes: Arc<[u8]>,
    pub expected_model_crc32: u32,
    pub expected_model_bytes: u64,
    pub dataset: SharedEvidence,
    pub geometry: [MlGeometry; 2],
    pub mirrors: [bool; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    pub photometric: PhotometricCorrection,
    pub current: CalibStore,
    pub open_deadzone: f32,
}

#[derive(Debug)]
pub struct StartError {
    pub message: String,
    pub inputs: EndpointFitInputs,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlockStats {
    pub frames: usize,
    pub admitted: usize,
    pub median: f32,
    pub iqr: f32,
    pub p10: f32,
    pub p80: f32,
    pub p90: f32,
}

#[derive(Clone, Debug, Default)]
pub struct EyeEndpointEvidence {
    pub open: Vec<BlockStats>,
    pub half: Vec<BlockStats>,
    pub closed: Vec<BlockStats>,
    pub holdout_open: Vec<BlockStats>,
    pub holdout_half: Vec<BlockStats>,
    pub holdout_closed: Vec<BlockStats>,
    pub native_closed_disable_rate: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndpointCandidate {
    pub baseline: f32,
    pub closed_ref: f32,
    pub blink_depth: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HoldoutPrediction {
    pub open_median: f32,
    pub open_p10: f32,
    pub half_median: f32,
    pub closed_median: f32,
    pub closed_p90: f32,
    pub endpoint_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EndpointReject {
    TooFewFrames {
        family: SampleFamily,
        block: usize,
        admitted: usize,
        needed: usize,
    },
    UnstableBlock {
        family: SampleFamily,
        block: usize,
        iqr: f32,
        max: f32,
    },
    InconsistentBlocks {
        family: SampleFamily,
        delta: f32,
        max: f32,
    },
    Inverted,
    WeakSeparation {
        separation: f32,
    },
    ExcessiveSeparation {
        separation: f32,
    },
    NonMonotonicHalf {
        closed: f32,
        half: f32,
        open: f32,
    },
    EffectiveSpanTooSmall {
        span: f32,
        open_deadzone: f32,
    },
    BaselineOutOfRange {
        baseline: f32,
    },
    HoldoutFailed {
        detail: String,
    },
    NotBetterThanCurrent {
        current_error: f32,
        candidate_error: f32,
    },
}

impl EndpointReject {
    pub fn user_message(&self) -> String {
        match self {
            Self::TooFewFrames {
                family,
                block,
                admitted,
                needed,
            } => format!(
                "{family:?} block {} had only {admitted}/{needed} usable frames",
                block + 1
            ),
            Self::UnstableBlock {
                family,
                block,
                iqr,
                max,
            } => format!(
                "{family:?} block {} moved too much (spread {iqr:.3}, need <= {max:.3})",
                block + 1
            ),
            Self::InconsistentBlocks { family, delta, max } => format!(
                "Repeated {family:?} poses disagreed by {delta:.3} (need <= {max:.3})"
            ),
            Self::Inverted => "Closed was not below relaxed-open in the raw model coordinate".into(),
            Self::WeakSeparation { separation } => {
                format!("Open/closed separation was too small ({separation:.3}, need >= 0.060)")
            }
            Self::ExcessiveSeparation { separation } => format!(
                "Open/closed separation was outside the supported range ({separation:.3})"
            ),
            Self::NonMonotonicHalf { closed, half, open } => format!(
                "Half-open was not between closed and open ({closed:.3} < {half:.3} < {open:.3})"
            ),
            Self::EffectiveSpanTooSmall {
                span,
                open_deadzone,
            } => format!(
                "Usable ramp span is only {span:.3} with Open dead-zone {open_deadzone:.3}; reduce the dead-zone or improve eye-image alignment"
            ),
            Self::BaselineOutOfRange { baseline } => {
                format!("Relaxed-open raw endpoint {baseline:.3} is outside 0.32..0.80")
            }
            Self::HoldoutFailed { detail } => format!("Untouched holdout failed: {detail}"),
            Self::NotBetterThanCurrent {
                current_error,
                candidate_error,
            } => format!(
                "Candidate endpoint error {candidate_error:.3} was worse than current {current_error:.3}"
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct EyeEndpointFit {
    pub evidence: EyeEndpointEvidence,
    pub proposed: Option<EndpointCandidate>,
    pub accepted: bool,
    pub rejections: Vec<EndpointReject>,
    pub holdout_candidate: HoldoutPrediction,
    pub holdout_current: HoldoutPrediction,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct EndpointFitResult {
    pub eyes: [EyeEndpointFit; 2],
    pub protocol: &'static str,
}

impl EndpointFitResult {
    pub fn accepted_eyes(&self) -> [bool; 2] {
        [self.eyes[0].accepted, self.eyes[1].accepted]
    }

    pub fn any_accepted(&self) -> bool {
        self.eyes.iter().any(|eye| eye.accepted)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EndpointObservation {
    pub evidence: EvidenceLabel,
    pub raw: [Option<f32>; 2],
    pub presence: [f32; 2],
    pub native_open: [Option<f32>; 2],
}

#[derive(Clone, Debug)]
pub enum EndpointFitStatus {
    Idle,
    Running { completed: usize, total: usize },
    Done { result: EndpointFitResult },
    Failed { message: String },
    Cancelled,
}

impl EndpointFitStatus {
    pub fn is_running(&self) -> bool {
        matches!(self, Self::Running { .. })
    }
}

#[derive(Debug)]
pub struct EndpointApply {
    pub baseline: f32,
    pub blink_depth: f32,
    pub calibrated_unix: u64,
}

#[derive(Debug)]
pub struct EndpointApplyRequest {
    pub eyes: [Option<EndpointApply>; 2],
    /// Per-eye removal of an existing explicit fit. Reset takes precedence over
    /// `eyes` and restarts adaptive endpoint learning from stock bounds.
    pub reset_to_adaptive: [bool; 2],
}

struct Shared {
    status: EndpointFitStatus,
}

pub struct EndpointFitter {
    shared: Arc<Mutex<Shared>>,
    cancel: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl Default for EndpointFitter {
    fn default() -> Self {
        Self::new()
    }
}

impl EndpointFitter {
    pub fn new() -> Self {
        Self {
            shared: Arc::new(Mutex::new(Shared {
                status: EndpointFitStatus::Idle,
            })),
            cancel: Arc::new(AtomicBool::new(false)),
            handle: None,
        }
    }

    pub fn status(&self) -> EndpointFitStatus {
        lock(&self.shared).status.clone()
    }

    pub fn is_running(&self) -> bool {
        self.status().is_running()
    }

    pub fn clear_finished(&mut self) {
        if !self.is_running() {
            lock(&self.shared).status = EndpointFitStatus::Idle;
            self.handle.take();
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn start(&mut self, inputs: EndpointFitInputs) -> Result<(), StartError> {
        if self.is_running() {
            return Err(StartError {
                message: "an endpoint fit is already running".into(),
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
        self.cancel.store(false, Ordering::Relaxed);
        let total = scored_endpoint_indices(inputs.dataset.samples()).len();
        lock(&self.shared).status = EndpointFitStatus::Running {
            completed: 0,
            total,
        };

        let slot = Arc::new(Mutex::new(Some(inputs)));
        let worker_slot = slot.clone();
        let shared = self.shared.clone();
        let cancel = self.cancel.clone();
        match std::thread::Builder::new()
            .name("eyelid-endpoint-fitter".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    // Model replay is background calibration work; keep the live
                    // camera, inference, output, and compositor threads responsive.
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let Some(inputs) = worker_slot.lock().ok().and_then(|mut slot| slot.take()) else {
                    lock(&shared).status = EndpointFitStatus::Failed {
                        message: "endpoint worker lost its immutable input snapshot".into(),
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
                    .expect("failed spawn must leave endpoint inputs available");
                lock(&self.shared).status = EndpointFitStatus::Idle;
                Err(StartError {
                    message: format!("could not spawn endpoint worker: {error}"),
                    inputs,
                })
            }
        }
    }
}

impl Drop for EndpointFitter {
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

fn run(shared: Arc<Mutex<Shared>>, cancel: Arc<AtomicBool>, inputs: EndpointFitInputs) {
    let map = match tvm_params::parse_map_bytes(&inputs.model_bytes) {
        Ok(map) => map,
        Err(error) => {
            lock(&shared).status = EndpointFitStatus::Failed {
                message: format!("EyePrediction model parse failed: {error}"),
            };
            return;
        }
    };
    let mut net = match EyeNet::new(map) {
        Ok(net) => net,
        Err(error) => {
            lock(&shared).status = EndpointFitStatus::Failed {
                message: format!("EyePrediction model is incompatible: {error}"),
            };
            return;
        }
    };
    let scored = scored_endpoint_indices(inputs.dataset.samples());
    let total = scored.len();
    let mut observations = Vec::with_capacity(total);
    for index in scored {
        let sample = &inputs.dataset.samples()[index];
        if cancel.load(Ordering::Relaxed) {
            lock(&shared).status = EndpointFitStatus::Cancelled;
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
        observations.push(EndpointObservation {
            evidence: sample.evidence_label(),
            raw: [finite(output[1]), finite(output[2])],
            presence: [output[0], output[0]],
            native_open: sample.native_open,
        });
        let completed = observations.len();
        lock(&shared).status = EndpointFitStatus::Running { completed, total };
    }
    if cancel.load(Ordering::Relaxed) {
        lock(&shared).status = EndpointFitStatus::Cancelled;
        return;
    }
    let result = fit_endpoints(&observations, &inputs.current, inputs.open_deadzone);
    lock(&shared).status = EndpointFitStatus::Done { result };
}

fn finite(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

fn is_endpoint_evidence(label: EvidenceLabel) -> bool {
    matches!(
        label.action,
        EvidenceAction::RelaxedOpen | EvidenceAction::HalfOpen | EvidenceAction::GentleClosed
    )
}

fn scored_endpoint_indices(samples: &[crate::geometry_calib::GeometrySample]) -> Vec<usize> {
    let mut blocks = BTreeMap::<(u8, usize), Vec<usize>>::new();
    for (index, sample) in samples.iter().enumerate() {
        let label = sample.evidence_label();
        if is_endpoint_evidence(label) && sample.phase_time_s.is_finite() {
            blocks
                .entry((
                    matches!(label.split, EvidenceSplit::Validation) as u8,
                    label.block_id,
                ))
                .or_default()
                .push(index);
        }
    }
    let mut selected = Vec::new();
    for indices in blocks.into_values() {
        let end = indices
            .iter()
            .map(|index| samples[*index].phase_time_s)
            .fold(f32::NEG_INFINITY, f32::max);
        let start = (end - STATIC_TAIL_S).max(0.0);
        selected.extend(
            indices
                .into_iter()
                .filter(|index| samples[*index].phase_time_s >= start),
        );
    }
    selected.sort_unstable();
    selected
}

pub fn fit_endpoints(
    observations: &[EndpointObservation],
    current: &CalibStore,
    open_deadzone: f32,
) -> EndpointFitResult {
    let current = [current.left, current.right];
    EndpointFitResult {
        eyes: std::array::from_fn(|eye| fit_eye(observations, current[eye], eye, open_deadzone)),
        protocol: "eyelid_endpoints_v1",
    }
}

fn fit_eye(
    observations: &[EndpointObservation],
    current: CalibSnapshot,
    eye: usize,
    open_deadzone: f32,
) -> EyeEndpointFit {
    let evidence = EyeEndpointEvidence {
        open: block_stats_by_label(
            observations,
            EvidenceAction::RelaxedOpen,
            EvidenceSplit::Fit,
            eye,
        ),
        half: block_stats_by_label(
            observations,
            EvidenceAction::HalfOpen,
            EvidenceSplit::Fit,
            eye,
        ),
        closed: block_stats_by_label(
            observations,
            EvidenceAction::GentleClosed,
            EvidenceSplit::Fit,
            eye,
        ),
        holdout_open: block_stats_by_label(
            observations,
            EvidenceAction::RelaxedOpen,
            EvidenceSplit::Validation,
            eye,
        ),
        holdout_half: block_stats_by_label(
            observations,
            EvidenceAction::HalfOpen,
            EvidenceSplit::Validation,
            eye,
        ),
        holdout_closed: block_stats_by_label(
            observations,
            EvidenceAction::GentleClosed,
            EvidenceSplit::Validation,
            eye,
        ),
        native_closed_disable_rate: native_closed_disable_rate(observations, eye),
    };
    let mut rejections = Vec::new();
    for (family, blocks, max_iqr) in [
        (SampleFamily::Neutral, evidence.open.as_slice(), 0.06),
        (SampleFamily::HalfOpen, evidence.half.as_slice(), 0.10),
        (SampleFamily::Closed, evidence.closed.as_slice(), 0.08),
    ] {
        for block in blocks.len()..MIN_FIT_BLOCKS {
            rejections.push(EndpointReject::TooFewFrames {
                family,
                block,
                admitted: 0,
                needed: TRAIN_MIN,
            });
        }
        for (block, stats) in blocks.iter().enumerate() {
            if stats.admitted < TRAIN_MIN {
                rejections.push(EndpointReject::TooFewFrames {
                    family,
                    block,
                    admitted: stats.admitted,
                    needed: TRAIN_MIN,
                });
            } else if stats.iqr > max_iqr {
                rejections.push(EndpointReject::UnstableBlock {
                    family,
                    block,
                    iqr: stats.iqr,
                    max: max_iqr,
                });
            }
        }
    }
    for (family, blocks) in [
        (SampleFamily::Neutral, evidence.holdout_open.as_slice()),
        (SampleFamily::HalfOpen, evidence.holdout_half.as_slice()),
        (SampleFamily::Closed, evidence.holdout_closed.as_slice()),
    ] {
        for block in blocks.len()..MIN_VALIDATION_BLOCKS {
            rejections.push(EndpointReject::TooFewFrames {
                family,
                block,
                admitted: 0,
                needed: HOLDOUT_MIN,
            });
        }
        for (block, stats) in blocks.iter().enumerate() {
            if stats.admitted < HOLDOUT_MIN {
                rejections.push(EndpointReject::TooFewFrames {
                    family,
                    block,
                    admitted: stats.admitted,
                    needed: HOLDOUT_MIN,
                });
            }
        }
    }
    for (family, blocks, max) in [
        (SampleFamily::Neutral, evidence.open.as_slice(), 0.04),
        (SampleFamily::Closed, evidence.closed.as_slice(), 0.05),
        (SampleFamily::HalfOpen, evidence.half.as_slice(), 0.08),
    ] {
        let medians = blocks
            .iter()
            .map(|block| block.median)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if medians.len() >= MIN_FIT_BLOCKS {
            let min = medians.iter().copied().fold(f32::INFINITY, f32::min);
            let max_value = medians.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let delta = max_value - min;
            if delta > max {
                rejections.push(EndpointReject::InconsistentBlocks { family, delta, max });
            }
        }
    }

    let open_values = values_for(
        observations,
        EvidenceAction::RelaxedOpen,
        EvidenceSplit::Fit,
        eye,
    );
    let half_values = values_for(
        observations,
        EvidenceAction::HalfOpen,
        EvidenceSplit::Fit,
        eye,
    );
    let closed_values = values_for(
        observations,
        EvidenceAction::GentleClosed,
        EvidenceSplit::Fit,
        eye,
    );
    let open = percentile(&open_values, 0.5);
    let half = percentile(&half_values, 0.5);
    let closed = percentile(&closed_values, 0.5);
    let closed_p80 = percentile(&closed_values, 0.8);
    let closed_iqr = percentile(&closed_values, 0.75) - percentile(&closed_values, 0.25);
    let margin = (0.5 * closed_iqr).clamp(0.010, 0.030);
    let closed_ref = (closed_p80 + margin).min(half - 0.02);
    let separation = open - closed;
    let candidate = EndpointCandidate {
        baseline: open,
        closed_ref,
        blink_depth: open - closed_ref,
    };
    if [open, half, closed, closed_ref]
        .iter()
        .all(|value| value.is_finite())
    {
        if open <= closed {
            rejections.push(EndpointReject::Inverted);
        }
        if separation < 0.06 {
            rejections.push(EndpointReject::WeakSeparation { separation });
        } else if separation > 0.40 {
            rejections.push(EndpointReject::ExcessiveSeparation { separation });
        }
        if !(closed + 0.02 <= half && half <= open - 0.02) {
            rejections.push(EndpointReject::NonMonotonicHalf { closed, half, open });
        }
        let effective_span = (open - open_deadzone) - closed_ref;
        if effective_span < MIN_CLOSE_SPAN {
            rejections.push(EndpointReject::EffectiveSpanTooSmall {
                span: effective_span,
                open_deadzone,
            });
        }
        if !(0.32..=0.80).contains(&open) {
            rejections.push(EndpointReject::BaselineOutOfRange { baseline: open });
        }
    }

    let holdout_open = values_for(
        observations,
        EvidenceAction::RelaxedOpen,
        EvidenceSplit::Validation,
        eye,
    );
    let holdout_half = values_for(
        observations,
        EvidenceAction::HalfOpen,
        EvidenceSplit::Validation,
        eye,
    );
    let holdout_closed = values_for(
        observations,
        EvidenceAction::GentleClosed,
        EvidenceSplit::Validation,
        eye,
    );
    let holdout_candidate = predict(
        &holdout_open,
        &holdout_half,
        &holdout_closed,
        candidate.baseline,
        candidate.closed_ref,
        open_deadzone,
        current.mid_anchor,
    );
    let current_closed = current.baseline - current.blink_depth;
    let holdout_current = predict(
        &holdout_open,
        &holdout_half,
        &holdout_closed,
        current.baseline,
        current_closed,
        open_deadzone,
        current.mid_anchor,
    );
    if holdout_candidate.closed_median > 0.05
        || holdout_candidate.closed_p90 > 0.15
        || holdout_candidate.open_median < 0.90
        || holdout_candidate.open_p10 < 0.70
        || !(0.15..=0.85).contains(&holdout_candidate.half_median)
    {
        rejections.push(EndpointReject::HoldoutFailed {
            detail: format!(
                "open median/p10 {:.2}/{:.2}, half {:.2}, closed median/p90 {:.2}/{:.2}",
                holdout_candidate.open_median,
                holdout_candidate.open_p10,
                holdout_candidate.half_median,
                holdout_candidate.closed_median,
                holdout_candidate.closed_p90
            ),
        });
    }
    if holdout_candidate.endpoint_error > holdout_current.endpoint_error + 0.01 {
        rejections.push(EndpointReject::NotBetterThanCurrent {
            current_error: holdout_current.endpoint_error,
            candidate_error: holdout_candidate.endpoint_error,
        });
    }

    let mut notes = Vec::new();
    if let Some(rate) = evidence.native_closed_disable_rate {
        if rate < 0.20 {
            notes.push(format!(
                "Native Tobii reported closed/Disable on only {:.0}% of held-close frames",
                rate * 100.0
            ));
        }
    }
    EyeEndpointFit {
        evidence,
        proposed: candidate.baseline.is_finite().then_some(candidate),
        accepted: rejections.is_empty(),
        rejections,
        holdout_candidate,
        holdout_current,
        notes,
    }
}

fn block_stats_by_label(
    observations: &[EndpointObservation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Vec<BlockStats> {
    let mut block_ids = observations
        .iter()
        .filter(|observation| label_matches(observation.evidence, action, split))
        .map(|observation| observation.evidence.block_id)
        .collect::<Vec<_>>();
    block_ids.sort_unstable();
    block_ids.dedup();
    block_ids
        .into_iter()
        .map(|block_id| block_stats(observations, action, split, block_id, eye))
        .collect()
}

fn block_stats(
    observations: &[EndpointObservation],
    action: EvidenceAction,
    split: EvidenceSplit,
    block_id: usize,
    eye: usize,
) -> BlockStats {
    let frames = observations
        .iter()
        .filter(|observation| {
            label_matches(observation.evidence, action, split)
                && observation.evidence.block_id == block_id
        })
        .count();
    let values = admitted_values(observations, action, split, Some(block_id), eye);
    BlockStats {
        frames,
        admitted: values.len(),
        median: percentile(&values, 0.5),
        iqr: percentile(&values, 0.75) - percentile(&values, 0.25),
        p10: percentile(&values, 0.10),
        p80: percentile(&values, 0.80),
        p90: percentile(&values, 0.90),
    }
}

fn values_for(
    observations: &[EndpointObservation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Vec<f32> {
    admitted_values(observations, action, split, None, eye)
}

fn admitted_values(
    observations: &[EndpointObservation],
    action: EvidenceAction,
    split: EvidenceSplit,
    block_id: Option<usize>,
    eye: usize,
) -> Vec<f32> {
    observations
        .iter()
        .filter(|observation| label_matches(observation.evidence, action, split))
        .filter(|observation| {
            block_id.is_none_or(|block_id| observation.evidence.block_id == block_id)
        })
        .filter(|observation| {
            observation.presence[eye].is_finite() && observation.presence[eye] > PRESENCE_MIN
        })
        .filter_map(|observation| observation.raw[eye].filter(|value| value.is_finite()))
        .collect()
}

fn label_matches(label: EvidenceLabel, action: EvidenceAction, split: EvidenceSplit) -> bool {
    label.action == action && label.split == split
}

fn native_closed_disable_rate(observations: &[EndpointObservation], eye: usize) -> Option<f32> {
    let values = observations
        .iter()
        .filter(|observation| observation.evidence.action == EvidenceAction::GentleClosed)
        .filter_map(|observation| observation.native_open[eye])
        .collect::<Vec<_>>();
    (!values.is_empty()).then(|| {
        values.iter().filter(|value| **value <= 0.001).count() as f32 / values.len() as f32
    })
}

fn predict(
    open: &[f32],
    half: &[f32],
    closed: &[f32],
    baseline: f32,
    closed_ref: f32,
    open_deadzone: f32,
    mid_anchor: f32,
) -> HoldoutPrediction {
    let map = |raw: f32| {
        let open_full = baseline - open_deadzone;
        let denom = (open_full - closed_ref).max(MIN_CLOSE_SPAN);
        let x = ((raw - closed_ref) / denom).clamp(0.0, 1.0);
        apply_anchor(x, mid_anchor)
    };
    let open = open.iter().copied().map(map).collect::<Vec<_>>();
    let half = half.iter().copied().map(map).collect::<Vec<_>>();
    let closed = closed.iter().copied().map(map).collect::<Vec<_>>();
    let open_median = percentile(&open, 0.5);
    let closed_median = percentile(&closed, 0.5);
    HoldoutPrediction {
        open_median,
        open_p10: percentile(&open, 0.10),
        half_median: percentile(&half, 0.5),
        closed_median,
        closed_p90: percentile(&closed, 0.90),
        endpoint_error: (1.0 - open_median).abs() + closed_median.abs(),
    }
}

fn apply_anchor(x: f32, anchor: f32) -> f32 {
    let anchor = if anchor.is_finite() {
        anchor.clamp(0.30, 0.70)
    } else {
        0.5
    };
    if x <= anchor {
        0.5 * x / anchor.max(1e-3)
    } else {
        0.5 + 0.5 * (x - anchor) / (1.0 - anchor).max(1e-3)
    }
}

fn percentile(values: &[f32], quantile: f32) -> f32 {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return f32::NAN;
    }
    values.sort_by(f32::total_cmp);
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let t = position - low as f32;
    values[low] * (1.0 - t) + values[high] * t
}

pub fn commit_request(
    result: &EndpointFitResult,
    eyes: [bool; 2],
    permit: CommitPermit<EndpointChange>,
) -> Option<EndpointApplyRequest> {
    let _session_id = permit.session_id();
    let calibrated_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let eyes = std::array::from_fn(|eye| {
        (eyes[eye] && result.eyes[eye].accepted)
            .then_some(result.eyes[eye].proposed)
            .flatten()
            .map(|candidate| EndpointApply {
                baseline: candidate.baseline,
                blink_depth: candidate.blink_depth,
                calibrated_unix,
            })
    });
    eyes.iter()
        .any(Option::is_some)
        .then_some(EndpointApplyRequest {
            eyes,
            reset_to_adaptive: [false; 2],
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry_calib::SampleKind;

    const OPEN_FIT_BLOCKS: [usize; 2] = [901, 17];
    const HALF_FIT_BLOCKS: [usize; 2] = [42, 990];
    const CLOSED_FIT_BLOCKS: [usize; 2] = [808, 6];
    const OPEN_VALIDATION_BLOCK: usize = 7001;
    const HALF_VALIDATION_BLOCK: usize = 73;
    const CLOSED_VALIDATION_BLOCK: usize = 602;

    fn geometry_sample(
        kind: SampleKind,
        phase_index: usize,
        phase_time_s: f32,
    ) -> crate::geometry_calib::GeometrySample {
        crate::geometry_calib::GeometrySample {
            kind,
            commanded_target: None,
            expected_open: None,
            phase_time_s,
            left: vec![0],
            right: vec![0],
            left_size: (1, 1),
            right_size: (1, 1),
            brightness_affine: [[1.0, 0.0]; 2],
            native_open: [None; 2],
            native_gaze: [None; 2],
            native_pupil_pos: [None; 2],
            frame_generation: [phase_index as u64 + 1; 2],
            native_timestamp_us: None,
            phase_index,
        }
    }

    #[test]
    fn static_pose_scoring_is_relative_to_each_block_end() {
        let samples = vec![
            geometry_sample(SampleKind::Neutral, 1, 0.0),
            geometry_sample(SampleKind::Neutral, 1, 2.70),
            geometry_sample(SampleKind::Neutral, 1, 2.80),
            geometry_sample(SampleKind::Neutral, 1, 4.00),
            geometry_sample(SampleKind::HoldoutClosed, 2, 0.0),
            geometry_sample(SampleKind::HoldoutClosed, 2, 0.90),
            geometry_sample(SampleKind::HoldoutClosed, 2, 1.00),
            geometry_sample(SampleKind::HoldoutClosed, 2, 2.20),
            geometry_sample(SampleKind::GazeSweep, 3, 9.0),
            geometry_sample(SampleKind::Closed, 4, f32::NAN),
        ];
        // Fit tail starts at 2.75; validation tail starts at 0.95. The two
        // protocols can therefore have different absolute phase durations.
        assert_eq!(scored_endpoint_indices(&samples), vec![2, 3, 6, 7]);
    }

    fn snapshot(baseline: f32, depth: f32) -> CalibSnapshot {
        CalibSnapshot {
            baseline,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: depth,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        }
    }

    fn push_block(
        observations: &mut Vec<EndpointObservation>,
        action: EvidenceAction,
        split: EvidenceSplit,
        block_id: usize,
        raw: [f32; 2],
        count: usize,
    ) {
        for n in 0..count {
            let noise = ((n % 7) as f32 - 3.0) * 0.0005;
            observations.push(EndpointObservation {
                evidence: EvidenceLabel {
                    action,
                    split,
                    target: None,
                    block_id,
                },
                raw: [Some(raw[0] + noise), Some(raw[1] - noise)],
                presence: [1.0; 2],
                native_open: [None; 2],
            });
        }
    }

    fn healthy_observations() -> Vec<EndpointObservation> {
        let mut observations = Vec::new();
        for block_id in OPEN_FIT_BLOCKS {
            push_block(
                &mut observations,
                EvidenceAction::RelaxedOpen,
                EvidenceSplit::Fit,
                block_id,
                [0.60, 0.59],
                60,
            );
        }
        for block_id in HALF_FIT_BLOCKS {
            push_block(
                &mut observations,
                EvidenceAction::HalfOpen,
                EvidenceSplit::Fit,
                block_id,
                [0.46, 0.45],
                60,
            );
        }
        for block_id in CLOSED_FIT_BLOCKS {
            push_block(
                &mut observations,
                EvidenceAction::GentleClosed,
                EvidenceSplit::Fit,
                block_id,
                [0.30, 0.31],
                60,
            );
        }
        for (action, block_id, raw) in [
            (
                EvidenceAction::RelaxedOpen,
                OPEN_VALIDATION_BLOCK,
                [0.60, 0.59],
            ),
            (
                EvidenceAction::HalfOpen,
                HALF_VALIDATION_BLOCK,
                [0.46, 0.45],
            ),
            (
                EvidenceAction::GentleClosed,
                CLOSED_VALIDATION_BLOCK,
                [0.29, 0.30],
            ),
        ] {
            push_block(
                &mut observations,
                action,
                EvidenceSplit::Validation,
                block_id,
                raw,
                45,
            );
        }
        observations
    }

    #[test]
    fn healthy_per_eye_endpoints_pass_untouched_holdout() {
        let current = CalibStore {
            left: snapshot(0.60, 0.40),
            right: snapshot(0.59, 0.39),
        };
        let result = fit_endpoints(&healthy_observations(), &current, 0.08);
        assert_eq!(result.accepted_eyes(), [true, true]);
        for eye in &result.eyes {
            let candidate = eye.proposed.unwrap();
            assert!((0.25..0.34).contains(&candidate.closed_ref));
            assert!(eye.holdout_candidate.closed_median <= 0.05);
            assert!(eye.holdout_candidate.open_median >= 0.90);
        }
    }

    #[test]
    fn shallow_right_eye_rejects_without_poisoning_left() {
        let mut observations = healthy_observations();
        for observation in &mut observations {
            if observation.evidence.action == EvidenceAction::GentleClosed {
                observation.raw[1] = Some(0.50);
            }
            if observation.evidence.action == EvidenceAction::HalfOpen {
                observation.raw[1] = Some(0.54);
            }
        }
        let current = CalibStore {
            left: snapshot(0.60, 0.40),
            right: snapshot(0.59, 0.20),
        };
        let result = fit_endpoints(&observations, &current, 0.08);
        assert!(result.eyes[0].accepted);
        assert!(!result.eyes[1].accepted);
        assert!(result.eyes[1]
            .rejections
            .iter()
            .any(|reject| matches!(reject, EndpointReject::EffectiveSpanTooSmall { .. })));
    }

    #[test]
    fn inconsistent_repeated_pose_and_bad_holdout_are_rejected() {
        let mut observations = healthy_observations();
        for observation in &mut observations {
            if observation.evidence.action == EvidenceAction::RelaxedOpen
                && observation.evidence.split == EvidenceSplit::Fit
                && observation.evidence.block_id == OPEN_FIT_BLOCKS[1]
            {
                observation.raw = observation.raw.map(|raw| raw.map(|raw| raw - 0.07));
            }
            if observation.evidence.action == EvidenceAction::GentleClosed
                && observation.evidence.split == EvidenceSplit::Validation
            {
                observation.raw = [Some(0.47), Some(0.47)];
            }
        }
        let current = CalibStore {
            left: snapshot(0.60, 0.40),
            right: snapshot(0.59, 0.39),
        };
        let result = fit_endpoints(&observations, &current, 0.08);
        for eye in &result.eyes {
            assert!(!eye.accepted);
            assert!(eye
                .rejections
                .iter()
                .any(|reject| matches!(reject, EndpointReject::InconsistentBlocks { .. })));
            assert!(eye
                .rejections
                .iter()
                .any(|reject| matches!(reject, EndpointReject::HoldoutFailed { .. })));
        }
    }

    #[test]
    fn sparse_presence_is_not_relabelled_or_fitted() {
        let mut observations = healthy_observations();
        for observation in &mut observations {
            if observation.evidence.action == EvidenceAction::GentleClosed
                && observation.evidence.split == EvidenceSplit::Fit
                && observation.evidence.block_id == CLOSED_FIT_BLOCKS[0]
            {
                observation.presence = [0.0; 2];
            }
        }
        let current = CalibStore {
            left: snapshot(0.60, 0.40),
            right: snapshot(0.59, 0.39),
        };
        let result = fit_endpoints(&observations, &current, 0.08);
        assert!(result.eyes.iter().all(|eye| eye
            .rejections
            .iter()
            .any(|reject| matches!(reject, EndpointReject::TooFewFrames { .. }))));
    }
}
