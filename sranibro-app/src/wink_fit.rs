//! Per-eye held-wink calibration.
//!
//! A wink floor is distinct from the ordinary bilateral closed endpoint. The
//! labelled eye must close while its partner remains open, repeated twice for
//! training and once in untouched holdout. Each eye is accepted independently.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::calib_session::{CommitPermit, WinkChange};
use crate::core::eye_state::{CalibSnapshot, CalibStore};
use crate::core::types::{
    DespeckleParams, FlattenParams, MlGeometry, PhotometricCorrection, WinkEyeProfile, WinkProfile,
};
use crate::geometry_calib::{EvidenceAction, EvidenceLabel, EvidenceSplit, SharedEvidence};
use crate::ml::{brightness, eye_net::EyeNet, preprocess, tvm_params};

const PRESENCE_MIN: f32 = 0.05;
const STABLE_SUFFIX_S: f32 = 1.5;
const TRAIN_MIN: usize = 30;
const HOLDOUT_MIN: usize = 25;
const MIN_CLOSE_SPAN: f32 = 0.05;
const MIN_FIT_BLOCKS: usize = 2;
const MIN_VALIDATION_BLOCKS: usize = 1;
const MIN_SQUEEZE_TRAIN_DELTA: f32 = 0.08;
const MIN_SQUEEZE_HOLDOUT_MARGIN: f32 = 0.025;

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
    pub open_deadzone: f32,
}

#[derive(Debug)]
pub struct StartError {
    pub message: String,
    pub inputs: FitInputs,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub evidence: EvidenceLabel,
    pub phase_time_s: f32,
    pub raw: [Option<f32>; 2],
    pub squeeze_raw: [Option<f32>; 2],
    pub presence: [f32; 2],
    pub native_open: [Option<f32>; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub admitted: usize,
    pub median: f32,
    pub iqr: f32,
    pub p10: f32,
    pub p80: f32,
    pub p90: f32,
}

#[derive(Clone, Debug)]
pub struct EyeFit {
    pub outcome: EyeOutcome,
    pub accepted: bool,
    pub proposed: WinkEyeProfile,
    pub train_wink: Vec<Stats>,
    pub holdout_wink: Stats,
    pub train_open: Vec<Stats>,
    pub holdout_open: Stats,
    pub open_control_holdout: f32,
    pub open_control_holdout_p10: f32,
    pub partner_open_holdout: f32,
    pub partner_open_holdout_p10: f32,
    pub squeeze_train_delta: f32,
    pub squeeze_holdout_margin: f32,
    pub rejections: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeOutcome {
    ApplyFloor,
    ApplySignatureOnly,
    NoChangeNeeded,
    RecordAgain,
}

impl EyeOutcome {
    pub fn is_committable(self) -> bool {
        matches!(self, Self::ApplyFloor | Self::ApplySignatureOnly)
    }
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub profile: WinkProfile,
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
    pub profile: WinkProfile,
}

pub fn commit_request(
    result: &FitResult,
    apply_eyes: [bool; 2],
    current: WinkProfile,
    _permit: CommitPermit<WinkChange>,
) -> Option<ApplyRequest> {
    let mut profile = current;
    profile.schema_version = WinkProfile::SCHEMA_VERSION;
    profile.calibrated_unix = result.profile.calibrated_unix;
    let mut changed = false;
    for (eye, (apply, fit)) in apply_eyes.iter().zip(result.eyes.iter()).enumerate() {
        if *apply && fit.outcome.is_committable() {
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
                message: "a wink fit is already running".into(),
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
        let total = inputs
            .dataset
            .samples()
            .iter()
            .filter(|sample| is_wink_evidence(sample.evidence_label()))
            .count();
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
            .name("wink-fitter".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let Some(inputs) = worker_slot.lock().ok().and_then(|mut slot| slot.take()) else {
                    lock(&shared).status = Status::Failed {
                        message: "wink worker lost its immutable input snapshot".into(),
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
                    .expect("failed spawn must leave wink inputs available");
                lock(&self.shared).status = Status::Idle;
                Err(StartError {
                    message: format!("could not spawn wink worker: {error}"),
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
    let total = inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_wink_evidence(sample.evidence_label()))
        .count();
    let mut observations = Vec::with_capacity(total);
    for sample in inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_wink_evidence(sample.evidence_label()))
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
            evidence: sample.evidence_label(),
            phase_time_s: sample.phase_time_s,
            raw: [finite(output[1]), finite(output[2])],
            squeeze_raw: [finite(output[3]), finite(output[4])],
            presence: [output[0], output[0]],
            native_open: sample.native_open,
        });
        lock(&shared).status = Status::Running {
            completed: observations.len(),
            total,
        };
    }
    let result = fit_observations(
        &observations,
        inputs.current_endpoints,
        inputs.open_deadzone,
    );
    lock(&shared).status = Status::Done { result };
}

fn finite(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

fn is_wink_evidence(label: EvidenceLabel) -> bool {
    matches!(
        label.action,
        EvidenceAction::LeftWink | EvidenceAction::RightWink
    )
}

pub fn fit_observations(
    observations: &[Observation],
    current: CalibStore,
    open_deadzone: f32,
) -> FitResult {
    let calibrated_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let endpoints = [current.left, current.right];
    let eyes = std::array::from_fn(|eye| fit_eye(observations, eye, endpoints, open_deadzone));
    let mut profile = WinkProfile {
        calibrated_unix,
        schema_version: WinkProfile::SCHEMA_VERSION,
        ..WinkProfile::default()
    };
    for (profile_eye, fit) in profile.eyes.iter_mut().zip(eyes.iter()) {
        *profile_eye = fit.proposed;
    }
    FitResult { profile, eyes }
}

fn fit_eye(
    observations: &[Observation],
    eye: usize,
    endpoints: [CalibSnapshot; 2],
    open_deadzone: f32,
) -> EyeFit {
    let endpoint = endpoints[eye];
    let partner_endpoint = endpoints[1 - eye];
    let (wink_action, open_action) = if eye == 0 {
        (EvidenceAction::LeftWink, EvidenceAction::RightWink)
    } else {
        (EvidenceAction::RightWink, EvidenceAction::LeftWink)
    };
    let train_wink = stats_by_label(observations, wink_action, EvidenceSplit::Fit, eye);
    let train_open = stats_by_label(observations, open_action, EvidenceSplit::Fit, eye);
    let holdout_wink_blocks =
        stats_by_label(observations, wink_action, EvidenceSplit::Validation, eye);
    let holdout_open_blocks =
        stats_by_label(observations, open_action, EvidenceSplit::Validation, eye);
    let partner_holdout_blocks = stats_by_label(
        observations,
        wink_action,
        EvidenceSplit::Validation,
        1 - eye,
    );
    let holdout_wink = stats_for_label(observations, wink_action, EvidenceSplit::Validation, eye);
    let holdout_open = stats_for_label(observations, open_action, EvidenceSplit::Validation, eye);
    let partner_holdout = stats_for_label(
        observations,
        wink_action,
        EvidenceSplit::Validation,
        1 - eye,
    );
    let train_wink_squeeze =
        squeeze_stats_by_label(observations, wink_action, EvidenceSplit::Fit, eye);
    let train_open_squeeze =
        squeeze_stats_by_label(observations, open_action, EvidenceSplit::Fit, eye);
    let holdout_wink_squeeze =
        squeeze_stats_for_label(observations, wink_action, EvidenceSplit::Validation, eye);
    let holdout_open_squeeze =
        squeeze_stats_for_label(observations, open_action, EvidenceSplit::Validation, eye);
    let mut rejections = Vec::new();
    let mut notes = Vec::new();

    for (label, blocks) in [("wink", &train_wink), ("open control", &train_open)] {
        for index in blocks.len()..MIN_FIT_BLOCKS {
            rejections.push(format!(
                "{label} block {} had only 0/{} usable frames",
                index + 1,
                TRAIN_MIN
            ));
        }
        for (index, block) in blocks.iter().enumerate() {
            if block.admitted < TRAIN_MIN {
                rejections.push(format!(
                    "{label} block {} had only {}/{} usable frames",
                    index + 1,
                    block.admitted,
                    TRAIN_MIN
                ));
            }
        }
    }
    for (label, blocks) in [
        ("holdout wink", &holdout_wink_blocks),
        ("holdout open", &holdout_open_blocks),
        ("holdout partner-open", &partner_holdout_blocks),
    ] {
        for index in blocks.len()..MIN_VALIDATION_BLOCKS {
            rejections.push(format!(
                "{label} block {} had only 0/{} usable frames",
                index + 1,
                HOLDOUT_MIN
            ));
        }
        for (index, block) in blocks.iter().enumerate() {
            if block.admitted < HOLDOUT_MIN {
                rejections.push(format!(
                    "{label} block {} had only {}/{} usable frames",
                    index + 1,
                    block.admitted,
                    HOLDOUT_MIN
                ));
            }
        }
    }
    for (label, blocks, max_iqr, max_repeat) in [
        ("wink", train_wink.as_slice(), 0.08, 0.05),
        ("open control", train_open.as_slice(), 0.06, 0.04),
    ] {
        for (index, block) in blocks.iter().enumerate() {
            if block.iqr.is_finite() && block.iqr > max_iqr {
                rejections.push(format!(
                    "{label} block {} spread {:.3} exceeded {:.3}",
                    index + 1,
                    block.iqr,
                    max_iqr
                ));
            }
        }
        let medians = blocks
            .iter()
            .map(|block| block.median)
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if medians.len() >= MIN_FIT_BLOCKS {
            let min = medians.iter().copied().fold(f32::INFINITY, f32::min);
            let max = medians.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let repeat = max - min;
            if repeat > max_repeat {
                rejections.push(format!(
                    "{label} repetitions differed by {repeat:.3} (need <= {max_repeat:.3})"
                ));
            }
        }
    }

    let open = median(
        &train_open
            .iter()
            .map(|block| block.median)
            .collect::<Vec<_>>(),
    );
    let wink_values = observations
        .iter()
        .filter_map(|observation| {
            (label_matches(observation.evidence, wink_action, EvidenceSplit::Fit)
                && observation.phase_time_s >= STABLE_SUFFIX_S
                && observation.presence[eye] > PRESENCE_MIN)
                .then_some(observation.raw[eye]?)
        })
        .collect::<Vec<_>>();
    let wink_median = percentile(&wink_values, 0.5);
    let wink_p80 = percentile(&wink_values, 0.8);
    let wink_iqr = percentile(&wink_values, 0.75) - percentile(&wink_values, 0.25);
    let margin = (0.5 * wink_iqr).clamp(0.005, 0.025);
    let open_full = endpoint.baseline - open_deadzone.clamp(0.03, 0.20);
    let wink_ref = (wink_p80 + margin).min(open_full - MIN_CLOSE_SPAN);
    let wink_depth = endpoint.baseline - wink_ref;
    let current_ref = endpoint.baseline - endpoint.blink_depth;
    let squeeze_open = median(
        &train_open_squeeze
            .iter()
            .map(|block| block.median)
            .collect::<Vec<_>>(),
    );
    let squeeze_wink = median(
        &train_wink_squeeze
            .iter()
            .map(|block| block.median)
            .collect::<Vec<_>>(),
    );
    let squeeze_train_delta = squeeze_wink - squeeze_open;
    let squeeze_holdout_margin = holdout_wink_squeeze.p10 - holdout_open_squeeze.p90;
    let squeeze_enabled = squeeze_train_delta.is_finite()
        && squeeze_holdout_margin.is_finite()
        && squeeze_train_delta >= MIN_SQUEEZE_TRAIN_DELTA
        && squeeze_holdout_margin >= MIN_SQUEEZE_HOLDOUT_MARGIN;
    let squeeze_enter_delta = if squeeze_enabled {
        (0.45 * squeeze_train_delta).clamp(0.03, 0.30)
    } else {
        WinkEyeProfile::default().squeeze_enter_delta
    };
    let squeeze_release_delta = if squeeze_enabled {
        (0.50 * squeeze_enter_delta).clamp(0.01, squeeze_enter_delta - 0.005)
    } else {
        WinkEyeProfile::default().squeeze_release_delta
    };
    let mut proposed = WinkEyeProfile {
        enabled: false,
        floor_enabled: false,
        wink_depth,
        squeeze_enabled,
        squeeze_enter_delta,
        squeeze_release_delta,
        holdout_before: predict_stats(holdout_wink, current_ref, open_full).0,
        holdout_after: predict_stats(holdout_wink, wink_ref, open_full).0,
    };

    if ![open, wink_median, wink_ref, wink_depth]
        .iter()
        .all(|value| value.is_finite())
    {
        rejections.push("wink evidence contained no finite stable endpoint".into());
    } else {
        if open - wink_median < 0.06 {
            rejections.push(format!(
                "labelled wink separated from open by only {:.3}",
                open - wink_median
            ));
        }
    }

    let (before_median, before_p90) = predict_stats(holdout_wink, current_ref, open_full);
    let (after_median, after_p90) = predict_stats(holdout_wink, wink_ref, open_full);
    let open_control_holdout = predict_open(holdout_open.median, current_ref, open_full);
    let open_control_holdout_p10 = predict_open(holdout_open.p10, current_ref, open_full);
    let partner_open_full = partner_endpoint.baseline - open_deadzone.clamp(0.03, 0.20);
    let partner_current_ref = partner_endpoint.baseline - partner_endpoint.blink_depth;
    let partner_open_holdout = predict_open(
        partner_holdout.median,
        partner_current_ref,
        partner_open_full,
    );
    let partner_open_holdout_p10 =
        predict_open(partner_holdout.p10, partner_current_ref, partner_open_full);
    let current_good = before_median <= 0.08 && before_p90 <= 0.18;
    if !current_good {
        if !(0.05..=0.40).contains(&wink_depth) {
            rejections.push(format!(
                "candidate wink depth {wink_depth:.3} is outside 0.05..0.40"
            ));
        }
        if wink_ref <= current_ref + 0.01 {
            rejections.push(
                "candidate did not provide a genuinely shallower unilateral wink floor".into(),
            );
        }
        if after_median > 0.08 || after_p90 > 0.18 {
            rejections.push(format!(
                "candidate holdout wink remained {:.2} median / {:.2} p90 (need <=0.08/0.18)",
                after_median, after_p90
            ));
        }
        if before_median - after_median < 0.08 {
            rejections.push(format!(
                "holdout improvement was only {:.2} (need >=0.08)",
                before_median - after_median
            ));
        }
    }
    if open_control_holdout_p10 < 0.85 {
        rejections.push(format!(
            "the calibrated eye was not consistently relaxed open during the opposite wink (median {:.2}, p10 {:.2}; need p10 >=0.85)",
            open_control_holdout, open_control_holdout_p10
        ));
    }
    if partner_open_holdout_p10 < 0.85 {
        rejections.push(format!(
            "the partner eye did not consistently remain open in holdout (median {:.2}, p10 {:.2}; need p10 >=0.85)",
            partner_open_holdout, partner_open_holdout_p10
        ));
    }
    let native_disable = native_disable_rate(observations, wink_action, eye);
    if let Some(rate) = native_disable {
        notes.push(format!(
            "native closed classifier agreed on {:.0}% of held-wink frames (advisory)",
            rate * 100.0
        ));
    }
    if squeeze_enabled {
        notes.push(format!(
            "native squeeze corroboration enabled (train delta {squeeze_train_delta:.3}, holdout margin {squeeze_holdout_margin:.3})"
        ));
    } else {
        notes.push(format!(
            "native squeeze was not repeatably separated (train delta {squeeze_train_delta:.3}, holdout margin {squeeze_holdout_margin:.3}); openness-only fallback retained"
        ));
    }

    let outcome = if !rejections.is_empty() {
        EyeOutcome::RecordAgain
    } else if current_good {
        notes.push(format!(
            "ordinary closed endpoint already reaches this wink ({before_median:.2} median / {before_p90:.2} p90)"
        ));
        if squeeze_enabled {
            proposed.enabled = true;
            proposed.floor_enabled = false;
            proposed.wink_depth = endpoint.blink_depth.clamp(0.05, 0.40);
            proposed.holdout_after = before_median;
            notes.push(
                "validated squeeze signature will be applied without changing the eyelid floor"
                    .into(),
            );
            EyeOutcome::ApplySignatureOnly
        } else {
            notes.push(
                "current wink response is already good; no separate profile is needed".into(),
            );
            EyeOutcome::NoChangeNeeded
        }
    } else {
        proposed.enabled = true;
        proposed.floor_enabled = true;
        proposed.wink_depth = wink_depth;
        proposed.holdout_after = after_median;
        EyeOutcome::ApplyFloor
    };
    let accepted = outcome.is_committable();
    EyeFit {
        outcome,
        accepted,
        proposed,
        train_wink,
        holdout_wink,
        train_open,
        holdout_open,
        open_control_holdout,
        open_control_holdout_p10,
        partner_open_holdout,
        partner_open_holdout_p10,
        squeeze_train_delta,
        squeeze_holdout_margin,
        rejections,
        notes,
    }
}

fn stats_by_label(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Vec<Stats> {
    let mut block_ids = observations
        .iter()
        .filter(|observation| label_matches(observation.evidence, action, split))
        .map(|observation| observation.evidence.block_id)
        .collect::<Vec<_>>();
    block_ids.sort_unstable();
    block_ids.dedup();
    block_ids
        .into_iter()
        .map(|block_id| stats_for_block(observations, action, split, block_id, eye))
        .collect()
}

fn squeeze_stats_by_label(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Vec<Stats> {
    let mut block_ids = observations
        .iter()
        .filter(|observation| label_matches(observation.evidence, action, split))
        .map(|observation| observation.evidence.block_id)
        .collect::<Vec<_>>();
    block_ids.sort_unstable();
    block_ids.dedup();
    block_ids
        .into_iter()
        .map(|block_id| {
            let values = admitted_squeeze_values(observations, action, split, Some(block_id), eye);
            stats_from_values(&values)
        })
        .collect()
}

fn squeeze_stats_for_label(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Stats {
    let values = admitted_squeeze_values(observations, action, split, None, eye);
    stats_from_values(&values)
}

fn stats_for_block(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    block_id: usize,
    eye: usize,
) -> Stats {
    let values = admitted_values(observations, action, split, Some(block_id), eye);
    stats_from_values(&values)
}

fn stats_for_label(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    eye: usize,
) -> Stats {
    let values = admitted_values(observations, action, split, None, eye);
    stats_from_values(&values)
}

fn admitted_values(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    block_id: Option<usize>,
    eye: usize,
) -> Vec<f32> {
    let values = observations
        .iter()
        .filter_map(|observation| {
            (label_matches(observation.evidence, action, split)
                && block_id.is_none_or(|block_id| observation.evidence.block_id == block_id)
                && observation.phase_time_s >= STABLE_SUFFIX_S
                && observation.presence[eye] > PRESENCE_MIN)
                .then_some(observation.raw[eye]?)
        })
        .collect::<Vec<_>>();
    values
}

fn admitted_squeeze_values(
    observations: &[Observation],
    action: EvidenceAction,
    split: EvidenceSplit,
    block_id: Option<usize>,
    eye: usize,
) -> Vec<f32> {
    observations
        .iter()
        .filter_map(|observation| {
            (label_matches(observation.evidence, action, split)
                && block_id.is_none_or(|block_id| observation.evidence.block_id == block_id)
                && observation.phase_time_s >= STABLE_SUFFIX_S
                && observation.presence[eye] > PRESENCE_MIN)
                .then_some(observation.squeeze_raw[eye]?)
        })
        .collect()
}

fn stats_from_values(values: &[f32]) -> Stats {
    Stats {
        admitted: values.len(),
        median: percentile(values, 0.5),
        iqr: percentile(values, 0.75) - percentile(values, 0.25),
        p10: percentile(values, 0.1),
        p80: percentile(values, 0.8),
        p90: percentile(values, 0.9),
    }
}

fn label_matches(label: EvidenceLabel, action: EvidenceAction, split: EvidenceSplit) -> bool {
    label.action == action && label.split == split
}

fn predict_stats(stats: Stats, closed_ref: f32, open_full: f32) -> (f32, f32) {
    (
        predict_open(stats.median, closed_ref, open_full),
        predict_open(stats.p90, closed_ref, open_full),
    )
}

fn predict_open(raw: f32, closed_ref: f32, open_full: f32) -> f32 {
    ((raw - closed_ref) / (open_full - closed_ref).max(MIN_CLOSE_SPAN)).clamp(0.0, 1.0)
}

fn native_disable_rate(
    observations: &[Observation],
    action: EvidenceAction,
    eye: usize,
) -> Option<f32> {
    let values = observations
        .iter()
        .filter(|observation| {
            label_matches(observation.evidence, action, EvidenceSplit::Validation)
                && observation.phase_time_s >= STABLE_SUFFIX_S
                && observation.native_open[eye].is_some()
        })
        .map(|observation| observation.native_open[eye] == Some(0.0))
        .collect::<Vec<_>>();
    (!values.is_empty())
        .then_some(values.iter().filter(|value| **value).count() as f32 / values.len() as f32)
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
    values[((values.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize]
}

fn median(values: &[f32]) -> f32 {
    percentile(values, 0.5)
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

    fn synthetic(right_holdout: f32, partner_bad: bool) -> Vec<Observation> {
        let mut out = Vec::new();
        for (action, split, block_id, raw) in [
            (
                EvidenceAction::LeftWink,
                EvidenceSplit::Fit,
                803,
                [0.45, 0.60],
            ),
            (
                EvidenceAction::RightWink,
                EvidenceSplit::Fit,
                91,
                [0.60, 0.46],
            ),
            (
                EvidenceAction::LeftWink,
                EvidenceSplit::Fit,
                11,
                [0.451, 0.60],
            ),
            (
                EvidenceAction::RightWink,
                EvidenceSplit::Fit,
                5001,
                [0.60, 0.461],
            ),
            (
                EvidenceAction::LeftWink,
                EvidenceSplit::Validation,
                7777,
                [0.452, if partner_bad { 0.48 } else { 0.60 }],
            ),
            (
                EvidenceAction::RightWink,
                EvidenceSplit::Validation,
                33,
                [0.60, right_holdout],
            ),
        ] {
            for frame in 0..36 {
                out.push(Observation {
                    evidence: EvidenceLabel {
                        action,
                        split,
                        target: None,
                        block_id,
                    },
                    phase_time_s: 1.5 + frame as f32 * 0.05,
                    raw: raw.map(|value| Some(value + frame as f32 * 0.00002)),
                    squeeze_raw: match action {
                        EvidenceAction::LeftWink => [Some(0.56), Some(0.12)],
                        EvidenceAction::RightWink => [Some(0.12), Some(0.57)],
                        _ => [Some(0.12); 2],
                    },
                    presence: [1.0; 2],
                    native_open: [None; 2],
                });
            }
        }
        out
    }

    #[test]
    fn arbitrary_block_ids_do_not_define_wink_meaning() {
        let result = fit_observations(
            &synthetic(0.462, false),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(result.eyes[0].accepted, "{:?}", result.eyes[0].rejections);
        assert!(result.eyes[1].accepted, "{:?}", result.eyes[1].rejections);
        assert_eq!(result.eyes[0].outcome, EyeOutcome::ApplyFloor);
        assert_eq!(result.eyes[1].outcome, EyeOutcome::ApplyFloor);
        assert!(result.profile.eyes[0].floor_enabled);
        assert!(result.profile.eyes[1].floor_enabled);
        assert!(result.profile.eyes[0].wink_depth < 0.25);
        assert!(result.profile.eyes[0].squeeze_enabled);
        assert!(result.profile.eyes[1].squeeze_enabled);
        assert!(result.eyes[0].squeeze_holdout_margin > 0.30);
    }

    #[test]
    fn holdout_failure_isolated_to_one_eye() {
        let result = fit_observations(
            &synthetic(0.54, false),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(result.eyes[0].accepted, "{:?}", result.eyes[0].rejections);
        assert!(!result.eyes[1].accepted);
    }

    #[test]
    fn partner_not_open_rejects_affected_wink() {
        let result = fit_observations(
            &synthetic(0.462, true),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(!result.eyes[0].accepted);
        assert!(!result.eyes[1].accepted);
    }

    #[test]
    fn absent_squeeze_keeps_the_validated_openness_fallback() {
        let mut observations = synthetic(0.462, false);
        for observation in &mut observations {
            observation.squeeze_raw = [Some(0.12); 2];
        }
        let result = fit_observations(
            &observations,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(result.eyes[0].accepted, "{:?}", result.eyes[0].rejections);
        assert!(result.eyes[1].accepted, "{:?}", result.eyes[1].rejections);
        assert!(!result.profile.eyes[0].squeeze_enabled);
        assert!(!result.profile.eyes[1].squeeze_enabled);
    }

    fn deep_current_zero(mut observations: Vec<Observation>, squeeze: bool) -> Vec<Observation> {
        for observation in &mut observations {
            match observation.evidence.action {
                EvidenceAction::LeftWink => observation.raw[0] = Some(0.30),
                EvidenceAction::RightWink => observation.raw[1] = Some(0.30),
                _ => {}
            }
            if !squeeze {
                observation.squeeze_raw = [Some(0.12); 2];
            }
        }
        observations
    }

    #[test]
    fn current_zero_wink_applies_signature_without_a_floor() {
        let observations = deep_current_zero(synthetic(0.30, false), true);
        let result = fit_observations(
            &observations,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        for eye in &result.eyes {
            assert_eq!(eye.outcome, EyeOutcome::ApplySignatureOnly);
            assert!(eye.accepted);
            assert!(eye.proposed.enabled);
            assert!(eye.proposed.squeeze_enabled);
            assert!(!eye.proposed.floor_enabled);
            assert!(eye.rejections.is_empty(), "{:?}", eye.rejections);
        }
    }

    #[test]
    fn current_zero_wink_without_squeeze_needs_no_change() {
        let observations = deep_current_zero(synthetic(0.30, false), false);
        let result = fit_observations(
            &observations,
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        for eye in &result.eyes {
            assert_eq!(eye.outcome, EyeOutcome::NoChangeNeeded);
            assert!(!eye.accepted);
            assert!(!eye.proposed.enabled);
            assert!(!eye.proposed.floor_enabled);
            assert!(eye.rejections.is_empty(), "{:?}", eye.rejections);
        }
        assert!(!result.any_accepted());
    }

    #[test]
    fn partner_open_uses_the_partner_endpoint_coordinate() {
        let mut observations = synthetic(0.462, false);
        for observation in &mut observations {
            observation.raw[0] = observation.raw[0].map(|raw| raw + 0.20);
        }
        let mut left_endpoint = endpoint();
        left_endpoint.baseline += 0.20;
        let result = fit_observations(
            &observations,
            CalibStore {
                left: left_endpoint,
                right: endpoint(),
            },
            0.08,
        );

        let left = &result.eyes[0];
        assert_eq!(
            left.outcome,
            EyeOutcome::ApplyFloor,
            "{:?}",
            left.rejections
        );
        assert!(left.partner_open_holdout_p10 > 0.99);
        let wrong_target_coordinate = predict_open(
            left.holdout_open.median - 0.20,
            left_endpoint.baseline - left_endpoint.blink_depth,
            left_endpoint.baseline - 0.08,
        );
        assert!(wrong_target_coordinate < 0.85);
    }
}
