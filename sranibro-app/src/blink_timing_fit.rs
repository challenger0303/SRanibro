//! Natural-blink evidence and the visible-bottom timing profile.
//!
//! The image recording is replayed through the exact live EyeNet preprocessing
//! path. Train and untouched holdout must both contain repeated, bilateral,
//! short close/open episodes. The fitted value changes only how long an already
//! confirmed fast bilateral blink remains visibly at zero.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use crate::calib_session::{BlinkTimingChange, CommitPermit};
use crate::core::eye_state::{CalibSnapshot, CalibStore};
use crate::core::types::{
    BlinkTimingProfile, DespeckleParams, FlattenParams, MlGeometry, PhotometricCorrection,
};
use crate::geometry_calib::{EvidenceAction, EvidenceLabel, EvidenceSplit, SharedEvidence};
use crate::ml::{brightness, eye_net::EyeNet, preprocess, tvm_params};

const PRESENCE_MIN: f32 = 0.05;
const MIN_TRAIN_BLINKS: usize = 3;
const MIN_HOLDOUT_BLINKS: usize = 2;

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
    pub presence: [f32; 2],
    pub native_open: [Option<f32>; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BlinkEpisode {
    pub duration_ms: f32,
    pub min_open: [f32; 2],
    pub bottom_skew_ms: f32,
}

#[derive(Clone, Debug, Default)]
pub struct Evidence {
    pub episodes: Vec<BlinkEpisode>,
    pub median_duration_ms: f32,
    pub median_bottom_skew_ms: f32,
    pub native_closed_coverage: f32,
}

#[derive(Clone, Debug)]
pub struct FitResult {
    pub accepted: bool,
    pub profile: BlinkTimingProfile,
    pub train: Evidence,
    pub holdout: Evidence,
    pub rejections: Vec<String>,
    pub notes: Vec<String>,
}

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
    pub profile: BlinkTimingProfile,
}

pub fn commit_request(
    result: &FitResult,
    _permit: CommitPermit<BlinkTimingChange>,
) -> Option<ApplyRequest> {
    result.accepted.then_some(ApplyRequest {
        profile: result.profile,
    })
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
                message: "a natural-blink analysis is already running".into(),
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
            .filter(|sample| is_blink_evidence(sample.evidence_label()))
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
            .name("blink-timing-fitter".into())
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
                        message: "natural-blink worker lost its immutable input snapshot".into(),
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
                    .expect("failed spawn must leave natural-blink inputs available");
                lock(&self.shared).status = Status::Idle;
                Err(StartError {
                    message: format!("could not spawn natural-blink worker: {error}"),
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
        .filter(|sample| is_blink_evidence(sample.evidence_label()))
        .count();
    let mut observations = Vec::with_capacity(total);
    for sample in inputs
        .dataset
        .samples()
        .iter()
        .filter(|sample| is_blink_evidence(sample.evidence_label()))
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

fn is_blink_evidence(label: EvidenceLabel) -> bool {
    label.action == EvidenceAction::NaturalBlink
}

pub fn fit_observations(
    observations: &[Observation],
    endpoints: CalibStore,
    open_deadzone: f32,
) -> FitResult {
    let endpoint = [endpoints.left, endpoints.right];
    let train = evidence(observations, EvidenceSplit::Fit, endpoint, open_deadzone);
    let holdout = evidence(
        observations,
        EvidenceSplit::Validation,
        endpoint,
        open_deadzone,
    );
    let mut rejections = Vec::new();
    let mut notes = Vec::new();
    if train.episodes.len() < MIN_TRAIN_BLINKS {
        rejections.push(format!(
            "training contained only {} clean bilateral blinks (need at least {MIN_TRAIN_BLINKS})",
            train.episodes.len()
        ));
    }
    if holdout.episodes.len() < MIN_HOLDOUT_BLINKS {
        rejections.push(format!(
            "untouched holdout contained only {} clean bilateral blinks (need at least {MIN_HOLDOUT_BLINKS})",
            holdout.episodes.len()
        ));
    }
    if train.median_bottom_skew_ms > 70.0 {
        rejections.push(format!(
            "training left/right blink bottoms differed by {:.0} ms (need <=70 ms)",
            train.median_bottom_skew_ms
        ));
    }
    if holdout.median_bottom_skew_ms > 70.0 {
        rejections.push(format!(
            "holdout left/right blink bottoms differed by {:.0} ms (need <=70 ms)",
            holdout.median_bottom_skew_ms
        ));
    }
    if train.median_duration_ms.is_finite()
        && holdout.median_duration_ms.is_finite()
        && (train.median_duration_ms - holdout.median_duration_ms).abs() > 120.0
    {
        rejections.push(format!(
            "training/holdout blink duration differed by {:.0} ms (need <=120 ms)",
            (train.median_duration_ms - holdout.median_duration_ms).abs()
        ));
    }
    if holdout.native_closed_coverage > 0.0 {
        notes.push(format!(
            "native Tobii closed evidence covered {:.0}% of admitted holdout blinks (advisory)",
            holdout.native_closed_coverage * 100.0
        ));
    }

    let source_duration = median(&[train.median_duration_ms, holdout.median_duration_ms]);
    let desired = if source_duration.is_finite() {
        (source_duration * 0.30).clamp(33.0, 58.0)
    } else {
        BlinkTimingProfile::default().min_closed_ms
    };
    let frame_ms = 1000.0 / 120.0;
    let min_closed_ms = (desired / frame_ms).round() * frame_ms;
    let accepted = rejections.is_empty();
    let profile = BlinkTimingProfile {
        enabled: accepted,
        min_closed_ms,
        calibrated_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        train_blinks: train.episodes.len().min(u16::MAX as usize) as u16,
        holdout_blinks: holdout.episodes.len().min(u16::MAX as usize) as u16,
        ..BlinkTimingProfile::default()
    };
    FitResult {
        accepted,
        profile,
        train,
        holdout,
        rejections,
        notes,
    }
}

fn evidence(
    observations: &[Observation],
    split: EvidenceSplit,
    endpoints: [CalibSnapshot; 2],
    open_deadzone: f32,
) -> Evidence {
    #[derive(Clone, Copy)]
    struct Active {
        start_s: f32,
        last_s: f32,
        min_open: [f32; 2],
        min_at: [f32; 2],
        native_closed: bool,
    }

    let mut episodes = Vec::new();
    let mut active: Option<Active> = None;
    let mut native_agree = 0usize;
    let mut block_id = None;
    for observation in observations.iter().filter(|observation| {
        observation.evidence.action == EvidenceAction::NaturalBlink
            && observation.evidence.split == split
    }) {
        // `phase_time_s` restarts for every capture block. Never let an
        // unfinished close in one block become a synthetic blink when the next
        // block begins open.
        if block_id != Some(observation.evidence.block_id) {
            active = None;
            block_id = Some(observation.evidence.block_id);
        }
        let [Some(left_raw), Some(right_raw)] = observation.raw else {
            active = None;
            continue;
        };
        let raw = [left_raw, right_raw];
        if observation
            .presence
            .iter()
            .any(|value| *value <= PRESENCE_MIN)
        {
            active = None;
            continue;
        }
        let openness = std::array::from_fn(|eye| {
            let open_full = endpoints[eye].baseline - open_deadzone.clamp(0.03, 0.20);
            let closed_ref = endpoints[eye].baseline - endpoints[eye].blink_depth;
            ((raw[eye] - closed_ref) / (open_full - closed_ref).max(0.05)).clamp(0.0, 1.0)
        });
        let both_narrow = openness.iter().all(|value| *value < 0.75);
        let both_open = openness.iter().all(|value| *value > 0.88);
        match active.as_mut() {
            None if both_narrow => {
                active = Some(Active {
                    start_s: observation.phase_time_s,
                    last_s: observation.phase_time_s,
                    min_open: openness,
                    min_at: [observation.phase_time_s; 2],
                    native_closed: observation
                        .native_open
                        .iter()
                        .all(|value| value.is_some_and(|value| value <= 0.05)),
                });
            }
            Some(current) => {
                current.last_s = observation.phase_time_s;
                for (eye, value) in openness.into_iter().enumerate() {
                    if value < current.min_open[eye] {
                        current.min_open[eye] = value;
                        current.min_at[eye] = observation.phase_time_s;
                    }
                }
                current.native_closed |= observation
                    .native_open
                    .iter()
                    .all(|value| value.is_some_and(|value| value <= 0.05));
                if both_open {
                    let duration_ms = (observation.phase_time_s - current.start_s) * 1000.0;
                    let bottom_skew_ms = (current.min_at[0] - current.min_at[1]).abs() * 1000.0;
                    let deep_enough = current.min_open.iter().all(|value| *value <= 0.72)
                        && current.min_open.iter().any(|value| *value <= 0.55);
                    if (35.0..=350.0).contains(&duration_ms)
                        && bottom_skew_ms <= 120.0
                        && deep_enough
                    {
                        episodes.push(BlinkEpisode {
                            duration_ms,
                            min_open: current.min_open,
                            bottom_skew_ms,
                        });
                        native_agree += usize::from(current.native_closed);
                    }
                    active = None;
                } else if observation.phase_time_s - current.start_s > 0.45 {
                    // A held close/squint is not a natural blink.
                    active = None;
                }
            }
            None => {}
        }
    }
    let durations = episodes
        .iter()
        .map(|episode| episode.duration_ms)
        .collect::<Vec<_>>();
    let skews = episodes
        .iter()
        .map(|episode| episode.bottom_skew_ms)
        .collect::<Vec<_>>();
    Evidence {
        median_duration_ms: median(&durations),
        median_bottom_skew_ms: median(&skews),
        native_closed_coverage: if episodes.is_empty() {
            0.0
        } else {
            native_agree as f32 / episodes.len() as f32
        },
        episodes,
    }
}

fn median(values: &[f32]) -> f32 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f32::NAN;
    }
    finite.sort_by(f32::total_cmp);
    let middle = finite.len() / 2;
    if finite.len() % 2 == 0 {
        0.5 * (finite[middle - 1] + finite[middle])
    } else {
        finite[middle]
    }
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

    fn synthetic(train_count: usize, holdout_count: usize, one_eye_only: bool) -> Vec<Observation> {
        let mut observations = Vec::new();
        for (split, block_id, count) in [
            (EvidenceSplit::Fit, 731, train_count),
            (EvidenceSplit::Validation, 9901, holdout_count),
        ] {
            let mut time = 0.0f32;
            for _ in 0..count {
                for raw in [0.60, 0.54, 0.43, 0.38, 0.45, 0.56, 0.60] {
                    observations.push(Observation {
                        evidence: EvidenceLabel {
                            action: EvidenceAction::NaturalBlink,
                            split,
                            target: None,
                            block_id,
                        },
                        phase_time_s: time,
                        raw: [
                            Some(raw),
                            Some(if one_eye_only { 0.60 } else { raw + 0.002 }),
                        ],
                        presence: [1.0; 2],
                        native_open: [None; 2],
                    });
                    time += 0.020;
                }
                time += 0.25;
            }
        }
        observations
    }

    #[test]
    fn repeatable_bilateral_blinks_pass_train_and_holdout() {
        let result = fit_observations(
            &synthetic(5, 3, false),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(result.accepted, "{:?}", result.rejections);
        assert!((33.0..=58.0).contains(&result.profile.min_closed_ms));
    }

    #[test]
    fn unilateral_or_missing_holdout_fails_safely() {
        let unilateral = fit_observations(
            &synthetic(5, 3, true),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(!unilateral.accepted);
        let sparse = fit_observations(
            &synthetic(5, 1, false),
            CalibStore {
                left: endpoint(),
                right: endpoint(),
            },
            0.08,
        );
        assert!(!sparse.accepted);
    }

    #[test]
    fn blink_episode_never_crosses_capture_block_boundary() {
        let observations = vec![
            Observation {
                evidence: EvidenceLabel {
                    action: EvidenceAction::NaturalBlink,
                    split: EvidenceSplit::Fit,
                    target: None,
                    block_id: 101,
                },
                phase_time_s: 0.0,
                raw: [Some(0.38), Some(0.382)],
                presence: [1.0; 2],
                native_open: [None; 2],
            },
            Observation {
                evidence: EvidenceLabel {
                    action: EvidenceAction::NaturalBlink,
                    split: EvidenceSplit::Fit,
                    target: None,
                    block_id: 909,
                },
                phase_time_s: 0.10,
                raw: [Some(0.60), Some(0.60)],
                presence: [1.0; 2],
                native_open: [None; 2],
            },
        ];
        let result = evidence(
            &observations,
            EvidenceSplit::Fit,
            [endpoint(), endpoint()],
            0.08,
        );
        assert!(
            result.episodes.is_empty(),
            "a close in one block and open in another must not form a blink"
        );
    }
}
