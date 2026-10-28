//! Appearance-conditioned, explicitly taught wearing-position calibration memory.
//!
//! A profile is created only after the user explicitly confirms that the complete
//! live response is correct. Recenter, threshold edits and Wide-neutral are draft
//! operations and never teach the matcher on their own. Runtime matching is
//! deliberately incapable of creating or mutating profiles: an uncertain image can
//! therefore fail closed, but can never teach SRanibro a progressively worse baseline.

use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::config::EyelidResponseProfile;
use crate::core::eye_state::CalibStore;
use crate::pipeline::{CalibrationFrameSnapshot, WearingCalibrationTarget};

// Schema 1 could contain partial profiles captured after individual editing actions.
// Do not import those into the explicit good-state memory.
const SCHEMA_VERSION: u32 = 2;
const SIDE: usize = 24;
const PIXELS: usize = SIDE * SIDE;
const RECENT_WINDOW: Duration = Duration::from_millis(900);
const CAPTURE_SETTLE: Duration = Duration::from_millis(450);
const CAPTURE_DURATION: Duration = Duration::from_millis(1450);
const MATCH_INTERVAL: Duration = Duration::from_millis(220);
const MIN_DESCRIPTOR_FRAMES: usize = 18;
const MAX_DESCRIPTOR_FRAMES: usize = 42;
const MAX_PROFILES: usize = 8;
const MATCH_MIN: f32 = 0.75;
const MATCH_MARGIN: f32 = 0.018;
/// Saving uses an alignment-sensitive duplicate test, unlike live matching. A
/// small physical reseat should create another learnable state, while repeated
/// captures of the same state update it instead of creating ambiguous twins.
const UPDATE_EXISTING_MIN: f32 = 0.965;
/// Near-identical pictures alone do not establish that a newly verified response
/// is a correction of the same state. Retain materially different coordinates.
const SAME_STATE_COORD_TOLERANCE: f32 = 0.015;
const CANDIDATE_CONFIRMATIONS: u8 = 6;
const LOST_CONFIRMATIONS: u8 = 12;
const CANDIDATE_MAX_GAP: Duration = Duration::from_secs(2);
const ACTIVE_MAX_GAP: Duration = Duration::from_secs(8);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub device_key: String,
    pub unit_id: String,
    #[serde(with = "fingerprint_serde")]
    pub pipeline_fingerprint: u64,
}

// TOML integers are signed 64-bit, but an image-domain fingerprint uses all
// 64 bits. Keep the hash and filename unchanged; only encode the stored value
// as fixed-width hex. Previously saved nonnegative integer values still load.
mod fingerprint_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{value:016x}"))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<u64, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StoredFingerprint {
            Hex(String),
            Legacy(u64),
        }

        match StoredFingerprint::deserialize(deserializer)? {
            StoredFingerprint::Legacy(value) => Ok(value),
            StoredFingerprint::Hex(value) => {
                if value.len() != 16 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return Err(serde::de::Error::custom(
                        "pipeline fingerprint must contain exactly 16 hexadecimal digits",
                    ));
                }
                u64::from_str_radix(&value, 16).map_err(serde::de::Error::custom)
            }
        }
    }
}

impl Context {
    pub fn new(device_key: &str, unit_id: String, pipeline_fingerprint: u64) -> Self {
        Self {
            device_key: crate::config::canonical_device_key(device_key),
            unit_id,
            pipeline_fingerprint,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LiveCalibration {
    pub calibration: CalibStore,
    pub response: EyelidResponseProfile,
    pub wide_baseline: [f32; 2],
    pub wide_entry_ref: [f32; 2],
}

#[derive(Clone, Copy, Debug)]
pub struct ClosedPointEstimate {
    pub median: f32,
    pub spread: f32,
    pub sample_count: usize,
}

/// Select the longest stable *final* portion of a closed-point recording. Times
/// must identify fresh model observations, not repeated UI reads of one frame.
/// Four samples at 10 Hz are sufficient: duration and continuity, not a nominal
/// 120 Hz camera quota, establish coverage. The caller checks that the last sample
/// is recent enough relative to its capture deadline before committing anything.
pub fn closed_point_from_samples(
    samples: &[(Duration, f32)],
) -> Result<ClosedPointEstimate, String> {
    const MIN_SAMPLES: usize = 4;
    const MIN_STABLE: Duration = Duration::from_millis(300);
    const MAX_GAP: Duration = Duration::from_millis(200);
    const MAX_SPREAD: f32 = 0.040;
    const MAX_DRIFT: f32 = 0.015;

    if samples.len() < MIN_SAMPLES {
        return Err("Too few fresh model samples. Hold the eye closed and try again.".into());
    }
    if samples.iter().any(|(_, value)| !value.is_finite())
        || samples.windows(2).any(|pair| pair[1].0 <= pair[0].0)
    {
        return Err("Closed-point samples were invalid or repeated. Try again.".into());
    }
    let median = |values: &[(Duration, f32)]| {
        let mut sorted: Vec<_> = values.iter().map(|(_, value)| *value).collect();
        sorted.sort_by(f32::total_cmp);
        sorted[sorted.len() / 2]
    };
    for start in 0..=samples.len() - MIN_SAMPLES {
        let tail = &samples[start..];
        if tail.last().unwrap().0.saturating_sub(tail[0].0) < MIN_STABLE {
            break;
        }
        if tail.windows(2).any(|pair| pair[1].0 - pair[0].0 > MAX_GAP) {
            continue;
        }
        let mut sorted: Vec<_> = tail.iter().map(|(_, value)| *value).collect();
        sorted.sort_by(f32::total_cmp);
        let spread = sorted[sorted.len() * 9 / 10] - sorted[sorted.len() / 10];
        let middle = sorted[sorted.len() / 2];
        let third = (tail.len() / 3).max(1);
        let drift = (median(&tail[..third]) - median(&tail[tail.len() - third..])).abs();
        // Robust quantiles may discard isolated noise inside the window, but must
        // not hide a reopening at its end, even at a high sample rate.
        let end_is_stable = tail
            .iter()
            .rev()
            .take(3)
            .all(|(_, value)| (value - middle).abs() <= MAX_SPREAD);
        if spread <= MAX_SPREAD && drift <= MAX_DRIFT && end_is_stable {
            return Ok(ClosedPointEstimate {
                median: middle,
                spread,
                sample_count: tail.len(),
            });
        }
    }
    Err("The eye did not stay steadily closed long enough. Keep it closed through the final tone and try again; the previous point was kept.".into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureReason {
    GoodState,
}

impl CaptureReason {
    fn label(self) -> &'static str {
        match self {
            Self::GoodState => "Good wearing state",
        }
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Saved { count: usize, updated: bool },
    Apply(Option<WearingCalibrationTarget>),
    Warning(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EyeDescriptor {
    census: Vec<u8>,
    #[serde(default)]
    thumbnail: Vec<u8>,
    mean: f32,
    contrast: f32,
    field: [f32; 16],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Descriptor {
    dimensions: [[u32; 2]; 2],
    eyes: [EyeDescriptor; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Profile {
    id: u64,
    captured_unix: u64,
    descriptor: Descriptor,
    calibration: CalibStore,
    response: EyelidResponseProfile,
    wide_baseline: [f32; 2],
    wide_entry_ref: [f32; 2],
}

impl Profile {
    fn target(&self, confidence: f32) -> WearingCalibrationTarget {
        WearingCalibrationTarget {
            baseline: [
                self.calibration.left.baseline,
                self.calibration.right.baseline,
            ],
            wide_baseline: self.wide_baseline,
            wide_entry_ref: self.wide_entry_ref,
            manual_range: self.response.manual_range,
            open_point_offset: self.response.open_point_offset,
            closed_point_depth: self.response.closed_point_depth,
            profile_id: self.id,
            confidence,
        }
    }

    fn live_calibration(&self) -> LiveCalibration {
        LiveCalibration {
            calibration: self.calibration,
            response: self.response,
            wide_baseline: self.wide_baseline,
            wide_entry_ref: self.wide_entry_ref,
        }
    }
}

fn calibrations_compatible(a: LiveCalibration, b: LiveCalibration) -> bool {
    let a_response = a.response.sanitized();
    let b_response = b.response.sanitized();
    if a_response.manual_range != b_response.manual_range {
        return false;
    }
    let a_baseline = [a.calibration.left.baseline, a.calibration.right.baseline];
    let b_baseline = [b.calibration.left.baseline, b.calibration.right.baseline];
    (0..2).all(|eye| {
        [
            (a_baseline[eye], b_baseline[eye]),
            (
                a_baseline[eye] - a_response.open_point_offset[eye],
                b_baseline[eye] - b_response.open_point_offset[eye],
            ),
            (
                a_baseline[eye] - a_response.closed_point_depth[eye],
                b_baseline[eye] - b_response.closed_point_depth[eye],
            ),
            (a.wide_baseline[eye], b.wide_baseline[eye]),
            (a.wide_entry_ref[eye], b.wide_entry_ref[eye]),
        ]
        .iter()
        .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= SAME_STATE_COORD_TOLERANCE)
    })
}

/// Pure staging: a failed persistence operation never mutates the live store.
fn stage_remember(
    store: &Store,
    descriptor: Descriptor,
    live: LiveCalibration,
    now: u64,
    next_id: u64,
) -> Result<(Store, usize, bool), String> {
    let existing = store
        .profiles
        .iter()
        .enumerate()
        .filter(|(_, profile)| calibrations_compatible(profile.live_calibration(), live))
        .filter_map(|(index, profile)| {
            descriptor_duplicate_score(&profile.descriptor, &descriptor).map(|score| (index, score))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|(_, score)| *score >= UPDATE_EXISTING_MIN)
        .map(|(index, _)| index);
    let profile = Profile {
        id: existing
            .map(|index| store.profiles[index].id)
            .unwrap_or(next_id),
        captured_unix: now,
        descriptor,
        calibration: live.calibration,
        response: live.response.sanitized(),
        wide_baseline: live.wide_baseline,
        wide_entry_ref: live.wide_entry_ref,
    };
    let mut next = store.clone();
    if let Some(index) = existing {
        next.profiles[index] = profile;
        Ok((next, index, true))
    } else if next.profiles.len() >= MAX_PROFILES {
        Err("Memory is full (8 states). Delete an unused state before saving another.".into())
    } else {
        let index = next.profiles.len();
        next.profiles.push(profile);
        Ok((next, index, false))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Store {
    schema_version: u32,
    context: Context,
    profiles: Vec<Profile>,
}

impl Store {
    fn empty(context: Context) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            context,
            profiles: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct PendingCapture {
    reason: CaptureReason,
    started: Instant,
    live: Option<LiveCalibration>,
}

pub struct Memory {
    store: Store,
    recent: VecDeque<CalibrationFrameSnapshot>,
    last_generation: [u64; 2],
    pending: Option<PendingCapture>,
    last_match: Option<Instant>,
    last_accepted: Option<Instant>,
    candidate: Option<u64>,
    candidate_count: u8,
    active: Option<u64>,
    lost_count: u8,
    last_confidence: f32,
    load_warning: Option<String>,
    descriptor_job: Option<(Instant, Receiver<Result<Analysis, String>>)>,
}

struct Analysis {
    descriptor: Descriptor,
    scores: Vec<(usize, f32)>,
}

/// Native invalid data must not be mistaken for an unsupported capability.
/// The fallback is supplied only when the processed/model evidence is valid.
pub fn neutral_openness(
    native: &crate::core::types::EyeSample,
    fallback: Option<f32>,
    teaching: bool,
) -> bool {
    if native.openness_reported {
        native.openness_valid && native.openness.is_finite() && native.openness >= 0.88
    } else {
        fallback
            .is_some_and(|value| value.is_finite() && value >= if teaching { 0.85 } else { 0.60 })
    }
}

impl Memory {
    pub fn load(context: Context) -> Self {
        let (store, load_warning) = match load_store(&context) {
            Ok((Some(store), recovered)) => (
                store,
                recovered.then(|| "Recovered saved wearing states from an interrupted save. The backup is retained until the next successful save.".into()),
            ),
            Ok((None, _)) => (Store::empty(context.clone()), None),
            Err(error) => (Store::empty(context.clone()), Some(error)),
        };
        Self {
            store,
            recent: VecDeque::new(),
            last_generation: [0; 2],
            pending: None,
            last_match: None,
            last_accepted: None,
            candidate: None,
            candidate_count: 0,
            active: None,
            lost_count: 0,
            last_confidence: 0.0,
            load_warning,
            descriptor_job: None,
        }
    }

    pub fn profile_count(&self) -> usize {
        self.store.profiles.len()
    }

    pub fn context(&self) -> &Context {
        &self.store.context
    }

    pub fn thumbnail(&self, id: u64) -> Option<[Vec<u8>; 2]> {
        let profile = self.store.profiles.iter().find(|p| p.id == id)?;
        if profile
            .descriptor
            .eyes
            .iter()
            .any(|e| e.thumbnail.len() != PIXELS)
        {
            return None;
        }
        Some(std::array::from_fn(|i| {
            profile.descriptor.eyes[i].thumbnail.clone()
        }))
    }

    pub fn profiles(&self) -> Vec<(u64, [f32; 2], [f32; 2])> {
        self.store
            .profiles
            .iter()
            .map(|p| {
                let b = [p.calibration.left.baseline, p.calibration.right.baseline];
                (
                    p.id,
                    std::array::from_fn(|i| b[i] - p.response.open_point_offset[i]),
                    std::array::from_fn(|i| b[i] - p.response.closed_point_depth[i]),
                )
            })
            .collect()
    }

    pub fn remove(&mut self, id: u64) -> Result<(), String> {
        let mut next = self.store.clone();
        next.profiles.retain(|p| p.id != id);
        save_store(&next)?;
        self.store = next;
        self.disable();
        Ok(())
    }

    pub fn trial(&mut self, id: u64) -> Option<WearingCalibrationTarget> {
        let target = self.store.profiles.iter().find(|p| p.id == id)?.target(0.0);
        self.disable();
        Some(target)
    }

    pub fn last_generation(&self) -> [u64; 2] {
        self.last_generation
    }

    pub fn active_confidence(&self) -> Option<f32> {
        self.active.map(|_| self.last_confidence)
    }

    pub fn capture_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn take_load_warning(&mut self) -> Option<String> {
        self.load_warning.take()
    }

    pub fn begin_capture(&mut self, reason: CaptureReason, now: Instant) {
        self.descriptor_job = None;
        self.pending = Some(PendingCapture {
            reason,
            started: now,
            live: None,
        });
        self.candidate = None;
        self.candidate_count = 0;
        self.active = None;
        self.lost_count = 0;
    }

    pub fn cancel_active(&mut self) -> Event {
        self.last_accepted = None;
        self.pending = None;
        self.descriptor_job = None;
        self.active = None;
        self.candidate = None;
        self.candidate_count = 0;
        self.lost_count = 0;
        Event::Apply(None)
    }

    pub fn disable(&mut self) -> Event {
        self.pending = None;
        self.cancel_active()
    }

    pub fn ingest(&mut self, samples: Vec<CalibrationFrameSnapshot>, now: Instant) {
        for sample in samples {
            if sample.source_generation[0] <= self.last_generation[0]
                || sample.source_generation[1] <= self.last_generation[1]
            {
                continue;
            }
            self.last_generation = sample.source_generation;
            self.recent.push_back(sample);
        }
        while self
            .recent
            .front()
            .is_some_and(|sample| now.saturating_duration_since(sample.captured_at) > RECENT_WINDOW)
        {
            self.recent.pop_front();
        }
    }

    pub fn update(
        &mut self,
        now: Instant,
        eligible_neutral: bool,
        live: Option<LiveCalibration>,
    ) -> Vec<Event> {
        let mut events = Vec::new();
        self.expire_evidence(now, &mut events);
        if let Some(pending) = self.pending {
            // Validate throughout the settled capture, not just at its final frame.
            // In particular, unknown native openness must not silently authorize
            // a closed-eye reference on devices using the UI's ML fallback.
            if now.saturating_duration_since(pending.started) >= CAPTURE_SETTLE && !eligible_neutral
            {
                self.pending = None;
                self.descriptor_job = None;
                events.push(Event::Warning(
                    "Nothing saved: keep both eyes relaxed and open throughout the recording, then try again.".into(),
                ));
                return events;
            }
            if now.saturating_duration_since(pending.started) >= CAPTURE_DURATION {
                if !eligible_neutral {
                    self.pending = None;
                    self.descriptor_job = None;
                    events.push(Event::Warning(format!(
                        "{} was not memorized: look straight ahead with both eyes relaxed and try again",
                        pending.reason.label()
                    )));
                } else if let Some(live) = live {
                    if self.pending.as_ref().is_some_and(|p| p.live.is_none()) {
                        self.pending.as_mut().unwrap().live = Some(live);
                    }
                    let capture_live = self.pending.as_ref().and_then(|p| p.live).unwrap_or(live);
                    let frames: Vec<_> = self
                        .recent
                        .iter()
                        .filter(|sample| {
                            sample.captured_at >= pending.started + CAPTURE_SETTLE
                                && sample_is_neutral(sample)
                        })
                        .cloned()
                        .collect();
                    let Some(result) = self.background_descriptor(frames, now) else {
                        return events;
                    };
                    self.pending = None;
                    match result {
                        Ok(analysis) => match self.remember(analysis.descriptor, capture_live) {
                            Ok((index, updated)) => {
                                let _ = index;
                                events.push(Event::Saved {
                                    count: self.store.profiles.len(),
                                    updated,
                                });
                            }
                            Err(error) => events.push(Event::Warning(error)),
                        },
                        Err(error) => events.push(Event::Warning(format!(
                            "{} was not memorized: {error}",
                            pending.reason.label()
                        ))),
                    }
                } else {
                    self.pending = None;
                    self.descriptor_job = None;
                    events.push(Event::Warning(
                        "Tracking is not ready. Nothing was saved.".into(),
                    ));
                }
            }
            return events;
        }

        if self.store.profiles.is_empty()
            || self
                .last_match
                .is_some_and(|last| now.saturating_duration_since(last) < MATCH_INTERVAL)
        {
            return events;
        }
        self.last_match = Some(now);
        if !eligible_neutral {
            self.descriptor_job = None;
            return events;
        }

        let frames: Vec<_> = self
            .recent
            .iter()
            .filter(|sample| sample_is_neutral(sample))
            .cloned()
            .collect();
        let Some(Ok(analysis)) = self.background_descriptor(frames, now) else {
            return events;
        };
        let mut scored = analysis.scores;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        let best = scored.first().copied();
        let second = scored.get(1).map(|value| value.1).unwrap_or(0.0);
        let accepted = best.filter(|(index, score)| {
            let conflicting_tie = scored.iter().skip(1).any(|(other_index, other_score)| {
                *score - *other_score < MATCH_MARGIN
                    && !calibrations_compatible(
                        self.store.profiles[*index].live_calibration(),
                        self.store.profiles[*other_index].live_calibration(),
                    )
            });
            *score >= MATCH_MIN
                && (*score - second >= MATCH_MARGIN
                    || (self.active == Some(self.store.profiles[*index].id) && !conflicting_tie)
                    || scored.len() == 1)
        });

        let Some((index, score)) = accepted else {
            self.candidate = None;
            self.candidate_count = 0;
            self.lost_count = self.lost_count.saturating_add(1);
            if self.lost_count >= LOST_CONFIRMATIONS && self.active.take().is_some() {
                self.last_confidence = 0.0;
                events.push(Event::Apply(None));
            }
            return events;
        };
        self.lost_count = 0;
        self.last_accepted = Some(now);
        let id = self.store.profiles[index].id;
        if self.candidate == Some(id) {
            self.candidate_count = self.candidate_count.saturating_add(1);
        } else {
            self.candidate = Some(id);
            self.candidate_count = 1;
        }
        self.last_confidence = score;
        if self.candidate_count >= CANDIDATE_CONFIRMATIONS && self.active != Some(id) {
            self.active = Some(id);
            events.push(Event::Apply(Some(self.store.profiles[index].target(score))));
        }
        events
    }

    fn expire_evidence(&mut self, now: Instant, events: &mut Vec<Event>) {
        let age = self
            .last_accepted
            .map(|last| now.saturating_duration_since(last));
        if age.is_none_or(|age| age > CANDIDATE_MAX_GAP) {
            self.candidate = None;
            self.candidate_count = 0;
        }
        if age.is_none_or(|age| age > ACTIVE_MAX_GAP) && self.active.take().is_some() {
            self.last_confidence = 0.0;
            events.push(Event::Apply(None));
        }
    }

    fn background_descriptor(
        &mut self,
        frames: Vec<CalibrationFrameSnapshot>,
        now: Instant,
    ) -> Option<Result<Analysis, String>> {
        if let Some((started, receiver)) = &self.descriptor_job {
            let result = match receiver.try_recv() {
                Ok(result) => result,
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => Err("Image analysis stopped; try again.".into()),
            };
            let stale = now.saturating_duration_since(*started) > Duration::from_secs(2);
            self.descriptor_job = None;
            return Some(if stale {
                Err("Image analysis was too old; try again.".into())
            } else {
                result
            });
        }
        let (sender, receiver) = mpsc::sync_channel(1);
        let profiles = self.store.profiles.clone();
        match std::thread::Builder::new()
            .name("wear-descriptor".into())
            .spawn(move || {
                let result = build_descriptor(&frames).map(|descriptor| {
                    let scores = profiles
                        .iter()
                        .enumerate()
                        .filter_map(|(index, profile)| {
                            recovery_score(&profile.descriptor, &descriptor)
                                .map(|score| (index, score))
                        })
                        .collect();
                    Analysis { descriptor, scores }
                });
                let _ = sender.send(result);
            }) {
            Ok(_) => self.descriptor_job = Some((now, receiver)),
            Err(error) => return Some(Err(format!("Could not start image analysis: {error}"))),
        }
        None
    }

    fn remember(
        &mut self,
        descriptor: Descriptor,
        live: LiveCalibration,
    ) -> Result<(usize, bool), String> {
        let now = unix_time();
        let next_id = unix_time_millis().max(
            self.store
                .profiles
                .iter()
                .map(|profile| profile.id)
                .max()
                .unwrap_or(0)
                .saturating_add(1),
        );
        let (next, index, updated) = stage_remember(&self.store, descriptor, live, now, next_id)?;
        save_store(&next)?;
        self.store = next;
        Ok((index, updated))
    }
}

fn sample_is_neutral(sample: &CalibrationFrameSnapshot) -> bool {
    for eye in [sample.gaze.left, sample.gaze.right] {
        if eye.openness_reported
            && (!eye.openness_valid || !eye.openness.is_finite() || eye.openness < 0.86)
        {
            return false;
        }
        if eye.gaze_valid && eye.gaze.iter().all(|value| value.is_finite()) {
            let transverse = (eye.gaze[0] * eye.gaze[0] + eye.gaze[1] * eye.gaze[1]).sqrt();
            if transverse.atan2(eye.gaze[2].abs().max(1e-4)).to_degrees() > 18.0 {
                return false;
            }
        }
    }
    true
}

fn build_descriptor(samples: &[CalibrationFrameSnapshot]) -> Result<Descriptor, String> {
    let mut samples: Vec<_> = samples.iter().rev().take(MAX_DESCRIPTOR_FRAMES).collect();
    samples.reverse();
    if samples.len() < MIN_DESCRIPTOR_FRAMES {
        return Err(format!(
            "only {}/{} fresh neutral frames were available",
            samples.len(),
            MIN_DESCRIPTOR_FRAMES
        ));
    }
    let dimensions = [
        [samples[0].frames[0].width, samples[0].frames[0].height],
        [samples[0].frames[1].width, samples[0].frames[1].height],
    ];
    if samples.iter().any(|sample| {
        [
            [sample.frames[0].width, sample.frames[0].height],
            [sample.frames[1].width, sample.frames[1].height],
        ] != dimensions
    }) {
        return Err("eye-camera dimensions changed during capture".into());
    }
    let eyes: [Result<EyeDescriptor, String>; 2] = std::array::from_fn(|eye| {
        let frames: Vec<Vec<u8>> = samples
            .iter()
            .map(|sample| downsample(&sample.frames[eye]))
            .collect::<Result<_, _>>()?;
        let mut median = vec![0u8; PIXELS];
        let mut column = Vec::with_capacity(frames.len());
        for pixel in 0..PIXELS {
            column.clear();
            column.extend(frames.iter().map(|frame| frame[pixel]));
            let middle = column.len() / 2;
            column.select_nth_unstable(middle);
            median[pixel] = column[middle];
        }
        let mean = median.iter().map(|value| *value as f32).sum::<f32>() / PIXELS as f32;
        let contrast = (median
            .iter()
            .map(|value| (*value as f32 - mean).powi(2))
            .sum::<f32>()
            / PIXELS as f32)
            .sqrt();
        let mut field = [0.0; 16];
        for gy in 0..4 {
            for gx in 0..4 {
                let mut sum = 0.0;
                let mut n = 0usize;
                for y in gy * SIDE / 4..(gy + 1) * SIDE / 4 {
                    for x in gx * SIDE / 4..(gx + 1) * SIDE / 4 {
                        sum += median[y * SIDE + x] as f32;
                        n += 1;
                    }
                }
                field[gy * 4 + gx] = sum / n.max(1) as f32;
            }
        }
        Ok(EyeDescriptor {
            census: census(&median),
            thumbnail: median,
            mean,
            contrast,
            field,
        })
    });
    Ok(Descriptor {
        dimensions,
        eyes: [eyes[0].clone()?, eyes[1].clone()?],
    })
}

fn downsample(frame: &crate::pipeline::EyeFrame) -> Result<Vec<u8>, String> {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if width == 0 || height == 0 || frame.pixels.len() < width * height {
        return Err("an eye-camera frame was incomplete".into());
    }
    let mut output = vec![0u8; PIXELS];
    for y in 0..SIDE {
        let sy0 = y * height / SIDE;
        let sy1 = ((y + 1) * height / SIDE).max(sy0 + 1).min(height);
        for x in 0..SIDE {
            let sx0 = x * width / SIDE;
            let sx1 = ((x + 1) * width / SIDE).max(sx0 + 1).min(width);
            let mut sum = 0u64;
            let mut count = 0u64;
            for sy in sy0..sy1 {
                for sx in sx0..sx1 {
                    sum += frame.pixels[sy * width + sx] as u64;
                    count += 1;
                }
            }
            output[y * SIDE + x] = (sum / count.max(1)) as u8;
        }
    }
    Ok(output)
}

fn census(image: &[u8]) -> Vec<u8> {
    let mut output = vec![0u8; PIXELS];
    let neighbours = [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ];
    for y in 1..SIDE - 1 {
        for x in 1..SIDE - 1 {
            let centre = image[y * SIDE + x];
            let mut bits = 0u8;
            for (bit, (dx, dy)) in neighbours.iter().enumerate() {
                let xx = (x as isize + dx) as usize;
                let yy = (y as isize + dy) as usize;
                if image[yy * SIDE + xx] > centre.saturating_add(2) {
                    bits |= 1 << bit;
                }
            }
            output[y * SIDE + x] = bits;
        }
    }
    output
}

fn descriptor_score(reference: &Descriptor, current: &Descriptor) -> Option<f32> {
    if reference.dimensions != current.dimensions {
        return None;
    }
    let eyes: [f32; 2] =
        std::array::from_fn(|eye| eye_score(&reference.eyes[eye], &current.eyes[eye]));
    Some((0.65 * eyes[0].min(eyes[1]) + 0.35 * (eyes[0] + eyes[1]) * 0.5).clamp(0.0, 1.0))
}

fn recovery_score(reference: &Descriptor, current: &Descriptor) -> Option<f32> {
    Some(
        0.35 * descriptor_score(reference, current)?
            + 0.65 * descriptor_duplicate_score(reference, current)?,
    )
}

/// Stricter, alignment-sensitive comparison used only when deciding whether a
/// Save is a duplicate. Live recovery intentionally searches a ±2 px window;
/// doing that here merged distinct wearing positions into a permanent "1 saved".
fn descriptor_duplicate_score(reference: &Descriptor, current: &Descriptor) -> Option<f32> {
    if reference.dimensions != current.dimensions {
        return None;
    }
    let eyes: [f32; 2] = std::array::from_fn(|eye| {
        let reference_eye = &reference.eyes[eye];
        let current_eye = &current.eyes[eye];
        let mut different = 0u32;
        let mut bits = 0u32;
        for y in 3..SIDE - 3 {
            for x in 3..SIDE - 3 {
                different += (reference_eye.census[y * SIDE + x]
                    ^ current_eye.census[y * SIDE + x])
                    .count_ones();
                bits += 8;
            }
        }
        let exact_shape = if bits > 0 {
            1.0 - different as f32 / bits as f32
        } else {
            0.0
        };
        0.90 * exact_shape + 0.10 * photometric_score(reference_eye, current_eye)
    });
    Some((0.65 * eyes[0].min(eyes[1]) + 0.35 * (eyes[0] + eyes[1]) * 0.5).clamp(0.0, 1.0))
}

fn eye_score(reference: &EyeDescriptor, current: &EyeDescriptor) -> f32 {
    let mut best_shape = 0.0f32;
    for dy in -2isize..=2 {
        for dx in -2isize..=2 {
            let mut different = 0u32;
            let mut bits = 0u32;
            for y in 3..SIDE - 3 {
                for x in 3..SIDE - 3 {
                    let xx = x as isize + dx;
                    let yy = y as isize + dy;
                    if xx < 1 || yy < 1 || xx >= (SIDE - 1) as isize || yy >= (SIDE - 1) as isize {
                        continue;
                    }
                    different += (reference.census[y * SIDE + x]
                        ^ current.census[yy as usize * SIDE + xx as usize])
                        .count_ones();
                    bits += 8;
                }
            }
            if bits > 0 {
                best_shape = best_shape.max(1.0 - different as f32 / bits as f32);
            }
        }
    }
    0.82 * best_shape + 0.18 * photometric_score(reference, current)
}

fn photometric_score(reference: &EyeDescriptor, current: &EyeDescriptor) -> f32 {
    let mean_error = (reference.mean - current.mean).abs() / 90.0;
    let contrast_error = (reference.contrast - current.contrast).abs() / 60.0;
    let field_error = reference
        .field
        .iter()
        .zip(current.field)
        .map(|(a, b)| (a - b).abs() / 90.0)
        .sum::<f32>()
        / 16.0;
    (1.0 - (0.25 * mean_error + 0.20 * contrast_error + 0.55 * field_error)).clamp(0.0, 1.0)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn unix_time_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

fn path_for(context: &Context) -> PathBuf {
    let safe = |value: &str| -> String {
        value
            .chars()
            .map(|ch| {
                if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                    ch
                } else {
                    '_'
                }
            })
            .collect()
    };
    crate::config::base_dir().join("wear-memory").join(format!(
        "{}-{}-{:016x}.toml",
        safe(&context.device_key),
        safe(&context.unit_id),
        context.pipeline_fingerprint
    ))
}

fn load_store(context: &Context) -> Result<(Option<Store>, bool), String> {
    load_store_with_recovery_at(context, &path_for(context))
}

#[cfg(test)]
fn load_store_at(context: &Context, path: &Path) -> Result<Option<Store>, String> {
    load_store_with_recovery_at(context, path).map(|(store, _)| store)
}

fn load_store_with_recovery_at(
    context: &Context,
    path: &Path,
) -> Result<(Option<Store>, bool), String> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Versions using two renames could be interrupted after moving the
            // primary file to .bak. Recover only a valid, matching-context backup,
            // and keep it intact until a subsequent save has really published.
            // An existing but invalid primary is never silently hidden by this.
            match fs::read_to_string(path.with_extension("bak")) {
                Ok(text) => {
                    return decode_store(context, &text)
                        .map_err(|error| format!("wearing memory backup could not be recovered: {error}"))?
                        .map(|store| (Some(store), true))
                        .ok_or_else(|| "Saved wearing memory backup belongs to a different context or schema; it has not been applied.".into());
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("wearing memory backup could not be read: {error}"))
                }
            }
            // Read the old shared file only when its complete context matches.
            let suffix = format!("-{:016x}.toml", context.pipeline_fingerprint);
            let name = path.file_name().unwrap().to_string_lossy();
            let Some(stem) = name.strip_suffix(&suffix) else {
                return Ok((None, false));
            };
            let legacy = path.with_file_name(format!("{stem}.toml"));
            match fs::read_to_string(legacy) {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    return Ok((None, false))
                }
                Err(error) => return Err(format!("wearing memory could not be read: {error}")),
            }
        }
        Err(error) => return Err(format!("wearing memory could not be read: {error}")),
    };
    decode_store(context, &text).map(|store| (store, false))
}

fn decode_store(context: &Context, text: &str) -> Result<Option<Store>, String> {
    let store: Store =
        toml::from_str(text).map_err(|error| format!("wearing memory is invalid: {error}"))?;
    if store.schema_version != SCHEMA_VERSION || store.context != *context {
        return Ok(None);
    }
    if store.profiles.len() > MAX_PROFILES
        || store.profiles.iter().any(|p| {
            p.descriptor.eyes.iter().any(|eye| {
                eye.census.len() != PIXELS
                    || !eye.mean.is_finite()
                    || !eye.contrast.is_finite()
                    || eye.field.iter().any(|value| !value.is_finite())
            }) || [p.calibration.left.baseline, p.calibration.right.baseline]
                .into_iter()
                .chain(p.wide_baseline)
                .chain(p.wide_entry_ref)
                .any(|value| !value.is_finite())
        })
    {
        return Err("Saved wearing memory is invalid; it has not been applied.".into());
    }
    Ok(Some(store))
}

fn save_store(store: &Store) -> Result<(), String> {
    save_store_at(store, &path_for(&store.context))
}

fn save_store_at(store: &Store, path: &Path) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("wearing memory folder could not be created: {error}"))?;
    }
    let text = toml::to_string(store)
        .map_err(|error| format!("wearing memory could not be encoded: {error}"))?;
    let partial = path.with_extension("partial");
    let mut file = fs::File::create(&partial)
        .map_err(|error| format!("wearing memory could not be written: {error}"))?;
    file.write_all(text.as_bytes())
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("wearing memory could not be flushed: {error}"))?;
    drop(file);
    publish_atomic(&partial, path)
        .map_err(|error| format!("wearing memory could not be published: {error}"))
}

fn publish_atomic(partial: &Path, path: &Path) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::{
            MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
        };
        let from: Vec<u16> = partial.as_os_str().encode_wide().chain(Some(0)).collect();
        let to: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        // Same-directory replacement keeps the old primary present until the
        // replacement succeeds. Do not delete it or its recovery backup first.
        if unsafe {
            MoveFileExW(
                from.as_ptr(),
                to.as_ptr(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error());
        }
    }
    #[cfg(not(windows))]
    fs::rename(partial, path)?;
    #[cfg(unix)]
    if let Some(parent) = path.parent() {
        if let Ok(directory) = fs::File::open(parent) {
            let _ = directory.sync_all();
        }
    }
    let _ = fs::remove_file(path.with_extension("bak"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::core::types::GazeSample;
    use crate::pipeline::EyeFrame;

    #[test]
    fn closed_point_accepts_a_stable_low_rate_tail_without_a_camera_hz_quota() {
        let samples: Vec<_> = [0.201, 0.199, 0.202, 0.200]
            .into_iter()
            .enumerate()
            .map(|(i, value)| (Duration::from_millis(i as u64 * 100), value))
            .collect();
        let result = closed_point_from_samples(&samples).unwrap();
        assert_eq!(result.sample_count, 4);
        assert!((result.median - 0.201).abs() < 0.001);
        assert!(result.spread < 0.004);
    }

    #[test]
    fn closed_point_discards_initial_motion_but_never_a_late_reopening() {
        let mut samples: Vec<_> = [0.50, 0.40, 0.30, 0.201, 0.199, 0.200, 0.202, 0.201]
            .into_iter()
            .enumerate()
            .map(|(i, value)| (Duration::from_millis(i as u64 * 100), value))
            .collect();
        let result = closed_point_from_samples(&samples).unwrap();
        assert_eq!(result.sample_count, 5);
        assert!((result.median - 0.201).abs() < 0.001);
        samples.push((Duration::from_millis(800), 0.40));
        assert!(closed_point_from_samples(&samples).is_err());
        let mut high_rate: Vec<_> = (0..75)
            .map(|i| (Duration::from_millis(i * 10), 0.2))
            .collect();
        high_rate.last_mut().unwrap().1 = 0.35;
        assert!(closed_point_from_samples(&high_rate).is_err());
    }

    #[test]
    fn closed_point_rejects_moving_or_sparse_or_repeated_samples() {
        let ramp: Vec<_> = (0..=10)
            .map(|i| (Duration::from_millis(i * 75), 0.15 + i as f32 * 0.024))
            .collect();
        assert!(closed_point_from_samples(&ramp).is_err());
        let reversed: Vec<_> = ramp.iter().map(|(t, v)| (*t, 0.54 - *v)).collect();
        assert!(closed_point_from_samples(&reversed).is_err());
        let sparse: Vec<_> = (0..4)
            .map(|i| (Duration::from_millis(i * 250), 0.2))
            .collect();
        assert!(closed_point_from_samples(&sparse).is_err());
        let repeated = [(Duration::ZERO, 0.2); 40];
        assert!(closed_point_from_samples(&repeated).is_err());
        let short: Vec<_> = (0..20)
            .map(|i| (Duration::from_millis(i * 10), 0.2))
            .collect();
        assert!(closed_point_from_samples(&short).is_err());
        let mut invalid = ramp;
        invalid[3].1 = f32::NAN;
        assert!(closed_point_from_samples(&invalid).is_err());
    }

    fn fixture_live() -> LiveCalibration {
        LiveCalibration {
            calibration: crate::core::eye_state::SRanipalState::new().snapshot_all(),
            response: EyelidResponseProfile::default(),
            wide_baseline: [0.61, 0.63],
            wide_entry_ref: [0.71, 0.73],
        }
    }

    fn fixture_descriptor(bias: f32) -> Descriptor {
        build_descriptor(
            &(1..=24)
                .map(|n| sample(n, 0, 1.0, bias))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    #[test]
    fn duplicate_save_preserves_materially_different_verified_lighting_states() {
        let descriptor = fixture_descriptor(0.0);
        let brighter = fixture_descriptor(20.0);
        assert!(descriptor_duplicate_score(&descriptor, &brighter).unwrap() > UPDATE_EXISTING_MIN);
        let empty = Store::empty(Context::new("pimax_vr4", "unit-test".into(), 1));
        let live = fixture_live();
        let (first, _, _) = stage_remember(&empty, descriptor.clone(), live, 1, 1).unwrap();
        let mut changed = live;
        changed.response.closed_point_depth = [0.28, 0.30];
        let (second, index, updated) = stage_remember(&first, brighter, changed, 2, 2).unwrap();
        assert!(!updated);
        assert_eq!(index, 1);
        assert_eq!(second.profiles.len(), 2);
        assert_eq!(second.profiles[0].response, live.response);
        assert_eq!(second.profiles[1].response, changed.response);
        let (refreshed, index, updated) = stage_remember(&second, descriptor, live, 3, 3).unwrap();
        assert!(updated);
        assert_eq!(index, 0);
        assert_eq!(refreshed.profiles.len(), 2);
        assert_eq!(refreshed.profiles[0].id, 1);
        assert_eq!(refreshed.profiles[1].response, changed.response);
    }

    #[test]
    fn conflicting_save_at_capacity_keeps_all_old_states_but_an_actual_update_fits() {
        let descriptor = fixture_descriptor(0.0);
        let mut store = Store::empty(Context::new("pimax_vr4", "unit-test".into(), 1));
        for i in 0..MAX_PROFILES {
            let mut live = fixture_live();
            live.calibration.left.baseline = 0.40 + i as f32 * 0.04;
            store = stage_remember(&store, descriptor.clone(), live, 1, i as u64 + 1)
                .unwrap()
                .0;
        }
        assert_eq!(store.profiles.len(), MAX_PROFILES);
        let before = toml::to_string(&store).unwrap();
        let mut conflict = fixture_live();
        conflict.calibration.left.baseline = 0.90;
        assert!(stage_remember(&store, descriptor.clone(), conflict, 2, 99).is_err());
        assert_eq!(toml::to_string(&store).unwrap(), before);
        let old = store.profiles[0].live_calibration();
        let (updated, index, replaced) = stage_remember(&store, descriptor, old, 2, 99).unwrap();
        assert!(replaced);
        assert_eq!(index, 0);
        assert_eq!(updated.profiles.len(), MAX_PROFILES);
    }

    #[test]
    fn indistinguishable_conflicting_states_do_not_keep_an_active_match_by_tie_break() {
        let now = Instant::now();
        let mut memory = empty_memory();
        let descriptor = fixture_descriptor(0.0);
        let live = fixture_live();
        memory.store = stage_remember(&memory.store, descriptor.clone(), live, 1, 1)
            .unwrap()
            .0;
        let mut changed = live;
        changed.response.closed_point_depth = [0.28; 2];
        memory.store = stage_remember(&memory.store, descriptor.clone(), changed, 2, 2)
            .unwrap()
            .0;
        memory.active = Some(1);
        memory.last_accepted = Some(now);
        for i in 0..LOST_CONFIRMATIONS {
            let time = now + Duration::from_millis(i as u64 * 221);
            let (sender, receiver) = mpsc::sync_channel(1);
            sender
                .send(Ok(Analysis {
                    descriptor: descriptor.clone(),
                    scores: vec![(0, 1.0), (1, 1.0)],
                }))
                .unwrap();
            memory.descriptor_job = Some((time, receiver));
            let events = memory.update(time, true, None);
            if i == LOST_CONFIRMATIONS - 1 {
                assert!(matches!(events.as_slice(), [Event::Apply(None)]));
            } else {
                assert!(events.is_empty());
            }
        }
        assert!(memory.active.is_none());
        assert!(memory.candidate.is_none());
    }

    struct RecoveryTestDirectory(PathBuf);

    impl RecoveryTestDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "sranibro-wearing-recovery-test-{}-{nonce}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for RecoveryTestDirectory {
        fn drop(&mut self) {
            for name in ["state.toml", "state.partial", "state.bak"] {
                let _ = fs::remove_file(self.0.join(name));
            }
            let _ = fs::remove_dir(&self.0);
        }
    }

    #[test]
    fn missing_primary_recovers_valid_backup_until_next_successful_publish() {
        let dir = RecoveryTestDirectory::new();
        let path = dir.0.join("state.toml");
        let context = Context::new("pimax_vr4", "unit-test".into(), 1);
        let (store, _, _) = stage_remember(
            &Store::empty(context.clone()),
            fixture_descriptor(0.0),
            fixture_live(),
            1,
            1,
        )
        .unwrap();
        save_store_at(&store, &path).unwrap();
        fs::rename(&path, path.with_extension("bak")).unwrap();
        let (restored, recovered) = load_store_with_recovery_at(&context, &path).unwrap();
        assert!(recovered);
        assert_eq!(restored.unwrap().profiles.len(), 1);
        assert!(path.with_extension("bak").exists());
        assert!(!path.exists());
        save_store_at(&store, &path).unwrap();
        assert!(path.exists());
        assert!(!path.with_extension("bak").exists());
        assert!(!load_store_with_recovery_at(&context, &path).unwrap().1);
    }

    #[test]
    fn backup_never_hides_corrupt_primary_or_invalid_or_wrong_context_evidence() {
        let dir = RecoveryTestDirectory::new();
        let path = dir.0.join("state.toml");
        let context = Context::new("pimax_vr4", "unit-test".into(), 1);
        let text = toml::to_string(&Store::empty(context.clone())).unwrap();
        fs::write(path.with_extension("bak"), &text).unwrap();
        fs::write(&path, "[not valid TOML").unwrap();
        assert!(load_store_with_recovery_at(&context, &path)
            .unwrap_err()
            .contains("invalid"));
        assert_eq!(fs::read_to_string(&path).unwrap(), "[not valid TOML");
        fs::remove_file(&path).unwrap();
        let other = Context::new("pimax_vr4", "different-unit".into(), 1);
        assert!(load_store_with_recovery_at(&other, &path)
            .unwrap_err()
            .contains("different context"));
        fs::write(path.with_extension("bak"), "[invalid backup").unwrap();
        assert!(load_store_with_recovery_at(&context, &path)
            .unwrap_err()
            .contains("backup"));
        assert_eq!(
            fs::read_to_string(path.with_extension("bak")).unwrap(),
            "[invalid backup"
        );
    }

    #[test]
    fn failed_publish_leaves_both_existing_primary_and_recovery_backup_untouched() {
        let dir = RecoveryTestDirectory::new();
        let path = dir.0.join("state.toml");
        fs::write(&path, "primary evidence").unwrap();
        fs::write(path.with_extension("bak"), "backup evidence").unwrap();
        assert!(publish_atomic(&path.with_extension("partial"), &path).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "primary evidence");
        assert_eq!(
            fs::read_to_string(path.with_extension("bak")).unwrap(),
            "backup evidence"
        );
    }

    #[test]
    fn fingerprint_round_trips_across_the_full_u64_range() {
        for fingerprint in [0, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
            let context = Context::new("pimax_vr4", "unit-test".into(), fingerprint);
            let text = toml::to_string(&context).unwrap();
            assert!(text.contains(&format!("pipeline_fingerprint = \"{fingerprint:016x}\"")));
            let restored: Context = toml::from_str(&text).unwrap();
            assert_eq!(restored, context);
            assert_eq!(path_for(&restored), path_for(&context));
        }
    }

    #[test]
    fn fingerprint_accepts_legacy_nonnegative_toml_integers() {
        for fingerprint in [0, 1, i64::MAX as u64] {
            let text = format!(
                "device_key = \"pimax_vr4\"\nunit_id = \"unit-test\"\npipeline_fingerprint = {fingerprint}\n"
            );
            let context: Context = toml::from_str(&text).unwrap();
            assert_eq!(context.pipeline_fingerprint, fingerprint);
        }
    }

    #[test]
    fn fingerprint_rejects_malformed_stored_values() {
        for value in [
            "\"\"",
            "\"1\"",
            "\"fffffffffffffff\"",
            "\"fffffffffffffffff\"",
            "\"0xffffffffffffffff\"",
            "\"gggggggggggggggg\"",
            "\"-000000000000001\"",
            "\"000000000000000 \"",
            "-1",
            "1.0",
            "true",
        ] {
            let text = format!(
                "device_key = \"pimax_vr4\"\nunit_id = \"unit-test\"\npipeline_fingerprint = {value}\n"
            );
            assert!(
                toml::from_str::<Context>(&text).is_err(),
                "accepted {value}"
            );
        }
    }

    #[test]
    fn store_save_load_and_overwrite_support_high_bit_fingerprints() {
        // Only this newly created temporary directory is written. Never use
        // path_for/save_store here: tests must not touch the user's memory.
        struct TestDirectory(PathBuf);
        impl Drop for TestDirectory {
            fn drop(&mut self) {
                for name in ["state.toml", "state.partial", "state.bak"] {
                    let _ = fs::remove_file(self.0.join(name));
                }
                let _ = fs::remove_dir(&self.0);
            }
        }
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "sranibro-wear-memory-test-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let directory = TestDirectory(directory);
        let path = directory.0.join("state.toml");
        let descriptor = build_descriptor(
            &(1..=24)
                .map(|generation| sample(generation, 0, 1.0, 0.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let calibration = crate::core::eye_state::SRanipalState::new().snapshot_all();

        for fingerprint in [0, i64::MAX as u64, i64::MAX as u64 + 1, u64::MAX] {
            let context = Context::new("pimax_vr4", "unit-test".into(), fingerprint);
            let mut store = Store::empty(context.clone());
            store.profiles.push(Profile {
                id: 1,
                captured_unix: 1,
                descriptor: descriptor.clone(),
                calibration,
                response: EyelidResponseProfile::default(),
                wide_baseline: [0.61, 0.63],
                wide_entry_ref: [0.71, 0.73],
            });
            save_store_at(&store, &path).unwrap();
            let restored = load_store_at(&context, &path).unwrap().unwrap();
            assert_eq!(restored.context, context);
            assert_eq!(restored.profiles.len(), 1);
            assert_eq!(restored.profiles[0].wide_baseline, [0.61, 0.63]);
            assert_eq!(restored.profiles[0].wide_entry_ref, [0.71, 0.73]);

            store.profiles[0].response.closed_point_depth = [0.31, 0.37];
            save_store_at(&store, &path).unwrap();
            let restored = load_store_at(&context, &path).unwrap().unwrap();
            assert_eq!(
                restored.profiles[0].response.closed_point_depth,
                [0.31, 0.37]
            );
            assert!(!path.with_extension("partial").exists());
            assert!(!path.with_extension("bak").exists());
        }
    }

    #[test]
    fn image_domains_use_different_files() {
        let a = Context::new("pimax_vr4", "test-unit".into(), 1);
        let b = Context::new("pimax_vr4", "test-unit".into(), 2);
        assert_ne!(path_for(&a), path_for(&b));
        for hmd in ["pimax_xr5", "psvr2", "varjo", "starvr"] {
            let other=Context::new(hmd,"test-unit".into(),1);
            assert_ne!(path_for(&a),path_for(&other));
            let text=toml::to_string(&Store::empty(a.clone())).unwrap();
            assert!(decode_store(&other,&text).unwrap().is_none());
        }
        assert!(decode_store(&b,&toml::to_string(&Store::empty(a)).unwrap()).unwrap().is_none());
    }

    #[test]
    fn manual_edit_cancels_pending_capture_and_match() {
        let mut memory = Memory {
            store: Store::empty(Context::new("pimax_vr4", "test-unit".into(), 1)),
            recent: VecDeque::new(),
            last_generation: [0; 2],
            pending: None,
            last_match: None,
            last_accepted: Some(Instant::now()),
            candidate: Some(7),
            candidate_count: 4,
            active: Some(7),
            lost_count: 0,
            last_confidence: 0.9,
            load_warning: None,
            descriptor_job: None,
        };
        memory.begin_capture(CaptureReason::GoodState, Instant::now());
        assert!(memory.capture_pending());
        memory.cancel_active();
        assert!(!memory.capture_pending());
        assert!(memory.active_confidence().is_none());
        assert!(memory.candidate.is_none());
    }

    fn empty_memory() -> Memory {
        Memory {
            store: Store::empty(Context::new("pimax_vr4", "unit-test".into(), 1)),
            recent: VecDeque::new(),
            last_generation: [0; 2],
            pending: None,
            last_match: None,
            last_accepted: None,
            candidate: None,
            candidate_count: 0,
            active: None,
            lost_count: 0,
            last_confidence: 0.0,
            load_warning: None,
            descriptor_job: None,
        }
    }

    #[test]
    fn short_blinks_hold_but_long_missing_evidence_expires_once() {
        let now = Instant::now();
        let mut memory = empty_memory();
        memory.active = Some(7);
        memory.candidate = Some(7);
        memory.candidate_count = 5;
        memory.last_confidence = 0.9;
        memory.last_accepted = Some(now);
        assert!(memory
            .update(now + Duration::from_millis(500), false, None)
            .is_empty());
        assert_eq!(memory.active, Some(7));
        assert_eq!(memory.candidate_count, 5);
        assert!(memory
            .update(now + Duration::from_secs(3), false, None)
            .is_empty());
        assert_eq!(memory.active, Some(7));
        assert_eq!(memory.candidate_count, 0);
        let events = memory.update(now + Duration::from_secs(9), false, None);
        assert!(matches!(events.as_slice(), [Event::Apply(None)]));
        assert!(memory.active_confidence().is_none());
        assert!(memory
            .update(now + Duration::from_secs(10), false, None)
            .is_empty());
    }

    #[test]
    fn bad_pose_during_settled_capture_cannot_be_hidden_by_final_open_frame() {
        let now = Instant::now();
        let mut memory = empty_memory();
        memory.begin_capture(CaptureReason::GoodState, now);
        assert!(memory
            .update(now + Duration::from_millis(100), false, None)
            .is_empty());
        assert!(memory.capture_pending());
        let events = memory.update(now + Duration::from_millis(600), false, None);
        assert!(matches!(events.as_slice(), [Event::Warning(_)]));
        assert!(!memory.capture_pending());
        assert!(memory
            .update(now + Duration::from_secs(2), true, None)
            .is_empty());
        assert_eq!(memory.profile_count(), 0);
    }

    #[test]
    fn teaching_requires_open_evidence_even_without_native_openness() {
        let mut native = crate::core::types::EyeSample::default();
        assert!(!neutral_openness(&native, None, true));
        for value in [0.0, 0.5, f32::NAN, f32::INFINITY] {
            assert!(!neutral_openness(&native, Some(value), true));
        }
        assert!(neutral_openness(&native, Some(0.9), true));
        assert!(neutral_openness(&native, Some(0.7), false));
        assert!(!neutral_openness(&native, Some(0.7), true));
        native.openness_reported = true;
        assert!(!neutral_openness(&native, Some(1.0), true));
        native.openness_valid = true;
        native.openness = 0.1;
        assert!(!neutral_openness(&native, Some(1.0), true));
        native.openness = 0.95;
        assert!(neutral_openness(&native, None, true));
    }

    fn sample(generation: u64, shift: isize, gain: f32, bias: f32) -> CalibrationFrameSnapshot {
        let make = |eye: usize| {
            let mut pixels = vec![0u8; 48 * 48];
            for y in 0..48isize {
                for x in 0..48isize {
                    let xx = (x - shift).clamp(0, 47);
                    let base = (((xx * 5 + y * 3 + eye as isize * 17) % 180) + 30) as f32;
                    pixels[y as usize * 48 + x as usize] =
                        (base * gain + bias).clamp(0.0, 255.0) as u8;
                }
            }
            EyeFrame {
                generation,
                width: 48,
                height: 48,
                pixels: Arc::from(pixels),
            }
        };
        CalibrationFrameSnapshot {
            captured_at: Instant::now(),
            source_generation: [generation; 2],
            affine: [[1.0, 0.0]; 2],
            frames: [make(0), make(1)],
            gaze: GazeSample::default(),
        }
    }

    #[test]
    fn descriptor_tolerates_brightness_and_small_translation() {
        let a: Vec<_> = (1..=24).map(|n| sample(n, 0, 1.0, 0.0)).collect();
        let b: Vec<_> = (25..=48).map(|n| sample(n, 1, 0.78, 20.0)).collect();
        let da = build_descriptor(&a).unwrap();
        let db = build_descriptor(&b).unwrap();
        assert!(descriptor_score(&da, &db).unwrap() > MATCH_MIN);
        let duplicate_score = descriptor_duplicate_score(&da, &db).unwrap();
        assert!(duplicate_score < descriptor_score(&da, &db).unwrap());
        assert!(duplicate_score < UPDATE_EXISTING_MIN);
    }

    #[test]
    fn duplicate_score_is_exact_for_the_same_capture() {
        let samples: Vec<_> = (1..=24).map(|n| sample(n, 0, 1.0, 0.0)).collect();
        let descriptor = build_descriptor(&samples).unwrap();
        assert_eq!(
            descriptor_duplicate_score(&descriptor, &descriptor),
            Some(1.0)
        );
    }

    #[test]
    fn remembered_positions_can_be_distinguished_after_restart() {
        let a = build_descriptor(&(1..=24).map(|n| sample(n, 0, 1.0, 0.0)).collect::<Vec<_>>())
            .unwrap();
        let b = build_descriptor(&(1..=24).map(|n| sample(n, 2, 1.0, 0.0)).collect::<Vec<_>>())
            .unwrap();
        assert!(descriptor_duplicate_score(&a, &b).unwrap() < UPDATE_EXISTING_MIN);
        assert!(recovery_score(&a, &a).unwrap() - recovery_score(&b, &a).unwrap() >= MATCH_MARGIN);
        assert!(recovery_score(&b, &b).unwrap() - recovery_score(&a, &b).unwrap() >= MATCH_MARGIN);
    }

    #[test]
    fn mismatched_dimensions_fail_closed() {
        let a: Vec<_> = (1..=24).map(|n| sample(n, 0, 1.0, 0.0)).collect();
        let mut b = a.clone();
        b[0].frames[0].width = 47;
        assert!(build_descriptor(&b).is_err());
    }
}
