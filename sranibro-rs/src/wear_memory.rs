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
const CANDIDATE_CONFIRMATIONS: u8 = 6;
const LOST_CONFIRMATIONS: u8 = 12;
const CANDIDATE_MAX_GAP: Duration = Duration::from_secs(2);
const ACTIVE_MAX_GAP: Duration = Duration::from_secs(8);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Context {
    pub device_key: String,
    pub unit_id: String,
    pub pipeline_fingerprint: u64,
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
            Ok(Some(store)) => (store, None),
            Ok(None) => (Store::empty(context.clone()), None),
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
            *score >= MATCH_MIN
                && (*score - second >= MATCH_MARGIN
                    || self.active == Some(self.store.profiles[*index].id)
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
        let existing = self
            .store
            .profiles
            .iter()
            .enumerate()
            .filter_map(|(index, profile)| {
                descriptor_duplicate_score(&profile.descriptor, &descriptor)
                    .map(|score| (index, score))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .filter(|(_, score)| *score >= UPDATE_EXISTING_MIN)
            .map(|(index, _)| index);
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
        let profile = Profile {
            id: existing
                .map(|index| self.store.profiles[index].id)
                .unwrap_or(next_id),
            captured_unix: now,
            descriptor,
            calibration: live.calibration,
            response: live.response.sanitized(),
            wide_baseline: live.wide_baseline,
            wide_entry_ref: live.wide_entry_ref,
        };
        let mut next = self.store.clone();
        let (index, updated) = if let Some(index) = existing {
            next.profiles[index] = profile;
            (index, true)
        } else {
            if next.profiles.len() >= MAX_PROFILES {
                return Err(
                    "Memory is full (8 states). Delete an unused state before saving another."
                        .into(),
                );
            }
            next.profiles.push(profile);
            (next.profiles.len() - 1, false)
        };
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

fn load_store(context: &Context) -> Result<Option<Store>, String> {
    let path = path_for(context);
    let text = match fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Read the old shared file only when its complete context matches.
            let suffix = format!("-{:016x}.toml", context.pipeline_fingerprint);
            let name = path.file_name().unwrap().to_string_lossy();
            let legacy = path.with_file_name(format!("{}.toml", name.trim_end_matches(&suffix)));
            match fs::read_to_string(legacy) {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
                Err(error) => return Err(format!("wearing memory could not be read: {error}")),
            }
        }
        Err(error) => return Err(format!("wearing memory could not be read: {error}")),
    };
    let store: Store =
        toml::from_str(&text).map_err(|error| format!("wearing memory is invalid: {error}"))?;
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
    let path = path_for(&store.context);
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
    publish_atomic(&partial, &path)
        .map_err(|error| format!("wearing memory could not be published: {error}"))
}

fn publish_atomic(partial: &Path, path: &Path) -> std::io::Result<()> {
    let backup = path.with_extension("bak");
    let _ = fs::remove_file(&backup);
    if path.exists() {
        fs::rename(path, &backup)?;
    }
    match fs::rename(partial, path) {
        Ok(()) => {
            let _ = fs::remove_file(backup);
            Ok(())
        }
        Err(error) => {
            let _ = fs::rename(&backup, path);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::core::types::GazeSample;
    use crate::pipeline::EyeFrame;

    #[test]
    fn image_domains_use_different_files() {
        let a = Context::new("pimax_vr4", "test-unit".into(), 1);
        let b = Context::new("pimax_vr4", "test-unit".into(), 2);
        assert_ne!(path_for(&a), path_for(&b));
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
