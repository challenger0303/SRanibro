//! All-HMD categorical gaze/eyelid capture.
//!
//! This protocol is deliberately independent of Safe Geometry Fit. It records
//! categorical nine-point gaze instructions plus eyelid actions for the
//! gaze-dependent eyelid fitter, but never changes geometry, trains a model, alters
//! gaze output, or treats the instructed target as calibrated gaze truth.

use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

use crate::geometry_calib::{
    export_dataset_recording, GazeTarget, GeometryDataset, GeometrySample, SampleKind,
};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
// Natural blinks need enough temporal resolution to measure the visible bottom.
// All other phases remain at 20 Hz to bound memory and CPU usage.
const BLINK_SAMPLE_INTERVAL: Duration = Duration::from_millis(15);
const STALL_THRESHOLD: Duration = Duration::from_secs(1);
/// A phase is usable only when it retained at least half of its nominal 20 Hz
/// samples. This tolerates ordinary UI scheduling jitter while detecting a
/// stopped eye camera or a capture that ran while frames were unavailable.
const MIN_SAMPLE_COVERAGE: f32 = 0.50;
const WARMUP_SECONDS: f32 = 3.0;
/// Dedicated no-sampling instruction screen before each eyelid-action group.
/// The old protocol changed action and began settling immediately, so the user
/// could not finish reading the eyelid directions before the target advanced.
const PROMPT_SECONDS: f32 = 6.0;
/// Automatic pose previews use a real three-second visual countdown. Group
/// explanations are manual and remain visible until the user continues.
const POSE_COUNTDOWN_SECONDS: f32 = 3.0;
// 2.20 s leaves at least 0.90 s of stable, post-reaction evidence after the
// analysis trim. The former 1.60 s phase could never form one valid one-second
// gaze block, making the pooled residual model mathematically unreachable.
const OPEN_SECONDS: f32 = 2.20;
const OPEN_SETTLE_SECONDS: f32 = 0.55;
const SLOW_SECONDS: f32 = 3.40;
const ACTION_SETTLE_SECONDS: f32 = 0.45;
const BLINK_SECONDS: f32 = 2.00;

const TRAIN_ORDER_A: [GazeTarget; 9] = [
    GazeTarget::Center,
    GazeTarget::Left,
    GazeTarget::Right,
    GazeTarget::Up,
    GazeTarget::Down,
    GazeTarget::UpLeft,
    GazeTarget::UpRight,
    GazeTarget::DownLeft,
    GazeTarget::DownRight,
];

const TRAIN_ORDER_B: [GazeTarget; 9] = [
    GazeTarget::Center,
    GazeTarget::DownLeft,
    GazeTarget::UpRight,
    GazeTarget::Down,
    GazeTarget::Left,
    GazeTarget::Up,
    GazeTarget::Right,
    GazeTarget::DownRight,
    GazeTarget::UpLeft,
];

// A different deterministic order prevents the holdout from being an immediate replay
// of the train sequence while keeping instructions reproducible across machines.
const HOLDOUT_ORDER_A: [GazeTarget; 9] = [
    GazeTarget::Center,
    GazeTarget::DownRight,
    GazeTarget::UpLeft,
    GazeTarget::Right,
    GazeTarget::Down,
    GazeTarget::UpRight,
    GazeTarget::Left,
    GazeTarget::DownLeft,
    GazeTarget::Up,
];

const HOLDOUT_ORDER_B: [GazeTarget; 9] = [
    GazeTarget::Center,
    GazeTarget::Right,
    GazeTarget::UpLeft,
    GazeTarget::Down,
    GazeTarget::DownRight,
    GazeTarget::Left,
    GazeTarget::UpRight,
    GazeTarget::Up,
    GazeTarget::DownLeft,
];

const CARDINALS: [GazeTarget; 5] = [
    GazeTarget::Center,
    GazeTarget::Left,
    GazeTarget::Right,
    GazeTarget::Up,
    GazeTarget::Down,
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CaptureProtocol {
    #[default]
    GazeDirections,
    InitialSetup,
    /// XR5-only biometric evidence for the offline Python eyelid-model factory.
    /// This reuses the broad master pose coverage but never applies a live fit.
    PythonEyelidDataset,
}

impl CaptureProtocol {
    pub const fn id(self) -> &'static str {
        match self {
            Self::GazeDirections => "gaze_eyelid_compensation_v1",
            Self::InitialSetup => "eyelid_master_v1",
            Self::PythonEyelidDataset => "xr5_python_eyelid_dataset_v1",
        }
    }

    pub fn total_seconds(self) -> f32 {
        protocol_for(self).iter().map(|phase| phase.seconds).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptureAction {
    RelaxedOpen,
    HalfOpen,
    GentleClosed,
    SlowCloseOpen,
    NaturalBlink,
    LeftWink,
    RightWink,
}

impl CaptureAction {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RelaxedOpen => "relaxed_open",
            Self::HalfOpen => "half_open",
            Self::GentleClosed => "gentle_closed",
            Self::SlowCloseOpen => "slow_close_open",
            Self::NaturalBlink => "natural_blink",
            Self::LeftWink => "left_wink",
            Self::RightWink => "right_wink",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Phase {
    target: Option<GazeTarget>,
    action: Option<CaptureAction>,
    /// Instruction-only action. Prompt phases never record frames and therefore
    /// keep `action=None`, preserving the meaning of a recording phase.
    prompt_action: Option<CaptureAction>,
    holdout: bool,
    /// Whether this phase contributes labelled evidence. Practice phases expose the
    /// real target/action timing but deliberately never clone or retain camera frames.
    recording: bool,
    /// An instruction screen that does not advance until the user confirms it.
    /// Its nominal duration is retained only for deterministic progress estimates.
    wait_for_continue: bool,
    seconds: f32,
    settle_seconds: f32,
}

impl Phase {
    fn warmup() -> Self {
        Self {
            target: None,
            action: None,
            prompt_action: None,
            holdout: false,
            recording: false,
            wait_for_continue: false,
            seconds: WARMUP_SECONDS,
            settle_seconds: WARMUP_SECONDS,
        }
    }

    fn prompt(action: CaptureAction, holdout: bool) -> Self {
        Self {
            target: None,
            action: None,
            prompt_action: Some(action),
            holdout,
            recording: false,
            wait_for_continue: true,
            seconds: PROMPT_SECONDS,
            settle_seconds: PROMPT_SECONDS,
        }
    }

    fn capture(
        target: GazeTarget,
        action: CaptureAction,
        holdout: bool,
        seconds: f32,
        settle_seconds: f32,
    ) -> Self {
        Self {
            target: Some(target),
            action: Some(action),
            prompt_action: None,
            holdout,
            recording: true,
            wait_for_continue: false,
            seconds,
            settle_seconds,
        }
    }

    fn practice(
        target: GazeTarget,
        action: CaptureAction,
        holdout: bool,
        seconds: f32,
        settle_seconds: f32,
    ) -> Self {
        Self {
            target: Some(target),
            action: Some(action),
            prompt_action: None,
            holdout,
            recording: false,
            wait_for_continue: false,
            seconds,
            settle_seconds,
        }
    }

    fn records(self, elapsed_s: f32) -> bool {
        self.recording && self.action.is_some() && elapsed_s >= self.settle_seconds
    }

    fn sample_interval(self) -> Duration {
        if self.action == Some(CaptureAction::NaturalBlink) {
            BLINK_SAMPLE_INTERVAL
        } else {
            SAMPLE_INTERVAL
        }
    }

    fn minimum_samples(self) -> usize {
        if !self.recording {
            return 0;
        }
        let active_seconds = (self.seconds - self.settle_seconds).max(0.0);
        ((active_seconds / self.sample_interval().as_secs_f32()) * MIN_SAMPLE_COVERAGE).ceil()
            as usize
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Idle,
    /// Session resources are locked, but no timer or sampling begins until the
    /// user confirms that the complete protocol explanation has been read.
    Ready {
        instruction: String,
    },
    Running {
        phase_index: usize,
        progress: f32,
        remaining_s: f32,
        target: Option<GazeTarget>,
        action: Option<CaptureAction>,
        holdout: bool,
        recording: bool,
        settling: bool,
        paused: bool,
        samples_in_phase: usize,
        stereo_stalled: bool,
        /// Manual explanation screen. No phase clock or sampling advances until
        /// `continue_step` is called.
        awaiting_confirmation: bool,
        /// Time left in the current automatic phase, quantized by the UI for the
        /// VR countdown so the SteamVR texture is replaced at most once per second.
        phase_remaining_s: f32,
        /// Expected eyelid openness for the slow-close visual, from 0 to 100.
        pose_progress: Option<u8>,
        instruction: String,
    },
    Done {
        samples: usize,
        evidence_complete: bool,
        missing_phases: Vec<usize>,
    },
}

pub struct GazeResidualCapture {
    protocol: CaptureProtocol,
    phases: Vec<Phase>,
    phase: Option<usize>,
    entered: Instant,
    completed_seconds: f32,
    last_sample: Instant,
    last_generation: [u64; 2],
    accepted_per_phase: Vec<usize>,
    quota_met_at: Vec<Option<Instant>>,
    awaiting_ready: bool,
    paused_at: Option<Instant>,
    dataset: GeometryDataset,
    completed_sample_count: usize,
    pub last_error: Option<String>,
}

impl Default for GazeResidualCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl GazeResidualCapture {
    pub fn new() -> Self {
        let protocol = CaptureProtocol::GazeDirections;
        let phases = protocol_for(protocol);
        let accepted_per_phase = vec![0; phases.len()];
        let quota_met_at = vec![None; phases.len()];
        Self {
            protocol,
            phases,
            phase: None,
            entered: Instant::now(),
            completed_seconds: 0.0,
            last_sample: Instant::now(),
            last_generation: [0; 2],
            accepted_per_phase,
            quota_met_at,
            awaiting_ready: false,
            paused_at: None,
            dataset: GeometryDataset::default(),
            completed_sample_count: 0,
            last_error: None,
        }
    }

    pub fn start(&mut self, generation: [u64; 2]) {
        self.start_protocol(CaptureProtocol::GazeDirections, generation);
    }

    pub fn start_protocol(&mut self, protocol: CaptureProtocol, generation: [u64; 2]) {
        self.protocol = protocol;
        self.phases = protocol_for(protocol);
        self.accepted_per_phase = vec![0; self.phases.len()];
        self.quota_met_at = vec![None; self.phases.len()];
        self.phase = Some(0);
        self.entered = Instant::now();
        self.completed_seconds = 0.0;
        self.last_sample = Instant::now() - SAMPLE_INTERVAL;
        self.last_generation = generation;
        self.awaiting_ready = true;
        self.paused_at = None;
        self.dataset.samples.clear();
        self.completed_sample_count = 0;
        self.last_error = None;
    }

    pub fn abort(&mut self) {
        self.phase = None;
        self.awaiting_ready = false;
        self.paused_at = None;
        self.accepted_per_phase.fill(0);
        self.quota_met_at.fill(None);
        self.dataset.samples.clear();
        self.dataset.samples.shrink_to_fit();
        self.completed_sample_count = 0;
        self.completed_seconds = 0.0;
        self.last_error = None;
    }

    pub const fn protocol(&self) -> CaptureProtocol {
        self.protocol
    }

    pub fn is_running(&self) -> bool {
        self.phase.is_some_and(|index| index < self.phases.len())
    }

    pub fn is_done(&self) -> bool {
        self.phase == Some(self.phases.len())
    }

    pub fn is_ready(&self) -> bool {
        self.is_running() && self.awaiting_ready
    }

    /// Start the first timed phase after the user has read the protocol. This is
    /// deliberately separate from `start`, which only reserves the session and
    /// snapshots calibration state.
    pub fn begin(&mut self) -> bool {
        if !self.is_ready() {
            return false;
        }
        self.awaiting_ready = false;
        self.entered = Instant::now();
        self.last_sample = Instant::now() - SAMPLE_INTERVAL;
        true
    }

    pub fn is_awaiting_confirmation(&self) -> bool {
        if self.awaiting_ready || self.is_paused() {
            return false;
        }
        self.phase
            .and_then(|index| self.phases.get(index))
            .is_some_and(|phase| phase.wait_for_continue)
    }

    /// Leave a manual action-family explanation and start the next timed phase.
    /// Manual reading time is deliberately not charged to the capture clock.
    pub fn continue_step(&mut self) -> bool {
        if !self.is_awaiting_confirmation() {
            return false;
        }
        let index = self.phase.expect("confirmation requires an active phase");
        self.completed_seconds += self.phases[index].seconds;
        self.phase = Some(index + 1);
        self.entered = Instant::now();
        self.last_sample = Instant::now() - SAMPLE_INTERVAL;
        true
    }

    pub fn is_paused(&self) -> bool {
        self.paused_at.is_some()
    }

    /// Freeze the current phase clock and sampling gate. The raw camera pipeline
    /// may continue running, but no hidden/minimized frame can acquire a target label.
    pub fn pause(&mut self) {
        if self.is_running() && !self.awaiting_ready && self.paused_at.is_none() {
            self.paused_at = Some(Instant::now());
        }
    }

    /// Resume without charging the paused wall-clock interval to the active phase
    /// or to the stereo-stall timer.
    pub fn resume(&mut self) {
        let Some(paused_at) = self.paused_at.take() else {
            return;
        };
        let now = Instant::now();
        let paused_for = now.saturating_duration_since(paused_at);
        self.entered = self.entered.checked_add(paused_for).unwrap_or(now);
        self.last_sample = self.last_sample.checked_add(paused_for).unwrap_or(now);
        for at in self.quota_met_at.iter_mut().flatten() {
            *at = at.checked_add(paused_for).unwrap_or(*at);
        }
    }

    /// Remove an event-loop hiatus from the active prompt and discard the hidden
    /// camera generations produced while no usable instruction was visible.
    pub fn suspend_for(&mut self, duration: Duration, generation: [u64; 2]) {
        if !self.is_running() || self.awaiting_ready || self.is_paused() || duration.is_zero() {
            return;
        }
        let now = Instant::now();
        self.entered = self.entered.checked_add(duration).unwrap_or(now);
        self.last_sample = self.last_sample.checked_add(duration).unwrap_or(now);
        for at in self.quota_met_at.iter_mut().flatten() {
            *at = at.checked_add(duration).unwrap_or(*at);
        }
        self.last_generation = generation;
    }

    pub fn last_generation(&self) -> [u64; 2] {
        self.last_generation
    }

    pub fn discard_through(&mut self, generation: [u64; 2]) {
        self.last_generation[0] = self.last_generation[0].max(generation[0]);
        self.last_generation[1] = self.last_generation[1].max(generation[1]);
    }

    fn phase_elapsed_at(&self, now: Instant) -> Duration {
        self.paused_at
            .unwrap_or(now)
            .saturating_duration_since(self.entered)
    }

    fn phase_elapsed(&self) -> Duration {
        self.phase_elapsed_at(Instant::now())
    }

    pub fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    pub fn tick_at(&mut self, now: Instant) {
        if self.awaiting_ready || self.is_paused() {
            return;
        }
        for _ in 0..=self.phases.len() {
            let Some(index) = self.phase else { return };
            if index >= self.phases.len() {
                return;
            }
            let phase = self.phases[index];
            if phase.wait_for_continue {
                return;
            }
            let deadline = self
                .entered
                .checked_add(Duration::from_secs_f32(phase.seconds))
                .unwrap_or(now);
            if now < deadline {
                return;
            }
            let quota_met =
                self.accepted_per_phase.get(index).copied().unwrap_or(0) >= phase.minimum_samples();
            if phase.recording && !quota_met {
                return;
            }
            let next_entered = self
                .quota_met_at
                .get(index)
                .copied()
                .flatten()
                .filter(|at| *at > deadline)
                .unwrap_or(deadline);
            self.completed_seconds += phase.seconds;
            self.phase = Some(index + 1);
            self.entered = next_entered;
            let interval = self
                .phases
                .get(index + 1)
                .copied()
                .map(Phase::sample_interval)
                .unwrap_or(SAMPLE_INTERVAL);
            self.last_sample = next_entered.checked_sub(interval).unwrap_or(next_entered);
        }
    }

    /// True only when cloning the latest stereo frame can result in an accepted
    /// 20 Hz sample. UI callers use this before copying the two 200x200 buffers.
    pub fn wants_frame(&self, generation: [u64; 2]) -> bool {
        self.wants_frame_at(generation, Instant::now())
    }

    pub fn wants_frame_at(&self, generation: [u64; 2], captured_at: Instant) -> bool {
        if self.awaiting_ready || self.is_paused() {
            return false;
        }
        let Some(index) = self.phase.filter(|index| *index < self.phases.len()) else {
            return false;
        };
        let phase = self.phases[index];
        if phase.wait_for_continue {
            return false;
        }
        phase.records(
            self.phase_elapsed_at(captured_at)
                .as_secs_f32()
                .min(phase.seconds),
        ) && generation[0] > self.last_generation[0]
            && generation[1] > self.last_generation[1]
            && captured_at.saturating_duration_since(self.last_sample) >= phase.sample_interval()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_frame(
        &mut self,
        generation: [u64; 2],
        native_timestamp_us: Option<u64>,
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
        brightness_affine: [[f32; 2]; 2],
        native_open: [Option<f32>; 2],
        native_gaze: [Option<[f32; 3]>; 2],
        native_pupil_pos: [Option<[f32; 2]>; 2],
    ) -> bool {
        self.on_frame_at(
            Instant::now(),
            generation,
            native_timestamp_us,
            left,
            right,
            brightness_affine,
            native_open,
            native_gaze,
            native_pupil_pos,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_frame_at(
        &mut self,
        captured_at: Instant,
        generation: [u64; 2],
        native_timestamp_us: Option<u64>,
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
        brightness_affine: [[f32; 2]; 2],
        native_open: [Option<f32>; 2],
        native_gaze: [Option<[f32; 3]>; 2],
        native_pupil_pos: [Option<[f32; 2]>; 2],
    ) -> bool {
        if !self.wants_frame_at(generation, captured_at) {
            return false;
        }
        let index = self.phase.expect("wants_frame requires an active phase");
        let phase = self.phases[index];
        let elapsed = self
            .phase_elapsed_at(captured_at)
            .as_secs_f32()
            .min(phase.seconds);
        let (Some((lw, lh, left)), Some((rw, rh, right))) = (left, right) else {
            return false;
        };
        let left_len = (lw as usize).saturating_mul(lh as usize);
        let right_len = (rw as usize).saturating_mul(rh as usize);
        if lw == 0
            || lh == 0
            || rw == 0
            || rh == 0
            || left.len() < left_len
            || right.len() < right_len
        {
            self.last_error = Some("The newest stereo eye frame is incomplete.".into());
            return false;
        }
        let target = phase.target;
        let action = phase.action.expect("recording phase has an action");
        let active_time = (elapsed - phase.settle_seconds).max(0.0);
        let active_duration = (phase.seconds - phase.settle_seconds).max(0.001);
        let progress = (active_time / active_duration).clamp(0.0, 1.0);
        let expected_open = match action {
            CaptureAction::RelaxedOpen if target.is_none() => Some(1.0),
            CaptureAction::HalfOpen => Some(0.5),
            CaptureAction::GentleClosed => Some(0.0),
            CaptureAction::SlowCloseOpen => Some(slow_target(progress)),
            _ => None,
        };
        let kind = sample_kind(action, target, phase.holdout);
        self.dataset.samples.push(GeometrySample {
            kind,
            commanded_target: target,
            expected_open,
            phase_time_s: active_time,
            left: left[..left_len].to_vec(),
            right: right[..right_len].to_vec(),
            left_size: (lw, lh),
            right_size: (rw, rh),
            brightness_affine,
            native_open,
            native_gaze,
            native_pupil_pos,
            frame_generation: generation,
            native_timestamp_us,
            phase_index: index,
        });
        self.accepted_per_phase[index] += 1;
        if self.quota_met_at[index].is_none()
            && self.accepted_per_phase[index] >= phase.minimum_samples()
        {
            self.quota_met_at[index] = Some(captured_at);
        }
        self.last_generation = generation;
        self.last_sample = self
            .last_sample
            .checked_add(phase.sample_interval())
            .filter(|scheduled| *scheduled <= captured_at)
            .unwrap_or(captured_at);
        self.last_error = None;
        true
    }

    pub fn export_recording(&self, path: &Path, metadata: &str) -> io::Result<()> {
        if !self.is_done() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "finish the landmark/residual capture before exporting it",
            ));
        }
        let metadata = self.evidence_metadata(metadata);
        export_dataset_recording(path, &self.dataset, &metadata)
    }

    /// Move a completed biometric dataset into a background exporter without cloning
    /// its image buffers. The caller must restore the dataset if the export fails.
    pub(crate) fn take_export_dataset(
        &mut self,
        metadata: &str,
    ) -> io::Result<(GeometryDataset, String)> {
        if !self.is_done() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "finish the landmark/residual capture before exporting it",
            ));
        }
        if self.dataset.samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "the completed research dataset is not available",
            ));
        }
        self.completed_sample_count = self.dataset.samples.len();
        let metadata = self.evidence_metadata(metadata);
        Ok((std::mem::take(&mut self.dataset), metadata))
    }

    /// Return ownership after a failed background export so Retry remains lossless.
    pub(crate) fn restore_export_dataset(&mut self, dataset: GeometryDataset) {
        if self.is_done() && self.dataset.samples.is_empty() {
            self.completed_sample_count = dataset.samples.len();
            self.dataset = dataset;
        }
    }

    /// Drop biometric image buffers immediately after a successful ZIP export while
    /// retaining the completed status and sample count for the UI.
    pub fn release_exported_frames(&mut self) {
        if !self.is_done() {
            return;
        }
        self.completed_sample_count = self.dataset.samples.len();
        self.dataset.samples.clear();
        self.dataset.samples.shrink_to_fit();
    }

    pub fn status(&self) -> Status {
        let Some(index) = self.phase else {
            return Status::Idle;
        };
        if self.awaiting_ready {
            return Status::Ready {
                instruction: match self.protocol {
                    CaptureProtocol::GazeDirections => "Read the complete protocol before beginning. Keep your head still and move only your eyes toward each target. Recording does not start until you press Begin.",
                    CaptureProtocol::InitialSetup => "This single recording covers open, half, closed, gaze directions, slow close, winks and natural blinks. Follow the foreground or SteamVR guide; recording does not start until you press Begin.",
                    CaptureProtocol::PythonEyelidDataset => "This XR5 recording covers open, half-open, gentle closed, gaze directions, slow close/open, winks and natural blinks for the offline Python eyelid-model dataset. Recording does not start until you press Begin.",
                }
                .into(),
            };
        }
        if index >= self.phases.len() {
            let missing_phases = self.missing_phases();
            return Status::Done {
                samples: self.completed_sample_count.max(self.dataset.samples.len()),
                evidence_complete: missing_phases.is_empty(),
                missing_phases,
            };
        }
        let phase = self.phases[index];
        let elapsed = self.phase_elapsed().as_secs_f32().min(phase.seconds);
        let total = self.protocol.total_seconds();
        let progress = ((self.completed_seconds + elapsed) / total.max(0.001)).clamp(0.0, 1.0);
        let settling = elapsed < phase.settle_seconds;
        let paused = self.is_paused();
        let samples_in_phase = self.accepted_per_phase.get(index).copied().unwrap_or(0);
        let quota_overdue = phase.recording
            && elapsed >= phase.seconds
            && samples_in_phase < phase.minimum_samples();
        let active_elapsed = (elapsed - phase.settle_seconds).max(0.0);
        let since_sample = if samples_in_phase == 0 {
            Duration::from_secs_f32(active_elapsed)
        } else {
            self.last_sample.elapsed()
        };
        let stereo_stalled = !paused
            && !settling
            && phase.recording
            && (quota_overdue || since_sample > STALL_THRESHOLD);
        let active_progress = if phase.seconds > phase.settle_seconds {
            ((elapsed - phase.settle_seconds).max(0.0) / (phase.seconds - phase.settle_seconds))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
        Status::Running {
            phase_index: index,
            progress,
            remaining_s: (total - self.completed_seconds - elapsed).max(0.0),
            target: phase.target,
            action: phase.action.or(phase.prompt_action),
            holdout: phase.holdout,
            recording: phase.recording,
            settling,
            paused,
            samples_in_phase,
            stereo_stalled,
            awaiting_confirmation: phase.wait_for_continue,
            phase_remaining_s: if phase.wait_for_continue {
                0.0
            } else {
                (phase.seconds - elapsed).max(0.0)
            },
            pose_progress: (phase.action == Some(CaptureAction::SlowCloseOpen)).then(|| {
                let percent = (slow_target(active_progress) * 100.0).round() as u8;
                ((percent.saturating_add(5) / 10) * 10).min(100)
            }),
            instruction: instruction(phase, settling),
        }
    }

    fn missing_phases(&self) -> Vec<usize> {
        self.phases
            .iter()
            .enumerate()
            .filter_map(|(index, phase)| {
                let accepted = self.accepted_per_phase.get(index).copied().unwrap_or(0);
                (phase.recording && accepted < phase.minimum_samples()).then_some(index)
            })
            .collect()
    }

    fn evidence_metadata(&self, metadata: &str) -> String {
        const GENERATED_KEYS: [&str; 3] = [
            "capture_evidence_complete=",
            "capture_missing_phase_ids=",
            "capture_phase_sample_counts=",
        ];
        let mut out = String::new();
        for line in metadata.lines() {
            if !GENERATED_KEYS.iter().any(|key| line.starts_with(key)) {
                writeln!(out, "{line}").expect("writing to String cannot fail");
            }
        }
        let missing = self.missing_phases();
        writeln!(out, "capture_evidence_complete={}", missing.is_empty())
            .expect("writing to String cannot fail");
        writeln!(
            out,
            "capture_missing_phase_ids={}",
            missing
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        )
        .expect("writing to String cannot fail");
        let counts = self
            .phases
            .iter()
            .enumerate()
            .filter(|(_, phase)| phase.recording)
            .map(|(index, phase)| {
                format!(
                    "{index}:{}/{}",
                    self.accepted_per_phase.get(index).copied().unwrap_or(0),
                    phase.minimum_samples()
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        writeln!(out, "capture_phase_sample_counts={counts}")
            .expect("writing to String cannot fail");
        out
    }
}

pub fn total_seconds() -> f32 {
    CaptureProtocol::GazeDirections.total_seconds()
}

pub fn total_seconds_for(protocol: CaptureProtocol) -> f32 {
    protocol.total_seconds()
}

#[cfg(test)]
fn protocol() -> Vec<Phase> {
    protocol_for(CaptureProtocol::GazeDirections)
}

fn protocol_for(protocol: CaptureProtocol) -> Vec<Phase> {
    match protocol {
        CaptureProtocol::GazeDirections => gaze_protocol(),
        CaptureProtocol::InitialSetup | CaptureProtocol::PythonEyelidDataset => {
            eyelid_master_protocol()
        }
    }
}

fn gaze_protocol() -> Vec<Phase> {
    let mut phases = Vec::with_capacity(69);
    phases.push(Phase::warmup());
    append_pass(&mut phases, false, [&TRAIN_ORDER_A, &TRAIN_ORDER_B]);
    append_pass(&mut phases, true, [&HOLDOUT_ORDER_A, &HOLDOUT_ORDER_B]);
    phases
}

fn instruction_phase(
    action: CaptureAction,
    holdout: bool,
    seconds: f32,
    wait_for_continue: bool,
) -> Phase {
    Phase {
        target: None,
        action: None,
        prompt_action: Some(action),
        holdout,
        recording: false,
        wait_for_continue,
        seconds,
        settle_seconds: seconds,
    }
}

/// Put an untimed reading screen in front of a visible three-second pose
/// countdown.  The user, not the render clock, decides when the instruction has
/// been understood; recording still cannot begin immediately after the click.
fn append_ready_countdown(phases: &mut Vec<Phase>, action: CaptureAction, holdout: bool) {
    phases.push(instruction_phase(action, holdout, 0.0, true));
    phases.push(instruction_phase(
        action,
        holdout,
        POSE_COUNTDOWN_SECONDS,
        false,
    ));
}

fn static_capture(
    action: CaptureAction,
    holdout: bool,
    seconds: f32,
    settle_seconds: f32,
) -> Phase {
    Phase {
        target: None,
        action: Some(action),
        prompt_action: None,
        holdout,
        recording: true,
        wait_for_continue: false,
        seconds,
        settle_seconds,
    }
}

fn target_capture(
    target: GazeTarget,
    action: CaptureAction,
    holdout: bool,
    seconds: f32,
    settle_seconds: f32,
) -> Phase {
    Phase::capture(target, action, holdout, seconds, settle_seconds)
}

fn append_static_fit(phases: &mut Vec<Phase>, holdout: bool) {
    let blocks: &[CaptureAction] = if holdout {
        &[
            CaptureAction::RelaxedOpen,
            CaptureAction::HalfOpen,
            CaptureAction::GentleClosed,
        ]
    } else {
        &[
            CaptureAction::RelaxedOpen,
            CaptureAction::HalfOpen,
            CaptureAction::GentleClosed,
            CaptureAction::GentleClosed,
            CaptureAction::HalfOpen,
            CaptureAction::RelaxedOpen,
        ]
    };
    for &action in blocks {
        append_ready_countdown(phases, action, holdout);
        phases.push(static_capture(
            action,
            holdout,
            if holdout { 2.6 } else { 3.6 },
            if holdout { 0.4 } else { 0.5 },
        ));
    }
}

fn append_open_grid(phases: &mut Vec<Phase>, holdout: bool, orders: [&[GazeTarget; 9]; 2]) {
    append_ready_countdown(phases, CaptureAction::RelaxedOpen, holdout);
    for order in orders {
        for &target in order {
            phases.push(target_capture(
                target,
                CaptureAction::RelaxedOpen,
                holdout,
                1.35,
                0.45,
            ));
        }
    }
}

fn append_winks(phases: &mut Vec<Phase>, holdout: bool) {
    let actions: &[CaptureAction] = if holdout {
        &[CaptureAction::LeftWink, CaptureAction::RightWink]
    } else {
        &[
            CaptureAction::LeftWink,
            CaptureAction::RightWink,
            CaptureAction::LeftWink,
            CaptureAction::RightWink,
        ]
    };
    for &action in actions {
        append_ready_countdown(phases, action, holdout);
        phases.push(static_capture(
            action,
            holdout,
            if holdout { 3.2 } else { 3.4 },
            0.2,
        ));
    }
}

fn append_slow_and_blinks(phases: &mut Vec<Phase>, holdout: bool) {
    append_ready_countdown(phases, CaptureAction::SlowCloseOpen, holdout);
    phases.push(target_capture(
        GazeTarget::Center,
        CaptureAction::SlowCloseOpen,
        holdout,
        if holdout { 4.5 } else { 8.0 },
        0.4,
    ));
    append_ready_countdown(phases, CaptureAction::NaturalBlink, holdout);
    phases.push(target_capture(
        GazeTarget::Center,
        CaptureAction::NaturalBlink,
        holdout,
        if holdout { 4.5 } else { 6.5 },
        0.35,
    ));
}

fn eyelid_master_protocol() -> Vec<Phase> {
    let mut phases = Vec::with_capacity(80);
    phases.push(Phase::warmup());
    append_static_fit(&mut phases, false);
    append_open_grid(&mut phases, false, [&TRAIN_ORDER_A, &TRAIN_ORDER_B]);
    append_slow_and_blinks(&mut phases, false);
    append_winks(&mut phases, false);
    append_static_fit(&mut phases, true);
    append_open_grid(&mut phases, true, [&HOLDOUT_ORDER_A, &HOLDOUT_ORDER_B]);
    append_slow_and_blinks(&mut phases, true);
    append_winks(&mut phases, true);
    phases
}

fn append_pass(phases: &mut Vec<Phase>, holdout: bool, open_orders: [&[GazeTarget; 9]; 2]) {
    phases.push(Phase::prompt(CaptureAction::RelaxedOpen, holdout));
    phases.push(Phase::practice(
        GazeTarget::UpLeft,
        CaptureAction::RelaxedOpen,
        holdout,
        OPEN_SECONDS,
        OPEN_SETTLE_SECONDS,
    ));
    for open_order in open_orders {
        for &target in open_order {
            phases.push(Phase::capture(
                target,
                CaptureAction::RelaxedOpen,
                holdout,
                OPEN_SECONDS,
                OPEN_SETTLE_SECONDS,
            ));
        }
    }
    phases.push(Phase::prompt(CaptureAction::SlowCloseOpen, holdout));
    phases.push(Phase::practice(
        GazeTarget::Center,
        CaptureAction::SlowCloseOpen,
        holdout,
        SLOW_SECONDS,
        ACTION_SETTLE_SECONDS,
    ));
    for &target in &CARDINALS {
        phases.push(Phase::capture(
            target,
            CaptureAction::SlowCloseOpen,
            holdout,
            SLOW_SECONDS,
            ACTION_SETTLE_SECONDS,
        ));
    }
    phases.push(Phase::prompt(CaptureAction::NaturalBlink, holdout));
    phases.push(Phase::practice(
        GazeTarget::Center,
        CaptureAction::NaturalBlink,
        holdout,
        BLINK_SECONDS,
        ACTION_SETTLE_SECONDS,
    ));
    for &target in &CARDINALS {
        phases.push(Phase::capture(
            target,
            CaptureAction::NaturalBlink,
            holdout,
            BLINK_SECONDS,
            ACTION_SETTLE_SECONDS,
        ));
    }
}

fn sample_kind(action: CaptureAction, target: Option<GazeTarget>, holdout: bool) -> SampleKind {
    match (action, target, holdout) {
        (CaptureAction::RelaxedOpen, None | Some(GazeTarget::Center), false) => SampleKind::Neutral,
        (CaptureAction::RelaxedOpen, None | Some(GazeTarget::Center), true) => {
            SampleKind::HoldoutNeutral
        }
        (CaptureAction::RelaxedOpen, Some(_), false) => SampleKind::GazeSweep,
        (CaptureAction::RelaxedOpen, Some(_), true) => SampleKind::HoldoutGazeSweep,
        (CaptureAction::HalfOpen, _, false) => SampleKind::HalfOpen,
        (CaptureAction::HalfOpen, _, true) => SampleKind::HoldoutHalfOpen,
        (CaptureAction::GentleClosed, _, false) => SampleKind::Closed,
        (CaptureAction::GentleClosed, _, true) => SampleKind::HoldoutClosed,
        (CaptureAction::SlowCloseOpen, _, false) => SampleKind::SlowClose,
        (CaptureAction::SlowCloseOpen, _, true) => SampleKind::HoldoutSlowClose,
        (CaptureAction::NaturalBlink, _, false) => SampleKind::NaturalBlinks,
        (CaptureAction::NaturalBlink, _, true) => SampleKind::HoldoutNaturalBlinks,
        (CaptureAction::LeftWink, _, false) => SampleKind::LeftWink,
        (CaptureAction::LeftWink, _, true) => SampleKind::HoldoutLeftWink,
        (CaptureAction::RightWink, _, false) => SampleKind::RightWink,
        (CaptureAction::RightWink, _, true) => SampleKind::HoldoutRightWink,
    }
}

fn slow_target(progress: f32) -> f32 {
    let progress = progress.clamp(0.0, 1.0);
    if progress < 0.5 {
        1.0 - progress * 2.0
    } else {
        (progress - 0.5) * 2.0
    }
}

fn instruction(phase: Phase, settling: bool) -> String {
    if phase.target.is_none() && phase.action.is_none() {
        if let Some(action) = phase.prompt_action {
            let pass = if phase.holdout {
                "validation"
            } else {
                "training"
            };
            let direction = match action {
                CaptureAction::RelaxedOpen => "Keep both eyelids comfortably open. When a target appears, move only your eyes toward it.",
                CaptureAction::HalfOpen => "Lower both eyelids to a comfortable half-open position and hold them steady.",
                CaptureAction::GentleClosed => "Close both eyes gently without squeezing and hold them closed.",
                CaptureAction::SlowCloseOpen => "Slowly close both eyelids all the way, then slowly open them.",
                CaptureAction::NaturalBlink => "Blink naturally several times with a relaxed open pause between blinks.",
                CaptureAction::LeftWink => "Close only the left eye; keep the right eye comfortably open.",
                CaptureAction::RightWink => "Close only the right eye; keep the left eye comfortably open.",
            };
            return format!("Read before the {pass} section: {direction}");
        }
        return "Wear the headset normally. Relax both eyes and prepare for the next step.".into();
    }
    let pass = if !phase.recording {
        "practice (not recorded)"
    } else if phase.holdout {
        "validation"
    } else {
        "training"
    };
    let verb = match phase.action.expect("recording phase has an action") {
        CaptureAction::RelaxedOpen => "keep both eyelids comfortably open",
        CaptureAction::HalfOpen => "hold both eyelids halfway open",
        CaptureAction::GentleClosed => "keep both eyes gently closed without squeezing",
        CaptureAction::SlowCloseOpen => {
            if settling {
                "keep both eyelids comfortably open"
            } else {
                "slowly close both eyelids, then slowly open them"
            }
        }
        CaptureAction::NaturalBlink => {
            if settling {
                "keep both eyelids comfortably open"
            } else {
                "blink naturally now"
            }
        }
        CaptureAction::LeftWink => "hold the left eye closed and the right eye open",
        CaptureAction::RightWink => "hold the right eye closed and the left eye open",
    };
    if let Some(target) = phase.target {
        format!(
            "{pass}: look at {} ({:+.0},{:+.0}); {verb}{}.",
            target.as_str(),
            target.screen_xy()[0],
            target.screen_xy()[1],
            if settling {
                " after the target settles"
            } else {
                ""
            }
        )
    } else {
        format!(
            "{pass}: {verb}{}.",
            if settling { " after the cue" } else { "" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn first_recording_phase(capture: &GazeResidualCapture) -> usize {
        capture
            .phases
            .iter()
            .position(|phase| phase.recording)
            .unwrap()
    }

    #[test]
    fn initial_setup_is_one_bounded_mixed_rate_recording() {
        let phases = protocol_for(CaptureProtocol::InitialSetup);
        let seconds = CaptureProtocol::InitialSetup.total_seconds();
        assert!((150.0..=230.0).contains(&seconds), "{seconds}");
        assert!(phases.iter().any(|phase| {
            phase.action == Some(CaptureAction::NaturalBlink)
                && phase.sample_interval() == BLINK_SAMPLE_INTERVAL
        }));
        assert!(
            phases
                .iter()
                .filter(|phase| {
                    phase.action == Some(CaptureAction::RelaxedOpen)
                        && phase.target.is_some()
                        && !phase.holdout
                        && phase.recording
                })
                .count()
                >= 18
        );
        assert!(
            phases
                .iter()
                .filter(|phase| {
                    phase.action == Some(CaptureAction::RelaxedOpen)
                        && phase.target.is_some()
                        && phase.holdout
                        && phase.recording
                })
                .count()
                >= 18
        );
        for action in [
            CaptureAction::RelaxedOpen,
            CaptureAction::HalfOpen,
            CaptureAction::GentleClosed,
            CaptureAction::LeftWink,
            CaptureAction::RightWink,
            CaptureAction::NaturalBlink,
        ] {
            assert!(phases.iter().any(|phase| phase.action == Some(action)));
        }
    }

    #[test]
    fn python_dataset_has_master_coverage_and_an_independent_protocol_id() {
        let phases = protocol_for(CaptureProtocol::PythonEyelidDataset);
        assert_eq!(
            CaptureProtocol::PythonEyelidDataset.id(),
            "xr5_python_eyelid_dataset_v1"
        );
        assert_eq!(
            CaptureProtocol::PythonEyelidDataset.total_seconds(),
            CaptureProtocol::InitialSetup.total_seconds()
        );
        for action in [
            CaptureAction::RelaxedOpen,
            CaptureAction::HalfOpen,
            CaptureAction::GentleClosed,
            CaptureAction::SlowCloseOpen,
            CaptureAction::NaturalBlink,
            CaptureAction::LeftWink,
            CaptureAction::RightWink,
        ] {
            assert!(phases
                .iter()
                .any(|phase| phase.recording && phase.action == Some(action)));
            assert!(phases
                .iter()
                .any(|phase| phase.recording && phase.holdout && phase.action == Some(action)));
        }
    }

    #[test]
    fn protocol_is_separate_fixed_and_has_untouched_holdout() {
        let phases = protocol();
        assert_eq!(phases.len(), 69);
        assert_eq!(
            phases
                .iter()
                .filter(|phase| phase.holdout && phase.recording)
                .count(),
            28
        );
        assert_eq!(
            phases
                .iter()
                .filter(|phase| phase.action == Some(CaptureAction::RelaxedOpen))
                .filter(|phase| phase.recording)
                .count(),
            36
        );
        assert!((total_seconds() - 187.4).abs() < 0.01);
        assert_eq!(
            phases
                .iter()
                .filter(|phase| phase.prompt_action.is_some())
                .count(),
            6
        );
        assert_eq!(
            phases
                .iter()
                .filter(|phase| phase.action.is_some() && !phase.recording)
                .count(),
            6
        );
        assert_ne!(TRAIN_ORDER_A, TRAIN_ORDER_B);
        assert_ne!(HOLDOUT_ORDER_A, HOLDOUT_ORDER_B);
        for holdout in [false, true] {
            for target in GazeTarget::ALL {
                assert_eq!(
                    phases
                        .iter()
                        .filter(|phase| {
                            phase.holdout == holdout
                                && phase.recording
                                && phase.action == Some(CaptureAction::RelaxedOpen)
                                && phase.target == Some(target)
                        })
                        .count(),
                    2,
                    "{holdout:?} {target:?}"
                );
            }
        }
        for phase in phases.iter().filter(|phase| phase.recording) {
            let expected = match phase.action.unwrap() {
                CaptureAction::RelaxedOpen => 17,
                CaptureAction::SlowCloseOpen => 30,
                CaptureAction::NaturalBlink => 52,
                CaptureAction::HalfOpen
                | CaptureAction::GentleClosed
                | CaptureAction::LeftWink
                | CaptureAction::RightWink => unreachable!("not present in gaze protocol"),
            };
            assert_eq!(phase.minimum_samples(), expected);
        }
    }

    #[test]
    fn settle_frames_are_never_recordable() {
        for phase in protocol().into_iter().filter(|phase| phase.recording) {
            assert!(!phase.records((phase.settle_seconds - 0.001).max(0.0)));
            assert!(phase.records(phase.settle_seconds));
        }
    }

    #[test]
    fn practice_phases_never_record_or_require_a_quota() {
        for phase in protocol()
            .into_iter()
            .filter(|phase| phase.action.is_some() && !phase.recording)
        {
            assert!(!phase.records(phase.seconds));
            assert_eq!(phase.minimum_samples(), 0);
        }
    }

    #[test]
    fn practice_status_is_explicit_and_never_reports_a_camera_stall() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        let index = capture
            .phases
            .iter()
            .position(|phase| phase.action.is_some() && !phase.recording)
            .unwrap();
        capture.phase = Some(index);
        let phase = capture.phases[index];
        capture.entered = Instant::now()
            - Duration::from_secs_f32(phase.settle_seconds + STALL_THRESHOLD.as_secs_f32());
        assert!(!capture.wants_frame([1, 1]));
        match capture.status() {
            Status::Running {
                recording,
                stereo_stalled,
                samples_in_phase,
                ..
            } => {
                assert!(!recording);
                assert!(!stereo_stalled);
                assert_eq!(samples_in_phase, 0);
            }
            status => panic!("unexpected status: {status:?}"),
        }
    }

    #[test]
    fn action_family_prompt_waits_for_explicit_confirmation() {
        let mut capture = GazeResidualCapture::new();
        capture.start_protocol(CaptureProtocol::InitialSetup, [0, 0]);
        assert!(capture.begin());
        let index = capture
            .phases
            .iter()
            .position(|phase| phase.wait_for_continue)
            .unwrap();
        capture.phase = Some(index);
        capture.entered = Instant::now() - Duration::from_secs(30);
        capture.tick();
        assert_eq!(capture.phase, Some(index));
        assert!(capture.is_awaiting_confirmation());
        match capture.status() {
            Status::Running {
                awaiting_confirmation,
                phase_remaining_s,
                ..
            } => {
                assert!(awaiting_confirmation);
                assert_eq!(phase_remaining_s, 0.0);
            }
            status => panic!("unexpected status: {status:?}"),
        }
        assert!(capture.continue_step());
        assert_eq!(capture.phase, Some(index + 1));
    }

    #[test]
    fn automatic_pose_preview_is_a_three_second_countdown() {
        let phases = protocol_for(CaptureProtocol::InitialSetup);
        let phase = phases
            .iter()
            .find(|phase| phase.prompt_action.is_some() && !phase.wait_for_continue)
            .unwrap();
        assert_eq!(phase.seconds, POSE_COUNTDOWN_SECONDS);
        assert_eq!(phase.settle_seconds, POSE_COUNTDOWN_SECONDS);
        assert!(!phase.recording);
    }

    #[test]
    fn every_initial_setup_instruction_precedes_a_three_second_countdown() {
        let phases = protocol_for(CaptureProtocol::InitialSetup);
        let mut manual_prompts = 0;
        for pair in phases.windows(2) {
            let [prompt, countdown] = pair else {
                unreachable!()
            };
            if !prompt.wait_for_continue {
                continue;
            }
            manual_prompts += 1;
            assert_eq!(prompt.seconds, 0.0);
            assert_eq!(countdown.prompt_action, prompt.prompt_action);
            assert!(!countdown.wait_for_continue);
            assert!(!countdown.recording);
            assert_eq!(countdown.seconds, POSE_COUNTDOWN_SECONDS);
        }
        assert!(manual_prompts >= 20, "found only {manual_prompts} prompts");
    }

    #[test]
    fn slow_close_target_has_open_closed_open_shape() {
        assert!((slow_target(0.0) - 1.0).abs() < 1.0e-6);
        assert!(slow_target(0.5) < 1.0e-6);
        assert!((slow_target(1.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn successful_export_release_drops_images_but_keeps_completion_count() {
        let mut capture = GazeResidualCapture::new();
        capture.phase = Some(capture.phases.len());
        capture.dataset.samples.push(GeometrySample {
            kind: SampleKind::Neutral,
            commanded_target: Some(GazeTarget::Center),
            expected_open: None,
            phase_time_s: 0.0,
            left: vec![1],
            right: vec![2],
            left_size: (1, 1),
            right_size: (1, 1),
            brightness_affine: [[1.0, 0.0]; 2],
            native_open: [None; 2],
            native_gaze: [None; 2],
            native_pupil_pos: [None; 2],
            frame_generation: [0; 2],
            native_timestamp_us: None,
            phase_index: 1,
        });
        capture.release_exported_frames();
        assert!(capture.dataset.samples.is_empty());
        match capture.status() {
            Status::Done {
                samples,
                evidence_complete,
                missing_phases,
            } => {
                assert_eq!(samples, 1);
                assert!(!evidence_complete);
                assert!(!missing_phases.is_empty());
            }
            status => panic!("unexpected status: {status:?}"),
        }
    }

    #[test]
    fn background_export_transfer_is_zero_copy_and_restorable() {
        let mut capture = GazeResidualCapture::new();
        capture.phase = Some(capture.phases.len());
        capture.dataset.samples.push(GeometrySample {
            kind: SampleKind::Neutral,
            commanded_target: Some(GazeTarget::Center),
            expected_open: None,
            phase_time_s: 0.25,
            left: vec![7, 8],
            right: vec![9, 10],
            left_size: (2, 1),
            right_size: (2, 1),
            brightness_affine: [[1.0, 0.0]; 2],
            native_open: [None; 2],
            native_gaze: [None; 2],
            native_pupil_pos: [None; 2],
            frame_generation: [4, 5],
            native_timestamp_us: Some(6),
            phase_index: 1,
        });
        let left_ptr = capture.dataset.samples[0].left.as_ptr();

        let (dataset, metadata) = capture
            .take_export_dataset("capture_protocol=test\n")
            .unwrap();
        assert!(capture.dataset.samples.is_empty());
        assert_eq!(dataset.samples[0].left.as_ptr(), left_ptr);
        assert!(metadata.contains("capture_evidence_complete=false"));
        assert!(matches!(capture.status(), Status::Done { samples: 1, .. }));

        capture.restore_export_dataset(dataset);
        assert_eq!(capture.dataset.samples.len(), 1);
        assert_eq!(capture.dataset.samples[0].left.as_ptr(), left_ptr);
    }

    #[test]
    fn wants_frame_gates_clone_and_tracks_accepted_phase_samples() {
        let mut capture = GazeResidualCapture::new();
        capture.start([10, 20]);
        assert!(capture.begin());
        let index = first_recording_phase(&capture);
        capture.phase = Some(index);
        let phase = capture.phases[index];
        capture.entered = Instant::now()
            - Duration::from_secs_f32(phase.settle_seconds + SAMPLE_INTERVAL.as_secs_f32());
        capture.last_sample = Instant::now() - SAMPLE_INTERVAL;

        assert!(!capture.wants_frame([10, 21]));
        assert!(capture.wants_frame([11, 21]));
        let pixels = [128u8];
        assert!(capture.on_frame(
            [11, 21],
            Some(123_456),
            Some((1, 1, &pixels)),
            Some((1, 1, &pixels)),
            [[1.0, 0.0]; 2],
            [None; 2],
            [None; 2],
            [None; 2],
        ));
        assert_eq!(capture.accepted_per_phase[index], 1);
        assert_eq!(capture.dataset.samples[0].frame_generation, [11, 21]);
        assert_eq!(
            capture.dataset.samples[0].native_timestamp_us,
            Some(123_456)
        );
        assert!(!capture.wants_frame([11, 21]));
        match capture.status() {
            Status::Running {
                samples_in_phase,
                stereo_stalled,
                ..
            } => {
                assert_eq!(samples_in_phase, 1);
                assert!(!stereo_stalled);
            }
            status => panic!("unexpected status: {status:?}"),
        }
    }

    #[test]
    fn pause_freezes_phase_clock_and_sampling_until_resume() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        let index = first_recording_phase(&capture);
        capture.phase = Some(index);
        capture.entered = Instant::now() - Duration::from_millis(700);
        capture.pause();
        let frozen = capture.phase_elapsed();
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(capture.phase_elapsed(), frozen);
        assert!(!capture.wants_frame([1, 1]));
        capture.tick();
        assert_eq!(capture.phase, Some(index));
        match capture.status() {
            Status::Running { paused, .. } => assert!(paused),
            status => panic!("unexpected status: {status:?}"),
        }

        capture.resume();
        assert!(!capture.is_paused());
        assert!(capture.phase_elapsed().abs_diff(frozen) < Duration::from_millis(2));
        capture.abort();
        assert!(!capture.is_paused());
        assert!(capture.accepted_per_phase.iter().all(|count| *count == 0));
    }

    #[test]
    fn blink_evidence_drains_source_history_on_a_24_hz_ui() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        let index = capture
            .phases
            .iter()
            .position(|phase| {
                phase.recording
                    && phase.action == Some(CaptureAction::NaturalBlink)
                    && !phase.holdout
            })
            .unwrap();
        capture.phase = Some(index);
        let phase = capture.phases[index];
        let base = Instant::now();
        capture.entered = base;
        capture.last_sample = base.checked_sub(phase.sample_interval()).unwrap_or(base);

        let pixels = [128u8];
        let duration_ms = (phase.seconds * 1_000.0) as u64 + 100;
        let mut source_ms = 0u64;
        let mut generation = 1u64;
        for ui_ms in (0..=duration_ms).step_by(42) {
            while source_ms <= ui_ms {
                let at = base + Duration::from_millis(source_ms);
                capture.tick_at(at);
                capture.on_frame_at(
                    at,
                    [generation; 2],
                    None,
                    Some((1, 1, &pixels)),
                    Some((1, 1, &pixels)),
                    [[1.0, 0.0]; 2],
                    [None; 2],
                    [None; 2],
                    [None; 2],
                );
                capture.discard_through([generation; 2]);
                capture.tick_at(at);
                generation += 1;
                source_ms += 16;
            }
        }

        // 1.55 active seconds at the source cadence yields roughly 97 samples;
        // sampling only the 24 Hz UI callbacks would produce fewer than 40.
        assert!(capture.accepted_per_phase[index] >= 90);
        assert!(capture.phase.unwrap() > index);
    }

    #[test]
    fn start_and_abort_reset_session_bookkeeping() {
        let mut capture = GazeResidualCapture::new();
        capture.phase = Some(capture.phases.len());
        capture.accepted_per_phase.fill(7);
        capture.completed_sample_count = 42;
        capture.completed_seconds = total_seconds();
        capture.paused_at = Some(Instant::now());
        capture.last_error = Some("old session".into());

        capture.start([70, 90]);
        assert_eq!(capture.phase, Some(0));
        assert_eq!(capture.last_generation, [70, 90]);
        assert!(capture.accepted_per_phase.iter().all(|count| *count == 0));
        assert_eq!(capture.completed_sample_count, 0);
        assert_eq!(capture.completed_seconds, 0.0);
        assert!(!capture.is_paused());
        assert!(capture.last_error.is_none());

        capture.accepted_per_phase[1] = 3;
        capture.completed_sample_count = 5;
        capture.abort();
        assert_eq!(capture.status(), Status::Idle);
        assert!(capture.accepted_per_phase.iter().all(|count| *count == 0));
        assert_eq!(capture.completed_sample_count, 0);
        assert_eq!(capture.completed_seconds, 0.0);
    }

    #[test]
    fn active_phase_reports_stereo_stall_after_one_second_without_samples() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        let index = first_recording_phase(&capture);
        capture.phase = Some(index);
        let phase = capture.phases[index];
        capture.entered = Instant::now() - Duration::from_secs_f32(phase.settle_seconds + 1.05);
        match capture.status() {
            Status::Running {
                samples_in_phase,
                stereo_stalled,
                ..
            } => {
                assert_eq!(samples_in_phase, 0);
                assert!(stereo_stalled);
            }
            status => panic!("unexpected status: {status:?}"),
        }
    }

    #[test]
    fn recording_phase_waits_at_deadline_until_its_sample_quota_is_met() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());

        // The non-recording warmup remains purely time-driven.
        capture.entered = Instant::now() - Duration::from_secs_f32(WARMUP_SECONDS);
        capture.tick();
        assert_eq!(capture.phase, Some(1));

        // The separate explanation screen is user-driven and records nothing.
        capture.entered = Instant::now() - Duration::from_secs_f32(PROMPT_SECONDS);
        capture.tick();
        assert_eq!(capture.phase, Some(1));
        assert!(capture.continue_step());
        let index = first_recording_phase(&capture);
        assert_eq!(capture.phase, Some(index - 1));
        let practice = capture.phases[index - 1];
        assert!(!practice.recording);
        capture.entered = Instant::now() - Duration::from_secs_f32(practice.seconds);
        capture.tick();
        assert_eq!(capture.phase, Some(index));

        let phase = capture.phases[index];
        let minimum = phase.minimum_samples();
        capture.accepted_per_phase[index] = minimum - 1;
        capture.entered = Instant::now() - Duration::from_secs_f32(phase.seconds);
        capture.last_sample = Instant::now();
        capture.tick();
        assert_eq!(capture.phase, Some(index));
        match capture.status() {
            Status::Running {
                samples_in_phase,
                stereo_stalled,
                ..
            } => {
                assert_eq!(samples_in_phase, minimum - 1);
                assert!(stereo_stalled);
            }
            status => panic!("unexpected status: {status:?}"),
        }

        capture.accepted_per_phase[index] = minimum;
        capture.tick();
        assert_eq!(capture.phase, Some(index + 1));
    }

    #[test]
    fn zero_frame_completion_is_explicitly_incomplete_but_exportable() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        // Exercise the defensive completed-state handling directly. Normal runtime
        // progression now refuses to cross a recording phase with a missing quota.
        capture.phase = Some(capture.phases.len());
        assert!(capture.is_done());
        let missing = match capture.status() {
            Status::Done {
                samples,
                evidence_complete,
                missing_phases,
            } => {
                assert_eq!(samples, 0);
                assert!(!evidence_complete);
                missing_phases
            }
            status => panic!("unexpected status: {status:?}"),
        };
        assert_eq!(missing.len(), 56);
        assert_eq!(missing.first(), Some(&3));
        assert_eq!(missing.last(), Some(&(capture.phases.len() - 1)));

        let path = std::env::temp_dir().join(format!(
            "sranibro_gaze_residual_incomplete_{}.zip",
            std::process::id()
        ));
        capture
            .export_recording(&path, "capture_protocol=test\n")
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for needle in [
            b"capture_evidence_complete=false".as_slice(),
            b"capture_missing_phase_ids=3,4,5".as_slice(),
            b"capture_phase_sample_counts=3:0/17".as_slice(),
        ] {
            assert!(bytes.windows(needle.len()).any(|window| window == needle));
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn satisfying_every_phase_quota_marks_evidence_complete() {
        let mut capture = GazeResidualCapture::new();
        capture.start([0, 0]);
        assert!(capture.begin());
        for (index, phase) in capture.phases.iter().copied().enumerate() {
            capture.accepted_per_phase[index] = phase.minimum_samples();
        }
        capture.phase = Some(capture.phases.len());
        assert_eq!(
            capture.status(),
            Status::Done {
                samples: 0,
                evidence_complete: true,
                missing_phases: Vec::new(),
            }
        );
    }
}
