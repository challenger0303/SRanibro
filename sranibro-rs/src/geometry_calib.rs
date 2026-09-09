//! Guided eye-camera capture shared by geometry and photometric fitting.
//!
//! Raw stereo frames remain in memory during capture. The UI automatically exports the
//! completed dataset to a local feedback ZIP before fit/audit can consume it.
//! A separate holdout tail is never exposed to the search and is used only to decide
//! whether a candidate is safer than the geometry that was active when capture started.

use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);
const PREPARE_COUNTDOWN_SECONDS: f32 = 3.0;
// Slightly below one 60 Hz UI interval so scheduler jitter does not make the
// collector accept only every second repaint (~30 Hz).
const BLINK_SAMPLE_INTERVAL: Duration = Duration::from_millis(15);

/// Categorical target shown during a gaze/landmark capture. Coordinates are
/// normalized screen coordinates: X grows to the right and Y grows downward.
/// These are instructed UI positions, not a claim about calibrated gaze angle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GazeTarget {
    Center,
    Left,
    Right,
    Up,
    Down,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

impl GazeTarget {
    pub const ALL: [Self; 9] = [
        Self::Center,
        Self::Left,
        Self::Right,
        Self::Up,
        Self::Down,
        Self::UpLeft,
        Self::UpRight,
        Self::DownLeft,
        Self::DownRight,
    ];

    /// Stable recording/schema spelling. Do not change existing strings.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Center => "center",
            Self::Left => "left",
            Self::Right => "right",
            Self::Up => "up",
            Self::Down => "down",
            Self::UpLeft => "up_left",
            Self::UpRight => "up_right",
            Self::DownLeft => "down_left",
            Self::DownRight => "down_right",
        }
    }

    pub const fn screen_xy(self) -> [f32; 2] {
        match self {
            Self::Center => [0.0, 0.0],
            Self::Left => [-1.0, 0.0],
            Self::Right => [1.0, 0.0],
            Self::Up => [0.0, -1.0],
            Self::Down => [0.0, 1.0],
            Self::UpLeft => [-1.0, -1.0],
            Self::UpRight => [1.0, -1.0],
            Self::DownLeft => [-1.0, 1.0],
            Self::DownRight => [1.0, 1.0],
        }
    }

    pub fn from_stable_str(value: &str) -> Option<Self> {
        Some(match value {
            "center" => Self::Center,
            "left" => Self::Left,
            "right" => Self::Right,
            "up" => Self::Up,
            "down" => Self::Down,
            "up_left" => Self::UpLeft,
            "up_right" => Self::UpRight,
            "down_left" => Self::DownLeft,
            "down_right" => Self::DownRight,
            _ => return None,
        })
    }
}

impl std::str::FromStr for GazeTarget {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::from_stable_str(value).ok_or(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleKind {
    Neutral,
    GazeSweep,
    SlowClose,
    NaturalBlinks,
    Closed,
    HalfOpen,
    HoldoutNeutral,
    HoldoutGazeSweep,
    HoldoutSlowClose,
    HoldoutNaturalBlinks,
    HoldoutClosed,
    HoldoutHalfOpen,
    LeftWink,
    RightWink,
    HoldoutLeftWink,
    HoldoutRightWink,
}

impl SampleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::GazeSweep => "gaze_sweep",
            Self::SlowClose => "slow_close",
            Self::NaturalBlinks => "natural_blinks",
            Self::Closed => "closed",
            Self::HalfOpen => "half_open",
            Self::HoldoutNeutral => "holdout_neutral",
            Self::HoldoutGazeSweep => "holdout_gaze_sweep",
            Self::HoldoutSlowClose => "holdout_slow_close",
            Self::HoldoutNaturalBlinks => "holdout_natural_blinks",
            Self::HoldoutClosed => "holdout_closed",
            Self::HoldoutHalfOpen => "holdout_half_open",
            Self::LeftWink => "left_wink",
            Self::RightWink => "right_wink",
            Self::HoldoutLeftWink => "holdout_left_wink",
            Self::HoldoutRightWink => "holdout_right_wink",
        }
    }

    pub fn is_holdout(self) -> bool {
        matches!(
            self,
            Self::HoldoutNeutral
                | Self::HoldoutGazeSweep
                | Self::HoldoutSlowClose
                | Self::HoldoutNaturalBlinks
                | Self::HoldoutClosed
                | Self::HoldoutHalfOpen
                | Self::HoldoutLeftWink
                | Self::HoldoutRightWink
        )
    }

    pub fn family(self) -> SampleFamily {
        match self {
            Self::Neutral | Self::HoldoutNeutral => SampleFamily::Neutral,
            Self::GazeSweep | Self::HoldoutGazeSweep => SampleFamily::GazeSweep,
            Self::SlowClose | Self::HoldoutSlowClose => SampleFamily::SlowClose,
            Self::NaturalBlinks | Self::HoldoutNaturalBlinks => SampleFamily::NaturalBlinks,
            Self::Closed | Self::HoldoutClosed => SampleFamily::Closed,
            Self::HalfOpen | Self::HoldoutHalfOpen => SampleFamily::HalfOpen,
            // Wink fitting distinguishes these by exact SampleKind. Mapping them to
            // Closed keeps legacy geometry scorers exhaustive while the Full plan
            // continues to exclude every wink phase.
            Self::LeftWink | Self::RightWink | Self::HoldoutLeftWink | Self::HoldoutRightWink => {
                SampleFamily::Closed
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleFamily {
    Neutral,
    GazeSweep,
    SlowClose,
    NaturalBlinks,
    Closed,
    HalfOpen,
}

/// Semantic role of one recorded evidence block.
///
/// Fitters must use this label instead of relying on a capture protocol's
/// numeric phase positions. `block_id` remains opaque: it identifies repeated
/// contiguous blocks, but its numeric value has no pose meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceAction {
    RelaxedOpen,
    GazeOpen,
    HalfOpen,
    GentleClosed,
    SlowCloseOpen,
    NaturalBlink,
    LeftWink,
    RightWink,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EvidenceSplit {
    Fit,
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceLabel {
    pub action: EvidenceAction,
    pub split: EvidenceSplit,
    pub target: Option<GazeTarget>,
    pub block_id: usize,
}

#[derive(Clone, Debug)]
pub struct GeometrySample {
    pub kind: SampleKind,
    /// Explicit capture instruction when a categorical gaze target was shown.
    /// `None` means that the capture protocol did not command a gaze position.
    pub commanded_target: Option<GazeTarget>,
    /// Expected open fraction for the metronome-driven slow-close phases and the
    /// explicit held-half evidence. The normal fitter ignores the separate HalfOpen
    /// family; only the diagnostic audit treats its 0.5 label as an absolute target.
    pub expected_open: Option<f32>,
    /// Seconds on the capture source clock since this evidence block became
    /// recordable. Preparation/rest time and UI/event-loop suspension are excluded.
    pub phase_time_s: f32,
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub left_size: (u32, u32),
    pub right_size: (u32, u32),
    /// Per-frame brightness affine captured from the live pipeline.  The fitter applies
    /// the configured deterministic filters, then this affine, before trying geometries.
    pub brightness_affine: [[f32; 2]; 2],
    /// Native Tobii openness captured only as a compliance cross-check. It never enters
    /// a geometry score or candidate selection. Reported Disable is represented as 0.
    pub native_open: [Option<f32>; 2],
    /// Native per-eye gaze direction before SRanibro output correction. Invalid or
    /// unreported vectors stay absent; angles are derived deterministically by the fitter.
    pub native_gaze: [Option<[f32; 3]>; 2],
    /// Native wearable pupil position in the EyeChip's normalized, unmapped
    /// coordinate space. A missing or reported-invalid value remains absent.
    pub native_pupil_pos: [Option<[f32; 2]>; 2],
    /// Monotonic camera generations copied with this stereo sample.
    pub frame_generation: [u64; 2],
    /// Latest native device timestamp associated with the sampled native evidence.
    /// It is not assumed to be synchronized exactly to the camera exposure.
    pub native_timestamp_us: Option<u64>,
    /// PHASES index, retained so repeated OPEN/HALF/CLOSED blocks can be compared.
    pub phase_index: usize,
}

impl GeometrySample {
    /// Derive a stable semantic label from fields already present in schema-v2+
    /// recordings. This keeps old ZIPs readable while allowing new protocols to
    /// reorder phases freely.
    pub fn evidence_label(&self) -> EvidenceLabel {
        let action = match self.kind {
            SampleKind::Neutral | SampleKind::HoldoutNeutral => {
                if self.commanded_target.is_some() {
                    EvidenceAction::GazeOpen
                } else {
                    EvidenceAction::RelaxedOpen
                }
            }
            SampleKind::GazeSweep | SampleKind::HoldoutGazeSweep => EvidenceAction::GazeOpen,
            SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen => EvidenceAction::HalfOpen,
            SampleKind::Closed | SampleKind::HoldoutClosed => EvidenceAction::GentleClosed,
            SampleKind::SlowClose | SampleKind::HoldoutSlowClose => EvidenceAction::SlowCloseOpen,
            SampleKind::NaturalBlinks | SampleKind::HoldoutNaturalBlinks => {
                EvidenceAction::NaturalBlink
            }
            SampleKind::LeftWink | SampleKind::HoldoutLeftWink => EvidenceAction::LeftWink,
            SampleKind::RightWink | SampleKind::HoldoutRightWink => EvidenceAction::RightWink,
        };
        EvidenceLabel {
            action,
            split: if self.kind.is_holdout() {
                EvidenceSplit::Validation
            } else {
                EvidenceSplit::Fit
            },
            target: self.commanded_target,
            block_id: self.phase_index,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GeometryDataset {
    pub samples: Vec<GeometrySample>,
}

impl GeometryDataset {
    pub fn train_len(&self) -> usize {
        self.samples
            .iter()
            .filter(|sample| !sample.kind.is_holdout())
            .count()
    }

    pub fn holdout_len(&self) -> usize {
        self.samples
            .iter()
            .filter(|sample| sample.kind.is_holdout())
            .count()
    }
}

/// Immutable, cheaply cloneable ownership of one captured eye-image dataset.
///
/// Fitters receive this handle instead of taking a deep copy of every raw frame.
/// The capture remains immutable while one or more analyses replay the same
/// evidence, and cloning this value only increments an [`Arc`] reference count.
#[derive(Clone, Debug)]
pub struct SharedEvidence(Arc<GeometryDataset>);

impl SharedEvidence {
    pub fn new(dataset: GeometryDataset) -> Self {
        Self(Arc::new(dataset))
    }

    /// Captured samples in their original temporal order.
    pub fn samples(&self) -> &[GeometrySample] {
        &self.0.samples
    }

    pub fn train_len(&self) -> usize {
        self.0.train_len()
    }

    pub fn holdout_len(&self) -> usize {
        self.0.holdout_len()
    }

    pub fn as_dataset(&self) -> &GeometryDataset {
        &self.0
    }

    /// Recover owned evidence for legacy single-capture retry paths.
    ///
    /// A start failure normally leaves this as the only handle, so no frame data
    /// is copied. If another analysis intentionally retained a handle, cloning is
    /// the safe fallback needed to restore the legacy mutable capture container.
    pub fn into_dataset(self) -> GeometryDataset {
        Arc::try_unwrap(self.0).unwrap_or_else(|shared| (*shared).clone())
    }

    #[cfg(test)]
    fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    #[cfg(test)]
    fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl From<GeometryDataset> for SharedEvidence {
    fn from(dataset: GeometryDataset) -> Self {
        Self::new(dataset)
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Rest {
        seconds: f32,
        instruction: &'static str,
    },
    Capture {
        seconds: f32,
        kind: SampleKind,
        instruction: &'static str,
    },
    Done,
}

const PHASES: &[Phase] = &[
    Phase::Rest {
        seconds: 3.0,
        instruction: "Wear the headset normally. Look straight ahead and relax your eyelids.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Neutral,
        instruction: "OPEN 1 - look straight ahead with both eyes comfortably open and relaxed.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Next, lower both eyelids to about halfway and hold them steady.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HalfOpen,
        instruction: "HALF 1 - hold both eyelids about halfway, like relaxed sleepy eyes.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Next, close both eyes gently without squeezing.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Closed,
        instruction: "CLOSED 1 - gently close both eyes and hold; do not squeeze.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, reopen and keep your eyelids relaxed while moving only your gaze.",
    },
    Phase::Capture {
        seconds: 8.0,
        kind: SampleKind::GazeSweep,
        instruction: "GAZE SWEEP - slowly look left, right, up, and down. Do not widen or squint.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, follow three slow close/open cycles. Each half takes two seconds.",
    },
    Phase::Capture {
        seconds: 10.0,
        kind: SampleKind::SlowClose,
        instruction:
            "SLOW CLOSE - follow the on-screen target bar through three smooth close/open cycles.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, blink naturally five times with a relaxed open pause between blinks.",
    },
    Phase::Capture {
        seconds: 7.0,
        kind: SampleKind::NaturalBlinks,
        instruction: "NATURAL BLINKS - blink five times; fully relax open between blinks.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Repeat the static poses in reverse order. First, close gently.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Closed,
        instruction: "CLOSED 2 - gently close both eyes and hold; do not squeeze.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Again, hold both eyelids about halfway.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HalfOpen,
        instruction: "HALF 2 - hold both eyelids about halfway and steady.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Reopen both eyes comfortably and relax.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Neutral,
        instruction: "OPEN 2 - look straight ahead, comfortably open and relaxed.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Untouched holdout begins. Keep both eyes naturally open.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutNeutral,
        instruction: "HOLDOUT OPEN - naturally open, looking straight ahead.",
    },
    Phase::Rest {
        seconds: 1.0,
        instruction: "Hold both eyelids about halfway again.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutHalfOpen,
        instruction: "HOLDOUT HALF - hold halfway and steady.",
    },
    Phase::Rest {
        seconds: 1.0,
        instruction: "Close both eyes gently without squeezing.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutClosed,
        instruction: "HOLDOUT CLOSED - keep both eyes gently closed. Do not squeeze.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Reopen. Holdout gaze sweep next with relaxed eyelids.",
    },
    Phase::Capture {
        seconds: 5.0,
        kind: SampleKind::HoldoutGazeSweep,
        instruction: "HOLDOUT GAZE - slowly look left, right, up, and down.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "One slow close/open cycle next.",
    },
    Phase::Capture {
        seconds: 5.0,
        kind: SampleKind::HoldoutSlowClose,
        instruction: "HOLDOUT SLOW CLOSE - follow one smooth close/open cycle.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Finally, blink naturally three times.",
    },
    Phase::Capture {
        seconds: 6.0,
        kind: SampleKind::HoldoutNaturalBlinks,
        instruction: "HOLDOUT BLINKS - blink three times with open pauses.",
    },
    Phase::Done,
    // Wink phases are appended after the legacy Done sentinel so every pre-v1.6
    // canonical sample phase ID (0..29) remains byte-for-byte stable.
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, make a natural, comfortable LEFT wink. Keep the RIGHT eye open; do not force it or squeeze hard.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::LeftWink,
        instruction:
            "LEFT WINK 1 - hold a natural left wink; keep the right eye relaxed open. Do not force it or squeeze hard.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Reopen both eyes. Next, make a natural, comfortable RIGHT wink.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::RightWink,
        instruction:
            "RIGHT WINK 1 - hold a natural right wink; keep the left eye relaxed open. Do not force it or squeeze hard.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Reopen both eyes. Repeat the same comfortable LEFT wink.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::LeftWink,
        instruction:
            "LEFT WINK 2 - hold the same natural left wink. Do not force it or squeeze hard.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Reopen both eyes. Repeat the same comfortable RIGHT wink.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::RightWink,
        instruction:
            "RIGHT WINK 2 - hold the same natural right wink. Do not force it or squeeze hard.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction:
            "Untouched holdout begins. Reopen, then make the same comfortable LEFT wink.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HoldoutLeftWink,
        instruction:
            "HOLDOUT LEFT WINK - natural left wink, right relaxed open. Do not force it or squeeze hard.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Reopen both eyes, then make the same comfortable RIGHT wink.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HoldoutRightWink,
        instruction:
            "HOLDOUT RIGHT WINK - natural right wink, left relaxed open. Do not force it or squeeze hard.",
    },
    Phase::Done,
];

/// Ordered views over the canonical [`PHASES`] table.
///
/// A sample always stores its canonical `PHASES` index, never the cursor inside
/// one of these plans.  Old schema-v3 recordings and the fit/audit block grouping
/// therefore keep exactly the same phase IDs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CapturePlan {
    /// The original Safe Geometry Fit / Photometric Fit protocol.
    #[default]
    Full,
    /// Repeated open, half-open and gently-closed endpoint evidence.
    EyelidEndpoints,
    /// Relaxed-open plus gaze-direction evidence.
    GazeDirections,
    /// Repeated held left/right winks with untouched per-eye holdout.
    Winks,
    /// Relaxed-open plus deliberately slow close/open evidence.
    SlowClose,
    /// Relaxed-open plus natural-blink evidence.
    NaturalBlinks,
}

const FULL_PLAN: &[usize] = &[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30,
];
const EYELID_ENDPOINT_PLAN: &[usize] = &[
    0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30,
];
const GAZE_DIRECTION_PLAN: &[usize] = &[0, 1, 6, 7, 24, 25, 30];
const SLOW_CLOSE_PLAN: &[usize] = &[0, 1, 8, 9, 26, 27, 30];
// The blink scorer uses the open pauses inside each instructed blink block and
// the already-locked endpoints. Excluding the separate 4 s neutral-image block
// keeps this 60 Hz recording bounded to roughly 62 MB for 200x200 stereo frames.
const NATURAL_BLINK_PLAN: &[usize] = &[0, 10, 11, 28, 29, 30];
const WINK_PLAN: &[usize] = &[0, 1, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43];

impl CapturePlan {
    pub const ALL: [Self; 6] = [
        Self::Full,
        Self::EyelidEndpoints,
        Self::GazeDirections,
        Self::Winks,
        Self::SlowClose,
        Self::NaturalBlinks,
    ];

    pub const fn id(self) -> &'static str {
        match self {
            Self::Full => "full_v1",
            Self::EyelidEndpoints => "eyelid_endpoints_v1",
            Self::GazeDirections => "gaze_directions_v1",
            Self::Winks => "winks_v1",
            Self::SlowClose => "slow_close_v1",
            Self::NaturalBlinks => "natural_blinks_v1",
        }
    }

    pub const fn phase_indices(self) -> &'static [usize] {
        match self {
            Self::Full => FULL_PLAN,
            Self::EyelidEndpoints => EYELID_ENDPOINT_PLAN,
            Self::GazeDirections => GAZE_DIRECTION_PLAN,
            Self::Winks => WINK_PLAN,
            Self::SlowClose => SLOW_CLOSE_PLAN,
            Self::NaturalBlinks => NATURAL_BLINK_PLAN,
        }
    }

    pub const fn capture_hz(self) -> u32 {
        match self {
            Self::NaturalBlinks => 60,
            _ => 20,
        }
    }

    fn sample_interval(self) -> Duration {
        match self {
            Self::NaturalBlinks => BLINK_SAMPLE_INTERVAL,
            _ => SAMPLE_INTERVAL,
        }
    }

    pub fn total_seconds(self) -> f32 {
        self.phase_indices()
            .iter()
            .copied()
            .map(|index| phase_seconds(PHASES[index]))
            .sum()
    }
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Rest {
        instruction: &'static str,
        remaining_s: f32,
        overall: f32,
        /// Reading time is unlimited. Once confirmed, the existing Rest duration
        /// becomes the visible pose-preparation countdown.
        awaiting_confirmation: bool,
        next_kind: Option<SampleKind>,
    },
    Capture {
        instruction: &'static str,
        kind: SampleKind,
        remaining_s: f32,
        phase_progress: f32,
        overall: f32,
        samples: usize,
        target_open: Option<f32>,
        stereo_stalled: bool,
    },
    Done {
        train_samples: usize,
        holdout_samples: usize,
    },
}

pub struct GeometryCapture {
    /// Cursor in `plan.phase_indices()`, not a canonical PHASES index.
    phase: Option<usize>,
    plan: CapturePlan,
    entered: Instant,
    last_sample: Instant,
    last_generation: [u64; 2],
    accepted_in_phase: usize,
    quota_met_at: Option<Instant>,
    paused_at: Option<Instant>,
    /// Every Rest phase starts as an untimed instruction screen. The user must
    /// confirm it before its short preparation countdown begins.
    rest_confirmed: bool,
    dataset: GeometryDataset,
    /// Counts retained while the completed dataset is temporarily owned by the
    /// background ZIP writer. This keeps the Done UI stable without cloning frames.
    completed_train_samples: usize,
    completed_holdout_samples: usize,
    completed_seconds: f32,
    pub last_error: Option<String>,
}

impl Default for GeometryCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryCapture {
    pub fn new() -> Self {
        Self {
            phase: None,
            plan: CapturePlan::Full,
            entered: Instant::now(),
            last_sample: Instant::now() - SAMPLE_INTERVAL,
            last_generation: [0; 2],
            accepted_in_phase: 0,
            quota_met_at: None,
            paused_at: None,
            rest_confirmed: false,
            dataset: GeometryDataset::default(),
            completed_train_samples: 0,
            completed_holdout_samples: 0,
            completed_seconds: 0.0,
            last_error: None,
        }
    }

    pub fn start(&mut self, generation: [u64; 2]) {
        self.start_plan(CapturePlan::Full, generation);
    }

    pub fn start_plan(&mut self, plan: CapturePlan, generation: [u64; 2]) {
        debug_assert!(matches!(
            plan.phase_indices()
                .last()
                .and_then(|index| PHASES.get(*index)),
            Some(Phase::Done)
        ));
        self.plan = plan;
        self.phase = Some(0);
        self.entered = Instant::now();
        self.last_sample = Instant::now() - plan.sample_interval();
        self.last_generation = generation;
        self.accepted_in_phase = 0;
        self.quota_met_at = None;
        self.paused_at = None;
        self.rest_confirmed = false;
        self.dataset.samples.clear();
        self.completed_train_samples = 0;
        self.completed_holdout_samples = 0;
        self.completed_seconds = 0.0;
        self.accepted_in_phase = 0;
        self.quota_met_at = None;
        self.last_error = None;
    }

    pub fn abort(&mut self) {
        self.phase = None;
        self.paused_at = None;
        self.dataset.samples.clear();
        self.dataset.samples.shrink_to_fit();
        self.completed_train_samples = 0;
        self.completed_holdout_samples = 0;
        self.completed_seconds = 0.0;
        self.rest_confirmed = false;
        self.last_error = None;
    }

    pub fn is_running(&self) -> bool {
        matches!(self.current_phase(), Some(phase) if !matches!(phase, Phase::Done))
    }

    pub fn is_done(&self) -> bool {
        matches!(self.current_phase(), Some(Phase::Done))
    }

    pub fn is_paused(&self) -> bool {
        self.paused_at.is_some()
    }

    pub fn is_awaiting_confirmation(&self) -> bool {
        !self.is_paused()
            && !self.rest_confirmed
            && matches!(self.current_phase(), Some(Phase::Rest { .. }))
    }

    /// Start the short preparation countdown for the current instruction. No
    /// image is recorded until that countdown has completed.
    pub fn continue_step(&mut self) -> bool {
        if !self.is_awaiting_confirmation() {
            return false;
        }
        let now = Instant::now();
        self.rest_confirmed = true;
        self.entered = now;
        self.last_sample = now.checked_sub(self.plan.sample_interval()).unwrap_or(now);
        true
    }

    pub fn pause(&mut self) {
        if self.is_running() && self.paused_at.is_none() {
            self.paused_at = Some(Instant::now());
        }
    }

    pub fn resume(&mut self) {
        let Some(paused_at) = self.paused_at.take() else {
            return;
        };
        let now = Instant::now();
        let paused_for = now.saturating_duration_since(paused_at);
        self.entered = self.entered.checked_add(paused_for).unwrap_or(now);
        self.last_sample = self.last_sample.checked_add(paused_for).unwrap_or(now);
        self.quota_met_at = self
            .quota_met_at
            .and_then(|at| at.checked_add(paused_for))
            .or(self.quota_met_at);
    }

    /// Exclude a UI/event-loop hiatus discovered after the fact. The generation
    /// cursor discards frames produced while the wearer could not see the prompt.
    pub fn suspend_for(&mut self, duration: Duration, generation: [u64; 2]) {
        if !self.is_running() || self.is_paused() || duration.is_zero() {
            return;
        }
        let now = Instant::now();
        self.entered = self.entered.checked_add(duration).unwrap_or(now);
        self.last_sample = self.last_sample.checked_add(duration).unwrap_or(now);
        if let Some(at) = self.quota_met_at {
            self.quota_met_at = at.checked_add(duration).or(Some(at));
        }
        self.last_generation = generation;
    }

    pub fn last_generation(&self) -> [u64; 2] {
        self.last_generation
    }

    /// Advance the bounded-history cursor even when a valid frame belonged to a
    /// rest/settle period or was skipped by the sampling cadence.
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

    fn minimum_phase_samples(&self, phase: Phase) -> usize {
        let Phase::Capture { seconds, .. } = phase else {
            return 0;
        };
        // Retain a real temporal-rate floor while allowing modest scheduler jitter.
        // Natural-blink timing needs substantially denser evidence than image fits.
        let coverage = if self.plan == CapturePlan::NaturalBlinks {
            0.75
        } else {
            0.70
        };
        (seconds * self.plan.capture_hz() as f32 * coverage).ceil() as usize
    }

    pub fn tick(&mut self) {
        self.tick_at(Instant::now());
    }

    pub fn tick_at(&mut self, now: Instant) {
        if self.is_paused() {
            return;
        }
        // A delayed UI callback may cross a Rest boundary. Advance through rests,
        // but never cross a capture whose generation quota is still missing.
        for _ in 0..self.plan.phase_indices().len() {
            let Some(cursor) = self.phase else { return };
            let Some(phase) = self.current_phase() else {
                self.abort();
                self.last_error =
                    Some("The selected capture plan contains an invalid phase.".into());
                return;
            };
            if matches!(phase, Phase::Rest { .. }) && !self.rest_confirmed {
                return;
            }
            let seconds = match phase {
                Phase::Rest { seconds, .. } => seconds.max(PREPARE_COUNTDOWN_SECONDS),
                Phase::Capture { seconds, .. } => seconds,
                Phase::Done => return,
            };
            let deadline = self
                .entered
                .checked_add(Duration::from_secs_f32(seconds))
                .unwrap_or(now);
            if now < deadline {
                return;
            }
            if matches!(phase, Phase::Capture { .. })
                && self.accepted_in_phase < self.minimum_phase_samples(phase)
            {
                return;
            }
            let next_entered = self
                .quota_met_at
                .filter(|at| *at > deadline)
                .unwrap_or(deadline);
            self.completed_seconds += seconds;
            self.phase = Some((cursor + 1).min(self.plan.phase_indices().len() - 1));
            self.rest_confirmed = !matches!(self.current_phase(), Some(Phase::Rest { .. }));
            self.entered = next_entered;
            self.last_sample = next_entered
                .checked_sub(self.plan.sample_interval())
                .unwrap_or(next_entered);
            self.accepted_in_phase = 0;
            self.quota_met_at = None;
        }
    }

    /// True only when the plan's collector can accept this exact stereo generation.
    /// Natural-blink timing uses 60 Hz; image-fit plans remain at 20 Hz.
    pub fn wants_frame(&self, generation: [u64; 2]) -> bool {
        self.wants_frame_at(generation, Instant::now())
    }

    pub fn wants_frame_at(&self, generation: [u64; 2], captured_at: Instant) -> bool {
        if self.is_paused() {
            return false;
        }
        matches!(self.current_phase(), Some(Phase::Capture { .. }))
            && generation[0] > self.last_generation[0]
            && generation[1] > self.last_generation[1]
            && captured_at.saturating_duration_since(self.last_sample)
                >= self.plan.sample_interval()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_frame(
        &mut self,
        generation: [u64; 2],
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
        brightness_affine: [[f32; 2]; 2],
        native_open: [Option<f32>; 2],
        native_gaze: [Option<[f32; 3]>; 2],
        native_timestamp_us: Option<u64>,
    ) -> bool {
        self.on_frame_at(
            Instant::now(),
            generation,
            left,
            right,
            brightness_affine,
            native_open,
            native_gaze,
            native_timestamp_us,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_frame_at(
        &mut self,
        captured_at: Instant,
        generation: [u64; 2],
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
        brightness_affine: [[f32; 2]; 2],
        native_open: [Option<f32>; 2],
        native_gaze: [Option<[f32; 3]>; 2],
        native_timestamp_us: Option<u64>,
    ) -> bool {
        if !self.wants_frame_at(generation, captured_at) {
            return false;
        }
        let index = self
            .canonical_phase_index()
            .expect("wants_frame requires an active canonical phase");
        let Phase::Capture { seconds, kind, .. } = PHASES[index] else {
            return false;
        };
        let (Some((lw, lh, left)), Some((rw, rh, right))) = (left, right) else {
            return false;
        };
        let l_len = (lw as usize).saturating_mul(lh as usize);
        let r_len = (rw as usize).saturating_mul(rh as usize);
        if lw == 0 || lh == 0 || rw == 0 || rh == 0 || left.len() < l_len || right.len() < r_len {
            self.last_error = Some("The newest stereo eye frame is incomplete.".into());
            return false;
        }

        let elapsed = self
            .phase_elapsed_at(captured_at)
            .as_secs_f32()
            .min(seconds);
        let progress = (elapsed / seconds.max(0.001)).clamp(0.0, 1.0);
        let expected_open = match kind {
            SampleKind::SlowClose => Some(slow_target(progress, 3)),
            SampleKind::HoldoutSlowClose => Some(slow_target(progress, 1)),
            SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen => Some(0.5),
            _ => None,
        };
        self.dataset.samples.push(GeometrySample {
            kind,
            commanded_target: None,
            expected_open,
            phase_time_s: elapsed,
            left: left[..l_len].to_vec(),
            right: right[..r_len].to_vec(),
            left_size: (lw, lh),
            right_size: (rw, rh),
            brightness_affine,
            native_open,
            native_gaze,
            native_pupil_pos: [None, None],
            frame_generation: generation,
            native_timestamp_us,
            phase_index: index,
        });
        self.last_generation = generation;
        self.last_sample = self
            .last_sample
            .checked_add(self.plan.sample_interval())
            .filter(|scheduled| *scheduled <= captured_at)
            .unwrap_or(captured_at);
        self.accepted_in_phase += 1;
        if self.quota_met_at.is_none()
            && self.accepted_in_phase >= self.minimum_phase_samples(PHASES[index])
        {
            self.quota_met_at = Some(captured_at);
        }
        self.last_error = None;
        true
    }

    pub fn take_dataset(&mut self) -> Option<GeometryDataset> {
        if !self.is_done() || self.dataset.samples.is_empty() {
            return None;
        }
        self.phase = None;
        self.paused_at = None;
        self.rest_confirmed = false;
        self.completed_seconds = 0.0;
        self.completed_train_samples = 0;
        self.completed_holdout_samples = 0;
        self.accepted_in_phase = 0;
        self.quota_met_at = None;
        Some(std::mem::take(&mut self.dataset))
    }

    /// Put a completed in-memory capture back after the background fitter could not
    /// be started. This avoids making the wearer repeat the guided sequence for a
    /// transient thread/model error; no frames are written to disk.
    pub fn restore_completed_dataset(&mut self, dataset: GeometryDataset) {
        self.completed_train_samples = dataset.train_len();
        self.completed_holdout_samples = dataset.holdout_len();
        self.dataset = dataset;
        self.phase = Some(self.plan.phase_indices().len() - 1);
        self.paused_at = None;
        self.rest_confirmed = true;
        self.entered = Instant::now();
        self.completed_seconds = self.plan.total_seconds();
        self.accepted_in_phase = 0;
        self.quota_met_at = None;
        self.last_error = None;
    }

    pub fn plan(&self) -> CapturePlan {
        self.plan
    }

    fn canonical_phase_index(&self) -> Option<usize> {
        self.phase
            .and_then(|cursor| self.plan.phase_indices().get(cursor).copied())
    }

    fn current_phase(&self) -> Option<Phase> {
        self.canonical_phase_index()
            .and_then(|index| PHASES.get(index).copied())
    }

    /// Move a completed biometric dataset into a background exporter without
    /// cloning its image buffers. The Done status and counts remain visible while
    /// ownership is in flight. The caller must restore the returned dataset on
    /// both success and failure because fit/audit consumes it later.
    pub(crate) fn take_export_dataset(
        &mut self,
        metadata: &str,
    ) -> io::Result<(GeometryDataset, String)> {
        if !self.is_done() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "finish the geometry capture before exporting it",
            ));
        }
        if self.dataset.samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "the completed geometry dataset is not available",
            ));
        }
        self.completed_train_samples = self.dataset.train_len();
        self.completed_holdout_samples = self.dataset.holdout_len();
        Ok((std::mem::take(&mut self.dataset), metadata_v3(metadata)))
    }

    /// Return ownership from the background exporter. This is intentionally used
    /// after a successful save too: Safe Geometry Fit and Objective Audit still
    /// need the exact in-memory frames that were written to the ZIP.
    pub(crate) fn restore_export_dataset(&mut self, dataset: GeometryDataset) {
        if self.is_done() && self.dataset.samples.is_empty() {
            self.completed_train_samples = dataset.train_len();
            self.completed_holdout_samples = dataset.holdout_len();
            self.dataset = dataset;
        }
    }

    /// Export the exact completed dataset used by the geometry fitter. The caller owns the
    /// disclosure policy because the archive contains raw eye-camera images (biometric data).
    pub fn export_recording(&self, path: &Path, metadata: &str) -> io::Result<()> {
        if !self.is_done() || self.dataset.samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "finish the geometry capture before exporting it",
            ));
        }
        export_dataset_recording(path, &self.dataset, metadata)
    }

    pub fn status(&self) -> Status {
        let Some(phase) = self.current_phase() else {
            return Status::Idle;
        };
        let elapsed = self.phase_elapsed().as_secs_f32();
        match phase {
            Phase::Rest {
                seconds,
                instruction,
            } => {
                let seconds = seconds.max(PREPARE_COUNTDOWN_SECONDS);
                Status::Rest {
                    instruction,
                    remaining_s: if self.rest_confirmed {
                        (seconds - elapsed).max(0.0)
                    } else {
                        seconds
                    },
                    overall: self.overall_progress(if self.rest_confirmed {
                        elapsed.min(seconds)
                    } else {
                        0.0
                    }),
                    awaiting_confirmation: !self.rest_confirmed,
                    next_kind: self
                        .phase
                        .and_then(|cursor| self.plan.phase_indices().get(cursor + 1))
                        .and_then(|index| PHASES.get(*index))
                        .and_then(|phase| match phase {
                            Phase::Capture { kind, .. } => Some(*kind),
                            _ => None,
                        }),
                }
            }
            Phase::Capture {
                seconds,
                kind,
                instruction,
            } => {
                let progress = (elapsed / seconds.max(0.001)).clamp(0.0, 1.0);
                Status::Capture {
                    instruction,
                    kind,
                    remaining_s: (seconds - elapsed).max(0.0),
                    phase_progress: progress,
                    overall: self.overall_progress(elapsed.min(seconds)),
                    samples: self.dataset.samples.len(),
                    stereo_stalled: !self.is_paused()
                        && self.last_sample.elapsed() >= Duration::from_secs(1),
                    target_open: match kind {
                        SampleKind::SlowClose => Some(slow_target(progress, 3)),
                        SampleKind::HoldoutSlowClose => Some(slow_target(progress, 1)),
                        SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen => Some(0.5),
                        _ => None,
                    },
                }
            }
            Phase::Done => Status::Done {
                train_samples: self.completed_train_samples.max(self.dataset.train_len()),
                holdout_samples: self
                    .completed_holdout_samples
                    .max(self.dataset.holdout_len()),
            },
        }
    }

    fn overall_progress(&self, current_seconds: f32) -> f32 {
        ((self.completed_seconds + current_seconds) / self.plan.total_seconds()).clamp(0.0, 1.0)
    }
}

fn phase_seconds(phase: Phase) -> f32 {
    match phase {
        Phase::Rest { seconds, .. } => seconds.max(PREPARE_COUNTDOWN_SECONDS),
        Phase::Capture { seconds, .. } => seconds,
        Phase::Done => 0.0,
    }
}

pub fn total_seconds() -> f32 {
    CapturePlan::Full.total_seconds()
}

fn slow_target(progress: f32, cycles: u32) -> f32 {
    let cycle = ((progress.clamp(0.0, 0.999_999) * cycles.max(1) as f32).fract()).clamp(0.0, 1.0);
    if cycle < 0.5 {
        1.0 - cycle * 2.0
    } else {
        (cycle - 0.5) * 2.0
    }
}

fn csv_optional(value: Option<f32>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default()
}

fn csv_gaze(value: Option<[f32; 3]>) -> [String; 3] {
    value
        .filter(|value| value.iter().all(|component| component.is_finite()))
        .map(|value| value.map(|component| format!("{component:.9}")))
        .unwrap_or_default()
}

fn csv_pupil(value: Option<[f32; 2]>) -> [String; 2] {
    value
        .filter(|value| value.iter().all(|component| component.is_finite()))
        .map(|value| value.map(|component| format!("{component:.9}")))
        .unwrap_or_default()
}

fn metadata_v3(metadata: &str) -> String {
    let mut body = String::new();
    let mut has_schema = false;
    let mut has_protocol = false;
    let mut has_pupil_space = false;
    for line in metadata.lines() {
        if line.starts_with("schema_version=") {
            writeln!(body, "schema_version=3").expect("writing to String cannot fail");
            has_schema = true;
        } else {
            writeln!(body, "{line}").expect("writing to String cannot fail");
            has_protocol |= line.starts_with("capture_protocol=");
            has_pupil_space |= line.starts_with("pupil_pos_space=");
        }
    }

    if !has_schema {
        body.insert_str(0, "schema_version=3\n");
    }
    if !has_protocol {
        body.push_str("capture_protocol=safe_geometry_fit_v1\n");
    }
    if !has_pupil_space {
        body.push_str("pupil_pos_space=tobii_wearable_normalized_unmapped\n");
    }
    body
}

fn encode_gray_png(width: u32, height: u32, pixels: &[u8]) -> io::Result<Vec<u8>> {
    let need = (width as usize).saturating_mul(height as usize);
    if width == 0 || height == 0 || pixels.len() < need {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "geometry recording contains an incomplete image",
        ));
    }
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|error| io::Error::other(error.to_string()))?;
        writer
            .write_image_data(&pixels[..need])
            .map_err(|error| io::Error::other(error.to_string()))?;
        writer
            .finish()
            .map_err(|error| io::Error::other(error.to_string()))?;
    }
    Ok(encoded)
}

pub(crate) fn export_dataset_recording(
    path: &Path,
    dataset: &GeometryDataset,
    metadata: &str,
) -> io::Result<()> {
    let mut csv = String::from(
        "index,kind,holdout,phase_index,phase_time_s,expected_open,left_file,right_file,left_width,left_height,right_width,right_height,left_gain,left_bias,right_gain,right_bias,native_open_left,native_open_right,gaze_l_x,gaze_l_y,gaze_l_z,gaze_r_x,gaze_r_y,gaze_r_z,commanded_target,pupil_l_x,pupil_l_y,pupil_r_x,pupil_r_y,frame_generation_l,frame_generation_r,native_timestamp_us\n",
    );
    for (index, sample) in dataset.samples.iter().enumerate() {
        let gaze_l = csv_gaze(sample.native_gaze[0]);
        let gaze_r = csv_gaze(sample.native_gaze[1]);
        let pupil_l = csv_pupil(sample.native_pupil_pos[0]);
        let pupil_r = csv_pupil(sample.native_pupil_pos[1]);
        let target = sample
            .commanded_target
            .map(GazeTarget::as_str)
            .unwrap_or_default();
        writeln!(
            csv,
            "{index},{},{},{},{:.6},{},frames/{index:06}_left.png,frames/{index:06}_right.png,{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            sample.kind.as_str(),
            sample.kind.is_holdout(),
            sample.phase_index,
            sample.phase_time_s,
            csv_optional(sample.expected_open),
            sample.left_size.0,
            sample.left_size.1,
            sample.right_size.0,
            sample.right_size.1,
            sample.brightness_affine[0][0],
            sample.brightness_affine[0][1],
            sample.brightness_affine[1][0],
            sample.brightness_affine[1][1],
            csv_optional(sample.native_open[0]),
            csv_optional(sample.native_open[1]),
            gaze_l[0],
            gaze_l[1],
            gaze_l[2],
            gaze_r[0],
            gaze_r[1],
            gaze_r[2],
            target,
            pupil_l[0],
            pupil_l[1],
            pupil_r[0],
            pupil_r[1],
            sample.frame_generation[0],
            sample.frame_generation[1],
            sample
                .native_timestamp_us
                .map(|value| value.to_string())
                .unwrap_or_default(),
        )
        .map_err(|_| io::Error::other("failed to build geometry recording manifest"))?;
    }

    const README: &str = "SRanibro eye-camera calibration recording\n\nThis archive contains raw grayscale eye-camera images and therefore biometric data.\nShare it only when you intend to provide debugging or model-fit feedback.\n\nframes/*.png are the exact mapped stereo frames collected by the capture protocol before crop, rotation, fitted photometric correction, and model inference.\nsamples.csv contains pose labels, holdout membership, timing, dimensions, per-frame adaptive-brightness affine, Tobii native openness/gaze/pupil position when reported, and an optional categorical target.\nmetadata.txt identifies the capture protocol and contains non-secret runtime settings plus a pseudonymous unit identifier; it intentionally omits asset paths and the real device serial.\n";
    let metadata = metadata_v3(metadata);
    let mut zip = crate::diagnostics::StoredZipWriter::create(path)?;
    zip.add("README.txt", README.as_bytes())?;
    zip.add("metadata.txt", metadata.as_bytes())?;
    zip.add("samples.csv", csv.as_bytes())?;
    for (index, sample) in dataset.samples.iter().enumerate() {
        let left = encode_gray_png(sample.left_size.0, sample.left_size.1, &sample.left)?;
        zip.add(&format!("frames/{index:06}_left.png"), &left)?;
        let right = encode_gray_png(sample.right_size.0, sample.right_size.1, &sample.right)?;
        zip.add(&format!("frames/{index:06}_right.png"), &right)?;
    }
    zip.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_evidence_clone_reuses_the_same_dataset_allocation() {
        let evidence = SharedEvidence::new(GeometryDataset::default());
        let clone = evidence.clone();

        assert!(evidence.ptr_eq(&clone));
        assert_eq!(evidence.strong_count(), 2);
        assert_eq!(evidence.samples().as_ptr(), clone.samples().as_ptr());

        drop(clone);
        let recovered = evidence.into_dataset();
        assert!(recovered.samples.is_empty());
    }

    #[test]
    fn holdout_is_separate_from_search_families() {
        assert!(!SampleKind::Neutral.is_holdout());
        assert!(!SampleKind::HalfOpen.is_holdout());
        assert!(SampleKind::HoldoutNeutral.is_holdout());
        assert!(SampleKind::HoldoutHalfOpen.is_holdout());
        assert_eq!(SampleKind::Neutral.family(), SampleFamily::Neutral);
        assert_eq!(SampleKind::HoldoutNeutral.family(), SampleFamily::Neutral);
        assert_eq!(SampleKind::HalfOpen.family(), SampleFamily::HalfOpen);
        assert_eq!(SampleKind::HoldoutHalfOpen.family(), SampleFamily::HalfOpen);
    }

    #[test]
    fn slow_target_is_a_scale_free_close_open_metronome() {
        assert!((slow_target(0.0, 1) - 1.0).abs() < 1e-6);
        assert!(slow_target(0.5, 1) < 0.01);
        assert!(slow_target(0.999, 1) > 0.99);
        assert!(slow_target(1.0 / 6.0, 3) < 0.01);
        assert!(slow_target(1.0 / 3.0 - 0.001, 3) > 0.98);
    }

    #[test]
    fn protocol_contains_a_real_holdout_and_no_squeeze_or_wide() {
        let mut train = Vec::new();
        let mut holdout = Vec::new();
        for phase in PHASES {
            if let Phase::Capture { kind, .. } = phase {
                if kind.is_holdout() {
                    holdout.push(*kind);
                } else {
                    train.push(*kind);
                }
            }
        }
        assert_eq!(train.len(), 13);
        assert_eq!(holdout.len(), 8);
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::Neutral)
                .count(),
            2
        );
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::HalfOpen)
                .count(),
            2
        );
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::Closed)
                .count(),
            2
        );
        assert!(holdout.contains(&SampleKind::HoldoutHalfOpen));
        assert!((118.0..=120.0).contains(&total_seconds()));
    }

    #[test]
    fn capture_plans_preserve_canonical_phase_ids() {
        assert_eq!(
            CapturePlan::Full.phase_indices(),
            &(0..=30).collect::<Vec<_>>()
        );
        for plan in CapturePlan::ALL {
            let indices = plan.phase_indices();
            assert!(!indices.is_empty(), "{plan:?}");
            assert!(matches!(PHASES[*indices.last().unwrap()], Phase::Done));
            assert!(indices[..indices.len() - 1]
                .iter()
                .all(|index| *index < PHASES.len() - 1));
            assert!(plan.total_seconds() > 0.0);
        }
        assert_eq!(CapturePlan::EyelidEndpoints.total_seconds(), 60.0);
        assert_eq!(CapturePlan::GazeDirections.total_seconds(), 26.0);
        assert_eq!(CapturePlan::Winks.total_seconds(), 49.0);
        assert_eq!(CapturePlan::SlowClose.total_seconds(), 28.0);
        assert_eq!(CapturePlan::NaturalBlinks.total_seconds(), 22.0);
    }

    fn drive_source_history_at_24_hz_ui(plan: CapturePlan, duration_ms: u64) -> GeometryCapture {
        let mut capture = GeometryCapture::new();
        capture.start_plan(plan, [0, 0]);
        let base = Instant::now();
        capture.entered = base;
        capture.last_sample = base.checked_sub(plan.sample_interval()).unwrap_or(base);

        let pixels = [0u8; 4];
        let mut source_ms = 0u64;
        let mut generation = 1u64;
        // A 24 Hz present loop drains every coherent ~60 Hz source sample that
        // arrived since its previous callback.
        for ui_ms in (0..=duration_ms).step_by(42) {
            while source_ms <= ui_ms {
                let at = base + Duration::from_millis(source_ms);
                capture.tick_at(at);
                // The camera-clock test is not a reading-speed test: simulate an
                // immediate Continue click whenever an instruction appears.
                if capture.is_awaiting_confirmation() {
                    capture.rest_confirmed = true;
                    capture.entered = at;
                    capture.last_sample = at.checked_sub(plan.sample_interval()).unwrap_or(at);
                }
                capture.on_frame_at(
                    at,
                    [generation; 2],
                    Some((2, 2, &pixels)),
                    Some((2, 2, &pixels)),
                    [[1.0, 0.0]; 2],
                    [None; 2],
                    [None; 2],
                    None,
                );
                capture.tick_at(at);
                capture.discard_through([generation; 2]);
                generation += 1;
                source_ms += 16;
            }
            capture.tick_at(base + Duration::from_millis(ui_ms));
        }
        capture
    }

    #[test]
    fn twenty_hz_capture_is_camera_clocked_on_a_24_hz_monitor() {
        let capture = drive_source_history_at_24_hz_ui(CapturePlan::EyelidEndpoints, 8_000);
        // Rest 3 s + first 4 s OPEN block. A reset-to-now UI clock would record
        // only ~48 frames here; the deadline accumulator stays near 20 Hz.
        assert!((76..=82).contains(&capture.dataset.samples.len()));
        assert_eq!(capture.canonical_phase_index(), Some(2));
    }

    #[test]
    fn blink_capture_keeps_source_rate_above_fifty_hz_on_a_24_hz_monitor() {
        let capture = drive_source_history_at_24_hz_ui(CapturePlan::NaturalBlinks, 10_100);
        // Two three-second preparation phases consume 6 s; the remaining 4.1 s is sampled at
        // the 16 ms source cadence, not at the 24 Hz presentation cadence.
        assert!((250..=265).contains(&capture.dataset.samples.len()));
        assert_eq!(capture.canonical_phase_index(), Some(11));
    }

    #[test]
    fn rest_waits_for_continue_then_runs_a_three_second_countdown() {
        let mut capture = GeometryCapture::new();
        capture.start_plan(CapturePlan::EyelidEndpoints, [0, 0]);
        let base = Instant::now();
        capture.entered = base;

        capture.tick_at(base + Duration::from_secs(60));
        assert_eq!(capture.canonical_phase_index(), Some(0));
        assert!(capture.is_awaiting_confirmation());
        match capture.status() {
            Status::Rest {
                awaiting_confirmation,
                remaining_s,
                ..
            } => {
                assert!(awaiting_confirmation);
                assert_eq!(remaining_s, PREPARE_COUNTDOWN_SECONDS);
            }
            status => panic!("unexpected status: {status:?}"),
        }

        assert!(capture.continue_step());
        capture.entered = base;
        capture.tick_at(base + Duration::from_millis(2_999));
        assert_eq!(capture.canonical_phase_index(), Some(0));
        capture.tick_at(base + Duration::from_secs(3));
        assert_eq!(capture.canonical_phase_index(), Some(1));
        assert!(!capture.is_awaiting_confirmation());
    }

    #[test]
    fn native_ui_hiccup_does_not_consume_the_visible_phase() {
        let mut capture = GeometryCapture::new();
        capture.start_plan(CapturePlan::EyelidEndpoints, [1, 1]);
        let base = Instant::now();
        capture.phase = Some(1); // OPEN capture
        capture.entered = base;
        capture.suspend_for(Duration::from_secs(2), [99, 99]);
        assert!(
            capture.phase_elapsed_at(base + Duration::from_millis(2_100))
                < Duration::from_millis(110)
        );
        assert_eq!(capture.last_generation(), [99, 99]);
    }

    #[test]
    fn feedback_zip_contains_stereo_pngs_labels_and_metadata() {
        let path = std::env::temp_dir().join(format!(
            "sranibro_geometry_recording_test_{}.zip",
            std::process::id()
        ));
        let dataset = GeometryDataset {
            samples: vec![GeometrySample {
                kind: SampleKind::HalfOpen,
                commanded_target: Some(GazeTarget::UpRight),
                expected_open: Some(0.5),
                phase_time_s: 1.25,
                left: vec![0, 64, 128, 255],
                right: vec![255, 128, 64, 0],
                left_size: (2, 2),
                right_size: (2, 2),
                brightness_affine: [[1.0, 0.0], [0.9, 0.1]],
                native_open: [Some(0.52), None],
                native_gaze: [Some([0.1, -0.2, 0.97]), None],
                native_pupil_pos: [Some([0.4, 0.6]), None],
                frame_generation: [17, 19],
                native_timestamp_us: Some(123_456),
                phase_index: 3,
            }],
        };
        export_dataset_recording(&path, &dataset, "version=test\nunit_id=unit-test\n").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for needle in [
            b"samples.csv".as_slice(),
            b"frames/000000_left.png".as_slice(),
            b"frames/000000_right.png".as_slice(),
            b"half_open".as_slice(),
            b"unit-test".as_slice(),
            b"gaze_l_x,gaze_l_y,gaze_l_z,gaze_r_x,gaze_r_y,gaze_r_z".as_slice(),
            b"0.100000001,-0.200000003,0.970000029,,,".as_slice(),
            b"commanded_target,pupil_l_x,pupil_l_y,pupil_r_x,pupil_r_y".as_slice(),
            b"up_right,0.400000006,0.600000024,,".as_slice(),
            b"frame_generation_l,frame_generation_r,native_timestamp_us".as_slice(),
            b",17,19,123456".as_slice(),
            b"schema_version=3".as_slice(),
            b"capture_protocol=safe_geometry_fit_v1".as_slice(),
            b"pupil_pos_space=tobii_wearable_normalized_unmapped".as_slice(),
            b"\x89PNG\r\n\x1a\n".as_slice(),
        ] {
            assert!(
                bytes.windows(needle.len()).any(|window| window == needle),
                "missing {:?}",
                String::from_utf8_lossy(needle)
            );
        }
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn gaze_targets_have_stable_strings_and_screen_coordinates() {
        for target in GazeTarget::ALL {
            assert_eq!(GazeTarget::from_stable_str(target.as_str()), Some(target));
            assert_eq!(target.as_str().parse::<GazeTarget>(), Ok(target));
        }
        assert_eq!(GazeTarget::Left.screen_xy(), [-1.0, 0.0]);
        assert_eq!(GazeTarget::Up.screen_xy(), [0.0, -1.0]);
        assert_eq!(GazeTarget::DownRight.screen_xy(), [1.0, 1.0]);
        assert_eq!(GazeTarget::from_stable_str(""), None);
    }

    #[test]
    fn schema_v3_metadata_replaces_old_version_and_preserves_explicit_protocol() {
        let metadata = metadata_v3(
            "schema_version=2\ncapture_protocol=xr5_landmark_residual_audit_v1\npupil_pos_space=custom_test_space\n",
        );
        assert!(metadata.contains("schema_version=3\n"));
        assert!(!metadata.contains("schema_version=2"));
        assert!(metadata.contains("capture_protocol=xr5_landmark_residual_audit_v1\n"));
        assert!(metadata.contains("pupil_pos_space=custom_test_space\n"));
        assert_eq!(metadata.matches("capture_protocol=").count(), 1);
        assert_eq!(metadata.matches("pupil_pos_space=").count(), 1);
    }

    #[test]
    fn background_export_transfer_keeps_done_counts_and_restores_fit_dataset() {
        let train = GeometrySample {
            kind: SampleKind::Neutral,
            commanded_target: None,
            expected_open: None,
            phase_time_s: 0.25,
            left: vec![1, 2, 3, 4],
            right: vec![4, 3, 2, 1],
            left_size: (2, 2),
            right_size: (2, 2),
            brightness_affine: [[1.0, 0.0], [1.0, 0.0]],
            native_open: [None; 2],
            native_gaze: [None; 2],
            native_pupil_pos: [None; 2],
            frame_generation: [10, 11],
            native_timestamp_us: None,
            phase_index: 1,
        };
        let mut holdout = train.clone();
        holdout.kind = SampleKind::HoldoutNeutral;
        holdout.phase_index = 29;

        let mut capture = GeometryCapture::new();
        capture.phase = Some(CapturePlan::Full.phase_indices().len() - 1);
        capture.dataset.samples = vec![train, holdout];
        let (dataset, metadata) = capture
            .take_export_dataset("schema_version=2\nunit_id=test\n")
            .unwrap();
        assert!(capture.dataset.samples.is_empty());
        assert!(capture.take_dataset().is_none());
        assert!(metadata.contains("schema_version=3\n"));
        match capture.status() {
            Status::Done {
                train_samples,
                holdout_samples,
            } => assert_eq!((train_samples, holdout_samples), (1, 1)),
            status => panic!("unexpected status while export owns dataset: {status:?}"),
        }

        capture.restore_export_dataset(dataset);
        let restored = capture.take_dataset().expect("fit dataset restored");
        assert_eq!(restored.train_len(), 1);
        assert_eq!(restored.holdout_len(), 1);
        assert!(matches!(capture.status(), Status::Idle));
    }
}
