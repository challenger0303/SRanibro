//! Guided Dream Air / XR5 onboarding and live quality evaluation.
//!
//! The fixed XR5 warp is only a starting point.  This module deliberately does not
//! auto-edit geometry: it measures the user's real relaxed/closed/wide ranges, seeds
//! the existing post-processor, and reports when the source signal is too weak.  All
//! calculations are UI-independent so recorded sessions can exercise them in tests.

use crate::core::eye_state::{CalibSnapshot, CalibStore};
use crate::core::types::{EyeResult, GazeSample};

const PRESENT_GATE: f32 = 0.05;
/// The dual-eye SRanipal network takes ~24 ms per stereo inference on the
/// reference Dream Air machine (about 42/s).  Requiring 45/s made a healthy,
/// stable stream impossible to pass.  35/s preserves headroom for short load
/// spikes while still rejecting a genuinely stalled/overloaded model.
const MIN_ML_RATE: f32 = 35.0;
const SETTLE_SAMPLES: u32 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuidedStep {
    Relaxed,
    SlowBlinks,
    Closed,
    Wide,
    Squeeze,
    LeftWink,
    RightWink,
    Complete,
}

impl GuidedStep {
    pub const ORDER: [GuidedStep; 7] = [
        GuidedStep::Relaxed,
        GuidedStep::SlowBlinks,
        GuidedStep::Closed,
        GuidedStep::Wide,
        GuidedStep::Squeeze,
        GuidedStep::LeftWink,
        GuidedStep::RightWink,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Self::Relaxed => "Relaxed open",
            Self::SlowBlinks => "Slow blinks",
            Self::Closed => "Eyes closed",
            Self::Wide => "Eyes wide",
            Self::Squeeze => "Squeeze",
            Self::LeftWink => "Left wink",
            Self::RightWink => "Right wink",
            Self::Complete => "Complete",
        }
    }

    pub fn instruction(self) -> &'static str {
        match self {
            Self::Relaxed => "Look straight ahead and keep both eyes naturally open.",
            Self::SlowBlinks => "Blink slowly three times, fully open between blinks.",
            Self::Closed => "Close both eyes gently and hold.",
            Self::Wide => "Open both eyes as wide as is comfortable and hold.",
            Self::Squeeze => "Close both eyes firmly and hold.",
            Self::LeftWink => "Wink only your LEFT eye.",
            Self::RightWink => "Wink only your RIGHT eye.",
            Self::Complete => "Review the measurements, then apply or discard them.",
        }
    }

    pub fn sample_budget(self) -> u32 {
        // EyeNet runs at ~60 Hz.  The whole flow is about 20 seconds; the first
        // half-second of every gesture is ignored so instruction reaction time does
        // not contaminate the measurement.
        match self {
            Self::Relaxed => 180,
            Self::SlowBlinks => 330,
            Self::Closed => 150,
            Self::Wide => 180,
            Self::Squeeze => 150,
            Self::LeftWink | Self::RightWink => 150,
            Self::Complete => 0,
        }
    }

    fn index(self) -> Option<usize> {
        Self::ORDER.iter().position(|s| *s == self)
    }

    fn next(self) -> Self {
        match self.index() {
            Some(i) if i + 1 < Self::ORDER.len() => Self::ORDER[i + 1],
            _ => Self::Complete,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct PhaseData {
    raw: [Vec<f32>; 2],
    presence_total: [u32; 2],
    presence_good: [u32; 2],
    native_total: [u32; 2],
    native_enabled: [u32; 2],
    pupil_x: [Vec<f32>; 2],
    pupil_y: [Vec<f32>; 2],
}

impl PhaseData {
    fn push(&mut self, ml: [[f32; 5]; 2], gaze: &GazeSample) {
        for i in 0..2 {
            let raw = ml[i][1];
            if raw.is_finite() {
                self.raw[i].push(raw);
            }
            self.presence_total[i] += 1;
            if ml[i][0].is_finite() && ml[i][0] > PRESENT_GATE {
                self.presence_good[i] += 1;
            }
            let eye = if i == 0 { &gaze.left } else { &gaze.right };
            if eye.openness_reported {
                self.native_total[i] += 1;
                if eye.openness_valid {
                    self.native_enabled[i] += 1;
                }
            }
            if eye.pupil_pos_valid && eye.pupil_pos[0].is_finite() && eye.pupil_pos[1].is_finite() {
                self.pupil_x[i].push(eye.pupil_pos[0]);
                self.pupil_y[i].push(eye.pupil_pos[1]);
            }
        }
    }
}

pub struct GuidedCalibration {
    step: GuidedStep,
    step_samples: u32,
    phases: [PhaseData; 7],
}

impl Default for GuidedCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl GuidedCalibration {
    pub fn new() -> Self {
        Self {
            step: GuidedStep::Relaxed,
            step_samples: 0,
            phases: std::array::from_fn(|_| PhaseData::default()),
        }
    }

    pub fn step(&self) -> GuidedStep {
        self.step
    }

    pub fn progress(&self) -> f32 {
        if self.step == GuidedStep::Complete {
            return 1.0;
        }
        let done: u32 = GuidedStep::ORDER
            .iter()
            .take(self.step.index().unwrap_or(0))
            .map(|s| s.sample_budget())
            .sum();
        let total: u32 = GuidedStep::ORDER.iter().map(|s| s.sample_budget()).sum();
        (done + self.step_samples).min(total) as f32 / total.max(1) as f32
    }

    pub fn step_progress(&self) -> f32 {
        let budget = self.step.sample_budget();
        if budget == 0 {
            1.0
        } else {
            self.step_samples as f32 / budget as f32
        }
    }

    /// Add exactly one new EyeNet inference.  The caller must de-duplicate on the
    /// pipeline ML generation counter; UI repaint cadence is not a sampling clock.
    pub fn push(&mut self, ml: [[f32; 5]; 2], gaze: &GazeSample) {
        let Some(index) = self.step.index() else {
            return;
        };
        self.step_samples += 1;
        if self.step_samples > SETTLE_SAMPLES {
            self.phases[index].push(ml, gaze);
        }
        if self.step_samples >= self.step.sample_budget() {
            self.step = self.step.next();
            self.step_samples = 0;
        }
    }

    pub fn is_complete(&self) -> bool {
        self.step == GuidedStep::Complete
    }

    pub fn report(&self) -> Option<CalibrationReport> {
        if !self.is_complete() {
            return None;
        }
        let relaxed = &self.phases[GuidedStep::Relaxed.index().unwrap()];
        let closed = &self.phases[GuidedStep::Closed.index().unwrap()];
        let wide = &self.phases[GuidedStep::Wide.index().unwrap()];
        let left_wink = &self.phases[GuidedStep::LeftWink.index().unwrap()];
        let right_wink = &self.phases[GuidedStep::RightWink.index().unwrap()];

        let mut baseline = [0.6; 2];
        let mut blink_depth = [0.2; 2];
        let mut wide_span = [0.0; 2];
        let mut wide_snr = [0.0; 2];
        let mut wide_supported = [false; 2];
        let mut presence_rate = [0.0; 2];
        let mut native_closed_rate = [0.0; 2];
        let mut pupil_center = [[0.5; 2]; 2];
        let mut pupil_center_valid = [false; 2];
        for i in 0..2 {
            baseline[i] = median(&relaxed.raw[i]).unwrap_or(0.6);
            let closed_raw = percentile(&closed.raw[i], 0.35).unwrap_or(baseline[i] - 0.2);
            blink_depth[i] = (baseline[i] - closed_raw).clamp(0.05, 0.40);
            let wide_raw = median(&wide.raw[i]).unwrap_or(baseline[i]);
            wide_span[i] = (wide_raw - baseline[i]).max(0.0);
            let noise = stddev(&relaxed.raw[i]).unwrap_or(0.0).max(0.005);
            wide_snr[i] = wide_span[i] / noise;
            wide_supported[i] = wide_span[i] >= 0.015 && wide_snr[i] >= 2.0;
            presence_rate[i] = ratio(relaxed.presence_good[i], relaxed.presence_total[i]);
            native_closed_rate[i] = if closed.native_total[i] == 0 {
                0.0
            } else {
                1.0 - ratio(closed.native_enabled[i], closed.native_total[i])
            };
            if let (Some(x), Some(y)) = (median(&relaxed.pupil_x[i]), median(&relaxed.pupil_y[i])) {
                pupil_center[i] = [x, y];
                pupil_center_valid[i] = true;
            }
        }

        let left_drop = [
            baseline[0] - median(&left_wink.raw[0]).unwrap_or(baseline[0]),
            baseline[1] - median(&left_wink.raw[1]).unwrap_or(baseline[1]),
        ];
        let right_drop = [
            baseline[0] - median(&right_wink.raw[0]).unwrap_or(baseline[0]),
            baseline[1] - median(&right_wink.raw[1]).unwrap_or(baseline[1]),
        ];
        let mapping =
            if left_drop[0] > left_drop[1] + 0.025 && right_drop[1] > right_drop[0] + 0.025 {
                MappingVerdict::Correct
            } else if left_drop[1] > left_drop[0] + 0.025 && right_drop[0] > right_drop[1] + 0.025 {
                MappingVerdict::Swapped
            } else {
                MappingVerdict::Ambiguous
            };

        let separation_score = blink_depth
            .iter()
            .map(|d| (d / 0.15).clamp(0.0, 1.0))
            .sum::<f32>()
            / 2.0;
        let presence_score = (presence_rate[0] + presence_rate[1]) * 0.5;
        let native_score = (native_closed_rate[0] + native_closed_rate[1]) * 0.5;
        let quality_score =
            100.0 * (0.55 * separation_score + 0.25 * presence_score + 0.20 * native_score);
        let passed = blink_depth.iter().all(|d| *d >= 0.06)
            && presence_rate.iter().all(|p| *p >= 0.90)
            && native_closed_rate.iter().all(|p| *p >= 0.50)
            && mapping != MappingVerdict::Swapped;

        Some(CalibrationReport {
            baseline,
            blink_depth,
            wide_span,
            wide_snr,
            wide_supported,
            presence_rate,
            native_closed_rate,
            pupil_center,
            pupil_center_valid,
            mapping,
            quality_score,
            passed,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingVerdict {
    Correct,
    Swapped,
    Ambiguous,
}

#[derive(Clone, Copy, Debug)]
pub struct CalibrationReport {
    pub baseline: [f32; 2],
    pub blink_depth: [f32; 2],
    pub wide_span: [f32; 2],
    pub wide_snr: [f32; 2],
    pub wide_supported: [bool; 2],
    pub presence_rate: [f32; 2],
    pub native_closed_rate: [f32; 2],
    pub pupil_center: [[f32; 2]; 2],
    pub pupil_center_valid: [bool; 2],
    pub mapping: MappingVerdict,
    pub quality_score: f32,
    pub passed: bool,
}

impl CalibrationReport {
    pub fn calibration_store(&self, previous: Option<CalibStore>) -> CalibStore {
        let prev = previous.unwrap_or_else(default_calib_store);
        let frame_count = prev.left.frame_count.max(prev.right.frame_count).max(200);
        let make = |i: usize, old: CalibSnapshot| CalibSnapshot {
            baseline: self.baseline[i],
            baseline_n: 100,
            frame_count,
            blink_depth: self.blink_depth[i],
            mid_anchor: old.mid_anchor,
            learned_once: true,
        };
        CalibStore {
            left: make(0, prev.left),
            right: make(1, prev.right),
        }
    }
}

fn default_calib_store() -> CalibStore {
    let snap = CalibSnapshot {
        baseline: 0.6,
        baseline_n: 0,
        frame_count: 0,
        blink_depth: 0.2,
        mid_anchor: 0.5,
        learned_once: false,
    };
    CalibStore {
        left: snap,
        right: snap,
    }
}

#[derive(Clone, Debug)]
pub struct PreflightInput {
    pub rates: [f32; 5],
    pub frame_dims: [Option<(u32, u32)>; 2],
    pub ml_loaded: bool,
    pub ml: [[f32; 5]; 2],
    pub gaze: GazeSample,
}

#[derive(Clone, Debug)]
pub struct PreflightCheck {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct PreflightReport {
    pub ready: bool,
    pub checks: Vec<PreflightCheck>,
}

pub fn evaluate_preflight(input: &PreflightInput) -> PreflightReport {
    let mut checks = Vec::new();
    let cam_rate = input.rates[0].min(input.rates[1]);
    checks.push(PreflightCheck {
        name: "Eye cameras",
        passed: cam_rate >= 90.0,
        detail: format!(
            "L {:.0}/s, R {:.0}/s (need >=90/s)",
            input.rates[0], input.rates[1]
        ),
    });
    let dims_ok = input.frame_dims.iter().all(|d| *d == Some((200, 200)));
    checks.push(PreflightCheck {
        name: "XR5 image format",
        passed: dims_ok,
        detail: format!(
            "L {:?}, R {:?} (expected 200x200)",
            input.frame_dims[0], input.frame_dims[1]
        ),
    });
    let ml_ok = input.ml_loaded
        && input.rates[3] >= MIN_ML_RATE
        && input
            .ml
            .iter()
            .all(|v| v[0].is_finite() && v[0] > PRESENT_GATE);
    checks.push(PreflightCheck {
        name: "Eyelid model",
        passed: ml_ok,
        detail: format!(
            "{:.0}/s (need >={:.0}/s), presence L {:.3}, R {:.3}",
            input.rates[3], MIN_ML_RATE, input.ml[0][0], input.ml[1][0]
        ),
    });
    let gaze_ok =
        input.rates[2] >= 60.0 && input.gaze.left.gaze_valid && input.gaze.right.gaze_valid;
    checks.push(PreflightCheck {
        name: "Native gaze",
        passed: gaze_ok,
        detail: format!(
            "{:.0}/s, valid L {}, R {}",
            input.rates[2], input.gaze.left.gaze_valid, input.gaze.right.gaze_valid
        ),
    });
    let native_ok = input.gaze.left.openness_reported && input.gaze.right.openness_reported;
    checks.push(PreflightCheck {
        name: "Tobii openness endpoints",
        passed: native_ok,
        detail: format!(
            "reported L {}, R {}",
            input.gaze.left.openness_reported, input.gaze.right.openness_reported
        ),
    });
    let vergence = if gaze_ok {
        let l = gaze_x_deg(input.gaze.left.gaze);
        let r = gaze_x_deg(input.gaze.right.gaze);
        Some(shortest_angle_delta_deg(l, r))
    } else {
        None
    };
    let calibration_ok = vergence.map(|v| v <= 8.0).unwrap_or(false);
    checks.push(PreflightCheck {
        name: "Pimax / Tobii calibration",
        passed: calibration_ok,
        detail: match vergence {
            Some(v) => format!("straight-ahead L/R residual {v:.1} deg (need <=8 deg)"),
            None => "look straight ahead; both gaze vectors must be valid".into(),
        },
    });
    PreflightReport {
        ready: checks.iter().all(|c| c.passed),
        checks,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QualityLevel {
    Green,
    Yellow,
    Red,
}

#[derive(Clone, Debug)]
pub struct QualityInput {
    pub rates: [f32; 5],
    pub ml: [[f32; 5]; 2],
    pub gaze: GazeSample,
    pub baselines: [f32; 2],
    pub results: [EyeResult; 2],
}

#[derive(Clone, Debug)]
pub struct QualityReport {
    pub score: f32,
    pub level: QualityLevel,
    pub reasons: Vec<String>,
}

pub fn evaluate_quality(input: &QualityInput) -> QualityReport {
    let mut reasons = Vec::new();
    let cam = (input.rates[0].min(input.rates[1]) / 90.0).clamp(0.0, 1.0);
    if cam < 0.8 {
        reasons.push(format!(
            "camera rate low ({:.0}/{:.0} per sec)",
            input.rates[0], input.rates[1]
        ));
    }
    let ml_rate = (input.rates[3] / 45.0).clamp(0.0, 1.0);
    let presence = input
        .ml
        .iter()
        .map(|v| {
            if v[0].is_finite() {
                (v[0] / 0.25).clamp(0.0, 1.0)
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / 2.0;
    if presence < 0.6 {
        reasons.push("eye image confidence is low".into());
    }
    let gaze = match (input.gaze.left.gaze_valid, input.gaze.right.gaze_valid) {
        (true, true) => 1.0,
        (true, false) | (false, true) => 0.5,
        _ => 0.0,
    };
    if gaze < 1.0 {
        reasons.push("native gaze is missing for one or both eyes".into());
    }
    let native = if input.gaze.left.openness_reported && input.gaze.right.openness_reported {
        1.0
    } else {
        reasons.push("Tobii openness endpoint stream is missing".into());
        0.0
    };
    // Only inspect baseline drift while both eyes are visibly relaxed. Expressions are
    // expected to move the raw value and must never turn the quality lamp red.
    let relaxed = input
        .results
        .iter()
        .all(|r| r.openness > 0.85 && r.wide < 0.10 && r.squeeze < 0.10 && !r.blink);
    let baseline = if relaxed {
        let drift = (0..2)
            .map(|i| (input.ml[i][1] - input.baselines[i]).abs())
            .fold(0.0f32, f32::max);
        if drift > 0.10 {
            reasons.push(format!("relaxed baseline drifted by {drift:.3}; Recenter"));
        }
        (1.0 - drift / 0.15).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let score = 100.0
        * (0.25 * cam
            + 0.15 * ml_rate
            + 0.20 * presence
            + 0.20 * gaze
            + 0.10 * native
            + 0.10 * baseline);
    let level = if score >= 80.0 {
        QualityLevel::Green
    } else if score >= 55.0 {
        QualityLevel::Yellow
    } else {
        QualityLevel::Red
    };
    QualityReport {
        score,
        level,
        reasons,
    }
}

fn gaze_x_deg(gaze: [f32; 3]) -> f32 {
    gaze[0].atan2(-gaze[2]).to_degrees()
}

/// Smallest separation on a circular degree axis. A raw subtraction across the
/// -180/+180 seam turns a real 7.4-degree L/R residual into the bogus 352.6 degrees
/// observed on XR5 when looking slightly away from centre.
fn shortest_angle_delta_deg(a: f32, b: f32) -> f32 {
    let delta = (a - b).rem_euclid(360.0);
    delta.min(360.0 - delta)
}

fn ratio(n: u32, d: u32) -> f32 {
    if d == 0 {
        0.0
    } else {
        n as f32 / d as f32
    }
}

fn median(values: &[f32]) -> Option<f32> {
    percentile(values, 0.5)
}

fn percentile(values: &[f32], q: f32) -> Option<f32> {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let p = q.clamp(0.0, 1.0) * (v.len() - 1) as f32;
    let lo = p.floor() as usize;
    let hi = p.ceil() as usize;
    let t = p - lo as f32;
    Some(v[lo] * (1.0 - t) + v[hi] * t)
}

fn stddev(values: &[f32]) -> Option<f32> {
    let v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 2 {
        return None;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (v.len() - 1) as f32;
    Some(var.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Eye;

    fn gaze(native_enabled: bool) -> GazeSample {
        let mut g = GazeSample::default();
        for e in [&mut g.left, &mut g.right] {
            e.gaze = [0.0, 0.0, -1.0];
            e.gaze_valid = true;
            e.gaze_reported = true;
            e.openness_reported = true;
            e.openness_valid = native_enabled;
            e.pupil_pos = [0.5, 0.5];
            e.pupil_pos_valid = true;
        }
        g
    }

    fn ml(l: f32, r: f32) -> [[f32; 5]; 2] {
        [[1.0, l, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]]
    }

    fn fill_step(c: &mut GuidedCalibration, values: (f32, f32), native: bool) {
        let step = c.step();
        for _ in 0..step.sample_budget() {
            c.push(ml(values.0, values.1), &gaze(native));
        }
    }

    #[test]
    fn guided_flow_measures_ranges_and_mapping() {
        let mut c = GuidedCalibration::new();
        fill_step(&mut c, (0.65, 0.66), true); // relaxed
        fill_step(&mut c, (0.50, 0.51), true); // slow blinks
        fill_step(&mut c, (0.30, 0.31), false); // closed
        fill_step(&mut c, (0.72, 0.73), true); // wide
        fill_step(&mut c, (0.28, 0.29), false); // squeeze
        fill_step(&mut c, (0.32, 0.65), true); // left wink
        fill_step(&mut c, (0.65, 0.33), true); // right wink
        let report = c.report().unwrap();
        assert!(report.passed, "{report:?}");
        assert_eq!(report.mapping, MappingVerdict::Correct);
        assert!(report.wide_supported.iter().all(|v| *v));
        assert!(report.blink_depth[0] > 0.30);
    }

    #[test]
    fn weak_wide_is_disabled_instead_of_amplified() {
        let mut c = GuidedCalibration::new();
        fill_step(&mut c, (0.65, 0.65), true);
        fill_step(&mut c, (0.50, 0.50), true);
        fill_step(&mut c, (0.30, 0.30), false);
        fill_step(&mut c, (0.655, 0.656), true);
        fill_step(&mut c, (0.28, 0.28), false);
        fill_step(&mut c, (0.30, 0.65), true);
        fill_step(&mut c, (0.65, 0.30), true);
        let report = c.report().unwrap();
        assert_eq!(report.wide_supported, [false, false]);
    }

    #[test]
    fn swapped_winks_block_application() {
        let mut c = GuidedCalibration::new();
        fill_step(&mut c, (0.65, 0.65), true);
        fill_step(&mut c, (0.50, 0.50), true);
        fill_step(&mut c, (0.30, 0.30), false);
        fill_step(&mut c, (0.72, 0.72), true);
        fill_step(&mut c, (0.28, 0.28), false);
        fill_step(&mut c, (0.65, 0.30), true);
        fill_step(&mut c, (0.30, 0.65), true);
        let report = c.report().unwrap();
        assert_eq!(report.mapping, MappingVerdict::Swapped);
        assert!(!report.passed);
    }

    #[test]
    fn seeded_store_preserves_curve_anchor() {
        let old = CalibStore {
            left: CalibSnapshot {
                baseline: 0.6,
                baseline_n: 100,
                frame_count: 500,
                blink_depth: 0.2,
                mid_anchor: 0.43,
                learned_once: true,
            },
            right: CalibSnapshot {
                baseline: 0.6,
                baseline_n: 100,
                frame_count: 500,
                blink_depth: 0.2,
                mid_anchor: 0.56,
                learned_once: true,
            },
        };
        let report = CalibrationReport {
            baseline: [0.67, 0.64],
            blink_depth: [0.24, 0.19],
            wide_span: [0.04, 0.03],
            wide_snr: [3.0, 2.5],
            wide_supported: [true, true],
            presence_rate: [1.0, 1.0],
            native_closed_rate: [1.0, 1.0],
            pupil_center: [[0.5; 2]; 2],
            pupil_center_valid: [true; 2],
            mapping: MappingVerdict::Correct,
            quality_score: 100.0,
            passed: true,
        };
        let seeded = report.calibration_store(Some(old));
        assert_eq!(seeded.left.mid_anchor, 0.43);
        assert_eq!(seeded.right.mid_anchor, 0.56);
        assert_eq!(seeded.left.baseline, 0.67);
        assert!(seeded.left.learned_once);
    }

    #[test]
    fn quality_ignores_expected_expression_motion() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for r in &mut results {
            r.openness = 0.0;
            r.blink = true;
        }
        let q = evaluate_quality(&QualityInput {
            rates: [120.0, 120.0, 120.0, 60.0, 120.0],
            ml: ml(0.25, 0.25),
            gaze: gaze(true),
            baselines: [0.65, 0.65],
            results,
        });
        assert_eq!(q.level, QualityLevel::Green);
    }

    #[test]
    fn xr5_reference_rate_passes_preflight_but_an_overloaded_model_does_not() {
        let input = |ml_rate| PreflightInput {
            rates: [120.0, 120.0, 120.0, ml_rate, 120.0],
            frame_dims: [Some((200, 200)); 2],
            ml_loaded: true,
            ml: [[0.349, 0.65, 0.0, 0.0, 0.0]; 2],
            gaze: gaze(true),
        };
        let reference = evaluate_preflight(&input(41.86));
        assert!(
            reference.ready,
            "real XR5 reference rate must pass: {reference:?}"
        );

        let overloaded = evaluate_preflight(&input(34.0));
        assert!(!overloaded.ready);
        assert!(
            !overloaded
                .checks
                .iter()
                .find(|check| check.name == "Eyelid model")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn preflight_vergence_wraps_across_the_180_degree_seam() {
        assert!((shortest_angle_delta_deg(176.3, -176.3) - 7.4).abs() < 0.001);
        assert!((shortest_angle_delta_deg(-176.3, 176.3) - 7.4).abs() < 0.001);
    }
}
