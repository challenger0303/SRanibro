//! Renderer-only preregistration for the Phase 1.1 whole-image-mean experiment.
//!
//! This module never calls EyeNet. It deterministically selects the feasible aperture
//! range and solves bounded sclera levels before any model inference is allowed.

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::renderer::{render_stereo, ImageCovariates, StereoPolicy, SyntheticEyeSpec};

pub const VERSION: &str = "synthetic-eye-luminance-match-v1";
pub const SOLVER_VERSION: &str = "quantized-mean-bisection-v1";
pub const APERTURE_STEPS: usize = 40;
pub const APERTURE_MAX: f32 = 1.30;
pub const REFERENCE_TARGET: f32 = 1.0;
pub const SCLERA_LOW: f32 = 0.55;
pub const SCLERA_HIGH: f32 = 0.95;
pub const DEFAULT_SCLERA: f32 = 0.78;
pub const SOLVER_ITERATIONS: usize = 24;
pub const MAX_MEAN_ERROR: f32 = 0.5 / 255.0;
pub const BOUND_PROXIMITY: f32 = 1.0 / 255.0;
pub const MAX_SATURATION: f32 = 0.05;
pub const MIN_SELECTED_POINTS: usize = 17;
pub const MIN_SELECTED_SPAN: f32 = 0.50;
pub const MIN_D_COVERAGE: f32 = 0.80;
pub const MIN_D_MATCHED_CASES: usize = 5;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeanMatchKind {
    ConstantMeanGeometry,
    FixedGeometrySameSclera,
    FixedGeometryOriginalMean,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolverCandidate {
    pub sclera_level: f32,
    pub achieved_mean: f32,
    pub absolute_error: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeanMatchRecord {
    pub kind: MeanMatchKind,
    pub source_aperture_index: usize,
    pub source_aperture: f32,
    pub reference_aperture: f32,
    pub sclera_level: f32,
    pub target_mean: Option<f32>,
    pub achieved_mean: f32,
    pub absolute_error: Option<f32>,
    pub within_one_u8_of_bound: bool,
    pub solver_version: String,
    pub bracket_low: Option<SolverCandidate>,
    pub bracket_high: Option<SolverCandidate>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UnmatchedMeanTarget {
    pub source_aperture_index: usize,
    pub source_aperture: f32,
    pub target_mean: f32,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LuminancePlan {
    pub version: &'static str,
    pub aperture_grid_formula: &'static str,
    pub reference_selection_rule: &'static str,
    pub reference_index: usize,
    pub reference_aperture: f32,
    pub reference_mean_default_sclera: f32,
    pub sclera_bounds: [f32; 2],
    pub selected_start_index: usize,
    pub selected_end_index: usize,
    pub selected_aperture_range: [f32; 2],
    pub selected_count: usize,
    pub selected_span: f32,
    pub common_achievable_mean_interval: [f32; 2],
    pub constant_mean_target: f32,
    pub solver_version: &'static str,
    pub solver_iterations: usize,
    pub maximum_absolute_mean_error: f32,
    pub bound_proximity: f32,
    pub d_matched_count: usize,
    pub d_unmatched_count: usize,
    pub d_renderer_coverage: f32,
    pub d_interpretation_preregistered: bool,
    pub d_unmatched: Vec<UnmatchedMeanTarget>,
    pub prior_aperture_spearman_lr: [f32; 2],
    pub prior_sclera_spearman_lr: [f32; 2],
    pub prior_uniform_offset_spearman_lr: [f32; 2],
    pub signed_interpretation_contract: Vec<&'static str>,
    pub limitations: Vec<&'static str>,
}

#[derive(Clone, Debug)]
pub struct PreparedPoint {
    pub spec: SyntheticEyeSpec,
    pub record: MeanMatchRecord,
}

#[derive(Clone, Debug)]
pub struct PreparedLuminance {
    pub plan: LuminancePlan,
    pub constant_mean: Vec<PreparedPoint>,
    pub same_sclera_control: Vec<PreparedPoint>,
    pub original_mean_control: Vec<PreparedPoint>,
}

#[derive(Clone, Copy, Debug)]
struct Interval {
    low_mean: f32,
    high_mean: f32,
}

#[derive(Clone, Debug)]
struct SolvedLevel {
    level: f32,
    mean: f32,
    error: f32,
    low: SolverCandidate,
    high: SolverCandidate,
}

pub fn aperture(index: usize) -> f32 {
    APERTURE_MAX * index as f32 / APERTURE_STEPS as f32
}

pub fn reference_index() -> usize {
    (0..=APERTURE_STEPS)
        .min_by(|&left, &right| {
            let left_distance = (aperture(left) - REFERENCE_TARGET).abs();
            let right_distance = (aperture(right) - REFERENCE_TARGET).abs();
            left_distance
                .total_cmp(&right_distance)
                .then_with(|| left.cmp(&right))
        })
        .expect("nonempty fixed aperture grid")
}

pub fn prepare() -> Result<PreparedLuminance, String> {
    static PREPARED: OnceLock<Result<PreparedLuminance, String>> = OnceLock::new();
    PREPARED.get_or_init(prepare_uncached).clone()
}

fn prepare_uncached() -> Result<PreparedLuminance, String> {
    let reference_index = reference_index();
    let reference_aperture = aperture(reference_index);
    let mut reference_spec = SyntheticEyeSpec::default();
    reference_spec.aperture = reference_aperture;
    reference_spec.sclera_level = DEFAULT_SCLERA;
    let reference_mean = guarded_mean(&reference_spec)?;

    let mut intervals = vec![None; APERTURE_STEPS + 1];
    for (index, slot) in intervals.iter_mut().enumerate() {
        let aperture = aperture(index);
        let low_mean = guarded_mean(&spec(aperture, SCLERA_LOW))?;
        let high_mean = guarded_mean(&spec(aperture, SCLERA_HIGH))?;
        if low_mean > high_mean {
            return Err(format!(
                "renderer mean was non-monotone at aperture {aperture:.9}: {low_mean:.9} > {high_mean:.9}"
            ));
        }
        *slot = Some(Interval {
            low_mean,
            high_mean,
        });
    }

    let (start, end, common_low, common_high, target) =
        select_range(&intervals, reference_index, reference_mean)
            .ok_or("no contiguous luminance-match range contains the reference aperture")?;
    let selected_count = end - start + 1;
    let selected_span = aperture(end) - aperture(start);
    if selected_count < MIN_SELECTED_POINTS || selected_span < MIN_SELECTED_SPAN {
        return Err(format!(
            "feasible range too small: {selected_count} points, span {selected_span:.9}; need at least {MIN_SELECTED_POINTS} and {MIN_SELECTED_SPAN:.2}"
        ));
    }

    let mut constant_mean = Vec::with_capacity(selected_count);
    let mut same_sclera_control = Vec::with_capacity(selected_count);
    let mut original_mean_control = Vec::with_capacity(selected_count);
    let mut d_unmatched = Vec::new();

    for source_index in start..=end {
        let source_aperture = aperture(source_index);
        let solved = solve_level(source_aperture, target)?;
        let b_spec = spec(source_aperture, solved.level);
        let b_record = solved_record(
            MeanMatchKind::ConstantMeanGeometry,
            source_index,
            source_aperture,
            reference_aperture,
            target,
            &solved,
        );
        constant_mean.push(PreparedPoint {
            spec: b_spec,
            record: b_record,
        });

        let c_spec = spec(reference_aperture, solved.level);
        let c_mean = guarded_mean(&c_spec)?;
        same_sclera_control.push(PreparedPoint {
            spec: c_spec,
            record: MeanMatchRecord {
                kind: MeanMatchKind::FixedGeometrySameSclera,
                source_aperture_index: source_index,
                source_aperture,
                reference_aperture,
                sclera_level: solved.level,
                target_mean: None,
                achieved_mean: c_mean,
                absolute_error: None,
                within_one_u8_of_bound: at_bound(solved.level),
                solver_version: "reuse-constant-mean-sclera-v1".into(),
                bracket_low: None,
                bracket_high: None,
            },
        });

        let original_target = guarded_mean(&spec(source_aperture, DEFAULT_SCLERA))?;
        match solve_level(reference_aperture, original_target) {
            Ok(d_solved) => {
                original_mean_control.push(PreparedPoint {
                    spec: spec(reference_aperture, d_solved.level),
                    record: solved_record(
                        MeanMatchKind::FixedGeometryOriginalMean,
                        source_index,
                        source_aperture,
                        reference_aperture,
                        original_target,
                        &d_solved,
                    ),
                });
            }
            Err(reason) => d_unmatched.push(UnmatchedMeanTarget {
                source_aperture_index: source_index,
                source_aperture,
                target_mean: original_target,
                reason,
            }),
        }
    }

    let d_matched_count = original_mean_control.len();
    let d_renderer_coverage = d_matched_count as f32 / selected_count as f32;
    let plan = LuminancePlan {
        version: VERSION,
        aperture_grid_formula: "a_i = 1.30 * i / 40, i=0..40",
        reference_selection_rule:
            "minimum |a_i-1.0|; exact distance tie resolves to lower index",
        reference_index,
        reference_aperture,
        reference_mean_default_sclera: reference_mean,
        sclera_bounds: [SCLERA_LOW, SCLERA_HIGH],
        selected_start_index: start,
        selected_end_index: end,
        selected_aperture_range: [aperture(start), aperture(end)],
        selected_count,
        selected_span,
        common_achievable_mean_interval: [common_low, common_high],
        constant_mean_target: target,
        solver_version: SOLVER_VERSION,
        solver_iterations: SOLVER_ITERATIONS,
        maximum_absolute_mean_error: MAX_MEAN_ERROR,
        bound_proximity: BOUND_PROXIMITY,
        d_matched_count,
        d_unmatched_count: d_unmatched.len(),
        d_renderer_coverage,
        d_interpretation_preregistered: d_matched_count >= MIN_D_MATCHED_CASES
            && d_renderer_coverage >= MIN_D_COVERAGE,
        d_unmatched,
        prior_aperture_spearman_lr: [0.9688153, 0.90749127],
        prior_sclera_spearman_lr: [1.0, 1.0],
        prior_uniform_offset_spearman_lr: [-0.8787879, -0.8181818],
        signed_interpretation_contract: vec![
            "B positive while C negative: geometry/visible-area contribution overcame the opposing sclera-intensity contribution",
            "B and C both negative with comparable sign: compensating sclera photometrics dominate; no geometry claim",
            "B near zero while C negative: cancellation; inconclusive",
            "A positive and D positive/comparable on matched cases: original mean trajectory reproduced through sclera can substantially explain A",
            "A positive and D weak/nonmonotone: tested whole-mean-via-sclera path does not explain A",
            "unregistered sign patterns are anomalous/inconclusive and receive no mechanism claim",
        ],
        limitations: vec![
            "B aperture and solved sclera level are perfectly coupled by construction",
            "whole-image mean matching does not match bright-pixel mass, histogram, contrast, edge energy, or visible sclera area",
            "all conclusions are limited to this synthetic renderer family",
        ],
    };
    Ok(PreparedLuminance {
        plan,
        constant_mean,
        same_sclera_control,
        original_mean_control,
    })
}

fn select_range(
    intervals: &[Option<Interval>],
    reference_index: usize,
    reference_mean: f32,
) -> Option<(usize, usize, f32, f32, f32)> {
    let mut best: Option<(usize, usize, f32, f32, f32)> = None;
    for start in 0..=reference_index {
        for end in reference_index..intervals.len() {
            let mut common_low = f32::NEG_INFINITY;
            let mut common_high = f32::INFINITY;
            let mut eligible = true;
            for interval in &intervals[start..=end] {
                let Some(interval) = interval else {
                    eligible = false;
                    break;
                };
                common_low = common_low.max(interval.low_mean);
                common_high = common_high.min(interval.high_mean);
            }
            if !eligible || common_low > common_high {
                continue;
            }
            let target = reference_mean.clamp(common_low, common_high);
            let candidate = (start, end, common_low, common_high, target);
            if range_better(candidate, best, reference_mean) {
                best = Some(candidate);
            }
        }
    }
    best
}

fn range_better(
    candidate: (usize, usize, f32, f32, f32),
    current: Option<(usize, usize, f32, f32, f32)>,
    reference_mean: f32,
) -> bool {
    let Some(current) = current else {
        return true;
    };
    let candidate_count = candidate.1 - candidate.0 + 1;
    let current_count = current.1 - current.0 + 1;
    candidate_count > current_count
        || (candidate_count == current_count
            && ((candidate.4 - reference_mean)
                .abs()
                .total_cmp(&(current.4 - reference_mean).abs())
                .is_lt()
                || ((candidate.4 - reference_mean).abs().to_bits()
                    == (current.4 - reference_mean).abs().to_bits()
                    && candidate.0 < current.0)))
}

fn solve_level(aperture: f32, target: f32) -> Result<SolvedLevel, String> {
    let mut low_level = SCLERA_LOW;
    let mut high_level = SCLERA_HIGH;
    let mut low_mean = guarded_mean(&spec(aperture, low_level))?;
    let mut high_mean = guarded_mean(&spec(aperture, high_level))?;
    if low_mean > high_mean {
        return Err(format!(
            "non-monotone endpoint means {low_mean:.9} > {high_mean:.9}"
        ));
    }

    for _ in 0..SOLVER_ITERATIONS {
        let midpoint = (low_level + high_level) * 0.5;
        let midpoint_mean = guarded_mean(&spec(aperture, midpoint))?;
        if midpoint_mean < target {
            low_level = midpoint;
            low_mean = midpoint_mean;
        } else {
            high_level = midpoint;
            high_mean = midpoint_mean;
        }
    }

    let low = SolverCandidate {
        sclera_level: low_level,
        achieved_mean: low_mean,
        absolute_error: (low_mean - target).abs(),
    };
    let high = SolverCandidate {
        sclera_level: high_level,
        achieved_mean: high_mean,
        absolute_error: (high_mean - target).abs(),
    };
    let selected = if low.absolute_error.total_cmp(&high.absolute_error).is_lt()
        || (low.absolute_error.to_bits() == high.absolute_error.to_bits()
            && low.sclera_level.total_cmp(&high.sclera_level).is_le())
    {
        &low
    } else {
        &high
    };
    if selected.absolute_error > MAX_MEAN_ERROR {
        return Err(format!(
            "best bounded mean error {:.9} exceeds {:.9}",
            selected.absolute_error, MAX_MEAN_ERROR
        ));
    }
    // Re-render the selected value so every accepted point independently passes guards.
    let selected_mean = guarded_mean(&spec(aperture, selected.sclera_level))?;
    Ok(SolvedLevel {
        level: selected.sclera_level,
        mean: selected_mean,
        error: (selected_mean - target).abs(),
        low,
        high,
    })
}

fn solved_record(
    kind: MeanMatchKind,
    source_aperture_index: usize,
    source_aperture: f32,
    reference_aperture: f32,
    target_mean: f32,
    solved: &SolvedLevel,
) -> MeanMatchRecord {
    MeanMatchRecord {
        kind,
        source_aperture_index,
        source_aperture,
        reference_aperture,
        sclera_level: solved.level,
        target_mean: Some(target_mean),
        achieved_mean: solved.mean,
        absolute_error: Some(solved.error),
        within_one_u8_of_bound: at_bound(solved.level),
        solver_version: SOLVER_VERSION.into(),
        bracket_low: Some(solved.low.clone()),
        bracket_high: Some(solved.high.clone()),
    }
}

fn spec(aperture: f32, sclera_level: f32) -> SyntheticEyeSpec {
    SyntheticEyeSpec {
        aperture,
        sclera_level,
        ..SyntheticEyeSpec::default()
    }
}

fn guarded_mean(spec: &SyntheticEyeSpec) -> Result<f32, String> {
    let pair = render_stereo(spec, spec, StereoPolicy::AnatomicalMirror);
    guard_eye("left", &pair.left.covariates, pair.left.eye_like)?;
    guard_eye("right", &pair.right.covariates, pair.right.eye_like)?;
    if pair.left.covariates.mean.to_bits() != pair.right.covariates.mean.to_bits() {
        return Err("anatomical mirror changed the quantized image mean".into());
    }
    Ok(pair.left.covariates.mean)
}

fn guard_eye(side: &str, covariates: &ImageCovariates, eye_like: bool) -> Result<(), String> {
    let finite = [
        covariates.mean,
        covariates.stddev,
        covariates.edge_energy,
        covariates.saturation_fraction,
        covariates.visible_area_fraction,
        covariates.measured_aperture_geometry,
        covariates.measured_aperture_raster,
    ]
    .iter()
    .all(|value| value.is_finite());
    if !eye_like {
        Err(format!(
            "{side} renderer output failed eye-like constraints"
        ))
    } else if !finite {
        Err(format!("{side} renderer covariates were non-finite"))
    } else if covariates.saturation_fraction > MAX_SATURATION {
        Err(format!(
            "{side} saturation {:.9} exceeded {:.9}",
            covariates.saturation_fraction, MAX_SATURATION
        ))
    } else if covariates.frame_truncated {
        Err(format!("{side} renderer primitive contacted the frame"))
    } else {
        Ok(())
    }
}

fn at_bound(level: f32) -> bool {
    (level - SCLERA_LOW).abs() <= BOUND_PROXIMITY || (level - SCLERA_HIGH).abs() <= BOUND_PROXIMITY
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_is_the_nearest_real_grid_point() {
        assert_eq!(reference_index(), 31);
        assert!((aperture(reference_index()) - 1.0075).abs() <= f32::EPSILON);
    }

    #[test]
    fn solver_is_bit_deterministic_and_within_tolerance() {
        let target = guarded_mean(&spec(aperture(reference_index()), DEFAULT_SCLERA)).unwrap();
        let first = solve_level(aperture(reference_index()), target).unwrap();
        let second = solve_level(aperture(reference_index()), target).unwrap();
        assert_eq!(first.level.to_bits(), second.level.to_bits());
        assert_eq!(first.mean.to_bits(), second.mean.to_bits());
        assert!(first.error <= MAX_MEAN_ERROR);
    }

    #[test]
    fn quantized_plateau_tie_resolves_to_the_lower_sclera_level() {
        let aperture = aperture(reference_index());
        let target = guarded_mean(&spec(aperture, SCLERA_LOW)).unwrap();
        let solved = solve_level(aperture, target).unwrap();
        assert_eq!(solved.level.to_bits(), SCLERA_LOW.to_bits());
        assert!(at_bound(solved.level));
    }

    #[test]
    fn prepared_plan_meets_every_renderer_preregistration_gate() {
        let prepared = prepare().unwrap();
        assert!(prepared.plan.selected_count >= MIN_SELECTED_POINTS);
        assert!(prepared.plan.selected_span >= MIN_SELECTED_SPAN);
        assert_eq!(prepared.constant_mean.len(), prepared.plan.selected_count);
        assert_eq!(
            prepared.same_sclera_control.len(),
            prepared.plan.selected_count
        );
        for point in &prepared.constant_mean {
            assert!(point.record.absolute_error.unwrap() <= MAX_MEAN_ERROR);
            assert_eq!(
                guarded_mean(&point.spec).unwrap().to_bits(),
                point.record.achieved_mean.to_bits()
            );
        }
        for (b, c) in prepared
            .constant_mean
            .iter()
            .zip(&prepared.same_sclera_control)
        {
            assert_eq!(
                b.record.sclera_level.to_bits(),
                c.record.sclera_level.to_bits()
            );
            assert_eq!(
                c.spec.aperture.to_bits(),
                prepared.plan.reference_aperture.to_bits()
            );
        }
    }
}
