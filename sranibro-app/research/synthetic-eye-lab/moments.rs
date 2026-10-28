//! Renderer-only preparation for the preregistered Phase 1.2 two-moment study.
//!
//! This module never loads or calls EyeNet. It solves every photometric trajectory,
//! applies the frozen renderer gates, and freezes the common index set first.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::luminance::aperture;
use crate::model::sha256_hex;
use crate::renderer::{
    gray_moments, render, render_stereo, GrayMoments, ImageCovariates, PhotometricBasis,
    StereoPolicy, SyntheticEyeSpec,
};

pub const VERSION: &str = "synthetic-eye-two-moment-v1";
pub const PREREGISTRATION_COMMIT: &str = "386e8b157cd675c9a140fe5c85051f622a4fbc43";
pub const UNIVERSE_START: usize = 7;
pub const UNIVERSE_END: usize = 40;
pub const UNIVERSE_COUNT: usize = UNIVERSE_END - UNIVERSE_START + 1;
pub const REFERENCE_INDICES: [usize; 2] = [15, 31];
pub const SKIN_BOUNDS: [f64; 2] = [0.30, 0.60];
pub const SCLERA_BOUNDS: [f64; 2] = [0.65, 0.95];
pub const DEFAULT_SKIN: f32 = 0.46;
pub const DEFAULT_SCLERA: f32 = 0.78;
pub const COARSE_GRID_POINTS: usize = 65;
pub const REFINEMENT_LEVELS: usize = 3;
pub const REFINEMENT_RADIUS: i32 = 8;
pub const MAX_MOMENT_RESIDUAL_GRAY: f64 = 0.25;
pub const BOUND_MARGIN_FRACTION: f64 = 0.01;
pub const MAX_SATURATION: f32 = 0.01;
pub const MAX_SATURATION_RANGE: f32 = 0.005;
pub const TRAJECTORY_JUMP_MULTIPLIER: f64 = 5.0;
pub const MIN_COMMON_INDICES: usize = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TwoMomentKind {
    Baseline,
    Matched,
    FixedGeometryReplay,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TwoMomentRecord {
    pub kind: TwoMomentKind,
    pub source_aperture_index: usize,
    pub source_aperture: f32,
    pub reference_index: Option<usize>,
    pub reference_aperture: Option<f32>,
    pub solver_skin_f64: Option<f64>,
    pub solver_sclera_f64: Option<f64>,
    pub applied_skin_f32: f32,
    pub applied_sclera_f32: f32,
    pub target_mean_gray: Option<f64>,
    pub target_stddev_gray: Option<f64>,
    pub achieved_mean_gray: f64,
    pub achieved_stddev_gray: f64,
    pub absolute_mean_residual_gray: Option<f64>,
    pub absolute_stddev_residual_gray: Option<f64>,
    pub objective: Option<f64>,
    pub near_bound: bool,
    pub predicted_sha256: String,
    pub canonical_sha256: String,
    pub solver_version: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct InvalidIndex {
    pub index: usize,
    pub aperture: f32,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReferencePlan {
    pub reference_index: usize,
    pub reference_aperture: f32,
    pub target_mean_gray: f64,
    pub target_stddev_gray: f64,
    pub matched_saturation_range: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TwoMomentPlan {
    pub version: &'static str,
    pub preregistration_commit: &'static str,
    pub aperture_formula: &'static str,
    pub frozen_universe: Vec<usize>,
    pub reference_indices: [usize; 2],
    pub reference_apertures: [f32; 2],
    pub skin_bounds: [f64; 2],
    pub sclera_bounds: [f64; 2],
    pub default_levels: [f32; 2],
    pub solver_version: &'static str,
    pub coarse_grid_points: usize,
    pub refinement_levels: usize,
    pub maximum_moment_residual_gray: f64,
    pub bound_margin_fraction: f64,
    pub maximum_saturation: f32,
    pub maximum_saturation_range: f32,
    pub trajectory_jump_multiplier: f64,
    pub minimum_common_indices: usize,
    pub references: Vec<ReferencePlan>,
    pub invalid_indices: Vec<InvalidIndex>,
    pub common_indices: Vec<usize>,
    pub preparation_repetitions: usize,
    pub renderer_decision: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedPoint {
    pub spec: SyntheticEyeSpec,
    pub record: TwoMomentRecord,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedReference {
    pub reference_index: usize,
    pub matched: Vec<PreparedPoint>,
    pub replay: Vec<PreparedPoint>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedTwoMoment {
    pub plan: TwoMomentPlan,
    pub baseline: Vec<PreparedPoint>,
    pub references: Vec<PreparedReference>,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    skin: f64,
    sclera: f64,
    applied_skin: f32,
    applied_sclera: f32,
    objective: f64,
    default_distance: f64,
}

pub fn prepare() -> Result<PreparedTwoMoment, String> {
    let first = prepare_once();
    let second = prepare_once();
    match (first, second) {
        (Ok(first), Ok(second)) if first == second => Ok(first),
        (Err(first), Err(second)) if first == second => Err(first),
        _ => Err("G1 repeated preparation produced different outcomes".into()),
    }
}

fn prepare_once() -> Result<PreparedTwoMoment, String> {
    let universe: Vec<_> = (UNIVERSE_START..=UNIVERSE_END).collect();
    let mut baseline = Vec::with_capacity(UNIVERSE_COUNT);
    let mut baseline_rendered = BTreeMap::new();
    let mut baseline_hashes = BTreeSet::new();
    let mut previous_geometry = None;
    let mut previous_raster = None;

    for &index in &universe {
        let spec = spec(index, DEFAULT_SKIN, DEFAULT_SCLERA);
        assert_mirror(&spec)?;
        let rendered = render(&spec);
        if let Some(previous) = previous_geometry {
            if rendered.covariates.measured_aperture_geometry <= previous {
                return Err(format!(
                    "G4 measured_aperture_geometry was not strictly increasing at index {index}"
                ));
            }
        }
        if let Some(previous) = previous_raster {
            if rendered.covariates.measured_aperture_raster <= previous {
                return Err(format!(
                    "G4 measured_aperture_raster was not strictly increasing at index {index}"
                ));
            }
        }
        previous_geometry = Some(rendered.covariates.measured_aperture_geometry);
        previous_raster = Some(rendered.covariates.measured_aperture_raster);
        let hash = sha256_hex(&rendered.pixels);
        if !baseline_hashes.insert(hash.clone()) {
            return Err(format!("G4 duplicate canonical S3 hash at index {index}"));
        }
        let moments = gray_moments(&rendered.pixels);
        baseline.push(PreparedPoint {
            spec: spec.clone(),
            record: TwoMomentRecord {
                kind: TwoMomentKind::Baseline,
                source_aperture_index: index,
                source_aperture: aperture(index),
                reference_index: None,
                reference_aperture: None,
                solver_skin_f64: None,
                solver_sclera_f64: None,
                applied_skin_f32: DEFAULT_SKIN,
                applied_sclera_f32: DEFAULT_SCLERA,
                target_mean_gray: None,
                target_stddev_gray: None,
                achieved_mean_gray: moments.mean,
                achieved_stddev_gray: moments.stddev,
                absolute_mean_residual_gray: None,
                absolute_stddev_residual_gray: None,
                objective: None,
                near_bound: false,
                predicted_sha256: hash.clone(),
                canonical_sha256: hash,
                solver_version: "not_applicable_baseline".into(),
            },
        });
        baseline_rendered.insert(index, rendered);
    }

    let mut invalid = BTreeMap::<usize, Vec<String>>::new();
    for point in &baseline {
        let rendered = baseline_rendered
            .get(&point.record.source_aperture_index)
            .expect("baseline render exists");
        add_image_failures(
            point.record.source_aperture_index,
            "S3",
            &rendered.covariates,
            rendered.eye_like,
            &mut invalid,
        );
    }

    let mut references = Vec::with_capacity(REFERENCE_INDICES.len());
    let mut reference_plans = Vec::with_capacity(REFERENCE_INDICES.len());
    for &reference_index in &REFERENCE_INDICES {
        let target_render = baseline_rendered
            .get(&reference_index)
            .ok_or_else(|| format!("missing reference index {reference_index}"))?;
        let target = gray_moments(&target_render.pixels);
        let mut matched = Vec::with_capacity(UNIVERSE_COUNT);
        let mut replay = Vec::with_capacity(UNIVERSE_COUNT);

        for &index in &universe {
            let base_spec = spec(index, DEFAULT_SKIN, DEFAULT_SCLERA);
            let basis = PhotometricBasis::from_spec(&base_spec);
            assert_basis_bound_corners(&base_spec, &basis)?;
            let solved = solve(&basis, target);
            if index == reference_index
                && (solved.applied_skin.to_bits() != DEFAULT_SKIN.to_bits()
                    || solved.applied_sclera.to_bits() != DEFAULT_SCLERA.to_bits())
            {
                return Err(format!(
                    "reference index {reference_index} did not select exact defaults"
                ));
            }

            let matched_spec = spec(index, solved.applied_skin, solved.applied_sclera);
            let predicted = basis.predict_pixels(solved.applied_skin, solved.applied_sclera);
            let canonical = render(&matched_spec);
            if predicted != canonical.pixels {
                return Err(format!(
                    "G2 fast/canonical byte mismatch at ref {reference_index} index {index}"
                ));
            }
            let fast_moments = basis.moments(solved.applied_skin, solved.applied_sclera);
            let canonical_moments = gray_moments(&canonical.pixels);
            if fast_moments.mean.to_bits() != canonical_moments.mean.to_bits()
                || fast_moments.stddev.to_bits() != canonical_moments.stddev.to_bits()
            {
                return Err(format!(
                    "G2 fast/canonical moment mismatch at ref {reference_index} index {index}"
                ));
            }
            assert_mirror(&matched_spec)?;
            let baseline_case = baseline_rendered
                .get(&index)
                .expect("baseline render exists");
            if canonical.covariates.measured_aperture_geometry.to_bits()
                != baseline_case
                    .covariates
                    .measured_aperture_geometry
                    .to_bits()
                || canonical.covariates.measured_aperture_raster.to_bits()
                    != baseline_case.covariates.measured_aperture_raster.to_bits()
            {
                return Err(format!(
                    "G3 S1 geometry changed at ref {reference_index} index {index}"
                ));
            }
            let achieved = canonical_moments;
            let mean_residual = (achieved.mean - target.mean).abs();
            let stddev_residual = (achieved.stddev - target.stddev).abs();
            let near_bound = is_near_bound(solved.skin, SKIN_BOUNDS)
                || is_near_bound(solved.sclera, SCLERA_BOUNDS);
            let predicted_hash = sha256_hex(&predicted);
            let canonical_hash = sha256_hex(&canonical.pixels);
            let record = TwoMomentRecord {
                kind: TwoMomentKind::Matched,
                source_aperture_index: index,
                source_aperture: aperture(index),
                reference_index: Some(reference_index),
                reference_aperture: Some(aperture(reference_index)),
                solver_skin_f64: Some(solved.skin),
                solver_sclera_f64: Some(solved.sclera),
                applied_skin_f32: solved.applied_skin,
                applied_sclera_f32: solved.applied_sclera,
                target_mean_gray: Some(target.mean),
                target_stddev_gray: Some(target.stddev),
                achieved_mean_gray: achieved.mean,
                achieved_stddev_gray: achieved.stddev,
                absolute_mean_residual_gray: Some(mean_residual),
                absolute_stddev_residual_gray: Some(stddev_residual),
                objective: Some(solved.objective),
                near_bound,
                predicted_sha256: predicted_hash,
                canonical_sha256: canonical_hash,
                solver_version: "two-moment-grid-65-refine17x3-v1".into(),
            };
            if near_bound {
                add_reason(
                    index,
                    format!("P1 ref{reference_index} solved level near bound"),
                    &mut invalid,
                );
            }
            if mean_residual > MAX_MOMENT_RESIDUAL_GRAY
                || stddev_residual > MAX_MOMENT_RESIDUAL_GRAY
            {
                add_reason(
                    index,
                    format!(
                        "P2 ref{reference_index} residual mean={mean_residual:.9} stddev={stddev_residual:.9}"
                    ),
                    &mut invalid,
                );
            }
            add_image_failures(
                index,
                &format!("S1_ref{reference_index}"),
                &canonical.covariates,
                canonical.eye_like,
                &mut invalid,
            );
            matched.push(PreparedPoint {
                spec: matched_spec,
                record,
            });

            let replay_spec = spec(reference_index, solved.applied_skin, solved.applied_sclera);
            let replay_render = render(&replay_spec);
            if replay_render
                .covariates
                .measured_aperture_geometry
                .to_bits()
                != target_render
                    .covariates
                    .measured_aperture_geometry
                    .to_bits()
                || replay_render.covariates.measured_aperture_raster.to_bits()
                    != target_render.covariates.measured_aperture_raster.to_bits()
            {
                return Err(format!(
                    "G3 S2 geometry changed at ref {reference_index} source index {index}"
                ));
            }
            assert_mirror(&replay_spec)?;
            add_image_failures(
                index,
                &format!("S2_ref{reference_index}"),
                &replay_render.covariates,
                replay_render.eye_like,
                &mut invalid,
            );
            let replay_moments = gray_moments(&replay_render.pixels);
            let replay_hash = sha256_hex(&replay_render.pixels);
            replay.push(PreparedPoint {
                spec: replay_spec,
                record: TwoMomentRecord {
                    kind: TwoMomentKind::FixedGeometryReplay,
                    source_aperture_index: index,
                    source_aperture: aperture(index),
                    reference_index: Some(reference_index),
                    reference_aperture: Some(aperture(reference_index)),
                    solver_skin_f64: Some(solved.skin),
                    solver_sclera_f64: Some(solved.sclera),
                    applied_skin_f32: solved.applied_skin,
                    applied_sclera_f32: solved.applied_sclera,
                    target_mean_gray: None,
                    target_stddev_gray: None,
                    achieved_mean_gray: replay_moments.mean,
                    achieved_stddev_gray: replay_moments.stddev,
                    absolute_mean_residual_gray: None,
                    absolute_stddev_residual_gray: None,
                    objective: None,
                    near_bound,
                    predicted_sha256: replay_hash.clone(),
                    canonical_sha256: replay_hash,
                    solver_version: "replay-paired-s1-levels-v1".into(),
                },
            });
        }

        mark_trajectory_discontinuities(reference_index, &matched, &mut invalid);
        references.push(PreparedReference {
            reference_index,
            matched,
            replay,
        });
        reference_plans.push(ReferencePlan {
            reference_index,
            reference_aperture: aperture(reference_index),
            target_mean_gray: target.mean,
            target_stddev_gray: target.stddev,
            matched_saturation_range: 0.0,
        });
    }

    for (prepared, plan) in references.iter().zip(&mut reference_plans) {
        let saturations: Vec<_> = prepared
            .matched
            .iter()
            .filter(|point| !invalid.contains_key(&point.record.source_aperture_index))
            .map(|point| render(&point.spec).covariates.saturation_fraction)
            .collect();
        plan.matched_saturation_range = saturation_range(&saturations);
        if plan.matched_saturation_range > MAX_SATURATION_RANGE {
            return Err(format!(
                "S1 ref{} saturation range {:.9} exceeded {:.9}",
                prepared.reference_index, plan.matched_saturation_range, MAX_SATURATION_RANGE
            ));
        }
    }

    let common_indices: Vec<_> = universe
        .iter()
        .copied()
        .filter(|index| !invalid.contains_key(index))
        .collect();
    if common_indices.len() < MIN_COMMON_INDICES
        || REFERENCE_INDICES
            .iter()
            .any(|reference| !common_indices.contains(reference))
    {
        return Err(format!(
            "common renderer-valid set was {:?} ({} of {}; refs {:?} required); exclusions: {:?}",
            common_indices,
            common_indices.len(),
            UNIVERSE_COUNT,
            REFERENCE_INDICES,
            invalid
        ));
    }

    let invalid_indices = invalid
        .into_iter()
        .map(|(index, reasons)| InvalidIndex {
            index,
            aperture: aperture(index),
            reasons,
        })
        .collect();
    let plan = TwoMomentPlan {
        version: VERSION,
        preregistration_commit: PREREGISTRATION_COMMIT,
        aperture_formula: "a_i = 1.30 * i / 40, frozen i=7..40",
        frozen_universe: universe,
        reference_indices: REFERENCE_INDICES,
        reference_apertures: [
            aperture(REFERENCE_INDICES[0]),
            aperture(REFERENCE_INDICES[1]),
        ],
        skin_bounds: SKIN_BOUNDS,
        sclera_bounds: SCLERA_BOUNDS,
        default_levels: [DEFAULT_SKIN, DEFAULT_SCLERA],
        solver_version: "two-moment-grid-65-refine17x3-v1",
        coarse_grid_points: COARSE_GRID_POINTS,
        refinement_levels: REFINEMENT_LEVELS,
        maximum_moment_residual_gray: MAX_MOMENT_RESIDUAL_GRAY,
        bound_margin_fraction: BOUND_MARGIN_FRACTION,
        maximum_saturation: MAX_SATURATION,
        maximum_saturation_range: MAX_SATURATION_RANGE,
        trajectory_jump_multiplier: TRAJECTORY_JUMP_MULTIPLIER,
        minimum_common_indices: MIN_COMMON_INDICES,
        references: reference_plans,
        invalid_indices,
        common_indices,
        preparation_repetitions: 2,
        renderer_decision: "go",
    };
    Ok(PreparedTwoMoment {
        plan,
        baseline,
        references,
    })
}

fn solve(basis: &PhotometricBasis, target: GrayMoments) -> Candidate {
    let skin_step = (SKIN_BOUNDS[1] - SKIN_BOUNDS[0]) / (COARSE_GRID_POINTS - 1) as f64;
    let sclera_step = (SCLERA_BOUNDS[1] - SCLERA_BOUNDS[0]) / (COARSE_GRID_POINTS - 1) as f64;
    let mut best = candidate(basis, target, DEFAULT_SKIN as f64, DEFAULT_SCLERA as f64);
    for skin_index in 0..COARSE_GRID_POINTS {
        let skin = SKIN_BOUNDS[0] + skin_step * skin_index as f64;
        for sclera_index in 0..COARSE_GRID_POINTS {
            let sclera = SCLERA_BOUNDS[0] + sclera_step * sclera_index as f64;
            let next = candidate(basis, target, skin, sclera);
            if candidate_better(next, best) {
                best = next;
            }
        }
    }

    let mut current_skin_step = skin_step;
    let mut current_sclera_step = sclera_step;
    for _ in 0..REFINEMENT_LEVELS {
        let refined_skin_step = current_skin_step / 4.0;
        let refined_sclera_step = current_sclera_step / 4.0;
        let center = best;
        for skin_offset in -REFINEMENT_RADIUS..=REFINEMENT_RADIUS {
            let skin = center.skin + skin_offset as f64 * refined_skin_step;
            if !(SKIN_BOUNDS[0]..=SKIN_BOUNDS[1]).contains(&skin) {
                continue;
            }
            for sclera_offset in -REFINEMENT_RADIUS..=REFINEMENT_RADIUS {
                let sclera = center.sclera + sclera_offset as f64 * refined_sclera_step;
                if !(SCLERA_BOUNDS[0]..=SCLERA_BOUNDS[1]).contains(&sclera) {
                    continue;
                }
                let next = candidate(basis, target, skin, sclera);
                if candidate_better(next, best) {
                    best = next;
                }
            }
        }
        let defaults = candidate(basis, target, DEFAULT_SKIN as f64, DEFAULT_SCLERA as f64);
        if candidate_better(defaults, best) {
            best = defaults;
        }
        current_skin_step = refined_skin_step;
        current_sclera_step = refined_sclera_step;
    }
    best
}

fn candidate(basis: &PhotometricBasis, target: GrayMoments, skin: f64, sclera: f64) -> Candidate {
    let applied_skin = skin as f32;
    let applied_sclera = sclera as f32;
    let moments = basis.moments(applied_skin, applied_sclera);
    let mean_residual = moments.mean - target.mean;
    let stddev_residual = moments.stddev - target.stddev;
    let objective = mean_residual * mean_residual + stddev_residual * stddev_residual;
    let skin_norm = (skin - DEFAULT_SKIN as f64) / (SKIN_BOUNDS[1] - SKIN_BOUNDS[0]);
    let sclera_norm = (sclera - DEFAULT_SCLERA as f64) / (SCLERA_BOUNDS[1] - SCLERA_BOUNDS[0]);
    Candidate {
        skin,
        sclera,
        applied_skin,
        applied_sclera,
        objective,
        default_distance: skin_norm * skin_norm + sclera_norm * sclera_norm,
    }
}

fn candidate_better(candidate: Candidate, current: Candidate) -> bool {
    candidate
        .objective
        .total_cmp(&current.objective)
        .then_with(|| {
            candidate
                .default_distance
                .total_cmp(&current.default_distance)
        })
        .then_with(|| candidate.skin.total_cmp(&current.skin))
        .then_with(|| candidate.sclera.total_cmp(&current.sclera))
        .is_lt()
}

fn spec(index: usize, skin: f32, sclera: f32) -> SyntheticEyeSpec {
    SyntheticEyeSpec {
        aperture: aperture(index),
        skin_level: skin,
        sclera_level: sclera,
        ..SyntheticEyeSpec::default()
    }
}

fn assert_basis_bound_corners(
    base_spec: &SyntheticEyeSpec,
    basis: &PhotometricBasis,
) -> Result<(), String> {
    for skin in SKIN_BOUNDS.map(|value| value as f32) {
        for sclera in SCLERA_BOUNDS.map(|value| value as f32) {
            let mut corner_spec = base_spec.clone();
            corner_spec.skin_level = skin;
            corner_spec.sclera_level = sclera;
            let predicted = basis.predict_pixels(skin, sclera);
            let canonical = render(&corner_spec);
            if predicted != canonical.pixels {
                return Err(format!(
                    "G2 photometric basis changed at bound corner skin={skin:.9} sclera={sclera:.9}"
                ));
            }
            let fast = basis.moments(skin, sclera);
            let checked = gray_moments(&canonical.pixels);
            if fast.mean.to_bits() != checked.mean.to_bits()
                || fast.stddev.to_bits() != checked.stddev.to_bits()
            {
                return Err(format!(
                    "G2 photometric basis moments changed at bound corner skin={skin:.9} sclera={sclera:.9}"
                ));
            }
        }
    }
    Ok(())
}

fn is_near_bound(level: f64, bounds: [f64; 2]) -> bool {
    let margin = BOUND_MARGIN_FRACTION * (bounds[1] - bounds[0]);
    (level - bounds[0]).min(bounds[1] - level) <= margin
}

fn finite_covariates(covariates: &ImageCovariates) -> bool {
    [
        covariates.mean,
        covariates.stddev,
        covariates.edge_energy,
        covariates.saturation_fraction,
        covariates.visible_area_fraction,
        covariates.measured_aperture_geometry,
        covariates.measured_aperture_raster,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn add_image_failures(
    index: usize,
    suite: &str,
    covariates: &ImageCovariates,
    eye_like: bool,
    invalid: &mut BTreeMap<usize, Vec<String>>,
) {
    if !eye_like {
        add_reason(index, format!("P3 {suite} not eye-like"), invalid);
    }
    if !finite_covariates(covariates) {
        add_reason(index, format!("P3 {suite} nonfinite covariate"), invalid);
    }
    if covariates.frame_truncated {
        add_reason(index, format!("P3 {suite} frame contact"), invalid);
    }
    if covariates.saturation_fraction > MAX_SATURATION {
        add_reason(
            index,
            format!(
                "P3 {suite} saturation {:.9} > {:.9}",
                covariates.saturation_fraction, MAX_SATURATION
            ),
            invalid,
        );
    }
}

fn add_reason(index: usize, reason: String, invalid: &mut BTreeMap<usize, Vec<String>>) {
    invalid.entry(index).or_default().push(reason);
}

fn mark_trajectory_discontinuities(
    reference_index: usize,
    matched: &[PreparedPoint],
    invalid: &mut BTreeMap<usize, Vec<String>>,
) {
    for (name, values) in [
        (
            "skin",
            matched
                .iter()
                .map(|point| point.record.solver_skin_f64.unwrap())
                .collect::<Vec<_>>(),
        ),
        (
            "sclera",
            matched
                .iter()
                .map(|point| point.record.solver_sclera_f64.unwrap())
                .collect::<Vec<_>>(),
        ),
    ] {
        let steps: Vec<_> = values
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();
        let nonzero: Vec<_> = steps.iter().copied().filter(|step| *step != 0.0).collect();
        let Some(median) = median(nonzero) else {
            continue;
        };
        for (offset, step) in steps.into_iter().enumerate() {
            if step > TRAJECTORY_JUMP_MULTIPLIER * median {
                let index = UNIVERSE_START + offset + 1;
                add_reason(
                    index,
                    format!(
                        "P4 ref{reference_index} {name} step {step:.12} > {} * median {median:.12}",
                        TRAJECTORY_JUMP_MULTIPLIER
                    ),
                    invalid,
                );
            }
        }
    }
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

fn saturation_range(values: &[f32]) -> f32 {
    if values.len() <= 1 {
        return 0.0;
    }
    values.iter().copied().reduce(f32::max).unwrap()
        - values.iter().copied().reduce(f32::min).unwrap()
}

fn assert_mirror(spec: &SyntheticEyeSpec) -> Result<(), String> {
    let pair = render_stereo(spec, spec, StereoPolicy::AnatomicalMirror);
    for y in 0..crate::renderer::SIDE {
        for x in 0..crate::renderer::SIDE {
            if pair.left.pixels[y * crate::renderer::SIDE + x]
                != pair.right.pixels[y * crate::renderer::SIDE + (crate::renderer::SIDE - 1 - x)]
            {
                return Err("anatomical right eye was not an exact pixel mirror".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_reference_indices_and_frozen_universe_are_exact() {
        assert_eq!(UNIVERSE_COUNT, 34);
        assert!((aperture(REFERENCE_INDICES[0]) - 0.4875).abs() <= f32::EPSILON);
        assert!((aperture(REFERENCE_INDICES[1]) - 1.0075).abs() <= f32::EPSILON);
    }

    #[test]
    fn candidate_ties_prefer_defaults_then_lower_levels() {
        let mut spec = SyntheticEyeSpec::default();
        spec.aperture = aperture(REFERENCE_INDICES[0]);
        let basis = PhotometricBasis::from_spec(&spec);
        let target = basis.moments(DEFAULT_SKIN, DEFAULT_SCLERA);
        let solved = solve(&basis, target);
        assert_eq!(solved.applied_skin.to_bits(), DEFAULT_SKIN.to_bits());
        assert_eq!(solved.applied_sclera.to_bits(), DEFAULT_SCLERA.to_bits());
    }

    #[test]
    fn median_convention_is_frozen_for_even_and_odd_counts() {
        assert_eq!(median(vec![]), None);
        assert_eq!(median(vec![3.0]), Some(3.0));
        assert_eq!(median(vec![4.0, 1.0, 3.0, 2.0]), Some(2.5));
    }

    #[test]
    fn near_bound_uses_one_percent_of_each_full_range() {
        assert!(is_near_bound(SKIN_BOUNDS[0] + 0.0029, SKIN_BOUNDS));
        assert!(!is_near_bound(SKIN_BOUNDS[0] + 0.0031, SKIN_BOUNDS));
        assert!(is_near_bound(SCLERA_BOUNDS[1] - 0.0029, SCLERA_BOUNDS));
        assert!(!is_near_bound(SCLERA_BOUNDS[1] - 0.0031, SCLERA_BOUNDS));
    }

    #[test]
    fn full_renderer_preparation_deterministically_reports_instrument_no_go() {
        let reason = prepare().expect_err("frozen dual-reference instrument is infeasible");
        assert!(
            reason.contains("common renderer-valid set was [18, 19, 20, 21, 22, 23, 24, 25, 26]")
        );
        assert!(reason.contains("9 of 34"));
        assert!(reason.contains("refs [15, 31] required"));
    }
}
