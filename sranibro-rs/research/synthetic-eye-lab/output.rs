use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use serde::Serialize;

use crate::experiment::CaseDefinition;
use crate::luminance::{aperture, LuminancePlan};
use crate::model::sha256_hex;
use crate::renderer::{RenderedStereo, SIDE};

pub const PRODUCTION_PRESENCE_GATE: f32 = 0.05;
pub const ANCHOR_PRESENCE_MARGIN: f32 = 0.10;
pub const BRIGHTNESS_CLIP_FLAG: f32 = 0.05;
pub const MIN_INTERPRETATION_COVERAGE: f32 = 0.80;
pub const MIN_CORRELATION_CASES: usize = 5;
const CASE_ID_SCHEMA_VERSION: &str = "synthetic-eye-case-v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Recognition {
    RecognizedSynthetic,
    NotRecognized,
    NonFinitePresence,
    NotEyeLike,
}

impl Recognition {
    fn as_str(self) -> &'static str {
        match self {
            Self::RecognizedSynthetic => "recognized_synthetic",
            Self::NotRecognized => "not_recognized",
            Self::NonFinitePresence => "nonfinite_presence",
            Self::NotEyeLike => "not_eye_like",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CaseResult {
    pub definition: CaseDefinition,
    pub case_id: String,
    pub rendered: RenderedStereo,
    pub tensor_sha256: String,
    pub raw: [f32; 5],
    pub recognition: Recognition,
}

impl CaseResult {
    pub fn all_finite(&self) -> bool {
        self.raw.iter().all(|value| value.is_finite())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase0Decision {
    Go,
    NoGo,
}

#[derive(Serialize)]
pub struct Manifest {
    pub tool: &'static str,
    pub suite_version: &'static str,
    pub generated_unix_s: u64,
    pub repository_commit: String,
    pub repository_dirty: bool,
    pub model_identity_sha256: String,
    pub model_bytes_written: bool,
    pub canonical_input: &'static str,
    pub input_normalization: &'static str,
    pub presence_gate: f32,
    pub anchor_presence_margin: f32,
    pub minimum_interpretation_coverage: f32,
    pub anchor_rule: &'static str,
    pub requested_experiment: &'static str,
    pub suites_evaluated: Vec<&'static str>,
    pub luminance_plan: Option<LuminancePlan>,
    pub luminance_renderer_no_go_reason: Option<String>,
    pub phase0_decision: Phase0Decision,
    pub total_cases: usize,
    pub recognized_cases: usize,
    pub sanitized_cli: Vec<String>,
    pub deterministic_scope: &'static str,
    pub interpretation_limit: &'static str,
}

pub fn classify(raw: [f32; 5], eye_like: bool) -> Recognition {
    if !raw[0].is_finite() {
        Recognition::NonFinitePresence
    } else if !eye_like {
        Recognition::NotEyeLike
    } else if raw[0] > PRODUCTION_PRESENCE_GATE {
        Recognition::RecognizedSynthetic
    } else {
        Recognition::NotRecognized
    }
}

pub fn case_id(definition: &CaseDefinition) -> Result<String, serde_json::Error> {
    let canonical = serde_json::to_vec(&(CASE_ID_SCHEMA_VERSION, definition))?;
    Ok(sha256_hex(&canonical)[..20].to_string())
}

pub fn phase0_decision(results: &[CaseResult]) -> Phase0Decision {
    let anchors: Vec<_> = results
        .iter()
        .filter(|result| result.definition.experiment == "anchor_family")
        .collect();
    if anchors.len() >= 3
        && anchors.iter().all(|result| {
            result.raw[0].is_finite()
                && result.raw[0] >= ANCHOR_PRESENCE_MARGIN
                && result.rendered.left.eye_like
                && result.rendered.right.eye_like
        })
    {
        Phase0Decision::Go
    } else {
        Phase0Decision::NoGo
    }
}

pub fn write_run(
    out_dir: &Path,
    manifest: &Manifest,
    results: &[CaseResult],
) -> Result<(), Box<dyn Error>> {
    if out_dir.exists() && std::fs::read_dir(out_dir)?.next().is_some() {
        return Err(format!(
            "output directory is not empty: {} (choose a new directory to avoid mixing runs)",
            out_dir.display()
        )
        .into());
    }
    let inputs = out_dir.join("inputs");
    std::fs::create_dir_all(&inputs)?;
    for result in results {
        write_png(
            &inputs.join(format!("{}_left.png", result.case_id)),
            &result.rendered.left.pixels,
        )?;
        write_png(
            &inputs.join(format!("{}_right.png", result.case_id)),
            &result.rendered.right.pixels,
        )?;
    }
    std::fs::write(
        out_dir.join("manifest.json"),
        serde_json::to_vec_pretty(manifest)?,
    )?;
    write_csv(&out_dir.join("results.csv"), results)?;
    std::fs::write(
        out_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summaries(results, manifest.luminance_plan.as_ref()))?,
    )?;
    if let Some(plan) = manifest.luminance_plan.as_ref() {
        if let Some(comparison) = luminance_comparison(results, plan) {
            std::fs::write(
                out_dir.join("luminance_analysis.json"),
                serde_json::to_vec_pretty(&comparison)?,
            )?;
        }
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct LuminanceComparison {
    matched_index_range: [usize; 2],
    matched_aperture_range: [f32; 2],
    planned_matched_cases: usize,
    paired_usable_cases: usize,
    a_spearman_lr_matched_indices: [Option<f32>; 2],
    b_spearman_lr: [Option<f32>; 2],
    c_spearman_lr: [Option<f32>; 2],
    d_spearman_lr: [Option<f32>; 2],
    a_open_span_lr_matched_indices: [Option<f32>; 2],
    d_open_span_lr: [Option<f32>; 2],
    paired_mean_absolute_a_minus_d_lr: [Option<f32>; 2],
    paired_mean_signed_a_minus_d_lr: [Option<f32>; 2],
    paired_max_absolute_a_minus_d_lr: [Option<f32>; 2],
    a_d_range_overlap_fraction_lr: [Option<f32>; 2],
    rubric_line_1_sign_pattern_observed: bool,
    interpretation_record: String,
    d_incremental_evidence_limit: &'static str,
    tested_aperture_scope: String,
    untested_near_closed_scope: String,
}

fn luminance_comparison(
    results: &[CaseResult],
    plan: &LuminancePlan,
) -> Option<LuminanceComparison> {
    let mut a_by_index = BTreeMap::<usize, &CaseResult>::new();
    let mut d_by_index = BTreeMap::<usize, &CaseResult>::new();
    for index in plan.selected_start_index..=plan.selected_end_index {
        let expected = aperture(index);
        if let Some(case) = results.iter().find(|case| {
            case.definition.experiment == "aperture_geometry"
                && case
                    .definition
                    .factor_x
                    .is_some_and(|value| value.to_bits() == expected.to_bits())
                && usable_for_interpretation(case)
        }) {
            a_by_index.insert(index, case);
        }
    }
    for case in results.iter().filter(|case| {
        case.definition.experiment == "fixed_geometry_original_mean_control"
            && usable_for_interpretation(case)
    }) {
        if let Some(record) = case.definition.mean_match.as_ref() {
            d_by_index.insert(record.source_aperture_index, case);
        }
    }

    let paired: Vec<_> = (plan.selected_start_index..=plan.selected_end_index)
        .filter_map(|index| Some((index, *a_by_index.get(&index)?, *d_by_index.get(&index)?)))
        .collect();
    if paired.is_empty() {
        return None;
    }
    let x: Vec<_> = paired
        .iter()
        .map(|(index, _, _)| aperture(*index))
        .collect();
    let a_left: Vec<_> = paired.iter().map(|(_, a, _)| a.raw[1]).collect();
    let a_right: Vec<_> = paired.iter().map(|(_, a, _)| a.raw[2]).collect();
    let d_left: Vec<_> = paired.iter().map(|(_, _, d)| d.raw[1]).collect();
    let d_right: Vec<_> = paired.iter().map(|(_, _, d)| d.raw[2]).collect();
    let a_spearman = [
        correlation(&ranks(&x), &ranks(&a_left)),
        correlation(&ranks(&x), &ranks(&a_right)),
    ];
    let d_spearman = [
        correlation(&ranks(&x), &ranks(&d_left)),
        correlation(&ranks(&x), &ranks(&d_right)),
    ];
    let b_spearman = experiment_spearman(results, "aperture_constant_mean");
    let c_spearman = experiment_spearman(results, "fixed_geometry_same_sclera_control");
    let rubric_line_1_sign_pattern_observed = b_spearman
        .iter()
        .all(|value| value.is_some_and(|value| value > 0.0))
        && c_spearman
            .iter()
            .all(|value| value.is_some_and(|value| value < 0.0));

    Some(LuminanceComparison {
        matched_index_range: [plan.selected_start_index, plan.selected_end_index],
        matched_aperture_range: plan.selected_aperture_range,
        planned_matched_cases: plan.selected_count,
        paired_usable_cases: paired.len(),
        a_spearman_lr_matched_indices: a_spearman,
        b_spearman_lr: b_spearman,
        c_spearman_lr: c_spearman,
        d_spearman_lr: d_spearman,
        a_open_span_lr_matched_indices: [span(&a_left), span(&a_right)],
        d_open_span_lr: [span(&d_left), span(&d_right)],
        paired_mean_absolute_a_minus_d_lr: [
            mean_pair_delta(&a_left, &d_left, true, false),
            mean_pair_delta(&a_right, &d_right, true, false),
        ],
        paired_mean_signed_a_minus_d_lr: [
            mean_pair_delta(&a_left, &d_left, false, false),
            mean_pair_delta(&a_right, &d_right, false, false),
        ],
        paired_max_absolute_a_minus_d_lr: [
            mean_pair_delta(&a_left, &d_left, true, true),
            mean_pair_delta(&a_right, &d_right, true, true),
        ],
        a_d_range_overlap_fraction_lr: [
            range_overlap_fraction(&a_left, &d_left),
            range_overlap_fraction(&a_right, &d_right),
        ],
        rubric_line_1_sign_pattern_observed,
        interpretation_record: if rubric_line_1_sign_pattern_observed {
            "Observed B-positive/C-negative signs match preregistered rubric line 1. No B residual effect-size threshold was preregistered, so the weak nonmonotone B residual is conservatively labelled inconclusive after inspection; this is an explicit post-hoc demotion, not a preregistered decision.".into()
        } else {
            "Observed B/C signs did not match preregistered rubric line 1; the result is anomalous/inconclusive with no mechanism claim.".into()
        },
        d_incremental_evidence_limit: "D mainly establishes that A's whole-image-mean trajectory is feasible within bounds and reproduces direction and magnitude through sclera. The positive direction was expected from the prior fixed-geometry sclera result and is not independent proof.",
        tested_aperture_scope: format!(
            "B/C/D cover indices {}..{} only ({:.9}..{:.9})",
            plan.selected_start_index,
            plan.selected_end_index,
            plan.selected_aperture_range[0],
            plan.selected_aperture_range[1]
        ),
        untested_near_closed_scope: if plan.selected_start_index == 0 {
            "none; the matched suites cover the full preregistered lower range".into()
        } else {
            format!(
                "indices 0..{} (aperture below {:.9}) were infeasible under fixed sclera bounds and remain untested by B/C/D",
                plan.selected_start_index - 1,
                plan.selected_aperture_range[0]
            )
        },
    })
}

fn experiment_spearman(results: &[CaseResult], experiment: &str) -> [Option<f32>; 2] {
    let cases: Vec<_> = results
        .iter()
        .filter(|case| case.definition.experiment == experiment && usable_for_interpretation(case))
        .collect();
    let x: Vec<_> = cases
        .iter()
        .filter_map(|case| case.definition.factor_x)
        .collect();
    let left: Vec<_> = cases.iter().map(|case| case.raw[1]).collect();
    let right: Vec<_> = cases.iter().map(|case| case.raw[2]).collect();
    [
        correlation(&ranks(&x), &ranks(&left)),
        correlation(&ranks(&x), &ranks(&right)),
    ]
}

fn mean_pair_delta(a: &[f32], b: &[f32], absolute: bool, maximum: bool) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let values = a.iter().zip(b).map(|(&a, &b)| {
        let delta = a - b;
        if absolute {
            delta.abs()
        } else {
            delta
        }
    });
    if maximum {
        values.reduce(f32::max)
    } else {
        Some(values.sum::<f32>() / a.len() as f32)
    }
}

fn range_overlap_fraction(a: &[f32], b: &[f32]) -> Option<f32> {
    let a_min = a.iter().copied().reduce(f32::min)?;
    let a_max = a.iter().copied().reduce(f32::max)?;
    let b_min = b.iter().copied().reduce(f32::min)?;
    let b_max = b.iter().copied().reduce(f32::max)?;
    let overlap = (a_max.min(b_max) - a_min.max(b_min)).max(0.0);
    let union = a_max.max(b_max) - a_min.min(b_min);
    (union > 0.0).then_some(overlap / union)
}

#[derive(Debug, Serialize)]
struct ExperimentSummary {
    experiment: String,
    total_cases: usize,
    preregistered_cases: usize,
    renderer_unmatched_cases: usize,
    recognized_eye_like_cases: usize,
    usable_cases: usize,
    usable_coverage: f32,
    analysis_cases: usize,
    analysis_scope: String,
    presence_min_all_finite: Option<f32>,
    presence_max_all_finite: Option<f32>,
    open_l_min_usable: Option<f32>,
    open_l_max_usable: Option<f32>,
    open_l_span_usable: Option<f32>,
    open_r_min_usable: Option<f32>,
    open_r_max_usable: Option<f32>,
    open_r_span_usable: Option<f32>,
    factor_x_name: Option<String>,
    factor_y_name: Option<String>,
    pearson_factor_vs_open_l: Option<f32>,
    pearson_factor_vs_open_r: Option<f32>,
    spearman_factor_vs_open_l: Option<f32>,
    spearman_factor_vs_open_r: Option<f32>,
    spearman_factor_vs_sclera_level: Option<f32>,
    mean_match_cases: usize,
    mean_match_at_bound_cases: usize,
    mean_match_at_bound_fraction: Option<f32>,
    mean_match_error_max: Option<f32>,
    mean_match_error_median: Option<f32>,
    interpretation_withheld: bool,
}

fn summaries(
    results: &[CaseResult],
    luminance_plan: Option<&LuminancePlan>,
) -> Vec<ExperimentSummary> {
    let mut groups = BTreeMap::<&str, Vec<&CaseResult>>::new();
    for result in results {
        groups
            .entry(&result.definition.experiment)
            .or_default()
            .push(result);
    }
    groups
        .into_iter()
        .map(|(experiment, cases)| {
            let (preregistered_cases, renderer_unmatched_cases) =
                if experiment == "fixed_geometry_original_mean_control" {
                    luminance_plan
                        .map(|plan| (plan.selected_count, plan.d_unmatched_count))
                        .unwrap_or((cases.len(), 0))
                } else {
                    (cases.len(), 0)
                };
            let recognized_eye_like_cases = cases
                .iter()
                .filter(|case| case.recognition == Recognition::RecognizedSynthetic)
                .count();
            let usable: Vec<_> = cases
                .iter()
                .filter(|case| usable_for_interpretation(case))
                .copied()
                .collect();
            let usable_coverage = usable.len() as f32 / preregistered_cases.max(1) as f32;
            let interpretation_withheld = usable_coverage < MIN_INTERPRETATION_COVERAGE;
            let analysis_usable: Vec<_> = if experiment == "aperture_geometry" {
                if let Some(plan) = luminance_plan {
                    usable
                        .iter()
                        .filter(|case| {
                            case.definition.factor_x.is_some_and(|value| {
                                value >= plan.selected_aperture_range[0]
                                    && value <= plan.selected_aperture_range[1]
                            })
                        })
                        .copied()
                        .collect()
                } else {
                    usable.clone()
                }
            } else {
                usable.clone()
            };
            let analysis_scope = if experiment == "aperture_geometry" {
                luminance_plan
                    .map(|plan| {
                        format!(
                            "luminance matched indices {}..{} only",
                            plan.selected_start_index, plan.selected_end_index
                        )
                    })
                    .unwrap_or_else(|| "all usable preregistered cases".into())
            } else {
                "all usable preregistered cases".into()
            };
            let finite_presence: Vec<_> = cases
                .iter()
                .map(|case| case.raw[0])
                .filter(|value| value.is_finite())
                .collect();
            let usable_open_l: Vec<_> = analysis_usable
                .iter()
                .map(|case| case.raw[1])
                .filter(|value| value.is_finite())
                .collect();
            let usable_open_r: Vec<_> = analysis_usable
                .iter()
                .map(|case| case.raw[2])
                .filter(|value| value.is_finite())
                .collect();
            let factor_x_name = cases
                .iter()
                .find_map(|case| case.definition.factor_x_name.clone());
            let factor_y_name = cases
                .iter()
                .find_map(|case| case.definition.factor_y_name.clone());
            let triples: Vec<_> = analysis_usable
                .iter()
                .filter_map(|case| {
                    Some((
                        case.definition.factor_x?,
                        case.raw[1].is_finite().then_some(case.raw[1])?,
                        case.raw[2].is_finite().then_some(case.raw[2])?,
                    ))
                })
                .collect();
            let (x, left, right): (Vec<_>, Vec<_>, Vec<_>) = triples.into_iter().fold(
                (Vec::new(), Vec::new(), Vec::new()),
                |mut acc, (x, left, right)| {
                    acc.0.push(x);
                    acc.1.push(left);
                    acc.2.push(right);
                    acc
                },
            );
            // A single flattened correlation is misleading for a 2D grid because each
            // x value is repeated across different y strata. The plotting script emits
            // the 2D response surface instead.
            let allow_correlation = !interpretation_withheld
                && factor_y_name.is_none()
                && x.len() >= MIN_CORRELATION_CASES;
            let factor_sclera: Vec<_> = usable
                .iter()
                .filter_map(|case| {
                    Some((
                        case.definition.factor_x?,
                        case.definition.mean_match.as_ref()?.sclera_level,
                    ))
                })
                .collect();
            let (sclera_x, sclera_level): (Vec<_>, Vec<_>) = factor_sclera.into_iter().unzip();
            let mean_match: Vec<_> = cases
                .iter()
                .filter_map(|case| case.definition.mean_match.as_ref())
                .collect();
            let mean_match_at_bound_cases = mean_match
                .iter()
                .filter(|record| record.within_one_u8_of_bound)
                .count();
            let mean_match_errors: Vec<_> = mean_match
                .iter()
                .filter_map(|record| record.absolute_error)
                .filter(|error| error.is_finite())
                .collect();
            ExperimentSummary {
                experiment: experiment.into(),
                total_cases: cases.len(),
                preregistered_cases,
                renderer_unmatched_cases,
                recognized_eye_like_cases,
                usable_cases: usable.len(),
                usable_coverage,
                analysis_cases: analysis_usable.len(),
                analysis_scope,
                presence_min_all_finite: finite_presence.iter().copied().reduce(f32::min),
                presence_max_all_finite: finite_presence.iter().copied().reduce(f32::max),
                open_l_min_usable: usable_open_l.iter().copied().reduce(f32::min),
                open_l_max_usable: usable_open_l.iter().copied().reduce(f32::max),
                open_l_span_usable: span(&usable_open_l),
                open_r_min_usable: usable_open_r.iter().copied().reduce(f32::min),
                open_r_max_usable: usable_open_r.iter().copied().reduce(f32::max),
                open_r_span_usable: span(&usable_open_r),
                factor_x_name,
                factor_y_name,
                pearson_factor_vs_open_l: allow_correlation
                    .then(|| correlation(&x, &left))
                    .flatten(),
                pearson_factor_vs_open_r: allow_correlation
                    .then(|| correlation(&x, &right))
                    .flatten(),
                spearman_factor_vs_open_l: allow_correlation
                    .then(|| correlation(&ranks(&x), &ranks(&left)))
                    .flatten(),
                spearman_factor_vs_open_r: allow_correlation
                    .then(|| correlation(&ranks(&x), &ranks(&right)))
                    .flatten(),
                spearman_factor_vs_sclera_level: (allow_correlation
                    && sclera_x.len() >= MIN_CORRELATION_CASES)
                    .then(|| correlation(&ranks(&sclera_x), &ranks(&sclera_level)))
                    .flatten(),
                mean_match_cases: mean_match.len(),
                mean_match_at_bound_cases,
                mean_match_at_bound_fraction: (!mean_match.is_empty())
                    .then_some(mean_match_at_bound_cases as f32 / mean_match.len() as f32),
                mean_match_error_max: mean_match_errors.iter().copied().reduce(f32::max),
                mean_match_error_median: median(mean_match_errors),
                interpretation_withheld,
            }
        })
        .collect()
}

fn median(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

fn span(values: &[f32]) -> Option<f32> {
    Some(values.iter().copied().reduce(f32::max)? - values.iter().copied().reduce(f32::min)?)
}

fn correlation(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.len() < 3 {
        return None;
    }
    let mean_a = a.iter().sum::<f32>() / a.len() as f32;
    let mean_b = b.iter().sum::<f32>() / b.len() as f32;
    let mut covariance = 0.0;
    let mut variance_a = 0.0;
    let mut variance_b = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        let da = a - mean_a;
        let db = b - mean_b;
        covariance += da * db;
        variance_a += da * da;
        variance_b += db * db;
    }
    let denom = (variance_a * variance_b).sqrt();
    (denom > 0.0 && denom.is_finite())
        .then_some(covariance / denom)
        .filter(|value| value.is_finite())
}

fn usable_for_interpretation(case: &CaseResult) -> bool {
    case.recognition == Recognition::RecognizedSynthetic
        && case.rendered.left.covariates.saturation_fraction <= BRIGHTNESS_CLIP_FLAG
        && case.rendered.right.covariates.saturation_fraction <= BRIGHTNESS_CLIP_FLAG
        && !case.rendered.left.covariates.frame_truncated
        && !case.rendered.right.covariates.frame_truncated
}

fn ranks(values: &[f32]) -> Vec<f32> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]].to_bits() == values[order[start]].to_bits() {
            end += 1;
        }
        let rank = (start + end - 1) as f32 * 0.5;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    ranks
}

pub fn write_png(path: &Path, pixels: &[u8]) -> Result<(), Box<dyn Error>> {
    if pixels.len() != SIDE * SIDE {
        return Err(format!(
            "PNG input has {} pixels, expected {}",
            pixels.len(),
            SIDE * SIDE
        )
        .into());
    }
    let file = File::create(path)?;
    let mut encoder = png::Encoder::new(BufWriter::new(file), SIDE as u32, SIDE as u32);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(pixels)?;
    Ok(())
}

fn write_csv(path: &Path, results: &[CaseResult]) -> Result<(), Box<dyn Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "experiment,case_name,case_id,interpretation_scope,stereo_policy,photometric_json,mean_match_json,source_aperture_index,source_aperture,match_target_mean,match_achieved_mean,match_abs_error,match_sclera_level,match_at_bound,factor_x_name,factor_x,factor_y_name,factor_y,left_spec_json,right_spec_json,requested_aperture_l,requested_aperture_r,measured_aperture_geometry_l,measured_aperture_geometry_r,measured_aperture_raster_l,measured_aperture_raster_r,presence,open_l,open_r,squeeze_l,squeeze_r,raw_bits_hex,finite,recognition,eye_like,usable_for_interpretation,input_sha256,mean_l,mean_r,stddev_l,stddev_r,edge_energy_l,edge_energy_r,saturation_l,saturation_r,visible_area_l,visible_area_r,brightness_saturated,frame_truncated_l,frame_truncated_r"
    )?;
    for result in results {
        let left_json = serde_json::to_string(&result.rendered.left_spec)?;
        let right_json = serde_json::to_string(&result.rendered.right_spec)?;
        let raw_bits = result
            .raw
            .iter()
            .map(|value| format!("{:08x}", value.to_bits()))
            .collect::<Vec<_>>()
            .join(";");
        let saturated = result.rendered.left.covariates.saturation_fraction > BRIGHTNESS_CLIP_FLAG
            || result.rendered.right.covariates.saturation_fraction > BRIGHTNESS_CLIP_FLAG;
        let mean_match = result.definition.mean_match.as_ref();
        let fields = vec![
            result.definition.experiment.clone(),
            result.definition.case_name.clone(),
            result.case_id.clone(),
            result.definition.interpretation_scope.clone(),
            serde_json::to_string(&result.definition.stereo_policy)?,
            serde_json::to_string(&result.definition.photometric)?,
            mean_match
                .map(serde_json::to_string)
                .transpose()?
                .unwrap_or_default(),
            mean_match
                .map(|record| record.source_aperture_index.to_string())
                .unwrap_or_default(),
            mean_match
                .map(|record| f(record.source_aperture))
                .unwrap_or_default(),
            mean_match
                .and_then(|record| record.target_mean)
                .map(f)
                .unwrap_or_default(),
            mean_match
                .map(|record| f(record.achieved_mean))
                .unwrap_or_default(),
            mean_match
                .and_then(|record| record.absolute_error)
                .map(f)
                .unwrap_or_default(),
            mean_match
                .map(|record| f(record.sclera_level))
                .unwrap_or_default(),
            mean_match
                .map(|record| record.within_one_u8_of_bound.to_string())
                .unwrap_or_default(),
            result.definition.factor_x_name.clone().unwrap_or_default(),
            result.definition.factor_x.map(f).unwrap_or_default(),
            result.definition.factor_y_name.clone().unwrap_or_default(),
            result.definition.factor_y.map(f).unwrap_or_default(),
            left_json,
            right_json,
            f(result.rendered.left_spec.aperture),
            f(result.rendered.right_spec.aperture),
            f(result.rendered.left.covariates.measured_aperture_geometry),
            f(result.rendered.right.covariates.measured_aperture_geometry),
            f(result.rendered.left.covariates.measured_aperture_raster),
            f(result.rendered.right.covariates.measured_aperture_raster),
            raw(result.raw[0]),
            raw(result.raw[1]),
            raw(result.raw[2]),
            raw(result.raw[3]),
            raw(result.raw[4]),
            raw_bits,
            result.all_finite().to_string(),
            result.recognition.as_str().into(),
            (result.rendered.left.eye_like && result.rendered.right.eye_like).to_string(),
            usable_for_interpretation(result).to_string(),
            result.tensor_sha256.clone(),
            f(result.rendered.left.covariates.mean),
            f(result.rendered.right.covariates.mean),
            f(result.rendered.left.covariates.stddev),
            f(result.rendered.right.covariates.stddev),
            f(result.rendered.left.covariates.edge_energy),
            f(result.rendered.right.covariates.edge_energy),
            f(result.rendered.left.covariates.saturation_fraction),
            f(result.rendered.right.covariates.saturation_fraction),
            f(result.rendered.left.covariates.visible_area_fraction),
            f(result.rendered.right.covariates.visible_area_fraction),
            saturated.to_string(),
            result.rendered.left.covariates.frame_truncated.to_string(),
            result.rendered.right.covariates.frame_truncated.to_string(),
        ];
        writeln!(
            writer,
            "{}",
            fields
                .into_iter()
                .map(|field| csv_escape(&field))
                .collect::<Vec<_>>()
                .join(",")
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn f(value: f32) -> String {
    format!("{value:.9}")
}

fn raw(value: f32) -> String {
    if value.is_nan() {
        "NaN".into()
    } else if value == f32::INFINITY {
        "+Infinity".into()
    } else if value == f32::NEG_INFINITY {
        "-Infinity".into()
    } else {
        f(value)
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

pub fn repository_state(repo: &Path) -> (String, bool) {
    let commit = git(repo, &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let dirty = git(repo, &["status", "--porcelain"])
        .map(|value| !value.is_empty())
        .unwrap_or(true);
    (commit, dirty)
}

fn git(repo: &Path, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(args)
        .current_dir(repo)
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

pub fn generated_unix_s() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::{apply_photometric, render_stereo, StereoPolicy, SyntheticEyeSpec};

    #[test]
    fn nonfinite_presence_is_preserved_and_never_recognized() {
        let raw = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0];
        assert_eq!(classify(raw, true), Recognition::NonFinitePresence);
        assert_eq!(raw[0].to_bits(), f32::NAN.to_bits());
        assert_eq!(raw[1].to_bits(), f32::INFINITY.to_bits());
        assert_eq!(raw[2].to_bits(), f32::NEG_INFINITY.to_bits());
    }

    #[test]
    fn production_presence_gate_is_strict() {
        assert_eq!(
            classify([0.05, 0.0, 0.0, 0.0, 0.0], true),
            Recognition::NotRecognized
        );
        assert_eq!(
            classify(
                [f32::from_bits(0.05f32.to_bits() + 1), 0.0, 0.0, 0.0, 0.0],
                true
            ),
            Recognition::RecognizedSynthetic
        );
    }

    #[test]
    fn csv_escaping_is_rfc4180_compatible_for_specs() {
        assert_eq!(csv_escape("plain"), "plain");
        assert_eq!(csv_escape("a,b"), "\"a,b\"");
        assert_eq!(csv_escape("a\"b"), "\"a\"\"b\"");
    }

    #[test]
    fn rank_correlation_handles_monotone_and_reversed_values() {
        let x = [0.0, 1.0, 2.0, 3.0];
        assert!((correlation(&ranks(&x), &ranks(&x)).unwrap() - 1.0).abs() < 1e-6);
        let reversed = [3.0, 2.0, 1.0, 0.0];
        assert!((correlation(&ranks(&x), &ranks(&reversed)).unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn written_png_decodes_to_the_exact_rendered_u8_pixels() {
        let rendered = render_stereo(
            &SyntheticEyeSpec::default(),
            &SyntheticEyeSpec::default(),
            StereoPolicy::AnatomicalMirror,
        );
        let path = std::env::temp_dir().join(format!(
            "sranibro-synthetic-eye-png-roundtrip-{}.png",
            std::process::id()
        ));
        write_png(&path, &rendered.left.pixels).unwrap();

        let decoder = png::Decoder::new(File::open(&path).unwrap());
        let mut reader = decoder.read_info().unwrap();
        let mut decoded = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut decoded).unwrap();
        assert_eq!(info.width, SIDE as u32);
        assert_eq!(info.height, SIDE as u32);
        assert_eq!(info.color_type, png::ColorType::Grayscale);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);
        assert_eq!(&decoded[..info.buffer_size()], &rendered.left.pixels);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn case_id_is_stable_for_the_default_anchor() {
        let definition = crate::experiment::phase0_cases()
            .into_iter()
            .find(|case| case.case_name == "anchor_center_100")
            .unwrap();
        assert_eq!(case_id(&definition).unwrap(), "ed3c2e7d8a3ea53626b6");
    }

    #[test]
    fn nonempty_output_directory_is_rejected_to_prevent_mixed_runs() {
        let path = std::env::temp_dir().join(format!(
            "sranibro-synthetic-eye-nonempty-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("old-result"), b"stale").unwrap();
        let manifest = Manifest {
            tool: "test",
            suite_version: "test",
            generated_unix_s: 0,
            repository_commit: "test".into(),
            repository_dirty: false,
            model_identity_sha256: "test".into(),
            model_bytes_written: false,
            canonical_input: "test",
            input_normalization: "test",
            presence_gate: PRODUCTION_PRESENCE_GATE,
            anchor_presence_margin: ANCHOR_PRESENCE_MARGIN,
            minimum_interpretation_coverage: MIN_INTERPRETATION_COVERAGE,
            anchor_rule: "test",
            requested_experiment: "test",
            suites_evaluated: vec!["phase0"],
            luminance_plan: None,
            luminance_renderer_no_go_reason: None,
            phase0_decision: Phase0Decision::NoGo,
            total_cases: 0,
            recognized_cases: 0,
            sanitized_cli: vec![],
            deterministic_scope: "test",
            interpretation_limit: "test",
        };
        let error = write_run(&path, &manifest, &[]).unwrap_err().to_string();
        assert!(error.contains("not empty"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn two_dimensional_grid_does_not_emit_a_misleading_flat_correlation() {
        let results: Vec<_> = crate::experiment::milestone1_cases()
            .into_iter()
            .filter(|definition| definition.experiment == "stretch_grid")
            .map(|definition| {
                let mut rendered = render_stereo(
                    &definition.left,
                    &definition.right,
                    definition.stereo_policy,
                );
                apply_photometric(&mut rendered.left, definition.photometric);
                apply_photometric(&mut rendered.right, definition.photometric);
                CaseResult {
                    case_id: case_id(&definition).unwrap(),
                    definition,
                    rendered,
                    tensor_sha256: "test".into(),
                    raw: [0.5, 0.5, 0.5, 0.0, 0.0],
                    recognition: Recognition::RecognizedSynthetic,
                }
            })
            .collect();
        let summary = summaries(&results, None).pop().unwrap();
        assert_eq!(summary.factor_x_name.as_deref(), Some("scale_x"));
        assert_eq!(summary.factor_y_name.as_deref(), Some("scale_y"));
        assert_eq!(summary.usable_cases, 20);
        assert!((summary.usable_coverage - MIN_INTERPRETATION_COVERAGE).abs() < f32::EPSILON);
        assert_eq!(summary.pearson_factor_vs_open_l, None);
        assert_eq!(summary.spearman_factor_vs_open_r, None);
        assert!(!summary.interpretation_withheld);
    }

    #[test]
    fn three_point_anchor_family_never_emits_scalar_correlations() {
        let results: Vec<_> = crate::experiment::phase0_cases()
            .into_iter()
            .filter(|definition| definition.experiment == "anchor_family")
            .map(|definition| {
                let rendered = render_stereo(
                    &definition.left,
                    &definition.right,
                    definition.stereo_policy,
                );
                let aperture = definition.factor_x.unwrap();
                CaseResult {
                    case_id: case_id(&definition).unwrap(),
                    definition,
                    rendered,
                    tensor_sha256: "test".into(),
                    raw: [0.7, aperture, aperture, 0.0, 0.0],
                    recognition: Recognition::RecognizedSynthetic,
                }
            })
            .collect();
        let summary = summaries(&results, None).pop().unwrap();
        assert_eq!(summary.usable_cases, 3);
        assert_eq!(summary.pearson_factor_vs_open_l, None);
        assert_eq!(summary.spearman_factor_vs_open_r, None);
    }

    #[test]
    fn luminance_d_summary_uses_preregistered_denominator_and_bound_report() {
        let (definitions, plan) = crate::experiment::luminance_match_cases().unwrap();
        let results: Vec<_> = definitions
            .into_iter()
            .filter(|definition| definition.experiment == "fixed_geometry_original_mean_control")
            .map(|definition| {
                let rendered = render_stereo(
                    &definition.left,
                    &definition.right,
                    definition.stereo_policy,
                );
                CaseResult {
                    case_id: case_id(&definition).unwrap(),
                    definition,
                    rendered,
                    tensor_sha256: "test".into(),
                    raw: [0.7, 0.5, 0.5, 0.0, 0.0],
                    recognition: Recognition::RecognizedSynthetic,
                }
            })
            .collect();
        let summary = summaries(&results, Some(&plan)).pop().unwrap();
        assert_eq!(summary.total_cases, plan.d_matched_count);
        assert_eq!(summary.preregistered_cases, plan.selected_count);
        assert_eq!(summary.renderer_unmatched_cases, plan.d_unmatched_count);
        assert!((summary.usable_coverage - plan.d_renderer_coverage).abs() < 1e-6);
        assert_eq!(summary.mean_match_cases, plan.d_matched_count);
        assert!(summary.mean_match_error_max.unwrap() <= crate::luminance::MAX_MEAN_ERROR);
    }
}
