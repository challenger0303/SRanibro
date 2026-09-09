//! Research-only counterfactual replay for the XR5 nine-target residual recording.
//!
//! The tool applies one bounded intervention at a time after the captured adaptive
//! brightness affine and before production geometry, then replays the exact EyeNet.
//! Discovery-train rows alone rank probes. Holdout rows are report-only and become
//! statistically consumed once displayed. No probe is applied to live configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

#[path = "xr5_recording.rs"]
mod xr5_recording;

use sranibro_rs::core::types::{FlattenParams, MlGeometry};
use sranibro_rs::geometry_calib::{GeometryDataset, SampleFamily};
use sranibro_rs::geometry_fitrun::{
    research_replay_detailed, ResearchCoordinateWarp, ResearchObservation, SpatialGainField,
};
use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

const CONTRACT: &str = "xr5-residual-counterfactual-v1";
const MIN_STABLE_ROWS_PER_PHASE: usize = 11;
const EXPECTED_TARGET_PHASES_PER_SPLIT: usize = 18;
const MIN_TRAIN_SLOW_PHASE_RHO: f32 = 0.50;

#[derive(Clone)]
struct Probe {
    name: String,
    family: &'static str,
    geometry: [MlGeometry; 2],
    post_flatten: FlattenParams,
    affine: [[f32; 2]; 2],
    field: [Option<SpatialGainField>; 2],
    warp: [Option<ResearchCoordinateWarp>; 2],
}

#[derive(Clone, Copy, Debug, Default)]
struct Anchor {
    closed: f32,
    span: f32,
    valid: bool,
}

#[derive(Clone, Debug, Default)]
struct ResidualMetrics {
    score: f32,
    valid: bool,
    eligible_phases: usize,
    anchors: [Anchor; 2],
    gaze_abs_error: [f32; 2],
    worst_target_open: [f32; 2],
    gaze_asymmetry: f32,
    slow_mae: [f32; 2],
    slow_rho: [f32; 2],
    blink_p10: [f32; 2],
    squeeze_p90: [f32; 2],
    presence_rate: f32,
}

struct ResultRow {
    probe: Probe,
    train: ResidualMetrics,
    holdout: ResidualMetrics,
    train_safe: bool,
    holdout_safe: bool,
}

struct Args {
    recording: PathBuf,
    model: PathBuf,
    out: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("xr5 residual counterfactual failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let loaded = xr5_recording::load_recording(&args.recording)?;
    let protocol = loaded.metadata.get("capture_protocol");
    if !matches!(
        protocol,
        Some(
            "xr5_landmark_residual_audit_v1"
                | "xr5_landmark_residual_audit_v2"
                | "xr5_landmark_residual_audit_v3"
        )
    ) {
        return Err(format!(
            "unsupported capture protocol {:?}; a complete XR5 residual recording is required",
            protocol
        ));
    }
    if loaded.metadata.get("capture_evidence_complete") != Some("true") {
        return Err("capture metadata does not declare complete phase evidence".into());
    }
    let model_bytes = std::fs::read(&args.model)
        .map_err(|error| format!("read model {}: {error}", args.model.display()))?;
    verify_model_fingerprint(&loaded.metadata, &model_bytes)?;
    let map = tvm_params::parse_map_bytes(&model_bytes)
        .map_err(|error| format!("parse EyeNet model: {error}"))?;
    let mut net = EyeNet::new(map).map_err(|error| format!("EyeNet model invalid: {error}"))?;

    std::fs::create_dir(&args.out)
        .map_err(|error| format!("create output {}: {error}", args.out.display()))?;
    let probes = probes(loaded.baseline);
    let probe_count = probes.len();
    println!(
        "loaded {} labelled stereo frames; evaluating {} isolated probes",
        loaded.dataset.samples.len(),
        probe_count
    );
    let mut rows = Vec::with_capacity(probes.len());
    for (index, probe) in probes.into_iter().enumerate() {
        println!("probe {}/{} {}", index + 1, probe_count, probe.name);
        let replay = research_replay_detailed(
            &mut net,
            &loaded.dataset,
            loaded.baseline,
            probe.geometry,
            loaded.mirrors,
            loaded.despeckle,
            loaded.flatten,
            probe.post_flatten,
            probe.affine,
            probe.field,
            probe.warp,
        );
        let anchors = train_anchors(&replay.observations);
        let train = residual_metrics(&replay.observations, &loaded.dataset, anchors, false);
        let holdout = residual_metrics(&replay.observations, &loaded.dataset, anchors, true);
        rows.push(ResultRow {
            probe,
            train,
            holdout,
            train_safe: false,
            holdout_safe: false,
        });
    }
    let baseline_train = rows
        .first()
        .map(|row| row.train.clone())
        .ok_or_else(|| "probe set is empty".to_owned())?;
    let baseline_holdout = rows[0].holdout.clone();
    let selection_allowed = baseline_train.eligible_phases == EXPECTED_TARGET_PHASES_PER_SPLIT
        && baseline_holdout.eligible_phases == EXPECTED_TARGET_PHASES_PER_SPLIT
        && baseline_train
            .slow_rho
            .iter()
            .all(|value| *value >= MIN_TRAIN_SLOW_PHASE_RHO);
    for row in &mut rows {
        row.train_safe = selection_allowed && safe_against(&row.train, &baseline_train);
        row.holdout_safe = safe_against(&row.holdout, &baseline_holdout);
    }
    write_results(&args, &rows, selection_allowed)?;
    write_report(&args, &rows, selection_allowed)?;
    println!("results={}", args.out.display());
    Ok(())
}

fn identity_probe(geometry: [MlGeometry; 2]) -> Probe {
    Probe {
        name: "identity".into(),
        family: "baseline",
        geometry,
        post_flatten: FlattenParams::default(),
        affine: [[1.0, 0.0]; 2],
        field: [None; 2],
        warp: [None; 2],
    }
}

fn probes(geometry: [MlGeometry; 2]) -> Vec<Probe> {
    let mut probes = vec![identity_probe(geometry)];
    for (name, affine) in [
        ("shared_bias_-12", [[1.0, -12.0]; 2]),
        ("shared_bias_+12", [[1.0, 12.0]; 2]),
        ("left_bias_-12", [[1.0, -12.0], [1.0, 0.0]]),
        ("left_bias_+12", [[1.0, 12.0], [1.0, 0.0]]),
        ("right_bias_-12", [[1.0, 0.0], [1.0, -12.0]]),
        ("right_bias_+12", [[1.0, 0.0], [1.0, 12.0]]),
    ] {
        let mut probe = identity_probe(geometry);
        probe.name = format!("postnorm_{name}");
        probe.family = "global_affine";
        probe.affine = affine;
        probes.push(probe);
    }
    for (radius, strength) in [(0.20, 0.35), (0.20, 0.70), (0.33, 0.35), (0.33, 0.70)] {
        let mut probe = identity_probe(geometry);
        probe.name = format!("postnorm_flatten_r{radius:.2}_s{strength:.2}");
        probe.family = "low_frequency_flatten";
        probe.post_flatten = FlattenParams {
            enabled: true,
            radius,
            strength,
        };
        probes.push(probe);
    }
    for (axis, value) in [
        ("horizontal", -0.12),
        ("horizontal", 0.12),
        ("vertical", -0.12),
        ("vertical", 0.12),
        ("horizontal_curve", -0.08),
        ("horizontal_curve", 0.08),
        ("vertical_curve", -0.08),
        ("vertical_curve", 0.08),
    ] {
        for (target, eyes) in [
            ("shared", [true, true]),
            ("left", [true, false]),
            ("right", [false, true]),
        ] {
            let mut field = SpatialGainField::default();
            match axis {
                "horizontal" => field.horizontal = value,
                "vertical" => field.vertical = value,
                "horizontal_curve" => field.horizontal_curve = value,
                _ => field.vertical_curve = value,
            }
            let mut probe = identity_probe(geometry);
            probe.name = format!("postnorm_field_{target}_{axis}_{value:+.2}");
            probe.family = "spatial_gain_field";
            probe.field = std::array::from_fn(|eye| eyes[eye].then_some(field));
            probes.push(probe);
        }
    }
    for (axis, value) in [
        ("vertical_bow", -0.08),
        ("vertical_bow", 0.08),
        ("radial_k1", -0.08),
        ("radial_k1", 0.08),
    ] {
        let warp = if axis == "vertical_bow" {
            ResearchCoordinateWarp {
                vertical_bow: value,
                radial_k1: 0.0,
            }
        } else {
            ResearchCoordinateWarp {
                vertical_bow: 0.0,
                radial_k1: value,
            }
        };
        let mut probe = identity_probe(geometry);
        probe.name = format!("postnorm_warp_shared_{axis}_{value:+.2}");
        probe.family = "coordinate_warp";
        probe.warp = [Some(warp); 2];
        probes.push(probe);
    }
    for (name, delta) in [
        ("vertical_-0.02", GeometryDelta::Vertical(-0.02)),
        ("vertical_+0.02", GeometryDelta::Vertical(0.02)),
        ("scale_y_-0.08", GeometryDelta::ScaleY(-0.08)),
        ("scale_y_+0.08", GeometryDelta::ScaleY(0.08)),
        ("rotation_mag_-6", GeometryDelta::RotationMagnitude(-6.0)),
        ("rotation_mag_+6", GeometryDelta::RotationMagnitude(6.0)),
    ] {
        let mut candidate = geometry;
        apply_geometry_delta(&mut candidate, delta);
        let mut probe = identity_probe(candidate);
        probe.name = format!("geometry_{name}");
        probe.family = "geometry";
        probes.push(probe);
    }
    probes
}

#[derive(Clone, Copy)]
enum GeometryDelta {
    Vertical(f32),
    ScaleY(f32),
    RotationMagnitude(f32),
}

fn apply_geometry_delta(geometry: &mut [MlGeometry; 2], delta: GeometryDelta) {
    for eye in geometry {
        match delta {
            GeometryDelta::Vertical(value) => {
                translate_crop_axis(&mut eye.crop_top, &mut eye.crop_bottom, value)
            }
            GeometryDelta::ScaleY(value) => eye.scale_y = (eye.scale_y + value).clamp(0.5, 2.0),
            GeometryDelta::RotationMagnitude(value) => {
                let sign = if eye.rotate_deg < 0.0 { -1.0 } else { 1.0 };
                eye.rotate_deg += sign * value;
            }
        }
    }
}

fn translate_crop_axis(leading: &mut f32, trailing: &mut f32, requested: f32) {
    let leading0 = leading.clamp(0.0, 0.8);
    let trailing0 = trailing.clamp(0.0, 0.8);
    let delta = requested.clamp(-leading0, trailing0);
    *leading = leading0 + delta;
    *trailing = trailing0 - delta;
}

fn train_anchors(observations: &[ResearchObservation]) -> [Anchor; 2] {
    std::array::from_fn(|eye| {
        let open = observations
            .iter()
            .filter(|row| {
                !row.kind.is_holdout() && row.stable && row.kind.family() == SampleFamily::Neutral
            })
            .map(|row| row.open[eye])
            .collect::<Vec<_>>();
        let closed = observations
            .iter()
            .filter(|row| {
                !row.kind.is_holdout()
                    && row.kind.family() == SampleFamily::SlowClose
                    && row.expected_open.is_some_and(|value| value <= 0.10)
            })
            .map(|row| row.open[eye])
            .collect::<Vec<_>>();
        let open = percentile(&open, 0.50).unwrap_or(0.0);
        let closed = percentile(&closed, 0.50).unwrap_or(0.0);
        let span = open - closed;
        Anchor {
            closed,
            span,
            valid: span.is_finite() && span > 0.001,
        }
    })
}

fn residual_metrics(
    observations: &[ResearchObservation],
    dataset: &GeometryDataset,
    anchors: [Anchor; 2],
    holdout: bool,
) -> ResidualMetrics {
    if anchors.iter().any(|anchor| !anchor.valid) {
        return ResidualMetrics {
            anchors,
            ..ResidualMetrics::default()
        };
    }
    let mut stable_per_phase = BTreeMap::<usize, usize>::new();
    for row in observations.iter().filter(|row| {
        row.kind.is_holdout() == holdout
            && row.stable
            && matches!(
                row.kind.family(),
                SampleFamily::Neutral | SampleFamily::GazeSweep
            )
    }) {
        *stable_per_phase.entry(row.phase_index).or_default() += 1;
    }
    let eligible = stable_per_phase
        .into_iter()
        .filter_map(|(phase, count)| (count >= MIN_STABLE_ROWS_PER_PHASE).then_some(phase))
        .collect::<BTreeSet<_>>();

    let mut gaze_error = [Vec::new(), Vec::new()];
    let mut target_values = [BTreeMap::<String, Vec<f32>>::new(), BTreeMap::new()];
    let mut asymmetry = Vec::new();
    let mut slow_expected = [Vec::new(), Vec::new()];
    let mut slow_output = [Vec::new(), Vec::new()];
    let mut slow_by_phase = [
        BTreeMap::<usize, (Vec<f32>, Vec<f32>)>::new(),
        BTreeMap::<usize, (Vec<f32>, Vec<f32>)>::new(),
    ];
    let mut blink = [Vec::new(), Vec::new()];
    let mut squeeze = [Vec::new(), Vec::new()];
    let mut present = 0usize;
    let mut seen = 0usize;

    for row in observations
        .iter()
        .filter(|row| row.kind.is_holdout() == holdout)
    {
        seen += 1;
        present += usize::from(row.presence.is_finite() && row.presence >= 0.02);
        let normalized: [f32; 2] =
            std::array::from_fn(|eye| (row.open[eye] - anchors[eye].closed) / anchors[eye].span);
        match row.kind.family() {
            SampleFamily::Neutral | SampleFamily::GazeSweep
                if row.stable && eligible.contains(&row.phase_index) =>
            {
                let target = dataset.samples[row.sample_index]
                    .commanded_target
                    .map(|value| value.as_str().to_owned())
                    .unwrap_or_else(|| "unlabelled".into());
                for eye in 0..2 {
                    if normalized[eye].is_finite() {
                        gaze_error[eye].push((normalized[eye] - 1.0).abs());
                        target_values[eye]
                            .entry(target.clone())
                            .or_default()
                            .push(normalized[eye]);
                    }
                    if row.squeeze[eye].is_finite() {
                        squeeze[eye].push(row.squeeze[eye]);
                    }
                }
                if normalized.iter().all(|value| value.is_finite()) {
                    asymmetry.push((normalized[0] - normalized[1]).abs());
                }
            }
            SampleFamily::SlowClose => {
                if let Some(expected) = row.expected_open.filter(|value| value.is_finite()) {
                    for eye in 0..2 {
                        if normalized[eye].is_finite() {
                            slow_expected[eye].push(expected);
                            slow_output[eye].push(normalized[eye]);
                            let phase = slow_by_phase[eye].entry(row.phase_index).or_default();
                            phase.0.push(expected);
                            phase.1.push(normalized[eye]);
                        }
                    }
                }
            }
            SampleFamily::NaturalBlinks => {
                for eye in 0..2 {
                    if normalized[eye].is_finite() {
                        blink[eye].push(normalized[eye]);
                    }
                }
            }
            _ => {}
        }
    }

    let gaze_abs_error =
        std::array::from_fn(|eye| percentile(&gaze_error[eye], 0.50).unwrap_or(f32::INFINITY));
    let worst_target_open = std::array::from_fn(|eye| {
        target_values[eye]
            .values()
            .filter_map(|values| percentile(values, 0.50))
            .min_by(f32::total_cmp)
            .unwrap_or(f32::NEG_INFINITY)
    });
    let slow_mae = std::array::from_fn(|eye| {
        let errors = slow_expected[eye]
            .iter()
            .zip(&slow_output[eye])
            .map(|(expected, output)| (expected - output).abs())
            .collect::<Vec<_>>();
        percentile(&errors, 0.50).unwrap_or(f32::INFINITY)
    });
    // Score each instructed close/open cycle independently so a target-specific
    // openness offset cannot manufacture or erase trajectory compliance.
    let slow_rho = std::array::from_fn(|eye| {
        let correlations = slow_by_phase[eye]
            .values()
            .filter_map(|(expected, output)| spearman(expected, output))
            .collect::<Vec<_>>();
        percentile(&correlations, 0.50).unwrap_or(f32::NEG_INFINITY)
    });
    let blink_p10 =
        std::array::from_fn(|eye| percentile(&blink[eye], 0.10).unwrap_or(f32::INFINITY));
    let squeeze_p90 =
        std::array::from_fn(|eye| percentile(&squeeze[eye], 0.90).unwrap_or(f32::INFINITY));
    let gaze_asymmetry = percentile(&asymmetry, 0.50).unwrap_or(f32::INFINITY);
    let worst_drop = worst_target_open.map(|value| (1.0 - value).max(0.0));
    let score = mean2(gaze_abs_error)
        + 0.50 * mean2(worst_drop)
        + 0.35 * gaze_asymmetry
        + 0.20 * mean2(slow_mae)
        + 0.10 * mean2(slow_rho.map(|value| (1.0 - value).max(0.0)));
    let valid = eligible.len() >= 9
        && gaze_abs_error.iter().all(|value| value.is_finite())
        && slow_mae.iter().all(|value| value.is_finite())
        && slow_rho.iter().all(|value| value.is_finite());
    ResidualMetrics {
        score,
        valid,
        eligible_phases: eligible.len(),
        anchors,
        gaze_abs_error,
        worst_target_open,
        gaze_asymmetry,
        slow_mae,
        slow_rho,
        blink_p10,
        squeeze_p90,
        presence_rate: if seen == 0 {
            0.0
        } else {
            present as f32 / seen as f32
        },
    }
}

fn safe_against(candidate: &ResidualMetrics, baseline: &ResidualMetrics) -> bool {
    candidate.valid
        && baseline.valid
        && candidate.presence_rate + 0.01 >= baseline.presence_rate
        && (0..2).all(|eye| {
            candidate.anchors[eye].span >= baseline.anchors[eye].span * 0.75
                && candidate.slow_rho[eye] + 0.05 >= baseline.slow_rho[eye]
                && candidate.slow_mae[eye] <= baseline.slow_mae[eye] + 0.05
                && candidate.blink_p10[eye] <= baseline.blink_p10[eye] + 0.10
                && candidate.squeeze_p90[eye] <= baseline.squeeze_p90[eye] + 0.03
        })
}

fn write_results(args: &Args, rows: &[ResultRow], selection_allowed: bool) -> Result<(), String> {
    let mut text = String::from(
        "name,family,selection_allowed,train_safe,holdout_safe,train_score,holdout_score,train_eligible_phases,holdout_eligible_phases,train_gaze_abs_l,train_gaze_abs_r,holdout_gaze_abs_l,holdout_gaze_abs_r,train_worst_target_open_l,train_worst_target_open_r,holdout_worst_target_open_l,holdout_worst_target_open_r,train_asymmetry,holdout_asymmetry,train_slow_mae_l,train_slow_mae_r,holdout_slow_mae_l,holdout_slow_mae_r,train_slow_rho_l,train_slow_rho_r,holdout_slow_rho_l,holdout_slow_rho_r,train_blink_p10_l,train_blink_p10_r,holdout_blink_p10_l,holdout_blink_p10_r,train_squeeze_p90_l,train_squeeze_p90_r,holdout_squeeze_p90_l,holdout_squeeze_p90_r,train_span_l,train_span_r,train_presence,holdout_presence\n",
    );
    for row in rows {
        let t = &row.train;
        let h = &row.holdout;
        let fields = vec![
            row.probe.name.clone(),
            row.probe.family.into(),
            selection_allowed.to_string(),
            row.train_safe.to_string(),
            row.holdout_safe.to_string(),
            f(t.score),
            f(h.score),
            t.eligible_phases.to_string(),
            h.eligible_phases.to_string(),
            f(t.gaze_abs_error[0]),
            f(t.gaze_abs_error[1]),
            f(h.gaze_abs_error[0]),
            f(h.gaze_abs_error[1]),
            f(t.worst_target_open[0]),
            f(t.worst_target_open[1]),
            f(h.worst_target_open[0]),
            f(h.worst_target_open[1]),
            f(t.gaze_asymmetry),
            f(h.gaze_asymmetry),
            f(t.slow_mae[0]),
            f(t.slow_mae[1]),
            f(h.slow_mae[0]),
            f(h.slow_mae[1]),
            f(t.slow_rho[0]),
            f(t.slow_rho[1]),
            f(h.slow_rho[0]),
            f(h.slow_rho[1]),
            f(t.blink_p10[0]),
            f(t.blink_p10[1]),
            f(h.blink_p10[0]),
            f(h.blink_p10[1]),
            f(t.squeeze_p90[0]),
            f(t.squeeze_p90[1]),
            f(h.squeeze_p90[0]),
            f(h.squeeze_p90[1]),
            f(t.anchors[0].span),
            f(t.anchors[1].span),
            f(t.presence_rate),
            f(h.presence_rate),
        ];
        text.push_str(&fields.join(","));
        text.push('\n');
    }
    std::fs::write(args.out.join("counterfactuals.csv"), text)
        .map_err(|error| format!("write counterfactuals.csv: {error}"))
}

fn write_report(args: &Args, rows: &[ResultRow], selection_allowed: bool) -> Result<(), String> {
    let baseline = rows.first().ok_or_else(|| "missing baseline".to_owned())?;
    let mut ranked = rows.iter().skip(1).collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.train.score.total_cmp(&right.train.score));
    let mut report = format!(
        "# XR5 residual counterfactual replay\n\nResearch-only. No live or saved configuration was changed. Lower score is better. Probes were ranked on discovery-train only; displayed holdout is report-only and is now consumed for later hypothesis design.\n\n## Baseline\n\n- train score {:.4}; holdout score {:.4}\n- gaze absolute error L/R train {:.3}/{:.3}, holdout {:.3}/{:.3}\n- worst target openness L/R train {:.3}/{:.3}, holdout {:.3}/{:.3}\n- slow-close rho L/R train {:.3}/{:.3}, holdout {:.3}/{:.3}\n- eligible stable target phases train/holdout {}/{}\n\n## Train-ranked safe single interventions\n\n| Probe | Family | Train | Δtrain | Holdout | Δholdout | Holdout safe |\n|---|---|---:|---:|---:|---:|---|\n",
        baseline.train.score,
        baseline.holdout.score,
        baseline.train.gaze_abs_error[0], baseline.train.gaze_abs_error[1],
        baseline.holdout.gaze_abs_error[0], baseline.holdout.gaze_abs_error[1],
        baseline.train.worst_target_open[0], baseline.train.worst_target_open[1],
        baseline.holdout.worst_target_open[0], baseline.holdout.worst_target_open[1],
        baseline.train.slow_rho[0], baseline.train.slow_rho[1],
        baseline.holdout.slow_rho[0], baseline.holdout.slow_rho[1],
        baseline.train.eligible_phases, baseline.holdout.eligible_phases,
    );
    let decision = format!(
        "## Decision\n\n- Candidate selection allowed: **{selection_allowed}**\n- Required: all {EXPECTED_TARGET_PHASES_PER_SPLIT} stable target phases in both splits and median train slow-close phase rho >= {MIN_TRAIN_SLOW_PHASE_RHO:.2} in both eyes.\n- When this gate is false, every ranking below is diagnostic only.\n\n## Baseline"
    );
    report = report.replacen("## Baseline", &decision, 1);
    report = report.replace(
        "## Train-ranked safe single interventions",
        "## Diagnostic train ranking",
    );
    report = report.replace(
        "slow-close rho L/R",
        "median phase-local slow-close rho L/R",
    );
    report = report
        .lines()
        .map(|line| {
            if line.starts_with("| Probe |") {
                "| Probe | Family | Train | delta train | Holdout | delta holdout | Holdout guard |"
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    report.push('\n');
    for row in ranked.iter().take(12) {
        report.push_str(&format!(
            "| {} | {} | {:.4} | {:+.4} | {:.4} | {:+.4} | {} |\n",
            row.probe.name,
            row.probe.family,
            row.train.score,
            row.train.score - baseline.train.score,
            row.holdout.score,
            row.holdout.score - baseline.holdout.score,
            row.holdout_safe,
        ));
    }
    report.push_str(
        "\n## Interpretation limits\n\n- This is one user/session and cannot establish transfer.\n- Three discovery target phases lacked enough stable rows; only eligible phases enter the score.\n- A successful gain field identifies a useful intervention class, not proof of a photometric physical cause.\n- No combination search was run. A single-axis result must be repeated on independent re-wears before freezing a combined candidate.\n",
    );
    std::fs::write(args.out.join("report.md"), report)
        .map_err(|error| format!("write report.md: {error}"))?;
    std::fs::write(
        args.out.join("manifest.txt"),
        format!(
            "research_only=true\ncontract={CONTRACT}\nrecording={}\nmodel={}\nprobes={}\nselection_allowed={}\nselection=train_only_when_gate_passes\nholdout=report_only_consumed_after_display\nintervention_seam=after_captured_adaptive_brightness_before_geometry\nproduction_config_changed=false\n",
            args.recording.display(),
            args.model.display(),
            rows.len(),
            selection_allowed,
        ),
    )
    .map_err(|error| format!("write manifest.txt: {error}"))
}

fn verify_model_fingerprint(
    metadata: &xr5_recording::RecordingMetadata,
    bytes: &[u8],
) -> Result<(), String> {
    let crc = format!("{:08x}", crc32_ieee(bytes));
    let recorded_crc = metadata.get("eyelid_model_crc32");
    let recorded_bytes = metadata
        .get("eyelid_model_bytes")
        .and_then(|value| value.parse::<usize>().ok());
    if recorded_crc != Some(crc.as_str()) || recorded_bytes != Some(bytes.len()) {
        return Err(format!(
            "EyeNet fingerprint mismatch: recording {:?}/{:?}, supplied {}/{}",
            recorded_crc,
            recorded_bytes,
            crc,
            bytes.len()
        ));
    }
    Ok(())
}

fn parse_args() -> Result<Args, String> {
    const USAGE: &str =
        "Usage: xr5-residual-counterfactual --recording <dir> --model <params> --out <new-dir>";
    let mut recording = None;
    let mut model = None;
    let mut out = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(raw) = args.next() {
        let flag = raw.to_string_lossy();
        let mut value = || {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{flag} requires a path\n{USAGE}"))
        };
        match flag.as_ref() {
            "--recording" => recording = Some(value()?),
            "--model" => model = Some(value()?),
            "--out" => out = Some(value()?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    let args = Args {
        recording: recording.ok_or_else(|| format!("missing --recording\n{USAGE}"))?,
        model: model.ok_or_else(|| format!("missing --model\n{USAGE}"))?,
        out: out.ok_or_else(|| format!("missing --out\n{USAGE}"))?,
    };
    if !args.recording.is_dir() || !args.model.is_file() {
        return Err("recording directory or model file does not exist".into());
    }
    if args.out.exists() {
        return Err(format!("output already exists: {}", args.out.display()));
    }
    Ok(args)
}

fn percentile(values: &[f32], quantile: f32) -> Option<f32> {
    let mut values = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f32;
    let lo = position.floor() as usize;
    let hi = position.ceil() as usize;
    let t = position - lo as f32;
    Some(values[lo] * (1.0 - t) + values[hi] * t)
}

fn spearman(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let left = average_ranks(left);
    let right = average_ranks(right);
    let lm = left.iter().sum::<f32>() / left.len() as f32;
    let rm = right.iter().sum::<f32>() / right.len() as f32;
    let covariance = left
        .iter()
        .zip(&right)
        .map(|(l, r)| (l - lm) * (r - rm))
        .sum::<f32>();
    let lv = left.iter().map(|value| (value - lm).powi(2)).sum::<f32>();
    let rv = right.iter().map(|value| (value - rm).powi(2)).sum::<f32>();
    let denominator = (lv * rv).sqrt();
    (denominator > 1.0e-12).then_some(covariance / denominator)
}

fn average_ranks(values: &[f32]) -> Vec<f32> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| values[*left].total_cmp(&values[*right]));
    let mut ranks = vec![0.0; values.len()];
    let mut begin = 0;
    while begin < order.len() {
        let mut end = begin + 1;
        while end < order.len() && values[order[end]] == values[order[begin]] {
            end += 1;
        }
        let rank = ((begin + 1) as f32 + end as f32) * 0.5;
        for index in begin..end {
            ranks[order[index]] = rank;
        }
        begin = end;
    }
    ranks
}

fn mean2(values: [f32; 2]) -> f32 {
    (values[0] + values[1]) * 0.5
}

fn f(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.9}")
    } else {
        String::new()
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_and_ignores_nonfinite() {
        assert_eq!(percentile(&[1.0, f32::NAN, 3.0], 0.5), Some(2.0));
    }

    #[test]
    fn spearman_recovers_monotonic_order() {
        assert!(spearman(&[0.0, 1.0, 2.0], &[5.0, 7.0, 9.0]).is_some_and(|v| v > 0.99));
    }

    #[test]
    fn probe_inventory_is_stable_and_identity_first() {
        let probes = probes([MlGeometry::default(); 2]);
        assert_eq!(
            probes.first().map(|probe| probe.name.as_str()),
            Some("identity")
        );
        assert_eq!(probes.len(), 45);
    }

    #[test]
    fn mirrored_rotation_magnitude_preserves_signs() {
        let mut geometry = [MlGeometry::default(); 2];
        geometry[0].rotate_deg = -30.0;
        geometry[1].rotate_deg = 30.0;
        apply_geometry_delta(&mut geometry, GeometryDelta::RotationMagnitude(6.0));
        assert_eq!(geometry[0].rotate_deg, -36.0);
        assert_eq!(geometry[1].rotate_deg, 36.0);
    }
}
