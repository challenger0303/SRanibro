//! Research-only replay of an exported Safe Geometry Fit recording.
//!
//! This compares bounded post-normalization photometric, low-frequency flattening,
//! affine-geometry, and landmark-independent coordinate-warp counterfactuals with the
//! exact production EyeNet scorer. Recorded preprocessing and adaptive-brightness
//! affines are reconstructed first; research interventions are then applied at an
//! explicit seam before geometry. It never changes SRanibro configuration or model
//! bytes and never writes camera images.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

#[path = "xr5_recording.rs"]
mod xr5_recording;

use sha2::{Digest, Sha256};
use sranibro_rs::core::types::{DespeckleParams, FlattenParams, MlGeometry};
use sranibro_rs::geometry_calib::GeometryDataset;
use sranibro_rs::geometry_fitrun::{
    research_candidate_admissible, research_evaluate_photometric, research_stability_report,
    GeometryMetrics, ResearchCoordinateWarp, SpatialGainField, StabilityReport,
};
use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

use xr5_recording::load_recording;

const REPLAY_CONTRACT: &str = "xr5-photometric-replay-v3";

#[derive(Clone)]
struct Probe {
    name: String,
    geometry: [MlGeometry; 2],
    post_flatten: FlattenParams,
    affine: [[f32; 2]; 2],
    field: [Option<SpatialGainField>; 2],
    warp: [Option<ResearchCoordinateWarp>; 2],
}

struct ResultRow {
    probe: Probe,
    train: GeometryMetrics,
    holdout: GeometryMetrics,
    train_safe: bool,
    holdout_safe: bool,
}

struct Args {
    recording: PathBuf,
    model: PathBuf,
    out: PathBuf,
    phase0_only: bool,
    confirmation: bool,
}

struct ReplayParity {
    status: &'static str,
    parity_selection_allowed: bool,
    detail: String,
}

struct ArtifactIdentity {
    recording_evidence_sha256: String,
    metadata_sha256: String,
    samples_sha256: String,
    model_sha256: String,
    model_bytes: u64,
    executable_sha256: String,
    parity_status: &'static str,
    parity_selection_allowed: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("xr5 photometric replay failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    std::fs::create_dir(&args.out)
        .map_err(|error| format!("create new output {}: {error}", args.out.display()))?;
    let recording = load_recording(&args.recording)?;
    let bytes = std::fs::read(&args.model)
        .map_err(|error| format!("read model {}: {error}", args.model.display()))?;
    let parity = replay_parity(&recording.metadata, &bytes);
    // Bind every replay result to the exact evidence, network and executable before
    // moving any fields out of LoadedRecording. Paths remain useful for humans, but
    // only these content identities are authoritative.
    let identity = ArtifactIdentity {
        recording_evidence_sha256: hash_recording_evidence(&args.recording, &recording.frames)?,
        metadata_sha256: hash_file(&args.recording.join("metadata.txt"))?,
        samples_sha256: hash_file(&args.recording.join("samples.csv"))?,
        model_sha256: hash_bytes(&bytes),
        model_bytes: bytes.len() as u64,
        executable_sha256: hash_current_executable()?,
        parity_status: parity.status,
        parity_selection_allowed: parity.parity_selection_allowed,
    };
    std::fs::write(
        args.out.join("replay_parity.txt"),
        format!(
            "status={}\nparity_selection_allowed={}\ndetail={}\n",
            parity.status, parity.parity_selection_allowed, parity.detail
        ),
    )
    .map_err(|error| format!("write replay_parity.txt: {error}"))?;
    println!("replay parity {}: {}", parity.status, parity.detail);
    let map = tvm_params::parse_map_bytes(&bytes)
        .map_err(|error| format!("parse EyeNet model: {error}"))?;
    let mut net = EyeNet::new(map).map_err(|error| format!("EyeNet model invalid: {error}"))?;

    let dataset = recording.dataset;
    let mirrors = recording.mirrors;
    println!("loaded {} labelled stereo frames", dataset.samples.len());
    let baseline_geometry = recording.baseline;
    let despeckle = recording.despeckle;
    let identity_affine = [[1.0, 0.0]; 2];
    let captured_flatten = recording.flatten;
    let no_post_flatten = FlattenParams::default();
    let stability = research_stability_report(&dataset, baseline_geometry);
    let baseline_probe = Probe {
        name: "identity".into(),
        geometry: baseline_geometry,
        post_flatten: no_post_flatten,
        affine: identity_affine,
        field: [None; 2],
        warp: [None; 2],
    };
    let (baseline_train, baseline_holdout) = evaluate(
        &mut net,
        &dataset,
        &baseline_probe,
        baseline_geometry,
        &stability,
        mirrors,
        despeckle,
        captured_flatten,
    );
    for note in &stability.notes {
        println!("capture evidence: {note}");
    }
    println!(
        "baseline train={:.4} holdout={:.4} gaze L/R={:.3}/{:.3} squeeze={:.3}/{:.3}",
        baseline_train.score,
        baseline_holdout.score,
        baseline_holdout.gaze_retention[0],
        baseline_holdout.gaze_retention[1],
        baseline_holdout.gaze_squeeze_fp[0],
        baseline_holdout.gaze_squeeze_fp[1],
    );
    if args.phase0_only {
        write_results(
            &args,
            &identity,
            false,
            "DIAGNOSTIC_PHASE0_ONLY",
            "phase-0 baseline replay does not evaluate or permit candidate selection",
            &baseline_probe,
            &baseline_train,
            &baseline_holdout,
            &[],
        )?;
        println!("phase-0 only; no candidate probes were evaluated");
        return Ok(());
    }
    if !parity.parity_selection_allowed {
        write_results(
            &args,
            &identity,
            false,
            "REJECTED_PARITY",
            &parity.detail,
            &baseline_probe,
            &baseline_train,
            &baseline_holdout,
            &[],
        )?;
        return Err(format!(
            "candidate selection/confirmation is disabled: {}. Baseline replay was written as diagnostic-only evidence",
            parity.detail
        ));
    }
    if args.confirmation {
        // The legacy fixed values were discovered while candidate flatten/fields
        // lived before the captured adaptive-brightness affine. Moving candidates
        // to the honest post-normalization seam changes the intervention itself, so
        // those values cannot be treated as a preregistered confirmation here.
        write_results(
            &args,
            &identity,
            false,
            "REJECTED_CONFIRMATION_INVALIDATED",
            "the legacy preregistration describes a different intervention seam and cannot select or confirm a candidate",
            &baseline_probe,
            &baseline_train,
            &baseline_holdout,
            &[],
        )?;
        return Err(
            "confirmation is intentionally disabled: the legacy frozen probes were invalidated by the post-normalization seam. Run discovery on independent matching recordings, freeze a new postnorm candidate outside this tool, then add a sealed confirmation specification"
                .into(),
        );
    }
    if !args.confirmation
        && (stability.invalid_static_phases >= 2
            || stability
                .valid_closed_phases
                .iter()
                .any(|count| *count == 0))
    {
        let evidence_reason = format!(
            "capture evidence cannot rank candidates: invalid static phases={}, valid Closed train/holdout={}/{}",
            stability.invalid_static_phases,
            stability.valid_closed_phases[0],
            stability.valid_closed_phases[1],
        );
        write_results(
            &args,
            &identity,
            false,
            "REJECTED_CAPTURE_EVIDENCE",
            &evidence_reason,
            &baseline_probe,
            &baseline_train,
            &baseline_holdout,
            &[],
        )?;
        return Err(format!(
            "exploration stopped: capture evidence is not fit for ranking \
             (invalid static phases={}, valid Closed train/holdout={}/{})",
            stability.invalid_static_phases,
            stability.valid_closed_phases[0],
            stability.valid_closed_phases[1],
        ));
    }
    let mut probes = photometric_probes(baseline_geometry, no_post_flatten);
    probes.extend(geometry_probes(
        baseline_geometry,
        mirrors,
        identity_affine,
        no_post_flatten,
        "geometry",
    ));
    let phase_a_len = probes.len();
    let mut rows = Vec::new();
    for (index, probe) in probes.into_iter().enumerate() {
        println!("phase A {}/{} {}", index + 1, phase_a_len, probe.name);
        rows.push(evaluate_row(
            &mut net,
            &dataset,
            probe,
            baseline_geometry,
            &stability,
            mirrors,
            despeckle,
            captured_flatten,
            &baseline_train,
            &baseline_holdout,
        ));
    }

    // Select only on train evidence. Holdout remains a report-only safety check.
    let best_photometric = rows
        .iter()
        .filter(|row| row.train_safe && !row.probe.name.starts_with("geometry_"))
        .max_by(|left, right| left.train.score.total_cmp(&right.train.score))
        .map(|row| row.probe.clone())
        .unwrap_or_else(|| baseline_probe.clone());
    println!(
        "train-selected photometric probe: {}",
        best_photometric.name
    );

    let mut phase_b = Vec::new();
    for radius in [0.20, 0.33, 0.45] {
        for strength in [0.35, 0.70, 1.00] {
            phase_b.push(Probe {
                name: format!(
                    "{}_plus_postnorm_flatten_r{radius:.2}_s{strength:.2}",
                    best_photometric.name
                ),
                geometry: baseline_geometry,
                post_flatten: FlattenParams {
                    enabled: true,
                    strength,
                    radius,
                },
                affine: best_photometric.affine,
                field: [None; 2],
                warp: [None; 2],
            });
        }
    }
    if best_photometric.name != "identity" {
        phase_b.extend(geometry_probes(
            baseline_geometry,
            mirrors,
            best_photometric.affine,
            no_post_flatten,
            "combined_geometry",
        ));
    }
    let phase_b_len = phase_b.len();
    for (index, probe) in phase_b.into_iter().enumerate() {
        println!("phase B {}/{} {}", index + 1, phase_b_len, probe.name);
        rows.push(evaluate_row(
            &mut net,
            &dataset,
            probe,
            baseline_geometry,
            &stability,
            mirrors,
            despeckle,
            captured_flatten,
            &baseline_train,
            &baseline_holdout,
        ));
    }

    // Keep the first local-field experiment monocular and causally isolated. A
    // train-selected shared/per-eye affine would otherwise change the opposite eye
    // while a probe name claimed to test only one local field.
    let phase_c = spatial_field_probes(baseline_geometry, no_post_flatten);
    let phase_c_len = phase_c.len();
    let mut phase_c_rows = Vec::new();
    for (index, probe) in phase_c.into_iter().enumerate() {
        println!("phase C {}/{} {}", index + 1, phase_c_len, probe.name);
        phase_c_rows.push(evaluate_row(
            &mut net,
            &dataset,
            probe,
            baseline_geometry,
            &stability,
            mirrors,
            despeckle,
            captured_flatten,
            &baseline_train,
            &baseline_holdout,
        ));
    }
    let combinations = field_combo_probes(&phase_c_rows, baseline_geometry, no_post_flatten);
    rows.extend(phase_c_rows);
    let combinations_len = combinations.len();
    for (index, probe) in combinations.into_iter().enumerate() {
        println!(
            "phase C combo {}/{} {}",
            index + 1,
            combinations_len,
            probe.name
        );
        rows.push(evaluate_row(
            &mut net,
            &dataset,
            probe,
            baseline_geometry,
            &stability,
            mirrors,
            despeckle,
            captured_flatten,
            &baseline_train,
            &baseline_holdout,
        ));
    }

    // Coordinate interventions are evaluated as their own causal family.  They use
    // identity research photometry so a successful warp cannot be mistaken for a
    // gain/contrast effect.  Landmarks never parameterize these fixed-coordinate
    // probes; commanded labels and the same EyeNet safety metrics remain the judge.
    let phase_d = coordinate_warp_probes(baseline_geometry, identity_affine, no_post_flatten);
    let phase_d_len = phase_d.len();
    for (index, probe) in phase_d.into_iter().enumerate() {
        println!("phase D {}/{} {}", index + 1, phase_d_len, probe.name);
        rows.push(evaluate_row(
            &mut net,
            &dataset,
            probe,
            baseline_geometry,
            &stability,
            mirrors,
            despeckle,
            captured_flatten,
            &baseline_train,
            &baseline_holdout,
        ));
    }

    write_results(
        &args,
        &identity,
        true,
        "COMPLETED_EXPLORATION_TRAIN_RANKING",
        "normal exploration completed; candidates were ranked on train evidence and holdout remained report-only",
        &baseline_probe,
        &baseline_train,
        &baseline_holdout,
        &rows,
    )?;
    let mut ranked: Vec<_> = rows.iter().collect();
    ranked.sort_by(|left, right| right.train.score.total_cmp(&left.train.score));
    println!("top train-selected probes (holdout was not used for ranking):");
    for row in ranked.into_iter().take(8) {
        println!(
            "  {:<46} train {:.4} holdout {:.4} safe {}/{} gaze {:.3}/{:.3} squeeze {:.3}/{:.3}",
            row.probe.name,
            row.train.score,
            row.holdout.score,
            row.train_safe,
            row.holdout_safe,
            row.holdout.gaze_retention[0],
            row.holdout.gaze_retention[1],
            row.holdout.gaze_squeeze_fp[0],
            row.holdout.gaze_squeeze_fp[1],
        );
    }
    println!("results={}", args.out.join("results.csv").display());
    Ok(())
}

fn evaluate(
    net: &mut EyeNet,
    dataset: &GeometryDataset,
    probe: &Probe,
    stability_baseline: [MlGeometry; 2],
    stability: &StabilityReport,
    mirrors: [bool; 2],
    despeckle: DespeckleParams,
    captured_flatten: FlattenParams,
) -> (GeometryMetrics, GeometryMetrics) {
    research_evaluate_photometric(
        net,
        dataset,
        stability_baseline,
        stability,
        probe.geometry,
        mirrors,
        despeckle,
        captured_flatten,
        probe.post_flatten,
        probe.affine,
        probe.field,
        probe.warp,
    )
}

fn evaluate_row(
    net: &mut EyeNet,
    dataset: &GeometryDataset,
    probe: Probe,
    stability_baseline: [MlGeometry; 2],
    stability: &StabilityReport,
    mirrors: [bool; 2],
    despeckle: DespeckleParams,
    captured_flatten: FlattenParams,
    baseline_train: &GeometryMetrics,
    baseline_holdout: &GeometryMetrics,
) -> ResultRow {
    let (train, holdout) = evaluate(
        net,
        dataset,
        &probe,
        stability_baseline,
        stability,
        mirrors,
        despeckle,
        captured_flatten,
    );
    let train_safe = research_candidate_admissible(&train, baseline_train)
        && research_always_open_guards(&train, baseline_train);
    let holdout_safe = research_candidate_admissible(&holdout, baseline_holdout)
        && research_always_open_guards(&holdout, baseline_holdout);
    ResultRow {
        probe,
        train,
        holdout,
        train_safe,
        holdout_safe,
    }
}

fn research_always_open_guards(candidate: &GeometryMetrics, baseline: &GeometryMetrics) -> bool {
    candidate.saturation_rate <= baseline.saturation_rate + 0.10
        && (0..2).all(|eye| {
            let span = (baseline.open_ref[eye] - baseline.closed_ref[eye]).max(0.001);
            candidate.closed_ref[eye] <= baseline.closed_ref[eye] + 0.05 * span
                && candidate.slow_close_std[eye] >= baseline.slow_close_std[eye] * 0.60
                && candidate.monotonicity[eye] + 0.05 >= baseline.monotonicity[eye]
        })
}

fn photometric_probes(geometry: [MlGeometry; 2], base_post_flatten: FlattenParams) -> Vec<Probe> {
    let mut probes = vec![Probe {
        name: "identity".into(),
        geometry,
        post_flatten: base_post_flatten,
        affine: [[1.0, 0.0]; 2],
        field: [None; 2],
        warp: [None; 2],
    }];
    for gain in [0.85, 1.15] {
        probes.push(affine_probe(
            format!("postnorm_shared_gain_{gain:.2}"),
            geometry,
            [[gain, 0.0]; 2],
            base_post_flatten,
        ));
    }
    for bias in [-15.0, -8.0, 8.0, 15.0] {
        probes.push(affine_probe(
            format!("postnorm_shared_bias_{bias:+.0}"),
            geometry,
            [[1.0, bias]; 2],
            base_post_flatten,
        ));
    }
    for (gain, bias) in [(0.85, -15.0), (0.85, 15.0), (1.15, -15.0), (1.15, 15.0)] {
        probes.push(affine_probe(
            format!("postnorm_shared_gain_{gain:.2}_bias_{bias:+.0}"),
            geometry,
            [[gain, bias]; 2],
            base_post_flatten,
        ));
    }
    for (eye, eye_name) in [(0usize, "l"), (1usize, "r")] {
        for gain in [0.85, 1.0, 1.15] {
            for bias in [-15.0, -8.0, 0.0, 8.0, 15.0] {
                if gain == 1.0 && bias == 0.0 {
                    continue;
                }
                let mut affine = [[1.0, 0.0]; 2];
                affine[eye] = [gain, bias];
                probes.push(affine_probe(
                    format!("postnorm_eye_{eye_name}_gain_{gain:.2}_bias_{bias:+.0}"),
                    geometry,
                    affine,
                    base_post_flatten,
                ));
            }
        }
    }
    probes
}

fn spatial_field_probes(geometry: [MlGeometry; 2], base_post_flatten: FlattenParams) -> Vec<Probe> {
    let mut probes = Vec::new();
    for (eye, eye_name) in [(0usize, "l"), (1usize, "r")] {
        for (axis, values) in [
            ("horizontal_tilt", vec![-0.12, -0.06, 0.06, 0.12]),
            ("vertical_tilt", vec![-0.12, -0.06, 0.06, 0.12]),
            ("horizontal_curve", vec![-0.08, 0.08]),
            ("vertical_curve", vec![-0.08, 0.08]),
        ] {
            for value in values {
                let mut candidate = SpatialGainField::default();
                match axis {
                    "horizontal_tilt" => candidate.horizontal = value,
                    "vertical_tilt" => candidate.vertical = value,
                    "horizontal_curve" => candidate.horizontal_curve = value,
                    _ => candidate.vertical_curve = value,
                }
                let mut field = [None; 2];
                field[eye] = Some(candidate);
                probes.push(Probe {
                    name: format!("postnorm_field_eye_{eye_name}_{axis}_{value:+.2}"),
                    geometry,
                    post_flatten: base_post_flatten,
                    affine: [[1.0, 0.0]; 2],
                    field,
                    warp: [None; 2],
                });
            }
        }
    }
    probes
}

fn field_combo_probes(
    rows: &[ResultRow],
    geometry: [MlGeometry; 2],
    post_flatten: FlattenParams,
) -> Vec<Probe> {
    let mut combinations = Vec::new();
    for (eye, eye_name) in [(0usize, "l"), (1usize, "r")] {
        let other_eye = 1 - eye;
        let mut tilts = rows
            .iter()
            .filter(|row| {
                row.train_safe
                    && row.probe.field[other_eye].is_none()
                    && row.probe.field[eye]
                        .is_some_and(|field| field.horizontal != 0.0 || field.vertical != 0.0)
            })
            .collect::<Vec<_>>();
        let mut curves = rows
            .iter()
            .filter(|row| {
                row.train_safe
                    && row.probe.field[other_eye].is_none()
                    && row.probe.field[eye].is_some_and(|field| {
                        field.horizontal_curve != 0.0 || field.vertical_curve != 0.0
                    })
            })
            .collect::<Vec<_>>();
        tilts.sort_by(|left, right| right.train.score.total_cmp(&left.train.score));
        curves.sort_by(|left, right| right.train.score.total_cmp(&left.train.score));
        for tilt in tilts.into_iter().take(2) {
            for curve in curves.iter().take(2) {
                let tilt_field = tilt.probe.field[eye].unwrap();
                let curve_field = curve.probe.field[eye].unwrap();
                let mut field = [None; 2];
                field[eye] = Some(SpatialGainField {
                    horizontal: tilt_field.horizontal,
                    vertical: tilt_field.vertical,
                    horizontal_curve: curve_field.horizontal_curve,
                    vertical_curve: curve_field.vertical_curve,
                });
                combinations.push(Probe {
                    name: format!(
                        "postnorm_field_combo_eye_{eye_name}_tilt_{}_curve_{}",
                        tilt.probe.name, curve.probe.name
                    ),
                    geometry,
                    post_flatten,
                    affine: [[1.0, 0.0]; 2],
                    field,
                    warp: [None; 2],
                });
            }
        }
    }
    combinations
}

fn affine_probe(
    name: String,
    geometry: [MlGeometry; 2],
    affine: [[f32; 2]; 2],
    post_flatten: FlattenParams,
) -> Probe {
    Probe {
        name,
        geometry,
        post_flatten,
        affine,
        field: [None; 2],
        warp: [None; 2],
    }
}

fn geometry_probes(
    baseline: [MlGeometry; 2],
    mirrors: [bool; 2],
    affine: [[f32; 2]; 2],
    post_flatten: FlattenParams,
    prefix: &str,
) -> Vec<Probe> {
    let mut probes = Vec::new();
    for (name, horizontal, vertical, scale_x, scale_y, rotation) in [
        ("horizontal_negative", -0.04, 0.0, 0.0, 0.0, 0.0),
        ("horizontal_positive", 0.04, 0.0, 0.0, 0.0, 0.0),
        ("vertical_up", 0.0, -0.04, 0.0, 0.0, 0.0),
        ("vertical_down", 0.0, 0.04, 0.0, 0.0, 0.0),
        ("scale_x_less", 0.0, 0.0, -0.10, 0.0, 0.0),
        ("scale_x_more", 0.0, 0.0, 0.10, 0.0, 0.0),
        ("scale_y_less", 0.0, 0.0, 0.0, -0.10, 0.0),
        ("scale_y_more", 0.0, 0.0, 0.0, 0.10, 0.0),
        ("rotation_less", 0.0, 0.0, 0.0, 0.0, -6.0),
        ("rotation_more", 0.0, 0.0, 0.0, 0.0, 6.0),
    ] {
        for (target_name, targets) in [
            ("shared", [true, true]),
            ("eye_l", [true, false]),
            ("eye_r", [false, true]),
        ] {
            let mut geometry = baseline;
            for eye in 0..2 {
                if targets[eye] {
                    apply_geometry_delta(
                        &mut geometry[eye],
                        if mirrors[eye] {
                            -horizontal
                        } else {
                            horizontal
                        },
                        vertical,
                        scale_x,
                        scale_y,
                        rotation,
                    );
                }
            }
            // A crop-window translation that would cross the recorded frame edge is
            // not silently converted into a crop/scale intervention. Omit that probe
            // unless every targeted eye can realize it while preserving window size.
            if targets
                .iter()
                .enumerate()
                .any(|(eye, target)| *target && geometry[eye] == baseline[eye])
            {
                continue;
            }
            probes.push(Probe {
                name: format!("{prefix}_{target_name}_{name}"),
                geometry,
                post_flatten,
                affine,
                field: [None; 2],
                warp: [None; 2],
            });
        }
    }
    probes
}

/// Fixed-coordinate, landmark-independent probes for the nonlinear spatial class.
/// Each magnitude is evaluated independently so the report exposes dose response;
/// there is deliberately no train-selected combination search at this stage.
fn coordinate_warp_probes(
    geometry: [MlGeometry; 2],
    affine: [[f32; 2]; 2],
    post_flatten: FlattenParams,
) -> Vec<Probe> {
    let mut probes = Vec::new();
    for (axis, values) in [
        ("vertical_bow", [-0.08, -0.04, 0.04, 0.08]),
        ("radial_k1", [-0.08, -0.04, 0.04, 0.08]),
    ] {
        for value in values {
            let candidate = match axis {
                "vertical_bow" => ResearchCoordinateWarp {
                    vertical_bow: value,
                    radial_k1: 0.0,
                },
                _ => ResearchCoordinateWarp {
                    vertical_bow: 0.0,
                    radial_k1: value,
                },
            };
            for (target_name, targets) in [
                ("shared", [true, true]),
                ("eye_l", [true, false]),
                ("eye_r", [false, true]),
            ] {
                let warp = std::array::from_fn(|eye| targets[eye].then_some(candidate));
                probes.push(Probe {
                    name: format!("postnorm_warp_{target_name}_{axis}_{value:+.2}"),
                    geometry,
                    post_flatten,
                    affine,
                    field: [None; 2],
                    warp,
                });
            }
        }
    }
    probes
}

fn apply_geometry_delta(
    geometry: &mut MlGeometry,
    horizontal: f32,
    vertical: f32,
    scale_x: f32,
    scale_y: f32,
    rotation: f32,
) {
    translate_crop_axis(
        &mut geometry.crop_left,
        &mut geometry.crop_right,
        horizontal,
    );
    translate_crop_axis(&mut geometry.crop_top, &mut geometry.crop_bottom, vertical);
    geometry.scale_x = (geometry.scale_x + scale_x).clamp(0.50, 2.0);
    geometry.scale_y = (geometry.scale_y + scale_y).clamp(0.50, 2.0);
    let rotation_sign = if geometry.rotate_deg < 0.0 { -1.0 } else { 1.0 };
    geometry.rotate_deg += rotation_sign * rotation;
}

fn translate_crop_axis(leading: &mut f32, trailing: &mut f32, requested: f32) {
    let original_leading = if leading.is_finite() {
        leading.clamp(0.0, 0.8)
    } else {
        0.0
    };
    let original_trailing = if trailing.is_finite() {
        trailing.clamp(0.0, 0.8)
    } else {
        0.0
    };
    let delta = if requested.is_finite() {
        requested.clamp(-original_leading, original_trailing)
    } else {
        0.0
    };
    *leading = original_leading + delta;
    *trailing = original_trailing - delta;
}

fn write_results(
    args: &Args,
    identity: &ArtifactIdentity,
    selection_allowed: bool,
    selection_status: &str,
    selection_reason: &str,
    baseline: &Probe,
    baseline_train: &GeometryMetrics,
    baseline_holdout: &GeometryMetrics,
    rows: &[ResultRow],
) -> Result<(), String> {
    let results_path = args.out.join("results.csv");
    let mut file =
        File::create(&results_path).map_err(|error| format!("create results.csv: {error}"))?;
    writeln!(file, "{RESULTS_HEADER}").map_err(|error| error.to_string())?;
    let baseline_train_safe = selection_allowed
        && research_candidate_admissible(baseline_train, baseline_train)
        && research_always_open_guards(baseline_train, baseline_train);
    let baseline_holdout_safe = selection_allowed
        && research_candidate_admissible(baseline_holdout, baseline_holdout)
        && research_always_open_guards(baseline_holdout, baseline_holdout);
    write_row(
        &mut file,
        baseline,
        baseline_train,
        baseline_holdout,
        selection_allowed,
        baseline_train_safe,
        baseline_holdout_safe,
    )?;
    for row in rows {
        write_row(
            &mut file,
            &row.probe,
            &row.train,
            &row.holdout,
            selection_allowed,
            selection_allowed && row.train_safe,
            selection_allowed && row.holdout_safe,
        )?;
    }
    file.flush()
        .map_err(|error| format!("flush results.csv: {error}"))?;
    drop(file);
    let results_sha256 = hash_file(&results_path)?;
    std::fs::write(
        args.out.join("manifest.txt"),
        format!(
            "research_only=true\nreplay_contract={}\ncrate_version={}\ncontent_hashes_authoritative=true\nrecording={}\nrecording_evidence_contract=samples_csv_then_ordered_source_index_left_right_frame_bytes_v1\nrecording_evidence_sha256={}\nmetadata_sha256={}\nsamples_sha256={}\nmodel={}\nmodel_sha256={}\nmodel_bytes={}\nexecutable_sha256={}\nreplay_parity_status={}\nparity_selection_allowed={}\nselection_allowed={}\nselection_status={}\nselection_reason={}\nresults_sha256={}\nprobes={}\nmode={}\nholdout={}\nintervention_seam=after_captured_adaptive_brightness_before_geometry\ncaptured_affine_replayed=true\nstateful_brightness_recomputed=false\nprobe_parameters_are_post_normalization=true\ngain_field_coordinates=recorded_baseline_crop_extended_over_full_frame\ncoordinate_warp_landmark_driven=false\ncoordinate_warp_full_frame_fixed_coordinates=true\ncoordinate_warp_vertical_bow_bound=0.12\ncoordinate_warp_radial_k1_bound=0.10\nproduction_config_changed=false\n",
            REPLAY_CONTRACT,
            env!("CARGO_PKG_VERSION"),
            args.recording.display(),
            identity.recording_evidence_sha256,
            identity.metadata_sha256,
            identity.samples_sha256,
            args.model.display(),
            identity.model_sha256,
            identity.model_bytes,
            identity.executable_sha256,
            identity.parity_status,
            identity.parity_selection_allowed,
            selection_allowed,
            manifest_value(selection_status),
            manifest_value(selection_reason),
            results_sha256,
            rows.len(),
            if args.confirmation { "invalidated_confirmation_request" } else if args.phase0_only { "phase0" } else { "exploration" },
            "report_only",
        ),
    )
    .map_err(|error| format!("write manifest: {error}"))?;
    Ok(())
}

const RESULTS_HEADER: &str = concat!(
    "name,selection_allowed,train_score,holdout_score,train_safe,holdout_safe,",
    "train_evidence_valid,holdout_evidence_valid,train_finite_rate,holdout_finite_rate,",
    "train_presence_rate,holdout_presence_rate,train_image_std,holdout_image_std,",
    "train_motion_energy,holdout_motion_energy,train_saturation_rate,holdout_saturation_rate,",
    "train_separation_l,train_separation_r,holdout_separation_l,holdout_separation_r,",
    "train_monotonicity_l,train_monotonicity_r,holdout_monotonicity_l,holdout_monotonicity_r,",
    "train_gaze_retention_l,train_gaze_retention_r,holdout_gaze_retention_l,holdout_gaze_retention_r,",
    "train_gaze_squeeze_fp_l,train_gaze_squeeze_fp_r,holdout_gaze_squeeze_fp_l,holdout_gaze_squeeze_fp_r,",
    "train_gaze_asymmetry,holdout_gaze_asymmetry,",
    "train_open_ref_l,train_open_ref_r,holdout_open_ref_l,holdout_open_ref_r,",
    "train_closed_ref_l,train_closed_ref_r,holdout_closed_ref_l,holdout_closed_ref_r,",
    "train_slow_close_std_l,train_slow_close_std_r,holdout_slow_close_std_l,holdout_slow_close_std_r,",
    "postnorm_flatten_enabled,postnorm_flatten_radius,postnorm_flatten_strength,",
    "postnorm_left_gain,postnorm_left_bias,postnorm_right_gain,postnorm_right_bias,",
    "left_crop_left,left_crop_right,left_crop_top,left_crop_bottom,left_scale_x,left_scale_y,left_rotation,left_mirror_h,",
    "right_crop_left,right_crop_right,right_crop_top,right_crop_bottom,right_scale_x,right_scale_y,right_rotation,right_mirror_h,",
    "postnorm_field_l_horizontal,postnorm_field_l_vertical,postnorm_field_l_horizontal_curve,postnorm_field_l_vertical_curve,",
    "postnorm_field_r_horizontal,postnorm_field_r_vertical,postnorm_field_r_horizontal_curve,postnorm_field_r_vertical_curve,",
    "postnorm_warp_l_vertical_bow,postnorm_warp_l_radial_k1,postnorm_warp_r_vertical_bow,postnorm_warp_r_radial_k1"
);

fn write_row<W: Write>(
    file: &mut W,
    probe: &Probe,
    train: &GeometryMetrics,
    holdout: &GeometryMetrics,
    selection_allowed: bool,
    train_safe: bool,
    holdout_safe: bool,
) -> Result<(), String> {
    // A rejected/diagnostic run can never expose a row as safe, even if a caller
    // passes the pre-gate metric result through unchanged.
    let train_safe = selection_allowed && train_safe;
    let holdout_safe = selection_allowed && holdout_safe;
    let l = probe.geometry[0];
    let r = probe.geometry[1];
    let fields = [
        probe.field[0].unwrap_or_default(),
        probe.field[1].unwrap_or_default(),
    ];
    let warps = [
        probe.warp[0].unwrap_or_default(),
        probe.warp[1].unwrap_or_default(),
    ];
    let mirror_name = |value: Option<bool>| match value {
        Some(true) => "true",
        Some(false) => "false",
        None => "inherit",
    };
    let columns = vec![
        probe.name.clone(),
        selection_allowed.to_string(),
        format!("{:.9}", train.score),
        format!("{:.9}", holdout.score),
        train_safe.to_string(),
        holdout_safe.to_string(),
        train.evidence_valid.to_string(),
        holdout.evidence_valid.to_string(),
        format!("{:.9}", train.finite_rate),
        format!("{:.9}", holdout.finite_rate),
        format!("{:.9}", train.presence_rate),
        format!("{:.9}", holdout.presence_rate),
        format!("{:.9}", train.image_std),
        format!("{:.9}", holdout.image_std),
        format!("{:.9}", train.motion_energy),
        format!("{:.9}", holdout.motion_energy),
        format!("{:.9}", train.saturation_rate),
        format!("{:.9}", holdout.saturation_rate),
        format!("{:.9}", train.separation[0]),
        format!("{:.9}", train.separation[1]),
        format!("{:.9}", holdout.separation[0]),
        format!("{:.9}", holdout.separation[1]),
        format!("{:.9}", train.monotonicity[0]),
        format!("{:.9}", train.monotonicity[1]),
        format!("{:.9}", holdout.monotonicity[0]),
        format!("{:.9}", holdout.monotonicity[1]),
        format!("{:.9}", train.gaze_retention[0]),
        format!("{:.9}", train.gaze_retention[1]),
        format!("{:.9}", holdout.gaze_retention[0]),
        format!("{:.9}", holdout.gaze_retention[1]),
        format!("{:.9}", train.gaze_squeeze_fp[0]),
        format!("{:.9}", train.gaze_squeeze_fp[1]),
        format!("{:.9}", holdout.gaze_squeeze_fp[0]),
        format!("{:.9}", holdout.gaze_squeeze_fp[1]),
        format!("{:.9}", train.gaze_asymmetry),
        format!("{:.9}", holdout.gaze_asymmetry),
        format!("{:.9}", train.open_ref[0]),
        format!("{:.9}", train.open_ref[1]),
        format!("{:.9}", holdout.open_ref[0]),
        format!("{:.9}", holdout.open_ref[1]),
        format!("{:.9}", train.closed_ref[0]),
        format!("{:.9}", train.closed_ref[1]),
        format!("{:.9}", holdout.closed_ref[0]),
        format!("{:.9}", holdout.closed_ref[1]),
        format!("{:.9}", train.slow_close_std[0]),
        format!("{:.9}", train.slow_close_std[1]),
        format!("{:.9}", holdout.slow_close_std[0]),
        format!("{:.9}", holdout.slow_close_std[1]),
        probe.post_flatten.enabled.to_string(),
        format!("{:.9}", probe.post_flatten.radius),
        format!("{:.9}", probe.post_flatten.strength),
        format!("{:.9}", probe.affine[0][0]),
        format!("{:.9}", probe.affine[0][1]),
        format!("{:.9}", probe.affine[1][0]),
        format!("{:.9}", probe.affine[1][1]),
        format!("{:.9}", l.crop_left),
        format!("{:.9}", l.crop_right),
        format!("{:.9}", l.crop_top),
        format!("{:.9}", l.crop_bottom),
        format!("{:.9}", l.scale_x),
        format!("{:.9}", l.scale_y),
        format!("{:.9}", l.rotate_deg),
        mirror_name(l.mirror_h).to_owned(),
        format!("{:.9}", r.crop_left),
        format!("{:.9}", r.crop_right),
        format!("{:.9}", r.crop_top),
        format!("{:.9}", r.crop_bottom),
        format!("{:.9}", r.scale_x),
        format!("{:.9}", r.scale_y),
        format!("{:.9}", r.rotate_deg),
        mirror_name(r.mirror_h).to_owned(),
        format!("{:.9}", fields[0].horizontal),
        format!("{:.9}", fields[0].vertical),
        format!("{:.9}", fields[0].horizontal_curve),
        format!("{:.9}", fields[0].vertical_curve),
        format!("{:.9}", fields[1].horizontal),
        format!("{:.9}", fields[1].vertical),
        format!("{:.9}", fields[1].horizontal_curve),
        format!("{:.9}", fields[1].vertical_curve),
        format!("{:.9}", warps[0].vertical_bow),
        format!("{:.9}", warps[0].radial_k1),
        format!("{:.9}", warps[1].vertical_bow),
        format!("{:.9}", warps[1].radial_k1),
    ];
    writeln!(file, "{}", columns.join(",")).map_err(|error| error.to_string())
}

fn manifest_value(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '\r' | '\n' => ' ',
            other => other,
        })
        .collect()
}

fn hash_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("read hash input {}: {error}", path.display()))?;
    Ok(hash_bytes(&bytes))
}

fn hash_current_executable() -> Result<String, String> {
    let path = std::env::current_exe()
        .map_err(|error| format!("resolve current replay executable: {error}"))?;
    hash_file(&path).map_err(|error| format!("hash replay executable: {error}"))
}

/// Hash the exact ordered evidence consumed by this replay. `samples.csv` binds
/// labels and relative paths; the explicit ordinal/source-index/side records below
/// bind each referenced image byte stream without relying on filesystem metadata.
fn hash_recording_evidence(
    root: &Path,
    frames: &[xr5_recording::RecordingFrame],
) -> Result<String, String> {
    let samples = std::fs::read(root.join("samples.csv"))
        .map_err(|error| format!("read recording evidence samples.csv: {error}"))?;
    let mut hasher = Sha256::new();
    hash_evidence_part(&mut hasher, b"samples.csv", &samples);
    for (ordinal, frame) in frames.iter().enumerate() {
        hasher.update((ordinal as u64).to_le_bytes());
        hasher.update((frame.source_index as u64).to_le_bytes());
        for (side, relative) in [
            (b"left".as_slice(), &frame.left_file),
            (b"right".as_slice(), &frame.right_file),
        ] {
            let path = root.join(relative);
            let bytes = std::fs::read(&path)
                .map_err(|error| format!("read recording evidence {}: {error}", path.display()))?;
            hash_evidence_part(&mut hasher, side, &bytes);
        }
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}

fn hash_evidence_part(hasher: &mut Sha256, tag: &[u8], bytes: &[u8]) {
    hasher.update((tag.len() as u64).to_le_bytes());
    hasher.update(tag);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn replay_parity(metadata: &xr5_recording::RecordingMetadata, model_bytes: &[u8]) -> ReplayParity {
    let supplied_crc32 = format!("{:08x}", crc32_ieee(model_bytes));
    let capture_crc32 = metadata
        .get("eyelid_model_crc32")
        .map(|value| value.trim().to_ascii_lowercase());
    let capture_bytes = metadata
        .get("eyelid_model_bytes")
        .and_then(|value| value.trim().parse::<u64>().ok());
    let model_match = capture_crc32.as_deref() == Some(supplied_crc32.as_str())
        && capture_bytes == Some(model_bytes.len() as u64);
    let protocol_match = metadata.get("capture_protocol") == Some("safe_geometry_fit_v1")
        && metadata.get("frame_stage") == Some("after_eye_mapping_before_ml_geometry");
    let (status, detail) = if capture_crc32.is_none() || capture_bytes.is_none() {
        (
            "UNKNOWN",
            "capture lacks the load-time EyeNet fingerprint required for replay selection"
                .to_owned(),
        )
    } else if !model_match {
        (
            "MISMATCH",
            format!(
                "capture EyeNet fingerprint does not match supplied model (supplied {supplied_crc32}/{} bytes)",
                model_bytes.len()
            ),
        )
    } else if !protocol_match {
        (
            "PROTOCOL_MISMATCH",
            "capture protocol/frame stage is not the Safe Geometry pre-geometry seam".to_owned(),
        )
    } else {
        (
            "MATCH",
            "capture-time EyeNet fingerprint and Safe Geometry frame-stage contract match"
                .to_owned(),
        )
    };
    ReplayParity {
        status,
        parity_selection_allowed: status == "MATCH",
        detail,
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

fn parse_args() -> Result<Args, String> {
    let mut recording = None;
    let mut model = None;
    let mut out = None;
    let mut phase0_only = false;
    let mut confirmation = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--phase0-only" {
            phase0_only = true;
            continue;
        }
        if argument == "--confirmation" {
            confirmation = true;
            continue;
        }
        let destination = match argument.to_str() {
            Some("--recording") => &mut recording,
            Some("--model") => &mut model,
            Some("--out") => &mut out,
            Some("--help" | "-h") => {
                println!(
                    "xr5-photometric-replay --recording <extracted ZIP directory> --model <EyeNet params> --out <new directory>"
                );
                std::process::exit(0);
            }
            _ => {
                return Err(format!(
                    "unknown argument {}",
                    PathBuf::from(argument).display()
                ))
            }
        };
        *destination = Some(PathBuf::from(args.next().ok_or("option requires a path")?));
    }
    let result = Args {
        recording: recording.ok_or("missing --recording")?,
        model: model.ok_or("missing --model")?,
        out: out.ok_or("missing --out")?,
        phase0_only,
        confirmation,
    };
    if result.phase0_only && result.confirmation {
        return Err("--phase0-only and --confirmation are mutually exclusive".into());
    }
    if result.out.exists() {
        return Err(format!(
            "output already exists; use a new directory: {}",
            result.out.display()
        ));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn sha256_matches_standard_vector() {
        assert_eq!(
            hash_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn results_header_and_row_have_matching_audit_columns() {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let probe = Probe {
            name: "audit".into(),
            geometry,
            post_flatten: FlattenParams::default(),
            affine: [[1.0, 0.0]; 2],
            field: [None; 2],
            warp: [None; 2],
        };
        let train = GeometryMetrics {
            evidence_valid: true,
            open_ref: [0.8, 0.9],
            ..GeometryMetrics::default()
        };
        let holdout = GeometryMetrics {
            evidence_valid: true,
            open_ref: [0.7, 0.6],
            ..GeometryMetrics::default()
        };
        let mut row = Vec::new();
        // Even deliberately passing pre-gate `true` flags cannot make a rejected
        // artifact advertise either split as safe.
        write_row(&mut row, &probe, &train, &holdout, false, true, true).unwrap();
        let row = String::from_utf8(row).unwrap();
        assert_eq!(
            RESULTS_HEADER.split(',').count(),
            row.trim_end().split(',').count()
        );
        for required in [
            "selection_allowed",
            "train_evidence_valid",
            "holdout_motion_energy",
            "train_gaze_squeeze_fp_l",
            "holdout_open_ref_r",
            "holdout_slow_close_std_r",
        ] {
            assert!(RESULTS_HEADER.split(',').any(|column| column == required));
        }
        let values = row.split(',').collect::<Vec<_>>();
        assert_eq!(values[1], "false");
        assert_eq!(values[4], "false");
        assert_eq!(values[5], "false");
    }

    #[test]
    fn rejected_artifact_separates_parity_from_overall_selection() {
        let unique = format!(
            "sranibro-replay-rejected-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let out = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&out).unwrap();
        let args = Args {
            recording: PathBuf::from("human-readable-recording-path"),
            model: PathBuf::from("human-readable-model-path"),
            out: out.clone(),
            phase0_only: false,
            confirmation: false,
        };
        let identity = ArtifactIdentity {
            recording_evidence_sha256: "11".repeat(32),
            metadata_sha256: "22".repeat(32),
            samples_sha256: "33".repeat(32),
            model_sha256: "44".repeat(32),
            model_bytes: 42,
            executable_sha256: "55".repeat(32),
            parity_status: "MATCH",
            parity_selection_allowed: true,
        };
        let probe = Probe {
            name: "identity".into(),
            geometry: sranibro_rs::config::default_ml_geometry("pimax_xr5"),
            post_flatten: FlattenParams::default(),
            affine: [[1.0, 0.0]; 2],
            field: [None; 2],
            warp: [None; 2],
        };
        let metrics = GeometryMetrics {
            evidence_valid: true,
            finite_rate: 1.0,
            ..GeometryMetrics::default()
        };
        write_results(
            &args,
            &identity,
            false,
            "REJECTED_TEST",
            "first line\nsecond line",
            &probe,
            &metrics,
            &metrics,
            &[],
        )
        .unwrap();
        let manifest = std::fs::read_to_string(out.join("manifest.txt")).unwrap();
        assert!(manifest.contains("parity_selection_allowed=true\n"));
        assert!(manifest.contains("selection_allowed=false\n"));
        assert!(manifest.contains("selection_status=REJECTED_TEST\n"));
        assert!(manifest.contains("selection_reason=first line second line\n"));
        assert!(!manifest.contains("selection=train_only"));
        let results = std::fs::read_to_string(out.join("results.csv")).unwrap();
        let values = results
            .lines()
            .nth(1)
            .unwrap()
            .split(',')
            .collect::<Vec<_>>();
        assert_eq!(values[1], "false");
        assert_eq!(values[4], "false");
        assert_eq!(values[5], "false");
        std::fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn recording_evidence_hash_binds_source_index_and_frame_bytes() {
        let unique = format!(
            "sranibro-replay-hash-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(root.join("frames")).unwrap();
        std::fs::write(root.join("samples.csv"), b"index,left,right\n7,l,r\n").unwrap();
        std::fs::write(root.join("frames/l.png"), b"left-bytes").unwrap();
        std::fs::write(root.join("frames/r.png"), b"right-bytes").unwrap();
        let mut frames = vec![xr5_recording::RecordingFrame {
            source_index: 7,
            holdout: false,
            left_file: PathBuf::from("frames/l.png"),
            right_file: PathBuf::from("frames/r.png"),
            fields: BTreeMap::new(),
        }];
        let initial = hash_recording_evidence(&root, &frames).unwrap();
        frames[0].source_index = 8;
        assert_ne!(initial, hash_recording_evidence(&root, &frames).unwrap());
        frames[0].source_index = 7;
        std::fs::write(root.join("frames/r.png"), b"changed").unwrap();
        assert_ne!(initial, hash_recording_evidence(&root, &frames).unwrap());
        std::fs::remove_dir_all(root).unwrap();
    }

    fn assert_field_eq(left: Option<SpatialGainField>, right: Option<SpatialGainField>) {
        match (left, right) {
            (None, None) => {}
            (Some(left), Some(right)) => {
                assert_eq!(left.horizontal, right.horizontal);
                assert_eq!(left.vertical, right.vertical);
                assert_eq!(left.horizontal_curve, right.horizontal_curve);
                assert_eq!(left.vertical_curve, right.vertical_curve);
            }
            _ => panic!("field presence differs"),
        }
    }

    fn find_probe<'a>(probes: &'a [Probe], name: &str) -> &'a Probe {
        probes
            .iter()
            .find(|probe| probe.name == name)
            .unwrap_or_else(|| panic!("missing probe {name}"))
    }

    #[test]
    fn always_open_guards_reject_closed_lift_and_lost_slow_response() {
        let baseline = GeometryMetrics {
            open_ref: [0.8; 2],
            closed_ref: [0.2; 2],
            slow_close_std: [0.15; 2],
            monotonicity: [0.8; 2],
            saturation_rate: 0.05,
            ..GeometryMetrics::default()
        };
        let mut candidate = baseline.clone();
        assert!(research_always_open_guards(&candidate, &baseline));
        candidate.closed_ref[1] = 0.231;
        assert!(!research_always_open_guards(&candidate, &baseline));
        candidate = baseline.clone();
        candidate.slow_close_std[1] = 0.08;
        assert!(!research_always_open_guards(&candidate, &baseline));
    }

    #[test]
    fn affine_probe_biases_use_pixel_units() {
        let probes = photometric_probes(
            sranibro_rs::config::default_ml_geometry("pimax_xr5"),
            FlattenParams::default(),
        );
        assert!(probes.iter().any(|probe| probe.affine[1][1] == -15.0));
        assert!(probes.iter().any(|probe| probe.affine[1][1] == 15.0));
        assert!(probes
            .iter()
            .all(|probe| probe.affine[0][1].abs() >= 1.0 || probe.affine[0][1] == 0.0));
    }

    #[test]
    fn probe_set_has_mirrored_left_and_right_counterparts() {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let flatten = FlattenParams::default();
        let affine_probes = photometric_probes(geometry, flatten);
        let mut affine_pairs = 0;
        for left in affine_probes
            .iter()
            .filter(|probe| probe.name.starts_with("postnorm_eye_l_"))
        {
            let right_name = left.name.replacen("postnorm_eye_l_", "postnorm_eye_r_", 1);
            let right = find_probe(&affine_probes, &right_name);
            assert_eq!(left.affine[0], right.affine[1]);
            assert_eq!(left.affine[1], right.affine[0]);
            affine_pairs += 1;
        }
        assert_eq!(affine_pairs, 14);

        let field_probes = spatial_field_probes(geometry, flatten);
        let mut field_pairs = 0;
        for left in field_probes
            .iter()
            .filter(|probe| probe.name.starts_with("postnorm_field_eye_l_"))
        {
            let right_name =
                left.name
                    .replacen("postnorm_field_eye_l_", "postnorm_field_eye_r_", 1);
            let right = find_probe(&field_probes, &right_name);
            assert_field_eq(left.field[0], right.field[1]);
            assert_field_eq(left.field[1], right.field[0]);
            field_pairs += 1;
        }
        assert_eq!(field_pairs, 12);

        let xr5_baseline = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let geometry_probes = geometry_probes(
            xr5_baseline,
            [false, true],
            [[1.0, 0.0]; 2],
            flatten,
            "symmetry",
        );
        let mut geometry_pairs = 0;
        for left in geometry_probes
            .iter()
            .filter(|probe| probe.name.starts_with("symmetry_eye_l_"))
        {
            let right_name = left.name.replacen("symmetry_eye_l_", "symmetry_eye_r_", 1);
            let right = find_probe(&geometry_probes, &right_name);
            let left_changed = left.geometry[0];
            let right_changed = right.geometry[1];
            assert_eq!(left.geometry[1], xr5_baseline[1]);
            assert_eq!(right.geometry[0], xr5_baseline[0]);
            assert_eq!(left_changed.crop_left, right_changed.crop_right);
            assert_eq!(left_changed.crop_right, right_changed.crop_left);
            assert_eq!(left_changed.crop_top, right_changed.crop_top);
            assert_eq!(left_changed.crop_bottom, right_changed.crop_bottom);
            assert_eq!(left_changed.scale_x, right_changed.scale_x);
            assert_eq!(left_changed.scale_y, right_changed.scale_y);
            assert_eq!(left_changed.rotate_deg, -right_changed.rotate_deg);
            geometry_pairs += 1;
        }
        assert_eq!(geometry_pairs, 9);

        let shared_shift = find_probe(&geometry_probes, "symmetry_shared_horizontal_positive");
        for eye in 0..2 {
            let before_width = 1.0 - xr5_baseline[eye].crop_left - xr5_baseline[eye].crop_right;
            let after_width =
                1.0 - shared_shift.geometry[eye].crop_left - shared_shift.geometry[eye].crop_right;
            assert!((before_width - after_width).abs() < f32::EPSILON);
        }
        assert!(geometry_probes
            .iter()
            .all(|probe| probe.name != "symmetry_shared_horizontal_negative"));
    }

    #[test]
    fn coordinate_warp_probes_are_bounded_symmetric_and_monocularly_isolated() {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let probes = coordinate_warp_probes(geometry, [[1.0, 0.0]; 2], FlattenParams::default());
        assert_eq!(probes.len(), 24);

        let mut monocular_pairs = 0;
        for left in probes
            .iter()
            .filter(|probe| probe.name.starts_with("postnorm_warp_eye_l_"))
        {
            let right_name = left
                .name
                .replacen("postnorm_warp_eye_l_", "postnorm_warp_eye_r_", 1);
            let right = find_probe(&probes, &right_name);
            let left_warp = left.warp[0].expect("left probe must modify left only");
            let right_warp = right.warp[1].expect("right probe must modify right only");
            assert!(left.warp[1].is_none());
            assert!(right.warp[0].is_none());
            assert_eq!(left_warp.vertical_bow, right_warp.vertical_bow);
            assert_eq!(left_warp.radial_k1, right_warp.radial_k1);
            assert!(left_warp.vertical_bow.abs() <= ResearchCoordinateWarp::MAX_VERTICAL_BOW);
            assert!(left_warp.radial_k1.abs() <= ResearchCoordinateWarp::MAX_RADIAL_K1);
            monocular_pairs += 1;
        }
        assert_eq!(monocular_pairs, 8);

        for shared in probes
            .iter()
            .filter(|probe| probe.name.starts_with("postnorm_warp_shared_"))
        {
            let left = shared.warp[0].expect("shared probe must modify left");
            let right = shared.warp[1].expect("shared probe must modify right");
            assert_eq!(left.vertical_bow, right.vertical_bow);
            assert_eq!(left.radial_k1, right.radial_k1);
        }
    }

    #[test]
    fn monocular_probes_do_not_modify_the_other_eye() {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let flatten = FlattenParams::default();
        for probe in photometric_probes(geometry, flatten) {
            if probe.name.starts_with("postnorm_eye_l_") {
                assert_ne!(probe.affine[0], [1.0, 0.0]);
                assert_eq!(probe.affine[1], [1.0, 0.0]);
            } else if probe.name.starts_with("postnorm_eye_r_") {
                assert_eq!(probe.affine[0], [1.0, 0.0]);
                assert_ne!(probe.affine[1], [1.0, 0.0]);
            }
        }
        for probe in spatial_field_probes(geometry, flatten) {
            if probe.name.starts_with("postnorm_field_eye_l_") {
                assert!(probe.field[0].is_some());
                assert!(probe.field[1].is_none());
            } else if probe.name.starts_with("postnorm_field_eye_r_") {
                assert!(probe.field[0].is_none());
                assert!(probe.field[1].is_some());
            }
        }
        for probe in geometry_probes(
            geometry,
            [false, true],
            [[1.0, 0.0]; 2],
            flatten,
            "isolation",
        ) {
            if probe.name.starts_with("isolation_eye_l_") {
                assert_ne!(probe.geometry[0], geometry[0]);
                assert_eq!(probe.geometry[1], geometry[1]);
            } else if probe.name.starts_with("isolation_eye_r_") {
                assert_eq!(probe.geometry[0], geometry[0]);
                assert_ne!(probe.geometry[1], geometry[1]);
            }
        }
    }

    #[test]
    fn field_combos_never_mix_eyes() {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        let flatten = FlattenParams::default();
        let rows = spatial_field_probes(geometry, flatten)
            .into_iter()
            .enumerate()
            .map(|(index, probe)| ResultRow {
                probe,
                train: GeometryMetrics {
                    score: index as f32,
                    ..GeometryMetrics::default()
                },
                holdout: GeometryMetrics::default(),
                train_safe: true,
                holdout_safe: true,
            })
            .collect::<Vec<_>>();
        let combos = field_combo_probes(&rows, geometry, FlattenParams::default());
        assert_eq!(combos.len(), 8);
        for combo in combos {
            let populated = combo
                .field
                .iter()
                .enumerate()
                .filter_map(|(eye, field)| field.map(|_| eye))
                .collect::<Vec<_>>();
            assert_eq!(populated.len(), 1);
            let (eye_name, other_name) = if populated[0] == 0 {
                ("l", "r")
            } else {
                ("r", "l")
            };
            assert!(combo.name.contains(&format!("combo_eye_{eye_name}_")));
            assert!(combo
                .name
                .contains(&format!("tilt_postnorm_field_eye_{eye_name}_")));
            assert!(combo
                .name
                .contains(&format!("curve_postnorm_field_eye_{eye_name}_")));
            assert!(!combo
                .name
                .contains(&format!("postnorm_field_eye_{other_name}_")));
        }
    }

    #[test]
    fn selection_requires_capture_model_and_frame_stage_parity() {
        let bytes = b"123456789";
        let mut fields = BTreeMap::new();
        fields.insert("eyelid_model_crc32".into(), "cbf43926".into());
        fields.insert("eyelid_model_bytes".into(), "9".into());
        fields.insert("capture_protocol".into(), "safe_geometry_fit_v1".into());
        fields.insert(
            "frame_stage".into(),
            "after_eye_mapping_before_ml_geometry".into(),
        );
        let metadata = xr5_recording::RecordingMetadata {
            schema_version: 2,
            raw: String::new(),
            fields,
        };
        assert!(replay_parity(&metadata, bytes).parity_selection_allowed);
        assert!(!replay_parity(&metadata, b"different").parity_selection_allowed);

        let mut wrong_stage = metadata.clone();
        wrong_stage
            .fields
            .insert("frame_stage".into(), "post_geometry".into());
        assert!(!replay_parity(&wrong_stage, bytes).parity_selection_allowed);
    }
}
