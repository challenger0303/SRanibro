//! Research-only, non-mutating audit of XR5 image evidence and EyeNet residuals.
//!
//! Raw-image landmark values are deliberately QA-only: the real-image detector is not
//! calibrated and is excluded from every association, regression, and selection.  The
//! only enabled explanatory variables are candidate-independent raw photometric
//! measurements. The supplied EyeNet model follows the production preprocessing and
//! inference seam, but legacy recordings cannot prove capture-time model-hash parity.

#[path = "xr5_recording.rs"]
mod xr5_recording;

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs::File;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use sranibro_rs::core::types::FlattenParams;
use sranibro_rs::geometry_calib::{GazeTarget, GeometryDataset, GeometrySample, SampleFamily};
use sranibro_rs::geometry_fitrun::{research_replay_detailed, ResearchObservation};
use sranibro_rs::geometry_landmarks::{detect_eye_landmarks, EyeSide, GrayFrame, LandmarkDecision};
use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

const AUDIT_SCHEMA: &str = "xr5-landmark-residual-audit-v1";
const EXPECTED_RECORDED_PHASES: usize = 56;
// The 2.20 s RelaxedOpen phase leaves enough post-reaction evidence for eleven
// stable 20 Hz rows while still tolerating the static end guard.
const MIN_COMMANDED_STABLE_ROWS_PER_PHASE: usize = 11;
const BASE_PHOTO_NAMES: [&str; 9] = [
    "p10",
    "median",
    "p90",
    "mad",
    "horizontal_gradient",
    "vertical_gradient",
    "vertical_curvature",
    "saturation_fraction",
    "glint_fraction",
];
const PHOTO_COUNT: usize = 36;
const RIDGE_LAMBDA: f64 = 1.0;
const MIN_RIDGE_BLOCKS: usize = PHOTO_COUNT + 5;
const MIN_DISCOVERY_SESSIONS_FOR_RIDGE: usize = 3;
const MIN_ASSOCIATION_SELECTION_BLOCKS: usize = 10;
const MIN_BLOCK_FRAMES: usize = 10;
const MIN_BLOCK_SPAN_S: f32 = 0.90;
// A nominal one-second block must represent a continuous observation, not two dense
// bursts separated by UI/worker starvation. This still leaves >3x headroom over the
// real 64 ms capture cadence exercised by the reachability regression test.
const MAX_BLOCK_GAP_S: f32 = 0.20;

/// v2 adds no-sampling instruction phases and a wider SteamVR presentation. v3 adds
/// unrecorded action rehearsals. All retain the same 56 labelled recording phases,
/// target repetitions, and sample quotas consumed by this audit.
fn residual_protocol_supported(protocol: Option<&str>) -> bool {
    matches!(
        protocol,
        Some(
            "xr5_landmark_residual_audit_v1"
                | "xr5_landmark_residual_audit_v2"
                | "xr5_landmark_residual_audit_v3"
        )
    )
}

#[derive(Debug)]
struct Args {
    recording: PathBuf,
    discoveries: Vec<PathBuf>,
    confirmations: Vec<PathBuf>,
    model: PathBuf,
    out: PathBuf,
    emit_overlays: bool,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct TrainAnchor {
    open: f32,
    closed: f32,
    span: f32,
    neutral_squeeze: f32,
    valid: bool,
}

impl Default for TrainAnchor {
    fn default() -> Self {
        Self {
            open: 0.0,
            closed: 0.0,
            span: 0.0,
            neutral_squeeze: 0.0,
            valid: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct CaptureTimingQuality {
    frame_generation_nonzero_coverage: [f32; 2],
    frame_generation_strict_increase_rate: [Option<f32>; 2],
    native_timestamp_coverage: f32,
    native_timestamp_nondecreasing_rate: Option<f32>,
    native_timestamp_strict_increase_rate: Option<f32>,
    interpretation: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelParity {
    status: String,
    capture_crc32: Option<String>,
    capture_bytes: Option<u64>,
    supplied_crc32: String,
    supplied_bytes: u64,
    capture_fingerprint_match: Option<bool>,
    replay_interpretation: String,
}

#[derive(Clone, Debug, Serialize)]
struct LandmarkQc {
    status: String,
    selection_scope: String,
    calibrated: bool,
    pupil_detection_neutral_gaze: [f32; 2],
    closed_like_false_pupil: [f32; 2],
    closed_like_frames: [usize; 2],
    temporal_jump_rate: [f32; 2],
    stereo_implausible_rate: f32,
    diagnostic_pupil_frames: [usize; 2],
    accepted_pupil_frames: [usize; 2],
    reasons: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct SessionSummary {
    session: usize,
    role: String,
    recording: String,
    schema_version: u32,
    unit_id: Option<String>,
    capture_protocol: Option<String>,
    capture_evidence_complete: bool,
    capture_timing_quality: CaptureTimingQuality,
    model_parity: ModelParity,
    frames: usize,
    train_frames: usize,
    holdout_frames: usize,
    full_population_replay_train_score: Option<f32>,
    full_population_replay_holdout_score: Option<f32>,
    full_population_replay_evidence_valid: [bool; 2],
    safe_fit_protocol_compatible: bool,
    production_stratified_fit_score_comparable: bool,
    replay_metric_scope: String,
    anchors: [TrainAnchor; 2],
    landmark_qc: LandmarkQc,
    direction_mode: String,
    absolute_direction_supported: bool,
    limitations: Vec<String>,
}

#[derive(Clone, Debug)]
struct FrameRow {
    session: usize,
    role: String,
    schema: u32,
    sample_index: usize,
    source_index: usize,
    split: &'static str,
    kind: String,
    family: SampleFamily,
    phase_index: usize,
    phase_time_s: f32,
    frame_generation: [u64; 2],
    native_timestamp_us: Option<u64>,
    eye: usize,
    stable: bool,
    commanded_target: Option<String>,
    direction_source: String,
    direction_bin: String,
    presence: f32,
    open: f32,
    squeeze: f32,
    normalized_open: Option<f32>,
    gaze_error: Option<f32>,
    slow_error: Option<f32>,
    closed_lift: Option<f32>,
    squeeze_error: Option<f32>,
    native_gaze_yaw: Option<f32>,
    native_gaze_pitch: Option<f32>,
    native_pupil_x: Option<f32>,
    native_pupil_y: Option<f32>,
    pupil_status: String,
    pupil_reason: String,
    pupil_quality: f32,
    pupil_x: Option<f32>,
    pupil_y: Option<f32>,
    pupil_major: Option<f32>,
    pupil_minor: Option<f32>,
    pupil_angle: Option<f32>,
    pupil_contrast: Option<f32>,
    pupil_boundary: Option<f32>,
    pupil_consensus: Option<usize>,
    lid_status: String,
    lid_reason: String,
    lid_quality: f32,
    lid_aperture: Option<f32>,
    lid_path_strength: Option<f32>,
    occlusion_proxy: f32,
    temporal_jump_px: Option<f32>,
    stereo_implausible: Option<bool>,
    photo: [f32; PHOTO_COUNT],
    fixed_ir_flare_proxy: f32,
}

#[derive(Clone, Debug, Serialize)]
struct PhaseQualityRow {
    session: usize,
    role: String,
    split: String,
    kind: String,
    phase_index: usize,
    eye: String,
    frames: usize,
    stable_frames: usize,
    pupil_diagnostic_rate: f32,
    pupil_accepted_rate: f32,
    lid_diagnostic_rate: f32,
    median_pupil_quality: Option<f32>,
    median_lid_quality: Option<f32>,
    median_photo_vertical_gradient: Option<f32>,
    abstentions: String,
    interpretation: String,
}

#[derive(Clone, Debug)]
struct BlockRow {
    session: usize,
    role: String,
    split: String,
    phase_index: usize,
    second_block: usize,
    eye: String,
    direction_source: String,
    direction_bin: String,
    frames: usize,
    span_s: f32,
    eligible_one_second_support: bool,
    normalized_open_median: Option<f32>,
    gaze_error_median: Option<f32>,
    slow_error_median: Option<f32>,
    squeeze_error_median: Option<f32>,
    photo_medians: [Option<f32>; PHOTO_COUNT],
}

#[derive(Clone, Debug, Serialize)]
struct DirectionRow {
    session: usize,
    role: String,
    split: String,
    eye: String,
    source: String,
    bin: String,
    frames: usize,
    normalized_open_median: Option<f32>,
    normalized_open_p10: Option<f32>,
    gaze_error_median: Option<f32>,
    squeeze_p90: Option<f32>,
    status: String,
}

#[derive(Clone, Debug, Serialize)]
struct AssociationRow {
    session: usize,
    role: String,
    split: String,
    eye: String,
    outcome: String,
    feature: String,
    n: usize,
    spearman_rho: Option<f32>,
    eligible_for_selection: bool,
    selected_on_discovery_train: bool,
    interpretation: String,
}

#[derive(Clone, Debug, Serialize)]
struct ModelRow {
    session: usize,
    role: String,
    split: String,
    eye: String,
    model: String,
    n: usize,
    mae: Option<f32>,
    r2: Option<f32>,
    frozen_from_discovery_train: bool,
    eligible_for_production: bool,
    status: String,
}

#[derive(Clone, Debug, Serialize)]
struct DiscoveryPoolSummary {
    minimum_independent_sessions: usize,
    contributing_session_ids_per_eye: [Vec<usize>; 2],
    one_second_blocks_per_eye: [usize; 2],
    fitted_per_eye: [bool; 2],
    session_wise_feature_stability_evaluated: bool,
    eligible_for_hypothesis_freeze: bool,
    excluded_sessions: Vec<DiscoveryPoolExclusion>,
    policy: String,
}

#[derive(Clone, Debug, Serialize)]
struct DiscoveryPoolExclusion {
    session: usize,
    reason: String,
}

struct SessionAnalysis {
    root: PathBuf,
    evidence_sha256: String,
    loaded: xr5_recording::LoadedRecording,
    summary: SessionSummary,
    rows: Vec<FrameRow>,
}

#[derive(Clone, Serialize)]
struct RidgeModel {
    means: Vec<f64>,
    scales: Vec<f64>,
    y_mean: f64,
    beta: Vec<f64>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("xr5 landmark residual audit failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let model_bytes = std::fs::read(&args.model)
        .map_err(|error| format!("read model {}: {error}", args.model.display()))?;
    let mut inputs = vec![("discovery".to_owned(), args.recording.clone())];
    inputs.extend(
        args.discoveries
            .iter()
            .cloned()
            .map(|path| ("discovery".to_owned(), path)),
    );
    inputs.extend(
        args.confirmations
            .iter()
            .cloned()
            .map(|path| ("confirmation".to_owned(), path)),
    );
    let mut sessions = Vec::new();
    for (session, (role, root)) in inputs.into_iter().enumerate() {
        println!("audit session {session} ({role}): {}", root.display());
        sessions.push(analyse_session(session, role, root, &model_bytes)?);
    }

    let mut associations = associations(&sessions);
    mark_selected_association(&mut associations);
    let (ridge, model_rows, discovery_pool) = model_comparison(&sessions);
    let phase_rows = phase_quality(&sessions);
    let block_rows = block_features(&sessions);
    let direction_rows = direction_bins(&sessions);

    std::fs::create_dir(&args.out)
        .map_err(|error| format!("create output {}: {error}", args.out.display()))?;
    write_frame_features(&args.out, &sessions)?;
    write_phase_quality(&args.out, &phase_rows)?;
    write_block_features(&args.out, &block_rows)?;
    write_direction_bins(&args.out, &direction_rows)?;
    write_associations(&args.out, &associations)?;
    write_models(&args.out, &model_rows)?;
    if args.emit_overlays {
        write_overlays(&args.out, &sessions)?;
    }
    write_summary(
        &args,
        &sessions,
        &associations,
        &model_rows,
        &ridge,
        &discovery_pool,
    )?;
    write_report(
        &args.out,
        &sessions,
        &associations,
        &model_rows,
        &discovery_pool,
    )?;
    write_manifest(&args, &sessions, &model_bytes)?;
    println!("audit results={}", args.out.display());
    Ok(())
}

fn analyse_session(
    session: usize,
    role: String,
    root: PathBuf,
    model_bytes: &[u8],
) -> Result<SessionAnalysis, String> {
    let loaded = xr5_recording::load_recording(&root)?;
    if loaded.frames.len() != loaded.dataset.samples.len() {
        return Err("recording frame metadata and decoded samples differ in length".into());
    }
    let evidence_sha256 = hash_recording_evidence(&root, &loaded.frames)?;
    let capture_evidence_complete =
        capture_evidence_metadata_complete(&loaded.metadata, &loaded.dataset);
    let map = tvm_params::parse_map_bytes(model_bytes)
        .map_err(|error| format!("parse EyeNet model: {error}"))?;
    let mut net = EyeNet::new(map).map_err(|error| format!("EyeNet model invalid: {error}"))?;
    let replay = research_replay_detailed(
        &mut net,
        &loaded.dataset,
        loaded.baseline,
        loaded.baseline,
        loaded.mirrors,
        loaded.despeckle,
        loaded.flatten,
        FlattenParams::default(),
        [[1.0, 0.0]; 2],
        [None; 2],
        [None; 2],
    );
    let anchors = train_anchors(&replay.observations);
    let observation_by_sample = replay
        .observations
        .iter()
        .map(|observation| (observation.sample_index, *observation))
        .collect::<BTreeMap<_, _>>();
    let mut rows = Vec::with_capacity(loaded.dataset.samples.len() * 2);
    let mut previous = BTreeMap::<(usize, usize), [f32; 2]>::new();
    for (sample_index, sample) in loaded.dataset.samples.iter().enumerate() {
        let observation = observation_by_sample
            .get(&sample_index)
            .ok_or_else(|| format!("production replay omitted sample {sample_index}"))?;
        let reports = [
            detect_eye_landmarks(
                GrayFrame {
                    pixels: &sample.left,
                    width: sample.left_size.0 as usize,
                    height: sample.left_size.1 as usize,
                },
                EyeSide::Left,
            ),
            detect_eye_landmarks(
                GrayFrame {
                    pixels: &sample.right,
                    width: sample.right_size.0 as usize,
                    height: sample.right_size.1 as usize,
                },
                EyeSide::Right,
            ),
        ];
        let centers = reports
            .each_ref()
            .map(|report| report.pupil.value().map(|pupil| pupil.center_px));
        let stereo_implausible = centers[0].zip(centers[1]).map(|(left, right)| {
            (left[0] + right[0] - 198.0).abs() > 30.0 || (left[1] - right[1]).abs() > 25.0
        });
        for eye in 0..2 {
            let report = &reports[eye];
            let pupil = report.pupil.value();
            let lid = report.lids.value();
            let temporal_jump_px = pupil.map(|pupil| {
                previous
                    .insert((sample.phase_index, eye), pupil.center_px)
                    .map(|prior| {
                        (prior[0] - pupil.center_px[0]).hypot(prior[1] - pupil.center_px[1])
                    })
                    .unwrap_or(0.0)
            });
            let normalized_open = anchors[eye]
                .valid
                .then_some((observation.open[eye] - anchors[eye].closed) / anchors[eye].span);
            let family = sample.kind.family();
            let gaze_error = (family == SampleFamily::GazeSweep)
                .then_some(normalized_open.map(|value| 1.0 - value))
                .flatten();
            let slow_error = (family == SampleFamily::SlowClose)
                .then_some(
                    normalized_open
                        .zip(sample.expected_open)
                        .map(|(value, expected)| value - expected),
                )
                .flatten();
            let closed_like = family == SampleFamily::Closed
                || (family == SampleFamily::SlowClose
                    && sample.expected_open.is_some_and(|value| value <= 0.10));
            let closed_lift = closed_like.then_some(normalized_open).flatten();
            let squeeze_error = anchors[eye]
                .valid
                .then_some(observation.squeeze[eye] - anchors[eye].neutral_squeeze);
            let photo = &report.photometric;
            let mut photo_values = [0.0; PHOTO_COUNT];
            photo_values[..9].copy_from_slice(&[
                photo.p10,
                photo.median,
                photo.p90,
                photo.mad,
                photo.horizontal_gradient,
                photo.vertical_gradient,
                photo.vertical_curvature,
                photo.saturation_fraction,
                photo.glint_fraction,
            ]);
            photo_values[9..18].copy_from_slice(&photo.local_median_3x3);
            photo_values[18..27].copy_from_slice(&photo.local_p90_3x3);
            photo_values[27..36].copy_from_slice(&photo.local_saturation_3x3);
            rows.push(FrameRow {
                session,
                role: role.clone(),
                schema: loaded.metadata.schema_version,
                sample_index,
                source_index: loaded.frames[sample_index].source_index,
                split: if sample.kind.is_holdout() {
                    "holdout"
                } else {
                    "train"
                },
                kind: sample.kind.as_str().to_owned(),
                family,
                phase_index: sample.phase_index,
                phase_time_s: sample.phase_time_s,
                frame_generation: sample.frame_generation,
                native_timestamp_us: sample.native_timestamp_us,
                eye,
                stable: observation.stable,
                commanded_target: sample
                    .commanded_target
                    .map(GazeTarget::as_str)
                    .map(str::to_owned),
                direction_source: "unassigned".into(),
                direction_bin: "unassigned".into(),
                presence: observation.presence,
                open: observation.open[eye],
                squeeze: observation.squeeze[eye],
                normalized_open,
                gaze_error,
                slow_error,
                closed_lift,
                squeeze_error,
                native_gaze_yaw: observation.native_gaze_deg[eye].map(|gaze| gaze[0]),
                native_gaze_pitch: observation.native_gaze_deg[eye].map(|gaze| gaze[1]),
                native_pupil_x: sample.native_pupil_pos[eye].map(|pupil| pupil[0]),
                native_pupil_y: sample.native_pupil_pos[eye].map(|pupil| pupil[1]),
                pupil_status: decision_status(&report.pupil).into(),
                pupil_reason: decision_reason(&report.pupil),
                pupil_quality: report.pupil.quality(),
                pupil_x: pupil.map(|value| value.center_px[0]),
                pupil_y: pupil.map(|value| value.center_px[1]),
                pupil_major: pupil.map(|value| value.radii_px[0]),
                pupil_minor: pupil.map(|value| value.radii_px[1]),
                pupil_angle: pupil.map(|value| value.angle_deg),
                pupil_contrast: pupil.map(|value| value.contrast),
                pupil_boundary: pupil.map(|value| value.boundary_coverage),
                pupil_consensus: pupil.map(|value| value.threshold_consensus),
                lid_status: decision_status(&report.lids).into(),
                lid_reason: decision_reason(&report.lids),
                lid_quality: report.lids.quality(),
                lid_aperture: lid.map(|value| value.median_aperture_px),
                lid_path_strength: lid.map(|value| value.path_strength),
                occlusion_proxy: report.occlusion_fraction,
                temporal_jump_px,
                stereo_implausible,
                photo: photo_values,
                fixed_ir_flare_proxy: photo.fixed_ir_flare_fraction,
            });
        }
    }
    // All mode-selection QC is frozen from discovery-train evidence. Holdout rows remain
    // available in the CSV/report, but can never rescue or tune a failed detector.
    let qc = landmark_qc(&rows, "train");
    let (direction_mode, absolute_direction_supported, mut limitations) = assign_direction_bins(
        &mut rows,
        loaded.metadata.schema_version,
        loaded.metadata.get("capture_protocol"),
        capture_evidence_complete,
        &qc,
    );
    limitations.push(
        "Pupil and lid detections are UNCALIBRATED_QA_ONLY and excluded from ranking and regression."
            .into(),
    );
    limitations.push(
        "pupil_minus_global is deliberately excluded because it depends on an uncalibrated pupil lock."
            .into(),
    );
    limitations.push(
        "fixed_ir_flare_fraction is exported descriptively but excluded from the compact first-pass model."
            .into(),
    );
    if loaded.metadata.schema_version == 1 {
        limitations.push(
            "Schema v1 has no native gaze or commanded target; it cannot support absolute direction claims."
                .into(),
        );
    }
    if loaded.metadata.schema_version >= 3 && !capture_evidence_complete {
        limitations.push(
            "Capture quota metadata or observed per-phase counts are incomplete; commanded-target mode and discovery pooling are disabled."
                .into(),
        );
    }
    let model_parity = model_parity(&loaded.metadata, model_bytes);
    match model_parity.status.as_str() {
        "MATCH" => limitations.push(
            "Capture CRC32/byte-count fingerprint matches the supplied EyeNet model; CRC32 is an integrity fingerprint, not a cryptographic identity proof."
                .into(),
        ),
        "MISMATCH" => limitations.push(
            "Supplied EyeNet model fingerprint does not match capture metadata; replay outputs are non-parity diagnostics."
                .into(),
        ),
        "UNKNOWN" => limitations.push(
            "Capture-time EyeNet CRC32/byte-count metadata is absent or incomplete; replay outputs are non-parity diagnostics."
                .into(),
        ),
        _ => limitations.push(
            "Capture-time EyeNet fingerprint metadata is malformed; replay outputs are non-parity diagnostics."
                .into(),
        ),
    }
    limitations.push(
        "native_timestamp_us is latest reported native evidence timing, not camera-exposure synchronization ground truth."
            .into(),
    );
    limitations.push(
        "Direction/QC selection uses train evidence only; holdout target-phase presence is checked solely as v3 protocol-integrity evidence and never tunes a bin or threshold."
            .into(),
    );
    let capture_protocol = loaded.metadata.get("capture_protocol");
    let safe_fit_protocol_compatible = capture_protocol == Some("safe_geometry_fit_v1")
        || (capture_protocol.is_none() && loaded.metadata.schema_version < 3);
    let replay_metric_scope = if safe_fit_protocol_compatible {
        "FULL_POPULATION_REPLAY_NOT_PRODUCTION_STRATIFIED_FIT_SCORE"
    } else {
        "FULL_RECORDING_METRIC_NOT_SAFE_FIT_COMPARABLE"
    };
    let summary = SessionSummary {
        session,
        role: role.clone(),
        recording: root.display().to_string(),
        schema_version: loaded.metadata.schema_version,
        unit_id: loaded.metadata.get("unit_id").map(str::to_owned),
        capture_protocol: loaded.metadata.get("capture_protocol").map(str::to_owned),
        capture_evidence_complete,
        capture_timing_quality: capture_timing_quality(&loaded.dataset.samples),
        model_parity,
        frames: loaded.dataset.samples.len(),
        train_frames: loaded.dataset.train_len(),
        holdout_frames: loaded.dataset.holdout_len(),
        full_population_replay_train_score: replay
            .train
            .evidence_valid
            .then_some(replay.train.score),
        full_population_replay_holdout_score: replay
            .holdout
            .evidence_valid
            .then_some(replay.holdout.score),
        full_population_replay_evidence_valid: [
            replay.train.evidence_valid,
            replay.holdout.evidence_valid,
        ],
        safe_fit_protocol_compatible,
        production_stratified_fit_score_comparable: false,
        replay_metric_scope: replay_metric_scope.into(),
        anchors,
        landmark_qc: qc,
        direction_mode,
        absolute_direction_supported,
        limitations,
    };
    Ok(SessionAnalysis {
        root,
        evidence_sha256,
        loaded,
        summary,
        rows,
    })
}

fn train_anchors(observations: &[ResearchObservation]) -> [TrainAnchor; 2] {
    std::array::from_fn(|eye| {
        let mut open = observations
            .iter()
            .filter(|observation| {
                !observation.kind.is_holdout()
                    && observation.stable
                    && observation.kind.family() == SampleFamily::Neutral
            })
            .map(|observation| observation.open[eye])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if open.len() < 5 {
            open = observations
                .iter()
                .filter(|observation| {
                    !observation.kind.is_holdout()
                        && observation.kind.family() == SampleFamily::SlowClose
                        && observation.expected_open.is_some_and(|value| value >= 0.90)
                })
                .map(|observation| observation.open[eye])
                .filter(|value| value.is_finite())
                .collect();
        }
        let mut closed = observations
            .iter()
            .filter(|observation| {
                !observation.kind.is_holdout()
                    && observation.stable
                    && observation.kind.family() == SampleFamily::Closed
            })
            .map(|observation| observation.open[eye])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        if closed.len() < 5 {
            closed = observations
                .iter()
                .filter(|observation| {
                    !observation.kind.is_holdout()
                        && observation.kind.family() == SampleFamily::SlowClose
                        && observation.expected_open.is_some_and(|value| value <= 0.10)
                })
                .map(|observation| observation.open[eye])
                .filter(|value| value.is_finite())
                .collect();
        }
        let neutral_squeeze = observations
            .iter()
            .filter(|observation| {
                !observation.kind.is_holdout()
                    && observation.stable
                    && observation.kind.family() == SampleFamily::Neutral
            })
            .map(|observation| observation.squeeze[eye])
            .filter(|value| value.is_finite())
            .collect::<Vec<_>>();
        let open_ref = percentile(&open, 0.50).unwrap_or(0.0);
        let closed_ref = percentile(&closed, 0.50).unwrap_or(0.0);
        let span = open_ref - closed_ref;
        TrainAnchor {
            open: open_ref,
            closed: closed_ref,
            span,
            neutral_squeeze: percentile(&neutral_squeeze, 0.50).unwrap_or(0.0),
            valid: open.len() >= 5 && closed.len() >= 5 && span.is_finite() && span > 0.001,
        }
    })
}

fn capture_timing_quality(samples: &[GeometrySample]) -> CaptureTimingQuality {
    let generation_nonzero = std::array::from_fn(|eye| {
        ratio(
            samples
                .iter()
                .filter(|sample| sample.frame_generation[eye] > 0)
                .count(),
            samples.len(),
        )
    });
    let generation_increase = std::array::from_fn(|eye| {
        let transitions = samples
            .windows(2)
            .filter(|pair| pair[0].frame_generation[eye] > 0 && pair[1].frame_generation[eye] > 0)
            .collect::<Vec<_>>();
        (!transitions.is_empty()).then(|| {
            ratio(
                transitions
                    .iter()
                    .filter(|pair| pair[1].frame_generation[eye] > pair[0].frame_generation[eye])
                    .count(),
                transitions.len(),
            )
        })
    });
    let timestamp_coverage = ratio(
        samples
            .iter()
            .filter(|sample| sample.native_timestamp_us.is_some())
            .count(),
        samples.len(),
    );
    let timestamp_transitions = samples
        .windows(2)
        .filter_map(|pair| pair[0].native_timestamp_us.zip(pair[1].native_timestamp_us))
        .collect::<Vec<_>>();
    let timestamp_nondecreasing = (!timestamp_transitions.is_empty()).then(|| {
        ratio(
            timestamp_transitions
                .iter()
                .filter(|(prior, current)| current >= prior)
                .count(),
            timestamp_transitions.len(),
        )
    });
    let timestamp_strict = (!timestamp_transitions.is_empty()).then(|| {
        ratio(
            timestamp_transitions
                .iter()
                .filter(|(prior, current)| current > prior)
                .count(),
            timestamp_transitions.len(),
        )
    });
    CaptureTimingQuality {
        frame_generation_nonzero_coverage: generation_nonzero,
        frame_generation_strict_increase_rate: generation_increase,
        native_timestamp_coverage: timestamp_coverage,
        native_timestamp_nondecreasing_rate: timestamp_nondecreasing,
        native_timestamp_strict_increase_rate: timestamp_strict,
        interpretation: "Capture-order diagnostics only; native timestamps are not camera synchronization ground truth."
            .into(),
    }
}

fn model_parity(metadata: &xr5_recording::RecordingMetadata, supplied: &[u8]) -> ModelParity {
    let capture_crc_raw = metadata.get("eyelid_model_crc32").map(str::to_owned);
    let capture_bytes_raw = metadata.get("eyelid_model_bytes");
    let capture_crc_valid = capture_crc_raw.as_deref().is_some_and(|value| {
        value.len() == 8
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    let capture_bytes = capture_bytes_raw.and_then(|value| value.parse::<u64>().ok());
    let supplied_crc32 = format!("{:08x}", crc32_ieee(supplied));
    let supplied_bytes = supplied.len() as u64;
    let (status, capture_fingerprint_match, replay_interpretation) =
        match (capture_crc_raw.as_deref(), capture_bytes_raw) {
            (None, None) => (
                "UNKNOWN",
                None,
                "NON_PARITY_DIAGNOSTIC_CAPTURE_FINGERPRINT_ABSENT",
            ),
            (Some(_), Some(_)) if !capture_crc_valid || capture_bytes.is_none() => (
                "INVALID_CAPTURE_METADATA",
                None,
                "NON_PARITY_DIAGNOSTIC_CAPTURE_FINGERPRINT_INVALID",
            ),
            (Some(capture_crc), Some(_)) => {
                let matched = capture_crc == supplied_crc32
                    && capture_bytes.is_some_and(|bytes| bytes == supplied_bytes);
                if matched {
                    (
                        "MATCH",
                        Some(true),
                        "CAPTURE_FINGERPRINT_MATCH_PRODUCTION_INFERENCE_SEAM",
                    )
                } else {
                    (
                        "MISMATCH",
                        Some(false),
                        "NON_PARITY_DIAGNOSTIC_CAPTURE_FINGERPRINT_MISMATCH",
                    )
                }
            }
            _ => (
                "UNKNOWN",
                None,
                "NON_PARITY_DIAGNOSTIC_CAPTURE_FINGERPRINT_INCOMPLETE",
            ),
        };
    ModelParity {
        status: status.into(),
        capture_crc32: capture_crc_raw,
        capture_bytes,
        supplied_crc32,
        supplied_bytes,
        capture_fingerprint_match,
        replay_interpretation: replay_interpretation.into(),
    }
}

fn crc32_ieee(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = 0u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn decision_status<T>(decision: &LandmarkDecision<T>) -> &'static str {
    match decision {
        LandmarkDecision::Accepted { .. } => "accepted",
        LandmarkDecision::DiagnosticOnly { .. } => "diagnostic_only",
        LandmarkDecision::Abstained { .. } => "abstained",
    }
}

fn decision_reason<T>(decision: &LandmarkDecision<T>) -> String {
    match decision {
        LandmarkDecision::Accepted { .. } => String::new(),
        LandmarkDecision::DiagnosticOnly { reason, .. }
        | LandmarkDecision::Abstained { reason, .. } => format!("{reason:?}"),
    }
}

fn landmark_qc(rows: &[FrameRow], selection_split: &str) -> LandmarkQc {
    let mut eligible = [0usize; 2];
    let mut detected = [0usize; 2];
    let mut closed = [0usize; 2];
    let mut closed_detected = [0usize; 2];
    let mut jumps = [0usize; 2];
    let mut jump_denominator = [0usize; 2];
    let mut diagnostic = [0usize; 2];
    let mut accepted = [0usize; 2];
    let mut stereo_seen = BTreeSet::new();
    let mut stereo_bad = BTreeSet::new();
    for row in rows {
        if row.split != selection_split {
            continue;
        }
        let eye = row.eye;
        if matches!(row.family, SampleFamily::Neutral | SampleFamily::GazeSweep) && row.stable {
            eligible[eye] += 1;
            detected[eye] += usize::from(row.pupil_x.is_some());
        }
        if row.closed_lift.is_some() {
            closed[eye] += 1;
            closed_detected[eye] += usize::from(row.pupil_x.is_some());
        }
        if let Some(jump) = row.temporal_jump_px {
            jump_denominator[eye] += 1;
            jumps[eye] += usize::from(jump > 15.0);
        }
        diagnostic[eye] += usize::from(row.pupil_status == "diagnostic_only");
        accepted[eye] += usize::from(row.pupil_status == "accepted");
        if let Some(bad) = row.stereo_implausible {
            let key = (row.sample_index, row.session);
            stereo_seen.insert(key);
            if bad {
                stereo_bad.insert(key);
            }
        }
    }
    let pupil_detection_neutral_gaze =
        std::array::from_fn(|eye| ratio(detected[eye], eligible[eye]));
    let closed_like_false_pupil =
        std::array::from_fn(|eye| ratio(closed_detected[eye], closed[eye]));
    let temporal_jump_rate = std::array::from_fn(|eye| ratio(jumps[eye], jump_denominator[eye]));
    let stereo_implausible_rate = ratio(stereo_bad.len(), stereo_seen.len());
    let calibrated = accepted.iter().sum::<usize>() > 0;
    let mut reasons = Vec::new();
    if !calibrated {
        reasons.push("detector has no calibrated Accepted real-image decisions".into());
    }
    for eye in 0..2 {
        let name = if eye == 0 { "L" } else { "R" };
        if pupil_detection_neutral_gaze[eye] < 0.80 {
            reasons.push(format!(
                "{name} Neutral/Gaze pupil coverage {:.1}% < 80%",
                pupil_detection_neutral_gaze[eye] * 100.0
            ));
        }
        if closed[eye] < 20 {
            reasons.push(format!(
                "{name} has only {} closed-like QA frames; at least 20 are required",
                closed[eye]
            ));
        } else if closed_like_false_pupil[eye] > 0.05 {
            reasons.push(format!(
                "{name} closed-like false pupil {:.1}% > 5%",
                closed_like_false_pupil[eye] * 100.0
            ));
        }
        if temporal_jump_rate[eye] > 0.05 {
            reasons.push(format!(
                "{name} >15px temporal jumps {:.1}% > 5%",
                temporal_jump_rate[eye] * 100.0
            ));
        }
    }
    if stereo_implausible_rate > 0.05 {
        reasons.push(format!(
            "stereo landmark inconsistency {:.1}% > 5%",
            stereo_implausible_rate * 100.0
        ));
    }
    LandmarkQc {
        status: if calibrated && reasons.is_empty() {
            "LANDMARK_QC_GO"
        } else {
            "LANDMARK_QC_NO_GO"
        }
        .into(),
        selection_scope: format!("{selection_split}_only"),
        calibrated,
        pupil_detection_neutral_gaze,
        closed_like_false_pupil,
        closed_like_frames: closed,
        temporal_jump_rate,
        stereo_implausible_rate,
        diagnostic_pupil_frames: diagnostic,
        accepted_pupil_frames: accepted,
        reasons,
    }
}

fn capture_evidence_metadata_complete(
    metadata: &xr5_recording::RecordingMetadata,
    dataset: &GeometryDataset,
) -> bool {
    if metadata.schema_version < 3
        || !residual_protocol_supported(metadata.get("capture_protocol"))
        || metadata.get("target_repetitions_per_split") != Some("2")
        || metadata.get("capture_evidence_complete") != Some("true")
        || metadata.get("capture_missing_phase_ids") != Some("")
    {
        return false;
    }
    let Some(encoded_counts) = metadata.get("capture_phase_sample_counts") else {
        return false;
    };
    let mut observed = BTreeMap::<usize, (usize, SampleFamily)>::new();
    for sample in &dataset.samples {
        let family = sample.kind.family();
        let entry = observed.entry(sample.phase_index).or_insert((0, family));
        if entry.1 != family {
            return false;
        }
        entry.0 += 1;
    }
    if observed.len() != EXPECTED_RECORDED_PHASES {
        return false;
    }
    let mut family_counts = [0usize; 3];
    for (_, family) in observed.values() {
        match family {
            SampleFamily::Neutral | SampleFamily::GazeSweep => family_counts[0] += 1,
            SampleFamily::SlowClose => family_counts[1] += 1,
            SampleFamily::NaturalBlinks => family_counts[2] += 1,
            SampleFamily::Closed | SampleFamily::HalfOpen => return false,
        }
    }
    if family_counts != [36, 10, 10] {
        return false;
    }
    let mut declared = BTreeMap::<usize, (usize, usize)>::new();
    for item in encoded_counts.split(',').filter(|item| !item.is_empty()) {
        let Some((phase, counts)) = item.split_once(':') else {
            return false;
        };
        let Some((accepted, required)) = counts.split_once('/') else {
            return false;
        };
        let (Ok(phase), Ok(accepted), Ok(required)) = (
            phase.parse::<usize>(),
            accepted.parse::<usize>(),
            required.parse::<usize>(),
        ) else {
            return false;
        };
        let expected_required = match observed.get(&phase).map(|(_, family)| family) {
            Some(SampleFamily::Neutral | SampleFamily::GazeSweep) => 17,
            Some(SampleFamily::SlowClose) => 30,
            Some(SampleFamily::NaturalBlinks) => 16,
            Some(SampleFamily::Closed | SampleFamily::HalfOpen) | None => return false,
        };
        if required != expected_required
            || accepted < required
            || observed.get(&phase).map(|entry| entry.0) != Some(accepted)
            || declared.insert(phase, (accepted, required)).is_some()
        {
            return false;
        }
    }
    declared.len() == EXPECTED_RECORDED_PHASES
}

fn assign_direction_bins(
    rows: &mut [FrameRow],
    schema: u32,
    capture_protocol: Option<&str>,
    capture_evidence_complete: bool,
    qc: &LandmarkQc,
) -> (String, bool, Vec<String>) {
    let commanded_protocol_complete = commanded_protocol_evidence_complete(
        schema,
        capture_protocol,
        capture_evidence_complete,
        rows.iter().map(|row| {
            (
                row.eye,
                row.stable,
                row.split,
                row.phase_index,
                row.family,
                row.commanded_target.as_deref(),
            )
        }),
    );
    if commanded_protocol_complete {
        for row in rows {
            row.direction_source = "commanded_target".into();
            row.direction_bin = row
                .commanded_target
                .clone()
                .unwrap_or_else(|| "unavailable".into());
        }
        return (
            "COMMANDED_TARGET_PRIMARY".into(),
            true,
            vec![
                "Commanded bins are instructed screen categories, not calibrated gaze-angle ground truth."
                    .into(),
            ],
        );
    }

    let gaze_coverage = ratio(
        rows.iter()
            .filter(|row| {
                row.split == "train"
                    && row.family == SampleFamily::GazeSweep
                    && row.stable
                    && row.native_gaze_yaw.is_some()
                    && row.native_gaze_pitch.is_some()
            })
            .count(),
        rows.iter()
            .filter(|row| {
                row.split == "train" && row.family == SampleFamily::GazeSweep && row.stable
            })
            .count(),
    );
    if schema >= 2 && gaze_coverage >= 0.60 {
        let centers: [(Option<f32>, Option<f32>); 2] = std::array::from_fn(|eye| {
            let yaw = rows
                .iter()
                .filter(|row| {
                    row.eye == eye
                        && row.split == "train"
                        && row.family == SampleFamily::Neutral
                        && row.stable
                })
                .filter_map(|row| row.native_gaze_yaw)
                .collect::<Vec<_>>();
            let pitch = rows
                .iter()
                .filter(|row| {
                    row.eye == eye
                        && row.split == "train"
                        && row.family == SampleFamily::Neutral
                        && row.stable
                })
                .filter_map(|row| row.native_gaze_pitch)
                .collect::<Vec<_>>();
            (percentile(&yaw, 0.50), percentile(&pitch, 0.50))
        });
        for row in rows {
            row.direction_source = "native_gaze_descriptive".into();
            row.direction_bin = match (
                row.native_gaze_yaw,
                row.native_gaze_pitch,
                centers[row.eye].0,
                centers[row.eye].1,
            ) {
                (Some(yaw), Some(pitch), Some(cy), Some(cp)) => {
                    native_direction_bin(yaw - cy, pitch - cp)
                }
                _ => "unavailable".into(),
            };
        }
        return (
            "NATIVE_GAZE_DESCRIPTIVE".into(),
            false,
            vec![if schema >= 3 {
                "Commanded-target mode was rejected because the exact residual-audit protocol or complete repeated nine-target train/holdout evidence was missing; native gaze bins are descriptive device-axis measurements."
                    .into()
            } else {
                "Native gaze bins are descriptive device-axis measurements; no complete commanded-target protocol was recorded."
                    .into()
            }],
        );
    }

    if schema == 1 {
        if qc.status != "LANDMARK_QC_GO" {
            for row in rows {
                row.direction_source = "relative_pupil_proxy_disabled".into();
                row.direction_bin = "unavailable_landmark_qc".into();
            }
            return (
                "RELATIVE_PUPIL_PROXY_ONLY".into(),
                false,
                vec![
                    "Relative pupil direction proxy was disabled because landmark QC is NO-GO."
                        .into(),
                    "No left/right/up/down or absolute direction claim is available.".into(),
                ],
            );
        }
        let centers: [(Option<f32>, Option<f32>); 2] = std::array::from_fn(|eye| {
            let x = rows
                .iter()
                .filter(|row| {
                    row.eye == eye
                        && row.split == "train"
                        && row.family == SampleFamily::Neutral
                        && row.stable
                })
                .filter_map(|row| row.pupil_x)
                .collect::<Vec<_>>();
            let y = rows
                .iter()
                .filter(|row| {
                    row.eye == eye
                        && row.split == "train"
                        && row.family == SampleFamily::Neutral
                        && row.stable
                })
                .filter_map(|row| row.pupil_y)
                .collect::<Vec<_>>();
            (percentile(&x, 0.50), percentile(&y, 0.50))
        });
        for row in rows {
            row.direction_source = "relative_pupil_proxy".into();
            row.direction_bin = match (row.pupil_x, row.pupil_y, centers[row.eye]) {
                (Some(x), Some(y), (Some(cx), Some(cy))) => relative_pupil_bin(x - cx, y - cy),
                _ => "unavailable".into(),
            };
        }
        return (
            "RELATIVE_PUPIL_PROXY_ONLY".into(),
            false,
            vec!["Raw sensor-axis proxy bins must not be renamed left/right/up/down.".into()],
        );
    }
    for row in rows {
        row.direction_source = "unavailable".into();
        row.direction_bin = "unavailable".into();
    }
    (
        "DIRECTION_UNAVAILABLE".into(),
        false,
        vec!["Neither commanded targets nor sufficient native gaze evidence were recorded.".into()],
    )
}

fn commanded_protocol_evidence_complete<'a>(
    schema: u32,
    capture_protocol: Option<&str>,
    capture_evidence_complete: bool,
    evidence: impl IntoIterator<Item = (usize, bool, &'a str, usize, SampleFamily, Option<&'a str>)>,
) -> bool {
    if schema < 3 || !residual_protocol_supported(capture_protocol) || !capture_evidence_complete {
        return false;
    }
    let evidence = evidence.into_iter().collect::<Vec<_>>();
    let phase_targets = evidence
        .iter()
        .filter(|(eye, _, _, _, _, _)| *eye == 0)
        .fold(
            BTreeMap::<(&str, usize), (BTreeSet<&str>, BTreeSet<u8>)>::new(),
            |mut phases, (_, _, split, phase_index, family, target)| {
                if let Some(target) = target {
                    let phase = phases.entry((*split, *phase_index)).or_default();
                    phase.0.insert(*target);
                    phase.1.insert(*family as u8);
                }
                phases
            },
        );
    if phase_targets
        .values()
        .any(|(targets, families)| targets.len() != 1 || families.len() != 1)
    {
        return false;
    }
    GazeTarget::ALL.iter().all(|target| {
        let expected_family = if *target == GazeTarget::Center {
            SampleFamily::Neutral
        } else {
            SampleFamily::GazeSweep
        };
        ["train", "holdout"].iter().all(|split| {
            let per_phase = evidence
                .iter()
                .filter(|(eye, stable, row_split, _, family, row_target)| {
                    *eye == 0
                        && *stable
                        && row_split == split
                        && *family == expected_family
                        && *row_target == Some(target.as_str())
                })
                .fold(BTreeMap::<usize, usize>::new(), |mut counts, row| {
                    *counts.entry(row.3).or_default() += 1;
                    counts
                });
            per_phase.len() == 2
                && per_phase
                    .values()
                    .all(|count| *count >= MIN_COMMANDED_STABLE_ROWS_PER_PHASE)
        })
    })
}

fn native_direction_bin(yaw: f32, pitch: f32) -> String {
    fn axis(value: f32) -> Option<i8> {
        if value <= -7.0 {
            Some(-1)
        } else if value >= 7.0 {
            Some(1)
        } else if value.abs() <= 5.0 {
            Some(0)
        } else {
            None
        }
    }
    match (axis(yaw), axis(pitch)) {
        (Some(0), Some(0)) => "center".into(),
        (Some(x), Some(y)) => format!("yaw_{}_pitch_{}", axis_name(x), axis_name(y)),
        _ => "transition".into(),
    }
}

fn axis_name(value: i8) -> &'static str {
    match value {
        -1 => "neg",
        1 => "pos",
        _ => "center",
    }
}

fn relative_pupil_bin(dx: f32, dy: f32) -> String {
    let axis = |value: f32| {
        if value <= -4.0 {
            -1
        } else if value >= 4.0 {
            1
        } else {
            0
        }
    };
    let (x, y) = (axis(dx), axis(dy));
    if x == 0 && y == 0 {
        "raw_center".into()
    } else {
        format!("raw_x_{}_y_{}", axis_name(x), axis_name(y))
    }
}

fn photo_name(index: usize) -> String {
    if index < BASE_PHOTO_NAMES.len() {
        BASE_PHOTO_NAMES[index].into()
    } else if index < 18 {
        format!("local_median_r{}_c{}", (index - 9) / 3, (index - 9) % 3)
    } else if index < 27 {
        format!("local_p90_r{}_c{}", (index - 18) / 3, (index - 18) % 3)
    } else {
        format!(
            "local_saturation_r{}_c{}",
            (index - 27) / 3,
            (index - 27) % 3
        )
    }
}

fn associations(sessions: &[SessionAnalysis]) -> Vec<AssociationRow> {
    let mut output = Vec::new();
    for session in sessions {
        for split in ["train", "holdout"] {
            for eye in 0..2 {
                for outcome in ["gaze_error", "slow_error"] {
                    let blocks = regression_blocks(session, split, eye, outcome);
                    for feature in 0..PHOTO_COUNT {
                        let x = blocks
                            .iter()
                            .map(|pair| pair.0[feature])
                            .collect::<Vec<_>>();
                        let y = blocks.iter().map(|pair| pair.1).collect::<Vec<_>>();
                        output.push(AssociationRow {
                            session: session.summary.session,
                            role: session.summary.role.clone(),
                            split: split.into(),
                            eye: eye_name(eye).into(),
                            outcome: outcome.into(),
                            feature: photo_name(feature),
                            n: blocks.len(),
                            spearman_rho: (blocks.len() >= 5)
                                .then(|| spearman(&x, &y))
                                .flatten(),
                            eligible_for_selection: blocks.len()
                                >= MIN_ASSOCIATION_SELECTION_BLOCKS
                                && session.summary.session == 0
                                && session.summary.role == "discovery"
                                && session.summary.model_parity.status == "MATCH"
                                && split == "train"
                                && outcome == "gaze_error",
                            selected_on_discovery_train: false,
                            interpretation:
                                "per-eye 1-second block association; candidate-independent, descriptive, not causal"
                                    .into(),
                        });
                    }
                }
            }
        }
    }
    output
}

fn mark_selected_association(rows: &mut [AssociationRow]) {
    for eye in ["L", "R"] {
        let selected = rows
            .iter()
            .filter(|row| {
                row.eye == eye
                    && row.eligible_for_selection
                    && row.outcome == "gaze_error"
                    && row.n >= MIN_ASSOCIATION_SELECTION_BLOCKS
            })
            .filter_map(|row| row.spearman_rho.map(|rho| (row.feature.clone(), rho.abs())))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|value| value.0);
        if let Some(selected) = selected {
            for row in rows.iter_mut().filter(|row| row.eye == eye) {
                row.selected_on_discovery_train = row.eligible_for_selection
                    && row.feature == selected
                    && row.outcome == "gaze_error";
            }
        }
    }
}

/// Collapse autocorrelated 20 Hz frames into phase-local one-second blocks before any
/// correlation or regression. Eyes are never pooled.
fn regression_blocks(
    session: &SessionAnalysis,
    split: &str,
    eye: usize,
    outcome: &str,
) -> Vec<([f32; PHOTO_COUNT], f32)> {
    regression_blocks_from_rows(&session.rows, split, eye, outcome)
}

fn regression_blocks_from_rows(
    rows: &[FrameRow],
    split: &str,
    eye: usize,
    outcome: &str,
) -> Vec<([f32; PHOTO_COUNT], f32)> {
    let mut phase_origins = BTreeMap::<usize, f32>::new();
    for row in rows {
        let selected_outcome = match outcome {
            "gaze_error" => row.gaze_error,
            _ => row.slow_error,
        };
        if row.split == split
            && row.eye == eye
            && row.stable
            && selected_outcome.is_some_and(f32::is_finite)
            && row.photo.iter().all(|value| value.is_finite())
        {
            phase_origins
                .entry(row.phase_index)
                .and_modify(|origin| *origin = origin.min(row.phase_time_s))
                .or_insert(row.phase_time_s);
        }
    }
    let mut groups = BTreeMap::<(usize, usize), Vec<&FrameRow>>::new();
    for row in rows {
        let selected_outcome = match outcome {
            "gaze_error" => row.gaze_error,
            _ => row.slow_error,
        };
        if row.split == split
            && row.eye == eye
            && row.stable
            && selected_outcome.is_some_and(f32::is_finite)
            && row.photo.iter().all(|value| value.is_finite())
        {
            let Some(origin) = phase_origins.get(&row.phase_index) else {
                continue;
            };
            groups
                .entry((
                    row.phase_index,
                    (row.phase_time_s - origin).max(0.0).floor() as usize,
                ))
                .or_default()
                .push(row);
        }
    }
    groups
        .into_values()
        .filter_map(|rows| {
            if !block_has_one_second_support(&rows) {
                return None;
            }
            let photo = std::array::from_fn(|feature| {
                percentile(
                    &rows
                        .iter()
                        .map(|row| row.photo[feature])
                        .collect::<Vec<_>>(),
                    0.50,
                )
                .unwrap_or(0.0)
            });
            let values = rows
                .iter()
                .filter_map(|row| match outcome {
                    "gaze_error" => row.gaze_error,
                    _ => row.slow_error,
                })
                .collect::<Vec<_>>();
            percentile(&values, 0.50).map(|value| (photo, value))
        })
        .collect()
}

fn discovery_pool_exclusion(session: &SessionAnalysis) -> Option<String> {
    if session.summary.role != "discovery" {
        return Some("confirmation sessions are frozen evaluation only".into());
    }
    if session.summary.schema_version < 3
        || !residual_protocol_supported(session.summary.capture_protocol.as_deref())
        || !session.summary.capture_evidence_complete
        || !session.summary.absolute_direction_supported
    {
        return Some("not a supported, complete nine-point residual capture".into());
    }
    if session.summary.model_parity.status != "MATCH" {
        return Some("capture-time EyeNet fingerprint does not match the supplied model".into());
    }
    if session.summary.anchors.iter().any(|anchor| !anchor.valid) {
        return Some("one or both train-only openness anchors are invalid".into());
    }
    None
}

fn fit_pooled_ridge(
    data: &[([f32; PHOTO_COUNT], f32)],
    independent_sessions: usize,
) -> Option<RidgeModel> {
    if independent_sessions < MIN_DISCOVERY_SESSIONS_FOR_RIDGE {
        return None;
    }
    fit_ridge(data)
}

fn model_comparison(
    sessions: &[SessionAnalysis],
) -> ([Option<RidgeModel>; 2], Vec<ModelRow>, DiscoveryPoolSummary) {
    let mut training: [Vec<([f32; PHOTO_COUNT], f32)>; 2] = std::array::from_fn(|_| Vec::new());
    let mut contributing: [Vec<usize>; 2] = std::array::from_fn(|_| Vec::new());
    let mut excluded_sessions = Vec::new();
    let mut seen_discovery_evidence = BTreeMap::<&str, usize>::new();
    for session in sessions
        .iter()
        .filter(|session| session.summary.role == "discovery")
    {
        if let Some(first_session) =
            seen_discovery_evidence.insert(&session.evidence_sha256, session.summary.session)
        {
            excluded_sessions.push(DiscoveryPoolExclusion {
                session: session.summary.session,
                reason: format!(
                    "statistical evidence is byte-identical to discovery session {first_session}"
                ),
            });
            continue;
        }
        if let Some(reason) = discovery_pool_exclusion(session) {
            excluded_sessions.push(DiscoveryPoolExclusion {
                session: session.summary.session,
                reason,
            });
            continue;
        }
        for eye in 0..2 {
            let blocks = regression_blocks(session, "train", eye, "gaze_error");
            if !blocks.is_empty() {
                contributing[eye].push(session.summary.session);
                training[eye].extend(blocks);
            }
        }
    }
    let ridge: [Option<RidgeModel>; 2] =
        std::array::from_fn(|eye| fit_pooled_ridge(&training[eye], contributing[eye].len()));
    let y_mean: [Option<f32>; 2] = std::array::from_fn(|eye| {
        (!training[eye].is_empty()).then(|| {
            training[eye].iter().map(|value| value.1).sum::<f32>() / training[eye].len() as f32
        })
    });
    let mut rows = Vec::new();
    for session in sessions {
        for split in ["train", "holdout"] {
            for eye in 0..2 {
                let data = regression_blocks(session, split, eye, "gaze_error");
                let role = session.summary.role.clone();
                let frozen = session.summary.role != "discovery" || split == "holdout";
                let model_parity_match = session.summary.model_parity.status == "MATCH";
                let intercept_metrics =
                    y_mean[eye].and_then(|mean| evaluate_predictions(&data, |_| mean));
                rows.push(ModelRow {
                    session: session.summary.session,
                    role: role.clone(),
                    split: split.into(),
                    eye: eye_name(eye).into(),
                    model: "intercept_only".into(),
                    n: data.len(),
                    mae: intercept_metrics.map(|value| value.0),
                    r2: intercept_metrics.map(|value| value.1),
                    frozen_from_discovery_train: frozen,
                    eligible_for_production: false,
                    status: if !model_parity_match {
                        "MODEL_FINGERPRINT_NOT_MATCHED_DIAGNOSTIC_ONLY"
                    } else if intercept_metrics.is_some() {
                        "DESCRIPTIVE_BLOCK_LEVEL"
                    } else {
                        "INSUFFICIENT_EVIDENCE"
                    }
                    .into(),
                });
                let ridge_metrics = ridge[eye]
                    .as_ref()
                    .and_then(|model| evaluate_predictions(&data, |photo| predict(model, photo)));
                rows.push(ModelRow {
                    session: session.summary.session,
                    role,
                    split: split.into(),
                    eye: eye_name(eye).into(),
                    model: "photometric_ridge_lambda_1".into(),
                    n: data.len(),
                    mae: ridge_metrics.map(|value| value.0),
                    r2: ridge_metrics.map(|value| value.1),
                    frozen_from_discovery_train: frozen,
                    eligible_for_production: false,
                    status: if !model_parity_match {
                        "MODEL_FINGERPRINT_NOT_MATCHED_DIAGNOSTIC_ONLY"
                    } else if ridge_metrics.is_some() {
                        "RESEARCH_ONLY_BLOCK_LEVEL"
                    } else if ridge[eye].is_none() {
                        "DISCOVERY_POOL_INSUFFICIENT"
                    } else {
                        "INSUFFICIENT_EVALUATION_EVIDENCE"
                    }
                    .into(),
                });
            }
        }
    }
    let discovery_pool = DiscoveryPoolSummary {
        minimum_independent_sessions: MIN_DISCOVERY_SESSIONS_FOR_RIDGE,
        contributing_session_ids_per_eye: contributing,
        one_second_blocks_per_eye: std::array::from_fn(|eye| training[eye].len()),
        fitted_per_eye: std::array::from_fn(|eye| ridge[eye].is_some()),
        // The current ridge is deliberately exploratory. A future preregistered
        // implementation must add leave-one-session-out feature/prediction stability
        // before this can ever become true.
        session_wise_feature_stability_evaluated: false,
        eligible_for_hypothesis_freeze: false,
        excluded_sessions,
        policy: "Only exact schema-v3 discovery-train blocks with matching capture-time EyeNet fingerprints are pooled. Every holdout and every confirmation session remains frozen evaluation only. The 41-block/three-session ridge gate is a numerical exploration floor, not a promotion gate; session-wise leave-one-session-out feature and prediction stability is not yet evaluated. Distinct paths are required, but independent re-wear/user provenance remains the operator's responsibility.".into(),
    };
    (ridge, rows, discovery_pool)
}

fn fit_ridge(data: &[([f32; PHOTO_COUNT], f32)]) -> Option<RidgeModel> {
    // Do not fit a nominally regularized 36-variable model to a handful of highly
    // autocorrelated blocks.  Ridge stabilizes coefficients; it does not create
    // independent evidence.  Future repeated nine-point sessions can clear this gate.
    if data.len() < MIN_RIDGE_BLOCKS {
        return None;
    }
    let mut means = [0.0; PHOTO_COUNT];
    for (x, _) in data {
        for feature in 0..PHOTO_COUNT {
            means[feature] += x[feature] as f64;
        }
    }
    for mean in &mut means {
        *mean /= data.len() as f64;
    }
    let mut scales = [0.0; PHOTO_COUNT];
    for (x, _) in data {
        for feature in 0..PHOTO_COUNT {
            scales[feature] += (x[feature] as f64 - means[feature]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / data.len() as f64).sqrt().max(1.0e-6);
    }
    let y_mean = data.iter().map(|value| value.1 as f64).sum::<f64>() / data.len() as f64;
    let mut matrix = vec![vec![0.0f64; PHOTO_COUNT]; PHOTO_COUNT];
    let mut rhs = vec![0.0f64; PHOTO_COUNT];
    for (x, y) in data {
        let z = std::array::from_fn::<_, PHOTO_COUNT, _>(|feature| {
            (x[feature] as f64 - means[feature]) / scales[feature]
        });
        let centered_y = *y as f64 - y_mean;
        for row in 0..PHOTO_COUNT {
            rhs[row] += z[row] * centered_y;
            for column in 0..PHOTO_COUNT {
                matrix[row][column] += z[row] * z[column];
            }
        }
    }
    for feature in 0..PHOTO_COUNT {
        matrix[feature][feature] += RIDGE_LAMBDA;
    }
    let solved = solve_linear(matrix, rhs)?;
    Some(RidgeModel {
        means: means.to_vec(),
        scales: scales.to_vec(),
        y_mean,
        beta: solved,
    })
}

fn predict(model: &RidgeModel, photo: [f32; PHOTO_COUNT]) -> f32 {
    let mut value = model.y_mean;
    for feature in 0..PHOTO_COUNT {
        value += model.beta[feature] * (photo[feature] as f64 - model.means[feature])
            / model.scales[feature];
    }
    value as f32
}

fn evaluate_predictions(
    data: &[([f32; PHOTO_COUNT], f32)],
    prediction: impl Fn([f32; PHOTO_COUNT]) -> f32,
) -> Option<(f32, f32)> {
    if data.len() < 5 {
        return None;
    }
    let mean = data.iter().map(|value| value.1).sum::<f32>() / data.len() as f32;
    let mut absolute = 0.0;
    let mut squared = 0.0;
    let mut total = 0.0;
    for (x, y) in data {
        let residual = *y - prediction(*x);
        absolute += residual.abs();
        squared += residual * residual;
        total += (*y - mean).powi(2);
    }
    Some((
        absolute / data.len() as f32,
        if total > 1.0e-8 {
            1.0 - squared / total
        } else {
            0.0
        },
    ))
}

fn solve_linear(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    let n = rhs.len();
    for pivot in 0..n {
        let best = (pivot..n).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1.0e-10 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let divisor = matrix[pivot][pivot];
        for column in pivot..n {
            matrix[pivot][column] /= divisor;
        }
        rhs[pivot] /= divisor;
        for row in 0..n {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for column in pivot..n {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn phase_quality(sessions: &[SessionAnalysis]) -> Vec<PhaseQualityRow> {
    let mut groups =
        BTreeMap::<(usize, String, String, String, usize, usize), Vec<&FrameRow>>::new();
    for session in sessions {
        for row in &session.rows {
            groups
                .entry((
                    row.session,
                    row.role.clone(),
                    row.split.into(),
                    row.kind.clone(),
                    row.phase_index,
                    row.eye,
                ))
                .or_default()
                .push(row);
        }
    }
    groups
        .into_iter()
        .map(|((session, role, split, kind, phase_index, eye), rows)| {
            let mut abstentions = BTreeMap::<String, usize>::new();
            for row in &rows {
                if row.pupil_status == "abstained" {
                    *abstentions.entry(row.pupil_reason.clone()).or_default() += 1;
                }
            }
            PhaseQualityRow {
                session,
                role,
                split,
                kind,
                phase_index,
                eye: eye_name(eye).into(),
                frames: rows.len(),
                stable_frames: rows.iter().filter(|row| row.stable).count(),
                pupil_diagnostic_rate: ratio(
                    rows.iter().filter(|row| row.pupil_x.is_some()).count(),
                    rows.len(),
                ),
                pupil_accepted_rate: ratio(
                    rows.iter()
                        .filter(|row| row.pupil_status == "accepted")
                        .count(),
                    rows.len(),
                ),
                lid_diagnostic_rate: ratio(
                    rows.iter().filter(|row| row.lid_aperture.is_some()).count(),
                    rows.len(),
                ),
                median_pupil_quality: median_options(
                    rows.iter().map(|row| Some(row.pupil_quality)),
                ),
                median_lid_quality: median_options(rows.iter().map(|row| Some(row.lid_quality))),
                median_photo_vertical_gradient: median_options(
                    rows.iter().map(|row| Some(row.photo[5])),
                ),
                abstentions: abstentions
                    .into_iter()
                    .map(|(reason, count)| format!("{reason}:{count}"))
                    .collect::<Vec<_>>()
                    .join("|"),
                interpretation: "landmark values are UNCALIBRATED_QA_ONLY".into(),
            }
        })
        .collect()
}

fn block_features(sessions: &[SessionAnalysis]) -> Vec<BlockRow> {
    let mut phase_origins = BTreeMap::<(usize, usize, usize), f32>::new();
    for session in sessions {
        for row in &session.rows {
            phase_origins
                .entry((row.session, row.phase_index, row.eye))
                .and_modify(|origin| *origin = origin.min(row.phase_time_s))
                .or_insert(row.phase_time_s);
        }
    }
    let mut groups = BTreeMap::<
        (usize, String, String, usize, usize, usize, String, String),
        Vec<&FrameRow>,
    >::new();
    for session in sessions {
        for row in &session.rows {
            let origin = phase_origins
                .get(&(row.session, row.phase_index, row.eye))
                .copied()
                .unwrap_or(row.phase_time_s);
            groups
                .entry((
                    row.session,
                    row.role.clone(),
                    row.split.into(),
                    row.phase_index,
                    (row.phase_time_s - origin).max(0.0).floor() as usize,
                    row.eye,
                    row.direction_source.clone(),
                    row.direction_bin.clone(),
                ))
                .or_default()
                .push(row);
        }
    }
    groups
        .into_iter()
        .map(
            |((session, role, split, phase_index, second_block, eye, source, bin), rows)| {
                BlockRow {
                    session,
                    role,
                    split,
                    phase_index,
                    second_block,
                    eye: eye_name(eye).into(),
                    direction_source: source,
                    direction_bin: bin,
                    frames: rows.len(),
                    span_s: block_span_s(&rows),
                    eligible_one_second_support: block_has_one_second_support(&rows),
                    normalized_open_median: median_options(
                        rows.iter().map(|row| row.normalized_open),
                    ),
                    gaze_error_median: median_options(rows.iter().map(|row| row.gaze_error)),
                    slow_error_median: median_options(rows.iter().map(|row| row.slow_error)),
                    squeeze_error_median: median_options(rows.iter().map(|row| row.squeeze_error)),
                    photo_medians: std::array::from_fn(|feature| {
                        median_options(rows.iter().map(|row| Some(row.photo[feature])))
                    }),
                }
            },
        )
        .collect()
}

fn direction_bins(sessions: &[SessionAnalysis]) -> Vec<DirectionRow> {
    let mut groups =
        BTreeMap::<(usize, String, String, usize, String, String), Vec<&FrameRow>>::new();
    for session in sessions {
        for row in &session.rows {
            // Direction summaries must contain only relaxed-open evidence. The v3
            // protocol also attaches targets to slow-close and blink actions; mixing
            // those actions here would manufacture a target-dependent openness loss.
            if !matches!(row.family, SampleFamily::Neutral | SampleFamily::GazeSweep) {
                continue;
            }
            groups
                .entry((
                    row.session,
                    row.role.clone(),
                    row.split.into(),
                    row.eye,
                    row.direction_source.clone(),
                    row.direction_bin.clone(),
                ))
                .or_default()
                .push(row);
        }
    }
    groups
        .into_iter()
        .map(
            |((session, role, split, eye, source, bin), rows)| DirectionRow {
                session,
                role,
                split,
                eye: eye_name(eye).into(),
                source: source.clone(),
                bin,
                frames: rows.len(),
                normalized_open_median: median_options(rows.iter().map(|row| row.normalized_open)),
                normalized_open_p10: percentile_options(
                    rows.iter().map(|row| row.normalized_open),
                    0.10,
                ),
                gaze_error_median: median_options(rows.iter().map(|row| row.gaze_error)),
                squeeze_p90: percentile_options(rows.iter().map(|row| Some(row.squeeze)), 0.90),
                status: if source == "relative_pupil_proxy_disabled" {
                    "DISABLED_LANDMARK_QC_NO_GO"
                } else if source == "commanded_target" {
                    "INSTRUCTED_CATEGORY_DESCRIPTIVE"
                } else {
                    "DESCRIPTIVE_ONLY"
                }
                .into(),
            },
        )
        .collect()
}

fn write_frame_features(out: &Path, sessions: &[SessionAnalysis]) -> Result<(), String> {
    let mut text = String::from(
        "session,role,schema,sample_index,source_index,split,kind,phase_index,phase_time_s,frame_generation_l,frame_generation_r,native_timestamp_us,eye,stable,commanded_target,direction_source,direction_bin,presence,open,squeeze,normalized_open,gaze_error,slow_error,closed_lift,squeeze_error,native_gaze_yaw,native_gaze_pitch,native_pupil_x,native_pupil_y,pupil_status,pupil_reason,pupil_quality,pupil_x,pupil_y,pupil_major,pupil_minor,pupil_angle,pupil_contrast,pupil_boundary,pupil_consensus,lid_status,lid_reason,lid_quality,lid_aperture,lid_path_strength,occlusion_proxy,temporal_jump_px,stereo_implausible,fixed_ir_flare_fraction,pupil_minus_global",
    );
    for feature in 0..PHOTO_COUNT {
        write!(text, ",{}", photo_name(feature)).unwrap();
    }
    text.push('\n');
    for session in sessions {
        for row in &session.rows {
            let mut fields = vec![
                row.session.to_string(),
                row.role.clone(),
                row.schema.to_string(),
                row.sample_index.to_string(),
                row.source_index.to_string(),
                row.split.into(),
                row.kind.clone(),
                row.phase_index.to_string(),
                format!("{:.6}", row.phase_time_s),
                row.frame_generation[0].to_string(),
                row.frame_generation[1].to_string(),
                row.native_timestamp_us
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                eye_name(row.eye).into(),
                row.stable.to_string(),
                row.commanded_target.clone().unwrap_or_default(),
                row.direction_source.clone(),
                row.direction_bin.clone(),
                format!("{:.9}", row.presence),
                format!("{:.9}", row.open),
                format!("{:.9}", row.squeeze),
                csv_opt(row.normalized_open),
                csv_opt(row.gaze_error),
                csv_opt(row.slow_error),
                csv_opt(row.closed_lift),
                csv_opt(row.squeeze_error),
                csv_opt(row.native_gaze_yaw),
                csv_opt(row.native_gaze_pitch),
                csv_opt(row.native_pupil_x),
                csv_opt(row.native_pupil_y),
                row.pupil_status.clone(),
                row.pupil_reason.clone(),
                format!("{:.9}", row.pupil_quality),
                csv_opt(row.pupil_x),
                csv_opt(row.pupil_y),
                csv_opt(row.pupil_major),
                csv_opt(row.pupil_minor),
                csv_opt(row.pupil_angle),
                csv_opt(row.pupil_contrast),
                csv_opt(row.pupil_boundary),
                row.pupil_consensus
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                row.lid_status.clone(),
                row.lid_reason.clone(),
                format!("{:.9}", row.lid_quality),
                csv_opt(row.lid_aperture),
                csv_opt(row.lid_path_strength),
                format!("{:.9}", row.occlusion_proxy),
                csv_opt(row.temporal_jump_px),
                row.stereo_implausible
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                format!("{:.9}", row.fixed_ir_flare_proxy),
                // Intentionally blank until the pupil detector is calibrated.
                String::new(),
            ];
            fields.extend(row.photo.iter().map(|value| format!("{value:.9}")));
            text.push_str(&csv_line(fields));
        }
    }
    write_text(out.join("frame_features.csv"), text)
}

fn write_phase_quality(out: &Path, rows: &[PhaseQualityRow]) -> Result<(), String> {
    let mut text = String::from("session,role,split,kind,phase_index,eye,frames,stable_frames,pupil_diagnostic_rate,pupil_accepted_rate,lid_diagnostic_rate,median_pupil_quality,median_lid_quality,median_photo_vertical_gradient,abstentions,interpretation\n");
    for row in rows {
        text.push_str(&csv_line([
            row.session.to_string(),
            row.role.clone(),
            row.split.clone(),
            row.kind.clone(),
            row.phase_index.to_string(),
            row.eye.clone(),
            row.frames.to_string(),
            row.stable_frames.to_string(),
            format!("{:.9}", row.pupil_diagnostic_rate),
            format!("{:.9}", row.pupil_accepted_rate),
            format!("{:.9}", row.lid_diagnostic_rate),
            csv_opt(row.median_pupil_quality),
            csv_opt(row.median_lid_quality),
            csv_opt(row.median_photo_vertical_gradient),
            row.abstentions.clone(),
            row.interpretation.clone(),
        ]));
    }
    write_text(out.join("phase_quality.csv"), text)
}

fn write_block_features(out: &Path, rows: &[BlockRow]) -> Result<(), String> {
    let mut text = String::from("session,role,split,phase_index,second_block,eye,direction_source,direction_bin,frames,span_s,eligible_one_second_support,normalized_open_median,gaze_error_median,slow_error_median,squeeze_error_median");
    for feature in 0..PHOTO_COUNT {
        write!(text, ",{}_median", photo_name(feature)).unwrap();
    }
    text.push('\n');
    for row in rows {
        let mut fields = vec![
            row.session.to_string(),
            row.role.clone(),
            row.split.clone(),
            row.phase_index.to_string(),
            row.second_block.to_string(),
            row.eye.clone(),
            row.direction_source.clone(),
            row.direction_bin.clone(),
            row.frames.to_string(),
            format!("{:.6}", row.span_s),
            row.eligible_one_second_support.to_string(),
            csv_opt(row.normalized_open_median),
            csv_opt(row.gaze_error_median),
            csv_opt(row.slow_error_median),
            csv_opt(row.squeeze_error_median),
        ];
        fields.extend(row.photo_medians.iter().copied().map(csv_opt));
        text.push_str(&csv_line(fields));
    }
    write_text(out.join("block_features.csv"), text)
}

fn write_direction_bins(out: &Path, rows: &[DirectionRow]) -> Result<(), String> {
    let mut text = String::from("session,role,split,eye,source,bin,frames,normalized_open_median,normalized_open_p10,gaze_error_median,squeeze_p90,status\n");
    for row in rows {
        text.push_str(&csv_line([
            row.session.to_string(),
            row.role.clone(),
            row.split.clone(),
            row.eye.clone(),
            row.source.clone(),
            row.bin.clone(),
            row.frames.to_string(),
            csv_opt(row.normalized_open_median),
            csv_opt(row.normalized_open_p10),
            csv_opt(row.gaze_error_median),
            csv_opt(row.squeeze_p90),
            row.status.clone(),
        ]));
    }
    write_text(out.join("direction_bins.csv"), text)
}

fn write_associations(out: &Path, rows: &[AssociationRow]) -> Result<(), String> {
    let mut text = String::from("session,role,split,eye,outcome,feature,n,spearman_rho,eligible_for_selection,selected_on_discovery_train,interpretation\n");
    for row in rows {
        text.push_str(&csv_line([
            row.session.to_string(),
            row.role.clone(),
            row.split.clone(),
            row.eye.clone(),
            row.outcome.clone(),
            row.feature.clone(),
            row.n.to_string(),
            csv_opt(row.spearman_rho),
            row.eligible_for_selection.to_string(),
            row.selected_on_discovery_train.to_string(),
            row.interpretation.clone(),
        ]));
    }
    write_text(out.join("associations.csv"), text)
}

fn write_models(out: &Path, rows: &[ModelRow]) -> Result<(), String> {
    let mut text = String::from("session,role,split,eye,model,n,mae,r2,frozen_from_discovery_train,eligible_for_production,status\n");
    for row in rows {
        text.push_str(&csv_line([
            row.session.to_string(),
            row.role.clone(),
            row.split.clone(),
            row.eye.clone(),
            row.model.clone(),
            row.n.to_string(),
            csv_opt(row.mae),
            csv_opt(row.r2),
            row.frozen_from_discovery_train.to_string(),
            row.eligible_for_production.to_string(),
            row.status.clone(),
        ]));
    }
    write_text(out.join("model_comparison.csv"), text)
}

fn write_summary(
    args: &Args,
    sessions: &[SessionAnalysis],
    associations: &[AssociationRow],
    model_rows: &[ModelRow],
    ridge: &[Option<RidgeModel>; 2],
    discovery_pool: &DiscoveryPoolSummary,
) -> Result<(), String> {
    let selected_features = associations
        .iter()
        .filter(|row| row.eligible_for_selection && row.selected_on_discovery_train)
        .map(|row| json!({"eye": row.eye, "feature": row.feature, "rho": row.spearman_rho, "blocks": row.n}))
        .collect::<Vec<_>>();
    let value = json!({
        "audit_schema": AUDIT_SCHEMA,
        "terminal_status": "RESEARCH_AUDIT_COMPLETE",
        "research_only": true,
        "production_config_changed": false,
        "recording": args.recording,
        "additional_discovery_recordings": args.discoveries,
        "confirmation_recordings": args.confirmations,
        "sessions": sessions.iter().map(|session| &session.summary).collect::<Vec<_>>(),
        "landmark_policy": {
            "status": "UNCALIBRATED_QA_ONLY",
            "included_in_ranking": false,
            "included_in_models": false,
            "pupil_minus_global": null,
        },
        "photometric_policy": {
            "candidate_independent": true,
            "raw_feature_source": "recorded camera pixels before production preprocessing and before every research intervention",
            "stability_source": "raw recorded pixels plus protocol labels and immutable capture geometry; independent of Tobii measurements and EyeNet output",
            "feature_count": PHOTO_COUNT,
            "association_selection_min_one_second_blocks_per_eye": MIN_ASSOCIATION_SELECTION_BLOCKS,
            "one_second_block_min_frames": MIN_BLOCK_FRAMES,
            "one_second_block_min_observed_span_s": MIN_BLOCK_SPAN_S,
            "one_second_block_max_adjacent_gap_s": MAX_BLOCK_GAP_S,
            "ridge_fit_min_one_second_blocks_per_eye": MIN_RIDGE_BLOCKS,
            "ridge_fit_min_independent_discovery_sessions": MIN_DISCOVERY_SESSIONS_FOR_RIDGE,
            "ridge_training_discovery_sessions_pooled": true,
            "selected_features_discovery_train_only": selected_features,
            "fixed_ir_flare_in_compact_model": false,
            "holdout_used_for_selection": false,
            "confirmation_used_for_selection": false,
            "ridge_minimum_interpretation": "numerical exploratory floor only; 48 blocks for 36 predictors remains statistically fragile",
            "session_wise_feature_stability_gate": {
                "required_for_hypothesis_freeze": true,
                "evaluated": false,
                "status": "NOT_EVALUATED_NO_GO",
                "requirements": "predeclared leave-one-session-out feature direction/rank stability; every training fold >=41 blocks; intercept beaten on every held-out session; no laterality reversal",
            },
        },
        "evidence_reuse_policy": {
            "displayed_discovery_holdout": "not selected on inside this run, but statistically consumed for any later human-revised hypothesis",
            "fresh_sealed_confirmation_required": true,
            "model_byte_or_crc_update_invalidates_prior_confirmation": true,
            "same_user_rewears_are_multi_user_generalization": false,
            "additional_users_units_required": "at least 2-3, including opposite failure laterality",
        },
        "direction_regression": {
            "enabled": false,
            "reason": "first-pass audit reports bins descriptively; landmark-driven direction regression is forbidden while QC is NO-GO",
        },
        "discovery_training_pool": discovery_pool,
        "ridge_model_per_eye": {"L": &ridge[0], "R": &ridge[1]},
        "model_comparison": model_rows,
        "promotion": "NO_GO_FROM_AUDIT_ALONE",
    });
    write_json(args.out.join("summary.json"), &value)
}

fn write_manifest(
    args: &Args,
    sessions: &[SessionAnalysis],
    model_bytes: &[u8],
) -> Result<(), String> {
    let input_manifest = sessions
        .iter()
        .map(|session| {
            Ok(json!({
                "session": session.summary.session,
                "role": session.summary.role,
                "path": session.root,
                "schema_version": session.summary.schema_version,
                "unit_id": session.summary.unit_id,
                "capture_evidence_complete": session.summary.capture_evidence_complete,
                "capture_timing_quality": &session.summary.capture_timing_quality,
                "model_parity": &session.summary.model_parity,
                "metadata_sha256": hash_file(&session.root.join("metadata.txt"))?,
                "samples_sha256": hash_file(&session.root.join("samples.csv"))?,
                "statistical_evidence_sha256": session.evidence_sha256,
                "ordered_frame_payloads_sha256": hash_session_frames(session)?,
                "frames": session.summary.frames,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let output_names = [
        "summary.json",
        "phase_quality.csv",
        "frame_features.csv",
        "block_features.csv",
        "direction_bins.csv",
        "associations.csv",
        "model_comparison.csv",
        "report.md",
    ];
    let output_inventory = output_names
        .iter()
        .map(|name| Ok(json!({"file": name, "sha256": hash_file(&args.out.join(name))?})))
        .collect::<Result<Vec<_>, String>>()?;
    let value = json!({
        "audit_schema": AUDIT_SCHEMA,
        "research_only": true,
        "production_config_changed": false,
        "supplied_model": {
            "path": args.model,
            "sha256": hash_bytes(model_bytes),
            "crc32": format!("{:08x}", crc32_ieee(model_bytes)),
            "bytes": model_bytes.len(),
        },
        "inputs": input_manifest,
        "outputs": output_inventory,
        "emit_overlays": args.emit_overlays,
        "analysis_input_contract": {
            "stability": "raw recorded pixels plus protocol labels and immutable capture geometry; no Tobii measurements or EyeNet output",
            "photometric_features": "raw recorded camera pixels before production preprocessing and before every research intervention",
            "outcomes": "EyeNet replay outputs joined by immutable sample_index; never fed back into stability or explanatory features",
        },
        "predeclared_thresholds": {
            "neutral_gaze_pupil_coverage_min": 0.80,
            "closed_like_false_pupil_max": 0.05,
            "temporal_jump_px": 15.0,
            "temporal_jump_rate_max": 0.05,
            "stereo_implausible_rate_max": 0.05,
            "native_gaze_coverage_min": 0.60,
            "native_direction_enter_deg": 7.0,
            "native_direction_center_deg": 5.0,
            "ridge_lambda": RIDGE_LAMBDA,
            "ridge_min_one_second_blocks_per_eye": MIN_RIDGE_BLOCKS,
            "ridge_min_independent_discovery_sessions": MIN_DISCOVERY_SESSIONS_FOR_RIDGE,
            "association_selection_min_one_second_blocks_per_eye": MIN_ASSOCIATION_SELECTION_BLOCKS,
            "one_second_block_min_frames": MIN_BLOCK_FRAMES,
            "one_second_block_min_observed_span_s": MIN_BLOCK_SPAN_S,
            "one_second_block_max_adjacent_gap_s": MAX_BLOCK_GAP_S,
            "commanded_stable_rows_per_phase_min": MIN_COMMANDED_STABLE_ROWS_PER_PHASE,
            "capture_expected_recorded_phases": EXPECTED_RECORDED_PHASES,
            "capture_required_rows_relaxed_open": 17,
            "capture_required_rows_slow_close_open": 30,
            "capture_required_rows_natural_blink": 16,
        },
        "promotion_gates": {
            "current_status": "NO_GO",
            "session_wise_feature_stability": "required but not evaluated",
            "fresh_externally_sealed_confirmation": "required; displayed discovery holdout is not a substitute",
            "model_fingerprint": "any model byte/CRC update invalidates prior confirmation for the new condition",
            "multi_user_transfer": "same-user re-wears are insufficient; require at least 2-3 additional users/units including opposite failure laterality",
        },
        "selection_policy": {
            "associations": "exploratory per-eye one-second-block correlations; strongest-feature selection requires the predeclared minimum and uses the primary discovery train only",
            "ridge": "pooled only from eligible independent discovery-train sessions; never from holdout or confirmation rows; 41 blocks/3 sessions is numerical exploration only and session-wise stability remains an unimplemented NO-GO promotion gate",
            "holdout": "report only inside this run; once displayed and used to revise a later hypothesis it is statistically consumed, not fresh confirmation",
            "confirmation": "fresh externally sealed report-only evidence; a model byte/CRC update invalidates prior confirmation for the new condition",
            "landmarks": "never selected; uncalibrated QA only",
            "generalization": "same-user re-wears are repeatability only; require at least 2-3 additional users/units including opposite failure laterality",
        },
    });
    write_json(args.out.join("manifest.json"), &value)
}

fn write_report(
    out: &Path,
    sessions: &[SessionAnalysis],
    associations: &[AssociationRow],
    models: &[ModelRow],
    discovery_pool: &DiscoveryPoolSummary,
) -> Result<(), String> {
    let mut text = String::from("# XR5 landmark and residual audit\n\n");
    text.push_str("Research-only diagnostic. It did not search, save, or apply a transform. Pupil and lid results are **UNCALIBRATED_QA_ONLY** and were excluded from associations and models.\n\n");
    for session in sessions {
        let summary = &session.summary;
        writeln!(text, "## Session {} - {}\n", summary.session, summary.role).unwrap();
        writeln!(text, "- Input schema: {}", summary.schema_version).unwrap();
        writeln!(
            text,
            "- Capture quota evidence complete: {}",
            summary.capture_evidence_complete
        )
        .unwrap();
        writeln!(
            text,
            "- Frames: {} train / {} holdout",
            summary.train_frames, summary.holdout_frames
        )
        .unwrap();
        writeln!(
            text,
            "- Full-population replay metric: {} train / {} holdout (evidence valid: {}/{})",
            csv_opt(summary.full_population_replay_train_score),
            csv_opt(summary.full_population_replay_holdout_score),
            summary.full_population_replay_evidence_valid[0],
            summary.full_population_replay_evidence_valid[1],
        )
        .unwrap();
        writeln!(
            text,
            "- Replay metric scope: `{}`; safe-fit protocol compatible: {}; directly comparable to production stratified fit score: {}",
            summary.replay_metric_scope,
            summary.safe_fit_protocol_compatible,
            summary.production_stratified_fit_score_comparable,
        )
        .unwrap();
        writeln!(
            text,
            "- EyeNet capture fingerprint parity: `{}` (`{}`)",
            summary.model_parity.status, summary.model_parity.replay_interpretation
        )
        .unwrap();
        writeln!(
            text,
            "- Capture generation nonzero coverage L/R: {:.1}% / {:.1}%; native timestamp coverage: {:.1}%",
            summary.capture_timing_quality.frame_generation_nonzero_coverage[0] * 100.0,
            summary.capture_timing_quality.frame_generation_nonzero_coverage[1] * 100.0,
            summary.capture_timing_quality.native_timestamp_coverage * 100.0,
        )
        .unwrap();
        writeln!(
            text,
            "- Direction mode: `{}`; absolute direction supported: {}",
            summary.direction_mode, summary.absolute_direction_supported
        )
        .unwrap();
        writeln!(
            text,
            "- Landmark QA: `{}` (`{}` selection scope)",
            summary.landmark_qc.status, summary.landmark_qc.selection_scope
        )
        .unwrap();
        writeln!(
            text,
            "- Neutral/Gaze diagnostic pupil coverage L/R: {:.1}% / {:.1}%",
            summary.landmark_qc.pupil_detection_neutral_gaze[0] * 100.0,
            summary.landmark_qc.pupil_detection_neutral_gaze[1] * 100.0
        )
        .unwrap();
        writeln!(
            text,
            "- Closed-like false pupil L/R: {:.1}% / {:.1}%",
            summary.landmark_qc.closed_like_false_pupil[0] * 100.0,
            summary.landmark_qc.closed_like_false_pupil[1] * 100.0
        )
        .unwrap();
        for reason in &summary.landmark_qc.reasons {
            writeln!(text, "  - {reason}").unwrap();
        }
        text.push('\n');
    }
    text.push_str("## Photometric audit\n\nOnly raw, candidate-independent global and 3x3 local photometric features were eligible. They were computed directly from recorded camera pixels before production preprocessing or any research intervention. Stability was frozen from raw pixels, labels, and immutable capture geometry without Tobii measurements or EyeNet output. `pupil_minus_global` and all pupil/lid values were excluded. Correlations are exploratory, diagnostic, and not causal. Strongest-feature selection requires at least 10 independent, cadence-continuous one-second blocks per eye.\n\n");
    let selected = associations
        .iter()
        .filter(|row| row.eligible_for_selection && row.selected_on_discovery_train)
        .collect::<Vec<_>>();
    if selected.is_empty() {
        text.push_str("There was insufficient discovery-train evidence to identify a strongest photometric association.\n\n");
    } else {
        for row in selected {
            writeln!(text,"Discovery-train {}-eye strongest predeclared feature: `{}` (block-level Spearman rho {}, blocks={}). Holdout and confirmation values were not consulted when choosing it.",row.eye,row.feature,csv_opt(row.spearman_rho),row.n).unwrap();
        }
        text.push('\n');
    }
    text.push_str("### Compact model comparison\n\nAll fits and evaluations use phase-local one-second medians and never pool eyes. The 36-feature ridge pools eligible discovery-train sessions only; every holdout and confirmation remains evaluation-only. It is not fitted until each eye has at least 41 blocks from at least three independent discovery recordings. That is only a numerical exploration floor: even a typical 48-block/36-predictor fit is fragile. Session-wise leave-one-out feature/prediction stability has not been evaluated, so no fitted ridge is eligible to freeze a hypothesis.\n\n");
    writeln!(
        text,
        "Discovery pool L: sessions {:?}, blocks {}, fitted {}; R: sessions {:?}, blocks {}, fitted {}.\n",
        discovery_pool.contributing_session_ids_per_eye[0],
        discovery_pool.one_second_blocks_per_eye[0],
        discovery_pool.fitted_per_eye[0],
        discovery_pool.contributing_session_ids_per_eye[1],
        discovery_pool.one_second_blocks_per_eye[1],
        discovery_pool.fitted_per_eye[1],
    )
    .unwrap();
    for excluded in &discovery_pool.excluded_sessions {
        writeln!(
            text,
            "- Discovery session {} excluded from fitting: {}",
            excluded.session, excluded.reason
        )
        .unwrap();
    }
    if !discovery_pool.excluded_sessions.is_empty() {
        text.push('\n');
    }
    text.push_str("| Session | Role | Split | Eye | Model | blocks | MAE | R2 |\n|---:|---|---|---|---|---:|---:|---:|\n");
    for row in models {
        writeln!(
            text,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            row.session,
            row.role,
            row.split,
            row.eye,
            row.model,
            row.n,
            csv_opt(row.mae),
            csv_opt(row.r2)
        )
        .unwrap();
    }
    text.push_str("\n## Evidence reuse and decision\n\nHoldout is not optimized on inside this run. However, once these displayed values are used to revise a candidate, threshold, or implementation, they are statistically consumed for that later hypothesis and are not fresh sealed confirmation. Any EyeNet model-byte/CRC update also invalidates prior confirmation for the new condition. Same-user re-wears establish repeatability only; generalization needs at least 2--3 additional users/units including opposite failure laterality.\n\nThis audit cannot authorize a production warp. A correction family must be frozen on discovery-train evidence, clear a predeclared session-wise stability gate, and pass fresh externally sealed confirmation.\n");
    write_text(out.join("report.md"), text)
}

fn write_overlays(out: &Path, sessions: &[SessionAnalysis]) -> Result<(), String> {
    let directory = out.join("overlays");
    std::fs::create_dir(&directory)
        .map_err(|error| format!("create overlays directory: {error}"))?;
    write_text(
        directory.join("README.txt"),
        "Explicitly requested biometric QA overlays. Crosses are uncalibrated diagnostic pupil locks, not ground truth.\n".into(),
    )?;
    for session in sessions {
        let mut emitted = BTreeSet::new();
        for row in &session.rows {
            if row.pupil_x.is_none() || !emitted.insert((row.phase_index, row.eye)) {
                continue;
            }
            let sample = &session.loaded.dataset.samples[row.sample_index];
            let (pixels, size) = if row.eye == 0 {
                (&sample.left, sample.left_size)
            } else {
                (&sample.right, sample.right_size)
            };
            let mut marked = pixels.clone();
            draw_cross(
                &mut marked,
                size,
                row.pupil_x.unwrap_or(0.0),
                row.pupil_y.unwrap_or(0.0),
            );
            let path = directory.join(format!(
                "session{}_{}_phase{:02}_{}_sample{:06}.png",
                row.session,
                row.role,
                row.phase_index,
                eye_name(row.eye),
                row.source_index
            ));
            write_gray_png(&path, size, &marked)?;
        }
    }
    Ok(())
}

fn draw_cross(pixels: &mut [u8], size: (u32, u32), x: f32, y: f32) {
    let (width, height) = (size.0 as isize, size.1 as isize);
    let (cx, cy) = (x.round() as isize, y.round() as isize);
    for delta in -7isize..=7 {
        for (xx, yy) in [(cx + delta, cy), (cx, cy + delta)] {
            if xx >= 0 && yy >= 0 && xx < width && yy < height {
                let index = yy as usize * width as usize + xx as usize;
                pixels[index] = if delta % 2 == 0 { 255 } else { 0 };
            }
        }
    }
}

fn write_gray_png(path: &Path, size: (u32, u32), pixels: &[u8]) -> Result<(), String> {
    let file = File::create(path).map_err(|error| format!("create {}: {error}", path.display()))?;
    let mut encoder = png::Encoder::new(file, size.0, size.1);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| format!("PNG header {}: {error}", path.display()))?;
    writer
        .write_image_data(pixels)
        .map_err(|error| format!("PNG data {}: {error}", path.display()))
}

fn parse_args() -> Result<Args, String> {
    const USAGE: &str = "Usage: xr5-landmark-residual-audit \
--recording <recording-dir> \
--model <EyePrediction.params> \
--out <new-output-dir> \
[--discovery-recording <independent-recording-dir>]... \
[--confirmation-recording <recording-dir>]... \
[--emit-overlays accepted]";

    let mut recording = None;
    let mut discoveries = Vec::new();
    let mut confirmations = Vec::new();
    let mut model = None;
    let mut out = None;
    let mut emit_overlays = false;
    let mut args = std::env::args_os().skip(1);
    while let Some(raw) = args.next() {
        let flag = raw.to_string_lossy();
        let mut next_path = |name: &str| -> Result<PathBuf, String> {
            args.next()
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} requires a path\n{USAGE}"))
        };
        match flag.as_ref() {
            "--recording" => recording = Some(next_path("--recording")?),
            "--discovery-recording" => discoveries.push(next_path("--discovery-recording")?),
            "--confirmation-recording" => {
                confirmations.push(next_path("--confirmation-recording")?)
            }
            "--model" => model = Some(next_path("--model")?),
            "--out" => out = Some(next_path("--out")?),
            "--emit-overlays" => {
                let consent = args
                    .next()
                    .ok_or_else(|| "--emit-overlays requires the literal 'accepted'".to_owned())?;
                if consent != "accepted" {
                    return Err(
                        "biometric overlays require explicit '--emit-overlays accepted' consent"
                            .into(),
                    );
                }
                emit_overlays = true;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {flag}\n{USAGE}")),
        }
    }

    let recording = recording.ok_or_else(|| format!("missing --recording\n{USAGE}"))?;
    let model = model.ok_or_else(|| format!("missing --model\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("missing --out\n{USAGE}"))?;
    if !recording.is_dir() {
        return Err(format!(
            "recording directory does not exist: {}",
            recording.display()
        ));
    }
    for discovery in &discoveries {
        if !discovery.is_dir() {
            return Err(format!(
                "discovery recording directory does not exist: {}",
                discovery.display()
            ));
        }
    }
    for confirmation in &confirmations {
        if !confirmation.is_dir() {
            return Err(format!(
                "confirmation recording directory does not exist: {}",
                confirmation.display()
            ));
        }
    }
    let mut unique_inputs = BTreeSet::new();
    for path in std::iter::once(&recording)
        .chain(discoveries.iter())
        .chain(confirmations.iter())
    {
        let canonical = std::fs::canonicalize(path)
            .map_err(|error| format!("resolve recording {}: {error}", path.display()))?;
        if !unique_inputs.insert(canonical) {
            return Err(format!(
                "the same recording cannot appear twice or in both discovery and confirmation: {}",
                path.display()
            ));
        }
    }
    if !model.is_file() {
        return Err(format!("model file does not exist: {}", model.display()));
    }
    if out.exists() {
        return Err(format!(
            "output path already exists; choose a new directory to preserve audit provenance: {}",
            out.display()
        ));
    }
    Ok(Args {
        recording,
        discoveries,
        confirmations,
        model,
        out,
        emit_overlays,
    })
}

fn percentile(values: &[f32], quantile: f32) -> Option<f32> {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return None;
    }
    finite.sort_by(f32::total_cmp);
    let position = quantile.clamp(0.0, 1.0) * (finite.len() - 1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    Some(finite[lower] + (finite[upper] - finite[lower]) * fraction)
}

fn percentile_options(values: impl Iterator<Item = Option<f32>>, quantile: f32) -> Option<f32> {
    percentile(&values.flatten().collect::<Vec<_>>(), quantile)
}

fn median_options(values: impl Iterator<Item = Option<f32>>) -> Option<f32> {
    percentile_options(values, 0.50)
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn block_span_s(rows: &[&FrameRow]) -> f32 {
    let minimum = rows
        .iter()
        .map(|row| row.phase_time_s)
        .min_by(f32::total_cmp);
    let maximum = rows
        .iter()
        .map(|row| row.phase_time_s)
        .max_by(f32::total_cmp);
    minimum.zip(maximum).map_or(0.0, |(min, max)| max - min)
}

fn block_has_one_second_support(rows: &[&FrameRow]) -> bool {
    if rows.len() < MIN_BLOCK_FRAMES || block_span_s(rows) < MIN_BLOCK_SPAN_S {
        return false;
    }
    let mut times = rows.iter().map(|row| row.phase_time_s).collect::<Vec<_>>();
    if times.iter().any(|time| !time.is_finite()) {
        return false;
    }
    times.sort_by(f32::total_cmp);
    times
        .windows(2)
        .all(|pair| pair[1] - pair[0] <= MAX_BLOCK_GAP_S)
}

fn spearman(left: &[f32], right: &[f32]) -> Option<f32> {
    if left.len() != right.len() || left.len() < 2 {
        return None;
    }
    let pairs = left
        .iter()
        .copied()
        .zip(right.iter().copied())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .collect::<Vec<_>>();
    if pairs.len() < 2 {
        return None;
    }
    let x = pairs.iter().map(|pair| pair.0).collect::<Vec<_>>();
    let y = pairs.iter().map(|pair| pair.1).collect::<Vec<_>>();
    let xr = average_ranks(&x);
    let yr = average_ranks(&y);
    let xmean = xr.iter().sum::<f32>() / xr.len() as f32;
    let ymean = yr.iter().sum::<f32>() / yr.len() as f32;
    let mut covariance = 0.0;
    let mut xvariance = 0.0;
    let mut yvariance = 0.0;
    for index in 0..xr.len() {
        let dx = xr[index] - xmean;
        let dy = yr[index] - ymean;
        covariance += dx * dy;
        xvariance += dx * dx;
        yvariance += dy * dy;
    }
    let denominator = (xvariance * yvariance).sqrt();
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
        let average = ((begin + 1) as f32 + end as f32) * 0.5;
        for index in begin..end {
            ranks[order[index]] = average;
        }
        begin = end;
    }
    ranks
}

fn eye_name(eye: usize) -> &'static str {
    if eye == 0 {
        "L"
    } else {
        "R"
    }
}

fn csv_opt(value: Option<f32>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.9}"))
        .unwrap_or_default()
}

fn csv_field(value: impl AsRef<str>) -> String {
    let value = value.as_ref();
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn csv_line(fields: impl IntoIterator<Item = String>) -> String {
    let mut line = fields
        .into_iter()
        .map(csv_field)
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

fn write_text(path: PathBuf, text: String) -> Result<(), String> {
    std::fs::write(&path, text).map_err(|error| format!("write {}: {error}", path.display()))
}

fn write_json(path: PathBuf, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("serialize {}: {error}", path.display()))?;
    std::fs::write(&path, bytes).map_err(|error| format!("write {}: {error}", path.display()))
}

fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_file(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(hash_bytes(&bytes))
}

/// Fingerprint the actual statistical evidence, excluding mutable path/provenance
/// metadata. This prevents an identical recording copied to another directory from
/// being counted as another independent discovery session.
fn hash_recording_evidence(
    root: &Path,
    frames: &[xr5_recording::RecordingFrame],
) -> Result<String, String> {
    let mut digest = Sha256::new();
    let samples_path = root.join("samples.csv");
    let samples = std::fs::read(&samples_path)
        .map_err(|error| format!("read {}: {error}", samples_path.display()))?;
    digest.update((samples.len() as u64).to_le_bytes());
    digest.update(samples);
    for frame in frames {
        digest.update((frame.source_index as u64).to_le_bytes());
        for path in [&frame.left_file, &frame.right_file] {
            let resolved = if path.is_absolute() {
                path.clone()
            } else {
                root.join(path)
            };
            let bytes = std::fs::read(&resolved)
                .map_err(|error| format!("read frame payload {}: {error}", resolved.display()))?;
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn hash_session_frames(session: &SessionAnalysis) -> Result<String, String> {
    let mut digest = Sha256::new();
    for frame in &session.loaded.frames {
        digest.update((frame.source_index as u64).to_le_bytes());
        for path in [&frame.left_file, &frame.right_file] {
            let resolved = if path.is_absolute() {
                path.clone()
            } else {
                session.root.join(path)
            };
            let bytes = std::fs::read(&resolved)
                .map_err(|error| format!("read frame payload {}: {error}", resolved.display()))?;
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sranibro_rs::core::types::MlGeometry;
    use sranibro_rs::geometry_calib::SampleKind;
    use sranibro_rs::geometry_fitrun::research_stability_report;

    #[test]
    fn percentile_interpolates_and_ignores_non_finite_values() {
        assert_eq!(percentile(&[0.0, f32::NAN, 10.0], 0.5), Some(5.0));
        assert_eq!(percentile(&[2.0], 0.1), Some(2.0));
        assert_eq!(percentile(&[], 0.5), None);
    }

    #[test]
    fn spearman_uses_average_tie_ranks() {
        let rho = spearman(&[1.0, 1.0, 2.0, 3.0], &[4.0, 4.0, 2.0, 1.0]).unwrap();
        assert!((rho + 1.0).abs() < 1.0e-6);
        assert_eq!(spearman(&[1.0, 1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn native_direction_dead_band_is_explicit() {
        assert_eq!(native_direction_bin(0.0, 0.0), "center");
        assert_eq!(native_direction_bin(6.0, 0.0), "transition");
        assert_eq!(native_direction_bin(8.0, -8.0), "yaw_pos_pitch_neg");
    }

    #[test]
    fn commanded_mode_requires_exact_complete_repeated_protocol() {
        let mut evidence = Vec::new();
        let mut phase = 0usize;
        for split in ["train", "holdout"] {
            for target in GazeTarget::ALL {
                for _ in 0..2 {
                    let family = if target == GazeTarget::Center {
                        SampleFamily::Neutral
                    } else {
                        SampleFamily::GazeSweep
                    };
                    // Mirror a minimally complete 17-frame RelaxedOpen phase after
                    // production stability selection: eleven retained, six excluded.
                    for sample in 0..17 {
                        evidence.push((
                            0,
                            sample < MIN_COMMANDED_STABLE_ROWS_PER_PHASE,
                            split,
                            phase,
                            family,
                            Some(target.as_str()),
                        ));
                    }
                    phase += 1;
                }
            }
        }
        assert!(commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v1"),
            true,
            evidence.iter().copied(),
        ));
        assert!(commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v2"),
            true,
            evidence.iter().copied(),
        ));
        assert!(commanded_protocol_evidence_complete(
            4,
            Some("xr5_landmark_residual_audit_v3"),
            true,
            evidence.iter().copied(),
        ));
        assert!(!commanded_protocol_evidence_complete(
            3,
            Some("safe_geometry_fit_v1"),
            true,
            evidence.iter().copied(),
        ));
        assert!(!commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v1"),
            false,
            evidence.iter().copied(),
        ));

        let mut incomplete = evidence.clone();
        let missing_stable = incomplete
            .iter()
            .rposition(|row| row.3 == phase - 1 && row.1)
            .unwrap();
        incomplete.remove(missing_stable);
        assert!(!commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v1"),
            true,
            incomplete.iter().copied(),
        ));

        let mut inconsistent = evidence.clone();
        inconsistent.push((0, true, "train", 0, SampleFamily::GazeSweep, Some("left")));
        assert!(!commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v1"),
            true,
            inconsistent,
        ));

        let mut wrong_family = evidence;
        for row in wrong_family.iter_mut().filter(|row| row.3 == phase - 1) {
            row.4 = SampleFamily::NaturalBlinks;
        }
        assert!(!commanded_protocol_evidence_complete(
            3,
            Some("xr5_landmark_residual_audit_v1"),
            true,
            wrong_family,
        ));
    }

    #[test]
    fn capture_quota_metadata_must_match_every_observed_phase() {
        let mut fields = BTreeMap::new();
        fields.insert(
            "capture_protocol".into(),
            "xr5_landmark_residual_audit_v1".into(),
        );
        fields.insert("target_repetitions_per_split".into(), "2".into());
        fields.insert("capture_evidence_complete".into(), "true".into());
        fields.insert("capture_missing_phase_ids".into(), String::new());
        let phase_specs = (1..=EXPECTED_RECORDED_PHASES)
            .map(|phase| {
                if phase <= 4 {
                    (phase, sranibro_rs::geometry_calib::SampleKind::Neutral, 17)
                } else if phase <= 36 {
                    (
                        phase,
                        sranibro_rs::geometry_calib::SampleKind::GazeSweep,
                        17,
                    )
                } else if phase <= 46 {
                    (
                        phase,
                        sranibro_rs::geometry_calib::SampleKind::SlowClose,
                        30,
                    )
                } else {
                    (
                        phase,
                        sranibro_rs::geometry_calib::SampleKind::NaturalBlinks,
                        16,
                    )
                }
            })
            .collect::<Vec<_>>();
        fields.insert(
            "capture_phase_sample_counts".into(),
            phase_specs
                .iter()
                .map(|(phase, _, required)| format!("{phase}:{required}/{required}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        let mut metadata = xr5_recording::RecordingMetadata {
            schema_version: 3,
            raw: String::new(),
            fields,
        };
        let mut samples = Vec::new();
        for (phase_index, kind, required) in phase_specs {
            for _ in 0..required {
                samples.push(GeometrySample {
                    kind,
                    commanded_target: Some(GazeTarget::Center),
                    expected_open: None,
                    phase_time_s: 1.0,
                    left: Vec::new(),
                    right: Vec::new(),
                    left_size: (0, 0),
                    right_size: (0, 0),
                    brightness_affine: [[1.0, 0.0]; 2],
                    native_open: [None; 2],
                    native_gaze: [None; 2],
                    native_pupil_pos: [None; 2],
                    frame_generation: [1; 2],
                    native_timestamp_us: None,
                    phase_index,
                });
            }
        }
        let mut dataset = GeometryDataset { samples };
        assert!(capture_evidence_metadata_complete(&metadata, &dataset));
        metadata.fields.insert(
            "capture_protocol".into(),
            "xr5_landmark_residual_audit_v2".into(),
        );
        assert!(capture_evidence_metadata_complete(&metadata, &dataset));
        metadata.fields.insert(
            "capture_protocol".into(),
            "xr5_landmark_residual_audit_v3".into(),
        );
        assert!(capture_evidence_metadata_complete(&metadata, &dataset));
        dataset.samples.pop();
        assert!(!capture_evidence_metadata_complete(&metadata, &dataset));
    }

    #[test]
    fn relative_proxy_never_claims_anatomical_direction() {
        let label = relative_pupil_bin(10.0, -10.0);
        assert!(label.starts_with("raw_x_"));
        assert!(!label.contains("left"));
        assert!(!label.contains("right"));
    }

    fn timed_train_gaze_row(
        sample_index: usize,
        eye: usize,
        phase_index: usize,
        phase_time_s: f32,
        stable: bool,
    ) -> FrameRow {
        let mut photo = [0.0; PHOTO_COUNT];
        for (feature, value) in photo.iter_mut().enumerate() {
            *value = phase_index as f32 * 0.10 + feature as f32 * 0.001 + phase_time_s * 0.0001;
        }
        FrameRow {
            session: 0,
            role: "discovery".into(),
            schema: 3,
            sample_index,
            source_index: sample_index,
            split: "train",
            kind: "gaze_sweep".into(),
            family: SampleFamily::GazeSweep,
            phase_index,
            phase_time_s,
            frame_generation: [sample_index as u64 + 1; 2],
            native_timestamp_us: Some(sample_index as u64 + 1),
            eye,
            stable,
            commanded_target: Some("left".into()),
            direction_source: "commanded_target".into(),
            direction_bin: "left".into(),
            presence: 1.0,
            open: 0.8,
            squeeze: 0.0,
            normalized_open: Some(0.8),
            gaze_error: Some(phase_index as f32 * 0.01 + phase_time_s * 0.001),
            slow_error: None,
            closed_lift: None,
            squeeze_error: Some(0.0),
            native_gaze_yaw: None,
            native_gaze_pitch: None,
            native_pupil_x: None,
            native_pupil_y: None,
            pupil_status: "not_evaluated".into(),
            pupil_reason: String::new(),
            pupil_quality: 0.0,
            pupil_x: None,
            pupil_y: None,
            pupil_major: None,
            pupil_minor: None,
            pupil_angle: None,
            pupil_contrast: None,
            pupil_boundary: None,
            pupil_consensus: None,
            lid_status: "not_evaluated".into(),
            lid_reason: String::new(),
            lid_quality: 0.0,
            lid_aperture: None,
            lid_path_strength: None,
            occlusion_proxy: 0.0,
            temporal_jump_px: None,
            stereo_implausible: None,
            photo,
            fixed_ir_flare_proxy: 0.0,
        }
    }

    #[test]
    fn one_second_support_rejects_clustered_frames_across_a_large_gap() {
        let continuous = (0..16)
            .map(|index| timed_train_gaze_row(index, 0, 2, index as f32 * 0.064, true))
            .collect::<Vec<_>>();
        let continuous_refs = continuous.iter().collect::<Vec<_>>();
        assert!(block_has_one_second_support(&continuous_refs));

        // Ten frames and 0.98 s of endpoint span used to pass even though the middle
        // 0.78 s had no evidence. The cadence gap must now make it ineligible.
        let times = [0.00, 0.04, 0.08, 0.12, 0.16, 0.94, 0.95, 0.96, 0.97, 0.98];
        let clustered = times
            .iter()
            .enumerate()
            .map(|(index, time)| timed_train_gaze_row(index, 0, 2, *time, true))
            .collect::<Vec<_>>();
        let clustered_refs = clustered.iter().collect::<Vec<_>>();
        assert_eq!(clustered_refs.len(), MIN_BLOCK_FRAMES);
        assert!(block_span_s(&clustered_refs) >= MIN_BLOCK_SPAN_S);
        assert!(!block_has_one_second_support(&clustered_refs));
    }

    #[test]
    fn real_protocol_timing_yields_sixteen_train_gaze_blocks_per_eye_and_reaches_ridge() {
        // OPEN_SECONDS=2.20 and OPEN_SETTLE_SECONDS=0.55 leave 1.65 s of
        // recorded evidence. Exercise the production stability classifier at a
        // realistic 64 ms UI cadence: after the 0.50 s trim and two-frame gaze
        // departure, 0.576..1.536 still supports a 0.96 s block with 16 frames.
        // Each train pass has eight non-centre targets.
        let non_center_phases = (1usize..=18)
            .filter(|phase| !matches!(phase, 1 | 10))
            .collect::<Vec<_>>();
        assert_eq!(non_center_phases.len(), 16);

        let mut dataset = GeometryDataset::default();
        for phase_index in 1usize..=18 {
            let center = matches!(phase_index, 1 | 10);
            for frame in 0..=25 {
                let phase_time_s = frame as f32 * 0.064;
                let pixels = vec![if center { 0 } else { 200 }; 16];
                dataset.samples.push(GeometrySample {
                    kind: if center {
                        SampleKind::Neutral
                    } else {
                        SampleKind::GazeSweep
                    },
                    commanded_target: Some(if center {
                        GazeTarget::Center
                    } else {
                        GazeTarget::Left
                    }),
                    expected_open: None,
                    phase_time_s,
                    left: pixels.clone(),
                    right: pixels,
                    left_size: (4, 4),
                    right_size: (4, 4),
                    brightness_affine: [[1.0, 0.0]; 2],
                    native_open: [None; 2],
                    native_gaze: [None; 2],
                    native_pupil_pos: [None; 2],
                    frame_generation: [dataset.samples.len() as u64 + 1; 2],
                    native_timestamp_us: None,
                    phase_index,
                });
            }
        }
        let stability = research_stability_report(&dataset, [MlGeometry::default(); 2]);
        assert_eq!(stability.invalid_static_phases, 0);

        let mut rows = Vec::new();
        for (sample_index, sample) in dataset.samples.iter().enumerate() {
            if sample.kind != SampleKind::GazeSweep {
                continue;
            }
            for eye in 0..2 {
                rows.push(timed_train_gaze_row(
                    sample_index,
                    eye,
                    sample.phase_index,
                    sample.phase_time_s,
                    stability.flags[sample_index],
                ));
            }
        }

        let per_eye = std::array::from_fn::<_, 2, _>(|eye| {
            regression_blocks_from_rows(&rows, "train", eye, "gaze_error")
        });
        assert_eq!(per_eye[0].len(), 16);
        assert_eq!(per_eye[1].len(), 16);

        let pooled =
            (0..3)
                .flat_map(|session| {
                    per_eye[0].iter().copied().enumerate().map(
                        move |(block, (mut photo, outcome))| {
                            photo[(session * 7 + block) % PHOTO_COUNT] += session as f32 * 0.01;
                            (photo, outcome + session as f32 * 0.005)
                        },
                    )
                })
                .collect::<Vec<_>>();
        assert_eq!(pooled.len(), 48);
        assert_eq!(MIN_RIDGE_BLOCKS, 41);
        assert!(fit_pooled_ridge(&pooled, 3).is_some());
    }

    #[test]
    fn ridge_refuses_underdetermined_evidence() {
        let rows = vec![([0.0; PHOTO_COUNT], 0.0); MIN_RIDGE_BLOCKS - 1];
        assert!(fit_ridge(&rows).is_none());
    }

    #[test]
    fn pooled_ridge_requires_three_independent_discovery_sessions() {
        let rows = (0..MIN_RIDGE_BLOCKS)
            .map(|index| {
                let mut photo = [0.0; PHOTO_COUNT];
                photo[index % PHOTO_COUNT] = index as f32 + 1.0;
                (photo, index as f32 * 0.01)
            })
            .collect::<Vec<_>>();
        assert!(fit_pooled_ridge(&rows, 2).is_none());
        assert!(fit_pooled_ridge(&rows, 3).is_some());
    }

    #[test]
    fn capture_model_fingerprint_is_gated() {
        let bytes = b"123456789";
        assert_eq!(crc32_ieee(bytes), 0xcbf4_3926);
        let mut fields = BTreeMap::new();
        fields.insert("eyelid_model_crc32".into(), "cbf43926".into());
        fields.insert("eyelid_model_bytes".into(), "9".into());
        let metadata = xr5_recording::RecordingMetadata {
            schema_version: 3,
            raw: String::new(),
            fields,
        };
        assert_eq!(model_parity(&metadata, bytes).status, "MATCH");
        assert_eq!(model_parity(&metadata, b"different").status, "MISMATCH");

        let unknown = xr5_recording::RecordingMetadata {
            schema_version: 1,
            raw: String::new(),
            fields: BTreeMap::new(),
        };
        assert_eq!(model_parity(&unknown, bytes).status, "UNKNOWN");
    }

    #[test]
    fn csv_fields_escape_commas_quotes_and_newlines() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("a\"b"), "\"a\"\"b\"");
    }
}
