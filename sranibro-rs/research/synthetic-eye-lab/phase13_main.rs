//! Preregistered Phase 1.3 synthetic EyeNet study entry point.
//!
//! This feature-gated binary accepts only the sealed renderer atlas, the frozen
//! model, and a new artifact destination. It has no recording or production-state
//! input.

mod model;
mod phase13;
mod phase13_output;
mod renderer;

use std::error::Error;
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sranibro_rs::ml::{eye_net::EyeNet, tvm_params};

use phase13::{
    CaseRecord, PreparedPlan, RawCaseResult, RendererPlan, StageAnalysis, StageCase, StageStatus,
};
use phase13_output::{
    repository_state, require_release_build, validate_atlas_dir, verify_decision_artifact,
    verify_preregistration, verify_runtime_identity, write_stage_artifact, RepositoryState,
    ValidatedAtlas, VerifiedArtifact, AMENDMENT_COMMIT, BUILD_COMMIT, BUILD_PROFILE,
    PREREGISTRATION_COMMIT,
};

const FROZEN_MODEL_BYTES: usize = phase13::MODEL_LENGTH as usize;
const FROZEN_MODEL_SHA256: &str = phase13::MODEL_SHA256;

fn main() {
    match run() {
        Ok(RunOutcome::Completed(path)) => println!("Phase 1.3 artifact: {}", path.display()),
        Ok(RunOutcome::Help) => print_help(),
        Err(error) => {
            eprintln!("synthetic-eye-phase13: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<RunOutcome, Box<dyn Error>> {
    let Some(args) = Args::parse(std::env::args().skip(1))? else {
        return Ok(RunOutcome::Help);
    };
    execute(args)
}

fn execute(args: Args) -> Result<RunOutcome, Box<dyn Error>> {
    require_release_build()?;
    reject_existing_output(&args.out)?;

    let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repository = repository_state(repo)?;
    verify_runtime_identity(&repository)?;
    verify_preregistration(repo)?;

    let decision = match (&args.stage, &args.decision, &args.decision_seal) {
        (Stage::Decision, None, None) => None,
        (Stage::Confirmation, Some(path), Some(expected_seal)) => {
            let artifact = verify_decision_artifact(path, expected_seal)?;
            let disposition = require_compatible_decision(&artifact, &repository)?;
            if disposition == DecisionDisposition::Inconclusive {
                return seal_inconclusive_decision_confirmation(
                    &args,
                    repo,
                    &repository,
                    &artifact,
                );
            }
            Some(artifact)
        }
        _ => return Err("internal command/decision-seal mismatch".into()),
    };

    let atlas = match validate_atlas_dir(&args.atlas) {
        Ok(atlas) => atlas,
        Err(error) => {
            return seal_renderer_failure(
                &args,
                repo,
                &repository,
                decision.as_ref(),
                None,
                "atlas_validation_failed",
                &error.to_string(),
            )
        }
    };
    execute_with_core(args, repo, repository, atlas, decision)
}

fn execute_with_core(
    args: Args,
    repo: &Path,
    repository: RepositoryState,
    atlas: ValidatedAtlas,
    decision: Option<VerifiedArtifact>,
) -> Result<RunOutcome, Box<dyn Error>> {
    let prepared = match phase13::validate_atlas_and_prepare(&args.atlas) {
        Ok(prepared) => prepared,
        Err(error) => {
            return seal_renderer_failure(
                &args,
                repo,
                &repository,
                decision.as_ref(),
                Some(&atlas),
                "renderer_preparation_failed",
                &error,
            )
        }
    };
    // A mismatch with the already sealed decision is invalid/tampered input,
    // not a new confirmation-side renderer result.  It must not publish an
    // artifact under the total renderer-failure procedure.
    if let Some(decision) = decision.as_ref() {
        require_decision_plan(decision, &prepared)?;
    }
    let cases = match finish_renderer_preparation(&args, &prepared) {
        Ok(cases) => cases,
        Err(error) => {
            return seal_renderer_failure(
                &args,
                repo,
                &repository,
                decision.as_ref(),
                Some(&atlas),
                "renderer_integrity_failed",
                &error,
            )
        }
    };
    let core_stage = args.stage.core();

    recheck_repository(repo, &repository, "before model loading")?;
    let mut models = match load_frozen_models(&args.model) {
        Ok(models) => models,
        Err(error) => {
            let analysis = artifact_analysis(core_stage, error.to_string());
            return seal_analysis_artifact(
                &args,
                repo,
                &repository,
                &atlas,
                decision.as_ref(),
                &prepared,
                &cases,
                &analysis,
                false,
                None,
            );
        }
    };

    let first = infer_in_order(&mut models.first, cases.iter());
    let second_reverse = infer_in_order(&mut models.second, cases.iter().rev());
    let model_change = verify_model_unchanged(&args.model, &models.bytes).err();
    let mut analysis = phase13::analyze_stage(core_stage, &cases, &first, &second_reverse);
    if let Some(error) = model_change {
        analysis.status = StageStatus::InconclusiveArtifact;
        analysis.response_class = None;
        analysis.artifact_error = Some(error.to_string());
        analysis.eyes = None;
    }
    let model_sha256 = models.sha256.clone();
    drop(models);

    seal_analysis_artifact(
        &args,
        repo,
        &repository,
        &atlas,
        decision.as_ref(),
        &prepared,
        &cases,
        &analysis,
        true,
        Some(&model_sha256),
    )
}

fn finish_renderer_preparation(
    args: &Args,
    prepared: &PreparedPlan,
) -> Result<Vec<StageCase>, String> {
    validate_prepared_identity(prepared).map_err(|error| error.to_string())?;
    let cases = match args.stage {
        Stage::Decision => prepared.decision_cases.clone(),
        Stage::Confirmation => prepared.confirmation_cases.clone(),
    };
    let rebuilt = phase13::build_stage_cases(&prepared.plan, args.stage.core())
        .map_err(|error| format!("rebuild stage cases: {error}"))?;
    require_same_cases(&cases, &rebuilt).map_err(|error| error.to_string())?;
    Ok(cases)
}

fn validate_prepared_identity(prepared: &PreparedPlan) -> Result<(), Box<dyn Error>> {
    if prepared.plan.version != phase13::VERSION
        || prepared.plan.preregistration_commit != PREREGISTRATION_COMMIT
        || prepared.plan.global_axis != "whole_image_mean_gray"
        || prepared.plan.renderer_decision != "go"
        || prepared.plan.excluded_pairs != vec![[36, 40]]
        || prepared.plan.preparation_repetitions != 2
    {
        return Err("renderer plan identity did not match the frozen protocol".into());
    }
    if phase13_output::sha256_hex(&prepared.plan_json) == prepared.plan_sha256 {
        return Err(
            "renderer plan used an unqualified raw hash instead of its frozen domain".into(),
        );
    }
    Ok(())
}

fn require_same_cases(first: &[StageCase], second: &[StageCase]) -> Result<(), Box<dyn Error>> {
    if first.len() != second.len() {
        return Err("stage case rebuild length mismatch".into());
    }
    for (index, (first, second)) in first.iter().zip(second).enumerate() {
        if first.record != second.record
            || first.stereo.left.pixels != second.stereo.left.pixels
            || first.stereo.right.pixels != second.stereo.right.pixels
            || !same_f32_bits(&first.tensor, &second.tensor)
        {
            return Err(format!("stage case rebuild mismatch at {index}").into());
        }
    }
    Ok(())
}

fn same_f32_bits(first: &[f32], second: &[f32]) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(first, second)| first.to_bits() == second.to_bits())
}

fn recheck_repository(
    repo: &Path,
    expected: &RepositoryState,
    boundary: &str,
) -> Result<(), Box<dyn Error>> {
    let current = repository_state(repo)?;
    if current != *expected {
        return Err(format!("repository identity changed {boundary}").into());
    }
    verify_runtime_identity(&current)
}

fn infer_in_order<'a>(
    net: &mut EyeNet,
    cases: impl IntoIterator<Item = &'a StageCase>,
) -> Vec<RawCaseResult> {
    cases
        .into_iter()
        .map(|case| RawCaseResult {
            id: case.id().to_owned(),
            raw: net.forward_one(&case.tensor),
        })
        .collect()
}

fn artifact_analysis(stage: phase13::Stage, error: String) -> StageAnalysis {
    StageAnalysis {
        stage,
        status: StageStatus::InconclusiveArtifact,
        response_class: None,
        flags: Vec::new(),
        artifact_error: Some(error),
        first_raw_bits: Vec::new(),
        second_raw_bits: Vec::new(),
        eyes: None,
    }
}

#[allow(clippy::too_many_arguments)]
fn seal_analysis_artifact(
    args: &Args,
    repo: &Path,
    repository: &RepositoryState,
    atlas: &ValidatedAtlas,
    decision: Option<&VerifiedArtifact>,
    prepared: &PreparedPlan,
    cases: &[StageCase],
    analysis: &StageAnalysis,
    model_loaded: bool,
    model_sha256: Option<&str>,
) -> Result<RunOutcome, Box<dyn Error>> {
    let case_records: Vec<_> = cases.iter().map(|case| &case.record).collect();
    let analysis_bytes = pretty_serialized(analysis)?;
    let cases_bytes = pretty_serialized(&case_records)?;
    let inference_bits = json!({
        "first_fixed_order": analysis.first_raw_bits,
        "second_reverse_order": analysis.second_raw_bits,
    });
    let inference_bytes = pretty_json(&inference_bits)?;

    let terminal_status = enum_string(&analysis.status)?;
    let response_class = match analysis.response_class {
        Some(response) => json!(enum_string(&response)?),
        None => Value::Null,
    };
    let confirmation = if args.stage == Stage::Confirmation {
        Some(confirmation_status(
            decision.ok_or("confirmation lacked its decision artifact")?,
            analysis,
        )?)
    } else {
        None
    };

    let mut manifest = base_manifest(args.stage, repository, json!(prepared.plan_sha256.clone()));
    let object = manifest
        .as_object_mut()
        .ok_or("base manifest was not an object")?;
    object.insert("terminal_status".into(), json!(terminal_status));
    object.insert("response_class".into(), response_class);
    object.insert("model_loaded".into(), json!(model_loaded));
    object.insert(
        "model_identity_sha256".into(),
        model_sha256.map_or(Value::Null, |hash| json!(hash)),
    );
    object.insert(
        "model_byte_length".into(),
        model_loaded
            .then_some(FROZEN_MODEL_BYTES)
            .map_or(Value::Null, |length| json!(length)),
    );
    object.insert(
        "expected_model_byte_length".into(),
        json!(FROZEN_MODEL_BYTES),
    );
    object.insert(
        "renderer_version".into(),
        json!(prepared.plan.renderer_version),
    );
    object.insert("global_axis".into(), json!(prepared.plan.global_axis));
    object.insert("case_count".into(), json!(cases.len()));
    object.insert("analysis_file".into(), json!("analysis.json"));
    object.insert("case_records_file".into(), json!("stage_cases.json"));
    object.insert("raw_bits_file".into(), json!("raw_bits.json"));
    insert_atlas_identity(object, atlas);
    if let Some(decision) = decision {
        insert_decision_seal(object, args, decision)?;
    }
    if let Some(status) = confirmation {
        object.insert("confirmation_status".into(), json!(status));
    }

    let seal = write_stage_artifact(
        &args.out,
        repo,
        repository,
        manifest,
        vec![
            ("analysis.json".into(), analysis_bytes),
            ("raw_bits.json".into(), inference_bytes),
            ("renderer_plan.json".into(), prepared.plan_json.clone()),
            ("stage_cases.json".into(), cases_bytes),
        ],
    )?;
    print_stage_seal(args.stage, &seal.manifest_sha256);
    Ok(RunOutcome::Completed(seal.output_dir))
}

fn insert_atlas_identity(object: &mut serde_json::Map<String, Value>, atlas: &ValidatedAtlas) {
    object.insert("atlas_manifest_sha256".into(), json!(atlas.manifest_sha256));
    object.insert(
        "atlas_candidate_stream_sha256".into(),
        json!(atlas.candidate_sha256),
    );
    object.insert(
        "atlas_aperture_summaries_sha256".into(),
        json!(atlas.aperture_sha256),
    );
    object.insert(
        "atlas_pair_summaries_sha256".into(),
        json!(atlas.pair_sha256),
    );
    object.insert(
        "atlas_canonical_checks_sha256".into(),
        json!(atlas.canonical_sha256),
    );
}

fn confirmation_status(
    decision: &VerifiedArtifact,
    confirmation: &StageAnalysis,
) -> Result<&'static str, Box<dyn Error>> {
    let decision_status = decision
        .manifest
        .get("terminal_status")
        .and_then(Value::as_str)
        .ok_or("decision terminal_status is missing")?;
    let confirmation_status = enum_string(&confirmation.status)?;
    if !is_conclusive_status(decision_status) || !is_conclusive_status(&confirmation_status) {
        return Ok("CONFIRMATION_INCONCLUSIVE");
    }
    let decision_response = decision
        .manifest
        .get("response_class")
        .and_then(Value::as_str)
        .ok_or("conclusive decision response_class is missing")?;
    let confirmation_response = confirmation
        .response_class
        .ok_or("conclusive confirmation response_class is missing")?;
    if decision_response == enum_string(&confirmation_response)? {
        Ok("REPLICATED")
    } else {
        Ok("NOT_REPLICATED")
    }
}

fn is_conclusive_status(status: &str) -> bool {
    matches!(
        status,
        "GEOMETRY_SUPPORTED" | "ALTERNATIVE_PHOTOMETRIC_PATH" | "NO_EVIDENCE"
    )
}

fn enum_string(value: &impl Serialize) -> Result<String, Box<dyn Error>> {
    serde_json::to_value(value)?
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| "enum did not serialize as a string".into())
}

fn pretty_serialized(value: &impl Serialize) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn reject_existing_output(path: &Path) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("output path already exists: {}", path.display()).into());
    }
    if path.file_name().is_none() || path.parent().is_none() {
        return Err("--out must be a named new path with a parent directory".into());
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecisionDisposition {
    Conclusive,
    Inconclusive,
}

fn require_compatible_decision(
    artifact: &VerifiedArtifact,
    repository: &RepositoryState,
) -> Result<DecisionDisposition, Box<dyn Error>> {
    let manifest = artifact
        .manifest
        .as_object()
        .ok_or("decision manifest is not an object")?;
    require_manifest_string(
        manifest,
        "artifact_schema",
        "synthetic-eye-phase13-stage-v1",
    )?;
    require_manifest_string(manifest, "stage", "decision")?;
    require_manifest_string(manifest, "preregistration_commit", PREREGISTRATION_COMMIT)?;
    require_manifest_string(manifest, "amendment_commit", AMENDMENT_COMMIT)?;
    require_manifest_string(manifest, "repository_commit", &repository.commit)?;
    require_manifest_string(manifest, "build_repository_commit", BUILD_COMMIT)?;
    require_manifest_string(
        manifest,
        "implementation_source_sha256",
        &repository.implementation_source_sha256,
    )?;
    require_manifest_string(manifest, "build_profile", "release")?;
    if manifest.get("repository_dirty").and_then(Value::as_bool) != Some(false)
        || manifest.get("debug_assertions").and_then(Value::as_bool) != Some(false)
        || manifest
            .get("real_recordings_loaded")
            .and_then(Value::as_bool)
            != Some(false)
        || manifest
            .get("expected_model_byte_length")
            .and_then(Value::as_u64)
            != Some(FROZEN_MODEL_BYTES as u64)
    {
        return Err("decision provenance fields were invalid".into());
    }
    require_optional_manifest_string(
        manifest,
        "atlas_manifest_sha256",
        phase13_output::ATLAS_MANIFEST_SHA256,
    )?;
    require_optional_manifest_string(
        manifest,
        "atlas_candidate_stream_sha256",
        phase13_output::ATLAS_CANDIDATE_SHA256,
    )?;
    require_optional_manifest_string(
        manifest,
        "atlas_aperture_summaries_sha256",
        phase13_output::ATLAS_APERTURE_SHA256,
    )?;
    require_optional_manifest_string(
        manifest,
        "atlas_pair_summaries_sha256",
        phase13_output::ATLAS_PAIR_SHA256,
    )?;
    require_optional_manifest_string(
        manifest,
        "atlas_canonical_checks_sha256",
        phase13_output::ATLAS_CANONICAL_SHA256,
    )?;
    require_optional_manifest_string(
        manifest,
        "renderer_version",
        "synthetic-eye-renderer-100x100-4x-v1",
    )?;
    require_optional_manifest_string(manifest, "global_axis", "whole_image_mean_gray")?;

    let disposition = require_valid_decision_result(manifest)?;
    require_decision_payload_consistency(artifact)?;
    if disposition == DecisionDisposition::Conclusive {
        for (key, expected) in [
            (
                "atlas_manifest_sha256",
                phase13_output::ATLAS_MANIFEST_SHA256,
            ),
            (
                "atlas_candidate_stream_sha256",
                phase13_output::ATLAS_CANDIDATE_SHA256,
            ),
            (
                "atlas_aperture_summaries_sha256",
                phase13_output::ATLAS_APERTURE_SHA256,
            ),
            (
                "atlas_pair_summaries_sha256",
                phase13_output::ATLAS_PAIR_SHA256,
            ),
            (
                "atlas_canonical_checks_sha256",
                phase13_output::ATLAS_CANONICAL_SHA256,
            ),
            ("renderer_version", "synthetic-eye-renderer-100x100-4x-v1"),
            ("global_axis", "whole_image_mean_gray"),
            ("model_identity_sha256", FROZEN_MODEL_SHA256),
        ] {
            require_manifest_string(manifest, key, expected)?;
        }
        if manifest.get("model_loaded").and_then(Value::as_bool) != Some(true)
            || manifest.get("model_byte_length").and_then(Value::as_u64)
                != Some(FROZEN_MODEL_BYTES as u64)
        {
            return Err("conclusive decision did not use the exact frozen model".into());
        }
    }
    Ok(disposition)
}

fn require_valid_decision_result(
    manifest: &serde_json::Map<String, Value>,
) -> Result<DecisionDisposition, Box<dyn Error>> {
    let status = manifest
        .get("terminal_status")
        .and_then(Value::as_str)
        .ok_or("decision terminal_status is missing")?;
    let response = manifest.get("response_class").unwrap_or(&Value::Null);
    match status {
        "RENDERER_NO_GO" => {
            if !response.is_null() {
                return Err("renderer NO-GO must not carry a response_class".into());
            }
            return Ok(DecisionDisposition::Inconclusive);
        }
        "GEOMETRY_SUPPORTED" | "ALTERNATIVE_PHOTOMETRIC_PATH" | "NO_EVIDENCE" | "INCONCLUSIVE" => {
            if response.as_str() != Some(status) {
                return Err("decision response_class does not match terminal_status".into());
            }
            return Ok(if status == "INCONCLUSIVE" {
                DecisionDisposition::Inconclusive
            } else {
                DecisionDisposition::Conclusive
            });
        }
        "INCONCLUSIVE_ARTIFACT" | "INCONCLUSIVE_RECOGNITION" | "INCONCLUSIVE_INSENSITIVE" => {
            if !response.is_null() {
                return Err("decision inconclusive gate must not carry a response_class".into());
            }
            return Ok(DecisionDisposition::Inconclusive);
        }
        _ => return Err(format!("unknown decision terminal_status: {status}").into()),
    }
}

fn require_optional_manifest_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    if object.contains_key(key) {
        require_manifest_string(object, key, expected)?;
    }
    Ok(())
}

fn require_decision_payload_consistency(artifact: &VerifiedArtifact) -> Result<(), Box<dyn Error>> {
    let manifest = artifact
        .manifest
        .as_object()
        .ok_or("decision manifest is not an object")?;
    let terminal = manifest
        .get("terminal_status")
        .and_then(Value::as_str)
        .ok_or("decision terminal_status is missing")?;
    let analysis_file = manifest
        .get("analysis_file")
        .and_then(Value::as_str)
        .ok_or("decision analysis_file is missing")?;
    if analysis_file == "renderer_failure.json" {
        if terminal != "RENDERER_NO_GO"
            || manifest.get("renderer_plan_sha256") != Some(&Value::Null)
            || manifest.get("model_loaded").and_then(Value::as_bool) != Some(false)
            || manifest.get("model_identity_sha256") != Some(&Value::Null)
            || manifest.get("model_byte_length") != Some(&Value::Null)
            || !manifest.get("response_class").is_some_and(Value::is_null)
        {
            return Err("renderer failure manifest fields are inconsistent".into());
        }
        let failure: Value = serde_json::from_slice(
            artifact
                .files
                .get(analysis_file)
                .ok_or("decision renderer_failure.json is missing")?,
        )?;
        if failure.get("classification").and_then(Value::as_str) != Some("RENDERER_NO_GO")
            || failure.get("model_loaded").and_then(Value::as_bool) != Some(false)
            || failure.get("model_identity_sha256") != Some(&Value::Null)
            || failure.get("code").and_then(Value::as_str).is_none()
            || failure.get("detail").and_then(Value::as_str).is_none()
        {
            return Err("renderer failure payload is inconsistent".into());
        }
        return Ok(());
    }
    if analysis_file != "analysis.json" {
        return Err(format!("unknown decision analysis file: {analysis_file}").into());
    }
    if manifest.get("case_records_file").and_then(Value::as_str) != Some("stage_cases.json")
        || manifest.get("raw_bits_file").and_then(Value::as_str) != Some("raw_bits.json")
    {
        return Err("decision payload filenames differ from the manifest contract".into());
    }

    let analysis: StageAnalysis = serde_json::from_slice(
        artifact
            .files
            .get("analysis.json")
            .ok_or("decision analysis.json is missing")?,
    )?;
    if analysis.stage != phase13::Stage::Decision || analysis.status.as_str() != terminal {
        return Err("decision analysis stage/status differs from manifest".into());
    }
    let manifest_response = manifest.get("response_class").unwrap_or(&Value::Null);
    let analysis_response = analysis
        .response_class
        .map(|value| Value::String(value.as_str().to_owned()))
        .unwrap_or(Value::Null);
    if *manifest_response != analysis_response {
        return Err("decision analysis response differs from manifest".into());
    }
    let raw_bits: Value = serde_json::from_slice(
        artifact
            .files
            .get("raw_bits.json")
            .ok_or("decision raw_bits.json is missing")?,
    )?;
    let expected_raw_bits = json!({
        "first_fixed_order": &analysis.first_raw_bits,
        "second_reverse_order": &analysis.second_raw_bits,
    });
    if raw_bits != expected_raw_bits {
        return Err("decision raw_bits.json differs from analysis.json".into());
    }

    let plan_bytes = artifact
        .files
        .get("renderer_plan.json")
        .ok_or("decision renderer_plan.json is missing")?;
    let recorded_plan_hash = manifest
        .get("renderer_plan_sha256")
        .and_then(Value::as_str)
        .ok_or("decision renderer_plan_sha256 is missing")?;
    if phase13_plan_sha256(plan_bytes) != recorded_plan_hash {
        return Err("decision renderer plan hash differs from its bytes".into());
    }
    let plan: RendererPlan = serde_json::from_slice(plan_bytes)?;
    require_frozen_renderer_plan_identity(&plan, manifest)?;
    let cases: Vec<CaseRecord> = serde_json::from_slice(
        artifact
            .files
            .get("stage_cases.json")
            .ok_or("decision stage_cases.json is missing")?,
    )?;
    if cases != plan.decision_case_order
        || manifest.get("case_count").and_then(Value::as_u64) != Some(cases.len() as u64)
        || cases
            .iter()
            .any(|case| case.stage != phase13::Stage::Decision)
    {
        return Err("decision cases differ from manifest or renderer plan".into());
    }
    require_analysis_model_state(manifest, &analysis, &cases)?;
    Ok(())
}

fn require_frozen_renderer_plan_identity(
    plan: &RendererPlan,
    manifest: &serde_json::Map<String, Value>,
) -> Result<(), Box<dyn Error>> {
    let pair_shape_ok = plan.pair_plans.len() == 30
        && plan.pair_plans.iter().enumerate().all(|(offset, pair)| {
            let lower = 7 + offset;
            pair.lower_index == lower
                && pair.higher_index == lower + 4
                && pair.retained == (lower != 36)
                && pair.selected.is_some() == pair.retained
        });
    let case_shape_ok = plan.decision_case_order.len() == 77
        && plan.confirmation_case_order.len() == 73
        && plan
            .decision_case_order
            .iter()
            .all(|case| case.stage == phase13::Stage::Decision)
        && plan
            .confirmation_case_order
            .iter()
            .all(|case| case.stage == phase13::Stage::Confirmation);
    if plan.version != phase13::VERSION
        || plan.preregistration_commit != PREREGISTRATION_COMMIT
        || plan.atlas_repository_commit != phase13::ATLAS_REPOSITORY_COMMIT
        || plan.atlas_preregistration_commit != phase13::ATLAS_PREREGISTRATION_COMMIT
        || plan.atlas_manifest_sha256 != phase13::ATLAS_MANIFEST_SHA256
        || plan.candidate_stream_sha256 != phase13::CANDIDATE_STREAM_SHA256
        || plan.renderer_version != "synthetic-eye-renderer-100x100-4x-v1"
        || plan.global_axis != "whole_image_mean_gray"
        || plan.boundary_margin.to_bits() != ((0.30f64 / 128.0f64) * 2.0f64).to_bits()
        || plan.maximum_condition_number.to_bits() != 20.0f64.to_bits()
        || plan.moment_tolerance_gray.to_bits() != 0.001f64.to_bits()
        || plan.excluded_pairs != vec![[36, 40]]
        || plan.preparation_repetitions != 2
        || plan.renderer_decision != "go"
        || !pair_shape_ok
        || !case_shape_ok
    {
        return Err("decision renderer plan differs from the frozen protocol identity".into());
    }
    for (key, expected) in [
        ("atlas_manifest_sha256", plan.atlas_manifest_sha256.as_str()),
        (
            "atlas_candidate_stream_sha256",
            plan.candidate_stream_sha256.as_str(),
        ),
        (
            "atlas_aperture_summaries_sha256",
            phase13::APERTURE_SUMMARIES_SHA256,
        ),
        (
            "atlas_pair_summaries_sha256",
            phase13::PAIR_SUMMARIES_SHA256,
        ),
        (
            "atlas_canonical_checks_sha256",
            phase13::CANONICAL_CHECKS_SHA256,
        ),
        ("renderer_version", plan.renderer_version.as_str()),
        ("global_axis", plan.global_axis.as_str()),
    ] {
        require_manifest_string(manifest, key, expected)?;
    }
    Ok(())
}

fn require_analysis_model_state(
    manifest: &serde_json::Map<String, Value>,
    analysis: &StageAnalysis,
    cases: &[CaseRecord],
) -> Result<(), Box<dyn Error>> {
    let model_loaded = manifest
        .get("model_loaded")
        .and_then(Value::as_bool)
        .ok_or("decision model_loaded is missing")?;
    if !model_loaded {
        if manifest.get("model_identity_sha256") != Some(&Value::Null)
            || manifest.get("model_byte_length") != Some(&Value::Null)
            || !analysis.first_raw_bits.is_empty()
            || !analysis.second_raw_bits.is_empty()
            || analysis.status != StageStatus::InconclusiveArtifact
            || analysis.artifact_error.as_deref().is_none_or(str::is_empty)
            || analysis.eyes.is_some()
        {
            return Err("pre-model decision artifact state is inconsistent".into());
        }
        return Ok(());
    }

    require_manifest_string(manifest, "model_identity_sha256", FROZEN_MODEL_SHA256)?;
    if manifest.get("model_byte_length").and_then(Value::as_u64) != Some(FROZEN_MODEL_BYTES as u64)
    {
        return Err("loaded-model decision identity is inconsistent".into());
    }
    // Length, ID, nonfinite, and repeat failures are themselves legitimate
    // INCONCLUSIVE_ARTIFACT evidence under the frozen protocol. Preserve those
    // failing arrays rather than reclassifying the scientific failure as
    // decision-artifact tampering.
    if analysis.status == StageStatus::InconclusiveArtifact {
        if analysis.artifact_error.as_deref().is_none_or(str::is_empty) || analysis.eyes.is_some() {
            return Err("artifact-inconclusive analysis lacks its error state".into());
        }
        return Ok(());
    }
    if analysis.first_raw_bits.len() != cases.len() || analysis.second_raw_bits.len() != cases.len()
    {
        return Err("usable decision raw-result counts are inconsistent".into());
    }
    for (index, case) in cases.iter().enumerate() {
        let reverse_index = cases.len() - 1 - index;
        if analysis.first_raw_bits[index].0 != case.id
            || analysis.second_raw_bits[reverse_index].0 != case.id
        {
            return Err(format!("decision raw-result case identity mismatch at {index}").into());
        }
    }

    let has_integrity_failure = cases.iter().enumerate().any(|(index, _)| {
        let reverse_index = cases.len() - 1 - index;
        let first = analysis.first_raw_bits[index].1;
        let second = analysis.second_raw_bits[reverse_index].1;
        first != second
            || first
                .into_iter()
                .chain(second)
                .any(|bits| !f32::from_bits(bits).is_finite())
    });
    if has_integrity_failure {
        return Err("usable decision status contains nonrepeatable or nonfinite inference".into());
    }
    if analysis.artifact_error.is_some() {
        return Err("non-artifact decision unexpectedly contains artifact_error".into());
    }
    match analysis.status {
        StageStatus::InconclusiveRecognition if analysis.eyes.is_some() => {
            return Err("recognition-inconclusive analysis unexpectedly contains metrics".into())
        }
        StageStatus::InconclusiveInsensitive
        | StageStatus::GeometrySupported
        | StageStatus::AlternativePhotometricPath
        | StageStatus::NoEvidence
        | StageStatus::Inconclusive
            if analysis.eyes.is_none() =>
        {
            return Err("post-recognition analysis is missing eye metrics".into())
        }
        StageStatus::RendererNoGo => {
            return Err("renderer NO-GO cannot be stored as analysis.json".into())
        }
        _ => {}
    }
    Ok(())
}

fn phase13_plan_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"sranibro-synthetic-eye-phase13-plan-v1\0");
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn require_decision_plan(
    artifact: &VerifiedArtifact,
    prepared: &PreparedPlan,
) -> Result<(), Box<dyn Error>> {
    let recorded_hash = artifact
        .manifest
        .get("renderer_plan_sha256")
        .and_then(Value::as_str)
        .ok_or("decision renderer_plan_sha256 is missing")?;
    if recorded_hash != prepared.plan_sha256 {
        return Err("decision renderer plan hash differs from confirmation preparation".into());
    }
    let recorded_plan = artifact
        .files
        .get("renderer_plan.json")
        .ok_or("decision renderer_plan.json is missing")?;
    if recorded_plan.as_slice() != prepared.plan_json.as_slice() {
        return Err("decision renderer plan bytes differ from confirmation preparation".into());
    }
    if artifact.manifest.get("case_count").and_then(Value::as_u64)
        != Some(prepared.decision_cases.len() as u64)
    {
        return Err("decision case_count differs from the sealed renderer plan".into());
    }
    let expected_records: Vec<_> = prepared
        .decision_cases
        .iter()
        .map(|case| &case.record)
        .collect();
    let recorded_cases = artifact
        .files
        .get("stage_cases.json")
        .ok_or("decision stage_cases.json is missing")?;
    if recorded_cases.as_slice() != pretty_serialized(&expected_records)?.as_slice() {
        return Err("decision stage case records differ from the sealed renderer plan".into());
    }
    Ok(())
}

fn require_manifest_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    if object.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(format!("decision manifest {key} mismatch").into());
    }
    Ok(())
}

fn base_manifest(stage: Stage, repository: &RepositoryState, renderer_plan_sha256: Value) -> Value {
    json!({
        "artifact_schema": "synthetic-eye-phase13-stage-v1",
        "stage": stage.as_str(),
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "amendment_commit": AMENDMENT_COMMIT,
        "repository_commit": repository.commit,
        "build_repository_commit": BUILD_COMMIT,
        "implementation_source_sha256": repository.implementation_source_sha256,
        "repository_dirty": repository.dirty,
        "build_profile": BUILD_PROFILE,
        "debug_assertions": cfg!(debug_assertions),
        "renderer_plan_sha256": renderer_plan_sha256,
        "real_recordings_loaded": false,
        "sanitized_cli": match stage {
            Stage::Decision => json!([
                "decision", "--atlas", "<sealed-atlas-directory>", "--model",
                "<EyeNet-params-file>", "--out", "<new-directory>"
            ]),
            Stage::Confirmation => json!([
                "confirmation", "--atlas", "<sealed-atlas-directory>", "--model",
                "<same-EyeNet-params-file>", "--decision",
                "<sealed-decision-directory>", "--decision-seal",
                "<decision-manifest-sha256>", "--out", "<new-directory>"
            ]),
        },
    })
}

fn seal_renderer_failure(
    args: &Args,
    repo: &Path,
    repository: &RepositoryState,
    decision: Option<&VerifiedArtifact>,
    atlas: Option<&ValidatedAtlas>,
    code: &str,
    detail: &str,
) -> Result<RunOutcome, Box<dyn Error>> {
    let failure = json!({
        "classification": "RENDERER_NO_GO",
        "code": code,
        "detail": detail,
        "model_loaded": false,
        "model_identity_sha256": Value::Null,
    });
    let bytes = pretty_json(&failure)?;
    let mut manifest = base_manifest(args.stage, repository, Value::Null);
    let object = manifest
        .as_object_mut()
        .ok_or("base manifest was not an object")?;
    object.insert("terminal_status".into(), json!("RENDERER_NO_GO"));
    object.insert("model_loaded".into(), json!(false));
    object.insert("model_identity_sha256".into(), Value::Null);
    object.insert("model_byte_length".into(), Value::Null);
    object.insert(
        "expected_model_byte_length".into(),
        json!(FROZEN_MODEL_BYTES),
    );
    object.insert("response_class".into(), Value::Null);
    object.insert("analysis_file".into(), json!("renderer_failure.json"));
    if let Some(atlas) = atlas {
        insert_atlas_identity(object, atlas);
    }
    if args.stage == Stage::Confirmation {
        let decision = decision.ok_or("confirmation renderer failure lacked decision seal")?;
        insert_decision_seal(object, args, decision)?;
        object.insert(
            "confirmation_status".into(),
            json!("CONFIRMATION_INCONCLUSIVE"),
        );
    }
    let seal = write_stage_artifact(
        &args.out,
        repo,
        repository,
        manifest,
        vec![("renderer_failure.json".into(), bytes)],
    )?;
    print_stage_seal(args.stage, &seal.manifest_sha256);
    Ok(RunOutcome::Completed(seal.output_dir))
}

fn seal_inconclusive_decision_confirmation(
    args: &Args,
    repo: &Path,
    repository: &RepositoryState,
    decision: &VerifiedArtifact,
) -> Result<RunOutcome, Box<dyn Error>> {
    if args.stage != Stage::Confirmation {
        return Err("metadata-only confirmation requires confirmation stage".into());
    }
    let decision_status = decision
        .manifest
        .get("terminal_status")
        .and_then(Value::as_str)
        .ok_or("decision terminal_status is missing")?;
    let payload = json!({
        "classification": "CONFIRMATION_INCONCLUSIVE",
        "reason": "sealed decision was not conclusive and cannot be rescued",
        "decision_terminal_status": decision_status,
        "model_loaded": false,
        "renderer_prepared": false,
    });
    let plan_hash = decision
        .manifest
        .get("renderer_plan_sha256")
        .cloned()
        .unwrap_or(Value::Null);
    let mut manifest = base_manifest(Stage::Confirmation, repository, plan_hash);
    let object = manifest
        .as_object_mut()
        .ok_or("base manifest was not an object")?;
    object.insert("terminal_status".into(), json!("INCONCLUSIVE"));
    object.insert("response_class".into(), Value::Null);
    object.insert(
        "confirmation_status".into(),
        json!("CONFIRMATION_INCONCLUSIVE"),
    );
    object.insert("decision_terminal_status".into(), json!(decision_status));
    object.insert("model_loaded".into(), json!(false));
    object.insert("model_identity_sha256".into(), Value::Null);
    object.insert("model_byte_length".into(), Value::Null);
    object.insert(
        "expected_model_byte_length".into(),
        json!(FROZEN_MODEL_BYTES),
    );
    object.insert("analysis_file".into(), json!("confirmation.json"));
    insert_decision_seal(object, args, decision)?;
    let seal = write_stage_artifact(
        &args.out,
        repo,
        repository,
        manifest,
        vec![("confirmation.json".into(), pretty_json(&payload)?)],
    )?;
    print_stage_seal(Stage::Confirmation, &seal.manifest_sha256);
    Ok(RunOutcome::Completed(seal.output_dir))
}

fn insert_decision_seal(
    object: &mut serde_json::Map<String, Value>,
    args: &Args,
    decision: &VerifiedArtifact,
) -> Result<(), Box<dyn Error>> {
    let external = args
        .decision_seal
        .as_deref()
        .ok_or("confirmation lacked --decision-seal")?;
    if external != decision.manifest_sha256 {
        return Err("external decision seal differs after verification".into());
    }
    object.insert("decision_external_seal_sha256".into(), json!(external));
    object.insert(
        "decision_manifest_sha256".into(),
        json!(decision.manifest_sha256),
    );
    Ok(())
}

fn print_stage_seal(stage: Stage, seal: &str) {
    match stage {
        Stage::Decision => println!("Decision manifest SHA-256: {seal}"),
        Stage::Confirmation => println!("Confirmation manifest SHA-256: {seal}"),
    }
}

fn pretty_json(value: &Value) -> Result<Vec<u8>, Box<dyn Error>> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

struct FrozenModels {
    first: EyeNet,
    second: EyeNet,
    bytes: Vec<u8>,
    sha256: String,
}

/// This is intentionally called only after the renderer preparation gate.
fn load_frozen_models(path: &Path) -> Result<FrozenModels, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() != FROZEN_MODEL_BYTES {
        return Err(format!(
            "model byte length was {}, expected exactly {FROZEN_MODEL_BYTES}",
            bytes.len()
        )
        .into());
    }
    let sha256 = sha256_hex(&bytes);
    if sha256 != FROZEN_MODEL_SHA256 {
        return Err(format!("model SHA-256 was {sha256}, expected {FROZEN_MODEL_SHA256}").into());
    }
    let first = EyeNet::new(tvm_params::parse_map_bytes(&bytes)?)
        .map_err(|error| format!("frozen EyeNet model invalid: {error}"))?;
    let second = EyeNet::new(tvm_params::parse_map_bytes(&bytes)?)
        .map_err(|error| format!("second frozen EyeNet model invalid: {error}"))?;
    Ok(FrozenModels {
        first,
        second,
        bytes,
        sha256,
    })
}

fn verify_model_unchanged(path: &Path, expected: &[u8]) -> Result<(), Box<dyn Error>> {
    let current = std::fs::read(path)?;
    if current != expected {
        return Err("model bytes changed during inference".into());
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    phase13_output::sha256_hex(bytes)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stage {
    Decision,
    Confirmation,
}

impl Stage {
    fn as_str(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Confirmation => "confirmation",
        }
    }

    fn core(self) -> phase13::Stage {
        match self {
            Self::Decision => phase13::Stage::Decision,
            Self::Confirmation => phase13::Stage::Confirmation,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    stage: Stage,
    atlas: PathBuf,
    model: PathBuf,
    decision: Option<PathBuf>,
    decision_seal: Option<String>,
    out: PathBuf,
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, Box<dyn Error>> {
        let mut args = args.into_iter();
        let Some(command) = args.next() else {
            return Err("missing required command: decision or confirmation".into());
        };
        if matches!(command.as_str(), "-h" | "--help") {
            if args.next().is_some() {
                return Err("--help cannot be combined with other arguments".into());
            }
            return Ok(None);
        }
        let stage = match command.as_str() {
            "decision" => Stage::Decision,
            "confirmation" => Stage::Confirmation,
            unknown => return Err(format!("unknown command: {unknown}").into()),
        };

        let mut atlas = None;
        let mut model = None;
        let mut decision = None;
        let mut decision_seal = None;
        let mut out = None;
        while let Some(option) = args.next() {
            match option.as_str() {
                "--atlas" => set_once(&mut atlas, next_path(&mut args, "--atlas")?, "--atlas")?,
                "--model" => set_once(&mut model, next_path(&mut args, "--model")?, "--model")?,
                "--decision" => set_once(
                    &mut decision,
                    next_path(&mut args, "--decision")?,
                    "--decision",
                )?,
                "--decision-seal" => {
                    let seal = next_string(&mut args, "--decision-seal")?;
                    validate_decision_seal(&seal)?;
                    set_once(&mut decision_seal, seal, "--decision-seal")?;
                }
                "--out" => set_once(&mut out, next_path(&mut args, "--out")?, "--out")?,
                "-h" | "--help" => {
                    return Err("--help must be used alone, without a command".into())
                }
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }

        match stage {
            Stage::Decision if decision.is_some() || decision_seal.is_some() => {
                return Err(
                    "--decision and --decision-seal are prohibited for the decision stage".into(),
                )
            }
            Stage::Confirmation if decision.is_none() || decision_seal.is_none() => return Err(
                "confirmation requires --decision <directory> and --decision-seal <64-hex-sha256>"
                    .into(),
            ),
            _ => {}
        }
        Ok(Some(Self {
            stage,
            atlas: atlas.ok_or("missing required --atlas <sealed-atlas-directory>")?,
            model: model.ok_or("missing required --model <EyeNet-params-file>")?,
            decision,
            decision_seal,
            out: out.ok_or("missing required --out <new-directory>")?,
        }))
    }
}

fn next_string(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<String, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value after {option}"))?;
    if value.starts_with('-') {
        return Err(format!("missing value after {option}; {value} is another option").into());
    }
    Ok(value)
}

fn validate_decision_seal(value: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("--decision-seal must be exactly 64 lowercase hexadecimal characters".into());
    }
    Ok(())
}

fn next_path(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value after {option}"))?;
    if value.starts_with('-') {
        return Err(format!("missing value after {option}; {value} is another option").into());
    }
    Ok(PathBuf::from(value))
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), Box<dyn Error>> {
    if slot.replace(value).is_some() {
        return Err(format!("{option} may be specified only once").into());
    }
    Ok(())
}

fn print_help() {
    println!(
        "{}",
        concat!(
            "SRanibro synthetic EyeNet Phase 1.3 study (research only)\n\n",
            "Decision:\n",
            "  synthetic-eye-phase13 decision --atlas <sealed-atlas-directory> ",
            "--model <EyeNet-params-file> --out <new-directory>\n\n",
            "Confirmation:\n",
            "  synthetic-eye-phase13 confirmation --atlas <sealed-atlas-directory> ",
            "--model <same-EyeNet-params-file> --decision <sealed-decision-directory> ",
            "--decision-seal <64-hex-sha256> --out <new-directory>\n\n",
            "Recording and production-state inputs are not accepted."
        )
    );
}

#[derive(Debug, PartialEq, Eq)]
enum RunOutcome {
    Completed(PathBuf),
    Help,
}

#[cfg(test)]
mod tests {
    use super::*;
    use phase13::{CaseKind, ResponseClass};

    const TEST_SEAL: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    fn analysis(status: StageStatus, response_class: Option<ResponseClass>) -> StageAnalysis {
        StageAnalysis {
            stage: phase13::Stage::Confirmation,
            status,
            response_class,
            flags: Vec::new(),
            artifact_error: None,
            first_raw_bits: Vec::new(),
            second_raw_bits: Vec::new(),
            eyes: None,
        }
    }

    fn decision_artifact(status: &str, response_class: Value) -> VerifiedArtifact {
        VerifiedArtifact {
            directory: PathBuf::from("decision"),
            manifest: json!({
                "terminal_status": status,
                "response_class": response_class,
            }),
            manifest_bytes: Vec::new(),
            manifest_sha256: "0".repeat(64),
            inventory: Vec::new(),
            files: Default::default(),
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn decision_case(id: &str) -> CaseRecord {
        CaseRecord {
            id: id.to_owned(),
            stage: phase13::Stage::Decision,
            kind: CaseKind::DefaultS3,
            aperture_index: 7,
            pair_lower_index: None,
            target_number: None,
            aperture_role: None,
            skin_bits: 0,
            sclera_bits: 0,
            left_pixels_sha256: String::new(),
            right_pixels_sha256: String::new(),
            tensor_sha256: String::new(),
            left_covariates: renderer::ImageCovariates::default(),
        }
    }

    fn loaded_recognition_analysis(ids: &[&str]) -> StageAnalysis {
        let raw = [0.90f32, 0.25, 0.25, 0.0, 0.0].map(f32::to_bits);
        StageAnalysis {
            stage: phase13::Stage::Decision,
            status: StageStatus::InconclusiveRecognition,
            response_class: None,
            flags: Vec::new(),
            artifact_error: None,
            first_raw_bits: ids.iter().map(|id| ((*id).to_owned(), raw)).collect(),
            second_raw_bits: ids.iter().rev().map(|id| ((*id).to_owned(), raw)).collect(),
            eyes: None,
        }
    }

    #[test]
    fn decision_cli_accepts_exact_required_surface() {
        let args = Args::parse(strings(&[
            "decision",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--out",
            "result",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(args.stage, Stage::Decision);
        assert_eq!(args.atlas, PathBuf::from("atlas"));
        assert_eq!(args.model, PathBuf::from("model.params"));
        assert_eq!(args.decision, None);
        assert_eq!(args.decision_seal, None);
        assert_eq!(args.out, PathBuf::from("result"));
    }

    #[test]
    fn confirmation_cli_requires_a_sealed_decision() {
        let args = Args::parse(strings(&[
            "confirmation",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--decision",
            "decision-result",
            "--decision-seal",
            TEST_SEAL,
            "--out",
            "confirmation-result",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(args.stage, Stage::Confirmation);
        assert_eq!(args.decision, Some(PathBuf::from("decision-result")));
        assert_eq!(args.decision_seal.as_deref(), Some(TEST_SEAL));

        let error = Args::parse(strings(&[
            "confirmation",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--out",
            "result",
        ]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("requires --decision"));

        let error = Args::parse(strings(&[
            "confirmation",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--decision",
            "decision-result",
            "--out",
            "result",
        ]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("--decision-seal"));
    }

    #[test]
    fn decision_rejects_confirmation_and_recording_arguments() {
        let error = Args::parse(strings(&[
            "decision",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--decision",
            "old-result",
            "--out",
            "result",
        ]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("prohibited"));

        let error = Args::parse(strings(&[
            "decision",
            "--atlas",
            "atlas",
            "--model",
            "model.params",
            "--recording",
            "session",
            "--out",
            "result",
        ]))
        .unwrap_err()
        .to_string();
        assert!(error.contains("unknown argument"));

        let upper = TEST_SEAL.to_ascii_uppercase().replace('0', "A");
        let error = Args::parse(vec![
            "confirmation".into(),
            "--atlas".into(),
            "atlas".into(),
            "--model".into(),
            "model.params".into(),
            "--decision".into(),
            "decision-result".into(),
            "--decision-seal".into(),
            upper,
            "--out".into(),
            "result".into(),
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("lowercase hexadecimal"));
    }

    #[test]
    fn cli_rejects_missing_duplicate_and_unknown_arguments() {
        assert!(Args::parse(strings(&[]))
            .unwrap_err()
            .to_string()
            .contains("command"));
        assert!(Args::parse(strings(&["other"]))
            .unwrap_err()
            .to_string()
            .contains("unknown"));
        let duplicate = Args::parse(strings(&[
            "decision", "--atlas", "a", "--atlas", "b", "--model", "m", "--out", "o",
        ]))
        .unwrap_err()
        .to_string();
        assert!(duplicate.contains("only once"));
        assert!(Args::parse(strings(&["decision", "--atlas", "--model"]))
            .unwrap_err()
            .to_string()
            .contains("another option"));
    }

    #[test]
    fn help_is_standalone() {
        assert_eq!(Args::parse(strings(&["--help"])).unwrap(), None);
        assert!(Args::parse(strings(&["--help", "decision"])).is_err());
        assert!(Args::parse(strings(&["decision", "--help"])).is_err());
    }

    #[test]
    fn bit_identity_distinguishes_signed_zero_and_nan_payloads() {
        assert!(same_f32_bits(
            &[f32::from_bits(0x7fc0_0001)],
            &[f32::from_bits(0x7fc0_0001)]
        ));
        assert!(!same_f32_bits(&[0.0], &[-0.0]));
        assert!(!same_f32_bits(
            &[f32::from_bits(0x7fc0_0001)],
            &[f32::from_bits(0x7fc0_0002)]
        ));
    }

    #[test]
    fn decision_result_requires_a_known_consistent_status_and_response() {
        for (status, response) in [
            ("GEOMETRY_SUPPORTED", json!("GEOMETRY_SUPPORTED")),
            (
                "ALTERNATIVE_PHOTOMETRIC_PATH",
                json!("ALTERNATIVE_PHOTOMETRIC_PATH"),
            ),
            ("NO_EVIDENCE", json!("NO_EVIDENCE")),
            ("INCONCLUSIVE", json!("INCONCLUSIVE")),
            ("INCONCLUSIVE_ARTIFACT", Value::Null),
            ("INCONCLUSIVE_RECOGNITION", Value::Null),
            ("INCONCLUSIVE_INSENSITIVE", Value::Null),
        ] {
            let manifest = json!({
                "terminal_status": status,
                "response_class": response,
            });
            assert!(require_valid_decision_result(manifest.as_object().unwrap()).is_ok());
        }

        for manifest in [
            json!({
                "terminal_status": "GEOMETRY_SUPPORTED",
                "response_class": "NO_EVIDENCE",
            }),
            json!({
                "terminal_status": "INCONCLUSIVE_ARTIFACT",
                "response_class": "INCONCLUSIVE",
            }),
            json!({
                "terminal_status": "UNKNOWN",
                "response_class": null,
            }),
        ] {
            assert!(require_valid_decision_result(manifest.as_object().unwrap()).is_err());
        }

        let renderer_no_go = json!({
            "terminal_status": "RENDERER_NO_GO",
            "response_class": null,
        });
        assert_eq!(
            require_valid_decision_result(renderer_no_go.as_object().unwrap()).unwrap(),
            DecisionDisposition::Inconclusive
        );
    }

    #[test]
    fn decision_model_state_is_bound_to_frozen_model_and_case_order() {
        let manifest = json!({
            "model_loaded": true,
            "model_identity_sha256": FROZEN_MODEL_SHA256,
            "model_byte_length": FROZEN_MODEL_BYTES,
        });
        let cases = vec![decision_case("a"), decision_case("b")];
        let analysis = loaded_recognition_analysis(&["a", "b"]);
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,).is_ok()
        );

        let mut wrong_forward_id = analysis.clone();
        wrong_forward_id.first_raw_bits[1].0 = "a".into();
        assert!(require_analysis_model_state(
            manifest.as_object().unwrap(),
            &wrong_forward_id,
            &cases,
        )
        .is_err());

        let mut wrong_reverse_order = analysis.clone();
        wrong_reverse_order.second_raw_bits.reverse();
        assert!(require_analysis_model_state(
            manifest.as_object().unwrap(),
            &wrong_reverse_order,
            &cases,
        )
        .is_err());

        let wrong_model = json!({
            "model_loaded": true,
            "model_identity_sha256": "0".repeat(64),
            "model_byte_length": FROZEN_MODEL_BYTES,
        });
        assert!(
            require_analysis_model_state(wrong_model.as_object().unwrap(), &analysis, &cases,)
                .is_err()
        );
    }

    #[test]
    fn pre_model_artifact_requires_null_model_fields_and_empty_results() {
        let manifest = json!({
            "model_loaded": false,
            "model_identity_sha256": null,
            "model_byte_length": null,
        });
        let cases = vec![decision_case("a")];
        let mut analysis = StageAnalysis {
            stage: phase13::Stage::Decision,
            status: StageStatus::InconclusiveArtifact,
            response_class: None,
            flags: Vec::new(),
            artifact_error: Some("model could not be loaded".into()),
            first_raw_bits: Vec::new(),
            second_raw_bits: Vec::new(),
            eyes: None,
        };
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,).is_ok()
        );

        analysis.status = StageStatus::InconclusiveRecognition;
        analysis.artifact_error = None;
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,)
                .is_err()
        );
    }

    #[test]
    fn usable_status_cannot_hide_nonrepeatable_or_nonfinite_bits() {
        let manifest = json!({
            "model_loaded": true,
            "model_identity_sha256": FROZEN_MODEL_SHA256,
            "model_byte_length": FROZEN_MODEL_BYTES,
        });
        let cases = vec![decision_case("a")];
        let mut analysis = loaded_recognition_analysis(&["a"]);
        analysis.second_raw_bits[0].1[1] ^= 1;
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,)
                .is_err()
        );

        analysis.status = StageStatus::InconclusiveArtifact;
        analysis.artifact_error = Some("repeat mismatch".into());
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,).is_ok()
        );

        analysis.artifact_error = Some("case/result length and identity mismatch".into());
        analysis.first_raw_bits.clear();
        analysis.second_raw_bits[0].0 = "wrong-id".into();
        assert!(
            require_analysis_model_state(manifest.as_object().unwrap(), &analysis, &cases,).is_ok()
        );
    }

    #[test]
    #[ignore = "full sealed-atlas plan fixture is an explicit release verification"]
    fn sealed_plan_identity_rejects_tampered_metadata() {
        let prepared =
            phase13::validate_atlas_and_prepare(Path::new("research-output/moment-atlas-49e13f0"))
                .unwrap();
        let manifest = json!({
            "atlas_manifest_sha256": phase13::ATLAS_MANIFEST_SHA256,
            "atlas_candidate_stream_sha256": phase13::CANDIDATE_STREAM_SHA256,
            "atlas_aperture_summaries_sha256": phase13::APERTURE_SUMMARIES_SHA256,
            "atlas_pair_summaries_sha256": phase13::PAIR_SUMMARIES_SHA256,
            "atlas_canonical_checks_sha256": phase13::CANONICAL_CHECKS_SHA256,
            "renderer_version": "synthetic-eye-renderer-100x100-4x-v1",
            "global_axis": "whole_image_mean_gray",
        });
        assert!(require_frozen_renderer_plan_identity(
            &prepared.plan,
            manifest.as_object().unwrap(),
        )
        .is_ok());

        let mut tampered = prepared.plan;
        tampered.global_axis = "per_pair_axis".into();
        assert!(
            require_frozen_renderer_plan_identity(&tampered, manifest.as_object().unwrap(),)
                .is_err()
        );
    }

    #[test]
    fn confirmation_status_is_the_frozen_total_procedure() {
        let matching = decision_artifact("GEOMETRY_SUPPORTED", json!("GEOMETRY_SUPPORTED"));
        assert_eq!(
            confirmation_status(
                &matching,
                &analysis(
                    StageStatus::GeometrySupported,
                    Some(ResponseClass::GeometrySupported),
                ),
            )
            .unwrap(),
            "REPLICATED"
        );

        assert_eq!(
            confirmation_status(
                &matching,
                &analysis(StageStatus::NoEvidence, Some(ResponseClass::NoEvidence)),
            )
            .unwrap(),
            "NOT_REPLICATED"
        );

        let inconclusive = decision_artifact("INCONCLUSIVE_RECOGNITION", Value::Null);
        assert_eq!(
            confirmation_status(
                &inconclusive,
                &analysis(
                    StageStatus::GeometrySupported,
                    Some(ResponseClass::GeometrySupported),
                ),
            )
            .unwrap(),
            "CONFIRMATION_INCONCLUSIVE"
        );
        assert_eq!(
            confirmation_status(
                &matching,
                &analysis(StageStatus::InconclusiveInsensitive, None),
            )
            .unwrap(),
            "CONFIRMATION_INCONCLUSIVE"
        );
    }
}
