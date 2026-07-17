//! Preregistered Phase 1.4 XR5 real-recording audit entry point.

#![recursion_limit = "256"]

mod xr5_transfer;
mod xr5_transfer_output;

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use xr5_transfer::{AnalysisTerminalStatus, BuiltInputPlan};
use xr5_transfer_output::{FileIdentity, RepositoryState, VerifiedInputSeal};

fn main() {
    match run() {
        Ok(RunOutcome::Completed { stage, path, seal }) => {
            println!("Phase 1.4 {stage} artifact: {}", path.display());
            println!("External manifest SHA-256: {seal}");
        }
        Ok(RunOutcome::Help) => print_help(),
        Err(error) => {
            eprintln!("synthetic-eye-xr5-transfer: {error}");
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
    match args {
        Args::SealInput(paths) => execute_seal_input(paths),
        Args::Analyze(paths) => execute_analyze(paths),
    }
}

fn execute_seal_input(paths: CommonPaths) -> Result<RunOutcome, Box<dyn Error>> {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // This preflight intentionally has no recording-root parameter.  Source,
    // build, repository, Phase 1.3, and output trust are established before the
    // private XR5 tree is even enumerated.
    let repository = xr5_transfer_output::preflight(
        &repository_root,
        &paths.phase13_decision,
        &paths.phase13_confirmation,
        &paths.out,
    )?;
    require_output_outside_protected_roots(
        &paths.out,
        &[
            ("recording root", paths.wide_data.as_path()),
            (
                "Phase 1.3 decision artifact",
                paths.phase13_decision.as_path(),
            ),
            (
                "Phase 1.3 confirmation artifact",
                paths.phase13_confirmation.as_path(),
            ),
        ],
    )?;

    let first = xr5_transfer::build_input_plan(&paths.wide_data)?;
    let second = xr5_transfer::build_input_plan(&paths.wide_data)?;
    require_matching_plan("recording input during sealing", &first, &second)?;

    // Close every external-state verification window immediately before the
    // destination is created.  No frame pixels have been decoded here.
    xr5_transfer_output::verify_phase13_artifacts(
        &paths.phase13_decision,
        &paths.phase13_confirmation,
    )?;
    xr5_transfer_output::recheck(&repository_root, &repository)?;

    let manifest = input_manifest(&repository, &first);
    let seal = xr5_transfer_output::write_sealed_artifact_with_guard(
        &paths.out,
        first.input_payloads(),
        manifest,
        xr5_transfer_output::INPUT_SEAL_ALLOWLIST,
        || {
            let final_plan = xr5_transfer::build_input_plan(&paths.wide_data)?;
            require_matching_plan(
                "recording input before seal publication",
                &first,
                &final_plan,
            )?;
            xr5_transfer_output::verify_phase13_artifacts(
                &paths.phase13_decision,
                &paths.phase13_confirmation,
            )?;
            xr5_transfer_output::recheck(&repository_root, &repository)
        },
    )?;
    Ok(RunOutcome::Completed {
        stage: "input seal",
        path: seal.output_dir,
        seal: seal.manifest_sha256,
    })
}

fn execute_analyze(paths: AnalyzePaths) -> Result<RunOutcome, Box<dyn Error>> {
    let repository_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // As with seal-input, this finishes before the recording root is opened.
    let repository = xr5_transfer_output::preflight(
        &repository_root,
        &paths.common.phase13_decision,
        &paths.common.phase13_confirmation,
        &paths.common.out,
    )?;
    require_output_outside_protected_roots(
        &paths.common.out,
        &[
            ("recording root", paths.common.wide_data.as_path()),
            (
                "Phase 1.3 decision artifact",
                paths.common.phase13_decision.as_path(),
            ),
            (
                "Phase 1.3 confirmation artifact",
                paths.common.phase13_confirmation.as_path(),
            ),
            ("sealed input artifact", paths.input.as_path()),
        ],
    )?;
    let input = xr5_transfer_output::verify_input_seal(&paths.input, &paths.input_seal)?;
    require_input_implementation_identity(&input, &repository)?;

    let before = xr5_transfer::build_input_plan(&paths.common.wide_data)?;
    require_plan_matches_seal(&before, &input)?;
    let repeated = xr5_transfer::build_input_plan(&paths.common.wide_data)?;
    require_matching_plan("recording input before analysis", &before, &repeated)?;

    let mut model_observation = ModelObservation::NotInvoked;
    let load_model = || read_stable_model(&paths.model, &mut model_observation);
    let outcome = xr5_transfer::analyze_with_model_loader(&before, load_model);
    if outcome.publication_forbidden {
        return Err(
            "recording or model input changed during analysis; no artifact was published".into(),
        );
    }
    if !outcome.payload_allowlist_is_exact() {
        return Err("analysis produced a non-preregistered payload allowlist".into());
    }

    // Reconstruct and bit-compare the private tree after all inference.  This
    // happens before creating the output destination, so a race is a hard
    // failure and cannot leave a scientific-looking artifact.
    let after = xr5_transfer::build_input_plan(&paths.common.wide_data)?;
    require_matching_plan("recording input during analysis", &before, &after)?;
    require_plan_matches_seal(&after, &input)?;
    recheck_model_observation(&paths.model, &model_observation)?;
    xr5_transfer_output::verify_phase13_artifacts(
        &paths.common.phase13_decision,
        &paths.common.phase13_confirmation,
    )?;
    xr5_transfer_output::recheck(&repository_root, &repository)?;

    let manifest = analysis_manifest(&repository, &input, &before, &after, &outcome);
    let allowlist = if outcome.terminal_status == AnalysisTerminalStatus::AuditComplete {
        xr5_transfer_output::ANALYSIS_COMPLETE_ALLOWLIST
    } else {
        xr5_transfer_output::ANALYSIS_REDUCED_ALLOWLIST
    };
    let seal = xr5_transfer_output::write_sealed_artifact_with_guard(
        &paths.common.out,
        outcome.payloads,
        manifest,
        allowlist,
        || {
            let final_input =
                xr5_transfer_output::verify_input_seal(&paths.input, &paths.input_seal)?;
            require_same_input_seal(&input, &final_input)?;
            require_input_implementation_identity(&final_input, &repository)?;
            let final_plan = xr5_transfer::build_input_plan(&paths.common.wide_data)?;
            require_matching_plan(
                "recording input before analysis publication",
                &before,
                &final_plan,
            )?;
            require_plan_matches_seal(&final_plan, &final_input)?;
            recheck_model_observation(&paths.model, &model_observation)?;
            xr5_transfer_output::verify_phase13_artifacts(
                &paths.common.phase13_decision,
                &paths.common.phase13_confirmation,
            )?;
            xr5_transfer_output::recheck(&repository_root, &repository)
        },
    )?;
    Ok(RunOutcome::Completed {
        stage: "analysis",
        path: seal.output_dir,
        seal: seal.manifest_sha256,
    })
}

fn require_matching_plan(
    label: &str,
    expected: &BuiltInputPlan,
    observed: &BuiltInputPlan,
) -> Result<(), Box<dyn Error>> {
    if !expected.bit_exact_artifacts_match(observed) {
        return Err(format!("{label} changed").into());
    }
    Ok(())
}

/// The artifact writer creates a deterministic staging sibling before rename,
/// so validating the canonical output parent also protects that staging path.
/// This runs only after the source/build/seal preflight, but before enumeration
/// or any write involving the private XR5 root.
fn require_output_outside_protected_roots(
    output: &Path,
    protected_roots: &[(&str, &Path)],
) -> Result<(), Box<dyn Error>> {
    let parent = output.parent().ok_or("output path has no parent")?;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    let canonical_parent = fs::canonicalize(parent)?;
    for (label, root) in protected_roots {
        let canonical_root = fs::canonicalize(root)?;
        if path_is_within(&canonical_parent, &canonical_root) {
            return Err(
                format!("output and staging parent must be outside the protected {label}").into(),
            );
        }
    }
    Ok(())
}

#[cfg(windows)]
fn path_is_within(candidate: &Path, root: &Path) -> bool {
    let normalize = |path: &Path| {
        path.as_os_str()
            .to_string_lossy()
            .replace('/', "\\")
            .to_lowercase()
            .trim_end_matches('\\')
            .to_owned()
    };
    let candidate = normalize(candidate);
    let root = normalize(root);
    candidate == root
        || candidate
            .strip_prefix(&root)
            .is_some_and(|suffix| suffix.starts_with('\\'))
}

#[cfg(not(windows))]
fn path_is_within(candidate: &Path, root: &Path) -> bool {
    candidate.starts_with(root)
}

fn require_plan_matches_seal(
    plan: &BuiltInputPlan,
    input: &VerifiedInputSeal,
) -> Result<(), Box<dyn Error>> {
    let inventory = input
        .files
        .get("recording_inventory.json")
        .ok_or("sealed input is missing recording_inventory.json")?;
    let session_plan = input
        .files
        .get("session_plan.json")
        .ok_or("sealed input is missing session_plan.json")?;
    if inventory != &plan.recording_inventory_json || session_plan != &plan.session_plan_json {
        return Err("current recording tree differs from the externally sealed input".into());
    }
    require_manifest_string(
        &input.manifest,
        "recording_tree_sha256",
        &plan.recording_tree_sha256,
    )?;
    require_manifest_u64(
        &input.manifest,
        "inventory_entry_count",
        plan.inventory.entries.len() as u64,
    )?;
    require_manifest_u64(
        &input.manifest,
        "completed_session_count",
        plan.session_plan.completed_sessions.len() as u64,
    )?;
    require_manifest_u64(
        &input.manifest,
        "partial_session_count",
        plan.session_plan.partial_session_ids.len() as u64,
    )?;
    let pair_count: usize = plan
        .session_plan
        .completed_sessions
        .iter()
        .map(|session| session.pair_count)
        .sum();
    require_manifest_u64(&input.manifest, "stereo_pair_count", pair_count as u64)?;
    require_manifest_u64(&input.manifest, "csv_row_count", (pair_count * 2) as u64)?;
    if input.manifest.get("invalid_reason_codes")
        != Some(&serde_json::to_value(&plan.session_plan.errors)?)
    {
        return Err("sealed input manifest has inconsistent invalid_reason_codes".into());
    }
    Ok(())
}

fn require_same_input_seal(
    expected: &VerifiedInputSeal,
    observed: &VerifiedInputSeal,
) -> Result<(), Box<dyn Error>> {
    if expected.manifest_sha256 != observed.manifest_sha256
        || expected.manifest_bytes != observed.manifest_bytes
        || expected.inventory != observed.inventory
        || expected.files != observed.files
    {
        return Err("externally sealed input changed during analysis".into());
    }
    Ok(())
}

fn require_input_implementation_identity(
    input: &VerifiedInputSeal,
    repository: &RepositoryState,
) -> Result<(), Box<dyn Error>> {
    for (key, expected) in [
        ("schema", "sranibro.phase14.input-manifest.v1"),
        ("stage", "seal-input"),
        (
            "preregistration_commit",
            xr5_transfer_output::PREREGISTRATION_COMMIT,
        ),
        (
            "preregistration_blob_oid",
            xr5_transfer_output::PREREGISTRATION_BLOB_OID,
        ),
        ("implementation_commit", repository.commit.as_str()),
        ("build_commit", repository.commit.as_str()),
        ("runtime_commit", repository.commit.as_str()),
        (
            "compiled_source_sha256",
            repository.implementation_source_sha256.as_str(),
        ),
        (
            "phase13_decision_manifest_sha256",
            xr5_transfer_output::PHASE13_DECISION_MANIFEST_SHA256,
        ),
        (
            "phase13_confirmation_manifest_sha256",
            xr5_transfer_output::PHASE13_CONFIRMATION_MANIFEST_SHA256,
        ),
        ("build_profile", "release"),
        ("independence", "independence_unproven"),
    ] {
        require_manifest_string(&input.manifest, key, expected)?;
    }
    require_manifest_bool(&input.manifest, "clean_worktree", true)?;
    require_manifest_bool(&input.manifest, "debug_assertions", false)?;
    require_manifest_bool(&input.manifest, "model_loaded", false)?;
    require_manifest_bool(&input.manifest, "png_pixels_decoded", false)?;
    Ok(())
}

fn input_manifest(repository: &RepositoryState, plan: &BuiltInputPlan) -> Value {
    let pair_count: usize = plan
        .session_plan
        .completed_sessions
        .iter()
        .map(|session| session.pair_count)
        .sum();
    json!({
        "schema": "sranibro.phase14.input-manifest.v1",
        "stage": "seal-input",
        "terminal_status": plan.terminal_status(),
        "preregistration_commit": xr5_transfer_output::PREREGISTRATION_COMMIT,
        "preregistration_blob_oid": xr5_transfer_output::PREREGISTRATION_BLOB_OID,
        "implementation_commit": repository.commit,
        "build_commit": xr5_transfer_output::BUILD_COMMIT,
        "runtime_commit": repository.commit,
        "clean_worktree": !repository.dirty,
        "compiled_source_sha256": repository.implementation_source_sha256,
        "build_profile": xr5_transfer_output::BUILD_PROFILE,
        "debug_assertions": cfg!(debug_assertions),
        "phase13_decision_manifest_sha256": xr5_transfer_output::PHASE13_DECISION_MANIFEST_SHA256,
        "phase13_confirmation_manifest_sha256": xr5_transfer_output::PHASE13_CONFIRMATION_MANIFEST_SHA256,
        "model_loaded": false,
        "png_pixels_decoded": false,
        "recording_tree_sha256": plan.recording_tree_sha256,
        "inventory_entry_count": plan.inventory.entries.len(),
        "completed_session_count": plan.session_plan.completed_sessions.len(),
        "partial_session_count": plan.session_plan.partial_session_ids.len(),
        "stereo_pair_count": pair_count,
        "csv_row_count": pair_count * 2,
        "invalid_reason_codes": plan.session_plan.errors,
        "independence": "independence_unproven",
        "sanitized_cli": {
            "command": "seal-input",
            "phase13_decision": "<sealed-artifact>",
            "phase13_confirmation": "<sealed-artifact>",
            "wide_data": "<private-recording-root>",
            "out": "<new-artifact-directory>"
        }
    })
}

fn analysis_manifest(
    repository: &RepositoryState,
    input: &VerifiedInputSeal,
    before: &BuiltInputPlan,
    after: &BuiltInputPlan,
    outcome: &xr5_transfer::AnalysisOutcome,
) -> Value {
    let inventory_before = xr5_transfer_output::sha256_bytes(&before.recording_inventory_json);
    let inventory_after = xr5_transfer_output::sha256_bytes(&after.recording_inventory_json);
    let plan_before = xr5_transfer_output::sha256_bytes(&before.session_plan_json);
    let plan_after = xr5_transfer_output::sha256_bytes(&after.session_plan_json);
    let model_byte_len = outcome
        .metadata
        .model_bytes_obtained
        .then_some(outcome.metadata.model_byte_len);
    let model_sha256 = outcome
        .metadata
        .model_bytes_obtained
        .then_some(outcome.metadata.model_sha256.as_str());
    json!({
        "schema": "sranibro.phase14.analysis-manifest.v1",
        "stage": "analyze",
        "terminal_status": outcome.terminal_status,
        "high_open_annotation": outcome.high_open_annotation,
        "preregistration_commit": xr5_transfer_output::PREREGISTRATION_COMMIT,
        "preregistration_blob_oid": xr5_transfer_output::PREREGISTRATION_BLOB_OID,
        "implementation_commit": repository.commit,
        "build_commit": xr5_transfer_output::BUILD_COMMIT,
        "runtime_commit": repository.commit,
        "clean_worktree": !repository.dirty,
        "compiled_source_sha256": repository.implementation_source_sha256,
        "build_profile": xr5_transfer_output::BUILD_PROFILE,
        "debug_assertions": cfg!(debug_assertions),
        "phase13_decision_manifest_sha256": xr5_transfer_output::PHASE13_DECISION_MANIFEST_SHA256,
        "phase13_confirmation_manifest_sha256": xr5_transfer_output::PHASE13_CONFIRMATION_MANIFEST_SHA256,
        "input_manifest_sha256": input.manifest_sha256,
        "recording_tree_sha256": outcome.metadata.recording_tree_sha256,
        "recording_tree_sha256_before": before.recording_tree_sha256,
        "recording_tree_sha256_after": after.recording_tree_sha256,
        "recording_inventory_sha256_before": inventory_before,
        "recording_inventory_sha256_after": inventory_after,
        "session_plan_sha256_before": plan_before,
        "session_plan_sha256_after": plan_after,
        "model_loaded": outcome.metadata.model_loaded,
        "model_byte_len": model_byte_len,
        "model_sha256": model_sha256,
        "model_loader_invoked": outcome.metadata.model_loader_invoked,
        "model_bytes_obtained": outcome.metadata.model_bytes_obtained,
        "model_loader_error_kind": outcome.metadata.model_loader_error_kind,
        "completed_session_count": outcome.metadata.completed_session_count,
        "frame_count": outcome.metadata.frame_count,
        "retained_pairs_per_session": outcome.metadata.retained_pairs_per_session,
        "association_pairs_per_session": outcome.metadata.association_pairs_per_session,
        "preprocessing": outcome.metadata.preprocessing,
        "preprocessing_bit_identical": outcome.metadata.preprocessing_bit_identical,
        "model_bit_identical": outcome.metadata.model_bit_identical,
        "ordered_stream_digests": outcome.metadata.digests,
        "independence": outcome.metadata.independence,
        "sanitized_cli": {
            "command": "analyze",
            "phase13_decision": "<sealed-artifact>",
            "phase13_confirmation": "<sealed-artifact>",
            "wide_data": "<same-private-recording-root>",
            "input": "<sealed-input-artifact>",
            "input_seal": input.manifest_sha256,
            "model": "<fixed-model-file>",
            "out": "<new-artifact-directory>"
        }
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ModelObservation {
    NotInvoked,
    Stable(FileIdentity),
    ReadOrIdentityFailure(ModelEntrySnapshot),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ModelEntrySnapshot {
    metadata_error_kind: Option<String>,
    raw_os_error: Option<i32>,
    is_file: bool,
    is_dir: bool,
    is_symlink: bool,
    is_reparse: bool,
    byte_len: u64,
    readonly: bool,
    modified_from_unix_epoch: Option<(bool, u128)>,
}

fn read_stable_model(
    path: &Path,
    observation: &mut ModelObservation,
) -> Result<Vec<u8>, xr5_transfer::ModelLoaderError> {
    let entry_before = model_entry_snapshot(path);
    let before = match xr5_transfer_output::file_identity(path) {
        Ok(identity) => identity,
        Err(error) if error.to_string().contains("file changed while hashing") => {
            return Err(xr5_transfer::ModelLoaderError::changed_during_read(
                "model changed while its initial identity was being established",
            ))
        }
        Err(_) => {
            let entry_after = model_entry_snapshot(path);
            if entry_before != entry_after {
                return Err(xr5_transfer::ModelLoaderError::changed_during_read(
                    "model entry state changed while its failed identity was being established",
                ));
            }
            *observation = ModelObservation::ReadOrIdentityFailure(entry_after);
            return Err(xr5_transfer::ModelLoaderError::read_or_identity(
                "model file was unavailable, unreadable, or not an ordinary file",
            ));
        }
    };
    let bytes = fs::read(path).map_err(|_| {
        xr5_transfer::ModelLoaderError::changed_during_read(
            "model became unreadable after its initial identity was established",
        )
    })?;
    let from_bytes = FileIdentity {
        length: bytes.len() as u64,
        sha256: xr5_transfer_output::sha256_bytes(&bytes),
    };
    let after = xr5_transfer_output::file_identity(path).map_err(|_| {
        xr5_transfer::ModelLoaderError::changed_during_read(
            "model identity could not be re-established after reading",
        )
    })?;
    if before != from_bytes || before != after {
        return Err(xr5_transfer::ModelLoaderError::changed_during_read(
            "model content or identity changed while it was being read",
        ));
    }
    *observation = ModelObservation::Stable(before);
    Ok(bytes)
}

fn recheck_model_observation(
    path: &Path,
    observation: &ModelObservation,
) -> Result<(), Box<dyn Error>> {
    match observation {
        ModelObservation::NotInvoked => Ok(()),
        ModelObservation::Stable(expected) => {
            let observed = xr5_transfer_output::file_identity(path)?;
            xr5_transfer_output::require_unchanged("model file", expected, &observed)
        }
        ModelObservation::ReadOrIdentityFailure(expected) => {
            match xr5_transfer_output::file_identity(path) {
                Ok(_) => return Err("model availability changed during analysis".into()),
                Err(error) if error.to_string().contains("file changed while hashing") => {
                    return Err("model changed while its failed identity was rechecked".into())
                }
                Err(_) => {}
            }
            let observed = model_entry_snapshot(path);
            if &observed != expected {
                return Err("unavailable model entry state changed during analysis".into());
            }
            Ok(())
        }
    }
}

fn model_entry_snapshot(path: &Path) -> ModelEntrySnapshot {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            let modified_from_unix_epoch = metadata.modified().ok().map(|modified| match modified
                .duration_since(std::time::UNIX_EPOCH)
            {
                Ok(duration) => (false, duration.as_nanos()),
                Err(error) => (true, error.duration().as_nanos()),
            });
            #[cfg(windows)]
            let is_reparse = {
                use std::os::windows::fs::MetadataExt;
                metadata.file_attributes() & 0x0000_0400 != 0
            };
            #[cfg(not(windows))]
            let is_reparse = false;
            ModelEntrySnapshot {
                metadata_error_kind: None,
                raw_os_error: None,
                is_file: metadata.is_file(),
                is_dir: metadata.is_dir(),
                is_symlink: metadata.file_type().is_symlink(),
                is_reparse,
                byte_len: metadata.len(),
                readonly: metadata.permissions().readonly(),
                modified_from_unix_epoch,
            }
        }
        Err(error) => ModelEntrySnapshot {
            metadata_error_kind: Some(format!("{:?}", error.kind())),
            raw_os_error: error.raw_os_error(),
            is_file: false,
            is_dir: false,
            is_symlink: false,
            is_reparse: false,
            byte_len: 0,
            readonly: false,
            modified_from_unix_epoch: None,
        },
    }
}

fn require_manifest_string(
    manifest: &Value,
    key: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(format!("sealed input manifest has unexpected {key}").into());
    }
    Ok(())
}

fn require_manifest_bool(
    manifest: &Value,
    key: &str,
    expected: bool,
) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("sealed input manifest has unexpected {key}").into());
    }
    Ok(())
}

fn require_manifest_u64(manifest: &Value, key: &str, expected: u64) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(format!("sealed input manifest has unexpected {key}").into());
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CommonPaths {
    phase13_decision: PathBuf,
    phase13_confirmation: PathBuf,
    wide_data: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AnalyzePaths {
    common: CommonPaths,
    input: PathBuf,
    input_seal: String,
    model: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Args {
    SealInput(CommonPaths),
    Analyze(AnalyzePaths),
}

impl Args {
    fn parse(args: impl IntoIterator<Item = String>) -> Result<Option<Self>, Box<dyn Error>> {
        let mut args = args.into_iter();
        let Some(command) = args.next() else {
            return Err("missing required command: seal-input or analyze".into());
        };
        if matches!(command.as_str(), "-h" | "--help") {
            if args.next().is_some() {
                return Err("--help cannot be combined with other arguments".into());
            }
            return Ok(None);
        }
        let analyze = match command.as_str() {
            "seal-input" => false,
            "analyze" => true,
            unknown => return Err(format!("unknown command: {unknown}").into()),
        };

        let mut phase13_decision = None;
        let mut phase13_confirmation = None;
        let mut wide_data = None;
        let mut input = None;
        let mut input_seal = None;
        let mut model = None;
        let mut out = None;
        while let Some(option) = args.next() {
            match option.as_str() {
                "--phase13-decision" => set_once(
                    &mut phase13_decision,
                    next_path(&mut args, "--phase13-decision")?,
                    "--phase13-decision",
                )?,
                "--phase13-confirmation" => set_once(
                    &mut phase13_confirmation,
                    next_path(&mut args, "--phase13-confirmation")?,
                    "--phase13-confirmation",
                )?,
                "--wide-data" => set_once(
                    &mut wide_data,
                    next_path(&mut args, "--wide-data")?,
                    "--wide-data",
                )?,
                "--input" if analyze => {
                    set_once(&mut input, next_path(&mut args, "--input")?, "--input")?
                }
                "--input-seal" if analyze => {
                    let seal = next_string(&mut args, "--input-seal")?;
                    validate_sha256("--input-seal", &seal)?;
                    set_once(&mut input_seal, seal, "--input-seal")?;
                }
                "--model" if analyze => {
                    set_once(&mut model, next_path(&mut args, "--model")?, "--model")?
                }
                "--out" => set_once(&mut out, next_path(&mut args, "--out")?, "--out")?,
                "-h" | "--help" => {
                    return Err("--help must be used alone, without a command".into())
                }
                unknown => return Err(format!("unknown argument: {unknown}").into()),
            }
        }

        let common = CommonPaths {
            phase13_decision: phase13_decision
                .ok_or("missing required --phase13-decision <directory>")?,
            phase13_confirmation: phase13_confirmation
                .ok_or("missing required --phase13-confirmation <directory>")?,
            wide_data: wide_data.ok_or("missing required --wide-data <wide_data-root>")?,
            out: out.ok_or("missing required --out <new-directory>")?,
        };
        if analyze {
            Ok(Some(Self::Analyze(AnalyzePaths {
                common,
                input: input.ok_or("missing required --input <sealed-input-directory>")?,
                input_seal: input_seal.ok_or("missing required --input-seal <64-lowercase-hex>")?,
                model: model.ok_or("missing required --model <EyeNet-params-file>")?,
            })))
        } else {
            Ok(Some(Self::SealInput(common)))
        }
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

fn next_path(
    args: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, Box<dyn Error>> {
    Ok(PathBuf::from(next_string(args, option)?))
}

fn set_once<T>(slot: &mut Option<T>, value: T, option: &str) -> Result<(), Box<dyn Error>> {
    if slot.replace(value).is_some() {
        return Err(format!("{option} may be specified only once").into());
    }
    Ok(())
}

fn validate_sha256(option: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{option} must be exactly 64 lowercase hexadecimal characters").into());
    }
    Ok(())
}

fn print_help() {
    println!(
        "{}",
        concat!(
            "SRanibro XR5 EyeNet transfer/stability audit (research only)\n\n",
            "Input sealing (does not decode PNG pixels or accept a model):\n",
            "  synthetic-eye-xr5-transfer seal-input ",
            "--phase13-decision <directory> --phase13-confirmation <directory> ",
            "--wide-data <wide_data-root> --out <new-directory>\n\n",
            "Analysis:\n",
            "  synthetic-eye-xr5-transfer analyze ",
            "--phase13-decision <directory> --phase13-confirmation <directory> ",
            "--wide-data <same-wide_data-root> --input <sealed-input-directory> ",
            "--input-seal <64-lowercase-hex> --model <EyeNet-params-file> ",
            "--out <new-directory>\n\n",
            "Session selection, mutable configuration, geometry search, and production writes are not accepted."
        )
    );
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum RunOutcome {
    Completed {
        stage: &'static str,
        path: PathBuf,
        seal: String,
    },
    Help,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    const SEAL: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn seal_input_surface_has_no_model_or_session_selector() {
        let ok = Args::parse(strings(&[
            "seal-input",
            "--phase13-decision",
            "decision",
            "--phase13-confirmation",
            "confirmation",
            "--wide-data",
            "wide",
            "--out",
            "out",
        ]))
        .unwrap();
        assert!(matches!(ok, Some(Args::SealInput(_))));
        for forbidden in [
            "--model",
            "--input",
            "--input-seal",
            "--session",
            "--sessions",
        ] {
            let mut args = strings(&[
                "seal-input",
                "--phase13-decision",
                "decision",
                "--phase13-confirmation",
                "confirmation",
                "--wide-data",
                "wide",
                "--out",
                "out",
            ]);
            args.push(forbidden.into());
            args.push("value".into());
            assert!(Args::parse(args).is_err(), "accepted {forbidden}");
        }
    }

    #[test]
    fn analyze_requires_fixed_surface_and_lowercase_seal() {
        let ok = Args::parse(strings(&[
            "analyze",
            "--phase13-decision",
            "decision",
            "--phase13-confirmation",
            "confirmation",
            "--wide-data",
            "wide",
            "--input",
            "input",
            "--input-seal",
            SEAL,
            "--model",
            "model",
            "--out",
            "out",
        ]))
        .unwrap();
        assert!(matches!(ok, Some(Args::Analyze(_))));

        let uppercase = SEAL.to_uppercase();
        let result = Args::parse(strings(&[
            "analyze",
            "--phase13-decision",
            "decision",
            "--phase13-confirmation",
            "confirmation",
            "--wide-data",
            "wide",
            "--input",
            "input",
            "--input-seal",
            &uppercase,
            "--model",
            "model",
            "--out",
            "out",
        ]));
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_unknown_and_positional_arguments_are_rejected() {
        assert!(Args::parse(strings(&["seal-input", "wide"])).is_err());
        assert!(Args::parse(strings(&["seal-input", "--force", "yes"])).is_err());
        assert!(Args::parse(strings(&[
            "seal-input",
            "--phase13-decision",
            "a",
            "--phase13-decision",
            "b",
        ]))
        .is_err());
    }

    #[test]
    fn output_and_staging_parent_must_be_disjoint_from_protected_roots() {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "sranibro-phase14-main-disjoint-{}-{sequence}",
            std::process::id()
        ));
        let wide = root.join("wide_data");
        let nested_parent = wide.join("research-output");
        let phase13 = root.join("phase13");
        let safe_parent = root.join("safe-output");
        for directory in [&wide, &nested_parent, &phase13, &safe_parent] {
            std::fs::create_dir_all(directory).unwrap();
        }

        assert!(require_output_outside_protected_roots(
            &nested_parent.join("result"),
            &[("recording root", &wide)],
        )
        .is_err());
        assert!(require_output_outside_protected_roots(
            &phase13.join("nested-result"),
            &[("sealed artifact", &phase13)],
        )
        .is_err());
        require_output_outside_protected_roots(
            &safe_parent.join("result"),
            &[("recording root", &wide), ("sealed artifact", &phase13)],
        )
        .unwrap();

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn unavailable_model_entry_state_cannot_change_before_publication() {
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "sranibro-phase14-main-model-state-{}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let model = root.join("missing-model.params");
        let missing = model_entry_snapshot(&model);
        let observation = ModelObservation::ReadOrIdentityFailure(missing);
        recheck_model_observation(&model, &observation).unwrap();

        std::fs::create_dir(&model).unwrap();
        assert!(recheck_model_observation(&model, &observation).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
