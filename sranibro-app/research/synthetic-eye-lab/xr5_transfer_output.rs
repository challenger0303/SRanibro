//! Provenance, immutable-input verification, and sealed artifacts for Phase 1.4.
//!
//! This module deliberately has no dependency on the XR5 parser, image decoder,
//! preprocessing code, or model wrapper.  In particular, [`preflight`] can run
//! before the recording root is even constructed by the caller.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const BUILD_PROFILE: &str = env!("SRANIBRO_BUILD_PROFILE");
pub const BUILD_COMMIT: &str = env!("SRANIBRO_BUILD_COMMIT");

pub const PREREGISTRATION_COMMIT: &str = "d60b44666bc53f786679e0c49bbb0b08d7ad71a2";
pub const PREREGISTRATION_BLOB_OID: &str = "53491a99366c9d8203e0003bc5fd6ac740ba2960";
const PREREGISTRATION_REPOSITORY_PATH: &str =
    "./research/synthetic-eye-lab/PHASE1_4_XR5_TRANSFER_PREREG.md";
pub const PHASE13_DECISION_MANIFEST_SHA256: &str =
    "17291d72ab05034ea5c047225c6868ea714c6dbbebe2122beb41519dc02dab48";
pub const PHASE13_CONFIRMATION_MANIFEST_SHA256: &str =
    "2e26e3d94c9ab267862869cf7fc3cc8740a15f94e4dad7cf6df10b2e254da93c";
pub const FROZEN_MODEL_BYTE_LENGTH: u64 = 51_423_934;
pub const FROZEN_MODEL_BYTES: u64 = FROZEN_MODEL_BYTE_LENGTH;
pub const FROZEN_MODEL_SHA256: &str =
    "bac8013e0423068924f190a1de44afd5e1dd0c7c10d1d394926e46fc1b075ded";

pub const INPUT_SEAL_ALLOWLIST: &[&str] = &[
    "recording_inventory.json",
    "session_plan.json",
    "manifest.json",
];
pub const ANALYSIS_COMPLETE_ALLOWLIST: &[&str] = &[
    "frames.csv",
    "phase_summaries.json",
    "temporal_blocks.csv",
    "associations.csv",
    "gaze_and_session_differences.csv",
    "interpretation.txt",
    "manifest.json",
];
pub const ANALYSIS_REDUCED_ALLOWLIST: &[&str] =
    &["diagnostic.json", "interpretation.txt", "manifest.json"];
pub const PHASE13_DECISION_ALLOWLIST: &[&str] = &[
    "analysis.json",
    "raw_bits.json",
    "renderer_plan.json",
    "stage_cases.json",
    "manifest.json",
];
pub const PHASE13_CONFIRMATION_ALLOWLIST: &[&str] = &["confirmation.json", "manifest.json"];

const MANIFEST_FILE: &str = "manifest.json";
const STAGING_SUFFIX: &str = ".phase14-staging";
const SOURCE_FINGERPRINT_DOMAIN: &[u8] = b"SRanibro\0Phase-1.4\0compiled-source-fingerprint\0v1\0";

/// The exact source allowlist frozen by the Phase 1.4 preregistration.
pub const COMPILED_SOURCE_PATHS: [&str; 16] = [
    "Cargo.toml",
    "Cargo.lock",
    "build.rs",
    "research/synthetic-eye-lab/PHASE1_4_XR5_TRANSFER_PREREG.md",
    "research/synthetic-eye-lab/xr5_transfer_main.rs",
    "research/synthetic-eye-lab/xr5_transfer.rs",
    "research/synthetic-eye-lab/xr5_transfer_output.rs",
    "src/lib.rs",
    "src/config.rs",
    "src/core/types.rs",
    "src/ml/mod.rs",
    "src/ml/eye_net.rs",
    "src/ml/tvm_params.rs",
    "src/ml/preprocess.rs",
    "src/wide_calib.rs",
    "src/pipeline.rs",
];

const COMPILED_SOURCE_BYTES: [&[u8]; 16] = [
    include_bytes!("../../Cargo.toml"),
    include_bytes!("../../Cargo.lock"),
    include_bytes!("../../build.rs"),
    include_bytes!("PHASE1_4_XR5_TRANSFER_PREREG.md"),
    include_bytes!("xr5_transfer_main.rs"),
    include_bytes!("xr5_transfer.rs"),
    include_bytes!("xr5_transfer_output.rs"),
    include_bytes!("../../src/lib.rs"),
    include_bytes!("../../src/config.rs"),
    include_bytes!("../../src/core/types.rs"),
    include_bytes!("../../src/ml/mod.rs"),
    include_bytes!("../../src/ml/eye_net.rs"),
    include_bytes!("../../src/ml/tvm_params.rs"),
    include_bytes!("../../src/ml/preprocess.rs"),
    include_bytes!("../../src/wide_calib.rs"),
    include_bytes!("../../src/pipeline.rs"),
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RepositoryState {
    pub commit: String,
    pub dirty: bool,
    pub implementation_source_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileIdentity {
    pub length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileInventoryEntry {
    pub path: String,
    pub length: u64,
    pub sha256: String,
}

#[derive(Clone, Debug)]
pub struct ArtifactSeal {
    pub output_dir: PathBuf,
    pub manifest_bytes: Vec<u8>,
    pub manifest_sha256: String,
    pub inventory: Vec<FileInventoryEntry>,
}

#[derive(Clone, Debug)]
pub struct VerifiedArtifact {
    pub directory: PathBuf,
    pub manifest: Value,
    pub manifest_bytes: Vec<u8>,
    pub manifest_sha256: String,
    pub inventory: Vec<FileInventoryEntry>,
    /// Exact non-manifest bytes, keyed by normalized artifact-relative path.
    pub files: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct VerifiedInputSeal {
    pub directory: PathBuf,
    pub manifest: Value,
    pub manifest_bytes: Vec<u8>,
    pub manifest_sha256: String,
    pub inventory: Vec<FileInventoryEntry>,
    pub files: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone, Debug)]
pub struct VerifiedPhase13Artifacts {
    pub decision: VerifiedArtifact,
    pub confirmation: VerifiedArtifact,
}

/// Run every check that must precede construction or enumeration of the XR5
/// recording root. No argument to this function can name that root.
pub fn preflight(
    repository_root: &Path,
    decision_directory: &Path,
    confirmation_directory: &Path,
    new_output: &Path,
) -> Result<RepositoryState, Box<dyn Error>> {
    require_release_build()?;
    let repository = repository_state(repository_root)?;
    verify_runtime_identity(&repository)?;
    verify_preregistration(repository_root, &repository)?;
    verify_phase13_artifacts(decision_directory, confirmation_directory)?;
    validate_new_output_destination(new_output)?;
    recheck(repository_root, &repository)?;
    Ok(repository)
}

pub fn require_release_build() -> Result<(), Box<dyn Error>> {
    if BUILD_PROFILE != "release" || cfg!(debug_assertions) {
        return Err(format!(
            "recorded Phase 1.4 runs require release with debug assertions disabled; profile={BUILD_PROFILE}"
        )
        .into());
    }
    Ok(())
}

pub fn repository_state(repository_root: &Path) -> Result<RepositoryState, Box<dyn Error>> {
    let commit = git(repository_root, &["rev-parse", "--verify", "HEAD"])?;
    validate_git_commit("repository HEAD", &commit)?;
    let status = git(
        repository_root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    Ok(RepositoryState {
        commit,
        dirty: !status.is_empty(),
        implementation_source_sha256: checkout_source_fingerprint(repository_root)?,
    })
}

pub fn verify_runtime_identity(repository: &RepositoryState) -> Result<(), Box<dyn Error>> {
    if repository.dirty {
        return Err("recorded Phase 1.4 runs require a clean Git worktree".into());
    }
    if BUILD_COMMIT != repository.commit {
        return Err(format!(
            "executable build commit {BUILD_COMMIT} differs from runtime HEAD {}",
            repository.commit
        )
        .into());
    }
    let compiled = compiled_source_fingerprint();
    if compiled != repository.implementation_source_sha256 {
        return Err(format!(
            "compiled source fingerprint {compiled} differs from checkout source {}",
            repository.implementation_source_sha256
        )
        .into());
    }
    Ok(())
}

pub fn verify_preregistration(
    repository_root: &Path,
    repository: &RepositoryState,
) -> Result<(), Box<dyn Error>> {
    if repository.commit == PREREGISTRATION_COMMIT {
        return Err("Phase 1.4 implementation must be a descendant of, not the preregistration commit itself".into());
    }
    let status = Command::new("git")
        .args([
            "merge-base",
            "--is-ancestor",
            PREREGISTRATION_COMMIT,
            &repository.commit,
        ])
        .current_dir(repository_root)
        .status()?;
    if !status.success() {
        return Err(format!(
            "frozen Phase 1.4 preregistration {PREREGISTRATION_COMMIT} is not an ancestor of runtime HEAD {}",
            repository.commit
        )
        .into());
    }
    // Ancestry alone is insufficient: a descendant could rewrite the protocol
    // and still have the original commit as an ancestor. Pin the committed Git
    // blob as well. The clean-worktree and compiled-source checks separately
    // bind these frozen committed bytes to the file used by this executable.
    let frozen_spec = format!("{PREREGISTRATION_COMMIT}:{PREREGISTRATION_REPOSITORY_PATH}");
    let frozen_blob = git(repository_root, &["rev-parse", &frozen_spec])?;
    if frozen_blob != PREREGISTRATION_BLOB_OID {
        return Err(
            "frozen Phase 1.4 preregistration blob identity is unavailable or unexpected".into(),
        );
    }
    let runtime_spec = format!("{}:{PREREGISTRATION_REPOSITORY_PATH}", repository.commit);
    let runtime_blob = git(repository_root, &["rev-parse", &runtime_spec])?;
    if runtime_blob != PREREGISTRATION_BLOB_OID {
        return Err("Phase 1.4 preregistration bytes changed after the frozen commit".into());
    }
    Ok(())
}

/// Recompute the complete repository/runtime identity and require exact equality
/// with the preflight snapshot.
pub fn recheck(repository_root: &Path, expected: &RepositoryState) -> Result<(), Box<dyn Error>> {
    let observed = repository_state(repository_root)?;
    if &observed != expected {
        return Err("repository identity changed during the Phase 1.4 command".into());
    }
    verify_runtime_identity(&observed)
}

pub fn compiled_source_fingerprint() -> String {
    source_fingerprint(
        COMPILED_SOURCE_PATHS
            .iter()
            .copied()
            .zip(COMPILED_SOURCE_BYTES.iter().copied()),
    )
}

pub fn checkout_source_fingerprint(repository_root: &Path) -> Result<String, Box<dyn Error>> {
    let mut entries = Vec::with_capacity(COMPILED_SOURCE_PATHS.len());
    for path in COMPILED_SOURCE_PATHS {
        let source = repository_root.join(path);
        ensure_ordinary_file(&source, "compiled-source member")?;
        entries.push((path, std::fs::read(source)?));
    }
    Ok(source_fingerprint(
        entries
            .iter()
            .map(|(path, bytes)| (*path, bytes.as_slice())),
    ))
}

/// Domain-separated source hash over UTF-8-byte-sorted paths and exact bytes.
pub fn source_fingerprint<'a>(entries: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut entries: Vec<_> = entries.into_iter().collect();
    entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(SOURCE_FINGERPRINT_DOMAIN);
    hasher.update((entries.len() as u64).to_le_bytes());
    for (path, bytes) in entries {
        hasher.update(b"file\0");
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    digest_to_hex(hasher.finalize())
}

pub fn verify_phase13_artifacts(
    decision_directory: &Path,
    confirmation_directory: &Path,
) -> Result<VerifiedPhase13Artifacts, Box<dyn Error>> {
    let decision = verify_sealed_artifact(
        decision_directory,
        PHASE13_DECISION_MANIFEST_SHA256,
        PHASE13_DECISION_ALLOWLIST,
    )?;
    validate_phase13_decision(&decision.manifest)?;

    let confirmation = verify_sealed_artifact(
        confirmation_directory,
        PHASE13_CONFIRMATION_MANIFEST_SHA256,
        PHASE13_CONFIRMATION_ALLOWLIST,
    )?;
    validate_phase13_confirmation(&confirmation.manifest)?;

    // Close the pair-verification window as well as each artifact's internal
    // verification window. Fixed external seals make any changed byte fatal.
    let decision_after = verify_sealed_artifact(
        decision_directory,
        PHASE13_DECISION_MANIFEST_SHA256,
        PHASE13_DECISION_ALLOWLIST,
    )?;
    let confirmation_after = verify_sealed_artifact(
        confirmation_directory,
        PHASE13_CONFIRMATION_MANIFEST_SHA256,
        PHASE13_CONFIRMATION_ALLOWLIST,
    )?;
    require_same_verified_artifact("Phase 1.3 decision", &decision, &decision_after)?;
    require_same_verified_artifact("Phase 1.3 confirmation", &confirmation, &confirmation_after)?;

    Ok(VerifiedPhase13Artifacts {
        decision,
        confirmation,
    })
}

fn require_same_verified_artifact(
    label: &str,
    first: &VerifiedArtifact,
    second: &VerifiedArtifact,
) -> Result<(), Box<dyn Error>> {
    if first.manifest_bytes != second.manifest_bytes
        || first.inventory != second.inventory
        || first.files != second.files
    {
        return Err(format!("{label} changed while the sealed pair was verified").into());
    }
    Ok(())
}

fn validate_phase13_decision(manifest: &Value) -> Result<(), Box<dyn Error>> {
    require_manifest_string(
        manifest,
        "artifact_schema",
        "synthetic-eye-phase13-stage-v1",
    )?;
    require_manifest_string(manifest, "stage", "decision")?;
    require_manifest_string(manifest, "terminal_status", "INCONCLUSIVE_INSENSITIVE")?;
    require_manifest_string(manifest, "analysis_file", "analysis.json")?;
    require_manifest_string(manifest, "case_records_file", "stage_cases.json")?;
    require_manifest_string(manifest, "raw_bits_file", "raw_bits.json")?;
    require_manifest_string(manifest, "build_profile", "release")?;
    require_manifest_bool(manifest, "repository_dirty", false)?;
    require_manifest_bool(manifest, "debug_assertions", false)?;
    require_manifest_bool(manifest, "model_loaded", true)?;
    require_manifest_bool(manifest, "real_recordings_loaded", false)?;
    require_manifest_string(manifest, "model_identity_sha256", FROZEN_MODEL_SHA256)?;
    require_manifest_u64(manifest, "model_byte_length", FROZEN_MODEL_BYTE_LENGTH)?;
    require_manifest_u64(
        manifest,
        "expected_model_byte_length",
        FROZEN_MODEL_BYTE_LENGTH,
    )?;
    require_manifest_null(manifest, "response_class")?;
    Ok(())
}

fn validate_phase13_confirmation(manifest: &Value) -> Result<(), Box<dyn Error>> {
    require_manifest_string(
        manifest,
        "artifact_schema",
        "synthetic-eye-phase13-stage-v1",
    )?;
    require_manifest_string(manifest, "stage", "confirmation")?;
    require_manifest_string(manifest, "terminal_status", "INCONCLUSIVE")?;
    require_manifest_string(manifest, "confirmation_status", "CONFIRMATION_INCONCLUSIVE")?;
    require_manifest_string(manifest, "analysis_file", "confirmation.json")?;
    require_manifest_string(manifest, "build_profile", "release")?;
    require_manifest_bool(manifest, "repository_dirty", false)?;
    require_manifest_bool(manifest, "debug_assertions", false)?;
    require_manifest_bool(manifest, "model_loaded", false)?;
    require_manifest_bool(manifest, "real_recordings_loaded", false)?;
    require_manifest_string(
        manifest,
        "decision_external_seal_sha256",
        PHASE13_DECISION_MANIFEST_SHA256,
    )?;
    require_manifest_string(
        manifest,
        "decision_manifest_sha256",
        PHASE13_DECISION_MANIFEST_SHA256,
    )?;
    require_manifest_string(
        manifest,
        "decision_terminal_status",
        "INCONCLUSIVE_INSENSITIVE",
    )?;
    require_manifest_u64(
        manifest,
        "expected_model_byte_length",
        FROZEN_MODEL_BYTE_LENGTH,
    )?;
    require_manifest_null(manifest, "model_identity_sha256")?;
    require_manifest_null(manifest, "model_byte_length")?;
    require_manifest_null(manifest, "response_class")?;
    Ok(())
}

/// Validate and hash a file without retaining its bytes.
pub fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    Ok(file_identity(path)?.sha256)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    digest_to_hex(Sha256::digest(bytes))
}

pub fn file_identity(path: &Path) -> Result<FileIdentity, Box<dyn Error>> {
    ensure_ordinary_file(path, "identity input")?;
    let before = std::fs::metadata(path)?;
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut length = 0u64;
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(count as u64)
            .ok_or("file length overflow while hashing")?;
        hasher.update(&buffer[..count]);
    }
    let after = std::fs::metadata(path)?;
    if before.len() != length || after.len() != length {
        return Err(format!("file changed while hashing: {}", path.display()).into());
    }
    Ok(FileIdentity {
        length,
        sha256: digest_to_hex(hasher.finalize()),
    })
}

pub fn verify_frozen_model(path: &Path) -> Result<FileIdentity, Box<dyn Error>> {
    let identity = file_identity(path)?;
    require_frozen_model_identity(&identity)?;
    Ok(identity)
}

pub fn read_frozen_model(path: &Path) -> Result<(Vec<u8>, FileIdentity), Box<dyn Error>> {
    ensure_ordinary_file(path, "model")?;
    let bytes = std::fs::read(path)?;
    let identity = FileIdentity {
        length: bytes.len() as u64,
        sha256: sha256_bytes(&bytes),
    };
    require_frozen_model_identity(&identity)?;
    let reopened = file_identity(path)?;
    if reopened != identity {
        return Err("model changed while it was being read".into());
    }
    Ok((bytes, identity))
}

pub fn recheck_model(path: &Path, expected: &FileIdentity) -> Result<(), Box<dyn Error>> {
    require_frozen_model_identity(expected)?;
    let observed = verify_frozen_model(path)?;
    require_unchanged("fixed model", expected, &observed)
}

fn require_frozen_model_identity(identity: &FileIdentity) -> Result<(), Box<dyn Error>> {
    if identity.length != FROZEN_MODEL_BYTE_LENGTH || identity.sha256 != FROZEN_MODEL_SHA256 {
        return Err(format!(
            "model identity mismatch: length={}, sha256={}",
            identity.length, identity.sha256
        )
        .into());
    }
    Ok(())
}

/// Core-type-independent equality guard used for recomputed recording-tree or
/// plan identities. It intentionally omits values from the error to avoid
/// leaking original session names through a higher-level identity type.
pub fn require_unchanged<T: PartialEq>(
    label: &str,
    expected: &T,
    observed: &T,
) -> Result<(), Box<dyn Error>> {
    if expected != observed {
        return Err(format!("{label} changed during the Phase 1.4 command").into());
    }
    Ok(())
}

pub fn recheck_tree<T, F>(label: &str, expected: &T, recompute: F) -> Result<T, Box<dyn Error>>
where
    T: PartialEq,
    F: FnOnce() -> Result<T, Box<dyn Error>>,
{
    let observed = recompute()?;
    require_unchanged(label, expected, &observed)?;
    Ok(observed)
}

/// Reject an existing output *and* the deterministic staging name. This is
/// non-mutating and is safe to call before any private input is opened.
pub fn validate_new_output_destination(output: &Path) -> Result<(), Box<dyn Error>> {
    if path_entry_exists(output)? {
        return Err(format!("output path already exists: {}", output.display()).into());
    }
    let parent = normalized_parent(output)?;
    ensure_ordinary_directory(parent, "output parent")?;
    let staging = staging_path(output)?;
    if path_entry_exists(&staging)? {
        return Err(format!(
            "partial staging directory already exists: {}",
            staging.display()
        )
        .into());
    }
    Ok(())
}

/// Write an exact-allowlist artifact through a private staging directory. All
/// non-manifest files are reopened and hashed, their inventory is injected into
/// the manifest, and `manifest.json` is written last. The returned SHA-256 is the
/// external seal over the raw manifest bytes and is never embedded in it.
pub fn write_sealed_artifact(
    output: &Path,
    nonmanifest_files: BTreeMap<String, Vec<u8>>,
    manifest: Value,
    exact_allowlist: &[&str],
) -> Result<ArtifactSeal, Box<dyn Error>> {
    write_sealed_artifact_with_guard(output, nonmanifest_files, manifest, exact_allowlist, || {
        Ok(())
    })
}

/// Identical to [`write_sealed_artifact`], with one final caller-supplied trust
/// guard. The guard runs exactly once after the complete staged artifact has
/// been verified and immediately before the destination check and atomic rename.
/// A guard or other pre-rename failure removes only this writer's deterministic
/// staging directory; a successfully renamed output is never removed.
pub fn write_sealed_artifact_with_guard<F>(
    output: &Path,
    nonmanifest_files: BTreeMap<String, Vec<u8>>,
    manifest: Value,
    exact_allowlist: &[&str],
    guard: F,
) -> Result<ArtifactSeal, Box<dyn Error>>
where
    F: FnOnce() -> Result<(), Box<dyn Error>>,
{
    validate_new_output_destination(output)?;
    let allowlist = validate_allowlist(exact_allowlist)?;
    let expected_nonmanifest: BTreeSet<String> = allowlist
        .iter()
        .filter(|path| path.as_str() != MANIFEST_FILE)
        .cloned()
        .collect();
    let provided: BTreeSet<String> = nonmanifest_files.keys().cloned().collect();
    if provided != expected_nonmanifest {
        return Err(format!(
            "artifact non-manifest allowlist mismatch: provided={provided:?}, expected={expected_nonmanifest:?}"
        )
        .into());
    }
    for path in nonmanifest_files.keys() {
        validate_relative_file(path)?;
    }

    let mut manifest_object = manifest
        .as_object()
        .cloned()
        .ok_or("artifact manifest fields must be a JSON object")?;
    for reserved in ["file_inventory", "manifest_self_sha256", "manifest_sha256"] {
        if manifest_object.contains_key(reserved) {
            return Err(format!("manifest field is reserved: {reserved}").into());
        }
    }

    let staging = staging_path(output)?;
    std::fs::create_dir(&staging)?;
    let mut published = false;
    let result = (|| -> Result<ArtifactSeal, Box<dyn Error>> {
        for (relative, bytes) in &nonmanifest_files {
            let path = staging.join(relative);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_new_sync(&path, bytes)?;
        }

        let inventory = inventory_from_tree(&staging, &expected_nonmanifest)?;
        for entry in &inventory {
            let expected = nonmanifest_files
                .get(&entry.path)
                .ok_or("internal artifact inventory mismatch")?;
            if entry.length != expected.len() as u64 || entry.sha256 != sha256_bytes(expected) {
                return Err(format!("staged artifact bytes changed: {}", entry.path).into());
            }
        }
        manifest_object.insert("file_inventory".into(), serde_json::to_value(&inventory)?);
        let manifest_value = Value::Object(manifest_object);
        let mut manifest_bytes = serde_json::to_vec_pretty(&manifest_value)?;
        manifest_bytes.push(b'\n');
        write_new_sync(&staging.join(MANIFEST_FILE), &manifest_bytes)?;
        sync_directory(&staging)?;

        let manifest_sha256 = sha256_bytes(&manifest_bytes);
        let verified = verify_sealed_artifact(&staging, &manifest_sha256, exact_allowlist)?;
        if verified.manifest_bytes != manifest_bytes || verified.inventory != inventory {
            return Err("staged artifact changed before publication".into());
        }

        let parent = normalized_parent(output)?;
        ensure_ordinary_directory(parent, "output parent")?;
        sync_directory(parent)?;

        guard()?;
        if path_entry_exists(output)? {
            return Err(format!(
                "output path appeared during artifact publication: {}",
                output.display()
            )
            .into());
        }
        std::fs::rename(&staging, output)?;
        published = true;
        sync_directory(output)?;
        sync_directory(parent)?;

        let published = verify_sealed_artifact(output, &manifest_sha256, exact_allowlist)?;
        if published.manifest_bytes != manifest_bytes || published.inventory != inventory {
            return Err("published artifact differs from the sealed staging artifact".into());
        }
        Ok(ArtifactSeal {
            output_dir: output.to_path_buf(),
            manifest_bytes,
            manifest_sha256,
            inventory,
        })
    })();
    match result {
        Ok(seal) => Ok(seal),
        Err(error) if published => Err(error),
        Err(error) => match cleanup_deterministic_staging(output, &staging) {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(format!(
                "{error}; deterministic staging cleanup also failed: {cleanup_error}"
            )
            .into()),
        },
    }
}

/// Verify the external raw-manifest seal before parsing JSON, then enforce the
/// exact stage allowlist and every non-manifest length/hash in `file_inventory`.
pub fn verify_sealed_artifact(
    directory: &Path,
    external_manifest_sha256: &str,
    exact_allowlist: &[&str],
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    validate_lower_sha256("external manifest seal", external_manifest_sha256)?;
    ensure_ordinary_directory(directory, "sealed artifact")?;
    let manifest_path = directory.join(MANIFEST_FILE);
    ensure_ordinary_file(&manifest_path, "sealed manifest")?;
    let manifest_bytes = std::fs::read(&manifest_path)?;
    let observed_manifest_sha256 = sha256_bytes(&manifest_bytes);
    if observed_manifest_sha256 != external_manifest_sha256 {
        return Err(format!(
            "raw manifest SHA-256 {observed_manifest_sha256} did not match external seal {external_manifest_sha256}"
        )
        .into());
    }

    // The seal comparison above intentionally precedes this parse.
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    let object = manifest
        .as_object()
        .ok_or("sealed manifest must be a JSON object")?;
    for forbidden in ["manifest_self_sha256", "manifest_sha256"] {
        if object.contains_key(forbidden) {
            return Err(
                format!("sealed manifest contains forbidden self hash: {forbidden}").into(),
            );
        }
    }

    let allowlist = validate_allowlist(exact_allowlist)?;
    let actual = scan_artifact_tree(directory)?;
    require_exact_tree(&actual, &allowlist)?;

    let inventory: Vec<FileInventoryEntry> = serde_json::from_value(
        object
            .get("file_inventory")
            .cloned()
            .ok_or("sealed manifest file_inventory is missing")?,
    )?;
    let expected_nonmanifest: BTreeSet<String> = allowlist
        .iter()
        .filter(|path| path.as_str() != MANIFEST_FILE)
        .cloned()
        .collect();
    validate_inventory(&inventory, &expected_nonmanifest)?;

    let mut files = BTreeMap::new();
    for entry in &inventory {
        let path = directory.join(&entry.path);
        ensure_ordinary_file(&path, "sealed artifact member")?;
        let bytes = std::fs::read(&path)?;
        if bytes.len() as u64 != entry.length || sha256_bytes(&bytes) != entry.sha256 {
            return Err(format!("sealed artifact member changed: {}", entry.path).into());
        }
        files.insert(entry.path.clone(), bytes);
    }

    // Close the verification window: the manifest, exact tree and every member
    // identity must still match after the first complete read.
    if std::fs::read(&manifest_path)? != manifest_bytes {
        return Err("sealed manifest changed during verification".into());
    }
    let final_tree = scan_artifact_tree(directory)?;
    require_exact_tree(&final_tree, &allowlist)?;
    for entry in &inventory {
        let observed = file_identity(&directory.join(&entry.path))?;
        if observed.length != entry.length || observed.sha256 != entry.sha256 {
            return Err(format!(
                "sealed artifact member changed during verification: {}",
                entry.path
            )
            .into());
        }
    }

    Ok(VerifiedArtifact {
        directory: directory.to_path_buf(),
        manifest,
        manifest_bytes,
        manifest_sha256: observed_manifest_sha256,
        inventory,
        files,
    })
}

pub fn verify_input_seal(
    directory: &Path,
    external_manifest_sha256: &str,
) -> Result<VerifiedInputSeal, Box<dyn Error>> {
    let artifact =
        verify_sealed_artifact(directory, external_manifest_sha256, INPUT_SEAL_ALLOWLIST)?;
    require_manifest_string(&artifact.manifest, "terminal_status", "INPUT_SEALED")?;
    Ok(VerifiedInputSeal {
        directory: artifact.directory,
        manifest: artifact.manifest,
        manifest_bytes: artifact.manifest_bytes,
        manifest_sha256: artifact.manifest_sha256,
        inventory: artifact.inventory,
        files: artifact.files,
    })
}

/// Validate a completed or reduced Phase 1.4 analysis artifact using the
/// status-selected exact allowlist. The external seal is checked before status
/// is parsed.
pub fn verify_analysis_artifact(
    directory: &Path,
    external_manifest_sha256: &str,
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    let (manifest_bytes, manifest) = read_sealed_manifest(directory, external_manifest_sha256)?;
    let status = manifest
        .get("terminal_status")
        .and_then(Value::as_str)
        .ok_or("analysis manifest terminal_status is missing")?
        .to_owned();
    let allowlist = match status.as_str() {
        "AUDIT_COMPLETE" => ANALYSIS_COMPLETE_ALLOWLIST,
        "INPUT_INVALID" | "INCONCLUSIVE_DETERMINISM" | "INCONCLUSIVE_ARTIFACT" => {
            ANALYSIS_REDUCED_ALLOWLIST
        }
        other => return Err(format!("unknown Phase 1.4 analysis terminal status: {other}").into()),
    };
    let artifact = verify_sealed_artifact(directory, external_manifest_sha256, allowlist)?;
    if artifact.manifest_bytes != manifest_bytes || artifact.manifest != manifest {
        return Err("analysis manifest changed during verification".into());
    }
    if status != "AUDIT_COMPLETE"
        && artifact.manifest.get("high_open_annotation") != Some(&Value::Null)
    {
        return Err("non-complete analysis must have a null high_open_annotation".into());
    }
    Ok(artifact)
}

fn read_sealed_manifest(
    directory: &Path,
    external_manifest_sha256: &str,
) -> Result<(Vec<u8>, Value), Box<dyn Error>> {
    validate_lower_sha256("external manifest seal", external_manifest_sha256)?;
    ensure_ordinary_directory(directory, "sealed artifact")?;
    let path = directory.join(MANIFEST_FILE);
    ensure_ordinary_file(&path, "sealed manifest")?;
    let bytes = std::fs::read(path)?;
    let observed = sha256_bytes(&bytes);
    if observed != external_manifest_sha256 {
        return Err(format!(
            "raw manifest SHA-256 {observed} did not match external seal {external_manifest_sha256}"
        )
        .into());
    }
    let value = serde_json::from_slice(&bytes)?;
    Ok((bytes, value))
}

fn validate_inventory(
    inventory: &[FileInventoryEntry],
    expected: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let mut observed = BTreeSet::new();
    let mut prior: Option<&str> = None;
    for entry in inventory {
        validate_relative_file(&entry.path)?;
        validate_lower_sha256("artifact member SHA-256", &entry.sha256)?;
        if entry.path == MANIFEST_FILE {
            return Err("manifest must not include its own inventory entry".into());
        }
        if prior.is_some_and(|value| value >= entry.path.as_str()) {
            return Err("artifact file_inventory is not strictly sorted".into());
        }
        prior = Some(&entry.path);
        if !observed.insert(entry.path.clone()) {
            return Err("artifact file_inventory contains a duplicate path".into());
        }
    }
    if &observed != expected {
        return Err(format!(
            "artifact file_inventory allowlist mismatch: observed={observed:?}, expected={expected:?}"
        )
        .into());
    }
    Ok(())
}

fn inventory_from_tree(
    root: &Path,
    expected_nonmanifest: &BTreeSet<String>,
) -> Result<Vec<FileInventoryEntry>, Box<dyn Error>> {
    let tree = scan_artifact_tree(root)?;
    require_exact_tree(&tree, expected_nonmanifest)?;
    let mut inventory = Vec::with_capacity(expected_nonmanifest.len());
    for relative in expected_nonmanifest {
        let path = root.join(relative);
        let identity = file_identity(&path)?;
        inventory.push(FileInventoryEntry {
            path: relative.clone(),
            length: identity.length,
            sha256: identity.sha256,
        });
    }
    Ok(inventory)
}

fn validate_allowlist(paths: &[&str]) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut allowlist = BTreeSet::new();
    for path in paths {
        validate_relative_file(path)?;
        if !allowlist.insert((*path).to_owned()) {
            return Err(format!("duplicate artifact allowlist path: {path}").into());
        }
    }
    if !allowlist.contains(MANIFEST_FILE) {
        return Err("artifact allowlist must contain manifest.json".into());
    }
    Ok(allowlist)
}

fn require_manifest_string(
    manifest: &Value,
    key: &str,
    expected: &str,
) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(format!("manifest {key} mismatch").into());
    }
    Ok(())
}

fn require_manifest_bool(
    manifest: &Value,
    key: &str,
    expected: bool,
) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("manifest {key} mismatch").into());
    }
    Ok(())
}

fn require_manifest_u64(manifest: &Value, key: &str, expected: u64) -> Result<(), Box<dyn Error>> {
    if manifest.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(format!("manifest {key} mismatch").into());
    }
    Ok(())
}

fn require_manifest_null(manifest: &Value, key: &str) -> Result<(), Box<dyn Error>> {
    if manifest.get(key) != Some(&Value::Null) {
        return Err(format!("manifest {key} must be null").into());
    }
    Ok(())
}

fn validate_git_commit(label: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if !matches!(value.len(), 40 | 64)
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid {label}: {value:?}").into());
    }
    Ok(())
}

fn validate_lower_sha256(label: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "invalid {label}: expected exactly 64 lowercase hexadecimal characters"
        )
        .into());
    }
    Ok(())
}

fn validate_relative_file(path: &str) -> Result<(), Box<dyn Error>> {
    if path.is_empty() || path.contains('\\') {
        return Err(format!("artifact path must use nonempty forward-slash form: {path:?}").into());
    }
    let parsed = Path::new(path);
    if parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("unsafe artifact path: {path:?}").into());
    }
    Ok(())
}

fn normalized_parent(path: &Path) -> Result<&Path, Box<dyn Error>> {
    if path.file_name().is_none() {
        return Err("output must be a named path".into());
    }
    let parent = path.parent().ok_or("output path has no parent")?;
    Ok(if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    })
}

fn path_entry_exists(path: &Path) -> Result<bool, Box<dyn Error>> {
    match std::fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn staging_path(output: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("output file name must be UTF-8")?;
    Ok(normalized_parent(output)?.join(format!(".{name}{STAGING_SUFFIX}")))
}

fn cleanup_deterministic_staging(output: &Path, staging: &Path) -> Result<(), Box<dyn Error>> {
    let expected = staging_path(output)?;
    if staging != expected {
        return Err("refusing to clean a non-deterministic staging path".into());
    }
    if !path_entry_exists(staging)? {
        return Ok(());
    }

    let parent = normalized_parent(output)?;
    ensure_ordinary_directory(parent, "output parent")?;
    ensure_ordinary_directory(staging, "artifact staging directory")?;

    // Resolve both paths immediately before recursive removal. The staging
    // target must still be an ordinary direct child of the explicitly named
    // output parent and retain the exact deterministic basename.
    let resolved_parent = std::fs::canonicalize(parent)?;
    let resolved_staging = std::fs::canonicalize(staging)?;
    if resolved_staging.parent() != Some(resolved_parent.as_path())
        || resolved_staging.file_name() != expected.file_name()
    {
        return Err("refusing to clean staging outside the validated output parent".into());
    }

    // Reject nested symlinks, junctions/reparse points and special entries
    // before remove_dir_all is allowed to recurse.
    let _ = scan_artifact_tree(staging)?;
    std::fs::remove_dir_all(staging)?;
    sync_directory(parent)?;
    Ok(())
}

fn write_new_sync(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct ArtifactTree {
    files: BTreeSet<String>,
    directories: BTreeSet<String>,
}

fn scan_artifact_tree(root: &Path) -> Result<ArtifactTree, Box<dyn Error>> {
    fn visit(
        root: &Path,
        directory: &Path,
        files: &mut BTreeSet<String>,
        directories: &mut BTreeSet<String>,
    ) -> Result<(), Box<dyn Error>> {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let path = entry.path();
            let relative = path
                .strip_prefix(root)?
                .to_str()
                .ok_or("artifact tree contains a non-UTF8 path")?
                .replace('\\', "/");
            validate_relative_file(&relative)?;
            let file_type = entry.file_type()?;
            if file_type.is_symlink() || is_reparse_point(&path)? {
                return Err("artifact tree contains a symlink or reparse point".into());
            }
            if file_type.is_dir() {
                if !directories.insert(relative) {
                    return Err("artifact tree contains a duplicate directory".into());
                }
                visit(root, &path, files, directories)?;
            } else if file_type.is_file() {
                if !files.insert(relative) {
                    return Err("artifact tree contains a duplicate file".into());
                }
            } else {
                return Err("artifact tree contains a special entry".into());
            }
        }
        Ok(())
    }

    let mut files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    visit(root, root, &mut files, &mut directories)?;
    Ok(ArtifactTree { files, directories })
}

fn require_exact_tree(
    tree: &ArtifactTree,
    expected_files: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let expected_directories = required_directories(expected_files)?;
    if tree.files != *expected_files || tree.directories != expected_directories {
        return Err(format!(
            "artifact tree allowlist mismatch: files={:?}, directories={:?}",
            tree.files, tree.directories
        )
        .into());
    }
    Ok(())
}

fn required_directories(
    expected_files: &BTreeSet<String>,
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut directories = BTreeSet::new();
    for path in expected_files {
        validate_relative_file(path)?;
        let components: Vec<_> = Path::new(path)
            .components()
            .map(|component| match component {
                Component::Normal(value) => value
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "artifact component is not UTF-8".to_owned()),
                _ => Err("artifact path is not normalized".to_owned()),
            })
            .collect::<Result<_, _>>()?;
        for length in 1..components.len() {
            directories.insert(components[..length].join("/"));
        }
    }
    Ok(directories)
}

fn ensure_ordinary_file(path: &Path, label: &str) -> Result<(), Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("{label} unavailable at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file()
        || metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
    {
        return Err(format!(
            "{label} is not an ordinary regular file: {}",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn ensure_ordinary_directory(path: &Path, label: &str) -> Result<(), Box<dyn Error>> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("{label} unavailable at {}: {error}", path.display()))?;
    if !metadata.file_type().is_dir()
        || metadata.file_type().is_symlink()
        || metadata_is_reparse_point(&metadata)
    {
        return Err(format!("{label} is not an ordinary directory: {}", path.display()).into());
    }
    Ok(())
}

fn is_reparse_point(path: &Path) -> Result<bool, Box<dyn Error>> {
    Ok(metadata_is_reparse_point(&std::fs::symlink_metadata(path)?))
}

#[cfg(windows)]
fn metadata_is_reparse_point(metadata: &std::fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn metadata_is_reparse_point(_metadata: &std::fs::Metadata) -> bool {
    false
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(windows)]
fn sync_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
    let directory = match OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
        .open(path)
    {
        Ok(directory) => directory,
        Err(error) if windows_directory_sync_unavailable(&error) => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    match directory.sync_all() {
        Ok(()) => Ok(()),
        Err(error) if windows_directory_sync_unavailable(&error) => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(windows)]
fn windows_directory_sync_unavailable(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::Unsupported
        || error.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(error.raw_os_error(), Some(1) | Some(5) | Some(6) | Some(50))
}

#[cfg(not(any(unix, windows)))]
fn sync_directory(_path: &Path) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn digest_to_hex(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut output = String::with_capacity(bytes.len() * 2);
    use std::fmt::Write as _;
    for byte in bytes {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn git(repository_root: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(repository_root)
        .output()?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn unique_temp(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sranibro-phase14-output-{label}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn make_parent(label: &str) -> PathBuf {
        let root = unique_temp(label);
        if root.exists() {
            std::fs::remove_dir_all(&root).unwrap();
        }
        std::fs::create_dir(&root).unwrap();
        root
    }

    #[test]
    fn source_allowlist_is_exact_and_fingerprint_is_order_independent() {
        assert_eq!(
            COMPILED_SOURCE_PATHS,
            [
                "Cargo.toml",
                "Cargo.lock",
                "build.rs",
                "research/synthetic-eye-lab/PHASE1_4_XR5_TRANSFER_PREREG.md",
                "research/synthetic-eye-lab/xr5_transfer_main.rs",
                "research/synthetic-eye-lab/xr5_transfer.rs",
                "research/synthetic-eye-lab/xr5_transfer_output.rs",
                "src/lib.rs",
                "src/config.rs",
                "src/core/types.rs",
                "src/ml/mod.rs",
                "src/ml/eye_net.rs",
                "src/ml/tvm_params.rs",
                "src/ml/preprocess.rs",
                "src/wide_calib.rs",
                "src/pipeline.rs",
            ]
        );
        let forward = source_fingerprint([("b", b"two".as_slice()), ("a", b"one".as_slice())]);
        let reverse = source_fingerprint([("a", b"one".as_slice()), ("b", b"two".as_slice())]);
        assert_eq!(forward, reverse);
        assert_ne!(
            forward,
            source_fingerprint([("a", b"onet".as_slice()), ("b", b"wo".as_slice())])
        );
        assert_eq!(compiled_source_fingerprint().len(), 64);
    }

    #[test]
    fn strict_lowercase_seals_and_safe_relative_paths_are_enforced() {
        assert!(validate_lower_sha256("seal", &"a".repeat(64)).is_ok());
        assert!(validate_lower_sha256("seal", &"A".repeat(64)).is_err());
        assert!(validate_lower_sha256("seal", &"g".repeat(64)).is_err());
        assert!(validate_lower_sha256("seal", &"a".repeat(63)).is_err());
        assert!(validate_relative_file("nested/result.json").is_ok());
        assert!(validate_relative_file("../escape").is_err());
        assert!(validate_relative_file("nested\\escape").is_err());
        assert!(validate_relative_file("/absolute").is_err());
    }

    #[test]
    fn writer_injects_nonmanifest_hashes_and_external_seal_only() {
        let root = make_parent("writer");
        let output = root.join("sealed");
        let files = BTreeMap::from([
            (
                "recording_inventory.json".to_owned(),
                b"inventory\n".to_vec(),
            ),
            ("session_plan.json".to_owned(), b"plan\n".to_vec()),
        ]);
        let seal = write_sealed_artifact(
            &output,
            files,
            json!({"terminal_status": "INPUT_SEALED"}),
            INPUT_SEAL_ALLOWLIST,
        )
        .unwrap();
        assert_eq!(sha256_bytes(&seal.manifest_bytes), seal.manifest_sha256);
        let text = String::from_utf8(seal.manifest_bytes.clone()).unwrap();
        assert!(text.ends_with('\n'));
        assert!(!text.contains("manifest_self_sha256"));
        assert!(!text.contains(seal.manifest_sha256.as_str()));
        assert_eq!(seal.inventory.len(), 2);

        let verified = verify_input_seal(&output, &seal.manifest_sha256).unwrap();
        assert_eq!(verified.files.len(), 2);
        assert_eq!(verified.manifest_sha256, seal.manifest_sha256);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publication_guard_failure_is_called_once_and_cleans_only_staging() {
        let root = make_parent("guard-failure");
        let output = root.join("sealed");
        let staging = staging_path(&output).unwrap();
        let calls = AtomicUsize::new(0);
        let files = BTreeMap::from([
            ("recording_inventory.json".to_owned(), b"{}\n".to_vec()),
            ("session_plan.json".to_owned(), b"{}\n".to_vec()),
        ]);

        let error = write_sealed_artifact_with_guard(
            &output,
            files,
            json!({"terminal_status": "INPUT_SEALED"}),
            INPUT_SEAL_ALLOWLIST,
            || -> Result<(), Box<dyn Error>> {
                assert!(staging.join(MANIFEST_FILE).is_file());
                assert!(!output.exists());
                calls.fetch_add(1, Ordering::SeqCst);
                Err("publication guard rejected the run".into())
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("publication guard rejected"));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(!output.exists());
        assert!(!staging.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publication_guard_runs_once_after_staging_verification_on_success() {
        let root = make_parent("guard-success");
        let output = root.join("sealed");
        let staging = staging_path(&output).unwrap();
        let calls = AtomicUsize::new(0);
        let files = BTreeMap::from([
            ("recording_inventory.json".to_owned(), b"{}\n".to_vec()),
            ("session_plan.json".to_owned(), b"{}\n".to_vec()),
        ]);

        let seal = write_sealed_artifact_with_guard(
            &output,
            files,
            json!({"terminal_status": "INPUT_SEALED"}),
            INPUT_SEAL_ALLOWLIST,
            || -> Result<(), Box<dyn Error>> {
                assert!(staging.join(MANIFEST_FILE).is_file());
                assert!(!output.exists());
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(output.join(MANIFEST_FILE).is_file());
        assert!(!staging.exists());
        verify_input_seal(&output, &seal.manifest_sha256).unwrap();
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn destination_appearing_after_guard_is_never_deleted() {
        let root = make_parent("guard-destination-race");
        let output = root.join("sealed");
        let staging = staging_path(&output).unwrap();
        let files = BTreeMap::from([
            ("recording_inventory.json".to_owned(), b"{}\n".to_vec()),
            ("session_plan.json".to_owned(), b"{}\n".to_vec()),
        ]);

        let error = write_sealed_artifact_with_guard(
            &output,
            files,
            json!({"terminal_status": "INPUT_SEALED"}),
            INPUT_SEAL_ALLOWLIST,
            || -> Result<(), Box<dyn Error>> {
                std::fs::create_dir(&output)?;
                Ok(())
            },
        )
        .unwrap_err()
        .to_string();

        assert!(error.contains("output path appeared"));
        assert!(output.is_dir());
        assert!(!staging.exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn writer_rejects_inexact_allowlist_before_creating_staging() {
        let root = make_parent("allowlist");
        let output = root.join("sealed");
        let files = BTreeMap::from([("recording_inventory.json".to_owned(), Vec::new())]);
        assert!(write_sealed_artifact(&output, files, json!({}), INPUT_SEAL_ALLOWLIST,).is_err());
        assert!(!output.exists());
        assert!(!staging_path(&output).unwrap().exists());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_seal_is_checked_before_json_parse() {
        let root = make_parent("seal-before-json");
        write_new_sync(&root.join(MANIFEST_FILE), b"not json").unwrap();
        let error = verify_sealed_artifact(&root, &"0".repeat(64), &[MANIFEST_FILE])
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not match external seal"));
        assert!(!error.contains("JSON"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn input_invalid_cannot_enter_analyze() {
        let root = make_parent("input-invalid");
        let output = root.join("sealed");
        let files = BTreeMap::from([
            ("recording_inventory.json".to_owned(), b"{}\n".to_vec()),
            ("session_plan.json".to_owned(), b"{}\n".to_vec()),
        ]);
        let seal = write_sealed_artifact(
            &output,
            files,
            json!({"terminal_status": "INPUT_INVALID"}),
            INPUT_SEAL_ALLOWLIST,
        )
        .unwrap();
        let error = verify_input_seal(&output, &seal.manifest_sha256)
            .unwrap_err()
            .to_string();
        assert!(error.contains("terminal_status"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn verifier_rejects_extra_entries_and_member_tampering() {
        let root = make_parent("tamper");
        let output = root.join("sealed");
        let files = BTreeMap::from([
            ("recording_inventory.json".to_owned(), b"{}\n".to_vec()),
            ("session_plan.json".to_owned(), b"{}\n".to_vec()),
        ]);
        let seal = write_sealed_artifact(
            &output,
            files,
            json!({"terminal_status": "INPUT_SEALED"}),
            INPUT_SEAL_ALLOWLIST,
        )
        .unwrap();
        std::fs::write(output.join("extra.txt"), b"extra").unwrap();
        assert!(verify_input_seal(&output, &seal.manifest_sha256).is_err());
        std::fs::remove_file(output.join("extra.txt")).unwrap();
        std::fs::write(output.join("session_plan.json"), b"changed").unwrap();
        assert!(verify_input_seal(&output, &seal.manifest_sha256).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn manifest_may_not_embed_a_self_hash() {
        for key in ["manifest_self_sha256", "manifest_sha256", "file_inventory"] {
            let root = make_parent(&format!("reserved-{key}"));
            let output = root.join("sealed");
            let files = BTreeMap::from([
                ("recording_inventory.json".to_owned(), Vec::new()),
                ("session_plan.json".to_owned(), Vec::new()),
            ]);
            let mut object = serde_json::Map::new();
            object.insert(key.to_owned(), Value::Null);
            assert!(write_sealed_artifact(
                &output,
                files,
                Value::Object(object),
                INPUT_SEAL_ALLOWLIST,
            )
            .is_err());
            std::fs::remove_dir_all(root).unwrap();
        }
    }

    #[test]
    fn new_output_rejects_existing_output_and_staging() {
        let root = make_parent("new-output");
        let output = root.join("result");
        assert!(validate_new_output_destination(&output).is_ok());
        std::fs::create_dir(staging_path(&output).unwrap()).unwrap();
        assert!(validate_new_output_destination(&output).is_err());
        std::fs::remove_dir_all(staging_path(&output).unwrap()).unwrap();
        std::fs::create_dir(&output).unwrap();
        assert!(validate_new_output_destination(&output).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn file_hash_and_generic_recheck_helpers_are_exact() {
        let root = make_parent("hash");
        let path = root.join("value.bin");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        let before = file_identity(&path).unwrap();
        assert_eq!(before.length, 3);
        assert!(require_unchanged("tree", &before, &before).is_ok());
        std::fs::write(&path, b"abd").unwrap();
        let after = file_identity(&path).unwrap();
        assert!(require_unchanged("tree", &before, &after).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn analysis_verifier_selects_status_specific_allowlist() {
        let root = make_parent("analysis-status");
        let output = root.join("reduced");
        let files = BTreeMap::from([
            ("diagnostic.json".to_owned(), b"{}\n".to_vec()),
            ("interpretation.txt".to_owned(), b"inconclusive\n".to_vec()),
        ]);
        let seal = write_sealed_artifact(
            &output,
            files,
            json!({
                "terminal_status": "INCONCLUSIVE_ARTIFACT",
                "high_open_annotation": null
            }),
            ANALYSIS_REDUCED_ALLOWLIST,
        )
        .unwrap();
        assert!(verify_analysis_artifact(&output, &seal.manifest_sha256).is_ok());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    #[ignore = "explicit local verification of the published Phase 1.3 sealed pair"]
    fn published_phase13_pair_matches_the_preregistered_raw_seals() {
        let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
        let verified = verify_phase13_artifacts(
            &repository.join("research-output/phase13-decision-243ab4d"),
            &repository.join("research-output/phase13-confirmation-243ab4d"),
        )
        .unwrap();
        assert_eq!(
            verified.decision.manifest_sha256,
            PHASE13_DECISION_MANIFEST_SHA256
        );
        assert_eq!(
            verified.confirmation.manifest_sha256,
            PHASE13_CONFIRMATION_MANIFEST_SHA256
        );
    }
}
