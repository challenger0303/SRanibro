//! Provenance, sealed-input validation, and atomic artifacts for Phase 1.3.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub const BUILD_PROFILE: &str = env!("SRANIBRO_BUILD_PROFILE");
pub const BUILD_COMMIT: &str = env!("SRANIBRO_BUILD_COMMIT");
pub const PREREGISTRATION_COMMIT: &str = "f7cf3686d8025631a8e83442a69c43858402cd6c";
pub const AMENDMENT_COMMIT: &str = "d32c7349c6bf7fbfb8398dd804e4158c4a052db5";
pub const AMENDMENT2_COMMIT: &str = "3fa96e76d7e02f3051644a72da219da3b9b9eba1";

pub const ATLAS_REPOSITORY_COMMIT: &str = "49e13f0eb2b78f84a387de8a46e7309257c9304e";
pub const ATLAS_PREREGISTRATION_COMMIT: &str = "c92dbb2411c13d2f055ee7c1a67ee2b956d1e1a1";
pub const ATLAS_MANIFEST_SHA256: &str =
    "13d67e09915faa22f89f482ae3960703d1b610e519e5868ddcf884df0542e347";
pub const ATLAS_CANDIDATE_SHA256: &str =
    "4eb662658c7997d37d53acc6daa8af9045e9ae2386784e39f3d37b4145313139";
pub const ATLAS_APERTURE_SHA256: &str =
    "ed8a7c468c301124bd26de9c56baa6755c9b4d82b728e4cfcf39e2bc820ad2f1";
pub const ATLAS_PAIR_SHA256: &str =
    "d7ce595f354b8dacd5f19c04d1a0f04f24aadad585fc0c580de29f334efd1480";
pub const ATLAS_CANONICAL_SHA256: &str =
    "5c67d57e6321f00f7e1e6e4e8a98de5d0788670e3afb176421ecc4afb135453d";
pub const ATLAS_RENDERER_SOURCE_SHA256: &str =
    "9fdeca8c45fa6c56d7721e0a2a2d10e1b9c19799ff528ccc5d059f7031c056bd";
pub const ATLAS_IMPLEMENTATION_SOURCE_SHA256: &str =
    "d3cfd3a4669d30663bcbd4072e2a8584f979e9e5e5b6c789864f424dc1f1bdc8";

const CANDIDATE_FILE: &str = "candidate_stream.bin";
const APERTURE_FILE: &str = "aperture_summaries.json";
const PAIR_FILE: &str = "pair_summaries.json";
const CANONICAL_FILE: &str = "canonical_checks.json";
const MANIFEST_FILE: &str = "manifest.json";
const EXPECTED_CANDIDATES: usize = 1_125_876;
const EXPECTED_PAIRS: u64 = 1_683;
const CANDIDATE_HEADER_LEN: usize = 24;
const CANDIDATE_RECORD_LEN: usize = 28;

const IMPLEMENTATION_SOURCES: [(&str, &[u8]); 16] = [
    ("Cargo.toml", include_bytes!("../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ("build.rs", include_bytes!("../../build.rs")),
    (
        "research/synthetic-eye-lab/PHASE1_3_PREREG.md",
        include_bytes!("PHASE1_3_PREREG.md"),
    ),
    (
        "research/synthetic-eye-lab/PHASE1_3_AMENDMENT1.md",
        include_bytes!("PHASE1_3_AMENDMENT1.md"),
    ),
    (
        "research/synthetic-eye-lab/PHASE1_3_AMENDMENT2.md",
        include_bytes!("PHASE1_3_AMENDMENT2.md"),
    ),
    (
        "research/synthetic-eye-lab/phase13.rs",
        include_bytes!("phase13.rs"),
    ),
    (
        "research/synthetic-eye-lab/phase13_main.rs",
        include_bytes!("phase13_main.rs"),
    ),
    (
        "research/synthetic-eye-lab/phase13_output.rs",
        include_bytes!("phase13_output.rs"),
    ),
    (
        "research/synthetic-eye-lab/renderer.rs",
        include_bytes!("renderer.rs"),
    ),
    (
        "research/synthetic-eye-lab/model.rs",
        include_bytes!("model.rs"),
    ),
    (
        "research/synthetic-eye-lab/moments.rs",
        include_bytes!("moments.rs"),
    ),
    ("src/lib.rs", include_bytes!("../../src/lib.rs")),
    ("src/ml/mod.rs", include_bytes!("../../src/ml/mod.rs")),
    (
        "src/ml/eye_net.rs",
        include_bytes!("../../src/ml/eye_net.rs"),
    ),
    (
        "src/ml/tvm_params.rs",
        include_bytes!("../../src/ml/tvm_params.rs"),
    ),
];

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct RepositoryState {
    pub commit: String,
    pub dirty: bool,
    pub implementation_source_sha256: String,
}

#[derive(Clone, Debug)]
pub struct ValidatedAtlas {
    pub candidate_stream: Vec<u8>,
    pub manifest: Value,
    pub manifest_sha256: String,
    pub candidate_sha256: String,
    pub aperture_sha256: String,
    pub pair_sha256: String,
    pub canonical_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
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
    pub files: BTreeMap<String, Vec<u8>>,
}

pub fn require_release_build() -> Result<(), Box<dyn Error>> {
    if BUILD_PROFILE != "release" || cfg!(debug_assertions) {
        return Err(format!(
            "recorded Phase 1.3 runs require release with debug assertions disabled; profile={BUILD_PROFILE}"
        )
        .into());
    }
    Ok(())
}

pub fn repository_state(repo: &Path) -> Result<RepositoryState, Box<dyn Error>> {
    let commit = git(repo, &["rev-parse", "--verify", "HEAD"])?;
    validate_hex("repository commit", &commit, &[40, 64])?;
    let status = git(repo, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    Ok(RepositoryState {
        commit,
        dirty: !status.is_empty(),
        implementation_source_sha256: checkout_source_fingerprint(repo)?,
    })
}

pub fn verify_runtime_identity(repository: &RepositoryState) -> Result<(), Box<dyn Error>> {
    if repository.dirty {
        return Err("recorded Phase 1.3 runs require a clean Git worktree".into());
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
            "compiled implementation source {compiled} differs from checkout source {}",
            repository.implementation_source_sha256
        )
        .into());
    }
    Ok(())
}

pub fn verify_preregistration(repo: &Path) -> Result<(), Box<dyn Error>> {
    for (label, commit) in [
        ("preregistration", PREREGISTRATION_COMMIT),
        ("amendment", AMENDMENT_COMMIT),
        ("amendment 2", AMENDMENT2_COMMIT),
    ] {
        let status = Command::new("git")
            .args(["merge-base", "--is-ancestor", commit, "HEAD"])
            .current_dir(repo)
            .status()?;
        if !status.success() {
            return Err(format!("frozen {label} {commit} is not an ancestor of HEAD").into());
        }
    }
    Ok(())
}

pub fn validate_atlas_dir(directory: &Path) -> Result<ValidatedAtlas, Box<dyn Error>> {
    if !directory.is_dir() {
        return Err(format!("atlas directory not found: {}", directory.display()).into());
    }
    let expected: BTreeSet<&str> = [
        CANDIDATE_FILE,
        APERTURE_FILE,
        PAIR_FILE,
        CANONICAL_FILE,
        MANIFEST_FILE,
    ]
    .into_iter()
    .collect();
    let actual: BTreeSet<String> = std::fs::read_dir(directory)?
        .map(|entry| {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "atlas contains a non-file entry",
                ));
            }
            entry.file_name().into_string().map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "non-UTF8 atlas name")
            })
        })
        .collect::<Result<_, _>>()?;
    if actual != expected.iter().map(|name| (*name).to_owned()).collect() {
        return Err(format!("atlas file inventory mismatch: {actual:?}").into());
    }

    let candidate_stream = std::fs::read(directory.join(CANDIDATE_FILE))?;
    let aperture = std::fs::read(directory.join(APERTURE_FILE))?;
    let pair = std::fs::read(directory.join(PAIR_FILE))?;
    let canonical = std::fs::read(directory.join(CANONICAL_FILE))?;
    let manifest_bytes = std::fs::read(directory.join(MANIFEST_FILE))?;

    let candidate_sha256 = require_sha(CANDIDATE_FILE, &candidate_stream, ATLAS_CANDIDATE_SHA256)?;
    let aperture_sha256 = require_sha(APERTURE_FILE, &aperture, ATLAS_APERTURE_SHA256)?;
    let pair_sha256 = require_sha(PAIR_FILE, &pair, ATLAS_PAIR_SHA256)?;
    let canonical_sha256 = require_sha(CANONICAL_FILE, &canonical, ATLAS_CANONICAL_SHA256)?;
    let manifest_sha256 = require_sha(MANIFEST_FILE, &manifest_bytes, ATLAS_MANIFEST_SHA256)?;
    validate_candidate_stream_header(&candidate_stream)?;

    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    require_json_string(&manifest, "atlas_version", "synthetic-eye-moment-atlas-v1")?;
    require_json_string(
        &manifest,
        "renderer_version",
        "synthetic-eye-renderer-100x100-4x-v1",
    )?;
    require_json_string(
        &manifest,
        "renderer_source_sha256",
        ATLAS_RENDERER_SOURCE_SHA256,
    )?;
    require_json_string(
        &manifest,
        "preregistration_commit",
        ATLAS_PREREGISTRATION_COMMIT,
    )?;
    require_json_string(&manifest, "repository_commit", ATLAS_REPOSITORY_COMMIT)?;
    require_json_string(
        &manifest,
        "implementation_source_sha256",
        ATLAS_IMPLEMENTATION_SOURCE_SHA256,
    )?;
    require_json_string(&manifest, "build_profile", "release")?;
    require_json_u64(&manifest, "candidate_count", EXPECTED_CANDIDATES as u64)?;
    require_json_u64(
        &manifest,
        "candidate_record_bytes",
        CANDIDATE_RECORD_LEN as u64,
    )?;
    require_json_u64(&manifest, "pair_count", EXPECTED_PAIRS)?;
    require_json_bool(&manifest, "repository_dirty", false)?;
    require_json_bool(&manifest, "debug_assertions", false)?;
    require_json_bool(&manifest, "model_loaded", false)?;
    require_json_bool(&manifest, "phase0_evaluated", false)?;
    require_json_bool(&manifest, "real_recordings_loaded", false)?;
    if manifest.get("model_identity_sha256") != Some(&Value::Null) {
        return Err("atlas model_identity_sha256 must be null".into());
    }
    let artifact = manifest
        .get("artifact_sha256")
        .and_then(Value::as_object)
        .ok_or("atlas artifact_sha256 is missing")?;
    for (key, expected) in [
        ("candidate_stream_bin", ATLAS_CANDIDATE_SHA256),
        ("aperture_summaries_json", ATLAS_APERTURE_SHA256),
        ("pair_summaries_json", ATLAS_PAIR_SHA256),
        ("canonical_checks_json", ATLAS_CANONICAL_SHA256),
    ] {
        if artifact.get(key).and_then(Value::as_str) != Some(expected) {
            return Err(format!("atlas manifest artifact hash mismatch: {key}").into());
        }
    }

    Ok(ValidatedAtlas {
        candidate_stream,
        manifest,
        manifest_sha256,
        candidate_sha256,
        aperture_sha256,
        pair_sha256,
        canonical_sha256,
    })
}

pub fn write_stage_artifact(
    output: &Path,
    repo: &Path,
    expected_repository: &RepositoryState,
    manifest_fields: Value,
    mut files: Vec<(String, Vec<u8>)>,
) -> Result<ArtifactSeal, Box<dyn Error>> {
    require_release_build()?;
    reject_existing_output(output)?;
    let repository = repository_state(repo)?;
    if repository != *expected_repository {
        return Err("repository identity changed before artifact publication".into());
    }
    verify_runtime_identity(&repository)?;

    let mut object = manifest_fields
        .as_object()
        .cloned()
        .ok_or("manifest_fields must be a JSON object")?;
    for reserved in ["file_inventory", "manifest_self_sha256"] {
        if object.contains_key(reserved) {
            return Err(format!("manifest field is reserved: {reserved}").into());
        }
    }

    files.sort_by(|left, right| left.0.cmp(&right.0));
    let mut seen = BTreeSet::new();
    for (name, _) in &files {
        validate_relative_file(name)?;
        if name == MANIFEST_FILE || !seen.insert(name.clone()) {
            return Err(format!("invalid or duplicate artifact path: {name}").into());
        }
    }

    let staging = staging_path(output)?;
    if staging.exists() {
        return Err(format!(
            "partial staging directory already exists: {}",
            staging.display()
        )
        .into());
    }
    std::fs::create_dir(&staging)?;

    let result = (|| -> Result<ArtifactSeal, Box<dyn Error>> {
        for (name, bytes) in &files {
            let path = staging.join(name);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            write_new_sync(&path, bytes)?;
        }
        let expected_files: BTreeSet<String> = files.iter().map(|(name, _)| name.clone()).collect();
        let inventory = inventory_from_staging(&staging, &expected_files)?;
        for ((name, expected_bytes), entry) in files.iter().zip(&inventory) {
            if entry.path != *name
                || entry.length != expected_bytes.len() as u64
                || entry.sha256 != sha256_hex(expected_bytes)
            {
                return Err(format!("staged artifact bytes changed after writing: {name}").into());
            }
        }
        object.insert("file_inventory".into(), serde_json::to_value(&inventory)?);
        let mut manifest_bytes = serde_json::to_vec_pretty(&Value::Object(object))?;
        manifest_bytes.push(b'\n');
        write_new_sync(&staging.join(MANIFEST_FILE), &manifest_bytes)?;
        sync_directory(&staging)?;

        let final_repository = repository_state(repo)?;
        if final_repository != *expected_repository {
            return Err("repository identity changed during artifact publication".into());
        }
        verify_runtime_identity(&final_repository)?;
        verify_staging_tree(&staging, &inventory, &manifest_bytes)?;
        sync_directory(&staging)?;
        reject_existing_output(output)?;
        let parent = output.parent().ok_or("output has no parent")?;
        let parent_for_sync = if parent.as_os_str().is_empty() {
            Path::new(".")
        } else {
            parent
        };
        sync_directory(parent_for_sync)?;
        std::fs::rename(&staging, output)?;
        sync_directory(output)?;
        sync_directory(parent_for_sync)?;
        Ok(ArtifactSeal {
            output_dir: output.to_path_buf(),
            manifest_sha256: sha256_hex(&manifest_bytes),
            manifest_bytes,
            inventory,
        })
    })();
    result
}

pub fn verify_decision_artifact(
    directory: &Path,
    expected_manifest_sha256: &str,
) -> Result<VerifiedArtifact, Box<dyn Error>> {
    validate_lower_sha256("external decision seal", expected_manifest_sha256)?;
    if !directory.is_dir() {
        return Err(format!("decision artifact not found: {}", directory.display()).into());
    }
    let manifest_bytes = std::fs::read(directory.join(MANIFEST_FILE))?;
    let manifest_sha256 = sha256_hex(&manifest_bytes);
    if manifest_sha256 != expected_manifest_sha256 {
        return Err(format!(
            "decision manifest SHA-256 {manifest_sha256} did not match external seal {expected_manifest_sha256}"
        )
        .into());
    }
    let manifest: Value = serde_json::from_slice(&manifest_bytes)?;
    if manifest.get("stage").and_then(Value::as_str) != Some("decision") {
        return Err("artifact is not a Phase 1.3 decision".into());
    }
    let inventory: Vec<FileInventoryEntry> = serde_json::from_value(
        manifest
            .get("file_inventory")
            .cloned()
            .ok_or("decision file_inventory is missing")?,
    )?;
    let mut prior = None;
    let mut expected = BTreeSet::new();
    let mut files = BTreeMap::new();
    for entry in &inventory {
        validate_relative_file(&entry.path)?;
        if prior
            .as_deref()
            .is_some_and(|value| value >= entry.path.as_str())
        {
            return Err("decision inventory is not strictly sorted".into());
        }
        prior = Some(entry.path.clone());
        if !expected.insert(entry.path.clone()) || entry.path == MANIFEST_FILE {
            return Err("decision inventory has duplicate or reserved paths".into());
        }
        let bytes = std::fs::read(directory.join(&entry.path))?;
        if bytes.len() as u64 != entry.length || sha256_hex(&bytes) != entry.sha256 {
            return Err(format!("decision artifact file changed: {}", entry.path).into());
        }
        files.insert(entry.path.clone(), bytes);
    }
    expected.insert(MANIFEST_FILE.into());
    let actual = scan_artifact_tree(directory)?;
    require_exact_tree(&actual, &expected)?;
    Ok(VerifiedArtifact {
        directory: directory.to_path_buf(),
        manifest,
        manifest_bytes,
        manifest_sha256,
        inventory,
        files,
    })
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn compiled_source_fingerprint() -> String {
    source_fingerprint(IMPLEMENTATION_SOURCES.iter().copied())
}

fn checkout_source_fingerprint(repo: &Path) -> Result<String, Box<dyn Error>> {
    let mut owned = Vec::with_capacity(IMPLEMENTATION_SOURCES.len());
    for (name, _) in IMPLEMENTATION_SOURCES {
        owned.push((name, std::fs::read(repo.join(name))?));
    }
    Ok(source_fingerprint(
        owned.iter().map(|(name, bytes)| (*name, bytes.as_slice())),
    ))
}

fn source_fingerprint<'a>(entries: impl IntoIterator<Item = (&'a str, &'a [u8])>) -> String {
    let mut hasher = Sha256::new();
    for (name, bytes) in entries {
        hasher.update((name.len() as u64).to_le_bytes());
        hasher.update(name.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    use std::fmt::Write as _;
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_candidate_stream_header(bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let expected_len = CANDIDATE_HEADER_LEN + EXPECTED_CANDIDATES * CANDIDATE_RECORD_LEN;
    if bytes.len() != expected_len {
        return Err(format!("candidate stream length {} != {expected_len}", bytes.len()).into());
    }
    if &bytes[0..8] != b"SRATL3A1"
        || read_u32(bytes, 8)? != 1
        || read_u32(bytes, 12)? != CANDIDATE_RECORD_LEN as u32
        || read_u64(bytes, 16)? != EXPECTED_CANDIDATES as u64
    {
        return Err("candidate stream header mismatch".into());
    }
    Ok(())
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, Box<dyn Error>> {
    Ok(u32::from_le_bytes(
        bytes
            .get(offset..offset + 4)
            .ok_or("truncated u32")?
            .try_into()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, Box<dyn Error>> {
    Ok(u64::from_le_bytes(
        bytes
            .get(offset..offset + 8)
            .ok_or("truncated u64")?
            .try_into()?,
    ))
}

fn require_sha(name: &str, bytes: &[u8], expected: &str) -> Result<String, Box<dyn Error>> {
    let actual = sha256_hex(bytes);
    if actual != expected {
        return Err(format!("{name} SHA-256 {actual} != {expected}").into());
    }
    Ok(actual)
}

fn require_json_string(root: &Value, key: &str, expected: &str) -> Result<(), Box<dyn Error>> {
    if root.get(key).and_then(Value::as_str) != Some(expected) {
        return Err(format!("atlas manifest {key} mismatch").into());
    }
    Ok(())
}

fn require_json_u64(root: &Value, key: &str, expected: u64) -> Result<(), Box<dyn Error>> {
    if root.get(key).and_then(Value::as_u64) != Some(expected) {
        return Err(format!("atlas manifest {key} mismatch").into());
    }
    Ok(())
}

fn require_json_bool(root: &Value, key: &str, expected: bool) -> Result<(), Box<dyn Error>> {
    if root.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("atlas manifest {key} mismatch").into());
    }
    Ok(())
}

fn validate_hex(name: &str, value: &str, lengths: &[usize]) -> Result<(), Box<dyn Error>> {
    if !lengths.contains(&value.len()) || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("invalid {name}: {value:?}").into());
    }
    Ok(())
}

fn validate_lower_sha256(name: &str, value: &str) -> Result<(), Box<dyn Error>> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("invalid {name}: expected 64 lowercase hexadecimal characters").into());
    }
    Ok(())
}

fn reject_existing_output(output: &Path) -> Result<(), Box<dyn Error>> {
    if output.exists() {
        return Err(format!("output path already exists: {}", output.display()).into());
    }
    if output.file_name().is_none() || output.parent().is_none() {
        return Err("output must be a named path with a parent directory".into());
    }
    Ok(())
}

fn staging_path(output: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or("output file name must be UTF-8")?;
    Ok(output
        .parent()
        .ok_or("output has no parent")?
        .join(format!(".{name}.phase13-staging")))
}

fn validate_relative_file(name: &str) -> Result<(), Box<dyn Error>> {
    if name.is_empty() || name.contains('\\') {
        return Err(format!("artifact path must use nonempty forward-slash form: {name:?}").into());
    }
    let path = Path::new(name);
    if path.is_absolute()
        || path
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(format!("unsafe artifact path: {name:?}").into());
    }
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

fn inventory_from_staging(
    staging: &Path,
    expected_files: &BTreeSet<String>,
) -> Result<Vec<FileInventoryEntry>, Box<dyn Error>> {
    let tree = scan_artifact_tree(staging)?;
    require_exact_tree(&tree, expected_files)?;
    let mut inventory = Vec::with_capacity(expected_files.len());
    for name in expected_files {
        let bytes = std::fs::read(staging.join(name))?;
        inventory.push(FileInventoryEntry {
            path: name.clone(),
            length: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        });
    }
    Ok(inventory)
}

fn verify_staging_tree(
    staging: &Path,
    inventory: &[FileInventoryEntry],
    expected_manifest_bytes: &[u8],
) -> Result<(), Box<dyn Error>> {
    let mut expected_files = BTreeSet::new();
    let mut prior: Option<&str> = None;
    for entry in inventory {
        validate_relative_file(&entry.path)?;
        if prior.is_some_and(|value| value >= entry.path.as_str())
            || !expected_files.insert(entry.path.clone())
        {
            return Err("staging inventory is not strictly sorted and unique".into());
        }
        prior = Some(&entry.path);
    }
    expected_files.insert(MANIFEST_FILE.to_owned());
    let tree = scan_artifact_tree(staging)?;
    require_exact_tree(&tree, &expected_files)?;
    for entry in inventory {
        let bytes = std::fs::read(staging.join(&entry.path))?;
        if bytes.len() as u64 != entry.length || sha256_hex(&bytes) != entry.sha256 {
            return Err(format!("staging artifact file changed: {}", entry.path).into());
        }
    }
    let manifest_bytes = std::fs::read(staging.join(MANIFEST_FILE))?;
    if manifest_bytes != expected_manifest_bytes {
        return Err("staging manifest changed before atomic publication".into());
    }
    Ok(())
}

fn require_exact_tree(
    actual: &ArtifactTree,
    expected_files: &BTreeSet<String>,
) -> Result<(), Box<dyn Error>> {
    let expected_directories = required_directories(expected_files)?;
    if actual.files != *expected_files || actual.directories != expected_directories {
        return Err(format!(
            "artifact tree mismatch: files={:?}, directories={:?}",
            actual.files, actual.directories
        )
        .into());
    }
    Ok(())
}

fn required_directories(
    expected_files: &BTreeSet<String>,
) -> Result<BTreeSet<String>, Box<dyn Error>> {
    let mut directories = BTreeSet::new();
    for name in expected_files {
        validate_relative_file(name)?;
        let components: Vec<_> = Path::new(name)
            .components()
            .map(|component| match component {
                Component::Normal(value) => value
                    .to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| "artifact path component is not UTF-8".to_string()),
                _ => Err("artifact path was not normalized".to_string()),
            })
            .collect::<Result<_, _>>()?;
        for count in 1..components.len() {
            directories.insert(components[..count].join("/"));
        }
    }
    Ok(directories)
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
                return Err("artifact tree contains a link or special entry".into());
            }
        }
        Ok(())
    }

    let mut files = BTreeSet::new();
    let mut directories = BTreeSet::new();
    visit(root, root, &mut files, &mut directories)?;
    Ok(ArtifactTree { files, directories })
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::File::open(path)?.sync_all()?;
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

fn git(repo: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(args).current_dir(repo).output()?;
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

    fn unique_temp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "sranibro-phase13-output-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn source_fingerprint_has_frozen_length_encoding_and_order() {
        let first = source_fingerprint([("a", b"bc".as_slice()), ("de", b"f".as_slice())]);
        let second = source_fingerprint([("ab", b"c".as_slice()), ("d", b"ef".as_slice())]);
        assert_ne!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn implementation_fingerprint_has_the_exact_twice_amended_file_order() {
        let names: Vec<_> = IMPLEMENTATION_SOURCES
            .iter()
            .map(|(name, _)| *name)
            .collect();
        assert_eq!(
            names,
            vec![
                "Cargo.toml",
                "Cargo.lock",
                "build.rs",
                "research/synthetic-eye-lab/PHASE1_3_PREREG.md",
                "research/synthetic-eye-lab/PHASE1_3_AMENDMENT1.md",
                "research/synthetic-eye-lab/PHASE1_3_AMENDMENT2.md",
                "research/synthetic-eye-lab/phase13.rs",
                "research/synthetic-eye-lab/phase13_main.rs",
                "research/synthetic-eye-lab/phase13_output.rs",
                "research/synthetic-eye-lab/renderer.rs",
                "research/synthetic-eye-lab/model.rs",
                "research/synthetic-eye-lab/moments.rs",
                "src/lib.rs",
                "src/ml/mod.rs",
                "src/ml/eye_net.rs",
                "src/ml/tvm_params.rs",
            ]
        );
    }

    #[test]
    fn artifact_relative_paths_reject_escape_and_windows_separators() {
        assert!(validate_relative_file("renderer_plan.json").is_ok());
        assert!(validate_relative_file("nested/results.json").is_ok());
        assert!(validate_relative_file("../escape").is_err());
        assert!(validate_relative_file("nested\\results.json").is_err());
        assert!(validate_relative_file("/absolute").is_err());
    }

    #[test]
    fn external_seal_is_strict_lowercase_sha256() {
        assert!(validate_lower_sha256("seal", &"a".repeat(64)).is_ok());
        assert!(validate_lower_sha256("seal", &"A".repeat(64)).is_err());
        assert!(validate_lower_sha256("seal", &"g".repeat(64)).is_err());
        assert!(validate_lower_sha256("seal", &"a".repeat(63)).is_err());
    }

    #[test]
    fn decision_seal_is_checked_before_manifest_json_is_trusted() {
        let root = unique_temp("seal-before-parse");
        std::fs::create_dir(&root).unwrap();
        write_new_sync(&root.join(MANIFEST_FILE), b"not json").unwrap();
        let error = verify_decision_artifact(&root, &"0".repeat(64))
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not match external seal"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staged_inventory_is_reopened_and_revalidated_before_publication() {
        let root = unique_temp("inventory");
        std::fs::create_dir(&root).unwrap();
        std::fs::create_dir(root.join("nested")).unwrap();
        write_new_sync(&root.join("a.json"), b"first").unwrap();
        write_new_sync(&root.join("nested/b.bin"), b"second").unwrap();
        let expected: BTreeSet<String> = ["a.json", "nested/b.bin"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let inventory = inventory_from_staging(&root, &expected).unwrap();
        assert_eq!(inventory[0].path, "a.json");
        assert_eq!(inventory[0].length, 5);
        assert_eq!(inventory[0].sha256, sha256_hex(b"first"));
        assert_eq!(inventory[1].path, "nested/b.bin");

        let manifest = b"{}\n";
        write_new_sync(&root.join(MANIFEST_FILE), manifest).unwrap();
        verify_staging_tree(&root, &inventory, manifest).unwrap();
        std::fs::write(root.join("a.json"), b"changed").unwrap();
        assert!(verify_staging_tree(&root, &inventory, manifest).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn staged_tree_rejects_unlisted_empty_directories() {
        let root = unique_temp("extra-directory");
        std::fs::create_dir(&root).unwrap();
        write_new_sync(&root.join("result.json"), b"result").unwrap();
        let expected: BTreeSet<String> = ["result.json"].into_iter().map(str::to_owned).collect();
        std::fs::create_dir(root.join("unexpected")).unwrap();
        assert!(inventory_from_staging(&root, &expected).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_sync_is_supported_or_safely_unavailable() {
        let root = unique_temp("directory-sync");
        std::fs::create_dir(&root).unwrap();
        sync_directory(&root).unwrap();
        std::fs::remove_dir(root).unwrap();
    }

    #[test]
    fn candidate_header_contract_is_exact() {
        let mut bytes =
            vec![0u8; CANDIDATE_HEADER_LEN + EXPECTED_CANDIDATES * CANDIDATE_RECORD_LEN];
        bytes[..8].copy_from_slice(b"SRATL3A1");
        bytes[8..12].copy_from_slice(&1u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&(CANDIDATE_RECORD_LEN as u32).to_le_bytes());
        bytes[16..24].copy_from_slice(&(EXPECTED_CANDIDATES as u64).to_le_bytes());
        assert!(validate_candidate_stream_header(&bytes).is_ok());
        bytes[8] = 2;
        assert!(validate_candidate_stream_header(&bytes).is_err());
    }

    #[test]
    fn compiled_fingerprint_is_stable() {
        assert_eq!(compiled_source_fingerprint(), compiled_source_fingerprint());
        assert_eq!(compiled_source_fingerprint().len(), 64);
    }
}
