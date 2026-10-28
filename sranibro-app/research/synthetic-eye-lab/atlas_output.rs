//! Deterministic artifact and provenance boundary for the renderer-only atlas.

use std::error::Error;
use std::path::Path;
use std::process::Command;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::atlas::{
    PreparedAtlas, CANDIDATE_RECORD_LEN, CANDIDATE_STREAM_HEADER_LEN, PREREGISTRATION_COMMIT,
    VERSION,
};

const CANDIDATE_STREAM_FILE: &str = "candidate_stream.bin";
const APERTURE_SUMMARIES_FILE: &str = "aperture_summaries.json";
const PAIR_SUMMARIES_FILE: &str = "pair_summaries.json";
const CANONICAL_CHECKS_FILE: &str = "canonical_checks.json";
const MANIFEST_FILE: &str = "manifest.json";
const MANIFEST_TEMP_FILE: &str = ".manifest.json.tmp";
const EXPECTED_CANDIDATE_COUNT: usize = 1_125_876;
const EXPECTED_PAIR_COUNT: usize = 1_683;
const RENDERER_VERSION: &str = "synthetic-eye-renderer-100x100-4x-v1";
pub const BUILD_PROFILE: &str = env!("SRANIBRO_BUILD_PROFILE");
pub const BUILD_COMMIT: &str = env!("SRANIBRO_BUILD_COMMIT");

const EMBEDDED_IMPLEMENTATION_SOURCES: [(&str, &[u8]); 8] = [
    ("Cargo.toml", include_bytes!("../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../Cargo.lock")),
    ("build.rs", include_bytes!("../../build.rs")),
    (
        "research/synthetic-eye-lab/PHASE1_3A_PREREG.md",
        include_bytes!("PHASE1_3A_PREREG.md"),
    ),
    (
        "research/synthetic-eye-lab/atlas.rs",
        include_bytes!("atlas.rs"),
    ),
    (
        "research/synthetic-eye-lab/atlas_main.rs",
        include_bytes!("atlas_main.rs"),
    ),
    (
        "research/synthetic-eye-lab/atlas_output.rs",
        include_bytes!("atlas_output.rs"),
    ),
    (
        "research/synthetic-eye-lab/renderer.rs",
        include_bytes!("renderer.rs"),
    ),
];

const INTERPRETATION_LIMIT: &str = "finite synthetic renderer candidate sets only; no continuous-domain reachability, renderer-realism, EyeNet-sensitivity, or real-camera-coverage claim";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepositoryState {
    pub commit: String,
    pub dirty: bool,
    pub implementation_source_sha256: String,
}

#[derive(Debug, Serialize)]
struct ArtifactHashes {
    candidate_stream_bin: String,
    aperture_summaries_json: String,
    pair_summaries_json: String,
    canonical_checks_json: String,
}

#[derive(Debug, Serialize)]
struct AtlasManifest<'a> {
    tool: &'static str,
    atlas_version: &'static str,
    renderer_version: &'static str,
    renderer_source_sha256: String,
    preregistration_commit: &'static str,
    repository_commit: &'a str,
    build_repository_commit: &'static str,
    implementation_source_sha256: &'a str,
    repository_dirty: bool,
    build_profile: &'static str,
    debug_assertions: bool,
    preparation_repetitions: usize,
    renderer_preparation_status: &'static str,
    candidate_count: usize,
    candidate_record_bytes: usize,
    candidate_stream_bytes: usize,
    pair_count: usize,
    candidate_stream_sha256_repetitions: &'a [String; 2],
    pair_summaries_sha256_repetitions: &'a [String; 2],
    artifact_sha256: ArtifactHashes,
    sanitized_cli: [&'static str; 2],
    model_loaded: bool,
    model_identity_sha256: Option<&'static str>,
    phase0_evaluated: bool,
    real_recordings_loaded: bool,
    interpretation_limit: &'static str,
}

/// Reject a destination that could mix this run with an earlier or partial run.
/// This check is intentionally repeated immediately before writing.
pub fn ensure_output_empty(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    if out_dir.exists() {
        if !out_dir.is_dir() {
            return Err(format!("output path is not a directory: {}", out_dir.display()).into());
        }
        if std::fs::read_dir(out_dir)?.next().is_some() {
            return Err(format!(
                "output directory is not empty: {} (choose a new directory)",
                out_dir.display()
            )
            .into());
        }
    }
    Ok(())
}

/// Resolve the exact source commit and dirty state without importing the model runner.
pub fn repository_state(repo: &Path) -> Result<RepositoryState, Box<dyn Error>> {
    let commit = git(repo, &["rev-parse", "--verify", "HEAD"])?;
    if !matches!(commit.len(), 40 | 64) || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("Git returned an invalid HEAD identity: {commit:?}").into());
    }
    let status = git(repo, &["status", "--porcelain=v1", "--untracked-files=all"])?;
    Ok(RepositoryState {
        commit,
        dirty: !status.is_empty(),
        implementation_source_sha256: checkout_implementation_source_sha256(repo)?,
    })
}

/// Bind the executable's compiled implementation bytes to the clean checkout that
/// will be named in the manifest. This rejects stale and dirty-built executables.
pub fn verify_compiled_identity(repository: &RepositoryState) -> Result<(), Box<dyn Error>> {
    if BUILD_COMMIT != repository.commit {
        return Err(format!(
            "executable was built from commit {BUILD_COMMIT}, but the checkout is {}",
            repository.commit
        )
        .into());
    }
    let compiled = compiled_implementation_source_sha256();
    if compiled != repository.implementation_source_sha256 {
        return Err(format!(
            "compiled implementation source {compiled} does not match checkout source {}",
            repository.implementation_source_sha256
        )
        .into());
    }
    Ok(())
}

/// Ensure that the frozen protocol is a real ancestor of the implementation run.
pub fn verify_preregistration(repo: &Path) -> Result<(), Box<dyn Error>> {
    let status = Command::new("git")
        .args([
            "merge-base",
            "--is-ancestor",
            PREREGISTRATION_COMMIT,
            "HEAD",
        ])
        .current_dir(repo)
        .status()?;
    if !status.success() {
        return Err(format!(
            "frozen preregistration {PREREGISTRATION_COMMIT} is not an ancestor of HEAD"
        )
        .into());
    }
    Ok(())
}

/// Write deterministic atlas artifacts. `manifest.json` is the completion marker and
/// is deliberately written last. No path or wall-clock value enters its bytes.
pub fn write_atlas(
    out_dir: &Path,
    repo_dir: &Path,
    expected_repository: &RepositoryState,
    prepared: &PreparedAtlas,
) -> Result<(), Box<dyn Error>> {
    ensure_output_empty(out_dir)?;
    if expected_repository.dirty {
        return Err("recorded atlas run requires a clean Git worktree".into());
    }
    if BUILD_PROFILE != "release" || cfg!(debug_assertions) {
        return Err(
            "recorded atlas runs require Cargo profile release with debug assertions disabled"
                .into(),
        );
    }
    let repository = repository_state(repo_dir)?;
    if repository != *expected_repository || repository.dirty {
        return Err("Git worktree changed immediately before artifact writing".into());
    }
    verify_compiled_identity(&repository)?;
    validate_prepared(prepared)?;

    std::fs::create_dir_all(out_dir)?;
    write_new(
        &out_dir.join(CANDIDATE_STREAM_FILE),
        &prepared.candidate_stream,
    )?;
    write_new(
        &out_dir.join(APERTURE_SUMMARIES_FILE),
        &prepared.aperture_summaries_json,
    )?;
    write_new(
        &out_dir.join(PAIR_SUMMARIES_FILE),
        &prepared.pair_summaries_json,
    )?;
    write_new(
        &out_dir.join(CANONICAL_CHECKS_FILE),
        &prepared.canonical_checks_json,
    )?;

    let artifact_sha256 = ArtifactHashes {
        candidate_stream_bin: sha256_hex(&prepared.candidate_stream),
        aperture_summaries_json: sha256_hex(&prepared.aperture_summaries_json),
        pair_summaries_json: sha256_hex(&prepared.pair_summaries_json),
        canonical_checks_json: sha256_hex(&prepared.canonical_checks_json),
    };
    let manifest = AtlasManifest {
        tool: "SRanibro renderer moment-feasibility atlas",
        atlas_version: VERSION,
        renderer_version: RENDERER_VERSION,
        renderer_source_sha256: sha256_hex(include_bytes!("renderer.rs")),
        preregistration_commit: PREREGISTRATION_COMMIT,
        repository_commit: &repository.commit,
        build_repository_commit: BUILD_COMMIT,
        implementation_source_sha256: &repository.implementation_source_sha256,
        repository_dirty: false,
        build_profile: BUILD_PROFILE,
        debug_assertions: cfg!(debug_assertions),
        preparation_repetitions: 2,
        renderer_preparation_status: "complete",
        candidate_count: prepared.candidate_count,
        candidate_record_bytes: CANDIDATE_RECORD_LEN,
        candidate_stream_bytes: prepared.candidate_stream.len(),
        pair_count: prepared.pair_count,
        candidate_stream_sha256_repetitions: &prepared.candidate_stream_sha256_repetitions,
        pair_summaries_sha256_repetitions: &prepared.pair_summaries_sha256_repetitions,
        artifact_sha256,
        sanitized_cli: ["--out", "<redacted-output-path>"],
        model_loaded: false,
        model_identity_sha256: None,
        phase0_evaluated: false,
        real_recordings_loaded: false,
        interpretation_limit: INTERPRETATION_LIMIT,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let final_repository = repository_state(repo_dir)?;
    if final_repository != repository || final_repository.dirty {
        return Err(
            "Git worktree changed before manifest sealing; data files remain unsealed".into(),
        );
    }
    verify_compiled_identity(&final_repository)?;
    publish_manifest(out_dir, &manifest_bytes)?;
    Ok(())
}

fn validate_prepared(prepared: &PreparedAtlas) -> Result<(), Box<dyn Error>> {
    if prepared.candidate_count != EXPECTED_CANDIDATE_COUNT {
        return Err(format!(
            "atlas returned {} candidates, expected {EXPECTED_CANDIDATE_COUNT}",
            prepared.candidate_count
        )
        .into());
    }
    if prepared.pair_count != EXPECTED_PAIR_COUNT {
        return Err(format!(
            "atlas returned {} pair summaries, expected {EXPECTED_PAIR_COUNT}",
            prepared.pair_count
        )
        .into());
    }
    let payload_bytes = EXPECTED_CANDIDATE_COUNT
        .checked_mul(CANDIDATE_RECORD_LEN)
        .ok_or("candidate stream size overflow")?;
    let expected_stream_bytes = CANDIDATE_STREAM_HEADER_LEN
        .checked_add(payload_bytes)
        .ok_or("candidate stream size overflow")?;
    if prepared.candidate_stream.len() != expected_stream_bytes {
        return Err(format!(
            "candidate stream has {} bytes; expected exactly {expected_stream_bytes}",
            prepared.candidate_stream.len(),
        )
        .into());
    }
    let candidate_hash = sha256_hex(&prepared.candidate_stream);
    if prepared.candidate_stream_sha256_repetitions[0]
        != prepared.candidate_stream_sha256_repetitions[1]
        || prepared.candidate_stream_sha256_repetitions[0] != candidate_hash
    {
        return Err("candidate stream did not match both preparation hashes".into());
    }
    let pair_hash = sha256_hex(&prepared.pair_summaries_json);
    if prepared.pair_summaries_sha256_repetitions[0]
        != prepared.pair_summaries_sha256_repetitions[1]
        || prepared.pair_summaries_sha256_repetitions[0] != pair_hash
    {
        return Err("pair summary bytes did not match both preparation hashes".into());
    }
    for (name, bytes) in [
        (APERTURE_SUMMARIES_FILE, &prepared.aperture_summaries_json),
        (PAIR_SUMMARIES_FILE, &prepared.pair_summaries_json),
        (CANONICAL_CHECKS_FILE, &prepared.canonical_checks_json),
    ] {
        let _: serde_json::Value = serde_json::from_slice(bytes)
            .map_err(|error| format!("{name} is not valid JSON: {error}"))?;
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    use std::io::Write;

    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options.open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn publish_manifest(out_dir: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    publish_manifest_with(out_dir, bytes, || Ok(()))
}

fn publish_manifest_with(
    out_dir: &Path,
    bytes: &[u8],
    before_publish: impl FnOnce() -> Result<(), Box<dyn Error>>,
) -> Result<(), Box<dyn Error>> {
    let temporary = out_dir.join(MANIFEST_TEMP_FILE);
    let final_path = out_dir.join(MANIFEST_FILE);
    write_new(&temporary, bytes)?;
    before_publish()?;
    if final_path.try_exists()? {
        return Err("manifest completion marker already exists".into());
    }
    std::fs::rename(&temporary, &final_path)?;
    sync_directory(out_dir)?;
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), Box<dyn Error>> {
    std::fs::File::open(path)?.sync_all()?;
    Ok(())
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), Box<dyn Error>> {
    Ok(())
}

fn git(repo: &Path, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new("git").args(args).current_dir(repo).output()?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed with status {}: {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )
        .into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_string())
}

fn compiled_implementation_source_sha256() -> String {
    implementation_source_sha256(
        EMBEDDED_IMPLEMENTATION_SOURCES
            .iter()
            .map(|(path, bytes)| (*path, *bytes)),
    )
}

fn checkout_implementation_source_sha256(repo: &Path) -> Result<String, Box<dyn Error>> {
    let mut sources = Vec::with_capacity(EMBEDDED_IMPLEMENTATION_SOURCES.len());
    for (path, _) in EMBEDDED_IMPLEMENTATION_SOURCES {
        sources.push((path, std::fs::read(repo.join(path))?));
    }
    Ok(implementation_source_sha256(
        sources
            .iter()
            .map(|(path, bytes)| (*path, bytes.as_slice())),
    ))
}

fn implementation_source_sha256<'a>(
    sources: impl IntoIterator<Item = (&'a str, &'a [u8])>,
) -> String {
    let mut hasher = Sha256::new();
    for (path, bytes) in sources {
        hasher.update((path.len() as u64).to_le_bytes());
        hasher.update(path.as_bytes());
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }
    digest_hex(hasher.finalize())
}

fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(Sha256::digest(bytes))
}

fn digest_hex(digest: impl AsRef<[u8]>) -> String {
    use std::fmt::Write;

    let digest = digest.as_ref();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "sranibro-atlas-{name}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    #[test]
    fn nonempty_output_directory_is_rejected() {
        let path = unique_temp("nonempty");
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("existing.txt"), b"existing").unwrap();
        let error = ensure_output_empty(&path).unwrap_err().to_string();
        assert!(error.contains("not empty"));
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn prepared_stream_requires_the_exact_header_and_payload_length() {
        let bytes =
            vec![
                0;
                CANDIDATE_STREAM_HEADER_LEN + EXPECTED_CANDIDATE_COUNT * CANDIDATE_RECORD_LEN - 1
            ];
        let prepared = PreparedAtlas {
            candidate_stream_sha256_repetitions: [sha256_hex(&bytes), sha256_hex(&bytes)],
            candidate_stream: bytes,
            aperture_summaries_json: b"[]".to_vec(),
            pair_summaries_json: b"[]".to_vec(),
            canonical_checks_json: b"[]".to_vec(),
            pair_summaries_sha256_repetitions: [sha256_hex(b"[]"), sha256_hex(b"[]")],
            candidate_count: EXPECTED_CANDIDATE_COUNT,
            pair_count: EXPECTED_PAIR_COUNT,
        };
        assert!(validate_prepared(&prepared)
            .unwrap_err()
            .to_string()
            .contains("expected exactly"));
    }

    #[test]
    fn dirty_repository_is_not_accepted_by_writer() {
        let prepared = PreparedAtlas {
            candidate_stream: vec![1],
            aperture_summaries_json: b"[]".to_vec(),
            pair_summaries_json: b"[]".to_vec(),
            canonical_checks_json: b"[]".to_vec(),
            candidate_stream_sha256_repetitions: [sha256_hex(&[1]), sha256_hex(&[1])],
            pair_summaries_sha256_repetitions: [sha256_hex(b"[]"), sha256_hex(b"[]")],
            candidate_count: EXPECTED_CANDIDATE_COUNT,
            pair_count: EXPECTED_PAIR_COUNT,
        };
        let repository = RepositoryState {
            commit: "0".repeat(40),
            dirty: true,
            implementation_source_sha256: "0".repeat(64),
        };
        let path =
            std::env::temp_dir().join(format!("sranibro-atlas-dirty-test-{}", std::process::id()));
        let error = write_atlas(&path, Path::new("."), &repository, &prepared)
            .unwrap_err()
            .to_string();
        assert!(error.contains("clean Git worktree"));
        assert!(!path.exists());
    }

    #[test]
    fn manifest_schema_pins_model_and_recording_isolation() {
        let manifest = AtlasManifest {
            tool: "test",
            atlas_version: VERSION,
            renderer_version: RENDERER_VERSION,
            renderer_source_sha256: sha256_hex(include_bytes!("renderer.rs")),
            preregistration_commit: PREREGISTRATION_COMMIT,
            repository_commit: "0123456789012345678901234567890123456789",
            build_repository_commit: BUILD_COMMIT,
            implementation_source_sha256:
                "0123456789012345678901234567890123456789012345678901234567890123",
            repository_dirty: false,
            build_profile: BUILD_PROFILE,
            debug_assertions: cfg!(debug_assertions),
            preparation_repetitions: 2,
            renderer_preparation_status: "complete",
            candidate_count: EXPECTED_CANDIDATE_COUNT,
            candidate_record_bytes: CANDIDATE_RECORD_LEN,
            candidate_stream_bytes: CANDIDATE_RECORD_LEN * EXPECTED_CANDIDATE_COUNT
                + CANDIDATE_STREAM_HEADER_LEN,
            pair_count: EXPECTED_PAIR_COUNT,
            candidate_stream_sha256_repetitions: &["a".into(), "a".into()],
            pair_summaries_sha256_repetitions: &["b".into(), "b".into()],
            artifact_sha256: ArtifactHashes {
                candidate_stream_bin: "a".into(),
                aperture_summaries_json: "b".into(),
                pair_summaries_json: "c".into(),
                canonical_checks_json: "d".into(),
            },
            sanitized_cli: ["--out", "<redacted-output-path>"],
            model_loaded: false,
            model_identity_sha256: None,
            phase0_evaluated: false,
            real_recordings_loaded: false,
            interpretation_limit: INTERPRETATION_LIMIT,
        };
        let value = serde_json::to_value(manifest).unwrap();
        assert_eq!(value["model_loaded"], false);
        assert!(value["model_identity_sha256"].is_null());
        assert_eq!(value["phase0_evaluated"], false);
        assert_eq!(value["real_recordings_loaded"], false);
        assert_eq!(value["build_profile"], BUILD_PROFILE);
        assert_eq!(value["debug_assertions"], cfg!(debug_assertions));
        assert!(value.get("generated_unix_s").is_none());
        assert!(value.to_string().find("model.params").is_none());
    }

    #[test]
    fn compiled_identity_rejects_stale_commit_and_source() {
        let repo = Path::new(env!("CARGO_MANIFEST_DIR"));
        let current = repository_state(repo).unwrap();
        verify_compiled_identity(&current).unwrap();

        let mut stale_commit = current.clone();
        stale_commit.commit = "0".repeat(current.commit.len());
        assert!(verify_compiled_identity(&stale_commit)
            .unwrap_err()
            .to_string()
            .contains("built from commit"));

        let mut stale_source = current;
        stale_source.implementation_source_sha256 = "0".repeat(64);
        assert!(verify_compiled_identity(&stale_source)
            .unwrap_err()
            .to_string()
            .contains("does not match checkout source"));
    }

    #[test]
    fn manifest_is_not_visible_when_prepublication_fails() {
        let path = unique_temp("manifest-publish-failure");
        std::fs::create_dir_all(&path).unwrap();
        let error = publish_manifest_with(&path, b"complete manifest", || {
            Err("injected prepublication failure".into())
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("injected"));
        assert!(!path.join(MANIFEST_FILE).exists());
        assert_eq!(
            std::fs::read(path.join(MANIFEST_TEMP_FILE)).unwrap(),
            b"complete manifest"
        );
        std::fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn manifest_publish_atomically_replaces_the_synced_temporary_name() {
        let path = unique_temp("manifest-publish-success");
        std::fs::create_dir_all(&path).unwrap();
        publish_manifest(&path, b"complete manifest").unwrap();
        assert_eq!(
            std::fs::read(path.join(MANIFEST_FILE)).unwrap(),
            b"complete manifest"
        );
        assert!(!path.join(MANIFEST_TEMP_FILE).exists());
        std::fs::remove_dir_all(path).unwrap();
    }
}
