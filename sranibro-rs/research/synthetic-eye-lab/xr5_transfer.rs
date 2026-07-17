//! Preregistered Phase 1.4 XR5 real-recording transfer audit core.
//!
//! This module deliberately owns no command-line or publication code.  Its public
//! API separates privacy-safe, JSON-ready artifacts from runtime-only source paths,
//! and returns complete file payloads to the publisher.  It never writes the input
//! tree or the application configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsStr;
use std::fmt::{self, Write as _};
use std::fs;
use std::io::Cursor;
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sranibro_rs::core::types::{DespeckleParams, FlattenParams, MlGeometry};
use sranibro_rs::ml::{eye_net::EyeNet, preprocess, tvm_params};

pub const PREREGISTRATION_FILE: &str = "research/synthetic-eye-lab/PHASE1_4_XR5_TRANSFER_PREREG.md";
pub const EXPECTED_MODEL_LEN: u64 = 51_423_934;
pub const EXPECTED_MODEL_SHA256: &str =
    "bac8013e0423068924f190a1de44afd5e1dd0c7c10d1d394926e46fc1b075ded";
pub const EXPECTED_PHASE13_DECISION_SEAL: &str =
    "17291d72ab05034ea5c047225c6868ea714c6dbbebe2122beb41519dc02dab48";
pub const EXPECTED_PHASE13_CONFIRMATION_SEAL: &str =
    "2e26e3d94c9ab267862869cf7fc3cc8740a15f94e4dad7cf6df10b2e254da93c";
pub const INPUT_SIDE: usize = 200;
pub const INPUT_PIXELS: usize = INPUT_SIDE * INPUT_SIDE;
pub const MODEL_SIDE: usize = 100;
pub const MODEL_INPUT_LEN: usize = 2 * MODEL_SIDE * MODEL_SIDE;
pub const PAIRS_PER_SESSION: usize = 2_040;
pub const CSV_ROWS_PER_SESSION: usize = 4_080;
pub const RETAINED_PAIRS_PER_SESSION: usize = 1_224;
pub const ASSOCIATION_PAIRS_PER_SESSION: usize = 104;
pub const RECOGNITION_THRESHOLD: f32 = 0.05;
pub const RECOGNITION_PERCENT: usize = 95;
pub const OPENNESS_DEADBAND: f64 = 0.004;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct PhaseSpec {
    pub name: &'static str,
    pub label: f32,
    pub pairs: usize,
    pub first_sequence: usize,
}

impl PhaseSpec {
    pub const fn last_sequence(self) -> usize {
        self.first_sequence + self.pairs - 1
    }
}

pub const PHASE_SCHEDULE: [PhaseSpec; 14] = [
    PhaseSpec {
        name: "neutral_center",
        label: 0.0,
        pairs: 180,
        first_sequence: 0,
    },
    PhaseSpec {
        name: "wide_soft",
        label: 0.5,
        pairs: 150,
        first_sequence: 180,
    },
    PhaseSpec {
        name: "wide_max",
        label: 1.0,
        pairs: 150,
        first_sequence: 330,
    },
    PhaseSpec {
        name: "gaze_up_neutral",
        label: 0.0,
        pairs: 240,
        first_sequence: 480,
    },
    PhaseSpec {
        name: "gaze_down_neutral",
        label: 0.0,
        pairs: 150,
        first_sequence: 720,
    },
    PhaseSpec {
        name: "gaze_left_neutral",
        label: 0.0,
        pairs: 120,
        first_sequence: 870,
    },
    PhaseSpec {
        name: "gaze_right_neutral",
        label: 0.0,
        pairs: 120,
        first_sequence: 990,
    },
    PhaseSpec {
        name: "wide_gaze_up",
        label: 1.0,
        pairs: 120,
        first_sequence: 1_110,
    },
    PhaseSpec {
        name: "wide_gaze_down",
        label: 1.0,
        pairs: 120,
        first_sequence: 1_230,
    },
    PhaseSpec {
        name: "blink_negative",
        label: 0.0,
        pairs: 180,
        first_sequence: 1_350,
    },
    PhaseSpec {
        name: "closed_negative",
        label: 0.0,
        pairs: 120,
        first_sequence: 1_530,
    },
    PhaseSpec {
        name: "squint_negative",
        label: 0.0,
        pairs: 150,
        first_sequence: 1_650,
    },
    PhaseSpec {
        name: "left_wink_negative",
        label: 0.0,
        pairs: 120,
        first_sequence: 1_800,
    },
    PhaseSpec {
        name: "right_wink_negative",
        label: 0.0,
        pairs: 120,
        first_sequence: 1_920,
    },
];

pub fn phase_spec(name: &str) -> Option<&'static PhaseSpec> {
    PHASE_SCHEDULE.iter().find(|spec| spec.name == name)
}

pub fn phase_for_sequence(sequence: usize) -> Option<&'static PhaseSpec> {
    PHASE_SCHEDULE
        .iter()
        .find(|spec| (spec.first_sequence..=spec.last_sequence()).contains(&sequence))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum InputTerminalStatus {
    InputSealed,
    InputInvalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InventoryKind {
    Directory,
    File,
    SymlinkOrReparse,
    Special,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InventoryEntry {
    pub path: String,
    pub kind: InventoryKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_len: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordingInventoryArtifact {
    pub schema: String,
    pub entries: Vec<InventoryEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPlanEntry {
    pub session_id: String,
    pub pair_count: usize,
    pub retained_pair_count: usize,
    pub association_pair_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionPlanArtifact {
    pub schema: String,
    pub terminal_status: InputTerminalStatus,
    pub errors: Vec<String>,
    pub completed_sessions: Vec<SessionPlanEntry>,
    pub partial_session_ids: Vec<String>,
    pub recording_tree_sha256: String,
    pub phase_schedule: Vec<PhasePlanEntry>,
    pub independence: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhasePlanEntry {
    pub phase: String,
    pub label_bits: u32,
    pub pair_count: usize,
    pub first_sequence: usize,
    pub last_sequence: usize,
    pub trim_each_end: usize,
    pub retained_pair_count: usize,
    pub association_pair_count: usize,
    pub block_size: usize,
}

#[derive(Clone, Debug)]
pub struct RuntimeFile {
    pub safe_path: String,
    pub source_path: PathBuf,
    pub byte_len: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeCase {
    pub session_id: String,
    pub phase: String,
    pub sequence: usize,
    pub label_bits: u32,
    pub left_safe_path: String,
    pub right_safe_path: String,
}

impl RuntimeCase {
    pub fn key(&self) -> String {
        format!("{}/{}/{:08}", self.session_id, self.phase, self.sequence)
    }
}

#[derive(Clone, Debug)]
pub struct RuntimeSession {
    pub session_id: String,
    pub source_dir: PathBuf,
    pub cases: Vec<RuntimeCase>,
}

#[derive(Clone, Debug)]
pub struct RuntimeInputPlan {
    pub wide_data_root: PathBuf,
    pub sessions_root: PathBuf,
    pub files: BTreeMap<String, RuntimeFile>,
    pub completed_sessions: Vec<RuntimeSession>,
}

#[derive(Clone, Debug)]
pub struct BuiltInputPlan {
    pub inventory: RecordingInventoryArtifact,
    pub session_plan: SessionPlanArtifact,
    pub recording_inventory_json: Vec<u8>,
    pub session_plan_json: Vec<u8>,
    pub recording_tree_sha256: String,
    pub runtime: RuntimeInputPlan,
}

impl BuiltInputPlan {
    pub fn terminal_status(&self) -> InputTerminalStatus {
        self.session_plan.terminal_status
    }

    pub fn input_payloads(&self) -> BTreeMap<String, Vec<u8>> {
        BTreeMap::from([
            (
                "recording_inventory.json".into(),
                self.recording_inventory_json.clone(),
            ),
            ("session_plan.json".into(), self.session_plan_json.clone()),
        ])
    }

    pub fn bit_exact_artifacts_match(&self, other: &Self) -> bool {
        self.recording_inventory_json == other.recording_inventory_json
            && self.session_plan_json == other.session_plan_json
            && self.recording_tree_sha256 == other.recording_tree_sha256
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HardInputError {
    pub operation: &'static str,
    pub detail: String,
}

impl fmt::Display for HardInputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.operation, self.detail)
    }
}

impl Error for HardInputError {}

fn hard(operation: &'static str, error: impl fmt::Display) -> HardInputError {
    HardInputError {
        operation,
        detail: error.to_string(),
    }
}

fn canonical_json<T: Serialize>(value: &T) -> Result<Vec<u8>, HardInputError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| hard("serialize JSON", e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn domain_sha256_hex(domain: &[u8], fields: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    hash.update((domain.len() as u64).to_le_bytes());
    hash.update(domain);
    for field in fields {
        hash.update((field.len() as u64).to_le_bytes());
        hash.update(field);
    }
    format!("{:x}", hash.finalize())
}

fn is_valid_session_name(name: &str) -> bool {
    let bytes = name.as_bytes();
    bytes.len() == 39
        && &bytes[..8] == b"session-"
        && bytes[8..21].iter().all(u8::is_ascii_digit)
        && bytes[21] == b'-'
        && bytes[22..32].iter().all(u8::is_ascii_digit)
        && bytes[32] == b'-'
        && bytes[33..39].iter().all(u8::is_ascii_digit)
}

#[cfg(windows)]
fn is_reparse(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x0000_0400;
    metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
}

#[cfg(not(windows))]
fn is_reparse(_metadata: &fs::Metadata) -> bool {
    false
}

#[derive(Clone, Debug)]
struct SortedDirEntry {
    utf8_name: Option<String>,
    path: PathBuf,
    raw_sort_key: Vec<u8>,
}

#[cfg(unix)]
fn os_name_sort_key(name: &OsStr) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    name.as_bytes().to_vec()
}

#[cfg(windows)]
fn os_name_sort_key(name: &OsStr) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    name.encode_wide()
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>()
}

#[cfg(not(any(unix, windows)))]
fn os_name_sort_key(name: &OsStr) -> Vec<u8> {
    format!("{name:?}").into_bytes()
}

/// Enumerate a directory without requiring every source name to be UTF-8.
/// Valid UTF-8 names retain the preregistered byte ordering. Non-UTF-8 names
/// sort deterministically after them using a runtime-only OS representation;
/// that raw key is never serialized or included in an error.
fn sorted_dir(path: &Path) -> Result<Vec<SortedDirEntry>, HardInputError> {
    let iter = fs::read_dir(path).map_err(|e| hard("enumerate directory", e))?;
    let mut entries = Vec::new();
    for entry in iter {
        let entry = entry.map_err(|e| hard("enumerate directory entry", e))?;
        let source_name = entry.file_name();
        entries.push(SortedDirEntry {
            utf8_name: source_name.to_str().map(str::to_owned),
            path: entry.path(),
            raw_sort_key: os_name_sort_key(&source_name),
        });
    }
    entries.sort_by(|left, right| match (&left.utf8_name, &right.utf8_name) {
        (Some(left), Some(right)) => left.as_bytes().cmp(right.as_bytes()),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => left.raw_sort_key.cmp(&right.raw_sort_key),
    });
    Ok(entries)
}

fn anonymized_non_utf8_component(rank: usize, occupied: &mut BTreeSet<String>) -> String {
    let base = format!("__non_utf8_{rank:04}__");
    if occupied.insert(base.clone()) {
        return base;
    }
    for disambiguator in 1usize.. {
        let candidate = format!("{base}_{disambiguator:04}");
        if occupied.insert(candidate.clone()) {
            return candidate;
        }
    }
    unreachable!("finite directory cannot exhaust anonymized names")
}

fn entry_kind(metadata: &fs::Metadata) -> InventoryKind {
    if metadata.file_type().is_symlink() || is_reparse(metadata) {
        InventoryKind::SymlinkOrReparse
    } else if metadata.is_dir() {
        InventoryKind::Directory
    } else if metadata.is_file() {
        InventoryKind::File
    } else {
        InventoryKind::Special
    }
}

fn safe_join(prefix: &str, components: &[String]) -> String {
    if components.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}/{}", components.join("/"))
    }
}

fn inventory_subtree(
    source_root: &Path,
    safe_root: &str,
    components: &mut Vec<String>,
    inventory: &mut Vec<InventoryEntry>,
    runtime_files: &mut BTreeMap<String, RuntimeFile>,
    errors: &mut Vec<String>,
) -> Result<(), HardInputError> {
    let entries = sorted_dir(source_root)?;
    let mut occupied_names: BTreeSet<String> = entries
        .iter()
        .filter_map(|entry| entry.utf8_name.clone())
        .collect();
    let mut non_utf8_rank = 0usize;
    for entry in entries {
        let (name, non_utf8_name) = match entry.utf8_name {
            Some(name) => (name, false),
            None => {
                let name = anonymized_non_utf8_component(non_utf8_rank, &mut occupied_names);
                non_utf8_rank += 1;
                (name, true)
            }
        };
        let path = entry.path;
        components.push(name);
        let safe_path = safe_join(safe_root, components);
        if non_utf8_name {
            errors.push(format!("non_utf8_entry_name:{safe_path}"));
        }
        let metadata = fs::symlink_metadata(&path).map_err(|e| hard("inspect entry", e))?;
        let kind = entry_kind(&metadata);
        match kind {
            InventoryKind::Directory => {
                inventory.push(InventoryEntry {
                    path: safe_path,
                    kind,
                    byte_len: None,
                    sha256: None,
                });
                inventory_subtree(
                    &path,
                    safe_root,
                    components,
                    inventory,
                    runtime_files,
                    errors,
                )?;
            }
            InventoryKind::File => {
                let bytes = fs::read(&path).map_err(|e| hard("read regular file", e))?;
                let byte_len = bytes.len() as u64;
                let sha256 = sha256_hex(&bytes);
                inventory.push(InventoryEntry {
                    path: safe_path.clone(),
                    kind,
                    byte_len: Some(byte_len),
                    sha256: Some(sha256.clone()),
                });
                runtime_files.insert(
                    safe_path.clone(),
                    RuntimeFile {
                        safe_path,
                        source_path: path,
                        byte_len,
                        sha256,
                    },
                );
            }
            InventoryKind::SymlinkOrReparse | InventoryKind::Special => {
                errors.push(format!("forbidden_entry_kind:{safe_path}:{kind:?}"));
                inventory.push(InventoryEntry {
                    path: safe_path,
                    kind,
                    byte_len: None,
                    sha256: None,
                });
            }
        }
        components.pop();
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct RootSession {
    source_name: String,
    source_dir: PathBuf,
    completed: bool,
}

fn ordinary_labels_file(dir: &Path) -> Result<bool, HardInputError> {
    let path = dir.join("labels.csv");
    match fs::symlink_metadata(&path) {
        Ok(metadata) => Ok(entry_kind(&metadata) == InventoryKind::File),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(hard("inspect labels.csv", error)),
    }
}

fn phase_plan_entries() -> Vec<PhasePlanEntry> {
    PHASE_SCHEDULE
        .iter()
        .map(|spec| {
            let trim = spec.pairs / 5;
            let retained = spec.pairs - 2 * trim;
            PhasePlanEntry {
                phase: spec.name.into(),
                label_bits: spec.label.to_bits(),
                pair_count: spec.pairs,
                first_sequence: spec.first_sequence,
                last_sequence: spec.last_sequence(),
                trim_each_end: trim,
                retained_pair_count: retained,
                association_pair_count: (retained + 11) / 12,
                block_size: spec.pairs / 5,
            }
        })
        .collect()
}

fn validate_relative_csv_path(path: &str) -> bool {
    if path.is_empty() || path.contains('\\') || path.starts_with('/') || path.contains(':') {
        return false;
    }
    let parsed = Path::new(path);
    if parsed.is_absolute() {
        return false;
    }
    parsed
        .components()
        .all(|component| matches!(component, Component::Normal(_)))
}

fn expected_filename(spec: &PhaseSpec, side: char, sequence: usize) -> String {
    format!("{0}/{0}_{side}_{sequence:08}.png", spec.name)
}

fn parse_completed_csv(
    session_id: &str,
    bytes: &[u8],
    session_source_dir: &Path,
) -> Result<Vec<RuntimeCase>, Vec<String>> {
    let mut errors = Vec::new();
    if bytes.contains(&b'\r') {
        errors.push(format!("csv_not_lf_only:{session_id}"));
    }
    if !bytes.ends_with(b"\n") {
        errors.push(format!("csv_missing_final_lf:{session_id}"));
    }
    let text = match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(_) => {
            errors.push(format!("csv_not_utf8:{session_id}"));
            return Err(errors);
        }
    };
    let mut lines: Vec<&str> = text.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }
    if lines.first().copied() != Some("filename,wide,phase,side") {
        errors.push(format!("csv_header_mismatch:{session_id}"));
    }
    let rows = lines.get(1..).unwrap_or_default();
    if rows.len() != CSV_ROWS_PER_SESSION {
        errors.push(format!(
            "csv_row_count:{session_id}:expected={CSV_ROWS_PER_SESSION}:actual={}",
            rows.len()
        ));
    }
    if rows.iter().any(|row| row.is_empty()) {
        errors.push(format!("csv_blank_data_row:{session_id}"));
    }

    let mut cases = Vec::with_capacity(PAIRS_PER_SESSION);
    let mut seen = BTreeSet::new();
    for sequence in 0..PAIRS_PER_SESSION {
        let Some(spec) = phase_for_sequence(sequence) else {
            errors.push(format!("internal_schedule_gap:{sequence}"));
            continue;
        };
        let mut paths = [String::new(), String::new()];
        for (eye, side) in ['l', 'r'].into_iter().enumerate() {
            let row_index = sequence * 2 + eye;
            let Some(row) = rows.get(row_index) else {
                continue;
            };
            let fields: Vec<&str> = row.split(',').collect();
            if fields.len() != 4 {
                errors.push(format!(
                    "csv_field_count:{session_id}:row={}",
                    row_index + 2
                ));
                continue;
            }
            let expected = expected_filename(spec, side, sequence);
            if !validate_relative_csv_path(fields[0]) {
                errors.push(format!(
                    "csv_unsafe_path:{session_id}:row={}",
                    row_index + 2
                ));
            }
            if fields[0] != expected {
                errors.push(format!(
                    "csv_filename_mismatch:{session_id}:row={}",
                    row_index + 2
                ));
            }
            if !seen.insert(fields[0].to_owned()) {
                errors.push(format!(
                    "csv_duplicate_file:{session_id}:row={}",
                    row_index + 2
                ));
            }
            match fields[1].parse::<f32>() {
                Ok(value) if value.is_finite() && value == spec.label => {}
                _ => errors.push(format!(
                    "csv_label_mismatch:{session_id}:row={}",
                    row_index + 2
                )),
            }
            if fields[2] != spec.name {
                errors.push(format!(
                    "csv_phase_mismatch:{session_id}:row={}",
                    row_index + 2
                ));
            }
            let mut expected_side = [0u8; 4];
            let expected_side = side.encode_utf8(&mut expected_side);
            if fields[3] != expected_side {
                errors.push(format!(
                    "csv_side_mismatch:{session_id}:row={}",
                    row_index + 2
                ));
            }
            paths[eye] = fields[0].to_owned();
        }
        if !paths[0].is_empty() && !paths[1].is_empty() {
            cases.push(RuntimeCase {
                session_id: session_id.into(),
                phase: spec.name.into(),
                sequence,
                label_bits: spec.label.to_bits(),
                left_safe_path: format!("{session_id}/images/{}", paths[0]),
                right_safe_path: format!("{session_id}/images/{}", paths[1]),
            });
        }
    }
    if cases.len() != PAIRS_PER_SESSION {
        errors.push(format!(
            "csv_pair_count:{session_id}:actual={}",
            cases.len()
        ));
    }
    // The source directory is deliberately used only for runtime path construction.
    let _ = session_source_dir;
    if errors.is_empty() {
        Ok(cases)
    } else {
        Err(errors)
    }
}

fn expected_completed_tree(session_id: &str) -> BTreeSet<String> {
    let mut expected = BTreeSet::from([
        session_id.to_owned(),
        format!("{session_id}/labels.csv"),
        format!("{session_id}/images"),
    ]);
    for spec in &PHASE_SCHEDULE {
        expected.insert(format!("{session_id}/images/{}", spec.name));
        for sequence in spec.first_sequence..=spec.last_sequence() {
            expected.insert(format!(
                "{session_id}/images/{}",
                expected_filename(spec, 'l', sequence)
            ));
            expected.insert(format!(
                "{session_id}/images/{}",
                expected_filename(spec, 'r', sequence)
            ));
        }
    }
    expected
}

fn inventory_anonymized_root_entry(
    path: PathBuf,
    metadata: fs::Metadata,
    safe: String,
    inventory: &mut Vec<InventoryEntry>,
    runtime_files: &mut BTreeMap<String, RuntimeFile>,
    errors: &mut Vec<String>,
) -> Result<(), HardInputError> {
    let kind = entry_kind(&metadata);
    match kind {
        InventoryKind::Directory => {
            inventory.push(InventoryEntry {
                path: safe.clone(),
                kind,
                byte_len: None,
                sha256: None,
            });
            inventory_subtree(
                &path,
                &safe,
                &mut Vec::new(),
                inventory,
                runtime_files,
                errors,
            )?;
        }
        InventoryKind::File => {
            let bytes = fs::read(&path).map_err(|e| hard("read root regular file", e))?;
            let sha256 = sha256_hex(&bytes);
            inventory.push(InventoryEntry {
                path: safe.clone(),
                kind,
                byte_len: Some(bytes.len() as u64),
                sha256: Some(sha256.clone()),
            });
            runtime_files.insert(
                safe.clone(),
                RuntimeFile {
                    safe_path: safe,
                    source_path: path,
                    byte_len: bytes.len() as u64,
                    sha256,
                },
            );
        }
        InventoryKind::SymlinkOrReparse | InventoryKind::Special => {
            inventory.push(InventoryEntry {
                path: safe,
                kind,
                byte_len: None,
                sha256: None,
            });
        }
    }
    Ok(())
}

/// Enumerate, hash, and non-pixel-validate an XR5 `wide_data` tree.
///
/// Only inability to enumerate or read the tree returns `Err`.  Every fully
/// enumerable schema problem is represented by an `InputInvalid` plan, so the
/// caller can publish the preregistered sealed invalid-input artifact.
pub fn build_input_plan(wide_data_root: &Path) -> Result<BuiltInputPlan, HardInputError> {
    let sessions_root = wide_data_root.join("sessions");
    let mut errors = Vec::new();
    let mut inventory = Vec::new();
    let mut runtime_files = BTreeMap::new();
    let root_entries = match fs::symlink_metadata(&sessions_root) {
        Ok(metadata) if entry_kind(&metadata) == InventoryKind::Directory => {
            sorted_dir(&sessions_root)?
        }
        Ok(metadata) => {
            let kind = entry_kind(&metadata);
            errors.push(format!("sessions_root_not_ordinary:{kind:?}"));
            inventory_anonymized_root_entry(
                sessions_root.clone(),
                metadata,
                "sessions_root".into(),
                &mut inventory,
                &mut runtime_files,
                &mut errors,
            )?;
            Vec::new()
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            errors.push("sessions_root_missing".into());
            Vec::new()
        }
        Err(error) => return Err(hard("inspect wide_data/sessions", error)),
    };
    let mut candidates = Vec::new();
    let mut forbidden_roots = Vec::new();
    let mut non_utf8_roots = Vec::new();
    for entry in root_entries {
        let path = entry.path;
        let metadata = fs::symlink_metadata(&path).map_err(|e| hard("inspect root entry", e))?;
        match entry.utf8_name {
            Some(name) if entry_kind(&metadata) == InventoryKind::Directory => {
                candidates.push(RootSession {
                    source_name: name,
                    completed: ordinary_labels_file(&path)?,
                    source_dir: path,
                });
            }
            Some(_name) => forbidden_roots.push((path, metadata)),
            None => non_utf8_roots.push((path, metadata)),
        }
    }

    let completed: Vec<_> = candidates.iter().filter(|s| s.completed).cloned().collect();
    let partial: Vec<_> = candidates
        .iter()
        .filter(|s| !s.completed)
        .cloned()
        .collect();
    let mut mapped = Vec::new();
    for (rank, source) in completed.into_iter().enumerate() {
        mapped.push((format!("session_{rank:04}"), source));
    }
    for (rank, source) in partial.into_iter().enumerate() {
        mapped.push((format!("partial_{rank:04}"), source));
    }
    mapped.sort_by(|a, b| a.1.source_name.as_bytes().cmp(b.1.source_name.as_bytes()));

    let mut runtime_sessions = Vec::new();
    let mut partial_ids = Vec::new();
    for (safe_id, source) in mapped {
        if !is_valid_session_name(&source.source_name) {
            errors.push(format!("invalid_session_name:{safe_id}"));
        }
        inventory.push(InventoryEntry {
            path: safe_id.clone(),
            kind: InventoryKind::Directory,
            byte_len: None,
            sha256: None,
        });
        inventory_subtree(
            &source.source_dir,
            &safe_id,
            &mut Vec::new(),
            &mut inventory,
            &mut runtime_files,
            &mut errors,
        )?;

        if source.completed {
            let labels_safe = format!("{safe_id}/labels.csv");
            let Some(labels_file) = runtime_files.get(&labels_safe) else {
                errors.push(format!("missing_regular_labels:{safe_id}"));
                continue;
            };
            let labels =
                fs::read(&labels_file.source_path).map_err(|e| hard("reread labels.csv", e))?;
            if labels.len() as u64 != labels_file.byte_len
                || sha256_hex(&labels) != labels_file.sha256
            {
                return Err(hard(
                    "reread labels.csv",
                    "file changed during input sealing",
                ));
            }
            let cases = match parse_completed_csv(&safe_id, &labels, &source.source_dir) {
                Ok(cases) => cases,
                Err(mut csv_errors) => {
                    errors.append(&mut csv_errors);
                    Vec::new()
                }
            };
            runtime_sessions.push(RuntimeSession {
                session_id: safe_id,
                source_dir: source.source_dir,
                cases,
            });
        } else {
            partial_ids.push(safe_id);
        }
    }

    for (rank, (path, metadata)) in non_utf8_roots.into_iter().enumerate() {
        let safe = format!("non_utf8_root_{rank:04}");
        errors.push(format!("non_utf8_entry_name:{safe}"));
        inventory_anonymized_root_entry(
            path,
            metadata,
            safe,
            &mut inventory,
            &mut runtime_files,
            &mut errors,
        )?;
    }

    for (rank, (path, metadata)) in forbidden_roots.into_iter().enumerate() {
        let safe = format!("forbidden_root_{rank:04}");
        let kind = entry_kind(&metadata);
        errors.push(format!("forbidden_root_entry:{safe}:{kind:?}"));
        inventory_anonymized_root_entry(
            path,
            metadata,
            safe,
            &mut inventory,
            &mut runtime_files,
            &mut errors,
        )?;
    }

    inventory.sort_by(|a, b| a.path.as_bytes().cmp(b.path.as_bytes()));
    for session in &runtime_sessions {
        let actual: BTreeSet<_> = inventory
            .iter()
            .filter(|entry| {
                entry.path == session.session_id
                    || entry.path.starts_with(&format!("{}/", session.session_id))
            })
            .map(|entry| entry.path.clone())
            .collect();
        let expected = expected_completed_tree(&session.session_id);
        for missing in expected.difference(&actual) {
            errors.push(format!("missing_completed_entry:{missing}"));
        }
        for extra in actual.difference(&expected) {
            errors.push(format!("extra_completed_entry:{extra}"));
        }
        for expected_dir in std::iter::once(session.session_id.clone())
            .chain(std::iter::once(format!("{}/images", session.session_id)))
            .chain(
                PHASE_SCHEDULE
                    .iter()
                    .map(|s| format!("{}/images/{}", session.session_id, s.name)),
            )
        {
            if inventory
                .iter()
                .find(|e| e.path == expected_dir)
                .map(|e| e.kind)
                != Some(InventoryKind::Directory)
            {
                errors.push(format!("expected_directory_not_ordinary:{expected_dir}"));
            }
        }
        for case in &session.cases {
            for safe in [&case.left_safe_path, &case.right_safe_path] {
                if !runtime_files.contains_key(safe) {
                    errors.push(format!("referenced_png_missing:{safe}"));
                }
            }
        }
    }
    let inventory_artifact = RecordingInventoryArtifact {
        schema: "sranibro.phase14.recording-inventory.v1".into(),
        entries: inventory,
    };
    let recording_inventory_json = canonical_json(&inventory_artifact)?;
    let recording_tree_sha256 = domain_sha256_hex(
        b"sranibro.phase14.recording-tree.v1",
        &[&recording_inventory_json],
    );
    if runtime_sessions.len() < 2 {
        errors.push("insufficient_completed_sessions".into());
    }
    errors.sort_by(|a, b| a.as_bytes().cmp(b.as_bytes()));
    errors.dedup();
    let terminal_status = if errors.is_empty() {
        InputTerminalStatus::InputSealed
    } else {
        InputTerminalStatus::InputInvalid
    };
    let session_plan = SessionPlanArtifact {
        schema: "sranibro.phase14.session-plan.v1".into(),
        terminal_status,
        errors,
        completed_sessions: runtime_sessions
            .iter()
            .map(|session| SessionPlanEntry {
                session_id: session.session_id.clone(),
                pair_count: session.cases.len(),
                retained_pair_count: if session.cases.len() == PAIRS_PER_SESSION {
                    RETAINED_PAIRS_PER_SESSION
                } else {
                    0
                },
                association_pair_count: if session.cases.len() == PAIRS_PER_SESSION {
                    ASSOCIATION_PAIRS_PER_SESSION
                } else {
                    0
                },
            })
            .collect(),
        partial_session_ids: partial_ids,
        recording_tree_sha256: recording_tree_sha256.clone(),
        phase_schedule: phase_plan_entries(),
        independence: "independence_unproven".into(),
    };
    let session_plan_json = canonical_json(&session_plan)?;
    Ok(BuiltInputPlan {
        inventory: inventory_artifact,
        session_plan,
        recording_inventory_json,
        session_plan_json,
        recording_tree_sha256,
        runtime: RuntimeInputPlan {
            wide_data_root: wide_data_root.to_path_buf(),
            sessions_root,
            files: runtime_files,
            completed_sessions: runtime_sessions,
        },
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PlaneStats {
    pub mean: f64,
    pub population_stddev: f64,
    pub q01: f64,
    pub q16: f64,
    pub q50: f64,
    pub q84: f64,
    pub q99: f64,
    pub black_fraction: f64,
    pub saturated_fraction: f64,
    pub mean_abs_neighbor_gradient: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EyeInputStats {
    pub reconstructed_native: PlaneStats,
    pub post_despeckle_native: PlaneStats,
    pub final_model_channel: PlaneStats,
    pub despeckle_changed_pixel_fraction: f64,
    pub despeckle_mean_signed_change: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreparedFrame {
    pub case_key: String,
    pub session_id: String,
    pub phase: String,
    pub sequence: usize,
    pub label_bits: u32,
    pub left_native_sha256: String,
    pub right_native_sha256: String,
    pub tensor_sha256: String,
    pub left_stats: EyeInputStats,
    pub right_stats: EyeInputStats,
}

#[derive(Clone, Debug)]
struct DecodedPreparedCase {
    frame: PreparedFrame,
    tensor: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassFailure {
    pub case_key: String,
    pub reason: String,
    /// A sealed file vanished, could not be read, or no longer matched its
    /// sealed length/hash.  This is an untrusted recording-tree mutation and
    /// the caller must not publish even a reduced artifact.
    pub untrusted_input_change: bool,
}

fn pass_failure(case_key: String, reason: String) -> PassFailure {
    let untrusted_input_change = [
        "read_failed:",
        "sealed_length_mismatch:",
        "sealed_sha256_mismatch:",
        "sealed_file_missing:",
    ]
    .iter()
    .any(|marker| reason.contains(marker));
    PassFailure {
        case_key,
        reason,
        untrusted_input_change,
    }
}

fn read_verified_file(file: &RuntimeFile) -> Result<Vec<u8>, String> {
    let bytes = fs::read(&file.source_path)
        .map_err(|error| format!("read_failed:{}:{error}", file.safe_path))?;
    if bytes.len() as u64 != file.byte_len {
        return Err(format!("sealed_length_mismatch:{}", file.safe_path));
    }
    if sha256_hex(&bytes) != file.sha256 {
        return Err(format!("sealed_sha256_mismatch:{}", file.safe_path));
    }
    Ok(bytes)
}

fn decode_gray8_200(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let mut reader = decoder.read_info().map_err(|e| format!("png_header:{e}"))?;
    let info = reader.info();
    if info.width != INPUT_SIDE as u32 || info.height != INPUT_SIDE as u32 {
        return Err(format!("png_dimensions:{}x{}", info.width, info.height));
    }
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "png_format:{:?}:{:?}",
            info.color_type, info.bit_depth
        ));
    }
    if info.animation_control.is_some() || info.frame_control.is_some() {
        return Err("png_animation_or_extra_frame".into());
    }
    let mut output = vec![0u8; reader.output_buffer_size()];
    let frame = reader
        .next_frame(&mut output)
        .map_err(|e| format!("png_decode:{e}"))?;
    if frame.width != INPUT_SIDE as u32
        || frame.height != INPUT_SIDE as u32
        || frame.color_type != png::ColorType::Grayscale
        || frame.bit_depth != png::BitDepth::Eight
        || frame.buffer_size() != INPUT_PIXELS
    {
        return Err("png_decoded_frame_contract".into());
    }
    let buffer_size = frame.buffer_size();
    output.truncate(buffer_size);
    // `next_frame` can finish after IDAT/fdAT. `finish` is what consumes and
    // validates through IEND, so a truncated/corrupt tail cannot enter the
    // preregistered audit as if it were a complete PNG.
    reader.finish().map_err(|e| format!("png_finish:{e}"))?;
    Ok(output)
}

pub fn mirror_gray_horizontal(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; width.saturating_mul(height)];
    if pixels.len() < output.len() {
        return Vec::new();
    }
    for y in 0..height {
        for x in 0..width {
            output[y * width + x] = pixels[y * width + (width - 1 - x)];
        }
    }
    output
}

fn frozen_geometry() -> Result<[MlGeometry; 2], String> {
    let expected = [
        MlGeometry {
            crop_left: 0.0,
            crop_right: 0.40,
            crop_top: 0.15,
            crop_bottom: 0.15,
            scale_x: 1.0,
            scale_y: 1.20,
            rotate_deg: -30.0,
            mirror_h: None,
        },
        MlGeometry {
            crop_left: 0.40,
            crop_right: 0.0,
            crop_top: 0.15,
            crop_bottom: 0.15,
            scale_x: 1.0,
            scale_y: 1.20,
            rotate_deg: 30.0,
            mirror_h: Some(true),
        },
    ];
    let actual = sranibro_rs::config::default_ml_geometry("pimax_xr5");
    let geometry_bits = |g: &MlGeometry| {
        [
            g.crop_left.to_bits(),
            g.crop_right.to_bits(),
            g.crop_top.to_bits(),
            g.crop_bottom.to_bits(),
            g.scale_x.to_bits(),
            g.scale_y.to_bits(),
            g.rotate_deg.to_bits(),
            g.mirror_h.map(u32::from).unwrap_or(2),
        ]
    };
    if geometry_bits(&actual[0]) != geometry_bits(&expected[0])
        || geometry_bits(&actual[1]) != geometry_bits(&expected[1])
    {
        return Err("default_xr5_geometry_mismatch".into());
    }
    let despeckle = DespeckleParams::default();
    if !despeckle.enabled
        || despeckle.threshold.to_bits() != 0.15f32.to_bits()
        || despeckle.radius != 3
    {
        return Err("default_despeckle_mismatch".into());
    }
    let flatten = FlattenParams::default();
    if flatten.enabled
        || flatten.strength.to_bits() != 0.7f32.to_bits()
        || flatten.radius.to_bits() != 0.33f32.to_bits()
    {
        return Err("default_flatten_mismatch".into());
    }
    Ok(actual)
}

fn quantile_sorted(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || sorted.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = position.ceil() as usize;
    let fraction = position - low as f64;
    Some(sorted[low] * (1.0 - fraction) + sorted[high] * fraction)
}

pub fn quantile(values: &[f64], probability: f64) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    quantile_sorted(&sorted, probability)
}

pub fn median_absolute_deviation(values: &[f64]) -> Option<f64> {
    let median = quantile(values, 0.5)?;
    let deviations: Vec<_> = values.iter().map(|value| (value - median).abs()).collect();
    quantile(&deviations, 0.5)
}

fn plane_stats_f64(values: &[f64], width: usize, height: usize) -> Result<PlaneStats, String> {
    if width == 0
        || height == 0
        || values.len() != width * height
        || values.iter().any(|v| !v.is_finite())
    {
        return Err("invalid_plane".into());
    }
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count;
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let black_fraction = values.iter().filter(|&&v| v <= 5.0 / 255.0).count() as f64 / count;
    let saturated_fraction = values.iter().filter(|&&v| v >= 250.0 / 255.0).count() as f64 / count;
    let mut gradient_sum = 0.0;
    let mut gradient_edges = 0usize;
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            if x + 1 < width {
                gradient_sum += (values[index] - values[index + 1]).abs();
                gradient_edges += 1;
            }
            if y + 1 < height {
                gradient_sum += (values[index] - values[index + width]).abs();
                gradient_edges += 1;
            }
        }
    }
    Ok(PlaneStats {
        mean,
        population_stddev: variance.sqrt(),
        q01: quantile_sorted(&sorted, 0.01).unwrap(),
        q16: quantile_sorted(&sorted, 0.16).unwrap(),
        q50: quantile_sorted(&sorted, 0.50).unwrap(),
        q84: quantile_sorted(&sorted, 0.84).unwrap(),
        q99: quantile_sorted(&sorted, 0.99).unwrap(),
        black_fraction,
        saturated_fraction,
        mean_abs_neighbor_gradient: gradient_sum / gradient_edges as f64,
    })
}

fn plane_stats_u8(values: &[u8], width: usize, height: usize) -> Result<PlaneStats, String> {
    let normalized: Vec<_> = values.iter().map(|&v| v as f64 / 255.0).collect();
    plane_stats_f64(&normalized, width, height)
}

fn preprocess_decoded_case(
    case: &RuntimeCase,
    left_stored: &[u8],
    right_stored: &[u8],
    geometry: &[MlGeometry; 2],
) -> Result<DecodedPreparedCase, String> {
    if left_stored.len() != INPUT_PIXELS || right_stored.len() != INPUT_PIXELS {
        return Err("decoded_pixel_count".into());
    }
    let left_native = left_stored.to_vec();
    // Capture mirrors right on save.  Undo that exactly once before the ordinary
    // live right-eye geometry, whose frozen mirror remains enabled.
    let right_native = mirror_gray_horizontal(right_stored, INPUT_SIDE, INPUT_SIDE);
    let despeckle = DespeckleParams::default();
    let left_post = preprocess::despeckle(&left_native, INPUT_SIDE, INPUT_SIDE, &despeckle);
    let right_post = preprocess::despeckle(&right_native, INPUT_SIDE, INPUT_SIDE, &despeckle);
    let flatten = FlattenParams::default();
    let left_flat = preprocess::flatten(&left_post, INPUT_SIDE, INPUT_SIDE, &flatten);
    let right_flat = preprocess::flatten(&right_post, INPUT_SIDE, INPUT_SIDE, &flatten);
    if left_flat != left_post || right_flat != right_post {
        return Err("disabled_flatten_not_identity".into());
    }
    let tensor = preprocess::to_input_stereo_geom(
        &left_flat,
        INPUT_SIDE as u32,
        INPUT_SIDE as u32,
        &right_flat,
        INPUT_SIDE as u32,
        INPUT_SIDE as u32,
        false,
        true,
        &geometry[0],
        &geometry[1],
    );
    if tensor.len() != MODEL_INPUT_LEN || tensor.iter().any(|value| !value.is_finite()) {
        return Err("invalid_model_tensor".into());
    }
    let left_final: Vec<_> = tensor[..MODEL_SIDE * MODEL_SIDE]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let right_final: Vec<_> = tensor[MODEL_SIDE * MODEL_SIDE..]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let make_stats =
        |native: &[u8], post: &[u8], final_values: &[f64]| -> Result<EyeInputStats, String> {
            let changed = native.iter().zip(post).filter(|(a, b)| a != b).count();
            let signed = native
                .iter()
                .zip(post)
                .map(|(&a, &b)| (b as f64 - a as f64) / 255.0)
                .sum::<f64>()
                / native.len() as f64;
            Ok(EyeInputStats {
                reconstructed_native: plane_stats_u8(native, INPUT_SIDE, INPUT_SIDE)?,
                post_despeckle_native: plane_stats_u8(post, INPUT_SIDE, INPUT_SIDE)?,
                final_model_channel: plane_stats_f64(final_values, MODEL_SIDE, MODEL_SIDE)?,
                despeckle_changed_pixel_fraction: changed as f64 / native.len() as f64,
                despeckle_mean_signed_change: signed,
            })
        };
    let mut tensor_bytes = Vec::with_capacity(tensor.len() * 4);
    for value in &tensor {
        tensor_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    let key = case.key();
    let left_hash = domain_sha256_hex(
        b"sranibro.phase14.reconstructed-native.left.v1",
        &[key.as_bytes(), &left_native],
    );
    let right_hash = domain_sha256_hex(
        b"sranibro.phase14.reconstructed-native.right.v1",
        &[key.as_bytes(), &right_native],
    );
    let tensor_hash = domain_sha256_hex(
        b"sranibro.phase14.final-stereo-tensor.v1",
        &[key.as_bytes(), &tensor_bytes],
    );
    Ok(DecodedPreparedCase {
        frame: PreparedFrame {
            case_key: key,
            session_id: case.session_id.clone(),
            phase: case.phase.clone(),
            sequence: case.sequence,
            label_bits: case.label_bits,
            left_native_sha256: left_hash,
            right_native_sha256: right_hash,
            tensor_sha256: tensor_hash,
            left_stats: make_stats(&left_native, &left_post, &left_final)?,
            right_stats: make_stats(&right_native, &right_post, &right_final)?,
        },
        tensor,
    })
}

fn decode_runtime_case(
    plan: &BuiltInputPlan,
    case: &RuntimeCase,
    geometry: &[MlGeometry; 2],
) -> Result<DecodedPreparedCase, String> {
    let left = plan
        .runtime
        .files
        .get(&case.left_safe_path)
        .ok_or_else(|| format!("sealed_file_missing:{}", case.left_safe_path))?;
    let right = plan
        .runtime
        .files
        .get(&case.right_safe_path)
        .ok_or_else(|| format!("sealed_file_missing:{}", case.right_safe_path))?;
    let left_bytes = read_verified_file(left)?;
    let right_bytes = read_verified_file(right)?;
    let left_pixels =
        decode_gray8_200(&left_bytes).map_err(|e| format!("{}:{e}", left.safe_path))?;
    let right_pixels =
        decode_gray8_200(&right_bytes).map_err(|e| format!("{}:{e}", right.safe_path))?;
    preprocess_decoded_case(case, &left_pixels, &right_pixels, geometry)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreflightPass {
    pub pass_name: String,
    pub ordered_stream_sha256: String,
    pub frames: Vec<PreparedFrame>,
}

pub fn run_preflight_pass(
    plan: &BuiltInputPlan,
    pass_name: &str,
) -> Result<PreflightPass, PassFailure> {
    let geometry = frozen_geometry()
        .map_err(|reason| pass_failure("static_preprocessing_contract".into(), reason))?;
    let mut frames = Vec::new();
    let mut stream = Sha256::new();
    stream.update(b"sranibro.phase14.preflight-stream.v1");
    stream.update((pass_name.len() as u64).to_le_bytes());
    stream.update(pass_name.as_bytes());
    for session in &plan.runtime.completed_sessions {
        for case in &session.cases {
            let prepared = decode_runtime_case(plan, case, &geometry)
                .map_err(|reason| pass_failure(case.key(), reason))?;
            for field in [
                prepared.frame.case_key.as_bytes(),
                prepared.frame.left_native_sha256.as_bytes(),
                prepared.frame.right_native_sha256.as_bytes(),
                prepared.frame.tensor_sha256.as_bytes(),
            ] {
                stream.update((field.len() as u64).to_le_bytes());
                stream.update(field);
            }
            frames.push(prepared.frame);
        }
    }
    Ok(PreflightPass {
        pass_name: pass_name.into(),
        ordered_stream_sha256: format!("{:x}", stream.finalize()),
        frames,
    })
}

pub fn preprocessing_mismatches(
    first: &PreflightPass,
    second: &PreflightPass,
) -> Vec<PreprocessMismatch> {
    let first_by_key: BTreeMap<_, _> = first
        .frames
        .iter()
        .map(|f| (f.case_key.as_str(), f))
        .collect();
    let second_by_key: BTreeMap<_, _> = second
        .frames
        .iter()
        .map(|f| (f.case_key.as_str(), f))
        .collect();
    let keys: BTreeSet<_> = first_by_key
        .keys()
        .chain(second_by_key.keys())
        .copied()
        .collect();
    keys.into_iter()
        .filter_map(|key| {
            let a = first_by_key.get(key).copied();
            let b = second_by_key.get(key).copied();
            match (a, b) {
                (Some(a), Some(b))
                    if a.left_native_sha256 == b.left_native_sha256
                        && a.right_native_sha256 == b.right_native_sha256
                        && a.tensor_sha256 == b.tensor_sha256 =>
                {
                    None
                }
                _ => Some(PreprocessMismatch {
                    case_key: key.into(),
                    pass_a_left_native_sha256: a.map(|v| v.left_native_sha256.clone()),
                    pass_b_left_native_sha256: b.map(|v| v.left_native_sha256.clone()),
                    pass_a_right_native_sha256: a.map(|v| v.right_native_sha256.clone()),
                    pass_b_right_native_sha256: b.map(|v| v.right_native_sha256.clone()),
                    pass_a_tensor_sha256: a.map(|v| v.tensor_sha256.clone()),
                    pass_b_tensor_sha256: b.map(|v| v.tensor_sha256.clone()),
                }),
            }
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreprocessMismatch {
    pub case_key: String,
    pub pass_a_left_native_sha256: Option<String>,
    pub pass_b_left_native_sha256: Option<String>,
    pub pass_a_right_native_sha256: Option<String>,
    pub pass_b_right_native_sha256: Option<String>,
    pub pass_a_tensor_sha256: Option<String>,
    pub pass_b_tensor_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFrameBits {
    pub case_key: String,
    pub output_bits: [u32; 5],
    pub left_native_sha256: String,
    pub right_native_sha256: String,
    pub tensor_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelPass {
    pub pass_name: String,
    pub ordered_stream_sha256: String,
    pub frames: Vec<ModelFrameBits>,
}

pub fn run_model_pass(
    plan: &BuiltInputPlan,
    net: &mut EyeNet,
    pass_name: &str,
    reverse: bool,
) -> Result<ModelPass, PassFailure> {
    let geometry = frozen_geometry()
        .map_err(|reason| pass_failure("static_preprocessing_contract".into(), reason))?;
    let mut cases: Vec<&RuntimeCase> = plan
        .runtime
        .completed_sessions
        .iter()
        .flat_map(|session| session.cases.iter())
        .collect();
    if reverse {
        cases.reverse();
    }
    let mut frames = Vec::with_capacity(cases.len());
    let mut stream = Sha256::new();
    stream.update(b"sranibro.phase14.model-stream.v1");
    stream.update((pass_name.len() as u64).to_le_bytes());
    stream.update(pass_name.as_bytes());
    for case in cases {
        let prepared = decode_runtime_case(plan, case, &geometry)
            .map_err(|reason| pass_failure(case.key(), reason))?;
        let outputs = net.forward_one(&prepared.tensor);
        let bits = outputs.map(f32::to_bits);
        stream.update((prepared.frame.case_key.len() as u64).to_le_bytes());
        stream.update(prepared.frame.case_key.as_bytes());
        for value in bits {
            stream.update(value.to_le_bytes());
        }
        frames.push(ModelFrameBits {
            case_key: prepared.frame.case_key,
            output_bits: bits,
            left_native_sha256: prepared.frame.left_native_sha256,
            right_native_sha256: prepared.frame.right_native_sha256,
            tensor_sha256: prepared.frame.tensor_sha256,
        });
    }
    Ok(ModelPass {
        pass_name: pass_name.into(),
        ordered_stream_sha256: format!("{:x}", stream.finalize()),
        frames,
    })
}

pub fn average_ranks(values: &[f64]) -> Option<Vec<f64>> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        let rank = ((start + 1) as f64 + end as f64) / 2.0;
        for &index in &order[start..end] {
            ranks[index] = rank;
        }
        start = end;
    }
    Some(ranks)
}

pub fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 3 {
        return None;
    }
    let rx = average_ranks(x)?;
    let ry = average_ranks(y)?;
    let mx = rx.iter().sum::<f64>() / rx.len() as f64;
    let my = ry.iter().sum::<f64>() / ry.len() as f64;
    let mut covariance = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&a, &b) in rx.iter().zip(&ry) {
        covariance += (a - mx) * (b - my);
        vx += (a - mx).powi(2);
        vy += (b - my).powi(2);
    }
    if vx == 0.0 || vy == 0.0 {
        None
    } else {
        Some(covariance / (vx * vy).sqrt())
    }
}

pub fn retained_role(spec: &PhaseSpec, sequence: usize) -> Option<(usize, bool)> {
    if sequence < spec.first_sequence || sequence > spec.last_sequence() {
        return None;
    }
    let local = sequence - spec.first_sequence;
    let trim = spec.pairs / 5;
    if local < trim || local >= spec.pairs - trim {
        return None;
    }
    let retained_index = local - trim;
    Some((retained_index, retained_index % 12 == 0))
}

pub fn block_number(spec: &PhaseSpec, sequence: usize) -> Option<usize> {
    if sequence < spec.first_sequence || sequence > spec.last_sequence() {
        return None;
    }
    Some((sequence - spec.first_sequence) / (spec.pairs / 5) + 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AnalysisTerminalStatus {
    InputInvalid,
    InconclusiveDeterminism,
    InconclusiveArtifact,
    AuditComplete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ModelLoaderErrorKind {
    /// The model could not be opened, read, or assigned an ordinary-file
    /// identity. This is a trusted reduced `INCONCLUSIVE_ARTIFACT` outcome.
    ReadOrIdentityFailure,
    /// Metadata or content changed while the loader was reading. The caller
    /// must not publish any artifact for this invocation.
    ChangedDuringRead,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelLoaderError {
    pub kind: ModelLoaderErrorKind,
    pub detail: String,
}

impl ModelLoaderError {
    pub fn read_or_identity(detail: impl Into<String>) -> Self {
        Self {
            kind: ModelLoaderErrorKind::ReadOrIdentityFailure,
            detail: detail.into(),
        }
    }

    pub fn changed_during_read(detail: impl Into<String>) -> Self {
        Self {
            kind: ModelLoaderErrorKind::ChangedDuringRead,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for ModelLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.detail)
    }
}

impl Error for ModelLoaderError {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAcquisitionMetadata {
    pub loader_invoked: bool,
    pub model_bytes_obtained: bool,
    pub observed_byte_len: Option<u64>,
    pub observed_sha256: Option<String>,
    pub loader_error_kind: Option<ModelLoaderErrorKind>,
}

impl ModelAcquisitionMetadata {
    fn not_invoked() -> Self {
        Self {
            loader_invoked: false,
            model_bytes_obtained: false,
            observed_byte_len: None,
            observed_sha256: None,
            loader_error_kind: None,
        }
    }

    fn obtained(bytes: &[u8]) -> Self {
        Self {
            loader_invoked: true,
            model_bytes_obtained: true,
            observed_byte_len: Some(bytes.len() as u64),
            observed_sha256: Some(sha256_hex(bytes)),
            loader_error_kind: None,
        }
    }

    fn failed(error: &ModelLoaderError) -> Self {
        Self {
            loader_invoked: true,
            model_bytes_obtained: false,
            observed_byte_len: None,
            observed_sha256: None,
            loader_error_kind: Some(error.kind),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PrimaryCategory {
    Monotone,
    Plateau,
    Reversal,
    SoftTransferNotShown,
    NotEvaluable,
}

impl PrimaryCategory {
    fn is_evaluable(self) -> bool {
        self != Self::NotEvaluable
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HighOpenAnnotation {
    HighOpenReversalObserved,
    HighOpenPlateauObserved,
    HighOpenMonotoneAll,
    HighOpenTransferNotEstablished,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum EyeCategoryAnnotation {
    EyeCategoryAsymmetric,
    EyeCategoryMatched,
    EyeCategoryNotEvaluable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockAgreement {
    AllAgree,
    Disagree,
    NotEvaluable,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnalysisDigests {
    pub preflight_a_ordered_stream_sha256: Option<String>,
    pub preflight_b_ordered_stream_sha256: Option<String>,
    pub model_a_ordered_stream_sha256: Option<String>,
    pub model_b_ordered_stream_sha256: Option<String>,
}

impl Default for AnalysisDigests {
    fn default() -> Self {
        Self {
            preflight_a_ordered_stream_sha256: None,
            preflight_b_ordered_stream_sha256: None,
            model_a_ordered_stream_sha256: None,
            model_b_ordered_stream_sha256: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub terminal_status: AnalysisTerminalStatus,
    pub high_open_annotation: Option<HighOpenAnnotation>,
    pub model_loaded: bool,
    pub model_byte_len: u64,
    pub model_sha256: String,
    pub model_loader_invoked: bool,
    pub model_bytes_obtained: bool,
    pub model_loader_error_kind: Option<ModelLoaderErrorKind>,
    pub completed_session_count: usize,
    pub frame_count: usize,
    pub retained_pairs_per_session: usize,
    pub association_pairs_per_session: usize,
    pub recording_tree_sha256: String,
    pub preprocessing_bit_identical: Option<bool>,
    pub model_bit_identical: Option<bool>,
    pub digests: AnalysisDigests,
    pub independence: String,
    pub preprocessing: FrozenPreprocessingMetadata,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrozenPreprocessingMetadata {
    pub left_geometry: MlGeometry,
    pub right_geometry: MlGeometry,
    pub left_mirror: bool,
    pub right_mirror: bool,
    pub saved_right_unmirror_count: usize,
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    pub adaptive_brightness_enabled: bool,
    pub brightness_affine_lr: [[f32; 2]; 2],
    pub eye_swap: bool,
    pub whole_frame_mapping_flip: bool,
}

impl FrozenPreprocessingMetadata {
    fn observed() -> Self {
        let geometry = sranibro_rs::config::default_ml_geometry("pimax_xr5");
        Self {
            left_geometry: geometry[0],
            right_geometry: geometry[1],
            left_mirror: false,
            right_mirror: true,
            saved_right_unmirror_count: 1,
            despeckle: DespeckleParams::default(),
            flatten: FlattenParams::default(),
            adaptive_brightness_enabled: false,
            brightness_affine_lr: [[1.0, 0.0], [1.0, 0.0]],
            eye_swap: false,
            whole_frame_mapping_flip: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterminismMismatch {
    pub case_key: String,
    pub kind: String,
    pub expected: Option<String>,
    pub observed: Option<String>,
    pub model_a_bits: Option<[u32; 5]>,
    pub model_b_bits: Option<[u32; 5]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NonfiniteModelFrame {
    pub case_key: String,
    pub output_bits: [u32; 5],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiagnosticArtifact {
    pub schema: String,
    pub terminal_status: AnalysisTerminalStatus,
    pub reason: String,
    pub model_loaded: bool,
    pub pass_failure: Option<PassFailure>,
    pub preprocessing_mismatches: Vec<PreprocessMismatch>,
    pub determinism_mismatches: Vec<DeterminismMismatch>,
    pub nonfinite_model_frames: Vec<NonfiniteModelFrame>,
    pub digests: AnalysisDigests,
    pub high_open_annotation: Option<HighOpenAnnotation>,
}

#[derive(Clone, Debug)]
pub struct AnalysisOutcome {
    pub terminal_status: AnalysisTerminalStatus,
    pub high_open_annotation: Option<HighOpenAnnotation>,
    /// Exact bytes for the complete or reduced stage-specific allowlist, excluding
    /// `manifest.json`, which is assembled and published last by the output module.
    pub payloads: BTreeMap<String, Vec<u8>>,
    pub metadata: AnalysisMetadata,
    pub diagnostic: Option<DiagnosticArtifact>,
    /// `true` means an input file changed/disappeared during one of the four
    /// passes.  The CLI must return a hard error without creating an artifact.
    pub publication_forbidden: bool,
}

impl AnalysisOutcome {
    pub fn complete_payload_allowlist() -> BTreeSet<&'static str> {
        BTreeSet::from([
            "frames.csv",
            "phase_summaries.json",
            "temporal_blocks.csv",
            "associations.csv",
            "gaze_and_session_differences.csv",
            "interpretation.txt",
        ])
    }

    pub fn reduced_payload_allowlist() -> BTreeSet<&'static str> {
        BTreeSet::from(["diagnostic.json", "interpretation.txt"])
    }

    pub fn payload_allowlist_is_exact(&self) -> bool {
        let actual: BTreeSet<_> = self.payloads.keys().map(String::as_str).collect();
        if self.terminal_status == AnalysisTerminalStatus::AuditComplete {
            actual == Self::complete_payload_allowlist()
        } else {
            actual == Self::reduced_payload_allowlist()
        }
    }
}

#[derive(Clone, Debug)]
struct AnalyzedFrame {
    prepared_a: PreparedFrame,
    prepared_b: PreparedFrame,
    model_a: ModelFrameBits,
    model_b: ModelFrameBits,
}

impl AnalyzedFrame {
    fn spec(&self) -> &'static PhaseSpec {
        phase_spec(&self.prepared_a.phase).expect("validated phase")
    }

    fn retained(&self) -> bool {
        retained_role(self.spec(), self.prepared_a.sequence).is_some()
    }

    fn association(&self) -> bool {
        retained_role(self.spec(), self.prepared_a.sequence)
            .is_some_and(|(_, association)| association)
    }

    fn block(&self) -> usize {
        block_number(self.spec(), self.prepared_a.sequence).expect("validated sequence")
    }

    fn outputs(&self) -> [f64; 5] {
        self.model_a
            .output_bits
            .map(|bits| f32::from_bits(bits) as f64)
    }

    fn recognized(&self) -> bool {
        let output = self.outputs();
        output.iter().all(|value| value.is_finite()) && output[0] > RECOGNITION_THRESHOLD as f64
    }

    fn openness(&self, eye: Eye) -> f64 {
        self.outputs()[match eye {
            Eye::Left => 1,
            Eye::Right => 2,
        }]
    }

    fn squeeze(&self, eye: Eye) -> f64 {
        self.outputs()[match eye {
            Eye::Left => 3,
            Eye::Right => 4,
        }]
    }

    fn stats(&self, eye: Eye) -> &EyeInputStats {
        match eye {
            Eye::Left => &self.prepared_a.left_stats,
            Eye::Right => &self.prepared_a.right_stats,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Eye {
    Left,
    Right,
}

impl Eye {
    const ALL: [Self; 2] = [Self::Left, Self::Right];
    fn code(self) -> &'static str {
        match self {
            Self::Left => "L",
            Self::Right => "R",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalarSummary {
    pub count: usize,
    pub median: f64,
    pub mad: f64,
    pub q05: f64,
    pub q25: f64,
    pub q75: f64,
    pub q95: f64,
}

impl ScalarSummary {
    fn from_values(values: &[f64]) -> Self {
        debug_assert!(!values.is_empty());
        Self {
            count: values.len(),
            median: quantile(values, 0.5).unwrap(),
            mad: median_absolute_deviation(values).unwrap(),
            q05: quantile(values, 0.05).unwrap(),
            q25: quantile(values, 0.25).unwrap(),
            q75: quantile(values, 0.75).unwrap(),
            q95: quantile(values, 0.95).unwrap(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstDifferenceSummary {
    pub count: usize,
    pub median_absolute: Option<f64>,
    pub q95_absolute: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseEyeSummary {
    pub session_id: String,
    pub phase: String,
    pub eye: Eye,
    pub retained_count: usize,
    pub recognized_count: usize,
    pub recognition_gate_passed: bool,
    pub presence: ScalarSummary,
    pub openness: ScalarSummary,
    pub squeeze: ScalarSummary,
    pub input_statistics: BTreeMap<String, ScalarSummary>,
    pub openness_first_difference: FirstDifferenceSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommonDifferentialSummary {
    pub session_id: String,
    pub phase: String,
    pub retained_count: usize,
    pub common_openness: ScalarSummary,
    pub differential_openness: ScalarSummary,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionEyePrimary {
    pub session_id: String,
    pub eye: Eye,
    pub neutral_median: f64,
    pub soft_median: f64,
    pub max_median: f64,
    pub soft_minus_neutral: f64,
    pub max_minus_soft: f64,
    pub max_minus_neutral: f64,
    pub category: PrimaryCategory,
    pub block_categories: [PrimaryCategory; 5],
    pub block_agreement: BlockAgreement,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEyeCategoryComparison {
    pub session_id: String,
    pub left: PrimaryCategory,
    pub right: PrimaryCategory,
    pub annotation: EyeCategoryAnnotation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndpointCheck {
    pub session_id: String,
    pub eye: Eye,
    pub check: String,
    pub reference_median: f64,
    pub contrast_median: f64,
    pub difference: f64,
    pub annotation: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseSummariesArtifact {
    pub schema: String,
    pub recognition_threshold: f32,
    pub recognition_gate_percent: usize,
    pub openness_deadband: f64,
    pub high_open_annotation: HighOpenAnnotation,
    pub session_eye_primary: Vec<SessionEyePrimary>,
    pub session_eye_category_comparison: Vec<SessionEyeCategoryComparison>,
    pub phase_eye_summaries: Vec<PhaseEyeSummary>,
    pub common_differential_summaries: Vec<CommonDifferentialSummary>,
    pub endpoint_checks: Vec<EndpointCheck>,
    pub independence: String,
}

fn input_metric_values(stats: &EyeInputStats) -> Vec<(&'static str, f64)> {
    fn push_plane(output: &mut Vec<(&'static str, f64)>, prefix: &'static str, p: &PlaneStats) {
        let names = [
            "mean",
            "population_stddev",
            "q01",
            "q16",
            "q50",
            "q84",
            "q99",
            "black_fraction",
            "saturated_fraction",
            "mean_abs_neighbor_gradient",
        ];
        let values = [
            p.mean,
            p.population_stddev,
            p.q01,
            p.q16,
            p.q50,
            p.q84,
            p.q99,
            p.black_fraction,
            p.saturated_fraction,
            p.mean_abs_neighbor_gradient,
        ];
        for (name, value) in names.into_iter().zip(values) {
            let full = match (prefix, name) {
                ("raw", "mean") => "raw_mean",
                ("raw", "population_stddev") => "raw_population_stddev",
                ("raw", "q01") => "raw_q01",
                ("raw", "q16") => "raw_q16",
                ("raw", "q50") => "raw_q50",
                ("raw", "q84") => "raw_q84",
                ("raw", "q99") => "raw_q99",
                ("raw", "black_fraction") => "raw_black_fraction",
                ("raw", "saturated_fraction") => "raw_saturated_fraction",
                ("raw", _) => "raw_mean_abs_neighbor_gradient",
                ("post", "mean") => "post_mean",
                ("post", "population_stddev") => "post_population_stddev",
                ("post", "q01") => "post_q01",
                ("post", "q16") => "post_q16",
                ("post", "q50") => "post_q50",
                ("post", "q84") => "post_q84",
                ("post", "q99") => "post_q99",
                ("post", "black_fraction") => "post_black_fraction",
                ("post", "saturated_fraction") => "post_saturated_fraction",
                ("post", _) => "post_mean_abs_neighbor_gradient",
                ("final", "mean") => "final_mean",
                ("final", "population_stddev") => "final_population_stddev",
                ("final", "q01") => "final_q01",
                ("final", "q16") => "final_q16",
                ("final", "q50") => "final_q50",
                ("final", "q84") => "final_q84",
                ("final", "q99") => "final_q99",
                ("final", "black_fraction") => "final_black_fraction",
                ("final", "saturated_fraction") => "final_saturated_fraction",
                ("final", _) => "final_mean_abs_neighbor_gradient",
                _ => unreachable!(),
            };
            output.push((full, value));
        }
    }
    let mut output = Vec::with_capacity(32);
    push_plane(&mut output, "raw", &stats.reconstructed_native);
    push_plane(&mut output, "post", &stats.post_despeckle_native);
    push_plane(&mut output, "final", &stats.final_model_channel);
    output.push((
        "despeckle_changed_pixel_fraction",
        stats.despeckle_changed_pixel_fraction,
    ));
    output.push((
        "despeckle_mean_signed_change",
        stats.despeckle_mean_signed_change,
    ));
    output
}

fn coverage_gate(frames: &[&AnalyzedFrame]) -> bool {
    let recognized = frames.iter().filter(|frame| frame.recognized()).count();
    recognized * 100 >= RECOGNITION_PERCENT * frames.len()
}

fn category_from_medians(neutral: f64, soft: f64, max: f64, evaluable: bool) -> PrimaryCategory {
    if !evaluable {
        return PrimaryCategory::NotEvaluable;
    }
    let soft_delta = soft - neutral;
    let max_delta = max - soft;
    if soft_delta <= OPENNESS_DEADBAND {
        PrimaryCategory::SoftTransferNotShown
    } else if max_delta > OPENNESS_DEADBAND {
        PrimaryCategory::Monotone
    } else if max_delta < -OPENNESS_DEADBAND {
        PrimaryCategory::Reversal
    } else {
        PrimaryCategory::Plateau
    }
}

fn selected_frames<'a>(
    frames: &'a [AnalyzedFrame],
    session: &str,
    phase: &str,
    retained_only: bool,
) -> Vec<&'a AnalyzedFrame> {
    frames
        .iter()
        .filter(|frame| {
            frame.prepared_a.session_id == session
                && frame.prepared_a.phase == phase
                && (!retained_only || frame.retained())
        })
        .collect()
}

fn median_openness(frames: &[&AnalyzedFrame], eye: Eye) -> f64 {
    quantile(
        &frames
            .iter()
            .map(|frame| frame.openness(eye))
            .collect::<Vec<_>>(),
        0.5,
    )
    .expect("nonempty finite frame set")
}

fn classify_primary(
    frames: &[AnalyzedFrame],
    sessions: &[String],
) -> (
    Vec<SessionEyePrimary>,
    Vec<SessionEyeCategoryComparison>,
    HighOpenAnnotation,
) {
    let mut rows = Vec::new();
    for session in sessions {
        for eye in Eye::ALL {
            let n = selected_frames(frames, session, "neutral_center", true);
            let s = selected_frames(frames, session, "wide_soft", true);
            let m = selected_frames(frames, session, "wide_max", true);
            let nm = median_openness(&n, eye);
            let sm = median_openness(&s, eye);
            let mm = median_openness(&m, eye);
            let evaluable = coverage_gate(&n) && coverage_gate(&s) && coverage_gate(&m);
            let category = category_from_medians(nm, sm, mm, evaluable);
            let mut block_categories = [PrimaryCategory::NotEvaluable; 5];
            for block in 1..=5 {
                let nb: Vec<_> = selected_frames(frames, session, "neutral_center", false)
                    .into_iter()
                    .filter(|f| f.block() == block)
                    .collect();
                let sb: Vec<_> = selected_frames(frames, session, "wide_soft", false)
                    .into_iter()
                    .filter(|f| f.block() == block)
                    .collect();
                let mb: Vec<_> = selected_frames(frames, session, "wide_max", false)
                    .into_iter()
                    .filter(|f| f.block() == block)
                    .collect();
                let block_evaluable =
                    coverage_gate(&nb) && coverage_gate(&sb) && coverage_gate(&mb);
                block_categories[block - 1] = category_from_medians(
                    median_openness(&nb, eye),
                    median_openness(&sb, eye),
                    median_openness(&mb, eye),
                    block_evaluable,
                );
            }
            let block_agreement =
                if !category.is_evaluable() || block_categories.iter().any(|c| !c.is_evaluable()) {
                    BlockAgreement::NotEvaluable
                } else if block_categories.iter().all(|&c| c == category) {
                    BlockAgreement::AllAgree
                } else {
                    BlockAgreement::Disagree
                };
            rows.push(SessionEyePrimary {
                session_id: session.clone(),
                eye,
                neutral_median: nm,
                soft_median: sm,
                max_median: mm,
                soft_minus_neutral: sm - nm,
                max_minus_soft: mm - sm,
                max_minus_neutral: mm - nm,
                category,
                block_categories,
                block_agreement,
            });
        }
    }
    let comparisons = sessions
        .iter()
        .map(|session| {
            let left = rows
                .iter()
                .find(|r| r.session_id == *session && r.eye == Eye::Left)
                .unwrap()
                .category;
            let right = rows
                .iter()
                .find(|r| r.session_id == *session && r.eye == Eye::Right)
                .unwrap()
                .category;
            let annotation = if !left.is_evaluable() || !right.is_evaluable() {
                EyeCategoryAnnotation::EyeCategoryNotEvaluable
            } else if left == right {
                EyeCategoryAnnotation::EyeCategoryMatched
            } else {
                EyeCategoryAnnotation::EyeCategoryAsymmetric
            };
            SessionEyeCategoryComparison {
                session_id: session.clone(),
                left,
                right,
                annotation,
            }
        })
        .collect();
    let high = if rows.iter().any(|r| r.category == PrimaryCategory::Reversal) {
        HighOpenAnnotation::HighOpenReversalObserved
    } else if rows.iter().any(|r| r.category == PrimaryCategory::Plateau) {
        HighOpenAnnotation::HighOpenPlateauObserved
    } else if rows.len() == sessions.len() * 2
        && rows.iter().all(|r| r.category == PrimaryCategory::Monotone)
    {
        HighOpenAnnotation::HighOpenMonotoneAll
    } else {
        HighOpenAnnotation::HighOpenTransferNotEstablished
    };
    (rows, comparisons, high)
}

fn build_phase_eye_summaries(
    frames: &[AnalyzedFrame],
    sessions: &[String],
) -> (Vec<PhaseEyeSummary>, Vec<CommonDifferentialSummary>) {
    let mut eye_rows = Vec::new();
    let mut stereo_rows = Vec::new();
    for session in sessions {
        for spec in &PHASE_SCHEDULE {
            let selected = selected_frames(frames, session, spec.name, true);
            let common: Vec<_> = selected
                .iter()
                .map(|f| (f.openness(Eye::Left) + f.openness(Eye::Right)) / 2.0)
                .collect();
            let differential: Vec<_> = selected
                .iter()
                .map(|f| (f.openness(Eye::Left) - f.openness(Eye::Right)) / 2.0)
                .collect();
            stereo_rows.push(CommonDifferentialSummary {
                session_id: session.clone(),
                phase: spec.name.into(),
                retained_count: selected.len(),
                common_openness: ScalarSummary::from_values(&common),
                differential_openness: ScalarSummary::from_values(&differential),
            });
            for eye in Eye::ALL {
                let presence: Vec<_> = selected.iter().map(|f| f.outputs()[0]).collect();
                let openness: Vec<_> = selected.iter().map(|f| f.openness(eye)).collect();
                let squeeze: Vec<_> = selected.iter().map(|f| f.squeeze(eye)).collect();
                let mut metrics: BTreeMap<String, Vec<f64>> = BTreeMap::new();
                for frame in &selected {
                    for (name, value) in input_metric_values(frame.stats(eye)) {
                        metrics.entry(name.into()).or_default().push(value);
                    }
                }
                let input_statistics = metrics
                    .into_iter()
                    .map(|(name, values)| (name, ScalarSummary::from_values(&values)))
                    .collect();
                let differences: Vec<_> = openness
                    .windows(2)
                    .map(|pair| (pair[1] - pair[0]).abs())
                    .collect();
                eye_rows.push(PhaseEyeSummary {
                    session_id: session.clone(),
                    phase: spec.name.into(),
                    eye,
                    retained_count: selected.len(),
                    recognized_count: selected.iter().filter(|f| f.recognized()).count(),
                    recognition_gate_passed: coverage_gate(&selected),
                    presence: ScalarSummary::from_values(&presence),
                    openness: ScalarSummary::from_values(&openness),
                    squeeze: ScalarSummary::from_values(&squeeze),
                    input_statistics,
                    openness_first_difference: FirstDifferenceSummary {
                        count: differences.len(),
                        median_absolute: quantile(&differences, 0.5),
                        q95_absolute: quantile(&differences, 0.95),
                    },
                });
            }
        }
    }
    (eye_rows, stereo_rows)
}

fn endpoint_annotation(evaluable: bool, difference: f64) -> String {
    if !evaluable {
        "NOT_EVALUABLE".into()
    } else if difference > OPENNESS_DEADBAND {
        "PASS".into()
    } else {
        "FAIL".into()
    }
}

fn median_simultaneous_openness_difference(
    frames: &[&AnalyzedFrame],
    nonwink_eye: Eye,
    wink_eye: Eye,
) -> f64 {
    let nonwink: Vec<_> = frames
        .iter()
        .map(|frame| frame.openness(nonwink_eye))
        .collect();
    let wink: Vec<_> = frames
        .iter()
        .map(|frame| frame.openness(wink_eye))
        .collect();
    median_paired_difference(&nonwink, &wink).expect("nonempty finite simultaneous wink stratum")
}

fn median_paired_difference(nonwink: &[f64], wink: &[f64]) -> Option<f64> {
    if nonwink.len() != wink.len() || nonwink.is_empty() {
        return None;
    }
    let paired_differences: Vec<_> = nonwink.iter().zip(wink).map(|(a, b)| a - b).collect();
    quantile(&paired_differences, 0.5)
}

fn build_endpoint_checks(frames: &[AnalyzedFrame], sessions: &[String]) -> Vec<EndpointCheck> {
    let mut rows = Vec::new();
    for session in sessions {
        let neutral = selected_frames(frames, session, "neutral_center", true);
        let wide = selected_frames(frames, session, "wide_max", true);
        let closed = selected_frames(frames, session, "closed_negative", true);
        let left_wink = selected_frames(frames, session, "left_wink_negative", true);
        let right_wink = selected_frames(frames, session, "right_wink_negative", true);
        for eye in Eye::ALL {
            let neutral_median = median_openness(&neutral, eye);
            let closed_median = median_openness(&closed, eye);
            let difference = neutral_median - closed_median;
            rows.push(EndpointCheck {
                session_id: session.clone(),
                eye,
                check: "neutral_center_minus_closed_negative".into(),
                reference_median: closed_median,
                contrast_median: neutral_median,
                difference,
                annotation: endpoint_annotation(
                    coverage_gate(&neutral) && coverage_gate(&closed),
                    difference,
                ),
            });
            let wide_median = median_openness(&wide, eye);
            let difference = wide_median - neutral_median;
            rows.push(EndpointCheck {
                session_id: session.clone(),
                eye,
                check: "wide_max_minus_neutral_center".into(),
                reference_median: neutral_median,
                contrast_median: wide_median,
                difference,
                annotation: endpoint_annotation(
                    coverage_gate(&neutral) && coverage_gate(&wide),
                    difference,
                ),
            });
        }
        let left_closed = median_openness(&left_wink, Eye::Left);
        let left_open = median_openness(&left_wink, Eye::Right);
        let left_difference =
            median_simultaneous_openness_difference(&left_wink, Eye::Right, Eye::Left);
        rows.push(EndpointCheck {
            session_id: session.clone(),
            eye: Eye::Left,
            check: "left_wink_simultaneous_nonwink_minus_wink".into(),
            reference_median: left_closed,
            contrast_median: left_open,
            difference: left_difference,
            annotation: endpoint_annotation(coverage_gate(&left_wink), left_difference),
        });
        let right_closed = median_openness(&right_wink, Eye::Right);
        let right_open = median_openness(&right_wink, Eye::Left);
        let right_difference =
            median_simultaneous_openness_difference(&right_wink, Eye::Left, Eye::Right);
        rows.push(EndpointCheck {
            session_id: session.clone(),
            eye: Eye::Right,
            check: "right_wink_simultaneous_nonwink_minus_wink".into(),
            reference_median: right_closed,
            contrast_median: right_open,
            difference: right_difference,
            annotation: endpoint_annotation(coverage_gate(&right_wink), right_difference),
        });
    }
    rows
}

fn stats_metric_names() -> Vec<&'static str> {
    input_metric_values(&EyeInputStats::default())
        .into_iter()
        .map(|(name, _)| name)
        .collect()
}

fn write_frames_csv(frames: &[AnalyzedFrame]) -> Vec<u8> {
    let metric_names = stats_metric_names();
    let mut output = String::new();
    let mut header = vec![
        "session_id",
        "phase",
        "sequence",
        "label_bits",
        "retained",
        "association",
        "block",
        "recognized",
        "preflight_a_left_native_sha256",
        "preflight_b_left_native_sha256",
        "preflight_a_right_native_sha256",
        "preflight_b_right_native_sha256",
        "preflight_a_tensor_sha256",
        "preflight_b_tensor_sha256",
        "model_a_presence_bits",
        "model_a_left_openness_bits",
        "model_a_right_openness_bits",
        "model_a_left_squeeze_bits",
        "model_a_right_squeeze_bits",
        "model_b_presence_bits",
        "model_b_left_openness_bits",
        "model_b_right_openness_bits",
        "model_b_left_squeeze_bits",
        "model_b_right_squeeze_bits",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    for eye in ["left", "right"] {
        for metric in &metric_names {
            header.push(format!("{eye}_{metric}"));
        }
    }
    output.push_str(&header.join(","));
    output.push('\n');
    for frame in frames {
        let p = &frame.prepared_a;
        let fields = [
            p.session_id.clone(),
            p.phase.clone(),
            p.sequence.to_string(),
            p.label_bits.to_string(),
            frame.retained().to_string(),
            frame.association().to_string(),
            frame.block().to_string(),
            frame.recognized().to_string(),
            p.left_native_sha256.clone(),
            frame.prepared_b.left_native_sha256.clone(),
            p.right_native_sha256.clone(),
            frame.prepared_b.right_native_sha256.clone(),
            p.tensor_sha256.clone(),
            frame.prepared_b.tensor_sha256.clone(),
        ];
        output.push_str(&fields.join(","));
        for bits in frame
            .model_a
            .output_bits
            .into_iter()
            .chain(frame.model_b.output_bits)
        {
            write!(output, ",{bits}").unwrap();
        }
        for eye in Eye::ALL {
            for (_, value) in input_metric_values(frame.stats(eye)) {
                write!(output, ",{value}").unwrap();
            }
        }
        output.push('\n');
    }
    output.into_bytes()
}

fn frame_metric_values(frame: &AnalyzedFrame, eye: Eye) -> Vec<(&'static str, f64)> {
    let mut metrics = vec![
        ("presence", frame.outputs()[0]),
        ("openness", frame.openness(eye)),
        ("squeeze", frame.squeeze(eye)),
    ];
    metrics.extend(input_metric_values(frame.stats(eye)));
    metrics
}

fn write_temporal_blocks_csv(frames: &[AnalyzedFrame], sessions: &[String]) -> Vec<u8> {
    let mut output = String::from(
        "session_id,phase,eye,metric,block,block_count,recognized_count,recognition_gate_passed,median,block5_minus_block1,block_median_range\n",
    );
    for session in sessions {
        for spec in &PHASE_SCHEDULE {
            let full = selected_frames(frames, session, spec.name, false);
            for eye in Eye::ALL {
                let metric_names: Vec<_> = frame_metric_values(full[0], eye)
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                for metric in metric_names {
                    let mut block_medians = Vec::with_capacity(5);
                    let mut blocks = Vec::with_capacity(5);
                    for block in 1..=5 {
                        let block_frames: Vec<_> = full
                            .iter()
                            .copied()
                            .filter(|f| f.block() == block)
                            .collect();
                        let values: Vec<_> = block_frames
                            .iter()
                            .map(|frame| {
                                frame_metric_values(frame, eye)
                                    .into_iter()
                                    .find(|(name, _)| *name == metric)
                                    .unwrap()
                                    .1
                            })
                            .collect();
                        block_medians.push(quantile(&values, 0.5).unwrap());
                        blocks.push(block_frames);
                    }
                    let delta = block_medians[4] - block_medians[0];
                    let range = block_medians
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max)
                        - block_medians.iter().copied().fold(f64::INFINITY, f64::min);
                    for block in 1..=5 {
                        let block_frames = &blocks[block - 1];
                        let row = [
                            session.clone(),
                            spec.name.into(),
                            eye.code().into(),
                            metric.into(),
                            block.to_string(),
                            block_frames.len().to_string(),
                            block_frames
                                .iter()
                                .filter(|f| f.recognized())
                                .count()
                                .to_string(),
                            coverage_gate(block_frames).to_string(),
                            block_medians[block - 1].to_string(),
                            delta.to_string(),
                            range.to_string(),
                        ];
                        output.push_str(&row.join(","));
                        output.push('\n');
                    }
                }
            }
        }
    }
    output.into_bytes()
}

fn association_metric_values(stats: &EyeInputStats) -> [(&'static str, f64); 10] {
    [
        ("raw_mean", stats.reconstructed_native.mean),
        (
            "raw_population_stddev",
            stats.reconstructed_native.population_stddev,
        ),
        (
            "raw_mean_abs_neighbor_gradient",
            stats.reconstructed_native.mean_abs_neighbor_gradient,
        ),
        ("post_mean", stats.post_despeckle_native.mean),
        (
            "post_population_stddev",
            stats.post_despeckle_native.population_stddev,
        ),
        (
            "post_mean_abs_neighbor_gradient",
            stats.post_despeckle_native.mean_abs_neighbor_gradient,
        ),
        ("final_mean", stats.final_model_channel.mean),
        (
            "final_population_stddev",
            stats.final_model_channel.population_stddev,
        ),
        (
            "final_mean_abs_neighbor_gradient",
            stats.final_model_channel.mean_abs_neighbor_gradient,
        ),
        (
            "despeckle_changed_pixel_fraction",
            stats.despeckle_changed_pixel_fraction,
        ),
    ]
}

fn write_associations_csv(frames: &[AnalyzedFrame], sessions: &[String]) -> Vec<u8> {
    let mut output = String::from("session_id,phase,eye,n,metric,spearman\n");
    for session in sessions {
        for spec in &PHASE_SCHEDULE {
            let selected: Vec<_> = selected_frames(frames, session, spec.name, true)
                .into_iter()
                .filter(|frame| frame.association())
                .collect();
            for eye in Eye::ALL {
                let openness: Vec<_> = selected.iter().map(|frame| frame.openness(eye)).collect();
                for metric_index in 0..10 {
                    let name = association_metric_values(selected[0].stats(eye))[metric_index].0;
                    let values: Vec<_> = selected
                        .iter()
                        .map(|frame| association_metric_values(frame.stats(eye))[metric_index].1)
                        .collect();
                    let correlation = spearman(&openness, &values)
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    let row = [
                        session.clone(),
                        spec.name.into(),
                        eye.code().into(),
                        selected.len().to_string(),
                        name.into(),
                        correlation,
                    ];
                    output.push_str(&row.join(","));
                    output.push('\n');
                }
            }
        }
    }
    output.into_bytes()
}

#[derive(Clone, Copy)]
struct GazeContrast {
    kind: &'static str,
    base: &'static str,
    contrast: &'static str,
    ratio_to_soft: bool,
    wide_order: bool,
}

const GAZE_CONTRASTS: [GazeContrast; 8] = [
    GazeContrast {
        kind: "same_label_gaze",
        base: "neutral_center",
        contrast: "gaze_up_neutral",
        ratio_to_soft: true,
        wide_order: false,
    },
    GazeContrast {
        kind: "same_label_gaze",
        base: "neutral_center",
        contrast: "gaze_down_neutral",
        ratio_to_soft: true,
        wide_order: false,
    },
    GazeContrast {
        kind: "same_label_gaze",
        base: "neutral_center",
        contrast: "gaze_left_neutral",
        ratio_to_soft: true,
        wide_order: false,
    },
    GazeContrast {
        kind: "same_label_gaze",
        base: "neutral_center",
        contrast: "gaze_right_neutral",
        ratio_to_soft: true,
        wide_order: false,
    },
    GazeContrast {
        kind: "same_label_gaze",
        base: "wide_max",
        contrast: "wide_gaze_up",
        ratio_to_soft: false,
        wide_order: false,
    },
    GazeContrast {
        kind: "same_label_gaze",
        base: "wide_max",
        contrast: "wide_gaze_down",
        ratio_to_soft: false,
        wide_order: false,
    },
    GazeContrast {
        kind: "within_gaze_wide_order",
        base: "gaze_up_neutral",
        contrast: "wide_gaze_up",
        ratio_to_soft: false,
        wide_order: true,
    },
    GazeContrast {
        kind: "within_gaze_wide_order",
        base: "gaze_down_neutral",
        contrast: "wide_gaze_down",
        ratio_to_soft: false,
        wide_order: true,
    },
];

fn median_named_metric(frames: &[&AnalyzedFrame], eye: Eye, metric: &str) -> f64 {
    let values: Vec<_> = frames
        .iter()
        .map(|frame| {
            if metric == "openness" {
                frame.openness(eye)
            } else {
                input_metric_values(frame.stats(eye))
                    .into_iter()
                    .find(|(name, _)| *name == metric)
                    .unwrap()
                    .1
            }
        })
        .collect();
    quantile(&values, 0.5).unwrap()
}

fn write_gaze_and_session_differences_csv(
    frames: &[AnalyzedFrame],
    sessions: &[String],
    primary: &[SessionEyePrimary],
) -> Vec<u8> {
    let mut output = String::from(
        "kind,session_id,reference_session_id,eye,base_phase,contrast_phase,metric,base_median,contrast_median,difference,ratio,annotation\n",
    );
    let mut metrics = vec!["openness"];
    metrics.extend(stats_metric_names());
    for session in sessions {
        for contrast in GAZE_CONTRASTS {
            let base = selected_frames(frames, session, contrast.base, true);
            let next = selected_frames(frames, session, contrast.contrast, true);
            let evaluable = coverage_gate(&base) && coverage_gate(&next);
            for eye in Eye::ALL {
                let primary_row = primary
                    .iter()
                    .find(|r| r.session_id == *session && r.eye == eye)
                    .unwrap();
                for metric in &metrics {
                    let a = median_named_metric(&base, eye, metric);
                    let b = median_named_metric(&next, eye, metric);
                    let difference = b - a;
                    let ratio = if *metric == "openness"
                        && contrast.ratio_to_soft
                        && primary_row.category.is_evaluable()
                        && primary_row.soft_minus_neutral.abs() > OPENNESS_DEADBAND
                    {
                        Some(difference.abs() / primary_row.soft_minus_neutral.abs())
                    } else {
                        None
                    };
                    let annotation = if *metric != "openness" {
                        String::new()
                    } else if !evaluable {
                        "NOT_EVALUABLE".into()
                    } else if contrast.wide_order {
                        if difference > OPENNESS_DEADBAND {
                            "WIDE_ORDER_SHOWN".into()
                        } else {
                            "WIDE_ORDER_NOT_SHOWN".into()
                        }
                    } else if difference.abs() > OPENNESS_DEADBAND {
                        "POSE_OR_ORDER_SENSITIVE".into()
                    } else {
                        "WITHIN_DEADBAND".into()
                    };
                    let row = [
                        contrast.kind.into(),
                        session.clone(),
                        String::new(),
                        eye.code().into(),
                        contrast.base.into(),
                        contrast.contrast.into(),
                        (*metric).into(),
                        a.to_string(),
                        b.to_string(),
                        difference.to_string(),
                        ratio.map(|v| v.to_string()).unwrap_or_default(),
                        annotation,
                    ];
                    output.push_str(&row.join(","));
                    output.push('\n');
                }
            }
        }
    }
    if let Some(reference) = sessions.first() {
        for later in sessions.iter().skip(1) {
            for spec in &PHASE_SCHEDULE {
                let base = selected_frames(frames, reference, spec.name, true);
                let next = selected_frames(frames, later, spec.name, true);
                for eye in Eye::ALL {
                    for metric in &metrics {
                        let a = median_named_metric(&base, eye, metric);
                        let b = median_named_metric(&next, eye, metric);
                        let row = [
                            "session_difference_reseat_time_expression_confounded".into(),
                            later.clone(),
                            reference.clone(),
                            eye.code().into(),
                            spec.name.into(),
                            spec.name.into(),
                            (*metric).into(),
                            a.to_string(),
                            b.to_string(),
                            (b - a).to_string(),
                            String::new(),
                            String::new(),
                        ];
                        output.push_str(&row.join(","));
                        output.push('\n');
                    }
                }
            }
        }
    }
    output.into_bytes()
}

fn serde_payload<T: Serialize>(value: &T) -> Vec<u8> {
    let mut bytes = serde_json::to_vec_pretty(value).expect("finite JSON-ready analysis artifact");
    bytes.push(b'\n');
    bytes
}

fn analysis_metadata(
    plan: &BuiltInputPlan,
    model_acquisition: &ModelAcquisitionMetadata,
    terminal_status: AnalysisTerminalStatus,
    high_open_annotation: Option<HighOpenAnnotation>,
    model_loaded: bool,
    preprocessing_bit_identical: Option<bool>,
    model_bit_identical: Option<bool>,
    digests: AnalysisDigests,
) -> AnalysisMetadata {
    AnalysisMetadata {
        terminal_status,
        high_open_annotation,
        model_loaded,
        model_byte_len: model_acquisition.observed_byte_len.unwrap_or(0),
        model_sha256: model_acquisition
            .observed_sha256
            .clone()
            .unwrap_or_default(),
        model_loader_invoked: model_acquisition.loader_invoked,
        model_bytes_obtained: model_acquisition.model_bytes_obtained,
        model_loader_error_kind: model_acquisition.loader_error_kind,
        completed_session_count: plan.runtime.completed_sessions.len(),
        frame_count: plan
            .runtime
            .completed_sessions
            .iter()
            .map(|s| s.cases.len())
            .sum(),
        retained_pairs_per_session: RETAINED_PAIRS_PER_SESSION,
        association_pairs_per_session: ASSOCIATION_PAIRS_PER_SESSION,
        recording_tree_sha256: plan.recording_tree_sha256.clone(),
        preprocessing_bit_identical,
        model_bit_identical,
        digests,
        independence: "independence_unproven".into(),
        preprocessing: FrozenPreprocessingMetadata::observed(),
    }
}

fn reduced_outcome(
    plan: &BuiltInputPlan,
    model_acquisition: &ModelAcquisitionMetadata,
    status: AnalysisTerminalStatus,
    reason: impl Into<String>,
    model_loaded: bool,
    pass_failure: Option<PassFailure>,
    preprocessing_mismatches: Vec<PreprocessMismatch>,
    determinism_mismatches: Vec<DeterminismMismatch>,
    nonfinite_model_frames: Vec<NonfiniteModelFrame>,
    digests: AnalysisDigests,
    preprocessing_bit_identical: Option<bool>,
    model_bit_identical: Option<bool>,
) -> AnalysisOutcome {
    debug_assert_ne!(status, AnalysisTerminalStatus::AuditComplete);
    let reason = reason.into();
    let publication_forbidden = pass_failure
        .as_ref()
        .is_some_and(|failure| failure.untrusted_input_change);
    let diagnostic = DiagnosticArtifact {
        schema: "sranibro.phase14.diagnostic.v1".into(),
        terminal_status: status,
        reason: reason.clone(),
        model_loaded,
        pass_failure,
        preprocessing_mismatches,
        determinism_mismatches,
        nonfinite_model_frames,
        digests: digests.clone(),
        high_open_annotation: None,
    };
    let interpretation = format!(
        "Phase 1.4 XR5 transfer audit did not reach a scientific result.\nterminal_status={status:?}\nreason={reason}\nmodel_loaded={model_loaded}\nhigh_open_annotation=null\n"
    );
    let payloads = BTreeMap::from([
        ("diagnostic.json".into(), serde_payload(&diagnostic)),
        ("interpretation.txt".into(), interpretation.into_bytes()),
    ]);
    let metadata = analysis_metadata(
        plan,
        model_acquisition,
        status,
        None,
        model_loaded,
        preprocessing_bit_identical,
        model_bit_identical,
        digests,
    );
    AnalysisOutcome {
        terminal_status: status,
        high_open_annotation: None,
        payloads,
        metadata,
        diagnostic: Some(diagnostic),
        publication_forbidden,
    }
}

fn model_preflight_mismatches(
    preflight: &PreflightPass,
    model_a: &ModelPass,
    model_b: &ModelPass,
) -> Vec<DeterminismMismatch> {
    let prep: BTreeMap<_, _> = preflight
        .frames
        .iter()
        .map(|f| (f.case_key.as_str(), f))
        .collect();
    let a: BTreeMap<_, _> = model_a
        .frames
        .iter()
        .map(|f| (f.case_key.as_str(), f))
        .collect();
    let b: BTreeMap<_, _> = model_b
        .frames
        .iter()
        .map(|f| (f.case_key.as_str(), f))
        .collect();
    let keys: BTreeSet<_> = prep
        .keys()
        .chain(a.keys())
        .chain(b.keys())
        .copied()
        .collect();
    let mut mismatches = Vec::new();
    for key in keys {
        let expected = prep.get(key).copied();
        let av = a.get(key).copied();
        let bv = b.get(key).copied();
        if expected.is_none() || av.is_none() || bv.is_none() {
            mismatches.push(DeterminismMismatch {
                case_key: key.into(),
                kind: "missing_case_identity".into(),
                expected: expected.map(|_| "present".into()),
                observed: Some(format!("model_a={} model_b={}", av.is_some(), bv.is_some())),
                model_a_bits: av.map(|v| v.output_bits),
                model_b_bits: bv.map(|v| v.output_bits),
            });
            continue;
        }
        let expected = expected.unwrap();
        let av = av.unwrap();
        let bv = bv.unwrap();
        for (kind, expected_hash, a_hash, b_hash) in [
            (
                "left_native",
                &expected.left_native_sha256,
                &av.left_native_sha256,
                &bv.left_native_sha256,
            ),
            (
                "right_native",
                &expected.right_native_sha256,
                &av.right_native_sha256,
                &bv.right_native_sha256,
            ),
            (
                "tensor",
                &expected.tensor_sha256,
                &av.tensor_sha256,
                &bv.tensor_sha256,
            ),
        ] {
            if a_hash != expected_hash {
                mismatches.push(DeterminismMismatch {
                    case_key: key.into(),
                    kind: format!("model_a_{kind}_hash"),
                    expected: Some(expected_hash.clone()),
                    observed: Some(a_hash.clone()),
                    model_a_bits: Some(av.output_bits),
                    model_b_bits: Some(bv.output_bits),
                });
            }
            if b_hash != expected_hash {
                mismatches.push(DeterminismMismatch {
                    case_key: key.into(),
                    kind: format!("model_b_{kind}_hash"),
                    expected: Some(expected_hash.clone()),
                    observed: Some(b_hash.clone()),
                    model_a_bits: Some(av.output_bits),
                    model_b_bits: Some(bv.output_bits),
                });
            }
        }
        if av.output_bits != bv.output_bits {
            mismatches.push(DeterminismMismatch {
                case_key: key.into(),
                kind: "model_output_bits".into(),
                expected: None,
                observed: None,
                model_a_bits: Some(av.output_bits),
                model_b_bits: Some(bv.output_bits),
            });
        }
    }
    mismatches
}

fn join_analyzed_frames(
    preflight_a: PreflightPass,
    preflight_b: PreflightPass,
    model_a: ModelPass,
    model_b: ModelPass,
) -> Vec<AnalyzedFrame> {
    let bprep: BTreeMap<_, _> = preflight_b
        .frames
        .into_iter()
        .map(|f| (f.case_key.clone(), f))
        .collect();
    let amap: BTreeMap<_, _> = model_a
        .frames
        .into_iter()
        .map(|f| (f.case_key.clone(), f))
        .collect();
    let bmap: BTreeMap<_, _> = model_b
        .frames
        .into_iter()
        .map(|f| (f.case_key.clone(), f))
        .collect();
    preflight_a
        .frames
        .into_iter()
        .map(|prepared_a| {
            let key = &prepared_a.case_key;
            AnalyzedFrame {
                prepared_b: bprep.get(key).unwrap().clone(),
                model_a: amap.get(key).unwrap().clone(),
                model_b: bmap.get(key).unwrap().clone(),
                prepared_a,
            }
        })
        .collect()
}

struct CheckedPreflights {
    first: PreflightPass,
    second: PreflightPass,
    digests: AnalysisDigests,
}

fn run_checked_preflights(plan: &BuiltInputPlan) -> Result<CheckedPreflights, AnalysisOutcome> {
    let mut digests = AnalysisDigests::default();
    let not_invoked = ModelAcquisitionMetadata::not_invoked();
    if plan.terminal_status() != InputTerminalStatus::InputSealed {
        let mut outcome = reduced_outcome(
            plan,
            &not_invoked,
            AnalysisTerminalStatus::InputInvalid,
            "sealed_input_terminal_status_is_not_INPUT_SEALED",
            false,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digests,
            None,
            None,
        );
        outcome.publication_forbidden = true;
        return Err(outcome);
    }
    let expected_frames = plan.runtime.completed_sessions.len() * PAIRS_PER_SESSION;
    if plan.runtime.completed_sessions.len() < 2
        || plan
            .runtime
            .completed_sessions
            .iter()
            .any(|session| session.cases.len() != PAIRS_PER_SESSION)
    {
        let mut outcome = reduced_outcome(
            plan,
            &not_invoked,
            AnalysisTerminalStatus::InputInvalid,
            "runtime_session_plan_count_mismatch",
            false,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digests,
            None,
            None,
        );
        outcome.publication_forbidden = true;
        return Err(outcome);
    }

    let preflight_a = match run_preflight_pass(plan, "preflight_a") {
        Ok(pass) => pass,
        Err(failure) => {
            return Err(reduced_outcome(
                plan,
                &not_invoked,
                AnalysisTerminalStatus::InputInvalid,
                "first_decode_preprocess_preflight_failed",
                false,
                Some(failure),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                None,
                None,
            ))
        }
    };
    digests.preflight_a_ordered_stream_sha256 = Some(preflight_a.ordered_stream_sha256.clone());
    let preflight_b = match run_preflight_pass(plan, "preflight_b") {
        Ok(pass) => pass,
        Err(failure) => {
            return Err(reduced_outcome(
                plan,
                &not_invoked,
                AnalysisTerminalStatus::InputInvalid,
                "second_decode_preprocess_preflight_failed",
                false,
                Some(failure),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                None,
                None,
            ))
        }
    };
    digests.preflight_b_ordered_stream_sha256 = Some(preflight_b.ordered_stream_sha256.clone());
    if preflight_a.frames.len() != expected_frames || preflight_b.frames.len() != expected_frames {
        return Err(reduced_outcome(
            plan,
            &not_invoked,
            AnalysisTerminalStatus::InputInvalid,
            "preflight_frame_count_mismatch",
            false,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digests,
            None,
            None,
        ));
    }
    let preprocess_mismatches = preprocessing_mismatches(&preflight_a, &preflight_b);
    if !preprocess_mismatches.is_empty() {
        return Err(reduced_outcome(
            plan,
            &not_invoked,
            AnalysisTerminalStatus::InconclusiveDeterminism,
            "independent_preprocessing_passes_differ",
            false,
            None,
            preprocess_mismatches,
            Vec::new(),
            Vec::new(),
            digests,
            Some(false),
            None,
        ));
    }

    Ok(CheckedPreflights {
        first: preflight_a,
        second: preflight_b,
        digests,
    })
}

/// Production orchestration. The loader is invoked exactly once, and only after
/// two complete sealed PNG decode/preprocess passes have matched case-for-case.
/// It must return bytes read under the caller's ordinary-file identity guard.
pub fn analyze_with_model_loader<F>(plan: &BuiltInputPlan, loader: F) -> AnalysisOutcome
where
    F: FnOnce() -> Result<Vec<u8>, ModelLoaderError>,
{
    let preflights = match run_checked_preflights(plan) {
        Ok(preflights) => preflights,
        Err(outcome) => return outcome,
    };
    continue_after_checked_preflights(plan, preflights, loader)
}

fn continue_after_checked_preflights<F>(
    plan: &BuiltInputPlan,
    preflights: CheckedPreflights,
    loader: F,
) -> AnalysisOutcome
where
    F: FnOnce() -> Result<Vec<u8>, ModelLoaderError>,
{
    let CheckedPreflights {
        first: preflight_a,
        second: preflight_b,
        digests,
    } = preflights;
    let model_bytes = match loader() {
        Ok(bytes) => bytes,
        Err(error) => {
            let acquisition = ModelAcquisitionMetadata::failed(&error);
            let mut outcome = reduced_outcome(
                plan,
                &acquisition,
                AnalysisTerminalStatus::InconclusiveArtifact,
                format!("model_loader_failed:{error}"),
                false,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            );
            if error.kind == ModelLoaderErrorKind::ChangedDuringRead {
                outcome.publication_forbidden = true;
            }
            return outcome;
        }
    };
    let acquisition = ModelAcquisitionMetadata::obtained(&model_bytes);
    finish_analysis_after_preflights(
        plan,
        &model_bytes,
        &acquisition,
        preflight_a,
        preflight_b,
        digests,
    )
}

fn finish_analysis_after_preflights(
    plan: &BuiltInputPlan,
    model_bytes: &[u8],
    acquisition: &ModelAcquisitionMetadata,
    preflight_a: PreflightPass,
    preflight_b: PreflightPass,
    mut digests: AnalysisDigests,
) -> AnalysisOutcome {
    if model_bytes.len() as u64 != EXPECTED_MODEL_LEN
        || sha256_hex(model_bytes) != EXPECTED_MODEL_SHA256
    {
        return reduced_outcome(
            plan,
            acquisition,
            AnalysisTerminalStatus::InconclusiveArtifact,
            "fixed_model_identity_mismatch",
            false,
            None,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            digests,
            Some(true),
            None,
        );
    }

    let first_map = match tvm_params::parse_map_bytes(model_bytes) {
        Ok(map) => map,
        Err(error) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveArtifact,
                format!("model_a_parse_failed:{error}"),
                false,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            )
        }
    };
    let mut first = match EyeNet::new(first_map) {
        Ok(net) => net,
        Err(error) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveArtifact,
                format!("model_a_load_failed:{error}"),
                false,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            )
        }
    };
    let model_a = match run_model_pass(plan, &mut first, "model_a_forward", false) {
        Ok(pass) => pass,
        Err(failure) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveDeterminism,
                "model_a_could_not_reconstruct_sealed_case",
                true,
                Some(failure),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            )
        }
    };
    digests.model_a_ordered_stream_sha256 = Some(model_a.ordered_stream_sha256.clone());
    drop(first);

    let second_map = match tvm_params::parse_map_bytes(model_bytes) {
        Ok(map) => map,
        Err(error) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveArtifact,
                format!("model_b_parse_failed:{error}"),
                true,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            )
        }
    };
    let mut second = match EyeNet::new(second_map) {
        Ok(net) => net,
        Err(error) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveArtifact,
                format!("model_b_load_failed:{error}"),
                true,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                None,
            )
        }
    };
    let model_b = match run_model_pass(plan, &mut second, "model_b_reverse", true) {
        Ok(pass) => pass,
        Err(failure) => {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InconclusiveDeterminism,
                "model_b_could_not_reconstruct_sealed_case",
                true,
                Some(failure),
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                Some(false),
            )
        }
    };
    digests.model_b_ordered_stream_sha256 = Some(model_b.ordered_stream_sha256.clone());
    drop(second);

    let determinism_mismatches = model_preflight_mismatches(&preflight_a, &model_a, &model_b);
    if !determinism_mismatches.is_empty() {
        return reduced_outcome(
            plan,
            acquisition,
            AnalysisTerminalStatus::InconclusiveDeterminism,
            "model_reconstruction_or_output_bits_differ",
            true,
            None,
            Vec::new(),
            determinism_mismatches,
            Vec::new(),
            digests,
            Some(true),
            Some(false),
        );
    }
    let nonfinite: Vec<_> = model_a
        .frames
        .iter()
        .filter_map(|frame| {
            let finite = frame
                .output_bits
                .iter()
                .all(|&bits| f32::from_bits(bits).is_finite());
            (!finite).then(|| NonfiniteModelFrame {
                case_key: frame.case_key.clone(),
                output_bits: frame.output_bits,
            })
        })
        .collect();
    if !nonfinite.is_empty() {
        return reduced_outcome(
            plan,
            acquisition,
            AnalysisTerminalStatus::InconclusiveArtifact,
            "nonfinite_model_output",
            true,
            None,
            Vec::new(),
            Vec::new(),
            nonfinite,
            digests,
            Some(true),
            Some(true),
        );
    }

    let frames = join_analyzed_frames(preflight_a, preflight_b, model_a, model_b);
    let sessions: Vec<_> = plan
        .runtime
        .completed_sessions
        .iter()
        .map(|s| s.session_id.clone())
        .collect();
    for session in &sessions {
        let session_frames: Vec<_> = frames
            .iter()
            .filter(|f| f.prepared_a.session_id == *session)
            .collect();
        let retained = session_frames.iter().filter(|f| f.retained()).count();
        let association = session_frames.iter().filter(|f| f.association()).count();
        if session_frames.len() != PAIRS_PER_SESSION
            || retained != RETAINED_PAIRS_PER_SESSION
            || association != ASSOCIATION_PAIRS_PER_SESSION
        {
            return reduced_outcome(
                plan,
                acquisition,
                AnalysisTerminalStatus::InputInvalid,
                format!("frozen_selection_count_mismatch:{session}"),
                true,
                None,
                Vec::new(),
                Vec::new(),
                Vec::new(),
                digests,
                Some(true),
                Some(true),
            );
        }
    }

    let (primary, eye_comparison, high_open) = classify_primary(&frames, &sessions);
    let (phase_eye_summaries, common_differential_summaries) =
        build_phase_eye_summaries(&frames, &sessions);
    let endpoint_checks = build_endpoint_checks(&frames, &sessions);
    let phase_summaries = PhaseSummariesArtifact {
        schema: "sranibro.phase14.phase-summaries.v1".into(),
        recognition_threshold: RECOGNITION_THRESHOLD,
        recognition_gate_percent: RECOGNITION_PERCENT,
        openness_deadband: OPENNESS_DEADBAND,
        high_open_annotation: high_open,
        session_eye_primary: primary.clone(),
        session_eye_category_comparison: eye_comparison,
        phase_eye_summaries,
        common_differential_summaries,
        endpoint_checks,
        independence: "independence_unproven".into(),
    };
    let interpretation = format!(
        "Phase 1.4 XR5 real-recording transfer audit complete.\nterminal_status=AUDIT_COMPLETE\nhigh_open_annotation={high_open:?}\nsessions={}\nframes={}\nindependence_unproven\nThis is an in-sample, one-user/one-device descriptive audit. Fixed capture order confounds pose with order/time; 30 Hz frames cannot establish 120 Hz dynamics or hour-scale stability. No model, geometry, calibration, or production state was changed.\n",
        sessions.len(), frames.len(),
    );
    let payloads = BTreeMap::from([
        ("frames.csv".into(), write_frames_csv(&frames)),
        (
            "phase_summaries.json".into(),
            serde_payload(&phase_summaries),
        ),
        (
            "temporal_blocks.csv".into(),
            write_temporal_blocks_csv(&frames, &sessions),
        ),
        (
            "associations.csv".into(),
            write_associations_csv(&frames, &sessions),
        ),
        (
            "gaze_and_session_differences.csv".into(),
            write_gaze_and_session_differences_csv(&frames, &sessions, &primary),
        ),
        ("interpretation.txt".into(), interpretation.into_bytes()),
    ]);
    let metadata = analysis_metadata(
        plan,
        acquisition,
        AnalysisTerminalStatus::AuditComplete,
        Some(high_open),
        true,
        Some(true),
        Some(true),
        digests,
    );
    AnalysisOutcome {
        terminal_status: AnalysisTerminalStatus::AuditComplete,
        high_open_annotation: Some(high_open),
        payloads,
        metadata,
        diagnostic: None,
        publication_forbidden: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(label: &str) -> Self {
            let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "sranibro_phase14_{label}_{}_{}",
                std::process::id(),
                sequence
            ));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn valid_csv() -> Vec<u8> {
        let mut csv = String::from("filename,wide,phase,side\n");
        for sequence in 0..PAIRS_PER_SESSION {
            let spec = phase_for_sequence(sequence).unwrap();
            for side in ['l', 'r'] {
                writeln!(
                    csv,
                    "{},{},{},{}",
                    expected_filename(spec, side, sequence),
                    spec.label,
                    spec.name,
                    side
                )
                .unwrap();
            }
        }
        csv.into_bytes()
    }

    fn encode_png(color: png::ColorType, width: u32, height: u32, bytes: &[u8]) -> Vec<u8> {
        let mut output = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut output, width, height);
            encoder.set_color(color);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(bytes)
                .unwrap();
        }
        output
    }

    fn dummy_stats(value: f64) -> EyeInputStats {
        let plane = PlaneStats {
            mean: value,
            population_stddev: value,
            q01: value,
            q16: value,
            q50: value,
            q84: value,
            q99: value,
            black_fraction: value,
            saturated_fraction: value,
            mean_abs_neighbor_gradient: value,
        };
        EyeInputStats {
            reconstructed_native: plane,
            post_despeckle_native: plane,
            final_model_channel: plane,
            despeckle_changed_pixel_fraction: value,
            despeckle_mean_signed_change: value,
        }
    }

    fn analyzed_frame(
        session: &str,
        spec: &PhaseSpec,
        local: usize,
        left: f32,
        right: f32,
        presence: f32,
    ) -> AnalyzedFrame {
        let sequence = spec.first_sequence + local;
        let case_key = format!("{session}/{}/{sequence:08}", spec.name);
        let prepared = PreparedFrame {
            case_key: case_key.clone(),
            session_id: session.into(),
            phase: spec.name.into(),
            sequence,
            label_bits: spec.label.to_bits(),
            left_native_sha256: "l".into(),
            right_native_sha256: "r".into(),
            tensor_sha256: "t".into(),
            left_stats: dummy_stats(left as f64),
            right_stats: dummy_stats(right as f64),
        };
        let bits = [
            presence.to_bits(),
            left.to_bits(),
            right.to_bits(),
            0.1f32.to_bits(),
            0.1f32.to_bits(),
        ];
        let model = ModelFrameBits {
            case_key,
            output_bits: bits,
            left_native_sha256: "l".into(),
            right_native_sha256: "r".into(),
            tensor_sha256: "t".into(),
        };
        AnalyzedFrame {
            prepared_a: prepared.clone(),
            prepared_b: prepared,
            model_a: model.clone(),
            model_b: model,
        }
    }

    #[test]
    fn frozen_schedule_totals_and_selection_counts_are_exact() {
        assert_eq!(
            PHASE_SCHEDULE.iter().map(|p| p.pairs).sum::<usize>(),
            PAIRS_PER_SESSION
        );
        assert_eq!(PHASE_SCHEDULE[0].first_sequence, 0);
        assert_eq!(
            PHASE_SCHEDULE.last().unwrap().last_sequence(),
            PAIRS_PER_SESSION - 1
        );
        let retained: usize = PHASE_SCHEDULE
            .iter()
            .map(|spec| {
                (spec.first_sequence..=spec.last_sequence())
                    .filter(|&sequence| retained_role(spec, sequence).is_some())
                    .count()
            })
            .sum();
        let associations: usize = PHASE_SCHEDULE
            .iter()
            .map(|spec| {
                (spec.first_sequence..=spec.last_sequence())
                    .filter(|&sequence| retained_role(spec, sequence).is_some_and(|(_, a)| a))
                    .count()
            })
            .sum();
        assert_eq!(retained, RETAINED_PAIRS_PER_SESSION);
        assert_eq!(associations, ASSOCIATION_PAIRS_PER_SESSION);
        for spec in &PHASE_SCHEDULE {
            assert_eq!(spec.pairs % 5, 0);
            let blocks: Vec<_> = (spec.first_sequence..=spec.last_sequence())
                .map(|sequence| block_number(spec, sequence).unwrap())
                .collect();
            for block in 1..=5 {
                assert_eq!(
                    blocks.iter().filter(|&&value| value == block).count(),
                    spec.pairs / 5
                );
            }
        }
    }

    #[test]
    fn session_name_grammar_is_exact() {
        assert!(is_valid_session_name(
            "session-0000000000000-0000000000-000000"
        ));
        assert!(is_valid_session_name(
            "session-9999999999999-9999999999-999999"
        ));
        for invalid in [
            "session-000000000000-0000000000-000000",
            "session-0000000000000-000000000-000000",
            "session-0000000000000-0000000000-00000",
            "Session-0000000000000-0000000000-000000",
            "session-000000000000x-0000000000-000000",
        ] {
            assert!(!is_valid_session_name(invalid), "{invalid}");
        }
    }

    #[test]
    fn strict_csv_parser_accepts_only_the_frozen_schedule() {
        let bytes = valid_csv();
        let cases = parse_completed_csv("session_0000", &bytes, Path::new("unused")).unwrap();
        assert_eq!(cases.len(), PAIRS_PER_SESSION);
        assert_eq!(
            cases[0].left_safe_path,
            "session_0000/images/neutral_center/neutral_center_l_00000000.png"
        );
        assert_eq!(
            cases[PAIRS_PER_SESSION - 1].right_safe_path,
            "session_0000/images/right_wink_negative/right_wink_negative_r_00002039.png"
        );

        let mut crlf = bytes.clone();
        crlf.insert(crlf.iter().position(|&b| b == b'\n').unwrap(), b'\r');
        assert!(parse_completed_csv("session_0000", &crlf, Path::new("unused")).is_err());
        let mut no_final_lf = bytes.clone();
        no_final_lf.pop();
        assert!(parse_completed_csv("session_0000", &no_final_lf, Path::new("unused")).is_err());
        let wrong_header =
            bytes.replacen(b"filename,wide,phase,side", b"filename,label,phase,side", 1);
        assert!(parse_completed_csv("session_0000", &wrong_header, Path::new("unused")).is_err());
        let wrong_side = bytes.replacen(b",neutral_center,l\n", b",neutral_center,r\n", 1);
        assert!(parse_completed_csv("session_0000", &wrong_side, Path::new("unused")).is_err());
        let nonfinite = bytes.replacen(b",0,neutral_center,l\n", b",NaN,neutral_center,l\n", 1);
        assert!(parse_completed_csv("session_0000", &nonfinite, Path::new("unused")).is_err());
    }

    trait ReplaceBytes {
        fn replacen(&self, from: &[u8], to: &[u8], count: usize) -> Vec<u8>;
    }

    impl ReplaceBytes for Vec<u8> {
        fn replacen(&self, from: &[u8], to: &[u8], count: usize) -> Vec<u8> {
            let mut output = Vec::new();
            let mut rest = self.as_slice();
            let mut replaced = 0;
            while replaced < count {
                let Some(index) = rest.windows(from.len()).position(|window| window == from) else {
                    break;
                };
                output.extend_from_slice(&rest[..index]);
                output.extend_from_slice(to);
                rest = &rest[index + from.len()..];
                replaced += 1;
            }
            output.extend_from_slice(rest);
            output
        }
    }

    #[test]
    fn partial_fixture_is_sealed_as_invalid_without_decoding_pixels() {
        let root = TempDir::new("partial");
        let sessions = root.0.join("sessions");
        fs::create_dir_all(sessions.join("session-0000000000001-0000000001-000001/images"))
            .unwrap();
        fs::create_dir_all(sessions.join("session-0000000000002-0000000001-000002/images"))
            .unwrap();
        fs::write(
            sessions.join("session-0000000000001-0000000001-000001/images/raw.bin"),
            b"not a png",
        )
        .unwrap();
        let plan = build_input_plan(&root.0).unwrap();
        assert_eq!(plan.terminal_status(), InputTerminalStatus::InputInvalid);
        assert!(plan
            .session_plan
            .errors
            .contains(&"insufficient_completed_sessions".into()));
        assert_eq!(
            plan.session_plan.partial_session_ids,
            ["partial_0000", "partial_0001"]
        );
        assert!(plan.recording_inventory_json.ends_with(b"\n"));
        assert!(plan.session_plan_json.ends_with(b"\n"));
        assert!(String::from_utf8_lossy(&plan.session_plan_json)
            .contains("\"terminal_status\": \"INPUT_INVALID\""));
        assert!(!String::from_utf8_lossy(&plan.recording_inventory_json).contains("session-000000"));
    }

    #[test]
    fn inspectable_missing_and_regular_file_sessions_roots_are_sealed_invalid() {
        let missing = TempDir::new("missing-sessions-root");
        let first = build_input_plan(&missing.0).unwrap();
        let second = build_input_plan(&missing.0).unwrap();
        assert_eq!(first.terminal_status(), InputTerminalStatus::InputInvalid);
        assert!(first.bit_exact_artifacts_match(&second));
        assert!(first.inventory.entries.is_empty());
        assert!(first
            .session_plan
            .errors
            .contains(&"sessions_root_missing".into()));
        assert!(first
            .session_plan
            .errors
            .contains(&"insufficient_completed_sessions".into()));

        let regular = TempDir::new("regular-sessions-root");
        fs::write(
            regular.0.join("sessions"),
            b"inspectable but not a directory",
        )
        .unwrap();
        let plan = build_input_plan(&regular.0).unwrap();
        assert_eq!(plan.terminal_status(), InputTerminalStatus::InputInvalid);
        assert_eq!(plan.inventory.entries.len(), 1);
        let entry = &plan.inventory.entries[0];
        assert_eq!(entry.path, "sessions_root");
        assert_eq!(entry.kind, InventoryKind::File);
        assert_eq!(entry.byte_len, Some(31));
        assert_eq!(
            entry.sha256.as_deref(),
            Some(sha256_hex(b"inspectable but not a directory").as_str())
        );
        assert!(plan
            .session_plan
            .errors
            .contains(&"sessions_root_not_ordinary:File".into()));
    }

    #[cfg(unix)]
    #[test]
    fn sessions_root_symlink_is_sealed_invalid_without_following_it() {
        use std::os::unix::fs::symlink;

        let root = TempDir::new("symlink-sessions-root");
        let target = root.0.join("outside");
        fs::create_dir(&target).unwrap();
        fs::write(target.join("must-not-be-inventoried.bin"), b"private").unwrap();
        symlink(&target, root.0.join("sessions")).unwrap();

        let plan = build_input_plan(&root.0).unwrap();
        assert_eq!(plan.terminal_status(), InputTerminalStatus::InputInvalid);
        assert_eq!(plan.inventory.entries.len(), 1);
        assert_eq!(plan.inventory.entries[0].path, "sessions_root");
        assert_eq!(
            plan.inventory.entries[0].kind,
            InventoryKind::SymlinkOrReparse
        );
        assert!(!String::from_utf8_lossy(&plan.recording_inventory_json)
            .contains("must-not-be-inventoried"));
    }

    #[cfg(unix)]
    #[test]
    fn non_utf8_root_name_is_anonymized_and_valid_utf8_mapping_is_preserved() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let root = TempDir::new("non-utf8-root-name");
        let sessions = root.0.join("sessions");
        fs::create_dir(&sessions).unwrap();
        let first_valid = sessions.join("session-0000000000001-0000000000-000001");
        let second_valid = sessions.join("session-0000000000002-0000000000-000002");
        fs::create_dir(&first_valid).unwrap();
        fs::create_dir(&second_valid).unwrap();
        fs::write(first_valid.join("from_first.bin"), b"first").unwrap();
        fs::write(second_valid.join("from_second.bin"), b"second").unwrap();
        fs::write(first_valid.join("__non_utf8_0000__"), b"occupied").unwrap();
        fs::write(
            first_valid.join(OsString::from_vec(vec![0xfe])),
            b"nested non-utf8",
        )
        .unwrap();

        let non_utf8 = sessions.join(OsString::from_vec(b"session-invalid-\xff".to_vec()));
        fs::create_dir(&non_utf8).unwrap();
        fs::write(non_utf8.join("payload.bin"), b"payload").unwrap();

        let first = build_input_plan(&root.0).unwrap();
        let second = build_input_plan(&root.0).unwrap();
        assert_eq!(first.terminal_status(), InputTerminalStatus::InputInvalid);
        assert!(first.bit_exact_artifacts_match(&second));
        let paths: BTreeSet<_> = first
            .inventory
            .entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect();
        assert!(paths.contains("partial_0000/from_first.bin"));
        assert!(paths.contains("partial_0001/from_second.bin"));
        assert!(paths.contains("partial_0000/__non_utf8_0000__"));
        assert!(paths.contains("partial_0000/__non_utf8_0000___0001"));
        assert!(paths.contains("non_utf8_root_0000"));
        assert!(paths.contains("non_utf8_root_0000/payload.bin"));
        assert!(first
            .session_plan
            .errors
            .contains(&"non_utf8_entry_name:non_utf8_root_0000".into()));
        assert!(first
            .session_plan
            .errors
            .contains(&"non_utf8_entry_name:partial_0000/__non_utf8_0000___0001".into()));
        assert!(!String::from_utf8_lossy(&first.recording_inventory_json).contains('\u{fffd}'));
    }

    #[test]
    fn anonymized_non_utf8_components_do_not_collide_with_valid_names() {
        let mut occupied = BTreeSet::from(["__non_utf8_0000__".to_owned()]);
        assert_eq!(
            anonymized_non_utf8_component(0, &mut occupied),
            "__non_utf8_0000___0001"
        );
    }

    #[test]
    fn png_decoder_enforces_gray8_and_200_square() {
        let gray = vec![17u8; INPUT_PIXELS];
        let complete = encode_png(png::ColorType::Grayscale, 200, 200, &gray);
        assert_eq!(decode_gray8_200(&complete).unwrap(), gray);
        let rgb = vec![17u8; INPUT_PIXELS * 3];
        assert!(decode_gray8_200(&encode_png(png::ColorType::Rgb, 200, 200, &rgb)).is_err());
        let small = vec![17u8; 199 * 200];
        assert!(
            decode_gray8_200(&encode_png(png::ColorType::Grayscale, 199, 200, &small)).is_err()
        );
        let truncated = &encode_png(png::ColorType::Grayscale, 200, 200, &gray)[..30];
        assert!(decode_gray8_200(truncated).is_err());
        let without_iend = &complete[..complete.len() - 12];
        assert!(
            decode_gray8_200(without_iend).is_err(),
            "a complete IDAT without the final IEND must still be rejected"
        );
    }

    #[test]
    fn verified_read_rejects_length_and_hash_changes() {
        let root = TempDir::new("sealed-read");
        let path = root.0.join("frame.png");
        fs::write(&path, b"abc").unwrap();
        let file = RuntimeFile {
            safe_path: "session_0000/images/x.png".into(),
            source_path: path.clone(),
            byte_len: 3,
            sha256: sha256_hex(b"abc"),
        };
        assert_eq!(read_verified_file(&file).unwrap(), b"abc");
        fs::write(&path, b"abd").unwrap();
        assert!(read_verified_file(&file).unwrap_err().contains("sha256"));
        fs::write(&path, b"longer").unwrap();
        assert!(read_verified_file(&file).unwrap_err().contains("length"));
    }

    #[test]
    fn saved_right_unmirror_reproduces_direct_live_tensor_bit_for_bit() {
        let root = TempDir::new("capture-writer-parity");
        let mut left_live = vec![0u8; INPUT_PIXELS];
        let mut right_live = vec![0u8; INPUT_PIXELS];
        for y in 0..INPUT_SIDE {
            for x in 0..INPUT_SIDE {
                left_live[y * INPUT_SIDE + x] = ((x * 3 + y * 5) % 251) as u8;
                right_live[y * INPUT_SIDE + x] =
                    ((x * 11 + y * 7 + (x < 37) as usize * 53) % 256) as u8;
            }
        }
        let left_path = root.0.join("left.png");
        let right_path = root.0.join("right.png");
        sranibro_rs::wide_calib::research_write_gray_png(
            &left_path,
            INPUT_SIDE as u32,
            INPUT_SIDE as u32,
            &left_live,
            false,
        )
        .unwrap();
        sranibro_rs::wide_calib::research_write_gray_png(
            &right_path,
            INPUT_SIDE as u32,
            INPUT_SIDE as u32,
            &right_live,
            true,
        )
        .unwrap();
        let left_saved = decode_gray8_200(&fs::read(left_path).unwrap()).unwrap();
        let right_saved = decode_gray8_200(&fs::read(right_path).unwrap()).unwrap();
        assert_eq!(left_saved, left_live, "capture writer must preserve left");
        assert_ne!(right_live, right_saved, "fixture must be non-symmetric");
        let case = RuntimeCase {
            session_id: "session_0000".into(),
            phase: "neutral_center".into(),
            sequence: 0,
            label_bits: 0.0f32.to_bits(),
            left_safe_path: "l".into(),
            right_safe_path: "r".into(),
        };
        let geometry = frozen_geometry().unwrap();
        let audit = preprocess_decoded_case(&case, &left_saved, &right_saved, &geometry).unwrap();
        let despeckle = DespeckleParams::default();
        let l = preprocess::despeckle(&left_live, INPUT_SIDE, INPUT_SIDE, &despeckle);
        let r = preprocess::despeckle(&right_live, INPUT_SIDE, INPUT_SIDE, &despeckle);
        let direct = preprocess::to_input_stereo_geom(
            &l,
            200,
            200,
            &r,
            200,
            200,
            false,
            true,
            &geometry[0],
            &geometry[1],
        );
        assert_eq!(
            audit.tensor.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            direct.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn scalar_statistics_follow_type7_population_and_average_rank_rules() {
        let values = [0.0, 10.0, 20.0, 30.0];
        assert_eq!(quantile(&values, 0.25), Some(7.5));
        assert_eq!(quantile(&values, 0.5), Some(15.0));
        assert_eq!(median_absolute_deviation(&values), Some(10.0));
        let plane = plane_stats_f64(&[0.0, 1.0, 0.0, 1.0], 2, 2).unwrap();
        assert_eq!(plane.mean, 0.5);
        assert_eq!(plane.population_stddev, 0.5);
        assert_eq!(plane.mean_abs_neighbor_gradient, 0.5);
        assert_eq!(spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]), Some(-1.0));
        assert_eq!(spearman(&[1.0, 1.0, 2.0], &[2.0, 2.0, 3.0]), Some(1.0));
        assert_eq!(spearman(&[-0.0, 0.0, 1.0], &[0.0, 0.0, 2.0]), Some(1.0));
        assert_eq!(spearman(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]), None);
    }

    #[test]
    fn wink_difference_is_the_median_of_simultaneous_pairwise_differences() {
        let nonwink = [0.0, 100.0, 101.0];
        let wink = [0.0, 0.0, 100.0];
        assert_eq!(median_paired_difference(&nonwink, &wink), Some(1.0));
        assert_ne!(
            median_paired_difference(&nonwink, &wink),
            Some(quantile(&nonwink, 0.5).unwrap() - quantile(&wink, 0.5).unwrap()),
            "the preregistered simultaneous contrast is not a difference of marginal medians"
        );
    }

    #[test]
    fn coverage_gate_uses_frozen_integer_arithmetic() {
        let spec = phase_spec("neutral_center").unwrap();
        let frames: Vec<_> = (0..108)
            .map(|i| {
                analyzed_frame(
                    "session_0000",
                    spec,
                    36 + i,
                    0.3,
                    0.3,
                    if i < 103 { 0.051 } else { 0.05 },
                )
            })
            .collect();
        let refs: Vec<_> = frames.iter().collect();
        assert!(coverage_gate(&refs));
        let mut failed = frames;
        failed[102].model_a.output_bits[0] = 0.05f32.to_bits();
        failed[102].model_b.output_bits[0] = 0.05f32.to_bits();
        let refs: Vec<_> = failed.iter().collect();
        assert!(!coverage_gate(&refs));
    }

    #[test]
    fn primary_precedence_reports_any_evaluable_reversal() {
        let mut frames = Vec::new();
        for spec in PHASE_SCHEDULE.iter().take(3) {
            let (left, right) = match spec.name {
                "neutral_center" => (0.30, 0.30),
                "wide_soft" => (0.50, 0.50),
                "wide_max" => (0.45, 0.70),
                _ => unreachable!(),
            };
            for local in 0..spec.pairs {
                frames.push(analyzed_frame(
                    "session_0000",
                    spec,
                    local,
                    left,
                    right,
                    0.8,
                ));
            }
        }
        let (primary, comparison, high) = classify_primary(&frames, &["session_0000".into()]);
        assert_eq!(
            primary
                .iter()
                .find(|r| r.eye == Eye::Left)
                .unwrap()
                .category,
            PrimaryCategory::Reversal
        );
        assert_eq!(
            primary
                .iter()
                .find(|r| r.eye == Eye::Right)
                .unwrap()
                .category,
            PrimaryCategory::Monotone
        );
        assert_eq!(
            comparison[0].annotation,
            EyeCategoryAnnotation::EyeCategoryAsymmetric
        );
        assert_eq!(high, HighOpenAnnotation::HighOpenReversalObserved);
    }

    #[test]
    fn reduced_outcome_has_exact_allowlist_and_null_annotation() {
        let root = TempDir::new("reduced");
        let plan = BuiltInputPlan {
            inventory: RecordingInventoryArtifact {
                schema: "x".into(),
                entries: Vec::new(),
            },
            session_plan: SessionPlanArtifact {
                schema: "x".into(),
                terminal_status: InputTerminalStatus::InputInvalid,
                errors: vec!["x".into()],
                completed_sessions: Vec::new(),
                partial_session_ids: Vec::new(),
                recording_tree_sha256: "0".repeat(64),
                phase_schedule: phase_plan_entries(),
                independence: "independence_unproven".into(),
            },
            recording_inventory_json: b"{}\n".to_vec(),
            session_plan_json: b"{}\n".to_vec(),
            recording_tree_sha256: "0".repeat(64),
            runtime: RuntimeInputPlan {
                wide_data_root: root.0.clone(),
                sessions_root: root.0.join("sessions"),
                files: BTreeMap::new(),
                completed_sessions: Vec::new(),
            },
        };
        let loader_called = Cell::new(false);
        let outcome = analyze_with_model_loader(&plan, || {
            loader_called.set(true);
            Ok(b"wrong".to_vec())
        });
        assert!(
            !loader_called.get(),
            "invalid preflight must not invoke model loader"
        );
        assert_eq!(
            outcome.terminal_status,
            AnalysisTerminalStatus::InputInvalid
        );
        assert_eq!(outcome.high_open_annotation, None);
        assert!(outcome.payload_allowlist_is_exact());
        assert!(outcome.publication_forbidden);
        assert!(!outcome.metadata.model_loader_invoked);
        assert!(!outcome.metadata.model_bytes_obtained);
        assert_eq!(outcome.metadata.model_byte_len, 0);
        assert!(outcome.metadata.model_sha256.is_empty());
        assert!(
            !String::from_utf8_lossy(&outcome.payloads["diagnostic.json"]).contains("HIGH_OPEN_")
        );
    }

    #[test]
    fn checked_preflights_invoke_loader_once_and_classify_trusted_read_failure() {
        let root = TempDir::new("loader-trusted");
        let plan = dummy_plan_for_loader_test(&root);
        let calls = Cell::new(0usize);
        let outcome = continue_after_checked_preflights(&plan, empty_checked_preflights(), || {
            calls.set(calls.get() + 1);
            Err(ModelLoaderError::read_or_identity("model unavailable"))
        });
        assert_eq!(calls.get(), 1);
        assert_eq!(
            outcome.terminal_status,
            AnalysisTerminalStatus::InconclusiveArtifact
        );
        assert!(!outcome.publication_forbidden);
        assert!(outcome.metadata.model_loader_invoked);
        assert!(!outcome.metadata.model_bytes_obtained);
        assert_eq!(
            outcome.metadata.model_loader_error_kind,
            Some(ModelLoaderErrorKind::ReadOrIdentityFailure)
        );
        assert!(outcome.payload_allowlist_is_exact());
    }

    #[test]
    fn model_change_during_deferred_read_forbids_publication() {
        let root = TempDir::new("loader-change");
        let plan = dummy_plan_for_loader_test(&root);
        let outcome = continue_after_checked_preflights(&plan, empty_checked_preflights(), || {
            Err(ModelLoaderError::changed_during_read("identity changed"))
        });
        assert_eq!(
            outcome.terminal_status,
            AnalysisTerminalStatus::InconclusiveArtifact
        );
        assert!(outcome.publication_forbidden);
        assert!(outcome.metadata.model_loader_invoked);
        assert!(!outcome.metadata.model_bytes_obtained);
        assert_eq!(
            outcome.metadata.model_loader_error_kind,
            Some(ModelLoaderErrorKind::ChangedDuringRead)
        );
    }

    fn empty_checked_preflights() -> CheckedPreflights {
        CheckedPreflights {
            first: PreflightPass {
                pass_name: "preflight_a".into(),
                ordered_stream_sha256: "a".repeat(64),
                frames: Vec::new(),
            },
            second: PreflightPass {
                pass_name: "preflight_b".into(),
                ordered_stream_sha256: "b".repeat(64),
                frames: Vec::new(),
            },
            digests: AnalysisDigests {
                preflight_a_ordered_stream_sha256: Some("a".repeat(64)),
                preflight_b_ordered_stream_sha256: Some("b".repeat(64)),
                ..AnalysisDigests::default()
            },
        }
    }

    fn dummy_plan_for_loader_test(root: &TempDir) -> BuiltInputPlan {
        BuiltInputPlan {
            inventory: RecordingInventoryArtifact {
                schema: "x".into(),
                entries: Vec::new(),
            },
            session_plan: SessionPlanArtifact {
                schema: "x".into(),
                terminal_status: InputTerminalStatus::InputSealed,
                errors: Vec::new(),
                completed_sessions: Vec::new(),
                partial_session_ids: Vec::new(),
                recording_tree_sha256: "0".repeat(64),
                phase_schedule: phase_plan_entries(),
                independence: "independence_unproven".into(),
            },
            recording_inventory_json: b"{}\n".to_vec(),
            session_plan_json: b"{}\n".to_vec(),
            recording_tree_sha256: "0".repeat(64),
            runtime: RuntimeInputPlan {
                wide_data_root: root.0.clone(),
                sessions_root: root.0.join("sessions"),
                files: BTreeMap::new(),
                completed_sessions: Vec::new(),
            },
        }
    }
}
