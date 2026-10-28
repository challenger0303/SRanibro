//! Renderer preparation and deterministic analysis for the frozen Phase 1.3 study.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::model::{canonical_tensor, sha256_hex, tensor_sha256};
use crate::renderer::{
    gray_moments, render, render_stereo, GrayMoments, ImageCovariates, PhotometricBasis,
    RenderedStereo, StereoPolicy, SyntheticEyeSpec, SIDE,
};

pub const VERSION: &str = "synthetic-eye-phase13-v1";
pub const PREREGISTRATION_COMMIT: &str = "f7cf3686d8025631a8e83442a69c43858402cd6c";
pub const ATLAS_REPOSITORY_COMMIT: &str = "49e13f0eb2b78f84a387de8a46e7309257c9304e";
pub const ATLAS_PREREGISTRATION_COMMIT: &str = "c92dbb2411c13d2f055ee7c1a67ee2b956d1e1a1";
pub const ATLAS_MANIFEST_SHA256: &str =
    "13d67e09915faa22f89f482ae3960703d1b610e519e5868ddcf884df0542e347";
pub const CANDIDATE_STREAM_SHA256: &str =
    "4eb662658c7997d37d53acc6daa8af9045e9ae2386784e39f3d37b4145313139";
pub const APERTURE_SUMMARIES_SHA256: &str =
    "ed8a7c468c301124bd26de9c56baa6755c9b4d82b728e4cfcf39e2bc820ad2f1";
pub const PAIR_SUMMARIES_SHA256: &str =
    "d7ce595f354b8dacd5f19c04d1a0f04f24aadad585fc0c580de29f334efd1480";
pub const CANONICAL_CHECKS_SHA256: &str =
    "5c67d57e6321f00f7e1e6e4e8a98de5d0788670e3afb176421ecc4afb135453d";
pub const RENDERER_SOURCE_SHA256: &str =
    "9fdeca8c45fa6c56d7721e0a2a2d10e1b9c19799ff528ccc5d059f7031c056bd";
pub const ATLAS_IMPLEMENTATION_SOURCE_SHA256: &str =
    "d3cfd3a4669d30663bcbd4072e2a8584f979e9e5e5b6c789864f424dc1f1bdc8";
pub const MODEL_SHA256: &str = "bac8013e0423068924f190a1de44afd5e1dd0c7c10d1d394926e46fc1b075ded";
pub const MODEL_LENGTH: u64 = 51_423_934;
pub const FROZEN_PLAN_SHA256: &str =
    "d9312bd9434c25cab044977ffb96c3fea92fbdfa4beb12327f840f1b79552f42";

const ATLAS_VERSION: &str = "synthetic-eye-moment-atlas-v1";
const RENDERER_VERSION: &str = "synthetic-eye-renderer-100x100-4x-v1";
const CANDIDATE_MAGIC: &[u8; 8] = b"SRATL3A1";
const CANDIDATE_SCHEMA: u32 = 1;
const CANDIDATE_HEADER_LEN: usize = 24;
const CANDIDATE_RECORD_LEN: usize = 28;
const CANDIDATE_COUNT: usize = 1_125_876;
const PAIR_COUNT: u64 = 1_683;
const RECORDS_PER_APERTURE: usize = 33_114;
const D0_PER_APERTURE: usize = 16_642;
const D1_PER_APERTURE: usize = 24_858;
const APERTURE_START: usize = 7;
const APERTURE_END: usize = 40;
const PAIR_LOWER_END: usize = 36;
const MASK_D0: u8 = 0x01;
const KNOWN_MEMBERSHIP: u8 = 0x07;
const DEFAULT_SKIN: f32 = 0.46;
const DEFAULT_SCLERA: f32 = 0.78;
const D0_STEP: f64 = 0.30f64 / 128.0f64;
const BOUNDARY_MARGIN: f64 = D0_STEP * 2.0f64;
const MOMENT_TOLERANCE: f64 = 0.001f64;
const KAPPA_LIMIT: f64 = 20.0f64;
const RANGE_GATE: f64 = 7.0f64;
const TARGET_SEPARATION: f64 = 4.0f64;
const PLAN_HASH_DOMAIN: &[u8] = b"sranibro-synthetic-eye-phase13-plan-v1\0";

pub const DELTA4: f64 = 0.001f64 * 4.0f64;
pub const E4: f64 = (0.05f64 * 4.0f64) / 33.0f64;
pub const B4: f64 = (0.10f64 * 4.0f64) / 33.0f64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Decision,
    Confirmation,
}

impl Stage {
    pub fn expected_pair_count(self) -> usize {
        match self {
            Self::Decision => 15,
            Self::Confirmation => 14,
        }
    }

    fn accepts(self, index: usize) -> bool {
        match self {
            Self::Decision => index % 2 == 1,
            Self::Confirmation => index % 2 == 0,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Decision => "decision",
            Self::Confirmation => "confirmation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    aperture_index: usize,
    membership: u8,
    skin: f32,
    sclera: f32,
    moments: GrayMoments,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProbeRecord {
    pub skin_bits: u32,
    pub sclera_bits: u32,
    pub mean_bits: u64,
    pub stddev_bits: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConditioningRecord {
    pub probes: [ProbeRecord; 4],
    pub matrix: [[f64; 2]; 2],
    pub sigma_max: f64,
    pub sigma_min: f64,
    pub kappa: f64,
}

#[derive(Clone, Debug)]
struct EligibleCandidate {
    candidate: Candidate,
    conditioning: ConditioningRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeRecord {
    pub aperture_index: usize,
    pub skin_bits: u32,
    pub sclera_bits: u32,
    pub mean_bits: u64,
    pub stddev_bits: u64,
    pub normalized_default_distance: f64,
    pub conditioning: ConditioningRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalRecord {
    pub pixels_sha256: String,
    pub mean_bits: u64,
    pub stddev_bits: u64,
    pub covariates: ImageCovariates,
    pub descriptive_covariates: DescriptiveCovariates,
    pub eye_like: bool,
    #[serde(skip)]
    canonical_pixels: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DescriptiveCovariates {
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub quadrant_mean_gray: [f64; 4],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TargetEndpointRecord {
    pub recipe: RecipeRecord,
    pub canonical: CanonicalRecord,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeasibleTargetRecord {
    pub mean_bits: u64,
    pub stddev_bits: u64,
    pub endpoint_residuals: [f64; 4],
    pub lower: TargetEndpointRecord,
    pub higher: TargetEndpointRecord,
}

impl FeasibleTargetRecord {
    fn mean(&self) -> f64 {
        f64::from_bits(self.mean_bits)
    }

    fn stddev(&self) -> f64 {
        f64::from_bits(self.stddev_bits)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellRecord {
    pub target_number: usize,
    pub aperture_role: ApertureRole,
    pub target_mean_bits: u64,
    pub target_stddev_bits: u64,
    pub residual_mean: f64,
    pub residual_stddev: f64,
    pub recipe: RecipeRecord,
    pub canonical: CanonicalRecord,
    pub default_geometry_bits: u32,
    pub default_raster_aperture_bits: u32,
    pub tensor_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApertureRole {
    Lower,
    Higher,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectedTargetPair {
    pub target0: FeasibleTargetRecord,
    pub target1: FeasibleTargetRecord,
    pub q25: f64,
    pub q75: f64,
    pub cells: [CellRecord; 4],
    /// Target-major `[mean, stddev]` higher-minus-lower achieved imbalance.
    pub signed_cell_imbalances: [[f64; 2]; 2],
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairPlan {
    pub lower_index: usize,
    pub higher_index: usize,
    pub feasible_target_count: usize,
    pub feasible_mean_range: [f64; 2],
    pub feasible_stddev_range: [f64; 2],
    pub retained: bool,
    pub selected: Option<SelectedTargetPair>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    DefaultS3,
    Factorial,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseRecord {
    pub id: String,
    pub stage: Stage,
    pub kind: CaseKind,
    pub aperture_index: usize,
    pub pair_lower_index: Option<usize>,
    pub target_number: Option<usize>,
    pub aperture_role: Option<ApertureRole>,
    pub skin_bits: u32,
    pub sclera_bits: u32,
    pub left_pixels_sha256: String,
    pub right_pixels_sha256: String,
    pub tensor_sha256: String,
    pub left_covariates: ImageCovariates,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RendererPlan {
    pub version: String,
    pub preregistration_commit: String,
    pub atlas_repository_commit: String,
    pub atlas_preregistration_commit: String,
    pub atlas_manifest_sha256: String,
    pub candidate_stream_sha256: String,
    pub renderer_version: String,
    pub global_axis: String,
    pub boundary_margin: f64,
    pub maximum_condition_number: f64,
    pub moment_tolerance_gray: f64,
    pub pair_plans: Vec<PairPlan>,
    pub decision_case_order: Vec<CaseRecord>,
    pub confirmation_case_order: Vec<CaseRecord>,
    pub excluded_pairs: Vec<[usize; 2]>,
    pub preparation_repetitions: usize,
    pub renderer_decision: String,
}

#[derive(Clone, Debug)]
pub struct StageCase {
    pub record: CaseRecord,
    pub stereo: RenderedStereo,
    pub tensor: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct PreparedPlan {
    pub plan: RendererPlan,
    pub plan_json: Vec<u8>,
    pub plan_sha256: String,
    pub decision_cases: Vec<StageCase>,
    pub confirmation_cases: Vec<StageCase>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RawCaseResult {
    pub id: String,
    pub raw: [f32; 5],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ResponseClass {
    GeometrySupported,
    AlternativePhotometricPath,
    NoEvidence,
    Inconclusive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StageStatus {
    RendererNoGo,
    InconclusiveArtifact,
    InconclusiveRecognition,
    InconclusiveInsensitive,
    GeometrySupported,
    AlternativePhotometricPath,
    NoEvidence,
    Inconclusive,
}

impl ResponseClass {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeometrySupported => "GEOMETRY_SUPPORTED",
            Self::AlternativePhotometricPath => "ALTERNATIVE_PHOTOMETRIC_PATH",
            Self::NoEvidence => "NO_EVIDENCE",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

impl StageStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::RendererNoGo => "RENDERER_NO_GO",
            Self::InconclusiveArtifact => "INCONCLUSIVE_ARTIFACT",
            Self::InconclusiveRecognition => "INCONCLUSIVE_RECOGNITION",
            Self::InconclusiveInsensitive => "INCONCLUSIVE_INSENSITIVE",
            Self::GeometrySupported => "GEOMETRY_SUPPORTED",
            Self::AlternativePhotometricPath => "ALTERNATIVE_PHOTOMETRIC_PATH",
            Self::NoEvidence => "NO_EVIDENCE",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AnalysisFlag {
    EyeAsymmetric,
    TargetModulated,
    GeometryReversed,
    TargetDirectionMixed,
    PhotometricSensitivity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BaselineMetrics {
    pub span: f64,
    pub spearman_rho: Option<f64>,
    pub mean_absolute_gap4_delta: f64,
    pub positive_gap4_count: usize,
    pub required_positive_gap4_count: usize,
    pub competent: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorMetrics {
    pub values: Vec<f64>,
    pub mean: f64,
    pub mean_absolute: f64,
    pub mean_to_baseline: Option<f64>,
    pub mean_absolute_to_baseline: Option<f64>,
    pub positive_count: usize,
    pub negative_count: usize,
    pub flat_count: usize,
    pub required_concordance_count: usize,
    pub positive: bool,
    pub negative: bool,
    pub flat: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EyeAnalysis {
    pub baseline: BaselineMetrics,
    pub g0: VectorMetrics,
    pub g1: VectorMetrics,
    pub p_l: VectorMetrics,
    pub p_h: VectorMetrics,
    pub geometry: VectorMetrics,
    pub photometric: VectorMetrics,
    pub modulation: VectorMetrics,
    pub provisional_response: ResponseClass,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageAnalysis {
    pub stage: Stage,
    pub status: StageStatus,
    pub response_class: Option<ResponseClass>,
    pub flags: Vec<AnalysisFlag>,
    pub artifact_error: Option<String>,
    pub first_raw_bits: Vec<(String, [u32; 5])>,
    pub second_raw_bits: Vec<(String, [u32; 5])>,
    pub eyes: Option<[EyeAnalysis; 2]>,
}

#[derive(Deserialize)]
struct AtlasManifest {
    atlas_version: String,
    renderer_version: String,
    renderer_source_sha256: String,
    preregistration_commit: String,
    repository_commit: String,
    build_repository_commit: String,
    implementation_source_sha256: String,
    repository_dirty: bool,
    build_profile: String,
    debug_assertions: bool,
    preparation_repetitions: u64,
    renderer_preparation_status: String,
    candidate_count: u64,
    candidate_record_bytes: u64,
    candidate_stream_bytes: u64,
    pair_count: u64,
    candidate_stream_sha256_repetitions: [String; 2],
    pair_summaries_sha256_repetitions: [String; 2],
    artifact_sha256: AtlasArtifactHashes,
    model_loaded: Option<bool>,
    model_identity_sha256: Option<String>,
    phase0_evaluated: Option<bool>,
    real_recordings_loaded: Option<bool>,
}

#[derive(Deserialize)]
struct AtlasArtifactHashes {
    candidate_stream_bin: String,
    aperture_summaries_json: String,
    pair_summaries_json: String,
    canonical_checks_json: String,
}

struct ParsedAtlas {
    apertures: Vec<Vec<Candidate>>,
}

pub fn validate_atlas_and_prepare(atlas_dir: &Path) -> Result<PreparedPlan, String> {
    let first = prepare_once(atlas_dir)?;
    let second = prepare_once(atlas_dir)?;
    if first.plan_sha256 != FROZEN_PLAN_SHA256 {
        return Err("Phase 1.3 renderer plan differed from the frozen complete identity".into());
    }
    if first.plan_json != second.plan_json
        || first.plan_sha256 != second.plan_sha256
        || first.plan != second.plan
        || !same_cases(&first.decision_cases, &second.decision_cases)
        || !same_cases(&first.confirmation_cases, &second.confirmation_cases)
    {
        return Err("repeated Phase 1.3 renderer preparation was not bit-identical".into());
    }
    Ok(first)
}

fn prepare_once(atlas_dir: &Path) -> Result<PreparedPlan, String> {
    let parsed = validate_and_parse_atlas(atlas_dir)?;
    let eligible = build_eligible_clouds(&parsed)?;
    let mut pair_plans = Vec::with_capacity(30);
    for lower in APERTURE_START..=PAIR_LOWER_END {
        pair_plans.push(prepare_pair(lower, &eligible)?);
    }
    let failures: Vec<_> = pair_plans
        .iter()
        .filter(|pair| !pair.retained)
        .map(|pair| [pair.lower_index, pair.higher_index])
        .collect();
    if failures != vec![[36, 40]] {
        return Err(format!(
            "mean-axis range gate excluded {failures:?}, expected only [[36, 40]]"
        ));
    }
    let excluded = &pair_plans[29];
    if excluded.feasible_mean_range[1] - excluded.feasible_mean_range[0] >= RANGE_GATE {
        return Err("frozen pair (36,40) unexpectedly passed the mean range gate".into());
    }

    let mut plan = RendererPlan {
        version: VERSION.into(),
        preregistration_commit: PREREGISTRATION_COMMIT.into(),
        atlas_repository_commit: ATLAS_REPOSITORY_COMMIT.into(),
        atlas_preregistration_commit: ATLAS_PREREGISTRATION_COMMIT.into(),
        atlas_manifest_sha256: ATLAS_MANIFEST_SHA256.into(),
        candidate_stream_sha256: CANDIDATE_STREAM_SHA256.into(),
        renderer_version: RENDERER_VERSION.into(),
        global_axis: "whole_image_mean_gray".into(),
        boundary_margin: BOUNDARY_MARGIN,
        maximum_condition_number: KAPPA_LIMIT,
        moment_tolerance_gray: MOMENT_TOLERANCE,
        pair_plans,
        decision_case_order: Vec::new(),
        confirmation_case_order: Vec::new(),
        excluded_pairs: failures,
        preparation_repetitions: 2,
        renderer_decision: "go".into(),
    };
    let decision_cases = build_stage_cases(&plan, Stage::Decision)?;
    let confirmation_cases = build_stage_cases(&plan, Stage::Confirmation)?;
    plan.decision_case_order = decision_cases
        .iter()
        .map(|case| case.record.clone())
        .collect();
    plan.confirmation_case_order = confirmation_cases
        .iter()
        .map(|case| case.record.clone())
        .collect();
    let plan_json = serde_json::to_vec_pretty(&plan)
        .map_err(|error| format!("serialize renderer plan: {error}"))?;
    let plan_sha256 = domain_hash(PLAN_HASH_DOMAIN, &plan_json);
    Ok(PreparedPlan {
        plan,
        plan_json,
        plan_sha256,
        decision_cases,
        confirmation_cases,
    })
}

fn validate_and_parse_atlas(atlas_dir: &Path) -> Result<ParsedAtlas, String> {
    let expected_files: BTreeSet<String> = [
        "aperture_summaries.json",
        "candidate_stream.bin",
        "canonical_checks.json",
        "manifest.json",
        "pair_summaries.json",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let actual_files: BTreeSet<String> = fs::read_dir(atlas_dir)
        .map_err(|error| format!("read atlas directory: {error}"))?
        .map(|entry| {
            let entry = entry.map_err(|error| format!("read atlas entry: {error}"))?;
            if !entry
                .file_type()
                .map_err(|error| format!("read atlas entry type: {error}"))?
                .is_file()
            {
                return Err("atlas directory contains a non-file entry".into());
            }
            entry
                .file_name()
                .into_string()
                .map_err(|_| "atlas filename is not UTF-8".to_string())
        })
        .collect::<Result<_, _>>()?;
    if actual_files != expected_files {
        return Err(format!("atlas file inventory was {actual_files:?}"));
    }

    let manifest_bytes = read_exact_hash(
        &atlas_dir.join("manifest.json"),
        ATLAS_MANIFEST_SHA256,
        "atlas manifest",
    )?;
    let manifest: AtlasManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse atlas manifest: {error}"))?;
    validate_manifest(&manifest)?;
    let stream = read_exact_hash(
        &atlas_dir.join("candidate_stream.bin"),
        CANDIDATE_STREAM_SHA256,
        "candidate stream",
    )?;
    let _ = read_exact_hash(
        &atlas_dir.join("aperture_summaries.json"),
        APERTURE_SUMMARIES_SHA256,
        "aperture summaries",
    )?;
    let _ = read_exact_hash(
        &atlas_dir.join("pair_summaries.json"),
        PAIR_SUMMARIES_SHA256,
        "pair summaries",
    )?;
    let _ = read_exact_hash(
        &atlas_dir.join("canonical_checks.json"),
        CANONICAL_CHECKS_SHA256,
        "canonical checks",
    )?;
    parse_candidate_stream(&stream)
}

fn validate_manifest(manifest: &AtlasManifest) -> Result<(), String> {
    let valid = manifest.atlas_version == ATLAS_VERSION
        && manifest.renderer_version == RENDERER_VERSION
        && manifest.renderer_source_sha256 == RENDERER_SOURCE_SHA256
        && manifest.preregistration_commit == ATLAS_PREREGISTRATION_COMMIT
        && manifest.repository_commit == ATLAS_REPOSITORY_COMMIT
        && manifest.build_repository_commit == ATLAS_REPOSITORY_COMMIT
        && manifest.implementation_source_sha256 == ATLAS_IMPLEMENTATION_SOURCE_SHA256
        && !manifest.repository_dirty
        && manifest.build_profile == "release"
        && !manifest.debug_assertions
        && manifest.preparation_repetitions == 2
        && manifest.renderer_preparation_status == "complete"
        && manifest.candidate_count == CANDIDATE_COUNT as u64
        && manifest.candidate_record_bytes == CANDIDATE_RECORD_LEN as u64
        && manifest.candidate_stream_bytes
            == (CANDIDATE_HEADER_LEN + CANDIDATE_COUNT * CANDIDATE_RECORD_LEN) as u64
        && manifest.pair_count == PAIR_COUNT
        && manifest.candidate_stream_sha256_repetitions
            == [CANDIDATE_STREAM_SHA256, CANDIDATE_STREAM_SHA256]
        && manifest.pair_summaries_sha256_repetitions
            == [PAIR_SUMMARIES_SHA256, PAIR_SUMMARIES_SHA256]
        && manifest.artifact_sha256.candidate_stream_bin == CANDIDATE_STREAM_SHA256
        && manifest.artifact_sha256.aperture_summaries_json == APERTURE_SUMMARIES_SHA256
        && manifest.artifact_sha256.pair_summaries_json == PAIR_SUMMARIES_SHA256
        && manifest.artifact_sha256.canonical_checks_json == CANONICAL_CHECKS_SHA256
        && manifest.model_loaded != Some(true)
        && manifest.model_identity_sha256.is_none()
        && manifest.phase0_evaluated != Some(true)
        && manifest.real_recordings_loaded != Some(true);
    if !valid {
        return Err("atlas manifest did not match the frozen Phase 1.3 identity".into());
    }
    Ok(())
}

fn read_exact_hash(path: &Path, expected: &str, label: &str) -> Result<Vec<u8>, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {label}: {error}"))?;
    let actual = sha256_hex(&bytes);
    if actual != expected {
        return Err(format!("{label} SHA-256 was {actual}, expected {expected}"));
    }
    Ok(bytes)
}

fn parse_candidate_stream(bytes: &[u8]) -> Result<ParsedAtlas, String> {
    let expected_len = CANDIDATE_HEADER_LEN + CANDIDATE_COUNT * CANDIDATE_RECORD_LEN;
    if bytes.len() != expected_len {
        return Err(format!(
            "candidate stream length was {}, expected {expected_len}",
            bytes.len()
        ));
    }
    if &bytes[0..8] != CANDIDATE_MAGIC
        || le_u32(&bytes[8..12]) != CANDIDATE_SCHEMA
        || le_u32(&bytes[12..16]) != CANDIDATE_RECORD_LEN as u32
        || le_u64(&bytes[16..24]) != CANDIDATE_COUNT as u64
    {
        return Err("candidate stream header did not match schema 1".into());
    }

    let mut apertures = vec![Vec::with_capacity(D0_PER_APERTURE); APERTURE_END + 1];
    let mut per_aperture = [0usize; APERTURE_END + 1];
    let mut d0_counts = [0usize; APERTURE_END + 1];
    let mut d1_counts = [0usize; APERTURE_END + 1];
    let mut previous_key: Option<(usize, u32, u32)> = None;
    for record in bytes[CANDIDATE_HEADER_LEN..].chunks_exact(CANDIDATE_RECORD_LEN) {
        let aperture_index = record[0] as usize;
        let membership = record[1];
        let reserved = u16::from_le_bytes(record[2..4].try_into().unwrap());
        let skin = f32::from_bits(le_u32(&record[4..8]));
        let sclera = f32::from_bits(le_u32(&record[8..12]));
        let mean = f64::from_bits(le_u64(&record[12..20]));
        let stddev = f64::from_bits(le_u64(&record[20..28]));
        if !(APERTURE_START..=APERTURE_END).contains(&aperture_index)
            || reserved != 0
            || membership == 0
            || membership & !KNOWN_MEMBERSHIP != 0
            || membership & 0x04 == 0
            || !skin.is_finite()
            || !sclera.is_finite()
            || !mean.is_finite()
            || !stddev.is_finite()
        {
            return Err("invalid candidate record".into());
        }
        let key = (aperture_index, skin.to_bits(), sclera.to_bits());
        if previous_key.is_some_and(|previous| previous >= key) {
            return Err("candidate records were not in strict aperture/level order".into());
        }
        previous_key = Some(key);
        per_aperture[aperture_index] += 1;
        d0_counts[aperture_index] += usize::from(membership & MASK_D0 != 0);
        d1_counts[aperture_index] += usize::from(membership & 0x02 != 0);
        if membership & MASK_D0 != 0 {
            apertures[aperture_index].push(Candidate {
                aperture_index,
                membership,
                skin,
                sclera,
                moments: GrayMoments { mean, stddev },
            });
        }
    }
    for index in APERTURE_START..=APERTURE_END {
        if per_aperture[index] != RECORDS_PER_APERTURE
            || d0_counts[index] != D0_PER_APERTURE
            || d1_counts[index] != D1_PER_APERTURE
        {
            return Err(format!(
                "candidate counts failed at aperture {index}: total={}, D0={}, D1={}",
                per_aperture[index], d0_counts[index], d1_counts[index]
            ));
        }
    }
    Ok(ParsedAtlas { apertures })
}

fn le_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

fn le_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

fn build_eligible_clouds(parsed: &ParsedAtlas) -> Result<Vec<Vec<EligibleCandidate>>, String> {
    let mut clouds = vec![Vec::new(); APERTURE_END + 1];
    for index in APERTURE_START..=APERTURE_END {
        let basis = PhotometricBasis::from_spec(&spec(index, DEFAULT_SKIN, DEFAULT_SCLERA));
        let mut cloud = Vec::new();
        for &candidate in &parsed.apertures[index] {
            if candidate.membership & MASK_D0 == 0 {
                continue;
            }
            if !(0.30f32..=0.60f32).contains(&candidate.skin)
                || !(0.65f32..=0.95f32).contains(&candidate.sclera)
            {
                return Err(format!("D0 membership outside D0 at aperture {index}"));
            }
            let fast = basis.moments(candidate.skin, candidate.sclera);
            if fast.mean.to_bits() != candidate.moments.mean.to_bits()
                || fast.stddev.to_bits() != candidate.moments.stddev.to_bits()
            {
                return Err(format!(
                    "candidate stream moment parity failed at aperture {index}"
                ));
            }
            if !passes_boundary(candidate.skin, candidate.sclera) {
                continue;
            }
            if let Some(conditioning) = conditioning(candidate, &basis)? {
                cloud.push(EligibleCandidate {
                    candidate,
                    conditioning,
                });
            }
        }
        if cloud.is_empty() {
            return Err(format!("no eligible D0 recipes at aperture {index}"));
        }
        clouds[index] = cloud;
    }
    Ok(clouds)
}

fn passes_boundary(skin: f32, sclera: f32) -> bool {
    let skin = skin as f64;
    let sclera = sclera as f64;
    skin - 0.30f64 >= BOUNDARY_MARGIN
        && 0.60f64 - skin >= BOUNDARY_MARGIN
        && sclera - 0.65f64 >= BOUNDARY_MARGIN
        && 0.95f64 - sclera >= BOUNDARY_MARGIN
}

fn conditioning(
    center: Candidate,
    basis: &PhotometricBasis,
) -> Result<Option<ConditioningRecord>, String> {
    let sm_skin = (center.skin as f64 - D0_STEP) as f32;
    let sp_skin = (center.skin as f64 + D0_STEP) as f32;
    let cm_sclera = (center.sclera as f64 - D0_STEP) as f32;
    let cp_sclera = (center.sclera as f64 + D0_STEP) as f32;
    if sm_skin.to_bits() == sp_skin.to_bits() || cm_sclera.to_bits() == cp_sclera.to_bits() {
        return Ok(None);
    }
    let configs = [
        (sm_skin, center.sclera),
        (sp_skin, center.sclera),
        (center.skin, cm_sclera),
        (center.skin, cp_sclera),
    ];
    if configs.iter().any(|&(skin, sclera)| {
        !(0.30f32..=0.60f32).contains(&skin) || !(0.65f32..=0.95f32).contains(&sclera)
    }) {
        return Ok(None);
    }
    let moments = configs.map(|(skin, sclera)| basis.moments(skin, sclera));
    if moments
        .iter()
        .any(|moment| !moment.mean.is_finite() || !moment.stddev.is_finite())
    {
        return Ok(None);
    }
    let skin_denominator = sp_skin as f64 - sm_skin as f64;
    let sclera_denominator = cp_sclera as f64 - cm_sclera as f64;
    let matrix = [
        [
            (moments[1].mean - moments[0].mean) / skin_denominator,
            (moments[3].mean - moments[2].mean) / sclera_denominator,
        ],
        [
            (moments[1].stddev - moments[0].stddev) / skin_denominator,
            (moments[3].stddev - moments[2].stddev) / sclera_denominator,
        ],
    ];
    let [a, b] = matrix[0];
    let [c, d] = matrix[1];
    if [a, b, c, d].iter().any(|value| !value.is_finite()) {
        return Ok(None);
    }
    let r = (a + d).hypot(c - b);
    let s = (a - d).hypot(b + c);
    let sigma_max = (r + s) * 0.5;
    let sigma_min = (r - s).abs() * 0.5;
    let kappa = sigma_max / sigma_min;
    if !r.is_finite()
        || !s.is_finite()
        || !sigma_max.is_finite()
        || !sigma_min.is_finite()
        || sigma_min <= 0.0
        || !kappa.is_finite()
        || kappa > KAPPA_LIMIT
    {
        return Ok(None);
    }
    let probes = std::array::from_fn(|i| ProbeRecord {
        skin_bits: configs[i].0.to_bits(),
        sclera_bits: configs[i].1.to_bits(),
        mean_bits: moments[i].mean.to_bits(),
        stddev_bits: moments[i].stddev.to_bits(),
    });
    Ok(Some(ConditioningRecord {
        probes,
        matrix,
        sigma_max,
        sigma_min,
        kappa,
    }))
}

#[derive(Clone)]
struct TargetInternal {
    mean: f64,
    stddev: f64,
    residuals: [f64; 4],
    lower: EligibleCandidate,
    higher: EligibleCandidate,
}

fn prepare_pair(lower_index: usize, clouds: &[Vec<EligibleCandidate>]) -> Result<PairPlan, String> {
    let higher_index = lower_index + 4;
    let internals = enumerate_feasible_targets(&clouds[lower_index], &clouds[higher_index]);
    if internals.is_empty() {
        return Err(format!(
            "gap-four pair ({lower_index},{higher_index}) had no feasible target"
        ));
    }
    let lower_basis = PhotometricBasis::from_spec(&spec(lower_index, DEFAULT_SKIN, DEFAULT_SCLERA));
    let higher_basis =
        PhotometricBasis::from_spec(&spec(higher_index, DEFAULT_SKIN, DEFAULT_SCLERA));
    let mut targets = Vec::with_capacity(internals.len());
    for target in internals {
        let lower_canonical = canonical_record(target.lower.candidate, &lower_basis, false)?;
        let higher_canonical = canonical_record(target.higher.candidate, &higher_basis, false)?;
        targets.push(FeasibleTargetRecord {
            mean_bits: target.mean.to_bits(),
            stddev_bits: target.stddev.to_bits(),
            endpoint_residuals: target.residuals,
            lower: TargetEndpointRecord {
                recipe: recipe_record(&target.lower),
                canonical: lower_canonical,
            },
            higher: TargetEndpointRecord {
                recipe: recipe_record(&target.higher),
                canonical: higher_canonical,
            },
        });
    }
    let mean_range = component_range(&targets, FeasibleTargetRecord::mean);
    let stddev_range = component_range(&targets, FeasibleTargetRecord::stddev);
    let mean_width = mean_range[1] - mean_range[0];
    let retained = mean_width >= RANGE_GATE;
    let selected = if retained {
        if targets.len() < 2 {
            return Err(format!(
                "retained pair ({lower_index},{higher_index}) had fewer than two targets"
            ));
        }
        Some(select_target_pair(
            lower_index,
            &targets,
            mean_range,
            &clouds[lower_index],
            &clouds[higher_index],
            &lower_basis,
            &higher_basis,
        )?)
    } else {
        None
    };
    Ok(PairPlan {
        lower_index,
        higher_index,
        feasible_target_count: targets.len(),
        feasible_mean_range: mean_range,
        feasible_stddev_range: stddev_range,
        retained,
        selected,
    })
}

fn enumerate_feasible_targets(
    lower: &[EligibleCandidate],
    higher: &[EligibleCandidate],
) -> Vec<TargetInternal> {
    let mut higher_order: Vec<usize> = (0..higher.len()).collect();
    higher_order.sort_by(|&a, &b| {
        higher[a]
            .candidate
            .moments
            .mean
            .total_cmp(&higher[b].candidate.moments.mean)
            .then_with(|| recipe_order(&higher[a], &higher[b]))
    });
    let mut deduplicated = BTreeMap::<(u64, u64), TargetInternal>::new();
    for low in lower {
        let lo = low.candidate.moments.mean - 2.0f64 * MOMENT_TOLERANCE;
        let hi = low.candidate.moments.mean + 2.0f64 * MOMENT_TOLERANCE;
        let begin =
            higher_order.partition_point(|&index| higher[index].candidate.moments.mean < lo);
        let end = higher_order.partition_point(|&index| higher[index].candidate.moments.mean <= hi);
        for &high_index in &higher_order[begin..end] {
            let high = &higher[high_index];
            if (high.candidate.moments.stddev - low.candidate.moments.stddev).abs()
                > 2.0f64 * MOMENT_TOLERANCE
            {
                continue;
            }
            let target_mean = (low.candidate.moments.mean + high.candidate.moments.mean) * 0.5;
            let target_stddev =
                (low.candidate.moments.stddev + high.candidate.moments.stddev) * 0.5;
            let residuals = [
                (low.candidate.moments.mean - target_mean).abs(),
                (low.candidate.moments.stddev - target_stddev).abs(),
                (high.candidate.moments.mean - target_mean).abs(),
                (high.candidate.moments.stddev - target_stddev).abs(),
            ];
            if residuals.iter().any(|&value| value > MOMENT_TOLERANCE) {
                continue;
            }
            let candidate = TargetInternal {
                mean: target_mean,
                stddev: target_stddev,
                residuals,
                lower: low.clone(),
                higher: high.clone(),
            };
            let key = (target_mean.to_bits(), target_stddev.to_bits());
            match deduplicated.get(&key) {
                Some(current) if target_representative_order(&candidate, current).is_ge() => {}
                _ => {
                    deduplicated.insert(key, candidate);
                }
            }
        }
    }
    deduplicated.into_values().collect()
}

fn target_representative_order(a: &TargetInternal, b: &TargetInternal) -> Ordering {
    max_value(&a.residuals)
        .total_cmp(&max_value(&b.residuals))
        .then_with(|| sum_squares(&a.residuals).total_cmp(&sum_squares(&b.residuals)))
        .then_with(|| {
            let ad = normalized_default_distance(a.lower.candidate)
                + normalized_default_distance(a.higher.candidate);
            let bd = normalized_default_distance(b.lower.candidate)
                + normalized_default_distance(b.higher.candidate);
            ad.total_cmp(&bd)
        })
        .then_with(|| recipe_order(&a.lower, &b.lower))
        .then_with(|| recipe_order(&a.higher, &b.higher))
}

fn recipe_order(a: &EligibleCandidate, b: &EligibleCandidate) -> Ordering {
    a.candidate
        .skin
        .total_cmp(&b.candidate.skin)
        .then_with(|| a.candidate.sclera.total_cmp(&b.candidate.sclera))
}

fn component_range(
    targets: &[FeasibleTargetRecord],
    component: fn(&FeasibleTargetRecord) -> f64,
) -> [f64; 2] {
    let minimum = targets
        .iter()
        .map(component)
        .min_by(f64::total_cmp)
        .unwrap();
    let maximum = targets
        .iter()
        .map(component)
        .max_by(f64::total_cmp)
        .unwrap();
    [minimum, maximum]
}

fn select_target_pair(
    lower_index: usize,
    targets: &[FeasibleTargetRecord],
    mean_range: [f64; 2],
    lower_cloud: &[EligibleCandidate],
    higher_cloud: &[EligibleCandidate],
    lower_basis: &PhotometricBasis,
    higher_basis: &PhotometricBasis,
) -> Result<SelectedTargetPair, String> {
    let range = mean_range[1] - mean_range[0];
    let q25 = mean_range[0] + 0.25f64 * range;
    let q75 = mean_range[0] + 0.75f64 * range;
    let mut best: Option<(&FeasibleTargetRecord, &FeasibleTargetRecord)> = None;
    for target0 in targets {
        for target1 in targets {
            let separation = target1.mean() - target0.mean();
            if separation < TARGET_SEPARATION || separation < 0.25f64 * range {
                continue;
            }
            let proposed = (target0, target1);
            if best.is_none_or(|current| target_pair_order(proposed, current, q25, q75).is_lt()) {
                best = Some(proposed);
            }
        }
    }
    let (target0, target1) = best.ok_or_else(|| {
        format!(
            "pair ({lower_index},{}) had no valid ordered target pair",
            lower_index + 4
        )
    })?;
    if (target0.mean_bits, target0.stddev_bits) == (target1.mean_bits, target1.stddev_bits) {
        return Err("selected target bit pairs were identical".into());
    }
    let cells = [
        solve_cell(0, ApertureRole::Lower, target0, lower_cloud, lower_basis)?,
        solve_cell(0, ApertureRole::Higher, target0, higher_cloud, higher_basis)?,
        solve_cell(1, ApertureRole::Lower, target1, lower_cloud, lower_basis)?,
        solve_cell(1, ApertureRole::Higher, target1, higher_cloud, higher_basis)?,
    ];
    let signed_cell_imbalances = [
        [
            (f64::from_bits(cells[1].recipe.mean_bits) - f64::from_bits(cells[0].recipe.mean_bits)),
            (f64::from_bits(cells[1].recipe.stddev_bits)
                - f64::from_bits(cells[0].recipe.stddev_bits)),
        ],
        [
            (f64::from_bits(cells[3].recipe.mean_bits) - f64::from_bits(cells[2].recipe.mean_bits)),
            (f64::from_bits(cells[3].recipe.stddev_bits)
                - f64::from_bits(cells[2].recipe.stddev_bits)),
        ],
    ];
    Ok(SelectedTargetPair {
        target0: target0.clone(),
        target1: target1.clone(),
        q25,
        q75,
        cells,
        signed_cell_imbalances,
    })
}

fn target_pair_order(
    a: (&FeasibleTargetRecord, &FeasibleTargetRecord),
    b: (&FeasibleTargetRecord, &FeasibleTargetRecord),
    q25: f64,
    q75: f64,
) -> Ordering {
    let anchor = |pair: (&FeasibleTargetRecord, &FeasibleTargetRecord)| {
        [(pair.0.mean() - q25).abs(), (pair.1.mean() - q75).abs()]
    };
    let aa = anchor(a);
    let ba = anchor(b);
    max_value(&aa)
        .total_cmp(&max_value(&ba))
        .then_with(|| (aa[0] + aa[1]).total_cmp(&(ba[0] + ba[1])))
        .then_with(|| {
            (a.0.stddev() - a.1.stddev())
                .abs()
                .total_cmp(&(b.0.stddev() - b.1.stddev()).abs())
        })
        .then_with(|| {
            max_eight(a)
                .total_cmp(&max_eight(b))
                .then_with(|| sum_eight_squares(a).total_cmp(&sum_eight_squares(b)))
        })
        .then_with(|| four_recipe_distance(a).total_cmp(&four_recipe_distance(b)))
        .then_with(|| target_tie_order(a.0, b.0))
        .then_with(|| target_tie_order(a.1, b.1))
}

fn max_eight(pair: (&FeasibleTargetRecord, &FeasibleTargetRecord)) -> f64 {
    let values = [
        pair.0.endpoint_residuals[0],
        pair.0.endpoint_residuals[1],
        pair.0.endpoint_residuals[2],
        pair.0.endpoint_residuals[3],
        pair.1.endpoint_residuals[0],
        pair.1.endpoint_residuals[1],
        pair.1.endpoint_residuals[2],
        pair.1.endpoint_residuals[3],
    ];
    max_value(&values)
}

fn sum_eight_squares(pair: (&FeasibleTargetRecord, &FeasibleTargetRecord)) -> f64 {
    let values = [
        pair.0.endpoint_residuals[0],
        pair.0.endpoint_residuals[1],
        pair.0.endpoint_residuals[2],
        pair.0.endpoint_residuals[3],
        pair.1.endpoint_residuals[0],
        pair.1.endpoint_residuals[1],
        pair.1.endpoint_residuals[2],
        pair.1.endpoint_residuals[3],
    ];
    sum_squares(&values)
}

fn four_recipe_distance(pair: (&FeasibleTargetRecord, &FeasibleTargetRecord)) -> f64 {
    let values = [
        pair.0.lower.recipe.normalized_default_distance,
        pair.0.higher.recipe.normalized_default_distance,
        pair.1.lower.recipe.normalized_default_distance,
        pair.1.higher.recipe.normalized_default_distance,
    ];
    left_sum(&values)
}

fn target_tie_order(a: &FeasibleTargetRecord, b: &FeasibleTargetRecord) -> Ordering {
    a.mean_bits
        .cmp(&b.mean_bits)
        .then_with(|| a.stddev_bits.cmp(&b.stddev_bits))
        .then_with(|| recipe_record_order(&a.lower.recipe, &b.lower.recipe))
        .then_with(|| recipe_record_order(&a.higher.recipe, &b.higher.recipe))
}

fn recipe_record_order(a: &RecipeRecord, b: &RecipeRecord) -> Ordering {
    f32::from_bits(a.skin_bits)
        .total_cmp(&f32::from_bits(b.skin_bits))
        .then_with(|| f32::from_bits(a.sclera_bits).total_cmp(&f32::from_bits(b.sclera_bits)))
}

fn solve_cell(
    target_number: usize,
    aperture_role: ApertureRole,
    target: &FeasibleTargetRecord,
    cloud: &[EligibleCandidate],
    basis: &PhotometricBasis,
) -> Result<CellRecord, String> {
    let target_mean = target.mean();
    let target_stddev = target.stddev();
    let best = cloud
        .iter()
        .min_by(|a, b| cell_order(a, b, target_mean, target_stddev))
        .ok_or_else(|| "empty cell solver population".to_string())?;
    let residual_mean = best.candidate.moments.mean - target_mean;
    let residual_stddev = best.candidate.moments.stddev - target_stddev;
    if residual_mean.abs() > MOMENT_TOLERANCE || residual_stddev.abs() > MOMENT_TOLERANCE {
        return Err("independent finite-D0 cell solver failed the target tolerance".into());
    }
    let canonical = canonical_record(best.candidate, basis, true)?;
    let default = render(&spec(
        best.candidate.aperture_index,
        DEFAULT_SKIN,
        DEFAULT_SCLERA,
    ));
    if canonical.covariates.measured_aperture_geometry.to_bits()
        != default.covariates.measured_aperture_geometry.to_bits()
        || canonical.covariates.measured_aperture_raster.to_bits()
            != default.covariates.measured_aperture_raster.to_bits()
    {
        return Err("selected cell changed frozen geometry".into());
    }
    let stereo = canonical_stereo(best.candidate)?;
    let tensor = canonical_tensor(&stereo);
    Ok(CellRecord {
        target_number,
        aperture_role,
        target_mean_bits: target.mean_bits,
        target_stddev_bits: target.stddev_bits,
        residual_mean,
        residual_stddev,
        recipe: recipe_record(best),
        canonical,
        default_geometry_bits: default.covariates.measured_aperture_geometry.to_bits(),
        default_raster_aperture_bits: default.covariates.measured_aperture_raster.to_bits(),
        tensor_sha256: tensor_sha256(&tensor),
    })
}

fn cell_order(
    a: &EligibleCandidate,
    b: &EligibleCandidate,
    target_mean: f64,
    target_stddev: f64,
) -> Ordering {
    let residual = |candidate: &EligibleCandidate| {
        [
            (candidate.candidate.moments.mean - target_mean).abs(),
            (candidate.candidate.moments.stddev - target_stddev).abs(),
        ]
    };
    let ar = residual(a);
    let br = residual(b);
    max_value(&ar)
        .total_cmp(&max_value(&br))
        .then_with(|| sum_squares(&ar).total_cmp(&sum_squares(&br)))
        .then_with(|| {
            normalized_default_distance(a.candidate)
                .total_cmp(&normalized_default_distance(b.candidate))
        })
        .then_with(|| recipe_order(a, b))
}

fn recipe_record(candidate: &EligibleCandidate) -> RecipeRecord {
    RecipeRecord {
        aperture_index: candidate.candidate.aperture_index,
        skin_bits: candidate.candidate.skin.to_bits(),
        sclera_bits: candidate.candidate.sclera.to_bits(),
        mean_bits: candidate.candidate.moments.mean.to_bits(),
        stddev_bits: candidate.candidate.moments.stddev.to_bits(),
        normalized_default_distance: normalized_default_distance(candidate.candidate),
        conditioning: candidate.conditioning.clone(),
    }
}

fn normalized_default_distance(candidate: Candidate) -> f64 {
    let ds = (candidate.skin as f64 - DEFAULT_SKIN as f64) / 0.30f64;
    let dc = (candidate.sclera as f64 - DEFAULT_SCLERA as f64) / 0.30f64;
    ds * ds + dc * dc
}

fn canonical_record(
    candidate: Candidate,
    basis: &PhotometricBasis,
    require_geometry: bool,
) -> Result<CanonicalRecord, String> {
    let predicted = basis.predict_pixels(candidate.skin, candidate.sclera);
    let canonical = render(&spec(
        candidate.aperture_index,
        candidate.skin,
        candidate.sclera,
    ));
    if predicted != canonical.pixels {
        return Err(format!(
            "fast/canonical byte mismatch at aperture {}",
            candidate.aperture_index
        ));
    }
    let predicted_moments = basis.moments(candidate.skin, candidate.sclera);
    let canonical_moments = gray_moments(&canonical.pixels);
    if predicted_moments.mean.to_bits() != candidate.moments.mean.to_bits()
        || predicted_moments.stddev.to_bits() != candidate.moments.stddev.to_bits()
        || canonical_moments.mean.to_bits() != candidate.moments.mean.to_bits()
        || canonical_moments.stddev.to_bits() != candidate.moments.stddev.to_bits()
    {
        return Err("fast/canonical/stream moment mismatch".into());
    }
    if !canonical.eye_like
        || !finite_covariates(&canonical.covariates)
        || canonical.covariates.frame_truncated
        || canonical.covariates.saturation_fraction != 0.0
    {
        return Err(format!(
            "canonical image gate failed at aperture {}",
            candidate.aperture_index
        ));
    }
    if require_geometry {
        let default = render(&spec(
            candidate.aperture_index,
            DEFAULT_SKIN,
            DEFAULT_SCLERA,
        ));
        if canonical.covariates.measured_aperture_geometry.to_bits()
            != default.covariates.measured_aperture_geometry.to_bits()
            || canonical.covariates.measured_aperture_raster.to_bits()
                != default.covariates.measured_aperture_raster.to_bits()
        {
            return Err("canonical cell geometry parity failed".into());
        }
    }
    Ok(CanonicalRecord {
        pixels_sha256: sha256_hex(&canonical.pixels),
        mean_bits: canonical_moments.mean.to_bits(),
        stddev_bits: canonical_moments.stddev.to_bits(),
        covariates: canonical.covariates,
        descriptive_covariates: descriptive_covariates(&canonical.pixels, canonical_moments),
        eye_like: canonical.eye_like,
        canonical_pixels: canonical.pixels,
    })
}

fn descriptive_covariates(pixels: &[u8], moments: GrayMoments) -> DescriptiveCovariates {
    let mut third = 0.0f64;
    let mut fourth = 0.0f64;
    for &pixel in pixels {
        let centered = pixel as f64 - moments.mean;
        let squared = centered * centered;
        third = third + squared * centered;
        fourth = fourth + squared * squared;
    }
    let n = pixels.len() as f64;
    let variance = moments.stddev * moments.stddev;
    let skewness = if moments.stddev > 0.0 {
        (third / n) / (variance * moments.stddev)
    } else {
        0.0
    };
    let excess_kurtosis = if variance > 0.0 {
        (fourth / n) / (variance * variance) - 3.0
    } else {
        0.0
    };
    let mut quadrant_sums = [0.0f64; 4];
    let mut quadrant_counts = [0usize; 4];
    for y in 0..SIDE {
        for x in 0..SIDE {
            let quadrant = usize::from(y >= SIDE / 2) * 2 + usize::from(x >= SIDE / 2);
            quadrant_sums[quadrant] = quadrant_sums[quadrant] + pixels[y * SIDE + x] as f64;
            quadrant_counts[quadrant] += 1;
        }
    }
    let quadrant_mean_gray =
        std::array::from_fn(|index| quadrant_sums[index] / quadrant_counts[index] as f64);
    DescriptiveCovariates {
        skewness,
        excess_kurtosis,
        quadrant_mean_gray,
    }
}

fn finite_covariates(covariates: &ImageCovariates) -> bool {
    [
        covariates.mean,
        covariates.stddev,
        covariates.edge_energy,
        covariates.saturation_fraction,
        covariates.visible_area_fraction,
        covariates.measured_aperture_geometry,
        covariates.measured_aperture_raster,
    ]
    .iter()
    .all(|value| value.is_finite())
}

fn max_value(values: &[f64]) -> f64 {
    values.iter().copied().reduce(f64::max).unwrap_or(0.0)
}

fn left_sum(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0f64, |sum, value| sum + value)
}

fn sum_squares(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .fold(0.0f64, |sum, value| sum + value * value)
}

pub fn build_stage_cases(plan: &RendererPlan, stage: Stage) -> Result<Vec<StageCase>, String> {
    if plan.version != VERSION
        || plan.global_axis != "whole_image_mean_gray"
        || plan.renderer_decision != "go"
        || plan.excluded_pairs != vec![[36, 40]]
    {
        return Err("renderer plan identity or decision was invalid".into());
    }
    let mut cases = Vec::with_capacity(17 + stage.expected_pair_count() * 4);
    for index in APERTURE_START..=APERTURE_END {
        if stage.accepts(index) {
            let candidate = Candidate {
                aperture_index: index,
                membership: MASK_D0,
                skin: DEFAULT_SKIN,
                sclera: DEFAULT_SCLERA,
                moments: moments_for(index, DEFAULT_SKIN, DEFAULT_SCLERA),
            };
            cases.push(make_stage_case(
                stage,
                CaseKind::DefaultS3,
                candidate,
                None,
                None,
                None,
            )?);
        }
    }
    for pair in plan
        .pair_plans
        .iter()
        .filter(|pair| pair.retained && stage.accepts(pair.lower_index))
    {
        let selected = pair
            .selected
            .as_ref()
            .ok_or_else(|| "retained pair lacked selected targets".to_string())?;
        for cell in &selected.cells {
            let expected_index = match cell.aperture_role {
                ApertureRole::Lower => pair.lower_index,
                ApertureRole::Higher => pair.higher_index,
            };
            if cell.recipe.aperture_index != expected_index {
                return Err("selected cell aperture role was inconsistent".into());
            }
            let candidate = Candidate {
                aperture_index: expected_index,
                membership: MASK_D0,
                skin: f32::from_bits(cell.recipe.skin_bits),
                sclera: f32::from_bits(cell.recipe.sclera_bits),
                moments: GrayMoments {
                    mean: f64::from_bits(cell.recipe.mean_bits),
                    stddev: f64::from_bits(cell.recipe.stddev_bits),
                },
            };
            let case = make_stage_case(
                stage,
                CaseKind::Factorial,
                candidate,
                Some(pair.lower_index),
                Some(cell.target_number),
                Some(cell.aperture_role),
            )?;
            if case.record.left_pixels_sha256 != cell.canonical.pixels_sha256
                || case.record.tensor_sha256 != cell.tensor_sha256
                || case.record.left_covariates != cell.canonical.covariates
            {
                return Err("selected cell did not reconstruct from renderer plan".into());
            }
            cases.push(case);
        }
    }
    let expected = 17 + stage.expected_pair_count() * 4;
    if cases.len() != expected {
        return Err(format!(
            "{stage:?} case count was {}, expected {expected}",
            cases.len()
        ));
    }
    Ok(cases)
}

impl StageCase {
    pub fn id(&self) -> &str {
        &self.record.id
    }
}

fn make_stage_case(
    stage: Stage,
    kind: CaseKind,
    candidate: Candidate,
    pair_lower_index: Option<usize>,
    target_number: Option<usize>,
    aperture_role: Option<ApertureRole>,
) -> Result<StageCase, String> {
    let stereo = canonical_stereo(candidate)?;
    let tensor = canonical_tensor(&stereo);
    if tensor.len() != 2 * SIDE * SIDE {
        return Err("canonical tensor shape was not [2,100,100]".into());
    }
    let id = match kind {
        CaseKind::DefaultS3 => format!("s3_i{:02}", candidate.aperture_index),
        CaseKind::Factorial => {
            let pair = pair_lower_index.unwrap();
            let target = target_number.unwrap();
            let role = match aperture_role.unwrap() {
                ApertureRole::Lower => "l",
                ApertureRole::Higher => "h",
            };
            format!("pair_i{pair:02}_{role}{target}")
        }
    };
    let record = CaseRecord {
        id,
        stage,
        kind,
        aperture_index: candidate.aperture_index,
        pair_lower_index,
        target_number,
        aperture_role,
        skin_bits: candidate.skin.to_bits(),
        sclera_bits: candidate.sclera.to_bits(),
        left_pixels_sha256: sha256_hex(&stereo.left.pixels),
        right_pixels_sha256: sha256_hex(&stereo.right.pixels),
        tensor_sha256: tensor_sha256(&tensor),
        left_covariates: stereo.left.covariates,
    };
    Ok(StageCase {
        record,
        stereo,
        tensor,
    })
}

fn canonical_stereo(candidate: Candidate) -> Result<RenderedStereo, String> {
    let eye_spec = spec(candidate.aperture_index, candidate.skin, candidate.sclera);
    let stereo = render_stereo(&eye_spec, &eye_spec, StereoPolicy::AnatomicalMirror);
    if !stereo.left.eye_like
        || !stereo.right.eye_like
        || !finite_covariates(&stereo.left.covariates)
        || !finite_covariates(&stereo.right.covariates)
        || stereo.left.covariates.frame_truncated
        || stereo.right.covariates.frame_truncated
        || stereo.left.covariates.saturation_fraction != 0.0
        || stereo.right.covariates.saturation_fraction != 0.0
    {
        return Err("canonical stereo image gate failed".into());
    }
    for y in 0..SIDE {
        for x in 0..SIDE {
            if stereo.right.pixels[y * SIDE + x] != stereo.left.pixels[y * SIDE + (SIDE - 1 - x)] {
                return Err("right channel was not the exact anatomical pixel mirror".into());
            }
        }
    }
    let checked = gray_moments(&stereo.left.pixels);
    if checked.mean.to_bits() != candidate.moments.mean.to_bits()
        || checked.stddev.to_bits() != candidate.moments.stddev.to_bits()
    {
        return Err("stage case moments differed from the renderer plan".into());
    }
    Ok(stereo)
}

fn moments_for(index: usize, skin: f32, sclera: f32) -> GrayMoments {
    let eye_spec = spec(index, DEFAULT_SKIN, DEFAULT_SCLERA);
    PhotometricBasis::from_spec(&eye_spec).moments(skin, sclera)
}

fn spec(index: usize, skin: f32, sclera: f32) -> SyntheticEyeSpec {
    SyntheticEyeSpec {
        aperture: aperture(index),
        skin_level: skin,
        sclera_level: sclera,
        ..SyntheticEyeSpec::default()
    }
}

fn aperture(index: usize) -> f32 {
    1.30f32 * index as f32 / 40.0f32
}

fn same_cases(a: &[StageCase], b: &[StageCase]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            a.record == b.record
                && a.stereo.left_spec == b.stereo.left_spec
                && a.stereo.right_spec == b.stereo.right_spec
                && a.stereo.left == b.stereo.left
                && a.stereo.right == b.stereo.right
                && a.tensor
                    .iter()
                    .zip(&b.tensor)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                && a.tensor.len() == b.tensor.len()
        })
}

fn domain_hash(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    use std::fmt::Write;
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

pub fn analyze_stage(
    stage: Stage,
    cases: &[StageCase],
    first: &[RawCaseResult],
    second_reverse: &[RawCaseResult],
) -> StageAnalysis {
    let first_bits = raw_bits(first);
    let second_bits = raw_bits(second_reverse);
    if let Err(error) = validate_inference_integrity(stage, cases, first, second_reverse) {
        return StageAnalysis {
            stage,
            status: StageStatus::InconclusiveArtifact,
            response_class: None,
            flags: Vec::new(),
            artifact_error: Some(error),
            first_raw_bits: first_bits,
            second_raw_bits: second_bits,
            eyes: None,
        };
    }
    if first.iter().any(|result| result.raw[0] as f64 <= 0.05f64) {
        return StageAnalysis {
            stage,
            status: StageStatus::InconclusiveRecognition,
            response_class: None,
            flags: Vec::new(),
            artifact_error: None,
            first_raw_bits: first_bits,
            second_raw_bits: second_bits,
            eyes: None,
        };
    }
    let eyes = std::array::from_fn(|eye| analyze_eye(stage, cases, first, eye));
    if eyes.iter().any(|eye| !eye.baseline.competent) {
        return StageAnalysis {
            stage,
            status: StageStatus::InconclusiveInsensitive,
            response_class: None,
            flags: analysis_flags(&eyes, None),
            artifact_error: None,
            first_raw_bits: first_bits,
            second_raw_bits: second_bits,
            eyes: Some(eyes),
        };
    }
    let response = joint_response(&eyes);
    let status = match response {
        ResponseClass::GeometrySupported => StageStatus::GeometrySupported,
        ResponseClass::AlternativePhotometricPath => StageStatus::AlternativePhotometricPath,
        ResponseClass::NoEvidence => StageStatus::NoEvidence,
        ResponseClass::Inconclusive => StageStatus::Inconclusive,
    };
    StageAnalysis {
        stage,
        status,
        response_class: Some(response),
        flags: analysis_flags(&eyes, Some(response)),
        artifact_error: None,
        first_raw_bits: first_bits,
        second_raw_bits: second_bits,
        eyes: Some(eyes),
    }
}

fn raw_bits(results: &[RawCaseResult]) -> Vec<(String, [u32; 5])> {
    results
        .iter()
        .map(|result| (result.id.clone(), result.raw.map(f32::to_bits)))
        .collect()
}

fn validate_inference_integrity(
    stage: Stage,
    cases: &[StageCase],
    first: &[RawCaseResult],
    second_reverse: &[RawCaseResult],
) -> Result<(), String> {
    let expected = 17 + stage.expected_pair_count() * 4;
    if cases.len() != expected || first.len() != expected || second_reverse.len() != expected {
        return Err(format!(
            "case/result lengths were {}/{}/{}, expected {expected}",
            cases.len(),
            first.len(),
            second_reverse.len()
        ));
    }
    for index in 0..expected {
        if cases[index].record.stage != stage || first[index].id != cases[index].record.id {
            return Err(format!("forward case identity mismatch at {index}"));
        }
        let reverse_index = expected - 1 - index;
        if second_reverse[reverse_index].id != cases[index].record.id {
            return Err(format!("reverse case identity mismatch at {reverse_index}"));
        }
        if first[index].raw.iter().any(|value| !value.is_finite())
            || second_reverse[reverse_index]
                .raw
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(format!("nonfinite raw output for {}", first[index].id));
        }
        if first[index].raw.map(f32::to_bits) != second_reverse[reverse_index].raw.map(f32::to_bits)
        {
            return Err(format!(
                "repeated inference bits differed for {}",
                first[index].id
            ));
        }
    }
    Ok(())
}

fn analyze_eye(
    stage: Stage,
    cases: &[StageCase],
    results: &[RawCaseResult],
    eye: usize,
) -> EyeAnalysis {
    let openness_index = 1 + eye;
    let s3_len = 17;
    let s3: Vec<f32> = results[..s3_len]
        .iter()
        .map(|result| result.raw[openness_index])
        .collect();
    let s3_by_index: BTreeMap<usize, f32> = cases[..s3_len]
        .iter()
        .zip(&s3)
        .map(|(case, &value)| (case.record.aperture_index, value))
        .collect();
    let retained: Vec<_> = (APERTURE_START..=PAIR_LOWER_END)
        .filter(|&index| index != 36 && stage.accepts(index))
        .collect();
    let baseline = baseline_metrics(stage, &s3, &s3_by_index, &retained);

    let mut g0 = Vec::with_capacity(retained.len());
    let mut g1 = Vec::with_capacity(retained.len());
    let mut p_l = Vec::with_capacity(retained.len());
    let mut p_h = Vec::with_capacity(retained.len());
    let mut geometry = Vec::with_capacity(retained.len());
    let mut photometric = Vec::with_capacity(retained.len());
    let mut modulation = Vec::with_capacity(retained.len());
    for (pair_position, &lower_index) in retained.iter().enumerate() {
        let offset = s3_len + pair_position * 4;
        debug_assert_eq!(cases[offset].record.pair_lower_index, Some(lower_index));
        let y_l0 = results[offset].raw[openness_index] as f64;
        let y_h0 = results[offset + 1].raw[openness_index] as f64;
        let y_l1 = results[offset + 2].raw[openness_index] as f64;
        let y_h1 = results[offset + 3].raw[openness_index] as f64;
        let this_g0 = y_h0 - y_l0;
        let this_g1 = y_h1 - y_l1;
        let this_p_l = y_l1 - y_l0;
        let this_p_h = y_h1 - y_h0;
        g0.push(this_g0);
        g1.push(this_g1);
        p_l.push(this_p_l);
        p_h.push(this_p_h);
        geometry.push((this_g0 + this_g1) * 0.5f64);
        photometric.push((this_p_l + this_p_h) * 0.5f64);
        modulation.push((this_g1 - this_g0) * 0.5f64);
    }
    let b = baseline.mean_absolute_gap4_delta;
    let g0 = vector_metrics(g0, b);
    let g1 = vector_metrics(g1, b);
    let p_l = vector_metrics(p_l, b);
    let p_h = vector_metrics(p_h, b);
    let geometry = vector_metrics(geometry, b);
    let photometric = vector_metrics(photometric, b);
    let modulation = vector_metrics(modulation, b);
    let provisional_response =
        response_for(&g0, &g1, &p_l, &p_h, &geometry, &photometric, &modulation);
    EyeAnalysis {
        baseline,
        g0,
        g1,
        p_l,
        p_h,
        geometry,
        photometric,
        modulation,
        provisional_response,
    }
}

fn baseline_metrics(
    stage: Stage,
    s3: &[f32],
    s3_by_index: &BTreeMap<usize, f32>,
    retained: &[usize],
) -> BaselineMetrics {
    let minimum = s3
        .iter()
        .map(|&value| value as f64)
        .min_by(f64::total_cmp)
        .unwrap();
    let maximum = s3
        .iter()
        .map(|&value| value as f64)
        .max_by(f64::total_cmp)
        .unwrap();
    let span = maximum - minimum;
    let spearman_rho = average_rank_spearman(s3);
    let deltas: Vec<f64> = retained
        .iter()
        .map(|&lower| {
            *s3_by_index.get(&(lower + 4)).unwrap() as f64
                - *s3_by_index.get(&lower).unwrap() as f64
        })
        .collect();
    let mean_absolute_gap4_delta =
        left_sum(&deltas.iter().map(|value| value.abs()).collect::<Vec<_>>()) / deltas.len() as f64;
    let positive_gap4_count = deltas.iter().filter(|&&value| value > DELTA4).count();
    let required_positive_gap4_count = concordance_count(retained.len());
    let competent = span >= 0.10f64
        && spearman_rho.is_some_and(|rho| rho >= 0.70f64)
        && mean_absolute_gap4_delta >= B4
        && positive_gap4_count >= required_positive_gap4_count
        && retained.len() == stage.expected_pair_count();
    BaselineMetrics {
        span,
        spearman_rho,
        mean_absolute_gap4_delta,
        positive_gap4_count,
        required_positive_gap4_count,
        competent,
    }
}

fn average_rank_spearman(values: &[f32]) -> Option<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then_with(|| a.cmp(&b)));
    let mut ranks = vec![0.0f64; n];
    let mut begin = 0usize;
    while begin < n {
        let bits = values[order[begin]].to_bits();
        let mut end = begin + 1;
        while end < n && values[order[end]].to_bits() == bits {
            end += 1;
        }
        let first_rank = (begin + 1) as f64;
        let last_rank = end as f64;
        let rank = (first_rank + last_rank) * 0.5f64;
        for &index in &order[begin..end] {
            ranks[index] = rank;
        }
        begin = end;
    }
    let aperture_ranks: Vec<f64> = (1..=n).map(|rank| rank as f64).collect();
    let mean_x = left_sum(&aperture_ranks) / n as f64;
    let mean_y = left_sum(&ranks) / n as f64;
    let mut numerator = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_y2 = 0.0f64;
    for index in 0..n {
        let x = aperture_ranks[index] - mean_x;
        let y = ranks[index] - mean_y;
        numerator = numerator + x * y;
        sum_x2 = sum_x2 + x * x;
        sum_y2 = sum_y2 + y * y;
    }
    if sum_x2 == 0.0 || sum_y2 == 0.0 {
        None
    } else {
        Some(numerator / (sum_x2 * sum_y2).sqrt())
    }
}

fn vector_metrics(values: Vec<f64>, baseline: f64) -> VectorMetrics {
    let n = values.len();
    let mean = left_sum(&values) / n as f64;
    let absolute: Vec<f64> = values.iter().map(|value| value.abs()).collect();
    let mean_absolute = left_sum(&absolute) / n as f64;
    let positive_count = values.iter().filter(|&&value| value > DELTA4).count();
    let negative_count = values.iter().filter(|&&value| -value > DELTA4).count();
    let flat_count = absolute.iter().filter(|&&value| value <= DELTA4).count();
    let required_concordance_count = concordance_count(n);
    let negative_mean =
        left_sum(&values.iter().map(|value| -*value).collect::<Vec<_>>()) / n as f64;
    let mean_to_baseline = (baseline > 0.0).then_some(mean / baseline);
    let mean_absolute_to_baseline = (baseline > 0.0).then_some(mean_absolute / baseline);
    let positive = mean >= E4
        && mean_to_baseline.is_some_and(|ratio| ratio >= 0.20f64)
        && positive_count >= required_concordance_count;
    let negative = negative_mean >= E4
        && baseline > 0.0
        && negative_mean / baseline >= 0.20f64
        && negative_count >= required_concordance_count;
    let flat = mean_absolute < E4
        && mean_absolute_to_baseline.is_some_and(|ratio| ratio <= 0.10f64)
        && flat_count >= required_concordance_count;
    VectorMetrics {
        values,
        mean,
        mean_absolute,
        mean_to_baseline,
        mean_absolute_to_baseline,
        positive_count,
        negative_count,
        flat_count,
        required_concordance_count,
        positive,
        negative,
        flat,
    }
}

fn concordance_count(n: usize) -> usize {
    (9 * n + 9) / 10
}

fn response_for(
    g0: &VectorMetrics,
    g1: &VectorMetrics,
    p_l: &VectorMetrics,
    p_h: &VectorMetrics,
    geometry: &VectorMetrics,
    photometric: &VectorMetrics,
    modulation: &VectorMetrics,
) -> ResponseClass {
    if g0.positive && g1.positive && geometry.positive && modulation.flat {
        ResponseClass::GeometrySupported
    } else if g0.flat
        && g1.flat
        && geometry.flat
        && modulation.flat
        && ((p_l.positive && p_h.positive && photometric.positive)
            || (p_l.negative && p_h.negative && photometric.negative))
    {
        ResponseClass::AlternativePhotometricPath
    } else if g0.flat
        && g1.flat
        && p_l.flat
        && p_h.flat
        && geometry.flat
        && photometric.flat
        && modulation.flat
    {
        ResponseClass::NoEvidence
    } else {
        ResponseClass::Inconclusive
    }
}

fn joint_response(eyes: &[EyeAnalysis; 2]) -> ResponseClass {
    let both = |predicate: fn(&EyeAnalysis) -> bool| eyes.iter().all(predicate);
    if both(|eye| {
        eye.g0.positive && eye.g1.positive && eye.geometry.positive && eye.modulation.flat
    }) {
        ResponseClass::GeometrySupported
    } else if both(|eye| eye.g0.flat && eye.g1.flat && eye.geometry.flat && eye.modulation.flat)
        && (both(|eye| eye.p_l.positive && eye.p_h.positive && eye.photometric.positive)
            || both(|eye| eye.p_l.negative && eye.p_h.negative && eye.photometric.negative))
    {
        ResponseClass::AlternativePhotometricPath
    } else if both(|eye| {
        eye.g0.flat
            && eye.g1.flat
            && eye.p_l.flat
            && eye.p_h.flat
            && eye.geometry.flat
            && eye.photometric.flat
            && eye.modulation.flat
    }) {
        ResponseClass::NoEvidence
    } else {
        ResponseClass::Inconclusive
    }
}

fn analysis_flags(eyes: &[EyeAnalysis; 2], response: Option<ResponseClass>) -> Vec<AnalysisFlag> {
    let mut flags = Vec::new();
    if eyes[0].provisional_response != eyes[1].provisional_response {
        flags.push(AnalysisFlag::EyeAsymmetric);
    }
    if eyes.iter().any(|eye| !eye.modulation.flat) {
        flags.push(AnalysisFlag::TargetModulated);
    }
    if eyes
        .iter()
        .any(|eye| eye.g0.negative || eye.g1.negative || eye.geometry.negative)
    {
        flags.push(AnalysisFlag::GeometryReversed);
    }
    if (eyes[0].photometric.positive && eyes[1].photometric.negative)
        || (eyes[0].photometric.negative && eyes[1].photometric.positive)
    {
        flags.push(AnalysisFlag::TargetDirectionMixed);
    }
    let signed_photometric = eyes
        .iter()
        .all(|eye| eye.p_l.positive && eye.p_h.positive && eye.photometric.positive)
        || eyes
            .iter()
            .all(|eye| eye.p_l.negative && eye.p_h.negative && eye.photometric.negative);
    if response == Some(ResponseClass::GeometrySupported) && signed_photometric {
        flags.push(AnalysisFlag::PhotometricSensitivity);
    }
    flags
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_conditioning() -> ConditioningRecord {
        ConditioningRecord {
            probes: std::array::from_fn(|_| ProbeRecord {
                skin_bits: 0,
                sclera_bits: 0,
                mean_bits: 0,
                stddev_bits: 0,
            }),
            matrix: [[1.0, 0.0], [0.0, 1.0]],
            sigma_max: 1.0,
            sigma_min: 1.0,
            kappa: 1.0,
        }
    }

    fn eligible(index: usize, ordinal: usize, mean: f64, stddev: f64) -> EligibleCandidate {
        EligibleCandidate {
            candidate: Candidate {
                aperture_index: index,
                membership: MASK_D0,
                skin: 0.40 + ordinal as f32 * 0.001,
                sclera: 0.75 + ordinal as f32 * 0.001,
                moments: GrayMoments { mean, stddev },
            },
            conditioning: dummy_conditioning(),
        }
    }

    fn brute_targets(
        lower: &[EligibleCandidate],
        higher: &[EligibleCandidate],
    ) -> Vec<TargetInternal> {
        let mut deduplicated = BTreeMap::<(u64, u64), TargetInternal>::new();
        for low in lower {
            for high in higher {
                let mean = (low.candidate.moments.mean + high.candidate.moments.mean) * 0.5;
                let stddev = (low.candidate.moments.stddev + high.candidate.moments.stddev) * 0.5;
                let residuals = [
                    (low.candidate.moments.mean - mean).abs(),
                    (low.candidate.moments.stddev - stddev).abs(),
                    (high.candidate.moments.mean - mean).abs(),
                    (high.candidate.moments.stddev - stddev).abs(),
                ];
                if residuals.iter().any(|&value| value > MOMENT_TOLERANCE) {
                    continue;
                }
                let next = TargetInternal {
                    mean,
                    stddev,
                    residuals,
                    lower: low.clone(),
                    higher: high.clone(),
                };
                let key = (mean.to_bits(), stddev.to_bits());
                match deduplicated.get(&key) {
                    Some(current) if target_representative_order(&next, current).is_ge() => {}
                    _ => {
                        deduplicated.insert(key, next);
                    }
                }
            }
        }
        deduplicated.into_values().collect()
    }

    fn target_signature(targets: &[TargetInternal]) -> Vec<(u64, u64, u32, u32, u32, u32)> {
        targets
            .iter()
            .map(|target| {
                (
                    target.mean.to_bits(),
                    target.stddev.to_bits(),
                    target.lower.candidate.skin.to_bits(),
                    target.lower.candidate.sclera.to_bits(),
                    target.higher.candidate.skin.to_bits(),
                    target.higher.candidate.sclera.to_bits(),
                )
            })
            .collect()
    }

    #[test]
    fn indexed_enumeration_equals_exhaustive_with_boundary_equality_and_ties() {
        let lower = vec![
            eligible(7, 8, 0.0, 0.0),
            eligible(7, 0, 10.0000, 4.0000),
            eligible(7, 1, 10.0010, 4.0010),
            eligible(7, 2, 10.0040, 4.0040),
            eligible(7, 3, 12.0000, 7.0000),
        ];
        let higher = vec![
            eligible(11, 9, 2.0 * MOMENT_TOLERANCE, 2.0 * MOMENT_TOLERANCE),
            eligible(11, 4, 10.0020, 4.0020),
            eligible(11, 5, 10.0010, 4.0010),
            eligible(11, 6, 9.9980, 3.9980),
            eligible(11, 7, 12.0100, 7.0000),
        ];
        let indexed = enumerate_feasible_targets(&lower, &higher);
        let brute = brute_targets(&lower, &higher);
        assert_eq!(target_signature(&indexed), target_signature(&brute));
        assert!(indexed.iter().any(|target| {
            target
                .residuals
                .iter()
                .any(|value| value.to_bits() == MOMENT_TOLERANCE.to_bits())
        }));
    }

    #[test]
    fn boundary_and_threshold_operators_are_frozen() {
        let at_skin = (0.30f64 + BOUNDARY_MARGIN) as f32;
        let at_sclera = (0.65f64 + BOUNDARY_MARGIN) as f32;
        assert!(passes_boundary(at_skin, at_sclera));
        assert!(!passes_boundary(
            f32::from_bits(at_skin.to_bits() - 1),
            at_sclera
        ));
        assert_eq!(concordance_count(15), 14);
        assert_eq!(concordance_count(14), 13);
        assert_eq!(DELTA4, 0.004);
        assert_eq!(E4, 0.2 / 33.0);
        assert_eq!(B4, 0.4 / 33.0);
    }

    #[test]
    fn average_rank_spearman_uses_bit_identical_ties_only() {
        assert_eq!(average_rank_spearman(&[1.0, 2.0, 3.0]), Some(1.0));
        let tied = average_rank_spearman(&[1.0, 1.0, 3.0]).unwrap();
        assert!(tied > 0.86 && tied < 0.87);
        assert!(average_rank_spearman(&[2.0, 2.0, 2.0]).is_none());
        let signed_zero = average_rank_spearman(&[-0.0, 0.0, 1.0]).unwrap();
        assert_eq!(signed_zero, 1.0);
    }

    #[test]
    fn vector_predicates_preserve_strict_and_inclusive_edges() {
        let positive = vector_metrics(vec![0.01; 15], B4);
        assert!(positive.positive);
        let flat = vector_metrics(vec![DELTA4; 15], 0.10);
        assert!(flat.flat);
        let not_flat = vector_metrics(vec![E4; 15], 1.0);
        assert!(!not_flat.flat);
    }

    #[test]
    fn response_rubric_prioritizes_supported_geometry_even_with_photometry() {
        let positive = vector_metrics(vec![0.02; 15], 0.02);
        let flat = vector_metrics(vec![0.0; 15], 0.02);
        assert_eq!(
            response_for(&positive, &positive, &positive, &positive, &positive, &positive, &flat),
            ResponseClass::GeometrySupported
        );
        assert_eq!(
            response_for(&flat, &flat, &positive, &positive, &flat, &positive, &flat),
            ResponseClass::AlternativePhotometricPath
        );
        assert_eq!(
            response_for(&flat, &flat, &flat, &flat, &flat, &flat, &flat),
            ResponseClass::NoEvidence
        );
    }

    fn analysis_fixture(stage: Stage) -> (Vec<StageCase>, Vec<RawCaseResult>) {
        let base_spec = SyntheticEyeSpec::default();
        let stereo = render_stereo(&base_spec, &base_spec, StereoPolicy::AnatomicalMirror);
        let tensor = canonical_tensor(&stereo);
        let mut cases = Vec::new();
        let mut results = Vec::new();
        let s3_indices: Vec<_> = (APERTURE_START..=APERTURE_END)
            .filter(|&index| stage.accepts(index))
            .collect();
        for (position, index) in s3_indices.into_iter().enumerate() {
            let id = format!("s3_i{index:02}");
            cases.push(StageCase {
                record: CaseRecord {
                    id: id.clone(),
                    stage,
                    kind: CaseKind::DefaultS3,
                    aperture_index: index,
                    pair_lower_index: None,
                    target_number: None,
                    aperture_role: None,
                    skin_bits: DEFAULT_SKIN.to_bits(),
                    sclera_bits: DEFAULT_SCLERA.to_bits(),
                    left_pixels_sha256: String::new(),
                    right_pixels_sha256: String::new(),
                    tensor_sha256: String::new(),
                    left_covariates: stereo.left.covariates,
                },
                stereo: stereo.clone(),
                tensor: tensor.clone(),
            });
            let openness = 0.10f32 + position as f32 * 0.05f32;
            results.push(RawCaseResult {
                id,
                raw: [0.90, openness, openness, 0.10, 0.10],
            });
        }
        for lower in
            (APERTURE_START..=PAIR_LOWER_END).filter(|&index| index != 36 && stage.accepts(index))
        {
            for (target, role, openness) in [
                (0, ApertureRole::Lower, 0.20),
                (0, ApertureRole::Higher, 0.23),
                (1, ApertureRole::Lower, 0.30),
                (1, ApertureRole::Higher, 0.33),
            ] {
                let role_name = match role {
                    ApertureRole::Lower => "l",
                    ApertureRole::Higher => "h",
                };
                let id = format!("pair_i{lower:02}_{role_name}{target}");
                cases.push(StageCase {
                    record: CaseRecord {
                        id: id.clone(),
                        stage,
                        kind: CaseKind::Factorial,
                        aperture_index: if role == ApertureRole::Lower {
                            lower
                        } else {
                            lower + 4
                        },
                        pair_lower_index: Some(lower),
                        target_number: Some(target),
                        aperture_role: Some(role),
                        skin_bits: DEFAULT_SKIN.to_bits(),
                        sclera_bits: DEFAULT_SCLERA.to_bits(),
                        left_pixels_sha256: String::new(),
                        right_pixels_sha256: String::new(),
                        tensor_sha256: String::new(),
                        left_covariates: stereo.left.covariates,
                    },
                    stereo: stereo.clone(),
                    tensor: tensor.clone(),
                });
                results.push(RawCaseResult {
                    id,
                    raw: [0.90, openness, openness, 0.10, 0.10],
                });
            }
        }
        (cases, results)
    }

    #[test]
    fn stage_analysis_uses_frozen_n15_and_n14_populations() {
        for stage in [Stage::Decision, Stage::Confirmation] {
            let (cases, first) = analysis_fixture(stage);
            let mut second = first.clone();
            second.reverse();
            let analysis = analyze_stage(stage, &cases, &first, &second);
            assert_eq!(analysis.status, StageStatus::GeometrySupported);
            assert_eq!(
                analysis.response_class,
                Some(ResponseClass::GeometrySupported)
            );
            let eyes = analysis.eyes.unwrap();
            for eye in eyes {
                assert_eq!(eye.geometry.values.len(), stage.expected_pair_count());
                assert!(eye.baseline.competent);
                assert!(eye.geometry.positive);
                assert!(eye.modulation.flat);
            }
            assert!(analysis
                .flags
                .contains(&AnalysisFlag::PhotometricSensitivity));
        }
    }

    #[test]
    fn inference_precedence_is_artifact_then_recognition_then_insensitive() {
        let stage = Stage::Decision;
        let (cases, first) = analysis_fixture(stage);
        let mut second = first.clone();
        second.reverse();
        second[0].raw[1] = f32::from_bits(second[0].raw[1].to_bits() + 1);
        assert_eq!(
            analyze_stage(stage, &cases, &first, &second).status,
            StageStatus::InconclusiveArtifact
        );

        let mut low_presence = first.clone();
        low_presence[0].raw[0] = 0.049;
        let mut low_presence_reverse = low_presence.clone();
        low_presence_reverse.reverse();
        assert_eq!(
            analyze_stage(stage, &cases, &low_presence, &low_presence_reverse).status,
            StageStatus::InconclusiveRecognition
        );

        let mut insensitive = first.clone();
        for result in &mut insensitive[..17] {
            result.raw[1] = 0.5;
            result.raw[2] = 0.5;
        }
        let mut insensitive_reverse = insensitive.clone();
        insensitive_reverse.reverse();
        let analysis = analyze_stage(stage, &cases, &insensitive, &insensitive_reverse);
        assert_eq!(analysis.status, StageStatus::InconclusiveInsensitive);
        assert!(serde_json::to_vec(&analysis).is_ok());
    }

    #[test]
    fn plan_hash_is_domain_separated_and_bit_sensitive() {
        let first = domain_hash(PLAN_HASH_DOMAIN, b"{}");
        assert_ne!(first, sha256_hex(b"{}"));
        assert_ne!(first, domain_hash(PLAN_HASH_DOMAIN, b"{ }"));
    }

    #[test]
    #[ignore = "full sealed-atlas preparation is an explicit release verification"]
    fn sealed_atlas_reproduces_the_frozen_29_pair_plan() {
        let atlas = Path::new("research-output/moment-atlas-49e13f0");
        let prepared = validate_atlas_and_prepare(atlas).unwrap();
        assert_eq!(prepared.plan.pair_plans.len(), 30);
        assert_eq!(prepared.plan.excluded_pairs, vec![[36, 40]]);
        assert_eq!(prepared.plan_sha256, FROZEN_PLAN_SHA256);
        assert_eq!(prepared.decision_cases.len(), 77);
        assert_eq!(prepared.confirmation_cases.len(), 73);
    }
}
