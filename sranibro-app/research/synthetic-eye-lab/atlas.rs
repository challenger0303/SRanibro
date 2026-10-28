//! Deterministic renderer-only Phase 1.3A moment-feasibility atlas.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::renderer::{
    gray_moments, render, GrayMoments, ImageCovariates, PhotometricBasis, SyntheticEyeSpec,
};

pub const VERSION: &str = "synthetic-eye-moment-atlas-v1";
pub const PREREGISTRATION_COMMIT: &str = "c92dbb2411c13d2f055ee7c1a67ee2b956d1e1a1";
pub const CANDIDATE_RECORD_LEN: usize = 28;
pub const CANDIDATE_STREAM_HEADER_LEN: usize = 24;

const UNIVERSE_START: usize = 7;
const UNIVERSE_END: usize = 40;
const APERTURE_COUNT: usize = UNIVERSE_END - UNIVERSE_START + 1;
const GRID_INTERVALS: usize = 128;
const DEFAULT_SKIN: f32 = 0.46;
const DEFAULT_SCLERA: f32 = 0.78;
const D0_SKIN: [f64; 2] = [0.30, 0.60];
const D0_SCLERA: [f64; 2] = [0.65, 0.95];
const MASK_D0: u8 = 1;
const MASK_D1: u8 = 2;
const MASK_D2: u8 = 4;
const EXPECTED_LEVEL_PAIRS: usize = 33_114;
const EXPECTED_CANDIDATES: usize = 1_125_876;
const EXPECTED_PAIRS: usize = 1_683;

#[derive(Debug)]
pub struct PreparedAtlas {
    pub candidate_stream: Vec<u8>,
    pub aperture_summaries_json: Vec<u8>,
    pub pair_summaries_json: Vec<u8>,
    pub canonical_checks_json: Vec<u8>,
    pub candidate_stream_sha256_repetitions: [String; 2],
    pub pair_summaries_sha256_repetitions: [String; 2],
    pub candidate_count: usize,
    pub pair_count: usize,
}

struct Preparation {
    candidate_stream: Vec<u8>,
    aperture_summaries_json: Vec<u8>,
    pair_summaries_json: Vec<u8>,
    canonical_checks_json: Vec<u8>,
    candidate_count: usize,
    pair_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
enum Domain {
    D0Legacy,
    D1PolarityPreserving,
    D2Mathematical,
}

impl Domain {
    const ALL: [Self; 3] = [
        Self::D0Legacy,
        Self::D1PolarityPreserving,
        Self::D2Mathematical,
    ];

    fn mask(self) -> u8 {
        match self {
            Self::D0Legacy => MASK_D0,
            Self::D1PolarityPreserving => MASK_D1,
            Self::D2Mathematical => MASK_D2,
        }
    }

    fn width(self) -> [f64; 2] {
        match self {
            Self::D0Legacy => [0.30, 0.30],
            Self::D1PolarityPreserving | Self::D2Mathematical => [1.0, 1.0],
        }
    }

    fn axis_step(self) -> [f64; 2] {
        let width = self.width();
        [
            width[0] / GRID_INTERVALS as f64,
            width[1] / GRID_INTERVALS as f64,
        ]
    }

    fn contains(self, skin: f32, sclera: f32) -> bool {
        match self {
            Self::D0Legacy => {
                (D0_SKIN[0] as f32..=D0_SKIN[1] as f32).contains(&skin)
                    && (D0_SCLERA[0] as f32..=D0_SCLERA[1] as f32).contains(&sclera)
            }
            Self::D1PolarityPreserving => {
                (0.0..=1.0).contains(&skin) && (0.0..=1.0).contains(&sclera) && sclera >= skin
            }
            Self::D2Mathematical => (0.0..=1.0).contains(&skin) && (0.0..=1.0).contains(&sclera),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct LevelPair {
    skin: f32,
    sclera: f32,
    membership: u8,
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    skin: f32,
    sclera: f32,
    membership: u8,
    moments: GrayMoments,
}

#[derive(Serialize)]
struct ApertureSummary {
    aperture_index: usize,
    aperture: f32,
    domain: Domain,
    candidate_count: usize,
    representative_count: usize,
    mean_range: [f64; 2],
    stddev_range: [f64; 2],
}

#[derive(Clone, Debug, Serialize)]
struct BoundaryMargins {
    skin_to_lower: f64,
    skin_to_upper: f64,
    sclera_to_lower: f64,
    sclera_to_upper: f64,
    polarity_gap: Option<f64>,
    polarity_euclidean_distance: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalDetails {
    sha256: String,
    covariates: ImageCovariates,
    eye_like: bool,
    mean_bits: u64,
    stddev_bits: u64,
}

#[derive(Clone, Debug, Serialize)]
struct SelectedLevel {
    skin: f32,
    sclera: f32,
    skin_bits: u32,
    sclera_bits: u32,
    mean: f64,
    stddev: f64,
    mean_bits: u64,
    stddev_bits: u64,
    boundary_margins: BoundaryMargins,
    canonical: CanonicalDetails,
    conditioning: Conditioning,
}

#[derive(Clone, Debug, Serialize)]
struct PairSummary {
    domain: Domain,
    lower_aperture_index: usize,
    higher_aperture_index: usize,
    lower_aperture: f32,
    higher_aperture: f32,
    lower: SelectedLevel,
    higher: SelectedLevel,
    signed_mean_difference: f64,
    signed_stddev_difference: f64,
    max_component_distance: f64,
    squared_euclidean_distance: f64,
    summed_normalized_default_distance: f64,
    candidate_midpoint_mean: f64,
    candidate_midpoint_stddev: f64,
    candidate_midpoint_mean_bits: u64,
    candidate_midpoint_stddev_bits: u64,
    interpretation_limit: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct Conditioning {
    available: bool,
    unavailable_reason: Option<&'static str>,
    matrix: Option<[[f64; 2]; 2]>,
    determinant: Option<f64>,
    sigma_max: Option<f64>,
    sigma_min: Option<f64>,
    condition_number: Option<f64>,
    condition_number_reason: Option<&'static str>,
}

#[derive(Clone, Debug, Serialize)]
struct CanonicalCheck {
    kind: &'static str,
    aperture_index: usize,
    skin_bits: u32,
    sclera_bits: u32,
    sha256: String,
    mean_bits: u64,
    stddev_bits: u64,
    covariates: ImageCovariates,
    eye_like: bool,
}

#[derive(Clone, Copy, Debug)]
struct PairChoice {
    lower: Candidate,
    higher: Candidate,
    signed_mean: f64,
    signed_stddev: f64,
    linf: f64,
    l2_sq: f64,
    default_distance: f64,
}

#[derive(Clone, Copy, Debug)]
struct ChoiceRecord {
    domain: Domain,
    lower_index: usize,
    higher_index: usize,
    choice: PairChoice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ConfigKey {
    aperture_index: usize,
    skin_bits: u32,
    sclera_bits: u32,
}

pub fn prepare() -> Result<PreparedAtlas, String> {
    let first = prepare_once()?;
    let second = prepare_once()?;
    let first_candidate_hash = sha256_hex(&first.candidate_stream);
    let second_candidate_hash = sha256_hex(&second.candidate_stream);
    let first_pair_hash = sha256_hex(&first.pair_summaries_json);
    let second_pair_hash = sha256_hex(&second.pair_summaries_json);

    if first_candidate_hash != second_candidate_hash
        || first.candidate_stream != second.candidate_stream
        || first.pair_summaries_json != second.pair_summaries_json
        || first.aperture_summaries_json != second.aperture_summaries_json
        || first.canonical_checks_json != second.canonical_checks_json
        || first.candidate_count != second.candidate_count
        || first.pair_count != second.pair_count
    {
        return Err("repeated atlas preparation was not byte-identical".into());
    }

    Ok(PreparedAtlas {
        candidate_stream: first.candidate_stream,
        aperture_summaries_json: first.aperture_summaries_json,
        pair_summaries_json: first.pair_summaries_json,
        canonical_checks_json: first.canonical_checks_json,
        candidate_stream_sha256_repetitions: [first_candidate_hash, second_candidate_hash],
        pair_summaries_sha256_repetitions: [first_pair_hash, second_pair_hash],
        candidate_count: first.candidate_count,
        pair_count: first.pair_count,
    })
}

fn prepare_once() -> Result<Preparation, String> {
    let levels = generate_levels();
    if levels.len() != EXPECTED_LEVEL_PAIRS {
        return Err(format!(
            "candidate level count was {}, expected {EXPECTED_LEVEL_PAIRS}",
            levels.len()
        ));
    }
    let domain_counts = Domain::ALL.map(|domain| {
        levels
            .iter()
            .filter(|level| level.membership & domain.mask() != 0)
            .count()
    });
    if domain_counts != [16_642, 24_858, 33_114] {
        return Err(format!(
            "domain candidate counts were {domain_counts:?}, expected [16642, 24858, 33114]"
        ));
    }

    let expected_default = SyntheticEyeSpec::default();
    if expected_default.skin_level.to_bits() != DEFAULT_SKIN.to_bits()
        || expected_default.sclera_level.to_bits() != DEFAULT_SCLERA.to_bits()
    {
        return Err("atlas defaults diverged from SyntheticEyeSpec defaults".into());
    }

    let mut stream = candidate_stream_header();
    let mut apertures = Vec::with_capacity(APERTURE_COUNT);
    let mut canonical_checks = Vec::new();
    let anchors = parity_anchors();

    for index in UNIVERSE_START..=UNIVERSE_END {
        let base_spec = spec(index, DEFAULT_SKIN, DEFAULT_SCLERA);
        let basis = PhotometricBasis::from_spec(&base_spec);
        for &(skin, sclera) in &anchors {
            canonical_checks.push(check_canonical(
                "fixed_anchor",
                index,
                skin,
                sclera,
                &basis,
            )?);
        }

        let mut candidates = Vec::with_capacity(levels.len());
        for level in &levels {
            let moments = basis.moments(level.skin, level.sclera);
            if !moments.mean.is_finite() || !moments.stddev.is_finite() {
                return Err(format!("nonfinite fast moment at aperture {index}"));
            }
            let candidate = Candidate {
                skin: level.skin,
                sclera: level.sclera,
                membership: level.membership,
                moments,
            };
            append_candidate_record(&mut stream, index, candidate);
            candidates.push(candidate);
        }
        apertures.push((basis, candidates));
    }

    let candidate_count = levels.len() * APERTURE_COUNT;
    if candidate_count != EXPECTED_CANDIDATES {
        return Err(format!(
            "candidate stream count was {candidate_count}, expected {EXPECTED_CANDIDATES}"
        ));
    }
    if stream.len() != CANDIDATE_STREAM_HEADER_LEN + candidate_count * CANDIDATE_RECORD_LEN {
        return Err("candidate stream byte length did not match its fixed schema".into());
    }

    let mut aperture_summaries = Vec::with_capacity(APERTURE_COUNT * Domain::ALL.len());
    let mut choices = Vec::with_capacity(EXPECTED_PAIRS);
    for domain in Domain::ALL {
        let clouds: Vec<Vec<Candidate>> = apertures
            .iter()
            .map(|(_, candidates)| representatives(candidates, domain))
            .collect();
        for (offset, cloud) in clouds.iter().enumerate() {
            let index = UNIVERSE_START + offset;
            let all_count = apertures[offset]
                .1
                .iter()
                .filter(|candidate| candidate.membership & domain.mask() != 0)
                .count();
            aperture_summaries.push(aperture_summary(index, domain, all_count, cloud)?);
        }
        for higher_offset in 1..APERTURE_COUNT {
            let tree = KdTree::build(&clouds[higher_offset]);
            for lower_offset in 0..higher_offset {
                let choice =
                    exact_nearest(&clouds[lower_offset], &clouds[higher_offset], &tree, domain)
                        .ok_or_else(|| "empty representative cloud".to_string())?;
                choices.push(ChoiceRecord {
                    domain,
                    lower_index: UNIVERSE_START + lower_offset,
                    higher_index: UNIVERSE_START + higher_offset,
                    choice,
                });
            }
        }
    }
    choices.sort_by_key(|record| (record.domain, record.lower_index, record.higher_index));
    if choices.len() != EXPECTED_PAIRS {
        return Err(format!(
            "pair summary count was {}, expected {EXPECTED_PAIRS}",
            choices.len()
        ));
    }
    assert_nested_winners(&choices)?;

    let selected_keys: BTreeSet<_> = choices
        .iter()
        .flat_map(|record| {
            [
                ConfigKey {
                    aperture_index: record.lower_index,
                    skin_bits: record.choice.lower.skin.to_bits(),
                    sclera_bits: record.choice.lower.sclera.to_bits(),
                },
                ConfigKey {
                    aperture_index: record.higher_index,
                    skin_bits: record.choice.higher.skin.to_bits(),
                    sclera_bits: record.choice.higher.sclera.to_bits(),
                },
            ]
        })
        .collect();
    let mut selected_details = BTreeMap::new();
    for key in selected_keys {
        let basis = &apertures[key.aperture_index - UNIVERSE_START].0;
        let check = check_canonical(
            "selected_pair",
            key.aperture_index,
            f32::from_bits(key.skin_bits),
            f32::from_bits(key.sclera_bits),
            basis,
        )?;
        selected_details.insert(
            key,
            CanonicalDetails {
                sha256: check.sha256.clone(),
                covariates: check.covariates,
                eye_like: check.eye_like,
                mean_bits: check.mean_bits,
                stddev_bits: check.stddev_bits,
            },
        );
        canonical_checks.push(check);
    }

    let mut pair_summaries = Vec::with_capacity(choices.len());
    for record in choices {
        let lower_key = ConfigKey {
            aperture_index: record.lower_index,
            skin_bits: record.choice.lower.skin.to_bits(),
            sclera_bits: record.choice.lower.sclera.to_bits(),
        };
        let higher_key = ConfigKey {
            aperture_index: record.higher_index,
            skin_bits: record.choice.higher.skin.to_bits(),
            sclera_bits: record.choice.higher.sclera.to_bits(),
        };
        let lower_basis = &apertures[record.lower_index - UNIVERSE_START].0;
        let higher_basis = &apertures[record.higher_index - UNIVERSE_START].0;
        let midpoint_mean =
            (record.choice.lower.moments.mean + record.choice.higher.moments.mean) * 0.5;
        let midpoint_std =
            (record.choice.lower.moments.stddev + record.choice.higher.moments.stddev) * 0.5;
        pair_summaries.push(PairSummary {
            domain: record.domain,
            lower_aperture_index: record.lower_index,
            higher_aperture_index: record.higher_index,
            lower_aperture: aperture(record.lower_index),
            higher_aperture: aperture(record.higher_index),
            lower: selected_level(
                record.domain,
                record.choice.lower,
                selected_details.get(&lower_key).unwrap().clone(),
                lower_basis,
            ),
            higher: selected_level(
                record.domain,
                record.choice.higher,
                selected_details.get(&higher_key).unwrap().clone(),
                higher_basis,
            ),
            signed_mean_difference: record.choice.signed_mean,
            signed_stddev_difference: record.choice.signed_stddev,
            max_component_distance: record.choice.linf,
            squared_euclidean_distance: record.choice.l2_sq,
            summed_normalized_default_distance: record.choice.default_distance,
            candidate_midpoint_mean: midpoint_mean,
            candidate_midpoint_stddev: midpoint_std,
            candidate_midpoint_mean_bits: midpoint_mean.to_bits(),
            candidate_midpoint_stddev_bits: midpoint_std.to_bits(),
            interpretation_limit: "winner on the frozen finite candidate sets only",
        });
    }

    let aperture_summaries_json = serde_json::to_vec_pretty(&aperture_summaries)
        .map_err(|error| format!("serialize aperture summaries: {error}"))?;
    let pair_summaries_json = serde_json::to_vec_pretty(&pair_summaries)
        .map_err(|error| format!("serialize pair summaries: {error}"))?;
    let canonical_checks_json = serde_json::to_vec_pretty(&canonical_checks)
        .map_err(|error| format!("serialize canonical checks: {error}"))?;

    Ok(Preparation {
        candidate_stream: stream,
        aperture_summaries_json,
        pair_summaries_json,
        canonical_checks_json,
        candidate_count,
        pair_count: pair_summaries.len(),
    })
}

fn generate_levels() -> Vec<LevelPair> {
    let mut levels = BTreeMap::<(u32, u32), u8>::new();
    for skin_index in 0..=GRID_INTERVALS {
        let skin = (D0_SKIN[0]
            + (D0_SKIN[1] - D0_SKIN[0]) * skin_index as f64 / GRID_INTERVALS as f64)
            as f32;
        for sclera_index in 0..=GRID_INTERVALS {
            let sclera = (D0_SCLERA[0]
                + (D0_SCLERA[1] - D0_SCLERA[0]) * sclera_index as f64 / GRID_INTERVALS as f64)
                as f32;
            *levels
                .entry((skin.to_bits(), sclera.to_bits()))
                .or_default() |= MASK_D0 | MASK_D1 | MASK_D2;
        }
    }
    for skin_index in 0..=GRID_INTERVALS {
        let skin = (skin_index as f64 / GRID_INTERVALS as f64) as f32;
        for sclera_index in 0..=GRID_INTERVALS {
            let sclera = (sclera_index as f64 / GRID_INTERVALS as f64) as f32;
            let mut mask = MASK_D2;
            if sclera >= skin {
                mask |= MASK_D1;
            }
            *levels
                .entry((skin.to_bits(), sclera.to_bits()))
                .or_default() |= mask;
        }
    }
    *levels
        .entry((DEFAULT_SKIN.to_bits(), DEFAULT_SCLERA.to_bits()))
        .or_default() |= MASK_D0 | MASK_D1 | MASK_D2;
    levels
        .into_iter()
        .map(|((skin, sclera), membership)| LevelPair {
            skin: f32::from_bits(skin),
            sclera: f32::from_bits(sclera),
            membership,
        })
        .collect()
}

fn candidate_stream_header() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(
        CANDIDATE_STREAM_HEADER_LEN + EXPECTED_CANDIDATES * CANDIDATE_RECORD_LEN,
    );
    bytes.extend_from_slice(b"SRATL3A1");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(CANDIDATE_RECORD_LEN as u32).to_le_bytes());
    bytes.extend_from_slice(&(EXPECTED_CANDIDATES as u64).to_le_bytes());
    bytes
}

fn append_candidate_record(bytes: &mut Vec<u8>, index: usize, candidate: Candidate) {
    bytes.push(index as u8);
    bytes.push(candidate.membership);
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(&candidate.skin.to_bits().to_le_bytes());
    bytes.extend_from_slice(&candidate.sclera.to_bits().to_le_bytes());
    bytes.extend_from_slice(&candidate.moments.mean.to_bits().to_le_bytes());
    bytes.extend_from_slice(&candidate.moments.stddev.to_bits().to_le_bytes());
}

fn representatives(candidates: &[Candidate], domain: Domain) -> Vec<Candidate> {
    let mut by_moment = BTreeMap::<(u64, u64), Candidate>::new();
    for &candidate in candidates
        .iter()
        .filter(|candidate| candidate.membership & domain.mask() != 0)
    {
        let key = (
            candidate.moments.mean.to_bits(),
            candidate.moments.stddev.to_bits(),
        );
        match by_moment.get(&key).copied() {
            Some(current) if !representative_better(candidate, current, domain) => {}
            _ => {
                by_moment.insert(key, candidate);
            }
        }
    }
    by_moment.into_values().collect()
}

fn representative_better(candidate: Candidate, current: Candidate, domain: Domain) -> bool {
    default_distance(candidate, domain)
        .total_cmp(&default_distance(current, domain))
        .then_with(|| candidate.skin.total_cmp(&current.skin))
        .then_with(|| candidate.sclera.total_cmp(&current.sclera))
        .is_lt()
}

fn default_distance(candidate: Candidate, domain: Domain) -> f64 {
    let width = domain.width();
    let skin = (candidate.skin as f64 - DEFAULT_SKIN as f64) / width[0];
    let sclera = (candidate.sclera as f64 - DEFAULT_SCLERA as f64) / width[1];
    skin * skin + sclera * sclera
}

fn pair_choice(lower: Candidate, higher: Candidate, domain: Domain) -> PairChoice {
    let signed_mean = higher.moments.mean - lower.moments.mean;
    let signed_stddev = higher.moments.stddev - lower.moments.stddev;
    PairChoice {
        lower,
        higher,
        signed_mean,
        signed_stddev,
        linf: signed_mean.abs().max(signed_stddev.abs()),
        l2_sq: signed_mean * signed_mean + signed_stddev * signed_stddev,
        default_distance: default_distance(lower, domain) + default_distance(higher, domain),
    }
}

fn choice_better(candidate: PairChoice, current: PairChoice) -> bool {
    candidate
        .linf
        .total_cmp(&current.linf)
        .then_with(|| candidate.l2_sq.total_cmp(&current.l2_sq))
        .then_with(|| {
            candidate
                .default_distance
                .total_cmp(&current.default_distance)
        })
        .then_with(|| candidate.lower.skin.total_cmp(&current.lower.skin))
        .then_with(|| candidate.lower.sclera.total_cmp(&current.lower.sclera))
        .then_with(|| candidate.higher.skin.total_cmp(&current.higher.skin))
        .then_with(|| candidate.higher.sclera.total_cmp(&current.higher.sclera))
        .is_lt()
}

fn exact_nearest(
    lower: &[Candidate],
    higher: &[Candidate],
    tree: &KdTree,
    domain: Domain,
) -> Option<PairChoice> {
    let mut best = None;
    for &query in lower {
        tree.query(higher, query, domain, &mut best);
    }
    best
}

#[cfg(test)]
fn brute_nearest(lower: &[Candidate], higher: &[Candidate], domain: Domain) -> Option<PairChoice> {
    let mut best = None;
    for &low in lower {
        for &high in higher {
            let candidate = pair_choice(low, high, domain);
            if best.is_none_or(|current| choice_better(candidate, current)) {
                best = Some(candidate);
            }
        }
    }
    best
}

struct KdTree {
    nodes: Vec<KdNode>,
    root: Option<usize>,
}

struct KdNode {
    point: usize,
    left: Option<usize>,
    right: Option<usize>,
    min: [f64; 2],
    max: [f64; 2],
}

impl KdTree {
    fn build(points: &[Candidate]) -> Self {
        let mut indices: Vec<_> = (0..points.len()).collect();
        let mut nodes = Vec::with_capacity(points.len());
        let root = build_kd(points, &mut indices, 0, &mut nodes);
        Self { nodes, root }
    }

    fn query(
        &self,
        points: &[Candidate],
        query: Candidate,
        domain: Domain,
        best: &mut Option<PairChoice>,
    ) {
        if let Some(root) = self.root {
            self.visit(root, points, query, domain, best);
        }
    }

    fn visit(
        &self,
        node_index: usize,
        points: &[Candidate],
        query: Candidate,
        domain: Domain,
        best: &mut Option<PairChoice>,
    ) {
        let node = &self.nodes[node_index];
        let bound = bbox_bound(query.moments, node.min, node.max);
        if !bound_can_improve(bound, *best) {
            return;
        }
        let candidate = pair_choice(query, points[node.point], domain);
        if best.is_none_or(|current| choice_better(candidate, current)) {
            *best = Some(candidate);
        }

        let mut children = [node.left, node.right];
        children.sort_by(|left, right| {
            child_bound(*left, &self.nodes, query.moments).cmp_total(child_bound(
                *right,
                &self.nodes,
                query.moments,
            ))
        });
        for child in children.into_iter().flatten() {
            self.visit(child, points, query, domain, best);
        }
    }
}

#[derive(Clone, Copy)]
struct Bound {
    linf: f64,
    l2_sq: f64,
}

impl Bound {
    fn cmp_total(self, other: Self) -> std::cmp::Ordering {
        self.linf
            .total_cmp(&other.linf)
            .then_with(|| self.l2_sq.total_cmp(&other.l2_sq))
    }
}

fn child_bound(child: Option<usize>, nodes: &[KdNode], query: GrayMoments) -> Bound {
    child.map_or(
        Bound {
            linf: f64::INFINITY,
            l2_sq: f64::INFINITY,
        },
        |index| bbox_bound(query, nodes[index].min, nodes[index].max),
    )
}

fn bound_can_improve(bound: Bound, best: Option<PairChoice>) -> bool {
    let Some(best) = best else {
        return true;
    };
    match bound.linf.total_cmp(&best.linf) {
        std::cmp::Ordering::Less => true,
        std::cmp::Ordering::Greater => false,
        std::cmp::Ordering::Equal => !bound.l2_sq.total_cmp(&best.l2_sq).is_gt(),
    }
}

fn bbox_bound(query: GrayMoments, min: [f64; 2], max: [f64; 2]) -> Bound {
    let dx = axis_distance(query.mean, min[0], max[0]);
    let dy = axis_distance(query.stddev, min[1], max[1]);
    Bound {
        linf: dx.max(dy),
        l2_sq: dx * dx + dy * dy,
    }
}

fn axis_distance(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
}

fn build_kd(
    points: &[Candidate],
    indices: &mut [usize],
    depth: usize,
    nodes: &mut Vec<KdNode>,
) -> Option<usize> {
    if indices.is_empty() {
        return None;
    }
    let axis = depth % 2;
    indices.sort_by(|left, right| point_order(points[*left], points[*right], axis));
    let middle = indices.len() / 2;
    let point = indices[middle];
    let (left_indices, right_with_middle) = indices.split_at_mut(middle);
    let right_indices = &mut right_with_middle[1..];
    let left = build_kd(points, left_indices, depth + 1, nodes);
    let right = build_kd(points, right_indices, depth + 1, nodes);
    let value = [points[point].moments.mean, points[point].moments.stddev];
    let mut min = value;
    let mut max = value;
    for child in [left, right].into_iter().flatten() {
        for axis in 0..2 {
            min[axis] = min[axis].min(nodes[child].min[axis]);
            max[axis] = max[axis].max(nodes[child].max[axis]);
        }
    }
    let node_index = nodes.len();
    nodes.push(KdNode {
        point,
        left,
        right,
        min,
        max,
    });
    Some(node_index)
}

fn point_order(left: Candidate, right: Candidate, axis: usize) -> std::cmp::Ordering {
    let left_values = [left.moments.mean, left.moments.stddev];
    let right_values = [right.moments.mean, right.moments.stddev];
    left_values[axis]
        .total_cmp(&right_values[axis])
        .then_with(|| left_values[1 - axis].total_cmp(&right_values[1 - axis]))
        .then_with(|| left.skin.total_cmp(&right.skin))
        .then_with(|| left.sclera.total_cmp(&right.sclera))
}

fn aperture_summary(
    index: usize,
    domain: Domain,
    candidate_count: usize,
    representatives: &[Candidate],
) -> Result<ApertureSummary, String> {
    let first = representatives
        .first()
        .ok_or_else(|| format!("empty cloud at aperture {index}"))?;
    let mut mean_range = [first.moments.mean; 2];
    let mut stddev_range = [first.moments.stddev; 2];
    for candidate in representatives.iter().skip(1) {
        mean_range[0] = mean_range[0].min(candidate.moments.mean);
        mean_range[1] = mean_range[1].max(candidate.moments.mean);
        stddev_range[0] = stddev_range[0].min(candidate.moments.stddev);
        stddev_range[1] = stddev_range[1].max(candidate.moments.stddev);
    }
    Ok(ApertureSummary {
        aperture_index: index,
        aperture: aperture(index),
        domain,
        candidate_count,
        representative_count: representatives.len(),
        mean_range,
        stddev_range,
    })
}

fn assert_nested_winners(records: &[ChoiceRecord]) -> Result<(), String> {
    let mut by_pair = BTreeMap::<(usize, usize), [Option<PairChoice>; 3]>::new();
    for record in records {
        let slot = match record.domain {
            Domain::D0Legacy => 0,
            Domain::D1PolarityPreserving => 1,
            Domain::D2Mathematical => 2,
        };
        by_pair
            .entry((record.lower_index, record.higher_index))
            .or_insert([None, None, None])[slot] = Some(record.choice);
    }
    for ((lower, higher), values) in by_pair {
        let [Some(d0), Some(d1), Some(d2)] = values else {
            return Err(format!("missing nested-domain result for {lower},{higher}"));
        };
        if distance_worse(d1, d0) || distance_worse(d2, d1) {
            return Err(format!(
                "nested finite cloud distance increased for pair {lower},{higher}"
            ));
        }
    }
    Ok(())
}

fn distance_worse(candidate: PairChoice, prior: PairChoice) -> bool {
    candidate
        .linf
        .total_cmp(&prior.linf)
        .then_with(|| candidate.l2_sq.total_cmp(&prior.l2_sq))
        .is_gt()
}

fn selected_level(
    domain: Domain,
    candidate: Candidate,
    canonical: CanonicalDetails,
    basis: &PhotometricBasis,
) -> SelectedLevel {
    SelectedLevel {
        skin: candidate.skin,
        sclera: candidate.sclera,
        skin_bits: candidate.skin.to_bits(),
        sclera_bits: candidate.sclera.to_bits(),
        mean: candidate.moments.mean,
        stddev: candidate.moments.stddev,
        mean_bits: candidate.moments.mean.to_bits(),
        stddev_bits: candidate.moments.stddev.to_bits(),
        boundary_margins: boundary_margins(domain, candidate),
        canonical,
        conditioning: conditioning(domain, candidate, basis),
    }
}

fn boundary_margins(domain: Domain, candidate: Candidate) -> BoundaryMargins {
    let (skin_bounds, sclera_bounds) = match domain {
        Domain::D0Legacy => (D0_SKIN, D0_SCLERA),
        Domain::D1PolarityPreserving | Domain::D2Mathematical => ([0.0, 1.0], [0.0, 1.0]),
    };
    let gap = (domain == Domain::D1PolarityPreserving)
        .then_some(candidate.sclera as f64 - candidate.skin as f64);
    BoundaryMargins {
        skin_to_lower: candidate.skin as f64 - skin_bounds[0],
        skin_to_upper: skin_bounds[1] - candidate.skin as f64,
        sclera_to_lower: candidate.sclera as f64 - sclera_bounds[0],
        sclera_to_upper: sclera_bounds[1] - candidate.sclera as f64,
        polarity_gap: gap,
        polarity_euclidean_distance: gap.map(|value| value / 2.0f64.sqrt()),
    }
}

fn conditioning(domain: Domain, center: Candidate, basis: &PhotometricBasis) -> Conditioning {
    let step = domain.axis_step();
    let skin_minus = (center.skin as f64 - step[0]) as f32;
    let skin_plus = (center.skin as f64 + step[0]) as f32;
    let sclera_minus = (center.sclera as f64 - step[1]) as f32;
    let sclera_plus = (center.sclera as f64 + step[1]) as f32;
    let probes = [
        (skin_minus, center.sclera),
        (skin_plus, center.sclera),
        (center.skin, sclera_minus),
        (center.skin, sclera_plus),
    ];
    if probes
        .iter()
        .any(|(skin, sclera)| !domain.contains(*skin, *sclera))
    {
        return unavailable_conditioning("central_probe_outside_domain");
    }
    if skin_minus.to_bits() == skin_plus.to_bits()
        || sclera_minus.to_bits() == sclera_plus.to_bits()
    {
        return unavailable_conditioning("central_probe_collapsed_after_f32_cast");
    }
    let sm = basis.moments(skin_minus, center.sclera);
    let sp = basis.moments(skin_plus, center.sclera);
    let cm = basis.moments(center.skin, sclera_minus);
    let cp = basis.moments(center.skin, sclera_plus);
    let skin_denominator = skin_plus as f64 - skin_minus as f64;
    let sclera_denominator = sclera_plus as f64 - sclera_minus as f64;
    let matrix = [
        [
            (sp.mean - sm.mean) / skin_denominator,
            (cp.mean - cm.mean) / sclera_denominator,
        ],
        [
            (sp.stddev - sm.stddev) / skin_denominator,
            (cp.stddev - cm.stddev) / sclera_denominator,
        ],
    ];
    let [a, b] = matrix[0];
    let [c, d] = matrix[1];
    if [a, b, c, d].iter().any(|value| !value.is_finite()) {
        return unavailable_conditioning("central_secant_nonfinite");
    }
    let r = (a + d).hypot(c - b);
    let s = (a - d).hypot(b + c);
    let sigma_max = (r + s) * 0.5;
    let sigma_min = (r - s).abs() * 0.5;
    let condition_number = (sigma_min > 0.0).then_some(sigma_max / sigma_min);
    Conditioning {
        available: true,
        unavailable_reason: None,
        matrix: Some(matrix),
        determinant: Some(a * d - b * c),
        sigma_max: Some(sigma_max),
        sigma_min: Some(sigma_min),
        condition_number,
        condition_number_reason: condition_number
            .is_none()
            .then_some("zero_minimum_singular_value"),
    }
}

fn unavailable_conditioning(reason: &'static str) -> Conditioning {
    Conditioning {
        available: false,
        unavailable_reason: Some(reason),
        matrix: None,
        determinant: None,
        sigma_max: None,
        sigma_min: None,
        condition_number: None,
        condition_number_reason: None,
    }
}

fn parity_anchors() -> Vec<(f32, f32)> {
    let mut anchors = BTreeSet::new();
    for skin_index in [0usize, 32, 64, 96, 128] {
        let skin = (D0_SKIN[0]
            + (D0_SKIN[1] - D0_SKIN[0]) * skin_index as f64 / GRID_INTERVALS as f64)
            as f32;
        for sclera_index in [0usize, 32, 64, 96, 128] {
            let sclera = (D0_SCLERA[0]
                + (D0_SCLERA[1] - D0_SCLERA[0]) * sclera_index as f64 / GRID_INTERVALS as f64)
                as f32;
            anchors.insert((skin.to_bits(), sclera.to_bits()));
        }
    }
    for skin_index in [0usize, 32, 64, 96, 128] {
        let skin = (skin_index as f64 / GRID_INTERVALS as f64) as f32;
        for sclera_index in [0usize, 32, 64, 96, 128] {
            let sclera = (sclera_index as f64 / GRID_INTERVALS as f64) as f32;
            anchors.insert((skin.to_bits(), sclera.to_bits()));
        }
    }
    anchors.insert((DEFAULT_SKIN.to_bits(), DEFAULT_SCLERA.to_bits()));
    anchors
        .into_iter()
        .map(|(skin, sclera)| (f32::from_bits(skin), f32::from_bits(sclera)))
        .collect()
}

fn check_canonical(
    kind: &'static str,
    index: usize,
    skin: f32,
    sclera: f32,
    basis: &PhotometricBasis,
) -> Result<CanonicalCheck, String> {
    let predicted = basis.predict_pixels(skin, sclera);
    let canonical = render(&spec(index, skin, sclera));
    if predicted != canonical.pixels {
        return Err(format!(
            "fast/canonical byte mismatch at aperture {index}, skin={skin}, sclera={sclera}"
        ));
    }
    let fast = basis.moments(skin, sclera);
    let checked = gray_moments(&canonical.pixels);
    if fast.mean.to_bits() != checked.mean.to_bits()
        || fast.stddev.to_bits() != checked.stddev.to_bits()
    {
        return Err(format!(
            "fast/canonical moment mismatch at aperture {index}, skin={skin}, sclera={sclera}"
        ));
    }
    if !finite_covariates(&canonical.covariates) {
        return Err(format!("nonfinite canonical covariate at aperture {index}"));
    }
    Ok(CanonicalCheck {
        kind,
        aperture_index: index,
        skin_bits: skin.to_bits(),
        sclera_bits: sclera.to_bits(),
        sha256: sha256_hex(&canonical.pixels),
        mean_bits: checked.mean.to_bits(),
        stddev_bits: checked.stddev.to_bits(),
        covariates: canonical.covariates,
        eye_like: canonical.eye_like,
    })
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

fn spec(index: usize, skin: f32, sclera: f32) -> SyntheticEyeSpec {
    SyntheticEyeSpec {
        aperture: aperture(index),
        skin_level: skin,
        sclera_level: sclera,
        ..SyntheticEyeSpec::default()
    }
}

fn aperture(index: usize) -> f32 {
    1.30 * index as f32 / 40.0
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        let _ = write!(output, "{byte:02x}");
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(skin: f32, sclera: f32, mean: f64, stddev: f64) -> Candidate {
        Candidate {
            skin,
            sclera,
            membership: MASK_D0 | MASK_D1 | MASK_D2,
            moments: GrayMoments { mean, stddev },
        }
    }

    fn assert_same_choice(kd: PairChoice, brute: PairChoice) {
        assert_eq!(kd.linf.to_bits(), brute.linf.to_bits());
        assert_eq!(kd.l2_sq.to_bits(), brute.l2_sq.to_bits());
        assert_eq!(
            kd.default_distance.to_bits(),
            brute.default_distance.to_bits()
        );
        assert_eq!(kd.lower.skin.to_bits(), brute.lower.skin.to_bits());
        assert_eq!(kd.lower.sclera.to_bits(), brute.lower.sclera.to_bits());
        assert_eq!(kd.higher.skin.to_bits(), brute.higher.skin.to_bits());
        assert_eq!(kd.higher.sclera.to_bits(), brute.higher.sclera.to_bits());
    }

    fn reduced_renderer_candidates(index: usize) -> Vec<Candidate> {
        let mut levels = BTreeMap::<(u32, u32), u8>::new();
        let mut insert = |skin: f32, sclera: f32, membership: u8| {
            *levels
                .entry((skin.to_bits(), sclera.to_bits()))
                .or_insert(0) |= membership;
        };
        for skin_step in [0usize, 64, 128] {
            let skin = (D0_SKIN[0]
                + (D0_SKIN[1] - D0_SKIN[0]) * skin_step as f64 / GRID_INTERVALS as f64)
                as f32;
            for sclera_step in [0usize, 64, 128] {
                let sclera = (D0_SCLERA[0]
                    + (D0_SCLERA[1] - D0_SCLERA[0]) * sclera_step as f64 / GRID_INTERVALS as f64)
                    as f32;
                insert(skin, sclera, MASK_D0 | MASK_D1 | MASK_D2);
            }
        }
        for skin_step in [0usize, 64, 128] {
            let skin = (skin_step as f64 / GRID_INTERVALS as f64) as f32;
            for sclera_step in [0usize, 64, 128] {
                let sclera = (sclera_step as f64 / GRID_INTERVALS as f64) as f32;
                let membership = MASK_D2 | if sclera >= skin { MASK_D1 } else { 0 };
                insert(skin, sclera, membership);
            }
        }
        insert(DEFAULT_SKIN, DEFAULT_SCLERA, MASK_D0 | MASK_D1 | MASK_D2);

        let basis = PhotometricBasis::from_spec(&spec(index, DEFAULT_SKIN, DEFAULT_SCLERA));
        levels
            .into_iter()
            .map(|((skin_bits, sclera_bits), membership)| {
                let skin = f32::from_bits(skin_bits);
                let sclera = f32::from_bits(sclera_bits);
                Candidate {
                    skin,
                    sclera,
                    membership,
                    moments: basis.moments(skin, sclera),
                }
            })
            .collect()
    }

    #[test]
    fn nested_level_sets_have_the_frozen_exact_counts() {
        let levels = generate_levels();
        assert_eq!(levels.len(), 33_114);
        assert_eq!(
            levels
                .iter()
                .filter(|level| level.membership & MASK_D0 != 0)
                .count(),
            16_642
        );
        assert_eq!(
            levels
                .iter()
                .filter(|level| level.membership & MASK_D1 != 0)
                .count(),
            24_858
        );
        assert_eq!(
            levels
                .iter()
                .filter(|level| level.membership & MASK_D2 != 0)
                .count(),
            33_114
        );
        assert!(levels.iter().all(|level| {
            level.membership & MASK_D0 == 0
                || level.membership & (MASK_D1 | MASK_D2) == (MASK_D1 | MASK_D2)
        }));
        assert!(levels
            .iter()
            .all(|level| { level.membership & MASK_D1 == 0 || level.membership & MASK_D2 != 0 }));
    }

    #[test]
    fn candidate_binary_schema_is_fixed_little_endian() {
        let mut bytes = candidate_stream_header();
        let value = candidate(0.25, 0.75, 12.5, 3.25);
        append_candidate_record(&mut bytes, 7, value);
        assert_eq!(bytes.len(), CANDIDATE_STREAM_HEADER_LEN + 28);
        assert_eq!(&bytes[..8], b"SRATL3A1");
        assert_eq!(&bytes[8..12], &1u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &(CANDIDATE_RECORD_LEN as u32).to_le_bytes());
        assert_eq!(&bytes[16..24], &(EXPECTED_CANDIDATES as u64).to_le_bytes());
        let record = &bytes[CANDIDATE_STREAM_HEADER_LEN..];
        assert_eq!(record[0], 7);
        assert_eq!(record[1], 7);
        assert_eq!(&record[2..4], &[0, 0]);
        assert_eq!(&record[4..8], &0.25f32.to_bits().to_le_bytes());
        assert_eq!(&record[8..12], &0.75f32.to_bits().to_le_bytes());
        assert_eq!(&record[12..20], &12.5f64.to_bits().to_le_bytes());
        assert_eq!(&record[20..28], &3.25f64.to_bits().to_le_bytes());
    }

    #[test]
    fn exact_kd_tree_matches_brute_force_with_ties() {
        let lower = vec![
            candidate(0.40, 0.70, 10.0, 2.0),
            candidate(0.45, 0.75, 11.0, 3.0),
            candidate(0.50, 0.80, 12.0, 4.0),
        ];
        let higher = vec![
            candidate(0.42, 0.72, 10.5, 2.5),
            candidate(0.46, 0.78, 11.0, 3.0),
            candidate(0.48, 0.82, 11.0, 3.0),
            candidate(0.55, 0.85, 14.0, 6.0),
        ];
        let tree = KdTree::build(&higher);
        let kd = exact_nearest(&lower, &higher, &tree, Domain::D0Legacy).unwrap();
        let brute = brute_nearest(&lower, &higher, Domain::D0Legacy).unwrap();
        assert_same_choice(kd, brute);
    }

    #[test]
    fn exact_kd_tree_matches_brute_force_for_every_reduced_cloud_pair() {
        let clouds: Vec<Vec<Candidate>> = (0..5)
            .map(|aperture| {
                (0..9)
                    .map(|level| {
                        let skin = 0.30 + level as f32 * 0.025;
                        let sclera = 0.65 + ((level * 5 + aperture) % 9) as f32 * 0.025;
                        let x = aperture as f64 * 0.37 + level as f64 * 0.11;
                        candidate(
                            skin,
                            sclera,
                            80.0 + x + (level % 3) as f64 * 0.02,
                            20.0 + x * x * 0.07 - (level % 2) as f64 * 0.03,
                        )
                    })
                    .collect()
            })
            .collect();
        for domain in Domain::ALL {
            for higher in 1..clouds.len() {
                let tree = KdTree::build(&clouds[higher]);
                for lower in 0..higher {
                    let kd = exact_nearest(&clouds[lower], &clouds[higher], &tree, domain).unwrap();
                    let brute = brute_nearest(&clouds[lower], &clouds[higher], domain).unwrap();
                    assert_same_choice(kd, brute);
                }
            }
        }
    }

    #[test]
    fn exact_kd_tree_matches_brute_force_for_renderer_reduced_grids() {
        let raw_clouds: Vec<_> = [7usize, 15, 24, 32, 40]
            .into_iter()
            .map(reduced_renderer_candidates)
            .collect();
        for domain in Domain::ALL {
            let clouds: Vec<_> = raw_clouds
                .iter()
                .map(|cloud| representatives(cloud, domain))
                .collect();
            for higher in 1..clouds.len() {
                let tree = KdTree::build(&clouds[higher]);
                for lower in 0..higher {
                    let kd = exact_nearest(&clouds[lower], &clouds[higher], &tree, domain).unwrap();
                    let brute = brute_nearest(&clouds[lower], &clouds[higher], domain).unwrap();
                    assert_same_choice(kd, brute);
                }
            }
        }
    }

    #[test]
    fn complete_pair_order_exercises_every_tie_break() {
        let choice = |linf: f64,
                      l2_sq: f64,
                      default_distance: f64,
                      lower_skin: f32,
                      lower_sclera: f32,
                      higher_skin: f32,
                      higher_sclera: f32| PairChoice {
            lower: candidate(lower_skin, lower_sclera, 10.0, 2.0),
            higher: candidate(higher_skin, higher_sclera, 11.0, 3.0),
            signed_mean: 1.0,
            signed_stddev: 1.0,
            linf,
            l2_sq,
            default_distance,
        };
        let assert_preferred = |winner, loser| {
            assert!(choice_better(winner, loser));
            assert!(!choice_better(loser, winner));
        };

        assert_preferred(
            choice(0.5, 4.0, 4.0, 0.4, 0.7, 0.4, 0.7),
            choice(1.0, 0.5, 0.5, 0.3, 0.6, 0.3, 0.6),
        );
        assert_preferred(
            choice(1.0, 1.0, 4.0, 0.4, 0.7, 0.4, 0.7),
            choice(1.0, 2.0, 0.5, 0.3, 0.6, 0.3, 0.6),
        );
        assert_preferred(
            choice(1.0, 2.0, 1.0, 0.4, 0.7, 0.4, 0.7),
            choice(1.0, 2.0, 2.0, 0.3, 0.6, 0.3, 0.6),
        );
        assert_preferred(
            choice(1.0, 2.0, 3.0, 0.3, 0.8, 0.5, 0.8),
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.4, 0.7),
        );
        assert_preferred(
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.5, 0.8),
            choice(1.0, 2.0, 3.0, 0.4, 0.8, 0.3, 0.6),
        );
        assert_preferred(
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.3, 0.8),
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.5, 0.6),
        );
        assert_preferred(
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.5, 0.6),
            choice(1.0, 2.0, 3.0, 0.4, 0.7, 0.5, 0.8),
        );
    }

    #[test]
    fn frozen_pair_count_is_561_per_domain() {
        let unordered = APERTURE_COUNT * (APERTURE_COUNT - 1) / 2;
        assert_eq!(unordered, 561);
        assert_eq!(unordered * Domain::ALL.len(), EXPECTED_PAIRS);
    }

    #[test]
    fn representative_ties_use_default_distance_then_levels() {
        let moments = GrayMoments {
            mean: 100.0,
            stddev: 20.0,
        };
        let candidates = vec![
            Candidate {
                skin: 0.30,
                sclera: 0.65,
                membership: 7,
                moments,
            },
            Candidate {
                skin: DEFAULT_SKIN,
                sclera: DEFAULT_SCLERA,
                membership: 7,
                moments,
            },
        ];
        let representatives = representatives(&candidates, Domain::D0Legacy);
        assert_eq!(representatives.len(), 1);
        assert_eq!(representatives[0].skin.to_bits(), DEFAULT_SKIN.to_bits());
        assert_eq!(
            representatives[0].sclera.to_bits(),
            DEFAULT_SCLERA.to_bits()
        );
    }

    #[test]
    fn central_secant_is_null_at_a_domain_boundary() {
        let spec = spec(20, D0_SKIN[0] as f32, DEFAULT_SCLERA);
        let basis = PhotometricBasis::from_spec(&spec);
        let center = Candidate {
            skin: D0_SKIN[0] as f32,
            sclera: DEFAULT_SCLERA,
            membership: 7,
            moments: basis.moments(D0_SKIN[0] as f32, DEFAULT_SCLERA),
        };
        let result = conditioning(Domain::D0Legacy, center, &basis);
        assert!(!result.available);
        assert_eq!(
            result.unavailable_reason,
            Some("central_probe_outside_domain")
        );
    }

    #[test]
    fn fixed_anchor_fast_path_matches_canonical_renderer() {
        let index = 23;
        let base = spec(index, DEFAULT_SKIN, DEFAULT_SCLERA);
        let basis = PhotometricBasis::from_spec(&base);
        for (skin, sclera) in parity_anchors() {
            check_canonical("test", index, skin, sclera, &basis).unwrap();
        }
    }
}
