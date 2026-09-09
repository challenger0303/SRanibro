//! Shared loader for research-only XR5 recording replays.
//!
//! Recording CSV columns are resolved by name rather than position.  The production
//! exporter may append fields in newer schema versions without changing older replay
//! tools; every source column is also retained in [`RecordingFrame::fields`] for tools
//! which understand those extensions.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::BufReader;
use std::path::{Component, Path, PathBuf};

use sranibro_rs::core::types::{DespeckleParams, FlattenParams, MlGeometry};
use sranibro_rs::geometry_calib::{
    GazeTarget, GeometryDataset, GeometrySample, SampleFamily, SampleKind,
};

#[derive(Clone, Debug)]
pub struct RecordingMetadata {
    /// Metadata schema. Historical recordings without an explicit version are v1.
    pub schema_version: u32,
    /// Exact metadata contents as exported by SRanibro.
    pub raw: String,
    /// Parsed `key=value` entries. Unknown entries are intentionally retained.
    pub fields: BTreeMap<String, String>,
}

impl RecordingMetadata {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct RecordingFrame {
    /// Source index declared by `samples.csv`; it need not equal the vector position.
    pub source_index: usize,
    /// Explicit split label from the CSV. The replay dataset also carries the split in
    /// `SampleKind`, but retaining this value lets audit tools report malformed exports.
    pub holdout: bool,
    pub left_file: PathBuf,
    pub right_file: PathBuf,
    /// All source columns, including future schema additions unknown to this loader.
    pub fields: BTreeMap<String, String>,
}

impl RecordingFrame {
    pub fn get(&self, name: &str) -> Option<&str> {
        self.fields.get(name).map(String::as_str)
    }
}

#[derive(Clone, Debug)]
pub struct LoadedRecording {
    pub dataset: GeometryDataset,
    pub frames: Vec<RecordingFrame>,
    pub metadata: RecordingMetadata,
    pub mirrors: [bool; 2],
    pub baseline: [MlGeometry; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
}

pub fn load_recording(root: &Path) -> Result<LoadedRecording, String> {
    let raw_metadata = std::fs::read_to_string(root.join("metadata.txt"))
        .map_err(|error| format!("read metadata.txt: {error}"))?;
    let metadata = parse_metadata(raw_metadata)?;

    let mirror_line = metadata
        .get("ml_mirror")
        .and_then(|value| value.strip_prefix('['))
        .and_then(|value| value.strip_suffix(']'))
        .ok_or("metadata.txt is missing ml_mirror=[L, R]")?;
    let mirror_values = mirror_line.split(',').map(str::trim).collect::<Vec<_>>();
    if mirror_values.len() != 2 {
        return Err("metadata ml_mirror must contain two booleans".into());
    }
    let mirrors = [
        parse_value(mirror_values[0])?,
        parse_value(mirror_values[1])?,
    ];

    let geometry_line = metadata
        .get("baseline_geometry")
        .ok_or("metadata.txt is missing baseline_geometry")?;
    let geometry_blocks = debug_struct_blocks(geometry_line, "MlGeometry");
    if geometry_blocks.len() != 2 {
        return Err("metadata baseline_geometry must contain two MlGeometry values".into());
    }
    let baseline = [
        parse_debug_geometry(geometry_blocks[0])?,
        parse_debug_geometry(geometry_blocks[1])?,
    ];

    let filters_line = metadata
        .get("filters")
        .ok_or("metadata.txt is missing filters")?;
    let despeckle_blocks = debug_struct_blocks(filters_line, "DespeckleParams");
    let flatten_blocks = debug_struct_blocks(filters_line, "FlattenParams");
    if despeckle_blocks.len() != 1 || flatten_blocks.len() != 1 {
        return Err("metadata filters must contain DespeckleParams and FlattenParams".into());
    }
    let despeckle = DespeckleParams {
        enabled: parse_value(debug_field(despeckle_blocks[0], "enabled")?)?,
        threshold: parse_value(debug_field(despeckle_blocks[0], "threshold")?)?,
        radius: parse_value(debug_field(despeckle_blocks[0], "radius")?)?,
    };
    let flatten = FlattenParams {
        enabled: parse_value(debug_field(flatten_blocks[0], "enabled")?)?,
        strength: parse_value(debug_field(flatten_blocks[0], "strength")?)?,
        radius: parse_value(debug_field(flatten_blocks[0], "radius")?)?,
    };

    let csv = std::fs::read_to_string(root.join("samples.csv"))
        .map_err(|error| format!("read samples.csv: {error}"))?;
    let mut lines = csv.lines();
    let header_line = lines.next().ok_or("samples.csv has no header")?;
    let headers = CsvHeaders::parse(header_line)?;
    let required = RequiredColumns::resolve(&headers)?;
    let gaze = GazeColumns::resolve(&headers)?;
    let commanded_target = headers.optional("commanded_target");
    let pupil = PupilColumns::resolve(&headers)?;
    let frame_generation = [
        headers.optional("frame_generation_l"),
        headers.optional("frame_generation_r"),
    ];
    let native_timestamp_us = headers.optional("native_timestamp_us");

    let mut samples = Vec::new();
    let mut frames = Vec::new();
    let mut phase_membership = BTreeMap::<usize, (bool, SampleFamily)>::new();
    for (zero_based_row, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let line_number = zero_based_row + 2;
        let values = line.split(',').collect::<Vec<_>>();
        if values.len() < headers.names.len() {
            return Err(format!(
                "samples.csv line {line_number} has {} values for {} headers",
                values.len(),
                headers.names.len()
            ));
        }

        let value = |column: usize| values[column].trim();
        let kind = parse_kind(value(required.kind))?;
        let left_file = recording_relative_path(value(required.left_file))?;
        let right_file = recording_relative_path(value(required.right_file))?;
        let left_path = root.join(&left_file);
        let right_path = root.join(&right_file);
        let left = decode_gray_png(&left_path)?;
        let right = decode_gray_png(&right_path)?;
        let declared_left = (
            parse_csv(value(required.left_width), line_number)?,
            parse_csv(value(required.left_height), line_number)?,
        );
        let declared_right = (
            parse_csv(value(required.right_width), line_number)?,
            parse_csv(value(required.right_height), line_number)?,
        );
        if left.0 != declared_left || right.0 != declared_right {
            return Err(format!(
                "PNG dimensions disagree at samples.csv line {line_number}"
            ));
        }

        let native_gaze = gaze
            .map(|columns| {
                [
                    parse_gaze([
                        value(columns.left[0]),
                        value(columns.left[1]),
                        value(columns.left[2]),
                    ]),
                    parse_gaze([
                        value(columns.right[0]),
                        value(columns.right[1]),
                        value(columns.right[2]),
                    ]),
                ]
            })
            .unwrap_or([None; 2]);
        let commanded_target = commanded_target
            .map(|column| value(column))
            .filter(|value| !value.is_empty())
            .map(|target| {
                target.parse::<GazeTarget>().map_err(|_| {
                    format!("unknown commanded_target {target:?} at samples.csv line {line_number}")
                })
            })
            .transpose()?;
        let native_pupil_pos = pupil
            .map(|columns| {
                [
                    parse_pair([value(columns.left[0]), value(columns.left[1])]),
                    parse_pair([value(columns.right[0]), value(columns.right[1])]),
                ]
            })
            .unwrap_or([None; 2]);

        let source_index = parse_csv(value(required.index), line_number)?;
        let holdout = parse_csv(value(required.holdout), line_number)?;
        let phase_index = parse_csv(value(required.phase_index), line_number)?;
        validate_row_membership(
            &mut phase_membership,
            phase_index,
            kind,
            holdout,
            line_number,
        )?;
        let fields = headers
            .names
            .iter()
            .zip(values.iter())
            .map(|(name, value)| (name.clone(), value.trim().to_owned()))
            .collect();
        frames.push(RecordingFrame {
            source_index,
            holdout,
            left_file,
            right_file,
            fields,
        });
        samples.push(GeometrySample {
            kind,
            commanded_target,
            expected_open: parse_optional(value(required.expected_open), line_number)?,
            phase_time_s: parse_csv(value(required.phase_time_s), line_number)?,
            left: left.1,
            right: right.1,
            left_size: declared_left,
            right_size: declared_right,
            brightness_affine: [
                [
                    parse_csv(value(required.left_gain), line_number)?,
                    parse_csv(value(required.left_bias), line_number)?,
                ],
                [
                    parse_csv(value(required.right_gain), line_number)?,
                    parse_csv(value(required.right_bias), line_number)?,
                ],
            ],
            native_open: [
                parse_optional(value(required.native_open_left), line_number)?,
                parse_optional(value(required.native_open_right), line_number)?,
            ],
            native_gaze,
            native_pupil_pos,
            frame_generation: [
                frame_generation[0]
                    .and_then(|column| value(column).parse().ok())
                    .unwrap_or(0),
                frame_generation[1]
                    .and_then(|column| value(column).parse().ok())
                    .unwrap_or(0),
            ],
            native_timestamp_us: native_timestamp_us.and_then(|column| value(column).parse().ok()),
            phase_index,
        });
    }
    if samples.is_empty() {
        return Err("recording has no samples".into());
    }

    Ok(LoadedRecording {
        dataset: GeometryDataset { samples },
        frames,
        metadata,
        mirrors,
        baseline,
        despeckle,
        flatten,
    })
}

fn validate_row_membership(
    phases: &mut BTreeMap<usize, (bool, SampleFamily)>,
    phase_index: usize,
    kind: SampleKind,
    explicit_holdout: bool,
    line_number: usize,
) -> Result<(), String> {
    let kind_holdout = kind.is_holdout();
    if explicit_holdout != kind_holdout {
        return Err(format!(
            "samples.csv line {line_number} has holdout={explicit_holdout}, but kind {:?} implies holdout={kind_holdout}",
            kind
        ));
    }
    let membership = (kind_holdout, kind.family());
    if let Some(previous) = phases.get(&phase_index) {
        if *previous != membership {
            return Err(format!(
                "samples.csv line {line_number} reuses phase_index {phase_index} across split or family: expected holdout={} family={:?}, found holdout={} family={:?}",
                previous.0, previous.1, membership.0, membership.1
            ));
        }
    } else {
        phases.insert(phase_index, membership);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CsvHeaders {
    names: Vec<String>,
    indices: HashMap<String, usize>,
}

impl CsvHeaders {
    fn parse(line: &str) -> Result<Self, String> {
        let mut names = Vec::new();
        let mut indices = HashMap::new();
        for (index, raw_name) in line.trim_start_matches('\u{feff}').split(',').enumerate() {
            let name = raw_name.trim();
            if name.is_empty() {
                return Err(format!("samples.csv header {index} is empty"));
            }
            if indices.insert(name.to_owned(), index).is_some() {
                return Err(format!("samples.csv has duplicate header {name:?}"));
            }
            names.push(name.to_owned());
        }
        if names.is_empty() {
            return Err("samples.csv has no headers".into());
        }
        Ok(Self { names, indices })
    }

    fn require(&self, name: &str) -> Result<usize, String> {
        self.indices
            .get(name)
            .copied()
            .ok_or_else(|| format!("samples.csv is missing required column {name:?}"))
    }

    fn optional(&self, name: &str) -> Option<usize> {
        self.indices.get(name).copied()
    }
}

#[derive(Clone, Copy)]
struct RequiredColumns {
    index: usize,
    kind: usize,
    holdout: usize,
    phase_index: usize,
    phase_time_s: usize,
    expected_open: usize,
    left_file: usize,
    right_file: usize,
    left_width: usize,
    left_height: usize,
    right_width: usize,
    right_height: usize,
    left_gain: usize,
    left_bias: usize,
    right_gain: usize,
    right_bias: usize,
    native_open_left: usize,
    native_open_right: usize,
}

impl RequiredColumns {
    fn resolve(headers: &CsvHeaders) -> Result<Self, String> {
        Ok(Self {
            index: headers.require("index")?,
            kind: headers.require("kind")?,
            holdout: headers.require("holdout")?,
            phase_index: headers.require("phase_index")?,
            phase_time_s: headers.require("phase_time_s")?,
            expected_open: headers.require("expected_open")?,
            left_file: headers.require("left_file")?,
            right_file: headers.require("right_file")?,
            left_width: headers.require("left_width")?,
            left_height: headers.require("left_height")?,
            right_width: headers.require("right_width")?,
            right_height: headers.require("right_height")?,
            left_gain: headers.require("left_gain")?,
            left_bias: headers.require("left_bias")?,
            right_gain: headers.require("right_gain")?,
            right_bias: headers.require("right_bias")?,
            native_open_left: headers.require("native_open_left")?,
            native_open_right: headers.require("native_open_right")?,
        })
    }
}

#[derive(Clone, Copy)]
struct GazeColumns {
    left: [usize; 3],
    right: [usize; 3],
}

impl GazeColumns {
    fn resolve(headers: &CsvHeaders) -> Result<Option<Self>, String> {
        let names = [
            "gaze_l_x", "gaze_l_y", "gaze_l_z", "gaze_r_x", "gaze_r_y", "gaze_r_z",
        ];
        let resolved = names.map(|name| headers.optional(name));
        if resolved.iter().all(Option::is_none) {
            return Ok(None);
        }
        if resolved.iter().any(Option::is_none) {
            return Err("samples.csv contains an incomplete native gaze column group".into());
        }
        let resolved = resolved.map(Option::unwrap);
        Ok(Some(Self {
            left: [resolved[0], resolved[1], resolved[2]],
            right: [resolved[3], resolved[4], resolved[5]],
        }))
    }
}

#[derive(Clone, Copy)]
struct PupilColumns {
    left: [usize; 2],
    right: [usize; 2],
}

impl PupilColumns {
    fn resolve(headers: &CsvHeaders) -> Result<Option<Self>, String> {
        let names = ["pupil_l_x", "pupil_l_y", "pupil_r_x", "pupil_r_y"];
        let resolved = names.map(|name| headers.optional(name));
        if resolved.iter().all(Option::is_none) {
            return Ok(None);
        }
        if resolved.iter().any(Option::is_none) {
            return Err("samples.csv contains an incomplete native pupil column group".into());
        }
        let resolved = resolved.map(Option::unwrap);
        Ok(Some(Self {
            left: [resolved[0], resolved[1]],
            right: [resolved[2], resolved[3]],
        }))
    }
}

fn parse_metadata(raw: String) -> Result<RecordingMetadata, String> {
    let fields = raw
        .lines()
        .filter_map(|line| line.split_once('='))
        .map(|(key, value)| (key.trim().to_owned(), value.trim().to_owned()))
        .collect::<BTreeMap<_, _>>();
    let schema_version = fields
        .get("schema_version")
        .map(|value| {
            value
                .parse::<u32>()
                .map_err(|_| format!("invalid metadata schema_version {value:?}"))
        })
        .transpose()?
        .unwrap_or(1);
    Ok(RecordingMetadata {
        schema_version,
        raw,
        fields,
    })
}

fn recording_relative_path(value: &str) -> Result<PathBuf, String> {
    let normalized = value.replace('\\', "/");
    let path = PathBuf::from(normalized);
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("invalid recording-relative path {value:?}"));
    }
    Ok(path)
}

fn debug_struct_blocks<'a>(line: &'a str, name: &str) -> Vec<&'a str> {
    let marker = format!("{name} {{");
    let mut blocks = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find(&marker) {
        let body = &rest[start + marker.len()..];
        let Some(end) = body.find('}') else {
            break;
        };
        blocks.push(body[..end].trim());
        rest = &body[end + 1..];
    }
    blocks
}

fn debug_field<'a>(block: &'a str, name: &str) -> Result<&'a str, String> {
    let marker = format!("{name}: ");
    let value = block
        .find(&marker)
        .map(|start| &block[start + marker.len()..])
        .ok_or_else(|| format!("debug metadata is missing field {name}"))?;
    Ok(value.split(',').next().unwrap_or(value).trim())
}

fn parse_debug_geometry(block: &str) -> Result<MlGeometry, String> {
    let mirror = debug_field(block, "mirror_h")?;
    let mirror_h = match mirror {
        "None" => None,
        "Some(true)" => Some(true),
        "Some(false)" => Some(false),
        _ => return Err(format!("invalid MlGeometry mirror_h {mirror:?}")),
    };
    Ok(MlGeometry {
        crop_left: parse_value(debug_field(block, "crop_left")?)?,
        crop_right: parse_value(debug_field(block, "crop_right")?)?,
        crop_top: parse_value(debug_field(block, "crop_top")?)?,
        crop_bottom: parse_value(debug_field(block, "crop_bottom")?)?,
        scale_x: parse_value(debug_field(block, "scale_x")?)?,
        scale_y: parse_value(debug_field(block, "scale_y")?)?,
        rotate_deg: parse_value(debug_field(block, "rotate_deg")?)?,
        mirror_h,
    })
}

pub fn decode_gray_png(path: &Path) -> Result<((u32, u32), Vec<u8>), String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("PNG header {}: {error}", path.display()))?;
    let mut pixels = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut pixels)
        .map_err(|error| format!("PNG data {}: {error}", path.display()))?;
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return Err(format!("{} is not grayscale u8", path.display()));
    }
    pixels.truncate(info.buffer_size());
    Ok(((info.width, info.height), pixels))
}

fn parse_value<T: std::str::FromStr>(value: &str) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid recorded value {value:?}"))
}

fn parse_csv<T: std::str::FromStr>(value: &str, line_number: usize) -> Result<T, String> {
    value
        .parse()
        .map_err(|_| format!("invalid value {value:?} at samples.csv line {line_number}"))
}

fn parse_optional(value: &str, line_number: usize) -> Result<Option<f32>, String> {
    if value.is_empty() {
        Ok(None)
    } else {
        parse_csv(value, line_number).map(Some)
    }
}

fn parse_gaze(columns: [&str; 3]) -> Option<[f32; 3]> {
    if columns.iter().any(|value| value.is_empty()) {
        return None;
    }
    let gaze = [
        columns[0].parse().ok()?,
        columns[1].parse().ok()?,
        columns[2].parse().ok()?,
    ];
    gaze.iter()
        .all(|value: &f32| value.is_finite())
        .then_some(gaze)
}

fn parse_pair(columns: [&str; 2]) -> Option<[f32; 2]> {
    if columns.iter().any(|value| value.is_empty()) {
        return None;
    }
    let pair = [columns[0].parse().ok()?, columns[1].parse().ok()?];
    pair.iter()
        .all(|value: &f32| value.is_finite())
        .then_some(pair)
}

fn parse_kind(value: &str) -> Result<SampleKind, String> {
    match value {
        "neutral" => Ok(SampleKind::Neutral),
        "gaze_sweep" => Ok(SampleKind::GazeSweep),
        "slow_close" => Ok(SampleKind::SlowClose),
        "natural_blinks" => Ok(SampleKind::NaturalBlinks),
        "closed" => Ok(SampleKind::Closed),
        "half_open" => Ok(SampleKind::HalfOpen),
        "holdout_neutral" => Ok(SampleKind::HoldoutNeutral),
        "holdout_gaze_sweep" => Ok(SampleKind::HoldoutGazeSweep),
        "holdout_slow_close" => Ok(SampleKind::HoldoutSlowClose),
        "holdout_natural_blinks" => Ok(SampleKind::HoldoutNaturalBlinks),
        "holdout_closed" => Ok(SampleKind::HoldoutClosed),
        "holdout_half_open" => Ok(SampleKind::HoldoutHalfOpen),
        _ => Err(format!("unknown sample kind {value:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_are_name_based_and_keep_future_fields() {
        let headers = CsvHeaders::parse(
            "kind,index,future_value,holdout,phase_index,phase_time_s,expected_open,left_file,right_file,left_width,left_height,right_width,right_height,left_gain,left_bias,right_gain,right_bias,native_open_left,native_open_right",
        )
        .unwrap();
        let required = RequiredColumns::resolve(&headers).unwrap();
        assert_eq!(required.kind, 0);
        assert_eq!(required.index, 1);
        assert_eq!(headers.optional("future_value"), Some(2));
        assert!(GazeColumns::resolve(&headers).unwrap().is_none());
    }

    #[test]
    fn native_gaze_columns_are_all_or_none() {
        let partial = CsvHeaders::parse("gaze_l_x,gaze_l_y").unwrap();
        assert!(GazeColumns::resolve(&partial).is_err());
        let complete =
            CsvHeaders::parse("gaze_r_z,gaze_l_z,gaze_r_y,gaze_l_y,gaze_r_x,gaze_l_x").unwrap();
        let gaze = GazeColumns::resolve(&complete).unwrap().unwrap();
        assert_eq!(gaze.left, [5, 3, 1]);
        assert_eq!(gaze.right, [4, 2, 0]);
    }

    #[test]
    fn schema_three_extensions_are_optional_and_name_based() {
        let headers =
            CsvHeaders::parse("pupil_r_y,commanded_target,pupil_l_x,pupil_r_x,pupil_l_y,future_v4")
                .unwrap();
        let pupil = PupilColumns::resolve(&headers).unwrap().unwrap();
        assert_eq!(pupil.left, [2, 4]);
        assert_eq!(pupil.right, [3, 0]);
        assert_eq!(headers.optional("commanded_target"), Some(1));
        assert_eq!("up_left".parse::<GazeTarget>(), Ok(GazeTarget::UpLeft));
        assert_eq!(parse_pair(["0.25", "0.75"]), Some([0.25, 0.75]));
        assert_eq!(parse_pair(["", "0.75"]), None);
        assert_eq!(parse_pair(["NaN", "0.75"]), None);
    }

    #[test]
    fn parses_recorded_debug_geometry_instead_of_assuming_the_preset() {
        let line = "Some([MlGeometry { crop_left: 0.01, crop_right: 0.4, crop_top: 0.12, crop_bottom: 0.18, scale_x: 1.0, scale_y: 1.2, rotate_deg: -31.0, mirror_h: None }, MlGeometry { crop_left: 0.4, crop_right: 0.02, crop_top: 0.14, crop_bottom: 0.16, scale_x: 1.0, scale_y: 1.1, rotate_deg: 29.0, mirror_h: Some(true) }])";
        let blocks = debug_struct_blocks(line, "MlGeometry");
        assert_eq!(blocks.len(), 2);
        let left = parse_debug_geometry(blocks[0]).unwrap();
        let right = parse_debug_geometry(blocks[1]).unwrap();
        assert_eq!(left.crop_left, 0.01);
        assert_eq!(left.rotate_deg, -31.0);
        assert_eq!(left.mirror_h, None);
        assert_eq!(right.crop_right, 0.02);
        assert_eq!(right.scale_y, 1.1);
        assert_eq!(right.mirror_h, Some(true));
    }

    #[test]
    fn metadata_defaults_historical_recordings_to_schema_one() {
        let metadata = parse_metadata("device=pimax_xr5\nunknown=future\n".into()).unwrap();
        assert_eq!(metadata.schema_version, 1);
        assert_eq!(metadata.get("unknown"), Some("future"));
    }

    #[test]
    fn explicit_holdout_must_match_the_kind() {
        let mut phases = BTreeMap::new();
        assert!(validate_row_membership(&mut phases, 7, SampleKind::Neutral, false, 2).is_ok());
        let error = validate_row_membership(&mut phases, 8, SampleKind::HoldoutNeutral, false, 3)
            .unwrap_err();
        assert!(error.contains("holdout=false"));
        assert!(error.contains("implies holdout=true"));
    }

    #[test]
    fn phase_index_cannot_cross_a_split_or_family() {
        let mut phases = BTreeMap::new();
        validate_row_membership(&mut phases, 12, SampleKind::Neutral, false, 2).unwrap();
        validate_row_membership(&mut phases, 12, SampleKind::Neutral, false, 3).unwrap();

        let family_error =
            validate_row_membership(&mut phases, 12, SampleKind::Closed, false, 4).unwrap_err();
        assert!(family_error.contains("phase_index 12"));
        assert!(family_error.contains("family"));

        let mut phases = BTreeMap::new();
        validate_row_membership(&mut phases, 9, SampleKind::Neutral, false, 5).unwrap();
        let split_error =
            validate_row_membership(&mut phases, 9, SampleKind::HoldoutNeutral, true, 6)
                .unwrap_err();
        assert!(split_error.contains("phase_index 9"));
        assert!(split_error.contains("holdout=false"));
        assert!(split_error.contains("holdout=true"));
    }

    #[test]
    fn recording_paths_must_remain_inside_the_recording() {
        assert_eq!(
            recording_relative_path("frames/000001_left.png").unwrap(),
            PathBuf::from("frames/000001_left.png")
        );
        assert!(recording_relative_path("../outside.png").is_err());
        assert!(recording_relative_path("C:\\outside.png").is_err());
    }
}
