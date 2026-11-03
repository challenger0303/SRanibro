
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const MODEL_REL: &str = "model/EyePrediction/00-0000.params_opencl.params";

pub fn model_path_in(root: &Path) -> PathBuf {
    let direct = root.join(MODEL_REL);
    if direct.is_file() || !cfg!(target_os = "linux") {
        return direct;
    }
    let mut path = root.to_path_buf();
    for component in MODEL_REL.split('/') {
        let exact = path.join(component);
        if exact.exists() {
            path = exact;
            continue;
        }
        let matches: Vec<_> = std::fs::read_dir(&path)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .filter(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .eq_ignore_ascii_case(component)
            })
            .take(2)
            .collect();
        if matches.len() != 1 {
            return direct;
        }
        path = matches[0].path();
    }
    path
}

pub const PSVR2_ONLY_BUILD: bool = cfg!(feature = "psvr2-only");

pub const XR5_ONLY_BUILD: bool = cfg!(feature = "xr5-only");

pub const fn config_file_name() -> &'static str {
    if PSVR2_ONLY_BUILD {
        "sranibro-psvr2.toml"
    } else if cfg!(feature = "xr5-research-model") {
        "sranibro-xr5-research.toml"
    } else if XR5_ONLY_BUILD {
        "sranibro-xr5.toml"
    } else {
        "sranibro.toml"
    }
}

fn nonempty(o: &Option<String>) -> Option<String> {
    o.as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

pub fn base_dir() -> PathBuf {
    static CACHE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    CACHE.get_or_init(resolve_base_dir).clone()
}

fn dir_writable(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let probe = dir.join(format!(".sranibro_wtest_{}", std::process::id()));
    let ok = std::fs::write(&probe, b"x").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

fn resolve_base_dir() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join(config_file_name()).is_file() && dir_writable(dir) {
                return dir.to_path_buf();
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        for base in linux_config_candidates(
            std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
            std::env::var_os("HOME").map(PathBuf::from),
        ) {
            let dir = base.join("SRanibro");
            if dir_writable(&dir) {
                return dir;
            }
        }
    }
    #[cfg(windows)]
    for var in ["APPDATA", "LOCALAPPDATA"] {
        if let Ok(base) = std::env::var(var) {
            let d = PathBuf::from(base).join("SRanibro");
            if dir_writable(&d) {
                return d;
            }
        }
    }
    let tmp = std::env::temp_dir().join("SRanibro");
    if dir_writable(&tmp) {
        return tmp;
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
    {
        if dir_writable(&dir) {
            return dir;
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

#[cfg(target_os = "linux")]
fn linux_config_candidates(xdg: Option<PathBuf>, home: Option<PathBuf>) -> Vec<PathBuf> {
    xdg.filter(|path| path.is_absolute())
        .into_iter()
        .chain(
            home.filter(|path| path.is_absolute())
                .map(|path| path.join(".config")),
        )
        .collect()
}

pub fn config_path() -> PathBuf {
    base_dir().join(config_file_name())
}

pub fn calib_path() -> PathBuf {
    base_dir().join("sranibro_calib.toml")
}

pub fn calib_path_for(device: &str) -> PathBuf {
    let canonical = canonical_device_key(device);
    let safe: String = canonical
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();
    let safe = if safe.is_empty() {
        "unknown".to_string()
    } else {
        safe
    };
    base_dir().join(format!("sranibro_calib_{safe}.toml"))
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    #[cfg(any(feature = "xr5-research-model", feature = "external-eyelid-model"))]
    pub xr5_research: crate::ml::xr5_research::Settings,
    pub assets: Assets,
    pub hmd: Hmd,
    pub output: Output,
    pub ui: Ui,
    pub tuning: crate::core::eye_state::Tuning,
    #[serde(skip)]
    save_blocked_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Assets {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sranipal_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub varjo_lib: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brow_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wide_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python_exe: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vr_eyebrow_dir: Option<String>,
}

fn varjo_lib_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut push = |base: PathBuf| {
        out.push(
            base.join("Varjo")
                .join("varjo-compositor")
                .join("VarjoLib.dll"),
        );
        out.push(base.join("Varjo").join("varjo-openxr").join("VarjoLib.dll"));
    };
    for var in ["ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"] {
        if let Ok(p) = std::env::var(var) {
            if !p.is_empty() {
                push(PathBuf::from(p));
            }
        }
    }
    out.push(PathBuf::from(
        r"C:\Program Files\Varjo\varjo-compositor\VarjoLib.dll",
    ));
    out
}

fn bundled_brow_model_path_in(exe_dir: &Path) -> Option<PathBuf> {
    [
        exe_dir.join("eyebrow.bin"),
        exe_dir.join("brow.bin"),
        exe_dir.join("models").join("eyebrow.bin"),
        exe_dir.join("models").join("brow.bin"),
    ]
    .into_iter()
    .find(|path| path.is_file())
}

fn bundled_brow_model_path() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
        .and_then(|dir| bundled_brow_model_path_in(&dir))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EyeMapping {
    pub swap_eyes: bool,
    pub flip_image: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub swap_gaze_eyes: Option<bool>,
    pub flip_gaze_x: bool,
    pub ml_mirror_l: bool,
    pub ml_mirror_r: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeCorrection {
    pub enabled: bool,
    pub offset_x_deg: [f32; 2],
    pub offset_y_deg: [f32; 2],
    pub scale_x: [f32; 2],
    pub scale_y: [f32; 2],
    pub vergence_deg: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamAirProfile {
    pub schema_version: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eyechip_serial: Option<String>,
    pub calibrated_unix: u64,
    pub baseline: [f32; 2],
    pub blink_depth: [f32; 2],
    pub wide_supported: [bool; 2],
    pub wide_snr: [f32; 2],
    pub quality_score: f32,
    pub pupil_center: [[f32; 2]; 2],
    pub pupil_center_valid: [bool; 2],
}

impl Default for DreamAirProfile {
    fn default() -> Self {
        Self {
            schema_version: 1,
            eyechip_serial: None,
            calibrated_unix: 0,
            baseline: [0.6; 2],
            blink_depth: [0.2; 2],
            wide_supported: [true; 2],
            wide_snr: [0.0; 2],
            quality_score: 0.0,
            pupil_center: [[0.5; 2]; 2],
            pupil_center_valid: [false; 2],
        }
    }
}

impl Default for GazeCorrection {
    fn default() -> Self {
        Self {
            enabled: false,
            offset_x_deg: [0.0; 2],
            offset_y_deg: [0.0; 2],
            scale_x: [1.0; 2],
            scale_y: [1.0; 2],
            vergence_deg: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EyelidResponseProfile {
    pub schema_version: u32,
    pub manual_range: bool,
    pub open_point_offset: [f32; 2],
    pub closed_point_depth: [f32; 2],
    pub wide_start: [f32; 2],
    pub wide_full: [f32; 2],
    pub squeeze_start: [f32; 2],
    pub squeeze_full: [f32; 2],
    pub close_depth_scale: [f32; 2],
    pub curve_mid_output: [f32; 2],
    pub blink_close_ms: f32,
    pub snap_gate_open: f32,
    pub auto_reseat: bool,
}

impl Default for EyelidResponseProfile {
    fn default() -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            manual_range: true,
            open_point_offset: [0.03; 2],
            closed_point_depth: [0.40; 2],
            wide_start: [0.0; 2],
            wide_full: [1.0; 2],
            squeeze_start: [0.0; 2],
            squeeze_full: [1.0; 2],
            close_depth_scale: [1.0; 2],
            curve_mid_output: [0.5; 2],
            blink_close_ms: 0.0,
            snap_gate_open: 1.0,
            auto_reseat: false,
        }
    }
}

impl EyelidResponseProfile {
    pub const SCHEMA_VERSION: u32 = 1;
    pub const OPEN_POINT_OFFSET_MIN: f32 = 0.03;
    pub const OPEN_POINT_OFFSET_MAX: f32 = 0.20;
    pub const CLOSED_POINT_DEPTH_MIN: f32 = 0.08;
    pub const CLOSED_POINT_DEPTH_MAX: f32 = 0.46;
    pub const MIN_MANUAL_RANGE: f32 = 0.05;
    pub const EXPRESSION_START_MIN: f32 = 0.0;
    pub const EXPRESSION_START_MAX: f32 = 0.95;
    pub const EXPRESSION_FULL_MIN: f32 = 0.05;
    pub const EXPRESSION_FULL_MAX: f32 = 1.0;
    pub const MIN_EXPRESSION_RANGE: f32 = 0.05;
    pub const CLOSE_DEPTH_SCALE_MIN: f32 = 0.85;
    pub const CLOSE_DEPTH_SCALE_MAX: f32 = 1.15;
    pub const CURVE_MID_OUTPUT_MIN: f32 = 0.35;
    pub const CURVE_MID_OUTPUT_MAX: f32 = 0.65;
    pub const BLINK_CLOSE_MS_MIN: f32 = 0.0;
    pub const BLINK_CLOSE_MS_MAX: f32 = 160.0;
    pub const SNAP_GATE_OPEN_MIN: f32 = 0.15;
    pub const SNAP_GATE_OPEN_MAX: f32 = 1.0;

    pub fn sanitized(mut self) -> Self {
        if self.schema_version != Self::SCHEMA_VERSION {
            return Self::default();
        }
        let defaults = Self::default();
        for eye in 0..2 {
            self.open_point_offset[eye] = sanitize_profile_scalar(
                self.open_point_offset[eye],
                defaults.open_point_offset[eye],
                Self::OPEN_POINT_OFFSET_MIN,
                Self::OPEN_POINT_OFFSET_MAX,
            );
            self.closed_point_depth[eye] = sanitize_profile_scalar(
                self.closed_point_depth[eye],
                defaults.closed_point_depth[eye],
                Self::CLOSED_POINT_DEPTH_MIN,
                Self::CLOSED_POINT_DEPTH_MAX,
            )
            .max(self.open_point_offset[eye] + Self::MIN_MANUAL_RANGE);
            for (start, full) in [
                (&mut self.wide_start[eye], &mut self.wide_full[eye]),
                (&mut self.squeeze_start[eye], &mut self.squeeze_full[eye]),
            ] {
                *start = sanitize_profile_scalar(
                    *start,
                    0.0,
                    Self::EXPRESSION_START_MIN,
                    Self::EXPRESSION_START_MAX,
                );
                *full = sanitize_profile_scalar(
                    *full,
                    1.0,
                    Self::EXPRESSION_FULL_MIN,
                    Self::EXPRESSION_FULL_MAX,
                );
                *start = (*start).min(*full - Self::MIN_EXPRESSION_RANGE);
            }
            self.close_depth_scale[eye] = sanitize_profile_scalar(
                self.close_depth_scale[eye],
                defaults.close_depth_scale[eye],
                Self::CLOSE_DEPTH_SCALE_MIN,
                Self::CLOSE_DEPTH_SCALE_MAX,
            );
            self.curve_mid_output[eye] = sanitize_profile_scalar(
                self.curve_mid_output[eye],
                defaults.curve_mid_output[eye],
                Self::CURVE_MID_OUTPUT_MIN,
                Self::CURVE_MID_OUTPUT_MAX,
            );
        }
        self.blink_close_ms = sanitize_profile_scalar(
            self.blink_close_ms,
            defaults.blink_close_ms,
            Self::BLINK_CLOSE_MS_MIN,
            Self::BLINK_CLOSE_MS_MAX,
        );
        self.snap_gate_open = sanitize_profile_scalar(
            self.snap_gate_open,
            defaults.snap_gate_open,
            Self::SNAP_GATE_OPEN_MIN,
            Self::SNAP_GATE_OPEN_MAX,
        );
        self
    }

    pub fn is_compatible(&self) -> bool {
        self.schema_version == Self::SCHEMA_VERSION
            && self.open_point_offset.iter().all(|value| {
                value.is_finite()
                    && (Self::OPEN_POINT_OFFSET_MIN..=Self::OPEN_POINT_OFFSET_MAX).contains(value)
            })
            && self
                .closed_point_depth
                .iter()
                .enumerate()
                .all(|(eye, value)| {
                    value.is_finite()
                        && (Self::CLOSED_POINT_DEPTH_MIN..=Self::CLOSED_POINT_DEPTH_MAX)
                            .contains(value)
                        && *value >= self.open_point_offset[eye] + Self::MIN_MANUAL_RANGE
                })
            && [
                (&self.wide_start, &self.wide_full),
                (&self.squeeze_start, &self.squeeze_full),
            ]
            .into_iter()
            .all(|(starts, fulls)| {
                (0..2).all(|eye| {
                    starts[eye].is_finite()
                        && fulls[eye].is_finite()
                        && (Self::EXPRESSION_START_MIN..=Self::EXPRESSION_START_MAX)
                            .contains(&starts[eye])
                        && (Self::EXPRESSION_FULL_MIN..=Self::EXPRESSION_FULL_MAX)
                            .contains(&fulls[eye])
                        && fulls[eye] >= starts[eye] + Self::MIN_EXPRESSION_RANGE
                })
            })
            && self.close_depth_scale.iter().all(|value| {
                value.is_finite()
                    && (Self::CLOSE_DEPTH_SCALE_MIN..=Self::CLOSE_DEPTH_SCALE_MAX).contains(value)
            })
            && self.curve_mid_output.iter().all(|value| {
                value.is_finite()
                    && (Self::CURVE_MID_OUTPUT_MIN..=Self::CURVE_MID_OUTPUT_MAX).contains(value)
            })
            && self.blink_close_ms.is_finite()
            && (Self::BLINK_CLOSE_MS_MIN..=Self::BLINK_CLOSE_MS_MAX).contains(&self.blink_close_ms)
            && self.snap_gate_open.is_finite()
            && (Self::SNAP_GATE_OPEN_MIN..=Self::SNAP_GATE_OPEN_MAX).contains(&self.snap_gate_open)
    }
}

fn sanitize_profile_scalar(value: f32, default: f32, min: f32, max: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        default
    }
}

pub fn canonical_device_key(device: &str) -> String {
    match device.trim().to_lowercase().replace(' ', "_").as_str() {
        "pimax-xr5" | "pimax_xr5" | "xr5" | "dream-air" | "dream_air" | "dreamair" => {
            "pimax_xr5".into()
        }
        "pimax-vr4" | "pimax_vr4" | "vr4" | "crystal" | "crystal_super" => "pimax_vr4".into(),
        "starvr-one" | "starvr_one" | "starvr" => "starvr".into(),
        "varjo-native" | "varjo_native" | "varjo" => "varjo".into(),
        "varjo-stream" | "varjo_stream" | "varjo_mjpeg" => "varjo_mjpeg".into(),
        "pimax-vr4-dll" | "pimax_vr4_dll" | "pimax-stream" | "pimax_stream" | "pimax_dll" => {
            "pimax_dll".into()
        }
        "playstation-vr2" | "playstation_vr2" | "ps-vr2" | "ps_vr2" | "psvr2" => "psvr2".into(),
        "vive-pro-eye" | "vive_pro_eye" | "vpe" => "vpe".into(),
        other => other.replace('-', "_"),
    }
}

pub fn supports_gaze_correction(device: &str) -> bool {
    matches!(
        canonical_device_key(device).as_str(),
        "pimax_xr5" | "pimax_vr4"
    )
}

pub fn supports_photometric_fit(device: &str) -> bool {
    matches!(
        canonical_device_key(device).as_str(),
        "pimax_vr4" | "pimax_dll" | "varjo" | "varjo_mjpeg"
    )
}

pub fn running_device_key(configured: &str, adapter_name: &str) -> String {
    if canonical_device_key(configured) == "auto" {
        canonical_device_key(adapter_name)
    } else {
        canonical_device_key(configured)
    }
}

fn device_entry<'a, T>(map: &'a BTreeMap<String, T>, device: &str) -> Option<&'a T> {
    if let Some(value) = map.get(device) {
        return Some(value);
    }
    let key = canonical_device_key(device);
    map.get(&key).or_else(|| {
        map.iter()
            .find_map(|(saved, value)| (canonical_device_key(saved) == key).then_some(value))
    })
}

fn remove_device_entry<T>(map: &mut BTreeMap<String, T>, device: &str) -> bool {
    let key = canonical_device_key(device);
    let before = map.len();
    map.retain(|saved, _| canonical_device_key(saved) != key);
    map.len() != before
}

pub fn default_eye_mapping(device: &str) -> EyeMapping {
    let key = canonical_device_key(device);
    let is_varjo = matches!(key.as_str(), "varjo" | "varjo_mjpeg");
    EyeMapping {
        swap_gaze_eyes: None,
        flip_gaze_x: !is_varjo,
        ..EyeMapping::default()
    }
}

fn normalize_eye_mapping(mut mapping: EyeMapping) -> EyeMapping {
    mapping.swap_eyes |= mapping.swap_gaze_eyes.unwrap_or(false);
    mapping.swap_gaze_eyes = None;
    mapping
}

pub fn default_ml_geometry(device: &str) -> [crate::core::types::MlGeometry; 2] {
    use crate::core::types::MlGeometry;
    if canonical_device_key(device) != "pimax_xr5" {
        return [MlGeometry::default(); 2];
    }
    [
        MlGeometry {
            crop_right: 0.40,
            crop_top: 0.15,
            crop_bottom: 0.15,
            scale_y: 1.20,
            rotate_deg: -30.0,
            ..MlGeometry::default()
        },
        MlGeometry {
            crop_left: 0.40,
            crop_top: 0.15,
            crop_bottom: 0.15,
            scale_y: 1.20,
            rotate_deg: 30.0,
            mirror_h: Some(true),
            ..MlGeometry::default()
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WideSource {
    #[default]
    Sranipal,
    Auto,
    Custom,
}

impl WideSource {
    pub const ALL: [Self; 3] = [Self::Sranipal, Self::Auto, Self::Custom];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sranipal => "sranipal",
            Self::Auto => "auto",
            Self::Custom => "custom",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum GazeSource {
    #[default]
    PerEye,
    Combined,
}

impl GazeSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PerEye => "per-eye",
            Self::Combined => "combined",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Vr4GazeSource {
    #[default]
    V1,
    V2,
}

impl Vr4GazeSource {
    pub fn label(self) -> &'static str {
        match self {
            Self::V1 => "v1 (40°)",
            Self::V2 => "v2 (30°)",
        }
    }

    pub fn stream_id(self) -> u32 {
        match self {
            Self::V1 => 1285,
            Self::V2 => 1289,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Hmd {
    pub device: String,
    pub varjo_left_url: String,
    pub varjo_right_url: String,
    pub wide_source: WideSource,
    pub starvr_direct: bool,
    pub vr4_gaze_source: Vr4GazeSource,
    pub vr4_gaze_smoothing: bool,

    #[serde(default, skip_serializing)]
    pub swap_eyes: bool,
    #[serde(default, skip_serializing)]
    pub flip_image: bool,
    #[serde(default, skip_serializing)]
    pub flip_gaze_x: bool,
    #[serde(default, skip_serializing)]
    pub ml_mirror_l: bool,
    #[serde(default, skip_serializing)]
    pub ml_mirror_r: bool,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mappings: BTreeMap<String, EyeMapping>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub geometry: BTreeMap<String, crate::core::types::MlGeometry>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub geometry_r: BTreeMap<String, crate::core::types::MlGeometry>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub despeckle: BTreeMap<String, crate::core::types::DespeckleParams>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub brightness: BTreeMap<String, crate::core::types::BrightnessNorm>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub flatten: BTreeMap<String, crate::core::types::FlattenParams>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub photometric_correction: BTreeMap<String, crate::core::types::PhotometricCorrection>,

    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub gaze_correction: BTreeMap<String, GazeCorrection>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gaze_eyelid_profiles: BTreeMap<String, crate::core::types::GazeEyelidProfile>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub wink_profiles: BTreeMap<String, crate::core::types::WinkProfile>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub blink_timing_profiles: BTreeMap<String, crate::core::types::BlinkTimingProfile>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub eyelid_response_profiles: BTreeMap<String, EyelidResponseProfile>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub right_eye_left_head: BTreeMap<String, bool>,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gaze_source: BTreeMap<String, GazeSource>,

    pub xr5_model_gaze: bool,
    pub xr5_gaze_hybrid: bool,
    pub xr5_gaze_package: PathBuf,
    pub xr5_model_gaze_correction: GazeCorrection,
    pub xr5_hybrid_gaze_correction: GazeCorrection,

    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dream_air_profiles: BTreeMap<String, DreamAirProfile>,
}

impl Default for Hmd {
    fn default() -> Self {
        Self {
            device: "auto".into(),
            varjo_left_url: "http://localhost:8080".into(),
            varjo_right_url: "http://localhost:8081".into(),
            wide_source: WideSource::Sranipal,
            starvr_direct: false,
            vr4_gaze_source: Vr4GazeSource::default(),
            vr4_gaze_smoothing: true,
            swap_eyes: false,
            flip_image: false,
            flip_gaze_x: false,
            ml_mirror_l: false,
            ml_mirror_r: false,
            mappings: BTreeMap::new(),
            geometry: BTreeMap::new(),
            geometry_r: BTreeMap::new(),
            despeckle: BTreeMap::new(),
            brightness: BTreeMap::new(),
            flatten: BTreeMap::new(),
            photometric_correction: BTreeMap::new(),
            gaze_correction: BTreeMap::new(),
            gaze_eyelid_profiles: BTreeMap::new(),
            wink_profiles: BTreeMap::new(),
            blink_timing_profiles: BTreeMap::new(),
            eyelid_response_profiles: BTreeMap::new(),
            right_eye_left_head: BTreeMap::new(),
            gaze_source: BTreeMap::new(),
            xr5_model_gaze: false,
            xr5_gaze_hybrid: false,
            xr5_gaze_package: PathBuf::new(),
            xr5_model_gaze_correction: GazeCorrection::default(),
            xr5_hybrid_gaze_correction: GazeCorrection::default(),
            dream_air_profiles: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Output {
    pub brokeneye: bool,
    pub brokeneye_port: u16,
    pub vrcft_filter_samples: u8,
    pub vrcft_sranipal_brow_link: bool,
    pub osc: bool,
    pub vrchat_eye_osc: bool,
    pub eyebrow_osc: bool,
    pub osc_host: String,
    pub osc_port: u16,
    pub raw_gaze_osc: bool,
    pub raw_gaze_osc_port: u16,
    pub eye_image_http: bool,
    pub eye_image_host: String,
    pub eye_image_port: u16,
}

impl Default for Output {
    fn default() -> Self {
        Self {
            brokeneye: true,
            brokeneye_port: 5555,
            vrcft_filter_samples: 10,
            vrcft_sranipal_brow_link: false,
            osc: false,
            vrchat_eye_osc: cfg!(target_os = "linux"),
            eyebrow_osc: false,
            osc_host: "127.0.0.1".into(),
            osc_port: 9000,
            raw_gaze_osc: false,
            raw_gaze_osc_port: 9005,
            eye_image_http: false,
            eye_image_host: "127.0.0.1".into(),
            eye_image_port: 5556,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Ui {
    pub inference_enabled: bool,
    pub steamvr_overlay: bool,
    pub eye_camera_preview: bool,
    pub recording_audio_cues: bool,
    pub wearing_memory_enabled: bool,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub wearing_memory_devices: BTreeMap<String, bool>,
    pub eye_wide_enabled: bool,
    pub eyebrow_enabled: bool,
    pub eyelid_inference_backend: EyelidInferenceBackend,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub eyelid_response_links: BTreeMap<String, bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EyelidInferenceBackend {
    Auto,
    Gpu,
    Cpu,
}

impl Default for EyelidInferenceBackend {
    fn default() -> Self {
        Self::Gpu
    }
}

impl EyelidInferenceBackend {
    pub const fn resolved(self) -> Self {
        match self { Self::Auto => Self::Gpu, other => other }
    }
    pub const fn label(self) -> &'static str {
        match self {
            Self::Auto => "GPU",
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }
}

impl Default for Ui {
    fn default() -> Self {
        Self {
            inference_enabled: true,
            steamvr_overlay: true,
            eye_camera_preview: false,
            recording_audio_cues: true,
            wearing_memory_enabled: true,
            wearing_memory_devices: BTreeMap::new(),
            eye_wide_enabled: true,
            eyebrow_enabled: true,
            eyelid_inference_backend: EyelidInferenceBackend::default(),
            eyelid_response_links: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct AssetStatus {
    pub label: &'static str,
    pub path: Option<PathBuf>,
    pub present: bool,
    pub required: bool,
    pub gates: String,
}

impl Config {
    pub fn wearing_memory_enabled_for(&self, device: &str) -> bool {
        device_entry(&self.ui.wearing_memory_devices, device).copied()
            .unwrap_or(self.ui.wearing_memory_enabled)
    }

    pub fn set_wearing_memory_enabled_for(&mut self, device: &str, enabled: bool) {
        self.ui.wearing_memory_devices.insert(canonical_device_key(device), enabled);
    }

    pub fn eyelid_response_link_for(&self, device: &str) -> Option<bool> {
        device_entry(&self.ui.eyelid_response_links, device).copied()
    }

    pub fn set_eyelid_response_link(&mut self, device: &str, linked: bool) {
        let key = canonical_device_key(device);
        remove_device_entry(&mut self.ui.eyelid_response_links, &key);
        self.ui.eyelid_response_links.insert(key, linked);
    }

    pub fn load(path: &Path) -> (Config, Option<String>) {
        let (mut cfg, warning) = match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str::<Config>(&text) {
                Ok(mut cfg) => {
                    cfg.migrate_legacy_mapping();
                    (cfg, None)
                }
                Err(error) => match quarantine_invalid_config(path) {
                    Ok(backup) => {
                        let warning = format!(
                            "Invalid configuration was preserved as {}; using defaults until new settings are saved ({error})",
                            backup.display()
                        );
                        remember_primary_config_warning(path, &warning);
                        (Config::default(), Some(warning))
                    }
                    Err(backup_error) => {
                        let reason = format!(
                            "could not preserve unreadable {}: {backup_error}",
                            path.display()
                        );
                        let warning = format!(
                            "Invalid configuration was not overwritten; automatic saves are disabled ({reason}; parse error: {error})"
                        );
                        remember_primary_config_warning(path, &warning);
                        let cfg = Config {
                            save_blocked_reason: Some(reason),
                            ..Config::default()
                        };
                        (cfg, Some(warning))
                    }
                },
            },
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                (Config::default(), take_primary_config_warning(path))
            }
            Err(error) => {
                let reason = format!("could not read {}: {error}", path.display());
                let warning = format!(
                    "Configuration was not overwritten; automatic saves are disabled ({reason})"
                );
                remember_primary_config_warning(path, &warning);
                let cfg = Config {
                    save_blocked_reason: Some(reason),
                    ..Config::default()
                };
                (cfg, Some(warning))
            }
        };
        cfg.enforce_build_variant();
        (cfg, warning)
    }

    fn enforce_build_variant(&mut self) {
        if PSVR2_ONLY_BUILD {
            self.hmd.device = "psvr2".to_string();
            self.hmd.wide_source = WideSource::Sranipal;
        } else if XR5_ONLY_BUILD {
            self.hmd.device = "pimax_xr5".to_string();
        }
    }

    pub fn mapping_for(&self, device: &str) -> EyeMapping {
        let preset = default_eye_mapping(device);
        let mapping = device_entry(&self.hmd.mappings, device)
            .copied()
            .unwrap_or(preset);
        normalize_eye_mapping(mapping)
    }

    pub fn set_mapping(&mut self, device: &str, m: EyeMapping) {
        self.hmd
            .mappings
            .insert(canonical_device_key(device), normalize_eye_mapping(m));
    }

    pub fn wide_source_for(&self, device: &str) -> WideSource {
        if canonical_device_key(device) == "pimax_xr5" {
            self.hmd.wide_source
        } else {
            WideSource::Sranipal
        }
    }

    pub fn geometry_for(&self, device: &str) -> [crate::core::types::MlGeometry; 2] {
        let preset = default_ml_geometry(device);
        let saved_l = device_entry(&self.hmd.geometry, device).copied();
        let mut l = saved_l.unwrap_or(preset[0]);
        let mut r = device_entry(&self.hmd.geometry_r, device)
            .copied()
            .unwrap_or_else(|| if saved_l.is_some() { l } else { preset[1] });
        if l.mirror_h.is_none() {
            l.mirror_h = preset[0].mirror_h;
        }
        if r.mirror_h.is_none() {
            r.mirror_h = preset[1].mirror_h;
        }
        [l, r]
    }

    pub fn set_geometry(&mut self, device: &str, g: [crate::core::types::MlGeometry; 2]) {
        let key = canonical_device_key(device);
        self.hmd.geometry.insert(key.clone(), g[0]);
        self.hmd.geometry_r.insert(key, g[1]);
    }

    pub fn has_geometry_override(&self, device: &str) -> bool {
        device_entry(&self.hmd.geometry, device).is_some()
            || device_entry(&self.hmd.geometry_r, device).is_some()
    }

    pub fn clear_geometry(&mut self, device: &str) -> bool {
        remove_device_entry(&mut self.hmd.geometry, device)
            | remove_device_entry(&mut self.hmd.geometry_r, device)
    }

    pub fn despeckle_for(&self, device: &str) -> crate::core::types::DespeckleParams {
        device_entry(&self.hmd.despeckle, device)
            .copied()
            .unwrap_or_default()
    }

    pub fn set_despeckle(&mut self, device: &str, d: crate::core::types::DespeckleParams) {
        self.hmd.despeckle.insert(canonical_device_key(device), d);
    }

    pub fn brightness_for(&self, device: &str) -> crate::core::types::BrightnessNorm {
        let mut brightness = device_entry(&self.hmd.brightness, device)
            .copied()
            .unwrap_or_default();
        brightness.enabled = false;
        brightness.auto_learn = false;
        brightness.captured = false;
        brightness
    }

    pub fn set_brightness(&mut self, device: &str, mut b: crate::core::types::BrightnessNorm) {
        b.enabled = false;
        b.auto_learn = false;
        b.captured = false;
        self.hmd.brightness.insert(canonical_device_key(device), b);
    }

    pub fn flatten_for(&self, device: &str) -> crate::core::types::FlattenParams {
        device_entry(&self.hmd.flatten, device)
            .copied()
            .unwrap_or_default()
    }

    pub fn set_flatten(&mut self, device: &str, f: crate::core::types::FlattenParams) {
        self.hmd.flatten.insert(canonical_device_key(device), f);
    }

    pub fn photometric_correction_for(
        &self,
        device: &str,
    ) -> crate::core::types::PhotometricCorrection {
        device_entry(&self.hmd.photometric_correction, device)
            .copied()
            .unwrap_or_default()
    }

    pub fn set_photometric_correction(
        &mut self,
        device: &str,
        correction: crate::core::types::PhotometricCorrection,
    ) {
        self.hmd
            .photometric_correction
            .insert(canonical_device_key(device), correction);
    }

    pub fn has_photometric_correction(&self, device: &str) -> bool {
        device_entry(&self.hmd.photometric_correction, device).is_some()
    }

    pub fn clear_photometric_correction(&mut self, device: &str) -> bool {
        remove_device_entry(&mut self.hmd.photometric_correction, device)
    }

    pub fn gaze_correction_for(&self, device: &str) -> GazeCorrection {
        device_entry(&self.hmd.gaze_correction, device)
            .copied()
            .unwrap_or_default()
    }

    pub fn set_gaze_correction(&mut self, device: &str, correction: GazeCorrection) {
        self.hmd
            .gaze_correction
            .insert(canonical_device_key(device), correction);
    }

    pub fn gaze_eyelid_profile_for(&self, device: &str) -> crate::core::types::GazeEyelidProfile {
        device_entry(&self.hmd.gaze_eyelid_profiles, device)
            .copied()
            .filter(crate::core::types::GazeEyelidProfile::is_compatible)
            .unwrap_or_default()
    }

    pub fn set_gaze_eyelid_profile(
        &mut self,
        device: &str,
        profile: crate::core::types::GazeEyelidProfile,
    ) {
        self.hmd
            .gaze_eyelid_profiles
            .insert(canonical_device_key(device), profile);
    }

    pub fn clear_gaze_eyelid_profile(&mut self, device: &str) -> bool {
        remove_device_entry(&mut self.hmd.gaze_eyelid_profiles, device)
    }

    pub fn wink_profile_for(&self, device: &str) -> crate::core::types::WinkProfile {
        device_entry(&self.hmd.wink_profiles, device)
            .copied()
            .filter(crate::core::types::WinkProfile::is_compatible)
            .unwrap_or_default()
    }

    pub fn set_wink_profile(&mut self, device: &str, profile: crate::core::types::WinkProfile) {
        self.hmd
            .wink_profiles
            .insert(canonical_device_key(device), profile);
    }

    pub fn clear_wink_profile(&mut self, device: &str) -> bool {
        remove_device_entry(&mut self.hmd.wink_profiles, device)
    }

    pub fn blink_timing_profile_for(&self, device: &str) -> crate::core::types::BlinkTimingProfile {
        device_entry(&self.hmd.blink_timing_profiles, device)
            .copied()
            .filter(crate::core::types::BlinkTimingProfile::is_compatible)
            .unwrap_or_default()
    }

    pub fn set_blink_timing_profile(
        &mut self,
        device: &str,
        profile: crate::core::types::BlinkTimingProfile,
    ) {
        self.hmd
            .blink_timing_profiles
            .insert(canonical_device_key(device), profile);
    }

    pub fn clear_blink_timing_calibration(
        &mut self,
        device: &str,
    ) -> crate::core::types::BlinkTimingProfile {
        let enabled = self.blink_timing_profile_for(device).enabled;
        remove_device_entry(&mut self.hmd.blink_timing_profiles, device);
        let mut profile = crate::core::types::BlinkTimingProfile::default();
        profile.enabled = enabled;
        if !enabled {
            self.hmd
                .blink_timing_profiles
                .insert(canonical_device_key(device), profile);
        }
        profile
    }

    pub fn eyelid_response_profile_for(&self, device: &str) -> EyelidResponseProfile {
        device_entry(&self.hmd.eyelid_response_profiles, device)
            .copied()
            .map(EyelidResponseProfile::sanitized)
            .unwrap_or_default()
    }

    pub fn has_eyelid_response_profile(&self, device: &str) -> bool {
        device_entry(&self.hmd.eyelid_response_profiles, device).is_some()
    }

    pub fn set_eyelid_response_profile(&mut self, device: &str, profile: EyelidResponseProfile) {
        let key = canonical_device_key(device);
        remove_device_entry(&mut self.hmd.eyelid_response_profiles, &key);
        self.hmd
            .eyelid_response_profiles
            .insert(key, profile.sanitized());
    }

    pub fn clear_eyelid_response_profile(&mut self, device: &str) -> bool {
        remove_device_entry(&mut self.hmd.eyelid_response_profiles, device)
    }

    pub fn reset_eyelid_calibration_for(&mut self, device: &str) {
        self.set_eyelid_response_profile(device, EyelidResponseProfile::default());
        self.clear_gaze_eyelid_profile(device);
        self.clear_wink_profile(device);
        remove_device_entry(&mut self.hmd.blink_timing_profiles, device);
    }

    pub fn right_eye_left_head_for(&self, device: &str) -> bool {
        device_entry(&self.hmd.right_eye_left_head, device)
            .copied()
            .unwrap_or(true)
    }

    pub fn set_right_eye_left_head(&mut self, device: &str, enabled: bool) {
        let key = canonical_device_key(device);
        remove_device_entry(&mut self.hmd.right_eye_left_head, &key);
        self.hmd.right_eye_left_head.insert(key, enabled);
    }

    pub fn gaze_source_for(&self, device: &str) -> GazeSource {
        device_entry(&self.hmd.gaze_source, device)
            .copied()
            .unwrap_or_default()
    }

    pub fn set_gaze_source(&mut self, device: &str, source: GazeSource) {
        self.hmd
            .gaze_source
            .insert(canonical_device_key(device), source);
    }

    pub fn dream_air_profile_for(&self, device: &str) -> Option<&DreamAirProfile> {
        device_entry(&self.hmd.dream_air_profiles, device)
            .filter(|profile| profile.schema_version == 1)
    }

    pub fn set_dream_air_profile(&mut self, device: &str, profile: DreamAirProfile) {
        self.hmd
            .dream_air_profiles
            .insert(canonical_device_key(device), profile);
    }

    pub fn migrate_auto_device_settings(&mut self, resolved_device: &str) -> bool {
        let target = canonical_device_key(resolved_device);
        if target == "auto" {
            return false;
        }

        fn move_bucket<T>(map: &mut BTreeMap<String, T>, target: &str) -> bool {
            let stale: Vec<String> = map
                .keys()
                .filter(|key| canonical_device_key(key) == "auto")
                .cloned()
                .collect();
            let mut target_exists = map.keys().any(|key| canonical_device_key(key) == target);
            let mut changed = false;
            for key in stale {
                if let Some(value) = map.remove(&key) {
                    if !target_exists {
                        map.insert(target.to_string(), value);
                        target_exists = true;
                    }
                    changed = true;
                }
            }
            changed
        }

        let mut changed = false;
        changed |= move_bucket(&mut self.hmd.mappings, &target);
        changed |= move_bucket(&mut self.hmd.geometry, &target);
        changed |= move_bucket(&mut self.hmd.geometry_r, &target);
        changed |= move_bucket(&mut self.hmd.despeckle, &target);
        changed |= move_bucket(&mut self.hmd.brightness, &target);
        changed |= move_bucket(&mut self.hmd.flatten, &target);
        changed |= move_bucket(&mut self.hmd.photometric_correction, &target);
        changed |= move_bucket(&mut self.hmd.gaze_correction, &target);
        changed |= move_bucket(&mut self.hmd.gaze_eyelid_profiles, &target);
        changed |= move_bucket(&mut self.hmd.wink_profiles, &target);
        changed |= move_bucket(&mut self.hmd.blink_timing_profiles, &target);
        changed |= move_bucket(&mut self.hmd.eyelid_response_profiles, &target);
        changed |= move_bucket(&mut self.hmd.right_eye_left_head, &target);
        changed |= move_bucket(&mut self.hmd.gaze_source, &target);
        changed |= move_bucket(&mut self.hmd.dream_air_profiles, &target);
        changed |= move_bucket(&mut self.ui.eyelid_response_links, &target);
        changed
    }

    fn migrate_legacy_mapping(&mut self) {
        if !self.hmd.mappings.is_empty() {
            return;
        }
        let legacy = EyeMapping {
            swap_eyes: self.hmd.swap_eyes,
            flip_image: self.hmd.flip_image,
            swap_gaze_eyes: None,
            flip_gaze_x: self.hmd.flip_gaze_x,
            ml_mirror_l: self.hmd.ml_mirror_l,
            ml_mirror_r: self.hmd.ml_mirror_r,
        };
        if legacy != EyeMapping::default() {
            let dev = canonical_device_key(&self.hmd.device);
            self.hmd.mappings.insert(dev, legacy);
        }
    }

    pub fn ml_params_path(&self) -> Option<PathBuf> {
        let nonempty = |o: &Option<String>| {
            o.as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        if let Some(m) = nonempty(&self.assets.ml_model) {
            return Some(PathBuf::from(m));
        }
        nonempty(&self.assets.sranipal_dir).map(|d| model_path_in(Path::new(&d)))
    }

    pub fn tobii_runtime_path(&self) -> Option<PathBuf> {
        if PSVR2_ONLY_BUILD {
            None
        } else {
            crate::bundled_tobii::path()
        }
    }

    pub fn starvr_runtime_path(&self) -> Option<PathBuf> {
        if PSVR2_ONLY_BUILD {
            None
        } else {
            crate::bundled_tobii::starvr_path().or_else(|| crate::bundled_tobii::path())
        }
    }

    pub fn varjo_lib_path(&self) -> Option<PathBuf> {
        let nonempty = |o: &Option<String>| {
            o.as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        if let Some(p) = nonempty(&self.assets.varjo_lib) {
            return Some(PathBuf::from(p));
        }
        varjo_lib_candidates().into_iter().find(|p| p.is_file())
    }

    pub fn brow_model_path(&self) -> Option<PathBuf> {
        let explicit = self
            .assets
            .brow_model
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        explicit.or_else(bundled_brow_model_path)
    }

    pub fn wide_model_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.wide_model).map(PathBuf::from)
    }

    pub fn python_exe_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.python_exe).map(PathBuf::from)
    }

    pub fn vr_eyebrow_dir_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.vr_eyebrow_dir).map(PathBuf::from)
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(reason) = &self.save_blocked_reason {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("refusing to overwrite protected configuration: {reason}"),
            ));
        }
        let mut normalized = self.clone();
        normalized.enforce_build_variant();
        for mapping in normalized.hmd.mappings.values_mut() {
            *mapping = normalize_eye_mapping(*mapping);
        }
        for profile in normalized.hmd.eyelid_response_profiles.values_mut() {
            if profile.schema_version == EyelidResponseProfile::SCHEMA_VERSION {
                *profile = profile.sanitized();
            }
        }
        let text = toml::to_string_pretty(&normalized)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        write_atomic(path, text.as_bytes())
    }

    pub fn check_assets(&self) -> Vec<AssetStatus> {
        let exists = |p: &Option<PathBuf>| p.as_ref().map(|p| p.is_file()).unwrap_or(false);
        #[cfg(all(target_os = "linux", feature = "xr5-only", feature = "external-eyelid-model"))]
        if matches!(canonical_device_key(&self.hmd.device).as_str(), "auto" | "pimax_xr5") {
            if !self.ui.inference_enabled { return Vec::new(); }
            if self.xr5_research.enabled && self.xr5_research.avatar {
                let model = Some(crate::ml::external_eyelid::resolve(self.xr5_research.model_file()));
                let runtime = crate::ml::external_eyelid::runtime_library_path().ok();
                return vec![
                    AssetStatus { label: "XR5 eyelid model", present: exists(&model), path: model, required: true, gates: "Restore the models folder from the XR5 Linux package.".into() },
                    AssetStatus { label: "XR5 model runtime", present: exists(&runtime), path: runtime, required: true, gates: "Restore the model-runtime folder from the XR5 Linux package.".into() },
                ];
            }
        }

        let ml = self.ml_params_path();

        vec![AssetStatus {
            label: "SRanipal ML weights (common)",
            present: exists(&ml),
            path: ml,
            required: true,
            gates: "eyelid openness/wide/squeeze (core). Set [assets].ml_model (direct \
                        weights file) or [assets].sranipal_dir."
                .into(),
        }]
    }

    pub fn missing_required(&self) -> Vec<AssetStatus> {
        self.check_assets()
            .into_iter()
            .filter(|a| a.required && !a.present)
            .collect()
    }

    pub fn write_template_if_absent(path: &Path) -> std::io::Result<bool> {
        if path.exists() {
            return Ok(false);
        }
        if !PSVR2_ONLY_BUILD && !XR5_ONLY_BUILD {
            let psvr2_path = path.with_file_name("sranibro-psvr2.toml");
            if psvr2_path.is_file() {
                if let Ok(text) = std::fs::read_to_string(&psvr2_path) {
                    if let Ok(mut migrated) = toml::from_str::<Config>(&text) {
                        migrated.hmd.device = "psvr2".to_string();
                        migrated.hmd.wide_source = WideSource::Sranipal;
                        migrated.save_blocked_reason = None;
                        let encoded = toml::to_string_pretty(&migrated)
                            .map_err(|error| std::io::Error::other(error.to_string()))?;
                        write_atomic(path, encoded.as_bytes())?;
                        return Ok(true);
                    }
                }
            }
        }
        if PSVR2_ONLY_BUILD || XR5_ONLY_BUILD {
            let device = if PSVR2_ONLY_BUILD {
                "psvr2"
            } else {
                "pimax_xr5"
            };
            let template = TEMPLATE.replacen(
                "device = \u{22}auto\u{22}",
                &format!("device = \u{22}{device}\u{22}"),
                1,
            );
            std::fs::write(path, template)?;
        } else {
            std::fs::write(path, TEMPLATE)?;
        }
        Ok(true)
    }
}

static PENDING_CONFIG_WARNING: std::sync::OnceLock<std::sync::Mutex<Option<String>>> =
    std::sync::OnceLock::new();

fn is_primary_config(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case(config_file_name()))
}

fn remember_primary_config_warning(path: &Path, warning: &str) {
    if !is_primary_config(path) {
        return;
    }
    let slot = PENDING_CONFIG_WARNING.get_or_init(|| std::sync::Mutex::new(None));
    *slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(warning.to_owned());
}

fn take_primary_config_warning(path: &Path) -> Option<String> {
    if !is_primary_config(path) {
        return None;
    }
    PENDING_CONFIG_WARNING
        .get()
        .and_then(|slot| slot.lock().ok()?.take())
}

fn quarantine_invalid_config(path: &Path) -> std::io::Result<PathBuf> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("sranibro");
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("toml");
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let backup = parent.join(format!(
        "{stem}.invalid-{}-{nonce}.{extension}",
        std::process::id()
    ));
    std::fs::rename(path, &backup)?;
    Ok(backup)
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    match std::fs::rename(&tmp, path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn sanitized_label(label: &str) -> String {
    let value: String = label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect();
    value.trim_matches('-').to_string()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn is_state_file_name(name: &str) -> bool {
    name == "sranibro.toml"
        || name == "sranibro-psvr2.toml"
        || name == "sranibro-xr5.toml"
        || name == "sranibro_calib.toml"
        || (name.starts_with("sranibro_calib_") && name.ends_with(".toml"))
}

fn state_files(dir: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if let Some(name) = entry.file_name().to_str().map(str::to_owned) {
            let path = entry.path();
            if path.is_file() && is_state_file_name(&name) {
                files.push((name, path));
            }
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn reference_files(dir: &Path) -> std::io::Result<Vec<(String, PathBuf)>> {
    let dir = dir.join("reseat-references");
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if path.is_file() && name.ends_with(".bin") {
            files.push((name, path));
        }
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

fn create_state_backup_at(base: &Path, label: &str) -> std::io::Result<PathBuf> {
    let root = base.join("backups");
    std::fs::create_dir_all(&root)?;
    let label = sanitized_label(label);
    let stem = if label.is_empty() {
        unix_now().to_string()
    } else {
        format!("{}-{label}", unix_now())
    };
    let mut dir = root.join(&stem);
    let mut suffix = 2;
    while dir.exists() {
        dir = root.join(format!("{stem}-{suffix}"));
        suffix += 1;
    }
    std::fs::create_dir_all(&dir)?;
    for (name, source) in state_files(base)? {
        std::fs::copy(source, dir.join(name))?;
    }
    let references = reference_files(base)?;
    if !references.is_empty() {
        let destination = dir.join("reseat-references");
        std::fs::create_dir_all(&destination)?;
        for (name, source) in references {
            std::fs::copy(source, destination.join(name))?;
        }
    }
    Ok(dir)
}

pub fn create_state_backup(label: &str) -> std::io::Result<PathBuf> {
    create_state_backup_at(&base_dir(), label)
}

pub fn latest_state_backup() -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(base_dir().join("backups"))
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    entries.last().map(|entry| entry.path())
}

pub fn restore_state_backup(dir: &Path) -> std::io::Result<()> {
    let base = base_dir();
    for (name, source) in state_files(dir)? {
        let bytes = std::fs::read(source)?;
        write_atomic(&base.join(name), &bytes)?;
    }
    for (name, source) in reference_files(dir)? {
        let bytes = std::fs::read(source)?;
        write_atomic(&base.join("reseat-references").join(name), &bytes)?;
    }
    Ok(())
}

#[cfg(all(target_os = "linux", feature = "xr5-only"))]
pub const TEMPLATE: &str = r#"[hmd]
device = "auto"

[xr5_research]
enabled = true
avatar = true
model_file = "models/XR5-Eyelid-Balanced-20260928.sranimodel"
gpu = false

[ui]
eyelid_inference_backend = "cpu"
eye_camera_preview = false

[output]
vrchat_eye_osc = true
osc_host = "127.0.0.1"
osc_port = 9000
brokeneye = true
brokeneye_port = 5555
osc = false
eyebrow_osc = false
eye_image_http = false
"#;

#[cfg(all(target_os = "linux", not(feature = "xr5-only")))]
pub const TEMPLATE: &str = r#"[assets]

[hmd]
device = "auto"

[output]
vrchat_eye_osc = true
osc_host = "127.0.0.1"
osc_port = 9000
brokeneye = true
brokeneye_port = 5555
osc = false
eyebrow_osc = false
eye_image_http = false
eye_image_host = "127.0.0.1"
eye_image_port = 5556

[ui]
eye_camera_preview = false
"#;

#[cfg(not(target_os = "linux"))]
pub const TEMPLATE: &str = r#"
[assets]




[hmd]
device = "auto"
wide_source = "sranipal"



[output]
brokeneye = true
brokeneye_port = 5555
vrcft_filter_samples = 10
vrcft_sranipal_brow_link = false
osc = false
eyebrow_osc = false
osc_host = "127.0.0.1"
osc_port = 9000
eye_image_http = false
eye_image_host = "127.0.0.1"
eye_image_port = 5556

[ui]
steamvr_overlay = true
eye_camera_preview = false
recording_audio_cues = true
wearing_memory_enabled = true
eye_wide_enabled = true
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wearing_memory_switch_is_per_headset_and_camera_only_round_trips() {
        let mut c=Config::default();
        assert!(c.ui.inference_enabled);
        c.set_wearing_memory_enabled_for("xr5",false);
        assert!(!c.wearing_memory_enabled_for("pimax_xr5"));
        assert!(c.wearing_memory_enabled_for("pimax_vr4"));
        assert!(c.wearing_memory_enabled_for("psvr2"));
        c.ui.inference_enabled=false;
        let restored:Config=toml::from_str(&toml::to_string(&c).unwrap()).unwrap();
        assert!(!restored.ui.inference_enabled);
        assert!(!restored.wearing_memory_enabled_for("xr5"));
        assert!(restored.wearing_memory_enabled_for("varjo"));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_config_paths_reject_relative_xdg_and_keep_home_fallback() {
        assert_eq!(
            linux_config_candidates(Some("/data/config".into()), Some("/home/test".into())),
            vec![
                PathBuf::from("/data/config"),
                PathBuf::from("/home/test/.config")
            ]
        );
        assert_eq!(
            linux_config_candidates(Some("relative".into()), Some("/home/test".into())),
            vec![PathBuf::from("/home/test/.config")]
        );
        assert!(linux_config_candidates(Some("".into()), None).is_empty());
    }

    #[test]
    fn native_eye_osc_defaults_are_platform_specific_and_configurable() {
        assert_eq!(Output::default().vrchat_eye_osc, cfg!(target_os = "linux"));
        let disabled: Config = toml::from_str("[output]\nvrchat_eye_osc=false").unwrap();
        assert!(!disabled.output.vrchat_eye_osc);
        let enabled: Config = toml::from_str("[output]\nvrchat_eye_osc=true").unwrap();
        assert!(enabled.output.vrchat_eye_osc);
        assert!(!enabled.output.osc);
    }

    #[test]
    fn raw_gaze_osc_is_opt_in_and_has_an_independent_destination() {
        let old: Config = toml::from_str("[output]\nosc_port=9017").unwrap();
        assert!(!old.output.raw_gaze_osc);
        assert_eq!(old.output.raw_gaze_osc_port, 9005);
        let configured: Config = toml::from_str(
            "[output]\nraw_gaze_osc=true\nraw_gaze_osc_port=9012\nosc_port=9017",
        ).unwrap();
        let roundtrip: Config = toml::from_str(&toml::to_string(&configured).unwrap()).unwrap();
        assert!(roundtrip.output.raw_gaze_osc);
        assert_eq!(roundtrip.output.raw_gaze_osc_port, 9012);
        assert_eq!(roundtrip.output.osc_port, 9017);
        assert!(!roundtrip.output.osc);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn copied_windows_model_is_found_with_original_capitalization() {
        let root = std::env::temp_dir().join(format!("sranibro-model-case-{}", std::process::id()));
        let model = root.join("MODEL/EYEPREDICTION/00-0000.params_opencl.params");
        std::fs::create_dir_all(model.parent().unwrap()).unwrap();
        std::fs::write(&model, b"test").unwrap();
        assert_eq!(model_path_in(&root), model);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn packaged_eyebrow_model_is_discovered_with_stable_priority() {
        let root = std::env::temp_dir().join(format!(
            "sranibro_packaged_brow_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("models")).unwrap();

        let nested_legacy = root.join("models").join("brow.bin");
        std::fs::write(&nested_legacy, b"legacy").unwrap();
        assert_eq!(bundled_brow_model_path_in(&root), Some(nested_legacy));

        let packaged = root.join("eyebrow.bin");
        std::fs::write(&packaged, b"packaged").unwrap();
        assert_eq!(bundled_brow_model_path_in(&root), Some(packaged));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn explicit_eyebrow_model_override_wins_without_requiring_discovery() {
        let mut config = Config::default();
        let explicit = PathBuf::from(r"D:\models\personal-eyebrow.bin");
        config.assets.brow_model = Some(explicit.to_string_lossy().into_owned());
        assert_eq!(config.brow_model_path(), Some(explicit));
    }

    #[test]
    fn defaults_are_sane() {
        let c = Config::default();
        assert_eq!(c.hmd.device, "auto");
        assert_eq!(c.hmd.wide_source, WideSource::Sranipal);
        assert!(!c.ui.eye_camera_preview);
        assert!(c.ui.eye_wide_enabled);
        assert!(!c.output.vrcft_sranipal_brow_link);
        assert!(c.output.brokeneye);
        assert_eq!(c.output.brokeneye_port, 5555);
        assert!(!c.output.eyebrow_osc);
        assert!(!c.output.eye_image_http);
        assert_eq!(c.output.eye_image_host, "127.0.0.1");
        assert_eq!(c.output.eye_image_port, 5556);
        assert!(
            c.ml_params_path().is_none(),
            "no sranipal_dir -> no ML path"
        );
    }

    #[test]
    fn legacy_adaptive_brightness_is_always_resolved_as_manual_only() {
        let mut c = Config::default();
        c.hmd.brightness.insert(
            "pimax_vr4".into(),
            crate::core::types::BrightnessNorm {
                enabled: true,
                manual_gain: 1.5,
                auto_learn: true,
                captured: true,
                ..Default::default()
            },
        );

        let resolved = c.brightness_for("pimax_vr4");
        assert_eq!(resolved.manual_gain, 1.5);
        assert!(!resolved.enabled);
        assert!(!resolved.auto_learn);
        assert!(!resolved.captured);

        c.set_brightness("pimax_vr4", resolved);
        let saved = c.hmd.brightness.get("pimax_vr4").unwrap();
        assert!(!saved.enabled && !saved.auto_learn && !saved.captured);
    }

    #[test]
    fn xr5_wide_source_never_leaks_to_other_hmds() {
        let mut c = Config::default();
        c.hmd.wide_source = WideSource::Custom;

        for xr5 in ["pimax_xr5", "pimax-xr5", "xr5", "dream_air"] {
            assert_eq!(c.wide_source_for(xr5), WideSource::Custom, "{xr5}");
        }
        for other in ["auto", "pimax_vr4", "starvr", "varjo", "varjo_mjpeg", "vpe"] {
            assert_eq!(
                c.wide_source_for(other),
                WideSource::Sranipal,
                "{other} must retain SRanipal Wide"
            );
        }
    }

    #[test]
    fn calibration_paths_are_canonical_and_per_device() {
        assert_eq!(calib_path_for("dream_air"), calib_path_for("pimax_xr5"));
        assert_ne!(calib_path_for("pimax_xr5"), calib_path_for("pimax_vr4"));
        assert_ne!(calib_path_for("starvr"), calib_path_for("varjo"));
        assert_ne!(calib_path_for("pimax_xr5"), calib_path());
        assert_eq!(
            calib_path_for("pimax-xr5")
                .file_name()
                .and_then(|s| s.to_str()),
            Some("sranibro_calib_pimax_xr5.toml")
        );
    }

    #[test]
    fn parses_partial_toml_and_fills_defaults() {
        let text = r#"
            [assets]
            sranipal_dir = "X:\\SRanipal"
            [output]
            osc = true
        "#;
        let c: Config = toml::from_str(text).unwrap();
        assert!(c.output.osc, "explicit osc=true honored");
        assert!(
            !c.ui.eye_camera_preview,
            "older configs default the GPU-heavy dashboard preview off"
        );
        assert!(
            c.ui.eye_wide_enabled,
            "older configs without the field keep EyeWide enabled"
        );
        assert!(
            !c.output.eyebrow_osc,
            "older configs keep eyebrow-only OSC disabled"
        );
        assert!(
            c.output.brokeneye,
            "unspecified brokeneye keeps default true"
        );
        let ml = c.ml_params_path().unwrap();
        assert!(ml.ends_with("00-0000.params_opencl.params"));
        assert!(ml.to_string_lossy().contains("EyePrediction"));
    }

    #[test]
    fn eye_wide_master_false_round_trips_in_ui_config() {
        let mut original = Config::default();
        original.ui.eye_wide_enabled = false;

        let encoded = toml::to_string(&original).expect("config serializes");
        let decoded: Config = toml::from_str(&encoded).expect("config parses");

        assert!(!decoded.ui.eye_wide_enabled);
        assert!(encoded.contains("eye_wide_enabled = false"));
    }

    #[test]
    fn eye_camera_preview_true_round_trips_in_ui_config() {
        let mut original = Config::default();
        original.ui.eye_camera_preview = true;

        let encoded = toml::to_string(&original).expect("config serializes");
        let decoded: Config = toml::from_str(&encoded).expect("config parses");

        assert!(decoded.ui.eye_camera_preview);
        assert!(encoded.contains("eye_camera_preview = true"));
    }

    #[test]
    fn wearing_memory_master_defaults_on_and_round_trips_off() {
        let old: Config = toml::from_str("[ui]\neye_camera_preview = false\n")
            .expect("pre-wearing-memory configuration parses");
        assert!(old.ui.wearing_memory_enabled);

        let mut configured = Config::default();
        configured.ui.wearing_memory_enabled = false;
        let encoded = toml::to_string(&configured).expect("config serializes");
        let decoded: Config = toml::from_str(&encoded).expect("config parses");
        assert!(!decoded.ui.wearing_memory_enabled);
        assert!(encoded.contains("wearing_memory_enabled = false"));
    }

    #[test]
    fn eyelid_inference_backend_defaults_to_gpu_and_preserves_explicit_cpu() {
        let old: Config = toml::from_str("[ui]\neye_camera_preview = false\n")
            .expect("pre-GPU configuration parses");
        assert_eq!(
            old.ui.eyelid_inference_backend,
            EyelidInferenceBackend::Gpu
        );

        let mut configured = Config::default();
        configured.ui.eyelid_inference_backend = EyelidInferenceBackend::Gpu;
        let encoded = toml::to_string(&configured).expect("config serializes");
        let decoded: Config = toml::from_str(&encoded).expect("config parses");
        assert_eq!(
            decoded.ui.eyelid_inference_backend,
            EyelidInferenceBackend::Gpu
        );
        assert!(encoded.contains("eyelid_inference_backend = \"gpu\""));
        let legacy: Config = toml::from_str("[ui]\neyelid_inference_backend = 'auto'").unwrap();
        assert_eq!(legacy.ui.eyelid_inference_backend.resolved(), EyelidInferenceBackend::Gpu);
        let cpu: Config = toml::from_str("[ui]\neyelid_inference_backend = 'cpu'").unwrap();
        assert_eq!(cpu.ui.eyelid_inference_backend.resolved(), EyelidInferenceBackend::Cpu);
    }

    #[test]
    fn missing_assets_reported_not_panicked() {
        let mut c = Config::default();
        #[cfg(all(target_os = "linux", feature = "xr5-only", feature = "external-eyelid-model"))]
        { c.xr5_research.enabled = false; c.xr5_research.avatar = false; }
        let missing = c.missing_required();
        assert_eq!(missing.len(), 1, "only the eye model is user-configurable");
        let labels: Vec<&str> = missing.iter().map(|a| a.label).collect();
        assert!(
            labels.iter().any(|l| l.contains("ML weights")),
            "ML required: {labels:?}"
        );
        assert!(missing.iter().all(|a| !a.present));
    }

    #[cfg(all(target_os = "linux", feature = "xr5-only", feature = "external-eyelid-model"))]
    #[test]
    fn linux_xr5_readiness_uses_its_native_assets_and_camera_only_needs_none() {
        let mut config = Config::default();
        let assets = config.check_assets();
        assert_eq!(assets.len(), 2);
        assert_eq!(assets[0].label, "XR5 eyelid model");
        assert_eq!(assets[1].label, "XR5 model runtime");
        config.ui.inference_enabled = false;
        assert!(config.missing_required().is_empty());
    }

    #[test]
    fn psvr2_does_not_require_the_unrelated_tobii_runtime() {
        let mut config = Config::default();
        config.hmd.device = "PlayStation-VR2".into();
        assert_eq!(canonical_device_key(&config.hmd.device), "psvr2");
        let missing = config.missing_required();
        assert_eq!(missing.len(), 1, "only the eyelid model is required");
        assert!(missing[0].label.contains("ML weights"));
    }

    #[test]
    #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
    fn unified_build_promotes_an_existing_psvr2_only_config_once() {
        let root = std::env::temp_dir().join(format!(
            "sranibro-unified-psvr2-migration-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let normal = root.join("sranibro.toml");
        let dedicated = root.join("sranibro-psvr2.toml");
        let mut old = Config::default();
        old.hmd.device = "psvr2".into();
        old.output.osc_port = 9017;
        std::fs::write(&dedicated, toml::to_string_pretty(&old).unwrap()).unwrap();

        assert!(Config::write_template_if_absent(&normal).unwrap());
        let (migrated, warning) = Config::load(&normal);
        assert!(warning.is_none());
        assert_eq!(migrated.hmd.device, "psvr2");
        assert_eq!(migrated.output.osc_port, 9017);
        assert!(
            dedicated.is_file(),
            "old dedicated config remains as rollback"
        );

        let mut edited = migrated;
        edited.output.osc_port = 9020;
        edited.save(&normal).unwrap();
        assert!(!Config::write_template_if_absent(&normal).unwrap());
        let (reloaded, _) = Config::load(&normal);
        assert_eq!(reloaded.output.osc_port, 9020);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(feature = "psvr2-only")]
    fn psvr2_only_build_uses_isolated_config_and_enforces_its_route() {
        assert_eq!(config_file_name(), "sranibro-psvr2.toml");
        let root =
            std::env::temp_dir().join(format!("sranibro-psvr2-only-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        let path = root.join(config_file_name());
        std::fs::write(&path, "[hmd]\ndevice = 'starvr'\nwide_source = 'custom'\n").unwrap();

        let (loaded, warning) = Config::load(&path);
        assert!(warning.is_none());
        assert_eq!(loaded.hmd.device, "psvr2");
        assert_eq!(loaded.hmd.wide_source, WideSource::Sranipal);
        assert!(loaded.tobii_runtime_path().is_none());
        assert!(loaded.starvr_runtime_path().is_none());

        loaded.save(&path).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        let saved: Config = toml::from_str(&saved).unwrap();
        assert_eq!(saved.hmd.device, "psvr2");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    #[cfg(feature = "xr5-only")]
    fn xr5_only_build_uses_isolated_config_and_enforces_its_route() {
        assert_eq!(config_file_name(), if cfg!(feature = "xr5-research-model") { "sranibro-xr5-research.toml" } else { "sranibro-xr5.toml" });
        let root =
            std::env::temp_dir().join(format!("sranibro-xr5-only-config-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&root);
        let path = root.join(config_file_name());
        std::fs::write(&path, "[hmd]\ndevice = 'starvr'\nwide_source = 'custom'\n").unwrap();

        let (loaded, warning) = Config::load(&path);
        assert!(warning.is_none());
        assert_eq!(loaded.hmd.device, "pimax_xr5");
        assert_eq!(loaded.hmd.wide_source, WideSource::Custom);

        loaded.save(&path).unwrap();
        let saved = std::fs::read_to_string(&path).unwrap();
        let saved: Config = toml::from_str(&saved).unwrap();
        assert_eq!(saved.hmd.device, "pimax_xr5");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn obsolete_runtime_path_keys_are_ignored_and_removed_on_save() {
        let c: Config = toml::from_str(
            r#"
                [assets]
                tobii_dll = "C:\\legacy\\runtime.bin"
                starvr_dll = "C:\\legacy\\starvr.bin"
                pimax_vr4_dll = "C:\\legacy\\pimax.bin"
            "#,
        )
        .expect("old configuration still parses");
        let encoded = toml::to_string(&c).expect("config serializes");
        assert!(!encoded.contains("tobii_dll"));
        assert!(!encoded.contains("starvr_dll"));
        assert!(!encoded.contains("pimax_vr4_dll"));
    }

    #[test]
    fn varjo_lib_path_prefers_explicit() {
        let mut c = Config::default();
        c.assets.varjo_lib = Some("D:\\pack\\VarjoLib.dll".into());
        assert_eq!(
            c.varjo_lib_path(),
            Some(std::path::PathBuf::from("D:\\pack\\VarjoLib.dll"))
        );
        c.assets.varjo_lib = Some("   ".into());
        assert_ne!(c.varjo_lib_path(), Some(std::path::PathBuf::from("")));
    }

    #[test]
    fn template_is_valid_toml() {
        let c: Config = toml::from_str(TEMPLATE).expect("template parses");
        assert_eq!(c.hmd.device, "auto");
    }

    #[test]
    fn ml_model_takes_precedence_over_sranipal_dir() {
        let mut c = Config::default();
        c.assets.sranipal_dir = Some("X:\\SRanipal".into());
        c.assets.ml_model = Some("D:\\pack\\weights.params".into());
        let p = c.ml_params_path().unwrap();
        assert_eq!(
            p,
            std::path::PathBuf::from("D:\\pack\\weights.params"),
            "direct ml_model wins over sranipal_dir"
        );
        c.assets.ml_model = Some("   ".into());
        let p = c.ml_params_path().unwrap();
        assert!(p.ends_with("00-0000.params_opencl.params"));
    }

    #[test]
    fn default_eye_mapping_presets() {
        assert!(default_eye_mapping("auto").flip_gaze_x);
        assert!(default_eye_mapping("pimax_vr4").flip_gaze_x);
        assert!(default_eye_mapping("pimax_xr5").flip_gaze_x);
        assert!(default_eye_mapping("starvr").flip_gaze_x);
        assert!(default_eye_mapping("psvr2").flip_gaze_x);
        assert!(!default_eye_mapping("varjo").flip_gaze_x);
        assert!(!default_eye_mapping("varjo_mjpeg").flip_gaze_x);
        for device in [
            "auto",
            "pimax_vr4",
            "pimax_xr5",
            "starvr",
            "psvr2",
            "varjo",
            "varjo_mjpeg",
        ] {
            let m = default_eye_mapping(device);
            assert!(!m.swap_eyes);
            assert_eq!(m.swap_gaze_eyes, None);
            assert!(!m.flip_image && !m.ml_mirror_l && !m.ml_mirror_r);
        }
    }
    #[test]
    fn xr5_geometry_preset_and_aliases() {
        for key in ["pimax_xr5", "pimax-xr5", "xr5", "dream_air"] {
            let [l, r] = default_ml_geometry(key);
            assert_eq!(l.crop_right, 0.40);
            assert_eq!(r.crop_left, 0.40);
            assert_eq!(l.crop_top, 0.15);
            assert_eq!(r.crop_bottom, 0.15);
            assert_eq!(l.scale_y, 1.20);
            assert_eq!(r.scale_y, 1.20);
            assert_eq!(l.rotate_deg, -30.0);
            assert_eq!(r.rotate_deg, 30.0);
            assert_eq!(l.mirror_h, None);
            assert_eq!(r.mirror_h, Some(true));
            let mirrored = l.mirrored_x();
            assert_eq!(mirrored.crop_left, r.crop_left);
            assert_eq!(mirrored.crop_right, r.crop_right);
            assert_eq!(mirrored.rotate_deg, r.rotate_deg);
        }
        assert_eq!(default_ml_geometry("pimax_vr4"), [Default::default(); 2]);
        assert_eq!(running_device_key("auto", "pimax-xr5"), "pimax_xr5");
        assert_eq!(running_device_key("varjo_mjpeg", "varjo"), "varjo_mjpeg");
    }

    #[test]
    fn saved_xr5_geometry_overrides_preset_and_legacy_left_applies_to_both() {
        let mut c = Config::default();
        let custom_l = crate::core::types::MlGeometry {
            crop_left: 0.07,
            mirror_h: Some(false),
            ..Default::default()
        };
        let custom_r = crate::core::types::MlGeometry {
            crop_right: 0.09,
            mirror_h: Some(true),
            ..Default::default()
        };
        c.set_geometry("pimax-xr5", [custom_l, custom_r]);
        assert_eq!(c.geometry_for("xr5"), [custom_l, custom_r]);

        let legacy_l = crate::core::types::MlGeometry {
            crop_left: 0.07,
            ..Default::default()
        };
        let mut legacy = Config::default();
        legacy.hmd.geometry.insert("xr5".into(), legacy_l);
        let [l, r] = legacy.geometry_for("pimax_xr5");
        assert_eq!(l, legacy_l);
        assert_eq!(r.crop_left, legacy_l.crop_left);
        assert_eq!(
            r.mirror_h,
            Some(true),
            "old geometry inherits XR5 right mirror"
        );
    }

    #[test]
    fn xr5_gaze_correction_is_per_device_and_round_trips() {
        let dir = std::env::temp_dir();
        let path = dir.join("sranibro_test_gaze_correction_rt.toml");
        let mut c = Config::default();
        let correction = GazeCorrection {
            enabled: true,
            offset_x_deg: [1.25, -0.75],
            offset_y_deg: [0.5, 0.25],
            scale_x: [1.1, 0.9],
            scale_y: [1.0, 1.05],
            vergence_deg: -1.5,
        };
        c.set_gaze_correction("dream_air", correction);
        c.save(&path).expect("save ok");
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert_eq!(back.gaze_correction_for("pimax_xr5"), correction);
        assert_eq!(
            back.gaze_correction_for("pimax_vr4"),
            GazeCorrection::default()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn xr5_gaze_source_defaults_per_eye_and_round_trips_by_canonical_device() {
        let dir = std::env::temp_dir();
        let path = dir.join("sranibro_test_gaze_source_rt.toml");
        let mut c = Config::default();
        assert_eq!(c.gaze_source_for("pimax_xr5"), GazeSource::PerEye);
        c.set_gaze_source("dream_air", GazeSource::Combined);
        assert_eq!(c.gaze_source_for("xr5"), GazeSource::Combined);
        assert_eq!(c.gaze_source_for("pimax_vr4"), GazeSource::PerEye);

        c.save(&path).expect("save ok");
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert_eq!(back.gaze_source_for("pimax_xr5"), GazeSource::Combined);
        assert_eq!(back.gaze_source_for("starvr"), GazeSource::PerEye);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn vr4_gaze_source_defaults_v1_and_both_choices_round_trip_independently() {
        for text in ["", "[hmd]\ndevice = 'crystal_super'", "[hmd]\ndevice = 'psvr2'"] {
            let old: Config = toml::from_str(text).unwrap();
            assert_eq!(old.hmd.vr4_gaze_source, Vr4GazeSource::V1);
            assert!(old.hmd.vr4_gaze_smoothing);
        }
        for source in [Vr4GazeSource::V1, Vr4GazeSource::V2] {
            let mut config = Config::default();
            config.hmd.vr4_gaze_source = source;
            config.hmd.vr4_gaze_smoothing = source == Vr4GazeSource::V1;
            config.set_gaze_source("pimax_xr5", GazeSource::Combined);
            let saved = toml::to_string_pretty(&config).unwrap();
            let restored: Config = toml::from_str(&saved).unwrap();
            assert_eq!(restored.hmd.vr4_gaze_source, source);
            assert_eq!(restored.hmd.vr4_gaze_smoothing, config.hmd.vr4_gaze_smoothing);
            assert_eq!(restored.gaze_source_for("pimax_xr5"), GazeSource::Combined);
        }
        assert_eq!(Vr4GazeSource::V1.stream_id(), 1285);
        assert_eq!(Vr4GazeSource::V2.stream_id(), 1289);
    }

    #[test]
    fn right_eye_left_head_defaults_on_and_explicit_off_is_per_device() {
        let mut config = Config::default();
        assert!(config.right_eye_left_head_for("pimax_vr4"));
        assert!(config.right_eye_left_head_for("pimax_xr5"));

        config.set_right_eye_left_head("dream-air", false);
        assert!(!config.right_eye_left_head_for("pimax_xr5"));
        assert!(config.right_eye_left_head_for("pimax_vr4"));

        let text = toml::to_string_pretty(&config).unwrap();
        let decoded: Config = toml::from_str(&text).unwrap();
        assert!(!decoded.right_eye_left_head_for("xr5"));
        assert!(decoded.right_eye_left_head_for("pimax_vr4"));

        config.set_right_eye_left_head("pimax_xr5", true);
        assert!(config.right_eye_left_head_for("dream-air"));
        assert_eq!(config.hmd.right_eye_left_head.len(), 1);
    }

    #[test]
    fn wink_and_blink_timing_profiles_are_canonical_and_per_device() {
        let mut config = Config::default();
        let mut wink = crate::core::types::WinkProfile::default();
        wink.eyes[0].enabled = true;
        wink.eyes[0].wink_depth = 0.14;
        config.set_wink_profile("dream_air", wink);

        let blink = crate::core::types::BlinkTimingProfile {
            enabled: false,
            min_closed_ms: 50.0,
            ..Default::default()
        };
        config.set_blink_timing_profile("pimax-xr5", blink);

        let encoded = toml::to_string(&config).expect("serialize profiles");
        let decoded: Config = toml::from_str(&encoded).expect("deserialize profiles");
        assert_eq!(decoded.wink_profile_for("xr5"), wink);
        assert_eq!(decoded.blink_timing_profile_for("dream_air"), blink);
        assert_eq!(
            decoded.wink_profile_for("pimax_vr4"),
            crate::core::types::WinkProfile::default()
        );
        assert_eq!(
            decoded.blink_timing_profile_for("pimax_vr4"),
            crate::core::types::BlinkTimingProfile::default()
        );
    }

    #[test]
    fn eyelid_response_profile_defaults_and_sanitization_are_runtime_safe() {
        let defaults = EyelidResponseProfile::default();
        assert_eq!(
            defaults.schema_version,
            EyelidResponseProfile::SCHEMA_VERSION
        );
        assert!(defaults.manual_range);
        assert_eq!(defaults.open_point_offset, [0.03; 2]);
        assert_eq!(defaults.closed_point_depth, [0.40; 2]);
        assert_eq!(defaults.wide_start, [0.0; 2]);
        assert_eq!(defaults.wide_full, [1.0; 2]);
        assert_eq!(defaults.squeeze_start, [0.0; 2]);
        assert_eq!(defaults.squeeze_full, [1.0; 2]);
        assert_eq!(defaults.close_depth_scale, [1.0; 2]);
        assert_eq!(defaults.curve_mid_output, [0.5; 2]);
        assert_eq!(defaults.blink_close_ms, 0.0);
        assert_eq!(defaults.snap_gate_open, 1.0);
        assert!(!defaults.auto_reseat);
        assert!(defaults.is_compatible());

        let dirty = EyelidResponseProfile {
            open_point_offset: [f32::NAN, 0.19],
            closed_point_depth: [f32::INFINITY, 0.20],
            wide_start: [f32::NAN, 0.99],
            wide_full: [f32::INFINITY, 0.20],
            squeeze_start: [-1.0, 0.80],
            squeeze_full: [2.0, 0.81],
            close_depth_scale: [0.10, f32::NAN],
            curve_mid_output: [f32::NEG_INFINITY, 0.99],
            blink_close_ms: 300.0,
            snap_gate_open: f32::INFINITY,
            auto_reseat: false,
            ..defaults
        };
        assert!(!dirty.is_compatible());
        let sanitized = dirty.sanitized();
        assert_eq!(sanitized.open_point_offset, [0.03, 0.19]);
        assert_eq!(sanitized.closed_point_depth, [0.40, 0.24]);
        assert_eq!(sanitized.wide_start, [0.0, 0.15]);
        assert_eq!(sanitized.wide_full, [1.0, 0.20]);
        assert_eq!(sanitized.squeeze_start, [0.0, 0.76]);
        assert_eq!(sanitized.squeeze_full, [1.0, 0.81]);
        assert_eq!(sanitized.close_depth_scale, [0.85, 1.0]);
        assert_eq!(sanitized.curve_mid_output, [0.5, 0.65]);
        assert_eq!(sanitized.blink_close_ms, 160.0);
        assert_eq!(sanitized.snap_gate_open, 1.0);
        assert!(!sanitized.auto_reseat);
        assert!(sanitized.is_compatible());

        let future = EyelidResponseProfile {
            schema_version: EyelidResponseProfile::SCHEMA_VERSION + 1,
            close_depth_scale: [1.1; 2],
            ..defaults
        };
        assert_eq!(future.sanitized(), defaults);
        assert!(!future.is_compatible());

        let mut config = Config::default();
        config
            .hmd
            .eyelid_response_profiles
            .insert("xr5".into(), future);
        assert!(config.has_eyelid_response_profile("pimax_xr5"));
        assert_eq!(config.eyelid_response_profile_for("dream_air"), defaults);
    }

    #[test]
    fn eyelid_editor_link_choice_round_trips_independent_of_equal_endpoints() {
        let mut config: Config = toml::from_str("").unwrap();
        assert_eq!(config.eyelid_response_link_for("vr4"), None);
        config.set_eyelid_response_link("vr4", false);
        config.set_eyelid_response_link("psvr2", true);
        let text = toml::to_string(&config).unwrap();
        let mut decoded: Config = toml::from_str(&text).unwrap();
        assert_eq!(decoded.eyelid_response_link_for("pimax_vr4"), Some(false));
        assert_eq!(decoded.eyelid_response_link_for("psvr2"), Some(true));
        assert_eq!(decoded.eyelid_response_link_for("xr5"), None);
        assert_eq!(decoded.eyelid_response_profile_for("vr4"), EyelidResponseProfile::default());
        decoded.ui.eyelid_response_links.insert("vr4".into(), false);
        decoded.set_eyelid_response_link("pimax-vr4", true);
        assert_eq!(decoded.eyelid_response_link_for("vr4"), Some(true));
        assert_eq!(decoded.ui.eyelid_response_links.len(), 2);
    }

    #[test]
    fn eyelid_editor_link_auto_migration_keeps_explicit_device_choice() {
        let mut config = Config::default();
        config.set_eyelid_response_link("auto", true);
        config.set_eyelid_response_link("vr4", false);
        assert!(config.migrate_auto_device_settings("pimax_vr4"));
        assert_eq!(config.eyelid_response_link_for("auto"), None);
        assert_eq!(config.eyelid_response_link_for("vr4"), Some(false));
        assert_eq!(config.ui.eyelid_response_links.len(), 1);
    }

    #[test]
    fn eyelid_response_profiles_are_canonical_per_hmd_and_round_trip() {
        let mut config = Config::default();
        assert!(!config.has_eyelid_response_profile("pimax_xr5"));
        let xr5 = EyelidResponseProfile {
            close_depth_scale: [0.90, 1.10],
            curve_mid_output: [0.40, 0.60],
            blink_close_ms: 80.0,
            snap_gate_open: 0.25,
            auto_reseat: false,
            ..Default::default()
        };
        config
            .hmd
            .eyelid_response_profiles
            .insert("dream_air".into(), EyelidResponseProfile::default());
        assert!(config.has_eyelid_response_profile("pimax_xr5"));
        config.set_eyelid_response_profile("dream-air", xr5);
        assert_eq!(config.hmd.eyelid_response_profiles.len(), 1);
        assert!(config
            .hmd
            .eyelid_response_profiles
            .contains_key("pimax_xr5"));
        assert_eq!(config.eyelid_response_profile_for("xr5"), xr5);
        assert_eq!(
            config.eyelid_response_profile_for("pimax_vr4"),
            EyelidResponseProfile::default()
        );

        let encoded = toml::to_string(&config).expect("serialize response profile");
        let mut decoded: Config = toml::from_str(&encoded).expect("deserialize response profile");
        assert_eq!(decoded.eyelid_response_profile_for("pimax-xr5"), xr5);

        let vr4 = EyelidResponseProfile {
            blink_close_ms: 55.0,
            ..Default::default()
        };
        decoded.set_eyelid_response_profile("pimax_vr4", vr4);
        decoded
            .hmd
            .eyelid_response_profiles
            .insert("dream_air".into(), xr5);
        assert!(decoded.clear_eyelid_response_profile("pimax-xr5"));
        assert!(!decoded.has_eyelid_response_profile("dream_air"));
        assert_eq!(
            decoded.eyelid_response_profile_for("dream_air"),
            EyelidResponseProfile::default()
        );
        assert!(decoded.has_eyelid_response_profile("vr4"));
        assert_eq!(decoded.eyelid_response_profile_for("vr4"), vr4);
    }

    #[test]
    fn legacy_config_and_partial_response_profile_fill_safe_defaults() {
        let legacy: Config = toml::from_str(
            r#"
                [hmd]
                device = "pimax_vr4"
            "#,
        )
        .expect("pre-response-profile config remains readable");
        assert!(legacy.hmd.eyelid_response_profiles.is_empty());
        assert_eq!(
            legacy.eyelid_response_profile_for("pimax_vr4"),
            EyelidResponseProfile::default()
        );

        let partial: Config = toml::from_str(
            r#"
                [hmd.eyelid_response_profiles.xr5]
                close_depth_scale = [1.10, 0.90]
            "#,
        )
        .expect("partial response profile remains readable");
        let resolved = partial.eyelid_response_profile_for("dream_air");
        assert_eq!(
            resolved.schema_version,
            EyelidResponseProfile::SCHEMA_VERSION
        );
        assert_eq!(resolved.close_depth_scale, [1.10, 0.90]);
        assert_eq!(resolved.curve_mid_output, [0.5; 2]);
        assert_eq!(resolved.blink_close_ms, 0.0);
        assert_eq!(resolved.snap_gate_open, 1.0);
        assert!(!resolved.auto_reseat);
    }

    #[test]
    fn save_sanitizes_current_response_profiles_and_preserves_future_known_values() {
        let path = std::env::temp_dir().join(format!(
            "sranibro_response_profile_save_{}.toml",
            std::process::id()
        ));
        let mut config = Config::default();
        let dirty = EyelidResponseProfile {
            close_depth_scale: [0.10, f32::NAN],
            curve_mid_output: [0.10, 0.90],
            blink_close_ms: 500.0,
            snap_gate_open: f32::INFINITY,
            ..Default::default()
        };
        let future = EyelidResponseProfile {
            schema_version: EyelidResponseProfile::SCHEMA_VERSION + 1,
            manual_range: true,
            open_point_offset: [0.03; 2],
            closed_point_depth: [0.40; 2],
            wide_start: [0.0; 2],
            wide_full: [1.0; 2],
            squeeze_start: [0.0; 2],
            squeeze_full: [1.0; 2],
            close_depth_scale: [1.10, 0.90],
            curve_mid_output: [0.40, 0.60],
            blink_close_ms: 123.0,
            snap_gate_open: 0.42,
            auto_reseat: false,
        };
        config
            .hmd
            .eyelid_response_profiles
            .insert("xr5".into(), dirty);
        config
            .hmd
            .eyelid_response_profiles
            .insert("pimax_vr4".into(), future);

        config.save(&path).expect("save sanitizes current schema");
        let (back, warning) = Config::load(&path);
        assert!(warning.is_none(), "saved profile reloads: {warning:?}");
        assert_eq!(
            back.hmd.eyelid_response_profiles.get("xr5").copied(),
            Some(dirty.sanitized())
        );
        assert_eq!(
            back.hmd.eyelid_response_profiles.get("pimax_vr4").copied(),
            Some(future)
        );
        assert_eq!(
            back.eyelid_response_profile_for("pimax_vr4"),
            EyelidResponseProfile::default()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn auto_response_profile_migrates_once_to_resolved_hmd() {
        let mut config = Config::default();
        let profile = EyelidResponseProfile {
            close_depth_scale: [0.95, 1.05],
            blink_close_ms: 70.0,
            ..Default::default()
        };
        config
            .hmd
            .eyelid_response_profiles
            .insert("AUTO".into(), profile);

        assert!(config.migrate_auto_device_settings("pimax-xr5"));
        assert_eq!(config.eyelid_response_profile_for("dream_air"), profile);
        assert!(!config
            .hmd
            .eyelid_response_profiles
            .keys()
            .any(|key| canonical_device_key(key) == "auto"));
        assert!(!config.migrate_auto_device_settings("pimax_xr5"));

        let saved_alias = EyelidResponseProfile {
            curve_mid_output: [0.45, 0.55],
            blink_close_ms: 90.0,
            ..Default::default()
        };
        let mut alias_wins = Config::default();
        alias_wins
            .hmd
            .eyelid_response_profiles
            .insert("auto".into(), profile);
        alias_wins
            .hmd
            .eyelid_response_profiles
            .insert("xr5".into(), saved_alias);

        assert!(alias_wins.migrate_auto_device_settings("pimax_xr5"));
        assert_eq!(
            alias_wins.eyelid_response_profile_for("pimax_xr5"),
            saved_alias
        );
        assert_eq!(alias_wins.eyelid_response_profile_for("xr5"), saved_alias);
        assert_eq!(
            alias_wins
                .hmd
                .eyelid_response_profiles
                .keys()
                .filter(|key| canonical_device_key(key) == "pimax_xr5")
                .count(),
            1
        );
        assert!(!alias_wins
            .hmd
            .eyelid_response_profiles
            .keys()
            .any(|key| canonical_device_key(key) == "auto"));
    }

    #[test]
    fn clearing_calibration_entries_removes_legacy_aliases_for_only_that_hmd() {
        let mut config = Config::default();
        let xr5_geometry = crate::core::types::MlGeometry {
            crop_left: 0.07,
            ..Default::default()
        };
        let vr4_geometry = crate::core::types::MlGeometry {
            crop_right: 0.04,
            ..Default::default()
        };
        config.hmd.geometry.insert("xr5".into(), xr5_geometry);
        config
            .hmd
            .geometry_r
            .insert("dream_air".into(), xr5_geometry);
        config.hmd.geometry.insert("pimax_vr4".into(), vr4_geometry);

        assert!(config.has_geometry_override("pimax_xr5"));
        assert!(config.clear_geometry("pimax-xr5"));
        assert!(!config.has_geometry_override("dream_air"));
        assert_eq!(config.geometry_for("pimax_xr5"), default_ml_geometry("xr5"));
        assert_eq!(config.geometry_for("pimax_vr4")[0], vr4_geometry);
    }

    #[test]
    fn removing_fitted_blink_timing_restores_default_but_preserves_master_disable() {
        let mut config = Config::default();
        let fitted = crate::core::types::BlinkTimingProfile {
            enabled: true,
            min_closed_ms: 67.0,
            calibrated_unix: 123,
            ..Default::default()
        };
        config.set_blink_timing_profile("xr5", fitted);
        let restored = config.clear_blink_timing_calibration("dream_air");
        assert_eq!(restored, crate::core::types::BlinkTimingProfile::default());
        assert_eq!(
            config.blink_timing_profile_for("pimax_xr5"),
            crate::core::types::BlinkTimingProfile::default()
        );

        let disabled = crate::core::types::BlinkTimingProfile {
            enabled: false,
            min_closed_ms: 75.0,
            calibrated_unix: 456,
            ..Default::default()
        };
        config.set_blink_timing_profile("pimax_vr4", disabled);
        let restored = config.clear_blink_timing_calibration("vr4");
        assert!(!restored.enabled);
        assert_eq!(restored.min_closed_ms, 42.0);
        assert_eq!(restored.calibrated_unix, 0);
        assert!(!config.blink_timing_profile_for("pimax_vr4").enabled);
    }

    #[test]
    fn full_eyelid_reset_is_per_headset_and_removes_dependent_fits() {
        let mut config = Config::default();
        let response = EyelidResponseProfile {
            closed_point_depth: [0.19, 0.22],
            curve_mid_output: [0.4, 0.6],
            ..Default::default()
        };
        let blink = crate::core::types::BlinkTimingProfile {
            enabled: false,
            min_closed_ms: 65.0,
            calibrated_unix: 123,
            ..Default::default()
        };
        for device in ["vr4", "xr5"] {
            config.set_eyelid_response_profile(device, response);
            config.set_blink_timing_profile(device, blink);
            config.set_wink_profile(device, crate::core::types::WinkProfile {
                calibrated_unix: 123,
                ..Default::default()
            });
            config.set_gaze_eyelid_profile(device, crate::core::types::GazeEyelidProfile {
                calibrated_unix: 123,
                ..Default::default()
            });
        }
        let before = config.clone();
        config.reset_eyelid_calibration_for("pimax-vr4");
        assert_eq!(config.eyelid_response_profile_for("vr4"), EyelidResponseProfile::default());
        assert_eq!(config.blink_timing_profile_for("vr4"), Default::default());
        assert_eq!(config.wink_profile_for("vr4"), Default::default());
        assert_eq!(config.gaze_eyelid_profile_for("vr4"), Default::default());
        assert_eq!(config.eyelid_response_profile_for("xr5"), response);
        assert_eq!(config.blink_timing_profile_for("xr5"), blink);
        assert_eq!(config.wink_profile_for("xr5"), before.wink_profile_for("xr5"));
        assert_eq!(config.gaze_eyelid_profile_for("xr5"), before.gaze_eyelid_profile_for("xr5"));
        assert_eq!(config.geometry_for("vr4"), before.geometry_for("vr4"));
        assert_eq!(config.ui.wearing_memory_enabled, before.ui.wearing_memory_enabled);
        let decoded: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        assert_eq!(decoded.eyelid_response_profile_for("vr4"), EyelidResponseProfile::default());
        assert_eq!(decoded.wink_profile_for("vr4"), Default::default());
    }

    #[test]
    fn legacy_wink_eye_profile_defaults_to_openness_only() {
        let legacy = r#"
enabled = true
wink_depth = 0.14
holdout_before = 0.42
holdout_after = 0.03
"#;
        let decoded: crate::core::types::WinkEyeProfile =
            toml::from_str(legacy).expect("legacy wink profile remains readable");
        assert!(decoded.enabled);
        assert_eq!(decoded.wink_depth.to_bits(), 0.14_f32.to_bits());
        assert!(!decoded.squeeze_enabled);
        assert!(decoded.squeeze_enter_delta > decoded.squeeze_release_delta);
    }

    #[test]
    fn legacy_auto_buckets_migrate_once_without_overwriting_device_specific_values() {
        let mut c = Config::default();
        c.hmd.mappings.insert(
            "auto".into(),
            EyeMapping {
                swap_eyes: true,
                ..Default::default()
            },
        );
        c.hmd.geometry.insert(
            "auto".into(),
            crate::core::types::MlGeometry {
                crop_left: 0.12,
                ..Default::default()
            },
        );
        c.hmd.despeckle.insert(
            "auto".into(),
            crate::core::types::DespeckleParams {
                threshold: 0.22,
                ..Default::default()
            },
        );
        c.hmd.brightness.insert(
            "auto".into(),
            crate::core::types::BrightnessNorm {
                strength: 0.25,
                ..Default::default()
            },
        );
        c.hmd
            .gaze_source
            .insert("auto".into(), GazeSource::Combined);
        c.hmd.brightness.insert(
            "pimax_xr5".into(),
            crate::core::types::BrightnessNorm {
                strength: 0.75,
                ..Default::default()
            },
        );

        assert!(c.migrate_auto_device_settings("pimax-xr5"));
        assert!(c.mapping_for("pimax_xr5").swap_eyes);
        assert_eq!(c.geometry_for("pimax_xr5")[0].crop_left, 0.12);
        assert_eq!(c.despeckle_for("pimax_xr5").threshold, 0.22);
        assert_eq!(c.brightness_for("pimax_xr5").strength, 0.75);
        assert_eq!(c.gaze_source_for("pimax_xr5"), GazeSource::Combined);
        assert!(!c.hmd.mappings.contains_key("auto"));
        assert!(!c.hmd.geometry.contains_key("auto"));
        assert!(!c.hmd.despeckle.contains_key("auto"));
        assert!(!c.hmd.brightness.contains_key("auto"));
        assert!(!c.hmd.gaze_source.contains_key("auto"));
        assert!(!c.migrate_auto_device_settings("pimax_xr5"));
    }

    #[test]
    fn mapping_for_uses_stored_then_default() {
        let mut c = Config::default();
        let vr4 = c.mapping_for("pimax_vr4");
        assert!(vr4.flip_gaze_x);
        assert!(!vr4.swap_eyes);
        assert_eq!(vr4.swap_gaze_eyes, None);
        assert!(!c.mapping_for("varjo_mjpeg").flip_gaze_x);

        c.set_mapping(
            "varjo_mjpeg",
            EyeMapping {
                swap_eyes: true,
                ..Default::default()
            },
        );
        let v = c.mapping_for("varjo_mjpeg");
        assert!(v.swap_eyes && !v.flip_gaze_x);
        assert_eq!(v.swap_gaze_eyes, None);

        c.set_mapping(
            "pimax_vr4",
            EyeMapping {
                swap_eyes: false,
                swap_gaze_eyes: Some(true),
                flip_gaze_x: true,
                ..Default::default()
            },
        );
        let migrated = c.mapping_for("pimax_vr4");
        assert!(migrated.swap_eyes);
        assert_eq!(migrated.swap_gaze_eyes, None);
        assert!(migrated.flip_gaze_x);
    }
    #[test]
    fn split_gaze_only_config_is_rewritten_as_whole_stream_swap() {
        let path = std::env::temp_dir().join("sranibro_test_split_mapping.toml");
        let old = r#"
            [hmd]
            device = "pimax_vr4"

            [hmd.mappings.pimax_vr4]
            swap_eyes = false
            swap_gaze_eyes = true
            flip_gaze_x = true
        "#;
        std::fs::write(&path, old).unwrap();
        let (config, err) = Config::load(&path);
        assert!(err.is_none(), "legacy split config loads: {err:?}");
        let resolved = config.mapping_for("pimax_vr4");
        assert!(resolved.swap_eyes);
        assert_eq!(resolved.swap_gaze_eyes, None);

        config.save(&path).unwrap();
        let rewritten = std::fs::read_to_string(&path).unwrap();
        assert!(rewritten.contains("swap_eyes = true"));
        assert!(!rewritten.contains("swap_gaze_eyes"));
        let (back, err) = Config::load(&path);
        assert!(err.is_none());
        assert!(back.mapping_for("pimax_vr4").swap_eyes);
        let _ = std::fs::remove_file(&path);
    }
    #[test]
    fn legacy_mapping_migrates_to_active_device_only() {
        let old = r#"
            [hmd]
            device = "pimax_vr4"
            flip_gaze_x = true
            swap_eyes = true
        "#;
        let dir = std::env::temp_dir();
        let path = dir.join("sranibro_test_migrate.toml");
        std::fs::write(&path, old).unwrap();
        let (c, err) = Config::load(&path);
        assert!(err.is_none());
        let m = c.mapping_for("pimax_vr4");
        assert!(
            m.flip_gaze_x && m.swap_eyes,
            "legacy values carried over: {m:?}"
        );
        assert_eq!(m.swap_gaze_eyes, None);
        assert!(!c.mapping_for("varjo_mjpeg").flip_gaze_x);
        let _ = std::fs::remove_file(&path);

        std::fs::write(&path, "[hmd]\ndevice = \"pimax_vr4\"\n").unwrap();
        let (c2, _) = Config::load(&path);
        assert!(c2.hmd.mappings.is_empty(), "no spurious migration");
        assert!(c2.mapping_for("pimax_vr4").flip_gaze_x);
        assert!(!c2.mapping_for("pimax_vr4").swap_eyes);
        let _ = std::fs::remove_file(&path);
    }
    #[test]
    fn per_device_mapping_round_trips() {
        let dir = std::env::temp_dir();
        let path = dir.join("sranibro_test_mapping_rt.toml");
        let mut c = Config::default();
        c.set_mapping(
            "pimax_vr4",
            EyeMapping {
                swap_eyes: true,
                flip_gaze_x: true,
                ..Default::default()
            },
        );
        c.set_mapping("varjo_mjpeg", EyeMapping::default());
        c.save(&path).expect("save ok");
        let serialized = std::fs::read_to_string(&path).unwrap();
        assert!(!serialized.contains("swap_gaze_eyes"));
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        let vr4 = back.mapping_for("pimax_vr4");
        assert!(vr4.flip_gaze_x && vr4.swap_eyes);
        assert_eq!(vr4.swap_gaze_eyes, None);
        assert!(!back.mapping_for("varjo_mjpeg").flip_gaze_x);
        assert!(!back.mapping_for("varjo_mjpeg").swap_eyes);
        let _ = std::fs::remove_file(&path);
    }
    #[test]
    fn photometric_correction_is_per_device_and_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "sranibro_test_photometric_rt_{}.toml",
            std::process::id()
        ));
        let mut c = Config::default();
        let mut correction = crate::core::types::PhotometricCorrection {
            enabled: true,
            affine: [[1.08, -4.0], [0.94, 7.0]],
            ..Default::default()
        };
        correction.field[0].horizontal = 0.06;
        correction.flatten = crate::core::types::FlattenParams {
            enabled: true,
            strength: 0.35,
            radius: 0.33,
        };
        c.set_photometric_correction("pimax-vr4", correction);
        c.save(&path).expect("save ok");
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert_eq!(back.photometric_correction_for("pimax_vr4"), correction);
        assert_eq!(
            back.photometric_correction_for("varjo"),
            crate::core::types::PhotometricCorrection::default()
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn photometric_fit_support_is_limited_to_frontal_hotmirror_devices() {
        for device in ["pimax_vr4", "pimax_dll", "varjo", "varjo_mjpeg"] {
            assert!(supports_photometric_fit(device), "{device}");
        }
        for device in ["pimax_xr5", "starvr", "auto", "mock"] {
            assert!(!supports_photometric_fit(device), "{device}");
        }
    }

    #[test]
    #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
    fn save_round_trips_and_omits_unset_assets() {
        let dir = std::env::temp_dir();
        let path = dir.join("sranibro_test_save.toml");
        let mut c = Config::default();
        c.assets.ml_model = Some("D:\\pack\\weights.params".into());
        c.hmd.device = "starvr".into();
        c.save(&path).expect("save ok");
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("ml_model"), "set asset written");
        assert!(
            !text.contains("sranipal_dir"),
            "unset asset omitted (no null)"
        );
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert_eq!(back.hmd.device, "starvr");
        assert_eq!(
            back.assets.ml_model.as_deref(),
            Some("D:\\pack\\weights.params")
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn dream_air_profile_is_canonical_and_round_trips() {
        let path = std::env::temp_dir().join(format!(
            "sranibro_test_dream_profile_{}.toml",
            std::process::id()
        ));
        let mut c = Config::default();
        let profile = DreamAirProfile {
            eyechip_serial: Some("XR5-TEST".into()),
            calibrated_unix: 42,
            baseline: [0.61, 0.58],
            blink_depth: [0.22, 0.19],
            wide_supported: [true, false],
            wide_snr: [4.0, 1.2],
            quality_score: 87.0,
            pupil_center: [[0.48, 0.52], [0.51, 0.49]],
            pupil_center_valid: [true; 2],
            ..DreamAirProfile::default()
        };
        c.set_dream_air_profile("dream-air", profile.clone());
        c.save(&path).unwrap();
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert_eq!(back.dream_air_profile_for("pimax_xr5"), Some(&profile));
        assert!(back.dream_air_profile_for("pimax_vr4").is_none());
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn malformed_config_is_preserved_before_defaults_can_be_saved() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "sranibro_malformed_config_{}_{nonce}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("broken.toml");
        std::fs::write(&path, b"[hmd\ninvalid").unwrap();

        let (config, warning) = Config::load(&path);
        let warning = warning.expect("malformed config must be visible");
        assert!(warning.contains("preserved as"));
        assert!(!path.exists(), "the malformed primary must be quarantined");
        let backups: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .collect();
        assert_eq!(backups.len(), 1);
        assert_eq!(
            std::fs::read_to_string(&backups[0]).unwrap(),
            "[hmd\ninvalid"
        );

        config.save(&path).expect("quarantine permits a fresh save");
        let (_, reload_warning) = Config::load(&path);
        assert!(reload_warning.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn state_backup_copies_only_existing_state_files() {
        let root = std::env::temp_dir().join(format!(
            "sranibro_backup_test_{}_{}",
            std::process::id(),
            unix_now()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("sranibro.toml"), "config-v1").unwrap();
        std::fs::write(root.join("sranibro_calib_pimax_xr5.toml"), "xr5-calib").unwrap();
        std::fs::write(root.join("sranibro_calib_starvr.toml"), "starvr-calib").unwrap();
        std::fs::write(root.join("unrelated.toml"), "do-not-copy").unwrap();
        std::fs::create_dir_all(root.join("reseat-references")).unwrap();
        std::fs::write(
            root.join("reseat-references/pimax_xr5_unit.bin"),
            b"reference-v1",
        )
        .unwrap();
        let backup = create_state_backup_at(&root, "before calibration").unwrap();
        assert_eq!(
            std::fs::read_to_string(backup.join("sranibro.toml")).unwrap(),
            "config-v1"
        );
        assert!(!backup.join("sranibro_calib.toml").exists());
        assert_eq!(
            std::fs::read_to_string(backup.join("sranibro_calib_pimax_xr5.toml")).unwrap(),
            "xr5-calib"
        );
        assert_eq!(
            std::fs::read_to_string(backup.join("sranibro_calib_starvr.toml")).unwrap(),
            "starvr-calib"
        );
        assert!(!backup.join("unrelated.toml").exists());
        assert_eq!(
            std::fs::read(backup.join("reseat-references/pimax_xr5_unit.bin")).unwrap(),
            b"reference-v1"
        );
        assert!(backup
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("before-calibration"));
        let _ = std::fs::remove_dir_all(root);
    }
}
