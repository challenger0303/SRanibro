//! Configuration + asset validation (`sranibro.toml`).
//!
//! Distribution model: nothing proprietary ships with SRanibro. The user points
//! us at assets they already own — their SRanipal install (for the ML weights)
//! and the patched Tobii DLLs (for device access). This module loads that config
//! and validates the referenced paths *gracefully*: a missing asset is reported
//! (with the feature it gates) rather than crashing, so the UI can show exactly
//! what to add — the opposite of VRCFT's "it just doesn't work" opacity.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// ML weights location inside a SRanipal install directory.
pub const MODEL_REL: &str = "model/EyePrediction/00-0000.params_opencl.params";

/// Trim a config string option; an empty/whitespace value (a cleared UI field) counts
/// as unset. Shared by the asset-path resolvers.
fn nonempty(o: &Option<String>) -> Option<String> {
    o.as_ref()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Directory that holds `sranibro.toml` + calibration, resolved INDEPENDENTLY of the
/// current working directory (which is `System32`/home when launched from a shortcut,
/// not the exe folder — the #1 distribution bug). Portable mode wins: if a
/// `sranibro.toml` already sits next to the exe, use that folder; otherwise the
/// per-user `%APPDATA%\SRanibro` (always writable, even under Program Files / non-admin).
///
/// Resolved ONCE and cached: every call within a process returns the SAME dir, so the
/// UI's saves, calibration, and logs can't drift to a different folder if files
/// appear/disappear during the run.
pub fn base_dir() -> PathBuf {
    static CACHE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    CACHE.get_or_init(resolve_base_dir).clone()
}

/// True if `dir` exists (created if needed) AND is actually WRITABLE — a real probe,
/// because an existing dir (e.g. a read-only Program Files install) passes
/// `create_dir_all` yet rejects writes.
fn dir_writable(dir: &Path) -> bool {
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    // Per-process probe name (avoids colliding with a concurrent instance), and write a
    // real byte so a full-disk/quota condition is actually detected.
    let probe = dir.join(format!(".sranibro_wtest_{}", std::process::id()));
    let ok = std::fs::write(&probe, b"x").is_ok();
    let _ = std::fs::remove_file(&probe);
    ok
}

fn resolve_base_dir() -> PathBuf {
    // Portable: a config sitting next to the exe wins — but only if that dir is
    // actually writable (else a read-only Program Files install would break saves).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if dir.join("sranibro.toml").is_file() && dir_writable(dir) {
                return dir.to_path_buf();
            }
        }
    }
    // Per-user, always-writable (even under Program Files / non-admin): roaming/local
    // app-data, then TEMP — each accepted only if a write probe succeeds.
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
    // Final fallback: the exe dir if writable, else the ABSOLUTE current dir (resolved
    // now and cached, so it can't change meaning if CWD later moves).
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

/// Resolved path to `sranibro.toml` (see [`base_dir`]).
pub fn config_path() -> PathBuf {
    base_dir().join("sranibro.toml")
}

/// Resolved path to the persisted calibration file.
pub fn calib_path() -> PathBuf {
    base_dir().join("sranibro_calib.toml")
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub assets: Assets,
    pub hmd: Hmd,
    pub output: Output,
    pub ui: Ui,
    /// Persisted post-processing tuning (the calibration sliders).
    pub tuning: crate::core::eye_state::Tuning,
}

/// User-supplied asset paths. All optional — absent means "not configured yet".
/// Nothing here ships with SRanibro; the user points us at assets they own (e.g.
/// from the separate Discord asset pack), and can edit every path live in the UI.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Assets {
    /// Direct path to the EyePrediction weights file (the eye-tracking "recognition"
    /// model). Takes precedence over `sranipal_dir` — lets the user ship just the
    /// one `.params` file instead of a whole SRanipal install.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ml_model: Option<String>,
    /// SRanipal install dir; ML weights are read from `<dir>/MODEL_REL` (used only
    /// when `ml_model` is unset).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sranipal_dir: Option<String>,
    /// Common Tobii stream-engine DLL, shared by every device. REQUIRED to connect:
    /// SRanibro ships inert and only talks to the EyeChip once the user supplies this
    /// (e.g. the patched `tobii_stream_engine_full_unlock.dll` from the asset pack).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tobii_dll: Option<String>,
    /// Varjo SDK client DLL (`VarjoLib.dll`), for `device = "varjo"` (native eye
    /// cameras). Optional: if unset, it is auto-detected from a Varjo Base install
    /// (see [`Config::varjo_lib_path`]). NOT bundled with SRanibro — it ships with
    /// Varjo Base. Without it (and without Varjo Base), use `device = "varjo_mjpeg"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub varjo_lib: Option<String>,
    /// Optional eyebrow model: a `BROWNET1` weights file baked from the user's own
    /// calibrated TinyBrowNet (see tools/bake_brow_weights.py). Per-user/per-HMD, NOT
    /// bundled. When set, SRanibro infers brow expression from eye-shape and emits the
    /// FT-v2 Brow* OSC params. Absent = no brow output (eye tracking unaffected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brow_model: Option<String>,
    /// Optional Dream Air/XR5 image-based EyeWide model. The model is trained from
    /// guided, user-labelled eye-camera frames and never uses SRanipal's EyeWide as
    /// its teacher. Missing = keep the legacy SRanipal-derived EyeWide path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wide_model: Option<String>,
    /// Python interpreter (a user-supplied venv with torch) used to run the OFFLINE
    /// eyebrow training + bake subprocess (B-2). NOT bundled — SRanibro never ships a
    /// Python runtime; the user points us at e.g. `<vr_eyebrow>/venv_cpu/Scripts/python.exe`.
    /// Absent = the "Train & bake" action is disabled (capture still works).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub python_exe: Option<String>,
    /// The user's `vr_eyebrow` project directory — the folder holding `train.py`,
    /// `dataset.py`, and `model.py`. Training runs with this as the working dir (train.py
    /// imports dataset/model by name). NOT bundled. Absent = "Train & bake" disabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vr_eyebrow_dir: Option<String>,
    /// Deprecated — folded into `tobii_dll`; still read for back-compat with old
    /// configs (the UI migrates these into `tobii_dll` on save).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starvr_dll: Option<String>,
    /// Deprecated — folded into `tobii_dll` (see above).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pimax_vr4_dll: Option<String>,
}

/// Standard install locations of `VarjoLib.dll` inside a Varjo Base install, probed
/// (in order) when `[assets].varjo_lib` is unset. The DLL ships with Varjo Base, so a
/// Varjo user already has it — we never bundle it.
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
    // Literal fallback for the common default if the env vars are missing.
    out.push(PathBuf::from(
        r"C:\Program Files\Varjo\varjo-compositor\VarjoLib.dll",
    ));
    out
}

/// Per-device eye-image / gaze orientation. The eye cameras and gaze handedness are
/// wired differently on each supported HMD, so this is stored *per device* (see
/// [`Hmd::mappings`]) and applied automatically when the device is selected — e.g. the
/// Pimax / Tobii stream-engine path needs the gaze X flipped, the Varjo path does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct EyeMapping {
    /// Swap left/right eye images (units wired the other way round).
    pub swap_eyes: bool,
    /// Horizontally mirror each eye image (mirrored optics).
    pub flip_image: bool,
    /// Negate gaze X (left/right). Mirrored on the Tobii stream-engine / pimax path,
    /// not on Varjo — flip if the avatar looks the opposite way left/right.
    pub flip_gaze_x: bool,
    /// Mirror ONE eye's image into the ML only (experiment to steady L/R handedness).
    pub ml_mirror_l: bool,
    pub ml_mirror_r: bool,
}

/// Per-eye gaze trim applied after the HMD's native calibration. This is deliberately
/// a small, XR5-oriented finishing correction: Pimax/Tobii calibration still establishes
/// the real optical model, while these values remove the remaining centre/vergence and
/// range mismatch without touching eye-camera ML or eyelid processing.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeCorrection {
    pub enabled: bool,
    /// Per-eye angular centre trim in degrees, `[left, right]`.
    pub offset_x_deg: [f32; 2],
    pub offset_y_deg: [f32; 2],
    /// Per-eye angular range multiplier, `[left, right]`.
    pub scale_x: [f32; 2],
    pub scale_y: [f32; 2],
    /// Total opposite horizontal separation in degrees. Half is applied to each eye.
    pub vergence_deg: f32,
}

/// Per-headset result of the guided Dream Air / XR5 onboarding flow.  The
/// schema version lets future builds reject or migrate measurements instead of
/// silently treating an old profile as current.
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

/// Stable key used by all per-device settings. Adapter names use hyphens while the UI
/// uses underscores, and `auto` resolves to an adapter only after device sniffing. Keep
/// those spellings from silently creating separate calibration buckets.
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
        "vive-pro-eye" | "vive_pro_eye" | "vpe" => "vpe".into(),
        other => other.replace('-', "_"),
    }
}

/// Resolve the per-device key after an adapter has been constructed. Explicit device
/// selections keep their transport distinction; only `auto` follows the sniffed adapter.
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

/// Built-in default mapping for a device, used until the user overrides it (the override
/// then persists per-device in [`Hmd::mappings`]). The only axis that differs today is
/// gaze handedness: Pimax / Tobii stream-engine is mirrored left/right, Varjo is not.
pub fn default_eye_mapping(device: &str) -> EyeMapping {
    let key = canonical_device_key(device);
    let is_varjo = matches!(key.as_str(), "varjo" | "varjo_mjpeg");
    EyeMapping {
        flip_gaze_x: !is_varjo,
        ..EyeMapping::default()
    }
}

/// Fixed Dream Air/XR5 reconstruction found by evaluating the original SRanipal EyeNet
/// on labeled 200x200 XR5 captures. Other HMDs retain the byte-identical identity path.
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
            // Dream Air/XR5 cameras face in opposite anatomical directions. The
            // SRanipal EyeNet expects a shared handedness, so the right mirror is part
            // of the same preset as its crop/rotation, not a separable mapping default.
            mirror_h: Some(true),
            ..MlGeometry::default()
        },
    ]
}

/// Source of EyeWide. `Sranipal` is the compatibility default. `Auto` uses the
/// custom model only on XR5 when it loaded and is producing fresh finite values;
/// `Custom` is strict and refuses to start without a valid XR5 model.
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

/// Native gaze direction source used by Dream Air / XR5. `PerEye` preserves the
/// existing behaviour. `Combined` uses the EyeChip's own fused column-5 vector for
/// both output eyes; it is deliberately opt-in because it trades dynamic vergence
/// for a steadier signal and takes effect only when the device stream is restarted.
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Hmd {
    /// "auto" (sniff EyeChip serial → VR4/XR5) | "pimax_vr4" | "pimax_xr5" |
    /// "pimax_dll" | "starvr" | "varjo" (= Eye Streamer) | …
    pub device: String,
    /// `device = "varjo_mjpeg"`: the "Varjo Eye Streamer" MJPEG-over-HTTP endpoints for the
    /// left/right eye cameras. (`device = "varjo"` uses the native VarjoLib SDK instead.)
    pub varjo_left_url: String,
    pub varjo_right_url: String,
    /// EyeWide provider. The custom model is intentionally XR5-only; other HMDs
    /// always retain the existing SRanipal-derived path.
    pub wide_source: WideSource,

    // --- Legacy single-mapping fields (pre per-device map). Still read from old configs
    // and migrated into `mappings` on load (see `migrate_legacy_mapping`); never written
    // back, so a re-saved config carries only the per-device map.
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

    /// Per-device eye mapping (swap / flip / gaze / ml-mirror), keyed by the `device`
    /// string. Switching devices swaps in that device's mapping; missing entries fall
    /// back to [`default_eye_mapping`]. Editing the mapping in Settings writes the entry
    /// for the *running* device, so each HMD remembers its own orientation. Declared LAST
    /// so TOML emits all the scalar `[hmd]` keys before this `[hmd.mappings.*]` sub-table.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub mappings: BTreeMap<String, EyeMapping>,

    /// Per-device ML-input geometry (crop / stretch / rotation of the image fed to the
    /// eye model), keyed by the `device` string. A missing entry uses the built-in HMD
    /// preset (identity except for Dream Air/XR5). Edited live in the Calibration tab;
    /// declared LAST so TOML emits the scalar `[hmd]` keys and `[hmd.mappings.*]` before
    /// `[hmd.geometry.*]`.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub geometry: BTreeMap<String, crate::core::types::MlGeometry>,

    /// Per-device RIGHT-eye ML-input geometry. `geometry` holds the LEFT eye and
    /// doubles as the legacy shared value: configs saved before per-eye tuning
    /// have only `geometry`, which then applies to both eyes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub geometry_r: BTreeMap<String, crate::core::types::MlGeometry>,

    /// Per-device specular-dot suppression for the ML input (bright IR / glasses
    /// reflection removal). A missing entry = default (enabled). Declared last.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub despeckle: BTreeMap<String, crate::core::types::DespeckleParams>,

    /// Per-device adaptive brightness / contrast normalization for the ML input (the
    /// learned target re-anchors the input across sessions). Missing entry = default.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub brightness: BTreeMap<String, crate::core::types::BrightnessNorm>,

    /// Per-device illumination flatten (close-up shadow removal) for the ML input.
    /// Missing entry = default (disabled).
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub flatten: BTreeMap<String, crate::core::types::FlattenParams>,

    /// Per-device native-gaze finishing correction. The product UI currently exposes
    /// this only for Dream Air/XR5, but keeping the key explicit prevents it leaking to
    /// another HMD when the user switches devices.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub gaze_correction: BTreeMap<String, GazeCorrection>,

    /// Dream Air / XR5 gaze provider, stored per device so an experimental combined
    /// mode can never leak into VR4, StarVR, or Varjo. Missing = today's per-eye path.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub gaze_source: BTreeMap<String, GazeSource>,

    /// Guided calibration results keyed by the canonical HMD/device name.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub dream_air_profiles: BTreeMap<String, DreamAirProfile>,
}

impl Default for Hmd {
    fn default() -> Self {
        Self {
            // "auto": sniff the EyeChip serial to pick VR4 vs XR5 (Pimax-only). A fresh
            // user gets auto-detection; explicit values still override.
            device: "auto".into(),
            varjo_left_url: "http://localhost:8080".into(),
            varjo_right_url: "http://localhost:8081".into(),
            wide_source: WideSource::Sranipal,
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
            gaze_correction: BTreeMap::new(),
            gaze_source: BTreeMap::new(),
            dream_air_profiles: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Output {
    pub brokeneye: bool,
    pub brokeneye_port: u16,
    /// Live VRCFT-module openness moving-average window. 0/1 = pass-through.
    pub vrcft_filter_samples: u8,
    pub osc: bool,
    /// Send only the eight FT/v2 eyebrow parameters directly to VRChat OSC.
    /// This is independent from `osc`, which sends the complete eye/gaze set;
    /// it lets VRCFT remain the eye source while SRanibro supplies eyebrows.
    pub eyebrow_osc: bool,
    pub osc_host: String,
    pub osc_port: u16,
}

impl Default for Output {
    fn default() -> Self {
        Self {
            brokeneye: true,
            brokeneye_port: 5555,
            vrcft_filter_samples: 10,
            osc: false,
            eyebrow_osc: false,
            osc_host: "127.0.0.1".into(),
            osc_port: 9000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Ui {
    pub steamvr_overlay: bool,
    /// Master switch for eyebrow inference/output. The model stays loaded while this is
    /// off so tracking can be resumed instantly without reconnecting the HMD.
    pub eyebrow_enabled: bool,
}

impl Default for Ui {
    fn default() -> Self {
        Self {
            steamvr_overlay: false,
            eyebrow_enabled: true,
        }
    }
}

/// One validated asset and what it gates.
#[derive(Debug, Clone)]
pub struct AssetStatus {
    pub label: &'static str,
    pub path: Option<PathBuf>,
    pub present: bool,
    /// Whether the engine cannot run its core (ML) without this.
    pub required: bool,
    /// Human-readable note: what works / breaks depending on presence.
    pub gates: String,
}

impl Config {
    /// Load from a TOML file. A missing file yields defaults; a malformed file
    /// yields defaults too (with `Err` so the caller can warn) — never panics.
    pub fn load(path: &Path) -> (Config, Option<String>) {
        match std::fs::read_to_string(path) {
            Ok(text) => match toml::from_str::<Config>(&text) {
                Ok(mut cfg) => {
                    cfg.migrate_legacy_mapping();
                    (cfg, None)
                }
                Err(e) => (Config::default(), Some(format!("{}: {e}", path.display()))),
            },
            Err(_) => (Config::default(), None), // absent is fine -> defaults
        }
    }

    /// Resolved eye mapping for `device`: the user's saved per-device entry, else the
    /// built-in [`default_eye_mapping`] preset (Pimax flips gaze X, Varjo does not).
    pub fn mapping_for(&self, device: &str) -> EyeMapping {
        device_entry(&self.hmd.mappings, device)
            .copied()
            .unwrap_or_else(|| default_eye_mapping(device))
    }

    /// Store the eye mapping for `device` (called when the user edits the toggles for the
    /// currently-running device, so each HMD remembers its own orientation).
    pub fn set_mapping(&mut self, device: &str, m: EyeMapping) {
        self.hmd.mappings.insert(canonical_device_key(device), m);
    }

    /// Resolved PER-EYE ML-input geometry for `device` as `[left, right]`: the saved
    /// entries, else the built-in device preset. The right eye falls back to the LEFT
    /// entry so configs saved before per-eye tuning keep applying their single geometry
    /// to both eyes.
    pub fn geometry_for(&self, device: &str) -> [crate::core::types::MlGeometry; 2] {
        let preset = default_ml_geometry(device);
        let saved_l = device_entry(&self.hmd.geometry, device).copied();
        let mut l = saved_l.unwrap_or(preset[0]);
        let mut r = device_entry(&self.hmd.geometry_r, device)
            .copied()
            .unwrap_or_else(|| if saved_l.is_some() { l } else { preset[1] });
        // Old geometry tables predate `mirror_h`; inherit the per-device preset instead
        // of treating serde's missing field as an explicit `false` override.
        if l.mirror_h.is_none() {
            l.mirror_h = preset[0].mirror_h;
        }
        if r.mirror_h.is_none() {
            r.mirror_h = preset[1].mirror_h;
        }
        [l, r]
    }

    /// Store the per-eye ML-input geometry `[left, right]` for `device` (edited live
    /// in the gear modal, so each HMD remembers its own crop/stretch/angle per eye).
    pub fn set_geometry(&mut self, device: &str, g: [crate::core::types::MlGeometry; 2]) {
        let key = canonical_device_key(device);
        self.hmd.geometry.insert(key.clone(), g[0]);
        self.hmd.geometry_r.insert(key, g[1]);
    }

    /// Resolved specular-dot suppression for `device` (saved per-device entry, else the
    /// default = enabled). Applied to the eye frames before the ML.
    pub fn despeckle_for(&self, device: &str) -> crate::core::types::DespeckleParams {
        device_entry(&self.hmd.despeckle, device)
            .copied()
            .unwrap_or_default()
    }

    /// Store the specular-dot suppression params for `device`.
    pub fn set_despeckle(&mut self, device: &str, d: crate::core::types::DespeckleParams) {
        self.hmd.despeckle.insert(canonical_device_key(device), d);
    }

    /// Resolved brightness normalization for `device` (saved entry incl. the learned
    /// target, else the default). The persisted target re-anchors the input across sessions.
    pub fn brightness_for(&self, device: &str) -> crate::core::types::BrightnessNorm {
        device_entry(&self.hmd.brightness, device)
            .copied()
            .unwrap_or_default()
    }

    /// Store the brightness-normalization params (+ learned target) for `device`.
    pub fn set_brightness(&mut self, device: &str, b: crate::core::types::BrightnessNorm) {
        self.hmd.brightness.insert(canonical_device_key(device), b);
    }

    /// Resolved illumination-flatten params for `device` (saved entry, else default = off).
    pub fn flatten_for(&self, device: &str) -> crate::core::types::FlattenParams {
        device_entry(&self.hmd.flatten, device)
            .copied()
            .unwrap_or_default()
    }

    /// Store the illumination-flatten params for `device`.
    pub fn set_flatten(&mut self, device: &str, f: crate::core::types::FlattenParams) {
        self.hmd.flatten.insert(canonical_device_key(device), f);
    }

    /// Resolved native-gaze finishing correction for `device`.
    pub fn gaze_correction_for(&self, device: &str) -> GazeCorrection {
        device_entry(&self.hmd.gaze_correction, device)
            .copied()
            .unwrap_or_default()
    }

    /// Store the native-gaze finishing correction for `device`.
    pub fn set_gaze_correction(&mut self, device: &str, correction: GazeCorrection) {
        self.hmd
            .gaze_correction
            .insert(canonical_device_key(device), correction);
    }

    /// Resolved native gaze provider for `device`. Missing entries are intentionally
    /// per-eye so existing configs remain byte-for-byte compatible at runtime.
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

    /// Guided Dream Air profile for a device, if that device has completed the
    /// onboarding flow with a compatible schema.
    pub fn dream_air_profile_for(&self, device: &str) -> Option<&DreamAirProfile> {
        device_entry(&self.hmd.dream_air_profiles, device)
            .filter(|profile| profile.schema_version == 1)
    }

    pub fn set_dream_air_profile(&mut self, device: &str, profile: DreamAirProfile) {
        self.hmd
            .dream_air_profiles
            .insert(canonical_device_key(device), profile);
    }

    /// Move settings saved by older builds under the literal `auto` bucket to the HMD
    /// selected by the current serial sniff. A device-specific entry already present at
    /// the destination wins. Returns true when any stale `auto` entry was consumed.
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
            let mut changed = false;
            for key in stale {
                if let Some(value) = map.remove(&key) {
                    map.entry(target.to_string()).or_insert(value);
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
        changed |= move_bucket(&mut self.hmd.gaze_correction, &target);
        changed |= move_bucket(&mut self.hmd.gaze_source, &target);
        changed |= move_bucket(&mut self.hmd.dream_air_profiles, &target);
        changed
    }

    /// One-time migration from the old single global mapping (`[hmd].flip_gaze_x` etc.)
    /// to the per-device map: if no per-device entries exist yet and the legacy fields
    /// were customized, seed the active device's entry from them. An all-default legacy
    /// block (or a fresh template) is left empty so `mapping_for` uses the built-in preset.
    fn migrate_legacy_mapping(&mut self) {
        if !self.hmd.mappings.is_empty() {
            return;
        }
        let legacy = EyeMapping {
            swap_eyes: self.hmd.swap_eyes,
            flip_image: self.hmd.flip_image,
            flip_gaze_x: self.hmd.flip_gaze_x,
            ml_mirror_l: self.hmd.ml_mirror_l,
            ml_mirror_r: self.hmd.ml_mirror_r,
        };
        if legacy != EyeMapping::default() {
            let dev = canonical_device_key(&self.hmd.device);
            self.hmd.mappings.insert(dev, legacy);
        }
    }

    /// Resolved ML weights path: a direct `ml_model` file if set, else
    /// `<sranipal_dir>/MODEL_REL`. Empty strings (cleared UI fields) count as unset.
    pub fn ml_params_path(&self) -> Option<PathBuf> {
        let nonempty = |o: &Option<String>| {
            o.as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        if let Some(m) = nonempty(&self.assets.ml_model) {
            return Some(PathBuf::from(m));
        }
        nonempty(&self.assets.sranipal_dir).map(|d| Path::new(&d).join(MODEL_REL))
    }

    /// Resolved common Tobii stream-engine DLL path: `tobii_dll`, else the legacy
    /// `starvr_dll`/`pimax_vr4_dll` (back-compat). Empty strings count as unset.
    /// This is the user-supplied component that gates ALL device connection.
    pub fn tobii_dll_path(&self) -> Option<PathBuf> {
        let nonempty = |o: &Option<String>| {
            o.as_ref()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
        };
        nonempty(&self.assets.tobii_dll)
            .or_else(|| nonempty(&self.assets.starvr_dll))
            .or_else(|| nonempty(&self.assets.pimax_vr4_dll))
            .map(PathBuf::from)
    }

    /// Resolved `VarjoLib.dll` path for the native Varjo path (`device = "varjo"`):
    /// an explicit `[assets].varjo_lib` if set, else the first auto-detected Varjo
    /// Base copy (see [`varjo_lib_candidates`]). `None` if neither is present.
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

    /// Resolved eyebrow model path (`[assets].brow_model`), if set + non-empty.
    pub fn brow_model_path(&self) -> Option<PathBuf> {
        self.assets
            .brow_model
            .as_ref()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
    }

    /// Resolved custom Dream Air/XR5 EyeWide model path, if configured.
    pub fn wide_model_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.wide_model).map(PathBuf::from)
    }

    /// Resolved Python interpreter path for the offline eyebrow trainer
    /// (`[assets].python_exe`), if set + non-empty. This is the user's venv-with-torch.
    pub fn python_exe_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.python_exe).map(PathBuf::from)
    }

    /// Resolved `vr_eyebrow` project dir (`[assets].vr_eyebrow_dir`), if set + non-empty.
    /// Holds `train.py` / `dataset.py` / `model.py`; used as the trainer's working dir.
    pub fn vr_eyebrow_dir_path(&self) -> Option<PathBuf> {
        nonempty(&self.assets.vr_eyebrow_dir).map(PathBuf::from)
    }

    /// Serialize and write the config to `path` (used by the UI's live editor).
    /// Comments are not preserved (TOML serialize drops them); the file becomes a
    /// plain key/value document after the first save.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let text = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        write_atomic(path, text.as_bytes())
    }

    /// Validate every referenced asset. Order: ML (required) then the DLLs
    /// (required only for their device). Each entry says what it gates.
    pub fn check_assets(&self) -> Vec<AssetStatus> {
        let exists = |p: &Option<PathBuf>| p.as_ref().map(|p| p.is_file()).unwrap_or(false);

        let ml = self.ml_params_path();
        let tobii = self.tobii_dll_path();

        vec![
            AssetStatus {
                label: "SRanipal ML weights (common)",
                present: exists(&ml),
                path: ml,
                required: true,
                gates: "eyelid openness/wide/squeeze (core). Set [assets].ml_model (direct \
                        weights file) or [assets].sranipal_dir."
                    .into(),
            },
            AssetStatus {
                label: "Tobii DLL (common, required to connect)",
                present: exists(&tobii),
                path: tobii,
                // REQUIRED for every device now: SRanibro will not open the EyeChip
                // (Pimax or StarVR) without the user-supplied Tobii DLL.
                required: true,
                gates: "device connection (Pimax + StarVR). Without it SRanibro stays \
                        inert — set [assets].tobii_dll, then reload."
                    .into(),
            },
        ]
    }

    /// Assets that are required-but-missing (the startup blockers to surface).
    pub fn missing_required(&self) -> Vec<AssetStatus> {
        self.check_assets()
            .into_iter()
            .filter(|a| a.required && !a.present)
            .collect()
    }

    /// Write a commented starter config if none exists yet (first-run UX).
    pub fn write_template_if_absent(path: &Path) -> std::io::Result<bool> {
        if path.exists() {
            return Ok(false);
        }
        std::fs::write(path, TEMPLATE)?;
        Ok(true)
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&tmp, bytes)?;
    let rollback = path.with_extension(format!("rollback-{}", std::process::id()));
    let had_original = path.exists();
    if had_original {
        let _ = std::fs::remove_file(&rollback);
        std::fs::rename(path, &rollback)?;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&rollback);
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            if had_original {
                let _ = std::fs::rename(&rollback, path);
            }
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
    for name in ["sranibro.toml", "sranibro_calib.toml"] {
        let source = base.join(name);
        if source.is_file() {
            std::fs::copy(source, dir.join(name))?;
        }
    }
    Ok(dir)
}

/// Snapshot the two user-editable state files before applying a guided profile.
pub fn create_state_backup(label: &str) -> std::io::Result<PathBuf> {
    create_state_backup_at(&base_dir(), label)
}

/// Most recently named backup directory, if any.
pub fn latest_state_backup() -> Option<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(base_dir().join("backups"))
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .collect();
    entries.sort_by_key(|entry| entry.file_name());
    entries.last().map(|entry| entry.path())
}

/// Restore a prior state snapshot. Missing files are left unchanged so backups
/// made before a calibration file existed remain usable.
pub fn restore_state_backup(dir: &Path) -> std::io::Result<()> {
    let base = base_dir();
    for name in ["sranibro.toml", "sranibro_calib.toml"] {
        let source = dir.join(name);
        if source.is_file() {
            let bytes = std::fs::read(source)?;
            write_atomic(&base.join(name), &bytes)?;
        }
    }
    Ok(())
}

/// Commented first-run template. Hand-authored (toml serialize drops comments).
pub const TEMPLATE: &str = r#"# SRanibro configuration.
# Nothing proprietary ships with SRanibro — point it at assets you already own.

[assets]
# Direct path to the EyePrediction weights file (the eye-tracking "recognition"
# model). If set, this is used as-is — ship just this one file (e.g. from the
# Discord asset pack) instead of a whole SRanipal install. Takes precedence over
# sranipal_dir. All asset paths are also editable live in the Settings tab.
# ml_model = "C:\\sranibro-assets\\00-0000.params_opencl.params"

# Your SRanipal install directory. Used only when ml_model is unset; weights read
# from <sranipal_dir>/model/EyePrediction/00-0000.params_opencl.params
# sranipal_dir = "C:\\Program Files\\VIVE\\SRanipalRuntime"

# Common Tobii stream-engine DLL — REQUIRED to connect to ANY device (Pimax + StarVR).
# Supply your own from the asset pack; it is NOT distributed with SRanibro. Without it
# SRanibro stays inert and will not open the EyeChip. The file can be named anything —
# SRanibro loads whatever path you set here. Editable live in the Settings tab.
# tobii_dll = "C:\\sranibro-assets\\tobii_stream_engine.dll"
# (Legacy starvr_dll / pimax_vr4_dll are still read for back-compat and migrated to
# tobii_dll on first save from the UI.)

# Eyebrow (B-2) train-and-bake inputs — used ONLY by the "Train & bake" button on the
# Eyebrow-calibration tab. NOT bundled: point at a Python venv that has torch, and at
# your local vr_eyebrow project (the folder with train.py / dataset.py / model.py).
# python_exe = "C:\\vr_eyebrow\\venv_cpu\\Scripts\\python.exe"
# vr_eyebrow_dir = "C:\\vr_eyebrow"

# Optional Dream Air/XR5 image-based EyeWide model. This is produced from the
# guided Wide dataset and is never bundled with SRanibro.
# wide_model = "C:\\sranibro-assets\\wide.bin"

[hmd]
# device = which HMD adapter to use. All share the Tobii IS4 EyeChip core; only the
# transport differs. Selecting an unimplemented one fails with a clear message.
device = "auto"          # auto | pimax_vr4 | pimax_xr5 | varjo | varjo_mjpeg | starvr
                         # auto = sniff the EyeChip serial and pick Pimax VR4 (frontal)
                         #        vs XR5 (angled) automatically (falls back to VR4 if no
                         #        Pimax eyechip is present). Pimax-only — StarVR/Varjo/VPE
                         #        still need their explicit device= value below.
                         # pimax_vr4 = WinUSB-direct, frontal ML (Tobii DLL = connection gate).
                         # pimax_xr5 = WinUSB-direct, angled ML (crop + flip).
                         # varjo = native VarjoLib SDK eye cameras (auto-detects Varjo Base).
                         # varjo_mjpeg = Varjo Eye Streamer (MJPEG); run it + "Start Server".
                         # starvr = Tobii stream-engine DLL.
wide_source = "sranipal" # sranipal | auto | custom (custom is Dream Air/XR5 only)

# Eye mapping (swap L/R, flip image, flip gaze, ml-mirror) is stored PER DEVICE and set
# from a sensible default the first time you select a device (Pimax flips gaze X, Varjo
# does not). Edit it live in the Settings tab; each HMD remembers its own. Saved configs
# write the per-device tables here, e.g.:
#   [hmd.mappings.pimax_vr4]
#   flip_gaze_x = true
#   [hmd.mappings.varjo_mjpeg]
#   flip_gaze_x = false

# Dream Air / XR5 only: post-calibration gaze finishing correction. Normally edited
# live from Calibration -> XR5 Gaze correction, not by hand.
#   [hmd.gaze_correction.pimax_xr5]
#   enabled = true
#   offset_x_deg = [0.0, 0.0]
#   offset_y_deg = [0.0, 0.0]
#   scale_x = [1.0, 1.0]
#   scale_y = [1.0, 1.0]
#   vergence_deg = 0.0
# Optional XR5 EyeChip gaze provider (default is per-eye). The UI writes:
#   [hmd.gaze_source]
#   pimax_xr5 = "combined" # per_eye | combined

[output]
brokeneye = true         # VRCFT-compatible TCP sink (BrokenEye protocol, port 5555)
brokeneye_port = 5555
vrcft_filter_samples = 10 # VRCFT openness moving average; 0/1 = off, live-adjustable
osc = false              # set true to ALSO send VRChat OSC direct (/avatar/parameters/Eye*)
eyebrow_osc = false      # eyebrow-only OSC (FT/v2 Brow*); use with VRCFT eye tracking
osc_host = "127.0.0.1"
osc_port = 9000

[ui]
# NOTE: the SteamVR in-headset overlay is not yet ported to the Rust build
# (desktop dashboard only). Leaving this true just logs a notice.
steamvr_overlay = false
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_sane() {
        let c = Config::default();
        assert_eq!(c.hmd.device, "auto");
        assert_eq!(c.hmd.wide_source, WideSource::Sranipal);
        assert!(c.output.brokeneye);
        assert_eq!(c.output.brokeneye_port, 5555);
        assert!(!c.output.eyebrow_osc);
        assert!(
            c.ml_params_path().is_none(),
            "no sranipal_dir -> no ML path"
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
    fn missing_assets_reported_not_panicked() {
        // Nothing configured -> BOTH the ML weights and the Tobii DLL are required
        // and missing (the DLL now gates all connection).
        let c = Config::default();
        let missing = c.missing_required();
        assert_eq!(missing.len(), 2, "ML weights + Tobii DLL are both required");
        let labels: Vec<&str> = missing.iter().map(|a| a.label).collect();
        assert!(
            labels.iter().any(|l| l.contains("ML weights")),
            "ML required: {labels:?}"
        );
        assert!(
            labels.iter().any(|l| l.contains("Tobii DLL")),
            "Tobii DLL required: {labels:?}"
        );
        assert!(missing.iter().all(|a| !a.present));
    }

    #[test]
    fn tobii_dll_path_prefers_common_then_legacy() {
        let mut c = Config::default();
        // Legacy fields are read for back-compat when tobii_dll is unset.
        c.assets.starvr_dll = Some("L:\\old\\starvr.dll".into());
        assert_eq!(
            c.tobii_dll_path(),
            Some(std::path::PathBuf::from("L:\\old\\starvr.dll"))
        );
        // The common field wins once set.
        c.assets.tobii_dll = Some("C:\\pack\\tobii.dll".into());
        assert_eq!(
            c.tobii_dll_path(),
            Some(std::path::PathBuf::from("C:\\pack\\tobii.dll"))
        );
        // Empty common field falls back to legacy.
        c.assets.tobii_dll = Some("  ".into());
        assert_eq!(
            c.tobii_dll_path(),
            Some(std::path::PathBuf::from("L:\\old\\starvr.dll"))
        );
    }

    #[test]
    fn varjo_lib_path_prefers_explicit() {
        let mut c = Config::default();
        // An explicit path always wins over auto-detect.
        c.assets.varjo_lib = Some("D:\\pack\\VarjoLib.dll".into());
        assert_eq!(
            c.varjo_lib_path(),
            Some(std::path::PathBuf::from("D:\\pack\\VarjoLib.dll"))
        );
        // Empty string is treated as unset (falls through to auto-detect, which may or
        // may not find a Varjo Base install on this machine — so only assert it's not
        // the empty string echoed back).
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
        // An empty ml_model (cleared UI field) falls back to sranipal_dir.
        c.assets.ml_model = Some("   ".into());
        let p = c.ml_params_path().unwrap();
        assert!(p.ends_with("00-0000.params_opencl.params"));
    }

    #[test]
    fn default_eye_mapping_presets() {
        // Pimax / Tobii path is gaze-mirrored; Varjo is not. "auto" resolves to a Pimax
        // eyechip, so it keeps the gaze-mirrored preset.
        assert!(default_eye_mapping("auto").flip_gaze_x);
        assert!(default_eye_mapping("pimax_vr4").flip_gaze_x);
        assert!(default_eye_mapping("pimax_xr5").flip_gaze_x);
        assert!(default_eye_mapping("starvr").flip_gaze_x);
        assert!(!default_eye_mapping("varjo").flip_gaze_x);
        assert!(!default_eye_mapping("varjo_mjpeg").flip_gaze_x);
        // ML mirroring now lives atomically with per-eye geometry, not mapping.
        for d in ["pimax_vr4", "pimax_xr5", "varjo_mjpeg"] {
            let m = default_eye_mapping(d);
            assert!(!m.swap_eyes && !m.flip_image && !m.ml_mirror_l && !m.ml_mirror_r);
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
        // A real device-specific value wins over the stale shared bucket.
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
        // Unset -> built-in preset (Pimax flips, Varjo doesn't).
        assert!(c.mapping_for("pimax_vr4").flip_gaze_x);
        assert!(!c.mapping_for("varjo_mjpeg").flip_gaze_x);
        // A stored override wins, independently per device.
        c.set_mapping(
            "varjo_mjpeg",
            EyeMapping {
                swap_eyes: true,
                ..Default::default()
            },
        );
        let v = c.mapping_for("varjo_mjpeg");
        assert!(v.swap_eyes && !v.flip_gaze_x);
        // Pimax still uses its preset (untouched).
        assert!(c.mapping_for("pimax_vr4").flip_gaze_x);
    }

    #[test]
    fn legacy_mapping_migrates_to_active_device_only() {
        // An old config with the single global mapping + a non-default value migrates to
        // the active device; a fresh (all-default) legacy block does not (preset wins).
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
        // Varjo (never configured) still gets its built-in preset.
        assert!(!c.mapping_for("varjo_mjpeg").flip_gaze_x);
        let _ = std::fs::remove_file(&path);

        // All-default legacy block -> no migration -> built-in preset (gaze flipped).
        std::fs::write(&path, "[hmd]\ndevice = \"pimax_vr4\"\n").unwrap();
        let (c2, _) = Config::load(&path);
        assert!(c2.hmd.mappings.is_empty(), "no spurious migration");
        assert!(c2.mapping_for("pimax_vr4").flip_gaze_x);
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
                flip_gaze_x: true,
                ..Default::default()
            },
        );
        c.set_mapping("varjo_mjpeg", EyeMapping::default());
        c.save(&path).expect("save ok");
        let (back, err) = Config::load(&path);
        assert!(err.is_none(), "reloads cleanly: {err:?}");
        assert!(back.mapping_for("pimax_vr4").flip_gaze_x);
        assert!(!back.mapping_for("varjo_mjpeg").flip_gaze_x);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
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
    fn state_backup_copies_only_existing_state_files() {
        let root = std::env::temp_dir().join(format!(
            "sranibro_backup_test_{}_{}",
            std::process::id(),
            unix_now()
        ));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("sranibro.toml"), "config-v1").unwrap();
        let backup = create_state_backup_at(&root, "before calibration").unwrap();
        assert_eq!(
            std::fs::read_to_string(backup.join("sranibro.toml")).unwrap(),
            "config-v1"
        );
        assert!(!backup.join("sranibro_calib.toml").exists());
        assert!(backup
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("before-calibration"));
        let _ = std::fs::remove_dir_all(root);
    }
}
