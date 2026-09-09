//! Read-only PlayStation VR2 Toolkit CAPI bridge.
//!
//! The Toolkit publishes gaze state and the paired eye-camera preview through its
//! installed `psvr2_toolkit_capi.dll`. SRanibro discovers that DLL from the
//! Toolkit's temporary path hint, active OpenVR drivers, and every registered
//! Steam library. This keeps the application self-contained without copying or
//! redistributing Toolkit binaries.
//!
//! This low-level module deliberately exposes raw camera halves as `A` and `B`.
//! The product adapter applies the anatomical mapping established by a labelled
//! hardware recording; the standalone probe retains neutral A/B names so future
//! Toolkit or headset revisions can be audited without assuming that mapping.

#![cfg(windows)]

use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::io;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr;

use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
use windows_sys::Win32::System::LibraryLoader::{
    GetProcAddress, LoadLibraryExW, LOAD_WITH_ALTERED_SEARCH_PATH,
};

pub const GAZE_STATUS_BYTES: usize = 0x148;
pub const IMAGE_HEADER_BYTES: usize = 0x100;
pub const IMAGE_WIDTH: usize = 400;
pub const IMAGE_HEIGHT: usize = 200;
pub const CAMERA_WIDTH: usize = IMAGE_WIDTH / 2;
pub const CAMERA_HEIGHT: usize = IMAGE_HEIGHT;
pub const IMAGE_PIXELS: usize = IMAGE_WIDTH * IMAGE_HEIGHT;

pub const RESULT_OK: i32 = 0;
pub const RESULT_DRIVER_INACTIVE: i32 = -1;
pub const RESULT_NO_SLOT: i32 = -2;

const CAPI_DLL: &str = "psvr2_toolkit_capi.dll";
const PSVR2_APP_DIR: &str = "PlayStation VR2 App";
const TOOLKIT_DRIVER_REL: &str = r"SteamVR_Plug-In\bin\win64";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GazeVec2 {
    pub x: f32,
    pub y: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GazeVec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct LensConfig {
    pub left: GazeVec3,
    pub right: GazeVec3,
}

/// `hmd2_gaze_wearable_eye_t` from PSVR2Toolkit's current `hmd2_gaze.h`.
///
/// Toolkit booleans are explicitly 32-bit integers, not C/Rust `bool`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WearableEye {
    pub is_gaze_origin_valid: u32,
    pub gaze_origin_mm: GazeVec3,
    pub is_gaze_dir_valid: u32,
    pub gaze_dir_norm: GazeVec3,
    pub is_pupil_dia_valid: u32,
    pub pupil_dia_mm: f32,
    pub is_pupil_pos_in_sensor_area_valid: u32,
    pub pupil_pos_in_sensor_area: GazeVec2,
    pub is_pos_guide_valid: u32,
    pub pos_guide: GazeVec2,
    pub is_blink_valid: u32,
    pub blink: u32,
}

impl WearableEye {
    #[inline]
    pub fn gaze_valid(&self) -> bool {
        self.is_gaze_dir_valid != 0 && finite_vec3(self.gaze_dir_norm)
    }

    #[inline]
    pub fn pupil_valid(&self) -> bool {
        self.is_pupil_dia_valid != 0 && self.pupil_dia_mm.is_finite()
    }

    #[inline]
    pub fn pupil_position_valid(&self) -> bool {
        self.is_pupil_pos_in_sensor_area_valid != 0
            && self.pupil_pos_in_sensor_area.x.is_finite()
            && self.pupil_pos_in_sensor_area.y.is_finite()
    }

    #[inline]
    pub fn blink_valid(&self) -> bool {
        self.is_blink_valid != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct WearableData {
    pub timestamp: i64,
    pub frame_counter: u32,
    pub left: WearableEye,
    pub right: WearableEye,
    pub is_gaze_origin_combined_valid: u32,
    pub gaze_origin_combined_mm: GazeVec3,
    pub is_gaze_dir_combined_valid: u32,
    pub gaze_dir_combined_norm: GazeVec3,
    pub is_convergence_distance_valid: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FoveatedGaze {
    pub timestamp: i64,
    pub frame_counter: u32,
    pub tracking_state: u32,
    pub gaze_dir_left_norm: GazeVec3,
    pub gaze_dir_right_norm: GazeVec3,
    pub gaze_dir_combined_norm: GazeVec3,
    pub convergence_distance_mm: f32,
}

/// Exact 0x148-byte status block emitted by current PSVR2Toolkit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct GazeStatus {
    pub magic: [u8; 2],
    pub version: u16,
    pub size: u32,
    pub exp_l: f32,
    pub exp_r: f32,
    pub led_status: u32,
    pub exp_counter_l: u32,
    pub exp_counter_r: u32,
    pub led_counter: u32,
    pub dsp_return_code: i32,
    pub lens_config: LensConfig,
    pub user_calibration_id: u32,
    pub fr_gaze_origin: GazeVec3,
    pub enabled_eye: u8,
    pub motor_sequence: u8,
    pub motor_strength: u8,
    pub wearable: WearableData,
    pub foveated: FoveatedGaze,
}

const _: () = assert!(std::mem::size_of::<WearableEye>() == 72);
const _: () = assert!(std::mem::size_of::<WearableData>() == 192);
const _: () = assert!(std::mem::size_of::<FoveatedGaze>() == 56);
const _: () = assert!(std::mem::size_of::<GazeStatus>() == GAZE_STATUS_BYTES);

#[inline]
fn finite_vec3(v: GazeVec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

/// Copied, stable subset of one Toolkit eye-image buffer.
#[derive(Clone, Debug)]
pub struct EyeImageFrame {
    pub header: [u8; IMAGE_HEADER_BYTES],
    pub pixels: Vec<u8>,
    pub version: u16,
    pub total_size: u32,
    pub timestamp: u32,
    pub image_type: u16,
}

impl EyeImageFrame {
    /// Copy a CAPI-owned frame before the circular buffer advances.
    ///
    /// # Safety
    /// `source` must be the non-null pointer returned by
    /// `psvr2_toolkit_gaze_image` and remain valid for at least
    /// `IMAGE_HEADER_BYTES + IMAGE_PIXELS` readable bytes during this call.
    pub unsafe fn copy_from_capi(source: *const u8) -> io::Result<Self> {
        if source.is_null() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PSVR2 Toolkit returned a null image pointer",
            ));
        }

        let mut header = [0u8; IMAGE_HEADER_BYTES];
        ptr::copy_nonoverlapping(source, header.as_mut_ptr(), header.len());
        if header[0..2] != *b"VI" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected PSVR2 image magic {:02X} {:02X}",
                    header[0], header[1]
                ),
            ));
        }

        let version = u16::from_le_bytes([header[2], header[3]]);
        let total_size = u32::from_le_bytes(header[4..8].try_into().unwrap());
        let timestamp = u32::from_le_bytes(header[8..12].try_into().unwrap());
        let image_type = u16::from_le_bytes(header[16..18].try_into().unwrap());
        if total_size != 0 && total_size < (IMAGE_HEADER_BYTES + IMAGE_PIXELS) as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSVR2 image block is too small: {total_size} bytes (need at least {})",
                    IMAGE_HEADER_BYTES + IMAGE_PIXELS
                ),
            ));
        }

        let mut pixels = vec![0u8; IMAGE_PIXELS];
        ptr::copy_nonoverlapping(
            source.add(IMAGE_HEADER_BYTES),
            pixels.as_mut_ptr(),
            IMAGE_PIXELS,
        );
        Ok(Self {
            header,
            pixels,
            version,
            total_size,
            timestamp,
            image_type,
        })
    }

    /// Split the 400x200 preview into two 200x200 camera planes without claiming
    /// that either plane is anatomically left or right.
    pub fn split_camera_planes(&self) -> [Vec<u8>; 2] {
        let mut a = vec![0u8; CAMERA_WIDTH * CAMERA_HEIGHT];
        let mut b = vec![0u8; CAMERA_WIDTH * CAMERA_HEIGHT];
        for y in 0..CAMERA_HEIGHT {
            let src = &self.pixels[y * IMAGE_WIDTH..(y + 1) * IMAGE_WIDTH];
            let dst = y * CAMERA_WIDTH..(y + 1) * CAMERA_WIDTH;
            a[dst.clone()].copy_from_slice(&src[..CAMERA_WIDTH]);
            b[dst].copy_from_slice(&src[CAMERA_WIDTH..]);
        }
        [a, b]
    }
}

type InitFn = unsafe extern "C" fn() -> i32;
type DeinitFn = unsafe extern "C" fn();
type DriverActiveFn = unsafe extern "C" fn() -> u8;
type GazeStatusFn = unsafe extern "C" fn(*mut GazeStatus, u32) -> u8;
type GazeImageFn = unsafe extern "C" fn(*mut *mut u8, u32) -> u8;

/// Dynamically loaded, read-only subset of PSVR2Toolkit CAPI.
pub struct Psvr2Capi {
    module: HMODULE,
    module_path: PathBuf,
    init: InitFn,
    deinit: DeinitFn,
    get_driver_active: Option<DriverActiveFn>,
    gaze_status: GazeStatusFn,
    gaze_image: GazeImageFn,
    initialized: bool,
}

impl Psvr2Capi {
    pub fn load() -> io::Result<Self> {
        let candidates = capi_candidates();
        let mut load_errors = Vec::new();
        for path in &candidates {
            if !path.is_file() {
                continue;
            }
            match unsafe { Self::load_path(path) } {
                Ok(capi) => {
                    eprintln!("[psvr2] CAPI loaded: {}", path.display());
                    return Ok(capi);
                }
                Err(error) => {
                    eprintln!(
                        "[psvr2] CAPI candidate could not be loaded: {}: {error}",
                        path.display()
                    );
                    load_errors.push((path.clone(), error));
                }
            }
        }
        if let Some((path, error)) = load_errors.into_iter().next() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSVR2Toolkit CAPI was found at {} but could not be loaded: {error}. Keep its dependency DLLs beside it and use a compatible Toolkit version",
                    path.display()
                ),
            ));
        }
        eprintln!(
            "[psvr2] CAPI not found; checked {} candidate paths",
            candidates.len()
        );
        for path in candidates {
            eprintln!("[psvr2] CAPI checked: {}", path.display());
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            "PSVR2Toolkit CAPI was not found. Start SteamVR with PSVR2Toolkit installed; SRanibro also checked every detected Steam library.",
        ))
    }

    unsafe fn load_path(path: &Path) -> io::Result<Self> {
        let wide: Vec<u16> = path
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        let module = LoadLibraryExW(
            wide.as_ptr(),
            ptr::null_mut(),
            LOAD_WITH_ALTERED_SEARCH_PATH,
        );
        if module.is_null() {
            return Err(io::Error::last_os_error());
        }

        let result = (|| {
            let init = required_proc::<InitFn>(module, b"psvr2_toolkit_init\0")?;
            let deinit = required_proc::<DeinitFn>(module, b"psvr2_toolkit_deinit\0")?;
            let gaze_status =
                required_proc::<GazeStatusFn>(module, b"psvr2_toolkit_gaze_status\0")?;
            let gaze_image = required_proc::<GazeImageFn>(module, b"psvr2_toolkit_gaze_image\0")?;
            let get_driver_active =
                optional_proc::<DriverActiveFn>(module, b"psvr2_toolkit_get_driver_active\0");
            Ok(Self {
                module,
                module_path: path.to_path_buf(),
                init,
                deinit,
                get_driver_active,
                gaze_status,
                gaze_image,
                initialized: false,
            })
        })();
        if result.is_err() {
            FreeLibrary(module);
        }
        result
    }

    pub fn module_path(&self) -> &Path {
        &self.module_path
    }

    pub fn initialize(&mut self) -> i32 {
        if self.initialized {
            return RESULT_OK;
        }
        let result = unsafe { (self.init)() };
        self.initialized = result == RESULT_OK;
        result
    }

    pub fn driver_active(&self) -> Option<bool> {
        self.get_driver_active.map(|get| unsafe { get() != 0 })
    }

    pub fn next_gaze(&self, timeout_ms: u32) -> io::Result<Option<GazeStatus>> {
        let mut status = GazeStatus::default();
        let changed = unsafe { (self.gaze_status)(&mut status, timeout_ms) } != 0;
        if !changed {
            return Ok(None);
        }
        if status.magic != *b"GS" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected PSVR2 gaze magic {:02X} {:02X}",
                    status.magic[0], status.magic[1]
                ),
            ));
        }
        if status.size != 0 && status.size < GAZE_STATUS_BYTES as u32 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSVR2 gaze status size {} is smaller than {}",
                    status.size, GAZE_STATUS_BYTES
                ),
            ));
        }
        Ok(Some(status))
    }

    pub fn next_image(&self, timeout_ms: u32) -> io::Result<Option<EyeImageFrame>> {
        let mut source = ptr::null_mut();
        let changed = unsafe { (self.gaze_image)(&mut source, timeout_ms) } != 0;
        if !changed {
            return Ok(None);
        }
        unsafe { EyeImageFrame::copy_from_capi(source).map(Some) }
    }
}

impl Drop for Psvr2Capi {
    fn drop(&mut self) {
        unsafe {
            if self.initialized {
                (self.deinit)();
                self.initialized = false;
            }
            if !self.module.is_null() {
                FreeLibrary(self.module);
                self.module = ptr::null_mut();
            }
        }
    }
}

fn capi_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |path: PathBuf| {
        let key = path.to_string_lossy().to_ascii_lowercase();
        if !key.is_empty() && seen.insert(key) {
            out.push(path);
        }
    };

    if let Some(value) = std::env::var_os("PSVR2_TOOLKIT_CAPI") {
        push(capi_path_from_hint(PathBuf::from(value)));
    }

    let path_file = std::env::temp_dir().join("psvr2tk_capi_path.txt");
    if let Ok(text) = std::fs::read_to_string(path_file) {
        if let Some(line) = text.lines().next().map(str::trim).filter(|s| !s.is_empty()) {
            push(capi_path_from_hint(PathBuf::from(line)));
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            push(parent.join(CAPI_DLL));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        push(cwd.join(CAPI_DLL));
    }

    for root in openvr_registered_paths() {
        push(root.join(CAPI_DLL));
        push(root.join("bin").join("win64").join(CAPI_DLL));
        push(root.join(TOOLKIT_DRIVER_REL).join(CAPI_DLL));
    }

    for library in steam_library_roots() {
        let common = library.join("steamapps").join("common");
        push(
            common
                .join(PSVR2_APP_DIR)
                .join(TOOLKIT_DRIVER_REL)
                .join(CAPI_DLL),
        );
        for install_dir in psvr2_manifest_install_dirs(&library) {
            push(
                common
                    .join(install_dir)
                    .join(TOOLKIT_DRIVER_REL)
                    .join(CAPI_DLL),
            );
        }
    }
    out
}

fn capi_path_from_hint(path: PathBuf) -> PathBuf {
    if path
        .file_name()
        .is_some_and(|name| name.to_string_lossy().eq_ignore_ascii_case(CAPI_DLL))
    {
        path
    } else {
        path.join(CAPI_DLL)
    }
}

fn openvr_registered_paths() -> Vec<PathBuf> {
    let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) else {
        return Vec::new();
    };
    let path = local.join("openvr").join("openvrpaths.vrpath");
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    quoted_values(&text)
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .collect()
}

fn steam_library_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    for root in steam_registry_roots() {
        push_existing_path(&mut roots, &mut seen, root);
    }
    for variable in ["ProgramFiles(x86)", "ProgramFiles"] {
        if let Some(base) = std::env::var_os(variable) {
            push_existing_path(&mut roots, &mut seen, PathBuf::from(base).join("Steam"));
        }
    }

    // Registry discovery normally finds Steam. These bounded fallbacks also cover
    // portable installs and machines whose registry entry was removed.
    for letter in b'C'..=b'Z' {
        let drive = format!("{}:\\", letter as char);
        for relative in ["SteamLibrary", "Steam", r"Program Files (x86)\Steam"] {
            push_existing_path(&mut roots, &mut seen, PathBuf::from(&drive).join(relative));
        }
    }

    let primary = roots.clone();
    for root in primary {
        let vdf = root.join("steamapps").join("libraryfolders.vdf");
        let Ok(text) = std::fs::read_to_string(vdf) else {
            continue;
        };
        for path in steam_library_paths_from_vdf(&text) {
            push_existing_path(&mut roots, &mut seen, path);
        }
    }
    roots
}

fn push_existing_path(roots: &mut Vec<PathBuf>, seen: &mut HashSet<String>, path: PathBuf) {
    let key = path.to_string_lossy().to_ascii_lowercase();
    if path.is_dir() && seen.insert(key) {
        roots.push(path);
    }
}

fn steam_registry_roots() -> Vec<PathBuf> {
    let reg = ["SystemRoot", "WINDIR"]
        .into_iter()
        .filter_map(std::env::var_os)
        .map(PathBuf::from)
        .find(|path| path.is_absolute())
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
        .join("reg.exe");
    let queries = [
        (r"HKCU\SOFTWARE\Valve\Steam", "SteamPath"),
        (r"HKLM\SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
        (r"HKLM\SOFTWARE\Valve\Steam", "InstallPath"),
    ];
    let mut paths = Vec::new();
    for (key, value) in queries {
        let Ok(output) = Command::new(&reg)
            .args(["query", key, "/v", value])
            .output()
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let Some((_, path)) = line.split_once("REG_SZ") else {
                continue;
            };
            let path = path.trim().trim_matches('"');
            if !path.is_empty() {
                paths.push(PathBuf::from(path));
            }
        }
    }
    paths
}

fn steam_library_paths_from_vdf(text: &str) -> Vec<PathBuf> {
    let values = quoted_values(text);
    let mut paths = Vec::new();
    for pair in values.windows(2) {
        let key = &pair[0];
        let value = &pair[1];
        let modern = key.eq_ignore_ascii_case("path");
        let legacy = key.chars().all(|ch| ch.is_ascii_digit()) && Path::new(value).is_absolute();
        if modern || legacy {
            paths.push(PathBuf::from(value));
        }
    }
    paths
}

fn psvr2_manifest_install_dirs(library: &Path) -> Vec<String> {
    let steamapps = library.join("steamapps");
    let Ok(entries) = std::fs::read_dir(steamapps) else {
        return Vec::new();
    };
    let mut installs = Vec::new();
    for entry in entries.flatten().take(4_096) {
        let file_name = entry.file_name().to_string_lossy().to_ascii_lowercase();
        if !file_name.starts_with("appmanifest_") || !file_name.ends_with(".acf") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let values = quoted_values(&text);
        let mut name = None;
        let mut install_dir = None;
        for pair in values.windows(2) {
            if pair[0].eq_ignore_ascii_case("name") {
                name = Some(pair[1].as_str());
            } else if pair[0].eq_ignore_ascii_case("installdir") {
                install_dir = Some(pair[1].as_str());
            }
        }
        let is_psvr2 = name.is_some_and(|value| {
            let compact: String = value
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect();
            compact.contains("playstation") && compact.contains("vr2")
        });
        if is_psvr2 {
            if let Some(install_dir) = install_dir.filter(|value| !value.is_empty()) {
                installs.push(install_dir.to_owned());
            }
        }
    }
    installs
}

fn quoted_values(text: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch != '"' {
            continue;
        }
        let mut value = String::new();
        let mut escaped = false;
        for ch in chars.by_ref() {
            if escaped {
                match ch {
                    '\\' | '"' => value.push(ch),
                    _ => {
                        value.push('\\');
                        value.push(ch);
                    }
                }
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                break;
            } else {
                value.push(ch);
            }
        }
        values.push(value);
    }
    values
}

unsafe fn required_proc<T: Copy>(module: HMODULE, name: &'static [u8]) -> io::Result<T> {
    optional_proc(module, name).ok_or_else(|| {
        let printable = CStr::from_bytes_with_nul(name)
            .map(CStr::to_string_lossy)
            .unwrap_or_else(|_| "<invalid>".into());
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing CAPI export {printable}"),
        )
    })
}

unsafe fn optional_proc<T: Copy>(module: HMODULE, name: &'static [u8]) -> Option<T> {
    let address = GetProcAddress(module, name.as_ptr());
    address.map(|proc| {
        let raw = proc as *const c_void;
        std::mem::transmute_copy::<*const c_void, T>(&raw)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_layout_matches_current_toolkit_header() {
        assert_eq!(std::mem::size_of::<WearableEye>(), 72);
        assert_eq!(std::mem::size_of::<WearableData>(), 192);
        assert_eq!(std::mem::size_of::<FoveatedGaze>(), 56);
        assert_eq!(std::mem::size_of::<GazeStatus>(), 0x148);
    }

    #[test]
    fn image_copy_validates_header_and_splits_without_labeling_eyes() {
        let mut bytes = vec![0u8; IMAGE_HEADER_BYTES + IMAGE_PIXELS];
        bytes[0..2].copy_from_slice(b"VI");
        bytes[2..4].copy_from_slice(&3u16.to_le_bytes());
        let len = bytes.len() as u32;
        bytes[4..8].copy_from_slice(&len.to_le_bytes());
        bytes[8..12].copy_from_slice(&1234u32.to_le_bytes());
        bytes[16..18].copy_from_slice(&6u16.to_le_bytes());
        for y in 0..IMAGE_HEIGHT {
            bytes[IMAGE_HEADER_BYTES + y * IMAGE_WIDTH
                ..IMAGE_HEADER_BYTES + y * IMAGE_WIDTH + CAMERA_WIDTH]
                .fill(y as u8);
            bytes[IMAGE_HEADER_BYTES + y * IMAGE_WIDTH + CAMERA_WIDTH
                ..IMAGE_HEADER_BYTES + (y + 1) * IMAGE_WIDTH]
                .fill(255 - y as u8);
        }

        let frame = unsafe { EyeImageFrame::copy_from_capi(bytes.as_ptr()) }.unwrap();
        assert_eq!(frame.version, 3);
        assert_eq!(frame.timestamp, 1234);
        assert_eq!(frame.image_type, 6);
        let [a, b] = frame.split_camera_planes();
        assert_eq!(a[0], 0);
        assert_eq!(a[CAMERA_WIDTH * 17], 17);
        assert_eq!(b[0], 255);
        assert_eq!(b[CAMERA_WIDTH * 17], 238);
    }

    #[test]
    fn rejects_wrong_image_magic_before_copying_pixels() {
        let bytes = vec![0u8; IMAGE_HEADER_BYTES + IMAGE_PIXELS];
        let error = unsafe { EyeImageFrame::copy_from_capi(bytes.as_ptr()) }.unwrap_err();
        assert!(error.to_string().contains("magic"));
    }

    #[test]
    fn capi_hint_accepts_a_directory_or_full_dll_path() {
        assert_eq!(
            capi_path_from_hint(PathBuf::from(r"E:\SteamLibrary\driver")),
            PathBuf::from(r"E:\SteamLibrary\driver\psvr2_toolkit_capi.dll")
        );
        assert_eq!(
            capi_path_from_hint(PathBuf::from(
                r"E:\SteamLibrary\driver\PSVR2_TOOLKIT_CAPI.DLL"
            )),
            PathBuf::from(r"E:\SteamLibrary\driver\PSVR2_TOOLKIT_CAPI.DLL")
        );
    }

    #[test]
    fn steam_vdf_parser_finds_modern_and_legacy_library_paths() {
        let text = r#"
            "libraryfolders"
            {
                "1" { "path" "D:\\VR Games\\SteamLibrary" }
                "2" "E:\\PortableSteam"
            }
        "#;
        assert_eq!(
            steam_library_paths_from_vdf(text),
            vec![
                PathBuf::from(r"D:\VR Games\SteamLibrary"),
                PathBuf::from(r"E:\PortableSteam")
            ]
        );
    }

    #[test]
    fn quoted_value_parser_preserves_windows_paths() {
        assert_eq!(
            quoted_values(r#""path" "F:\\SteamLibrary""#),
            vec!["path".to_owned(), r"F:\SteamLibrary".to_owned()]
        );
    }
}
