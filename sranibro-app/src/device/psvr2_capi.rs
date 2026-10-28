//! Read-only PlayStation VR2 Toolkit CAPI bridge.
//!
//! The Toolkit publishes gaze state and the paired eye-camera preview through its
//! installed `psvr2_toolkit_capi.dll` (Windows) or native Linux `.so`. SRanibro discovers it from the
//! Toolkit's temporary path hint, active OpenVR drivers, and every registered
//! Steam library. This keeps the application self-contained without copying or
//! redistributing Toolkit binaries.
//!
//! This low-level module deliberately exposes raw camera halves as `A` and `B`.
//! The product adapter applies the anatomical mapping established by a labelled
//! hardware recording; the standalone probe retains neutral A/B names so future
//! Toolkit or headset revisions can be audited without assuming that mapping.

#![cfg(any(windows, target_os = "linux"))]

use std::collections::HashSet;
#[cfg(windows)]
use std::ffi::{c_void, CStr};
use std::io;
#[cfg(target_os = "linux")]
use std::io::Read;
#[cfg(target_os = "linux")]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
#[cfg(windows)]
use std::process::Command;
use std::ptr;

#[cfg(windows)]
use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
#[cfg(windows)]
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

#[cfg(windows)]
const CAPI_LIBRARIES: &[&str] = &["psvr2_toolkit_capi.dll"];
// Upstream CMake/loader use the lib prefix; some Ignition packages omit it.
#[cfg(target_os = "linux")]
const CAPI_LIBRARIES: &[&str] = &["libpsvr2_toolkit_capi.so", "psvr2_toolkit_capi.so"];
const PSVR2_APP_DIR: &str = "PlayStation VR2 App";

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
// The exported functions return C++ bool (one byte on both supported x86_64
// ABIs), unlike the uint32_t enum flags embedded inside GazeStatus.
type DriverActiveFn = unsafe extern "C" fn() -> u8;
type GazeStatusFn = unsafe extern "C" fn(*mut GazeStatus, u32) -> u8;
type GazeImageFn = unsafe extern "C" fn(*mut *mut u8, u32) -> u8;

/// Dynamically loaded, read-only subset of PSVR2Toolkit CAPI.
pub struct Psvr2Capi {
    #[cfg(windows)]
    module: HMODULE,
    // Fields drop in declaration order, after our Drop runs deinit: unload CAPI
    // before releasing its explicitly preloaded IPC dependency.
    #[cfg(target_os = "linux")]
    _module: libloading::os::unix::Library,
    #[cfg(target_os = "linux")]
    _crossipc: Option<libloading::os::unix::Library>,
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
        #[cfg(windows)]
        let candidates = capi_candidates();
        // An explicit Linux override is authoritative and intentionally trusted
        // by the user. Never silently fall back to a different automatic library
        // if the chosen file is absent or fails to load.
        #[cfg(target_os = "linux")]
        let explicit_override = std::env::var_os("PSVR2_TOOLKIT_CAPI");
        #[cfg(target_os = "linux")]
        let candidates = explicit_override
            .as_ref()
            .map_or_else(capi_candidates, |value| {
                capi_paths_from_hint(PathBuf::from(value))
            });
        let mut load_errors = Vec::new();
        for path in &candidates {
            if !path.is_file() {
                continue;
            }
            #[cfg(windows)]
            let result = unsafe { Self::load_path(path) };
            #[cfg(target_os = "linux")]
            let result = unsafe { Self::load_path(path, explicit_override.is_none()) };
            match result {
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
        if !load_errors.is_empty() {
            let details = load_errors
                .iter()
                .map(|(path, error)| format!("{}: {error}", path.display()))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "PSVR2Toolkit CAPI was found but could not be loaded: {details}. {}",
                    capi_install_hint()
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
        #[cfg(windows)]
        let search_detail = "SRanibro checked every detected Steam library and OpenVR driver.";
        #[cfg(target_os = "linux")]
        let search_detail = if explicit_override.is_some() {
            "Only the explicit PSVR2_TOOLKIT_CAPI override was checked; unset it to restore automatic discovery."
        } else {
            "SRanibro checked every detected Steam library and OpenVR driver."
        };
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("PSVR2Toolkit CAPI was not found. {} {search_detail} PSVR2_TOOLKIT_CAPI can specify the library or its directory.", capi_install_hint()),
        ))
    }

    #[cfg(windows)]
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

    #[cfg(target_os = "linux")]
    unsafe fn load_path(path: &Path, automatic: bool) -> io::Result<Self> {
        use libloading::os::unix::{Library, RTLD_LOCAL, RTLD_NOW};

        // An absolute path avoids dlopen's bare-filename search and makes a
        // relative PSVR2_TOOLKIT_CAPI override behave predictably.
        let path = if automatic {
            trusted_linux_library_path(path)?
        } else {
            std::fs::canonicalize(path)?
        };
        let dependency = path.parent().unwrap().join("libcrossipc.so");
        // Automatic discovery requires the known private dependency to be safe
        // as well. Load its validated canonical target, not the original symlink
        // path, so another user cannot replace a link between validation/dlopen.
        let dependency = if automatic {
            trusted_linux_library_path(&dependency).map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!(
                        "cannot trust Toolkit IPC dependency {}: {error}",
                        dependency.display()
                    ),
                )
            })?
        } else {
            dependency
        };
        let crossipc = if dependency.is_file() {
            Some(
                Library::open(Some(&dependency), RTLD_NOW | RTLD_LOCAL).map_err(|error| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cannot load Toolkit IPC dependency {}: {error}",
                            dependency.display()
                        ),
                    )
                })?,
            )
        } else {
            // Official builds have $ORIGIN RUNPATH. Let the linker report its
            // precise missing dependency/architecture/version error if absent.
            None
        };
        let module = Library::open(Some(&path), RTLD_NOW | RTLD_LOCAL).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                "dlopen {} failed: {error}; keep matching libcrossipc.so beside the native CAPI",
                path.display()
            ),
            )
        })?;
        let init = required_proc::<InitFn>(&module, b"psvr2_toolkit_init\0")?;
        let deinit = required_proc::<DeinitFn>(&module, b"psvr2_toolkit_deinit\0")?;
        let gaze_status = required_proc::<GazeStatusFn>(&module, b"psvr2_toolkit_gaze_status\0")?;
        let gaze_image = required_proc::<GazeImageFn>(&module, b"psvr2_toolkit_gaze_image\0")?;
        let get_driver_active = module
            .get::<DriverActiveFn>(b"psvr2_toolkit_get_driver_active\0")
            .ok()
            .map(|symbol| *symbol);
        Ok(Self {
            _module: module,
            _crossipc: crossipc,
            module_path: path,
            init,
            deinit,
            get_driver_active,
            gaze_status,
            gaze_image,
            initialized: false,
        })
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
            #[cfg(windows)]
            if !self.module.is_null() {
                FreeLibrary(self.module);
                self.module = ptr::null_mut();
            }
        }
    }
}

fn capi_install_hint() -> &'static str {
    #[cfg(windows)]
    {
        "Start SteamVR with PSVR2Toolkit installed; keep its matching dependency DLLs beside the CAPI."
    }
    #[cfg(target_os = "linux")]
    {
        "Install the Linux/Ignition PSVR2Toolkit release and start SteamVR through Ignition. Keep the native .so and matching libcrossipc.so together (normally SteamVR_Plug-In/bin/win64); Windows DLLs cannot be loaded by this native Linux app."
    }
}

fn capi_candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut push = |path: PathBuf| {
        if !path.as_os_str().is_empty() && seen.insert(path_key(&path)) {
            out.push(path);
        }
    };

    #[cfg(windows)]
    if let Some(value) = std::env::var_os("PSVR2_TOOLKIT_CAPI") {
        for path in capi_paths_from_hint(PathBuf::from(value)) {
            push(path);
        }
    }

    let mut temp_dirs = vec![std::env::temp_dir()];
    // Ignition's Windows driver writes the hint into /tmp even if this process
    // was launched with TMPDIR set to another location.
    #[cfg(target_os = "linux")]
    temp_dirs.push(PathBuf::from("/tmp"));
    for temp in temp_dirs.drain(..) {
        let hint_file = temp.join("psvr2tk_capi_path.txt");
        #[cfg(windows)]
        let text = std::fs::read_to_string(&hint_file);
        #[cfg(target_os = "linux")]
        let text = read_trusted_linux_hint(&hint_file);
        #[cfg(target_os = "linux")]
        if let Err(error) = &text {
            if error.kind() != io::ErrorKind::NotFound {
                eprintln!(
                    "[psvr2] ignoring unsafe/unreadable CAPI hint {}: {error}",
                    hint_file.display()
                );
            }
        }
        if let Ok(text) = text {
            if let Some(line) = text.lines().next().map(str::trim).filter(|s| !s.is_empty()) {
                for path in capi_paths_from_hint(PathBuf::from(line)) {
                    push(path);
                }
            }
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for path in capi_paths_from_hint(parent.to_path_buf()) {
                push(path);
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        for path in capi_paths_from_hint(cwd) {
            push(path);
        }
    }

    for root in openvr_registered_paths() {
        for path in driver_capi_candidates(&root) {
            push(path);
        }
    }

    for library in steam_library_roots() {
        let common = library.join("steamapps").join("common");
        for path in driver_capi_candidates(&common.join(PSVR2_APP_DIR).join("SteamVR_Plug-In")) {
            push(path);
        }
        for install_dir in psvr2_manifest_install_dirs(&library) {
            for path in driver_capi_candidates(&common.join(install_dir).join("SteamVR_Plug-In")) {
                push(path);
            }
        }
    }
    out
}

#[cfg(target_os = "linux")]
fn trusted_linux_owner(owner: u32, current_uid: u32) -> bool {
    owner == current_uid || owner == 0
}

#[cfg(target_os = "linux")]
fn check_linux_file_metadata(metadata: &std::fs::Metadata, current_uid: u32) -> io::Result<()> {
    if !metadata.is_file()
        || !trusted_linux_owner(metadata.uid(), current_uid)
        || metadata.mode() & 0o022 != 0
    {
        return Err(io::Error::new(io::ErrorKind::PermissionDenied,
            "automatic PSVR2 discovery requires a regular file owned by the current user or root, without group/world write permission"));
    }
    Ok(())
}

/// Upstream's Wine writer uses std::ofstream in /tmp (normally mode 0644 under
/// umask 022). Open without following the final symlink and inspect this exact
/// descriptor, not a separate pathname stat. NONBLOCK avoids hanging on a FIFO
/// maliciously placed at the well-known global hint path.
#[cfg(target_os = "linux")]
fn read_trusted_linux_hint(path: &Path) -> io::Result<String> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let uid = unsafe { libc::geteuid() };
    check_linux_file_metadata(&file.metadata()?, uid)?;
    let mut text = String::new();
    (&file).take(8193).read_to_string(&mut text)?;
    check_linux_file_metadata(&file.metadata()?, uid)?;
    if text.len() > 8192 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PSVR2 CAPI path hint is too long",
        ));
    }
    let hint = text.lines().next().unwrap_or_default().trim();
    if !Path::new(hint).is_absolute() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "PSVR2 CAPI path hint must be absolute",
        ));
    }
    Ok(text)
}

/// Canonicalize first and only pass this validated target to dlopen. Inspect
/// regular-file metadata from an opened descriptor, and reject directories that
/// let another user replace the file or inject $ORIGIN dependencies. Root-owned
/// sticky ancestors such as /tmp are fine, but the immediate library directory
/// must not be group/world writable. A user deliberately using a shared Steam
/// library can opt in with PSVR2_TOOLKIT_CAPI instead of automatic discovery.
#[cfg(target_os = "linux")]
fn trusted_linux_library_path(path: &Path) -> io::Result<PathBuf> {
    let canonical = std::fs::canonicalize(path)?;
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(&canonical)?;
    let uid = unsafe { libc::geteuid() };
    check_linux_file_metadata(&file.metadata()?, uid)?;
    for (index, directory) in canonical.ancestors().skip(1).enumerate() {
        let metadata = std::fs::symlink_metadata(directory)?;
        let writable = metadata.mode() & 0o022 != 0;
        let sticky_ancestor = index > 0 && metadata.mode() & 0o1000 != 0;
        if !metadata.is_dir()
            || !trusted_linux_owner(metadata.uid(), uid)
            || (writable && !sticky_ancestor)
        {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, format!(
                "unsafe automatic PSVR2 library directory {} (ownership/write permissions); use PSVR2_TOOLKIT_CAPI only if you intentionally trust this installation",
                directory.display()
            )));
        }
    }
    Ok(canonical)
}

fn capi_paths_from_hint(path: PathBuf) -> Vec<PathBuf> {
    let filename = path.file_name().map(|name| name.to_string_lossy());
    let is_library = filename.as_ref().is_some_and(|name| {
        #[cfg(windows)]
        {
            name.eq_ignore_ascii_case(CAPI_LIBRARIES[0])
        }
        #[cfg(target_os = "linux")]
        {
            name.ends_with(".so") || name.contains(".so.")
        }
    });
    if is_library || path.is_file() {
        vec![path]
    } else {
        CAPI_LIBRARIES.iter().map(|name| path.join(name)).collect()
    }
}

fn driver_capi_candidates(root: &Path) -> Vec<PathBuf> {
    let mut directories = vec![root.to_path_buf()];
    #[cfg(target_os = "linux")]
    directories.push(root.join("bin").join("linux64"));
    // Ignition's Linux libraries are distributed beside the Windows driver.
    directories.push(root.join("bin").join("win64"));
    #[cfg(target_os = "linux")]
    directories.push(root.join("SteamVR_Plug-In").join("bin").join("linux64"));
    directories.push(root.join("SteamVR_Plug-In").join("bin").join("win64"));
    directories
        .into_iter()
        .flat_map(capi_paths_from_hint)
        .collect()
}

fn path_key(path: &Path) -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from(path.to_string_lossy().to_ascii_lowercase())
    }
    #[cfg(target_os = "linux")]
    {
        path.to_path_buf()
    }
}

fn openvr_registered_paths() -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Some(path) = std::env::var_os("VR_PATHREG_OVERRIDE") {
        files.push(PathBuf::from(path));
    }
    #[cfg(windows)]
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        files.push(
            PathBuf::from(local)
                .join("openvr")
                .join("openvrpaths.vrpath"),
        );
    }
    #[cfg(target_os = "linux")]
    {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let config = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from);
        files.extend(linux_openvr_files(home.as_deref(), config.as_deref()));
    }
    files
        .into_iter()
        .filter_map(|path| std::fs::read_to_string(path).ok())
        .flat_map(|text| quoted_values(&text))
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .collect()
}

fn steam_library_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut seen = HashSet::new();

    #[cfg(windows)]
    {
        for root in steam_registry_roots() {
            push_existing_path(&mut roots, &mut seen, root);
        }
        for variable in ["ProgramFiles(x86)", "ProgramFiles"] {
            if let Some(base) = std::env::var_os(variable) {
                push_existing_path(&mut roots, &mut seen, PathBuf::from(base).join("Steam"));
            }
        }
        // Registry discovery normally finds Steam. These bounded fallbacks also
        // cover portable installs and machines whose registry entry was removed.
        for letter in b'C'..=b'Z' {
            let drive = format!("{}:\\", letter as char);
            for relative in ["SteamLibrary", "Steam", r"Program Files (x86)\Steam"] {
                push_existing_path(&mut roots, &mut seen, PathBuf::from(&drive).join(relative));
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        for variable in ["STEAM_DIR", "STEAM_PATH"] {
            if let Some(path) = std::env::var_os(variable) {
                push_existing_path(&mut roots, &mut seen, PathBuf::from(path));
            }
        }
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let data = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from);
        for root in linux_steam_roots(home.as_deref(), data.as_deref()) {
            push_existing_path(&mut roots, &mut seen, root);
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

fn push_existing_path(roots: &mut Vec<PathBuf>, seen: &mut HashSet<PathBuf>, path: PathBuf) {
    let key = path_key(&path);
    if path.is_dir() && seen.insert(key) {
        roots.push(path);
    }
}

#[cfg(target_os = "linux")]
fn linux_openvr_files(home: Option<&Path>, config: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(config) = config.filter(|path| path.is_absolute()) {
        roots.push(config.to_path_buf());
    }
    if let Some(home) = home.filter(|path| path.is_absolute()) {
        roots.push(home.join(".config"));
    }
    roots
        .into_iter()
        .map(|root| root.join("openvr").join("openvrpaths.vrpath"))
        .collect()
}

#[cfg(target_os = "linux")]
fn linux_steam_roots(home: Option<&Path>, data: Option<&Path>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(data) = data.filter(|path| path.is_absolute()) {
        roots.push(data.join("Steam"));
    }
    if let Some(home) = home.filter(|path| path.is_absolute()) {
        for relative in [
            ".local/share/Steam",
            ".steam/steam",
            ".steam/root",
            ".steam/debian-installation",
        ] {
            roots.push(home.join(relative));
        }
    }
    roots
}

#[cfg(windows)]
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
        if (modern || legacy) && Path::new(value).is_absolute() {
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

#[cfg(windows)]
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

#[cfg(windows)]
unsafe fn optional_proc<T: Copy>(module: HMODULE, name: &'static [u8]) -> Option<T> {
    let address = GetProcAddress(module, name.as_ptr());
    address.map(|proc| {
        let raw = proc as *const c_void;
        std::mem::transmute_copy::<*const c_void, T>(&raw)
    })
}

#[cfg(target_os = "linux")]
unsafe fn required_proc<T: Copy>(
    module: &libloading::os::unix::Library,
    name: &'static [u8],
) -> io::Result<T> {
    module
        .get::<T>(name)
        .map(|symbol| *symbol)
        .map_err(|error| {
            let printable = String::from_utf8_lossy(&name[..name.len().saturating_sub(1)]);
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing CAPI export {printable}: {error}"),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    struct SecurityTestDirectory(PathBuf);

    #[cfg(target_os = "linux")]
    impl SecurityTestDirectory {
        fn new() -> Self {
            use std::os::unix::fs::DirBuilderExt;
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let directory = PathBuf::from("/tmp").join(format!(
                "sranibro-capi-security-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::DirBuilder::new()
                .mode(0o700)
                .create(&directory)
                .unwrap();
            Self(directory)
        }

        fn file(&self, name: &str, bytes: &[u8], mode: u32) -> PathBuf {
            use std::io::Write;
            use std::os::unix::fs::PermissionsExt;
            let path = self.0.join(name);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(mode)
                .open(&path)
                .unwrap();
            file.write_all(bytes).unwrap();
            // Set exact test permissions independently of the runner's umask.
            file.set_permissions(std::fs::Permissions::from_mode(mode))
                .unwrap();
            path
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for SecurityTestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_hint_accepts_owner_only_and_normal_ignition_permissions() {
        let directory = SecurityTestDirectory::new();
        for (name, mode) in [("owner-only", 0o600), ("ignition-default", 0o644)] {
            let path = directory.file(name, b"/home/test/Steam/PSVR2/bin/win64\n", mode);
            assert_eq!(
                read_trusted_linux_hint(&path).unwrap(),
                "/home/test/Steam/PSVR2/bin/win64\n"
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_hint_rejects_group_or_world_writable_files() {
        let directory = SecurityTestDirectory::new();
        for (name, mode) in [("group-writable", 0o664), ("world-writable", 0o646)] {
            let path = directory.file(name, b"/untrusted/driver\n", mode);
            assert_eq!(
                read_trusted_linux_hint(&path).unwrap_err().kind(),
                io::ErrorKind::PermissionDenied
            );
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_hint_rejects_symlinks_nonfiles_relative_and_oversized_paths() {
        let directory = SecurityTestDirectory::new();
        let target = directory.file("target", b"/home/test/driver\n", 0o600);
        let link = directory.0.join("linked-hint");
        std::os::unix::fs::symlink(target, &link).unwrap();
        assert!(read_trusted_linux_hint(&link).is_err());
        assert!(read_trusted_linux_hint(&directory.0).is_err());
        let relative = directory.file("relative", b"../driver\n", 0o600);
        assert_eq!(
            read_trusted_linux_hint(&relative).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let oversized = directory.file("oversized", &vec![b'/'; 8193], 0o600);
        assert_eq!(
            read_trusted_linux_hint(&oversized).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_owner_policy_rejects_other_users_without_changing_process_uid() {
        assert!(trusted_linux_owner(42, 42));
        assert!(trusted_linux_owner(0, 42));
        assert!(!trusted_linux_owner(43, 42));
        assert!(!trusted_linux_owner(42, 0));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_automatic_library_checks_file_and_parent_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let directory = SecurityTestDirectory::new();
        let library = directory.file("libpsvr2_toolkit_capi.so", b"fixture", 0o644);
        assert_eq!(trusted_linux_library_path(&library).unwrap(), library);
        let writable = directory.file("writable.so", b"fixture", 0o666);
        assert_eq!(
            trusted_linux_library_path(&writable).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        let alias = directory.0.join("versioned.so");
        std::os::unix::fs::symlink(&library, &alias).unwrap();
        assert_eq!(trusted_linux_library_path(&alias).unwrap(), library);
        std::fs::set_permissions(&directory.0, std::fs::Permissions::from_mode(0o777)).unwrap();
        assert_eq!(
            trusted_linux_library_path(&library).unwrap_err().kind(),
            io::ErrorKind::PermissionDenied
        );
        // A sticky immediate directory still permits unrelated users to plant
        // dependencies, unlike a sticky ancestor protecting our owned subdir.
        std::fs::set_permissions(&directory.0, std::fs::Permissions::from_mode(0o1777)).unwrap();
        assert!(trusted_linux_library_path(&library).is_err());
        std::fs::set_permissions(&directory.0, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn ffi_layout_matches_current_toolkit_header() {
        assert_eq!(std::mem::size_of::<WearableEye>(), 72);
        assert_eq!(std::mem::size_of::<WearableData>(), 192);
        assert_eq!(std::mem::size_of::<FoveatedGaze>(), 56);
        assert_eq!(std::mem::size_of::<GazeStatus>(), 0x148);
        assert_eq!(std::mem::align_of::<GazeStatus>(), 8);
        assert_eq!(std::mem::offset_of!(GazeStatus, wearable), 80);
        assert_eq!(std::mem::offset_of!(GazeStatus, foveated), 272);
        assert_eq!(std::mem::offset_of!(WearableData, left), 12);
        assert_eq!(std::mem::offset_of!(WearableData, right), 84);
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
    #[cfg(windows)]
    fn capi_hint_accepts_a_directory_or_full_dll_path() {
        assert_eq!(
            capi_paths_from_hint(PathBuf::from(r"E:\SteamLibrary\driver")),
            vec![PathBuf::from(
                r"E:\SteamLibrary\driver\psvr2_toolkit_capi.dll"
            )]
        );
        assert_eq!(
            capi_paths_from_hint(PathBuf::from(
                r"E:\SteamLibrary\driver\PSVR2_TOOLKIT_CAPI.DLL"
            )),
            vec![PathBuf::from(
                r"E:\SteamLibrary\driver\PSVR2_TOOLKIT_CAPI.DLL"
            )]
        );
    }

    #[test]
    #[cfg(windows)]
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
    #[cfg(target_os = "linux")]
    fn linux_capi_hints_accept_both_package_names_and_explicit_versioned_libraries() {
        let directory = PathBuf::from("/home/user/VR driver");
        assert_eq!(
            capi_paths_from_hint(directory.clone()),
            vec![
                directory.join("libpsvr2_toolkit_capi.so"),
                directory.join("psvr2_toolkit_capi.so")
            ]
        );
        for name in [
            "libpsvr2_toolkit_capi.so",
            "psvr2_toolkit_capi.so",
            "custom-capi.so.1",
        ] {
            let library = directory.join(name);
            assert_eq!(capi_paths_from_hint(library.clone()), vec![library]);
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_driver_discovery_includes_ignition_win64_and_native_linux64() {
        let root = Path::new("/media/VR Games/PlayStation VR2 App/SteamVR_Plug-In");
        let candidates = driver_capi_candidates(root);
        for directory in ["bin/win64", "bin/linux64"] {
            for name in CAPI_LIBRARIES {
                assert!(candidates.contains(&root.join(directory).join(name)));
            }
        }
        assert_ne!(path_key(Path::new("/Steam")), path_key(Path::new("/steam")));
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_steam_and_openvr_discovery_respect_xdg_and_keep_standard_fallbacks() {
        let home = Path::new("/home/user");
        let roots = linux_steam_roots(Some(home), Some(Path::new("/custom/data")));
        assert_eq!(roots[0], PathBuf::from("/custom/data/Steam"));
        assert!(roots.contains(&home.join(".local/share/Steam")));
        assert!(roots.contains(&home.join(".steam/steam")));
        assert!(roots.contains(&home.join(".steam/root")));
        assert_eq!(
            linux_openvr_files(Some(home), Some(Path::new("/custom/config"))),
            vec![
                PathBuf::from("/custom/config/openvr/openvrpaths.vrpath"),
                PathBuf::from("/home/user/.config/openvr/openvrpaths.vrpath")
            ]
        );
        assert!(linux_steam_roots(None, Some(Path::new("relative"))).is_empty());
        assert!(linux_openvr_files(None, Some(Path::new("relative"))).is_empty());
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_steam_vdf_preserves_case_spaces_and_external_libraries() {
        assert_eq!(
            steam_library_paths_from_vdf(
                r#"
            "libraryfolders" {
                "0" { "path" "/home/user/.local/share/Steam" }
                "1" { "path" "/media/VR Games/SteamLibrary" }
                "2" "/mnt/LegacySteam"
                "3" { "path" "relative/untrusted" }
            }
        "#
            ),
            vec![
                PathBuf::from("/home/user/.local/share/Steam"),
                PathBuf::from("/media/VR Games/SteamLibrary"),
                PathBuf::from("/mnt/LegacySteam")
            ]
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_loader_reports_native_linker_errors() {
        let directory = SecurityTestDirectory::new();
        let temp = directory.file("invalid.so", b"not an ELF shared library", 0o600);
        let result = unsafe { Psvr2Capi::load_path(&temp, false) };
        let error = result.err().expect("an invalid ELF file must fail");
        assert!(error.to_string().contains("dlopen"));
        assert!(error.to_string().contains("libcrossipc.so"));
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn linux_missing_symbol_error_names_the_required_export() {
        let module = libloading::os::unix::Library::this();
        let error =
            unsafe { required_proc::<InitFn>(&module, b"sranibro_test_missing_capi_export\0") }
                .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(error
            .to_string()
            .contains("missing CAPI export sranibro_test_missing_capi_export"));
    }

    /// Explicit opt-in integration test: compile tools/linux/mock_psvr2_capi.c,
    /// set SRANIBRO_PSVR2_TEST_CAPI to the resulting file and
    /// SRANIBRO_PSVR2_MOCK=1, then run this test with --ignored.
    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "requires explicitly built test-only CAPI and SRANIBRO_PSVR2_MOCK=1"]
    fn linux_mock_capi_ffi_streams_and_releases_session() {
        let path = std::env::var_os("SRANIBRO_PSVR2_TEST_CAPI")
            .map(PathBuf::from)
            .expect("set SRANIBRO_PSVR2_TEST_CAPI to the test-only mock .so");
        let mut capi = unsafe { Psvr2Capi::load_path(&path, false) }.unwrap();
        assert_eq!(capi.initialize(), RESULT_OK);
        assert_eq!(capi.driver_active(), Some(true));
        let gaze = capi.next_gaze(0).unwrap().expect("mock gaze frame");
        assert_eq!(gaze.wearable.frame_counter, 1);
        assert!(gaze.wearable.left.gaze_valid());
        assert_eq!(gaze.wearable.left.gaze_origin_mm.x, -32.0);
        assert_eq!(gaze.wearable.right.gaze_origin_mm.x, 32.0);
        assert_eq!(gaze.wearable.left.pupil_dia_mm, 4.0);
        let image = capi.next_image(0).unwrap().expect("mock eye image");
        assert_eq!(image.pixels.len(), IMAGE_PIXELS);
        let [left, right] = image.split_camera_planes();
        assert_eq!(left, right);
        assert!(left.iter().any(|pixel| *pixel == 15));
        // Keep a second library handle so the shared mock state remains alive
        // while verifying that our Drop called deinit before unloading.
        let observer = unsafe {
            libloading::os::unix::Library::open(Some(&path), libloading::os::unix::RTLD_NOW)
        }
        .unwrap();
        let active =
            unsafe { observer.get::<DriverActiveFn>(b"psvr2_toolkit_get_driver_active\0") }
                .unwrap();
        drop(capi);
        assert_eq!(unsafe { active() }, 0);
    }

    #[test]
    fn quoted_value_parser_preserves_windows_paths() {
        assert_eq!(
            quoted_values(r#""path" "F:\\SteamLibrary""#),
            vec!["path".to_owned(), r"F:\SteamLibrary".to_owned()]
        );
    }
}
