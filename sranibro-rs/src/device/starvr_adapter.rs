//! StarVR One adapter — Tobii Stream Engine over the patched DLL.
//!
//! StarVR One uses the SAME Tobii IS4 EyeChip as the Pimax VR4/Crystal (200x200
//! @120Hz stereo IR, `slot A = LEFT`), but a DIFFERENT transport: instead of VR4's
//! DLL-free WinUSB+TTP path, StarVR is driven through the user-supplied patched
//! stream-engine DLL (`[assets].starvr_dll`, e.g. ReStar's
//! `tobii_stream_engine_full_unlock.dll`) via the standard Tobii C API.
//!
//! This is a FAITHFUL Rust port of the working Python reference
//! (`starvr_eye_viewer.py`): `tobii_api_create` -> `enumerate_local_device_urls`
//! -> `tobii_device_create` -> `tobii_image_subscribe` + `tobii_wearable_data_subscribe`
//! -> pump `tobii_device_process_callbacks`. Struct layouts verified against
//! ReStar's vendored `tobii_wearable.h` (compile-time size asserts below).
//!
//! UNVERIFIED ON RUST+HARDWARE: the Python path is known-good and this port matches
//! it 1:1, but run it once on a StarVR One to confirm before trusting. We bundle
//! nothing — the DLL is the user's, referenced by path.

#![cfg(windows)]

use std::ffi::{c_char, c_void, CStr, CString};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use windows_sys::Win32::Foundation::{FreeLibrary, GetLastError};
use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryA};

use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye, EyeSample, GazeSample};

// If a vendor initialization call exceeds our timeout, its detached thread may
// still own DLL state. Do not stack another potentially wedged API instance in
// the same process; a process restart is the only reliable isolation boundary.
static INIT_WEDGED: AtomicBool = AtomicBool::new(false);

// --- Tobii SDK 2.x struct layout (verified vs ReStar's tobii_wearable.h) ---
// Validity flag ALWAYS precedes its value.
#[repr(C)]
#[derive(Clone, Copy)]
struct TobiiWearableEye {
    gaze_origin_validity: i32,
    gaze_origin_mm: [f32; 3],
    gaze_direction_validity: i32,
    gaze_direction: [f32; 3],
    pupil_diameter_validity: i32,
    pupil_diameter_mm: f32,
    eye_openness_validity: i32,
    eye_openness: f32,
    pupil_position_validity: i32,
    pupil_position: [f32; 2],
}
const _: () = assert!(core::mem::size_of::<TobiiWearableEye>() == 60);

#[repr(C)]
#[derive(Clone, Copy)]
struct TobiiWearableData {
    timestamp_tracker_us: i64,
    timestamp_system_us: i64,
    frame_counter: u32,
    led_mode: u32,
    left: TobiiWearableEye,
    right: TobiiWearableEye,
}
const _: () = assert!(core::mem::size_of::<TobiiWearableData>() == 144);

impl TobiiWearableEye {
    fn to_eye_sample(&self) -> EyeSample {
        EyeSample {
            gaze: self.gaze_direction, // RAW; the output sink negates once.
            gaze_valid: self.gaze_direction_validity != 0,
            gaze_reported: true,
            origin_mm: self.gaze_origin_mm,
            origin_valid: self.gaze_origin_validity != 0,
            pupil_mm: self.pupil_diameter_mm,
            pupil_valid: self.pupil_diameter_validity != 0,
            pupil_pos: self.pupil_position,
            pupil_pos_valid: self.pupil_position_validity != 0,
            openness: self.eye_openness,
            openness_valid: self.eye_openness_validity != 0,
            openness_reported: true,
        }
    }
}

// --- Tobii C API function-pointer types (cdecl == C on x64 Windows) ---
type UrlCb = extern "C" fn(*const c_char, *mut c_void);
type ImgCb = extern "C" fn(*const c_void, *mut c_void);
type AdvCb = extern "C" fn(*const TobiiWearableData, *mut c_void);
type ApiCreate =
    unsafe extern "C" fn(*mut *mut c_void, *const c_void, *const TobiiCustomLog) -> i32;
type ApiDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type EnumUrls = unsafe extern "C" fn(*mut c_void, UrlCb, *mut c_void) -> i32;
type DeviceCreate = unsafe extern "C" fn(*mut c_void, *const c_char, *mut *mut c_void) -> i32;
type DeviceDestroy = unsafe extern "C" fn(*mut c_void) -> i32;
type ImgSub = unsafe extern "C" fn(*mut c_void, ImgCb, *mut c_void) -> i32;
type ImgUnsub = unsafe extern "C" fn(*mut c_void) -> i32;
type AdvSub = unsafe extern "C" fn(*mut c_void, AdvCb, *mut c_void) -> i32;
type AdvUnsub = unsafe extern "C" fn(*mut c_void) -> i32;
type ProcCb = unsafe extern "C" fn(*mut c_void) -> i32;

type TobiiLogFn = extern "C" fn(*mut c_void, i32, *const c_char);

#[repr(C)]
struct TobiiCustomLog {
    context: *mut c_void,
    log_func: Option<TobiiLogFn>,
}

/// Stable names from `tobii_error_t` in the SDK header. Keeping the numeric value
/// too is important when a vendor DLL adds a code newer than our header.
fn error_name(rc: i32) -> &'static str {
    match rc {
        0 => "NO_ERROR",
        1 => "INTERNAL",
        2 => "INSUFFICIENT_LICENSE",
        3 => "NOT_SUPPORTED",
        4 => "NOT_AVAILABLE",
        5 => "CONNECTION_FAILED",
        6 => "TIMED_OUT",
        7 => "ALLOCATION_FAILED",
        8 => "INVALID_PARAMETER",
        9 => "CALIBRATION_ALREADY_STARTED",
        10 => "CALIBRATION_NOT_STARTED",
        11 => "ALREADY_SUBSCRIBED",
        12 => "NOT_SUBSCRIBED",
        13 => "OPERATION_FAILED",
        14 => "CONFLICTING_API_INSTANCES",
        15 => "CALIBRATION_BUSY",
        16 => "CALLBACK_IN_PROGRESS",
        17 => "TOO_MANY_SUBSCRIBERS",
        18 => "CONNECTION_FAILED_DRIVER",
        19 => "UNAUTHORIZED",
        20 => "FIRMWARE_UPGRADE_IN_PROGRESS",
        21 => "INCOMPATIBLE_API_VERSION",
        _ => "UNKNOWN",
    }
}

extern "C" fn tobii_log(_context: *mut c_void, level: i32, text: *const c_char) {
    if text.is_null() {
        return;
    }
    let level = match level {
        0 => "ERROR",
        1 => "WARN",
        2 => "INFO",
        3 => "DEBUG",
        4 => "TRACE",
        _ => "UNKNOWN",
    };
    let message = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    eprintln!("[starvr:tobii] {level}: {message}");
}

/// Shared context handed to the C callbacks via the `user_data` pointer. Lives on
/// the pump thread for the whole subscription; callbacks fire synchronously inside
/// `tobii_device_process_callbacks` on that same thread (no cross-thread aliasing).
struct Ctx {
    on_frame: FrameFn,
    on_gaze: GazeFn,
    slot_a_eye: Eye,
    last_image_timestamp_us: u64,
    image_count: u64,
    wearable_count: u64,
}

type InitSender = mpsc::SyncSender<Result<(), String>>;

fn signal_init_error(sender: &mut Option<InitSender>, message: impl Into<String>) {
    if let Some(sender) = sender.take() {
        let _ = sender.send(Err(message.into()));
    }
}

fn signal_init_ok(sender: &mut Option<InitSender>) {
    if let Some(sender) = sender.take() {
        let _ = sender.send(Ok(()));
    }
}

/// The two eye images in one stereo capture arrive almost back-to-back, while
/// consecutive captures are about 8.3 ms apart at 120 Hz. Grouping by the
/// device timestamp lets the next capture re-synchronise even if a callback is
/// malformed or dropped; a global parity bit cannot recover from that.
const STEREO_PAIR_MAX_DELTA_US: u64 = 2_000;

fn route_image_eye(last_timestamp_us: &mut u64, slot_a_eye: Eye, timestamp_us: u64) -> Eye {
    let is_second_in_pair = *last_timestamp_us != 0
        && timestamp_us >= *last_timestamp_us
        && timestamp_us.saturating_sub(*last_timestamp_us) <= STEREO_PAIR_MAX_DELTA_US;
    *last_timestamp_us = timestamp_us;
    if is_second_in_pair {
        slot_a_eye.opposite()
    } else {
        slot_a_eye
    }
}

extern "C" fn url_cb(url: *const c_char, user: *mut c_void) {
    if url.is_null() || user.is_null() {
        return;
    }
    let urls = unsafe { &mut *(user as *mut Vec<CString>) };
    urls.push(unsafe { CStr::from_ptr(url) }.to_owned());
}

fn choose_device_url(urls: &[CString], preferred_marker: Option<&str>) -> Option<CString> {
    match preferred_marker {
        Some(marker) => urls
            .iter()
            .find(|url| url.to_string_lossy().contains(marker))
            .cloned(),
        None => urls.first().cloned(),
    }
}

extern "C" fn img_cb(img: *const c_void, user: *mut c_void) {
    if img.is_null() || user.is_null() {
        return;
    }
    let ctx = unsafe { &mut *(user as *mut Ctx) };
    // StarVR's tobii_image_t header: device timestamp @0x00 (u64), width @0x10
    // (u32), height @0x14 (u32), data ptr @0x18 (u64).
    let base = img as *const u8;
    let (timestamp_us, w, h, dptr) = unsafe {
        (
            core::ptr::read_unaligned(base as *const u64),
            core::ptr::read_unaligned(base.add(0x10) as *const u32),
            core::ptr::read_unaligned(base.add(0x14) as *const u32),
            core::ptr::read_unaligned(base.add(0x18) as *const u64) as *const u8,
        )
    };
    // Route before validating the payload. An invalid first-eye payload still
    // occupies its timestamp slot, so the valid mate must remain the other eye.
    let eye = route_image_eye(
        &mut ctx.last_image_timestamp_us,
        ctx.slot_a_eye,
        timestamp_us,
    );
    if w == 0 || h == 0 || dptr.is_null() || (w as u64) * (h as u64) > (1 << 20) {
        return;
    }
    let n = (w as usize) * (h as usize);
    let px = unsafe { std::slice::from_raw_parts(dptr, n) };
    ctx.image_count += 1;
    if ctx.image_count == 1 {
        eprintln!("[starvr] first image callback: {w}x{h} ({n} bytes)");
    }
    (ctx.on_frame)(eye, w, h, px); // on_frame copies the slice; borrow ends here.
}

extern "C" fn adv_cb(data: *const TobiiWearableData, user: *mut c_void) {
    if data.is_null() || user.is_null() {
        return;
    }
    let ctx = unsafe { &mut *(user as *mut Ctx) };
    let d = unsafe { &*data };
    ctx.wearable_count += 1;
    if ctx.wearable_count == 1 {
        eprintln!("[starvr] first wearable callback (gaze/pupil/openness)");
    }
    (ctx.on_gaze)(GazeSample {
        timestamp_us: d.timestamp_system_us as u64,
        left: d.left.to_eye_sample(),
        right: d.right.to_eye_sample(),
    });
}

pub struct StarVrAdapter {
    profile: DeviceProfile,
    dll_path: Option<String>,
    /// Whether to stop the Tobii Platform Runtime before connecting. True when the
    /// patched stream engine accesses the EyeChip DIRECTLY (Pimax `pimax_dll`); false
    /// for a service-routed StarVR (don't pull its service out from under it).
    handoff: bool,
    preferred_url_marker: Option<String>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}

impl StarVrAdapter {
    pub fn new(cfg: &crate::config::Config) -> Self {
        // verified: slot A (brighter) = LEFT on StarVR One. Service-routed: no handoff.
        Self::with_profile(cfg, "StarVR One", Eye::Left, false)
    }

    /// Generic Tobii stream-engine adapter for any IS4 HMD (StarVR, or Pimax via
    /// `device = "pimax_dll"`). `name` is the display profile; `slot_a` is which eye
    /// the first image of each callback pair belongs to (Left on the units verified
    /// so far — flip with the eye-swap option if a unit is wired the other way).
    /// `handoff` = stop the Tobii runtime first (direct EyeChip access, e.g. Pimax).
    pub fn with_profile(
        cfg: &crate::config::Config,
        name: &str,
        slot_a: Eye,
        handoff: bool,
    ) -> Self {
        Self {
            profile: DeviceProfile {
                name: name.into(),
                ml_device: "vr4".into(), // same IS4 frontal preprocessing as VR4
                slot_a_eye: slot_a,
                transport: "Tobii stream engine (DLL)".into(),
                streams: "Tobii image + wearable_data".into(),
                gaze_src: "gaze · pupil · openness (Tobii)".into(),
                ..DeviceProfile::default()
            },
            // Common Tobii DLL (shared with the Pimax gate); legacy starvr_dll is
            // still honored via the resolver for back-compat.
            dll_path: cfg
                .tobii_dll_path()
                .map(|p| p.to_string_lossy().into_owned()),
            handoff,
            // Observed StarVR serial URLs start with VRS. Prefer that device if
            // another Tobii runtime also exposes a local URL on the machine.
            preferred_url_marker: name.contains("StarVR").then(|| "VRS".into()),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }
}

fn set_status(s: &Arc<Mutex<String>>, msg: impl Into<String>) {
    if let Ok(mut g) = s.lock() {
        *g = msg.into();
    }
}

/// Resolve an `extern "C"` proc by NUL-terminated name; `None` if absent.
unsafe fn proc<T: Copy>(hmod: *mut c_void, name: &[u8]) -> Option<T> {
    debug_assert_eq!(*name.last().unwrap(), 0, "name must be NUL-terminated");
    GetProcAddress(hmod, name.as_ptr()).map(|f| std::mem::transmute_copy::<_, T>(&f))
}

impl HmdAdapter for StarVrAdapter {
    fn name(&self) -> &'static str {
        "starvr-one"
    }
    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> io::Result<()> {
        if INIT_WEDGED.load(Ordering::Relaxed) {
            return Err(io::Error::new(
                io::ErrorKind::ResourceBusy,
                "StarVR initialization previously timed out; restart SRanibro before retrying",
            ));
        }
        if !self.handoff {
            crate::platform::ensure_starvr_ready();
        }
        let dll_path = self.dll_path.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "StarVR: set [assets].tobii_dll to the Tobii stream-engine DLL",
            )
        })?;
        let stop = self.stop.clone();
        let status = self.status.clone();
        let slot_a = self.profile.slot_a_eye;
        let preferred_url_marker = self.preferred_url_marker.clone();
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        self.stop.store(false, Ordering::Relaxed);

        let handle = thread::spawn(move || {
            let mut init_tx = Some(init_tx);
            set_status(&status, "loading DLL…");
            let dll_c = match CString::new(dll_path.clone()) {
                Ok(c) => c,
                Err(_) => {
                    set_status(&status, "bad DLL path");
                    signal_init_error(&mut init_tx, "StarVR DLL path contains a NUL byte");
                    return;
                }
            };
            let hmod = unsafe { LoadLibraryA(dll_c.as_ptr() as *const u8) };
            if hmod.is_null() {
                let win32 = unsafe { GetLastError() };
                eprintln!("[starvr] LoadLibraryA failed: win32={win32}, path={dll_path}");
                set_status(&status, format!("DLL load failed (Win32 {win32})"));
                signal_init_error(
                    &mut init_tx,
                    format!("StarVR DLL load failed (Win32 {win32})"),
                );
                return;
            }

            // Resolve the C API.
            macro_rules! load {
                ($name:literal, $t:ty) => {
                    match unsafe { proc::<$t>(hmod, concat!($name, "\0").as_bytes()) } {
                        Some(f) => f,
                        None => {
                            eprintln!("[starvr] missing export: {}", $name);
                            set_status(&status, concat!("DLL missing ", $name));
                            signal_init_error(&mut init_tx, concat!("StarVR DLL missing ", $name));
                            unsafe { FreeLibrary(hmod) };
                            return;
                        }
                    }
                };
            }
            let api_create = load!("tobii_api_create", ApiCreate);
            let api_destroy = load!("tobii_api_destroy", ApiDestroy);
            let enum_urls = load!("tobii_enumerate_local_device_urls", EnumUrls);
            let device_create = load!("tobii_device_create", DeviceCreate);
            let device_destroy = load!("tobii_device_destroy", DeviceDestroy);
            let img_sub = load!("tobii_image_subscribe", ImgSub);
            let img_unsub = load!("tobii_image_unsubscribe", ImgUnsub);
            let adv_sub = load!("tobii_wearable_data_subscribe", AdvSub);
            let adv_unsub = load!("tobii_wearable_data_unsubscribe", AdvUnsub);
            let process = load!("tobii_device_process_callbacks", ProcCb);

            // api_create -> enumerate -> device_create
            let mut api: *mut c_void = std::ptr::null_mut();
            // Capture the Tobii stack's own diagnostics. The struct stays alive for
            // the entire API lifetime because this pump thread owns the stack frame.
            let custom_log = TobiiCustomLog {
                context: std::ptr::null_mut(),
                log_func: Some(tobii_log),
            };
            let api_rc = unsafe { api_create(&mut api, std::ptr::null(), &custom_log) };
            if api_rc != 0 || api.is_null() {
                eprintln!(
                    "[starvr] tobii_api_create failed: rc={api_rc} ({}), api_null={}",
                    error_name(api_rc),
                    api.is_null()
                );
                set_status(
                    &status,
                    format!("tobii_api_create failed: {api_rc} ({})", error_name(api_rc)),
                );
                signal_init_error(
                    &mut init_tx,
                    format!("tobii_api_create failed: {api_rc} ({})", error_name(api_rc)),
                );
                unsafe { FreeLibrary(hmod) };
                return;
            }
            eprintln!("[starvr] tobii_api_create ok");
            let mut urls: Vec<CString> = Vec::new();
            let enum_rc = unsafe { enum_urls(api, url_cb, &mut urls as *mut _ as *mut c_void) };
            if enum_rc != 0 {
                eprintln!(
                    "[starvr] tobii_enumerate_local_device_urls failed: rc={enum_rc} ({})",
                    error_name(enum_rc)
                );
                set_status(
                    &status,
                    format!(
                        "device enumeration failed: {enum_rc} ({})",
                        error_name(enum_rc)
                    ),
                );
                signal_init_error(
                    &mut init_tx,
                    format!(
                        "device enumeration failed: {enum_rc} ({})",
                        error_name(enum_rc)
                    ),
                );
                unsafe {
                    api_destroy(api);
                    FreeLibrary(hmod);
                }
                return;
            }
            let url = match choose_device_url(&urls, preferred_url_marker.as_deref()) {
                Some(u) => u,
                None => {
                    eprintln!("[starvr] enumeration succeeded but returned no device URL");
                    set_status(&status, "no StarVR device found (is it connected?)");
                    signal_init_error(&mut init_tx, "no StarVR device found (is it connected?)");
                    unsafe {
                        api_destroy(api);
                        FreeLibrary(hmod);
                    }
                    return;
                }
            };
            let mut dev: *mut c_void = std::ptr::null_mut();
            let device_rc = unsafe { device_create(api, url.as_ptr(), &mut dev) };
            if device_rc != 0 || dev.is_null() {
                eprintln!(
                    "[starvr] tobii_device_create failed: rc={device_rc} ({}), device_null={}, url={}",
                    error_name(device_rc),
                    dev.is_null(),
                    url.to_string_lossy()
                );
                set_status(
                    &status,
                    format!(
                        "tobii_device_create failed: {device_rc} ({})",
                        error_name(device_rc)
                    ),
                );
                signal_init_error(
                    &mut init_tx,
                    format!(
                        "tobii_device_create failed: {device_rc} ({})",
                        error_name(device_rc)
                    ),
                );
                unsafe {
                    api_destroy(api);
                    FreeLibrary(hmod);
                }
                return;
            }
            eprintln!("[starvr] device opened: {}", url.to_string_lossy());

            // Subscribe. Ctx lives on this thread for the whole loop; its raw ptr is
            // the callbacks' user_data.
            let mut ctx = Box::new(Ctx {
                on_frame,
                on_gaze,
                slot_a_eye: slot_a,
                last_image_timestamp_us: 0,
                image_count: 0,
                wearable_count: 0,
            });
            let ctx_ptr = &mut *ctx as *mut Ctx as *mut c_void;

            let image_rc = unsafe { img_sub(dev, img_cb, ctx_ptr) };
            if image_rc != 0 {
                eprintln!(
                    "[starvr] tobii_image_subscribe failed: rc={image_rc} ({})",
                    error_name(image_rc)
                );
                set_status(
                    &status,
                    format!(
                        "tobii_image_subscribe failed: {image_rc} ({})",
                        error_name(image_rc)
                    ),
                );
                signal_init_error(
                    &mut init_tx,
                    format!(
                        "tobii_image_subscribe failed: {image_rc} ({})",
                        error_name(image_rc)
                    ),
                );
                unsafe {
                    device_destroy(dev);
                    api_destroy(api);
                    FreeLibrary(hmod);
                }
                return;
            }
            // Gaze/pupil/openness is best-effort (the image alone still feeds the ML).
            let adv_rc = unsafe { adv_sub(dev, adv_cb, ctx_ptr) };
            let have_adv = adv_rc == 0;
            if !have_adv {
                eprintln!(
                    "[starvr] tobii_wearable_data_subscribe failed: rc={adv_rc} ({}) (no gaze overlay)",
                    error_name(adv_rc)
                );
            }
            set_status(&status, "streaming");
            eprintln!("[starvr] streaming");
            signal_init_ok(&mut init_tx);

            let mut consecutive_process_errors = 0u32;
            while !stop.load(Ordering::Relaxed) {
                let rc = unsafe { process(dev) };
                if rc != 0 {
                    consecutive_process_errors += 1;
                    eprintln!(
                        "[starvr] tobii_device_process_callbacks failed: rc={rc} ({}) (attempt {consecutive_process_errors}/5)",
                        error_name(rc),
                    );
                    if consecutive_process_errors < 5 {
                        set_status(
                            &status,
                            format!(
                                "callback error {rc} ({}); retrying {consecutive_process_errors}/5",
                                error_name(rc)
                            ),
                        );
                        thread::sleep(Duration::from_millis(
                            20 * u64::from(consecutive_process_errors),
                        ));
                        continue;
                    }
                    set_status(
                        &status,
                        format!("callback processing failed: {rc} ({})", error_name(rc)),
                    );
                    break;
                }
                if consecutive_process_errors != 0 {
                    consecutive_process_errors = 0;
                    set_status(&status, "streaming");
                    eprintln!("[starvr] callback processing recovered");
                }
                thread::sleep(Duration::from_millis(2));
            }

            // Teardown BEFORE dropping ctx, so no callback can fire on freed state.
            unsafe {
                if have_adv {
                    adv_unsub(dev);
                }
                img_unsub(dev);
                device_destroy(dev);
                api_destroy(api);
                FreeLibrary(hmod);
            }
            drop(ctx);
            if stop.load(Ordering::Relaxed) {
                set_status(&status, "stopped");
                eprintln!("[starvr] stopped");
            }
        });
        self.thread = Some(handle);
        match init_rx.recv_timeout(Duration::from_secs(10)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(message)) => {
                if let Some(handle) = self.thread.take() {
                    let _ = handle.join();
                }
                Err(io::Error::new(io::ErrorKind::ConnectionRefused, message))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.stop.store(true, Ordering::Relaxed);
                INIT_WEDGED.store(true, Ordering::Relaxed);
                set_status(
                    &self.status,
                    "initialization timed out; restart SRanibro before retrying",
                );
                // A vendor call may itself be wedged. Detach rather than block
                // the UI forever; the stop flag makes it exit if the call returns.
                self.thread.take();
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "StarVR initialization timed out after 10 seconds",
                ))
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if let Some(handle) = self.thread.take() {
                    let _ = handle.join();
                }
                Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "StarVR initialization thread exited unexpectedly",
                ))
            }
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }

    fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }

    fn needs_eyechip_handoff(&self) -> bool {
        self.handoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_pair_routing_recovers_after_a_missing_callback() {
        let mut last = 0;

        assert_eq!(route_image_eye(&mut last, Eye::Left, 10_000), Eye::Left);
        assert_eq!(route_image_eye(&mut last, Eye::Left, 10_085), Eye::Right);

        // The second callback of this capture never arrives. The long gap to
        // the next capture must reset routing to slot A instead of swapping all
        // subsequent eyes as parity routing would.
        assert_eq!(route_image_eye(&mut last, Eye::Left, 18_333), Eye::Left);
        assert_eq!(route_image_eye(&mut last, Eye::Left, 26_666), Eye::Left);
        assert_eq!(route_image_eye(&mut last, Eye::Left, 26_750), Eye::Right);
    }

    #[test]
    fn timestamp_pair_routing_counts_an_invalid_payload_slot() {
        let mut last = 0;

        // img_cb calls the router before payload validation. Even if the first
        // payload is rejected, its close-timestamp mate remains slot B.
        assert_eq!(route_image_eye(&mut last, Eye::Left, 50_000), Eye::Left);
        assert_eq!(route_image_eye(&mut last, Eye::Left, 50_090), Eye::Right);
    }

    #[test]
    fn device_selection_prefers_starvr_serial_marker() {
        let urls = vec![
            CString::new("tobii-ttp://PIMAX-OTHER").unwrap(),
            CString::new("tobii-ttp://VRS02-820F1C027600").unwrap(),
        ];
        assert_eq!(choose_device_url(&urls, Some("VRS")).unwrap(), urls[1]);
        assert!(choose_device_url(&urls, Some("STARVR-NOT-PRESENT")).is_none());
        assert_eq!(choose_device_url(&urls, None).unwrap(), urls[0]);
    }

    #[test]
    fn timestamp_reset_starts_a_new_stereo_pair() {
        let mut last = 90_000;
        assert_eq!(route_image_eye(&mut last, Eye::Left, 100), Eye::Left);
    }
}
