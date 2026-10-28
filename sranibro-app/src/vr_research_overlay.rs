//! Optional head-locked SteamVR target for XR5 residual research recordings.
//!
//! OpenVR is loaded dynamically and only while a target is requested. SRanibro
//! therefore keeps working on systems without SteamVR and does not redistribute
//! Valve's runtime DLL. All OpenVR calls live on one worker thread.

use crate::recorder_fixation::{self, Feedback, Matrix34, PoseHistory, WorldPose};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::geometry_calib::GazeTarget;

pub const TARGET_HORIZONTAL_DEG: f32 = 20.0;
pub const TARGET_VERTICAL_DEG: f32 = 11.0;

/// Transparent texture for the world-fixed recorder overlay, 0.075 m wide.
pub fn world_target_rgba() -> Vec<u8> {
    let mut rgba = vec![0; 128 * 128 * 4];
    for y in 0..128_i32 {
        for x in 0..128_i32 {
            let radius2 = (x - 64).pow(2) + (y - 64).pow(2);
            let color = if radius2 <= 6 * 6 {
                [255, 255, 255, 255]
            } else if (20 * 20..=22 * 22).contains(&radius2) {
                [110, 230, 190, 255]
            } else {
                [0, 0, 0, 0]
            };
            rgba[((y * 128 + x) * 4) as usize..((y * 128 + x) * 4 + 4) as usize]
                .copy_from_slice(&color);
        }
    }
    rgba
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VrGuideState {
    Ready,
    Prepare,
    Recording,
    Paused,
    Waiting,
    Adjust,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VrEyePose {
    #[default]
    None,
    Open,
    HalfOpen,
    Closed,
    /// Expected openness from 0 (closed) to 100 (open).
    Slow(u8),
    Blink,
    LeftWink,
    RightWink,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VrTargetFrame {
    pub target: Option<GazeTarget>,
    pub state: VrGuideState,
    pub eye_pose: VrEyePose,
    /// Integer-only so the SteamVR texture changes at most once per second.
    pub countdown: Option<u8>,
    pub progress_percent: Option<u8>,
    pub headline: String,
    pub instruction: String,
    pub footer: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VrOverlayState {
    Hidden,
    Starting,
    Visible,
    Error(String),
}

enum Command {
    Show(VrTargetFrame),
    World(u64),
    Hide,
    Shutdown,
}

/// Non-blocking UI-side controller. Rendering and all runtime calls happen on a
/// dedicated thread so a stopped or starting SteamVR process cannot stall egui.
pub struct VrResearchOverlay {
    tx: mpsc::Sender<Command>,
    state: Arc<Mutex<VrOverlayState>>,
    worker: Option<JoinHandle<()>>,
    last_frame: Option<VrTargetFrame>,
    hidden: bool,
    visible_during_session: Arc<std::sync::atomic::AtomicBool>,
    world_feedback: Feedback,
    world_session: Option<u64>,
}

impl Default for VrResearchOverlay {
    fn default() -> Self {
        Self::new()
    }
}

impl VrResearchOverlay {
    pub fn new() -> Self {
        Self::with_recorder_layout(false)
    }

    /// Compact timer layout for the standalone recorder; normal calibration
    /// overlays keep their existing renderer and target positioning.
    pub fn new_recorder() -> Self {
        Self::with_recorder_layout(true)
    }

    fn with_recorder_layout(recorder_layout: bool) -> Self {
        let (tx, rx) = mpsc::channel();
        let state = Arc::new(Mutex::new(VrOverlayState::Hidden));
        let thread_state = state.clone();
        let visible_during_session = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let thread_visible = visible_during_session.clone();
        let world_feedback = Arc::new(Mutex::new(PoseHistory::default()));
        let thread_feedback = world_feedback.clone();
        let worker = std::thread::Builder::new()
            .name("steamvr-research-overlay".into())
            .spawn(move || {
                worker_main(
                    rx,
                    thread_state,
                    thread_visible,
                    recorder_layout,
                    thread_feedback,
                )
            })
            .ok();
        if worker.is_none() {
            *state.lock().unwrap() =
                VrOverlayState::Error("could not start SteamVR overlay worker".into());
        }
        Self {
            tx,
            state,
            worker,
            last_frame: None,
            hidden: true,
            visible_during_session,
            world_feedback,
            world_session: None,
        }
    }

    pub fn present(&mut self, enabled: bool, frame: Option<VrTargetFrame>) {
        let Some(frame) = frame.filter(|_| enabled) else {
            self.hide();
            return;
        };
        if !self.hidden && self.last_frame.as_ref() == Some(&frame) {
            return;
        }
        self.hidden = false;
        self.last_frame = Some(frame.clone());
        if self.tx.send(Command::Show(frame)).is_err() {
            *self.state.lock().unwrap() =
                VrOverlayState::Error("SteamVR overlay worker stopped".into());
        }
    }

    pub fn world_feedback(&self) -> Feedback {
        self.world_feedback.clone()
    }

    /// One absolute target per session. Repeated calls never reposition it.
    pub fn present_world(&mut self, session: u64) {
        if self.world_session == Some(session) && !self.hidden {
            return;
        }
        self.hidden = false;
        self.last_frame = None;
        self.world_session = Some(session);
        let _ = self.tx.send(Command::World(session));
    }

    pub fn hide(&mut self) {
        if self.hidden {
            return;
        }
        self.hidden = true;
        self.world_session = None;
        self.last_frame = None;
        let _ = self.tx.send(Command::Hide);
    }

    pub fn reset_session(&self) {
        self.visible_during_session
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn was_visible_during_session(&self) -> bool {
        self.visible_during_session
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn is_visible(&self) -> bool {
        matches!(*self.state.lock().unwrap(), VrOverlayState::Visible)
    }

    pub fn status_text(&self) -> String {
        match &*self.state.lock().unwrap() {
            VrOverlayState::Hidden => "SteamVR calibration guide is idle.".into(),
            VrOverlayState::Starting => "Connecting to SteamVR...".into(),
            VrOverlayState::Visible => format!(
                "SteamVR calibration guide is visible (about +/-{TARGET_HORIZONTAL_DEG:.0} deg horizontal, +/-{TARGET_VERTICAL_DEG:.0} deg vertical)."
            ),
            VrOverlayState::Error(error) => {
                format!("SteamVR target unavailable: {error}. Desktop target remains active.")
            }
        }
    }
}

impl Drop for VrResearchOverlay {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(not(windows))]
fn worker_main(
    rx: mpsc::Receiver<Command>,
    state: Arc<Mutex<VrOverlayState>>,
    _visible: Arc<std::sync::atomic::AtomicBool>,
    _recorder_layout: bool,
    _feedback: Feedback,
) {
    while let Ok(command) = rx.recv() {
        match command {
            Command::Show(_) => {
                *state.lock().unwrap() =
                    VrOverlayState::Error("SteamVR overlay is Windows-only".into());
            }
            Command::World(_) => {
                *state.lock().unwrap() = VrOverlayState::Error("SteamVR is Windows-only".into());
            }
            Command::Hide => *state.lock().unwrap() = VrOverlayState::Hidden,
            Command::Shutdown => break,
        }
    }
}

#[cfg(windows)]
mod windows_impl {
    use super::*;
    use font8x8::{UnicodeFonts, BASIC_FONTS};
    use std::collections::HashSet;
    use std::ffi::{c_char, c_void, CString, OsStr};
    use std::os::windows::ffi::OsStrExt;
    use std::path::{Path, PathBuf};
    use windows_sys::Win32::Foundation::{FreeLibrary, HMODULE};
    use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    const WIDTH: usize = 1024;
    const HEIGHT: usize = 576;
    const EYE_POSE_CENTER_Y: i32 = 292;
    const OVERLAY_WIDTH_METERS: f32 = 1.45;
    const OVERLAY_DISTANCE_METERS: f32 = 1.40;
    const VR_APPLICATION_OVERLAY: i32 = 2;
    const VR_OVERLAY_OK: i32 = 0;
    const HMD_DEVICE_INDEX: u32 = 0;

    type VrInit = unsafe extern "system" fn(*mut i32, i32) -> isize;
    type VrGetInterface = unsafe extern "system" fn(*const c_char, *mut i32) -> isize;
    type VrShutdown = unsafe extern "system" fn();
    type CreateOverlay = unsafe extern "system" fn(*mut c_char, *mut c_char, *mut u64) -> i32;
    type DestroyOverlay = unsafe extern "system" fn(u64) -> i32;
    type SetOverlayAlpha = unsafe extern "system" fn(u64, f32) -> i32;
    type SetOverlayWidth = unsafe extern "system" fn(u64, f32) -> i32;
    type SetOverlayTransform = unsafe extern "system" fn(u64, u32, *mut HmdMatrix34) -> i32;
    type ShowOverlay = unsafe extern "system" fn(u64) -> i32;
    type HideOverlay = unsafe extern "system" fn(u64) -> i32;
    type SetOverlayRaw = unsafe extern "system" fn(u64, *mut c_void, u32, u32, u32) -> i32;
    type GetPoses = unsafe extern "system" fn(i32, f32, *mut TrackedPose, u32);
    type GetEye = unsafe extern "system" fn(i32) -> HmdMatrix34;
    type GetOrigin = unsafe extern "system" fn() -> HmdMatrix34;
    type IsVisible = unsafe extern "system" fn(u64) -> bool;

    #[repr(C)]
    #[derive(Default)]
    struct TrackedPose {
        transform: HmdMatrix34,
        velocity: [f32; 3],
        angular_velocity: [f32; 3],
        result: i32,
        valid: bool,
        connected: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct HmdMatrix34 {
        m: [[f32; 4]; 3],
    }

    struct Api {
        module: HMODULE,
        shutdown: VrShutdown,
        destroy: DestroyOverlay,
        show: ShowOverlay,
        hide: HideOverlay,
        set_raw: SetOverlayRaw,
        handle: u64,
        shown: bool,
        /// SetOverlayRaw completes asynchronously. Keep the preceding allocation alive
        /// while SteamVR imports the replacement instead of freeing it immediately.
        previous_rgba: Option<Vec<u8>>,
        rgba: Vec<u8>,
        world: Option<WorldApi>,
        reset_head_layout: bool,
    }

    struct WorldApi {
        session: u64,
        get_poses: GetPoses,
        get_eye: GetEye,
        get_origin: GetOrigin,
        is_visible: IsVisible,
        origin: Matrix34,
        target: [f32; 3],
        faulted: bool,
    }

    impl Api {
        unsafe fn connect() -> Result<Self, String> {
            let module = load_openvr()?;
            let result = Self::connect_loaded(module);
            if result.is_err() {
                FreeLibrary(module);
            }
            result
        }

        unsafe fn connect_loaded(module: HMODULE) -> Result<Self, String> {
            let init: VrInit = exported(module, b"VR_InitInternal\0")?;
            let get_interface: VrGetInterface = exported(module, b"VR_GetGenericInterface\0")?;
            let shutdown: VrShutdown = exported(module, b"VR_ShutdownInternal\0")?;

            let mut error = 0;
            let token = init(&mut error, VR_APPLICATION_OVERLAY);
            if error != 0 || token == 0 {
                return Err(format!("OpenVR initialization error {error}"));
            }

            let mut interface_error = 0;
            let table = get_interface(c"FnTable:IVROverlay_027".as_ptr(), &mut interface_error)
                as *const *const c_void;
            if interface_error != 0 || table.is_null() {
                shutdown();
                return Err(format!("OpenVR overlay interface error {interface_error}"));
            }

            let functions = (|| {
                Ok::<_, String>((
                    table_fn::<CreateOverlay>(table, 1)?,
                    table_fn::<DestroyOverlay>(table, 2)?,
                    table_fn::<SetOverlayAlpha>(table, 15)?,
                    table_fn::<SetOverlayWidth>(table, 21)?,
                    table_fn::<SetOverlayTransform>(table, 34)?,
                    table_fn::<ShowOverlay>(table, 41)?,
                    table_fn::<HideOverlay>(table, 42)?,
                    table_fn::<SetOverlayRaw>(table, 60)?,
                ))
            })();
            let (create, destroy, set_alpha, set_width, set_transform, show, hide, set_raw) =
                match functions {
                    Ok(functions) => functions,
                    Err(error) => {
                        shutdown();
                        return Err(error);
                    }
                };

            let key = CString::new(format!("sranibro.residual.{}", std::process::id())).unwrap();
            let name = CString::new("SRanibro calibration guide").unwrap();
            let mut handle = 0u64;
            if let Err(error) = check_overlay(
                "create overlay",
                create(
                    key.as_ptr() as *mut c_char,
                    name.as_ptr() as *mut c_char,
                    &mut handle,
                ),
            ) {
                shutdown();
                return Err(error);
            }
            let configured = (|| {
                check_overlay("set overlay alpha", set_alpha(handle, 1.0))?;
                check_overlay("set overlay width", set_width(handle, OVERLAY_WIDTH_METERS))?;
                let mut transform = HmdMatrix34 {
                    m: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, -OVERLAY_DISTANCE_METERS],
                    ],
                };
                check_overlay(
                    "set head-relative transform",
                    set_transform(handle, HMD_DEVICE_INDEX, &mut transform),
                )
            })();
            if let Err(error) = configured {
                let _ = destroy(handle);
                shutdown();
                return Err(error);
            }
            Ok(Self {
                module,
                shutdown,
                destroy,
                show,
                hide,
                set_raw,
                handle,
                shown: false,
                previous_rgba: None,
                rgba: Vec::new(),
                world: None,
                reset_head_layout: false,
            })
        }

        unsafe fn present(
            &mut self,
            frame: &VrTargetFrame,
            recorder_layout: bool,
        ) -> Result<(), String> {
            if self.reset_head_layout {
                let interface: VrGetInterface = exported(self.module, b"VR_GetGenericInterface\0")?;
                let mut error = 0;
                let table = interface(c"FnTable:IVROverlay_027".as_ptr(), &mut error)
                    as *const *const c_void;
                if error != 0 || table.is_null() {
                    return Err("Could not restore recorder timer layout".into());
                }
                let set_width = table_fn::<SetOverlayWidth>(table, 21)?;
                let relative = table_fn::<SetOverlayTransform>(table, 34)?;
                let mut transform = HmdMatrix34 {
                    m: [
                        [1., 0., 0., 0.],
                        [0., 1., 0., 0.],
                        [0., 0., 1., -OVERLAY_DISTANCE_METERS],
                    ],
                };
                check_overlay(
                    "restore timer width",
                    set_width(self.handle, OVERLAY_WIDTH_METERS),
                )?;
                check_overlay(
                    "restore head-relative timer",
                    relative(self.handle, 0, &mut transform),
                )?;
                self.reset_head_layout = false;
                self.world = None;
            }
            let next = if recorder_layout {
                render_recorder(frame)
            } else {
                render(frame)
            };
            self.previous_rgba = Some(std::mem::replace(&mut self.rgba, next));
            check_overlay(
                "upload RGBA target",
                (self.set_raw)(
                    self.handle,
                    self.rgba.as_mut_ptr().cast(),
                    WIDTH as u32,
                    HEIGHT as u32,
                    4,
                ),
            )?;
            if !self.shown {
                check_overlay("show overlay", (self.show)(self.handle))?;
                self.shown = true;
            }
            Ok(())
        }

        unsafe fn hide(&mut self) {
            self.world = None;
            if self.shown {
                let _ = (self.hide)(self.handle);
                self.shown = false;
            }
        }

        unsafe fn start_world(&mut self, session: u64) -> Result<(), String> {
            let interface: VrGetInterface = exported(self.module, b"VR_GetGenericInterface\0")?;
            let mut error = 0;
            // Function indexes and ABI are pinned to Valve openvr v2.5.1.
            let system =
                interface(c"FnTable:IVRSystem_022".as_ptr(), &mut error) as *const *const c_void;
            if error != 0 || system.is_null() {
                return Err(format!("OpenVR system interface error {error}"));
            }
            let overlay =
                interface(c"FnTable:IVROverlay_027".as_ptr(), &mut error) as *const *const c_void;
            if error != 0 || overlay.is_null() {
                return Err(format!("OpenVR overlay interface error {error}"));
            }
            let get_poses = table_fn::<GetPoses>(system, 11)?;
            let get_eye = table_fn::<GetEye>(system, 4)?;
            let get_origin = table_fn::<GetOrigin>(system, 13)?;
            let absolute = table_fn::<SetOverlayTransform>(overlay, 32)?;
            let set_width = table_fn::<SetOverlayWidth>(overlay, 21)?;
            let is_visible = table_fn::<IsVisible>(overlay, 43)?;
            let mut pose = TrackedPose::default();
            get_poses(1, 0.0, &mut pose, 1); // standing origin; zero prediction
            if !pose.valid
                || !pose.connected
                || pose.result != 200
                || !recorder_fixation::rigid(pose.transform.m)
            {
                return Err("Waiting for valid HMD tracking before placing the dot".into());
            }
            let origin = get_origin().m;
            if !recorder_fixation::rigid(origin) {
                return Err("Invalid standing-origin transform".into());
            }
            let mut anchor = pose.transform;
            let target =
                recorder_fixation::transform(anchor.m, [0., 0., -recorder_fixation::DISTANCE_M]);
            for i in 0..3 {
                anchor.m[i][3] = target[i];
            }
            check_overlay("set world dot width", set_width(self.handle, 0.075))?;
            check_overlay(
                "set world-fixed transform",
                absolute(self.handle, 1, &mut anchor),
            )?;
            self.previous_rgba = Some(std::mem::replace(
                &mut self.rgba,
                super::world_target_rgba(),
            ));
            check_overlay(
                "upload world dot",
                (self.set_raw)(self.handle, self.rgba.as_mut_ptr().cast(), 128, 128, 4),
            )?;
            check_overlay("show world dot", (self.show)(self.handle))?;
            self.shown = true;
            self.world = Some(WorldApi {
                session,
                get_poses,
                get_eye,
                get_origin,
                is_visible,
                origin,
                target,
                faulted: false,
            });
            self.reset_head_layout = true;
            Ok(())
        }

        unsafe fn world_pose(&self) -> Result<WorldPose, String> {
            let world = self.world.as_ref().ok_or("World target not placed")?;
            if world.faulted {
                return Err("World target stopped. Save/cancel and start a new recording.".into());
            }
            let origin = (world.get_origin)().m;
            if !recorder_fixation::same_origin(origin, world.origin) {
                return Err("Tracking origin changed. Stop/save and start a new recording.".into());
            }
            let mut pose = TrackedPose::default();
            (world.get_poses)(1, 0.0, &mut pose, 1);
            Ok(WorldPose {
                session: world.session,
                queried_at: Instant::now(),
                target_world: world.target,
                head_to_world: pose.transform.m,
                eye_to_head: [(world.get_eye)(0).m, (world.get_eye)(1).m],
                angular_velocity: pose.angular_velocity,
                tracking_result: pose.result,
                valid: pose.valid,
                connected: pose.connected,
                overlay_visible: (world.is_visible)(self.handle),
            })
        }
    }

    impl Drop for Api {
        fn drop(&mut self) {
            unsafe {
                let _ = (self.hide)(self.handle);
                let _ = (self.destroy)(self.handle);
                (self.shutdown)();
                FreeLibrary(self.module);
            }
        }
    }

    pub(super) fn run(
        rx: mpsc::Receiver<Command>,
        state: Arc<Mutex<VrOverlayState>>,
        visible: Arc<std::sync::atomic::AtomicBool>,
        recorder_layout: bool,
        feedback: Feedback,
    ) {
        let mut api: Option<Api> = None;
        let mut world_requested: Option<u64> = None;
        let mut retry_at = Instant::now() - Duration::from_secs(2);
        loop {
            let command = if world_requested.is_some() {
                match rx.recv_timeout(Duration::from_millis(8)) {
                    Ok(c) => Some(c),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                    Err(_) => break,
                }
            } else {
                match rx.recv() {
                    Ok(c) => Some(c),
                    Err(_) => break,
                }
            };
            if let Some(command) = command {
                match command {
                    Command::Show(frame) => {
                        world_requested = None;
                        if api.is_none() {
                            *state.lock().unwrap() = VrOverlayState::Starting;
                            match unsafe { Api::connect() } {
                                Ok(connected) => api = Some(connected),
                                Err(error) => {
                                    *state.lock().unwrap() = VrOverlayState::Error(error);
                                    continue;
                                }
                            }
                        }
                        match unsafe { api.as_mut().unwrap().present(&frame, recorder_layout) } {
                            Ok(()) => {
                                *state.lock().unwrap() = VrOverlayState::Visible;
                                visible.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            Err(error) => {
                                *state.lock().unwrap() = VrOverlayState::Error(error);
                                api = None;
                            }
                        }
                    }
                    Command::Hide => {
                        world_requested = None;
                        feedback.lock().unwrap().fail("Target hidden".into());
                        if let Some(api) = api.as_mut() {
                            unsafe { api.hide() };
                        }
                        *state.lock().unwrap() = VrOverlayState::Hidden;
                    }
                    Command::World(session) => {
                        if feedback.lock().unwrap().requested_session != Some(session) {
                            continue;
                        }
                        // Preserve the anchor across Pause/Resume and UI repaints.
                        if world_requested != Some(session) {
                            if let Some(api) = api.as_mut() {
                                unsafe { api.hide() };
                            }
                            feedback
                                .lock()
                                .unwrap()
                                .fail("Waiting for world target".into());
                            world_requested = Some(session);
                            retry_at = Instant::now() - Duration::from_secs(2);
                        }
                    }
                    Command::Shutdown => break,
                }
            }
            if let Some(session) = world_requested {
                if feedback.lock().unwrap().requested_session != Some(session) {
                    if let Some(a) = api.as_mut() {
                        unsafe { a.hide() };
                    }
                    world_requested = None;
                    *state.lock().unwrap() = VrOverlayState::Hidden;
                    continue;
                }
                if api.is_none() && retry_at.elapsed() >= Duration::from_secs(1) {
                    retry_at = Instant::now();
                    match unsafe { Api::connect() } {
                        Ok(a) => api = Some(a),
                        Err(e) => {
                            feedback.lock().unwrap().fail(e.clone());
                            *state.lock().unwrap() = VrOverlayState::Error(e);
                        }
                    }
                }
                if let Some(a) = api.as_mut() {
                    let result = unsafe {
                        if a.world.is_none() {
                            a.start_world(session)
                        } else {
                            Ok(())
                        }
                        .and_then(|()| a.world_pose())
                    };
                    match result {
                        Ok(p) => {
                            *state.lock().unwrap() = if p.overlay_visible {
                                VrOverlayState::Visible
                            } else {
                                VrOverlayState::Starting
                            };
                            if p.overlay_visible {
                                visible.store(true, std::sync::atomic::Ordering::Relaxed);
                            }
                            feedback.lock().unwrap().push(p);
                        }
                        Err(e) => {
                            if let Some(world) = &mut a.world {
                                if !world.faulted {
                                    unsafe {
                                        (a.hide)(a.handle);
                                    }
                                    a.shown = false;
                                    world.faulted = true;
                                }
                            }
                            feedback.lock().unwrap().fail(e.clone());
                            *state.lock().unwrap() = VrOverlayState::Error(e);
                            // Never reconnect/re-anchor during an anchored session.
                            // Cancel and start again if the origin or runtime changes.
                        }
                    }
                }
            }
        }
    }

    unsafe fn exported<T: Copy>(module: HMODULE, name: &[u8]) -> Result<T, String> {
        let function = GetProcAddress(module, name.as_ptr());
        function
            .map(|pointer| std::mem::transmute_copy(&pointer))
            .ok_or_else(|| {
                format!(
                    "SteamVR OpenVR runtime is missing the required symbol {}",
                    String::from_utf8_lossy(&name[..name.len().saturating_sub(1)])
                )
            })
    }

    unsafe fn table_fn<T: Copy>(table: *const *const c_void, index: usize) -> Result<T, String> {
        let pointer = *table.add(index);
        if pointer.is_null() {
            Err(format!("OpenVR overlay function {index} is null"))
        } else {
            Ok(std::mem::transmute_copy(&pointer))
        }
    }

    fn check_overlay(operation: &str, error: i32) -> Result<(), String> {
        if error == VR_OVERLAY_OK {
            Ok(())
        } else {
            Err(format!(
                "{operation} failed with OpenVR overlay error {error}"
            ))
        }
    }

    unsafe fn load_openvr() -> Result<HMODULE, String> {
        for path in openvr_candidates() {
            let wide: Vec<u16> = OsStr::new(&path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let module = LoadLibraryW(wide.as_ptr());
            if !module.is_null() {
                return Ok(module);
            }
        }
        Err("SteamVR OpenVR runtime was not found; start or install SteamVR".into())
    }

    fn openvr_candidates() -> Vec<PathBuf> {
        let mut candidates = Vec::new();
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            let registry = PathBuf::from(local)
                .join("openvr")
                .join("openvrpaths.vrpath");
            if let Ok(text) = std::fs::read_to_string(registry) {
                candidates.extend(
                    runtime_paths(&text)
                        .into_iter()
                        .map(|path| Path::new(&path).join(r"bin\win64\openvr_api.dll")),
                );
            }
        }
        if let Ok(program_files) = std::env::var("ProgramFiles(x86)") {
            candidates.push(
                PathBuf::from(program_files)
                    .join(r"Steam\steamapps\common\SteamVR\bin\win64\openvr_api.dll"),
            );
        }
        let mut seen = HashSet::new();
        candidates
            .into_iter()
            .filter(|path| path.is_absolute())
            .filter(|path| seen.insert(path.to_string_lossy().to_ascii_lowercase()))
            .collect()
    }

    fn runtime_paths(text: &str) -> Vec<String> {
        let Some(key) = text.find("\"runtime\"") else {
            return Vec::new();
        };
        let Some(open_rel) = text[key..].find('[') else {
            return Vec::new();
        };
        let tail = &text[key + open_rel + 1..];
        let Some(close) = tail.find(']') else {
            return Vec::new();
        };
        parse_json_strings(&tail[..close])
    }

    fn parse_json_strings(text: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut chars = text.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch != '"' {
                continue;
            }
            let mut value = String::new();
            while let Some(ch) = chars.next() {
                match ch {
                    '"' => break,
                    '\\' => match chars.next() {
                        Some('"') => value.push('"'),
                        Some('\\') => value.push('\\'),
                        Some('/') => value.push('/'),
                        Some('b') => value.push('\u{0008}'),
                        Some('f') => value.push('\u{000c}'),
                        Some('n') => value.push('\n'),
                        Some('r') => value.push('\r'),
                        Some('t') => value.push('\t'),
                        Some(other) => value.push(other),
                        None => break,
                    },
                    other => value.push(other),
                }
            }
            if !value.is_empty() {
                result.push(value);
            }
        }
        result
    }

    fn recorder_text(rgba: &mut [u8], y: f32, text: &str, size: f32, color: [u8; 4]) {
        use ab_glyph::{point, Font, FontArc, ScaleFont};
        static FONT: std::sync::OnceLock<FontArc> = std::sync::OnceLock::new();
        let font = FONT.get_or_init(|| {
            let definitions = eframe::egui::FontDefinitions::default();
            let name = &definitions.families[&eframe::egui::FontFamily::Proportional][0];
            FontArc::try_from_vec(definitions.font_data[name].font.to_vec())
                .expect("bundled UI font")
        });
        let scaled = font.as_scaled(size);
        let ids: Vec<_> = text.chars().map(|c| scaled.glyph_id(c)).collect();
        let width: f32 = ids
            .iter()
            .enumerate()
            .map(|(i, &id)| {
                scaled.h_advance(id)
                    + if i > 0 {
                        scaled.kern(ids[i - 1], id)
                    } else {
                        0.0
                    }
            })
            .sum();
        let mut x = (WIDTH as f32 - width) * 0.5;
        for (i, &id) in ids.iter().enumerate() {
            if i > 0 {
                x += scaled.kern(ids[i - 1], id);
            }
            let glyph = id.with_scale_and_position(size, point(x, y + scaled.ascent()));
            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                outlined.draw(|gx, gy, coverage| {
                    let px = bounds.min.x as i32 + gx as i32;
                    let py = bounds.min.y as i32 + gy as i32;
                    if px >= 0 && py >= 0 && px < WIDTH as i32 && py < HEIGHT as i32 {
                        let offset = (py as usize * WIDTH + px as usize) * 4;
                        for c in 0..4 {
                            rgba[offset + c] = (rgba[offset + c] as f32 * (1.0 - coverage)
                                + color[c] as f32 * coverage)
                                .round() as u8;
                        }
                    }
                });
            }
            x += scaled.h_advance(id);
        }
    }

    pub(super) fn render_recorder(frame: &VrTargetFrame) -> Vec<u8> {
        let mut rgba = vec![0; WIDTH * HEIGHT * 4];
        fill_rect(&mut rgba, 0, 0, WIDTH, HEIGHT, [9, 14, 21, 238]);
        let (label, color) = match frame.state {
            VrGuideState::Prepare => ("STARTING IN", [255, 195, 102, 255]),
            VrGuideState::Paused => ("PAUSED", [255, 195, 102, 255]),
            _ => ("RECORDING", [101, 227, 172, 255]),
        };
        recorder_text(&mut rgba, 40.0, label, 36.0, color);
        recorder_text(
            &mut rgba,
            116.0,
            &frame.headline,
            64.0,
            [239, 245, 251, 255],
        );
        let n = frame.countdown.unwrap_or(0);
        let clock = if frame.state == VrGuideState::Prepare {
            n.to_string()
        } else {
            format!("{}:{:02}", n / 60, n % 60)
        };
        recorder_text(&mut rgba, 210.0, &clock, 144.0, [239, 245, 251, 255]);
        recorder_text(
            &mut rgba,
            384.0,
            "Look around freely",
            38.0,
            [167, 187, 203, 255],
        );
        recorder_text(
            &mut rgba,
            450.0,
            "Blink and rest whenever needed",
            30.0,
            [167, 187, 203, 255],
        );
        fill_rect(&mut rgba, 48, 528, WIDTH - 48, 536, [31, 43, 56, 255]);
        fill_rect(
            &mut rgba,
            48,
            528,
            48 + (WIDTH - 96) * usize::from(frame.progress_percent.unwrap_or(0).min(100)) / 100,
            536,
            color,
        );
        rgba
    }

    fn render(frame: &VrTargetFrame) -> Vec<u8> {
        let mut rgba = vec![0u8; WIDTH * HEIGHT * 4];
        fill_rect(&mut rgba, 0, 0, WIDTH, HEIGHT, [6, 10, 17, 245]);
        fill_rect(&mut rgba, 0, 0, WIDTH, 88, [11, 21, 32, 255]);
        fill_rect(&mut rgba, 0, HEIGHT - 64, WIDTH, HEIGHT, [11, 21, 32, 255]);
        let state_color = match frame.state {
            VrGuideState::Ready => [75, 223, 255, 255],
            VrGuideState::Prepare => [255, 194, 77, 255],
            VrGuideState::Recording => [85, 235, 151, 255],
            VrGuideState::Paused => [255, 194, 77, 255],
            VrGuideState::Waiting => [255, 105, 105, 255],
            VrGuideState::Adjust => [75, 223, 255, 255],
        };
        fill_rect(&mut rgba, 0, 82, WIDTH, 88, state_color);
        draw_text_centered(&mut rgba, 20, &frame.headline, 4, state_color);

        if let Some(percent) = frame.progress_percent {
            fill_rect(
                &mut rgba,
                0,
                HEIGHT - 70,
                WIDTH,
                HEIGHT - 64,
                [30, 44, 58, 255],
            );
            fill_rect(
                &mut rgba,
                0,
                HEIGHT - 70,
                WIDTH * usize::from(percent.min(100)) / 100,
                HEIGHT - 64,
                state_color,
            );
        }

        if let Some(countdown) = frame.countdown {
            draw_text_centered(&mut rgba, 108, &countdown.to_string(), 12, state_color);
            draw_eye_pose(
                &mut rgba,
                frame.eye_pose,
                EYE_POSE_CENTER_Y,
                [240, 244, 249, 255],
                state_color,
            );
            draw_wrapped_centered(
                &mut rgba,
                370,
                900,
                &frame.instruction,
                4,
                [240, 244, 249, 255],
            );
        } else if let Some(target) = frame
            .target
            .filter(|_| matches!(frame.eye_pose, VrEyePose::None | VrEyePose::Open))
        {
            let [x, y] = target.screen_xy();
            let cx = (WIDTH as f32 * (0.5 + x * 0.45)).round() as i32;
            let cy = (HEIGHT as f32 * (0.5 + y * 0.43)).round() as i32;
            circle(&mut rgba, cx, cy, 25, [245, 249, 255, 255], false);
            circle(&mut rgba, cx, cy, 15, state_color, true);
            draw_wrapped_centered(
                &mut rgba,
                102,
                860,
                &frame.instruction,
                4,
                [240, 244, 249, 255],
            );
        } else if frame.eye_pose != VrEyePose::None {
            draw_eye_pose(
                &mut rgba,
                frame.eye_pose,
                EYE_POSE_CENTER_Y,
                [240, 244, 249, 255],
                state_color,
            );
            draw_wrapped_centered(
                &mut rgba,
                370,
                900,
                &frame.instruction,
                4,
                [240, 244, 249, 255],
            );
        } else {
            draw_wrapped_centered(
                &mut rgba,
                170,
                900,
                &frame.instruction,
                5,
                [240, 244, 249, 255],
            );
        }
        draw_wrapped_centered(
            &mut rgba,
            HEIGHT - 60,
            960,
            &frame.footer,
            3,
            [171, 185, 199, 255],
        );
        rgba
    }

    fn fill_rect(
        rgba: &mut [u8],
        left: usize,
        top: usize,
        right: usize,
        bottom: usize,
        color: [u8; 4],
    ) {
        for y in top.min(HEIGHT)..bottom.min(HEIGHT) {
            for x in left.min(WIDTH)..right.min(WIDTH) {
                let offset = (y * WIDTH + x) * 4;
                rgba[offset..offset + 4].copy_from_slice(&color);
            }
        }
    }

    fn put_pixel(rgba: &mut [u8], x: i32, y: i32, color: [u8; 4]) {
        if x < 0 || y < 0 || x >= WIDTH as i32 || y >= HEIGHT as i32 {
            return;
        }
        let offset = (y as usize * WIDTH + x as usize) * 4;
        rgba[offset..offset + 4].copy_from_slice(&color);
    }

    fn circle(rgba: &mut [u8], cx: i32, cy: i32, radius: i32, color: [u8; 4], fill: bool) {
        let inner = (radius - 4).max(0);
        for y in -radius..=radius {
            for x in -radius..=radius {
                let d = x * x + y * y;
                if d <= radius * radius && (fill || d >= inner * inner) {
                    put_pixel(rgba, cx + x, cy + y, color);
                }
            }
        }
    }

    fn draw_eye_pose(rgba: &mut [u8], pose: VrEyePose, cy: i32, color: [u8; 4], accent: [u8; 4]) {
        let (left, right) = match pose {
            VrEyePose::None => return,
            VrEyePose::Open | VrEyePose::Blink => (100, 100),
            VrEyePose::HalfOpen => (45, 45),
            VrEyePose::Closed => (0, 0),
            VrEyePose::Slow(openness) => (openness, openness),
            VrEyePose::LeftWink => (0, 100),
            VrEyePose::RightWink => (100, 0),
        };
        draw_text(rgba, 350, (cy - 118) as usize, "L", 3, accent);
        draw_text(rgba, 636, (cy - 118) as usize, "R", 3, accent);
        draw_single_eye(rgba, 365, cy, left, color, accent);
        draw_single_eye(rgba, 659, cy, right, color, accent);
        if pose == VrEyePose::Blink {
            draw_text_centered(rgba, (cy + 83) as usize, "RELAX BETWEEN BLINKS", 2, accent);
        }
    }

    fn draw_single_eye(
        rgba: &mut [u8],
        cx: i32,
        cy: i32,
        openness: u8,
        color: [u8; 4],
        accent: [u8; 4],
    ) {
        if openness <= 8 {
            line(rgba, cx - 112, cy - 5, cx - 45, cy + 10, color, 5);
            line(rgba, cx - 45, cy + 10, cx + 45, cy + 10, color, 5);
            line(rgba, cx + 45, cy + 10, cx + 112, cy - 5, color, 5);
            return;
        }
        let ry = 12 + i32::from(openness.min(100)) * 53 / 100;
        ellipse_outline(rgba, cx, cy, 112, ry, color, 5);
        circle(rgba, cx, cy, 22.min((ry - 4).max(8)), accent, true);
    }

    fn ellipse_outline(
        rgba: &mut [u8],
        cx: i32,
        cy: i32,
        rx: i32,
        ry: i32,
        color: [u8; 4],
        thickness: i32,
    ) {
        for degree in 0..360 {
            let angle = (degree as f32).to_radians();
            let x = cx + (rx as f32 * angle.cos()).round() as i32;
            let y = cy + (ry as f32 * angle.sin()).round() as i32;
            circle(rgba, x, y, thickness, color, true);
        }
    }

    fn line(
        rgba: &mut [u8],
        mut x0: i32,
        mut y0: i32,
        x1: i32,
        y1: i32,
        color: [u8; 4],
        thickness: i32,
    ) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            circle(rgba, x0, y0, thickness, color, true);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = error * 2;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn draw_text_centered(rgba: &mut [u8], y: usize, text: &str, scale: usize, color: [u8; 4]) {
        let text = text.to_ascii_uppercase();
        let width = text.chars().count() * 9 * scale;
        draw_text(
            rgba,
            WIDTH.saturating_sub(width) / 2,
            y,
            &text,
            scale,
            color,
        );
    }

    fn draw_wrapped_centered(
        rgba: &mut [u8],
        start_y: usize,
        max_width: usize,
        text: &str,
        scale: usize,
        color: [u8; 4],
    ) {
        let max_chars = (max_width / (9 * scale)).max(8);
        let mut lines = Vec::<String>::new();
        let mut line = String::new();
        for word in text.to_ascii_uppercase().split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > max_chars {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            lines.push(line);
        }
        for (index, line) in lines.iter().enumerate() {
            draw_text_centered(rgba, start_y + index * 12 * scale, line, scale, color);
        }
    }

    fn draw_text(rgba: &mut [u8], x: usize, y: usize, text: &str, scale: usize, color: [u8; 4]) {
        for (char_index, ch) in text.chars().enumerate() {
            let Some(glyph) = BASIC_FONTS.get(ch) else {
                continue;
            };
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..8 {
                    if bits & (1 << column) == 0 {
                        continue;
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            put_pixel(
                                rgba,
                                (x + (char_index * 9 + column) * scale + sx) as i32,
                                (y + row * scale + sy) as i32,
                                color,
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn world_pose_abi_and_transparent_dot_texture() {
            assert_eq!(std::mem::size_of::<TrackedPose>(), 80);
            assert_eq!(std::mem::size_of::<HmdMatrix34>(), 48);
            let pixels = super::super::world_target_rgba();
            assert_eq!(pixels.len(), 128 * 128 * 4);
            assert_eq!(pixels[3], 0);
            assert_eq!(
                &pixels[(64 * 128 + 64) * 4..(64 * 128 + 64) * 4 + 4],
                &[255, 255, 255, 255]
            );
        }

        #[test]
        fn reads_escaped_runtime_paths() {
            let text =
                r#"{"runtime":["C:\\Program Files (x86)\\Steam\\steamapps\\common\\SteamVR"]}"#;
            assert_eq!(
                runtime_paths(text),
                vec![r"C:\Program Files (x86)\Steam\steamapps\common\SteamVR"]
            );
        }

        #[test]
        fn openvr_dll_candidates_are_absolute() {
            assert!(openvr_candidates().iter().all(|path| path.is_absolute()));
        }

        #[test]
        fn corner_target_uses_most_of_the_vr_canvas() {
            let image = render(&VrTargetFrame {
                target: Some(GazeTarget::UpLeft),
                state: VrGuideState::Recording,
                eye_pose: VrEyePose::Open,
                countdown: None,
                progress_percent: Some(50),
                headline: "recording".into(),
                instruction: "look at target".into(),
                footer: "test".into(),
            });
            let x = (WIDTH as f32 * 0.05).round() as usize;
            let y = (HEIGHT as f32 * 0.07).round() as usize;
            let offset = (y * WIDTH + x) * 4;
            assert_eq!(&image[offset..offset + 3], &[85, 235, 151]);
        }

        #[test]
        fn closed_pose_draws_two_large_eye_guides() {
            let image = render(&VrTargetFrame {
                target: None,
                state: VrGuideState::Prepare,
                eye_pose: VrEyePose::Closed,
                countdown: Some(3),
                progress_percent: None,
                headline: "get ready".into(),
                instruction: "close gently".into(),
                footer: "test".into(),
            });
            let left = ((302usize * WIDTH + 365) * 4)..((302usize * WIDTH + 365) * 4 + 3);
            let right = ((302usize * WIDTH + 659) * 4)..((302usize * WIDTH + 659) * 4 + 3);
            assert_ne!(&image[left], &[6, 10, 17]);
            assert_ne!(&image[right], &[6, 10, 17]);
        }

        #[test]
        fn countdown_never_moves_the_eye_pose_guide() {
            let frame = |countdown| VrTargetFrame {
                target: None,
                state: VrGuideState::Prepare,
                eye_pose: VrEyePose::Closed,
                countdown,
                progress_percent: Some(25),
                headline: "get ready".into(),
                instruction: "close gently".into(),
                footer: "test".into(),
            };
            let steady = render(&frame(None));
            let countdown = render(&frame(Some(3)));
            for y in 230..350 {
                for x in (240..450).chain(575..785) {
                    let pixel = (y * WIDTH + x) * 4;
                    assert_eq!(
                        &steady[pixel..pixel + 4],
                        &countdown[pixel..pixel + 4],
                        "eye guide moved at ({x}, {y})"
                    );
                }
            }
        }
    }
}

#[cfg(windows)]
fn worker_main(
    rx: mpsc::Receiver<Command>,
    state: Arc<Mutex<VrOverlayState>>,
    visible: Arc<std::sync::atomic::AtomicBool>,
    recorder_layout: bool,
    feedback: Feedback,
) {
    windows_impl::run(rx, state, visible, recorder_layout, feedback)
}

/// Same pixels sent to SteamVR, exposed for offline visual verification.
#[cfg(windows)]
pub fn recorder_preview_rgba(frame: &VrTargetFrame) -> Vec<u8> {
    windows_impl::render_recorder(frame)
}
