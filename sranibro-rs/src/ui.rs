//! In-process egui dashboard — Fluent-dark, design-token driven, animated.
//!
//! Reads [`Telemetry`] from the running [`Pipeline`] directly (no HTTP). The
//! signal rail is the headline: VRCFT's #1 pain point is opacity, so SRanibro
//! always shows where data is flowing — here as a live animated pipeline plus a
//! one-line diagnostic banner with the fix. Built entirely from `theme` tokens.

use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicIsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Id, Pos2, Rect, Sense, Stroke};

use crate::blink_timing_fit::{
    commit_request as blink_timing_commit_request, FitInputs as BlinkTimingFitInputs,
    Fitter as BlinkTimingFitter, Status as BlinkTimingFitStatus,
};
use crate::brow_calib::{self, BrowCalib, Status as BrowStatus};
use crate::brow_fitrun::{BrowFitter, Status as FitStatus};
use crate::brow_train::{BrowTrainer, Status as TrainStatus, TrainInputs};
use crate::calib_session::{
    BlinkTimingChange, CalibrationView, CommitPermit, EndpointChange, GazeEyelidChange,
    RecipeDescriptor, SessionDescriptor, SessionKind, WinkChange, RECIPES,
};
use crate::config::{Config, EyelidResponseProfile, GazeCorrection, GazeSource, WideSource};
use crate::endpoint_fit::{
    commit_request as endpoint_commit_request, EndpointFitInputs, EndpointFitStatus, EndpointFitter,
};
use crate::gaze_eyelid_fit::{
    commit_request as gaze_eyelid_commit_request, FitInputs as GazeEyelidFitInputs,
    Fitter as GazeEyelidFitter, Status as GazeEyelidFitStatus,
};
use crate::gaze_residual_calib::{
    CaptureAction as GazeResidualAction, CaptureProtocol, GazeResidualCapture,
    Status as GazeResidualStatus,
};
use crate::geometry_calib::{
    CapturePlan, GeometryCapture, SampleFamily, SampleKind, SharedEvidence,
    Status as GeometryCaptureStatus,
};
use crate::geometry_fitrun::{
    FitInputs as GeometryFitInputs, GeometryFitResult, GeometryFitter, PhotometricFitInputs,
    PhotometricFitResult, PhotometricFitter, PhotometricStatus, Status as GeometryFitStatus,
};
use crate::output::BrokenEyeStatus;
use crate::pipeline::{EyeFrame, Pipeline, Telemetry};
use crate::recording_audio::{Cue as RecordingCue, RecordingAudio};
use crate::reseat_assist::{
    Assist as WearingPositionAssist, GazePoint as ReseatGazePoint, Guidance as ReseatGuidance,
    ReferenceContext as ReseatReferenceContext,
};
use crate::theme::*;
use crate::vr_research_overlay::{VrEyePose, VrGuideState, VrResearchOverlay, VrTargetFrame};
use crate::wear_memory::{
    CaptureReason as WearCaptureReason, Context as WearMemoryContext, Event as WearMemoryEvent,
    LiveCalibration as WearLiveCalibration, Memory as WearMemory,
};
use crate::wide_calib::{Status as WideCalibStatus, WideCalib};
use crate::wide_fitrun::{FitInputs as WideFitInputs, Status as WideFitStatus, WideFitter};
use crate::wink_fit::{
    commit_request as wink_commit_request, EyeOutcome, FitInputs as WinkFitInputs,
    Fitter as WinkFitter, Status as WinkFitStatus,
};

// The app deliberately keeps its dark borderless chrome. On Windows the blank
// title-bar rectangle is nevertheless exposed as a genuine non-client caption
// through WM_NCHITTEST. This is materially different from asking winit to begin
// a drag after egui's movement threshold: Windows owns the anchor from the first
// button-down, including physical/logical conversion across mixed-DPI monitors.
#[cfg(windows)]
static NATIVE_DRAG_HWND: AtomicIsize = AtomicIsize::new(0);
#[cfg(windows)]
static NATIVE_DRAG_OLD_PROC: AtomicIsize = AtomicIsize::new(0);
#[cfg(windows)]
static NATIVE_DRAG_ACTIVE: AtomicBool = AtomicBool::new(false);
#[cfg(windows)]
static NATIVE_DRAG_LEFT: AtomicI32 = AtomicI32::new(0);
#[cfg(windows)]
static NATIVE_DRAG_TOP: AtomicI32 = AtomicI32::new(0);
#[cfg(windows)]
static NATIVE_DRAG_RIGHT: AtomicI32 = AtomicI32::new(0);
#[cfg(windows)]
static NATIVE_DRAG_BOTTOM: AtomicI32 = AtomicI32::new(0);

#[cfg(windows)]
unsafe extern "system" fn sranibro_window_proc(
    hwnd: windows_sys::Win32::Foundation::HWND,
    message: u32,
    wparam: windows_sys::Win32::Foundation::WPARAM,
    lparam: windows_sys::Win32::Foundation::LPARAM,
) -> windows_sys::Win32::Foundation::LRESULT {
    use windows_sys::Win32::Foundation::POINT;
    use windows_sys::Win32::Graphics::Gdi::ScreenToClient;
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        CallWindowProcW, DefWindowProcW, HTCAPTION, HTCLIENT, WM_ENTERSIZEMOVE, WM_EXITSIZEMOVE,
        WM_NCDESTROY, WM_NCHITTEST,
    };

    let old = NATIVE_DRAG_OLD_PROC.load(Ordering::Acquire);
    let inherited = || {
        if old == 0 {
            DefWindowProcW(hwnd, message, wparam, lparam)
        } else {
            let previous: windows_sys::Win32::UI::WindowsAndMessaging::WNDPROC =
                std::mem::transmute(old);
            CallWindowProcW(previous, hwnd, message, wparam, lparam)
        }
    };

    match message {
        WM_NCHITTEST => {
            let base = inherited();
            if base != HTCLIENT as isize {
                return base;
            }
            // WM_NCHITTEST coordinates are signed physical screen pixels.
            let mut point = POINT {
                x: (lparam as u16 as i16) as i32,
                y: ((lparam >> 16) as u16 as i16) as i32,
            };
            if ScreenToClient(hwnd, &mut point) != 0
                && point.x >= NATIVE_DRAG_LEFT.load(Ordering::Relaxed)
                && point.x < NATIVE_DRAG_RIGHT.load(Ordering::Relaxed)
                && point.y >= NATIVE_DRAG_TOP.load(Ordering::Relaxed)
                && point.y < NATIVE_DRAG_BOTTOM.load(Ordering::Relaxed)
            {
                return HTCAPTION as isize;
            }
            base
        }
        WM_ENTERSIZEMOVE => {
            NATIVE_DRAG_ACTIVE.store(true, Ordering::Release);
            inherited()
        }
        WM_EXITSIZEMOVE => {
            NATIVE_DRAG_ACTIVE.store(false, Ordering::Release);
            inherited()
        }
        WM_NCDESTROY => {
            NATIVE_DRAG_ACTIVE.store(false, Ordering::Release);
            NATIVE_DRAG_HWND.store(0, Ordering::Release);
            inherited()
        }
        _ => inherited(),
    }
}

#[cfg(windows)]
fn install_native_caption_drag(cc: &eframe::CreationContext<'_>) {
    use raw_window_handle::{HasWindowHandle, RawWindowHandle};
    use windows_sys::Win32::UI::WindowsAndMessaging::{SetWindowLongPtrW, GWLP_WNDPROC};

    let Ok(handle) = cc.window_handle() else {
        return;
    };
    let RawWindowHandle::Win32(handle) = handle.as_raw() else {
        return;
    };
    let hwnd = handle.hwnd.get();
    if NATIVE_DRAG_HWND
        .compare_exchange(0, hwnd, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return;
    }
    let previous = unsafe {
        SetWindowLongPtrW(
            hwnd as _,
            GWLP_WNDPROC,
            sranibro_window_proc as *const () as isize,
        )
    };
    if previous == 0 {
        NATIVE_DRAG_HWND.store(0, Ordering::Release);
        return;
    }
    NATIVE_DRAG_OLD_PROC.store(previous, Ordering::Release);
}

#[cfg(not(windows))]
fn install_native_caption_drag(_cc: &eframe::CreationContext<'_>) {}

fn native_window_drag_active() -> bool {
    #[cfg(windows)]
    return NATIVE_DRAG_ACTIVE.load(Ordering::Acquire);
    #[cfg(not(windows))]
    return false;
}

fn set_native_caption_rect(rect: Rect, pixels_per_point: f32) {
    #[cfg(windows)]
    {
        NATIVE_DRAG_LEFT.store(
            (rect.left() * pixels_per_point).round() as i32,
            Ordering::Relaxed,
        );
        NATIVE_DRAG_TOP.store(
            (rect.top() * pixels_per_point).round() as i32,
            Ordering::Relaxed,
        );
        NATIVE_DRAG_RIGHT.store(
            (rect.right() * pixels_per_point).round() as i32,
            Ordering::Relaxed,
        );
        NATIVE_DRAG_BOTTOM.store(
            (rect.bottom() * pixels_per_point).round() as i32,
            Ordering::Relaxed,
        );
    }
    #[cfg(not(windows))]
    let _ = (rect, pixels_per_point);
}
#[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"));
#[cfg(feature = "psvr2-only")]
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"), " · PSVR2");
#[cfg(feature = "xr5-only")]
const APP_VERSION_LABEL: &str = concat!("v", env!("CARGO_PKG_VERSION"), " · XR5");
#[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
const APP_WINDOW_TITLE: &str = concat!("SRanibro v", env!("CARGO_PKG_VERSION"));
#[cfg(feature = "psvr2-only")]
const APP_WINDOW_TITLE: &str = concat!("SRanibro PSVR2 Beta v", env!("CARGO_PKG_VERSION"));
#[cfg(feature = "xr5-only")]
const APP_WINDOW_TITLE: &str = concat!("SRanibro XR5 Beta v", env!("CARGO_PKG_VERSION"));

#[cfg(any(test, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
const STANDARD_DEVICE_OPTIONS: &[&str] = &[
    "auto",
    "pimax_vr4",
    "varjo",
    "varjo_mjpeg",
    "starvr",
    "psvr2",
];

#[cfg(any(test, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
fn device_option_label(device: &str) -> &str {
    match crate::config::canonical_device_key(device).as_str() {
        "auto" => "Auto detect",
        "pimax_vr4" => "Pimax Crystal / Crystal Super (VR4)",
        "varjo" => "Varjo (native)",
        "varjo_mjpeg" => "Varjo Eye Streamer",
        "starvr" => "StarVR One",
        "psvr2" => "PlayStation VR2 (PSVR2Toolkit)",
        _ => device,
    }
}

// A visible wgpu swapchain competes with the VR compositor even when average GPU
// usage looks small. Normal tracking telemetry is intentionally presentation-light;
// acquisition, inference and output keep their native clocks on worker threads.
const TRACKING_UI_REPAINT_INTERVAL: Duration = Duration::from_millis(100);
// A visible preview is an explicit opt-in diagnostic, so present every new camera
// frame instead of imposing the normal low-rate dashboard cadence. Hidden previews
// still do no frame clones or texture uploads.
const LIVE_EYE_TEXTURE_INTERVAL: Duration = Duration::from_micros(8_333);

fn vr_eye_pose(action: Option<GazeResidualAction>, slow_openness: Option<u8>) -> VrEyePose {
    match action {
        Some(GazeResidualAction::RelaxedOpen) => VrEyePose::Open,
        Some(GazeResidualAction::HalfOpen) => VrEyePose::HalfOpen,
        Some(GazeResidualAction::GentleClosed) => VrEyePose::Closed,
        Some(GazeResidualAction::SlowCloseOpen) => VrEyePose::Slow(slow_openness.unwrap_or(100)),
        Some(GazeResidualAction::NaturalBlink) => VrEyePose::Blink,
        Some(GazeResidualAction::LeftWink) => VrEyePose::LeftWink,
        Some(GazeResidualAction::RightWink) => VrEyePose::RightWink,
        None => VrEyePose::None,
    }
}

fn vr_pose_instruction(action: Option<GazeResidualAction>, target_visible: bool) -> &'static str {
    match action {
        Some(GazeResidualAction::RelaxedOpen) if target_visible => {
            "KEEP BOTH EYES OPEN - MOVE ONLY YOUR EYES"
        }
        Some(GazeResidualAction::RelaxedOpen) => "KEEP BOTH EYES COMFORTABLY OPEN",
        Some(GazeResidualAction::HalfOpen) => "HOLD BOTH EYES HALF OPEN",
        Some(GazeResidualAction::GentleClosed) => "CLOSE GENTLY - DO NOT SQUEEZE",
        Some(GazeResidualAction::SlowCloseOpen) => "FOLLOW THE EYELID ANIMATION",
        Some(GazeResidualAction::NaturalBlink) => "BLINK NATURALLY - RELAX BETWEEN BLINKS",
        Some(GazeResidualAction::LeftWink) => "CLOSE LEFT EYE - KEEP RIGHT EYE OPEN",
        Some(GazeResidualAction::RightWink) => "CLOSE RIGHT EYE - KEEP LEFT EYE OPEN",
        None => "RELAX BOTH EYES AND HOLD STILL",
    }
}

fn geometry_vr_eye_pose(kind: Option<SampleKind>, target_open: Option<f32>) -> VrEyePose {
    match kind {
        Some(SampleKind::Neutral | SampleKind::HoldoutNeutral) => VrEyePose::Open,
        Some(SampleKind::GazeSweep | SampleKind::HoldoutGazeSweep) => VrEyePose::Open,
        Some(SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen) => VrEyePose::HalfOpen,
        Some(SampleKind::Closed | SampleKind::HoldoutClosed) => VrEyePose::Closed,
        Some(SampleKind::SlowClose | SampleKind::HoldoutSlowClose) => {
            VrEyePose::Slow((target_open.unwrap_or(1.0).clamp(0.0, 1.0) * 100.0).round() as u8)
        }
        Some(SampleKind::NaturalBlinks | SampleKind::HoldoutNaturalBlinks) => VrEyePose::Blink,
        Some(SampleKind::LeftWink | SampleKind::HoldoutLeftWink) => VrEyePose::LeftWink,
        Some(SampleKind::RightWink | SampleKind::HoldoutRightWink) => VrEyePose::RightWink,
        None => VrEyePose::None,
    }
}

#[derive(PartialEq, Clone, Copy)]
enum Page {
    Dashboard,
    Calibration,
    Console,
    Settings,
}

/// The window / taskbar icon, baked as raw RGBA (256×256) next to the .ico so no PNG
/// decoder is needed at runtime. Matches the exe's embedded icon.
fn app_icon() -> egui::IconData {
    const RGBA: &[u8] = include_bytes!("../assets/sranibro_rgba.bin");
    egui::IconData {
        rgba: RGBA.to_vec(),
        width: 256,
        height: 256,
    }
}

fn native_options() -> eframe::NativeOptions {
    let mut wgpu_options = eframe::egui_wgpu::WgpuConfiguration::default();
    // CursorMoved can arrive much faster than the monitor refresh rate. Use an
    // unambiguous blocking VSync mode and allow at most one queued frame, so an
    // interactive widget cannot build a long GPU presentation backlog.
    wgpu_options.present_mode = wgpu::PresentMode::Fifo;
    wgpu_options.desired_maximum_frame_latency = Some(1);
    eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // Fixed, non-resizable: sized to hold the capped content column
            // (NAV_W + margins + MAX_W) with a small balanced margin.
            .with_inner_size([WIN_W, WIN_H])
            // Fixed: the layout is fixed-metric and positions some elements from the
            // absolute window height (e.g. the nav LED), so a resizable window would
            // misplace them. True responsive layout is a separate refactor.
            .with_resizable(false)
            .with_maximize_button(false)
            // The native Windows title bar does not follow the app's dark theme.
            // Keep the window borderless and provide drag/minimize/close controls
            // in SRanibro's own top bar instead.
            .with_decorations(false)
            .with_icon(app_icon())
            .with_title(APP_WINDOW_TITLE),
        // wgpu backend: DX12->Vulkan->GL fallback, robust on VMs/RDP/weak drivers
        // where the default glow/OpenGL-3 path blank-windows or fails to launch.
        renderer: eframe::Renderer::Wgpu,
        vsync: true,
        wgpu_options,
        ..Default::default()
    }
}

#[cfg(windows)]
fn renderer_gpu_context(
    cc: &eframe::CreationContext<'_>,
) -> Option<crate::ml::eyelid_model::EyelidGpuContext> {
    let state = cc.wgpu_render_state.as_ref()?;
    let info = state.adapter.get_info();
    eprintln!(
        "[ui:gpu] adapter={} backend={:?}; FIFO renderer isolated from EyeNet compute queue",
        info.name, info.backend
    );
    // A shared queue prevented two producers from overfilling the driver, but synchronous
    // and mapped readbacks were then serialized behind the 120 Hz preview swapchain and
    // reduced EyeNet to 20-30 Hz. FIFO + one-frame latency already bounds the renderer;
    // keep EyeNet on its small private compute queue so presentation cannot delay ML.
    None
}

/// Compatibility entry point for callers that already constructed a pipeline.
/// New product startup uses [`run_ui_from_config`] so the renderer can be configured
/// before the tracking engine starts, without reconnecting the eye tracker afterward.
pub fn run_ui(
    pipeline: Pipeline,
    be_status: Option<Arc<BrokenEyeStatus>>,
    startup_notice: Option<String>,
) -> eframe::Result<()> {
    eframe::run_native(
        APP_WINDOW_TITLE,
        native_options(),
        Box::new(move |cc| {
            crate::theme::apply(&cc.egui_ctx);
            install_native_caption_drag(cc);
            let gpu_context = renderer_gpu_context(cc);
            Ok(Box::new(App::new(
                pipeline,
                be_status,
                startup_notice,
                gpu_context,
            )))
        }),
    )
}

/// Product GUI startup. The renderer is initialized first with a bounded FIFO queue,
/// then the tracking engine starts with its independent EyeNet compute queue.
#[cfg(windows)]
pub fn run_ui_from_config(config: Config) -> eframe::Result<()> {
    eframe::run_native(
        APP_WINDOW_TITLE,
        native_options(),
        Box::new(move |cc| {
            crate::theme::apply(&cc.egui_ctx);
            install_native_caption_drag(cc);
            let gpu_context = renderer_gpu_context(cc);
            let engine = match crate::engine::build_engine_with_gpu(&config, gpu_context.clone()) {
                Ok(engine) => engine,
                Err(error) => {
                    eprintln!("engine start failed: {error}");
                    crate::engine::build_recovery_engine(&config, error.to_string())?
                }
            };
            Ok(Box::new(App::new(
                engine.pipeline,
                engine.be_status,
                engine.startup_notice,
                gpu_context,
            )))
        }),
    )
}

/// Editable copies of the asset paths + device, bound to the Settings text fields.
/// Empty string == "not set"; applied back into [`Config`] on reload.
#[derive(Clone)]
struct SettingsEdit {
    sranipal_dir: String,
    /// Optional eyebrow model (BROWNET1 file baked from the user's calibrated model).
    brow_model: String,
    /// Optional task-tagged XR5 image-based EyeWide model.
    wide_model: String,
    wide_source: WideSource,
    /// XR5-only EyeChip gaze provider. Source changes are applied with an engine
    /// reload so per-eye and combined vectors can never flap within one session.
    gaze_source: GazeSource,
    /// B-2 train inputs: the venv-with-torch python + the user's vr_eyebrow project dir.
    python_exe: String,
    vr_eyebrow_dir: String,
    device: String,
    osc_host: String,
    osc_port: u16,
    eye_image_host: String,
    eye_image_port: u16,
}

impl SettingsEdit {
    fn from_cfg(c: &Config) -> Self {
        let g = |o: &Option<String>| o.clone().unwrap_or_default();
        Self {
            sranipal_dir: g(&c.assets.sranipal_dir),
            brow_model: g(&c.assets.brow_model),
            wide_model: g(&c.assets.wide_model),
            wide_source: c.hmd.wide_source,
            gaze_source: c.gaze_source_for("pimax_xr5"),
            python_exe: g(&c.assets.python_exe),
            vr_eyebrow_dir: g(&c.assets.vr_eyebrow_dir),
            device: c.hmd.device.clone(),
            osc_host: c.output.osc_host.clone(),
            osc_port: c.output.osc_port,
            eye_image_host: c.output.eye_image_host.clone(),
            eye_image_port: c.output.eye_image_port,
        }
    }
}

struct GazeCenterCapture {
    started: Instant,
    paused_at: Option<Instant>,
    sum_deg: [[f64; 2]; 2],
    count: u32,
    last_timestamp_us: u64,
}

impl GazeCenterCapture {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            paused_at: None,
            sum_deg: [[0.0; 2]; 2],
            count: 0,
            last_timestamp_us: 0,
        }
    }

    fn pause(&mut self, now: Instant) {
        if self.paused_at.is_none() {
            self.paused_at = Some(now);
        }
    }

    fn resume(&mut self, now: Instant, latest_timestamp_us: u64) {
        let Some(paused_at) = self.paused_at.take() else {
            return;
        };
        let paused_for = now.saturating_duration_since(paused_at);
        self.started = self.started.checked_add(paused_for).unwrap_or(now);
        self.last_timestamp_us = latest_timestamp_us;
    }

    fn suspend_for(&mut self, duration: Duration, now: Instant, latest_timestamp_us: u64) {
        if self.paused_at.is_none() && !duration.is_zero() {
            self.started = self.started.checked_add(duration).unwrap_or(now);
            self.last_timestamp_us = latest_timestamp_us;
        }
    }
}

#[derive(Clone, Copy)]
enum GazeRangeAxis {
    Horizontal,
    Vertical,
}

/// Percent values currently stored for each eye on one gaze axis.  Keep this
/// separate from the slider value so an old asymmetric/per-eye calibration is
/// never presented as if both eyes had the averaged setting.
fn gaze_range_percent(correction: &GazeCorrection, axis: GazeRangeAxis) -> [f32; 2] {
    let scale = match axis {
        GazeRangeAxis::Horizontal => correction.scale_x,
        GazeRangeAxis::Vertical => correction.scale_y,
    };
    [scale[0] * 100.0, scale[1] * 100.0]
}

fn shared_gaze_range_slider_value(correction: &GazeCorrection, axis: GazeRangeAxis) -> f32 {
    let per_eye = gaze_range_percent(correction, axis);
    ((per_eye[0] + per_eye[1]) * 0.5).clamp(25.0, 250.0)
}

/// Set a deliberately shared avatar movement range.  Moving the simple control
/// means “use this for both eyes”; advanced per-eye centre, the other axis, and
/// vergence remain untouched.
fn set_shared_gaze_range_percent(
    correction: &mut GazeCorrection,
    axis: GazeRangeAxis,
    percent: f32,
) {
    let scale = (percent.clamp(25.0, 250.0) / 100.0).clamp(0.25, 2.5);
    match axis {
        GazeRangeAxis::Horizontal => correction.scale_x = [scale; 2],
        GazeRangeAxis::Vertical => correction.scale_y = [scale; 2],
    }
    correction.enabled = true;
}

fn gaze_range_is_mixed(per_eye_percent: [f32; 2]) -> bool {
    (per_eye_percent[0] - per_eye_percent[1]).abs() > 0.5
}

fn supports_frontal_photometric_correction(device: &str) -> bool {
    crate::config::supports_photometric_fit(device)
}

/// Long guided recordings exist to rescue the non-frontal XR5 image path. A
/// frontal SRanipal-compatible HMD is tuned directly from its live eye image and
/// response rails, so presenting the recording menu there adds work without a
/// useful first-line benefit.
fn shows_advanced_recording_calibration(device: &str) -> bool {
    crate::config::canonical_device_key(device) == "pimax_xr5"
}

fn frontal_photometric_profile_name(device: &str) -> &'static str {
    match crate::config::canonical_device_key(device).as_str() {
        "pimax_vr4" | "pimax_dll" => "PIMAX VR4",
        "varjo" | "varjo_mjpeg" => "VARJO",
        _ => "FRONTAL",
    }
}

fn calibration_duration(seconds: f32) -> String {
    if seconds >= 60.0 {
        format!("{:.1} min", seconds / 60.0)
    } else {
        format!("{seconds:.0} sec")
    }
}

fn calibration_meta_label(text: &str) -> egui::RichText {
    egui::RichText::new(text)
        .size(10.0 * S)
        .strong()
        .color(ACCENT)
}

fn calibration_meta_cell(ui: &mut egui::Ui, text: &str) {
    ui.add_sized(
        [46.0 * S, 18.0 * S],
        egui::Label::new(calibration_meta_label(text)).truncate(),
    );
}

fn calibration_open_button(ui: &mut egui::Ui) -> bool {
    let button = egui::Button::new(
        egui::RichText::new("Open")
            .size(10.0 * S)
            .strong()
            .color(ACCENT),
    )
    .fill(ACCENT_BG)
    .stroke(egui::Stroke::new(1.0, ACCENT))
    .min_size(egui::vec2(54.0 * S, 22.0 * S));
    ui.add(button)
        .on_hover_text("Open instructions, progress, and results. Recording does not start yet.")
        .clicked()
}

fn calibration_entry_footer(ui: &mut egui::Ui, meta: &str, value: &str, hover: &str) -> bool {
    let mut start_clicked = false;
    let wide = ui.available_width() >= 480.0 * S;

    let show_change = |ui: &mut egui::Ui| {
        calibration_meta_cell(ui, meta);
        ui.label(prose(value)).on_hover_text(hover);
    };

    if wide {
        ui.horizontal(|ui| {
            show_change(ui);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                start_clicked = calibration_open_button(ui);
            });
        });
    } else {
        ui.horizontal_wrapped(show_change);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                start_clicked = calibration_open_button(ui);
            });
        });
    }

    start_clicked
}

/// Consistent, scannable entry used by Full setup and Single check. Technical
/// contract detail remains available as a hover instead of filling the page.
fn calibration_session_entry(
    ui: &mut egui::Ui,
    step: Option<usize>,
    descriptor: &SessionDescriptor,
) -> bool {
    let mut start_clicked = false;
    egui::Frame::default()
        .fill(INNER)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .rounding(R_INNER)
        .inner_margin(egui::Margin::symmetric(10.0 * S, 7.0 * S))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0 * S;
            ui.horizontal(|ui| {
                if let Some(step) = step {
                    calibration_meta_cell(ui, &format!("STEP {step}"));
                }
                ui.label(
                    egui::RichText::new(descriptor.title)
                        .size(11.5 * S)
                        .strong()
                        .color(TEXT1),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    start_clicked = calibration_open_button(ui);
                    ui.add_space(4.0 * S);
                    ui.label(
                        egui::RichText::new(calibration_duration(descriptor.estimated_seconds))
                            .monospace()
                            .size(9.0 * S)
                            .color(TEXT3),
                    );
                });
            });
            ui.label(prose(descriptor.checks));
            ui.horizontal_wrapped(|ui| {
                ui.label(calibration_meta_label("AFTER APPLY"));
                ui.label(prose(descriptor.after_apply));
            });
        });
    start_clicked
}

fn calibration_problem_entry(
    ui: &mut egui::Ui,
    recipe: RecipeDescriptor,
    descriptor: &SessionDescriptor,
) -> bool {
    let mut start_clicked = false;
    egui::Frame::default()
        .fill(INNER)
        .stroke(egui::Stroke::new(1.0, BORDER))
        .rounding(R_INNER)
        .inner_margin(egui::Margin::symmetric(10.0 * S, 7.0 * S))
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 4.0 * S;
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(recipe.title)
                        .size(11.5 * S)
                        .strong()
                        .color(TEXT1),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(calibration_duration(descriptor.estimated_seconds))
                            .monospace()
                            .size(9.0 * S)
                            .color(TEXT3),
                    );
                });
            });
            ui.label(prose(recipe.symptom));
            ui.horizontal_wrapped(|ui| {
                ui.label(calibration_meta_label("AFTER APPLY"));
                ui.label(prose(descriptor.after_apply));
            });
            start_clicked =
                calibration_entry_footer(ui, "CHECK", descriptor.title, descriptor.checks);
        });
    start_clicked
}

fn calibration_detail_intro(ui: &mut egui::Ui, descriptor: &SessionDescriptor) {
    ui.label(prose(descriptor.checks));
    ui.add_space(5.0 * S);
    ui.horizontal_wrapped(|ui| {
        ui.label(calibration_meta_label("CHANGES"));
        ui.label(prose(&descriptor.may_change.user_text()))
            .on_hover_text(format!(
                "Everything else stays unchanged: {}",
                descriptor.never_changes.user_text()
            ));
        ui.add_space(SP2);
        ui.label(calibration_meta_label("TIME"));
        ui.label(
            egui::RichText::new(calibration_duration(descriptor.estimated_seconds))
                .monospace()
                .size(9.0 * S)
                .color(TEXT3),
        );
    });
}

fn eyelid_reason_badge(ui: &mut egui::Ui, reason: &crate::core::eye_state::ClosureReason) {
    use crate::core::eye_state::ClosureReason;

    let (text, color) = match reason {
        ClosureReason::Normal => ("NORMAL", OK),
        ClosureReason::BlinkPending => ("BLINK PENDING", WARN),
        ClosureReason::FastBlink => ("FAST BLINK", ACCENT),
        ClosureReason::NativeDisable => ("NATIVE DISABLE", WARN),
        ClosureReason::Wink => ("WINK", ACCENT),
        ClosureReason::TrackingLost => ("TRACKING LOST", ERR),
    };
    egui::Frame::default()
        .fill(color.gamma_multiply(0.14))
        .stroke(Stroke::new(1.0, color.gamma_multiply(0.65)))
        .rounding(4.0 * S)
        .inner_margin(egui::Margin::symmetric(6.0 * S, 2.0 * S))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(text)
                    .monospace()
                    .strong()
                    .size(9.0 * S)
                    .color(color),
            );
        });
}

fn eyelid_live_metric(ui: &mut egui::Ui, name: &str, value: f32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(name).size(10.0 * S).color(TEXT2));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0}%", value.clamp(0.0, 1.0) * 100.0))
                    .monospace()
                    .strong()
                    .size(10.0 * S)
                    .color(TEXT1),
            );
        });
    });
    let width = ui.available_width();
    ui.add(
        egui::ProgressBar::new(value.clamp(0.0, 1.0))
            .desired_width(width)
            .show_percentage(),
    );
}

fn eyelid_live_eye(
    ui: &mut egui::Ui,
    eye_name: &str,
    diag: &crate::core::eye_state::EyelidLiveDiag,
) {
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(eye_name)
                .monospace()
                .strong()
                .size(10.0 * S)
                .color(TEXT1),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            eyelid_reason_badge(ui, &diag.reason);
        });
    });
    eyelid_live_metric(ui, "Avatar openness", diag.final_openness);
}

/// Project-Babble-style raw-model rail. The live signal moves between two draggable
/// endpoint handles; persisted values remain relative to the relaxed-open baseline.
struct WideRailEdit<'a> {
    input: f32,
    output: f32,
    start: &'a mut f32,
    full: &'a mut f32,
    entry_ref: f32,
    full_ref: f32,
}

fn wide_raw_range(
    diag: &crate::core::eye_state::EyelidLiveDiag,
    start: f32,
    full: f32,
) -> (f32, f32) {
    let entry = diag.wide_entry_ref.clamp(0.0, 1.0 - 1.0e-3);
    let ceiling = diag.wide_full_ref.clamp(0.0, 1.0).max(entry + 1.0e-3);
    let span = (ceiling - entry).max(1.0e-3);
    (
        entry + start.clamp(0.0, 1.0) * span,
        entry + full.clamp(0.0, 1.0) * span,
    )
}

fn openness_marker_value(raw_openness: f32, wide_entry: Option<f32>) -> f32 {
    wide_entry.map_or(raw_openness, |entry| raw_openness.min(entry))
}

/// ch1 openness and ch2 Wide are different model heads, so their raw 0..1 values
/// are not a shared physical coordinate. Preserve the familiar combined rail while
/// guaranteeing its semantic ordering: the red no-Wide band ends just beyond the
/// normal-open handle and the Wide lane retains enough visible travel.
fn wide_display_bounds(open_ref: f32, raw_entry: f32, raw_full: f32) -> (f32, f32) {
    let entry = raw_entry.max(open_ref + 0.015).clamp(0.0, 0.985);
    let full = raw_full.max(entry + 0.05).clamp(entry + 0.001, 1.0);
    (entry, full)
}

fn split_handle_hits(mut left: Rect, mut right: Rect) -> (Rect, Rect) {
    let split = (left.center().x + right.center().x) * 0.5;
    left.max.x = left.max.x.min(split);
    right.min.x = right.min.x.max(split);
    (left, right)
}

fn dragged_handle_x(ui: &egui::Ui, response: &egui::Response, center: f32) -> Option<f32> {
    if !response.dragged() {
        return None;
    }
    let key = response.id.with("grab-offset");
    if response.drag_started() {
        let origin = ui
            .input(|input| input.pointer.press_origin())
            .map_or(center, |p| p.x);
        ui.ctx()
            .data_mut(|data| data.insert_temp(key, origin - center));
    }
    let offset = ui
        .ctx()
        .data(|data| data.get_temp::<f32>(key))
        .unwrap_or(0.0);
    response
        .interact_pointer_pos()
        .map(|point| point.x - offset)
}

fn eyelid_threshold_rail(
    ui: &mut egui::Ui,
    eye: usize,
    diag: &crate::core::eye_state::EyelidLiveDiag,
    open_offset: &mut f32,
    closed_depth: &mut f32,
    enabled: bool,
    mut wide: Option<WideRailEdit<'_>>,
    comparison_raw: Option<f32>,
) -> (bool, bool, bool, bool) {
    let baseline = diag.effective_baseline;
    let open_ref = diag.effective_open_ref;
    let closed_ref = diag.effective_closed_ref;
    // ch1 openness and ch2 EyeWide share one physical response axis, but retain
    // separate live markers: green is confined to the normal range and orange owns
    // the Wide range. Custom XR5 Wide remains on its own score rail.
    let combined_wide = wide.is_some();
    let height = if combined_wide { 74.0 * S } else { 52.0 * S };
    let (outer, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    // Keep endpoint chevrons and their 30 px grab targets inside each eye column.
    // Without this inset, LEFT 1.0 and RIGHT 0.0 meet at the column boundary and
    // egui clips the half of each handle that extends into the neighbouring column.
    let edge_inset = 16.0 * S;
    let openness_top = if combined_wide { 38.0 * S } else { 19.0 * S };
    let rail = Rect::from_min_max(
        pos2(outer.left() + edge_inset, outer.top() + openness_top),
        pos2(
            outer.right() - edge_inset,
            outer.top() + openness_top + 12.0 * S,
        ),
    );
    let to_x = |value: f32| rail.left() + rail.width() * value.clamp(0.0, 1.0);
    let open_x = to_x(open_ref);
    let closed_x = to_x(closed_ref);
    // Once raw openness crosses into the ch2 Wide range, the normal-openness marker
    // stops at the boundary. Only the orange Wide marker is allowed to move beyond it.
    let wide_display_entry = wide
        .as_ref()
        .map(|wide| wide_display_bounds(open_ref, wide.entry_ref, wide.full_ref).0);
    let openness_marker = openness_marker_value(diag.raw_openness, wide_display_entry);
    let raw_x = to_x(openness_marker);

    // Keep the marker precise but make the invisible grab target generous.
    let handle_size = if combined_wide {
        vec2(30.0 * S, 24.0 * S)
    } else {
        vec2(30.0 * S, 44.0 * S)
    };
    let openness_handle_y = if combined_wide {
        rail.bottom() + 8.0 * S
    } else {
        rail.center().y
    };
    let open_hit = Rect::from_center_size(pos2(open_x, openness_handle_y), handle_size);
    let closed_hit = Rect::from_center_size(pos2(closed_x, openness_handle_y), handle_size);
    let (closed_hit, open_hit) = split_handle_hits(closed_hit, open_hit);
    let open_response = ui.interact(
        open_hit,
        ui.id().with(("eyelid-open-point", eye)),
        if enabled {
            Sense::drag()
        } else {
            Sense::hover()
        },
    );
    let closed_response = ui.interact(
        closed_hit,
        ui.id().with(("eyelid-closed-point", eye)),
        if enabled {
            Sense::drag()
        } else {
            Sense::hover()
        },
    );

    let mut open_changed = false;
    let mut closed_changed = false;
    let mut wide_start_changed = false;
    let mut wide_full_changed = false;
    if enabled && open_response.dragged() {
        if let Some(x) = dragged_handle_x(ui, &open_response, open_x) {
            let raw = ((x - rail.left()) / rail.width()).clamp(0.0, 1.0);
            let max_offset = (*closed_depth - EyelidResponseProfile::MIN_MANUAL_RANGE)
                .min(EyelidResponseProfile::OPEN_POINT_OFFSET_MAX);
            *open_offset = (baseline - raw).clamp(
                EyelidResponseProfile::OPEN_POINT_OFFSET_MIN,
                max_offset.max(EyelidResponseProfile::OPEN_POINT_OFFSET_MIN),
            );
            open_changed = true;
        }
    }
    if enabled && closed_response.dragged() {
        if let Some(x) = dragged_handle_x(ui, &closed_response, closed_x) {
            let raw = ((x - rail.left()) / rail.width()).clamp(0.0, 1.0);
            let min_depth = (*open_offset + EyelidResponseProfile::MIN_MANUAL_RANGE)
                .max(EyelidResponseProfile::CLOSED_POINT_DEPTH_MIN);
            *closed_depth = (baseline - raw).clamp(
                min_depth.min(EyelidResponseProfile::CLOSED_POINT_DEPTH_MAX),
                EyelidResponseProfile::CLOSED_POINT_DEPTH_MAX,
            );
            closed_changed = true;
        }
    }

    let painter = ui.painter();
    painter.rect_filled(rail, rail.height() * 0.5, INNER);
    let openness_range = Rect::from_min_max(
        pos2(closed_x.min(open_x), rail.top()),
        pos2(closed_x.max(open_x), rail.bottom()),
    );
    painter.rect_filled(
        openness_range,
        rail.height() * 0.5,
        Color32::from_rgb(28, 92, 112),
    );
    if let Some(wide) = wide.as_mut() {
        let wide_live_color = Color32::from_rgb(255, 174, 66);
        let (entry, ceiling) = wide_display_bounds(open_ref, wide.entry_ref, wide.full_ref);
        let span = (ceiling - entry).max(1.0e-3);
        let wide_lane = rail;
        let wide_band = Rect::from_min_max(
            wide_lane.left_top(),
            pos2(wide_lane.right(), wide_lane.center().y),
        );
        let wide_to_x = |value: f32| wide_lane.left() + wide_lane.width() * value.clamp(0.0, 1.0);
        let entry_x = wide_to_x(entry);
        let ceiling_x = wide_to_x(ceiling);
        let start_raw = entry + (*wide.start).clamp(0.0, 1.0) * span;
        let full_raw = entry + (*wide.full).clamp(0.0, 1.0) * span;
        let wide_start_x = wide_to_x(start_raw);
        let wide_full_x = wide_to_x(full_raw);

        let wide_handle_size = vec2(30.0 * S, 30.0 * S);
        let (start_hit, full_hit) = split_handle_hits(
            Rect::from_center_size(
                pos2(wide_start_x, wide_lane.top() - 8.0 * S),
                wide_handle_size,
            ),
            Rect::from_center_size(
                pos2(wide_full_x, wide_lane.top() - 8.0 * S),
                wide_handle_size,
            ),
        );
        let start_response = ui.interact(
            start_hit,
            ui.id().with(("wide-open-start", eye)),
            Sense::drag(),
        );
        let full_response = ui.interact(
            full_hit,
            ui.id().with(("wide-open-full", eye)),
            Sense::drag(),
        );
        if start_response.dragged() {
            if let Some(x) = dragged_handle_x(ui, &start_response, wide_start_x) {
                let raw = ((x - wide_lane.left()) / wide_lane.width()).clamp(entry, ceiling);
                let value = ((raw - entry) / span).clamp(0.0, 1.0);
                *wide.start = value.clamp(
                    EyelidResponseProfile::EXPRESSION_START_MIN,
                    (*wide.full - EyelidResponseProfile::MIN_EXPRESSION_RANGE)
                        .max(EyelidResponseProfile::EXPRESSION_START_MIN),
                );
                wide_start_changed = true;
            }
        }
        if full_response.dragged() {
            if let Some(x) = dragged_handle_x(ui, &full_response, wide_full_x) {
                let raw = ((x - wide_lane.left()) / wide_lane.width()).clamp(entry, ceiling);
                let value = ((raw - entry) / span).clamp(0.0, 1.0);
                *wide.full = value.clamp(
                    (*wide.start + EyelidResponseProfile::MIN_EXPRESSION_RANGE)
                        .min(EyelidResponseProfile::EXPRESSION_FULL_MAX),
                    EyelidResponseProfile::EXPRESSION_FULL_MAX,
                );
                wide_full_changed = true;
            }
        }

        painter.text(
            pos2(outer.left(), outer.top()),
            Align2::LEFT_TOP,
            format!("OPEN ML {:.3}", diag.raw_openness),
            FontId::monospace(9.0 * S),
            OK,
        );
        painter.text(
            pos2(outer.right(), outer.top()),
            Align2::RIGHT_TOP,
            format!(
                "WIDE CH2 {:.3}   OUT {:.0}%",
                wide.input,
                wide.output * 100.0
            ),
            FontId::monospace(9.0 * S),
            wide_live_color,
        );
        // The red lane is intentionally unavailable: SRanipal Wide cannot begin
        // below its raw-openness entry point. It makes the normal-open/Wide
        // boundary visible without placing another handle on top of Openness.
        painter.rect_filled(
            Rect::from_min_max(wide_band.left_top(), pos2(entry_x, wide_band.bottom())),
            wide_band.height() * 0.5,
            Color32::from_rgb(73, 29, 38),
        );
        painter.rect_filled(
            Rect::from_min_max(
                pos2(wide_start_x, wide_band.top()),
                pos2(wide_full_x, wide_band.bottom()),
            ),
            wide_band.height() * 0.5,
            Color32::from_rgb(38, 126, 158),
        );
        painter.line_segment(
            [
                pos2(entry_x, wide_lane.top() - 3.0 * S),
                pos2(entry_x, wide_lane.center().y),
            ],
            Stroke::new(2.0 * S, ERR),
        );
        painter.line_segment(
            [
                pos2(ceiling_x, wide_lane.top()),
                pos2(ceiling_x, wide_lane.center().y),
            ],
            Stroke::new(1.0 * S, TEXT3),
        );
        for x in [wide_start_x, wide_full_x] {
            painter.line_segment(
                [
                    pos2(x, wide_lane.top() - 4.0 * S),
                    pos2(x, wide_lane.center().y),
                ],
                Stroke::new(3.0 * S, ACCENT),
            );
            let tip = pos2(x, wide_lane.top() - 2.0 * S);
            painter.line_segment(
                [pos2(x - 6.0 * S, wide_lane.top() - 9.0 * S), tip],
                Stroke::new(3.0 * S, ACCENT),
            );
            painter.line_segment(
                [tip, pos2(x + 6.0 * S, wide_lane.top() - 9.0 * S)],
                Stroke::new(3.0 * S, ACCENT),
            );
        }
        let input_x = wide_to_x(entry + wide.input.clamp(0.0, 1.0) * span);
        painter.line_segment(
            [
                pos2(input_x, wide_lane.top() - 6.0 * S),
                pos2(input_x, wide_lane.center().y),
            ],
            Stroke::new(3.0 * S, wide_live_color),
        );
    }

    painter.rect_stroke(rail, rail.height() * 0.5, Stroke::new(1.0, BORDER));

    let threshold_color = if enabled { ACCENT } else { TEXT3 };
    for x in [closed_x, open_x] {
        painter.line_segment(
            [
                pos2(x, rail.top() - 5.0 * S),
                pos2(x, rail.bottom() + 5.0 * S),
            ],
            Stroke::new(3.0 * S, threshold_color),
        );
        let tip = pos2(x, rail.bottom() + 2.0 * S);
        let left = pos2(x - 6.0 * S, rail.bottom() + 9.0 * S);
        let right = pos2(x + 6.0 * S, rail.bottom() + 9.0 * S);
        painter.line_segment([left, tip], Stroke::new(3.0 * S, threshold_color));
        painter.line_segment([tip, right], Stroke::new(3.0 * S, threshold_color));
    }
    let raw_marker_top = if combined_wide {
        rail.center().y
    } else {
        rail.top() - 9.0 * S
    };
    painter.line_segment(
        [
            pos2(raw_x, raw_marker_top),
            pos2(raw_x, rail.bottom() + 9.0 * S),
        ],
        Stroke::new(3.0 * S, OK),
    );
    if let Some(value) = comparison_raw.filter(|value| value.is_finite()) {
        let comparison_x = to_x(value);
        let comparison_color = Color32::from_rgb(225, 105, 255);
        painter.line_segment(
            [
                pos2(comparison_x, rail.top() - 7.0 * S),
                pos2(comparison_x, rail.bottom() + 7.0 * S),
            ],
            Stroke::new(2.0 * S, comparison_color),
        );
    }
    if !combined_wide {
        painter.text(
            outer.right_top(),
            Align2::RIGHT_TOP,
            format!("ML {:.3}", diag.raw_openness),
            FontId::monospace(9.0 * S),
            OK,
        );
    }

    (
        open_changed,
        closed_changed,
        wide_start_changed,
        wide_full_changed,
    )
}

fn expression_axis_position(kind: &str, value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if kind == "squeeze" {
        const PAD: f32 = 0.10;
        PAD + (1.0 - 2.0 * PAD) * value.sqrt()
    } else {
        value
    }
}

fn expression_axis_value(kind: &str, position: f32) -> f32 {
    let position = position.clamp(0.0, 1.0);
    if kind == "squeeze" {
        const PAD: f32 = 0.10;
        let unpadded = ((position - PAD) / (1.0 - 2.0 * PAD)).clamp(0.0, 1.0);
        unpadded * unpadded
    } else {
        position
    }
}

fn expression_threshold_rail(
    ui: &mut egui::Ui,
    kind: &'static str,
    eye: usize,
    input: f32,
    output: f32,
    start: &mut f32,
    full: &mut f32,
) -> (bool, bool) {
    // Live IN/OUT and the rail are separate; endpoint values are printed below.
    let height = 68.0 * S;
    let (outer, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let edge_inset = 16.0 * S;
    let rail = Rect::from_min_max(
        pos2(outer.left() + edge_inset, outer.top() + 34.0 * S),
        pos2(outer.right() - edge_inset, outer.top() + 46.0 * S),
    );
    // Squeeze usually occupies only the bottom of the model's 0..1 output. A
    // square-root display axis expands that useful region without changing the
    // stored threshold or the value sent to the avatar.
    let to_x = |value: f32| rail.left() + rail.width() * expression_axis_position(kind, value);
    let start_x = to_x(*start);
    let full_x = to_x(*full);
    let input_x = to_x(input);
    let handle_size = vec2(30.0 * S, 44.0 * S);
    let (start_hit, full_hit) = split_handle_hits(
        Rect::from_center_size(pos2(start_x, rail.center().y), handle_size),
        Rect::from_center_size(pos2(full_x, rail.center().y), handle_size),
    );
    let start_response = ui.interact(
        start_hit,
        ui.id().with(("expression-start", kind, eye)),
        Sense::drag(),
    );
    let full_response = ui.interact(
        full_hit,
        ui.id().with(("expression-full", kind, eye)),
        Sense::drag(),
    );

    let mut start_changed = false;
    let mut full_changed = false;
    if start_response.dragged() {
        if let Some(x) = dragged_handle_x(ui, &start_response, start_x) {
            let position = ((x - rail.left()) / rail.width()).clamp(0.0, 1.0);
            let value = expression_axis_value(kind, position);
            *start = value.clamp(
                EyelidResponseProfile::EXPRESSION_START_MIN,
                (*full - EyelidResponseProfile::MIN_EXPRESSION_RANGE)
                    .max(EyelidResponseProfile::EXPRESSION_START_MIN),
            );
            start_changed = true;
        }
    }
    if full_response.dragged() {
        if let Some(x) = dragged_handle_x(ui, &full_response, full_x) {
            let position = ((x - rail.left()) / rail.width()).clamp(0.0, 1.0);
            let value = expression_axis_value(kind, position);
            *full = value.clamp(
                (*start + EyelidResponseProfile::MIN_EXPRESSION_RANGE)
                    .min(EyelidResponseProfile::EXPRESSION_FULL_MAX),
                EyelidResponseProfile::EXPRESSION_FULL_MAX,
            );
            full_changed = true;
        }
    }

    let painter = ui.painter();
    painter.rect_filled(rail, rail.height() * 0.5, INNER);
    painter.rect_filled(
        Rect::from_min_max(pos2(start_x, rail.top()), pos2(full_x, rail.bottom())),
        rail.height() * 0.5,
        Color32::from_rgb(28, 92, 112),
    );
    painter.rect_stroke(rail, rail.height() * 0.5, Stroke::new(1.0, BORDER));
    for x in [start_x, full_x] {
        painter.line_segment(
            [
                pos2(x, rail.top() - 5.0 * S),
                pos2(x, rail.bottom() + 5.0 * S),
            ],
            Stroke::new(3.0 * S, ACCENT),
        );
        let tip = pos2(x, rail.bottom() + 2.0 * S);
        painter.line_segment(
            [pos2(x - 6.0 * S, rail.bottom() + 9.0 * S), tip],
            Stroke::new(3.0 * S, ACCENT),
        );
        painter.line_segment(
            [tip, pos2(x + 6.0 * S, rail.bottom() + 9.0 * S)],
            Stroke::new(3.0 * S, ACCENT),
        );
    }
    painter.line_segment(
        [
            pos2(input_x, rail.top() - 9.0 * S),
            pos2(input_x, rail.bottom() + 9.0 * S),
        ],
        Stroke::new(4.0 * S, OK),
    );
    painter.circle_filled(pos2(input_x, rail.top() - 8.0 * S), 4.0 * S, OK);
    painter.text(
        pos2(outer.left(), outer.top()),
        Align2::LEFT_TOP,
        format!("OUT {:.2}", output),
        FontId::monospace(9.0 * S),
        TEXT2,
    );
    painter.text(
        pos2(outer.right(), outer.top()),
        Align2::RIGHT_TOP,
        format!("IN {:.3}", input),
        FontId::monospace(9.0 * S),
        OK,
    );
    (start_changed, full_changed)
}

fn memory_save_is_uncorrected(
    enabled: bool,
    suspended: bool,
    target_present: bool,
    effective: [f32; 2],
    calibrated: [f32; 2],
) -> bool {
    (suspended || !enabled)
        && !target_present
        && (0..2).all(|eye| {
            effective[eye].is_finite() && (effective[eye] - calibrated[eye]).abs() < 0.0001
        })
}

fn restore_thresholds_only(
    mut current: EyelidResponseProfile,
    saved: EyelidResponseProfile,
) -> EyelidResponseProfile {
    current.manual_range = saved.manual_range;
    current.open_point_offset = saved.open_point_offset;
    current.closed_point_depth = saved.closed_point_depth;
    current
}

#[cfg(test)]
mod gaze_range_tests {
    use super::*;

    #[test]
    fn memory_save_requires_verified_uncorrected_mode_and_settled_baseline() {
        let base = [0.5, 0.6];
        assert!(!memory_save_is_uncorrected(true, false, false, base, base));
        assert!(!memory_save_is_uncorrected(true, true, true, base, base));
        assert!(!memory_save_is_uncorrected(
            true,
            true,
            false,
            [0.51, 0.6],
            base
        ));
        assert!(!memory_save_is_uncorrected(
            true,
            true,
            false,
            [f32::NAN, 0.6],
            base
        ));
        assert!(memory_save_is_uncorrected(true, true, false, base, base));
        assert!(memory_save_is_uncorrected(false, false, false, base, base));
    }

    #[test]
    fn threshold_undo_preserves_later_expression_and_curve_edits() {
        let saved = EyelidResponseProfile::default();
        let mut current = saved;
        current.open_point_offset = [0.09, 0.1];
        current.closed_point_depth = [0.3, 0.4];
        current.wide_start = [0.2, 0.3];
        current.squeeze_full = [0.7, 0.8];
        current.curve_mid_output = [0.35, 0.65];
        let restored = restore_thresholds_only(current, saved);
        assert_eq!(restored.open_point_offset, saved.open_point_offset);
        assert_eq!(restored.closed_point_depth, saved.closed_point_depth);
        assert_eq!(restored.wide_start, current.wide_start);
        assert_eq!(restored.squeeze_full, current.squeeze_full);
        assert_eq!(restored.curve_mid_output, current.curve_mid_output);
    }

    #[test]
    fn close_threshold_handles_have_disjoint_hit_regions() {
        for distance in [0.0, 1.0, 8.0, 29.0, 60.0] {
            let (left, right) = split_handle_hits(
                Rect::from_center_size(pos2(100.0, 20.0), vec2(30.0, 30.0)),
                Rect::from_center_size(pos2(100.0 + distance, 20.0), vec2(30.0, 30.0)),
            );
            assert!(left.right() <= right.left());
            assert!(left.width() > 0.0 && right.width() > 0.0);
        }
    }

    #[test]
    fn interactive_ui_never_queues_more_than_one_vsynced_frame() {
        let options = native_options();
        assert!(options.vsync);
        assert_eq!(options.wgpu_options.present_mode, wgpu::PresentMode::Fifo);
        assert_eq!(options.wgpu_options.desired_maximum_frame_latency, Some(1));
    }

    #[test]
    fn squeeze_display_expands_low_input_without_changing_threshold_values() {
        let position = expression_axis_position("squeeze", 0.04);
        assert!((position - 0.26).abs() < 1.0e-6);
        assert!((expression_axis_value("squeeze", position) - 0.04).abs() < 1.0e-6);
        assert!((expression_axis_position("squeeze", 0.0) - 0.10).abs() < 1.0e-6);
        assert!((expression_axis_position("squeeze", 1.0) - 0.90).abs() < 1.0e-6);
        assert_eq!(expression_axis_position("wide", 0.04), 0.04);
        assert_eq!(expression_axis_value("wide", 0.04), 0.04);
    }

    #[test]
    fn openness_marker_stops_at_wide_entry() {
        assert_eq!(openness_marker_value(0.48, Some(0.60)), 0.48);
        assert_eq!(openness_marker_value(0.60, Some(0.60)), 0.60);
        assert_eq!(openness_marker_value(0.82, Some(0.60)), 0.60);
        assert_eq!(openness_marker_value(0.82, None), 0.82);
    }

    #[test]
    fn combined_wide_lane_never_crosses_the_normal_open_handle() {
        let (entry, full) = wide_display_bounds(0.72, 0.61, 0.69);
        assert!(entry > 0.72);
        assert!(full > entry);

        let (already_ordered, _) = wide_display_bounds(0.42, 0.63, 0.80);
        assert!((already_ordered - 0.63).abs() < 1.0e-6);
    }

    #[test]
    fn sranipal_wide_overlay_maps_into_raw_openness_above_its_entry() {
        let diag = crate::core::eye_state::EyelidLiveDiag {
            wide_entry_ref: 0.62,
            wide_full_ref: 0.77,
            ..Default::default()
        };
        let (start, full) = wide_raw_range(&diag, 0.20, 0.80);
        assert!((start - 0.65).abs() < 1.0e-6);
        assert!((full - 0.74).abs() < 1.0e-6);
        assert!(start >= diag.wide_entry_ref);
        assert!(full <= diag.wide_full_ref);
    }

    #[test]
    fn sranipal_wide_overlay_handles_a_saturated_entry_without_panicking() {
        let diag = crate::core::eye_state::EyelidLiveDiag {
            wide_entry_ref: 1.0,
            wide_full_ref: 0.0,
            ..Default::default()
        };
        let (start, full) = wide_raw_range(&diag, 0.0, 1.0);
        assert!(start.is_finite() && full.is_finite());
        assert!(start <= full && full <= 1.0);
    }

    #[test]
    fn frontal_photometric_card_is_scoped_to_vr4_and_varjo_paths() {
        for device in ["pimax_vr4", "pimax-vr4-dll", "varjo", "varjo_mjpeg"] {
            assert!(supports_frontal_photometric_correction(device), "{device}");
        }
        for device in ["pimax_xr5", "starvr", "vpe"] {
            assert!(!supports_frontal_photometric_correction(device), "{device}");
        }
    }

    #[test]
    fn long_recording_menu_is_only_exposed_for_the_non_frontal_xr5_path() {
        for device in ["pimax_xr5", "dream-air", "xr5"] {
            assert!(shows_advanced_recording_calibration(device), "{device}");
        }
        for device in ["pimax_vr4", "varjo", "starvr", "vpe", "psvr2"] {
            assert!(!shows_advanced_recording_calibration(device), "{device}");
        }
    }

    #[test]
    fn standard_headset_selector_does_not_offer_the_paused_xr5_route() {
        assert!(!STANDARD_DEVICE_OPTIONS.contains(&"pimax_xr5"));
        assert!(!STANDARD_DEVICE_OPTIONS
            .iter()
            .any(|device| { crate::config::canonical_device_key(device) == "pimax_xr5" }));
    }

    #[test]
    fn unified_headset_selector_exposes_psvr2_with_a_human_readable_label() {
        assert!(STANDARD_DEVICE_OPTIONS.contains(&"psvr2"));
        assert_eq!(
            device_option_label("psvr2"),
            "PlayStation VR2 (PSVR2Toolkit)"
        );
        assert_eq!(
            device_option_label("pimax_vr4"),
            "Pimax Crystal / Crystal Super (VR4)"
        );
    }

    #[test]
    fn saved_calibration_capture_routes_to_its_matching_analysis() {
        let expected = [
            SessionKind::SafeGeometry,
            SessionKind::Photometric,
            SessionKind::EyelidEndpoints,
            SessionKind::Winks,
            SessionKind::NaturalBlinks,
        ];
        assert_eq!(
            CalibrationCapturePurpose::ALL.map(CalibrationCapturePurpose::session_kind),
            expected
        );
    }

    #[test]
    fn shared_gaze_range_maps_percent_to_both_eyes_and_auto_enables() {
        let mut correction = GazeCorrection {
            enabled: false,
            offset_x_deg: [1.25, -2.5],
            offset_y_deg: [3.0, -4.0],
            scale_y: [0.8, 1.2],
            vergence_deg: 2.25,
            ..GazeCorrection::default()
        };

        for (percent, expected) in [(25.0, 0.25), (100.0, 1.0), (250.0, 2.5)] {
            correction.enabled = false;
            set_shared_gaze_range_percent(&mut correction, GazeRangeAxis::Horizontal, percent);
            assert!(correction.enabled);
            assert_eq!(correction.scale_x, [expected; 2]);
            assert_eq!(correction.scale_y, [0.8, 1.2]);
            assert_eq!(correction.offset_x_deg, [1.25, -2.5]);
            assert_eq!(correction.offset_y_deg, [3.0, -4.0]);
            assert_eq!(correction.vergence_deg, 2.25);
        }
    }

    #[test]
    fn shared_gaze_range_reports_mixed_per_eye_values_without_hiding_them() {
        let correction = GazeCorrection {
            scale_x: [0.25, 2.5],
            ..GazeCorrection::default()
        };
        let values = gaze_range_percent(&correction, GazeRangeAxis::Horizontal);
        assert_eq!(values, [25.0, 250.0]);
        assert!(gaze_range_is_mixed(values));
        assert_eq!(
            shared_gaze_range_slider_value(&correction, GazeRangeAxis::Horizontal),
            137.5
        );
    }

    #[test]
    fn gaze_center_pause_excludes_the_hidden_interval() {
        let base = Instant::now();
        let mut capture = GazeCenterCapture::new();
        capture.started = base;
        capture.pause(base + Duration::from_millis(100));
        capture.resume(base + Duration::from_millis(1_100), 77);
        assert_eq!(capture.started, base + Duration::from_secs(1));
        assert_eq!(capture.last_timestamp_us, 77);
    }
}

#[derive(Clone)]
struct GazeResidualSnapshot {
    device_key: String,
    unit_id: String,
    geometry: [crate::core::types::MlGeometry; 2],
    mirrors: [bool; 2],
    despeckle: crate::core::types::DespeckleParams,
    flatten: crate::core::types::FlattenParams,
    brightness: crate::core::types::BrightnessNorm,
    photometric: crate::core::types::PhotometricCorrection,
    mapping: crate::config::EyeMapping,
    wide_source: WideSource,
    gaze_source: GazeSource,
    eyelid_model_crc32: Option<u32>,
    eyelid_model_bytes: Option<u64>,
    steamvr_target_requested: bool,
}

struct GeometryExportWork {
    path: PathBuf,
    partial_path: PathBuf,
    dataset: crate::geometry_calib::GeometryDataset,
    metadata: String,
}

struct GeometryExportResult {
    path: PathBuf,
    dataset: crate::geometry_calib::GeometryDataset,
    result: Result<(), String>,
}

struct GazeResidualExportWork {
    path: PathBuf,
    partial_path: PathBuf,
    dataset: crate::geometry_calib::GeometryDataset,
    metadata: String,
}

struct GazeResidualExportResult {
    path: PathBuf,
    dataset: crate::geometry_calib::GeometryDataset,
    result: Result<(), String>,
}

fn reseat_reference_context(config: &Config, pipeline: &Pipeline) -> ReseatReferenceContext {
    let serial = config
        .dream_air_profile_for(&pipeline.device_key)
        .and_then(|profile| profile.eyechip_serial.clone())
        .or_else(crate::device::usb::peek_serial);
    let unit_id = crate::diagnostics::pseudonymous_unit_id(serial.as_deref());
    let mapping = config.mapping_for(&pipeline.device_key);
    let geometry = config.geometry_for(&pipeline.device_key);
    ReseatReferenceContext::new(
        &pipeline.device_key,
        unit_id,
        crate::reseat_assist::image_fingerprint(mapping, geometry),
    )
}

fn wear_memory_context(config: &Config, pipeline: &Pipeline) -> WearMemoryContext {
    let serial = config
        .dream_air_profile_for(&pipeline.device_key)
        .and_then(|profile| profile.eyechip_serial.clone())
        .or_else(crate::device::usb::peek_serial);
    let unit_id = crate::diagnostics::pseudonymous_unit_id(serial.as_deref());
    WearMemoryContext::new(
        &pipeline.device_key,
        unit_id,
        wear_memory_fingerprint(config, pipeline),
    )
}

fn wear_memory_fingerprint(config: &Config, pipeline: &Pipeline) -> u64 {
    // Calibration values are meaningful only under the exact image/model domain
    // that produced them. Raw-image matching itself survives lighting changes, but
    // a changed filter, geometry, mapping or model must start a separate memory.
    let material = format!(
        "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
        config.mapping_for(&pipeline.device_key),
        *pipeline.geometry.lock().unwrap(),
        *pipeline.despeckle.lock().unwrap(),
        *pipeline.flatten.lock().unwrap(),
        *pipeline.brightness.lock().unwrap(),
        *pipeline.photometric_correction.lock().unwrap(),
        pipeline.eyelid_model_identity,
        pipeline.uses_right_eye_left_head(),
    );
    let mut hash = 0xcbf29ce484222325u64;
    for byte in material.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationCapturePurpose {
    Geometry,
    Photometric,
    EyelidEndpoints,
    Winks,
    NaturalBlinks,
}

impl CalibrationCapturePurpose {
    #[cfg(test)]
    const ALL: [Self; 5] = [
        Self::Geometry,
        Self::Photometric,
        Self::EyelidEndpoints,
        Self::Winks,
        Self::NaturalBlinks,
    ];

    const fn session_kind(self) -> SessionKind {
        match self {
            Self::Geometry => SessionKind::SafeGeometry,
            Self::Photometric => SessionKind::Photometric,
            Self::EyelidEndpoints => SessionKind::EyelidEndpoints,
            Self::Winks => SessionKind::Winks,
            Self::NaturalBlinks => SessionKind::NaturalBlinks,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CalibrationDetail {
    // The old record-once transaction remains compiled so an in-flight analysis
    // can still finish safely, but the simplified UI no longer starts one.
    #[allow(dead_code)]
    InitialSetup,
    PythonEyelidDataset,
    ReseatAssist,
    Atomic(SessionKind),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnifiedStage {
    Capturing,
    Saving,
    Preprocessing,
    Endpoints,
    ReviewEndpoints,
    Gaze,
    Winks,
    Blinks,
    FinalReview,
    Complete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnifiedPreprocessing {
    None,
    Geometry,
    Photometric,
}

struct UnifiedCalibrationRun {
    protocol: CaptureProtocol,
    stage: UnifiedStage,
    preprocessing: UnifiedPreprocessing,
    evidence: Option<SharedEvidence>,
    snapshot: Option<GazeResidualSnapshot>,
    /// Endpoint inputs that were live when recording started. Unlike `endpoints`,
    /// this never changes when a candidate is staged.
    frozen_endpoints: Option<crate::core::eye_state::CalibStore>,
    endpoints: Option<crate::core::eye_state::CalibStore>,
    endpoint_apply: Option<crate::endpoint_fit::EndpointApplyRequest>,
    geometry: Option<[crate::core::types::MlGeometry; 2]>,
    photometric: Option<crate::core::types::PhotometricCorrection>,
    base_gaze_profile: crate::core::types::GazeEyelidProfile,
    base_wink_profile: crate::core::types::WinkProfile,
    base_blink_profile: crate::core::types::BlinkTimingProfile,
    gaze_profile: Option<crate::core::types::GazeEyelidProfile>,
    wink_profile: Option<crate::core::types::WinkProfile>,
    blink_profile: Option<crate::core::types::BlinkTimingProfile>,
    open_deadzone: f32,
    notes: Vec<String>,
    blocked: Option<String>,
}

impl UnifiedCalibrationRun {
    fn new(
        protocol: CaptureProtocol,
        endpoints: Option<crate::core::eye_state::CalibStore>,
        open_deadzone: f32,
        base_gaze_profile: crate::core::types::GazeEyelidProfile,
        base_wink_profile: crate::core::types::WinkProfile,
        base_blink_profile: crate::core::types::BlinkTimingProfile,
    ) -> Self {
        Self {
            protocol,
            stage: UnifiedStage::Capturing,
            preprocessing: UnifiedPreprocessing::None,
            evidence: None,
            snapshot: None,
            frozen_endpoints: endpoints,
            endpoints,
            endpoint_apply: None,
            geometry: None,
            photometric: None,
            base_gaze_profile,
            base_wink_profile,
            base_blink_profile,
            gaze_profile: None,
            wink_profile: None,
            blink_profile: None,
            open_deadzone,
            notes: Vec::new(),
            blocked: None,
        }
    }

    fn staged_change_count(&self) -> usize {
        usize::from(self.geometry.is_some())
            + usize::from(self.photometric.is_some())
            + usize::from(self.endpoint_apply.is_some())
            + usize::from(self.gaze_profile.is_some())
            + usize::from(self.wink_profile.is_some())
            + usize::from(self.blink_profile.is_some())
    }
}

#[derive(Default)]
struct RecordingCueState {
    initialized: bool,
    geometry: Option<String>,
    geometry_stalled: bool,
    geometry_error: Option<String>,
    residual: Option<String>,
    residual_paused: bool,
    residual_stalled: bool,
    residual_error: Option<String>,
    wide: Option<String>,
    brow: Option<String>,
    wide_error: Option<String>,
    brow_error: Option<String>,
    diagnostic: bool,
}

/// Live editor for the per-HMD eyelid response. Slider changes reach the
/// pipeline immediately; `dirty` delays the small config-file write until the
/// current pointer gesture has ended.
struct EyelidResponsePreview {
    edit: EyelidResponseProfile,
    link_eyes: bool,
    dirty: bool,
    message: Option<(String, Color32)>,
}

#[derive(Default)]
struct WearingMemoryDraft {
    recentered: bool,
    closed_set: [bool; 2],
    wide_neutral_set: bool,
    message: Option<(String, Color32)>,
}

struct ClosedPointCapture {
    eye: usize,
    started: Instant,
    samples: Vec<f32>,
    sampling_cued: bool,
    last_generation: u64,
}

impl EyelidResponsePreview {
    fn new(profile: EyelidResponseProfile) -> Self {
        Self {
            edit: profile,
            link_eyes: (profile.close_depth_scale[0] - profile.close_depth_scale[1]).abs() < 0.001
                && (profile.curve_mid_output[0] - profile.curve_mid_output[1]).abs() < 0.001
                && (profile.open_point_offset[0] - profile.open_point_offset[1]).abs() < 0.001
                && (profile.closed_point_depth[0] - profile.closed_point_depth[1]).abs() < 0.001
                && (profile.wide_start[0] - profile.wide_start[1]).abs() < 0.001
                && (profile.wide_full[0] - profile.wide_full[1]).abs() < 0.001
                && (profile.squeeze_start[0] - profile.squeeze_start[1]).abs() < 0.001
                && (profile.squeeze_full[0] - profile.squeeze_full[1]).abs() < 0.001,
            dirty: false,
            message: None,
        }
    }
}

struct App {
    pipeline: Pipeline,
    #[cfg(windows)]
    gpu_context: Option<crate::ml::eyelid_model::EyelidGpuContext>,
    tele: Arc<Telemetry>,
    config: Config,
    /// Live-editable asset/device fields (Settings tab).
    edit: SettingsEdit,
    /// Last "Apply & reload" result message (text, color).
    reload_msg: Option<(String, Color32)>,
    reload_job: Option<std::thread::JoinHandle<std::io::Result<crate::engine::Engine>>>,
    /// Paint-only snapshot, captured once when reload begins. No live widgets or
    /// camera texture uploads run behind the progress overlay.
    reload_backdrop: Vec<egui::epaint::ClippedShape>,
    /// On-demand SRanipal discovery runs off the UI thread because a bounded
    /// Downloads/Desktop fallback may take several seconds on a large profile.
    sranipal_discovery_job: Option<mpsc::Receiver<Option<crate::sranipal_discovery::Found>>>,
    page: Page,
    last: [u64; 5],
    rates: [f32; 5],
    last_t: Instant,
    tex_l: Option<egui::TextureHandle>,
    tex_r: Option<egui::TextureHandle>,
    /// Eye textures are the expensive part of a dashboard repaint (grayscale -> RGBA
    /// conversion plus a GPU upload). Input events may make egui repaint faster than
    /// `request_repaint_after`, so throttle the uploads independently of repaint rate.
    last_eye_texture_upload: Instant,
    /// Exact source generations represented by the current dashboard textures.
    /// Input/hover repaints must not upload the same camera payload again.
    last_eye_texture_generation: [u64; 2],
    /// Preview textures for the Calibration tab's ML-input geometry card (the processed
    /// 100x100 the eye model actually sees, per eye).
    tex_ml_l: Option<egui::TextureHandle>,
    tex_ml_r: Option<egui::TextureHandle>,
    /// Textures for the ML occlusion-heatmap overlay (eye + colormap), per eye.
    tex_heat_l: Option<egui::TextureHandle>,
    tex_heat_r: Option<egui::TextureHandle>,
    /// Colormap full-scale for the heatmap (|openness delta| that maps to full colour).
    heat_vmax: f32,
    /// Scale each computed map to its robust p98 response while still showing the absolute
    /// p98/peak values, so focused regions stay visible without hiding weak maps.
    heat_auto_scale: bool,
    // diagnostic event log (stage up/down, tracking toggles)
    start: Instant,
    /// Last eframe callback. A long gap while a calibration is active means the
    /// Windows compositor/event loop owned the UI (for example native window drag),
    /// not that the wearer completed an invisible prompt.
    last_ui_update: Instant,
    /// Camera generations seen by the previous UI callback. Diagnostic stall logs use
    /// this to distinguish a blocked presentation thread from a stopped tracker.
    last_ui_frame_generations: [u64; 2],
    /// Tracks the minimized transition so the last hidden camera generations are
    /// discarded exactly once before any paused calibration resumes.
    ui_was_minimized: bool,
    /// Start time of a Windows-owned title-bar drag. While set, capture clocks are
    /// paused and expensive camera texture uploads are suppressed.
    window_drag_started_at: Option<Instant>,
    window_drag_start_generations: [u64; 2],
    window_drag_update_count: u64,
    window_drag_max_ui_gap: Duration,
    prev_ok: Option<[bool; 6]>,
    prev_paused: bool,
    /// (wall-clock "HH:MM:SS" stamp, message, color).
    events: Vec<(String, String, Color32)>,
    /// Pipeline node whose detail card is open (click to toggle).
    sel_node: Option<usize>,
    /// Whether the top PIPELINE card is expanded. Default collapsed — the header
    /// (name + "n/6 nodes ok") is the at-a-glance summary; the flow diagram is opt-in.
    pipeline_open: bool,
    /// The ML-input geometry editor modal (opened by the gear on the eye-cameras card).
    show_geom_modal: bool,
    /// Dream Air/XR5 native-gaze finishing correction modal.
    show_gaze_modal: bool,
    /// Explicit good-state capture for appearance-conditioned wearing recovery.
    show_wear_memory_modal: bool,
    /// Live-preview transaction owned by the Open / closed detail modal.
    /// `None` elsewhere so an abandoned edit cannot leak into another workflow.
    eyelid_response_preview: Option<EyelidResponsePreview>,
    /// One-second straight-ahead capture used by the gaze Center action.
    gaze_center_capture: Option<GazeCenterCapture>,
    gaze_center_msg: Option<(String, Color32)>,
    /// Calibration-modal-only source edit. Discarded on close so it cannot be
    /// committed later by an unrelated Settings reload.
    gaze_source_modal_edit: Option<GazeSource>,
    dream_air_msg: Option<(String, Color32)>,
    recording_audio: RecordingAudio,
    recording_cues: RecordingCueState,
    /// In-memory, labelled stereo capture used only by the XR5 geometry search.
    geometry_capture: GeometryCapture,
    /// Background pure-Rust candidate search + untouched-holdout validation.
    geometry_fitter: GeometryFitter,
    /// Fixed-geometry brightness/contrast/illumination search for frontal eye cameras.
    photometric_fitter: PhotometricFitter,
    /// All-HMD explicit relaxed-open / gentle-close endpoint analysis.
    endpoint_fitter: EndpointFitter,
    /// All-HMD nine-direction relaxed-eyelid compensation analysis.
    gaze_eyelid_fitter: GazeEyelidFitter,
    /// All-HMD per-eye held-wink response analysis.
    wink_fitter: WinkFitter,
    /// All-HMD natural-blink visible-bottom timing analysis.
    blink_timing_fitter: BlinkTimingFitter,
    /// Immutable user-confirmed wearing-position reference and live physical reseat
    /// guidance. This never writes tracking calibration or adaptive baselines.
    reseat_assist: WearingPositionAssist,
    /// Explicitly taught appearance -> calibration associations. Matching is
    /// passive; only the explicit good-state confirmation may create memory.
    wear_memory: WearMemory,
    /// True while the user is deliberately tuning a new wearing state. Automatic
    /// matching resumes only after that complete state is explicitly confirmed.
    wear_memory_matching_suspended: bool,
    wear_response_before_edit: Option<EyelidResponseProfile>,
    wear_baseline_before_edit: Option<[f32; 2]>,
    wear_thumbnails: Vec<(u64, [egui::TextureHandle; 2])>,
    /// In-progress endpoint checks are deliberately separate from persisted memory:
    /// Recenter and slider edits prepare a candidate but never teach it themselves.
    wear_memory_draft: WearingMemoryDraft,
    wear_closed_capture: Option<ClosedPointCapture>,
    confirm_remove_reseat_reference: bool,
    /// The workflow currently owning `geometry_capture`.
    calibration_capture_purpose: Option<CalibrationCapturePurpose>,
    /// Compact navigation for Full / symptom-guided / individual calibration.
    calibration_view: CalibrationView,
    /// Dedicated foreground guide/progress window opened from the compact list.
    calibration_detail_window: Option<CalibrationDetail>,
    /// Two-step guard for removing an applied calibration. Upstream removals can
    /// invalidate dependent eyelid fits, so their exact cascade is shown before
    /// the user confirms the write.
    pending_calibration_removal: Option<SessionKind>,
    /// Record-once workflow. Raw frames are immutable and reused by each fitter.
    unified_calibration: Option<UnifiedCalibrationRun>,
    /// Geometry active at capture start. The search is always centred on this fallback.
    geometry_capture_baseline: Option<[crate::core::types::MlGeometry; 2]>,
    /// Photometric filters are snapshotted with the baseline so edits made after capture
    /// cannot mismatch the stored per-frame brightness affine during replay.
    geometry_capture_filters: Option<(
        crate::core::types::DespeckleParams,
        crate::core::types::FlattenParams,
    )>,
    /// Fitted correction active at photometric-capture start. It remains the exact
    /// fallback and must not change until the recording has been scored.
    photometric_capture_baseline: Option<crate::core::types::PhotometricCorrection>,
    /// Automatic feedback-archive state for the current completed capture. An attempt is
    /// made exactly once on completion; a failed attempt leaves the dataset in memory and
    /// exposes a retry action instead of silently letting fit/audit consume the only copy.
    geometry_recording_export_attempted: bool,
    geometry_recording_path: Option<PathBuf>,
    geometry_recording_export_job: Option<mpsc::Receiver<GeometryExportResult>>,
    geometry_recording_export_thread: Option<std::thread::JoinHandle<()>>,
    /// Independent nine-point research capture. It never feeds Safe Geometry Fit and is
    /// saved only as evidence for offline landmark/residual analysis.
    gaze_residual_capture: GazeResidualCapture,
    gaze_residual_export_attempted: bool,
    gaze_residual_recording_path: Option<PathBuf>,
    gaze_residual_snapshot: Option<GazeResidualSnapshot>,
    gaze_residual_export_job: Option<mpsc::Receiver<GazeResidualExportResult>>,
    gaze_residual_export_thread: Option<std::thread::JoinHandle<()>>,
    gaze_residual_window_elevated: bool,
    /// Optional head-locked wide-angle target. OpenVR is initialized lazily on
    /// its own thread and the desktop target remains the unconditional fallback.
    vr_research_overlay: VrResearchOverlay,
    /// Unsaved geometry restored when the user exits candidate preview.
    geometry_preview_restore: Option<[crate::core::types::MlGeometry; 2]>,
    /// Previous persisted geometry retained for one-click rollback after Apply.
    geometry_rollback: Option<[crate::core::types::MlGeometry; 2]>,
    /// Previous persisted fitted correction retained for one-click rollback.
    photometric_rollback: Option<crate::core::types::PhotometricCorrection>,
    /// Explicit acknowledgement required before persisting a candidate that failed
    /// untouched-holdout validation. Reset whenever a new capture or fit begins.
    geometry_unvalidated_ack: bool,
    /// Which tab of the ML-input modal is active: 0 = Image (crop/stretch/rotate), 1 = Filter.
    geom_tab: u8,
    /// Which eye the Image-tab sliders target: 0 = both, 1 = left, 2 = right. The
    /// sliders re-read the selected eye's live values every frame, so switching
    /// snaps them to that eye's numbers.
    geom_eye: u8,
    /// Dashboard eye-cameras source: false = raw cameras, true = the exact image
    /// sent to the eye net (un-mirrored).
    net_view: bool,
    /// BrokenEye (VRCFT) server status, shown in the OUTPUT node detail.
    be: Option<Arc<BrokenEyeStatus>>,
    /// One-shot guard for the fit-to-monitor scale (applied once when monitor size is
    /// known; set every frame would oscillate ppp — the documented flicker bug).
    fit_done: bool,
    /// Pointer position inside the custom title bar at drag start. Moving the
    /// undecorated window ourselves avoids Windows' modal WM_ENTERSIZEMOVE loop,
    /// which can suspend eframe redraws until another native input event arrives.
    /// Eyebrow-calibration data-collection controller (B-1 capture tab). Writes RAW eye
    /// frames + labels.csv under base_dir()/brow_data for offline training (B-2).
    brow: BrowCalib,
    /// Last c_frame_l/r counters seen by the brow-capture tab, so it saves exactly one frame
    /// per NEW device frame during a capture phase (not per repaint).
    brow_last_frames: [u64; 2],
    /// B-2: the offline train->bake subprocess runner (drives the user's PyTorch venv).
    trainer: BrowTrainer,
    /// True once we've consumed a `Done` from `trainer` and hot-loaded the model, so we
    /// don't re-load it every frame while the status stays `Done`.
    train_applied: bool,
    /// In-app pure-Rust head-fit runner: re-fits only the output head onto the captured
    /// brow_data, reusing an existing brow.bin as a frozen backbone (no Python). The lighter,
    /// recommended alternative to the full external `trainer`.
    fitter: BrowFitter,
    /// One-shot guard mirroring `train_applied` for the in-app fit's hot-load.
    fit_applied: bool,
    /// Dream Air/XR5 EyeWide capture controller. Each run creates an independent
    /// session so fitting can hold out a whole reseat/session for validation.
    wide: WideCalib,
    wide_last_frames: [u64; 2],
    wide_fitter: WideFitter,
    wide_fit_applied: bool,
    confirm_delete_wide: bool,
    /// The process's captured stdout/stderr ring buffer (see `logcap`), rendered by the
    /// Console tab so runtime logs are visible without launching from a terminal.
    log: Arc<Mutex<VecDeque<String>>>,
}

impl App {
    fn new(
        pipeline: Pipeline,
        be: Option<Arc<BrokenEyeStatus>>,
        startup_notice: Option<String>,
        #[cfg(windows)] gpu_context: Option<crate::ml::eyelid_model::EyelidGpuContext>,
    ) -> Self {
        let tele = pipeline.tele.clone();
        // Anchored path (not CWD), and SURFACE malformed or automatically repaired
        // configuration instead of silently resetting it.
        let config_path = crate::config::config_path();
        let (mut config, mut cfg_warn) = Config::load(&config_path);
        if crate::config::canonical_device_key(&pipeline.device_key) == "pimax_xr5" {
            let live_source = pipeline.selected_wide_source();
            if config.hmd.wide_source != live_source {
                let requested = config.hmd.wide_source;
                config.hmd.wide_source = live_source;
                let repair = format!(
                    "XR5 EyeWide source '{}' was unavailable and was reset to '{}'; fit or install a Wide model before selecting Auto/Custom",
                    requested.as_str(),
                    live_source.as_str()
                );
                if let Err(error) = config.save(&config_path) {
                    cfg_warn = Some(format!("{repair}; could not save repair: {error}"));
                } else {
                    cfg_warn = Some(repair);
                }
            }
        }
        let edit = SettingsEdit::from_cfg(&config);
        let reseat_context = reseat_reference_context(&config, &pipeline);
        let reseat_assist = WearingPositionAssist::load(reseat_context);
        let wear_memory = WearMemory::load(wear_memory_context(&config, &pipeline));
        let recovery_mode = startup_notice
            .as_deref()
            .is_some_and(|message| message.starts_with("Eye tracker connection failed"));
        let page = if recovery_mode {
            Page::Settings
        } else {
            match std::env::args().nth(2).as_deref() {
                Some("calibration") => Page::Calibration,
                // Keep the old command-line routes working after eyebrow calibration
                // moved into the shared XR5 calibration page.
                Some("brow") | Some("browcalib") => Page::Calibration,
                Some("console") => Page::Console,
                Some("settings") => Page::Settings,
                _ => Page::Dashboard,
            }
        };
        let reload_msg = match (startup_notice, cfg_warn) {
            (Some(notice), Some(warning)) => Some((
                format!("{notice}\nConfiguration note: {warning}"),
                if recovery_mode { ERR } else { WARN },
            )),
            (Some(notice), None) => Some((notice, if recovery_mode { ERR } else { OK })),
            (None, Some(warning)) => Some((warning, WARN)),
            (None, None) => None,
        };
        let last_ui_frame_generations = tele.frame_generations();
        Self {
            pipeline,
            #[cfg(windows)]
            gpu_context,
            tele,
            config,
            edit,
            reload_msg,
            reload_job: None,
            reload_backdrop: Vec::new(),
            sranipal_discovery_job: None,
            page,
            last: [0; 5],
            rates: [0.0; 5],
            last_t: Instant::now(),
            tex_l: None,
            tex_r: None,
            last_eye_texture_upload: Instant::now() - Duration::from_secs(1),
            last_eye_texture_generation: [0; 2],
            tex_ml_l: None,
            tex_ml_r: None,
            tex_heat_l: None,
            tex_heat_r: None,
            heat_vmax: 0.20,
            heat_auto_scale: true,
            show_geom_modal: false,
            show_gaze_modal: false,
            show_wear_memory_modal: false,
            eyelid_response_preview: None,
            gaze_center_capture: None,
            gaze_center_msg: None,
            gaze_source_modal_edit: None,
            dream_air_msg: None,
            recording_audio: RecordingAudio::new(),
            recording_cues: RecordingCueState::default(),
            geometry_capture: GeometryCapture::new(),
            geometry_fitter: GeometryFitter::new(),
            photometric_fitter: PhotometricFitter::new(),
            endpoint_fitter: EndpointFitter::new(),
            gaze_eyelid_fitter: GazeEyelidFitter::new(),
            wink_fitter: WinkFitter::new(),
            blink_timing_fitter: BlinkTimingFitter::new(),
            reseat_assist,
            wear_memory,
            wear_memory_matching_suspended: false,
            wear_response_before_edit: None,
            wear_baseline_before_edit: None,
            wear_thumbnails: Vec::new(),
            wear_memory_draft: WearingMemoryDraft::default(),
            wear_closed_capture: None,
            confirm_remove_reseat_reference: false,
            calibration_capture_purpose: None,
            calibration_view: CalibrationView::Problems,
            calibration_detail_window: None,
            pending_calibration_removal: None,
            unified_calibration: None,
            geometry_capture_baseline: None,
            geometry_capture_filters: None,
            photometric_capture_baseline: None,
            geometry_recording_export_attempted: false,
            geometry_recording_path: None,
            geometry_recording_export_job: None,
            geometry_recording_export_thread: None,
            gaze_residual_capture: GazeResidualCapture::new(),
            gaze_residual_export_attempted: false,
            gaze_residual_recording_path: None,
            gaze_residual_snapshot: None,
            gaze_residual_export_job: None,
            gaze_residual_export_thread: None,
            gaze_residual_window_elevated: false,
            vr_research_overlay: VrResearchOverlay::new(),
            geometry_preview_restore: None,
            geometry_rollback: None,
            photometric_rollback: None,
            geometry_unvalidated_ack: false,
            geom_tab: 0,
            geom_eye: 0,
            net_view: false,
            start: Instant::now(),
            last_ui_update: Instant::now(),
            last_ui_frame_generations,
            ui_was_minimized: false,
            window_drag_started_at: None,
            window_drag_start_generations: [0; 2],
            window_drag_update_count: 0,
            window_drag_max_ui_gap: Duration::ZERO,
            prev_ok: None,
            prev_paused: false,
            events: Vec::new(),
            sel_node: None,
            pipeline_open: false,
            be,
            fit_done: false,
            brow: BrowCalib::new(),
            brow_last_frames: [0; 2],
            trainer: BrowTrainer::new(),
            train_applied: false,
            fitter: BrowFitter::new(),
            fit_applied: false,
            wide: WideCalib::new(),
            wide_last_frames: [0; 2],
            wide_fitter: WideFitter::new(),
            wide_fit_applied: false,
            confirm_delete_wide: false,
            log: crate::logcap::log_buffer(),
        }
    }

    /// One-shot fit-to-monitor: if the fixed design window (WIN_W x WIN_H points) is
    /// bigger than the monitor (small/high-DPI laptops), shrink BOTH the window and the
    /// UI uniformly via a single zoom_factor so nothing clips off-screen. The layout
    /// stays in its WIN_W x WIN_H POINT space (zoom only changes points->pixels), so
    /// fixed-position elements like the nav LED at content_h() stay correct. Applied
    /// once (the guard) — setting zoom every frame oscillates ppp (the flicker bug).
    fn fit_to_monitor(&mut self, ctx: &egui::Context) {
        if self.fit_done {
            return;
        }
        let mon = ctx.input(|i| i.viewport().monitor_size);
        let Some(mon) = mon else { return }; // not known yet — retry next frame
        if mon.x < 1.0 || mon.y < 1.0 {
            return;
        }
        self.fit_done = true;
        // Percentage margins (not a hardcoded 64) leave room for window decorations +
        // the taskbar; shrink only (never enlarge past design). A low floor is allowed
        // so a very small screen still fits (readability suffers but nothing clips).
        // 0.85 vertical clears the taskbar (~40-48px) + title bar even on a 720-logical
        // -tall screen; 0.96 horizontal covers side decorations. monitor_size is full
        // (egui exposes no work area), so be conservative. Shrink-only via the 1.0 cap.
        let fit = (mon.x * 0.96 / WIN_W)
            .min(mon.y * 0.85 / WIN_H)
            .clamp(0.4, 1.0);
        if fit < 0.995 {
            ctx.set_zoom_factor(fit);
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(vec2(
                WIN_W * fit,
                WIN_H * fit,
            )));
        }
    }

    /// Current per-stage health (same logic as the signal rail).
    fn stage_oks(&self) -> [bool; 6] {
        let cam = self.rates[0] > 1.0 && self.rates[1] > 1.0;
        let gaze = self.rates[2] > 1.0;
        let ml = self.tele.ml_loaded && self.rates[3] > 1.0;
        let core = self.rates[4] > 1.0;
        let tracking = !self.pipeline.paused.load(Ordering::Relaxed);
        [cam || gaze, cam, gaze, ml, core, core && tracking]
    }

    /// Log stage transitions + tracking toggles to the event strip.
    fn detect_events(&mut self) {
        // Let rates settle before logging, so startup isn't noisy.
        if self.start.elapsed().as_secs_f32() < 1.5 {
            return;
        }
        let names = ["Device", "Camera", "Gaze", "ML", "Core", "Output"];
        let oks = self.stage_oks();
        if let Some(prev) = self.prev_ok {
            for i in 0..6 {
                if prev[i] != oks[i] {
                    let (msg, col) = if oks[i] {
                        (format!("{} restored", names[i]), OK)
                    } else {
                        (format!("{} stalled", names[i]), ERR)
                    };
                    self.events.push((now_hms(), msg, col));
                }
            }
        }
        self.prev_ok = Some(oks);
        let paused = self.pipeline.paused.load(Ordering::Relaxed);
        if paused != self.prev_paused {
            let (msg, col) = if paused {
                ("Tracking turned off".to_string(), WARN)
            } else {
                ("Tracking turned on".to_string(), OK)
            };
            self.events.push((now_hms(), msg, col));
            self.prev_paused = paused;
        }
        if self.events.len() > 60 {
            let drop = self.events.len() - 60;
            self.events.drain(0..drop);
        }
    }

    fn update_rates(&mut self) {
        let dt = self.last_t.elapsed().as_secs_f32();
        if dt < 0.5 {
            return;
        }
        let cur = [
            self.tele.c_frame_l.load(Ordering::Relaxed),
            self.tele.c_frame_r.load(Ordering::Relaxed),
            self.tele.c_gaze.load(Ordering::Relaxed),
            self.tele.c_ml.load(Ordering::Relaxed),
            self.tele.c_emit.load(Ordering::Relaxed),
        ];
        for i in 0..5 {
            self.rates[i] = cur[i].saturating_sub(self.last[i]) as f32 / dt;
        }
        self.last = cur;
        self.last_t = Instant::now();
    }

    fn sync_recording_audio_cues(&mut self) {
        let enabled = self.config.ui.recording_audio_cues;
        let initialized = self.recording_cues.initialized;
        let mut cues = Vec::new();

        let (geometry_key, geometry_cue, geometry_stalled) = match self.geometry_capture.status() {
            GeometryCaptureStatus::Idle => ("idle".to_owned(), None, false),
            GeometryCaptureStatus::Rest { instruction, .. } => (
                format!("rest:{instruction}"),
                Some(if instruction.to_ascii_lowercase().contains("holdout") {
                    RecordingCue::Holdout
                } else {
                    RecordingCue::Prepare
                }),
                false,
            ),
            GeometryCaptureStatus::Capture {
                instruction,
                stereo_stalled,
                ..
            } => (
                format!("capture:{instruction}"),
                Some(RecordingCue::Sampling),
                stereo_stalled,
            ),
            GeometryCaptureStatus::Done { .. } => {
                ("done".to_owned(), Some(RecordingCue::Complete), false)
            }
        };
        if initialized {
            if geometry_stalled && !self.recording_cues.geometry_stalled {
                cues.push(RecordingCue::Warning);
            } else if self.recording_cues.geometry.as_deref() != Some(&geometry_key) {
                let aborted =
                    geometry_key == "idle"
                        && self.recording_cues.geometry.as_deref().is_some_and(|key| {
                            key.starts_with("rest:") || key.starts_with("capture:")
                        });
                if aborted {
                    cues.push(RecordingCue::Cancelled);
                } else if let Some(cue) = geometry_cue {
                    cues.push(cue);
                }
            }
        }
        self.recording_cues.geometry = Some(geometry_key);
        self.recording_cues.geometry_stalled = geometry_stalled;
        let geometry_error = self.geometry_capture.last_error.clone();
        if initialized
            && geometry_error.is_some()
            && geometry_error != self.recording_cues.geometry_error
        {
            cues.push(RecordingCue::Warning);
        }
        self.recording_cues.geometry_error = geometry_error;

        let (residual_key, residual_cue, residual_paused, residual_stalled) =
            match self.gaze_residual_capture.status() {
                GazeResidualStatus::Idle => ("idle".to_owned(), None, false, false),
                GazeResidualStatus::Ready { .. } => {
                    ("ready".to_owned(), Some(RecordingCue::Ready), false, false)
                }
                GazeResidualStatus::Running {
                    phase_index,
                    holdout,
                    recording,
                    settling,
                    awaiting_confirmation,
                    paused,
                    stereo_stalled,
                    ..
                } => (
                    format!("running:{phase_index}"),
                    Some(if awaiting_confirmation {
                        RecordingCue::Ready
                    } else if recording && !settling {
                        RecordingCue::Sampling
                    } else if holdout {
                        RecordingCue::Holdout
                    } else {
                        RecordingCue::Prepare
                    }),
                    paused,
                    stereo_stalled,
                ),
                GazeResidualStatus::Done { .. } => (
                    "done".to_owned(),
                    Some(RecordingCue::Complete),
                    false,
                    false,
                ),
            };
        if initialized {
            if residual_paused && !self.recording_cues.residual_paused {
                cues.push(RecordingCue::Paused);
            } else if !residual_paused && self.recording_cues.residual_paused {
                cues.push(RecordingCue::Resumed);
            } else if residual_stalled && !self.recording_cues.residual_stalled {
                cues.push(RecordingCue::Warning);
            } else if self.recording_cues.residual.as_deref() != Some(&residual_key) {
                let aborted = residual_key == "idle"
                    && self
                        .recording_cues
                        .residual
                        .as_deref()
                        .is_some_and(|key| key == "ready" || key.starts_with("running:"));
                if aborted {
                    cues.push(RecordingCue::Cancelled);
                } else if let Some(cue) = residual_cue {
                    cues.push(cue);
                }
            }
        }
        self.recording_cues.residual = Some(residual_key);
        self.recording_cues.residual_paused = residual_paused;
        self.recording_cues.residual_stalled = residual_stalled;
        let residual_error = self.gaze_residual_capture.last_error.clone();
        if initialized
            && residual_error.is_some()
            && residual_error != self.recording_cues.residual_error
        {
            cues.push(RecordingCue::Warning);
        }
        self.recording_cues.residual_error = residual_error;

        let (wide_key, wide_cue) = match self.wide.status() {
            WideCalibStatus::Idle => ("idle".to_owned(), None),
            WideCalibStatus::Rest { instruction, .. } => {
                (format!("rest:{instruction}"), Some(RecordingCue::Prepare))
            }
            WideCalibStatus::Capture {
                folder,
                instruction,
                ..
            } => (
                format!("capture:{folder}:{instruction}"),
                Some(RecordingCue::Sampling),
            ),
            WideCalibStatus::Done { session } => (
                format!("done:{}", session.display()),
                Some(RecordingCue::Complete),
            ),
        };
        if initialized && self.recording_cues.wide.as_deref() != Some(&wide_key) {
            let aborted = wide_key == "idle"
                && self
                    .recording_cues
                    .wide
                    .as_deref()
                    .is_some_and(|key| key.starts_with("rest:") || key.starts_with("capture:"));
            if aborted {
                cues.push(RecordingCue::Cancelled);
            } else if let Some(cue) = wide_cue {
                cues.push(cue);
            }
        }
        self.recording_cues.wide = Some(wide_key);
        let wide_error = self.wide.last_error.clone();
        if initialized && wide_error.is_some() && wide_error != self.recording_cues.wide_error {
            cues.push(RecordingCue::Warning);
        }
        self.recording_cues.wide_error = wide_error;

        let (brow_key, brow_cue) = match self.brow.status() {
            BrowStatus::Idle => ("idle".to_owned(), None),
            BrowStatus::Rest { instruction, .. } => {
                (format!("rest:{instruction}"), Some(RecordingCue::Prepare))
            }
            BrowStatus::Capture {
                folder,
                instruction,
                ..
            } => (
                format!("capture:{folder}:{instruction}"),
                Some(RecordingCue::Sampling),
            ),
            BrowStatus::Done => ("done".to_owned(), Some(RecordingCue::Complete)),
        };
        if initialized && self.recording_cues.brow.as_deref() != Some(&brow_key) {
            let aborted = brow_key == "idle"
                && self
                    .recording_cues
                    .brow
                    .as_deref()
                    .is_some_and(|key| key.starts_with("rest:") || key.starts_with("capture:"));
            if aborted {
                cues.push(RecordingCue::Cancelled);
            } else if let Some(cue) = brow_cue {
                cues.push(cue);
            }
        }
        self.recording_cues.brow = Some(brow_key);
        let brow_error = self.brow.last_error.clone();
        if initialized && brow_error.is_some() && brow_error != self.recording_cues.brow_error {
            cues.push(RecordingCue::Warning);
        }
        self.recording_cues.brow_error = brow_error;

        let diagnostic = self.pipeline.diag_rec.load(Ordering::Relaxed);
        if initialized && diagnostic != self.recording_cues.diagnostic {
            cues.push(if diagnostic {
                RecordingCue::DiagnosticStarted
            } else {
                RecordingCue::DiagnosticStopped
            });
        }
        self.recording_cues.diagnostic = diagnostic;
        self.recording_cues.initialized = true;

        for cue in cues {
            self.recording_audio.cue(cue, enabled);
        }
    }

    fn compensate_calibration_ui_gap(&mut self, now: Instant) {
        const UI_HICCUP: Duration = Duration::from_millis(250);
        let gap = now.saturating_duration_since(self.last_ui_update);
        self.last_ui_update = now;
        if self.window_drag_started_at.is_some() {
            self.window_drag_update_count = self.window_drag_update_count.saturating_add(1);
            self.window_drag_max_ui_gap = self.window_drag_max_ui_gap.max(gap);
        }
        let generation = self.tele.frame_generations();
        let generation_delta = [
            generation[0].saturating_sub(self.last_ui_frame_generations[0]),
            generation[1].saturating_sub(self.last_ui_frame_generations[1]),
        ];
        self.last_ui_frame_generations = generation;
        if gap < UI_HICCUP {
            return;
        }
        let page = match self.page {
            Page::Dashboard => "dashboard",
            Page::Calibration => "calibration",
            Page::Console => "console",
            Page::Settings => "settings",
        };
        eprintln!(
            "[ui:stall] update gap {:.0}ms page={page} drag={} preview={} camera_delta={}/{}",
            gap.as_secs_f64() * 1000.0,
            self.window_drag_started_at.is_some(),
            self.config.ui.eye_camera_preview,
            generation_delta[0],
            generation_delta[1],
        );
        self.geometry_capture.suspend_for(gap, generation);
        self.gaze_residual_capture.suspend_for(gap, generation);
        self.reseat_assist.suspend_for(gap, generation);
        self.wide.suspend_for(gap);
        self.brow.suspend_for(gap);
        // Wide/Brow own only phase state; their generation cursors live in App.
        // Drop every frame produced while the prompt could not be presented.
        self.wide_last_frames = generation;
        self.brow_last_frames = generation;
        if let Some(capture) = self.gaze_center_capture.as_mut() {
            capture.suspend_for(gap, now, self.tele.fresh_gaze().timestamp_us);
        }
    }

    fn mark_window_drag_started(&mut self, now: Instant) {
        if self.window_drag_started_at.is_some() {
            return;
        }
        self.window_drag_started_at = Some(now);
        self.window_drag_start_generations = self.tele.frame_generations();
        self.window_drag_update_count = 0;
        self.window_drag_max_ui_gap = Duration::ZERO;
        self.geometry_capture.pause();
        self.gaze_residual_capture.pause();
        self.wide.pause();
        self.brow.pause();
        if let Some(capture) = self.gaze_center_capture.as_mut() {
            capture.pause(now);
        }
    }

    #[cfg(not(windows))]
    fn begin_fallback_window_drag(&mut self, ctx: &egui::Context) {
        self.mark_window_drag_started(Instant::now());
        ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
    }

    fn finish_window_drag(&mut self, ctx: &egui::Context, now: Instant) {
        let Some(started_at) = self.window_drag_started_at else {
            return;
        };

        self.window_drag_started_at = None;
        let generation = self.tele.frame_generations();
        let generation_delta = [
            generation[0].saturating_sub(self.window_drag_start_generations[0]),
            generation[1].saturating_sub(self.window_drag_start_generations[1]),
        ];
        eprintln!(
            "[ui:drag] native move ended after {:.0}ms; ui_updates={} max_gap={:.1}ms camera_delta={}/{}",
            now.saturating_duration_since(started_at).as_secs_f64() * 1000.0,
            self.window_drag_update_count,
            self.window_drag_max_ui_gap.as_secs_f64() * 1000.0,
            generation_delta[0],
            generation_delta[1],
        );
        self.last_ui_frame_generations = generation;
        self.geometry_capture.discard_through(generation);
        self.gaze_residual_capture.discard_through(generation);
        self.reseat_assist
            .suspend_for(now.saturating_duration_since(started_at), generation);
        self.wide_last_frames = generation;
        self.brow_last_frames = generation;
        let gaze_timestamp = self.tele.fresh_gaze().timestamp_us;
        if let Some(capture) = self.gaze_center_capture.as_mut() {
            capture.resume(now, gaze_timestamp);
        }

        // Pause/resume owns the capture clocks, so do not let the generic UI-gap
        // compensator charge the same native move interval a second time.
        self.last_ui_update = now;
        ctx.request_repaint();
    }
}

impl Drop for App {
    fn drop(&mut self) {
        if let Some(worker) = self.reload_job.take() {
            if let Ok(Ok(mut engine)) = worker.join() {
                engine.pipeline.stop();
            }
        }
        self.pipeline.stop();
        // A completed biometric capture is more valuable than a fast shutdown. The
        // worker owns the only frame buffers while saving, so wait for its atomic
        // publish/cleanup instead of detaching it and leaving a partial archive.
        if let Some(worker) = self.gaze_residual_export_thread.take() {
            let _ = worker.join();
        }
        if let Some(worker) = self.geometry_recording_export_thread.take() {
            let _ = worker.join();
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_reload();
        if self.reload_job.is_some() {
            if ctx.input(|input| input.viewport().minimized.unwrap_or(false)) {
                ctx.request_repaint_after(Duration::from_millis(250));
                return;
            }
            self.title_bar(ctx);
            self.reload_overlay(ctx);
            ctx.request_repaint_after(Duration::from_millis(if native_window_drag_active() {
                250
            } else {
                33
            }));
            return;
        }
        let now = Instant::now();
        let native_drag = native_window_drag_active();
        if native_drag {
            self.mark_window_drag_started(now);
        } else if self.window_drag_started_at.is_some()
            && !ctx.input(|input| input.pointer.primary_down())
        {
            self.finish_window_drag(ctx, now);
        }
        self.compensate_calibration_ui_gap(now);
        self.update_closed_point_capture(now);
        self.update_wear_memory(now);
        if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
            if self.gaze_residual_capture.is_ready() {
                self.begin_gaze_residual_capture();
            } else if self.gaze_residual_capture.continue_step() {
                self.dream_air_msg = Some(("Three-second preparation started.".into(), ACCENT));
            } else if self.geometry_capture.continue_step() {
                self.dream_air_msg = Some(("Three-second preparation started.".into(), ACCENT));
            }
        }
        self.sync_gaze_residual_vr_overlay();
        let residual_running = self.gaze_residual_capture.is_running();
        if residual_running != self.gaze_residual_window_elevated {
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(if residual_running {
                egui::viewport::WindowLevel::AlwaysOnTop
            } else {
                egui::viewport::WindowLevel::Normal
            }));
            self.gaze_residual_window_elevated = residual_running;
        }
        // eframe/wgpu skips presenting while a Windows viewport is minimized. Building the
        // complete dashboard anyway (especially `TextureHandle::set` for the live eye
        // cameras) leaves texture/paint deltas queued behind the unavailable surface. They
        // are not reclaimed on restore and Private Bytes can grow by hundreds of MiB per
        // minute. Keep the non-render tracking/capture controllers alive, but emit an empty
        // UI frame until Windows restores the surface.
        let minimized = ctx.input(|i| i.viewport().minimized.unwrap_or(false));
        if minimized {
            self.ui_was_minimized = true;
            // A visible head-locked SteamVR target remains valid while the desktop is
            // minimized. Without it, freeze the protocol: a hidden desktop point cannot
            // carry a valid commanded gaze label. The early return still avoids all egui
            // texture uploads, preserving the minimized-memory fix.
            if self.config.ui.steamvr_overlay && self.vr_research_overlay.is_visible() {
                self.gaze_residual_capture.resume();
            } else {
                self.gaze_residual_capture.pause();
            }
            self.geometry_capture.pause();
            self.wide.pause();
            self.brow.pause();
            if let Some(capture) = self.gaze_center_capture.as_mut() {
                capture.pause(Instant::now());
            }
            ctx.request_repaint_after(Duration::from_millis(
                if (self.gaze_residual_capture.is_running() || self.reseat_assist.is_active())
                    && self.config.ui.steamvr_overlay
                {
                    16
                } else {
                    250
                },
            ));
            self.update_rates();
            self.detect_events();
            self.update_geometry_capture();
            self.update_gaze_residual_capture();
            self.update_unified_calibration();
            self.update_reseat_assist();
            self.sync_gaze_residual_vr_overlay();
            self.update_wide_capture();
            self.update_brow_capture();
            self.apply_wide_fit_result_if_ready();
            self.sync_recording_audio_cues();
            return;
        }

        if std::mem::take(&mut self.ui_was_minimized) {
            let generation = self.tele.frame_generations();
            self.geometry_capture.discard_through(generation);
            self.gaze_residual_capture.discard_through(generation);
            self.reseat_assist.discard_through(generation);
            self.wide_last_frames = generation;
            self.brow_last_frames = generation;
            let gaze_timestamp = self.tele.fresh_gaze().timestamp_us;
            if let Some(capture) = self.gaze_center_capture.as_mut() {
                capture.resume(Instant::now(), gaze_timestamp);
            }
        }

        if self.window_drag_started_at.is_some() {
            self.gaze_residual_capture.pause();
            self.geometry_capture.pause();
            self.wide.pause();
            self.brow.pause();
        } else {
            self.gaze_residual_capture.resume();
            self.geometry_capture.resume();
            self.wide.resume();
            self.brow.resume();
        }

        // NOTE: we deliberately do NOT touch zoom_factor / pixels_per_point here.
        // Deriving zoom from native_pixels_per_point() each frame oscillates,
        // because egui feeds the *effective* ppp back through that call — the
        // result flickers between scales. We let egui keep its stable native ppp
        // and size everything from fixed point dimensions in `theme`.
        // ML/eye/brow inference run on their own threads. Windows owns title-bar
        // movement, so the compositor can present it at the monitor's native cadence.
        // A visible live-eye preview opts the dashboard into the cameras' 120 Hz
        // cadence; hidden preview keeps this presentation-light tracking cadence.
        // Calibration consumes bounded source-frame history, so this repaint cadence
        // never defines its sample rate.
        let redraw_interval = if self.window_drag_started_at.is_some() {
            // DWM moves the last presented surface while its native caption loop owns
            // the mouse. Do not submit another FIFO frame every 8 ms: those presents
            // blocked winit's move messages behind VSync and made the window trail the
            // cursor. A release/move event requests the one repaint needed afterwards.
            Duration::from_millis(250)
        } else if self.wide.is_running()
            || self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.reseat_assist.is_active()
            || self.brow.is_running()
        {
            Duration::from_millis(16)
        } else if self.config.ui.eye_camera_preview {
            LIVE_EYE_TEXTURE_INTERVAL
        } else {
            TRACKING_UI_REPAINT_INTERVAL
        };
        ctx.request_repaint_after(redraw_interval);
        self.fit_to_monitor(ctx);
        self.update_rates();
        self.detect_events();
        self.update_gaze_center_capture();
        self.update_geometry_capture();
        self.update_gaze_residual_capture();
        self.update_unified_calibration();
        self.update_reseat_assist();
        self.sync_gaze_residual_vr_overlay();
        self.update_wide_capture();
        self.update_brow_capture();
        self.apply_wide_fit_result_if_ready();
        self.sync_recording_audio_cues();
        self.title_bar(ctx);
        self.nav(ctx);
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(egui::Margin::same(MAIN_PAD)),
            )
            .show(ctx, |ui| match self.page {
                Page::Dashboard => self.dashboard(ui),
                Page::Calibration => self.calibration(ui),
                Page::Console => self.console(ui),
                Page::Settings => self.settings(ui),
            });
        // Guided calibration details are foreground navigation; nested image/gaze
        // editors are rendered after it so they remain the topmost modal.
        self.calibration_detail_modal(ctx);
        if self.show_wear_memory_modal {
            self.wearing_memory_modal(ctx);
        }
        // This editor is reachable from Dashboard's camera gear and from the
        // Calibration page's VR4/Varjo photometric card.
        if self.show_geom_modal {
            let frames = self.tele.stereo_frames();
            self.geom_modal(ctx, &frames);
        }
        if self.show_gaze_modal {
            self.gaze_correction_modal(ctx);
        }
        self.gaze_residual_capture_overlay(ctx);
        if self.reload_job.is_some() {
            // Keep a single snapshot of the just-rendered page. Draining a clone
            // leaves this frame untouched and preserves all modal/layer ordering.
            let (layers, transforms) = ctx.memory(|memory| {
                (
                    memory.layer_ids().collect::<Vec<_>>(),
                    memory.layer_transforms.clone(),
                )
            });
            self.reload_backdrop =
                ctx.graphics(|graphics| graphics.clone().drain(&layers, &transforms));
        }
        let frame_time = now.elapsed();
        if frame_time >= Duration::from_millis(50) {
            let page = match self.page {
                Page::Dashboard => "dashboard",
                Page::Calibration => "calibration",
                Page::Console => "console",
                Page::Settings => "settings",
            };
            eprintln!(
                "[ui:slow-frame] {:.1}ms page={page} preview={} drag={}",
                frame_time.as_secs_f64() * 1000.0,
                self.config.ui.eye_camera_preview,
                self.window_drag_started_at.is_some(),
            );
        }
    }
}

impl App {
    fn reload_overlay(&self, ctx: &egui::Context) {
        let body = ctx.available_rect();
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Middle,
            egui::Id::new("reload-backdrop"),
        ));
        painter.rect_filled(body, 0.0, BG);
        for clipped in &self.reload_backdrop {
            let clip = clipped.clip_rect.intersect(body);
            if clip.is_positive() {
                painter.with_clip_rect(clip).add(clipped.shape.clone());
            }
        }
        painter.rect_filled(body, 0.0, Color32::from_black_alpha(115));
        egui::Area::new(egui::Id::new("reload-progress"))
            .order(egui::Order::Foreground)
            .anchor(Align2::CENTER_CENTER, vec2(0.0, 0.0))
            .show(ctx, |ui| {
                card().show(ui, |ui| {
                    ui.set_width((300.0 * S).min((body.width() - 48.0 * S).max(140.0)));
                    ui.horizontal(|ui| {
                        ui.add(egui::Spinner::new().color(ACCENT));
                        ui.label(
                            egui::RichText::new("Reloading tracking…")
                                .color(TEXT1)
                                .strong(),
                        );
                    });
                    ui.add_space(SP2);
                    ui.label(prose("Please wait while tracking reconnects."));
                });
            });
    }

    #[cfg(any())]
    fn current_preflight(&self) -> PreflightReport {
        let frame_dims = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|frame| (frame.width, frame.height)),
                frames[1].as_ref().map(|frame| (frame.width, frame.height)),
            ]
        };
        evaluate_preflight(&PreflightInput {
            rates: self.rates,
            frame_dims,
            ml_loaded: self.tele.ml_loaded,
            ml: *self.tele.ml5.lock().unwrap(),
            gaze: self.tele.fresh_gaze(),
        })
    }

    #[cfg(any())]
    fn current_quality(&self) -> QualityReport {
        evaluate_quality(&QualityInput {
            rates: self.rates,
            ml: *self.tele.ml5.lock().unwrap(),
            gaze: self.tele.fresh_gaze(),
            baselines: *self.tele.baselines.lock().unwrap(),
            results: *self.tele.results.lock().unwrap(),
        })
    }

    /// The UI repaint clock is not a measurement clock. Sample the guided flow
    /// only when `c_ml` changes, then hand its completed report to the review UI.
    #[cfg(any())]
    fn update_dream_air_state(&mut self) {
        if self.pipeline.device_key != "pimax_xr5" {
            return;
        }
        if self.quality_last.elapsed() >= Duration::from_millis(250) {
            self.quality_report = Some(self.current_quality());
            self.quality_last = Instant::now();
        }

        let generation = self.tele.c_ml.load(Ordering::Relaxed);
        if generation == 0 || generation == self.guided_last_ml {
            return;
        }
        self.guided_last_ml = generation;
        let ml = *self.tele.ml5.lock().unwrap();
        let gaze = self.tele.fresh_gaze();
        let report = if let Some(session) = self.guided_calibration.as_mut() {
            session.push(ml, &gaze);
            session.report()
        } else {
            None
        };
        if let Some(report) = report {
            self.guided_report = Some(report);
            self.guided_calibration = None;
            self.dream_air_msg = Some((
                "Measurement complete - review before applying".into(),
                ACCENT,
            ));
        }
    }

    /// Drive the shared geometry/photometric evidence capture from real camera
    /// generations. The capture clock is capped at 20 Hz, so UI repaint bursts cannot
    /// duplicate a frame and an unfocused UI still records enough blink timing detail.
    fn update_geometry_capture(&mut self) {
        self.poll_geometry_recording_export();
        if self.geometry_capture.is_running() && !self.geometry_capture_state_unchanged() {
            self.geometry_capture.abort();
            self.geometry_capture_baseline = None;
            self.geometry_capture_filters = None;
            self.photometric_capture_baseline = None;
            self.calibration_capture_purpose = None;
            self.reset_geometry_recording_export();
            self.dream_air_msg = Some((
                "Calibration recording stopped because its frozen image pipeline changed. Start a new recording with the new settings."
                    .into(),
                ERR,
            ));
            return;
        }
        let device_supported = match self.calibration_capture_purpose {
            Some(CalibrationCapturePurpose::Geometry) => self.pipeline.device_key == "pimax_xr5",
            Some(CalibrationCapturePurpose::Photometric) => {
                crate::config::supports_photometric_fit(&self.pipeline.device_key)
            }
            Some(CalibrationCapturePurpose::EyelidEndpoints) => true,
            Some(CalibrationCapturePurpose::Winks) => true,
            Some(CalibrationCapturePurpose::NaturalBlinks) => true,
            None => false,
        };
        if !self.geometry_capture.is_running() || !device_supported {
            if self.geometry_capture.is_done() && !self.geometry_recording_export_attempted {
                self.export_geometry_recording();
            }
            return;
        }
        let history = self
            .tele
            .calibration_frames_after(self.geometry_capture.last_generation());
        for sample in history {
            self.geometry_capture.tick_at(sample.captured_at);
            if !self.geometry_capture.is_running() {
                break;
            }
            let gaze = sample.gaze;
            let native_open = [gaze.left, gaze.right].map(|eye| {
                if !eye.openness_reported {
                    None
                } else if !eye.openness_valid {
                    Some(0.0)
                } else {
                    eye.openness
                        .is_finite()
                        .then_some(eye.openness.clamp(0.0, 1.0))
                }
            });
            let native_gaze = [gaze.left, gaze.right].map(|eye| {
                (eye.gaze_reported
                    && eye.gaze_valid
                    && eye.gaze.iter().all(|component| component.is_finite()))
                .then_some(eye.gaze)
            });
            let left = Some(sample.frames[0].view());
            let right = Some(sample.frames[1].view());
            self.geometry_capture.on_frame_at(
                sample.captured_at,
                sample.source_generation,
                left,
                right,
                sample.affine,
                native_open,
                native_gaze,
                (gaze.timestamp_us != 0).then_some(gaze.timestamp_us),
            );
            self.geometry_capture.tick_at(sample.captured_at);
            self.geometry_capture
                .discard_through(sample.source_generation);
        }
        self.geometry_capture.tick_at(Instant::now());
        if self.geometry_capture.is_done() && !self.geometry_recording_export_attempted {
            self.export_geometry_recording();
        }
    }

    /// Drive the independent landmark/residual capture.  It shares only the live raw
    /// camera source with Safe Geometry Fit; its protocol, dataset, and export lifecycle
    /// are separate so running it cannot change fit evidence or live geometry.
    fn update_gaze_residual_capture(&mut self) {
        self.poll_gaze_residual_export();
        let unified_inputs_unchanged = self.unified_calibration.as_ref().is_none_or(|run| {
            run.protocol != self.gaze_residual_capture.protocol()
                || self.unified_runtime_inputs_compatible()
        });
        if self.gaze_residual_capture.is_running()
            && (!self.gaze_residual_state_unchanged() || !unified_inputs_unchanged)
        {
            let protocol = self.gaze_residual_capture.protocol();
            self.gaze_residual_capture.abort();
            self.gaze_residual_snapshot = None;
            self.gaze_residual_export_attempted = false;
            self.gaze_residual_recording_path = None;
            if self
                .unified_calibration
                .as_ref()
                .is_some_and(|run| run.protocol == protocol)
            {
                self.unified_calibration = None;
            }
            self.dream_air_msg = Some((
                "Calibration recording stopped because image geometry or filters changed. Start a new recording with the new settings."
                    .into(),
                ERR,
            ));
            return;
        }
        if !self.gaze_residual_capture.is_running() {
            if self.gaze_residual_capture.is_done() && !self.gaze_residual_export_attempted {
                self.export_gaze_residual_recording();
            }
            return;
        }
        let history = self
            .tele
            .calibration_frames_after(self.gaze_residual_capture.last_generation());
        for sample in history {
            self.gaze_residual_capture.tick_at(sample.captured_at);
            if !self.gaze_residual_capture.is_running() {
                break;
            }
            let gaze = sample.gaze;
            let native_open = [gaze.left, gaze.right].map(|eye| {
                if !eye.openness_reported {
                    None
                } else if !eye.openness_valid {
                    Some(0.0)
                } else {
                    eye.openness
                        .is_finite()
                        .then_some(eye.openness.clamp(0.0, 1.0))
                }
            });
            let native_gaze = [gaze.left, gaze.right].map(|eye| {
                (eye.gaze_reported
                    && eye.gaze_valid
                    && eye.gaze.iter().all(|component| component.is_finite()))
                .then_some(eye.gaze)
            });
            let native_pupil_pos = [gaze.left, gaze.right].map(|eye| {
                (eye.pupil_pos_reported
                    && eye.pupil_pos_valid
                    && eye.pupil_pos.iter().all(|component| component.is_finite()))
                .then_some(eye.pupil_pos)
            });
            self.gaze_residual_capture.on_frame_at(
                sample.captured_at,
                sample.source_generation,
                (gaze.timestamp_us != 0).then_some(gaze.timestamp_us),
                Some(sample.frames[0].view()),
                Some(sample.frames[1].view()),
                sample.affine,
                native_open,
                native_gaze,
                native_pupil_pos,
            );
            self.gaze_residual_capture.tick_at(sample.captured_at);
            self.gaze_residual_capture
                .discard_through(sample.source_generation);
        }
        self.gaze_residual_capture.tick_at(Instant::now());
        if self.gaze_residual_capture.is_done() && !self.gaze_residual_export_attempted {
            self.export_gaze_residual_recording();
        }
    }

    /// Mirror the independent research protocol into a large, head-locked SteamVR
    /// canvas. `VrResearchOverlay` de-duplicates identical frames, so calling this
    /// from every UI update does not upload textures at the dashboard redraw rate.
    fn sync_gaze_residual_vr_overlay(&mut self) {
        let enabled = self.config.ui.steamvr_overlay;
        let reseat_frame = self.reseat_assist.is_active().then(|| {
            let guidance = self.reseat_assist.guidance();
            let target = match guidance {
                ReseatGuidance::CapturingReference { progress, step, .. } => step.target(*progress),
                ReseatGuidance::WaitingForFrames
                | ReseatGuidance::LowConfidence { .. }
                | ReseatGuidance::Adjust(_)
                | ReseatGuidance::Aligned(_) => Some(crate::geometry_calib::GazeTarget::Center),
                ReseatGuidance::EyesClosed
                | ReseatGuidance::ReferenceReady { .. }
                | ReseatGuidance::ReferenceSaved { .. }
                | ReseatGuidance::Error(_) => None,
            };
            let (headline, instruction, footer) = match guidance {
                ReseatGuidance::WaitingForFrames => (
                    "WAITING FOR EYE CAMERAS".to_owned(),
                    "Keep the HMD on and look at the centre target.".to_owned(),
                    "NO SETTINGS ARE BEING CHANGED".to_owned(),
                ),
                ReseatGuidance::CapturingReference { step, .. } => (
                    "SAVING YOUR BEST FIT".to_owned(),
                    step.instruction().to_owned(),
                    "THIS REFERENCE CHANGES ONLY WHEN YOU SAVE IT".to_owned(),
                ),
                ReseatGuidance::EyesClosed => (
                    "OPEN BOTH EYES COMFORTABLY".to_owned(),
                    "Guidance is paused during a blink.".to_owned(),
                    "NO SETTINGS ARE BEING CHANGED".to_owned(),
                ),
                ReseatGuidance::LowConfidence { detail } => (
                    "HOLD STILL - MATCH UNCERTAIN".to_owned(),
                    detail.clone(),
                    "KEEP BOTH EYES COMFORTABLY OPEN".to_owned(),
                ),
                ReseatGuidance::Adjust(estimate) => (
                    "ADJUST THE HMD".to_owned(),
                    estimate.instruction(),
                    format!("MATCH CONFIDENCE {:.0}%", estimate.confidence * 100.0),
                ),
                ReseatGuidance::Aligned(estimate) => (
                    "POSITION MATCHED".to_owned(),
                    "Hold this position and tighten the HMD evenly.".to_owned(),
                    format!("MATCH CONFIDENCE {:.0}%", estimate.confidence * 100.0),
                ),
                ReseatGuidance::ReferenceReady { .. } => (
                    "REFERENCE CAPTURED".to_owned(),
                    "Review and save it on the desktop.".to_owned(),
                    "THE PREVIOUS REFERENCE IS STILL SAFE".to_owned(),
                ),
                ReseatGuidance::ReferenceSaved { .. } => (
                    "REFERENCE SAVED".to_owned(),
                    "This wearing position is now the fixed reference.".to_owned(),
                    "NO TRACKING CALIBRATION WAS CHANGED".to_owned(),
                ),
                ReseatGuidance::Error(error) => (
                    "RESEAT ASSIST STOPPED".to_owned(),
                    error.clone(),
                    "CHECK THE DESKTOP WINDOW".to_owned(),
                ),
            };
            VrTargetFrame {
                target,
                state: match guidance {
                    ReseatGuidance::WaitingForFrames
                    | ReseatGuidance::EyesClosed
                    | ReseatGuidance::LowConfidence { .. }
                    | ReseatGuidance::Error(_) => VrGuideState::Waiting,
                    ReseatGuidance::CapturingReference { .. } => VrGuideState::Recording,
                    ReseatGuidance::Adjust(_) | ReseatGuidance::Aligned(_) => VrGuideState::Adjust,
                    ReseatGuidance::ReferenceReady { .. }
                    | ReseatGuidance::ReferenceSaved { .. } => VrGuideState::Ready,
                },
                eye_pose: if target.is_some() {
                    VrEyePose::Open
                } else {
                    VrEyePose::None
                },
                countdown: None,
                progress_percent: None,
                headline,
                instruction,
                footer,
            }
        });
        let frame = reseat_frame
            .or_else(|| match self.gaze_residual_capture.status() {
                GazeResidualStatus::Ready { .. } => Some(VrTargetFrame {
                    target: None,
                    state: VrGuideState::Ready,
                    eye_pose: VrEyePose::Open,
                    countdown: None,
                    progress_percent: Some(0),
                    headline: "READY".into(),
                    instruction: "PUT ON THE HMD AND GET COMFORTABLE".into(),
                    footer: "PRESS SPACE OR BEGIN ON THE DESKTOP".into(),
                }),
                GazeResidualStatus::Running {
                    target,
                    action,
                    holdout,
                    recording,
                    settling,
                    paused,
                    stereo_stalled,
                    awaiting_confirmation,
                    phase_remaining_s,
                    pose_progress,
                    progress,
                    ..
                } => {
                    let (state, headline) = if paused {
                        (VrGuideState::Paused, "PAUSED")
                    } else if stereo_stalled {
                        (VrGuideState::Waiting, "WAITING FOR EYE CAMERAS")
                    } else if awaiting_confirmation {
                        (
                            VrGuideState::Ready,
                            if holdout {
                                "SAME ACTION AGAIN"
                            } else {
                                "NEXT ACTION"
                            },
                        )
                    } else if !recording && target.is_none() {
                        (VrGuideState::Prepare, "GET READY")
                    } else if settling {
                        (VrGuideState::Prepare, "MOVE INTO POSITION")
                    } else if !recording {
                        (VrGuideState::Prepare, "TRY IT ONCE")
                    } else {
                        (VrGuideState::Recording, "RECORDING")
                    };
                    let countdown = (!paused
                        && !stereo_stalled
                        && !awaiting_confirmation
                        && !recording
                        && target.is_none()
                        && phase_remaining_s > 0.0)
                        .then(|| phase_remaining_s.ceil().clamp(1.0, 9.0) as u8);
                    let target_visible =
                        target.is_some() && matches!(action, Some(GazeResidualAction::RelaxedOpen));
                    Some(VrTargetFrame {
                        target,
                        state,
                        eye_pose: vr_eye_pose(action, pose_progress),
                        countdown,
                        progress_percent: Some((progress * 100.0).round() as u8),
                        headline: headline.into(),
                        instruction: if paused {
                            "RETURN TO SRANIBRO TO CONTINUE"
                        } else if stereo_stalled {
                            "KEEP THE HMD ON AND HOLD STILL"
                        } else {
                            vr_pose_instruction(action, target_visible)
                        }
                        .into(),
                        footer: if awaiting_confirmation {
                            "PRESS SPACE OR CONTINUE ON THE DESKTOP"
                        } else if holdout {
                            "SAME ACTION AGAIN"
                        } else {
                            "KEEP YOUR HEAD STILL"
                        }
                        .into(),
                    })
                }
                _ => None,
            })
            .or_else(|| match self.geometry_capture.status() {
                GeometryCaptureStatus::Rest {
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                    next_kind,
                } => Some(VrTargetFrame {
                    target: None,
                    state: if awaiting_confirmation {
                        VrGuideState::Ready
                    } else {
                        VrGuideState::Prepare
                    },
                    eye_pose: geometry_vr_eye_pose(next_kind, None),
                    countdown: (!awaiting_confirmation)
                        .then(|| remaining_s.ceil().clamp(1.0, 9.0) as u8),
                    progress_percent: Some((overall * 100.0).round() as u8),
                    headline: if awaiting_confirmation {
                        "READ THIS STEP"
                    } else {
                        "GET READY"
                    }
                    .into(),
                    instruction: instruction.to_ascii_uppercase(),
                    footer: if awaiting_confirmation {
                        "PRESS SPACE OR CONTINUE ON THE DESKTOP"
                    } else {
                        "RECORDING STARTS AFTER THE COUNTDOWN"
                    }
                    .into(),
                }),
                GeometryCaptureStatus::Capture {
                    instruction,
                    kind,
                    remaining_s,
                    overall,
                    target_open,
                    stereo_stalled,
                    ..
                } => Some(VrTargetFrame {
                    target: None,
                    state: if stereo_stalled {
                        VrGuideState::Waiting
                    } else {
                        VrGuideState::Recording
                    },
                    eye_pose: geometry_vr_eye_pose(Some(kind), target_open),
                    countdown: None,
                    progress_percent: Some((overall * 100.0).round() as u8),
                    headline: if stereo_stalled {
                        "WAITING FOR EYE CAMERAS"
                    } else {
                        "RECORDING"
                    }
                    .into(),
                    instruction: instruction.to_ascii_uppercase(),
                    footer: if stereo_stalled {
                        "KEEP THE HMD ON AND HOLD STILL".into()
                    } else {
                        format!("{remaining_s:.0} SECONDS LEFT")
                    },
                }),
                _ => None,
            });
        self.vr_research_overlay.present(enabled, frame);
    }

    fn start_gaze_residual_capture(&mut self) {
        let _ = self.start_evidence_capture(CaptureProtocol::GazeDirections);
    }

    fn start_initial_setup_capture(&mut self) {
        let Some(endpoints) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some((
                "Wait for the live eyelid baseline to become ready before Initial setup.".into(),
                WARN,
            ));
            return;
        };
        let open_deadzone = self.pipeline.tuning.lock().unwrap().open_deadzone;
        if self.start_evidence_capture(CaptureProtocol::InitialSetup) {
            self.unified_calibration = Some(UnifiedCalibrationRun::new(
                CaptureProtocol::InitialSetup,
                Some(endpoints),
                open_deadzone,
                self.config
                    .gaze_eyelid_profile_for(&self.pipeline.device_key),
                self.config.wink_profile_for(&self.pipeline.device_key),
                self.config
                    .blink_timing_profile_for(&self.pipeline.device_key),
            ));
        }
    }

    fn start_python_eyelid_dataset_capture(&mut self) {
        if crate::config::canonical_device_key(&self.pipeline.device_key) != "pimax_xr5" {
            self.dream_air_msg = Some((
                "The Python eyelid dataset recorder is available only for Dream Air / XR5.".into(),
                WARN,
            ));
            return;
        }
        let _ = self.start_evidence_capture(CaptureProtocol::PythonEyelidDataset);
    }

    fn update_reseat_assist(&mut self) {
        let geometry = self.config.geometry_for(&self.pipeline.device_key);
        // The running headset cannot change without rebuilding `App`; keep the
        // pseudonymous unit id captured at startup instead of probing USB every repaint.
        let context = ReseatReferenceContext::new(
            &self.pipeline.device_key,
            self.reseat_assist.context().unit_id.clone(),
            crate::reseat_assist::image_fingerprint(
                self.config.mapping_for(&self.pipeline.device_key),
                geometry,
            ),
        );
        self.reseat_assist.sync_context(context);
        if !self.reseat_assist.is_active() {
            return;
        }
        let results = *self.tele.results.lock().unwrap();
        let fallback_blinking = results
            .iter()
            .all(|eye| eye.openness_valid)
            .then(|| results.iter().all(|eye| eye.blink || eye.openness < 0.30));
        let was_capturing = self.reseat_assist.is_capturing();
        let history = self
            .tele
            .calibration_frames_after(self.reseat_assist.last_generation());
        if history.is_empty() {
            // Preserve the no-EyeNet fallback. Normal eyelid-capable builds use the
            // coherent 60 Hz history above.
            let native = self.tele.fresh_capture_sample();
            let gaze = [native.left, native.right].map(|eye| {
                (eye.gaze_reported
                    && eye.gaze_valid
                    && eye.gaze.iter().all(|value| value.is_finite()))
                .then_some(ReseatGazePoint::Direction(eye.gaze))
            });
            self.reseat_assist.update(
                &self.tele.stereo_frames(),
                gaze,
                fallback_blinking,
                Instant::now(),
            );
        } else {
            for sample in history {
                let native = sample.gaze;
                let gaze = [native.left, native.right].map(|eye| {
                    // Pupil position also moves with the HMD. Eye-relative gaze is
                    // the only native signal safe enough to gate a reseat estimate.
                    (eye.gaze_reported
                        && eye.gaze_valid
                        && eye.gaze.iter().all(|value| value.is_finite()))
                    .then_some(ReseatGazePoint::Direction(eye.gaze))
                });
                let eyes = [native.left, native.right];
                let native_blinking = eyes
                    .iter()
                    .all(|eye| eye.openness_reported && eye.openness_valid)
                    .then(|| eyes.iter().all(|eye| eye.openness < 0.30));
                self.reseat_assist.update(
                    &sample.stereo_frames(),
                    gaze,
                    native_blinking.or(fallback_blinking),
                    sample.captured_at,
                );
                self.reseat_assist.discard_through(sample.source_generation);
                if !self.reseat_assist.is_active() {
                    break;
                }
            }
        }
        if was_capturing && !self.reseat_assist.is_capturing() {
            let cue = if matches!(
                self.reseat_assist.guidance(),
                ReseatGuidance::ReferenceReady { .. }
            ) {
                RecordingCue::Complete
            } else {
                RecordingCue::Warning
            };
            self.recording_audio
                .cue(cue, self.config.ui.recording_audio_cues);
        }
    }

    fn start_evidence_capture(&mut self, protocol: CaptureProtocol) -> bool {
        if self.pending_unified_review() {
            self.dream_air_msg = Some((
                "Review or finish the current Initial setup before starting another calibration."
                    .into(),
                WARN,
            ));
            return false;
        }
        if self.gaze_residual_capture.is_done() && self.gaze_residual_recording_path.is_none() {
            self.dream_air_msg = Some((
                "Retry saving or explicitly discard the existing calibration recording first."
                    .into(),
                ERR,
            ));
            return false;
        }
        if self.geometry_capture.is_running()
            || self.geometry_capture.is_done()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.gaze_residual_capture.is_running()
            || self.gaze_eyelid_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self.wide.is_running()
            || self.brow.is_running()
            || self.wide_fitter.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return false;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before starting calibration.".into(),
                WARN,
            ));
            return false;
        }
        let (ready, detail) = if protocol == CaptureProtocol::PythonEyelidDataset {
            self.python_dataset_capture_ready()
        } else {
            self.endpoint_capture_ready()
        };
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return false;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.show_gaze_modal = false;
        self.gaze_center_capture = None;
        let generation = self.tele.frame_generations();
        self.gaze_residual_export_attempted = false;
        self.gaze_residual_recording_path = None;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.gaze_eyelid_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        let snapshot = self.current_gaze_residual_snapshot();
        if protocol != CaptureProtocol::PythonEyelidDataset
            && (snapshot.eyelid_model_crc32.is_none() || snapshot.eyelid_model_bytes.is_none())
        {
            self.dream_air_msg = Some((
                "The live eyelid model has no load-time fingerprint; reload SRanibro before recording calibration evidence."
                    .into(),
                ERR,
            ));
            return false;
        }
        self.gaze_residual_snapshot = Some(snapshot);
        self.vr_research_overlay.reset_session();
        self.gaze_residual_capture
            .start_protocol(protocol, generation);
        self.dream_air_msg = Some((
            match protocol {
                CaptureProtocol::GazeDirections => "Gaze-direction recording is ready. Read the protocol, then press Begin.",
                CaptureProtocol::InitialSetup => "Initial setup is ready. One recording will be reused across every eyelid check.",
                CaptureProtocol::PythonEyelidDataset => "XR5 Python dataset recording is ready. Read the biometric-data notice and protocol, then press Begin.",
            }
            .into(),
            ACCENT,
        ));
        true
    }

    fn begin_gaze_residual_capture(&mut self) {
        if self.gaze_residual_capture.begin() {
            let name = match self.gaze_residual_capture.protocol() {
                CaptureProtocol::GazeDirections => "Gaze-direction eyelid recording",
                CaptureProtocol::InitialSetup => "Initial eyelid setup",
                CaptureProtocol::PythonEyelidDataset => "XR5 Python eyelid dataset recording",
            };
            let next =
                if self.gaze_residual_capture.protocol() == CaptureProtocol::PythonEyelidDataset {
                    "The ZIP will save automatically; no live setting will change."
                } else {
                    "The ZIP will save automatically before analysis."
                };
            self.dream_air_msg = Some((format!("{name} started. {next}"), ACCENT));
        }
    }

    fn export_gaze_residual_recording(&mut self) -> bool {
        if self.gaze_residual_export_job.is_some() {
            return true;
        }
        self.gaze_residual_export_attempted = true;
        if let Some(run) = self.unified_calibration.as_mut() {
            if run.protocol == self.gaze_residual_capture.protocol() {
                run.stage = UnifiedStage::Saving;
            }
        }
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let protocol = self.gaze_residual_capture.protocol();
        let (
            capture_hz,
            session_kind,
            session_may_change,
            session_never_changes,
            target_repetitions,
            offline_python_dataset,
        ) = match protocol {
                CaptureProtocol::GazeDirections => (
                    "20",
                    "GazeDirections",
                    "gaze-dependent eyelid compensation",
                    "image geometry, photometric correction, endpoint calibration, native/avatar gaze, Wide, brow, wink response, blink timing",
                    2,
                    false,
                ),
                CaptureProtocol::InitialSetup => (
                    "mixed_20_60",
                    "InitialSetup",
                    "validated image/lighting correction, endpoint calibration, gaze-dependent eyelids, wink response, blink timing",
                    "native/avatar gaze, Wide, squeeze, brow",
                    2,
                    false,
                ),
                CaptureProtocol::PythonEyelidDataset => (
                    "mixed_20_60",
                    "PythonEyelidDataset",
                    "none",
                    "all live calibration, image geometry, model and output settings",
                    2,
                    true,
                ),
            };
        let path = crate::config::base_dir()
            .join("calibration-recordings")
            .join(format!(
                "sranibro_{}_{}_{stamp}.zip",
                self.pipeline.device_key,
                protocol.id()
            ));
        let partial_path = path.with_extension("zip.partial");
        let Some(snapshot) = self.gaze_residual_snapshot.as_ref() else {
            self.dream_air_msg = Some((
                "Research capture snapshot is missing; the ZIP was not written.".into(),
                ERR,
            ));
            return false;
        };
        let baseline = Some(snapshot.geometry);
        let filters = Some((snapshot.despeckle, snapshot.flatten));
        let mapping = snapshot.mapping;
        let mirrors = snapshot.mirrors;
        let model_crc32 = snapshot
            .eyelid_model_crc32
            .map(|value| format!("{value:08x}"))
            .unwrap_or_default();
        let model_bytes = snapshot
            .eyelid_model_bytes
            .map(|value| value.to_string())
            .unwrap_or_default();
        let metadata = format!(
            "schema_version=6\nsranibro_version={}\ndevice={}\nunit_id={}\ncapture_hz={capture_hz}\ncapture_protocol={}\nsession_kind={session_kind}\nsession_may_change={session_may_change}\nsession_never_changes={session_never_changes}\ntarget_repetitions_per_split={target_repetitions}\nbiometric_data=true\noffline_python_dataset={offline_python_dataset}\nproduction_state_changed=false\nconsent_required_before_model_import=true\npupil_pos_space=native_unmapped_when_available\nnative_freshness_limit_ms=150\nnative_pupil_sampling=freshness_gated_sample_hold\nnative_gaze_sampling=freshness_gated_sample_hold\nnative_openness_sampling=freshness_gated_sample_hold\ncommanded_target_space=categorical_target_labels_y_down\ntarget_presentation_requested={}\nsteamvr_target_visible_during_session={}\nsteamvr_target_horizontal_span_deg=+/-{:.1}\nsteamvr_target_vertical_span_deg=+/-{:.1}\nframe_stage=after_eye_mapping_before_ml_geometry\neyelid_model_crc32={model_crc32}\neyelid_model_bytes={model_bytes}\nbaseline_geometry={baseline:?}\nbaseline_photometric={:?}\nfilters={filters:?}\nbrightness_at_capture_start={:?}\nbrightness_affine_sampling=per_frame\neye_mapping={mapping:?}\nflip_gaze_x={}\nml_mirror={mirrors:?}\nwide_source={}\ngaze_source={}\n",
            env!("CARGO_PKG_VERSION"),
            snapshot.device_key,
            snapshot.unit_id,
            protocol.id(),
            if snapshot.steamvr_target_requested {
                "steamvr_head_locked_with_desktop_fallback"
            } else {
                "desktop_window"
            },
            self.vr_research_overlay.was_visible_during_session(),
            crate::vr_research_overlay::TARGET_HORIZONTAL_DEG,
            crate::vr_research_overlay::TARGET_VERTICAL_DEG,
            snapshot.photometric,
            snapshot.brightness,
            mapping.flip_gaze_x,
            snapshot.wide_source.as_str(),
            snapshot.gaze_source.as_str(),
        );
        let (work_tx, work_rx) = mpsc::sync_channel::<GazeResidualExportWork>(1);
        let (result_tx, result_rx) = mpsc::channel::<GazeResidualExportResult>();
        let worker = match std::thread::Builder::new()
            .name("eyelid-evidence-zip".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    // PNG encoding is bulk background work; keep camera/inference and
                    // the desktop compositor ahead of it under CPU contention.
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let Ok(work) = work_rx.recv() else { return };
                // Keep `work` outside the unwind boundary so even an encoder panic
                // returns the only in-memory biometric dataset to the UI for Retry.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::geometry_calib::export_dataset_recording(
                        &work.partial_path,
                        &work.dataset,
                        &work.metadata,
                    )
                    .and_then(|()| std::fs::rename(&work.partial_path, &work.path))
                    .map_err(|error| error.to_string())
                }))
                .unwrap_or_else(|_| Err("research ZIP encoder panicked".into()));
                if result.is_err() {
                    let _ = std::fs::remove_file(&work.partial_path);
                }
                let _ = result_tx.send(GazeResidualExportResult {
                    path: work.path,
                    dataset: work.dataset,
                    result,
                });
            }) {
            Ok(worker) => worker,
            Err(error) => {
                self.dream_air_msg = Some((
                    format!("Could not start the research ZIP worker: {error}"),
                    ERR,
                ));
                return false;
            }
        };

        let (dataset, metadata) = match self.gaze_residual_capture.take_export_dataset(&metadata) {
            Ok(payload) => payload,
            Err(error) => {
                drop(work_tx);
                let _ = worker.join();
                self.dream_air_msg = Some((
                    format!("Landmark/residual recording export failed: {error}"),
                    ERR,
                ));
                return false;
            }
        };
        let work = GazeResidualExportWork {
            path,
            partial_path,
            dataset,
            metadata,
        };
        if let Err(error) = work_tx.send(work) {
            let work = error.0;
            let _ = worker.join();
            self.gaze_residual_capture
                .restore_export_dataset(work.dataset);
            self.dream_air_msg = Some((
                "Research ZIP worker stopped before receiving the recording.".into(),
                ERR,
            ));
            return false;
        }
        self.gaze_residual_export_job = Some(result_rx);
        self.gaze_residual_export_thread = Some(worker);
        self.dream_air_msg = Some((
            if protocol == CaptureProtocol::PythonEyelidDataset {
                "Saving the XR5 Python model dataset ZIP in the background..."
            } else {
                "Saving landmark/residual recording ZIP in the background..."
            }
            .into(),
            ACCENT,
        ));
        true
    }

    fn poll_gaze_residual_export(&mut self) {
        let Some(receiver) = self.gaze_residual_export_job.as_ref() else {
            return;
        };
        let outcome = match receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.gaze_residual_export_job = None;
                if let Some(worker) = self.gaze_residual_export_thread.take() {
                    let _ = worker.join();
                }
                self.gaze_residual_recording_path = None;
                self.dream_air_msg = Some((
                    "Research ZIP worker stopped unexpectedly; no completed ZIP was published."
                        .into(),
                    ERR,
                ));
                return;
            }
        };
        self.gaze_residual_export_job = None;
        if let Some(worker) = self.gaze_residual_export_thread.take() {
            let _ = worker.join();
        }
        match outcome.result {
            Ok(()) => {
                self.gaze_residual_recording_path = Some(outcome.path.clone());
                self.recording_audio
                    .cue(RecordingCue::Saved, self.config.ui.recording_audio_cues);
                let Some(snapshot) = self.gaze_residual_snapshot.clone() else {
                    self.gaze_residual_capture
                        .restore_export_dataset(outcome.dataset);
                    self.dream_air_msg = Some((
                        "The recording ZIP was saved, but its frozen image-path snapshot was lost; record again before applying a correction."
                            .into(),
                        ERR,
                    ));
                    return;
                };
                match self.gaze_residual_capture.protocol() {
                    CaptureProtocol::GazeDirections => {
                        match self.start_gaze_eyelid_fit_dataset(outcome.dataset, snapshot) {
                            Ok(()) => {
                                self.gaze_residual_snapshot = None;
                                self.dream_air_msg = Some((
                                    format!(
                                        "Gaze-direction ZIP saved; holdout analysis started: {}",
                                        outcome.path.display()
                                    ),
                                    OK,
                                ));
                            }
                            Err((message, dataset)) => {
                                self.gaze_residual_capture.restore_export_dataset(dataset);
                                self.dream_air_msg = Some((message, ERR));
                            }
                        }
                    }
                    CaptureProtocol::InitialSetup => {
                        let evidence = SharedEvidence::from(outcome.dataset);
                        let Some(run) = self.unified_calibration.as_mut() else {
                            self.gaze_residual_capture
                                .restore_export_dataset(evidence.into_dataset());
                            self.dream_air_msg = Some((
                                "The shared recording was saved, but its workflow state was lost. Open Initial setup and record again."
                                    .into(),
                                ERR,
                            ));
                            return;
                        };
                        if run.protocol != CaptureProtocol::InitialSetup {
                            self.gaze_residual_capture
                                .restore_export_dataset(evidence.into_dataset());
                            self.dream_air_msg = Some((
                                "The saved recording does not match the open calibration workflow."
                                    .into(),
                                ERR,
                            ));
                            return;
                        }
                        run.evidence = Some(evidence);
                        run.snapshot = Some(snapshot);
                        self.gaze_residual_snapshot = None;
                        match self.start_unified_preprocessing_analysis() {
                            Ok(()) => {
                                self.dream_air_msg = Some((
                                    format!(
                                        "Initial setup ZIP saved; shared analysis started: {}",
                                        outcome.path.display()
                                    ),
                                    OK,
                                ));
                            }
                            Err(message) => {
                                self.set_unified_blocked(
                                    UnifiedStage::FinalReview,
                                    format!(
                                        "Initial analysis could not start; no settings changed. Record again after fixing this: {message}"
                                    ),
                                );
                                self.dream_air_msg = Some((message, ERR));
                            }
                        }
                    }
                    CaptureProtocol::PythonEyelidDataset => {
                        drop(outcome.dataset);
                        self.gaze_residual_snapshot = None;
                        self.dream_air_msg = Some((
                            format!(
                                "XR5 Python eyelid dataset saved; no live settings changed: {}",
                                outcome.path.display()
                            ),
                            OK,
                        ));
                    }
                }
            }
            Err(error) => {
                self.gaze_residual_capture
                    .restore_export_dataset(outcome.dataset);
                self.gaze_residual_recording_path = None;
                self.dream_air_msg = Some((
                    format!("Landmark/residual recording export failed: {error}"),
                    ERR,
                ));
            }
        }
    }

    fn start_gaze_eyelid_fit_dataset(
        &mut self,
        dataset: crate::geometry_calib::GeometryDataset,
        snapshot: GazeResidualSnapshot,
    ) -> Result<(), (String, crate::geometry_calib::GeometryDataset)> {
        let fail = |message: String, dataset| Err((message, dataset));
        if !self.gaze_residual_state_matches(&snapshot) {
            return fail(
                "The frozen image path changed after recording. The ZIP is safe, but this evidence cannot be applied; record again."
                    .into(),
                dataset,
            );
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            return fail("EyePrediction model path is missing.".into(), dataset);
        };
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => return fail(error, dataset),
            };
        let Some(current_endpoints) = *self.tele.calibration.lock().unwrap() else {
            return fail(
                "Live eyelid endpoints are not ready; keep tracking active, then record again."
                    .into(),
                dataset,
            );
        };
        let inputs = GazeEyelidFitInputs {
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            geometry: snapshot.geometry,
            mirrors: snapshot.mirrors,
            despeckle: snapshot.despeckle,
            flatten: snapshot.flatten,
            photometric: snapshot.photometric,
            current_endpoints,
        };
        match self.gaze_eyelid_fitter.start(inputs) {
            Ok(()) => Ok(()),
            Err(error) => fail(
                format!(
                    "Gaze-direction eyelid analysis could not start: {}",
                    error.message
                ),
                error.inputs.dataset.into_dataset(),
            ),
        }
    }

    fn unified_snapshot_compatible(&self, snapshot: &GazeResidualSnapshot) -> bool {
        snapshot.device_key == self.pipeline.device_key
            && *self.pipeline.geometry.lock().unwrap() == snapshot.geometry
            && *self.pipeline.photometric_correction.lock().unwrap() == snapshot.photometric
            && [
                self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
                self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
            ] == snapshot.mirrors
            && *self.pipeline.despeckle.lock().unwrap() == snapshot.despeckle
            && *self.pipeline.flatten.lock().unwrap() == snapshot.flatten
            && self.config.mapping_for(&self.pipeline.device_key) == snapshot.mapping
            && self.pipeline.eyelid_model_identity.map(|value| value.crc32)
                == snapshot.eyelid_model_crc32
            && self.pipeline.eyelid_model_identity.map(|value| value.bytes)
                == snapshot.eyelid_model_bytes
    }

    fn unified_profile_inputs_compatible(&self) -> bool {
        let Some(run) = self.unified_calibration.as_ref() else {
            return false;
        };
        let device = &self.pipeline.device_key;
        (run.gaze_profile.is_none()
            || self.config.gaze_eyelid_profile_for(device) == run.base_gaze_profile)
            && (run.wink_profile.is_none()
                || self.config.wink_profile_for(device) == run.base_wink_profile)
            && (run.blink_profile.is_none()
                || self.config.blink_timing_profile_for(device) == run.base_blink_profile)
    }

    fn unified_runtime_inputs_compatible(&self) -> bool {
        let Some(run) = self.unified_calibration.as_ref() else {
            return false;
        };
        let live_endpoints = *self.tele.calibration.lock().unwrap();
        let endpoints_match = match (run.frozen_endpoints, live_endpoints) {
            (Some(frozen), Some(live)) => calibration_fit_inputs_equal(frozen, live),
            (None, None) => true,
            _ => false,
        };
        let live_deadzone = self.pipeline.tuning.lock().unwrap().open_deadzone;
        endpoints_match && live_deadzone.to_bits() == run.open_deadzone.to_bits()
    }

    fn unified_evidence_snapshot(&self) -> Result<(SharedEvidence, GazeResidualSnapshot), String> {
        let run = self
            .unified_calibration
            .as_ref()
            .ok_or_else(|| "The initial-setup workflow is not active.".to_owned())?;
        let evidence = run
            .evidence
            .clone()
            .ok_or_else(|| "The shared recording is not available.".to_owned())?;
        let snapshot = run
            .snapshot
            .clone()
            .ok_or_else(|| "The frozen recording snapshot is not available.".to_owned())?;
        if !self.unified_snapshot_compatible(&snapshot) {
            return Err(
                "The HMD, model, mapping or image path changed after recording. The ZIP is safe, but a fresh recording is required."
                    .into(),
            );
        }
        if !self.unified_runtime_inputs_compatible() {
            return Err(
                "Recenter, eyelid endpoints or the open dead-zone changed after recording. No result from the older coordinate system can be applied; record again."
                    .into(),
            );
        }
        Ok((evidence, snapshot))
    }

    fn unified_model_snapshot(&self) -> Result<(PathBuf, Arc<[u8]>, u32, u64), String> {
        let model_path = self
            .config
            .ml_params_path()
            .filter(|path| path.is_file())
            .ok_or_else(|| "EyePrediction model path is missing.".to_owned())?;
        let (model_bytes, crc32, bytes) = self.geometry_model_snapshot(&model_path)?;
        Ok((model_path, model_bytes, crc32, bytes))
    }

    fn unified_selected_image_path(
        &self,
        snapshot: &GazeResidualSnapshot,
    ) -> (
        [crate::core::types::MlGeometry; 2],
        crate::core::types::PhotometricCorrection,
    ) {
        self.unified_calibration
            .as_ref()
            .map(|run| {
                (
                    run.geometry.unwrap_or(snapshot.geometry),
                    run.photometric.unwrap_or(snapshot.photometric),
                )
            })
            .unwrap_or((snapshot.geometry, snapshot.photometric))
    }

    fn set_unified_blocked(&mut self, stage: UnifiedStage, message: impl Into<String>) {
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = stage;
            run.blocked = Some(message.into());
        }
        if matches!(
            stage,
            UnifiedStage::ReviewEndpoints | UnifiedStage::FinalReview
        ) {
            self.release_unified_replay_evidence();
        }
    }

    fn start_unified_preprocessing_analysis(&mut self) -> Result<(), String> {
        let (evidence, snapshot) = self.unified_evidence_snapshot()?;
        let (model_path, model_bytes, expected_model_crc32, expected_model_bytes) =
            self.unified_model_snapshot()?;
        if self.pipeline.device_key == "pimax_xr5" {
            self.geometry_unvalidated_ack = false;
            self.geometry_fitter
                .start(GeometryFitInputs {
                    model_path,
                    model_bytes,
                    expected_model_crc32,
                    expected_model_bytes,
                    dataset: evidence,
                    baseline: snapshot.geometry,
                    mirrors: snapshot.mirrors,
                    despeckle: snapshot.despeckle,
                    flatten: snapshot.flatten,
                })
                .map_err(|error| {
                    format!(
                        "Image-alignment analysis could not start: {}",
                        error.message
                    )
                })?;
            if let Some(run) = self.unified_calibration.as_mut() {
                run.preprocessing = UnifiedPreprocessing::Geometry;
                run.stage = UnifiedStage::Preprocessing;
            }
            return Ok(());
        }
        if crate::config::supports_photometric_fit(&self.pipeline.device_key) {
            self.photometric_fitter
                .start(PhotometricFitInputs {
                    model_path,
                    model_bytes,
                    expected_model_crc32,
                    expected_model_bytes,
                    dataset: evidence,
                    geometry: snapshot.geometry,
                    mirrors: snapshot.mirrors,
                    despeckle: snapshot.despeckle,
                    flatten: snapshot.flatten,
                    baseline: snapshot.photometric,
                })
                .map_err(|error| format!("Lighting analysis could not start: {}", error.message))?;
            if let Some(run) = self.unified_calibration.as_mut() {
                run.preprocessing = UnifiedPreprocessing::Photometric;
                run.stage = UnifiedStage::Preprocessing;
            }
            return Ok(());
        }
        if let Some(run) = self.unified_calibration.as_mut() {
            run.preprocessing = UnifiedPreprocessing::None;
            run.notes
                .push("Image path: no automatic correction is needed for this HMD.".into());
        }
        self.start_unified_endpoint_analysis()
    }

    fn start_unified_endpoint_analysis(&mut self) -> Result<(), String> {
        let (evidence, snapshot) = self.unified_evidence_snapshot()?;
        let (model_path, model_bytes, expected_model_crc32, expected_model_bytes) =
            self.unified_model_snapshot()?;
        let (geometry, photometric) = self.unified_selected_image_path(&snapshot);
        let current = self
            .unified_calibration
            .as_ref()
            .and_then(|run| run.endpoints)
            .or_else(|| *self.tele.calibration.lock().unwrap())
            .ok_or_else(|| "Live eyelid endpoints are not available.".to_owned())?;
        self.endpoint_fitter
            .start(EndpointFitInputs {
                model_path,
                model_bytes,
                expected_model_crc32,
                expected_model_bytes,
                dataset: evidence,
                geometry,
                mirrors: snapshot.mirrors,
                despeckle: snapshot.despeckle,
                flatten: snapshot.flatten,
                photometric,
                current,
                open_deadzone: self
                    .unified_calibration
                    .as_ref()
                    .map(|run| run.open_deadzone)
                    .ok_or_else(|| "The calibration workflow is not active.".to_owned())?,
            })
            .map_err(|error| format!("Endpoint analysis could not start: {}", error.message))?;
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = UnifiedStage::Endpoints;
        }
        Ok(())
    }

    fn start_unified_gaze_analysis(&mut self) -> Result<(), String> {
        let (evidence, snapshot) = self.unified_evidence_snapshot()?;
        let (_, model_bytes, expected_model_crc32, expected_model_bytes) =
            self.unified_model_snapshot()?;
        let (geometry, photometric) = self.unified_selected_image_path(&snapshot);
        let current_endpoints = self
            .unified_calibration
            .as_ref()
            .and_then(|run| run.endpoints)
            .ok_or_else(|| "Validated open / closed endpoints are unavailable.".to_owned())?;
        self.gaze_eyelid_fitter
            .start(GazeEyelidFitInputs {
                model_bytes,
                expected_model_crc32,
                expected_model_bytes,
                dataset: evidence,
                geometry,
                mirrors: snapshot.mirrors,
                despeckle: snapshot.despeckle,
                flatten: snapshot.flatten,
                photometric,
                current_endpoints,
            })
            .map_err(|error| {
                format!(
                    "Gaze-dependent eyelid analysis could not start: {}",
                    error.message
                )
            })?;
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = UnifiedStage::Gaze;
        }
        Ok(())
    }

    fn start_unified_wink_analysis(&mut self) -> Result<(), String> {
        let (evidence, snapshot) = self.unified_evidence_snapshot()?;
        let (_, model_bytes, expected_model_crc32, expected_model_bytes) =
            self.unified_model_snapshot()?;
        let (geometry, photometric) = self.unified_selected_image_path(&snapshot);
        let current_endpoints = self
            .unified_calibration
            .as_ref()
            .and_then(|run| run.endpoints)
            .ok_or_else(|| "Validated open / closed endpoints are unavailable.".to_owned())?;
        self.wink_fitter
            .start(WinkFitInputs {
                model_bytes,
                expected_model_crc32,
                expected_model_bytes,
                dataset: evidence,
                geometry,
                mirrors: snapshot.mirrors,
                despeckle: snapshot.despeckle,
                flatten: snapshot.flatten,
                photometric,
                current_endpoints,
                open_deadzone: self
                    .unified_calibration
                    .as_ref()
                    .map(|run| run.open_deadzone)
                    .ok_or_else(|| "The calibration workflow is not active.".to_owned())?,
            })
            .map_err(|error| format!("Wink analysis could not start: {}", error.message))?;
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = UnifiedStage::Winks;
        }
        Ok(())
    }

    fn start_unified_blink_analysis(&mut self) -> Result<(), String> {
        let (evidence, snapshot) = self.unified_evidence_snapshot()?;
        let (_, model_bytes, expected_model_crc32, expected_model_bytes) =
            self.unified_model_snapshot()?;
        let (geometry, photometric) = self.unified_selected_image_path(&snapshot);
        let current_endpoints = self
            .unified_calibration
            .as_ref()
            .and_then(|run| run.endpoints)
            .ok_or_else(|| "Validated open / closed endpoints are unavailable.".to_owned())?;
        self.blink_timing_fitter
            .start(BlinkTimingFitInputs {
                model_bytes,
                expected_model_crc32,
                expected_model_bytes,
                dataset: evidence,
                geometry,
                mirrors: snapshot.mirrors,
                despeckle: snapshot.despeckle,
                flatten: snapshot.flatten,
                photometric,
                current_endpoints,
                open_deadzone: self
                    .unified_calibration
                    .as_ref()
                    .map(|run| run.open_deadzone)
                    .ok_or_else(|| "The calibration workflow is not active.".to_owned())?,
            })
            .map_err(|error| {
                format!("Natural-blink analysis could not start: {}", error.message)
            })?;
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = UnifiedStage::Blinks;
        }
        Ok(())
    }

    fn apply_endpoint_request_to_store(
        mut store: crate::core::eye_state::CalibStore,
        request: &crate::endpoint_fit::EndpointApplyRequest,
    ) -> crate::core::eye_state::CalibStore {
        for (index, apply) in request.eyes.iter().enumerate() {
            let Some(apply) = apply else { continue };
            let snapshot = if index == 0 {
                &mut store.left
            } else {
                &mut store.right
            };
            snapshot.baseline = apply.baseline;
            snapshot.blink_depth = apply.blink_depth;
            snapshot.learned_once = true;
            snapshot.endpoint_locked = true;
            snapshot.endpoint_calibrated_unix = apply.calibrated_unix;
        }
        store
    }

    fn update_unified_calibration(&mut self) {
        let Some((protocol, stage)) = self
            .unified_calibration
            .as_ref()
            .map(|run| (run.protocol, run.stage))
        else {
            return;
        };
        match protocol {
            CaptureProtocol::InitialSetup => match stage {
                UnifiedStage::Preprocessing => self.advance_unified_preprocessing(),
                UnifiedStage::Endpoints => self.advance_unified_endpoints(),
                UnifiedStage::Gaze => self.advance_unified_gaze(),
                UnifiedStage::Winks => self.advance_unified_winks(),
                UnifiedStage::Blinks => self.advance_unified_blinks(),
                _ => {}
            },
            CaptureProtocol::GazeDirections | CaptureProtocol::PythonEyelidDataset => {}
        }
    }

    fn advance_unified_preprocessing(&mut self) {
        let preprocessing = self
            .unified_calibration
            .as_ref()
            .map(|run| run.preprocessing)
            .unwrap_or(UnifiedPreprocessing::None);
        let finished = match preprocessing {
            UnifiedPreprocessing::Geometry => match self.geometry_fitter.status() {
                GeometryFitStatus::Done { result, .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        if result.accepted {
                            run.geometry = Some(result.candidate);
                            run.notes.push(format!(
                                "Image alignment: validated candidate staged ({:+.3} holdout).",
                                result.holdout_improvement
                            ));
                        } else {
                            run.notes.push(format!(
                                "Image alignment: kept current ({:+.3} holdout; {}).",
                                result.holdout_improvement, result.reason
                            ));
                        }
                    }
                    true
                }
                GeometryFitStatus::Failed { message, .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        run.notes
                            .push(format!("Image alignment: kept current ({message})."));
                    }
                    true
                }
                GeometryFitStatus::Cancelled { .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        run.notes
                            .push("Image alignment: analysis cancelled; kept current.".into());
                    }
                    true
                }
                _ => false,
            },
            UnifiedPreprocessing::Photometric => match self.photometric_fitter.status() {
                PhotometricStatus::Done { result, .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        if result.accepted {
                            run.photometric = Some(result.candidate);
                            run.notes.push(format!(
                                "Lighting correction: validated candidate staged ({:+.3} holdout).",
                                result.holdout_improvement
                            ));
                        } else {
                            run.notes.push(format!(
                                "Lighting correction: kept current ({:+.3} holdout; {}).",
                                result.holdout_improvement, result.reason
                            ));
                        }
                    }
                    true
                }
                PhotometricStatus::Failed { message, .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        run.notes
                            .push(format!("Lighting correction: kept current ({message})."));
                    }
                    true
                }
                PhotometricStatus::Cancelled { .. } => {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        run.notes
                            .push("Lighting correction: analysis cancelled; kept current.".into());
                    }
                    true
                }
                _ => false,
            },
            UnifiedPreprocessing::None => true,
        };
        if finished {
            if let Err(message) = self.start_unified_endpoint_analysis() {
                self.set_unified_blocked(UnifiedStage::ReviewEndpoints, message);
            }
        }
    }

    fn advance_unified_endpoints(&mut self) {
        let status = self.endpoint_fitter.status();
        let finished = match status {
            EndpointFitStatus::Done { result } => {
                let current = self
                    .unified_calibration
                    .as_ref()
                    .and_then(|run| run.endpoints)
                    .or_else(|| *self.tele.calibration.lock().unwrap());
                let mut staged = current;
                let eyes = result.accepted_eyes();
                if eyes.iter().any(|accepted| *accepted) {
                    let permit = CommitPermit::<EndpointChange>::new(unified_session_id());
                    if let Some(request) = endpoint_commit_request(&result, eyes, permit) {
                        staged = staged
                            .map(|store| Self::apply_endpoint_request_to_store(store, &request));
                        if let Some(run) = self.unified_calibration.as_mut() {
                            run.endpoint_apply = Some(request);
                            run.notes.push(format!(
                                "Open / closed range: validated {} staged.",
                                accepted_eye_text(eyes)
                            ));
                        }
                    }
                } else if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Open / closed range: no candidate passed; kept current.".into());
                }
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.endpoints = staged;
                }
                true
            }
            EndpointFitStatus::Failed { message } => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push(format!("Open / closed range: kept current ({message})."));
                }
                true
            }
            EndpointFitStatus::Cancelled => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Open / closed range: analysis cancelled; kept current.".into());
                }
                true
            }
            _ => false,
        };
        if !finished {
            return;
        }
        let endpoints_ready = self
            .unified_calibration
            .as_ref()
            .and_then(|run| run.endpoints)
            .is_some_and(|store| store.left.endpoint_locked && store.right.endpoint_locked);
        if !endpoints_ready {
            self.set_unified_blocked(
                UnifiedStage::ReviewEndpoints,
                "Both eyes need a validated open / closed range before the remaining checks. Use Fix a problem -> Eyelid open / closed range, then rerun Initial setup.",
            );
            return;
        }
        if let Err(message) = self.start_unified_gaze_analysis() {
            self.set_unified_blocked(UnifiedStage::ReviewEndpoints, message);
        }
    }

    fn advance_unified_gaze(&mut self) {
        let finished = match self.gaze_eyelid_fitter.status() {
            GazeEyelidFitStatus::Done { result } => {
                let eyes = result.accepted_eyes();
                if eyes.iter().any(|accepted| *accepted) {
                    let permit = CommitPermit::<GazeEyelidChange>::new(unified_session_id());
                    let current = self
                        .unified_calibration
                        .as_ref()
                        .map(|run| run.base_gaze_profile)
                        .unwrap_or_else(|| {
                            self.config
                                .gaze_eyelid_profile_for(&self.pipeline.device_key)
                        });
                    if let Some(request) =
                        gaze_eyelid_commit_request(&result, eyes, current, permit)
                    {
                        if let Some(run) = self.unified_calibration.as_mut() {
                            run.gaze_profile = Some(request.profile);
                            run.notes.push(format!(
                                "Eyelids while looking around: validated {} staged.",
                                accepted_eye_text(eyes)
                            ));
                        }
                    }
                } else if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes.push(
                        "Eyelids while looking around: no repeatable correction; kept current."
                            .into(),
                    );
                }
                true
            }
            GazeEyelidFitStatus::Failed { message } => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes.push(format!(
                        "Eyelids while looking around: kept current ({message})."
                    ));
                }
                true
            }
            GazeEyelidFitStatus::Cancelled => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes.push(
                        "Eyelids while looking around: analysis cancelled; kept current.".into(),
                    );
                }
                true
            }
            _ => false,
        };
        if finished {
            if let Err(message) = self.start_unified_wink_analysis() {
                self.set_unified_blocked(UnifiedStage::ReviewEndpoints, message);
            }
        }
    }

    fn advance_unified_winks(&mut self) {
        let finished = match self.wink_fitter.status() {
            WinkFitStatus::Done { result } => {
                let eyes = result.accepted_eyes();
                if eyes.iter().any(|accepted| *accepted) {
                    let permit = CommitPermit::<WinkChange>::new(unified_session_id());
                    let current = self
                        .unified_calibration
                        .as_ref()
                        .map(|run| run.base_wink_profile)
                        .unwrap_or_else(|| self.config.wink_profile_for(&self.pipeline.device_key));
                    if let Some(request) = wink_commit_request(&result, eyes, current, permit) {
                        if let Some(run) = self.unified_calibration.as_mut() {
                            run.wink_profile = Some(request.profile);
                            run.notes.push(format!(
                                "Left / right wink: validated {} staged.",
                                accepted_eye_text(eyes)
                            ));
                        }
                    }
                } else if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Left / right wink: no candidate passed; kept current.".into());
                }
                true
            }
            WinkFitStatus::Failed { message } => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push(format!("Left / right wink: kept current ({message})."));
                }
                true
            }
            WinkFitStatus::Cancelled => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Left / right wink: analysis cancelled; kept current.".into());
                }
                true
            }
            _ => false,
        };
        if finished {
            if let Err(message) = self.start_unified_blink_analysis() {
                self.set_unified_blocked(UnifiedStage::ReviewEndpoints, message);
            }
        }
    }

    fn advance_unified_blinks(&mut self) {
        let finished = match self.blink_timing_fitter.status() {
            BlinkTimingFitStatus::Done { result } => {
                if result.accepted {
                    if let Some(run) = self.unified_calibration.as_mut() {
                        run.blink_profile = Some(result.profile);
                        run.notes.push(format!(
                            "Blink timing: validated {:.0} ms visible bottom staged.",
                            result.profile.min_closed_ms
                        ));
                    }
                } else if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Blink timing: holdout did not pass; kept current.".into());
                }
                true
            }
            BlinkTimingFitStatus::Failed { message } => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push(format!("Blink timing: kept current ({message})."));
                }
                true
            }
            BlinkTimingFitStatus::Cancelled => {
                if let Some(run) = self.unified_calibration.as_mut() {
                    run.notes
                        .push("Blink timing: analysis cancelled; kept current.".into());
                }
                true
            }
            _ => false,
        };
        if finished {
            if let Some(run) = self.unified_calibration.as_mut() {
                run.stage = UnifiedStage::FinalReview;
            }
            // Every candidate has been copied into the small staged result. Raw
            // biometric frames are no longer needed for Apply/Keep, so do not retain
            // the large shared capture while the review window is left open.
            self.release_unified_replay_evidence();
        }
    }

    fn cancel_unified_analysis_keep_current(&mut self) {
        self.geometry_fitter.cancel();
        self.photometric_fitter.cancel();
        self.endpoint_fitter.cancel();
        self.gaze_eyelid_fitter.cancel();
        self.wink_fitter.cancel();
        self.blink_timing_fitter.cancel();
        self.reseat_assist.stop();
        self.finish_unified_without_changes();
    }

    fn release_unified_replay_evidence(&mut self) {
        if let Some(run) = self.unified_calibration.as_mut() {
            run.evidence = None;
        }
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.gaze_eyelid_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
    }

    fn release_unified_evidence(&mut self) {
        self.release_unified_replay_evidence();
        if let Some(run) = self.unified_calibration.as_mut() {
            run.snapshot = None;
        }
    }

    fn commit_unified_calibration(&mut self) {
        let Some(run) = self.unified_calibration.as_ref() else {
            return;
        };
        let Some(snapshot) = run.snapshot.as_ref() else {
            self.set_unified_blocked(
                UnifiedStage::FinalReview,
                "The frozen recording snapshot is missing; nothing was applied.",
            );
            return;
        };
        if !self.unified_snapshot_compatible(snapshot) {
            self.set_unified_blocked(
                UnifiedStage::FinalReview,
                "The live image path changed after recording. Nothing was applied; record again.",
            );
            return;
        }
        if !self.unified_runtime_inputs_compatible() {
            self.set_unified_blocked(
                UnifiedStage::FinalReview,
                "Recenter, eyelid endpoints or the open dead-zone changed after recording. Nothing was applied; record again.",
            );
            return;
        }
        if !self.unified_profile_inputs_compatible() {
            self.set_unified_blocked(
                UnifiedStage::FinalReview,
                "A staged eyelid correction was edited separately after recording. Nothing was overwritten; record Initial setup again.",
            );
            return;
        }
        let geometry = run.geometry;
        let photometric = run.photometric;
        let gaze_profile = run.gaze_profile;
        let wink_profile = run.wink_profile;
        let blink_profile = run.blink_profile;
        let staged_endpoints = run.endpoints;
        let endpoint_changed = run.endpoint_apply.is_some();
        let change_count = run.staged_change_count();
        if change_count == 0 {
            if let Some(run) = self.unified_calibration.as_mut() {
                run.stage = UnifiedStage::Complete;
            }
            self.release_unified_evidence();
            self.dream_air_msg = Some((
                "Initial setup complete; all current settings were retained.".into(),
                OK,
            ));
            return;
        }
        let backup = match crate::config::create_state_backup("before-initial-setup") {
            Ok(path) => path,
            Err(error) => {
                self.set_unified_blocked(
                    UnifiedStage::FinalReview,
                    format!("Backup failed; nothing was applied: {error}"),
                );
                return;
            }
        };
        let previous = self.config.clone();
        let previous_endpoints = *self.tele.calibration.lock().unwrap();
        let device = self.pipeline.device_key.clone();
        if let Some(value) = geometry {
            self.config.set_geometry(&device, value);
        }
        if let Some(value) = photometric {
            self.config.set_photometric_correction(&device, value);
        }
        if let Some(value) = gaze_profile {
            self.config.set_gaze_eyelid_profile(&device, value);
        }
        if let Some(value) = wink_profile {
            self.config.set_wink_profile(&device, value);
        }
        if let Some(value) = blink_profile {
            self.config.set_blink_timing_profile(&device, value);
        }
        let calib_path = crate::config::calib_path_for(&device)
            .to_string_lossy()
            .into_owned();
        if endpoint_changed {
            let Some(store) = staged_endpoints else {
                self.config = previous;
                self.set_unified_blocked(
                    UnifiedStage::FinalReview,
                    "Validated endpoints were staged without a complete calibration store; nothing was applied.",
                );
                return;
            };
            if let Err(error) = crate::core::eye_state::save_calib_checked(&calib_path, &store) {
                self.config = previous;
                self.set_unified_blocked(
                    UnifiedStage::FinalReview,
                    format!("Endpoint save failed; nothing was applied: {error}"),
                );
                return;
            }
        }
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.config = previous;
            let rollback = if endpoint_changed {
                previous_endpoints
                    .ok_or_else(|| "the previous endpoint snapshot was unavailable".to_owned())
                    .and_then(|store| {
                        crate::core::eye_state::save_calib_checked(&calib_path, &store)
                            .map_err(|rollback| rollback.to_string())
                    })
            } else {
                Ok(())
            };
            let detail = match rollback {
                Ok(()) => format!("Config save failed; nothing was applied: {error}"),
                Err(rollback) => format!(
                    "Config save failed and endpoint rollback also failed ({rollback}). Live tracking was not changed; restore the safety backup if needed: {error}"
                ),
            };
            self.set_unified_blocked(UnifiedStage::FinalReview, detail);
            return;
        }
        if let Some(value) = geometry {
            self.set_live_geometry(value);
        }
        if let Some(value) = photometric {
            *self.pipeline.photometric_correction.lock().unwrap() = value;
        }
        let endpoint_apply = self
            .unified_calibration
            .as_mut()
            .and_then(|run| run.endpoint_apply.take());
        if let Some(request) = endpoint_apply {
            *self.pipeline.endpoint_apply.lock().unwrap() = Some(request);
        }
        if let Some(profile) = gaze_profile {
            *self.pipeline.gaze_eyelid_apply.lock().unwrap() =
                Some(crate::gaze_eyelid_fit::ApplyRequest { profile });
        }
        if let Some(profile) = wink_profile {
            *self.pipeline.wink_apply.lock().unwrap() =
                Some(crate::wink_fit::ApplyRequest { profile });
        }
        if let Some(profile) = blink_profile {
            *self.pipeline.blink_timing_apply.lock().unwrap() =
                Some(crate::blink_timing_fit::ApplyRequest { profile });
        }
        if let Some(run) = self.unified_calibration.as_mut() {
            run.stage = UnifiedStage::Complete;
            run.notes.push(format!(
                "Applied {change_count} validated change group(s). Backup: {}",
                backup.display()
            ));
        }
        self.release_unified_evidence();
        self.dream_air_msg = Some((
            format!(
                "Initial setup applied {change_count} validated change group(s). Safety backup: {}",
                backup.display()
            ),
            OK,
        ));
    }

    fn finish_unified_without_changes(&mut self) {
        if let Some(run) = self.unified_calibration.as_mut() {
            run.geometry = None;
            run.photometric = None;
            run.endpoint_apply = None;
            run.gaze_profile = None;
            run.wink_profile = None;
            run.blink_profile = None;
            run.stage = UnifiedStage::Complete;
            run.notes
                .push("Review finished without applying staged candidates.".into());
        }
        self.release_unified_evidence();
        self.dream_air_msg = Some((
            "Initial setup finished; existing settings were kept.".into(),
            WARN,
        ));
    }

    fn gaze_residual_export_failed(&self) -> bool {
        self.gaze_residual_capture.is_done()
            && self.gaze_residual_export_attempted
            && self.gaze_residual_export_job.is_none()
            && self.gaze_residual_recording_path.is_none()
    }

    fn discard_failed_gaze_residual_recording(&mut self) {
        if self.gaze_residual_export_job.is_some() {
            return;
        }
        let protocol = self.gaze_residual_capture.protocol();
        self.gaze_residual_capture.abort();
        self.gaze_residual_snapshot = None;
        self.gaze_residual_export_attempted = false;
        self.gaze_residual_recording_path = None;
        if self
            .unified_calibration
            .as_ref()
            .is_some_and(|run| run.protocol == protocol)
        {
            self.unified_calibration = None;
        }
        self.release_unified_evidence();
        self.dream_air_msg = Some((
            "Unsaved calibration frames were discarded. Start a fresh recording when ready.".into(),
            WARN,
        ));
    }

    fn current_gaze_residual_snapshot(&self) -> GazeResidualSnapshot {
        let serial = self
            .config
            .dream_air_profile_for(&self.pipeline.device_key)
            .and_then(|profile| profile.eyechip_serial.clone())
            .or_else(crate::device::usb::peek_serial);
        GazeResidualSnapshot {
            device_key: self.pipeline.device_key.clone(),
            unit_id: crate::diagnostics::pseudonymous_unit_id(serial.as_deref()),
            geometry: *self.pipeline.geometry.lock().unwrap(),
            mirrors: [
                self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
                self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
            ],
            despeckle: *self.pipeline.despeckle.lock().unwrap(),
            flatten: *self.pipeline.flatten.lock().unwrap(),
            brightness: *self.pipeline.brightness.lock().unwrap(),
            photometric: *self.pipeline.photometric_correction.lock().unwrap(),
            mapping: self.config.mapping_for(&self.pipeline.device_key),
            wide_source: self.config.hmd.wide_source,
            gaze_source: self.config.gaze_source_for(&self.pipeline.device_key),
            eyelid_model_crc32: self.pipeline.eyelid_model_identity.map(|value| value.crc32),
            eyelid_model_bytes: self.pipeline.eyelid_model_identity.map(|value| value.bytes),
            steamvr_target_requested: self.config.ui.steamvr_overlay,
        }
    }

    fn gaze_residual_state_unchanged(&self) -> bool {
        let Some(snapshot) = self.gaze_residual_snapshot.as_ref() else {
            return false;
        };
        self.gaze_residual_state_matches(snapshot)
    }

    fn gaze_residual_state_matches(&self, snapshot: &GazeResidualSnapshot) -> bool {
        snapshot.device_key == self.pipeline.device_key
            && *self.pipeline.geometry.lock().unwrap() == snapshot.geometry
            && [
                self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
                self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
            ] == snapshot.mirrors
            && *self.pipeline.despeckle.lock().unwrap() == snapshot.despeckle
            && *self.pipeline.flatten.lock().unwrap() == snapshot.flatten
            && *self.pipeline.photometric_correction.lock().unwrap() == snapshot.photometric
            && self.config.mapping_for(&self.pipeline.device_key) == snapshot.mapping
            && self.config.hmd.wide_source == snapshot.wide_source
            && self.config.gaze_source_for(&self.pipeline.device_key) == snapshot.gaze_source
            && self.pipeline.eyelid_model_identity.map(|value| value.crc32)
                == snapshot.eyelid_model_crc32
            && self.pipeline.eyelid_model_identity.map(|value| value.bytes)
                == snapshot.eyelid_model_bytes
    }

    /// The instructed point is part of the evidence, so it must remain visible even if
    /// the user changes pages or collapses the research card. The OS window is raised to
    /// AlwaysOnTop for the same interval by `update`.
    fn gaze_residual_capture_overlay(&mut self, ctx: &egui::Context) {
        let status = self.gaze_residual_capture.status();
        if !matches!(
            status,
            GazeResidualStatus::Ready { .. } | GazeResidualStatus::Running { .. }
        ) {
            return;
        }

        let mut begin = false;
        let mut continue_step = false;
        let mut cancel = false;
        let title = match self.gaze_residual_capture.protocol() {
            CaptureProtocol::GazeDirections => "Gaze-direction eyelid recording",
            CaptureProtocol::InitialSetup => "Initial eyelid setup",
            CaptureProtocol::PythonEyelidDataset => "XR5 Python eyelid dataset",
        };
        egui::Window::new(title)
            .id(egui::Id::new("xr5_residual_capture_overlay"))
            .anchor(Align2::CENTER_TOP, vec2(0.0, 44.0 * S))
            .order(egui::Order::Foreground)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .show(ctx, |ui| {
                ui.set_min_width(390.0 * S);
                match &status {
                    GazeResidualStatus::Ready { instruction } => {
                        ui.label(
                            egui::RichText::new("READY - RECORDING HAS NOT STARTED")
                                .monospace()
                                .strong()
                                .color(ACCENT),
                        );
                        ui.label(label(instruction));
                        ui.add_space(SP2);
                        ui.label(num(match self.gaze_residual_capture.protocol() {
                            CaptureProtocol::GazeDirections => {
                                "Move only your eyes toward each target. The explanation and practice screens are not recorded."
                            }
                            CaptureProtocol::InitialSetup => {
                                "Follow each eye or eyelid pose. Open, half, closed, gaze, wink and blink evidence is recorded once and reused."
                            }
                            CaptureProtocol::PythonEyelidDataset => {
                                "Follow every pose carefully. Raw biometric eye images and labels are saved for offline Python training; SRanibro settings will not change."
                            }
                        }));
                        ui.add_space(SP2);
                        if ui
                            .add_sized(
                                [210.0 * S, 32.0 * S],
                                egui::Button::new("Begin recording (Space)"),
                            )
                            .clicked()
                        {
                            begin = true;
                        }
                    }
                    GazeResidualStatus::Running {
                        progress,
                        remaining_s,
                        target,
                        holdout,
                        recording,
                        settling,
                        paused,
                        samples_in_phase,
                        stereo_stalled,
                        awaiting_confirmation,
                        phase_remaining_s,
                        instruction,
                        ..
                    } => {
                        ui.label(
                            egui::RichText::new(if *paused {
                                "PAUSED - restore the SRanibro window"
                            } else if *stereo_stalled {
                                "WAITING FOR FRESH STEREO FRAMES"
                            } else if *awaiting_confirmation {
                                "READY FOR THE NEXT ACTION"
                            } else if target.is_none() && !recording {
                                "GET READY"
                            } else if target.is_none() && *settling {
                                "SET EYELID POSE"
                            } else if target.is_none() {
                                "RECORDING EYELID POSE"
                            } else if !recording {
                                "PRACTICE - NOT RECORDED"
                            } else if *settling {
                                "SET TARGET"
                            } else {
                                "RECORDING"
                            })
                            .monospace()
                            .strong()
                            .color(if *stereo_stalled { ERR } else { ACCENT }),
                        );
                        ui.label(label(instruction));
                        if *awaiting_confirmation {
                            ui.add_space(SP2);
                            if ui
                                .add_sized(
                                    [210.0 * S, 32.0 * S],
                                    egui::Button::new("Continue (Space)"),
                                )
                                .clicked()
                            {
                                continue_step = true;
                            }
                        }
                        if let Some(target) = target {
                            let (rect, _) = ui.allocate_exact_size(
                                vec2(370.0 * S, 208.0 * S),
                                Sense::hover(),
                            );
                            let painter = ui.painter_at(rect);
                            painter.rect_filled(rect, R_INNER, INNER);
                            painter.rect_stroke(rect, R_INNER, Stroke::new(1.0, BORDER));
                            let [tx, ty] = target.screen_xy();
                            let point = pos2(
                                rect.center().x + tx * rect.width() * 0.45,
                                rect.center().y + ty * rect.height() * 0.43,
                            );
                            painter.circle_filled(point, 9.0 * S, ACCENT);
                            painter.circle_stroke(point, 14.0 * S, Stroke::new(1.5, TEXT1));
                        }
                        ui.label(num(&if *awaiting_confirmation {
                            "Read at your own pace. The timer and camera recording are stopped."
                                .to_owned()
                        } else if !recording && target.is_none() {
                            format!(
                                "Get into position    recording starts in {phase_remaining_s:.1}s"
                            )
                        } else if !recording {
                            format!("practice pass    no frames saved    {remaining_s:.1}s left")
                        } else {
                            format!(
                                "{} pass    {samples_in_phase} fresh samples    {remaining_s:.1}s left",
                                if *holdout {
                                    "untouched holdout"
                                } else {
                                    "training"
                                }
                            )
                        }));
                        ui.add(egui::ProgressBar::new(*progress).show_percentage());
                        if *stereo_stalled {
                            ui.label(
                                egui::RichText::new(
                                    "The phase timer is held until enough fresh stereo frames arrive.",
                                )
                                .monospace()
                                .color(ERR),
                            );
                        }
                    }
                    _ => {}
                }
                if ui.button("Cancel and discard recording").clicked() {
                    cancel = true;
                }
            });
        if begin {
            self.begin_gaze_residual_capture();
        }
        if continue_step {
            self.gaze_residual_capture.continue_step();
        }
        if cancel {
            self.gaze_residual_capture.abort();
            self.gaze_residual_snapshot = None;
            self.gaze_residual_export_attempted = false;
            self.gaze_residual_recording_path = None;
            if self
                .unified_calibration
                .as_ref()
                .is_some_and(|run| run.protocol == self.gaze_residual_capture.protocol())
            {
                self.unified_calibration = None;
            }
            self.dream_air_msg = Some(("Calibration recording cancelled.".into(), WARN));
        }
    }

    fn geometry_capture_ready(&self) -> (bool, String) {
        let dimensions = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|frame| (frame.width, frame.height)),
                frames[1].as_ref().map(|frame| (frame.width, frame.height)),
            ]
        };
        if dimensions != [Some((200, 200)); 2] {
            return (
                false,
                format!("Waiting for XR5 stereo 200x200 frames (now {dimensions:?})"),
            );
        }
        if !self.tele.ml_loaded {
            return (false, "The SRanipal eyelid model is not loaded.".into());
        }
        match self.config.ml_params_path() {
            Some(path) if path.is_file() => (true, "Camera and eyelid model are ready.".into()),
            Some(path) => (
                false,
                format!("EyePrediction model not found: {}", path.display()),
            ),
            None => (
                false,
                "Configure the SRanipal EyePrediction model first.".into(),
            ),
        }
    }

    fn photometric_capture_ready(&self) -> (bool, String) {
        if !crate::config::supports_photometric_fit(&self.pipeline.device_key) {
            return (
                false,
                "Photometric Fit is available only for frontal Hotmirror eye cameras (VR4/Varjo)."
                    .into(),
            );
        }
        let dimensions = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|frame| (frame.width, frame.height)),
                frames[1].as_ref().map(|frame| (frame.width, frame.height)),
            ]
        };
        if dimensions.iter().any(Option::is_none) {
            return (
                false,
                format!("Waiting for stereo eye-camera frames (now {dimensions:?})"),
            );
        }
        if !self.tele.ml_loaded {
            return (false, "The SRanipal eyelid model is not loaded.".into());
        }
        match self.config.ml_params_path() {
            Some(path) if path.is_file() => (
                true,
                "Frontal eye cameras and the eyelid model are ready.".into(),
            ),
            Some(path) => (
                false,
                format!("EyePrediction model not found: {}", path.display()),
            ),
            None => (
                false,
                "Configure the SRanipal EyePrediction model first.".into(),
            ),
        }
    }

    fn endpoint_capture_ready(&self) -> (bool, String) {
        let dimensions = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|frame| (frame.width, frame.height)),
                frames[1].as_ref().map(|frame| (frame.width, frame.height)),
            ]
        };
        if dimensions.iter().any(Option::is_none) {
            return (
                false,
                format!("Waiting for stereo eye-camera frames (now {dimensions:?})"),
            );
        }
        if !self.tele.ml_loaded {
            return (false, "The SRanipal eyelid model is not loaded.".into());
        }
        match self.config.ml_params_path() {
            Some(path) if path.is_file() => (
                true,
                "Stereo eye cameras and the eyelid model are ready.".into(),
            ),
            Some(path) => (
                false,
                format!("EyePrediction model not found: {}", path.display()),
            ),
            None => (
                false,
                "Configure the SRanipal EyePrediction model first.".into(),
            ),
        }
    }

    fn python_dataset_capture_ready(&self) -> (bool, String) {
        if crate::config::canonical_device_key(&self.pipeline.device_key) != "pimax_xr5" {
            return (
                false,
                "Connect a Pimax Dream Air / XR5 before recording this dataset.".into(),
            );
        }
        let dimensions = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|frame| (frame.width, frame.height)),
                frames[1].as_ref().map(|frame| (frame.width, frame.height)),
            ]
        };
        if dimensions != [Some((200, 200)); 2] {
            return (
                false,
                format!(
                    "Waiting for two native XR5 200x200 eye-camera frames (now {dimensions:?})"
                ),
            );
        }
        (
            true,
            "Native XR5 stereo cameras are ready. Python is not required on this recording PC."
                .into(),
        )
    }

    fn pending_unified_review(&self) -> bool {
        self.unified_calibration
            .as_ref()
            .is_some_and(|run| !matches!(run.stage, UnifiedStage::Complete))
    }

    fn reject_pending_unified_review(&mut self) -> bool {
        if !self.pending_unified_review() {
            return false;
        }
        self.dream_air_msg = Some((
            "Review or finish the current Initial setup before starting another calibration."
                .into(),
            WARN,
        ));
        true
    }

    fn start_geometry_capture(&mut self) {
        if self.reject_pending_unified_review() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || (self.gaze_residual_capture.is_done() && self.gaze_residual_recording_path.is_none())
        {
            self.dream_air_msg = Some((
                "Save or discard the landmark/residual recording first.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
        {
            self.dream_air_msg = Some(("Cancel the active image fit first.".into(), WARN));
            return;
        }
        if self.wide.is_running()
            || self.wide_fitter.is_running()
            || self.brow.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
            || self.gaze_center_capture.is_some()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before starting image-alignment capture.".into(),
                WARN,
            ));
            return;
        }
        let (ready, detail) = self.geometry_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        self.geometry_unvalidated_ack = false;
        self.reset_geometry_recording_export();
        let generation = self.tele.frame_generations();
        self.calibration_capture_purpose = Some(CalibrationCapturePurpose::Geometry);
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.photometric_capture_baseline = None;
        self.geometry_capture.start(generation);
        self.dream_air_msg = Some((
            "Guided image-alignment capture started; the completed recording will be saved automatically.".into(),
            ACCENT,
        ));
    }

    fn start_photometric_capture(&mut self) {
        if self.reject_pending_unified_review() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self.wide.is_running()
            || self.wide_fitter.is_running()
            || self.brow.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
            || self.gaze_center_capture.is_some()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before starting Photometric Fit capture.".into(),
                WARN,
            ));
            return;
        }
        let (ready, detail) = self.photometric_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        self.reset_geometry_recording_export();
        let generation = self.tele.frame_generations();
        self.calibration_capture_purpose = Some(CalibrationCapturePurpose::Photometric);
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.photometric_capture_baseline =
            Some(*self.pipeline.photometric_correction.lock().unwrap());
        self.geometry_capture.start(generation);
        self.dream_air_msg = Some((
            "Photometric Fit recording started. Geometry stays fixed; the completed biometric recording will be saved automatically."
                .into(),
            ACCENT,
        ));
    }

    fn start_eyelid_endpoint_capture(&mut self) {
        if self.reject_pending_unified_review() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self.wide.is_running()
            || self.wide_fitter.is_running()
            || self.brow.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
            || self.gaze_center_capture.is_some()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before recording eyelid endpoints.".into(),
                WARN,
            ));
            return;
        }
        let (ready, detail) = self.endpoint_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        self.reset_geometry_recording_export();
        let generation = self.tele.frame_generations();
        self.calibration_capture_purpose = Some(CalibrationCapturePurpose::EyelidEndpoints);
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.photometric_capture_baseline =
            Some(*self.pipeline.photometric_correction.lock().unwrap());
        self.geometry_capture
            .start_plan(CapturePlan::EyelidEndpoints, generation);
        self.dream_air_msg = Some((
            "Endpoint recording started. Hold each instructed pose; the ZIP is saved before analysis."
                .into(),
            ACCENT,
        ));
    }

    fn start_wink_capture(&mut self) {
        if self.reject_pending_unified_review() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self.wide.is_running()
            || self.wide_fitter.is_running()
            || self.brow.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
            || self.gaze_center_capture.is_some()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before recording winks.".into(),
                WARN,
            ));
            return;
        }
        let (ready, detail) = self.endpoint_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        let Some(endpoints) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some((
                "Live eyelid endpoints are not ready yet. Keep tracking active for a moment."
                    .into(),
                WARN,
            ));
            return;
        };
        if !endpoints.left.endpoint_locked || !endpoints.right.endpoint_locked {
            self.dream_air_msg = Some((
                "Calibrate and apply Open / closed endpoints first. Wink response is measured relative to those fixed endpoints."
                    .into(),
                WARN,
            ));
            return;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        self.reset_geometry_recording_export();
        let generation = self.tele.frame_generations();
        self.calibration_capture_purpose = Some(CalibrationCapturePurpose::Winks);
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.photometric_capture_baseline =
            Some(*self.pipeline.photometric_correction.lock().unwrap());
        self.geometry_capture
            .start_plan(CapturePlan::Winks, generation);
        self.dream_air_msg = Some((
            "Wink recording started. Hold only the instructed eye closed and keep the partner eye relaxed open."
                .into(),
            ACCENT,
        ));
    }

    fn start_natural_blink_capture(&mut self) {
        if self.reject_pending_unified_review() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving.".into(),
                WARN,
            ));
            return;
        }
        if self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self.wide.is_running()
            || self.wide_fitter.is_running()
            || self.brow.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running()
            || self.gaze_center_capture.is_some()
        {
            self.dream_air_msg = Some((
                "Finish or cancel the active calibration capture/fit first.".into(),
                WARN,
            ));
            return;
        }
        if self.pipeline.diag_rec.load(Ordering::Relaxed) {
            self.dream_air_msg = Some((
                "Stop the diagnostic CSV recorder before recording natural blinks.".into(),
                WARN,
            ));
            return;
        }
        let (ready, detail) = self.endpoint_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        let Some(endpoints) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some((
                "Live eyelid endpoints are not ready yet. Keep tracking active for a moment."
                    .into(),
                WARN,
            ));
            return;
        };
        if !endpoints.left.endpoint_locked || !endpoints.right.endpoint_locked {
            self.dream_air_msg = Some((
                "Calibrate and apply Open / closed endpoints first. Blink timing is validated against those fixed endpoints."
                    .into(),
                WARN,
            ));
            return;
        }
        self.restore_geometry_preview(false);
        self.show_geom_modal = false;
        self.geometry_fitter.clear_finished();
        self.photometric_fitter.clear_finished();
        self.endpoint_fitter.clear_finished();
        self.wink_fitter.clear_finished();
        self.blink_timing_fitter.clear_finished();
        self.reset_geometry_recording_export();
        let generation = self.tele.frame_generations();
        self.calibration_capture_purpose = Some(CalibrationCapturePurpose::NaturalBlinks);
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.photometric_capture_baseline =
            Some(*self.pipeline.photometric_correction.lock().unwrap());
        self.geometry_capture
            .start_plan(CapturePlan::NaturalBlinks, generation);
        self.dream_air_msg = Some((
            "Natural-blink recording started at 60 Hz. Blink normally only when instructed and fully relax open between blinks."
                .into(),
            ACCENT,
        ));
    }

    fn geometry_evidence_locked(&self) -> bool {
        self.geometry_capture.is_running()
            || self.geometry_capture.is_done()
            || self.geometry_recording_export_job.is_some()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
    }

    fn guided_capture_running(&self) -> bool {
        self.geometry_capture.is_running()
            || self.gaze_residual_capture.is_running()
            || self.wide.is_running()
            || self.brow.is_running()
    }

    fn geometry_filters_match_snapshot(&self) -> bool {
        self.geometry_capture_filters
            .is_some_and(|(despeckle, flatten)| {
                *self.pipeline.despeckle.lock().unwrap() == despeckle
                    && *self.pipeline.flatten.lock().unwrap() == flatten
            })
    }

    fn geometry_capture_state_unchanged(&self) -> bool {
        self.geometry_capture_baseline
            .is_some_and(|baseline| *self.pipeline.geometry.lock().unwrap() == baseline)
            && self.geometry_filters_match_snapshot()
            && match self.calibration_capture_purpose {
                Some(CalibrationCapturePurpose::Geometry) => true,
                Some(CalibrationCapturePurpose::Photometric) => {
                    self.photometric_capture_baseline.is_some_and(|baseline| {
                        *self.pipeline.photometric_correction.lock().unwrap() == baseline
                    })
                }
                Some(CalibrationCapturePurpose::EyelidEndpoints) => {
                    self.photometric_capture_baseline.is_some_and(|baseline| {
                        *self.pipeline.photometric_correction.lock().unwrap() == baseline
                    })
                }
                Some(CalibrationCapturePurpose::Winks) => {
                    self.photometric_capture_baseline.is_some_and(|baseline| {
                        *self.pipeline.photometric_correction.lock().unwrap() == baseline
                    })
                }
                Some(CalibrationCapturePurpose::NaturalBlinks) => {
                    self.photometric_capture_baseline.is_some_and(|baseline| {
                        *self.pipeline.photometric_correction.lock().unwrap() == baseline
                    })
                }
                None => false,
            }
    }

    fn reset_geometry_recording_export(&mut self) {
        debug_assert!(
            self.geometry_recording_export_job.is_none(),
            "cannot reset geometry export state while its worker owns the dataset"
        );
        self.geometry_recording_export_attempted = false;
        self.geometry_recording_path = None;
    }

    /// Move the current completed capture into a below-normal-priority ZIP worker. The
    /// result always carries the dataset back to the UI; even a successful export restores
    /// it because fit/audit still needs the exact frames written to disk.
    fn export_geometry_recording(&mut self) -> bool {
        if self.geometry_recording_export_job.is_some() {
            return true;
        }
        let Some(purpose) = self.calibration_capture_purpose else {
            self.dream_air_msg = Some((
                "Calibration recording has no workflow identity; discard it and record again."
                    .into(),
                ERR,
            ));
            return false;
        };
        self.geometry_recording_export_attempted = true;
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let (session_kind, filename_kind) = match purpose {
            CalibrationCapturePurpose::Geometry => (SessionKind::SafeGeometry, "geometry"),
            CalibrationCapturePurpose::Photometric => (SessionKind::Photometric, "photometric"),
            CalibrationCapturePurpose::EyelidEndpoints => {
                (SessionKind::EyelidEndpoints, "eyelid_endpoints")
            }
            CalibrationCapturePurpose::Winks => (SessionKind::Winks, "winks"),
            CalibrationCapturePurpose::NaturalBlinks => {
                (SessionKind::NaturalBlinks, "natural_blinks")
            }
        };
        let descriptor = session_kind.descriptor();
        let protocol = descriptor.protocol;
        let path = crate::config::base_dir()
            .join("calibration-recordings")
            .join(format!(
                "sranibro_{}_{}_{stamp}.zip",
                self.pipeline.device_key, filename_kind
            ));
        let partial_path = path.with_extension("zip.partial");
        let serial = self
            .config
            .dream_air_profile_for(&self.pipeline.device_key)
            .and_then(|profile| profile.eyechip_serial.clone())
            .or_else(crate::device::usb::peek_serial);
        let unit_id = crate::diagnostics::pseudonymous_unit_id(serial.as_deref());
        let baseline = self.geometry_capture_baseline;
        let filters = self.geometry_capture_filters;
        let baseline_photometric = self.photometric_capture_baseline;
        let mapping = self.config.mapping_for(&self.pipeline.device_key);
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let model_crc32 = self
            .pipeline
            .eyelid_model_identity
            .map(|value| format!("{:08x}", value.crc32))
            .unwrap_or_default();
        let model_bytes = self
            .pipeline
            .eyelid_model_identity
            .map(|value| value.bytes.to_string())
            .unwrap_or_default();
        let metadata = format!(
            "schema_version=2\nsranibro_version={}\ndevice={}\nunit_id={}\ncapture_hz={}\ncapture_protocol={protocol}\nsession_id=s{stamp}\nsession_kind={session_kind:?}\nsession_plan={}\nsession_may_change={}\nsession_never_changes={}\nframe_stage=after_eye_mapping_before_ml_geometry\neyelid_model_crc32={model_crc32}\neyelid_model_bytes={model_bytes}\nbaseline_geometry={baseline:?}\nbaseline_photometric={baseline_photometric:?}\nfilters={filters:?}\neye_mapping={mapping:?}\nflip_gaze_x={}\nml_mirror={mirrors:?}\nwide_source={:?}\ngaze_source={:?}\n",
            env!("CARGO_PKG_VERSION"),
            self.pipeline.device_key,
            unit_id,
            self.geometry_capture.plan().capture_hz(),
            self.geometry_capture.plan().id(),
            descriptor.may_change.user_text(),
            descriptor.never_changes.user_text(),
            mapping.flip_gaze_x,
            self.config.hmd.wide_source,
            self.config.gaze_source_for(&self.pipeline.device_key),
        );
        let (work_tx, work_rx) = mpsc::sync_channel::<GeometryExportWork>(1);
        let (result_tx, result_rx) = mpsc::channel::<GeometryExportResult>();
        let worker = match std::thread::Builder::new()
            .name("calibration-eye-zip".into())
            .spawn(move || {
                #[cfg(windows)]
                unsafe {
                    // PNG/ZIP encoding is bulk work. Keep live camera, inference and
                    // desktop-compositor threads ahead of it under CPU contention.
                    let thread = windows_sys::Win32::System::Threading::GetCurrentThread();
                    let _ = windows_sys::Win32::System::Threading::SetThreadPriority(
                        thread,
                        windows_sys::Win32::System::Threading::THREAD_PRIORITY_BELOW_NORMAL,
                    );
                }
                let Ok(work) = work_rx.recv() else { return };
                // Keep `work` outside the unwind boundary so an encoder panic still
                // returns ownership of the only in-memory capture to the UI.
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::geometry_calib::export_dataset_recording(
                        &work.partial_path,
                        &work.dataset,
                        &work.metadata,
                    )
                    .and_then(|()| std::fs::rename(&work.partial_path, &work.path))
                    .map_err(|error| error.to_string())
                }))
                .unwrap_or_else(|_| Err("calibration ZIP encoder panicked".into()));
                if result.is_err() {
                    let _ = std::fs::remove_file(&work.partial_path);
                }
                let _ = result_tx.send(GeometryExportResult {
                    path: work.path,
                    dataset: work.dataset,
                    result,
                });
            }) {
            Ok(worker) => worker,
            Err(error) => {
                self.geometry_recording_path = None;
                self.dream_air_msg = Some((
                    format!("Could not start the calibration ZIP worker: {error}"),
                    ERR,
                ));
                return false;
            }
        };

        let (dataset, metadata) = match self.geometry_capture.take_export_dataset(&metadata) {
            Ok(payload) => payload,
            Err(error) => {
                drop(work_tx);
                let _ = worker.join();
                self.geometry_recording_path = None;
                self.dream_air_msg =
                    Some((format!("Calibration recording export failed: {error}"), ERR));
                return false;
            }
        };
        let work = GeometryExportWork {
            path,
            partial_path,
            dataset,
            metadata,
        };
        if let Err(error) = work_tx.send(work) {
            let work = error.0;
            let _ = worker.join();
            self.geometry_capture.restore_export_dataset(work.dataset);
            self.geometry_recording_path = None;
            self.dream_air_msg = Some((
                "Calibration ZIP worker stopped before receiving the recording.".into(),
                ERR,
            ));
            return false;
        }
        self.geometry_recording_export_job = Some(result_rx);
        self.geometry_recording_export_thread = Some(worker);
        self.geometry_recording_path = None;
        self.dream_air_msg = Some((
            "Saving calibration recording ZIP in the background...".into(),
            ACCENT,
        ));
        true
    }

    fn poll_geometry_recording_export(&mut self) {
        let Some(receiver) = self.geometry_recording_export_job.as_ref() else {
            return;
        };
        let outcome = match receiver.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => {
                self.geometry_recording_export_job = None;
                if let Some(worker) = self.geometry_recording_export_thread.take() {
                    let _ = worker.join();
                }
                self.geometry_recording_path = None;
                self.dream_air_msg = Some((
                    "Calibration ZIP worker stopped unexpectedly; no completed ZIP was published."
                        .into(),
                    ERR,
                ));
                return;
            }
        };
        self.geometry_recording_export_job = None;
        if let Some(worker) = self.geometry_recording_export_thread.take() {
            let _ = worker.join();
        }

        // Restore before publishing either outcome. Fit/audit and Retry must see the
        // exact dataset that the worker encoded, never an empty placeholder.
        self.geometry_capture
            .restore_export_dataset(outcome.dataset);
        match outcome.result {
            Ok(()) => {
                self.geometry_recording_path = Some(outcome.path.clone());
                self.recording_audio
                    .cue(RecordingCue::Saved, self.config.ui.recording_audio_cues);
                self.dream_air_msg = Some((
                    format!(
                        "Calibration recording ZIP saved automatically: {}",
                        outcome.path.display()
                    ),
                    OK,
                ));
                self.start_saved_calibration_analysis();
            }
            Err(error) => {
                self.geometry_recording_path = None;
                self.dream_air_msg =
                    Some((format!("Calibration recording export failed: {error}"), ERR));
            }
        }
    }

    /// A problem/session Start button owns the complete recording-to-review flow.
    /// ZIP durability is the boundary: analysis never consumes in-memory biometric
    /// evidence until the automatic export has completed successfully.
    fn start_saved_calibration_analysis(&mut self) {
        let Some(purpose) = self.calibration_capture_purpose else {
            return;
        };
        match purpose.session_kind() {
            SessionKind::SafeGeometry => self.start_geometry_fit(),
            SessionKind::Photometric => self.start_photometric_fit(),
            SessionKind::EyelidEndpoints => self.start_endpoint_fit(),
            SessionKind::Winks => self.start_wink_fit(),
            SessionKind::NaturalBlinks => self.start_blink_timing_fit(),
            SessionKind::GazeDirections => {
                unreachable!("gaze capture has its own saved-dataset path")
            }
        }
    }

    fn ensure_geometry_recording_saved(&mut self) -> bool {
        if self.geometry_recording_path.is_some() {
            return true;
        }
        if self.geometry_recording_export_job.is_some() {
            self.dream_air_msg = Some((
                "Wait for the calibration recording ZIP to finish saving before fit/audit.".into(),
                WARN,
            ));
            return false;
        }
        // A newly queued async save is not durable until poll publishes its atomic
        // rename, so the caller must wait for the next completed UI poll.
        self.export_geometry_recording();
        false
    }

    fn geometry_model_snapshot(
        &self,
        model_path: &std::path::Path,
    ) -> Result<(Arc<[u8]>, u32, u64), String> {
        let expected = self.pipeline.eyelid_model_identity.ok_or_else(|| {
            "The live EyeNet has no load-time identity; reload SRanibro before fitting.".to_owned()
        })?;
        let bytes = std::fs::read(model_path)
            .map_err(|error| format!("Read EyePrediction model: {error}"))?;
        let crc32 = crate::diagnostics::crc32_fingerprint(&bytes);
        if crc32 != expected.crc32 || bytes.len() as u64 != expected.bytes {
            return Err(
                "EyePrediction model changed after SRanibro loaded it. Apply & reload, then record again."
                    .into(),
            );
        }
        Ok((Arc::from(bytes), expected.crc32, expected.bytes))
    }

    fn start_geometry_fit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::Geometry) {
            self.dream_air_msg = Some((
                "This recording belongs to a different calibration workflow.".into(),
                ERR,
            ));
            return;
        }
        if self.gaze_residual_capture.is_running() {
            self.dream_air_msg = Some((
                "Finish or cancel the landmark/residual recording first.".into(),
                WARN,
            ));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(baseline) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture baseline is missing; record again.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some((
                "Capture filter snapshot is missing; record again.".into(),
                ERR,
            ));
            return;
        };
        if *self.pipeline.geometry.lock().unwrap() != baseline {
            self.dream_air_msg = Some((
                "Image geometry changed after recording. Discard this capture and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        if !self.geometry_filters_match_snapshot() {
            self.dream_air_msg = Some((
                "Image filters changed after recording. Discard this capture and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some(("Finish the guided capture before fitting.".into(), WARN));
            return;
        };
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let inputs = GeometryFitInputs {
            model_path,
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            baseline,
            mirrors,
            despeckle,
            flatten,
        };
        self.geometry_unvalidated_ack = false;
        match self.geometry_fitter.start(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Geometry search started. Tracking stays live; fitting may take several minutes."
                        .into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Geometry fit could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn start_geometry_audit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::Geometry) {
            self.dream_air_msg = Some((
                "This recording belongs to a different calibration workflow.".into(),
                ERR,
            ));
            return;
        }
        if self.gaze_residual_capture.is_running() {
            self.dream_air_msg = Some((
                "Finish or cancel the landmark/residual recording first.".into(),
                WARN,
            ));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(baseline) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture baseline is missing; record again.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some((
                "Capture filter snapshot is missing; record again.".into(),
                ERR,
            ));
            return;
        };
        if *self.pipeline.geometry.lock().unwrap() != baseline {
            self.dream_air_msg = Some((
                "Image geometry changed after recording. Discard this capture and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        if !self.geometry_filters_match_snapshot() {
            self.dream_air_msg = Some((
                "Image filters changed after recording. Discard this capture and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some(("Finish the guided capture before auditing.".into(), WARN));
            return;
        };
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let inputs = GeometryFitInputs {
            model_path,
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            baseline,
            mirrors,
            despeckle,
            flatten,
        };
        match self.geometry_fitter.start_audit(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Objective audit started. It compares the current scorer with the method that found the XR5 preset; live geometry will not change."
                        .into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Geometry audit could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn start_photometric_fit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::Photometric) {
            self.dream_air_msg = Some(("Record a Photometric Fit sequence first.".into(), WARN));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(geometry) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture geometry snapshot is missing.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some(("Capture filter snapshot is missing.".into(), ERR));
            return;
        };
        let Some(baseline) = self.photometric_capture_baseline else {
            self.dream_air_msg = Some(("Capture photometric baseline is missing.".into(), ERR));
            return;
        };
        if *self.pipeline.geometry.lock().unwrap() != geometry
            || !self.geometry_filters_match_snapshot()
            || *self.pipeline.photometric_correction.lock().unwrap() != baseline
        {
            self.dream_air_msg = Some((
                "The frozen image path changed after recording. Discard this evidence and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some(("Finish the guided capture before fitting.".into(), WARN));
            return;
        };
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let inputs = PhotometricFitInputs {
            model_path,
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            geometry,
            mirrors,
            despeckle,
            flatten,
            baseline,
        };
        match self.photometric_fitter.start(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Photometric search started. Geometry remains fixed and tracking stays live."
                        .into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Photometric Fit could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn start_endpoint_fit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::EyelidEndpoints) {
            self.dream_air_msg = Some((
                "Record an Open / closed endpoints session first.".into(),
                WARN,
            ));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(geometry) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture geometry snapshot is missing.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some(("Capture filter snapshot is missing.".into(), ERR));
            return;
        };
        let Some(photometric) = self.photometric_capture_baseline else {
            self.dream_air_msg = Some(("Capture photometric snapshot is missing.".into(), ERR));
            return;
        };
        let Some(current) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some((
                "Live eyelid calibration is not available yet; keep tracking active for a moment."
                    .into(),
                WARN,
            ));
            return;
        };
        if *self.pipeline.geometry.lock().unwrap() != geometry
            || !self.geometry_filters_match_snapshot()
            || *self.pipeline.photometric_correction.lock().unwrap() != photometric
        {
            self.dream_air_msg = Some((
                "The frozen image path changed after recording. Discard this evidence and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some((
                "Finish the endpoint recording before analysis.".into(),
                WARN,
            ));
            return;
        };
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let open_deadzone = self.pipeline.tuning.lock().unwrap().open_deadzone;
        let inputs = EndpointFitInputs {
            model_path,
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            geometry,
            mirrors,
            despeckle,
            flatten,
            photometric,
            current,
            open_deadzone,
        };
        match self.endpoint_fitter.start(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Endpoint analysis started. Only labelled endpoint phases are replayed; holdout stays decision-only."
                        .into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Endpoint analysis could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn apply_endpoint_result(&mut self, result: &crate::endpoint_fit::EndpointFitResult) {
        let eyes = result.accepted_eyes();
        if !eyes.iter().any(|accepted| *accepted) {
            self.dream_air_msg = Some((
                "Neither eye passed untouched holdout validation; nothing was applied.".into(),
                WARN,
            ));
            return;
        }
        if let Err(error) = crate::config::create_state_backup("before-eyelid-endpoints") {
            self.dream_air_msg =
                Some((format!("Could not create calibration backup: {error}"), ERR));
            return;
        }
        let session_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let permit = CommitPermit::<EndpointChange>::new(session_id);
        let Some(request) = endpoint_commit_request(result, eyes, permit) else {
            self.dream_air_msg =
                Some(("No validated endpoint candidate was available.".into(), ERR));
            return;
        };
        *self.pipeline.endpoint_apply.lock().unwrap() = Some(request);
        let applied = match eyes {
            [true, true] => "left and right eyes",
            [true, false] => "left eye only",
            [false, true] => "right eye only",
            [false, false] => unreachable!(),
        };
        self.dream_air_msg = Some((
            format!(
                "Validated endpoints queued for {applied}. Background close-endpoint learning is locked; Recenter still updates relaxed-open position."
            ),
            OK,
        ));
    }

    fn start_wink_fit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::Winks) {
            self.dream_air_msg = Some(("Record a Left / right wink session first.".into(), WARN));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(geometry) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture geometry snapshot is missing.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some(("Capture filter snapshot is missing.".into(), ERR));
            return;
        };
        let Some(photometric) = self.photometric_capture_baseline else {
            self.dream_air_msg = Some(("Capture photometric snapshot is missing.".into(), ERR));
            return;
        };
        let Some(current_endpoints) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some((
                "Live eyelid endpoints are not available. Keep tracking active for a moment."
                    .into(),
                WARN,
            ));
            return;
        };
        if !current_endpoints.left.endpoint_locked || !current_endpoints.right.endpoint_locked {
            self.dream_air_msg = Some((
                "The explicit endpoint calibration changed or is no longer active. Apply endpoints and record winks again."
                    .into(),
                WARN,
            ));
            return;
        }
        if *self.pipeline.geometry.lock().unwrap() != geometry
            || !self.geometry_filters_match_snapshot()
            || *self.pipeline.photometric_correction.lock().unwrap() != photometric
        {
            self.dream_air_msg = Some((
                "The frozen image path changed after recording. Discard this evidence and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some(("Finish the wink recording before analysis.".into(), WARN));
            return;
        };
        let inputs = WinkFitInputs {
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            geometry,
            mirrors: [
                self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
                self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
            ],
            despeckle,
            flatten,
            photometric,
            current_endpoints,
            open_deadzone: self.pipeline.tuning.lock().unwrap().open_deadzone,
        };
        match self.wink_fitter.start(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Wink analysis started. Each eye must pass its own untouched holdout.".into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Wink analysis could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn apply_wink_result(&mut self, result: &crate::wink_fit::FitResult) {
        let eyes = result.accepted_eyes();
        if !eyes.iter().any(|accepted| *accepted) {
            self.dream_air_msg = Some((
                "Neither eye passed the held-wink holdout; nothing was applied.".into(),
                WARN,
            ));
            return;
        }
        if let Err(error) = crate::config::create_state_backup("before-wink-response") {
            self.dream_air_msg =
                Some((format!("Could not create calibration backup: {error}"), ERR));
            return;
        }
        let session_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let permit = CommitPermit::<WinkChange>::new(session_id);
        let current = self.config.wink_profile_for(&self.pipeline.device_key);
        let Some(request) = wink_commit_request(result, eyes, current, permit) else {
            self.dream_air_msg = Some(("No validated wink response was available.".into(), ERR));
            return;
        };
        self.config
            .set_wink_profile(&self.pipeline.device_key, request.profile);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.dream_air_msg = Some((format!("Could not save the wink response: {error}"), ERR));
            return;
        }
        *self.pipeline.wink_apply.lock().unwrap() = Some(request);
        let applied = match eyes {
            [true, true] => "left and right eyes",
            [true, false] => "left eye only",
            [false, true] => "right eye only",
            [false, false] => unreachable!(),
        };
        self.dream_air_msg = Some((
            format!(
                "Validated wink response applied to {applied}. The partner eye and ordinary bilateral close endpoint are unchanged."
            ),
            OK,
        ));
    }

    fn calibration_result_change_busy(&self) -> bool {
        self.pending_unified_review()
            || self.geometry_evidence_locked()
            || self.gaze_residual_capture.is_running()
            || self.gaze_eyelid_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self
                .pipeline
                .endpoint_apply
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_some()
    }

    fn explicit_endpoint_locks(&self) -> [bool; 2] {
        self.tele
            .calibration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .map(|store| [store.left.endpoint_locked, store.right.endpoint_locked])
            .unwrap_or([false; 2])
    }

    /// Explain the exact per-HMD change before showing the destructive confirmation.
    /// Only downstream results which actually exist are named.
    fn calibration_removal_preview(&self, kind: SessionKind) -> Option<String> {
        let device = &self.pipeline.device_key;
        let endpoints = self.explicit_endpoint_locks();
        let gaze = self.config.gaze_eyelid_profile_for(device);
        let gaze_active = gaze.calibrated_unix != 0 || gaze.eyes.iter().any(|eye| eye.enabled);
        let wink = self.config.wink_profile_for(device);
        let wink_active = wink.calibrated_unix != 0 || wink.eyes.iter().any(|eye| eye.enabled);
        let blink = self.config.blink_timing_profile_for(device);
        let blink_fitted = blink.calibrated_unix != 0;

        let (direct_exists, first, cascades) = match kind {
            SessionKind::SafeGeometry => (
                self.config.has_geometry_override(device),
                "Restore the built-in XR5 image alignment.",
                true,
            ),
            SessionKind::Photometric => (
                self.config.has_photometric_correction(device),
                "Restore the built-in brightness and illumination path.",
                true,
            ),
            SessionKind::EyelidEndpoints => (
                endpoints.iter().any(|locked| *locked),
                "Remove the fitted 100% / 0% range and restart adaptive endpoint learning from standard bounds.",
                true,
            ),
            SessionKind::GazeDirections => (
                gaze_active,
                "Remove the saved looking-around eyelid correction.",
                false,
            ),
            SessionKind::Winks => (
                wink_active,
                "Remove the saved left / right wink response.",
                false,
            ),
            SessionKind::NaturalBlinks => (
                blink_fitted,
                "Remove the fitted blink timing and restore the built-in 42 ms timing. The separate enabled / disabled choice is preserved.",
                false,
            ),
        };
        if !direct_exists {
            return None;
        }

        let mut dependent = Vec::new();
        if cascades {
            if kind != SessionKind::EyelidEndpoints && endpoints.iter().any(|locked| *locked) {
                dependent.push("open / closed range");
            }
            if gaze_active {
                dependent.push("looking-around eyelids");
            }
            if wink_active {
                dependent.push("left / right wink");
            }
            if blink_fitted {
                dependent.push("fitted blink timing");
            }
        }
        Some(if dependent.is_empty() {
            format!("{first} This affects only the current HMD.")
        } else {
            format!(
                "{first} Because those results were fitted against it, this also resets: {}. This affects only the current HMD.",
                dependent.join(", ")
            )
        })
    }

    fn calibration_removal_controls(
        &mut self,
        ui: &mut egui::Ui,
        kind: SessionKind,
        button_text: &str,
    ) {
        let Some(preview) = self.calibration_removal_preview(kind) else {
            if self.pending_calibration_removal == Some(kind) {
                self.pending_calibration_removal = None;
            }
            return;
        };
        let busy = self.calibration_result_change_busy();
        if self.pending_calibration_removal != Some(kind) {
            let response = ui.add_enabled(!busy, egui::Button::new(button_text));
            if busy {
                response.on_hover_text(
                    "Finish, apply, or discard the active calibration recording / analysis first.",
                );
            } else if response.clicked() {
                self.pending_calibration_removal = Some(kind);
            }
            return;
        }

        ui.add_space(SP2);
        egui::Frame::none()
            .fill(Color32::from_rgb(35, 22, 20))
            .stroke(Stroke::new(1.0, WARN))
            .inner_margin(egui::Margin::same(10.0))
            .rounding(6.0)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new("CONFIRM REMOVAL")
                        .monospace()
                        .strong()
                        .color(WARN),
                );
                ui.label(label(&preview));
                ui.label(num(
                    "A safety copy is saved under SRanibro\\backups. This window has no one-click Undo.",
                ));
                let mut confirm = false;
                let mut cancel = false;
                ui.horizontal(|ui| {
                    confirm = ui
                        .add_enabled(!busy, egui::Button::new("Remove calibration"))
                        .clicked();
                    cancel = ui.button("Cancel").clicked();
                });
                if confirm {
                    self.pending_calibration_removal = None;
                    self.remove_calibration_result(kind);
                } else if cancel {
                    self.pending_calibration_removal = None;
                }
            });
    }

    /// Remove one persisted result and every result whose coordinate system it
    /// invalidates. UI capture/fit starts and this method run on the same egui
    /// thread, so the busy check and state transition cannot race each other.
    fn remove_calibration_result(&mut self, kind: SessionKind) {
        if self.calibration_result_change_busy() {
            self.dream_air_msg = Some((
                "Finish, apply, or discard the active calibration recording / analysis first."
                    .into(),
                WARN,
            ));
            return;
        }
        if self.calibration_removal_preview(kind).is_none() {
            self.dream_air_msg = Some((
                "That calibration is no longer active for this HMD.".into(),
                WARN,
            ));
            return;
        }

        let device = self.pipeline.device_key.clone();
        let calib_path = crate::config::calib_path_for(&device)
            .to_string_lossy()
            .into_owned();
        let telemetry_store = *self
            .tele
            .calibration
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous_store =
            telemetry_store.or_else(|| crate::core::eye_state::load_calib(&calib_path));
        let mut next_store = previous_store;
        let mut next_config = self.config.clone();
        let mut changed = Vec::<&'static str>::new();
        let cascade = matches!(
            kind,
            SessionKind::SafeGeometry | SessionKind::Photometric | SessionKind::EyelidEndpoints
        );
        let reset_endpoints = matches!(kind, SessionKind::SafeGeometry | SessionKind::Photometric)
            || kind == SessionKind::EyelidEndpoints;

        match kind {
            SessionKind::SafeGeometry => {
                if next_config.clear_geometry(&device) {
                    changed.push("image alignment");
                }
            }
            SessionKind::Photometric => {
                if next_config.clear_photometric_correction(&device) {
                    changed.push("lighting correction");
                }
            }
            SessionKind::EyelidEndpoints => {}
            SessionKind::GazeDirections => {
                if next_config.clear_gaze_eyelid_profile(&device) {
                    changed.push("looking-around eyelids");
                }
            }
            SessionKind::Winks => {
                if next_config.clear_wink_profile(&device) {
                    changed.push("left / right wink");
                }
            }
            SessionKind::NaturalBlinks => {
                next_config.clear_blink_timing_calibration(&device);
                changed.push("fitted blink timing");
            }
        }

        let mut reset_eyes = [false; 2];
        if reset_endpoints {
            if let Some(store) = next_store.as_mut() {
                reset_eyes = [store.left.endpoint_locked, store.right.endpoint_locked];
                if reset_eyes[0] {
                    store.left = store.left.without_explicit_endpoint();
                }
                if reset_eyes[1] {
                    store.right = store.right.without_explicit_endpoint();
                }
                if reset_eyes.iter().any(|reset| *reset) {
                    changed.push("open / closed range");
                }
            }
        }
        if cascade {
            let gaze = next_config.gaze_eyelid_profile_for(&device);
            if (gaze.calibrated_unix != 0 || gaze.eyes.iter().any(|eye| eye.enabled))
                && next_config.clear_gaze_eyelid_profile(&device)
            {
                changed.push("looking-around eyelids");
            }
            let wink = next_config.wink_profile_for(&device);
            if (wink.calibrated_unix != 0 || wink.eyes.iter().any(|eye| eye.enabled))
                && next_config.clear_wink_profile(&device)
            {
                changed.push("left / right wink");
            }
            let blink = next_config.blink_timing_profile_for(&device);
            if blink.calibrated_unix != 0 {
                next_config.clear_blink_timing_calibration(&device);
                changed.push("fitted blink timing");
            }
        }

        let backup = match crate::config::create_state_backup("before-remove-calibration") {
            Ok(path) => path,
            Err(error) => {
                self.dream_air_msg = Some((
                    format!("Backup failed; calibration was not removed: {error}"),
                    ERR,
                ));
                return;
            }
        };
        if reset_eyes.iter().any(|reset| *reset) {
            let Some(store) = next_store else {
                self.dream_air_msg = Some((
                    "Live endpoint state was unavailable; nothing was removed.".into(),
                    ERR,
                ));
                return;
            };
            if let Err(error) = crate::core::eye_state::save_calib_checked(&calib_path, &store) {
                self.dream_air_msg = Some((
                    format!("Endpoint reset could not be saved; nothing was removed: {error}"),
                    ERR,
                ));
                return;
            }
        }
        if let Err(error) = next_config.save(&crate::config::config_path()) {
            let rollback = if reset_eyes.iter().any(|reset| *reset) {
                previous_store
                    .ok_or_else(|| "previous endpoint state was unavailable".to_owned())
                    .and_then(|store| {
                        crate::core::eye_state::save_calib_checked(&calib_path, &store)
                            .map_err(|rollback| rollback.to_string())
                    })
            } else {
                Ok(())
            };
            self.dream_air_msg = Some((
                match rollback {
                    Ok(()) => format!("Config save failed; removal was rolled back: {error}"),
                    Err(rollback) => format!(
                        "Config save failed and endpoint rollback failed ({rollback}). Restore the safety backup at {}: {error}",
                        backup.display()
                    ),
                },
                ERR,
            ));
            return;
        }

        self.config = next_config;
        if kind == SessionKind::SafeGeometry {
            self.set_live_geometry(self.config.geometry_for(&device));
            self.geometry_preview_restore = None;
            self.geometry_rollback = None;
            self.geometry_fitter.clear_finished();
        }
        if kind == SessionKind::Photometric {
            *self
                .pipeline
                .photometric_correction
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                self.config.photometric_correction_for(&device);
            self.photometric_rollback = None;
            self.photometric_fitter.clear_finished();
        }
        if reset_eyes.iter().any(|reset| *reset) {
            *self
                .pipeline
                .endpoint_apply
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                Some(crate::endpoint_fit::EndpointApplyRequest {
                    eyes: [None, None],
                    reset_to_adaptive: reset_eyes,
                });
            self.endpoint_fitter.clear_finished();
        }
        let gaze_profile = self.config.gaze_eyelid_profile_for(&device);
        *self
            .pipeline
            .gaze_eyelid_apply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(crate::gaze_eyelid_fit::ApplyRequest {
                profile: gaze_profile,
            });
        let wink_profile = self.config.wink_profile_for(&device);
        *self
            .pipeline
            .wink_apply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(crate::wink_fit::ApplyRequest {
                profile: wink_profile,
            });
        let blink_profile = self.config.blink_timing_profile_for(&device);
        *self
            .pipeline
            .blink_timing_apply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(crate::blink_timing_fit::ApplyRequest {
                profile: blink_profile,
            });
        if cascade || kind == SessionKind::GazeDirections {
            self.gaze_eyelid_fitter.clear_finished();
        }
        if cascade || kind == SessionKind::Winks {
            self.wink_fitter.clear_finished();
        }
        if cascade || kind == SessionKind::NaturalBlinks {
            self.blink_timing_fitter.clear_finished();
        }

        self.dream_air_msg = Some((
            format!(
                "Reverted for this HMD: {}. Safety backup: {}",
                changed.join(", "),
                backup.display()
            ),
            OK,
        ));
    }

    fn start_blink_timing_fit(&mut self) {
        if self.calibration_capture_purpose != Some(CalibrationCapturePurpose::NaturalBlinks) {
            self.dream_air_msg =
                Some(("Record a Natural blink timing session first.".into(), WARN));
            return;
        }
        let Some(model_path) = self.config.ml_params_path().filter(|path| path.is_file()) else {
            self.dream_air_msg = Some(("EyePrediction model path is missing.".into(), ERR));
            return;
        };
        let Some(geometry) = self.geometry_capture_baseline else {
            self.dream_air_msg = Some(("Capture geometry snapshot is missing.".into(), ERR));
            return;
        };
        let Some((despeckle, flatten)) = self.geometry_capture_filters else {
            self.dream_air_msg = Some(("Capture filter snapshot is missing.".into(), ERR));
            return;
        };
        let Some(photometric) = self.photometric_capture_baseline else {
            self.dream_air_msg = Some(("Capture photometric snapshot is missing.".into(), ERR));
            return;
        };
        let Some(current_endpoints) = *self.tele.calibration.lock().unwrap() else {
            self.dream_air_msg = Some(("Live eyelid endpoints are unavailable.".into(), WARN));
            return;
        };
        if !current_endpoints.left.endpoint_locked || !current_endpoints.right.endpoint_locked {
            self.dream_air_msg = Some((
                "Explicit endpoints are no longer active. Apply endpoints and record natural blinks again."
                    .into(),
                WARN,
            ));
            return;
        }
        if *self.pipeline.geometry.lock().unwrap() != geometry
            || !self.geometry_filters_match_snapshot()
            || *self.pipeline.photometric_correction.lock().unwrap() != photometric
        {
            self.dream_air_msg = Some((
                "The frozen image path changed after recording. Discard this evidence and record again."
                    .into(),
                WARN,
            ));
            return;
        }
        let (model_bytes, expected_model_crc32, expected_model_bytes) =
            match self.geometry_model_snapshot(&model_path) {
                Ok(snapshot) => snapshot,
                Err(error) => {
                    self.dream_air_msg = Some((error, ERR));
                    return;
                }
            };
        if !self.ensure_geometry_recording_saved() {
            return;
        }
        let Some(dataset) = self.geometry_capture.take_dataset() else {
            self.dream_air_msg = Some(("Finish the natural-blink recording first.".into(), WARN));
            return;
        };
        let inputs = BlinkTimingFitInputs {
            model_bytes,
            expected_model_crc32,
            expected_model_bytes,
            dataset: dataset.into(),
            geometry,
            mirrors: [
                self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
                self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
            ],
            despeckle,
            flatten,
            photometric,
            current_endpoints,
            open_deadzone: self.pipeline.tuning.lock().unwrap().open_deadzone,
        };
        match self.blink_timing_fitter.start(inputs) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    "Natural-blink analysis started. Untouched holdout must repeat the bilateral fast-blink evidence."
                        .into(),
                    ACCENT,
                ));
            }
            Err(error) => {
                self.geometry_capture
                    .restore_completed_dataset(error.inputs.dataset.into_dataset());
                self.dream_air_msg = Some((
                    format!("Natural-blink analysis could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn apply_blink_timing_result(&mut self, result: &crate::blink_timing_fit::FitResult) {
        if !result.accepted {
            self.dream_air_msg = Some((
                "The natural-blink holdout did not pass; timing was not changed.".into(),
                WARN,
            ));
            return;
        }
        if let Err(error) = crate::config::create_state_backup("before-natural-blink-timing") {
            self.dream_air_msg =
                Some((format!("Could not create calibration backup: {error}"), ERR));
            return;
        }
        let session_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let permit = CommitPermit::<BlinkTimingChange>::new(session_id);
        let Some(request) = blink_timing_commit_request(result, permit) else {
            self.dream_air_msg = Some((
                "No validated natural-blink timing was available.".into(),
                ERR,
            ));
            return;
        };
        self.config
            .set_blink_timing_profile(&self.pipeline.device_key, request.profile);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.dream_air_msg =
                Some((format!("Could not save natural-blink timing: {error}"), ERR));
            return;
        }
        *self.pipeline.blink_timing_apply.lock().unwrap() = Some(request);
        self.dream_air_msg = Some((
            format!(
                "Natural-blink bottom guarantee applied: {:.0} ms after both eyes reach zero. Slow closes and one-eye winks are unchanged.",
                request.profile.min_closed_ms
            ),
            OK,
        ));
    }

    /// The timing feature's master switch is deliberately separate from removing
    /// fitted timing. Disabling it preserves the fitted milliseconds so enabling
    /// it again is reversible; removing calibration preserves this switch.
    fn set_blink_timing_enabled(&mut self, enabled: bool) {
        let device = self.pipeline.device_key.clone();
        let mut profile = self.config.blink_timing_profile_for(&device);
        if profile.enabled == enabled {
            return;
        }
        if let Err(error) = crate::config::create_state_backup("before-blink-timing-toggle") {
            self.dream_air_msg =
                Some((format!("Could not create calibration backup: {error}"), ERR));
            return;
        }
        let previous = profile;
        profile.enabled = enabled;
        self.config.set_blink_timing_profile(&device, profile);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.config.set_blink_timing_profile(&device, previous);
            self.dream_air_msg = Some((
                format!("Could not save the blink timing switch: {error}"),
                ERR,
            ));
            return;
        }
        *self
            .pipeline
            .blink_timing_apply
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) =
            Some(crate::blink_timing_fit::ApplyRequest { profile });
        self.dream_air_msg = Some((
            format!(
                "Natural-blink visible-bottom timing {} for this HMD.",
                if enabled { "enabled" } else { "disabled" }
            ),
            OK,
        ));
    }

    fn apply_gaze_eyelid_result(&mut self, result: &crate::gaze_eyelid_fit::FitResult) {
        let eyes = result.accepted_eyes();
        if !eyes.iter().any(|accepted| *accepted) {
            self.dream_air_msg = Some((
                "Neither eye showed a repeatable, safely correctable directional droop.".into(),
                WARN,
            ));
            return;
        }
        if let Err(error) = crate::config::create_state_backup("before-gaze-eyelid") {
            self.dream_air_msg =
                Some((format!("Could not create calibration backup: {error}"), ERR));
            return;
        }
        let session_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let permit = CommitPermit::<GazeEyelidChange>::new(session_id);
        let current = self
            .config
            .gaze_eyelid_profile_for(&self.pipeline.device_key);
        let Some(request) = gaze_eyelid_commit_request(result, eyes, current, permit) else {
            self.dream_air_msg = Some(("No validated correction was available.".into(), ERR));
            return;
        };
        self.config
            .set_gaze_eyelid_profile(&self.pipeline.device_key, request.profile);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.dream_air_msg = Some((
                format!("Could not save the gaze-direction eyelid profile: {error}"),
                ERR,
            ));
            return;
        }
        *self.pipeline.gaze_eyelid_apply.lock().unwrap() = Some(request);
        let applied = match eyes {
            [true, true] => "left and right eyes",
            [true, false] => "left eye only",
            [false, true] => "right eye only",
            [false, false] => unreachable!(),
        };
        self.dream_air_msg = Some((
            format!(
                "Validated gaze-dependent eyelid correction applied to {applied}. Native/avatar gaze and EyeWide settings were not changed."
            ),
            OK,
        ));
    }

    fn apply_photometric_candidate(&mut self, result: &PhotometricFitResult) {
        if !result.accepted {
            self.dream_air_msg = Some((
                "This candidate did not pass untouched holdout validation and cannot be applied."
                    .into(),
                WARN,
            ));
            return;
        }
        if !self.geometry_filters_match_snapshot()
            || self
                .geometry_capture_baseline
                .is_none_or(|geometry| *self.pipeline.geometry.lock().unwrap() != geometry)
        {
            self.dream_air_msg = Some((
                "Geometry or upstream filters changed after scoring. Record again before applying."
                    .into(),
                WARN,
            ));
            return;
        }
        let live = *self.pipeline.photometric_correction.lock().unwrap();
        if live != result.baseline && live != result.candidate {
            self.dream_air_msg = Some((
                "The fitted correction changed after capture. Record again before applying.".into(),
                WARN,
            ));
            return;
        }
        let device = self.pipeline.device_key.clone();
        if self.config.photometric_correction_for(&device) == result.candidate {
            self.dream_air_msg =
                Some(("This photometric correction is already applied.".into(), OK));
            return;
        }
        let backup = match crate::config::create_state_backup("before-photometric-fit") {
            Ok(path) => path,
            Err(error) => {
                self.dream_air_msg = Some((
                    format!("Backup failed; candidate was not applied: {error}"),
                    ERR,
                ));
                return;
            }
        };
        self.config
            .set_photometric_correction(&device, result.candidate);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.config.set_photometric_correction(&device, live);
            self.dream_air_msg = Some((
                format!("Config save failed; current correction was kept: {error}"),
                ERR,
            ));
            return;
        }
        *self.pipeline.photometric_correction.lock().unwrap() = result.candidate;
        self.photometric_rollback = Some(live);
        self.dream_air_msg = Some((
            format!(
                "Validated photometric correction applied. Safety backup: {}",
                backup.display()
            ),
            OK,
        ));
    }

    fn rollback_photometric(&mut self) {
        if self.geometry_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
        {
            self.dream_air_msg = Some((
                "Cancel the active capture or fit before rolling the correction back.".into(),
                WARN,
            ));
            return;
        }
        let Some(previous) = self.photometric_rollback.take() else {
            return;
        };
        let device = self.pipeline.device_key.clone();
        self.config.set_photometric_correction(&device, previous);
        match self.config.save(&crate::config::config_path()) {
            Ok(()) => {
                *self.pipeline.photometric_correction.lock().unwrap() = previous;
                self.dream_air_msg = Some((
                    "Previous fitted photometric correction restored.".into(),
                    OK,
                ));
            }
            Err(error) => {
                self.photometric_rollback = Some(previous);
                self.dream_air_msg = Some((format!("Rollback save failed: {error}"), ERR));
            }
        }
    }

    fn set_live_geometry(&mut self, geometry: [crate::core::types::MlGeometry; 2]) {
        *self
            .pipeline
            .geometry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = geometry;
        if let Some(mirror) = geometry[0].mirror_h {
            self.pipeline.ml_mirror_l.store(mirror, Ordering::Relaxed);
        }
        if let Some(mirror) = geometry[1].mirror_h {
            self.pipeline.ml_mirror_r.store(mirror, Ordering::Relaxed);
        }
    }

    fn preview_geometry_candidate(&mut self, result: &GeometryFitResult) {
        if self.gaze_residual_capture.is_running() {
            self.dream_air_msg = Some((
                "Image geometry is locked during landmark/residual recording.".into(),
                WARN,
            ));
            return;
        }
        if !self.geometry_filters_match_snapshot() {
            self.dream_air_msg = Some((
                "Image filters changed since this result was scored. Record again before previewing it."
                    .into(),
                WARN,
            ));
            return;
        }
        let live = *self.pipeline.geometry.lock().unwrap();
        if live != result.baseline && live != result.candidate {
            self.dream_air_msg = Some((
                "Image geometry changed after capture. Record again before previewing this result."
                    .into(),
                WARN,
            ));
            return;
        }
        if self.geometry_preview_restore.is_none() {
            self.geometry_preview_restore = Some(live);
        }
        self.set_live_geometry(result.candidate);
        self.dream_air_msg = Some((
            "Candidate preview is live but not saved. Blink and slowly close once to compare."
                .into(),
            ACCENT,
        ));
    }

    fn restore_geometry_preview(&mut self, show_message: bool) {
        let Some(previous) = self.geometry_preview_restore.take() else {
            return;
        };
        self.set_live_geometry(previous);
        if show_message {
            self.dream_air_msg = Some(("Unsaved preview discarded.".into(), WARN));
        }
    }

    fn apply_geometry_candidate(&mut self, result: &GeometryFitResult, allow_unvalidated: bool) {
        if self.gaze_residual_capture.is_running() {
            self.dream_air_msg = Some((
                "Image geometry is locked during landmark/residual recording.".into(),
                WARN,
            ));
            return;
        }
        if !self.geometry_filters_match_snapshot() {
            self.dream_air_msg = Some((
                "Image filters changed since this result was scored. Record again before applying it."
                    .into(),
                WARN,
            ));
            return;
        }
        if !result.accepted && !allow_unvalidated {
            self.dream_air_msg = Some((
                "Confirm the holdout warning before applying this candidate.".into(),
                ERR,
            ));
            return;
        }
        let device = self.pipeline.device_key.clone();
        if self.config.geometry_for(&device) == result.candidate {
            self.dream_air_msg = Some(("This geometry is already applied.".into(), OK));
            return;
        }
        let live = *self.pipeline.geometry.lock().unwrap();
        if live != result.baseline && live != result.candidate {
            self.dream_air_msg = Some((
                "Image geometry changed after capture. Record again before applying this result."
                    .into(),
                WARN,
            ));
            return;
        }
        let previous = self.geometry_preview_restore.take().unwrap_or(live);
        let backup = match crate::config::create_state_backup("before-xr5-geometry-fit") {
            Ok(path) => path,
            Err(error) => {
                self.set_live_geometry(previous);
                self.dream_air_msg = Some((
                    format!("Backup failed; candidate was not applied: {error}"),
                    ERR,
                ));
                return;
            }
        };
        self.config.set_geometry(&device, result.candidate);
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            self.config.set_geometry(&device, previous);
            self.set_live_geometry(previous);
            self.dream_air_msg = Some((
                format!("Config save failed; candidate was rolled back: {error}"),
                ERR,
            ));
            return;
        }
        self.set_live_geometry(result.candidate);
        self.geometry_rollback = Some(previous);
        self.geometry_unvalidated_ack = false;
        self.dream_air_msg = Some((
            format!(
                "{} candidate applied. Safety backup: {}",
                if result.accepted {
                    "Validated"
                } else {
                    "Unvalidated"
                },
                backup.display()
            ),
            if result.accepted { OK } else { WARN },
        ));
    }

    fn rollback_geometry(&mut self) {
        if self.geometry_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.gaze_residual_capture.is_running()
        {
            self.dream_air_msg = Some((
                "Cancel the active capture or fit before rolling geometry back.".into(),
                WARN,
            ));
            return;
        }
        let Some(previous) = self.geometry_rollback.take() else {
            return;
        };
        let device = self.pipeline.device_key.clone();
        self.config.set_geometry(&device, previous);
        match self.config.save(&crate::config::config_path()) {
            Ok(()) => {
                self.set_live_geometry(previous);
                self.geometry_preview_restore = None;
                self.dream_air_msg = Some(("Previous image geometry restored.".into(), OK));
            }
            Err(error) => {
                self.geometry_rollback = Some(previous);
                self.dream_air_msg = Some((format!("Rollback save failed: {error}"), ERR));
            }
        }
    }

    /// Drain the coherent ML-source history. UI repaint rate is presentation-only:
    /// a 24 Hz desktop can still consume every eligible camera-clocked Wide frame.
    fn update_wide_capture(&mut self) {
        if self.pipeline.device_key != "pimax_xr5" {
            return;
        }
        if !self.wide.is_running() {
            self.wide_last_frames = self.tele.frame_generations();
            return;
        }
        let history = self.tele.calibration_frames_after(self.wide_last_frames);
        for sample in history {
            self.wide.tick_at(sample.captured_at);
            self.wide_last_frames = sample.source_generation;
            if !self.wide.is_running() {
                continue;
            }
            let frames = sample.stereo_frames();
            let left = frames[0].as_ref().map(EyeFrame::view);
            let right = frames[1].as_ref().map(EyeFrame::view);
            self.wide.on_frame_at(sample.captured_at, left, right);
        }
        self.wide.tick_at(Instant::now());
    }

    /// Eyebrow datasets are count-driven, but their source is still the bounded
    /// camera history so collection continues at the device rate on low-Hz desktops.
    fn update_brow_capture(&mut self) {
        if !self.brow.is_running() {
            self.brow_last_frames = self.tele.frame_generations();
            return;
        }
        let history = self.tele.calibration_frames_after(self.brow_last_frames);
        for sample in history {
            self.brow.tick_at(sample.captured_at);
            self.brow_last_frames = sample.source_generation;
            if !self.brow.is_running() {
                continue;
            }
            let frames = sample.stereo_frames();
            let left = frames[0].as_ref().map(EyeFrame::view);
            let right = frames[1].as_ref().map(EyeFrame::view);
            self.brow.on_frame_at(sample.captured_at, left, right);
        }
        self.brow.tick_at(Instant::now());
    }

    fn hot_load_wide_model(&mut self, wide_bin: &std::path::Path) {
        match crate::ml::wide_net::WideNet::load(wide_bin) {
            Ok(net) => {
                let active_xr5 = self.pipeline.device_key == "pimax_xr5";
                if active_xr5 {
                    self.pipeline.set_wide(Some(net));
                }
                let path = wide_bin.to_string_lossy().into_owned();
                self.edit.wide_model = path.clone();
                self.config.assets.wide_model = Some(path);
                match self.config.save(&crate::config::config_path()) {
                    Ok(()) => {
                        let message = if active_xr5 {
                            "Wide model fitted and loaded for A/B comparison; choose Auto or Custom to output it"
                        } else {
                            "Wide model fitted and saved, but not loaded because the active HMD is not XR5; switch back to XR5 and Apply & reload"
                        };
                        self.dream_air_msg = Some((message.into(), OK));
                    }
                    Err(error) => {
                        let action = if active_xr5 {
                            "loaded"
                        } else {
                            "validated but not loaded on this non-XR5 HMD"
                        };
                        self.dream_air_msg = Some((
                            format!("Wide model {action}, but config save failed: {error}"),
                            ERR,
                        ));
                    }
                }
            }
            Err(error) => {
                self.dream_air_msg = Some((
                    format!("Fitted Wide model could not be loaded: {error}"),
                    ERR,
                ));
            }
        }
    }

    fn apply_wide_fit_result_if_ready(&mut self) {
        if self.wide_fit_applied {
            return;
        }
        let WideFitStatus::Done { wide_bin, .. } = self.wide_fitter.status() else {
            return;
        };
        self.wide_fit_applied = true;
        self.hot_load_wide_model(&wide_bin);
    }

    #[cfg(any())]
    fn start_guided_calibration(&mut self) {
        self.guided_calibration = Some(GuidedCalibration::new());
        self.guided_report = None;
        self.guided_last_ml = self.tele.c_ml.load(Ordering::Relaxed);
        self.dream_air_msg = Some((
            "Follow each gesture and hold it until the next instruction".into(),
            ACCENT,
        ));
    }

    #[cfg(any())]
    fn apply_guided_calibration(&mut self, report: CalibrationReport) {
        if report.mapping == MappingVerdict::Swapped || !report.passed {
            self.dream_air_msg = Some((
                "Calibration did not pass; fix the failed checks and rerun".into(),
                ERR,
            ));
            return;
        }
        let backup = match crate::config::create_state_backup("before-dream-air-calibration") {
            Ok(path) => path,
            Err(e) => {
                self.dream_air_msg = Some((
                    format!("Backup failed; calibration was not applied: {e}"),
                    ERR,
                ));
                return;
            }
        };
        let previous = self.tele.calibration.lock().ok().and_then(|guard| *guard);
        let store = report.calibration_store(previous);
        let calibrated_unix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let profile = DreamAirProfile {
            schema_version: 1,
            eyechip_serial: self.eyechip_serial.clone(),
            calibrated_unix,
            baseline: report.baseline,
            blink_depth: report.blink_depth,
            wide_supported: report.wide_supported,
            wide_snr: report.wide_snr,
            quality_score: report.quality_score,
            pupil_center: report.pupil_center,
            pupil_center_valid: report.pupil_center_valid,
        };
        let device = self.pipeline.device_key.clone();
        self.config.set_dream_air_profile(&device, profile);
        match self.config.save(&crate::config::config_path()) {
            Ok(()) => {
                if let Ok(mut pending) = self.pipeline.guided_calibration.lock() {
                    *pending = Some(store);
                }
                if let Ok(mut enabled) = self.pipeline.wide_enabled.lock() {
                    *enabled = report.wide_supported;
                }
                self.guided_report = None;
                self.dream_air_msg = Some((
                    format!("Applied - backup saved at {}", backup.display()),
                    OK,
                ));
            }
            Err(e) => {
                self.dream_air_msg = Some((format!("Profile save failed: {e}"), ERR));
            }
        }
    }

    #[cfg(any())]
    fn export_dream_air_support_bundle(&mut self) {
        let preflight = self.current_preflight();
        let quality = self.current_quality();
        let geometry = *self.pipeline.geometry.lock().unwrap();
        let mapping = self.config.mapping_for(&self.pipeline.device_key);
        let correction = *self.pipeline.gaze_correction.lock().unwrap();
        let wide_enabled = *self.pipeline.wide_enabled.lock().unwrap();
        let unit_id = crate::diagnostics::pseudonymous_unit_id(self.eyechip_serial.as_deref());
        let mut summary = format!(
            "SRanibro {}\ndevice={}\nunit_id={}\nrates={:?}\nquality={:.1} {:?}\nquality_reasons={:?}\ngeometry={:?}\nmapping={:?}\ngaze_correction={:?}\nwide_enabled={:?}\nbaselines={:?}\nprofile_quality={:?}\n\nPREFLIGHT\n",
            env!("CARGO_PKG_VERSION"),
            self.pipeline.device_key,
            unit_id,
            self.rates,
            quality.score,
            quality.level,
            quality.reasons,
            geometry,
            mapping,
            correction,
            wide_enabled,
            *self.tele.baselines.lock().unwrap(),
            self.config
                .dream_air_profile_for(&self.pipeline.device_key)
                .map(|profile| (profile.schema_version, profile.calibrated_unix, profile.quality_score, profile.wide_snr)),
        );
        for check in preflight.checks {
            summary.push_str(&format!(
                "{}: {} - {}\n",
                check.name, check.passed, check.detail
            ));
        }
        let log_tail = self
            .log
            .lock()
            .map(|log| {
                let start = log.len().saturating_sub(400);
                log.iter()
                    .skip(start)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();
        let summary =
            crate::diagnostics::redact_support_text(&summary, self.eyechip_serial.as_deref());
        let log_tail =
            crate::diagnostics::redact_support_text(&log_tail, self.eyechip_serial.as_deref());
        match crate::diagnostics::export_support_bundle(&summary, &log_tail) {
            Ok(path) => {
                self.dream_air_msg = Some((format!("Support ZIP saved: {}", path.display()), OK));
            }
            Err(e) => {
                self.dream_air_msg = Some((format!("Support ZIP failed: {e}"), ERR));
            }
        }
    }

    fn title_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("bar")
            .frame(
                egui::Frame::default()
                    .fill(NAV_BG)
                    .inner_margin(egui::Margin::symmetric(14.0 * S, 6.0 * S)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    // Establish the control-row height before laying out the shorter
                    // wordmark labels. egui cannot move earlier widgets when a taller
                    // widget is added later, which left the text slightly top-aligned.
                    ui.allocate_space(vec2(0.0, 22.0 * S));
                    ui.label(
                        egui::RichText::new("SRani")
                            .monospace()
                            .size(13.0 * S)
                            .strong()
                            .color(TEXT1),
                    );
                    ui.label(
                        egui::RichText::new("bro")
                            .monospace()
                            .size(13.0 * S)
                            .strong()
                            .color(ACCENT),
                    );
                    ui.add_space(8.0 * S);
                    ui.label(
                        egui::RichText::new(APP_VERSION_LABEL)
                            .monospace()
                            .size(9.0 * S)
                            .color(TEXT3),
                    );

                    // Allocate only the middle blank area for native window dragging.
                    // The two control rectangles remain independent click targets.
                    let controls_width = 70.0 * S;
                    let drag_width = (ui.available_width() - controls_width).max(24.0 * S);
                    let drag_sense = if cfg!(windows) {
                        Sense::hover()
                    } else {
                        Sense::click_and_drag()
                    };
                    let drag = ui.allocate_response(vec2(drag_width, 22.0 * S), drag_sense);
                    set_native_caption_rect(drag.rect, ui.ctx().pixels_per_point());
                    #[cfg(not(windows))]
                    if drag.drag_started() {
                        self.begin_fallback_window_drag(ctx);
                    }

                    let control = |ui: &mut egui::Ui, danger: bool| {
                        let (rect, response) =
                            ui.allocate_exact_size(vec2(35.0 * S, 22.0 * S), Sense::click());
                        let fill = if response.hovered() {
                            if danger {
                                Color32::from_rgb(196, 43, 28)
                            } else {
                                SURFACE
                            }
                        } else {
                            NAV_BG
                        };
                        ui.painter().rect_filled(rect, 4.0 * S, fill);
                        // Font glyphs such as × and − sit above the typographic
                        // line-box centre, so CENTER_CENTER still looks too high.
                        // Draw the chrome as geometry to keep its visual centre
                        // exact across fonts, DPI scales and fallback renderers.
                        let centre = ui.painter().round_pos_to_pixels(rect.center());
                        let radius = 4.5 * S;
                        let stroke = Stroke::new(1.5 * S, TEXT1);
                        if danger {
                            ui.painter().line_segment(
                                [
                                    pos2(centre.x - radius, centre.y - radius),
                                    pos2(centre.x + radius, centre.y + radius),
                                ],
                                stroke,
                            );
                            ui.painter().line_segment(
                                [
                                    pos2(centre.x - radius, centre.y + radius),
                                    pos2(centre.x + radius, centre.y - radius),
                                ],
                                stroke,
                            );
                        } else {
                            ui.painter().hline(
                                centre.x - radius..=centre.x + radius,
                                centre.y,
                                stroke,
                            );
                        }
                        response.widget_info(|| {
                            egui::WidgetInfo::labeled(
                                egui::WidgetType::Button,
                                ui.is_enabled(),
                                if danger { "Close" } else { "Minimize" },
                            )
                        });
                        response
                    };

                    if control(ui, false).on_hover_text("Minimize").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                    }
                    if control(ui, true).on_hover_text("Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
            });
    }

    fn nav(&mut self, ctx: &egui::Context) {
        let live = self.rates[4] > 1.0 && !self.pipeline.paused.load(Ordering::Relaxed);
        let mut reload = false;
        egui::SidePanel::left("nav")
            .exact_width(NAV_W)
            .resizable(false)
            .frame(
                egui::Frame::default()
                    .fill(NAV_BG)
                    .inner_margin(egui::Margin::symmetric(9.0 * S, 10.0 * S)),
            )
            .show(ctx, |ui| {
                // Bottom "system live" dot: paint at the true client bottom
                // (egui's available_height over-reports the surface here).
                let cx = ui.max_rect().center().x;
                let bottom = content_h();
                let dot = pos2(cx, bottom - 13.0 * S);
                // Shape-coded (not color-only): filled green = live, hollow ring = idle
                // — the old LED_OFF fill was ~1.7:1 and read as "no dot at all".
                if live {
                    ui.painter().circle_filled(dot, 4.0 * S, OK);
                } else {
                    ui.painter()
                        .circle_stroke(dot, 4.0 * S, Stroke::new(1.5 * S, TEXT3));
                }

                // Keep Reload in its own fixed bottom slot, above the live-status light.
                // It does not participate in the vertical tab layout and therefore cannot
                // overlap a page icon or a collapsible-card click target.
                let reload_rect =
                    Rect::from_center_size(pos2(cx, bottom - 47.0 * S), vec2(30.0 * S, 30.0 * S));
                let reload_response =
                    ui.interact(reload_rect, Id::new("nav_reload"), Sense::click());
                let reload_fill = if reload_response.hovered() {
                    SURFACE
                } else {
                    NAV_BG
                };
                ui.painter().rect_filled(reload_rect, 8.0 * S, reload_fill);
                if reload_response.hovered() {
                    ui.painter()
                        .rect_stroke(reload_rect, 8.0 * S, Stroke::new(1.0, BORDER));
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                draw_icon(
                    ui.painter(),
                    Icon::Reload,
                    reload_rect.center(),
                    18.0 * S,
                    if reload_response.hovered() {
                        ACCENT
                    } else {
                        TEXT3
                    },
                );
                let reload_response = reload_response
                    .on_hover_text("Apply settings, reconnect the HMD, and reload models.");
                reload = reload_response.clicked();

                ui.vertical_centered(|ui| {
                    self.nav_icon(ui, Page::Dashboard, Icon::Activity, 0);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Calibration, Icon::Sliders, 1);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Console, Icon::Console, 2);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Settings, Icon::Gear, 3);
                });
            });
        if reload {
            self.apply_and_reload();
        }
    }

    fn nav_icon(&mut self, ui: &mut egui::Ui, page: Page, icon: Icon, idx: u32) {
        let selected = self.page == page;
        let (rect, resp) = ui.allocate_exact_size(vec2(30.0 * S, 30.0 * S), Sense::click());
        let t = ui.ctx().animate_bool(Id::new(("nav", idx)), selected);
        let hover = if resp.hovered() { 0.4 } else { 0.0 };
        // Active slot: SURFACE fill + 1px BORDER (mockup). No left accent bar.
        let fill = lerp_color(NAV_BG, SURFACE, (t + hover * (1.0 - t)).min(1.0));
        ui.painter().rect_filled(rect, 8.0 * S, fill);
        if t > 0.02 {
            ui.painter().rect_stroke(
                rect,
                8.0 * S,
                Stroke::new(1.0, lerp_color(NAV_BG, BORDER, t)),
            );
        }
        let col = lerp_color(TEXT3, ACCENT, t);
        draw_icon(ui.painter(), icon, rect.center(), 18.0 * S, col);
        if resp.clicked() {
            self.page = page;
        }
    }

    // ----------------------------------------------------------------- pages

    fn dashboard(&mut self, ui: &mut egui::Ui) {
        let (gutter, cw) = stage_metrics(ui.ctx());
        let results = *self.tele.results.lock().unwrap();
        // Raw model outputs (per eye: [presence, openness, _, squeeze, _]) + raw brow, for the
        // thin raw bars under each ML-parameters gauge.
        let ml5 = *self.tele.ml5.lock().unwrap();
        let brow_raw = *self.tele.brow_raw.lock().unwrap();
        // Learned per-eye openness baseline, shown as a red tick on the wide raw bar.
        let baselines = *self.tele.baselines.lock().unwrap();
        let pupil = *self.tele.pupil.lock().unwrap();
        // The live camera preview is presentation-only and defaults off. Do not even
        // clone the Arc-backed raw frames or the processed NET Vecs while hidden.
        // Acquisition, inference, recordings, and HTTP output remain native-rate.
        let mut eye_camera_preview = self.config.ui.eye_camera_preview;
        let eye_camera_preview_before = eye_camera_preview;
        let frames = if eye_camera_preview {
            self.tele.stereo_frames()
        } else {
            [None, None]
        };
        // Input/hover events can repaint independently of the scheduled UI cadence.
        // Upload only a genuinely newer camera pair, at the visible preview's 120 Hz
        // ceiling. Hiding preview removes both the frame clone and texture upload.
        let eye_generations = std::array::from_fn(|eye| {
            frames[eye]
                .as_ref()
                .map(|frame| frame.generation)
                .unwrap_or(0)
        });
        let newer_eye_frame = eye_generations != self.last_eye_texture_generation;
        let eye_texture_missing = self.tex_l.is_none() || self.tex_r.is_none();
        let upload_eye_textures = eye_camera_preview
            && self.window_drag_started_at.is_none()
            && (newer_eye_frame || eye_texture_missing)
            && (self.last_eye_texture_upload.elapsed() >= LIVE_EYE_TEXTURE_INTERVAL
                || eye_texture_missing);
        if upload_eye_textures {
            self.last_eye_texture_upload = Instant::now();
            self.last_eye_texture_generation = eye_generations;
        }
        let ctx = ui.ctx().clone();
        let drop = self.drop_pct();
        // Live rates + nominal dims for the per-HMD labels (resolution/fps vary by HMD).
        let cam_rates = [self.rates[0], self.rates[1]];
        let ml_rate = self.rates[3];
        let (eye_w, eye_h) = (self.tele.eye_w, self.tele.eye_h);

        ui.horizontal_top(|ui| {
            ui.add_space(gutter);
            ui.vertical(|ui| {
                ui.set_width(cw);
                self.console_pipeline(ui, cw);
                ui.add_space(SP3);

                // Two side-by-side cards of EQUAL width (1:1) so the eye images
                // get as much room as the parameters. egui's `Frame` over-reports
                // available width inside a horizontal row, so we pin each card into
                // an explicit fixed-width rect rather than trusting the row.
                let cams_w = (cw - SP3) / 2.0;
                let ml_w = cw - cams_w - SP3;
                // Optional eyebrow stereo sync is a live output preference. Each eye
                // still runs its own model/post-process; only the emitted pair is averaged.
                let mut brow_lr_sync = self.pipeline.tuning.lock().unwrap().brow_lr_sync;
                let brow_lr_sync_before = brow_lr_sync;
                let (tex_l, tex_r) = (&mut self.tex_l, &mut self.tex_r);
                let ml_frames = if eye_camera_preview && self.net_view {
                    self.tele.ml_input.lock().unwrap().clone()
                } else {
                    [None, None]
                };
                let net_view = &mut self.net_view;
                // Render the (taller) eye card at its natural height, then force the
                // ML card to MATCH it within the same frame — so bottoms align like
                // the mockup's `align-items: stretch`, with NO cross-frame feedback
                // (a max()-cache loop here compounds tiny overflow and runs away).
                let start = ui.cursor().min;
                let cams = ui.allocate_new_ui(
                    egui::UiBuilder::new().max_rect(Rect::from_min_size(start, vec2(cams_w, 1.0))),
                    |ui| {
                        eye_cams_card(
                            ui,
                            cams_w,
                            0.0,
                            &frames,
                            &ml_frames,
                            net_view,
                            &pupil,
                            cam_rates,
                            eye_w,
                            eye_h,
                            tex_l,
                            tex_r,
                            upload_eye_textures,
                            &mut eye_camera_preview,
                            &ctx,
                        )
                    },
                );
                let h_eye = cams.response.rect.height();
                if eye_camera_preview != eye_camera_preview_before {
                    self.config.ui.eye_camera_preview = eye_camera_preview;
                    if !eye_camera_preview {
                        // Release the GPU textures immediately. Re-enabling starts from
                        // the newest source generation instead of showing a stale eye.
                        self.tex_l = None;
                        self.tex_r = None;
                        self.last_eye_texture_generation = [0; 2];
                        self.last_eye_texture_upload =
                            Instant::now() - LIVE_EYE_TEXTURE_INTERVAL;
                    }
                    let _ = self.config.save(&crate::config::config_path());
                    ctx.request_repaint();
                }
                if cams.inner {
                    if self.gaze_residual_capture.is_running() {
                        self.dream_air_msg = Some((
                            "Image controls are locked during landmark/residual recording.".into(),
                            WARN,
                        ));
                    } else if self.geometry_evidence_locked() {
                        self.dream_air_msg = Some((
                            "Image controls are locked until the current image-alignment capture is fitted, audited, or discarded."
                                .into(),
                            WARN,
                        ));
                    } else {
                        self.show_geom_modal = true;
                    }
                }
                let brow_legacy_before = self
                    .be
                    .as_ref()
                    .map(|status| status.sranipal_brow_link.load(Ordering::Relaxed))
                    .unwrap_or(self.config.output.vrcft_sranipal_brow_link);
                let mut brow_legacy = brow_legacy_before;
                let brow_model_loaded = self.tele.brow_loaded.load(Ordering::Relaxed);
                let mlr = ui.allocate_new_ui(
                    egui::UiBuilder::new().max_rect(Rect::from_min_size(
                        pos2(start.x + cams_w + SP3, start.y),
                        vec2(ml_w, 1.0),
                    )),
                    |ui| {
                        ml_params_card(
                            ui,
                            ml_w,
                            h_eye - 2.0 * CARD_PAD,
                            &results,
                            &ml5,
                            brow_raw,
                            baselines,
                            drop,
                            ml_rate,
                            &mut brow_lr_sync,
                            &mut brow_legacy,
                            brow_model_loaded,
                        )
                    },
                );
                if brow_lr_sync != brow_lr_sync_before {
                    self.pipeline.tuning.lock().unwrap().brow_lr_sync = brow_lr_sync;
                    self.config.tuning.brow_lr_sync = brow_lr_sync;
                    let _ = self.config.save(&crate::config::config_path());
                }
                if brow_legacy != brow_legacy_before {
                    self.config.output.vrcft_sranipal_brow_link = brow_legacy;
                    if let Some(status) = &self.be {
                        status
                            .sranipal_brow_link
                            .store(brow_legacy, Ordering::Relaxed);
                    }
                    // The dashboard source selector is deliberately exclusive: Legacy
                    // lets the VRCFT module derive brows from Wide/Squint; Estimate uses
                    // the independently trained eyebrow model instead.
                    let estimate_enabled = !brow_legacy;
                    self.pipeline
                        .eyebrow_enabled
                        .store(estimate_enabled, Ordering::Relaxed);
                    self.config.ui.eyebrow_enabled = estimate_enabled;
                    let color = if brow_legacy || brow_model_loaded {
                        OK
                    } else {
                        WARN
                    };
                    let mode = if brow_legacy {
                        "Legacy eyebrow"
                    } else {
                        "Estimate (Python)"
                    };
                    match self.config.save(&crate::config::config_path()) {
                        Ok(()) => self.events.push((
                            now_hms(),
                            format!("Eyebrow source: {mode}"),
                            color,
                        )),
                        Err(e) => self.events.push((
                            now_hms(),
                            format!("Eyebrow source changed live only; save failed: {e}"),
                            ERR,
                        )),
                    }
                }
                let row_h = h_eye.max(mlr.response.rect.height());
                // Reset the cursor below the taller card so the log doesn't overlap.
                ui.allocate_rect(Rect::from_min_size(start, vec2(cw, row_h)), Sense::hover());

                ui.add_space(SP3);
                self.terminal_log(ui, cw);
                // ML-input geometry modal (opened by the eye-cameras gear) — overlays all.
            });
        });
    }

    fn console_pipeline(&mut self, ui: &mut egui::Ui, cw: f32) {
        let cam = self.rates[0] > 1.0 && self.rates[1] > 1.0;
        let gaze = self.rates[2] > 1.0;
        let ml = self.tele.ml_loaded && self.rates[3] > 1.0;
        let core = self.rates[4] > 1.0;
        let tracking = !self.pipeline.paused.load(Ordering::Relaxed);
        let names = ["DEVICE", "CAMERA", "GAZE", "ML", "CORE", "OUTPUT"];
        let oks = [cam || gaze, cam, gaze, ml, core, core && tracking];
        // DEVICE sub-label reflects the active transport, not a hardcoded "USB".
        let dev_sub = match self.pipeline.device_key.as_str() {
            "varjo" | "varjo_native" => "SDK",        // native VarjoLib
            "varjo_mjpeg" | "varjo_stream" => "HTTP", // Eye Streamer MJPEG
            "psvr2" => "CAPI",
            "starvr" | "starvr_one" | "pimax_dll" | "pimax_vr4_dll" | "pimax_stream" => "TSE",
            _ => "USB", // auto / pimax_vr4 = WinUSB
        };
        let subs = [
            dev_sub.to_string(),
            format!("{:.0}/s", self.rates[0]),
            format!("{:.0}/s", self.rates[2]),
            if self.tele.ml_loaded {
                format!("{:.0}/s", self.rates[3])
            } else {
                "off".into()
            },
            format!("{:.0}/s", self.rates[4]),
            if tracking {
                "LIVE".into()
            } else {
                "off".into()
            },
        ];
        let icons = [
            Icon::Usb,
            Icon::Camera,
            Icon::Eye,
            Icon::Cpu,
            Icon::Stack,
            Icon::Broadcast,
        ];
        let first_broken = oks.iter().position(|o| !*o);
        let n_ok = oks.iter().filter(|&&o| o).count();
        let inner = cw - 2.0 * CARD_PAD;

        card().show(ui, |ui| {
            ui.set_width(inner);
            // Header: chevron + "PIPELINE" left, "n/6 nodes ok" pinned right. Clicking
            // the header toggles the flow diagram; default collapsed, so the header alone
            // (name + n/6 nodes ok) is the at-a-glance summary and the diagram is opt-in.
            let hdr = ui.horizontal(|ui| {
                let chev = if self.pipeline_open {
                    "\u{25be}"
                } else {
                    "\u{25b8}"
                }; // ▾ open / ▸ collapsed
                ui.label(
                    egui::RichText::new(chev)
                        .monospace()
                        .size(10.0 * S)
                        .color(TEXT2),
                );
                ui.label(
                    egui::RichText::new("PIPELINE")
                        .monospace()
                        .size(10.0 * S)
                        .color(TEXT2),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let col = if n_ok == 6 { OK } else { WARN };
                    ui.label(
                        egui::RichText::new(format!("{n_ok}/6 nodes ok"))
                            .monospace()
                            .size(10.0 * S)
                            .color(col),
                    );
                });
            });
            let toggle = ui.interact(hdr.response.rect, Id::new("pipeline_hdr"), Sense::click());
            if toggle.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
            }
            if toggle.clicked() {
                self.pipeline_open = !self.pipeline_open;
            }
            if !self.pipeline_open {
                return; // collapsed — header only
            }
            ui.add_space(8.0 * S);
            // Branched flow: DEVICE fans out to (CAMERA -> ML) and (GAZE direct);
            // both merge at CORE -> OUTPUT. ML runs on the camera images, gaze
            // comes straight off the device — they're PARALLEL inputs to CORE,
            // not a single chain.
            let nh = 44.0 * S;
            let rg = 10.0 * S;
            let cg = 28.0 * S;
            let area_h = 2.0 * nh + rg;
            let (area, _) = ui.allocate_exact_size(vec2(inner, area_h), Sense::hover());
            let nw = (inner - 4.0 * cg) / 5.0;
            let colx = |i: usize| area.left() + i as f32 * (nw + cg);
            let cam_cy = area.top() + nh / 2.0;
            let gaze_cy = area.top() + nh + rg + nh / 2.0;
            let mid_cy = area.top() + area_h / 2.0;
            let nrect =
                |x: f32, cy: f32| Rect::from_center_size(pos2(x + nw / 2.0, cy), vec2(nw, nh));
            let r_dev = nrect(colx(0), mid_cy);
            let r_cam = nrect(colx(1), cam_cy);
            let r_gaze = nrect(colx(1), gaze_cy);
            let r_ml = nrect(colx(2), cam_cy);
            let r_core = nrect(colx(3), mid_cy);
            let r_out = nrect(colx(4), mid_cy);

            // Connectors first (so the opaque nodes paint over the joins).
            let painter = ui.painter().clone();
            let lc = |a: bool, b: bool| if a && b { OK } else { DECO };
            let seg = |a: Pos2, b: Pos2, c: Color32| {
                painter.line_segment([a, b], Stroke::new(1.6 * S, c))
            };
            let trunk = if oks[0] { OK } else { DECO };
            let sx = r_dev.right() + cg / 2.0;
            seg(pos2(r_dev.right(), mid_cy), pos2(sx, mid_cy), trunk);
            seg(pos2(sx, cam_cy), pos2(sx, gaze_cy), trunk);
            seg(
                pos2(sx, cam_cy),
                pos2(r_cam.left(), cam_cy),
                lc(oks[0], oks[1]),
            );
            seg(
                pos2(sx, gaze_cy),
                pos2(r_gaze.left(), gaze_cy),
                lc(oks[0], oks[2]),
            );
            seg(
                pos2(r_cam.right(), cam_cy),
                pos2(r_ml.left(), cam_cy),
                lc(oks[1], oks[3]),
            );
            let mx = r_core.left() - cg / 2.0;
            let core_in = if oks[4] { OK } else { DECO };
            seg(
                pos2(r_ml.right(), cam_cy),
                pos2(mx, cam_cy),
                lc(oks[3], oks[4]),
            );
            seg(
                pos2(r_gaze.right(), gaze_cy),
                pos2(mx, gaze_cy),
                lc(oks[2], oks[4]),
            );
            seg(pos2(mx, cam_cy), pos2(mx, gaze_cy), core_in);
            seg(pos2(mx, mid_cy), pos2(r_core.left(), mid_cy), core_in);
            seg(
                pos2(r_core.right(), mid_cy),
                pos2(r_out.left(), mid_cy),
                lc(oks[4], oks[5]),
            );

            // Per-node detail rows (device name, stream ids, output state, …) now
            // live here instead of the top bar — click a node to open its card.
            let details = self.node_details();
            let rects = [r_dev, r_cam, r_gaze, r_ml, r_core, r_out];

            for (i, &r) in rects.iter().enumerate() {
                let is_out = i == 5;
                let is_first_broken = Some(i) == first_broken;
                // Healthy nodes get a faint chip (NODE_BG, no border) so they read as
                // grouped without the boxed-tile-grid look; the FIRST broken stage gets
                // a real container (darker fill + amber border) to pull attention.
                let fill = if is_first_broken { INNER } else { NODE_BG };
                let border = if is_first_broken {
                    WARN
                } else {
                    Color32::TRANSPARENT
                };
                let dot = if oks[i] {
                    OK
                } else if is_first_broken {
                    WARN
                } else {
                    LED_OFF
                };
                let icol = if oks[i] {
                    if is_out {
                        OK
                    } else {
                        TEXT2
                    }
                } else {
                    TEXT3
                };
                let vcol = if oks[i] {
                    if is_out {
                        OK
                    } else {
                        TEXT1
                    }
                } else {
                    TEXT3
                };
                pipeline_node(
                    ui, r, icons[i], names[i], &subs[i], dot, icol, vcol, fill, border,
                );
                let resp = ui.interact(r, Id::new(("pnode", i)), Sense::click());
                if resp.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                // Accent border on hover or when this node's card is open.
                if resp.hovered() || self.sel_node == Some(i) {
                    ui.painter()
                        .rect_stroke(r, R_INNER, Stroke::new(1.5 * S, ACCENT));
                }
                if resp.clicked() {
                    self.sel_node = if self.sel_node == Some(i) {
                        None
                    } else {
                        Some(i)
                    };
                }
            }

            // Open detail card (own dark frame so values are readable).
            if let Some(i) = self.sel_node {
                let area = egui::Area::new(Id::new("nodedetail"))
                    .order(egui::Order::Foreground)
                    .fixed_pos(rects[i].left_bottom() + vec2(0.0, 8.0 * S))
                    .show(ui.ctx(), |ui| {
                        // Distinct floating look: darker than the SURFACE cards it
                        // overlays, accent border, drop shadow.
                        egui::Frame::default()
                            .fill(INNER)
                            .stroke(Stroke::new(1.5, ACCENT))
                            .rounding(R_CARD)
                            .inner_margin(egui::Margin::same(CARD_PAD))
                            .shadow(egui::epaint::Shadow {
                                offset: vec2(0.0, 4.0 * S),
                                blur: 14.0 * S,
                                spread: 0.0,
                                color: Color32::from_black_alpha(140),
                            })
                            .show(ui, |ui| node_detail_card(ui, names[i], oks[i], &details[i]));
                    });
                // Dismiss when clicking outside the card and the nodes.
                if ui.input(|inp| inp.pointer.any_pressed()) {
                    if let Some(pos) = ui.ctx().pointer_interact_pos() {
                        let on_node = rects.iter().any(|r| r.contains(pos));
                        if !area.response.rect.contains(pos) && !on_node {
                            self.sel_node = None;
                        }
                    }
                }
            }
        });
    }

    /// Per-pipeline-node detail rows (label, value), surfaced on hover.
    fn node_details(&self) -> [Vec<(&'static str, String)>; 6] {
        let r = self.rates;
        let rate = |x: f32| {
            if x > 1.0 {
                format!("{x:.0}/s")
            } else {
                "—".to_string()
            }
        };
        let cam = r[0] > 1.0 && r[1] > 1.0;
        let gaze = r[2] > 1.0;
        let mlon = self.tele.ml_loaded;
        let pu = *self.tele.pupil.lock().unwrap();
        let pufmt = |p: (f32, bool)| {
            if p.1 {
                format!("{:.1}mm", p.0)
            } else {
                "—".to_string()
            }
        };
        // Live eye-camera dims (first present frame), else the device profile's nominal —
        // so the CAMERA node shows the ACTIVE HMD's resolution, not a hardcoded 200×200.
        let (dw, dh) = {
            let f = self.tele.frames.lock().unwrap();
            f.iter()
                .flatten()
                .next()
                .map(|frame| (frame.width, frame.height))
                .unwrap_or((self.tele.eye_w, self.tele.eye_h))
        };
        [
            vec![
                ("device", self.tele.device_name.clone()),
                ("transport", self.tele.transport.clone()),
                ("streams", self.tele.streams.clone()),
                (
                    "link",
                    if cam || gaze {
                        "streaming".into()
                    } else {
                        "no link".into()
                    },
                ),
            ],
            vec![
                ("format", format!("{dw}×{dh} IR")),
                ("rate L / R", format!("{} / {}", rate(r[0]), rate(r[1]))),
            ],
            vec![
                ("source", self.tele.gaze_src.clone()),
                ("rate", rate(r[2])),
                (
                    "pupil L / R",
                    format!("{} / {}", pufmt(pu[0]), pufmt(pu[1])),
                ),
            ],
            {
                let mut ml = vec![
                    (
                        "model",
                        if mlon {
                            "TVM eyelid net".into()
                        } else {
                            "not loaded".into()
                        },
                    ),
                    ("rate", if mlon { rate(r[3]) } else { "off".into() }),
                    ("outputs", "openness · wide · squeeze".into()),
                ];
                // Eyebrow CNN (optional) — show its live signed output per eye.
                if self.tele.brow_loaded.load(Ordering::Relaxed) {
                    let res = *self.tele.results.lock().unwrap();
                    ml.push(("brow net", "TinyBrowNet (eye-shape)".to_string()));
                    ml.push((
                        "brow L / R",
                        format!("{:+.2} / {:+.2}", res[0].brow, res[1].brow),
                    ));
                }
                ml
            },
            vec![
                ("post-proc", "SRanipal-style".into()),
                ("emit", rate(r[4])),
                (
                    "frame",
                    format!("#{}", self.tele.c_emit.load(Ordering::Relaxed)),
                ),
                ("drop", format!("{:.1}%", self.drop_pct())),
            ],
            {
                let mut out = vec![("sink", "SRanibro → VRCFT".to_string())];
                if let Some(be) = &self.be {
                    let n = be.clients.load(Ordering::Relaxed);
                    out.push(("server", format!("tcp :{}", be.port)));
                    out.push(("clients", n.to_string()));
                    out.push((
                        "state",
                        if n > 0 {
                            "LIVE".into()
                        } else {
                            "waiting".into()
                        },
                    ));
                    out.push(("rate", rate(r[4])));
                } else {
                    out.push(("server", "not started".into()));
                }
                out
            },
        ]
    }

    fn terminal_log(&self, ui: &mut egui::Ui, cw: f32) {
        // A FAULT SUMMARY (first broken stage + plain cause + direct action) over the
        // real EVENT HISTORY (the transitions collected in detect_events). Previously
        // this rendered only current-state rows and the history was never shown.
        let oks = self.stage_oks();
        let names = ["Device", "Camera", "Gaze", "ML", "Core", "Output"];
        let first_broken = oks.iter().position(|o| !*o);
        let ml_loaded = self.tele.ml_loaded;
        let paused = self.pipeline.paused.load(Ordering::Relaxed);
        let drop = self.drop_pct();

        // Plain-language cause + the exact next action for the first broken stage.
        let (sum_col, sum_tag, sum_msg) = if let Some(i) = first_broken {
            let msg = match i {
                0 | 1 | 2 => {
                    // Prefer the ACTIVE adapter's own status line — it's already
                    // device-specific and actionable (e.g. Varjo: "put the headset on";
                    // Pimax/StarVR: "DLL load failed"). Only fall back to a generic,
                    // device-aware hint when the status carries no information.
                    let st = self
                        .pipeline
                        .device_status
                        .lock()
                        .map(|s| s.clone())
                        .unwrap_or_default();
                    let uninformative =
                        st.is_empty() || st == "idle" || st == "n/a" || st == "streaming";
                    let is_varjo = self.pipeline.device_key.starts_with("varjo");
                    let is_psvr2 = self.pipeline.device_key == "psvr2";
                    if !uninformative {
                        format!("{}: {st}", names[i])
                    } else if is_varjo {
                        format!(
                            "{}: no stream → start Varjo Base and put the headset on",
                            names[i]
                        )
                    } else if is_psvr2 {
                        format!(
                            "{}: no stream → start SteamVR and PSVR2Toolkit, then reload",
                            names[i]
                        )
                    } else {
                        format!(
                            "{}: no stream → check the headset connection, then reload",
                            names[i]
                        )
                    }
                }
                3 => {
                    if !ml_loaded {
                        "ML: model not loaded → Settings: set the Eye model (weights file)"
                            .to_string()
                    } else {
                        "ML: loaded but no inferences → waiting on camera frames".to_string()
                    }
                }
                4 => "Core: post-processor stalled → no frames reaching the emitter".to_string(),
                _ => {
                    if paused {
                        "Output: tracking is OFF → turn tracking on to emit".to_string()
                    } else {
                        "Output: not emitting → core stalled upstream".to_string()
                    }
                }
            };
            // Amber (not red) when the only "fault" is the user turning tracking off.
            (if i == 5 && paused { WARN } else { ERR }, "[!!]", msg)
        } else {
            (
                OK,
                "[ok]",
                format!(
                    "all systems nominal · emit {:.0}/s · drop {drop:.1}%",
                    self.rates[4]
                ),
            )
        };

        egui::Frame::default()
            .fill(INNER)
            .stroke(Stroke::new(1.0, BORDER))
            .rounding(R_CARD)
            .inner_margin(egui::Margin::symmetric(12.0 * S, 10.0 * S))
            .show(ui, |ui| {
                ui.set_width(cw - 24.0 * S);
                ui.spacing_mut().item_spacing.y = 3.0 * S;
                // Fault summary (prominent), then a hairline, then the event history.
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0 * S;
                    ui.label(
                        egui::RichText::new(sum_tag)
                            .monospace()
                            .size(10.0 * S)
                            .color(sum_col),
                    );
                    ui.label(
                        egui::RichText::new(&sum_msg)
                            .monospace()
                            .size(10.0 * S)
                            .strong()
                            .color(TEXT1),
                    );
                });
                ui.add_space(4.0 * S);
                let (sep, _) = ui.allocate_exact_size(vec2(cw - 24.0 * S, 1.0), Sense::hover());
                ui.painter().rect_filled(sep, 0.0, BORDER);
                ui.add_space(4.0 * S);
                if self.events.is_empty() {
                    log_line(
                        ui,
                        "[··]",
                        TEXT3,
                        "events",
                        "no transitions yet — stage up/down and tracking toggles log here",
                    );
                } else {
                    // Full chronological history in a bounded, scrollable box (newest at
                    // the bottom) so it never overflows the dashboard and older events stay
                    // reachable. Stamp = wall-clock HH:MM:SS captured when the event fired.
                    egui::ScrollArea::vertical()
                        .max_height(96.0 * S)
                        .auto_shrink([false, true])
                        .stick_to_bottom(true)
                        .show(ui, |ui| {
                            for (ts, msg, col) in &self.events {
                                log_line(ui, ts, *col, "event", msg);
                            }
                        });
                }
            });
    }

    /// Real dropped-frame ratio over the last window (emit cycles that overran the
    /// 120Hz period). 0.0 when the pipeline keeps up.
    fn drop_pct(&self) -> f32 {
        let dropped = self.tele.c_drop.load(Ordering::Relaxed);
        let total = self.tele.c_emit.load(Ordering::Relaxed).max(1);
        (dropped as f32 / total as f32) * 100.0
    }

    /// The Console tab: a live dump of the process's own stdout/stderr (`[xr5]`, `[vr4]`,
    /// `[ml]`, `[brokeneye]`, …), captured by `logcap` and rendered here so runtime logs
    /// are visible without a terminal. Monospace, auto-scrolls to the newest line.
    fn console(&mut self, ui: &mut egui::Ui) {
        let (gutter, cw) = stage_metrics(ui.ctx());
        // Snapshot the last ~1000 lines under the lock, then release it before painting.
        let (lines, total): (Vec<String>, usize) = {
            let q = self.log.lock().unwrap();
            let total = q.len();
            let start = total.saturating_sub(1000);
            (q.iter().skip(start).cloned().collect(), total)
        };
        ui.horizontal_top(|ui| {
            ui.add_space(gutter);
            ui.vertical(|ui| {
                ui.set_width(cw);
                card().show(ui, |ui| {
                    ui.set_width(cw - 2.0 * CARD_PAD);
                    ui.horizontal(|ui| {
                        ui.label(h3("Console"));
                        ui.add_space(SP2);
                        ui.label(label(&format!("{total} lines")));
                        // Right-aligned Clear button empties the ring buffer.
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let clear = egui::Button::new(
                                egui::RichText::new("Clear")
                                    .monospace()
                                    .size(11.0 * S)
                                    .color(TEXT2),
                            )
                            .fill(INNER)
                            .stroke(Stroke::new(1.0, BORDER));
                            if ui.add(clear).clicked() {
                                if let Ok(mut q) = self.log.lock() {
                                    q.clear();
                                }
                            }
                        });
                    });
                    ui.add_space(SP2);
                    // Monospace, auto-following log surface. Fixed height so the card fits
                    // the fixed window; the ScrollArea scrolls within it.
                    egui::Frame::default()
                        .fill(INNER)
                        .stroke(Stroke::new(1.0, BORDER))
                        .rounding(R_INNER)
                        .inner_margin(egui::Margin::symmetric(10.0 * S, 8.0 * S))
                        .show(ui, |ui| {
                            let w = cw - 2.0 * CARD_PAD - 20.0 * S;
                            ui.set_width(w);
                            ui.set_height(content_h() - 150.0 * S);
                            egui::ScrollArea::vertical()
                                .auto_shrink([false, false])
                                .stick_to_bottom(true)
                                .show(ui, |ui| {
                                    ui.set_width(w);
                                    ui.spacing_mut().item_spacing.y = 1.0 * S;
                                    if lines.is_empty() {
                                        ui.label(
                                            egui::RichText::new("(no output captured yet)")
                                                .monospace()
                                                .size(10.5 * S)
                                                .color(TEXT3),
                                        );
                                    }
                                    for l in &lines {
                                        ui.label(
                                            egui::RichText::new(l)
                                                .monospace()
                                                .size(10.5 * S)
                                                .color(log_color(l)),
                                        );
                                    }
                                });
                        });
                });
            });
        });
    }

    fn dream_air_onboarding_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let capture_status =
            if self.calibration_capture_purpose == Some(CalibrationCapturePurpose::Geometry) {
                self.geometry_capture.status()
            } else {
                GeometryCaptureStatus::Idle
            };
        let fit_status = self.geometry_fitter.status();
        let (ready, ready_detail) = self.geometry_capture_ready();
        let recording_path = self.geometry_recording_path.clone();
        let recording_export_attempted = self.geometry_recording_export_attempted;
        let recording_export_in_flight = self.geometry_recording_export_job.is_some();
        let residual_status = self.gaze_residual_capture.status();
        let residual_recording_path = self.gaze_residual_recording_path.clone();
        let residual_export_attempted = self.gaze_residual_export_attempted;
        let residual_export_in_flight = self.gaze_residual_export_job.is_some();

        ui.set_width(width);
        calibration_detail_intro(ui, SessionKind::SafeGeometry.descriptor());
        ui.label(prose(
                        "XR5 only. The candidate must improve untouched validation frames before it can be applied.",
                    ))
                    .on_hover_text(
                        "The inner IR-LED and lens region is excluded. This does not train a model, change Tobii gaze calibration, or use squeeze / Wide as geometry targets.",
                    );
        ui.label(prose(
                        "The labelled recording is saved locally as a ZIP when capture finishes.",
                    ))
                    .on_hover_text(
                        "The ZIP contains raw eye images (biometric data). It stays local unless you choose to share it.",
                    );
        if self.config.has_geometry_override(&self.pipeline.device_key) {
            ui.label(
                egui::RichText::new("SAVED IMAGE ALIGNMENT ACTIVE")
                    .monospace()
                    .strong()
                    .color(OK),
            );
            self.calibration_removal_controls(
                ui,
                SessionKind::SafeGeometry,
                "Remove saved image alignment",
            );
        }
        ui.add_space(SP2);

        match capture_status {
            GeometryCaptureStatus::Rest {
                instruction,
                remaining_s,
                overall,
                awaiting_confirmation,
                ..
            } => {
                self.geometry_rest_guide(
                    ui,
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                );
                if ui.button("Cancel and discard in-memory frames").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                    self.dream_air_msg = Some(("Image-alignment capture cancelled.".into(), WARN));
                }
            }
            GeometryCaptureStatus::Capture {
                kind,
                instruction,
                remaining_s,
                phase_progress,
                overall,
                samples,
                target_open,
                stereo_stalled,
                ..
            } => {
                ui.label(
                    egui::RichText::new("RECORDING")
                        .monospace()
                        .size(14.0 * S)
                        .strong()
                        .color(ACCENT),
                );
                ui.label(label(instruction));
                if let Some(target) = target_open {
                    ui.horizontal(|ui| {
                        ui.label(label(if kind.family() == SampleFamily::HalfOpen {
                            "Hold both eyelids steady at the halfway target"
                        } else {
                            "Follow the slow close/open guide"
                        }));
                        ui.add(
                            egui::ProgressBar::new(target)
                                .desired_width(220.0 * S)
                                .text(format!("target {:.0}% open", target * 100.0)),
                        );
                    });
                } else {
                    ui.add(egui::ProgressBar::new(phase_progress).desired_width(300.0 * S));
                }
                ui.label(num(&format!(
                    "{remaining_s:.1}s left    {samples} stereo samples    overall {:.0}%",
                    overall * 100.0
                )));
                if stereo_stalled {
                    ui.label(
                                    egui::RichText::new(
                                        "No fresh stereo pair for 1 second. Check both XR5 camera streams; this phase is still advancing.",
                                    )
                                    .monospace()
                                    .color(ERR),
                                );
                }
                if ui.button("Cancel and discard in-memory frames").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                    self.dream_air_msg = Some(("Image-alignment capture cancelled.".into(), WARN));
                }
            }
            GeometryCaptureStatus::Done {
                train_samples,
                holdout_samples,
            } => {
                ui.label(
                    egui::RichText::new("CAPTURE COMPLETE")
                        .monospace()
                        .size(13.0 * S)
                        .strong()
                        .color(OK),
                );
                ui.label(num(&format!(
                    "search {train_samples} frames    untouched holdout {holdout_samples} frames"
                )));
                ui.label(label(
                                "Fitting is pure Rust and usually takes several minutes. The current geometry remains live until you explicitly preview or apply a passing result.",
                            ));
                ui.label(label(
                                "Objective audit uses this capture instead of fitting. It probes the active geometry and nearby alternatives with both the current score and the labelled method that originally found the XR5 preset; it never applies a result.",
                            ));
                if let Some(path) = &recording_path {
                    ui.label(
                        egui::RichText::new(format!(
                            "Recording saved automatically: {}",
                            path.display()
                        ))
                        .monospace()
                        .color(OK),
                    );
                } else if recording_export_in_flight {
                    ui.label(num("Saving calibration recording ZIP in the background..."));
                } else if recording_export_attempted {
                    ui.label(
                                    egui::RichText::new(
                                        "Automatic ZIP save failed. The capture is still in memory and fit/audit will not consume it until saving succeeds.",
                                    )
                                    .monospace()
                                    .color(ERR),
                                );
                    if ui.button("Retry recording ZIP save").clicked() {
                        self.export_geometry_recording();
                    }
                } else {
                    ui.label(num("Saving calibration recording ZIP..."));
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            !recording_export_in_flight,
                            egui::Button::new("Run objective audit (recommended)"),
                        )
                        .clicked()
                    {
                        self.start_geometry_audit();
                    }
                    if ui
                        .add_enabled(
                            !recording_export_in_flight,
                            egui::Button::new("Start safe geometry fit"),
                        )
                        .clicked()
                    {
                        self.start_geometry_fit();
                    }
                    if ui
                        .add_enabled(
                            !recording_export_in_flight,
                            egui::Button::new("Discard capture"),
                        )
                        .clicked()
                    {
                        self.geometry_capture.abort();
                        self.geometry_capture_baseline = None;
                        self.geometry_capture_filters = None;
                        self.photometric_capture_baseline = None;
                        self.calibration_capture_purpose = None;
                        self.reset_geometry_recording_export();
                        self.dream_air_msg = Some(("Captured frames discarded.".into(), WARN));
                    }
                });
            }
            GeometryCaptureStatus::Idle => match fit_status {
                GeometryFitStatus::Running {
                    stage,
                    completed,
                    total,
                    log,
                } => {
                    let auditing = stage.contains("audit");
                    ui.label(
                        egui::RichText::new(if auditing { "AUDITING" } else { "FITTING" })
                            .monospace()
                            .size(13.0 * S)
                            .strong()
                            .color(ACCENT),
                    );
                    ui.label(label(&stage));
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .show_percentage(),
                    );
                    ui.label(num(&format!(
                        "{completed} / {total} candidate-frame evaluations"
                    )));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    ui.label(label(
                                    "Tracking stays live, but CPU load is intentionally higher during the offline search.",
                                ));
                    if ui
                        .button(if auditing {
                            "Cancel audit; keep current geometry"
                        } else {
                            "Cancel fit; keep current geometry"
                        })
                        .clicked()
                    {
                        self.geometry_fitter.cancel();
                    }
                }
                GeometryFitStatus::Done { result, log } => {
                    let result_color = if result.accepted { OK } else { WARN };
                    ui.label(
                        egui::RichText::new(if result.accepted {
                            "HOLDOUT PASS"
                        } else {
                            "KEEP CURRENT GEOMETRY"
                        })
                        .monospace()
                        .size(13.0 * S)
                        .strong()
                        .color(result_color),
                    );
                    ui.label(label(&result.reason));
                    ui.add_space(SP2);
                    ui.label(num(&format!(
                        "search score   current {:.3}  candidate {:.3}",
                        result.baseline_train.score, result.candidate_train.score
                    )));
                    ui.label(num(&format!(
                        "holdout score  current {:.3}  candidate {:.3}  delta {:+.3}",
                        result.baseline_holdout.score,
                        result.candidate_holdout.score,
                        result.holdout_improvement
                    )));
                    ui.label(num(&format!(
                        "holdout separation L/R  {:.2}/{:.2} -> {:.2}/{:.2}",
                        result.baseline_holdout.separation[0],
                        result.baseline_holdout.separation[1],
                        result.candidate_holdout.separation[0],
                        result.candidate_holdout.separation[1]
                    )));
                    ui.label(num(&format!(
                        "holdout slow-close correlation L/R  {:.2}/{:.2} -> {:.2}/{:.2}",
                        result.baseline_holdout.monotonicity[0],
                        result.baseline_holdout.monotonicity[1],
                        result.candidate_holdout.monotonicity[0],
                        result.candidate_holdout.monotonicity[1]
                    )));
                    ui.label(num(&format!(
                        "holdout gaze-open retention L/R  {:.2}/{:.2} -> {:.2}/{:.2}",
                        result.baseline_holdout.gaze_retention[0],
                        result.baseline_holdout.gaze_retention[1],
                        result.candidate_holdout.gaze_retention[0],
                        result.candidate_holdout.gaze_retention[1]
                    )));
                    ui.label(num(&format!(
                                    "holdout false squeeze L/R  {:.3}/{:.3} -> {:.3}/{:.3}    native gaze {:.0}%",
                                    result.baseline_holdout.gaze_squeeze_fp[0],
                                    result.baseline_holdout.gaze_squeeze_fp[1],
                                    result.candidate_holdout.gaze_squeeze_fp[0],
                                    result.candidate_holdout.gaze_squeeze_fp[1],
                                    result.baseline_holdout.gaze_evidence_rate * 100.0,
                                )));
                    ui.label(num(&format!(
                                    "capture evidence  degraded {}  invalid {}  valid CLOSED train/holdout {}/{}",
                                    result.degraded_static_phases,
                                    result.invalid_static_phases,
                                    result.valid_closed_phases[0],
                                    result.valid_closed_phases[1],
                                )));
                    if let Some(seed) = &result.appearance_seed {
                        ui.add_space(SP2);
                        ui.label(
                            egui::RichText::new(if seed.search_eligible {
                                "NEUTRAL-APPEARANCE INITIAL GEOMETRY"
                            } else {
                                "NEUTRAL APPEARANCE: DIAGNOSTIC ONLY"
                            })
                            .monospace()
                            .size(11.0 * S)
                            .strong()
                            .color(if seed.search_eligible { OK } else { WARN }),
                        );
                        ui.label(num(&format!(
                            "confidence {:.0}%    {}{}",
                            seed.confidence * 100.0,
                            seed.reason,
                            if result.candidate_from_appearance_seed {
                                "    selected by training search"
                            } else {
                                ""
                            }
                        )));
                        for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                            let value = &seed.eyes[eye];
                            let descriptor = value.descriptor;
                            let g = value.geometry;
                            ui.label(num(&format!(
                                            "neutral {name} pupil {:.1}/{:.1} contrast {:.1} axis {:+.1} spread {:.1}px/{:.1}deg{}   crop {:.3}/{:.3}/{:.3}/{:.3} rot {:+.1}",
                                            descriptor.pupil_center_px[0],
                                            descriptor.pupil_center_px[1],
                                            descriptor.pupil_contrast,
                                            descriptor.aperture_angle_deg,
                                            descriptor.block_center_spread_px,
                                            descriptor.block_angle_spread_deg,
                                            if descriptor.stereo_recovered {
                                                " stereo-recovered"
                                            } else {
                                                ""
                                            },
                                            g.crop_left,
                                            g.crop_right,
                                            g.crop_top,
                                            g.crop_bottom,
                                            g.rotate_deg,
                                        )));
                        }
                    }
                    if let Some(seed) = &result.motion_seed {
                        ui.add_space(SP2);
                        ui.label(
                            egui::RichText::new(if seed.search_eligible {
                                "MOTION-DERIVED INITIAL GEOMETRY"
                            } else {
                                "MOTION GEOMETRY: DIAGNOSTIC ONLY"
                            })
                            .monospace()
                            .size(11.0 * S)
                            .strong()
                            .color(if seed.search_eligible { OK } else { WARN }),
                        );
                        ui.label(num(&format!(
                            "confidence {:.0}%    {}{}",
                            seed.confidence * 100.0,
                            seed.reason,
                            if result.candidate_from_motion_seed {
                                "    selected by training search"
                            } else {
                                ""
                            }
                        )));
                        for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                            let value = &seed.eyes[eye];
                            let g = value.geometry;
                            ui.label(num(&format!(
                                            "motion {name} crop {:.3}/{:.3}/{:.3}/{:.3}   rot {:+.1}   descriptor error {:.4}",
                                            g.crop_left,
                                            g.crop_right,
                                            g.crop_top,
                                            g.crop_bottom,
                                            g.rotate_deg,
                                            value.fit_error
                                        )));
                        }
                    }
                    ui.add_space(SP2);
                    for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                        let before = result.baseline[eye];
                        let after = result.candidate[eye];
                        ui.label(num(&format!(
                                        "{name} crop {:.3}/{:.3}/{:.3}/{:.3} -> {:.3}/{:.3}/{:.3}/{:.3}   scaleY {:.3}->{:.3}   rot {:.1}->{:.1}",
                                        before.crop_left,
                                        before.crop_right,
                                        before.crop_top,
                                        before.crop_bottom,
                                        after.crop_left,
                                        after.crop_right,
                                        after.crop_top,
                                        after.crop_bottom,
                                        before.scale_y,
                                        after.scale_y,
                                        before.rotate_deg,
                                        after.rotate_deg
                                    )));
                    }
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    ui.add_space(SP2);
                    let already_applied =
                        self.config.geometry_for(&self.pipeline.device_key) == result.candidate;
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                !already_applied,
                                egui::Button::new("Preview candidate live"),
                            )
                            .clicked()
                        {
                            self.preview_geometry_candidate(&result);
                        }
                        if result.accepted
                            && ui
                                .add_enabled(
                                    !already_applied,
                                    egui::Button::new("Apply validated candidate"),
                                )
                                .clicked()
                        {
                            self.apply_geometry_candidate(&result, false);
                        }
                        if ui
                            .add_enabled(
                                self.geometry_preview_restore.is_some(),
                                egui::Button::new("Discard preview"),
                            )
                            .clicked()
                        {
                            self.restore_geometry_preview(true);
                        }
                    });
                    if !result.accepted && !already_applied {
                        ui.label(
                                        egui::RichText::new(
                                            "The automatic recommendation is still to keep the current geometry. Preview the candidate before overriding it.",
                                        )
                                        .size(10.0 * S)
                                        .color(WARN),
                                    );
                        ui.checkbox(
                            &mut self.geometry_unvalidated_ack,
                            "I understand this candidate failed holdout validation",
                        );
                        if ui
                            .add_enabled(
                                self.geometry_unvalidated_ack,
                                egui::Button::new("Apply unvalidated candidate"),
                            )
                            .clicked()
                        {
                            self.apply_geometry_candidate(&result, true);
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui.button("Record again").clicked() {
                            self.start_geometry_capture();
                        }
                        if ui
                            .add_enabled(
                                !self.gaze_residual_capture.is_running(),
                                egui::Button::new("Open manual image controls"),
                            )
                            .clicked()
                        {
                            self.show_geom_modal = true;
                            self.geom_tab = 0;
                        }
                    });
                }
                GeometryFitStatus::AuditDone { result, log } => {
                    let no_go = !result.evidence_ready
                        || result.confident_wrong_count > 0
                        || !result.edge_drift_axes.is_empty();
                    ui.label(
                        egui::RichText::new(if no_go {
                            "OBJECTIVE AUDIT: NO-GO"
                        } else {
                            "OBJECTIVE AUDIT COMPLETE"
                        })
                        .monospace()
                        .size(13.0 * S)
                        .strong()
                        .color(if no_go { ERR } else { WARN }),
                    );
                    ui.label(label(&result.reason));
                    ui.add_space(SP2);
                    let reference = &result.cases[0];
                    let current_best = &result.cases[result.current_best];
                    let legacy_best = &result.cases[result.legacy_best];
                    ui.label(num(&format!(
                        "active reference  current {:.3} +/- {:.3}   legacy {:.3} +/- {:.3}",
                        reference.current_score.mean,
                        reference.current_score.stddev,
                        reference.legacy_score.mean,
                        reference.legacy_score.stddev,
                    )));
                    ui.label(num(&format!(
                        "current-objective best  {}  {:.3}   legacy-at-that-case {:.3}",
                        current_best.name,
                        current_best.current_score.mean,
                        current_best.legacy_score.mean,
                    )));
                    ui.label(num(&format!(
                        "legacy-discovery best   {}  {:.3}   current-at-that-case {:.3}",
                        legacy_best.name,
                        legacy_best.legacy_score.mean,
                        legacy_best.current_score.mean,
                    )));
                    ui.label(num(&format!(
                                    "reference span {:.3}   half error {:.3}   bimodality {:.3}   reproducibility {:.3}",
                                    reference.absolute_span.mean,
                                    reference.half_error.mean,
                                    reference.bimodality.mean,
                                    reference.reproducibility,
                                )));
                    ui.label(num(&format!(
                        "reference half position L {:.3} +/- {:.3}   R {:.3} +/- {:.3}",
                        reference.half_position[0].mean,
                        reference.half_position[0].stddev,
                        reference.half_position[1].mean,
                        reference.half_position[1].stddev,
                    )));
                    let half = &result.half_quality;
                    ui.label(num(&format!(
                                    "held-half quality  position L/R {:.3}/{:.3}   spread {:.3}/{:.3}   block delta {:.3}/{:.3}",
                                    half.position[0],
                                    half.position[1],
                                    half.normalized_stddev[0],
                                    half.normalized_stddev[1],
                                    half.block_disagreement[0],
                                    half.block_disagreement[1],
                                )));
                    ui.label(num(&format!(
                        "native openness cross-check coverage L/R {:.0}%/{:.0}% (warning only)",
                        half.native_coverage[0] * 100.0,
                        half.native_coverage[1] * 100.0,
                    )));
                    if !result.evidence_ready {
                        ui.label(
                                        egui::RichText::new(
                                            "Evidence is not repeatable enough to redesign the objective. Repeat the recording before drawing a conclusion.",
                                        )
                                        .monospace()
                                        .color(WARN),
                                    );
                    }
                    for warning in &half.warnings {
                        ui.label(
                            egui::RichText::new(format!("WARN: {warning}"))
                                .monospace()
                                .color(WARN),
                        );
                    }
                    if result.confident_wrong_count > 0 {
                        ui.label(
                                        egui::RichText::new(format!(
                                            "{} confident-wrong probe(s): current score improved beyond fold noise while legacy or unsupervised evidence regressed.",
                                            result.confident_wrong_count
                                        ))
                                        .monospace()
                                        .color(ERR),
                                    );
                        for case in result.cases.iter().filter(|case| case.confident_wrong) {
                            ui.label(num(&format!(
                                            "  {}  current {:.3}  legacy {:.3}  span {:.3}  half {:.3}  bimodal {:.3}",
                                            case.name,
                                            case.current_score.mean,
                                            case.legacy_score.mean,
                                            case.absolute_span.mean,
                                            case.half_error.mean,
                                            case.bimodality.mean,
                                        )));
                        }
                    }
                    if !result.edge_drift_axes.is_empty() {
                        ui.label(
                            egui::RichText::new(format!(
                                "Still improving at search boundary: {}",
                                result.edge_drift_axes.join(", ")
                            ))
                            .monospace()
                            .color(ERR),
                        );
                    }
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    ui.label(label(
                                    "This diagnostic never changed live or saved geometry. Record again to run a fit or repeat the audit.",
                                ));
                    if ui.button("Record again").clicked() {
                        self.start_geometry_capture();
                    }
                }
                GeometryFitStatus::Failed { message, log } => {
                    ui.label(
                        egui::RichText::new("FIT FAILED SAFELY")
                            .monospace()
                            .strong()
                            .color(ERR),
                    );
                    ui.label(label(&message));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    if ui.button("Record again").clicked() {
                        self.start_geometry_capture();
                    }
                }
                GeometryFitStatus::Cancelled { log } => {
                    ui.label(label("Fit cancelled. Current geometry was not changed."));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    if ui.button("Record again").clicked() {
                        self.start_geometry_capture();
                    }
                }
                GeometryFitStatus::Idle => {
                    let mark = if ready { "  OK" } else { "WAIT" };
                    ui.label(
                        egui::RichText::new(mark)
                            .monospace()
                            .strong()
                            .color(if ready { OK } else { WARN }),
                    );
                    ui.label(label(&ready_detail));
                    ui.label(num(&format!(
                        "guided capture about {:.0}s; fitting is normally several minutes",
                        crate::geometry_calib::total_seconds()
                    )));
                    if ui
                        .add_enabled(ready, egui::Button::new("Start automatic image alignment"))
                        .clicked()
                    {
                        self.start_geometry_capture();
                    }
                    ui.add_space(SP2);
                    if ui
                        .add_enabled(
                            !self.gaze_residual_capture.is_running(),
                            egui::Button::new("Open manual image controls"),
                        )
                        .clicked()
                    {
                        self.show_geom_modal = true;
                        self.geom_tab = 0;
                    }
                }
            },
        }

        if let Some(error) = &self.geometry_capture.last_error {
            ui.label(egui::RichText::new(error).monospace().color(ERR));
        }
        if self.geometry_rollback.is_some() {
            let rollback_ready = !self.geometry_capture.is_running()
                && !self.geometry_fitter.is_running()
                && !self.photometric_fitter.is_running();
            if ui
                .add_enabled(
                    rollback_ready,
                    egui::Button::new("Rollback last applied geometry"),
                )
                .clicked()
            {
                self.rollback_geometry();
            }
        }
        if let Some((message, color)) = &self.dream_air_msg {
            ui.label(
                egui::RichText::new(message)
                    .monospace()
                    .size(10.0 * S)
                    .color(*color),
            );
        }

        // Superseded by the all-HMD Gaze-direction eyelid correction card.
        if false {
            ui.add_space(SP2);
            egui::CollapsingHeader::new(h3("Landmark / residual research recording"))
                .id_salt("xr5_landmark_residual_capture_card")
                .default_open(false)
                .show(ui, |ui| {
                    ui.add_space(SP2);
                    ui.label(label(
                        "Dream Air / XR5 only. This separate diagnostic records nine categorical gaze targets, slow eyelid motion, natural blinks, raw eye images, and reported Tobii gaze/pupil data.",
                    ));
                    ui.label(label(
                        "It does not run Safe Geometry Fit, change geometry, train a model, or assume the instructed target is calibrated gaze truth.",
                    ));
                    let steamvr_changed = ui
                        .checkbox(
                            &mut self.config.ui.steamvr_overlay,
                            "Show the wide-angle target inside SteamVR (recommended)",
                        )
                        .changed();
                    if steamvr_changed {
                        match self.config.save(&crate::config::config_path()) {
                            Ok(()) => self.sync_gaze_residual_vr_overlay(),
                            Err(error) => {
                                self.dream_air_msg = Some((
                                    format!("Could not save SteamVR target setting: {error}"),
                                    ERR,
                                ));
                            }
                        }
                    }
                    ui.label(num(&self.vr_research_overlay.status_text()));
                    ui.label(num(
                        "If SteamVR is unavailable, the larger desktop target is used and minimizing pauses the recording.",
                    ));
                    match residual_status {
                        GazeResidualStatus::Idle => {
                            ui.label(num(&format!(
                                "about {:.0}s; saved automatically as a separate biometric-data ZIP",
                                crate::gaze_residual_calib::total_seconds()
                            )));
                            if ui
                                .add_enabled(
                                    ready
                                        && !self.geometry_capture.is_running()
                                        && !self.geometry_fitter.is_running()
                                        && !self.photometric_fitter.is_running(),
                                    egui::Button::new("Start nine-point research recording"),
                                )
                                .clicked()
                            {
                                self.start_gaze_residual_capture();
                            }
                        }
                        GazeResidualStatus::Ready { instruction } => {
                            ui.label(
                                egui::RichText::new("READY - TIMER STOPPED")
                                    .monospace()
                                    .size(14.0 * S)
                                    .strong()
                                    .color(ACCENT),
                            );
                            ui.label(label(&instruction));
                            ui.label(num(
                                "Read this first. Recording begins only after the button or Space key is pressed.",
                            ));
                            if ui.button("Begin recording (Space)").clicked() {
                                self.begin_gaze_residual_capture();
                            }
                            if ui.button("Cancel").clicked() {
                                self.gaze_residual_capture.abort();
                                self.gaze_residual_snapshot = None;
                            }
                        }
                        GazeResidualStatus::Running {
                            progress,
                            remaining_s,
                            target,
                            action,
                            holdout,
                            recording,
                            settling,
                            paused,
                            samples_in_phase,
                            stereo_stalled,
                            instruction,
                            ..
                        } => {
                            ui.label(
                                egui::RichText::new(if paused {
                                    "PAUSED"
                                } else if stereo_stalled {
                                    "WAITING FOR FRESH STEREO FRAMES"
                                } else if target.is_none() {
                                    "READ THE NEXT EYELID STEP"
                                } else if !recording {
                                    "PRACTICE - NOT RECORDED"
                                } else if settling {
                                    "SET TARGET"
                                } else {
                                    "RECORDING"
                                })
                                    .monospace()
                                    .size(14.0 * S)
                                    .strong()
                                    .color(if stereo_stalled { ERR } else { ACCENT }),
                            );
                            ui.label(label(&instruction));
                            if let Some(target) = target {
                                let (rect, _) = ui.allocate_exact_size(
                                    vec2(240.0 * S, 135.0 * S),
                                    Sense::hover(),
                                );
                                let painter = ui.painter_at(rect);
                                painter.rect_filled(rect, R_INNER, INNER);
                                painter.rect_stroke(
                                    rect,
                                    R_INNER,
                                    Stroke::new(1.0, BORDER),
                                );
                                let [tx, ty] = target.screen_xy();
                                let point = pos2(
                                    rect.center().x + tx * rect.width() * 0.38,
                                    rect.center().y + ty * rect.height() * 0.36,
                                );
                                painter.circle_filled(point, 8.0 * S, ACCENT);
                                painter.circle_stroke(point, 12.0 * S, Stroke::new(1.0, TEXT1));
                            }
                            let action_name = match action {
                                Some(GazeResidualAction::RelaxedOpen) => "relaxed open",
                                Some(GazeResidualAction::HalfOpen) => "half open",
                                Some(GazeResidualAction::GentleClosed) => "gently closed",
                                Some(GazeResidualAction::SlowCloseOpen) => "slow close/open",
                                Some(GazeResidualAction::NaturalBlink) => "one natural blink",
                                Some(GazeResidualAction::LeftWink) => "left wink",
                                Some(GazeResidualAction::RightWink) => "right wink",
                                None => "warm-up",
                            };
                            ui.label(num(&if !recording {
                                format!(
                                    "practice pass    {action_name}    no frames saved    {remaining_s:.1}s left"
                                )
                            } else {
                                format!(
                                    "{} pass    {action_name}    {samples_in_phase} fresh samples    {remaining_s:.1}s left",
                                    if holdout { "untouched holdout" } else { "training" }
                                )
                            }));
                            if stereo_stalled {
                                ui.label(
                                    egui::RichText::new(
                                        "This phase will not advance until fresh stereo frames meet its evidence quota.",
                                    )
                                    .monospace()
                                    .color(ERR),
                                );
                            }
                            ui.add(egui::ProgressBar::new(progress).show_percentage());
                            if ui.button("Cancel and discard research frames").clicked() {
                                self.gaze_residual_capture.abort();
                                self.gaze_residual_snapshot = None;
                                self.gaze_residual_export_attempted = false;
                                self.gaze_residual_recording_path = None;
                                self.dream_air_msg = Some((
                                    "Landmark/residual recording cancelled.".into(),
                                    WARN,
                                ));
                            }
                        }
                        GazeResidualStatus::Done {
                            samples,
                            evidence_complete,
                            missing_phases,
                        } => {
                            ui.label(
                                egui::RichText::new(if evidence_complete {
                                    "RESEARCH RECORDING COMPLETE"
                                } else {
                                    "RESEARCH RECORDING INCOMPLETE"
                                })
                                    .monospace()
                                    .size(13.0 * S)
                                    .strong()
                                    .color(if evidence_complete { OK } else { ERR }),
                            );
                            ui.label(num(&format!("{samples} labelled stereo frames")));
                            if !evidence_complete {
                                ui.label(
                                    egui::RichText::new(format!(
                                        "Missing evidence in phase IDs: {}",
                                        missing_phases
                                            .iter()
                                            .map(usize::to_string)
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    ))
                                    .monospace()
                                    .color(ERR),
                                );
                            }
                            if let Some(path) = &residual_recording_path {
                                ui.label(
                                    egui::RichText::new(format!("Saved: {}", path.display()))
                                        .monospace()
                                        .color(OK),
                                );
                            } else if residual_export_in_flight {
                                ui.label(num(
                                    "Saving research recording ZIP in the background...",
                                ));
                            } else if residual_export_attempted {
                                ui.label(
                                    egui::RichText::new(
                                        "Automatic ZIP save failed; the recording remains in memory.",
                                    )
                                    .monospace()
                                    .color(ERR),
                                );
                                if ui.button("Retry research ZIP save").clicked() {
                                    self.export_gaze_residual_recording();
                                }
                                if ui.button("Discard failed recording").clicked() {
                                    self.gaze_residual_capture.abort();
                                    self.gaze_residual_snapshot = None;
                                    self.gaze_residual_export_attempted = false;
                                    self.gaze_residual_recording_path = None;
                                    self.dream_air_msg = Some((
                                        "Unsaved landmark/residual recording discarded.".into(),
                                        WARN,
                                    ));
                                }
                            } else {
                                ui.label(num("Preparing research recording ZIP..."));
                            }
                            if residual_recording_path.is_some()
                                && ui.button("Record another session").clicked()
                            {
                                self.start_gaze_residual_capture();
                            }
                        }
                    }
                    if let Some(error) = &self.gaze_residual_capture.last_error {
                        ui.label(egui::RichText::new(error).monospace().color(ERR));
                    }
                });
        }
    }

    #[cfg(any())]
    fn dream_air_onboarding_card_legacy(&mut self, ui: &mut egui::Ui, width: f32) {
        let preflight = self.current_preflight();
        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            let title = self
                .quality_report
                .as_ref()
                .map(|quality| format!("Dream Air setup    QUALITY {:.0}", quality.score))
                .unwrap_or_else(|| "Dream Air setup".into());
            egui::CollapsingHeader::new(h3(&title))
                .id_salt("dream_air_setup_card")
                .default_open(false)
                .show(ui, |ui| {
            ui.add_space(SP2);
            ui.label(label(
                "Checks this headset, measures your real eyelid range, and disables EyeWide per eye when its signal is not reliable.",
            ));
            if let Some(serial) = &self.eyechip_serial {
                ui.label(num(&format!("EyeChip {serial}")));
            }

            ui.add_space(SP2);
            for check in &preflight.checks {
                let mark = if check.passed { "OK" } else { "WAIT" };
                let color = if check.passed { OK } else { WARN };
                ui.horizontal(|ui| {
                    ui.set_min_height(14.0 * S);
                    ui.label(
                        egui::RichText::new(format!("{mark:>4}"))
                            .monospace()
                            .size(10.0 * S)
                            .strong()
                            .color(color),
                    );
                    let full = format!("{} - {}", check.name, check.detail);
                    ui.add(
                        egui::Label::new(
                            egui::RichText::new(&full)
                                .monospace()
                                .size(10.0 * S)
                                .color(TEXT2),
                        )
                        .truncate(),
                    )
                    .on_hover_text(full);
                });
            }

            ui.add_space(SP3);
            if let Some(session) = &self.guided_calibration {
                ui.label(
                    egui::RichText::new(session.step().title())
                        .monospace()
                        .size(15.0 * S)
                        .strong()
                        .color(ACCENT),
                );
                ui.label(label(session.step().instruction()));
                ui.add(egui::ProgressBar::new(session.step_progress()).show_percentage());
                ui.label(num(&format!("overall {:.0}%", session.progress() * 100.0)));
                if ui.button("Cancel guided calibration").clicked() {
                    self.guided_calibration = None;
                    self.dream_air_msg = Some(("Guided calibration cancelled".into(), WARN));
                }
            } else if let Some(report) = self.guided_report {
                let report_color = if report.passed { OK } else { ERR };
                ui.label(
                    egui::RichText::new(format!(
                        "MEASUREMENT {:.0}/100  {}",
                        report.quality_score,
                        if report.passed { "PASS" } else { "RETRY" }
                    ))
                    .monospace()
                    .size(12.0 * S)
                    .strong()
                    .color(report_color),
                );
                ui.label(num(&format!(
                    "baseline L {:.3} R {:.3}   blink depth L {:.3} R {:.3}",
                    report.baseline[0],
                    report.baseline[1],
                    report.blink_depth[0],
                    report.blink_depth[1]
                )));
                ui.label(num(&format!(
                    "EyeWide L {} (SNR {:.1})   R {} (SNR {:.1})",
                    if report.wide_supported[0] { "YES" } else { "NO" },
                    report.wide_snr[0],
                    if report.wide_supported[1] { "YES" } else { "NO" },
                    report.wide_snr[1]
                )));
                let mapping_color = match report.mapping {
                    MappingVerdict::Correct => OK,
                    MappingVerdict::Ambiguous => WARN,
                    MappingVerdict::Swapped => ERR,
                };
                ui.label(
                    egui::RichText::new(format!("LEFT / RIGHT mapping: {:?}", report.mapping))
                        .monospace()
                        .size(10.0 * S)
                        .color(mapping_color),
                );
                ui.horizontal(|ui| {
                    let apply = egui::Button::new(
                        egui::RichText::new("Apply measured profile")
                            .monospace()
                            .strong()
                            .color(BG),
                    )
                    .fill(ACCENT);
                    if ui.add_enabled(report.passed, apply).clicked() {
                        self.apply_guided_calibration(report);
                    }
                    if ui.button("Discard").clicked() {
                        self.guided_report = None;
                    }
                });
                if report.mapping == MappingVerdict::Swapped
                    && ui.button("Fix L/R mapping and rerun").clicked()
                {
                    let swapped = !self.pipeline.swap_eyes.load(Ordering::Relaxed);
                    self.pipeline.swap_eyes.store(swapped, Ordering::Relaxed);
                    self.persist_mapping();
                    self.guided_report = None;
                    self.dream_air_msg = Some(("L/R mapping changed - rerun the guided measurement".into(), ACCENT));
                }
            } else {
                let start = egui::Button::new(
                    egui::RichText::new("Start guided calibration")
                        .monospace()
                        .strong()
                        .color(if preflight.ready { BG } else { TEXT3 }),
                )
                .fill(if preflight.ready { ACCENT } else { INNER });
                if ui.add_enabled(preflight.ready, start).clicked() {
                    self.start_guided_calibration();
                }
                if !preflight.ready {
                    ui.label(label("Look straight ahead and wait until every preflight row says OK."));
                }
            }

            if let Some(quality) = &self.quality_report {
                ui.add_space(SP2);
                ui.horizontal(|ui| {
                    let reason = quality
                        .reasons
                        .first()
                        .map(String::as_str)
                        .unwrap_or("No current warnings.");
                    let color = if quality.reasons.is_empty() { OK } else { WARN };
                    ui.label(
                        egui::RichText::new(if quality.reasons.is_empty() { "  OK" } else { "WAIT" })
                            .monospace()
                            .size(10.0 * S)
                            .strong()
                            .color(color),
                    );
                    ui.add(egui::Label::new(label(reason)).truncate())
                        .on_hover_text(reason);
                    if reason.contains("Recenter")
                        && ui
                            .add_enabled(
                                !self.guided_capture_running(),
                                egui::Button::new("Recenter now"),
                            )
                            .clicked()
                    {
                        self.pipeline.recenter.store(true, Ordering::Relaxed);
                    }
                });
            }
            ui.add_space(SP2);
            ui.horizontal(|ui| {
                if ui.button("Export support ZIP").clicked() {
                    self.export_dream_air_support_bundle();
                }
                let enabled = *self.pipeline.wide_enabled.lock().unwrap();
                ui.label(label(&format!(
                    "EyeWide output: L {} / R {}",
                    if enabled[0] { "on" } else { "off" },
                    if enabled[1] { "on" } else { "off" }
                )));
            });
            if let Some((message, color)) = &self.dream_air_msg {
                ui.label(
                    egui::RichText::new(message)
                        .monospace()
                        .size(10.0 * S)
                        .color(*color),
                );
            }
                });
        });
    }

    fn apply_wide_source_live(&mut self) {
        let requested = self.edit.wide_source;
        let previous = self.pipeline.selected_wide_source();
        if let Err(message) = self.pipeline.set_wide_source(requested) {
            self.edit.wide_source = previous;
            self.dream_air_msg = Some((message.into(), WARN));
            return;
        }

        let saved_previous = self.config.hmd.wide_source;
        self.config.hmd.wide_source = requested;
        if let Err(error) = self.config.save(&crate::config::config_path()) {
            let _ = self.pipeline.set_wide_source(previous);
            self.config.hmd.wide_source = saved_previous;
            self.edit.wide_source = previous;
            self.dream_air_msg = Some((
                format!(
                    "Could not save EyeWide source; kept {}: {error}",
                    previous.as_str()
                ),
                ERR,
            ));
            return;
        }

        self.dream_air_msg = Some((
            format!(
                "EyeWide source changed live to {}; the eye camera stayed connected",
                requested.as_str()
            ),
            OK,
        ));
        self.events.push((
            now_hms(),
            format!("XR5 EyeWide source: {}", requested.as_str()),
            OK,
        ));
    }
    fn dream_air_wide_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let session_count = crate::ml::wide_calfit::completed_sessions(self.wide.root())
            .map(|sessions| sessions.len())
            .unwrap_or(0);
        let capture_status = self.wide.status();
        let fit_status = self.wide_fitter.status();
        let raw = *self.tele.wide_raw.lock().unwrap();
        let custom = *self.tele.wide_custom.lock().unwrap();
        let sranipal = *self.tele.wide_sranipal.lock().unwrap();
        let model_loaded = self.tele.wide_loaded.load(Ordering::Relaxed);
        let model_active = self.tele.wide_custom_active.load(Ordering::Relaxed);
        let mut output_enabled = self.pipeline.eye_wide_enabled.load(Ordering::Relaxed);
        let wide_ready = *self.tele.wide_ready.lock().unwrap();
        let bootstrap_seen = *self.tele.wide_bootstrap_seen.lock().unwrap();
        let residual_busy = self.gaze_residual_capture.is_running()
            || (self.gaze_residual_capture.is_done()
                && self.gaze_residual_recording_path.is_none());
        let busy = self.wide.is_running() || self.wide_fitter.is_running() || residual_busy;
        let mut start_capture = false;
        let mut cancel_capture = false;
        let mut start_fit = false;
        let mut delete_all = false;
        let mut apply_source = false;

        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            let title = format!(
                "XR5 EyeWide    {}{}{}",
                if model_loaded {
                    "WIDE MODEL READY"
                } else {
                    "NO WIDE MODEL"
                },
                if model_active {
                    "    WIDE CUSTOM ACTIVE"
                } else {
                    ""
                },
                if output_enabled {
                    ""
                } else {
                    "    WIDE OUTPUT DISABLED"
                }
            );
            egui::CollapsingHeader::new(h3(&title))
                .id_salt("xr5_image_wide_card")
                .default_open(true)
                .show(ui, |ui| {
            ui.label(label(
                "Learns EyeWide directly from the 200x200 XR5 cameras instead of relying on SRanipal's unstable Wide channel.",
            ));
            ui.label(
                egui::RichText::new("Eye images stay on this PC and are never uploaded.")
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
            );
            if ui
                .checkbox(&mut output_enabled, "Enable EyeWide output")
                .on_hover_text(
                    "When off, SRanibro sends EyeWide = 0 for both eyes. Capture, fitting, and diagnostics remain available, and the loaded model stays ready.",
                )
                .changed()
            {
                self.pipeline
                    .eye_wide_enabled
                    .store(output_enabled, Ordering::Relaxed);
                self.config.ui.eye_wide_enabled = output_enabled;
                match self.config.save(&crate::config::config_path()) {
                    Ok(()) => self.events.push((
                        now_hms(),
                        format!(
                            "EyeWide output {}",
                            if output_enabled { "enabled" } else { "disabled" }
                        ),
                        if output_enabled { OK } else { WARN },
                    )),
                    Err(e) => {
                        let message = format!("EyeWide changed live only; save failed: {e}");
                        self.dream_air_msg = Some((message.clone(), ERR));
                        self.events.push((now_hms(), message, ERR));
                    }
                }
            }
            if !output_enabled {
                ui.label(
                    egui::RichText::new(
                        "EyeWide output is disabled. Capture, fitting, and diagnostics remain available.",
                    )
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
                );
            }
            ui.add_space(SP2);
            ui.label(num(&format!(
                "same-frame A/B   SRanipal L {:.3} R {:.3}   Custom L {:.3} R {:.3}   raw L {:.3} R {:.3}",
                sranipal[0], sranipal[1], custom[0], custom[1], raw[0], raw[1]
            )));
            if model_loaded && !wide_ready.iter().all(|ready| *ready) {
                let offset_note = if bootstrap_seen.iter().any(|seen| {
                    *seen >= crate::core::wide_state::bootstrap_fallback_after()
                }) {
                    " (adapting neutral offset)"
                } else {
                    ""
                };
                ui.label(
                    egui::RichText::new(format!(
                        "CUSTOM CALIBRATING   L {} samples R {} samples{offset_note}",
                        bootstrap_seen[0], bootstrap_seen[1]
                    ))
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
                );
            }

            ui.add_space(SP2);
            ui.horizontal(|ui| {
                ui.label(label("Output source"));
                egui::ComboBox::from_id_salt("wide_source")
                    .selected_text(self.edit.wide_source.as_str())
                    .show_ui(ui, |ui| {
                        for source in WideSource::ALL {
                            ui.add_enabled_ui(
                                source == WideSource::Sranipal || model_loaded,
                                |ui| {
                                    ui.selectable_value(
                                        &mut self.edit.wide_source,
                                        source,
                                        source.as_str(),
                                    );
                                },
                            );
                        }
                    });
                let source_available =
                    self.edit.wide_source == WideSource::Sranipal || model_loaded;
                let source_changed =
                    self.edit.wide_source != self.pipeline.selected_wide_source();
                if ui
                    .add_enabled(
                        !busy && source_available && source_changed,
                        egui::Button::new("Apply source"),
                    )
                    .on_hover_text("Changes the Wide provider live; the eye camera is not restarted.")
                    .clicked()
                {
                    apply_source = true;
                }
            });
            ui.label(label(
                "Auto uses a fresh calibrated custom result and otherwise falls back to SRanipal. Custom never silently falls back. Source changes are live and do not restart the eye camera.",
            ));
            if !model_loaded {
                ui.label(
                    egui::RichText::new(
                        "Fit or install an XR5 EyeWide model before selecting Auto or Custom.",
                    )
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
                );
            }

            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            ui.label(h3("1. Collect two sessions"));
            ui.label(label(
                "Complete one run, reseat the headset, then run it again. The newest whole session is kept out of training for validation.",
            ));
            ui.label(num(&format!(
                "completed sessions {session_count}   stereo progress {:.0}%",
                self.wide.progress() * 100.0
            )));
            match &capture_status {
                WideCalibStatus::Idle => {
                    if ui.add_enabled(!busy, egui::Button::new("Start Wide capture")).clicked() {
                        start_capture = true;
                    }
                }
                WideCalibStatus::Rest {
                    instruction,
                    remaining,
                } => {
                    ui.label(
                        egui::RichText::new(*instruction)
                            .monospace()
                            .size(13.0 * S)
                            .strong()
                            .color(ACCENT),
                    );
                    ui.label(num(&format!("starting in {remaining:.1}s")));
                    ui.add(egui::ProgressBar::new(self.wide.progress()).show_percentage());
                    if ui.button("Cancel Wide capture").clicked() {
                        cancel_capture = true;
                    }
                }
                WideCalibStatus::Capture {
                    instruction,
                    folder,
                    captured,
                    target,
                } => {
                    ui.label(
                        egui::RichText::new(*instruction)
                            .monospace()
                            .size(13.0 * S)
                            .strong()
                            .color(ACCENT),
                    );
                    ui.label(num(&format!("{folder}   {captured}/{target} stereo pairs")));
                    ui.add(egui::ProgressBar::new(self.wide.progress()).show_percentage());
                    if ui.button("Cancel Wide capture").clicked() {
                        cancel_capture = true;
                    }
                }
                WideCalibStatus::Done { session } => {
                    ui.label(
                        egui::RichText::new(format!("session saved: {}", session.display()))
                            .monospace()
                            .size(10.0 * S)
                            .color(OK),
                    );
                    if ui
                        .add_enabled(!self.wide_fitter.is_running(), egui::Button::new("Capture another reseat session"))
                        .clicked()
                    {
                        start_capture = true;
                    }
                }
            }
            if let Some(error) = &self.wide.last_error {
                ui.label(egui::RichText::new(error).monospace().size(10.0 * S).color(ERR));
            }

            ui.add_space(SP2);
            ui.horizontal(|ui| {
                if !self.confirm_delete_wide {
                    if ui
                        .add_enabled(!busy, egui::Button::new("Delete local Wide data..."))
                        .clicked()
                    {
                        self.confirm_delete_wide = true;
                    }
                } else {
                    ui.label(egui::RichText::new("Delete every Wide eye image?").color(ERR));
                    if ui.button("Delete permanently").clicked() {
                        delete_all = true;
                    }
                    if ui.button("Keep data").clicked() {
                        self.confirm_delete_wide = false;
                    }
                }
            });

            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            ui.label(h3("2. Fit from a generic base model"));
            let base = self.edit.wide_model.trim();
            let base_label = if base.is_empty() {
                "base model: not set (set Wide model in Settings first)".to_string()
            } else {
                format!("base model: {base}")
            };
            ui.label(num(&base_label));
            let can_fit = session_count >= 2
                && !base.is_empty()
                && std::path::Path::new(base).is_file()
                && !busy;
            if ui
                .add_enabled(can_fit, egui::Button::new("Fit Wide in app (no Python)"))
                .clicked()
            {
                start_fit = true;
            }
            if session_count < 2 {
                ui.label(label("Fit unlocks after two completed sessions."));
            } else if base.is_empty() {
                ui.label(label("A generic XR5 Wide backbone is required; personal fitting cannot start from nothing."));
            }
            match &fit_status {
                WideFitStatus::Idle => {}
                WideFitStatus::Running { log } => {
                    ui.label(egui::RichText::new("FITTING...").monospace().color(ACCENT));
                    if let Some(line) = log.last() {
                        ui.label(num(line));
                    }
                }
                WideFitStatus::Done {
                    sessions,
                    train_frames,
                    val_frames,
                    train_rmse,
                    val_rmse,
                    ..
                } => {
                    ui.label(
                        egui::RichText::new(format!(
                            "PASS   sessions {sessions}   train {train_frames} RMSE {train_rmse:.3}   held-out {val_frames} RMSE {val_rmse:.3}"
                        ))
                        .monospace()
                        .size(10.0 * S)
                        .color(OK),
                    );
                }
                WideFitStatus::Failed { msg, .. } => {
                    ui.label(egui::RichText::new(msg).monospace().size(10.0 * S).color(ERR));
                }
            }

                });
        });

        if start_capture {
            if self.gaze_residual_capture.is_running()
                || self.geometry_evidence_locked()
                || self.brow.is_running()
                || self.fitter.is_running()
                || self.trainer.is_running()
            {
                self.dream_air_msg = Some((
                    "Finish or cancel the active calibration capture/fit first.".into(),
                    WARN,
                ));
                return;
            }
            match self.wide.start() {
                Ok(()) => {
                    self.wide_last_frames = self.tele.frame_generations();
                    self.confirm_delete_wide = false;
                    self.dream_air_msg = Some(("Wide capture started".into(), ACCENT));
                }
                Err(error) => {
                    self.dream_air_msg = Some((format!("Wide capture failed: {error}"), ERR));
                }
            }
        }
        if cancel_capture {
            self.wide.abort();
            self.dream_air_msg = Some(("Partial Wide session deleted".into(), WARN));
        }
        if delete_all {
            match self.wide.delete_all() {
                Ok(()) => {
                    self.confirm_delete_wide = false;
                    self.dream_air_msg = Some(("All local Wide capture data deleted".into(), WARN));
                }
                Err(error) => {
                    self.dream_air_msg = Some((format!("Delete failed: {error}"), ERR));
                }
            }
        }
        if start_fit {
            if self.gaze_residual_capture.is_running()
                || self.geometry_evidence_locked()
                || self.brow.is_running()
                || self.fitter.is_running()
                || self.trainer.is_running()
            {
                self.dream_air_msg = Some((
                    "Finish or cancel the active calibration capture/fit first.".into(),
                    WARN,
                ));
                return;
            }
            self.wide_fit_applied = false;
            let result = self.wide_fitter.start(WideFitInputs {
                backbone_bin: std::path::PathBuf::from(self.edit.wide_model.trim()),
                wide_data_dir: self.wide.root().to_path_buf(),
                seed: 0x5759_4445,
            });
            if let Err(error) = result {
                self.dream_air_msg = Some((format!("Wide fit: {error}"), ERR));
            }
        }
        if apply_source {
            self.apply_wide_source_live();
        }
    }

    fn set_live_eyelid_response(&self, profile: EyelidResponseProfile) {
        self.pipeline.set_live_response(profile);
    }

    fn begin_wear_memory_capture(&mut self, reason: WearCaptureReason) {
        if !self.wear_memory_save_ready() {
            self.begin_wearing_memory_edit();
            self.wear_memory_draft.message = Some((
                "Recovery is now paused. Check open, closed and blinks without correction, then press Save current good state again.".into(),
                WARN,
            ));
            return;
        }
        // A new memory always records the user's uncorrected, explicitly tuned
        // state. Never learn the result of another recalled memory.
        self.wear_memory_matching_suspended = true;
        self.pipeline.clear_wearing_calibration_target_immediately();
        self.wear_memory.cancel_active();
        self.wear_memory.begin_capture(reason, Instant::now());
    }

    fn wear_memory_save_ready(&self) -> bool {
        let target_present = self.pipeline.wearing_calibration_target().is_some();
        let live = *self.tele.eyelid_live.lock().unwrap();
        memory_save_is_uncorrected(
            self.config.ui.wearing_memory_enabled,
            self.wear_memory_matching_suspended,
            target_present,
            live.map(|eye| eye.effective_baseline),
            live.map(|eye| eye.calibrated_baseline),
        )
    }

    fn begin_wearing_memory_edit(&mut self) {
        if self.wear_response_before_edit.is_none() {
            self.wear_response_before_edit = Some(*self.pipeline.eyelid_response.lock().unwrap());
        }
        self.wear_memory_matching_suspended = true;
        self.wear_memory.cancel_active();
        self.pipeline.clear_wearing_calibration_target_immediately();
    }

    fn resume_wearing_memory(&mut self) {
        self.wear_response_before_edit = None;
        self.wear_baseline_before_edit = None;
        self.wear_memory.disable();
        self.wear_closed_capture = None;
        self.wear_memory_matching_suspended = false;
        self.pipeline.clear_wearing_calibration_target_immediately();
        self.wear_memory_draft.message = Some((
            "Adjustment finished. Automatic recovery follows the main switch.".into(),
            TEXT2,
        ));
    }

    fn set_wearing_memory_enabled(&mut self, enabled: bool) {
        let mut next = self.config.clone();
        next.ui.wearing_memory_enabled = enabled;
        match next.save(&crate::config::config_path()) {
            Ok(()) => {
                self.config = next;
                if !enabled {
                    self.wear_memory.disable();
                    self.pipeline.clear_wearing_calibration_target_immediately();
                    self.wear_closed_capture = None;
                    self.wear_memory_draft.message = Some((
                        "Automatic wearing recovery is off. Saved states were kept.".into(),
                        TEXT2,
                    ));
                } else {
                    self.resume_wearing_memory();
                    self.wear_memory_draft.message = Some((
                        "Automatic wearing recovery is on. Only confirmed good states can be matched."
                            .into(),
                        OK,
                    ));
                }
            }
            Err(error) => {
                self.wear_memory_draft.message = Some((
                    format!("Could not save Wearing Memory setting: {error}"),
                    ERR,
                ));
            }
        }
    }

    fn begin_closed_point_capture(&mut self, eye: usize) {
        if eye >= 2 || self.wear_closed_capture.is_some() {
            return;
        }
        self.begin_wearing_memory_edit();
        self.begin_eyelid_response_preview();
        self.wear_closed_capture = Some(ClosedPointCapture {
            eye,
            started: Instant::now(),
            samples: Vec::with_capacity(96),
            sampling_cued: false,
            last_generation: self.tele.c_ml.load(Ordering::Acquire),
        });
        self.wear_memory_draft.message = Some((
            format!(
                "Close the {} eye. Sampling starts after the short preparation tone.",
                if eye == 0 { "left" } else { "right" }
            ),
            ACCENT,
        ));
        self.recording_audio
            .cue(RecordingCue::Prepare, self.config.ui.recording_audio_cues);
    }

    fn update_closed_point_capture(&mut self, now: Instant) {
        const PREPARE: Duration = Duration::from_millis(650);
        const SAMPLE: Duration = Duration::from_millis(750);
        let Some(capture) = self.wear_closed_capture.as_mut() else {
            return;
        };
        let elapsed = now.saturating_duration_since(capture.started);
        if elapsed >= PREPARE && !capture.sampling_cued {
            capture.sampling_cued = true;
            self.recording_audio
                .cue(RecordingCue::Sampling, self.config.ui.recording_audio_cues);
        }
        if elapsed >= PREPARE && elapsed < PREPARE + SAMPLE {
            let generation = self.tele.c_ml.load(Ordering::Acquire);
            if generation == capture.last_generation {
                return;
            }
            capture.last_generation = generation;
            let live = *self.tele.eyelid_live.lock().unwrap();
            let raw = live[capture.eye].raw_openness;
            let presence = self.tele.ml5.lock().unwrap()[capture.eye][0];
            if raw.is_finite() && presence.is_finite() && presence > 0.10 {
                capture.samples.push(raw);
            }
            return;
        }
        if elapsed < PREPARE + SAMPLE {
            return;
        }

        let Some(mut completed) = self.wear_closed_capture.take() else {
            return;
        };
        if completed.samples.len() < 6 {
            self.wear_memory_draft.message = Some((
                "Closed point was not set: too few valid model samples. Try again.".into(),
                WARN,
            ));
            self.recording_audio
                .cue(RecordingCue::Warning, self.config.ui.recording_audio_cues);
            return;
        }
        completed.samples.sort_by(f32::total_cmp);
        let median = completed.samples[completed.samples.len() / 2];
        let p10 = completed.samples[completed.samples.len() / 10];
        let p90 = completed.samples[completed.samples.len() * 9 / 10];
        let eye = completed.eye;
        let live = *self.tele.eyelid_live.lock().unwrap();
        let Some(mut edit) = self
            .eyelid_response_preview
            .as_ref()
            .map(|preview| preview.edit)
        else {
            return;
        };
        // Derive the currently effective baseline from the displayed open point.
        // This remains correct while a remembered appearance offset is fading.
        let effective_baseline = live[eye].effective_open_ref + edit.open_point_offset[eye];
        let raw_depth = effective_baseline - median;
        let minimum = (edit.open_point_offset[eye] + EyelidResponseProfile::MIN_MANUAL_RANGE)
            .max(EyelidResponseProfile::CLOSED_POINT_DEPTH_MIN);
        if !raw_depth.is_finite() || raw_depth < minimum {
            self.wear_memory_draft.message = Some((
                format!(
                    "{} eye did not close far enough in the raw model ({median:.3}). Try again or drag the 0% point manually.",
                    if eye == 0 { "Left" } else { "Right" }
                ),
                WARN,
            ));
            self.recording_audio
                .cue(RecordingCue::Warning, self.config.ui.recording_audio_cues);
            return;
        }
        edit.manual_range = true;
        edit.closed_point_depth[eye] =
            raw_depth.clamp(minimum, EyelidResponseProfile::CLOSED_POINT_DEPTH_MAX);
        if let Some(preview) = self.eyelid_response_preview.as_mut() {
            preview.edit = edit;
            preview.link_eyes = false;
            preview.dirty = true;
            preview.message = None;
        }
        self.set_live_eyelid_response(edit);
        self.save_eyelid_response_preview();
        self.wear_memory_draft.closed_set[eye] = true;
        self.wear_memory_draft.message = Some((
            format!(
                "{} 0% point set from {:.3} (spread {:.3}). Test it live, then save only if it feels correct.",
                if eye == 0 { "Left" } else { "Right" },
                median,
                p90 - p10
            ),
            OK,
        ));
        self.recording_audio
            .cue(RecordingCue::Complete, self.config.ui.recording_audio_cues);
    }

    fn wear_neutral_eligible(&self) -> bool {
        let results = *self.tele.results.lock().unwrap();
        let ml = *self.tele.ml5.lock().unwrap();
        let native = self.tele.fresh_runtime_sample();
        for eye in 0..2 {
            let result = results[eye];
            if !ml[eye][0].is_finite() || ml[eye][0] <= 0.10 {
                return false;
            }
            let sample = if eye == 0 { native.left } else { native.right };
            // Teaching requires stronger evidence than recovery. Missing native
            // capability can use the manually verified response, but invalid
            // reported data is not an unsupported capability.
            let fallback = (ml[eye][1].is_finite() && result.openness_valid && !result.blink)
                .then_some(result.openness);
            if !crate::wear_memory::neutral_openness(
                &sample,
                fallback,
                self.wear_memory.capture_pending(),
            ) {
                return false;
            }
            if sample.gaze_valid && sample.gaze.iter().all(|value| value.is_finite()) {
                let transverse =
                    (sample.gaze[0] * sample.gaze[0] + sample.gaze[1] * sample.gaze[1]).sqrt();
                if transverse
                    .atan2(sample.gaze[2].abs().max(1e-4))
                    .to_degrees()
                    > 18.0
                {
                    return false;
                }
            }
        }
        true
    }

    fn update_wear_memory(&mut self, now: Instant) {
        // Unit discovery can enumerate USB devices; only startup/reload does it.
        let context = WearMemoryContext::new(
            &self.pipeline.device_key,
            self.wear_memory.context().unit_id.clone(),
            wear_memory_fingerprint(&self.config, &self.pipeline),
        );
        if self.wear_memory.context() != &context {
            self.pipeline.clear_wearing_calibration_target_immediately();
            self.wear_memory = WearMemory::load(context);
            self.wear_thumbnails.clear();
            self.wear_memory_draft = WearingMemoryDraft::default();
            self.wear_closed_capture = None;
            self.wear_memory_matching_suspended = false;
            self.wear_response_before_edit = None;
            self.wear_baseline_before_edit = None;
        }
        let cursor = self.wear_memory.last_generation();
        self.wear_memory
            .ingest(self.tele.calibration_frames_after(cursor), now);
        if !self.config.ui.wearing_memory_enabled && !self.wear_memory.capture_pending() {
            return;
        }
        if self.wear_memory_matching_suspended && !self.wear_memory.capture_pending() {
            return;
        }
        let calibration = *self.tele.calibration.lock().unwrap();
        let live = calibration.map(|calibration| {
            let eyelid_live = *self.tele.eyelid_live.lock().unwrap();
            WearLiveCalibration {
                calibration,
                response: *self.pipeline.eyelid_response.lock().unwrap(),
                wide_baseline: *self.tele.wide_baselines.lock().unwrap(),
                wide_entry_ref: [eyelid_live[0].wide_entry_ref, eyelid_live[1].wide_entry_ref],
            }
        });
        let eligible = self.wear_neutral_eligible();
        let mut events = Vec::new();
        if let Some(warning) = self.wear_memory.take_load_warning() {
            events.push(WearMemoryEvent::Warning(warning));
        }
        events.extend(self.wear_memory.update(now, eligible, live));
        for event in events {
            match event {
                WearMemoryEvent::Saved { count, updated } => {
                    self.wear_thumbnails.clear();
                    self.wear_memory_matching_suspended = false;
                    self.wear_response_before_edit = None;
                    self.wear_baseline_before_edit = None;
                    let detail = if updated {
                        format!("Good wearing state updated ({count} saved)")
                    } else {
                        format!("Good wearing state saved ({count} saved)")
                    };
                    self.wear_memory_draft.message = Some((detail.clone(), OK));
                    self.events.push((now_hms(), detail, OK));
                    self.recording_audio
                        .cue(RecordingCue::Saved, self.config.ui.recording_audio_cues);
                }
                WearMemoryEvent::Apply(target) => {
                    if !self.config.ui.wearing_memory_enabled || self.wear_memory_matching_suspended
                    {
                        continue;
                    }
                    let detail = target.map(|target| {
                        format!(
                            "Wearing memory matched {:.0}%",
                            target.confidence.clamp(0.0, 1.0) * 100.0
                        )
                    });
                    self.pipeline.set_wearing_calibration_target(target);
                    if let Some(detail) = detail {
                        self.events.push((now_hms(), detail, ACCENT));
                    }
                }
                WearMemoryEvent::Warning(message) => {
                    self.wear_memory_draft.message = Some((message.clone(), WARN));
                    self.events.push((now_hms(), message, WARN));
                }
            }
        }
    }

    fn begin_eyelid_response_preview(&mut self) {
        if self.eyelid_response_preview.is_some() {
            return;
        }
        let profile = *self.pipeline.eyelid_response.lock().unwrap();
        self.eyelid_response_preview = Some(EyelidResponsePreview::new(profile));
    }

    fn reset_eyelid_response_preview(&mut self) {
        let profile = EyelidResponseProfile::default();
        if let Some(preview) = self.eyelid_response_preview.as_mut() {
            preview.edit = profile;
            preview.dirty = true;
            preview.message = Some((
                "Calibrated response restored live; saving automatically.".into(),
                ACCENT,
            ));
        }
        self.set_live_eyelid_response(profile);
    }

    fn save_eyelid_response_preview(&mut self) {
        let Some(requested) = self
            .eyelid_response_preview
            .as_ref()
            .map(|preview| preview.edit)
        else {
            return;
        };
        let device = self.pipeline.device_key.clone();
        let mut next = self.config.clone();
        next.set_eyelid_response_profile(&device, requested);
        let committed = next.eyelid_response_profile_for(&device);
        match next.save(&crate::config::config_path()) {
            Ok(()) => {
                self.config = next;
                self.set_live_eyelid_response(committed);
                if let Some(preview) = self.eyelid_response_preview.as_mut() {
                    preview.edit = committed;
                    preview.dirty = false;
                    preview.message = Some(("Applied live and saved automatically.".into(), OK));
                }
            }
            Err(error) => {
                if let Some(preview) = self.eyelid_response_preview.as_mut() {
                    // Keep the requested runtime value live. A subsequent edit retries
                    // persistence without surprising the user with a visual rollback.
                    preview.dirty = false;
                    preview.message = Some((
                        format!("Applied live, but automatic save failed: {error}"),
                        ERR,
                    ));
                }
            }
        }
    }

    fn automatic_fit_recovery_control(&mut self, ui: &mut egui::Ui) {
        let mut enabled = self.config.ui.wearing_memory_enabled;
        let response = ui
            .checkbox(&mut enabled, "Automatic wearing-position recovery")
            .on_hover_text(
                "Matches the current eye-camera appearance to states you explicitly saved as working correctly. Saved states remain on this PC while disabled.",
            );
        if response.changed() {
            self.set_wearing_memory_enabled(enabled);
        }
        ui.label(label(if enabled {
            "Uses only complete states confirmed with Save current good state."
        } else {
            "Recovery is off; saved wearing states are kept."
        }));
    }

    fn close_calibration_detail(&mut self) {
        if self
            .eyelid_response_preview
            .as_ref()
            .is_some_and(|preview| preview.dirty)
        {
            self.save_eyelid_response_preview();
        }
        self.eyelid_response_preview = None;
        self.calibration_detail_window = None;
    }

    fn open_calibration_detail(&mut self, detail: CalibrationDetail) {
        if self.eyelid_response_preview.is_some() {
            self.close_calibration_detail();
        }
        if detail == CalibrationDetail::Atomic(SessionKind::EyelidEndpoints) {
            self.begin_eyelid_response_preview();
        }
        self.calibration_detail_window = Some(detail);
    }

    /// Foreground home for guided workflows and individual calibration checks.
    fn calibration_detail_modal(&mut self, ctx: &egui::Context) {
        let Some(detail) = self.calibration_detail_window else {
            return;
        };
        let (title, supported) = match detail {
            CalibrationDetail::InitialSetup => ("Initial setup", true),
            CalibrationDetail::PythonEyelidDataset => (
                "XR5 Python eyelid dataset",
                crate::config::canonical_device_key(&self.pipeline.device_key) == "pimax_xr5",
            ),
            CalibrationDetail::ReseatAssist => ("Wearing position assist", true),
            CalibrationDetail::Atomic(kind) => (
                kind.descriptor().title,
                kind.descriptor()
                    .applicability
                    .supports(&self.pipeline.device_key),
            ),
        };
        let screen = ctx.screen_rect();
        let closed = egui::Area::new(egui::Id::new("calibration_detail_modal"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                ui.painter()
                    .rect_filled(screen, 0.0, Color32::from_black_alpha(205));
                let _scrim = ui.interact(
                    screen,
                    egui::Id::new("calibration_detail_scrim"),
                    Sense::click(),
                );
                let panel_width = (screen.width() * 0.78).clamp(420.0, 780.0 * S);
                let panel_height = (screen.height() * 0.90).clamp(340.0, 820.0 * S);
                let panel =
                    Rect::from_center_size(screen.center(), vec2(panel_width, panel_height));
                ui.painter().rect_filled(panel, 14.0 * S, NAV_BG);
                ui.painter()
                    .rect_stroke(panel, 14.0 * S, Stroke::new(1.0, BORDER));

                let mut close_button = false;
                let inner = panel.shrink(CARD_PAD);
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(inner), |ui| {
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.label(calibration_meta_label("GUIDED CALIBRATION"));
                            ui.label(
                                egui::RichText::new(title)
                                    .size(16.0 * S)
                                    .strong()
                                    .color(TEXT1),
                            );
                        });
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            close_button = ui.button("Close").clicked()
                        });
                    });
                    ui.add_space(SP2);
                    ui.separator();
                    ui.add_space(SP2);
                    egui::ScrollArea::vertical()
                        .id_salt("calibration_detail_scroll")
                        .show(ui, |ui| {
                            if !supported {
                                card().show(ui, |ui| {
                                    ui.label(
                                        egui::RichText::new("NOT AVAILABLE FOR THIS HMD")
                                            .monospace()
                                            .strong()
                                            .color(TEXT3),
                                    );
                                    ui.label(prose(
                                        "Switch to a supported HMD and reopen this check.",
                                    ));
                                    ui.add_enabled(false, egui::Button::new("Record"));
                                });
                                return;
                            }
                            let width = ui.available_width();
                            match detail {
                                CalibrationDetail::InitialSetup => {
                                    self.unified_initial_setup_card(ui, width)
                                }
                                CalibrationDetail::PythonEyelidDataset => {
                                    self.python_eyelid_dataset_card(ui, width)
                                }
                                CalibrationDetail::ReseatAssist => {
                                    self.reseat_assist_card(ui, width)
                                }
                                CalibrationDetail::Atomic(kind) => match kind {
                                    SessionKind::SafeGeometry => {
                                        self.dream_air_onboarding_card(ui, width)
                                    }
                                    SessionKind::Photometric => {
                                        self.photometric_correction_card(ui, width)
                                    }
                                    SessionKind::EyelidEndpoints => {
                                        self.eyelid_endpoint_card(ui, width)
                                    }
                                    SessionKind::GazeDirections => self.gaze_eyelid_card(ui, width),
                                    SessionKind::Winks => self.wink_response_card(ui, width),
                                    SessionKind::NaturalBlinks => {
                                        self.natural_blink_timing_card(ui, width)
                                    }
                                },
                            }
                            ui.add_space(SP2);
                        });
                });
                close_button
            })
            .inner;
        if closed || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.close_calibration_detail();
        }
    }

    fn calibration_workflow_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let device = self.pipeline.device_key.clone();
        let mut requested: Option<CalibrationDetail> = None;
        let endpoints_ready = self
            .tele
            .calibration
            .lock()
            .unwrap()
            .is_some_and(|value| value.left.endpoint_locked && value.right.endpoint_locked);

        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            ui.label(h3("Guided calibration"));
            ui.label(prose(
                "Use this only when you need to repair a tracking symptom, return the HMD to a saved good wearing position, or adjust a supported eye-image path.",
            ));
            ui.add_space(SP2);
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::Problems,
                    "Fix a problem",
                );
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::FitAssist,
                    "Fit assist",
                );
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::Individual,
                    "Advanced image",
                );
            });
            ui.add_space(SP2);

            match self.calibration_view {
                CalibrationView::Problems => {
                    ui.label(prose(
                        "Choose the symptom you actually see. Only that evidence is recorded; a prerequisite is requested only when it is missing.",
                    ));
                    ui.add_space(SP2);
                    for recipe in RECIPES.iter().skip(1) {
                        let steps = crate::calib_session::applicable_steps(*recipe, &device);
                        let Some(kind) = steps.first().copied() else {
                            continue;
                        };
                        let needs_endpoints = matches!(
                            kind,
                            SessionKind::GazeDirections
                                | SessionKind::Winks
                                | SessionKind::NaturalBlinks
                        );
                        let enabled = !needs_endpoints || endpoints_ready;
                        let response = ui.add_enabled_ui(enabled, |ui| {
                            calibration_problem_entry(ui, *recipe, kind.descriptor())
                        });
                        if response.inner {
                            requested = Some(CalibrationDetail::Atomic(kind));
                        }
                        if !enabled {
                            ui.label(
                                egui::RichText::new(
                                    "Open / closed range must be calibrated first.",
                                )
                                .monospace()
                                .size(9.0 * S)
                                .color(WARN),
                            );
                        }
                        ui.add_space(SP2);
                    }
                }
                CalibrationView::FitAssist => {
                    ui.label(
                        egui::RichText::new("Return to your best wearing position")
                            .size(12.0 * S)
                            .strong()
                            .color(TEXT1),
                    );
                    ui.label(prose(
                        "Save a user-confirmed good fit once, then use live guidance after putting the HMD back on. It never changes tracking settings.",
                    ));
                    ui.horizontal(|ui| {
                        ui.label(num("2.5 sec reference"));
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                if calibration_open_button(ui) {
                                    requested = Some(CalibrationDetail::ReseatAssist);
                                }
                            },
                        );
                    });
                }
                CalibrationView::Individual => {
                    ui.label(prose(
                        "Advanced eye-image correction. It is shown only for the current HMD and is not part of normal eyelid setup.",
                    ));
                    ui.add_space(SP2);
                    let mut shown = false;
                    for kind in [SessionKind::SafeGeometry, SessionKind::Photometric] {
                        let descriptor = kind.descriptor();
                        if !descriptor.applicability.supports(&device) {
                            continue;
                        }
                        shown = true;
                        let response = ui.add_enabled_ui(true, |ui| {
                            calibration_session_entry(ui, None, descriptor)
                        });
                        if response.inner {
                            requested = Some(CalibrationDetail::Atomic(kind));
                        }
                        ui.add_space(SP2);
                    }
                    if !shown {
                        ui.label(prose(
                            "No automatic image-correction workflow is required for this HMD.",
                        ));
                    }
                    if crate::config::canonical_device_key(&device) == "pimax_xr5" {
                        ui.add_space(SP2);
                        ui.separator();
                        ui.add_space(SP2);
                        ui.label(
                            egui::RichText::new("Offline model data")
                                .size(12.0 * S)
                                .strong()
                                .color(TEXT1),
                        );
                        egui::Frame::default()
                            .fill(INNER)
                            .stroke(egui::Stroke::new(1.0, BORDER))
                            .rounding(R_INNER)
                            .inner_margin(egui::Margin::symmetric(10.0 * S, 7.0 * S))
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new("Python eyelid dataset")
                                            .size(11.5 * S)
                                            .strong()
                                            .color(TEXT1),
                                    );
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if calibration_open_button(ui) {
                                                requested = Some(
                                                    CalibrationDetail::PythonEyelidDataset,
                                                );
                                            }
                                            ui.label(num(&calibration_duration(
                                                CaptureProtocol::PythonEyelidDataset
                                                    .total_seconds(),
                                            )));
                                        },
                                    );
                                });
                                ui.label(prose(
                                    "Records labelled raw XR5 eye images for offline Python training. It saves a ZIP and never changes live settings.",
                                ));
                            });
                    }
                }
            }
            ui.label(prose(
                "Every recording is saved as a ZIP before analysis. Existing settings remain active until you review and apply a validated result.",
            ));
        });
        if let Some(detail) = requested {
            self.open_calibration_detail(detail);
        }
    }

    fn python_eyelid_dataset_card(&mut self, ui: &mut egui::Ui, width: f32) {
        ui.set_width(width);
        ui.label(prose(
            "Records raw stereo XR5 eye images with explicit pose labels for the offline Python eyelid-model factory.",
        ));
        ui.label(num(
            "Python is not required here. This recorder only creates a training ZIP; it never trains, applies, or replaces a model.",
        ));
        ui.add_space(SP2);
        ui.label(
            egui::RichText::new("BIOMETRIC DATA")
                .monospace()
                .strong()
                .color(WARN),
        );
        ui.label(prose(
            "The ZIP contains raw infrared images of both eyes. Share it only when the wearer explicitly agrees to model development.",
        ));
        ui.add_space(SP2);

        let active = self.gaze_residual_capture.protocol() == CaptureProtocol::PythonEyelidDataset;
        let (ready, ready_detail) = self.python_dataset_capture_ready();
        if !active {
            ui.label(num(&ready_detail));
            ui.label(num(&format!(
                "about {:.0}s; training and untouched holdout evidence are saved together",
                CaptureProtocol::PythonEyelidDataset.total_seconds()
            )));
            if ui
                .add_enabled(
                    ready && !self.geometry_evidence_locked(),
                    egui::Button::new("Start XR5 dataset recording"),
                )
                .clicked()
            {
                self.start_python_eyelid_dataset_capture();
            }
            return;
        }

        match self.gaze_residual_capture.status() {
            GazeResidualStatus::Idle => {
                ui.label(num(&ready_detail));
                if ui
                    .add_enabled(ready, egui::Button::new("Start XR5 dataset recording"))
                    .clicked()
                {
                    self.start_python_eyelid_dataset_capture();
                }
            }
            GazeResidualStatus::Ready { instruction } => {
                ui.label(
                    egui::RichText::new("READY - TIMER STOPPED")
                        .monospace()
                        .strong()
                        .color(ACCENT),
                );
                ui.label(label(&instruction));
                if ui.button("Begin recording (Space)").clicked() {
                    self.begin_gaze_residual_capture();
                }
                if ui.button("Cancel").clicked() {
                    self.gaze_residual_capture.abort();
                    self.gaze_residual_snapshot = None;
                }
            }
            GazeResidualStatus::Running {
                progress,
                remaining_s,
                instruction,
                recording,
                holdout,
                samples_in_phase,
                stereo_stalled,
                ..
            } => {
                ui.label(
                    egui::RichText::new(if stereo_stalled {
                        "WAITING FOR FRESH STEREO FRAMES"
                    } else if recording {
                        "RECORDING DATASET"
                    } else {
                        "PREPARE - NOT RECORDED"
                    })
                    .monospace()
                    .strong()
                    .color(if stereo_stalled { ERR } else { ACCENT }),
                );
                ui.label(label(&instruction));
                ui.label(num(&format!(
                    "{}    {samples_in_phase} samples    {remaining_s:.1}s left",
                    if !recording {
                        "instruction"
                    } else if holdout {
                        "untouched holdout"
                    } else {
                        "training"
                    }
                )));
                ui.add(egui::ProgressBar::new(progress).show_percentage());
                if ui.button("Cancel and discard recording").clicked() {
                    self.gaze_residual_capture.abort();
                    self.gaze_residual_snapshot = None;
                    self.gaze_residual_export_attempted = false;
                    self.gaze_residual_recording_path = None;
                }
            }
            GazeResidualStatus::Done {
                samples,
                evidence_complete,
                ..
            } => {
                ui.label(
                    egui::RichText::new(if evidence_complete {
                        "PYTHON DATASET READY"
                    } else {
                        "DATASET INCOMPLETE"
                    })
                    .monospace()
                    .strong()
                    .color(if evidence_complete { OK } else { ERR }),
                );
                ui.label(num(&format!("{samples} labelled stereo frames")));
                if let Some(path) = &self.gaze_residual_recording_path {
                    ui.label(num(&format!("Saved: {}", path.display())));
                    ui.label(num(
                        "This ZIP can be imported by research/xr5-eyelid-model. No SRanibro setting was changed.",
                    ));
                    if ui.button("Record another session").clicked() {
                        self.start_python_eyelid_dataset_capture();
                    }
                } else if self.gaze_residual_export_job.is_some() {
                    ui.label(num("Saving the biometric ZIP in the background..."));
                } else if self.gaze_residual_export_attempted {
                    ui.label(num("ZIP save failed; the recording remains in memory."));
                    if ui.button("Retry ZIP save").clicked() {
                        self.export_gaze_residual_recording();
                    }
                }
            }
        }
    }

    fn unified_initial_setup_card(&mut self, ui: &mut egui::Ui, width: f32) {
        ui.set_width(width);
        ui.label(prose(
            "Record once. SRanibro replays the same untouched frames through image/lighting, open/closed range, gaze-dependent eyelids, winks and blink timing, then presents one final review.",
        ));
        ui.add_space(SP2);
        let active = self
            .unified_calibration
            .as_ref()
            .is_some_and(|run| run.protocol == CaptureProtocol::InitialSetup);
        if !active {
            let (ready, detail) = self.endpoint_capture_ready();
            ui.label(prose(&detail));
            ui.label(num(&format!(
                "about {:.0}s recording; analysis runs in the background",
                CaptureProtocol::InitialSetup.total_seconds()
            )));
            if ui
                .add_enabled(ready, egui::Button::new("Start one-time recording"))
                .clicked()
            {
                self.start_initial_setup_capture();
            }
            return;
        }

        let run = self.unified_calibration.as_ref().unwrap();
        let stage = run.stage;
        let preprocessing = run.preprocessing;
        let notes = run.notes.clone();
        let blocked = run.blocked.clone();
        let change_count = run.staged_change_count();
        let (stage_name, stage_index) = match stage {
            UnifiedStage::Capturing | UnifiedStage::Saving => ("Recording", 0),
            UnifiedStage::Preprocessing => ("Image / lighting", 1),
            UnifiedStage::Endpoints | UnifiedStage::ReviewEndpoints => ("Open / closed range", 2),
            UnifiedStage::Gaze => ("Looking around", 3),
            UnifiedStage::Winks => ("Left / right wink", 4),
            UnifiedStage::Blinks => ("Blink timing", 5),
            UnifiedStage::FinalReview => ("Final review", 6),
            UnifiedStage::Complete => ("Complete", 7),
        };
        ui.label(
            egui::RichText::new(format!(
                "STEP {stage_index}/7    {}",
                stage_name.to_uppercase()
            ))
            .monospace()
            .strong()
            .color(if blocked.is_some() { ERR } else { ACCENT }),
        );
        ui.add_space(SP2);

        match stage {
            UnifiedStage::Capturing => {
                ui.label(prose(
                    "Follow the foreground or SteamVR guide. Recording pauses if no valid target is visible.",
                ));
            }
            UnifiedStage::Saving => {
                if self.gaze_residual_export_failed() {
                    ui.label(
                        egui::RichText::new("THE RECORDING COULD NOT BE SAVED")
                            .monospace()
                            .strong()
                            .color(ERR),
                    );
                    ui.label(prose(
                        "The frames are kept in memory when possible. Retry the ZIP before analysis, or discard them and record again.",
                    ));
                } else {
                    ui.label(prose(
                        "Saving the biometric ZIP atomically before analysis. The only in-memory copy is not released until this finishes.",
                    ));
                }
            }
            UnifiedStage::Preprocessing => match preprocessing {
                UnifiedPreprocessing::Geometry => match self.geometry_fitter.status() {
                    GeometryFitStatus::Running {
                        stage,
                        completed,
                        total,
                        ..
                    } => {
                        ui.label(prose(&stage));
                        ui.add(
                            egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                                .text(format!("{completed}/{total} candidates")),
                        );
                    }
                    _ => {
                        ui.label(prose("Finishing image-alignment validation..."));
                    }
                },
                UnifiedPreprocessing::Photometric => match self.photometric_fitter.status() {
                    PhotometricStatus::Running {
                        stage,
                        completed,
                        total,
                        ..
                    } => {
                        ui.label(prose(&stage));
                        ui.add(
                            egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                                .text(format!("{completed}/{total} candidates")),
                        );
                    }
                    _ => {
                        ui.label(prose("Finishing lighting-correction validation..."));
                    }
                },
                UnifiedPreprocessing::None => {
                    ui.label(prose("No image correction is required for this HMD."));
                }
            },
            UnifiedStage::Endpoints => match self.endpoint_fitter.status() {
                EndpointFitStatus::Running { completed, total } => {
                    ui.label(prose(
                        "Measuring relaxed open, half-open and gentle full-close for each eye.",
                    ));
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} frames")),
                    );
                }
                _ => {
                    ui.label(prose("Finishing open / closed holdout validation..."));
                }
            },
            UnifiedStage::Gaze => match self.gaze_eyelid_fitter.status() {
                GazeEyelidFitStatus::Running { completed, total } => {
                    ui.label(prose(
                        "Checking whether gaze direction falsely changes eyelid openness.",
                    ));
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} EyeNet replays")),
                    );
                }
                _ => {
                    ui.label(prose("Finishing gaze-direction holdout validation..."));
                }
            },
            UnifiedStage::Winks => match self.wink_fitter.status() {
                WinkFitStatus::Running { completed, total } => {
                    ui.label(prose(
                        "Measuring each wink while protecting the eye that stays open.",
                    ));
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} frames")),
                    );
                }
                _ => {
                    ui.label(prose("Finishing wink holdout validation..."));
                }
            },
            UnifiedStage::Blinks => match self.blink_timing_fitter.status() {
                BlinkTimingFitStatus::Running { completed, total } => {
                    ui.label(prose(
                        "Checking that natural blinks reach a visible closed bottom before reopening.",
                    ));
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} frames")),
                    );
                }
                _ => {
                    ui.label(prose("Finishing blink-timing holdout validation..."));
                }
            },
            UnifiedStage::ReviewEndpoints => {
                ui.label(
                    egui::RichText::new("SETUP NEEDS ONE REPAIR")
                        .monospace()
                        .strong()
                        .color(ERR),
                );
            }
            UnifiedStage::FinalReview => {
                ui.label(
                    egui::RichText::new(format!("{change_count} VALIDATED CHANGE GROUP(S) READY"))
                        .monospace()
                        .strong()
                        .color(if change_count == 0 { TEXT2 } else { OK }),
                );
                ui.label(prose(
                    "Nothing has changed yet. Apply once to save the staged results together, or keep every current setting.",
                ));
            }
            UnifiedStage::Complete => {
                ui.label(
                    egui::RichText::new("INITIAL SETUP COMPLETE")
                        .monospace()
                        .strong()
                        .color(OK),
                );
            }
        }

        if !notes.is_empty() {
            ui.add_space(SP2);
            ui.label(calibration_meta_label("RESULTS SO FAR"));
            for note in notes {
                ui.label(num(&format!("• {note}")));
            }
        }
        if let Some(message) = blocked {
            ui.add_space(SP2);
            ui.label(egui::RichText::new(message).size(10.0 * S).color(ERR));
        }
        ui.add_space(SP2);
        match stage {
            UnifiedStage::FinalReview => {
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            change_count > 0,
                            egui::Button::new(format!(
                                "Apply {change_count} validated change group(s)"
                            )),
                        )
                        .clicked()
                    {
                        self.commit_unified_calibration();
                    }
                    if ui.button("Keep all current settings").clicked() {
                        self.finish_unified_without_changes();
                    }
                });
                if change_count == 0 && ui.button("Finish").clicked() {
                    self.commit_unified_calibration();
                }
            }
            UnifiedStage::ReviewEndpoints => {
                if ui.button("Finish without applying").clicked() {
                    self.finish_unified_without_changes();
                }
            }
            UnifiedStage::Complete => {
                if ui.button("Run initial setup again").clicked() {
                    self.start_initial_setup_capture();
                }
            }
            UnifiedStage::Saving if self.gaze_residual_export_failed() => {
                ui.horizontal(|ui| {
                    if ui.button("Retry saving ZIP").clicked() {
                        self.export_gaze_residual_recording();
                    }
                    if ui.button("Discard frames and record again").clicked() {
                        self.discard_failed_gaze_residual_recording();
                    }
                });
            }
            UnifiedStage::Preprocessing
            | UnifiedStage::Endpoints
            | UnifiedStage::Gaze
            | UnifiedStage::Winks
            | UnifiedStage::Blinks => {
                if ui
                    .button("Cancel analysis and keep current settings")
                    .clicked()
                {
                    self.cancel_unified_analysis_keep_current();
                }
            }
            _ => {}
        }
    }

    fn reseat_assist_card(&mut self, ui: &mut egui::Ui, width: f32) {
        ui.set_width(width);
        ui.label(prose(
            "Save a wearing position only when tracking feels its best. After putting the HMD back on, live guidance helps you return to that fixed reference.",
        ));
        ui.label(
            egui::RichText::new(
                "It stores a compact eye-camera reference on this PC only. Geometry, lighting correction, eyelid calibration and continuous baselines are never changed.",
            )
            .size(9.0 * S)
            .color(TEXT2),
        );
        if let Some(warning) = self.reseat_assist.load_warning() {
            ui.add_space(SP2);
            ui.label(
                egui::RichText::new(warning)
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
            );
        }

        ui.add_space(SP3);
        match self.reseat_assist.guidance() {
            ReseatGuidance::WaitingForFrames if self.reseat_assist.is_active() => {
                ui.label(
                    egui::RichText::new("WAITING FOR STEREO EYE CAMERAS")
                        .monospace()
                        .strong()
                        .color(WARN),
                );
            }
            ReseatGuidance::CapturingReference {
                progress,
                frames,
                step,
            } => {
                ui.label(
                    egui::RichText::new("SAVING THIS GOOD FIT")
                        .monospace()
                        .strong()
                        .color(ACCENT),
                );
                ui.label(prose(step.instruction()));
                ui.add(
                    egui::ProgressBar::new(*progress)
                        .show_percentage()
                        .text(format!("{frames} stable stereo samples")),
                );
            }
            ReseatGuidance::ReferenceSaved { path } => {
                ui.label(
                    egui::RichText::new("GOOD FIT SAVED")
                        .monospace()
                        .strong()
                        .color(OK),
                );
                ui.label(num(&path.display().to_string()));
            }
            ReseatGuidance::EyesClosed => {
                ui.label(
                    egui::RichText::new("OPEN BOTH EYES COMFORTABLY")
                        .monospace()
                        .strong()
                        .color(WARN),
                );
                ui.label(prose(
                    "Guidance pauses during a blink so eyelid motion is never mistaken for HMD movement.",
                ));
            }
            ReseatGuidance::LowConfidence { detail } => {
                ui.label(
                    egui::RichText::new("MATCH UNCERTAIN")
                        .monospace()
                        .strong()
                        .color(WARN),
                );
                ui.label(prose(detail));
            }
            ReseatGuidance::Adjust(estimate) => {
                ui.label(
                    egui::RichText::new(estimate.instruction().to_uppercase())
                        .size(18.0 * S)
                        .strong()
                        .color(ACCENT),
                );
                ui.label(num(&format!(
                    "offset x/y {:+.1}/{:+.1}   scale {:.3}   roll {:+.1} deg   confidence {:.0}%",
                    estimate.correction_px[0],
                    estimate.correction_px[1],
                    estimate.scale,
                    estimate.rotation_deg,
                    estimate.confidence * 100.0,
                )));
            }
            ReseatGuidance::Aligned(estimate) => {
                ui.label(
                    egui::RichText::new("POSITION MATCHED")
                        .size(18.0 * S)
                        .strong()
                        .color(OK),
                );
                ui.label(prose(
                    "Hold this position and tighten the HMD evenly. The match stayed stable for one second.",
                ));
                ui.label(num(&format!(
                    "confidence {:.0}%   L/R score {:.2}/{:.2}",
                    estimate.confidence * 100.0,
                    estimate.eyes[0].score,
                    estimate.eyes[1].score,
                )));
            }
            ReseatGuidance::ReferenceReady { weighted_pixels } => {
                ui.label(
                    egui::RichText::new("REFERENCE CAPTURED - REVIEW BEFORE SAVING")
                        .monospace()
                        .strong()
                        .color(ACCENT),
                );
                ui.label(prose(
                    "The capture passed its coverage checks. Your previous saved reference has not been replaced yet.",
                ));
                ui.label(num(&format!(
                    "stable face-feature pixels L/R  {} / {}",
                    weighted_pixels[0], weighted_pixels[1]
                )));
            }
            ReseatGuidance::Error(error) => {
                ui.label(
                    egui::RichText::new("RESEAT ASSIST STOPPED")
                        .monospace()
                        .strong()
                        .color(ERR),
                );
                ui.label(prose(error));
            }
            ReseatGuidance::WaitingForFrames => {}
        }

        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP2);
        let frames_ready = self
            .tele
            .stereo_frames()
            .iter()
            .all(|frame| frame.is_some());
        let other_calibration_active = self.geometry_capture.is_running()
            || self.geometry_capture.is_done()
            || self.geometry_recording_export_job.is_some()
            || self.gaze_residual_capture.is_running()
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.gaze_eyelid_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.wide.is_running()
            || self.brow.is_running()
            || self.wide_fitter.is_running()
            || self.fitter.is_running()
            || self.trainer.is_running();
        let assist_ready = frames_ready && !other_calibration_active;
        if self.reseat_assist.is_capturing() {
            if ui.button("Cancel reference capture").clicked() {
                self.reseat_assist.stop();
                self.recording_audio
                    .cue(RecordingCue::Cancelled, self.config.ui.recording_audio_cues);
            }
            return;
        }
        if self.reseat_assist.is_active() {
            if ui.button("Stop live assist").clicked() {
                self.reseat_assist.stop();
                self.vr_research_overlay.hide();
            }
            return;
        }
        if self.reseat_assist.has_pending_reference() {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Save as my best fit").clicked() {
                    match self.reseat_assist.confirm_reference() {
                        Ok(_) => self
                            .recording_audio
                            .cue(RecordingCue::Saved, self.config.ui.recording_audio_cues),
                        Err(error) => self.dream_air_msg = Some((error, ERR)),
                    }
                }
                if ui.button("Record again").clicked() {
                    self.reseat_assist.begin_reference_capture(Instant::now());
                    self.reseat_assist
                        .discard_through(self.tele.frame_generations());
                    self.recording_audio
                        .cue(RecordingCue::Prepare, self.config.ui.recording_audio_cues);
                }
                if ui.button("Discard capture").clicked() {
                    self.reseat_assist.discard_pending_reference();
                }
            });
            return;
        }

        if self.reseat_assist.has_reference() {
            ui.horizontal_wrapped(|ui| {
                if ui
                    .add_enabled(assist_ready, egui::Button::new("Start live fit assist"))
                    .clicked()
                {
                    if let Err(error) = self.reseat_assist.begin_assist() {
                        self.dream_air_msg = Some((error, ERR));
                    } else {
                        self.reseat_assist
                            .discard_through(self.tele.frame_generations());
                    }
                }
                if ui
                    .add_enabled(
                        assist_ready,
                        egui::Button::new("Replace good-fit reference"),
                    )
                    .clicked()
                {
                    self.confirm_remove_reseat_reference = false;
                    self.reseat_assist.begin_reference_capture(Instant::now());
                    self.reseat_assist
                        .discard_through(self.tele.frame_generations());
                    self.recording_audio.cue(
                        RecordingCue::Prepare,
                        self.config.ui.recording_audio_cues,
                    );
                }
                if ui.button("Remove reference").clicked() {
                    self.confirm_remove_reseat_reference = true;
                }
            });
            if self.confirm_remove_reseat_reference {
                ui.add_space(SP2);
                ui.label(
                    egui::RichText::new(
                        "Remove only the saved wearing-position image reference? Tracking calibration is unaffected.",
                    )
                    .size(10.0 * S)
                    .color(WARN),
                );
                ui.horizontal(|ui| {
                    if ui.button("Yes, remove reference").clicked() {
                        match self.reseat_assist.remove_reference() {
                            Ok(_) => self.confirm_remove_reseat_reference = false,
                            Err(error) => self.dream_air_msg = Some((error, ERR)),
                        }
                    }
                    if ui.button("Keep reference").clicked() {
                        self.confirm_remove_reseat_reference = false;
                    }
                });
            }
        } else if ui
            .add_enabled(assist_ready, egui::Button::new("Save current good fit"))
            .on_hover_text(
                "Use this only when eyelids are tracking well and the HMD feels correctly positioned.",
            )
            .clicked()
        {
            self.reseat_assist.begin_reference_capture(Instant::now());
            self.reseat_assist
                .discard_through(self.tele.frame_generations());
            self.recording_audio.cue(
                RecordingCue::Prepare,
                self.config.ui.recording_audio_cues,
            );
        }
        if !frames_ready {
            ui.label(prose(
                "Waiting for both eye-camera streams before a reference can be saved or matched.",
            ));
        } else if other_calibration_active {
            ui.label(prose(
                "Finish or cancel the active calibration recording or analysis first.",
            ));
        }
    }

    #[cfg(any())]
    fn calibration_workflow_card_legacy(&mut self, ui: &mut egui::Ui, width: f32) {
        let device = self.pipeline.device_key.clone();
        let mut start_endpoints = false;
        let mut start_gaze_directions = false;
        let mut start_winks = false;
        let mut start_natural_blinks = false;
        let mut start_geometry = false;
        let mut start_photometric = false;
        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            ui.label(h3("Guided calibration"));
            ui.label(prose(
                "Choose a complete setup, fix one symptom, or run a single check.",
            ));
            ui.add_space(SP2);
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::Full,
                    "Full setup",
                );
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::Problems,
                    "Fix a problem",
                );
                ui.selectable_value(
                    &mut self.calibration_view,
                    CalibrationView::Individual,
                    "Single check",
                );
            });
            ui.add_space(SP2);

            match self.calibration_view {
                CalibrationView::Full => {
                    let recipe = RECIPES[0];
                    let steps = crate::calib_session::applicable_steps(recipe, &device);
                    ui.label(label(recipe.symptom));
                    for (index, kind) in steps.iter().enumerate() {
                        let descriptor = kind.descriptor();
                        ui.label(num(&format!(
                            "{}. {} — checks {}",
                            index + 1,
                            descriptor.title,
                            descriptor.checks
                        )));
                        let enabled = match kind {
                            SessionKind::GazeDirections => {
                                !self.gaze_residual_capture.is_running()
                                    && !self.gaze_eyelid_fitter.is_running()
                            }
                            _ => !self.geometry_evidence_locked(),
                        };
                        if ui
                            .add_enabled(
                                enabled,
                                egui::Button::new(format!("Start step {}", index + 1)),
                            )
                            .clicked()
                        {
                            match kind {
                                SessionKind::SafeGeometry => start_geometry = true,
                                SessionKind::Photometric => start_photometric = true,
                                SessionKind::EyelidEndpoints => start_endpoints = true,
                                SessionKind::GazeDirections => start_gaze_directions = true,
                                SessionKind::Winks => start_winks = true,
                                SessionKind::NaturalBlinks => start_natural_blinks = true,
                            }
                        }
                    }
                    ui.label(label(
                        "Run these in order. Review each holdout result and apply or keep the current value before starting the next step. Full uses the same atomic checks as Individual mode.",
                    ));
                }
                CalibrationView::Problems => {
                    for recipe in RECIPES.iter().skip(1) {
                        let steps = crate::calib_session::applicable_steps(*recipe, &device);
                        if steps.is_empty() {
                            continue;
                        }
                        ui.label(
                            egui::RichText::new(recipe.title)
                                .monospace()
                                .strong()
                                .color(TEXT1),
                        );
                        ui.label(label(recipe.symptom));
                        ui.label(num(&format!(
                            "Runs: {}",
                            steps
                                .iter()
                                .map(|kind| kind.descriptor().title)
                                .collect::<Vec<_>>()
                                .join(" -> ")
                        )));
                        if steps.as_slice() == [SessionKind::EyelidEndpoints]
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start this check"),
                                )
                                .clicked()
                        {
                            start_endpoints = true;
                        }
                        if steps.as_slice() == [SessionKind::GazeDirections]
                            && ui
                                .add_enabled(
                                    !self.gaze_residual_capture.is_running()
                                        && !self.gaze_eyelid_fitter.is_running(),
                                    egui::Button::new("Start this check"),
                                )
                                .clicked()
                        {
                            start_gaze_directions = true;
                        }
                        if steps.as_slice() == [SessionKind::Winks]
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start this check"),
                                )
                                .clicked()
                        {
                            start_winks = true;
                        }
                        if steps.as_slice() == [SessionKind::NaturalBlinks]
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start this check"),
                                )
                                .clicked()
                        {
                            start_natural_blinks = true;
                        }
                        ui.add_space(SP2);
                    }
                }
                CalibrationView::Individual => {
                    for kind in SessionKind::ALL {
                        let descriptor = kind.descriptor();
                        if !descriptor.applicability.supports(&device) {
                            continue;
                        }
                        ui.label(
                            egui::RichText::new(descriptor.title)
                                .monospace()
                                .strong()
                                .color(TEXT1),
                        );
                        ui.label(label(descriptor.checks));
                        ui.label(num(&format!(
                            "May change: {}    Never changes: {}    about {:.0}s",
                            descriptor.may_change.user_text(),
                            descriptor.never_changes.user_text(),
                            descriptor.estimated_seconds
                        )));
                        if kind == SessionKind::SafeGeometry
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start image-alignment recording"),
                                )
                                .clicked()
                        {
                            start_geometry = true;
                        }
                        if kind == SessionKind::Photometric
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start lighting recording"),
                                )
                                .clicked()
                        {
                            start_photometric = true;
                        }
                        if kind == SessionKind::EyelidEndpoints
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start endpoint recording"),
                                )
                                .clicked()
                        {
                            start_endpoints = true;
                        }
                        if kind == SessionKind::GazeDirections
                            && ui
                                .add_enabled(
                                    !self.gaze_residual_capture.is_running()
                                        && !self.gaze_eyelid_fitter.is_running(),
                                    egui::Button::new("Start nine-direction recording"),
                                )
                                .clicked()
                        {
                            start_gaze_directions = true;
                        }
                        if kind == SessionKind::Winks
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start wink recording"),
                                )
                                .clicked()
                        {
                            start_winks = true;
                        }
                        if kind == SessionKind::NaturalBlinks
                            && ui
                                .add_enabled(
                                    !self.geometry_evidence_locked(),
                                    egui::Button::new("Start natural-blink recording"),
                                )
                                .clicked()
                        {
                            start_natural_blinks = true;
                        }
                        ui.add_space(SP2);
                    }
                }
            }
            ui.label(
                egui::RichText::new(
                    "Every recording is saved locally as a ZIP before any fit can consume it.",
                )
                .monospace()
                .size(10.0 * S)
                .color(TEXT3),
            );
        });
        if start_endpoints {
            self.start_eyelid_endpoint_capture();
        }
        if start_gaze_directions {
            self.start_gaze_residual_capture();
        }
        if start_winks {
            self.start_wink_capture();
        }
        if start_natural_blinks {
            self.start_natural_blink_capture();
        }
        if start_geometry {
            self.start_geometry_capture();
        }
        if start_photometric {
            self.start_photometric_capture();
        }
    }

    fn wearing_memory_modal(&mut self, ctx: &egui::Context) {
        let screen = ctx.screen_rect();
        let closed = egui::Area::new(egui::Id::new("wearing_memory_modal"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                ui.painter()
                    .rect_filled(screen, 0.0, Color32::from_black_alpha(205));
                let scrim = ui.interact(
                    screen,
                    egui::Id::new("wearing_memory_scrim"),
                    Sense::click(),
                );
                let panel_width = (screen.width() * 0.58).clamp(390.0, 540.0 * S);
                let panel_height = (screen.height() * 0.72).clamp(390.0, 580.0 * S);
                let panel = Rect::from_center_size(
                    screen.center(),
                    vec2(panel_width, panel_height.min(screen.height() - 24.0 * S)),
                );
                ui.painter().rect_filled(panel, 14.0 * S, NAV_BG);
                ui.painter()
                    .rect_stroke(panel, 14.0 * S, Stroke::new(1.0, BORDER));
                let mut close_button = false;
                ui.allocate_new_ui(
                    egui::UiBuilder::new().max_rect(panel.shrink(CARD_PAD)),
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                ui.label(calibration_meta_label("AUTOMATIC RECOVERY"));
                                ui.label(
                                    egui::RichText::new("Wearing memory")
                                        .size(16.0 * S)
                                        .strong()
                                        .color(TEXT1),
                                );
                            });
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| close_button = ui.button("Close").clicked(),
                            );
                        });
                        ui.add_space(SP2);
                        ui.separator();
                        ui.add_space(SP2);
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            ui.label(prose(
                                "Recenter and tune the live bars first. Nothing is learned until you confirm that the complete response is correct.",
                            ));
                            ui.horizontal_wrapped(|ui| {
                                if ui.button("Adjust without recovery").clicked() {
                                    self.begin_wearing_memory_edit();
                                }
                                if self.wear_memory_matching_suspended && ui.button("Finish without saving memory").clicked() {
                                    self.resume_wearing_memory();
                                }
                                if self.wear_response_before_edit.is_some() && ui.button("Undo threshold edits").clicked() {
                                    if let Some(saved) = self.wear_response_before_edit.take() {
                                        let profile = restore_thresholds_only(
                                            *self.pipeline.eyelid_response.lock().unwrap(), saved,
                                        );
                                        if let Some(baseline) = self.wear_baseline_before_edit.take() {
                                            self.pipeline.edit_manual_endpoints(baseline, profile);
                                        }
                                        self.set_live_eyelid_response(profile);
                                        self.eyelid_response_preview = Some(EyelidResponsePreview::new(profile));
                                        self.save_eyelid_response_preview();
                                    }
                                    self.resume_wearing_memory();
                                }
                            });
                            ui.add_space(SP2);
                            let state_row = |ui: &mut egui::Ui, tag: &str, name: &str, color| {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        egui::RichText::new(tag)
                                            .monospace()
                                            .strong()
                                            .size(9.0 * S)
                                            .color(color),
                                    );
                                    ui.label(
                                        egui::RichText::new(name).size(10.0 * S).color(TEXT2),
                                    );
                                });
                            };
                            state_row(
                                ui,
                                if self.wear_memory_draft.recentered { "READY" } else { "NEEDED" },
                                "Open reference",
                                if self.wear_memory_draft.recentered { OK } else { WARN },
                            );
                            for (eye, name) in ["Left 0% point", "Right 0% point"].into_iter().enumerate() {
                                state_row(
                                    ui,
                                    if self.wear_memory_draft.closed_set[eye] { "SET" } else { "CHECK" },
                                    name,
                                    if self.wear_memory_draft.closed_set[eye] { OK } else { TEXT3 },
                                );
                            }
                            state_row(
                                ui,
                                if self.wear_memory_draft.wide_neutral_set { "SET" } else { "OPTIONAL" },
                                "Wide neutral",
                                if self.wear_memory_draft.wide_neutral_set { OK } else { TEXT3 },
                            );

                            ui.add_space(SP3);
                            ui.label(
                                egui::RichText::new("Set a 0% point")
                                    .strong()
                                    .size(11.0 * S)
                                    .color(TEXT1),
                            );
                            ui.label(label(
                                "Close the selected eye and keep it closed through the two tones. Its stable raw green value becomes 0%.",
                            ));
                            ui.add_space(SP2);
                            let capture_busy = self.wear_closed_capture.is_some();
                            let open_ready = self.wear_memory_draft.recentered
                                && self.tele.calibration.lock().unwrap().as_ref().is_some_and(|c|
                                    c.left.baseline_n >= 100 && c.right.baseline_n >= 100);
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        open_ready
                                            && !capture_busy,
                                        egui::Button::new("Set L closed"),
                                    )
                                    .clicked()
                                {
                                    self.begin_closed_point_capture(0);
                                }
                                if ui
                                    .add_enabled(
                                        open_ready
                                            && !capture_busy,
                                        egui::Button::new("Set R closed"),
                                    )
                                    .clicked()
                                {
                                    self.begin_closed_point_capture(1);
                                }
                            });
                            if let Some(capture) = self.wear_closed_capture.as_ref() {
                                let elapsed = Instant::now().saturating_duration_since(capture.started);
                                let remaining = Duration::from_millis(1400).saturating_sub(elapsed);
                                ui.label(num(&format!(
                                    "{} eye · {:.1}s",
                                    if capture.eye == 0 { "LEFT" } else { "RIGHT" },
                                    remaining.as_secs_f32()
                                )));
                            } else if !self.wear_memory_draft.recentered {
                                ui.label(label("Recenter first while looking straight ahead."));
                            } else if !open_ready {
                                ui.label(label("Learning the open reference. Keep both eyes relaxed."));
                            }

                            ui.add_space(SP3);
                            ui.separator();
                            ui.add_space(SP3);
                            ui.label(
                                egui::RichText::new("Everything works correctly?")
                                    .strong()
                                    .size(11.0 * S)
                                    .color(TEXT1),
                            );
                            ui.label(label(
                                "Test open, closed and natural blinks. Only this confirmation teaches automatic recovery.",
                            ));
                            ui.add_space(SP2);
                            let save_ready = open_ready
                                && !capture_busy
                                && self.wear_memory_save_ready()
                                && !self.wear_memory.capture_pending();
                            if !self.wear_memory_save_ready() {
                                ui.label(label("Choose Adjust without recovery, then verify open, closed and blinks before saving."));
                            }
                            let save = egui::Button::new(
                                egui::RichText::new("Save current good state")
                                    .monospace()
                                    .strong()
                                    .color(BG),
                            )
                            .fill(ACCENT);
                            if ui
                                .add_enabled_ui(save_ready, |ui| {
                                    ui.add_sized([ui.available_width(), 34.0 * S], save)
                                })
                                .inner
                                .clicked()
                            {
                                self.begin_wear_memory_capture(WearCaptureReason::GoodState);
                                self.wear_memory_draft.message = Some((
                                    "Keep both eyes relaxed and look straight ahead for one second."
                                        .into(),
                                    ACCENT,
                                ));
                                self.recording_audio.cue(
                                    RecordingCue::Prepare,
                                    self.config.ui.recording_audio_cues,
                                );
                            }
                            if self.wear_memory.capture_pending() {
                                ui.label(num(
                                    "Saving neutral appearance · keep both eyes relaxed",
                                ));
                            }
                            ui.add_space(SP2);
                            let memory_status = if self.wear_memory_matching_suspended {
                                format!(
                                    "WAITING FOR CONFIRMATION · {} SAVED",
                                    self.wear_memory.profile_count()
                                )
                            } else if let Some(confidence) =
                                self.wear_memory.active_confidence()
                            {
                                format!(
                                    "MATCHED {:.0}% · {} SAVED",
                                    confidence.clamp(0.0, 1.0) * 100.0,
                                    self.wear_memory.profile_count()
                                )
                            } else {
                                format!("{} SAVED", self.wear_memory.profile_count())
                            };
                            ui.label(num(&memory_status));
                            ui.add_space(SP2);
                            ui.label(label("Saved states · for the current headset and image settings"));
                            for (index, (id, open, closed)) in self.wear_memory.profiles().into_iter().enumerate() {
                                if !self.wear_thumbnails.iter().any(|(key, _)| *key == id) {
                                    if let Some(pixels) = self.wear_memory.thumbnail(id) {
                                        let textures = std::array::from_fn(|eye| ui.ctx().load_texture(
                                            format!("wear-{id}-{eye}"),
                                            egui::ColorImage::from_gray([24, 24], &pixels[eye]),
                                            egui::TextureOptions::LINEAR));
                                        self.wear_thumbnails.push((id, textures));
                                    }
                                }
                                ui.push_id(id, |ui| {
                                    if let Some((_, textures)) = self.wear_thumbnails.iter().find(|(key, _)| *key == id) {
                                        ui.horizontal(|ui| {
                                            for texture in textures {
                                                ui.image((texture.id(), vec2(64.0 * S, 64.0 * S)));
                                            }
                                        });
                                    }
                                    ui.label(label(&format!("State {} · L {:.3}–{:.3} · R {:.3}–{:.3}", index + 1, closed[0], open[0], closed[1], open[1])));
                                    ui.horizontal(|ui| {
                                        if ui.button("Try").clicked() {
                                            self.begin_wearing_memory_edit();
                                            if let Some(target) = self.wear_memory.trial(id) {
                                                self.pipeline.set_wearing_calibration_target(Some(target));
                                                self.wear_memory_draft.message = Some(("Trying saved state. Adjust without recovery ends the trial.".into(), ACCENT));
                                            }
                                        }
                                        if ui.button("Delete").clicked() {
                                            match self.wear_memory.remove(id) {
                                                Ok(()) => {
                                                    self.wear_thumbnails.retain(|(key, _)| *key != id);
                                                    self.pipeline.clear_wearing_calibration_target_immediately();
                                                },
                                                Err(error) => self.wear_memory_draft.message = Some((error, ERR)),
                                            }
                                        }
                                    });
                                });
                            }
                            if let Some((text, color)) = &self.wear_memory_draft.message {
                                ui.add_space(SP2);
                                ui.label(egui::RichText::new(text).size(10.0 * S).color(*color));
                            }
                        });
                    },
                );
                let outside = scrim.clicked()
                    && scrim
                        .interact_pointer_pos()
                        .is_some_and(|point| !panel.contains(point));
                close_button || outside
            })
            .inner;
        if closed || ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
            self.show_wear_memory_modal = false;
        }
    }

    fn eyelid_response_controls(&mut self, ui: &mut egui::Ui) {
        if self.eyelid_response_preview.is_none() {
            self.begin_eyelid_response_preview();
        }
        let Some((mut edit, mut link_eyes, message)) = self
            .eyelid_response_preview
            .as_ref()
            .map(|preview| (preview.edit, preview.link_eyes, preview.message.clone()))
        else {
            return;
        };
        let expression_live = *self.tele.expression_live.lock().unwrap();
        let stored_edit = edit;
        let rail_live = *self.tele.eyelid_live.lock().unwrap();
        // Render the actual coordinates applied by the emit worker. If the user
        // drags a handle, convert those coordinates back to uncorrected offsets
        // before releasing recovery, so the untouched eye does not jump.
        if edit.manual_range || self.pipeline.wearing_calibration_target().is_some() {
            let live = rail_live;
            edit.manual_range = true;
            for eye in 0..2 {
                let baseline = live[eye].effective_baseline;
                edit.open_point_offset[eye] = baseline - live[eye].effective_open_ref;
                edit.closed_point_depth[eye] = baseline - live[eye].effective_closed_ref;
            }
        }
        let shown_edit = edit;
        let sranipal_wide_axis = self.pipeline.selected_wide_source() == WideSource::Sranipal;

        let mut profile_changed = false;
        let mut reset_clicked = false;
        let mut closed_capture_request = None;
        let mut save_good_state = false;

        card().show(ui, |ui| {
            let total_width = ui.available_width();
            let column_gap = SP2;
            let right_width = 0.0;
            let left_width = total_width;
            ui.horizontal_top(|ui| {
                ui.spacing_mut().item_spacing.x = column_gap;
                ui.allocate_ui_with_layout(
                    vec2(left_width, 0.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                        ui.set_min_width(left_width);
                        ui.set_max_width(left_width);
            ui.label(h3("Live eyelid response"));
            ui.label(prose(
                "Set where the raw eyelid model becomes 100% open and 0% closed. Green shows normal openness and stops at the Wide boundary; orange shows ch2 EyeWide. Drag either blue threshold and watch the avatar response. Changes apply immediately and save per HMD.",
            ));
            ui.add_space(SP2);

            ui.horizontal(|ui| {
                profile_changed |= ui
                    .checkbox(&mut edit.manual_range, "Use manual open / closed points")
                    .on_hover_text(
                        "On: the two visible blue points directly define 100% open and 0% closed. Off: use recorded or adaptively learned endpoints instead.",
                    )
                    .changed();
                ui.add_space(SP2);
                let link_response = ui
                    .checkbox(&mut link_eyes, "Link left / right")
                    .on_hover_text("Dragging either eye applies the same relative point to both eyes.");
                if link_response.changed() && link_eyes {
                    let differs = (edit.open_point_offset[0] - edit.open_point_offset[1]).abs()
                        > 0.0001
                        || (edit.closed_point_depth[0] - edit.closed_point_depth[1]).abs()
                            > 0.0001
                        || (edit.close_depth_scale[0] - edit.close_depth_scale[1]).abs()
                            > 0.0001
                        || (edit.curve_mid_output[0] - edit.curve_mid_output[1]).abs()
                            > 0.0001
                        || (edit.wide_start[0] - edit.wide_start[1]).abs() > 0.0001
                        || (edit.wide_full[0] - edit.wide_full[1]).abs() > 0.0001
                        || (edit.squeeze_start[0] - edit.squeeze_start[1]).abs() > 0.0001
                        || (edit.squeeze_full[0] - edit.squeeze_full[1]).abs() > 0.0001;
                    edit.open_point_offset[1] = edit.open_point_offset[0];
                    edit.closed_point_depth[1] = edit.closed_point_depth[0];
                    edit.close_depth_scale[1] = edit.close_depth_scale[0];
                    edit.curve_mid_output[1] = edit.curve_mid_output[0];
                    edit.wide_start[1] = edit.wide_start[0];
                    edit.wide_full[1] = edit.wide_full[0];
                    edit.squeeze_start[1] = edit.squeeze_start[0];
                    edit.squeeze_full[1] = edit.squeeze_full[0];
                    profile_changed |= differs;
                }
            });
            if !edit.manual_range {
                ui.label(label(
                    "Recorded/adaptive endpoints are active; the blue open/closed points do not drive output.",
                ));
            }
            ui.add_space(SP2);

            let live = rail_live;
            let mut range_change = [(false, false, false, false); 2];
            ui.columns(2, |columns| {
                for (eye, name) in ["LEFT", "RIGHT"].into_iter().enumerate() {
                    eyelid_live_eye(&mut columns[eye], name, &live[eye]);
                    let wide_edit = sranipal_wide_axis.then(|| WideRailEdit {
                        input: expression_live[eye].wide_input,
                        output: expression_live[eye].wide_output,
                        start: &mut edit.wide_start[eye],
                        full: &mut edit.wide_full[eye],
                        entry_ref: live[eye].wide_entry_ref,
                        full_ref: live[eye].wide_full_ref,
                    });
                    range_change[eye] = eyelid_threshold_rail(
                        &mut columns[eye],
                        eye,
                        &live[eye],
                        &mut edit.open_point_offset[eye],
                        &mut edit.closed_point_depth[eye],
                        edit.manual_range,
                        wide_edit,
                        None,
                    );
                    let closed_value = live[eye].effective_closed_ref;
                    let open_value = live[eye].effective_open_ref;
                    if sranipal_wide_axis {
                        let (wide_start, wide_full) = wide_raw_range(
                            &live[eye],
                            edit.wide_start[eye],
                            edit.wide_full[eye],
                        );
                        columns[eye].label(num(&format!(
                            "OPEN {:.3} → {:.3}    WIDE {:.3} → {:.3}",
                            closed_value, open_value, wide_start, wide_full
                        )));
                    } else {
                        columns[eye].label(num(&format!(
                            "OPEN {:.3} → {:.3}",
                            closed_value, open_value
                        )));
                    }
                }
            });
            if range_change[0].0 || range_change[0].1 {
                if link_eyes {
                    edit.open_point_offset[1] = edit.open_point_offset[0];
                    edit.closed_point_depth[1] = edit.closed_point_depth[0];
                }
                profile_changed = true;
            } else if range_change[1].0 || range_change[1].1 {
                if link_eyes {
                    edit.open_point_offset[0] = edit.open_point_offset[1];
                    edit.closed_point_depth[0] = edit.closed_point_depth[1];
                }
                profile_changed = true;
            }

            if range_change[0].2 || range_change[0].3 {
                if link_eyes {
                    edit.wide_start[1] = edit.wide_start[0];
                    edit.wide_full[1] = edit.wide_full[0];
                }
                profile_changed = true;
            } else if range_change[1].2 || range_change[1].3 {
                if link_eyes {
                    edit.wide_start[0] = edit.wide_start[1];
                    edit.wide_full[0] = edit.wide_full[1];
                }
                profile_changed = true;
            }

            ui.add_space(SP2);
            for kind in ["WIDE", "SQUEEZE"] {
                if kind == "WIDE" && sranipal_wide_axis {
                    continue;
                }
                ui.add_space(SP2);
                ui.label(
                    egui::RichText::new(kind)
                        .monospace()
                        .strong()
                        .size(10.0 * S)
                        .color(TEXT1),
                );
                let mut rail_changed = [(false, false); 2];
                ui.columns(2, |columns| {
                    for eye in 0..2 {
                        let (input, output) = if kind == "WIDE" {
                            (
                                expression_live[eye].wide_input,
                                expression_live[eye].wide_output,
                            )
                        } else {
                            (
                                expression_live[eye].squeeze_input,
                                expression_live[eye].squeeze_output,
                            )
                        };
                        let (start, full) = if kind == "WIDE" {
                            (&mut edit.wide_start[eye], &mut edit.wide_full[eye])
                        } else {
                            (
                                &mut edit.squeeze_start[eye],
                                &mut edit.squeeze_full[eye],
                            )
                        };
                        rail_changed[eye] = expression_threshold_rail(
                            &mut columns[eye],
                            if kind == "WIDE" { "wide" } else { "squeeze" },
                            eye,
                            input,
                            output,
                            start,
                            full,
                        );
                        columns[eye].label(num(&format!("{:.2} → {:.2}", *start, *full)));
                    }
                });
                if rail_changed[0].0 || rail_changed[0].1 {
                    if link_eyes {
                        if kind == "WIDE" {
                            edit.wide_start[1] = edit.wide_start[0];
                            edit.wide_full[1] = edit.wide_full[0];
                        } else {
                            edit.squeeze_start[1] = edit.squeeze_start[0];
                            edit.squeeze_full[1] = edit.squeeze_full[0];
                        }
                    }
                    profile_changed = true;
                } else if rail_changed[1].0 || rail_changed[1].1 {
                    if link_eyes {
                        if kind == "WIDE" {
                            edit.wide_start[0] = edit.wide_start[1];
                            edit.wide_full[0] = edit.wide_full[1];
                        } else {
                            edit.squeeze_start[0] = edit.squeeze_start[1];
                            edit.squeeze_full[0] = edit.squeeze_full[1];
                        }
                    }
                    profile_changed = true;
                }
            }

            if !edit.manual_range {
                ui.add_space(SP2);
                ui.separator();
                ui.add_space(SP2);
                ui.label(
                    egui::RichText::new("Recorded full-close trim")
                        .strong()
                        .size(11.0 * S)
                        .color(TEXT1),
                );
                ui.label(label(
                    "Legacy mode only: trim the endpoint learned by a recording.",
                ));
                ui.columns(2, |columns| {
                    for (eye, name) in ["LEFT", "RIGHT"].into_iter().enumerate() {
                        let mut percent = edit.close_depth_scale[eye] * 100.0;
                        let response = columns[eye].add(
                            egui::Slider::new(
                                &mut percent,
                                EyelidResponseProfile::CLOSE_DEPTH_SCALE_MIN * 100.0
                                    ..=EyelidResponseProfile::CLOSE_DEPTH_SCALE_MAX * 100.0,
                            )
                            .step_by(1.0)
                            .suffix("%")
                            .text(name),
                        );
                        if response.changed() {
                            edit.close_depth_scale[eye] = percent / 100.0;
                            if link_eyes {
                                edit.close_depth_scale[1 - eye] = edit.close_depth_scale[eye];
                            }
                            profile_changed = true;
                        }
                    }
                });
                ui.add_space(SP2);
            }

            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            ui.label(
                egui::RichText::new("Eyelid response")
                    .strong()
                    .size(11.0 * S)
                    .color(TEXT1),
            );
            ui.label(label(
                "Close earlier  |  response at the 50% model point  |  keep open longer",
            ));
            ui.columns(2, |columns| {
                for (eye, name) in ["LEFT", "RIGHT"].into_iter().enumerate() {
                    let mut percent = edit.curve_mid_output[eye] * 100.0;
                    let response = columns[eye].add(
                        egui::Slider::new(
                            &mut percent,
                            EyelidResponseProfile::CURVE_MID_OUTPUT_MIN * 100.0
                                ..=EyelidResponseProfile::CURVE_MID_OUTPUT_MAX * 100.0,
                        )
                        .step_by(1.0)
                        .suffix("%")
                        .text(name),
                    );
                    if response.changed() {
                        edit.curve_mid_output[eye] = percent / 100.0;
                        if link_eyes {
                            edit.curve_mid_output[1 - eye] = edit.curve_mid_output[eye];
                        }
                        profile_changed = true;
                    }
                }
            });

            ui.add_space(SP2);
            egui::CollapsingHeader::new("Advanced blink timing")
                .id_salt("advanced_blink_timing")
                .default_open(false)
                .show(ui, |ui| {
                    ui.add_space(SP2);
                    ui.label(
                        egui::RichText::new("Minimum closing time")
                            .strong()
                            .size(11.0 * S)
                            .color(TEXT1),
                    );
                    ui.label(label(
                        "Adds a lower bound to close duration. 0 ms is instant/off.",
                    ));
                    profile_changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut edit.blink_close_ms,
                                EyelidResponseProfile::BLINK_CLOSE_MS_MIN
                                    ..=EyelidResponseProfile::BLINK_CLOSE_MS_MAX,
                            )
                            .step_by(1.0)
                            .suffix(" ms"),
                        )
                        .changed();

                    ui.add_space(SP2);
                    ui.label(
                        egui::RichText::new("Fast-blink snap timing")
                            .strong()
                            .size(11.0 * S)
                            .color(TEXT1),
                    );
                    ui.label(label("Close later  |  pending gate  |  close earlier"));
                    let mut snap_gate_percent = edit.snap_gate_open * 100.0;
                    let snap_gate_response = ui
                        .add(
                            egui::Slider::new(
                                &mut snap_gate_percent,
                                EyelidResponseProfile::SNAP_GATE_OPEN_MIN * 100.0
                                    ..=EyelidResponseProfile::SNAP_GATE_OPEN_MAX * 100.0,
                            )
                            .step_by(1.0)
                            .suffix("%"),
                        )
                        .on_hover_text(
                            "A Tobii Disable may force 0% only after adjusted openness drops below this level.",
                        );
                    if snap_gate_response.changed() {
                        edit.snap_gate_open = snap_gate_percent / 100.0;
                        profile_changed = true;
                    }
                    ui.add_space(SP2);
                });

            ui.add_space(SP2);
            ui.horizontal(|ui| {
                reset_clicked = ui.button("Reset to standard").clicked();
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new("LIVE · AUTO-SAVE")
                            .monospace()
                            .strong()
                            .size(9.0 * S)
                            .color(OK),
                    );
                });
            });
            if let Some((text, color)) = &message {
                ui.label(
                    egui::RichText::new(text)
                        .monospace()
                        .size(10.0 * S)
                        .color(*color),
                );
            }
                    },
                );
                if right_width > 0.0 {
                ui.allocate_ui_with_layout(
                    vec2(right_width, 0.0),
                    egui::Layout::top_down(egui::Align::Min),
                    |ui| {
                    ui.set_min_width(right_width);
                    ui.set_max_width(right_width);
                    ui.label(h3("Wearing memory"));
                    ui.label(prose(
                        "Recenter and tune the live bars first. Nothing is learned until you confirm that the complete response is correct.",
                    ));
                    ui.add_space(SP2);

                    let status = |ui: &mut egui::Ui, name: &str, ready: bool| {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(if ready { "SET" } else { "CHECK" })
                                    .monospace()
                                    .strong()
                                    .size(9.0 * S)
                                    .color(if ready { OK } else { TEXT3 }),
                            );
                            ui.label(egui::RichText::new(name).size(10.0 * S).color(TEXT2));
                        });
                    };
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(if self.wear_memory_draft.recentered {
                                "READY"
                            } else {
                                "NEEDED"
                            })
                            .monospace()
                            .strong()
                            .size(9.0 * S)
                            .color(if self.wear_memory_draft.recentered { OK } else { WARN }),
                        );
                        ui.label(egui::RichText::new("Open reference").size(10.0 * S).color(TEXT2));
                    });
                    status(ui, "Left 0% point", self.wear_memory_draft.closed_set[0]);
                    status(ui, "Right 0% point", self.wear_memory_draft.closed_set[1]);
                    status(ui, "Wide neutral (optional)", self.wear_memory_draft.wide_neutral_set);

                    ui.add_space(SP3);
                    ui.label(
                        egui::RichText::new("Set a 0% point")
                            .strong()
                            .size(11.0 * S)
                            .color(TEXT1),
                    );
                    ui.label(label(
                        "Close the selected eye and keep it closed through the two tones. The stable raw green value becomes that eye's 0% point.",
                    ));
                    ui.add_space(SP2);
                    let capture_busy = self.wear_closed_capture.is_some();
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                self.wear_memory_draft.recentered && !capture_busy,
                                egui::Button::new("Set L closed"),
                            )
                            .clicked()
                        {
                            closed_capture_request = Some(0);
                        }
                        if ui
                            .add_enabled(
                                self.wear_memory_draft.recentered && !capture_busy,
                                egui::Button::new("Set R closed"),
                            )
                            .clicked()
                        {
                            closed_capture_request = Some(1);
                        }
                    });
                    if let Some(capture) = self.wear_closed_capture.as_ref() {
                        let elapsed = Instant::now().saturating_duration_since(capture.started);
                        let remaining = Duration::from_millis(1400).saturating_sub(elapsed);
                        ui.label(num(&format!(
                            "{} eye · {:.1}s",
                            if capture.eye == 0 { "LEFT" } else { "RIGHT" },
                            remaining.as_secs_f32()
                        )));
                    } else if !self.wear_memory_draft.recentered {
                        ui.label(label("Recenter first while looking straight ahead."));
                    }

                    ui.add_space(SP3);
                    ui.separator();
                    ui.add_space(SP3);
                    ui.label(
                        egui::RichText::new("Everything works correctly?")
                            .strong()
                            .size(11.0 * S)
                            .color(TEXT1),
                    );
                    ui.label(label(
                        "Test open, closed and normal blinks. This is the only action that saves an appearance for automatic recovery.",
                    ));
                    ui.add_space(SP2);
                    let save_ready = self.wear_memory_draft.recentered
                        && !capture_busy
                        && !self.wear_memory.capture_pending();
                    let save = egui::Button::new(
                        egui::RichText::new("Save current good state")
                            .monospace()
                            .strong()
                            .color(BG),
                    )
                    .fill(ACCENT);
                    if ui
                        .add_enabled_ui(save_ready, |ui| {
                            ui.add_sized([right_width, 34.0 * S], save)
                        })
                        .inner
                        .clicked()
                    {
                        save_good_state = true;
                    }
                    if self.wear_memory.capture_pending() {
                        ui.label(num("Saving neutral appearance · keep both eyes relaxed"));
                    }
                    ui.add_space(SP2);
                    let memory_status = if self.wear_memory_matching_suspended {
                        format!(
                            "WAITING FOR CONFIRMATION · {} SAVED",
                            self.wear_memory.profile_count()
                        )
                    } else if let Some(confidence) = self.wear_memory.active_confidence() {
                        format!(
                            "MATCHED {:.0}% · {} SAVED",
                            confidence.clamp(0.0, 1.0) * 100.0,
                            self.wear_memory.profile_count()
                        )
                    } else {
                        format!("{} SAVED", self.wear_memory.profile_count())
                    };
                    ui.label(num(&memory_status));
                    if let Some((text, color)) = &self.wear_memory_draft.message {
                        ui.add_space(SP2);
                        ui.label(
                            egui::RichText::new(text)
                                .size(10.0 * S)
                                .color(*color),
                        );
                    }
                    },
                );
                }
            });
        });

        if let Some(eye) = closed_capture_request {
            self.begin_closed_point_capture(eye);
        }
        if save_good_state {
            self.begin_wear_memory_capture(WearCaptureReason::GoodState);
            self.wear_memory_draft.message = Some((
                "Keep both eyes relaxed and look straight ahead for one second.".into(),
                ACCENT,
            ));
            self.recording_audio
                .cue(RecordingCue::Prepare, self.config.ui.recording_audio_cues);
        }

        let endpoints_changed = edit.manual_range != shown_edit.manual_range
            || edit.open_point_offset != shown_edit.open_point_offset
            || edit.closed_point_depth != shown_edit.closed_point_depth;
        if !endpoints_changed {
            // Expression/curve edits must not persist recovered endpoint values.
            edit.manual_range = stored_edit.manual_range;
            edit.open_point_offset = stored_edit.open_point_offset;
            edit.closed_point_depth = stored_edit.closed_point_depth;
        }
        if let Some(preview) = self.eyelid_response_preview.as_mut() {
            preview.link_eyes = link_eyes;
            if profile_changed {
                preview.edit = edit;
                preview.dirty = true;
                preview.message = None;
            }
        }
        if profile_changed {
            if endpoints_changed {
                if self.wear_response_before_edit.is_none() {
                    self.wear_response_before_edit = Some(stored_edit);
                }
                if self.wear_baseline_before_edit.is_none() {
                    self.wear_baseline_before_edit = Some(rail_live.map(|d| d.calibrated_baseline));
                }
                self.wear_memory_matching_suspended = true;
                self.wear_memory.disable();
                self.wear_closed_capture = None;
                self.pipeline
                    .edit_manual_endpoints(rail_live.map(|d| d.effective_baseline), edit);
            } else {
                self.set_live_eyelid_response(edit);
            }
        }
        if reset_clicked {
            self.begin_wearing_memory_edit();
            self.reset_eyelid_response_preview();
        }

        // Do not rewrite the TOML at display refresh rate while a slider is moving.
        // The runtime still receives every intermediate value above; persistence
        // happens once when the pointer gesture ends (or immediately for keyboard
        // and checkbox changes).
        let save_now = !ui.input(|input| input.pointer.primary_down())
            && self
                .eyelid_response_preview
                .as_ref()
                .is_some_and(|preview| preview.dirty);
        if save_now {
            self.save_eyelid_response_preview();
        }
    }

    fn geometry_rest_guide(
        &mut self,
        ui: &mut egui::Ui,
        instruction: &str,
        remaining_s: f32,
        overall: f32,
        awaiting_confirmation: bool,
    ) {
        ui.label(
            egui::RichText::new(if awaiting_confirmation {
                "READ THIS STEP"
            } else {
                "GET READY"
            })
            .monospace()
            .size(14.0 * S)
            .strong()
            .color(ACCENT),
        );
        ui.label(label(instruction));
        if awaiting_confirmation {
            ui.label(num(
                "Take your time. The timer and camera recording are stopped until Continue.",
            ));
            ui.add_space(SP2);
            if ui
                .add_sized([210.0 * S, 32.0 * S], egui::Button::new("Continue (Space)"))
                .clicked()
            {
                self.geometry_capture.continue_step();
            }
            ui.add(egui::ProgressBar::new(overall).show_percentage());
        } else {
            ui.label(num(&format!(
                "Get into position - recording starts in {remaining_s:.1}s"
            )));
            ui.add(egui::ProgressBar::new(overall).show_percentage());
        }
    }

    fn eyelid_endpoint_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let descriptor = SessionKind::EyelidEndpoints.descriptor();
        let capture_status = if self.calibration_capture_purpose
            == Some(CalibrationCapturePurpose::EyelidEndpoints)
        {
            self.geometry_capture.status()
        } else {
            GeometryCaptureStatus::Idle
        };
        let fit_status = self.endpoint_fitter.status();
        let (ready, ready_detail) = self.endpoint_capture_ready();
        let saved_path = self.geometry_recording_path.clone();
        let save_running = self.geometry_recording_export_job.is_some();
        let save_attempted = self.geometry_recording_export_attempted;
        let current = *self.tele.calibration.lock().unwrap();

        ui.set_width(width);
        calibration_detail_intro(ui, descriptor);
        ui.label(prose(
            "Each eye is checked independently. Move into each pose during the first part; only \
             the final steady second is scored. Half-open is used only as a consistency check.",
        ))
        .on_hover_text("Half-open is not forced to a 50% model output.");
        ui.add_space(SP2);

        match capture_status {
            GeometryCaptureStatus::Rest {
                instruction,
                remaining_s,
                overall,
                awaiting_confirmation,
                ..
            } => {
                self.geometry_rest_guide(
                    ui,
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                );
                if ui.button("Cancel endpoint recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Capture {
                instruction,
                remaining_s,
                overall,
                samples,
                target_open,
                stereo_stalled,
                ..
            } => {
                ui.label(
                    egui::RichText::new("RECORDING")
                        .monospace()
                        .strong()
                        .color(if stereo_stalled { WARN } else { ACCENT }),
                );
                ui.label(label(instruction));
                if let Some(target) = target_open {
                    ui.add(
                        egui::ProgressBar::new(target)
                            .text(format!("instruction target {:.0}% open", target * 100.0)),
                    );
                }
                ui.add(
                    egui::ProgressBar::new(overall)
                        .text(format!("{remaining_s:.1}s    {samples} stereo frames")),
                );
                if stereo_stalled {
                    ui.label(
                        egui::RichText::new("WAITING FOR FRESH STEREO FRAMES")
                            .monospace()
                            .color(ERR),
                    );
                }
                if ui.button("Cancel endpoint recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Done {
                train_samples,
                holdout_samples,
            } => {
                ui.label(
                    egui::RichText::new("ENDPOINT RECORDING COMPLETE")
                        .monospace()
                        .strong()
                        .color(OK),
                );
                ui.label(num(&format!(
                    "{train_samples} train + {holdout_samples} untouched holdout frames"
                )));
                if let Some(path) = &saved_path {
                    ui.label(num(&format!("Saved: {}", path.display())));
                } else if save_running {
                    ui.label(label("Saving the biometric ZIP before analysis..."));
                } else if save_attempted {
                    ui.label(
                        egui::RichText::new("Automatic ZIP save failed. Retry before analysis.")
                            .monospace()
                            .color(ERR),
                    );
                    if ui.button("Retry ZIP save").clicked() {
                        self.export_geometry_recording();
                    }
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            saved_path.is_some() && !save_running,
                            egui::Button::new("Analyze endpoints"),
                        )
                        .clicked()
                    {
                        self.start_endpoint_fit();
                    }
                    if ui.button("Discard recording").clicked() {
                        self.geometry_capture.abort();
                        self.geometry_capture_baseline = None;
                        self.geometry_capture_filters = None;
                        self.photometric_capture_baseline = None;
                        self.calibration_capture_purpose = None;
                        self.reset_geometry_recording_export();
                    }
                });
            }
            GeometryCaptureStatus::Idle => {
                match fit_status {
                    EndpointFitStatus::Idle => {
                        ui.label(
                            egui::RichText::new(if ready { "OK" } else { "WAIT" })
                                .monospace()
                                .strong()
                                .color(if ready { OK } else { WARN }),
                        );
                        ui.label(label(&ready_detail));
                        if ui
                            .add_enabled(
                                ready && !self.geometry_evidence_locked(),
                                egui::Button::new("Record open / closed endpoints"),
                            )
                            .clicked()
                        {
                            self.start_eyelid_endpoint_capture();
                        }
                    }
                    EndpointFitStatus::Running { completed, total } => {
                        ui.label(
                            egui::RichText::new("ANALYZING ENDPOINTS")
                                .monospace()
                                .strong()
                                .color(ACCENT),
                        );
                        ui.add(
                            egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                                .text(format!("{completed}/{total} labelled frames")),
                        );
                        if ui.button("Cancel analysis").clicked() {
                            self.endpoint_fitter.cancel();
                        }
                    }
                    EndpointFitStatus::Done { result } => {
                        ui.label(
                            egui::RichText::new(if result.any_accepted() {
                                "ENDPOINT HOLDOUT PASS"
                            } else {
                                "KEEP CURRENT ENDPOINTS"
                            })
                            .monospace()
                            .strong()
                            .color(if result.any_accepted() { OK } else { WARN }),
                        );
                        for (eye_index, eye_name) in ["LEFT", "RIGHT"].into_iter().enumerate() {
                            let eye = &result.eyes[eye_index];
                            ui.add_space(SP2);
                            ui.label(
                                egui::RichText::new(format!(
                                    "{eye_name}    {}",
                                    if eye.accepted { "PASS" } else { "RECORD AGAIN" }
                                ))
                                .monospace()
                                .strong()
                                .color(if eye.accepted {
                                    OK
                                } else {
                                    ERR
                                }),
                            );
                            if let Some(candidate) = eye.proposed {
                                ui.label(num(&format!(
                                    "raw open {:.3}    close ref {:.3}    depth {:.3}",
                                    candidate.baseline, candidate.closed_ref, candidate.blink_depth
                                )));
                            }
                            ui.label(num(&format!(
                                        "holdout open/half/closed {:.2}/{:.2}/{:.2}    current endpoint error {:.3} -> candidate {:.3}",
                                        eye.holdout_candidate.open_median,
                                        eye.holdout_candidate.half_median,
                                        eye.holdout_candidate.closed_median,
                                        eye.holdout_current.endpoint_error,
                                        eye.holdout_candidate.endpoint_error,
                                    )));
                            for rejection in &eye.rejections {
                                ui.label(
                                    egui::RichText::new(rejection.user_message())
                                        .monospace()
                                        .size(10.0 * S)
                                        .color(ERR),
                                );
                            }
                            for note in &eye.notes {
                                ui.label(
                                    egui::RichText::new(note)
                                        .monospace()
                                        .size(10.0 * S)
                                        .color(WARN),
                                );
                            }
                        }
                        ui.add_space(SP2);
                        ui.horizontal(|ui| {
                            if ui
                                .add_enabled(
                                    result.any_accepted(),
                                    egui::Button::new(match result.accepted_eyes() {
                                        [true, true] => "Apply both eyes",
                                        [true, false] => "Apply left eye only",
                                        [false, true] => "Apply right eye only",
                                        [false, false] => "Apply",
                                    }),
                                )
                                .clicked()
                            {
                                self.apply_endpoint_result(&result);
                            }
                            if ui.button("Record again").clicked() {
                                self.endpoint_fitter.clear_finished();
                                self.start_eyelid_endpoint_capture();
                            }
                            if ui.button("Discard result").clicked() {
                                self.endpoint_fitter.clear_finished();
                                self.calibration_capture_purpose = None;
                            }
                        });
                    }
                    EndpointFitStatus::Failed { message } => {
                        ui.label(
                            egui::RichText::new("ENDPOINT ANALYSIS FAILED SAFELY")
                                .monospace()
                                .strong()
                                .color(ERR),
                        );
                        ui.label(label(&message));
                        if ui.button("Record again").clicked() {
                            self.endpoint_fitter.clear_finished();
                            self.start_eyelid_endpoint_capture();
                        }
                    }
                    EndpointFitStatus::Cancelled => {
                        ui.label(label(
                            "Analysis cancelled; live endpoints were not changed.",
                        ));
                        if ui.button("Record again").clicked() {
                            self.endpoint_fitter.clear_finished();
                            self.start_eyelid_endpoint_capture();
                        }
                    }
                }
            }
        }

        if let Some(store) = current {
            let locked = [store.left.endpoint_locked, store.right.endpoint_locked];
            if locked.iter().any(|locked| *locked) {
                ui.add_space(SP2);
                ui.label(
                    egui::RichText::new(format!(
                        "Explicit endpoint lock    L {}    R {}",
                        if locked[0] { "ON" } else { "adaptive" },
                        if locked[1] { "ON" } else { "adaptive" },
                    ))
                    .monospace()
                    .color(OK),
                );
                self.calibration_removal_controls(
                    ui,
                    SessionKind::EyelidEndpoints,
                    "Remove open / closed calibration",
                );
            }
        }
    }

    fn wink_response_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let descriptor = SessionKind::Winks.descriptor();
        let capture_status =
            if self.calibration_capture_purpose == Some(CalibrationCapturePurpose::Winks) {
                self.geometry_capture.status()
            } else {
                GeometryCaptureStatus::Idle
            };
        let fit_status = self.wink_fitter.status();
        let saved_path = self.geometry_recording_path.clone();
        let save_running = self.geometry_recording_export_job.is_some();
        let save_attempted = self.geometry_recording_export_attempted;
        let profile = self.config.wink_profile_for(&self.pipeline.device_key);
        let active = profile.eyes.iter().any(|eye| eye.enabled);
        let (ready, ready_detail) = self.endpoint_capture_ready();

        ui.set_width(width);
        calibration_detail_intro(ui, descriptor);
        ui.label(prose(
            "Learns each eye separately, including its squeeze signature, and activates only while \
             the other eye stays open.",
        ))
        .on_hover_text(
            "Squeeze corroborates a wink but cannot create one by itself. This never moves the \
             partner eye or replaces the bilateral closed endpoints.",
        );
        if active {
            let state = |eye: &crate::core::types::WinkEyeProfile| match (
                eye.enabled,
                eye.floor_enabled,
                eye.squeeze_enabled,
            ) {
                (true, true, true) => "FLOOR ON + SIGNATURE",
                (true, true, false) => "FLOOR ON",
                (true, false, true) => "SIGNATURE ONLY",
                (true, false, false) => "ON",
                _ => "off",
            };
            ui.label(
                egui::RichText::new(format!(
                    "ACTIVE    L {} depth {:.3}    R {} depth {:.3}",
                    state(&profile.eyes[0]),
                    profile.eyes[0].wink_depth,
                    state(&profile.eyes[1]),
                    profile.eyes[1].wink_depth,
                ))
                .monospace()
                .color(OK),
            );
            self.calibration_removal_controls(ui, SessionKind::Winks, "Remove wink calibration");
        }
        ui.add_space(SP2);

        match capture_status {
            GeometryCaptureStatus::Rest {
                instruction,
                remaining_s,
                overall,
                awaiting_confirmation,
                ..
            } => {
                self.geometry_rest_guide(
                    ui,
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                );
                if ui.button("Cancel wink recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Capture {
                instruction,
                remaining_s,
                overall,
                samples,
                stereo_stalled,
                ..
            } => {
                ui.label(
                    egui::RichText::new("RECORDING")
                        .monospace()
                        .strong()
                        .color(if stereo_stalled { WARN } else { ACCENT }),
                );
                ui.label(label(instruction));
                ui.add(
                    egui::ProgressBar::new(overall)
                        .text(format!("{remaining_s:.1}s    {samples} stereo frames")),
                );
                if stereo_stalled {
                    ui.label(
                        egui::RichText::new("WAITING FOR FRESH STEREO FRAMES")
                            .monospace()
                            .color(ERR),
                    );
                }
                if ui.button("Cancel wink recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Done {
                train_samples,
                holdout_samples,
            } => {
                ui.label(
                    egui::RichText::new("WINK RECORDING COMPLETE")
                        .monospace()
                        .strong()
                        .color(OK),
                );
                ui.label(num(&format!(
                    "{train_samples} train + {holdout_samples} untouched holdout frames"
                )));
                if let Some(path) = &saved_path {
                    ui.label(num(&format!("Saved: {}", path.display())));
                } else if save_running {
                    ui.label(label("Saving the biometric ZIP before analysis..."));
                } else if save_attempted {
                    ui.label(
                        egui::RichText::new("Automatic ZIP save failed. Retry before analysis.")
                            .monospace()
                            .color(ERR),
                    );
                    if ui.button("Retry ZIP save").clicked() {
                        self.export_geometry_recording();
                    }
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            saved_path.is_some() && !save_running,
                            egui::Button::new("Analyze winks"),
                        )
                        .clicked()
                    {
                        self.start_wink_fit();
                    }
                    if ui.button("Discard recording").clicked() {
                        self.geometry_capture.abort();
                        self.geometry_capture_baseline = None;
                        self.geometry_capture_filters = None;
                        self.photometric_capture_baseline = None;
                        self.calibration_capture_purpose = None;
                        self.reset_geometry_recording_export();
                    }
                });
            }
            GeometryCaptureStatus::Idle => match fit_status {
                WinkFitStatus::Idle => {
                    let endpoints = *self.tele.calibration.lock().unwrap();
                    let endpoints_locked = endpoints.is_some_and(|store| {
                        store.left.endpoint_locked && store.right.endpoint_locked
                    });
                    let can_record = ready && endpoints_locked && !self.geometry_evidence_locked();
                    ui.label(
                        egui::RichText::new(if can_record { "OK" } else { "WAIT" })
                            .monospace()
                            .strong()
                            .color(if can_record { OK } else { WARN }),
                    );
                    ui.label(label(if endpoints_locked {
                                    &ready_detail
                                } else {
                                    "Apply Open / closed endpoints first; wink depth is relative to those locked endpoints."
                                }));
                    if ui
                        .add_enabled(can_record, egui::Button::new("Record left / right winks"))
                        .clicked()
                    {
                        self.start_wink_capture();
                    }
                }
                WinkFitStatus::Running { completed, total } => {
                    ui.label(
                        egui::RichText::new("ANALYZING WINK RESPONSE")
                            .monospace()
                            .strong()
                            .color(ACCENT),
                    );
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} labelled frames")),
                    );
                    if ui.button("Cancel analysis").clicked() {
                        self.wink_fitter.cancel();
                    }
                }
                WinkFitStatus::Done { result } => {
                    let ready_eyes = [
                        result.eyes[0].outcome.is_committable(),
                        result.eyes[1].outcome.is_committable(),
                    ];
                    let any_ready = result.eyes.iter().any(|eye| eye.outcome.is_committable());
                    let all_already_complete = result
                        .eyes
                        .iter()
                        .all(|eye| matches!(eye.outcome, EyeOutcome::NoChangeNeeded));
                    let (heading, heading_color) = if any_ready {
                        ("WINK RESPONSE READY", OK)
                    } else if all_already_complete {
                        ("WINK RESPONSE ALREADY COMPLETE", OK)
                    } else {
                        ("WINK RECORD AGAIN", WARN)
                    };
                    ui.label(
                        egui::RichText::new(heading)
                            .monospace()
                            .strong()
                            .color(heading_color),
                    );
                    for (eye_index, eye_name) in ["LEFT", "RIGHT"].into_iter().enumerate() {
                        let eye = &result.eyes[eye_index];
                        let (status, status_color) = match eye.outcome {
                            EyeOutcome::ApplyFloor if eye.proposed.squeeze_enabled => {
                                ("FLOOR + SIGNATURE READY", OK)
                            }
                            EyeOutcome::ApplyFloor => ("FLOOR READY", OK),
                            EyeOutcome::ApplySignatureOnly => {
                                ("SIGNATURE READY (NO FLOOR CHANGE)", OK)
                            }
                            EyeOutcome::NoChangeNeeded => ("ALREADY REACHES 0%", OK),
                            EyeOutcome::RecordAgain => ("RECORD AGAIN", ERR),
                        };
                        ui.add_space(SP2);
                        ui.label(
                            egui::RichText::new(format!("{eye_name}    {status}"))
                                .monospace()
                                .strong()
                                .color(status_color),
                        );
                        ui.label(num(&format!(
                                "wink depth {:.3}    holdout raw {:.3}    partner open median/p10 {:.2}/{:.2}",
                                eye.proposed.wink_depth,
                                eye.holdout_wink.median,
                                eye.partner_open_holdout,
                                eye.partner_open_holdout_p10,
                            )));
                        ui.label(num(&format!(
                            "holdout predicted current {:.3} -> candidate {:.3}",
                            eye.proposed.holdout_before, eye.proposed.holdout_after,
                        )));
                        ui.label(num(&format!(
                            "squeeze {}    train delta {:.3}    holdout margin {:.3}",
                            if eye.proposed.squeeze_enabled {
                                "corroboration on"
                            } else {
                                "fallback"
                            },
                            eye.squeeze_train_delta,
                            eye.squeeze_holdout_margin,
                        )));
                        for rejection in &eye.rejections {
                            ui.label(
                                egui::RichText::new(rejection)
                                    .monospace()
                                    .size(10.0 * S)
                                    .color(ERR),
                            );
                        }
                        for note in &eye.notes {
                            ui.label(
                                egui::RichText::new(note)
                                    .monospace()
                                    .size(10.0 * S)
                                    .color(WARN),
                            );
                        }
                    }
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(
                                any_ready,
                                egui::Button::new(match ready_eyes {
                                    [true, true] => "Apply both eyes",
                                    [true, false] => "Apply left eye only",
                                    [false, true] => "Apply right eye only",
                                    [false, false] => "Apply",
                                }),
                            )
                            .clicked()
                        {
                            self.apply_wink_result(&result);
                        }
                        if ui.button("Record again").clicked() {
                            self.wink_fitter.clear_finished();
                            self.start_wink_capture();
                        }
                        if ui.button("Discard result").clicked() {
                            self.wink_fitter.clear_finished();
                            self.calibration_capture_purpose = None;
                        }
                    });
                }
                WinkFitStatus::Failed { message } => {
                    ui.label(
                        egui::RichText::new("WINK ANALYSIS FAILED SAFELY")
                            .monospace()
                            .strong()
                            .color(ERR),
                    );
                    ui.label(label(&message));
                    if ui.button("Record again").clicked() {
                        self.wink_fitter.clear_finished();
                        self.start_wink_capture();
                    }
                }
                WinkFitStatus::Cancelled => {
                    ui.label(label(
                        "Analysis cancelled; the live wink response was not changed.",
                    ));
                    if ui.button("Record again").clicked() {
                        self.wink_fitter.clear_finished();
                        self.start_wink_capture();
                    }
                }
            },
        }
    }

    fn natural_blink_timing_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let descriptor = SessionKind::NaturalBlinks.descriptor();
        let capture_status =
            if self.calibration_capture_purpose == Some(CalibrationCapturePurpose::NaturalBlinks) {
                self.geometry_capture.status()
            } else {
                GeometryCaptureStatus::Idle
            };
        let fit_status = self.blink_timing_fitter.status();
        let saved_path = self.geometry_recording_path.clone();
        let save_running = self.geometry_recording_export_job.is_some();
        let save_attempted = self.geometry_recording_export_attempted;
        let profile = self
            .config
            .blink_timing_profile_for(&self.pipeline.device_key);
        let (ready, ready_detail) = self.endpoint_capture_ready();

        ui.set_width(width);
        calibration_detail_intro(ui, descriptor);
        ui.label(prose(
            "Only fast two-eye blinks are affected; slow closes and winks are unchanged.",
        ))
        .on_hover_text("Reopening is held briefly until the emitted eyelids visibly reach zero.");
        ui.label(
            egui::RichText::new(if profile.calibrated_unix != 0 {
                format!(
                    "CALIBRATED    {}    visible bottom {:.0} ms",
                    if profile.enabled {
                        "enabled"
                    } else {
                        "disabled"
                    },
                    profile.min_closed_ms
                )
            } else if profile.enabled {
                format!("BUILT-IN    visible bottom {:.0} ms", profile.min_closed_ms)
            } else {
                "DISABLED".to_owned()
            })
            .monospace()
            .color(if profile.enabled { OK } else { WARN }),
        );
        if ui
            .button(if profile.enabled {
                "Disable blink bottom hold"
            } else {
                "Enable blink bottom hold"
            })
            .clicked()
        {
            self.set_blink_timing_enabled(!profile.enabled);
        }
        if profile.calibrated_unix != 0 {
            self.calibration_removal_controls(
                ui,
                SessionKind::NaturalBlinks,
                "Remove fitted blink timing",
            );
        }
        ui.add_space(SP2);

        match capture_status {
            GeometryCaptureStatus::Rest {
                instruction,
                remaining_s,
                overall,
                awaiting_confirmation,
                ..
            } => {
                self.geometry_rest_guide(
                    ui,
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                );
                if ui.button("Cancel natural-blink recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Capture {
                instruction,
                remaining_s,
                overall,
                samples,
                stereo_stalled,
                ..
            } => {
                ui.label(
                    egui::RichText::new("RECORDING AT 60 HZ")
                        .monospace()
                        .strong()
                        .color(if stereo_stalled { WARN } else { ACCENT }),
                );
                ui.label(label(instruction));
                ui.add(
                    egui::ProgressBar::new(overall)
                        .text(format!("{remaining_s:.1}s    {samples} stereo frames")),
                );
                if stereo_stalled {
                    ui.label(
                        egui::RichText::new("WAITING FOR FRESH STEREO FRAMES")
                            .monospace()
                            .color(ERR),
                    );
                }
                if ui.button("Cancel natural-blink recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Done {
                train_samples,
                holdout_samples,
            } => {
                ui.label(
                    egui::RichText::new("NATURAL-BLINK RECORDING COMPLETE")
                        .monospace()
                        .strong()
                        .color(OK),
                );
                ui.label(num(&format!(
                    "{train_samples} train + {holdout_samples} untouched holdout frames"
                )));
                if let Some(path) = &saved_path {
                    ui.label(num(&format!("Saved: {}", path.display())));
                } else if save_running {
                    ui.label(label("Saving the biometric ZIP before analysis..."));
                } else if save_attempted {
                    ui.label(
                        egui::RichText::new("Automatic ZIP save failed. Retry before analysis.")
                            .monospace()
                            .color(ERR),
                    );
                    if ui.button("Retry ZIP save").clicked() {
                        self.export_geometry_recording();
                    }
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            saved_path.is_some() && !save_running,
                            egui::Button::new("Analyze natural blinks"),
                        )
                        .clicked()
                    {
                        self.start_blink_timing_fit();
                    }
                    if ui.button("Discard recording").clicked() {
                        self.geometry_capture.abort();
                        self.geometry_capture_baseline = None;
                        self.geometry_capture_filters = None;
                        self.photometric_capture_baseline = None;
                        self.calibration_capture_purpose = None;
                        self.reset_geometry_recording_export();
                    }
                });
            }
            GeometryCaptureStatus::Idle => match fit_status {
                BlinkTimingFitStatus::Idle => {
                    let endpoints = *self.tele.calibration.lock().unwrap();
                    let endpoints_locked = endpoints.is_some_and(|store| {
                        store.left.endpoint_locked && store.right.endpoint_locked
                    });
                    let can_record = ready && endpoints_locked && !self.geometry_evidence_locked();
                    ui.label(
                        egui::RichText::new(if can_record { "OK" } else { "WAIT" })
                            .monospace()
                            .strong()
                            .color(if can_record { OK } else { WARN }),
                    );
                    ui.label(label(if endpoints_locked {
                        &ready_detail
                    } else {
                        "Apply Open / closed endpoints before validating blink timing."
                    }));
                    if ui
                        .add_enabled(can_record, egui::Button::new("Record natural blinks"))
                        .clicked()
                    {
                        self.start_natural_blink_capture();
                    }
                }
                BlinkTimingFitStatus::Running { completed, total } => {
                    ui.label(
                        egui::RichText::new("ANALYZING NATURAL BLINKS")
                            .monospace()
                            .strong()
                            .color(ACCENT),
                    );
                    ui.add(
                        egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                            .text(format!("{completed}/{total} frames")),
                    );
                    if ui.button("Cancel analysis").clicked() {
                        self.blink_timing_fitter.cancel();
                    }
                }
                BlinkTimingFitStatus::Done { result } => {
                    ui.label(
                        egui::RichText::new(if result.accepted {
                            "NATURAL-BLINK HOLDOUT PASS"
                        } else {
                            "KEEP CURRENT BLINK TIMING"
                        })
                        .monospace()
                        .strong()
                        .color(if result.accepted { OK } else { WARN }),
                    );
                    ui.label(num(&format!(
                        "train {} blinks, {:.0} ms median    holdout {} blinks, {:.0} ms median",
                        result.train.episodes.len(),
                        result.train.median_duration_ms,
                        result.holdout.episodes.len(),
                        result.holdout.median_duration_ms,
                    )));
                    ui.label(num(&format!(
                        "proposed visible bottom {:.0} ms",
                        result.profile.min_closed_ms
                    )));
                    for rejection in &result.rejections {
                        ui.label(
                            egui::RichText::new(rejection)
                                .monospace()
                                .size(10.0 * S)
                                .color(ERR),
                        );
                    }
                    for note in &result.notes {
                        ui.label(
                            egui::RichText::new(note)
                                .monospace()
                                .size(10.0 * S)
                                .color(WARN),
                        );
                    }
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(result.accepted, egui::Button::new("Apply blink timing"))
                            .clicked()
                        {
                            self.apply_blink_timing_result(&result);
                        }
                        if ui.button("Record again").clicked() {
                            self.blink_timing_fitter.clear_finished();
                            self.start_natural_blink_capture();
                        }
                        if ui.button("Discard result").clicked() {
                            self.blink_timing_fitter.clear_finished();
                            self.calibration_capture_purpose = None;
                        }
                    });
                }
                BlinkTimingFitStatus::Failed { message } => {
                    ui.label(
                        egui::RichText::new("BLINK ANALYSIS FAILED SAFELY")
                            .monospace()
                            .strong()
                            .color(ERR),
                    );
                    ui.label(label(&message));
                    if ui.button("Record again").clicked() {
                        self.blink_timing_fitter.clear_finished();
                        self.start_natural_blink_capture();
                    }
                }
                BlinkTimingFitStatus::Cancelled => {
                    ui.label(label(
                        "Analysis cancelled; live blink timing was not changed.",
                    ));
                    if ui.button("Record again").clicked() {
                        self.blink_timing_fitter.clear_finished();
                        self.start_natural_blink_capture();
                    }
                }
            },
        }
    }

    fn gaze_eyelid_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let descriptor = SessionKind::GazeDirections.descriptor();
        let capture_status = self.gaze_residual_capture.status();
        let fit_status = self.gaze_eyelid_fitter.status();
        let (ready, ready_detail) = self.endpoint_capture_ready();
        let active = self
            .config
            .gaze_eyelid_profile_for(&self.pipeline.device_key);

        ui.set_width(width);
        calibration_detail_intro(ui, descriptor);
        ui.label(prose(
                        "Follow nine targets with your eyes while keeping both eyelids comfortably open.",
                    ))
                    .on_hover_text(
                        "Untouched passes validate the correction. It can only lift repeatable open-eye droop, is capped at the relaxed baseline, and fades to zero at the closed endpoint.",
                    );
        ui.label(num(&format!(
            "Active profile    L {}    R {}",
            if active.eyes[0].enabled { "ON" } else { "off" },
            if active.eyes[1].enabled { "ON" } else { "off" }
        )));
        if active.calibrated_unix != 0 || active.eyes.iter().any(|eye| eye.enabled) {
            self.calibration_removal_controls(
                ui,
                SessionKind::GazeDirections,
                "Remove saved correction",
            );
        }
        ui.add_space(SP2);

        match fit_status {
            GazeEyelidFitStatus::Running { completed, total } => {
                ui.label(
                    egui::RichText::new("ANALYZING SAVED RECORDING")
                        .monospace()
                        .strong()
                        .color(ACCENT),
                );
                ui.add(
                    egui::ProgressBar::new(completed as f32 / total.max(1) as f32)
                        .text(format!("{completed}/{total} EyeNet replays")),
                );
                ui.label(num(
                                "Tracking stays live. Image geometry, photometric settings and gaze output are frozen inputs, not search parameters.",
                            ));
                if ui.button("Cancel analysis").clicked() {
                    self.gaze_eyelid_fitter.cancel();
                }
            }
            GazeEyelidFitStatus::Done { result } => {
                ui.label(
                    egui::RichText::new(if result.any_accepted() {
                        "HOLDOUT PASS"
                    } else {
                        "KEEP CURRENT EYELID BEHAVIOUR"
                    })
                    .monospace()
                    .strong()
                    .color(if result.any_accepted() { OK } else { WARN }),
                );
                for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                    let fit = &result.eyes[eye];
                    ui.label(num(&format!(
                                    "{name} {}    train max droop {:.3}    holdout {:.3} -> {:.3}    close-safety {} frames",
                                    if fit.accepted { "PASS" } else { "REJECT" },
                                    fit.max_train_droop,
                                    fit.holdout_before,
                                    fit.holdout_after,
                                    fit.safety_frames,
                                )));
                    for rejection in fit.rejections.iter().take(3) {
                        ui.label(
                            egui::RichText::new(format!("{name}: {rejection}"))
                                .monospace()
                                .size(10.0 * S)
                                .color(WARN),
                        );
                    }
                }
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(
                            result.any_accepted(),
                            egui::Button::new("Apply validated correction"),
                        )
                        .clicked()
                    {
                        self.apply_gaze_eyelid_result(&result);
                    }
                    if ui.button("Record again").clicked() {
                        self.start_gaze_residual_capture();
                    }
                    if ui.button("Clear result").clicked() {
                        self.gaze_eyelid_fitter.clear_finished();
                    }
                });
            }
            GazeEyelidFitStatus::Failed { message } => {
                ui.label(
                    egui::RichText::new("ANALYSIS FAILED SAFELY")
                        .monospace()
                        .strong()
                        .color(ERR),
                );
                ui.label(label(&message));
                if ui.button("Record again").clicked() {
                    self.start_gaze_residual_capture();
                }
            }
            GazeEyelidFitStatus::Cancelled => {
                ui.label(num("Analysis cancelled; no profile was changed."));
                if ui.button("Record again").clicked() {
                    self.start_gaze_residual_capture();
                }
            }
            GazeEyelidFitStatus::Idle => match capture_status {
                GazeResidualStatus::Idle => {
                    ui.label(num(&ready_detail));
                    ui.label(num(&format!(
                        "about {:.0}s; the biometric ZIP is saved automatically before analysis",
                        crate::gaze_residual_calib::total_seconds()
                    )));
                    if ui
                        .add_enabled(
                            ready && !self.geometry_evidence_locked(),
                            egui::Button::new("Start nine-direction recording"),
                        )
                        .clicked()
                    {
                        self.start_gaze_residual_capture();
                    }
                }
                GazeResidualStatus::Ready { instruction } => {
                    ui.label(
                        egui::RichText::new("READY - TIMER STOPPED")
                            .monospace()
                            .strong()
                            .color(ACCENT),
                    );
                    ui.label(label(&instruction));
                    if ui.button("Begin recording (Space)").clicked() {
                        self.begin_gaze_residual_capture();
                    }
                    if ui.button("Cancel").clicked() {
                        self.gaze_residual_capture.abort();
                        self.gaze_residual_snapshot = None;
                    }
                }
                GazeResidualStatus::Running {
                    progress,
                    remaining_s,
                    instruction,
                    holdout,
                    recording,
                    settling,
                    stereo_stalled,
                    samples_in_phase,
                    ..
                } => {
                    ui.label(
                        egui::RichText::new(if stereo_stalled {
                            "WAITING FOR FRESH STEREO FRAMES"
                        } else if !recording {
                            "PRACTICE - NOT RECORDED"
                        } else if settling {
                            "MOVE YOUR EYES TO THE TARGET"
                        } else {
                            "RECORDING"
                        })
                        .monospace()
                        .strong()
                        .color(if stereo_stalled { ERR } else { ACCENT }),
                    );
                    ui.label(label(&instruction));
                    ui.label(num(&format!(
                        "{}    {samples_in_phase} samples    {remaining_s:.1}s left",
                        if holdout {
                            "untouched holdout"
                        } else {
                            "training"
                        }
                    )));
                    ui.add(egui::ProgressBar::new(progress).show_percentage());
                    if ui.button("Cancel and discard recording").clicked() {
                        self.gaze_residual_capture.abort();
                        self.gaze_residual_snapshot = None;
                        self.gaze_residual_export_attempted = false;
                        self.gaze_residual_recording_path = None;
                    }
                }
                GazeResidualStatus::Done {
                    samples,
                    evidence_complete,
                    ..
                } => {
                    ui.label(
                        egui::RichText::new(if evidence_complete {
                            "RECORDING COMPLETE"
                        } else {
                            "RECORDING INCOMPLETE"
                        })
                        .monospace()
                        .strong()
                        .color(if evidence_complete { OK } else { ERR }),
                    );
                    ui.label(num(&format!("{samples} labelled stereo frames")));
                    if let Some(path) = &self.gaze_residual_recording_path {
                        ui.label(num(&format!("Saved: {}", path.display())));
                    } else if self.gaze_residual_export_job.is_some() {
                        ui.label(num("Saving biometric ZIP before analysis..."));
                    } else if self.gaze_residual_export_attempted {
                        ui.label(num(
                            "ZIP save failed; the only recording remains in memory.",
                        ));
                        if ui.button("Retry ZIP save").clicked() {
                            self.export_gaze_residual_recording();
                        }
                    }
                }
            },
        }
    }

    fn calibration(&mut self, ui: &mut egui::Ui) {
        let baselines = *self.tele.baselines.lock().unwrap();
        let (gutter, cw) = stage_metrics(ui.ctx());
        egui::ScrollArea::vertical().show(ui, |ui| {
        ui.horizontal_top(|ui| {
        ui.add_space(gutter);
        ui.vertical(|ui| {
        ui.set_width(cw);
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.add_space(SP2);
            ui.label(label("Look straight ahead, relaxed, then Recenter to re-learn each eye's open baseline."));
            ui.add_space(SP2);
            // Primary action: accent fill + dark text so it's clearly readable (the old
            // white-on-default-grey button was low-contrast).
            ui.horizontal(|ui| {
                let recenter = egui::Button::new(
                    egui::RichText::new("Recenter").monospace().size(13.0 * S).strong().color(BG),
                )
                .fill(ACCENT);
                let recenter_response = ui
                    .add_enabled_ui(!self.guided_capture_running(), |ui| {
                        ui.add_sized([150.0 * S, 32.0 * S], recenter)
                    })
                    .inner;
                if recenter_response.clicked() {
                    self.begin_wearing_memory_edit();
                    self.pipeline.recenter.store(true, Ordering::Relaxed);
                    self.wear_memory_draft = WearingMemoryDraft {
                        recentered: true,
                        message: Some((
                            "Open reference ready. Verify or set both 0% points, then confirm the complete state beside Live eyelid response."
                                .into(),
                            ACCENT,
                        )),
                        ..Default::default()
                    };
                }
                ui.add_space(SP2);
                let wide_neutral = egui::Button::new(
                    egui::RichText::new("Set Wide neutral")
                        .monospace()
                        .size(13.0 * S)
                        .strong(),
                );
                let wide_neutral_response = ui
                    .add_enabled_ui(!self.guided_capture_running(), |ui| {
                        ui.add_sized([170.0 * S, 32.0 * S], wide_neutral)
                    })
                    .inner
                    .on_hover_text(
                        "Keep both eyes comfortably open and relaxed. Relearns only the EyeWide neutral baseline; eyelid and brow calibration stay unchanged.",
                    );
                if wide_neutral_response.clicked() {
                    self.begin_wearing_memory_edit();
                    self.pipeline.recenter_wide_neutral();
                    self.wear_memory_draft.wide_neutral_set = true;
                    self.events.push((
                        now_hms(),
                        "Learning Wide neutral - keep both eyes relaxed for one second".into(),
                        ACCENT,
                    ));
                }
                ui.add_space(SP2);
                // Diagnostic recorder: while on, the emit thread writes raw +
                // every post-processing internal to a CSV in the app dir.
                let rec_on = self.pipeline.diag_rec.load(Ordering::Relaxed);
                let (rec_text, rec_fill) = if rec_on {
                    ("■ STOP", egui::Color32::from_rgb(0xd9, 0x53, 0x4f))
                } else {
                    ("● REC", egui::Color32::from_rgb(0x3a, 0x3f, 0x4a))
                };
                let rec = egui::Button::new(
                    egui::RichText::new(rec_text).monospace().size(13.0 * S).strong().color(
                        if rec_on { BG } else { egui::Color32::from_rgb(0xd9, 0x53, 0x4f) },
                    ),
                )
                .fill(rec_fill);
                let rec_response = ui
                    .add_enabled_ui(!self.guided_capture_running(), |ui| {
                        ui.add_sized([110.0 * S, 32.0 * S], rec)
                    })
                    .inner;
                if rec_response.clicked() {
                    self.pipeline.diag_rec.store(!rec_on, Ordering::Relaxed);
                }
            });
            ui.add_space(SP2);
            ui.label(num(&format!("baseline   L {:.3}    R {:.3}", baselines[0], baselines[1])));
            let memory_status = if !self.config.ui.wearing_memory_enabled {
                format!(
                    "wearing memory   off · {} saved",
                    self.wear_memory.profile_count()
                )
            } else if self.wear_memory_matching_suspended {
                format!(
                    "wearing memory   waiting for confirmation · {} saved",
                    self.wear_memory.profile_count()
                )
            } else if let Some(confidence) = self.wear_memory.active_confidence() {
                format!(
                    "wearing memory   matched {:.0}% · {} saved",
                    confidence * 100.0,
                    self.wear_memory.profile_count()
                )
            } else if self.wear_memory.profile_count() > 0 {
                format!(
                    "wearing memory   watching · {} saved",
                    self.wear_memory.profile_count()
                )
            } else {
                "wearing memory   no confirmed good states yet".into()
            };
            ui.label(num(&memory_status));
            ui.add_space(SP2);
            self.automatic_fit_recovery_control(ui);
            ui.add_space(SP2);
            ui.separator();
            ui.add_space(SP2);
            let supports_gaze = crate::config::supports_gaze_correction(&self.pipeline.device_key);
            let correction = *self.pipeline.gaze_correction.lock().unwrap();
            ui.horizontal(|ui| {
                if supports_gaze
                    && ui
                        .add_enabled(
                            !self.guided_capture_running(),
                            egui::Button::new("Gaze centre & movement range..."),
                        )
                        .clicked()
                {
                    self.show_gaze_modal = true;
                }
                if ui.button("Wearing memory...").clicked() {
                    self.show_wear_memory_modal = true;
                }
            });
            ui.horizontal(|ui| {
                if supports_gaze {
                    let range_x = gaze_range_percent(&correction, GazeRangeAxis::Horizontal);
                    let range_y = gaze_range_percent(&correction, GazeRangeAxis::Vertical);
                    let gaze_status = if correction.enabled {
                        if gaze_range_is_mixed(range_x) || gaze_range_is_mixed(range_y) {
                            format!(
                                "gaze per-eye L/R {:.0}/{:.0}% U/D {:.0}/{:.0}%",
                                range_x[0], range_x[1], range_y[0], range_y[1]
                            )
                        } else {
                            format!("gaze L/R {:.0}% U/D {:.0}%", range_x[0], range_y[0])
                        }
                    } else {
                        "gaze native range".to_owned()
                    };
                    ui.label(label(&gaze_status));
                    ui.add_space(SP2);
                }
                let memory_status = if !self.config.ui.wearing_memory_enabled {
                    format!("wearing off · {} saved", self.wear_memory.profile_count())
                } else if self.wear_memory_matching_suspended {
                    format!(
                        "wearing waiting for confirmation · {} saved",
                        self.wear_memory.profile_count()
                    )
                } else if let Some(confidence) = self.wear_memory.active_confidence() {
                    format!(
                        "wearing matched {:.0}% · {} saved",
                        confidence.clamp(0.0, 1.0) * 100.0,
                        self.wear_memory.profile_count()
                    )
                } else {
                    format!("wearing {} saved", self.wear_memory.profile_count())
                };
                ui.label(label(&memory_status));
            });
        });
        ui.add_space(SP3);

        // Primary, recording-free eyelid setup. The advanced capture workflows remain
        // available below for XR5 research and targeted diagnostics, but ordinary users
        // can now fit the model response directly from the live threshold rail.
        self.eyelid_response_controls(ui);
        ui.add_space(SP3);

        let calibration_recording = self.geometry_capture.is_running()
            || matches!(
                self.gaze_residual_capture.status(),
                GazeResidualStatus::Ready { .. } | GazeResidualStatus::Running { .. }
            );
        let calibration_analysing = self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.gaze_eyelid_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running();
        let calibration_result_ready =
            self.geometry_capture.is_done() || self.gaze_residual_capture.is_done();
        let calibration_title = if calibration_recording {
            "Advanced calibration    RECORDING"
        } else if calibration_analysing {
            "Advanced calibration    ANALYSING"
        } else if calibration_result_ready {
            "Advanced calibration    RESULT READY"
        } else {
            "Advanced calibration & recordings"
        };

        if shows_advanced_recording_calibration(&self.pipeline.device_key) {
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            egui::CollapsingHeader::new(h3(calibration_title))
                .id_salt("calibration_workflows_group")
                .default_open(false)
                .open(
                    (calibration_recording
                        || calibration_analysing
                        || calibration_result_ready)
                        .then_some(true),
                )
                .show(ui, |ui| {
                    let inner_width = cw - 2.0 * CARD_PAD;
                    ui.add_space(SP2);
                    let mut audio_cues = self.config.ui.recording_audio_cues;
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut audio_cues, "Recording notification sounds")
                            .on_hover_text(
                                "Short sounds mark prepare, sampling, holdout, completion, cancellation, and errors. No speech is used.",
                            )
                            .changed()
                        {
                            self.config.ui.recording_audio_cues = audio_cues;
                            let _ = self.config.save(&crate::config::config_path());
                        }
                        if ui.button("Test sound").clicked() {
                            self.recording_audio.cue(RecordingCue::Complete, true);
                        }
                    });
                    ui.add_space(SP3);
                    self.calibration_workflow_card(ui, inner_width);
                });
        });
        ui.add_space(SP3);
        }
        if self.pipeline.device_key == "pimax_xr5" {
            self.dream_air_wide_card(ui, cw);
            ui.add_space(SP3);
        }
        self.eyebrow_calib_card(ui, cw);
        ui.add_space(SP3);
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            egui::CollapsingHeader::new(h3("Tuning"))
                .id_salt("tuning_card")
                .default_open(false)
                .show(ui, |ui| {
            ui.add_space(SP2);
            let visual_range_active = self.pipeline.eyelid_response.lock().unwrap().manual_range;
            let mut t = self.pipeline.tuning.lock().unwrap();
            ui.spacing_mut().slider_width = 240.0 * S;
            let mut ch = false;
            ch |= ui.add(egui::Slider::new(&mut t.alpha_open, 0.05..=1.0).text("open speed")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.alpha_close, 0.05..=1.0).text("close speed")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.squeeze_deadzone, 0.0..=0.9).text("squeeze deadzone")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.squeeze_gain, 0.0..=1.0).text("squeeze gain")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.wide_gain, 0.0..=1.0).text("wide gain")).changed();
            ch |= ui
                .add_enabled(
                    !visual_range_active,
                    egui::Slider::new(&mut t.open_deadzone, 0.03..=0.20)
                        .text("legacy eye-open dead-zone"),
                )
                .changed();
            if visual_range_active {
                ui.label(label(
                    "Open / closed points are controlled by Live eyelid response above.",
                ));
            }
            ui.add_space(SP2);
            // Live A/B of the RE'd channels vs the legacy derivation.
            ch |= ui.checkbox(&mut t.native_squeeze, "Native squeeze (model ch3/ch4)").changed();
            ch |= ui.checkbox(&mut t.adaptive_kalman, "Adaptive Kalman openness").changed();
            ch |= ui
                .checkbox(&mut t.couple_eyes, "Couple eyes (shared baseline)")
                .on_hover_text(
                    "Moves both open baselines toward their mean. Adaptive blink-bound learning pauses while enabled.",
                )
                .changed();
            ch |= ui
                .checkbox(&mut t.continuous_calib, "Adaptive blink bounds")
                .on_hover_text(
                    "Use learned per-eye full-close bounds. The relaxed-open baseline is calibrated separately.",
                )
                .changed();
            // Robust fallback: plain per-eye ramp only (skips the curve equalizer + fast-
            // blink latch), like native SRanipal / BrokenEye — fixes "one eye breaks".
            ch |= ui.checkbox(&mut t.wide_requires_both, "Wide needs both eyes (symmetric)").changed();
            ch |= ui.checkbox(&mut t.gaze_yoke, "Gaze yoke (squint eye follows open eye)").changed();
            let snapshot = *t;
            drop(t);
            // Persist tuning so it survives a restart (it's a tiny file; OK to write on change).
            if ch {
                self.config.tuning = snapshot;
                let _ = self.config.save(&crate::config::config_path());
            }
            ui.add_space(SP2);
            if ui.button("Reset to defaults").clicked() {
                let def = crate::core::eye_state::Tuning::default();
                *self.pipeline.tuning.lock().unwrap() = def;
                self.config.tuning = def;
                let _ = self.config.save(&crate::config::config_path());
            }
                // One-click eyelid feel: native-style crisp close (simple mode + reachable
                // 0-point, no curve/latch = BrokenEye-stable, no one-eye breaks) with the
                // teleport-snap taken down a notch (close 0.85). Keeps squeeze/wide/gaze.
                });
        });
        });
        });
        });
    }

    /// VR4 and Varjo keep their frontal geometry frozen. This workflow searches only
    /// bounded brightness, contrast, weak flattening and low-frequency illumination
    /// differences, then admits a candidate only after untouched holdout validation.
    fn photometric_correction_card(&mut self, ui: &mut egui::Ui, width: f32) {
        let device = self.pipeline.device_key.clone();
        let capture_status = self.geometry_capture.status();
        let fit_status = self.photometric_fitter.status();
        let (ready, ready_detail) = self.photometric_capture_ready();

        ui.set_width(width);
        calibration_detail_intro(ui, SessionKind::Photometric.descriptor());
        ui.label(prose(
                    "VR4 / Varjo only. The candidate must improve untouched validation frames before it can be applied.",
                ))
                .on_hover_text(
                    "Crop, rotation, scale, distortion, native Tobii gaze and raw eye-image output stay unchanged. Weak results remain diagnostic only.",
                );
        if self.config.has_photometric_correction(&device) {
            let correction = self.config.photometric_correction_for(&device);
            ui.label(
                egui::RichText::new(if correction.is_identity() {
                    "SAVED LIGHTING PROFILE (IDENTITY)"
                } else {
                    "SAVED LIGHTING CORRECTION ACTIVE"
                })
                .monospace()
                .strong()
                .color(OK),
            );
            self.calibration_removal_controls(
                ui,
                SessionKind::Photometric,
                "Remove saved lighting correction",
            );
        }
        ui.add_space(SP2);

        match capture_status {
            GeometryCaptureStatus::Rest {
                instruction,
                remaining_s,
                overall,
                awaiting_confirmation,
                ..
            } if self.calibration_capture_purpose
                == Some(CalibrationCapturePurpose::Photometric) =>
            {
                self.geometry_rest_guide(
                    ui,
                    instruction,
                    remaining_s,
                    overall,
                    awaiting_confirmation,
                );
                if ui.button("Cancel recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Capture {
                instruction,
                remaining_s,
                overall,
                samples,
                target_open,
                stereo_stalled,
                ..
            } if self.calibration_capture_purpose
                == Some(CalibrationCapturePurpose::Photometric) =>
            {
                ui.label(
                    egui::RichText::new(instruction)
                        .monospace()
                        .strong()
                        .color(if stereo_stalled { WARN } else { ACCENT }),
                );
                if let Some(target) = target_open {
                    ui.label(num(&format!(
                        "target eyelid openness {:.0}%",
                        target * 100.0
                    )));
                }
                ui.add(
                    egui::ProgressBar::new(overall)
                        .text(format!("{remaining_s:.1}s  {samples} stereo frames")),
                );
                if stereo_stalled {
                    ui.label(
                        egui::RichText::new("WAITING FOR FRESH STEREO FRAMES")
                            .monospace()
                            .color(WARN),
                    );
                }
                if ui.button("Cancel recording").clicked() {
                    self.geometry_capture.abort();
                    self.geometry_capture_baseline = None;
                    self.geometry_capture_filters = None;
                    self.photometric_capture_baseline = None;
                    self.calibration_capture_purpose = None;
                    self.reset_geometry_recording_export();
                }
            }
            GeometryCaptureStatus::Done {
                train_samples,
                holdout_samples,
            } if self.calibration_capture_purpose
                == Some(CalibrationCapturePurpose::Photometric) =>
            {
                ui.label(
                    egui::RichText::new("PHOTOMETRIC RECORDING COMPLETE")
                        .monospace()
                        .strong()
                        .color(OK),
                );
                ui.label(num(&format!(
                    "{train_samples} training + {holdout_samples} untouched holdout frames"
                )));
                if let Some(path) = &self.geometry_recording_path {
                    ui.label(num(&format!("Saved: {}", path.display())));
                } else if self.geometry_recording_export_job.is_some() {
                    ui.label(label("Saving the biometric feedback ZIP..."));
                }
                if ui
                    .add_enabled(
                        self.geometry_recording_path.is_some()
                            && self.geometry_recording_export_job.is_none(),
                        egui::Button::new("Start safe photometric fit"),
                    )
                    .clicked()
                {
                    self.start_photometric_fit();
                }
            }
            _ => match fit_status {
                PhotometricStatus::Running {
                    stage,
                    completed,
                    total,
                    log,
                } => {
                    ui.label(
                        egui::RichText::new("PHOTOMETRIC FIT RUNNING")
                            .monospace()
                            .strong()
                            .color(ACCENT),
                    );
                    let progress = if total == 0 {
                        0.0
                    } else {
                        completed as f32 / total as f32
                    };
                    ui.add(
                        egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                            .text(format!("{stage}  {completed}/{total}")),
                    );
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    if ui.button("Cancel fit").clicked() {
                        self.photometric_fitter.cancel();
                    }
                }
                PhotometricStatus::Done { result, log } => {
                    ui.label(
                        egui::RichText::new(if result.accepted {
                            "PHOTOMETRIC FIT: ACCEPT"
                        } else {
                            "PHOTOMETRIC FIT: KEEP CURRENT"
                        })
                        .monospace()
                        .strong()
                        .color(if result.accepted { OK } else { WARN }),
                    );
                    ui.label(label(&result.reason));
                    let relative =
                        result.holdout_improvement / result.baseline_holdout.score.abs().max(0.001);
                    ui.label(num(&format!(
                        "holdout score {:.3} -> {:.3}  delta {:+.3} ({:+.1}%)",
                        result.baseline_holdout.score,
                        result.candidate_holdout.score,
                        result.holdout_improvement,
                        relative * 100.0,
                    )));
                    ui.label(num(&format!(
                        "holdout separation L/R {:.2}/{:.2} -> {:.2}/{:.2}",
                        result.baseline_holdout.separation[0],
                        result.baseline_holdout.separation[1],
                        result.candidate_holdout.separation[0],
                        result.candidate_holdout.separation[1],
                    )));
                    let c = result.candidate;
                    ui.label(num(&format!(
                        "affine L x{:.3} {:+.1}  R x{:.3} {:+.1}",
                        c.affine[0][0], c.affine[0][1], c.affine[1][0], c.affine[1][1],
                    )));
                    ui.label(num(&format!(
                        "weak flatten {} strength {:.2} radius {:.2}",
                        if c.flatten.enabled { "on" } else { "off" },
                        c.flatten.strength,
                        c.flatten.radius,
                    )));
                    ui.label(num(&format!(
                                "field L h/v/cx/cy {:+.3}/{:+.3}/{:+.3}/{:+.3}  R {:+.3}/{:+.3}/{:+.3}/{:+.3}",
                                c.field[0].horizontal,
                                c.field[0].vertical,
                                c.field[0].horizontal_curve,
                                c.field[0].vertical_curve,
                                c.field[1].horizontal,
                                c.field[1].vertical,
                                c.field[1].horizontal_curve,
                                c.field[1].vertical_curve,
                            )));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    ui.horizontal(|ui| {
                        if result.accepted && ui.button("Apply validated correction").clicked() {
                            self.apply_photometric_candidate(&result);
                        }
                        if ui.button("Record again").clicked() {
                            self.start_photometric_capture();
                        }
                    });
                }
                PhotometricStatus::Failed { message, log } => {
                    ui.label(
                        egui::RichText::new("PHOTOMETRIC FIT FAILED SAFELY")
                            .monospace()
                            .strong()
                            .color(ERR),
                    );
                    ui.label(label(&message));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    if ui.button("Record again").clicked() {
                        self.start_photometric_capture();
                    }
                }
                PhotometricStatus::Cancelled { log } => {
                    ui.label(label(
                        "Fit cancelled. Live and saved correction were not changed.",
                    ));
                    if let Some(last) = log.last() {
                        ui.label(num(last));
                    }
                    if ui.button("Record again").clicked() {
                        self.start_photometric_capture();
                    }
                }
                PhotometricStatus::Idle => {
                    ui.label(
                        egui::RichText::new(if ready { "OK" } else { "WAIT" })
                            .monospace()
                            .strong()
                            .color(if ready { OK } else { WARN }),
                    );
                    ui.label(label(&ready_detail));
                    ui.label(num(&format!(
                        "guided recording about {:.0}s; fit runs in the background",
                        crate::geometry_calib::total_seconds()
                    )));
                    if ui
                        .add_enabled(ready, egui::Button::new("Start photometric recording"))
                        .clicked()
                    {
                        self.start_photometric_capture();
                    }
                }
            },
        }

        if self.photometric_rollback.is_some()
            && ui.button("Rollback last applied correction").clicked()
        {
            self.rollback_photometric();
        }
    }

    fn persist_gaze_correction(&mut self, correction: GazeCorrection) -> Result<(), String> {
        *self.pipeline.gaze_correction.lock().unwrap() = correction;
        let device = self.pipeline.device_key.clone();
        self.config.set_gaze_correction(&device, correction);
        if let Err(e) = self.config.save(&crate::config::config_path()) {
            let message = format!("Gaze adjustment is live only; save failed: {e}");
            self.gaze_center_msg = Some((message.clone(), ERR));
            self.reload_msg = Some((message.clone(), ERR));
            self.events.push((now_hms(), message.clone(), ERR));
            return Err(message);
        }
        Ok(())
    }

    /// Advance the one-second straight-ahead capture using fresh, de-duplicated native
    /// gaze frames. The source snapshot is before correction, but we apply the running
    /// HMD handedness first so the learned offsets live in the same space as the sliders.
    fn update_gaze_center_capture(&mut self) {
        let Some(capture) = self.gaze_center_capture.as_mut() else {
            return;
        };
        let gaze = self.tele.fresh_gaze();
        if gaze.timestamp_us != 0
            && gaze.timestamp_us != capture.last_timestamp_us
            && gaze.left.gaze_valid
            && gaze.right.gaze_valid
        {
            let mut dirs = [gaze.left.gaze, gaze.right.gaze];
            if self.pipeline.flip_gaze_x.load(Ordering::Relaxed) {
                dirs[0][0] = -dirs[0][0];
                dirs[1][0] = -dirs[1][0];
            }
            if let (Some(l), Some(r)) = (
                crate::pipeline::gaze_angles_deg(dirs[0]),
                crate::pipeline::gaze_angles_deg(dirs[1]),
            ) {
                for axis in 0..2 {
                    capture.sum_deg[0][axis] += l[axis] as f64;
                    capture.sum_deg[1][axis] += r[axis] as f64;
                }
                capture.count += 1;
                capture.last_timestamp_us = gaze.timestamp_us;
            }
        }
        if capture.started.elapsed() < Duration::from_secs(1) {
            return;
        }

        let capture = self.gaze_center_capture.take().unwrap();
        if capture.count < 10 {
            self.gaze_center_msg = Some((
                "Center failed: not enough valid gaze samples (keep eyes open and retry)".into(),
                ERR,
            ));
            return;
        }
        let mut correction = *self.pipeline.gaze_correction.lock().unwrap();
        correction.enabled = true;
        for eye in 0..2 {
            let yaw = capture.sum_deg[eye][0] as f32 / capture.count as f32;
            let pitch = capture.sum_deg[eye][1] as f32 / capture.count as f32;
            let sign = if eye == 0 { -0.5 } else { 0.5 };
            correction.offset_x_deg[eye] = (-(yaw * correction.scale_x[eye]
                + correction.vergence_deg * sign))
                .clamp(-15.0, 15.0);
            correction.offset_y_deg[eye] = (-(pitch * correction.scale_y[eye])).clamp(-15.0, 15.0);
        }
        if self.persist_gaze_correction(correction).is_ok() {
            self.gaze_center_msg =
                Some((format!("Centered from {} fresh samples", capture.count), OK));
        }
    }

    /// Returns the source to apply when the user explicitly requests a reload.
    /// Correction sliders remain live and do not require reload.
    fn gaze_correction_body(&mut self, ui: &mut egui::Ui) -> Option<GazeSource> {
        let is_xr5 = crate::config::canonical_device_key(&self.pipeline.device_key) == "pimax_xr5";
        ui.label(h3(if is_xr5 {
            "Dream Air / XR5 gaze correction"
        } else {
            "Pimax VR4 gaze correction"
        }));
        ui.add_space(SP2);
        ui.label(label(
            "Run Pimax/Tobii calibration first. This is a saved finishing trim for residual centre, vergence, and range mismatch; it affects gaze output only.",
        ));
        ui.add_space(SP3);

        let mut reload_source = None;
        if is_xr5 {
            let active_source = if self.tele.gaze_src.contains("combined") {
                GazeSource::Combined
            } else {
                GazeSource::PerEye
            };
            let mut selected_source = self.gaze_source_modal_edit.unwrap_or(active_source);
            let mut use_combined = selected_source == GazeSource::Combined;
            if ui
                .checkbox(
                    &mut use_combined,
                    "Use EyeChip combined gaze for both eyes (steadier)",
                )
                .on_hover_text(
                    "Dream Air / XR5 only. Uses Tobii's fused column-5 gaze, not an average made by SRanibro.",
                )
                .changed()
            {
                selected_source = if use_combined {
                    GazeSource::Combined
                } else {
                    GazeSource::PerEye
                };
                self.gaze_source_modal_edit = Some(selected_source);
            }
            ui.label(label(
                "Combined mode can stay stable when one eye is lost, but removes natural dynamic cross-eye/near-focus motion. Openness is unchanged.",
            ));
            ui.label(num(&format!("active source: {}", active_source.as_str())));
            let source_pending = selected_source != active_source;
            if ui
                .add_enabled(
                    source_pending && self.gaze_center_capture.is_none(),
                    egui::Button::new(if source_pending {
                        if self.gaze_center_capture.is_some() {
                            "Finish Center before reloading"
                        } else {
                            "Apply gaze source & reload"
                        }
                    } else {
                        "Gaze source already active"
                    }),
                )
                .clicked()
            {
                reload_source = Some(selected_source);
            }
            if source_pending {
                ui.label(
                    egui::RichText::new(
                        "Reload prevents per-eye/combined switching jitter. Run Center again afterwards.",
                    )
                    .monospace()
                    .size(10.0 * S)
                    .color(WARN),
                );
            }
            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP3);
        }

        let mut correction = *self.pipeline.gaze_correction.lock().unwrap();
        let mut changed = ui
            .checkbox(&mut correction.enabled, "Enable gaze adjustment")
            .on_hover_text(
                "Disabling this returns to the native Pimax/Tobii gaze range and centre.",
            )
            .changed();
        ui.add_space(SP2);
        let capturing = self.gaze_center_capture.is_some();
        ui.horizontal(|ui| {
            let button = egui::Button::new(
                egui::RichText::new(if capturing {
                    "Capturing straight-ahead…"
                } else {
                    "Center (look straight for 1s)"
                })
                .monospace()
                .strong()
                .color(if capturing { TEXT2 } else { BG }),
            )
            .fill(if capturing { INNER } else { ACCENT });
            if ui
                .add_enabled(!capturing && !self.guided_capture_running(), button)
                .clicked()
            {
                self.gaze_center_capture = Some(GazeCenterCapture::new());
                self.gaze_center_msg = Some(("Hold a relaxed straight-ahead gaze…".into(), ACCENT));
            }
            if let Some((msg, col)) = &self.gaze_center_msg {
                ui.label(
                    egui::RichText::new(msg)
                        .monospace()
                        .size(10.0 * S)
                        .color(*col),
                );
            }
        });

        ui.add_space(SP3);
        ui.label(h3("Avatar gaze movement"));
        ui.label(label(
            "100% keeps the calibrated Tobii range. Increase it if the avatar's eyes barely move, then run Center again.",
        ));
        ui.spacing_mut().slider_width = 260.0 * S;
        let current_x = gaze_range_percent(&correction, GazeRangeAxis::Horizontal);
        let current_y = gaze_range_percent(&correction, GazeRangeAxis::Vertical);
        let mut shared_x = shared_gaze_range_slider_value(&correction, GazeRangeAxis::Horizontal);
        let mut shared_y = shared_gaze_range_slider_value(&correction, GazeRangeAxis::Vertical);
        let x_changed = ui
            .add(
                egui::Slider::new(&mut shared_x, 25.0..=250.0)
                    .suffix("%")
                    .text("Left / right movement"),
            )
            .on_hover_text(
                "Increase this if the avatar's eyes barely move left/right. This changes gaze only, not eyelids.",
            )
            .changed();
        let y_changed = ui
            .add(
                egui::Slider::new(&mut shared_y, 25.0..=250.0)
                    .suffix("%")
                    .text("Up / down movement"),
            )
            .on_hover_text(
                "Increase this if the avatar's eyes barely move up/down. This changes gaze only, not eyelids.",
            )
            .changed();
        if x_changed {
            set_shared_gaze_range_percent(&mut correction, GazeRangeAxis::Horizontal, shared_x);
            changed = true;
        }
        if y_changed {
            set_shared_gaze_range_percent(&mut correction, GazeRangeAxis::Vertical, shared_y);
            changed = true;
        }
        if gaze_range_is_mixed(current_x) {
            ui.label(label(&format!(
                "Current L/R is mixed: L {:.0}% / R {:.0}%. Moving the shared slider sets both.",
                current_x[0], current_x[1]
            )));
        }
        if gaze_range_is_mixed(current_y) {
            ui.label(label(&format!(
                "Current U/D is mixed: L {:.0}% / R {:.0}%. Moving the shared slider sets both.",
                current_y[0], current_y[1]
            )));
        }

        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP3);
        ui.spacing_mut().slider_width = 260.0 * S;
        let vergence_changed = ui
            .add(
                egui::Slider::new(&mut correction.vergence_deg, -10.0..=10.0)
                    .text("vergence trim (deg)"),
            )
            .changed();
        if vergence_changed {
            correction.enabled = true;
            changed = true;
        }
        ui.label(label("Moves left/right gaze in opposite directions; use after Center if the avatar still looks cross-eyed."));
        ui.label(label("Advanced per-eye adjustment"));

        for (eye, name) in [(0usize, "LEFT EYE"), (1usize, "RIGHT EYE")] {
            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            ui.label(
                egui::RichText::new(name)
                    .monospace()
                    .size(11.0 * S)
                    .strong()
                    .color(TEXT1),
            );
            let offset_x_changed = ui
                .add(
                    egui::Slider::new(&mut correction.offset_x_deg[eye], -15.0..=15.0)
                        .text("X centre (deg)"),
                )
                .changed();
            let offset_y_changed = ui
                .add(
                    egui::Slider::new(&mut correction.offset_y_deg[eye], -15.0..=15.0)
                        .text("Y centre (deg)"),
                )
                .changed();
            let scale_x_changed = ui
                .add(
                    egui::Slider::new(&mut correction.scale_x[eye], 0.25..=2.5)
                        .text("X movement (per eye)"),
                )
                .changed();
            let scale_y_changed = ui
                .add(
                    egui::Slider::new(&mut correction.scale_y[eye], 0.25..=2.5)
                        .text("Y movement (per eye)"),
                )
                .changed();
            if offset_x_changed || offset_y_changed || scale_x_changed || scale_y_changed {
                correction.enabled = true;
                changed = true;
            }
        }

        ui.add_space(SP3);
        if ui.button("Reset gaze correction").clicked() {
            correction = GazeCorrection::default();
            self.gaze_center_capture = None;
            self.gaze_center_msg = Some(("Reset to native Pimax/Tobii gaze".into(), TEXT2));
            changed = true;
        }
        if changed {
            let _ = self.persist_gaze_correction(correction);
        }
        reload_source
    }

    fn gaze_correction_modal(&mut self, ctx: &egui::Context) {
        if self.gaze_residual_capture.is_running() {
            self.show_gaze_modal = false;
            self.gaze_source_modal_edit = None;
            self.gaze_center_capture = None;
            return;
        }
        let screen = ctx.screen_rect();
        let (closed, reload_source) = egui::Area::new(egui::Id::new("gaze_correction_modal"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                ui.painter()
                    .rect_filled(screen, 0.0, Color32::from_black_alpha(195));
                let scrim = ui.interact(
                    screen,
                    egui::Id::new("gaze_correction_scrim"),
                    Sense::click(),
                );
                let pw = (screen.width() * 0.70).clamp(360.0, 640.0 * S);
                let ph = (screen.height() * 0.88).clamp(320.0, 760.0 * S);
                let panel = Rect::from_center_size(screen.center(), vec2(pw, ph));
                ui.painter().rect_filled(panel, 14.0 * S, NAV_BG);
                ui.painter()
                    .rect_stroke(panel, 14.0 * S, Stroke::new(1.0, BORDER));
                let mut close_btn = false;
                let mut reload_source = None;
                ui.allocate_new_ui(
                    egui::UiBuilder::new().max_rect(panel.shrink(CARD_PAD)),
                    |ui| {
                        egui::ScrollArea::vertical().show(ui, |ui| {
                            reload_source = self.gaze_correction_body(ui);
                            ui.add_space(SP3);
                            close_btn = ui.button("Close").clicked();
                        });
                    },
                );
                let outside = scrim.clicked()
                    && scrim
                        .interact_pointer_pos()
                        .map_or(false, |p| !panel.contains(p));
                (close_btn || outside, reload_source)
            })
            .inner;
        if let Some(source) = reload_source {
            self.show_gaze_modal = false;
            self.gaze_source_modal_edit = None;
            self.apply_gaze_source_and_reload(source);
            return;
        }
        if closed || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.show_gaze_modal = false;
            self.gaze_source_modal_edit = None;
            if self.gaze_center_capture.take().is_some() {
                self.gaze_center_msg = Some(("Center capture cancelled".into(), TEXT2));
            }
        }
    }

    /// Apply only the XR5 gaze-provider choice from the Calibration modal. Preserve
    /// unrelated half-edited Settings fields instead of silently committing them.
    fn apply_gaze_source_and_reload(&mut self, source: GazeSource) {
        let mut pending_settings = self.edit.clone();
        let mut source_only = SettingsEdit::from_cfg(&self.config);
        source_only.gaze_source = source;
        self.edit = source_only;
        self.apply_and_reload();
        pending_settings.gaze_source = source;
        self.edit = pending_settings;

        if let Some((message, color)) = self.reload_msg.clone() {
            self.gaze_center_msg = Some((message, color));
        }
    }

    /// The ML-input settings body (tabbed) shown inside the gear modal: an "Image" tab
    /// (crop / stretch / rotate) and a "Filter" tab (reflection despeckle + response
    /// heatmap), with a shared live preview of the processed model input below.
    fn ml_geometry_body(&mut self, ui: &mut egui::Ui, frames: &[Option<EyeFrame>; 2]) {
        if self.gaze_residual_capture.is_running() {
            ui.label(
                egui::RichText::new(
                    "Image geometry and filters are locked until the research recording finishes.",
                )
                .monospace()
                .color(WARN),
            );
            return;
        }
        if self.geometry_evidence_locked() {
            ui.label(
                egui::RichText::new(
                    "Image geometry and filters are locked while the current image-alignment evidence is active.",
                )
                .monospace()
                .color(WARN),
            );
            return;
        }
        ui.spacing_mut().slider_width = 240.0 * S;
        ui.horizontal(|ui| {
            if ui.selectable_label(self.geom_tab == 0, "Image").clicked() {
                self.geom_tab = 0;
            }
            if ui.selectable_label(self.geom_tab == 1, "Filter").clicked() {
                self.geom_tab = 1;
            }
        });
        ui.add_space(SP2);
        ui.separator();
        ui.add_space(SP2);
        if self.geom_tab == 0 {
            // Image tab: geometry controls, then a preview of the warped MODEL input.
            self.geom_image_controls(ui);
            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            self.geom_preview(ui, frames, true);
        } else {
            // Filter tab: despeckle controls, then the FILTERED-EYE preview RIGHT below it,
            // then the response heatmap.
            self.geom_filter_controls(ui);
            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
            self.geom_preview(ui, frames, false);
            self.geom_heatmap(ui);
        }
    }

    /// "Image" tab: crop / stretch / rotate the image fed to the eye model — per device,
    /// per EYE (both / left / right target selector; the cameras can sit at different
    /// angles per eye, so each side gets its own values).
    fn geom_image_controls(&mut self, ui: &mut egui::Ui) {
        // Manual edits save immediately. Never let an explicitly unsaved fit preview
        // leak into config merely because the user opens this editor and moves a slider.
        self.restore_geometry_preview(false);
        ui.label(label(
            "Crop / stretch / rotate the image fed to the eye model — tune out per-person / per-HMD variance. Reset restores this HMD's built-in preset (Dream Air/XR5 uses an angled-camera reconstruction; other HMDs use no warp).",
        ));
        ui.add_space(SP2);
        let mut gs = *self.pipeline.geometry.lock().unwrap();
        ui.horizontal(|ui| {
            ui.label(label("apply to"));
            ui.add_space(4.0 * S);
            for (v, name) in [(0u8, "both"), (1u8, "left"), (2u8, "right")] {
                let sel = self.geom_eye == v;
                let txt = egui::RichText::new(name).monospace().size(11.0 * S);
                if ui
                    .selectable_label(sel, if sel { txt.strong().color(ACCENT) } else { txt })
                    .clicked()
                {
                    self.geom_eye = v;
                }
            }
            if self.geom_eye == 0 && gs[0] != gs[1] {
                ui.add_space(6.0 * S);
                ui.label(label(
                    "(L/R asymmetric — XR5 edits are mirrored onto right)",
                ));
            }
        });
        ui.add_space(SP2);
        // Sliders bind to the SELECTED eye's live values, re-read every frame — so
        // switching the target snaps them to that eye's numbers. "both" shows the
        // left eye's values and writes to both eyes.
        let mut g = if self.geom_eye == 2 { gs[1] } else { gs[0] };
        let mut ch = false;
        ch |= ui
            .add(egui::Slider::new(&mut g.crop_left, 0.0..=0.45).text("crop left"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.crop_right, 0.0..=0.45).text("crop right"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.crop_top, 0.0..=0.45).text("crop top"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.crop_bottom, 0.0..=0.45).text("crop bottom"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.scale_x, 0.5..=2.0).text("stretch X"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.scale_y, 0.5..=2.0).text("stretch Y"))
            .changed();
        ch |= ui
            .add(egui::Slider::new(&mut g.rotate_deg, -45.0..=45.0).text("rotate deg"))
            .changed();
        ui.add_space(SP2);
        let reset = ui
            .button(match self.geom_eye {
                1 => "Reset image (left)",
                2 => "Reset image (right)",
                _ => "Reset image (both)",
            })
            .clicked();
        if reset {
            let preset = crate::config::default_ml_geometry(&self.pipeline.device_key);
            match self.geom_eye {
                1 => gs[0] = preset[0],
                2 => gs[1] = preset[1],
                _ => gs = preset,
            }
            if let Some(v) = gs[0].mirror_h {
                self.pipeline.ml_mirror_l.store(v, Ordering::Relaxed);
            }
            if let Some(v) = gs[1].mirror_h {
                self.pipeline.ml_mirror_r.store(v, Ordering::Relaxed);
            }
        } else if ch {
            match self.geom_eye {
                1 => gs[0] = g,
                2 => gs[1] = g,
                _ => {
                    let mirror_state = [gs[0].mirror_h, gs[1].mirror_h];
                    gs[0] = g;
                    gs[0].mirror_h = mirror_state[0];
                    if self.pipeline.device_key == "pimax_xr5" {
                        // XR5 cameras are a mirrored physical pair. "both" edits the
                        // left shape and applies its horizontal counterpart to the right
                        // instead of destroying the asymmetric device reconstruction.
                        let mut right = g.mirrored_x();
                        right.mirror_h = mirror_state[1];
                        gs[1] = right;
                    } else {
                        gs[1] = g;
                        gs[1].mirror_h = mirror_state[1];
                    }
                }
            }
        }
        if ch || reset {
            *self.pipeline.geometry.lock().unwrap() = gs;
            let device = self.pipeline.device_key.clone();
            self.config.set_geometry(&device, gs);
            let _ = self.config.save(&crate::config::config_path());
        }
    }

    /// "Filter" tab: specular-dot despeckle + the ML response heatmap that diagnoses it.
    fn geom_filter_controls(&mut self, ui: &mut egui::Ui) {
        let device = crate::config::canonical_device_key(&self.pipeline.device_key);
        if supports_frontal_photometric_correction(&device) {
            ui.label(h3(&format!(
                "{} eye-image correction",
                frontal_photometric_profile_name(&device)
            )));
            ui.label(label(
                "Frontal-camera profile: brightness statistics are measured before the identity geometry transform and saved only for this HMD.",
            ));
            ui.add_space(SP3);
            ui.separator();
            ui.add_space(SP2);
        }
        ui.label(h3("Reflection filter (despeckle)"));
        ui.add_space(SP2);
        ui.label(label(
            "Removes bright IR / glasses reflection dots from the ML input — the heatmap showed the model reads brightness as 'more open', so glints inflate and destabilize openness. Applied before the model (see the preview); the live camera images stay raw.",
        ));
        ui.add_space(SP2);
        let mut dsp = *self.pipeline.despeckle.lock().unwrap();
        let mut dch = false;
        dch |= ui
            .checkbox(&mut dsp.enabled, "Enabled (removes reflection dots)")
            .changed();
        dch |= ui
            .add(egui::Slider::new(&mut dsp.threshold, 0.05..=0.4).text("spot threshold"))
            .changed();
        dch |= ui
            .add(egui::Slider::new(&mut dsp.radius, 2..=6).text("spot radius"))
            .changed();
        if dch {
            *self.pipeline.despeckle.lock().unwrap() = dsp;
            let device = self.pipeline.device_key.clone();
            self.config.set_despeckle(&device, dsp);
            let _ = self.config.save(&crate::config::config_path());
        }

        // --- Flatten shadows (illumination) ---------------------------------------------
        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP2);
        ui.label(h3("Flatten shadows (illumination)"));
        ui.add_space(SP2);
        ui.label(label(
            "Removes a low-frequency shadow / gradient -- like the dark centre band that appears when the eye is close to the lens -- while keeping the eye's structure. Experimental; enable when the close-up shadow is the problem. Shown in the preview below.",
        ));
        ui.add_space(SP2);
        let mut flt = *self.pipeline.flatten.lock().unwrap();
        let mut fch = false;
        fch |= ui.checkbox(&mut flt.enabled, "Enabled").changed();
        fch |= ui
            .add(egui::Slider::new(&mut flt.strength, 0.0..=1.0).text("strength"))
            .changed();
        fch |= ui
            .add(egui::Slider::new(&mut flt.radius, 0.1..=0.5).text("smooth radius"))
            .changed();
        if fch {
            *self.pipeline.flatten.lock().unwrap() = flt;
            let device = self.pipeline.device_key.clone();
            self.config.set_flatten(&device, flt);
            let _ = self.config.save(&crate::config::config_path());
        }

        // --- Fixed manual brightness ----------------------------------------------------
        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP2);
        ui.label(h3("Eye-image brightness"));
        ui.add_space(SP2);
        ui.label(label(
            "Fixed brightness for the image sent to the eyelid model. It never learns or changes automatically. Raw eye-image output and Tobii gaze stay unchanged.",
        ));
        ui.add_space(SP2);
        let dev = self.pipeline.device_key.clone();
        let mut bn = *self.pipeline.brightness.lock().unwrap();
        let mut bch = false;
        let mut brightness_percent =
            crate::ml::brightness::sanitize_manual_gain(bn.manual_gain) * 100.0;
        if ui
            .add(
                egui::Slider::new(
                    &mut brightness_percent,
                    crate::ml::brightness::MANUAL_GAIN_MIN * 100.0
                        ..=crate::ml::brightness::MANUAL_GAIN_MAX * 100.0,
                )
                .step_by(1.0)
                .suffix("%")
                .text("brightness"),
            )
            .on_hover_text("50% darkens, 100% is unchanged, and up to 200% brightens.")
            .changed()
        {
            bn.manual_gain = brightness_percent / 100.0;
            bch = true;
        }
        if ui
            .add_enabled(
                (brightness_percent - 160.0).abs() > 0.01,
                egui::Button::new("Reset to 160% standard"),
            )
            .clicked()
        {
            bn.manual_gain = crate::core::types::BrightnessNorm::default().manual_gain;
            bch = true;
        }
        if bch {
            let mut g = self.pipeline.brightness.lock().unwrap();
            g.manual_gain = bn.manual_gain;
            g.enabled = false;
            g.auto_learn = false;
            g.captured = false;
            let live = *g;
            drop(g);
            self.config.set_brightness(&dev, live);
            let _ = self.config.save(&crate::config::config_path());
        }
    }

    /// The ML response heatmap section (occlusion sensitivity), shown in the Filter tab
    /// BELOW the filtered-eye preview.
    fn geom_heatmap(&mut self, ui: &mut egui::Ui) {
        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP2);
        ui.label(h3("ML response heatmap"));
        ui.add_space(SP2);
        ui.label(label(
            "High-resolution probes over the exact 100x100 model input. Occlusion shows relied-on structure; Brightness isolates where lighter/darker pixels move openness; Contrast isolates local boundaries; Glint inject tests reflection damage. Diagnostic only: it never changes or saves a filter.",
        ));
        ui.add_space(SP2);
        let ml_loaded = self.tele.ml_loaded;
        let computing = self.pipeline.heatmap.computing.load(Ordering::Relaxed);
        let mode = self.pipeline.heatmap.mode.load(Ordering::Relaxed);
        ui.horizontal(|ui| {
            if ui.selectable_label(mode == 0, "Occlusion").clicked() {
                self.pipeline.heatmap.mode.store(0, Ordering::Relaxed);
            }
            if ui.selectable_label(mode == 1, "Glint inject").clicked() {
                self.pipeline.heatmap.mode.store(1, Ordering::Relaxed);
            }
            if ui.selectable_label(mode == 2, "Brightness").clicked() {
                self.pipeline.heatmap.mode.store(2, Ordering::Relaxed);
            }
            if ui.selectable_label(mode == 3, "Contrast").clicked() {
                self.pipeline.heatmap.mode.store(3, Ordering::Relaxed);
            }
            ui.add_space(SP2);
            let btn = egui::Button::new(
                egui::RichText::new(if computing { "computing…" } else { "Compute" })
                    .monospace()
                    .size(12.0 * S),
            );
            if ui.add_enabled(ml_loaded && !computing, btn).clicked() {
                self.pipeline.heatmap.req.store(true, Ordering::Relaxed);
            }
            if !ml_loaded {
                ui.label(label("(load an ML model first)"));
            }
        });
        ui.add_space(SP2);
        let heat_meta = {
            let guard = self.pipeline.heatmap.result.lock().unwrap();
            guard.as_ref().map(|result| {
                (
                    result.mode,
                    result.p98_abs_delta,
                    result.peak_abs_delta,
                    result.baseline_openness,
                    result.brightness_response,
                    result.contrast_response,
                )
            })
        };
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.heat_auto_scale, "Auto scale focus");
            if let Some((_, p98, peak, _, _, _)) = heat_meta {
                ui.label(num(&format!("p98 {p98:.4}   peak {peak:.4}")));
            }
        });
        if !self.heat_auto_scale {
            ui.add(
                egui::Slider::new(&mut self.heat_vmax, 0.001..=0.5)
                    .logarithmic(true)
                    .text("heat scale (VMAX)"),
            );
        }
        if let Some((result_mode, _, _, _, _, _)) = heat_meta {
            let legend = match result_mode {
                crate::ml::heatmap::HeatMode::OcclusionMean => {
                    "Red: erasing this area lowered openness. Blue: erasing it raised openness."
                }
                crate::ml::heatmap::HeatMode::GlintInject => {
                    "Red: a bright glint here raised openness. Blue: it lowered openness."
                }
                crate::ml::heatmap::HeatMode::BrightnessSensitivity => {
                    "Red: making this area brighter raises openness. Blue: darker raises openness."
                }
                crate::ml::heatmap::HeatMode::ContrastSensitivity => {
                    "Red: stronger local contrast raises openness. Blue: weaker contrast raises openness."
                }
            };
            ui.label(label(legend));
        }
        ui.add_space(SP2);
        let vmax = if self.heat_auto_scale {
            heat_meta
                .map(|(_, p98, _, _, _, _)| p98)
                .unwrap_or(self.heat_vmax)
        } else {
            self.heat_vmax
        }
        .max(1e-3);
        let n = crate::ml::preprocess::DST;
        let imgs: Option<(egui::ColorImage, egui::ColorImage)> = {
            let guard = self.pipeline.heatmap.result.lock().unwrap();
            guard.as_ref().map(|res| {
                let mk = |i: usize| {
                    let mut img = egui::ColorImage::new([n, n], Color32::BLACK);
                    for k in 0..n * n {
                        img.pixels[k] = heat_color(res.base[i][k], res.delta[i][k], vmax);
                    }
                    img
                };
                (mk(0), mk(1))
            })
        };
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            let side = 160.0 * S;
            for i in 0..2 {
                let (name, slot): (&str, &mut Option<egui::TextureHandle>) = if i == 0 {
                    ("heat_l", &mut self.tex_heat_l)
                } else {
                    ("heat_r", &mut self.tex_heat_r)
                };
                if let Some((ref l, ref r)) = imgs {
                    let img = if i == 0 { l.clone() } else { r.clone() };
                    match slot {
                        Some(h) => h.set(img, egui::TextureOptions::LINEAR),
                        None => {
                            *slot = Some(ctx.load_texture(name, img, egui::TextureOptions::LINEAR))
                        }
                    }
                }
                let sz = vec2(side, side);
                let (rect, _) = ui.allocate_exact_size(sz, Sense::hover());
                if imgs.is_some() {
                    if let Some(h) = slot.as_ref() {
                        egui::Image::new(egui::load::SizedTexture::new(h.id(), sz))
                            .rounding(R_BOX)
                            .paint_at(ui, rect);
                    }
                } else {
                    ui.painter().rect_filled(rect, R_BOX, INNER);
                    ui.painter().text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        if computing {
                            "computing…"
                        } else {
                            "press Compute"
                        },
                        FontId::monospace(10.0 * S),
                        TEXT3,
                    );
                }
                ui.painter()
                    .rect_stroke(rect, R_BOX, Stroke::new(1.0, BORDER));
                ui.painter().text(
                    rect.left_top() + vec2(5.0 * S, 4.0 * S),
                    Align2::LEFT_TOP,
                    if i == 0 { "L" } else { "R" },
                    FontId::monospace(9.0 * S),
                    TEXT3,
                );
                if i == 0 {
                    ui.add_space(SP2);
                }
            }
        });
        if let Some((_, _, _, baseline, brightness, contrast)) = heat_meta {
            ui.add_space(SP2);
            ui.label(label(
                "Current-frame response only. The factor changes one eye image at a time; a larger openness value is not automatically a better setting.",
            ));
            egui::Grid::new("heat_photometric_response")
                .num_columns(6)
                .spacing(vec2(12.0 * S, 3.0 * S))
                .striped(true)
                .show(ui, |ui| {
                    ui.label(num("factor"));
                    for factor in crate::ml::heatmap::RESPONSE_FACTORS {
                        ui.label(num(&format!("{factor:.2}")));
                    }
                    ui.end_row();
                    for (name, values) in [
                        ("brightness L", brightness[0]),
                        ("brightness R", brightness[1]),
                        ("contrast L", contrast[0]),
                        ("contrast R", contrast[1]),
                    ] {
                        ui.label(num(name));
                        for value in values {
                            ui.label(num(&format!("{value:.3}")));
                        }
                        ui.end_row();
                    }
                });
            ui.label(num(&format!(
                "current raw openness   L {:.3}   R {:.3}",
                baseline[0], baseline[1]
            )));
        }
    }

    /// Live preview inside the modal. `warped=true` (Image tab) shows the geometry-warped
    /// MODEL input (100x100) so crop/rotate is visible; `warped=false` (Filter tab) shows
    /// the despeckled REAL eye at its native resolution and NATURAL orientation — L on the
    /// left, R on the right, matching the live cameras — so you can see the reflection dots
    /// removed. Neither applies the ML left/right mirror.
    fn geom_preview(&mut self, ui: &mut egui::Ui, frames: &[Option<EyeFrame>; 2], warped: bool) {
        ui.label(label(if warped {
            "Model input preview (crop / rotate / stretch applied):"
        } else {
            "Filtered eye preview — dots removed (this is your real eye, not the model view):"
        }));
        ui.add_space(SP2);
        let g = *self.pipeline.geometry.lock().unwrap();
        let dsp = *self.pipeline.despeckle.lock().unwrap();
        let flt = *self.pipeline.flatten.lock().unwrap();
        // Preview is intentionally best-effort and may use the newest affine with
        // the newest camera frame. Persistent recordings use the bundled source pair.
        let aff = self.pipeline.bright_affine.lock().unwrap().affine;
        let fitted = *self.pipeline.photometric_correction.lock().unwrap();
        let side = 150.0 * S;
        let ctx = ui.ctx().clone();
        let n = crate::ml::preprocess::DST;
        ui.horizontal(|ui| {
            for i in 0..2 {
                let (name, slot): (&str, &mut Option<egui::TextureHandle>) = if i == 0 {
                    ("ml_prev_l", &mut self.tex_ml_l)
                } else {
                    ("ml_prev_r", &mut self.tex_ml_r)
                };
                if let Some(frame) = &frames[i] {
                    let (fw, fh, px) = frame.view();
                    // Match the ML input pipeline: despeckle -> flatten -> adaptive
                    // brightness -> fitted photometric correction -> fixed geometry.
                    let filtered =
                        crate::ml::preprocess::despeckle(px, fw as usize, fh as usize, &dsp);
                    let flat =
                        crate::ml::preprocess::flatten(&filtered, fw as usize, fh as usize, &flt);
                    let normed = crate::ml::brightness::apply(&flat, aff[i][0], aff[i][1]);
                    let photometric = crate::ml::preprocess::fitted_photometric(
                        &normed,
                        fw as usize,
                        fh as usize,
                        &g[i],
                        &fitted,
                        i,
                    );
                    // Build the display image: the warped model input (geometry, NO mirror),
                    // or the despeckled + brightness-matched real eye at native res.
                    let built = if warped {
                        let prev = crate::ml::preprocess::ml_input_preview(
                            &photometric,
                            fw,
                            fh,
                            &g[i],
                            false,
                        );
                        (prev.len() == n * n).then(|| {
                            let mut im = egui::ColorImage::new([n, n], Color32::BLACK);
                            for k in 0..n * n {
                                im.pixels[k] = Color32::from_gray(prev[k]);
                            }
                            im
                        })
                    } else {
                        let (w, h) = (fw as usize, fh as usize);
                        (w > 0 && h > 0 && photometric.len() >= w * h).then(|| {
                            let mut im = egui::ColorImage::new([w, h], Color32::BLACK);
                            for k in 0..w * h {
                                im.pixels[k] = Color32::from_gray(photometric[k]);
                            }
                            im
                        })
                    };
                    if let Some(im) = built {
                        match slot {
                            Some(h) => h.set(im, egui::TextureOptions::LINEAR),
                            None => {
                                *slot =
                                    Some(ctx.load_texture(name, im, egui::TextureOptions::LINEAR))
                            }
                        }
                    }
                }
                let sz = vec2(side, side);
                let (rect, _) = ui.allocate_exact_size(sz, Sense::hover());
                if let Some(h) = slot.as_ref() {
                    egui::Image::new(egui::load::SizedTexture::new(h.id(), sz))
                        .rounding(R_BOX)
                        .paint_at(ui, rect);
                } else {
                    ui.painter().rect_filled(rect, R_BOX, INNER);
                    ui.painter().text(
                        rect.center(),
                        Align2::CENTER_CENTER,
                        "no signal",
                        FontId::monospace(10.0 * S),
                        TEXT3,
                    );
                }
                ui.painter()
                    .rect_stroke(rect, R_BOX, Stroke::new(1.0, BORDER));
                ui.painter().text(
                    rect.left_top() + vec2(5.0 * S, 4.0 * S),
                    Align2::LEFT_TOP,
                    if i == 0 { "L" } else { "R" },
                    FontId::monospace(9.0 * S),
                    TEXT3,
                );
                if i == 0 {
                    ui.add_space(SP2);
                }
            }
        });
    }

    /// Full-screen modal for the ML-input geometry editor (opened by the eye-cameras
    /// gear). Dims the app behind it to focus attention (egui has no true backdrop blur)
    /// and centers a SCROLLABLE panel so the controls never clip. Dismiss by the Close
    /// button, a click on the dimmed area, or Esc.
    fn geom_modal(&mut self, ctx: &egui::Context, frames: &[Option<EyeFrame>; 2]) {
        let screen = ctx.screen_rect();
        let closed = egui::Area::new(egui::Id::new("geom_modal"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                // Dim scrim over the whole app, and swallow clicks meant for behind it.
                ui.painter()
                    .rect_filled(screen, 0.0, Color32::from_black_alpha(195));
                let scrim = ui.interact(screen, egui::Id::new("geom_scrim"), Sense::click());
                // Centered panel, capped so it fits small screens; its content scrolls.
                let pw = (screen.width() * 0.72).clamp(340.0, 660.0 * S);
                let ph = (screen.height() * 0.88).clamp(300.0, 780.0 * S);
                let panel = Rect::from_center_size(screen.center(), vec2(pw, ph));
                ui.painter().rect_filled(panel, 14.0 * S, NAV_BG);
                ui.painter()
                    .rect_stroke(panel, 14.0 * S, Stroke::new(1.0, BORDER));
                let mut close_btn = false;
                let inner = panel.shrink(CARD_PAD);
                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(inner), |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.ml_geometry_body(ui, frames);
                        ui.add_space(SP3);
                        close_btn = ui.button("Close").clicked();
                    });
                });
                // Close on the Close button, or a click on the dimmed area OUTSIDE the panel.
                let outside = scrim.clicked()
                    && scrim
                        .interact_pointer_pos()
                        .map_or(false, |p| !panel.contains(p));
                close_btn || outside
            })
            .inner;
        if closed || ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.show_geom_modal = false;
        }
    }

    /// Eyebrow calibration card shared by every HMD. Guides the user through the vr_eyebrow capture protocol
    /// and writes RAW eye frames + labels.csv under base_dir()/brow_data.
    fn eyebrow_calib_card(&mut self, ui: &mut egui::Ui, cw: f32) {
        let status = self.brow.status();
        let total_cap = self.brow.total_captured();
        let total_target = brow_calib::total_capture_target();
        let root = self.brow.root().to_path_buf();
        let err = self.brow.last_error.clone();
        let model_loaded = self.tele.brow_loaded.load(Ordering::Relaxed);
        let mut reload_eyebrow_output = false;

        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            let title = format!(
                "Eyebrow training    {}",
                if model_loaded {
                    "BROW MODEL READY"
                } else {
                    "NO BROW MODEL"
                }
            );
            egui::CollapsingHeader::new(h3(&title))
                .id_salt("eyebrow_card")
                .default_open(false)
                .show(ui, |ui| {
                    ui.label(h3("VRChat eyebrow OSC"));
                    let full_osc = self.config.output.osc;
                    let mut eyebrow_osc =
                        self.config.output.eyebrow_osc || full_osc;
                    if ui
                        .add_enabled(
                            !full_osc,
                            egui::Checkbox::new(
                                &mut eyebrow_osc,
                                "Send eyebrows directly to VRChat OSC",
                            ),
                        )
                        .on_hover_text(
                            "VRCFT continues to handle eyes and gaze; SRanibro sends only FT/v2 brow parameters.",
                        )
                        .changed()
                    {
                        self.config.output.eyebrow_osc = eyebrow_osc;
                        reload_eyebrow_output = true;
                    }
                    ui.horizontal(|ui| {
                        ui.label(label("Address"));
                        ui.add(
                            egui::TextEdit::singleline(&mut self.edit.osc_host)
                                .desired_width(180.0 * S),
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.edit.osc_port)
                                .range(1..=u16::MAX),
                        );
                        if ui.button("Apply & reload").clicked() {
                            reload_eyebrow_output = true;
                        }
                    });
                    if full_osc {
                        ui.label(label(
                            "Eyebrows are already included in the full OSC output.",
                        ));
                    } else {
                        ui.label(label(
                            "Default VRChat target: 127.0.0.1 : 9000.",
                        ));
                    }
                    ui.add_space(SP3);

        // --- Current-phase card ------------------------------------------------------
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.label(h3("Eyebrow calibration — data collection"));
            ui.add_space(SP2);
            ui.label(prose(
                "Records training frames for your personal eyebrow model. Follow each prompt; \
                 the run captures a fixed number of frames per expression, then writes a \
                 brow_data folder ready for training.",
            ));
            ui.add_space(SP3);

            // Big instruction + per-phase state, color-coded by phase kind.
            let (accent, big, sub, frac) = match status {
                BrowStatus::Idle => (
                    TEXT2,
                    "Ready".to_string(),
                    "Press Start and follow the prompts.".to_string(),
                    0.0,
                ),
                BrowStatus::Rest { instruction, remaining } => (
                    WARN,
                    instruction.to_string(),
                    format!("REST — starting in {remaining:.1}s"),
                    0.0,
                ),
                BrowStatus::Capture { instruction, folder, captured, target } => (
                    ACCENT,
                    instruction.to_string(),
                    format!("CAPTURING {folder} — {captured}/{target} frames"),
                    if target > 0 { captured as f32 / target as f32 } else { 0.0 },
                ),
                BrowStatus::Done => (
                    OK,
                    "Done — dataset captured".to_string(),
                    "Frames + labels.csv written.".to_string(),
                    1.0,
                ),
            };
            ui.label(egui::RichText::new(big).monospace().size(16.0 * S).strong().color(accent));
            ui.add_space(SP2);
            ui.label(egui::RichText::new(sub).monospace().size(11.0 * S).color(TEXT2));
            ui.add_space(SP2);
            // Per-phase progress bar (only meaningful during a capture phase).
            bar(ui, cw - 2.0 * CARD_PAD, 12.0 * S, frac, accent);
            ui.add_space(SP2);
            // Overall progress across ALL capture phases.
            let ofrac = if total_target > 0 { total_cap as f32 / total_target as f32 } else { 0.0 };
            ui.label(egui::RichText::new(
                format!("total {total_cap}/{total_target} frames"),
            ).monospace().size(10.0 * S).color(TEXT3));
            ui.add_space(2.0 * S);
            bar(ui, cw - 2.0 * CARD_PAD, 8.0 * S, ofrac, OK);

            ui.add_space(SP3);
            // Buttons: Start (idle/done) | Abort (running).
            ui.horizontal(|ui| {
                let running = self.brow.is_running();
                let start_txt = if self.brow.is_done() { "Capture again" } else { "Start" };
                let start = egui::Button::new(
                    egui::RichText::new(start_txt).monospace().size(13.0 * S).strong().color(BG),
                ).fill(ACCENT);
                let start_resp = ui.add_enabled_ui(
                    !running
                        && !self.gaze_residual_capture.is_running()
                        && !self.geometry_evidence_locked()
                        && !self.wide.is_running()
                        && !self.wide_fitter.is_running(),
                    |ui| {
                    ui.add_sized([150.0 * S, 32.0 * S], start)
                    },
                ).inner;
                if start_resp.clicked() && !running {
                    if let Err(e) = self.brow.start() {
                        self.brow.last_error = Some(format!("could not start: {e}"));
                    }
                    self.brow_last_frames = self.tele.frame_generations();
                }
                let abort = egui::Button::new(
                    egui::RichText::new("Abort")
                        .monospace()
                        .size(13.0 * S)
                        .strong()
                        .color(if running { TEXT1 } else { TEXT3 }),
                )
                // Never fall back to egui's default disabled-button surface: on
                // this dark theme it can render as a white rectangle with the
                // disabled label effectively invisible.
                .fill(INNER)
                .stroke(Stroke::new(1.0, if running { ERR } else { BORDER }));
                let abort_resp = ui.add_enabled_ui(running, |ui| {
                    ui.add_sized([110.0 * S, 32.0 * S], abort)
                }).inner;
                if abort_resp.clicked() {
                    self.brow.abort();
                }
            });

            if let Some(e) = &err {
                ui.add_space(SP2);
                ui.label(egui::RichText::new(format!("write error: {e}")).monospace().size(10.0 * S).color(ERR));
            }

            if self.brow.is_done() {
                ui.add_space(SP2);
                ui.label(egui::RichText::new(format!("saved to: {}", root.display()))
                    .monospace().size(10.0 * S).color(TEXT2));
                ui.label(prose("Next: Fit in app (fast, no Python) — or Train & bake for a full retrain — to turn this brow_data folder into your live eyebrow model."));
            }
        });

        ui.add_space(SP3);

        // --- Fit in app (pure-Rust head-fit, no Python) — the recommended lighter path ---
        self.brow_fit_ui(ui, cw);
        ui.add_space(SP3);

        // --- B-2: Train & bake (full external PyTorch retrain) -----------------------
        self.brow_train_ui(ui, cw);
                });
        });
        if reload_eyebrow_output {
            self.apply_and_reload();
        }
    }

    /// The "Fit in app (no Python)" card: re-fits the output head onto the captured brow_data
    /// using an existing brow.bin as a FROZEN conv backbone — pure Rust, seconds. The lighter,
    /// recommended path for a quick per-user recalibration; the full retrain lives below. On
    /// success the produced brow.bin is hot-loaded into the LIVE pipeline (no reconnect).
    fn brow_fit_ui(&mut self, ui: &mut egui::Ui, cw: f32) {
        // Consume a completed fit once: persist the model path + hot-swap it in.
        self.apply_fit_result_if_ready();

        let status = self.fitter.status();
        let running = status.is_running();

        // Preconditions: a captured dataset + a base model to reuse as the frozen backbone.
        let labels = self.brow.root().join("labels.csv");
        let has_labels = labels.is_file();
        // Base backbone = the configured model, else the Settings text field — whichever is a file.
        let backbone: Option<std::path::PathBuf> = self
            .config
            .brow_model_path()
            .filter(|p| p.is_file())
            .or_else(|| {
                let p = self.edit.brow_model.trim();
                if p.is_empty() {
                    None
                } else {
                    let p = std::path::PathBuf::from(p);
                    p.is_file().then_some(p)
                }
            });

        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.label(h3("Fit in app (no Python)"));
            ui.add_space(SP2);
            ui.label(prose(
                "Re-fits the output head to your captured brow_data using the existing eyebrow \
                 model as a frozen backbone — pure Rust, runs in seconds, no venv. Best for a \
                 quick per-user recalibration; use Train & bake below for a full retrain.",
            ));
            ui.add_space(SP2);

            // No base model yet? Say exactly how to get one, and render nothing else.
            let Some(backbone_bin) = backbone else {
                ui.label(egui::RichText::new(
                    "needs: a base eyebrow model (brow.bin) — bake once, or set Settings → Eyebrow model",
                ).monospace().size(10.0 * S).color(WARN));
                return;
            };

            let can_fit = has_labels
                && !running
                && !self.gaze_residual_capture.is_running()
                && !self.geometry_evidence_locked()
                && !self.wide.is_running()
                && !self.wide_fitter.is_running();
            ui.horizontal(|ui| {
                let btn = egui::Button::new(
                    egui::RichText::new("Fit in app").monospace().size(13.0 * S).strong().color(BG),
                ).fill(ACCENT);
                let resp = ui.add_enabled_ui(can_fit, |ui| {
                    ui.add_sized([160.0 * S, 32.0 * S], btn)
                }).inner;
                if resp.clicked() && can_fit {
                    let inputs = crate::brow_fitrun::FitInputs {
                        backbone_bin: backbone_bin.clone(),
                        brow_data_dir: self.brow.root().to_path_buf(),
                        seed: 0x5241_4942,
                    };
                    match self.fitter.start(inputs) {
                        Ok(()) => {
                            self.fit_applied = false;
                            self.events.push((now_hms(), "In-app eyebrow fit started".into(), ACCENT));
                        }
                        Err(e) => {
                            self.events.push((now_hms(), format!("Fit failed to start: {e}"), ERR));
                        }
                    }
                }
                // Stage / result label next to the button.
                match &status {
                    FitStatus::Idle => {}
                    FitStatus::Running { .. } => {
                        ui.label(egui::RichText::new("running — fitting head")
                            .monospace().size(11.0 * S).color(ACCENT));
                    }
                    FitStatus::Done { brow_bin, .. } => {
                        ui.label(egui::RichText::new(format!("done ✓  {}", brow_bin.display()))
                            .monospace().size(11.0 * S).color(OK));
                    }
                    FitStatus::Failed { msg, .. } => {
                        ui.label(egui::RichText::new(format!("failed — {msg}"))
                            .monospace().size(11.0 * S).color(ERR));
                    }
                }
            });

            // Precondition hint when the button is disabled (and not already running).
            if !has_labels && !running {
                ui.add_space(SP2);
                ui.label(egui::RichText::new("needs: capture a dataset (labels.csv) first")
                    .monospace().size(10.0 * S).color(WARN));
            }

            // Live log (bounded, scrollable, newest at the bottom).
            let log: &[String] = match &status {
                FitStatus::Running { log }
                | FitStatus::Done { log, .. }
                | FitStatus::Failed { log, .. } => log,
                FitStatus::Idle => &[],
            };
            if !log.is_empty() {
                ui.add_space(SP2);
                egui::ScrollArea::vertical()
                    .id_salt("brow_fit_log")
                    .max_height(150.0 * S)
                    .auto_shrink([false, true])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in log {
                            ui.label(egui::RichText::new(line).monospace().size(9.5 * S).color(TEXT2));
                        }
                    });
            }
        });
    }

    /// B-2: the "Train & bake" card on the Eyebrow-calibration tab. Drives the external
    /// PyTorch trainer + bake as a subprocess (via [`BrowTrainer`]) and, on success,
    /// hot-loads the produced brow.bin into the LIVE pipeline (no reconnect).
    fn brow_train_ui(&mut self, ui: &mut egui::Ui, cw: f32) {
        // Consume a completed run once: persist the model path + hot-swap it in.
        self.apply_train_result_if_ready();

        let status = self.trainer.status();
        // Preconditions for the button (report exactly what's missing, VRCFT-anti-opacity).
        let labels = self.brow.root().join("labels.csv");
        let has_labels = labels.is_file();
        let py = self.edit.python_exe.trim().to_string();
        let veb = self.edit.vr_eyebrow_dir.trim().to_string();
        let mut missing: Vec<&str> = Vec::new();
        if !has_labels {
            missing.push("capture a dataset (labels.csv)");
        }
        if py.is_empty() {
            missing.push("set the Python venv path (Settings)");
        }
        if veb.is_empty() {
            missing.push("set the vr_eyebrow dir (Settings)");
        }
        let running = status.is_running();
        let can_train = missing.is_empty()
            && !running
            && !self.gaze_residual_capture.is_running()
            && !self.geometry_evidence_locked()
            && !self.wide.is_running()
            && !self.wide_fitter.is_running();

        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.label(h3("Train & bake"));
            ui.add_space(SP2);
            ui.label(prose(
                "Trains your personal eyebrow model from the captured brow_data folder \
                 (PyTorch, in your configured venv) and bakes it into a live model — then \
                 loads it without restarting. Training runs as an external process.",
            ));
            ui.add_space(SP2);

            // The action button + a one-line state summary.
            ui.horizontal(|ui| {
                let btn = egui::Button::new(
                    egui::RichText::new("Train & bake")
                        .monospace()
                        .size(13.0 * S)
                        .strong()
                        .color(BG),
                )
                .fill(ACCENT);
                let resp = ui
                    .add_enabled_ui(can_train, |ui| ui.add_sized([160.0 * S, 32.0 * S], btn))
                    .inner;
                if resp.clicked() && can_train {
                    let inputs = TrainInputs {
                        python_exe: py.clone().into(),
                        vr_eyebrow_dir: veb.clone().into(),
                        brow_data_dir: self.brow.root().to_path_buf(),
                    };
                    match self.trainer.start(inputs) {
                        Ok(()) => {
                            self.train_applied = false;
                            self.events.push((
                                now_hms(),
                                "Eyebrow training started".into(),
                                ACCENT,
                            ));
                        }
                        Err(e) => {
                            self.events.push((
                                now_hms(),
                                format!("Train failed to start: {e}"),
                                ERR,
                            ));
                        }
                    }
                }
                // Stage / result label next to the button.
                match &status {
                    TrainStatus::Idle => {}
                    TrainStatus::Running { stage, .. } => {
                        ui.label(
                            egui::RichText::new(format!("running — {}", stage.label()))
                                .monospace()
                                .size(11.0 * S)
                                .color(ACCENT),
                        );
                    }
                    TrainStatus::Done { brow_bin, .. } => {
                        ui.label(
                            egui::RichText::new(format!("done ✓  {}", brow_bin.display()))
                                .monospace()
                                .size(11.0 * S)
                                .color(OK),
                        );
                    }
                    TrainStatus::Failed { msg, .. } => {
                        ui.label(
                            egui::RichText::new(format!("failed — {msg}"))
                                .monospace()
                                .size(11.0 * S)
                                .color(ERR),
                        );
                    }
                }
            });

            // What's missing (only when the button is disabled and not already running).
            if !missing.is_empty() && !running {
                ui.add_space(SP2);
                ui.label(
                    egui::RichText::new(format!("needs: {}", missing.join(" · ")))
                        .monospace()
                        .size(10.0 * S)
                        .color(WARN),
                );
            }

            // Live streamed log (bounded, scrollable, newest at the bottom).
            let log: &[String] = match &status {
                TrainStatus::Running { log, .. }
                | TrainStatus::Done { log, .. }
                | TrainStatus::Failed { log, .. } => log,
                TrainStatus::Idle => &[],
            };
            if !log.is_empty() {
                ui.add_space(SP2);
                egui::ScrollArea::vertical()
                    .id_salt("brow_train_log")
                    .max_height(150.0 * S)
                    .auto_shrink([false, true])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        for line in log {
                            ui.label(
                                egui::RichText::new(line)
                                    .monospace()
                                    .size(9.5 * S)
                                    .color(TEXT2),
                            );
                        }
                    });
            }
        });
    }

    /// If the trainer just finished, set `brow_model` to the baked `brow.bin`, persist the
    /// config, and hot-load it into the running pipeline (no reconnect). Guarded so it runs
    /// exactly once per completed run.
    fn apply_train_result_if_ready(&mut self) {
        if self.train_applied {
            return;
        }
        let TrainStatus::Done { brow_bin, .. } = self.trainer.status() else {
            return;
        };
        self.train_applied = true;
        self.hot_load_brow_model(&brow_bin);
    }

    /// Persist a freshly-produced `brow.bin` as the active eyebrow model and hot-swap it into
    /// the LIVE pipeline (no device reconnect, no port re-bind). Shared by the Train & bake and
    /// the in-app Fit paths.
    fn hot_load_brow_model(&mut self, brow_bin: &std::path::Path) {
        // Persist the produced model path so it survives a restart.
        let path_str = brow_bin.to_string_lossy().into_owned();
        self.config.assets.brow_model = Some(path_str.clone());
        self.edit.brow_model = path_str;
        let _ = self.config.save(&crate::config::config_path());
        // Hot-swap the model into the LIVE pipeline: load the BrowNet and hand it to the
        // running ML thread via the shared handle (no device reconnect, no port re-bind).
        match crate::ml::brow_net::BrowNet::load(brow_bin) {
            Ok(net) => {
                let out_dim = net.out_dim();
                // set_brow flips `tele.brow_loaded` on the existing telemetry Arc (the UI
                // already reads it), so no tele/engine rebuild is needed — the dashboard's
                // ML node starts showing live brow L/R on the next frame.
                self.pipeline.set_brow(Some(net));
                self.events.push((
                    now_hms(),
                    format!("Eyebrow model loaded live (out_dim={out_dim})"),
                    OK,
                ));
            }
            Err(e) => {
                self.events
                    .push((now_hms(), format!("Baked model invalid: {e}"), ERR));
            }
        }
    }

    /// If the in-app fit just finished, persist + hot-load the produced brow.bin. Guarded so it
    /// runs exactly once per completed fit (mirrors [`Self::apply_train_result_if_ready`]).
    fn apply_fit_result_if_ready(&mut self) {
        if self.fit_applied {
            return;
        }
        let FitStatus::Done { brow_bin, .. } = self.fitter.status() else {
            return;
        };
        self.fit_applied = true;
        self.hot_load_brow_model(&brow_bin);
    }

    fn settings(&mut self, ui: &mut egui::Ui) {
        let discovery_poll = self
            .sranipal_discovery_job
            .as_ref()
            .map(mpsc::Receiver::try_recv);
        match discovery_poll {
            Some(Ok(Some(found))) => {
                self.edit.sranipal_dir = found.root.to_string_lossy().into_owned();
                self.reload_msg = Some((
                    format!("Found SRanipal from {}. Use Apply & reload.", found.source),
                    OK,
                ));
                self.sranipal_discovery_job = None;
            }
            Some(Ok(None)) => {
                self.reload_msg = Some((
                    "SRanipal was not found. Choose sr_runtime.exe manually.".into(),
                    WARN,
                ));
                self.sranipal_discovery_job = None;
            }
            Some(Err(mpsc::TryRecvError::Disconnected)) => {
                self.reload_msg = Some(("SRanipal search stopped unexpectedly.".into(), WARN));
                self.sranipal_discovery_job = None;
            }
            Some(Err(mpsc::TryRecvError::Empty)) | None => {}
        }

        if self.gaze_residual_capture.is_running() {
            ui.label(
                egui::RichText::new(
                    "Settings are locked while the nine-point research target is being recorded.",
                )
                .monospace()
                .color(WARN),
            );
            ui.disable();
        }
        let (gutter, cw) = stage_metrics(ui.ctx());
        let mut do_reload = false;

        ui.horizontal_top(|ui| {
            ui.add_space(gutter);
            ui.vertical(|ui| {
                ui.set_width(cw);

                // Keep the only destructive/engine-level action visible while the
                // individual setting groups scroll underneath it.
                card().show(ui, |ui| {
                    ui.set_width(cw - 2.0 * CARD_PAD);
                    ui.horizontal(|ui| {
                        ui.label(h3("Settings"));
                        ui.add_space(SP2);
                        if ui.button("Apply & reload").clicked() {
                            do_reload = true;
                        }
                        if let Some((msg, col)) = &self.reload_msg {
                            ui.label(egui::RichText::new(msg).size(11.0 * S).color(*col));
                        }
                    });
                    ui.label(prose(
                        "Live switches save immediately. Device, model, path, and local-output changes use Apply & reload.",
                    ));
                });
                ui.add_space(SP3);

                egui::ScrollArea::vertical()
                    .id_salt("settings_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_width(cw);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            ui.label(h3("Tracking & device"));
                            ui.add_space(SP2);
                                    ui.label(prose("The controls normally used when changing HMD or tuning live output."));
                                    ui.add_space(SP2);

                                     ui.horizontal(|ui| {
                                         ui.label(label("Headset"));
                                        #[cfg(feature = "psvr2-only")]
                                        ui.label(num("psvr2 (fixed by this build)"));
                                        #[cfg(feature = "xr5-only")]
                                        ui.label(num("pimax_xr5 (fixed by this build)"));
                                        #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
                                        egui::ComboBox::from_id_salt("device_select")
                                            .selected_text(device_option_label(&self.edit.device))
                                            .show_ui(ui, |ui| {
                                                for &d in STANDARD_DEVICE_OPTIONS {
                                                    ui.selectable_value(
                                                        &mut self.edit.device,
                                                        d.to_string(),
                                                        device_option_label(d),
                                                    );
                                                }
                                             });
                                     });

                                     ui.add_space(SP2);
                                     ui.horizontal(|ui| {
                                         ui.label(label("Eyelid inference"));
                                         let previous = self.config.ui.eyelid_inference_backend;
                                         egui::ComboBox::from_id_salt(
                                             "settings_eyelid_inference_backend",
                                         )
                                         .selected_text(previous.label())
                                         .show_ui(ui, |ui| {
                                             for backend in [
                                                 crate::config::EyelidInferenceBackend::Auto,
                                                 crate::config::EyelidInferenceBackend::Gpu,
                                                 crate::config::EyelidInferenceBackend::Cpu,
                                             ] {
                                                 ui.selectable_value(
                                                     &mut self
                                                         .config
                                                         .ui
                                                         .eyelid_inference_backend,
                                                     backend,
                                                     backend.label(),
                                                 );
                                             }
                                         });
                                         if self.config.ui.eyelid_inference_backend != previous {
                                             do_reload = true;
                                         }
                                         let report = &self.pipeline.eyelid_backend_report;
                                         let active = report.active_label();
                                         ui.label(
                                             egui::RichText::new(format!("ACTIVE {active}"))
                                                 .monospace()
                                                 .size(9.0 * S)
                                                 .color(if active == "GPU" { OK } else { TEXT3 }),
                                         )
                                         .on_hover_text(format!(
                                             "{}{}",
                                             report.note,
                                             report
                                                 .adapter
                                                 .as_deref()
                                                 .map(|adapter| format!("\nAdapter: {adapter}"))
                                                 .unwrap_or_default()
                                         ));
                                     });
                                     ui.label(label(
                                         "Auto validates GPU output against CPU and chooses the faster safe backend. Changing it reconnects the HMD once.",
                                     ));

                                     let mut right_left_head =
                                         self.pipeline.uses_right_eye_left_head();
                                     let route_response = ui
                                         .add_enabled_ui(self.tele.ml_loaded, |ui| {
                                             ui.checkbox(
                                                 &mut right_left_head,
                                                 "Use LEFT model head for RIGHT eyelid",
                                             )
                                             .on_hover_text(
                                                 "Mirrors the physical right eye into EyeNet's left input and uses the left openness/squeeze head. Gaze, camera identity, Wide and brow are unchanged.",
                                             )
                                         })
                                         .inner;
                                     if route_response.changed() {
                                         let previous = self.pipeline.uses_right_eye_left_head();
                                         let device = self.pipeline.device_key.clone();
                                         self.config
                                             .set_right_eye_left_head(&device, right_left_head);
                                         match self.config.save(&crate::config::config_path()) {
                                             Ok(()) => {
                                                 self.pipeline
                                                     .set_right_eye_left_head(right_left_head);
                                                 self.reload_msg = Some((
                                                     "RIGHT eyelid source changed live; keep both eyes relaxed while its baseline re-centers."
                                                         .into(),
                                                     OK,
                                                 ));
                                             }
                                             Err(error) => {
                                                 self.config
                                                     .set_right_eye_left_head(&device, previous);
                                                 self.reload_msg = Some((
                                                     format!(
                                                         "Could not save eyelid source: {error}"
                                                     ),
                                                     ERR,
                                                 ));
                                             }
                                         }
                                     }

                                     #[cfg(feature = "xr5-only")]
                                    {
                                        ui.horizontal(|ui| {
                                            ui.label(label("XR5 EyeWide source"));
                                            ui.label(num(self.pipeline.selected_wide_source().as_str()));
                                        });
                                        ui.label(label(
                                            "Change this live in Calibration > XR5 EyeWide; it never needs a camera reload.",
                                        ));
                                        let mut use_combined =
                                            self.edit.gaze_source == GazeSource::Combined;
                                        if ui
                                            .checkbox(
                                                &mut use_combined,
                                                "XR5: use EyeChip combined gaze (steadier)",
                                            )
                                            .on_hover_text(
                                                "Default off. Applies after reload; does not affect openness or non-XR5 headsets.",
                                            )
                                            .changed()
                                        {
                                            self.edit.gaze_source = if use_combined {
                                                GazeSource::Combined
                                            } else {
                                                GazeSource::PerEye
                                            };
                                        }
                                        ui.label(label(
                                            "Combined mode trades natural near-focus convergence for lower per-eye jitter. Re-run gaze Center after changing it.",
                                        ));
                                    }

                                    ui.add_space(SP2);
                                    let mut filter_samples = self
                                        .be
                                        .as_ref()
                                        .map(|s| {
                                            s.filter_samples.load(Ordering::Relaxed)
                                        })
                                        .unwrap_or(u32::from(
                                            self.config.output.vrcft_filter_samples,
                                        ))
                                        .min(30);
                                    ui.label(label("VRCFT openness low-pass"));
                                    ui.horizontal(|ui| {
                                        let changed = ui
                                            .add(
                                                egui::Slider::new(
                                                    &mut filter_samples,
                                                    0..=30,
                                                )
                                                .suffix(" samples")
                                                .show_value(true),
                                            )
                                            .on_hover_text(
                                                "0/1 = pass-through. Larger values are smoother but add latency.",
                                            )
                                            .changed();
                                        let detail = if filter_samples <= 1 {
                                            "OFF".to_string()
                                        } else {
                                            let lag_ms = (filter_samples - 1) as f32
                                                * 1000.0
                                                / (2.0 * 120.0);
                                            format!("~{lag_ms:.0} ms delay")
                                        };
                                        ui.label(
                                            egui::RichText::new(detail)
                                                .monospace()
                                                .size(10.0 * S)
                                                .color(TEXT3),
                                        );
                                        if changed {
                                            self.config.output.vrcft_filter_samples =
                                                filter_samples as u8;
                                            if let Some(status) = &self.be {
                                                status.filter_samples.store(
                                                    filter_samples,
                                                    Ordering::Relaxed,
                                                );
                                            }
                                            let _ = self
                                                .config
                                                .save(&crate::config::config_path());
                                            self.events.push((
                                                now_hms(),
                                                format!(
                                                    "VRCFT low-pass set to {filter_samples} samples"
                                                ),
                                                ACCENT,
                                            ));
                                        }
                                    });

                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                                    ui.label(h3("Connection & models"));
                                    ui.add_space(SP2);
                                    ui.label(prose(
                                        "Eye model and optional trained models. These normally stay unchanged after setup.",
                                    ));
                                    ui.add_space(SP2);
                                    if let Some(action) = sranipal_path_row(
                                        ui,
                                        &mut self.edit.sranipal_dir,
                                        self.sranipal_discovery_job.is_some(),
                                    ) {
                                        match action {
                                            SranipalPathAction::StartAuto => {
                                                let (sender, receiver) = mpsc::channel();
                                                let context = ui.ctx().clone();
                                                std::thread::spawn(move || {
                                                    let found =
                                                        crate::sranipal_discovery::discover();
                                                    let _ = sender.send(found);
                                                    context.request_repaint();
                                                });
                                                self.sranipal_discovery_job = Some(receiver);
                                                self.reload_msg = Some((
                                                    "Searching for sr_runtime.exe...".into(),
                                                    ACCENT,
                                                ));
                                            }
                                            SranipalPathAction::Message(result) => {
                                                self.reload_msg = Some(match result {
                                                    Ok(message) => (message, OK),
                                                    Err(message) => (message, WARN),
                                                });
                                            }
                                        }
                                    }
                                    settings_path_row(
                                        ui,
                                        "Eyebrow model (brow.bin)",
                                        &mut self.edit.brow_model,
                                        false,
                                    );
                                    #[cfg(feature = "xr5-only")]
                                    {
                                        settings_path_row(
                                            ui,
                                            "XR5 EyeWide model (wide.bin)",
                                            &mut self.edit.wide_model,
                                            false,
                                        );
                                    }

                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            ui.label(h3("Eye image output"));
                            ui.add_space(SP2);
                                    let mut eye_image_http =
                                        self.config.output.eye_image_http;
                                    if ui
                                        .checkbox(
                                            &mut eye_image_http,
                                            "Enable local eye-image HTTP output",
                                        )
                                        .on_hover_text(
                                            "Publishes the latest mapped left/right camera images as local MJPEG streams. JPEG encoding runs only while a client is connected.",
                                        )
                                        .changed()
                                    {
                                        self.config.output.eye_image_http = eye_image_http;
                                        do_reload = true;
                                    }
                                    ui.horizontal(|ui| {
                                        ui.label(label("Local address"));
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self.edit.eye_image_host,
                                            )
                                            .desired_width(150.0 * S),
                                        );
                                        ui.add(
                                            egui::DragValue::new(
                                                &mut self.edit.eye_image_port,
                                            )
                                            .range(1..=u16::MAX),
                                        );
                                    });
                                    let display_host =
                                        if self.edit.eye_image_host.trim().contains(':') {
                                            format!("[{}]", self.edit.eye_image_host.trim())
                                        } else {
                                            self.edit.eye_image_host.trim().to_owned()
                                        };
                                    let origin = format!(
                                        "http://{}:{}/",
                                        display_host, self.edit.eye_image_port
                                    );
                                    ui.horizontal_wrapped(|ui| {
                                        if let Some(server) =
                                            self.pipeline.eye_image_http.as_ref()
                                        {
                                            let running =
                                                format!("http://{}/", server.local_addr());
                                            ui.label(
                                                egui::RichText::new("RUNNING")
                                                    .monospace()
                                                    .size(10.0 * S)
                                                    .strong()
                                                    .color(OK),
                                            );
                                            ui.hyperlink_to(&running, &running);
                                        } else {
                                            ui.label(label("Preview"));
                                            ui.add_enabled_ui(false, |ui| {
                                                ui.hyperlink_to(&origin, &origin);
                                            });
                                        }
                                    });
                                    if eye_image_http
                                        && self.pipeline.eye_image_http.is_none()
                                    {
                                        ui.label(
                                            egui::RichText::new(
                                                "Eye image output is not running. Check the Console for a bind/address error.",
                                            )
                                            .monospace()
                                            .size(10.0 * S)
                                            .color(ERR),
                                        );
                                    }
                                    ui.label(label(
                                        "Streams: /left.mjpg and /right.mjpg. Only 127.0.0.1 or ::1 is accepted. Use Apply & reload after editing the address.",
                                    ));


                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            ui.label(h3("Eye mapping"));
                            ui.add_space(SP2);
                                    ui.label(prose(&format!(
                                        "Orientation saved for the active device: {}",
                                        self.pipeline.device_key
                                    )));
                                    ui.add_space(SP2);

                                    let mut swap =
                                        self.pipeline.swap_eyes.load(Ordering::Relaxed);
                                    if ui
                                        .checkbox(&mut swap, "Swap left/right eye streams")
                                        .changed()
                                    {
                                        self.pipeline.swap_eyes.store(swap, Ordering::Relaxed);
                                        self.persist_mapping();
                                    }
                                    ui.label(label(
                                        "For Pimax units with reversed L/R camera images. Swaps camera, eyelid, pupil and gaze together.",
                                    ));

                                    ui.add_space(SP2);
                                    egui::CollapsingHeader::new("Advanced orientation")
                                        .default_open(false)
                                        .show(ui, |ui| {
                                            ui.label(prose(
                                                "Driver and ML diagnostics. These do not fix reversed left/right eye streams.",
                                            ));

                                            let mut flip = self
                                                .pipeline
                                                .flip_image
                                                .load(Ordering::Relaxed);
                                            if ui
                                                .checkbox(
                                                    &mut flip,
                                                    "Flip camera image horizontally",
                                                )
                                                .changed()
                                            {
                                                self.pipeline
                                                    .flip_image
                                                    .store(flip, Ordering::Relaxed);
                                                self.persist_mapping();
                                            }

                                            let mut fgx = self
                                                .pipeline
                                                .flip_gaze_x
                                                .load(Ordering::Relaxed);
                                            if ui
                                                .checkbox(
                                                    &mut fgx,
                                                    "Flip gaze horizontal direction",
                                                )
                                                .changed()
                                            {
                                                self.pipeline
                                                    .flip_gaze_x
                                                    .store(fgx, Ordering::Relaxed);
                                                self.persist_mapping();
                                            }

                                            ui.add_space(SP2);
                                            ui.label(
                                                egui::RichText::new("Per-eye ML handedness")
                                                    .size(11.0 * S)
                                                    .color(WARN),
                                            );
                                            let mut mml = self
                                                .pipeline
                                                .ml_mirror_l
                                                .load(Ordering::Relaxed);
                                            if ui
                                                .checkbox(&mut mml, "Mirror LEFT eye for ML")
                                                .changed()
                                            {
                                                self.pipeline
                                                    .ml_mirror_l
                                                    .store(mml, Ordering::Relaxed);
                                                self.persist_mapping();
                                            }
                                            let mut mmr = self
                                                .pipeline
                                                .ml_mirror_r
                                                .load(Ordering::Relaxed);
                                            if ui
                                                .checkbox(&mut mmr, "Mirror RIGHT eye for ML")
                                                .changed()
                                            {
                                                self.pipeline
                                                    .ml_mirror_r
                                                    .store(mmr, Ordering::Relaxed);
                                                self.persist_mapping();
                                            }
                                        });

                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            ui.label(h3("Training tools"));
                            ui.add_space(SP2);
                                    ui.label(prose(
                                        "Only required by the full Python Train & bake workflow.",
                                    ));
                                    ui.add_space(SP2);
                                    settings_path_row(
                                        ui,
                                        "Python with PyTorch",
                                        &mut self.edit.python_exe,
                                        false,
                                    );
                                    settings_path_row(
                                        ui,
                                        "vr_eyebrow project folder",
                                        &mut self.edit.vr_eyebrow_dir,
                                        true,
                                    );

                        });
                        ui.add_space(SP3);
                    });
            });
        });

        if do_reload {
            self.apply_and_reload();
        }
    }

    fn persist_mapping(&mut self) {
        let dev = self.pipeline.device_key.clone();
        let requested_mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        // A mapping checkbox is another persistence path. Discard an unsaved geometry
        // preview first, then re-apply the checkbox value that triggered this call.
        self.restore_geometry_preview(false);
        self.pipeline
            .ml_mirror_l
            .store(requested_mirrors[0], Ordering::Relaxed);
        self.pipeline
            .ml_mirror_r
            .store(requested_mirrors[1], Ordering::Relaxed);
        // Mirror is persisted beside crop/rotation as a tri-state so an older config's
        // missing field inherits the XR5 preset, while a new explicit "off" remains off.
        let mut geometry = *self.pipeline.geometry.lock().unwrap();
        geometry[0].mirror_h = Some(self.pipeline.ml_mirror_l.load(Ordering::Relaxed));
        geometry[1].mirror_h = Some(self.pipeline.ml_mirror_r.load(Ordering::Relaxed));
        *self.pipeline.geometry.lock().unwrap() = geometry;
        self.config.set_geometry(&dev, geometry);
        let m = crate::config::EyeMapping {
            swap_eyes: self.pipeline.swap_eyes.load(Ordering::Relaxed),
            flip_image: self.pipeline.flip_image.load(Ordering::Relaxed),
            swap_gaze_eyes: None,
            flip_gaze_x: self.pipeline.flip_gaze_x.load(Ordering::Relaxed),
            ml_mirror_l: self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            ml_mirror_r: self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        };
        self.config.set_mapping(&dev, m);
        let _ = self.config.save(&crate::config::config_path());
    }

    /// Apply the edited asset paths: write sranibro.toml, tear down the running
    /// engine (frees the EyeChip + TCP port), and rebuild it in-process from the
    /// new config. On failure the message is shown and the user can fix + retry.
    fn apply_and_reload(&mut self) {
        if self.reload_job.is_some() {
            return;
        }
        if self.geometry_recording_export_job.is_some() {
            let message =
                "Wait for the calibration recording ZIP to finish saving before reloading.";
            self.reload_msg = Some((message.into(), WARN));
            self.dream_air_msg = Some((message.into(), WARN));
            return;
        }
        if self.gaze_residual_capture.is_running()
            || (self.gaze_residual_capture.is_done() && self.gaze_residual_recording_path.is_none())
            || self.geometry_fitter.is_running()
            || self.photometric_fitter.is_running()
            || self.endpoint_fitter.is_running()
            || self.gaze_eyelid_fitter.is_running()
            || self.wink_fitter.is_running()
            || self.blink_timing_fitter.is_running()
            || self.reseat_assist.is_active()
            || self
                .unified_calibration
                .as_ref()
                .is_some_and(|run| !matches!(run.stage, UnifiedStage::Complete))
        {
            let message =
                "Finish or cancel the gaze-direction eyelid recording/analysis before reloading.";
            self.reload_msg = Some((message.into(), ERR));
            self.dream_air_msg = Some((message.into(), ERR));
            return;
        }
        if self.gaze_center_capture.take().is_some() {
            self.gaze_center_msg = Some(("Center capture cancelled by reload".into(), TEXT2));
        }
        self.restore_geometry_preview(false);
        self.geometry_capture.abort();
        self.gaze_residual_capture.abort();
        self.gaze_residual_snapshot = None;
        self.gaze_residual_export_attempted = false;
        self.gaze_residual_recording_path = None;
        self.wide.abort();
        self.geometry_capture_baseline = None;
        self.geometry_capture_filters = None;
        self.photometric_capture_baseline = None;
        self.calibration_capture_purpose = None;
        self.reset_geometry_recording_export();
        self.geometry_fitter.cancel();
        self.photometric_fitter.cancel();
        self.endpoint_fitter.cancel();
        self.gaze_eyelid_fitter.cancel();
        self.wink_fitter.cancel();
        self.blink_timing_fitter.cancel();
        self.reseat_assist.stop();
        self.unified_calibration = None;
        self.close_calibration_detail();
        self.geometry_rollback = None;
        self.photometric_rollback = None;
        self.geometry_unvalidated_ack = false;
        let norm = |s: &str| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        };
        self.config.assets.sranipal_dir = norm(&self.edit.sranipal_dir);
        self.config.assets.brow_model = norm(&self.edit.brow_model);
        self.config.assets.wide_model = norm(&self.edit.wide_model);
        self.config.assets.python_exe = norm(&self.edit.python_exe);
        self.config.assets.vr_eyebrow_dir = norm(&self.edit.vr_eyebrow_dir);
        #[cfg(feature = "psvr2-only")]
        {
            self.edit.device = "psvr2".to_string();
        }
        #[cfg(feature = "xr5-only")]
        {
            self.edit.device = "pimax_xr5".to_string();
        }
        self.config.hmd.device = self.edit.device.trim().to_string();
        self.config.hmd.wide_source = self.edit.wide_source;
        self.config
            .set_gaze_source("pimax_xr5", self.edit.gaze_source);
        self.config.output.osc_host = self.edit.osc_host.trim().to_string();
        self.config.output.osc_port = self.edit.osc_port;
        self.config.output.eye_image_host = self.edit.eye_image_host.trim().to_string();
        self.config.output.eye_image_port = self.edit.eye_image_port;

        let path = crate::config::config_path();
        if let Err(e) = self.config.save(&path) {
            self.reload_msg = Some((format!("save failed: {e}"), ERR));
            return;
        }

        // Tear down first: releases the WinUSB handle and TCP port 5555 so the
        // rebuild can re-acquire them cleanly (the user opted into live re-init).
        let shutdown = self.pipeline.take_shutdown();
        let config = self.config.clone();
        #[cfg(windows)]
        let gpu = self.gpu_context.clone();
        self.reload_msg = Some(("Reloading tracking…".into(), TEXT2));
        self.reload_job = Some(std::thread::spawn(move || {
            shutdown();
            #[cfg(windows)]
            {
                crate::engine::build_engine_with_gpu(&config, gpu)
            }
            #[cfg(not(windows))]
            {
                crate::engine::build_engine(&config)
            }
        }));
    }

    fn poll_reload(&mut self) {
        if !self
            .reload_job
            .as_ref()
            .is_some_and(|worker| worker.is_finished())
        {
            return;
        }
        let rebuilt = self.reload_job.take().unwrap().join().unwrap_or_else(|_| {
            Err(std::io::Error::other(
                "Tracking reload worker stopped unexpectedly",
            ))
        });
        self.reload_backdrop.clear();
        match rebuilt {
            Ok(eng) => {
                if let Some(direct) = eng.starvr_direct_mode {
                    self.config.hmd.starvr_direct = direct;
                }
                let startup_notice = eng.startup_notice.clone();
                self.pipeline = eng.pipeline;
                self.tele = self.pipeline.tele.clone();
                self.wear_memory =
                    WearMemory::load(wear_memory_context(&self.config, &self.pipeline));
                self.wear_thumbnails.clear();
                self.wear_memory_draft = WearingMemoryDraft::default();
                self.wear_memory_matching_suspended = false;
                self.wear_closed_capture = None;
                self.wear_response_before_edit = None;
                self.wear_baseline_before_edit = None;
                self.eyelid_response_preview = None;
                self.be = eng.be_status;
                self.tex_l = None;
                self.tex_r = None;
                self.last_eye_texture_upload = Instant::now() - Duration::from_secs(1);
                self.last_eye_texture_generation = [0; 2];
                self.last = [0; 5];
                self.last_t = Instant::now();
                self.events.push((now_hms(), "Assets reloaded".into(), OK));
                self.reload_msg = Some((startup_notice.unwrap_or_else(|| "reloaded ✓".into()), OK));
            }
            Err(e) => {
                eprintln!("[ui] reload failed: {e}");
                self.reload_msg = Some((format!("reload failed: {e}"), ERR));
            }
        }
    }
}

fn calibration_fit_inputs_equal(
    frozen: crate::core::eye_state::CalibStore,
    live: crate::core::eye_state::CalibStore,
) -> bool {
    let eye_equal = |frozen: crate::core::eye_state::CalibSnapshot,
                     live: crate::core::eye_state::CalibSnapshot| {
        frozen.baseline.to_bits() == live.baseline.to_bits()
            && frozen.blink_depth.to_bits() == live.blink_depth.to_bits()
            && frozen.mid_anchor.to_bits() == live.mid_anchor.to_bits()
            && frozen.learned_once == live.learned_once
            && frozen.endpoint_locked == live.endpoint_locked
            && frozen.endpoint_calibrated_unix == live.endpoint_calibrated_unix
    };
    eye_equal(frozen.left, live.left) && eye_equal(frozen.right, live.right)
}

fn unified_session_id() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn accepted_eye_text(eyes: [bool; 2]) -> &'static str {
    match eyes {
        [true, true] => "both eyes",
        [true, false] => "left eye",
        [false, true] => "right eye",
        [false, false] => "no eyes",
    }
}

// ---------------------------------------------------------------------------
// Widgets + icons
// ---------------------------------------------------------------------------

enum SranipalPathAction {
    StartAuto,
    Message(Result<String, String>),
}

fn sranipal_path_row(
    ui: &mut egui::Ui,
    value: &mut String,
    searching: bool,
) -> Option<SranipalPathAction> {
    ui.label(label("SRanipal runtime"));
    let mut outcome = None;
    ui.horizontal(|ui| {
        let find_width = 116.0 * S;
        let browse_width = 92.0 * S;
        let edit_width =
            (ui.available_width() - find_width - browse_width - 2.0 * SP2).max(160.0 * S);
        ui.add_sized(
            [edit_width, 24.0 * S],
            egui::TextEdit::singleline(value).hint_text("folder containing sr_runtime.exe"),
        );
        if ui
            .add_enabled_ui(!searching, |ui| {
                ui.add_sized(
                    [find_width, 24.0 * S],
                    egui::Button::new(if searching {
                        "Searching..."
                    } else {
                        "Find automatically"
                    }),
                )
            })
            .inner
            .on_hover_text(
                "Checks a running sr_runtime.exe, installed-program records, common locations, Downloads and Desktop.",
            )
            .clicked()
        {
            outcome = Some(SranipalPathAction::StartAuto);
        }
        if ui
            .add_sized([browse_width, 24.0 * S], egui::Button::new("Choose EXE..."))
            .on_hover_text("Select sr_runtime.exe; SRanibro derives the model folder.")
            .clicked()
        {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("SRanipal runtime", &["exe"])
                .set_file_name("sr_runtime.exe")
                .pick_file()
            {
                let is_runtime = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.eq_ignore_ascii_case("sr_runtime.exe"));
                outcome = Some(SranipalPathAction::Message(if !is_runtime {
                    Err("Select sr_runtime.exe itself, not another executable.".into())
                } else {
                    match crate::sranipal_discovery::resolve(&path) {
                        Some(root) => {
                            *value = root.to_string_lossy().into_owned();
                            Ok("SRanipal runtime and EyePrediction model found. Use Apply & reload."
                                .into())
                        }
                        None => Err(
                            "The folder containing that sr_runtime.exe has no compatible EyePrediction model."
                                .into(),
                        ),
                    }
                }));
            }
        }
    });
    ui.label(label(
        "Use the folder containing sr_runtime.exe. SRanibro reads the EyePrediction model inside it.",
    ));
    ui.add_space(4.0 * S);
    outcome
}

fn settings_path_row(ui: &mut egui::Ui, name: &str, value: &mut String, pick_directory: bool) {
    ui.label(label(name));
    ui.horizontal(|ui| {
        let button_width = 78.0 * S;
        let edit_width = (ui.available_width() - button_width - SP2).max(160.0 * S);

        ui.add_sized(
            [edit_width, 24.0 * S],
            egui::TextEdit::singleline(value).hint_text("(not set)"),
        );
        if ui
            .add_sized([button_width, 24.0 * S], egui::Button::new("Browse..."))
            .clicked()
        {
            let dialog = rfd::FileDialog::new();
            let picked = if pick_directory {
                dialog.pick_folder()
            } else {
                dialog.pick_file()
            };
            if let Some(path) = picked {
                *value = path.to_string_lossy().into_owned();
            }
        }
    });
    ui.add_space(4.0 * S);
}

/// (left gutter, content column width) for the central panel, in points.
/// Derived from the configured window width and the real ppp — never from
/// egui's `available_*`, which over-reports the surface on this display.
/// `content_w` is the full visible width; subtract the nav rail and the
/// central panel's own margins to get the usable column.
fn stage_metrics(ctx: &egui::Context) -> (f32, f32) {
    let cw = content_w(ctx) - NAV_W - 2.0 * MAIN_PAD;
    (0.0, cw)
}

/// Local wall-clock "HH:MM:SS" for event timestamps (captured when the event fires).
#[cfg(windows)]
fn now_hms() -> String {
    use windows_sys::Win32::System::SystemInformation::GetLocalTime;
    let mut st = unsafe { std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>() };
    unsafe { GetLocalTime(&mut st) };
    format!("{:02}:{:02}:{:02}", st.wHour, st.wMinute, st.wSecond)
}
/// Non-Windows fallback (compile-only; the app runs on Windows) — UTC time-of-day.
#[cfg(not(windows))]
fn now_hms() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{:02}:{:02}:{:02}", (s / 3600) % 24, (s / 60) % 60, s % 60)
}

fn log_line(ui: &mut egui::Ui, tag: &str, col: Color32, module: &str, msg: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0 * S;
        ui.label(
            egui::RichText::new(tag)
                .monospace()
                .size(10.0 * S)
                .color(col),
        );
        ui.label(
            egui::RichText::new(format!("{module:<11}"))
                .monospace()
                .size(10.0 * S)
                .color(TEXT3),
        );
        ui.label(
            egui::RichText::new(msg)
                .monospace()
                .size(10.0 * S)
                .color(TEXT3),
        );
    });
}

/// Per-line tint for the Console tab: errors stand out (ERR), device-stage lines
/// (`[xr5]`/`[vr4]`) get the ACCENT so pipeline chatter is easy to follow, everything
/// else is the muted-but-legible TEXT2. Case-insensitive on the error keywords.
fn log_color(line: &str) -> Color32 {
    let lo = line.to_ascii_lowercase();
    if lo.contains("error") || lo.contains("fail") || lo.contains("panic") || line.contains("!!") {
        ERR
    } else if line.contains("[xr5]") || line.contains("[vr4]") {
        ACCENT
    } else {
        TEXT2
    }
}

fn bar(ui: &mut egui::Ui, w: f32, h: f32, frac: f32, col: Color32) {
    let (rect, _) = ui.allocate_exact_size(vec2(w, h), Sense::hover());
    draw_bar(ui.painter(), rect, frac, col);
}

/// Draw a horizontal fill bar into `rect` (rounded track + clipped fill). No allocation, so
/// it can be composed (e.g. a main bar + a thin raw bar in one column).
fn draw_bar(p: &egui::Painter, rect: Rect, frac: f32, col: Color32) {
    let rad = (rect.height() * 0.33).min(4.0 * S);
    p.rect_filled(rect, rad, INNER);
    let f = frac.clamp(0.0, 1.0);
    let fw = rect.width() * f;
    if fw > 0.5 {
        // Left corners rounded; right edge square (like a clipped track fill),
        // except when full where it matches the track's right radius.
        let r = if f > 0.995 { rad } else { 0.0 };
        let rounding = egui::Rounding {
            nw: rad,
            sw: rad,
            ne: r,
            se: r,
        };
        let fill = Rect::from_min_size(rect.min, vec2(fw, rect.height()));
        p.rect_filled(fill, rounding, col);
    }
}

/// A corrected bar with a THIN raw (model-output) bar just below it, in a `w x row_h` column
/// — so you can see whether a value is shaped at the MODEL or in POST-PROCESSING. `marker`
/// (0..1) draws a small red vertical tick on the RAW bar (used to show the learned baseline
/// on the wide row: if it creeps up toward the raw-openness fill, the calibration is drifting).
fn bar_with_raw(
    ui: &mut egui::Ui,
    w: f32,
    row_h: f32,
    bar_h: f32,
    corrected: f32,
    raw: f32,
    marker: Option<f32>,
    fill: Color32,
) {
    let (rect, _) = ui.allocate_exact_size(vec2(w, row_h), Sense::hover());
    let gap = 2.0 * S;
    let raw_h = (bar_h * 0.34).clamp(3.0 * S, 7.0 * S);
    let top = rect.top() + ((row_h - (bar_h + gap + raw_h)) * 0.5).max(0.0);
    let main = Rect::from_min_size(pos2(rect.left(), top), vec2(w, bar_h));
    let rawr = Rect::from_min_size(pos2(rect.left(), top + bar_h + gap), vec2(w, raw_h));
    let p = ui.painter();
    draw_bar(p, main, corrected.clamp(0.0, 1.0), fill);
    draw_bar(p, rawr, raw.clamp(0.0, 1.0), lerp_color(fill, INNER, 0.5));
    if let Some(m) = marker {
        let mx = rawr.left() + rawr.width() * m.clamp(0.0, 1.0);
        p.line_segment(
            [
                pos2(mx, rawr.top() - 1.5 * S),
                pos2(mx, rawr.bottom() + 1.5 * S),
            ],
            Stroke::new(1.5 * S, Color32::from_rgb(235, 70, 70)),
        );
    }
}

/// Center-origin (bipolar) bar: `val` in [-1,1], 0 = center. Fills rightward for
/// positive, leftward for negative; |val| scales the half-width. Same track
/// background/rounding as `bar`. The inner-edge corners (at center) stay square so
/// the fill reads as growing out from the middle line.
/// Draw a center-origin (bipolar) fill bar into `rect`. No allocation (composable).
fn draw_bar_bipolar(p: &egui::Painter, rect: Rect, val: f32, col: Color32) {
    let rad = (rect.height() * 0.33).min(4.0 * S);
    p.rect_filled(rect, rad, INNER);
    let v = val.clamp(-1.0, 1.0);
    let cx = rect.center().x;
    let half = rect.width() * 0.5;
    let fw = half * v.abs();
    if fw > 0.5 {
        if v > 0.0 {
            let outer = if v > 0.995 { rad } else { 0.0 };
            let rounding = egui::Rounding {
                nw: 0.0,
                sw: 0.0,
                ne: outer,
                se: outer,
            };
            let fill = Rect::from_min_max(pos2(cx, rect.top()), pos2(cx + fw, rect.bottom()));
            p.rect_filled(fill, rounding, col);
        } else {
            let outer = if v < -0.995 { rad } else { 0.0 };
            let rounding = egui::Rounding {
                nw: outer,
                sw: outer,
                ne: 0.0,
                se: 0.0,
            };
            let fill = Rect::from_min_max(pos2(cx - fw, rect.top()), pos2(cx, rect.bottom()));
            p.rect_filled(fill, rounding, col);
        }
    }
}

/// Bipolar corrected bar + a thin bipolar raw bar below it (see `bar_with_raw`).
fn bar_bipolar_with_raw(
    ui: &mut egui::Ui,
    w: f32,
    row_h: f32,
    bar_h: f32,
    corrected: f32,
    raw: f32,
    fill: Color32,
) {
    let (rect, _) = ui.allocate_exact_size(vec2(w, row_h), Sense::hover());
    let gap = 2.0 * S;
    let raw_h = (bar_h * 0.34).clamp(3.0 * S, 7.0 * S);
    let top = rect.top() + ((row_h - (bar_h + gap + raw_h)) * 0.5).max(0.0);
    let main = Rect::from_min_size(pos2(rect.left(), top), vec2(w, bar_h));
    let rawr = Rect::from_min_size(pos2(rect.left(), top + bar_h + gap), vec2(w, raw_h));
    let p = ui.painter();
    draw_bar_bipolar(p, main, corrected.clamp(-1.0, 1.0), fill);
    draw_bar_bipolar(p, rawr, raw.clamp(-1.0, 1.0), lerp_color(fill, INNER, 0.5));
}

/// 3-tier value color (mockup): exactly-0 dim, small-but-present mid, larger accent.
fn grad_col(v: f32) -> Color32 {
    if v < 0.005 {
        TEXT3
    } else if v < 0.05 {
        TEXT2
    } else {
        ACCENT
    }
}

/// One ML table row: 64*S label | bar | 38*S value | bar | 38*S value, gap 6*S.
/// `row_h`/`bar_h` let rows grow to fill a stretched card.
#[allow(clippy::too_many_arguments)]
fn ml_row(
    ui: &mut egui::Ui,
    name: &str,
    l: f32,
    r: f32,
    raw_l: f32,
    raw_r: f32,
    marker_l: Option<f32>,
    marker_r: Option<f32>,
    fill: Color32,
    bar_w: f32,
    lcol: Color32,
    rcol: Color32,
    row_h: f32,
    bar_h: f32,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0 * S;
        let (lab, _) = ui.allocate_exact_size(vec2(64.0 * S, row_h), Sense::hover());
        ui.painter().text(
            lab.left_center(),
            Align2::LEFT_CENTER,
            name,
            FontId::monospace(10.0 * S),
            TEXT1,
        );
        bar_with_raw(ui, bar_w, row_h, bar_h, l, raw_l, marker_l, fill);
        let (lv, _) = ui.allocate_exact_size(vec2(38.0 * S, row_h), Sense::hover());
        ui.painter().text(
            lv.right_center(),
            Align2::RIGHT_CENTER,
            format!("{l:.2}"),
            FontId::monospace(11.0 * S),
            lcol,
        );
        bar_with_raw(ui, bar_w, row_h, bar_h, r, raw_r, marker_r, fill);
        let (rv, _) = ui.allocate_exact_size(vec2(38.0 * S, row_h), Sense::hover());
        ui.painter().text(
            rv.right_center(),
            Align2::RIGHT_CENTER,
            format!("{r:.2}"),
            FontId::monospace(11.0 * S),
            rcol,
        );
    });
}

/// Like `ml_row` but the L/R column bars are center-origin/bipolar (`bar_bipolar`):
/// `l`/`r` in [-1,1], 0 = center, positive extends RIGHT, negative extends LEFT.
#[allow(clippy::too_many_arguments)]
fn ml_row_bipolar(
    ui: &mut egui::Ui,
    name: &str,
    l: f32,
    r: f32,
    raw_l: f32,
    raw_r: f32,
    fill: Color32,
    bar_w: f32,
    lcol: Color32,
    rcol: Color32,
    row_h: f32,
    bar_h: f32,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0 * S;
        let (lab, _) = ui.allocate_exact_size(vec2(64.0 * S, row_h), Sense::hover());
        ui.painter().text(
            lab.left_center(),
            Align2::LEFT_CENTER,
            name,
            FontId::monospace(10.0 * S),
            TEXT1,
        );
        bar_bipolar_with_raw(ui, bar_w, row_h, bar_h, l, raw_l, fill);
        let (lv, _) = ui.allocate_exact_size(vec2(38.0 * S, row_h), Sense::hover());
        ui.painter().text(
            lv.right_center(),
            Align2::RIGHT_CENTER,
            format!("{l:.2}"),
            FontId::monospace(11.0 * S),
            lcol,
        );
        bar_bipolar_with_raw(ui, bar_w, row_h, bar_h, r, raw_r, fill);
        let (rv, _) = ui.allocate_exact_size(vec2(38.0 * S, row_h), Sense::hover());
        ui.painter().text(
            rv.right_center(),
            Align2::RIGHT_CENTER,
            format!("{r:.2}"),
            FontId::monospace(11.0 * S),
            rcol,
        );
    });
}

/// Hover card for a pipeline node: title + label/value rows. Uses a fixed label
/// column (NOT right_to_left, which over-reports width in a tooltip and clips).
fn node_detail_card(ui: &mut egui::Ui, title: &str, ok: bool, rows: &[(&'static str, String)]) {
    ui.spacing_mut().item_spacing.y = 3.0 * S;
    ui.horizontal(|ui| {
        let (d, _) = ui.allocate_exact_size(vec2(8.0 * S, 8.0 * S), Sense::hover());
        ui.painter()
            .circle_filled(d.center(), 3.0 * S, if ok { OK } else { WARN });
        ui.label(
            egui::RichText::new(title)
                .monospace()
                .size(11.0 * S)
                .strong()
                .color(TEXT1),
        );
    });
    ui.add_space(3.0 * S);
    for (k, v) in rows {
        ui.horizontal(|ui| {
            let (lr, _) = ui.allocate_exact_size(vec2(86.0 * S, 14.0 * S), Sense::hover());
            ui.painter().text(
                lr.left_center(),
                Align2::LEFT_CENTER,
                *k,
                FontId::monospace(10.0 * S),
                TEXT3,
            );
            ui.label(
                egui::RichText::new(v)
                    .monospace()
                    .size(10.0 * S)
                    .color(TEXT1),
            );
        });
    }
}

/// Paint one pipeline node into an exact rect (icon over name over dot+value).
/// Used by the branched pipeline, which positions nodes with explicit geometry.
#[allow(clippy::too_many_arguments)]
fn pipeline_node(
    ui: &egui::Ui,
    rect: Rect,
    icon: Icon,
    name: &str,
    sub: &str,
    dot: Color32,
    icol: Color32,
    vcol: Color32,
    fill: Color32,
    border: Color32,
) {
    let p = ui.painter();
    // `fill` masks the connectors that pass under the node; SURFACE == the card, so a
    // healthy node reads as floating icon+label (no boxed tile). A transparent border
    // is skipped — only the first-broken stage gets a visible container.
    p.rect_filled(rect, R_INNER, fill);
    if border != Color32::TRANSPARENT {
        p.rect_stroke(rect, R_INNER, Stroke::new(1.0, border));
    }
    let cx = rect.center().x;
    let icon_cy = rect.top() + 5.0 * S + 6.5 * S;
    draw_icon(p, icon, pos2(cx, icon_cy), 13.0 * S, icol);
    let name_y = icon_cy + 6.5 * S + 2.5 * S;
    p.text(
        pos2(cx, name_y),
        Align2::CENTER_TOP,
        name,
        FontId::monospace(9.0 * S),
        TEXT2,
    );
    let gw = ui.fonts(|f| {
        f.layout_no_wrap(sub.to_string(), FontId::monospace(10.0 * S), vcol)
            .size()
            .x
    });
    let left = cx - (10.0 * S + gw) / 2.0;
    let val_cy = name_y + 9.0 * S + 6.0 * S;
    p.circle_filled(pos2(left + 3.0 * S, val_cy), 3.0 * S, dot);
    p.text(
        pos2(left + 10.0 * S, val_cy),
        Align2::LEFT_CENTER,
        sub,
        FontId::monospace(10.0 * S),
        vcol,
    );
}

#[allow(clippy::too_many_arguments)]
/// Composite one heatmap pixel: the grayscale eye under a diverging colour overlay whose
/// alpha grows with |delta|/vmax. Warm (red) = occluding/glinting here RAISED the signed
/// delta, cool (blue) = lowered it; transparent where the model barely reacts.
fn heat_color(gray: u8, delta: f32, vmax: f32) -> Color32 {
    let base = gray as f32;
    let v = (delta / vmax).clamp(-1.0, 1.0);
    let a = v.abs() * 0.6;
    let (cr, cg, cb) = if v >= 0.0 {
        (255.0, 70.0, 40.0)
    } else {
        (40.0, 130.0, 255.0)
    };
    let mix = |c: f32| (((1.0 - a) * base + a * c).clamp(0.0, 255.0)) as u8;
    Color32::from_rgb(mix(cr), mix(cg), mix(cb))
}

/// Compact two-way source selector used in the ML diagnostics footer. Its bounds are
/// explicit so it cannot push into the DROP readout when the card is narrow.
fn brow_source_switch(
    ui: &mut egui::Ui,
    rect: Rect,
    brow_legacy: &mut bool,
    brow_model_loaded: bool,
) {
    let split_x = rect.center().x;
    let legacy_rect = Rect::from_min_max(rect.min, pos2(split_x, rect.max.y));
    let estimate_rect = Rect::from_min_max(pos2(split_x, rect.min.y), rect.max);
    let legacy_response = ui.interact(
        legacy_rect,
        Id::new("ml_brow_source_legacy"),
        Sense::click(),
    );
    let estimate_response = ui.interact(
        estimate_rect,
        Id::new("ml_brow_source_estimate"),
        if brow_model_loaded {
            Sense::click()
        } else {
            Sense::hover()
        },
    );

    if legacy_response.clicked() {
        *brow_legacy = true;
    }
    if brow_model_loaded && estimate_response.clicked() {
        *brow_legacy = false;
    }
    if legacy_response.hovered() || estimate_response.hovered() && brow_model_loaded {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let painter = ui.painter();
    painter.rect_filled(rect, 4.0 * S, INNER);
    let legacy_fill = if *brow_legacy {
        ACCENT_BG
    } else if legacy_response.hovered() {
        SURFACE
    } else {
        INNER
    };
    let estimate_fill = if !*brow_legacy {
        ACCENT_BG
    } else if estimate_response.hovered() && brow_model_loaded {
        SURFACE
    } else {
        INNER
    };
    painter.rect_filled(legacy_rect.shrink(1.0), 3.0 * S, legacy_fill);
    painter.rect_filled(estimate_rect.shrink(1.0), 3.0 * S, estimate_fill);
    painter.rect_stroke(rect, 4.0 * S, Stroke::new(1.0, BORDER_STRONG));
    painter.line_segment(
        [
            pos2(split_x, rect.top() + 2.0),
            pos2(split_x, rect.bottom() - 2.0),
        ],
        Stroke::new(1.0, BORDER),
    );
    painter.text(
        legacy_rect.center(),
        Align2::CENTER_CENTER,
        "LEGACY",
        FontId::monospace(7.5 * S),
        if *brow_legacy { ACCENT } else { TEXT2 },
    );
    painter.text(
        estimate_rect.center(),
        Align2::CENTER_CENTER,
        "ESTIMATE",
        FontId::monospace(7.5 * S),
        if !brow_model_loaded {
            TEXT3.gamma_multiply(0.45)
        } else if !*brow_legacy {
            ACCENT
        } else {
            TEXT2
        },
    );

    legacy_response
        .on_hover_text("Derives eyebrow motion from EyeWide and EyeSquint in the VRCFT module.");
    estimate_response.on_hover_text(if brow_model_loaded {
        "Uses the independent eyebrow model trained from the Python dataset. Python is not run during tracking."
    } else {
        "Estimate is unavailable because no independent eyebrow model is loaded."
    });
}

/// Conventional pill switch for optional eyebrow stereo synchronization. The
/// label and pill share one hit target so the compact footer remains easy to use.
fn brow_lr_sync_switch(ui: &mut egui::Ui, hit_rect: Rect, switch_rect: Rect, value: &mut bool) {
    let response = ui.interact(hit_rect, Id::new("ml_brow_lr_sync"), Sense::click());
    if response.clicked() {
        *value = !*value;
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }

    let painter = ui.painter();
    let fill = if *value { ACCENT_BG } else { INNER };
    let outline = if *value { ACCENT } else { BORDER_STRONG };
    painter.rect_filled(switch_rect, switch_rect.height() * 0.5, fill);
    painter.rect_stroke(
        switch_rect,
        switch_rect.height() * 0.5,
        Stroke::new(1.0, outline),
    );
    let knob_radius = switch_rect.height() * 0.33;
    let knob_x = if *value {
        switch_rect.right() - switch_rect.height() * 0.5
    } else {
        switch_rect.left() + switch_rect.height() * 0.5
    };
    painter.circle_filled(
        pos2(knob_x, switch_rect.center().y),
        knob_radius,
        if *value { ACCENT } else { TEXT3 },
    );

    response.on_hover_text(if *value {
        "Brow L/R sync is ON: the two independently processed brow values are averaged and emitted equally."
    } else {
        "Brow L/R sync is OFF: left and right eyebrow expressions remain independent."
    });
}

fn ml_params_card(
    ui: &mut egui::Ui,
    w: f32,
    min_inner: f32,
    results: &[crate::core::types::EyeResult; 2],
    raw: &[[f32; 5]; 2],
    brow_raw: [f32; 2],
    baselines: [f32; 2],
    drop: f32,
    ml_rate: f32,
    brow_lr_sync: &mut bool,
    brow_legacy: &mut bool,
    brow_model_loaded: bool,
) {
    let inner = w - 2.0 * CARD_PAD;
    card().show(ui, |ui| {
        ui.set_width(inner);
        ui.set_min_height(min_inner);
        // Header: title left, "L / R · <ml>/s" (live rate) pinned right.
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new("ML PARAMETERS")
                    .monospace()
                    .size(10.0 * S)
                    .color(TEXT2),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("L / R · {ml_rate:.0}/s"))
                        .monospace()
                        .size(9.0 * S)
                        .color(TEXT3),
                );
            });
        });
        ui.add_space(11.0 * S);
        // Columns: label 64 | bar 1fr | value 38 | bar 1fr | value 38, gap 6.
        let bar_w = ((inner - 64.0 * S - 2.0 * 38.0 * S - 4.0 * 6.0 * S) / 2.0).max(40.0 * S);
        // Grow the 4 ML data rows (openness/wide/squeeze/brow —
        // pupil is NOT an ML output, it lives in the eye-cameras card) + taller
        // bars to fill a stretched card. The math sums EXACTLY to min_inner
        // (header + lr + 4*row + 4*gap + footer); the footer pad below clamps at
        // 0 to avoid overflow.
        let lr_h = 11.0 * S;
        let row_gap = 8.0 * S;
        // Footer = separator(1) + gap(10*S) + one flat diagnostics row(20*S). No
        // nested tiles (was bordered mini-cells inside this bordered card).
        let footer_h = 30.0 * S + 1.0;
        let header_block = ui.cursor().top() - ui.min_rect().top();
        // egui won't render a row shorter than its interact_size.y (~18 logical pts),
        // so PREDICT every row at >= that height. 5 row_gaps actually render (before the
        // LEFT/RIGHT header row + between the 4 data rows), not 4. Under-predicting any
        // of these overshoots row_h and pushes the card past min_inner — the trailing
        // `pad` only absorbs OVER-prediction, never overflow.
        let row_min = 18.0_f32;
        let region = (min_inner - header_block - footer_h - lr_h.max(row_min) - 5.0 * row_gap)
            .max(4.0 * row_min);
        let row_h = (region / 4.0).clamp(row_min, 52.0 * S);
        let bar_h = (row_h * 0.4).clamp(9.0 * S, 24.0 * S);
        ui.spacing_mut().item_spacing.y = row_gap;
        // Header row: LEFT / RIGHT centered over their bar columns.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0 * S;
            ui.allocate_exact_size(vec2(64.0 * S, lr_h), Sense::hover());
            let (lh, _) = ui.allocate_exact_size(vec2(bar_w, lr_h), Sense::hover());
            ui.painter().text(
                lh.center(),
                Align2::CENTER_CENTER,
                "LEFT",
                FontId::monospace(9.0 * S),
                TEXT3,
            );
            ui.allocate_exact_size(vec2(38.0 * S, lr_h), Sense::hover());
            let (rh, _) = ui.allocate_exact_size(vec2(bar_w, lr_h), Sense::hover());
            ui.painter().text(
                rh.center(),
                Align2::CENTER_CENTER,
                "RIGHT",
                FontId::monospace(9.0 * S),
                TEXT3,
            );
            ui.allocate_exact_size(vec2(38.0 * S, lr_h), Sense::hover());
        });
        // Blink is folded into openness: a blinking eye reads ~0 and its value
        // turns amber, so no separate blink indicator is needed.
        let l_open = if results[0].blink { WARN } else { OK };
        let r_open = if results[1].blink { WARN } else { OK };
        // Thin RAW bar under each gauge = the model's direct output (ch1 openness, ch3
        // squeeze, raw brow); wide has no raw channel so it shows the raw openness it
        // derives from. Lets you see whether a value is shaped at the model or in post.
        ml_row(
            ui,
            "openness",
            results[0].openness,
            results[1].openness,
            raw[0][1],
            raw[1][1],
            None,
            None,
            OK,
            bar_w,
            l_open,
            r_open,
            row_h,
            bar_h,
        );
        // wide's raw bar = raw openness, with a red tick at the learned BASELINE per eye:
        // if the tick creeps up toward the raw fill while you hold wide, the calibration is
        // drifting (wide will collapse); if it stays put, the wide-freeze fix is working.
        ml_row(
            ui,
            "wide",
            results[0].wide,
            results[1].wide,
            raw[0][1],
            raw[1][1],
            Some(baselines[0]),
            Some(baselines[1]),
            ACCENT,
            bar_w,
            grad_col(results[0].wide),
            grad_col(results[1].wide),
            row_h,
            bar_h,
        );
        ml_row(
            ui,
            "squeeze",
            results[0].squeeze,
            results[1].squeeze,
            raw[0][3],
            raw[1][3],
            None,
            None,
            ACCENT,
            bar_w,
            grad_col(results[0].squeeze),
            grad_col(results[1].squeeze),
            row_h,
            bar_h,
        );
        // Eyebrow (signed brow in [-1,1]) shown as ONE center-origin bar: 0 = center,
        // positive extends RIGHT, negative extends LEFT. 0 when no brow model is loaded.
        let brow_l = results[0].brow;
        let brow_r = results[1].brow;
        ml_row_bipolar(
            ui,
            "brow",
            brow_l,
            brow_r,
            brow_raw[0],
            brow_raw[1],
            ACCENT,
            bar_w,
            grad_col(brow_l.abs()),
            grad_col(brow_r.abs()),
            row_h,
            bar_h,
        );
        // Footer (separator + mini-row) pinned to the bottom of the stretched
        // card. Measure content height from the cursor, NOT min_rect (set_min_height
        // already inflated min_rect to min_inner, which would zero out the pad).
        // Zero item-spacing so the footer height is exactly footer_h.
        ui.spacing_mut().item_spacing.y = 0.0;
        let used = ui.cursor().top() - ui.min_rect().top();
        let pad = (min_inner - used - footer_h).max(0.0);
        ui.add_space(pad);
        let (sep, _) = ui.allocate_exact_size(vec2(inner, 1.0), Sense::hover());
        ui.painter().rect_filled(sep, 0.0, BORDER);
        ui.add_space(10.0 * S);
        // Flat diagnostics + live switches row. DROP = % of 120Hz emit cycles that
        // ran late (0 = keeping up). Explicit rectangles keep every control clear of
        // DROP even at the card's minimum supported width.
        let (row, _) = ui.allocate_exact_size(vec2(inner, 20.0 * S), Sense::hover());
        let cy = row.center().y;
        let dc = if drop < 0.5 { OK } else { WARN };
        ui.painter().text(
            pos2(row.left(), cy),
            Align2::LEFT_CENTER,
            "DROP",
            FontId::monospace(9.0 * S),
            TEXT2,
        );
        ui.painter().text(
            pos2(row.left() + 42.0 * S, cy),
            Align2::LEFT_CENTER,
            format!("{drop:.1}%"),
            FontId::monospace(10.0 * S),
            dc,
        );

        // Right-aligned controls: eyebrow source plus optional stereo output sync.
        let switch_size = vec2(28.0 * S, 16.0 * S);
        let switch_rect =
            Rect::from_center_size(pos2(row.right() - switch_size.x * 0.5, cy), switch_size);
        let sync_label_right = switch_rect.left() - 4.0 * S;
        ui.painter().text(
            pos2(sync_label_right, cy),
            Align2::RIGHT_CENTER,
            "BROW L/R SYNC",
            FontId::monospace(8.0 * S),
            TEXT3,
        );
        let sync_hit = Rect::from_min_max(
            pos2(sync_label_right - 76.0 * S, row.top()),
            pos2(row.right(), row.bottom()),
        );
        brow_lr_sync_switch(ui, sync_hit, switch_rect, brow_lr_sync);

        let source_size = vec2(96.0 * S, 18.0 * S);
        let source_right = sync_hit.left() - 8.0 * S;
        let source_rect =
            Rect::from_center_size(pos2(source_right - source_size.x * 0.5, cy), source_size);
        brow_source_switch(ui, source_rect, brow_legacy, brow_model_loaded);
        ui.painter().text(
            pos2(source_rect.left() - 4.0 * S, cy),
            Align2::RIGHT_CENTER,
            "BROW",
            FontId::monospace(8.0 * S),
            if brow_model_loaded { TEXT3 } else { WARN },
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn eye_cams_card(
    ui: &mut egui::Ui,
    w: f32,
    min_inner: f32,
    frames: &[Option<EyeFrame>; 2],
    ml_frames: &[Option<Vec<u8>>; 2],
    net_view: &mut bool,
    pupil: &[(f32, bool); 2],
    cam_rates: [f32; 2],
    eye_w: u32,
    eye_h: u32,
    tex_l: &mut Option<egui::TextureHandle>,
    tex_r: &mut Option<egui::TextureHandle>,
    upload_textures: bool,
    preview_enabled: &mut bool,
    ctx: &egui::Context,
) -> bool {
    let inner = w - 2.0 * CARD_PAD;
    // Resolution label: the live frame's real dims when streaming, else the device
    // profile's nominal — per-HMD (VR4/StarVR 200x200, Varjo higher), not hardcoded.
    let (dw, dh) = frames
        .iter()
        .flatten()
        .next()
        .map(|frame| (frame.width, frame.height))
        .unwrap_or((eye_w, eye_h));
    let n = crate::ml::preprocess::DST as u32;
    card().show(ui, |ui| {
        ui.set_width(inner);
        ui.set_min_height(min_inner);
        // Header: a small gear (opens ML-input settings) at the far left, then the title.
        // The dashboard preview switch affects presentation only; tracking stays live.
        let clicked = ui
            .horizontal(|ui| {
                let (grect, gresp) = ui.allocate_exact_size(vec2(18.0 * S, 14.0 * S), Sense::click());
                let hov = gresp.hovered();
                if hov {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                draw_icon(ui.painter(), Icon::Gear, grect.center(), 15.0 * S, if hov { TEXT1 } else { ACCENT });
                let gr = gresp.on_hover_text("ML input settings");
                ui.add_space(3.0 * S);
                ui.label(egui::RichText::new("EYE CAMERAS").monospace().size(10.0 * S).color(TEXT2));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let state_text = if *preview_enabled { "ON" } else { "OFF" };
                    let state_color = if *preview_enabled { ACCENT } else { TEXT3 };
                    let preview_toggle = ui
                        .add(egui::Button::new(
                            egui::RichText::new(state_text)
                                .monospace()
                                .size(9.0 * S)
                                .strong()
                                .color(state_color),
                        )
                        .small()
                        .fill(INNER))
                        .on_hover_text(
                            "Dashboard eye-image preview only. Tracking and recordings stay live.",
                        );
                    if preview_toggle.clicked() {
                        *preview_enabled = !*preview_enabled;
                    }
                    ui.add_space(4.0 * S);
                    ui.label(
                        egui::RichText::new("PREVIEW")
                            .monospace()
                            .size(9.0 * S)
                            .color(TEXT3),
                    );
                    if *preview_enabled {
                        ui.add_space(4.0 * S);
                        // Source toggle: RAW cameras vs the exact image sent to the eye
                        // net (all filters + geometry, shown un-mirrored).
                        let txt = if *net_view { "NET" } else { "RAW" };
                        let color = if *net_view { ACCENT } else { TEXT3 };
                        let tr = ui
                            .add(egui::Button::new(
                                egui::RichText::new(txt)
                                    .monospace()
                                    .size(9.0 * S)
                                    .strong()
                                    .color(color),
                            )
                            .small()
                            .fill(INNER))
                            .on_hover_text(
                                "Toggle: raw cameras / the exact image sent to the eye net (correct orientation)",
                            );
                        if tr.clicked() {
                            *net_view = !*net_view;
                        }
                        ui.add_space(4.0 * S);
                        let res = if *net_view {
                            format!("{n}x{n} NET")
                        } else {
                            format!("{dw}x{dh} IR")
                        };
                        ui.label(
                            egui::RichText::new(res)
                                .monospace()
                                .size(9.0 * S)
                                .color(TEXT3),
                        );
                    }
                });
                gr.clicked()
            })
            .inner;
        ui.add_space(9.0 * S);
        let box_w = ((inner - 9.0 * S) / 2.0).floor();
        // Keep the original two-eye layout while preview is off. Only the image
        // texture disappears; L/R identity, camera rate, spacing, and card geometry
        // stay unchanged, so toggling preview cannot make the dashboard jump.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 9.0 * S;
            if *preview_enabled && *net_view {
                let fl = ml_frames[0].as_ref().map(|p| (n, n, p.as_slice()));
                let fr = ml_frames[1].as_ref().map(|p| (n, n, p.as_slice()));
                eye_cam_box(ui, "L", fl, pupil[0], cam_rates[0], tex_l, upload_textures, true, ctx, "eye_l", box_w);
                eye_cam_box(ui, "R", fr, pupil[1], cam_rates[1], tex_r, upload_textures, true, ctx, "eye_r", box_w);
            } else if *preview_enabled {
                eye_cam_box(ui, "L", frames[0].as_ref().map(EyeFrame::view), pupil[0], cam_rates[0], tex_l, upload_textures, true, ctx, "eye_l", box_w);
                eye_cam_box(ui, "R", frames[1].as_ref().map(EyeFrame::view), pupil[1], cam_rates[1], tex_r, upload_textures, true, ctx, "eye_r", box_w);
            } else {
                eye_cam_box(ui, "L", None, pupil[0], cam_rates[0], tex_l, false, false, ctx, "eye_l", box_w);
                eye_cam_box(ui, "R", None, pupil[1], cam_rates[1], tex_r, false, false, ctx, "eye_r", box_w);
            }
        });
        clicked
    })
    .inner
}

#[allow(clippy::too_many_arguments)]
fn eye_cam_box(
    ui: &mut egui::Ui,
    label: &str,
    frame: Option<(u32, u32, &[u8])>,
    pupil: (f32, bool),
    rate: f32,
    slot: &mut Option<egui::TextureHandle>,
    upload_texture: bool,
    show_texture: bool,
    ctx: &egui::Context,
    name: &str,
    w: f32,
) {
    ui.vertical(|ui| {
        ui.set_width(w);
        if upload_texture {
            if let Some((fw, fh, px)) = frame {
                // Build the texture at the frame's NATIVE resolution (per-HMD); the box
                // displays it scaled to the square slot.
                let (iw, ih) = (fw as usize, fh as usize);
                if iw > 0 && ih > 0 && px.len() >= iw * ih {
                    let mut img = egui::ColorImage::new([iw, ih], Color32::BLACK);
                    for i in 0..iw * ih {
                        img.pixels[i] = Color32::from_gray(px[i]);
                    }
                    match slot {
                        Some(h) => h.set(img, egui::TextureOptions::LINEAR),
                        None => {
                            *slot = Some(ctx.load_texture(name, img, egui::TextureOptions::LINEAR))
                        }
                    }
                }
            }
        }
        // Allocate an exact w x w rect FIRST, then paint into it. egui::Image with
        // a texture otherwise ignores its size and inflates to phantom available
        // width, which would widen the whole card.
        let sz = vec2(w, w);
        let (rect, _) = ui.allocate_exact_size(sz, Sense::hover());
        if show_texture {
            if let Some(h) = slot.as_ref() {
                egui::Image::new(egui::load::SizedTexture::new(h.id(), sz))
                    .rounding(R_BOX)
                    .paint_at(ui, rect);
            } else {
                ui.painter().rect_filled(rect, R_BOX, INNER);
                ui.painter().text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    "no signal",
                    FontId::monospace(10.0 * S),
                    TEXT3,
                );
            }
        } else {
            // Preview OFF deliberately looks like the normal L/R camera slots with
            // their video layer removed, not one merged placeholder panel.
            ui.painter().rect_filled(rect, R_BOX, INNER);
        }
        // Box frame + overlays (L/R top-left, fps bottom-right). Each overlay gets a
        // dark scrim chip so it stays legible over bright IR (token contrast can't be
        // guaranteed against live imagery).
        ui.painter()
            .rect_stroke(rect, R_BOX, Stroke::new(1.0, BORDER));
        let p = ui.painter().clone();
        let chip = |anchor: Pos2, align: Align2, txt: &str, sz: f32, col: Color32| {
            let fid = FontId::monospace(sz);
            let g = ui.fonts(|f| f.layout_no_wrap(txt.to_string(), fid.clone(), col));
            let ts = g.size();
            // Place the text rect per alignment, then a padded dark backing behind it.
            let min = pos2(
                anchor.x
                    - if align.x() == egui::Align::Max {
                        ts.x
                    } else {
                        0.0
                    },
                anchor.y
                    - if align.y() == egui::Align::Max {
                        ts.y
                    } else {
                        0.0
                    },
            );
            let pad = vec2(4.0 * S, 2.0 * S);
            p.rect_filled(
                Rect::from_min_size(min - pad, ts + 2.0 * pad),
                4.0 * S,
                Color32::from_black_alpha(150),
            );
            p.text(anchor, align, txt, fid, col);
        };
        let rate_txt = format!("{rate:.0}/s");
        chip(
            rect.left_top() + vec2(6.0 * S, 5.0 * S),
            Align2::LEFT_TOP,
            label,
            9.0 * S,
            TEXT1,
        );
        chip(
            rect.right_bottom() + vec2(-6.0 * S, -5.0 * S),
            Align2::RIGHT_BOTTOM,
            &rate_txt,
            8.0 * S,
            TEXT3,
        );
        // Under-box line: pupil Ø only (blink lives in the ML card now).
        ui.add_space(5.0 * S);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0 * S;
            ui.label(
                egui::RichText::new("pupil")
                    .monospace()
                    .size(9.0 * S)
                    .color(TEXT2),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let (mm, valid) = pupil;
                let txt = if valid {
                    format!("Ø {mm:.1} mm")
                } else {
                    "Ø — mm".to_string()
                };
                ui.label(
                    egui::RichText::new(txt)
                        .monospace()
                        .size(9.0 * S)
                        .color(if valid { ACCENT } else { TEXT3 }),
                );
            });
        });
    });
}

#[derive(Clone, Copy)]
enum Icon {
    Eye,
    Activity,
    Sliders,
    Gear,
    Usb,
    Camera,
    Cpu,
    Stack,
    Broadcast,
    Reload,
    Console,
}

/// Hand-drawn vector glyphs (Tabler-equivalents) so the console matches the
/// mockup's icon set without bundling an icon font. Coordinates are in a unit
/// box [0,1] mapped onto the `size` square centered at `c`.
fn draw_icon(p: &egui::Painter, icon: Icon, c: Pos2, size: f32, col: Color32) {
    let st = Stroke::new(1.6, col);
    let thin = Stroke::new(1.3, col);
    let bold = Stroke::new(1.9, col);
    let pt = |ux: f32, uy: f32| pos2(c.x + (ux - 0.5) * size, c.y + (uy - 0.5) * size);
    let line = |pts: Vec<Pos2>, s: Stroke| {
        p.add(egui::Shape::line(pts, s));
    };
    let arc = |ucx: f32, ucy: f32, ur: f32, a0: f32, a1: f32| -> Vec<Pos2> {
        let n = 16;
        (0..=n)
            .map(|i| {
                let a = a0 + (a1 - a0) * (i as f32 / n as f32);
                pt(ucx + ur * a.cos(), ucy + ur * a.sin())
            })
            .collect()
    };
    match icon {
        Icon::Eye => {
            line(
                vec![
                    pt(0.10, 0.50),
                    pt(0.30, 0.30),
                    pt(0.50, 0.27),
                    pt(0.70, 0.30),
                    pt(0.90, 0.50),
                ],
                st,
            );
            line(
                vec![
                    pt(0.10, 0.50),
                    pt(0.30, 0.70),
                    pt(0.50, 0.73),
                    pt(0.70, 0.70),
                    pt(0.90, 0.50),
                ],
                st,
            );
            p.circle_stroke(pt(0.5, 0.5), size * 0.16, st);
            p.circle_filled(pt(0.5, 0.5), size * 0.09, col);
        }
        Icon::Activity => {
            line(
                vec![
                    pt(0.05, 0.50),
                    pt(0.30, 0.50),
                    pt(0.40, 0.22),
                    pt(0.52, 0.80),
                    pt(0.62, 0.50),
                    pt(0.95, 0.50),
                ],
                bold,
            );
        }
        Icon::Sliders => {
            let half = size * 0.46;
            for (i, kt) in [(-1.0, 0.7), (0.0, 0.35), (1.0, 0.55)] {
                let y = c.y + i * size * 0.30;
                p.line_segment(
                    [pos2(c.x - half, y), pos2(c.x + half, y)],
                    Stroke::new(1.5, col),
                );
                let kx = c.x - half + (2.0 * half) * kt;
                p.circle_filled(pos2(kx, y), size * 0.11, col);
            }
        }
        Icon::Gear => {
            let r = size * 0.32;
            p.circle_stroke(c, r, st);
            p.circle_filled(c, size * 0.12, col);
            for k in 0..8 {
                let a = k as f32 / 8.0 * TAU;
                let d = vec2(a.cos(), a.sin());
                p.line_segment([c + d * r, c + d * (r + size * 0.16)], st);
            }
        }
        Icon::Usb => {
            line(vec![pt(0.5, 0.16), pt(0.5, 0.90)], st);
            line(vec![pt(0.40, 0.27), pt(0.5, 0.13), pt(0.60, 0.27)], st);
            p.circle_filled(pt(0.5, 0.90), size * 0.06, col);
            line(vec![pt(0.5, 0.46), pt(0.30, 0.46), pt(0.30, 0.33)], thin);
            p.rect_filled(
                Rect::from_center_size(pt(0.30, 0.30), vec2(size * 0.11, size * 0.11)),
                0.0,
                col,
            );
            line(vec![pt(0.5, 0.62), pt(0.70, 0.62), pt(0.70, 0.45)], thin);
            p.circle_filled(pt(0.70, 0.42), size * 0.06, col);
        }
        Icon::Camera => {
            p.rect_stroke(
                Rect::from_min_max(pt(0.12, 0.36), pt(0.88, 0.82)),
                size * 0.06,
                st,
            );
            line(
                vec![
                    pt(0.34, 0.36),
                    pt(0.39, 0.25),
                    pt(0.54, 0.25),
                    pt(0.59, 0.36),
                ],
                thin,
            );
            p.circle_stroke(pt(0.5, 0.59), size * 0.16, st);
            p.circle_filled(pt(0.5, 0.59), size * 0.05, col);
        }
        Icon::Cpu => {
            p.rect_stroke(
                Rect::from_min_max(pt(0.28, 0.28), pt(0.72, 0.72)),
                size * 0.04,
                st,
            );
            p.rect_stroke(
                Rect::from_min_max(pt(0.40, 0.40), pt(0.60, 0.60)),
                0.0,
                thin,
            );
            for x in [0.38, 0.5, 0.62] {
                line(vec![pt(x, 0.28), pt(x, 0.18)], thin);
                line(vec![pt(x, 0.72), pt(x, 0.82)], thin);
            }
            for y in [0.38, 0.5, 0.62] {
                line(vec![pt(0.28, y), pt(0.18, y)], thin);
                line(vec![pt(0.72, y), pt(0.82, y)], thin);
            }
        }
        Icon::Stack => {
            line(
                vec![
                    pt(0.5, 0.20),
                    pt(0.84, 0.39),
                    pt(0.5, 0.58),
                    pt(0.16, 0.39),
                    pt(0.5, 0.20),
                ],
                st,
            );
            line(vec![pt(0.16, 0.52), pt(0.5, 0.71), pt(0.84, 0.52)], st);
        }
        Icon::Broadcast => {
            p.circle_filled(pt(0.5, 0.5), size * 0.10, col);
            line(arc(0.5, 0.5, 0.24, 0.75 * PI, 1.25 * PI), thin);
            line(arc(0.5, 0.5, 0.40, 0.80 * PI, 1.20 * PI), thin);
            line(arc(0.5, 0.5, 0.24, -0.25 * PI, 0.25 * PI), thin);
            line(arc(0.5, 0.5, 0.40, -0.20 * PI, 0.20 * PI), thin);
        }
        Icon::Reload => {
            line(arc(0.5, 0.5, 0.31, -0.72 * PI, 0.82 * PI), bold);
            line(vec![pt(0.17, 0.26), pt(0.18, 0.51), pt(0.40, 0.40)], bold);
        }
        Icon::Console => {
            // A terminal window with a `>_` prompt: rounded frame + chevron + caret line.
            p.rect_stroke(
                Rect::from_min_max(pt(0.14, 0.20), pt(0.86, 0.80)),
                size * 0.08,
                st,
            );
            line(vec![pt(0.28, 0.38), pt(0.42, 0.50), pt(0.28, 0.62)], bold);
            line(vec![pt(0.50, 0.62), pt(0.70, 0.62)], bold);
        }
    }
}
