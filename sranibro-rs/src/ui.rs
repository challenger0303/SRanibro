//! In-process egui dashboard — Fluent-dark, design-token driven, animated.
//!
//! Reads [`Telemetry`] from the running [`Pipeline`] directly (no HTTP). The
//! signal rail is the headline: VRCFT's #1 pain point is opacity, so SRanibro
//! always shows where data is flowing — here as a live animated pipeline plus a
//! one-line diagnostic banner with the fix. Built entirely from `theme` tokens.

use std::collections::VecDeque;
use std::f32::consts::{PI, TAU};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Id, Pos2, Rect, Sense, Stroke};

use crate::brow_calib::{self, BrowCalib, Status as BrowStatus};
use crate::brow_fitrun::{BrowFitter, Status as FitStatus};
use crate::brow_train::{BrowTrainer, Status as TrainStatus, TrainInputs};
use crate::config::{Config, GazeCorrection, GazeSource, WideSource};
use crate::geometry_calib::{GeometryCapture, SampleFamily, Status as GeometryCaptureStatus};
use crate::geometry_fitrun::{
    FitInputs as GeometryFitInputs, GeometryFitResult, GeometryFitter, Status as GeometryFitStatus,
};
use crate::output::BrokenEyeStatus;
use crate::pipeline::{Pipeline, Telemetry};
use crate::theme::*;
use crate::wide_calib::{Status as WideCalibStatus, WideCalib};
use crate::wide_fitrun::{FitInputs as WideFitInputs, Status as WideFitStatus, WideFitter};

#[derive(PartialEq, Clone, Copy)]
enum Page {
    Dashboard,
    Calibration,
    BrowCalib,
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

pub fn run_ui(pipeline: Pipeline, be_status: Option<Arc<BrokenEyeStatus>>) -> eframe::Result<()> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // Fixed, non-resizable: sized to hold the capped content column
            // (NAV_W + margins + MAX_W) with a small balanced margin.
            .with_inner_size([WIN_W, WIN_H])
            // Fixed: the layout is fixed-metric and positions some elements from the
            // absolute window height (e.g. the nav LED), so a resizable window would
            // misplace them. True responsive layout is a separate refactor.
            .with_resizable(false)
            .with_maximize_button(false)
            .with_icon(app_icon())
            .with_title("SRanibro"),
        // wgpu backend: DX12->DX11->Vulkan->GL fallback, robust on VMs/RDP/weak drivers
        // where the default glow/OpenGL-3 path blank-windows or fails to launch.
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native(
        "SRanibro",
        native,
        Box::new(|cc| {
            crate::theme::apply(&cc.egui_ctx);
            Ok(Box::new(App::new(pipeline, be_status)))
        }),
    )
}

/// Editable copies of the asset paths + device, bound to the Settings text fields.
/// Empty string == "not set"; applied back into [`Config`] on reload.
#[derive(Clone)]
struct SettingsEdit {
    sranipal_dir: String,
    /// Common Tobii DLL (migrates the legacy per-device fields in from_cfg).
    tobii_dll: String,
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
}

impl SettingsEdit {
    fn from_cfg(c: &Config) -> Self {
        let g = |o: &Option<String>| o.clone().unwrap_or_default();
        Self {
            sranipal_dir: g(&c.assets.sranipal_dir),
            // Surface the resolved common DLL (incl. legacy starvr_dll/pimax_vr4_dll).
            tobii_dll: c
                .tobii_dll_path()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default(),
            brow_model: g(&c.assets.brow_model),
            wide_model: g(&c.assets.wide_model),
            wide_source: c.hmd.wide_source,
            gaze_source: c.gaze_source_for("pimax_xr5"),
            python_exe: g(&c.assets.python_exe),
            vr_eyebrow_dir: g(&c.assets.vr_eyebrow_dir),
            device: c.hmd.device.clone(),
            osc_host: c.output.osc_host.clone(),
            osc_port: c.output.osc_port,
        }
    }
}

struct GazeCenterCapture {
    started: Instant,
    sum_deg: [[f64; 2]; 2],
    count: u32,
    last_timestamp_us: u64,
}

impl GazeCenterCapture {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            sum_deg: [[0.0; 2]; 2],
            count: 0,
            last_timestamp_us: 0,
        }
    }
}

struct App {
    pipeline: Pipeline,
    tele: Arc<Telemetry>,
    config: Config,
    /// Live-editable asset/device fields (Settings tab).
    edit: SettingsEdit,
    /// Last "Apply & reload" result message (text, color).
    reload_msg: Option<(String, Color32)>,
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
    /// Preview textures for the Calibration tab's ML-input geometry card (the processed
    /// 100x100 the eye model actually sees, per eye).
    tex_ml_l: Option<egui::TextureHandle>,
    tex_ml_r: Option<egui::TextureHandle>,
    /// Textures for the ML occlusion-heatmap overlay (eye + colormap), per eye.
    tex_heat_l: Option<egui::TextureHandle>,
    tex_heat_r: Option<egui::TextureHandle>,
    /// Colormap full-scale for the heatmap (|openness delta| that maps to full colour).
    heat_vmax: f32,
    // diagnostic event log (stage up/down, tracking toggles)
    start: Instant,
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
    /// One-second straight-ahead capture used by the gaze Center action.
    gaze_center_capture: Option<GazeCenterCapture>,
    gaze_center_msg: Option<(String, Color32)>,
    /// Calibration-modal-only source edit. Discarded on close so it cannot be
    /// committed later by an unrelated Settings reload.
    gaze_source_modal_edit: Option<GazeSource>,
    dream_air_msg: Option<(String, Color32)>,
    /// In-memory, labelled stereo capture used only by the XR5 geometry search.
    geometry_capture: GeometryCapture,
    /// Background pure-Rust candidate search + untouched-holdout validation.
    geometry_fitter: GeometryFitter,
    /// Geometry active at capture start. The search is always centred on this fallback.
    geometry_capture_baseline: Option<[crate::core::types::MlGeometry; 2]>,
    /// Photometric filters are snapshotted with the baseline so edits made after capture
    /// cannot mismatch the stored per-frame brightness affine during replay.
    geometry_capture_filters: Option<(
        crate::core::types::DespeckleParams,
        crate::core::types::FlattenParams,
    )>,
    /// Unsaved geometry restored when the user exits candidate preview.
    geometry_preview_restore: Option<[crate::core::types::MlGeometry; 2]>,
    /// Previous persisted geometry retained for one-click rollback after Apply.
    geometry_rollback: Option<[crate::core::types::MlGeometry; 2]>,
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
    fn new(pipeline: Pipeline, be: Option<Arc<BrokenEyeStatus>>) -> Self {
        let tele = pipeline.tele.clone();
        // Anchored path (not CWD), and SURFACE the malformed-config warning instead of
        // silently resetting to defaults.
        let (config, cfg_warn) = Config::load(&crate::config::config_path());
        let edit = SettingsEdit::from_cfg(&config);
        let page = match std::env::args().nth(2).as_deref() {
            Some("calibration") => Page::Calibration,
            Some("brow") | Some("browcalib") => Page::BrowCalib,
            Some("console") => Page::Console,
            Some("settings") => Page::Settings,
            _ => Page::Dashboard,
        };
        Self {
            pipeline,
            tele,
            config,
            edit,
            reload_msg: cfg_warn.map(|w| (format!("config reset to defaults — {w}"), WARN)),
            page,
            last: [0; 5],
            rates: [0.0; 5],
            last_t: Instant::now(),
            tex_l: None,
            tex_r: None,
            last_eye_texture_upload: Instant::now() - Duration::from_secs(1),
            tex_ml_l: None,
            tex_ml_r: None,
            tex_heat_l: None,
            tex_heat_r: None,
            heat_vmax: 0.20,
            show_geom_modal: false,
            show_gaze_modal: false,
            gaze_center_capture: None,
            gaze_center_msg: None,
            gaze_source_modal_edit: None,
            dream_air_msg: None,
            geometry_capture: GeometryCapture::new(),
            geometry_fitter: GeometryFitter::new(),
            geometry_capture_baseline: None,
            geometry_capture_filters: None,
            geometry_preview_restore: None,
            geometry_rollback: None,
            geom_tab: 0,
            geom_eye: 0,
            net_view: false,
            start: Instant::now(),
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
}

impl Drop for App {
    fn drop(&mut self) {
        self.pipeline.stop();
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // eframe/wgpu skips presenting while a Windows viewport is minimized. Building the
        // complete dashboard anyway (especially `TextureHandle::set` for the live eye
        // cameras) leaves texture/paint deltas queued behind the unavailable surface. They
        // are not reclaimed on restore and Private Bytes can grow by hundreds of MiB per
        // minute. Keep the non-render tracking/capture controllers alive, but emit an empty
        // UI frame until Windows restores the surface.
        let minimized = ctx.input(|i| i.viewport().minimized.unwrap_or(false));
        if minimized {
            let poll_ms = if self.wide.is_running() || self.geometry_capture.is_running() {
                16
            } else {
                250
            };
            ctx.request_repaint_after(Duration::from_millis(poll_ms));
            self.update_rates();
            self.detect_events();
            self.update_gaze_center_capture();
            self.update_geometry_capture();
            self.update_wide_capture();
            self.apply_wide_fit_result_if_ready();
            return;
        }

        // NOTE: we deliberately do NOT touch zoom_factor / pixels_per_point here.
        // Deriving zoom from native_pixels_per_point() each frame oscillates,
        // because egui feeds the *effective* ppp back through that call — the
        // result flickers between scales. We let egui keep its stable native ppp
        // and size everything from fixed point dimensions in `theme`.
        // UI redraw rate. ML/eye/brow inference run on their OWN threads. Keep focused
        // presentation at 60 fps: Windows/winit processes native move/size interaction
        // on this UI thread, so reducing it to 30 fps also makes the WINDOW itself move
        // in visibly coarse 33 ms steps. The expensive live-eye texture uploads are
        // throttled separately to 30 Hz in `dashboard`, which avoids paying their full
        // GPU cost on every input-triggered repaint. Unfocused stays at 10 fps.
        // NOTE: dragging the window on Windows enters winit's modal move/size loop, which
        // pauses redraws regardless — the in-drag freeze is a winit/Windows limit.
        // Wide capture is intentionally clocked by camera generations, but its collector
        // is polled here. Keep polling at 60 Hz while the user is wearing the HMD and the
        // desktop window is naturally unfocused; otherwise a 10 Hz UI would make the
        // guided capture take several minutes.
        let redraw_ms = if self.wide.is_running()
            || self.geometry_capture.is_running()
            || ctx.input(|i| i.focused)
        {
            16
        } else {
            100
        };
        ctx.request_repaint_after(std::time::Duration::from_millis(redraw_ms));
        self.fit_to_monitor(ctx);
        self.update_rates();
        self.detect_events();
        self.update_gaze_center_capture();
        self.update_geometry_capture();
        self.update_wide_capture();
        self.apply_wide_fit_result_if_ready();
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
                Page::BrowCalib => self.brow_calib(ui),
                Page::Console => self.console(ui),
                Page::Settings => self.settings(ui),
            });
        if self.show_gaze_modal {
            self.gaze_correction_modal(ctx);
        }
    }
}

impl App {
    #[cfg(any())]
    fn current_preflight(&self) -> PreflightReport {
        let frame_dims = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0].as_ref().map(|(w, h, _)| (*w, *h)),
                frames[1].as_ref().map(|(w, h, _)| (*w, *h)),
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

    /// Drive the XR5 geometry capture from real camera generations. The capture clock is
    /// capped at 20 Hz, so UI repaint bursts cannot duplicate a frame and an unfocused UI
    /// still records enough temporal detail for blink timing.
    fn update_geometry_capture(&mut self) {
        self.geometry_capture.tick();
        if !self.geometry_capture.is_running() || self.pipeline.device_key != "pimax_xr5" {
            return;
        }
        let generation = [
            self.tele.c_frame_l.load(Ordering::Relaxed),
            self.tele.c_frame_r.load(Ordering::Relaxed),
        ];
        let frames = self.tele.frames.lock().unwrap().clone();
        let affine = *self.pipeline.bright_affine.lock().unwrap();
        let gaze = *self.tele.gaze.lock().unwrap();
        let native_open = [gaze.left, gaze.right].map(|eye| {
            if !eye.openness_reported {
                None
            } else if !eye.openness_valid {
                Some(0.0)
            } else if eye.openness.is_finite() {
                Some(eye.openness.clamp(0.0, 1.0))
            } else {
                None
            }
        });
        let left = frames[0]
            .as_ref()
            .map(|(width, height, pixels)| (*width, *height, pixels.as_slice()));
        let right = frames[1]
            .as_ref()
            .map(|(width, height, pixels)| (*width, *height, pixels.as_slice()));
        self.geometry_capture
            .on_frame(generation, left, right, affine, native_open);
    }

    fn geometry_capture_ready(&self) -> (bool, String) {
        let dimensions = {
            let frames = self.tele.frames.lock().unwrap();
            [
                frames[0]
                    .as_ref()
                    .map(|(width, height, _)| (*width, *height)),
                frames[1]
                    .as_ref()
                    .map(|(width, height, _)| (*width, *height)),
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

    fn start_geometry_capture(&mut self) {
        if self.geometry_fitter.is_running() {
            self.dream_air_msg = Some(("Cancel the active geometry fit first.".into(), WARN));
            return;
        }
        let (ready, detail) = self.geometry_capture_ready();
        if !ready {
            self.dream_air_msg = Some((detail, WARN));
            return;
        }
        self.restore_geometry_preview(false);
        let generation = [
            self.tele.c_frame_l.load(Ordering::Relaxed),
            self.tele.c_frame_r.load(Ordering::Relaxed),
        ];
        self.geometry_capture_baseline = Some(*self.pipeline.geometry.lock().unwrap());
        self.geometry_capture_filters = Some((
            *self.pipeline.despeckle.lock().unwrap(),
            *self.pipeline.flatten.lock().unwrap(),
        ));
        self.geometry_capture.start(generation);
        self.dream_air_msg = Some((
            "Guided image-alignment capture started; raw frames stay in memory unless you export the completed recording.".into(),
            ACCENT,
        ));
    }

    fn export_geometry_recording(&mut self) {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = crate::config::base_dir()
            .join("calibration-recordings")
            .join(format!("sranibro_xr5_geometry_{stamp}.zip"));
        let serial = self
            .config
            .dream_air_profile_for(&self.pipeline.device_key)
            .and_then(|profile| profile.eyechip_serial.clone())
            .or_else(crate::device::usb::peek_serial);
        let unit_id = crate::diagnostics::pseudonymous_unit_id(serial.as_deref());
        let baseline = self.geometry_capture_baseline;
        let filters = self.geometry_capture_filters;
        let mapping = self.config.mapping_for(&self.pipeline.device_key);
        let mirrors = [
            self.pipeline.ml_mirror_l.load(Ordering::Relaxed),
            self.pipeline.ml_mirror_r.load(Ordering::Relaxed),
        ];
        let metadata = format!(
            "schema_version=1\nsranibro_version={}\ndevice={}\nunit_id={}\ncapture_hz=20\nframe_stage=after_eye_mapping_before_ml_geometry\nbaseline_geometry={baseline:?}\nfilters={filters:?}\neye_mapping={mapping:?}\nml_mirror={mirrors:?}\nwide_source={:?}\ngaze_source={:?}\n",
            env!("CARGO_PKG_VERSION"),
            self.pipeline.device_key,
            unit_id,
            self.config.hmd.wide_source,
            self.config.gaze_source_for(&self.pipeline.device_key),
        );
        match self.geometry_capture.export_recording(&path, &metadata) {
            Ok(()) => {
                self.dream_air_msg = Some((
                    format!("Calibration recording ZIP saved: {}", path.display()),
                    OK,
                ));
            }
            Err(error) => {
                self.dream_air_msg =
                    Some((format!("Calibration recording export failed: {error}"), ERR));
            }
        }
    }

    fn start_geometry_fit(&mut self) {
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
            dataset,
            baseline,
            mirrors,
            despeckle,
            flatten,
        };
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
                    .restore_completed_dataset(error.inputs.dataset);
                self.dream_air_msg = Some((
                    format!("Geometry fit could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn start_geometry_audit(&mut self) {
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
            dataset,
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
                    .restore_completed_dataset(error.inputs.dataset);
                self.dream_air_msg = Some((
                    format!("Geometry audit could not start: {}", error.message),
                    ERR,
                ));
            }
        }
    }

    fn set_live_geometry(&mut self, geometry: [crate::core::types::MlGeometry; 2]) {
        *self.pipeline.geometry.lock().unwrap() = geometry;
        if let Some(mirror) = geometry[0].mirror_h {
            self.pipeline.ml_mirror_l.store(mirror, Ordering::Relaxed);
        }
        if let Some(mirror) = geometry[1].mirror_h {
            self.pipeline.ml_mirror_r.store(mirror, Ordering::Relaxed);
        }
    }

    fn preview_geometry_candidate(&mut self, result: &GeometryFitResult) {
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

    fn apply_geometry_candidate(&mut self, result: &GeometryFitResult) {
        if !result.accepted {
            self.dream_air_msg = Some((
                "This candidate did not pass holdout validation and cannot be applied.".into(),
                ERR,
            ));
            return;
        }
        let device = self.pipeline.device_key.clone();
        if self.config.geometry_for(&device) == result.candidate {
            self.dream_air_msg = Some(("This validated geometry is already applied.".into(), OK));
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
        self.dream_air_msg = Some((
            format!("Candidate applied. Safety backup: {}", backup.display()),
            OK,
        ));
    }

    fn rollback_geometry(&mut self) {
        if self.geometry_capture.is_running() || self.geometry_fitter.is_running() {
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

    /// Capture exactly one stereo pair per pair of NEW device frames. The UI repaint
    /// rate is irrelevant; the camera counters are the sampling clock.
    fn update_wide_capture(&mut self) {
        if self.pipeline.device_key != "pimax_xr5" {
            return;
        }
        self.wide.tick();
        if !self.wide.is_running() {
            return;
        }
        let generation = [
            self.tele.c_frame_l.load(Ordering::Relaxed),
            self.tele.c_frame_r.load(Ordering::Relaxed),
        ];
        if generation[0] == self.wide_last_frames[0] || generation[1] == self.wide_last_frames[1] {
            return;
        }
        self.wide_last_frames = generation;
        let frames = self.tele.frames.lock().unwrap().clone();
        let left = frames[0]
            .as_ref()
            .map(|(w, h, pixels)| (*w, *h, pixels.as_slice()));
        let right = frames[1]
            .as_ref()
            .map(|(w, h, pixels)| (*w, *h, pixels.as_slice()));
        self.wide.on_frame(left, right);
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
                    .inner_margin(egui::Margin::symmetric(14.0 * S, 9.0 * S)),
            )
            .show(ctx, |ui| {
                // Minimal bar: just the wordmark. Device name, output state,
                // and per-stage detail live in the pipeline nodes (hover a node).
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
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
                });
            });
    }

    fn nav(&mut self, ctx: &egui::Context) {
        let live = self.rates[4] > 1.0 && !self.pipeline.paused.load(Ordering::Relaxed);
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
                ui.vertical_centered(|ui| {
                    self.nav_icon(ui, Page::Dashboard, Icon::Activity, 0);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Calibration, Icon::Sliders, 1);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::BrowCalib, Icon::Brow, 2);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Console, Icon::Console, 3);
                    ui.add_space(4.0 * S);
                    self.nav_icon(ui, Page::Settings, Icon::Gear, 4);
                });
            });
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
        let frames = {
            let f = self.tele.frames.lock().unwrap();
            [f[0].clone(), f[1].clone()]
        };
        // A focused window can receive hundreds of mouse events per second. Egui may
        // repaint for those events even though the scheduled cadence is 30 fps; keep
        // camera texture conversion/uploads at a hard 30 Hz ceiling regardless.
        let upload_eye_textures = self.last_eye_texture_upload.elapsed()
            >= Duration::from_millis(33)
            || self.tex_l.is_none()
            || self.tex_r.is_none();
        if upload_eye_textures {
            self.last_eye_texture_upload = Instant::now();
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
                // wide/squeeze "chain" toggle state, read from live tuning; persisted
                // below if the ML-parameters card's chain glyph flips it this frame.
                let mut chain = self.pipeline.tuning.lock().unwrap().wide_squeeze_exclusive;
                let chain_prev = chain;
                let (tex_l, tex_r) = (&mut self.tex_l, &mut self.tex_r);
                let ml_frames = self.tele.ml_input.lock().unwrap().clone();
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
                            &ctx,
                        )
                    },
                );
                let h_eye = cams.response.rect.height();
                if cams.inner {
                    self.show_geom_modal = true;
                }
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
                            &mut chain,
                        )
                    },
                );
                // Persist the chain toggle if the card flipped it (survives restart).
                if chain != chain_prev {
                    self.pipeline.tuning.lock().unwrap().wide_squeeze_exclusive = chain;
                    self.config.tuning.wide_squeeze_exclusive = chain;
                    let _ = self.config.save(&crate::config::config_path());
                }
                let row_h = h_eye.max(mlr.response.rect.height());
                // Reset the cursor below the taller card so the log doesn't overlap.
                ui.allocate_rect(Rect::from_min_size(start, vec2(cw, row_h)), Sense::hover());

                ui.add_space(SP3);
                self.terminal_log(ui, cw);
                // ML-input geometry modal (opened by the eye-cameras gear) — overlays all.
                if self.show_geom_modal {
                    self.geom_modal(&ctx, &frames);
                }
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
            "starvr" | "starvr_one" | "pimax_dll" | "pimax_vr4_dll" | "pimax_stream" => "DLL",
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
                .map(|(fw, fh, _)| (*fw, *fh))
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
                    if !uninformative {
                        format!("{}: {st}", names[i])
                    } else if is_varjo {
                        format!(
                            "{}: no stream → start Varjo Base and put the headset on",
                            names[i]
                        )
                    } else {
                        format!(
                            "{}: no stream → set the Tobii DLL (Settings), then check the headset",
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
        let capture_status = self.geometry_capture.status();
        let fit_status = self.geometry_fitter.status();
        let (ready, ready_detail) = self.geometry_capture_ready();
        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            egui::CollapsingHeader::new(h3("Automatic eye image alignment"))
                .id_salt("xr5_geometry_setup_card")
                .default_open(false)
                .show(ui, |ui| {
                    ui.add_space(SP2);
                    ui.label(label(
                        "Dream Air / XR5 only. Records labelled stereo frames in memory, estimates absolute crop/angle hypotheses from repeated neutral-eye appearance and eyelid motion, compares them with a bounded ML search, then validates the winner on untouched holdout frames.",
                    ));
                    ui.label(label(
                        "This does not train a model, change Tobii gaze calibration, or use squeeze/Wide as geometry targets.",
                    ));
                    ui.label(label(
                        "The fixed inner XR5 IR-LED/lens region is excluded before extracting geometry evidence and search candidates cannot reduce the inner hardware crop below 40%.",
                    ));
                    ui.label(label(
                        "After capture you can export the exact labelled stereo recording for feedback. The ZIP contains raw eye images (biometric data) and is never created automatically.",
                    ));
                    ui.add_space(SP2);

                    match capture_status {
                        GeometryCaptureStatus::Rest {
                            instruction,
                            remaining_s,
                            overall,
                        } => {
                            ui.label(
                                egui::RichText::new("GET READY")
                                    .monospace()
                                    .size(14.0 * S)
                                    .strong()
                                    .color(ACCENT),
                            );
                            ui.label(label(instruction));
                            ui.label(num(&format!("next phase in {remaining_s:.1}s")));
                            ui.add(egui::ProgressBar::new(overall).show_percentage());
                            if ui.button("Cancel and discard in-memory frames").clicked() {
                                self.geometry_capture.abort();
                                self.geometry_capture_baseline = None;
                                self.geometry_capture_filters = None;
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
                                ui.add(
                                    egui::ProgressBar::new(phase_progress)
                                        .desired_width(300.0 * S),
                                );
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
                            ui.label(label(
                                "Save the recording before starting a fit or audit; those operations consume the in-memory dataset.",
                            ));
                            if ui.button("Save calibration recording ZIP for feedback").clicked() {
                                self.export_geometry_recording();
                            }
                            ui.horizontal(|ui| {
                                if ui.button("Run objective audit (recommended)").clicked() {
                                    self.start_geometry_audit();
                                }
                                if ui.button("Start safe geometry fit").clicked() {
                                    self.start_geometry_fit();
                                }
                                if ui.button("Discard capture").clicked() {
                                    self.geometry_capture.abort();
                                    self.geometry_capture_baseline = None;
                                    self.geometry_capture_filters = None;
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
                                    egui::ProgressBar::new(
                                        completed as f32 / total.max(1) as f32,
                                    )
                                    .show_percentage(),
                                );
                                ui.label(num(&format!("{completed} / {total} candidate-frame evaluations")));
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
                                if result.accepted {
                                    let already_applied = self
                                        .config
                                        .geometry_for(&self.pipeline.device_key)
                                        == result.candidate;
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
                                        if ui
                                            .add_enabled(
                                                !already_applied,
                                                egui::Button::new("Apply validated candidate"),
                                            )
                                            .clicked()
                                        {
                                            self.apply_geometry_candidate(&result);
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
                                }
                                ui.horizontal(|ui| {
                                    if ui.button("Record again").clicked() {
                                        self.start_geometry_capture();
                                    }
                                    if ui.button("Open manual image controls").clicked() {
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
                                    .add_enabled(
                                        ready,
                                        egui::Button::new("Start automatic image alignment"),
                                    )
                                    .clicked()
                                {
                                    self.start_geometry_capture();
                                }
                                ui.add_space(SP2);
                                if ui.button("Open manual image controls").clicked() {
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
                            && !self.geometry_fitter.is_running();
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
                });
        });
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
                    if reason.contains("Recenter") && ui.button("Recenter now").clicked() {
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
        let wide_ready = *self.tele.wide_ready.lock().unwrap();
        let bootstrap_seen = *self.tele.wide_bootstrap_seen.lock().unwrap();
        let busy = self.wide.is_running() || self.wide_fitter.is_running();
        let mut start_capture = false;
        let mut cancel_capture = false;
        let mut start_fit = false;
        let mut delete_all = false;
        let mut apply_source = false;

        card().show(ui, |ui| {
            ui.set_width(width - 2.0 * CARD_PAD);
            let title = format!(
                "XR5 image EyeWide    {}{}",
                if model_loaded { "MODEL READY" } else { "NO MODEL" },
                if model_active { "    CUSTOM ACTIVE" } else { "" }
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
                            ui.selectable_value(
                                &mut self.edit.wide_source,
                                source,
                                source.as_str(),
                            );
                        }
                    });
                if ui
                    .add_enabled(!busy, egui::Button::new("Apply source & reload"))
                    .clicked()
                {
                    apply_source = true;
                }
            });
            ui.label(label(
                "Auto uses a fresh calibrated custom result and otherwise falls back to SRanipal. Custom never silently falls back.",
            ));

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
            match self.wide.start() {
                Ok(()) => {
                    self.wide_last_frames = [
                        self.tele.c_frame_l.load(Ordering::Relaxed),
                        self.tele.c_frame_r.load(Ordering::Relaxed),
                    ];
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
            self.apply_and_reload();
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
            egui::CollapsingHeader::new(h3("Calibration"))
                .id_salt("calibration_card")
                .default_open(true)
                .show(ui, |ui| {
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
                if ui.add_sized([150.0 * S, 32.0 * S], recenter).clicked() {
                    self.pipeline.recenter.store(true, Ordering::Relaxed);
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
                if ui.add_sized([110.0 * S, 32.0 * S], rec).clicked() {
                    self.pipeline.diag_rec.store(!rec_on, Ordering::Relaxed);
                }
            });
            ui.add_space(SP2);
            ui.label(num(&format!("baseline   L {:.3}    R {:.3}", baselines[0], baselines[1])));
            if self.pipeline.device_key == "pimax_xr5" {
                ui.add_space(SP2);
                ui.separator();
                ui.add_space(SP2);
                let correction = *self.pipeline.gaze_correction.lock().unwrap();
                ui.horizontal(|ui| {
                    if ui.button("Dream Air / XR5 gaze correction").clicked() {
                        self.show_gaze_modal = true;
                    }
                    ui.add_space(SP2);
                    ui.label(label(if correction.enabled { "enabled" } else { "disabled" }));
                });
            }
                });
        });
        ui.add_space(SP3);
        if self.pipeline.device_key == "pimax_xr5" {
            self.dream_air_onboarding_card(ui, cw);
            ui.add_space(SP3);
            self.dream_air_wide_card(ui, cw);
            ui.add_space(SP3);
        }
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            egui::CollapsingHeader::new(h3("Tuning"))
                .id_salt("tuning_card")
                .default_open(false)
                .show(ui, |ui| {
            ui.add_space(SP2);
            let mut t = self.pipeline.tuning.lock().unwrap();
            ui.spacing_mut().slider_width = 240.0 * S;
            let mut ch = false;
            ch |= ui.add(egui::Slider::new(&mut t.alpha_open, 0.05..=1.0).text("open speed")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.alpha_close, 0.05..=1.0).text("close speed")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.squeeze_deadzone, 0.0..=0.9).text("squeeze deadzone")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.squeeze_gain, 0.0..=1.0).text("squeeze gain")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.wide_gain, 0.0..=1.0).text("wide gain")).changed();
            ch |= ui.add(egui::Slider::new(&mut t.open_deadzone, 0.03..=0.20).text("eye-open dead-zone")).changed();
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

    fn persist_gaze_correction(&mut self, correction: GazeCorrection) {
        *self.pipeline.gaze_correction.lock().unwrap() = correction;
        let device = self.pipeline.device_key.clone();
        self.config.set_gaze_correction(&device, correction);
        if let Err(e) = self.config.save(&crate::config::config_path()) {
            self.gaze_center_msg = Some((format!("save failed: {e}"), ERR));
        }
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
        self.persist_gaze_correction(correction);
        self.gaze_center_msg = Some((format!("Centered from {} fresh samples", capture.count), OK));
    }

    /// Returns the source to apply when the user explicitly requests a reload.
    /// Correction sliders remain live and do not require reload.
    fn gaze_correction_body(&mut self, ui: &mut egui::Ui) -> Option<GazeSource> {
        ui.label(h3("Dream Air / XR5 gaze correction"));
        ui.add_space(SP2);
        ui.label(label(
            "Run Pimax/Tobii calibration first. This is a saved finishing trim for residual centre, vergence, and range mismatch; it affects gaze output only.",
        ));
        ui.add_space(SP3);

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
        let reload_source = ui
            .add_enabled(
                source_pending,
                egui::Button::new(if source_pending {
                    "Apply gaze source & reload"
                } else {
                    "Gaze source already active"
                }),
            )
            .clicked();
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

        let mut correction = *self.pipeline.gaze_correction.lock().unwrap();
        let mut changed = ui.checkbox(&mut correction.enabled, "Enabled").changed();
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
            if ui.add_enabled(!capturing, button).clicked() {
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
        ui.spacing_mut().slider_width = 260.0 * S;
        changed |= ui
            .add(
                egui::Slider::new(&mut correction.vergence_deg, -10.0..=10.0)
                    .text("vergence trim (deg)"),
            )
            .changed();
        ui.label(label("Moves left/right gaze in opposite directions; use after Center if the avatar still looks cross-eyed."));

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
            changed |= ui
                .add(
                    egui::Slider::new(&mut correction.offset_x_deg[eye], -15.0..=15.0)
                        .text("X centre (deg)"),
                )
                .changed();
            changed |= ui
                .add(
                    egui::Slider::new(&mut correction.offset_y_deg[eye], -15.0..=15.0)
                        .text("Y centre (deg)"),
                )
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut correction.scale_x[eye], 0.5..=1.5).text("X range"))
                .changed();
            changed |= ui
                .add(egui::Slider::new(&mut correction.scale_y[eye], 0.5..=1.5).text("Y range"))
                .changed();
        }

        ui.add_space(SP3);
        if ui.button("Reset XR5 gaze correction").clicked() {
            correction = GazeCorrection::default();
            self.gaze_center_capture = None;
            self.gaze_center_msg = Some(("Reset to native Pimax/Tobii gaze".into(), TEXT2));
            changed = true;
        }
        if changed {
            self.persist_gaze_correction(correction);
        }
        reload_source.then_some(selected_source)
    }

    fn gaze_correction_modal(&mut self, ctx: &egui::Context) {
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
    fn ml_geometry_body(&mut self, ui: &mut egui::Ui, frames: &[Option<(u32, u32, Vec<u8>)>; 2]) {
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

        // --- Brightness match (adaptive normalization) ----------------------------------
        ui.add_space(SP3);
        ui.separator();
        ui.add_space(SP2);
        ui.label(h3("Brightness match"));
        ui.add_space(SP2);
        ui.label(label(
            "Auto-normalizes the ML input's brightness + contrast to a target learned from your own settled frames, so lens-to-eye distance (and HMD reseats) don't bias openness. Slow, so it never cancels a blink; complements the camera's own auto-exposure.",
        ));
        ui.add_space(SP2);
        let dev = self.pipeline.device_key.clone();
        let mut bn = *self.pipeline.brightness.lock().unwrap();
        let mut bch = false;
        bch |= ui.checkbox(&mut bn.enabled, "Enabled").changed();
        bch |= ui
            .checkbox(&mut bn.auto_learn, "Auto-learn target from settled frames")
            .changed();
        bch |= ui
            .add(egui::Slider::new(&mut bn.strength, 0.0..=1.0).text("strength"))
            .changed();
        bch |= ui
            .add(egui::Slider::new(&mut bn.adapt, 0.005..=0.1).text("adapt speed"))
            .changed();
        let recap = ui.button("Recapture reference").clicked();
        if bn.captured {
            ui.label(label(&format!(
                "reference: L {:.0}/{:.0}   R {:.0}/{:.0}  (level / spread)",
                bn.tgt_level[0], bn.tgt_spread[0], bn.tgt_level[1], bn.tgt_spread[1],
            )));
        } else {
            ui.label(label(
                "reference: learning… hold a relaxed, open gaze for ~2s",
            ));
        }
        if bch || recap {
            // Overwrite only the params we edit (preserve the ML thread's captured target);
            // drop `captured` on Recapture so it re-learns.
            let mut g = self.pipeline.brightness.lock().unwrap();
            g.enabled = bn.enabled;
            g.auto_learn = bn.auto_learn;
            g.strength = bn.strength;
            g.adapt = bn.adapt;
            if recap {
                g.captured = false;
            }
            let live = *g;
            drop(g);
            self.config.set_brightness(&dev, live);
            let _ = self.config.save(&crate::config::config_path());
        }
        // Persist the auto-captured target (set by the ML thread) when it appears / changes.
        let live = *self.pipeline.brightness.lock().unwrap();
        if live != self.config.brightness_for(&dev) {
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
            "How the model's openness reacts to each region — to check whether the reflection dots corrupt tracking, and whether the filter fixed it. Occlusion = what the model relies on; Glint inject = paint a fake reflection and see where openness breaks. One-shot (~2s; openness freezes while it computes).",
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
        ui.add(egui::Slider::new(&mut self.heat_vmax, 0.03..=0.5).text("heat scale (VMAX)"));
        ui.add_space(SP2);
        let vmax = self.heat_vmax.max(1e-3);
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
    }

    /// Live preview inside the modal. `warped=true` (Image tab) shows the geometry-warped
    /// MODEL input (100x100) so crop/rotate is visible; `warped=false` (Filter tab) shows
    /// the despeckled REAL eye at its native resolution and NATURAL orientation — L on the
    /// left, R on the right, matching the live cameras — so you can see the reflection dots
    /// removed. Neither applies the ML left/right mirror.
    fn geom_preview(
        &mut self,
        ui: &mut egui::Ui,
        frames: &[Option<(u32, u32, Vec<u8>)>; 2],
        warped: bool,
    ) {
        ui.label(label(if warped {
            "Model input preview (crop / rotate / stretch applied):"
        } else {
            "Filtered eye preview — dots removed (this is your real eye, not the model view):"
        }));
        ui.add_space(SP2);
        let g = *self.pipeline.geometry.lock().unwrap();
        let dsp = *self.pipeline.despeckle.lock().unwrap();
        let flt = *self.pipeline.flatten.lock().unwrap();
        let aff = *self.pipeline.bright_affine.lock().unwrap();
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
                if let Some((fw, fh, px)) = &frames[i] {
                    // Match the ML input pipeline: despeckle -> flatten -> brightness normalize.
                    let filtered =
                        crate::ml::preprocess::despeckle(px, *fw as usize, *fh as usize, &dsp);
                    let flat =
                        crate::ml::preprocess::flatten(&filtered, *fw as usize, *fh as usize, &flt);
                    let normed = crate::ml::brightness::apply(&flat, aff[i][0], aff[i][1]);
                    // Build the display image: the warped model input (geometry, NO mirror),
                    // or the despeckled + brightness-matched real eye at native res.
                    let built = if warped {
                        let prev = crate::ml::preprocess::ml_input_preview(
                            &normed, *fw, *fh, &g[i], false,
                        );
                        (prev.len() == n * n).then(|| {
                            let mut im = egui::ColorImage::new([n, n], Color32::BLACK);
                            for k in 0..n * n {
                                im.pixels[k] = Color32::from_gray(prev[k]);
                            }
                            im
                        })
                    } else {
                        let (w, h) = (*fw as usize, *fh as usize);
                        (w > 0 && h > 0 && normed.len() >= w * h).then(|| {
                            let mut im = egui::ColorImage::new([w, h], Color32::BLACK);
                            for k in 0..w * h {
                                im.pixels[k] = Color32::from_gray(normed[k]);
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
    fn geom_modal(&mut self, ctx: &egui::Context, frames: &[Option<(u32, u32, Vec<u8>)>; 2]) {
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

    /// B-1: eyebrow-calibration DATA COLLECTION tab. Guides the user through the
    /// vr_eyebrow capture protocol and writes RAW eye frames + labels.csv under
    /// base_dir()/brow_data (drop-in for the Python train.py). Capture-only here;
    /// B-2 (train/bake subprocess) is a later task.
    fn brow_calib(&mut self, ui: &mut egui::Ui) {
        // --- Drive the state machine (egui repaints continuously) ---------------------
        // 1) Wall-clock rest phases advance here.
        self.brow.tick();
        // 2) On a NEW device frame (either eye's counter moved), save one stereo pair.
        //    Using the frame counters (not the repaint) means we save exactly one frame per
        //    real camera frame during a capture phase, regardless of UI fps.
        let cur = [
            self.tele.c_frame_l.load(Ordering::Relaxed),
            self.tele.c_frame_r.load(Ordering::Relaxed),
        ];
        let fresh = cur != self.brow_last_frames;
        self.brow_last_frames = cur;
        // Snapshot the newest frames once (short lock).
        let frames = {
            let f = self.tele.frames.lock().unwrap();
            [f[0].clone(), f[1].clone()]
        };
        if fresh && self.brow.is_running() {
            let l = frames[0].as_ref().map(|(w, h, p)| (*w, *h, p.as_slice()));
            let r = frames[1].as_ref().map(|(w, h, p)| (*w, *h, p.as_slice()));
            self.brow.on_frame(l, r);
        }

        let (gutter, cw) = stage_metrics(ui.ctx());
        let status = self.brow.status();
        let total_cap = self.brow.total_captured();
        let total_target = brow_calib::total_capture_target();
        let root = self.brow.root().to_path_buf();
        let err = self.brow.last_error.clone();

        // Scrollable page body: the three cards (capture + Fit + Train & bake) can exceed the
        // panel height, so wrap them — otherwise the bottom card's last label clips.
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
        ui.horizontal_top(|ui| {
        ui.add_space(gutter);
        ui.vertical(|ui| {
        ui.set_width(cw);

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
                let start_resp = ui.add_enabled_ui(!running, |ui| {
                    ui.add_sized([150.0 * S, 32.0 * S], start)
                }).inner;
                if start_resp.clicked() && !running {
                    if let Err(e) = self.brow.start() {
                        self.brow.last_error = Some(format!("could not start: {e}"));
                    }
                    self.brow_last_frames = cur;
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
        });
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

            let can_fit = has_labels && !running;
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
        let can_train = missing.is_empty() && !running;

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
                        "Live switches save immediately. Device, model, path, and OSC target changes use Apply & reload.",
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
                            egui::CollapsingHeader::new(h3("Tracking & device"))
                                .id_salt("settings_tracking")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.label(prose("The controls normally used when changing HMD or tuning live output."));
                                    ui.add_space(SP2);

                                    ui.horizontal(|ui| {
                                        ui.label(label("Headset"));
                                        egui::ComboBox::from_id_salt("device_select")
                                            .selected_text(self.edit.device.clone())
                                            .show_ui(ui, |ui| {
                                                for d in [
                                                    "auto",
                                                    "pimax_vr4",
                                                    "pimax_xr5",
                                                    "varjo",
                                                    "varjo_mjpeg",
                                                    "starvr",
                                                ] {
                                                    ui.selectable_value(
                                                        &mut self.edit.device,
                                                        d.to_string(),
                                                        d,
                                                    );
                                                }
                                            });
                                    });
                                    let selected_device =
                                        crate::config::canonical_device_key(&self.edit.device);
                                    let xr5_settings = selected_device == "pimax_xr5"
                                        || (selected_device == "auto"
                                            && self.pipeline.device_key == "pimax_xr5");
                                    if xr5_settings {
                                        ui.horizontal(|ui| {
                                            ui.label(label("XR5 EyeWide source"));
                                            egui::ComboBox::from_id_salt("settings_wide_source")
                                                .selected_text(self.edit.wide_source.as_str())
                                                .show_ui(ui, |ui| {
                                                    for source in WideSource::ALL {
                                                        ui.selectable_value(
                                                            &mut self.edit.wide_source,
                                                            source,
                                                            source.as_str(),
                                                        );
                                                    }
                                                });
                                        });
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

                                    let mut eyebrow_enabled = self
                                        .pipeline
                                        .eyebrow_enabled
                                        .load(Ordering::Relaxed);
                                    if ui
                                        .checkbox(
                                            &mut eyebrow_enabled,
                                            "Enable eyebrow tracking",
                                        )
                                        .on_hover_text(
                                            "Turns eyebrow inference and output on or off without unloading the model.",
                                        )
                                        .changed()
                                    {
                                        self.pipeline
                                            .eyebrow_enabled
                                            .store(eyebrow_enabled, Ordering::Relaxed);
                                        self.config.ui.eyebrow_enabled = eyebrow_enabled;
                                        let _ = self.config.save(&crate::config::config_path());
                                        self.events.push((
                                            now_hms(),
                                            format!(
                                                "Eyebrow tracking {}",
                                                if eyebrow_enabled {
                                                    "enabled"
                                                } else {
                                                    "disabled"
                                                }
                                            ),
                                            if eyebrow_enabled { OK } else { WARN },
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
                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            egui::CollapsingHeader::new(h3("Connection & models"))
                                .id_salt("settings_assets")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label(prose(
                                        "Runtime files and trained models. These normally stay unchanged after setup.",
                                    ));
                                    ui.add_space(SP2);
                                    settings_path_row(
                                        ui,
                                        "SRanipal model folder",
                                        &mut self.edit.sranipal_dir,
                                        true,
                                    );
                                    settings_path_row(
                                        ui,
                                        "Tobii runtime DLL",
                                        &mut self.edit.tobii_dll,
                                        false,
                                    );
                                    settings_path_row(
                                        ui,
                                        "Eyebrow model (brow.bin)",
                                        &mut self.edit.brow_model,
                                        false,
                                    );
                                    let selected_device =
                                        crate::config::canonical_device_key(&self.edit.device);
                                    if selected_device == "pimax_xr5"
                                        || (selected_device == "auto"
                                            && self.pipeline.device_key == "pimax_xr5")
                                    {
                                        settings_path_row(
                                            ui,
                                            "XR5 EyeWide model (wide.bin)",
                                            &mut self.edit.wide_model,
                                            false,
                                        );
                                    }
                                });
                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            egui::CollapsingHeader::new(h3("Output"))
                                .id_salt("settings_output")
                                .default_open(false)
                                .show(ui, |ui| {
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
                                        do_reload = true;
                                    }
                                    ui.add_space(SP2);
                                    ui.label(label("OSC target"));
                                    ui.horizontal(|ui| {
                                        ui.add(
                                            egui::TextEdit::singleline(
                                                &mut self.edit.osc_host,
                                            )
                                            .desired_width(180.0 * S),
                                        );
                                        ui.add(
                                            egui::DragValue::new(&mut self.edit.osc_port)
                                                .range(1..=u16::MAX),
                                        );
                                        if full_osc {
                                            ui.label(label("included in full OSC"));
                                        }
                                    });
                                });
                        });
                        ui.add_space(SP3);

                        card().show(ui, |ui| {
                            ui.set_width(cw - 2.0 * CARD_PAD);
                            egui::CollapsingHeader::new(h3("Eye mapping"))
                                .id_salt("settings_mapping")
                                .default_open(false)
                                .show(ui, |ui| {
                                    ui.label(prose(&format!(
                                        "Orientation saved for the active device: {}",
                                        self.pipeline.device_key
                                    )));
                                    ui.add_space(SP2);
                                    let mut swap =
                                        self.pipeline.swap_eyes.load(Ordering::Relaxed);
                                    if ui.checkbox(&mut swap, "Swap left / right").changed() {
                                        self.pipeline.swap_eyes.store(swap, Ordering::Relaxed);
                                        self.persist_mapping();
                                    }
                                    let mut flip =
                                        self.pipeline.flip_image.load(Ordering::Relaxed);
                                    if ui
                                        .checkbox(&mut flip, "Flip camera image horizontally")
                                        .changed()
                                    {
                                        self.pipeline
                                            .flip_image
                                            .store(flip, Ordering::Relaxed);
                                        self.persist_mapping();
                                    }
                                    let mut fgx =
                                        self.pipeline.flip_gaze_x.load(Ordering::Relaxed);
                                    if ui
                                        .checkbox(&mut fgx, "Flip gaze left / right")
                                        .changed()
                                    {
                                        self.pipeline
                                            .flip_gaze_x
                                            .store(fgx, Ordering::Relaxed);
                                        self.persist_mapping();
                                    }

                                    ui.add_space(SP2);
                                    ui.label(
                                        egui::RichText::new("Advanced ML handedness")
                                            .size(11.0 * S)
                                            .color(WARN),
                                    );
                                    ui.label(prose(
                                        "Use only when one eye behaves differently because its camera orientation is reversed.",
                                    ));
                                    let mut mml =
                                        self.pipeline.ml_mirror_l.load(Ordering::Relaxed);
                                    if ui
                                        .checkbox(&mut mml, "Mirror LEFT eye for ML")
                                        .changed()
                                    {
                                        self.pipeline
                                            .ml_mirror_l
                                            .store(mml, Ordering::Relaxed);
                                        self.persist_mapping();
                                    }
                                    let mut mmr =
                                        self.pipeline.ml_mirror_r.load(Ordering::Relaxed);
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
                            egui::CollapsingHeader::new(h3("Training tools"))
                                .id_salt("settings_training")
                                .default_open(false)
                                .show(ui, |ui| {
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
                        });
                        ui.add_space(SP3);
                    });
            });
        });

        if do_reload {
            self.apply_and_reload();
        }
    }

    #[allow(dead_code)]
    fn settings_legacy(&mut self, ui: &mut egui::Ui) {
        let (gutter, cw) = stage_metrics(ui.ctx());
        ui.horizontal_top(|ui| {
        ui.add_space(gutter);
        ui.vertical(|ui| {
        ui.set_width(cw);
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.label(h3("Eye mapping"));
            ui.add_space(SP2);
            ui.label(prose(&format!(
                "Per-device orientation for {} — each HMD remembers its own (Pimax flips gaze, Varjo doesn't).",
                self.pipeline.device_key
            )));
            ui.add_space(SP2);
            let mut swap = self.pipeline.swap_eyes.load(Ordering::Relaxed);
            if ui.checkbox(&mut swap, "Swap left / right").changed() {
                self.pipeline.swap_eyes.store(swap, Ordering::Relaxed);
                self.persist_mapping();
            }
            let mut flip = self.pipeline.flip_image.load(Ordering::Relaxed);
            if ui.checkbox(&mut flip, "Flip image horizontally").changed() {
                self.pipeline.flip_image.store(flip, Ordering::Relaxed);
                self.persist_mapping();
            }
            // Gaze handedness differs per device (the pimax / Tobii stream-engine path is
            // mirrored vs Varjo). Live + persisted per-device so it sticks across restarts.
            let mut fgx = self.pipeline.flip_gaze_x.load(Ordering::Relaxed);
            if ui.checkbox(&mut fgx, "Flip gaze left / right").changed() {
                self.pipeline.flip_gaze_x.store(fgx, Ordering::Relaxed);
                self.persist_mapping();
            }
            ui.add_space(SP2);
            ui.label(prose("Experiment: mirror ONE eye's image into the ML only (the eyes are mirror images — matching the model's handedness may steady L/R)."));
            let mut mml = self.pipeline.ml_mirror_l.load(Ordering::Relaxed);
            if ui.checkbox(&mut mml, "Mirror LEFT eye for ML").changed() {
                self.pipeline.ml_mirror_l.store(mml, Ordering::Relaxed);
                self.persist_mapping();
            }
            let mut mmr = self.pipeline.ml_mirror_r.load(Ordering::Relaxed);
            if ui.checkbox(&mut mmr, "Mirror RIGHT eye for ML").changed() {
                self.pipeline.ml_mirror_r.store(mmr, Ordering::Relaxed);
                self.persist_mapping();
            }
        });
        ui.add_space(SP3);
        let mut do_reload = false;
        card().show(ui, |ui| {
            ui.set_width(cw - 2.0 * CARD_PAD);
            ui.label(h3("Assets"));
            ui.add_space(SP2);
            ui.label(prose(
                "The Tobii DLL is distributed separately via Discord — set it below.",
            ));
            ui.add_space(SP2);

            // One editable path row: label + text field + native Browse picker.
            let row = |ui: &mut egui::Ui, name: &str, buf: &mut String, pick_dir: bool| {
                ui.horizontal(|ui| {
                    ui.label(label(name));
                    ui.add(
                        egui::TextEdit::singleline(buf)
                            .desired_width(220.0 * S)
                            .hint_text("(not set)"),
                    );
                    if ui.button("Browse…").clicked() {
                        let dlg = rfd::FileDialog::new();
                        let picked = if pick_dir { dlg.pick_folder() } else { dlg.pick_file() };
                        if let Some(p) = picked {
                            *buf = p.to_string_lossy().into_owned();
                        }
                    }
                });
            };
            // Eye model: point at the SRanipal folder (the weights live inside).
            row(ui, "SRanipal folder", &mut self.edit.sranipal_dir, true);
            ui.add_space(2.0);
            // Common Tobii DLL — REQUIRED to connect (gates Pimax + StarVR).
            row(ui, "Tobii DLL (required to connect Pimax)", &mut self.edit.tobii_dll, false);
            ui.add_space(2.0);
            // Eyebrow model (BROWNET1). Set automatically after a successful Train & bake,
            // but also editable here (point at a brow.bin you already have).
            row(ui, "Eyebrow model (brow.bin)", &mut self.edit.brow_model, false);
            ui.add_space(2.0);
            row(
                ui,
                "XR5 Wide model (wide.bin)",
                &mut self.edit.wide_model,
                false,
            );
            ui.horizontal(|ui| {
                ui.label(label("XR5 Wide source"));
                egui::ComboBox::from_id_salt("settings_wide_source")
                    .selected_text(self.edit.wide_source.as_str())
                    .show_ui(ui, |ui| {
                        for source in WideSource::ALL {
                            ui.selectable_value(
                                &mut self.edit.wide_source,
                                source,
                                source.as_str(),
                            );
                        }
                    });
            });
            let mut eyebrow_enabled = self.pipeline.eyebrow_enabled.load(Ordering::Relaxed);
            if ui
                .checkbox(&mut eyebrow_enabled, "Enable eyebrow tracking")
                .on_hover_text("Turns eyebrow inference and output on or off. The loaded model is kept ready.")
                .changed()
            {
                self.pipeline
                    .eyebrow_enabled
                    .store(eyebrow_enabled, Ordering::Relaxed);
                self.config.ui.eyebrow_enabled = eyebrow_enabled;
                let _ = self.config.save(&crate::config::config_path());
                self.events.push((
                    now_hms(),
                    format!("Eyebrow tracking {}", if eyebrow_enabled { "enabled" } else { "disabled" }),
                    if eyebrow_enabled { OK } else { WARN },
                ));
            }
            let full_osc = self.config.output.osc;
            let mut eyebrow_osc = self.config.output.eyebrow_osc || full_osc;
            let osc_changed = ui
                .add_enabled(
                    !full_osc,
                    egui::Checkbox::new(
                        &mut eyebrow_osc,
                        "Send eyebrows directly to VRChat OSC (FT/v2)",
                    ),
                )
                .on_hover_text(
                    "Sends BrowExpressionLeft/Right, BrowUpLeft/Right, BrowDownLeft/Right, BrowUp, and BrowDown. VRCFT keeps handling eyes and gaze.",
                )
                .changed();
            if osc_changed {
                self.config.output.eyebrow_osc = eyebrow_osc;
                do_reload = true;
            }
            ui.horizontal(|ui| {
                ui.label(label("OSC target"));
                ui.add(
                    egui::TextEdit::singleline(&mut self.edit.osc_host)
                        .desired_width(150.0 * S),
                );
                ui.add(egui::DragValue::new(&mut self.edit.osc_port).range(1..=u16::MAX));
                if full_osc {
                    ui.label(label("included in full OSC"));
                }
            });
            ui.add_space(2.0);
            // B-2 train inputs: the venv-with-torch python + the vr_eyebrow project dir.
            // Used ONLY by the "Train & bake" action on the Eyebrow-calibration tab.
            row(ui, "Python venv (torch, for training)", &mut self.edit.python_exe, false);
            ui.add_space(2.0);
            row(ui, "vr_eyebrow project dir", &mut self.edit.vr_eyebrow_dir, true);

            ui.add_space(SP2);
            ui.horizontal(|ui| {
                ui.label(label("Device:"));
                egui::ComboBox::from_id_salt("device_select")
                    .selected_text(self.edit.device.clone())
                    .show_ui(ui, |ui| {
                        for d in ["auto", "pimax_vr4", "pimax_xr5", "varjo", "varjo_mjpeg", "starvr"] {
                            ui.selectable_value(&mut self.edit.device, d.to_string(), d);
                        }
                    });
            });

            ui.add_space(SP2);
            let mut filter_samples = self
                .be
                .as_ref()
                .map(|s| s.filter_samples.load(Ordering::Relaxed))
                .unwrap_or(u32::from(self.config.output.vrcft_filter_samples))
                .min(30);
            ui.horizontal(|ui| {
                ui.label(label("VRCFT openness low-pass:"));
                let changed = ui
                    .add(
                        egui::Slider::new(&mut filter_samples, 0..=30)
                            .suffix(" samples")
                            .show_value(true),
                    )
                    .on_hover_text(
                        "Live VRCFT-module moving average. 0/1 = pass-through; larger values are smoother but add latency.",
                    )
                    .changed();
                let detail = if filter_samples <= 1 {
                    "OFF".to_string()
                } else {
                    let lag_ms = (filter_samples - 1) as f32 * 1000.0 / (2.0 * 120.0);
                    format!("≈{lag_ms:.0} ms group delay")
                };
                ui.label(egui::RichText::new(detail).monospace().size(10.0 * S).color(TEXT3));
                if changed {
                    self.config.output.vrcft_filter_samples = filter_samples as u8;
                    if let Some(status) = &self.be {
                        status.filter_samples.store(filter_samples, Ordering::Relaxed);
                    }
                    let _ = self.config.save(&crate::config::config_path());
                    self.events.push((
                        now_hms(),
                        format!("VRCFT low-pass set to {filter_samples} samples"),
                        ACCENT,
                    ));
                }
            });

            ui.add_space(SP2);
            ui.horizontal(|ui| {
                if ui.button("Apply & reload").clicked() {
                    do_reload = true;
                }
                if let Some((msg, col)) = &self.reload_msg {
                    ui.label(egui::RichText::new(msg).size(12.0).color(*col));
                }
            });
            ui.add_space(2.0);
            ui.label(prose("Saves to sranibro.toml and restarts the engine in place (no app restart)."));
        });
        if do_reload {
            self.apply_and_reload();
        }
        });
        });
    }

    /// Snapshot the live eye-mapping toggles into the *running* device's saved mapping
    /// and persist. Reading the atomics (not local vars) keeps the stored entry exactly
    /// in sync with the pipeline regardless of which checkbox changed.
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
        self.restore_geometry_preview(false);
        self.geometry_capture.abort();
        self.wide.abort();
        self.geometry_capture_baseline = None;
        self.geometry_capture_filters = None;
        self.geometry_fitter.cancel();
        self.geometry_rollback = None;
        let norm = |s: &str| {
            let t = s.trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        };
        self.config.assets.sranipal_dir = norm(&self.edit.sranipal_dir);
        // Common Tobii DLL; migrate (clear) the legacy per-device fields so the
        // saved config has a single source of truth.
        self.config.assets.tobii_dll = norm(&self.edit.tobii_dll);
        self.config.assets.starvr_dll = None;
        self.config.assets.pimax_vr4_dll = None;
        self.config.assets.brow_model = norm(&self.edit.brow_model);
        self.config.assets.wide_model = norm(&self.edit.wide_model);
        self.config.assets.python_exe = norm(&self.edit.python_exe);
        self.config.assets.vr_eyebrow_dir = norm(&self.edit.vr_eyebrow_dir);
        self.config.hmd.device = self.edit.device.trim().to_string();
        self.config.hmd.wide_source = self.edit.wide_source;
        self.config
            .set_gaze_source("pimax_xr5", self.edit.gaze_source);
        self.config.output.osc_host = self.edit.osc_host.trim().to_string();
        self.config.output.osc_port = self.edit.osc_port;

        let path = crate::config::config_path();
        if let Err(e) = self.config.save(&path) {
            self.reload_msg = Some((format!("save failed: {e}"), ERR));
            return;
        }

        // Tear down first: releases the WinUSB handle and TCP port 5555 so the
        // rebuild can re-acquire them cleanly (the user opted into live re-init).
        self.pipeline.stop();
        match crate::engine::build_engine(&self.config) {
            Ok(eng) => {
                self.pipeline = eng.pipeline;
                self.tele = self.pipeline.tele.clone();
                self.be = eng.be_status;
                self.tex_l = None;
                self.tex_r = None;
                self.last_eye_texture_upload = Instant::now() - Duration::from_secs(1);
                self.last = [0; 5];
                self.last_t = Instant::now();
                self.events.push((now_hms(), "Assets reloaded".into(), OK));
                self.reload_msg = Some(("reloaded ✓".into(), OK));
            }
            Err(e) => {
                self.reload_msg = Some((
                    format!("reload failed: {e} — fix the path and Apply again"),
                    ERR,
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Widgets + icons
// ---------------------------------------------------------------------------

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

/// The wide/squeeze LINK toggle, drawn as a small padlock at the seam between the two
/// rows: closed shackle + ACCENT body = LOCKED (exclusive), raised/open shackle + dim =
/// unlocked (independent). Painted as an overlay (no layout allocated); returns true on
/// click.
fn chain_marker(ui: &mut egui::Ui, center: Pos2, on: bool) -> bool {
    let hit = Rect::from_center_size(center, vec2(20.0 * S, 26.0 * S));
    let resp = ui.interact(hit, Id::new("ws_chain"), Sense::click());
    let (hov, clicked) = (resp.hovered(), resp.clicked());
    if hov {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let col = if on {
        ACCENT
    } else if hov {
        TEXT2
    } else {
        TEXT3
    };
    let stroke = Stroke::new(1.5 * S, col);
    let p = ui.painter();
    let cx = center.x;
    // Body (rounded rect) — the lower half of the padlock.
    let (bw, bh) = (11.0 * S, 8.5 * S);
    let body = Rect::from_min_size(pos2(cx - bw * 0.5, center.y + 1.0 * S), vec2(bw, bh));
    if on {
        p.rect_filled(body, 2.0 * S, col);
        p.circle_filled(pos2(cx, body.center().y), 1.3 * S, BG); // keyhole punched out
    } else {
        p.rect_stroke(body, 2.0 * S, stroke);
        p.circle_filled(pos2(cx, body.center().y), 1.1 * S, col);
    }
    // Shackle: inverted-U (top arc + two legs). Locked = legs meet the body; unlocked =
    // whole shackle raised (a visible gap) with the right leg swung short = "open".
    let sr = bw * 0.30;
    let lift = if on { 0.0 } else { 2.6 * S };
    let arc_cy = body.top() - 3.0 * S - lift;
    let leg_bottom = body.top() - lift;
    let n = 12;
    let arc: Vec<Pos2> = (0..=n)
        .map(|i| {
            let a = std::f32::consts::PI * (1.0 + i as f32 / n as f32);
            pos2(cx + sr * a.cos(), arc_cy + sr * a.sin())
        })
        .collect();
    p.add(egui::Shape::line(arc, stroke));
    p.line_segment([pos2(cx - sr, arc_cy), pos2(cx - sr, leg_bottom)], stroke);
    let right_bottom = if on { leg_bottom } else { arc_cy + 2.0 * S };
    p.line_segment([pos2(cx + sr, arc_cy), pos2(cx + sr, right_bottom)], stroke);
    resp.on_hover_text(if on {
        "wide / squeeze: LOCKED — only the stronger fires"
    } else {
        "wide / squeeze: unlocked (independent) — click to link"
    });
    clicked
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
    chain: &mut bool,
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
        // Footer = separator(1) + gap(10*S) + one flat diagnostics row(14*S). No
        // nested tiles (was bordered mini-cells inside this bordered card).
        let footer_h = 24.0 * S + 1.0;
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
        // Seam between the wide and squeeze rows: where the wide/squeeze "chain" sits.
        let seam_y = ui.cursor().top() + row_gap * 0.5;
        let label_right = ui.min_rect().left() + 64.0 * S;
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
        // Overlay the wide/squeeze link glyph — painted + interacted only (no row
        // allocated), so the 4-row height math above stays exact. Right of the label
        // gutter, centered on the seam.
        if chain_marker(ui, pos2(label_right - 11.0 * S, seam_y), *chain) {
            *chain = !*chain;
        }
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
        // Flat diagnostics row (no nested tiles): real-time health only. DROP = % of
        // 120Hz emit cycles that ran late (0 = keeping up). FRAME counter removed.
        let (row, _) = ui.allocate_exact_size(vec2(inner, 14.0 * S), Sense::hover());
        let p = ui.painter();
        let cy = row.center().y;
        let dc = if drop < 0.5 { OK } else { WARN };
        p.text(
            pos2(row.left(), cy),
            Align2::LEFT_CENTER,
            "DROP",
            FontId::monospace(9.0 * S),
            TEXT2,
        );
        p.text(
            pos2(row.left() + 42.0 * S, cy),
            Align2::LEFT_CENTER,
            format!("{drop:.1}%"),
            FontId::monospace(10.0 * S),
            dc,
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn eye_cams_card(
    ui: &mut egui::Ui,
    w: f32,
    min_inner: f32,
    frames: &[Option<(u32, u32, Vec<u8>)>; 2],
    ml_frames: &[Option<Vec<u8>>; 2],
    net_view: &mut bool,
    pupil: &[(f32, bool); 2],
    cam_rates: [f32; 2],
    eye_w: u32,
    eye_h: u32,
    tex_l: &mut Option<egui::TextureHandle>,
    tex_r: &mut Option<egui::TextureHandle>,
    upload_textures: bool,
    ctx: &egui::Context,
) -> bool {
    let inner = w - 2.0 * CARD_PAD;
    // Resolution label: the live frame's real dims when streaming, else the device
    // profile's nominal — per-HMD (VR4/StarVR 200x200, Varjo higher), not hardcoded.
    let (dw, dh) = frames
        .iter()
        .flatten()
        .next()
        .map(|(fw, fh, _)| (*fw, *fh))
        .unwrap_or((eye_w, eye_h));
    let n = crate::ml::preprocess::DST as u32;
    card().show(ui, |ui| {
        ui.set_width(inner);
        ui.set_min_height(min_inner);
        // Header: a small gear (opens ML-input settings) at the far left, then the title,
        // then "<w>x<h> IR" + the RAW/NET source toggle pinned right.
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
                    // Source toggle: RAW cameras vs the exact image sent to the eye
                    // net (all filters + geometry, shown un-mirrored).
                    let txt = if *net_view { "NET" } else { "RAW" };
                    let color = if *net_view { ACCENT } else { TEXT3 };
                    let tr = ui
                        .add(egui::Button::new(
                            egui::RichText::new(txt).monospace().size(9.0 * S).strong().color(color),
                        )
                        .small()
                        .fill(INNER))
                        .on_hover_text("Toggle: raw cameras / the exact image sent to the eye net (correct orientation)");
                    if tr.clicked() {
                        *net_view = !*net_view;
                    }
                    ui.add_space(4.0 * S);
                    let res = if *net_view { format!("{n}x{n} NET") } else { format!("{dw}x{dh} IR") };
                    ui.label(egui::RichText::new(res).monospace().size(9.0 * S).color(TEXT3));
                });
                gr.clicked()
            })
            .inner;
        ui.add_space(9.0 * S);
        // Big L/R images: each fills half the (now equal-width) card.
        let box_w = ((inner - 9.0 * S) / 2.0).floor();
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 9.0 * S;
            if *net_view {
                let fl = ml_frames[0].as_ref().map(|p| (n, n, p.clone()));
                let fr = ml_frames[1].as_ref().map(|p| (n, n, p.clone()));
                eye_cam_box(ui, "L", &fl, pupil[0], cam_rates[0], tex_l, upload_textures, ctx, "eye_l", box_w);
                eye_cam_box(ui, "R", &fr, pupil[1], cam_rates[1], tex_r, upload_textures, ctx, "eye_r", box_w);
            } else {
                eye_cam_box(ui, "L", &frames[0], pupil[0], cam_rates[0], tex_l, upload_textures, ctx, "eye_l", box_w);
                eye_cam_box(ui, "R", &frames[1], pupil[1], cam_rates[1], tex_r, upload_textures, ctx, "eye_r", box_w);
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
    frame: &Option<(u32, u32, Vec<u8>)>,
    pupil: (f32, bool),
    rate: f32,
    slot: &mut Option<egui::TextureHandle>,
    upload_texture: bool,
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
                let (iw, ih) = (*fw as usize, *fh as usize);
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
    Brow,
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
        Icon::Brow => {
            // A raised eyebrow arc over a simple eye — distinguishes the brow-calibration
            // tab from the (Sliders) eye-tuning tab.
            line(
                vec![
                    pt(0.18, 0.34),
                    pt(0.38, 0.22),
                    pt(0.62, 0.22),
                    pt(0.82, 0.34),
                ],
                bold,
            );
            line(
                vec![
                    pt(0.14, 0.62),
                    pt(0.30, 0.50),
                    pt(0.50, 0.47),
                    pt(0.70, 0.50),
                    pt(0.86, 0.62),
                ],
                st,
            );
            line(
                vec![
                    pt(0.14, 0.62),
                    pt(0.30, 0.74),
                    pt(0.50, 0.77),
                    pt(0.70, 0.74),
                    pt(0.86, 0.62),
                ],
                st,
            );
            p.circle_filled(pt(0.5, 0.62), size * 0.10, col);
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
