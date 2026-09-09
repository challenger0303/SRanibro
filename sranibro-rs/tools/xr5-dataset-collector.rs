//! Standalone Dream Air / XR5 eyelid-model dataset collector.
//!
//! This binary deliberately does not construct SRanibro's model, post-processor,
//! VRCFT server, OSC output, fitters, or saved calibration. It opens only the XR5
//! camera/native-data adapter, guides one labelled session, and writes a local ZIP.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("SRanibro XR5 Dataset Collector is available on Windows only.");
}

#[cfg(all(windows, feature = "xr5-model-service-client"))]
#[path = "xr5-model-service-client.rs"]
mod model_service_client;

#[cfg(windows)]
mod app {
    #[cfg(feature = "xr5-model-service-client")]
    use crate::model_service_client;

    use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2};
    use sranibro_rs::config::{Config, EyeMapping};
    use sranibro_rs::core::types::{Eye, EyeSample, GazeSample};
    use sranibro_rs::device::{self, HmdAdapter};
    use sranibro_rs::gaze_residual_calib::{
        CaptureAction, CaptureProtocol, GazeResidualCapture, Status as CaptureStatus,
    };
    use sranibro_rs::geometry_calib::GazeTarget;
    use sranibro_rs::recording_audio::{Cue as RecordingCue, RecordingAudio};
    use sranibro_rs::vr_research_overlay::{
        VrEyePose, VrGuideState, VrResearchOverlay, VrTargetFrame,
    };
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

    const DEVICE_KEY: &str = "pimax_xr5";
    const NATIVE_FRESHNESS: Duration = Duration::from_millis(150);
    const FRAME_FRESHNESS: Duration = Duration::from_secs(1);
    const UI_HEARTBEAT_LIMIT: Duration = Duration::from_millis(300);
    const PREVIEW_INTERVAL: Duration = Duration::from_millis(67);

    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    enum FitCondition {
        #[default]
        Normal,
        Reseated,
        AlternateOptics,
    }

    impl FitCondition {
        const ALL: [Self; 3] = [Self::Normal, Self::Reseated, Self::AlternateOptics];

        fn label(self) -> &'static str {
            match self {
                Self::Normal => "Normal comfortable fit",
                Self::Reseated => "After removing and reseating the HMD",
                Self::AlternateOptics => "Glasses / contacts / alternate fit",
            }
        }

        fn slug(self) -> &'static str {
            match self {
                Self::Normal => "normal",
                Self::Reseated => "reseated",
                Self::AlternateOptics => "alternate_optics",
            }
        }
    }

    #[derive(Clone)]
    struct RawFrame {
        generation: u64,
        width: u32,
        height: u32,
        pixels: Arc<[u8]>,
        received_at: Instant,
    }

    struct SavedCapture {
        path: PathBuf,
        session_id: String,
    }

    #[derive(Clone)]
    struct NativeState {
        sample: GazeSample,
        gaze_valid_at: [Option<Instant>; 2],
        pupil_pos_at: [Option<Instant>; 2],
        openness_at: [Option<Instant>; 2],
        timestamp_at: Option<Instant>,
    }

    impl Default for NativeState {
        fn default() -> Self {
            Self {
                sample: GazeSample::default(),
                gaze_valid_at: [None; 2],
                pupil_pos_at: [None; 2],
                openness_at: [None; 2],
                timestamp_at: None,
            }
        }
    }

    impl NativeState {
        fn merge(&mut self, mut source: GazeSample, swap_eyes: bool, now: Instant) {
            if swap_eyes {
                std::mem::swap(&mut source.left, &mut source.right);
            }
            if source.timestamp_us != 0 {
                self.sample.timestamp_us = source.timestamp_us;
                self.timestamp_at = Some(now);
            }
            merge_eye(
                &mut self.sample.left,
                source.left,
                0,
                now,
                &mut self.gaze_valid_at,
                &mut self.pupil_pos_at,
                &mut self.openness_at,
            );
            merge_eye(
                &mut self.sample.right,
                source.right,
                1,
                now,
                &mut self.gaze_valid_at,
                &mut self.pupil_pos_at,
                &mut self.openness_at,
            );
        }

        fn fresh_sample(&self, now: Instant) -> GazeSample {
            let mut sample = self.sample;
            for index in 0..2 {
                let eye = if index == 0 {
                    &mut sample.left
                } else {
                    &mut sample.right
                };
                if !is_fresh(self.gaze_valid_at[index], now) {
                    eye.gaze_valid = false;
                    eye.gaze_reported = false;
                }
                if !is_fresh(self.pupil_pos_at[index], now) {
                    eye.pupil_pos_valid = false;
                    eye.pupil_pos_reported = false;
                }
                if !is_fresh(self.openness_at[index], now) {
                    eye.openness_valid = false;
                    eye.openness_reported = false;
                }
            }
            if !is_fresh(self.timestamp_at, now) {
                sample.timestamp_us = 0;
            }
            sample
        }
    }

    #[derive(Default)]
    struct SourceState {
        frames: [Option<RawFrame>; 2],
        native: NativeState,
        frame_count: [u64; 2],
        gaze_count: u64,
    }

    impl SourceState {
        fn generations(&self) -> [u64; 2] {
            std::array::from_fn(|eye| {
                self.frames[eye]
                    .as_ref()
                    .map(|frame| frame.generation)
                    .unwrap_or(0)
            })
        }

        fn stereo_ready(&self, now: Instant) -> bool {
            self.frames.iter().all(|frame| {
                frame
                    .as_ref()
                    .is_some_and(|frame| now.duration_since(frame.received_at) <= FRAME_FRESHNESS)
            })
        }
    }

    #[derive(Clone)]
    struct LiveState {
        connection: String,
        detail: String,
        connected: bool,
        stereo_ready: bool,
        mock: bool,
        frame_hz: [f32; 2],
        gaze_hz: f32,
        frames: [Option<RawFrame>; 2],
        capture_status: CaptureStatus,
        capture_error: Option<String>,
        saving: bool,
        saved_path: Option<PathBuf>,
        saved_session_id: Option<String>,
        saved_condition: Option<FitCondition>,
        save_error: Option<String>,
        #[cfg(feature = "xr5-model-service-client")]
        service_origin: Option<String>,
        #[cfg(feature = "xr5-model-service-client")]
        service_error: Option<String>,
        #[cfg(feature = "xr5-model-service-client")]
        upload_state: UploadState,
    }

    impl LiveState {
        fn new(mock: bool) -> Self {
            Self {
                connection: if mock {
                    "Starting simulated XR5 camera...".into()
                } else {
                    "Connecting to Dream Air / XR5...".into()
                },
                detail: String::new(),
                connected: false,
                stereo_ready: false,
                mock,
                frame_hz: [0.0; 2],
                gaze_hz: 0.0,
                frames: [None, None],
                capture_status: CaptureStatus::Idle,
                capture_error: None,
                saving: false,
                saved_path: None,
                saved_session_id: None,
                saved_condition: None,
                save_error: None,
                #[cfg(feature = "xr5-model-service-client")]
                service_origin: None,
                #[cfg(feature = "xr5-model-service-client")]
                service_error: None,
                #[cfg(feature = "xr5-model-service-client")]
                upload_state: UploadState::Unavailable,
            }
        }
    }

    #[cfg(feature = "xr5-model-service-client")]
    #[derive(Clone, Debug)]
    enum UploadState {
        Unavailable,
        Ready,
        Uploading {
            sent: u64,
            total: u64,
        },
        Uploaded {
            contribution_id: String,
            sample_count: Option<usize>,
        },
        Failed(String),
    }

    enum WorkerCommand {
        Prepare(FitCondition),
        Begin,
        Continue,
        TogglePause,
        Abort,
        RetrySave,
        #[cfg(feature = "xr5-model-service-client")]
        Upload {
            path: PathBuf,
            session_id: String,
            condition: FitCondition,
        },
        Reconnect,
        Stop,
    }

    struct Shared {
        ui: Mutex<LiveState>,
        source: Mutex<SourceState>,
        ui_heartbeat: Mutex<Instant>,
        overlay_requested: AtomicBool,
        overlay_visible_ever: AtomicBool,
        #[cfg(feature = "xr5-model-service-client")]
        service: Option<model_service_client::ServiceClient>,
    }

    impl Shared {
        fn new(mock: bool) -> Self {
            #[allow(unused_mut)]
            let mut ui = LiveState::new(mock);
            #[cfg(feature = "xr5-model-service-client")]
            let service = match model_service_client::ServiceClient::configured(mock) {
                Ok(service) => {
                    ui.service_origin = service
                        .as_ref()
                        .map(|client| client.display_origin().to_owned());
                    ui.upload_state = if service.is_some() {
                        UploadState::Ready
                    } else {
                        UploadState::Unavailable
                    };
                    service
                }
                Err(error) => {
                    ui.service_error = Some(error);
                    None
                }
            };
            Self {
                ui: Mutex::new(ui),
                source: Mutex::new(SourceState::default()),
                ui_heartbeat: Mutex::new(Instant::now()),
                overlay_requested: AtomicBool::new(true),
                overlay_visible_ever: AtomicBool::new(false),
                #[cfg(feature = "xr5-model-service-client")]
                service,
            }
        }

        fn heartbeat(&self) {
            *self
                .ui_heartbeat
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Instant::now();
        }

        fn heartbeat_fresh(&self, now: Instant) -> bool {
            now.duration_since(
                *self
                    .ui_heartbeat
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            ) <= UI_HEARTBEAT_LIMIT
        }
    }

    struct CollectorApp {
        shared: Arc<Shared>,
        commands: Sender<WorkerCommand>,
        worker: Option<thread::JoinHandle<()>>,
        texture: [Option<egui::TextureHandle>; 2],
        texture_generation: [u64; 2],
        last_preview_upload: Instant,
        show_preview: bool,
        consent: bool,
        condition: FitCondition,
        steamvr_guide: bool,
        overlay: VrResearchOverlay,
        audio: RecordingAudio,
        last_audio_key: String,
        last_error_key: String,
        always_on_top: bool,
        #[cfg(feature = "xr5-model-service-client")]
        upload_consent: bool,
    }

    impl CollectorApp {
        fn new(ctx: &egui::Context, mock: bool) -> Self {
            sranibro_rs::theme::apply(ctx);
            let shared = Arc::new(Shared::new(mock));
            let (commands, receiver) = mpsc::channel();
            let worker_shared = shared.clone();
            let worker = thread::Builder::new()
                .name("xr5-dataset-collector".into())
                .spawn(move || worker_main(worker_shared, receiver, mock))
                .ok();
            if worker.is_none() {
                let mut state = shared.ui.lock().unwrap();
                state.connection = "Collector worker could not start".into();
                state.detail = "Close the app and try again.".into();
            }
            Self {
                shared,
                commands,
                worker,
                texture: [None, None],
                texture_generation: [0; 2],
                last_preview_upload: Instant::now() - PREVIEW_INTERVAL,
                show_preview: true,
                consent: false,
                condition: FitCondition::Normal,
                steamvr_guide: true,
                overlay: VrResearchOverlay::new(),
                audio: RecordingAudio::new(),
                last_audio_key: String::new(),
                last_error_key: String::new(),
                always_on_top: false,
                #[cfg(feature = "xr5-model-service-client")]
                upload_consent: false,
            }
        }

        fn upload_preview(&mut self, ctx: &egui::Context, state: &LiveState) {
            if !self.show_preview
                || capture_is_active(&state.capture_status)
                || self.last_preview_upload.elapsed() < PREVIEW_INTERVAL
            {
                return;
            }
            self.last_preview_upload = Instant::now();
            for eye in 0..2 {
                let Some(frame) = state.frames[eye].as_ref() else {
                    continue;
                };
                if frame.generation == self.texture_generation[eye] {
                    continue;
                }
                let need = frame.width as usize * frame.height as usize;
                if frame.width == 0 || frame.height == 0 || frame.pixels.len() < need {
                    continue;
                }
                self.texture_generation[eye] = frame.generation;
                let image = egui::ColorImage::from_gray(
                    [frame.width as usize, frame.height as usize],
                    &frame.pixels[..need],
                );
                match self.texture[eye].as_mut() {
                    Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                    None => {
                        self.texture[eye] = Some(ctx.load_texture(
                            if eye == 0 {
                                "xr5_collector_left"
                            } else {
                                "xr5_collector_right"
                            },
                            image,
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
            }
        }

        fn sync_overlay(&mut self, state: &LiveState) {
            let frame = overlay_frame(&state.capture_status);
            self.overlay.present(self.steamvr_guide, frame);
            self.shared
                .overlay_requested
                .store(self.steamvr_guide, Ordering::Relaxed);
            if self.overlay.is_visible() {
                self.shared
                    .overlay_visible_ever
                    .store(true, Ordering::Relaxed);
            }
        }

        fn sync_audio(&mut self, state: &LiveState) {
            let (key, cue) = audio_state(&state.capture_status, state.saving, &state.saved_path);
            if key != self.last_audio_key {
                if !self.last_audio_key.is_empty() {
                    if let Some(cue) = cue {
                        self.audio.cue(cue, true);
                    }
                }
                self.last_audio_key = key;
            }
            if state.capture_error.is_some() || state.save_error.is_some() {
                let error_key = format!("{:?}:{:?}", state.capture_error, state.save_error);
                if self.last_error_key != error_key {
                    self.audio.cue(RecordingCue::Warning, true);
                    self.last_error_key = error_key;
                }
            } else {
                self.last_error_key.clear();
            }
        }

        fn sync_window_level(&mut self, ctx: &egui::Context, state: &LiveState) {
            let active = capture_is_active(&state.capture_status) || state.saving;
            if active == self.always_on_top {
                return;
            }
            self.always_on_top = active;
            ctx.send_viewport_cmd(egui::ViewportCommand::WindowLevel(if active {
                egui::viewport::WindowLevel::AlwaysOnTop
            } else {
                egui::viewport::WindowLevel::Normal
            }));
        }

        fn draw_header(&self, ui: &mut egui::Ui, state: &LiveState) {
            ui.horizontal(|ui| {
                ui.heading("SRanibro XR5 Dataset Collector");
                ui.label(
                    RichText::new(if state.mock {
                        "SIMULATION"
                    } else {
                        "COLLECTION ONLY"
                    })
                    .strong()
                    .color(if state.mock {
                        Color32::YELLOW
                    } else {
                        Color32::from_rgb(95, 220, 150)
                    }),
                );
            });
            ui.label(
                "Collects raw Dream Air / XR5 stereo eye images for the native eyelid model. It never changes calibration, geometry, tracking output, or model settings.",
            );
            ui.add_space(8.0);
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new(&state.connection).strong().color(
                        if state.stereo_ready {
                            Color32::from_rgb(95, 220, 150)
                        } else {
                            Color32::from_rgb(255, 194, 90)
                        },
                    ));
                    ui.separator();
                    ui.label(format!(
                        "camera L {:.0}/s  R {:.0}/s    native data {:.0}/s",
                        state.frame_hz[0], state.frame_hz[1], state.gaze_hz
                    ));
                });
                if !state.detail.is_empty() {
                    ui.label(RichText::new(&state.detail).color(Color32::LIGHT_GRAY));
                }
                if !state.stereo_ready
                    && !capture_is_active(&state.capture_status)
                    && ui.button("Reconnect").clicked()
                {
                    let _ = self.commands.send(WorkerCommand::Reconnect);
                }
            });
        }

        fn draw_preview(&mut self, ui: &mut egui::Ui, state: &LiveState) {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_preview, "Show eye-camera preview");
                ui.label(
                    RichText::new("Automatically hidden while recording to reduce GPU load.")
                        .small(),
                );
            });
            if !self.show_preview || capture_is_active(&state.capture_status) {
                return;
            }
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let side = ((available - 16.0) * 0.5).clamp(160.0, 240.0);
                for eye in 0..2 {
                    ui.vertical(|ui| {
                        ui.label(RichText::new(if eye == 0 { "LEFT" } else { "RIGHT" }).strong());
                        let (rect, _) = ui.allocate_exact_size(Vec2::splat(side), Sense::hover());
                        ui.painter().rect_filled(rect, 8.0, Color32::from_gray(12));
                        if let Some(texture) = self.texture[eye].as_ref() {
                            egui::Image::new(egui::load::SizedTexture::new(
                                texture.id(),
                                Vec2::splat(side),
                            ))
                            .paint_at(ui, rect);
                        } else {
                            ui.painter().text(
                                rect.center(),
                                Align2::CENTER_CENTER,
                                "waiting for image",
                                FontId::proportional(14.0),
                                Color32::GRAY,
                            );
                        }
                        ui.painter().rect_stroke(
                            rect,
                            8.0,
                            Stroke::new(1.0, Color32::from_gray(65)),
                        );
                    });
                }
            });
        }

        fn draw_idle(&mut self, ui: &mut egui::Ui, state: &LiveState) {
            ui.heading("One complete recording");
            ui.label(format!(
                "About {:.1} minutes plus time to read each instruction. The session contains separate training and untouched validation repetitions.",
                CaptureProtocol::PythonEyelidDataset.total_seconds() / 60.0
            ));
            ui.add_space(8.0);
            ui.label(RichText::new("Before starting").strong());
            ui.label("1. Complete the normal Pimax/Tobii gaze calibration if available.");
            ui.label("2. Wear the HMD in a comfortable, stable position.");
            ui.label("3. Keep your head still and move only your eyes when a target appears.");
            ui.label(
                "4. Use gentle eyelid motion unless the instruction explicitly asks for a wink.",
            );
            ui.add_space(10.0);

            ui.horizontal(|ui| {
                ui.label("Session condition");
                egui::ComboBox::from_id_salt("xr5_dataset_session_condition")
                    .selected_text(self.condition.label())
                    .show_ui(ui, |ui| {
                        for condition in FitCondition::ALL {
                            ui.selectable_value(&mut self.condition, condition, condition.label());
                        }
                    });
            });
            ui.label(
                RichText::new("For useful generalization, repeat later after taking the HMD off and putting it back on.")
                    .small()
                    .color(Color32::LIGHT_GRAY),
            );
            ui.add_space(10.0);

            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Biometric-data consent").strong());
                ui.label(
                    "The ZIP contains raw infrared images of both eyes. It stays on this PC and is never uploaded automatically. Share it only if you want it used to develop and evaluate SRanibro's XR5 eyelid model.",
                );
                ui.add_space(4.0);
                ui.checkbox(
                    &mut self.consent,
                    "I understand and consent to this recording being used for SRanibro eyelid-model development.",
                );
            });

            ui.add_space(8.0);
            ui.checkbox(
                &mut self.steamvr_guide,
                "Show the large head-locked guide inside SteamVR",
            );
            ui.label(RichText::new(self.overlay.status_text()).small());
            ui.add_space(8.0);
            let can_start = state.stereo_ready && self.consent && !state.saving;
            if ui
                .add_enabled(
                    can_start,
                    egui::Button::new("Prepare recording").min_size(Vec2::new(190.0, 38.0)),
                )
                .clicked()
            {
                self.overlay.reset_session();
                self.shared
                    .overlay_visible_ever
                    .store(false, Ordering::Relaxed);
                let _ = self.commands.send(WorkerCommand::Prepare(self.condition));
            }
            if !state.stereo_ready {
                ui.colored_label(
                    Color32::from_rgb(255, 194, 90),
                    "Both fresh 200x200 camera streams are required before recording.",
                );
            } else if !self.consent {
                ui.small("Confirm the biometric-data notice to enable recording.");
            }
        }

        fn draw_capture(&mut self, ui: &mut egui::Ui, state: &LiveState) {
            match &state.capture_status {
                CaptureStatus::Ready { instruction } => {
                    ui.heading("Ready to begin");
                    ui.label(instruction);
                    ui.add_space(8.0);
                    ui.label("The first tone announces a new instruction. The higher tone marks recording.");
                    ui.label("Manual explanation screens wait until you press Continue, so there is no reading-time limit.");
                    ui.add_space(10.0);
                    if ui
                        .add(egui::Button::new("Begin  (Space)").min_size(Vec2::new(180.0, 40.0)))
                        .clicked()
                    {
                        let _ = self.commands.send(WorkerCommand::Begin);
                    }
                    if ui.button("Cancel").clicked() {
                        let _ = self.commands.send(WorkerCommand::Abort);
                    }
                }
                CaptureStatus::Running {
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
                    awaiting_confirmation,
                    phase_remaining_s,
                    pose_progress,
                    instruction,
                    ..
                } => {
                    let state_label = if *paused {
                        "PAUSED"
                    } else if *awaiting_confirmation {
                        "READ THIS STEP"
                    } else if *recording && !*settling {
                        "RECORDING"
                    } else {
                        "GET READY"
                    };
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new(state_label).strong().size(19.0).color(
                            if *recording && !*settling && !*paused {
                                Color32::from_rgb(95, 220, 150)
                            } else {
                                Color32::from_rgb(80, 195, 255)
                            },
                        ));
                        ui.separator();
                        ui.label(RichText::new(action_title(*action, *target)).size(19.0));
                        if *holdout {
                            ui.label(RichText::new("REPEAT CHECK").small().strong());
                        }
                    });
                    ui.label(RichText::new(instruction).size(16.0));
                    ui.add_space(7.0);
                    draw_target(ui, *target, *pose_progress, *paused);
                    ui.add_space(7.0);
                    ui.add(egui::ProgressBar::new(*progress).show_percentage());
                    ui.horizontal_wrapped(|ui| {
                        if !*awaiting_confirmation {
                            ui.label(format!(
                                "Current step {:.1}s    whole session {:.0}s remaining    {} samples",
                                phase_remaining_s, remaining_s, samples_in_phase
                            ));
                        }
                    });
                    if *stereo_stalled {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            "Eye-camera frames stopped. The timer is waiting; check the headset connection.",
                        );
                    }
                    if let Some(error) = &state.capture_error {
                        ui.colored_label(Color32::LIGHT_RED, error);
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if *awaiting_confirmation {
                            if ui
                                .add(
                                    egui::Button::new("Continue  (Space)")
                                        .min_size(Vec2::new(180.0, 38.0)),
                                )
                                .clicked()
                            {
                                let _ = self.commands.send(WorkerCommand::Continue);
                            }
                        } else if ui
                            .button(if *paused { "Resume" } else { "Pause" })
                            .clicked()
                        {
                            let _ = self.commands.send(WorkerCommand::TogglePause);
                        }
                        if ui.button("Cancel and discard").clicked() {
                            let _ = self.commands.send(WorkerCommand::Abort);
                        }
                    });
                }
                CaptureStatus::Done {
                    samples,
                    evidence_complete,
                    missing_phases,
                } => {
                    ui.heading(if state.saving {
                        "Saving biometric ZIP..."
                    } else if state.saved_path.is_some() {
                        "Recording saved"
                    } else {
                        "Recording complete"
                    });
                    ui.label(format!("{samples} labelled stereo frames collected."));
                    if !*evidence_complete {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            format!(
                                "Evidence is incomplete; missing phase IDs: {missing_phases:?}"
                            ),
                        );
                    }
                    if state.saving {
                        ui.spinner();
                        ui.label("PNG and ZIP encoding runs in the background-priority collector thread. Tracking output is not running in this app.");
                    }
                    if let Some(path) = &state.saved_path {
                        ui.colored_label(
                            Color32::from_rgb(95, 220, 150),
                            format!("Saved: {}", path.display()),
                        );
                        #[cfg(feature = "xr5-model-service-client")]
                        self.draw_upload(ui, state, *evidence_complete, path);
                        #[cfg(not(feature = "xr5-model-service-client"))]
                        ui.label("Send this ZIP only to the model maintainer. This collector contains no upload or training code in the current build.");
                        ui.horizontal(|ui| {
                            if ui.button("Open folder").clicked() {
                                open_parent(path);
                            }
                            #[cfg(feature = "xr5-model-service-client")]
                            let upload_in_progress =
                                matches!(state.upload_state, UploadState::Uploading { .. });
                            #[cfg(not(feature = "xr5-model-service-client"))]
                            let upload_in_progress = false;
                            if ui
                                .add_enabled(
                                    !upload_in_progress,
                                    egui::Button::new("Record another session"),
                                )
                                .clicked()
                            {
                                self.consent = false;
                                #[cfg(feature = "xr5-model-service-client")]
                                {
                                    self.upload_consent = false;
                                }
                                let _ = self.commands.send(WorkerCommand::Abort);
                            }
                        });
                    }
                    if let Some(error) = &state.save_error {
                        ui.colored_label(Color32::LIGHT_RED, format!("Save failed: {error}"));
                        if ui.button("Retry save").clicked() {
                            let _ = self.commands.send(WorkerCommand::RetrySave);
                        }
                    }
                }
                CaptureStatus::Idle => self.draw_idle(ui, state),
            }
        }

        #[cfg(feature = "xr5-model-service-client")]
        fn draw_upload(
            &mut self,
            ui: &mut egui::Ui,
            state: &LiveState,
            evidence_complete: bool,
            path: &Path,
        ) {
            ui.add_space(8.0);
            let Some(origin) = state.service_origin.as_deref() else {
                ui.label("Online contribution is not configured in this build. Send the ZIP to the model maintainer manually.");
                if let Some(error) = &state.service_error {
                    ui.colored_label(Color32::LIGHT_RED, error);
                }
                return;
            };
            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Optional model-development contribution").strong());
                ui.label(format!("Destination: {origin}"));
                ui.label("This sends the saved raw infrared eye-image ZIP. It does not train anything on this PC and never changes your current tracking model.");
                ui.checkbox(
                    &mut self.upload_consent,
                    "I choose to send this biometric recording for SRanibro XR5 model development and later model distribution.",
                );
                match &state.upload_state {
                    UploadState::Uploading { sent, total } => {
                        let progress = if *total == 0 {
                            0.0
                        } else {
                            *sent as f32 / *total as f32
                        };
                        ui.add(egui::ProgressBar::new(progress.clamp(0.0, 1.0)).show_percentage());
                        ui.label(format!("Sending {:.1} / {:.1} MiB", *sent as f64 / 1_048_576.0, *total as f64 / 1_048_576.0));
                    }
                    UploadState::Uploaded {
                        contribution_id,
                        sample_count,
                    } => {
                        ui.colored_label(
                            Color32::from_rgb(95, 220, 150),
                            format!(
                                "Received by the model service: {contribution_id}{}",
                                sample_count
                                    .map(|count| format!("  ({count} stereo samples)"))
                                    .unwrap_or_default()
                            ),
                        );
                        ui.label("A private upload receipt, including the deletion token, was saved next to the ZIP.");
                    }
                    UploadState::Failed(error) => {
                        ui.colored_label(Color32::LIGHT_RED, format!("Send failed: {error}"));
                    }
                    UploadState::Ready | UploadState::Unavailable => {}
                }
                let can_send = evidence_complete
                    && self.upload_consent
                    && matches!(
                        state.upload_state,
                        UploadState::Ready | UploadState::Failed(_)
                    );
                if ui
                    .add_enabled(
                        can_send,
                        egui::Button::new("Send recording").min_size(Vec2::new(170.0, 36.0)),
                    )
                    .clicked()
                {
                    if let (Some(session_id), Some(condition)) =
                        (&state.saved_session_id, state.saved_condition)
                    {
                        let _ = self.commands.send(WorkerCommand::Upload {
                            path: path.to_path_buf(),
                            session_id: session_id.clone(),
                            condition,
                        });
                    }
                }
                if !evidence_complete {
                    ui.small("Incomplete evidence cannot be sent to the service.");
                }
            });
        }
    }

    impl eframe::App for CollectorApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            apply_dark_visuals(ctx);
            self.shared.heartbeat();
            let state = self
                .shared
                .ui
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            self.upload_preview(ctx, &state);
            self.sync_overlay(&state);
            self.sync_audio(&state);
            self.sync_window_level(ctx, &state);

            if ctx.input(|input| input.key_pressed(egui::Key::Space)) {
                match state.capture_status {
                    CaptureStatus::Ready { .. } => {
                        let _ = self.commands.send(WorkerCommand::Begin);
                    }
                    CaptureStatus::Running {
                        awaiting_confirmation: true,
                        ..
                    } => {
                        let _ = self.commands.send(WorkerCommand::Continue);
                    }
                    _ => {}
                }
            }

            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_header(ui, &state);
                    ui.add_space(9.0);
                    if matches!(state.capture_status, CaptureStatus::Idle) {
                        self.draw_preview(ui, &state);
                        ui.add_space(10.0);
                    }
                    ui.group(|ui| {
                        ui.set_width(ui.available_width());
                        self.draw_capture(ui, &state);
                    });
                    ui.add_space(8.0);
                    ui.small(
                        "No automatic upload. No SRanipal model output is used as training truth. Native Tobii values are stored only as advisory diagnostics.",
                    );
                });
            });

            ctx.request_repaint_after(Duration::from_millis(
                if capture_is_active(&state.capture_status) {
                    16
                } else {
                    50
                },
            ));
        }
    }

    impl Drop for CollectorApp {
        fn drop(&mut self) {
            self.overlay.hide();
            let _ = self.commands.send(WorkerCommand::Stop);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn capture_is_active(status: &CaptureStatus) -> bool {
        matches!(
            status,
            CaptureStatus::Ready { .. } | CaptureStatus::Running { .. }
        )
    }

    fn action_title(action: Option<CaptureAction>, target: Option<GazeTarget>) -> String {
        if let Some(target) = target {
            return format!("LOOK {}", target.as_str().replace('_', " ").to_uppercase());
        }
        match action {
            Some(CaptureAction::RelaxedOpen) => "RELAXED OPEN".into(),
            Some(CaptureAction::HalfOpen) => "HALF OPEN".into(),
            Some(CaptureAction::GentleClosed) => "GENTLE CLOSED".into(),
            Some(CaptureAction::SlowCloseOpen) => "SLOW CLOSE / OPEN".into(),
            Some(CaptureAction::NaturalBlink) => "NATURAL BLINKS".into(),
            Some(CaptureAction::LeftWink) => "LEFT WINK".into(),
            Some(CaptureAction::RightWink) => "RIGHT WINK".into(),
            None => "PREPARE".into(),
        }
    }

    fn draw_target(
        ui: &mut egui::Ui,
        target: Option<GazeTarget>,
        pose_progress: Option<u8>,
        paused: bool,
    ) {
        let height = 235.0;
        let (rect, _) =
            ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
        ui.painter().rect_filled(rect, 10.0, Color32::from_gray(9));
        ui.painter()
            .rect_stroke(rect, 10.0, Stroke::new(1.0, Color32::from_gray(54)));
        for t in [1.0 / 3.0, 2.0 / 3.0] {
            let x = egui::lerp(rect.left()..=rect.right(), t);
            let y = egui::lerp(rect.top()..=rect.bottom(), t);
            ui.painter().line_segment(
                [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
                Stroke::new(1.0, Color32::from_gray(25)),
            );
            ui.painter().line_segment(
                [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
                Stroke::new(1.0, Color32::from_gray(25)),
            );
        }
        if let Some(target) = target {
            let [x, y] = target.screen_xy();
            let center = Pos2::new(
                rect.center().x + x * rect.width() * 0.36,
                rect.center().y + y * rect.height() * 0.34,
            );
            ui.painter()
                .circle_filled(center, 18.0, Color32::from_rgb(70, 195, 255));
            ui.painter()
                .circle_stroke(center, 27.0, Stroke::new(3.0, Color32::WHITE));
        } else if let Some(open) = pose_progress {
            let width = rect.width().min(420.0);
            let bar = Rect::from_center_size(rect.center(), Vec2::new(width, 34.0));
            ui.painter().rect_filled(bar, 7.0, Color32::from_gray(35));
            let fill = Rect::from_min_max(
                bar.min,
                Pos2::new(bar.left() + bar.width() * open as f32 / 100.0, bar.bottom()),
            );
            ui.painter()
                .rect_filled(fill, 7.0, Color32::from_rgb(70, 195, 255));
            ui.painter().text(
                bar.center(),
                Align2::CENTER_CENTER,
                format!("EYELIDS {open}%"),
                FontId::proportional(18.0),
                Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                Align2::CENTER_CENTER,
                if paused {
                    "PAUSED"
                } else {
                    "FOLLOW THE INSTRUCTION ABOVE"
                },
                FontId::proportional(18.0),
                Color32::LIGHT_GRAY,
            );
        }
    }

    fn overlay_frame(status: &CaptureStatus) -> Option<VrTargetFrame> {
        match status {
            CaptureStatus::Idle | CaptureStatus::Done { .. } => None,
            CaptureStatus::Ready { instruction } => Some(VrTargetFrame {
                target: None,
                state: VrGuideState::Ready,
                eye_pose: VrEyePose::Open,
                countdown: None,
                progress_percent: Some(0),
                headline: "XR5 EYELID DATASET".into(),
                instruction: instruction.clone(),
                footer: "Return to the collector window and press Begin (Space).".into(),
            }),
            CaptureStatus::Running {
                progress,
                target,
                action,
                recording,
                settling,
                paused,
                awaiting_confirmation,
                phase_remaining_s,
                pose_progress,
                instruction,
                ..
            } => {
                let state = if *paused {
                    VrGuideState::Paused
                } else if *awaiting_confirmation {
                    VrGuideState::Waiting
                } else if *recording && !*settling {
                    VrGuideState::Recording
                } else {
                    VrGuideState::Prepare
                };
                let eye_pose = match action {
                    Some(CaptureAction::RelaxedOpen) => VrEyePose::Open,
                    Some(CaptureAction::HalfOpen) => VrEyePose::HalfOpen,
                    Some(CaptureAction::GentleClosed) => VrEyePose::Closed,
                    Some(CaptureAction::SlowCloseOpen) => {
                        VrEyePose::Slow(pose_progress.unwrap_or(100))
                    }
                    Some(CaptureAction::NaturalBlink) => VrEyePose::Blink,
                    Some(CaptureAction::LeftWink) => VrEyePose::LeftWink,
                    Some(CaptureAction::RightWink) => VrEyePose::RightWink,
                    None => VrEyePose::None,
                };
                Some(VrTargetFrame {
                    target: *target,
                    state,
                    eye_pose,
                    countdown: (!*awaiting_confirmation)
                        .then_some(phase_remaining_s.ceil().clamp(0.0, u8::MAX as f32) as u8),
                    progress_percent: Some((progress * 100.0).round() as u8),
                    headline: action_title(*action, *target),
                    instruction: instruction.clone(),
                    footer: if *awaiting_confirmation {
                        "Read this fully, then press Continue (Space).".into()
                    } else {
                        "Keep your head still. Follow the target with your eyes only.".into()
                    },
                })
            }
        }
    }

    fn audio_state(
        status: &CaptureStatus,
        saving: bool,
        saved: &Option<PathBuf>,
    ) -> (String, Option<RecordingCue>) {
        if saving {
            return ("saving".into(), Some(RecordingCue::Complete));
        }
        if saved.is_some() {
            return ("saved".into(), Some(RecordingCue::Saved));
        }
        match status {
            CaptureStatus::Idle => ("idle".into(), None),
            CaptureStatus::Ready { .. } => ("ready".into(), Some(RecordingCue::Ready)),
            CaptureStatus::Done { .. } => ("done".into(), Some(RecordingCue::Complete)),
            CaptureStatus::Running {
                phase_index,
                recording,
                settling,
                paused,
                awaiting_confirmation,
                holdout,
                ..
            } => {
                let state = if *paused {
                    "paused"
                } else if *awaiting_confirmation || *settling || !*recording {
                    "prepare"
                } else {
                    "sampling"
                };
                let cue = if *paused {
                    RecordingCue::Paused
                } else if *awaiting_confirmation || *settling || !*recording {
                    if *holdout {
                        RecordingCue::Holdout
                    } else {
                        RecordingCue::Prepare
                    }
                } else {
                    RecordingCue::Sampling
                };
                (format!("{phase_index}:{state}"), Some(cue))
            }
        }
    }

    fn worker_main(shared: Arc<Shared>, commands: Receiver<WorkerCommand>, mock: bool) {
        loop {
            clear_source(&shared);
            let connection = if mock {
                start_mock_source(shared.clone())
            } else {
                start_real_source(shared.clone())
            };
            match connection {
                Ok(mut adapter) => {
                    let reconnect = run_connected(shared.clone(), &commands, mock);
                    if let Some(adapter) = adapter.as_deref_mut() {
                        adapter.stop();
                    }
                    if !reconnect {
                        return;
                    }
                }
                Err(error) => {
                    {
                        let mut state = shared.ui.lock().unwrap();
                        state.connection = "XR5 connection failed".into();
                        state.detail = error;
                        state.connected = false;
                        state.stereo_ready = false;
                    }
                    loop {
                        match commands.recv_timeout(Duration::from_millis(500)) {
                            Ok(WorkerCommand::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                                return;
                            }
                            Ok(WorkerCommand::Reconnect) => break,
                            Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
                        }
                    }
                }
            }
        }
    }

    fn start_real_source(shared: Arc<Shared>) -> Result<Option<Box<dyn HmdAdapter>>, String> {
        let (mut config, warning) = Config::load(&sranibro_rs::config::config_path());
        config.hmd.device = DEVICE_KEY.into();
        let mapping = config.mapping_for(DEVICE_KEY);
        let runtime = config.tobii_runtime_path();
        if runtime.as_ref().is_none_or(|path| !path.is_file()) {
            return Err(
                "This source build does not contain the authorized runtime component. Use an official SRanibro XR5 Dataset Collector build."
                    .into(),
            );
        }
        let mut adapter = device::make_adapter(&config).map_err(|error| error.to_string())?;
        if adapter.name() != "pimax-xr5" {
            return Err(format!(
                "The collector selected an unexpected adapter: {}",
                adapter.name()
            ));
        }
        if adapter.needs_eyechip_handoff() {
            sranibro_rs::platform::ensure_capture_ready();
        }

        let source_frames = shared.clone();
        let frame_mapping = mapping;
        let on_frame = Box::new(move |eye: Eye, width: u32, height: u32, pixels: &[u8]| {
            publish_frame(
                &source_frames.source,
                frame_mapping,
                eye,
                width,
                height,
                pixels,
            );
        });
        let source_gaze = shared.clone();
        let on_gaze = Box::new(move |sample: GazeSample| {
            let now = Instant::now();
            let mut source = source_gaze
                .source
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            source.native.merge(sample, mapping.swap_eyes, now);
            source.gaze_count = source.gaze_count.wrapping_add(1);
        });

        adapter
            .start(on_frame, on_gaze)
            .map_err(|error| error.to_string())?;
        let status = adapter.status_arc();
        {
            let mut state = shared.ui.lock().unwrap();
            state.connection = "Dream Air / XR5 adapter connected".into();
            state.detail = warning
                .map(|warning| format!("Configuration note: {warning}"))
                .unwrap_or_else(|| {
                    status
                        .lock()
                        .map(|value| value.clone())
                        .unwrap_or_else(|_| "Waiting for stereo camera frames...".into())
                });
            state.connected = true;
        }
        Ok(Some(adapter))
    }

    fn start_mock_source(shared: Arc<Shared>) -> Result<Option<Box<dyn HmdAdapter>>, String> {
        let mut state = shared.ui.lock().unwrap();
        state.connection = "Simulated Dream Air / XR5 connected".into();
        state.detail = "Synthetic frames are for UI testing only and must never be imported as biometric data.".into();
        state.connected = true;
        Ok(None)
    }

    fn run_connected(shared: Arc<Shared>, commands: &Receiver<WorkerCommand>, mock: bool) -> bool {
        let mut capture = GazeResidualCapture::new();
        let mut condition = FitCondition::Normal;
        let mut save_attempted = false;
        let mut rate_at = Instant::now();
        let mut rate_counts = [0u64; 3];
        let mut mock_at = Instant::now();
        let mut ui_pause_owned = false;

        loop {
            while let Ok(command) = commands.try_recv() {
                match command {
                    WorkerCommand::Prepare(next_condition) => {
                        let ready = shared
                            .source
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .stereo_ready(Instant::now());
                        if ready && !capture.is_running() {
                            condition = next_condition;
                            let generation = shared
                                .source
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .generations();
                            capture
                                .start_protocol(CaptureProtocol::PythonEyelidDataset, generation);
                            save_attempted = false;
                            let mut state = shared.ui.lock().unwrap();
                            state.saved_path = None;
                            state.saved_session_id = None;
                            state.saved_condition = None;
                            state.save_error = None;
                            state.capture_error = None;
                            #[cfg(feature = "xr5-model-service-client")]
                            {
                                state.upload_state = if shared.service.is_some() {
                                    UploadState::Ready
                                } else {
                                    UploadState::Unavailable
                                };
                            }
                        }
                    }
                    WorkerCommand::Begin => {
                        capture.begin();
                    }
                    WorkerCommand::Continue => {
                        capture.continue_step();
                    }
                    WorkerCommand::TogglePause => {
                        if capture.is_paused() {
                            capture.resume();
                        } else {
                            capture.pause();
                        }
                    }
                    WorkerCommand::Abort => {
                        capture.abort();
                        save_attempted = false;
                        let mut state = shared.ui.lock().unwrap();
                        state.saved_path = None;
                        state.saved_session_id = None;
                        state.saved_condition = None;
                        state.save_error = None;
                        state.saving = false;
                        #[cfg(feature = "xr5-model-service-client")]
                        {
                            state.upload_state = if shared.service.is_some() {
                                UploadState::Ready
                            } else {
                                UploadState::Unavailable
                            };
                        }
                    }
                    WorkerCommand::RetrySave => {
                        if capture.is_done() {
                            save_attempted = false;
                        }
                    }
                    #[cfg(feature = "xr5-model-service-client")]
                    WorkerCommand::Upload {
                        path,
                        session_id,
                        condition,
                    } => {
                        start_upload(shared.clone(), path, session_id, condition);
                    }
                    WorkerCommand::Reconnect => {
                        if !capture.is_running() {
                            return true;
                        }
                    }
                    WorkerCommand::Stop => return false,
                }
            }

            let now = Instant::now();
            if mock && now.duration_since(mock_at) >= Duration::from_millis(8) {
                mock_at = now;
                publish_mock_pair(&shared, &capture.status(), now);
            }

            if capture.is_running() && !capture.is_ready() {
                let heartbeat_ok = shared.heartbeat_fresh(now);
                if !heartbeat_ok && !capture.is_paused() {
                    capture.pause();
                    ui_pause_owned = true;
                } else if heartbeat_ok && ui_pause_owned {
                    capture.resume();
                    ui_pause_owned = false;
                }
            }

            capture.tick_at(now);
            if capture.is_running() && !capture.is_ready() && !capture.is_paused() {
                ingest_latest(&shared, &mut capture, now);
                capture.tick_at(now);
            }

            if capture.is_done() && !save_attempted {
                save_attempted = true;
                {
                    let mut state = shared.ui.lock().unwrap();
                    state.capture_status = capture.status();
                    state.saving = true;
                    state.save_error = None;
                }
                let result = save_capture(&shared, &capture, condition, mock);
                let mut state = shared.ui.lock().unwrap();
                state.saving = false;
                match result {
                    Ok(saved) => {
                        state.saved_path = Some(saved.path);
                        state.saved_session_id = Some(saved.session_id);
                        state.saved_condition = Some(condition);
                        state.save_error = None;
                        capture.release_exported_frames();
                    }
                    Err(error) => {
                        state.saved_path = None;
                        state.save_error = Some(error);
                    }
                }
            }

            let (frame_counts, gaze_count, ready, frames) = {
                let source = shared
                    .source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (
                    source.frame_count,
                    source.gaze_count,
                    source.stereo_ready(now),
                    source.frames.clone(),
                )
            };
            if rate_at.elapsed() >= Duration::from_secs(1) {
                let seconds = rate_at.elapsed().as_secs_f32();
                let next = [frame_counts[0], frame_counts[1], gaze_count];
                let rates = std::array::from_fn::<_, 3, _>(|index| {
                    next[index].wrapping_sub(rate_counts[index]) as f32 / seconds
                });
                rate_counts = next;
                rate_at = now;
                let mut state = shared.ui.lock().unwrap();
                state.frame_hz = [rates[0], rates[1]];
                state.gaze_hz = rates[2];
            }
            {
                let mut state = shared.ui.lock().unwrap();
                state.stereo_ready = ready;
                state.frames = frames;
                state.capture_status = capture.status();
                state.capture_error = capture.last_error.clone();
                if state.connected && !ready {
                    state.detail = "Waiting for fresh LEFT and RIGHT eye-camera frames...".into();
                } else if state.connected && ready {
                    state.detail =
                        "Raw stereo cameras are live. No model or tracking output is running."
                            .into();
                }
            }

            thread::sleep(Duration::from_millis(2));
        }
    }

    fn ingest_latest(shared: &Shared, capture: &mut GazeResidualCapture, now: Instant) {
        let generation = shared
            .source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .generations();
        if !capture.wants_frame_at(generation, now) {
            return;
        }
        let (frames, native) = {
            let source = shared
                .source
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !source.stereo_ready(now) {
                return;
            }
            (source.frames.clone(), source.native.fresh_sample(now))
        };
        let native_open = [native.left, native.right].map(|eye| {
            eye.openness_reported.then_some(if eye.openness_valid {
                eye.openness
            } else {
                0.0
            })
        });
        let native_gaze = [native.left, native.right].map(|eye| eye.gaze_valid.then_some(eye.gaze));
        let native_pupil = [native.left, native.right]
            .map(|eye| (eye.pupil_pos_reported && eye.pupil_pos_valid).then_some(eye.pupil_pos));
        let left = frames[0]
            .as_ref()
            .map(|frame| (frame.width, frame.height, frame.pixels.as_ref()));
        let right = frames[1]
            .as_ref()
            .map(|frame| (frame.width, frame.height, frame.pixels.as_ref()));
        capture.on_frame_at(
            now,
            generation,
            (native.timestamp_us != 0).then_some(native.timestamp_us),
            left,
            right,
            [[1.0, 0.0]; 2],
            native_open,
            native_gaze,
            native_pupil,
        );
    }

    #[cfg(feature = "xr5-model-service-client")]
    fn start_upload(
        shared: Arc<Shared>,
        path: PathBuf,
        session_id: String,
        condition: FitCondition,
    ) {
        let Some(service) = shared.service.clone() else {
            let mut state = shared
                .ui
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.upload_state =
                UploadState::Failed("online contribution is not configured in this build".into());
            return;
        };
        let total = match path.metadata() {
            Ok(metadata) => metadata.len(),
            Err(error) => {
                let mut state = shared
                    .ui
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                state.upload_state =
                    UploadState::Failed(format!("cannot inspect the saved recording: {error}"));
                return;
            }
        };
        {
            let mut state = shared
                .ui
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !matches!(
                state.upload_state,
                UploadState::Ready | UploadState::Failed(_)
            ) || state.saved_path.as_deref() != Some(path.as_path())
            {
                return;
            }
            state.upload_state = UploadState::Uploading { sent: 0, total };
        }
        let progress_shared = shared.clone();
        let error_shared = shared.clone();
        let expected_path = path.clone();
        if let Err(error) = thread::Builder::new()
            .name("xr5-dataset-upload".into())
            .spawn(move || {
                let result = service.upload(&path, &session_id, condition.slug(), |sent, total| {
                    let mut state = progress_shared
                        .ui
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if state.saved_path.as_deref() == Some(expected_path.as_path()) {
                        state.upload_state = UploadState::Uploading { sent, total };
                    }
                });
                let mut state = progress_shared
                    .ui
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if state.saved_path.as_deref() != Some(expected_path.as_path()) {
                    return;
                }
                state.upload_state = match result {
                    Ok(receipt) => UploadState::Uploaded {
                        contribution_id: receipt.contribution_id,
                        sample_count: receipt.sample_count,
                    },
                    Err(error) => UploadState::Failed(error),
                };
            })
        {
            let mut state = error_shared
                .ui
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.upload_state =
                UploadState::Failed(format!("could not start the upload worker: {error}"));
        }
    }

    fn save_capture(
        shared: &Shared,
        capture: &GazeResidualCapture,
        condition: FitCondition,
        mock: bool,
    ) -> Result<SavedCapture, String> {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let directory = output_directory();
        std::fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
        let session_id = format!("s{stamp}");
        let path = directory.join(format!(
            "sranibro_xr5_eyelid_dataset_{stamp}{}.zip",
            if mock {
                "_SIMULATION_DO_NOT_IMPORT"
            } else {
                ""
            }
        ));
        let partial = path.with_extension("zip.partial");
        let (mapping, unit_id) = if mock {
            (EyeMapping::default(), "simulation".to_string())
        } else {
            let (mut config, _) = Config::load(&sranibro_rs::config::config_path());
            config.hmd.device = DEVICE_KEY.into();
            (
                config.mapping_for(DEVICE_KEY),
                sranibro_rs::diagnostics::pseudonymous_unit_id(
                    device::usb::peek_serial().as_deref(),
                ),
            )
        };
        let metadata = format!(
            "schema_version=3\nsranibro_version={}\nbuild_commit={}\ndevice={}\nunit_id={}\ncapture_hz=mixed_20_67\ncapture_protocol={}\nsession_id={}\nsession_kind=PythonEyelidDataset\nsession_condition={}\nbiometric_data=true\ncollection_consent=true\nconsent_text_version=xr5_dataset_collector_v1\noffline_python_dataset=true\nproduction_state_changed=false\nno_automatic_upload=true\ncontributor_id_assigned_at_import=true\npupil_pos_space=native_unmapped_when_available\nnative_freshness_limit_ms={}\ncommanded_target_space=categorical_target_labels_y_down\ntarget_presentation_requested={}\nsteamvr_target_visible_during_session={}\nframe_stage=after_eye_mapping_before_ml_geometry\nbrightness_affine_sampling=identity_raw_capture\neye_mapping={:?}\nmock_recording={}\n",
            env!("CARGO_PKG_VERSION"),
            env!("SRANIBRO_BUILD_COMMIT"),
            DEVICE_KEY,
            unit_id,
            CaptureProtocol::PythonEyelidDataset.id(),
            session_id,
            condition.slug(),
            NATIVE_FRESHNESS.as_millis(),
            if shared.overlay_requested.load(Ordering::Relaxed) {
                "steamvr_head_locked_with_desktop_fallback"
            } else {
                "desktop_window"
            },
            shared.overlay_visible_ever.load(Ordering::Relaxed),
            mapping,
            mock,
        );
        let result = capture
            .export_recording(&partial, &metadata)
            .and_then(|()| std::fs::rename(&partial, &path))
            .map_err(|error| error.to_string());
        if result.is_err() {
            let _ = std::fs::remove_file(&partial);
        }
        result.map(|()| SavedCapture { path, session_id })
    }

    fn output_directory() -> PathBuf {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .map(|path| path.join("Desktop"))
            .filter(|path| path.is_dir())
            .unwrap_or_else(sranibro_rs::config::base_dir)
            .join("SRanibro XR5 Dataset")
    }

    fn open_parent(path: &Path) {
        if let Some(parent) = path.parent() {
            let _ = std::process::Command::new("explorer.exe")
                .arg(parent)
                .spawn();
        }
    }

    fn clear_source(shared: &Shared) {
        *shared
            .source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SourceState::default();
        let mut state = shared.ui.lock().unwrap();
        state.connected = false;
        state.stereo_ready = false;
        state.frame_hz = [0.0; 2];
        state.gaze_hz = 0.0;
        state.frames = [None, None];
    }

    fn publish_frame(
        source: &Mutex<SourceState>,
        mapping: EyeMapping,
        source_eye: Eye,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) {
        let need = width as usize * height as usize;
        if width == 0 || height == 0 || pixels.len() < need {
            return;
        }
        let eye = if mapping.swap_eyes {
            source_eye.opposite()
        } else {
            source_eye
        };
        let stored: Arc<[u8]> = if mapping.flip_image {
            Arc::from(mirror_h(&pixels[..need], width as usize, height as usize))
        } else {
            Arc::from(&pixels[..need])
        };
        let now = Instant::now();
        let mut source = source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        source.frame_count[eye.idx()] = source.frame_count[eye.idx()].wrapping_add(1);
        let generation = source.frame_count[eye.idx()];
        source.frames[eye.idx()] = Some(RawFrame {
            generation,
            width,
            height,
            pixels: stored,
            received_at: now,
        });
    }

    fn publish_mock_pair(shared: &Shared, status: &CaptureStatus, now: Instant) {
        let (openness, target) = match status {
            CaptureStatus::Running {
                action,
                target,
                pose_progress,
                ..
            } => {
                let openness = match action {
                    Some(CaptureAction::HalfOpen) => 0.5,
                    Some(CaptureAction::GentleClosed) => 0.0,
                    Some(CaptureAction::SlowCloseOpen) => {
                        pose_progress.unwrap_or(100) as f32 / 100.0
                    }
                    Some(CaptureAction::LeftWink | CaptureAction::RightWink) => 0.35,
                    _ => 1.0,
                };
                (openness, *target)
            }
            _ => (1.0, Some(GazeTarget::Center)),
        };
        let mut source = shared
            .source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for eye in 0..2 {
            source.frame_count[eye] = source.frame_count[eye].wrapping_add(1);
            let pixels = mock_eye(openness, target, eye);
            source.frames[eye] = Some(RawFrame {
                generation: source.frame_count[eye],
                width: 200,
                height: 200,
                pixels: Arc::from(pixels),
                received_at: now,
            });
        }
        source.gaze_count = source.gaze_count.wrapping_add(1);
        source.native.sample.timestamp_us = source.gaze_count * 8_333;
        source.native.timestamp_at = Some(now);
    }

    fn mock_eye(openness: f32, target: Option<GazeTarget>, eye: usize) -> Vec<u8> {
        const SIDE: usize = 200;
        let [tx, ty] = target.unwrap_or(GazeTarget::Center).screen_xy();
        let cx = 100.0 + tx * 31.0 + if eye == 0 { -2.0 } else { 2.0 };
        let cy = 100.0 + ty * 19.0;
        let lid = (8.0 + openness.clamp(0.0, 1.0) * 72.0).max(6.0);
        let mut pixels = vec![34u8; SIDE * SIDE];
        for y in 0..SIDE {
            for x in 0..SIDE {
                let dx = (x as f32 - cx) / 74.0;
                let dy = (y as f32 - cy) / lid;
                let index = y * SIDE + x;
                if dx * dx + dy * dy <= 1.0 {
                    let pupil = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                    pixels[index] = if pupil < 22.0 { 22 } else { 150 };
                } else {
                    pixels[index] = (50.0 + y as f32 * 0.42).clamp(0.0, 255.0) as u8;
                }
            }
        }
        pixels
    }

    fn mirror_h(pixels: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut mirrored = vec![0u8; width * height];
        for y in 0..height {
            let row = y * width;
            for x in 0..width {
                mirrored[row + width - 1 - x] = pixels[row + x];
            }
        }
        mirrored
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_eye(
        destination: &mut EyeSample,
        source: EyeSample,
        index: usize,
        now: Instant,
        gaze_valid_at: &mut [Option<Instant>; 2],
        pupil_pos_at: &mut [Option<Instant>; 2],
        openness_at: &mut [Option<Instant>; 2],
    ) {
        if source.gaze_reported {
            if source.gaze_valid {
                destination.gaze = source.gaze;
                destination.gaze_valid = true;
                gaze_valid_at[index] = Some(now);
            }
            destination.gaze_reported = true;
        }
        if source.origin_valid {
            destination.origin_mm = source.origin_mm;
            destination.origin_valid = true;
        }
        if source.pupil_valid {
            destination.pupil_mm = source.pupil_mm;
            destination.pupil_valid = true;
        }
        if source.pupil_pos_reported {
            if source.pupil_pos_valid {
                destination.pupil_pos = source.pupil_pos;
            }
            destination.pupil_pos_valid = source.pupil_pos_valid;
            destination.pupil_pos_reported = true;
            pupil_pos_at[index] = Some(now);
        }
        if source.openness_reported {
            destination.openness = source.openness;
            destination.openness_valid = source.openness_valid;
            destination.openness_reported = true;
            openness_at[index] = Some(now);
        }
    }

    fn is_fresh(timestamp: Option<Instant>, now: Instant) -> bool {
        timestamp.is_some_and(|timestamp| now.duration_since(timestamp) <= NATIVE_FRESHNESS)
    }

    fn apply_dark_visuals(ctx: &egui::Context) {
        use sranibro_rs::theme::{ACCENT, BG, BORDER, INNER, SURFACE, TEXT1};

        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(TEXT1);
        visuals.panel_fill = BG;
        visuals.window_fill = BG;
        visuals.faint_bg_color = SURFACE;
        visuals.extreme_bg_color = INNER;
        visuals.selection.bg_fill = ACCENT.gamma_multiply(0.35);
        visuals.selection.stroke = Stroke::new(1.0, ACCENT);
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
        visuals.widgets.active.bg_stroke = Stroke::new(1.0, ACCENT);
        ctx.set_visuals(visuals);
    }

    fn app_icon() -> egui::IconData {
        const RGBA: &[u8] = include_bytes!("../assets/sranibro_rgba.bin");
        egui::IconData {
            rgba: RGBA.to_vec(),
            width: 256,
            height: 256,
        }
    }

    pub fn run() -> eframe::Result<()> {
        let mock = std::env::args().any(|argument| argument == "--mock");
        let options = eframe::NativeOptions {
            renderer: eframe::Renderer::Wgpu,
            viewport: egui::ViewportBuilder::default()
                .with_title("SRanibro XR5 Dataset Collector")
                .with_inner_size([1040.0, 860.0])
                .with_min_inner_size([820.0, 680.0])
                .with_icon(app_icon()),
            ..Default::default()
        };
        eframe::run_native(
            "SRanibro XR5 Dataset Collector",
            options,
            Box::new(move |cc| Ok(Box::new(CollectorApp::new(&cc.egui_ctx, mock)))),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mirror_is_dimension_aware() {
            assert_eq!(mirror_h(&[1, 2, 3, 4, 5, 6], 3, 2), [3, 2, 1, 6, 5, 4]);
        }

        #[test]
        fn condition_labels_are_stable_and_non_identifying() {
            let slugs: Vec<_> = FitCondition::ALL
                .into_iter()
                .map(FitCondition::slug)
                .collect();
            assert_eq!(slugs, ["normal", "reseated", "alternate_optics"]);
        }

        #[test]
        fn stale_native_fields_are_removed_instead_of_repeated() {
            let now = Instant::now();
            let mut native = NativeState::default();
            let mut sample = GazeSample::default();
            sample.left.gaze_reported = true;
            sample.left.gaze_valid = true;
            sample.left.gaze = [0.2, 0.1, 0.9];
            native.merge(sample, false, now);
            assert!(native.fresh_sample(now).left.gaze_valid);
            assert!(
                !native
                    .fresh_sample(now + NATIVE_FRESHNESS + Duration::from_millis(1))
                    .left
                    .gaze_valid
            );
        }

        #[test]
        fn complete_protocol_is_not_a_random_frame_split() {
            assert_eq!(
                CaptureProtocol::PythonEyelidDataset.id(),
                "xr5_python_eyelid_dataset_v1"
            );
            assert!(CaptureProtocol::PythonEyelidDataset.total_seconds() > 150.0);
        }
    }
}

#[cfg(windows)]
fn main() -> eframe::Result<()> {
    app::run()
}
