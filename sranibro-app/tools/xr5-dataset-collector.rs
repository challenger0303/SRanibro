//! Standalone Dream Air / XR5 eyelid-model dataset collector.
//!
//! This binary deliberately does not construct SRanibro's model, post-processor,
//! VRCFT server, OSC output, fitters, or saved calibration. It opens only the XR5
//! camera/native-data adapter, records eye movement and optional world fixation,
//! and writes a local ZIP.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("SRanibro XR5 Dataset Collector is available on Windows only.");
}

#[cfg(all(windows, feature = "xr5-model-service-client"))]
#[path = "xr5-model-service-client.rs"]
mod model_service_client;

#[cfg(windows)]
#[path = "xr5-native-evidence.rs"]
mod native_evidence;

#[cfg(windows)]
#[path = "xr5-tobii-reference.rs"]
mod tobii_reference;

#[cfg(windows)]
#[path = "xr5-free-capture.rs"]
mod free_capture;

#[cfg(windows)]
mod app {
    use crate::free_capture::{FreeCapture, RecordingPose};
    #[cfg(feature = "xr5-model-service-client")]
    use crate::model_service_client;
    use crate::native_evidence::{self, EvidenceRow};
    use crate::tobii_reference;

    use eframe::egui::{self, Align2, Color32, FontId, RichText, Sense, Stroke, Vec2};
    use sranibro_rs::config::{Config, EyeMapping};
    use sranibro_rs::core::types::{Eye, EyeSample, GazeSample};
    use sranibro_rs::device::{self, HmdAdapter};
    use sranibro_rs::gaze_residual_calib::{CaptureAction, Status as CaptureStatus};
    use sranibro_rs::geometry_calib::GazeTarget;
    use sranibro_rs::recorder_fixation::Feedback;
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
    const FRAME_FRESHNESS: Duration = Duration::from_millis(150);
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
        openness_timestamp_us: [Option<u64>; 2],
        timestamp_at: Option<Instant>,
    }

    impl Default for NativeState {
        fn default() -> Self {
            Self {
                sample: GazeSample::default(),
                gaze_valid_at: [None; 2],
                pupil_pos_at: [None; 2],
                openness_at: [None; 2],
                openness_timestamp_us: [None; 2],
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
            for (i, eye) in [source.left, source.right].iter().enumerate() {
                if eye.openness_reported {
                    self.openness_timestamp_us[i] =
                        (source.timestamp_us != 0).then_some(source.timestamp_us);
                }
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
        tobii: tobii_reference::ReferenceState,
        mapping_swap_eyes: bool,
        frames: [Option<RawFrame>; 2],
        native: NativeState,
        frame_count: [u64; 2],
        gaze_count: u64,
        wearable_packet: Option<(Arc<[u8]>, Instant)>,
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
                frame.as_ref().is_some_and(|frame| {
                    frame.width == 200
                        && frame.height == 200
                        && now.saturating_duration_since(frame.received_at) <= FRAME_FRESHNESS
                })
            }) && match (&self.frames[0], &self.frames[1]) {
                (Some(l), Some(r)) => {
                    l.received_at
                        .max(r.received_at)
                        .duration_since(l.received_at.min(r.received_at))
                        <= Duration::from_millis(25)
                }
                _ => false,
            }
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
        tobii_status: [&'static str; 2],
        frames: [Option<RawFrame>; 2],
        capture_status: CaptureStatus,
        world_session: Option<u64>,
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
                    "Ready to connect to Dream Air / XR5".into()
                },
                detail: String::new(),
                connected: false,
                stereo_ready: false,
                mock,
                frame_hz: [0.0; 2],
                gaze_hz: 0.0,
                tobii_status: ["not received"; 2],
                frames: [None, None],
                capture_status: CaptureStatus::Idle,
                world_session: None,
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
        Prepare(FitCondition, u32, RecordingPose, bool),
        Finish,
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
        /// All item recordings from one app run stay together when selecting
        /// independent evaluation sessions; no random adjacent-frame splitting.
        collection_group: String,
        ui: Mutex<LiveState>,
        source: Mutex<SourceState>,
        device_status: Mutex<Option<Arc<Mutex<String>>>>,
        overlay_requested: AtomicBool,
        overlay_visible_ever: AtomicBool,
        world_feedback: Feedback,
        #[cfg(feature = "xr5-model-service-client")]
        service: Option<model_service_client::ServiceClient>,
    }

    impl Shared {
        fn new(mock: bool, world_feedback: Feedback) -> Self {
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
                collection_group: format!(
                    "g{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos()
                ),
                ui: Mutex::new(ui),
                source: Mutex::new(SourceState::default()),
                device_status: Mutex::new(None),
                overlay_requested: AtomicBool::new(true),
                overlay_visible_ever: AtomicBool::new(false),
                world_feedback,
                #[cfg(feature = "xr5-model-service-client")]
                service,
            }
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
        recording_seconds: u32,
        recording_pose: RecordingPose,
        steamvr_guide: bool,
        world_fixation: bool,
        overlay: VrResearchOverlay,
        audio: RecordingAudio,
        last_audio_key: String,
        last_error_key: String,
        always_on_top: bool,
        ui_smoke: Option<(Instant, PathBuf, u32)>,
        ui_smoke_requested: bool,
        #[cfg(feature = "xr5-model-service-client")]
        upload_consent: bool,
    }

    impl CollectorApp {
        fn new(ctx: &egui::Context, mock: bool) -> Self {
            sranibro_rs::theme::apply(ctx);
            let overlay = VrResearchOverlay::new_recorder();
            let shared = Arc::new(Shared::new(mock, overlay.world_feedback()));
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
                recording_seconds: 30,
                recording_pose: RecordingPose::Relaxed,
                steamvr_guide: false,
                world_fixation: true,
                overlay,
                audio: RecordingAudio::new(),
                last_audio_key: String::new(),
                last_error_key: String::new(),
                always_on_top: false,
                ui_smoke: {
                    let args: Vec<_> = std::env::args().collect();
                    args.iter()
                        .position(|a| a == "--ui-smoke-test")
                        .and_then(|i| args.get(i + 1))
                        .map(|p| (Instant::now(), PathBuf::from(p), 0))
                },
                ui_smoke_requested: false,
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
            if let Some(session) = state
                .world_session
                .filter(|_| capture_is_active(&state.capture_status))
            {
                self.overlay.present_world(session);
                self.shared.overlay_requested.store(true, Ordering::Relaxed);
                if self.overlay.is_visible() {
                    self.shared
                        .overlay_visible_ever
                        .store(true, Ordering::Relaxed);
                }
                return;
            }
            let frame = overlay_frame(&state.capture_status).map(|mut f| {
                f.headline = self.recording_pose.label().into();
                f
            });
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
                ui.heading("XR5 Recorder");
                ui.small("v0.4.1 Tobii gaze reference");
                ui.label(
                    RichText::new(if state.mock { "SIMULATION" } else { "SRanibro" })
                        .strong()
                        .color(if state.mock {
                            Color32::YELLOW
                        } else {
                            Color32::from_rgb(95, 220, 150)
                        }),
                );
            });
            ui.horizontal_wrapped(|ui| {
                ui.label("Eye recordings for model development. Saved locally.");
                if ui.button("Open recordings folder").clicked() {
                    open_recordings_folder(&output_directory());
                }
            });
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
                        "L {:.0}/s   R {:.0}/s",
                        state.frame_hz[0], state.frame_hz[1]
                    ));
                    ui.separator();
                    ui.label(format!("Tobii gaze {:.0}/s | L: {} | R: {}",state.gaze_hz,state.tobii_status[0],state.tobii_status[1]));
                });
                ui.small("Tobii gaze is saved as a reference alongside eye images. Missing gaze does not stop recording.");
                if !state.detail.is_empty() && !state.stereo_ready {
                    ui.label(RichText::new(&state.detail).color(Color32::LIGHT_GRAY));
                }
                if !state.stereo_ready
                    && !capture_is_active(&state.capture_status)
                {
                    ui.label("Close SRanibro first. Connecting takes the eye cameras from Pimax tracking and may ask for administrator permission.");
                    if ui.button("Connect / reconnect cameras").clicked() {
                        let _ = self.commands.send(WorkerCommand::Reconnect);
                    }
                }
            });
        }

        fn draw_preview(&mut self, ui: &mut egui::Ui, state: &LiveState) {
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_preview, "Camera preview");
            });
            if !self.show_preview || capture_is_active(&state.capture_status) {
                return;
            }
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                let available = ui.available_width();
                let side = ((available - 16.0) * 0.5).clamp(120.0, 156.0);
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
            ui.heading("Eye recording");
            ui.horizontal_wrapped(|ui| {
                ui.selectable_value(&mut self.world_fixation, true, "Fixed dot + head movement");
                ui.selectable_value(&mut self.world_fixation, false, "Free eye movement");
            });
            if self.world_fixation {
                ui.label("Look at one floating dot while slowly turning and tilting your head. The dot stays in the same place in the room.");
                ui.small("SteamVR required. At Start, face forward comfortably: the dot is placed about 2m ahead. It stays there through pauses.");
            } else {
                ui.label("Look around freely; there are no gaze targets.");
            }
            ui.add_space(8.0);
            for pair in RecordingPose::ALL.chunks(2) {
                ui.horizontal(|ui| {
                    let width = (ui.available_width() - ui.spacing().item_spacing.x) * 0.5;
                    for &pose in pair {
                        let selected = self.recording_pose == pose;
                        if ui
                            .add_sized(
                                [width, 52.0],
                                egui::Button::new(RichText::new(pose.label()).size(19.0))
                                    .selected(selected),
                            )
                            .clicked()
                        {
                            self.recording_pose = pose;
                        }
                    }
                });
            }
            ui.add_space(6.0);
            if self.world_fixation {
                ui.label("Keep the selected eyelid posture comfortably while watching the same dot. Blink and take breaks whenever needed.");
            } else {
                ui.label(self.recording_pose.instruction());
            }
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.label("Recording time");
                ui.add(
                    egui::DragValue::new(&mut self.recording_seconds)
                        .range(10..=120)
                        .suffix(" sec"),
                );
                for seconds in [30, 60, 120] {
                    if ui
                        .selectable_label(self.recording_seconds == seconds, format!("{seconds}s"))
                        .clicked()
                    {
                        self.recording_seconds = seconds;
                    }
                }
            });
            ui.small("Each item starts only when you press Start. 5-second countdown, then automatic ZIP save.");
            ui.add_space(10.0);

            ui.collapsing("Recording details", |ui| {
                ui.label("Headset fit");
                egui::ComboBox::from_id_salt("xr5_dataset_session_condition")
                    .selected_text(self.condition.label())
                    .show_ui(ui, |ui| {
                        for condition in FitCondition::ALL {
                            ui.selectable_value(&mut self.condition, condition, condition.label());
                        }
                    });
                ui.small("A separate recording after reseating the headset is also useful.");
            });
            ui.add_space(10.0);

            ui.group(|ui| {
                ui.set_width(ui.available_width());
                ui.small("The ZIP contains images of your eyes (biometric data). Nothing is uploaded automatically.");
                ui.checkbox(
                    &mut self.consent,
                    "I agree to record my eye images for SRanibro model development.",
                );
            });

            ui.add_space(8.0);
            if !self.world_fixation {
                ui.checkbox(&mut self.steamvr_guide, "Show recording timer in SteamVR");
            }
            if self.world_fixation {
                ui.small("Only the dot floats in VR. A countdown and completion sound guide the recording. Tracking loss pauses the timer.");
            } else if self.steamvr_guide {
                ui.small(if self.overlay.is_visible() {
                    "VR timer visible"
                } else {
                    "Optional; recording also works without SteamVR."
                });
            }
            ui.add_space(8.0);
            let can_start = state.stereo_ready && self.consent && !state.saving;
            if ui
                .add_enabled(
                    can_start,
                    egui::Button::new("Start recording").min_size(Vec2::new(190.0, 38.0)),
                )
                .clicked()
            {
                self.overlay.reset_session();
                self.shared
                    .overlay_visible_ever
                    .store(false, Ordering::Relaxed);
                let _ = self.commands.send(WorkerCommand::Prepare(
                    self.condition,
                    self.recording_seconds,
                    self.recording_pose,
                    self.world_fixation,
                ));
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
                CaptureStatus::Ready { .. } => {
                    ui.label("Preparing recording...");
                }
                CaptureStatus::Running {
                    progress,
                    remaining_s,
                    recording,
                    settling,
                    paused,
                    samples_in_phase,
                    stereo_stalled,
                    phase_remaining_s,
                    instruction,
                    ..
                } => {
                    let state_label = if *paused {
                        "PAUSED"
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
                        ui.label(RichText::new(self.recording_pose.label()).size(19.0));
                    });
                    ui.label(RichText::new(instruction).size(16.0));
                    ui.add_space(7.0);
                    ui.label(
                        RichText::new(if *settling {
                            format!("Starting in {}", phase_remaining_s.ceil() as u32)
                        } else {
                            format!(
                                "{}:{:02} remaining",
                                *remaining_s as u32 / 60,
                                *remaining_s as u32 % 60
                            )
                        })
                        .size(32.0),
                    );
                    ui.add_space(7.0);
                    ui.add(egui::ProgressBar::new(*progress).show_percentage());
                    ui.horizontal_wrapped(|ui| {
                        ui.label(format!("{samples_in_phase} stereo frames"));
                    });
                    if *stereo_stalled {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            "Eye-camera frames stopped. Check the connection or stop and save what was recorded.",
                        );
                    }
                    if let Some(error) = &state.capture_error {
                        ui.colored_label(Color32::LIGHT_RED, error);
                    }
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add_enabled(*samples_in_phase > 0, egui::Button::new("Stop and save"))
                            .clicked()
                        {
                            let _ = self.commands.send(WorkerCommand::Finish);
                        }
                        if ui
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
                    ..
                } => {
                    ui.heading(if state.saving {
                        "Saving biometric ZIP..."
                    } else if state.saved_path.is_some() {
                        "Recording saved"
                    } else {
                        "Recording complete"
                    });
                    ui.label(format!("{samples} stereo frames collected."));
                    if !*evidence_complete {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            "Camera coverage was low. The available frames are kept; please check the connection before another recording.",
                        );
                    }
                    if state.saving {
                        ui.spinner();
                        ui.label("Saving eye images. Please keep this window open.");
                    }
                    if let Some(path) = &state.saved_path {
                        ui.colored_label(
                            Color32::from_rgb(95, 220, 150),
                            format!("Saved: {}", path.display()),
                        );
                        #[cfg(feature = "xr5-model-service-client")]
                        self.draw_upload(ui, state, *evidence_complete, path);
                        #[cfg(not(feature = "xr5-model-service-client"))]
                        ui.label("Send the ZIP to the model maintainer when you are ready.");
                        ui.horizontal(|ui| {
                            if ui.button("Open folder").clicked() {
                                if let Some(parent) = path.parent() {
                                    open_recordings_folder(parent);
                                }
                            }
                            #[cfg(feature = "xr5-model-service-client")]
                            let upload_in_progress =
                                matches!(state.upload_state, UploadState::Uploading { .. });
                            #[cfg(not(feature = "xr5-model-service-client"))]
                            let upload_in_progress = false;
                            if ui
                                .add_enabled(
                                    !upload_in_progress,
                                    egui::Button::new("Choose another item"),
                                )
                                .clicked()
                            {
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
                        if ui.button("Discard and record again").clicked() {
                            let _ = self.commands.send(WorkerCommand::Abort);
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
            let mut state = self
                .shared
                .ui
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            // Synthetic visual test only. Never opens the real camera adapter.
            if self.ui_smoke.is_some() && std::env::args().any(|a| a == "--ui-smoke-recording") {
                let now = Instant::now();
                let mut capture = FreeCapture::new();
                capture.start(
                    30,
                    RecordingPose::HalfOpen,
                    [0; 2],
                    now - Duration::from_secs(17),
                );
                capture.tick_at(now);
                if std::env::args().any(|a| a == "--ui-smoke-world") {
                    use sranibro_rs::recorder_fixation::{PoseHistory, WorldPose};
                    let feedback = Arc::new(Mutex::new(PoseHistory::default()));
                    let start = now - Duration::from_secs(17);
                    capture.start_world(
                        30,
                        RecordingPose::Relaxed,
                        [0; 2],
                        start,
                        333,
                        feedback.clone(),
                    );
                    let m = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]];
                    for i in 0..=340 {
                        let at = start + Duration::from_millis(i * 50);
                        feedback.lock().unwrap().push(WorldPose {
                            session: 333,
                            queried_at: at,
                            target_world: [0., 0., -2.],
                            head_to_world: m,
                            eye_to_head: [m; 2],
                            angular_velocity: [0., 0.2, 0.],
                            tracking_result: 200,
                            valid: true,
                            connected: true,
                            overlay_visible: true,
                        });
                        capture.tick_at(at);
                    }
                    capture.set_camera_times([now; 2]);
                }
                let pixels = mock_eye(0.5, None, 0);
                capture.on_frame_at(
                    now,
                    [1; 2],
                    None,
                    Some((200, 200, &pixels)),
                    Some((200, 200, &pixels)),
                    [[1.0, 0.0]; 2],
                    [None; 2],
                    [None; 2],
                    [None; 2],
                );
                state.capture_status = capture.status();
                state.mock = true;
                state.connection = "Simulated cameras".into();
                state.stereo_ready = true;
                let mut reference = tobii_reference::ReferenceState::default();
                if !std::env::args().any(|a| a == "--ui-smoke-tobii-missing") {
                    reference.receive(&tobii_reference::simulation_packet(123, true), now);
                    state.gaze_hz = 120.0; // Explicit simulated display fixture.
                } else {
                    state.gaze_hz = 0.0;
                }
                state.tobii_status = reference.status(now, false);
                self.recording_pose = if std::env::args().any(|a| a == "--ui-smoke-world") {
                    RecordingPose::Relaxed
                } else {
                    RecordingPose::HalfOpen
                };
            }
            if ctx.input(|i| i.viewport().close_requested())
                && (state.saving || capture_is_active(&state.capture_status))
            {
                ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
                // Finish saving or explicitly cancel the recording first.
            }
            self.upload_preview(ctx, &state);
            self.sync_overlay(&state);
            self.sync_audio(&state);
            self.sync_window_level(ctx, &state);

            egui::CentralPanel::default().show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.draw_header(ui, &state);
                    ui.add_space(9.0);
                    let idle=matches!(state.capture_status,CaptureStatus::Idle);
                    if idle && ui.available_width()>=780.0 {
                        let width=ui.available_width();
                        ui.horizontal_top(|ui| {
                            ui.vertical(|ui| {
                                ui.set_width(272.0);
                                self.draw_preview(ui,&state);
                            });
                            ui.add_space(12.0);
                            ui.vertical(|ui| {
                                ui.set_width(width-292.0);
                                self.draw_capture(ui,&state);
                            });
                        });
                    } else {
                        if idle {self.draw_preview(ui,&state);ui.add_space(10.0);}
                        self.draw_capture(ui,&state);
                    }
                    ui.add_space(8.0);
                    ui.small(
                        "Saved on this PC only. Share the ZIP only if you agree to eye-model research use.",
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
            if let Some((started, path, frames)) = &mut self.ui_smoke {
                *frames += 1;
                let screenshot = ctx.input(|i| {
                    i.events.iter().find_map(|e| {
                        if let egui::Event::Screenshot { image, .. } = e {
                            Some(image.clone())
                        } else {
                            None
                        }
                    })
                });
                if let Some(image) = screenshot {
                    let rgba: Vec<u8> = image.pixels.iter().flat_map(|p| p.to_array()).collect();
                    if write_rgba_png(
                        &path.with_extension("png"),
                        image.size[0] as u32,
                        image.size[1] as u32,
                        &rgba,
                    )
                    .is_err()
                    {
                        std::process::exit(2);
                    }
                    let result = std::fs::write(
                        path,
                        format!(
                            "PASS\nrenderer=Glow\nui_updates={}\nhardware_connected={}\n",
                            frames, state.connected
                        ),
                    );
                    if result.is_err() || state.connected {
                        std::process::exit(2);
                    }
                    self.ui_smoke = None;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else if started.elapsed() >= Duration::from_secs(2)
                    && *frames >= 10
                    && !self.ui_smoke_requested
                {
                    self.ui_smoke_requested = true;
                    ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot);
                } else if started.elapsed() > Duration::from_secs(20) {
                    std::process::exit(2);
                }
            }
        }
    }

    fn write_rgba_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> std::io::Result<()> {
        let mut encoder = png::Encoder::new(std::fs::File::create(path)?, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_image_data(rgba)?;
        Ok(())
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

    /// Full accelerated recording/export smoke test. Uses synthetic data only,
    /// no hardware, service handoff, native runtime, window or user settings.
    pub fn self_test(directory: &Path) -> Result<(), String> {
        use std::io::Read;
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let preview = VrTargetFrame {
            target: None,
            state: VrGuideState::Recording,
            eye_pose: VrEyePose::None,
            countdown: Some(24),
            progress_percent: Some(20),
            headline: "Half open".into(),
            instruction: String::new(),
            footer: String::new(),
        };
        write_rgba_png(
            &directory.join("vr-timer.png"),
            1024,
            576,
            &sranibro_rs::vr_research_overlay::recorder_preview_rgba(&preview),
        )
        .map_err(|e| e.to_string())?;
        let mut now = Instant::now();
        let started = now;
        let mut tobii = tobii_reference::ReferenceState::default();
        let mut capture = FreeCapture::new();
        capture.start(10, RecordingPose::Blinks, [0, 0], now);
        let mut generation = 0u64;
        let mut rows = Vec::new();
        let pixels = mock_eye(0.5, None, 0);
        for _ in 0..100_000 {
            if capture.is_done() {
                break;
            }
            now += Duration::from_millis(16);
            capture.tick_at(now);
            generation += 1;
            if !std::env::args().any(|a| a == "--self-test-tobii-missing")
                && !(30..46).contains(&(generation % 120))
            {
                tobii.receive(
                    &tobii_reference::simulation_packet(
                        (generation * 16_000) as i64,
                        generation % 120 >= 15,
                    ),
                    now - Duration::from_millis(2),
                );
            }
            let mut sample = GazeSample::default();
            sample.left.openness = 0.75;
            sample.left.openness_reported = true;
            sample.left.openness_valid = generation % 2 == 0;
            if capture.on_frame_at(
                now,
                [generation; 2],
                Some(generation * 16_000),
                Some((200, 200, &pixels)),
                Some((200, 200, &pixels)),
                [[1.0, 0.0]; 2],
                [None; 2],
                [None; 2],
                [None; 2],
            ) {
                rows.push(EvidenceRow {
                    tobii: Some(tobii.row(
                        started,
                        [
                            now - Duration::from_millis(3),
                            now - Duration::from_millis(1),
                        ],
                        [generation; 2],
                        false,
                    )),
                    generation: [generation; 2],
                    camera_age_us: [0; 2],
                    sample,
                    openness_age_us: [Some(0), None],
                    openness_timestamp_us: [Some(generation * 16_000), None],
                    gaze_age_us: [None; 2],
                    packet: None,
                    packet_age_us: None,
                });
            }
        }
        if !matches!(
            capture.status(),
            CaptureStatus::Done {
                evidence_complete: true,
                ..
            }
        ) {
            return Err(format!(
                "incomplete simulated protocol: {:?}",
                capture.status()
            ));
        }
        let path = directory.join("SIMULATION_DO_NOT_IMPORT.zip");
        capture.export_recording(&path,"device=pimax_xr5\nmock_recording=true\ncapture_protocol=xr5_free_gaze_pose_recording_v1\n")
            .and_then(|()|native_evidence::append_to_archive(&path,&rows)).map_err(|e|e.to_string())?;
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let mut csv = String::new();
        zip.by_name("native_evidence.csv")
            .map_err(|e| e.to_string())?
            .read_to_string(&mut csv)
            .map_err(|e| e.to_string())?;
        if csv.lines().count() != rows.len() * 2 + 1 || !csv.contains(",0.75,3f400000,true,false,")
        {
            return Err("native evidence did not survive export".into());
        }
        std::fs::write(directory.join("self-test.txt"),format!("PASS\nstereo_frames={}\nnative_rows={}\nfree_recording=true\ninvalid_raw_value_preserved=true\nhardware_accessed=false\n",rows.len(),rows.len()*2)).map_err(|e|e.to_string())?;
        self_test_world(directory)?;
        Ok(())
    }

    fn self_test_world(directory: &Path) -> Result<(), String> {
        use sranibro_rs::recorder_fixation::{PoseHistory, WorldPose};
        let t = Instant::now();
        let mut tobii = tobii_reference::ReferenceState::default();
        let feedback = Arc::new(Mutex::new(PoseHistory::default()));
        let mut capture = FreeCapture::new();
        capture.start_world(10, RecordingPose::Relaxed, [0; 2], t, 123, feedback.clone());
        let identity = [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]];
        let mut eyes = [identity; 2];
        eyes[0][0][3] = -0.032;
        eyes[1][0][3] = 0.032;
        let pixels = mock_eye(0.8, None, 0);
        let mut rows = Vec::new();
        for i in 1..2000_u64 {
            let now = t + Duration::from_millis(i * 16);
            if !std::env::args().any(|a| a == "--self-test-tobii-missing")
                && !(30..46).contains(&(i % 120))
            {
                tobii.receive(
                    &tobii_reference::simulation_packet((i * 16_000) as i64, i % 120 >= 15),
                    now - Duration::from_millis(2),
                );
            }
            let angle = ((i as f32) * 0.012).sin() * 0.3;
            let p = WorldPose {
                session: 123,
                queried_at: now,
                target_world: [0., 1.6, -2.],
                head_to_world: [
                    [angle.cos(), 0., angle.sin(), 0.1 * angle.sin()],
                    [0., 1., 0., 1.6],
                    [-angle.sin(), 0., angle.cos(), 0.],
                ],
                eye_to_head: eyes,
                angular_velocity: [0., 0.3, 0.],
                tracking_result: 200,
                valid: !(450..480).contains(&i),
                connected: true,
                overlay_visible: true,
            };
            feedback.lock().unwrap().push(p);
            capture.tick_at(now);
            if capture.is_done() {
                break;
            }
            if !capture.set_camera_times([now; 2]) {
                continue;
            }
            if capture.on_frame_at(
                now,
                [i; 2],
                Some(i * 16000),
                Some((200, 200, &pixels)),
                Some((200, 200, &pixels)),
                [[1., 0.]; 2],
                [None; 2],
                [None; 2],
                [None; 2],
            ) {
                rows.push(EvidenceRow {
                    tobii: Some(tobii.row(t, [now; 2], [i; 2], false)),
                    generation: [i; 2],
                    camera_age_us: [0; 2],
                    sample: GazeSample::default(),
                    openness_age_us: [None; 2],
                    openness_timestamp_us: [None; 2],
                    gaze_age_us: [None; 2],
                    packet: None,
                    packet_age_us: None,
                });
            }
        }
        if !capture.is_done() || rows.len() < 600 {
            return Err("Incomplete world fixation simulation".into());
        }
        let path = directory.join("WORLD_SIMULATION_DO_NOT_IMPORT.zip");
        capture.export_recording(&path,"capture_protocol=xr5_world_fixation_v1\nmock_recording=true\nfixation_confirmed=false\n")
            .and_then(|()|native_evidence::append_to_archive(&path,&rows)).map_err(|e|e.to_string())?;
        write_rgba_png(
            &directory.join("world-dot.png"),
            128,
            128,
            &sranibro_rs::vr_research_overlay::world_target_rgba(),
        )
        .map_err(|e| e.to_string())?;
        std::fs::write(directory.join("world-self-test.txt"),format!("PASS\nstereo_frames={}\nfixed_world_target=true\nhead_rotation_and_translation=true\ntracking_loss_injected=true\nhardware_accessed=false\n",rows.len())).map_err(|e|e.to_string())?;
        Ok(())
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
            None => "FREE RECORDING".into(),
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
                        "Blink and rest whenever you need. No order to follow.".into()
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
        // Opening the app alone must not stop services or claim the user's HMD.
        if !mock {
            loop {
                match commands.recv() {
                    Ok(WorkerCommand::Reconnect) => break,
                    Ok(WorkerCommand::Stop) | Err(_) => return,
                    _ => {}
                }
            }
        }
        loop {
            clear_source(&shared);
            shared.ui.lock().unwrap().detail = "Connecting... please wait.".into();
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
        let serial = device::usb::peek_serial();
        if device::auto_device_key_from_serial(serial.as_deref()) != DEVICE_KEY {
            return Err("No Dream Air / XR5 EyeChip detected. Connect that headset and retry; this recorder does not support other HMDs.".into());
        }
        // Collector is isolated from the user's tracking/calibration settings.
        let mut config = Config::default();
        config.hmd.device = DEVICE_KEY.into();
        let mapping = config.mapping_for(DEVICE_KEY);
        shared
            .source
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .mapping_swap_eyes = mapping.swap_eyes;
        let runtime = config.tobii_runtime_path();
        if runtime.as_ref().is_none_or(|path| !path.is_file()) {
            return Err(
                "This source build does not contain the authorized runtime component. Use an official SRanibro XR5 Dataset Collector build."
                    .into(),
            );
        }
        let mut adapter = device::xr5_adapter::Xr5Adapter::new(&config);
        let packet_source = shared.clone();
        adapter.set_wearable_packet_callback(Box::new(move |bytes| {
            let mut source = packet_source
                .source
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            source.wearable_packet = (bytes.len() <= native_evidence::MAX_PACKET_BYTES)
                .then(|| (Arc::from(bytes), Instant::now()));
        }));
        let gaze_packets = shared.clone();
        adapter.set_gaze_packet_callback(Box::new(move |bytes| {
            let received = Instant::now();
            let mut source = gaze_packets
                .source
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            source.tobii.receive(bytes, received);
        }));
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
        *shared.device_status.lock().unwrap() = Some(status.clone());
        {
            let mut state = shared.ui.lock().unwrap();
            state.connection = "Dream Air / XR5 adapter connected".into();
            state.detail = status
                .lock()
                .map(|value| value.clone())
                .unwrap_or_else(|_| "Waiting for stereo camera frames...".into());
            state.connected = true;
        }
        Ok(Some(Box::new(adapter)))
    }

    fn start_mock_source(shared: Arc<Shared>) -> Result<Option<Box<dyn HmdAdapter>>, String> {
        let mut state = shared.ui.lock().unwrap();
        state.connection = "Simulated Dream Air / XR5 connected".into();
        state.detail = "Synthetic frames are for UI testing only and must never be imported as biometric data.".into();
        state.connected = true;
        Ok(None)
    }

    fn run_connected(shared: Arc<Shared>, commands: &Receiver<WorkerCommand>, mock: bool) -> bool {
        let mut capture = FreeCapture::new();
        let mut condition = FitCondition::Normal;
        let mut save_attempted = false;
        let mut rate_at = Instant::now();
        let mut rate_counts = [0u64; 3];
        let mut mock_at = Instant::now();
        let mut evidence = Vec::<EvidenceRow>::new();

        loop {
            while let Ok(command) = commands.try_recv() {
                match command {
                    WorkerCommand::Prepare(next_condition, seconds, pose, world_fixation) => {
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
                            let now = Instant::now();
                            if world_fixation {
                                let session = SystemTime::now()
                                    .duration_since(UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_nanos()
                                    as u64;
                                capture.start_world(
                                    seconds,
                                    pose,
                                    generation,
                                    now,
                                    session,
                                    shared.world_feedback.clone(),
                                );
                            } else {
                                capture.start(seconds, pose, generation, now);
                            }
                            evidence.clear();
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
                    WorkerCommand::Finish => {
                        capture.finish();
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
                        evidence.clear();
                        evidence.shrink_to_fit();
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

            // The world-dot worker polls tracking independently of desktop repaint.
            // Free recordings retain their existing minimized behavior.

            capture.tick_at(now);
            if capture.is_running() && !capture.is_ready() && !capture.is_paused() {
                ingest_latest(&shared, &mut capture, &mut evidence, now);
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
                let result = save_capture(&shared, &capture, &evidence, condition, mock);
                let mut state = shared.ui.lock().unwrap();
                state.saving = false;
                match result {
                    Ok(saved) => {
                        state.saved_path = Some(saved.path);
                        state.saved_session_id = Some(saved.session_id);
                        state.saved_condition = Some(condition);
                        state.save_error = None;
                        capture.release_exported_frames();
                        evidence.clear();
                        evidence.shrink_to_fit();
                    }
                    Err(error) => {
                        state.saved_path = None;
                        state.save_error = Some(error);
                    }
                }
            }

            let (frame_counts, gaze_count, tobii_status, ready, frames) = {
                let source = shared
                    .source
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                (
                    source.frame_count,
                    source.tobii.packet_count,
                    source.tobii.status(now, source.mapping_swap_eyes),
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
                state.tobii_status = tobii_status;
                state.frames = frames;
                state.capture_status = capture.status();
                state.world_session = capture.world_session();
                state.capture_error = capture.last_error.clone();
                if state.connected && !ready {
                    state.detail = shared
                        .device_status
                        .lock()
                        .unwrap()
                        .as_ref()
                        .and_then(|s| s.lock().ok().map(|s| s.clone()))
                        .unwrap_or_else(|| {
                            "Waiting for fresh LEFT and RIGHT eye-camera frames...".into()
                        });
                } else if state.connected && ready {
                    state.detail =
                        "Raw stereo cameras are live. No model or tracking output is running."
                            .into();
                }
            }

            thread::sleep(Duration::from_millis(2));
        }
    }

    fn ingest_latest(
        shared: &Shared,
        capture: &mut FreeCapture,
        evidence: &mut Vec<EvidenceRow>,
        now: Instant,
    ) {
        if evidence.len() >= native_evidence::MAX_ROWS {
            capture.pause();
            capture.last_error =
                Some("Recording memory limit reached. Cancel and record again.".into());
            return;
        }
        let generation = shared
            .source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .generations();
        if !capture.wants_frame_at(generation, now) {
            return;
        }
        let (frames, native, row) = {
            let source = shared
                .source
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !source.stereo_ready(now) {
                return;
            }
            let age = |t: Instant| {
                now.saturating_duration_since(t)
                    .as_micros()
                    .min(u64::MAX as u128) as u64
            };
            let row = EvidenceRow {
                tobii: Some(source.tobii.row(
                    capture.started_at(),
                    std::array::from_fn(|i| source.frames[i].as_ref().unwrap().received_at),
                    source.generations(),
                    source.mapping_swap_eyes,
                )),
                generation: source.generations(),
                camera_age_us: std::array::from_fn(|i| {
                    age(source.frames[i].as_ref().unwrap().received_at)
                }),
                sample: source.native.sample,
                openness_age_us: source.native.openness_at.map(|t| t.map(age)),
                openness_timestamp_us: source.native.openness_timestamp_us,
                gaze_age_us: source.native.gaze_valid_at.map(|t| t.map(age)),
                packet: source.wearable_packet.as_ref().map(|(p, _)| p.clone()),
                packet_age_us: source.wearable_packet.as_ref().map(|(_, t)| age(*t)),
            };
            (source.frames.clone(), source.native.fresh_sample(now), row)
        };
        let generation = row.generation;
        if !capture.set_camera_times(std::array::from_fn(|i| {
            frames[i].as_ref().unwrap().received_at
        })) {
            return;
        }
        let native_open = [native.left, native.right].map(|eye| {
            (eye.openness_reported && eye.openness_valid && eye.openness.is_finite())
                .then_some(eye.openness)
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
        if capture.on_frame_at(
            now,
            generation,
            (native.timestamp_us != 0).then_some(native.timestamp_us),
            left,
            right,
            [[1.0, 0.0]; 2],
            native_open,
            native_gaze,
            native_pupil,
        ) {
            evidence.push(row);
        }
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
        capture: &FreeCapture,
        evidence: &[EvidenceRow],
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
            "sranibro_xr5_eyelid_dataset_{stamp}_{}{}.zip",
            capture.pose().slug(),
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
            let mut config = Config::default();
            config.hmd.device = DEVICE_KEY.into();
            (
                config.mapping_for(DEVICE_KEY),
                sranibro_rs::diagnostics::pseudonymous_unit_id(
                    device::usb::peek_serial().as_deref(),
                ),
            )
        };
        let mut metadata = format!(
            "schema_version=4\nsranibro_version={}\nbuild_commit={}\ndevice={}\nunit_id={}\ncapture_hz=up_to_67\ncapture_protocol={}\nsession_id={}\nsession_kind=FreeEyeRecording\nsession_condition={}\nbiometric_data=true\ncollection_consent=true\nconsent_text_version=xr5_dataset_collector_v1\noffline_python_dataset=true\nproduction_state_changed=false\nno_automatic_upload=true\ncontributor_id_assigned_at_import=true\npupil_pos_space=native_unmapped_when_available\nnative_freshness_limit_ms={}\ncommanded_target_space=none\ntimer_presentation_requested={}\nsteamvr_timer_visible_during_session={}\nframe_stage=after_eye_mapping_before_ml_geometry\nbrightness_affine_sampling=identity_raw_capture\neye_mapping={:?}\nmock_recording={}\n",
            env!("CARGO_PKG_VERSION"),
            env!("SRANIBRO_BUILD_COMMIT"),
            DEVICE_KEY,
            unit_id,
            capture.protocol(),
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
        if capture.world_session().is_some() {
            metadata = metadata
                .replace("schema_version=4\n", "schema_version=5\n")
                .replace(
                    "session_kind=FreeEyeRecording\n",
                    "session_kind=WorldFixedDotHeadMovement\n",
                )
                .replace(
                    "commanded_target_space=none\n",
                    "commanded_target_space=openvr_standing_world_m\n",
                )
                .replace(
                    "timer_presentation_requested=steamvr_head_locked_with_desktop_fallback\n",
                    "timer_presentation_requested=steamvr_world_fixed_dot_no_desktop_substitute\n",
                );
            metadata.push_str("fixation_schema=xr5_world_fixation_v1\ntarget_initial_distance_m=2\ntarget_width_m=0.075\ntarget_anchor=initial_tracked_hmd_pose_forward\ntracking_origin=standing\npose_prediction_seconds=0\npose_pairing=latest_preceding_query_at_each_camera_receive\npose_max_age_us=50000\npose_pairing_is_exposure_sync=false\nview_origins=runtime_virtual_eye_to_head_not_measured_anatomical_centers\nnominal_ray_space=runtime_eye_view_right_up_minus_z_forward\nfixation_confirmed=false\ngeometry_calibration_available=false\nevent_log_capacity=8192\n");
        }
        metadata.push_str("collector_version=0.4.1\nnative_evidence_schema=xr5_native_evidence_v1\nnative_flags_preserved=true\nnative_packet_stream=1285\nnative_sample_pairing=latest_received_with_age_not_exposure_synchronized\ntobii_gaze_schema=xr5_tobii_gaze_reference_v1\ntobii_gaze_stream=1289\ntobii_gaze_pairing=latest_preceding_packet_at_each_camera_receive\ntobii_gaze_freshness_limit_ms=150\ntobii_gaze_clock=local_monotonic_receive_not_exposure_sync\ntobii_gaze_validity=adapter_vector_plausibility_not_native_confidence\ntobii_gaze_coordinate_frame=device_raw_unverified\ntobii_gaze_eye_mapping=decoder_column3_left_column4_right_then_recording_swap\ntobii_gaze_hardware_eye_assignment_verified=false\ntobii_gaze_is_training_label=false\n");
        metadata.push_str(&format!("collection_group={}\n", shared.collection_group));
        let result = capture
            .export_recording(&partial, &metadata)
            .and_then(|()| native_evidence::append_to_archive(&partial, evidence))
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

    fn open_recordings_folder(directory: &Path) {
        // Also usable before the first recording. Use the same destination as
        // the writer, including its fallback when Desktop is unavailable.
        let result = std::fs::create_dir_all(directory).and_then(|()| {
            std::process::Command::new("explorer.exe")
                .arg(directory)
                .spawn()
        });
        if let Err(error) = result {
            rfd::MessageDialog::new()
                .set_title("Could not open recordings folder")
                .set_description(format!("{}\n\n{error}", directory.display()))
                .set_level(rfd::MessageLevel::Error)
                .show();
        }
    }

    fn clear_source(shared: &Shared) {
        *shared.device_status.lock().unwrap() = None;
        *shared
            .source
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = SourceState::default();
        let mut state = shared.ui.lock().unwrap();
        state.connected = false;
        state.stereo_ready = false;
        state.frame_hz = [0.0; 2];
        state.gaze_hz = 0.0;
        state.tobii_status = ["not received"; 2];
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
        let mock_timestamp = (source.gaze_count * 8_333) as i64;
        if !std::env::args().any(|a| a == "--ui-smoke-tobii-missing") {
            source.tobii.receive(
                &tobii_reference::simulation_packet(mock_timestamp, true),
                now,
            );
        }
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
            destination.gaze = source.gaze;
            destination.gaze_valid = source.gaze_valid;
            gaze_valid_at[index] = Some(now);
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
            renderer: eframe::Renderer::Glow,
            viewport: egui::ViewportBuilder::default()
                .with_title("SRanibro XR5 Recorder")
                .with_inner_size([840.0, 860.0])
                .with_min_inner_size([680.0, 640.0])
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
        fn invalid_native_replaces_valid_without_losing_raw_value() {
            let now = Instant::now();
            let mut state = NativeState::default();
            let mut sample = GazeSample::default();
            sample.timestamp_us = 11;
            sample.left.openness = 0.75;
            sample.left.openness_reported = true;
            sample.left.openness_valid = true;
            sample.left.gaze_reported = true;
            sample.left.gaze_valid = true;
            state.merge(sample, false, now);
            sample.timestamp_us = 12;
            sample.left.openness_valid = false;
            sample.left.gaze_valid = false;
            state.merge(sample, false, now + Duration::from_millis(1));
            assert!(!state.sample.left.openness_valid);
            assert!(!state.sample.left.gaze_valid);
            assert_eq!(state.sample.left.openness, 0.75);
            assert_eq!(state.openness_timestamp_us[0], Some(12));
            sample.timestamp_us = 13;
            sample.left.openness_reported = false;
            state.merge(sample, false, now + Duration::from_millis(2));
            assert_eq!(state.openness_timestamp_us[0], Some(12));
        }

        #[test]
        fn free_recording_has_its_own_protocol() {
            assert_eq!(
                crate::free_capture::PROTOCOL,
                "xr5_free_gaze_pose_recording_v1"
            );
        }
    }
}

#[cfg(windows)]
fn main() -> eframe::Result<()> {
    // The platform handoff relaunches this exact executable. Without these
    // subcommands an elevated second GUI would be opened instead of the helper.
    match std::env::args().nth(1).as_deref() {
        Some("custom") => {
            sranibro_rs::platform::cmd_custom();
            return Ok(());
        }
        Some("restore") => {
            sranibro_rs::platform::cmd_restore();
            return Ok(());
        }
        Some("--check-runtime") => {
            // Release gate: check the finished executable, not just a test binary.
            // No camera is opened and no service handoff is performed.
            let report = std::env::args()
                .nth(2)
                .expect("--check-runtime needs a report path");
            let result = (|| -> Result<(), String> {
                let path = sranibro_rs::config::Config::default()
                    .tobii_runtime_path()
                    .filter(|path| path.is_file())
                    .ok_or_else(|| "official runtime component is missing".to_string())?;
                use std::os::windows::ffi::OsStrExt;
                use windows_sys::Win32::Foundation::FreeLibrary;
                use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
                let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
                // SAFETY: the provider supplies an absolute path, terminated above.
                // No exported function is invoked; the handle is released below.
                let library = unsafe { LoadLibraryW(wide.as_ptr()) };
                if library.is_null() {
                    return Err(format!(
                        "runtime could not load: {}",
                        std::io::Error::last_os_error()
                    ));
                }
                let export = unsafe { GetProcAddress(library, b"tobii_api_create\0".as_ptr()) };
                unsafe {
                    FreeLibrary(library);
                }
                if export.is_none() {
                    return Err("runtime API export missing".into());
                }
                Ok(())
            })();
            let message = match &result {
                Ok(()) => "PASS\nruntime_materialized=true\nruntime_loadable=true\nhardware_accessed=false\n".to_string(),
                Err(error) => format!("FAIL\n{error}\nhardware_accessed=false\n"),
            };
            if std::fs::write(report, message).is_err() || result.is_err() {
                std::process::exit(1);
            }
            return Ok(());
        }
        Some("--self-test-output") => {
            let path = std::env::args()
                .nth(2)
                .expect("--self-test-output needs a directory");
            if let Err(error) = app::self_test(std::path::Path::new(&path)) {
                eprintln!("self-test failed: {error}");
                std::process::exit(1);
            }
            return Ok(());
        }
        _ => {}
    }
    let result = app::run();
    if let Err(error) = &result {
        let message = format!("XR5 Recorder could not start: {error}");
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            let dir = std::path::PathBuf::from(local).join("SRanibro-XR5-Recorder");
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(dir.join("startup-error.txt"), &message);
        }
        rfd::MessageDialog::new()
            .set_title("XR5 Recorder")
            .set_description(&message)
            .set_level(rfd::MessageLevel::Error)
            .show();
    }
    result
}
