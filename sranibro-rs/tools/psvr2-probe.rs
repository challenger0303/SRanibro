//! Standalone PSVR2Toolkit hardware probe for SRanibro.
//!
//! This is intentionally read-only: it never sends calibration, USB-state,
//! haptic, or driver commands. The returned archive establishes the remaining
//! hardware facts (camera half order, mirroring, gaze sign, frame rate and
//! availability) before the adapter is enabled in the main application.

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("SRanibro PSVR2 Probe is available on Windows only.");
}

#[cfg(windows)]
mod app {
    use eframe::egui::{self, Color32, RichText, Sense, Stroke, Vec2};
    use sranibro_rs::device::psvr2_capi::{
        EyeImageFrame, GazeStatus, GazeVec3, Psvr2Capi, CAMERA_HEIGHT, CAMERA_WIDTH, IMAGE_HEIGHT,
        IMAGE_WIDTH, RESULT_DRIVER_INACTIVE, RESULT_NO_SLOT, RESULT_OK,
    };
    use std::fmt::Write as FmtWrite;
    use std::fs::{self, File};
    use std::io::{self, Write};
    use std::path::{Path, PathBuf};
    use std::sync::mpsc::{self, Receiver, Sender};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
    use windows_sys::Win32::System::Diagnostics::Debug::MessageBeep;
    use windows_sys::Win32::UI::WindowsAndMessaging::MB_OK;
    use zip::write::SimpleFileOptions;

    const TOOLKIT_CONTRACT_COMMIT: &str = "9e24e6ef475660481e8b46366aaa3cb24d0b4fde";
    /// Every pose gets a quiet preparation interval before any labelled evidence
    /// is accepted. This keeps reading/repositioning motion out of the dataset.
    const PREP_SECONDS: f32 = 4.0;
    const CAPTURE_SECONDS: f32 = 8.0;
    const STEP_SECONDS: f32 = PREP_SECONDS + CAPTURE_SECONDS;

    #[derive(Clone, Copy)]
    struct Phase {
        slug: &'static str,
        title: &'static str,
        instruction: &'static str,
    }

    const PHASES: [Phase; 7] = [
        Phase {
            slug: "neutral",
            title: "Look straight / relaxed eyes",
            instruction: "Keep both eyes comfortably open and look straight ahead.",
        },
        Phase {
            slug: "look_left",
            title: "Look left",
            instruction: "Move only your eyes to the left. Keep both eyelids open.",
        },
        Phase {
            slug: "look_right",
            title: "Look right",
            instruction: "Move only your eyes to the right. Keep both eyelids open.",
        },
        Phase {
            slug: "look_up_down",
            title: "Look up, then down",
            instruction: "Alternate slowly between looking up and looking down.",
        },
        Phase {
            slug: "close_both",
            title: "Close both eyes",
            instruction: "Close both eyes gently, hold briefly, then reopen and blink naturally.",
        },
        Phase {
            slug: "wink_left",
            title: "Hold a LEFT wink",
            instruction: "Close only your anatomical LEFT eye and hold it gently.",
        },
        Phase {
            slug: "wink_right",
            title: "Hold a RIGHT wink",
            instruction: "Close only your anatomical RIGHT eye and hold it gently.",
        },
    ];

    const TOTAL_SECONDS: f32 = STEP_SECONDS * PHASES.len() as f32;

    fn phase_index_at(elapsed: f32) -> usize {
        ((elapsed.max(0.0) / STEP_SECONDS) as usize).min(PHASES.len() - 1)
    }

    fn step_elapsed_at(elapsed: f32) -> f32 {
        let elapsed = elapsed.max(0.0);
        elapsed - phase_index_at(elapsed) as f32 * STEP_SECONDS
    }

    fn capture_active_at(elapsed: f32) -> bool {
        step_elapsed_at(elapsed) >= PREP_SECONDS
    }

    #[derive(Clone)]
    struct LiveState {
        ready: bool,
        mock: bool,
        connection: String,
        detail: String,
        capi_path: Option<PathBuf>,
        latest_gaze: Option<GazeStatus>,
        camera_a: Arc<Vec<u8>>,
        camera_b: Arc<Vec<u8>>,
        image_generation: u64,
        gaze_hz: f32,
        image_hz: f32,
        last_gaze_at: Option<Instant>,
        last_image_at: Option<Instant>,
        recording: Option<RecordingProgress>,
        report_path: Option<PathBuf>,
        report_error: Option<String>,
    }

    impl Default for LiveState {
        fn default() -> Self {
            Self {
                ready: false,
                mock: false,
                connection: "Starting PSVR2 probe...".into(),
                detail: String::new(),
                capi_path: None,
                latest_gaze: None,
                camera_a: Arc::new(Vec::new()),
                camera_b: Arc::new(Vec::new()),
                image_generation: 0,
                gaze_hz: 0.0,
                image_hz: 0.0,
                last_gaze_at: None,
                last_image_at: None,
                recording: None,
                report_path: None,
                report_error: None,
            }
        }
    }

    #[derive(Clone)]
    struct RecordingProgress {
        phase_index: usize,
        elapsed: f32,
        step_elapsed: f32,
        capturing: bool,
    }

    enum WorkerCommand {
        StartRecording,
        Stop,
    }

    struct ProbeApp {
        shared: Arc<Mutex<LiveState>>,
        commands: Sender<WorkerCommand>,
        worker: Option<thread::JoinHandle<()>>,
        tex_a: Option<egui::TextureHandle>,
        tex_b: Option<egui::TextureHandle>,
        texture_generation: u64,
    }

    impl ProbeApp {
        fn new(ctx: &egui::Context, mock: bool) -> Self {
            ctx.set_visuals(egui::Visuals::dark());
            let shared = Arc::new(Mutex::new(LiveState {
                mock,
                ..LiveState::default()
            }));
            let (commands, receiver) = mpsc::channel();
            let worker_state = shared.clone();
            let worker = thread::spawn(move || worker_main(worker_state, receiver, mock));
            Self {
                shared,
                commands,
                worker: Some(worker),
                tex_a: None,
                tex_b: None,
                texture_generation: 0,
            }
        }

        fn upload_images(&mut self, ctx: &egui::Context, state: &LiveState) {
            if state.image_generation == self.texture_generation
                || state.camera_a.len() != CAMERA_WIDTH * CAMERA_HEIGHT
                || state.camera_b.len() != CAMERA_WIDTH * CAMERA_HEIGHT
            {
                return;
            }
            self.texture_generation = state.image_generation;
            for (name, pixels, slot) in [
                ("psvr2_camera_a", &state.camera_a, &mut self.tex_a),
                ("psvr2_camera_b", &state.camera_b, &mut self.tex_b),
            ] {
                let image =
                    egui::ColorImage::from_gray([CAMERA_WIDTH, CAMERA_HEIGHT], pixels.as_slice());
                match slot {
                    Some(texture) => texture.set(image, egui::TextureOptions::LINEAR),
                    None => {
                        *slot = Some(ctx.load_texture(name, image, egui::TextureOptions::LINEAR))
                    }
                }
            }
        }

        fn camera_panel(
            ui: &mut egui::Ui,
            label: &str,
            texture: Option<&egui::TextureHandle>,
            size: f32,
        ) {
            ui.vertical(|ui| {
                ui.label(RichText::new(label).strong().size(16.0));
                let (rect, _) = ui.allocate_exact_size(Vec2::splat(size), Sense::hover());
                if let Some(texture) = texture {
                    egui::Image::new(egui::load::SizedTexture::new(
                        texture.id(),
                        Vec2::splat(size),
                    ))
                    .paint_at(ui, rect);
                } else {
                    ui.painter().rect_filled(rect, 8.0, Color32::from_gray(18));
                    ui.painter().text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "waiting for eye image",
                        egui::FontId::proportional(14.0),
                        Color32::GRAY,
                    );
                }
                ui.painter()
                    .rect_stroke(rect, 8.0, Stroke::new(1.0, Color32::from_gray(65)));
            });
        }

        fn gaze_panel(ui: &mut egui::Ui, status: Option<GazeStatus>) {
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.set_min_width(330.0);
                    ui.heading("Native gaze data");
                    if let Some(status) = status {
                        let left = status.wearable.left;
                        let right = status.wearable.right;
                        vector_row(ui, "Left gaze", left.gaze_valid(), left.gaze_dir_norm);
                        vector_row(ui, "Right gaze", right.gaze_valid(), right.gaze_dir_norm);
                        vector_row(
                            ui,
                            "Combined",
                            status.wearable.is_gaze_dir_combined_valid != 0,
                            status.wearable.gaze_dir_combined_norm,
                        );
                        ui.separator();
                        ui.label(format!(
                            "Pupil L: {} {:.3} mm",
                            yes_no(left.pupil_valid()),
                            left.pupil_dia_mm,
                        ));
                        ui.label(format!(
                            "Sensor L: {} ({:.3}, {:.3})",
                            yes_no(left.pupil_position_valid()),
                            left.pupil_pos_in_sensor_area.x,
                            left.pupil_pos_in_sensor_area.y
                        ));
                        ui.label(format!(
                            "Pupil R: {} {:.3} mm",
                            yes_no(right.pupil_valid()),
                            right.pupil_dia_mm,
                        ));
                        ui.label(format!(
                            "Sensor R: {} ({:.3}, {:.3})",
                            yes_no(right.pupil_position_valid()),
                            right.pupil_pos_in_sensor_area.x,
                            right.pupil_pos_in_sensor_area.y
                        ));
                        ui.label(format!(
                            "Blink L/R: {} / {}    valid {} / {}",
                            left.blink != 0,
                            right.blink != 0,
                            left.blink_valid(),
                            right.blink_valid()
                        ));
                        ui.label(format!(
                            "Gaze block: v{} / {} B / frame {}",
                            status.version, status.size, status.wearable.frame_counter
                        ));
                        ui.label(format!(
                            "Exposure L/R: {:.2} / {:.2}    DSP: {}",
                            status.exp_l, status.exp_r, status.dsp_return_code
                        ));
                    } else {
                        ui.label("No gaze status received yet.");
                    }
                });
            });
        }
    }

    impl eframe::App for ProbeApp {
        fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
            // Keep the probe visually consistent even when the platform integration
            // refreshes system theme settings after app construction.
            ctx.set_visuals(egui::Visuals::dark());
            ctx.request_repaint_after(Duration::from_millis(16));
            let state = self.shared.lock().unwrap().clone();
            self.upload_images(ctx, &state);

            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.heading("SRanibro PSVR2 Hardware Probe");
                    ui.label(
                        RichText::new(if state.mock { "MOCK" } else { "READ ONLY" })
                            .color(if state.mock {
                                Color32::YELLOW
                            } else {
                                Color32::LIGHT_GREEN
                            })
                            .strong(),
                    );
                });
                ui.label(
                    "Validates PSVR2Toolkit gaze and eye-camera data before PSVR2 support is enabled in SRanibro.",
                );
                ui.add_space(8.0);

                ui.group(|ui| {
                    ui.horizontal_wrapped(|ui| {
                        let status_color = if state.ready {
                            Color32::LIGHT_GREEN
                        } else {
                            Color32::YELLOW
                        };
                        ui.label(RichText::new(&state.connection).color(status_color).strong());
                        ui.separator();
                        ui.label(format!(
                            "gaze {:.1}/s    image {:.1}/s",
                            state.gaze_hz, state.image_hz
                        ));
                    });
                    if !state.detail.is_empty() {
                        ui.label(RichText::new(&state.detail).color(Color32::LIGHT_GRAY));
                    }
                    if let Some(path) = &state.capi_path {
                        ui.small(format!("CAPI: {}", path.display()));
                    }
                });

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let available = (ui.available_width() - 24.0).max(400.0);
                    let camera_size = ((available - 350.0) / 2.0).clamp(190.0, 300.0);
                    Self::camera_panel(ui, "Camera A (unassigned)", self.tex_a.as_ref(), camera_size);
                    Self::camera_panel(ui, "Camera B (unassigned)", self.tex_b.as_ref(), camera_size);
                    Self::gaze_panel(ui, state.latest_gaze);
                });

                ui.add_space(10.0);
                ui.group(|ui| {
                    ui.set_width(ui.available_width());
                    ui.heading(format!("{}-second labelled recording", TOTAL_SECONDS as u32));
                    ui.label(
                        "The camera halves stay A/B until the LEFT and RIGHT wink phases prove their anatomical order.",
                    );

                    if let Some(progress) = &state.recording {
                        let phase = PHASES[progress.phase_index.min(PHASES.len() - 1)];
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(format!(
                                "{}    STEP {} / {}    {}",
                                if progress.capturing {
                                    "RECORDING"
                                } else {
                                    "GET READY"
                                },
                                progress.phase_index + 1,
                                PHASES.len(),
                                phase.title
                            ))
                            .size(20.0)
                            .strong()
                            .color(if progress.capturing {
                                Color32::from_rgb(90, 225, 145)
                            } else {
                                Color32::from_rgb(70, 195, 255)
                            }),
                        );
                        ui.label(RichText::new(phase.instruction).size(16.0));
                        let remaining = if progress.capturing {
                            (STEP_SECONDS - progress.step_elapsed).max(0.0)
                        } else {
                            (PREP_SECONDS - progress.step_elapsed).max(0.0)
                        };
                        ui.label(if progress.capturing {
                            format!("Hold the requested pose — {remaining:.1} seconds remaining")
                        } else {
                            format!(
                                "Recording begins after the interval — {remaining:.1} seconds"
                            )
                        });
                        ui.add(
                            egui::ProgressBar::new((progress.elapsed / TOTAL_SECONDS).clamp(0.0, 1.0))
                                .show_percentage(),
                        );
                        ui.small(
                            "A tone marks the new instruction; a second tone marks the start of recording.",
                        );
                    } else {
                        let can_start = state.ready;
                        if ui
                            .add_enabled(
                                can_start,
                                egui::Button::new("Start recording").min_size(Vec2::new(150.0, 34.0)),
                            )
                            .clicked()
                        {
                            let _ = self.commands.send(WorkerCommand::StartRecording);
                        }
                        if !can_start {
                            ui.small(
                                "Start SteamVR, connect PSVR2, and confirm PSVR2Toolkit is active.",
                            );
                        }
                    }

                    if let Some(error) = &state.report_error {
                        ui.colored_label(Color32::LIGHT_RED, format!("Report failed: {error}"));
                    }
                    if let Some(path) = &state.report_path {
                        ui.add_space(5.0);
                        ui.label(
                            RichText::new(format!("Saved feedback ZIP: {}", path.display()))
                                .color(Color32::LIGHT_GREEN),
                        );
                        if ui.button("Open report folder").clicked() {
                            if let Some(parent) = path.parent() {
                                let _ = std::process::Command::new("explorer.exe").arg(parent).spawn();
                            }
                        }
                    }
                });

                ui.add_space(8.0);
                ui.small(
                    "No calibration, USB-state, haptic, driver, or SteamVR setting is changed. \
                     PSVR2Toolkit eye-tracking data is for non-commercial use only.",
                );
            });
        }
    }

    impl Drop for ProbeApp {
        fn drop(&mut self) {
            let _ = self.commands.send(WorkerCommand::Stop);
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }

    fn vector_row(ui: &mut egui::Ui, label: &str, valid: bool, value: GazeVec3) {
        ui.label(format!(
            "{label}: {}  ({:+.4}, {:+.4}, {:+.4})",
            yes_no(valid),
            value.x,
            value.y,
            value.z
        ));
    }

    fn yes_no(value: bool) -> &'static str {
        if value {
            "valid"
        } else {
            "invalid"
        }
    }

    struct ActiveRecording {
        started: Instant,
        csv: String,
        snapshots: Vec<CaptureSnapshot>,
        captured_phase: [bool; PHASES.len()],
        last_phase: usize,
        last_capturing: bool,
        gaze_samples: u64,
        image_frames: u64,
        valid_gaze_l: u64,
        valid_gaze_r: u64,
        valid_pupil_l: u64,
        valid_pupil_r: u64,
        capi_path: Option<PathBuf>,
        last_status: Option<GazeStatus>,
    }

    impl ActiveRecording {
        fn new(capi_path: Option<PathBuf>) -> Self {
            let mut csv = String::new();
            csv.push_str(
                "elapsed_ms,phase,device_timestamp,frame_counter,\
left_gaze_valid,left_gaze_x,left_gaze_y,left_gaze_z,\
right_gaze_valid,right_gaze_x,right_gaze_y,right_gaze_z,\
combined_valid,combined_x,combined_y,combined_z,\
left_pupil_valid,left_pupil_mm,left_pupil_pos_valid,left_pupil_x,left_pupil_y,left_blink_valid,left_blink,\
right_pupil_valid,right_pupil_mm,right_pupil_pos_valid,right_pupil_x,right_pupil_y,right_blink_valid,right_blink,\
foveated_tracking_state,convergence_mm,exp_l,exp_r,dsp_return_code\n",
            );
            Self {
                started: Instant::now(),
                csv,
                snapshots: Vec::new(),
                captured_phase: [false; PHASES.len()],
                last_phase: 0,
                last_capturing: false,
                gaze_samples: 0,
                image_frames: 0,
                valid_gaze_l: 0,
                valid_gaze_r: 0,
                valid_pupil_l: 0,
                valid_pupil_r: 0,
                capi_path,
                last_status: None,
            }
        }

        fn elapsed(&self) -> f32 {
            self.started.elapsed().as_secs_f32()
        }

        fn phase_index(&self) -> usize {
            phase_index_at(self.elapsed())
        }

        fn step_elapsed(&self) -> f32 {
            step_elapsed_at(self.elapsed())
        }

        fn capture_active(&self) -> bool {
            capture_active_at(self.elapsed())
        }

        fn push_gaze(&mut self, status: GazeStatus) {
            if !self.capture_active() {
                return;
            }
            self.gaze_samples += 1;
            self.valid_gaze_l += status.wearable.left.gaze_valid() as u64;
            self.valid_gaze_r += status.wearable.right.gaze_valid() as u64;
            self.valid_pupil_l += status.wearable.left.pupil_valid() as u64;
            self.valid_pupil_r += status.wearable.right.pupil_valid() as u64;
            self.last_status = Some(status);

            let left = status.wearable.left;
            let right = status.wearable.right;
            let combined_valid = status.wearable.is_gaze_dir_combined_valid != 0;
            let combined = status.wearable.gaze_dir_combined_norm;
            let elapsed_ms = self.started.elapsed().as_micros() as f64 / 1000.0;
            let _ = writeln!(
                self.csv,
                "{elapsed_ms:.3},{},{},{},{},{:.7},{:.7},{:.7},{},{:.7},{:.7},{:.7},{},{:.7},{:.7},{:.7},{},{:.7},{},{:.7},{:.7},{},{},{},{:.7},{},{:.7},{:.7},{},{},{},{:.7},{:.7},{:.7},{}",
                PHASES[self.phase_index()].slug,
                status.wearable.timestamp,
                status.wearable.frame_counter,
                left.gaze_valid() as u8,
                left.gaze_dir_norm.x,
                left.gaze_dir_norm.y,
                left.gaze_dir_norm.z,
                right.gaze_valid() as u8,
                right.gaze_dir_norm.x,
                right.gaze_dir_norm.y,
                right.gaze_dir_norm.z,
                combined_valid as u8,
                combined.x,
                combined.y,
                combined.z,
                left.pupil_valid() as u8,
                left.pupil_dia_mm,
                left.pupil_position_valid() as u8,
                left.pupil_pos_in_sensor_area.x,
                left.pupil_pos_in_sensor_area.y,
                left.blink_valid() as u8,
                (left.blink != 0) as u8,
                right.pupil_valid() as u8,
                right.pupil_dia_mm,
                right.pupil_position_valid() as u8,
                right.pupil_pos_in_sensor_area.x,
                right.pupil_pos_in_sensor_area.y,
                right.blink_valid() as u8,
                (right.blink != 0) as u8,
                status.foveated.tracking_state,
                status.foveated.convergence_distance_mm,
                status.exp_l,
                status.exp_r,
                status.dsp_return_code,
            );
        }

        fn push_image(&mut self, frame: &EyeImageFrame) {
            if !self.capture_active() {
                return;
            }
            self.image_frames += 1;
            let phase = self.phase_index();
            let within_capture = self.step_elapsed() - PREP_SECONDS;
            if within_capture >= CAPTURE_SECONDS * 0.55 && !self.captured_phase[phase] {
                let [camera_a, camera_b] = frame.split_camera_planes();
                self.snapshots.push(CaptureSnapshot {
                    phase,
                    header: frame.header,
                    combined: frame.pixels.clone(),
                    camera_a,
                    camera_b,
                    version: frame.version,
                    total_size: frame.total_size,
                    timestamp: frame.timestamp,
                    image_type: frame.image_type,
                });
                self.captured_phase[phase] = true;
            }
        }
    }

    struct CaptureSnapshot {
        phase: usize,
        header: [u8; 0x100],
        combined: Vec<u8>,
        camera_a: Vec<u8>,
        camera_b: Vec<u8>,
        version: u16,
        total_size: u32,
        timestamp: u32,
        image_type: u16,
    }

    enum Backend {
        Real(Psvr2Capi),
        Mock(MockBackend),
    }

    impl Backend {
        fn capi_path(&self) -> Option<PathBuf> {
            match self {
                Self::Real(capi) => Some(capi.module_path().to_path_buf()),
                Self::Mock(_) => None,
            }
        }

        fn driver_active(&self) -> bool {
            match self {
                Self::Real(capi) => capi.driver_active().unwrap_or(true),
                Self::Mock(_) => true,
            }
        }

        fn poll(&mut self) -> io::Result<(Option<GazeStatus>, Option<EyeImageFrame>)> {
            match self {
                Self::Real(capi) => Ok((capi.next_gaze(0)?, capi.next_image(0)?)),
                Self::Mock(mock) => Ok(mock.poll()),
            }
        }
    }

    struct MockBackend {
        next: Instant,
        counter: u32,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                next: Instant::now(),
                counter: 0,
            }
        }

        fn poll(&mut self) -> (Option<GazeStatus>, Option<EyeImageFrame>) {
            let now = Instant::now();
            if now < self.next {
                return (None, None);
            }
            self.next += Duration::from_micros(8_333);
            self.counter = self.counter.wrapping_add(1);
            let t = self.counter as f32 / 120.0;

            let mut gaze = GazeStatus {
                magic: *b"GS",
                version: 1,
                size: 0x148,
                exp_l: 42.0 + t.sin(),
                exp_r: 43.0 - t.sin(),
                ..GazeStatus::default()
            };
            gaze.wearable.timestamp = (t * 1_000_000.0) as i64;
            gaze.wearable.frame_counter = self.counter;
            for (eye, sign) in [
                (&mut gaze.wearable.left, 1.0f32),
                (&mut gaze.wearable.right, -1.0f32),
            ] {
                eye.is_gaze_dir_valid = 1;
                eye.gaze_dir_norm = GazeVec3 {
                    x: sign * 0.12 * (t * 0.8).sin(),
                    y: 0.08 * (t * 0.5).cos(),
                    z: 0.99,
                };
                eye.is_pupil_dia_valid = 1;
                eye.pupil_dia_mm = 3.5 + 0.2 * t.sin();
                eye.is_pupil_pos_in_sensor_area_valid = 1;
                eye.pupil_pos_in_sensor_area.x = 0.5 + sign * 0.1 * t.sin();
                eye.pupil_pos_in_sensor_area.y = 0.5 + 0.08 * t.cos();
                eye.is_blink_valid = 1;
                eye.blink = ((self.counter / 180) % 2 == 1 && self.counter % 180 < 8) as u32;
            }
            gaze.wearable.is_gaze_dir_combined_valid = 1;
            gaze.wearable.gaze_dir_combined_norm = GazeVec3 {
                x: 0.12 * (t * 0.8).sin(),
                y: 0.08 * (t * 0.5).cos(),
                z: 0.99,
            };

            let mut header = [0u8; 0x100];
            header[0..2].copy_from_slice(b"VI");
            header[2..4].copy_from_slice(&1u16.to_le_bytes());
            header[4..8].copy_from_slice(&0x200100u32.to_le_bytes());
            header[8..12].copy_from_slice(&self.counter.to_le_bytes());
            header[16..18].copy_from_slice(&6u16.to_le_bytes());
            let mut pixels = vec![20u8; IMAGE_WIDTH * IMAGE_HEIGHT];
            for y in 0..IMAGE_HEIGHT {
                for x in 0..IMAGE_WIDTH {
                    let local_x = (x % CAMERA_WIDTH) as f32;
                    let cx = 100.0 + 28.0 * (t * 0.7).sin();
                    let cy = 100.0 + 12.0 * (t * 0.4).cos();
                    let d = ((local_x - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
                    pixels[y * IMAGE_WIDTH + x] = if d < 24.0 {
                        35
                    } else if d < 55.0 {
                        150
                    } else {
                        (70.0 + y as f32 * 0.35).min(180.0) as u8
                    };
                }
            }
            let image = EyeImageFrame {
                header,
                pixels,
                version: 1,
                total_size: 0x200100,
                timestamp: self.counter,
                image_type: 6,
            };
            (Some(gaze), Some(image))
        }
    }

    fn worker_main(shared: Arc<Mutex<LiveState>>, commands: Receiver<WorkerCommand>, mock: bool) {
        if mock {
            let backend = Backend::Mock(MockBackend::new());
            set_connected(&shared, &backend, "Synthetic 120 Hz input");
            run_connected(shared, commands, backend);
            return;
        }

        loop {
            if matches!(commands.try_recv(), Ok(WorkerCommand::Stop)) {
                return;
            }
            set_waiting(
                &shared,
                "Looking for PSVR2Toolkit...",
                "Start SteamVR with the updated PSVR2Toolkit driver installed.",
            );
            match Psvr2Capi::load() {
                Ok(mut capi) => {
                    let result = capi.initialize();
                    if result == RESULT_OK {
                        let backend = Backend::Real(capi);
                        set_connected(&shared, &backend, "PSVR2Toolkit CAPI connected");
                        if !run_connected(shared.clone(), &commands, backend) {
                            return;
                        }
                    } else {
                        let detail = match result {
                            RESULT_DRIVER_INACTIVE => {
                                "Toolkit CAPI loaded, but its SteamVR driver is inactive."
                            }
                            RESULT_NO_SLOT => {
                                "All eight Toolkit client slots are occupied. Close another eye-tracking client."
                            }
                            _ => "Toolkit initialization returned an unknown error.",
                        };
                        set_waiting(
                            &shared,
                            &format!("PSVR2Toolkit init failed ({result})"),
                            detail,
                        );
                    }
                }
                Err(error) => {
                    set_waiting(&shared, "PSVR2Toolkit CAPI not found", &error.to_string());
                }
            }

            for _ in 0..20 {
                match commands.try_recv() {
                    Ok(WorkerCommand::Stop) => return,
                    Ok(WorkerCommand::StartRecording) | Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => return,
                }
                thread::sleep(Duration::from_millis(100));
            }
        }
    }

    /// Returns true when the real backend should reconnect, false on app shutdown.
    fn run_connected(
        shared: Arc<Mutex<LiveState>>,
        commands: impl CommandSource,
        mut backend: Backend,
    ) -> bool {
        let mut recording: Option<ActiveRecording> = None;
        let mut rate_started = Instant::now();
        let mut gaze_count = 0u32;
        let mut image_count = 0u32;
        let mut consecutive_errors = 0u32;

        loop {
            loop {
                match commands.try_command() {
                    Ok(WorkerCommand::StartRecording) if recording.is_none() => {
                        recording = Some(ActiveRecording::new(backend.capi_path()));
                        beep();
                        let mut state = shared.lock().unwrap();
                        state.recording = Some(RecordingProgress {
                            phase_index: 0,
                            elapsed: 0.0,
                            step_elapsed: 0.0,
                            capturing: false,
                        });
                        state.report_error = None;
                    }
                    Ok(WorkerCommand::Stop) => return false,
                    Ok(WorkerCommand::StartRecording) => {}
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(mpsc::TryRecvError::Disconnected) => return false,
                }
            }

            if !backend.driver_active() {
                set_waiting(
                    &shared,
                    "PSVR2Toolkit driver stopped",
                    "Waiting for SteamVR and the PSVR2Toolkit driver to return.",
                );
                return true;
            }

            match backend.poll() {
                Ok((gaze, image)) => {
                    consecutive_errors = 0;
                    if let Some(status) = gaze {
                        gaze_count += 1;
                        if let Some(capture) = recording.as_mut() {
                            capture.push_gaze(status);
                        }
                        let mut state = shared.lock().unwrap();
                        state.latest_gaze = Some(status);
                        state.last_gaze_at = Some(Instant::now());
                        if state.last_image_at.is_some() {
                            state.detail =
                                "Raw gaze and 400x200 eye-image streams are live.".into();
                        } else {
                            state.detail =
                                "Native gaze is live; waiting for the eye-image stream.".into();
                        }
                    }
                    if let Some(frame) = image {
                        image_count += 1;
                        if let Some(capture) = recording.as_mut() {
                            capture.push_image(&frame);
                        }
                        let [a, b] = frame.split_camera_planes();
                        let mut state = shared.lock().unwrap();
                        state.camera_a = Arc::new(a);
                        state.camera_b = Arc::new(b);
                        state.image_generation = state.image_generation.wrapping_add(1);
                        state.last_image_at = Some(Instant::now());
                        if state.last_gaze_at.is_some() {
                            state.detail =
                                "Raw gaze and 400x200 eye-image streams are live.".into();
                        } else {
                            state.detail =
                                "Eye images are live; waiting for native gaze data.".into();
                        }
                    }
                }
                Err(error) => {
                    consecutive_errors += 1;
                    let mut state = shared.lock().unwrap();
                    state.detail = format!("Data validation error: {error}");
                    if consecutive_errors >= 30 {
                        return true;
                    }
                }
            }

            if rate_started.elapsed() >= Duration::from_secs(1) {
                let seconds = rate_started.elapsed().as_secs_f32();
                let mut state = shared.lock().unwrap();
                state.gaze_hz = gaze_count as f32 / seconds;
                state.image_hz = image_count as f32 / seconds;
                gaze_count = 0;
                image_count = 0;
                rate_started = Instant::now();
            }

            if let Some(capture) = recording.as_mut() {
                let elapsed = capture.elapsed();
                let phase = capture.phase_index();
                let capturing = capture.capture_active();
                if phase != capture.last_phase {
                    capture.last_phase = phase;
                    capture.last_capturing = false;
                    beep();
                }
                if capturing && !capture.last_capturing {
                    beep();
                }
                capture.last_capturing = capturing;
                {
                    let mut state = shared.lock().unwrap();
                    state.recording = Some(RecordingProgress {
                        phase_index: phase,
                        elapsed,
                        step_elapsed: capture.step_elapsed(),
                        capturing,
                    });
                }
                if elapsed >= TOTAL_SECONDS {
                    let completed = recording.take().unwrap();
                    let result = save_report(completed);
                    let mut state = shared.lock().unwrap();
                    state.recording = None;
                    match result {
                        Ok(path) => {
                            state.report_path = Some(path);
                            state.report_error = None;
                            beep();
                        }
                        Err(error) => {
                            state.report_error = Some(error.to_string());
                        }
                    }
                }
            }

            thread::sleep(Duration::from_millis(1));
        }
    }

    trait CommandSource {
        fn try_command(&self) -> Result<WorkerCommand, mpsc::TryRecvError>;
    }

    impl CommandSource for Receiver<WorkerCommand> {
        fn try_command(&self) -> Result<WorkerCommand, mpsc::TryRecvError> {
            self.try_recv()
        }
    }

    impl CommandSource for &Receiver<WorkerCommand> {
        fn try_command(&self) -> Result<WorkerCommand, mpsc::TryRecvError> {
            self.try_recv()
        }
    }

    fn set_connected(shared: &Arc<Mutex<LiveState>>, backend: &Backend, label: &str) {
        let mut state = shared.lock().unwrap();
        state.ready = true;
        state.connection = label.into();
        state.detail = "Waiting for gaze and 400x200 eye-image frames...".into();
        state.capi_path = backend.capi_path();
    }

    fn set_waiting(shared: &Arc<Mutex<LiveState>>, label: &str, detail: &str) {
        let mut state = shared.lock().unwrap();
        state.ready = false;
        state.connection = label.into();
        state.detail = detail.into();
        state.capi_path = None;
        state.gaze_hz = 0.0;
        state.image_hz = 0.0;
    }

    fn beep() {
        unsafe {
            let _ = MessageBeep(MB_OK);
        }
    }

    fn report_directory() -> PathBuf {
        let base = std::env::var_os("USERPROFILE")
            .map(PathBuf::from)
            .map(|p| p.join("Desktop"))
            .filter(|p| p.is_dir())
            .unwrap_or_else(std::env::temp_dir);
        base.join("SRanibro PSVR2 reports")
    }

    fn save_report(capture: ActiveRecording) -> io::Result<PathBuf> {
        let directory = report_directory();
        fs::create_dir_all(&directory)?;
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = directory.join(format!("sranibro_psvr2_probe_{stamp}.zip"));
        let file = File::create(&path)?;
        let mut zip = zip::ZipWriter::new(file);
        let options = SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);

        let mut summary = String::new();
        let _ = writeln!(summary, "SRanibro PSVR2 Hardware Probe");
        let _ = writeln!(summary, "SRanibro version: {}", env!("CARGO_PKG_VERSION"));
        let _ = writeln!(
            summary,
            "PSVR2Toolkit contract commit: {TOOLKIT_CONTRACT_COMMIT}"
        );
        let _ = writeln!(
            summary,
            "CAPI path: {}",
            capture
                .capi_path
                .as_deref()
                .map(Path::display)
                .map(|p| p.to_string())
                .unwrap_or_else(|| "MOCK".into())
        );
        let _ = writeln!(summary, "Duration: {:.3} s", capture.elapsed());
        let _ = writeln!(summary, "Gaze samples: {}", capture.gaze_samples);
        let _ = writeln!(summary, "Eye-image frames: {}", capture.image_frames);
        let denominator = capture.gaze_samples.max(1) as f64;
        let _ = writeln!(
            summary,
            "Valid gaze L/R: {:.1}% / {:.1}%",
            capture.valid_gaze_l as f64 * 100.0 / denominator,
            capture.valid_gaze_r as f64 * 100.0 / denominator
        );
        let _ = writeln!(
            summary,
            "Valid pupil L/R: {:.1}% / {:.1}%",
            capture.valid_pupil_l as f64 * 100.0 / denominator,
            capture.valid_pupil_r as f64 * 100.0 / denominator
        );
        let _ = writeln!(
            summary,
            "Expected image preview: {}x{} split into A/B {}x{}",
            IMAGE_WIDTH, IMAGE_HEIGHT, CAMERA_WIDTH, CAMERA_HEIGHT
        );
        let _ = writeln!(
            summary,
            "Captured phase snapshots: {}",
            capture.snapshots.len()
        );
        if let Some(status) = capture.last_status {
            let _ = writeln!(
                summary,
                "Last gaze block: magic={:?} version={} size={} frame={} calibration_id={} dsp={}",
                status.magic,
                status.version,
                status.size,
                status.wearable.frame_counter,
                status.user_calibration_id,
                status.dsp_return_code
            );
        }
        summary.push_str(
            "\nInterpretation rule:\n\
             Camera A/B are intentionally NOT called left/right. The held LEFT and RIGHT wink\n\
             snapshots establish the anatomical order and whether either camera needs mirroring.\n",
        );

        zip_start(&mut zip, "summary.txt", options)?;
        zip.write_all(summary.as_bytes())?;
        zip_start(&mut zip, "gaze.csv", options)?;
        zip.write_all(capture.csv.as_bytes())?;

        for snapshot in capture.snapshots {
            let phase = PHASES[snapshot.phase];
            let prefix = format!("{:02}_{}", snapshot.phase + 1, phase.slug);
            zip_start(&mut zip, &format!("{prefix}/header.bin"), options)?;
            zip.write_all(&snapshot.header)?;

            let metadata = format!(
                "phase={}\nversion={}\ntotal_size={}\ntimestamp={}\nimage_type={}\n",
                phase.title,
                snapshot.version,
                snapshot.total_size,
                snapshot.timestamp,
                snapshot.image_type
            );
            zip_start(&mut zip, &format!("{prefix}/metadata.txt"), options)?;
            zip.write_all(metadata.as_bytes())?;
            write_png_to_zip(
                &mut zip,
                &format!("{prefix}/combined_400x200.png"),
                IMAGE_WIDTH as u32,
                IMAGE_HEIGHT as u32,
                &snapshot.combined,
                options,
            )?;
            write_png_to_zip(
                &mut zip,
                &format!("{prefix}/camera_a_200x200.png"),
                CAMERA_WIDTH as u32,
                CAMERA_HEIGHT as u32,
                &snapshot.camera_a,
                options,
            )?;
            write_png_to_zip(
                &mut zip,
                &format!("{prefix}/camera_b_200x200.png"),
                CAMERA_WIDTH as u32,
                CAMERA_HEIGHT as u32,
                &snapshot.camera_b,
                options,
            )?;
        }

        zip.finish().map_err(zip_error)?;
        Ok(path)
    }

    fn zip_start(
        zip: &mut zip::ZipWriter<File>,
        name: &str,
        options: SimpleFileOptions,
    ) -> io::Result<()> {
        zip.start_file(name, options).map_err(zip_error)
    }

    fn write_png_to_zip(
        zip: &mut zip::ZipWriter<File>,
        name: &str,
        width: u32,
        height: u32,
        pixels: &[u8],
        options: SimpleFileOptions,
    ) -> io::Result<()> {
        let mut encoded = Vec::new();
        {
            let mut encoder = png::Encoder::new(&mut encoded, width, height);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header()?;
            writer.write_image_data(pixels)?;
        }
        zip_start(zip, name, options)?;
        zip.write_all(&encoded)
    }

    fn zip_error(error: zip::result::ZipError) -> io::Error {
        io::Error::new(io::ErrorKind::Other, error)
    }

    pub fn run() -> eframe::Result<()> {
        let mock = std::env::args().any(|arg| arg == "--mock");
        let options = eframe::NativeOptions {
            renderer: eframe::Renderer::Glow,
            viewport: egui::ViewportBuilder::default()
                .with_title("SRanibro PSVR2 Hardware Probe")
                .with_inner_size([1120.0, 820.0])
                .with_min_inner_size([920.0, 680.0]),
            ..Default::default()
        };
        eframe::run_native(
            "SRanibro PSVR2 Hardware Probe",
            options,
            Box::new(move |cc| Ok(Box::new(ProbeApp::new(&cc.egui_ctx, mock)))),
        )
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn every_phase_has_a_prepare_interval_before_capture() {
            for phase in 0..PHASES.len() {
                let start = phase as f32 * STEP_SECONDS;
                assert_eq!(phase_index_at(start), phase);
                assert!(!capture_active_at(start));
                assert!(!capture_active_at(start + PREP_SECONDS - 0.001));
                assert!(capture_active_at(start + PREP_SECONDS));
                assert!(capture_active_at(start + STEP_SECONDS - 0.001));
            }
        }

        #[test]
        fn phase_boundaries_return_to_prepare_state() {
            for phase in 1..PHASES.len() {
                let boundary = phase as f32 * STEP_SECONDS;
                assert_eq!(phase_index_at(boundary), phase);
                assert_eq!(step_elapsed_at(boundary), 0.0);
                assert!(!capture_active_at(boundary));
            }
        }

        #[test]
        fn schedule_is_four_seconds_prepare_then_eight_seconds_capture() {
            assert_eq!(PREP_SECONDS, 4.0);
            assert_eq!(CAPTURE_SECONDS, 8.0);
            assert_eq!(STEP_SECONDS, 12.0);
            assert_eq!(TOTAL_SECONDS, 84.0);
        }
    }
}

#[cfg(windows)]
fn main() -> eframe::Result<()> {
    app::run()
}
