//! PlayStation VR2 acquisition through PSVR2Toolkit's installed read-only CAPI.
//!
//! The Toolkit owns the headset and SteamVR integration. SRanibro only consumes
//! the paired eye-camera preview and wearable gaze/pupil/blink data. A labelled
//! 120 Hz hardware recording established the product mapping: preview half A is
//! anatomical left, half B is anatomical right, neither image needs mirroring,
//! and native gaze X uses SRanibro's ordinary per-device X flip.

#![cfg(windows)]

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::psvr2_capi::{
    GazeStatus, GazeVec3, Psvr2Capi, WearableEye, CAMERA_HEIGHT, CAMERA_WIDTH,
    RESULT_DRIVER_INACTIVE, RESULT_NO_SLOT, RESULT_OK,
};
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye, EyeSample, GazeSample};

pub struct Psvr2Adapter {
    profile: DeviceProfile,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}

impl Default for Psvr2Adapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Psvr2Adapter {
    pub fn new() -> Self {
        Self {
            profile: DeviceProfile {
                name: "PlayStation VR2".into(),
                // The captured cameras are frontal, centered 200x200 views. Start
                // from the identity/VR4 SRanipal preprocessing route; PSVR2 keeps
                // its own calibration and geometry bucket.
                ml_device: "vr4".into(),
                image_w: CAMERA_WIDTH as u32,
                image_h: CAMERA_HEIGHT as u32,
                slot_a_eye: Eye::Left,
                transport: "PSVR2Toolkit CAPI".into(),
                streams: "native gaze + paired 400x200 eye-camera preview".into(),
                gaze_src: "PSVR2 gaze · pupil · blink".into(),
            },
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }
}

fn set_status(status: &Arc<Mutex<String>>, message: impl Into<String>) {
    if let Ok(mut value) = status.lock() {
        *value = message.into();
    }
}

fn finite_vec3(value: GazeVec3) -> bool {
    value.x.is_finite() && value.y.is_finite() && value.z.is_finite()
}

fn eye_sample(native: WearableEye) -> EyeSample {
    let blink_reported = native.blink_valid();
    let blink = native.blink != 0;
    EyeSample {
        gaze: [
            native.gaze_dir_norm.x,
            native.gaze_dir_norm.y,
            native.gaze_dir_norm.z,
        ],
        gaze_valid: native.gaze_valid(),
        // The status block always contains an explicit per-eye gaze validity bit.
        gaze_reported: true,
        origin_mm: [
            native.gaze_origin_mm.x,
            native.gaze_origin_mm.y,
            native.gaze_origin_mm.z,
        ],
        origin_valid: native.is_gaze_origin_valid != 0 && finite_vec3(native.gaze_origin_mm),
        pupil_mm: native.pupil_dia_mm,
        pupil_valid: native.pupil_valid(),
        pupil_pos: [
            native.pupil_pos_in_sensor_area.x,
            native.pupil_pos_in_sensor_area.y,
        ],
        pupil_pos_valid: native.pupil_position_valid(),
        // Invalidity is explicit and must clear a stale pupil position.
        pupil_pos_reported: true,
        // PSVR2 exposes a binary native blink classifier, not continuous
        // openness. Feed it only as the absolute open/closed classifier used by
        // SRanibro's blink latch. Camera ML remains the continuous eyelid signal.
        openness: if blink { 0.0 } else { 1.0 },
        openness_valid: blink_reported && !blink,
        openness_reported: blink_reported,
    }
}

fn gaze_sample(status: GazeStatus) -> GazeSample {
    GazeSample {
        timestamp_us: status.wearable.timestamp.max(0) as u64,
        left: eye_sample(status.wearable.left),
        right: eye_sample(status.wearable.right),
    }
}

fn init_error(code: i32) -> io::Error {
    let detail = match code {
        RESULT_DRIVER_INACTIVE => {
            "PSVR2Toolkit is installed, but its SteamVR driver is inactive"
        }
        RESULT_NO_SLOT => {
            "all PSVR2Toolkit client slots are occupied; close the hardware probe or another eye-tracking client"
        }
        _ => "PSVR2Toolkit initialization returned an unknown error",
    };
    io::Error::new(
        io::ErrorKind::ConnectionRefused,
        format!("{detail} (code {code})"),
    )
}

impl HmdAdapter for Psvr2Adapter {
    fn name(&self) -> &'static str {
        "psvr2"
    }

    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, mut on_frame: FrameFn, mut on_gaze: GazeFn) -> io::Result<()> {
        if self.thread.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "PSVR2 adapter is already running",
            ));
        }

        self.stop.store(false, Ordering::Relaxed);
        set_status(&self.status, "loading PSVR2Toolkit CAPI");

        let stop = self.stop.clone();
        let status = self.status.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<io::Result<()>>(1);
        self.thread = Some(thread::spawn(move || {
            let mut capi = match Psvr2Capi::load() {
                Ok(capi) => capi,
                Err(error) => {
                    eprintln!("[psvr2] CAPI discovery/load failed: {error}");
                    set_status(&status, error.to_string());
                    let _ = ready_tx.send(Err(error));
                    return;
                }
            };
            let result = capi.initialize();
            if result != RESULT_OK {
                let error = init_error(result);
                eprintln!("[psvr2] CAPI initialization failed: {error}");
                set_status(&status, error.to_string());
                let _ = ready_tx.send(Err(error));
                return;
            }
            if capi.driver_active() == Some(false) {
                let error = init_error(RESULT_DRIVER_INACTIVE);
                eprintln!("[psvr2] driver check failed: {error}");
                set_status(&status, error.to_string());
                let _ = ready_tx.send(Err(error));
                return;
            }

            eprintln!("[psvr2] CAPI connected");
            set_status(&status, "connected · waiting for gaze and eye images");
            if ready_tx.send(Ok(())).is_err() {
                return;
            }

            let mut saw_gaze = false;
            let mut saw_image = false;
            let mut reported_streams = (false, false);
            let mut consecutive_errors = 0u32;
            while !stop.load(Ordering::Relaxed) {
                if capi.driver_active() == Some(false) {
                    set_status(&status, "PSVR2Toolkit SteamVR driver stopped");
                    break;
                }

                let mut received = false;
                match capi.next_gaze(0) {
                    Ok(Some(native)) => {
                        on_gaze(gaze_sample(native));
                        saw_gaze = true;
                        received = true;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        consecutive_errors = consecutive_errors.saturating_add(1);
                        set_status(&status, format!("invalid PSVR2 gaze data: {error}"));
                    }
                }

                match capi.next_image(0) {
                    Ok(Some(frame)) => {
                        let [left, right] = frame.split_camera_planes();
                        on_frame(Eye::Left, CAMERA_WIDTH as u32, CAMERA_HEIGHT as u32, &left);
                        on_frame(
                            Eye::Right,
                            CAMERA_WIDTH as u32,
                            CAMERA_HEIGHT as u32,
                            &right,
                        );
                        saw_image = true;
                        received = true;
                    }
                    Ok(None) => {}
                    Err(error) => {
                        consecutive_errors = consecutive_errors.saturating_add(1);
                        set_status(&status, format!("invalid PSVR2 eye image: {error}"));
                    }
                }

                if received {
                    consecutive_errors = 0;
                    let streams = (saw_gaze, saw_image);
                    if streams != reported_streams {
                        set_status(
                            &status,
                            match streams {
                                (true, true) => "streaming gaze + eye images at native rate",
                                (true, false) => "gaze live · waiting for eye-camera access",
                                (false, true) => "eye images live · waiting for native gaze",
                                (false, false) => "connected · waiting for data",
                            },
                        );
                        reported_streams = streams;
                    }
                } else {
                    thread::sleep(Duration::from_millis(1));
                }

                if consecutive_errors >= 30 {
                    set_status(&status, "PSVR2Toolkit data validation repeatedly failed");
                    break;
                }
            }
            eprintln!("[psvr2] stopped");
        }));

        match ready_rx.recv_timeout(Duration::from_secs(8)) {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.stop();
                Err(error)
            }
            Err(_) => {
                self.stop();
                let error = io::Error::new(
                    io::ErrorKind::TimedOut,
                    "timed out while starting PSVR2Toolkit",
                );
                eprintln!("[psvr2] startup failed: {error}");
                Err(error)
            }
        }
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        set_status(&self.status, "stopped");
    }

    fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }
}

impl Drop for Psvr2Adapter {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::super::psvr2_capi::GazeVec2;
    use super::*;

    fn tracked_eye(gaze_x: f32, blink: bool) -> WearableEye {
        WearableEye {
            is_gaze_origin_valid: 1,
            gaze_origin_mm: GazeVec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            is_gaze_dir_valid: 1,
            gaze_dir_norm: GazeVec3 {
                x: gaze_x,
                y: -0.2,
                z: 0.97,
            },
            is_pupil_dia_valid: 1,
            pupil_dia_mm: 4.2,
            is_pupil_pos_in_sensor_area_valid: 1,
            pupil_pos_in_sensor_area: GazeVec2 { x: 0.4, y: 0.6 },
            is_blink_valid: 1,
            blink: blink as u32,
            ..WearableEye::default()
        }
    }

    #[test]
    fn hardware_mapping_keeps_raw_gaze_and_routes_a_b_as_left_right() {
        let mut status = GazeStatus::default();
        status.wearable.timestamp = 123_456;
        status.wearable.left = tracked_eye(0.5, false);
        status.wearable.right = tracked_eye(-0.4, false);
        let sample = gaze_sample(status);

        assert_eq!(sample.timestamp_us, 123_456);
        assert_eq!(sample.left.gaze[0], 0.5);
        assert_eq!(sample.right.gaze[0], -0.4);
        assert!(sample.left.gaze_valid && sample.right.gaze_valid);
        assert_eq!(Psvr2Adapter::new().profile().slot_a_eye, Eye::Left);
    }

    #[test]
    fn native_blink_is_absolute_closed_evidence_not_analog_openness() {
        let open = eye_sample(tracked_eye(0.0, false));
        assert!(open.openness_reported);
        assert!(open.openness_valid);
        assert_eq!(open.openness, 1.0);

        let closed = eye_sample(tracked_eye(0.0, true));
        assert!(closed.openness_reported);
        assert!(!closed.openness_valid);
        assert_eq!(closed.openness, 0.0);
    }

    #[test]
    fn unreported_blink_does_not_fabricate_native_openness() {
        let mut eye = tracked_eye(0.0, false);
        eye.is_blink_valid = 0;
        let sample = eye_sample(eye);
        assert!(!sample.openness_reported);
        assert!(!sample.openness_valid);
    }
}
