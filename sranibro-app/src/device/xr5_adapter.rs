
#![cfg(any(windows, target_os = "linux"))]

use super::vr4_adapter::Vr4Adapter;
use crate::config::GazeSource;
use crate::core::types::{DeviceProfile, Eye};

pub struct Xr5Adapter(Vr4Adapter);

impl Xr5Adapter {
    #[cfg(feature = "xr5-only")]
    pub fn set_wearable_packet_callback(&mut self, callback: Box<dyn FnMut(&[u8]) + Send>) {
        self.0.set_wearable_packet_callback(callback);
    }
    #[cfg(feature = "xr5-only")]
    pub fn set_gaze_packet_callback(&mut self, callback: Box<dyn FnMut(&[u8]) + Send>) {
        self.0.set_gaze_packet_callback(callback);
    }
    fn profile(gaze_source: GazeSource) -> DeviceProfile {
        DeviceProfile {
            name: "Pimax XR5".into(),
            ml_device: "xr5".into(),
            image_w: 200,
            image_h: 200,
            slot_a_eye: Eye::Left,
            transport: if cfg!(target_os = "linux") { "libusb + native TTP" } else { "WinUSB + native TTP" }.into(),
            streams: "TTP image 1291 + wearable 1285".into(),
            gaze_src: match gaze_source {
                GazeSource::PerEye => "per-eye gaze · pupil · openness (Tobii)".into(),
                GazeSource::Combined => "combined gaze · pupil · openness (Tobii)".into(),
            },
        }
    }

    pub fn new(cfg: &crate::config::Config) -> Self {
        let gaze_source = cfg.gaze_source_for("pimax_xr5");
        let dll_path = if cfg!(windows) { cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned()) } else { None };
        Self(Vr4Adapter::with_profile(
            Self::profile(gaze_source),
            "pimax-xr5",
            "[xr5]",
            dll_path,
            cfg!(windows),
            false,
            gaze_source == GazeSource::Combined,
        ))
    }
}

impl super::HmdAdapter for Xr5Adapter {
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn profile(&self) -> &DeviceProfile {
        self.0.profile()
    }

    fn set_raw_gaze_callback(&mut self, callback: super::GazeFn) -> bool {
        self.0.set_raw_gaze_callback(callback)
    }
    fn set_native_evidence_callback(&mut self, callback: super::GazeFn) -> bool {
        self.0.set_native_evidence_callback(callback)
    }

    fn start(&mut self, on_frame: super::FrameFn, on_gaze: super::GazeFn) -> std::io::Result<()> {
        self.0.start(on_frame, on_gaze)
    }

    fn stop(&mut self) {
        self.0.stop()
    }

    fn status_arc(&self) -> std::sync::Arc<std::sync::Mutex<String>> {
        self.0.status_arc()
    }

    fn needs_eyechip_handoff(&self) -> bool {
        cfg!(windows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::HmdAdapter;

    #[test]
    fn xr5_profile_is_angled_dll_free_and_left_slot() {
        let p = Xr5Adapter::profile(GazeSource::PerEye);
        assert_eq!(p.name, "Pimax XR5");
        assert_eq!(p.ml_device, "xr5", "angled-optics ML route");
        assert_eq!(p.slot_a_eye, Eye::Left, "region 1 = LEFT");
        assert!(
            p.transport.contains("native TTP"),
            "native WinUSB+TTP transport"
        );
        assert!(p.streams.contains("1291"), "TTP image stream 1291");
    }

    #[test]
    fn xr5_adapter_delegates_identity_and_handoff() {
        let cfg = crate::config::Config::default();
        let a = Xr5Adapter::new(&cfg);
        assert_eq!(a.name(), "pimax-xr5");
        assert_eq!(a.profile().ml_device, "xr5");
        assert!(
            !a.0.uses_wearable_gaze(),
            "XR5 gaze must come only from stream 1289"
        );

        assert!(!a.0.uses_combined_gaze());
        assert_eq!(a.needs_eyechip_handoff(), cfg!(windows));
    }

    #[test]
    fn xr5_combined_gaze_is_opt_in_and_visible_in_the_profile() {
        let mut cfg = crate::config::Config::default();
        cfg.set_gaze_source("dream_air", GazeSource::Combined);
        let combined = Xr5Adapter::new(&cfg);
        assert!(combined.0.uses_combined_gaze());
        assert!(combined.profile().gaze_src.contains("combined"));

        let default = Xr5Adapter::new(&crate::config::Config::default());
        assert!(!default.0.uses_combined_gaze());
        assert!(default.profile().gaze_src.contains("per-eye"));
    }

    #[test]
    fn xr5_combined_gaze_accepts_an_independent_raw_callback() {
        let mut cfg = crate::config::Config::default();
        cfg.set_gaze_source("pimax_xr5", GazeSource::Combined);
        let mut adapter = Xr5Adapter::new(&cfg);
        assert!(adapter.set_raw_gaze_callback(Box::new(|_| {})));
        assert!(adapter.0.uses_combined_gaze());
    }
}
