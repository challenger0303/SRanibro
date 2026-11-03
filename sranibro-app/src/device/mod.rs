
pub(crate) mod runtime_access;
pub mod gaze;
pub mod ttp;
pub mod vr4;
pub mod vr4_gaze_source;

#[cfg(any(windows, target_os = "linux"))]
pub mod psvr2_adapter;
#[cfg(any(windows, target_os = "linux"))]
pub mod psvr2_capi;
#[cfg(any(windows, target_os = "linux"))]
pub mod psvr2_worker;
#[cfg(windows)]
pub mod starvr_adapter;
#[cfg(windows)]
pub mod usb;
#[cfg(all(target_os = "linux", feature = "xr5-only"))]
#[path = "usb_linux.rs"]
pub mod usb;
#[cfg(windows)]
pub mod varjo_adapter;
#[cfg(any(windows, all(target_os = "linux", feature = "xr5-only")))]
pub mod vr4_adapter;
#[cfg(windows)]
pub mod varjo_native_adapter;
#[cfg(windows)]
pub mod varjo_sdk;
#[cfg(windows)]
pub mod vpe_adapter;
#[cfg(windows)]
pub mod vpe_hid;
#[cfg(windows)]
pub mod vpe_probe;
#[cfg(any(all(windows, any(feature = "xr5-only", feature = "external-eyelid-model")), all(target_os = "linux", feature = "xr5-only")))]
pub mod xr5_adapter;

use std::sync::{Arc, Mutex};

use crate::core::types::{DeviceProfile, Eye, GazeSample};

pub fn auto_device_key_from_serial(serial: Option<&str>) -> &'static str {
    match serial.map(str::trim).filter(|serial| !serial.is_empty()) {
        Some(serial) if serial.to_ascii_uppercase().starts_with("XR5") => "pimax_xr5",
        Some(serial) if serial.to_ascii_uppercase().starts_with("VRS") => "starvr",
        _ => "pimax_vr4",
    }
}

pub const XR5_PAUSED_MESSAGE: &str =
    "Dream Air / XR5 support is separated from the normal build and is currently paused";

pub type FrameFn = Box<dyn FnMut(Eye, u32, u32, &[u8]) + Send>;
pub type GazeFn = Box<dyn FnMut(GazeSample) + Send>;

pub trait HmdAdapter: Send {
    fn name(&self) -> &'static str;
    fn profile(&self) -> &DeviceProfile;
    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> std::io::Result<()>;
    fn set_raw_gaze_callback(&mut self, _callback: GazeFn) -> bool {
        false
    }
    fn set_native_evidence_callback(&mut self, _callback: GazeFn) -> bool {
        false
    }
    fn stop(&mut self);
    fn needs_eyechip_handoff(&self) -> bool {
        false
    }
    fn status_arc(&self) -> Arc<Mutex<String>> {
        Arc::new(Mutex::new("n/a".into()))
    }
}

#[cfg(all(
    any(windows, target_os = "linux"),
    feature = "psvr2-only",
    not(feature = "xr5-only")
))]
pub fn make_adapter(_cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    Ok(Box::new(psvr2_adapter::Psvr2Adapter::new()))
}

#[cfg(all(
    target_os = "linux",
    not(any(feature = "psvr2-only", feature = "xr5-only"))
))]
pub fn make_adapter(cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    let device = crate::config::canonical_device_key(&cfg.hmd.device);
    match device.as_str() {
        "auto" | "psvr2" => Ok(Box::new(psvr2_adapter::Psvr2Adapter::new())),
        other => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!("device '{other}' is not supported on Linux; select psvr2 (PSVR2Toolkit with Ignition)"),
        )),
    }
}

#[cfg(all(target_os = "linux", feature = "xr5-only", not(feature = "psvr2-only")))]
pub fn make_adapter(cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    Ok(Box::new(xr5_adapter::Xr5Adapter::new(cfg)))
}

#[cfg(all(test, target_os = "linux", feature = "xr5-only", not(feature = "psvr2-only")))]
#[test]
fn linux_xr5_build_constructs_native_adapter_without_windows_handoff() {
    let adapter = make_adapter(&crate::config::Config::default()).unwrap();
    assert_eq!(adapter.name(), "pimax-xr5");
    assert!(adapter.profile().transport.contains("libusb"));
    assert!(!adapter.needs_eyechip_handoff());
}

#[cfg(all(windows, feature = "xr5-only", not(feature = "psvr2-only")))]
pub fn make_adapter(cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    Ok(Box::new(xr5_adapter::Xr5Adapter::new(cfg)))
}

#[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
pub fn make_adapter(cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    use std::io::{Error, ErrorKind};
    let device = crate::config::canonical_device_key(&cfg.hmd.device);
    match device.as_str() {
        "auto" => match usb::peek_serial() {
            Some(s) if auto_device_key_from_serial(Some(&s)) == "pimax_xr5" => {
                #[cfg(feature = "external-eyelid-model")]
                { eprintln!("[auto] eyechip serial={s} -> Dream Air / XR5"); Ok(Box::new(xr5_adapter::Xr5Adapter::new(cfg))) }
                #[cfg(not(feature = "external-eyelid-model"))]
                {
                eprintln!("[auto] eyechip serial={s} -> XR5 (separate build paused)");
                Err(Error::new(ErrorKind::Unsupported, XR5_PAUSED_MESSAGE))
                }
            }
            Some(s) if auto_device_key_from_serial(Some(&s)) == "starvr" => {
                eprintln!("[auto] eyechip serial={s} → StarVR One");
                Ok(Box::new(starvr_adapter::StarVrAdapter::new(cfg)))
            }
            Some(s) => {
                eprintln!("[auto] eyechip serial={s} → VR4 (frontal)");
                let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
                Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)
                    .with_gaze_source(cfg.hmd.vr4_gaze_source)))
            }
            None => {
                eprintln!("[auto] no eyechip found yet → defaulting to VR4");
                let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
                Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)
                    .with_gaze_source(cfg.hmd.vr4_gaze_source)))
            }
        },
        "pimax_vr4" => {
            let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
            Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)
                .with_gaze_source(cfg.hmd.vr4_gaze_source)))
        }
        "pimax_dll" => Ok(Box::new(
            starvr_adapter::StarVrAdapter::with_profile(
                cfg, "Pimax (Tobii stream engine)", Eye::Left, true,
            ),
        )),
        "starvr" => Ok(Box::new(starvr_adapter::StarVrAdapter::new(cfg))),
        "varjo" => match cfg.varjo_lib_path() {
            Some(p) => Ok(Box::new(varjo_native_adapter::VarjoNativeAdapter::new(
                p.to_string_lossy().into_owned(),
            ))),
            None => Err(Error::new(
                ErrorKind::Unsupported,
                "device=\"varjo\": Varjo Base native runtime not found — install Varjo Base, or use \
                 device=\"varjo_mjpeg\" (Varjo Eye Streamer)",
            )),
        },
        "varjo_mjpeg" => Ok(Box::new(varjo_adapter::VarjoAdapter::new(cfg))),
        "psvr2" => Ok(Box::new(psvr2_adapter::Psvr2Adapter::new())),
        #[cfg(feature = "external-eyelid-model")]
        "pimax_xr5" => Ok(Box::new(xr5_adapter::Xr5Adapter::new(cfg))),
        #[cfg(not(feature = "external-eyelid-model"))]
        "pimax_xr5" => Err(Error::new(ErrorKind::Unsupported, XR5_PAUSED_MESSAGE)),
        "vpe" => Ok(Box::new(vpe_adapter::VpeAdapter::new())),
        other => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("unknown [hmd].device '{other}' — use one of: auto, pimax_vr4, starvr, varjo, varjo_mjpeg, psvr2"),
        )),
    }
}

#[cfg(all(test, any(windows, target_os = "linux")))]
mod adapter_alias_tests {
    use super::*;

    #[test]
    #[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
    fn vr4_aliases_use_the_saved_source_and_do_not_affect_other_devices() {
        use crate::config::Vr4GazeSource;
        for source in [Vr4GazeSource::V1, Vr4GazeSource::V2] {
            let mut cfg = crate::config::Config::default();
            cfg.hmd.vr4_gaze_source = source;
            for alias in ["pimax_vr4", "pimax-vr4", "vr4", "crystal", "Crystal Super"] {
                cfg.hmd.device = alias.into();
                let adapter = make_adapter(&cfg).unwrap();
                assert_eq!(adapter.name(), "pimax-vr4");
                assert_eq!(adapter.profile().gaze_src, format!("Tobii gaze {}", source.label()));
            }
            cfg.hmd.device = "psvr2".into();
            assert_eq!(make_adapter(&cfg).unwrap().name(), "psvr2");
            cfg.hmd.device = "starvr".into();
            assert_eq!(make_adapter(&cfg).unwrap().name(), "starvr-one");
        }
    }

    #[test]
    fn auto_serial_classification_keeps_starvr_out_of_the_vr4_path() {
        assert_eq!(
            auto_device_key_from_serial(Some("VRS02-834F1B587600")),
            "starvr"
        );
        assert_eq!(auto_device_key_from_serial(Some("xr5-1234")), "pimax_xr5");
        assert_eq!(auto_device_key_from_serial(Some("VRU02-1234")), "pimax_vr4");
        assert_eq!(auto_device_key_from_serial(None), "pimax_vr4");
    }

    #[test]
    #[cfg(all(windows, feature = "xr5-only"))]
    fn xr5_aliases_reach_the_same_adapter() {
        for alias in ["xr5", "pimax_xr5", "pimax-xr5", "dream_air", "dream-air"] {
            let mut cfg = crate::config::Config::default();
            cfg.hmd.device = alias.into();
            let adapter = make_adapter(&cfg).unwrap_or_else(|e| panic!("{alias}: {e}"));
            assert_eq!(adapter.name(), "pimax-xr5", "alias {alias}");
        }
    }

    #[test]
    #[cfg(not(feature = "xr5-only"))]
    fn psvr2_aliases_reach_the_toolkit_adapter_without_hardware_access() {
        for alias in ["psvr2", "ps-vr2", "playstation-vr2"] {
            let mut cfg = crate::config::Config::default();
            cfg.hmd.device = alias.into();
            let adapter = make_adapter(&cfg).unwrap_or_else(|e| panic!("{alias}: {e}"));
            assert_eq!(adapter.name(), "psvr2", "alias {alias}");
            assert_eq!(adapter.profile().slot_a_eye, Eye::Left);
            assert_eq!(adapter.profile().ml_device, "vr4");
        }
    }

    #[test]
    #[cfg(feature = "psvr2-only")]
    fn psvr2_only_build_ignores_a_copied_non_psvr2_device_setting() {
        let mut cfg = crate::config::Config::default();
        cfg.hmd.device = "starvr".into();
        let adapter = make_adapter(&cfg).unwrap();
        assert_eq!(adapter.name(), "psvr2");
    }

    #[test]
    #[cfg(all(windows, feature = "xr5-only"))]
    fn xr5_only_build_ignores_a_copied_non_xr5_device_setting() {
        let mut cfg = crate::config::Config::default();
        cfg.hmd.device = "starvr".into();
        let adapter = make_adapter(&cfg).unwrap();
        assert_eq!(adapter.name(), "pimax-xr5");
    }

    #[test]
    #[cfg(all(windows, not(any(feature = "psvr2-only", feature = "xr5-only"))))]
    fn normal_build_rejects_xr5_instead_of_misrouting_it() {
        for alias in ["xr5", "pimax_xr5", "pimax-xr5", "dream_air", "dream-air"] {
            let mut cfg = crate::config::Config::default();
            cfg.hmd.device = alias.into();
            #[cfg(feature = "external-eyelid-model")]
            { assert_eq!(make_adapter(&cfg).unwrap().name(), "pimax-xr5"); continue; }
            #[cfg(not(feature = "external-eyelid-model"))]
            {
            let error = match make_adapter(&cfg) {
                Ok(_) => panic!("{alias}: XR5 adapter unexpectedly available"),
                Err(error) => error,
            };
            assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
            assert!(
                error.to_string().contains("currently paused"),
                "{alias}: {error}"
            );
            }
        }
    }

    #[test]
    #[cfg(all(
        target_os = "linux",
        not(any(feature = "psvr2-only", feature = "xr5-only"))
    ))]
    fn linux_auto_uses_psvr2_and_other_headsets_fail_explicitly() {
        let mut cfg = crate::config::Config::default();
        cfg.hmd.device = "auto".into();
        assert_eq!(make_adapter(&cfg).unwrap().name(), "psvr2");
        for device in ["pimax_vr4", "starvr", "varjo", "vpe", "xr5"] {
            cfg.hmd.device = device.into();
            let error = make_adapter(&cfg)
                .err()
                .expect("Windows devices must not be constructed on Linux");
            assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
            assert!(error.to_string().contains("Linux"));
        }
    }
}
