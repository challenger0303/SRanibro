//! Device acquisition: the HMD-specific seam (the only place HMDs differ).
//!
//! Status: the pure TTP protocol ([`ttp`]) and the VR4 data layout ([`vr4`]) are
//! ported and unit-tested offline. The WinUSB + named-pipe + libloading glue that
//! drives real hardware is the next step (it cannot be verified without a
//! physical VR4: the platform service must be stopped and the patched DLL present).

pub mod auth;
pub mod gaze;
pub mod ttp;
pub mod vr4;

#[cfg(windows)]
pub mod psvr2_adapter;
#[cfg(windows)]
pub mod psvr2_capi;
#[cfg(windows)]
pub mod starvr_adapter;
#[cfg(windows)]
pub mod usb;
#[cfg(windows)]
pub mod varjo_adapter;
#[cfg(windows)]
pub mod vr4_adapter;
// Native Varjo SDK path RE-ENABLED 2026-06-29: the crash-dump analysis cleared it (the
// MEMORY_MANAGEMENT BSODs were attributed to RAM/an ImDisk RAM disk, no Varjo driver in the
// faulting stack). `varjo_mjpeg` stays as the safe fallback.
#[cfg(windows)]
pub mod varjo_native_adapter;
#[cfg(windows)]
pub mod varjo_sdk;
#[cfg(windows)]
pub mod vpe_adapter;
/// `vpewake` — VPE eyechip wake via the Windows HID API (the 0BB4:0309 HidUsb interface).
#[cfg(windows)]
pub mod vpe_hid;
/// `vpetest`/`vpescan` diagnostics — validate the native VPE eyechip wake (USB keepalive).
#[cfg(windows)]
pub mod vpe_probe;
#[cfg(all(windows, feature = "xr5-only"))]
pub mod xr5_adapter;

use std::sync::{Arc, Mutex};

use crate::core::types::{DeviceProfile, Eye, GazeSample};

/// Classify the Pimax/StarVR EyeChip serial without opening the device. StarVR
/// serials use the `VRS` family; treating them as the old VR4 fallback makes the
/// WinUSB adapter try to open a service-owned EyeChip and fail with access denied.
pub fn auto_device_key_from_serial(serial: Option<&str>) -> &'static str {
    match serial.map(str::trim).filter(|serial| !serial.is_empty()) {
        Some(serial) if serial.to_ascii_uppercase().starts_with("XR5") => "pimax_xr5",
        Some(serial) if serial.to_ascii_uppercase().starts_with("VRS") => "starvr",
        _ => "pimax_vr4",
    }
}

/// The normal Hotmirror build still recognizes XR5 serials so it can refuse the
/// route explicitly instead of accidentally treating an angled XR5 as a VR4.
pub const XR5_PAUSED_MESSAGE: &str =
    "Dream Air / XR5 support is separated from the normal build and is currently paused";

/// Called by an adapter for each decoded camera frame: `(eye, width, height, bytes)`
/// where `bytes` is `width*height` grayscale, row-major. Dimensions travel WITH the
/// frame so the resolution varies per HMD (VR4/StarVR 200x200, Varjo higher) without
/// any hardcoding downstream.
pub type FrameFn = Box<dyn FnMut(Eye, u32, u32, &[u8]) + Send>;
/// Called by an adapter for each gaze/advanced-data sample (raw gaze sign).
pub type GazeFn = Box<dyn FnMut(GazeSample) + Send>;

/// The acquisition seam: produce (frame, gaze); everything downstream
/// (ML -> SRanipalState -> outputs) is HMD-agnostic.
// Adapters own configuration and worker handles; native sessions remain on their
// acquisition workers. Send permits shutdown/reload ownership off the UI thread.
pub trait HmdAdapter: Send {
    fn name(&self) -> &'static str;
    fn profile(&self) -> &DeviceProfile;
    /// Begin streaming; the adapter invokes the callbacks from its own threads.
    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> std::io::Result<()>;
    fn stop(&mut self);
    /// Whether this adapter needs the Tobii Platform Runtime stopped (the "CUSTOM"
    /// pre-flight) so it can claim the EyeChip directly. True for the DLL-free WinUSB
    /// path AND for the direct stream-engine path (`pimax_dll`), which both talk to a
    /// device the running Tobii runtime is holding. A real StarVR / service-routed
    /// path leaves it false so the service isn't stopped out from under it. Default:
    /// false.
    fn needs_eyechip_handoff(&self) -> bool {
        false
    }
    /// Live human-readable device status (for the UI's diagnostic line). Adapters
    /// that track state override this; the default is a static placeholder.
    fn status_arc(&self) -> Arc<Mutex<String>> {
        Arc::new(Mutex::new("n/a".into()))
    }
}

/// Construct the adapter for the configured `[hmd].device`. Everything
/// downstream (ML -> SRanipalState -> sinks) is identical across adapters; only
/// this acquisition seam differs. Unimplemented devices return a clear error
/// instead of silently falling back to VR4 — so the config never lies about what
/// actually runs.
#[cfg(all(windows, feature = "psvr2-only", not(feature = "xr5-only")))]
pub fn make_adapter(_cfg: &crate::config::Config) -> std::io::Result<Box<dyn HmdAdapter>> {
    Ok(Box::new(psvr2_adapter::Psvr2Adapter::new()))
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
        // `auto`: sniff the EyeChip serial (CM enumeration only, no device open) and
        // route VR4 vs XR5 — the ONLY difference is the ML preprocessing (VR4 frontal
        // vs XR5 angled); acquisition is byte-identical. Serial prefix `XR5` → XR5
        // (angled); anything else (e.g. `VRU02-…`) → the gated VR4 path (frontal).
        // If no Pimax EyeChip is found yet, default to VR4 (harmless — it just won't
        // connect until one appears). This sniffing is Pimax-only; StarVR/Varjo/VPE
        // keep their own explicit `device=` values below.
        "auto" => match usb::peek_serial() {
            Some(s) if auto_device_key_from_serial(Some(&s)) == "pimax_xr5" => {
                eprintln!("[auto] eyechip serial={s} -> XR5 (separate build paused)");
                Err(Error::new(ErrorKind::Unsupported, XR5_PAUSED_MESSAGE))
            }
            Some(s) if auto_device_key_from_serial(Some(&s)) == "starvr" => {
                eprintln!("[auto] eyechip serial={s} → StarVR One");
                Ok(Box::new(starvr_adapter::StarVrAdapter::new(cfg)))
            }
            Some(s) => {
                eprintln!("[auto] eyechip serial={s} → VR4 (frontal)");
                let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
                Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)))
            }
            None => {
                eprintln!("[auto] no eyechip found yet → defaulting to VR4");
                let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
                Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)))
            }
        },
        // `pimax_vr4`: DLL-free WinUSB acquisition, GATED on the user-supplied common
        // Tobii DLL (loaded as the connection authorization; data via WinUSB). Manual
        // override of the VR4 (frontal) path (`auto` picks this from the serial).
        "pimax_vr4" => {
            let dll = cfg.tobii_runtime_path().map(|p| p.to_string_lossy().into_owned());
            Ok(Box::new(vr4_adapter::Vr4Adapter::new_gated(dll)))
        }
        // `pimax_dll`: drive the Pimax EyeChip THROUGH the Tobii stream-engine DLL
        // (same generic adapter as StarVR — the DLL is the actual data path, not just
        // a gate). Use this once verified to stream on the Pimax hardware.
        "pimax_dll" => Ok(Box::new(
            // handoff=true: the patched stream engine accesses the Pimax EyeChip
            // directly, so the Tobii Platform Runtime must be stopped first.
            starvr_adapter::StarVrAdapter::with_profile(
                cfg, "Pimax (Tobii stream engine)", Eye::Left, true,
            ),
        )),
        "starvr" => Ok(Box::new(starvr_adapter::StarVrAdapter::new(cfg))),
        // `varjo`: native VarjoLib SDK eye cameras (re-enabled — the BSODs were RAM/RAM-disk,
        // not this). Gated on a VarjoLib.dll (auto-detected from Varjo Base, or
        // [assets].varjo_lib). Errors to `varjo_mjpeg` if the DLL is missing.
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
        // `varjo_mjpeg`: the zero-SDK fallback — consume the Varjo Eye Streamer's MJPEG feed.
        "varjo_mjpeg" => Ok(Box::new(varjo_adapter::VarjoAdapter::new(cfg))),
        // `psvr2`: consume the read-only native gaze + paired eye-camera preview
        // exposed by the user's installed PSVR2Toolkit SteamVR driver.
        "psvr2" => Ok(Box::new(psvr2_adapter::Psvr2Adapter::new())),
        // `xr5`: DLL-free WinUSB + native TTP, built on the VR4 adapter (same IS4
        // EyeChip, same GUID; only the "xr5" angled-optics ML route differs). NOT a
        // stream-engine DLL call — the DLL is only the distribution gate. If XR5 auth
        // differs on hardware, the documented fallback is a named-pipe + DLL relay
        // (see xr5_adapter.rs). Needs the Tobii Platform Runtime stopped (handoff).
        "pimax_xr5" => Err(Error::new(ErrorKind::Unsupported, XR5_PAUSED_MESSAGE)),
        "vpe" => Ok(Box::new(vpe_adapter::VpeAdapter::new())),
        other => Err(Error::new(
            ErrorKind::InvalidInput,
            format!("unknown [hmd].device '{other}' — use one of: auto, pimax_vr4, starvr, varjo, varjo_mjpeg, psvr2"),
        )),
    }
}

#[cfg(all(test, windows))]
mod adapter_alias_tests {
    use super::*;

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
    #[cfg(feature = "xr5-only")]
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
    #[cfg(feature = "xr5-only")]
    fn xr5_only_build_ignores_a_copied_non_xr5_device_setting() {
        let mut cfg = crate::config::Config::default();
        cfg.hmd.device = "starvr".into();
        let adapter = make_adapter(&cfg).unwrap();
        assert_eq!(adapter.name(), "pimax-xr5");
    }

    #[test]
    #[cfg(not(any(feature = "psvr2-only", feature = "xr5-only")))]
    fn normal_build_rejects_xr5_instead_of_misrouting_it() {
        for alias in ["xr5", "pimax_xr5", "pimax-xr5", "dream_air", "dream-air"] {
            let mut cfg = crate::config::Config::default();
            cfg.hmd.device = alias.into();
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
