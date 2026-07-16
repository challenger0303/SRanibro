//! Pimax XR5 adapter — DLL-FREE WinUSB + native TTP, built on the VR4 adapter.
//!
//! The XR5 eyechip is the SAME Tobii IS4 sensor on the SAME WinUSB device-interface
//! GUID `{85C0F97C-E2B1-422A-92A9-5F96072E79D8}` as the Pimax VR4 / Crystal Super
//! and StarVR One. Only the ML preprocessing angle differs (`ml_device = "xr5"` →
//! angled crop + flip). Everything about *acquisition* — opening the device by GUID,
//! the control-transfer init, the channel upgrade, the HMAC-MD5 auth, subscribing to
//! the TTP streams, and decoding images (pid 1291) / gaze (1289) / wearable (1285) —
//! is IDENTICAL to the VR4 path. A working Python bridge confirmed the XR5 image
//! stream is pid 1291, 200×200 grayscale blobs, region byte `region = 1 - (byte&1)`,
//! region 1 = LEFT, region 0 = RIGHT — the exact same decode as VR4.
//!
//! So this adapter is literally the VR4 adapter with a different [`DeviceProfile`].
//! It delegates to [`Vr4Adapter::with_profile`], which owns the whole DLL-free
//! WinUSB+TTP flow (see `vr4_adapter.rs`). We do NOT load the Tobii stream-engine
//! DLL for data — the DLL is only the distribution gate (the user's licensed
//! component), exactly as on the gated VR4 product path. NOTHING here is a stream
//! engine call.
//!
//! ## TODO(hardware) — the three unknowns, in priority order
//!  1. **Auth/subscribe parity (THE big one).** We assume the XR5 EyeChip's
//!     control-init + HMAC-MD5 challenge/response + subscribe sequence is byte-
//!     identical to VR4 (`vr4_adapter::handshake`). It is on the Python bridge; if a
//!     real XR5 unit REJECTS the auth or never acks the subscribe, the documented
//!     fallback is a **named-pipe + DLL relay** (the Python `ttp_bridge_xr5.py`
//!     approach) — a separate adapter that shells the frames through the DLL. We try
//!     DLL-free FIRST because it's far simpler and reuses this entire VR4 stack.
//!  2. **Resolution.** Assumed 200×200 like the IS4, but the XR5 primary optics MAY
//!     publish 400×400. We do NOT hardcode: `ttp::decode_image` reads the real blob
//!     size and the adapter logs the first frame's actual `W×H`. (Note: the current
//!     `decode_image` keys on a 40000-byte blob = 200×200; if XR5 turns out to be
//!     400×400, `decode_image` needs a second size branch — flagged in that fn.)
//!  3. **Region→eye mapping.** Assumed region 1 → LEFT, region 0 → RIGHT (matches the
//!     Python bridge and VR4's `region_to_eye`). Confirm on hardware from the logged
//!     first-few-frames region→eye lines.

#![cfg(windows)]

use super::vr4_adapter::Vr4Adapter;
use crate::config::GazeSource;
use crate::core::types::{DeviceProfile, Eye};

/// The Pimax XR5 acquisition adapter.
///
/// Thin newtype over [`Vr4Adapter`]: XR5 and VR4 share the entire DLL-free
/// WinUSB+TTP acquisition path and differ ONLY in the [`DeviceProfile`], so this
/// constructs a `Vr4Adapter` seeded with the XR5 profile and forwards the
/// [`HmdAdapter`](super::HmdAdapter) trait to it.
pub struct Xr5Adapter(Vr4Adapter);

impl Xr5Adapter {
    /// The XR5 device profile: same EyeChip/transport as VR4, but the angled-optics
    /// ML route (`ml_device = "xr5"` → crop 40% + flip) and XR5-specific display
    /// strings. Resolution defaults to 200×200 but the per-frame decode passes the
    /// REAL W×H through, so a 400×400 XR5 primary camera would be honored.
    fn profile(gaze_source: GazeSource) -> DeviceProfile {
        DeviceProfile {
            name: "Pimax XR5".into(),
            // Angled optics → crop 40% + flip preprocessing (see ml::preprocess).
            ml_device: "xr5".into(),
            // TODO(hardware): the XR5 primary optics MAY be 400×400; the frame decode
            // passes the real per-frame W×H through, so this is only the default.
            image_w: 200,
            image_h: 200,
            // region 1 → LEFT, region 0 → RIGHT (see vr4_adapter::region_to_eye); the
            // pair's "slot A" is the LEFT eye, matching the Python XR5 bridge.
            slot_a_eye: Eye::Left,
            transport: "WinUSB + native TTP (DLL-free)".into(),
            streams: "TTP image 1291 + wearable 1285".into(),
            gaze_src: match gaze_source {
                GazeSource::PerEye => "per-eye gaze · pupil · openness (Tobii)".into(),
                GazeSource::Combined => "combined gaze · pupil · openness (Tobii)".into(),
            },
        }
    }

    /// Product constructor. Like the gated VR4 path, the connection REQUIRES the
    /// user-supplied common Tobii DLL (`[assets].tobii_dll`) as the distribution gate
    /// — data still flows over WinUSB, DLL-free; the DLL is only the authorization to
    /// open the EyeChip. Without it configured, the adapter refuses to connect.
    pub fn new(cfg: &crate::config::Config) -> Self {
        // The option is XR5-only and always lives in the canonical XR5 bucket,
        // including when the active device selector is `auto`.
        let gaze_source = cfg.gaze_source_for("pimax_xr5");
        let dll_path = cfg
            .tobii_dll_path()
            .map(|p| p.to_string_lossy().into_owned());
        Self(Vr4Adapter::with_profile(
            Self::profile(gaze_source),
            "pimax-xr5",
            "[xr5]",
            dll_path,
            // require_dll = true: same distribution stance as the VR4 product path.
            true,
            // 1289 is the sole source. XR5 validates each eye's direction vector
            // because this firmware can emit usable vectors with packet status 0.
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

    fn start(&mut self, on_frame: super::FrameFn, on_gaze: super::GazeFn) -> std::io::Result<()> {
        self.0.start(on_frame, on_gaze)
    }

    fn stop(&mut self) {
        self.0.stop()
    }

    fn status_arc(&self) -> std::sync::Arc<std::sync::Mutex<String>> {
        self.0.status_arc()
    }

    /// XR5 needs the Tobii Platform Runtime stopped before we can claim the EyeChip
    /// over WinUSB — the Python launcher kills `platform_runtime_XR5EYECHIP` before
    /// connecting. Same handoff requirement as the DLL-free VR4 path.
    fn needs_eyechip_handoff(&self) -> bool {
        true
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
            p.transport.contains("DLL-free"),
            "DLL-free WinUSB+TTP transport"
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
        // DLL-free WinUSB path claims the EyeChip directly → runtime must be stopped.
        assert!(a.needs_eyechip_handoff());
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
}
