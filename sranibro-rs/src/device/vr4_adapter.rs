//! Pimax VR4 adapter: DLL-free, service-free eye acquisition over WinUSB.
//!
//! Ties [`super::usb`] (WinUSB transport) to the TTP protocol: runs the EyeChip
//! handshake (control init -> channel upgrade -> HMAC-MD5 auth -> subscribe),
//! then streams, routing gaze (1289) and camera images (1291) to the callbacks.
//! Requires the platform service stopped: `net stop "Tobii VR4PIMAXP3B Platform Runtime"`.

#![cfg(windows)]

use std::ffi::{c_void, CString};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use windows_sys::Win32::Foundation::FreeLibrary;
use windows_sys::Win32::System::LibraryLoader::LoadLibraryA;

use super::auth::{auth_digest, AUTH_PAD};
use super::gaze::{decode_gaze_1289, decode_wearable_1285};
use super::ttp::{
    self, decode_value, encode_blob, encode_u32, encode_u32_vector, Value, PID_GAZE, PID_IMAGE,
    PID_WEARABLE,
};
use super::usb::UsbDevice;
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye};

// Region->eye mapping for the standard EyeChip unit (verified on a Crystal Super):
// region 0 = Right, region 1 = Left — see `region_to_eye` below. Units that ship
// with the cameras swapped are corrected at the pipeline level via `swap_eyes`.
pub struct Vr4Adapter {
    profile: DeviceProfile,
    /// Stable adapter identity for [`HmdAdapter::name`] (the trait wants a
    /// `&'static str`, so this is a fixed tag chosen at construction, NOT the
    /// user-facing `profile.name`). The XR5 delegate picks a different tag.
    tag: &'static str,
    /// Log prefix (`"[vr4]"` / `"[xr5]"`) so the same pump/handshake code emits
    /// device-appropriate bring-up lines.
    log_tag: &'static str,
    /// User-supplied common Tobii DLL path. When `require_dll`, the connection is
    /// gated on loading it — SRanibro ships inert and only opens the EyeChip once
    /// the user adds this (distribution stance; see config.tobii_dll).
    dll_path: Option<String>,
    require_dll: bool,
    /// Select the one canonical gaze stream for this hardware. Current VR4/XR5
    /// firmware uses 1289. Never merge both streams: their timing and L/R
    /// conventions are not interchangeable.
    wearable_gaze_source: bool,
    /// XR5-only source selection inside pid 1289. When true, shape the EyeChip's
    /// fused column-5 direction as both eyes instead of columns 3/4. Fixed for the
    /// lifetime of the stream to avoid source-switch jitter.
    combined_gaze_source: bool,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}

impl Default for Vr4Adapter {
    fn default() -> Self {
        Self::new()
    }
}

impl Vr4Adapter {
    /// The default Pimax VR4 / Crystal Super profile. Factored out so the XR5
    /// delegate can build its own profile without duplicating the shared fields.
    fn vr4_profile() -> DeviceProfile {
        DeviceProfile {
            // Display name; the EyeChip/transport is shared with the Crystal Super,
            // which is the unit this adapter serves. `ml_device` ("vr4") is the
            // preprocessing route and is intentionally separate.
            name: "Pimax Crystal Super".into(),
            ml_device: "vr4".into(),
            transport: "WinUSB · DLL-free".into(),
            streams: "Tobii TTP: gaze 1289 · adv 1285 · img 1291".into(),
            gaze_src: "gaze · pupil · openness (Tobii)".into(),
            ..DeviceProfile::default()
        }
    }

    /// DLL-free constructor for the developer raw-capture subcommands (vr4/mlcheck/
    /// capture). The product connection path uses [`Vr4Adapter::new_gated`] instead.
    pub fn new() -> Self {
        Self::build(
            Self::vr4_profile(),
            "pimax-vr4",
            "[vr4]",
            None,
            false,
            false,
            false,
        )
    }

    /// Product constructor: the connection REQUIRES the user-supplied Tobii DLL
    /// (`dll_path`). Without it the adapter refuses to open the EyeChip — so the
    /// distributed binary cannot connect until the user adds the DLL.
    pub fn new_gated(dll_path: Option<String>) -> Self {
        Self::build(
            Self::vr4_profile(),
            "pimax-vr4",
            "[vr4]",
            dll_path,
            true,
            false,
            false,
        )
    }

    /// Build the DLL-free WinUSB+TTP adapter with an arbitrary [`DeviceProfile`].
    /// This is the seam the XR5 adapter delegates through: the acquisition path
    /// (open → init → auth → subscribe → pump 1289/1285/1291) is byte-identical to
    /// VR4; only the profile (`ml_device`, name, transport strings) differs. `tag`
    /// is the static [`HmdAdapter::name`] and `log_tag` is the `eprintln!` prefix.
    pub fn with_profile(
        profile: DeviceProfile,
        tag: &'static str,
        log_tag: &'static str,
        dll_path: Option<String>,
        require_dll: bool,
        wearable_gaze_source: bool,
        combined_gaze_source: bool,
    ) -> Self {
        Self::build(
            profile,
            tag,
            log_tag,
            dll_path,
            require_dll,
            wearable_gaze_source,
            combined_gaze_source,
        )
    }

    fn build(
        profile: DeviceProfile,
        tag: &'static str,
        log_tag: &'static str,
        dll_path: Option<String>,
        require_dll: bool,
        wearable_gaze_source: bool,
        combined_gaze_source: bool,
    ) -> Self {
        Self {
            profile,
            tag,
            log_tag,
            dll_path,
            require_dll,
            wearable_gaze_source,
            combined_gaze_source,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }

    /// Live device status string (for the UI's diagnostic line).
    pub fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }

    #[cfg(test)]
    pub(crate) fn uses_wearable_gaze(&self) -> bool {
        self.wearable_gaze_source
    }

    #[cfg(test)]
    pub(crate) fn uses_combined_gaze(&self) -> bool {
        self.combined_gaze_source
    }
}

fn set_status(s: &Arc<Mutex<String>>, msg: impl Into<String>) {
    if let Ok(mut g) = s.lock() {
        *g = msg.into();
    }
}

/// RAII handle to the loaded Tobii DLL: FreeLibrary on drop (at the pump thread's
/// end). Created and dropped on the same thread, so it never crosses threads.
struct DllGuard(*mut c_void);
impl Drop for DllGuard {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.0) };
    }
}

/// Distribution gate: load the user-supplied Tobii DLL, returning a guard that frees
/// it when the connection ends. `None` (with a status message) means "no DLL -> do
/// not connect". The DLL is the user's licensed component; SRanibro bundles nothing.
fn load_required_dll(
    dll_path: &Option<String>,
    status: &Arc<Mutex<String>>,
    log_tag: &str,
) -> Option<DllGuard> {
    let path = match dll_path {
        Some(p) if !p.trim().is_empty() => p.clone(),
        _ => {
            set_status(
                status,
                "Tobii DLL required — set it in Settings, then reload",
            );
            eprintln!(
                "{log_tag} refusing to connect: no Tobii DLL configured ([assets].tobii_dll)"
            );
            return None;
        }
    };
    if !std::path::Path::new(&path).is_file() {
        set_status(status, format!("Tobii DLL not found: {path}"));
        return None;
    }
    let c = match CString::new(path.clone()) {
        Ok(c) => c,
        Err(_) => {
            set_status(status, "bad Tobii DLL path");
            return None;
        }
    };
    let hmod = unsafe { LoadLibraryA(c.as_ptr() as *const u8) };
    if hmod.is_null() {
        set_status(status, format!("Tobii DLL load failed: {path}"));
        eprintln!("{log_tag} LoadLibrary failed for {path}");
        return None;
    }
    eprintln!("{log_tag} Tobii DLL loaded (connection authorized): {path}");
    Some(DllGuard(hmod))
}

/// Subscribe-command payload for a stream id.
fn subscribe_payload(stream_id: u32) -> Vec<u8> {
    let mut p = encode_u32(stream_id);
    p.extend(encode_u32_vector(&[]));
    p
}

/// Run the full EyeChip bring-up handshake. `log_tag` (`"[vr4]"` / `"[xr5]"`)
/// prefixes the bring-up lines so the shared flow logs per-device.
///
/// TODO(hardware, XR5): this init → channel-upgrade → HMAC-MD5 auth → subscribe
/// sequence is confirmed on VR4/Crystal Super and on the XR5 EyeChip via a Python
/// bridge; if a real XR5 unit ever rejects the auth/subscribe here, the documented
/// fallback is a named-pipe + DLL relay (the Python `ttp_bridge_xr5.py` approach).
/// We do DLL-free FIRST because it reuses this whole VR4 stack and is far simpler.
fn handshake(dev: &mut UsbDevice, log_tag: &str) -> io::Result<()> {
    dev.init_device()?;
    eprintln!("{log_tag} init_device ok");

    // Channel upgrade (offered feature ids 65536..65544).
    let upgrade: Vec<u32> = (0..9).map(|i| 65536 + i).collect();
    let mid = dev.send(1000, &encode_u32_vector(&upgrade))?;
    dev.recv(mid, 1500)?;

    // Auth: request challenge, answer with HMAC-MD5(challenge).
    let mut req = encode_u32(1001);
    req.extend(encode_u32_vector(&[0]));
    let mid = dev.send(1900, &req)?;
    let mut authed = false;
    if let Some(resp) = dev.recv(mid, 1500)? {
        // payload @26: echoed u32 (1001), echoed u32 (0), then the challenge blob.
        let mut pos = 26;
        if let Some((_, p)) = decode_value(&resp, pos) {
            pos = p;
        }
        if let Some((_, p)) = decode_value(&resp, pos) {
            pos = p;
        }
        if let Some((Value::Blob(challenge), _)) = decode_value(&resp, pos) {
            let digest = auth_digest(&challenge);
            let mut rp = encode_u32(1001);
            rp.extend(encode_u32(0));
            rp.extend(encode_blob(&digest));
            rp.extend(encode_blob(&AUTH_PAD));
            let mid = dev.send(1911, &rp)?;
            let _ = dev.recv(mid, 1000)?;
            authed = true;
        }
    }
    eprintln!(
        "{log_tag} HMAC-MD5 auth {}",
        if authed {
            "answered (challenge/response sent)"
        } else {
            "SKIPPED — no challenge blob in response"
        }
    );

    // Subscribe gaze + wearable-advanced (pupil/openness) + image, then unlock.
    for sid in [PID_GAZE, PID_WEARABLE, PID_IMAGE] {
        let mid = dev.send(1220, &subscribe_payload(sid))?;
        let acked = dev.recv(mid, 1000)?.is_some();
        eprintln!(
            "{log_tag} subscribe stream {sid} {}",
            if acked { "ack" } else { "(no ack within 1s)" }
        );
    }
    let mid = dev.send(1915, &encode_u32(1001))?;
    let _ = dev.recv(mid, 1000)?;
    Ok(())
}

/// Region→eye mapping shared by VR4 and the XR5 delegate. The TTP image decoder
/// yields `region = 1 - (region_byte & 1)` (see `ttp::decode_image`), so this maps
/// region 1 → LEFT, region 0 (and unknown) → RIGHT — matching the Python XR5
/// bridge's `1-(byte&1)` (region 1 = LEFT). Same EyeChip, same mapping.
fn region_to_eye(region: i32) -> Eye {
    if region == 1 {
        Eye::Left
    } else {
        Eye::Right
    }
}

impl HmdAdapter for Vr4Adapter {
    fn name(&self) -> &'static str {
        self.tag
    }

    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, mut on_frame: FrameFn, mut on_gaze: GazeFn) -> io::Result<()> {
        let stop = self.stop.clone();
        let status = self.status.clone();
        let require_dll = self.require_dll;
        let dll_path = self.dll_path.clone();
        let log_tag = self.log_tag;
        let wearable_gaze_source = self.wearable_gaze_source;
        let combined_gaze_source = self.combined_gaze_source;
        let handle = thread::spawn(move || {
            // Distribution gate: the product path requires the user-supplied Tobii
            // DLL before any EyeChip access. Held for the whole session (freed on
            // exit). Dev raw-capture tools pass require_dll=false (DLL-free).
            let _dll_guard = if require_dll {
                match load_required_dll(&dll_path, &status, log_tag) {
                    Some(g) => Some(g),
                    None => return, // status set; no connection without the DLL
                }
            } else {
                None
            };
            set_status(&status, "opening EyeChip…");
            let mut dev = match UsbDevice::open() {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("{log_tag} open failed: {e}");
                    set_status(
                        &status,
                        "no EyeChip — connect the headset and stop the Tobii platform service",
                    );
                    return;
                }
            };
            eprintln!("{log_tag} opened EyeChip (serial={})", dev.serial);
            set_status(&status, "authenticating…");
            if let Err(e) = handshake(&mut dev, log_tag) {
                eprintln!("{log_tag} handshake failed: {e}");
                set_status(&status, format!("handshake failed: {e}"));
                return;
            }
            eprintln!("{log_tag} streaming");
            set_status(&status, "streaming");

            // Bring-up counters: log the first image's real W×H, the first few
            // region→eye routing decisions, and first gaze/wearable arrival — the
            // logs ARE the debugger for hardware bring-up (esp. the XR5 delegate).
            let mut frames_seen: u32 = 0;
            let mut gaze_seen = false;
            let mut gaze_diag_seen: u8 = 0;
            let mut gaze_packets: u32 = 0;
            let mut combined_valid_packets: u32 = 0;
            let mut combined_rescues_per_eye_loss: u32 = 0;
            let mut wearable_seen = false;
            while !stop.load(Ordering::Relaxed) {
                let msg = match dev.next_message(500) {
                    Ok(Some(m)) => m,
                    Ok(None) => continue,
                    Err(e) => {
                        eprintln!("{log_tag} read error: {e}");
                        set_status(&status, format!("stream error: {e}"));
                        break;
                    }
                };
                // Streaming frames carry msg_id 0; skip command responses.
                if msg.len() < 8 || u32::from_be_bytes(msg[4..8].try_into().unwrap()) != 0 {
                    continue;
                }
                match ttp::pid(&msg) {
                    PID_GAZE => {
                        if let Some(g) = decode_gaze_1289(&msg) {
                            if !gaze_seen {
                                gaze_seen = true;
                                eprintln!("{log_tag} first gaze sample (stream 1289) present");
                            }
                            if !wearable_gaze_source {
                                let per_eye_sample = g.to_xr5_gaze_sample();
                                let combined_sample = g.to_xr5_combined_gaze_sample();
                                let sample = if log_tag == "[xr5]" && combined_gaze_source {
                                    combined_sample
                                } else if log_tag == "[xr5]" {
                                    per_eye_sample
                                } else {
                                    g.to_gaze_sample()
                                };
                                if log_tag == "[xr5]" && gaze_diag_seen < 8 {
                                    eprintln!(
                                        "{log_tag} gaze diag #{gaze_diag_seen}: mode={} status={} L={:?} valid={} R={:?} valid={} C={:?} valid={} convergence_raw={}",
                                        if combined_gaze_source { "combined" } else { "per-eye" },
                                        g.status,
                                        g.l_gaze,
                                        per_eye_sample.left.gaze_valid,
                                        g.r_gaze,
                                        per_eye_sample.right.gaze_valid,
                                        g.combined,
                                        combined_sample.left.gaze_valid,
                                        g.convergence,
                                    );
                                    gaze_diag_seen += 1;
                                }
                                if log_tag == "[xr5]" {
                                    gaze_packets = gaze_packets.saturating_add(1);
                                    if combined_sample.left.gaze_valid {
                                        combined_valid_packets =
                                            combined_valid_packets.saturating_add(1);
                                        if !per_eye_sample.left.gaze_valid
                                            || !per_eye_sample.right.gaze_valid
                                        {
                                            combined_rescues_per_eye_loss =
                                                combined_rescues_per_eye_loss.saturating_add(1);
                                        }
                                    }
                                    if gaze_packets == 600 {
                                        eprintln!(
                                            "{log_tag} combined gaze audit (600 packets): valid={combined_valid_packets}, valid-while-per-eye-missing={combined_rescues_per_eye_loss}, active_mode={}",
                                            if combined_gaze_source { "combined" } else { "per-eye" },
                                        );
                                    }
                                }
                                on_gaze(sample);
                            }
                        }
                    }
                    PID_WEARABLE => {
                        if let Some(w) = decode_wearable_1285(&msg) {
                            if !wearable_seen {
                                wearable_seen = true;
                                eprintln!("{log_tag} first wearable/advanced sample (stream 1285) present");
                            }
                            if wearable_gaze_source {
                                // Hardware explicitly configured for wearable gaze:
                                // keep 1285 as the sole gaze source.
                                on_gaze(w.to_gaze_sample());
                            } else {
                                // VR4: 1289 is canonical; merge only pupil/openness
                                // from the duplicate wearable stream.
                                on_gaze(w.to_aux_sample());
                            }
                        }
                    }
                    PID_IMAGE => {
                        if let Some(img) = ttp::decode_image(&msg) {
                            let eye = region_to_eye(img.region);
                            if frames_seen == 0 {
                                eprintln!(
                                    "{log_tag} first image frame (pid 1291): {}x{} ({} bytes)",
                                    img.width,
                                    img.height,
                                    img.pixels.len()
                                );
                            }
                            if frames_seen < 4 {
                                eprintln!(
                                    "{log_tag} frame {frames_seen}: region {} -> {} eye",
                                    img.region,
                                    eye.as_str()
                                );
                            }
                            frames_seen = frames_seen.saturating_add(1);
                            on_frame(eye, img.width, img.height, &img.pixels);
                        }
                    }
                    _ => {}
                }
            }
            set_status(&status, "stopped");
            eprintln!("{log_tag} stopped");
        });
        self.thread = Some(handle);
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }

    fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }

    // DLL-free WinUSB path: needs the Tobii Platform Runtime stopped to claim the
    // EyeChip directly.
    fn needs_eyechip_handoff(&self) -> bool {
        true
    }
}
