
#![cfg(any(windows, target_os = "linux"))]

#[cfg(windows)]
use std::ffi::{c_void, CString};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

#[cfg(windows)]
use windows_sys::Win32::Foundation::FreeLibrary;
#[cfg(windows)]
use windows_sys::Win32::System::LibraryLoader::LoadLibraryA;

use super::gaze::{decode_gaze_1289, decode_wearable_1285};
use super::ttp::{
    self, decode_value, encode_blob, encode_u32, encode_u32_vector, Value, PID_GAZE, PID_IMAGE,
    PID_WEARABLE,
};
use super::usb::{self, UsbDevice};
use super::vr4_gaze_source::Vr4GazeRouter;
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::config::Vr4GazeSource;
use crate::core::types::{DeviceProfile, Eye};

pub struct Vr4Adapter {
    profile: DeviceProfile,
    tag: &'static str,
    log_tag: &'static str,
    dll_path: Option<String>,
    require_dll: bool,
    wearable_gaze_source: bool,
    vr4_gaze_source: Option<Vr4GazeSource>,

    combined_gaze_source: bool,
    raw_gaze_callback: Option<GazeFn>,
    native_evidence_callback: Option<GazeFn>,
    #[cfg(feature = "xr5-only")]
    wearable_packet_callback: Option<Box<dyn FnMut(&[u8]) + Send>>,
    #[cfg(feature = "xr5-only")]
    gaze_packet_callback: Option<Box<dyn FnMut(&[u8]) + Send>>,
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
    #[cfg(feature = "xr5-only")]
    pub fn set_wearable_packet_callback(&mut self, callback: Box<dyn FnMut(&[u8]) + Send>) {
        self.wearable_packet_callback = Some(callback);
    }
    #[cfg(feature = "xr5-only")]
    pub fn set_gaze_packet_callback(&mut self, callback: Box<dyn FnMut(&[u8]) + Send>) {
        self.gaze_packet_callback = Some(callback);
    }
    fn vr4_profile() -> DeviceProfile {
        DeviceProfile {
            name: "Pimax Crystal Super".into(),
            ml_device: "vr4".into(),
            transport: "WinUSB · native TTP".into(),
            streams: "Tobii TTP: gaze 1289 · adv 1285 · img 1291".into(),
            gaze_src: "gaze · pupil · openness (Tobii)".into(),
            ..DeviceProfile::default()
        }
    }

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
        .with_gaze_source(Vr4GazeSource::default())
    }

    pub fn with_gaze_source(mut self, source: Vr4GazeSource) -> Self {
        if self.tag == "pimax-vr4" {
            self.vr4_gaze_source = Some(source);
            self.profile.gaze_src = format!("Tobii gaze {}", source.label());
        }
        self
    }

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
            vr4_gaze_source: None,
            raw_gaze_callback: None,
            native_evidence_callback: None,
            #[cfg(feature = "xr5-only")]
            wearable_packet_callback: None,
            #[cfg(feature = "xr5-only")]
            gaze_packet_callback: None,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }

    pub fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }

    #[cfg(test)]
    pub(crate) fn uses_wearable_gaze(&self) -> bool {
        self.vr4_gaze_source
            .map(|source| source == Vr4GazeSource::V1)
            .unwrap_or(self.wearable_gaze_source)
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

#[cfg(windows)]
struct DllGuard(*mut c_void);
#[cfg(windows)]
impl Drop for DllGuard {
    fn drop(&mut self) {
        unsafe { FreeLibrary(self.0) };
    }
}

#[cfg(windows)]
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
                "Tobii runtime unavailable — reinstall the official SRanibro build",
            );
            eprintln!("{log_tag} refusing to connect: Tobii runtime unavailable in this build");
            return None;
        }
    };
    if !std::path::Path::new(&path).is_file() {
        set_status(status, "Tobii runtime file not found");
        eprintln!("{log_tag} Tobii runtime file not found");
        return None;
    }
    let c = match CString::new(path.clone()) {
        Ok(c) => c,
        Err(_) => {
            set_status(status, "Tobii runtime path is invalid");
            return None;
        }
    };
    let hmod = unsafe { LoadLibraryA(c.as_ptr() as *const u8) };
    if hmod.is_null() {
        set_status(status, "Tobii runtime load failed");
        eprintln!("{log_tag} Tobii runtime load failed");
        return None;
    }
    Some(DllGuard(hmod))
}

#[cfg(target_os = "linux")]
struct DllGuard;
#[cfg(target_os = "linux")]
fn load_required_dll(_: &Option<String>, status: &Arc<Mutex<String>>, _: &str) -> Option<DllGuard> {
    set_status(status, "Windows runtime loading is unavailable on Linux");
    None
}


fn subscribe_payload(stream_id: u32) -> Vec<u8> {
    let mut p = encode_u32(stream_id);
    p.extend(encode_u32_vector(&[]));
    p
}

fn handshake(dev: &mut UsbDevice, log_tag: &str, stop: &AtomicBool) -> io::Result<()> {
    let _ = (dev, log_tag, stop);
    Err(super::runtime_access::unavailable())
}

fn wait_for_reopen_quiet_period(
    stop: &AtomicBool,
    status: &Arc<Mutex<String>>,
    log_tag: &str,
) -> bool {
    let initial = usb::reopen_quiet_remaining();
    if initial.is_zero() {
        return true;
    }

    eprintln!(
        "{log_tag} waiting {:.1}s for EyeChip firmware to release the previous session",
        initial.as_secs_f32()
    );
    loop {
        if stop.load(Ordering::Relaxed) {
            return false;
        }
        let remaining = usb::reopen_quiet_remaining();
        if remaining.is_zero() {
            return true;
        }
        set_status(
            status,
            format!(
                "waiting for EyeChip release... {:.1}s",
                remaining.as_secs_f32()
            ),
        );
        thread::sleep(remaining.min(Duration::from_millis(50)));
    }
}

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

    fn set_raw_gaze_callback(&mut self, callback: GazeFn) -> bool {
        if self.thread.is_some() {
            return false;
        }
        self.raw_gaze_callback = Some(callback);
        true
    }

    fn set_native_evidence_callback(&mut self, callback: GazeFn) -> bool {
        if self.tag != "pimax-xr5" || self.thread.is_some() {
            return false;
        }
        self.native_evidence_callback = Some(callback);
        true
    }

    fn start(&mut self, mut on_frame: FrameFn, mut on_gaze: GazeFn) -> io::Result<()> {
        if !super::runtime_access::direct_access_available() {
            set_status(&self.status, "Direct acquisition excluded from source build");
            return Err(super::runtime_access::unavailable());
        }

        let stop = self.stop.clone();
        let status = self.status.clone();
        let require_dll = self.require_dll;
        let dll_path = self.dll_path.clone();
        let log_tag = self.log_tag;
        let wearable_gaze_source = self.wearable_gaze_source;
        let vr4_gaze_source = self.vr4_gaze_source;
        let combined_gaze_source = self.combined_gaze_source;
        let mut on_raw_gaze = self.raw_gaze_callback.take();
        let mut on_native_evidence = self.native_evidence_callback.take();
        #[cfg(feature = "xr5-only")]
        let mut on_wearable_packet = self.wearable_packet_callback.take();
        #[cfg(feature = "xr5-only")]
        let mut on_gaze_packet = self.gaze_packet_callback.take();
        let handle = thread::spawn(move || {
            let mut vr4_router = vr4_gaze_source.map(Vr4GazeRouter::new);
            if let Some(source) = vr4_gaze_source {
                eprintln!(
                    "{log_tag} gaze source={} stream={} (avatar + native OSC)",
                    source.label(),
                    source.stream_id()
                );
            }
            let _dll_guard = if require_dll {
                match load_required_dll(&dll_path, &status, log_tag) {
                    Some(g) => Some(g),
                    None => return,
                }
            } else {
                None
            };
            const START_ATTEMPTS: u8 = 4;
            const RETRY_STEPS: u8 = 10;
            let mut ready = None;
            for attempt in 1..=START_ATTEMPTS {
                if stop.load(Ordering::Relaxed) {
                    return;
                }
                if !wait_for_reopen_quiet_period(&stop, &status, log_tag) {
                    return;
                }
                set_status(
                    &status,
                    format!("opening EyeChip… attempt {attempt}/{START_ATTEMPTS}"),
                );
                match UsbDevice::open() {
                    Ok(mut dev) => {
                        eprintln!(
                            "{log_tag} opened EyeChip (serial={}, attempt {attempt}/{START_ATTEMPTS})",
                            dev.serial
                        );
                        set_status(
                            &status,
                            format!("authenticating… attempt {attempt}/{START_ATTEMPTS}"),
                        );
                        match handshake(&mut dev, log_tag, &stop) {
                            Ok(()) => {
                                ready = Some(dev);
                                break;
                            }
                            Err(error) => {
                                if stop.load(Ordering::Relaxed) {
                                    set_status(&status, "stopped");
                                    return;
                                }
                                eprintln!(
                                    "{log_tag} handshake attempt {attempt}/{START_ATTEMPTS} failed: {error}"
                                );
                                set_status(
                                    &status,
                                    format!("handshake retry {attempt}/{START_ATTEMPTS}: {error}"),
                                );
                            }
                        }
                    }
                    Err(error) => {
                        eprintln!(
                            "{log_tag} open attempt {attempt}/{START_ATTEMPTS} failed: {error}"
                        );
                        set_status(
                            &status,
                            format!("EyeChip open retry {attempt}/{START_ATTEMPTS}: {error}"),
                        );
                    }
                }
                if attempt < START_ATTEMPTS {
                    for _ in 0..RETRY_STEPS {
                        if stop.load(Ordering::Relaxed) {
                            return;
                        }
                        thread::sleep(Duration::from_millis(50));
                    }
                }
            }
            let Some(mut dev) = ready else {
                set_status(
                    &status,
                    format!("EyeChip connection failed after {START_ATTEMPTS} attempts"),
                );
                eprintln!("{log_tag} connection failed after {START_ATTEMPTS} bring-up attempts");
                return;
            };
            eprintln!("{log_tag} streaming");
            set_status(&status, "streaming");

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
                if msg.len() < 8 || u32::from_be_bytes(msg[4..8].try_into().unwrap()) != 0 {
                    continue;
                }
                match ttp::pid(&msg) {
                    PID_GAZE => {
                        #[cfg(feature = "xr5-only")]
                        if let Some(callback) = &mut on_gaze_packet {
                            callback(&msg);
                        }
                        if let Some(g) = decode_gaze_1289(&msg) {
                            if !gaze_seen {
                                gaze_seen = true;
                                eprintln!("{log_tag} first gaze sample (stream 1289) present");
                            }
                            if let Some(router) = &vr4_router {
                                if let Some(sample) = router.foveated(&g) {
                                    if let Some(callback) = &mut on_raw_gaze {
                                        callback(sample);
                                    }
                                    on_gaze(sample);
                                }
                            } else if !wearable_gaze_source {
                                let per_eye_sample = g.to_xr5_gaze_sample();
                                if let Some(callback) = &mut on_raw_gaze {
                                    callback(per_eye_sample);
                                }
                                let combined_sample = g.to_xr5_combined_gaze_sample();
                                let sample = if log_tag == "[xr5]" && combined_gaze_source {
                                    combined_sample
                                } else {
                                    per_eye_sample
                                };
                                if gaze_diag_seen < 8 {
                                    eprintln!(
                                        "{log_tag} gaze diag #{gaze_diag_seen}: mode={} status={} L={:?} valid={} R={:?} valid={} C={:?} valid={} convergence_raw={}",
                                        if combined_gaze_source { "combined" } else { "per-eye" },
                                        g.status,
                                        per_eye_sample.left.gaze,
                                        per_eye_sample.left.gaze_valid,
                                        per_eye_sample.right.gaze,
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
                        #[cfg(feature = "xr5-only")]
                        if let Some(callback) = &mut on_wearable_packet {
                            callback(&msg);
                        }
                        if let Some(w) = decode_wearable_1285(&msg) {
                            if let Some(callback) = &mut on_native_evidence {
                                callback(w.to_gaze_sample());
                            }
                            if !wearable_seen {
                                wearable_seen = true;
                                eprintln!("{log_tag} first wearable/advanced sample (stream 1285) present");
                            }
                            if let Some(router) = &mut vr4_router {
                                let (raw, tracking) = router.wearable(&w);
                                if let Some(callback) = &mut on_raw_gaze {
                                    callback(raw);
                                }
                                on_gaze(tracking);
                            } else if wearable_gaze_source {
                                if let Some(callback) = &mut on_raw_gaze {
                                    callback(w.to_gaze_sample());
                                }
                                on_gaze(w.to_gaze_sample());
                            } else {
                                if let Some(callback) = &mut on_raw_gaze {
                                    callback(w.to_aux_sample());
                                }
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

    fn needs_eyechip_handoff(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vr4_adapter_routes_only_stream_1289_gaze_for_pipeline_mapping() {
        let adapter = Vr4Adapter::new();
        assert!(
            !adapter.uses_wearable_gaze(),
            "1285 stays auxiliary pupil/openness data"
        );
        assert!(!adapter.uses_combined_gaze());
    }

    #[test]
    fn vr4_product_defaults_v1_and_can_select_v2_without_combined_gaze() {
        let adapter = Vr4Adapter::new_gated(None);
        assert!(adapter.uses_wearable_gaze());
        assert!(!adapter.uses_combined_gaze());
        let adapter = adapter.with_gaze_source(Vr4GazeSource::V2);
        assert!(!adapter.uses_wearable_gaze());
        assert!(!adapter.uses_combined_gaze());
    }

    #[test]
    fn vr4_source_choice_does_not_change_the_xr5_delegate() {
        let adapter = Vr4Adapter::with_profile(
            DeviceProfile::default(),
            "pimax-xr5",
            "[xr5]",
            None,
            false,
            false,
            true,
        )
        .with_gaze_source(Vr4GazeSource::V1);
        assert_eq!(adapter.vr4_gaze_source, None);
        assert!(!adapter.uses_wearable_gaze());
        assert!(adapter.uses_combined_gaze());
    }
}
