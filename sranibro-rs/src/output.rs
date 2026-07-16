//! Output sinks: where post-processed [`EyeResult`]s go (avatar / debug / etc.).
//!
//! The pipeline feeds every sink each frame. `DebugSink` prints throttled state;
//! `BrokenEyeSink` is a TCP server speaking the BrokenEye protocol so
//! VRCFaceTracking's TobiiAdvanced module connects and consumes our eye data.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::core::types::{Eye, EyeResult};

/// A consumer of post-processed per-eye results. Sinks run on the emit thread.
pub trait OutputSink: Send {
    fn name(&self) -> &str;
    fn on_eye(&mut self, result: &EyeResult);

    /// Publish one coherent stereo frame. Shared-state sinks can override this so a
    /// reader never observes a new left eye together with a stale right eye.
    fn on_frame(&mut self, results: &[EyeResult; 2]) {
        for result in results {
            self.on_eye(result);
        }
    }
}

/// Prints throttled left-eye state — for end-to-end verification.
pub struct DebugSink {
    every: u64,
    count: u64,
}

impl DebugSink {
    /// `every` = print once per this many left-eye frames (e.g. 60 ≈ 0.5s @120Hz).
    pub fn new(every: u64) -> Self {
        Self {
            every: every.max(1),
            count: 0,
        }
    }
}

impl OutputSink for DebugSink {
    fn name(&self) -> &str {
        "debug"
    }

    fn on_eye(&mut self, r: &EyeResult) {
        if r.eye != Eye::Left {
            return; // throttle on the left eye only
        }
        self.count += 1;
        if self.count % self.every == 0 {
            println!(
                "[debug] L open={:.2} wide={:.2} sqz={:.2} blink={:<5} gaze=({:+.2},{:+.2},{:+.2}) gv={}",
                r.openness, r.wide, r.squeeze, r.blink,
                r.gaze[0], r.gaze[1], r.gaze[2], r.gaze_valid,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// BrokenEye TCP sink (VRCFaceTracking compatibility)
// ---------------------------------------------------------------------------

/// Live status of the BrokenEye server, for the UI's OUTPUT node detail.
#[derive(Default)]
pub struct BrokenEyeStatus {
    pub port: u16,
    pub clients: AtomicUsize,
    pub frames: AtomicU64,
    /// Live filter window sent to the VRCFT module in every JSON frame.
    pub filter_samples: AtomicU32,
}

/// One eye's already-clamped values, ready to serialize (BrokenEye convention:
/// gaze X is negated, openness 0 on blink, everything clamped).
#[derive(Clone, Copy)]
struct EyeJson {
    gaze: [f32; 2],
    gaze_valid: bool,
    pupil_mm: f32,
    pupil_valid: bool,
    openness: f32,
    openness_valid: bool,
    wide: f32,
    squeeze: f32,
    /// Signed per-eye eyebrow expression: +1 raised, -1 lowered.
    brow: f32,
    brow_valid: bool,
    frown: f32,
}

impl Default for EyeJson {
    fn default() -> Self {
        Self {
            gaze: [0.0, 0.0],
            gaze_valid: false,
            pupil_mm: 0.0,
            pupil_valid: false,
            openness: 0.0,
            openness_valid: false,
            wide: 0.0,
            squeeze: 0.0,
            brow: 0.0,
            brow_valid: false,
            frown: 0.0,
        }
    }
}

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.max(lo).min(hi)
}

fn append_eye(buf: &mut String, e: &EyeJson) {
    use std::fmt::Write;
    let _ = write!(
        buf,
        "{{\"gaze_direction_is_valid\":{},\"gaze_direction\":[{:.5},{:.5}],\
         \"pupil_diameter_is_valid\":{},\"pupil_diameter_mm\":{:.4},\
         \"openness_is_valid\":{},\"openness\":{:.4},\
         \"wide\":{:.4},\"squeeze\":{:.4},\
         \"brow_is_valid\":{},\"brow\":{:.4},\"frown\":{:.4}}}",
        e.gaze_valid,
        e.gaze[0],
        e.gaze[1],
        e.pupil_valid,
        e.pupil_mm,
        e.openness_valid,
        e.openness,
        e.wide,
        e.squeeze,
        e.brow_valid,
        e.brow,
        e.frown,
    );
}

fn eye_json(r: &EyeResult) -> EyeJson {
    EyeJson {
        gaze: [clamp(-r.gaze[0], -1.0, 1.0), clamp(r.gaze[1], -1.0, 1.0)],
        gaze_valid: r.gaze_valid && (!r.blink || r.gaze_yoked),
        pupil_mm: r.pupil_mm.max(0.0),
        pupil_valid: r.pupil_valid,
        openness: clamp(if r.blink { 0.0 } else { r.openness }, 0.0, 1.0),
        openness_valid: r.openness_valid || r.blink,
        wide: clamp(r.wide, 0.0, 1.0),
        squeeze: clamp(r.squeeze, 0.0, 1.0),
        brow: clamp(r.brow, -1.0, 1.0),
        brow_valid: r.brow_valid,
        frown: clamp(r.frown, 0.0, 1.0),
    }
}

#[cfg(test)]
mod broken_eye_json_tests {
    use super::*;

    #[test]
    fn extended_frame_serializes_signed_brow_and_validity() {
        let mut json = String::new();
        append_eye(
            &mut json,
            &EyeJson {
                brow: -0.625,
                brow_valid: true,
                ..EyeJson::default()
            },
        );
        assert!(json.contains("\"brow_is_valid\":true"), "{json}");
        assert!(json.contains("\"brow\":-0.6250"), "{json}");
    }

    #[test]
    fn brokeneye_stereo_frame_publishes_both_wide_values_together() {
        let mut sink = BrokenEyeSink::new(0, 0).unwrap();
        let mut left = EyeResult::new(Eye::Left);
        let mut right = EyeResult::new(Eye::Right);
        left.wide = 0.75;
        right.wide = 0.75;

        sink.on_frame(&[left, right]);

        let frame = *sink.data.lock().unwrap();
        assert_eq!(frame[0].wide, 0.75);
        assert_eq!(frame[1].wide, 0.75);
    }
}

/// BrokenEye-compatible TCP server. VRCFT connects to `port` (default 5555),
/// sends a 1-byte mode (0x00 = JSON), then receives `[0x00][u32 LE len][JSON]`
/// frames at 120Hz. Image modes (0x01-0x03) are accepted but not yet served.
pub struct BrokenEyeSink {
    data: Arc<Mutex<[EyeJson; 2]>>,
    status: Arc<BrokenEyeStatus>,
    stop: Arc<AtomicBool>,
    /// Accept-loop handle, joined on Drop so the `TcpListener` (and thus the port)
    /// is released BEFORE Drop returns — required for the UI's live "Apply & reload"
    /// to rebind the same port without a race.
    accept: Option<std::thread::JoinHandle<()>>,
}

const MODE_JSON: u8 = 0x00;

impl BrokenEyeSink {
    pub fn new(port: u16, filter_samples: u8) -> std::io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        listener.set_nonblocking(true)?;
        let data = Arc::new(Mutex::new([EyeJson::default(); 2]));
        let status = Arc::new(BrokenEyeStatus {
            port,
            filter_samples: AtomicU32::new(u32::from(filter_samples)),
            ..Default::default()
        });
        let stop = Arc::new(AtomicBool::new(false));

        let (a_data, a_status, a_stop) = (data.clone(), status.clone(), stop.clone());
        let accept = thread::Builder::new()
            .name("brokeneye-accept".into())
            .spawn(move || {
                while !a_stop.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _addr)) => {
                            spawn_client(stream, a_data.clone(), a_status.clone(), a_stop.clone());
                        }
                        Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(50));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(200)),
                    }
                }
                // listener drops here (loop exited) -> port freed.
            })?;
        eprintln!("[vrcft] VRCFT server listening on 0.0.0.0:{port}");
        Ok(Self {
            data,
            status,
            stop,
            accept: Some(accept),
        })
    }

    pub fn status(&self) -> Arc<BrokenEyeStatus> {
        self.status.clone()
    }
}

impl Drop for BrokenEyeSink {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Wait for the accept loop to observe `stop` and drop the listener, so the
        // port is free the instant Drop returns (the loop polls every <=50ms).
        if let Some(h) = self.accept.take() {
            let _ = h.join();
        }
    }
}

/// Read the 1-byte mode, then (for JSON) stream frames at 120Hz until the peer
/// drops or the server stops.
fn spawn_client(
    mut stream: TcpStream,
    data: Arc<Mutex<[EyeJson; 2]>>,
    status: Arc<BrokenEyeStatus>,
    stop: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        stream.set_read_timeout(Some(Duration::from_secs(3))).ok();
        let mut mode = [0u8; 1];
        if stream.read_exact(&mut mode).is_err() {
            return;
        }
        if mode[0] != MODE_JSON {
            // Image modes (0x01-0x03) not served yet; close politely.
            return;
        }
        stream
            .set_write_timeout(Some(Duration::from_millis(500)))
            .ok();
        status.clients.fetch_add(1, Ordering::Relaxed);
        let peer = stream
            .peer_addr()
            .map(|a| a.to_string())
            .unwrap_or_default();
        eprintln!("[vrcft] VRCFT client connected: {peer}");

        let mut json = String::with_capacity(512);
        while !stop.load(Ordering::Relaxed) {
            let [l, r] = *data.lock().unwrap();
            json.clear();
            use std::fmt::Write as _;
            let samples = status.filter_samples.load(Ordering::Relaxed).min(30);
            let _ = write!(json, "{{\"noise_filter_samples\":{samples},\"left\":");
            append_eye(&mut json, &l);
            json.push_str(",\"right\":");
            append_eye(&mut json, &r);
            json.push('}');

            let mut frame = Vec::with_capacity(5 + json.len());
            frame.push(MODE_JSON);
            frame.extend_from_slice(&(json.len() as u32).to_le_bytes());
            frame.extend_from_slice(json.as_bytes());
            if stream.write_all(&frame).is_err() {
                break;
            }
            status.frames.fetch_add(1, Ordering::Relaxed);
            thread::sleep(Duration::from_micros(8333));
        }
        status.clients.fetch_sub(1, Ordering::Relaxed);
        eprintln!("[vrcft] VRCFT client disconnected: {peer}");
    });
}

impl OutputSink for BrokenEyeSink {
    fn name(&self) -> &str {
        "brokeneye"
    }

    fn on_eye(&mut self, r: &EyeResult) {
        let e = EyeJson {
            // Single canonical gaze sign (decoders emit RAW). BrokenEye/VRCFT
            // convention = X mirrored, Y as-is -> net vs the chip's raw vector is
            // [-raw_x, +raw_y]. Matches brokeneye_compat.py and the live avatar's
            // vertical; L/R worth one on-avatar confirm.
            gaze: [clamp(-r.gaze[0], -1.0, 1.0), clamp(r.gaze[1], -1.0, 1.0)],
            // Suppress gaze while blinking so a garbage mid-blink gaze doesn't leak —
            // EXCEPT a yoked gaze (mirrored from the open eye), which IS good and must
            // still drive the squinting eye so it follows instead of freezing.
            gaze_valid: r.gaze_valid && (!r.blink || r.gaze_yoked),
            pupil_mm: r.pupil_mm.max(0.0),
            pupil_valid: r.pupil_valid,
            openness: clamp(if r.blink { 0.0 } else { r.openness }, 0.0, 1.0),
            openness_valid: r.openness_valid || r.blink,
            wide: clamp(r.wide, 0.0, 1.0),
            squeeze: clamp(r.squeeze, 0.0, 1.0),
            brow: clamp(r.brow, -1.0, 1.0),
            brow_valid: r.brow_valid,
            frown: clamp(r.frown, 0.0, 1.0),
        };
        if let Ok(mut d) = self.data.lock() {
            d[r.eye.idx()] = e;
        }
    }

    fn on_frame(&mut self, results: &[EyeResult; 2]) {
        let mut frame = [EyeJson::default(); 2];
        for result in results {
            frame[result.eye.idx()] = eye_json(result);
        }
        // Publish the stereo pair with one assignment under one lock. The TCP writer
        // sees either the previous complete frame or the next complete frame.
        if let Ok(mut data) = self.data.lock() {
            *data = frame;
        }
    }
}

// ---------------------------------------------------------------------------
// OSC sink (VRChat-direct — the "no VRCFT" wing of the hybrid output)
// ---------------------------------------------------------------------------

/// Sends gaze + eye shapes straight to VRChat over OSC (UnifiedExpressions-style
/// `/avatar/parameters/Eye*`), so SRanibro can drive an avatar WITHOUT VRCFT.
/// Self-contained OSC encoder (no deps). Uses the SAME canonical gaze sign as
/// [`BrokenEyeSink`] (X mirrored, Y as-is) so both outputs agree. Faithful port
/// of the reference `sranibro/outputs/osc.py`.
pub struct OscSink {
    sock: std::net::UdpSocket,
    host: String,
    port: u16,
    prefix: String,
    /// Full eye/gaze OSC when true; eyebrow-only FT/v2 sender when false.
    eyes_enabled: bool,
    latest: [[f32; 2]; 2],
    have: [bool; 2],
    /// Per-eye signed brow [-1,1] + presence, for the combined BrowUp/Down averages.
    brow: [f32; 2],
    brow_have: [bool; 2],
    brow_active: [bool; 2],
}

impl OscSink {
    /// Bind an ephemeral UDP socket; datagrams are sent to `host:port`.
    pub fn new(
        host: impl Into<String>,
        port: u16,
        prefix: impl Into<String>,
    ) -> std::io::Result<Self> {
        Self::new_mode(host, port, prefix, true)
    }

    /// VRCFT remains responsible for eyes/gaze; send only the eight eyebrow
    /// parameters used by the original `vr_eyebrow` application.
    pub fn new_brow_only(
        host: impl Into<String>,
        port: u16,
        prefix: impl Into<String>,
    ) -> std::io::Result<Self> {
        Self::new_mode(host, port, prefix, false)
    }

    fn new_mode(
        host: impl Into<String>,
        port: u16,
        prefix: impl Into<String>,
        eyes_enabled: bool,
    ) -> std::io::Result<Self> {
        let sock = std::net::UdpSocket::bind(("0.0.0.0", 0))?;
        let (host, prefix) = (host.into(), prefix.into());
        let family = if eyes_enabled {
            "Eye* + FT/v2/Brow*"
        } else {
            "FT/v2/Brow* only"
        };
        eprintln!("[osc] -> {host}:{port} {prefix}/{family}");
        Ok(Self {
            sock,
            host,
            port,
            prefix,
            eyes_enabled,
            latest: [[0.0; 2]; 2],
            have: [false; 2],
            brow: [0.0; 2],
            brow_have: [false; 2],
            brow_active: [false; 2],
        })
    }

    fn send(&self, name: &str, value: f32) {
        let msg = osc_message(&format!("{}/{}", self.prefix, name), value);
        let _ = self.sock.send_to(&msg, (self.host.as_str(), self.port));
    }
}

/// Encode a single-float OSC message: address + ",f" type tag + big-endian f32,
/// each string null-terminated and padded to a 4-byte boundary.
fn osc_message(address: &str, value: f32) -> Vec<u8> {
    fn padded(bytes: &[u8]) -> Vec<u8> {
        let pad = 4 - (bytes.len() % 4); // 1..=4: always at least one null terminator
        let mut v = Vec::with_capacity(bytes.len() + pad);
        v.extend_from_slice(bytes);
        v.extend(std::iter::repeat(0u8).take(pad));
        v
    }
    let mut msg = padded(address.as_bytes());
    msg.extend(padded(b",f"));
    msg.extend_from_slice(&value.to_be_bytes());
    msg
}

impl OutputSink for OscSink {
    fn name(&self) -> &str {
        "osc"
    }

    fn on_eye(&mut self, r: &EyeResult) {
        let side = match r.eye {
            Eye::Left => "Left",
            Eye::Right => "Right",
        };
        let gx = clamp(-r.gaze[0], -1.0, 1.0);
        let gy = clamp(r.gaze[1], -1.0, 1.0);
        if self.eyes_enabled {
            self.send(
                &format!("EyeLid{side}"),
                clamp(if r.blink { 0.0 } else { r.openness }, 0.0, 1.0),
            );
            self.send(&format!("EyeWide{side}"), clamp(r.wide, 0.0, 1.0));
            self.send(&format!("EyeSquint{side}"), clamp(r.squeeze, 0.0, 1.0));

            // Canonical gaze sign, matching BrokenEyeSink: X mirrored, Y as-is.
            // Hold gaze during a blink, but yoked gaze from the open eye is usable.
            if r.gaze_valid && (!r.blink || r.gaze_yoked) {
                self.send(&format!("Eye{side}X"), gx);
                self.send(&format!("Eye{side}Y"), gy);
            }
            if r.pupil_valid {
                const PUPIL_MIN_MM: f32 = 2.0;
                const PUPIL_MAX_MM: f32 = 7.0;
                let t =
                    ((r.pupil_mm - PUPIL_MIN_MM) / (PUPIL_MAX_MM - PUPIL_MIN_MM)).clamp(0.0, 1.0);
                self.send("PupilDilation", t);
            }
        }

        // Eyebrow (FT v2). Signed BrowExpression + split Up/Down per eye; the combined
        // BrowUp/BrowDown averages are emitted once per fresh L+R brow pair. Only sent
        // when a brow model is producing values, so we never force a 0 over another source.
        if r.brow_valid {
            let b = clamp(r.brow, -1.0, 1.0);
            self.send(&format!("FT/v2/BrowExpression{side}"), b);
            self.send(&format!("FT/v2/BrowUp{side}"), b.max(0.0));
            self.send(&format!("FT/v2/BrowDown{side}"), (-b).max(0.0));
            let bi = r.eye.idx();
            self.brow_active[bi] = true;
            self.brow[bi] = b;
            self.brow_have[bi] = true;
            if self.brow_have[0] && self.brow_have[1] {
                // vr_eyebrow combines the signed values first. Opposite
                // unilateral expressions therefore cancel in the combined pair.
                let avg = 0.5 * (self.brow[0] + self.brow[1]);
                let up = avg.max(0.0);
                let dn = (-avg).max(0.0);
                self.send("FT/v2/BrowUp", up);
                self.send("FT/v2/BrowDown", dn);
                self.brow_have = [false, false];
            }
        } else {
            // Clear values once when eyebrow tracking is turned off. Without this edge,
            // direct OSC avatars would keep the last non-zero brow pose indefinitely.
            let bi = r.eye.idx();
            if self.brow_active[bi] {
                self.send(&format!("FT/v2/BrowExpression{side}"), 0.0);
                self.send(&format!("FT/v2/BrowUp{side}"), 0.0);
                self.send(&format!("FT/v2/BrowDown{side}"), 0.0);
                self.brow_active[bi] = false;
                self.brow[bi] = 0.0;
                self.brow_have[bi] = true;
                if self.brow_have[0] && self.brow_have[1] {
                    self.send("FT/v2/BrowUp", 0.0);
                    self.send("FT/v2/BrowDown", 0.0);
                    self.brow_have = [false, false];
                }
            }
        }

        if self.eyes_enabled {
            let i = r.eye.idx();
            self.latest[i] = [gx, gy];
            self.have[i] = true;
            // Emit combined gaze once per FRESH L+R pair, then require a new pair.
            // Previously it re-sent every call, mixing the current eye with a stale
            // opposite eye and introducing avoidable gaze jitter.
            if self.have[0] && self.have[1] {
                self.send("EyeX", 0.5 * (self.latest[0][0] + self.latest[1][0]));
                self.send("EyeY", 0.5 * (self.latest[0][1] + self.latest[1][1]));
                self.have = [false, false];
            }
        }
    }
}

#[cfg(test)]
mod eyebrow_osc_tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::time::Duration;

    fn decode(packet: &[u8]) -> (String, f32) {
        let end = packet.iter().position(|byte| *byte == 0).unwrap();
        let address = String::from_utf8(packet[..end].to_vec()).unwrap();
        let value = f32::from_be_bytes(packet[packet.len() - 4..].try_into().unwrap());
        (address, value)
    }

    #[test]
    fn brow_only_mode_matches_vr_eyebrow_eight_parameter_protocol() {
        let receiver = std::net::UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let port = receiver.local_addr().unwrap().port();
        let mut sink = OscSink::new_brow_only("127.0.0.1", port, "/avatar/parameters").unwrap();
        assert!(!sink.eyes_enabled);

        let mut left = EyeResult::new(Eye::Left);
        left.brow = 0.8;
        left.brow_valid = true;
        let mut right = EyeResult::new(Eye::Right);
        right.brow = -0.4;
        right.brow_valid = true;
        sink.on_eye(&left);
        sink.on_eye(&right);

        let mut got = BTreeMap::new();
        for _ in 0..8 {
            let mut packet = [0u8; 256];
            let n = receiver.recv(&mut packet).unwrap();
            let (address, value) = decode(&packet[..n]);
            assert!(
                !address.contains("EyeLid"),
                "brow-only leaked eye OSC: {address}"
            );
            got.insert(address, value);
        }
        let prefix = "/avatar/parameters/FT/v2/";
        assert_eq!(got.len(), 8);
        assert_eq!(got[&format!("{prefix}BrowExpressionLeft")], 0.8);
        assert_eq!(got[&format!("{prefix}BrowExpressionRight")], -0.4);
        assert_eq!(got[&format!("{prefix}BrowUpLeft")], 0.8);
        assert_eq!(got[&format!("{prefix}BrowDownRight")], 0.4);
        // vr_eyebrow: avg=(0.8-0.4)/2=0.2, then split by sign.
        assert!((got[&format!("{prefix}BrowUp")] - 0.2).abs() < 1e-6);
        assert_eq!(got[&format!("{prefix}BrowDown")], 0.0);
    }
}
