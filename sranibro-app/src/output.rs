//! Output sinks: where post-processed [`EyeResult`]s go (avatar / debug / etc.).
//!
//! The pipeline feeds every sink each frame. `DebugSink` prints throttled state;
//! `BrokenEyeSink` is a TCP server speaking the BrokenEye protocol so
//! VRCFaceTracking's TobiiAdvanced module connects and consumes our eye data.
//! `VrchatEyeOscSink` sends native eye OSC without requiring a VRCFT module;
//! `OscSink` is the separate, avatar-parameter-specific expression protocol.

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
    /// Live native-SRanipal-style EyeWide/EyeSquint -> eyebrow mapping request.
    pub sranipal_brow_link: AtomicBool,
}

/// One eye's already-clamped values, ready to serialize (BrokenEye convention:
/// gaze X is negated and every numeric field is clamped).
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

/// The core has already resolved blink/dropout policy and any user-selected close
/// slew into `EyeResult::openness`. `blink` remains semantic state for gaze gating;
/// forcing zero again in a sink would bypass the configured minimum closing time.
fn emitted_openness(r: &EyeResult) -> f32 {
    let openness = clamp(r.openness, 0.0, 1.0);
    if r.blink && openness < 0.08 {
        0.0
    } else {
        openness
    }
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
        openness: emitted_openness(r),
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
    fn brokeneye_loopback_server_reports_and_releases_ephemeral_port() {
        let sink = BrokenEyeSink::new(0, 0).unwrap();
        let port = sink.status().port;
        assert_ne!(port, 0);
        let client = TcpStream::connect(("127.0.0.1", port)).unwrap();
        assert!(client.local_addr().unwrap().ip().is_loopback());
        drop(client);
        drop(sink);
        TcpListener::bind(("127.0.0.1", port))
            .expect("dropping the VRCFT sink must release its loopback port");
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

    #[test]
    fn brokeneye_stream_switches_sranipal_brow_link_live() {
        fn read_json(stream: &mut TcpStream) -> String {
            let mut header = [0u8; 5];
            stream.read_exact(&mut header).unwrap();
            assert_eq!(header[0], MODE_JSON);
            let len = u32::from_le_bytes(header[1..5].try_into().unwrap()) as usize;
            let mut payload = vec![0u8; len];
            stream.read_exact(&mut payload).unwrap();
            String::from_utf8(payload).unwrap()
        }

        let sink = BrokenEyeSink::new(0, 0).unwrap();
        let status = sink.status();
        let mut client = TcpStream::connect(("127.0.0.1", status.port)).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        client.write_all(&[MODE_JSON]).unwrap();

        assert!(
            read_json(&mut client).contains("\"sranipal_brow_link\":false"),
            "the safe default must leave eyebrow shapes untouched"
        );

        status.sranipal_brow_link.store(true, Ordering::Relaxed);
        let switched =
            (0..4).any(|_| read_json(&mut client).contains("\"sranipal_brow_link\":true"));
        assert!(
            switched,
            "the VRCFT control flag must change without reconnecting"
        );
    }

    #[test]
    fn blink_semantics_do_not_bypass_the_core_close_slew() {
        let mut result = EyeResult::new(Eye::Left);
        result.openness = 0.625;
        result.openness_valid = true;
        result.gaze_valid = true;
        result.blink = true;

        let encoded = eye_json(&result);
        assert_eq!(encoded.openness, 0.625);
        assert!(encoded.openness_valid);
        assert!(
            !encoded.gaze_valid,
            "blink remains available for gaze gating"
        );

        result.openness = 0.04;
        assert_eq!(eye_json(&result).openness, 0.0);
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
        // VRCFaceTracking is a same-machine consumer. Eye/gaze/expression data must
        // never be exposed to the LAN merely because the host has a public interface.
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        let port = listener.local_addr()?.port();
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
        eprintln!("[vrcft] VRCFT server listening on 127.0.0.1:{port}");
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
            let brow_link = status.sranipal_brow_link.load(Ordering::Relaxed);
            let _ = write!(
                json,
                "{{\"noise_filter_samples\":{samples},\"sranipal_brow_link\":{brow_link},\"left\":"
            );
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
            openness: emitted_openness(r),
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
    osc_float_message(address, &[value])
}

/// Encode OSC float arguments without converting direction vectors into angles
/// or losing the six independent components of a stereo gaze sample.
fn osc_float_message(address: &str, values: &[f32]) -> Vec<u8> {
    fn padded(bytes: &[u8]) -> Vec<u8> {
        let pad = 4 - (bytes.len() % 4); // 1..=4: always at least one null terminator
        let mut v = Vec::with_capacity(bytes.len() + pad);
        v.extend_from_slice(bytes);
        v.extend(std::iter::repeat(0u8).take(pad));
        v
    }
    let mut msg = padded(address.as_bytes());
    let mut tags = vec![b','; values.len() + 1];
    tags[1..].fill(b'f');
    msg.extend(padded(&tags));
    for value in values {
        msg.extend_from_slice(&value.to_be_bytes());
    }
    msg
}

// ---------------------------------------------------------------------------
// VRChat native eye OSC (no custom avatar parameters or VRCFT module required)
// ---------------------------------------------------------------------------

/// VRChat's documented `/tracking/eye` protocol, distinct from [`OscSink`].
///
/// Sends `LeftRightVec` (six unit-vector components) and `EyesClosedAmount` (one
/// shared eyelid value). The native protocol does not expose independent winks,
/// pupil dilation, wide/squint, or eyebrows, so none of those are synthesized.
/// See <https://docs.vrchat.com/docs/osc-eye-tracking>.
///
/// The input must be a real HMD-local direction with +Z forward, as required by
/// SRanibro's gaze-vector contract. The pipeline has already applied per-device
/// mappings; this sink applies the same final X sign as the existing OSC sink,
/// then normalizes the vector. It never treats normalized avatar parameters as
/// degrees, invents a missing Z component, or applies a gaze-strength multiplier.
/// Invalid gaze/eyelids are not sent, allowing VRChat's independent ten-second
/// tracking timeouts to restore automatic eye behavior after tracking is lost.
pub struct VrchatEyeOscSink {
    sock: std::net::UdpSocket,
    pending: [Option<EyeResult>; 2],
}

impl VrchatEyeOscSink {
    pub fn new(host: impl Into<String>, port: u16) -> std::io::Result<Self> {
        use std::net::ToSocketAddrs;

        if port == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "VRChat OSC destination port must not be zero",
            ));
        }
        let host = host.into();
        let mut last_error = std::io::Error::new(
            std::io::ErrorKind::AddrNotAvailable,
            "VRChat OSC host resolved to no addresses",
        );
        // Resolve once, outside the emit loop. Match the socket's address family
        // to the destination (including explicit IPv6 loopback configurations).
        for target in (host.as_str(), port).to_socket_addrs()? {
            let bind = if target.is_ipv6() {
                "[::]:0"
            } else {
                "0.0.0.0:0"
            };
            let connected = (|| {
                let sock = std::net::UdpSocket::bind(bind)?;
                sock.connect(target)?;
                sock.set_nonblocking(true)?;
                Ok::<_, std::io::Error>(sock)
            })();
            match connected {
                Ok(sock) => {
                    eprintln!("[vrchat-eye-osc] -> {target} /tracking/eye");
                    return Ok(Self {
                        sock,
                        pending: [None, None],
                    });
                }
                Err(error) => last_error = error,
            }
        }
        Err(last_error)
    }

    fn emit_pair(&self, left: &EyeResult, right: &EyeResult) {
        if let (Some(left), Some(right)) = (native_eye_vector(left), native_eye_vector(right)) {
            let message = osc_float_message(
                "/tracking/eye/LeftRightVec",
                &[left[0], left[1], left[2], right[0], right[1], right[2]],
            );
            // OSC is best-effort UDP. Never block inference/emit on a slow peer.
            let _ = self.sock.send(&message);
        }
        if let Some(closed) = native_eyes_closed(left, right) {
            let _ = self
                .sock
                .send(&osc_message("/tracking/eye/EyesClosedAmount", closed));
        }
    }
}

fn native_eye_vector(result: &EyeResult) -> Option<[f32; 3]> {
    if !result.gaze_valid
        || (result.blink && !result.gaze_yoked)
        || !result.gaze.iter().all(|value| value.is_finite())
        || result.gaze[2] <= 0.0
    {
        return None;
    }
    let [x, y, z] = result.gaze;
    let length = x.hypot(y).hypot(z);
    if !length.is_finite() || length <= 1.0e-6 {
        return None;
    }
    Some([-x / length, y / length, z / length])
}

fn native_eyes_closed(left: &EyeResult, right: &EyeResult) -> Option<f32> {
    if [left, right]
        .iter()
        .any(|eye| !(eye.openness_valid || eye.blink) || !eye.openness.is_finite())
    {
        return None;
    }
    // VRChat currently has only one eyelid channel. Explicit policy: average the
    // core's clamped/slew-limited openness, then invert (0=open, 1=closed).
    Some(1.0 - 0.5 * (emitted_openness(left) + emitted_openness(right)))
}

impl OutputSink for VrchatEyeOscSink {
    fn name(&self) -> &str {
        "vrchat-eye-osc"
    }

    fn on_eye(&mut self, result: &EyeResult) {
        self.pending[result.eye.idx()] = Some(*result);
        if let [Some(left), Some(right)] = self.pending {
            self.pending = [None, None];
            self.emit_pair(&left, &right);
        }
    }

    fn on_frame(&mut self, results: &[EyeResult; 2]) {
        // Never mix a partially collected on_eye pair with a coherent on_frame.
        self.pending = [None, None];
        for result in results {
            self.on_eye(result);
        }
    }
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
            self.send(&format!("EyeLid{side}"), emitted_openness(r));
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

#[cfg(test)]
mod native_eye_osc_tests {
    use super::*;
    use std::net::UdpSocket;

    fn tracked(eye: Eye) -> EyeResult {
        let mut result = EyeResult::new(eye);
        result.gaze = [0.0, 0.0, 1.0];
        result.gaze_valid = true;
        result.openness_valid = true;
        result
    }

    fn loopback() -> (UdpSocket, VrchatEyeOscSink) {
        let receiver = UdpSocket::bind(("127.0.0.1", 0)).unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let sink =
            VrchatEyeOscSink::new("127.0.0.1", receiver.local_addr().unwrap().port()).unwrap();
        (receiver, sink)
    }

    fn receive(receiver: &UdpSocket) -> (String, Vec<f32>) {
        let mut packet = [0u8; 512];
        let n = receiver.recv(&mut packet).unwrap();
        let packet = &packet[..n];
        assert_eq!(n % 4, 0, "OSC packets are four-byte aligned");
        let address_end = packet.iter().position(|byte| *byte == 0).unwrap();
        let address = String::from_utf8(packet[..address_end].to_vec()).unwrap();
        let tags_start = (address_end + 4) & !3;
        assert!(packet[address_end..tags_start]
            .iter()
            .all(|byte| *byte == 0));
        let tags_end = tags_start
            + packet[tags_start..]
                .iter()
                .position(|byte| *byte == 0)
                .unwrap();
        let tags = &packet[tags_start..tags_end];
        assert_eq!(tags[0], b',');
        assert!(tags[1..].iter().all(|byte| *byte == b'f'));
        let values_start = (tags_end + 4) & !3;
        assert!(packet[tags_end..values_start].iter().all(|byte| *byte == 0));
        assert_eq!(n - values_start, 4 * (tags.len() - 1));
        let values = packet[values_start..]
            .chunks_exact(4)
            .map(|bytes| f32::from_be_bytes(bytes.try_into().unwrap()))
            .collect();
        (address, values)
    }

    fn assert_no_packets(receiver: &UdpSocket) {
        receiver.set_nonblocking(true).unwrap();
        let mut packet = [0u8; 512];
        let error = receiver
            .recv(&mut packet)
            .expect_err("unexpected additional OSC message");
        assert_eq!(error.kind(), std::io::ErrorKind::WouldBlock);
        receiver.set_nonblocking(false).unwrap();
    }

    #[test]
    fn single_float_osc_keeps_legacy_wire_format() {
        assert_eq!(osc_message("/a", 1.0), b"/a\0\0,f\0\0\x3f\x80\0\0");
        assert_eq!(osc_message("/abc", -0.5), b"/abc\0\0\0\0,f\0\0\xbf\0\0\0");
    }

    #[test]
    fn native_osc_uses_documented_stereo_vectors_and_shared_eyelid_only() {
        let (receiver, mut sink) = loopback();
        let mut left = tracked(Eye::Left);
        left.gaze = [-0.6, 0.0, 0.8];
        left.openness = 0.25;
        left.brow = 0.75;
        left.brow_valid = true;
        left.pupil_mm = 4.0;
        left.pupil_valid = true;
        let mut right = tracked(Eye::Right);
        right.gaze = [0.0, 6.0, 8.0];
        right.openness = 0.75;
        // Order is anatomical, not array order.
        sink.on_frame(&[right, left]);

        let (address, values) = receive(&receiver);
        assert_eq!(address, "/tracking/eye/LeftRightVec");
        assert_eq!(values, vec![0.6, 0.0, 0.8, 0.0, 0.6, 0.8]);
        assert_eq!(
            receive(&receiver),
            ("/tracking/eye/EyesClosedAmount".into(), vec![0.5])
        );
        assert_no_packets(&receiver);
    }

    #[cfg(feature = "external-eyelid-model")]
    #[test]
    fn model_gaze_axes_match_vrcft_and_vrchat_without_changing_eyelids() {
        use crate::ml::xr5_gaze::Snapshot;
        let (receiver,mut sink)=loopback();
        let now=std::time::Instant::now();
        let snapshot=Snapshot {states:[1,1],directions:[[0.6,0.,-0.8],[-0.6,0.,-0.8]],captured:Some(now),..Default::default()};
        let mut eyes=[tracked(Eye::Left),tracked(Eye::Right)];
        eyes[0].openness=0.25;eyes[1].openness=0.75;
        snapshot.apply(&mut eyes,now,true);
        assert_eq!(eye_json(&eyes[0]).gaze,[0.6,0.]);
        assert_eq!(eye_json(&eyes[1]).gaze,[-0.6,0.]);
        sink.on_frame(&eyes);
        assert_eq!(receive(&receiver),("/tracking/eye/LeftRightVec".into(),vec![0.6,0.,0.8,-0.6,0.,0.8]));
        assert_eq!(receive(&receiver),("/tracking/eye/EyesClosedAmount".into(),vec![0.5]));
        snapshot.apply(&mut eyes,now+std::time::Duration::from_millis(51),true);
        assert!(!eye_json(&eyes[0]).gaze_valid);
        sink.on_frame(&eyes);
        assert_eq!(receive(&receiver).0,"/tracking/eye/EyesClosedAmount");
        assert_no_packets(&receiver);
    }

    #[test]
    fn native_osc_requires_a_fresh_pair_and_does_not_mix_frame_entrypoints() {
        let (receiver, mut sink) = loopback();
        let mut left = tracked(Eye::Left);
        let right = tracked(Eye::Right);
        sink.on_eye(&left);
        assert_no_packets(&receiver);
        left.gaze = [0.6, 0.0, 0.8];
        sink.on_eye(&left);
        assert_no_packets(&receiver);
        sink.on_eye(&right);
        assert_eq!(receive(&receiver).1[0], -0.6);
        receive(&receiver);
        sink.on_eye(&right);
        assert_no_packets(&receiver);
        left.gaze = [-0.6, 0.0, 0.8];
        sink.on_frame(&[right, left]);
        assert_eq!(receive(&receiver).1[0], 0.6);
        receive(&receiver);
        assert_no_packets(&receiver);
    }

    #[test]
    fn native_gaze_rejects_invalid_nonfinite_zero_and_backward_vectors() {
        let mut eye = tracked(Eye::Left);
        for gaze in [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [f32::NAN, 0.0, 1.0],
            [0.0, f32::INFINITY, 1.0],
            [0.0, 0.0, f32::NAN],
        ] {
            eye.gaze = gaze;
            assert_eq!(native_eye_vector(&eye), None);
        }
        eye.gaze = [0.0, 0.0, 1.0];
        eye.gaze_valid = false;
        assert_eq!(native_eye_vector(&eye), None);
        eye.gaze_valid = true;
        eye.blink = true;
        assert_eq!(native_eye_vector(&eye), None);
        eye.gaze_yoked = true;
        assert_eq!(native_eye_vector(&eye), Some([0.0, 0.0, 1.0]));
    }

    #[test]
    fn invalid_native_gaze_does_not_stop_valid_eyelids_or_reuse_stale_gaze() {
        let (receiver, mut sink) = loopback();
        let mut left = tracked(Eye::Left);
        let right = tracked(Eye::Right);
        sink.on_frame(&[left, right]);
        receive(&receiver);
        receive(&receiver);
        left.gaze_valid = false;
        sink.on_frame(&[left, right]);
        assert_eq!(
            receive(&receiver),
            ("/tracking/eye/EyesClosedAmount".into(), vec![0.0])
        );
        assert_no_packets(&receiver);
    }

    #[test]
    fn invalid_native_eyelids_do_not_stop_valid_gaze() {
        let (receiver, mut sink) = loopback();
        let left = tracked(Eye::Left);
        let mut right = tracked(Eye::Right);
        right.openness_valid = false;
        sink.on_frame(&[left, right]);
        assert_eq!(receive(&receiver).0, "/tracking/eye/LeftRightVec");
        assert_no_packets(&receiver);
    }

    #[test]
    fn native_eyelids_preserve_core_slew_clamp_and_validity() {
        let mut left = tracked(Eye::Left);
        let mut right = tracked(Eye::Right);
        left.blink = true;
        right.blink = true;
        left.openness = 0.625;
        right.openness = 0.625;
        assert_eq!(native_eyes_closed(&left, &right), Some(0.375));
        left.openness = 0.04;
        right.openness = 0.04;
        assert_eq!(native_eyes_closed(&left, &right), Some(1.0));
        left.openness = -4.0;
        right.openness = 4.0;
        assert_eq!(native_eyes_closed(&left, &right), Some(0.5));
        right.openness = f32::NAN;
        assert_eq!(native_eyes_closed(&left, &right), None);
        right.openness = 1.0;
        right.blink = false;
        right.openness_valid = false;
        assert_eq!(native_eyes_closed(&left, &right), None);
    }

    #[test]
    fn native_osc_rejects_zero_destination_port() {
        match VrchatEyeOscSink::new("127.0.0.1", 0) {
            Err(error) => assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput),
            Ok(_) => panic!("port zero is not a valid OSC destination"),
        }
    }
}
