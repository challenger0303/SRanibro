//! Varjo adapter — consume the **Varjo Eye Streamer** (by DimeRhyme) MJPEG-over-HTTP
//! eye-camera feed instead of binding the Varjo Native SDK. That tool drives the Varjo
//! SDK and re-publishes both eye cameras as MJPEG (left `:8080`, right `:8081` by
//! default). We GET each stream, slice out JPEG frames (SOI `FF D8` .. EOI `FF D9`),
//! decode to grayscale, and feed the existing EyePrediction ML (variable-resolution).
//!
//! Requires: Varjo Base + a Varjo HMD + the Varjo Eye Streamer running ("Start
//! Server"). No Varjo SDK FFI, no Tobii DLL. Gaze is NOT provided by the streamer
//! (images only) — native Varjo gaze is a future add via the SDK. This path's job is
//! to verify the SRanipal eye-model on Varjo's eye images (openness/wide/squeeze).

#![cfg(windows)]

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye};

pub struct VarjoAdapter {
    profile: DeviceProfile,
    left_url: String,
    right_url: String,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}

impl VarjoAdapter {
    pub fn new(cfg: &crate::config::Config) -> Self {
        Self {
            profile: DeviceProfile {
                name: "Varjo (Eye Streamer)".into(),
                // Resize-only preprocessing; the real frame dims are carried per frame,
                // so the model sees Varjo's native resolution downscaled to 100x100.
                ml_device: "vr4".into(),
                slot_a_eye: Eye::Left,
                image_w: 640,
                image_h: 400,
                transport: "MJPEG over HTTP".into(),
                streams: "eye-camera MJPEG (Varjo Eye Streamer)".into(),
                gaze_src: "— (images only, no gaze)".into(),
                ..DeviceProfile::default()
            },
            left_url: cfg.hmd.varjo_left_url.clone(),
            right_url: cfg.hmd.varjo_right_url.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            threads: Vec::new(),
            status: Arc::new(Mutex::new("idle".into())),
        }
    }
}

fn set_status(s: &Arc<Mutex<String>>, msg: impl Into<String>) {
    if let Ok(mut g) = s.lock() {
        *g = msg.into();
    }
}

/// "http://host:port/path" -> "host:port" (path ignored; default port 80).
fn host_port(url: &str) -> String {
    let s = url.trim();
    let s = s
        .strip_prefix("http://")
        .or_else(|| s.strip_prefix("https://"))
        .unwrap_or(s);
    let s = s.split('/').next().unwrap_or(s);
    if s.contains(':') {
        s.to_string()
    } else {
        format!("{s}:80")
    }
}

/// First index of `needle` in `hay`.
fn find_sub(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Parse `Content-Length` (case-insensitive) from a multipart part-header block.
fn parse_content_length(headers: &[u8]) -> Option<usize> {
    for line in headers.split(|&b| b == b'\n') {
        let line = String::from_utf8_lossy(line);
        let l = line.trim();
        let b = l.as_bytes();
        // Byte-compare the prefix (no str slicing — avoids a panic if byte 15 isn't a
        // UTF-8 char boundary on a garbage line). The 15 matched bytes are ASCII, so
        // `l[15..]` is then on a char boundary and safe.
        if b.len() >= 15 && b[..15].eq_ignore_ascii_case(b"content-length:") {
            return l[15..].trim().parse::<usize>().ok();
        }
    }
    None
}

const MAX_JPEG_BYTES: usize = 8 << 20; // sane cap on one compressed frame (8 MB)
const MAX_DECODE_BYTES: usize = 16 << 20; // decoded-pixel cap — rejects bogus huge dims

/// Decode one JPEG to grayscale and hand it to `on_frame` with its native dims.
fn decode_and_emit(eye: Eye, jpeg: &[u8], on_frame: &Arc<Mutex<FrameFn>>) {
    let mut dec = jpeg_decoder::Decoder::new(jpeg);
    dec.set_max_decoding_buffer_size(MAX_DECODE_BYTES);
    // Read just the header first and REJECT implausible dimensions BEFORE decode()
    // allocates dimension-sized planes — so a bogus huge-dim JPEG can't OOM/abort
    // Real eye cameras are well under 4096².
    if dec.read_info().is_err() {
        return;
    }
    let info = match dec.info() {
        Some(i) => i,
        None => return,
    };
    let (w, h) = (info.width as usize, info.height as usize);
    if w == 0 || h == 0 || w > 4096 || h > 4096 {
        return;
    }
    let pixels = match dec.decode() {
        Ok(p) => p,
        Err(_) => return,
    };
    let gray: Vec<u8> = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => {
            if pixels.len() < w * h {
                return;
            }
            pixels
        }
        jpeg_decoder::PixelFormat::RGB24 => {
            if pixels.len() < w * h * 3 {
                return;
            }
            pixels
                .chunks_exact(3)
                .map(|c| ((c[0] as u32 * 77 + c[1] as u32 * 150 + c[2] as u32 * 29) >> 8) as u8)
                .collect()
        }
        jpeg_decoder::PixelFormat::L16 => {
            if pixels.len() < w * h * 2 {
                return;
            }
            pixels.chunks_exact(2).map(|c| c[1]).collect() // high byte
        }
        _ => return, // CMYK32 etc. — not expected from an eye camera
    };
    if let Ok(mut guard) = on_frame.lock() {
        let f: &mut FrameFn = &mut guard;
        f(eye, w as u32, h as u32, &gray);
    }
}

/// One eye's MJPEG pump: connect -> GET -> slice JPEG frames -> decode -> on_frame.
/// Reconnects on error until `stop`.
fn pump_eye(
    eye: Eye,
    url: String,
    on_frame: Arc<Mutex<FrameFn>>,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<String>>,
) {
    use std::net::ToSocketAddrs;
    let backoff = |status: &Arc<Mutex<String>>, stop: &Arc<AtomicBool>| -> bool {
        for _ in 0..20 {
            if stop.load(Ordering::Relaxed) {
                return false;
            }
            thread::sleep(Duration::from_millis(100));
        }
        let _ = status;
        true
    };
    while !stop.load(Ordering::Relaxed) {
        let hp = host_port(&url);
        let addr = match hp.to_socket_addrs().ok().and_then(|mut a| a.next()) {
            Some(a) => a,
            None => {
                set_status(&status, format!("bad Varjo URL: {hp}"));
                if !backoff(&status, &stop) {
                    return;
                }
                continue;
            }
        };
        // connect_timeout (not blocking connect) so stop()/join() can't hang on a dead host.
        let mut stream = match TcpStream::connect_timeout(&addr, Duration::from_secs(1)) {
            Ok(s) => s,
            Err(e) => {
                set_status(
                    &status,
                    format!("Varjo Eye Streamer not reachable at {hp} — start it ({e})"),
                );
                if !backoff(&status, &stop) {
                    return;
                }
                continue;
            }
        };
        // If the read timeout can't be set, don't proceed — a blocking read would make
        // stop()/join() hang. Reconnect instead.
        if stream
            .set_read_timeout(Some(Duration::from_millis(1000)))
            .is_err()
        {
            if !backoff(&status, &stop) {
                return;
            }
            continue;
        }
        let req = format!("GET / HTTP/1.1\r\nHost: {hp}\r\nConnection: keep-alive\r\n\r\n");
        if stream.write_all(req.as_bytes()).is_err() {
            continue;
        }
        set_status(&status, "streaming");

        // Parse the multipart/x-mixed-replace MJPEG stream by Content-Length (exact
        // framing — avoids the FF-D9-in-an-APP/COM-segment truncation that a blind
        // marker scan would hit). Accumulate raw bytes; read timeouts just keep waiting.
        let mut buf: Vec<u8> = Vec::with_capacity(1 << 18);
        let mut http_done = false;
        let mut tmp = [0u8; 16384];
        'conn: while !stop.load(Ordering::Relaxed) {
            match stream.read(&mut tmp) {
                Ok(0) => break, // peer closed
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(ref e)
                    if e.kind() == std::io::ErrorKind::WouldBlock
                        || e.kind() == std::io::ErrorKind::TimedOut =>
                {
                    continue; // read timeout — re-check stop, keep reading
                }
                Err(_) => break,
            }
            // Phase 1: drop the HTTP response headers (up to the first blank line).
            if !http_done {
                match find_sub(&buf, b"\r\n\r\n") {
                    Some(i) => {
                        buf.drain(..i + 4);
                        http_done = true;
                    }
                    None => {
                        if buf.len() > 16384 {
                            break; // headers absurdly large — drop the connection
                        }
                        continue;
                    }
                }
            }
            // Phase 2: extract each part — header block ends at \r\n\r\n; Content-Length
            // gives the exact JPEG byte count that follows.
            loop {
                let h = match find_sub(&buf, b"\r\n\r\n") {
                    Some(i) => i,
                    None => {
                        if buf.len() > 65536 {
                            buf.clear(); // no part header in 64 KB — resync
                        }
                        break;
                    }
                };
                let clen = match parse_content_length(&buf[..h]) {
                    Some(n) if n > 0 && n <= MAX_JPEG_BYTES => n,
                    _ => {
                        // No usable Content-Length — not the expected multipart MJPEG.
                        // Reconnect rather than risk scanning \r\n\r\n into JPEG body bytes.
                        break 'conn;
                    }
                };
                let body = h + 4;
                if buf.len() < body + clen {
                    break; // wait for the full JPEG
                }
                decode_and_emit(eye, &buf[body..body + clen], &on_frame);
                buf.drain(..body + clen);
            }
        }
        set_status(&status, "reconnecting…");
    }
    set_status(&status, "stopped");
}

impl HmdAdapter for VarjoAdapter {
    fn name(&self) -> &'static str {
        "varjo"
    }
    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, on_frame: FrameFn, _on_gaze: GazeFn) -> std::io::Result<()> {
        // Two HTTP pumps (left/right) share the single FrameFn via a mutex (each call
        // just stashes the newest frame, so contention is negligible). The streamer
        // provides images only — no gaze.
        let on_frame = Arc::new(Mutex::new(on_frame));
        for (eye, url) in [
            (Eye::Left, self.left_url.clone()),
            (Eye::Right, self.right_url.clone()),
        ] {
            let (f, stop, status) = (on_frame.clone(), self.stop.clone(), self.status.clone());
            self.threads
                .push(thread::spawn(move || pump_eye(eye, url, f, stop, status)));
        }
        Ok(())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }

    fn status_arc(&self) -> Arc<Mutex<String>> {
        self.status.clone()
    }
}
