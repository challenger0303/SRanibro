//! Varjo NATIVE adapter — drive the Varjo eye cameras directly through `VarjoLib.dll`
//! (the Varjo SDK data-stream API), absorbing the job of the external "Varjo Eye
//! Streamer" tool. No HTTP, no JPEG: we subscribe to the eye-camera data stream and
//! get the raw Y8 grayscale frames straight from the runtime into the existing ML.
//!
//! The MJPEG path ([`super::varjo_adapter`]) stays as the zero-FFI fallback under
//! `device = "varjo_mjpeg"`; this native path is `device = "varjo"` and is gated on a
//! VarjoLib.dll (auto-detected from Varjo Base, or user-supplied `[assets].varjo_lib`).
//!
//! Threading: VarjoLib delivers frames via a callback on a Varjo-OWNED worker thread,
//! so (unlike the StarVR pump, whose callbacks fire synchronously on our thread) the
//! `FrameFn` is shared through an `Arc<Mutex<>>` — same as the MJPEG adapter's two HTTP
//! pumps. Our own capture thread just owns the session lifecycle and idles until stop.

#![cfg(windows)]

use std::ffi::c_void;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

// One-time diagnostic counters (so the log shows whether the callback fires, what
// buffer ids/metadata come back, and whether a frame is emitted — without per-frame
// spam). Reset is not needed; they only gate the first few log lines.
static DBG_CB: AtomicU32 = AtomicU32::new(0);
static DBG_META: AtomicU32 = AtomicU32::new(0);
static DBG_EMIT: AtomicU32 = AtomicU32::new(0);

use super::varjo_sdk::*;
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye, EyeSample, GazeSample};

/// Handed to the Varjo frame callback via `user_data`. Holds the few fn pointers the
/// callback needs plus the shared sink. Lives in a `Box` on the capture thread for the
/// whole subscription; we stop the stream BEFORE dropping it, so no callback ever runs
/// on freed state.
struct VarjoCtx {
    on_frame: Arc<Mutex<FrameFn>>,
    get_buffer_id: PFN_varjo_GetBufferId,
    get_metadata: PFN_varjo_GetBufferMetadata,
    lock_buffer: PFN_varjo_LockDataStreamBuffer,
    get_cpu_data: PFN_varjo_GetBufferCPUData,
    unlock_buffer: PFN_varjo_UnlockDataStreamBuffer,
    /// Per-instance emitted-frame counter (the loop's watchdog reads it). Per-instance,
    /// NOT a process-global static, so a reloaded adapter can't see a stale session's
    /// frames and think a stalled one is alive.
    frame_count: Arc<AtomicU64>,
    /// Gate read at the top of the callback: the loop clears it BEFORE StopDataStream, so
    /// any in-flight callback (which can run during Stop) bails immediately without making
    /// eye-subsystem calls — closing the doff/Stop transient window.
    stream_live: Arc<AtomicBool>,
}

pub struct VarjoNativeAdapter {
    profile: DeviceProfile,
    dll_path: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
}

impl VarjoNativeAdapter {
    pub fn new(dll_path: String) -> Self {
        Self {
            profile: DeviceProfile {
                name: "Varjo (Native SDK)".into(),
                // Frontal IR eye cams like the IS4 — resize-only preprocessing; the real
                // per-frame dims travel with each frame, so resolution is not hardcoded.
                ml_device: "vr4".into(),
                slot_a_eye: Eye::Left,
                image_w: 640,
                image_h: 400,
                transport: "VarjoLib SDK (data stream)".into(),
                streams: "eye-camera Y8 + gaze (Native SDK)".into(),
                gaze_src: "gaze direction (Native SDK)".into(),
                ..DeviceProfile::default()
            },
            dll_path,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
        }
    }
}

fn set_status(s: &Arc<Mutex<String>>, msg: impl Into<String>) {
    if let Ok(mut g) = s.lock() {
        *g = msg.into();
    }
}

/// Sane upper bound on an eye-camera dimension; rejects bogus metadata before any
/// allocation (real Varjo eye cams are ~640x400).
const MAX_DIM: usize = 8192;

/// Validate the buffer metadata and copy the luma out into a fresh `w*h` grayscale
/// Vec, honoring rowStride. Returns `None` (caller skips this eye) on any implausible
/// metadata so a bad/unspecified descriptor can never drive an out-of-bounds read or a
/// process-killing allocation. The buffer must be LOCKED for the duration of this call.
///
/// # Safety
/// `bid` must be a currently-locked buffer of `session`; the fn pointers in `ctx` must
/// be valid.
unsafe fn copy_eye_frame(
    ctx: &VarjoCtx,
    session: *mut varjo_Session,
    bid: i64,
) -> Option<(u32, u32, Vec<u8>)> {
    let meta = (ctx.get_metadata)(session, bid);
    let dbg = DBG_META.fetch_add(1, Ordering::Relaxed) < 2; // log the first couple
    if dbg {
        eprintln!(
            "[varjo] meta: {}x{} stride={} byteSize={} fmt={} type={}",
            meta.width, meta.height, meta.row_stride, meta.byte_size, meta.format, meta.buffer_type
        );
    }
    // Positive dims, and rowStride large enough for one row of luma. `row_stride <
    // width` also rejects a negative stride (it would be < the positive width).
    if meta.width <= 0 || meta.height <= 0 || meta.byte_size <= 0 || meta.row_stride < meta.width {
        if dbg {
            eprintln!("[varjo]  -> rejected: bad dims/stride");
        }
        return None;
    }
    let (w, h) = (meta.width as usize, meta.height as usize);
    if w > MAX_DIM || h > MAX_DIM {
        if dbg {
            eprintln!("[varjo]  -> rejected: dims > {MAX_DIM}");
        }
        return None;
    }
    // Reject multi-byte (color) pixel layouts: a clean 1-byte/px luma row has stride ~=
    // width; RGB/RGBA would be ~3-4x. NV12's luma plane has stride == width, so it
    // passes and we read just the luma. This is robust to texture-format enum drift.
    let stride = meta.row_stride as usize;
    if stride >= 2 * w {
        if dbg {
            eprintln!("[varjo]  -> rejected: multi-byte stride ({stride} >= 2*{w})");
        }
        return None;
    }
    let avail = meta.byte_size as usize;
    // Highest byte index we touch is (h-1)*stride + w; require the runtime's reported
    // byteSize to cover it (checked arithmetic — no overflow).
    let last = (h - 1).checked_mul(stride)?.checked_add(w)?;
    if last > avail {
        if dbg {
            eprintln!("[varjo]  -> rejected: byteSize {avail} < needed {last}");
        }
        return None;
    }
    let ptr = (ctx.get_cpu_data)(session, bid) as *const u8;
    if ptr.is_null() {
        if dbg {
            eprintln!("[varjo]  -> rejected: null CPU data ptr");
        }
        return None;
    }
    let src = std::slice::from_raw_parts(ptr, avail);
    let mut gray = vec![0u8; w * h];
    for row in 0..h {
        let so = row * stride; // so + w <= last <= avail == src.len()  (bounds-safe)
        gray[row * w..row * w + w].copy_from_slice(&src[so..so + w]);
    }
    Some((w as u32, h as u32, gray))
}

/// The Varjo frame-listener callback (runs on a Varjo worker thread). For each eye
/// channel: get buffer id -> lock -> copy grayscale -> UNLOCK -> emit. The copy is done
/// while locked and the buffer is released BEFORE the (potentially slow) ML sink runs,
/// so we never hold the Varjo lock across downstream work. Unlocked on every path.
extern "C" fn varjo_frame_cb(
    frame: *const varjo_StreamFrame,
    session: *mut varjo_Session,
    user: *mut c_void,
) {
    if frame.is_null() || user.is_null() {
        return;
    }
    // SAFETY: `user` is the Box<VarjoCtx> pointer we passed to StartDataStream; it
    // outlives the stream (StopDataStream is a barrier before we drop it). The frame
    // pointer is valid for the duration of this call.
    let ctx = unsafe { &*(user as *const VarjoCtx) };
    let f = unsafe { &*frame };
    let first = DBG_CB.fetch_add(1, Ordering::Relaxed) == 0;
    if first {
        eprintln!(
            "[varjo] callback FIRED: stream_type={} id={} frame={} channels={:#x} flags={:#x}",
            f.stream_type, f.id, f.frame_number, f.channels, f.data_flags
        );
    }
    if f.stream_type != varjo_StreamType_EyeCamera {
        return; // not our stream
    }
    // The loop clears stream_live BEFORE StopDataStream / on any pause, so once it decides
    // to stop we make NO eye-subsystem (GetBufferId/Lock/Metadata/CPUData) calls — even for
    // a callback already in flight during the Stop or the brief doff transient.
    if !ctx.stream_live.load(Ordering::Acquire) {
        return;
    }
    // Per-eye presence is signalled authoritatively by GetBufferId == InvalidId below
    // (more reliable than gating on f.data_flags, which an eye-camera frame may not
    // set); metadata is fully validated in copy_eye_frame regardless, so this is safe.
    for (index, base_eye) in [
        (varjo_ChannelIndex_Left, Eye::Left),
        (varjo_ChannelIndex_Right, Eye::Right),
    ] {
        let bid = unsafe { (ctx.get_buffer_id)(session, f.id, f.frame_number, index) };
        if first {
            eprintln!("[varjo]  ch{index} bufferId={bid}");
        }
        if bid == varjo_InvalidId {
            continue; // that eye absent this frame
        }
        unsafe { (ctx.lock_buffer)(session, bid) }; // MUST lock before reading CPU data
        let emit = unsafe { copy_eye_frame(ctx, session, bid) };
        unsafe { (ctx.unlock_buffer)(session, bid) }; // release ASAP, before emitting
        if let Some((w, h, gray)) = emit {
            ctx.frame_count.fetch_add(1, Ordering::Relaxed); // watchdog activity signal
            if DBG_EMIT.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!("[varjo] FIRST EMIT {w}x{h} (ch{index})");
            }
            // index 0 = Left, 1 = Right (Varjo channels are explicit L/R). Any unit-level
            // L/R inversion is handled downstream by the pipeline's swap_eyes, same as
            // every other adapter.
            if let Ok(mut g) = ctx.on_frame.lock() {
                (g)(base_eye, w, h, &gray);
            }
        }
    }
}

impl HmdAdapter for VarjoNativeAdapter {
    fn name(&self) -> &'static str {
        "varjo-native"
    }
    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> io::Result<()> {
        // One subscription per adapter instance: refuse a second start (the engine
        // builds a fresh adapter on reload, so this never blocks normal use).
        if self.thread.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Varjo capture already running",
            ));
        }
        self.stop.store(false, Ordering::Relaxed); // fresh run
        let dll_path = self.dll_path.clone();
        let stop = self.stop.clone();
        let status = self.status.clone();
        let mut on_gaze = on_gaze; // called from this thread's poll loop (no mutex needed)

        let handle = thread::spawn(move || {
            set_status(&status, "loading VarjoLib…");
            let lib = match unsafe { VarjoLib::load(&dll_path) } {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[varjo] {e}");
                    set_status(&status, format!("VarjoLib load failed: {e}"));
                    return;
                }
            };
            if unsafe { (lib.is_available)() } == varjo_False {
                set_status(
                    &status,
                    "Varjo runtime not available — start Varjo Base & connect the HMD",
                );
                return;
            }
            let session = unsafe { (lib.session_init)() };
            if session.is_null() {
                set_status(&status, "varjo_SessionInit failed");
                return;
            }

            // Gaze symbols present? We init gaze LAZILY (only once the HMD is worn) —
            // calling GazeInit/GetGaze off-head is part of what crashes the driver.
            let gaze_symbols = lib.gaze_init.is_some() && lib.get_gaze.is_some();
            if !gaze_symbols {
                eprintln!("[varjo] gaze API not available in this VarjoLib (images only)");
            }

            // Enumerate streams; find the eye camera by type. Log every config so a
            // missing eye-camera stream is diagnosable from the log.
            let n = unsafe { (lib.cfg_count)(session) };
            if n <= 0 {
                set_status(&status, "no Varjo data streams available");
                unsafe { (lib.session_shutdown)(session) };
                return;
            }
            let mut cfgs: Vec<varjo_StreamConfig> = (0..n as usize)
                .map(|_| unsafe { std::mem::zeroed() })
                .collect();
            unsafe { (lib.get_configs)(session, cfgs.as_mut_ptr(), n) };
            for c in &cfgs {
                eprintln!(
                    "[varjo] stream id={} type={} fmt={} {}x{} stride={} @{}Hz",
                    c.stream_id,
                    c.stream_type,
                    c.format,
                    c.width,
                    c.height,
                    c.row_stride,
                    c.frame_rate
                );
            }
            let eye_cfg = match cfgs
                .iter()
                .find(|c| c.stream_type == varjo_StreamType_EyeCamera)
            {
                Some(c) => *c,
                None => {
                    set_status(
                        &status,
                        "no eye-camera stream — enable eye tracking & 'Allow eye tracking data streams' in Varjo Base",
                    );
                    unsafe { (lib.session_shutdown)(session) };
                    return;
                }
            };
            let stream_id = eye_cfg.stream_id;

            // Callback context (Box keeps its address stable; lives for the whole thread).
            let frame_count = Arc::new(AtomicU64::new(0));
            let stream_live = Arc::new(AtomicBool::new(false));
            let mut ctx = Box::new(VarjoCtx {
                on_frame: Arc::new(Mutex::new(on_frame)),
                get_buffer_id: lib.get_buffer_id,
                get_metadata: lib.get_metadata,
                lock_buffer: lib.lock_buffer,
                get_cpu_data: lib.get_cpu_data,
                unlock_buffer: lib.unlock_buffer,
                frame_count: frame_count.clone(),
                stream_live: stream_live.clone(),
            });
            let ctx_ptr = &mut *ctx as *mut VarjoCtx as *mut c_void;
            // The frame-listener fn pointer, passed BY VALUE (the SDK example passes the
            // function directly). It's a static item, so no lifetime to manage.
            let cb: varjo_FrameListener = varjo_frame_cb;

            // Can we read the UserPresence property (the authoritative "HMD worn" gate)?
            // The UserPresence property (0x2000) is the authoritative "is the HMD worn"
            // signal and the ONLY safe gate (StandbyStatus is about render consumers, not
            // wear). It is REQUIRED: if this VarjoLib can't report it we REFUSE to run
            // rather than fail open and risk driving the eye cameras off-head (the
            // MEMORY_MANAGEMENT BSOD). Standard Varjo Base exports it.
            let presence_ok = match (lib.sync_properties, lib.get_property_bool) {
                (Some(sync), Some(_)) => {
                    unsafe { sync(session) };
                    lib.has_property
                        .map(|h| {
                            (unsafe { h(session, varjo_PropertyKey_UserPresence) }) == varjo_True
                        })
                        .unwrap_or(true) // has_property missing -> assume the key works
                }
                _ => false,
            };
            if !presence_ok {
                set_status(
                    &status,
                    "Varjo runtime can't report UserPresence — refusing (can't safely gate the eye cameras on HMD-worn). Update Varjo Base.",
                );
                eprintln!("[varjo] UserPresence unavailable — refusing to run (fail-closed)");
                unsafe { (lib.session_shutdown)(session) };
                drop(ctx);
                return;
            }
            // Guaranteed present (else we returned above).
            let sync_props = lib.sync_properties.unwrap();
            let get_bool = lib.get_property_bool.unwrap();
            eprintln!("[varjo] gating on UserPresence (HMD worn) — fail-closed");

            // ---- FAIL-CLOSED lifecycle loop ----
            // CRITICAL: Varjo only runs the eye cameras while the HMD is WORN. Driving the
            // data stream / gaze / GazeInit off-head crashes the Varjo driver (observed:
            // Windows MEMORY_MANAGEMENT BSOD). Nothing that touches the eye-tracking
            // subsystem (GazeInit / StartDataStream / GetGaze / buffer calls) runs unless
            // `present` (UserPresence) is true. While paused we make ONLY control-plane
            // calls (poll_event / sync_properties / get_property_bool), safe off-head.
            let mut evt = varjo_Event::default();
            let mut gaze_logged = false;
            let mut present = false; // authoritative wear state (starts unknown -> closed)
            let mut stream_running = false; // do we currently hold a stream subscription
            let mut gaze_attempted = false; // GazeInit tried once (lazily, when first worn)
            let mut gaze_active = false; // GazeInit succeeded -> GetGaze is allowed
            let mut ever_started = false; // StartDataStream ever called (teardown barrier)
            let mut blocked = false; // latched off after a stall; cleared only by a real doff
            let mut retry_after: Option<Instant> = None; // back-off after a Start failure
            let mut last_presence = Instant::now() - Duration::from_secs(1); // check immediately
            let mut last_evidence = Instant::now();
            let mut last_frames = 0u64;
            set_status(&status, "waiting for headset (put it on)…");

            // Local pause helper: stop the stream (idempotent barrier) and mark not-worn.
            // (Inlined as a macro because closures can't easily borrow the many locals.)
            macro_rules! pause_now {
                ($msg:expr) => {{
                    // Clear the callback gate FIRST so any in-flight callback bails before
                    // (and during) StopDataStream — no eye-subsystem calls past this point.
                    stream_live.store(false, Ordering::Release);
                    if stream_running {
                        unsafe { (lib.stop_stream)(session, stream_id) };
                        stream_running = false;
                    }
                    present = false;
                    set_status(&status, $msg);
                }};
            }

            while !stop.load(Ordering::Relaxed) {
                let mut paused_this_iter = false;

                // 1) Drain events — immediate pause hints (faster than the presence poll).
                //    standby==false is only a hint; the UserPresence poll decides resume.
                while unsafe { (lib.poll_event)(session, &mut evt) } == varjo_True {
                    let p = evt._buf.as_ptr();
                    let etype = unsafe { (p as *const i64).read() }; // header.type @0
                    if etype == varjo_EventType_StandbyStatus {
                        let on_standby =
                            unsafe { (p.add(VARJO_EVENT_PAYLOAD_OFFSET) as *const i32).read() }
                                != 0;
                        if on_standby {
                            pause_now!("standby — paused");
                            paused_this_iter = true;
                            // Back off restart ~1s so a chattering standby can't flap Start/Stop.
                            retry_after = Some(Instant::now() + Duration::from_secs(1));
                            eprintln!("[varjo] standby event — paused");
                        }
                    } else if etype == varjo_EventType_DataStreamStop {
                        let sid =
                            unsafe { (p.add(VARJO_EVENT_PAYLOAD_OFFSET) as *const i64).read() };
                        if sid == stream_id {
                            pause_now!("stream stopped by runtime — paused");
                            paused_this_iter = true;
                            retry_after = Some(Instant::now() + Duration::from_secs(1));
                            eprintln!("[varjo] DataStreamStop event — paused");
                        }
                    }
                }

                // 2) Presence poll (~150ms) — the authoritative worn gate.
                if last_presence.elapsed() >= Duration::from_millis(50) {
                    last_presence = Instant::now();
                    unsafe { sync_props(session) };
                    let worn = (unsafe { get_bool(session, varjo_PropertyKey_UserPresence) })
                        == varjo_True;
                    if !worn {
                        // A real doff re-arms everything (clears the stall latch + backoff).
                        blocked = false;
                        retry_after = None;
                        if present || stream_running {
                            pause_now!("headset off — paused");
                            eprintln!("[varjo] headset off (UserPresence) — paused");
                        }
                    } else {
                        present = true;
                    }
                }

                // 3) Reconcile: start only when worn, not stall-latched, not in back-off,
                //    and we didn't just pause this iteration (avoids stop/restart flap).
                let retry_ok = retry_after.map(|t| Instant::now() >= t).unwrap_or(true);
                if present && !blocked && retry_ok && !paused_this_iter && !stream_running {
                    // Lazily init gaze ONCE (only now that the HMD is worn). GetGaze is
                    // gated on gaze_active, set only if GazeInit actually succeeded.
                    if gaze_symbols && !gaze_attempted {
                        gaze_attempted = true;
                        let allowed = lib
                            .is_gaze_allowed
                            .map(|f| (unsafe { f(session) }) == varjo_True)
                            .unwrap_or(true);
                        if allowed {
                            if let Some(gi) = lib.gaze_init {
                                unsafe { gi(session) };
                            }
                            gaze_active = (unsafe { (lib.get_error)(session) }) == varjo_NoError;
                            eprintln!(
                                "[varjo] gaze init {}",
                                if gaze_active { "ok" } else { "failed" }
                            );
                        } else {
                            eprintln!(
                                "[varjo] gaze not allowed (enable eye tracking in Varjo Base)"
                            );
                        }
                    }
                    let _ = unsafe { (lib.get_error)(session) }; // clear stale error
                    unsafe {
                        (lib.start_stream)(
                            session,
                            stream_id,
                            varjo_ChannelFlag_Left | varjo_ChannelFlag_Right,
                            cb,
                            ctx_ptr,
                        );
                    }
                    ever_started = true;
                    let err = unsafe { (lib.get_error)(session) };
                    if err == varjo_NoError {
                        stream_running = true;
                        stream_live.store(true, Ordering::Release); // arm the callback gate
                        retry_after = None;
                        last_evidence = Instant::now();
                        last_frames = frame_count.load(Ordering::Relaxed);
                        set_status(
                            &status,
                            format!("streaming {}x{} eye camera", eye_cfg.width, eye_cfg.height),
                        );
                        eprintln!("[varjo] headset on — streaming eye camera id={stream_id}");
                    } else {
                        // Start failed — Stop as a barrier (Start may have registered the cb),
                        // then back off ~2s so we don't hammer Start/Stop.
                        unsafe { (lib.stop_stream)(session, stream_id) };
                        retry_after = Some(Instant::now() + Duration::from_secs(2));
                        set_status(
                            &status,
                            format!("stream start failed (varjo error {err}) — retrying"),
                        );
                        eprintln!("[varjo] StartDataStream error {err} — backing off 2s");
                    }
                }

                // 4) Watchdog (secondary fail-safe): streaming but no frames for 3s -> stall.
                //    Latch off until a real doff so it can't flap or keep re-driving a bad
                //    device (e.g. UserPresence misreports worn while the cameras are dark).
                if stream_running {
                    let now_frames = frame_count.load(Ordering::Relaxed);
                    if now_frames != last_frames {
                        last_frames = now_frames;
                        last_evidence = Instant::now();
                    }
                    if last_evidence.elapsed() >= Duration::from_secs(3) {
                        stream_live.store(false, Ordering::Release); // gate callbacks first
                        unsafe { (lib.stop_stream)(session, stream_id) };
                        stream_running = false;
                        blocked = true;
                        set_status(&status, "no frames — paused (re-seat the headset)");
                        eprintln!("[varjo] no frames 3s — paused (stall latch)");
                    }
                }

                // 5) Gaze (only while streaming AND gaze actually initialized). openness/pupil
                //    left invalid so the merge keeps the ML-derived openness — only DIRECTION.
                if stream_running && gaze_active {
                    if let Some(get_gaze) = lib.get_gaze {
                        let gz = unsafe { get_gaze(session) };
                        if !gaze_logged {
                            gaze_logged = true;
                            eprintln!(
                                "[varjo] first gaze: status={} L{}({:.2},{:.2},{:.2}) R{}({:.2},{:.2},{:.2})",
                                gz.status, gz.left_status,
                                gz.left_eye.forward[0], gz.left_eye.forward[1], gz.left_eye.forward[2],
                                gz.right_status,
                                gz.right_eye.forward[0], gz.right_eye.forward[1], gz.right_eye.forward[2]
                            );
                        }
                        // Gaze direction only. Pupil is left out: GetGaze's normalized
                        // pupil_size doesn't update on this setup (the real pupil would need
                        // GetGazeData/EyeMeasurements), so we don't fake one. openness/pupil
                        // invalid -> the merge keeps the ML-derived openness untouched.
                        let mk = |ray: &varjo_Ray, eye_status: i64| {
                            let tracked = eye_status != varjo_GazeEyeStatus_Invalid;
                            EyeSample {
                                gaze: [
                                    ray.forward[0] as f32,
                                    ray.forward[1] as f32,
                                    ray.forward[2] as f32,
                                ],
                                gaze_valid: tracked,
                                gaze_reported: true,
                                origin_mm: [
                                    (ray.origin[0] * 1000.0) as f32,
                                    (ray.origin[1] * 1000.0) as f32,
                                    (ray.origin[2] * 1000.0) as f32,
                                ],
                                origin_valid: tracked,
                                pupil_mm: 0.0,
                                pupil_valid: false,
                                pupil_pos: [0.0, 0.0],
                                pupil_pos_valid: false,
                                openness: 0.0,
                                openness_valid: false,
                                openness_reported: false,
                            }
                        };
                        let valid = gz.status != varjo_GazeStatus_Invalid;
                        on_gaze(GazeSample {
                            timestamp_us: (gz.capture_time / 1000).max(0) as u64,
                            left: mk(
                                &gz.left_eye,
                                if valid {
                                    gz.left_status
                                } else {
                                    varjo_GazeEyeStatus_Invalid
                                },
                            ),
                            right: mk(
                                &gz.right_eye,
                                if valid {
                                    gz.right_status
                                } else {
                                    varjo_GazeEyeStatus_Invalid
                                },
                            ),
                        });
                    }
                }

                // Responsive (~200Hz) while streaming; relaxed (20Hz) while paused —
                // enough to catch the "worn" transition without hammering a sleeping device.
                thread::sleep(Duration::from_millis(if stream_running { 5 } else { 50 }));
            }

            // Teardown: gate the callback off first, then Stop as a barrier (safe even if
            // already stopped) so no callback can run on the freed ctx, then shut down.
            stream_live.store(false, Ordering::Release);
            if ever_started {
                unsafe { (lib.stop_stream)(session, stream_id) };
            }
            unsafe { (lib.session_shutdown)(session) };
            drop(ctx); // StopDataStream is a barrier -> ctx outlived every callback
                       // `lib` drops here -> FreeLibrary.
            set_status(&status, "stopped");
            eprintln!("[varjo] stopped");
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
}
