
#![cfg(windows)]

use std::ffi::c_void;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static DBG_CB: AtomicU32 = AtomicU32::new(0);
static DBG_META: AtomicU32 = AtomicU32::new(0);
static DBG_EMIT: AtomicU32 = AtomicU32::new(0);

use super::varjo_sdk::*;
use super::{FrameFn, GazeFn, HmdAdapter};
use crate::core::types::{DeviceProfile, Eye, EyeSample, GazeSample};

struct VarjoCtx {
    on_frame: Arc<Mutex<FrameFn>>,
    get_buffer_id: PFN_varjo_GetBufferId,
    get_metadata: PFN_varjo_GetBufferMetadata,
    lock_buffer: PFN_varjo_LockDataStreamBuffer,
    get_cpu_data: PFN_varjo_GetBufferCPUData,
    unlock_buffer: PFN_varjo_UnlockDataStreamBuffer,
    frame_count: Arc<AtomicU64>,
    stream_live: Arc<AtomicBool>,
}

pub struct VarjoNativeAdapter {
    profile: DeviceProfile,
    dll_path: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    status: Arc<Mutex<String>>,
    raw_gaze_callback: Option<GazeFn>,
}

impl VarjoNativeAdapter {
    pub fn new(dll_path: String) -> Self {
        Self {
            profile: DeviceProfile {
                name: "Varjo (Native SDK)".into(),
                ml_device: "vr4".into(),
                slot_a_eye: Eye::Left,
                image_w: 640,
                image_h: 400,
                transport: "Varjo Base SDK (data stream)".into(),
                streams: "eye-camera Y8 + gaze (Native SDK)".into(),
                gaze_src: "gaze direction (Native SDK)".into(),
                ..DeviceProfile::default()
            },
            dll_path,
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            status: Arc::new(Mutex::new("idle".into())),
            raw_gaze_callback: None,
        }
    }
}

#[derive(Default)]
struct RawGazePoll {
    last_key: Option<(i64, i64, i64, i64)>,
}

impl RawGazePoll {
    fn sample(&mut self, gaze: &varjo_Gaze) -> Option<GazeSample> {
        let key = (
            gaze.capture_time,
            gaze.status,
            gaze.left_status,
            gaze.right_status,
        );
        if self.last_key == Some(key) {
            return None;
        }
        self.last_key = Some(key);
        let eye = |ray: &varjo_Ray, status: i64| EyeSample {
            gaze: ray.forward.map(|value| value as f32),
            gaze_valid: gaze.status == varjo_GazeStatus_Valid
                && status >= varjo_GazeEyeStatus_Compensated,
            gaze_reported: true,
            ..EyeSample::default()
        };
        Some(GazeSample {
            timestamp_us: (gaze.capture_time / 1000).max(0) as u64,
            left: eye(&gaze.left_eye, gaze.left_status),
            right: eye(&gaze.right_eye, gaze.right_status),
        })
    }
}

fn set_status(s: &Arc<Mutex<String>>, msg: impl Into<String>) {
    if let Ok(mut g) = s.lock() {
        *g = msg.into();
    }
}

const MAX_DIM: usize = 8192;

unsafe fn copy_eye_frame(
    ctx: &VarjoCtx,
    session: *mut varjo_Session,
    bid: i64,
) -> Option<(u32, u32, Vec<u8>)> {
    let meta = (ctx.get_metadata)(session, bid);
    let dbg = DBG_META.fetch_add(1, Ordering::Relaxed) < 2;
    if dbg {
        eprintln!(
            "[varjo] meta: {}x{} stride={} byteSize={} fmt={} type={}",
            meta.width, meta.height, meta.row_stride, meta.byte_size, meta.format, meta.buffer_type
        );
    }
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
    let stride = meta.row_stride as usize;
    if stride >= 2 * w {
        if dbg {
            eprintln!("[varjo]  -> rejected: multi-byte stride ({stride} >= 2*{w})");
        }
        return None;
    }
    let avail = meta.byte_size as usize;
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
        let so = row * stride;
        gray[row * w..row * w + w].copy_from_slice(&src[so..so + w]);
    }
    Some((w as u32, h as u32, gray))
}

extern "C" fn varjo_frame_cb(
    frame: *const varjo_StreamFrame,
    session: *mut varjo_Session,
    user: *mut c_void,
) {
    if frame.is_null() || user.is_null() {
        return;
    }
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
        return;
    }
    if !ctx.stream_live.load(Ordering::Acquire) {
        return;
    }
    for (index, base_eye) in [
        (varjo_ChannelIndex_Left, Eye::Left),
        (varjo_ChannelIndex_Right, Eye::Right),
    ] {
        let bid = unsafe { (ctx.get_buffer_id)(session, f.id, f.frame_number, index) };
        if first {
            eprintln!("[varjo]  ch{index} bufferId={bid}");
        }
        if bid == varjo_InvalidId {
            continue;
        }
        unsafe { (ctx.lock_buffer)(session, bid) };
        let emit = unsafe { copy_eye_frame(ctx, session, bid) };
        unsafe { (ctx.unlock_buffer)(session, bid) };
        if let Some((w, h, gray)) = emit {
            ctx.frame_count.fetch_add(1, Ordering::Relaxed);
            if DBG_EMIT.fetch_add(1, Ordering::Relaxed) == 0 {
                eprintln!("[varjo] FIRST EMIT {w}x{h} (ch{index})");
            }
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

    fn set_raw_gaze_callback(&mut self, callback: GazeFn) -> bool {
        if self.thread.is_some() {
            return false;
        }
        self.raw_gaze_callback = Some(callback);
        true
    }

    fn start(&mut self, on_frame: FrameFn, on_gaze: GazeFn) -> io::Result<()> {
        if self.thread.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "Varjo capture already running",
            ));
        }
        self.stop.store(false, Ordering::Relaxed);
        let dll_path = self.dll_path.clone();
        let stop = self.stop.clone();
        let status = self.status.clone();
        let mut on_gaze = on_gaze;
        let mut on_raw_gaze = self.raw_gaze_callback.take();

        let handle = thread::spawn(move || {
            let mut raw_gaze_poll = RawGazePoll::default();
            set_status(&status, "loading Varjo Base runtime…");
            let lib = match unsafe { VarjoLib::load(&dll_path) } {
                Ok(l) => l,
                Err(e) => {
                    eprintln!("[varjo] {e}");
                    set_status(&status, "Varjo Base runtime load failed");
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

            let gaze_symbols = lib.gaze_init.is_some() && lib.get_gaze.is_some();
            if !gaze_symbols {
                eprintln!("[varjo] gaze API not available in this VarjoLib (images only)");
            }

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
            let cb: varjo_FrameListener = varjo_frame_cb;

            let presence_ok = match (lib.sync_properties, lib.get_property_bool) {
                (Some(sync), Some(_)) => {
                    unsafe { sync(session) };
                    lib.has_property
                        .map(|h| {
                            (unsafe { h(session, varjo_PropertyKey_UserPresence) }) == varjo_True
                        })
                        .unwrap_or(true)
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
            let sync_props = lib.sync_properties.unwrap();
            let get_bool = lib.get_property_bool.unwrap();
            eprintln!("[varjo] gating on UserPresence (HMD worn) — fail-closed");

            let mut evt = varjo_Event::default();
            let mut gaze_logged = false;
            let mut present = false;
            let mut stream_running = false;
            let mut gaze_attempted = false;
            let mut gaze_active = false;
            let mut ever_started = false;
            let mut blocked = false;
            let mut retry_after: Option<Instant> = None;
            let mut last_presence = Instant::now() - Duration::from_secs(1);
            let mut last_evidence = Instant::now();
            let mut last_frames = 0u64;
            set_status(&status, "waiting for headset (put it on)…");

            macro_rules! pause_now {
                ($msg:expr) => {{
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

                while unsafe { (lib.poll_event)(session, &mut evt) } == varjo_True {
                    let p = evt._buf.as_ptr();
                    let etype = unsafe { (p as *const i64).read() };
                    if etype == varjo_EventType_StandbyStatus {
                        let on_standby =
                            unsafe { (p.add(VARJO_EVENT_PAYLOAD_OFFSET) as *const i32).read() }
                                != 0;
                        if on_standby {
                            pause_now!("standby — paused");
                            paused_this_iter = true;
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

                if last_presence.elapsed() >= Duration::from_millis(50) {
                    last_presence = Instant::now();
                    unsafe { sync_props(session) };
                    let worn = (unsafe { get_bool(session, varjo_PropertyKey_UserPresence) })
                        == varjo_True;
                    if !worn {
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

                let retry_ok = retry_after.map(|t| Instant::now() >= t).unwrap_or(true);
                if present && !blocked && retry_ok && !paused_this_iter && !stream_running {
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
                    let _ = unsafe { (lib.get_error)(session) };
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
                        stream_live.store(true, Ordering::Release);
                        retry_after = None;
                        last_evidence = Instant::now();
                        last_frames = frame_count.load(Ordering::Relaxed);
                        set_status(
                            &status,
                            format!("streaming {}x{} eye camera", eye_cfg.width, eye_cfg.height),
                        );
                        eprintln!("[varjo] headset on — streaming eye camera id={stream_id}");
                    } else {
                        unsafe { (lib.stop_stream)(session, stream_id) };
                        retry_after = Some(Instant::now() + Duration::from_secs(2));
                        set_status(
                            &status,
                            format!("stream start failed (varjo error {err}) — retrying"),
                        );
                        eprintln!("[varjo] StartDataStream error {err} — backing off 2s");
                    }
                }

                if stream_running {
                    let now_frames = frame_count.load(Ordering::Relaxed);
                    if now_frames != last_frames {
                        last_frames = now_frames;
                        last_evidence = Instant::now();
                    }
                    if last_evidence.elapsed() >= Duration::from_secs(3) {
                        stream_live.store(false, Ordering::Release);
                        unsafe { (lib.stop_stream)(session, stream_id) };
                        stream_running = false;
                        blocked = true;
                        set_status(&status, "no frames — paused (re-seat the headset)");
                        eprintln!("[varjo] no frames 3s — paused (stall latch)");
                    }
                }

                if stream_running && gaze_active {
                    if let Some(get_gaze) = lib.get_gaze {
                        let gz = unsafe { get_gaze(session) };
                        if let Some(callback) = &mut on_raw_gaze {
                            if let Some(sample) = raw_gaze_poll.sample(&gz) {
                                callback(sample);
                            }
                        }
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
                                pupil_pos_reported: false,
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

                thread::sleep(Duration::from_millis(if stream_running { 5 } else { 50 }));
            }

            stream_live.store(false, Ordering::Release);
            if ever_started {
                unsafe { (lib.stop_stream)(session, stream_id) };
            }
            unsafe { (lib.session_shutdown)(session) };
            drop(ctx);
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

#[cfg(test)]
mod raw_gaze_tests {
    use super::*;

    fn gaze() -> varjo_Gaze {
        varjo_Gaze {
            left_eye: varjo_Ray {
                origin: [-0.032, 0.0, 0.0],
                forward: [0.8, 0.0, 0.6],
            },
            right_eye: varjo_Ray {
                origin: [0.032, 0.0, 0.0],
                forward: [-0.6, 0.0, 0.8],
            },
            gaze: varjo_Ray::default(),
            focus_distance: 0.0,
            stability: 0.0,
            capture_time: 12_345_000,
            left_status: varjo_GazeEyeStatus_Tracked,
            right_status: varjo_GazeEyeStatus_Compensated,
            status: varjo_GazeStatus_Valid,
            frame_number: 123,
            left_pupil_size: 0.0,
            right_pupil_size: 0.0,
        }
    }

    #[test]
    fn raw_eyes_preserve_native_directions_and_have_no_fabricated_openness() {
        let native = gaze();
        let sample = RawGazePoll::default().sample(&native).unwrap();
        assert_eq!(sample.timestamp_us, 12_345);
        assert_eq!(sample.left.gaze, [0.8, 0.0, 0.6]);
        assert_eq!(sample.right.gaze, [-0.6, 0.0, 0.8]);
        for eye in [sample.left, sample.right] {
            assert!(eye.gaze_valid && eye.gaze_reported);
            assert!(!eye.openness_valid && !eye.openness_reported);
        }
    }

    #[test]
    fn raw_validity_rejects_visible_only_and_calibrating_states() {
        let mut native = gaze();
        native.left_status = varjo_GazeEyeStatus_Visible;
        let mut poll = RawGazePoll::default();
        let sample = poll.sample(&native).unwrap();
        assert!(!sample.left.gaze_valid);
        assert!(sample.right.gaze_valid);
        native.status = varjo_GazeStatus_Adjust;
        let sample = poll.sample(&native).unwrap();
        assert!(!sample.left.gaze_valid && !sample.right.gaze_valid);
        native.status = varjo_GazeStatus_Invalid;
        let sample = poll.sample(&native).unwrap();
        assert!(!sample.left.gaze_valid && !sample.right.gaze_valid);
    }

    #[test]
    fn raw_poll_deduplicates_capture_but_keeps_same_frame_status_changes() {
        let mut native = gaze();
        let mut poll = RawGazePoll::default();
        assert!(poll.sample(&native).is_some());
        assert!(poll.sample(&native).is_none());
        native.right_status = varjo_GazeEyeStatus_Invalid;
        assert!(!poll.sample(&native).unwrap().right.gaze_valid);
        assert!(poll.sample(&native).is_none());
        native.capture_time += 1;
        assert!(poll.sample(&native).is_some());
        native.status = varjo_GazeStatus_Adjust;
        assert!(poll.sample(&native).is_some());
        native.left_status = varjo_GazeEyeStatus_Invalid;
        assert!(poll.sample(&native).is_some());
        native.capture_time = 0;
        assert!(poll.sample(&native).is_some());
        assert!(poll.sample(&native).is_none());
    }

    #[test]
    fn native_varjo_accepts_the_dedicated_raw_hook_before_start() {
        let mut adapter = VarjoNativeAdapter::new(String::new());
        assert!(adapter.set_raw_gaze_callback(Box::new(|_| {})));
        assert!(adapter.raw_gaze_callback.is_some());
    }
}
