//! Exercises the real pipeline's native gaze tap using only an in-memory adapter
//! and local UDP receivers. No device, model, GPU, UI, or installed config is used.

use std::io;
use std::net::{Ipv4Addr, UdpSocket};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use sranibro_rs::core::types::{DeviceProfile, EyeSample, GazeSample};
use sranibro_rs::device::{FrameFn, GazeFn, HmdAdapter};
use sranibro_rs::pipeline::{DeviceMap, Pipeline, PipelineInit};

fn isolated_state() {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = std::env::temp_dir().join(format!(
            "sranibro-raw-gaze-pipeline-tests-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();
        // This integration test is its own process. Every test passes through
        // this OnceLock before the library resolves its cached state directory.
        std::env::set_var("APPDATA", &root);
        std::env::set_var("LOCALAPPDATA", &root);
        std::env::set_var("XDG_CONFIG_HOME", &root);
        std::env::set_var("HOME", &root);
        assert_eq!(sranibro_rs::config::base_dir(), root.join("SRanibro"));
        root
    });
}

#[derive(Default)]
struct Callbacks {
    regular: Option<GazeFn>,
    raw: Option<GazeFn>,
}

struct FakeAdapter {
    profile: DeviceProfile,
    callbacks: Arc<Mutex<Callbacks>>,
    dedicated: bool,
    stopped: Arc<AtomicBool>,
}

impl HmdAdapter for FakeAdapter {
    fn name(&self) -> &'static str {
        "raw-gaze-test"
    }

    fn profile(&self) -> &DeviceProfile {
        &self.profile
    }

    fn start(&mut self, _on_frame: FrameFn, on_gaze: GazeFn) -> io::Result<()> {
        self.callbacks.lock().unwrap().regular = Some(on_gaze);
        Ok(())
    }

    fn set_raw_gaze_callback(&mut self, callback: GazeFn) -> bool {
        if self.dedicated {
            self.callbacks.lock().unwrap().raw = Some(callback);
        }
        self.dedicated
    }

    fn stop(&mut self) {
        // Retain callbacks deliberately: tests also simulate a late callback
        // after stop and verify that the pipeline's output guard suppresses it.
        self.stopped.store(true, Ordering::Release);
    }
}

struct Harness {
    pipeline: Pipeline,
    callbacks: Arc<Mutex<Callbacks>>,
    stopped: Arc<AtomicBool>,
}

impl Harness {
    fn new(device: &str, dedicated: bool, map: DeviceMap, port: Option<u16>) -> Self {
        Self::with_init(
            device,
            dedicated,
            map,
            PipelineInit {
                raw_gaze_osc_port: port,
                ..PipelineInit::default()
            },
        )
    }

    fn with_init(device: &str, dedicated: bool, map: DeviceMap, init: PipelineInit) -> Self {
        isolated_state();
        let callbacks = Arc::new(Mutex::new(Callbacks::default()));
        let stopped = Arc::new(AtomicBool::new(false));
        let pipeline = Pipeline::run(
            Box::new(FakeAdapter {
                profile: DeviceProfile::default(),
                callbacks: callbacks.clone(),
                dedicated,
                stopped: stopped.clone(),
            }),
            None,
            None,
            None,
            Vec::new(),
            map,
            Arc::new(Mutex::new(String::new())),
            device.into(),
            init,
        )
        .unwrap();
        assert!(
            !pipeline.tele.ml_loaded,
            "no calibration persistence without ML"
        );
        Self {
            pipeline,
            callbacks,
            stopped,
        }
    }

    fn regular(&self, sample: GazeSample) {
        self.callbacks.lock().unwrap().regular.as_mut().unwrap()(sample);
    }

    fn raw(&self, sample: GazeSample) {
        self.callbacks.lock().unwrap().raw.as_mut().unwrap()(sample);
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.pipeline.stop();
    }
}

fn receiver() -> UdpSocket {
    UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap()
}

fn port(socket: &UdpSocket) -> u16 {
    socket.local_addr().unwrap().port()
}

#[derive(Debug)]
struct Message {
    address: String,
    direction: [f32; 3],
    valid: i32,
    openness: f32,
    counter: i64,
}

fn osc_string<'a>(packet: &'a [u8], offset: &mut usize) -> &'a str {
    let end = *offset + packet[*offset..].iter().position(|v| *v == 0).unwrap();
    let value = std::str::from_utf8(&packet[*offset..end]).unwrap();
    let next = (end + 4) & !3;
    assert!(packet[end..next].iter().all(|v| *v == 0));
    *offset = next;
    value
}

fn receive(socket: &UdpSocket) -> Message {
    socket
        .set_read_timeout(Some(Duration::from_millis(700)))
        .unwrap();
    let mut packet = [0u8; 256];
    let (len, from) = socket.recv_from(&mut packet).unwrap();
    assert!(from.ip().is_loopback());
    assert_ne!(from.port(), 0);
    let packet = &packet[..len];
    let mut offset = 0;
    let address = osc_string(packet, &mut offset).to_owned();
    assert_eq!(osc_string(packet, &mut offset), ",fffifh");
    assert_eq!(packet.len() - offset, 28);
    let float = |start| f32::from_be_bytes(packet[start..start + 4].try_into().unwrap());
    Message {
        address,
        direction: [float(offset), float(offset + 4), float(offset + 8)],
        valid: i32::from_be_bytes(packet[offset + 12..offset + 16].try_into().unwrap()),
        openness: float(offset + 16),
        counter: i64::from_be_bytes(packet[offset + 20..offset + 28].try_into().unwrap()),
    }
}

fn pair(socket: &UdpSocket) -> [Message; 2] {
    let mut received = [receive(socket), receive(socket)];
    received.sort_by(|a, b| a.address.cmp(&b.address));
    assert_eq!(received[0].address, "/sranibro/gaze/left");
    assert_eq!(received[1].address, "/sranibro/gaze/right");
    assert_eq!(received[0].counter, received[1].counter);
    assert!(received[0].counter > 0);
    received
}

fn quiet(socket: &UdpSocket) {
    socket
        .set_read_timeout(Some(Duration::from_millis(70)))
        .unwrap();
    let error = socket
        .recv(&mut [0u8; 256])
        .expect_err("unexpected raw gaze packet");
    assert!(matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
    ));
}

fn eye(gaze: [f32; 3]) -> EyeSample {
    EyeSample {
        gaze,
        gaze_valid: true,
        gaze_reported: true,
        ..EyeSample::default()
    }
}

fn stereo() -> GazeSample {
    GazeSample {
        timestamp_us: 9_876_543_210,
        left: eye([0.6, 0.0, 0.8]),
        right: eye([0.0, 0.8, 0.6]),
    }
}

fn direction(actual: [f32; 3], expected: [f32; 3]) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() < 0.000001,
            "actual {actual:?}, expected {expected:?}"
        );
    }
}

#[test]
fn output_is_opt_in_and_live_enable_disable_needs_no_reload() {
    let socket = receiver();
    let harness = Harness::new("psvr2", false, DeviceMap::default(), None);
    assert_eq!(harness.pipeline.raw_gaze_osc_port(), None);
    harness.regular(stereo());
    quiet(&socket);

    harness
        .pipeline
        .set_raw_gaze_osc(Some(port(&socket)))
        .unwrap();
    assert_eq!(harness.pipeline.raw_gaze_osc_port(), Some(port(&socket)));
    harness.regular(stereo());
    assert_eq!(pair(&socket)[0].counter, 1);

    harness.pipeline.set_raw_gaze_osc(None).unwrap();
    assert_eq!(harness.pipeline.raw_gaze_osc_port(), None);
    harness.regular(stereo());
    quiet(&socket);
    assert!(!harness.stopped.load(Ordering::Acquire));
}

#[test]
fn psvr2_exports_native_coordinates_and_binary_blink_without_ml() {
    let socket = receiver();
    let harness = Harness::new("psvr2", false, DeviceMap::default(), Some(port(&socket)));
    let mut sample = stereo();
    sample.left.openness_reported = true;
    sample.left.openness_valid = true;
    sample.left.openness = 1.0;
    // This is the actual PSVR2 adapter's known-closed classifier representation.
    sample.right.openness_reported = true;
    sample.right.openness_valid = false;
    sample.right.openness = 0.0;
    harness.regular(sample);
    let [left, right] = pair(&socket);
    direction(left.direction, [-0.6, 0.0, -0.8]);
    direction(right.direction, [0.0, 0.8, -0.6]);
    assert_eq!((left.valid, right.valid), (1, 1));
    assert_eq!((left.openness, right.openness), (1.0, 0.0));
    assert_ne!(left.counter as u64, sample.timestamp_us);
    // Loss of PSVR2's explicit blink-valid bit is not an auxiliary-only gap:
    // both formerly open and formerly closed values become unknown immediately.
    sample.left.openness_reported = false;
    sample.right.openness_reported = false;
    harness.regular(sample);
    let [left, right] = pair(&socket);
    assert_eq!((left.openness, right.openness), (-1.0, -1.0));
    assert_eq!((left.valid, right.valid), (1, 1));
}

#[test]
fn varjo_axes_and_avatar_gaze_flip_remain_independent() {
    let socket = receiver();
    let harness = Harness::new(
        "varjo",
        false,
        DeviceMap {
            flip_gaze_x: true,
            ..DeviceMap::default()
        },
        Some(port(&socket)),
    );
    harness.regular(stereo());
    let [left, right] = pair(&socket);
    direction(left.direction, [0.6, 0.0, -0.8]);
    direction(right.direction, [0.0, 0.8, -0.6]);
    assert_eq!((left.openness, right.openness), (-1.0, -1.0));
}

#[test]
fn physical_eye_swap_routes_the_complete_native_eye_once() {
    let socket = receiver();
    let harness = Harness::new(
        "psvr2",
        false,
        DeviceMap {
            swap_eyes: true,
            flip_image: true,
            flip_gaze_x: true,
        },
        Some(port(&socket)),
    );
    let mut sample = stereo();
    sample.left.openness_reported = true;
    sample.left.openness_valid = true;
    sample.left.openness = 1.0;
    sample.right.openness_reported = true;
    sample.right.openness_valid = false;
    harness.regular(sample);
    let [left, right] = pair(&socket);
    direction(left.direction, [0.0, 0.8, -0.6]);
    direction(right.direction, [-0.6, 0.0, -0.8]);
    assert_eq!((left.openness, right.openness), (0.0, 1.0));
}

#[test]
fn dedicated_per_eye_tap_bypasses_the_combined_avatar_callback() {
    let socket = receiver();
    let harness = Harness::new("pimax_xr5", true, DeviceMap::default(), Some(port(&socket)));
    let combined = GazeSample {
        left: eye([0.0, 0.0, 1.0]),
        right: eye([0.0, 0.0, 1.0]),
        ..GazeSample::default()
    };
    harness.regular(combined);
    quiet(&socket);
    harness.raw(stereo());
    let [left, right] = pair(&socket);
    direction(left.direction, [-0.6, 0.0, -0.8]);
    direction(right.direction, [0.0, 0.8, -0.6]);
    assert_eq!(left.counter, 1);
    let merged = *harness.pipeline.tele.gaze.lock().unwrap();
    assert_eq!(merged.left.gaze, combined.left.gaze);
    assert_eq!(merged.right.gaze, combined.right.gaze);
    quiet(&socket);
}

#[test]
fn invalid_or_unreported_eye_never_reuses_merged_last_good_gaze() {
    let socket = receiver();
    let harness = Harness::new("psvr2", false, DeviceMap::default(), Some(port(&socket)));
    harness.regular(stereo());
    pair(&socket);
    let mut missing = stereo();
    missing.left = EyeSample::default();
    harness.regular(missing);
    let [left, right] = pair(&socket);
    assert_eq!(left.valid, 0);
    assert_eq!(left.direction, [0.0; 3]);
    assert_eq!(right.valid, 1);
    // The ordinary telemetry merge intentionally retains an auxiliary-unreported
    // eye. The raw datagram must not borrow that retained valid sample.
    assert!(harness.pipeline.tele.gaze.lock().unwrap().left.gaze_valid);

    let mut invalid = stereo();
    invalid.right.gaze_valid = false;
    invalid.right.gaze = [f32::NAN, f32::INFINITY, 0.0];
    harness.regular(invalid);
    let [left, right] = pair(&socket);
    assert_eq!((left.valid, right.valid), (1, 0));
    assert_eq!(right.direction, [0.0; 3]);
}

#[test]
fn auxiliary_native_openness_does_not_duplicate_gaze_packets() {
    let socket = receiver();
    let harness = Harness::new("pimax_vr4", true, DeviceMap::default(), Some(port(&socket)));
    let mut auxiliary = GazeSample::default();
    auxiliary.left.openness_reported = true;
    auxiliary.left.openness_valid = true;
    auxiliary.left.openness = 0.25;
    auxiliary.right.openness_reported = true;
    auxiliary.right.openness_valid = true;
    auxiliary.right.openness = 0.75;
    harness.raw(auxiliary);
    quiet(&socket);
    // Refresh after the quiet check, so assertion is not sensitive to a slow
    // scheduler crossing the sender's native-openness freshness deadline.
    harness.raw(auxiliary);
    harness.raw(stereo());
    let [left, right] = pair(&socket);
    assert_eq!(left.counter, 1);
    assert_eq!((left.openness, right.openness), (0.25, 0.75));
}

#[test]
fn bad_port_preserves_existing_feed_and_live_switch_stops_old_destination() {
    let old = receiver();
    let new = receiver();
    let harness = Harness::new("psvr2", false, DeviceMap::default(), Some(port(&old)));
    assert_eq!(
        harness
            .pipeline
            .set_raw_gaze_osc(Some(0))
            .unwrap_err()
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert_eq!(harness.pipeline.raw_gaze_osc_port(), Some(port(&old)));
    harness.regular(stereo());
    pair(&old);

    harness.pipeline.set_raw_gaze_osc(Some(port(&new))).unwrap();
    harness.regular(stereo());
    assert_eq!(pair(&new)[0].counter, 1);
    quiet(&old);
    assert!(!harness.stopped.load(Ordering::Acquire));
}

#[test]
fn pause_and_stop_suppress_even_late_callbacks() {
    let socket = receiver();
    let mut harness = Harness::new("psvr2", false, DeviceMap::default(), Some(port(&socket)));
    harness.regular(stereo());
    pair(&socket);
    harness.pipeline.paused.store(true, Ordering::Relaxed);
    harness.regular(stereo());
    quiet(&socket);
    harness.pipeline.paused.store(false, Ordering::Relaxed);
    harness.regular(stereo());
    assert_eq!(pair(&socket)[0].counter, 2);
    harness.pipeline.stop();
    assert!(harness.stopped.load(Ordering::Acquire));
    assert_eq!(harness.pipeline.raw_gaze_osc_port(), None);
    harness.regular(stereo());
    quiet(&socket);
    assert_eq!(
        harness
            .pipeline
            .set_raw_gaze_osc(Some(port(&socket)))
            .unwrap_err()
            .kind(),
        io::ErrorKind::NotConnected
    );
}

#[test]
fn every_native_callback_emits_once_without_a_postprocess_tick_or_rate_cap() {
    let socket = receiver();
    let harness = Harness::new("psvr2", false, DeviceMap::default(), Some(port(&socket)));
    // No wall-clock sleep or device-timestamp change between callbacks. A paced,
    // timestamp-deduplicated, or latest-only emit loop cannot pass this assertion.
    for _ in 0..16 {
        harness.regular(stereo());
    }
    for counter in 1..=16 {
        assert_eq!(pair(&socket)[0].counter, counter);
    }
    quiet(&socket);
}

#[test]
fn optional_gaze_smoothing_survives_live_osc_enable_and_swap_without_rate_limiting() {
    use sranibro_rs::config::Config;
    use sranibro_rs::core::gaze_smoothing::{adaptive_gaze_step, INITIAL_GAZE};
    let sample = GazeSample {
        left: EyeSample {
            gaze: [0.6, 0.0, 0.8],
            gaze_reported: true,
            gaze_valid: true,
            ..Default::default()
        },
        right: EyeSample {
            gaze: [-0.4, 0.1, 0.9],
            gaze_reported: true,
            gaze_valid: true,
            ..Default::default()
        },
        ..Default::default()
    };
    for enabled in [true, false] {
        let mut config = Config::default();
        config.hmd.vr4_gaze_smoothing = enabled;
        let (_, init) = sranibro_rs::engine::pipeline_start_settings(&config, "pimax_vr4");
        let harness = Harness::with_init("pimax_vr4", true, DeviceMap::default(), init);
        let socket = receiver();
        // The output was OFF at startup; a later Output toggle must retain the
        // applied smoothing choice, as must a port change/re-enable.
        for _ in 0..2 {
            harness
                .pipeline
                .set_raw_gaze_osc(Some(port(&socket)))
                .unwrap();
            harness.pipeline.swap_eyes.store(false, Ordering::Relaxed);
            let mut history = [INITIAL_GAZE; 2];
            for counter in 1..=16 {
                harness.raw(sample);
                let messages = pair(&socket);
                for (eye, source) in [sample.left, sample.right].into_iter().enumerate() {
                    let filtered = if enabled {
                        adaptive_gaze_step(history[eye], source.gaze)
                    } else {
                        source.gaze
                    };
                    history[eye] = filtered;
                    let len = filtered[0].hypot(filtered[1]).hypot(filtered[2]);
                    direction(
                        messages[eye].direction,
                        [-filtered[0] / len, filtered[1] / len, -filtered[2] / len],
                    );
                    assert_eq!(messages[eye].counter, counter);
                    assert_eq!(messages[eye].valid, 1);
                }
            }
            harness.pipeline.swap_eyes.store(true, Ordering::Relaxed);
            harness.raw(sample);
            let messages = pair(&socket);
            for (eye, source) in [sample.right, sample.left].into_iter().enumerate() {
                let filtered = if enabled {
                    adaptive_gaze_step(INITIAL_GAZE, source.gaze)
                } else {
                    source.gaze
                };
                let len = filtered[0].hypot(filtered[1]).hypot(filtered[2]);
                direction(
                    messages[eye].direction,
                    [-filtered[0] / len, filtered[1] / len, -filtered[2] / len],
                );
                assert_eq!(messages[eye].counter, 17);
            }
            let mut lost = sample;
            lost.left.gaze_valid = false;
            harness.raw(lost);
            let messages = pair(&socket);
            assert_eq!((messages[0].valid, messages[1].valid), (1, 0));
            assert_eq!(messages[1].direction, [0.0; 3]);
            harness.pipeline.set_raw_gaze_osc(None).unwrap();
            harness.raw(sample);
            quiet(&socket);
        }
    }
}

#[test]
fn crystal_source_choice_reaches_avatar_input_and_dedicated_osc_without_mixing() {
    use sranibro_rs::config::{Config, Vr4GazeSource};
    use sranibro_rs::device::gaze::{GazeData, WearableData};
    use sranibro_rs::device::vr4_gaze_source::Vr4GazeRouter;

    let native = WearableData {
        timestamp_us: 100,
        gaze: [[0.64278764, 0.0, 0.76604444], [-0.6, 0.0, 0.8]],
        gaze_valid: [true; 2],
        openness: [0.25, 0.75],
        openness_valid: [true; 2],
        openness_reported: [true; 2],
        ..Default::default()
    };
    let old = GazeData {
        timestamp: 100,
        l_gaze: [0.5, 0.0, 0.8660254],
        r_gaze: [-0.5, 0.0, 0.8660254],
        combined: [0.0, 0.0, 1.0],
        ..Default::default()
    };
    // Reconstruct both pipeline and router as Apply & reload does, including
    // switching back. No process environment override, sensor, ML or GPU needed.
    for source in [Vr4GazeSource::V1, Vr4GazeSource::V2, Vr4GazeSource::V1] {
        let mut config = Config::default();
        config.hmd.vr4_gaze_source = source;
        config.hmd.vr4_gaze_smoothing = false;
        let restored: Config = toml::from_str(&toml::to_string(&config).unwrap()).unwrap();
        let socket = receiver();
        let (_, mut init) = sranibro_rs::engine::pipeline_start_settings(&restored, "pimax_vr4");
        init.raw_gaze_osc_port = Some(port(&socket));
        let harness = Harness::with_init("pimax_vr4", true, DeviceMap::default(), init);
        let mut router = Vr4GazeRouter::new(restored.hmd.vr4_gaze_source);
        let (raw, tracking) = router.wearable(&native);
        harness.raw(raw);
        harness.regular(tracking);
        if let Some(sample) = router.foveated(&old) {
            harness.raw(sample);
            harness.regular(sample);
        }
        let [left, right] = pair(&socket);
        let expected = if source == Vr4GazeSource::V1 {
            native.gaze
        } else {
            [old.l_gaze, old.r_gaze]
        };
        direction(
            left.direction,
            [-expected[0][0], expected[0][1], -expected[0][2]],
        );
        direction(
            right.direction,
            [-expected[1][0], expected[1][1], -expected[1][2]],
        );
        assert_eq!((left.valid, right.valid), (1, 1));
        assert_eq!((left.openness, right.openness), (0.25, 0.75));
        assert_eq!(
            left.counter, 1,
            "one selected provider, not duplicate callbacks"
        );
        let gaze = *harness.pipeline.tele.gaze.lock().unwrap();
        direction(gaze.left.gaze, expected[0]);
        direction(gaze.right.gaze, expected[1]);
        quiet(&socket);

        if source == Vr4GazeSource::V1 {
            let mut lost = native;
            lost.timestamp_us += 1;
            lost.gaze_valid[0] = false;
            let (raw, tracking) = router.wearable(&lost);
            harness.raw(raw);
            harness.regular(tracking);
            assert!(
                router.foveated(&old).is_none(),
                "must not borrow v2 on loss"
            );
            let [left, right] = pair(&socket);
            assert_eq!((left.valid, right.valid), (0, 1));
            assert_eq!(left.direction, [0.0; 3]);
            quiet(&socket);
        }
    }
}
