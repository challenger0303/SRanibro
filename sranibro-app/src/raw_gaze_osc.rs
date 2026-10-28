//! Loopback-only OSC output for native, pre-avatar gaze samples.
//!
//! This protocol deliberately carries the two eyes independently.  It does not
//! apply avatar gaze strength, angular clamps, yoking, or synthesize a missing
//! eye. VR4 may opt into the avatar's legacy adaptive EMA; it runs once per native
//! gaze callback, independently per eye. The final OSC `h` argument is a counter, not
//! the device timestamp; both eye messages from one gaze callback share it.
//! The opt-in XR5 research provider instead supplies freshness-checked model rays
//! before avatar trim. Native callbacks are disconnected from this publisher in
//! that mode. Unknown native openness remains -1; invalid rays stay invalid.

use std::io;
use std::net::{Ipv4Addr, SocketAddrV4, UdpSocket};
use std::time::{Duration, Instant};

use crate::core::gaze_smoothing::{adaptive_gaze_step, INITIAL_GAZE};
use crate::core::types::{EyeSample, GazeSample};

pub const LEFT_ADDRESS: &str = "/sranibro/gaze/left";
pub const RIGHT_ADDRESS: &str = "/sranibro/gaze/right";

const OPENNESS_FRESHNESS: Duration = Duration::from_millis(150);
const UNKNOWN_OPENNESS: f32 = -1.0;
const OSC_TAGS: &str = ",fffifh";
const MAX_PACKET_LEN: usize = 64;

/// Native tracker coordinate convention to convert into the wire convention:
/// +X right, +Y up, and -Z forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RawGazeAxes {
    /// Tobii wearable gaze is +X left, +Y up, +Z forward.
    Tobii,
    /// Varjo gaze (not its matrix convention) is +X right, +Y up, +Z forward.
    Varjo,
}

impl RawGazeAxes {
    #[inline]
    fn to_wire(self, [x, y, z]: [f32; 3]) -> [f32; 3] {
        match self {
            Self::Tobii => [-x, y, -z],
            Self::Varjo => [x, y, -z],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct CachedOpenness {
    value: f32,
    observed_at: Instant,
}

/// Best-effort native gaze publisher with a fixed loopback destination.
///
/// The socket is deliberately unconnected: on Windows this prevents an ICMP
/// "port unreachable" response from poisoning later UDP sends.  It is also
/// nonblocking, so a missing receiver cannot stall a camera callback.
pub struct RawGazeOsc {
    socket: UdpSocket,
    destination: SocketAddrV4,
    openness: [Option<CachedOpenness>; 2],
    sample_counter: i64,
    gaze_smoothing: bool,
    gaze_history: [[f32; 3]; 2],
}

impl RawGazeOsc {
    pub fn new(port: u16) -> io::Result<Self> {
        if port == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "raw gaze OSC destination port must not be zero",
            ));
        }
        let socket = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0))?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket,
            destination: SocketAddrV4::new(Ipv4Addr::LOCALHOST, port),
            openness: [None, None],
            sample_counter: 0,
            gaze_smoothing: false,
            gaze_history: [INITIAL_GAZE; 2],
        })
    }

    pub fn port(&self) -> u16 {
        self.destination.port()
    }

    pub fn with_gaze_smoothing(mut self, enabled: bool) -> Self {
        self.gaze_smoothing = enabled;
        self
    }

    /// A physical L/R routing change invalidates all values cached under
    /// the previous eye identities. It must not reset the gaze pair counter.
    pub fn clear_eye_history(&mut self) {
        self.openness = [None, None];
        self.gaze_history = [INITIAL_GAZE; 2];
    }

    /// Consume one adapter callback. Auxiliary-only callbacks refresh native
    /// openness but produce no packets. A gaze callback always attempts both
    /// eye datagrams, even if the first send fails.
    pub fn on_sample(&mut self, sample: GazeSample, axes: RawGazeAxes) -> io::Result<()> {
        self.on_sample_at(sample, axes, Instant::now())
    }

    fn on_sample_at(
        &mut self,
        sample: GazeSample,
        axes: RawGazeAxes,
        now: Instant,
    ) -> io::Result<()> {
        self.update_openness(0, &sample.left, now);
        self.update_openness(1, &sample.right, now);

        if !sample.left.gaze_reported && !sample.right.gaze_reported {
            return Ok(());
        }

        self.sample_counter = self.sample_counter.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "raw gaze OSC sample counter exhausted",
            )
        })?;
        let counter = self.sample_counter;

        let left_eye = self.smooth_eye(0, sample.left);
        let right_eye = self.smooth_eye(1, sample.right);
        let left = gaze_payload(&left_eye, axes, self.fresh_openness(0, now), counter);
        let right = gaze_payload(&right_eye, axes, self.fresh_openness(1, now), counter);
        let mut left_packet = [0_u8; MAX_PACKET_LEN];
        let mut right_packet = [0_u8; MAX_PACKET_LEN];
        let left_len = encode_packet(LEFT_ADDRESS, left, &mut left_packet);
        let right_len = encode_packet(RIGHT_ADDRESS, right, &mut right_packet);

        let left_result = self
            .socket
            .send_to(&left_packet[..left_len], self.destination);
        let right_result = self
            .socket
            .send_to(&right_packet[..right_len], self.destination);
        match (left_result, right_result) {
            (Err(error), _) => Err(error),
            (_, Err(error)) => Err(error),
            (Ok(_), Ok(_)) => Ok(()),
        }
    }

    fn update_openness(&mut self, index: usize, eye: &EyeSample, now: Instant) {
        if !eye.openness_reported {
            return;
        }
        self.openness[index] =
            (eye.openness_valid && eye.openness.is_finite()).then_some(CachedOpenness {
                value: eye.openness.clamp(0.0, 1.0),
                observed_at: now,
            });
    }

    fn smooth_eye(&mut self, index: usize, mut eye: EyeSample) -> EyeSample {
        // A loss/auxiliary packet must remain invalid and never poison history.
        // Smooth before normalization, exactly as in the avatar post-processor.
        if self.gaze_smoothing && valid_gaze_length(&eye).is_some() {
            eye.gaze = adaptive_gaze_step(self.gaze_history[index], eye.gaze);
            self.gaze_history[index] = eye.gaze;
        }
        eye
    }

    fn fresh_openness(&self, index: usize, now: Instant) -> f32 {
        self.openness[index]
            .filter(|cached| {
                now.saturating_duration_since(cached.observed_at) <= OPENNESS_FRESHNESS
            })
            .map(|cached| cached.value)
            .unwrap_or(UNKNOWN_OPENNESS)
    }
}

#[derive(Clone, Copy, Debug)]
struct GazePayload {
    direction: [f32; 3],
    valid: i32,
    openness: f32,
    counter: i64,
}

fn valid_gaze_length(eye: &EyeSample) -> Option<f32> {
    let finite = eye.gaze.iter().all(|component| component.is_finite());
    let length = if finite {
        eye.gaze[0].hypot(eye.gaze[1]).hypot(eye.gaze[2])
    } else {
        0.0
    };
    let valid = eye.gaze_reported && eye.gaze_valid && length.is_finite() && length > 1.0e-6;
    valid.then_some(length)
}

fn gaze_payload(eye: &EyeSample, axes: RawGazeAxes, openness: f32, counter: i64) -> GazePayload {
    let length = valid_gaze_length(eye);
    let direction = if let Some(length) = length {
        let normalized = eye.gaze.map(|component| component / length);
        axes.to_wire(normalized)
    } else {
        [0.0; 3]
    };
    GazePayload {
        direction,
        valid: i32::from(length.is_some()),
        openness,
        counter,
    }
}

fn append_osc_string(buffer: &mut [u8], offset: &mut usize, value: &str) {
    let bytes = value.as_bytes();
    buffer[*offset..*offset + bytes.len()].copy_from_slice(bytes);
    *offset += bytes.len();
    buffer[*offset] = 0;
    *offset += 1;
    while *offset % 4 != 0 {
        buffer[*offset] = 0;
        *offset += 1;
    }
}

fn encode_packet(address: &str, payload: GazePayload, buffer: &mut [u8; MAX_PACKET_LEN]) -> usize {
    let mut offset = 0;
    append_osc_string(buffer, &mut offset, address);
    append_osc_string(buffer, &mut offset, OSC_TAGS);
    for value in payload.direction {
        buffer[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
        offset += 4;
    }
    buffer[offset..offset + 4].copy_from_slice(&payload.valid.to_be_bytes());
    offset += 4;
    buffer[offset..offset + 4].copy_from_slice(&payload.openness.to_be_bytes());
    offset += 4;
    buffer[offset..offset + 8].copy_from_slice(&payload.counter.to_be_bytes());
    offset + 8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_alloc::count_current_thread_allocations;
    use std::net::UdpSocket;

    #[cfg(feature = "external-eyelid-model")]
    #[test]
    fn model_gaze_loopback_keeps_worker_axes_and_expires() {
        use crate::core::types::{Eye,EyeResult};
        use crate::ml::xr5_gaze::Snapshot;
        let receiver=UdpSocket::bind(("127.0.0.1",0)).unwrap();
        receiver.set_read_timeout(Some(Duration::from_secs(1))).unwrap();
        let mut sender=RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        let now=Instant::now();
        let snap=Snapshot {states:[1,1],directions:[[0.6,0.,-0.8],[-0.6,0.,-0.8]],captured:Some(now),..Default::default()};
        let mut eyes=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
        for (time,valid) in [(now,1),(now+Duration::from_millis(51),0)] {
            snap.apply(&mut eyes,time,true);
            sender.on_sample(snap.raw_sample(&eyes),RawGazeAxes::Tobii).unwrap();
            for eye in 0..2 {
                let mut bytes=[0;64];let size=receiver.recv(&mut bytes).unwrap();let value=decode(&bytes[..size]);
                assert_eq!(value.3,valid);assert_eq!(value.4,-1.);
                assert_eq!(value.2,if valid==1 {snap.directions[eye]} else {[0.;3]});
            }
        }
    }

    fn reported(gaze: [f32; 3]) -> EyeSample {
        EyeSample {
            gaze,
            gaze_valid: true,
            gaze_reported: true,
            ..EyeSample::default()
        }
    }

    fn decode(packet: &[u8]) -> (&str, &str, [f32; 3], i32, f32, i64) {
        fn osc_string<'a>(packet: &'a [u8], offset: &mut usize) -> &'a str {
            let start = *offset;
            let end = packet[start..]
                .iter()
                .position(|byte| *byte == 0)
                .map(|relative| start + relative)
                .unwrap();
            *offset = (end + 4) & !3;
            std::str::from_utf8(&packet[start..end]).unwrap()
        }
        fn take<const N: usize>(packet: &[u8], offset: &mut usize) -> [u8; N] {
            let bytes = packet[*offset..*offset + N].try_into().unwrap();
            *offset += N;
            bytes
        }
        let mut offset = 0;
        let address = osc_string(packet, &mut offset);
        let tags = osc_string(packet, &mut offset);
        let direction = std::array::from_fn(|_| f32::from_be_bytes(take(packet, &mut offset)));
        let valid = i32::from_be_bytes(take(packet, &mut offset));
        let openness = f32::from_be_bytes(take(packet, &mut offset));
        let counter = i64::from_be_bytes(take(packet, &mut offset));
        assert_eq!(offset, packet.len());
        (address, tags, direction, valid, openness, counter)
    }

    fn packet(address: &str, payload: GazePayload) -> Vec<u8> {
        let mut bytes = [0_u8; MAX_PACKET_LEN];
        let len = encode_packet(address, payload, &mut bytes);
        bytes[..len].to_vec()
    }

    #[test]
    fn exact_wire_layout_has_expected_address_alignment_and_types() {
        let bytes = packet(
            LEFT_ADDRESS,
            GazePayload {
                direction: [1.0, -0.5, 0.25],
                valid: 1,
                openness: 0.75,
                counter: 0x0102_0304_0506_0708,
            },
        );
        assert_eq!(&bytes[..20], b"/sranibro/gaze/left\0");
        assert_eq!(&bytes[20..28], b",fffifh\0");
        assert_eq!(bytes.len(), 56);
        let decoded = decode(&bytes);
        assert_eq!(decoded.0, LEFT_ADDRESS);
        assert_eq!(decoded.1, OSC_TAGS);
        assert_eq!(decoded.2, [1.0, -0.5, 0.25]);
        assert_eq!(decoded.3, 1);
        assert_eq!(decoded.4, 0.75);
        assert_eq!(decoded.5, 0x0102_0304_0506_0708);

        let right = packet(
            RIGHT_ADDRESS,
            GazePayload {
                direction: [0.0; 3],
                valid: 0,
                openness: -1.0,
                counter: 1,
            },
        );
        assert_eq!(&right[..24], b"/sranibro/gaze/right\0\0\0\0");
        assert_eq!(&right[24..32], b",fffifh\0");
        assert_eq!(right.len(), 60);
    }

    #[test]
    fn axis_conventions_normalize_without_an_angular_cap() {
        let eye = reported([10.0, 2.0, 1.0]);
        let tobii = gaze_payload(&eye, RawGazeAxes::Tobii, -1.0, 1);
        let varjo = gaze_payload(&eye, RawGazeAxes::Varjo, -1.0, 1);
        let length = 10.0_f32.hypot(2.0).hypot(1.0);
        assert_eq!(
            tobii.direction,
            [-10.0 / length, 2.0 / length, -1.0 / length]
        );
        assert_eq!(
            varjo.direction,
            [10.0 / length, 2.0 / length, -1.0 / length]
        );
        assert!(tobii.direction[0].abs() > 0.9, "must not cap at 30 degrees");
    }

    #[test]
    fn gaze_smoothing_matches_avatar_ema_and_off_is_native() {
        use crate::core::eye_state::SRanipalState;
        let mut sender = RawGazeOsc::new(9005).unwrap();
        assert!(!sender.gaze_smoothing);
        let native = reported([0.6, 0.0, 0.8]);
        assert_eq!(sender.smooth_eye(0, native).gaze, native.gaze);
        sender = sender.with_gaze_smoothing(true);
        let mut avatar = SRanipalState::new();
        avatar.tuning.gaze_yoke = false;
        for i in 0..240 {
            let t = i as f32 * 0.03;
            let gaze = GazeSample {
                left: reported([t.sin() * 0.6, 0.1, 0.8]),
                right: reported([-t.sin() * 0.4, -0.2, 0.9]),
                ..GazeSample::default()
            };
            let out = avatar.process_frame([[1.0, 0.62, 0.0, 0.0, 0.0]; 2], &gaze, true);
            for (i, eye) in [gaze.left, gaze.right].into_iter().enumerate() {
                let filtered = sender.smooth_eye(i, eye);
                assert_eq!(
                    filtered.gaze.map(f32::to_bits),
                    out[i].gaze.map(f32::to_bits)
                );
                assert!(filtered.gaze_reported && filtered.gaze_valid);
                let payload = gaze_payload(&filtered, RawGazeAxes::Tobii, -1.0, 1);
                assert!((payload.direction.iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn gaze_smoothing_never_fills_invalid_eyes_or_ingests_auxiliary_and_resets_on_swap() {
        let mut sender = RawGazeOsc::new(9005).unwrap().with_gaze_smoothing(true);
        sender.smooth_eye(0, reported([0.6, 0.0, 0.8]));
        let history = sender.gaze_history;
        for eye in [
            EyeSample {
                gaze_valid: false,
                ..reported([0.4, 0.0, 0.9])
            },
            EyeSample {
                gaze_reported: false,
                ..reported([0.4, 0.0, 0.9])
            },
            reported([0.0; 3]),
            reported([f32::NAN, 0.0, 1.0]),
            reported([f32::INFINITY, 0.0, 1.0]),
        ] {
            let filtered = sender.smooth_eye(0, eye);
            assert_eq!(sender.gaze_history, history);
            let payload = gaze_payload(&filtered, RawGazeAxes::Tobii, -1.0, 1);
            assert_eq!(payload.valid, 0);
            assert_eq!(payload.direction, [0.0; 3]);
        }
        sender.sample_counter = 123;
        sender.clear_eye_history();
        assert_eq!(sender.gaze_history, [INITIAL_GAZE; 2]);
        assert_eq!(sender.sample_counter, 123);
        assert_eq!(
            sender.smooth_eye(0, reported([-0.6, 0.0, 0.8])).gaze,
            adaptive_gaze_step(INITIAL_GAZE, [-0.6, 0.0, 0.8])
        );
    }

    #[test]
    fn invalid_or_unreported_gaze_emits_zero_and_never_reuses_gaze() {
        let stale = reported([1.0, 0.0, 1.0]);
        assert_eq!(gaze_payload(&stale, RawGazeAxes::Tobii, -1.0, 1).valid, 1);
        for eye in [
            EyeSample {
                gaze: [f32::NAN, 0.0, 1.0],
                gaze_valid: true,
                gaze_reported: true,
                ..EyeSample::default()
            },
            EyeSample {
                gaze: [1.0, 0.0, 1.0],
                gaze_valid: false,
                gaze_reported: true,
                ..EyeSample::default()
            },
            EyeSample {
                gaze: [1.0, 0.0, 1.0],
                gaze_valid: true,
                gaze_reported: false,
                ..EyeSample::default()
            },
            EyeSample {
                gaze: [0.0; 3],
                gaze_valid: true,
                gaze_reported: true,
                ..EyeSample::default()
            },
        ] {
            let payload = gaze_payload(&eye, RawGazeAxes::Tobii, -1.0, 2);
            assert_eq!(payload.valid, 0);
            assert_eq!(payload.direction, [0.0; 3]);
        }
    }

    #[test]
    fn auxiliary_openness_is_cached_clamped_expired_and_disable_is_unknown() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        let now = Instant::now();
        let aux = GazeSample {
            left: EyeSample {
                openness: 1.4,
                openness_valid: true,
                openness_reported: true,
                ..EyeSample::default()
            },
            right: EyeSample {
                openness: 0.0,
                openness_valid: false,
                openness_reported: true,
                ..EyeSample::default()
            },
            ..GazeSample::default()
        };
        sender.on_sample_at(aux, RawGazeAxes::Tobii, now).unwrap();
        assert_eq!(
            sender.fresh_openness(0, now + Duration::from_millis(150)),
            1.0
        );
        assert_eq!(
            sender.fresh_openness(0, now + Duration::from_millis(151)),
            -1.0
        );
        assert_eq!(sender.fresh_openness(1, now), -1.0);
    }

    #[test]
    fn clearing_routed_openness_does_not_reset_the_pair_counter() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        let now = Instant::now();
        sender.update_openness(
            0,
            &EyeSample {
                openness: 0.6,
                openness_valid: true,
                openness_reported: true,
                ..EyeSample::default()
            },
            now,
        );
        sender.sample_counter = 41;

        sender.clear_eye_history();

        assert_eq!(sender.fresh_openness(0, now), UNKNOWN_OPENNESS);
        assert_eq!(sender.fresh_openness(1, now), UNKNOWN_OPENNESS);
        assert_eq!(sender.sample_counter, 41);
        sender
            .on_sample_at(
                GazeSample {
                    left: reported([0.0, 0.0, 1.0]),
                    right: reported([0.0, 0.0, 1.0]),
                    ..GazeSample::default()
                },
                RawGazeAxes::Tobii,
                now,
            )
            .unwrap();
        assert_eq!(sender.sample_counter, 42);
    }

    #[test]
    fn aux_only_sends_nothing_then_gaze_sends_both_eyes_with_one_counter() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        receiver.set_nonblocking(true).unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        assert_eq!(sender.port(), receiver.local_addr().unwrap().port());
        sender
            .on_sample(
                GazeSample {
                    left: EyeSample {
                        openness: 0.4,
                        openness_valid: true,
                        openness_reported: true,
                        ..EyeSample::default()
                    },
                    ..GazeSample::default()
                },
                RawGazeAxes::Tobii,
            )
            .unwrap();
        let mut buffer = [0_u8; 128];
        assert_eq!(
            receiver.recv(&mut buffer).unwrap_err().kind(),
            io::ErrorKind::WouldBlock
        );

        sender
            .on_sample(
                GazeSample {
                    left: reported([1.0, 0.0, 1.0]),
                    ..GazeSample::default()
                },
                RawGazeAxes::Tobii,
            )
            .unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let first_len = receiver.recv(&mut buffer).unwrap();
        let first = decode(&buffer[..first_len]);
        let mut second_buffer = [0_u8; 128];
        let second_len = receiver.recv(&mut second_buffer).unwrap();
        let second = decode(&second_buffer[..second_len]);
        assert_eq!((first.0, second.0), (LEFT_ADDRESS, RIGHT_ADDRESS));
        assert_eq!(first.5, second.5);
        assert_eq!(first.4, 0.4);
        assert_eq!(second.4, -1.0);
        assert_eq!(second.2, [0.0; 3]);
        assert_eq!(second.3, 0);
    }

    #[test]
    fn counter_is_i64_and_can_exceed_i32() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        sender.sample_counter = i32::MAX as i64 + 10;
        sender
            .on_sample(
                GazeSample {
                    left: reported([0.0, 0.0, 1.0]),
                    right: reported([0.0, 0.0, 1.0]),
                    ..GazeSample::default()
                },
                RawGazeAxes::Tobii,
            )
            .unwrap();
        let mut buffer = [0_u8; 128];
        let len = receiver.recv(&mut buffer).unwrap();
        assert_eq!(decode(&buffer[..len]).5, i32::MAX as i64 + 11);
        let len = receiver.recv(&mut buffer).unwrap();
        assert_eq!(decode(&buffer[..len]).5, i32::MAX as i64 + 11);
    }

    #[test]
    fn input_sample_is_not_modified() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        let sample = GazeSample {
            timestamp_us: 987,
            left: reported([0.2, -0.3, 0.9]),
            right: reported([-0.4, 0.1, 0.8]),
        };
        let before = sample;
        sender.on_sample(sample, RawGazeAxes::Varjo).unwrap();
        assert_eq!(sample.timestamp_us, before.timestamp_us);
        assert_eq!(sample.left.gaze, before.left.gaze);
        assert_eq!(sample.right.gaze, before.right.gaze);
        assert_eq!(sample.left.gaze_reported, before.left.gaze_reported);
    }

    #[test]
    fn warmed_gaze_send_has_no_current_thread_allocations() {
        let receiver = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        receiver
            .set_read_timeout(Some(Duration::from_secs(1)))
            .unwrap();
        let mut sender = RawGazeOsc::new(receiver.local_addr().unwrap().port()).unwrap();
        let sample = GazeSample {
            left: reported([0.3, -0.2, 0.9]),
            right: reported([-0.1, 0.4, 0.8]),
            ..GazeSample::default()
        };

        // Warm up the platform socket path and drain both datagrams before the
        // measured callback. The allocator counter is per-thread, so kernel and
        // receiver activity cannot be attributed to the sender hot path.
        sender.on_sample(sample, RawGazeAxes::Tobii).unwrap();
        let mut packet = [0_u8; MAX_PACKET_LEN];
        receiver.recv(&mut packet).unwrap();
        receiver.recv(&mut packet).unwrap();

        let (result, allocations) =
            count_current_thread_allocations(|| sender.on_sample(sample, RawGazeAxes::Tobii));
        result.unwrap();
        assert_eq!(
            allocations, 0,
            "gaze OSC callback must remain allocation-free"
        );
        receiver.recv(&mut packet).unwrap();
        receiver.recv(&mut packet).unwrap();
        let mut smoothed = sender.with_gaze_smoothing(true);
        let (result, allocations) =
            count_current_thread_allocations(|| smoothed.on_sample(sample, RawGazeAxes::Tobii));
        result.unwrap();
        assert_eq!(
            allocations, 0,
            "optional gaze smoothing must remain allocation-free"
        );
    }

    #[test]
    fn zero_destination_port_is_rejected() {
        assert_eq!(
            RawGazeOsc::new(0).err().unwrap().kind(),
            io::ErrorKind::InvalidInput
        );
    }
}
