//! Tobii TTP protocol: USB framing, the typed-value (TLV) codec, image decode,
//! and packet builders — the pure, host-independent core of the EyeChip transport.
//!
//! Ported from `ttp_bridge.py` (framing/image) and `tobii_usb_direct.py` (the
//! DLL-free direct-USB path: TLV codec, gaze stream 1289, subscribe/auth packets).
//! Everything here is unit-testable offline; the WinUSB + auth glue drives it.
//!
//! Conventions: this module works on *deframed* TTP messages — the 8-byte USB
//! header is stripped by [`Deframer`], so within a message: big-endian `0x51` at
//! offset 0, msg_id @4, pid @12, payload_len @20, and the TLV payload at offset 26.

/// Fixed-point scale for tag-3 floats: value = i32 * (1/65536).
const FIXED_SCALE: f32 = 1.5258789e-05;

// An operational bound for this EyeChip transport, not a limit on all Tobii
// protocols. Supported grayscale frames are 40,000 bytes; this also leaves ample
// space for a 160,000-byte image plus metadata. Reject implausible lengths before
// retaining a never-ending USB stream or allocating a declared vector.
const MAX_TTP_MESSAGE_BYTES: usize = 1024 * 1024;
const MAX_BUFFER_BYTES: usize = 2 * (MAX_TTP_MESSAGE_BYTES + 8);

/// Camera image stream id.
pub const PID_IMAGE: u32 = 1291;
/// Raw gaze stream id (gaze vectors + convergence only).
pub const PID_GAZE: u32 = 1289;
/// Wearable "advanced data" stream id: gaze L/R + PUPIL diameter + pupil position
/// + eyelid openness. The DLL gated this behind a PROFESSIONAL license check, but
/// that check is in the DLL — subscribing directly over USB bypasses it.
pub const PID_WEARABLE: u32 = 1285;
/// Subscribe-command pid.
pub const PID_SUBSCRIBE: u32 = 1220;
/// Column-prologue magic that precedes every gaze row.
pub const GAZE_COLUMN_MAGIC: u32 = 134073;

// ---------------------------------------------------------------------------
// USB framing
// ---------------------------------------------------------------------------

/// Reassembles TTP messages from a stream of USB bulk reads (strips the 8-byte
/// USB header so callers see the TTP message starting at the `0x51` magic).
#[derive(Default)]
pub struct Deframer {
    buf: Vec<u8>,
    error: Option<&'static str>,
}

impl Deframer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, data: &[u8]) {
        if self.error.is_some() {
            return;
        }
        if data.len() > MAX_BUFFER_BYTES.saturating_sub(self.buf.len()) {
            self.reject("TTP receive buffer exceeded the EyeChip transport limit");
            return;
        }
        self.buf.extend_from_slice(data);
        self.validate_outer_length();
    }

    /// A malformed stream is terminal for this deframer. The transport must close
    /// the session; do not silently discard bytes and interpret pixels as headers.
    pub fn error(&self) -> Option<&'static str> {
        self.error
    }

    fn reject(&mut self, error: &'static str) {
        self.error = Some(error);
        self.buf = Vec::new();
    }

    fn validate_outer_length(&mut self) {
        if self.error.is_none() && self.buf.len() >= 8 {
            let declared = u32::from_le_bytes(self.buf[4..8].try_into().unwrap()) as usize;
            if declared > MAX_TTP_MESSAGE_BYTES + 8 {
                self.reject("TTP frame length exceeded the EyeChip transport limit");
            }
        }
    }

    /// Pop the next complete TTP message, or `None` if more data is needed.
    /// Empty frames are consumed and skipped (matches the Python relay).
    pub fn pop(&mut self) -> Option<Vec<u8>> {
        loop {
            self.validate_outer_length();
            if self.error.is_some() || self.buf.len() < 8 {
                return None;
            }
            let frame_len = u32::from_le_bytes(self.buf[4..8].try_into().unwrap()) as usize;
            // For a real TTP header, its payload length disambiguates the two USB
            // length conventions. The old "8 bytes short means total length"
            // fallback consumed incomplete camera images at certain chunk splits.
            // Unknown/legacy metadata retains the previous framing convention.
            // Do not reject a no-body acknowledgement on an unverified inner field.
            let hinted_total = self
                .buf
                .get(8..12)
                .filter(|magic| *magic == &0x51u32.to_be_bytes())
                .and_then(|_| be_u32(&self.buf, 28))
                .map(|payload_len| payload_len as usize)
                .filter(|payload_len| *payload_len <= MAX_TTP_MESSAGE_BYTES - 24)
                .map(|payload_len| 24 + payload_len)
                .filter(|message_len| frame_len == *message_len || frame_len == *message_len + 8)
                .map(|message_len| message_len + 8);
            let total = if let Some(total) = hinted_total {
                if self.buf.len() < total {
                    return None;
                }
                total
            } else if frame_len + 8 <= self.buf.len() {
                frame_len + 8
            } else if frame_len <= self.buf.len() && frame_len >= 8 {
                frame_len // protocol quirk: frame_len is the total size here
            } else {
                return None;
            };
            let msg = self.buf[8..total].to_vec();
            self.buf.drain(0..total);
            if msg.is_empty() {
                continue;
            }
            return Some(msg);
        }
    }
}

/// Big-endian pid at TTP offset 12 (0 if too short).
pub fn pid(msg: &[u8]) -> u32 {
    if msg.len() >= 16 {
        u32::from_be_bytes(msg[12..16].try_into().unwrap())
    } else {
        0
    }
}

/// True for a camera-image message (decode locally; never forward to a DLL).
pub fn is_image(msg: &[u8]) -> bool {
    pid(msg) == PID_IMAGE && msg.len() > 10000
}

// ---------------------------------------------------------------------------
// TLV typed-value codec
// ---------------------------------------------------------------------------

/// A decoded TTP typed value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U32(u32),
    F32(f32),
    /// 64-bit fixed-point value used by some wearable vectors.
    F64(f64),
    /// Prologue/structure marker; the row/sub count is `value >> 16`.
    Pro(u32),
    I64(i64),
    Blob(Vec<u8>),
    Vec(Vec<u32>),
    Unk(u8),
}

fn be_u32(d: &[u8], p: usize) -> Option<u32> {
    d.get(p..p.checked_add(4)?)
        .map(|s| u32::from_be_bytes(s.try_into().unwrap()))
}
fn be_i32(d: &[u8], p: usize) -> Option<i32> {
    d.get(p..p.checked_add(4)?)
        .map(|s| i32::from_be_bytes(s.try_into().unwrap()))
}
fn be_i64(d: &[u8], p: usize) -> Option<i64> {
    d.get(p..p.checked_add(8)?)
        .map(|s| i64::from_be_bytes(s.try_into().unwrap()))
}

/// Decode one typed value at `pos`, returning the value and the next position.
/// Bounds-checked: returns `None` on truncation.
pub fn decode_value(data: &[u8], pos: usize) -> Option<(Value, usize)> {
    let t = *data.get(pos)?;
    if !matches!(t, 2..=6 | 21 | 23) {
        return Some((Value::Unk(t), pos.checked_add(1)?));
    }
    // Keep the established tolerance for redundant length metadata. Device
    // variants may encode this differently; only actual consumed byte counts
    // below control allocation. No allocation is based on this ignored field.
    let _declared_len = be_u32(data, pos.checked_add(1)?)?;
    let start = pos.checked_add(5)?;
    match t {
        2 => Some((Value::U32(be_u32(data, start)?), start.checked_add(4)?)),
        3 => Some((
            Value::F32(be_i32(data, start)? as f32 * FIXED_SCALE),
            start.checked_add(4)?,
        )),
        4 => Some((
            Value::F64(be_i64(data, start)? as f64 / 4_294_967_296.0),
            start.checked_add(8)?,
        )),
        5 => Some((Value::Pro(be_u32(data, start)?), start.checked_add(4)?)),
        6 => Some((Value::I64(be_i64(data, start)?), start.checked_add(8)?)),
        21 | 23 => {
            let count = be_u32(data, start)? as usize;
            let byte_count = if t == 23 {
                count.checked_mul(4)?
            } else {
                count
            };
            if byte_count > MAX_TTP_MESSAGE_BYTES {
                return None;
            }
            let payload_start = start.checked_add(4)?;
            let end = payload_start.checked_add(byte_count)?;
            // Validate all remaining bytes BEFORE reserving either vector.
            let body = data.get(payload_start..end)?;
            let value = if t == 21 {
                Value::Blob(body.to_vec())
            } else {
                Value::Vec(
                    body.chunks_exact(4)
                        .map(|word| u32::from_be_bytes(word.try_into().unwrap()))
                        .collect(),
                )
            };
            Some((value, end))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Image decode (stream 1291)
// ---------------------------------------------------------------------------

/// A decoded camera frame.
pub struct DecodedImage {
    /// Bridge convention: 0 = RIGHT, 1 = LEFT, -1 = unknown.
    pub region: i32,
    /// Frame dimensions (VR4 EyeChip is fixed 200x200).
    pub width: u32,
    pub height: u32,
    /// `width*height` grayscale bytes, row-major.
    pub pixels: Vec<u8>,
}

/// Decode a pid-1291 message into (region, 200x200): scan for blob marker 0x15,
/// require a 40000-byte payload. Offsets are for a deframed message.
///
/// TODO(hardware, XR5): this keys on a 40000-byte blob = 200×200. The XR5 primary
/// optics MAY publish 400×400 (160000-byte blob). If bring-up logs show the XR5
/// frame is not 200×200, add a second size branch here (e.g. accept 160000 →
/// 400×400) and set width/height accordingly — everything downstream already reads
/// W/H from the frame, so only this decode is size-specific.
pub fn decode_image(msg: &[u8]) -> Option<DecodedImage> {
    let upper = msg.len().saturating_sub(12).min(2000);
    let mut i = 26usize;
    while i < upper {
        if msg[i] == 21 && i + 9 <= msg.len() {
            let data_len = u32::from_be_bytes(msg[i + 5..i + 9].try_into().unwrap()) as usize;
            if data_len == 40000 && i + 9 + 40000 <= msg.len() {
                let pixels = msg[i + 9..i + 9 + 40000].to_vec();
                let region = if msg.len() > 281 {
                    1 - (msg[281] & 1) as i32
                } else {
                    -1
                };
                return Some(DecodedImage {
                    region,
                    width: 200,
                    height: 200,
                    pixels,
                });
            }
        }
        i += 1;
    }
    None
}

// ---------------------------------------------------------------------------
// Packet builders (host -> device)
// ---------------------------------------------------------------------------

/// Encode a u32 TTP value (tag 2).
pub fn encode_u32(v: u32) -> Vec<u8> {
    let mut o = vec![2u8];
    o.extend_from_slice(&4u32.to_be_bytes());
    o.extend_from_slice(&v.to_be_bytes());
    o
}

/// Encode a u32-vector TTP value (tag 23).
pub fn encode_u32_vector(vals: &[u32]) -> Vec<u8> {
    let n = vals.len() as u32;
    let mut o = vec![23u8];
    o.extend_from_slice(&(n * 4 + 4).to_be_bytes());
    o.extend_from_slice(&n.to_be_bytes());
    for v in vals {
        o.extend_from_slice(&v.to_be_bytes());
    }
    o
}

/// Encode a blob TTP value (tag 21).
pub fn encode_blob(data: &[u8]) -> Vec<u8> {
    let mut o = vec![21u8];
    o.extend_from_slice(&((data.len() + 4) as u32).to_be_bytes());
    o.extend_from_slice(&(data.len() as u32).to_be_bytes());
    o.extend_from_slice(data);
    o
}

/// Build a full USB packet (8-byte USB header + TTP header + payload), matching
/// `tobii_usb_direct.ttp_build_packet`.
pub fn build_packet(msg_id: u32, packet_id: u32, payload: &[u8]) -> Vec<u8> {
    let pl_len = (payload.len() + 2) as u32;
    let msg_len = (payload.len() + 26) as u32;
    let mut o = Vec::with_capacity(8 + msg_len as usize);
    o.extend_from_slice(&0u32.to_le_bytes());
    o.extend_from_slice(&msg_len.to_le_bytes());
    for v in [0x51u32, msg_id, 0, packet_id, 0, pl_len] {
        o.extend_from_slice(&v.to_be_bytes());
    }
    o.extend_from_slice(&[0u8, 0]);
    o.extend_from_slice(payload);
    o
}

/// Subscribe to a stream (packet id 1220).
pub fn subscribe_packet(msg_id: u32, stream_id: u32) -> Vec<u8> {
    let mut payload = encode_u32(stream_id);
    payload.extend(encode_u32_vector(&[]));
    build_packet(msg_id, PID_SUBSCRIBE, &payload)
}

/// The image-stream subscribe the bridge injects (fixed msg_id 9999).
pub fn image_subscribe_packet(stream_id: u32) -> Vec<u8> {
    subscribe_packet(9999, stream_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn usb_frame(msg: &[u8]) -> Vec<u8> {
        let mut f = Vec::new();
        f.extend_from_slice(&0u32.to_le_bytes());
        f.extend_from_slice(&(msg.len() as u32).to_le_bytes());
        f.extend_from_slice(msg);
        f
    }

    fn msg_with_pid(p: u32, len: usize) -> Vec<u8> {
        let mut m = vec![0u8; len.max(16)];
        m[12..16].copy_from_slice(&p.to_be_bytes());
        m
    }

    #[test]
    fn deframer_splits_complete_and_waits_for_truly_partial() {
        let a = msg_with_pid(PID_GAZE, 32);
        let b = msg_with_pid(7, 20);

        let mut full = usb_frame(&a);
        full.extend(usb_frame(&b));
        let mut d = Deframer::new();
        d.push(&full);
        assert_eq!(pid(&d.pop().expect("a")), PID_GAZE);
        assert_eq!(pid(&d.pop().expect("b")), 7);
        assert!(d.pop().is_none());

        let mut d = Deframer::new();
        let fb = usb_frame(&b);
        d.push(&usb_frame(&a));
        d.push(&fb[..10]);
        assert_eq!(pid(&d.pop().expect("a")), PID_GAZE);
        assert!(d.pop().is_none(), "partial frame b waits");
        d.push(&fb[10..]);
        assert_eq!(pid(&d.pop().expect("b after rest")), 7);
    }

    #[test]
    fn deframer_fallback_treats_frame_len_as_total() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&20u32.to_le_bytes());
        buf.extend_from_slice(&[0u8; 12]);
        let mut d = Deframer::new();
        d.push(&buf);
        assert_eq!(d.pop().expect("fallback").len(), 12);
    }

    #[test]
    fn impossible_frame_length_is_terminal_and_retains_no_stream_data() {
        let mut header = [0u8; 8];
        header[4..].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut d = Deframer::new();
        d.push(&header);
        assert!(d.error().is_some());
        for _ in 0..128 {
            d.push(&[0; 4096]);
            assert!(d.pop().is_none());
        }
        assert_eq!(d.buf.len(), 0);
        assert_eq!(d.buf.capacity(), 0);
    }

    #[test]
    fn deframer_rejects_a_buffer_flood_before_growing() {
        let mut d = Deframer::new();
        let flood = vec![0; MAX_BUFFER_BYTES + 1];
        d.push(&flood);
        assert!(d.error().is_some());
        assert_eq!(d.buf.capacity(), 0);
    }

    #[test]
    fn supported_camera_payload_survives_fragmentation_near_either_length() {
        let packet = build_packet(0, PID_IMAGE, &encode_blob(&vec![127; 40_000]));
        for chunk_size in [1, 7, 31, 32, 16_384, packet.len() - 8, packet.len() - 1] {
            let mut d = Deframer::new();
            let mut messages = Vec::new();
            for chunk in packet.chunks(chunk_size) {
                d.push(chunk);
                while let Some(message) = d.pop() {
                    messages.push(message);
                }
                assert!(d.error().is_none());
            }
            assert_eq!(messages.len(), 1, "chunk size {chunk_size}");
            assert_eq!(messages[0], packet[8..], "chunk size {chunk_size}");
            assert_eq!(decode_image(&messages[0]).unwrap().pixels.len(), 40_000);
        }
    }

    #[test]
    fn total_length_quirk_does_not_swallow_the_next_coalesced_frame() {
        let mut first = build_packet(1, PID_GAZE, &encode_u32(7));
        let first_len = first.len() as u32;
        first[4..8].copy_from_slice(&first_len.to_le_bytes());
        let second = build_packet(2, PID_WEARABLE, &encode_u32(9));
        let mut d = Deframer::new();
        d.push(&[first.as_slice(), second.as_slice()].concat());
        assert_eq!(d.pop().unwrap(), first[8..]);
        assert_eq!(d.pop().unwrap(), second[8..]);
        assert!(d.pop().is_none());
        assert!(d.error().is_none());
    }

    #[test]
    fn deframer_preserves_legacy_inner_metadata_and_no_body_ack() {
        let mut packet = build_packet(1, PID_GAZE, &encode_u32(7));
        packet[28..32].copy_from_slice(&u32::MAX.to_be_bytes());
        let mut d = Deframer::new();
        d.push(&packet);
        assert_eq!(d.pop().unwrap(), packet[8..]);
        assert!(d.error().is_none());
        let mut ack = vec![0u8; 24];
        ack[..4].copy_from_slice(&0x51u32.to_be_bytes());
        d.push(&usb_frame(&ack));
        assert_eq!(d.pop().unwrap(), ack);
        assert!(d.error().is_none());
    }

    #[test]
    fn image_classified_and_decoded_with_region() {
        let blob_at = 30usize;
        let mut m = vec![0u8; blob_at + 9 + 40000];
        m[12..16].copy_from_slice(&PID_IMAGE.to_be_bytes());
        m[blob_at] = 21;
        m[blob_at + 5..blob_at + 9].copy_from_slice(&40000u32.to_be_bytes());
        m[blob_at + 9] = 123;
        m[281] = 0;
        assert!(is_image(&m));
        let img = decode_image(&m).expect("decodes");
        assert_eq!(img.region, 1);
        assert_eq!(img.pixels.len(), 40000);
        assert_eq!(img.pixels[0], 123);
        m[281] = 1;
        assert_eq!(decode_image(&m).unwrap().region, 0);
    }

    #[test]
    fn value_codec_round_trips() {
        let buf = encode_u32(0xDEAD_BEEF);
        assert_eq!(decode_value(&buf, 0).unwrap().0, Value::U32(0xDEAD_BEEF));

        let v = encode_u32_vector(&[1, 2, 3]);
        assert_eq!(decode_value(&v, 0).unwrap().0, Value::Vec(vec![1, 2, 3]));

        let b = encode_blob(&[9, 8, 7]);
        assert_eq!(decode_value(&b, 0).unwrap().0, Value::Blob(vec![9, 8, 7]));

        // tag-3 fixed-point float: 32768 -> ~0.5
        let mut f = vec![3u8];
        f.extend_from_slice(&4u32.to_be_bytes());
        f.extend_from_slice(&32768i32.to_be_bytes());
        match decode_value(&f, 0).unwrap().0 {
            Value::F32(x) => assert!((x - 0.5).abs() < 1e-3, "got {x}"),
            other => panic!("expected F32, got {other:?}"),
        }
    }

    #[test]
    fn decode_value_is_truncation_safe() {
        // A u32 tag with only 2 bytes of value -> None, not a panic.
        let mut t = vec![2u8];
        t.extend_from_slice(&4u32.to_be_bytes());
        t.extend_from_slice(&[0u8, 0]); // short
        assert!(decode_value(&t, 0).is_none());
    }

    #[test]
    fn declared_vector_and_blob_counts_are_checked_before_allocating() {
        for tag in [21, 23] {
            let mut truncated = vec![tag];
            truncated.extend_from_slice(&4u32.to_be_bytes());
            truncated.extend_from_slice(&u32::MAX.to_be_bytes());
            assert!(decode_value(&truncated, 0).is_none());
            truncated[1..5].copy_from_slice(&u32::MAX.to_be_bytes());
            assert!(decode_value(&truncated, 0).is_none());
        }
        assert!(decode_value(&[23; 9], usize::MAX).is_none());
        assert!(be_u32(&[0; 8], usize::MAX).is_none());
        assert!(be_i64(&[0; 8], usize::MAX).is_none());
    }

    #[test]
    fn tlv_preserves_metadata_tolerance_but_bounds_actual_vector_payload() {
        let mut scalar = encode_u32(7);
        scalar[1..5].copy_from_slice(&8u32.to_be_bytes());
        scalar.extend_from_slice(&[1; 8]);
        assert_eq!(decode_value(&scalar, 0), Some((Value::U32(7), 9)));
        let mut vector = encode_u32_vector(&[1, 2]);
        vector[5..9].copy_from_slice(&3u32.to_be_bytes());
        assert!(decode_value(&vector, 0).is_none());
        let mut blob = encode_blob(&[1, 2]);
        blob[5..9].copy_from_slice(&3u32.to_be_bytes());
        assert!(decode_value(&blob, 0).is_none());
        let mut padded_metadata = encode_blob(&[1, 2]);
        padded_metadata[1..5].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(
            decode_value(&padded_metadata, 0),
            Some((Value::Blob(vec![1, 2]), 11))
        );
        let large_image = encode_blob(&vec![127; 160_000]);
        assert!(
            matches!(decode_value(&large_image, 0), Some((Value::Blob(bytes), _)) if bytes.len() == 160_000)
        );
    }

    #[test]
    fn subscribe_packet_is_byte_exact() {
        let pkt = image_subscribe_packet(PID_IMAGE);
        assert_eq!(pkt.len(), 52);
        assert_eq!(&pkt[4..8], &44u32.to_le_bytes(), "USB len = 44 (LE)");
        assert_eq!(&pkt[8..12], &0x51u32.to_be_bytes(), "TTP magic");
        assert_eq!(&pkt[20..24], &PID_SUBSCRIBE.to_be_bytes(), "pid 1220 @20");
        assert_eq!(
            &pkt[39..43],
            &PID_IMAGE.to_be_bytes(),
            "subscribed stream id"
        );
    }
}
