//! Decode the raw TTP gaze stream (pid 1289) into per-eye gaze directions.
//!
//! Faithful port of `tobii_usb_direct.py::decode_gaze_packet`. The packet is a
//! row/column TLV structure: a row prologue gives the row count; each row is a
//! column prologue (magic 134073), a column id, and a column value. Vector
//! columns (gaze directions) arrive as a sub-prologue followed by 3 floats.
//!
//! Column ids: 1=timestamp, 2=status, 3=right-eye gaze, 4=left-eye gaze,
//! 5=combined gaze, 6=convergence. Gaze x/y are negated to the chip's canonical
//! sign here (as in the Python reference); the SRanibro sink's own negation must
//! be reconciled against this once verified on hardware.
//!
//! This module decodes raw gaze stream 1289 and wearable-advanced stream 1285
//! (native pupil diameter, pupil position, and native openness).

use super::ttp::{decode_value, Value, GAZE_COLUMN_MAGIC};
use crate::core::types::{Eye, EyeSample, GazeSample};

/// Per-frame gaze, as recovered from a 1289 packet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GazeData {
    pub timestamp: i64,
    pub status: u32,
    pub l_gaze: [f32; 3],
    pub r_gaze: [f32; 3],
    pub combined: [f32; 3],
    pub convergence: f32,
}

impl Default for GazeData {
    fn default() -> Self {
        Self {
            timestamp: 0,
            status: 0,
            l_gaze: [0.0; 3],
            r_gaze: [0.0; 3],
            combined: [0.0; 3],
            convergence: -1.0,
        }
    }
}

impl GazeData {
    /// Map to the core contract. 1289 carries only gaze (openness comes from the
    /// ML on the camera images); pupil/origin are left invalid. `status > 0`
    /// marks the gaze valid.
    pub fn to_gaze_sample(&self) -> GazeSample {
        let valid = self.status > 0;
        let mk = |g: [f32; 3]| EyeSample {
            gaze: g,
            gaze_valid: valid,
            gaze_reported: true,
            ..Default::default()
        };
        GazeSample {
            timestamp_us: self.timestamp.max(0) as u64,
            left: mk(self.l_gaze),
            right: mk(self.r_gaze),
        }
    }

    /// XR5 firmware can publish a usable per-eye direction while the packet-level
    /// status is zero. Conversely, that status is shared by both eyes and cannot
    /// describe a missing vector for just one eye. Validate each direction itself
    /// on XR5 so a real vector keeps flowing, while the all-zero sentinel remains
    /// invalid. Other HMDs retain [`Self::to_gaze_sample`] and its native status rule.
    pub fn to_xr5_gaze_sample(&self) -> GazeSample {
        let mk = |g: [f32; 3]| EyeSample {
            gaze: g,
            gaze_valid: plausible_gaze_direction(g),
            gaze_reported: true,
            ..Default::default()
        };
        GazeSample {
            timestamp_us: self.timestamp.max(0) as u64,
            left: mk(self.l_gaze),
            right: mk(self.r_gaze),
        }
    }

    /// Shape the EyeChip's fused/cyclopean direction as an ordinary stereo sample.
    /// Both eyes intentionally receive the same native vector; the existing per-eye
    /// centre/range/static-vergence correction remains downstream. There is no
    /// per-frame fallback to columns 3/4, so source disagreement cannot cause flapping.
    pub fn to_xr5_combined_gaze_sample(&self) -> GazeSample {
        let valid = plausible_gaze_direction(self.combined);
        let eye = EyeSample {
            gaze: self.combined,
            gaze_valid: valid,
            gaze_reported: true,
            ..Default::default()
        };
        GazeSample {
            timestamp_us: self.timestamp.max(0) as u64,
            left: eye,
            right: eye,
        }
    }
}

fn plausible_gaze_direction(g: [f32; 3]) -> bool {
    if !g.iter().all(|v| v.is_finite()) || g[2] <= 0.1 {
        return false;
    }
    let norm_sq = g.iter().map(|v| v * v).sum::<f32>();
    (0.25..=2.25).contains(&norm_sq) && g[0].abs() <= 1.5 && g[1].abs() <= 1.5 && g[2] <= 1.5
}

/// Per-frame wearable advanced data, as recovered from stream 1285.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WearableData {
    pub timestamp_us: u64,
    pub origin_mm: [[f32; 3]; 2],
    pub origin_valid: [bool; 2],
    pub gaze: [[f32; 3]; 2],
    pub gaze_valid: [bool; 2],
    pub pupil_mm: [f32; 2],
    pub pupil_valid: [bool; 2],
    pub pupil_pos: [[f32; 2]; 2],
    pub pupil_pos_valid: [bool; 2],
    pub openness: [f32; 2],
    pub openness_valid: [bool; 2],
}

impl Default for WearableData {
    fn default() -> Self {
        Self {
            timestamp_us: 0,
            origin_mm: [[0.0; 3]; 2],
            origin_valid: [false; 2],
            gaze: [[0.0; 3]; 2],
            gaze_valid: [false; 2],
            pupil_mm: [0.0; 2],
            pupil_valid: [false; 2],
            pupil_pos: [[0.5; 2]; 2],
            pupil_pos_valid: [false; 2],
            openness: [0.0; 2],
            openness_valid: [false; 2],
        }
    }
}

impl WearableData {
    pub fn to_gaze_sample(&self) -> GazeSample {
        let mk = |eye: Eye| {
            let i = eye.idx();
            EyeSample {
                gaze: self.gaze[i],
                gaze_valid: self.gaze_valid[i],
                gaze_reported: true,
                origin_mm: self.origin_mm[i],
                origin_valid: self.origin_valid[i],
                pupil_mm: self.pupil_mm[i],
                pupil_valid: self.pupil_valid[i],
                pupil_pos: self.pupil_pos[i],
                pupil_pos_valid: self.pupil_pos_valid[i],
                openness: self.openness[i],
                openness_valid: self.openness_valid[i],
                openness_reported: true,
            }
        };
        GazeSample {
            timestamp_us: self.timestamp_us,
            left: mk(Eye::Left),
            right: mk(Eye::Right),
        }
    }

    /// Convert wearable stream 1285 to its auxiliary contribution only. Stream
    /// 1289 is the canonical gaze source; accepting the duplicate 1285 gaze into
    /// the same latest-value slot made the two differently timed/LR-mapped streams
    /// alternate and produced large apparent gaze jumps on XR5.
    pub fn to_aux_sample(&self) -> GazeSample {
        let mut sample = self.to_gaze_sample();
        sample.timestamp_us = 0;
        for eye in [&mut sample.left, &mut sample.right] {
            eye.gaze_valid = false;
            eye.gaze_reported = false;
        }
        sample
    }

    fn any_valid(&self) -> bool {
        self.origin_valid.iter().any(|&v| v)
            || self.gaze_valid.iter().any(|&v| v)
            || self.pupil_valid.iter().any(|&v| v)
            || self.pupil_pos_valid.iter().any(|&v| v)
            || self.openness_valid.iter().any(|&v| v)
    }
}

#[derive(Debug, Clone)]
enum ColumnPayload {
    Scalar(Value),
    Struct(Vec<Value>),
}

fn decode_tlv_columns(msg: &[u8]) -> Option<Vec<(u32, ColumnPayload)>> {
    let mut pos = 26;
    let (row, np) = decode_value(msg, pos)?;
    pos = np;
    let row_count = match row {
        Value::Pro(v) => v >> 16,
        _ => return None,
    };

    let mut columns = Vec::with_capacity(row_count as usize);
    for _ in 0..row_count {
        if pos + 10 >= msg.len() {
            break;
        }
        let (cp, np) = decode_value(msg, pos)?;
        pos = np;
        if cp != Value::Pro(GAZE_COLUMN_MAGIC) {
            break;
        }
        let (col_id_v, np) = decode_value(msg, pos)?;
        pos = np;
        let col_id = match col_id_v {
            Value::U32(v) => v,
            _ => break,
        };
        let (val, np) = decode_value(msg, pos)?;
        pos = np;
        let payload = match val {
            Value::Pro(p) => {
                let sub_count = p >> 16;
                let mut subs = Vec::with_capacity(sub_count as usize);
                for _ in 0..sub_count {
                    let (sv, np) = decode_value(msg, pos)?;
                    pos = np;
                    subs.push(sv);
                }
                ColumnPayload::Struct(subs)
            }
            other => ColumnPayload::Scalar(other),
        };
        columns.push((col_id, payload));
    }
    Some(columns)
}

/// Decode pid-1285 wearable advanced data. This first tries the 160-byte VR4
/// callback payload layout seen in prior DLL captures, then the observed direct
/// TTP row/column layout from the EyeChip's wearable-advanced stream.
pub fn decode_wearable_1285(msg: &[u8]) -> Option<WearableData> {
    decode_wearable_struct_payload(msg).or_else(|| decode_wearable_tlv_layout(msg))
}

fn decode_wearable_tlv_layout(msg: &[u8]) -> Option<WearableData> {
    let columns = decode_tlv_columns(msg)?;
    let mut w = WearableData::default();

    if let Some(t) = scalar_i64(&columns, 1) {
        w.timestamp_us = t.max(0) as u64;
    }

    read_tlv_eye(
        &columns,
        Eye::Left.idx(),
        EyeTlvCols {
            origin: 2,
            origin_valid: 3,
            gaze: 4,
            gaze_valid: 5,
            pupil_mm: 6,
            openness: 7,
            pupil_pos: 23,
            pupil_pos_valid: 22,
        },
        &mut w,
    );
    read_tlv_eye(
        &columns,
        Eye::Right.idx(),
        EyeTlvCols {
            origin: 8,
            origin_valid: 9,
            gaze: 10,
            gaze_valid: 11,
            pupil_mm: 12,
            openness: 13,
            pupil_pos: 25,
            pupil_pos_valid: 24,
        },
        &mut w,
    );

    w.any_valid().then_some(w)
}

#[derive(Clone, Copy)]
struct EyeTlvCols {
    origin: u32,
    origin_valid: u32,
    gaze: u32,
    gaze_valid: u32,
    pupil_mm: u32,
    openness: u32,
    pupil_pos: u32,
    pupil_pos_valid: u32,
}

fn read_tlv_eye(
    columns: &[(u32, ColumnPayload)],
    idx: usize,
    cols: EyeTlvCols,
    out: &mut WearableData,
) {
    if let Some(gaze) = struct_vec3(columns, cols.gaze) {
        // RAW gaze. The single canonical sign is applied once downstream in the
        // output sink (BrokenEyeSink), per the adapter contract — never here.
        out.gaze[idx] = gaze;
        out.gaze_valid[idx] = scalar_u32(columns, cols.gaze_valid) == Some(1);
    }
    if let Some(origin) = struct_vec3(columns, cols.origin) {
        out.origin_mm[idx] = origin;
        out.origin_valid[idx] = scalar_u32(columns, cols.origin_valid) == Some(1);
    }
    if let Some(mm) = scalar_f32(columns, cols.pupil_mm) {
        out.pupil_mm[idx] = mm;
        out.pupil_valid[idx] = (0.5..=12.0).contains(&mm);
    }
    if let Some(openness) = scalar_f32(columns, cols.openness) {
        out.openness[idx] = openness;
        out.openness_valid[idx] = (0.0..=1.0).contains(&openness);
    }
    if let Some(pos) = struct_vec2(columns, cols.pupil_pos) {
        out.pupil_pos[idx] = pos;
        out.pupil_pos_valid[idx] = scalar_u32(columns, cols.pupil_pos_valid) == Some(1)
            && pos.iter().all(|v| (0.0..=1.0).contains(v));
    }
}

fn column_payload<'a>(
    columns: &'a [(u32, ColumnPayload)],
    target: u32,
) -> Option<&'a ColumnPayload> {
    columns
        .iter()
        .find_map(|(col, payload)| (*col == target).then_some(payload))
}

fn scalar_u32(columns: &[(u32, ColumnPayload)], target: u32) -> Option<u32> {
    match column_payload(columns, target)? {
        ColumnPayload::Scalar(Value::U32(v)) => Some(*v),
        _ => None,
    }
}

fn scalar_i64(columns: &[(u32, ColumnPayload)], target: u32) -> Option<i64> {
    match column_payload(columns, target)? {
        ColumnPayload::Scalar(Value::I64(v)) => Some(*v),
        ColumnPayload::Scalar(Value::U32(v)) => Some(*v as i64),
        _ => None,
    }
}

fn scalar_f32(columns: &[(u32, ColumnPayload)], target: u32) -> Option<f32> {
    match column_payload(columns, target)? {
        ColumnPayload::Scalar(Value::F32(v)) if v.is_finite() => Some(*v),
        ColumnPayload::Scalar(Value::F64(v)) if v.is_finite() => Some(*v as f32),
        _ => None,
    }
}

fn struct_vec3(columns: &[(u32, ColumnPayload)], target: u32) -> Option<[f32; 3]> {
    match column_payload(columns, target)? {
        ColumnPayload::Struct(subs) => vec3_raw(subs),
        _ => None,
    }
}

fn struct_vec2(columns: &[(u32, ColumnPayload)], target: u32) -> Option<[f32; 2]> {
    match column_payload(columns, target)? {
        ColumnPayload::Struct(subs) => vec2_raw(subs),
        _ => None,
    }
}

fn vec3_raw(subs: &[Value]) -> Option<[f32; 3]> {
    if subs.len() != 3 {
        return None;
    }
    let mut out = [0.0; 3];
    for (i, value) in subs.iter().enumerate() {
        out[i] = match value {
            Value::F32(v) if v.is_finite() => *v,
            Value::F64(v) if v.is_finite() => *v as f32,
            _ => return None,
        };
    }
    Some(out)
}

fn vec2_raw(subs: &[Value]) -> Option<[f32; 2]> {
    if subs.len() != 2 {
        return None;
    }
    let mut out = [0.0; 2];
    for (i, value) in subs.iter().enumerate() {
        out[i] = match value {
            Value::F32(v) if v.is_finite() => *v,
            Value::F64(v) if v.is_finite() => *v as f32,
            _ => return None,
        };
    }
    Some(out)
}
const WEARABLE_STRUCT_LEN: usize = 160;
const WEARABLE_EYE_LEN: usize = 72;

fn decode_wearable_struct_payload(msg: &[u8]) -> Option<WearableData> {
    if is_ttp_message(msg) && msg.len() >= 26 + WEARABLE_STRUCT_LEN {
        if let Some(w) = parse_wearable_struct(&msg[26..26 + WEARABLE_STRUCT_LEN]) {
            return Some(w);
        }
    }
    if msg.len() >= WEARABLE_STRUCT_LEN {
        parse_wearable_struct(&msg[..WEARABLE_STRUCT_LEN])
    } else {
        None
    }
}

fn is_ttp_message(msg: &[u8]) -> bool {
    msg.len() >= 4 && u32::from_be_bytes(msg[0..4].try_into().unwrap()) == 0x51
}

fn parse_wearable_struct(bytes: &[u8]) -> Option<WearableData> {
    if bytes.len() < WEARABLE_STRUCT_LEN {
        return None;
    }
    let mut w = WearableData::default();
    let tracker_ts = read_i64_le(bytes, 0)?;
    let system_ts = read_i64_le(bytes, 8)?;
    w.timestamp_us = system_ts.max(tracker_ts).max(0) as u64;
    read_eye_struct(bytes, 16, Eye::Left.idx(), &mut w)?;
    read_eye_struct(bytes, 16 + WEARABLE_EYE_LEN, Eye::Right.idx(), &mut w)?;
    w.any_valid().then_some(w)
}

fn read_eye_struct(bytes: &[u8], base: usize, idx: usize, out: &mut WearableData) -> Option<()> {
    let origin_valid = read_i32_le(bytes, base)?;
    let gaze_valid = read_i32_le(bytes, base + 16)?;
    let pupil_valid = read_i32_le(bytes, base + 32)?;
    let pupil_pos_valid = read_i32_le(bytes, base + 40)?;
    let openness_valid = read_i32_le(bytes, base + 52)?;
    for flag in [
        origin_valid,
        gaze_valid,
        pupil_valid,
        pupil_pos_valid,
        openness_valid,
    ] {
        if flag != 0 && flag != 1 {
            return None;
        }
    }

    let origin = [
        read_f32_le(bytes, base + 4)?,
        read_f32_le(bytes, base + 8)?,
        read_f32_le(bytes, base + 12)?,
    ];
    let gaze = [
        read_f32_le(bytes, base + 20)?,
        read_f32_le(bytes, base + 24)?,
        read_f32_le(bytes, base + 28)?,
    ];
    let pupil_mm = read_f32_le(bytes, base + 36)?;
    let pupil_pos = [
        read_f32_le(bytes, base + 44)?,
        read_f32_le(bytes, base + 48)?,
    ];
    let openness = read_f32_le(bytes, base + 56)?;

    if !origin.iter().chain(gaze.iter()).all(|v| v.is_finite())
        || !pupil_mm.is_finite()
        || !pupil_pos.iter().all(|v| v.is_finite())
        || !openness.is_finite()
    {
        return None;
    }
    if pupil_valid == 1 && !(0.5..=12.0).contains(&pupil_mm) {
        return None;
    }
    if pupil_pos_valid == 1 && !pupil_pos.iter().all(|v| (-0.2..=1.2).contains(v)) {
        return None;
    }
    if openness_valid == 1 && !(-0.05..=1.2).contains(&openness) {
        return None;
    }

    out.origin_mm[idx] = origin;
    out.origin_valid[idx] = origin_valid == 1;
    out.gaze[idx] = gaze;
    out.gaze_valid[idx] = gaze_valid == 1;
    out.pupil_mm[idx] = pupil_mm;
    out.pupil_valid[idx] = pupil_valid == 1;
    out.pupil_pos[idx] = pupil_pos;
    out.pupil_pos_valid[idx] = pupil_pos_valid == 1;
    out.openness[idx] = openness;
    out.openness_valid[idx] = openness_valid == 1;
    Some(())
}

fn read_i32_le(bytes: &[u8], pos: usize) -> Option<i32> {
    bytes
        .get(pos..pos + 4)
        .map(|s| i32::from_le_bytes(s.try_into().unwrap()))
}

fn read_i64_le(bytes: &[u8], pos: usize) -> Option<i64> {
    bytes
        .get(pos..pos + 8)
        .map(|s| i64::from_le_bytes(s.try_into().unwrap()))
}

fn read_f32_le(bytes: &[u8], pos: usize) -> Option<f32> {
    bytes
        .get(pos..pos + 4)
        .map(|s| f32::from_le_bytes(s.try_into().unwrap()))
}

/// First three sub-values as a RAW vector (lenient: missing -> 0.0). The single
/// canonical gaze sign is applied once downstream in the output sink, not here.
fn vec3_lenient(subs: &[Value]) -> [f32; 3] {
    let f = |i: usize| match subs.get(i) {
        Some(Value::F32(x)) => *x,
        _ => 0.0,
    };
    [f(0), f(1), f(2)]
}

/// Decode a deframed pid-1289 message. The TLV payload begins at offset 26
/// (the 8-byte USB header is already stripped by `Deframer`).
pub fn decode_gaze_1289(msg: &[u8]) -> Option<GazeData> {
    let mut g = GazeData::default();
    let mut pos = 26;

    let (row, np) = decode_value(msg, pos)?;
    pos = np;
    let row_count = match row {
        Value::Pro(v) => v >> 16,
        _ => return None,
    };

    for _ in 0..row_count {
        if pos + 10 >= msg.len() {
            break;
        }
        // Column prologue (must match the gaze magic).
        let (cp, np) = decode_value(msg, pos)?;
        pos = np;
        match cp {
            Value::Pro(GAZE_COLUMN_MAGIC) => {}
            _ => break,
        }
        // Column id, then the column value.
        let (col_id_v, np) = decode_value(msg, pos)?;
        pos = np;
        let col_id = match col_id_v {
            Value::U32(v) => v,
            _ => break,
        };
        let (val, np) = decode_value(msg, pos)?;
        pos = np;

        match val {
            Value::Pro(p) => {
                let sub_count = p >> 16;
                let mut subs = Vec::with_capacity(sub_count as usize);
                for _ in 0..sub_count {
                    let (sv, np) = decode_value(msg, pos)?;
                    pos = np;
                    subs.push(sv);
                }
                match col_id {
                    3 => g.r_gaze = vec3_lenient(&subs),
                    4 => g.l_gaze = vec3_lenient(&subs),
                    5 => g.combined = vec3_lenient(&subs),
                    _ => {}
                }
            }
            other => match col_id {
                1 => {
                    g.timestamp = match other {
                        Value::I64(t) => t,
                        Value::U32(t) => t as i64,
                        _ => g.timestamp,
                    }
                }
                2 => {
                    if let Value::U32(s) = other {
                        g.status = s;
                    }
                }
                6 => {
                    g.convergence = match other {
                        Value::F32(c) => c,
                        Value::U32(c) => c as f32,
                        _ => g.convergence,
                    }
                }
                _ => {}
            },
        }
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc_pro(v: u32) -> Vec<u8> {
        let mut o = vec![5u8];
        o.extend_from_slice(&4u32.to_be_bytes());
        o.extend_from_slice(&v.to_be_bytes());
        o
    }
    fn enc_u32(v: u32) -> Vec<u8> {
        let mut o = vec![2u8];
        o.extend_from_slice(&4u32.to_be_bytes());
        o.extend_from_slice(&v.to_be_bytes());
        o
    }
    fn enc_f(i: i32) -> Vec<u8> {
        let mut o = vec![3u8];
        o.extend_from_slice(&4u32.to_be_bytes());
        o.extend_from_slice(&i.to_be_bytes());
        o
    }
    fn enc_i64(v: i64) -> Vec<u8> {
        let mut o = vec![6u8];
        o.extend_from_slice(&8u32.to_be_bytes());
        o.extend_from_slice(&v.to_be_bytes());
        o
    }

    #[test]
    fn decodes_left_gaze_status_and_timestamp() {
        // 5 rows: left-eye gaze vector, status, timestamp, combined, convergence.
        let mut tlv = enc_pro(5 << 16); // row_count = 5

        // Row 1: left eye (col 4) = vector [0.5, -0.25, 1.0] (pre-negation).
        tlv.extend(enc_pro(GAZE_COLUMN_MAGIC));
        tlv.extend(enc_u32(4));
        tlv.extend(enc_pro(3 << 16)); // sub_count = 3
        tlv.extend(enc_f(32768)); //  0.5
        tlv.extend(enc_f(-16384)); // -0.25
        tlv.extend(enc_f(65536)); //  1.0

        // Row 2: status (col 2) = 7.
        tlv.extend(enc_pro(GAZE_COLUMN_MAGIC));
        tlv.extend(enc_u32(2));
        tlv.extend(enc_u32(7));

        // Row 3: timestamp (col 1) = 123456789.
        tlv.extend(enc_pro(GAZE_COLUMN_MAGIC));
        tlv.extend(enc_u32(1));
        tlv.extend(enc_i64(123_456_789));

        // Row 4: EyeChip fused gaze (col 5).
        tlv.extend(enc_pro(GAZE_COLUMN_MAGIC));
        tlv.extend(enc_u32(5));
        tlv.extend(enc_pro(3 << 16));
        tlv.extend(enc_f(16384)); //  0.25
        tlv.extend(enc_f(-8192)); // -0.125
        tlv.extend(enc_f(65536)); //  1.0

        // Row 5: convergence scalar (col 6). Its physical unit is intentionally
        // not interpreted by SRanibro until verified on hardware.
        tlv.extend(enc_pro(GAZE_COLUMN_MAGIC));
        tlv.extend(enc_u32(6));
        tlv.extend(enc_f(131072)); // 2.0 raw scalar

        // Prepend a 26-byte TTP header (contents irrelevant to the decoder).
        let mut msg = vec![0u8; 26];
        msg.extend(tlv);

        let g = decode_gaze_1289(&msg).expect("decodes");
        // RAW (decoder no longer negates; sink applies the sign once): [0.5, -0.25, 1.0].
        assert!((g.l_gaze[0] - 0.5).abs() < 1e-3, "lx {}", g.l_gaze[0]);
        assert!((g.l_gaze[1] - (-0.25)).abs() < 1e-3, "ly {}", g.l_gaze[1]);
        assert!((g.l_gaze[2] - 1.0).abs() < 1e-3, "lz {}", g.l_gaze[2]);
        assert_eq!(g.status, 7);
        assert_eq!(g.timestamp, 123_456_789);
        assert_eq!(g.combined, [0.25, -0.125, 1.0]);
        assert!((g.convergence - 2.0).abs() < 1e-6);
        assert_eq!(
            g.r_gaze,
            [0.0, 0.0, 0.0],
            "right eye not present this frame"
        );
    }

    #[test]
    fn bad_column_magic_stops_cleanly() {
        let mut tlv = enc_pro(1 << 16);
        tlv.extend(enc_pro(999)); // wrong magic -> break
        let mut msg = vec![0u8; 26];
        msg.extend(tlv);
        let g = decode_gaze_1289(&msg).expect("returns default-ish");
        assert_eq!(g.status, 0);
    }

    #[test]
    fn xr5_uses_per_eye_vector_validity_when_packet_status_is_zero() {
        let g = GazeData {
            status: 0,
            l_gaze: [0.2, -0.1, 0.97],
            r_gaze: [0.0, 0.0, 0.0],
            ..GazeData::default()
        };
        let sample = g.to_xr5_gaze_sample();
        assert!(sample.left.gaze_valid);
        assert!(
            !sample.right.gaze_valid,
            "zero-vector sentinel must stay invalid"
        );
        assert!(sample.left.gaze_reported && sample.right.gaze_reported);
    }

    #[test]
    fn xr5_combined_mode_duplicates_chip_fusion_even_when_per_eye_is_lost() {
        let g = GazeData {
            timestamp: 42,
            status: 0,
            l_gaze: [0.0; 3],
            r_gaze: [0.0; 3],
            combined: [0.12, -0.08, 0.99],
            ..GazeData::default()
        };
        let sample = g.to_xr5_combined_gaze_sample();
        assert_eq!(sample.timestamp_us, 42);
        assert_eq!(sample.left.gaze, g.combined);
        assert_eq!(sample.right.gaze, g.combined);
        assert!(sample.left.gaze_valid && sample.right.gaze_valid);
        assert!(sample.left.gaze_reported && sample.right.gaze_reported);
    }

    #[test]
    fn xr5_combined_mode_never_falls_back_to_valid_per_eye_vectors() {
        for bad_combined in [[0.0, 0.0, 0.0], [f32::NAN, 0.0, 1.0], [0.0, 0.0, -1.0]] {
            let g = GazeData {
                l_gaze: [0.1, 0.0, 0.99],
                r_gaze: [-0.1, 0.0, 0.99],
                combined: bad_combined,
                ..GazeData::default()
            };
            let sample = g.to_xr5_combined_gaze_sample();
            assert!(!sample.left.gaze_valid && !sample.right.gaze_valid);
            assert_eq!(
                sample.left.gaze.map(f32::to_bits),
                bad_combined.map(f32::to_bits)
            );
            assert_eq!(
                sample.right.gaze.map(f32::to_bits),
                bad_combined.map(f32::to_bits)
            );
        }
    }

    #[test]
    fn xr5_rejects_non_finite_or_backward_gaze_vectors() {
        for bad in [
            [f32::NAN, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [4.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ] {
            assert!(!plausible_gaze_direction(bad), "accepted {bad:?}");
        }
    }

    fn enc_col(col: u32, val: Vec<u8>) -> Vec<u8> {
        let mut out = enc_pro(GAZE_COLUMN_MAGIC);
        out.extend(enc_u32(col));
        out.extend(val);
        out
    }

    fn enc_vec_f(vals: &[i32]) -> Vec<u8> {
        let mut out = enc_pro((vals.len() as u32) << 16);
        for v in vals {
            out.extend(enc_f(*v));
        }
        out
    }

    #[test]
    fn decodes_wearable_1285_observed_layout() {
        let rows = 17u32;
        let mut tlv = enc_pro(rows << 16);
        tlv.extend(enc_col(1, enc_i64(123_456)));
        tlv.extend(enc_col(2, enc_vec_f(&[65536, 131072, 196608])));
        tlv.extend(enc_col(3, enc_u32(1)));
        tlv.extend(enc_col(4, enc_vec_f(&[6554, -13107, 58982])));
        tlv.extend(enc_col(5, enc_u32(1)));
        tlv.extend(enc_col(6, enc_f((3.25 * 65536.0) as i32)));
        tlv.extend(enc_col(7, enc_f((0.75 * 65536.0) as i32)));
        tlv.extend(enc_col(22, enc_u32(1)));
        tlv.extend(enc_col(
            23,
            enc_vec_f(&[(0.4 * 65536.0) as i32, (0.6 * 65536.0) as i32]),
        ));
        tlv.extend(enc_col(8, enc_vec_f(&[262144, 327680, 393216])));
        tlv.extend(enc_col(9, enc_u32(1)));
        tlv.extend(enc_col(10, enc_vec_f(&[-6554, 13107, 55706])));
        tlv.extend(enc_col(11, enc_u32(1)));
        tlv.extend(enc_col(12, enc_f((4.5 * 65536.0) as i32)));
        tlv.extend(enc_col(13, enc_f((0.5 * 65536.0) as i32)));
        tlv.extend(enc_col(24, enc_u32(1)));
        tlv.extend(enc_col(
            25,
            enc_vec_f(&[(0.45 * 65536.0) as i32, (0.55 * 65536.0) as i32]),
        ));

        let mut msg = vec![0u8; 26];
        msg.extend(tlv);
        let w = decode_wearable_1285(&msg).expect("wearable 1285 decodes");

        assert_eq!(w.timestamp_us, 123_456);
        assert!(w.gaze_valid[0] && w.gaze_valid[1]);
        assert!((w.gaze[0][0] - 0.1000061).abs() < 1e-4);
        assert!((w.gaze[1][1] - 0.19999695).abs() < 1e-4);
        assert_eq!(w.origin_mm[0], [1.0, 2.0, 3.0]);
        assert_eq!(w.origin_mm[1], [4.0, 5.0, 6.0]);
        assert_eq!(w.pupil_mm, [3.25, 4.5]);
        assert_eq!(w.pupil_valid, [true, true]);
        assert_eq!(w.openness_valid, [true, true]);
        assert!((w.pupil_pos[0][0] - 0.3999939).abs() < 1e-4);
        assert!((w.pupil_pos[1][1] - 0.5499878).abs() < 1e-4);
    }
}
