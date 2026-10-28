//! Recorder-only EyeChip1289 reference, never measured fixation or a fit label.
//! Keep raw columns/packet/status so adapter eye labels can be audited later.
use sranibro_rs::device::{
    gaze::{decode_gaze_1289, GazeData},
    ttp,
};
use std::{
    collections::{BTreeMap, VecDeque},
    fmt::Write as _,
    io::{self, Seek, Write},
    sync::Arc,
    time::{Duration, Instant},
};

pub const MAX_PACKET_BYTES: usize = 16_384;
pub const MAX_HISTORY: usize = 512;
pub const FRESHNESS: Duration = Duration::from_millis(150);

#[derive(Clone)]
pub struct Packet {
    pub sequence: u64,
    pub received: Instant,
    pub data: Option<GazeData>,
    pub raw: Option<Arc<[u8]>>,
}
#[derive(Default)]
pub struct ReferenceState {
    pub packet_count: u64,
    history: VecDeque<Arc<Packet>>,
}
#[derive(Clone)]
pub struct ReferenceRow {
    pub started: Instant,
    pub camera_received: [Instant; 2],
    pub generation: [u64; 2],
    pub packets: [Option<Arc<Packet>>; 2],
    pub swap_eyes: bool,
}
impl ReferenceState {
    pub fn receive(&mut self, bytes: &[u8], now: Instant) {
        // Only streaming gaze data, never control/authentication or other PIDs.
        if bytes.len() < 16 || ttp::pid(bytes) != ttp::PID_GAZE || bytes[4..8] != [0, 0, 0, 0] {
            return;
        }
        self.packet_count = self.packet_count.wrapping_add(1);
        let raw = (bytes.len() <= MAX_PACKET_BYTES).then(|| Arc::<[u8]>::from(bytes));
        let data = raw.as_ref().and_then(|b| decode_gaze_1289(b));
        if self.history.len() == MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(Arc::new(Packet {
            sequence: self.packet_count,
            received: now,
            data,
            raw,
        }));
    }
    fn preceding(&self, camera: Instant) -> Option<Arc<Packet>> {
        self.history
            .iter()
            .rev()
            .find(|p| p.received <= camera)
            .cloned()
    }
    pub fn row(
        &self,
        started: Instant,
        cameras: [Instant; 2],
        generation: [u64; 2],
        swap_eyes: bool,
    ) -> ReferenceRow {
        ReferenceRow {
            started,
            camera_received: cameras,
            generation,
            packets: cameras.map(|c| self.preceding(c)),
            swap_eyes,
        }
    }
    pub fn status(&self, now: Instant, swap_eyes: bool) -> [&'static str; 2] {
        std::array::from_fn(|eye| {
            reason(
                self.preceding(now).as_deref(),
                now,
                eye ^ usize::from(swap_eyes),
            )
        })
    }
}
fn plausible(data: &GazeData, eye: usize) -> bool {
    let sample = data.to_xr5_gaze_sample();
    if eye == 0 {
        sample.left.gaze_valid
    } else {
        sample.right.gaze_valid
    }
}
fn reason(packet: Option<&Packet>, camera: Instant, source_eye: usize) -> &'static str {
    let Some(p) = packet else {
        return "not received";
    };
    if p.raw.is_none() {
        return "packet too large";
    }
    let Some(data) = p.data.as_ref() else {
        return "decode failed";
    };
    if camera.saturating_duration_since(p.received) > FRESHNESS {
        return "stale";
    }
    if !plausible(data, source_eye) {
        return "invalid vector";
    }
    "available"
}
fn micros(t: Instant, origin: Instant) -> String {
    if t >= origin {
        t.duration_since(origin).as_micros().to_string()
    } else {
        format!("-{}", origin.duration_since(t).as_micros())
    }
}
fn values(data: Option<GazeData>, source_eye: usize) -> Vec<String> {
    let Some(d) = data else {
        return vec![String::new(); 18];
    };
    let raw = if source_eye == 0 { d.l_gaze } else { d.r_gaze };
    let mut out = vec![
        d.timestamp.to_string(),
        d.status.to_string(),
        plausible(&d, source_eye).to_string(),
    ];
    for v in raw {
        out.push(v.to_string());
        out.push(format!("{:08x}", v.to_bits()));
    }
    for v in d.combined {
        out.push(v.to_string());
        out.push(format!("{:08x}", v.to_bits()));
    }
    out.push(d.to_xr5_combined_gaze_sample().left.gaze_valid.to_string());
    out.push(d.convergence.to_string());
    out.push(format!("{:08x}", d.convergence.to_bits()));
    out
}
pub fn csv(rows: &[Option<ReferenceRow>]) -> String {
    let mut out=String::from("index,eye,frame_generation,camera_received_elapsed_us,reference_received_elapsed_us,reference_age_us,packet_sequence,packet_decoded,reference_fresh,reference_usable,reference_state,source_column,recording_swap_eyes,device_timestamp_us_raw,packet_status_raw,vector_plausible,gaze_x,gaze_x_bits,gaze_y,gaze_y_bits,gaze_z,gaze_z_bits,combined_x,combined_x_bits,combined_y,combined_y_bits,combined_z,combined_z_bits,combined_plausible,convergence_raw,convergence_bits,raw_packet\n");
    for (index, row) in rows.iter().enumerate() {
        for eye in 0..2 {
            let mut fields = vec![
                index.to_string(),
                if eye == 0 { "left" } else { "right" }.into(),
            ];
            if let Some(row) = row {
                let p = row.packets[eye].as_deref();
                let source_eye = eye ^ usize::from(row.swap_eyes);
                let state = reason(p, row.camera_received[eye], source_eye);
                fields.extend([
                    row.generation[eye].to_string(),
                    micros(row.camera_received[eye], row.started),
                    p.map(|p| micros(p.received, row.started))
                        .unwrap_or_default(),
                    p.map(|p| {
                        row.camera_received[eye]
                            .saturating_duration_since(p.received)
                            .as_micros()
                            .to_string()
                    })
                    .unwrap_or_default(),
                    p.map(|p| p.sequence.to_string()).unwrap_or_default(),
                    p.is_some_and(|p| p.data.is_some()).to_string(),
                    p.is_some_and(|p| {
                        row.camera_received[eye].saturating_duration_since(p.received) <= FRESHNESS
                    })
                    .to_string(),
                    (state == "available").to_string(),
                    state.into(),
                    (3 + source_eye).to_string(),
                    row.swap_eyes.to_string(),
                ]);
                fields.extend(values(p.and_then(|p| p.data), source_eye));
                fields.push(
                    p.filter(|p| p.raw.is_some())
                        .map(|p| format!("tobii_packets/{:016}.bin", p.sequence))
                        .unwrap_or_default(),
                );
            } else {
                fields.extend(vec![String::new(); 30]);
                fields[7] = "false".into();
                fields[8] = "false".into();
                fields[9] = "false".into();
                fields[10] = "not collected".into();
            }
            debug_assert_eq!(fields.len(), 32);
            writeln!(out, "{}", fields.join(",")).unwrap();
        }
    }
    out
}
pub fn write_archive<W: Write + Seek>(
    zip: &mut zip::ZipWriter<W>,
    rows: &[Option<ReferenceRow>],
) -> io::Result<()> {
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("tobii_gaze.csv", options)?;
    zip.write_all(csv(rows).as_bytes())?;
    zip.start_file("TOBII-GAZE.txt", options)?;
    zip.write_all(b"Schema: xr5_tobii_gaze_reference_v1\nSource: unmodified EyeChip/Tobii streaming PID1289, NOT the custom model. Two rows per stereo image pair; join index/eye/frame_generation to samples.csv and fixation.csv. Each eye uses the latest packet RECEIVED NO LATER THAN that camera image, with independent age. No exposure synchronization or hardware-clock synchronization is claimed. Negative receive elapsed time denotes a reference received before recording Start. No packet, undecodable packet, implausible vector and stale packet are explicit. Raw device timestamp/status/direction/combined/convergence and float bits are retained. reference_usable is ONLY adapter vector plausibility plus <=150ms receive age, not a native Tobii confidence/validity certification, user attention or fixation truth. Packet status is preserved uninterpreted; status0 alone is not invalid.\nThe existing decoder assigns raw column3 to its left slot and column4 to its right slot. source_column plus recording_swap_eyes records the exact adapter mapping; hardware eye assignment and coordinate frame remain unverified. No axis/sign conversion or automatic comparison with OpenVR target rays is performed. Combined gaze repeats one fused observation and is NOT independent left/right gaze. Convergence physical unit is unverified. Raw selected PID1289 packets are preserved once per sequence under tobii_packets; control/auth traffic and other PIDs are excluded. This is a frame-associated reference stream, not a complete continuous event log. Missing/stale Tobii gaze never blocks eye-image recording. No reference value is a training label.\n")?;
    let mut packets = BTreeMap::new();
    for row in rows.iter().flatten() {
        for p in row.packets.iter().flatten() {
            if let Some(raw) = &p.raw {
                packets.entry(p.sequence).or_insert(raw);
            }
        }
    }
    for (sequence, raw) in packets {
        zip.start_file(format!("tobii_packets/{sequence:016}.bin"), options)?;
        zip.write_all(raw)?;
    }
    Ok(())
}

/// Explicit simulation only; valid TTP fixed-point encoding exercises the decoder.
pub fn simulation_packet(timestamp: i64, valid: bool) -> Vec<u8> {
    fn enc(tag: u8, b: &[u8]) -> Vec<u8> {
        let mut o = vec![tag];
        o.extend((b.len() as u32).to_be_bytes());
        o.extend(b);
        o
    }
    let pro = |v: u32| enc(5, &v.to_be_bytes());
    let mut msg = vec![0; 26];
    msg[12..16].copy_from_slice(&ttp::PID_GAZE.to_be_bytes());
    msg.extend(pro(6 << 16));
    for col in 1u32..=6 {
        msg.extend(pro(ttp::GAZE_COLUMN_MAGIC));
        msg.extend(ttp::encode_u32(col));
        match col {
            1 => msg.extend(enc(6, &timestamp.to_be_bytes())),
            2 => msg.extend(ttp::encode_u32(0)),
            3..=5 => {
                msg.extend(pro(3 << 16));
                for v in if valid {
                    match col {
                        3 => [8192, -4096, 65536],
                        4 => [-16384, 8192, 65536],
                        _ => [0, 4096, 65536],
                    }
                } else {
                    [0; 3]
                } {
                    msg.extend(enc(3, &i32::to_be_bytes(v)));
                }
            }
            _ => msg.extend(enc(3, &131072_i32.to_be_bytes())),
        }
    }
    msg
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_is_explicit_and_not_zeros() {
        let s = ReferenceState::default();
        let t = Instant::now();
        let text = csv(&[Some(s.row(t, [t; 2], [1; 2], false))]);
        assert!(text.contains("not received"));
        let lines: Vec<_> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines.iter().all(|s| s.split(',').count() == 32));
    }
    #[test]
    fn status_zero_with_usable_direction_is_retained() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(123, true), t);
        assert_eq!(s.status(t, false), ["available"; 2]);
        let text = csv(&[Some(s.row(t, [t; 2], [1; 2], false))]);
        assert!(text.contains(",123,0,true,0.125,3e000000,"));
    }
    #[test]
    fn invalid_latest_does_not_resurrect_previous_good() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(1, true), t);
        s.receive(&simulation_packet(2, false), t + Duration::from_millis(1));
        assert_eq!(
            s.status(t + Duration::from_millis(2), false),
            ["invalid vector"; 2]
        );
    }
    #[test]
    fn future_gaze_is_never_paired_to_camera() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(1, true), t);
        s.receive(&simulation_packet(2, true), t + Duration::from_millis(20));
        let row = s.row(
            t,
            [t + Duration::from_millis(5), t + Duration::from_millis(25)],
            [1, 2],
            false,
        );
        assert_eq!(row.packets[0].as_ref().unwrap().sequence, 1);
        assert_eq!(row.packets[1].as_ref().unwrap().sequence, 2);
    }
    #[test]
    fn stale_reference_is_not_usable() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(1, true), t);
        assert_eq!(
            s.status(t + FRESHNESS + Duration::from_micros(1), false),
            ["stale"; 2]
        );
    }
    #[test]
    fn eye_swap_records_source_columns_without_flipping_vectors() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(1, true), t);
        let text = csv(&[Some(s.row(t, [t; 2], [1; 2], true))]);
        let left: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(left[11], "4");
        assert_eq!(left[16], "-0.25");
    }
    #[test]
    fn history_is_bounded() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        for i in 0..600 {
            s.receive(&simulation_packet(i, true), t);
        }
        assert_eq!(s.history.len(), MAX_HISTORY);
    }
    #[test]
    fn control_and_other_streams_are_excluded() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        let mut p = simulation_packet(1, true);
        p[4..8].copy_from_slice(&1u32.to_be_bytes());
        s.receive(&p, t);
        p[4..8].fill(0);
        p[12..16].copy_from_slice(&1285u32.to_be_bytes());
        s.receive(&p, t);
        assert_eq!(s.packet_count, 0);
    }
    #[test]
    fn malformed_latest_clears_availability() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(1, true), t);
        let mut bad = vec![0; 26];
        bad[12..16].copy_from_slice(&1289u32.to_be_bytes());
        s.receive(&bad, t);
        assert_eq!(s.status(t, false), ["decode failed"; 2]);
    }
    #[test]
    fn missing_fixture_rows_have_same_schema() {
        let text = csv(&[None]);
        assert!(text.lines().all(|s| s.split(',').count() == 32));
        assert!(text.contains("not collected"));
    }
    #[test]
    fn signed_device_clock_and_prestart_receipt_are_preserved() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        s.receive(&simulation_packet(-7, true), t);
        let text = csv(&[Some(s.row(
            t + Duration::from_millis(10),
            [t + Duration::from_millis(11); 2],
            [1; 2],
            false,
        ))]);
        let f: Vec<_> = text.lines().nth(1).unwrap().split(',').collect();
        assert_eq!(f[4], "-10000");
        assert_eq!(f[13], "-7");
        assert_eq!(f[5], "11000");
    }
    #[test]
    fn oversized_packet_is_explicit_and_not_saved() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        let mut packet = simulation_packet(1, true);
        packet.resize(MAX_PACKET_BYTES + 1, 0);
        s.receive(&packet, t);
        assert_eq!(s.status(t, false), ["packet too large"; 2]);
        assert!(s.history[0].raw.is_none());
    }
    #[test]
    fn raw_packet_archive_deduplicates_shared_eye_reference() {
        let t = Instant::now();
        let mut s = ReferenceState::default();
        let packet = simulation_packet(8, true);
        s.receive(&packet, t);
        let row = s.row(t, [t; 2], [1; 2], false);
        let mut zip = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
        write_archive(&mut zip, &[Some(row)]).unwrap();
        let mut zip = zip::ZipArchive::new(zip.finish().unwrap()).unwrap();
        assert_eq!(zip.len(), 3);
        let mut saved = Vec::new();
        use std::io::Read;
        zip.by_name("tobii_packets/0000000000000001.bin")
            .unwrap()
            .read_to_end(&mut saved)
            .unwrap();
        assert_eq!(saved, packet);
    }
}
