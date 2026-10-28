//! Lossless-at-the-adapter-boundary evidence for the standalone recorder.
//! Flags are separate from values. Raw wearable messages allow later decoder
//! audits; these are data messages only, never authentication/control traffic.
use sranibro_rs::core::types::GazeSample;
use std::fmt::Write as _;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::Arc;

pub const MAX_ROWS: usize = 10_000;
pub const MAX_PACKET_BYTES: usize = 16_384;

#[derive(Clone)]
pub struct EvidenceRow {
    pub tobii: Option<crate::tobii_reference::ReferenceRow>,
    pub generation: [u64; 2],
    pub camera_age_us: [u64; 2],
    pub sample: GazeSample,
    pub openness_age_us: [Option<u64>; 2],
    pub openness_timestamp_us: [Option<u64>; 2],
    pub gaze_age_us: [Option<u64>; 2],
    pub packet: Option<Arc<[u8]>>,
    pub packet_age_us: Option<u64>,
}

fn optional(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

pub fn csv(rows: &[EvidenceRow]) -> String {
    let mut out = String::from("index,eye,frame_generation,camera_age_us,openness_raw,openness_bits,openness_reported,openness_valid,openness_age_us,openness_timestamp_us,gaze_reported,gaze_valid,gaze_age_us,gaze_x,gaze_y,gaze_z,pupil_pos_reported,pupil_pos_valid,pupil_x,pupil_y,wearable_packet,wearable_packet_age_us\n");
    for (index, row) in rows.iter().enumerate() {
        for (eye, sample) in [row.sample.left, row.sample.right].iter().enumerate() {
            // Nonfinite values remain explicit; bit representation also preserves
            // exact decoder output. A false validity flag NEVER changes the value.
            writeln!(
                out,
                "{index},{},{},{},{},{:08x},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                if eye == 0 { "left" } else { "right" },
                row.generation[eye],
                row.camera_age_us[eye],
                sample.openness,
                sample.openness.to_bits(),
                sample.openness_reported,
                sample.openness_valid,
                optional(row.openness_age_us[eye]),
                optional(row.openness_timestamp_us[eye]),
                sample.gaze_reported,
                sample.gaze_valid,
                optional(row.gaze_age_us[eye]),
                sample.gaze[0],
                sample.gaze[1],
                sample.gaze[2],
                sample.pupil_pos_reported,
                sample.pupil_pos_valid,
                sample.pupil_pos[0],
                sample.pupil_pos[1],
                if row.packet.is_some() {
                    format!("native_packets/{index:06}.bin")
                } else {
                    String::new()
                },
                optional(row.packet_age_us)
            )
            .unwrap();
        }
    }
    out
}

pub fn append_to_archive(path: &Path, rows: &[EvidenceRow]) -> io::Result<()> {
    if rows.is_empty() || rows.len() > MAX_ROWS {
        return Err(io::Error::other(
            "native evidence count outside recorder limits",
        ));
    }
    // Verify alignment before modifying the incomplete archive.
    {
        let mut archive = zip::ZipArchive::new(std::fs::File::open(path)?)?;
        let mut samples = String::new();
        archive
            .by_name("samples.csv")?
            .read_to_string(&mut samples)?;
        let recorded: Vec<_> = samples.lines().skip(1).collect();
        if recorded.len() != rows.len()
            || recorded
                .iter()
                .enumerate()
                .any(|(i, line)| line.split(',').next() != Some(i.to_string().as_str()))
        {
            return Err(io::Error::other("image/native evidence index mismatch"));
        }
    }
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)?;
    let mut archive = zip::ZipWriter::new_append(file)?;
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    archive.start_file("native_evidence.csv", options)?;
    archive.write_all(csv(rows).as_bytes())?;
    archive.start_file("NATIVE-EVIDENCE.txt", options)?;
    archive.write_all(b"Schema: xr5_native_evidence_v1\nJoin index/eye to samples.csv and frames. Values and reported/valid flags are kept separately. openness_valid is the existing adapter interpretation, NOT an independently verified eye-present or closed classifier. TLV decoders may derive validity from numeric range; native_packets preserves original wearable data (stream 1285) for re-decoding. Age columns use local monotonic receive times, not hardware exposure synchronization. Latest packet and merged sample are independently timestamped and need not be the same event. Stale values are retained with their age; missing age means never received. A gaze-only event does not refresh openness. No native value is a training label. No authentication/control packets are included. Missing packet means no wearable data arrived or its size exceeded the bounded capture limit.\n")?;
    for (index, row) in rows.iter().enumerate() {
        if let Some(packet) = &row.packet {
            if packet.len() > MAX_PACKET_BYTES {
                return Err(io::Error::other(
                    "wearable packet exceeds bounded capture limit",
                ));
            }
            archive.start_file(format!("native_packets/{index:06}.bin"), options)?;
            archive.write_all(packet)?;
        }
    }
    crate::tobii_reference::write_archive(
        &mut archive,
        &rows.iter().map(|r| r.tobii.clone()).collect::<Vec<_>>(),
    )?;
    archive.finish()?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn row() -> EvidenceRow {
        EvidenceRow {
            tobii: None,
            generation: [1, 2],
            camera_age_us: [10, 20],
            sample: GazeSample::default(),
            openness_age_us: [Some(170_000), None],
            openness_timestamp_us: [Some(123), None],
            gaze_age_us: [None, None],
            packet: None,
            packet_age_us: None,
        }
    }
    #[test]
    fn disable_does_not_replace_raw_value_with_zero() {
        let mut r = row();
        r.sample.left.openness = 0.75;
        r.sample.left.openness_reported = true;
        let text = csv(&[r]);
        assert!(text.contains(",0.75,3f400000,true,false,170000,123,"));
        assert!(text.contains("0,right,2,20,0,00000000,false,false,,,"));
    }
    #[test]
    fn nonfinite_bits_are_preserved() {
        let mut r = row();
        r.sample.left.openness = f32::from_bits(0x7fc01234);
        assert!(csv(&[r]).contains("NaN,7fc01234"));
    }
    #[test]
    fn packet_link_and_staleness_are_explicit() {
        let mut r = row();
        r.packet = Some(Arc::from([1u8, 2, 3]));
        r.packet_age_us = Some(200_000);
        assert!(csv(&[r]).contains("native_packets/000000.bin,200000"));
    }
    #[test]
    fn zip_roundtrip_preserves_packet_and_rejects_wrong_row_count() {
        let path =
            std::env::temp_dir().join(format!("xr5-evidence-test-{}.zip", std::process::id()));
        let file = std::fs::File::create(&path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        zip.start_file("samples.csv", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(b"index,kind\n0,neutral\n").unwrap();
        zip.finish().unwrap();
        assert!(append_to_archive(&path, &[row(), row()]).is_err());
        let mut r = row();
        r.packet = Some(Arc::from([5u8, 6, 7]));
        append_to_archive(&path, &[r]).unwrap();
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut packet = Vec::new();
        zip.by_name("native_packets/000000.bin")
            .unwrap()
            .read_to_end(&mut packet)
            .unwrap();
        assert_eq!(packet, [5, 6, 7]);
        assert!(zip.by_name("NATIVE-EVIDENCE.txt").is_ok());
        drop(zip);
        std::fs::remove_file(path).unwrap();
    }
}
