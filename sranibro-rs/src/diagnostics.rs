//! Privacy-conscious support bundle writer.
//!
//! The archive intentionally omits `sranibro.toml`: it can contain local asset
//! paths and user names.  The caller supplies a hardware/settings summary, the
//! persisted eyelid calibration, and a bounded log tail.

use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

fn put_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xedb8_8320 & (0u32.wrapping_sub(crc & 1)));
        }
    }
    !crc
}

/// Stable label for comparing reports from one physical unit without exporting
/// its real hardware serial.
pub fn pseudonymous_unit_id(serial: Option<&str>) -> String {
    serial
        .filter(|value| !value.is_empty())
        .map(|value| format!("unit-{:08x}", crc32(value.as_bytes())))
        .unwrap_or_else(|| "unavailable".into())
}

/// Remove the common personally identifying fragments from captured logs.  The
/// bundle still contains technical paths below the profile root, which are useful
/// for diagnosis, but not the Windows account name or real EyeChip serial.
pub fn redact_support_text(text: &str, eyechip_serial: Option<&str>) -> String {
    let mut redacted = text.to_string();
    if let Some(serial) = eyechip_serial.filter(|value| !value.is_empty()) {
        redacted = redacted.replace(serial, "%EYECHIP_SERIAL%");
    }
    for (var, marker) in [
        ("USERPROFILE", "%USERPROFILE%"),
        ("APPDATA", "%APPDATA%"),
        ("LOCALAPPDATA", "%LOCALAPPDATA%"),
        ("HOME", "%HOME%"),
    ] {
        if let Ok(value) = std::env::var(var) {
            if !value.is_empty() {
                redacted = redacted.replace(&value, marker);
                redacted = redacted.replace(&value.replace('\\', "/"), marker);
            }
        }
    }
    redacted
}

/// Incremental standards-compatible ZIP writer using the uncompressed "store" method.
/// Keeping only the central directory in memory lets calibration exports stream thousands
/// of already-compressed PNG frames without building a second copy of the whole recording.
pub struct StoredZipWriter {
    file: BufWriter<std::fs::File>,
    central: Vec<u8>,
    offset: u64,
    entries: u16,
}

impl StoredZipWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(Self {
            file: BufWriter::new(std::fs::File::create(path)?),
            central: Vec::new(),
            offset: 0,
            entries: 0,
        })
    }

    pub fn add(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
        if name.is_empty() || name.len() > u16::MAX as usize || data.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ZIP entry is too large",
            ));
        }
        if self.entries == u16::MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ZIP has too many entries",
            ));
        }
        let offset = u32::try_from(self.offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
        let crc = crc32(data);
        let len = data.len() as u32;
        let name_len = name.len() as u16;

        let mut header = Vec::with_capacity(30 + name.len());
        put_u32(&mut header, 0x0403_4b50);
        put_u16(&mut header, 20);
        put_u16(&mut header, 0);
        put_u16(&mut header, 0);
        put_u16(&mut header, 0);
        put_u16(&mut header, 0);
        put_u32(&mut header, crc);
        put_u32(&mut header, len);
        put_u32(&mut header, len);
        put_u16(&mut header, name_len);
        put_u16(&mut header, 0);
        header.extend_from_slice(name.as_bytes());
        self.file.write_all(&header)?;
        self.file.write_all(data)?;
        self.offset = self
            .offset
            .checked_add(header.len() as u64)
            .and_then(|value| value.checked_add(data.len() as u64))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;

        put_u32(&mut self.central, 0x0201_4b50);
        put_u16(&mut self.central, 20);
        put_u16(&mut self.central, 20);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u32(&mut self.central, crc);
        put_u32(&mut self.central, len);
        put_u32(&mut self.central, len);
        put_u16(&mut self.central, name_len);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u16(&mut self.central, 0);
        put_u32(&mut self.central, 0);
        put_u32(&mut self.central, offset);
        self.central.extend_from_slice(name.as_bytes());
        self.entries += 1;
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        let central_offset = u32::try_from(self.offset)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
        let central_len = u32::try_from(self.central.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
        self.file.write_all(&self.central)?;
        let mut end = Vec::with_capacity(22);
        put_u32(&mut end, 0x0605_4b50);
        put_u16(&mut end, 0);
        put_u16(&mut end, 0);
        put_u16(&mut end, self.entries);
        put_u16(&mut end, self.entries);
        put_u32(&mut end, central_len);
        put_u32(&mut end, central_offset);
        put_u16(&mut end, 0);
        self.file.write_all(&end)?;
        self.file.flush()?;
        self.file.get_ref().sync_all()
    }
}

/// Write a small standards-compatible ZIP using the incremental writer above.
pub fn write_zip(path: &Path, entries: &[(&str, &[u8])]) -> io::Result<()> {
    let mut writer = StoredZipWriter::create(path)?;
    for (name, data) in entries {
        writer.add(name, data)?;
    }
    writer.finish()
}

pub fn export_support_bundle(summary: &str, log_tail: &str) -> io::Result<PathBuf> {
    let base = crate::config::base_dir();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = base.join(format!("sranibro_support_{stamp}.zip"));
    let calibration = std::fs::read(crate::config::calib_path()).unwrap_or_default();
    let entries = [
        ("diagnostics.txt", summary.as_bytes()),
        ("log-tail.txt", log_tail.as_bytes()),
        ("sranibro_calib.toml", calibration.as_slice()),
    ];
    write_zip(&path, &entries)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc_matches_the_standard_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn store_zip_has_local_central_and_end_records() {
        let path =
            std::env::temp_dir().join(format!("sranibro_zip_test_{}.zip", std::process::id()));
        write_zip(&path, &[("hello.txt", b"hello"), ("empty.txt", b"")]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.starts_with(&0x0403_4b50u32.to_le_bytes()));
        assert!(bytes.windows(9).any(|w| w == b"hello.txt"));
        assert!(bytes.windows(4).any(|w| w == 0x0201_4b50u32.to_le_bytes()));
        assert!(bytes.windows(4).any(|w| w == 0x0605_4b50u32.to_le_bytes()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn support_text_hides_serial_and_profile_root() {
        let profile = std::env::var("USERPROFILE").unwrap_or_else(|_| "C:\\Users\\someone".into());
        let text = format!("opened XR5DA-SECRET from {profile}\\assets\\model.bin");
        let redacted = redact_support_text(&text, Some("XR5DA-SECRET"));
        assert!(!redacted.contains("XR5DA-SECRET"));
        assert!(!redacted.contains(&profile));
        assert!(redacted.contains("%EYECHIP_SERIAL%"));
        assert_eq!(
            pseudonymous_unit_id(Some("XR5DA-SECRET")),
            pseudonymous_unit_id(Some("XR5DA-SECRET"))
        );
    }
}
