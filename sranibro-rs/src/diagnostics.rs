//! Privacy-conscious support bundle writer.
//!
//! The archive intentionally omits `sranibro.toml`: it can contain local asset
//! paths and user names.  The caller supplies a hardware/settings summary, the
//! persisted eyelid calibration, and a bounded log tail.

use std::io::{self, Write};
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

/// Write a standards-compatible ZIP using the uncompressed "store" method.
/// This keeps support export dependency-free and makes corruption obvious.
pub fn write_zip(path: &Path, entries: &[(&str, &[u8])]) -> io::Result<()> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        if name.is_empty() || name.len() > u16::MAX as usize || data.len() > u32::MAX as usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "ZIP entry is too large",
            ));
        }
        let offset = u32::try_from(out.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
        let crc = crc32(data);
        let len = data.len() as u32;
        let name_len = name.len() as u16;

        put_u32(&mut out, 0x0403_4b50);
        put_u16(&mut out, 20);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u16(&mut out, 0);
        put_u32(&mut out, crc);
        put_u32(&mut out, len);
        put_u32(&mut out, len);
        put_u16(&mut out, name_len);
        put_u16(&mut out, 0);
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(data);

        put_u32(&mut central, 0x0201_4b50);
        put_u16(&mut central, 20);
        put_u16(&mut central, 20);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u32(&mut central, crc);
        put_u32(&mut central, len);
        put_u32(&mut central, len);
        put_u16(&mut central, name_len);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u16(&mut central, 0);
        put_u32(&mut central, 0);
        put_u32(&mut central, offset);
        central.extend_from_slice(name.as_bytes());
    }

    let central_offset = u32::try_from(out.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
    let central_len = u32::try_from(central.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "ZIP is too large"))?;
    out.extend_from_slice(&central);
    put_u32(&mut out, 0x0605_4b50);
    put_u16(&mut out, 0);
    put_u16(&mut out, 0);
    put_u16(&mut out, entries.len() as u16);
    put_u16(&mut out, entries.len() as u16);
    put_u32(&mut out, central_len);
    put_u32(&mut out, central_offset);
    put_u16(&mut out, 0);

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(path)?;
    file.write_all(&out)?;
    file.sync_all()
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
