//! `vpewake` — Vive Pro Eye eyechip native wake via the **Windows HID API** (2026-06-30).
//!
//! The vpescan map showed the always-present eyechip control interface is
//! `USB\VID_0BB4&PID_0309`, class HIDClass, driver **HidUsb** — i.e. it lives on the HID
//! stack, NOT WinUSB. testvpe.exe (Go/gousb) reaches it through libusb's Windows HID
//! backend. So SRanibro must drive the wake with the **HID API** (`HidD_SetFeature` +
//! `ReadFile` for input reports), which needs NO service pause and NO exclusive claim.
//!
//! Protocol (RE'd from testvpe, see project_vpe_native_wake): the chip emits 64-B input
//! reports beginning `03 D0 2C` (challenge); reply with a Feature report (ID 0x04, 64 B)
//! whose 3 checksum bytes are CRC32(serial)+challenge-tail. Sustain @500 ms and the chip
//! stays awake → the `2104:020F` WinUSB camera device appears for SRanibro to stream.

#![cfg(windows)]

use std::time::{Duration, Instant};

use windows_sys::core::GUID;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInterfaces, SetupDiGetClassDevsW,
    SetupDiGetDeviceInterfaceDetailW, DIGCF_DEVICEINTERFACE, DIGCF_PRESENT,
    SP_DEVICE_INTERFACE_DATA,
};
use windows_sys::Win32::Devices::HumanInterfaceDevice::{
    HidD_FreePreparsedData, HidD_GetAttributes, HidD_GetHidGuid, HidD_GetPreparsedData,
    HidD_SetFeature, HidP_GetCaps, HIDD_ATTRIBUTES, HIDP_CAPS,
};
use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0,
};
use windows_sys::Win32::Storage::FileSystem::{
    CreateFileW, ReadFile, FILE_FLAG_OVERLAPPED, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
};
use windows_sys::Win32::System::Threading::{CreateEventW, ResetEvent, WaitForSingleObject};
use windows_sys::Win32::System::IO::{CancelIo, GetOverlappedResult, OVERLAPPED};

use super::vpe_probe::{crc32_table, crc32_update};

const VPE_VID: u16 = 0x0BB4;
const VPE_PID: u16 = 0x0309;
const GENERIC_RW: u32 = 0xC000_0000;
const ERROR_IO_PENDING: u32 = 997;
const HIDP_STATUS_SUCCESS: i32 = 0x0011_0000;

/// Find + open the VPE HID control interface (0BB4:0309). Returns its raw HANDLE.
unsafe fn open_vpe_hid() -> Option<(HANDLE, String, bool)> {
    let mut hid_guid: GUID = std::mem::zeroed();
    HidD_GetHidGuid(&mut hid_guid);

    let dset = SetupDiGetClassDevsW(
        &hid_guid,
        std::ptr::null(),
        std::ptr::null_mut(),
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE,
    );
    if dset == -1 || dset == 0 {
        println!("[hid] SetupDiGetClassDevs(HID) failed: {}", GetLastError());
        return None;
    }

    let mut result = None;
    let mut idx = 0u32;
    loop {
        let mut ifd: SP_DEVICE_INTERFACE_DATA = std::mem::zeroed();
        ifd.cbSize = core::mem::size_of::<SP_DEVICE_INTERFACE_DATA>() as u32;
        if SetupDiEnumDeviceInterfaces(dset, std::ptr::null(), &hid_guid, idx, &mut ifd) == 0 {
            break;
        }
        idx += 1;

        // Get the device path (variable-length detail struct; cbSize=8 on x64).
        let mut detail = [0u8; 1024];
        // SP_DEVICE_INTERFACE_DETAIL_DATA_W { cbSize: u32, DevicePath: [u16;1] }
        detail[0..4].copy_from_slice(&8u32.to_le_bytes());
        let mut req = 0u32;
        if SetupDiGetDeviceInterfaceDetailW(
            dset,
            &mut ifd,
            detail.as_mut_ptr() as *mut _,
            detail.len() as u32,
            &mut req,
            std::ptr::null_mut(),
        ) == 0
        {
            continue;
        }
        // DevicePath wide string starts at byte offset 4.
        let path_u16: Vec<u16> = detail[4..]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .take_while(|&c| c != 0)
            .collect();
        if path_u16.is_empty() {
            continue;
        }
        let path: Vec<u16> = path_u16.iter().copied().chain(std::iter::once(0)).collect();

        // Open shared (HID class allows it); try RW then read-only fallback.
        let mut writable = true;
        let mut h = CreateFileW(
            path.as_ptr(),
            GENERIC_RW,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            std::ptr::null_mut(),
        );
        if h == INVALID_HANDLE_VALUE {
            writable = false;
            h = CreateFileW(
                path.as_ptr(),
                0,
                FILE_SHARE_READ | FILE_SHARE_WRITE,
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            );
        }
        if h == INVALID_HANDLE_VALUE {
            continue;
        }
        let mut attr: HIDD_ATTRIBUTES = std::mem::zeroed();
        attr.Size = core::mem::size_of::<HIDD_ATTRIBUTES>() as u32;
        if HidD_GetAttributes(h, &mut attr) != 0
            && attr.VendorID == VPE_VID
            && attr.ProductID == VPE_PID
        {
            let p = String::from_utf16_lossy(&path[..path.len() - 1]);
            result = Some((h, p, writable));
            break;
        }
        CloseHandle(h);
    }
    SetupDiDestroyDeviceInfoList(dset);
    result
}

unsafe fn caps(h: HANDLE) -> Option<HIDP_CAPS> {
    let mut pp: isize = 0;
    if HidD_GetPreparsedData(h, &mut pp) == 0 {
        return None;
    }
    let mut c: HIDP_CAPS = std::mem::zeroed();
    let st = HidP_GetCaps(pp, &mut c);
    HidD_FreePreparsedData(pp);
    if st == HIDP_STATUS_SUCCESS {
        Some(c)
    } else {
        None
    }
}

/// One overlapped input report read with a timeout. Ok(0) on timeout.
unsafe fn read_report(
    h: HANDLE,
    ev: HANDLE,
    buf: &mut [u8],
    timeout_ms: u32,
) -> std::io::Result<usize> {
    ResetEvent(ev);
    let mut ov: OVERLAPPED = std::mem::zeroed();
    ov.hEvent = ev;
    let mut got = 0u32;
    let ok = ReadFile(h, buf.as_mut_ptr(), buf.len() as u32, &mut got, &mut ov);
    if ok == 0 {
        let e = GetLastError();
        if e == ERROR_IO_PENDING {
            if WaitForSingleObject(ev, timeout_ms) == WAIT_OBJECT_0 {
                if GetOverlappedResult(h, &ov, &mut got, 0) == 0 {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("GetOverlappedResult: {}", GetLastError()),
                    ));
                }
            } else {
                // Timeout: cancel AND wait for the cancel to finish before `ov` (a stack
                // local) drops — otherwise the kernel writes to freed memory later.
                CancelIo(h);
                let _ = GetOverlappedResult(h, &ov, &mut got, 1);
                return Ok(0);
            }
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("ReadFile: {e}"),
            ));
        }
    }
    Ok(got as usize)
}

/// `vpewake` subcommand entry.
pub fn run() {
    println!("=== SRanibro VPE eyechip wake (HID API) ===");
    println!("Opens 0BB4:0309 via the HID stack (no service pause). Ctrl-C to stop.\n");

    let (h, ev) = unsafe {
        let Some((h, path, writable)) = open_vpe_hid() else {
            println!("[hid] VPE HID device 0BB4:0309 not found. Is the headset connected?");
            return;
        };
        println!("[hid] opened {path}");
        if !writable {
            println!("[hid] WARNING: opened WITHOUT write access -> SetFeature will fail. Close other openers (or run elevated).");
        }
        let ev = CreateEventW(std::ptr::null(), 1, 0, std::ptr::null());
        (h, ev)
    };

    // Report lengths + serial (for the CRC).
    let (in_len, feat_len) = unsafe {
        match caps(h) {
            Some(c) => {
                println!(
                    "[hid] report lengths: input={} output={} feature={}",
                    c.InputReportByteLength, c.OutputReportByteLength, c.FeatureReportByteLength
                );
                (
                    c.InputReportByteLength as usize,
                    c.FeatureReportByteLength as usize,
                )
            }
            None => {
                println!("[hid] HidP_GetCaps failed; assuming 64/64");
                (64, 64)
            }
        }
    };
    let serial = unsafe { hid_serial(h) };
    println!("[hid] serial string = {:?}", serial);
    let serial_bytes = serial.into_bytes();
    println!("[hid] serial bytes for CRC = {}", hex(&serial_bytes));
    if serial_bytes.is_empty() {
        println!("[hid] WARNING: empty serial -> CRC seed is empty; the keepalive will likely be rejected.");
    }

    let table = crc32_table();
    let in_n = in_len.max(64).max(1);
    let mut buf = vec![0u8; in_n];
    let (mut challenges, mut acks, mut reads, mut errs) = (0u32, 0u32, 0u32, 0u32);
    let start = Instant::now();
    let mut iter = 0u32;
    println!("[hid] running ~30 s (wear the headset)...\n");
    while start.elapsed() < Duration::from_secs(30) {
        iter += 1;
        let n = match unsafe { read_report(h, ev, &mut buf, 1000) } {
            Ok(n) => n,
            Err(e) => {
                errs += 1;
                if errs <= 5 {
                    println!("  [{iter}] read err: {e}");
                }
                continue;
            }
        };
        if n == 0 {
            continue;
        }
        reads += 1;
        // Input report may carry a leading report-ID byte; match the 03 D0 2C anywhere in the first 4.
        let head = &buf[..n.min(6)];
        let off = head.windows(3).position(|w| w == [0x03, 0xD0, 0x2C]);
        let Some(_p) = off else {
            if reads <= 8 {
                println!("  [{iter}] input n={n} head={} (no 03 D0 2C)", hex(head));
            }
            continue;
        };
        challenges += 1;

        // CRC = update(0, serial) then update(crc, report[n-2..n]).
        let mut crc = crc32_update(&table, 0, &serial_bytes);
        if n >= 2 {
            crc = crc32_update(&table, crc, &buf[n - 2..n]);
        }
        let mut rep = vec![0u8; feat_len.max(64)];
        rep[0] = 0x04;
        rep[1] = 0x78;
        rep[2] = 0x29;
        rep[3] = 0x38;
        rep[0x0a] = 0x08;
        rep[0x2f] = ((crc >> 25) | 0x80) as u8;
        let x = ((crc >> 9) & 0xFFFF) as u16;
        rep[0x30] = (x >> 8) as u8;
        rep[0x31] = (x & 0xFF) as u8;

        let ok = unsafe { HidD_SetFeature(h, rep.as_mut_ptr() as *mut _, rep.len() as u32) };
        if ok != 0 {
            acks += 1;
            if challenges <= 6 {
                println!(
                    "  [{iter}] challenge {} -> SetFeature OK (crc {crc:08x})",
                    hex(&buf[..4])
                );
            }
        } else if challenges <= 6 {
            println!("  [{iter}] SetFeature FAILED: {}", unsafe {
                GetLastError()
            });
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    println!("\n[result] {:.0}s: iters={iter} reads={reads} challenges(03D02C)={challenges} setfeature_ok={acks} read_errs={errs}", start.elapsed().as_secs_f32());
    if challenges > 0 && acks > 0 {
        println!("[result] PROMISING — challenges arrive and SetFeature is accepted.");
        println!("[result] Now check (another terminal): does `sranibro-rs vpescan` show 2104:020F (WinUSB) appear,");
        println!("[result] WITHOUT SRanipal running? If yes -> native wake works; we fold this into the adapter.");
    } else if reads > 0 {
        println!("[result] Reads work but no 03 D0 2C challenge yet — paste the head= lines so I can adjust the match.");
    } else {
        println!(
            "[result] No input reports — the chip may need an initial poke; paste this output."
        );
    }
    unsafe {
        if !ev.is_null() {
            CloseHandle(ev);
        }
        CloseHandle(h);
    }
}

unsafe fn hid_serial(h: HANDLE) -> String {
    use windows_sys::Win32::Devices::HumanInterfaceDevice::HidD_GetSerialNumberString;
    let mut buf = [0u16; 256];
    if HidD_GetSerialNumberString(h, buf.as_mut_ptr() as *mut _, (buf.len() * 2) as u32) != 0 {
        let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        String::from_utf16_lossy(&buf[..n])
    } else {
        String::new()
    }
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
