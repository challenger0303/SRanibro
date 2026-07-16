//! `vpetest` — Vive Pro Eye eyechip native-wake VALIDATOR (diagnostic, 2026-06-30).
//!
//! Goal: prove (or disprove) that SRanibro can keep the VPE eyechip registered over
//! USB by itself — replacing the "run SRanipal in the background" workaround. It uses
//! the SAME WinUSB transport SRanibro ships, so a pass here means the real adapter can
//! do it too.
//!
//! Protocol RE'd from GhostIam's `testvpe.exe` (Go/gousb) — see project_vpe_native_wake:
//! every ~500 ms the chip emits a 64-B interrupt-IN "challenge" starting `03 D0 2C`; the
//! host must reply with a HID **SET_REPORT** (bmRequestType 0x21, bRequest 0x09,
//! wValue 0x0304, wIndex = HID interface) carrying a 64-B feature report whose 3 checksum
//! bytes are a CRC32(serial)+challenge-tail fold. Sustain it and the chip stays alive.
//!
//! This tool is DIAGNOSTIC: it dumps the device/interface/endpoint/serial layout (the
//! unknowns the RE couldn't pin without hardware) and then runs the keepalive for ~30 s,
//! logging every step. Even if the keepalive guess is slightly off, the dumped layout is
//! what we need to finish the real adapter. It does not change anything permanently; run
//! `sranibro-rs restore` afterwards to hand the chip back to the Tobii runtime.

#![cfg(windows)]

use std::io;
use std::time::{Duration, Instant};

use windows_sys::core::GUID;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_Device_Interface_ListW, CM_Get_Device_Interface_List_SizeW,
    SetupDiDestroyDeviceInfoList, SetupDiEnumDeviceInfo, SetupDiGetClassDevsW,
    SetupDiGetDeviceInstanceIdW, SetupDiGetDeviceRegistryPropertyW,
    CM_GET_DEVICE_INTERFACE_LIST_PRESENT, DIGCF_ALLCLASSES, DIGCF_PRESENT, HDEVINFO,
    SP_DEVINFO_DATA,
};
use windows_sys::Win32::Devices::Usb::{
    WinUsb_ControlTransfer, WinUsb_Free, WinUsb_GetAssociatedInterface, WinUsb_Initialize,
    WinUsb_QueryInterfaceSettings, WinUsb_QueryPipe, WinUsb_ReadPipe, WinUsb_SetPipePolicy,
    PIPE_TRANSFER_TIMEOUT, USB_INTERFACE_DESCRIPTOR, WINUSB_INTERFACE_HANDLE,
    WINUSB_PIPE_INFORMATION, WINUSB_SETUP_PACKET,
};
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_FLAG_OVERLAPPED, OPEN_EXISTING};

/// EyeChip device-interface GUID {85C0F97C-E2B1-422A-92A9-5F96072E79D8} (same as usb.rs;
/// Sepfox confirmed this resolves to the VPE eyechip too — VPE runs as `pimax_vr4`).
const GUID_EYECHIP: GUID = GUID {
    data1: 0x85C0_F97C,
    data2: 0xE2B1,
    data3: 0x422A,
    data4: [0x92, 0xA9, 0x5F, 0x96, 0x07, 0x2E, 0x79, 0xD8],
};
const GENERIC_RW: u32 = 0xC000_0000;
const ERROR_SEM_TIMEOUT: u32 = 121;
const USBD_PIPE_TYPE_INTERRUPT: i32 = 3;

// ---- CRC32 IEEE, reflected, RAW running update (no pre/post inversion) ----
// Matches Go's `hash/crc32.update` called directly (NOT ChecksumIEEE). Seed is 0.
pub(crate) fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0u32;
    while i < 256 {
        let mut c = i;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i as usize] = c;
        i += 1;
    }
    t
}
pub(crate) fn crc32_update(table: &[u32; 256], mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc = (crc >> 8) ^ table[((crc ^ b as u32) & 0xff) as usize];
    }
    crc
}

fn last_err(ctx: &str) -> io::Error {
    let e = unsafe { GetLastError() };
    io::Error::new(io::ErrorKind::Other, format!("{ctx}: win32 error {e}"))
}

/// First present EyeChip interface path (NUL-terminated UTF-16), like usb.rs::find_device.
fn find_eyechip() -> io::Result<Vec<u16>> {
    let mut len: u32 = 0;
    let cr = unsafe {
        CM_Get_Device_Interface_List_SizeW(
            &mut len,
            &GUID_EYECHIP,
            std::ptr::null(),
            CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
        )
    };
    if cr != 0 || len <= 1 {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "EyeChip interface not present (connect the VPE and stop the Tobii service)",
        ));
    }
    let mut buf = vec![0u16; len as usize];
    let cr = unsafe {
        CM_Get_Device_Interface_ListW(
            &GUID_EYECHIP,
            std::ptr::null(),
            buf.as_mut_ptr(),
            len,
            CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
        )
    };
    if cr != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "CM_Get_Device_Interface_ListW failed",
        ));
    }
    Ok(buf)
}

unsafe fn ctrl_in(
    h: WINUSB_INTERFACE_HANDLE,
    rtype: u8,
    req: u8,
    val: u16,
    idx: u16,
    len: u16,
) -> io::Result<Vec<u8>> {
    let setup = WINUSB_SETUP_PACKET {
        RequestType: rtype,
        Request: req,
        Value: val,
        Index: idx,
        Length: len,
    };
    let mut buf = vec![0u8; len as usize];
    let mut got = 0u32;
    let ok = WinUsb_ControlTransfer(
        h,
        setup,
        buf.as_mut_ptr(),
        buf.len() as u32,
        &mut got,
        std::ptr::null_mut(),
    );
    if ok == 0 {
        return Err(last_err("WinUsb_ControlTransfer(in)"));
    }
    buf.truncate(got as usize);
    Ok(buf)
}

unsafe fn ctrl_out(
    h: WINUSB_INTERFACE_HANDLE,
    rtype: u8,
    req: u8,
    val: u16,
    idx: u16,
    data: &[u8],
) -> io::Result<()> {
    let setup = WINUSB_SETUP_PACKET {
        RequestType: rtype,
        Request: req,
        Value: val,
        Index: idx,
        Length: data.len() as u16,
    };
    let mut buf = data.to_vec();
    let mut got = 0u32;
    let ok = WinUsb_ControlTransfer(
        h,
        setup,
        buf.as_mut_ptr(),
        buf.len() as u32,
        &mut got,
        std::ptr::null_mut(),
    );
    if ok == 0 {
        return Err(last_err("WinUsb_ControlTransfer(out)"));
    }
    Ok(())
}

unsafe fn set_timeout(h: WINUSB_INTERFACE_HANDLE, ep: u8, ms: u32) {
    let v = ms;
    WinUsb_SetPipePolicy(
        h,
        ep,
        PIPE_TRANSFER_TIMEOUT,
        4,
        &v as *const u32 as *const core::ffi::c_void,
    );
}

unsafe fn read_pipe(h: WINUSB_INTERFACE_HANDLE, ep: u8, buf: &mut [u8]) -> io::Result<usize> {
    let mut got = 0u32;
    let ok = WinUsb_ReadPipe(
        h,
        ep,
        buf.as_mut_ptr(),
        buf.len() as u32,
        &mut got,
        std::ptr::null_mut(),
    );
    if ok == 0 {
        let e = GetLastError();
        if e == ERROR_SEM_TIMEOUT {
            return Ok(0);
        }
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("WinUsb_ReadPipe(ep 0x{ep:02x}): error {e}"),
        ));
    }
    Ok(got as usize)
}

/// Decode a USB string descriptor (UTF-16LE after the 2-byte header) to a String.
fn decode_string_desc(d: &[u8]) -> String {
    if d.len() < 2 {
        return String::new();
    }
    let n = (d[0] as usize).min(d.len());
    let units: Vec<u16> = d[2..n]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    String::from_utf16_lossy(&units)
}

/// One discovered interface: its WinUSB handle + descriptor + pipes.
struct IfaceInfo {
    handle: WINUSB_INTERFACE_HANDLE,
    number: u8,
    class: u8,
    pipes: Vec<WINUSB_PIPE_INFORMATION>,
}

unsafe fn query_iface(h: WINUSB_INTERFACE_HANDLE) -> Option<IfaceInfo> {
    let mut desc: USB_INTERFACE_DESCRIPTOR = std::mem::zeroed();
    if WinUsb_QueryInterfaceSettings(h, 0, &mut desc) == 0 {
        return None;
    }
    let mut pipes = Vec::new();
    for pi in 0..desc.bNumEndpoints {
        let mut info: WINUSB_PIPE_INFORMATION = std::mem::zeroed();
        if WinUsb_QueryPipe(h, 0, pi, &mut info) != 0 {
            pipes.push(info);
        }
    }
    Some(IfaceInfo {
        handle: h,
        number: desc.bInterfaceNumber,
        class: desc.bInterfaceClass,
        pipes,
    })
}

/// Entry point for the `vpetest` subcommand.
pub fn run() {
    println!("=== SRanibro VPE eyechip wake VALIDATOR ===");
    println!("(diagnostic only; run `sranibro-rs restore` afterwards to hand the chip back)\n");

    // Read-only interface/driver map FIRST (before any service change), so we always get
    // the layout even if the exclusive open below fails.
    println!("[scan] present eyechip USB interfaces + drivers:");
    unsafe { scan_usb() };
    println!();

    // Free the EyeChip from the Tobii runtime so WinUSB can claim it exclusively.
    crate::platform::ensure_capture_ready();

    let path = match find_eyechip() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[open] {e}");
            eprintln!("[open] If a VPE is connected, the Tobii service may still hold it — try `sranibro-rs custom` first.");
            return;
        }
    };

    let file: HANDLE = unsafe {
        CreateFileW(
            path.as_ptr(),
            GENERIC_RW,
            0,
            std::ptr::null(),
            OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED,
            std::ptr::null_mut(),
        )
    };
    if file == INVALID_HANDLE_VALUE || file.is_null() {
        eprintln!("[open] {} (the interface is busy — is SRanipal or the Tobii service still holding it?)", last_err("CreateFileW"));
        return;
    }
    let mut base: WINUSB_INTERFACE_HANDLE = std::ptr::null_mut();
    if unsafe { WinUsb_Initialize(file, &mut base) } == 0 {
        eprintln!("[open] {}", last_err("WinUsb_Initialize"));
        unsafe { CloseHandle(file) };
        return;
    }
    println!("[open] WinUSB interface opened.");

    // --- Device descriptor: confirm VID/PID + find the serial-string index. ---
    let mut serial_bytes: Vec<u8> = Vec::new();
    match unsafe { ctrl_in(base, 0x80, 0x06, 0x0100, 0x0000, 18) } {
        Ok(dd) if dd.len() >= 18 => {
            let vid = u16::from_le_bytes([dd[8], dd[9]]);
            let pid = u16::from_le_bytes([dd[10], dd[11]]);
            let iserial = dd[16];
            println!("[desc] device: VID 0x{vid:04X} PID 0x{pid:04X}  iSerial idx={iserial}");
            if vid != 0x0BB4 || pid != 0x0309 {
                println!("[desc] WARNING: expected VPE 0BB4:0309 — this may be a different (e.g. Pimax) eyechip.");
            }
            if iserial != 0 {
                match unsafe { ctrl_in(base, 0x80, 0x06, 0x0300 | iserial as u16, 0x0409, 255) } {
                    Ok(sd) => {
                        let s = decode_string_desc(&sd);
                        serial_bytes = s.clone().into_bytes();
                        println!("[desc] serial string = {:?}", s);
                        println!("[desc] serial raw desc = {}", hex(&sd));
                        println!("[desc] serial bytes used for CRC = {}", hex(&serial_bytes));
                    }
                    Err(e) => println!("[desc] could not read serial string descriptor: {e}"),
                }
            } else {
                println!("[desc] iSerial=0 (no serial string) — CRC pass-1 input will be empty; flag if keepalive is rejected.");
            }
        }
        Ok(dd) => println!(
            "[desc] short device descriptor ({} bytes): {}",
            dd.len(),
            hex(&dd)
        ),
        Err(e) => println!("[desc] GET_DESCRIPTOR(device) failed: {e}"),
    }

    // --- Enumerate the base interface + every associated interface; dump pipes. ---
    println!("\n[iface] enumerating interfaces (base + associated):");
    let mut ifaces: Vec<IfaceInfo> = Vec::new();
    if let Some(info) = unsafe { query_iface(base) } {
        print_iface("base", &info);
        ifaces.push(info);
    }
    let mut ai = 0u8;
    loop {
        let mut h: WINUSB_INTERFACE_HANDLE = std::ptr::null_mut();
        if unsafe { WinUsb_GetAssociatedInterface(base, ai, &mut h) } == 0 {
            break; // no more associated interfaces
        }
        if let Some(info) = unsafe { query_iface(h) } {
            print_iface(&format!("assoc[{ai}]"), &info);
            ifaces.push(info);
        }
        ai += 1;
        if ai > 16 {
            break;
        }
    }

    // --- Pick the keepalive target: an interface with an interrupt-IN endpoint
    //     (prefer HID class 3). That endpoint carries the chip's challenge. ---
    let target = ifaces
        .iter()
        .find(|i| i.class == 3 && interrupt_in(i).is_some())
        .or_else(|| ifaces.iter().find(|i| interrupt_in(i).is_some()));
    let Some(t) = target else {
        println!("\n[keepalive] NO interrupt-IN endpoint found on any WinUSB-claimable interface.");
        println!("[keepalive] => the HID challenge interface is likely bound to hidclass.sys, not WinUSB.");
        println!("[keepalive] This is the key finding: native wake needs that interface WinUSB-accessible.");
        cleanup(base, file);
        return;
    };
    let in_ep = interrupt_in(t).unwrap();
    let iface_no = t.number;
    println!(
        "\n[keepalive] target interface #{iface_no} (class {}), interrupt-IN ep 0x{in_ep:02X}",
        t.class
    );
    unsafe { set_timeout(t.handle, in_ep, 200) };

    // --- Run the RE'd challenge-response keepalive for ~30 s. ---
    let table = crc32_table();
    let mut buf = [0u8; 64];
    let (mut challenges, mut acks, mut reads, mut errs) = (0u32, 0u32, 0u32, 0u32);
    let start = Instant::now();
    let mut iter = 0u32;
    println!("[keepalive] running ~30 s (wear the headset)...");
    while start.elapsed() < Duration::from_secs(30) {
        iter += 1;
        let n = match unsafe { read_pipe(t.handle, in_ep, &mut buf) } {
            Ok(n) => n,
            Err(e) => {
                errs += 1;
                if errs <= 5 {
                    println!("  [{iter}] read err: {e}");
                }
                std::thread::sleep(Duration::from_millis(100));
                continue;
            }
        };
        if n == 0 {
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        reads += 1;
        let valid = n >= 3 && buf[0] == 0x03 && buf[1] == 0xD0 && buf[2] == 0x2C;
        if !valid {
            if reads <= 8 {
                println!(
                    "  [{iter}] IN n={n} head={} (no 03 D0 2C challenge)",
                    hex(&buf[..n.min(8)])
                );
            }
            std::thread::sleep(Duration::from_millis(50));
            continue;
        }
        challenges += 1;

        // CRC = update(0, serial) then update(crc, challenge[n-2..n]) — raw reflected, no XOR.
        let mut crc = crc32_update(&table, 0, &serial_bytes);
        if n >= 2 {
            crc = crc32_update(&table, crc, &buf[n - 2..n]);
        }
        let mut rep = [0u8; 64];
        rep[0] = 0x04;
        rep[1] = 0x78;
        rep[2] = 0x29;
        rep[3] = 0x38;
        rep[0x0a] = 0x08;
        rep[0x2f] = ((crc >> 25) | 0x80) as u8;
        let x = ((crc >> 9) & 0xFFFF) as u16;
        rep[0x30] = (x >> 8) as u8;
        rep[0x31] = (x & 0xFF) as u8;

        match unsafe { ctrl_out(t.handle, 0x21, 0x09, 0x0304, iface_no as u16, &rep) } {
            Ok(()) => {
                acks += 1;
                if challenges <= 6 {
                    println!(
                        "  [{iter}] challenge {} -> SET_REPORT ok (crc {:08x})",
                        hex(&buf[..4]),
                        crc
                    );
                }
            }
            Err(e) => {
                if challenges <= 6 {
                    println!("  [{iter}] SET_REPORT FAILED: {e}");
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }

    println!("\n[result] {:.0}s: iters={iter} reads={reads} challenges(03D02C)={challenges} set_report_ok={acks} read_errs={errs}", start.elapsed().as_secs_f32());
    if challenges > 0 && acks > 0 {
        println!("[result] LOOKS GOOD: the chip is emitting challenges and accepting our SET_REPORT replies.");
        println!("[result] Watch whether the eyechip stays registered without SRanipal — if so, native wake is viable.");
    } else if reads > 0 && challenges == 0 {
        println!("[result] Reads work but NO 03 D0 2C challenge — the interrupt-IN/report format differs; check the [iface] dump above.");
    } else {
        println!("[result] No IN data — wrong endpoint/interface, or the chip needs SRanipal's init first. See the [iface] dump.");
    }
    cleanup(base, file);
    println!("\nDone. Run `sranibro-rs restore` to re-enable the Tobii runtime.");
}

fn interrupt_in(i: &IfaceInfo) -> Option<u8> {
    i.pipes
        .iter()
        .find(|p| p.PipeType == USBD_PIPE_TYPE_INTERRUPT && p.PipeId & 0x80 != 0)
        .map(|p| p.PipeId)
}

fn print_iface(tag: &str, i: &IfaceInfo) {
    let pipes: Vec<String> = i
        .pipes
        .iter()
        .map(|p| {
            let ty = match p.PipeType {
                0 => "ctrl",
                1 => "iso",
                2 => "bulk",
                3 => "intr",
                _ => "?",
            };
            let dir = if p.PipeId & 0x80 != 0 { "IN" } else { "OUT" };
            format!("0x{:02X}({ty},{dir},max{})", p.PipeId, p.MaximumPacketSize)
        })
        .collect();
    println!(
        "  {tag}: iface#{} class=0x{:02X} pipes=[{}]",
        i.number,
        i.class,
        pipes.join(", ")
    );
}

fn cleanup(base: WINUSB_INTERFACE_HANDLE, file: HANDLE) {
    unsafe {
        if !base.is_null() {
            WinUsb_Free(base);
        }
        if file != INVALID_HANDLE_VALUE && !file.is_null() {
            CloseHandle(file);
        }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

// ---- read-only SetupAPI scan: which USB interfaces the eyechip exposes + their drivers ----
const SPDRP_SERVICE: u32 = 4;
const SPDRP_CLASS: u32 = 7;

unsafe fn dev_instance_id(h: HDEVINFO, d: *mut SP_DEVINFO_DATA) -> String {
    let mut buf = [0u16; 512];
    let mut req = 0u32;
    if SetupDiGetDeviceInstanceIdW(h, d, buf.as_mut_ptr(), buf.len() as u32, &mut req) == 0 {
        return String::new();
    }
    let n = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    String::from_utf16_lossy(&buf[..n])
}

unsafe fn dev_prop(h: HDEVINFO, d: *mut SP_DEVINFO_DATA, prop: u32) -> String {
    let mut ty = 0u32;
    let mut buf = [0u8; 512];
    let mut req = 0u32;
    if SetupDiGetDeviceRegistryPropertyW(
        h,
        d,
        prop,
        &mut ty,
        buf.as_mut_ptr(),
        buf.len() as u32,
        &mut req,
    ) == 0
    {
        return String::new();
    }
    let u: Vec<u16> = buf[..(req as usize).min(buf.len())]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    let n = u.iter().position(|&c| c == 0).unwrap_or(u.len());
    String::from_utf16_lossy(&u[..n])
}

/// List every PRESENT USB device whose hardware id is the eyechip (VID 0BB4 = VPE,
/// 2104 = Pimax), with its interface number (MI), device class and bound driver. This
/// is the make-or-break map: it shows whether the HID (challenge) interface exists and
/// whether it is WinUSB-claimable (driver = WinUSB/libusbK) vs HID-stack-bound (HidUsb).
unsafe fn scan_usb() {
    let enumerator: Vec<u16> = "USB".encode_utf16().chain(std::iter::once(0)).collect();
    let h = SetupDiGetClassDevsW(
        std::ptr::null(),
        enumerator.as_ptr(),
        std::ptr::null_mut(),
        DIGCF_PRESENT | DIGCF_ALLCLASSES,
    );
    if h == -1 || h == 0 {
        println!("  [scan] SetupDiGetClassDevs failed: {}", GetLastError());
        return;
    }
    let mut idx = 0u32;
    let mut found = 0;
    loop {
        let mut data: SP_DEVINFO_DATA = std::mem::zeroed();
        data.cbSize = core::mem::size_of::<SP_DEVINFO_DATA>() as u32;
        if SetupDiEnumDeviceInfo(h, idx, &mut data) == 0 {
            break;
        }
        idx += 1;
        let id = dev_instance_id(h, &mut data);
        let up = id.to_uppercase();
        if up.contains("VID_0BB4") || up.contains("VID_2104") {
            let service = dev_prop(h, &mut data, SPDRP_SERVICE);
            let class = dev_prop(h, &mut data, SPDRP_CLASS);
            let mi = up
                .find("MI_")
                .map(|i| up[i..].chars().take(5).collect::<String>())
                .unwrap_or_else(|| "(single)".into());
            println!("  {id}");
            println!(
                "      iface={mi}  class={}  driver={}",
                if class.is_empty() { "?" } else { &class },
                if service.is_empty() {
                    "(none)"
                } else {
                    &service
                }
            );
            found += 1;
        }
    }
    if found == 0 {
        println!("  [scan] no VID_0BB4 (VPE) / VID_2104 (Pimax) USB devices present.");
    }
    SetupDiDestroyDeviceInfoList(h);
}

/// `vpescan` subcommand — READ-ONLY interface/driver map (no service changes). Run this
/// both WITH SRanipal running and WITHOUT, to compare which interfaces appear/vanish.
pub fn scan() {
    println!("=== VPE / eyechip USB interface map (read-only — nothing is changed) ===");
    println!("(driver = WinUSB/libusbK -> we can claim it; HidUsb/mshidkmdf -> HID-stack-bound)\n");
    unsafe { scan_usb() };
    println!("\nTip: run once with SRanipal running and once without — the interface that is\nALWAYS present (even dormant) is the one the native wake must open.");
}
