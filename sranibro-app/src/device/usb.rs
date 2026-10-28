//! WinUSB transport to the Tobii EyeChip — no libusb, no patched DLL.
//!
//! Ported from `winusb_device.py` (device enumeration + WinUSB I/O) and the
//! handshake in `tobii_usb_direct.py` (control init + channel upgrade + auth +
//! subscribe). Requires the Tobii platform service to be stopped (WinUSB takes
//! the interface exclusively):  `net stop "Tobii VR4PIMAXP3B Platform Runtime"`.

#![cfg(windows)]

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows_sys::core::GUID;
use windows_sys::Win32::Devices::DeviceAndDriverInstallation::{
    CM_Get_Device_Interface_ListW, CM_Get_Device_Interface_List_SizeW,
    CM_GET_DEVICE_INTERFACE_LIST_PRESENT,
};
use windows_sys::Win32::Devices::Usb::{
    WinUsb_ControlTransfer, WinUsb_Free, WinUsb_Initialize, WinUsb_ReadPipe, WinUsb_ResetPipe,
    WinUsb_SetPipePolicy, WinUsb_WritePipe, PIPE_TRANSFER_TIMEOUT, WINUSB_INTERFACE_HANDLE,
    WINUSB_SETUP_PACKET,
};
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{CreateFileW, FILE_FLAG_OVERLAPPED, OPEN_EXISTING};

use super::ttp::{build_packet, Deframer};

/// EyeChip device-interface GUID {85C0F97C-E2B1-422A-92A9-5F96072E79D8}.
const GUID_EYECHIP: GUID = GUID {
    data1: 0x85C0_F97C,
    data2: 0xE2B1,
    data3: 0x422A,
    data4: [0x92, 0xA9, 0x5F, 0x96, 0x07, 0x2E, 0x79, 0xD8],
};

const EP_IN: u8 = 0x83;
const EP_OUT: u8 = 0x05;
const GENERIC_RW: u32 = 0xC000_0000; // GENERIC_READ | GENERIC_WRITE
const ERROR_SEM_TIMEOUT: u32 = 121;

// The EyeChip firmware keeps the previous streaming session alive briefly after
// WinUsb_Free/CloseHandle return. Reopening during that interval succeeds at the
// Windows handle level but the first bulk read commonly fails with ERROR_GEN_FAILURE
// (31), or the authentication challenge is missing. Remember the close time inside
// this process so a live engine reload can give the firmware a quiet interval. A
// cold process start has no recorded close and therefore pays no delay.
const REOPEN_QUIET_PERIOD: Duration = Duration::from_secs(5);
static LAST_INTERFACE_CLOSE: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

fn last_interface_close() -> &'static Mutex<Option<Instant>> {
    LAST_INTERFACE_CLOSE.get_or_init(|| Mutex::new(None))
}

fn record_interface_close() {
    if let Ok(mut last) = last_interface_close().lock() {
        *last = Some(Instant::now());
    }
}

/// Remaining firmware quiet time before another WinUSB session should begin.
/// Used by the adapter thread so waiting remains stop-responsive and visible in
/// device status instead of blocking the UI thread.
pub fn reopen_quiet_remaining() -> Duration {
    let last = last_interface_close().lock().ok().and_then(|last| *last);
    quiet_remaining(last, Instant::now())
}

fn quiet_remaining(last: Option<Instant>, now: Instant) -> Duration {
    last.map(|closed| REOPEN_QUIET_PERIOD.saturating_sub(now.saturating_duration_since(closed)))
        .unwrap_or(Duration::ZERO)
}

/// An open WinUSB connection to the EyeChip, with a TTP deframer and msg-id seq.
pub struct UsbDevice {
    file: HANDLE,
    handle: WINUSB_INTERFACE_HANDLE,
    pub serial: String,
    msg_id: u32,
    deframer: Deframer,
}

// The raw handles are only ever used from the single thread that opens the
// device; we never share a UsbDevice across threads.
unsafe impl Send for UsbDevice {}

fn last_err(ctx: &str) -> io::Error {
    let e = unsafe { GetLastError() };
    io::Error::new(io::ErrorKind::Other, format!("{ctx}: win32 error {e}"))
}

impl UsbDevice {
    /// Enumerate + open the EyeChip and bring up the WinUSB interface.
    pub fn open() -> io::Result<Self> {
        let (path, serial) = find_device()?;

        // SAFETY: path is a NUL-terminated UTF-16 buffer from the CM API.
        let file = unsafe {
            CreateFileW(
                path.as_ptr(),
                GENERIC_RW,
                0, // exclusive (no sharing)
                std::ptr::null(),
                OPEN_EXISTING,
                FILE_FLAG_OVERLAPPED,
                std::ptr::null_mut(),
            )
        };
        if file == INVALID_HANDLE_VALUE || file.is_null() {
            return Err(last_err("CreateFileW(EyeChip)"));
        }

        let mut handle: WINUSB_INTERFACE_HANDLE = std::ptr::null_mut();
        if unsafe { WinUsb_Initialize(file, &mut handle) } == 0 {
            let e = last_err("WinUsb_Initialize");
            unsafe { CloseHandle(file) };
            record_interface_close();
            return Err(e);
        }

        let dev = UsbDevice {
            file,
            handle,
            serial,
            msg_id: 0,
            deframer: Deframer::new(),
        };
        unsafe {
            for ep in [EP_IN, EP_OUT, 0x04] {
                WinUsb_ResetPipe(dev.handle, ep);
            }
        }
        dev.set_timeout(EP_IN, 5000)?;
        dev.set_timeout(EP_OUT, 5000)?;
        Ok(dev)
    }

    fn set_timeout(&self, ep: u8, ms: u32) -> io::Result<()> {
        // Zero disables the WinUSB timeout. Never turn a short remaining deadline
        // into an unbounded native read by rounding it down to zero milliseconds.
        let v = ms.max(1);
        let ok = unsafe {
            WinUsb_SetPipePolicy(
                self.handle,
                ep,
                PIPE_TRANSFER_TIMEOUT,
                4,
                &v as *const u32 as *const core::ffi::c_void,
            )
        };
        if ok == 0 {
            return Err(last_err("WinUsb_SetPipePolicy(timeout)"));
        }
        Ok(())
    }

    /// Control OUT transfer (e.g. device init commands).
    pub fn control_out(
        &self,
        req_type: u8,
        req: u8,
        value: u16,
        index: u16,
        data: &[u8],
    ) -> io::Result<()> {
        let setup = WINUSB_SETUP_PACKET {
            RequestType: req_type,
            Request: req,
            Value: value,
            Index: index,
            Length: data.len() as u16,
        };
        let mut transferred = 0u32;
        let mut buf = data.to_vec();
        let ok = unsafe {
            WinUsb_ControlTransfer(
                self.handle,
                setup,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut transferred,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(last_err("WinUsb_ControlTransfer(out)"));
        }
        Ok(())
    }

    /// Control IN transfer; returns the bytes read.
    pub fn control_in(
        &self,
        req_type: u8,
        req: u8,
        value: u16,
        index: u16,
        len: u16,
    ) -> io::Result<Vec<u8>> {
        let setup = WINUSB_SETUP_PACKET {
            RequestType: req_type,
            Request: req,
            Value: value,
            Index: index,
            Length: len,
        };
        let mut buf = vec![0u8; len as usize];
        let mut transferred = 0u32;
        let ok = unsafe {
            WinUsb_ControlTransfer(
                self.handle,
                setup,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut transferred,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(last_err("WinUsb_ControlTransfer(in)"));
        }
        buf.truncate(transferred as usize);
        Ok(buf)
    }

    /// The fixed control-transfer init sequence (from winusb_device.init_device).
    pub fn init_device(&self) -> io::Result<()> {
        Err(super::runtime_access::unavailable())
}

    fn write_bulk(&self, data: &[u8]) -> io::Result<()> {
        let mut transferred = 0u32;
        let ok = unsafe {
            WinUsb_WritePipe(
                self.handle,
                EP_OUT,
                data.as_ptr(),
                data.len() as u32,
                &mut transferred,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            return Err(last_err("WinUsb_WritePipe"));
        }
        Ok(())
    }

    /// Read one bulk transfer (up to `buf.len()`), returning the count. A pipe
    /// timeout returns Ok(0) rather than an error.
    fn read_bulk(&self, buf: &mut [u8], timeout_ms: u32) -> io::Result<usize> {
        self.set_timeout(EP_IN, timeout_ms)?;
        let mut transferred = 0u32;
        let ok = unsafe {
            WinUsb_ReadPipe(
                self.handle,
                EP_IN,
                buf.as_mut_ptr(),
                buf.len() as u32,
                &mut transferred,
                std::ptr::null_mut(),
            )
        };
        if ok == 0 {
            let e = unsafe { GetLastError() };
            if e == ERROR_SEM_TIMEOUT {
                return Ok(0);
            }
            return Err(io::Error::new(
                io::ErrorKind::Other,
                format!("WinUsb_ReadPipe: error {e}"),
            ));
        }
        Ok(transferred as usize)
    }

    /// Send a TTP command, returning its msg_id.
    pub fn send(&mut self, packet_id: u32, payload: &[u8]) -> io::Result<u32> {
        self.msg_id += 1;
        let pkt = build_packet(self.msg_id, packet_id, payload);
        self.write_bulk(&pkt)?;
        Ok(self.msg_id)
    }

    /// Pull the next deframed TTP message (any), reading more USB if needed.
    /// Returns Ok(None) on a read timeout with nothing buffered.
    pub fn next_message(&mut self, timeout_ms: u32) -> io::Result<Option<Vec<u8>>> {
        let buffered = self.deframer.pop();
        if let Some(error) = self.deframer.error() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, error));
        }
        if let Some(m) = buffered {
            return Ok(Some(m));
        }
        let mut rbuf = vec![0u8; 16384];
        let n = self.read_bulk(&mut rbuf, timeout_ms)?;
        if n == 0 {
            return Ok(None);
        }
        self.deframer.push(&rbuf[..n]);
        let next = self.deframer.pop();
        if let Some(error) = self.deframer.error() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, error));
        }
        Ok(next)
    }

    /// Wait for a command response with the given msg_id (skipping other frames).
    pub fn recv(&mut self, expected_mid: u32, timeout_ms: u32) -> io::Result<Option<Vec<u8>>> {
        receive_matching_response(
            expected_mid,
            Duration::from_millis(timeout_ms as u64),
            |remaining_ms| self.next_message(remaining_ms),
            Instant::now,
            || false,
        )
    }

    /// Command wait used by the adapter handshake. Stop and total deadline are
    /// checked after EVERY packet, including a continuous unrelated stream.
    pub fn recv_with_stop(
        &mut self,
        expected_mid: u32,
        timeout_ms: u32,
        stop: &AtomicBool,
    ) -> io::Result<Option<Vec<u8>>> {
        receive_matching_response(
            expected_mid,
            Duration::from_millis(timeout_ms as u64),
            |remaining_ms| self.next_message(remaining_ms),
            Instant::now,
            || stop.load(Ordering::Relaxed),
        )
    }
}

fn receive_matching_response(
    expected_mid: u32,
    timeout: Duration,
    mut next: impl FnMut(u32) -> io::Result<Option<Vec<u8>>>,
    mut now: impl FnMut() -> Instant,
    mut stopped: impl FnMut() -> bool,
) -> io::Result<Option<Vec<u8>>> {
    let started = now();
    loop {
        if stopped() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "EyeChip wait cancelled",
            ));
        }
        let remaining = timeout.saturating_sub(now().saturating_duration_since(started));
        if remaining.is_zero() {
            return Ok(None);
        }
        let wait_ms = remaining.as_millis().clamp(1, 200) as u32;
        let message = next(wait_ms)?;
        if stopped() {
            return Err(io::Error::new(
                io::ErrorKind::Interrupted,
                "EyeChip wait cancelled",
            ));
        }
        // A late matching response cannot extend the original command deadline.
        if now().saturating_duration_since(started) >= timeout {
            return Ok(None);
        }
        if let Some(message) = message {
            if message.len() >= 8
                && u32::from_be_bytes(message[4..8].try_into().unwrap()) == expected_mid
            {
                return Ok(Some(message));
            }
        }
    }
}

impl Drop for UsbDevice {
    fn drop(&mut self) {
        unsafe {
            if !self.handle.is_null() {
                WinUsb_Free(self.handle);
            }
            if self.file != INVALID_HANDLE_VALUE && !self.file.is_null() {
                CloseHandle(self.file);
            }
        }
        record_interface_close();
    }
}

#[cfg(test)]
mod reopen_quiet_tests {
    use super::*;
    use std::cell::Cell;

    fn response(mid: u32) -> Vec<u8> {
        let mut message = vec![0; 8];
        message[4..8].copy_from_slice(&mid.to_be_bytes());
        message
    }

    #[test]
    fn unrelated_stream_flood_cannot_extend_command_deadline() {
        let start = Instant::now();
        let elapsed = Cell::new(0u64);
        let result = receive_matching_response(
            42,
            Duration::from_millis(1000),
            |wait_ms| {
                assert!((1..=200).contains(&wait_ms));
                elapsed.set(elapsed.get() + 1);
                Ok(Some(response(0)))
            },
            || start + Duration::from_millis(elapsed.get()),
            || false,
        )
        .unwrap();
        assert!(result.is_none());
        assert_eq!(elapsed.get(), 1000);
    }

    #[test]
    fn stop_is_checked_while_unrelated_messages_keep_arriving() {
        let start = Instant::now();
        let packets = Cell::new(0u64);
        let result = receive_matching_response(
            42,
            Duration::from_secs(5),
            |_| {
                packets.set(packets.get() + 1);
                Ok(Some(response(0)))
            },
            || start + Duration::from_millis(packets.get()),
            || packets.get() >= 7,
        );
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::Interrupted);
        assert_eq!(packets.get(), 7);
    }

    #[test]
    fn read_timeout_is_capped_to_remaining_total_budget() {
        let start = Instant::now();
        let elapsed = Cell::new(0u64);
        let mut waits = Vec::new();
        let result = receive_matching_response(
            42,
            Duration::from_millis(251),
            |wait_ms| {
                waits.push(wait_ms);
                elapsed.set(elapsed.get() + wait_ms as u64);
                Ok(None)
            },
            || start + Duration::from_millis(elapsed.get()),
            || false,
        )
        .unwrap();
        assert!(result.is_none());
        assert_eq!(waits, [200, 51]);
    }

    #[test]
    fn late_matching_response_is_not_accepted() {
        let start = Instant::now();
        let read = Cell::new(false);
        let result = receive_matching_response(
            42,
            Duration::from_millis(10),
            |_| {
                read.set(true);
                Ok(Some(response(42)))
            },
            || start + Duration::from_millis(if read.get() { 10 } else { 0 }),
            || false,
        )
        .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn zero_timeout_does_not_touch_usb_and_valid_response_returns() {
        let start = Instant::now();
        assert!(receive_matching_response(
            42,
            Duration::ZERO,
            |_| panic!("zero deadline must not issue a USB read"),
            || start,
            || false,
        )
        .unwrap()
        .is_none());
        assert_eq!(
            receive_matching_response(
                42,
                Duration::from_millis(1),
                |wait_ms| {
                    assert_eq!(wait_ms, 1);
                    Ok(Some(response(42)))
                },
                || start,
                || false,
            )
            .unwrap(),
            Some(response(42))
        );
    }

    #[test]
    fn cold_start_has_no_quiet_delay() {
        let now = Instant::now();
        assert_eq!(quiet_remaining(None, now), Duration::ZERO);
    }

    #[test]
    fn recent_close_requires_only_the_remaining_quiet_time() {
        let closed = Instant::now();
        let now = closed + Duration::from_millis(1250);
        assert_eq!(
            quiet_remaining(Some(closed), now),
            Duration::from_millis(3750)
        );
        assert_eq!(
            quiet_remaining(Some(closed), closed + REOPEN_QUIET_PERIOD),
            Duration::ZERO
        );
    }
}

/// Run the CM device-interface enumeration for the EyeChip GUID and return the
/// first interface's NUL-terminated UTF-16 path buffer. This is the shared core of
/// both [`find_device`] (which then opens the device) and [`peek_serial`] (which
/// only wants the serial) — enumeration only, no CreateFile/WinUsb_Initialize.
fn enumerate_eyechip() -> io::Result<Vec<u16>> {
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
            "EyeChip not found (is it connected and the Tobii platform service stopped?)",
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

/// Extract the serial from an EyeChip device-interface path: it is the 3rd
/// '#'-separated segment (e.g. `...#XR5DA-12345#...`).
fn serial_from_path(buf: &[u16]) -> String {
    // First NUL-terminated string in the multi-sz.
    let first_len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
    let path_str = String::from_utf16_lossy(&buf[..first_len]);
    path_str
        .split('#')
        .nth(2)
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Enumerate the EyeChip interface; return (NUL-terminated UTF-16 path, serial).
fn find_device() -> io::Result<(Vec<u16>, String)> {
    let buf = enumerate_eyechip()?;
    let serial = serial_from_path(&buf);
    Ok((buf, serial))
}

/// Peek the EyeChip serial WITHOUT opening the device — CM enumeration only (no
/// CreateFile / WinUsb_Initialize), so it's cheap and safe to call before deciding
/// which adapter to build. Returns `None` if no EyeChip is present (or the serial
/// segment is empty). Used by `make_adapter` to auto-route VR4 vs XR5 from the
/// serial prefix. Shares the exact CM enumeration `find_device` uses.
pub fn peek_serial() -> Option<String> {
    let buf = enumerate_eyechip().ok()?;
    let serial = serial_from_path(&buf);
    if serial.is_empty() {
        None
    } else {
        Some(serial)
    }
}
