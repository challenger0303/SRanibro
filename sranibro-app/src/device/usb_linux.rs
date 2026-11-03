use std::io;
use std::sync::{Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use rusb::{Context, DeviceHandle, Direction, TransferType, UsbContext};
use super::ttp::{build_packet, Deframer};

const VID: u16 = 0x2104;
const PID: u16 = 0x0561;
const EP_IN: u8 = 0x83;
const EP_OUT: u8 = 0x05;
const REOPEN_QUIET_PERIOD: Duration = Duration::from_secs(5);
static LAST_INTERFACE_CLOSE: OnceLock<Mutex<Option<Instant>>> = OnceLock::new();

fn usb_error(error: rusb::Error) -> io::Error {
    let kind = match error {
        rusb::Error::Access => io::ErrorKind::PermissionDenied,
        rusb::Error::NoDevice => io::ErrorKind::NotFound,
        rusb::Error::Timeout => io::ErrorKind::TimedOut,
        rusb::Error::Busy => io::ErrorKind::ResourceBusy,
        _ => io::ErrorKind::Other,
    };
    let detail = match error {
        rusb::Error::Access => "XR5 USB permission denied; install the supplied udev rule and reconnect the headset",
        rusb::Error::Busy => "XR5 eye-camera interface is in use; close the other eye-camera application",
        _ => "XR5 USB transfer failed",
    };
    io::Error::new(kind, format!("{detail}: {error}"))
}

fn quiet_remaining(last: Option<Instant>, now: Instant) -> Duration {
    last.map(|closed| REOPEN_QUIET_PERIOD.saturating_sub(now.saturating_duration_since(closed)))
        .unwrap_or(Duration::ZERO)
}

pub fn reopen_quiet_remaining() -> Duration {
    let last = LAST_INTERFACE_CLOSE.get_or_init(|| Mutex::new(None)).lock().ok().and_then(|v| *v);
    quiet_remaining(last, Instant::now())
}

fn xr5_identity(vendor: u16, product: u16, serial: &str) -> bool {
    vendor == VID && product == PID && serial.trim().to_ascii_uppercase().starts_with("XR5")
}

fn bulk_pair(endpoints: impl Iterator<Item = (u8, Direction, TransferType)>) -> bool {
    let mut input = false;
    let mut output = false;
    for (address, direction, transfer) in endpoints {
        input |= address == EP_IN && direction == Direction::In && transfer == TransferType::Bulk;
        output |= address == EP_OUT && direction == Direction::Out && transfer == TransferType::Bulk;
    }
    input && output
}

pub struct UsbDevice {
    handle: DeviceHandle<Context>,
    interface: u8,
    pub serial: String,
    msg_id: u32,
    deframer: Deframer,
}

impl UsbDevice {
    pub fn open() -> io::Result<Self> {
        let context = Context::new().map_err(usb_error)?;
        let devices = context.devices().map_err(usb_error)?;
        let mut candidates = Vec::new();
        let mut denied = false;
        for device in devices.iter() {
            let descriptor = match device.device_descriptor() { Ok(value) => value, Err(_) => continue };
            if descriptor.vendor_id() != VID || descriptor.product_id() != PID { continue; }
            let handle = match device.open() {
                Ok(handle) => handle,
                Err(rusb::Error::Access) => { denied = true; continue; }
                Err(error) => return Err(usb_error(error)),
            };
            let serial = handle.read_serial_number_string_ascii(&descriptor).map_err(usb_error)?;
            if !xr5_identity(descriptor.vendor_id(), descriptor.product_id(), &serial) { continue; }
            let configuration = device.active_config_descriptor().map_err(usb_error)?;
            let mut interfaces = Vec::new();
            for interface in configuration.interfaces() {
                for alternate in interface.descriptors() {
                    if bulk_pair(alternate.endpoint_descriptors().map(|ep| (ep.address(), ep.direction(), ep.transfer_type()))) {
                        interfaces.push((alternate.interface_number(), alternate.setting_number()));
                    }
                }
            }
            if interfaces.len() != 1 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "XR5 needs one unambiguous bulk interface (IN 0x83, OUT 0x05)"));
            }
            candidates.push((handle, serial, interfaces[0]));
        }
        if candidates.len() > 1 || (denied && !candidates.is_empty()) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "Multiple possible XR5 EyeChips; connect only one headset"));
        }
        let Some((handle, serial, (interface, alternate))) = candidates.pop() else {
            return Err(if denied { usb_error(rusb::Error::Access) } else {
                io::Error::new(io::ErrorKind::NotFound, "No XR5 EyeChip found (USB 2104:0561, XR5 serial)")
            });
        };
        if handle.kernel_driver_active(interface).unwrap_or(false) {
            return Err(io::Error::new(io::ErrorKind::ResourceBusy, "XR5 bulk interface has a kernel driver; no driver was detached"));
        }
        handle.claim_interface(interface).map_err(usb_error)?;
        if alternate != 0 { handle.set_alternate_setting(interface, alternate).map_err(usb_error)?; }
        for endpoint in [EP_IN, EP_OUT] { handle.clear_halt(endpoint).map_err(usb_error)?; }
        Ok(Self { handle, interface, serial, msg_id: 0, deframer: Deframer::new() })
    }

    pub fn control_out(&self, req_type: u8, req: u8, value: u16, index: u16, data: &[u8]) -> io::Result<()> {
        let count = self.handle.write_control(req_type, req, value, index, data, Duration::from_secs(5)).map_err(usb_error)?;
        if count != data.len() { return Err(io::Error::new(io::ErrorKind::WriteZero, "Incomplete XR5 control write")); }
        Ok(())
    }

    pub fn control_in(&self, req_type: u8, req: u8, value: u16, index: u16, len: u16) -> io::Result<Vec<u8>> {
        let mut data = vec![0; len as usize];
        let count = self.handle.read_control(req_type, req, value, index, &mut data, Duration::from_secs(5)).map_err(usb_error)?;
        data.truncate(count); Ok(data)
    }

    pub fn init_device(&self) -> io::Result<()> {
        self.control_out(0x41, 0x30, 0, 0, &[1,0,0,0,0,0,0,0,1,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0])?;
        self.control_in(0xC1, 0x30, 0, 0, 512)?;
        self.control_in(0xC1, 0x46, 1, 0, 8)?;
        self.control_out(0x41, 0x41, 0, 0, &[])
    }

    fn write_bulk(&self, data: &[u8]) -> io::Result<()> {
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut offset = 0;
        while offset < data.len() {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining < Duration::from_millis(1) { return Err(usb_error(rusb::Error::Timeout)); }
            let count = self.handle.write_bulk(EP_OUT, &data[offset..], remaining).map_err(usb_error)?;
            if count == 0 { return Err(io::Error::new(io::ErrorKind::WriteZero, "Empty XR5 bulk write")); }
            offset += count;
        }
        Ok(())
    }

    fn read_bulk(&self, data: &mut [u8], timeout_ms: u32) -> io::Result<usize> {
        match self.handle.read_bulk(EP_IN, data, Duration::from_millis(timeout_ms.max(1) as u64)) {
            Ok(count) => Ok(count), Err(rusb::Error::Timeout) => Ok(0), Err(error) => Err(usb_error(error)),
        }
    }

    pub fn send(&mut self, packet_id: u32, payload: &[u8]) -> io::Result<u32> {
        self.msg_id = self.msg_id.checked_add(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "XR5 message sequence exhausted"))?;
        self.write_bulk(&build_packet(self.msg_id, packet_id, payload))?; Ok(self.msg_id)
    }

    pub fn next_message(&mut self, timeout_ms: u32) -> io::Result<Option<Vec<u8>>> {
        let buffered = self.deframer.pop();
        if let Some(error) = self.deframer.error() { return Err(io::Error::new(io::ErrorKind::InvalidData, error)); }
        if buffered.is_some() { return Ok(buffered); }
        let mut buffer = vec![0; 16384];
        let count = self.read_bulk(&mut buffer, timeout_ms)?;
        if count == 0 { return Ok(None); }
        self.deframer.push(&buffer[..count]);
        let next = self.deframer.pop();
        if let Some(error) = self.deframer.error() { return Err(io::Error::new(io::ErrorKind::InvalidData, error)); }
        Ok(next)
    }

    pub fn recv_with_stop(&mut self, expected: u32, timeout_ms: u32, stop: &AtomicBool) -> io::Result<Option<Vec<u8>>> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
        loop {
            if stop.load(Ordering::Relaxed) { return Err(io::Error::new(io::ErrorKind::Interrupted, "XR5 wait cancelled")); }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() { return Ok(None); }
            let message = self.next_message(remaining.as_millis().clamp(1, 200) as u32)?;
            if stop.load(Ordering::Relaxed) { return Err(io::Error::new(io::ErrorKind::Interrupted, "XR5 wait cancelled")); }
            if Instant::now() >= deadline { return Ok(None); }
            if let Some(message) = message {
                if message.len() >= 8 && u32::from_be_bytes(message[4..8].try_into().unwrap()) == expected { return Ok(Some(message)); }
            }
        }
    }
}

impl Drop for UsbDevice {
    fn drop(&mut self) {
        let _ = self.handle.release_interface(self.interface);
        if let Ok(mut last) = LAST_INTERFACE_CLOSE.get_or_init(|| Mutex::new(None)).lock() { *last = Some(Instant::now()); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_the_xr5_family_is_admitted() {
        assert!(xr5_identity(VID, PID, "XR5-fixture"));
        for serial in ["", "VR4-fixture", "VRS-fixture"] { assert!(!xr5_identity(VID, PID, serial)); }
        assert!(!xr5_identity(0, PID, "XR5-fixture"));
        assert!(!xr5_identity(VID, 0, "XR5-fixture"));
    }
    #[test]
    fn interface_requires_both_expected_bulk_endpoints() {
        let pair = [(EP_IN, Direction::In, TransferType::Bulk), (EP_OUT, Direction::Out, TransferType::Bulk)];
        assert!(bulk_pair(pair.into_iter()));
        assert!(!bulk_pair(pair[..1].iter().copied()));
        assert!(!bulk_pair([(EP_IN, Direction::In, TransferType::Interrupt), pair[1]].into_iter()));
        assert!(!bulk_pair([(0x81, Direction::In, TransferType::Bulk), pair[1]].into_iter()));
    }
    #[test]
    fn reconnect_quiet_period_is_bounded() {
        let start = Instant::now();
        assert_eq!(quiet_remaining(None, start), Duration::ZERO);
        assert_eq!(quiet_remaining(Some(start), start + Duration::from_secs(1)), Duration::from_secs(4));
        assert_eq!(quiet_remaining(Some(start), start + Duration::from_secs(8)), Duration::ZERO);
    }
}
