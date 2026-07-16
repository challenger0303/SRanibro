//! Pimax VR4 device data structures — LEGACY (Option A, reference-only).
//!
//! NOT in the live path: VR4/Crystal acquisition is DLL-free over WinUSB + TTP
//! (see [`super::gaze`] / [`super::vr4_adapter`]). This 160-byte DLL-callback
//! struct is retained only as a layout reference (with round-trip tests) in case
//! the patched-DLL path is ever revisited.
//!
//! `WearableAdvancedData` is the 160-byte struct the patched Pimax DLL
//! (`tobii_stream_engine_pimax_unlock.dll`) hands back via
//! `tobii_wearable_advanced_data_subscribe`. The `repr(C)` layout here is
//! byte-compatible with the ctypes definition in `sranibro/adapters/pimax_vr4.py`
//! (16-byte header + 72 bytes/eye); the compile-time `assert!`s guard that.
//!
//! Gaze is left RAW here (the output sink applies the single canonical negation),
//! matching every other adapter.

use crate::core::types::{EyeSample, GazeSample};

/// Per-eye block (72 bytes). Field order matches the DLL exactly: each value is
/// preceded by its validity int.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EyeBlock {
    pub origin_valid: i32,
    pub origin: [f32; 3],
    pub gaze_valid: i32,
    pub gaze: [f32; 3],
    pub pupil_valid: i32,
    pub pupil_diam: f32,
    pub pupil_pos_valid: i32,
    pub pupil_pos: [f32; 2],
    pub openness_valid: i32,
    pub openness: f32,
    pub _extra: [i32; 3],
}
const _: () = assert!(core::mem::size_of::<EyeBlock>() == 72);

/// Full advanced-data record (160 bytes): 16-byte timestamp header + 2 eyes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WearableAdvancedData {
    pub timestamp_tracker_us: i64,
    pub timestamp_system_us: i64,
    pub left: EyeBlock,
    pub right: EyeBlock,
}
const _: () = assert!(core::mem::size_of::<WearableAdvancedData>() == 160);

impl EyeBlock {
    fn to_eye_sample(&self) -> EyeSample {
        EyeSample {
            gaze: self.gaze, // RAW; sink negates once.
            gaze_valid: self.gaze_valid != 0,
            gaze_reported: true,
            origin_mm: self.origin,
            origin_valid: self.origin_valid != 0,
            pupil_mm: self.pupil_diam,
            pupil_valid: self.pupil_valid != 0,
            pupil_pos: self.pupil_pos,
            pupil_pos_valid: self.pupil_pos_valid != 0,
            openness: self.openness,
            openness_valid: self.openness_valid != 0,
            openness_reported: true,
        }
    }
}

impl WearableAdvancedData {
    pub fn to_gaze_sample(&self) -> GazeSample {
        GazeSample {
            timestamp_us: self.timestamp_system_us as u64,
            left: self.left.to_eye_sample(),
            right: self.right.to_eye_sample(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_match_dll_layout() {
        assert_eq!(core::mem::size_of::<EyeBlock>(), 72);
        assert_eq!(core::mem::size_of::<WearableAdvancedData>(), 160);
    }

    #[test]
    fn raw_bytes_decode_at_correct_offsets() {
        // Mirror the ctypes field offsets: header 16, then left EyeBlock.
        //   left.gaze.x  = 16 + 4(origin_valid) + 12(origin) + 4(gaze_valid) = 36
        //   left.openness = 36 + 12(gaze) + 4 + 4 + 4 + 8 + 4 = 72
        //   left.gaze_valid = 32
        let mut b = [0u8; 160];
        b[32..36].copy_from_slice(&1i32.to_le_bytes());
        b[36..40].copy_from_slice(&1.5f32.to_le_bytes());
        b[72..76].copy_from_slice(&0.8f32.to_le_bytes());
        b[68..72].copy_from_slice(&1i32.to_le_bytes()); // left.openness_valid

        // SAFETY: 160 bytes, correct alignment (array is at least 4-aligned for
        // the i32/f32 fields; the leading i64s read from offset 0/8 which are
        // 8-aligned for a stack array). repr(C) layout asserted above.
        let d: WearableAdvancedData = unsafe { std::ptr::read_unaligned(b.as_ptr() as *const _) };
        assert_eq!(d.left.gaze[0], 1.5);
        assert_eq!(d.left.openness, 0.8);
        assert!(d.left.gaze_valid != 0);

        let gs = d.to_gaze_sample();
        assert_eq!(gs.left.gaze[0], 1.5);
        assert!(gs.left.gaze_valid);
        assert_eq!(gs.left.openness, 0.8);
        assert!(gs.left.openness_valid);
        assert!(!gs.right.gaze_valid, "right block left zeroed");
    }
}
