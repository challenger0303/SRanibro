
use crate::core::types::{EyeSample, GazeSample};

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
            gaze: self.gaze,
            gaze_valid: self.gaze_valid != 0,
            gaze_reported: true,
            origin_mm: self.origin,
            origin_valid: self.origin_valid != 0,
            pupil_mm: self.pupil_diam,
            pupil_valid: self.pupil_valid != 0,
            pupil_pos: self.pupil_pos,
            pupil_pos_valid: self.pupil_pos_valid != 0,
            pupil_pos_reported: true,
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
        let mut b = [0u8; 160];
        b[32..36].copy_from_slice(&1i32.to_le_bytes());
        b[36..40].copy_from_slice(&1.5f32.to_le_bytes());
        b[72..76].copy_from_slice(&0.8f32.to_le_bytes());
        b[68..72].copy_from_slice(&1i32.to_le_bytes());

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
