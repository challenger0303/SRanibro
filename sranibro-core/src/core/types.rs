
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Eye {
    Left = 0,
    Right = 1,
}

impl Eye {
    pub const ALL: [Eye; 2] = [Eye::Left, Eye::Right];
    #[inline]
    pub fn idx(self) -> usize {
        self as usize
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Eye::Left => "left",
            Eye::Right => "right",
        }
    }
    pub fn opposite(self) -> Eye {
        match self {
            Eye::Left => Eye::Right,
            Eye::Right => Eye::Left,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EyeSample {
    pub gaze: [f32; 3],
    pub gaze_valid: bool,
    pub origin_mm: [f32; 3],
    pub origin_valid: bool,
    pub pupil_mm: f32,
    pub pupil_valid: bool,
    pub pupil_pos: [f32; 2],
    pub pupil_pos_valid: bool,
    pub openness: f32,
    pub openness_valid: bool,
}

impl Default for EyeSample {
    fn default() -> Self {
        Self {
            gaze: [0.0, 0.0, 0.0],
            gaze_valid: false,
            origin_mm: [0.0, 0.0, 0.0],
            origin_valid: false,
            pupil_mm: 0.0,
            pupil_valid: false,
            pupil_pos: [0.5, 0.5],
            pupil_pos_valid: false,
            openness: 0.0,
            openness_valid: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GazeSample {
    pub timestamp_us: u64,
    pub left: EyeSample,
    pub right: EyeSample,
}

impl GazeSample {
    pub fn eye(&self, e: Eye) -> &EyeSample {
        match e {
            Eye::Left => &self.left,
            Eye::Right => &self.right,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MlGeometry {
    pub crop_left: f32,
    pub crop_right: f32,
    pub crop_top: f32,
    pub crop_bottom: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotate_deg: f32,
}

impl Default for MlGeometry {
    fn default() -> Self {
        Self {
            crop_left: 0.0,
            crop_right: 0.0,
            crop_top: 0.0,
            crop_bottom: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            rotate_deg: 0.0,
        }
    }
}

impl MlGeometry {
    pub fn is_identity(&self) -> bool {
        self.crop_left == 0.0
            && self.crop_right == 0.0
            && self.crop_top == 0.0
            && self.crop_bottom == 0.0
            && self.scale_x == 1.0
            && self.scale_y == 1.0
            && self.rotate_deg == 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DespeckleParams {
    pub enabled: bool,
    pub threshold: f32,
    pub radius: u32,
}

impl Default for DespeckleParams {
    fn default() -> Self {
        Self { enabled: true, threshold: 0.15, radius: 3 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FlattenParams {
    pub enabled: bool,
    pub strength: f32,
    pub radius: f32,
}

impl Default for FlattenParams {
    fn default() -> Self {
        Self { enabled: false, strength: 0.7, radius: 0.33 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BrightnessNorm {
    pub enabled: bool,
    pub adapt: f32,
    pub strength: f32,
    pub auto_learn: bool,
    pub tgt_level: [f32; 2],
    pub tgt_spread: [f32; 2],
    pub captured: bool,
}

impl Default for BrightnessNorm {
    fn default() -> Self {
        Self {
            enabled: false,
            adapt: 0.02,
            strength: 1.0,
            auto_learn: true,
            tgt_level: [128.0, 128.0],
            tgt_spread: [40.0, 40.0],
            captured: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DeviceProfile {
    pub name: String,
    pub ml_device: String,
    pub image_w: u32,
    pub image_h: u32,
    pub slot_a_eye: Eye,
    pub transport: String,
    pub streams: String,
    pub gaze_src: String,
}

impl Default for DeviceProfile {
    fn default() -> Self {
        Self {
            name: "unknown".into(),
            ml_device: "vr4".into(),
            image_w: 200,
            image_h: 200,
            slot_a_eye: Eye::Left,
            transport: "—".into(),
            streams: "—".into(),
            gaze_src: "gaze · pupil · openness".into(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EyeResult {
    pub eye: Eye,
    pub gaze: [f32; 3],
    pub gaze_valid: bool,
    pub openness: f32,
    pub openness_valid: bool,
    pub wide: f32,
    pub squeeze: f32,
    pub frown: f32,
    pub pupil_mm: f32,
    pub pupil_valid: bool,
    pub blink: bool,
    pub gaze_yoked: bool,
    pub brow: f32,
    pub brow_valid: bool,
}

impl EyeResult {
    pub fn new(eye: Eye) -> Self {
        Self {
            eye,
            gaze: [0.0, 0.0, 0.0],
            gaze_valid: false,
            openness: 1.0,
            openness_valid: false,
            wide: 0.0,
            squeeze: 0.0,
            frown: 0.0,
            pupil_mm: 0.0,
            pupil_valid: false,
            blink: false,
            gaze_yoked: false,
            brow: 0.0,
            brow_valid: false,
        }
    }
}
