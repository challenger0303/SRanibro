
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
    pub gaze_reported: bool,
    pub origin_mm: [f32; 3],
    pub origin_valid: bool,
    pub pupil_mm: f32,
    pub pupil_valid: bool,
    pub pupil_pos: [f32; 2],
    pub pupil_pos_valid: bool,
    pub pupil_pos_reported: bool,
    pub openness: f32,
    pub openness_valid: bool,
    pub openness_reported: bool,
}

impl Default for EyeSample {
    fn default() -> Self {
        Self {
            gaze: [0.0, 0.0, 0.0],
            gaze_valid: false,
            gaze_reported: false,
            origin_mm: [0.0, 0.0, 0.0],
            origin_valid: false,
            pupil_mm: 0.0,
            pupil_valid: false,
            pupil_pos: [0.5, 0.5],
            pupil_pos_valid: false,
            pupil_pos_reported: false,
            openness: 0.0,
            openness_valid: false,
            openness_reported: false,
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
    pub mirror_h: Option<bool>,
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
            mirror_h: None,
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

    pub fn mirrored_x(mut self) -> Self {
        std::mem::swap(&mut self.crop_left, &mut self.crop_right);
        self.rotate_deg = -self.rotate_deg;
        self
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
        Self {
            enabled: true,
            threshold: 0.15,
            radius: 3,
        }
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
        Self {
            enabled: false,
            strength: 0.7,
            radius: 0.33,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BrightnessNorm {
    pub enabled: bool,
    pub manual_gain: f32,
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
            manual_gain: 1.6,
            adapt: 0.02,
            strength: 1.0,
            auto_learn: true,
            tgt_level: [128.0, 128.0],
            tgt_spread: [40.0, 40.0],
            captured: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GazeEyelidEyeProfile {
    pub enabled: bool,
    pub center_deg: [f32; 2],
    pub coefficients: [f32; 5],
    pub max_lift: f32,
    pub train_error: f32,
    pub holdout_error_before: f32,
    pub holdout_error_after: f32,
}

impl Default for GazeEyelidEyeProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            center_deg: [0.0; 2],
            coefficients: [0.0; 5],
            max_lift: 0.0,
            train_error: 0.0,
            holdout_error_before: 0.0,
            holdout_error_after: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct GazeEyelidProfile {
    pub schema_version: u32,
    pub calibrated_unix: u64,
    pub angle_scale_deg: f32,
    pub eyes: [GazeEyelidEyeProfile; 2],
}

impl Default for GazeEyelidProfile {
    fn default() -> Self {
        Self {
            schema_version: 2,
            calibrated_unix: 0,
            angle_scale_deg: 20.0,
            eyes: [GazeEyelidEyeProfile::default(); 2],
        }
    }
}

impl GazeEyelidProfile {
    pub fn is_compatible(&self) -> bool {
        self.schema_version == 2
            && self.angle_scale_deg.is_finite()
            && (5.0..=60.0).contains(&self.angle_scale_deg)
            && self.eyes.iter().all(|eye| {
                eye.center_deg.iter().all(|value| value.is_finite())
                    && eye.coefficients.iter().all(|value| value.is_finite())
                    && eye.max_lift.is_finite()
                    && (0.0..=0.12).contains(&eye.max_lift)
            })
    }

    pub fn gaze_angles_deg(gaze: [f32; 3]) -> Option<[f32; 2]> {
        if !gaze.iter().all(|value| value.is_finite()) {
            return None;
        }
        let horizontal_radius = gaze[0].hypot(gaze[2]);
        if horizontal_radius <= 1.0e-6 || gaze[2] <= 0.0 {
            return None;
        }
        Some([
            gaze[0].atan2(gaze[2]).to_degrees(),
            gaze[1].atan2(horizontal_radius).to_degrees(),
        ])
    }

    pub fn predicted_lift(&self, eye: usize, gaze: [f32; 3]) -> Option<f32> {
        if !self.is_compatible() {
            return None;
        }
        let profile = self.eyes.get(eye)?;
        if !profile.enabled {
            return None;
        }
        let angle = Self::gaze_angles_deg(gaze)?;
        let scale = self.angle_scale_deg;
        let x = (angle[0] - profile.center_deg[0]) / scale;
        let y = (angle[1] - profile.center_deg[1]) / scale;
        let basis = [x, y, x * y, x * x, y * y];
        let lift = profile
            .coefficients
            .iter()
            .zip(basis)
            .map(|(coefficient, value)| coefficient * value)
            .sum::<f32>();
        if !lift.is_finite() {
            return None;
        }
        Some(lift.clamp(0.0, profile.max_lift.clamp(0.0, 0.12)))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WinkEyeProfile {
    pub enabled: bool,
    pub floor_enabled: bool,
    pub wink_depth: f32,
    pub squeeze_enabled: bool,
    pub squeeze_enter_delta: f32,
    pub squeeze_release_delta: f32,
    pub holdout_before: f32,
    pub holdout_after: f32,
}

impl Default for WinkEyeProfile {
    fn default() -> Self {
        Self {
            enabled: false,
            floor_enabled: false,
            wink_depth: 0.20,
            squeeze_enabled: false,
            squeeze_enter_delta: 0.14,
            squeeze_release_delta: 0.07,
            holdout_before: 0.0,
            holdout_after: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct WinkProfile {
    pub schema_version: u32,
    pub calibrated_unix: u64,
    pub eyes: [WinkEyeProfile; 2],
}

impl Default for WinkProfile {
    fn default() -> Self {
        Self {
            schema_version: Self::SCHEMA_VERSION,
            calibrated_unix: 0,
            eyes: [WinkEyeProfile::default(); 2],
        }
    }
}

impl WinkProfile {
    pub const SCHEMA_VERSION: u32 = 2;

    pub fn is_compatible(&self) -> bool {
        self.schema_version == Self::SCHEMA_VERSION
            && self.eyes.iter().all(|eye| {
                eye.wink_depth.is_finite()
                    && (0.05..=0.40).contains(&eye.wink_depth)
                    && (!eye.squeeze_enabled
                        || (eye.squeeze_enter_delta.is_finite()
                            && eye.squeeze_release_delta.is_finite()
                            && (0.03..=0.60).contains(&eye.squeeze_enter_delta)
                            && (0.0..eye.squeeze_enter_delta).contains(&eye.squeeze_release_delta)))
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BlinkTimingProfile {
    pub schema_version: u32,
    pub enabled: bool,
    pub min_closed_ms: f32,
    pub calibrated_unix: u64,
    pub train_blinks: u16,
    pub holdout_blinks: u16,
}

impl Default for BlinkTimingProfile {
    fn default() -> Self {
        Self {
            schema_version: 1,
            enabled: true,
            min_closed_ms: 42.0,
            calibrated_unix: 0,
            train_blinks: 0,
            holdout_blinks: 0,
        }
    }
}

impl BlinkTimingProfile {
    pub fn is_compatible(&self) -> bool {
        self.schema_version == 1
            && self.min_closed_ms.is_finite()
            && (25.0..=100.0).contains(&self.min_closed_ms)
    }

    pub fn minimum_frames(&self) -> u8 {
        ((self.min_closed_ms * 0.120).round() as u8).clamp(3, 12)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PhotometricGainField {
    pub horizontal: f32,
    pub vertical: f32,
    pub horizontal_curve: f32,
    pub vertical_curve: f32,
}

impl PhotometricGainField {
    pub fn is_identity(self) -> bool {
        self.horizontal == 0.0
            && self.vertical == 0.0
            && self.horizontal_curve == 0.0
            && self.vertical_curve == 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PhotometricCorrection {
    pub enabled: bool,
    pub affine: [[f32; 2]; 2],
    pub flatten: FlattenParams,
    pub field: [PhotometricGainField; 2],
}

impl Default for PhotometricCorrection {
    fn default() -> Self {
        Self {
            enabled: false,
            affine: [[1.0, 0.0]; 2],
            flatten: FlattenParams::default(),
            field: [PhotometricGainField::default(); 2],
        }
    }
}

impl PhotometricCorrection {
    pub fn is_identity(self) -> bool {
        !self.enabled
            || (self.affine == [[1.0, 0.0]; 2]
                && !self.flatten.enabled
                && self.field.iter().all(|field| field.is_identity()))
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
