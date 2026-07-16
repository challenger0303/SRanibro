//! The data contract that flows adapter -> core -> output.
//!
//! Ported from the Python `sranibro/core/types.py`. Camera frames travel
//! out-of-band (raw bytes via the frame callback); only gaze/expression scalars
//! are modelled here. Gaze is the *true* (raw-sign) direction — output sinks
//! apply their own sign conventions (e.g. BrokenEye negates X internally).

/// Which eye. Doubles as an index (Left=0, Right=1) into per-eye state arrays.
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

/// Per-eye gaze/pupil as reported by the HMD's native tracker. `openness` here
/// is the device's native value — the core ignores it and derives openness from
/// the ML signal instead.
#[derive(Clone, Copy, Debug)]
pub struct EyeSample {
    pub gaze: [f32; 3],
    pub gaze_valid: bool,
    /// True when this sample actually carried a gaze validity field. This is
    /// separate from `gaze_valid`: an auxiliary packet must leave the current
    /// gaze untouched, while a reported-invalid gaze must clear stale validity.
    pub gaze_reported: bool,
    pub origin_mm: [f32; 3],
    pub origin_valid: bool,
    pub pupil_mm: f32,
    pub pupil_valid: bool,
    pub pupil_pos: [f32; 2],
    pub pupil_pos_valid: bool,
    pub openness: f32,
    pub openness_valid: bool,
    /// True when this sample actually carried a native openness validity field.
    /// False means "not provided", while `reported && !valid` is Tobii Disable.
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
            openness: 0.0,
            openness_valid: false,
            openness_reported: false,
        }
    }
}

/// A full gaze frame: both eyes + timestamp. Adapters emit one per device tick.
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

/// Geometry of the image fed to the eye ML, tunable PER DEVICE to cut per-person
/// variance (esp. Varjo, whose eye cameras sit at odd angles). Applied to each eye
/// frame BEFORE the resize to the model's 100x100: crop a sub-window, rotate, and
/// stretch. Default = identity (no crop, unit scale, 0°) → the preprocessing takes
/// the legacy area-resize path unchanged, so VR4/StarVR stay byte-identical (no ML
/// regression). Persisted per-device in `[hmd.geometry.*]`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct MlGeometry {
    /// Fraction of the frame cropped off each edge before resizing (0..0.9).
    pub crop_left: f32,
    pub crop_right: f32,
    pub crop_top: f32,
    pub crop_bottom: f32,
    /// Horizontal / vertical stretch of the sampled window (1.0 = none; >1 zooms in
    /// on that axis so content appears stretched along it).
    pub scale_x: f32,
    pub scale_y: f32,
    /// Rotation applied to the sampled window, in degrees.
    pub rotate_deg: f32,
    /// Horizontal mirror for this eye's ML channel. `None` means the value came from an
    /// older config and should inherit the device preset / legacy EyeMapping value;
    /// `Some(false)` is an explicit user override. Keeping this tri-state beside the
    /// geometry makes the asymmetric XR5 reconstruction atomic and migration-safe.
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
    /// True when this geometry is a no-op, so the preprocessor can take the exact
    /// legacy resize path (byte-identical, no ML regression) instead of the warp.
    pub fn is_identity(&self) -> bool {
        self.crop_left == 0.0
            && self.crop_right == 0.0
            && self.crop_top == 0.0
            && self.crop_bottom == 0.0
            && self.scale_x == 1.0
            && self.scale_y == 1.0
            && self.rotate_deg == 0.0
    }

    /// Horizontal counterpart used when a paired HMD's right camera is the mirror image
    /// of its left camera. Photometric mirror state is preserved independently.
    pub fn mirrored_x(mut self) -> Self {
        std::mem::swap(&mut self.crop_left, &mut self.crop_right);
        self.rotate_deg = -self.rotate_deg;
        self
    }
}

/// Specular-dot suppression for the eye ML input. Bright IR / glasses reflections create
/// small hot spots that the openness model latches onto — the occlusion + glint heatmaps
/// showed it reads "brighter = more open" almost everywhere, so a reflection inflates and
/// destabilizes the reading. This removes isolated bright OUTLIERS (pixels more than
/// `threshold` above their local mean within `radius`) by replacing them with the local
/// median, applied to the ML INPUT only — the displayed eye camera stays raw and gaze
/// (native Tobii) is untouched. Per-device; ON by default (near-identity when no spots).
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct DespeckleParams {
    pub enabled: bool,
    /// How much brighter than the local mean (0..1 of full scale) a pixel must be to count
    /// as a specular spot. Lower removes more (risks real highlights); higher = only the
    /// brightest glints.
    pub threshold: f32,
    /// Local window radius in pixels (window = 2r+1). Should exceed the glint size.
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

/// Illumination "flatten" (flat-field) for the eye ML input. When the eye is CLOSE to the
/// lens a low-frequency shadow (e.g. a dark vertical band down the centre) appears that the
/// model reacts to. This estimates the smooth illumination with a large-kernel local mean
/// and subtracts its deviation from the global mean, so slowly-varying shadows/gradients are
/// removed while the eye's high-frequency structure (lid, lashes) is preserved. A LARGE
/// radius keeps a blink (a mid-frequency lid motion) mostly intact. Per-device; OFF by
/// default (experimental; enable when the close-up shadow is the problem). Runs AFTER
/// despeckle, BEFORE brightness normalization.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct FlattenParams {
    pub enabled: bool,
    /// How much of the estimated illumination deviation to subtract, 0..1 (1 = full flatten).
    pub strength: f32,
    /// Local-mean window radius as a FRACTION of the frame's short side (bigger = smoother
    /// illumination estimate, less blink impact). ~0.33 is a good start.
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

/// Adaptive brightness + contrast normalization of the eye ML input. Lens-to-eye
/// distance varies per person (and per HMD reseat), so the IR illumination — hence the
/// image brightness/contrast — drifts; and the openness model reads "brighter = more
/// open" (per the response heatmap), so that drift biases openness. This holds the input
/// at a learned target: a SLOWLY-adapted per-user baseline (the source) is mapped by an
/// affine onto the target, so lens-distance drift is corrected while fast changes (blinks)
/// pass through unchanged. The target is auto-captured from the user's own settled frames
/// (`captured`) and persisted per-device, so it re-anchors across sessions. Applied to the
/// ML input only, AFTER despeckle; complements (does not replace) the camera's own AE.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct BrightnessNorm {
    pub enabled: bool,
    /// Baseline EMA rate per ML frame (~60Hz). Small = slow (won't chase blinks). ~0.02
    /// ≈ a couple-second time constant.
    pub adapt: f32,
    /// Blend of the normalization, 0..1 (1 = full).
    pub strength: f32,
    /// Auto-capture the target from the settled baseline after warmup (vs. keep a manually
    /// captured one).
    pub auto_learn: bool,
    /// Learned target level (median brightness) per eye `[L, R]`.
    pub tgt_level: [f32; 2],
    /// Learned target spread (robust contrast) per eye `[L, R]`.
    pub tgt_spread: [f32; 2],
    /// Whether the target has been captured yet (identity until it has).
    pub captured: bool,
}

impl Default for BrightnessNorm {
    fn default() -> Self {
        Self {
            // OFF by default: still being tuned, and normalizing to a stale target after a
            // big lens-distance change can misbehave — opt-in per device.
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

/// Per-HMD knobs the core needs to interpret an adapter's stream.
#[derive(Clone, Debug)]
pub struct DeviceProfile {
    pub name: String,
    /// EyePredictor preprocessing: "vr4" = frontal (resize only),
    /// "xr5" = angled (crop 40% + flip).
    pub ml_device: String,
    pub image_w: u32,
    pub image_h: u32,
    /// Which eye the first frame of a stereo pair belongs to (StarVR/VR4: slot A = left).
    pub slot_a_eye: Eye,
    /// Human-readable acquisition transport for the dashboard's DEVICE node (e.g.
    /// "WinUSB · DLL-free", "Tobii stream engine (DLL)", "VarjoLib SDK"). Set per-adapter
    /// so the UI is NOT hardcoded to one HMD.
    pub transport: String,
    /// Short description of the device's data streams for the DEVICE node (e.g. the Tobii
    /// TTP stream ids, or "eye-camera (Y8) + gaze").
    pub streams: String,
    /// What the device's gaze sample carries (GAZE node): e.g. "gaze · pupil · openness"
    /// for Tobii, "gaze direction" for Varjo (openness comes from the ML there).
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

/// Post-processed per-eye output handed to every OutputSink.
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
    /// This eye's gaze was MIRRORED from the other (open, tracking) eye because this
    /// one is squinting/closed and its own gaze went stale (cross-eye gaze yoke, see
    /// `Tuning.gaze_yoke`). Lets the output sinks emit the gaze even though `blink`
    /// is set — they otherwise suppress gaze while blinking — so a squinting eye
    /// FOLLOWS the tracked eye instead of freezing.
    pub gaze_yoked: bool,
    /// Eyebrow expression for this eye, signed [-1,+1] (+ = raised, - = lowered),
    /// inferred by the optional brow CNN from eye-shape/eyelid deformation. 0 + invalid
    /// when no brow model is loaded.
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
