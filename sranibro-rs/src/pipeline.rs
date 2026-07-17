//! Orchestration: adapter -> ML -> SRanipalState -> output sinks, with live
//! telemetry + controls for the UI.
//!
//! Thread layout mirrors the Python `core/pipeline.py`:
//!   adapter --frames--> latest[L/R] --(60Hz)--> EyeNet --> openness raw[L/R]
//!   adapter --gaze----> GazeSample
//!                            |
//!                       (120Hz emit)
//!         SRanipalState::process_frame -> [EyeResult;2] -> Telemetry + sinks

use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::config::{GazeCorrection, WideSource};
use crate::core::brow_state::BrowState;
use crate::core::eye_state::{CalibStore, SRanipalState, Tuning};
use crate::core::types::{
    BrightnessNorm, DespeckleParams, DeviceProfile, Eye, EyeResult, EyeSample, FlattenParams,
    GazeSample, MlGeometry,
};
use crate::core::wide_state::WideState;
use crate::device::HmdAdapter;
use crate::ml::brow_net::BrowNet;
use crate::ml::eye_net::EyeNet;
use crate::ml::eyelid_model::{
    CanonicalStereoInput, EyelidModel, LegacyEyelidModel, EYELID_INPUT_LEN,
};
use crate::ml::preprocess;
use crate::ml::wide_net::WideNet;
use crate::output::OutputSink;

/// Live snapshot of the running pipeline, shared with the UI (read-only there).
pub struct Telemetry {
    /// Newest eye frame per eye as `(width, height, bytes)` — dimensions travel with
    /// the frame so resolution is per-HMD (VR4/StarVR 200x200, Varjo higher).
    pub frames: Mutex<[Option<(u32, u32, Vec<u8>)>; 2]>,
    /// Nominal eye-camera resolution from the device profile — the UI's fallback
    /// label before any frame arrives (the live frame's own dims win once streaming).
    pub eye_w: u32,
    pub eye_h: u32,
    pub gaze: Mutex<GazeSample>,
    /// Wall-clock time of the newest VALID gaze report for each eye. Tobii 1289
    /// interleaves short invalid-status packets with valid samples; validity is
    /// therefore freshness-based instead of being cleared by one transient packet.
    gaze_last_valid: Mutex<[Option<Instant>; 2]>,
    pub ml_raw: Mutex<[f32; 2]>,
    /// Legacy per-eye post-processor input layout reconstructed from the stereo
    /// EyeNet output. Only presence, openness, and squeeze are populated in Phase 1;
    /// structural channels 2 and 4 remain zero exactly as in the previous path.
    pub ml5: Mutex<[[f32; 5]; 2]>,
    /// Native pupil diameter (mm, valid) per eye [L, R], from stream 1285 when available.
    pub pupil: Mutex<[(f32, bool); 2]>,
    /// RAW brow CNN output per eye [L, R] (one per inference, NOT smoothed/clamped). The
    /// emit thread does the blink-gated EMA + baseline. `c_brow` is the inference
    /// generation (so the emit thread advances the EMA once per inference, not per tick).
    pub brow_raw: Mutex<[f32; 2]>,
    pub c_brow: AtomicU64,
    /// Whether an eyebrow model is currently loaded. An `AtomicBool` (not a fixed flag) so
    /// [`Pipeline::set_brow`] can hot-load a freshly trained model into the LIVE pipeline
    /// without a device reconnect — the emit thread and the UI both observe the flip.
    pub brow_loaded: AtomicBool,
    /// Custom XR5 image-based EyeWide: raw model score, post-processed comparison,
    /// and the legacy SRanipal-derived value from the same emit frame.
    pub wide_raw: Mutex<[f32; 2]>,
    pub wide_custom: Mutex<[f32; 2]>,
    pub wide_sranipal: Mutex<[f32; 2]>,
    pub c_wide: AtomicU64,
    pub wide_loaded: AtomicBool,
    /// True only while the custom result is fresh, calibrated, and selected for output.
    pub wide_custom_active: AtomicBool,
    /// Custom model neutral-bootstrap visibility for the XR5 card.
    pub wide_ready: Mutex<[bool; 2]>,
    pub wide_bootstrap_seen: Mutex<[u32; 2]>,
    pub results: Mutex<[EyeResult; 2]>,
    pub baselines: Mutex<[f32; 2]>,
    /// Full post-processor calibration snapshot.  The guided onboarding flow
    /// reads this to preserve learned curve anchors when replacing baseline and
    /// blink depth.
    pub calibration: Mutex<Option<CalibStore>>,
    /// The EXACT per-eye image last fed to the eye net (100x100 u8, all filters +
    /// geometry applied), un-mirrored back to natural orientation — the dashboard's
    /// "NET" view shows what the model actually sees.
    pub ml_input: Mutex<[Option<Vec<u8>>; 2]>,
    pub ml_loaded: bool,
    pub device_name: String,
    /// Device-supplied descriptors for the dashboard's pipeline-node detail cards, so the
    /// UI reflects the ACTIVE adapter (Varjo/StarVR/VR4) instead of hardcoding Pimax.
    pub transport: String,
    pub streams: String,
    pub gaze_src: String,
    // Monotonic counters; the UI derives rates from deltas over time.
    pub c_frame_l: AtomicU64,
    pub c_frame_r: AtomicU64,
    pub c_gaze: AtomicU64,
    pub c_ml: AtomicU64,
    pub c_emit: AtomicU64,
    /// Emit cycles that overran the 120Hz period (a real dropped-frame count).
    pub c_drop: AtomicU64,
}

impl Telemetry {
    fn new(
        ml_loaded: bool,
        brow_loaded: bool,
        wide_loaded: bool,
        profile: &DeviceProfile,
    ) -> Arc<Self> {
        Arc::new(Telemetry {
            device_name: profile.name.clone(),
            transport: profile.transport.clone(),
            streams: profile.streams.clone(),
            gaze_src: profile.gaze_src.clone(),
            eye_w: profile.image_w,
            eye_h: profile.image_h,
            brow_raw: Mutex::new([0.0, 0.0]),
            c_brow: AtomicU64::new(0),
            brow_loaded: AtomicBool::new(brow_loaded),
            wide_raw: Mutex::new([0.0, 0.0]),
            wide_custom: Mutex::new([0.0, 0.0]),
            wide_sranipal: Mutex::new([0.0, 0.0]),
            c_wide: AtomicU64::new(0),
            wide_loaded: AtomicBool::new(wide_loaded),
            wide_custom_active: AtomicBool::new(false),
            wide_ready: Mutex::new([false; 2]),
            wide_bootstrap_seen: Mutex::new([0; 2]),
            frames: Mutex::new([None, None]),
            gaze: Mutex::new(GazeSample::default()),
            gaze_last_valid: Mutex::new([None, None]),
            ml_raw: Mutex::new([0.5, 0.5]),
            // Neutral seed: ch0=1.0 (eye present, so the gate doesn't force-close
            // before the first inference / when ML is absent), ch1=0.5 (neutral
            // openness), matching the `ml_raw` [0.5,0.5] no-ML path.
            ml5: Mutex::new([[1.0, 0.5, 0.0, 0.0, 0.0]; 2]),
            pupil: Mutex::new([(0.0, false), (0.0, false)]),
            results: Mutex::new([EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)]),
            baselines: Mutex::new([0.6, 0.6]),
            calibration: Mutex::new(None),
            ml_input: Mutex::new([None, None]),
            ml_loaded,
            c_frame_l: AtomicU64::new(0),
            c_frame_r: AtomicU64::new(0),
            c_gaze: AtomicU64::new(0),
            c_ml: AtomicU64::new(0),
            c_emit: AtomicU64::new(0),
            c_drop: AtomicU64::new(0),
        })
    }

    /// Latest gaze with the same freshness rule used by the 120 Hz emit thread.
    /// The UI uses this for XR5 centre capture so a stale-but-sticky coordinate cannot
    /// be mistaken for a full second of valid calibration samples.
    pub fn fresh_gaze(&self) -> GazeSample {
        let mut gaze = *self.gaze.lock().unwrap();
        if let Ok(last) = self.gaze_last_valid.lock() {
            let now = Instant::now();
            gaze.left.gaze_valid &= gaze_is_fresh(last[0], now);
            gaze.right.gaze_valid &= gaze_is_fresh(last[1], now);
        }
        gaze
    }
}

/// Per-unit eye-image mapping (handles hardware variants).
#[derive(Clone, Copy, Default)]
pub struct DeviceMap {
    /// Swap which physical camera maps to left/right.
    pub swap_eyes: bool,
    /// Horizontally mirror each eye image.
    pub flip_image: bool,
    /// Negate the gaze X (left/right) sign at the output.
    pub flip_gaze_x: bool,
}

/// Complete live state installed before the adapter and worker threads start. Keeping
/// this separate from [`DeviceMap`] avoids a one-frame identity/default window at startup
/// and keeps the product UI and diagnostic `run` entrypoint on the same initialization.
#[derive(Clone, Copy)]
pub struct PipelineInit {
    pub eyebrow_enabled: bool,
    pub ml_mirror: [bool; 2],
    pub tuning: Tuning,
    pub geometry: [MlGeometry; 2],
    pub despeckle: DespeckleParams,
    pub flatten: FlattenParams,
    pub brightness: BrightnessNorm,
    pub gaze_correction: GazeCorrection,
    pub wide_enabled: [bool; 2],
    pub wide_source: WideSource,
}

impl Default for PipelineInit {
    fn default() -> Self {
        Self {
            eyebrow_enabled: true,
            ml_mirror: [false; 2],
            tuning: Tuning::default(),
            geometry: [MlGeometry::default(); 2],
            despeckle: DespeckleParams::default(),
            flatten: FlattenParams::default(),
            brightness: BrightnessNorm::default(),
            gaze_correction: GazeCorrection::default(),
            wide_enabled: [true; 2],
            wide_source: WideSource::Sranipal,
        }
    }
}

/// On-demand occlusion-heatmap request/result, shared with the ML thread (which owns the
/// net). The UI sets `req` + `mode`; the ML thread computes both eyes on its next tick — a
/// ~2s blocking diagnostic — and publishes `result`, toggling `computing` around it.
pub struct HeatState {
    pub req: AtomicBool,
    pub mode: std::sync::atomic::AtomicU8, // 0 = occlusion (erase-to-mean), 1 = glint inject
    pub computing: AtomicBool,
    pub result: Mutex<Option<crate::ml::heatmap::HeatResult>>,
}

impl HeatState {
    fn new() -> Self {
        Self {
            req: AtomicBool::new(false),
            mode: std::sync::atomic::AtomicU8::new(0),
            computing: AtomicBool::new(false),
            result: Mutex::new(None),
        }
    }
}

pub struct Pipeline {
    adapter: Box<dyn HmdAdapter>,
    /// Canonical key for the adapter that is actually running. Unlike `[hmd].device`,
    /// this is resolved after `auto` sniffing and therefore selects the correct per-HMD
    /// geometry/mapping bucket in the live UI.
    pub device_key: String,
    stop: Arc<AtomicBool>,
    threads: Vec<JoinHandle<()>>,
    pub tele: Arc<Telemetry>,
    /// When true, the emit thread keeps telemetry live but stops feeding sinks.
    pub paused: Arc<AtomicBool>,
    /// Set to request a one-shot calibration recenter (cleared by the engine).
    pub recenter: Arc<AtomicBool>,
    /// One-shot calibrated state installed by the guided Dream Air flow.  The
    /// emit thread owns `SRanipalState`, so UI code hands the value across here
    /// rather than mutating processor internals concurrently.
    pub guided_calibration: Arc<Mutex<Option<CalibStore>>>,
    /// Diagnostic CSV recorder toggle (dashboard REC button): while true, the
    /// emit thread writes one row per frame — raw ml values next to every
    /// post-processing internal — to `sranibro_diag_<unix>.csv` in the app dir.
    pub diag_rec: Arc<AtomicBool>,
    /// Live-toggleable eye-image mapping (UI exposes these as checkboxes).
    pub swap_eyes: Arc<AtomicBool>,
    pub flip_image: Arc<AtomicBool>,
    /// Per-eye horizontal mirror applied ONLY to the ML input (A/B experiment for
    /// matching the model's expected eye handedness; see preprocess::vr4_to_input_stereo_flip).
    pub ml_mirror_l: Arc<AtomicBool>,
    pub ml_mirror_r: Arc<AtomicBool>,
    /// Negate gaze X (left/right) at the output — per-device gaze handedness fix.
    pub flip_gaze_x: Arc<AtomicBool>,
    /// Live Dream Air/XR5 post-calibration gaze trim.
    pub gaze_correction: Arc<Mutex<GazeCorrection>>,
    /// Per-eye EyeWide capability gate. Weak source signals are reported and
    /// disabled instead of being amplified into false expressions.
    pub wide_enabled: Arc<Mutex<[bool; 2]>>,
    /// Live master switch for eyebrow inference/output. The model remains loaded while
    /// disabled so tracking can resume without rebuilding the pipeline.
    pub eyebrow_enabled: Arc<AtomicBool>,
    /// Live calibration parameters (UI exposes these as sliders).
    pub tuning: Arc<Mutex<Tuning>>,
    /// Live PER-EYE ML-input geometry `[left, right]` (crop/stretch/rotation), read by
    /// the ML thread each frame and edited live in the gear modal. Identity = legacy
    /// resize for that eye.
    pub geometry: Arc<Mutex<[crate::core::types::MlGeometry; 2]>>,
    /// Live per-device specular-dot suppression applied to the ML input (before geometry).
    pub despeckle: Arc<Mutex<crate::core::types::DespeckleParams>>,
    /// Live per-device illumination flatten (close-up shadow removal), after despeckle.
    pub flatten: Arc<Mutex<crate::core::types::FlattenParams>>,
    /// Live per-device adaptive brightness normalization (params + learned target).
    pub brightness: Arc<Mutex<crate::core::types::BrightnessNorm>>,
    /// The per-eye brightness affine (a,b) the ML thread last applied — for the UI preview.
    pub bright_affine: Arc<Mutex<[[f32; 2]; 2]>>,
    /// On-demand ML occlusion heatmap (diagnostic), computed by the ML thread.
    pub heatmap: Arc<HeatState>,
    /// Live device status string (for the UI's diagnostic line).
    pub device_status: Arc<Mutex<String>>,
    /// The live eyebrow model, shared with the ML thread so a freshly trained model can be
    /// hot-swapped in ([`Pipeline::set_brow`]) with no device reconnect. `None` = no brow.
    brow: Arc<Mutex<Option<BrowNet>>>,
    /// Optional custom XR5 EyeWide model; shared so a fitted model can be hot-loaded later.
    wide: Arc<Mutex<Option<WideNet>>>,
    /// Dedicated reset for custom Wide normalization when its model is hot-swapped.
    wide_recenter: Arc<AtomicBool>,
}

/// Horizontally mirror a `w`x`h` grayscale image (row-major).
fn mirror_h(px: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let row = y * w;
        for x in 0..w {
            out[row + (w - 1 - x)] = px[row + x];
        }
    }
    out
}

fn merge_eye_sample(dst: &mut EyeSample, src: EyeSample) {
    if src.gaze_reported {
        if src.gaze_valid {
            dst.gaze = src.gaze;
            dst.gaze_valid = true;
        }
        dst.gaze_reported = true;
    }
    if src.origin_valid {
        dst.origin_mm = src.origin_mm;
        dst.origin_valid = true;
    }
    if src.pupil_valid {
        dst.pupil_mm = src.pupil_mm;
        dst.pupil_valid = true;
    }
    if src.pupil_pos_valid {
        dst.pupil_pos = src.pupil_pos;
        dst.pupil_pos_valid = true;
    }
    if src.openness_reported {
        dst.openness = src.openness;
        dst.openness_valid = src.openness_valid;
        dst.openness_reported = true;
    }
}

fn merge_gaze_sample(dst: &mut GazeSample, src: GazeSample) {
    if src.timestamp_us != 0 {
        dst.timestamp_us = src.timestamp_us;
    }
    merge_eye_sample(&mut dst.left, src.left);
    merge_eye_sample(&mut dst.right, src.right);
}

/// Blend the legacy openness/squeeze pose toward fully open in proportion to an ACTIVE
/// Custom Wide gesture. The output envelope has a short release tail for visual smoothness;
/// that tail must not pull a genuinely narrowed lid back toward 1.0 after the Wide gesture
/// has ended. When the bilateral Wide values match, use one shared base openness so
/// gaze-dependent L/R dips cannot split the pose.
fn apply_custom_wide_pose(results: &mut [EyeResult; 2], use_custom: bool, active: [bool; 2]) {
    if !use_custom {
        return;
    }
    let bilateral = !results[0].blink
        && !results[1].blink
        && active[0]
        && active[1]
        && results[0].wide > 0.002
        && results[1].wide > 0.002
        && (results[0].wide - results[1].wide).abs() < 1.0e-4;
    if bilateral {
        let wide = results[0].wide.max(results[1].wide).clamp(0.0, 1.0);
        let natural = results[0].openness.min(results[1].openness).clamp(0.0, 1.0);
        let openness = natural + wide * (1.0 - natural);
        for result in results {
            result.openness = openness;
            result.openness_valid = true;
            result.squeeze *= 1.0 - wide;
        }
        return;
    }
    for (i, result) in results.iter_mut().enumerate() {
        if active[i] && !result.blink && result.wide > 0.002 {
            let wide = result.wide.clamp(0.0, 1.0);
            let natural = result.openness.clamp(0.0, 1.0);
            result.openness = natural + wide * (1.0 - natural);
            result.openness_valid = true;
            result.squeeze *= 1.0 - wide;
        }
    }
}

#[cfg(test)]
mod merge_tests {
    use super::*;

    #[test]
    fn native_openness_disable_survives_mixed_stream_merge() {
        let mut dst = EyeSample::default();
        merge_eye_sample(
            &mut dst,
            EyeSample {
                openness: 0.8,
                openness_valid: true,
                openness_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.openness_valid);

        // A gaze-only packet must not erase the last wearable state.
        merge_eye_sample(&mut dst, EyeSample::default());
        assert!(dst.openness_valid && dst.openness_reported);

        // A wearable packet carrying Disable must replace the previous Enable.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                openness: 0.0,
                openness_valid: false,
                openness_reported: true,
                ..Default::default()
            },
        );
        assert!(!dst.openness_valid && dst.openness_reported);
    }

    #[test]
    fn transient_invalid_gaze_holds_last_valid_but_aux_packet_does_not_disturb_it() {
        let mut dst = EyeSample::default();
        merge_eye_sample(
            &mut dst,
            EyeSample {
                gaze: [0.2, -0.1, 0.97],
                gaze_valid: true,
                gaze_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid && dst.gaze_reported);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);

        // Wearable 1285 contributes pupil/openness only and must not disturb
        // the canonical 1289 gaze state.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                pupil_mm: 3.4,
                pupil_valid: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);

        // A single 1289 invalid status is only a transient. Freshness aging in
        // the emit thread clears it if no later valid sample arrives.
        merge_eye_sample(
            &mut dst,
            EyeSample {
                gaze_valid: false,
                gaze_reported: true,
                ..Default::default()
            },
        );
        assert!(dst.gaze_valid);
        assert_eq!(dst.gaze, [0.2, -0.1, 0.97]);
    }

    #[test]
    fn gaze_validity_uses_a_short_freshness_grace() {
        let now = Instant::now();
        assert!(!gaze_is_fresh(None, now));
        assert!(gaze_is_fresh(Some(now), now));
        let recent = now.checked_sub(Duration::from_millis(100)).unwrap();
        let stale = now.checked_sub(Duration::from_millis(200)).unwrap();
        assert!(gaze_is_fresh(Some(recent), now));
        assert!(!gaze_is_fresh(Some(stale), now));
    }

    #[test]
    fn gaze_correction_centres_and_scales_in_angle_space() {
        let mut gaze = [10.0f32.to_radians().sin(), 0.0, 10.0f32.to_radians().cos()];
        apply_gaze_correction(
            &mut gaze,
            Eye::Left,
            GazeCorrection {
                enabled: true,
                offset_x_deg: [-20.0, 0.0],
                scale_x: [2.0, 1.0],
                ..GazeCorrection::default()
            },
        );
        let [yaw, pitch] = gaze_angles_deg(gaze).unwrap();
        assert!(yaw.abs() < 1.0e-3, "yaw={yaw}");
        assert!(pitch.abs() < 1.0e-3, "pitch={pitch}");
    }

    #[test]
    fn gaze_vergence_moves_eyes_in_opposite_directions() {
        let correction = GazeCorrection {
            enabled: true,
            vergence_deg: 4.0,
            ..Default::default()
        };
        let mut left = [0.0, 0.0, 1.0];
        let mut right = [0.0, 0.0, 1.0];
        apply_gaze_correction(&mut left, Eye::Left, correction);
        apply_gaze_correction(&mut right, Eye::Right, correction);
        assert!(gaze_angles_deg(left).unwrap()[0] < 0.0);
        assert!(gaze_angles_deg(right).unwrap()[0] > 0.0);
    }

    #[test]
    fn custom_wide_overrides_gaze_dependent_openness_dips() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.80;
        results[1].openness = 0.91;
        results[0].squeeze = 0.25;
        results[1].squeeze = 0.10;
        results[0].wide = 0.7;
        results[1].wide = 0.7;

        apply_custom_wide_pose(&mut results, true, [true; 2]);

        assert!((results[0].openness - 0.94).abs() < 1.0e-6);
        assert!((results[1].openness - 0.94).abs() < 1.0e-6);
        assert!((results[0].squeeze - 0.075).abs() < 1.0e-6);
        assert!((results[1].squeeze - 0.03).abs() < 1.0e-6);
    }

    #[test]
    fn custom_wide_release_tail_does_not_pull_narrowed_lids_open() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.30;
        results[1].openness = 0.40;
        results[0].wide = 0.20;
        results[1].wide = 0.20;

        apply_custom_wide_pose(&mut results, true, [false; 2]);

        assert!((results[0].openness - 0.30).abs() < 1.0e-6);
        assert!((results[1].openness - 0.40).abs() < 1.0e-6);
    }

    #[test]
    fn inactive_wide_tail_does_not_open_partner_during_a_blink() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.0;
        results[0].blink = true;
        results[1].openness = 0.35;
        results[0].wide = 0.20;
        results[1].wide = 0.20;

        apply_custom_wide_pose(&mut results, true, [false; 2]);

        assert_eq!(results[0].openness, 0.0);
        assert_eq!(results[1].openness, 0.35);
    }

    #[test]
    fn active_wide_still_keeps_partner_open_during_a_blink() {
        let mut results = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        results[0].openness = 0.0;
        results[0].blink = true;
        results[1].openness = 0.82;
        results[0].wide = 0.0;
        results[1].wide = 0.70;

        apply_custom_wide_pose(&mut results, true, [true; 2]);

        assert_eq!(results[0].openness, 0.0);
        assert!((results[1].openness - 0.946).abs() < 1.0e-6);
    }
}

const GAZE_VALID_GRACE: Duration = Duration::from_millis(150);

fn gaze_is_fresh(last_valid: Option<Instant>, now: Instant) -> bool {
    last_valid.is_some_and(|t| now.saturating_duration_since(t) <= GAZE_VALID_GRACE)
}

/// Convert a direction vector to yaw/pitch degrees. `None` rejects the zero/non-finite
/// sentinel so correction can never turn missing gaze into a valid-looking direction.
pub(crate) fn gaze_angles_deg(gaze: [f32; 3]) -> Option<[f32; 2]> {
    if !gaze.iter().all(|v| v.is_finite()) {
        return None;
    }
    let horizontal = gaze[0].hypot(gaze[2]);
    if horizontal <= 1.0e-6 || gaze[2] <= 0.0 {
        return None;
    }
    Some([
        gaze[0].atan2(gaze[2]).to_degrees(),
        gaze[1].atan2(horizontal).to_degrees(),
    ])
}

/// Apply angular centre/range/vergence correction while preserving a normalized 3-D
/// direction. This operates after per-device handedness mapping and before every sink.
pub(crate) fn apply_gaze_correction(gaze: &mut [f32; 3], eye: Eye, correction: GazeCorrection) {
    if !correction.enabled {
        return;
    }
    let Some([yaw_deg, pitch_deg]) = gaze_angles_deg(*gaze) else {
        return;
    };
    let i = eye.idx();
    let sx = if correction.scale_x[i].is_finite() {
        correction.scale_x[i].clamp(0.25, 2.5)
    } else {
        1.0
    };
    let sy = if correction.scale_y[i].is_finite() {
        correction.scale_y[i].clamp(0.25, 2.5)
    } else {
        1.0
    };
    let ox = if correction.offset_x_deg[i].is_finite() {
        correction.offset_x_deg[i].clamp(-30.0, 30.0)
    } else {
        0.0
    };
    let oy = if correction.offset_y_deg[i].is_finite() {
        correction.offset_y_deg[i].clamp(-30.0, 30.0)
    } else {
        0.0
    };
    let vergence = if correction.vergence_deg.is_finite() {
        correction.vergence_deg.clamp(-20.0, 20.0)
    } else {
        0.0
    };
    let eye_sign = if eye == Eye::Left { -0.5 } else { 0.5 };
    let yaw = (yaw_deg * sx + ox + vergence * eye_sign).to_radians();
    let pitch = (pitch_deg * sy + oy).to_radians();
    let cp = pitch.cos();
    *gaze = [yaw.sin() * cp, pitch.sin(), yaw.cos() * cp];
}

impl Pipeline {
    /// Start the adapter and the ML + emit threads. `net` is `None` to run
    /// without ML (gaze still flows; openness held at the neutral 0.5).
    pub fn run(
        mut adapter: Box<dyn HmdAdapter>,
        net: Option<EyeNet>,
        brow: Option<BrowNet>,
        wide: Option<WideNet>,
        mut sinks: Vec<Box<dyn OutputSink>>,
        map: DeviceMap,
        device_status: Arc<Mutex<String>>,
        device_key: String,
        init: PipelineInit,
    ) -> io::Result<Pipeline> {
        let stop = Arc::new(AtomicBool::new(false));
        let paused = Arc::new(AtomicBool::new(false));
        let recenter = Arc::new(AtomicBool::new(false));
        let wide_recenter = Arc::new(AtomicBool::new(false));
        let guided_calibration = Arc::new(Mutex::new(None));
        let diag_rec = Arc::new(AtomicBool::new(false));
        let swap_eyes = Arc::new(AtomicBool::new(map.swap_eyes));
        let flip_image = Arc::new(AtomicBool::new(map.flip_image));
        let ml_mirror_l = Arc::new(AtomicBool::new(init.ml_mirror[0]));
        let ml_mirror_r = Arc::new(AtomicBool::new(init.ml_mirror[1]));
        let flip_gaze_x = Arc::new(AtomicBool::new(map.flip_gaze_x));
        let gaze_correction = Arc::new(Mutex::new(init.gaze_correction));
        let wide_enabled = Arc::new(Mutex::new(init.wide_enabled));
        let eyebrow_enabled = Arc::new(AtomicBool::new(init.eyebrow_enabled));
        let tuning = Arc::new(Mutex::new(init.tuning));
        let geometry = Arc::new(Mutex::new(init.geometry));
        let despeckle = Arc::new(Mutex::new(init.despeckle));
        let flatten = Arc::new(Mutex::new(init.flatten));
        let brightness = Arc::new(Mutex::new(init.brightness));
        let bright_affine = Arc::new(Mutex::new([[1.0f32, 0.0]; 2]));
        let heatmap = Arc::new(HeatState::new());
        let ml_loaded = net.is_some();
        let net = net.map(|net| Box::new(LegacyEyelidModel::new(net)) as Box<dyn EyelidModel>);
        let brow_loaded = brow.is_some();
        let wide_loaded = wide.is_some();
        // Brow lives behind a shared handle so a freshly trained model can be hot-swapped
        // into the running ML thread (see `set_brow`) without a device reconnect.
        let brow = Arc::new(Mutex::new(brow));
        let wide = Arc::new(Mutex::new(wide));
        let tele = Telemetry::new(ml_loaded, brow_loaded, wide_loaded, adapter.profile());

        // Adapter callbacks: apply per-unit mapping, then stash newest frame per eye.
        let t_cb = tele.clone();
        let (cb_swap, cb_flip) = (swap_eyes.clone(), flip_image.clone());
        let on_frame = Box::new(move |eye: Eye, w: u32, h: u32, px: &[u8]| {
            let eye = if cb_swap.load(Ordering::Relaxed) {
                eye.opposite()
            } else {
                eye
            };
            let sz = (w as usize) * (h as usize);
            // Mirror per the ACTUAL frame width (not a hardcoded 200) so the flip is
            // correct at any resolution.
            let stored = if cb_flip.load(Ordering::Relaxed) && px.len() >= sz {
                mirror_h(&px[..sz], w as usize, h as usize)
            } else {
                px.to_vec()
            };
            if let Ok(mut f) = t_cb.frames.lock() {
                f[eye.idx()] = Some((w, h, stored));
            }
            match eye {
                Eye::Left => &t_cb.c_frame_l,
                Eye::Right => &t_cb.c_frame_r,
            }
            .fetch_add(1, Ordering::Relaxed);
        });
        let t_g = tele.clone();
        let g_swap = swap_eyes.clone();
        let on_gaze = Box::new(move |s: GazeSample| {
            // Apply the SAME L/R swap as the image (on_frame), so a backwards-wired unit
            // corrects its gaze + pupil together with the image-derived openness/wide/
            // squeeze/brow — swapping now flips ALL per-eye parameters consistently.
            let s = if g_swap.load(Ordering::Relaxed) {
                GazeSample {
                    timestamp_us: s.timestamp_us,
                    left: s.right,
                    right: s.left,
                }
            } else {
                s
            };
            let valid = [
                s.left.gaze_reported && s.left.gaze_valid,
                s.right.gaze_reported && s.right.gaze_valid,
            ];
            if let Ok(mut g) = t_g.gaze.lock() {
                merge_gaze_sample(&mut g, s);
                if valid[0] || valid[1] {
                    if let Ok(mut last) = t_g.gaze_last_valid.lock() {
                        let now = Instant::now();
                        for i in 0..2 {
                            if valid[i] {
                                last[i] = Some(now);
                            }
                        }
                    }
                }
                if let Ok(mut pu) = t_g.pupil.lock() {
                    *pu = [
                        (g.left.pupil_mm, g.left.pupil_valid),
                        (g.right.pupil_mm, g.right.pupil_valid),
                    ];
                }
            }
            t_g.c_gaze.fetch_add(1, Ordering::Relaxed);
        });
        adapter.start(on_frame, on_gaze)?;

        let mut threads = Vec::new();

        // ML thread @ ~60Hz: newest stereo pair -> eye openness/squeeze + optional brow.
        // ALWAYS spawned (even with neither model): it reads `brow` from the shared handle
        // each iteration, so `set_brow` can hot-load a freshly trained model into a running
        // pipeline with no reconnect. Idle cost is a 60Hz sleep loop with a cheap frame peek.
        {
            let (t_ml, ms) = (tele.clone(), stop.clone());
            let (mml, mmr) = (ml_mirror_l.clone(), ml_mirror_r.clone());
            let mut net = net;
            let brow = brow.clone();
            let wide = wide.clone();
            let mgeo = geometry.clone();
            let mdsp = despeckle.clone();
            let mflt = flatten.clone();
            let mbn = brightness.clone();
            let maff = bright_affine.clone();
            let hm = heatmap.clone();
            threads.push(thread::spawn(move || {
                const BN: usize = preprocess::BROW_SIDE * preprocess::BROW_SIDE;
                let period = Duration::from_millis(16);
                let mut bin = [0.0f32; BN];
                let mut win = [0.0f32; BN];
                // Per-eye brightness baselines (runtime; the learned target lives in `mbn`).
                let mut bright_st = [crate::ml::brightness::BrightState::default(); 2];
                // Previous frame's raw model openness (s[1]/s[2]) — gates the brightness target
                // capture to genuinely relaxed-OPEN frames.
                let mut prev_open = [0.5f32; 2];
                let mut model_error_reported = false;
                while !ms.load(Ordering::Relaxed) {
                    let (l, r) = {
                        let f = t_ml.frames.lock().unwrap();
                        (f[0].clone(), f[1].clone())
                    };
                    if let (Some((lw, lh, l)), Some((rw, rh, r))) = (l, r) {
                        if l.len() >= (lw as usize) * (lh as usize)
                            && r.len() >= (rw as usize) * (rh as usize)
                        {
                            // Eye net: ONE pass over both eyes (L in ch0, R in ch1), each
                            // resized to 100x100. Outputs (RE'd 2026-06-26): s0=presence,
                            // s1=L openness, s2=R openness, s3=L squeeze, s4=R squeeze.
                            if let Some(net) = net.as_mut() {
                                let geom = *mgeo.lock().unwrap();
                                // Suppress specular spots (glasses / IR glints) BEFORE the
                                // model — the heatmaps showed the net reads brightness as
                                // "more open", so a reflection inflates/destabilizes openness.
                                let dsp = *mdsp.lock().unwrap();
                                let lf = preprocess::despeckle(&l, lw as usize, lh as usize, &dsp);
                                let rf = preprocess::despeckle(&r, rw as usize, rh as usize, &dsp);
                                // Illumination flatten (close-up shadow removal), after despeckle.
                                let flt = *mflt.lock().unwrap();
                                let lf = preprocess::flatten(&lf, lw as usize, lh as usize, &flt);
                                let rf = preprocess::flatten(&rf, rw as usize, rh as usize, &flt);
                                // Adaptive brightness/contrast normalization (per user, slow
                                // baseline) — holds the input at a learned target so lens-
                                // distance drift doesn't bias openness; blinks pass through.
                                let aff = {
                                    let mut norm = mbn.lock().unwrap();
                                    crate::ml::brightness::step(
                                        &mut bright_st,
                                        [
                                            (&lf, lw as usize, lh as usize),
                                            (&rf, rw as usize, rh as usize),
                                        ],
                                        prev_open,
                                        &mut norm,
                                    )
                                };
                                if let Ok(mut a) = maff.lock() {
                                    *a = [[aff[0].0, aff[0].1], [aff[1].0, aff[1].1]];
                                }
                                let nlf = crate::ml::brightness::apply(&lf, aff[0].0, aff[0].1);
                                let nrf = crate::ml::brightness::apply(&rf, aff[1].0, aff[1].1);
                                let mirr =
                                    [mml.load(Ordering::Relaxed), mmr.load(Ordering::Relaxed)];
                                let input = preprocess::to_input_stereo_geom(
                                    &nlf, lw, lh, &nrf, rw, rh, mirr[0], mirr[1], &geom[0],
                                    &geom[1],
                                );
                                // Publish the exact net input for the dashboard's NET
                                // view, un-mirrored back to natural orientation.
                                if let Ok(mut mi) = t_ml.ml_input.lock() {
                                    let n = preprocess::DST;
                                    for e in 0..2 {
                                        let sl = &input[e * n * n..(e + 1) * n * n];
                                        let mut px = vec![0u8; n * n];
                                        for y in 0..n {
                                            for x in 0..n {
                                                let sx = if mirr[e] { n - 1 - x } else { x };
                                                px[y * n + x] = (sl[y * n + sx] * 255.0)
                                                    .clamp(0.0, 255.0)
                                                    as u8;
                                            }
                                        }
                                        mi[e] = Some(px);
                                    }
                                }
                                // Preprocessing is defined to produce exactly CHW [2, 100, 100].
                                // The old EyeNet path also relied on this invariant and would fail
                                // rather than silently retaining a stale sample if it were broken.
                                debug_assert_eq!(input.len(), EYELID_INPUT_LEN);
                                let canonical = CanonicalStereoInput::try_from(input.as_slice())
                                    .expect("eyelid preprocessing must produce CHW [2, 100, 100]");
                                let publish = match net.infer(canonical) {
                                    Ok(prediction) => prediction.require_legacy_frame().ok(),
                                    Err(_) => None,
                                };
                                if let Some(frame) = publish {
                                    // Preserve the raw-openness feedback loop byte-for-byte:
                                    // this gates next frame's brightness baseline capture.
                                    prev_open = frame.ml_raw;
                                    if let Ok(mut o) = t_ml.ml_raw.lock() {
                                        *o = frame.ml_raw;
                                    }
                                    if let Ok(mut o5) = t_ml.ml5.lock() {
                                        *o5 = frame.ml5;
                                    }
                                    t_ml.c_ml.fetch_add(1, Ordering::Relaxed);
                                } else if !model_error_reported {
                                    // Unreachable for the only Phase 1 backend. Do not invent
                                    // zeroes for a missing capability or failed inference.
                                    eprintln!(
                                        "[ml] eyelid backend did not provide the required legacy frame; retaining the previous eyelid sample"
                                    );
                                    model_error_reported = true;
                                }
                                // On-demand occlusion heatmap: reuse this frame's input +
                                // net. Blocks the ML loop ~2s (openness freezes) — a manual
                                // one-shot diagnostic, so that's acceptable.
                                if hm.req.swap(false, Ordering::Relaxed) {
                                    hm.computing.store(true, Ordering::Relaxed);
                                    let mode = crate::ml::heatmap::HeatMode::from_u8(
                                        hm.mode.load(Ordering::Relaxed),
                                    );
                                    let res = crate::ml::heatmap::compute_model(
                                        net.as_mut(),
                                        &input,
                                        mode,
                                    );
                                    if let (Some(res), Ok(mut r)) = (res, hm.result.lock()) {
                                        *r = Some(res);
                                    }
                                    hm.computing.store(false, Ordering::Relaxed);
                                }
                            }
                            // Brow net: per eye, on the same frames (right eye flipped to
                            // the model's left-canonical orientation). Publish the RAW
                            // output (NOT clamped/smoothed) + a generation; the emit thread
                            // does the blink-gated EMA + baseline. Drop non-finite outputs
                            // so a bad weight/frame can never poison the downstream filter.
                            // Read the model from the SHARED handle so a hot-swapped model
                            // (set_brow) takes effect on the next iteration with no reconnect.
                            if let Ok(mut guard) = brow.lock() {
                                if let Some(brow) = guard.as_mut() {
                                    preprocess::brow_input(
                                        &l,
                                        lw as usize,
                                        lh as usize,
                                        false,
                                        &mut bin,
                                    );
                                    let bl = brow.forward_one(&bin)[0];
                                    preprocess::brow_input(
                                        &r,
                                        rw as usize,
                                        rh as usize,
                                        true,
                                        &mut bin,
                                    );
                                    let br = brow.forward_one(&bin)[0];
                                    if bl.is_finite() && br.is_finite() {
                                        if let Ok(mut o) = t_ml.brow_raw.lock() {
                                            *o = [bl, br];
                                        }
                                        t_ml.c_brow.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                            // Custom Dream Air/XR5 EyeWide model. It shares the TinyEyeNet
                            // runtime with brow but uses a full-eye crop and its own task-tagged
                            // weights. The legacy EyeWide path continues in parallel for A/B.
                            if let Ok(mut guard) = wide.lock() {
                                if let Some(wide) = guard.as_mut() {
                                    preprocess::wide_input(
                                        &l,
                                        lw as usize,
                                        lh as usize,
                                        false,
                                        &mut win,
                                    );
                                    let wl = wide.forward_one(&win);
                                    preprocess::wide_input(
                                        &r,
                                        rw as usize,
                                        rh as usize,
                                        true,
                                        &mut win,
                                    );
                                    let wr = wide.forward_one(&win);
                                    if wl.is_finite() && wr.is_finite() {
                                        if let Ok(mut raw) = t_ml.wide_raw.lock() {
                                            *raw = [wl, wr];
                                        }
                                        t_ml.c_wide.fetch_add(1, Ordering::Relaxed);
                                    }
                                }
                            }
                        }
                    }
                    thread::sleep(period);
                }
            }));
        }

        // Emit thread @ ~120Hz: post-process -> telemetry + sinks.
        let (t_em, es, ep, er) = (tele.clone(), stop.clone(), paused.clone(), recenter.clone());
        let edr = diag_rec.clone();
        let t_tune = tuning.clone();
        let fgx = flip_gaze_x.clone();
        let gaze_trim = gaze_correction.clone();
        let guided_seed = guided_calibration.clone();
        let wide_on = wide_enabled.clone();
        let wide_reset = wide_recenter.clone();
        let brow_on = eyebrow_enabled.clone();
        let wide_source = init.wide_source;
        threads.push(thread::spawn(move || {
            let mut state = SRanipalState::new();
            let mut brow_state = BrowState::default();
            let mut custom_wide_state = WideState::default();
            let mut last_brow_gen = 0u64;
            let mut last_wide_gen = 0u64;
            let mut last_wide_infer = Instant::now();
            let mut wide_fresh_at = Instant::now() - Duration::from_secs(10);
            let period = Duration::from_micros(8333);
            let mut last = std::time::Instant::now();
            // Persisted calibration: load the relaxed-open baseline on start, then
            // re-save periodically (and on stop) so it survives restarts. Anchored to
            // the app dir (not CWD) so it persists when launched from a shortcut.
            let calib_path = crate::config::calib_path().to_string_lossy().into_owned();
            if let Some(store) = crate::core::eye_state::load_calib(&calib_path) {
                state.restore_all(&store);
            }
            let mut since_save = 0u32;
            // Diagnostic CSV recorder state (REC button; see Pipeline::diag_rec).
            let mut diag_file: Option<std::io::BufWriter<std::fs::File>> = None;
            let mut diag_t0 = std::time::Instant::now();
            while !es.load(Ordering::Relaxed) {
                // A cycle that took >=2x the target period means we missed one or
                // more 120Hz slots — count the shortfall as real dropped frames.
                let elapsed = last.elapsed();
                last = std::time::Instant::now();
                let missed = (elapsed.as_micros() / 8333).saturating_sub(1);
                if missed > 0 {
                    t_em.c_drop.fetch_add(missed as u64, Ordering::Relaxed);
                }
                state.tuning = *t_tune.lock().unwrap();
                if let Ok(mut seed) = guided_seed.lock() {
                    if let Some(store) = seed.take() {
                        state.restore_all(&store);
                        crate::core::eye_state::save_calib(&calib_path, &store);
                        since_save = 0;
                    }
                }
                let did_recenter = er.swap(false, Ordering::Relaxed);
                if did_recenter {
                    state.recenter();
                    brow_state.recenter(); // re-baseline the brow neutral too
                }
                if did_recenter || wide_reset.swap(false, Ordering::Relaxed) {
                    custom_wide_state.recenter();
                    *t_em.wide_ready.lock().unwrap() = [false; 2];
                    *t_em.wide_bootstrap_seen.lock().unwrap() = [0; 2];
                }
                let g = t_em.fresh_gaze();
                let m5 = *t_em.ml5.lock().unwrap();
                let mut results = state.process_frame(m5, &g, t_em.ml_loaded);
                let sranipal_wide = [results[0].wide, results[1].wide];
                if let Ok(mut value) = t_em.wide_sranipal.lock() {
                    *value = sranipal_wide;
                }

                // Always calculate Custom Wide in parallel when a model is loaded, even
                // while SRanipal remains selected. This gives us same-frame A/B telemetry
                // before the user entrusts VRCFT output to the new model.
                let mut custom = [None, None];
                let mut custom_pose_active = [false; 2];
                let mut custom_fresh = false;
                if t_em.wide_loaded.load(Ordering::Relaxed) {
                    let raw = *t_em.wide_raw.lock().unwrap();
                    let generation = t_em.c_wide.load(Ordering::Relaxed);
                    let is_new = generation != last_wide_gen;
                    let now = Instant::now();
                    let infer_dt = if is_new {
                        let dt = now.duration_since(last_wide_infer).as_secs_f32();
                        last_wide_infer = now;
                        wide_fresh_at = now;
                        last_wide_gen = generation;
                        dt
                    } else {
                        0.0
                    };
                    custom = custom_wide_state.process_pair(
                        raw,
                        is_new,
                        [results[0].blink, results[1].blink],
                        infer_dt,
                        elapsed.as_secs_f32(),
                        state.tuning.wide_requires_both,
                    );
                    let wide_diag = custom_wide_state.diag();
                    custom_pose_active = wide_diag.active;
                    *t_em.wide_ready.lock().unwrap() = wide_diag.ready;
                    *t_em.wide_bootstrap_seen.lock().unwrap() = wide_diag.bootstrap_seen;
                    custom_fresh = now.duration_since(wide_fresh_at) <= Duration::from_millis(500)
                        && custom[0].is_some()
                        && custom[1].is_some();
                    if let Ok(mut value) = t_em.wide_custom.lock() {
                        *value = [custom[0].unwrap_or(0.0), custom[1].unwrap_or(0.0)];
                    }
                }

                let use_custom = match wide_source {
                    WideSource::Sranipal => false,
                    WideSource::Auto | WideSource::Custom => custom_fresh,
                };
                if use_custom {
                    results[0].wide = custom[0].unwrap_or(0.0);
                    results[1].wide = custom[1].unwrap_or(0.0);
                } else if wide_source == WideSource::Custom {
                    // Strict selection never silently falls back to SRanipal. During the
                    // short neutral bootstrap (or a stale model) it emits a safe zero.
                    results[0].wide = 0.0;
                    results[1].wide = 0.0;
                }
                t_em.wide_custom_active.store(use_custom, Ordering::Relaxed);

                // The guided XR5 capability result gates whichever provider was selected.
                let wide = *wide_on.lock().unwrap();
                for i in 0..2 {
                    if !wide[i] {
                        results[i].wide = 0.0;
                    }
                }
                apply_custom_wide_pose(&mut results, use_custom, custom_pose_active);
                // Brow: blink-gated EMA + baseline of the ML thread's raw brow, per eye.
                // `is_new` advances the EMA once per inference (not per 120Hz tick); the
                // result is None until the first open inference, so we never emit a
                // baseline derived from the zero placeholder.
                if brow_on.load(Ordering::Relaxed) && t_em.brow_loaded.load(Ordering::Relaxed) {
                    let raw = *t_em.brow_raw.lock().unwrap();
                    let gen = t_em.c_brow.load(Ordering::Relaxed);
                    let is_new = gen != last_brow_gen;
                    last_brow_gen = gen;
                    for (i, r) in results.iter_mut().enumerate() {
                        if let Some(b) =
                            brow_state.process(i, raw[i], is_new, r.blink, did_recenter)
                        {
                            r.brow = b;
                            r.brow_valid = true;
                        }
                    }
                }
                // Per-device gaze handedness: negate X so left/right isn't mirrored.
                // Applied here so telemetry AND all sinks see the same corrected gaze.
                if fgx.load(Ordering::Relaxed) {
                    for r in results.iter_mut() {
                        r.gaze[0] = -r.gaze[0];
                    }
                }
                let correction = *gaze_trim.lock().unwrap();
                apply_gaze_correction(&mut results[0].gaze, Eye::Left, correction);
                apply_gaze_correction(&mut results[1].gaze, Eye::Right, correction);
                if let Ok(mut pu) = t_em.pupil.lock() {
                    *pu = [
                        (results[0].pupil_mm, results[0].pupil_valid),
                        (results[1].pupil_mm, results[1].pupil_valid),
                    ];
                }
                if let Ok(mut slot) = t_em.results.lock() {
                    *slot = results;
                }
                if let Ok(mut b) = t_em.baselines.lock() {
                    *b = [state.baseline(Eye::Left), state.baseline(Eye::Right)];
                }
                if let Ok(mut calibration) = t_em.calibration.lock() {
                    *calibration = Some(state.snapshot_all());
                }
                // Diagnostic recorder: raw ml values next to every post-processing
                // internal, one CSV row per emit frame, so a mis-correction can be
                // diagnosed offline from a short recorded session.
                if edr.load(Ordering::Relaxed) {
                    use std::io::Write;
                    if diag_file.is_none() {
                        let ts = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_secs())
                            .unwrap_or(0);
                        let path =
                            crate::config::base_dir().join(format!("sranibro_diag_{ts}.csv"));
                        match std::fs::File::create(&path) {
                            Ok(f) => {
                                let mut w = std::io::BufWriter::new(f);
                                let _ = writeln!(
                                    w,
                                    "t_ms,raw_l,raw_r,ch0_l,ch0_r,sq3_l,sq3_r,\
                                     x_l,x_r,anchor_l,anchor_r,staged_l,staged_r,\
                                     open_l,open_r,wide_l,wide_r,squeeze_l,squeeze_r,\
                                     blink_l,blink_r,baseline_l,baseline_r,\
                                     closed_ref_l,closed_ref_r,is_wide_l,is_wide_r,\
                                     latched_l,latched_r,fall_run_l,fall_run_r,\
                                     since_down_l,since_down_r,blink_len_l,blink_len_r,\
                                     ep_exit_l,ep_exit_r,pend_w,\
                                     gin_lx,gin_ly,gin_rx,gin_ry,gv_l,gv_r,\
                                     gout_lx,gout_ly,gout_rx,gout_ry,\
                                     yoked_l,yoked_r,yhold_l,yhold_r,\
                                     ch2_l,ch2_r,ch4_l,ch4_r,\
                                     native_open_l,native_open_r,\
                                     native_open_valid_l,native_open_valid_r,\
                                     native_open_reported_l,native_open_reported_r,\
                                     wide_raw_l,wide_raw_r,wide_custom_l,wide_custom_r,\
                                     wide_sranipal_l,wide_sranipal_r,wide_custom_active"
                                );
                                println!("[diag] recording to {}", path.display());
                                diag_t0 = std::time::Instant::now();
                                diag_file = Some(w);
                            }
                            Err(e) => {
                                eprintln!("[diag] could not create the recording file: {e}");
                                edr.store(false, Ordering::Relaxed);
                            }
                        }
                    }
                    if let Some(w) = diag_file.as_mut() {
                        let d = state.diag();
                        let wide_raw_diag = *t_em.wide_raw.lock().unwrap();
                        let wide_custom_diag = *t_em.wide_custom.lock().unwrap();
                        let _ = writeln!(
                            w,
                            "{},{:.4},{:.4},{:.3},{:.3},{:.4},{:.4},\
                             {:.4},{:.4},{:.4},{:.4},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},\
                             {},{},{:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{},{},{},{},{},{},{:.2},\
                             {:.4},{:.4},{:.4},{:.4},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},\
                             {:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                            diag_t0.elapsed().as_millis(),
                            m5[0][1],
                            m5[1][1],
                            m5[0][0],
                            m5[1][0],
                            m5[0][3],
                            m5[1][3],
                            d.ramp_pre[0],
                            d.ramp_pre[1],
                            d.mid_anchor[0],
                            d.mid_anchor[1],
                            d.staged[0] as u8,
                            d.staged[1] as u8,
                            results[0].openness,
                            results[1].openness,
                            results[0].wide,
                            results[1].wide,
                            results[0].squeeze,
                            results[1].squeeze,
                            results[0].blink as u8,
                            results[1].blink as u8,
                            d.baseline[0],
                            d.baseline[1],
                            d.closed_ref[0],
                            d.closed_ref[1],
                            d.is_wide[0] as u8,
                            d.is_wide[1] as u8,
                            d.latched[0] as u8,
                            d.latched[1] as u8,
                            d.fall_run[0],
                            d.fall_run[1],
                            d.since_down[0].min(9999),
                            d.since_down[1].min(9999),
                            d.blink_len[0],
                            d.blink_len[1],
                            d.ep_exit[0] as u8,
                            d.ep_exit[1] as u8,
                            d.pend_w,
                            // Gaze diagnostics: device-in vs emitted-out + yoke state
                            // (the emitted gaze has flip_gaze_x already applied above,
                            // matching exactly what the sinks send).
                            g.left.gaze[0],
                            g.left.gaze[1],
                            g.right.gaze[0],
                            g.right.gaze[1],
                            g.left.gaze_valid as u8,
                            g.right.gaze_valid as u8,
                            results[0].gaze[0],
                            results[0].gaze[1],
                            results[1].gaze[0],
                            results[1].gaze[1],
                            results[0].gaze_yoked as u8,
                            results[1].gaze_yoked as u8,
                            d.yoke_hold[0] as u8,
                            d.yoke_hold[1] as u8,
                            m5[0][2],
                            m5[1][2],
                            m5[0][4],
                            m5[1][4],
                            g.left.openness,
                            g.right.openness,
                            g.left.openness_valid as u8,
                            g.right.openness_valid as u8,
                            g.left.openness_reported as u8,
                            g.right.openness_reported as u8,
                            wide_raw_diag[0],
                            wide_raw_diag[1],
                            wide_custom_diag[0],
                            wide_custom_diag[1],
                            sranipal_wide[0],
                            sranipal_wide[1],
                            use_custom as u8,
                        );
                    }
                } else if let Some(mut w) = diag_file.take() {
                    use std::io::Write;
                    let _ = w.flush();
                    println!("[diag] recording stopped");
                }
                t_em.c_emit.fetch_add(1, Ordering::Relaxed);
                if !ep.load(Ordering::Relaxed) {
                    for s in sinks.iter_mut() {
                        s.on_frame(&results);
                    }
                }
                since_save += 1;
                if since_save >= 2400 {
                    since_save = 0;
                    crate::core::eye_state::save_calib(&calib_path, &state.snapshot_all());
                }
                thread::sleep(period);
            }
            crate::core::eye_state::save_calib(&calib_path, &state.snapshot_all());
        }));

        Ok(Pipeline {
            adapter,
            device_key,
            stop,
            threads,
            tele,
            paused,
            recenter,
            guided_calibration,
            diag_rec,
            swap_eyes,
            flip_image,
            ml_mirror_l,
            ml_mirror_r,
            flip_gaze_x,
            gaze_correction,
            eyebrow_enabled,
            tuning,
            geometry,
            despeckle,
            flatten,
            brightness,
            bright_affine,
            heatmap,
            device_status,
            brow,
            wide,
            wide_recenter,
            wide_enabled,
        })
    }

    /// Hot-swap the eyebrow model into the LIVE pipeline with no device reconnect. Pass
    /// `Some(net)` to load a freshly trained model (e.g. right after B-2 train+bake) or
    /// `None` to drop brow output. The running ML thread picks it up on its next iteration
    /// (it reads the shared handle), and `brow_loaded` flips so the emit thread + UI react.
    ///
    /// Note: brow is only *useful* when the eye (SRanipal) model is also loaded — the emit
    /// thread blink-gates brow on the eye net's openness. Loading brow without the eye net
    /// still stores it, but no brow is emitted until the eye net is present.
    pub fn set_brow(&self, net: Option<BrowNet>) {
        let loaded = net.is_some();
        if let Ok(mut g) = self.brow.lock() {
            *g = net;
        }
        self.tele.brow_loaded.store(loaded, Ordering::Relaxed);
    }

    /// Hot-swap the custom XR5 EyeWide model. The next ML iteration starts producing
    /// A/B telemetry; source selection still follows `[hmd].wide_source` after reload.
    pub fn set_wide(&self, net: Option<WideNet>) {
        let loaded = net.is_some();
        if let Ok(mut guard) = self.wide.lock() {
            *guard = net;
        }
        self.tele.wide_loaded.store(loaded, Ordering::Relaxed);
        self.tele.wide_custom_active.store(false, Ordering::Relaxed);
        *self.tele.wide_ready.lock().unwrap() = [false; 2];
        *self.tele.wide_bootstrap_seen.lock().unwrap() = [0; 2];
        self.wide_recenter.store(true, Ordering::Relaxed);
    }

    /// Stop all threads and the adapter.
    pub fn stop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
        self.adapter.stop();
    }
}
