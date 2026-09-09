//! Spatial response heatmaps for the eye model — diagnostics that show how the openness
//! output RESPONDS to each region of the eye image (e.g. the lid/iris boundary or bright
//! specular dots glasses / IR-LEDs throw onto the IR frame).
//!
//! The net is forward-pass only (no autograd), so instead of a gradient we PERTURB a
//! patch of the input and watch the openness output move. Besides erase-to-mean and glint
//! injection, centred brighten/darken and local-contrast probes distinguish "the model
//! needs this shape" from "the model merely reacts to this region becoming brighter".
//! The 8 px / 4 px grid is deliberately high-resolution on the canonical 100x100 input;
//! it runs once on demand rather than as a live per-frame overlay.

use super::eye_net::EyeNet;
use super::eyelid_model::{CanonicalStereoInput, EyelidModel};
use super::preprocess::DST;

/// Which perturbation the heatmap applies to each patch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeatMode {
    /// Erase each patch to the channel mean and measure how much openness moves — the
    /// regions the model actually relies on for its openness estimate.
    OcclusionMean,
    /// Paint each patch bright (simulate a specular glint) and measure how much openness
    /// is corrupted — where a glasses / IR-LED reflection actually hurts the reading.
    GlintInject,
    /// Brighten and darken the same patch by equal amounts. Positive response means local
    /// brightness raises openness; negative means it lowers openness.
    BrightnessSensitivity,
    /// Increase and decrease contrast inside a patch around that patch's own mean. This
    /// keeps local brightness approximately fixed and exposes sensitivity to boundaries.
    ContrastSensitivity,
}

impl HeatMode {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => HeatMode::GlintInject,
            2 => HeatMode::BrightnessSensitivity,
            3 => HeatMode::ContrastSensitivity,
            _ => HeatMode::OcclusionMean,
        }
    }
}

/// Whole-channel response probes shown beside the spatial map. Brightness uses the same
/// multiplicative definition as the live manual-brightness control. Contrast is centred
/// on the eye channel's median, so it does not intentionally move the median brightness.
pub const RESPONSE_FACTORS: [f32; 5] = [0.70, 0.85, 1.00, 1.25, 1.50];

/// One computed heatmap pair (both eyes) plus the grayscale model input it was scored
/// over, so the UI can composite the overlay 1:1 with no resampling offset.
pub struct HeatResult {
    /// Signed openness delta per pixel (`DST*DST` each). `[left, right]`.
    pub delta: [Vec<f32>; 2],
    /// The grayscale model input (`DST*DST` u8) each eye was scored on. `[left, right]`.
    pub base: [Vec<u8>; 2],
    pub mode: HeatMode,
    /// Robust display scale (98th percentile of finite absolute spatial response) and the
    /// true maximum. Auto-scaling uses p98; peak remains visible so weak maps are honest.
    pub p98_abs_delta: f32,
    pub peak_abs_delta: f32,
    /// Raw current-frame openness and one-eye-at-a-time global response curves.
    pub baseline_openness: [f32; 2],
    pub brightness_response: [[f32; RESPONSE_FACTORS.len()]; 2],
    pub contrast_response: [[f32; RESPONSE_FACTORS.len()]; 2],
}

// Patch = 8 px on the 100x100 grid: still wider than conv1's 5x5 receptive field, but
// narrow enough to separate lid margin, iris and glints. Stride 4 gives 50% overlap and
// a 24x24 probe grid instead of the old ~14x14 grid.
const P: usize = 8;
const S: usize = 4;
const BRIGHTNESS_STEP: f32 = 0.12;
const CONTRAST_HIGH: f32 = 1.35;
const CONTRAST_LOW: f32 = 0.65;

/// Top-left patch positions along one axis: 0,4,…,DST-P, with the last clamped to the
/// edge so the whole grid is covered.
fn positions() -> Vec<usize> {
    let mut v: Vec<usize> = (0..=DST - P).step_by(S).collect();
    if v.last() != Some(&(DST - P)) {
        v.push(DST - P);
    }
    v
}

/// Occlusion-sensitivity map for ONE eye. `input` is the live 2x100x100 stereo buffer
/// (`[c][h][w]`, L=ch0 / R=ch1); `eye_ch` = the channel to perturb (0/1); `out_idx` =
/// the openness output to read (1 for L, 2 for R). The OTHER channel is left at its real
/// value throughout (the net is dual-eye — perturbing it would move openness for reasons
/// unrelated to the patch). Returns a `DST*DST` signed delta map.
fn occlusion_map_with(
    input: &[f32],
    eye_ch: usize,
    mode: HeatMode,
    mut infer_openness: impl FnMut(&[f32]) -> Option<f32>,
) -> Option<Vec<f32>> {
    let n = DST;
    let off = eye_ch * n * n;
    let base = infer_openness(input)?;
    let fill = match mode {
        HeatMode::OcclusionMean => {
            let sum: f32 = input[off..off + n * n].iter().sum();
            sum / (n * n) as f32 // erase to the average grey — neutral, no injected feature
        }
        HeatMode::GlintInject => 1.0, // brightest = a specular reflection
        HeatMode::BrightnessSensitivity | HeatMode::ContrastSensitivity => 0.0,
    };
    let mut work = input.to_vec();
    let mut heat = vec![0f32; n * n];
    let mut cover = vec![0f32; n * n];
    let mut saved = vec![0f32; P * P];
    let pos = positions();
    for &cy in &pos {
        for &cx in &pos {
            for y in 0..P {
                for x in 0..P {
                    let idx = off + (cy + y) * n + (cx + x);
                    saved[y * P + x] = work[idx];
                }
            }
            let delta = match mode {
                HeatMode::OcclusionMean | HeatMode::GlintInject => {
                    for y in 0..P {
                        for x in 0..P {
                            work[off + (cy + y) * n + (cx + x)] = fill;
                        }
                    }
                    let o = infer_openness(&work)?;
                    // OcclusionMean: base - o (positive = erasing here lowered openness).
                    // GlintInject: o - base (signed shift a fake glint induces).
                    if mode == HeatMode::OcclusionMean {
                        base - o
                    } else {
                        o - base
                    }
                }
                HeatMode::BrightnessSensitivity => {
                    for y in 0..P {
                        for x in 0..P {
                            let k = y * P + x;
                            work[off + (cy + y) * n + (cx + x)] =
                                (saved[k] + BRIGHTNESS_STEP).clamp(0.0, 1.0);
                        }
                    }
                    let brighter = infer_openness(&work)?;
                    for y in 0..P {
                        for x in 0..P {
                            let k = y * P + x;
                            work[off + (cy + y) * n + (cx + x)] =
                                (saved[k] - BRIGHTNESS_STEP).clamp(0.0, 1.0);
                        }
                    }
                    let darker = infer_openness(&work)?;
                    0.5 * (brighter - darker)
                }
                HeatMode::ContrastSensitivity => {
                    let patch_mean = saved.iter().sum::<f32>() / saved.len() as f32;
                    for y in 0..P {
                        for x in 0..P {
                            let k = y * P + x;
                            work[off + (cy + y) * n + (cx + x)] = (patch_mean
                                + CONTRAST_HIGH * (saved[k] - patch_mean))
                                .clamp(0.0, 1.0);
                        }
                    }
                    let higher = infer_openness(&work)?;
                    for y in 0..P {
                        for x in 0..P {
                            let k = y * P + x;
                            work[off + (cy + y) * n + (cx + x)] = (patch_mean
                                + CONTRAST_LOW * (saved[k] - patch_mean))
                                .clamp(0.0, 1.0);
                        }
                    }
                    let lower = infer_openness(&work)?;
                    0.5 * (higher - lower)
                }
            };
            for y in 0..P {
                for x in 0..P {
                    let k = (cy + y) * n + (cx + x);
                    heat[k] += delta;
                    cover[k] += 1.0;
                    work[off + k] = saved[y * P + x]; // restore
                }
            }
        }
    }
    for k in 0..n * n {
        if cover[k] > 0.0 {
            heat[k] /= cover[k];
        }
    }
    Some(heat)
}

fn median(channel: &[f32]) -> f32 {
    let mut values: Vec<f32> = channel.iter().copied().filter(|v| v.is_finite()).collect();
    if values.is_empty() {
        return 0.5;
    }
    let mid = values.len() / 2;
    values.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    values[mid]
}

fn response_curves(
    input: &[f32],
    mut infer_openness: impl FnMut(&[f32], usize) -> Option<f32>,
) -> Option<([f32; 2], [[f32; 5]; 2], [[f32; 5]; 2])> {
    let plane = DST * DST;
    let mut baseline = [0.0; 2];
    let mut brightness = [[0.0; RESPONSE_FACTORS.len()]; 2];
    let mut contrast = [[0.0; RESPONSE_FACTORS.len()]; 2];
    let mut work = input.to_vec();
    for eye in 0..2 {
        let off = eye * plane;
        let source = &input[off..off + plane];
        let pivot = median(source);
        baseline[eye] = infer_openness(input, eye)?;
        for (index, factor) in RESPONSE_FACTORS.iter().copied().enumerate() {
            for (dst, &src) in work[off..off + plane].iter_mut().zip(source) {
                *dst = (src * factor).clamp(0.0, 1.0);
            }
            brightness[eye][index] = infer_openness(&work, eye)?;
            work[off..off + plane].copy_from_slice(source);

            for (dst, &src) in work[off..off + plane].iter_mut().zip(source) {
                *dst = (pivot + factor * (src - pivot)).clamp(0.0, 1.0);
            }
            contrast[eye][index] = infer_openness(&work, eye)?;
            work[off..off + plane].copy_from_slice(source);
        }
    }
    Some((baseline, brightness, contrast))
}

fn response_scale(delta: &[Vec<f32>; 2]) -> (f32, f32) {
    let mut values: Vec<f32> = delta
        .iter()
        .flat_map(|eye| eye.iter().copied())
        .filter(|v| v.is_finite())
        .map(f32::abs)
        .collect();
    if values.is_empty() {
        return (0.001, 0.0);
    }
    values.sort_by(f32::total_cmp);
    let peak = *values.last().unwrap_or(&0.0);
    let index = ((values.len() - 1) as f32 * 0.98).round() as usize;
    (values[index].max(0.001), peak)
}

pub fn occlusion_map(
    net: &mut EyeNet,
    input: &[f32],
    eye_ch: usize,
    out_idx: usize,
    mode: HeatMode,
) -> Vec<f32> {
    occlusion_map_with(input, eye_ch, mode, |sample| {
        Some(net.forward_one(sample)[out_idx])
    })
    .expect("the legacy EyeNet always provides openness")
}

fn compute_with(
    input: &[f32],
    mode: HeatMode,
    mut infer_openness: impl FnMut(&[f32], usize) -> Option<f32>,
) -> Option<HeatResult> {
    let n = DST;
    let gray = |ch: usize| -> Vec<u8> {
        input[ch * n * n..(ch + 1) * n * n]
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    };
    let dl = occlusion_map_with(input, 0, mode, |sample| infer_openness(sample, 0))?;
    let dr = occlusion_map_with(input, 1, mode, |sample| infer_openness(sample, 1))?;
    let delta = [dl, dr];
    let (p98_abs_delta, peak_abs_delta) = response_scale(&delta);
    let (baseline_openness, brightness_response, contrast_response) =
        response_curves(input, |sample, eye| infer_openness(sample, eye))?;
    Some(HeatResult {
        delta,
        base: [gray(0), gray(1)],
        mode,
        p98_abs_delta,
        peak_abs_delta,
        baseline_openness,
        brightness_response,
        contrast_response,
    })
}

/// Compute both eyes' heatmaps for `mode` over the live `input` (2x100x100), and copy the
/// grayscale model input per eye for the UI overlay base.
pub fn compute(net: &mut EyeNet, input: &[f32], mode: HeatMode) -> HeatResult {
    compute_with(input, mode, |sample, eye| {
        // left eye: input ch0 -> out[1]; right eye: input ch1 -> out[2]
        Some(net.forward_one(sample)[eye + 1])
    })
    .expect("the legacy EyeNet always provides openness")
}

fn model_openness(model: &mut dyn EyelidModel, input: &[f32], eye: usize) -> Option<f32> {
    let input = CanonicalStereoInput::try_from(input).ok()?;
    model.infer(input).ok()?.openness[eye]
}

/// Contract-backed heatmap used by the live runtime. The public legacy API above remains
/// unchanged; both routes share the exact perturbation loop and arithmetic order.
pub(crate) fn compute_model(
    model: &mut dyn EyelidModel,
    input: &[f32],
    mode: HeatMode,
) -> Option<HeatResult> {
    compute_with(input, mode, |sample, eye| {
        model_openness(model, sample, eye)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::eyelid_model::{ModelError, RawEyelidPrediction, EYELID_INPUT_LEN};

    struct MeanModel;

    fn mean_for_eye(input: &[f32], eye: usize) -> f32 {
        let start = eye * DST * DST;
        input[start..start + DST * DST].iter().sum::<f32>() / (DST * DST) as f32
    }

    impl EyelidModel for MeanModel {
        fn infer(
            &mut self,
            input: CanonicalStereoInput<'_>,
        ) -> Result<RawEyelidPrediction, ModelError> {
            let input = input.as_slice();
            Ok(RawEyelidPrediction {
                presence: Some(1.0),
                openness: [Some(mean_for_eye(input, 0)), Some(mean_for_eye(input, 1))],
                squeeze: [None; 2],
            })
        }
    }

    #[test]
    fn positions_cover_the_grid() {
        let pos = positions();
        assert_eq!(*pos.last().unwrap(), DST - P, "last patch reaches the edge");
        let mut covered = vec![false; DST];
        for &c in &pos {
            for x in c..c + P {
                covered[x] = true;
            }
        }
        assert!(
            covered.iter().all(|&b| b),
            "every column is covered by some patch"
        );
    }

    #[test]
    fn mode_from_u8_maps() {
        assert_eq!(HeatMode::from_u8(0), HeatMode::OcclusionMean);
        assert_eq!(HeatMode::from_u8(1), HeatMode::GlintInject);
        assert_eq!(HeatMode::from_u8(2), HeatMode::BrightnessSensitivity);
        assert_eq!(HeatMode::from_u8(3), HeatMode::ContrastSensitivity);
        assert_eq!(HeatMode::from_u8(200), HeatMode::OcclusionMean);
    }

    #[test]
    fn high_resolution_grid_has_at_least_twenty_four_probes_per_axis() {
        assert!(positions().len() >= 24);
    }

    #[test]
    fn trait_heatmap_matches_the_same_direct_inference_seam() {
        let mut input = vec![0.0f32; EYELID_INPUT_LEN];
        for (index, value) in input.iter_mut().enumerate() {
            *value = (index % 251) as f32 / 250.0;
        }
        for mode in [HeatMode::OcclusionMean, HeatMode::GlintInject] {
            let direct =
                compute_with(&input, mode, |sample, eye| Some(mean_for_eye(sample, eye))).unwrap();
            let contract = compute_model(&mut MeanModel, &input, mode).unwrap();
            assert_eq!(contract.delta, direct.delta);
            assert_eq!(contract.base, direct.base);
            assert_eq!(contract.mode, direct.mode);
            assert_eq!(contract.p98_abs_delta, direct.p98_abs_delta);
            assert_eq!(contract.peak_abs_delta, direct.peak_abs_delta);
            assert_eq!(contract.baseline_openness, direct.baseline_openness);
            assert_eq!(contract.brightness_response, direct.brightness_response);
            assert_eq!(contract.contrast_response, direct.contrast_response);
        }
    }

    #[test]
    fn local_brightness_map_reports_positive_response_for_a_mean_model() {
        let input = vec![0.5f32; EYELID_INPUT_LEN];
        let result = compute_with(&input, HeatMode::BrightnessSensitivity, |sample, eye| {
            Some(mean_for_eye(sample, eye))
        })
        .unwrap();
        assert!(result.delta[0].iter().all(|value| *value > 0.0));
        assert!(result.delta[1].iter().all(|value| *value > 0.0));
    }
}
