//! Occlusion-sensitivity heatmaps for the eye model — a diagnostic that shows how the
//! openness output RESPONDS to each region of the eye image (e.g. the bright specular
//! dots glasses / IR-LEDs throw onto the IR frame, which degrade tracking when worn).
//!
//! The net is forward-pass only (no autograd), so instead of a gradient we PERTURB a
//! patch of the input and watch the openness output move — a causal "damage here changes
//! the prediction this much" map, which is exactly the question "how does the model react
//! to this dot?". Two modes: erase-to-mean (what the model relies on) and glint-inject
//! (paint a fake specular reflection and see where it breaks openness). ~196 forward
//! passes per eye (~1s), run once on demand — not a live per-frame overlay.

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
}

impl HeatMode {
    pub fn from_u8(v: u8) -> Self {
        if v == 1 {
            HeatMode::GlintInject
        } else {
            HeatMode::OcclusionMean
        }
    }
}

/// One computed heatmap pair (both eyes) plus the grayscale model input it was scored
/// over, so the UI can composite the overlay 1:1 with no resampling offset.
pub struct HeatResult {
    /// Signed openness delta per pixel (`DST*DST` each). `[left, right]`.
    pub delta: [Vec<f32>; 2],
    /// The grayscale model input (`DST*DST` u8) each eye was scored on. `[left, right]`.
    pub base: [Vec<u8>; 2],
    pub mode: HeatMode,
}

// Patch = 14 px on the 100x100 grid (~ a real glint after the 200->100 downscale, and
// wider than conv1's 5x5 receptive field so an occluded patch truly denies signal).
// Stride 7 = 50% overlap -> the averaged map is smooth, not blocky. ~14x14 positions.
const P: usize = 14;
const S: usize = 7;

/// Top-left patch positions along one axis: 0,7,…,DST-P, with the last clamped to the
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
                    work[idx] = fill;
                }
            }
            let o = infer_openness(&work)?;
            // OcclusionMean: base - o  (positive = erasing here LOWERED openness = relied on it).
            // GlintInject:   o - base  (signed shift a fake glint here induces in openness).
            let delta = match mode {
                HeatMode::OcclusionMean => base - o,
                HeatMode::GlintInject => o - base,
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
    Some(HeatResult {
        delta: [dl, dr],
        base: [gray(0), gray(1)],
        mode,
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
        assert_eq!(HeatMode::from_u8(200), HeatMode::OcclusionMean);
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
        }
    }
}
