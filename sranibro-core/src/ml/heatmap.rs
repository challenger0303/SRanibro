
use super::eye_net::EyeNet;
use super::preprocess::DST;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HeatMode {
    OcclusionMean,
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

pub struct HeatResult {
    pub delta: [Vec<f32>; 2],
    pub base: [Vec<u8>; 2],
    pub mode: HeatMode,
}

const P: usize = 14;
const S: usize = 7;

fn positions() -> Vec<usize> {
    let mut v: Vec<usize> = (0..=DST - P).step_by(S).collect();
    if v.last() != Some(&(DST - P)) {
        v.push(DST - P);
    }
    v
}

pub fn occlusion_map(
    net: &mut EyeNet,
    input: &[f32],
    eye_ch: usize,
    out_idx: usize,
    mode: HeatMode,
) -> Vec<f32> {
    let n = DST;
    let off = eye_ch * n * n;
    let base = net.forward_one(input)[out_idx];
    let fill = match mode {
        HeatMode::OcclusionMean => {
            let sum: f32 = input[off..off + n * n].iter().sum();
            sum / (n * n) as f32
        }
        HeatMode::GlintInject => 1.0,
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
            let o = net.forward_one(&work)[out_idx];
            let delta = match mode {
                HeatMode::OcclusionMean => base - o,
                HeatMode::GlintInject => o - base,
            };
            for y in 0..P {
                for x in 0..P {
                    let k = (cy + y) * n + (cx + x);
                    heat[k] += delta;
                    cover[k] += 1.0;
                    work[off + k] = saved[y * P + x];
                }
            }
        }
    }
    for k in 0..n * n {
        if cover[k] > 0.0 {
            heat[k] /= cover[k];
        }
    }
    heat
}

pub fn compute(net: &mut EyeNet, input: &[f32], mode: HeatMode) -> HeatResult {
    let n = DST;
    let gray = |ch: usize| -> Vec<u8> {
        input[ch * n * n..(ch + 1) * n * n]
            .iter()
            .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
            .collect()
    };
    let dl = occlusion_map(net, input, 0, 1, mode);
    let dr = occlusion_map(net, input, 1, 2, mode);
    HeatResult { delta: [dl, dr], base: [gray(0), gray(1)], mode }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(covered.iter().all(|&b| b), "every column is covered by some patch");
    }

    #[test]
    fn mode_from_u8_maps() {
        assert_eq!(HeatMode::from_u8(0), HeatMode::OcclusionMean);
        assert_eq!(HeatMode::from_u8(1), HeatMode::GlintInject);
        assert_eq!(HeatMode::from_u8(200), HeatMode::OcclusionMean);
    }
}
