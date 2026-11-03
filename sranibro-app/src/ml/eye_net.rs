
use std::collections::HashMap;

use super::tvm_params::Tensor;

pub const OUT_DIM: usize = 5;

pub struct EyeNet {
    conv1_w: Vec<f32>,
    conv1_b: Vec<f32>,
    conv2_w: Vec<f32>,
    conv2_b: Vec<f32>,
    conv3_w: Vec<f32>,
    conv3_b: Vec<f32>,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    col: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

pub(crate) struct EyeNetWeights<'a> {
    pub(crate) conv1_w: &'a [f32],
    pub(crate) conv1_b: &'a [f32],
    pub(crate) conv2_w: &'a [f32],
    pub(crate) conv2_b: &'a [f32],
    pub(crate) conv3_w: &'a [f32],
    pub(crate) conv3_b: &'a [f32],
    pub(crate) fc1_w: &'a [f32],
    pub(crate) fc1_b: &'a [f32],
    pub(crate) fc2_w: &'a [f32],
    pub(crate) fc2_b: &'a [f32],
}

const SPEC: &[(&str, &[i64])] = &[
    ("conv1_weight", &[20, 2, 5, 5]),
    ("conv1_bias", &[20]),
    ("conv2_weight", &[48, 20, 5, 5]),
    ("conv2_bias", &[48]),
    ("conv3_weight", &[64, 48, 3, 3]),
    ("conv3_bias", &[64]),
    ("fc5_ft_weight", &[500, 25600]),
    ("fc5_ft_bias", &[500]),
    ("fc6_10_weight", &[5, 500]),
    ("fc6_10_bias", &[5]),
];

impl EyeNet {
    pub fn new(map: HashMap<String, Tensor>) -> Result<Self, String> {
        for (name, want) in SPEC {
            let t = map
                .get(*name)
                .ok_or_else(|| format!("weight '{name}' missing — not an EyePrediction model?"))?;
            if t.shape != *want {
                return Err(format!(
                    "weight '{name}' shape {:?}, expected {want:?} — incompatible SRanipal model",
                    t.shape
                ));
            }
        }
        let take = |name: &str| map[name].data.clone();
        Ok(Self {
            conv1_w: take("conv1_weight"),
            conv1_b: take("conv1_bias"),
            conv2_w: take("conv2_weight"),
            conv2_b: take("conv2_bias"),
            conv3_w: take("conv3_weight"),
            conv3_b: take("conv3_bias"),
            fc1_w: take("fc5_ft_weight"),
            fc1_b: take("fc5_ft_bias"),
            fc2_w: take("fc6_10_weight"),
            fc2_b: take("fc6_10_bias"),
            col: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    }

    pub(crate) fn weights(&self) -> EyeNetWeights<'_> {
        EyeNetWeights {
            conv1_w: &self.conv1_w,
            conv1_b: &self.conv1_b,
            conv2_w: &self.conv2_w,
            conv2_b: &self.conv2_b,
            conv3_w: &self.conv3_w,
            conv3_b: &self.conv3_b,
            fc1_w: &self.fc1_w,
            fc1_b: &self.fc1_b,
            fc2_w: &self.fc2_w,
            fc2_b: &self.fc2_b,
        }
    }

    pub fn forward_one(&mut self, input: &[f32]) -> [f32; OUT_DIM] {
        let (h, w) = conv_relu(
            input,
            2,
            100,
            100,
            &self.conv1_w,
            20,
            5,
            &self.conv1_b,
            &mut self.col,
            &mut self.a,
        );
        let (h, w) = maxpool2(&self.a, 20, h, w, &mut self.b);

        let (h, w) = conv_relu(
            &self.b,
            20,
            h,
            w,
            &self.conv2_w,
            48,
            5,
            &self.conv2_b,
            &mut self.col,
            &mut self.a,
        );
        let (h, w) = maxpool2(&self.a, 48, h, w, &mut self.b);

        let _ = conv_relu(
            &self.b,
            48,
            h,
            w,
            &self.conv3_w,
            64,
            3,
            &self.conv3_b,
            &mut self.col,
            &mut self.a,
        );

        linear_relu(
            &self.a,
            &self.fc1_w,
            &self.fc1_b,
            500,
            64 * 20 * 20,
            &mut self.b,
        );
        linear_relu(&self.b, &self.fc2_w, &self.fc2_b, OUT_DIM, 500, &mut self.a);
        [self.a[0], self.a[1], self.a[2], self.a[3], self.a[4]]
    }
}

fn ensure(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

fn im2col(
    inp: &[f32],
    ic: usize,
    ih: usize,
    iw: usize,
    k: usize,
    col: &mut Vec<f32>,
) -> (usize, usize) {
    let oh = ih - k + 1;
    let ow = iw - k + 1;
    let kk = k * k;
    let n = oh * ow;
    ensure(col, ic * kk * n);
    for c in 0..ic {
        let i_c = c * ih * iw;
        for ky in 0..k {
            for kx in 0..k {
                let row = (c * kk + ky * k + kx) * n;
                for oy in 0..oh {
                    let src = i_c + (oy + ky) * iw + kx;
                    let dst = row + oy * ow;
                    col[dst..dst + ow].copy_from_slice(&inp[src..src + ow]);
                }
            }
        }
    }
    (oh, ow)
}

#[allow(clippy::too_many_arguments)]
fn conv_relu(
    inp: &[f32],
    ic: usize,
    ih: usize,
    iw: usize,
    w: &[f32],
    oc: usize,
    k: usize,
    bias: &[f32],
    col: &mut Vec<f32>,
    out: &mut Vec<f32>,
) -> (usize, usize) {
    let (oh, ow) = im2col(inp, ic, ih, iw, k, col);
    let kdim = ic * k * k;
    let n = oh * ow;
    ensure(out, oc * n);
    unsafe {
        matrixmultiply::sgemm(
            oc,
            kdim,
            n,
            1.0,
            w.as_ptr(),
            kdim as isize,
            1,
            col.as_ptr(),
            n as isize,
            1,
            0.0,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    for o in 0..oc {
        let b = bias[o];
        for v in out[o * n..(o + 1) * n].iter_mut() {
            let s = *v + b;
            *v = if s < 0.0 { 0.0 } else { s };
        }
    }
    (oh, ow)
}

fn maxpool2(inp: &[f32], c: usize, h: usize, w: usize, out: &mut Vec<f32>) -> (usize, usize) {
    let oh = h / 2;
    let ow = w / 2;
    ensure(out, c * oh * ow);
    for ch in 0..c {
        let base = ch * h * w;
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f32::MIN;
                for dy in 0..2 {
                    let row = base + (oy * 2 + dy) * w + ox * 2;
                    m = m.max(inp[row]).max(inp[row + 1]);
                }
                out[(ch * oh + oy) * ow + ox] = m;
            }
        }
    }
    (oh, ow)
}

fn linear_relu(
    inp: &[f32],
    w: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
    out: &mut Vec<f32>,
) {
    ensure(out, out_dim);
    unsafe {
        matrixmultiply::sgemm(
            out_dim,
            in_dim,
            1,
            1.0,
            w.as_ptr(),
            in_dim as isize,
            1,
            inp.as_ptr(),
            1,
            1,
            0.0,
            out.as_mut_ptr(),
            1,
            1,
        );
    }
    for o in 0..out_dim {
        let s = out[o] + bias[o];
        out[o] = if s < 0.0 { 0.0 } else { s };
    }
}
