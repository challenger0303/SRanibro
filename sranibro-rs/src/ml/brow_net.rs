//! TinyBrowNet — pure-Rust forward pass for the eyebrow model ported from the
//! `vr_eyebrow` project. The eyebrow is NOT in the eye-camera FOV; this CNN infers
//! brow expression from eye-shape / upper-eyelid deformation, on a per-user calibrated
//! model. Input is 1x64x64 grayscale (per-frame z-scored); output is 1 or 3 continuous
//! brow values ([brow] or [brow, inner, outer]).
//!
//! Architecture (PyTorch `TinyBrowNet`, model.py):
//!   input [1,64,64]
//!   4x { Conv2d(3x3, pad 1) -> BatchNorm -> ReLU -> MaxPool2 }
//!       channels 1->16->32->64->64, spatial 64->32->16->8->4
//!   flatten [1024]
//!   fc1 Linear(1024,128) -> BatchNorm1d -> ReLU -> Dropout(inference no-op)
//!   fc2 Linear(128, OUT)  (linear, NOT relu — final unclamped output)
//!
//! BatchNorm is FOLDED into the preceding conv/linear at bake time (see
//! tools/bake_brow_weights.py), so this runs just conv+bias+relu / linear. Convs use
//! the same im2col+GEMM (matrixmultiply) approach as [`super::eye_net`], with an added
//! zero-pad-by-1 so the 3x3 convs keep their spatial size. Weights load from a flat
//! little-endian binary; nothing proprietary ships — the user bakes their own model.

use std::io::Read;
use std::path::Path;

/// Per-stage (in_ch, out_ch) for the 4 conv stages; kernel is 3x3 pad-1 throughout.
const CONV: [(usize, usize); 4] = [(1, 16), (16, 32), (32, 64), (64, 64)];
const FC1_IN: usize = 64 * 4 * 4; // 1024
const FC1_OUT: usize = 128;

pub struct BrowNet {
    out_dim: usize,
    conv_w: [Vec<f32>; 4],
    conv_b: [Vec<f32>; 4],
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    // Reusable scratch (steady-state zero-alloc): pad buffer, im2col, two ping-pongs.
    pad: Vec<f32>,
    col: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl BrowNet {
    /// Number of model outputs (1 = brow only, 3 = [brow, inner, outer]).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Load a baked weights file. Format (little-endian):
    ///   magic "BROWNET1" (8 bytes), u32 out_dim, then f32 arrays in order:
    ///   conv1_w[16,1,3,3] conv1_b[16] conv2_w[32,16,3,3] conv2_b[32]
    ///   conv3_w[64,32,3,3] conv3_b[64] conv4_w[64,64,3,3] conv4_b[64]
    ///   fc1_w[128,1024] fc1_b[128] fc2_w[out,128] fc2_b[out]
    /// (BatchNorm already folded into the conv/fc weights+biases.)
    pub fn load(path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(|e| format!("brow weights '{}': {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[..8] != b"BROWNET1" {
            return Err("not a BROWNET1 weights file".into());
        }
        let out_dim = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        if out_dim != 1 && out_dim != 3 {
            return Err(format!("brow out_dim {out_dim}, expected 1 or 3"));
        }
        // Cursor over the f32 payload.
        let mut off = 12usize;
        let mut take = |n: usize| -> Result<Vec<f32>, String> {
            let end = off + n * 4;
            if end > bytes.len() {
                return Err(format!(
                    "brow weights truncated (need {end}, have {})",
                    bytes.len()
                ));
            }
            let v: Vec<f32> = bytes[off..end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off = end;
            Ok(v)
        };
        let mut conv_w: [Vec<f32>; 4] = Default::default();
        let mut conv_b: [Vec<f32>; 4] = Default::default();
        for (i, (ic, oc)) in CONV.iter().enumerate() {
            conv_w[i] = take(oc * ic * 9)?;
            conv_b[i] = take(*oc)?;
        }
        let fc1_w = take(FC1_OUT * FC1_IN)?;
        let fc1_b = take(FC1_OUT)?;
        let fc2_w = take(out_dim * FC1_OUT)?;
        let fc2_b = take(out_dim)?;
        if off != bytes.len() {
            return Err(format!(
                "brow weights: {} trailing bytes",
                bytes.len() - off
            ));
        }
        // Reject NaN/Inf weights up front — a non-finite weight would silently poison
        // every inference (NaN survives clamp and reads as full brow-down downstream).
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !(conv_w.iter().all(|w| finite(w))
            && conv_b.iter().all(|b| finite(b))
            && finite(&fc1_w)
            && finite(&fc1_b)
            && finite(&fc2_w)
            && finite(&fc2_b))
        {
            return Err("brow weights contain NaN/Inf".into());
        }
        Ok(Self {
            out_dim,
            conv_w,
            conv_b,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            pad: Vec::new(),
            col: Vec::new(),
            a: Vec::new(),
            b: Vec::new(),
        })
    }

    /// Run the (frozen) conv+maxpool backbone on one 64x64 z-scored frame, leaving the
    /// 1024-d flatten in `self.b[..FC1_IN]` — the exact tensor that feeds fc1. Shared by
    /// [`Self::forward_one`] and [`Self::brow_features`] so inference and the per-user head
    /// trainer ([`crate::ml::brow_fit`]) see byte-identical features.
    fn conv_features(&mut self, input64: &[f32]) {
        debug_assert_eq!(input64.len(), 64 * 64);
        let mut h = 64;
        let mut w = 64;
        // Stage 1 reads `input64`; later stages ping-pong a <-> b. After each conv the
        // activation is in `a`; maxpool writes to `b`; next conv reads `b`.
        conv_relu_pad(
            input64,
            1,
            h,
            w,
            &self.conv_w[0],
            16,
            &self.conv_b[0],
            &mut self.pad,
            &mut self.col,
            &mut self.a,
        );
        let (mut hh, mut ww) = maxpool2(&self.a, 16, h, w, &mut self.b);
        h = hh;
        w = ww;
        for s in 1..4 {
            let (ic, oc) = CONV[s];
            conv_relu_pad(
                &self.b,
                ic,
                h,
                w,
                &self.conv_w[s],
                oc,
                &self.conv_b[s],
                &mut self.pad,
                &mut self.col,
                &mut self.a,
            );
            let r = maxpool2(&self.a, oc, h, w, &mut self.b);
            hh = r.0;
            ww = r.1;
            h = hh;
            w = ww;
        }
        debug_assert_eq!(64 * h * w, FC1_IN); // 64x4x4 = 1024, now in self.b
    }

    /// Forward one 64x64 grayscale frame (already z-scored), row-major [h][w].
    /// Returns `out_dim` brow values (raw, unclamped — the caller clips/post-processes).
    pub fn forward_one(&mut self, input64: &[f32]) -> Vec<f32> {
        self.conv_features(input64); // conv backbone -> 1024-d flatten in self.b
                                     // fc1+relu (b -> a), fc2 linear (a -> b).
        linear(
            &self.b,
            &self.fc1_w,
            &self.fc1_b,
            FC1_OUT,
            FC1_IN,
            true,
            &mut self.a,
        );
        linear(
            &self.a,
            &self.fc2_w,
            &self.fc2_b,
            self.out_dim,
            FC1_OUT,
            false,
            &mut self.b,
        );
        self.b[..self.out_dim].to_vec()
    }

    /// The 1024-d flatten (frozen conv-backbone output) for one 64x64 z-scored frame — the
    /// exact input fc1 consumes. This is what the per-user head trainer
    /// ([`crate::ml::brow_fit`]) caches and regresses on; running the SAME conv path as
    /// [`Self::forward_one`] keeps train-time and inference-time features identical.
    pub fn brow_features(&mut self, input64: &[f32]) -> Vec<f32> {
        self.conv_features(input64);
        self.b[..FC1_IN].to_vec()
    }

    /// Serialize this net's FROZEN conv backbone + a freshly-fitted `head` (fc1 + fc2) into
    /// the exact [`Self::from_bytes`] `BROWNET1` layout, ready to write as a `brow.bin`. The
    /// conv weights are reused byte-for-byte (the backbone is frozen); only the head is
    /// swapped in. Every head tensor's shape is checked against this net's dims and any
    /// non-finite value (conv OR head) is rejected, so a corrupt file is never emitted — the
    /// loader would refuse a NaN/Inf weight anyway, so we catch it at the source.
    pub fn to_bytes_with_head(
        &self,
        head: &crate::ml::brow_fit::HeadWeights,
    ) -> Result<Vec<u8>, String> {
        // Head shapes must match the frozen layout exactly (fc1 fan-in is FC1_IN=1024).
        if head.fc1_w.len() != FC1_OUT * FC1_IN {
            return Err(format!(
                "fc1_w len {} != {}",
                head.fc1_w.len(),
                FC1_OUT * FC1_IN
            ));
        }
        if head.fc1_b.len() != FC1_OUT {
            return Err(format!("fc1_b len {} != {}", head.fc1_b.len(), FC1_OUT));
        }
        if head.fc2_w.len() != self.out_dim * FC1_OUT {
            return Err(format!(
                "fc2_w len {} != {}",
                head.fc2_w.len(),
                self.out_dim * FC1_OUT
            ));
        }
        if head.fc2_b.len() != self.out_dim {
            return Err(format!(
                "fc2_b len {} != {}",
                head.fc2_b.len(),
                self.out_dim
            ));
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !(self.conv_w.iter().all(|w| finite(w))
            && self.conv_b.iter().all(|b| finite(b))
            && finite(&head.fc1_w)
            && finite(&head.fc1_b)
            && finite(&head.fc2_w)
            && finite(&head.fc2_b))
        {
            return Err("brow weights contain NaN/Inf".into());
        }
        // magic (8) + u32 out_dim + f32 payload in from_bytes order.
        let n_f32 = self.conv_w.iter().map(|w| w.len()).sum::<usize>()
            + self.conv_b.iter().map(|b| b.len()).sum::<usize>()
            + head.fc1_w.len()
            + head.fc1_b.len()
            + head.fc2_w.len()
            + head.fc2_b.len();
        let mut out = Vec::with_capacity(12 + n_f32 * 4);
        out.extend_from_slice(b"BROWNET1");
        out.extend_from_slice(&(self.out_dim as u32).to_le_bytes());
        for (w, b) in self.conv_w.iter().zip(self.conv_b.iter()) {
            push_f32_le(&mut out, w);
            push_f32_le(&mut out, b);
        }
        push_f32_le(&mut out, &head.fc1_w);
        push_f32_le(&mut out, &head.fc1_b);
        push_f32_le(&mut out, &head.fc2_w);
        push_f32_le(&mut out, &head.fc2_b);
        Ok(out)
    }
}

/// Append `v`'s f32 values to `out` as little-endian bytes (the `BROWNET1` payload encoding).
fn push_f32_le(out: &mut Vec<u8>, v: &[f32]) {
    for &x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn ensure(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

/// Zero-pad [ic][ih][iw] by 1 on all sides into `out` -> [ic][ih+2][iw+2].
fn pad1(inp: &[f32], ic: usize, ih: usize, iw: usize, out: &mut Vec<f32>) {
    let (ph, pw) = (ih + 2, iw + 2);
    ensure(out, ic * ph * pw);
    for v in out[..ic * ph * pw].iter_mut() {
        *v = 0.0;
    }
    for c in 0..ic {
        for y in 0..ih {
            let src = c * ih * iw + y * iw;
            let dst = c * ph * pw + (y + 1) * pw + 1;
            out[dst..dst + iw].copy_from_slice(&inp[src..src + iw]);
        }
    }
}

/// im2col (k=3 valid) into `col`; inp [ic][ih][iw]; col [ic*9, oh*ow] row-major.
fn im2col3(inp: &[f32], ic: usize, ih: usize, iw: usize, col: &mut Vec<f32>) -> (usize, usize) {
    let (k, oh, ow) = (3usize, ih - 2, iw - 2);
    let n = oh * ow;
    ensure(col, ic * 9 * n);
    for c in 0..ic {
        let i_c = c * ih * iw;
        for ky in 0..k {
            for kx in 0..k {
                let row = (c * 9 + ky * k + kx) * n;
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

/// Conv2d 3x3 stride-1 SAME (pad 1) + bias + ReLU. `w` is [oc, ic*9] row-major (PyTorch
/// conv weight [oc,ic,3,3] flattens to exactly this). Output [oc][ih][iw] in `out`.
#[allow(clippy::too_many_arguments)]
fn conv_relu_pad(
    inp: &[f32],
    ic: usize,
    ih: usize,
    iw: usize,
    w: &[f32],
    oc: usize,
    bias: &[f32],
    pad: &mut Vec<f32>,
    col: &mut Vec<f32>,
    out: &mut Vec<f32>,
) {
    pad1(inp, ic, ih, iw, pad);
    let (oh, ow) = im2col3(pad, ic, ih + 2, iw + 2, col); // -> (ih, iw)
    let kdim = ic * 9;
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
}

/// 2x2 max pool, stride 2, [c][h][w] -> [c][h/2][w/2] into `out`.
fn maxpool2(inp: &[f32], c: usize, h: usize, w: usize, out: &mut Vec<f32>) -> (usize, usize) {
    let (oh, ow) = (h / 2, w / 2);
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

/// y = W x + b, with optional ReLU. W is [out_dim, in_dim] row-major (PyTorch Linear).
fn linear(
    inp: &[f32],
    w: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
    relu: bool,
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
        out[o] = if relu && s < 0.0 { 0.0 } else { s };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_magic() {
        assert!(BrowNet::from_bytes(b"nope").is_err());
    }

    /// Deterministic finite f32s in [-0.5, 0.5) from a seed (splitmix-ish LCG) — for
    /// synthetic weights/heads without pulling in `rand`.
    fn det_f32s(n: usize, seed: u64) -> Vec<f32> {
        let mut st = seed | 1;
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (st >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            })
            .collect()
    }

    fn push_f32s(out: &mut Vec<u8>, v: &[f32]) {
        for &x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// A synthetic `BROWNET1` blob (magic + out_dim + conv backbone + head). Conv weights are
    /// scaled small so the untrained backbone's activations stay O(0.1) — that keeps the
    /// f32-vs-f64 head comparison in `to_bytes_with_head_roundtrips` well inside tolerance.
    fn synth_backbone_bytes(out_dim: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BROWNET1");
        bytes.extend_from_slice(&(out_dim as u32).to_le_bytes());
        let mut seed = 1u64;
        for (ic, oc) in CONV {
            let w: Vec<f32> = det_f32s(oc * ic * 9, seed)
                .iter()
                .map(|x| x * 0.05)
                .collect();
            push_f32s(&mut bytes, &w);
            seed += 1;
            let b: Vec<f32> = det_f32s(oc, seed).iter().map(|x| x * 0.05).collect();
            push_f32s(&mut bytes, &b);
            seed += 1;
        }
        push_f32s(&mut bytes, &det_f32s(FC1_OUT * FC1_IN, seed));
        seed += 1;
        push_f32s(&mut bytes, &det_f32s(FC1_OUT, seed));
        seed += 1;
        push_f32s(&mut bytes, &det_f32s(out_dim * FC1_OUT, seed));
        seed += 1;
        push_f32s(&mut bytes, &det_f32s(out_dim, seed));
        bytes
    }

    /// Writing a frozen backbone + a fresh head, then reloading, must (a) preserve the conv
    /// backbone byte-for-byte and (b) apply the head in the correct layout — so the reloaded
    /// net's forward equals the head applied to the base net's features.
    #[test]
    fn to_bytes_with_head_roundtrips() {
        let base_blob = synth_backbone_bytes(3);
        let mut base = BrowNet::from_bytes(&base_blob).expect("base loads");

        // A DIFFERENT head (deterministic, correct shapes).
        let new_head = crate::ml::brow_fit::HeadWeights {
            fc1_w: det_f32s(FC1_OUT * FC1_IN, 0xA1),
            fc1_b: det_f32s(FC1_OUT, 0xB2),
            fc2_w: det_f32s(3 * FC1_OUT, 0xC3),
            fc2_b: det_f32s(3, 0xD4),
        };

        let bytes = base.to_bytes_with_head(&new_head).unwrap();
        let mut reloaded = BrowNet::from_bytes(&bytes).unwrap();
        assert_eq!(reloaded.out_dim(), 3);

        // Sample input; features come from the (shared, frozen) conv backbone.
        let x: Vec<f32> = (0..64 * 64).map(|i| (i % 37) as f32 * 0.03 - 0.5).collect();
        let feats = base.brow_features(&x);
        assert_eq!(feats.len(), FC1_IN);

        // Apply new_head to feats by hand (fc1 row-major + ReLU, then fc2), f64 accumulation.
        let mut h = vec![0.0f64; FC1_OUT];
        for o in 0..FC1_OUT {
            let mut s = new_head.fc1_b[o] as f64;
            let base_i = o * FC1_IN;
            for i in 0..FC1_IN {
                s += new_head.fc1_w[base_i + i] as f64 * feats[i] as f64;
            }
            h[o] = if s > 0.0 { s } else { 0.0 };
        }
        let mut expected = vec![0.0f64; 3];
        for k in 0..3 {
            let mut s = new_head.fc2_b[k] as f64;
            let base_o = k * FC1_OUT;
            for o in 0..FC1_OUT {
                s += new_head.fc2_w[base_o + o] as f64 * h[o];
            }
            expected[k] = s;
        }

        let got = reloaded.forward_one(&x);
        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| (*g as f64 - *e).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_diff < 1e-4,
            "head/backbone roundtrip diff {max_diff:e} (got {got:?}, want {expected:?})"
        );

        // Frozen backbone: the conv byte-region must be byte-identical to the base blob.
        let conv_f32: usize = CONV.iter().map(|(ic, oc)| oc * ic * 9 + oc).sum();
        let conv_end = 12 + conv_f32 * 4;
        assert_eq!(
            &bytes[12..conv_end],
            &base_blob[12..conv_end],
            "frozen conv backbone bytes preserved"
        );
    }

    /// Numerical parity against PyTorch. Skipped unless the bake tool has written a
    /// fixture (set BROW_FIXTURE_DIR to the folder holding brow.bin + brow_in.f32 +
    /// brow_out.f32). Run after `python tools/bake_brow_weights.py ...`.
    #[test]
    fn pytorch_parity() {
        let dir = match std::env::var("BROW_FIXTURE_DIR") {
            Ok(d) => d,
            Err(_) => return, // no fixture -> skip (normal `cargo test` stays green)
        };
        let p = std::path::Path::new(&dir);
        let read_f32 = |name: &str| -> Vec<f32> {
            let b = std::fs::read(p.join(name)).expect("fixture file");
            b.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let mut net = BrowNet::load(&p.join("brow.bin")).expect("load brow.bin");
        let input = read_f32("brow_in.f32");
        let expected = read_f32("brow_out.f32");
        assert_eq!(input.len(), 64 * 64, "fixture input must be 64x64");
        let got = net.forward_one(&input);
        assert_eq!(got.len(), expected.len(), "output dim mismatch");
        let max_diff = got
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "Rust vs PyTorch max diff {max_diff} (got {got:?}, want {expected:?})"
        );
        eprintln!("[brow_parity] max abs diff = {max_diff:e}  out={got:?}");
    }
}
