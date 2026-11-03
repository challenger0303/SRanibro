
use std::io::Read;
use std::path::Path;

const CONV: [(usize, usize); 4] = [(1, 16), (16, 32), (32, 64), (64, 64)];
const FC1_IN: usize = 64 * 4 * 4;
const FC1_OUT: usize = 128;

pub struct BrowNet {
    out_dim: usize,
    conv_w: [Vec<f32>; 4],
    conv_b: [Vec<f32>; 4],
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: Vec<f32>,
    pad: Vec<f32>,
    col: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl BrowNet {
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

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
        let mut off = 12usize;
        let mut take = |n: usize| -> Result<Vec<f32>, String> {
            let end = off + n * 4;
            if end > bytes.len() {
                return Err(format!("brow weights truncated (need {end}, have {})", bytes.len()));
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
            return Err(format!("brow weights: {} trailing bytes", bytes.len() - off));
        }
        let finite = |v: &[f32]| v.iter().all(|x| x.is_finite());
        if !(conv_w.iter().all(|w| finite(w))
            && conv_b.iter().all(|b| finite(b))
            && finite(&fc1_w) && finite(&fc1_b) && finite(&fc2_w) && finite(&fc2_b))
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

    pub fn forward_one(&mut self, input64: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input64.len(), 64 * 64);
        let mut h = 64;
        let mut w = 64;
        conv_relu_pad(input64, 1, h, w, &self.conv_w[0], 16, &self.conv_b[0], &mut self.pad, &mut self.col, &mut self.a);
        let (mut hh, mut ww) = maxpool2(&self.a, 16, h, w, &mut self.b);
        h = hh; w = ww;
        for s in 1..4 {
            let (ic, oc) = CONV[s];
            conv_relu_pad(&self.b, ic, h, w, &self.conv_w[s], oc, &self.conv_b[s], &mut self.pad, &mut self.col, &mut self.a);
            let r = maxpool2(&self.a, oc, h, w, &mut self.b);
            hh = r.0; ww = r.1;
            h = hh; w = ww;
        }
        debug_assert_eq!(64 * h * w, FC1_IN);
        linear(&self.b, &self.fc1_w, &self.fc1_b, FC1_OUT, FC1_IN, true, &mut self.a);
        linear(&self.a, &self.fc2_w, &self.fc2_b, self.out_dim, FC1_OUT, false, &mut self.b);
        self.b[..self.out_dim].to_vec()
    }
}

fn ensure(v: &mut Vec<f32>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

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

#[allow(clippy::too_many_arguments)]
fn conv_relu_pad(
    inp: &[f32], ic: usize, ih: usize, iw: usize,
    w: &[f32], oc: usize, bias: &[f32],
    pad: &mut Vec<f32>, col: &mut Vec<f32>, out: &mut Vec<f32>,
) {
    pad1(inp, ic, ih, iw, pad);
    let (oh, ow) = im2col3(pad, ic, ih + 2, iw + 2, col);
    let kdim = ic * 9;
    let n = oh * ow;
    ensure(out, oc * n);
    unsafe {
        matrixmultiply::sgemm(
            oc, kdim, n,
            1.0,
            w.as_ptr(), kdim as isize, 1,
            col.as_ptr(), n as isize, 1,
            0.0,
            out.as_mut_ptr(), n as isize, 1,
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

fn linear(inp: &[f32], w: &[f32], bias: &[f32], out_dim: usize, in_dim: usize, relu: bool, out: &mut Vec<f32>) {
    ensure(out, out_dim);
    unsafe {
        matrixmultiply::sgemm(
            out_dim, in_dim, 1,
            1.0,
            w.as_ptr(), in_dim as isize, 1,
            inp.as_ptr(), 1, 1,
            0.0,
            out.as_mut_ptr(), 1, 1,
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

    #[test]
    fn pytorch_parity() {
        let dir = match std::env::var("BROW_FIXTURE_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let p = std::path::Path::new(&dir);
        let read_f32 = |name: &str| -> Vec<f32> {
            let b = std::fs::read(p.join(name)).expect("fixture file");
            b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        };
        let mut net = BrowNet::load(&p.join("brow.bin")).expect("load brow.bin");
        let input = read_f32("brow_in.f32");
        let expected = read_f32("brow_out.f32");
        assert_eq!(input.len(), 64 * 64, "fixture input must be 64x64");
        let got = net.forward_one(&input);
        assert_eq!(got.len(), expected.len(), "output dim mismatch");
        let max_diff = got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_diff < 1e-3, "Rust vs PyTorch max diff {max_diff} (got {got:?}, want {expected:?})");
        eprintln!("[brow_parity] max abs diff = {max_diff:e}  out={got:?}");
    }
}
