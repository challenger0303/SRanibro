//! Per-user TinyBrowNet HEAD trainer — pure Rust, no deps, no Python.
//!
//! The eventual per-user calibration re-fits [`crate::ml::brow_net::BrowNet`] to one
//! wearer. This first increment trains ONLY the head (fc1 + ReLU + fc2) on CACHED 1024-d
//! features from the FROZEN conv backbone ([`crate::ml::brow_net::BrowNet::brow_features`]),
//! which is exactly the top of the full backprop — so nothing here is throwaway. It is a
//! plain MSE regression driven by Adam over shuffled mini-batches with early stopping.
//!
//! Layout is bit-for-bit the `BROWNET1` head layout so the exported weights can later be
//! concatenated with the frozen conv weights and written as a `brow.bin`:
//!   * `fc1_w` is `[128, in_dim]` row-major (index `o*in_dim + i`), `fc1_b[128]`,
//!   * `fc2_w` is `[out, 128]` row-major (index `k*128 + o`),   `fc2_b[out]`.
//! (`in_dim` is 1024 for the real backbone; the tiny-net gradient-check uses smaller dims.)
//!
//! All training arithmetic is `f64` for a tight gradient-check and stable accumulation; the
//! exported [`HeadWeights`] are `f32` (what the runtime net and `brow.bin` use). This is a
//! once-per-user fit, so the inner loops are plain nested loops — correctness over speed;
//! we deliberately do NOT reach for sgemm/BLAS here (it would only complicate the check).
//!
//! There is NO data loading, PNG decoding, or `brow.bin` writing here — that is the next
//! increment. This module is the tested library core: `fit_head` + the gradient-check.

/// Hidden width of the brow head (fc1 out / fc2 in); fixed by the `BROWNET1` layout.
const HIDDEN: usize = 128;
/// Feature dim fed to fc1 (the conv flatten). Only a fallback for the empty-input guard;
/// the real dim is read from the samples.
const FEAT_DIM: usize = 64 * 4 * 4; // 1024

// Adam + schedule constants (a once-per-user fit; sane fixed values, no tuning surface).
const LR: f64 = 1e-3;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
/// Decoupled (AdamW-style) weight decay, applied to the WEIGHT tensors only so the
/// gradient-check can compare against the bare MSE gradient.
const WEIGHT_DECAY: f64 = 1e-5;
const BATCH: usize = 32;
const MAX_EPOCHS: usize = 60;
/// Stop after this many epochs with no val-MSE improvement (best weights are kept).
const PATIENCE: usize = 5;
/// Fraction of samples held out for early-stopping validation.
const VAL_FRAC: f64 = 0.15;

/// The trained head weights, in the exact `BROWNET1` head layout (see module docs). `f32`
/// so they drop straight into the runtime net / `brow.bin` writer (the next increment).
#[derive(Clone)]
pub struct HeadWeights {
    /// fc1 weight `[HIDDEN, in_dim]` row-major, index `o*in_dim + i`.
    pub fc1_w: Vec<f32>,
    /// fc1 bias `[HIDDEN]`.
    pub fc1_b: Vec<f32>,
    /// fc2 weight `[out, HIDDEN]` row-major, index `k*HIDDEN + o`.
    pub fc2_w: Vec<f32>,
    /// fc2 bias `[out]`.
    pub fc2_b: Vec<f32>,
}

impl HeadWeights {
    /// Zeroed head of the right shape — the degenerate return when there is nothing to fit.
    fn zeros(in_dim: usize, out_dim: usize) -> Self {
        HeadWeights {
            fc1_w: vec![0.0; HIDDEN * in_dim],
            fc1_b: vec![0.0; HIDDEN],
            fc2_w: vec![0.0; out_dim * HIDDEN],
            fc2_b: vec![0.0; out_dim],
        }
    }
}

/// Tiny seeded PRNG (splitmix64) — deterministic init/shuffle without pulling in `rand`.
/// splitmix64 has no bad-seed / zero-state trap, so any `seed` is fine.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform f64 in `[0, 1)` from the top 53 bits (full mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in `[-b, b)`.
    fn uniform(&mut self, b: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * b
    }

    /// Uniform integer in `[0, n)` (n > 0).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Fisher-Yates in-place shuffle of `idx` using `rng`.
fn shuffle(rng: &mut Rng, idx: &mut [usize]) {
    for i in (1..idx.len()).rev() {
        let j = rng.below(i + 1);
        idx.swap(i, j);
    }
}

/// The head parameters + shapes, held in `f64` during training.
#[derive(Clone)]
struct Head {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    fc1_w: Vec<f64>, // [hidden, in_dim], index o*in_dim + i
    fc1_b: Vec<f64>, // [hidden]
    fc2_w: Vec<f64>, // [out_dim, hidden], index k*hidden + o
    fc2_b: Vec<f64>, // [out_dim]
}

impl Head {
    /// PyTorch `nn.Linear`-default init (Kaiming-uniform, a=sqrt(5), which reduces to
    /// `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`) for both weights and biases.
    fn init(in_dim: usize, hidden: usize, out_dim: usize, rng: &mut Rng) -> Head {
        let b1 = 1.0 / (in_dim as f64).sqrt(); // fc1 fan_in = in_dim
        let b2 = 1.0 / (hidden as f64).sqrt(); // fc2 fan_in = hidden
        let fill = |n: usize, bound: f64, rng: &mut Rng| -> Vec<f64> {
            (0..n).map(|_| rng.uniform(bound)).collect()
        };
        Head {
            in_dim,
            hidden,
            out_dim,
            fc1_w: fill(hidden * in_dim, b1, rng),
            fc1_b: fill(hidden, b1, rng),
            fc2_w: fill(out_dim * hidden, b2, rng),
            fc2_b: fill(out_dim, b2, rng),
        }
    }

    /// Forward one sample `x[in_dim]`, returning `(pre1[hidden], h[hidden], y[out_dim])`
    /// where `pre1` is the fc1 pre-activation, `h = relu(pre1)`, and `y` is the linear fc2
    /// output. `pre1` is returned because backward needs its sign for the ReLU derivative.
    fn forward(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut pre1 = vec![0.0f64; self.hidden];
        let mut h = vec![0.0f64; self.hidden];
        for o in 0..self.hidden {
            let base = o * self.in_dim;
            let mut s = self.fc1_b[o];
            for i in 0..self.in_dim {
                s += self.fc1_w[base + i] * x[i];
            }
            pre1[o] = s;
            h[o] = if s > 0.0 { s } else { 0.0 };
        }
        let mut y = vec![0.0f64; self.out_dim];
        for k in 0..self.out_dim {
            let base = k * self.hidden;
            let mut s = self.fc2_b[k];
            for o in 0..self.hidden {
                s += self.fc2_w[base + o] * h[o];
            }
            y[k] = s;
        }
        (pre1, h, y)
    }

    /// fc2 output only (used by the loss-only path).
    fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.forward(x).2
    }

    /// Export to the `f32` `BROWNET1` head layout (indices already match — a straight cast).
    fn to_weights(&self) -> HeadWeights {
        let cast = |v: &[f64]| v.iter().map(|&x| x as f32).collect();
        HeadWeights {
            fc1_w: cast(&self.fc1_w),
            fc1_b: cast(&self.fc1_b),
            fc2_w: cast(&self.fc2_w),
            fc2_b: cast(&self.fc2_b),
        }
    }
}

/// Gradient accumulator mirroring [`Head`]'s parameter shapes.
#[derive(Clone)]
struct Grads {
    fc1_w: Vec<f64>,
    fc1_b: Vec<f64>,
    fc2_w: Vec<f64>,
    fc2_b: Vec<f64>,
}

impl Grads {
    fn zeros(in_dim: usize, hidden: usize, out_dim: usize) -> Grads {
        Grads {
            fc1_w: vec![0.0; hidden * in_dim],
            fc1_b: vec![0.0; hidden],
            fc2_w: vec![0.0; out_dim * hidden],
            fc2_b: vec![0.0; out_dim],
        }
    }
}

/// Mean MSE over the sample subset `idx` (loss = mean over the batch of `sum_k (y-t)^2`).
fn batch_loss(head: &Head, xs: &[Vec<f64>], ts: &[Vec<f64>], idx: &[usize]) -> f64 {
    let mut loss = 0.0f64;
    for &s in idx {
        let y = head.predict(&xs[s]);
        for k in 0..head.out_dim {
            let e = y[k] - ts[s][k];
            loss += e * e;
        }
    }
    loss / idx.len().max(1) as f64
}

/// Mean MSE and its gradient over the sample subset `idx`. Pure MSE — weight decay lives in
/// the Adam step so this is exactly the loss the gradient-check finite-differences.
///
/// Per sample: `dy[k] = 2*(y[k]-t[k])`; `dfc2_w[k*H+o] += dy[k]*h[o]`; `dfc2_b[k] += dy[k]`;
/// `dh[o] = sum_k fc2_w[k*H+o]*dy[k]`; `dpre1[o] = dh[o]*(pre1[o]>0)`;
/// `dfc1_w[o*I+i] += dpre1[o]*x[i]`; `dfc1_b[o] += dpre1[o]`. Accumulate, then divide by
/// the batch size (matching the `1/batch` in the mean loss).
fn batch_loss_grad(head: &Head, xs: &[Vec<f64>], ts: &[Vec<f64>], idx: &[usize]) -> (f64, Grads) {
    let mut g = Grads::zeros(head.in_dim, head.hidden, head.out_dim);
    let mut loss = 0.0f64;
    for &s in idx {
        let x = &xs[s];
        let t = &ts[s];
        let (pre1, h, y) = head.forward(x);
        // Output error + its gradient dL/dy.
        let mut dy = vec![0.0f64; head.out_dim];
        for k in 0..head.out_dim {
            let e = y[k] - t[k];
            loss += e * e;
            dy[k] = 2.0 * e;
        }
        // fc2 grads, and backprop into the hidden activation (dh).
        let mut dh = vec![0.0f64; head.hidden];
        for k in 0..head.out_dim {
            let base = k * head.hidden;
            g.fc2_b[k] += dy[k];
            for o in 0..head.hidden {
                g.fc2_w[base + o] += dy[k] * h[o];
                dh[o] += head.fc2_w[base + o] * dy[k];
            }
        }
        // Through ReLU (dead unit -> zero grad), then fc1 grads.
        for o in 0..head.hidden {
            let dpre = if pre1[o] > 0.0 { dh[o] } else { 0.0 };
            g.fc1_b[o] += dpre;
            let base = o * head.in_dim;
            for i in 0..head.in_dim {
                g.fc1_w[base + i] += dpre * x[i];
            }
        }
    }
    // Mean over the batch (both the loss and every gradient).
    let inv = 1.0 / idx.len().max(1) as f64;
    loss *= inv;
    for v in g.fc1_w.iter_mut() {
        *v *= inv;
    }
    for v in g.fc1_b.iter_mut() {
        *v *= inv;
    }
    for v in g.fc2_w.iter_mut() {
        *v *= inv;
    }
    for v in g.fc2_b.iter_mut() {
        *v *= inv;
    }
    (loss, g)
}

/// Adam moment state, one `(m, v)` pair per parameter tensor + a global step counter.
struct Adam {
    t: i32,
    m: Grads,
    v: Grads,
}

/// One Adam update over a flat tensor (with optional decoupled weight decay). `bc1`/`bc2`
/// are the bias-correction denominators `1 - beta^t`, precomputed once per step.
#[allow(clippy::too_many_arguments)]
fn adam_update(
    p: &mut [f64],
    g: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    bc1: f64,
    bc2: f64,
    decay: bool,
) {
    for i in 0..p.len() {
        let gi = g[i];
        m[i] = BETA1 * m[i] + (1.0 - BETA1) * gi;
        v[i] = BETA2 * v[i] + (1.0 - BETA2) * gi * gi;
        let mhat = m[i] / bc1;
        let vhat = v[i] / bc2;
        p[i] -= LR * mhat / (vhat.sqrt() + EPS);
        if decay {
            // Decoupled (AdamW) weight decay — applied to weights only, keeping the MSE
            // gradient (and hence the gradient-check) untouched.
            p[i] -= LR * WEIGHT_DECAY * p[i];
        }
    }
}

impl Adam {
    fn new(in_dim: usize, hidden: usize, out_dim: usize) -> Adam {
        Adam {
            t: 0,
            m: Grads::zeros(in_dim, hidden, out_dim),
            v: Grads::zeros(in_dim, hidden, out_dim),
        }
    }

    fn step(&mut self, head: &mut Head, g: &Grads) {
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(self.t);
        let bc2 = 1.0 - BETA2.powi(self.t);
        adam_update(
            &mut head.fc1_w,
            &g.fc1_w,
            &mut self.m.fc1_w,
            &mut self.v.fc1_w,
            bc1,
            bc2,
            true,
        );
        adam_update(
            &mut head.fc1_b,
            &g.fc1_b,
            &mut self.m.fc1_b,
            &mut self.v.fc1_b,
            bc1,
            bc2,
            false,
        );
        adam_update(
            &mut head.fc2_w,
            &g.fc2_w,
            &mut self.m.fc2_w,
            &mut self.v.fc2_w,
            bc1,
            bc2,
            true,
        );
        adam_update(
            &mut head.fc2_b,
            &g.fc2_b,
            &mut self.m.fc2_b,
            &mut self.v.fc2_b,
            bc1,
            bc2,
            false,
        );
    }
}

/// Fit the brow head (fc1 + ReLU + fc2) by MSE regression + Adam on cached 1024-d
/// `features` -> `out_dim` `labels`. `seed` makes init + shuffling deterministic.
///
/// Trains on an 85/15 train/val split (shuffled batches of [`BATCH`]), early-stopping on
/// val MSE (patience [`PATIENCE`], up to [`MAX_EPOCHS`]) and returning the BEST-val weights.
/// Degenerate inputs (no samples / zero dims) return a zeroed head of the right shape.
pub fn fit_head(
    features: &[Vec<f32>],
    labels: &[Vec<f32>],
    out_dim: usize,
    seed: u64,
) -> HeadWeights {
    let n = features.len();
    let in_dim = features.first().map(|f| f.len()).unwrap_or(FEAT_DIM);
    debug_assert!(labels.len() == n, "features/labels count mismatch");
    debug_assert!(
        labels.first().map(|l| l.len()).unwrap_or(out_dim) == out_dim,
        "label dim != out_dim"
    );
    if n == 0 || out_dim == 0 || in_dim == 0 {
        return HeadWeights::zeros(in_dim, out_dim);
    }

    // Promote to f64 once (stable accumulation; the gradient-check needs f64 anyway).
    let xs: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.iter().map(|&v| v as f64).collect())
        .collect();
    let ts: Vec<Vec<f64>> = labels
        .iter()
        .map(|l| l.iter().map(|&v| v as f64).collect())
        .collect();

    // Train/val split: shuffle once, hold out the first VAL_FRAC as validation. With <2
    // samples we can't split, so validate on the (whole) training set instead.
    let mut rng = Rng::new(seed);
    let mut order: Vec<usize> = (0..n).collect();
    shuffle(&mut rng, &mut order);
    let n_val = if n < 2 {
        0
    } else {
        ((n as f64 * VAL_FRAC).round() as usize).clamp(1, n - 1)
    };
    let val_idx: Vec<usize> = order[..n_val].to_vec();
    let train_idx: Vec<usize> = order[n_val..].to_vec();
    let eval_idx: &[usize] = if val_idx.is_empty() {
        &train_idx
    } else {
        &val_idx
    };

    fit_indices(&xs, &ts, &train_idx, eval_idx, out_dim, seed).to_weights()
}

/// Result of a head fit with an explicit validation set. Wide calibration uses this
/// instead of a random adjacent-frame split so a whole later capture session can be
/// held out, which is a much stronger check of headset reseating/generalization.
pub struct FitHeadReport {
    pub weights: HeadWeights,
    pub train_mse: f64,
    pub val_mse: f64,
}

/// Fit the shared TinyEyeNet head with caller-provided train and validation sessions.
/// The validation samples are never optimized against. If `val_features` is empty the
/// training set is also used for early stopping, matching [`fit_head`]'s tiny-data guard.
pub fn fit_head_with_validation(
    train_features: &[Vec<f32>],
    train_labels: &[Vec<f32>],
    val_features: &[Vec<f32>],
    val_labels: &[Vec<f32>],
    out_dim: usize,
    seed: u64,
) -> FitHeadReport {
    let in_dim = train_features.first().map(|f| f.len()).unwrap_or(FEAT_DIM);
    if train_features.is_empty() || out_dim == 0 || in_dim == 0 {
        return FitHeadReport {
            weights: HeadWeights::zeros(in_dim, out_dim),
            train_mse: f64::INFINITY,
            val_mse: f64::INFINITY,
        };
    }
    assert_eq!(train_features.len(), train_labels.len());
    assert_eq!(val_features.len(), val_labels.len());
    let xs: Vec<Vec<f64>> = train_features
        .iter()
        .chain(val_features.iter())
        .map(|f| f.iter().map(|&v| v as f64).collect())
        .collect();
    let ts: Vec<Vec<f64>> = train_labels
        .iter()
        .chain(val_labels.iter())
        .map(|l| l.iter().map(|&v| v as f64).collect())
        .collect();
    // Keep the bindings mutable only long enough for shape assertions below; the
    // optimizer receives immutable slices.
    debug_assert!(xs.iter().all(|x| x.len() == in_dim));
    debug_assert!(ts.iter().all(|t| t.len() == out_dim));
    let train_idx: Vec<usize> = (0..train_features.len()).collect();
    let val_idx: Vec<usize> = if val_features.is_empty() {
        train_idx.clone()
    } else {
        (train_features.len()..xs.len()).collect()
    };
    let best = fit_indices(&xs, &ts, &train_idx, &val_idx, out_dim, seed);
    let train_mse = batch_loss(&best, &xs, &ts, &train_idx);
    let val_mse = batch_loss(&best, &xs, &ts, &val_idx);
    FitHeadReport {
        weights: best.to_weights(),
        train_mse,
        val_mse,
    }
}

fn fit_indices(
    xs: &[Vec<f64>],
    ts: &[Vec<f64>],
    train_idx: &[usize],
    eval_idx: &[usize],
    out_dim: usize,
    seed: u64,
) -> Head {
    let in_dim = xs.first().map(|f| f.len()).unwrap_or(FEAT_DIM);
    let mut rng = Rng::new(seed);
    let mut head = Head::init(in_dim, HIDDEN, out_dim, &mut rng);
    let mut adam = Adam::new(in_dim, HIDDEN, out_dim);

    // Keep the best-val snapshot (early stopping returns it, not the last epoch's weights).
    let mut best = head.clone();
    let mut best_val = batch_loss(&head, &xs, &ts, eval_idx);
    let mut since_improve = 0usize;
    let mut batch_order = train_idx.to_vec();
    for _epoch in 0..MAX_EPOCHS {
        shuffle(&mut rng, &mut batch_order);
        for chunk in batch_order.chunks(BATCH) {
            let (_l, g) = batch_loss_grad(&head, &xs, &ts, chunk);
            adam.step(&mut head, &g);
        }
        let val = batch_loss(&head, &xs, &ts, eval_idx);
        if val < best_val - 1e-9 {
            best_val = val;
            best = head.clone();
            since_improve = 0;
        } else {
            since_improve += 1;
            if since_improve >= PATIENCE {
                break;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Head forward from exported `f32` [`HeadWeights`] (mirrors the runtime layout), used by
    /// the convergence test to measure the fitted head's MSE. Accumulates in f64.
    fn hw_predict(hw: &HeadWeights, x: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = x.len();
        let mut h = vec![0.0f64; HIDDEN];
        for o in 0..HIDDEN {
            let base = o * in_dim;
            let mut s = hw.fc1_b[o] as f64;
            for i in 0..in_dim {
                s += hw.fc1_w[base + i] as f64 * x[i] as f64;
            }
            h[o] = if s > 0.0 { s } else { 0.0 };
        }
        let mut y = vec![0.0f32; out_dim];
        for k in 0..out_dim {
            let base = k * HIDDEN;
            let mut s = hw.fc2_b[k] as f64;
            for o in 0..HIDDEN {
                s += hw.fc2_w[base + o] as f64 * h[o];
            }
            y[k] = s as f32;
        }
        y
    }

    /// THE correctness gate: analytic backward vs central finite differences of the loss.
    ///
    /// Tiny net (6->4->2) with seeded random weights + a few random (input, label) samples.
    /// For every weight and bias of both layers, the analytic gradient must match the
    /// central difference `(L(+eps) - L(-eps)) / (2*eps)` to a max relative error < 1e-5.
    #[test]
    fn head_gradient_check() {
        let (in_dim, hidden, out_dim) = (6usize, 4usize, 2usize);
        let mut rng = Rng::new(0xC0FF_EE12_3456_789A);
        let head = Head::init(in_dim, hidden, out_dim, &mut rng);
        // A few random samples; inputs at O(1) scale so units straddle the ReLU kink.
        let mut xs: Vec<Vec<f64>> = Vec::new();
        let mut ts: Vec<Vec<f64>> = Vec::new();
        for _ in 0..3 {
            xs.push((0..in_dim).map(|_| rng.uniform(1.5)).collect());
            ts.push((0..out_dim).map(|_| rng.uniform(1.0)).collect());
        }
        let idx: Vec<usize> = (0..xs.len()).collect();
        let (_loss, g) = batch_loss_grad(&head, &xs, &ts, &idx);

        let eps = 1e-5f64;
        let mut max_rel = 0.0f64;
        // Central-difference each element of a tensor and compare to its analytic grad.
        macro_rules! check_tensor {
            ($field:ident) => {{
                for e in 0..head.$field.len() {
                    let orig = head.$field[e];
                    let mut hp = head.clone();
                    let mut hm = head.clone();
                    hp.$field[e] = orig + eps;
                    hm.$field[e] = orig - eps;
                    let lp = batch_loss(&hp, &xs, &ts, &idx);
                    let lm = batch_loss(&hm, &xs, &ts, &idx);
                    let num = (lp - lm) / (2.0 * eps);
                    let ana = g.$field[e];
                    let rel = (ana - num).abs() / ana.abs().max(num.abs()).max(1e-8);
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
            }};
        }
        check_tensor!(fc1_w);
        check_tensor!(fc1_b);
        check_tensor!(fc2_w);
        check_tensor!(fc2_b);

        eprintln!("[brow_fit] gradient-check max relative error = {max_rel:e}");
        assert!(
            max_rel < 1e-5,
            "analytic vs finite-diff gradient mismatch: max rel err {max_rel:e}"
        );
    }

    /// Convergence smoke test: fitting a random LINEAR target (which the head easily
    /// represents) must drive MSE far below the zero-prediction baseline.
    #[test]
    fn head_converges_on_linear_data() {
        let (in_dim, out_dim, n) = (8usize, 2usize, 240usize);
        let mut rng = Rng::new(0xABCD_1234_5678_9012);
        // Random affine target t = A x + c.
        let a: Vec<f64> = (0..out_dim * in_dim).map(|_| rng.uniform(1.0)).collect();
        let c: Vec<f64> = (0..out_dim).map(|_| rng.uniform(0.5)).collect();
        let mut feats: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut labels: Vec<Vec<f32>> = Vec::with_capacity(n);
        for _ in 0..n {
            let x: Vec<f64> = (0..in_dim).map(|_| rng.uniform(1.0)).collect();
            let mut y = c.clone();
            for k in 0..out_dim {
                for i in 0..in_dim {
                    y[k] += a[k * in_dim + i] * x[i];
                }
            }
            feats.push(x.iter().map(|&v| v as f32).collect());
            labels.push(y.iter().map(|&v| v as f32).collect());
        }
        // Baseline: predict zero (mean sum-of-squares of the labels).
        let before: f64 = labels
            .iter()
            .map(|l| l.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>())
            .sum::<f64>()
            / n as f64;

        let hw = fit_head(&feats, &labels, out_dim, 42);

        // After: MSE of the fitted head over the whole set.
        let mut after = 0.0f64;
        for s in 0..n {
            let y = hw_predict(&hw, &feats[s], out_dim);
            for k in 0..out_dim {
                let e = y[k] as f64 - labels[s][k] as f64;
                after += e * e;
            }
        }
        after /= n as f64;

        eprintln!("[brow_fit] convergence: before(zero-pred) MSE = {before:.5}, after(fit) MSE = {after:.5}");
        assert!(before > 1e-3, "degenerate baseline — labels ~0");
        assert!(
            after < before * 0.2,
            "fit did not markedly reduce MSE: {before:.5} -> {after:.5}"
        );
    }

    /// Exported weights are the exact `BROWNET1` head shape (so the next increment can
    /// concatenate them with the frozen conv weights and write `brow.bin`).
    #[test]
    fn fit_head_output_shapes() {
        let in_dim = 1024usize;
        let out_dim = 3usize;
        let feats: Vec<Vec<f32>> = (0..40)
            .map(|s| (0..in_dim).map(|i| ((s + i) % 7) as f32 * 0.1).collect())
            .collect();
        let labels: Vec<Vec<f32>> = (0..40)
            .map(|s| (0..out_dim).map(|k| ((s + k) % 3) as f32 * 0.2).collect())
            .collect();
        let hw = fit_head(&feats, &labels, out_dim, 7);
        assert_eq!(hw.fc1_w.len(), HIDDEN * in_dim);
        assert_eq!(hw.fc1_b.len(), HIDDEN);
        assert_eq!(hw.fc2_w.len(), out_dim * HIDDEN);
        assert_eq!(hw.fc2_b.len(), out_dim);
        assert!(
            hw.fc1_w.iter().all(|v| v.is_finite()),
            "weights must be finite"
        );
        assert!(hw.fc2_w.iter().all(|v| v.is_finite()));
    }

    /// Degenerate inputs must not panic and must yield a correctly-shaped zeroed head.
    #[test]
    fn fit_head_empty_is_zeroed() {
        let hw = fit_head(&[], &[], 1, 0);
        assert_eq!(hw.fc1_w.len(), HIDDEN * FEAT_DIM);
        assert_eq!(hw.fc2_b.len(), 1);
        assert!(hw.fc1_w.iter().all(|&v| v == 0.0));
    }
}
