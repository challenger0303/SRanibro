
const HIDDEN: usize = 128;
const FEAT_DIM: usize = 64 * 4 * 4;

const LR: f64 = 1e-3;
const BETA1: f64 = 0.9;
const BETA2: f64 = 0.999;
const EPS: f64 = 1e-8;
const WEIGHT_DECAY: f64 = 1e-5;
const BATCH: usize = 32;
const MAX_EPOCHS: usize = 60;
const PATIENCE: usize = 5;
const VAL_FRAC: f64 = 0.15;

#[derive(Clone)]
pub struct HeadWeights {
    pub fc1_w: Vec<f32>,
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,
    pub fc2_b: Vec<f32>,
}

impl HeadWeights {
    fn zeros(in_dim: usize, out_dim: usize) -> Self {
        HeadWeights {
            fc1_w: vec![0.0; HIDDEN * in_dim],
            fc1_b: vec![0.0; HIDDEN],
            fc2_w: vec![0.0; out_dim * HIDDEN],
            fc2_b: vec![0.0; out_dim],
        }
    }
}

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

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn uniform(&mut self, b: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * b
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

fn shuffle(rng: &mut Rng, idx: &mut [usize]) {
    for i in (1..idx.len()).rev() {
        let j = rng.below(i + 1);
        idx.swap(i, j);
    }
}

#[derive(Clone)]
struct Head {
    in_dim: usize,
    hidden: usize,
    out_dim: usize,
    fc1_w: Vec<f64>,
    fc1_b: Vec<f64>,
    fc2_w: Vec<f64>,
    fc2_b: Vec<f64>,
}

impl Head {
    fn init(in_dim: usize, hidden: usize, out_dim: usize, rng: &mut Rng) -> Head {
        let b1 = 1.0 / (in_dim as f64).sqrt();
        let b2 = 1.0 / (hidden as f64).sqrt();
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

    fn predict(&self, x: &[f64]) -> Vec<f64> {
        self.forward(x).2
    }

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

fn batch_loss_grad(head: &Head, xs: &[Vec<f64>], ts: &[Vec<f64>], idx: &[usize]) -> (f64, Grads) {
    let mut g = Grads::zeros(head.in_dim, head.hidden, head.out_dim);
    let mut loss = 0.0f64;
    for &s in idx {
        let x = &xs[s];
        let t = &ts[s];
        let (pre1, h, y) = head.forward(x);
        let mut dy = vec![0.0f64; head.out_dim];
        for k in 0..head.out_dim {
            let e = y[k] - t[k];
            loss += e * e;
            dy[k] = 2.0 * e;
        }
        let mut dh = vec![0.0f64; head.hidden];
        for k in 0..head.out_dim {
            let base = k * head.hidden;
            g.fc2_b[k] += dy[k];
            for o in 0..head.hidden {
                g.fc2_w[base + o] += dy[k] * h[o];
                dh[o] += head.fc2_w[base + o] * dy[k];
            }
        }
        for o in 0..head.hidden {
            let dpre = if pre1[o] > 0.0 { dh[o] } else { 0.0 };
            g.fc1_b[o] += dpre;
            let base = o * head.in_dim;
            for i in 0..head.in_dim {
                g.fc1_w[base + i] += dpre * x[i];
            }
        }
    }
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

struct Adam {
    t: i32,
    m: Grads,
    v: Grads,
}

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

    let xs: Vec<Vec<f64>> = features
        .iter()
        .map(|f| f.iter().map(|&v| v as f64).collect())
        .collect();
    let ts: Vec<Vec<f64>> = labels
        .iter()
        .map(|l| l.iter().map(|&v| v as f64).collect())
        .collect();

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

pub struct FitHeadReport {
    pub weights: HeadWeights,
    pub train_mse: f64,
    pub val_mse: f64,
}

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

    #[test]
    fn head_gradient_check() {
        let (in_dim, hidden, out_dim) = (6usize, 4usize, 2usize);
        let mut rng = Rng::new(0xC0FF_EE12_3456_789A);
        let head = Head::init(in_dim, hidden, out_dim, &mut rng);
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

    #[test]
    fn head_converges_on_linear_data() {
        let (in_dim, out_dim, n) = (8usize, 2usize, 240usize);
        let mut rng = Rng::new(0xABCD_1234_5678_9012);
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
        let before: f64 = labels
            .iter()
            .map(|l| l.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>())
            .sum::<f64>()
            / n as f64;

        let hw = fit_head(&feats, &labels, out_dim, 42);

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

    #[test]
    fn fit_head_empty_is_zeroed() {
        let hw = fit_head(&[], &[], 1, 0);
        assert_eq!(hw.fc1_w.len(), HIDDEN * FEAT_DIM);
        assert_eq!(hw.fc2_b.len(), 1);
        assert!(hw.fc1_w.iter().all(|&v| v == 0.0));
    }
}
