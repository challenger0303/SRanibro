//! Brow post-processing: turn the raw brow CNN output into the emitted signed brow
//! expression per eye. Runs in the emit thread (which owns the blink flag + recenter).
//!
//! The signal is "deviation of your eye-shape from neutral". This stage:
//! 1. EMA-smooths the RAW output, but ONLY on a new, NON-blink inference — a blink
//!    deforms the eye, so a blink sample must never enter the filter or the baseline
//!    (otherwise the eye spikes on reopening).
//! 2. Subtracts a frozen NEUTRAL baseline captured (from the smoothed value) on the
//!    first open frame and re-captured on recenter.
//! 3. BLINK-GATES: during a blink hold the last open value briefly, then decay to neutral.
//! 4. Applies an optional power curve (deadzone).
//!
//! `process` returns `None` until the first real inference + neutral are available, so the
//! pipeline never emits a brow value derived from a zero placeholder.

/// Frames (at the ~120 Hz emit rate) to HOLD the last value when a blink starts
/// (~125 ms), then to DECAY it to neutral over (~375 ms more).
const HOLD_FRAMES: u16 = 15;
const DECAY_FRAMES: u16 = 45;
/// A raw-model residual above this is deliberate motion and keeps the fast EMA.
const MOTION_RAW: f32 = 0.10;
/// Position-independent output hysteresis. Unlike the neutral deadzone this follows the
/// expression across the whole range, suppressing small oscillation after the brow arrives.
const SETTLE_DEADBAND: f32 = 0.015;

#[derive(Default)]
struct Eye {
    ema: f32,
    have_ema: bool,
    neutral: Option<f32>,
    last: f32, // last non-blink centered value (held through a blink)
    have_last: bool,
    blink_frames: u16,
    output: f32,
    have_output: bool,
}

pub struct BrowState {
    eyes: [Eye; 2],
    /// EMA factor per NEW inference (0..1).
    alpha: f32,
    /// Stronger smoothing for small residuals around a held expression.
    calm_alpha: f32,
    /// Exact neutral zone before the power curve.
    deadzone: f32,
    /// Moving deadband around the last published non-neutral expression.
    settle_deadband: f32,
    /// Power-curve exponent. Guarded: non-finite or <=0 falls back to linear.
    gamma: f32,
}

impl Default for BrowState {
    fn default() -> Self {
        Self {
            eyes: Default::default(),
            alpha: 0.5,
            calm_alpha: 0.08,
            deadzone: 0.03,
            settle_deadband: SETTLE_DEADBAND,
            // Matches vr_eyebrow's default Curve=5 (gamma=1.5): small model noise is
            // compressed while full gestures still reach exactly +/-1.
            gamma: 1.5,
        }
    }
}

impl BrowState {
    /// Re-capture the neutral baseline (and clear blink/hold state) on the next open
    /// sample for both eyes. Keeps the EMA (only the baseline is re-learned).
    pub fn recenter(&mut self) {
        for e in &mut self.eyes {
            e.neutral = None;
            e.have_last = false;
            e.blink_frames = 0;
            e.have_output = false;
        }
    }

    fn curve(&self, x: f32) -> f32 {
        let x = x.clamp(-1.0, 1.0);
        let deadzone = if self.deadzone.is_finite() {
            self.deadzone.clamp(0.0, 0.30)
        } else {
            0.0
        };
        let magnitude = x.abs();
        if magnitude <= deadzone {
            return 0.0;
        }
        let magnitude = ((magnitude - deadzone) / (1.0 - deadzone)).clamp(0.0, 1.0);
        let g = self.gamma;
        if !g.is_finite() || g <= 0.0 || (g - 1.0).abs() < 1e-3 {
            x.signum() * magnitude
        } else {
            x.signum() * magnitude.powf(g)
        }
    }

    /// Process one eye. `is_new` = a fresh inference arrived this tick (gate the EMA so
    /// it advances once per inference, not per emit tick). `blink` holds/decays the
    /// output and is excluded from the EMA + baseline. Returns `None` until ready.
    pub fn process(
        &mut self,
        eye: usize,
        raw: f32,
        is_new: bool,
        blink: bool,
        recenter: bool,
    ) -> Option<f32> {
        if recenter {
            let e = &mut self.eyes[eye];
            e.neutral = None;
            e.have_last = false;
            e.blink_frames = 0;
            e.have_output = false;
        }
        // Advance the EMA + capture neutral ONLY on a fresh, non-blink, finite sample.
        if is_new && !blink && raw.is_finite() {
            let e = &mut self.eyes[eye];
            if e.have_ema {
                let residual = raw - e.ema;
                let alpha = if residual.abs() >= MOTION_RAW {
                    self.alpha
                } else {
                    self.calm_alpha.min(self.alpha)
                }
                .clamp(0.0, 1.0);
                e.ema += alpha * residual;
            } else {
                e.ema = raw;
                e.have_ema = true;
            }
            if e.neutral.is_none() {
                e.neutral = Some(e.ema);
            }
        }
        let (ema, neutral) = match (self.eyes[eye].have_ema, self.eyes[eye].neutral) {
            (true, Some(n)) => (self.eyes[eye].ema, n),
            _ => return None, // not ready: no open inference / baseline yet
        };
        if !blink {
            let c = (ema - neutral).clamp(-1.0, 1.0);
            let target = self.curve(c);
            let settle_deadband = if self.settle_deadband.is_finite() {
                self.settle_deadband.clamp(0.0, 0.10)
            } else {
                0.0
            };
            let e = &mut self.eyes[eye];
            e.blink_frames = 0;
            e.last = c;
            e.have_last = true;

            // A neutral target must remain an exact zero. Away from neutral, hold the
            // previous output inside a narrow moving deadband. When deliberate motion
            // exceeds it, consume only the deadband portion and follow the remainder;
            // therefore small noise cannot twitch the avatar, while larger changes do
            // not acquire an accumulating time lag.
            if !e.have_output || target == 0.0 {
                e.output = target;
                e.have_output = true;
            } else {
                let delta = target - e.output;
                if delta.abs() > settle_deadband {
                    e.output = target - delta.signum() * settle_deadband;
                }
            }
            return Some(e.output);
        }
        // Blink: hold the last OPEN value, then decay to neutral. If we've never seen an
        // open frame since (re)start, just sit at neutral.
        let e = &mut self.eyes[eye];
        if !e.have_last {
            return Some(0.0);
        }
        let bf = e.blink_frames;
        e.blink_frames = bf.saturating_add(1);
        // `output` is already curved + settled. Holding that exact published value avoids
        // a one-frame jump when a blink begins.
        let held = e.output;
        let v = if bf < HOLD_FRAMES {
            held
        } else {
            let t = ((bf - HOLD_FRAMES + 1) as f32 / DECAY_FRAMES as f32).min(1.0);
            held * (1.0 - t)
        };
        Some(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_ready_until_first_open_inference() {
        let mut b = BrowState::default();
        // No inference yet -> None.
        assert!(b.process(0, 0.0, false, false, false).is_none());
        // A blink-only inference must not establish a baseline.
        assert!(b.process(0, 0.4, true, true, false).is_none());
        // First OPEN inference establishes ema+neutral -> ~0.
        assert!(b.process(0, 0.4, true, false, false).unwrap().abs() < 1e-6);
    }

    #[test]
    fn neutral_subtracts_and_recenters() {
        let mut b = BrowState::default();
        b.alpha = 1.0; // make the EMA track exactly for the test
        b.deadzone = 0.0;
        b.settle_deadband = 0.0;
        b.gamma = 1.0;
        assert!(b.process(0, 0.4, true, false, false).unwrap().abs() < 1e-6); // neutral=0.4
        assert!((b.process(0, 0.7, true, false, false).unwrap() - 0.3).abs() < 1e-6);
        // Recenter re-baselines to the next open sample.
        assert!(b.process(0, 0.7, true, false, true).unwrap().abs() < 1e-6);
    }

    #[test]
    fn blink_holds_then_decays_and_excludes_ema() {
        let mut b = BrowState::default();
        b.alpha = 1.0;
        b.deadzone = 0.0;
        b.settle_deadband = 0.0;
        b.gamma = 1.0;
        b.process(0, 0.0, true, false, false); // neutral 0
        assert!((b.process(0, 0.8, true, false, false).unwrap() - 0.8).abs() < 1e-6);
        // A blink sample with a wild raw must NOT move the EMA.
        for _ in 0..HOLD_FRAMES {
            assert!((b.process(0, 5.0, true, true, false).unwrap() - 0.8).abs() < 1e-6);
        }
        let d = b.process(0, 5.0, true, true, false).unwrap();
        assert!(d < 0.8, "decaying: {d}");
        // On reopen the EMA is still 0.8 (blink excluded), not contaminated by 5.0.
        let r = b.process(0, 0.8, true, false, false).unwrap();
        assert!((r - 0.8).abs() < 1e-6, "post-blink {r}");
    }

    #[test]
    fn neutral_jitter_is_an_exact_zero() {
        let mut b = BrowState::default();
        b.process(0, 0.40, true, false, false);
        for i in 0..200 {
            let noise = ((i % 5) as f32 - 2.0) * 0.008;
            let out = b.process(0, 0.40 + noise, true, false, false).unwrap();
            assert_eq!(out, 0.0, "sample {i}: {out}");
        }
    }

    #[test]
    fn deliberate_brow_motion_remains_responsive() {
        let mut b = BrowState::default();
        b.process(0, 0.0, true, false, false);
        let mut out = 0.0;
        for _ in 0..4 {
            out = b.process(0, 0.8, true, false, false).unwrap();
        }
        assert!(out > 0.5, "four inferences should be visibly raised: {out}");
    }

    #[test]
    fn held_expression_rejects_destination_jitter() {
        let mut b = BrowState::default();
        b.process(0, 0.0, true, false, false);
        for _ in 0..120 {
            b.process(0, 0.65, true, false, false);
        }

        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..240 {
            let noise = ((i % 7) as f32 - 3.0) * 0.006;
            let out = b.process(0, 0.65 + noise, true, false, false).unwrap();
            lo = lo.min(out);
            hi = hi.max(out);
        }
        assert!(hi - lo < 0.006, "held output still jitters: {lo}..{hi}");
    }

    #[test]
    fn settle_deadband_does_not_block_a_small_deliberate_move() {
        let mut b = BrowState::default();
        b.process(0, 0.0, true, false, false);
        for _ in 0..20 {
            b.process(0, 0.45, true, false, false);
        }
        let before = b.process(0, 0.45, true, false, false).unwrap();
        let mut after = before;
        for _ in 0..12 {
            after = b.process(0, 0.60, true, false, false).unwrap();
        }
        assert!(
            after > before + 0.08,
            "small deliberate move was stuck: {before} -> {after}"
        );
    }
}
