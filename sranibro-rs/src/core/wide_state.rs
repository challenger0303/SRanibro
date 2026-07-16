//! Post-processing for the Dream Air/XR5 image-based EyeWide model.
//!
//! This intentionally does not reuse the eyelid state machine. The custom model already
//! predicts a wide score; this layer only removes per-user offset/range, neutral jitter,
//! blink contamination, and optional bilateral imbalance.

const BOOTSTRAP_SAMPLES: u32 = 24; // about 0.57 s at 42 inference/s
const BOOTSTRAP_MAX_RAW: f32 = 0.35;
const BOOTSTRAP_FALLBACK_AFTER: u32 = BOOTSTRAP_SAMPLES * 3;
const BOOTSTRAP_FALLBACK_BAND: f32 = 0.10;
const MIN_SPAN: f32 = 0.25;
const ENTER: f32 = 0.08;
const EXIT: f32 = 0.04;
const MOTION_RAW: f32 = 0.04;
const CALM_RAW: f32 = 0.03;
// A deliberate wide expression is a large raw step. Follow that edge quickly while
// retaining the much slower calm path below for neutral jitter suppression.
const FILTER_MOVE_TAU: f32 = 0.020;
const FILTER_CALM_TAU: f32 = 0.180;
const BASELINE_TAU: f32 = 30.0;
const SPAN_RELAX_TAU: f32 = 180.0;
const OUTPUT_ATTACK_TAU: f32 = 0.025;
const OUTPUT_RELEASE_TAU: f32 = 0.060;
const BILATERAL_GATE_TAU: f32 = 0.030;

pub const fn bootstrap_fallback_after() -> u32 {
    BOOTSTRAP_FALLBACK_AFTER
}

fn alpha(dt: f32, tau: f32) -> f32 {
    let dt = if dt.is_finite() {
        dt.clamp(0.0, 0.05)
    } else {
        0.0
    };
    if dt <= 0.0 || tau <= 0.0 {
        0.0
    } else {
        1.0 - (-dt / tau).exp()
    }
}

#[derive(Clone, Copy, Debug)]
struct Eye {
    filtered: f32,
    neutral: f32,
    span: f32,
    target: f32,
    output: f32,
    neutral_n: u32,
    bootstrap_seen: u32,
    bootstrap_min: f32,
    active: bool,
}

impl Default for Eye {
    fn default() -> Self {
        Self {
            filtered: 0.0,
            neutral: 0.0,
            // Start at the safe floor. The first real wide observation expands this
            // immediately, so a user whose fitted head reaches only e.g. 0.45 still gets
            // full usable range instead of waiting minutes for a 1.0 seed to decay.
            span: MIN_SPAN,
            target: 0.0,
            output: 0.0,
            neutral_n: 0,
            bootstrap_seen: 0,
            bootstrap_min: f32::INFINITY,
            active: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct WideDiag {
    pub filtered: [f32; 2],
    pub neutral: [f32; 2],
    pub span: [f32; 2],
    pub active: [bool; 2],
    pub gate: f32,
    pub bootstrap_seen: [u32; 2],
    pub ready: [bool; 2],
}

pub struct WideState {
    eyes: [Eye; 2],
    gate: f32,
}

impl Default for WideState {
    fn default() -> Self {
        Self {
            eyes: [Eye::default(); 2],
            gate: 0.0,
        }
    }
}

impl WideState {
    pub fn recenter(&mut self) {
        for eye in &mut self.eyes {
            eye.neutral_n = 0;
            eye.bootstrap_seen = 0;
            eye.bootstrap_min = f32::INFINITY;
            eye.active = false;
            eye.target = 0.0;
            eye.output = 0.0;
        }
        self.gate = 0.0;
    }

    /// Update a fresh stereo prediction and advance the 120 Hz output envelope.
    /// `infer_dt` matters only when `is_new`; `emit_dt` advances the final slew every call.
    /// Returns `None` per eye until a relaxed neutral bootstrap has completed.
    pub fn process_pair(
        &mut self,
        raw: [f32; 2],
        is_new: bool,
        blink: [bool; 2],
        infer_dt: f32,
        emit_dt: f32,
        requires_both: bool,
    ) -> [Option<f32>; 2] {
        if is_new {
            for i in 0..2 {
                if blink[i] || !raw[i].is_finite() {
                    continue;
                }
                let eye = &mut self.eyes[i];
                eye.bootstrap_seen = eye.bootstrap_seen.saturating_add(1);
                if eye.bootstrap_seen == 1 {
                    eye.filtered = raw[i];
                } else {
                    let residual = raw[i] - eye.filtered;
                    let tau = if residual.abs() >= MOTION_RAW {
                        FILTER_MOVE_TAU
                    } else {
                        FILTER_CALM_TAU
                    };
                    eye.filtered += alpha(infer_dt, tau) * residual;
                }
                eye.bootstrap_min = eye.bootstrap_min.min(eye.filtered);

                if eye.neutral_n < BOOTSTRAP_SAMPLES {
                    // A user who starts while holding wide cannot become the neutral.
                    // Normally enforce an absolute score ceiling. A generic model can
                    // have a harmless per-wearer offset, though, so after three expected
                    // bootstrap windows accept only values close to the lowest score seen.
                    // This avoids a permanent Custom=0/Auto fallback deadlock while still
                    // rejecting transient high/wide samples during the normal window.
                    let offset_fallback = eye.bootstrap_seen >= BOOTSTRAP_FALLBACK_AFTER
                        && eye.filtered <= eye.bootstrap_min + BOOTSTRAP_FALLBACK_BAND;
                    if eye.filtered <= BOOTSTRAP_MAX_RAW || offset_fallback {
                        eye.neutral_n += 1;
                        eye.neutral += (eye.filtered - eye.neutral) / eye.neutral_n as f32;
                    }
                    eye.target = 0.0;
                    continue;
                }

                let dev = eye.filtered - eye.neutral;
                let norm = (dev.max(0.0) / eye.span.max(MIN_SPAN)).clamp(0.0, 1.0);

                // Learn neutral only inside the already-neutral basin and only while
                // measurement motion is calm. A held real wide expression is outside the
                // basin and therefore cannot be silently calibrated away.
                if !eye.active && norm < ENTER && (raw[i] - eye.filtered).abs() < CALM_RAW {
                    eye.neutral += alpha(infer_dt, BASELINE_TAU) * (eye.filtered - eye.neutral);
                }

                // Per-eye range: reach a newly-observed maximum immediately, then relax
                // toward a safe floor over minutes so one outlier cannot desensitize forever.
                let positive = (eye.filtered - eye.neutral).max(0.0);
                if positive > eye.span {
                    eye.span = positive;
                } else {
                    let floor = positive.max(MIN_SPAN);
                    eye.span += alpha(infer_dt, SPAN_RELAX_TAU) * (floor - eye.span);
                }

                let x = ((eye.filtered - eye.neutral).max(0.0) / eye.span.max(MIN_SPAN))
                    .clamp(0.0, 1.0);
                if eye.active {
                    if x <= EXIT {
                        eye.active = false;
                        eye.target = 0.0;
                    } else {
                        eye.target = ((x - EXIT) / (1.0 - EXIT)).clamp(0.0, 1.0);
                    }
                } else if x >= ENTER {
                    eye.active = true;
                    eye.target = ((x - EXIT) / (1.0 - EXIT)).clamp(0.0, 1.0);
                } else {
                    eye.target = 0.0;
                }
            }
        }

        let mut ready = [false; 2];
        for i in 0..2 {
            let eye = &mut self.eyes[i];
            ready[i] = eye.neutral_n >= BOOTSTRAP_SAMPLES;
            if blink[i] {
                // EyeWide must never remain visible through a closed eye. The measurement,
                // baseline, and range were already frozen above.
                eye.output = 0.0;
                continue;
            }
            let tau = if eye.target > eye.output {
                OUTPUT_ATTACK_TAU
            } else {
                OUTPUT_RELEASE_TAU
            };
            eye.output += alpha(emit_dt, tau) * (eye.target - eye.output);
            if !eye.active && eye.output < 0.002 {
                eye.output = 0.0;
            }
        }

        let any_blink = blink[0] || blink[1];
        if requires_both {
            let target = if self.eyes[0].active && self.eyes[1].active {
                1.0
            } else {
                0.0
            };
            if !any_blink {
                self.gate += alpha(emit_dt, BILATERAL_GATE_TAU) * (target - self.gate);
            }
            let value = if any_blink {
                0.0
            } else {
                self.eyes[0].output.max(self.eyes[1].output) * self.gate
            };
            return [ready[0].then_some(value), ready[1].then_some(value)];
        }

        self.gate = 1.0;
        [
            ready[0].then_some(self.eyes[0].output),
            ready[1].then_some(self.eyes[1].output),
        ]
    }

    pub fn diag(&self) -> WideDiag {
        WideDiag {
            filtered: [self.eyes[0].filtered, self.eyes[1].filtered],
            neutral: [self.eyes[0].neutral, self.eyes[1].neutral],
            span: [self.eyes[0].span, self.eyes[1].span],
            active: [self.eyes[0].active, self.eyes[1].active],
            gate: self.gate,
            bootstrap_seen: [self.eyes[0].bootstrap_seen, self.eyes[1].bootstrap_seen],
            ready: [
                self.eyes[0].neutral_n >= BOOTSTRAP_SAMPLES,
                self.eyes[1].neutral_n >= BOOTSTRAP_SAMPLES,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IDT: f32 = 1.0 / 42.0;
    const EDT: f32 = 1.0 / 120.0;

    fn infer(state: &mut WideState, raw: [f32; 2], both: bool) -> [Option<f32>; 2] {
        let mut out = state.process_pair(raw, true, [false; 2], IDT, EDT, both);
        for _ in 0..2 {
            out = state.process_pair(raw, false, [false; 2], IDT, EDT, both);
        }
        out
    }

    fn bootstrap(state: &mut WideState) {
        for i in 0..BOOTSTRAP_SAMPLES + 4 {
            let n = if i % 2 == 0 { 0.015 } else { -0.015 };
            infer(state, [0.10 + n, 0.12 - n], true);
        }
    }

    #[test]
    fn neutral_jitter_is_exact_zero() {
        let mut state = WideState::default();
        bootstrap(&mut state);
        for i in 0..200 {
            let n = ((i % 7) as f32 - 3.0) * 0.006;
            let out = infer(&mut state, [0.10 + n, 0.12 - n], true);
            assert_eq!(out, [Some(0.0), Some(0.0)]);
        }
    }

    #[test]
    fn offset_neutral_cannot_deadlock_bootstrap_forever() {
        let mut state = WideState::default();
        let mut out = [None; 2];
        for _ in 0..BOOTSTRAP_FALLBACK_AFTER - 1 {
            out = infer(&mut state, [0.52, 0.56], false);
        }
        assert_eq!(out, [None, None]);
        for _ in 0..BOOTSTRAP_SAMPLES + 2 {
            out = infer(&mut state, [0.52, 0.56], false);
        }
        assert_eq!(out, [Some(0.0), Some(0.0)]);
        assert_eq!(state.diag().ready, [true, true]);
    }

    #[test]
    fn bilateral_step_starts_within_two_inferences_and_returns_to_zero() {
        let mut state = WideState::default();
        bootstrap(&mut state);
        let _ = infer(&mut state, [0.80, 0.82], true);
        let out = infer(&mut state, [0.80, 0.82], true);
        assert!(
            out[0].unwrap() > 0.55,
            "second inference should already be strongly active: {out:?}"
        );
        for _ in 0..30 {
            infer(&mut state, [0.10, 0.12], true);
        }
        let out = infer(&mut state, [0.10, 0.12], true);
        assert_eq!(out, [Some(0.0), Some(0.0)]);
    }

    #[test]
    fn sustained_wide_does_not_teach_neutral() {
        let mut state = WideState::default();
        bootstrap(&mut state);
        let before = state.diag().neutral;
        for _ in 0..42 * 30 {
            infer(&mut state, [0.75, 0.80], false);
        }
        let after = state.diag().neutral;
        assert!((after[0] - before[0]).abs() < 0.002);
        assert!((after[1] - before[1]).abs() < 0.002);
    }

    #[test]
    fn blink_contamination_cannot_move_calibration() {
        let mut state = WideState::default();
        bootstrap(&mut state);
        let before = state.diag();
        for _ in 0..100 {
            let out = state.process_pair([9.0, -9.0], true, [true; 2], IDT, EDT, true);
            assert_eq!(out, [Some(0.0), Some(0.0)]);
        }
        let after = state.diag();
        assert_eq!(before.neutral, after.neutral);
        assert_eq!(before.span, after.span);
    }

    #[test]
    fn unilateral_expression_is_preserved_only_when_requested() {
        let mut free = WideState::default();
        let mut bilateral = WideState::default();
        bootstrap(&mut free);
        bootstrap(&mut bilateral);
        let mut a = [None; 2];
        let mut b = [None; 2];
        for _ in 0..10 {
            a = infer(&mut free, [0.85, 0.12], false);
            b = infer(&mut bilateral, [0.85, 0.12], true);
        }
        assert!(a[0].unwrap() > 0.2 && a[1].unwrap() == 0.0);
        assert_eq!(b, [Some(0.0), Some(0.0)]);
    }
}
