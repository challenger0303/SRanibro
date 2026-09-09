//! Brow post-processing: turn the raw brow CNN output into the emitted signed
//! expression per eye. This runs in the 120 Hz emit thread, but advances only
//! when the event-driven brow worker publishes a genuinely new stereo inference.
//!
//! The old standalone eyebrow tracker was stable because it used a continuous,
//! fixed-response EMA and a real neutral deadzone. The previous in-app path used
//! a fast/calm EMA switch plus a moving output deadband; noisy CLAHE input could
//! therefore sit still for several frames and then jump ("stick-slip"). This
//! implementation restores the old signal shape without changing the input
//! preprocessing expected by the current trained model:
//!
//! 1. EMA response is time-based, so 60, 90 and 120 Hz camera sources have the
//!    same latency.
//! 2. Neutral is averaged from a short run of open-eye samples, not one frame.
//! 3. A fixed neutral deadzone removes small noise continuously; there is no
//!    moving deadband at the destination.
//! 4. Blink samples never enter the EMA or baseline. The last open expression is
//!    held for the entire blink instead of decaying and visibly twitching.
//!
//! `process` returns `None` until a stable neutral baseline exists, so the
//! pipeline never emits a value derived from a zero placeholder.

/// Old tracker setting `smooth = 68` meant alpha=0.32 at its effective 100 Hz
/// update rate (about 26 ms). The current model must retain its noisier CLAHE
/// preprocessing, so replay of its recorded sequences uses twice that constant:
/// still responsive, but with old-tracker-like peak step noise.
const DEFAULT_EMA_TAU_S: f32 = 0.052;
/// Match the old tracker's per-eye neutral deadzone.
const DEFAULT_DEADZONE: f32 = 0.10;
/// Roughly 100 ms at 120 Hz: long enough not to use a noisy single frame, short
/// enough that brow output becomes ready immediately after startup/recenter.
const BASELINE_SAMPLES: u16 = 12;
/// If the filtered signal moves substantially while neutral is being captured,
/// restart the short window instead of averaging an eyebrow gesture into neutral.
const BASELINE_MAX_SPAN: f32 = 0.12;
const FALLBACK_INFER_DT_S: f32 = 1.0 / 120.0;

#[derive(Default)]
struct Eye {
    ema: f32,
    have_ema: bool,
    neutral: Option<f32>,
    neutral_sum: f32,
    neutral_samples: u16,
    neutral_min: f32,
    neutral_max: f32,
    output: f32,
    have_output: bool,
}

impl Eye {
    fn reset_neutral(&mut self) {
        self.neutral = None;
        self.neutral_sum = 0.0;
        self.neutral_samples = 0;
        self.neutral_min = 0.0;
        self.neutral_max = 0.0;
        self.output = 0.0;
        self.have_output = false;
    }

    fn add_neutral_sample(&mut self) {
        let value = self.ema;
        if self.neutral_samples == 0 {
            self.neutral_sum = value;
            self.neutral_samples = 1;
            self.neutral_min = value;
            self.neutral_max = value;
            return;
        }

        let min = self.neutral_min.min(value);
        let max = self.neutral_max.max(value);
        if max - min > BASELINE_MAX_SPAN {
            // Start a fresh stable suffix at the newest value.
            self.neutral_sum = value;
            self.neutral_samples = 1;
            self.neutral_min = value;
            self.neutral_max = value;
            return;
        }

        self.neutral_sum += value;
        self.neutral_samples += 1;
        self.neutral_min = min;
        self.neutral_max = max;
        if self.neutral_samples >= BASELINE_SAMPLES {
            self.neutral = Some(self.neutral_sum / self.neutral_samples as f32);
        }
    }
}

pub struct BrowState {
    eyes: [Eye; 2],
    /// Continuous-time EMA constant in seconds.
    tau_s: f32,
    /// Exact neutral zone before the response curve.
    deadzone: f32,
    /// Same optional power curve used by the old tracker (default Curve=5).
    gamma: f32,
}

impl Default for BrowState {
    fn default() -> Self {
        Self {
            eyes: Default::default(),
            tau_s: DEFAULT_EMA_TAU_S,
            deadzone: DEFAULT_DEADZONE,
            gamma: 1.5,
        }
    }
}

impl BrowState {
    /// Re-capture the neutral baseline on the next stable run of open samples.
    /// The EMA itself is retained so recenter never introduces a filter jump.
    pub fn recenter(&mut self) {
        for eye in &mut self.eyes {
            eye.reset_neutral();
        }
    }

    fn alpha(&self, infer_dt_s: f32) -> f32 {
        let tau = self.tau_s;
        if !tau.is_finite() || tau <= 1.0e-6 {
            return 1.0;
        }
        let dt = if infer_dt_s.is_finite() && infer_dt_s > 0.0 {
            infer_dt_s
        } else {
            FALLBACK_INFER_DT_S
        }
        .clamp(1.0 / 1000.0, 0.100);
        (1.0 - (-dt / tau).exp()).clamp(0.0, 1.0)
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
        let gamma = self.gamma;
        if !gamma.is_finite() || gamma <= 0.0 || (gamma - 1.0).abs() < 1.0e-3 {
            x.signum() * magnitude
        } else {
            x.signum() * magnitude.powf(gamma)
        }
    }

    /// Process one eye.
    ///
    /// `is_new` means the worker produced a new stereo inference; `infer_dt_s`
    /// is the elapsed time since the previous inference. Emit-only ticks never
    /// advance the EMA. Blink frames are held and excluded from all learning.
    pub fn process(
        &mut self,
        eye: usize,
        raw: f32,
        is_new: bool,
        blink: bool,
        recenter: bool,
        infer_dt_s: f32,
    ) -> Option<f32> {
        if recenter {
            self.eyes[eye].reset_neutral();
        }

        if is_new && !blink && raw.is_finite() {
            let alpha = self.alpha(infer_dt_s);
            let state = &mut self.eyes[eye];
            if state.have_ema {
                state.ema += alpha * (raw - state.ema);
            } else {
                state.ema = raw;
                state.have_ema = true;
            }
            if state.neutral.is_none() {
                state.add_neutral_sample();
            }
        }

        let (ema, neutral) = match (self.eyes[eye].have_ema, self.eyes[eye].neutral) {
            (true, Some(neutral)) => (self.eyes[eye].ema, neutral),
            _ => return None,
        };

        let state = &mut self.eyes[eye];
        if blink {
            // The brow model sees an eye-shaped crop and is not trained to interpret
            // eyelid occlusion. Holding the last open result is safer than allowing a
            // blink to pull the brow down and then spring back.
            return Some(if state.have_output { state.output } else { 0.0 });
        }

        let centered = (ema - neutral).clamp(-1.0, 1.0);
        let output = self.curve(centered);
        let state = &mut self.eyes[eye];
        state.output = output;
        state.have_output = true;
        Some(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DT_120: f32 = 1.0 / 120.0;

    fn establish_neutral(state: &mut BrowState, eye: usize, raw: f32) {
        for index in 0..BASELINE_SAMPLES {
            let output = state.process(eye, raw, true, false, false, DT_120);
            if index + 1 < BASELINE_SAMPLES {
                assert!(output.is_none());
            } else {
                assert!(output.unwrap().abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn not_ready_until_short_open_baseline_exists() {
        let mut state = BrowState::default();
        assert!(state.process(0, 0.0, false, false, false, 0.0).is_none());
        assert!(state.process(0, 0.4, true, true, false, DT_120).is_none());
        establish_neutral(&mut state, 0, 0.4);
    }

    #[test]
    fn neutral_subtracts_and_recenters() {
        let mut state = BrowState::default();
        state.tau_s = 0.0;
        state.deadzone = 0.0;
        state.gamma = 1.0;
        establish_neutral(&mut state, 0, 0.4);
        assert!((state.process(0, 0.7, true, false, false, DT_120).unwrap() - 0.3).abs() < 1.0e-6);

        assert!(state.process(0, 0.7, true, false, true, DT_120).is_none());
        for _ in 1..BASELINE_SAMPLES {
            state.process(0, 0.7, true, false, false, DT_120);
        }
        assert!(
            state
                .process(0, 0.7, false, false, false, 0.0)
                .unwrap()
                .abs()
                < 1.0e-6
        );
    }

    #[test]
    fn blink_holds_forever_and_never_contaminates_ema() {
        let mut state = BrowState::default();
        state.tau_s = 0.0;
        state.deadzone = 0.0;
        state.gamma = 1.0;
        establish_neutral(&mut state, 0, 0.0);
        assert_eq!(state.process(0, 0.8, true, false, false, DT_120), Some(0.8));
        for _ in 0..240 {
            assert_eq!(state.process(0, 5.0, true, true, false, DT_120), Some(0.8));
        }
        assert_eq!(state.process(0, 0.8, true, false, false, DT_120), Some(0.8));
    }

    #[test]
    fn neutral_jitter_is_an_exact_zero() {
        let mut state = BrowState::default();
        establish_neutral(&mut state, 0, 0.40);
        for index in 0..200 {
            let noise = ((index % 5) as f32 - 2.0) * 0.008;
            let output = state
                .process(0, 0.40 + noise, true, false, false, DT_120)
                .unwrap();
            assert_eq!(output, 0.0, "sample {index}: {output}");
        }
    }

    #[test]
    fn deliberate_brow_motion_remains_responsive() {
        let mut state = BrowState::default();
        establish_neutral(&mut state, 0, 0.0);
        let mut output = 0.0;
        for _ in 0..12 {
            output = state.process(0, 0.8, true, false, false, DT_120).unwrap();
        }
        assert!(
            output > 0.5,
            "100 ms of 120 Hz inferences should visibly move the brow: {output}"
        );
    }

    #[test]
    fn held_expression_noise_stays_small_without_stick_slip() {
        let mut state = BrowState::default();
        establish_neutral(&mut state, 0, 0.0);
        for _ in 0..120 {
            state.process(0, 0.65, true, false, false, DT_120);
        }

        let mut low = f32::INFINITY;
        let mut high = f32::NEG_INFINITY;
        for index in 0..240 {
            let noise = ((index % 7) as f32 - 3.0) * 0.006;
            let output = state
                .process(0, 0.65 + noise, true, false, false, DT_120)
                .unwrap();
            low = low.min(output);
            high = high.max(output);
        }
        assert!(
            high - low < 0.025,
            "filtered expression still jitters too much: {low}..{high}"
        );
    }

    #[test]
    fn slow_motion_advances_continuously_instead_of_sticking_then_jumping() {
        let mut state = BrowState::default();
        state.gamma = 1.0;
        establish_neutral(&mut state, 0, 0.0);
        let mut previous = 0.0;
        let mut changed = 0;
        let mut largest_step = 0.0f32;
        for index in 0..60 {
            let raw = 0.20 + 0.60 * index as f32 / 59.0;
            let output = state.process(0, raw, true, false, false, DT_120).unwrap();
            let step = (output - previous).abs();
            if step > 1.0e-5 {
                changed += 1;
            }
            largest_step = largest_step.max(step);
            previous = output;
        }
        assert!(
            changed > 50,
            "slow motion stuck on too many frames: {changed}"
        );
        assert!(
            largest_step < 0.06,
            "slow motion produced a visible jump: {largest_step}"
        );
    }

    #[test]
    fn ema_response_is_rate_independent() {
        fn response(rate: usize) -> f32 {
            let mut state = BrowState::default();
            state.deadzone = 0.0;
            state.gamma = 1.0;
            establish_neutral(&mut state, 0, 0.0);
            let dt = 1.0 / rate as f32;
            let mut output = 0.0;
            for _ in 0..(rate / 10) {
                output = state.process(0, 1.0, true, false, false, dt).unwrap();
            }
            output
        }

        let at_60 = response(60);
        let at_120 = response(120);
        assert!(
            (at_60 - at_120).abs() < 0.01,
            "time response changed with rate: 60 Hz={at_60}, 120 Hz={at_120}"
        );
    }

    #[test]
    fn non_finite_sample_is_ignored() {
        let mut state = BrowState::default();
        establish_neutral(&mut state, 0, 0.2);
        let before = state.process(0, 0.6, true, false, false, DT_120);
        assert_eq!(
            state.process(0, f32::NAN, true, false, false, DT_120),
            before
        );
    }
}
