
use crate::core::types::BrightnessNorm;

const ROI: f32 = 0.625;
pub const WARMUP: u32 = 90;

pub const MANUAL_GAIN_MIN: f32 = 0.5;
pub const MANUAL_GAIN_MAX: f32 = 2.0;

pub fn sanitize_manual_gain(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(MANUAL_GAIN_MIN, MANUAL_GAIN_MAX)
    } else {
        BrightnessNorm::default().manual_gain
    }
}

pub fn compose_manual_gain(affine: &mut [(f32, f32); 2], value: f32) {
    let gain = sanitize_manual_gain(value);
    for (a, b) in affine {
        *a *= gain;
        *b *= gain;
    }
}

#[derive(Clone, Copy)]
pub struct BrightState {
    base_level: f32,
    base_spread: f32,
    warm: u32,
    init: bool,
    in_band: bool,
    omax: f32,
    omax_init: bool,
    good_streak: u32,
    target_seen: bool,
}

impl Default for BrightState {
    fn default() -> Self {
        Self {
            base_level: 128.0,
            base_spread: 40.0,
            warm: 0,
            init: false,
            in_band: true,
            omax: 0.5,
            omax_init: false,
            good_streak: 0,
            target_seen: false,
        }
    }
}

const GOOD_CAP: u32 = 45;

fn roi_stats(frame: &[u8], w: usize, h: usize) -> (f32, f32) {
    let mx = (w as f32 * (1.0 - ROI) * 0.5) as usize;
    let my = (h as f32 * (1.0 - ROI) * 0.5) as usize;
    let (x0, x1) = (mx.min(w / 2), (w - mx).max(w / 2 + 1).min(w));
    let (y0, y1) = (my.min(h / 2), (h - my).max(h / 2 + 1).min(h));
    let mut hist = [0u32; 256];
    let mut n = 0u32;
    for y in y0..y1 {
        let row = y * w;
        for x in x0..x1 {
            hist[frame[row + x] as usize] += 1;
            n += 1;
        }
    }
    if n == 0 {
        return (128.0, 40.0);
    }
    let pct = |p: f32| -> f32 {
        let target = (p * n as f32) as u32;
        let mut acc = 0u32;
        for (v, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= target {
                return v as f32;
            }
        }
        255.0
    };
    let level = pct(0.5);
    let spread = ((pct(0.84) - pct(0.16)) * 0.5).max(1.0);
    (level, spread)
}

pub fn update(st: &mut BrightState, frame: &[u8], w: usize, h: usize, adapt: f32) {
    if w == 0 || h == 0 || frame.len() < w * h {
        return;
    }
    let (lvl, spr) = roi_stats(frame, w, h);
    if !st.init {
        st.base_level = lvl;
        st.base_spread = spr;
        st.init = true;
        st.warm = 1;
        return;
    }
    let dev = lvl - st.base_level;
    let gate = (0.8 * st.base_spread).clamp(6.0, 22.0);
    st.in_band = dev.abs() <= gate;
    let a = (if st.in_band { adapt } else { adapt * 0.005 }).clamp(0.0, 0.2);
    st.base_level += a * dev;
    st.base_spread += a * (spr - st.base_spread);
    st.warm = st.warm.saturating_add(1);
}

pub fn warmed(st: &BrightState) -> bool {
    st.init && st.warm >= WARMUP
}

pub fn capture(st: &BrightState) -> (f32, f32) {
    (st.base_level, st.base_spread)
}

pub fn affine(st: &BrightState, tgt_level: f32, tgt_spread: f32, strength: f32) -> (f32, f32) {
    if !tgt_level.is_finite() || !tgt_spread.is_finite() {
        return (1.0, 0.0);
    }
    let gain = (tgt_spread / st.base_spread.max(1.0)).clamp(0.3, 3.0);
    let off = tgt_level - gain * st.base_level;
    let s = if strength.is_finite() {
        strength.clamp(0.0, 1.0)
    } else {
        1.0
    };
    (1.0 + s * (gain - 1.0), s * off)
}

pub fn apply(frame: &[u8], a: f32, b: f32) -> Vec<u8> {
    if (a - 1.0).abs() < 1e-4 && b.abs() < 1e-3 {
        return frame.to_vec();
    }
    frame
        .iter()
        .map(|&v| (a * v as f32 + b).clamp(0.0, 255.0) as u8)
        .collect()
}

pub fn step(
    states: &mut [BrightState; 2],
    frames: [(&[u8], usize, usize); 2],
    prev_open: [f32; 2],
    norm: &mut BrightnessNorm,
) -> [(f32, f32); 2] {
    if !norm.enabled {
        return [(1.0, 0.0), (1.0, 0.0)];
    }
    if !norm.captured && states.iter().any(|state| state.target_seen) {
        *states = [BrightState::default(); 2];
    }
    for i in 0..2 {
        let (f, w, h) = frames[i];
        update(&mut states[i], f, w, h, norm.adapt);
        let o = prev_open[i];
        let st = &mut states[i];
        if o.is_finite() {
            if !st.omax_init {
                st.omax = o.max(0.1);
                st.omax_init = true;
            } else {
                st.omax = (st.omax * 0.999).max(o);
            }
        }
        let good = warmed(st) && st.in_band && o.is_finite() && st.omax >= 0.5 && o > 0.8 * st.omax;
        st.good_streak = if good {
            st.good_streak.saturating_add(1)
        } else {
            0
        };
    }
    if norm.auto_learn
        && !norm.captured
        && states[0].good_streak >= GOOD_CAP
        && states[1].good_streak >= GOOD_CAP
    {
        for i in 0..2 {
            let (l, s) = capture(&states[i]);
            norm.tgt_level[i] = l;
            norm.tgt_spread[i] = s;
        }
        norm.captured = true;
    }
    let mut out = [(1.0, 0.0); 2];
    if norm.captured {
        for i in 0..2 {
            out[i] = affine(
                &states[i],
                norm.tgt_level[i],
                norm.tgt_spread[i],
                norm.strength,
            );
        }
    }
    for state in states.iter_mut() {
        state.target_seen = norm.captured;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roi_stats_flat_and_gradient() {
        let flat = vec![100u8; 200 * 200];
        let (l, s) = roi_stats(&flat, 200, 200);
        assert!(
            (l - 100.0).abs() < 1.0 && s <= 1.0,
            "flat: level~100 spread~0, got {l},{s}"
        );
    }

    #[test]
    fn affine_preserves_darkening() {
        let st = BrightState {
            base_level: 160.0,
            base_spread: 40.0,
            warm: 200,
            init: true,
            ..Default::default()
        };
        let (a, b) = affine(&st, 120.0, 40.0, 1.0);
        let at_base = a * 160.0 + b;
        assert!(
            (at_base - 120.0).abs() < 1.0,
            "baseline maps onto target, got {at_base}"
        );
        let at_dark = a * 80.0 + b;
        assert!(
            at_dark < at_base - 30.0,
            "darker input stays clearly darker, got {at_dark}"
        );
    }

    #[test]
    fn dim_frame_is_brightened_toward_target() {
        let st = BrightState {
            base_level: 90.0,
            base_spread: 30.0,
            warm: 200,
            init: true,
            ..Default::default()
        };
        let (a, b) = affine(&st, 140.0, 40.0, 1.0);
        let mapped = a * 90.0 + b;
        assert!(
            mapped > 130.0,
            "dim baseline brightened toward target, got {mapped}"
        );
    }

    #[test]
    fn manual_gain_is_post_adaptive_and_safely_bounded() {
        let mut affine = [(1.2, -8.0), (0.8, 12.0)];
        compose_manual_gain(&mut affine, 1.25);
        assert_eq!(affine, [(1.5, -10.0), (1.0, 15.0)]);

        let mut invalid = [(1.0, 3.0), (2.0, -4.0)];
        compose_manual_gain(&mut invalid, f32::NAN);
        assert_eq!(invalid, [(1.6, 4.8), (3.2, -6.4)]);
        assert_eq!(sanitize_manual_gain(0.1), MANUAL_GAIN_MIN);
        assert_eq!(sanitize_manual_gain(9.0), MANUAL_GAIN_MAX);
    }

    #[test]
    fn recapture_clears_the_stale_runtime_baseline_once() {
        let mut norm = BrightnessNorm {
            enabled: true,
            captured: true,
            ..BrightnessNorm::default()
        };
        let frame = vec![80u8; 200 * 200];
        let pair = [
            (&frame[..], 200usize, 200usize),
            (&frame[..], 200usize, 200usize),
        ];
        let mut states = [BrightState::default(); 2];
        step(&mut states, pair, [0.8, 0.8], &mut norm);
        assert!(states.iter().all(|state| state.target_seen));

        norm.captured = false;
        step(&mut states, pair, [0.8, 0.8], &mut norm);
        assert!(states.iter().all(|state| state.warm == 1));
        assert!(states.iter().all(|state| !state.target_seen));

        step(&mut states, pair, [0.8, 0.8], &mut norm);
        assert!(states.iter().all(|state| state.warm == 2));
    }

    #[test]
    fn baseline_resists_held_expression() {
        let mut st = BrightState::default();
        let mut relaxed = vec![0u8; 200 * 200];
        for (i, p) in relaxed.iter_mut().enumerate() {
            *p = ((i % 200) as u8 / 2).wrapping_add(90);
        }
        for _ in 0..250 {
            update(&mut st, &relaxed, 200, 200, 0.05);
        }
        let base_before = st.base_level;
        let dark = vec![30u8; 200 * 200];
        for _ in 0..180 {
            update(&mut st, &dark, 200, 200, 0.05);
        }
        let drift = (st.base_level - base_before).abs();
        assert!(
            drift < 6.0,
            "a held wink must not drag the baseline (drifted {drift})"
        );
    }

    #[test]
    fn step_captures_only_from_good_open_frames() {
        let mut norm = BrightnessNorm {
            enabled: true,
            ..BrightnessNorm::default()
        };
        let mut st = [BrightState::default(); 2];
        let mut frame = vec![0u8; 200 * 200];
        for (i, p) in frame.iter_mut().enumerate() {
            *p = ((i % 200) as u8).wrapping_add(40);
        }
        let f = [
            (&frame[..], 200usize, 200usize),
            (&frame[..], 200usize, 200usize),
        ];
        for _ in 0..WARMUP + GOOD_CAP + 20 {
            step(&mut st, f, [0.2, 0.2], &mut norm);
        }
        assert!(!norm.captured, "closed eyes must not trigger a capture");
        let mut last = [(1.0, 0.0); 2];
        for _ in 0..GOOD_CAP + 5 {
            last = step(&mut st, f, [0.8, 0.8], &mut norm);
        }
        assert!(
            norm.captured,
            "target captured from sustained good open frames"
        );
        assert!(
            (last[0].0 - 1.0).abs() < 0.15 && last[0].1.abs() < 8.0,
            "near-identity at capture: {:?}",
            last[0]
        );
    }
}
