//! Adaptive per-user brightness + contrast normalization for the eye ML input.
//!
//! Lens-to-eye distance varies per person (and per HMD reseat), so the IR image's overall
//! brightness/contrast drifts — and the openness model reads "brighter = more open" (per
//! the response heatmap), so that drift biases openness. We hold the input at a learned
//! target with an affine `out = a*in + b`, where the SOURCE is a SLOWLY-adapted per-user
//! baseline (so lens-distance drift is corrected) and fast changes — blinks — pass through
//! unchanged (the transform is fixed on the slow baseline, so a darker blink frame maps
//! below the target = still dark). Robust stats (median + inter-percentile spread) over a
//! central ROI make it insensitive to the few remaining specular pixels. Runs AFTER
//! despeckle, on the native-resolution frame. See [`crate::core::types::BrightnessNorm`].

use crate::core::types::BrightnessNorm;

/// Central fraction of the frame used for the robust stats (skips the dark periphery / mask).
const ROI: f32 = 0.625;
/// Frames (~60Hz ML) the baseline settles before the target is auto-captured (~1.5s).
pub const WARMUP: u32 = 90;

/// Runtime (non-persisted) per-eye baseline the normalizer adapts — the SOURCE. The
/// persisted target + params live in [`BrightnessNorm`] (config); this is transient state
/// owned by the ML thread.
#[derive(Clone, Copy)]
pub struct BrightState {
    base_level: f32,
    base_spread: f32,
    warm: u32,
    init: bool,
    /// Whether the last frame's brightness sat in the relaxed band (not a wink/wide).
    in_band: bool,
    /// Running MAX of the raw model openness (with slow decay) — the eye's own open peak,
    /// used to gate the target capture on "the eye is actually open".
    omax: f32,
    omax_init: bool,
    /// Consecutive frames this eye has been a GOOD capture frame (open + relaxed brightness).
    good_streak: u32,
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
        }
    }
}

/// Both eyes must be a GOOD frame (relaxed + open) for this many consecutive frames before
/// the target is captured (~0.75s @60Hz) — so the reference is a genuinely good frame, not a
/// blind post-warmup snapshot.
const GOOD_CAP: u32 = 45;

/// Robust (level, spread) of the central ROI via a 256-bin histogram: level = median,
/// spread = half the p16..p84 range (~1 std for a normal). O(n) + O(256).
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

/// Advance one eye's baseline from a frame (slow EMA). Call once per ML frame.
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
    // Gate the adaptation on how far this frame's level sits from the settled baseline.
    // Relaxed jitter (in-band) adapts normally; a WINK (much darker) or a WIDE (much
    // brighter) is an EXPRESSION, not a lens-distance change — adapt ~200x slower so a held
    // expression can't drag the baseline. Without this, the baseline chased the wink/wide and
    // the affine then BRIGHTENED a closed eye (openness spiked up while shut) or DIMMED a wide
    // eye (wide stopped firing after a few tries) — the two reported regressions. A genuine
    // reseat is a PERSISTENT shift, so the slow out-of-band rate still re-anchors it over ~a
    // minute (or hit "Recapture reference" for instant).
    let dev = lvl - st.base_level;
    let gate = (0.8 * st.base_spread).clamp(6.0, 22.0);
    st.in_band = dev.abs() <= gate;
    let a = (if st.in_band { adapt } else { adapt * 0.005 }).clamp(0.0, 0.2);
    st.base_level += a * dev;
    st.base_spread += a * (spr - st.base_spread);
    st.warm = st.warm.saturating_add(1);
}

/// Whether this eye's baseline has settled enough to capture a target.
pub fn warmed(st: &BrightState) -> bool {
    st.init && st.warm >= WARMUP
}

/// This eye's current baseline (level, spread) — the value latched as the target.
pub fn capture(st: &BrightState) -> (f32, f32) {
    (st.base_level, st.base_spread)
}

/// The affine `(a, b)` mapping the current baseline onto `(tgt_level, tgt_spread)`, blended
/// by `strength`. `out = a*in + b` (identity at strength 0). Gain is clamped so a bad
/// estimate can't blow the image out.
pub fn affine(st: &BrightState, tgt_level: f32, tgt_spread: f32, strength: f32) -> (f32, f32) {
    // A corrupt persisted target/strength could be non-finite; fall back to identity so it
    // can never NaN the ML input.
    if !tgt_level.is_finite() || !tgt_spread.is_finite() {
        return (1.0, 0.0);
    }
    let gain = (tgt_spread / st.base_spread.max(1.0)).clamp(0.3, 3.0);
    let off = tgt_level - gain * st.base_level;
    let s = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 1.0 };
    (1.0 + s * (gain - 1.0), s * off)
}

/// Apply an affine `(a, b)` to a grayscale frame: `out = clamp(a*in + b, 0, 255)`. Returns
/// a plain copy for the identity `(1, 0)`.
pub fn apply(frame: &[u8], a: f32, b: f32) -> Vec<u8> {
    if (a - 1.0).abs() < 1e-4 && b.abs() < 1e-3 {
        return frame.to_vec();
    }
    frame.iter().map(|&v| (a * v as f32 + b).clamp(0.0, 255.0) as u8).collect()
}

/// One ML-frame step for BOTH eyes: update baselines, capture the target from GOOD frames,
/// and return the per-eye affine `[(aL,bL),(aR,bR)]`. `prev_open` is the PREVIOUS frame's raw
/// model openness per eye (s[1]/s[2]); it gates the capture so the reference is a genuinely
/// relaxed-open frame, not a blind post-warmup snapshot. `norm` is the shared params+target
/// (mutated on capture). Identity per eye until a target exists / when disabled.
pub fn step(
    states: &mut [BrightState; 2],
    frames: [(&[u8], usize, usize); 2],
    prev_open: [f32; 2],
    norm: &mut BrightnessNorm,
) -> [(f32, f32); 2] {
    if !norm.enabled {
        return [(1.0, 0.0), (1.0, 0.0)];
    }
    for i in 0..2 {
        let (f, w, h) = frames[i];
        update(&mut states[i], f, w, h, norm.adapt);
        // Track the eye's own openness peak (slow decay) and flag GOOD capture frames:
        // OPEN (openness within 80% of its peak -> not a blink) AND relaxed brightness
        // (in-band -> not a wide/wink). Both eyes must stay GOOD for GOOD_CAP frames.
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
        // `omax >= 0.5` = the eye has genuinely reached an OPEN level at some point (the
        // model's raw openness is ~0.66 open / ~0.39 closed), so an all-closed session can't
        // capture a closed reference; `o > 0.8*omax` = currently near that open peak.
        let good = warmed(st) && st.in_band && o.is_finite() && st.omax >= 0.5 && o > 0.8 * st.omax;
        st.good_streak = if good { st.good_streak.saturating_add(1) } else { 0 };
    }
    // Capture ONCE, only after BOTH eyes have been relaxed-open for a sustained stretch.
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
            out[i] = affine(&states[i], norm.tgt_level[i], norm.tgt_spread[i], norm.strength);
        }
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
        assert!((l - 100.0).abs() < 1.0 && s <= 1.0, "flat: level~100 spread~0, got {l},{s}");
    }

    #[test]
    fn affine_preserves_darkening() {
        // With a captured baseline, the baseline maps ~to the target level, and a DARKER
        // frame (a blink) maps below it — the affine never cancels the darkening.
        let st = BrightState { base_level: 160.0, base_spread: 40.0, warm: 200, init: true, ..Default::default() };
        let (a, b) = affine(&st, 120.0, 40.0, 1.0);
        let at_base = a * 160.0 + b;
        assert!((at_base - 120.0).abs() < 1.0, "baseline maps onto target, got {at_base}");
        let at_dark = a * 80.0 + b;
        assert!(at_dark < at_base - 30.0, "darker input stays clearly darker, got {at_dark}");
    }

    #[test]
    fn dim_frame_is_brightened_toward_target() {
        // A user whose eye is systematically DIM (level ~90) gets pulled up toward a
        // brighter target (~140), i.e. gain>1 / positive offset.
        let st = BrightState { base_level: 90.0, base_spread: 30.0, warm: 200, init: true, ..Default::default() };
        let (a, b) = affine(&st, 140.0, 40.0, 1.0);
        let mapped = a * 90.0 + b;
        assert!(mapped > 130.0, "dim baseline brightened toward target, got {mapped}");
    }

    #[test]
    fn baseline_resists_held_expression() {
        // The regression fix: after a relaxed baseline settles, HOLDING a wink (much darker)
        // must barely move the baseline — otherwise the affine brightens the closed eye and
        // openness spikes up while shut.
        let mut st = BrightState::default();
        let mut relaxed = vec![0u8; 200 * 200];
        for (i, p) in relaxed.iter_mut().enumerate() {
            *p = ((i % 200) as u8 / 2).wrapping_add(90);
        }
        for _ in 0..250 {
            update(&mut st, &relaxed, 200, 200, 0.05);
        }
        let base_before = st.base_level;
        let dark = vec![30u8; 200 * 200]; // held wink (much darker than baseline)
        for _ in 0..180 {
            update(&mut st, &dark, 200, 200, 0.05);
        }
        let drift = (st.base_level - base_before).abs();
        assert!(drift < 6.0, "a held wink must not drag the baseline (drifted {drift})");
    }

    #[test]
    fn step_captures_only_from_good_open_frames() {
        let mut norm = BrightnessNorm { enabled: true, ..BrightnessNorm::default() };
        let mut st = [BrightState::default(); 2];
        // Steady gradient-ish frame so spread is non-degenerate (relaxed brightness = in-band).
        let mut frame = vec![0u8; 200 * 200];
        for (i, p) in frame.iter_mut().enumerate() {
            *p = ((i % 200) as u8).wrapping_add(40);
        }
        let f = [(&frame[..], 200usize, 200usize), (&frame[..], 200usize, 200usize)];
        // (a) Warmed but eyes CLOSED (low openness) -> must NOT capture.
        for _ in 0..WARMUP + GOOD_CAP + 20 {
            step(&mut st, f, [0.2, 0.2], &mut norm);
        }
        assert!(!norm.captured, "closed eyes must not trigger a capture");
        // (b) Now the eyes are OPEN (high openness) for a sustained stretch -> capture.
        let mut last = [(1.0, 0.0); 2];
        for _ in 0..GOOD_CAP + 5 {
            last = step(&mut st, f, [0.8, 0.8], &mut norm);
        }
        assert!(norm.captured, "target captured from sustained good open frames");
        // Right after capture, baseline == target, so the affine is ~identity.
        assert!((last[0].0 - 1.0).abs() < 0.15 && last[0].1.abs() < 8.0, "near-identity at capture: {:?}", last[0]);
    }
}
