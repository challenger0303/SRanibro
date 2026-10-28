//! The legacy motion-adaptive gaze EMA, shared by avatar and optional Gaze OSC
//! smoothing. Callers own independent per-eye history and validity policies.

pub const INITIAL_GAZE: [f32; 3] = [0.0, 0.0, -1.0];

#[inline]
pub fn adaptive_gaze_step([px, py, pz]: [f32; 3], [gx, gy, gz]: [f32; 3]) -> [f32; 3] {
    let delta = ((gx - px).powi(2) + (gy - py).powi(2) + (gz - pz).powi(2)).sqrt();
    let alpha = (delta * 3.0).clamp(0.08, 0.85);
    [
        px + alpha * (gx - px),
        py + alpha * (gy - py),
        pz + alpha * (gz - pz),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaze_smoothing_is_bit_identical_to_the_legacy_formula() {
        let mut previous = INITIAL_GAZE;
        for i in 0..1000 {
            let t = i as f32 * 0.019;
            let input = [t.sin() * 0.6, t.cos() * 0.2, 0.8];
            let [gx, gy, gz] = input;
            let [px, py, pz] = previous;
            let delta = ((gx - px).powi(2) + (gy - py).powi(2) + (gz - pz).powi(2)).sqrt();
            let alpha_g = (delta * 3.0).clamp(0.08, 0.85);
            let expected = [
                px + alpha_g * (gx - px),
                py + alpha_g * (gy - py),
                pz + alpha_g * (gz - pz),
            ];
            let actual = adaptive_gaze_step(previous, input);
            assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
            previous = actual;
        }
    }
}
