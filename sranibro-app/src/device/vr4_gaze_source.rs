//! A single, connection-scoped gaze provider for Crystal / Crystal Super.
//! Both avatar input and the independent OSC tap use the selected source; there
//! is no fallback on tracking loss, no combined gaze and no extra smoothing.

use super::gaze::{GazeData, WearableData};
use crate::config::Vr4GazeSource;
use crate::core::types::GazeSample;

pub struct Vr4GazeRouter {
    source: Vr4GazeSource,
    last_timestamp: u64,
}

impl Vr4GazeRouter {
    pub fn new(source: Vr4GazeSource) -> Self {
        Self {
            source,
            last_timestamp: 0,
        }
    }

    /// v1 never publishes gaze from 1289, even when 1285 tracking is invalid.
    pub fn foveated(&self, data: &GazeData) -> Option<GazeSample> {
        (self.source == Vr4GazeSource::V2).then(|| data.to_gaze_sample())
    }

    /// First result is the unfiltered native tap, second is avatar input.
    /// v2 still receives auxiliary pupil/openness, but no duplicate gaze event.
    pub fn wearable(&mut self, data: &WearableData) -> (GazeSample, GazeSample) {
        if self.source == Vr4GazeSource::V2 {
            let aux = data.to_aux_sample();
            return (aux, aux);
        }
        let raw = data.to_gaze_sample();
        let mut tracking = raw;
        let fresh = tracking.timestamp_us > self.last_timestamp;
        if fresh {
            self.last_timestamp = tracking.timestamp_us;
        }
        // Match the tested direct trial: guard avatar freshness/invalid rays,
        // but never turn a missing eye into its partner or limit its gaze range.
        for eye in [&mut tracking.left, &mut tracking.right] {
            let ray = eye.gaze;
            let norm_sq = ray.iter().map(|v| v * v).sum::<f32>();
            eye.gaze_valid &= fresh
                && ray.iter().all(|v| v.is_finite())
                && ray[2] > 0.1
                && (0.25..=2.25).contains(&norm_sq);
        }
        (raw, tracking)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native() -> WearableData {
        WearableData {
            timestamp_us: 100,
            gaze: [[0.64278764, 0.0, 0.76604444], [-0.6, 0.0, 0.8]],
            gaze_valid: [true; 2],
            pupil_mm: [3.2, 3.7],
            pupil_valid: [true; 2],
            openness: [0.0, 1.0],
            openness_valid: [true; 2],
            openness_reported: [true; 2],
            ..Default::default()
        }
    }

    fn legacy() -> GazeData {
        GazeData {
            timestamp: 100,
            l_gaze: [0.5, 0.0, 0.8660254],
            r_gaze: [-0.5, 0.0, 0.8660254],
            combined: [0.0, 0.0, 1.0],
            ..Default::default()
        }
    }

    #[test]
    fn v1_uses_only_native_independent_eyes_without_scaling_or_smoothing() {
        let mut router = Vr4GazeRouter::new(Vr4GazeSource::V1);
        assert!(router.foveated(&legacy()).is_none());
        let source = native();
        let (raw, tracking) = router.wearable(&source);
        for (i, eye) in [tracking.left, tracking.right].into_iter().enumerate() {
            assert_eq!(eye.gaze, source.gaze[i]);
            assert!(eye.gaze_valid && eye.gaze_reported);
            assert_eq!(eye.pupil_mm, source.pupil_mm[i]);
            assert_eq!(eye.openness, source.openness[i]);
        }
        assert_eq!(raw.left.gaze, tracking.left.gaze);
        let mut closed = source;
        closed.timestamp_us += 1;
        closed.gaze_valid[0] = false;
        let (raw, tracking) = router.wearable(&closed);
        assert!(!raw.left.gaze_valid && !tracking.left.gaze_valid);
        assert!(tracking.right.gaze_valid);
        assert!(
            router.foveated(&legacy()).is_none(),
            "no fallback while v1 is invalid"
        );
    }

    #[test]
    fn v2_retains_original_per_eye_rays_and_auxiliary_only_wearable_data() {
        let mut router = Vr4GazeRouter::new(Vr4GazeSource::V2);
        let expected = legacy().to_gaze_sample();
        let gaze = router.foveated(&legacy()).unwrap();
        assert_eq!(gaze.left.gaze, expected.left.gaze);
        assert_eq!(gaze.right.gaze, expected.right.gaze);
        assert_ne!(gaze.left.gaze, gaze.right.gaze);
        for sample in {
            let (raw, tracking) = router.wearable(&native());
            [raw, tracking]
        } {
            assert_eq!(sample.timestamp_us, 0);
            assert!(!sample.left.gaze_reported && !sample.right.gaze_reported);
            assert!(!sample.left.gaze_valid && !sample.right.gaze_valid);
            assert_eq!(sample.left.pupil_mm, 3.2);
            assert_eq!(sample.right.openness, 1.0);
        }
    }

    #[test]
    fn v1_rejects_replays_zero_and_nonfinite_but_never_changes_native_tap() {
        let mut router = Vr4GazeRouter::new(Vr4GazeSource::V1);
        for (timestamp, ray, expected) in [
            (0, [0.6, 0.0, 0.8], false),
            (10, [0.6, 0.0, 0.8], true),
            (10, [0.6, 0.0, 0.8], false),
            (9, [0.6, 0.0, 0.8], false),
            (11, [0.0; 3], false),
            (12, [f32::NAN, 0.0, 1.0], false),
            (13, [0.0, 0.0, -1.0], false),
            (14, [0.64278764, 0.0, 0.76604444], true),
        ] {
            let mut sample = native();
            sample.timestamp_us = timestamp;
            sample.gaze[0] = ray;
            let (raw, tracking) = router.wearable(&sample);
            assert_eq!(tracking.left.gaze_valid, expected);
            assert_eq!(raw.left.gaze.map(f32::to_bits), ray.map(f32::to_bits));
            assert!(raw.left.gaze_valid);
            assert_eq!(tracking.left.pupil_mm, raw.left.pupil_mm);
        }
        // A newly constructed connection has no stale timestamp/provider state.
        assert!(
            Vr4GazeRouter::new(Vr4GazeSource::V1)
                .wearable(&native())
                .1
                .left
                .gaze_valid
        );
    }
}
