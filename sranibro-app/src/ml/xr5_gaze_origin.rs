//! Independent reported-origin sharing. Never selects or modifies gaze direction.
//! Device-local mm, not a measured corneal center or validated world-space pose.
use crate::core::types::{EyeResult, GazeSample};
use std::time::{Duration, Instant};

const FRESH: Duration = Duration::from_millis(30);
const HOLD: Duration = Duration::from_millis(250);

#[derive(Clone, Copy)]
struct Point {
    raw: [f32; 3],
    at: Instant,
}

#[derive(Clone, Copy, Default)]
pub struct Origin {
    pub point_mm: Option<[f32; 3]>,
    pub received: Option<Instant>,
    pub fresh: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin_mm: [f32; 3],
    pub direction: [f32; 3],
    pub origin_fresh: bool,
    pub origin_age_ms: f64,
    pub partner_estimated: bool,
}

#[derive(Default)]
pub struct Tracker {
    latest: [Option<Point>; 2],
    valid: [bool; 2],
    timestamp: u64,
    mapping: Option<bool>,
}

fn age(at: Instant, now: Instant, limit: Duration) -> Option<Duration> {
    now.checked_duration_since(at).filter(|d| *d <= limit)
}
fn valid_point(point: [f32; 3]) -> bool {
    point.iter().all(|v| v.is_finite() && v.abs() <= 1000.)
}

impl Tracker {
    /// `sample` is original wearable data routed to physical L/R, never merged
    /// last-good values. A changed duplicate can invalidate but cannot refresh.
    pub fn push(&mut self, sample: GazeSample, at: Instant, swapped: bool) {
        if sample.timestamp_us == 0 {
            return;
        }
        if self.mapping != Some(swapped) || sample.timestamp_us < self.timestamp {
            *self = Self::default();
            self.mapping = Some(swapped);
        }
        let eyes = [sample.left, sample.right];
        if sample.timestamp_us == self.timestamp {
            for (i, e) in eyes.iter().enumerate() {
                self.valid[i] &= e.origin_valid
                    && valid_point(e.origin_mm)
                    && self.latest[i].is_some_and(|p| p.raw == e.origin_mm);
            }
            return;
        }
        self.timestamp = sample.timestamp_us;
        for (i, e) in eyes.iter().enumerate() {
            self.valid[i] = e.origin_valid && valid_point(e.origin_mm);
            if self.valid[i] {
                self.latest[i] = Some(Point {
                    raw: e.origin_mm,
                    at,
                });
            }
        }
    }

    pub fn snapshot(&self, now: Instant, flip_x: bool) -> [Origin; 2] {
        std::array::from_fn(|i| {
            let Some(p) = self.latest[i].filter(|p| age(p.at, now, HOLD).is_some()) else {
                return Origin::default();
            };
            let [x, y, z] = p.raw;
            Origin {
                point_mm: Some([if flip_x { x } else { -x }, y, -z]),
                received: Some(p.at),
                fresh: self.valid[i] && age(p.at, now, FRESH).is_some(),
            }
        })
    }

    /// Builds optional rays from already selected/processed output. Does not
    /// mutate eye state, smoothing, vergence, eyelids or OSC output preferences.
    pub fn rays(&self, eyes: &[EyeResult; 2], now: Instant, flip_x: bool) -> [Option<Ray>; 2] {
        let origins = self.snapshot(now, flip_x);
        std::array::from_fn(|i| {
            let e = &eyes[i];
            if !e.gaze_valid || !e.gaze.iter().all(|v| v.is_finite()) || e.gaze[2] <= 0. {
                return None;
            }
            let norm = e.gaze[0].hypot(e.gaze[1]).hypot(e.gaze[2]);
            if !(0.5..=1.5).contains(&norm) {
                return None;
            }
            let o = origins[i];
            Some(Ray {
                origin_mm: o.point_mm?,
                direction: [-e.gaze[0] / norm, e.gaze[1] / norm, -e.gaze[2] / norm],
                origin_fresh: o.fresh,
                origin_age_ms: age(o.received?, now, HOLD)?.as_secs_f64() * 1000.,
                partner_estimated: e.gaze_yoked,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Eye, EyeSample};
    fn sample(ts: u64) -> GazeSample {
        let eye = |x| EyeSample {
            origin_mm: [x, 2., 10.],
            origin_valid: true,
            ..Default::default()
        };
        GazeSample {
            timestamp_us: ts,
            left: eye(30.),
            right: eye(-32.),
        }
    }
    fn eyes() -> [EyeResult; 2] {
        let mut eyes = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for e in &mut eyes {
            e.gaze_valid = true;
            e.gaze = [0.2, 0.1, 0.97];
            e.openness = 0.42;
            e.wide = 0.8;
        }
        eyes
    }
    #[test]
    fn works_without_model_gaze_direction_or_pupil_flags() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(1), at, false);
        let s = t.snapshot(at, true);
        assert!(s.iter().all(|o| o.fresh));
        assert_eq!(s[0].point_mm, Some([30., 2., -10.]));
    }
    #[test]
    fn invalid_origin_is_estimated_only_then_expires() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(1), at, false);
        let mut s = sample(2);
        s.left.origin_valid = false;
        t.push(s, at + Duration::from_millis(8), false);
        let o = t.snapshot(at + Duration::from_millis(8), true);
        assert!(!o[0].fresh);
        assert!(o[1].fresh);
        assert!(t.snapshot(at + Duration::from_millis(251), true)[0]
            .point_mm
            .is_none());
    }
    #[test]
    fn duplicate_invalidity_no_refresh_or_resurrection() {
        let at = Instant::now();
        let mut t = Tracker::default();
        let mut s = sample(1);
        t.push(s, at, false);
        s.left.origin_valid = false;
        t.push(s, at + Duration::from_millis(8), false);
        s.left.origin_valid = true;
        t.push(s, at + Duration::from_millis(9), false);
        assert!(!t.snapshot(at + Duration::from_millis(9), true)[0].fresh);
        assert_eq!(
            t.snapshot(at + Duration::from_millis(9), true)[0].received,
            Some(at)
        );
    }
    #[test]
    fn changed_duplicate_point_is_not_a_measurement() {
        let at = Instant::now();
        let mut t = Tracker::default();
        let mut s = sample(1);
        t.push(s, at, false);
        s.left.origin_mm[0] = 99.;
        t.push(s, at + Duration::from_millis(8), false);
        let o = t.snapshot(at + Duration::from_millis(8), true)[0];
        assert!(!o.fresh);
        assert_eq!(o.point_mm, Some([30., 2., -10.]));
    }
    #[test]
    fn clock_and_mapping_reset_discard_old_origins() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(10), at, false);
        let mut s = sample(1);
        s.left.origin_valid = false;
        t.push(s, at + Duration::from_millis(8), false);
        assert!(t.snapshot(at + Duration::from_millis(8), true)[0]
            .point_mm
            .is_none());
        t.push(sample(2), at + Duration::from_millis(9), false);
        s.timestamp_us = 3;
        t.push(s, at + Duration::from_millis(10), true);
        assert!(t.snapshot(at + Duration::from_millis(10), true)[0]
            .point_mm
            .is_none());
    }
    #[test]
    fn future_zero_timestamp_and_bad_numbers_are_not_fresh() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(0), at, false);
        assert!(t.snapshot(at, true)[0].point_mm.is_none());
        t.push(sample(1), at + Duration::from_millis(1), false);
        assert!(t.snapshot(at, true)[0].point_mm.is_none());
        for bad in [f32::NAN, f32::INFINITY, 2000.] {
            let mut t = Tracker::default();
            let mut s = sample(1);
            s.left.origin_mm[0] = bad;
            t.push(s, at, false);
            assert!(t.snapshot(at, true)[0].point_mm.is_none());
        }
    }
    #[test]
    fn handedness_eye_identity_and_normalization_preserve_output() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(1), at, false);
        let e = eyes();
        let before = e.map(|e| (e.gaze, e.openness, e.wide));
        let r = t.rays(&e, at, true);
        assert_eq!(r[0].unwrap().origin_mm, [30., 2., -10.]);
        assert_eq!(r[1].unwrap().origin_mm, [-32., 2., -10.]);
        let d = r[0].unwrap().direction;
        assert!((d[0].hypot(d[1]).hypot(d[2]) - 1.).abs() < 1e-6);
        assert_eq!(e.map(|e| (e.gaze, e.openness, e.wide)), before);
        assert_eq!(
            t.rays(&e, at, false)[0].unwrap().origin_mm,
            [-30., 2., -10.]
        );
    }
    #[test]
    fn invalid_direction_does_not_create_a_ray_and_partner_is_explicit() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(1), at, false);
        let mut e = eyes();
        e[0].gaze_valid = false;
        e[1].gaze_yoked = true;
        let r = t.rays(&e, at, true);
        assert!(r[0].is_none());
        assert!(r[1].unwrap().partner_estimated);
    }
    #[test]
    fn stale_and_held_origins_are_not_refreshed_by_polling() {
        let at = Instant::now();
        let mut t = Tracker::default();
        t.push(sample(1), at, false);
        let o = t.snapshot(at + Duration::from_millis(31), true)[0];
        assert!(!o.fresh);
        assert_eq!(o.received, Some(at));
        assert!(
            !t.rays(&eyes(), at + Duration::from_millis(40), true)[0]
                .unwrap()
                .origin_fresh
        );
        assert!(t.rays(&eyes(), at + Duration::from_millis(251), true)[0].is_none());
    }
}
