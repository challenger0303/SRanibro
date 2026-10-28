//! Experimental stationary Tobii anchor. Receipt times are NOT exposure times.
//! Never used outside opt-in XR5 model gaze; never changes expression outputs.
use super::xr5_gaze::Snapshot;
use crate::core::types::GazeSample;
use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

const N: usize = 64;
#[derive(Clone, Copy)]
struct Native {
    sample: GazeSample,
    at: Instant,
    seq: u64,
}
#[derive(Default)]
pub struct NativeBuffer {
    queue: VecDeque<Native>,
    timestamp: u64,
    sequence: u64,
    epoch: u64,
}
impl NativeBuffer {
    pub fn push(&mut self, sample: GazeSample, at: Instant) {
        if !(sample.left.gaze_reported || sample.right.gaze_reported) {
            return;
        }
        if sample.timestamp_us == 0 || sample.timestamp_us <= self.timestamp {
            if sample.timestamp_us < self.timestamp {
                self.queue.clear();
                self.epoch = self.epoch.wrapping_add(1);
                self.timestamp = sample.timestamp_us;
            }
            return;
        }
        self.timestamp = sample.timestamp_us;
        self.sequence = self.sequence.wrapping_add(1);
        if self.queue.len() == N {
            self.queue.pop_front();
        }
        self.queue.push_back(Native {
            sample,
            at,
            seq: self.sequence,
        });
    }
    fn nearest(&self, at: Instant, now: Instant) -> Option<Native> {
        self.queue
            .iter()
            .filter(|n| {
                now.checked_duration_since(n.at)
                    .is_some_and(|d| d <= Duration::from_millis(75))
            })
            .min_by_key(|n| n.at.max(at).duration_since(n.at.min(at)))
            .copied()
            .filter(|n| n.at.max(at).duration_since(n.at.min(at)) <= Duration::from_millis(25))
    }
}
fn angles(v: [f32; 3]) -> Option<[f64; 2]> {
    let [x, y, z] = v.map(f64::from);
    let norm = x.hypot(y).hypot(z);
    if !v.iter().all(|v| v.is_finite()) || !(0.99..=1.01).contains(&norm) || z >= 0. {
        return None;
    }
    Some([x.atan2(-z).to_degrees(), y.atan2(x.hypot(z)).to_degrees()])
}
fn ray(a: [f64; 2]) -> [f32; 3] {
    let (y, p) = (a[0].to_radians(), a[1].to_radians());
    [
        (y.sin() * p.cos()) as f32,
        p.sin() as f32,
        (-y.cos() * p.cos()) as f32,
    ]
}
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
#[derive(Clone, Copy)]
struct Pair {
    time: f64,
    m: [f64; 2],
    n: [f64; 2],
}
#[derive(Clone, Copy)]
struct Fixation {
    time: f64,
    m: [f64; 2],
    n: [f64; 2],
    bias: [f64; 2],
}
#[derive(Default)]
struct EyeAnchor {
    pairs: VecDeque<Pair>,
    reference: Option<Fixation>,
    corroborated: Option<f64>,
    ids: [u64; 2],
    last_time: Option<f64>,
    bias: [f64; 2],
    updated: Option<f64>,
    diagnostic: Diagnostic,
}
#[derive(Clone, Copy, Debug)]
pub struct Diagnostic {
    pub gate: &'static str,
    pub samples: usize,
    pub span: f64,
    pub spread: [f64; 2],
    pub residual: f64,
    pub offset: f64,
    pub movement: [f64; 2],
    pub agreement: f64,
}
impl Default for Diagnostic {
    fn default() -> Self {
        Self {
            gate: "collecting",
            samples: 0,
            span: 0.,
            spread: [0.; 2],
            residual: 0.,
            offset: 0.,
            movement: [0.; 2],
            agreement: 0.,
        }
    }
}
impl EyeAnchor {
    fn feed(&mut self, now: f64, ids: [u64; 2], m: [f64; 2], n: [f64; 2]) -> bool {
        if ids[0] <= self.ids[0] || ids[1] <= self.ids[1] {
            return false;
        }
        let dt = self
            .last_time
            .map_or(0., |t| (now - t).clamp(0., 1. / 120.));
        if self.last_time.is_some_and(|t| now - t > 0.1 || now <= t) {
            self.pairs.clear();
            // A blink or short observation loss must not erase the first pose.
            // Never combine pre-gap samples or carry update permission across it.
            if self.last_time.is_some_and(|t| now - t > 1. || now <= t) {
                self.reference = None;
            }
            self.corroborated = None;
        }
        self.last_time = Some(now);
        self.ids = ids;
        if self.pairs.len() == N {
            self.pairs.pop_front();
        }
        self.pairs.push_back(Pair { time: now, m, n });
        self.diagnostic = Diagnostic {
            samples: self.pairs.len(),
            span: now - self.pairs[0].time,
            ..Default::default()
        };
        if self.pairs.len() != N {
            return false;
        }
        let span = now - self.pairs[0].time;
        if !(0.5..=2.).contains(&span) {
            self.diagnostic.gate = "window duration";
            return false;
        }
        let mut center = Fixation {
            time: now,
            m: [0.; 2],
            n: [0.; 2],
            bias: [0.; 2],
        };
        let (mut ms, mut ns) = ([0.; 2], [0.; 2]);
        for axis in 0..2 {
            let mut ma = [0.; N];
            let mut na = [0.; N];
            let mut diff = [0.; N];
            for (i, p) in self.pairs.iter().enumerate() {
                ma[i] = p.m[axis];
                na[i] = p.n[axis];
                diff[i] = na[i] - ma[i];
            }
            ma.sort_unstable_by(f64::total_cmp);
            na.sort_unstable_by(f64::total_cmp);
            diff.sort_unstable_by(f64::total_cmp);
            ms[axis] = ma[N - 1] - ma[0];
            ns[axis] = na[N - 1] - na[0];
            self.diagnostic.residual = self.diagnostic.residual.max(diff[57] - diff[6]);
            center.m[axis] = (ma[31] + ma[32]) * 0.5;
            center.n[axis] = (na[31] + na[32]) * 0.5;
            center.bias[axis] = (diff[31] + diff[32]) * 0.5;
        }
        self.diagnostic.spread = [ms[0].hypot(ms[1]), ns[0].hypot(ns[1])];
        self.diagnostic.offset = dist(center.bias, [0.; 2]);
        self.diagnostic.gate = if self.diagnostic.residual > 1.5 {
            "offset unstable"
        } else if self.diagnostic.spread[0] > 0.8 {
            "model moving"
        } else if self.diagnostic.spread[1] > 0.8 {
            "Tobii moving"
        } else if self.diagnostic.offset > 12. {
            "offset too large"
        } else {
            "second position needed"
        };
        if self.diagnostic.residual > 1.5
            || self.diagnostic.spread.iter().any(|x| *x > 0.8)
            || self.diagnostic.offset > 12.
        {
            return false;
        }
        match self.reference {
            Some(old) if now - old.time <= 10. => {
                self.diagnostic.movement = [dist(old.m, center.m), dist(old.n, center.n)];
                self.diagnostic.agreement = dist(old.bias, center.bias);
                if dist(old.m, center.m) >= 3. && dist(old.n, center.n) >= 3. {
                    if dist(old.bias, center.bias) <= 1.5 {
                        self.corroborated = Some(now);
                    } else {
                        self.diagnostic.gate = "positions disagree";
                    }
                    self.reference = Some(center);
                }
            }
            _ => self.reference = Some(center),
        }
        if !self.corroborated.is_some_and(|t| now - t <= 10.) {
            return false;
        }
        let delta = dist(center.bias, self.bias);
        let alpha = if delta > 0. { (dt / delta).min(1.) } else { 0. };
        for k in 0..2 {
            self.bias[k] += alpha * (center.bias[k] - self.bias[k]);
        }
        self.updated = Some(now);
        self.diagnostic.gate = "anchoring";
        true
    }
}
#[derive(Clone, Default)]
pub struct Status {
    pub message: String,
    pub bias: [[f64; 2]; 2],
    pub updating: [bool; 2],
    pub native_fresh: bool,
    pub diagnostics: [Diagnostic; 2],
}
pub struct Hybrid {
    start: Instant,
    eyes: [EyeAnchor; 2],
    context: Option<(u64, bool)>,
}
impl Default for Hybrid {
    fn default() -> Self {
        Self {
            start: Instant::now(),
            eyes: Default::default(),
            context: None,
        }
    }
}
impl Hybrid {
    pub fn apply(
        &mut self,
        model: &Snapshot,
        native: &NativeBuffer,
        now: Instant,
        flip_x: bool,
    ) -> (Snapshot, Status) {
        let context = (native.epoch, flip_x);
        if self.context != Some(context) {
            let keep_bias = self.context.is_some_and(|(_, old_flip)| old_flip == flip_x);
            for eye in &mut self.eyes {
                // A native clock reset may be a disconnect: discard its evidence,
                // not the already displayed offset. Explicit mapping/reload resets.
                let (bias, updated) = (eye.bias, eye.updated);
                *eye = Default::default();
                if keep_bias {
                    eye.bias = bias;
                    eye.updated = updated;
                }
            }
            self.context = Some(context);
        }
        let mut selected = model.clone();
        let mut status = Status::default();
        let Some(at) = model.captured.filter(|at| {
            now.checked_duration_since(*at)
                .is_some_and(|d| d <= Duration::from_millis(50))
        }) else {
            status.message = "Hybrid · model unavailable".into();
            status.bias = self.eyes.each_ref().map(|e| e.bias);
            status.diagnostics = [Diagnostic {
                gate: "model stale",
                ..Default::default()
            }; 2];
            return (selected, status);
        };
        let candidate = native.nearest(at, now);
        let t = at.saturating_duration_since(self.start).as_secs_f64();
        for eye in 0..2 {
            status.diagnostics[eye] = self.eyes[eye].diagnostic;
            if model.states[eye] != 1 && model.states[eye] != 2 {
                status.diagnostics[eye].gate = "model unavailable";
                continue;
            }
            let Some(a) = angles(model.directions[eye]) else {
                status.diagnostics[eye].gate = "model ray invalid";
                selected.states[eye] = 0;
                selected.directions[eye] = [0.; 3];
                continue;
            };
            if model.states[eye] == 1 {
                status.diagnostics[eye].gate = "no timed Tobii pair";
                if let Some(n) = candidate {
                    status.diagnostics[eye].gate = "Tobii invalid";
                    let e = if eye == 0 {
                        n.sample.left
                    } else {
                        n.sample.right
                    };
                    // Model uses raw camera eye identity; do not apply legacy stream swap.
                    // Match native handedness then convert the app's axes to OpenVR.
                    let v = [
                        if flip_x { e.gaze[0] } else { -e.gaze[0] },
                        e.gaze[1],
                        -e.gaze[2],
                    ];
                    if e.gaze_reported && e.gaze_valid {
                        if let Some(b) = angles(v) {
                            status.native_fresh = true;
                            status.updating[eye] =
                                self.eyes[eye].feed(t, [model.count, n.seq], a, b);
                            status.diagnostics[eye] = self.eyes[eye].diagnostic;
                        }
                    }
                }
            } else {
                status.diagnostics[eye].gate = "partner-estimated eye";
            }
            let bias = self.eyes[eye].bias;
            // Zero correction preserves model bits. Estimated eyes never teach.
            if bias != [0.; 2] {
                let adjusted = [a[0] + bias[0], a[1] + bias[1]];
                if adjusted.iter().any(|x| x.abs() >= 89.) {
                    selected.states[eye] = 0;
                    selected.directions[eye] = [0.; 3];
                } else {
                    selected.directions[eye] = ray(adjusted);
                }
            }
        }
        status.bias = self.eyes.each_ref().map(|e| e.bias);
        let learned = self.eyes.iter().any(|e| e.updated.is_some());
        status.message = if status.updating.iter().any(|x| *x) {
            "Hybrid · anchoring"
        } else if learned {
            "Hybrid · holding correction"
        } else if status.native_fresh {
            "Hybrid · look at two positions, pause at each"
        } else {
            "Hybrid · waiting for usable pair"
        }
        .into();
        selected.message = status.message.clone();
        (selected, status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(stuck: bool) -> EyeAnchor {
        let mut e = EyeAnchor::default();
        for i in 1..2400u64 {
            let x = if (i / 240) % 2 == 0 { 0. } else { 6. };
            let n = if stuck { 4. } else { x + 4. };
            e.feed(i as f64 / 120., [i, i], [x, 0.], [n, 1.]);
        }
        e
    }
    #[test]
    fn two_fixations_acquire() {
        let e = run(false);
        assert!(dist(e.bias, [4., 1.]) < 1e-8);
    }
    #[test]
    fn constant_native_never_acquires() {
        assert_eq!(run(true).bias, [0.; 2]);
    }
    #[test]
    fn one_stationary_pose_never_acquires() {
        let mut e = EyeAnchor::default();
        for i in 1..600 {
            e.feed(i as f64 / 120., [i, i], [0., 0.], [4., 0.]);
        }
        assert_eq!(e.bias, [0.; 2]);
    }
    #[test]
    fn rate_and_duplicates() {
        let mut e = EyeAnchor::default();
        let mut old = [0.; 2];
        for i in 1..1600 {
            let t = i as f64 / 120.;
            let x = if i < 240 { 0. } else { 6. };
            e.feed(t, [i, i], [x, 0.], [x + 4., 1.]);
            assert!(dist(e.bias, old) <= 1. / 120. + 1e-10);
            old = e.bias;
            assert!(!e.feed(t, [i, i], [x, 0.], [x + 4., 1.]));
            assert_eq!(e.bias, old);
        }
    }
    fn sample(ts: u64, x: f64) -> GazeSample {
        let d = ray([x, 0.]);
        let e = crate::core::types::EyeSample {
            gaze: [d[0], d[1], -d[2]],
            gaze_valid: true,
            gaze_reported: true,
            ..Default::default()
        };
        GazeSample {
            timestamp_us: ts,
            left: e,
            right: e,
        }
    }
    #[test]
    fn native_aux_duplicate_and_regression() {
        let mut b = NativeBuffer::default();
        let t = Instant::now();
        b.push(sample(10, 0.), t);
        b.push(sample(10, 4.), t);
        assert_eq!(b.queue.len(), 1);
        b.push(
            GazeSample {
                timestamp_us: 11,
                ..Default::default()
            },
            t,
        );
        assert_eq!(b.queue.len(), 1);
        b.push(sample(5, 0.), t);
        assert!(b.queue.is_empty());
        assert_eq!(b.epoch, 1);
    }
    #[test]
    fn buffer_bounded_and_no_future_or_stale() {
        let t = Instant::now();
        let mut b = NativeBuffer::default();
        for i in 1..1000 {
            b.push(sample(i, 0.), t);
        }
        assert_eq!(b.queue.len(), 64);
        assert!(b.nearest(t, t - Duration::from_millis(1)).is_none());
        assert!(b.nearest(t, t + Duration::from_millis(100)).is_none());
    }
    #[test]
    fn loss_and_estimates_never_train_preserve_bias() {
        let now = Instant::now();
        let mut h = Hybrid::default();
        let b = NativeBuffer::default();
        h.context = Some((0, true));
        h.eyes[0] = run(false);
        let bias = h.eyes[0].bias;
        let m = Snapshot {
            count: 10,
            states: [2, 0],
            directions: [ray([0., 0.]), [0.; 3]],
            captured: Some(now),
            ..Default::default()
        };
        let (o, s) = h.apply(&m, &b, now, true);
        assert_eq!(s.bias[0], bias);
        assert_eq!(o.states, [2, 0]);
        assert_eq!(s.updating, [false; 2]);
        let mut missing = m;
        missing.states = [0; 2];
        let (o, _) = h.apply(&missing, &b, now, true);
        assert_eq!(o.states, [0; 2]);
    }
    #[test]
    fn zero_bias_bit_exact_and_context_reset() {
        let now = Instant::now();
        let mut h = Hybrid::default();
        let b = NativeBuffer::default();
        let m = Snapshot {
            count: 1,
            states: [1; 2],
            directions: [ray([1., 2.]); 2],
            captured: Some(now),
            ..Default::default()
        };
        assert_eq!(h.apply(&m, &b, now, true).0.directions, m.directions);
        h.eyes[0] = run(false);
        assert_eq!(h.apply(&m, &b, now, false).0.directions, m.directions);
    }
    #[test]
    fn gap_does_not_join_fixations() {
        let mut e = EyeAnchor::default();
        for i in 1..200 {
            e.feed(i as f64 / 120., [i, i], [0., 0.], [4., 0.]);
        }
        for i in 200..400 {
            e.feed(5. + i as f64 / 120., [i, i], [6., 0.], [10., 0.]);
        }
        assert_eq!(e.bias, [0.; 2]);
    }
    #[test]
    fn short_blink_retains_first_pose_but_requires_new_window() {
        let mut e = EyeAnchor::default();
        for i in 1..200 {
            e.feed(i as f64 / 120., [i, i], [0., 0.], [4., 0.]);
        }
        assert!(e.reference.is_some());
        for i in 200..263 {
            assert!(!e.feed(0.3 + i as f64 / 120., [i, i], [6., 0.], [10., 0.]));
            assert_eq!(e.bias, [0.; 2]);
        }
        assert!(e.feed(0.3 + 263. / 120., [263, 263], [6., 0.], [10., 0.]));
        assert!(e.bias[0] > 0.);
    }
    #[test]
    fn short_blink_does_not_authorize_same_pose_or_stuck_native() {
        for stuck in [false, true] {
            let mut e = EyeAnchor::default();
            for i in 1..200 {
                e.feed(i as f64 / 120., [i, i], [0., 0.], [4., 0.]);
            }
            for i in 200..500 {
                let x = if stuck { 6. } else { 0. };
                e.feed(0.3 + i as f64 / 120., [i, i], [x, 0.], [4., 0.]);
            }
            assert_eq!(e.bias, [0.; 2]);
        }
    }
    #[test]
    fn diagnostic_identifies_spread_without_weakening_gate() {
        let mut e = EyeAnchor::default();
        for i in 1..200 {
            let noise = if i % 2 == 0 { 0.6 } else { -0.6 };
            e.feed(i as f64 / 120., [i, i], [0., 0.], [4. + noise, 0.]);
        }
        assert_eq!(e.diagnostic.gate, "Tobii moving");
        assert_eq!(e.bias, [0.; 2]);
        assert!((e.diagnostic.spread[1] - 1.2).abs() < 1e-9);
    }
    #[test]
    fn spike_blocks_window() {
        let mut e = run(false);
        let old = e.bias;
        for i in 2400..2430 {
            let x = if i == 2400 { 30. } else { 6. };
            assert!(!e.feed(i as f64 / 120., [i, i], [6., 0.], [x, 1.]));
        }
        assert_eq!(e.bias, old);
    }
    #[test]
    fn native_axes_independent_eyes_and_selected_sink() {
        let start = Instant::now();
        let mut h = Hybrid::default();
        h.start = start;
        let mut buffer = NativeBuffer::default();
        let mut selected = Snapshot::default();
        for i in 1..2400 {
            let now = start + Duration::from_secs_f64(i as f64 / 120.);
            let x = if (i / 240) % 2 == 0 { 0. } else { 6. };
            let mut n = sample(i, x + 4.);
            let r = ray([x - 3., 0.]);
            n.right.gaze = [r[0], r[1], -r[2]];
            buffer.push(n, now);
            let m = Snapshot {
                count: i,
                states: [1; 2],
                directions: [ray([x, 0.]); 2],
                captured: Some(now),
                ..Default::default()
            };
            selected = h.apply(&m, &buffer, now, true).0;
        }
        assert!((h.eyes[0].bias[0] - 4.).abs() < 1e-4);
        assert!((h.eyes[1].bias[0] + 3.).abs() < 1e-4);
        let mut eyes = [
            crate::core::types::EyeResult::new(crate::core::types::Eye::Left),
            crate::core::types::EyeResult::new(crate::core::types::Eye::Right),
        ];
        selected.apply(&mut eyes, selected.captured.unwrap(), false);
        let raw = selected.raw_sample(&eyes);
        assert_eq!(
            raw.left.gaze,
            [
                -selected.directions[0][0],
                selected.directions[0][1],
                -selected.directions[0][2]
            ]
        );
        assert_eq!(raw.right.gaze, eyes[1].gaze);
    }
    #[test]
    fn estimated_pairs_and_invalid_native_do_not_bootstrap() {
        let start = Instant::now();
        let mut h = Hybrid::default();
        h.start = start;
        let mut b = NativeBuffer::default();
        for i in 1..800 {
            let now = start + Duration::from_secs_f64(i as f64 / 120.);
            let x = if (i / 240) % 2 == 0 { 0. } else { 6. };
            let mut n = sample(i, x + 4.);
            n.right.gaze_valid = false;
            b.push(n, now);
            let m = Snapshot {
                count: i,
                states: [2, 1],
                directions: [ray([x, 0.]); 2],
                captured: Some(now),
                ..Default::default()
            };
            let (o, s) = h.apply(&m, &b, now, true);
            assert_eq!(s.bias, [[0.; 2]; 2]);
            assert_eq!(o.directions, m.directions);
        }
    }
    #[test]
    fn native_clock_reset_freezes_bias_not_output_snap() {
        let now = Instant::now();
        let mut h = Hybrid::default();
        let mut b = NativeBuffer::default();
        h.context = Some((0, true));
        h.eyes[0] = run(false);
        let bias = h.eyes[0].bias;
        b.push(sample(10, 0.), now);
        b.push(sample(5, 0.), now);
        let m = Snapshot {
            count: 100,
            states: [1; 2],
            directions: [ray([0., 0.]); 2],
            captured: Some(now),
            ..Default::default()
        };
        let (_, s) = h.apply(&m, &b, now, true);
        assert_eq!(s.bias[0], bias);
        assert_eq!(s.updating, [false; 2]);
    }
}
