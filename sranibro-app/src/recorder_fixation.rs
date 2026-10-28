//! World-fixed recorder target context. No measured eye gaze is inferred here.
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

pub const PROTOCOL: &str = "xr5_world_fixation_v1";
pub const DISTANCE_M: f32 = 2.0;
pub const POSE_MAX_AGE: Duration = Duration::from_millis(50);
pub type Matrix34 = [[f32; 4]; 3];
pub type Feedback = Arc<Mutex<PoseHistory>>;

#[derive(Clone, Debug)]
pub struct WorldPose {
    pub session: u64,
    pub queried_at: Instant,
    pub target_world: [f32; 3],
    pub head_to_world: Matrix34,
    /// Runtime view transforms; these are not measured anatomical eye centers.
    pub eye_to_head: [Matrix34; 2],
    pub angular_velocity: [f32; 3],
    pub tracking_result: i32,
    pub valid: bool,
    pub connected: bool,
    pub overlay_visible: bool,
}

pub fn transform(m: Matrix34, p: [f32; 3]) -> [f32; 3] {
    std::array::from_fn(|r| m[r][3] + (0..3).map(|c| m[r][c] * p[c]).sum::<f32>())
}
pub fn inverse_transform(m: Matrix34, p: [f32; 3]) -> [f32; 3] {
    let delta = std::array::from_fn::<_, 3, _>(|r| p[r] - m[r][3]);
    std::array::from_fn(|r| (0..3).map(|c| m[c][r] * delta[c]).sum())
}
pub fn unit(p: [f32; 3]) -> Option<[f32; 3]> {
    let norm = p.iter().map(|v| v * v).sum::<f32>().sqrt();
    (norm.is_finite() && norm > 0.0001).then(|| p.map(|v| v / norm))
}
pub fn rigid(m: Matrix34) -> bool {
    m.iter().flatten().all(|v| v.is_finite())
        && (0..3).all(|r| {
            (0..3).all(|s| {
                let dot = (0..3).map(|c| m[r][c] * m[s][c]).sum::<f32>();
                (dot - if r == s { 1.0 } else { 0.0 }).abs() < 0.01
            })
        })
        && {
            let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
                - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
            (det - 1.0).abs() < 0.01
        }
}
pub fn same_origin(a: Matrix34, b: Matrix34) -> bool {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() < 0.0001)
}
impl WorldPose {
    pub fn usable(&self) -> bool {
        self.valid
            && self.connected
            && self.overlay_visible
            && self.tracking_result == 200
            && rigid(self.head_to_world)
            && self.eye_to_head.iter().all(|m| rigid(*m))
            && self
                .angular_velocity
                .iter()
                .chain(self.target_world.iter())
                .all(|v| v.is_finite())
            && {
                let p = inverse_transform(self.head_to_world, self.target_world);
                p[2] < -0.1 && p[0].abs() <= -p[2] && p[1].abs() <= -p[2]
            }
    }
    pub fn eye_ray(&self, eye: usize) -> Option<[f32; 3]> {
        let in_head = inverse_transform(self.head_to_world, self.target_world);
        unit(inverse_transform(self.eye_to_head[eye], in_head))
    }
}

#[derive(Default)]
pub struct PoseHistory {
    pub poses: VecDeque<WorldPose>,
    pub error: Option<String>,
    pub requested_session: Option<u64>,
}
impl PoseHistory {
    pub fn push(&mut self, pose: WorldPose) {
        if self.poses.len() == 512 {
            self.poses.pop_front();
        }
        self.poses.push_back(pose);
        self.error = None;
    }
    pub fn fail(&mut self, error: String) {
        self.poses.clear();
        self.error = Some(error);
    }
    pub fn at(&self, session: u64, received_at: Instant) -> Option<WorldPose> {
        // Latest preceding query, never a future pose across a tracking gap.
        let pose = self
            .poses
            .iter()
            .rev()
            .find(|p| p.queried_at <= received_at)?;
        (pose.session == session
            && pose.usable()
            && received_at.saturating_duration_since(pose.queried_at) <= POSE_MAX_AGE)
            .then(|| pose.clone())
    }
}

#[derive(Clone, Debug)]
pub struct FixationPair {
    pub poses: [WorldPose; 2],
    pub camera_received_at: [Instant; 2],
}

pub struct WorldGuide {
    pub session: u64,
    pub feedback: Feedback,
    pub pair: Option<FixationPair>,
    pub events: Vec<String>,
    started: Instant,
    previous_available: bool,
    pub stable_since: Option<Instant>,
}
impl WorldGuide {
    pub fn new(session: u64, feedback: Feedback, now: Instant) -> Self {
        feedback.lock().unwrap().requested_session = Some(session);
        let mut g = Self {
            session,
            feedback,
            pair: None,
            events: Vec::new(),
            started: now,
            previous_available: false,
            stable_since: None,
        };
        g.event(now, "start_requested");
        g
    }
    pub fn available(&self, now: Instant) -> bool {
        self.feedback
            .lock()
            .unwrap()
            .at(self.session, now)
            .is_some()
    }
    pub fn event(&mut self, now: Instant, event: &str) {
        if self.events.len() < 8192 {
            self.events.push(format!(
                "{},{},{}\n",
                now.saturating_duration_since(self.started).as_micros(),
                self.session,
                event
            ));
        }
    }
    pub fn stop_display(&self) {
        let mut h = self.feedback.lock().unwrap();
        if h.requested_session == Some(self.session) {
            h.requested_session = None;
        }
    }
    pub fn tick(&mut self, now: Instant, paused: bool) -> bool {
        let available = !paused && self.available(now);
        if available != self.previous_available {
            self.event(
                now,
                if available {
                    "tracking_available"
                } else {
                    "tracking_wait"
                },
            );
            self.stable_since = available.then_some(now);
            self.previous_available = available;
        }
        available
    }
    pub fn set_camera_pair(&mut self, received_at: [Instant; 2]) -> bool {
        let h = self.feedback.lock().unwrap();
        self.pair = h
            .at(self.session, received_at[0])
            .zip(h.at(self.session, received_at[1]))
            .map(|(l, r)| FixationPair {
                poses: [l, r],
                camera_received_at: received_at,
            });
        self.pair.is_some()
    }
}

impl Drop for WorldGuide {
    fn drop(&mut self) {
        self.stop_display();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    pub fn identity() -> Matrix34 {
        [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]
    }
    fn pose(t: Instant) -> WorldPose {
        let mut eyes = [identity(); 2];
        eyes[0][0][3] = -0.032;
        eyes[1][0][3] = 0.032;
        WorldPose {
            session: 7,
            queried_at: t,
            target_world: [0., 0., -2.],
            head_to_world: identity(),
            eye_to_head: eyes,
            angular_velocity: [0.; 3],
            tracking_result: 200,
            valid: true,
            connected: true,
            overlay_visible: true,
        }
    }
    #[test]
    fn world_target_stays_fixed_as_head_rotates_and_translates() {
        let mut p = pose(Instant::now());
        assert!(p.eye_ray(0).unwrap()[0] > 0.);
        assert!(p.eye_ray(1).unwrap()[0] < 0.);
        let a = 20_f32.to_radians();
        p.head_to_world = [
            [a.cos(), 0., a.sin(), 0.1],
            [0., 1., 0., 0.],
            [-a.sin(), 0., a.cos(), 0.],
        ];
        assert_eq!(p.target_world, [0., 0., -2.]);
        assert!(p.usable());
        assert!(p.eye_ray(0).unwrap()[0] > 0.2);
        let local = inverse_transform(p.head_to_world, p.target_world);
        let recovered = transform(p.head_to_world, local);
        for axis in 0..3 {
            assert!((recovered[axis] - p.target_world[axis]).abs() < 1e-5);
        }
    }
    #[test]
    fn stale_future_invalid_and_wrong_session_are_rejected() {
        let t = Instant::now();
        let mut h = PoseHistory::default();
        h.push(pose(t));
        assert!(h.at(7, t).is_some());
        assert!(h.at(8, t).is_none());
        assert!(h.at(7, t - Duration::from_millis(1)).is_none());
        assert!(h.at(7, t + Duration::from_millis(51)).is_none());
        let mut bad = pose(t + Duration::from_millis(10));
        bad.valid = false;
        h.push(bad);
        assert!(h.at(7, t + Duration::from_millis(12)).is_none());
        assert!(h.at(7, t + Duration::from_millis(5)).is_some());
        h.fail("origin changed".into());
        assert!(h.at(7, t).is_none());
    }
    #[test]
    fn bounded_history_and_unusable_geometry() {
        let t = Instant::now();
        let mut h = PoseHistory::default();
        for i in 0..600 {
            h.push(pose(t + Duration::from_millis(i)));
        }
        assert_eq!(h.poses.len(), 512);
        let mut p = pose(t);
        p.target_world[2] = 2.;
        assert!(!p.usable());
        p = pose(t);
        p.head_to_world[0][0] = f32::NAN;
        assert!(!p.usable());
        p = pose(t);
        p.overlay_visible = false;
        assert!(!p.usable());
    }
    #[test]
    fn origin_change_is_detected_and_guide_drop_stops_only_its_session() {
        let m = identity();
        let mut moved = m;
        moved[0][3] = 0.001;
        assert!(same_origin(m, m));
        assert!(!same_origin(m, moved));
        let feedback = Arc::new(Mutex::new(PoseHistory::default()));
        let guide = WorldGuide::new(8, feedback.clone(), Instant::now());
        assert_eq!(feedback.lock().unwrap().requested_session, Some(8));
        feedback.lock().unwrap().requested_session = Some(9);
        drop(guide);
        assert_eq!(feedback.lock().unwrap().requested_session, Some(9));
        drop(WorldGuide::new(10, feedback.clone(), Instant::now()));
        assert_eq!(feedback.lock().unwrap().requested_session, None);
    }
}
