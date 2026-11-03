
use super::types::{Eye, EyeResult, GazeSample};

const OPEN_TRUST_FRAMES: u32 = 20;
const BLINK_RESET_FRAMES: u32 = 5;
const WARMUP_FRAMES: u32 = 200;
const KALMAN_ALPHA: f32 = 0.30;
const UPPER_OFFSET: f32 = 0.02;
const SQUEEZE_TOP_OFFSET: f32 = 0.03;
const BLINK_OFFSET: f32 = 0.20;
const BLINK_RELEASE_OFFSET: f32 = 0.10;
const WIDE_RELEASE_OFFSET: f32 = 0.01;
const WIDE_CEIL_MIN_SPAN: f32 = 0.15;
const WIDE_CEIL_DECAY: f32 = 0.0008;
const SQ_FLOOR_TRACK: f32 = 0.005;
const BLINK_BOTTOM_SMOOTH: f32 = 0.3;
const BLINK_LEARN_MIN_FRAMES: u32 = 12;
const BLINK_LEARN_MAX_FRAMES: u32 = 360;
const BLINK_DEPTH_DEEPEN: f32 = 0.4;
const BLINK_DEPTH_RELAX: f32 = 0.2;
const BLINK_VEL_NORM: f32 = 0.19;
const BLINK_VEL_FLOOR: f32 = 0.045;
const BLINK_VEL_CAP: f32 = 0.075;
const BLINK_FALL2_NORM: f32 = 0.31;
const BLINK_FALL2_FLOOR: f32 = 0.060;
const BLINK_FALL2_CAP: f32 = 0.115;
const FALL2_COMMIT_DEPTH: f32 = 0.60;
const FAST_BLINK_MAX_FRAMES: u32 = 24;
const FAST_BLINK_CAP_SHALLOW: u32 = 12;
const SHALLOW_COMMIT_DIP: f32 = 0.5;
const FAST_BLINK_ARM_FRAMES: u32 = 2;
const FAST_BLINK_ARM_MAX: u32 = 3;
const RELEASE_RISE_NORM: f32 = 0.05;
const RELEASE_RISE_FLOOR: f32 = 0.012;
const LATCH_EXIT_NORM: f32 = 0.25;
const ASSIST_DIP_NORM: f32 = 0.40;
const ASSIST_FALL2_NORM: f32 = 0.20;
const DETECT_GRACE_FRAMES: u32 = 24;
const YOKE_RELEASE_OPEN: f32 = 0.25;

const MID_ANCHOR_MIN: f32 = 0.30;
const MID_ANCHOR_MAX: f32 = 0.70;
const MID_ANCHOR_LEARN: f32 = 0.3;
const PAIR_MAX_DIFF: f32 = 0.30;
const CURVE_DOWN_RECENT: u32 = 4;
const CURVE_PRE_WINDOW: u32 = 60;
const PAIR_MID_WINDOW: f32 = 0.25;
const PAIR_MIN_WEIGHT: f32 = 4.0;
const PAIR_STALE_FRAMES: u32 = 600;
const EP_SYNC_SKEW: u32 = 18;
const FAST_CLOSING_RAW: f32 = -0.05;
const BASELINE_RANGE_LO: f32 = 0.32;
const BASELINE_RANGE_HI: f32 = 0.80;
const WIDE_LEARN_CAP: f32 = 0.10;
const BOOTSTRAP_ANCHOR_N: u32 = 8;
const BASELINE_BOOTSTRAP_N: u32 = 100;
const BASELINE_ADAPT: f32 = 0.003;
const BASELINE_LEARN_BAND: f32 = UPPER_OFFSET;
const BASELINE_SLOW_FACTOR: f32 = 0.02;
const WIDE_STUCK_FRAMES: u32 = 1800;
const WIDE_CLEAR_FRAMES: u32 = 48;
const WIDE_DISCHARGE: u32 = WIDE_STUCK_FRAMES / WIDE_CLEAR_FRAMES + 1;
const RECENTER_PRIOR_N: u32 = 8;
const RESTORE_PRIOR_N: u32 = 50;
const WIDE_GATE_STEP: f32 = 0.08;
const COUPLE_RATE: f32 = 0.001;
const COUPLE_MIN_OPEN_STREAK: u32 = 30;
const SQUEEZE_DWELL_MIN: u32 = 8;
const SQUEEZE_HARDCLOSE_T: f32 = 0.70;
const CH0_PRESENT_GATE: f32 = 0.05;

const KAL_MOTION_DEADZONE: f32 = 0.05;
const KAL_Q_FAST: f32 = 0.25;
const KAL_R_FAST: f32 = 0.5;
const KAL_Q_SLOW: f32 = 0.001;
const KAL_R_SLOW: f32 = 20.5;
const KAL_P0: f32 = 1.0;

const NATIVE_SQ_FLOOR: f32 = 0.18;
const NATIVE_SQ_SPAN: f32 = 0.45;

fn apply_anchor(x: f32, a: f32) -> f32 {
    if x <= a {
        0.5 * x / a.max(1e-3)
    } else {
        0.5 + 0.5 * (x - a) / (1.0 - a).max(1e-3)
    }
}

#[derive(Clone, Copy, Debug)]
struct ScalarKalman {
    x: f32,
    p: f32,
    init: bool,
}

impl Default for ScalarKalman {
    fn default() -> Self {
        Self { x: 1.0, p: KAL_P0, init: false }
    }
}

impl ScalarKalman {
    fn update(&mut self, z: f32) -> f32 {
        if !self.init {
            self.x = z;
            self.init = true;
            return self.x;
        }
        let dev = (z - self.x).abs();
        let (q, r) = if dev >= KAL_MOTION_DEADZONE {
            (KAL_Q_FAST, KAL_R_FAST)
        } else {
            (KAL_Q_SLOW, KAL_R_SLOW)
        };
        self.p += q;
        let k = self.p / (self.p + r);
        self.x += k * (z - self.x);
        self.p *= 1.0 - k;
        self.x
    }
    fn reset(&mut self) {
        self.x = 1.0;
        self.p = KAL_P0;
        self.init = false;
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Tuning {
    pub alpha_open: f32,
    pub alpha_close: f32,
    pub squeeze_deadzone: f32,
    pub squeeze_gain: f32,
    pub adaptive_kalman: bool,
    pub native_squeeze: bool,
    pub wide_gain: f32,
    pub couple_eyes: bool,
    pub continuous_calib: bool,
    pub wide_requires_both: bool,
    pub gaze_yoke: bool,
    pub wide_squeeze_exclusive: bool,
    pub open_deadzone: f32,
    pub blink_reopen_ms: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            alpha_open: 0.35,
            alpha_close: 0.75,
            squeeze_deadzone: 0.40,
            squeeze_gain: 0.65,
            adaptive_kalman: false,
            native_squeeze: true,
            wide_gain: 1.0,
            couple_eyes: false,
            continuous_calib: true,
            wide_requires_both: true,
            gaze_yoke: true,
            wide_squeeze_exclusive: false,
            open_deadzone: 0.08,
            blink_reopen_ms: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EyeState {
    baseline: f32,
    baseline_n: u32,
    upper: f32,
    wide_ceiling: f32,
    squeeze_floor: f32,
    closed_ref: f32,
    blink_depth: f32,
    blink_len: u32,
    blink_smooth: f32,
    blink_lo: f32,
    blink_dirty: bool,
    mid_anchor: f32,
    mid_anchor_staged: Option<f32>,
    last_ramp_pre: f32,
    ep_entry_frame: u32,
    ep_clean_exit: Option<(u32, u32)>,
    frame_count: u32,
    open_streak: u32,
    closed_streak: u32,
    squeeze_streak: u32,
    is_closed: bool,
    is_wide: bool,
    wide_frames: u32,
    fall_run: f32,
    fall_updates: u32,
    since_down: u32,
    prev_raw: f32,
    fast_blink_frames: u32,
    fast_blink_arm: u32,
    fast_blink_arm_raw: f32,
    arm_updates: u32,
    latch_bottom: f32,
    latch_cap: u32,
    rawu1: f32,
    rawu2: f32,
    detect_grace: u32,
    reopening: bool,
    yoke_hold: bool,
    kalman_x: f32,
    smooth_open: f32,
    kalman: ScalarKalman,
    gaze_smooth: [f32; 3],
    pupil_smooth: f32,
}

impl Default for EyeState {
    fn default() -> Self {
        Self {
            baseline: 0.60,
            baseline_n: 0,
            upper: 0.62,
            wide_ceiling: 0.0,
            squeeze_floor: NATIVE_SQ_FLOOR,
            closed_ref: 0.40,
            blink_depth: BLINK_OFFSET,
            blink_len: 0,
            blink_smooth: 0.40,
            blink_lo: 0.40,
            blink_dirty: false,
            mid_anchor: 0.5,
            mid_anchor_staged: None,
            last_ramp_pre: -1.0,
            ep_entry_frame: 0,
            ep_clean_exit: None,
            frame_count: 0,
            open_streak: 0,
            closed_streak: 0,
            squeeze_streak: 0,
            is_closed: false,
            is_wide: false,
            wide_frames: 0,
            fall_run: 0.0,
            fall_updates: 0,
            since_down: u32::MAX,
            prev_raw: 0.60,
            fast_blink_frames: 0,
            fast_blink_arm: 0,
            fast_blink_arm_raw: 0.60,
            arm_updates: 0,
            latch_bottom: 0.60,
            latch_cap: FAST_BLINK_MAX_FRAMES,
            rawu1: 0.60,
            rawu2: 0.60,
            detect_grace: DETECT_GRACE_FRAMES,
            reopening: false,
            yoke_hold: false,
            kalman_x: 0.0,
            smooth_open: 1.0,
            kalman: ScalarKalman::default(),
            gaze_smooth: [0.0, 0.0, -1.0],
            pupil_smooth: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalibSnapshot {
    pub baseline: f32,
    pub baseline_n: u32,
    pub frame_count: u32,
    #[serde(default = "default_blink_depth")]
    pub blink_depth: f32,
    #[serde(default = "default_mid_anchor")]
    pub mid_anchor: f32,
}

fn default_blink_depth() -> f32 {
    BLINK_OFFSET
}

fn default_mid_anchor() -> f32 {
    0.5
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalibStore {
    pub left: CalibSnapshot,
    pub right: CalibSnapshot,
}

pub fn load_calib(path: &str) -> Option<CalibStore> {
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

pub fn save_calib(path: &str, store: &CalibStore) {
    if let Ok(text) = toml::to_string(store) {
        let _ = std::fs::write(path, text);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DiagSnapshot {
    pub baseline: [f32; 2],
    pub closed_ref: [f32; 2],
    pub ramp_pre: [f32; 2],
    pub mid_anchor: [f32; 2],
    pub staged: [bool; 2],
    pub is_wide: [bool; 2],
    pub latched: [bool; 2],
    pub fall_run: [f32; 2],
    pub since_down: [u32; 2],
    pub blink_len: [u32; 2],
    pub ep_exit: [bool; 2],
    pub pend_w: f32,
    pub yoke_hold: [bool; 2],
}

#[derive(Clone, Copy, Debug, Default)]
struct PairPending {
    wx: [f32; 2],
    w_sum: f32,
    start_frame: u32,
    last_frame: u32,
    age: u32,
}

pub struct SRanipalState {
    eyes: [EyeState; 2],
    pub tuning: Tuning,
    wide_gate: f32,
    pend: PairPending,
}

impl Default for SRanipalState {
    fn default() -> Self {
        Self::new()
    }
}

impl SRanipalState {
    pub fn new() -> Self {
        Self {
            eyes: [EyeState::default(); 2],
            tuning: Tuning::default(),
            wide_gate: 0.0,
            pend: PairPending::default(),
        }
    }

    pub fn baseline(&self, e: Eye) -> f32 {
        self.eyes[e.idx()].baseline
    }

    pub fn snapshot(&self, e: Eye) -> CalibSnapshot {
        let s = &self.eyes[e.idx()];
        CalibSnapshot {
            baseline: s.baseline,
            baseline_n: s.baseline_n,
            frame_count: s.frame_count,
            blink_depth: s.blink_depth,
            mid_anchor: s.mid_anchor,
        }
    }

    pub fn recenter(&mut self) {
        self.wide_gate = 0.0;
        for s in self.eyes.iter_mut() {
            s.baseline_n = RECENTER_PRIOR_N;
            s.wide_frames = if s.is_wide { WIDE_STUCK_FRAMES } else { 0 };
            s.fall_run = 0.0;
            s.fall_updates = 0;
            s.is_closed = false;
            s.is_wide = false;
            s.fast_blink_frames = 0;
            s.fast_blink_arm = 0;
            s.arm_updates = 0;
            s.kalman_x = 0.0;
            s.kalman.reset();
            s.wide_ceiling = 0.0;
            s.squeeze_floor = NATIVE_SQ_FLOOR;
            s.blink_len = 0;
            s.mid_anchor_staged = None;
            s.ep_clean_exit = None;
            s.last_ramp_pre = -1.0;
            s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
        }
        self.pend = PairPending::default();
    }

    pub fn restore(&mut self, e: Eye, snap: CalibSnapshot) {
        let s = &mut self.eyes[e.idx()];
        s.baseline = snap.baseline;
        s.baseline_n = snap.baseline_n.min(RESTORE_PRIOR_N);
        s.frame_count = snap.frame_count;
        s.blink_depth = if snap.blink_depth.is_finite() {
            snap.blink_depth.clamp(BLINK_OFFSET, 0.40)
        } else {
            BLINK_OFFSET
        };
        s.mid_anchor = if snap.mid_anchor.is_finite() {
            snap.mid_anchor.clamp(MID_ANCHOR_MIN, MID_ANCHOR_MAX)
        } else {
            0.5
        };
        s.mid_anchor_staged = None;
        s.ep_clean_exit = None;
        s.prev_raw = s.baseline;
        s.rawu1 = s.baseline;
        s.rawu2 = s.baseline;
        s.detect_grace = DETECT_GRACE_FRAMES;
        s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
    }

    pub fn snapshot_all(&self) -> CalibStore {
        CalibStore { left: self.snapshot(Eye::Left), right: self.snapshot(Eye::Right) }
    }

    pub fn restore_all(&mut self, store: &CalibStore) {
        self.restore(Eye::Left, store.left);
        self.restore(Eye::Right, store.right);
        let fc = self.eyes[0].frame_count.max(self.eyes[1].frame_count);
        self.eyes[0].frame_count = fc;
        self.eyes[1].frame_count = fc;
    }

    fn update_calibration(&mut self, i: usize, raw: f32, present: bool) {
        let s = &mut self.eyes[i];
        s.frame_count += 1;
        if s.is_wide {
            s.wide_frames = s.wide_frames.saturating_add(1);
        } else if present
            && raw > s.baseline - BLINK_RELEASE_OFFSET
            && raw < BASELINE_RANGE_HI
        {
            s.wide_frames = s.wide_frames.saturating_sub(WIDE_DISCHARGE);
        }
        if present && raw > BASELINE_RANGE_LO && raw < BASELINE_RANGE_HI {
            let stuck = s.is_wide && s.wide_frames >= WIDE_STUCK_FRAMES;
            if s.baseline_n < BASELINE_BOOTSTRAP_N {
                if s.baseline_n < BOOTSTRAP_ANCHOR_N
                    || (raw - s.baseline).abs() < WIDE_LEARN_CAP
                {
                    s.baseline_n += 1;
                    s.baseline += (raw - s.baseline) / s.baseline_n as f32;
                } else {
                    let rate = if stuck {
                        BASELINE_ADAPT
                    } else {
                        BASELINE_ADAPT * BASELINE_SLOW_FACTOR
                    };
                    s.baseline += rate * (raw - s.baseline);
                }
            } else {
                let dev = raw - s.baseline;
                let rate = if stuck || dev.abs() <= BASELINE_LEARN_BAND {
                    BASELINE_ADAPT
                } else {
                    BASELINE_ADAPT * BASELINE_SLOW_FACTOR
                };
                s.baseline += rate * dev;
                s.baseline_n = s.baseline_n.saturating_add(1);
            }
        }
        let b = s.baseline;
        s.upper = b + UPPER_OFFSET;
        let entry = b - BLINK_OFFSET;
        let release = b - BLINK_RELEASE_OFFSET;
        if s.blink_len == 0 {
            if present && raw < entry && s.baseline_n >= BASELINE_BOOTSTRAP_N {
                s.blink_len = 1;
                s.blink_smooth = raw;
                s.blink_lo = raw;
                s.blink_dirty = s.fast_blink_frames > 0 || s.fast_blink_arm > 0;
                s.ep_entry_frame = s.frame_count;
                s.ep_clean_exit = None;
            }
        } else {
            s.blink_len = s.blink_len.saturating_add(1);
            if s.fast_blink_frames > 0 || s.fast_blink_arm > 0 || !present {
                s.blink_dirty = true;
            }
            if present {
                s.blink_smooth += BLINK_BOTTOM_SMOOTH * (raw - s.blink_smooth);
                s.blink_lo = s.blink_lo.min(s.blink_smooth);
            }
            if raw > release {
                if !s.blink_dirty
                    && s.blink_len >= BLINK_LEARN_MIN_FRAMES
                    && s.blink_len <= BLINK_LEARN_MAX_FRAMES
                {
                    s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                    let observed = (b - s.blink_lo).clamp(0.05, 0.40);
                    let rate = if observed > s.blink_depth {
                        BLINK_DEPTH_DEEPEN
                    } else {
                        BLINK_DEPTH_RELAX
                    };
                    s.blink_depth += rate * (observed - s.blink_depth);
                }
                s.blink_len = 0;
            }
        }
        s.closed_ref = (b - s.blink_depth).clamp(b - 0.40, b - 0.05);
    }

    pub fn closed_ref(&self, e: Eye) -> f32 {
        self.eyes[e.idx()].closed_ref
    }

    pub fn diag(&self) -> DiagSnapshot {
        let e = &self.eyes;
        DiagSnapshot {
            baseline: [e[0].baseline, e[1].baseline],
            closed_ref: [e[0].closed_ref, e[1].closed_ref],
            ramp_pre: [e[0].last_ramp_pre, e[1].last_ramp_pre],
            mid_anchor: [e[0].mid_anchor, e[1].mid_anchor],
            staged: [e[0].mid_anchor_staged.is_some(), e[1].mid_anchor_staged.is_some()],
            is_wide: [e[0].is_wide, e[1].is_wide],
            latched: [e[0].fast_blink_frames > 0, e[1].fast_blink_frames > 0],
            fall_run: [e[0].fall_run, e[1].fall_run],
            since_down: [e[0].since_down, e[1].since_down],
            blink_len: [e[0].blink_len, e[1].blink_len],
            ep_exit: [e[0].ep_clean_exit.is_some(), e[1].ep_clean_exit.is_some()],
            pend_w: self.pend.w_sum,
            yoke_hold: [e[0].yoke_hold, e[1].yoke_hold],
        }
    }

    fn normalize(&mut self, i: usize, raw: f32, in2: f32) -> (f32, f32, f32) {
        let tuning = self.tuning;
        let s = &mut self.eyes[i];
        s.last_ramp_pre = -1.0;
        let b = s.baseline;
        let upper = b + UPPER_OFFSET;
        let squeeze_top = b - SQUEEZE_TOP_OFFSET;
        let open_full = b - tuning.open_deadzone.clamp(SQUEEZE_TOP_OFFSET, 0.20);
        let blink_top = if tuning.continuous_calib {
            s.closed_ref.min(open_full - 1e-3)
        } else {
            b - BLINK_OFFSET
        };
        let blink_release = b - BLINK_RELEASE_OFFSET;

        if !tuning.continuous_calib && s.is_closed {
            if raw < blink_release {
                s.squeeze_streak = 0;
                s.kalman_x *= 0.5;
                return (0.0, 0.0, 0.0);
            }
            s.is_closed = false;
        }

        if raw >= squeeze_top {
            s.fast_blink_frames = 0;
            s.fast_blink_arm = 0;
        }

        let ceil_floor = upper + WIDE_CEIL_MIN_SPAN;
        if s.wide_ceiling < ceil_floor {
            s.wide_ceiling = ceil_floor;
        } else if raw > s.wide_ceiling {
            s.wide_ceiling = raw;
        } else {
            s.wide_ceiling += WIDE_CEIL_DECAY * (ceil_floor - s.wide_ceiling);
        }
        let wide_span = (s.wide_ceiling - upper).max(1e-3);
        let wide_gain = tuning.wide_gain;

        let wide_release = upper - WIDE_RELEASE_OFFSET;
        if s.is_wide {
            if raw < wide_release {
                s.is_wide = false;
            } else {
                let x = ((in2 - upper) / wide_span).max(0.0);
                let kx = (1.0 - KALMAN_ALPHA) * s.kalman_x + KALMAN_ALPHA * x;
                s.kalman_x = kx;
                return (1.0, (kx.min(1.0) * wide_gain).clamp(0.0, 1.0), 0.0);
            }
        } else if raw > upper {
            s.is_wide = true;
            let x = ((in2 - upper) / wide_span).max(0.0);
            let kx = (1.0 - KALMAN_ALPHA) * s.kalman_x + KALMAN_ALPHA * x;
            s.kalman_x = kx;
            return (1.0, (kx.min(1.0) * wide_gain).clamp(0.0, 1.0), 0.0);
        }

        if tuning.continuous_calib {
            s.kalman_x *= 1.0 - KALMAN_ALPHA;
            let d = s.blink_depth;
            let is_upd = (raw - s.prev_raw).abs() > 1e-6;
            if s.detect_grace > 0 {
                s.detect_grace -= 1;
            }
            if s.fast_blink_frames > 0 {
                s.latch_bottom = s.latch_bottom.min(raw);
                let rise = (RELEASE_RISE_NORM * d).max(RELEASE_RISE_FLOOR);
                let exit_zone = raw > blink_top + (LATCH_EXIT_NORM * d).clamp(0.03, 0.09);
                let deep_now = (b - raw) >= SHALLOW_COMMIT_DIP * d;
                let cap_hit = s.fast_blink_frames >= s.latch_cap && !deep_now;
                if (raw > s.latch_bottom + rise && exit_zone) || cap_hit {
                    s.fast_blink_frames = 0;
                    s.fall_run = 0.0;
                    s.fall_updates = 0;
                    s.rawu1 = raw;
                    s.rawu2 = raw;
                } else {
                    s.fast_blink_frames += 1;
                    s.squeeze_streak = 0;
                    return (0.0, 0.0, 0.0);
                }
            } else if s.fast_blink_arm > 0 {
                if raw > s.fast_blink_arm_raw {
                    s.fast_blink_arm = 0;
                } else {
                    if is_upd {
                        s.arm_updates = s.arm_updates.saturating_add(1);
                    }
                    let window_done = s.fast_blink_arm >= FAST_BLINK_ARM_FRAMES
                        && (s.arm_updates >= 1 || s.fast_blink_arm >= FAST_BLINK_ARM_MAX);
                    if window_done {
                        s.fast_blink_arm = 0;
                        s.fast_blink_frames = 1;
                        s.squeeze_streak = 0;
                        s.latch_bottom = raw;
                        s.latch_cap = if (b - raw) < SHALLOW_COMMIT_DIP * d {
                            FAST_BLINK_CAP_SHALLOW
                        } else {
                            FAST_BLINK_MAX_FRAMES
                        };
                        return (0.0, 0.0, 0.0);
                    }
                    s.fast_blink_arm += 1;
                }
            } else if is_upd && raw < squeeze_top && s.detect_grace == 0 {
                let v1 = s.prev_raw - raw;
                let fall2 = s.rawu2 - raw;
                let vel_thr = (BLINK_VEL_NORM * d).clamp(BLINK_VEL_FLOOR, BLINK_VEL_CAP);
                let fall2_thr = (BLINK_FALL2_NORM * d).clamp(BLINK_FALL2_FLOOR, BLINK_FALL2_CAP);
                let deep = (b - raw) >= FALL2_COMMIT_DEPTH * d;
                if v1 >= vel_thr || (fall2 >= fall2_thr && deep) {
                    s.fast_blink_arm = 1;
                    s.fast_blink_arm_raw = raw;
                    s.arm_updates = 0;
                }
            }
            let denom = (open_full - blink_top).max(1e-3);
            let opn_pre = ((raw - blink_top) / denom).clamp(0.0, 1.0);
            if s.fast_blink_arm == 0 {
                s.last_ramp_pre = opn_pre;
            }
            if let Some(a) = s.mid_anchor_staged {
                if opn_pre >= 0.99 {
                    s.mid_anchor = a;
                    s.mid_anchor_staged = None;
                }
            }
            let opn = apply_anchor(opn_pre, s.mid_anchor);
            let dz = tuning.squeeze_deadzone;
            let sqz = if opn >= 1.0 - dz {
                0.0
            } else {
                (((1.0 - dz) - opn) / (1.0 - dz).max(0.01) * tuning.squeeze_gain).clamp(0.0, 1.0)
            };
            if sqz > 0.0 {
                s.squeeze_streak += 1;
            } else {
                s.squeeze_streak = 0;
            }
            return (opn, (s.kalman_x.max(0.0) * wide_gain).min(1.0), sqz);
        }

        if raw >= open_full {
            s.squeeze_streak = 0;
            s.kalman_x *= 1.0 - KALMAN_ALPHA;
            return (1.0, (s.kalman_x.max(0.0) * wide_gain).min(1.0), 0.0);
        }

        if raw > blink_top {
            let t_raw = ((open_full - raw) / (open_full - blink_top)).clamp(0.0, 1.0);
            s.kalman_x *= 1.0 - KALMAN_ALPHA;
            s.squeeze_streak += 1;
            let opn = if t_raw >= SQUEEZE_HARDCLOSE_T {
                0.0
            } else {
                1.0 - (t_raw / SQUEEZE_HARDCLOSE_T)
            };
            let dz = tuning.squeeze_deadzone;
            let sqz = if t_raw <= dz || s.squeeze_streak < SQUEEZE_DWELL_MIN {
                0.0
            } else {
                let sqz_t = (t_raw - dz) / (1.0 - dz).max(0.01);
                sqz_t * tuning.squeeze_gain
            };
            return (opn, 0.0, sqz);
        }

        s.is_closed = true;
        s.squeeze_streak = 0;
        s.kalman_x *= 0.5;
        (0.0, 0.0, 0.0)
    }

    fn couple_baselines(&mut self, raw_l: f32, raw_r: f32) {
        let (l, r) = (self.eyes[0], self.eyes[1]);
        if l.frame_count < WARMUP_FRAMES || r.frame_count < WARMUP_FRAMES {
            return;
        }
        if l.open_streak < COUPLE_MIN_OPEN_STREAK || r.open_streak < COUPLE_MIN_OPEN_STREAK {
            return;
        }
        if !(raw_l > BASELINE_RANGE_LO && raw_l < BASELINE_RANGE_HI) {
            return;
        }
        if !(raw_r > BASELINE_RANGE_LO && raw_r < BASELINE_RANGE_HI) {
            return;
        }
        let mean = 0.5 * (l.baseline + r.baseline);
        self.eyes[0].baseline += COUPLE_RATE * (mean - l.baseline);
        self.eyes[1].baseline += COUPLE_RATE * (mean - r.baseline);
    }

    pub fn process_frame(
        &mut self,
        ml: [[f32; 5]; 2],
        gaze: &GazeSample,
        ml_loaded: bool,
    ) -> [EyeResult; 2] {
        let raws = [ml[0][1], ml[1][1]];
        let mut pe = [PerEye::default(); 2];

        for e in Eye::ALL {
            let i = e.idx();
            let raw = raws[i];
            let present = ml[i][0] > CH0_PRESENT_GATE;

            self.update_calibration(i, raw, present);

            let blink_threshold = self.eyes[i].baseline - 0.20;
            if raw < blink_threshold {
                self.eyes[i].closed_streak += 1;
                self.eyes[i].open_streak = 0;
                if self.eyes[i].closed_streak > BLINK_RESET_FRAMES {
                    self.eyes[i].kalman_x *= 0.5;
                }
            } else {
                self.eyes[i].closed_streak = 0;
                self.eyes[i].open_streak += 1;
            }

            let trusted = self.eyes[i].open_streak >= OPEN_TRUST_FRAMES
                || self.eyes[i].frame_count >= WARMUP_FRAMES;

            let (opn, wide_mag, sqz_mag) = if trusted {
                self.normalize(i, raw, raw)
            } else {
                self.eyes[i].is_wide = false;
                self.eyes[i].fast_blink_frames = 0;
                self.eyes[i].fast_blink_arm = 0;
                self.eyes[i].last_ramp_pre = -1.0;
                let held = if raw < self.eyes[i].baseline - BLINK_OFFSET {
                    0.0
                } else {
                    self.eyes[i].smooth_open
                };
                (held, 0.0, 0.0)
            };

            let squeeze_out = if self.tuning.native_squeeze {
                let s3 = ml[i][3];
                if raw >= self.eyes[i].baseline {
                    let f = &mut self.eyes[i].squeeze_floor;
                    *f += SQ_FLOOR_TRACK * (s3 - *f);
                    *f = f.clamp(0.0, 0.4);
                }
                let floor = self.eyes[i].squeeze_floor;
                (((s3 - floor) / NATIVE_SQ_SPAN).clamp(0.0, 1.0) * self.tuning.squeeze_gain)
                    .clamp(0.0, 1.0)
            } else {
                sqz_mag
            };

            let es = gaze.eye(e);
            let draw = raw - self.eyes[i].prev_raw;
            pe[i] = PerEye {
                raw,
                openness_target: opn.clamp(0.0, 1.0),
                wide_out: wide_mag,
                squeeze_out,
                gaze: es.gaze,
                gaze_v: es.gaze_valid,
                pupil_mm: es.pupil_mm,
                pupil_v: es.pupil_valid,
                closed: (if self.tuning.continuous_calib {
                    opn < 0.08
                } else {
                    raw < blink_threshold
                }) || !present,
                present,
                falling: draw <= 0.005,
                draw,
            };

            if draw < -0.001 {
                self.eyes[i].fall_run += -draw;
                self.eyes[i].fall_updates = self.eyes[i].fall_updates.saturating_add(1);
                self.eyes[i].since_down = 0;
            } else {
                self.eyes[i].since_down = self.eyes[i].since_down.saturating_add(1);
                if draw > 0.01 {
                    self.eyes[i].fall_run = 0.0;
                    self.eyes[i].fall_updates = 0;
                }
            }

            self.eyes[i].prev_raw = raw;
        }

        if self.tuning.continuous_calib {
            for e in Eye::ALL {
                let (i, j) = (e.idx(), e.opposite().idx());
                if self.eyes[i].fast_blink_frames > 0 && self.eyes[j].fast_blink_frames == 0 {
                    let d = self.eyes[j].blink_depth;
                    let dip = self.eyes[j].baseline - pe[j].raw;
                    let fall2 = self.eyes[j].rawu2 - pe[j].raw;
                    if pe[j].present
                        && pe[j].falling
                        && dip >= ASSIST_DIP_NORM * d
                        && fall2 >= ASSIST_FALL2_NORM * d
                    {
                        let sj = &mut self.eyes[j];
                        sj.fast_blink_frames = 1;
                        sj.fast_blink_arm = 0;
                        sj.squeeze_streak = 0;
                        sj.latch_bottom = pe[j].raw;
                        sj.latch_cap = if dip < SHALLOW_COMMIT_DIP * d {
                            FAST_BLINK_CAP_SHALLOW
                        } else {
                            FAST_BLINK_MAX_FRAMES
                        };
                        pe[j].openness_target = 0.0;
                        pe[j].closed = true;
                    }
                }
            }
        }
        for i in 0..2 {
            if pe[i].draw.abs() > 1e-6 {
                self.eyes[i].rawu2 = self.eyes[i].rawu1;
                self.eyes[i].rawu1 = pe[i].raw;
            }
        }

        if self.tuning.continuous_calib {
            let xs = [self.eyes[0].last_ramp_pre, self.eyes[1].last_ramp_pre];
            let pair_ok = (0..2).all(|i| {
                let s = &self.eyes[i];
                pe[i].present
                    && (s.open_streak >= OPEN_TRUST_FRAMES || s.frame_count >= WARMUP_FRAMES)
                    && s.baseline_n >= BASELINE_BOOTSTRAP_N
                    && s.fast_blink_frames == 0
                    && s.fast_blink_arm == 0
                    && s.since_down <= CURVE_DOWN_RECENT
                    && s.fall_run >= 0.03
                    && xs[i] >= 0.05
                    && xs[i] <= 0.95
            }) && (xs[0] - xs[1]).abs() <= PAIR_MAX_DIFF;
            if pair_ok {
                let m = 0.5
                    * (apply_anchor(xs[0], self.eyes[0].mid_anchor)
                        + apply_anchor(xs[1], self.eyes[1].mid_anchor));
                let w = (1.0 - (m - 0.5).abs() / PAIR_MID_WINDOW).max(0.0);
                if w > 0.0 {
                    if self.pend.w_sum == 0.0 {
                        self.pend.start_frame = self.eyes[0].frame_count;
                        self.pend.age = 0;
                    }
                    self.pend.wx[0] += w * xs[0];
                    self.pend.wx[1] += w * xs[1];
                    self.pend.w_sum += w;
                    self.pend.last_frame = self.eyes[0].frame_count;
                }
            }
            if let (Some((en_l, ex_l)), Some((en_r, ex_r))) =
                (self.eyes[0].ep_clean_exit, self.eyes[1].ep_clean_exit)
            {
                let min_en = en_l.min(en_r);
                let max_ex = ex_l.max(ex_r);
                if en_l.abs_diff(en_r) <= EP_SYNC_SKEW
                    && self.pend.w_sum >= PAIR_MIN_WEIGHT
                    && self.pend.start_frame + CURVE_PRE_WINDOW >= min_en
                    && self.pend.start_frame <= max_ex
                    && self.pend.last_frame <= max_ex
                {
                    let mut a = [0.0f32; 2];
                    for i in 0..2 {
                        let a_hat = self.pend.wx[i] / self.pend.w_sum;
                        a[i] = self.eyes[i].mid_anchor
                            + MID_ANCHOR_LEARN * (a_hat - self.eyes[i].mid_anchor);
                    }
                    let c = 0.5 * (a[0] + a[1]) - 0.5;
                    for i in 0..2 {
                        self.eyes[i].mid_anchor_staged =
                            Some((a[i] - c).clamp(MID_ANCHOR_MIN, MID_ANCHOR_MAX));
                    }
                }
                self.eyes[0].ep_clean_exit = None;
                self.eyes[1].ep_clean_exit = None;
                self.pend = PairPending::default();
            }
            if self.pend.w_sum > 0.0 {
                self.pend.age = self.pend.age.saturating_add(1);
                let abandoned = pe[0].raw > self.eyes[0].baseline - 0.05
                    && pe[1].raw > self.eyes[1].baseline - 0.05;
                if self.pend.age > PAIR_STALE_FRAMES || abandoned {
                    self.pend = PairPending::default();
                }
            }
        }

        if self.tuning.couple_eyes {
            self.couple_baselines(pe[0].raw, pe[1].raw);
        }

        if !self.tuning.native_squeeze {
            let both_squeezing = self.eyes[0].squeeze_streak >= SQUEEZE_DWELL_MIN
                && self.eyes[1].squeeze_streak >= SQUEEZE_DWELL_MIN;
            if both_squeezing {
                let m = pe[0].squeeze_out.max(pe[1].squeeze_out);
                pe[0].squeeze_out = m;
                pe[1].squeeze_out = m;
            }
        }
        if self.tuning.wide_requires_both {
            let both_wide = self.eyes[0].is_wide && self.eyes[1].is_wide;
            let blinking = pe[0].closed || pe[1].closed;
            let gate_target = if both_wide { 1.0 } else { 0.0 };
            if blinking {
            } else if self.wide_gate < gate_target {
                self.wide_gate = (self.wide_gate + WIDE_GATE_STEP).min(gate_target);
            } else if self.wide_gate > gate_target {
                self.wide_gate = (self.wide_gate - WIDE_GATE_STEP).max(gate_target);
            }
            let m = if blinking { 0.0 } else { pe[0].wide_out.max(pe[1].wide_out) * self.wide_gate };
            pe[0].wide_out = m;
            pe[1].wide_out = m;
        } else {
            let both_widening = pe[0].wide_out > 0.05 && pe[1].wide_out > 0.05;
            if both_widening {
                let m = pe[0].wide_out.max(pe[1].wide_out);
                pe[0].wide_out = m;
                pe[1].wide_out = m;
            }
        }

        if self.tuning.wide_squeeze_exclusive {
            for i in 0..2 {
                let (w, s) = (pe[i].wide_out, pe[i].squeeze_out);
                pe[i].wide_out = w * (1.0 - s).clamp(0.0, 1.0);
                pe[i].squeeze_out = s * (1.0 - w).clamp(0.0, 1.0);
            }
        }

        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for e in Eye::ALL {
            let i = e.idx();
            let s = pe[i];
            let target = s.openness_target;
            let prev = self.eyes[i].smooth_open;
            let gaze_invalid = !s.gaze_v;
            let fast_closing = s.draw <= FAST_CLOSING_RAW;
            let streak_gate = !self.tuning.continuous_calib && self.eyes[i].closed_streak >= 2;
            let forced_closed = streak_gate
                || target < 0.05
                || !s.present
                || (gaze_invalid && prev > 0.4 && fast_closing);
            let mut openness_out = if forced_closed {
                self.eyes[i].kalman.reset();
                0.0
            } else if self.tuning.adaptive_kalman {
                self.eyes[i].kalman.update(target)
            } else {
                let alpha_o = if target < prev { self.tuning.alpha_close } else { self.tuning.alpha_open };
                prev + alpha_o * (target - prev)
            };
            if forced_closed {
                self.eyes[i].reopening = true;
            } else if self.eyes[i].reopening {
                let ms = self.tuning.blink_reopen_ms;
                if ms >= 1.0 {
                    let step = 1000.0 / (120.0 * ms);
                    if openness_out > prev + step {
                        openness_out = prev + step;
                    }
                }
                if openness_out >= 0.9 {
                    self.eyes[i].reopening = false;
                }
            }
            self.eyes[i].smooth_open = openness_out;

            if self.tuning.gaze_yoke {
                if s.closed {
                    self.eyes[i].yoke_hold = true;
                } else if openness_out > YOKE_RELEASE_OPEN {
                    self.eyes[i].yoke_hold = false;
                }
            } else {
                self.eyes[i].yoke_hold = false;
            }

            let [gx, gy, gz] = s.gaze;
            let (sx, sy, sz) = if s.gaze_v && !self.eyes[i].yoke_hold {
                let [px, py, pz] = self.eyes[i].gaze_smooth;
                let delta = ((gx - px).powi(2) + (gy - py).powi(2) + (gz - pz).powi(2)).sqrt();
                let alpha_g = (delta * 3.0).clamp(0.15, 0.85);
                let sx = px + alpha_g * (gx - px);
                let sy = py + alpha_g * (gy - py);
                let sz = pz + alpha_g * (gz - pz);
                self.eyes[i].gaze_smooth = [sx, sy, sz];
                (sx, sy, sz)
            } else {
                let [px, py, pz] = self.eyes[i].gaze_smooth;
                (px, py, pz)
            };

            let pupil_out = if s.pupil_v {
                let pp = self.eyes[i].pupil_smooth;
                let po = pp + 0.25 * (s.pupil_mm - pp);
                self.eyes[i].pupil_smooth = po;
                po
            } else {
                self.eyes[i].pupil_smooth
            };

            out[i] = EyeResult {
                eye: e,
                gaze: [sx, sy, sz],
                gaze_valid: s.gaze_v,
                openness: openness_out,
                openness_valid: ml_loaded,
                wide: s.wide_out,
                squeeze: s.squeeze_out,
                frown: 0.0,
                pupil_mm: pupil_out,
                pupil_valid: pupil_out > 0.5,
                blink: s.closed,
                gaze_yoked: false,
                brow: 0.0,
                brow_valid: false,
            };
        }

        if self.tuning.gaze_yoke {
            for e in Eye::ALL {
                let (i, j) = (e.idx(), e.opposite().idx());
                if self.eyes[i].yoke_hold
                    && !self.eyes[j].yoke_hold
                    && !out[j].blink
                    && out[j].gaze_valid
                {
                    out[i].gaze = out[j].gaze;
                    out[i].gaze_valid = true;
                    out[i].gaze_yoked = true;
                    self.eyes[i].gaze_smooth = out[j].gaze;
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, Default)]
struct PerEye {
    raw: f32,
    openness_target: f32,
    wide_out: f32,
    squeeze_out: f32,
    gaze: [f32; 3],
    gaze_v: bool,
    pupil_mm: f32,
    pupil_v: bool,
    closed: bool,
    present: bool,
    falling: bool,
    draw: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(st: &mut SRanipalState, raw: f32, n: usize) -> [EyeResult; 2] {
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
        for _ in 0..n {
            last = st.process_frame(ml, &g, true);
        }
        last
    }


    #[test]
    fn blink_open_close_open_cycle() {
        let mut st = SRanipalState::new();

        let open = feed(&mut st, 0.62, 260);
        assert!(st.baseline(Eye::Left) > 0.58 && st.baseline(Eye::Left) < 0.66,
                "baseline learned ~0.62, got {}", st.baseline(Eye::Left));
        assert!(open[0].openness > 0.8, "warm open openness {}", open[0].openness);
        assert!(!open[0].blink, "should not be blinking when open");

        let closed = feed(&mut st, 0.30, 12);
        assert_eq!(closed[0].openness, 0.0, "blink should drive openness to 0");
        assert!(closed[0].blink, "blink flag set when closed");
        assert_eq!(closed[1].openness, 0.0, "both eyes closed");

        let reopened = feed(&mut st, 0.62, 40);
        assert!(reopened[0].openness > 0.5, "reopened openness {}", reopened[0].openness);
        assert!(!reopened[0].blink, "no longer blinking after reopen");
    }

    #[test]
    fn adaptive_kalman_tracks_then_deadzones() {
        let mut k = ScalarKalman::default();
        assert_eq!(k.update(0.70), 0.70);
        for _ in 0..40 {
            k.update(0.20);
        }
        assert!(k.x < 0.30, "fast regime should track the step down, got {}", k.x);
        let settled = k.x;
        for i in 0..60 {
            k.update(settled + if i % 2 == 0 { 0.02 } else { -0.02 });
        }
        assert!(
            (k.x - settled).abs() < 0.02,
            "deadzone should suppress jitter, drift {}",
            (k.x - settled).abs()
        );
    }

    #[test]
    fn adaptive_kalman_openness_path_matches_when_open() {
        let mut st = SRanipalState::new();
        st.tuning.adaptive_kalman = true;
        let open = feed(&mut st, 0.62, 300);
        assert!(open[0].openness > 0.7, "kalman-smoothed open openness {}", open[0].openness);
        assert!(!open[0].blink);
    }

    #[test]
    fn native_squeeze_from_channel() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            let ml = [[1.0, 0.55, 0.0, 0.60, 0.0], [1.0, 0.55, 0.0, 0.60, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(last[0].squeeze > 0.7, "native squeeze should fire, got {}", last[0].squeeze);
        assert!(last[1].squeeze > 0.7, "native squeeze (right), got {}", last[1].squeeze);
    }

    #[test]
    fn native_squeeze_per_eye_independent() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            let ml = [[1.0, 0.55, 0.0, 0.60, 0.0], [1.0, 0.62, 0.0, 0.05, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(last[0].squeeze > 0.6, "left squeeze fires, got {}", last[0].squeeze);
        assert!(last[1].squeeze < 0.1, "right squeeze stays low (no yoke), got {}", last[1].squeeze);
    }

    #[test]
    fn wide_fires_above_upper() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let wide = feed(&mut st, 0.79, 20);
        assert!(wide[0].wide > 0.05, "wide magnitude {}", wide[0].wide);
        assert!(wide[0].openness > 0.9, "wide implies fully open, got {}", wide[0].openness);
    }

    #[test]
    fn fast_blink_snaps_fully_closed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let blink = feed(&mut st, 0.48, 6);
        assert!(blink[0].openness < 0.1, "fast blink should snap closed, got {}", blink[0].openness);
        assert!(blink[0].blink, "blink flag set during fast blink");
        let reopened = feed(&mut st, 0.62, 40);
        assert!(reopened[0].openness > 0.5, "reopened after fast blink, got {}", reopened[0].openness);
        assert!(!reopened[0].blink, "no longer blinking after reopen");
    }

    #[test]
    fn blink_reopen_into_wide_then_settle_stays_open() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let blink = feed(&mut st, 0.48, 4);
        assert!(blink[0].openness < 0.1, "blink closed, got {}", blink[0].openness);
        feed(&mut st, 0.80, 6);
        let settled = feed(&mut st, 0.62, 8);
        assert!(settled[0].openness > 0.5, "settling out of wide must stay open, got {}", settled[0].openness);
        assert!(!settled[0].blink, "not blinking when open");
    }

    #[test]
    fn slow_partial_close_is_not_latched() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for k in 0..40 {
            let raw = (0.62 - 0.02 * (k.min(6) as f32)).max(0.50);
            let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(last[0].openness > 0.15, "slow partial close stays partly open, got {}", last[0].openness);
    }

    #[test]
    fn single_eye_wide_suppressed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            let ml = [[1.0, 0.79, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(last[0].wide < 1e-3, "single-eye (left) wide suppressed, got {}", last[0].wide);
        assert!(last[1].wide < 1e-3, "right stays 0, got {}", last[1].wide);
    }

    #[test]
    fn bilateral_wide_engages_smoothly_no_dump() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        for _ in 0..20 {
            let ml = [[1.0, 0.82, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
        }
        let mut first = 0.0;
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for k in 0..40 {
            let ml = [[1.0, 0.82, 0.0, 0.0, 0.0], [1.0, 0.82, 0.0, 0.0, 0.0]];
            last = st.process_frame(ml, &g, true);
            if k == 0 {
                first = last[0].wide;
            }
        }
        assert!(first < 0.25, "wide must not dump on engage, first-frame {}", first);
        assert!(last[0].wide > 0.6, "wide reaches full after the ramp, got {}", last[0].wide);
        assert!(last[0].wide > first + 0.3, "wide ramped up (from {} to {})", first, last[0].wide);
    }

    fn emit_cadence(st: &mut SRanipalState, raws_per_ml_update: &[f32]) -> Vec<f32> {
        let mut g = GazeSample::default();
        g.left.gaze_valid = true;
        g.right.gaze_valid = true;
        let mut out = Vec::new();
        for &raw in raws_per_ml_update {
            let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
            out.push(st.process_frame(ml, &g, true)[0].openness);
            out.push(st.process_frame(ml, &g, true)[0].openness);
        }
        out
    }

    fn warmed_open_state() -> SRanipalState {
        let mut st = SRanipalState::new();
        let g = GazeSample::default();
        let ml = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        for _ in 0..400 {
            st.process_frame(ml, &g, true);
        }
        st
    }

    #[test]
    fn saccade_dip_does_not_flicker_closed() {
        let mut st = warmed_open_state();
        let emitted = emit_cadence(&mut st, &[0.62, 0.55, 0.60, 0.62, 0.62]);
        let min_emit = emitted.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_emit > 0.5,
            "transient lid-dip must not flicker closed (min openness {}, want >0.5)",
            min_emit
        );
    }

    #[test]
    fn genuine_fast_blink_still_snaps_closed_under_cadence() {
        let mut st = warmed_open_state();
        let emitted = emit_cadence(&mut st, &[0.62, 0.45, 0.43, 0.43, 0.45]);
        let min_emit = emitted.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            min_emit < 0.01,
            "genuine fast blink must still reach fully closed (min openness {})",
            min_emit
        );
    }

    #[test]
    fn diag_blink_while_wide_envelope() {
        let g = GazeSample::default();
        fn step(st: &mut SRanipalState, g: &GazeSample, rl: f32, rr: f32) -> [EyeResult; 2] {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, g, true)
        }
        let mut st = SRanipalState::new();
        let mut r = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..130 {
            step(&mut st, &g, 0.62, 0.62);
            r = step(&mut st, &g, 0.62, 0.62);
        }
        for _ in 0..40 {
            step(&mut st, &g, 0.82, 0.82);
            r = step(&mut st, &g, 0.82, 0.82);
        }
        eprintln!("=== blink-while-wide envelope diagnostic ===");
        eprintln!(
            "after steady wide: gate={:.4} isW L={} R={} wideOut={:.4}",
            st.wide_gate, st.eyes[0].is_wide, st.eyes[1].is_wide, r[0].wide
        );
        let gate_held = st.wide_gate;

        eprintln!("--- blink (raw=0.39), 6 emit frames ---");
        let blink_raw = 0.39;
        for f in 0..6 {
            let rr = step(&mut st, &g, blink_raw, blink_raw);
            eprintln!(
                "  blink f{}: gate={:.4} isW L={} R={} wideOut={:.4}",
                f, st.wide_gate, st.eyes[0].is_wide, st.eyes[1].is_wide, rr[0].wide
            );
        }
        let gate_bottom = st.wide_gate;

        eprintln!("--- reopen into wide (raw=0.82) ---");
        for f in 0..10 {
            let rr = step(&mut st, &g, 0.82, 0.82);
            eprintln!(
                "  reopen f{}: gate={:.4} isW L={} R={} wideOut={:.4}",
                f, st.wide_gate, st.eyes[0].is_wide, st.eyes[1].is_wide, rr[0].wide
            );
        }
        eprintln!(
            "gate_held={:.4} gate_bottom={:.4} drop={:.4}",
            gate_held, gate_bottom, gate_held - gate_bottom
        );
        eprintln!("=============================================");
    }

    #[test]
    fn diag_genuine_disengage_still_fades() {
        let g = GazeSample::default();
        fn step(st: &mut SRanipalState, g: &GazeSample, rl: f32, rr: f32) -> [EyeResult; 2] {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, g, true)
        }
        let mut st = SRanipalState::new();
        for _ in 0..170 {
            step(&mut st, &g, 0.62, 0.62);
            step(&mut st, &g, 0.62, 0.62);
        }
        for _ in 0..40 {
            step(&mut st, &g, 0.82, 0.82);
            step(&mut st, &g, 0.82, 0.82);
        }
        eprintln!("=== genuine disengage control ===");
        eprintln!("steady wide: gate={:.4}", st.wide_gate);
        for f in 0..8 {
            let rr = step(&mut st, &g, 0.82, 0.62);
            eprintln!(
                "  disengage f{}: gate={:.4} isW L={} R={} closedR={} wideOut={:.4}",
                f, st.wide_gate, st.eyes[0].is_wide, st.eyes[1].is_wide,
                rr[1].blink, rr[0].wide
            );
        }
        eprintln!("=================================");
    }

    #[test]
    fn both_eye_wide_symmetric_pulls_up_weak_eye() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            let ml = [[1.0, 0.65, 0.0, 0.0, 0.0], [1.0, 0.82, 0.0, 0.0, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(last[1].wide > 0.05, "right wide should fire, got {}", last[1].wide);
        assert!(
            (last[0].wide - last[1].wide).abs() < 1e-6,
            "bilateral wide must be symmetric L {} R {}",
            last[0].wide,
            last[1].wide
        );
    }

    fn gaze_lr(l: [f32; 3], r: [f32; 3]) -> GazeSample {
        let mut g = GazeSample::default();
        g.left.gaze = l;
        g.left.gaze_valid = true;
        g.right.gaze = r;
        g.right.gaze_valid = true;
        g
    }

    #[test]
    fn gaze_yoke_squinting_eye_follows_open_eye() {
        let mut st = SRanipalState::new();
        let g = gaze_lr([0.4, 0.1, 0.9], [-0.4, 0.1, 0.9]);
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        for _ in 0..260 {
            st.process_frame(open, &g, true);
        }
        let squint = [[1.0, 0.30, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..12 {
            out = st.process_frame(squint, &g, true);
        }
        assert!(out[0].blink, "left eye reads squinting/closed");
        assert!(!out[1].blink, "right eye stays open");
        assert!(out[0].gaze_yoked, "left gaze is yoked to the tracked (right) eye");
        assert!(out[0].gaze_valid, "yoked gaze is emitted as valid (so it drives the avatar)");
        assert_eq!(out[0].gaze, out[1].gaze, "squinting eye mirrors the open eye's gaze exactly");
        assert!(!out[1].gaze_yoked, "the open eye is never the one yoked");
    }

    #[test]
    fn gaze_yoke_does_not_tremble_at_blink_boundary() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let own_garbage = [-0.5f32, 0.2, -1.0];
        let partner = [0.3f32, 0.0, -1.0];
        let g = gaze_lr(own_garbage, partner);
        for _ in 0..12 {
            st.process_frame([[1.0, 0.42, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]], &g, true);
        }
        let mut prev_x: Option<f32> = None;
        let mut max_step = 0.0f32;
        let mut flip = false;
        for _ in 0..60 {
            let rl = if flip { 0.425 } else { 0.435 };
            flip = !flip;
            let out = st.process_frame(
                [[1.0, rl, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
            if let Some(px) = prev_x {
                max_step = max_step.max((out[0].gaze[0] - px).abs());
            }
            prev_x = Some(out[0].gaze[0]);
            assert!(out[0].gaze_yoked, "squinting eye stays yoked through the hover");
        }
        assert!((prev_x.unwrap() - partner[0]).abs() < 0.02,
                "squinting eye pinned to the partner, got x {}", prev_x.unwrap());
        assert!(max_step < 0.02, "no gaze trembling at the boundary (max step {max_step})");
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..40 {
            out = st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        assert!(!out[0].gaze_yoked, "yoke releases after a real reopen");
        assert!((out[0].gaze[0] - own_garbage[0]).abs() < 0.05,
                "own gaze resumes after release, got x {}", out[0].gaze[0]);
    }

    #[test]
    fn bilateral_blink_recovery_does_not_cross_yoke() {
        let mut st = SRanipalState::new();
        st.tuning.blink_reopen_ms = 200.0;
        feed(&mut st, 0.62, 400);
        let left_dir = [-0.3f32, 0.1, -1.0];
        let right_dir = [0.3f32, -0.1, -1.0];
        let g = gaze_lr(left_dir, right_dir);
        for _ in 0..40 {
            st.process_frame([[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]], &g, true);
        }
        for _ in 0..8 {
            st.process_frame([[1.0, 0.36, 0.0, 0.0, 0.0], [1.0, 0.36, 0.0, 0.0, 0.0]], &g, true);
        }
        let mut crossed = false;
        for _ in 0..60 {
            let out = st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
            if out[0].gaze_yoked || out[1].gaze_yoked {
                crossed = true;
            }
            if out[0].gaze[0] > 0.0 || out[1].gaze[0] < 0.0 {
                crossed = true;
            }
        }
        assert!(!crossed, "recovering eyes must not mirror each other after a blink");
    }

    #[test]
    fn gaze_yoke_idle_when_both_eyes_track() {
        let mut st = SRanipalState::new();
        let g = gaze_lr([0.4, 0.0, 0.9], [-0.4, 0.0, 0.9]);
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..260 {
            out = st.process_frame(open, &g, true);
        }
        assert!(!out[0].gaze_yoked && !out[1].gaze_yoked, "no yoke while both eyes track");
        assert_ne!(out[0].gaze, out[1].gaze, "each open eye keeps its own gaze");
    }

    #[test]
    fn gaze_yoke_disabled_leaves_squint_eye_unyoked() {
        let mut st = SRanipalState::new();
        st.tuning.gaze_yoke = false;
        let g = gaze_lr([0.4, 0.0, 0.9], [-0.4, 0.0, 0.9]);
        for _ in 0..260 {
            st.process_frame([[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]], &g, true);
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..12 {
            out = st.process_frame([[1.0, 0.30, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]], &g, true);
        }
        assert!(out[0].blink, "left still squinting");
        assert!(!out[0].gaze_yoked, "yoke disabled → no mirroring");
        assert_ne!(out[0].gaze, out[1].gaze, "unyoked squint eye does not adopt the open eye's gaze");
    }

    #[test]
    fn wide_squeeze_chain_suppresses_the_collision() {
        let mut st = SRanipalState::new();
        st.tuning.wide_gain = 1.0;
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let ml = [[1.0, 0.82, 0.0, 0.60, 0.0], [1.0, 0.82, 0.0, 0.60, 0.0]];
        let mut off = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..25 {
            off = st.process_frame(ml, &g, true);
        }
        assert!(
            off[0].wide > 0.05 && off[0].squeeze > 0.05,
            "chain OFF: wide & squeeze both fire independently ({}, {})",
            off[0].wide, off[0].squeeze
        );

        st.tuning.wide_squeeze_exclusive = true;
        let mut on = off;
        for _ in 0..25 {
            on = st.process_frame(ml, &g, true);
        }
        assert!(on[0].wide < off[0].wide, "chain ON attenuates wide ({} -> {})", off[0].wide, on[0].wide);
        assert!(on[0].squeeze < off[0].squeeze, "chain ON attenuates squeeze ({} -> {})", off[0].squeeze, on[0].squeeze);
        assert!(on[0].wide <= off[0].wide + 1e-6 && on[0].squeeze <= off[0].squeeze + 1e-6, "chain never amplifies");
    }

    #[test]
    fn relaxed_below_baseline_reads_open_via_deadzone() {
        let mut st = SRanipalState::new();
        let g = GazeSample::default();
        for _ in 0..300 {
            st.process_frame([[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]], &g, true);
        }
        for k in 0..6 {
            let raw = 0.62 - 0.01 * (k as f32);
            st.process_frame([[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]], &g, true);
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..15 {
            out = st.process_frame([[1.0, 0.56, 0.0, 0.0, 0.0], [1.0, 0.56, 0.0, 0.0, 0.0]], &g, true);
        }
        assert!(
            out[0].openness > 0.95,
            "a relaxed eye ~0.06 below baseline must read fully open, got {}",
            out[0].openness
        );
        assert!(!out[0].blink, "not a blink — just relaxed slightly below baseline");
    }

    #[test]
    fn deadzone_still_blinks_and_wide_still_fires() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300);
        let closed = feed(&mut st, 0.30, 12);
        assert_eq!(closed[0].openness, 0.0, "blink still drives openness to 0");
        assert!(closed[0].blink);
        let wide = feed(&mut st, 0.80, 24);
        assert!(wide[0].openness > 0.95, "wide implies fully open, got {}", wide[0].openness);
        assert!(wide[0].wide > 0.05, "wide still fires with the wider dead-zone, got {}", wide[0].wide);
    }

    #[test]
    fn held_wide_does_not_drift_the_baseline() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300);
        let base0 = st.baseline(Eye::Left);
        let out = feed(&mut st, 0.68, 400);
        let base1 = st.baseline(Eye::Left);
        assert!((base1 - base0).abs() < 0.02, "held wide must not drift the baseline ({base0} -> {base1})");
        assert!(out[0].wide > 0.05, "wide still fires after a long hold, got {}", out[0].wide);
    }

    #[test]
    fn sustained_squint_does_not_sink_baseline_or_fire_wide() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.53, 1200);
        let b1 = st.baseline(Eye::Left);
        assert!((b1 - b0).abs() < 0.01, "squint sank the baseline ({b0} -> {b1})");
        let g = GazeSample::default();
        let ml = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        let mut max_wide: f32 = 0.0;
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..120 {
            last = st.process_frame(ml, &g, true);
            max_wide = max_wide.max(last[0].wide).max(last[1].wide);
        }
        assert!(max_wide < 0.01, "spurious wide after a squint: {max_wide}");
        assert!(last[0].openness > 0.9, "openness after squint {}", last[0].openness);
    }

    #[test]
    fn latched_wide_learns_slow_not_frozen() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.68, 1200);
        let b1 = st.baseline(Eye::Left);
        assert!(b1 - b0 > 5e-4, "learning froze while wide ({b0} -> {b1})");
        assert!(b1 - b0 < 0.02, "wide must learn slowly, not fast ({b0} -> {b1})");
    }

    #[test]
    fn stuck_wide_breaker_recovers_bounded() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot { baseline: 0.53, baseline_n: 5000, frame_count: 5000, blink_depth: 0.20, mid_anchor: 0.5 };
        st.restore_all(&CalibStore { left: snap, right: snap });
        let g = GazeSample::default();
        let relaxed = [[1.0, 0.66, 0.0, 0.0, 0.0], [1.0, 0.66, 0.0, 0.0, 0.0]];
        let blink = [[1.0, 0.39, 0.0, 0.0, 0.0], [1.0, 0.39, 0.0, 0.0, 0.0]];
        let mut recovered_at = None;
        let mut frame = 0u32;
        'outer: for _ in 0..12 {
            for _ in 0..300 {
                let r = st.process_frame(relaxed, &g, true);
                frame += 1;
                if r[0].wide < 1e-3 && (st.baseline(Eye::Left) - 0.66).abs() < 0.02 {
                    recovered_at = Some(frame);
                    break 'outer;
                }
            }
            for _ in 0..12 {
                st.process_frame(blink, &g, true);
                frame += 1;
            }
        }
        let at = recovered_at.expect("never recovered from a stale-low baseline");
        assert!(at <= 3000, "recovery took {at} frames (> 25s)");
    }

    #[test]
    fn short_wide_hold_unaffected_by_breaker() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        let last = feed(&mut st, 0.75, 1200);
        assert!(last[0].wide > 0.3, "wide collapsed during a 10s hold: {}", last[0].wide);
        let drift = (st.baseline(Eye::Left) - b0).abs();
        assert!(drift < 0.012, "baseline drifted {drift} under a held wide");
    }

    #[test]
    fn recenter_is_noise_tolerant() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.recenter();
        feed(&mut st, 0.70, 1);
        let b = st.baseline(Eye::Left);
        assert!(b < 0.65, "one frame after recenter snapped the baseline to {b}");
        feed(&mut st, 0.62, 300);
        let b2 = st.baseline(Eye::Left);
        assert!((b2 - 0.62).abs() < 0.01, "baseline should settle back to 0.62, got {b2}");
    }

    #[test]
    fn burst_wide_spam_does_not_trip_breaker() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        let mut last_wide = 0.0f32;
        for _ in 0..6 {
            let w = feed(&mut st, 0.75, 360);
            last_wide = w[0].wide;
            feed(&mut st, 0.62, 96);
        }
        assert!(last_wide > 0.3, "wide collapsed under burst spam: {last_wide}");
        let drift = (st.baseline(Eye::Left) - b0).abs();
        assert!(drift < 0.02, "baseline drifted {drift} under burst spam");
    }

    #[test]
    fn recenter_during_stuck_episode_recovers_bounded() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot { baseline: 0.53, baseline_n: 5000, frame_count: 5000, blink_depth: 0.20, mid_anchor: 0.5 };
        st.restore_all(&CalibStore { left: snap, right: snap });
        let g = GazeSample::default();
        let relaxed = [[1.0, 0.66, 0.0, 0.0, 0.0], [1.0, 0.66, 0.0, 0.0, 0.0]];
        for _ in 0..600 {
            st.process_frame(relaxed, &g, true);
        }
        st.recenter();
        let mut recovered_at = None;
        for f in 0..4200u32 {
            let r = st.process_frame(relaxed, &g, true);
            if r[0].wide < 1e-3 && (st.baseline(Eye::Left) - 0.66).abs() < 0.02 {
                recovered_at = Some(f);
                break;
            }
        }
        let at = recovered_at.expect("recenter wedged the stuck-wide recovery");
        assert!(at <= 1800, "recovery after recenter took {at} frames (> 15s)");
    }

    #[test]
    fn fast_blink_yokes_lagging_eye_closed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = GazeSample::default();
        let step = |st: &mut SRanipalState, rl: f32, rr: f32| {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true)
        };
        step(&mut st, 0.50, 0.57);
        step(&mut st, 0.40, 0.52);
        let mut min_right: f32 = 1.0;
        for _ in 0..6 {
            let out = step(&mut st, 0.40, 0.49);
            min_right = min_right.min(out[1].openness);
            assert_eq!(out[0].openness, 0.0, "latched left eye reads 0");
        }
        assert_eq!(min_right, 0.0, "lagging right eye must be yoked to 0, got {min_right}");
        step(&mut st, 0.62, 0.56);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            out = step(&mut st, 0.62, 0.62);
        }
        assert!(out[0].openness > 0.8 && out[1].openness > 0.8,
                "both eyes reopen after the yoked blink ({} / {})",
                out[0].openness, out[1].openness);
    }

    fn slow_blink(st: &mut SRanipalState, top: f32, bottom: f32, dwell: usize) {
        let mut raw = top;
        while raw > bottom {
            raw -= 0.02;
            feed(st, raw.max(bottom), 1);
        }
        feed(st, bottom, dwell);
        while raw < top {
            raw += 0.02;
            feed(st, raw.min(top), 1);
        }
    }

    #[test]
    fn slow_blink_calibrates_blink_depth_and_it_sticks() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        slow_blink(&mut st, 0.62, 0.36, 60);
        let after = st.closed_ref(Eye::Left);
        assert!(after < before - 0.015,
                "slow blink must deepen closed_ref ({before} -> {after})");
        slow_blink(&mut st, 0.62, 0.36, 60);
        let after2 = st.closed_ref(Eye::Left);
        assert!(after2 < after - 0.005,
                "second slow blink converges further ({after} -> {after2})");
        feed(&mut st, 0.62, 3600);
        let held = st.closed_ref(Eye::Left);
        assert!((held - after2).abs() < 0.006,
                "learned depth must not decay away ({after2} -> {held})");
    }

    #[test]
    fn fast_blink_does_not_teach_blink_depth() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        for _ in 0..5 {
            feed(&mut st, 0.30, 14);
            feed(&mut st, 0.62, 120);
        }
        let after = st.closed_ref(Eye::Left);
        assert!((after - before).abs() < 0.01,
                "fast blinks must not move closed_ref ({before} -> {after})");
    }

    #[test]
    fn resting_closed_eyes_do_not_teach_blink_depth() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        slow_blink(&mut st, 0.62, 0.34, 500);
        let after = st.closed_ref(Eye::Left);
        assert!((after - before).abs() < 0.01,
                "resting closed must not teach depth ({before} -> {after})");
    }

    #[test]
    fn blink_depth_survives_recenter_and_restore() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.36, 60);
        slow_blink(&mut st, 0.62, 0.36, 60);
        let learned = st.closed_ref(Eye::Left);
        assert!(learned < 0.415, "depth learned before the round-trip, got {learned}");
        st.recenter();
        feed(&mut st, 0.62, 300);
        let after_recenter = st.closed_ref(Eye::Left);
        assert!((after_recenter - learned).abs() < 0.01,
                "recenter must keep the learned depth ({learned} -> {after_recenter})");
        let store = st.snapshot_all();
        assert!(store.left.blink_depth > 0.21, "depth persisted: {}", store.left.blink_depth);
        let mut st2 = SRanipalState::new();
        st2.restore_all(&store);
        feed(&mut st2, 0.62, 300);
        let restored = st2.closed_ref(Eye::Left);
        assert!((restored - learned).abs() < 0.015,
                "restore must keep the learned depth ({learned} -> {restored})");
    }

    #[test]
    fn old_calib_file_without_depth_loads_default() {
        let toml_text = "[left]\nbaseline = 0.62\nbaseline_n = 5000\nframe_count = 5000\n\
                         [right]\nbaseline = 0.61\nbaseline_n = 5000\nframe_count = 5000\n";
        let store: CalibStore = toml::from_str(toml_text).expect("old calib file must parse");
        assert_eq!(store.left.blink_depth, BLINK_OFFSET);
        assert_eq!(store.left.mid_anchor, 0.5);
        let mut st = SRanipalState::new();
        st.restore_all(&store);
        assert!((st.closed_ref(Eye::Left) - (0.62 - BLINK_OFFSET)).abs() < 1e-4);
        assert_eq!(st.eyes[0].mid_anchor, 0.5);
    }

    fn warped_right(raw_l: f32) -> f32 {
        let t = ((raw_l - 0.36) / 0.26).clamp(0.0, 1.0);
        0.36 + 0.26 * t.powf(1.2)
    }

    fn step_pair(st: &mut SRanipalState, rl: f32, rr: f32) -> [EyeResult; 2] {
        let g = GazeSample::default();
        let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
        st.process_frame(ml, &g, true);
        st.process_frame(ml, &g, true)
    }

    fn warped_slow_close_cycle(st: &mut SRanipalState) {
        let mut raw = 0.62f32;
        while raw > 0.36 {
            raw -= 0.01;
            step_pair(st, raw.max(0.36), warped_right(raw.max(0.36)));
        }
        for _ in 0..25 {
            step_pair(st, 0.36, 0.36);
        }
        while raw < 0.62 {
            raw += 0.01;
            step_pair(st, raw.min(0.62), warped_right(raw.min(0.62)));
        }
        for _ in 0..60 {
            step_pair(st, 0.62, 0.62);
        }
    }

    fn mid_close_gap(st: &mut SRanipalState) -> f32 {
        let mut raw = 0.62f32;
        while raw > 0.48 {
            raw -= 0.01;
            step_pair(st, raw.max(0.48), warped_right(raw.max(0.48)));
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..30 {
            out = step_pair(st, 0.48, warped_right(0.48));
        }
        while raw < 0.62 {
            raw += 0.01;
            step_pair(st, raw.min(0.62), warped_right(raw.min(0.62)));
        }
        for _ in 0..30 {
            step_pair(st, 0.62, 0.62);
        }
        (out[0].openness - out[1].openness).abs()
    }

    #[test]
    fn symmetric_slow_close_learns_and_equalizes() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let gap_before = mid_close_gap(&mut st);
        assert!(gap_before > 0.10, "warp should produce a visible gap, got {gap_before}");
        for _ in 0..4 {
            warped_slow_close_cycle(&mut st);
        }
        let (a_l, a_r) = (st.eyes[0].mid_anchor, st.eyes[1].mid_anchor);
        assert!(a_l > 0.52 && a_r < 0.48, "anchors must move oppositely (L {a_l} / R {a_r})");
        assert!((a_l + a_r - 1.0).abs() < 1e-3,
                "pair mean pinned to identity (L {a_l} / R {a_r})");
        let gap_after = mid_close_gap(&mut st);
        assert!(gap_after < 0.06 && gap_after < 0.5 * gap_before,
                "mid-close gap must shrink ({gap_before} -> {gap_after})");
    }

    #[test]
    fn winks_never_teach_the_curve() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..6 {
            let mut raw = 0.62f32;
            let mut noisy = false;
            while raw > 0.36 {
                raw -= 0.01;
                let rr = if noisy { 0.60 } else { 0.64 };
                noisy = !noisy;
                step_pair(&mut st, raw.max(0.36), rr);
            }
            for _ in 0..25 {
                step_pair(&mut st, 0.36, 0.62);
            }
            while raw < 0.62 {
                raw += 0.01;
                step_pair(&mut st, raw.min(0.62), 0.62);
            }
            for _ in 0..40 {
                step_pair(&mut st, 0.62, 0.62);
            }
        }
        assert_eq!(st.eyes[0].mid_anchor, 0.5, "wink taught the left anchor");
        assert_eq!(st.eyes[1].mid_anchor, 0.5, "wink taught the right anchor");
        assert!(st.eyes[0].mid_anchor_staged.is_none() && st.eyes[1].mid_anchor_staged.is_none());
    }

    #[test]
    fn apply_anchor_identity_and_endpoints() {
        for k in 0..=20 {
            let x = k as f32 / 20.0;
            assert_eq!(apply_anchor(x, 0.5), x, "identity broken at {x}");
        }
        for a in [0.30_f32, 0.38, 0.5, 0.62, 0.70] {
            assert_eq!(apply_anchor(0.0, a), 0.0);
            assert_eq!(apply_anchor(1.0, a), 1.0);
            assert!((apply_anchor(a, a) - 0.5).abs() < 1e-6);
            let mut prev = -1.0;
            for k in 0..=40 {
                let y = apply_anchor(k as f32 / 40.0, a);
                assert!(y >= prev, "not monotone at anchor {a}");
                prev = y;
            }
        }
    }

    #[test]
    fn staged_anchor_swaps_only_at_full_open() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let mut raw = 0.62f32;
        while raw > 0.48 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.48), raw.max(0.48));
        }
        st.eyes[0].mid_anchor_staged = Some(0.62);
        for _ in 0..30 {
            step_pair(&mut st, 0.48, 0.48);
        }
        assert_eq!(st.eyes[0].mid_anchor, 0.5, "anchor swapped mid-close");
        assert!(st.eyes[0].mid_anchor_staged.is_some());
        while raw < 0.62 {
            raw += 0.01;
            step_pair(&mut st, raw.min(0.62), raw.min(0.62));
        }
        step_pair(&mut st, 0.62, 0.62);
        assert_eq!(st.eyes[0].mid_anchor, 0.62, "staged anchor must land at full open");
        assert!(st.eyes[0].mid_anchor_staged.is_none());
    }

    #[test]
    fn anchors_persist_and_survive_recenter() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.eyes[0].mid_anchor = 0.62;
        st.eyes[1].mid_anchor = 0.38;
        st.recenter();
        assert_eq!(st.eyes[0].mid_anchor, 0.62, "recenter must keep the anchor");
        let store = st.snapshot_all();
        assert_eq!(store.left.mid_anchor, 0.62);
        let mut st2 = SRanipalState::new();
        st2.restore_all(&store);
        assert_eq!(st2.eyes[0].mid_anchor, 0.62);
        assert_eq!(st2.eyes[1].mid_anchor, 0.38);
        let bad = CalibSnapshot {
            baseline: 0.62,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: f32::NAN,
        };
        st2.restore(Eye::Left, bad);
        assert_eq!(st2.eyes[0].mid_anchor, 0.5, "NaN anchor must sanitize to identity");
    }

    #[test]
    fn stale_partner_exit_does_not_deadlock_pairing() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        {
            let mut raw = 0.62f32;
            while raw > 0.36 {
                raw -= 0.01;
                step_pair(&mut st, 0.62, raw.max(0.36));
            }
            for _ in 0..20 {
                step_pair(&mut st, 0.62, 0.36);
            }
            while raw < 0.62 {
                raw += 0.01;
                step_pair(&mut st, 0.62, raw.min(0.62));
            }
            for _ in 0..40 {
                step_pair(&mut st, 0.62, 0.62);
            }
        }
        assert!(st.eyes[1].ep_clean_exit.is_some(), "stale right exit seeded");
        for _ in 0..2 {
            let mut raw = 0.62f32;
            while raw > 0.36 {
                raw -= 0.01;
                step_pair(&mut st, raw.max(0.36), (raw - 0.02).max(0.36));
            }
            for _ in 0..25 {
                step_pair(&mut st, 0.36, 0.36);
            }
            while raw < 0.62 {
                raw += 0.01;
                step_pair(&mut st, raw.min(0.62), (raw - 0.02).min(0.62));
            }
            for _ in 0..60 {
                step_pair(&mut st, 0.62, 0.62);
            }
        }
        let (a_l, a_r) = (st.eyes[0].mid_anchor, st.eyes[1].mid_anchor);
        assert!(a_l > 0.5 && a_r < 0.5,
                "offset gestures must still commit (L {a_l} / R {a_r})");
    }

    #[test]
    fn low_raw_setup_baseline_learns() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.43, 300);
        let b = st.baseline(Eye::Left);
        assert!((b - 0.43).abs() < 0.01, "cold start at raw 0.43 must learn, got {b}");
        let snap = CalibSnapshot {
            baseline: 0.51, baseline_n: 5000, frame_count: 5000,
            blink_depth: 0.29, mid_anchor: 0.5,
        };
        let mut st2 = SRanipalState::new();
        st2.restore_all(&CalibStore { left: snap, right: snap });
        st2.recenter();
        feed(&mut st2, 0.43, 300);
        let b2 = st2.baseline(Eye::Right);
        assert!((b2 - 0.43).abs() < 0.015,
                "recenter must re-anchor a stale-high baseline at raw 0.43, got {b2}");
    }

    #[test]
    fn asymmetric_hold_before_blink_does_not_teach() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..3 {
            let (mut rl, mut rr) = (0.62f32, 0.62f32);
            while rl > 0.50 {
                rl -= 0.01;
                rr -= 0.013;
                step_pair(&mut st, rl.max(0.50), rr.max(0.46));
            }
            for _ in 0..150 {
                step_pair(&mut st, 0.50, 0.46);
            }
            let mut raw = 0.48f32;
            while raw > 0.36 {
                raw -= 0.01;
                step_pair(&mut st, raw.max(0.36), raw.max(0.36));
            }
            for _ in 0..25 {
                step_pair(&mut st, 0.36, 0.36);
            }
            while raw < 0.62 {
                raw += 0.01;
                step_pair(&mut st, raw.min(0.62), raw.min(0.62));
            }
            for _ in 0..60 {
                step_pair(&mut st, 0.62, 0.62);
            }
        }
        let (a_l, a_r) = (st.eyes[0].mid_anchor, st.eyes[1].mid_anchor);
        assert!((a_l - 0.5).abs() < 0.015 && (a_r - 0.5).abs() < 0.015,
                "asymmetric hold taught the anchors (L {a_l} / R {a_r})");
    }

    #[test]
    fn underweight_episode_consumes_exits() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let mut raw = 0.62f32;
        while raw > 0.36 {
            raw -= 0.025;
            step_pair(&mut st, raw.max(0.36), raw.max(0.36));
        }
        for _ in 0..20 {
            step_pair(&mut st, 0.36, 0.36);
        }
        while raw < 0.62 {
            raw += 0.025;
            step_pair(&mut st, raw.min(0.62), raw.min(0.62));
        }
        for _ in 0..10 {
            step_pair(&mut st, 0.62, 0.62);
        }
        assert!(st.eyes[0].ep_clean_exit.is_none() && st.eyes[1].ep_clean_exit.is_none(),
                "an evaluated episode pair must consume its exits");
        let (mut rl, mut rr) = (0.62f32, 0.62f32);
        while rl > 0.49 {
            rl -= 0.01;
            rr -= 0.013;
            step_pair(&mut st, rl.max(0.49), rr.max(0.45));
        }
        for _ in 0..120 {
            step_pair(&mut st, 0.49, 0.45);
        }
        assert_eq!(st.eyes[0].mid_anchor, 0.5);
        assert!(st.eyes[0].mid_anchor_staged.is_none() && st.eyes[1].mid_anchor_staged.is_none(),
                "post-episode samples must never commit");
    }

    #[test]
    fn gaze_invalid_midclose_noise_does_not_force_close() {
        let mut st = SRanipalState::new();
        st.tuning.alpha_close = 1.0;
        feed(&mut st, 0.62, 400);
        st.eyes[0].mid_anchor = MID_ANCHOR_MIN;
        st.eyes[1].mid_anchor = MID_ANCHOR_MAX;
        let mut raw = 0.62f32;
        while raw > 0.48 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.48), raw.max(0.48));
        }
        let mut lo = 1.0f32;
        let mut flip = false;
        for _ in 0..120 {
            let r = if flip { 0.46 } else { 0.50 };
            flip = !flip;
            let out = step_pair(&mut st, r, r);
            for o in out.iter() {
                lo = lo.min(o.openness);
            }
        }
        assert!(lo > 0.15, "gaze-invalid mid-close noise forced a close (lo {lo})");
    }

    #[test]
    fn corrected_midclose_hold_never_flaps_at_alpha1() {
        let mut st = SRanipalState::new();
        st.tuning.alpha_close = 1.0;
        feed(&mut st, 0.62, 400);
        st.eyes[0].mid_anchor = MID_ANCHOR_MIN;
        st.eyes[1].mid_anchor = MID_ANCHOR_MAX;
        let mut raw = 0.62f32;
        while raw > 0.48 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.48), raw.max(0.48));
        }
        let (mut lo, mut hi) = (1.0f32, 0.0f32);
        let mut flip = false;
        for _ in 0..120 {
            let r = if flip { 0.46 } else { 0.50 };
            flip = !flip;
            let out = step_pair(&mut st, r, r);
            for o in out.iter() {
                lo = lo.min(o.openness);
                hi = hi.max(o.openness);
            }
        }
        assert!(lo > 0.02, "mid-close noise must not excurse to closed (lo {lo})");
        assert!(hi < 0.98, "mid-close noise must not excurse to open (hi {hi})");
    }

    #[test]
    fn weak_eye_blink_commits_via_relative_threshold() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let mut min_open = 1.0f32;
        for r in [0.57f32, 0.52, 0.50, 0.50, 0.52, 0.57, 0.62] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                min_open = min_open.min(out[0].openness);
            }
        }
        assert_eq!(min_open, 0.0, "a -0.05/update blink must commit (the old code missed it)");
    }

    #[test]
    fn latch_holds_through_bottom_tremor() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let seq = [0.50f32, 0.40, 0.395, 0.401, 0.397, 0.403, 0.398, 0.43, 0.50, 0.62];
        let mut opens = Vec::new();
        for r in seq {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                opens.push(out[0].openness);
            }
        }
        let committed = opens.iter().position(|&o| o == 0.0).expect("must commit");
        let reopen_idx = 14;
        for (k, &o) in opens.iter().enumerate().take(reopen_idx).skip(committed) {
            assert_eq!(o, 0.0, "latch must hold through bottom tremor (frame {k} read {o})");
        }
        assert!(*opens.last().unwrap() > 0.4, "reopens after the real rise");
    }

    #[test]
    fn blink_reopen_slider_slows_recovery_only() {
        let mut st = SRanipalState::new();
        st.tuning.blink_reopen_ms = 200.0;
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for r in [0.50f32, 0.38, 0.38] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                out = st.process_frame(ml, &g, true);
            }
        }
        assert_eq!(out[0].openness, 0.0, "close is not slowed");
        let mut prev = 0.0f32;
        let mut frames_to_recover = 0;
        for _ in 0..80 {
            let ml = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            let o = st.process_frame(ml, &g, true)[0].openness;
            if o < 0.9 {
                assert!(o - prev <= 1.0 / 24.0 + 1e-4,
                        "reopen must be slew-limited (step {})", o - prev);
                frames_to_recover += 1;
            }
            prev = o;
            if o >= 0.9 {
                break;
            }
        }
        assert!(frames_to_recover >= 20,
                "200ms recovery must take >= ~21 frames, took {frames_to_recover}");
        let mut st2 = SRanipalState::new();
        feed(&mut st2, 0.62, 400);
        for r in [0.50f32, 0.38, 0.38] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                st2.process_frame(ml, &g, true);
            }
        }
        let mut fast_frames = 0;
        for _ in 0..80 {
            let ml = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            let o = st2.process_frame(ml, &g, true)[0].openness;
            fast_frames += 1;
            if o >= 0.9 {
                break;
            }
        }
        assert!(fast_frames < 15, "default recovery stays fast, took {fast_frames}");
    }

    #[test]
    fn held_wink_does_not_bounce() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let mut hold_max: f32 = 0.0;
        let seq: Vec<f32> = std::iter::once(0.50)
            .chain(std::iter::repeat(0.45).take(40))
            .collect();
        for (k, r) in seq.iter().enumerate() {
            let ml = [[1.0, *r, 0.0, 0.0, 0.0], [1.0, *r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                if k > 3 {
                    hold_max = hold_max.max(out[0].openness);
                }
            }
        }
        assert_eq!(hold_max, 0.0, "a held deep close must stay 0 (no bounce), got {hold_max}");
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for r in [0.50f32, 0.56, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                out = st.process_frame(ml, &g, true);
            }
        }
        assert!(out[0].openness > 0.7, "reopens after the hold, got {}", out[0].openness);
    }

    #[test]
    fn hard_close_overshoot_recovery_stays_closed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let seq = [0.50f32, 0.38, 0.36, 0.40, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44];
        let mut after_commit_max: f32 = 0.0;
        for (k, r) in seq.iter().enumerate() {
            let ml = [[1.0, *r, 0.0, 0.0, 0.0], [1.0, *r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                if k >= 3 {
                    after_commit_max = after_commit_max.max(out[0].openness);
                }
            }
        }
        assert_eq!(after_commit_max, 0.0,
                   "overshoot recovery (+0.08, still in the closed zone) must not release");
    }

    #[test]
    fn release_resets_descent_evidence() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let seq = [0.50f32, 0.40, 0.43, 0.435, 0.43, 0.44, 0.50, 0.62];
        let mut committed = false;
        let mut released = false;
        let mut re_latched = false;
        for r in seq {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                if out[0].openness == 0.0 {
                    if released {
                        re_latched = true;
                    } else {
                        committed = true;
                    }
                } else if committed && out[0].openness > 0.02 && r >= 0.5 {
                    released = true;
                }
            }
        }
        assert!(committed && released, "must commit then release (c={committed} r={released})");
        assert!(!re_latched, "stale evidence must not re-latch after the release");
    }

    #[test]
    fn bilateral_squint_onset_never_commits() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let mut min_open = 1.0f32;
        for r in [0.58f32, 0.55, 0.51, 0.51, 0.51, 0.51, 0.51, 0.51] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                let out = st.process_frame(ml, &g, true);
                min_open = min_open.min(out[0].openness);
            }
        }
        assert!(min_open > 0.3, "bilateral squint must stay on the ramp, got {min_open}");
    }

    #[test]
    fn medium_blink_without_latch_closes_both() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = GazeSample::default();
        let step = |st: &mut SRanipalState, rl: f32, rr: f32| {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true)
        };
        step(&mut st, 0.57, 0.59);
        step(&mut st, 0.52, 0.56);
        step(&mut st, 0.47, 0.53);
        step(&mut st, 0.43, 0.50);
        let mut both_zero = false;
        for _ in 0..4 {
            let out = step(&mut st, 0.43, 0.50);
            both_zero |= out[0].openness == 0.0 && out[1].openness == 0.0;
        }
        assert!(both_zero, "medium-speed bilateral blink must close BOTH eyes");
        step(&mut st, 0.55, 0.56);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            out = step(&mut st, 0.62, 0.62);
        }
        assert!(out[0].openness > 0.8 && out[1].openness > 0.8,
                "both eyes reopen after the anchored blink ({} / {})",
                out[0].openness, out[1].openness);
    }

    #[test]
    fn deep_slow_droop_stays_on_ramp() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = GazeSample::default();
        let mut min_open: f32 = 1.0;
        let mut raw = 0.62f32;
        while raw > 0.44 {
            raw -= 0.01;
            let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true);
        }
        for _ in 0..30 {
            let ml = [[1.0, 0.44, 0.0, 0.0, 0.0], [1.0, 0.44, 0.0, 0.0, 0.0]];
            let out = st.process_frame(ml, &g, true);
            min_open = min_open.min(out[0].openness).min(out[1].openness);
        }
        assert!(min_open > 0.05,
                "a slow deep droop must stay on the ramp (never snap to 0), got {min_open}");
    }

    #[test]
    fn wink_is_not_yoked_by_fast_blink() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = GazeSample::default();
        let step = |st: &mut SRanipalState, rl: f32, rr: f32| {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true)
        };
        step(&mut st, 0.50, 0.60);
        step(&mut st, 0.40, 0.57);
        for _ in 0..8 {
            let out = step(&mut st, 0.40, 0.57);
            assert_eq!(out[0].openness, 0.0, "winking eye reads closed");
            assert!(out[1].openness > 0.5,
                    "open eye must NOT be yoked during a wink, got {}", out[1].openness);
        }
    }

    #[test]
    fn two_wide_holds_with_short_breath_keep_firing() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.75, 1200);
        feed(&mut st, 0.62, 42);
        let last = feed(&mut st, 0.75, 1200);
        assert!(last[0].wide > 0.3, "wide collapsed in hold #2: {}", last[0].wide);
        let drift = (st.baseline(Eye::Left) - b0).abs();
        assert!(drift < 0.02, "baseline drifted {drift} across breath-separated holds");
    }

    #[test]
    fn restore_unlearned_snapshot_keeps_cold_start() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot { baseline: 0.60, baseline_n: 0, frame_count: 0, blink_depth: 0.20, mid_anchor: 0.5 };
        st.restore_all(&CalibStore { left: snap, right: snap });
        feed(&mut st, 0.66, 300);
        let b = st.baseline(Eye::Left);
        assert!((b - 0.66).abs() < 0.01, "cold start after idle-session restore stuck at {b}");
        let out = feed(&mut st, 0.66, 120);
        assert!(out[0].wide < 0.01, "spurious wide after cold-start restore: {}", out[0].wide);
    }

    #[test]
    fn stale_high_restore_stays_open_and_heals() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot { baseline: 0.72, baseline_n: 5000, frame_count: 5000, blink_depth: 0.20, mid_anchor: 0.5 };
        st.restore_all(&CalibStore { left: snap, right: snap });
        let out = feed(&mut st, 0.66, 240);
        assert!(out[0].openness > 0.9, "stale-high restore must read open, got {}", out[0].openness);
        assert!(!out[0].blink, "relaxed eye under a stale-high baseline is not a blink");
    }



}
