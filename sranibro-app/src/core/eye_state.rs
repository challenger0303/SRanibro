
use super::gaze_smoothing::{adaptive_gaze_step, INITIAL_GAZE};
use super::types::{
    BlinkTimingProfile, Eye, EyeResult, GazeEyelidProfile, GazeSample, WinkProfile,
};

const OPEN_TRUST_FRAMES: u32 = 20;
const BLINK_RESET_FRAMES: u32 = 5;
const WARMUP_FRAMES: u32 = 200;
const KALMAN_ALPHA: f32 = 0.30;
const UPPER_OFFSET: f32 = 0.02;
const SQUEEZE_TOP_OFFSET: f32 = 0.03;
const DEFAULT_BASELINE: f32 = 0.60;
const BLINK_OFFSET: f32 = 0.20;
const BLINK_RELEASE_OFFSET: f32 = 0.10;
const WIDE_RELEASE_OFFSET: f32 = 0.01;
const WIDE_NEUTRAL_SAMPLE_CAP: usize = 96;
const WIDE_NEUTRAL_MIN_SAMPLES: usize = 48;
const WIDE_NEUTRAL_SIGMA_MULT: f32 = 3.0;
const WIDE_ENTRY_MIN_OFFSET: f32 = 0.006;
const WIDE_ENTRY_MAX_OFFSET: f32 = 0.030;
const WIDE_CEIL_MIN_SPAN: f32 = 0.15;
const NATIVE_WIDE_MIN_SPAN: f32 = 0.04;
const WIDE_CEIL_DECAY: f32 = 0.0008;
const SQ_FLOOR_TRACK: f32 = 0.005;
const BLINK_BOTTOM_SMOOTH: f32 = 0.3;
const BLINK_LEARN_MIN_FRAMES: u32 = 18;
const BLINK_LEARN_MAX_FRAMES: u32 = 240;

const FLOOR_DEEP_CONFIRM_N: u8 = 2;
const FLOOR_SHALLOW_CONFIRM_N: u8 = 3;
const FLOOR_DEEP_CONFIRM_TOL: f32 = 0.03;
const FLOOR_SHALLOW_CONFIRM_TOL: f32 = 0.02;
const FLOOR_SHALLOW_ESCAPE: f32 = 0.02;
const FLOOR_DEEP_COMMIT_CAP: f32 = 0.03;
const FLOOR_SHALLOW_COMMIT_CAP: f32 = 0.015;
const FLOOR_TRAVEL_BUDGET: f32 = 0.08;
const FLOOR_COMMIT_MIN_FRAMES: u32 = 600;
const FLOOR_CANDIDATE_MAX_AGE: u32 = 10_800;

const FLOOR_CURVE_NEAR: f32 = 0.02;
const ENV_ATTACK: f32 = 0.02;
const ENV_HEAL: f32 = 0.00005;
const ENV_SANITY_OFFSET: f32 = 0.40;
const GATE_FRAC: f32 = 0.55;
const GATE_MIN: f32 = 0.08;
const GENUINE_NEAR: f32 = 0.04;
const FIRST_CLOSE_CONFIRM_TOL: f32 = 0.02;
const MIN_CLOSE_SPAN: f32 = 0.05;
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
const YOKE_ENGAGE_OPEN: f32 = 0.35;
const YOKE_RELEASE_OPEN: f32 = 0.55;
const WINK_ARM_FRAMES: u8 = 5;
const WINK_OWN_ENTER_OPEN: f32 = 0.78;
const WINK_SQUEEZE_ENTER_OPEN: f32 = 0.90;
const WINK_PARTNER_OPEN: f32 = 0.86;
const WINK_RELEASE_OPEN: f32 = 0.92;
const WINK_LEARN_SQUEEZE_DELTA: f32 = 0.10;
const BLINK_PULSE_MAX_FRAMES: u16 = 48;

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
const NATIVE_SQUEEZE_CLOSED_DELAY: u16 = 8;
const NATIVE_SQUEEZE_ATTACK_ALPHA: f32 = 0.18;
const NATIVE_SQUEEZE_RELEASE_ALPHA: f32 = 0.30;
const NATIVE_SQUEEZE_PEAK_RELEASE_ALPHA: f32 = 0.025;
const SQUEEZE_HARDCLOSE_T: f32 = 0.70;
const CH0_PRESENT_GATE: f32 = 0.05;

const CLOSE_DEPTH_SCALE_MIN: f32 = 0.85;
const CLOSE_DEPTH_SCALE_MAX: f32 = 1.15;
const CURVE_MID_OUTPUT_MIN: f32 = 0.35;
const CURVE_MID_OUTPUT_MAX: f32 = 0.65;
const SNAP_GATE_OPEN_MIN: f32 = 0.15;
const SNAP_GATE_OPEN_MAX: f32 = 1.0;
const DEFAULT_SNAP_GATE_OPEN: f32 = 1.0;
const AUTO_RESEAT_NATIVE_OPEN_MIN: f32 = 0.95;
const BLINK_CLOSE_MS_MAX: f32 = 160.0;
const AUTO_RESEAT_STABLE_FRAMES: u16 = 480;
const AUTO_RESEAT_CANDIDATE_BAND: f32 = 0.015;
const AUTO_RESEAT_PAIR_COHERENCE: f32 = 0.20;
const AUTO_RESEAT_SQUEEZE_MAX: f32 = 0.05;
const AUTO_RESEAT_CENTER_DEG: f32 = 12.0;
const AUTO_RESEAT_OFFSET_CAP: f32 = 0.12;
const AUTO_RESEAT_DEPTH_CAP_FRAC: f32 = 0.60;
const AUTO_RESEAT_OFFSET_STEP: f32 = 0.0015;
const APPEARANCE_BASELINE_STEP: f32 = 0.0010;
const APPEARANCE_WIDE_STEP: f32 = 0.0015;
const APPEARANCE_OFFSET_CAP: f32 = 0.20;

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

fn apply_response_curve(x: f32, midpoint: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    let midpoint = midpoint.clamp(CURVE_MID_OUTPUT_MIN, CURVE_MID_OUTPUT_MAX);
    if midpoint.to_bits() == 0.5f32.to_bits() {
        x
    } else if x <= 0.5 {
        x * (2.0 * midpoint)
    } else {
        midpoint + (x - 0.5) * (2.0 * (1.0 - midpoint))
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
        Self {
            x: 1.0,
            p: KAL_P0,
            init: false,
        }
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
    pub smoothing: f32,
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
    pub brow_lr_sync: bool,
    pub open_deadzone: f32,
    pub blink_reopen_ms: f32,
    pub blink_close_ms: f32,












}

impl Default for Tuning {
    fn default() -> Self {
        Self {
            alpha_open: 0.35,
            alpha_close: 0.75,
            smoothing: 0.0,
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
            brow_lr_sync: false,
            open_deadzone: 0.08,
            blink_reopen_ms: 0.0,
            blink_close_ms: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct EyeState {
    baseline: f32,
    baseline_n: u32,
    recenter_active: bool,
    upper: f32,
    wide_entry_offset: f32,
    wide_neutral_samples: [f32; WIDE_NEUTRAL_SAMPLE_CAP],
    wide_neutral_n: u8,
    wide_ceiling: f32,
    squeeze_floor: f32,
    closed_ref: f32,
    blink_depth: f32,
    blink_len: u32,
    blink_smooth: f32,
    blink_lo: f32,
    blink_dirty: bool,
    reach_env: f32,
    learned_once: bool,
    endpoint_locked: bool,
    endpoint_calibrated_unix: u64,
    blink_candidate: Option<f32>,
    blink_candidate_count: u8,
    blink_candidate_max: f32,
    blink_candidate_frame: u32,
    floor_last_commit_frame: u32,
    floor_travel_deeper: f32,
    floor_travel_shallower: f32,
    mid_anchor: f32,
    mid_anchor_staged: Option<f32>,
    last_ramp_pre: f32,
    ep_entry_frame: u32,
    ep_clean_exit: Option<(u32, u32)>,
    frame_count: u32,
    open_streak: u32,
    closed_streak: u32,
    squeeze_streak: u32,
    native_squeeze_closed_frames: u16,
    native_squeeze_peak: f32,
    native_squeeze_envelope: f32,
    is_closed: bool,
    is_wide: bool,
    wide_frames: u32,
    fall_run: f32,
    fall_updates: u32,
    since_down: u32,
    prev_raw: f32,
    native_was_enabled: bool,
    native_disable_latch: bool,
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
    wink_arm: u8,
    wink_active: bool,
    blink_pulse_active: bool,
    blink_pulse_bottom_frames: u8,
    blink_pulse_age: u16,
    blink_pulse_consumed: bool,
    response_snap_committed: bool,

    kalman_x: f32,
    smooth_open: f32,
    kalman: ScalarKalman,
    gaze_smooth: [f32; 3],
    pupil_smooth: f32,
}

fn wide_entry_offset_from_neutral(samples: &[f32; WIDE_NEUTRAL_SAMPLE_CAP], n: usize) -> f32 {
    let n = n.min(WIDE_NEUTRAL_SAMPLE_CAP);
    if n < WIDE_NEUTRAL_MIN_SAMPLES {
        return UPPER_OFFSET;
    }

    let mut ordered = *samples;
    ordered[..n].sort_by(f32::total_cmp);
    let median = if n % 2 == 0 {
        (ordered[n / 2 - 1] + ordered[n / 2]) * 0.5
    } else {
        ordered[n / 2]
    };
    let p10 = ordered[(n - 1) * 10 / 100];
    let p90 = ordered[(n - 1) * 90 / 100];

    let mut deviations = [0.0; WIDE_NEUTRAL_SAMPLE_CAP];
    for (dst, value) in deviations[..n].iter_mut().zip(ordered[..n].iter()) {
        *dst = (*value - median).abs();
    }
    deviations[..n].sort_by(f32::total_cmp);
    let mad = if n % 2 == 0 {
        (deviations[n / 2 - 1] + deviations[n / 2]) * 0.5
    } else {
        deviations[n / 2]
    };

    let sigma = (1.4826 * mad).max((p90 - p10).max(0.0) / 2.563);
    (WIDE_NEUTRAL_SIGMA_MULT * sigma).clamp(WIDE_ENTRY_MIN_OFFSET, WIDE_ENTRY_MAX_OFFSET)
}

fn record_wide_neutral_sample(s: &mut EyeState, raw: f32) {
    let n = usize::from(s.wide_neutral_n).min(WIDE_NEUTRAL_SAMPLE_CAP);
    if n < WIDE_NEUTRAL_SAMPLE_CAP {
        s.wide_neutral_samples[n] = raw;
        s.wide_neutral_n = (n + 1) as u8;
    }
}

impl Default for EyeState {
    fn default() -> Self {
        Self {
            baseline: 0.60,
            baseline_n: 0,
            recenter_active: false,
            upper: 0.62,
            wide_entry_offset: UPPER_OFFSET,
            wide_neutral_samples: [0.0; WIDE_NEUTRAL_SAMPLE_CAP],
            wide_neutral_n: 0,
            wide_ceiling: 0.0,
            squeeze_floor: NATIVE_SQ_FLOOR,
            closed_ref: 0.40,
            blink_depth: BLINK_OFFSET,
            blink_len: 0,
            blink_smooth: 0.40,
            blink_lo: 0.40,
            blink_dirty: false,
            reach_env: 0.60,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
            blink_candidate: None,
            blink_candidate_count: 0,
            blink_candidate_max: 0.0,
            blink_candidate_frame: 0,
            floor_last_commit_frame: 0,
            floor_travel_deeper: 0.0,
            floor_travel_shallower: 0.0,
            mid_anchor: 0.5,
            mid_anchor_staged: None,
            last_ramp_pre: -1.0,
            ep_entry_frame: 0,
            ep_clean_exit: None,
            frame_count: 0,
            open_streak: 0,
            closed_streak: 0,
            squeeze_streak: 0,
            native_squeeze_closed_frames: 0,
            native_squeeze_peak: 0.0,
            native_squeeze_envelope: 0.0,
            is_closed: false,
            is_wide: false,
            wide_frames: 0,
            fall_run: 0.0,
            fall_updates: 0,
            since_down: u32::MAX,
            prev_raw: 0.60,
            native_was_enabled: false,
            native_disable_latch: false,
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
            wink_arm: 0,
            wink_active: false,
            blink_pulse_active: false,
            blink_pulse_bottom_frames: 0,
            blink_pulse_age: 0,
            blink_pulse_consumed: false,
            response_snap_committed: false,
            kalman_x: 0.0,
            smooth_open: 1.0,
            kalman: ScalarKalman::default(),
            gaze_smooth: INITIAL_GAZE,
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
    #[serde(default)]
    pub learned_once: bool,
    #[serde(default)]
    pub endpoint_locked: bool,
    #[serde(default)]
    pub endpoint_calibrated_unix: u64,
}

impl CalibSnapshot {
    pub fn without_explicit_endpoint(self) -> Self {
        Self {
            baseline: DEFAULT_BASELINE,
            baseline_n: 0,
            blink_depth: BLINK_OFFSET,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
            ..self
        }
    }
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

static CALIB_SAVE_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

pub fn save_calib_checked(path: &str, store: &CalibStore) -> std::io::Result<()> {
    let lock = CALIB_SAVE_LOCK.get_or_init(|| std::sync::Mutex::new(()));
    let _guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let text = toml::to_string(store).map_err(|error| std::io::Error::other(error.to_string()))?;
    let path = std::path::Path::new(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let suffix = format!("{}-{nonce}", std::process::id());

    let temp = path.with_extension(format!("tmp-{suffix}"));
    std::fs::write(&temp, text)?;
    match std::fs::rename(&temp, path) {
        Ok(()) => Ok(()),
        Err(error) => {
            let _ = std::fs::remove_file(&temp);
            Err(error)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ClosureReason {
    #[default]
    Normal,
    BlinkPending,
    FastBlink,
    NativeDisable,
    Wink,
    TrackingLost,
}

#[derive(Clone, Copy, Debug)]
pub struct EyelidLiveDiag {
    pub raw_openness: f32,
    pub model_response: f32,
    pub final_openness: f32,
    pub calibrated_baseline: f32,
    pub effective_baseline: f32,
    pub session_baseline_offset: f32,
    pub effective_open_ref: f32,
    pub effective_closed_ref: f32,
    pub wide_entry_ref: f32,
    pub wide_full_ref: f32,
    pub reason: ClosureReason,
}

impl Default for EyelidLiveDiag {
    fn default() -> Self {
        Self {
            raw_openness: 0.0,
            model_response: 1.0,
            final_openness: 1.0,
            calibrated_baseline: DEFAULT_BASELINE,
            effective_baseline: DEFAULT_BASELINE,
            session_baseline_offset: 0.0,
            effective_open_ref: DEFAULT_BASELINE - SQUEEZE_TOP_OFFSET,
            effective_closed_ref: DEFAULT_BASELINE - BLINK_OFFSET,
            wide_entry_ref: DEFAULT_BASELINE + UPPER_OFFSET,
            wide_full_ref: DEFAULT_BASELINE + UPPER_OFFSET + WIDE_CEIL_MIN_SPAN,
            reason: ClosureReason::Normal,
        }
    }
}

pub fn save_calib(path: &str, store: &CalibStore) {
    let _ = save_calib_checked(path, store);
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

#[derive(Clone)]
pub struct SRanipalState {
    eyes: [EyeState; 2],
    pub tuning: Tuning,
    gaze_smoothing_enabled: bool,
    wide_gate: f32,
    native_wide_response: bool,
    response_manual_range: bool,
    response_open_point_offset: [f32; 2],
    response_closed_point_depth: [f32; 2],
    response_close_depth_scale: [f32; 2],
    response_curve_mid_output: [f32; 2],
    response_blink_close_ms: f32,
    response_snap_gate_open: f32,
    response_auto_reseat: bool,
    response_profile_installed: bool,
    session_baseline_offset: [f32; 2],
    appearance_profile_active: bool,
    appearance_baseline_offset: [f32; 2],
    appearance_wide_entry_shift: [f32; 2],
    reseat_candidate: [f32; 2],
    reseat_prev_raw: [f32; 2],
    reseat_stable_frames: u16,
    eyelid_live: [EyelidLiveDiag; 2],
    pend: PairPending,
    gaze_eyelid_lift: [f32; 2],
}

impl Default for SRanipalState {
    fn default() -> Self {
        Self::new()
    }
}

impl SRanipalState {
    pub fn process_frame_with_model_evidence(
        &mut self, ml: [[f32; 5]; 2], gaze: &GazeSample, ml_loaded: bool,
        profile: &GazeEyelidProfile, wink: &WinkProfile, timing: &BlinkTimingProfile,
    ) -> [EyeResult; 2] {
        let missing = [ml[0][0] <= CH0_PRESENT_GATE, ml[1][0] <= CH0_PRESENT_GATE];
        if !missing.iter().any(|v| *v) {
            return self.process_frame_with_all_profiles(ml, gaze, ml_loaded, profile, wink, timing);
        }
        let saved = self.clone();
        let output = self.process_frame_with_all_profiles(ml, gaze, ml_loaded, profile, wink, timing);
        for i in 0..2 {
            if missing[i] {
                let current = self.eyes[i];
                self.eyes[i] = saved.eyes[i];
                self.eyes[i].gaze_smooth = current.gaze_smooth;
                self.eyes[i].pupil_smooth = current.pupil_smooth;
                self.eyes[i].yoke_hold = current.yoke_hold;
                self.eyelid_live[i] = saved.eyelid_live[i];
                self.eyelid_live[i].reason = ClosureReason::TrackingLost;
                self.gaze_eyelid_lift[i] = saved.gaze_eyelid_lift[i];
            }
        }
        self.pend = PairPending::default();
        self.wide_gate = saved.wide_gate;
        self.session_baseline_offset = saved.session_baseline_offset;
        self.reseat_candidate = saved.reseat_candidate;
        self.reseat_prev_raw = saved.reseat_prev_raw;
        self.reseat_stable_frames = saved.reseat_stable_frames;
        output
    }

    pub fn new() -> Self {
        Self {
            eyes: [EyeState::default(); 2],
            tuning: Tuning::default(),
            gaze_smoothing_enabled: true,
            wide_gate: 0.0,
            native_wide_response: false,
            response_manual_range: false,
            response_open_point_offset: [0.03; 2],
            response_closed_point_depth: [0.40; 2],
            response_close_depth_scale: [1.0; 2],
            response_curve_mid_output: [0.5; 2],
            response_blink_close_ms: 0.0,
            response_snap_gate_open: DEFAULT_SNAP_GATE_OPEN,
            response_profile_installed: false,
            response_auto_reseat: false,
            session_baseline_offset: [0.0; 2],
            appearance_profile_active: false,
            appearance_baseline_offset: [0.0; 2],
            appearance_wide_entry_shift: [0.0; 2],
            reseat_candidate: [0.0; 2],
            reseat_prev_raw: [DEFAULT_BASELINE; 2],
            reseat_stable_frames: 0,
            eyelid_live: [EyelidLiveDiag::default(); 2],
            pend: PairPending::default(),
            gaze_eyelid_lift: [0.0; 2],
        }
    }

    pub fn baseline(&self, e: Eye) -> f32 {
        self.eyes[e.idx()].baseline
    }

    pub(crate) fn set_native_wide_response(&mut self, enabled: bool) {
        if self.native_wide_response == enabled { return; }
        self.native_wide_response = enabled;
        self.wide_gate = 0.0;
        for eye in &mut self.eyes {
            eye.wide_ceiling = 0.0;
            eye.kalman_x = 0.0;
        }
    }

    pub fn set_gaze_smoothing(&mut self, enabled: bool) {
        self.gaze_smoothing_enabled = enabled;
    }

    pub fn set_eyelid_response_tuning(
        &mut self,
        manual_range: bool,
        open_point_offset: [f32; 2],
        closed_point_depth: [f32; 2],
        close_depth_scale: [f32; 2],
        curve_mid_output: [f32; 2],
        blink_close_ms: f32,
        snap_gate_open: f32,
        auto_reseat: bool,
    ) {
        self.response_manual_range = manual_range;
        for i in 0..2 {
            self.response_open_point_offset[i] = if open_point_offset[i].is_finite() {
                open_point_offset[i].clamp(0.03, 0.20)
            } else {
                0.03
            };
            self.response_closed_point_depth[i] = if closed_point_depth[i].is_finite() {
                closed_point_depth[i]
                    .clamp(0.08, 0.46)
                    .max(self.response_open_point_offset[i] + MIN_CLOSE_SPAN)
            } else {
                0.40
            };
            self.response_close_depth_scale[i] = if close_depth_scale[i].is_finite() {
                close_depth_scale[i].clamp(CLOSE_DEPTH_SCALE_MIN, CLOSE_DEPTH_SCALE_MAX)
            } else {
                1.0
            };
            self.response_curve_mid_output[i] = if curve_mid_output[i].is_finite() {
                curve_mid_output[i].clamp(CURVE_MID_OUTPUT_MIN, CURVE_MID_OUTPUT_MAX)
            } else {
                0.5
            };
        }
        self.response_blink_close_ms = if blink_close_ms.is_finite() {
            blink_close_ms.clamp(0.0, BLINK_CLOSE_MS_MAX)
        } else {
            0.0
        };
        self.response_snap_gate_open = if snap_gate_open.is_finite() {
            snap_gate_open.clamp(SNAP_GATE_OPEN_MIN, SNAP_GATE_OPEN_MAX)
        } else {
            DEFAULT_SNAP_GATE_OPEN
        };
        self.response_profile_installed = true;
        self.response_auto_reseat = auto_reseat;
        if !auto_reseat {
            self.reset_session_reseat();
        }
    }

    pub fn eyelid_live_diag(&self) -> [EyelidLiveDiag; 2] {
        self.eyelid_live
    }

    pub fn update_appearance_profile(
        &mut self,
        baseline: Option<[f32; 2]>,
        wide_entry_ref: Option<[f32; 2]>,
    ) {
        self.appearance_profile_active = baseline.is_some();
        if self.appearance_profile_active {
            self.reset_session_reseat();
        }
        for i in 0..2 {
            let desired_baseline = baseline
                .map(|target| {
                    (target[i] - self.eyes[i].baseline)
                        .clamp(-APPEARANCE_OFFSET_CAP, APPEARANCE_OFFSET_CAP)
                })
                .unwrap_or(0.0);
            self.appearance_baseline_offset[i] += (desired_baseline
                - self.appearance_baseline_offset[i])
                .clamp(-APPEARANCE_BASELINE_STEP, APPEARANCE_BASELINE_STEP);
            if self.appearance_baseline_offset[i].abs() < 1e-6 {
                self.appearance_baseline_offset[i] = 0.0;
            }

            let natural_wide = self.eyes[i].baseline
                + self.appearance_baseline_offset[i]
                + self.eyes[i].wide_entry_offset;
            let desired_wide = wide_entry_ref
                .map(|target| {
                    (target[i] - natural_wide).clamp(-APPEARANCE_OFFSET_CAP, APPEARANCE_OFFSET_CAP)
                })
                .unwrap_or(0.0);
            self.appearance_wide_entry_shift[i] += (desired_wide
                - self.appearance_wide_entry_shift[i])
                .clamp(-APPEARANCE_WIDE_STEP, APPEARANCE_WIDE_STEP);
            if self.appearance_wide_entry_shift[i].abs() < 1e-6 {
                self.appearance_wide_entry_shift[i] = 0.0;
            }
        }
    }

    pub fn clear_appearance_profile(&mut self) {
        self.appearance_profile_active = false;
        self.appearance_baseline_offset = [0.0; 2];
        self.appearance_wide_entry_shift = [0.0; 2];
    }

    pub fn adopt_appearance_baseline(&mut self, baseline: [f32; 2]) {
        for i in 0..2 {
            if baseline[i].is_finite() {
                let delta = baseline[i] - self.eyes[i].baseline;
                self.eyes[i].baseline = baseline[i];
                self.eyes[i].closed_ref += delta;
                self.eyes[i].wide_entry_offset += self.appearance_wide_entry_shift[i];
            }
        }
        self.clear_appearance_profile();
        self.reset_session_reseat();
    }

    pub fn adopt_current_appearance(&mut self) {
        self.adopt_appearance_baseline([self.effective_baseline(0), self.effective_baseline(1)]);
    }

    fn effective_baseline(&self, i: usize) -> f32 {
        self.eyes[i].baseline + self.appearance_baseline_offset[i] + self.session_baseline_offset[i]
    }

    fn uses_continuous_response(&self) -> bool {
        self.tuning.continuous_calib
            || (self.response_profile_installed && self.response_manual_range)
    }

    fn effective_open_ref(&self, i: usize) -> f32 {
        let offset = if self.response_profile_installed && self.response_manual_range {
            self.response_open_point_offset[i]
        } else {
            self.tuning.open_deadzone.clamp(SQUEEZE_TOP_OFFSET, 0.20)
        };
        self.effective_baseline(i) - offset
    }

    fn effective_closed_ref(&self, i: usize) -> f32 {
        let baseline = self.eyes[i].baseline + self.appearance_baseline_offset[i];
        if self.response_profile_installed && self.response_manual_range {
            return baseline - self.response_closed_point_depth[i];
        }
        let learned_depth = if self.tuning.continuous_calib {
            (self.eyes[i].baseline - self.eyes[i].closed_ref).clamp(0.05, 0.40)
        } else {
            BLINK_OFFSET
        };
        let depth = learned_depth * self.response_close_depth_scale[i];
        baseline - depth.clamp(0.05, 0.46)
    }

    fn model_response_for_raw(&self, i: usize, raw: f32, native_absolute: bool) -> f32 {
        if !raw.is_finite() {
            return 0.0;
        }
        let open_full = self.effective_open_ref(i);
        let closed_ref = self.effective_closed_ref(i).min(open_full - 1e-3);
        let denom = (open_full - closed_ref).max(MIN_CLOSE_SPAN);
        let pre = ((raw - closed_ref) / denom).clamp(0.0, 1.0);
        let calibrated = if native_absolute {
            pre
        } else {
            apply_anchor(pre, self.eyes[i].mid_anchor)
        };
        apply_response_curve(calibrated, self.response_curve_mid_output[i])
    }

    fn resolve_snap_gate(&mut self, samples: &mut [PerEye; 2]) {
        for (i, sample) in samples.iter_mut().enumerate() {
            if !sample.present {
                sample.openness_target = 0.0;
                sample.closed = true;
                sample.reason = ClosureReason::TrackingLost;
                self.eyes[i].response_snap_committed = false;
                continue;
            }

            if self.eyes[i].blink_pulse_active {
                sample.openness_target = 0.0;
                sample.closed = true;
                sample.reason = ClosureReason::FastBlink;
                self.eyes[i].response_snap_committed = true;
                continue;
            }

            if sample.snap_requested {
                if self.eyes[i].response_snap_committed
                    || sample.model_response <= self.response_snap_gate_open
                {
                    self.eyes[i].response_snap_committed = true;
                    sample.openness_target = 0.0;
                    sample.closed = true;
                    if sample.reason != ClosureReason::Wink {
                        sample.reason = sample.snap_reason;
                    }
                } else {
                    self.eyes[i].response_snap_committed = false;
                    sample.openness_target = sample.model_response;
                    sample.closed = sample.model_response < 0.08;
                    if sample.reason != ClosureReason::Wink {
                        sample.reason = ClosureReason::BlinkPending;
                    }
                }
            } else {
                self.eyes[i].response_snap_committed = false;
                if sample.reason != ClosureReason::Wink {
                    sample.reason = ClosureReason::Normal;
                }
            }
        }
    }

    fn reset_session_reseat(&mut self) {
        self.session_baseline_offset = [0.0; 2];
        self.reseat_candidate = [0.0; 2];
        self.reseat_prev_raw = [self.eyes[0].baseline, self.eyes[1].baseline];
        self.reseat_stable_frames = 0;
    }

    fn gaze_is_near_center(gaze: [f32; 3], valid: bool) -> bool {
        if !valid || !gaze.iter().all(|value| value.is_finite()) {
            return false;
        }
        let transverse = (gaze[0] * gaze[0] + gaze[1] * gaze[1]).sqrt();
        let forward = gaze[2].abs();
        forward > 1e-4 && transverse.atan2(forward).to_degrees() <= AUTO_RESEAT_CENTER_DEG
    }

    fn update_auto_reseat(&mut self, raw: [f32; 2], ml: &[[f32; 5]; 2], gaze: &GazeSample) {
        if self.appearance_profile_active {
            self.reset_session_reseat();
            return;
        }
        if !self.response_auto_reseat {
            self.reset_session_reseat();
            return;
        }

        let mut eligible = true;
        for e in Eye::ALL {
            let i = e.idx();
            let native = gaze.eye(e);
            let state = &self.eyes[i];
            let squeeze_rise = (ml[i][3] - state.squeeze_floor).max(0.0);
            let fixed_range_ready = (self.response_profile_installed && self.response_manual_range)
                || (state.learned_once && state.endpoint_locked);
            eligible &= state.baseline_n >= BASELINE_BOOTSTRAP_N
                && fixed_range_ready
                && ml[i][0] > CH0_PRESENT_GATE
                && raw[i].is_finite()
                && raw[i] > BASELINE_RANGE_LO - 0.12
                && raw[i] < BASELINE_RANGE_HI + 0.12
                && native.openness_reported
                && native.openness_valid
                && native.openness.is_finite()
                && native.openness >= AUTO_RESEAT_NATIVE_OPEN_MIN
                && Self::gaze_is_near_center(native.gaze, native.gaze_valid)
                && squeeze_rise <= AUTO_RESEAT_SQUEEZE_MAX
                && !state.wink_active
                && !state.blink_pulse_active;
        }

        let deltas = [
            raw[0] - self.eyes[0].baseline,
            raw[1] - self.eyes[1].baseline,
        ];
        let depth = [
            self.eyes[0].blink_depth.max(0.05),
            self.eyes[1].blink_depth.max(0.05),
        ];
        let coherent =
            ((deltas[0] / depth[0]) - (deltas[1] / depth[1])).abs() <= AUTO_RESEAT_PAIR_COHERENCE;
        let both_near_calibrated = deltas
            .iter()
            .all(|delta| delta.abs() <= AUTO_RESEAT_CANDIDATE_BAND);
        let both_shifted_same_direction = deltas
            .iter()
            .all(|delta| delta.abs() > AUTO_RESEAT_CANDIDATE_BAND)
            && deltas[0].signum() == deltas[1].signum();
        let same_direction = both_near_calibrated || both_shifted_same_direction;
        let low_motion = (raw[0] - self.reseat_prev_raw[0]).abs() <= AUTO_RESEAT_CANDIDATE_BAND
            && (raw[1] - self.reseat_prev_raw[1]).abs() <= AUTO_RESEAT_CANDIDATE_BAND;
        self.reseat_prev_raw = raw;

        if !eligible || !coherent || !same_direction || !low_motion {
            self.reseat_candidate = deltas;
            self.reseat_stable_frames = 0;
            return;
        }

        let candidate_matches = self.reseat_stable_frames == 0
            || (0..2).all(|i| {
                (deltas[i] - self.reseat_candidate[i]).abs() <= AUTO_RESEAT_CANDIDATE_BAND
            });
        if !candidate_matches {
            self.reseat_candidate = deltas;
            self.reseat_stable_frames = 1;
            return;
        }

        if self.reseat_stable_frames == 0 {
            self.reseat_candidate = deltas;
        } else {
            for i in 0..2 {
                self.reseat_candidate[i] += 0.05 * (deltas[i] - self.reseat_candidate[i]);
            }
        }
        self.reseat_stable_frames = self.reseat_stable_frames.saturating_add(1);
        if self.reseat_stable_frames < AUTO_RESEAT_STABLE_FRAMES {
            return;
        }

        for i in 0..2 {
            let cap = AUTO_RESEAT_OFFSET_CAP
                .min(AUTO_RESEAT_DEPTH_CAP_FRAC * self.eyes[i].blink_depth.max(0.05));
            let mut target = self.reseat_candidate[i].clamp(-cap, cap);
            if target.abs() <= AUTO_RESEAT_CANDIDATE_BAND {
                target = 0.0;
            }
            let delta = (target - self.session_baseline_offset[i])
                .clamp(-AUTO_RESEAT_OFFSET_STEP, AUTO_RESEAT_OFFSET_STEP);
            self.session_baseline_offset[i] += delta;
            if self.session_baseline_offset[i].abs() < 1e-5 {
                self.session_baseline_offset[i] = 0.0;
            }
        }
    }

    pub fn snapshot(&self, e: Eye) -> CalibSnapshot {
        let s = &self.eyes[e.idx()];
        CalibSnapshot {
            baseline: s.baseline,
            baseline_n: s.baseline_n,
            frame_count: s.frame_count,
            blink_depth: s.blink_depth,
            mid_anchor: s.mid_anchor,
            learned_once: s.learned_once,
            endpoint_locked: s.endpoint_locked,
            endpoint_calibrated_unix: s.endpoint_calibrated_unix,
        }
    }

    fn arm_eye_recenter(s: &mut EyeState) {
        s.baseline_n = RECENTER_PRIOR_N;
        s.recenter_active = true;
        s.wide_neutral_n = 0;
        s.wide_frames = if s.is_wide { WIDE_STUCK_FRAMES } else { 0 };
        s.fall_run = 0.0;
        s.fall_updates = 0;
        s.is_closed = false;
        s.is_wide = false;
        s.fast_blink_frames = 0;
        s.fast_blink_arm = 0;
        s.arm_updates = 0;
        s.wink_arm = 0;
        s.wink_active = false;
        s.native_squeeze_closed_frames = 0;
        s.native_squeeze_peak = 0.0;
        s.native_squeeze_envelope = 0.0;
        s.blink_pulse_active = false;
        s.blink_pulse_bottom_frames = 0;
        s.blink_pulse_age = 0;
        s.blink_pulse_consumed = false;
        s.response_snap_committed = false;
        s.kalman_x = 0.0;
        s.kalman.reset();
        s.wide_ceiling = 0.0;
        s.squeeze_floor = NATIVE_SQ_FLOOR;
        s.blink_len = 0;
        s.mid_anchor_staged = None;
        s.ep_clean_exit = None;
        s.last_ramp_pre = -1.0;
        s.reach_env = s.baseline;
        s.blink_candidate = None;
        s.blink_candidate_count = 0;
        s.blink_candidate_max = 0.0;
        s.blink_candidate_frame = 0;
        s.floor_last_commit_frame = s.frame_count;
        s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
    }

    pub fn recenter(&mut self) {
        self.wide_gate = 0.0;
        self.gaze_eyelid_lift = [0.0; 2];
        self.reset_session_reseat();
        self.clear_appearance_profile();
        for s in self.eyes.iter_mut() {
            Self::arm_eye_recenter(s);
        }
        self.pend = PairPending::default();
    }

    pub fn reset_eyelid_calibration(&mut self) {
        let mut fresh = Self::new();
        fresh.tuning = self.tuning;
        fresh.gaze_smoothing_enabled = self.gaze_smoothing_enabled;
        fresh.native_wide_response = self.native_wide_response;
        for (reset, previous) in fresh.eyes.iter_mut().zip(self.eyes.iter()) {
            reset.gaze_smooth = previous.gaze_smooth;
            reset.pupil_smooth = previous.pupil_smooth;
            reset.yoke_hold = previous.yoke_hold;
        }
        *self = fresh;
    }

    pub fn recenter_eye(&mut self, eye: Eye) {
        let i = eye.idx();
        self.wide_gate = 0.0;
        self.gaze_eyelid_lift[i] = 0.0;
        self.session_baseline_offset[i] = 0.0;
        self.appearance_baseline_offset[i] = 0.0;
        self.appearance_wide_entry_shift[i] = 0.0;
        self.reseat_candidate = [0.0; 2];
        self.reseat_prev_raw = [self.eyes[0].baseline, self.eyes[1].baseline];
        self.reseat_stable_frames = 0;
        Self::arm_eye_recenter(&mut self.eyes[i]);
        self.pend = PairPending::default();
    }

    pub fn restore(&mut self, e: Eye, snap: CalibSnapshot) {
        let i = e.idx();
        self.session_baseline_offset[i] = 0.0;
        self.appearance_baseline_offset[i] = 0.0;
        self.appearance_wide_entry_shift[i] = 0.0;
        self.reseat_stable_frames = 0;
        self.eyes[i].response_snap_committed = false;
        let s = &mut self.eyes[e.idx()];
        s.baseline = snap.baseline;
        s.recenter_active = false;
        s.wide_entry_offset = UPPER_OFFSET;
        s.wide_neutral_n = 0;
        s.baseline_n = snap.baseline_n.min(RESTORE_PRIOR_N);
        s.frame_count = snap.frame_count;
        s.blink_depth = if snap.blink_depth.is_finite() {
            snap.blink_depth.clamp(0.05, 0.40)
        } else {
            BLINK_OFFSET
        };
        s.learned_once = snap.learned_once
            || (snap.blink_depth.is_finite()
                && (snap.blink_depth - default_blink_depth()).abs() > 1e-4);
        s.endpoint_locked = snap.endpoint_locked;
        s.endpoint_calibrated_unix = snap.endpoint_calibrated_unix;
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
        s.wink_arm = 0;
        s.wink_active = false;
        s.blink_pulse_active = false;
        s.blink_pulse_bottom_frames = 0;
        s.blink_pulse_age = 0;
        s.blink_pulse_consumed = false;
        s.reach_env = s.baseline;
        s.blink_candidate = None;
        s.blink_candidate_count = 0;
        s.blink_candidate_max = 0.0;
        s.blink_candidate_frame = 0;
        s.floor_last_commit_frame = 0;
        s.floor_travel_deeper = 0.0;
        s.floor_travel_shallower = 0.0;
        s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
    }

    pub fn snapshot_all(&self) -> CalibStore {
        CalibStore {
            left: self.snapshot(Eye::Left),
            right: self.snapshot(Eye::Right),
        }
    }

    pub fn restore_all(&mut self, store: &CalibStore) {
        self.restore(Eye::Left, store.left);
        self.restore(Eye::Right, store.right);
        let fc = self.eyes[0].frame_count.max(self.eyes[1].frame_count);
        self.eyes[0].frame_count = fc;
        self.eyes[1].frame_count = fc;
    }

    pub fn apply_explicit_endpoint(
        &mut self,
        eye: Eye,
        baseline: f32,
        blink_depth: f32,
        calibrated_unix: u64,
    ) {
        if !baseline.is_finite() || !blink_depth.is_finite() {
            return;
        }
        let i = eye.idx();
        self.session_baseline_offset[i] = 0.0;
        self.appearance_baseline_offset[i] = 0.0;
        self.appearance_wide_entry_shift[i] = 0.0;
        self.reseat_stable_frames = 0;
        self.eyes[i].response_snap_committed = false;
        let s = &mut self.eyes[eye.idx()];
        s.baseline = baseline.clamp(BASELINE_RANGE_LO, BASELINE_RANGE_HI);
        s.baseline_n = s.baseline_n.max(BASELINE_BOOTSTRAP_N);
        s.blink_depth = blink_depth.clamp(0.05, 0.40);
        s.learned_once = true;
        s.endpoint_locked = true;
        s.endpoint_calibrated_unix = calibrated_unix;
        s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
    }

    pub fn remove_explicit_endpoint(&mut self, eye: Eye) {
        let reset = self.snapshot(eye).without_explicit_endpoint();
        self.restore(eye, reset);
    }

    fn update_calibration(
        &mut self,
        i: usize,
        raw: f32,
        present: bool,
        exclude_close_learning: bool,
    ) {
        let endpoint_learning_enabled = !self.tuning.couple_eyes && !self.eyes[i].endpoint_locked;
        let s = &mut self.eyes[i];
        s.frame_count += 1;
        if s.is_wide {
            s.wide_frames = s.wide_frames.saturating_add(1);
        } else if present && raw > s.baseline - BLINK_RELEASE_OFFSET && raw < BASELINE_RANGE_HI {
            s.wide_frames = s.wide_frames.saturating_sub(WIDE_DISCHARGE);
        }
        if present && raw > BASELINE_RANGE_LO && raw < BASELINE_RANGE_HI {
            let stuck = s.is_wide && s.wide_frames >= WIDE_STUCK_FRAMES;
            if s.baseline_n < BASELINE_BOOTSTRAP_N {
                if s.recenter_active
                    || s.baseline_n < BOOTSTRAP_ANCHOR_N
                    || (raw - s.baseline).abs() < WIDE_LEARN_CAP
                {
                    record_wide_neutral_sample(s, raw);
                    s.baseline_n += 1;
                    s.baseline += (raw - s.baseline) / s.baseline_n as f32;
                    if s.baseline_n >= BASELINE_BOOTSTRAP_N {
                        s.recenter_active = false;
                        s.wide_entry_offset = wide_entry_offset_from_neutral(
                            &s.wide_neutral_samples,
                            usize::from(s.wide_neutral_n),
                        );
                        s.wide_ceiling = 0.0;
                    }
                } else {
                    let rate = if stuck {
                        BASELINE_ADAPT
                    } else {
                        BASELINE_ADAPT * BASELINE_SLOW_FACTOR
                    };
                    s.baseline += rate * (raw - s.baseline);
                }
            } else {
                if stuck {
                    s.baseline += BASELINE_ADAPT * (raw - s.baseline);
                }
                s.baseline_n = s.baseline_n.saturating_add(1);
            }
        }
        let b = s.baseline;
        s.upper = b + s.wide_entry_offset;
        if present && !exclude_close_learning && raw > b - ENV_SANITY_OFFSET {
            let rate = if raw < s.reach_env {
                ENV_ATTACK
            } else {
                ENV_HEAL
            };
            s.reach_env += rate * (raw - s.reach_env);
        }
        if !present
            || (s.blink_candidate.is_some()
                && s.frame_count.saturating_sub(s.blink_candidate_frame) > FLOOR_CANDIDATE_MAX_AGE)
        {
            s.blink_candidate = None;
            s.blink_candidate_count = 0;
            s.blink_candidate_max = 0.0;
        }
        let entry = b - (GATE_FRAC * (b - s.reach_env)).clamp(GATE_MIN, BLINK_OFFSET);
        let release = b - BLINK_RELEASE_OFFSET;
        if s.blink_len == 0 {
            if present
                && !exclude_close_learning
                && raw < entry
                && s.baseline_n >= BASELINE_BOOTSTRAP_N
            {
                s.blink_len = 1;
                s.blink_smooth = raw;
                s.blink_lo = raw;
                s.blink_dirty = s.fast_blink_frames > 0 || s.fast_blink_arm > 0;
                s.ep_entry_frame = s.frame_count;
                s.ep_clean_exit = None;
            }
        } else {
            s.blink_len = s.blink_len.saturating_add(1);
            if exclude_close_learning || s.fast_blink_frames > 0 || s.fast_blink_arm > 0 || !present
            {
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
                    && s.blink_lo <= s.reach_env + GENUINE_NEAR
                {
                    let bottom = s.blink_lo.clamp(0.0, 1.0);
                    if !endpoint_learning_enabled {
                        s.blink_candidate = None;
                        s.blink_candidate_count = 0;
                        s.blink_candidate_max = 0.0;
                        if s.learned_once && (bottom - s.closed_ref).abs() <= FLOOR_CURVE_NEAR {
                            s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                        }
                    } else if !s.learned_once {
                        if let Some(candidate) = s.blink_candidate {
                            if (bottom - candidate).abs() <= FIRST_CLOSE_CONFIRM_TOL {
                                s.closed_ref = (0.5 * (candidate + bottom)).clamp(0.0, 1.0);
                                s.blink_depth = (b - s.closed_ref).clamp(0.05, 0.40);
                                s.learned_once = true;
                                s.blink_candidate = None;
                                s.blink_candidate_count = 0;
                                s.blink_candidate_max = 0.0;
                                s.floor_last_commit_frame = s.frame_count;
                                s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                            } else {
                                s.blink_candidate = Some(bottom);
                                s.blink_candidate_count = 1;
                                s.blink_candidate_max = bottom;
                                s.blink_candidate_frame = s.frame_count;
                            }
                        } else {
                            s.blink_candidate = Some(bottom);
                            s.blink_candidate_count = 1;
                            s.blink_candidate_max = bottom;
                            s.blink_candidate_frame = s.frame_count;
                        }
                    } else {
                        let delta = bottom - s.closed_ref;
                        let near_floor = delta.abs() <= FLOOR_CURVE_NEAR;
                        let deeper = delta < -FLOOR_CURVE_NEAR;
                        let shallow_escape = delta >= FLOOR_SHALLOW_ESCAPE;

                        if near_floor {
                            s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                            s.blink_candidate = None;
                            s.blink_candidate_count = 0;
                            s.blink_candidate_max = 0.0;
                        } else if deeper || shallow_escape {
                            let (need, tolerance) = if deeper {
                                (FLOOR_DEEP_CONFIRM_N, FLOOR_DEEP_CONFIRM_TOL)
                            } else {
                                (FLOOR_SHALLOW_CONFIRM_N, FLOOR_SHALLOW_CONFIRM_TOL)
                            };
                            let matches = s.blink_candidate.is_some_and(|candidate_min| {
                                s.blink_candidate_max.max(bottom) - candidate_min.min(bottom)
                                    <= tolerance
                            });
                            if matches {
                                s.blink_candidate =
                                    Some(s.blink_candidate.unwrap_or(bottom).min(bottom));
                                s.blink_candidate_max = s.blink_candidate_max.max(bottom);
                                s.blink_candidate_count = s.blink_candidate_count.saturating_add(1);
                            } else {
                                s.blink_candidate = Some(bottom);
                                s.blink_candidate_count = 1;
                                s.blink_candidate_max = bottom;
                                s.blink_candidate_frame = s.frame_count;
                            }

                            let gap_ok = s.floor_last_commit_frame == 0
                                || s.frame_count.saturating_sub(s.floor_last_commit_frame)
                                    >= FLOOR_COMMIT_MIN_FRAMES;
                            if s.blink_candidate_count >= need && gap_ok {
                                let candidate_min = s.blink_candidate.unwrap_or(bottom);
                                let target = 0.5 * (candidate_min + s.blink_candidate_max);
                                let requested = target - s.closed_ref;
                                let applied = if requested < 0.0 {
                                    let remaining =
                                        (FLOOR_TRAVEL_BUDGET - s.floor_travel_deeper).max(0.0);
                                    -((-requested).min(FLOOR_DEEP_COMMIT_CAP).min(remaining))
                                } else {
                                    let remaining =
                                        (FLOOR_TRAVEL_BUDGET - s.floor_travel_shallower).max(0.0);
                                    requested.min(FLOOR_SHALLOW_COMMIT_CAP).min(remaining)
                                };
                                if applied.abs() > f32::EPSILON {
                                    let old = s.closed_ref;
                                    s.closed_ref = (s.closed_ref + applied).clamp(0.0, 1.0);
                                    let actual = s.closed_ref - old;
                                    if actual < 0.0 {
                                        s.floor_travel_deeper += -actual;
                                    } else {
                                        s.floor_travel_shallower += actual;
                                    }
                                    s.blink_depth = (b - s.closed_ref).clamp(0.05, 0.40);
                                    s.floor_last_commit_frame = s.frame_count;
                                    s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                                }
                                s.blink_candidate = None;
                                s.blink_candidate_count = 0;
                                s.blink_candidate_max = 0.0;
                            }
                        } else {
                            s.blink_candidate = None;
                            s.blink_candidate_count = 0;
                            s.blink_candidate_max = 0.0;
                        }
                    }
                }
                s.blink_len = 0;
            }
        }
        if !s.learned_once || s.baseline_n <= BASELINE_BOOTSTRAP_N || s.endpoint_locked {
            s.closed_ref = (b - s.blink_depth).clamp(0.0, 1.0);
        } else if !s.closed_ref.is_finite() {
            s.closed_ref = (b - s.blink_depth).clamp(0.0, 1.0);
        } else {
            s.closed_ref = s.closed_ref.clamp(0.0, 1.0);
        }
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
            staged: [
                e[0].mid_anchor_staged.is_some(),
                e[1].mid_anchor_staged.is_some(),
            ],
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

    fn normalize(
        &mut self,
        i: usize,
        raw: f32,
        in2: f32,
        native_absolute: bool,
    ) -> (f32, f32, f32) {
        let tuning = self.tuning;
        let session_offset = self.session_baseline_offset[i];
        let manual_range = self.response_profile_installed && self.response_manual_range;
        let continuous_response = self.uses_continuous_response();
        let manual_open_offset = self.response_open_point_offset[i];
        let manual_closed_depth = self.response_closed_point_depth[i];
        let close_scale = self.response_close_depth_scale[i];
        let response_mid = self.response_curve_mid_output[i];
        let s = &mut self.eyes[i];
        s.last_ramp_pre = -1.0;
        let close_b = s.baseline + self.appearance_baseline_offset[i];
        let open_b = s.baseline + session_offset;
        let open_b = open_b + self.appearance_baseline_offset[i];
        let upper = open_b + s.wide_entry_offset + self.appearance_wide_entry_shift[i];
        let squeeze_top = close_b - SQUEEZE_TOP_OFFSET;
        let open_full = if manual_range {
            open_b - manual_open_offset
        } else {
            open_b - tuning.open_deadzone.clamp(SQUEEZE_TOP_OFFSET, 0.20)
        };
        let blink_top = if manual_range {
            (close_b - manual_closed_depth).min(open_full - 1e-3)
        } else if tuning.continuous_calib {
            let learned_depth = (s.baseline - s.closed_ref).clamp(0.05, 0.40);
            (close_b - learned_depth * close_scale).min(open_full - 1e-3)
        } else {
            close_b - BLINK_OFFSET * close_scale
        };
        let blink_release = close_b - BLINK_RELEASE_OFFSET;

        if !continuous_response && s.is_closed {
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

        let ceil_floor = upper + if self.native_wide_response {
            NATIVE_WIDE_MIN_SPAN
        } else {
            WIDE_CEIL_MIN_SPAN
        };
        if s.wide_ceiling < ceil_floor {
            s.wide_ceiling = ceil_floor;
        } else if raw > s.wide_ceiling {
            s.wide_ceiling = raw;
        } else {
            s.wide_ceiling += WIDE_CEIL_DECAY * (ceil_floor - s.wide_ceiling);
        }
        let wide_span = (s.wide_ceiling - upper).max(1e-3);
        let wide_gain = tuning.wide_gain;

        let wide_release = upper
            - (s.wide_entry_offset * 0.5).clamp(WIDE_ENTRY_MIN_OFFSET * 0.5, WIDE_RELEASE_OFFSET);
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

        if continuous_response {
            s.is_closed = false;
            s.kalman_x *= 1.0 - KALMAN_ALPHA;
            let d = s.blink_depth;
            let is_upd = (raw - s.prev_raw).abs() > 1e-6;
            if s.detect_grace > 0 {
                s.detect_grace -= 1;
            }
            if native_absolute {
                s.fast_blink_frames = 0;
                s.fast_blink_arm = 0;
            } else if s.fast_blink_frames > 0 {
                s.latch_bottom = s.latch_bottom.min(raw);
                let rise = (RELEASE_RISE_NORM * d).max(RELEASE_RISE_FLOOR);
                let exit_zone = raw > blink_top + (LATCH_EXIT_NORM * d).clamp(0.03, 0.09);
                let deep_now = (close_b - raw) >= SHALLOW_COMMIT_DIP * d;
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
                        s.latch_cap = if (close_b - raw) < SHALLOW_COMMIT_DIP * d {
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
                let deep = (close_b - raw) >= FALL2_COMMIT_DEPTH * d;
                if v1 >= vel_thr || (fall2 >= fall2_thr && deep) {
                    s.fast_blink_arm = 1;
                    s.fast_blink_arm_raw = raw;
                    s.arm_updates = 0;
                }
            }
            let denom = (open_full - blink_top).max(MIN_CLOSE_SPAN);
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
            let calibrated = if native_absolute {
                opn_pre
            } else {
                apply_anchor(opn_pre, s.mid_anchor)
            };
            let opn = apply_response_curve(calibrated, response_mid);
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
            let opn = apply_response_curve(opn, response_mid);
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
        self.process_frame_with_gaze_eyelid(ml, gaze, ml_loaded, &GazeEyelidProfile::default())
    }

    pub fn process_frame_with_gaze_eyelid(
        &mut self,
        ml: [[f32; 5]; 2],
        gaze: &GazeSample,
        ml_loaded: bool,
        profile: &GazeEyelidProfile,
    ) -> [EyeResult; 2] {
        self.process_frame_with_profiles(ml, gaze, ml_loaded, profile, &WinkProfile::default())
    }

    pub fn process_frame_with_profiles(
        &mut self,
        ml: [[f32; 5]; 2],
        gaze: &GazeSample,
        ml_loaded: bool,
        profile: &GazeEyelidProfile,
        wink_profile: &WinkProfile,
    ) -> [EyeResult; 2] {
        self.process_frame_with_all_profiles(
            ml,
            gaze,
            ml_loaded,
            profile,
            wink_profile,
            &BlinkTimingProfile::default(),
        )
    }

    pub fn process_frame_with_all_profiles(
        &mut self,
        ml: [[f32; 5]; 2],
        gaze: &GazeSample,
        ml_loaded: bool,
        profile: &GazeEyelidProfile,
        wink_profile: &WinkProfile,
        blink_timing: &BlinkTimingProfile,
    ) -> [EyeResult; 2] {
        let mut raws = [ml[0][1], ml[1][1]];
        self.update_auto_reseat(raws, &ml, gaze);
        for e in Eye::ALL {
            let i = e.idx();
            let native = gaze.eye(e);
            let model_ready = profile.is_compatible()
                && profile.eyes[i].enabled
                && self.eyes[i].baseline_n >= BASELINE_BOOTSTRAP_N
                && ml[i][0] > CH0_PRESENT_GATE;
            let evidence_valid = model_ready
                && native.gaze_valid
                && native.gaze.iter().all(|value| value.is_finite())
                && (!native.openness_reported || native.openness_valid);
            let target = if evidence_valid {
                profile.predicted_lift(i, native.gaze).unwrap_or(0.0)
            } else {
                0.0
            };

            if native.openness_reported && !native.openness_valid {
                self.gaze_eyelid_lift[i] = 0.0;
            } else {
                let alpha = if target > self.gaze_eyelid_lift[i] {
                    0.22
                } else {
                    0.14
                };
                self.gaze_eyelid_lift[i] += alpha * (target - self.gaze_eyelid_lift[i]);
                if self.gaze_eyelid_lift[i].abs() < 1e-5 {
                    self.gaze_eyelid_lift[i] = 0.0;
                }
            }

            let raw = raws[i];
            let baseline = self.effective_baseline(i);
            let closed_ref = self.effective_closed_ref(i).min(baseline - 0.05);
            let span = (baseline - closed_ref).max(0.05);
            let normalized = (raw - closed_ref) / span;
            let gate_x = ((normalized - 0.08) / 0.32).clamp(0.0, 1.0);
            let close_gate = gate_x * gate_x * (3.0 - 2.0 * gate_x);
            let lift = self.gaze_eyelid_lift[i] * close_gate;
            if raw < baseline && lift > 0.0 {
                raws[i] = raw + lift.min(baseline - raw);
            }
        }
        let squeeze_deltas: [f32; 2] =
            std::array::from_fn(|i| (ml[i][3] - self.eyes[i].squeeze_floor).max(0.0));
        let exclude_close_learning: [bool; 2] = std::array::from_fn(|i| {
            let j = 1 - i;
            let response = wink_profile.eyes[i];
            let squeeze_threshold =
                if wink_profile.is_compatible() && response.enabled && response.squeeze_enabled {
                    response.squeeze_release_delta
                } else {
                    WINK_LEARN_SQUEEZE_DELTA
                };
            let own_present = ml[i][0] > CH0_PRESENT_GATE;
            let partner_open =
                ml[j][0] > CH0_PRESENT_GATE && raws[j] > self.effective_baseline(j) - 0.04;
            self.eyes[i].wink_active
                || (own_present && partner_open && squeeze_deltas[i] >= squeeze_threshold)
        });
        let mut pe = [PerEye::default(); 2];

        for e in Eye::ALL {
            let i = e.idx();
            let raw = raws[i];
            let present = ml[i][0] > CH0_PRESENT_GATE;

            let transient_reseat = self.session_baseline_offset[i].abs() > 0.005;
            self.update_calibration(
                i,
                raw,
                present,
                exclude_close_learning[i] || transient_reseat,
            );

            let blink_threshold = self.eyes[i].baseline - 0.20 * self.response_close_depth_scale[i];
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
            let native_absolute = gaze.eye(e).openness_reported;
            let model_response = self.model_response_for_raw(i, raw, native_absolute);

            let (opn, wide_mag, sqz_mag) = if trusted {
                self.normalize(i, raw, raw, native_absolute)
            } else {
                self.eyes[i].is_wide = false;
                self.eyes[i].fast_blink_frames = 0;
                self.eyes[i].fast_blink_arm = 0;
                self.eyes[i].last_ramp_pre = -1.0;
                let held = if model_response <= 0.001 {
                    0.0
                } else {
                    self.eyes[i].smooth_open
                };
                (held, 0.0, 0.0)
            };

            let squeeze_out = if self.tuning.native_squeeze {
                let s3 = ml[i][3];
                if raw >= self.effective_baseline(i) {
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
                model_response,
                snap_requested: self.eyes[i].fast_blink_frames > 0,
                snap_reason: ClosureReason::FastBlink,
                reason: if present {
                    ClosureReason::Normal
                } else {
                    ClosureReason::TrackingLost
                },
                wide_out: wide_mag,
                squeeze_out,
                squeeze_delta: squeeze_deltas[i],
                gaze: es.gaze,
                gaze_v: es.gaze_valid,
                pupil_mm: es.pupil_mm,
                pupil_v: es.pupil_valid,
                closed: (if self.uses_continuous_response() {
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

        if wink_profile.is_compatible() {
            for e in Eye::ALL {
                let (i, j) = (e.idx(), e.opposite().idx());
                let response = wink_profile.eyes[i];
                if !response.enabled {
                    self.eyes[i].wink_arm = 0;
                    self.eyes[i].wink_active = false;
                    continue;
                }
                let own_native = gaze.eye(e);
                let partner_native = gaze.eye(e.opposite());
                let native_wink = own_native.openness_reported
                    && !own_native.openness_valid
                    && partner_native.openness_reported
                    && partner_native.openness_valid;
                let partner_open = pe[j].present
                    && pe[j].openness_target >= WINK_PARTNER_OPEN
                    && pe[j].draw > -0.01;
                let squeeze_enter =
                    response.squeeze_enabled && pe[i].squeeze_delta >= response.squeeze_enter_delta;
                let squeeze_held = response.squeeze_enabled
                    && pe[i].squeeze_delta >= response.squeeze_release_delta;
                let partner_response = wink_profile.eyes[j];
                let partner_squeeze_threshold = if partner_response.squeeze_enabled {
                    partner_response.squeeze_release_delta
                } else {
                    WINK_LEARN_SQUEEZE_DELTA
                };
                let unilateral_squeeze =
                    squeeze_enter && pe[j].squeeze_delta < partner_squeeze_threshold;
                let own_enter_open = if unilateral_squeeze {
                    WINK_SQUEEZE_ENTER_OPEN
                } else {
                    WINK_OWN_ENTER_OPEN
                };
                let own_narrow = pe[i].present
                    && pe[i].openness_target <= own_enter_open
                    && pe[i].raw < self.eyes[i].baseline - 0.06;

                if self.eyes[i].wink_active {
                    if !pe[i].present
                        || (!native_wink && !partner_open)
                        || (pe[i].openness_target >= WINK_RELEASE_OPEN
                            && pe[i].raw > self.effective_baseline(i) - 0.04
                            && !squeeze_held)
                    {
                        self.eyes[i].wink_active = false;
                        self.eyes[i].wink_arm = 0;
                    }
                } else if native_wink {
                    self.eyes[i].wink_active = true;
                    self.eyes[i].wink_arm = WINK_ARM_FRAMES;
                } else if own_narrow && partner_open {
                    let step = if unilateral_squeeze { 2 } else { 1 };
                    self.eyes[i].wink_arm = self.eyes[i].wink_arm.saturating_add(step);
                    if self.eyes[i].wink_arm >= WINK_ARM_FRAMES {
                        self.eyes[i].wink_active = true;
                    }
                } else {
                    self.eyes[i].wink_arm = 0;
                }

                if self.eyes[i].wink_active {
                    pe[i].reason = ClosureReason::Wink;
                }
                if self.eyes[i].wink_active && response.floor_enabled {
                    let baseline = self.eyes[i].baseline;
                    let open_full = self.effective_open_ref(i);
                    let wink_depth =
                        response.wink_depth.clamp(0.05, 0.40) * self.response_close_depth_scale[i];
                    let wink_ref = baseline - wink_depth;
                    let wink_open_pre = ((pe[i].raw - wink_ref)
                        / (open_full - wink_ref).max(MIN_CLOSE_SPAN))
                    .clamp(0.0, 1.0);
                    let wink_open =
                        apply_response_curve(wink_open_pre, self.response_curve_mid_output[i]);
                    pe[i].openness_target = pe[i].openness_target.min(wink_open);
                    pe[i].closed = pe[i].closed || wink_open < 0.08;
                }
            }
        } else {
            for eye in &mut self.eyes {
                eye.wink_arm = 0;
                eye.wink_active = false;
            }
        }

        let bilateral_squeezed_close = (0..2).all(|i| {
            let response = wink_profile.eyes[i];
            let squeeze_threshold =
                if wink_profile.is_compatible() && response.enabled && response.squeeze_enabled {
                    response.squeeze_release_delta
                } else {
                    WINK_LEARN_SQUEEZE_DELTA
                };
            pe[i].present
                && pe[i].raw < self.eyes[i].baseline - 0.06
                && pe[i].squeeze_delta >= squeeze_threshold
                && !self.eyes[i].wink_active
        });
        if bilateral_squeezed_close {
            let committed_frames = self
                .eyes
                .iter()
                .map(|eye| eye.fast_blink_frames)
                .max()
                .unwrap_or(0);
            if committed_frames > 0 {
                let pair_cap = if (0..2).any(|i| {
                    self.eyes[i].baseline - pe[i].raw
                        >= SHALLOW_COMMIT_DIP * self.eyes[i].blink_depth
                }) {
                    FAST_BLINK_MAX_FRAMES
                } else {
                    FAST_BLINK_CAP_SHALLOW
                };
                for i in 0..2 {
                    self.eyes[i].fast_blink_frames = committed_frames;
                    self.eyes[i].fast_blink_arm = 0;
                    self.eyes[i].squeeze_streak = 0;
                    self.eyes[i].latch_bottom = self.eyes[i].latch_bottom.min(pe[i].raw);
                    self.eyes[i].latch_cap = pair_cap;
                    pe[i].openness_target = 0.0;
                    pe[i].snap_requested = true;
                    pe[i].snap_reason = ClosureReason::FastBlink;
                    pe[i].closed = true;
                }
            }
            let pair_snap = pe.iter().any(|sample| sample.snap_requested);
            let target = pe[0].openness_target.min(pe[1].openness_target);
            for sample in &mut pe {
                sample.openness_target = target;
                sample.closed = sample.closed || target < 0.08;
                if pair_snap && target <= 0.001 {
                    sample.snap_requested = true;
                    sample.snap_reason = ClosureReason::FastBlink;
                }
            }
        }
        let bilateral_invalid_fast_close = bilateral_squeezed_close
            && (0..2).any(|i| {
                !pe[i].gaze_v && self.eyes[i].smooth_open > 0.4 && pe[i].draw <= FAST_CLOSING_RAW
            });

        let mut native_fast = [false; 2];
        for e in Eye::ALL {
            let i = e.idx();
            let native = gaze.eye(e);
            if !native.openness_reported {
                continue;
            }

            let enabled = native.openness_valid;
            let just_disabled = self.eyes[i].native_was_enabled && !enabled;
            if enabled {
                self.eyes[i].native_disable_latch = false;
            } else if just_disabled {
                let d = self.eyes[i].blink_depth.max(0.05);
                let v1 = (-pe[i].draw).max(0.0);
                let fall2 = (self.eyes[i].rawu2 - pe[i].raw).max(0.0);
                let vel_thr = (BLINK_VEL_NORM * d).clamp(BLINK_VEL_FLOOR, BLINK_VEL_CAP);
                let fall2_thr = (BLINK_FALL2_NORM * d).clamp(BLINK_FALL2_FLOOR, BLINK_FALL2_CAP);
                let recent_fast = self.eyes[i].since_down <= 2
                    && self.eyes[i].fall_updates > 0
                    && self.eyes[i].fall_run / self.eyes[i].fall_updates as f32 >= vel_thr;
                let deep = self.eyes[i].baseline - pe[i].raw >= FALL2_COMMIT_DEPTH * d;
                native_fast[i] = v1 >= vel_thr || recent_fast || (fall2 >= fall2_thr && deep);
                self.eyes[i].native_disable_latch = native_fast[i];
            }
            self.eyes[i].native_was_enabled = enabled;

            if self.eyes[i].native_disable_latch {
                pe[i].openness_target = 0.0;
                pe[i].snap_requested = true;
                pe[i].snap_reason = ClosureReason::NativeDisable;
                pe[i].closed = true;
            }
        }

        let both_native_disabled = Eye::ALL.iter().all(|&e| {
            let n = gaze.eye(e);
            n.openness_reported && !n.openness_valid
        });
        if both_native_disabled && native_fast.iter().any(|&fast| fast) {
            for i in 0..2 {
                self.eyes[i].native_disable_latch = true;
                pe[i].openness_target = 0.0;
                pe[i].snap_requested = true;
                pe[i].snap_reason = ClosureReason::NativeDisable;
                pe[i].closed = true;
            }
        }

        if self.uses_continuous_response()
            && !gaze.left.openness_reported
            && !gaze.right.openness_reported
        {
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
                        pe[j].snap_requested = true;
                        pe[j].snap_reason = ClosureReason::FastBlink;
                        pe[j].closed = true;
                    }
                }
            }
        }

        for i in 0..2 {
            let legacy_streak = !self.uses_continuous_response() && self.eyes[i].closed_streak >= 2;
            let gaze_loss_fast =
                !pe[i].gaze_v && self.eyes[i].smooth_open > 0.4 && pe[i].draw <= FAST_CLOSING_RAW;
            if legacy_streak || gaze_loss_fast || bilateral_invalid_fast_close {
                pe[i].snap_requested = true;
                pe[i].snap_reason = ClosureReason::FastBlink;
                pe[i].openness_target = 0.0;
                pe[i].closed = true;
            }
        }
        self.resolve_snap_gate(&mut pe);

        let blink_timing_enabled = blink_timing.is_compatible() && blink_timing.enabled;
        if blink_timing_enabled {
            let bilateral_fast = self.eyes.iter().all(|eye| eye.response_snap_committed);
            let no_wink = self.eyes.iter().all(|eye| !eye.wink_active);
            if !bilateral_fast
                && pe
                    .iter()
                    .all(|eye| eye.present && eye.openness_target > 0.85)
            {
                for eye in &mut self.eyes {
                    eye.blink_pulse_consumed = false;
                }
            }
            if bilateral_fast
                && no_wink
                && pe.iter().all(|eye| eye.present)
                && self.eyes.iter().all(|eye| !eye.blink_pulse_consumed)
            {
                for eye in &mut self.eyes {
                    eye.blink_pulse_active = true;
                    eye.blink_pulse_bottom_frames = 0;
                    eye.blink_pulse_age = 0;
                    eye.blink_pulse_consumed = true;
                }
            }
            for (eye, sample) in self.eyes.iter_mut().zip(pe.iter_mut()) {
                if eye.blink_pulse_active {
                    eye.blink_pulse_age = eye.blink_pulse_age.saturating_add(1);
                    if eye.blink_pulse_age > BLINK_PULSE_MAX_FRAMES {
                        eye.blink_pulse_active = false;
                        eye.blink_pulse_bottom_frames = 0;
                    } else {
                        sample.openness_target = 0.0;
                        sample.reason = ClosureReason::FastBlink;
                        eye.response_snap_committed = true;
                        sample.closed = true;
                    }
                }
            }
        } else {
            for eye in &mut self.eyes {
                eye.blink_pulse_active = false;
                eye.blink_pulse_bottom_frames = 0;
                eye.blink_pulse_age = 0;
                eye.blink_pulse_consumed = false;
            }
        }

        let bilateral_squeezed_commit = bilateral_squeezed_close
            && (0..2).any(|i| {
                let gaze_loss_fast_close = !pe[i].gaze_v
                    && self.eyes[i].smooth_open > 0.4
                    && pe[i].draw <= FAST_CLOSING_RAW;
                pe[i].openness_target < 0.05 || gaze_loss_fast_close
            });
        if bilateral_squeezed_commit {
            for i in 0..2 {
                pe[i].openness_target = 0.0;
                pe[i].snap_requested = true;
                pe[i].snap_reason = ClosureReason::FastBlink;
                pe[i].closed = true;
                self.eyes[i].wink_arm = 0;
                self.eyes[i].wink_active = false;
            }
        }
        if bilateral_squeezed_commit {
            self.resolve_snap_gate(&mut pe);
        }

        for (i, sample) in pe.iter().enumerate() {
            if sample.draw.abs() > 1e-6 {
                self.eyes[i].rawu2 = self.eyes[i].rawu1;
                self.eyes[i].rawu1 = sample.raw;
            }
        }

        if self.tuning.continuous_calib
            && self
                .session_baseline_offset
                .iter()
                .all(|value| value.abs() <= 0.005)
            && !gaze.left.openness_reported
            && !gaze.right.openness_reported
        {
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

        if self.tuning.couple_eyes
            && self
                .session_baseline_offset
                .iter()
                .all(|value| value.abs() <= 0.005)
        {
            self.couple_baselines(pe[0].raw, pe[1].raw);
        }

        if self.tuning.native_squeeze {
            for i in 0..2 {
                let state = &mut self.eyes[i];
                if !pe[i].present {
                    state.native_squeeze_closed_frames = 0;
                    state.native_squeeze_peak = 0.0;
                    state.native_squeeze_envelope = 0.0;
                    pe[i].squeeze_out = 0.0;
                    continue;
                }

                let observed = pe[i].squeeze_out;
                if !pe[i].closed && pe[i].openness_target >= 0.80 {
                    state.native_squeeze_closed_frames = 0;
                    state.native_squeeze_peak = 0.0;
                    state.native_squeeze_envelope = 0.0;
                    pe[i].squeeze_out = 0.0;
                    continue;
                }
                if observed >= state.native_squeeze_peak {
                    state.native_squeeze_peak = observed;
                } else {
                    state.native_squeeze_peak +=
                        NATIVE_SQUEEZE_PEAK_RELEASE_ALPHA * (observed - state.native_squeeze_peak);
                }

                if !pe[i].closed {
                    state.native_squeeze_closed_frames = 0;
                    state.native_squeeze_envelope = 0.0;
                    pe[i].squeeze_out = 0.0;
                    continue;
                }

                state.native_squeeze_closed_frames =
                    state.native_squeeze_closed_frames.saturating_add(1);
                if state.native_squeeze_closed_frames <= NATIVE_SQUEEZE_CLOSED_DELAY {
                    state.native_squeeze_envelope = 0.0;
                    pe[i].squeeze_out = 0.0;
                    continue;
                }

                let target = state.native_squeeze_peak;
                let alpha = if target >= state.native_squeeze_envelope {
                    NATIVE_SQUEEZE_ATTACK_ALPHA
                } else {
                    NATIVE_SQUEEZE_RELEASE_ALPHA
                };
                state.native_squeeze_envelope += alpha * (target - state.native_squeeze_envelope);
                if state.native_squeeze_envelope < 1e-4 {
                    state.native_squeeze_envelope = 0.0;
                }
                pe[i].squeeze_out = state.native_squeeze_envelope.clamp(0.0, 1.0);
            }
        } else {
            for i in 0..2 {
                self.eyes[i].native_squeeze_closed_frames = 0;
                self.eyes[i].native_squeeze_peak = 0.0;
                self.eyes[i].native_squeeze_envelope = 0.0;
                if !pe[i].closed || !pe[i].present {
                    pe[i].squeeze_out = 0.0;
                }
            }
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
            let strict_pair = self.eyes[0].is_wide && self.eyes[1].is_wide;
            let supported_native_pair = self.native_wide_response
                && pe.iter().all(|eye| eye.present && !eye.closed)
                && (0..2).any(|i| {
                    let j = 1 - i;
                    let own_margin = (self.eyes[i].wide_entry_offset
                        + self.appearance_wide_entry_shift[i]).max(WIDE_ENTRY_MIN_OFFSET);
                    let partner_margin = (self.eyes[j].wide_entry_offset
                        + self.appearance_wide_entry_shift[j]).max(WIDE_ENTRY_MIN_OFFSET);
                    pe[i].raw > self.effective_baseline(i) + 2.0 * own_margin
                        && pe[j].raw > self.effective_baseline(j) + 0.5 * partner_margin
                });
            let both_wide = strict_pair || supported_native_pair;
            let blinking = pe[0].closed || pe[1].closed;
            let gate_target = if both_wide { 1.0 } else { 0.0 };
            if blinking {
            } else if self.wide_gate < gate_target {
                self.wide_gate = (self.wide_gate + WIDE_GATE_STEP).min(gate_target);
            } else if self.wide_gate > gate_target {
                self.wide_gate = (self.wide_gate - WIDE_GATE_STEP).max(gate_target);
            }
            let m = if blinking {
                0.0
            } else {
                pe[0].wide_out.max(pe[1].wide_out) * self.wide_gate
            };
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
            let endpoint_closed = if self.response_profile_installed && self.response_manual_range {
                target <= 0.0
            } else {
                target < 0.05
            };
            let forced_closed = endpoint_closed
                || !s.present
                || self.eyes[i].response_snap_committed
                || self.eyes[i].blink_pulse_active;
            let mut openness_out = if forced_closed {
                self.eyes[i].kalman.reset();
                0.0
            } else if self.tuning.adaptive_kalman {
                self.eyes[i].kalman.update(target)
            } else {
                let base = if target < prev {
                    self.tuning.alpha_close
                } else {
                    self.tuning.alpha_open
                };
                let alpha_o =
                    (base * (1.0 - self.tuning.smoothing.clamp(0.0, 1.0))).clamp(0.03, 1.0);
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
            let close_ms = if self.response_profile_installed {
                self.response_blink_close_ms
            } else {
                self.tuning.blink_close_ms.clamp(0.0, BLINK_CLOSE_MS_MAX)
            };
            if close_ms >= 1.0 {
                let step = 1000.0 / (120.0 * close_ms);
                if openness_out < prev - step {
                    openness_out = prev - step;
                }
            }
            if self.eyes[i].blink_pulse_active {
                if openness_out <= 0.001 {
                    self.eyes[i].blink_pulse_bottom_frames =
                        self.eyes[i].blink_pulse_bottom_frames.saturating_add(1);
                    if self.eyes[i].blink_pulse_bottom_frames >= blink_timing.minimum_frames() {
                        self.eyes[i].blink_pulse_active = false;
                    }
                } else {
                    self.eyes[i].blink_pulse_bottom_frames = 0;
                }
            }
            self.eyes[i].smooth_open = openness_out;
            self.eyelid_live[i] = EyelidLiveDiag {
                raw_openness: s.raw,
                model_response: s.model_response,
                final_openness: openness_out,
                calibrated_baseline: self.eyes[i].baseline,
                effective_baseline: self.effective_baseline(i),
                session_baseline_offset: self.session_baseline_offset[i],
                effective_open_ref: self.effective_open_ref(i),
                effective_closed_ref: self.effective_closed_ref(i),
                wide_entry_ref: (self.effective_baseline(i)
                    + self.eyes[i].wide_entry_offset
                    + self.appearance_wide_entry_shift[i])
                    .clamp(0.0, 1.0),
                wide_full_ref: self.eyes[i].wide_ceiling.clamp(0.0, 1.0),
                reason: s.reason,
            };

            if self.tuning.gaze_yoke {
                if s.closed || openness_out < YOKE_ENGAGE_OPEN || !s.gaze_v {
                    self.eyes[i].yoke_hold = true;
                } else if openness_out > YOKE_RELEASE_OPEN && s.gaze_v {
                    self.eyes[i].yoke_hold = false;
                }
            } else {
                self.eyes[i].yoke_hold = false;
            }

            let (sx, sy, sz) = if s.gaze_v && !self.eyes[i].yoke_hold {
                let [sx, sy, sz] = if self.gaze_smoothing_enabled {
                    adaptive_gaze_step(self.eyes[i].gaze_smooth, s.gaze)
                } else {
                    s.gaze
                };
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
    model_response: f32,
    snap_requested: bool,
    snap_reason: ClosureReason,
    reason: ClosureReason,

    wide_out: f32,
    squeeze_out: f32,
    squeeze_delta: f32,
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

    fn native_wide_test_state(enabled: bool) -> SRanipalState {
        let mut state = SRanipalState::new();
        state.set_native_wide_response(enabled);
        for eye in &mut state.eyes {
            eye.baseline = 0.5;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            eye.frame_count = WARMUP_FRAMES;
            eye.prev_raw = 0.5;
            eye.wide_entry_offset = 0.02;
            eye.closed_ref = 0.3;
        }
        state
    }

    fn native_wide_test_step(state: &mut SRanipalState, raw: [f32; 2], present: [bool; 2]) -> [EyeResult; 2] {
        let mut gaze = GazeSample::default();
        gaze.left.openness_reported = true;
        gaze.right.openness_reported = true;
        gaze.left.openness_valid = present[0] && raw[0] > 0.1;
        gaze.right.openness_valid = present[1] && raw[1] > 0.1;
        state.process_frame_with_model_evidence(
            std::array::from_fn(|i| [if present[i] {1.0} else {0.0},raw[i],0.0,0.0,0.0]),
            &gaze,true,&GazeEyelidProfile::default(),&WinkProfile::default(),&BlinkTimingProfile::default(),
        )
    }

    #[test]
    fn native_wide_span_matches_aperture_units_without_changing_legacy() {
        let mut old = native_wide_test_state(false);
        let mut new = native_wide_test_state(true);
        let mut out_old = [EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
        let mut out_new = out_old;
        for _ in 0..80 {
            out_old = native_wide_test_step(&mut old,[0.54;2],[true;2]);
            out_new = native_wide_test_step(&mut new,[0.54;2],[true;2]);
        }
        assert!(out_old[0].wide < 0.15);
        assert!(out_new[0].wide > 0.45);
        assert_eq!(old.eyes[0].wide_ceiling,0.52+WIDE_CEIL_MIN_SPAN);
        assert_eq!(new.eyes[0].wide_ceiling,0.52+NATIVE_WIDE_MIN_SPAN);
        assert_eq!(out_old[0].openness.to_bits(),out_new[0].openness.to_bits());
        assert!(!SRanipalState::new().native_wide_response);
    }

    #[test]
    fn native_wide_accepts_a_weaker_positive_partner_but_not_neutral_partner() {
        for strong in 0..2 {
            let mut old = native_wide_test_state(false);
            let mut new = native_wide_test_state(true);
            let mut raw=[0.511;2];raw[strong]=0.56;
            let mut result=[EyeResult::new(Eye::Left),EyeResult::new(Eye::Right)];
            for _ in 0..40 {
                let old_result=native_wide_test_step(&mut old,raw,[true;2]);
                assert_eq!([old_result[0].wide,old_result[1].wide],[0.0;2]);
                result=native_wide_test_step(&mut new,raw,[true;2]);
            }
            assert!(result[0].wide>0.9);
            assert_eq!(result[0].wide.to_bits(),result[1].wide.to_bits());
            let mut neutral_partner=native_wide_test_state(true);
            raw[1-strong]=0.5;
            for _ in 0..80 {
                let out=native_wide_test_step(&mut neutral_partner,raw,[true;2]);
                assert_eq!([out[0].wide,out[1].wide],[0.0;2]);
            }
        }
    }

    #[test]
    fn native_wide_noise_closed_and_missing_partner_never_qualify() {
        let mut noise=native_wide_test_state(true);
        for n in 0..600 {
            let raw=[0.5+0.002*(n%3) as f32-0.002,0.5+0.002*(n%5) as f32-0.004];
            let out=native_wide_test_step(&mut noise,raw,[true;2]);
            assert_eq!([out[0].wide,out[1].wide],[0.0;2]);
        }
        for partner in 0..2 {
            for missing in [false,true] {
                let mut state=native_wide_test_state(true);
                let mut raw=[0.56;2];raw[partner]=0.1;
                let mut present=[true;2];present[partner]=!missing;
                for _ in 0..40 {
                    let out=native_wide_test_step(&mut state,raw,present);
                    assert_eq!([out[0].wide,out[1].wide],[0.0;2]);
                }
            }
        }
    }

    #[test]
    fn native_wide_reset_and_recenter_keep_decoder_selection() {
        let mut state=native_wide_test_state(true);
        native_wide_test_step(&mut state,[0.56;2],[true;2]);
        state.reset_eyelid_calibration();
        assert!(state.native_wide_response);
        assert_eq!(state.wide_gate,0.0);
        state.recenter();
        assert!(state.native_wide_response);
        state.set_native_wide_response(false);
        assert!(!state.native_wide_response);
        assert_eq!(state.eyes[0].wide_ceiling,0.0);
    }

    #[test]
    fn missing_model_evidence_discards_partial_learning_pair() {
        let mut state = SRanipalState::new();
        state.pend.w_sum = 5.0;
        state.pend.wx = [0.3, 0.4];
        state.process_frame_with_model_evidence(
            [[0.0, 0.5, 0.0, 0.0, 0.0]; 2], &GazeSample::default(), true,
            &GazeEyelidProfile::default(), &WinkProfile::default(), &BlinkTimingProfile::default(),
        );
        assert_eq!(state.pend.w_sum, 0.0);
        assert_eq!(state.pend.wx, [0.0; 2]);
    }

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
    fn wide_neutral_capture_learns_independent_per_eye_noise_bands() {
        let mut state = SRanipalState::new();
        let gaze = GazeSample::default();
        let left_noise = [-0.001, 0.0, 0.001, 0.0];
        let right_noise = [-0.008, -0.004, 0.0, 0.004, 0.008];

        for frame in 0..110 {
            let left = 0.62 + left_noise[frame % left_noise.len()];
            let right = 0.62 + right_noise[frame % right_noise.len()];
            state.process_frame(
                [[1.0, left, 0.0, 0.0, 0.0], [1.0, right, 0.0, 0.0, 0.0]],
                &gaze,
                true,
            );
        }

        let left = state.eyes[0].wide_entry_offset;
        let right = state.eyes[1].wide_entry_offset;
        assert!(
            (left - WIDE_ENTRY_MIN_OFFSET).abs() < 1e-4,
            "quiet eye should use the safe minimum, got {left}"
        );
        assert!(
            right > left + 0.008 && right <= WIDE_ENTRY_MAX_OFFSET,
            "noisy eye should receive its own wider band, left={left} right={right}"
        );
    }

    #[test]
    fn wide_neutral_single_close_outlier_does_not_desensitize_entry() {
        let mut samples = [0.0; WIDE_NEUTRAL_SAMPLE_CAP];
        for (i, sample) in samples.iter_mut().enumerate() {
            *sample = 0.62 + [-0.001, 0.0, 0.001, 0.0][i % 4];
        }
        samples[17] = 0.40;

        let offset = wide_entry_offset_from_neutral(&samples, samples.len());
        assert!(
            offset <= 0.008,
            "one close outlier must not make Wide unreachable, got {offset}"
        );
    }

    fn native_gaze(left_enabled: bool, right_enabled: bool) -> GazeSample {
        let eye = |enabled| crate::core::types::EyeSample {
            openness: if enabled { 1.0 } else { 0.0 },
            openness_valid: enabled,
            openness_reported: true,
            ..Default::default()
        };
        GazeSample {
            left: eye(left_enabled),
            right: eye(right_enabled),
            ..Default::default()
        }
    }

    fn native_step(
        st: &mut SRanipalState,
        left_raw: f32,
        right_raw: f32,
        left_enabled: bool,
        right_enabled: bool,
    ) -> [EyeResult; 2] {
        let ml = [
            [1.0, left_raw, 0.0, 0.0, 0.0],
            [1.0, right_raw, 0.0, 0.0, 0.0],
        ];
        st.process_frame(ml, &native_gaze(left_enabled, right_enabled), true)
    }

    #[test]
    fn native_slow_disable_keeps_following_ml_ramp() {
        let mut st = SRanipalState::new();
        st.tuning.alpha_close = 1.0;
        for _ in 0..260 {
            native_step(&mut st, 0.62, 0.62, true, true);
        }
        for raw in [0.60, 0.58, 0.56, 0.54, 0.52, 0.50] {
            native_step(&mut st, raw, raw, true, true);
        }
        let disabled = native_step(&mut st, 0.49, 0.49, false, false);
        assert!(
            disabled[0].openness > 0.1 && disabled[1].openness > 0.1,
            "slow Disable must stay on the ML ramp: {} / {}",
            disabled[0].openness,
            disabled[1].openness
        );
        let lower = native_step(&mut st, 0.47, 0.47, false, false);
        assert!(
            lower[0].openness < disabled[0].openness && lower[0].openness > 0.0,
            "disabled slow close must keep descending with ML"
        );
    }

    #[test]
    fn native_fast_disable_snaps_both_and_enable_releases() {
        let mut st = SRanipalState::new();
        st.tuning.alpha_open = 1.0;
        for _ in 0..260 {
            native_step(&mut st, 0.62, 0.62, true, true);
        }
        native_step(&mut st, 0.54, 0.54, true, true);
        let closed = native_step(&mut st, 0.42, 0.44, false, false);
        assert_eq!([closed[0].openness, closed[1].openness], [0.0, 0.0]);
        let held = native_step(&mut st, 0.48, 0.49, false, false);
        assert_eq!([held[0].openness, held[1].openness], [0.0, 0.0]);
        let first_enable = native_step(&mut st, 0.62, 0.62, true, true);
        assert_eq!(
            [first_enable[0].openness, first_enable[1].openness],
            [0.0, 0.0],
            "confirmed bilateral blink keeps its short visible bottom"
        );
        let mut reopened = first_enable;
        for _ in 0..12 {
            reopened = native_step(&mut st, 0.62, 0.62, true, true);
        }
        assert!(reopened[0].openness > 0.5 && reopened[1].openness > 0.5);
    }

    #[test]
    fn native_fast_wink_does_not_snap_other_eye() {
        let mut st = SRanipalState::new();
        for _ in 0..260 {
            native_step(&mut st, 0.62, 0.62, true, true);
        }
        native_step(&mut st, 0.54, 0.62, true, true);
        let wink = native_step(&mut st, 0.40, 0.62, false, true);
        assert_eq!(wink[0].openness, 0.0);
        assert!(wink[1].openness > 0.8, "open partner must remain open");
    }

    #[test]
    fn blink_open_close_open_cycle() {
        let mut st = SRanipalState::new();

        let open = feed(&mut st, 0.62, 260);
        assert!(
            st.baseline(Eye::Left) > 0.58 && st.baseline(Eye::Left) < 0.66,
            "baseline learned ~0.62, got {}",
            st.baseline(Eye::Left)
        );
        assert!(
            open[0].openness > 0.8,
            "warm open openness {}",
            open[0].openness
        );
        assert!(!open[0].blink, "should not be blinking when open");

        let closed = feed(&mut st, 0.30, 12);
        assert_eq!(closed[0].openness, 0.0, "blink should drive openness to 0");
        assert!(closed[0].blink, "blink flag set when closed");
        assert_eq!(closed[1].openness, 0.0, "both eyes closed");

        let reopened = feed(&mut st, 0.62, 40);
        assert!(
            reopened[0].openness > 0.5,
            "reopened openness {}",
            reopened[0].openness
        );
        assert!(!reopened[0].blink, "no longer blinking after reopen");
    }

    #[test]
    fn adaptive_kalman_tracks_then_deadzones() {
        let mut k = ScalarKalman::default();
        assert_eq!(k.update(0.70), 0.70);
        for _ in 0..40 {
            k.update(0.20);
        }
        assert!(
            k.x < 0.30,
            "fast regime should track the step down, got {}",
            k.x
        );
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
        assert!(
            open[0].openness > 0.7,
            "kalman-smoothed open openness {}",
            open[0].openness
        );
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
        assert_eq!(
            last[0].squeeze, 0.0,
            "open left eye must suppress native squeeze"
        );
        assert_eq!(
            last[1].squeeze, 0.0,
            "open right eye must suppress native squeeze"
        );

        for _ in 0..20 {
            let ml = [[1.0, 0.30, 0.0, 0.60, 0.0], [1.0, 0.30, 0.0, 0.60, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(
            last[0].blink && last[0].squeeze > 0.7,
            "closed left squeeze should fire, got {}",
            last[0].squeeze
        );
        assert!(
            last[1].blink && last[1].squeeze > 0.7,
            "closed right squeeze should fire, got {}",
            last[1].squeeze
        );
    }

    #[test]
    fn native_squeeze_starts_after_close_and_rises_smoothly() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut closed_values = Vec::new();

        for _ in 0..48 {
            let ml = [[1.0, 0.30, 0.0, 0.60, 0.0], [1.0, 0.30, 0.0, 0.60, 0.0]];
            let out = st.process_frame(ml, &g, true);
            if out[0].blink {
                closed_values.push(out[0].squeeze);
            }
        }

        assert!(
            closed_values.len() > NATIVE_SQUEEZE_CLOSED_DELAY as usize + 4,
            "sustained close must produce enough closed frames"
        );
        assert!(
            closed_values
                .iter()
                .take(NATIVE_SQUEEZE_CLOSED_DELAY as usize)
                .all(|value| *value == 0.0),
            "native squeeze must stay zero during the initial closed hold: {closed_values:?}"
        );
        let first_active = closed_values[NATIVE_SQUEEZE_CLOSED_DELAY as usize];
        assert!(
            first_active > 0.0 && first_active < 0.30,
            "first active squeeze frame must be intermediate, got {first_active}"
        );
        assert!(
            closed_values.last().copied().unwrap_or_default() > 0.7,
            "held squeeze must still reach full response"
        );

        let mut saw_open = false;
        for _ in 0..30 {
            let ml = [[1.0, 0.62, 0.0, 0.60, 0.0], [1.0, 0.62, 0.0, 0.60, 0.0]];
            let out = st.process_frame(ml, &g, true);
            if !out[0].blink {
                saw_open = true;
                assert_eq!(out[0].squeeze, 0.0, "reopen must clear squeeze");
                break;
            }
        }
        assert!(saw_open, "eye should reopen during the test window");
    }

    #[test]
    fn short_native_close_does_not_emit_squeeze() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut peak = 0.0f32;

        for _ in 0..NATIVE_SQUEEZE_CLOSED_DELAY.saturating_sub(2) {
            let ml = [[1.0, 0.20, 0.0, 0.70, 0.0], [1.0, 0.20, 0.0, 0.70, 0.0]];
            let out = st.process_frame(ml, &g, true);
            peak = peak.max(out[0].squeeze).max(out[1].squeeze);
        }
        for _ in 0..24 {
            let ml = [[1.0, 0.62, 0.0, 0.05, 0.0], [1.0, 0.62, 0.0, 0.05, 0.0]];
            let out = st.process_frame(ml, &g, true);
            peak = peak.max(out[0].squeeze).max(out[1].squeeze);
        }

        assert_eq!(peak, 0.0, "short close leaked squeeze {peak}");
    }

    #[test]
    fn native_squeeze_keeps_the_closing_stroke_peak_until_the_closed_gate() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();

        for _ in 0..4 {
            let ml = [[1.0, 0.52, 0.0, 0.63, 0.0], [1.0, 0.52, 0.0, 0.63, 0.0]];
            let out = st.process_frame(ml, &g, true);
            assert_eq!(out[0].squeeze, 0.0);
            assert_eq!(out[1].squeeze, 0.0);
        }

        let mut peak = 0.0f32;
        for _ in 0..24 {
            let ml = [[1.0, 0.20, 0.0, 0.24, 0.0], [1.0, 0.20, 0.0, 0.24, 0.0]];
            let out = st.process_frame(ml, &g, true);
            peak = peak.max(out[0].squeeze).max(out[1].squeeze);
        }
        assert!(
            peak > 0.45,
            "closing-stroke squeeze peak was lost at full closure: {peak}"
        );
    }

    #[test]
    fn native_squeeze_per_eye_independent() {
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            let ml = [[1.0, 0.30, 0.0, 0.60, 0.0], [1.0, 0.62, 0.0, 0.05, 0.0]];
            last = st.process_frame(ml, &g, true);
        }
        assert!(
            last[0].blink && last[0].squeeze > 0.6,
            "closed left squeeze fires, got {}",
            last[0].squeeze
        );
        assert!(!last[1].blink, "right eye remains open");
        assert!(
            last[1].squeeze < 0.1,
            "right squeeze stays low (no yoke), got {}",
            last[1].squeeze
        );
    }

    #[test]
    fn wide_fires_above_upper() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let wide = feed(&mut st, 0.79, 20);
        assert!(wide[0].wide > 0.05, "wide magnitude {}", wide[0].wide);
        assert!(
            wide[0].openness > 0.9,
            "wide implies fully open, got {}",
            wide[0].openness
        );
    }

    #[test]
    fn fast_blink_snaps_fully_closed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let blink = feed(&mut st, 0.48, 6);
        assert!(
            blink[0].openness < 0.1,
            "fast blink should snap closed, got {}",
            blink[0].openness
        );
        assert!(blink[0].blink, "blink flag set during fast blink");
        let reopened = feed(&mut st, 0.62, 40);
        assert!(
            reopened[0].openness > 0.5,
            "reopened after fast blink, got {}",
            reopened[0].openness
        );
        assert!(!reopened[0].blink, "no longer blinking after reopen");
    }

    #[test]
    fn blink_reopen_into_wide_then_settle_stays_open() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let blink = feed(&mut st, 0.48, 4);
        assert!(
            blink[0].openness < 0.1,
            "blink closed, got {}",
            blink[0].openness
        );
        feed(&mut st, 0.80, 6);
        let settled = feed(&mut st, 0.62, 8);
        assert!(
            settled[0].openness > 0.5,
            "settling out of wide must stay open, got {}",
            settled[0].openness
        );
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
        assert!(
            last[0].openness > 0.15,
            "slow partial close stays partly open, got {}",
            last[0].openness
        );
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
        assert!(
            last[0].wide < 1e-3,
            "single-eye (left) wide suppressed, got {}",
            last[0].wide
        );
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
        assert!(
            first < 0.25,
            "wide must not dump on engage, first-frame {}",
            first
        );
        assert!(
            last[0].wide > 0.6,
            "wide reaches full after the ramp, got {}",
            last[0].wide
        );
        assert!(
            last[0].wide > first + 0.3,
            "wide ramped up (from {} to {})",
            first,
            last[0].wide
        );
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
            gate_held,
            gate_bottom,
            gate_held - gate_bottom
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
                f, st.wide_gate, st.eyes[0].is_wide, st.eyes[1].is_wide, rr[1].blink, rr[0].wide
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
        assert!(
            last[1].wide > 0.05,
            "right wide should fire, got {}",
            last[1].wide
        );
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
        assert!(
            out[0].gaze_yoked,
            "left gaze is yoked to the tracked (right) eye"
        );
        assert!(
            out[0].gaze_valid,
            "yoked gaze is emitted as valid (so it drives the avatar)"
        );
        assert_eq!(
            out[0].gaze, out[1].gaze,
            "squinting eye mirrors the open eye's gaze exactly"
        );
        assert!(!out[1].gaze_yoked, "the open eye is never the one yoked");
    }

    #[test]
    fn gaze_yoke_engages_on_a_shallow_non_blink_squint() {
        let mut st = SRanipalState::new();
        let g = gaze_lr([0.4, 0.1, 0.9], [-0.4, 0.1, 0.9]);
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        for _ in 0..300 {
            st.process_frame(open, &g, true);
        }

        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for step in 1..=80 {
            let raw = 0.62 - 0.16 * (step as f32 / 80.0);
            let shallow = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            out = st.process_frame(shallow, &g, true);
        }
        assert!(
            !out[0].blink,
            "shallow squint must not be reclassified as a blink"
        );
        assert!(
            out[0].openness < YOKE_ENGAGE_OPEN,
            "left openness {}",
            out[0].openness
        );
        assert!(
            out[0].gaze_yoked,
            "shallow squint should use the open partner's gaze"
        );
        assert_eq!(out[0].gaze, out[1].gaze);
    }

    #[test]
    fn gaze_yoke_uses_partner_after_own_gaze_really_drops_out() {
        let mut st = SRanipalState::new();
        let valid = gaze_lr([0.35, 0.05, 0.93], [-0.35, 0.05, 0.93]);
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        for _ in 0..300 {
            st.process_frame(open, &valid, true);
        }

        let mut left_lost = valid;
        left_lost.left.gaze_valid = false;
        let out = st.process_frame(open, &left_lost, true);
        assert!(out[0].gaze_yoked);
        assert!(out[0].gaze_valid);
        assert_eq!(out[0].gaze, out[1].gaze);
        assert!(!out[1].gaze_yoked);
    }

    #[test]
    fn calm_gaze_noise_is_heavily_smoothed_without_blocking_motion() {
        let mut st = SRanipalState::new();
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        let centre = gaze_lr([0.2, 0.0, 0.98], [-0.2, 0.0, 0.98]);
        for _ in 0..300 {
            st.process_frame(open, &centre, true);
        }

        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for i in 0..120 {
            let noise = if i % 2 == 0 { 0.01 } else { -0.01 };
            let g = gaze_lr([0.2 + noise, 0.0, 0.98], [-0.2 + noise, 0.0, 0.98]);
            let out = st.process_frame(open, &g, true);
            lo = lo.min(out[0].gaze[0]);
            hi = hi.max(out[0].gaze[0]);
        }
        assert!(hi - lo < 0.004, "calm gaze still jitters: {lo}..{hi}");

        let moved = gaze_lr([0.5, 0.0, 0.86], [-0.5, 0.0, 0.86]);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..4 {
            out = st.process_frame(open, &moved, true);
        }
        assert!(
            out[0].gaze[0] > 0.45,
            "real gaze move became sluggish: {}",
            out[0].gaze[0]
        );
    }

    #[test]
    fn gaze_smoothing_off_bypasses_only_ema_and_survives_eyelid_reset() {
        let mut state = SRanipalState::new();
        state.tuning.gaze_yoke = false;
        state.set_gaze_smoothing(false);
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0]; 2];
        let gaze = gaze_lr([0.6, 0.2, 0.8], [-0.4, -0.1, 0.9]);
        for reset in [false, true] {
            if reset { state.reset_eyelid_calibration(); }
            let out = state.process_frame(open, &gaze, true);
            assert_eq!(out[0].gaze, gaze.left.gaze);
            assert_eq!(out[1].gaze, gaze.right.gaze);
            assert!(!out[0].gaze_yoked && !out[1].gaze_yoked);
        }
        let mut lost = gaze;
        lost.left.gaze_valid = false;
        lost.left.gaze = [f32::NAN; 3];
        let out = state.process_frame(open, &lost, true);
        assert!(!out[0].gaze_valid);
        assert_eq!(out[0].gaze, gaze.left.gaze);
        assert_eq!(out[1].gaze, gaze.right.gaze);

        state.set_gaze_smoothing(true);
        let moved = gaze_lr([0.61, 0.2, 0.8], [-0.39, -0.1, 0.9]);
        let out = state.process_frame(open, &moved, true);
        assert_eq!(out[0].gaze, adaptive_gaze_step(gaze.left.gaze, moved.left.gaze));
        assert_eq!(out[1].gaze, adaptive_gaze_step(gaze.right.gaze, moved.right.gaze));
    }

    #[test]
    fn gaze_yoke_does_not_tremble_at_blink_boundary() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let own_garbage = [-0.5f32, 0.2, -1.0];
        let partner = [0.3f32, 0.0, -1.0];
        let g = gaze_lr(own_garbage, partner);
        for _ in 0..12 {
            st.process_frame(
                [[1.0, 0.45, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        let mut prev_x: Option<f32> = None;
        let mut max_step = 0.0f32;
        let mut flip = false;
        for _ in 0..60 {
            let rl = if flip { 0.455 } else { 0.475 };
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
            assert!(
                out[0].gaze_yoked,
                "squinting eye stays yoked through the hover"
            );
        }
        assert!(
            (prev_x.unwrap() - partner[0]).abs() < 0.02,
            "squinting eye pinned to the partner, got x {}",
            prev_x.unwrap()
        );
        assert!(
            max_step < 0.02,
            "no gaze trembling at the boundary (max step {max_step})"
        );
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..40 {
            out = st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        assert!(!out[0].gaze_yoked, "yoke releases after a real reopen");
        assert!(
            (out[0].gaze[0] - own_garbage[0]).abs() < 0.05,
            "own gaze resumes after release, got x {}",
            out[0].gaze[0]
        );
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
            st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        for _ in 0..8 {
            st.process_frame(
                [[1.0, 0.36, 0.0, 0.0, 0.0], [1.0, 0.36, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
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
        assert!(
            !crossed,
            "recovering eyes must not mirror each other after a blink"
        );
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
        assert!(
            !out[0].gaze_yoked && !out[1].gaze_yoked,
            "no yoke while both eyes track"
        );
        assert_ne!(out[0].gaze, out[1].gaze, "each open eye keeps its own gaze");
    }

    #[test]
    fn gaze_yoke_disabled_leaves_squint_eye_unyoked() {
        let mut st = SRanipalState::new();
        st.tuning.gaze_yoke = false;
        let g = gaze_lr([0.4, 0.0, 0.9], [-0.4, 0.0, 0.9]);
        for _ in 0..260 {
            st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..12 {
            out = st.process_frame(
                [[1.0, 0.30, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        assert!(out[0].blink, "left still squinting");
        assert!(!out[0].gaze_yoked, "yoke disabled → no mirroring");
        assert_ne!(
            out[0].gaze, out[1].gaze,
            "unyoked squint eye does not adopt the open eye's gaze"
        );
    }

    #[test]
    fn open_eye_squeeze_noise_never_suppresses_wide() {
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
        assert!(off[0].wide > 0.05, "wide should fire, got {}", off[0].wide);
        assert_eq!(off[0].squeeze, 0.0, "open-eye squeeze noise must be gated");

        st.tuning.wide_squeeze_exclusive = true;
        let mut on = off;
        for _ in 0..25 {
            on = st.process_frame(ml, &g, true);
        }
        assert!(
            on[0].wide > 0.05,
            "open-eye squeeze noise must not suppress wide, got {}",
            on[0].wide
        );
        assert_eq!(
            on[0].squeeze, 0.0,
            "open-eye squeeze remains gated with chain enabled"
        );
    }

    #[test]
    fn relaxed_below_baseline_reads_open_via_deadzone() {
        let mut st = SRanipalState::new();
        let g = GazeSample::default();
        for _ in 0..300 {
            st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        for k in 0..6 {
            let raw = 0.62 - 0.01 * (k as f32);
            st.process_frame(
                [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..15 {
            out = st.process_frame(
                [[1.0, 0.56, 0.0, 0.0, 0.0], [1.0, 0.56, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        assert!(
            out[0].openness > 0.95,
            "a relaxed eye ~0.06 below baseline must read fully open, got {}",
            out[0].openness
        );
        assert!(
            !out[0].blink,
            "not a blink — just relaxed slightly below baseline"
        );
    }

    #[test]
    fn deadzone_still_blinks_and_wide_still_fires() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300);
        let closed = feed(&mut st, 0.30, 12);
        assert_eq!(closed[0].openness, 0.0, "blink still drives openness to 0");
        assert!(closed[0].blink);
        let wide = feed(&mut st, 0.80, 24);
        assert!(
            wide[0].openness > 0.95,
            "wide implies fully open, got {}",
            wide[0].openness
        );
        assert!(
            wide[0].wide > 0.05,
            "wide still fires with the wider dead-zone, got {}",
            wide[0].wide
        );
    }

    #[test]
    fn held_wide_does_not_drift_the_baseline() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300);
        let base0 = st.baseline(Eye::Left);
        let out = feed(&mut st, 0.68, 400);
        let base1 = st.baseline(Eye::Left);
        assert!(
            (base1 - base0).abs() < 0.02,
            "held wide must not drift the baseline ({base0} -> {base1})"
        );
        assert!(
            out[0].wide > 0.05,
            "wide still fires after a long hold, got {}",
            out[0].wide
        );
    }

    #[test]
    fn sustained_squint_does_not_sink_baseline_or_fire_wide() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.53, 1200);
        let b1 = st.baseline(Eye::Left);
        assert!(
            (b1 - b0).abs() < 0.01,
            "squint sank the baseline ({b0} -> {b1})"
        );
        let g = GazeSample::default();
        let ml = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        let mut max_wide: f32 = 0.0;
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..120 {
            last = st.process_frame(ml, &g, true);
            max_wide = max_wide.max(last[0].wide).max(last[1].wide);
        }
        assert!(max_wide < 0.01, "spurious wide after a squint: {max_wide}");
        assert!(
            last[0].openness > 0.9,
            "openness after squint {}",
            last[0].openness
        );
    }

    #[test]
    fn latched_wide_is_frozen_until_the_explicit_breaker() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.68, 1200);
        let b1 = st.baseline(Eye::Left);
        assert!(
            (b1 - b0).abs() < 1e-6,
            "sub-breaker wide moved the mature baseline ({b0} -> {b1})"
        );
    }

    #[test]
    fn stuck_wide_breaker_recovers_bounded() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.53,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
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
        assert!(
            last[0].wide > 0.3,
            "wide collapsed during a 10s hold: {}",
            last[0].wide
        );
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
        assert!(
            b < 0.65,
            "one frame after recenter snapped the baseline to {b}"
        );
        feed(&mut st, 0.62, 300);
        let b2 = st.baseline(Eye::Left);
        assert!(
            (b2 - 0.62).abs() < 0.01,
            "baseline should settle back to 0.62, got {b2}"
        );
    }

    #[test]
    fn recenter_reanchors_across_the_ordinary_outlier_window() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.recenter();
        feed(&mut st, 0.43, 100);

        let baseline = st.baseline(Eye::Left);
        assert!(
            (baseline - 0.43).abs() < 0.02,
            "explicit recenter did not acquire the reseated level: {baseline}"
        );
        assert!(
            !st.eyes[0].recenter_active,
            "explicit acquisition must end after the bounded bootstrap"
        );
    }

    #[test]
    fn recenter_ignores_closed_frames_then_acquires_relaxed_open() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.recenter();
        feed(&mut st, 0.20, 60);
        assert!((st.baseline(Eye::Left) - 0.62).abs() < 1e-5);
        assert!(st.eyes[0].recenter_active);

        feed(&mut st, 0.43, 100);
        assert!((st.baseline(Eye::Left) - 0.43).abs() < 0.02);
        assert!(!st.eyes[0].recenter_active);
    }

    #[test]
    fn one_eye_recenter_preserves_the_other_eye_baseline_and_state() {
        let mut state = SRanipalState::new();
        state.eyes[0].baseline = 0.53;
        state.eyes[0].baseline_n = 321;
        state.eyes[0].recenter_active = false;
        state.eyes[1].baseline = 0.44;
        state.eyes[1].baseline_n = 654;
        state.eyes[1].recenter_active = false;
        state.session_baseline_offset = [0.03, -0.02];

        state.recenter_eye(Eye::Right);

        assert_eq!(state.eyes[0].baseline.to_bits(), 0.53_f32.to_bits());
        assert_eq!(state.eyes[0].baseline_n, 321);
        assert!(!state.eyes[0].recenter_active);
        assert_eq!(
            state.session_baseline_offset[0].to_bits(),
            0.03_f32.to_bits()
        );
        assert_eq!(state.eyes[1].baseline.to_bits(), 0.44_f32.to_bits());
        assert_eq!(state.eyes[1].baseline_n, RECENTER_PRIOR_N);
        assert!(state.eyes[1].recenter_active);
        assert_eq!(state.session_baseline_offset[1], 0.0);
    }

    #[test]
    fn full_eyelid_reset_forgets_learned_locked_and_transient_expression_state() {
        let mut state = SRanipalState::new();
        feed(&mut state, 0.48, 240);
        state.apply_explicit_endpoint(Eye::Left, 0.49, 0.14, 1234);
        state.apply_explicit_endpoint(Eye::Right, 0.53, 0.19, 5678);
        for eye in &mut state.eyes {
            eye.mid_anchor = 0.72;
            eye.mid_anchor_staged = Some(0.68);
            eye.blink_len = 32;
            eye.blink_candidate = Some(0.25);
            eye.blink_candidate_count = 2;
            eye.floor_travel_deeper = 0.03;
            eye.wide_entry_offset = 0.09;
            eye.wide_neutral_n = 10;
            eye.wide_neutral_samples[0] = 0.51;
            eye.wide_ceiling = 0.82;
            eye.squeeze_floor = 0.29;
            eye.native_squeeze_closed_frames = 12;
            eye.native_squeeze_peak = 0.78;
            eye.native_squeeze_envelope = 0.67;
            eye.is_closed = true;
            eye.is_wide = true;
            eye.native_disable_latch = true;
            eye.fast_blink_frames = 5;
            eye.wink_active = true;
            eye.blink_pulse_active = true;
            eye.response_snap_committed = true;
            eye.kalman.update(0.23);
        }
        state.set_eyelid_response_tuning(
            true, [0.08; 2], [0.28; 2], [1.2; 2], [0.65; 2], 80.0, 0.4, true,
        );
        state.session_baseline_offset = [0.03, -0.02];
        state.appearance_profile_active = true;
        state.appearance_baseline_offset = [-0.06, 0.05];
        state.appearance_wide_entry_shift = [0.04, -0.03];
        state.reseat_candidate = [0.02, -0.03];
        state.reseat_prev_raw = [0.45, 0.48];
        state.reseat_stable_frames = 60;
        state.wide_gate = 0.8;
        state.gaze_eyelid_lift = [0.05, 0.04];
        state.pend = PairPending {
            wx: [1.2, 1.6], w_sum: 3.0, start_frame: 40, last_frame: 50, age: 10,
        };

        let mut fresh = SRanipalState::new();
        for (expected, previous) in fresh.eyes.iter_mut().zip(state.eyes.iter()) {
            expected.gaze_smooth = previous.gaze_smooth;
            expected.pupil_smooth = previous.pupil_smooth;
            expected.yoke_hold = previous.yoke_hold;
        }
        state.reset_eyelid_calibration();

        assert_eq!(format!("{:?}", state.eyes), format!("{:?}", fresh.eyes));
        assert_eq!(
            toml::to_string(&state.snapshot_all()).unwrap(),
            toml::to_string(&fresh.snapshot_all()).unwrap()
        );
        assert_eq!(format!("{:?}", state.pend), format!("{:?}", fresh.pend));
        assert_eq!(format!("{:?}", state.eyelid_live), format!("{:?}", fresh.eyelid_live));
        assert_eq!(state.wide_gate, 0.0);
        assert_eq!(state.gaze_eyelid_lift, [0.0; 2]);
        assert_eq!(state.session_baseline_offset, [0.0; 2]);
        assert!(!state.appearance_profile_active);
        assert_eq!(state.appearance_baseline_offset, [0.0; 2]);
        assert_eq!(state.appearance_wide_entry_shift, [0.0; 2]);
        assert_eq!(state.reseat_candidate, [0.0; 2]);
        assert_eq!(state.reseat_prev_raw, fresh.reseat_prev_raw);
        assert_eq!(state.reseat_stable_frames, 0);
        assert_eq!(state.response_manual_range, fresh.response_manual_range);
        assert_eq!(state.response_open_point_offset, fresh.response_open_point_offset);
        assert_eq!(state.response_closed_point_depth, fresh.response_closed_point_depth);
        assert_eq!(state.response_close_depth_scale, fresh.response_close_depth_scale);
        assert_eq!(state.response_curve_mid_output, fresh.response_curve_mid_output);
        assert_eq!(state.response_blink_close_ms, fresh.response_blink_close_ms);
        assert_eq!(state.response_snap_gate_open, fresh.response_snap_gate_open);
        assert!(!state.response_auto_reseat);
        assert!(!state.response_profile_installed);
    }

    #[test]
    fn full_eyelid_reset_preserves_tuning_and_gaze_pupil_filters() {
        let mut state = SRanipalState::new();
        state.tuning.alpha_open = 0.31;
        state.tuning.continuous_calib = false;
        state.tuning.gaze_yoke = false;
        state.tuning.brow_lr_sync = false;
        let tuning_before = toml::to_string(&state.tuning).unwrap();
        state.eyes[0].gaze_smooth = [0.21, -0.43, -0.82];
        state.eyes[1].gaze_smooth = [-0.38, 0.19, -0.77];
        state.eyes[0].pupil_smooth = 3.27;
        state.eyes[1].pupil_smooth = 4.61;
        state.eyes[0].yoke_hold = true;
        state.eyes[1].yoke_hold = false;
        let before = state.eyes;

        state.reset_eyelid_calibration();

        assert_eq!(toml::to_string(&state.tuning).unwrap(), tuning_before);
        for (current, previous) in state.eyes.iter().zip(before.iter()) {
            assert_eq!(current.gaze_smooth.map(f32::to_bits), previous.gaze_smooth.map(f32::to_bits));
            assert_eq!(current.pupil_smooth.to_bits(), previous.pupil_smooth.to_bits());
            assert_eq!(current.yoke_hold, previous.yoke_hold);
        }
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
        assert!(
            last_wide > 0.3,
            "wide collapsed under burst spam: {last_wide}"
        );
        let drift = (st.baseline(Eye::Left) - b0).abs();
        assert!(drift < 0.02, "baseline drifted {drift} under burst spam");
    }

    #[test]
    fn recenter_during_stuck_episode_recovers_bounded() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.53,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
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
        assert!(
            at <= 1800,
            "recovery after recenter took {at} frames (> 15s)"
        );
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
        assert_eq!(
            min_right, 0.0,
            "lagging right eye must be yoked to 0, got {min_right}"
        );
        step(&mut st, 0.62, 0.56);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            out = step(&mut st, 0.62, 0.62);
        }
        assert!(
            out[0].openness > 0.8 && out[1].openness > 0.8,
            "both eyes reopen after the yoked blink ({} / {})",
            out[0].openness,
            out[1].openness
        );
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
        assert!(
            (after - before).abs() < 0.01,
            "first close must only stage a candidate ({before} -> {after})"
        );
        slow_blink(&mut st, 0.62, 0.36, 60);
        let after2 = st.closed_ref(Eye::Left);
        assert!(
            after2 < before - 0.04,
            "second slow blink must confirm the calibrated depth ({after} -> {after2})"
        );
        feed(&mut st, 0.62, 3600);
        let held = st.closed_ref(Eye::Left);
        assert!(
            (held - after2).abs() < 0.006,
            "learned depth must not decay away ({after2} -> {held})"
        );
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
        assert!(
            (after - before).abs() < 0.01,
            "fast blinks must not move closed_ref ({before} -> {after})"
        );
    }

    #[test]
    fn resting_closed_eyes_do_not_teach_blink_depth() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        slow_blink(&mut st, 0.62, 0.34, 500);
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - before).abs() < 0.01,
            "resting closed must not teach depth ({before} -> {after})"
        );
    }

    #[test]
    fn blink_depth_survives_recenter_and_restore() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.36, 60);
        slow_blink(&mut st, 0.62, 0.36, 60);
        let learned = st.closed_ref(Eye::Left);
        assert!(
            learned < 0.415,
            "depth learned before the round-trip, got {learned}"
        );
        st.recenter();
        feed(&mut st, 0.62, 300);
        let after_recenter = st.closed_ref(Eye::Left);
        assert!(
            (after_recenter - learned).abs() < 0.01,
            "recenter must keep the learned depth ({learned} -> {after_recenter})"
        );
        let store = st.snapshot_all();
        assert!(
            store.left.blink_depth > 0.21,
            "depth persisted: {}",
            store.left.blink_depth
        );
        let mut st2 = SRanipalState::new();
        st2.restore_all(&store);
        feed(&mut st2, 0.62, 300);
        let restored = st2.closed_ref(Eye::Left);
        assert!(
            (restored - learned).abs() < 0.015,
            "restore must keep the learned depth ({learned} -> {restored})"
        );
    }

    #[test]
    fn mature_baseline_and_close_floor_do_not_follow_a_long_droop() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60);
        slow_blink(&mut st, 0.62, 0.40, 60);
        let baseline = st.baseline(Eye::Left);
        let learned = st.closed_ref(Eye::Left);
        assert!(learned < 0.44, "depth learned, got closed_ref {learned}");
        feed(&mut st, 0.55, 24000);
        assert!(
            (st.baseline(Eye::Left) - baseline).abs() < 1e-6,
            "mature baseline drifted ({} -> {})",
            baseline,
            st.baseline(Eye::Left)
        );
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - learned).abs() < 0.005,
            "close floor must not follow a sunk baseline ({learned} -> {after})"
        );
        let mut raw = 0.55f32;
        while raw > 0.40 {
            raw -= 0.02;
            feed(&mut st, raw.max(0.40), 1);
        }
        let out = feed(&mut st, 0.40, 30);
        assert!(
            out[0].openness < 0.05,
            "full close must still reach ~0 after the sink, got {}",
            out[0].openness
        );
    }

    #[test]
    fn long_stare_then_squint_does_not_erode_the_learned_depth() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.38, 60);
        slow_blink(&mut st, 0.62, 0.38, 60);
        let learned = st.closed_ref(Eye::Left);
        feed(&mut st, 0.62, 4800);
        slow_blink(&mut st, 0.62, 0.50, 80);
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - learned).abs() < 0.01,
            "a post-stare squint must not lift the close floor ({learned} -> {after})"
        );
    }

    #[test]
    fn field_partial_close_sequence_cannot_ratchet_floor_or_depth() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.7186, 400);
        for eye in &mut st.eyes {
            eye.baseline = 0.7186;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            eye.upper = eye.baseline + UPPER_OFFSET;
            eye.closed_ref = 0.3755;
            eye.blink_depth = eye.baseline - eye.closed_ref;
            eye.learned_once = true;
            eye.reach_env = 0.515;
            eye.blink_candidate = None;
            eye.blink_candidate_count = 0;
        }
        let ref0 = st.closed_ref(Eye::Right);
        let depth0 = st.eyes[1].blink_depth;
        let anchor0 = st.eyes[1].mid_anchor;
        for bottom in [0.515, 0.528, 0.541] {
            slow_blink(&mut st, 0.7186, bottom, 40);
            feed(&mut st, 0.7186, 120);
        }
        assert_eq!(st.closed_ref(Eye::Right), ref0);
        assert_eq!(st.eyes[1].blink_depth, depth0);
        assert_eq!(st.eyes[1].mid_anchor, anchor0);
    }

    #[test]
    fn repeated_long_consistent_squints_do_not_form_a_floor_candidate() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for eye in &mut st.eyes {
            eye.closed_ref = 0.40;
            eye.blink_depth = 0.22;
            eye.learned_once = true;
            eye.reach_env = 0.50;
        }
        let before = st.closed_ref(Eye::Left);
        for _ in 0..3 {
            slow_blink(&mut st, 0.62, 0.50, 300);
            feed(&mut st, 0.62, 120);
        }
        assert_eq!(st.closed_ref(Eye::Left), before);
        assert!(st.eyes[0].blink_candidate.is_none());
    }

    #[test]
    fn learned_deeper_floor_requires_a_pair_and_is_capped() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60);
        slow_blink(&mut st, 0.62, 0.40, 60);
        feed(&mut st, 0.62, FLOOR_COMMIT_MIN_FRAMES as usize);
        let before = st.closed_ref(Eye::Left);
        let depth_before = st.eyes[0].blink_depth;

        slow_blink(&mut st, 0.62, 0.34, 60);
        assert_eq!(st.closed_ref(Eye::Left), before, "one episode only stages");
        assert_eq!(st.eyes[0].blink_depth, depth_before);

        slow_blink(&mut st, 0.62, 0.34, 60);
        let after = st.closed_ref(Eye::Left);
        assert!((after - (before - FLOOR_DEEP_COMMIT_CAP)).abs() < 1e-4);
        assert!((st.eyes[0].blink_depth - (st.baseline(Eye::Left) - after)).abs() < 1e-4);
    }

    #[test]
    fn learned_floor_candidate_expires_without_a_confirming_close() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60);
        slow_blink(&mut st, 0.62, 0.40, 60);
        feed(&mut st, 0.62, FLOOR_COMMIT_MIN_FRAMES as usize);
        let before = st.closed_ref(Eye::Left);

        slow_blink(&mut st, 0.62, 0.34, 60);
        feed(&mut st, 0.62, FLOOR_CANDIDATE_MAX_AGE as usize + 1);
        slow_blink(&mut st, 0.62, 0.34, 60);
        assert_eq!(st.closed_ref(Eye::Left), before);
    }

    #[test]
    fn learned_floor_travel_budget_survives_recenter() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60);
        slow_blink(&mut st, 0.62, 0.40, 60);
        let before = st.closed_ref(Eye::Left);

        for _ in 0..5 {
            feed(&mut st, 0.62, FLOOR_COMMIT_MIN_FRAMES as usize);
            slow_blink(&mut st, 0.62, 0.25, 60);
            slow_blink(&mut st, 0.62, 0.25, 60);
        }
        let travel = before - st.closed_ref(Eye::Left);
        assert!(
            (travel - FLOOR_TRAVEL_BUDGET).abs() < 1e-4,
            "travel={travel}"
        );
        let at_budget = st.closed_ref(Eye::Left);
        st.recenter();
        feed(&mut st, 0.62, 300 + FLOOR_COMMIT_MIN_FRAMES as usize);
        slow_blink(&mut st, 0.62, 0.20, 60);
        slow_blink(&mut st, 0.62, 0.20, 60);
        assert!(
            (st.closed_ref(Eye::Left) - at_budget).abs() < 1e-4,
            "Recenter reset the endpoint travel budget"
        );
    }

    #[test]
    fn stale_too_deep_floor_has_a_capped_three_close_escape() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for eye in &mut st.eyes {
            eye.closed_ref = 0.30;
            eye.blink_depth = 0.32;
            eye.learned_once = true;
            eye.reach_env = 0.47;
            eye.floor_last_commit_frame = 0;
        }
        let before = st.closed_ref(Eye::Left);
        slow_blink(&mut st, 0.62, 0.47, 60);
        slow_blink(&mut st, 0.62, 0.47, 60);
        assert_eq!(
            st.closed_ref(Eye::Left),
            before,
            "two shallow closes only stage"
        );
        slow_blink(&mut st, 0.62, 0.47, 60);
        let after = st.closed_ref(Eye::Left);
        assert!((after - (before + FLOOR_SHALLOW_COMMIT_CAP)).abs() < 1e-4);
        assert!((st.eyes[0].blink_depth - (st.baseline(Eye::Left) - after)).abs() < 1e-4);
    }

    #[test]
    fn shallow_escape_converges_through_the_old_dead_band() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for eye in &mut st.eyes {
            eye.closed_ref = 0.425;
            eye.blink_depth = eye.baseline - eye.closed_ref;
            eye.learned_once = true;
            eye.reach_env = 0.47;
            eye.floor_last_commit_frame = 0;
        }
        for _ in 0..2 {
            for _ in 0..FLOOR_SHALLOW_CONFIRM_N {
                slow_blink(&mut st, 0.62, 0.47, 60);
            }
            feed(&mut st, 0.62, FLOOR_COMMIT_MIN_FRAMES as usize);
        }
        let residual = 0.47 - st.closed_ref(Eye::Left);
        assert!(
            residual <= FLOOR_CURVE_NEAR + 1e-4,
            "shallow correction stalled with residual {residual}"
        );
    }

    #[test]
    fn coupling_suppresses_endpoint_commits() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60);
        slow_blink(&mut st, 0.62, 0.40, 60);
        feed(&mut st, 0.62, FLOOR_COMMIT_MIN_FRAMES as usize);
        let before_ref = st.closed_ref(Eye::Left);
        let before_depth = st.eyes[0].blink_depth;
        st.tuning.couple_eyes = true;
        slow_blink(&mut st, 0.62, 0.34, 60);
        slow_blink(&mut st, 0.62, 0.34, 60);
        assert_eq!(st.closed_ref(Eye::Left), before_ref);
        assert_eq!(st.eyes[0].blink_depth, before_depth);
    }

    #[test]
    fn mature_floor_is_not_clamped_back_to_a_shifted_baseline() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.70, 400);
        for eye in &mut st.eyes {
            eye.learned_once = true;
            eye.closed_ref = 0.20;
            eye.blink_depth = 0.40;
        }
        feed(&mut st, 0.70, 1);
        assert_eq!(st.closed_ref(Eye::Left), 0.20);
        assert_eq!(st.closed_ref(Eye::Right), 0.20);
    }

    #[test]
    fn razor_thin_close_span_is_floored() {
        let mut st = SRanipalState::new();
        st.tuning.open_deadzone = 0.08;
        st.tuning.alpha_open = 1.0;
        st.tuning.alpha_close = 1.0;
        let snap = CalibSnapshot {
            baseline: 0.62,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.12,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
        let out = feed(&mut st, 0.62, 400);
        assert!(
            out[0].openness > 0.95,
            "relaxed reads open, got {}",
            out[0].openness
        );
        let blink_top = 0.62 - 0.12;
        let probe = blink_top + MIN_CLOSE_SPAN * 0.5;
        let mut raw = 0.62f32;
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        while raw > probe {
            raw -= 0.02;
            out = feed(&mut st, raw.max(probe), 1);
        }
        assert!(
            (out[0].openness - 0.5).abs() < 0.1,
            "floored slope: half-span reads ~0.5, got {}",
            out[0].openness
        );
        while raw > blink_top {
            raw -= 0.02;
            feed(&mut st, raw.max(blink_top), 1);
        }
        let out = feed(&mut st, blink_top, 20);
        assert!(
            out[0].openness < 0.05,
            "bottom reads 0, got {}",
            out[0].openness
        );
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
        assert!(
            gap_before > 0.10,
            "warp should produce a visible gap, got {gap_before}"
        );
        for _ in 0..5 {
            warped_slow_close_cycle(&mut st);
        }
        let (a_l, a_r) = (st.eyes[0].mid_anchor, st.eyes[1].mid_anchor);
        assert!(
            a_l > 0.52 && a_r < 0.48,
            "anchors must move oppositely (L {a_l} / R {a_r})"
        );
        assert!(
            (a_l + a_r - 1.0).abs() < 1e-3,
            "pair mean pinned to identity (L {a_l} / R {a_r})"
        );
        let gap_after = mid_close_gap(&mut st);
        assert!(
            gap_after < 0.06 && gap_after < 0.5 * gap_before,
            "mid-close gap must shrink ({gap_before} -> {gap_after})"
        );
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
            step_pair(&mut st, raw.max(0.48), 0.62);
        }
        st.eyes[0].mid_anchor_staged = Some(0.62);
        for _ in 0..30 {
            step_pair(&mut st, 0.48, 0.62);
        }
        assert_eq!(st.eyes[0].mid_anchor, 0.5, "anchor swapped mid-close");
        assert!(st.eyes[0].mid_anchor_staged.is_some());
        while raw < 0.62 {
            raw += 0.01;
            step_pair(&mut st, raw.min(0.62), 0.62);
        }
        step_pair(&mut st, 0.62, 0.62);
        assert_eq!(
            st.eyes[0].mid_anchor, 0.62,
            "staged anchor must land at full open"
        );
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
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st2.restore(Eye::Left, bad);
        assert_eq!(
            st2.eyes[0].mid_anchor, 0.5,
            "NaN anchor must sanitize to identity"
        );
    }

    #[test]
    fn stale_partner_exit_does_not_deadlock_pairing() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.eyes[0].learned_once = true;
        st.eyes[1].learned_once = true;
        st.eyes[0].closed_ref = 0.36;
        st.eyes[1].closed_ref = 0.36;
        st.eyes[0].blink_depth = 0.26;
        st.eyes[1].blink_depth = 0.26;
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
        assert!(
            st.eyes[1].ep_clean_exit.is_some(),
            "stale right exit seeded"
        );
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
        assert!(
            a_l > 0.5 && a_r < 0.5,
            "offset gestures must still commit (L {a_l} / R {a_r})"
        );
    }

    #[test]
    fn low_raw_setup_baseline_learns() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.43, 300);
        let b = st.baseline(Eye::Left);
        assert!(
            (b - 0.43).abs() < 0.01,
            "cold start at raw 0.43 must learn, got {b}"
        );
        let snap = CalibSnapshot {
            baseline: 0.51,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.29,
            mid_anchor: 0.5,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        let mut st2 = SRanipalState::new();
        st2.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
        st2.recenter();
        feed(&mut st2, 0.43, 300);
        let b2 = st2.baseline(Eye::Right);
        assert!(
            (b2 - 0.43).abs() < 0.015,
            "recenter must re-anchor a stale-high baseline at raw 0.43, got {b2}"
        );
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
        assert!(
            (a_l - 0.5).abs() < 0.015 && (a_r - 0.5).abs() < 0.015,
            "asymmetric hold taught the anchors (L {a_l} / R {a_r})"
        );
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
        assert!(
            st.eyes[0].ep_clean_exit.is_none() && st.eyes[1].ep_clean_exit.is_none(),
            "an evaluated episode pair must consume its exits"
        );
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
        assert!(
            st.eyes[0].mid_anchor_staged.is_none() && st.eyes[1].mid_anchor_staged.is_none(),
            "post-episode samples must never commit"
        );
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
        assert!(
            lo > 0.15,
            "gaze-invalid mid-close noise forced a close (lo {lo})"
        );
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
        assert!(
            lo > 0.02,
            "mid-close noise must not excurse to closed (lo {lo})"
        );
        assert!(
            hi < 0.98,
            "mid-close noise must not excurse to open (hi {hi})"
        );
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
        assert_eq!(
            min_open, 0.0,
            "a -0.05/update blink must commit (the old code missed it)"
        );
    }

    #[test]
    fn latch_holds_through_bottom_tremor() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let seq = [
            0.50f32, 0.40, 0.395, 0.401, 0.397, 0.403, 0.398, 0.43, 0.50, 0.62,
        ];
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
            assert_eq!(
                o, 0.0,
                "latch must hold through bottom tremor (frame {k} read {o})"
            );
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
                assert!(
                    o - prev <= 1.0 / 24.0 + 1e-4,
                    "reopen must be slew-limited (step {})",
                    o - prev
                );
                frames_to_recover += 1;
            }
            prev = o;
            if o >= 0.9 {
                break;
            }
        }
        assert!(
            frames_to_recover >= 20,
            "200ms recovery must take >= ~21 frames, took {frames_to_recover}"
        );
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
        assert!(
            fast_frames < 15,
            "default recovery stays fast, took {fast_frames}"
        );
    }

    #[test]
    fn confirmed_fast_blink_reaches_zero_before_reopening_with_close_slew() {
        let mut st = SRanipalState::new();
        st.tuning.blink_close_ms = 100.0;
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let mut samples = Vec::new();
        for raw in [0.50f32, 0.38, 0.38] {
            for _ in 0..2 {
                let ml = [[1.0, raw, 0.0, 0.0, 0.0]; 2];
                samples.push(st.process_frame(ml, &g, true)[0].openness);
            }
        }
        for _ in 0..60 {
            let ml = [[1.0, 0.62, 0.0, 0.0, 0.0]; 2];
            samples.push(st.process_frame(ml, &g, true)[0].openness);
        }
        let first_zero = samples
            .iter()
            .position(|value| *value <= 0.001)
            .expect("the close slew must still reach a visible zero");
        let zero_run = samples[first_zero..]
            .iter()
            .take_while(|value| **value <= 0.001)
            .count();
        assert!(
            zero_run >= BlinkTimingProfile::default().minimum_frames() as usize,
            "visible bottom lasted only {zero_run} frames"
        );
        assert!(
            samples
                .iter()
                .skip(first_zero + zero_run)
                .any(|value| *value > 0.5),
            "the eye must reopen after the guaranteed bottom"
        );
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
        assert_eq!(
            hold_max, 0.0,
            "a held deep close must stay 0 (no bounce), got {hold_max}"
        );
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for r in [0.50f32, 0.56, 0.62, 0.62, 0.62, 0.62, 0.62, 0.62] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                out = st.process_frame(ml, &g, true);
            }
        }
        assert!(
            out[0].openness > 0.7,
            "reopens after the hold, got {}",
            out[0].openness
        );
    }

    #[test]
    fn hard_close_overshoot_recovery_stays_closed() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let seq = [
            0.50f32, 0.38, 0.36, 0.40, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
        ];
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
        assert_eq!(
            after_commit_max, 0.0,
            "overshoot recovery (+0.08, still in the closed zone) must not release"
        );
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
        assert!(
            committed && released,
            "must commit then release (c={committed} r={released})"
        );
        assert!(
            !re_latched,
            "stale evidence must not re-latch after the release"
        );
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
        assert!(
            min_open > 0.3,
            "bilateral squint must stay on the ramp, got {min_open}"
        );
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
        assert!(
            both_zero,
            "medium-speed bilateral blink must close BOTH eyes"
        );
        step(&mut st, 0.55, 0.56);
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            out = step(&mut st, 0.62, 0.62);
        }
        assert!(
            out[0].openness > 0.8 && out[1].openness > 0.8,
            "both eyes reopen after the anchored blink ({} / {})",
            out[0].openness,
            out[1].openness
        );
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
        assert!(
            min_open > 0.05,
            "a slow deep droop must stay on the ramp (never snap to 0), got {min_open}"
        );
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
            assert!(
                out[1].openness > 0.5,
                "open eye must NOT be yoked during a wink, got {}",
                out[1].openness
            );
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
        assert!(
            last[0].wide > 0.3,
            "wide collapsed in hold #2: {}",
            last[0].wide
        );
        let drift = (st.baseline(Eye::Left) - b0).abs();
        assert!(
            drift < 0.02,
            "baseline drifted {drift} across breath-separated holds"
        );
    }

    #[test]
    fn restore_unlearned_snapshot_keeps_cold_start() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 0,
            frame_count: 0,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
        feed(&mut st, 0.66, 300);
        let b = st.baseline(Eye::Left);
        assert!(
            (b - 0.66).abs() < 0.01,
            "cold start after idle-session restore stuck at {b}"
        );
        let out = feed(&mut st, 0.66, 120);
        assert!(
            out[0].wide < 0.01,
            "spurious wide after cold-start restore: {}",
            out[0].wide
        );
    }

    #[test]
    fn stale_high_restore_stays_open_and_heals() {
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.72,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
        let out = feed(&mut st, 0.66, 240);
        assert!(
            out[0].openness > 0.9,
            "stale-high restore must read open, got {}",
            out[0].openness
        );
        assert!(
            !out[0].blink,
            "relaxed eye under a stale-high baseline is not a blink"
        );
    }


    fn slow_close_min(st: &mut SRanipalState, bottom: f32, dwell: usize) -> f32 {
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let frame = |st: &mut SRanipalState, r: f32| {
            st.process_frame([[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]], &g, true)
        };
        let mut raw = 0.62f32;
        while raw > bottom {
            raw -= 0.02;
            frame(st, raw.max(bottom));
        }
        let mut min_o = 1.0f32;
        for _ in 0..dwell {
            let out = frame(st, bottom);
            min_o = min_o.min(out[0].openness).min(out[1].openness);
        }
        while raw < 0.62 {
            raw += 0.02;
            frame(st, raw.min(0.62));
        }
        for _ in 0..40 {
            frame(st, 0.62);
        }
        min_o
    }

    fn probe_hold_open(st: &mut SRanipalState, raw_hold: f32) -> EyeResult {
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let frame = |st: &mut SRanipalState, r: f32| {
            st.process_frame([[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]], &g, true)
        };
        let mut raw = 0.62f32;
        while raw > raw_hold {
            raw -= 0.02;
            frame(st, raw.max(raw_hold));
        }
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            out = frame(st, raw_hold);
        }
        out[0]
    }

    #[test]
    fn shallow_eye_reaches_zero_after_two_close_confirmation() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let first = slow_close_min(&mut st, 0.47, 40);
        let second = slow_close_min(&mut st, 0.47, 40);
        let third = slow_close_min(&mut st, 0.47, 40);
        assert!(
            first > 0.05,
            "first shallow close bottoms partly-open, got {first}"
        );
        assert!(
            second > 0.05,
            "confirmation must wait for the true episode bottom"
        );
        assert_eq!(
            third, 0.0,
            "shallow eye must fully close after confirmation, got {third}"
        );
    }

    #[test]
    fn cold_start_squint_does_not_become_the_blink_floor() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before_depth = st.eyes[0].blink_depth;
        slow_close_min(&mut st, 0.50, 40);
        assert!(
            !st.eyes[0].learned_once,
            "one ambiguous squint must stay unconfirmed"
        );
        assert!((st.eyes[0].blink_depth - before_depth).abs() < 1e-4);

        slow_close_min(&mut st, 0.47, 40);
        assert!(
            !st.eyes[0].learned_once,
            "a different-depth close replaces the candidate"
        );
        slow_close_min(&mut st, 0.47, 40);
        assert!(st.eyes[0].learned_once);
        let confirmed = slow_close_min(&mut st, 0.47, 40);
        assert_eq!(confirmed, 0.0);
    }

    #[test]
    fn deeper_blink_passing_candidate_does_not_confirm_mid_descent() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_close_min(&mut st, 0.50, 40);

        let mut raw = 0.62f32;
        while raw > 0.50 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.50), raw.max(0.50));
        }
        for _ in 0..40 {
            step_pair(&mut st, 0.50, 0.50);
        }
        assert!(
            !st.eyes[0].learned_once,
            "must not confirm before the episode bottom is known"
        );

        while raw > 0.44 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.44), raw.max(0.44));
        }
        for _ in 0..40 {
            step_pair(&mut st, 0.44, 0.44);
        }
        while raw < 0.62 {
            raw += 0.01;
            step_pair(&mut st, raw.min(0.62), raw.min(0.62));
        }
        feed(&mut st, 0.62, 40);
        assert!(
            !st.eyes[0].learned_once,
            "deeper bottom must replace the shallow candidate"
        );
    }

    #[test]
    fn normal_eye_no_early_close_and_mild_squint_reads_open() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..3 {
            slow_close_min(&mut st, 0.40, 40);
        }
        let cr = st.closed_ref(Eye::Left);
        let b = st.baseline(Eye::Left);
        assert!(
            cr <= b - 0.19 && cr >= b - 0.23,
            "closed_ref must stay ~baseline-0.20..0.22, got {cr} (baseline {b})"
        );
        let out = probe_hold_open(&mut st, 0.46);
        assert!(
            out.openness > 0.25 && !out.blink,
            "a mild squint must read clearly open, got {}",
            out.openness
        );
    }

    #[test]
    fn squint_heavy_does_not_pull_floor_up() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..3 {
            slow_close_min(&mut st, 0.40, 40);
        }
        let cr0 = st.closed_ref(Eye::Left);
        for r in 0..12 {
            slow_close_min(&mut st, 0.50, 40);
            if r % 3 == 2 {
                slow_close_min(&mut st, 0.40, 40);
            }
        }
        let cr1 = st.closed_ref(Eye::Left);
        assert!(
            cr1 <= cr0 + 0.03,
            "squints must not pull the floor up (closed_ref {cr0} -> {cr1})"
        );
        let out = probe_hold_open(&mut st, 0.52);
        assert!(
            out.openness > 0.3,
            "narrowed-relaxed raw must read open, got {}",
            out.openness
        );
    }

    #[test]
    fn dropout_burst_does_not_disable_shallow_close() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_close_min(&mut st, 0.47, 40);
        slow_close_min(&mut st, 0.47, 40);
        let learned = slow_close_min(&mut st, 0.47, 40);
        assert_eq!(
            learned, 0.0,
            "shallow floor learned before the dropout, got {learned}"
        );
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        for _ in 0..60 {
            st.process_frame(
                [[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        for _ in 0..60 {
            st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        let after = slow_close_min(&mut st, 0.47, 40);
        assert_eq!(
            after, 0.0,
            "dropout must not disable the shallow-eye close, got {after}"
        );
    }

    #[test]
    fn explicit_endpoint_lock_survives_blinks_recenter_and_serde() {
        let mut state = SRanipalState::new();
        feed(&mut state, 0.62, 400);
        let mature_weight = state.snapshot(Eye::Left).baseline_n;
        state.apply_explicit_endpoint(Eye::Left, 0.60, 0.16, 1234);
        let applied = state.snapshot(Eye::Left);
        assert!(applied.endpoint_locked);
        assert_eq!(applied.endpoint_calibrated_unix, 1234);
        assert_eq!(applied.baseline_n, mature_weight);

        for _ in 0..4 {
            slow_close_min(&mut state, 0.34, 40);
            slow_close_min(&mut state, 0.48, 40);
        }
        assert!((state.snapshot(Eye::Left).blink_depth - 0.16).abs() < 1e-5);

        state.recenter();
        feed(&mut state, 0.64, 180);
        let recentered = state.snapshot(Eye::Left);
        assert!(recentered.endpoint_locked);
        assert!((recentered.blink_depth - 0.16).abs() < 1e-5);
        assert!((state.closed_ref(Eye::Left) - (state.baseline(Eye::Left) - 0.16)).abs() < 1e-4);

        let store = state.snapshot_all();
        let encoded = toml::to_string(&store).unwrap();
        let decoded: CalibStore = toml::from_str(&encoded).unwrap();
        assert!(decoded.left.endpoint_locked);
        assert_eq!(decoded.left.endpoint_calibrated_unix, 1234);
    }

    #[test]
    fn removing_explicit_endpoint_restarts_adaptive_defaults_without_touching_other_eye() {
        let mut state = SRanipalState::new();
        state.apply_explicit_endpoint(Eye::Left, 0.72, 0.11, 1234);
        state.apply_explicit_endpoint(Eye::Right, 0.57, 0.18, 5678);
        let right_before = state.snapshot(Eye::Right);
        let left_mid_anchor = state.snapshot(Eye::Left).mid_anchor;

        state.remove_explicit_endpoint(Eye::Left);

        let left = state.snapshot(Eye::Left);
        assert!(!left.endpoint_locked);
        assert_eq!(left.endpoint_calibrated_unix, 0);
        assert_eq!(left.baseline.to_bits(), DEFAULT_BASELINE.to_bits());
        assert_eq!(left.baseline_n, 0);
        assert_eq!(left.blink_depth.to_bits(), BLINK_OFFSET.to_bits());
        assert!(!left.learned_once);
        assert_eq!(left.mid_anchor.to_bits(), left_mid_anchor.to_bits());

        let right_after = state.snapshot(Eye::Right);
        assert_eq!(
            right_after.baseline.to_bits(),
            right_before.baseline.to_bits()
        );
        assert_eq!(
            right_after.blink_depth.to_bits(),
            right_before.blink_depth.to_bits()
        );
        assert!(right_after.endpoint_locked);
        assert_eq!(right_after.endpoint_calibrated_unix, 5678);
    }

    #[test]
    fn old_calibration_toml_defaults_to_adaptive_endpoints() {
        let text = r#"
[left]
baseline = 0.6
baseline_n = 100
frame_count = 500
blink_depth = 0.2
mid_anchor = 0.5
learned_once = true

[right]
baseline = 0.59
baseline_n = 100
frame_count = 500
blink_depth = 0.19
mid_anchor = 0.5
learned_once = true
"#;
        let store: CalibStore = toml::from_str(text).unwrap();
        assert!(!store.left.endpoint_locked);
        assert!(!store.right.endpoint_locked);
        assert_eq!(store.left.endpoint_calibrated_unix, 0);
    }

    #[test]
    fn gaze_eyelid_lift_restores_open_but_never_lifts_closed() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        let mut profile = GazeEyelidProfile::default();
        for eye in &mut profile.eyes {
            eye.enabled = true;
            eye.coefficients = [0.0, 0.0, 0.0, 0.08, 0.0];
            eye.max_lift = 0.09;
        }
        let angle = 20f32.to_radians();
        let sample = crate::core::types::EyeSample {
            gaze: [angle.sin(), 0.0, angle.cos()],
            gaze_valid: true,
            gaze_reported: true,
            ..crate::core::types::EyeSample::default()
        };
        let gaze = GazeSample {
            left: sample,
            right: sample,
            ..GazeSample::default()
        };
        let side_open = [[1.0, 0.53, 0.0, 0.0, 0.0]; 2];
        let mut open = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..40 {
            open = state.process_frame_with_gaze_eyelid(side_open, &gaze, true, &profile);
        }
        assert!(
            open[0].openness > 0.92,
            "repeatable side gaze should return close to relaxed open: {}",
            open[0].openness
        );

        let closed = [[1.0, 0.35, 0.0, 0.0, 0.0]; 2];
        let mut shut = open;
        for _ in 0..8 {
            shut = state.process_frame_with_gaze_eyelid(closed, &gaze, true, &profile);
        }
        assert_eq!(shut[0].openness, 0.0);
        assert_eq!(shut[1].openness, 0.0);
    }

    #[test]
    fn held_wink_profile_closes_only_the_instructed_eye() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.blink_close_ms = 0.0;
        let mut wink = WinkProfile::default();
        wink.eyes[0].enabled = true;
        wink.eyes[0].floor_enabled = true;
        wink.eyes[0].wink_depth = 0.15;
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();

        for _ in 0..120 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.0, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for step in 0..=15 {
            let left = 0.60 - 0.01 * step as f32;
            state.process_frame_with_profiles(
                [[1.0, left, 0.0, 0.0, 0.0], [1.0, 0.60, 0.0, 0.0, 0.0]],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        let mut result = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..12 {
            result = state.process_frame_with_profiles(
                [[1.0, 0.45, 0.0, 0.0, 0.0], [1.0, 0.60, 0.0, 0.0, 0.0]],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        assert!(state.eyes[0].wink_active);
        assert!(
            result[0].openness <= 0.02,
            "wink output {} baseline {} closed_ref {} raw {}",
            result[0].openness,
            state.baseline(Eye::Left),
            state.closed_ref(Eye::Left),
            state.eyes[0].prev_raw,
        );
        assert!(result[1].openness > 0.95);
        assert!(!state.eyes[1].wink_active);
    }

    #[test]
    fn wink_profile_does_not_reclassify_a_bilateral_slow_close() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.blink_close_ms = 0.0;
        let mut wink = WinkProfile::default();
        for eye in &mut wink.eyes {
            eye.enabled = true;
            eye.wink_depth = 0.15;
        }
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();
        let mut result = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..120 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.0, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for step in 0..=15 {
            let raw = 0.60 - 0.01 * step as f32;
            result = state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.0, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        assert!(!state.eyes[0].wink_active);
        assert!(!state.eyes[1].wink_active);
        assert!(result[0].openness > 0.20);
        assert!((result[0].openness - result[1].openness).abs() < 1e-5);
    }

    #[test]
    fn calibrated_unilateral_squeeze_accelerates_wink_entry() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.blink_close_ms = 0.0;
        let mut wink = WinkProfile::default();
        wink.eyes[0].enabled = true;
        wink.eyes[0].wink_depth = 0.15;
        wink.eyes[0].squeeze_enabled = true;
        wink.eyes[0].squeeze_enter_delta = 0.12;
        wink.eyes[0].squeeze_release_delta = 0.06;
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();

        for _ in 0..120 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for _ in 0..3 {
            state.process_frame_with_profiles(
                [[1.0, 0.50, 0.0, 0.56, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        assert!(
            state.eyes[0].wink_active,
            "repeatable unilateral squeeze should corroborate the shallow wink onset"
        );
        assert!(!state.eyes[1].wink_active);
    }

    #[test]
    fn signature_only_wink_profile_is_openness_bit_identical_to_no_profile() {
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        let mut plain = SRanipalState::new();
        let mut signature = SRanipalState::new();
        for state in [&mut plain, &mut signature] {
            state.restore(Eye::Left, endpoint);
            state.restore(Eye::Right, endpoint);
            state.tuning.alpha_close = 1.0;
            state.tuning.alpha_open = 1.0;
            state.tuning.smoothing = 0.0;
            state.tuning.blink_close_ms = 0.0;
            state.eyes[0].squeeze_floor = 0.12;
            state.eyes[1].squeeze_floor = 0.12;
        }

        let no_wink = WinkProfile::default();
        let mut signature_only = WinkProfile::default();
        signature_only.eyes[0].enabled = true;
        signature_only.eyes[0].floor_enabled = false;
        signature_only.eyes[0].wink_depth = 0.15;
        signature_only.eyes[0].squeeze_enabled = true;
        signature_only.eyes[0].squeeze_enter_delta = 0.06;
        signature_only.eyes[0].squeeze_release_delta = 0.03;
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();

        for _ in 0..16 {
            let ml = [[1.0, 0.60, 0.0, 0.12, 0.0]; 2];
            let a = plain.process_frame_with_profiles(ml, &gaze, true, &no_gaze_profile, &no_wink);
            let b = signature.process_frame_with_profiles(
                ml,
                &gaze,
                true,
                &no_gaze_profile,
                &signature_only,
            );
            for eye in 0..2 {
                assert_eq!(a[eye].openness.to_bits(), b[eye].openness.to_bits());
            }
        }

        for raw in [0.56f32, 0.53, 0.50, 0.48, 0.47, 0.47, 0.47, 0.47] {
            let ml = [[1.0, raw, 0.0, 0.20, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]];
            let a = plain.process_frame_with_profiles(ml, &gaze, true, &no_gaze_profile, &no_wink);
            let b = signature.process_frame_with_profiles(
                ml,
                &gaze,
                true,
                &no_gaze_profile,
                &signature_only,
            );
            for eye in 0..2 {
                assert_eq!(
                    a[eye].openness.to_bits(),
                    b[eye].openness.to_bits(),
                    "signature-only profile changed eye {eye} openness at raw {raw}"
                );
            }
        }
        assert!(
            signature.eyes[0].wink_active,
            "signature-only profiles must still run wink classification"
        );
        assert!(!plain.eyes[0].wink_active);
        assert_eq!(
            signature.eyes[0].reach_env.to_bits(),
            0.60f32.to_bits(),
            "validated low squeeze threshold must protect the reach learner"
        );
        assert_ne!(
            plain.eyes[0].reach_env.to_bits(),
            0.60f32.to_bits(),
            "control sequence should exercise the otherwise-unprotected reach learner"
        );
        assert_eq!(
            signature.eyes[0].blink_depth.to_bits(),
            0.25f32.to_bits(),
            "signature-only protection must not rewrite the ordinary endpoint"
        );
    }

    #[test]
    fn two_frame_asymmetric_lead_into_bilateral_squeezed_close_is_not_a_wink() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.blink_close_ms = 0.0;
        for eye in &mut state.eyes {
            eye.squeeze_floor = 0.12;
        }

        let mut wink = WinkProfile::default();
        for eye in &mut wink.eyes {
            eye.enabled = true;
            eye.floor_enabled = true;
            eye.wink_depth = 0.15;
            eye.squeeze_enabled = true;
            eye.squeeze_enter_delta = 0.12;
            eye.squeeze_release_delta = 0.06;
        }
        let gaze = native_gaze(true, true);
        let no_gaze_profile = GazeEyelidProfile::default();
        for _ in 0..16 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }

        for raw in [0.54f32, 0.50] {
            state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.30, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
            assert!(!state.eyes[0].wink_active);
            assert!(!state.eyes[1].wink_active);
        }

        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for raw in [0.48f32, 0.44, 0.40, 0.36, 0.36, 0.36, 0.36] {
            out = state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.30, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
            assert!(!state.eyes[0].wink_active);
            assert!(!state.eyes[1].wink_active);
            assert_eq!(
                out[0].openness.to_bits(),
                out[1].openness.to_bits(),
                "bilateral squeezed close became unilateral at raw {raw}"
            );
        }
        assert!(
            out[0].openness <= 0.08 && out[1].openness <= 0.08,
            "bilateral close did not settle near closed: L={} R={}",
            out[0].openness,
            out[1].openness
        );
    }

    #[test]
    fn ml_only_bilateral_squeeze_mirrors_fast_latch_without_creating_a_wink() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.blink_close_ms = 0.0;
        for eye in &mut state.eyes {
            eye.squeeze_floor = 0.12;
        }

        let mut wink = WinkProfile::default();
        for eye in &mut wink.eyes {
            eye.enabled = true;
            eye.floor_enabled = true;
            eye.wink_depth = 0.15;
            eye.squeeze_enabled = true;
            eye.squeeze_enter_delta = 0.12;
            eye.squeeze_release_delta = 0.06;
        }
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();
        for _ in 0..24 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for raw in [0.54f32, 0.50] {
            state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.30, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
            assert!(!state.eyes[0].wink_active);
            assert!(!state.eyes[1].wink_active);
        }

        let mut saw_fast_commit = false;
        for raw in [0.48f32, 0.44, 0.40, 0.36, 0.36, 0.36, 0.36] {
            let out = state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.30, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
            assert!(!state.eyes[0].wink_active);
            assert!(!state.eyes[1].wink_active);
            assert_eq!(out[0].openness.to_bits(), out[1].openness.to_bits());
            if state.eyes.iter().any(|eye| eye.fast_blink_frames > 0) {
                saw_fast_commit = true;
                assert!(
                    state.eyes.iter().all(|eye| eye.fast_blink_frames > 0),
                    "a bilateral squeezed close left one per-eye fast latch behind"
                );
            }
        }
        assert!(
            saw_fast_commit,
            "test sequence never exercised the fast latch"
        );
    }

    #[test]
    fn bilateral_squeeze_never_becomes_two_winks() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: true,
            endpoint_calibrated_unix: 1,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.alpha_close = 1.0;
        state.tuning.alpha_open = 1.0;
        state.tuning.smoothing = 0.0;
        let mut wink = WinkProfile::default();
        for eye in &mut wink.eyes {
            eye.enabled = true;
            eye.wink_depth = 0.15;
            eye.squeeze_enabled = true;
            eye.squeeze_enter_delta = 0.12;
            eye.squeeze_release_delta = 0.06;
        }
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let no_gaze_profile = GazeEyelidProfile::default();
        for _ in 0..120 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for step in 1..=10 {
            let raw = 0.60 - 0.01 * step as f32;
            state.process_frame_with_profiles(
                [[1.0, raw, 0.0, 0.56, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        for _ in 0..8 {
            state.process_frame_with_profiles(
                [[1.0, 0.50, 0.0, 0.56, 0.0]; 2],
                &gaze,
                true,
                &no_gaze_profile,
                &wink,
            );
        }
        assert!(!state.eyes[0].wink_active);
        assert!(!state.eyes[1].wink_active);
    }

    #[test]
    fn squeeze_correlated_wink_cannot_retrain_bilateral_endpoint() {
        let mut state = SRanipalState::new();
        let endpoint = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 100,
            frame_count: 1000,
            blink_depth: 0.25,
            mid_anchor: 0.5,
            learned_once: true,
            endpoint_locked: false,
            endpoint_calibrated_unix: 0,
        };
        state.restore(Eye::Left, endpoint);
        state.restore(Eye::Right, endpoint);
        state.tuning.continuous_calib = true;
        let gaze = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        let profile = GazeEyelidProfile::default();
        let wink = WinkProfile::default();
        for _ in 0..120 {
            state.process_frame_with_profiles(
                [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                &gaze,
                true,
                &profile,
                &wink,
            );
        }
        let before_depth = state.eyes[0].blink_depth;
        let before_ref = state.eyes[0].closed_ref;
        let before_reach = state.eyes[0].reach_env;

        for _ in 0..4 {
            for step in 1..=15 {
                let left = 0.60 - 0.01 * step as f32;
                state.process_frame_with_profiles(
                    [[1.0, left, 0.0, 0.56, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]],
                    &gaze,
                    true,
                    &profile,
                    &wink,
                );
            }
            for _ in 0..30 {
                state.process_frame_with_profiles(
                    [[1.0, 0.45, 0.0, 0.56, 0.0], [1.0, 0.60, 0.0, 0.12, 0.0]],
                    &gaze,
                    true,
                    &profile,
                    &wink,
                );
            }
            for _ in 0..35 {
                state.process_frame_with_profiles(
                    [[1.0, 0.60, 0.0, 0.12, 0.0]; 2],
                    &gaze,
                    true,
                    &profile,
                    &wink,
                );
            }
        }
        assert_eq!(state.eyes[0].blink_depth.to_bits(), before_depth.to_bits());
        assert_eq!(state.eyes[0].closed_ref.to_bits(), before_ref.to_bits());
        assert_eq!(state.eyes[0].reach_env.to_bits(), before_reach.to_bits());
    }

    #[test]
    fn response_curve_identity_and_fixed_endpoints() {
        for x in [0.0f32, 0.1, 0.49, 0.5, 0.73, 1.0] {
            assert_eq!(apply_response_curve(x, 0.5).to_bits(), x.to_bits());
        }
        for midpoint in [CURVE_MID_OUTPUT_MIN, CURVE_MID_OUTPUT_MAX] {
            assert_eq!(apply_response_curve(0.0, midpoint), 0.0);
            assert_eq!(apply_response_curve(0.5, midpoint), midpoint);
            assert_eq!(apply_response_curve(1.0, midpoint), 1.0);
        }
    }

    #[test]
    fn identity_response_profile_is_output_bit_identical() {
        let mut legacy = SRanipalState::new();
        let mut profiled = SRanipalState::new();
        profiled.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        let gaze = GazeSample::default();
        let sequence = [
            0.62f32, 0.62, 0.58, 0.52, 0.46, 0.39, 0.39, 0.48, 0.58, 0.62,
        ];
        for _ in 0..30 {
            for raw in sequence {
                let ml = [[1.0, raw, 0.0, 0.0, 0.0]; 2];
                let a = legacy.process_frame(ml, &gaze, true);
                let b = profiled.process_frame(ml, &gaze, true);
                for i in 0..2 {
                    assert_eq!(a[i].openness.to_bits(), b[i].openness.to_bits());
                    assert_eq!(a[i].blink, b[i].blink);
                }
            }
        }
    }

    fn manual_range_fixture(
        adaptive_bounds: bool,
        native_reported: bool,
    ) -> (SRanipalState, GazeSample) {
        let mut state = SRanipalState::new();
        state.tuning.continuous_calib = adaptive_bounds;
        state.tuning.alpha_open = 1.0;
        state.tuning.alpha_close = 1.0;
        state.tuning.smoothing = 0.0;
        state.tuning.adaptive_kalman = false;
        state.tuning.blink_reopen_ms = 0.0;
        state.tuning.blink_close_ms = 0.0;
        state.tuning.couple_eyes = false;
        state.tuning.gaze_yoke = false;
        state.set_eyelid_response_tuning(
            true, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        for eye in Eye::ALL {
            state.apply_explicit_endpoint(eye, 0.60, 0.40, 1);
        }
        let eye = crate::core::types::EyeSample {
            gaze: [0.0, 0.0, -1.0],
            gaze_valid: true,
            gaze_reported: true,
            openness: 1.0,
            openness_valid: true,
            openness_reported: native_reported,
            ..Default::default()
        };
        let gaze = GazeSample {
            left: eye,
            right: eye,
            ..Default::default()
        };
        for _ in 0..300 {
            state.process_frame([[1.0, 0.60, 0.0, 0.0, 0.0]; 2], &gaze, true);
        }
        (state, gaze)
    }

    #[test]
    fn manual_range_slow_ramps_are_continuous_with_adaptive_bounds_on_or_off() {
        for native_reported in [false, true] {
            let (off, gaze) = manual_range_fixture(false, native_reported);
            let (on, _) = manual_range_fixture(true, native_reported);
            let mut states = [off, on];
            let timing = BlinkTimingProfile {
                enabled: false,
                ..Default::default()
            };
            let mut previous = [1.0f32; 2];
            let mut max_step = [0.0f32; 2];
            for closing in [true, false] {
                for step in 1..=410 {
                    let raw = if closing {
                        (600 - step) as f32 / 1000.0
                    } else {
                        (190 + step) as f32 / 1000.0
                    };
                    let mut output = [0.0; 2];
                    for (index, state) in states.iter_mut().enumerate() {
                        let frame = state.process_frame_with_all_profiles(
                            [[1.0, raw, 0.0, 0.0, 0.0]; 2],
                            &gaze,
                            true,
                            &GazeEyelidProfile::default(),
                            &WinkProfile::default(),
                            &timing,
                        );
                        output[index] = frame[0].openness;
                        let expected = state.model_response_for_raw(0, raw, native_reported);
                        assert!((output[index] - expected).abs() < 1e-6,
                            "adaptive={index} native={native_reported} raw={raw} expected={expected} output={}", output[index]);
                        assert_eq!(frame[0].openness.to_bits(), frame[1].openness.to_bits());
                        max_step[index] =
                            max_step[index].max((output[index] - previous[index]).abs());
                        previous[index] = output[index];
                    }
                    assert_eq!(
                        output[0].to_bits(),
                        output[1].to_bits(),
                        "adaptive toggle changed manual output at {raw}"
                    );
                }
            }
            assert!(max_step.iter().all(|step| *step < 0.00271), "{max_step:?}");
        }
    }

    #[test]
    fn manual_range_adaptive_toggle_does_not_restore_legacy_closed_hold() {
        let (mut state, gaze) = manual_range_fixture(false, true);
        state.eyes[0].is_closed = true;
        for enabled in [false, true, false, true] {
            state.tuning.continuous_calib = enabled;
            let frame = state.process_frame([[1.0, 0.30, 0.0, 0.0, 0.0]; 2], &gaze, true);
            let expected = state.model_response_for_raw(0, 0.30, true);
            assert!((frame[0].openness - expected).abs() < 1e-6);
            assert!(!state.eyes[0].is_closed);
            assert!(!state.eyes[0].response_snap_committed);
        }
    }

    #[test]
    fn manual_range_keeps_the_last_five_percent_until_the_exact_zero_endpoint() {
        for adaptive_bounds in [false, true] {
            let (mut state, gaze) = manual_range_fixture(adaptive_bounds, true);
            for raw in [0.23, 0.22, 0.21, 0.205, 0.201] {
                let frame = state.process_frame([[1.0, raw, 0.0, 0.0, 0.0]; 2], &gaze, true);
                assert!(
                    frame[0].openness > 0.0,
                    "raw={raw} must not be rounded to closed"
                );
                assert!(
                    (frame[0].openness - state.model_response_for_raw(0, raw, true)).abs() < 1e-6
                );
            }
            let zero = state.effective_closed_ref(0);
            let frame = state.process_frame([[1.0, zero, 0.0, 0.0, 0.0]; 2], &gaze, true);
            assert_eq!(frame[0].openness, 0.0);
            assert_eq!(frame[1].openness, 0.0);
        }
    }

    #[test]
    fn manual_range_retains_native_fast_blink_and_unilateral_wink_overrides() {
        for adaptive_bounds in [false, true] {
            let (mut state, _) = manual_range_fixture(adaptive_bounds, true);
            native_step(&mut state, 0.54, 0.54, true, true);
            let closed = native_step(&mut state, 0.42, 0.44, false, false);
            assert_eq!([closed[0].openness, closed[1].openness], [0.0; 2]);
            let held = native_step(&mut state, 0.48, 0.49, false, false);
            assert_eq!([held[0].openness, held[1].openness], [0.0; 2]);
            let mut reopened = held;
            for _ in 0..12 {
                reopened = native_step(&mut state, 0.60, 0.60, true, true);
            }
            assert!(reopened.iter().all(|eye| eye.openness > 0.95));

            let (mut state, _) = manual_range_fixture(adaptive_bounds, true);
            native_step(&mut state, 0.54, 0.60, true, true);
            let wink = native_step(&mut state, 0.40, 0.60, false, true);
            assert_eq!(wink[0].openness, 0.0);
            assert!(
                wink[1].openness > 0.95,
                "native wink must not close its partner"
            );
        }
    }

    #[test]
    fn manual_range_retains_ml_fast_blink_and_dropout_closure() {
        for adaptive_bounds in [false, true] {
            let (mut state, gaze) = manual_range_fixture(adaptive_bounds, false);
            let mut lowest = 1.0f32;
            for raw in [0.48, 0.48, 0.46, 0.46, 0.46, 0.46] {
                let frame = state.process_frame([[1.0, raw, 0.0, 0.0, 0.0]; 2], &gaze, true);
                lowest = lowest.min(frame[0].openness).min(frame[1].openness);
            }
            assert_eq!(
                lowest, 0.0,
                "confirmed ML-only blink must still close fully"
            );
            for _ in 0..16 {
                state.process_frame([[1.0, 0.60, 0.0, 0.0, 0.0]; 2], &gaze, true);
            }
            let dropped = state.process_frame([[0.0, 0.60, 0.0, 0.0, 0.0]; 2], &gaze, true);
            assert!(dropped.iter().all(|eye| eye.openness == 0.0));
        }
    }

    #[test]
    fn full_close_scale_moves_only_the_close_endpoint() {
        let mut state = SRanipalState::new();
        for eye in &mut state.eyes {
            eye.baseline = 0.60;
            eye.closed_ref = 0.40;
            eye.blink_depth = 0.20;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            eye.learned_once = true;
        }
        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        let ordinary = state.model_response_for_raw(0, 0.46, false);
        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.15; 2], [0.5; 2], 0.0, 1.0, false,
        );
        let deeper = state.model_response_for_raw(0, 0.46, false);
        assert!(deeper > ordinary, "deeper zero point should close later");
        let effective_close = state.effective_closed_ref(0);
        assert_eq!(state.model_response_for_raw(0, effective_close, false), 0.0);
        assert_eq!(state.model_response_for_raw(0, 0.60, false), 1.0);
        assert_eq!(
            state.eyes[0].closed_ref, 0.40,
            "trim must not rewrite calibration"
        );
    }

    #[test]
    fn visual_range_auto_reseat_moves_only_the_open_point() {
        let mut state = SRanipalState::new();
        for eye in &mut state.eyes {
            eye.baseline = 0.60;
            eye.closed_ref = 0.41;
            eye.blink_depth = 0.19;
        }
        state.set_eyelid_response_tuning(
            true,
            [0.03, 0.05],
            [0.40, 0.32],
            [1.0; 2],
            [0.5; 2],
            0.0,
            1.0,
            false,
        );
        assert!((state.effective_open_ref(0) - 0.57).abs() < 1e-6);
        assert!((state.effective_closed_ref(0) - 0.20).abs() < 1e-6);
        assert!((state.model_response_for_raw(0, 0.57, false) - 1.0).abs() < 1e-5);
        assert!(state.model_response_for_raw(0, 0.20, false).abs() < 1e-5);
        assert!((state.effective_open_ref(1) - 0.55).abs() < 1e-6);
        assert!((state.effective_closed_ref(1) - 0.28).abs() < 1e-6);

        state.session_baseline_offset = [0.04, -0.02];
        assert!((state.effective_open_ref(0) - 0.61).abs() < 1e-6);
        assert!((state.effective_closed_ref(0) - 0.20).abs() < 1e-6);
        assert!((state.effective_open_ref(1) - 0.53).abs() < 1e-6);
        assert!((state.effective_closed_ref(1) - 0.28).abs() < 1e-6);
        assert_eq!(
            state.model_response_for_raw(0, 0.20, false),
            0.0,
            "auto reseat must never make the calibrated full close harder to reach"
        );
        assert_eq!(
            state.model_response_for_raw(1, 0.28, false),
            0.0,
            "the close floor remains fixed for either offset direction"
        );
    }

    #[test]
    fn shared_snap_gate_delays_special_zero_but_not_tracking_loss() {
        let mut state = SRanipalState::new();
        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 0.35, false,
        );
        let mut samples = [PerEye::default(); 2];
        for sample in &mut samples {
            sample.present = true;
            sample.openness_target = 1.0;
            sample.model_response = 1.0;
        }
        samples[0].openness_target = 0.0;
        samples[0].model_response = 0.60;
        samples[0].snap_requested = true;
        samples[0].snap_reason = ClosureReason::NativeDisable;
        state.resolve_snap_gate(&mut samples);
        assert_eq!(samples[0].reason, ClosureReason::BlinkPending);
        assert_eq!(samples[0].openness_target, 0.60);
        assert!(!state.eyes[0].response_snap_committed);

        samples[0].openness_target = 0.0;
        samples[0].model_response = 0.20;
        samples[0].snap_requested = true;
        state.resolve_snap_gate(&mut samples);
        assert_eq!(samples[0].reason, ClosureReason::NativeDisable);
        assert_eq!(samples[0].openness_target, 0.0);
        assert!(state.eyes[0].response_snap_committed);

        samples[0].present = false;
        samples[0].model_response = 0.90;
        samples[0].snap_requested = false;
        samples[0].openness_target = 0.90;
        state.resolve_snap_gate(&mut samples);
        assert_eq!(samples[0].reason, ClosureReason::TrackingLost);
        assert_eq!(samples[0].openness_target, 0.0);
    }

    #[test]
    fn response_profile_close_time_limits_a_tracking_loss_drop() {
        let mut instant = SRanipalState::new();
        let mut slowed = SRanipalState::new();
        instant.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        slowed.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 120.0, 1.0, false,
        );
        feed(&mut instant, 0.62, 240);
        feed(&mut slowed, 0.62, 240);
        let lost = [[0.0, 0.62, 0.0, 0.0, 0.0]; 2];
        let a = instant.process_frame(lost, &GazeSample::default(), true);
        let b = slowed.process_frame(lost, &GazeSample::default(), true);
        assert_eq!(a[0].openness, 0.0);
        assert!(b[0].openness > 0.0 && b[0].openness < 1.0);
        assert_eq!(
            slowed.eyelid_live_diag()[0].reason,
            ClosureReason::TrackingLost
        );
    }

    #[test]
    fn auto_reseat_offset_is_transient_and_never_changes_snapshot() {
        let mut state = SRanipalState::new();
        for eye in &mut state.eyes {
            eye.baseline = 0.60;
            eye.closed_ref = 0.40;
            eye.blink_depth = 0.20;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            eye.learned_once = true;
            eye.endpoint_locked = true;
        }
        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, true,
        );
        let eye = crate::core::types::EyeSample {
            gaze: [0.0, 0.0, -1.0],
            gaze_valid: true,
            openness: 1.0,
            openness_valid: true,
            openness_reported: true,
            ..Default::default()
        };
        let gaze = GazeSample {
            left: eye,
            right: eye,
            ..Default::default()
        };
        let ml = [[1.0, 0.54, 0.0, NATIVE_SQ_FLOOR, 0.0]; 2];
        let before = state.snapshot_all();
        for _ in 0..540 {
            state.update_auto_reseat([0.54; 2], &ml, &gaze);
        }
        assert!(state.session_baseline_offset[0] < -0.01);
        assert!(state.session_baseline_offset[1] < -0.01);
        let after = state.snapshot_all();
        assert_eq!(
            before.left.baseline.to_bits(),
            after.left.baseline.to_bits()
        );
        assert_eq!(
            before.left.blink_depth.to_bits(),
            after.left.blink_depth.to_bits()
        );
        assert_eq!(
            before.right.baseline.to_bits(),
            after.right.baseline.to_bits()
        );
        assert_eq!(
            before.right.blink_depth.to_bits(),
            after.right.blink_depth.to_bits()
        );

        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        assert_eq!(state.session_baseline_offset, [0.0; 2]);
    }

    #[test]
    fn appearance_profile_translates_both_endpoints_without_persisting() {
        let mut state = SRanipalState::new();
        let before = state.snapshot_all();
        let open_before = state.effective_open_ref(0);
        let closed_before = state.effective_closed_ref(0);
        let baseline_target = [before.left.baseline + 0.06, before.right.baseline - 0.04];
        for _ in 0..120 {
            state.update_appearance_profile(Some(baseline_target), Some([0.82, 0.79]));
        }
        let open_shift = state.effective_open_ref(0) - open_before;
        let closed_shift = state.effective_closed_ref(0) - closed_before;
        assert!((open_shift - 0.06).abs() < 0.002, "{open_shift}");
        assert!((closed_shift - 0.06).abs() < 0.002, "{closed_shift}");
        let after = state.snapshot_all();
        assert_eq!(
            before.left.baseline.to_bits(),
            after.left.baseline.to_bits()
        );
        assert_eq!(
            before.right.baseline.to_bits(),
            after.right.baseline.to_bits()
        );
    }

    #[test]
    fn appearance_profile_fades_out_and_never_stacks_auto_reseat() {
        let mut state = SRanipalState::new();
        let target = [state.eyes[0].baseline + 0.05, state.eyes[1].baseline + 0.05];
        for _ in 0..80 {
            state.update_appearance_profile(Some(target), None);
        }
        assert!(state.appearance_profile_active);
        state.session_baseline_offset = [0.03, 0.03];
        state.update_appearance_profile(Some(target), None);
        assert_eq!(state.session_baseline_offset, [0.0; 2]);
        for _ in 0..80 {
            state.update_appearance_profile(None, None);
        }
        assert!(!state.appearance_profile_active);
        assert_eq!(state.appearance_baseline_offset, [0.0; 2]);
    }

    #[test]
    fn adopting_recovered_coordinates_preserves_both_endpoints() {
        let mut state = SRanipalState::new();
        state.eyes[0].baseline = 0.50;
        state.eyes[1].baseline = 0.55;
        state.set_eyelid_response_tuning(
            true, [0.03; 2], [0.30; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, false,
        );
        for _ in 0..120 {
            state.update_appearance_profile(Some([0.60, 0.45]), None);
        }
        let open = [state.effective_open_ref(0), state.effective_open_ref(1)];
        let close = [state.effective_closed_ref(0), state.effective_closed_ref(1)];
        state.adopt_current_appearance();
        assert_eq!(state.appearance_baseline_offset, [0.0; 2]);
        for i in 0..2 {
            assert!((state.effective_open_ref(i) - open[i]).abs() < 1e-6);
            assert!((state.effective_closed_ref(i) - close[i]).abs() < 1e-6);
        }
        assert!((state.effective_open_ref(0) - 0.57).abs() < 1e-5);
    }

    #[test]
    fn auto_reseat_accepts_manual_range_without_recorded_endpoint_lock() {
        let mut state = SRanipalState::new();
        for eye in &mut state.eyes {
            eye.baseline = 0.60;
            eye.closed_ref = 0.40;
            eye.blink_depth = 0.20;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            assert!(!eye.learned_once);
            assert!(!eye.endpoint_locked);
        }
        state.set_eyelid_response_tuning(
            true, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, true,
        );
        let eye = crate::core::types::EyeSample {
            gaze: [0.0, 0.0, -1.0],
            gaze_valid: true,
            openness: 1.0,
            openness_valid: true,
            openness_reported: true,
            ..Default::default()
        };
        let gaze = GazeSample {
            left: eye,
            right: eye,
            ..Default::default()
        };
        let ml = [[1.0, 0.54, 0.0, NATIVE_SQ_FLOOR, 0.0]; 2];
        for _ in 0..540 {
            state.update_auto_reseat([0.54; 2], &ml, &gaze);
        }
        assert!(state.session_baseline_offset[0] < -0.01);
        assert!(state.session_baseline_offset[1] < -0.01);
    }

    #[test]
    fn auto_reseat_rejects_a_stable_unilateral_partial_close() {
        let mut state = SRanipalState::new();
        for eye in &mut state.eyes {
            eye.baseline = 0.60;
            eye.closed_ref = 0.40;
            eye.blink_depth = 0.20;
            eye.baseline_n = BASELINE_BOOTSTRAP_N;
            eye.learned_once = true;
            eye.endpoint_locked = true;
        }
        state.set_eyelid_response_tuning(
            false, [0.03; 2], [0.40; 2], [1.0; 2], [0.5; 2], 0.0, 1.0, true,
        );
        let open = crate::core::types::EyeSample {
            gaze: [0.0, 0.0, -1.0],
            gaze_valid: true,
            openness: 1.0,
            openness_valid: true,
            openness_reported: true,
            ..Default::default()
        };
        let gaze = GazeSample {
            left: open,
            right: open,
            ..Default::default()
        };
        let ml = [
            [1.0, 0.55, 0.0, NATIVE_SQ_FLOOR, 0.0],
            [1.0, 0.60, 0.0, NATIVE_SQ_FLOOR, 0.0],
        ];
        for _ in 0..600 {
            state.update_auto_reseat([0.55, 0.60], &ml, &gaze);
        }
        assert_eq!(state.session_baseline_offset, [0.0; 2]);
        assert_eq!(state.reseat_stable_frames, 0);
    }
}
