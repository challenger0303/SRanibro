//! SRanipalState — HMD-agnostic post-processor for the eyelid ML signal.
//!
//! Faithful port of `eye_state_processor.py::SRanipalState` plus the per-frame
//! driving logic from `core/pipeline.py::emit_thread` (streaks, blink gate,
//! cross-eye coupling/yoke, asymmetric EMA). The whole per-frame body lives in
//! [`SRanipalState::process_frame`] so it is unit-testable in isolation.
//!
//! Input: ML output index 1 (openness raw; relax≈0.66 wide≈0.71 squeeze≈0.55
//! closed≈0.39) per eye, plus the native gaze sample. Output: two [`EyeResult`].

use super::types::{Eye, EyeResult, GazeSample};

// --- tuning constants (identical to the Python reference) ---
const OPEN_TRUST_FRAMES: u32 = 20;
const BLINK_RESET_FRAMES: u32 = 5;
const WARMUP_FRAMES: u32 = 200;
const KALMAN_ALPHA: f32 = 0.30;
const UPPER_OFFSET: f32 = 0.02;
const SQUEEZE_TOP_OFFSET: f32 = 0.03;
const BLINK_OFFSET: f32 = 0.20; // raw < baseline-0.20 -> enter blink
const BLINK_RELEASE_OFFSET: f32 = 0.10; // raw must rise > baseline-0.10 to exit blink
const WIDE_RELEASE_OFFSET: f32 = 0.01; // raw must drop > upper-0.01 to exit wide
/// Adaptive wide ceiling: minimum openness span above `upper` for wide=1.0 (caps
/// how touchy wide can be — bigger = less sensitive), and how slowly the learned
/// ceiling relaxes back toward that floor.
const WIDE_CEIL_MIN_SPAN: f32 = 0.15;
const WIDE_CEIL_DECAY: f32 = 0.0008;
/// How fast the native-squeeze floor tracks the relaxed s3 baseline (auto-calib).
const SQ_FLOOR_TRACK: f32 = 0.005;
// --- blink-depth calibration (episode-based, v2 2026-07-07) ---
// The openness=0 point is learned PER BLINK EPISODE instead of the old per-frame
// ratchet + decay: the ratchet moved on a single sample and the decay (τ ~8s)
// erased a deliberate slow-blink calibration within seconds — the user-facing
// "calibrate by blinking slowly" gesture effectively did nothing. Now each
// two consistent qualifying episodes establish a persistent per-eye endpoint.
// `blink_depth` (= baseline − bottom) is the persisted coordinate; `closed_ref`
// is the live absolute coordinate. They commit atomically so Recenter cannot
// re-import an unconfirmed partial close through the persisted depth.
/// Episode-local smoothing of raw while closed (α per emit frame): the learned
/// bottom is the MIN of this smoothed trace, so a single noisy sample can't
/// define the calibration.
const BLINK_BOTTOM_SMOOTH: f32 = 0.3;
/// Episode length gates (emit frames below the entry threshold): shorter =
/// under-sampled bottom (fast blink that dodged the latch), longer = resting
/// with eyes closed (model output drifts there; not a calibration blink).
const BLINK_LEARN_MIN_FRAMES: u32 = 18; // 150 ms @120 Hz
const BLINK_LEARN_MAX_FRAMES: u32 = 240; // 2 s @120 Hz

// Learned-floor updates are transactional. A deeper change needs a confirming
// pair; the dangerous shallow direction needs three mutually-consistent
// episodes because a partial close cannot prove the true full-close floor.
const FLOOR_DEEP_CONFIRM_N: u8 = 2;
const FLOOR_SHALLOW_CONFIRM_N: u8 = 3;
const FLOOR_DEEP_CONFIRM_TOL: f32 = 0.03;
const FLOOR_SHALLOW_CONFIRM_TOL: f32 = 0.02;
// Ignore small shallow observations: they are ordinary lazy blinks/squints,
// not evidence that a persisted floor is genuinely too deep.
const FLOOR_SHALLOW_ESCAPE: f32 = 0.02;
// Bound both a single bad commit and cumulative travel between Recenters.
const FLOOR_DEEP_COMMIT_CAP: f32 = 0.03;
const FLOOR_SHALLOW_COMMIT_CAP: f32 = 0.015;
const FLOOR_TRAVEL_BUDGET: f32 = 0.08;
const FLOOR_COMMIT_MIN_FRAMES: u32 = 600; // 5 s @120 Hz
const FLOOR_CANDIDATE_MAX_AGE: u32 = 10_800; // 90 s @120 Hz

// A close may train the L/R mid-curve without moving the endpoint only when
// its bottom is already near the committed floor.
const FLOOR_CURVE_NEAR: f32 = 0.02;
// --- close-side per-eye auto-range floor (shallow-eye fix, 2026-07-09) ---
// openness reached 0 only when raw dropped to baseline − blink_depth, and blink_depth
// floored at BLINK_OFFSET while the episode learner only fired below baseline−0.20. So
// an eye whose full close drops <0.20 below its OWN baseline (a "shallow" eye) could
// never reach 0 (floored ~0.47) and never recalibrated — a self-locking dead zone. Fix
// = learn each eye's own reachable floor, like the reference SRanipal/BrokenEye per-eye
// auto-range. `reach_env` is a present-gated MIN-envelope of raw (TRANSIENT — never
// persisted): it settles at the full-close bottom (squints ride above it), so it is used
// ONLY as (a) a relative episode-ENTRY gate and (b) a genuine-close DISCRIMINATOR, never
// as the emitted value.
/// MIN-envelope rates per emit frame: attack DOWN fast toward a new low, heal UP slowly.
/// HEAL τ ~2.8min (2026-07-13; was τ~17s): the envelope is the genuine-close
/// discriminator, so it must outlive a STARE. At τ~17s a ~30s no-blink stretch healed
/// it nearly back to baseline, the genuine gate went soft (blink_lo <= reach_env +
/// GENUINE_NEAR passed for anything), and a slow half-squint then taught blink_depth
/// SHALLOWER — the measured right-eye depth had eroded to 0.120 vs the left's 0.219,
/// thinning that eye's ramp denominator and amplifying every baseline error into the
/// close floor. Natural blink gaps are 3-10s; stares run 30-60s; τ~2.8min keeps the
/// floor evidence alive across both.
const ENV_ATTACK: f32 = 0.02;
const ENV_HEAL: f32 = 0.00005;
/// The envelope only tracks raw within this much below baseline — a dropout / near-0
/// sample can't drag the floor to nonsense.
const ENV_SANITY_OFFSET: f32 = 0.40;
/// Relative episode-entry gate: enter below baseline − clamp(GATE_FRAC·(baseline −
/// reach_env), GATE_MIN, BLINK_OFFSET). A normal eye's envelope sits ~0.20 below
/// baseline so the gate stays ~0.20; a shallow eye's sits higher, tightening the gate
/// so its real closes still fire.
const GATE_FRAC: f32 = 0.55;
const GATE_MIN: f32 = 0.08;
/// First-stage discriminator: an episode is considered only if its bottom reaches
/// the envelope floor (blink_lo <= reach_env + GENUINE_NEAR). The envelope can itself
/// follow repeated squints, so this is only a prefilter; transactional confirmation
/// and travel caps below provide the actual commit safety.
const GENUINE_NEAR: f32 = 0.04;
/// First-close confirmation tolerance. A cold-start episode is only a candidate;
/// a second close must reach approximately the same floor before it is trusted.
const FIRST_CLOSE_CONFIRM_TOL: f32 = 0.02;
// --- close-side decoupling (v3, 2026-07-13) ---
// closed_ref used to be RE-DERIVED from the baseline every frame (baseline −
// blink_depth), which propagated every baseline error 1:1 into the openness=0 point.
// During play the baseline slowly sinks (down-gaze / squint dwell accumulating through
// the out-of-band slow path, and once >band even NEUTRAL frames heal it only at
// τ~2.3min), so a standing offset δ built up and full closes read δ/(depth −
// open_deadzone) open — "one eye stops closing over time; recentering the baseline
// fixes it" (user 2026-07-13). Measured depths L 0.219 / R 0.120 mean the same δ hits
// the right eye ~2x. Native SRanipal is structurally immune: its lower bound is
// learned DIRECTLY from observed lows (BlinkResetState / TrackLowStreak, RE'd
// 2026-07-11), independent of upper. v3 adopts that shape: while COUPLED (baseline
// bootstrap in progress, or no confirmed close yet) closed_ref stays baseline-derived
// — that is the seed path, and recenter/restore re-enter it via their bootstrap
// re-verification so "Recenter" remains the instant whole-reset. Once MATURE
// (learned_once && bootstrap done) closed_ref is an independent absolute level,
// updated only after multiple mutually-consistent clean slow-close episodes,
// with per-commit and cumulative travel caps. A shallow update needs three closes
// because one partial close cannot prove that the true endpoint moved. There is
// deliberately NO continuous envelope-follow here: a
// stare-healed reach_env is absence of evidence, and following it up would slowly
// lift the close floor — the very failure v3 removes.
/// Openness-ramp denominator floor (raw units), applied in `normalize`. With a shallow
/// learned depth and a wide open_deadzone the ramp span (depth − deadzone) can get
/// razor-thin (0.04 measured on a real config = ~25x error amplification). Flooring the
/// DENOMINATOR (not moving blink_top) keeps 0 reachable exactly at the learned bottom;
/// the only cost is dips near open_full reading slightly <1.0 in such degenerate
/// configs (the ramp tops out at blink_top + this span instead of open_full).
const MIN_CLOSE_SPAN: f32 = 0.05;
// --- fast-blink detector v2 (2026-07-09) ---
// The proportional continuous ramp under-reports fast blinks (the eye reopens
// before raw reaches closed_ref), so a latch snaps openness to 0 for the
// descent. v2 redesign, data-derived from two 120Hz diagnostic recordings with
// every eyelid event hand-labeled (30 per-eye fast blinks, 22 slow, 4 winks):
// - All evidence thresholds are RELATIVE to the per-eye learned blink_depth D:
//   the two eyes' dynamic ranges differ ~1.6x on real hardware, so the old
//   absolute 0.06 threshold needed the weaker eye to blink ~60% faster in
//   normalized terms (one full user-visible right-eye miss on record).
//   Measured, normalized by D: fast blinks step 0.24-0.58 per ML update; slow
//   blinks <= 0.14; relaxed noise p99 <= 0.03 with worst single steps 0.048 abs.
// - Evidence is evaluated per DISTINCT ML update (emit duplicates each 60Hz
//   sample ~2-3x; per-emit-frame counting is phase-dependent).
// - A cumulative 2-update fall path catches medium blinks whose descent splits
//   across updates (never one step over threshold); it is DEPTH-GATED so a
//   brisk squint onset can never commit through accumulation.
// - Release is RISE-OFF-BOTTOM: the old "first up-tick" release fired on
//   sub-noise bottom tremors (+0.001..0.012) — 11 re-latch flickers in one 28s
//   recording; real reopen steps measured >= 0.022.
/// Per-update fall threshold, normalized by blink_depth. 0.19 sits between the
/// slow-blink max (0.14, x1.36) and the fast-blink min (0.24, x1.26).
const BLINK_VEL_NORM: f32 = 0.19;
/// Absolute rails for the velocity threshold: the floor sits above relaxed-noise
/// p99 (0.030) and ±0.02 spec noise (0.04 single step; the one recorded 0.048
/// noise step is absorbed by the confirmation window); the cap keeps a
/// stale-large learned depth (0.40) from desensitizing below the old 0.06 era.
const BLINK_VEL_FLOOR: f32 = 0.045;
const BLINK_VEL_CAP: f32 = 0.075;
/// Net-fall-over-2-updates threshold (normalized by D, with absolute rails):
/// catches split-step medium blinks. Measured: fast >= 0.37, slow <= 0.25.
const BLINK_FALL2_NORM: f32 = 0.31;
const BLINK_FALL2_FLOOR: f32 = 0.060;
const BLINK_FALL2_CAP: f32 = 0.115;
/// The cumulative (fall2) path may only ARM once the eye is this deep (fraction
/// of D below baseline): a real medium blink passes it within one extra update,
/// while a brisk squint onset (shallow by definition) can never accumulate its
/// way into a phantom blink. The single-step velocity path is NOT depth-gated
/// (weak-eye fast blinks bottom at only 0.38-0.66 of D) — its false positives
/// are bounded by the shallow commit cap instead.
const FALL2_COMMIT_DEPTH: f32 = 0.60;
/// Safety cap: max frames the latch holds 0 before releasing to the ramp anyway
/// (~200ms). SHALLOW-ONLY (see the release logic): a deep hold is a genuinely
/// closed eye and keeps reading 0 until raw actually leaves the closed zone —
/// cap-releasing a held wink into the ramp was the "bounce after closing"
/// artifact (user 2026-07-09). The cap still bounds misclassified shallow
/// commits (blink-speed squint onsets) exactly as before.
const FAST_BLINK_MAX_FRAMES: u32 = 24;
/// Shallow-bottom commits (dip < SHALLOW_COMMIT_DIP of D) get a tighter cap:
/// bounds a blink-speed squint onset that sneaks past the depth gates to ~100ms
/// of forced close; real shallow-bottom weak-eye blinks dwell < 100ms anyway.
const FAST_BLINK_CAP_SHALLOW: u32 = 12;
const SHALLOW_COMMIT_DIP: f32 = 0.5;
/// Confirmation window: after arming we emit the ramp for this many emit frames
/// AND require at least one fresh ML update to confirm the descent (the old
/// emit-frame-only count committed blind on duplicated frames ~half the time).
/// If no fresh sample arrives by ARM_MAX emit frames, fall back to the legacy
/// blind expiry (commit if raw never recovered above the armed level).
const FAST_BLINK_ARM_FRAMES: u32 = 2;
const FAST_BLINK_ARM_MAX: u32 = 3;
/// Release when raw rises this far off the tracked latch bottom (normalized by
/// D, absolute floor above bottom-tremor amplitude 0.001-0.009).
const RELEASE_RISE_NORM: f32 = 0.05;
const RELEASE_RISE_FLOOR: f32 = 0.012;
/// ...AND the rise must actually LEAVE the closed zone: release only above
/// blink_top + this margin (x D, clamped 0.03..0.09 raw). Typical closure
/// bottoms sit +0.02..0.05 ABOVE the learned closed_ref (it learns the DEEPEST
/// slow-blink bottoms), and a hard close overshoots then recovers a few
/// hundredths while the lid is still shut — without this the latch released
/// mid-closure and the ramp rendered a 0.1-0.25 "bounce" right after closing,
/// most visibly on held winks (user 2026-07-09). A genuine reopen crosses the
/// margin within 1-2 ML updates (~8-33ms extra closed — imperceptible).
const LATCH_EXIT_NORM: f32 = 0.25;
/// Bilateral assist: a real blink is bilateral, and one eye's detection can
/// trail the other's. When the PARTNER has committed and this eye is itself
/// clearly mid-blink by its OWN relative evidence — at least this deep (x D)
/// and having fallen at least ASSIST_FALL2_NORM (x D) over recent updates —
/// commit it too. There is deliberately NO unlatched anchor path: two
/// sub-threshold eyes can never bootstrap each other, so bilateral squints are
/// structurally safe, and a wink's open eye (measured worst sympathetic droop
/// 0.31 x D, slow) can never qualify.
const ASSIST_DIP_NORM: f32 = 0.40;
const ASSIST_FALL2_NORM: f32 = 0.20;
/// Detector grace after construction / restore / recenter (~200ms): the first
/// real frame after any of those sees a PHANTOM step (raw vs a seeded prev_raw
/// or a stale restored baseline) that can arm-and-commit a fake blink at
/// session start (found by the golden-replay tests). No triggering until real
/// history exists; the smooth ramp still tracks meanwhile.
const DETECT_GRACE_FRAMES: u32 = 24;
/// Gaze-yoke hysteresis: the yoke engages the moment an eye reads closed (the
/// blink flag, openness < 0.08) but releases only once the eye has genuinely
/// reopened past THIS smoothed openness. Without it the yoke flapped at the
/// 0.08 boundary during squints — alternating each frame between the partner's
/// gaze and the eye's OWN gaze, which mid-squint is exactly the garbage the
/// yoke exists to replace (gaze_valid often stays true while the lid hides the
/// pupil) — rendering as visible gaze trembling (user 2026-07-08).
/// Engage before a shallow squint hides enough pupil for XR5 gaze to become noisy.
/// The old implementation waited for the blink flag (openness < 0.08), so the UI's
/// "squint eye follows open eye" setting was effectively inert through most squints.
const YOKE_ENGAGE_OPEN: f32 = 0.35;
/// Release with wide hysteresis and only after the eye also has valid gaze again.
const YOKE_RELEASE_OPEN: f32 = 0.55;

// --- XR5/native gaze stabilization ---
/// Calm lower bound for the motion-adaptive gaze EMA. Large motion still raises alpha
/// continuously up to 0.85, preserving saccades and smooth pursuit.
const GAZE_ALPHA_CALM: f32 = 0.08;

// --- L/R mid-close curve equalization (learned mid-anchor, 2026-07-07) ---
// On a SYMMETRIC slow close the two cameras' raw curves diverge mid-close (the
// right runs ahead: L read 0.8 while R read 0.65). The ramp denominator is only
// blink_depth - open_deadzone ~= 0.107, i.e. the ramp amplifies raw ~9.3x into
// openness — that reported gap is a raw divergence of just ~0.016. Endpoint
// calibration (deadzone, blink_depth) can never fix a MID-curve difference, and
// runtime cross-eye coupling of the targets cannot be made safe (close_yoke v1:
// gating a 9x-noise-amplified signal flapped visibly at alpha_close=1.0 and
// noise dips of a wink's open eye yanked the winking eye up). So instead each
// eye gets a STATIC monotone piecewise-linear correction of its ramp output —
// (0,0) -> (mid_anchor, 0.5) -> (1,1) — learned from the eyes themselves and
// applied with zero cross-eye terms at runtime: nothing exists to oscillate,
// and winks are untouched by construction.
/// Anchor clamp: covers the reported case's implied anchors (0.364/0.636) with
/// margin; hard slope cap 0.5/0.30 = 1.67x keeps mid-close noise amplification
/// bounded and the map safely monotone.
const MID_ANCHOR_MIN: f32 = 0.30;
const MID_ANCHOR_MAX: f32 = 0.70;
/// Per-committed-episode EMA blend (parity with BLINK_DEPTH_DEEPEN: 2-3
/// deliberate slow closes converge the correction).
const MID_ANCHOR_LEARN: f32 = 0.3;
/// Paired-sample gates: reject pairs whose PRE-correction ramp values differ
/// more than this (a wink pair differs by ~1.0; a symmetric close by ~0.15).
const PAIR_MAX_DIFF: f32 = 0.30;
/// Samples are collected only while the eye is ACTIVELY descending: at most this
/// many frames since the last real down-step (covers the 60Hz-duplicate cadence
/// and jittery slow descents). A static HOLD stops collecting within ~2 ML
/// updates — an asymmetric half-close held before a calibration blink must not
/// train the anchors (review 2026-07-08) — and slow REOPENS never collect
/// (their real steps are upward).
const CURVE_DOWN_RECENT: u32 = 4;
/// The pending pair-set must have STARTED no earlier than this many frames
/// (~0.5s) before the blink episodes' entry — samples from an unrelated earlier
/// descent or hold must not ride a later blink's clean episodes into a commit
/// (review 2026-07-08). Covers the mid-close descent phase that precedes
/// episode entry at any speed the collector accepts.
const CURVE_PRE_WINDOW: u32 = 60;
/// Samples are weighted by closeness of the corrected pair MEAN to half-closed
/// (triangle window; the anchor is defined at output 0.5), and a commit needs
/// at least this much accumulated weight (~8+ strong mid-close frames).
const PAIR_MID_WINDOW: f32 = 0.25;
const PAIR_MIN_WEIGHT: f32 = 4.0;
/// Pending pairs expire if the close drags on (~5s) or is abandoned (both eyes
/// back near baseline without completing clean blink episodes).
const PAIR_STALE_FRAMES: u32 = 600;
/// The two eyes' blink episodes must have STARTED within this many frames
/// (~150ms) of each other to count as one bilateral gesture — sequential winks
/// can never pair up. KNOWN BOUNDED LIMITATION (review 2026-07-08): a habitual
/// per-eye blink-timing LEAD inside this window reads as a curve difference and
/// is partially learned; that trades static-hold symmetry for dynamic-close
/// symmetry (the user's actual complaint), is bounded by the anchor clamps +
/// pair centering, and heals on lead-free slow blinks. Escalation if beta
/// reports asymmetric HOLDS after calibration: normalized-time realignment of
/// the pair samples. Do not tighten below ~12 frames (natural bilateral blink
/// onset skew).
const EP_SYNC_SKEW: u32 = 18;
/// Raw-domain "closing fast" threshold for the emit-stage gaze-invalid forced
/// close. The old test (target dropped > 0.10 openness/frame) was equivalent to
/// only ~0.011 of RAW at the nominal ramp slope — inside ordinary mid-close
/// noise (+-0.02), so a noisy half-closed hold with gaze lost FLAPPED to 0 and
/// back (pre-existing bug; the anchor map amplified it up to 1.67x — review
/// 2026-07-08). Raw-domain and above the WORST-CASE noise transition (a +-0.02
/// jitter flips into a 0.04 step between updates): only a genuinely fast
/// descent trips it (just under the 0.06 single-eye latch, whose forced-0 path
/// covers everything faster), and the learned map cannot change its
/// sensitivity.
const FAST_CLOSING_RAW: f32 = -0.05;
/// Absolute sanity band for baseline learning. LO was 0.45 from the raw~0.62
/// era; a 2026-07-08 diagnostic recording showed today's setup idles at raw
/// 0.42-0.50 — the right eye's relaxed level (0.43) sat BELOW the band, so its
/// baseline could never learn OR recenter and stuck 0.07 high, shifting every
/// derived threshold (the real reason the curve equalizer "didn't work").
/// Blink bottoms (0.14-0.25 across eras) stay safely below 0.32; the RELATIVE
/// band-gate does the fine rejection either way.
const BASELINE_RANGE_LO: f32 = 0.32;
const BASELINE_RANGE_HI: f32 = 0.80;
/// SYMMETRIC bootstrap window: past the anchor phase, a bootstrap sample enters
/// the running mean only within +-this of the current estimate — keeps wide
/// (+0.18) AND blinks/deep closes (-0.2..-0.35) out of the mean (the widened
/// absolute band no longer excludes blink raws by value — 2026-07-08). Rejected
/// samples still drift the baseline at the slow rate (never zero — see
/// BASELINE_LEARN_BAND); the mature path uses the band-gate instead.
const WIDE_LEARN_CAP: f32 = 0.10;
/// The first few bootstrap samples are UNCAPPED (the estimate may start
/// anywhere — a cold start's default 0.60 must anchor onto a real setup that
/// idles at 0.43 in one step). 8 samples of anchor, then the window guards.
const BOOTSTRAP_ANCHOR_N: u32 = 8;
/// Bootstrap length: samples averaged by the running mean before the relaxed
/// open reference is frozen for ordinary play.
const BASELINE_BOOTSTRAP_N: u32 = 100;
/// Recovery rate per emit frame @120Hz (τ ~2.8s). Used only by bootstrap's
/// guarded slow path and the explicit stuck-wide recovery state after maturity.
const BASELINE_ADAPT: f32 = 0.003;
/// Bootstrap outlier slowdown (50x → 6e-5/frame, τ ~2.3min). It prevents a
/// capped stale prior from becoming a zero-rate state; it is never used for
/// ordinary mature tracking.
const BASELINE_SLOW_FACTOR: f32 = 0.02;
/// Stuck-wide breaker: is_wide held NEAR-CONTINUOUSLY this long is the deadlock
/// signature — a corrupted-LOW baseline reading the RELAXED eye as wide (stale
/// restore, recenter mid-expression) — not a facial expression: the longest
/// user-confirmed deliberate wide hold is ~10s (1200 frames), so 15s adds 50%
/// margin, and deliberate wide BURSTS discharge the counter in their relaxed
/// gaps (see WIDE_CLEAR_FRAMES). While stuck, the baseline re-anchors at the
/// FAST rate until `upper` overtakes raw and the existing hysteresis releases
/// the latch naturally (smooth wide fade via the envelope — no output snap).
const WIDE_STUCK_FRAMES: u32 = 1800;
/// Span (in relaxed-open calm frames, ~0.4s) over which the stuck counter fully
/// discharges — see WIDE_DISCHARGE.
const WIDE_CLEAR_FRAMES: u32 = 48;
/// LEAKY discharge of the stuck counter: every RELAXED-OPEN calm frame (present,
/// raw back inside the absolute learning band) erases this much accumulation, so
/// a WIDE_CLEAR_FRAMES-long (~0.4s) relaxed pause erases an entire threshold's
/// worth — genuine wide bursts with >=0.4s gaps can never trip the breaker —
/// while a brief dip erases only its proportional share (no all-or-nothing
/// cliff: two 10s holds a 0.35s breath apart must NOT trip either, review
/// 2026-07-07). A blink also releases is_wide but sits BELOW the band, and a
/// dropout isn't present, so neither discharges — a stuck episode accumulates
/// right through both. KNOWN AMBIGUITY (accepted): during a stuck episode a
/// >=0.4s SQUINT also lands in-band (the corrupted baseline sits AT the squint
/// level — that's where it came from) and discharges the counter, so a user who
/// squints more often than every ~15s defers the AUTOMATIC breaker and falls
/// back to the slow path (minutes). Unresolvable locally (squint-at-corruption
/// is pixel-identical to relaxed-at-health); the instant remedy is recenter,
/// which pre-arms the breaker (see `recenter`).
const WIDE_DISCHARGE: u32 = WIDE_STUCK_FRAMES / WIDE_CLEAR_FRAMES + 1;
/// Recenter keeps the old baseline as this many samples of prior: one noisy or
/// mid-expression frame right after recenter moves the baseline by at most
/// dev/(prior+1) instead of becoming it verbatim (the old `baseline_n = 0` snap).
/// Re-anchoring still completes within ~1s (bootstrap mean over ~100 frames).
const RECENTER_PRIOR_N: u32 = 8;
/// A restored MATURE calibration re-enters a short bootstrap re-verification:
/// the persisted baseline acts as this many samples of prior — confirmed in
/// ~0.5s when still accurate, re-anchored in ~1s when moderately stale. Immature
/// snapshots keep their own count (an idle session persists baseline_n=0 and
/// must not skip its cold start — review 2026-07-06).
const RESTORE_PRIOR_N: u32 = 50;
/// Bilateral-wide engage/disengage envelope rate (per emit frame @120Hz). When the
/// gate opens (both eyes cross into wide), emitted wide ramps 0->1 over ~1/step
/// frames instead of dumping the magnitude the LEADING eye already charged up while
/// waiting for the lagging (weaker) eye — that dump looked like the eye snapping
/// wide open ("がくっと", user 2026-06-27). Also fades wide out on disengage so a
/// single eye relaxing doesn't snap wide off. ~0.08 -> ~100ms transition.
const WIDE_GATE_STEP: f32 = 0.08;
const COUPLE_RATE: f32 = 0.001; // ~10s convergence at 120Hz when 0.04 apart
const COUPLE_MIN_OPEN_STREAK: u32 = 30; // both eyes stable open >=0.25s
const SQUEEZE_DWELL_MIN: u32 = 8; // ~67ms @120Hz before squeeze fires
const SQUEEZE_HARDCLOSE_T: f32 = 0.70; // force openness=0 at this depth into squeeze
/// Model output ch0 = presence/confidence. The real SRanipal EyePredictionModule
/// binarizes it at this threshold (`FUN_180012430`) and treats the eye as lost
/// (forced closed) below it. In normal tracking ch0 stays high (~0.8), so this is
/// a dropout safety gate, not a blink source. (RE'd 2026-06-26.)
const CH0_PRESENT_GATE: f32 = 0.05;

// --- adaptive openness Kalman (RE'd FUN_180010ee0 feeding FUN_180010da0; opt-in) ---
const KAL_MOTION_DEADZONE: f32 = 0.05; // |Δ| >= this -> fast regime
const KAL_Q_FAST: f32 = 0.25; // process noise on motion
const KAL_R_FAST: f32 = 0.5; // measurement noise on motion
const KAL_Q_SLOW: f32 = 0.001; // process noise in deadzone (heavy smoothing)
const KAL_R_SLOW: f32 = 20.5; // measurement noise in deadzone
const KAL_P0: f32 = 1.0; // initial error covariance

// --- native per-eye squeeze (RE'd via interleaved capture 2026-06-26) ---
// Under interleaved L+R feeding the model emits a genuine per-eye squeeze/closure
// channel (s3=left, s4=right): ~0.1 relaxed, rising past ~0.6 on a hard squeeze.
// Map [floor..floor+span] -> [0,1]. Floor cuts the relaxed baseline; tunable.
const NATIVE_SQ_FLOOR: f32 = 0.18;
const NATIVE_SQ_SPAN: f32 = 0.45;

/// Monotone piecewise-linear curve correction through (0,0), (a,0.5), (1,1):
/// the L/R mid-close equalizer (see the MID_ANCHOR constants). `apply_anchor(x,
/// 0.5)` is the identity bit-exactly (0.5*x/0.5 == x for normal f32), so an
/// untrained anchor changes nothing. Endpoints are fixed points for every
/// anchor, so blink-latch 0.0 and full-open 1.0 pass through unchanged.
fn apply_anchor(x: f32, a: f32) -> f32 {
    if x <= a {
        0.5 * x / a.max(1e-3)
    } else {
        0.5 + 0.5 * (x - a) / (1.0 - a).max(1e-3)
    }
}

/// 1-D scalar Kalman with two-regime (motion-adaptive) noise — a faithful port of
/// the real SRanipal openness smoother (`FUN_180010ee0`/`FUN_180010da0`): in the
/// deadzone it smooths hard (Q slow, R large); on motion it tracks fast. Opt-in via
/// [`Tuning::adaptive_kalman`]; our asymmetric EMA is the default. `x` = posterior
/// estimate, `p` = error covariance.
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

/// Live-adjustable post-processing parameters (exposed as calibration sliders).
/// Persisted in `sranibro.toml`'s `[tuning]` section; `serde(default)` fills any field a
/// missing/old config doesn't have, so partial configs load cleanly.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// EMA toward a more-open target (smaller = smoother/slower).
    pub alpha_open: f32,
    /// EMA toward a more-closed target (larger = snappier blink).
    pub alpha_close: f32,
    /// Master smoothing applied to BOTH directions: the per-direction speed is scaled
    /// by `(1 - smoothing)`, so one knob makes the whole eyelid softer/snappier while
    /// `alpha_open`/`alpha_close` set the open-vs-close balance. 0 = off (use the raw
    /// per-direction speeds, i.e. no extra smoothing); larger = smoother/slower both ways.
    pub smoothing: f32,
    /// Fraction of the squeeze range ignored before squeeze rises.
    pub squeeze_deadzone: f32,
    /// Maximum squeeze output (1.0 = uncapped).
    pub squeeze_gain: f32,
    /// Use the RE'd two-regime adaptive Kalman for openness smoothing instead of
    /// the asymmetric EMA (closer to real SRanipal; off by default to preserve feel).
    pub adaptive_kalman: bool,
    /// Use the model's NATIVE per-eye squeeze channel (s3/s4 under interleaved
    /// feeding) instead of deriving squeeze from openness. On by default — it is
    /// the genuine signal (RE'd 2026-06-26); turn off to A/B the old derivation.
    pub native_squeeze: bool,
    /// Wide output scale (1.0 = full). Lower it if wide is too sensitive; works
    /// together with the adaptive wide ceiling (which auto-sets the threshold).
    pub wide_gain: f32,
    /// Cross-eye baseline coupling: drift both eyes' baselines toward their mean.
    /// OFF by default — when the eyes' raw openness differs, a shared baseline makes
    /// the lower eye read half-closed (L/R asymmetry). Off = each eye self-calibrates
    /// to its OWN relaxed level, so both read symmetric while keeping winks.
    pub couple_eyes: bool,
    /// Adaptive blink bounds (BrokenEye-style): the openness=0 point is each eye's
    /// confirmed blink minimum (`closed_ref`), not a fixed offset. Endpoint changes
    /// require repeated matching slow closes and are tightly bounded. On by default.
    pub continuous_calib: bool,
    /// Wide is a BILATERAL gesture: it fires only when BOTH eyes are in wide
    /// territory, and then both eyes emit the SAME magnitude (the max of the two).
    /// A single eye widening emits no wide. This kills unreliable one-eye wide and
    /// makes bilateral wide L/R-symmetric — the weaker eye (left has poorer wide
    /// sensitivity) is pulled up to the stronger reading instead of lagging. On by
    /// default (user 2026-06-27). Off = legacy per-eye wide.
    pub wide_requires_both: bool,
    /// Cross-eye GAZE yoke. When one eye squints/closes, its own eye camera can no
    /// longer recover a reliable gaze (the lid hides the pupil), so the native gaze
    /// goes stale and the avatar's eye FREEZES pointing wherever it was when the
    /// squint began — look elsewhere and the squinting eye lags behind (a beta
    /// tester's only complaint, 2026-07-04). With this on, a closed eye MIRRORS the
    /// OTHER eye's gaze when that one is open and tracking, so it follows instead of
    /// freezing. The discriminator for "which eye lost tracking" is the openness/blink
    /// flag, NOT the gaze-valid bit (a shallow squint can keep gaze_valid while the
    /// value is already wrong). Both eyes closed (a real blink) → no yoke (nothing
    /// good to copy, and the lids are shut anyway). On by default.
    pub gaze_yoke: bool,
    /// wide/squeeze exclusivity — the ML-parameters "chain". wide (from the openness
    /// channel ch1) and squeeze (native ch3/ch4) are computed INDEPENDENTLY and both
    /// emitted every frame, so a noisy frame can light up BOTH at once. With this ON
    /// (chain closed) each attenuates the other in proportion to its own strength
    /// (`wide *= 1-squeeze`, `squeeze *= 1-wide`) so the dominant gesture wins smoothly
    /// (no hard branch → no flicker) and they never both fire. OFF by default (chain
    /// open) = today's independent behavior. Toggled by the chain glyph drawn between
    /// the wide and squeeze rows in the ML PARAMETERS card.
    pub wide_squeeze_exclusive: bool,
    /// Eye-open dead-zone: openness reads FULLY open (1.0) once raw is within this much
    /// BELOW the learned relaxed baseline (default 0.08; the old behavior was effectively
    /// 0.03). The baseline is the MEAN of the relaxed raw, so with too tight a margin about
    /// HALF of the relaxed noise dips under the full-open point and — because the ramp
    /// clamps at 1.0 above but slopes down below — the smoothed openness settles well below
    /// full. On a noisy / dim eye (typically the right hot-mirror camera) this read as
    /// "stuck ~half-open" (multi-user report 2026-07-04). Widen if an eye still won't open
    /// fully at rest; lower toward 0.03 for the old, tighter behavior.
    pub open_deadzone: f32,
    /// Minimum time (ms) a blink's REOPEN takes, as a slew limit on how fast
    /// openness may RISE while recovering from a full close (closing is never
    /// limited). 0 = off (instant recovery, the old behavior). VRChat-side
    /// avatar animations can miss a blink whose recovery is near-instant — a
    /// 100-200ms recovery keeps the closed pose visible long enough for the
    /// avatar to actually render it (user 2026-07-08).
    pub blink_reopen_ms: f32,
    /// Minimum time (ms) a blink's CLOSE takes: a slew limit on how fast openness may
    /// FALL. Mirror of `blink_reopen_ms`. With `close speed` maxed the blink close is
    /// near-instant ("teleport"); this eases it shut over at least this long — the gentle
    /// close a low-pass gave — WITHOUT the general smoothing. Only limits closes FASTER
    /// than the cap; a slow squint passes through. 0 = off (instant, the raw speed).
    pub blink_close_ms: f32,
    /* Removed: the former Simple mode and close-reach tuning.
    /// latch, so openness is just the plain per-eye ramp — raw normalized through the
    /// learned floor/ceiling, which is structurally what native SRanipal / BrokenEye do
    /// per eye. A robust fallback for the "one eye breaks" reports: SRanibro's extra
    /// per-eye machinery is the likeliest cause, and native proves the plain map is
    /// robust. Off by default. Baseline/closed_ref auto-range, wide, squeeze, and gaze
    /// yoke stay ON (native learns per-eye bounds too).
    /// Simple-mode close-point reachability (0..1): how far UP toward the open point the
    /// openness=0 point is raised, so a fast blink's shallower (under-sampled) bottom still
    /// hits 0 WITHOUT a velocity latch. 0 = the deep learned floor (fast blinks may not
    /// close); higher = closes more readily; 1 = binary (0 below the open point). Only used
     */
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
            open_deadzone: 0.08,
            blink_reopen_ms: 0.0,
            blink_close_ms: 0.0,
        }
    }
}

/// Per-eye mutable calibration + smoothing state.
#[derive(Clone, Copy, Debug)]
struct EyeState {
    baseline: f32,
    baseline_n: u32,
    upper: f32,
    /// Adaptive wide ceiling: absolute openness at which wide reaches 1.0.
    wide_ceiling: f32,
    /// Adaptive relaxed-s3 floor for native squeeze.
    squeeze_floor: f32,
    /// Closed (lower) bound for openness. Derived from the baseline (baseline −
    /// blink_depth) each frame ONLY while coupled (bootstrap / unlearned); once mature
    /// it is an INDEPENDENT absolute reference taught per clean episode (v3 — see the
    /// close-side decoupling block), so baseline drift cannot move the close floor.
    closed_ref: f32,
    /// Learned blink depth = baseline − blink bottom (persisted; a physiological
    /// per-user property). Updated per qualifying slow-blink episode.
    blink_depth: f32,
    /// Blink-episode tracking: frames since entry (0 = no episode), the
    /// episode-local smoothed raw, its running minimum, and whether the episode
    /// was disqualified (fast-blink latch involved / presence lost).
    blink_len: u32,
    blink_smooth: f32,
    blink_lo: f32,
    blink_dirty: bool,
    /// Close-side auto-range MIN-envelope of raw (present-gated; TRANSIENT — never
    /// persisted): the entry-gate + genuine-close reference for the per-eye floor.
    reach_env: f32,
    /// Whether a confirmed close pair has established `blink_depth`. Persisted.
    learned_once: bool,
    /// Minimum bottom of the current uncommitted close candidate set. Cold-start
    /// squints and partial closes are indistinguishable in one episode, so the
    /// floor always requires repeat evidence before moving.
    blink_candidate: Option<f32>,
    /// Number/timestamp for the transactional floor candidate. The candidate
    /// value is an ABSOLUTE raw bottom, not a baseline-relative depth.
    blink_candidate_count: u8,
    blink_candidate_max: f32,
    blink_candidate_frame: u32,
    /// Last endpoint commit and bounded travel for this runtime session.
    floor_last_commit_frame: u32,
    floor_travel_deeper: f32,
    floor_travel_shallower: f32,
    /// L/R mid-close curve equalizer: the LIVE anchor used by `apply_anchor`
    /// (0.5 = identity; persisted), a learned anchor staged to swap in at the
    /// next full-open frame (where the swap is output-invariant), this frame's
    /// PRE-correction ramp value (-1.0 = no clean ramp sample this frame), and
    /// the frame numbers of the current/last clean blink episode (entry, exit).
    mid_anchor: f32,
    mid_anchor_staged: Option<f32>,
    last_ramp_pre: f32,
    ep_entry_frame: u32,
    ep_clean_exit: Option<(u32, u32)>,
    frame_count: u32,
    open_streak: u32,
    closed_streak: u32,
    squeeze_streak: u32,
    is_closed: bool, // sticky blink (hysteresis)
    is_wide: bool,   // sticky wide (hysteresis)
    /// Stuck-wide breaker bookkeeping (runtime-only, not persisted): accumulated
    /// emit frames with `is_wide` held; leaks away on relaxed-open calm frames.
    wide_frames: u32,
    /// Current uninterrupted descent: total raw fallen and ML updates it took
    /// (for the blink-anchor speed test). Reset the moment raw turns up.
    fall_run: f32,
    fall_updates: u32,
    /// Frames since the last REAL down-step (draw < -0.001). Small only while
    /// actively descending — the curve-equalizer collector's anti-hold gate.
    since_down: u32,
    /// Previous-frame raw openness (for the fast-blink velocity latch).
    prev_raw: f32,
    /// Last Tobii absolute-openness state. A valid -> invalid transition is the
    /// authoritative close event; ML velocity decides whether it snaps or ramps.
    native_was_enabled: bool,
    /// Fast native close stays pinned until Tobii reports Enable again.
    native_disable_latch: bool,
    /// Fast-blink latch: frames the velocity latch has held openness at 0 (0 = not
    /// latched). Continuous mode only; snaps quick blinks fully closed.
    fast_blink_frames: u32,
    /// Fast-blink ARMED but not yet committed (emit frames held in the confirmation
    /// window; 0 = not armed). A fast drop was seen, but we defer emitting 0 until the
    /// window proves the descent is sustained, so a single saccade lid-dip / noisy
    /// sample that recovers doesn't flicker closed.
    fast_blink_arm: u32,
    /// Raw at the moment the fast-blink latch armed; the confirm test compares the
    /// next ML update against this level (recover above = transient dip, release).
    fast_blink_arm_raw: f32,
    /// Fresh ML updates seen while armed (the confirm window must include >= 1).
    arm_updates: u32,
    /// Lowest raw seen while latched (release = rise off this bottom) and the
    /// active hold cap for this commit (shallow commits get the tighter cap).
    latch_bottom: f32,
    latch_cap: u32,
    /// The last two DISTINCT ML raw values (duplicated-cadence-safe descent
    /// history; `rawu2 - raw` = net fall over the last 2 updates).
    rawu1: f32,
    rawu2: f32,
    /// Frames left before the fast-blink detector may trigger (see
    /// DETECT_GRACE_FRAMES).
    detect_grace: u32,
    /// Recovering from a full close: the post-blink reopen slew limit
    /// (Tuning::blink_reopen_ms) applies until openness is back near full.
    reopening: bool,
    /// Gaze-yoke hold (hysteresis; see YOKE_RELEASE_OPEN): while set, this eye's
    /// own gaze is treated as unreliable — its gaze EMA is frozen and the yoke
    /// mirrors the partner whenever the partner is open and tracking.
    yoke_hold: bool,
    kalman_x: f32,
    smooth_open: f32,
    /// Adaptive openness Kalman (used only when `Tuning.adaptive_kalman`).
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
            wide_ceiling: 0.0, // lazy: set to upper+min_span on first use
            squeeze_floor: NATIVE_SQ_FLOOR,
            closed_ref: 0.40, // = default baseline 0.60 - blink_depth; derived while coupled
            blink_depth: BLINK_OFFSET,
            blink_len: 0,
            blink_smooth: 0.40,
            blink_lo: 0.40,
            blink_dirty: false,
            reach_env: 0.60, // = default baseline; re-seeded on restore/recenter
            learned_once: false,
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
            kalman_x: 0.0,
            smooth_open: 1.0,
            kalman: ScalarKalman::default(),
            gaze_smooth: [0.0, 0.0, -1.0],
            pupil_smooth: 0.0,
        }
    }
}

/// Calibration snapshot persisted per user (baseline learning state).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalibSnapshot {
    pub baseline: f32,
    pub baseline_n: u32,
    pub frame_count: u32,
    /// Learned blink depth (baseline − blink bottom). `serde(default)` so calib
    /// files written before 2026-07-07 load cleanly with the stock depth.
    #[serde(default = "default_blink_depth")]
    pub blink_depth: f32,
    /// Learned L/R mid-close curve anchor (0.5 = identity). `serde(default)` so
    /// older calib files load cleanly with no correction.
    #[serde(default = "default_mid_anchor")]
    pub mid_anchor: f32,
    /// Whether a genuine close ever set `blink_depth` (so restore does not re-snap a
    /// learned depth via the first-episode direct-set). `serde(default)` = false for
    /// calib files written before this field existed.
    #[serde(default)]
    pub learned_once: bool,
}

fn default_blink_depth() -> f32 {
    BLINK_OFFSET
}

fn default_mid_anchor() -> f32 {
    0.5
}

/// Persisted calibration for both eyes (written to disk so the relaxed-open
/// baseline survives restarts — no re-warmup each session).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalibStore {
    pub left: CalibSnapshot,
    pub right: CalibSnapshot,
}

/// Load persisted calibration from a TOML file (None if absent or invalid).
pub fn load_calib(path: &str) -> Option<CalibStore> {
    let text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&text).ok()
}

/// Save calibration to a TOML file (best-effort; errors ignored).
pub fn save_calib(path: &str, store: &CalibStore) {
    if let Ok(text) = toml::to_string(store) {
        let _ = std::fs::write(path, text);
    }
}

/// Per-frame post-processor internals exposed to the diagnostic CSV recorder.
#[derive(Clone, Copy, Debug)]
pub struct DiagSnapshot {
    pub baseline: [f32; 2],
    pub closed_ref: [f32; 2],
    /// Pre-correction ramp value this frame (-1 = no clean ramp sample).
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
    /// Gaze-yoke hysteresis hold per eye (own gaze treated as unreliable).
    pub yoke_hold: [bool; 2],
}

/// Pending paired mid-close samples for the L/R curve equalizer (runtime-only):
/// weighted sums of each eye's pre-correction ramp value, collected during a
/// synchronized descent, committed only when BOTH eyes complete clean
/// synchronized slow-blink episodes.
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
    /// Bilateral-wide engage envelope [0,1] (shared across eyes): smooths the
    /// both-eyes gate so wide ramps in/out instead of snapping (see WIDE_GATE_STEP).
    wide_gate: f32,
    /// L/R curve-equalizer training pairs in flight (see PairPending).
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

    /// Snapshot for persistence (one per eye).
    pub fn snapshot(&self, e: Eye) -> CalibSnapshot {
        let s = &self.eyes[e.idx()];
        CalibSnapshot {
            baseline: s.baseline,
            baseline_n: s.baseline_n,
            frame_count: s.frame_count,
            blink_depth: s.blink_depth,
            mid_anchor: s.mid_anchor,
            learned_once: s.learned_once,
        }
    }

    /// Recenter: re-learn each eye's baseline from upcoming frames. The old
    /// baseline is kept as a small prior (RECENTER_PRIOR_N samples) so one noisy /
    /// mid-expression frame can't become the baseline verbatim; the bootstrap mean
    /// still re-anchors within ~1s. Keeps frame_count so the eyes stay trusted.
    pub fn recenter(&mut self) {
        self.wide_gate = 0.0;
        for s in self.eyes.iter_mut() {
            s.baseline_n = RECENTER_PRIOR_N;
            // Recenter pressed while a wide is LATCHED is the user asserting this
            // wide is spurious (a stuck episode looks like rest to the user):
            // pre-arm the breaker so the fast re-anchor starts immediately (~3-5s)
            // instead of waiting out the full 15s detection — the press IS the
            // detection. A healthy recenter (no latch) clears the counter; a
            // pre-armed one drains harmlessly within ~0.4s of relaxed frames if
            // the wide was somehow genuine (review 2026-07-07).
            s.wide_frames = if s.is_wide { WIDE_STUCK_FRAMES } else { 0 };
            s.fall_run = 0.0;
            s.fall_updates = 0;
            s.is_closed = false;
            s.is_wide = false;
            s.fast_blink_frames = 0;
            s.fast_blink_arm = 0;
            s.arm_updates = 0;
            // Keep the LIVE raw history (prev_raw/rawu*) and grant NO detector
            // grace: unlike new()/restore(), a mid-session recenter has genuine
            // per-frame history, so there is no phantom step to guard against --
            // and a graced detector would turn a natural blink in the 200ms
            // after the button press into a half-blink (review 2026-07-09).
            s.kalman_x = 0.0;
            s.kalman.reset();
            s.wide_ceiling = 0.0;
            s.squeeze_floor = NATIVE_SQ_FLOOR;
            // Blink depth and the mid-close curve anchor are physiological /
            // per-camera properties: KEEP them across recenter (only learning in
            // flight is abandoned) and re-derive the closed bound.
            s.blink_len = 0;
            s.mid_anchor_staged = None;
            s.ep_clean_exit = None;
            s.last_ramp_pre = -1.0;
            // reach_env is transient and baseline-relative: re-seed it at the current
            // baseline so the entry gate starts from the normal ~0.20 and re-learns this
            // eye's floor live (learned_once + blink_depth are kept — physiological).
            s.reach_env = s.baseline;
            s.blink_candidate = None;
            s.blink_candidate_count = 0;
            s.blink_candidate_max = 0.0;
            s.blink_candidate_frame = 0;
            // Recenter only changes the open coordinate. Keep the endpoint travel
            // budget so repeated Recenters cannot launder a bad floor adjustment
            // into blink_depth and grant another full budget.
            s.floor_last_commit_frame = s.frame_count;
            s.closed_ref = (s.baseline - s.blink_depth).clamp(s.baseline - 0.40, s.baseline - 0.05);
        }
        self.pend = PairPending::default();
    }

    /// Restore a persisted snapshot (loaded calibration).
    pub fn restore(&mut self, e: Eye, snap: CalibSnapshot) {
        let s = &mut self.eyes[e.idx()];
        s.baseline = snap.baseline;
        // A mature persisted baseline re-enters a SHORT bootstrap re-verification
        // (the value becomes a RESTORE_PRIOR_N-sample prior): confirmed in ~0.5s
        // when still accurate, re-anchored in ~1s when moderately stale (a reseat
        // between sessions). An immature snapshot keeps its own count — an idle
        // session persists baseline_n=0 and must not skip its cold start. Extreme
        // stale-LOW priors that the bootstrap cap rejects are still bounded: the
        // slow path + the stuck-wide breaker recover automatically within ~20s
        // (no zero-rate state anywhere — review 2026-07-06).
        s.baseline_n = snap.baseline_n.min(RESTORE_PRIOR_N);
        s.frame_count = snap.frame_count;
        // Restore the learned blink depth (sanitized: old calib files serde-default
        // it, and a corrupt value must not poison the ramp) and derive the closed
        // bound from it — as a baseline-relative OFFSET it stays valid even when
        // the baseline itself re-anchors.
        // Clamp to [0.05, 0.40] only: the old 0.20 FLOOR blocked a genuinely shallow
        // eye's learned sub-0.20 depth from surviving a restart. The detector's
        // relative gates stay well-behaved for small depths (their (NORM·d).clamp(...)
        // rails hold the thresholds up), so a shallow depth no longer degenerates them.
        s.blink_depth = if snap.blink_depth.is_finite() {
            snap.blink_depth.clamp(0.05, 0.40)
        } else {
            BLINK_OFFSET
        };
        // A real persisted (learned) depth must NOT be re-snapped by the first-episode
        // direct-set — treat it as already learned. Honor the persisted flag, and for
        // calib files that predate it, infer "learned" from a non-default depth.
        s.learned_once = snap.learned_once
            || (snap.blink_depth.is_finite()
                && (snap.blink_depth - default_blink_depth()).abs() > 1e-4);
        // Learned L/R curve anchor: sanitized like blink_depth (old files
        // serde-default it to identity).
        s.mid_anchor = if snap.mid_anchor.is_finite() {
            snap.mid_anchor.clamp(MID_ANCHOR_MIN, MID_ANCHOR_MAX)
        } else {
            0.5
        };
        s.mid_anchor_staged = None;
        s.ep_clean_exit = None;
        // Seed the detector's history at the restored baseline and hold its
        // trigger for a moment: the first real frame otherwise reads as a
        // phantom step from the seed (or from a stale baseline) and can commit
        // a fake blink at session start (golden-replay finding, 2026-07-09).
        s.prev_raw = s.baseline;
        s.rawu1 = s.baseline;
        s.rawu2 = s.baseline;
        s.detect_grace = DETECT_GRACE_FRAMES;
        // reach_env is transient (never persisted): re-seed at the restored baseline so
        // the entry gate starts from the normal ~0.20 and re-learns this eye's floor.
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

    /// Snapshot both eyes for persistence.
    pub fn snapshot_all(&self) -> CalibStore {
        CalibStore {
            left: self.snapshot(Eye::Left),
            right: self.snapshot(Eye::Right),
        }
    }

    /// Restore both eyes from a persisted store (loaded calibration).
    pub fn restore_all(&mut self, store: &CalibStore) {
        self.restore(Eye::Left, store.left);
        self.restore(Eye::Right, store.right);
        // The curve-equalizer's episode-sync test compares frame numbers ACROSS
        // eyes — a snapshot with mismatched per-eye counts would silently block
        // learning forever, so normalize them (review 2026-07-08).
        let fc = self.eyes[0].frame_count.max(self.eyes[1].frame_count);
        self.eyes[0].frame_count = fc;
        self.eyes[1].frame_count = fc;
    }

    fn update_calibration(&mut self, i: usize, raw: f32, present: bool) {
        // Coupling deliberately moves the mature baseline after this function.
        // Without an uncoupled coordinate snapshot, committing an endpoint would
        // bake that artificial displacement into persisted blink_depth.
        let endpoint_learning_enabled = !self.tuning.couple_eyes;
        let s = &mut self.eyes[i];
        s.frame_count += 1;
        // Stuck-wide breaker bookkeeping: accumulate while the sticky `is_wide` is
        // held; LEAK on relaxed-open calm frames (present, raw back inside the
        // absolute band) — a blink also releases is_wide but sits BELOW the band,
        // and a dropout isn't present, so a stuck episode accumulates right
        // through both, while any >=0.4s relaxed gap between genuine wide bursts
        // drains a full threshold's worth (proportional, no cliff — see
        // WIDE_DISCHARGE, review 2026-07-07).
        if s.is_wide {
            s.wide_frames = s.wide_frames.saturating_add(1);
        } else if present && raw > s.baseline - BLINK_RELEASE_OFFSET && raw < BASELINE_RANGE_HI {
            // Relaxed-open calm is BASELINE-relative (not the absolute band —
            // after the 2026-07-08 band widening, blink raws would land inside
            // an absolute test and wrongly discharge the breaker).
            s.wide_frames = s.wide_frames.saturating_sub(WIDE_DISCHARGE);
        }
        // Relaxed-baseline calibration. Bootstrap/recenter may learn; after that,
        // ordinary frames are frozen because a true level shift is not observable
        // separately from down-gaze or a sustained squint. The existing stuck-wide
        // breaker remains the one explicit automatic recovery path for a stale-low
        // reference; a real mid-session reseat otherwise needs one Recenter.
        if present && raw > BASELINE_RANGE_LO && raw < BASELINE_RANGE_HI {
            let stuck = s.is_wide && s.wide_frames >= WIDE_STUCK_FRAMES;
            if s.baseline_n < BASELINE_BOOTSTRAP_N {
                if s.baseline_n < BOOTSTRAP_ANCHOR_N || (raw - s.baseline).abs() < WIDE_LEARN_CAP {
                    // Bootstrap running mean; after the uncapped anchor phase the
                    // symmetric window keeps wide AND blink frames out of it.
                    s.baseline_n += 1;
                    s.baseline += (raw - s.baseline) / s.baseline_n as f32;
                } else {
                    // NO zero-rate state even here: a steady level ABOVE the cap
                    // (a stale-low prior + recenter, or recenter mid-squint then
                    // relax) still drifts the baseline up slowly, and the stuck
                    // breaker escalates to the fast rate. Without this branch the
                    // cap wedged the bootstrap forever — pressing Recenter during
                    // a spurious-wide episode made it WORSE (review 2026-07-06).
                    let rate = if stuck {
                        BASELINE_ADAPT
                    } else {
                        BASELINE_ADAPT * BASELINE_SLOW_FACTOR
                    };
                    s.baseline += rate * (raw - s.baseline);
                }
            } else {
                // A mature open reference is immutable during ordinary play.
                // Down-gaze/squint and small relaxed droop are observationally
                // indistinguishable, so even a slow EMA eventually learns the
                // expression and creates per-eye bias. Restore/Recenter already
                // re-enter bootstrap; the sole automatic exception is the explicit
                // stuck-wide recovery state for a stale-low reference.
                if stuck {
                    s.baseline += BASELINE_ADAPT * (raw - s.baseline);
                }
                s.baseline_n = s.baseline_n.saturating_add(1);
            }
        }
        let b = s.baseline;
        s.upper = b + UPPER_OFFSET;
        // Close-side auto-range: update the present-gated MIN-envelope of raw. It
        // attacks DOWN fast toward a new low and heals UP slowly, so it settles at this
        // eye's full-close bottom (squints ride above it). Only tracks raw within
        // ENV_SANITY_OFFSET of baseline, so a dropout / near-0 sample can't corrupt it.
        if present && raw > b - ENV_SANITY_OFFSET {
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
        // Blink-depth calibration, per EPISODE (v2 2026-07-07; see the constants
        // block). An episode spans raw dipping below the entry threshold until it
        // rises back above the release level. Only clean SLOW episodes teach:
        // anything the fast-blink latch touched had an under-sampled bottom (that
        // is exactly why the latch exists), a dropout mid-episode makes the bottom
        // untrustworthy, and the length gates reject under-sampled flickers and
        // resting-closed eyes. The ENTRY threshold is now per-eye relative (so a
        // shallow eye's real closes fire) and the genuine-close discriminator below
        // rejects a mid squint that trips the loose gate — so squints can't poison
        // the depth even though they may now enter an episode.
        let entry = b - (GATE_FRAC * (b - s.reach_env)).clamp(GATE_MIN, BLINK_OFFSET);
        let release = b - BLINK_RELEASE_OFFSET;
        if s.blink_len == 0 {
            if present && raw < entry && s.baseline_n >= BASELINE_BOOTSTRAP_N {
                s.blink_len = 1;
                s.blink_smooth = raw;
                s.blink_lo = raw;
                s.blink_dirty = s.fast_blink_frames > 0 || s.fast_blink_arm > 0;
                s.ep_entry_frame = s.frame_count; // for the curve-equalizer sync test
                                                  // Starting a NEW episode supersedes this eye's previous clean
                                                  // exit. Without this, the eye with the LONGER episode (it exits
                                                  // after its partner) leaves a stale exit that pairs with the
                                                  // partner's NEXT-gesture exit, fails the sync check, and
                                                  // discards every gesture's samples forever — observed live in
                                                  // the 2026-07-08 diagnostic recording (right exits ~0.3s after
                                                  // left, every blink's pend was thrown away).
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
                    // Genuine-close discriminator: the episode bottom must have reached
                    // the envelope floor. A mid squint that tripped the (now looser)
                    // gate but never reached the floor is rejected — it can neither
                    // ratchet the depth up nor train the curve equalizer.
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
                        // Confirm only after the episode has ended and its true bottom
                        // is known. Confirming mid-descent could lock onto a shallow
                        // candidate while a deeper real blink merely passes through it.
                        if let Some(candidate) = s.blink_candidate {
                            if (bottom - candidate).abs() <= FIRST_CLOSE_CONFIRM_TOL {
                                s.closed_ref = (0.5 * (candidate + bottom)).clamp(0.0, 1.0);
                                s.blink_depth = (b - s.closed_ref).clamp(0.05, 0.40);
                                s.learned_once = true;
                                s.blink_candidate = None;
                                s.blink_candidate_count = 0;
                                s.blink_candidate_max = 0.0;
                                s.floor_last_commit_frame = s.frame_count;
                                // Seed the now-INDEPENDENT close reference at the
                                // confirmed level (v3): from here on only clean
                                // episodes move it, never baseline drift.
                                s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                            } else {
                                s.blink_candidate = Some(bottom);
                                s.blink_candidate_count = 1;
                                s.blink_candidate_max = bottom;
                                s.blink_candidate_frame = s.frame_count;
                            }
                        } else {
                            // One episode cannot distinguish a cold-start squint from a
                            // genuinely shallow close, so stage it for confirmation.
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
                            // This is a credible full close at the already committed
                            // endpoint. It may train the mid-curve, but cannot drift
                            // either endpoint state.
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
                                    // `blink_depth` and `closed_ref` encode the same
                                    // endpoint in different coordinates. Commit them
                                    // atomically so Recenter cannot re-import an
                                    // unconfirmed squint through blink_depth.
                                    s.blink_depth = (b - s.closed_ref).clamp(0.05, 0.40);
                                    s.floor_last_commit_frame = s.frame_count;
                                    s.ep_clean_exit = Some((s.ep_entry_frame, s.frame_count));
                                }
                                s.blink_candidate = None;
                                s.blink_candidate_count = 0;
                                s.blink_candidate_max = 0.0;
                            }
                        } else {
                            // A merely lazy/incomplete close cannot prove that the
                            // true floor moved. Do not let it accumulate evidence.
                            s.blink_candidate = None;
                            s.blink_candidate_count = 0;
                            s.blink_candidate_max = 0.0;
                        }
                    }
                }
                s.blink_len = 0;
            }
        }
        // Close-side reference maintenance (v3 — see the decoupling block by the
        // constants). COUPLED phase (bootstrap in progress, or no confirmed close
        // yet): derive from the baseline each frame, exactly the old behavior — it
        // follows a recenter/restore re-anchor (their bootstrap re-verification
        // re-enters this phase) and gives an unlearned eye a sane floor. MATURE
        // phase: closed_ref is independent — the episode teaching above is the only
        // writer, so baseline drift no longer moves the openness=0 point.
        if !s.learned_once || s.baseline_n <= BASELINE_BOOTSTRAP_N {
            s.closed_ref = (b - s.blink_depth).clamp(0.0, 1.0);
        } else if !s.closed_ref.is_finite() {
            s.closed_ref = (b - s.blink_depth).clamp(0.0, 1.0);
        } else {
            s.closed_ref = s.closed_ref.clamp(0.0, 1.0);
        }
    }

    /// The derived openness=0 point (blink bottom) for an eye — diagnostics.
    pub fn closed_ref(&self, e: Eye) -> f32 {
        self.eyes[e.idx()].closed_ref
    }

    /// Per-frame post-processor internals for the diagnostic CSV recorder (the
    /// dashboard REC button): one row of this next to the raw ml values is
    /// enough to reconstruct offline why any output came out the way it did.
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

    /// Region decode -> (openness, wide, squeeze). Mirrors `normalize` exactly.
    fn normalize(
        &mut self,
        i: usize,
        raw: f32,
        in2: f32,
        native_absolute: bool,
    ) -> (f32, f32, f32) {
        let tuning = self.tuning;
        let s = &mut self.eyes[i];
        // No clean ramp sample unless the continuous-ramp branch below produces
        // one (wide / latch / legacy early-returns leave the sentinel).
        s.last_ramp_pre = -1.0;
        let b = s.baseline;
        let upper = b + UPPER_OFFSET;
        let squeeze_top = b - SQUEEZE_TOP_OFFSET;
        // Openness saturates to 1.0 at `open_full` — a wider dead-zone below baseline than
        // squeeze_top. The baseline is the MEAN relaxed raw, so a too-tight full-open point
        // let ~half the relaxed noise read < 1.0 and the smoothed openness settled well
        // below full (the "one eye stuck ~half-open" report, 2026-07-04). squeeze_top is
        // kept as-is for the wide / fast-blink-latch guards (those are about narrowing
        // BELOW relaxed, not about where the eye counts as fully open).
        let open_full = b - tuning.open_deadzone.clamp(SQUEEZE_TOP_OFFSET, 0.20);
        // openness=0 point: per-eye learned blink minimum (continuous calib) or the fixed
        // offset. Kept just below `open_full` so the openness-ramp denominator is +ve.
        let blink_top = if tuning.continuous_calib {
            s.closed_ref.min(open_full - 1e-3)
        } else {
            b - BLINK_OFFSET
        };
        let blink_release = b - BLINK_RELEASE_OFFSET;

        // Blink hysteresis (legacy): stay closed until raw rises above blink_release.
        // Continuous-calib skips this sticky HOLD so reopen tracks smoothly (the
        // hold + a flat-0 ramp region were the dead-zone / catch-up jump on reopen).
        if !tuning.continuous_calib && s.is_closed {
            if raw < blink_release {
                s.squeeze_streak = 0;
                s.kalman_x *= 0.5;
                return (0.0, 0.0, 0.0);
            }
            s.is_closed = false; // exit blink, fall through
        }

        // Any clearly-open raw cancels a pending fast-blink latch. Runs BEFORE the
        // wide block below (which early-returns) so a blink that reopens straight
        // into wide can't leave a stale latch that later forces openness back to 0
        // when raw settles down out of wide. A real blink descent
        // sits below squeeze_top, so this never fires mid-blink.
        if raw >= squeeze_top {
            s.fast_blink_frames = 0;
            s.fast_blink_arm = 0;
        }

        // Adaptive wide ceiling: wide reaches 1.0 only at the user's OBSERVED max
        // openness (auto-calibrated threshold — the user's ask). A minimum span
        // above `upper` caps how touchy wide can get; the ceiling relaxes slowly
        // back toward that floor so a one-off peak doesn't desensitize forever.
        let ceil_floor = upper + WIDE_CEIL_MIN_SPAN;
        if s.wide_ceiling < ceil_floor {
            s.wide_ceiling = ceil_floor;
        } else if raw > s.wide_ceiling {
            s.wide_ceiling = raw; // ratchet up to a new observed max
        } else {
            s.wide_ceiling += WIDE_CEIL_DECAY * (ceil_floor - s.wide_ceiling);
        }
        let wide_span = (s.wide_ceiling - upper).max(1e-3);
        let wide_gain = tuning.wide_gain;

        // Wide hysteresis (mirror of blink): stay wide until raw < upper - release.
        let wide_release = upper - WIDE_RELEASE_OFFSET;
        if s.is_wide {
            if raw < wide_release {
                s.is_wide = false; // exit wide, fall through
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

        // Continuous openness ramp: closed_ref (0) -> squeeze_top (1) as a single
        // straight line — no flat-0 region, no sticky hold — so closing AND reopening
        // track smoothly with no dead-zone or catch-up jump. Blink-to-0 still happens
        // (opn hits 0 at/below closed_ref; the emit stage zeroes openness near 0).
        if tuning.continuous_calib {
            s.kalman_x *= 1.0 - KALMAN_ALPHA;
            // Fast-blink detector v2 (see the constants block for the data-derived
            // design): per-eye evidence normalized by the learned blink_depth,
            // evaluated per DISTINCT ML update, released by rise-off-bottom.
            let d = s.blink_depth;
            let is_upd = (raw - s.prev_raw).abs() > 1e-6;
            if s.detect_grace > 0 {
                s.detect_grace -= 1;
            }
            if native_absolute {
                // Simplest / native path: no fast-blink latch — the plain ramp handles
                // the close (raw dips below closed_ref -> 0). Clear any latch a toggle left.
                s.fast_blink_frames = 0;
                s.fast_blink_arm = 0;
            } else if s.fast_blink_frames > 0 {
                // Committed (emitting 0). Track the true bottom; release only when
                // raw rises meaningfully off it AND actually leaves the closed
                // zone — bottom wobble and the hard-close overshoot-recovery stay
                // pinned at 0 (the "bounce right after closing" report). The hold
                // cap applies only while SHALLOW: a deep hold is a genuinely
                // closed eye (held wink) and reads 0 until a real reopen.
                s.latch_bottom = s.latch_bottom.min(raw);
                let rise = (RELEASE_RISE_NORM * d).max(RELEASE_RISE_FLOOR);
                let exit_zone = raw > blink_top + (LATCH_EXIT_NORM * d).clamp(0.03, 0.09);
                let deep_now = (b - raw) >= SHALLOW_COMMIT_DIP * d;
                let cap_hit = s.fast_blink_frames >= s.latch_cap && !deep_now;
                if (raw > s.latch_bottom + rise && exit_zone) || cap_hit {
                    // Release -> fall through to the ramp, and RESET the descent
                    // evidence: a genuine re-close must earn >= 2 fresh updates,
                    // so stale evidence can never re-latch at the bottom (the
                    // recorded 17ms re-entry flickers).
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
                // Armed but not committed -- emit the ramp while the confirmation
                // window proves the descent:
                //  * raw recovers ABOVE the armed level -> transient dip (saccade
                //    lid-dip / one noisy sample): disarm, no 0 ever emitted.
                //  * window done (>= ARM_FRAMES emit frames AND >= 1 fresh ML
                //    update, or the ARM_MAX blind fallback) with raw still at/below
                //    the armed level -> a real fast blink: commit.
                if raw > s.fast_blink_arm_raw {
                    s.fast_blink_arm = 0; // recovered -> no flicker, fall to the ramp
                } else {
                    if is_upd {
                        s.arm_updates = s.arm_updates.saturating_add(1);
                    }
                    let window_done = s.fast_blink_arm >= FAST_BLINK_ARM_FRAMES
                        && (s.arm_updates >= 1 || s.fast_blink_arm >= FAST_BLINK_ARM_MAX);
                    if window_done {
                        // Commit. Shallow bottoms get the tighter hold cap.
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
                    s.fast_blink_arm += 1; // keep waiting; emit ramp this frame
                }
            } else if is_upd && raw < squeeze_top && s.detect_grace == 0 {
                // Trigger: single-update velocity (any depth -- weak-eye blinks
                // bottom shallow) OR cumulative 2-update fall (depth-gated so a
                // brisk squint onset can't accumulate into a phantom blink).
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
            // Ramp denominator floored (MIN_CLOSE_SPAN): a shallow learned depth minus a
            // wide open_deadzone left a razor span (0.04 measured) that amplified every
            // raw error ~25x into openness. The floor keeps the ZERO point at blink_top
            // (full closes still read 0) and only relaxes the slope.
            let denom = (open_full - blink_top).max(MIN_CLOSE_SPAN);
            let opn_pre = ((raw - blink_top) / denom).clamp(0.0, 1.0);
            if s.fast_blink_arm == 0 {
                // Clean pre-correction ramp sample for the curve-equalizer
                // learner (armed frames are mid-descent ambiguity — excluded).
                s.last_ramp_pre = opn_pre;
            }
            // A learned anchor swaps in only at full-open, where apply_anchor(1,a)
            // == 1 for every a — the swap is exactly output-invariant.
            if let Some(a) = s.mid_anchor_staged {
                if opn_pre >= 0.99 {
                    s.mid_anchor = a;
                    s.mid_anchor_staged = None;
                }
            }
            // L/R mid-close curve equalization (identity until learned); skipped in
            // simple mode so a mis-learned per-eye anchor can't warp one eye.
            let opn = if native_absolute {
                opn_pre
            } else {
                apply_anchor(opn_pre, s.mid_anchor)
            };
            // Derived squeeze (only used when native_squeeze is off): rises as the
            // eye narrows below relaxed.
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
            // Apply wide_gain + cap on the decay tail too.
            return (1.0, (s.kalman_x.max(0.0) * wide_gain).min(1.0), 0.0);
        }

        if raw > blink_top {
            // Squeeze territory (below the open dead-zone).
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

        // Blink: enter sticky closed state.
        s.is_closed = true;
        s.squeeze_streak = 0;
        s.kalman_x *= 0.5;
        (0.0, 0.0, 0.0)
    }

    /// Cross-eye live calibration: drift L/R baselines toward their mean when
    /// both eyes are stable+open in the learning band. Call once per frame.
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

    /// Full per-frame post-process. `ml` is the full per-eye model output
    /// `[[ch0..ch4]; 2]` (L, R). ch1 = openness (the primary signal, used as the
    /// `raw` below); ch0 = presence/confidence gate; ch3 = native per-eye squeeze.
    /// `gaze` is the native sample. Returns [Left, Right].
    ///
    /// Channel layout (RESOLVED via interleaved capture 2026-06-26, winks isolate
    /// each eye): the pipeline now feeds the model ONE interleaved pass (L in ch0,
    /// R in ch1) and packs the per-eye view as `[presence, openness, _, squeeze, _]`.
    /// So `ml[i] = [s0 presence, s1|s2 openness, _, s3|s4 squeeze, _]`. Openness
    /// (s1/s2) carries the wide range above the relaxed baseline; squeeze (s3/s4)
    /// is a genuine native channel (no derivation needed) — see `Tuning.native_squeeze`.
    pub fn process_frame(
        &mut self,
        ml: [[f32; 5]; 2],
        gaze: &GazeSample,
        ml_loaded: bool,
    ) -> [EyeResult; 2] {
        let raws = [ml[0][1], ml[1][1]]; // ch1 = openness
        let mut pe = [PerEye::default(); 2];

        for e in Eye::ALL {
            let i = e.idx();
            let raw = raws[i];
            // ch0 presence gate: eye lost (model returns ~0) -> force closed.
            let present = ml[i][0] > CH0_PRESENT_GATE;

            // Step 1: online calibration.
            self.update_calibration(i, raw, present);

            // Step 2: blink gate + streaks.
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
            let native_absolute = gaze.eye(e).openness_reported;

            // Step 3+4: region decode (or hold smoothed value until trusted).
            let (opn, wide_mag, sqz_mag) = if trusted {
                self.normalize(i, raw, raw, native_absolute)
            } else {
                // Untrusted (warmup): normalize() is skipped, so clear the sticky
                // is_wide flag — otherwise a stale `true` (set when this eye was
                // briefly trusted, then blinked back to untrusted before frame 200)
                // would make the bilateral-wide gate read true && true and leak wide
                // while this eye is actually closed.
                self.eyes[i].is_wide = false;
                self.eyes[i].fast_blink_frames = 0;
                self.eyes[i].fast_blink_arm = 0;
                self.eyes[i].last_ramp_pre = -1.0; // normalize() skipped: no ramp sample
                                                   // Hold the smoothed value, but still force 0 on a clear close so
                                                   // blinks register before trust is reached.
                let held = if raw < self.eyes[i].baseline - BLINK_OFFSET {
                    0.0
                } else {
                    self.eyes[i].smooth_open
                };
                (held, 0.0, 0.0)
            };

            // Native per-eye squeeze (s3=left/s4=right) when enabled, else the
            // openness-derived magnitude from normalize(). The floor auto-calibrates
            // to each user's relaxed s3 while the eye is open; `squeeze_gain` is the
            // sensitivity knob.
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
            // Per-frame raw delta (prev_raw is overwritten below).
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
                // Blink flag: in continuous mode tie it to the openness curve itself
                // (consistent with the forced-0 gate) instead of the fixed threshold.
                closed: (if self.tuning.continuous_calib {
                    opn < 0.08
                } else {
                    raw < blink_threshold
                }) || !present,
                present,
                // Not rising (<= +0.005 tolerates the duplicated-sample cadence,
                // where every other emit frame has draw exactly 0 mid-descent).
                falling: draw <= 0.005,
                draw,
            };

            // Track the current uninterrupted descent for the blink-anchor speed
            // test: accumulate on real down-steps (duplicated-sample frames have
            // draw == 0 and count toward neither), reset the moment raw turns up.
            // `since_down` ages every frame without a real down-step, so a static
            // hold or a reopen stops counting as "descending" within ~2 updates.
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

            // Remember this frame's raw for the next frame's fast-blink velocity.
            self.eyes[i].prev_raw = raw;
        }

        // Step 4.3: Tobii absolute-openness classifier. The native validity transition is
        // the authoritative "lid fully closed" event, while the unfiltered ML descent tells
        // us HOW it closed:
        //   * slow Disable: keep following the continuous ML ramp;
        //   * fast Disable: snap this eye to zero and hold until native Enable returns.
        // A bilateral fast event snaps both eyes together. Adapters without a native
        // openness field retain the ML-only fallback detector in `normalize`.
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
                pe[i].closed = true;
            }
        }

        // Step 4.4: bilateral fast-blink ASSIST (v2). A real blink is bilateral and
        // one eye's detection can trail the other's: when the PARTNER has
        // committed and this eye is itself clearly mid-blink by its OWN relative
        // evidence (deep by ASSIST_DIP_NORM x its blink_depth AND recently fallen
        // by ASSIST_FALL2_NORM x depth), commit it too. There is deliberately NO
        // unlatched anchor: two sub-threshold eyes can never bootstrap, so
        // bilateral squints are structurally safe, and a wink's open eye (worst
        // measured sympathetic droop 0.31 x depth, and slow) can never qualify.
        // Each committed eye releases via its own rise-off-bottom logic.
        if self.tuning.continuous_calib
            && !gaze.left.openness_reported
            && !gaze.right.openness_reported
        {
            for e in Eye::ALL {
                let (i, j) = (e.idx(), e.opposite().idx());
                if self.eyes[i].fast_blink_frames > 0 && self.eyes[j].fast_blink_frames == 0 {
                    let d = self.eyes[j].blink_depth;
                    let dip = self.eyes[j].baseline - pe[j].raw;
                    // History still excludes this frame (shifted after this step),
                    // so this spans exactly the last 2 distinct ML updates.
                    let fall2 = self.eyes[j].rawu2 - pe[j].raw;
                    if pe[j].present
                        && pe[j].falling
                        && dip >= ASSIST_DIP_NORM * d
                        && fall2 >= ASSIST_FALL2_NORM * d
                    {
                        let sj = &mut self.eyes[j];
                        sj.fast_blink_frames = 1;
                        sj.fast_blink_arm = 0; // stale-arm fix: a commit consumes any arm
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
        // Shift the distinct-ML-update history AFTER the assist (the fall2 tests
        // above and inside normalize must span the last 2 updates EXCLUDING the
        // current frame; the duplicated cadence makes per-emit deltas unusable).
        for i in 0..2 {
            if pe[i].draw.abs() > 1e-6 {
                self.eyes[i].rawu2 = self.eyes[i].rawu1;
                self.eyes[i].rawu1 = pe[i].raw;
            }
        }

        // Step 4.6: L/R mid-close curve-equalizer LEARNER (runtime output is the
        // static per-eye apply_anchor map in normalize; this step only gathers
        // statistics — none of its gates can affect the emitted frame). During a
        // synchronized bilateral descent, collect paired pre-correction ramp
        // values weighted toward the half-closed point; when BOTH eyes then
        // complete clean synchronized slow-blink episodes (the same qualifier
        // that teaches blink_depth — one deliberate gesture teaches both), blend
        // each eye's weighted mean into its anchor, CENTER the pair (the average
        // curve stays identity, so global close feel never drifts), and stage
        // the swap for the next full-open frame. Winks can never train it: the
        // open eye's ramp value is pinned at 1.0 (outside the sample window),
        // wink pairs differ by ~1.0 (PAIR_MAX_DIFF), and a wink never yields two
        // synchronized clean episodes (EP_SYNC_SKEW).
        if self.tuning.continuous_calib
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
                    // ACTIVE descent only (anti-hold + anti-reopen): a static
                    // hold stops within ~2 ML updates, and its up/down noise
                    // jitter is sampled symmetrically (no selection bias — the
                    // gate is "recently stepped down", not "this frame fell").
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
            // COMMIT — evaluated ONCE per episode pair: the moment both eyes hold
            // clean exits, the pend either commits or is discarded, and the exits
            // are consumed either way (an under-weighted or out-of-window pend
            // must not linger armed and pair with unrelated later samples —
            // review 2026-07-08). The samples must also lie inside the episodes'
            // own time window: started no earlier than CURVE_PRE_WINDOW before
            // entry (rejects an earlier unrelated descent/hold riding this
            // blink) and none collected after the last exit.
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
                    // Pin the pair-mean curve to identity: only the DIFFERENCE
                    // between the eyes is corrected, never the shared feel.
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
            // EXPIRE stale or abandoned pending samples (a close that never
            // completed clean synchronized episodes must not linger).
            if self.pend.w_sum > 0.0 {
                self.pend.age = self.pend.age.saturating_add(1);
                let abandoned = pe[0].raw > self.eyes[0].baseline - 0.05
                    && pe[1].raw > self.eyes[1].baseline - 0.05;
                if self.pend.age > PAIR_STALE_FRAMES || abandoned {
                    self.pend = PairPending::default();
                }
            }
        }

        // Step 4.5: cross-eye baseline coupling (opt-in — off by default so each
        // eye self-calibrates and L/R openness stays symmetric; see Tuning).
        if self.tuning.couple_eyes {
            self.couple_baselines(pe[0].raw, pe[1].raw);
        }

        // Squeeze is a closed-eye expression, not an alternate openness signal.
        // Gate the native channel before the wide/squeeze chain so open-eye noise
        // can neither leak to EyeSquint nor attenuate a genuine EyeWide reading.
        // Tracking loss also reports closed, but is not a real squeeze gesture.
        for eye in &mut pe {
            if !eye.closed || !eye.present {
                eye.squeeze_out = 0.0;
            }
        }

        // Step 5: cross-eye yoke for wide & squeeze (only when bilateral). Skip the
        // squeeze yoke under native squeeze — s3/s4 are already per-eye accurate
        // (confirmed by single-eye winks), so forcing symmetry would break winks.
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
            // Wide is a BILATERAL gesture (user 2026-06-27): fire only when BOTH
            // eyes are in wide territory, then emit the SAME magnitude to both.
            // Gate on the per-eye `is_wide` flag (raw > upper, a low +0.02 bar) — NOT
            // the wide magnitude — so the weaker LEFT eye (poor wide sensitivity)
            // still counts toward "both", and `max` pulls it up to the stronger
            // reading. A single eye widening is suppressed (片目だけでeyewide廃止).
            //
            // The gate is a smooth envelope, NOT a hard on/off: the leading eye's
            // per-eye magnitude charges up internally while it waits for the lagging
            // eye, so opening the gate hard would DUMP that charge in one frame (the
            // eye "snapped" wide — user 2026-06-27). Ramping wide = max(L,R) *
            // wide_gate from 0 removes the snap on engage and the snap-off on
            // disengage.
            let both_wide = self.eyes[0].is_wide && self.eyes[1].is_wide;
            // Don't bleed the wide envelope down for a BLINK. A blink clears both
            // eyes' is_wide, which would drive the gate toward 0 the whole time the
            // eyes are closed; on reopen straight back into wide the user sees wide
            // "swell in" over the re-engage ramp. While either eye is closed, HOLD
            // the gate; resume the slew once both eyes are open. Safe: per-eye
            // wide_out is 0 during a blink, so a held gate leaks no wide while closed.
            let blinking = pe[0].closed || pe[1].closed;
            let gate_target = if both_wide { 1.0 } else { 0.0 };
            if blinking {
                // hold wide_gate (don't decay during a blink-while-wide)
            } else if self.wide_gate < gate_target {
                self.wide_gate = (self.wide_gate + WIDE_GATE_STEP).min(gate_target);
            } else if self.wide_gate > gate_target {
                self.wide_gate = (self.wide_gate - WIDE_GATE_STEP).max(gate_target);
            }
            // Eyes shut -> emit no wide (and don't leak the decaying per-eye
            // kalman_x through the held gate). The frozen gate still snaps wide
            // back to full the instant the eyes reopen.
            let m = if blinking {
                0.0
            } else {
                pe[0].wide_out.max(pe[1].wide_out) * self.wide_gate
            };
            pe[0].wide_out = m;
            pe[1].wide_out = m;
        } else {
            // Legacy: per-eye wide, symmetrize only when both magnitudes are up.
            let both_widening = pe[0].wide_out > 0.05 && pe[1].wide_out > 0.05;
            if both_widening {
                let m = pe[0].wide_out.max(pe[1].wide_out);
                pe[0].wide_out = m;
                pe[1].wide_out = m;
            }
        }

        // Step 5.5: wide/squeeze exclusivity (the ML-parameters "chain"). wide and
        // squeeze come from different channels and are otherwise independent, so a
        // noisy frame can show both at once. When linked, each attenuates the other in
        // proportion to its own strength — the dominant gesture wins smoothly, and a
        // genuine wide no longer bleeds a little squeeze (or vice-versa). Uses the
        // pre-attenuation magnitudes for BOTH sides so it's symmetric (order-free).
        if self.tuning.wide_squeeze_exclusive {
            for i in 0..2 {
                let (w, s) = (pe[i].wide_out, pe[i].squeeze_out);
                pe[i].wide_out = w * (1.0 - s).clamp(0.0, 1.0);
                pe[i].squeeze_out = s * (1.0 - w).clamp(0.0, 1.0);
            }
        }

        // Step 6: asymmetric EMA on openness + adaptive gaze/pupil EMA -> emit.
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for e in Eye::ALL {
            let i = e.idx();
            let s = pe[i];
            let target = s.openness_target;
            let prev = self.eyes[i].smooth_open;
            let gaze_invalid = !s.gaze_v;
            // Raw-domain: map- and noise-invariant (see FAST_CLOSING_RAW — the
            // old corrected-target test flapped on ordinary mid-close jitter).
            let fast_closing = s.draw <= FAST_CLOSING_RAW;
            // In continuous mode, blink-to-0 is driven by the continuous openness
            // itself (target<0.05 at/near closed_ref) — NOT the fixed baseline-0.20
            // streak gate, which sits above closed_ref and would re-introduce the
            // reopen dead-zone/jump.
            let streak_gate = !self.tuning.continuous_calib && self.eyes[i].closed_streak >= 2;
            let forced_closed = streak_gate
                || target < 0.05
                || !s.present
                || (gaze_invalid && prev > 0.4 && fast_closing);
            let mut openness_out = if forced_closed {
                // Unconditional reset: avoids a stale openness jump if adaptive
                // mode is toggled across a forced-closed period.
                self.eyes[i].kalman.reset();
                0.0
            } else if self.tuning.adaptive_kalman {
                self.eyes[i].kalman.update(target)
            } else {
                // Per-direction speed, then the master `smoothing` damps BOTH equally
                // (floored so it never fully freezes). smoothing=0 -> the raw per-dir rate.
                let base = if target < prev {
                    self.tuning.alpha_close
                } else {
                    self.tuning.alpha_open
                };
                let alpha_o =
                    (base * (1.0 - self.tuning.smoothing.clamp(0.0, 1.0))).clamp(0.03, 1.0);
                prev + alpha_o * (target - prev)
            };
            // Post-blink reopen slew limit (Tuning::blink_reopen_ms): while
            // recovering from a full close, cap how fast openness may RISE so
            // the closed pose stays visible long enough for the receiving
            // avatar's animation (VRChat can miss a near-instant recovery).
            // Closing is never limited; normal (non-blink) motion is untouched.
            if forced_closed {
                self.eyes[i].reopening = true;
            } else if self.eyes[i].reopening {
                let ms = self.tuning.blink_reopen_ms;
                if ms >= 1.0 {
                    let step = 1000.0 / (120.0 * ms); // full range per emit frame
                    if openness_out > prev + step {
                        openness_out = prev + step;
                    }
                }
                if openness_out >= 0.9 {
                    self.eyes[i].reopening = false; // recovered — normal motion
                }
            }
            // Blink-close slew limit (Tuning::blink_close_ms): mirror of the reopen limit
            // — cap how fast openness may FALL so a teleport-fast blink close eases shut
            // over at least this long (the gentle close a low-pass gave), independent of
            // `smoothing`. Only bites when the close is FASTER than the cap; a slow squint
            // falls slower than `step` and passes through untouched. 0 = off (instant).
            let close_ms = self.tuning.blink_close_ms;
            if close_ms >= 1.0 {
                let step = 1000.0 / (120.0 * close_ms); // full range per emit frame
                if openness_out < prev - step {
                    openness_out = prev - step;
                }
            }
            self.eyes[i].smooth_open = openness_out;

            // Gaze-yoke hysteresis: engage BEFORE XR5 loses the pupil under a shallow
            // squint, or after the native gaze has genuinely aged invalid. `fresh_gaze`
            // already absorbs short invalid packets, so `!gaze_v` here means the dropout
            // outlived that grace. Release only after both a wide reopen and own-gaze
            // recovery; this prevents the old own/partner source flapping at the boundary.
            if self.tuning.gaze_yoke {
                if s.closed || openness_out < YOKE_ENGAGE_OPEN || !s.gaze_v {
                    self.eyes[i].yoke_hold = true;
                } else if openness_out > YOKE_RELEASE_OPEN && s.gaze_v {
                    self.eyes[i].yoke_hold = false;
                }
            } else {
                self.eyes[i].yoke_hold = false;
            }

            // Adaptive gaze EMA (alpha scales with motion). Frozen while the
            // yoke hold is active: mid-squint the eye's OWN gaze is garbage
            // (the lid hides the pupil while gaze_valid can stay true), and
            // ingesting it between yoked frames was the gaze-trembling bug.
            let [gx, gy, gz] = s.gaze;
            let (sx, sy, sz) = if s.gaze_v && !self.eyes[i].yoke_hold {
                let [px, py, pz] = self.eyes[i].gaze_smooth;
                let delta = ((gx - px).powi(2) + (gy - py).powi(2) + (gz - pz).powi(2)).sqrt();
                let alpha_g = (delta * 3.0).clamp(GAZE_ALPHA_CALM, 0.85);
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
                gaze_yoked: false, // may be set by the cross-eye gaze yoke below
                brow: 0.0,         // set by the pipeline's brow pass (if a model is loaded)
                brow_valid: false,
            };
        }

        // Step 7: cross-eye GAZE yoke. A squinting/closed eye can't recover a
        // reliable gaze from its own camera (the lid hides the pupil), so on its own
        // the avatar's eye FREEZES where it was when the squint began and lags when
        // you look away (beta-tester report 2026-07-04). The signal for "which eye
        // lost tracking" is the openness/blink flag (NOT the gaze-valid bit — a
        // shallow squint keeps gaze_valid while the value is already wrong). So when
        // exactly one eye is closed while the OTHER is open and tracking, mirror the
        // open eye's smoothed gaze onto the closed one and keep it valid, so the
        // squint eye FOLLOWS the tracked eye. Both eyes closed (real blink) → neither
        // branch fires (each partner is also blink) → no yoke. `gaze_yoked` lets the
        // sinks emit this gaze despite blink==true (they otherwise gate gaze on !blink).
        if self.tuning.gaze_yoke {
            for e in Eye::ALL {
                let (i, j) = (e.idx(), e.opposite().idx());
                // Mirror while the HYSTERESIS hold is active (engaged at blink,
                // released only after a genuine reopen) — not the raw blink flag,
                // which flaps at its 0.08 boundary mid-squint. The partner must
                // NOT itself be holding: after a bilateral blink both eyes hold
                // through the reopen, and without this exclusion they mirrored
                // EACH OTHER (order-dependent gaze cross-copy on every natural
                // blink — the residual trembling, user 2026-07-08).
                if self.eyes[i].yoke_hold
                    && !self.eyes[j].yoke_hold
                    && !out[j].blink
                    && out[j].gaze_valid
                {
                    out[i].gaze = out[j].gaze;
                    out[i].gaze_valid = true;
                    out[i].gaze_yoked = true;
                    // Anchor this eye's own smoother to the partner so it resumes from
                    // the mirrored direction (no snap) when the squint releases.
                    self.eyes[i].gaze_smooth = out[j].gaze;
                }
            }
        }
        out
    }
}

/// Per-eye scratch between the decode pass and the emit pass.
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
    /// raw was not rising this frame (for the bilateral fast-blink yoke).
    falling: bool,
    /// This frame's raw delta (for the raw-domain fast-closing test).
    draw: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(st: &mut SRanipalState, raw: f32, n: usize) -> [EyeResult; 2] {
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        // ch0 = 1.0 (eye present), ch1 = raw openness; other channels unused here.
        let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
        for _ in 0..n {
            last = st.process_frame(ml, &g, true);
        }
        last
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
        let reopened = native_step(&mut st, 0.62, 0.62, true, true);
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

        // Warm up at a relaxed-open raw (in the baseline band) -> trusted + open.
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

        // Close the eyes (raw well below baseline-0.20 blink threshold).
        let closed = feed(&mut st, 0.30, 12);
        assert_eq!(closed[0].openness, 0.0, "blink should drive openness to 0");
        assert!(closed[0].blink, "blink flag set when closed");
        assert_eq!(closed[1].openness, 0.0, "both eyes closed");

        // Reopen -> openness recovers above the halfway mark.
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
        // First sample initializes exactly.
        assert_eq!(k.update(0.70), 0.70);
        // A large step is in the fast regime -> tracks toward the new value.
        for _ in 0..40 {
            k.update(0.20);
        }
        assert!(
            k.x < 0.30,
            "fast regime should track the step down, got {}",
            k.x
        );
        // Small ±0.02 jitter sits in the deadzone -> heavily smoothed (barely moves).
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
        // With the flag on, a warmed-open eye still reports open (sanity: the
        // Kalman path doesn't collapse openness for a steadily-open eye).
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
        // s3/s4 drive squeeze only while the corresponding eye is closed.
        let mut st = SRanipalState::new();
        st.tuning.squeeze_gain = 1.0; // unity gain to verify the raw channel mapping
        feed(&mut st, 0.62, 260); // warm up open
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            // openness still in the open band, squeeze channel (idx 3) high.
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
        // (0.60 - 0.18) / 0.45 = 0.93
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
    fn native_squeeze_per_eye_independent() {
        // A single-eye squeeze (left only) must NOT bleed to the right (winks).
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
        feed(&mut st, 0.62, 260); // warm up / trusted
                                  // Raw clearly above upper (=baseline+0.02) but the decode uses in2=raw;
                                  // x=(raw-upper)/(mid-upper) so a raw past mid yields wide ~1.
                                  // feed() widens BOTH eyes, so the bilateral gate lets it through.
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
        // A quick blink whose raw only dips SHALLOW (0.48, not reaching closed_ref
        // ~0.42) must still read fully closed thanks to the velocity latch — the
        // proportional ramp alone would leave it ~0.35 open (user 2026-06-27).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260); // warm open / trusted, prev_raw settled at 0.62
                                  // Sharp step down to a shallow bottom (a -0.14 step, over the velocity threshold, raw <
                                  // squeeze_top) -> latch -> openness driven to ~0.
        let blink = feed(&mut st, 0.48, 6);
        assert!(
            blink[0].openness < 0.1,
            "fast blink should snap closed, got {}",
            blink[0].openness
        );
        assert!(blink[0].blink, "blink flag set during fast blink");
        // Reopen: raw rises -> latch releases at the bottom -> openness recovers.
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
        // Regression: a fast blink that reopens STRAIGHT into wide
        // must not leave a stale latch. The wide block early-returns, so the latch is
        // cleared by the `raw >= squeeze_top` guard; when raw later settles down out
        // of wide to normal-open, openness must stay open (NOT snap to 0).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260); // warm open
        let blink = feed(&mut st, 0.48, 4); // fast blink -> latched closed
        assert!(
            blink[0].openness < 0.1,
            "blink closed, got {}",
            blink[0].openness
        );
        feed(&mut st, 0.80, 6); // reopen straight into WIDE (raw > upper)
                                // Settle from wide back down to relaxed-open; raw is FALLING (raw < prev_raw)
                                // — the buggy stale latch would force openness to 0 here.
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
        // A SLOW descent to the same shallow level (small per-frame steps) must NOT
        // trip the velocity latch — it should follow the continuous ramp (so squeeze
        // / gradual closes keep their continuity), reading partly-open, not closed.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260);
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        // Ramp down 0.62 -> 0.50 in 0.02 steps (each under the velocity floor), then hold.
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
        // wide_requires_both is on by default: one eye widening must emit NO wide
        // on either eye (片目だけでeyewide廃止 — user 2026-06-27).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260); // warm up both / trusted
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            // Left wide (raw well above upper), right relaxed (in baseline band).
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
        // Regression (user 2026-06-27): one eye reaches wide first and its magnitude
        // charges up while gated to 0; when the lagging eye crosses, the gate must
        // RAMP wide in (not dump the charged value in one frame -> "がくっと見開く").
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260); // warm both open
        let g = GazeSample::default();
        // Eye0 strongly wide and held; eye1 still relaxed -> wide gated to 0, but
        // eye0's internal magnitude charges to full.
        for _ in 0..20 {
            let ml = [[1.0, 0.82, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
        }
        // Eye1 now crosses into wide too -> gate opens.
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

    // Feed a series of ML-update raws under the REAL lumpy cadence (emit @120Hz, ML
    // @60Hz: each ML value lands on TWO consecutive process_frame calls, the 2nd with
    // draw==0). Uses a VALID-gaze sample so the emit-stage gaze-invalid forced-close
    // guard does not fire — this isolates the fast-blink latch. Returns emitted
    // openness per emit frame (2 per ML update).
    fn emit_cadence(st: &mut SRanipalState, raws_per_ml_update: &[f32]) -> Vec<f32> {
        let mut g = GazeSample::default();
        g.left.gaze_valid = true;
        g.right.gaze_valid = true;
        let mut out = Vec::new();
        for &raw in raws_per_ml_update {
            let ml = [[1.0, raw, 0.0, 0.0, 0.0], [1.0, raw, 0.0, 0.0, 0.0]];
            out.push(st.process_frame(ml, &g, true)[0].openness); // ML-update frame
            out.push(st.process_frame(ml, &g, true)[0].openness); // repeat frame
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
        // REGRESSION (finding 2026-06-27): a single fast downward ML step where the
        // eye never actually closes (downward-saccade lid-dip / one-update signal dip
        // that recovers) must NOT trip the fast-blink latch into a phantom full-close.
        // Before the confirmation window the latch returned (0,0,0) on the entry frame,
        // emitting openness 0 for ~2 emit frames (~17ms) mid-open, then snapping back.
        let mut st = warmed_open_state();
        // ML-update sequence 0.62 -> 0.55(dip, a -0.07 step over the velocity threshold, < squeeze_top
        // ~0.59) -> 0.60(recover) -> 0.62. The eye dips to squeeze territory (0.55,
        // well above closed_ref ~0.41) and recovers on the very next ML update.
        let emitted = emit_cadence(&mut st, &[0.62, 0.55, 0.60, 0.62, 0.62]);
        let min_emit = emitted.iter().cloned().fold(f32::INFINITY, f32::min);
        // The dip must read like a shallow narrow (ramp ~0.78), NOT a full close. The
        // armed-but-deferred latch emits the ramp; recovery releases with no 0 ever.
        assert!(
            min_emit > 0.5,
            "transient lid-dip must not flicker closed (min openness {}, want >0.5)",
            min_emit
        );
    }

    #[test]
    fn genuine_fast_blink_still_snaps_closed_under_cadence() {
        // No-regression for the dip fix: a GENUINE fast blink (sharp drop that stays
        // shut for several ML updates) must STILL reach fully closed. The confirmation
        // window only delays the commit ~2 emit frames (~16ms) while emitting the
        // descending ramp, then snaps to 0.
        let mut st = warmed_open_state();
        // 0.62 -> 0.45 (sharp) then HELD ~shut (0.43) for several ML updates, reopen.
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
        // FINDING under test: blinking while holding wide collapses wide_gate, so
        // wide "swells back in" over ~50ms on reopen. Trace wide_gate + is_wide L/R
        // + emitted wide through: hold wide -> bilateral blink -> reopen into wide.
        // Realistic 120Hz emit / 60Hz ML cadence (raw repeats on the repeat frame).
        let g = GazeSample::default();
        fn step(st: &mut SRanipalState, g: &GazeSample, rl: f32, rr: f32) -> [EyeResult; 2] {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, g, true)
        }
        let mut st = SRanipalState::new();
        // warm both open at relaxed 0.62 (lumpy cadence: each ML update repeated).
        let mut r = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..130 {
            step(&mut st, &g, 0.62, 0.62);
            r = step(&mut st, &g, 0.62, 0.62);
        }
        // Engage steady wide at 0.82 long enough for the gate to reach 1.0.
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

        // Bilateral blink: raw drops to 0.39 (fully closed). A real blink is ~6 emit
        // frames = 3 ML updates. Trace each emit frame.
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

        // Reopen STRAIGHT back into wide.
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
        // CONTROL: a GENUINE wide-disengage (one eye relaxes out of wide while
        // staying OPEN) must still fade the gate down. The relaxing eye is OPEN
        // (not `closed`), so a "freeze while closed" fix must NOT freeze here.
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
        // Both eyes wide but LEFT under-reads (poor sensitivity: barely above
        // upper) while RIGHT reads strong. Output must be IDENTICAL on both eyes
        // (max of the two) so the weak left is pulled up — no L/R difference.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 260); // warm up both
        let g = GazeSample::default();
        let mut last = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for _ in 0..20 {
            // Left just over upper (~baseline+0.02), right strongly wide.
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

    // A valid gaze sample with DISTINCT left/right directions, so each eye's gaze
    // smoother settles somewhere different and a yoke is observable.
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
        // The beta-tester bug (2026-07-04): squint one eye while looking around and it
        // freezes where the squint began. With the yoke on (default), the squinting
        // eye must MIRROR the open, tracking eye's gaze instead.
        let mut st = SRanipalState::new();
        let g = gaze_lr([0.4, 0.1, 0.9], [-0.4, 0.1, 0.9]);
        // Warm both eyes open + trusted, each smoother settled on its own direction.
        let open = [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]];
        for _ in 0..260 {
            st.process_frame(open, &g, true);
        }
        // Squint the LEFT eye (ch1 low) while the RIGHT stays open and tracking.
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

        // This settles below the yoke engage threshold but remains above the blink
        // threshold. The old implementation did nothing here despite the UI saying
        // "squint eye follows open eye".
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

        // Pipeline freshness filtering happens before this state machine. A false flag
        // here therefore represents a dropout that already outlived the transient grace.
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

        // A real direction change still raises the adaptive alpha and catches up quickly.
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
    fn gaze_yoke_does_not_tremble_at_blink_boundary() {
        // The yoke once keyed on an instantaneous threshold: a squint hovering
        // at that boundary alternated every frame between the
        // partner's gaze and the eye's own (garbage) gaze — visible trembling
        // (user 2026-07-08). With the hysteresis hold, the squinting eye must
        // stay pinned to the partner through the hover.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        // Left squints to the blink-flag boundary with GARBAGE own gaze; right
        // stays open, tracking a distinct direction.
        let own_garbage = [-0.5f32, 0.2, -1.0];
        let partner = [0.3f32, 0.0, -1.0];
        let g = gaze_lr(own_garbage, partner);
        // Settle into the squint so the yoke engages...
        for _ in 0..12 {
            st.process_frame(
                [[1.0, 0.45, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        // ...then hover around the shallow-squint engage region. The separate 0.55
        // release threshold must keep the selected source fixed throughout.
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
        // Genuine reopen releases the hold: own gaze resumes.
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
        // After a NATURAL bilateral blink both eyes hold the yoke hysteresis
        // through the reopen (longer with blink_reopen_ms set). They must not
        // mirror EACH OTHER during that window — the order-dependent cross-copy
        // re-pointed both eyes on every blink (user 2026-07-08 residual
        // trembling after the first hysteresis fix).
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
        // Bilateral fast blink...
        for _ in 0..8 {
            st.process_frame(
                [[1.0, 0.36, 0.0, 0.0, 0.0], [1.0, 0.36, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        // ...then a slow (slew-limited) recovery: no cross-mirroring allowed.
        let mut crossed = false;
        for _ in 0..60 {
            let out = st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
            if out[0].gaze_yoked || out[1].gaze_yoked {
                crossed = true; // neither eye may claim the other mid-recovery
            }
            if out[0].gaze[0] > 0.0 || out[1].gaze[0] < 0.0 {
                crossed = true; // each eye stays on its OWN side
            }
        }
        assert!(
            !crossed,
            "recovering eyes must not mirror each other after a blink"
        );
    }

    #[test]
    fn gaze_yoke_idle_when_both_eyes_track() {
        // Both eyes open + tracking → no yoke, each keeps its own gaze.
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
        // With the yoke OFF the squinting eye is NOT mirrored (legacy freeze behavior:
        // the sink then suppresses its gaze on blink and the avatar holds last).
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
        // wide and squeeze come from different channels; a frame can present BOTH.
        // Chain OFF (default) → both pass independently. Chain ON → each attenuates
        // the other so the collision shrinks (neither is amplified).
        let mut st = SRanipalState::new();
        st.tuning.wide_gain = 1.0;
        st.tuning.squeeze_gain = 1.0;
        feed(&mut st, 0.62, 260); // warm both open + trusted
        let g = GazeSample::default();
        // Both eyes: raw well above upper (wide fires, bilateral gate passes) AND ch3
        // high (native squeeze fires) — the exact both-on case the chain is for.
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
        // The "one eye stuck ~half-open" repro: after the baseline learns ~0.62, a relaxed
        // eye that sits a little BELOW baseline (a noisy / dim right camera lives here) must
        // still read fully open. The old tight full-open margin (~0.03) read this ~0.8 and,
        // smoothed with deeper dips, sank toward half; the wider open dead-zone fixes it.
        let mut st = SRanipalState::new();
        let g = GazeSample::default();
        for _ in 0..300 {
            st.process_frame(
                [[1.0, 0.62, 0.0, 0.0, 0.0], [1.0, 0.62, 0.0, 0.0, 0.0]],
                &g,
                true,
            );
        }
        // Ease gently down to 0.56 (steps under the velocity floor so the fast-blink latch stays out),
        // then hold — the relaxed level is now ~0.06 under baseline.
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
        // Guard: widening the open dead-zone must NOT break blink (full close) or wide.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300); // warm open
        let closed = feed(&mut st, 0.30, 12); // deep close
        assert_eq!(closed[0].openness, 0.0, "blink still drives openness to 0");
        assert!(closed[0].blink);
        let wide = feed(&mut st, 0.80, 24); // reopen straight into bilateral wide
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
        // The wide-collapse fix (user 2026-07-05): a HELD wide whose raw sits WITHIN
        // WIDE_LEARN_CAP of the baseline (+0.06 here — the case the cap alone doesn't
        // exclude) must NOT drag the relaxed baseline up. If it did, `upper` would overtake
        // the wide level and wide would die after a few tries.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 300); // warm + trusted, baseline ~0.62
        let base0 = st.baseline(Eye::Left);
        let out = feed(&mut st, 0.68, 400); // hold bilateral wide (+0.06) for a long time
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
        // The 2026-07-06 instability root cause: squint raws (~0.53) sit INSIDE the
        // absolute learning band and used to be learned at the FULL rate — a 10s
        // squint sank the baseline ~0.09, the relaxed eye then read as wide
        // (raw > upper), and the old is_wide freeze made recovery impossible.
        // With the symmetric band-gate a 10s squint injects < 0.01 of error and
        // returning to relaxed shows NO wide at all.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.53, 1200); // 10s bilateral squint @120Hz
        let b1 = st.baseline(Eye::Left);
        assert!(
            (b1 - b0).abs() < 0.01,
            "squint sank the baseline ({b0} -> {b1})"
        );
        // Back to relaxed: no spurious wide, fully open within a second.
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
        // Ordinary mature learning is frozen. A deliberate hold shorter than the
        // stuck-wide threshold must not move the open reference at all; a truly
        // stale-low reference is still handled by the bounded breaker test below.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.68, 1200); // held just-wide, under the breaker threshold
        let b1 = st.baseline(Eye::Left);
        assert!(
            (b1 - b0).abs() < 1e-6,
            "sub-breaker wide moved the mature baseline ({b0} -> {b1})"
        );
    }

    #[test]
    fn stuck_wide_breaker_recovers_bounded() {
        // A stale-LOW persisted baseline (e.g. the setup changed between sessions)
        // makes the relaxed eye read wide from frame one. Natural blinks must NOT
        // reset the breaker, and recovery must be automatic and bounded (~20s):
        // the breaker re-anchors the baseline at the fast rate until the wide
        // hysteresis releases on its own.
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.53,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
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
                st.process_frame(blink, &g, true); // natural blink mid-episode
                frame += 1;
            }
        }
        let at = recovered_at.expect("never recovered from a stale-low baseline");
        assert!(at <= 3000, "recovery took {at} frames (> 25s)");
    }

    #[test]
    fn short_wide_hold_unaffected_by_breaker() {
        // Guards WIDE_STUCK_FRAMES against future tightening: the user-confirmed
        // ~10s deliberate wide hold must stay under the breaker — wide keeps firing
        // for the whole hold and the baseline barely moves.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        let last = feed(&mut st, 0.75, 1200); // 10s bilateral wide
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
        // recenter keeps the old baseline as an 8-sample prior: one outlier frame
        // right after (mid-expression / noise) must not become the new baseline
        // (the old baseline_n=0 snapped to it verbatim).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        st.recenter();
        feed(&mut st, 0.70, 1); // a single outlier frame
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
    fn burst_wide_spam_does_not_trip_breaker() {
        // Repeated deliberate wide bursts with short relaxed gaps must NOT
        // accumulate into the stuck-wide breaker (review 2026-07-06): a >=0.4s
        // relaxed-open gap discharges the counter, so only a near-continuous ~15s
        // wide can trip it. 6 x 3s bursts = 2160 wide frames total (> the 1800
        // threshold), so without the discharge this test fails with a mid-burst
        // wide collapse.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        let mut last_wide = 0.0f32;
        for _ in 0..6 {
            let w = feed(&mut st, 0.75, 360); // 3s bilateral wide burst
            last_wide = w[0].wide;
            feed(&mut st, 0.62, 96); // 0.8s relaxed gap
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
        // The user's habitual remedy: pressing Recenter DURING a spurious-wide
        // episode. That re-enters the capped bootstrap — which used to be a
        // zero-learning-rate wedge (review 2026-07-06). The slow path + breaker
        // must still recover automatically, bounded (~20s), never wedge.
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.53,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
        };
        st.restore_all(&CalibStore {
            left: snap,
            right: snap,
        });
        let g = GazeSample::default();
        let relaxed = [[1.0, 0.66, 0.0, 0.0, 0.0], [1.0, 0.66, 0.0, 0.0, 0.0]];
        for _ in 0..600 {
            st.process_frame(relaxed, &g, true); // 5s into the spurious-wide episode
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
        // Recenter while the wide is latched PRE-ARMS the breaker (the press is
        // the detection), so the fast re-anchor starts immediately: recovery is
        // seconds, not the 15s detection + heal.
        assert!(
            at <= 1800,
            "recovery after recenter took {at} frames (> 15s)"
        );
    }

    #[test]
    fn fast_blink_yokes_lagging_eye_closed() {
        // A real blink is bilateral, but one eye's descent can just miss the
        // velocity latch (slightly shallower/slower raw response): the latched eye
        // reads 0 while the lagging eye's ramp bottoms out ~half-open — the avatar
        // blinks with one eye (半目, user 2026-07-07). The bilateral yoke must snap
        // the lagging eye closed while the partner's latch is committed.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400); // warm both eyes, baseline ~0.62
        let g = GazeSample::default();
        let step = |st: &mut SRanipalState, rl: f32, rr: f32| {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            // Each ML sample lands on two emit frames (120Hz emit / 60Hz ML).
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true)
        };
        // Left: fast blink (per-update drop 0.12, over the velocity threshold). Right: lagging
        // descent (drops 0.05, under the OLD absolute threshold, bottom 0.49 — deep, but its ramp alone
        // would read it ~half-open).
        step(&mut st, 0.50, 0.57);
        step(&mut st, 0.40, 0.52);
        let mut min_right: f32 = 1.0;
        for _ in 0..6 {
            let out = step(&mut st, 0.40, 0.49); // both held at their bottoms
            min_right = min_right.min(out[1].openness);
            assert_eq!(out[0].openness, 0.0, "latched left eye reads 0");
        }
        assert_eq!(
            min_right, 0.0,
            "lagging right eye must be yoked to 0, got {min_right}"
        );
        // Reopen: both eyes release on their own rise and recover.
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

    /// Feed one deliberate SLOW blink: descend in -0.02 steps (under the velocity floor, no
    /// latch), dwell at `bottom`, rise back in +0.02 steps.
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
        // The user-facing calibration gesture: a deliberate slow deep blink must
        // move the openness=0 point toward the observed bottom AND the learning
        // must PERSIST (the old ratchet+decay erased it within ~10s, so the
        // gesture effectively did nothing — user 2026-07-07).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left); // ~baseline - 0.20 = 0.42
        slow_blink(&mut st, 0.62, 0.36, 60);
        let after = st.closed_ref(Eye::Left);
        // Cold-start safety: the first ambiguous close is only staged, because a
        // single squint cannot be distinguished from a genuinely shallow blink.
        assert!(
            (after - before).abs() < 0.01,
            "first close must only stage a candidate ({before} -> {after})"
        );
        // A comparable second close confirms and snaps the calibrated depth.
        slow_blink(&mut st, 0.62, 0.36, 60);
        let after2 = st.closed_ref(Eye::Left);
        assert!(
            after2 < before - 0.04,
            "second slow blink must confirm the calibrated depth ({after} -> {after2})"
        );
        // And it STICKS: 30s of relaxed tracking must not decay it back.
        feed(&mut st, 0.62, 3600);
        let held = st.closed_ref(Eye::Left);
        assert!(
            (held - after2).abs() < 0.006,
            "learned depth must not decay away ({after2} -> {held})"
        );
    }

    #[test]
    fn fast_blink_does_not_teach_blink_depth() {
        // A fast blink's raw bottom is under-sampled (that is why the velocity
        // latch exists) — episodes the latch touched must NOT move the calibration.
        // The old code ratcheted closed_ref to any single low sample.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        for _ in 0..5 {
            feed(&mut st, 0.30, 14); // snap shut (latch commits), hold ~120ms
            feed(&mut st, 0.62, 120); // reopen, relax
        }
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - before).abs() < 0.01,
            "fast blinks must not move closed_ref ({before} -> {after})"
        );
    }

    #[test]
    fn resting_closed_eyes_do_not_teach_blink_depth() {
        // Eyes closed for seconds (resting) is not a calibration blink: the model
        // output drifts there, so over-long episodes are discarded.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let before = st.closed_ref(Eye::Left);
        slow_blink(&mut st, 0.62, 0.34, 500); // 500 frames > BLINK_LEARN_MAX_FRAMES
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - before).abs() < 0.01,
            "resting closed must not teach depth ({before} -> {after})"
        );
    }

    #[test]
    fn blink_depth_survives_recenter_and_restore() {
        // The learned depth is a physiological per-user property: it must survive
        // recenter (which re-learns the baseline) and round-trip through the
        // persisted snapshot.
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
        // Snapshot -> fresh state -> restore.
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
        // The field failure: down-gaze / squint dwell used to sink each eye's
        // baseline at a different rate. Mature references must now be immutable.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.40, 60); // stage
        slow_blink(&mut st, 0.62, 0.40, 60); // confirm -> learned, decoupled
        let baseline = st.baseline(Eye::Left);
        let learned = st.closed_ref(Eye::Left);
        assert!(learned < 0.44, "depth learned, got closed_ref {learned}");
        // Long dwell below the old learning band: looking down / soft squint,
        // not a blink and not a reason to recalibrate.
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
        // And a full slow close still reads fully closed (the user-visible claim).
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
        // Companion fix (ENV_HEAL τ~17s -> ~2.8min): a ~30s no-blink stare used to heal
        // reach_env nearly back to baseline, the genuine-close gate went soft, and a
        // slow half-squint then taught blink_depth SHALLOWER — eroding the ramp span
        // (the measured R 0.120 vs L 0.219). The envelope must outlive a stare so the
        // squint is still rejected as "never reached the floor".
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_blink(&mut st, 0.62, 0.38, 60);
        slow_blink(&mut st, 0.62, 0.38, 60); // learned
        let learned = st.closed_ref(Eye::Left);
        feed(&mut st, 0.62, 4800); // 40s stare, zero blinks
        slow_blink(&mut st, 0.62, 0.50, 80); // deliberate half-squint, dwelled
        let after = st.closed_ref(Eye::Left);
        assert!(
            (after - learned).abs() < 0.01,
            "a post-stare squint must not lift the close floor ({learned} -> {after})"
        );
    }

    #[test]
    fn field_partial_close_sequence_cannot_ratchet_floor_or_depth() {
        // Exact shape of sranibro_diag_1783998233.csv: three episode exits in
        // 7.8 s previously moved R closed_ref .3755 -> .4034 -> .4284 -> .4509.
        // Their implied bottoms (.515/.528/.541) are progressively shallower;
        // they must never become a three-close confirmation.
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
            slow_blink(&mut st, 0.62, 0.50, 300); // >2 s: held squint, not a close episode
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
        slow_blink(&mut st, 0.62, 0.40, 60); // initial confirmation
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

        slow_blink(&mut st, 0.62, 0.34, 60); // candidate #1
        feed(&mut st, 0.62, FLOOR_CANDIDATE_MAX_AGE as usize + 1);
        slow_blink(&mut st, 0.62, 0.34, 60); // a new candidate, not confirmation
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
            eye.closed_ref = 0.20; // below the old baseline-0.40 rail (0.30)
            eye.blink_depth = 0.40;
        }
        feed(&mut st, 0.70, 1);
        assert_eq!(st.closed_ref(Eye::Left), 0.20);
        assert_eq!(st.closed_ref(Eye::Right), 0.20);
    }

    #[test]
    fn razor_thin_close_span_is_floored() {
        // A learned-shallow depth (0.12) under a wide open_deadzone (0.08) left a 0.04
        // ramp denominator — ~25x error amplification (the measured right-eye config).
        // The denominator is floored at MIN_CLOSE_SPAN with the ZERO point unchanged:
        // full closes read 0, relaxed reads 1, and the slope stays bounded.
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
        // Mid-ramp: half a floored span above the bottom must read ~0.5, not the razor
        // 0.04-denominator value (which would put +0.02 of raw at ~1.0 already).
        // Descend gradually (-0.02 steps, under the velocity floor and the emit-stage
        // fast-closing gate) so no latch / forced close fires on the way down.
        let blink_top = 0.62 - 0.12; // = closed_ref (learned)
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
        // Slow full close to the learned bottom still reads 0.
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
        // Calib files written before blink_depth / mid_anchor existed must load
        // with the stock values (serde default), not fail or zero them.
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

    /// The right eye's warped (ahead-running) raw trajectory for a symmetric
    /// close: same endpoints as the left, lower mid-range (t^1.2 warp) —
    /// reproduces the reported L0.8 / R0.65 class of divergence.
    fn warped_right(raw_l: f32) -> f32 {
        let t = ((raw_l - 0.36) / 0.26).clamp(0.0, 1.0);
        0.36 + 0.26 * t.powf(1.2)
    }

    /// Feed one paired frame twice (120Hz emit / 60Hz ML cadence).
    fn step_pair(st: &mut SRanipalState, rl: f32, rr: f32) -> [EyeResult; 2] {
        let g = GazeSample::default();
        let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
        st.process_frame(ml, &g, true);
        st.process_frame(ml, &g, true)
    }

    /// One full warped symmetric slow close: descend, dwell (clean episode on
    /// both eyes), reopen, relax (staged anchors swap in at full open).
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

    /// Descend to a mid-close hold (never deep enough for an episode) and
    /// measure the steady L/R openness gap there.
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
        // Reopen and relax so the next cycle starts clean.
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
        // THE feature test: a symmetric slow close whose right eye runs ahead
        // mid-range must, after a few deliberate slow-blink calibration cycles,
        // read (near-)symmetric — anchors move oppositely, their mean stays 0.5
        // (pair-mean feel pinned to identity), and the mid-close gap shrinks.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let gap_before = mid_close_gap(&mut st);
        assert!(
            gap_before > 0.10,
            "warp should produce a visible gap, got {gap_before}"
        );
        // Two episodes are now reserved for cold-start floor confirmation; the
        // following cycles train the curve equalizer from confirmed bottoms.
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
        // A wink can never train the equalizer: the open eye's pre-correction
        // ramp value is pinned at 1.0 (outside the sample window), the pair gap
        // is ~1.0 (PAIR_MAX_DIFF), and one eye never completes a synchronized
        // clean episode.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..6 {
            // Slow wink: left descends and dwells, right stays open with noise.
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
        // anchor 0.5 is the identity bit-exactly; endpoints are fixed points for
        // every anchor (blink 0 and full-open 1 pass through unchanged).
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
        // A staged anchor must not apply mid-close (the swap is only
        // output-invariant at full open, where f(1,a)=1 for every a).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        // Descend the LEFT eye only (a wink): the right stays open, so no bilateral
        // curve-equalizer commit can complete and re-stage the anchor we set by hand
        // (a fresh eye's relative entry gate now treats a mid-close as an episode).
        let mut raw = 0.62f32;
        while raw > 0.48 {
            raw -= 0.01;
            step_pair(&mut st, raw.max(0.48), 0.62);
        }
        // Stage while ALREADY mid-close: the swap must wait for full open.
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
        // Sanitize garbage on restore.
        let bad = CalibSnapshot {
            baseline: 0.62,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: f32::NAN,
            learned_once: false,
        };
        st2.restore(Eye::Left, bad);
        assert_eq!(
            st2.eyes[0].mid_anchor, 0.5,
            "NaN anchor must sanitize to identity"
        );
    }

    #[test]
    fn stale_partner_exit_does_not_deadlock_pairing() {
        // 2026-07-08 diagnostic recording: the eye with the LONGER episode exits
        // AFTER its partner, so its exit lingered into the next gesture, failed
        // the sync check there, and every gesture's samples were discarded —
        // forever. Entering a new episode must supersede that eye's old exit so
        // the eventual pairing is always same-gesture.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        // This test targets stale curve-pair exits, not cold-start floor
        // confirmation. Treat both eyes as already floor-calibrated.
        st.eyes[0].learned_once = true;
        st.eyes[1].learned_once = true;
        st.eyes[0].closed_ref = 0.36;
        st.eyes[1].closed_ref = 0.36;
        st.eyes[0].blink_depth = 0.26;
        st.eyes[1].blink_depth = 0.26;
        // Seed a stale RIGHT-only exit: a clean slow right-eye-only deep close.
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
        // Two symmetric-intent gestures where the right runs slightly ahead
        // (enters earlier, exits later — the recording's geometry).
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
        // 2026-07-08 recording: today's cameras idle at raw 0.42-0.50; the old
        // absolute band floor (0.45) silently excluded the right eye's relaxed
        // level — its baseline could neither learn nor recenter. The widened
        // band must track a 0.43 setup, from cold start AND from a stale-high
        // restored baseline via recenter.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.43, 300);
        let b = st.baseline(Eye::Left);
        assert!(
            (b - 0.43).abs() < 0.01,
            "cold start at raw 0.43 must learn, got {b}"
        );
        // Stale-high restore (the recording's exact state), then recenter.
        let snap = CalibSnapshot {
            baseline: 0.51,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.29,
            mid_anchor: 0.5,
            learned_once: false,
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
        // Review 2026-07-08: an asymmetric half-close HOLD used to keep
        // collecting pair samples (a hold counted as "falling" forever), and a
        // following calibration blink committed the hold-dominated statistics —
        // manufacturing an L/R gap out of an expression. The anti-hold gate
        // (since_down) plus the commit time-window must make this teach NOTHING.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..3 {
            // Descend asymmetrically to a half-close hold and stay 2.5s.
            let (mut rl, mut rr) = (0.62f32, 0.62f32);
            while rl > 0.50 {
                rl -= 0.01;
                rr -= 0.013; // right descends deeper: hold at L 0.50 / R 0.46
                step_pair(&mut st, rl.max(0.50), rr.max(0.46));
            }
            for _ in 0..150 {
                step_pair(&mut st, 0.50, 0.46);
            }
            // Then the documented calibration gesture straight from the hold.
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
        // Review 2026-07-08: a clean but brisk blink that gathers too little
        // weight must DISCARD its episode exits, not leave them armed to pair
        // with unrelated later samples.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        // Brisk clean close: -0.025/update (under all latch/speed gates) crosses
        // the mid window in a few frames -> w_sum < PAIR_MIN_WEIGHT.
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
        // A following asymmetric mid-close descent+hold must not commit anything.
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
        // Review 2026-07-08 (pre-existing bug, amplified by the anchor map): the
        // forced-close "fast_closing" test compared CORRECTED targets, which put
        // ordinary +-0.02 mid-close noise over its threshold whenever gaze was
        // invalid — one-frame snaps to 0 and back. The raw-domain test must not
        // fire on noise. (GazeSample::default() has gaze_valid = false.)
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
        // Minimum legitimate corrected target here is ~0.24 (R eye at x=0.33
        // through the 0.70 anchor); anything much below that means the forced
        // close fired on noise.
        assert!(
            lo > 0.15,
            "gaze-invalid mid-close noise forced a close (lo {lo})"
        );
    }

    #[test]
    fn corrected_midclose_hold_never_flaps_at_alpha1() {
        // Worst case: anchors at the clamps, alpha_close = 1.0 (no downward
        // smoothing), mid-close hold with alternating +-0.02 raw noise. The
        // static map may amplify noise by at most 1.67x — the output must stay
        // strictly mid-range (no excursion to 0 or 1, nothing to flap).
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
        // Detector v2 (2026-07-09): the old ABSOLUTE 0.06 threshold missed the
        // weaker eye's blinks (its dynamic range is ~2/3 of the other's, so its
        // fast steps sit ~0.05 — one full user-visible miss on record at
        // t=27.70s). The v2 threshold is relative to blink_depth with a 0.045
        // floor, so a -0.05/update descent commits.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        // Valid gaze: keep the emit-stage gaze-invalid forced-close guard out of
        // the picture so the test exercises the LATCH path alone.
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
        // v2 release is rise-off-bottom: the old first-up-tick release fired on
        // sub-noise bottom tremors (+0.002..0.009) and popped the latch open
        // mid-blink (11 recorded re-latch flickers in one 28s session).
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
        let reopen_idx = 14; // first frame of the 0.43 reopen step
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
        // Tuning::blink_reopen_ms slew-limits the RISE after a full close so the
        // receiving avatar can render the blink; closing stays instant and the
        // limit disengages once recovered (>= 0.9).
        let mut st = SRanipalState::new();
        st.tuning.blink_reopen_ms = 200.0; // step = 1/24 of full range per frame
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        // Fast blink down...
        let mut out = [EyeResult::new(Eye::Left), EyeResult::new(Eye::Right)];
        for r in [0.50f32, 0.38, 0.38] {
            let ml = [[1.0, r, 0.0, 0.0, 0.0], [1.0, r, 0.0, 0.0, 0.0]];
            for _ in 0..2 {
                out = st.process_frame(ml, &g, true);
            }
        }
        assert_eq!(out[0].openness, 0.0, "close is not slowed");
        // ...then reopen: the rise must be capped at ~1/24 per frame.
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
        // With the slider at 0 the recovery is fast (old behavior).
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
    fn held_wink_does_not_bounce() {
        // "Closes fully, then bounces back a little" (user 2026-07-09): a held
        // wink's bottom typically sits a few hundredths ABOVE the learned
        // closed_ref, so the old cap-release dropped the latch into a ramp
        // rendering 0.1-0.25 after ~100-200ms. A deep hold must read 0 for the
        // WHOLE hold — the cap is shallow-only and the release needs to leave
        // the closed zone.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        // Fast close, then hold at 0.45: above closed_ref (0.42), below the
        // exit zone (0.47) — the exact bounce regime.
        let mut hold_max: f32 = 0.0;
        let seq: Vec<f32> = std::iter::once(0.50)
            .chain(std::iter::repeat(0.45).take(40)) // ~660ms hold, way past caps
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
        // Genuine reopen releases and recovers.
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
        // A hard close overshoots below the settling level then recovers a few
        // hundredths while the lid is still shut — that recovery must NOT
        // release the latch (it exceeds the rise-off-bottom threshold but never
        // leaves the closed zone).
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
        // After a release, stale descent history must not re-commit while raw is
        // still low (the recorded 17ms re-entry flicker class): the release
        // re-seeds the update history, so a re-close must earn fresh evidence.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        // Wiggle stays ABOVE closed_ref (0.42) so a 0 can only come from a latch.
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
                        re_latched = true; // a zero AFTER the release = re-latch
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
        // A brisk bilateral squint (-0.04/update to ~baseline-0.11, held) must
        // never read as a blink: the single-step path sits under the velocity
        // floor and the cumulative fall2 path is depth-gated at 0.60 x depth.
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
        // A blink where NEITHER eye trips the per-eye velocity latch (both step
        // -0.05..-0.03 per ML update, under the OLD absolute threshold): one eye still reaches deep
        // blink territory while the other bottoms out half-open. The deep+fast
        // anchor must snap BOTH closed (user 2026-07-07: both eyes read half-shut
        // without this).
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
        step(&mut st, 0.43, 0.50); // L deep (b-0.19) at blink speed; R half-open
        let mut both_zero = false;
        for _ in 0..4 {
            let out = step(&mut st, 0.43, 0.50);
            both_zero |= out[0].openness == 0.0 && out[1].openness == 0.0;
        }
        assert!(
            both_zero,
            "medium-speed bilateral blink must close BOTH eyes"
        );
        // Reopen cleanly.
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
        // The anchor requires blink SPEED, not just depth (user 2026-07-07:
        // 降下スピードも考慮): a deliberate slow bilateral droop to deep-but-open
        // (~0.44, sleepy eyes) descends at ~0.01/update and must keep tracking the
        // smooth ramp — never snap to 0.
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
        // A fast WINK (one eye snaps shut, the other stays open — possibly with a
        // small sympathetic droop below its baseline) must NOT drag the open eye
        // closed: the yoke's depth gate (baseline - 0.10) sits well below any
        // sympathetic droop.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let g = GazeSample::default();
        let step = |st: &mut SRanipalState, rl: f32, rr: f32| {
            let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
            st.process_frame(ml, &g, true);
            st.process_frame(ml, &g, true)
        };
        // Left winks fast; right droops sympathetically to 0.57 (0.05 below
        // baseline — under squeeze_top but far above the yoke depth gate).
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
        // The 10s-hold guarantee must survive a sub-0.4s breath between holds:
        // the leaky discharge erases the first hold's accumulation proportionally
        // (42 calm frames wipe well over 1200), so the breaker can't trip
        // mid-expression and eat the baseline (review 2026-07-07 — the previous
        // all-or-nothing reset made this a hard cliff at the 0.4s boundary).
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let b0 = st.baseline(Eye::Left);
        feed(&mut st, 0.75, 1200); // 10s wide hold #1
        feed(&mut st, 0.62, 42); // 0.35s breath — below the full-discharge span
        let last = feed(&mut st, 0.75, 1200); // 10s wide hold #2
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
        // The pipeline persists calibration unconditionally, so an idle session
        // (HMD on the desk, nothing learned) writes baseline=0.60/n=0. Restoring
        // that must NOT skip the cold-start bootstrap (review 2026-07-06) — the
        // next real session converges in ~1s exactly like a fresh start.
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.60,
            baseline_n: 0,
            frame_count: 0,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
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
        // A stale-HIGH persisted baseline (relaxed raw now sits below it) must not
        // read as a half-closed eye: the open dead-zone masks the error while the
        // slow out-of-band path heals it. No manual recenter required.
        let mut st = SRanipalState::new();
        let snap = CalibSnapshot {
            baseline: 0.72,
            baseline_n: 5000,
            frame_count: 5000,
            blink_depth: 0.20,
            mid_anchor: 0.5,
            learned_once: false,
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

    // --- close-side per-eye auto-range floor (shallow-eye fix) ---------------------

    /// One deliberate SLOW bilateral close to `bottom` (−0.02/step so the fast-blink
    /// latch never trips), holding `dwell` frames while recording the MIN emitted
    /// openness at the bottom, then reopen and rest. Valid gaze isolates the ramp/floor
    /// path from the emit-stage gaze-invalid forced-close guard.
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

    /// Hold a steady raw (after a slow descent) and return the settled openness WITHOUT
    /// reopening — so no episode exit relearns the floor during the probe.
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
        // A "shallow" eye whose full close only reaches 0.47 (0.15 below baseline) used
        // to floor at openness ~0.42 forever: blink_depth floored at 0.20 and the
        // episode learner never fired above baseline−0.20, so it never recalibrated.
        // The per-eye auto-range must learn its OWN floor so a close reads fully 0.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        let first = slow_close_min(&mut st, 0.47, 40); // floor not learned yet
        let second = slow_close_min(&mut st, 0.47, 40); // confirms at episode exit
        let third = slow_close_min(&mut st, 0.47, 40); // starts with confirmed floor
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
        slow_close_min(&mut st, 0.50, 40); // deep squint before any real blink
        assert!(
            !st.eyes[0].learned_once,
            "one ambiguous squint must stay unconfirmed"
        );
        assert!((st.eyes[0].blink_depth - before_depth).abs() < 1e-4);

        // A deeper real close must replace, not confirm, the squint candidate.
        slow_close_min(&mut st, 0.47, 40);
        assert!(
            !st.eyes[0].learned_once,
            "a different-depth close replaces the candidate"
        );
        slow_close_min(&mut st, 0.47, 40); // confirms at the true bottom on exit
        assert!(st.eyes[0].learned_once);
        let confirmed = slow_close_min(&mut st, 0.47, 40);
        assert_eq!(confirmed, 0.0);
    }

    #[test]
    fn deeper_blink_passing_candidate_does_not_confirm_mid_descent() {
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_close_min(&mut st, 0.50, 40); // stage a shallow candidate

        // A real deeper blink pauses at the candidate depth before continuing.
        // The old mid-episode confirmation committed here, before seeing its bottom.
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
        // No regression for a NORMAL eye (bottom 0.40, 0.22 below baseline): closed_ref
        // must settle near baseline−0.20..0.22, and a mild squint (~baseline−0.16) must
        // still read clearly open — the looser entry gate must not close it early.
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
        let out = probe_hold_open(&mut st, 0.46); // mild squint ~baseline-0.16
        assert!(
            out.openness > 0.25 && !out.blink,
            "a mild squint must read clearly open, got {}",
            out.openness
        );
    }

    #[test]
    fn squint_heavy_does_not_pull_floor_up() {
        // A squint-heavy sequence must NOT ratchet the floor up. The mid squints (0.50)
        // trip the loose entry gate but are REJECTED by the genuine-close discriminator
        // (their bottom never reaches the envelope floor), so a narrowed-relaxed raw
        // still reads clearly open.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        for _ in 0..3 {
            slow_close_min(&mut st, 0.40, 40); // establish the genuine floor
        }
        let cr0 = st.closed_ref(Eye::Left);
        for r in 0..12 {
            slow_close_min(&mut st, 0.50, 40); // narrowed / squint hold
            if r % 3 == 2 {
                slow_close_min(&mut st, 0.40, 40); // occasional genuine blink pins the floor
            }
        }
        let cr1 = st.closed_ref(Eye::Left);
        assert!(
            cr1 <= cr0 + 0.03,
            "squints must not pull the floor up (closed_ref {cr0} -> {cr1})"
        );
        // A narrowed raw ABOVE the squint bottom still reads clearly open.
        let out = probe_hold_open(&mut st, 0.52);
        assert!(
            out.openness > 0.3,
            "narrowed-relaxed raw must read open, got {}",
            out.openness
        );
    }

    #[test]
    fn dropout_burst_does_not_disable_shallow_close() {
        // reach_env is present-gated and blink_depth persists, so a burst of lost-eye
        // frames (present=false, raw~0) must not disable the shallow-eye close.
        let mut st = SRanipalState::new();
        feed(&mut st, 0.62, 400);
        slow_close_min(&mut st, 0.47, 40);
        slow_close_min(&mut st, 0.47, 40); // confirms on exit
        let learned = slow_close_min(&mut st, 0.47, 40);
        assert_eq!(
            learned, 0.0,
            "shallow floor learned before the dropout, got {learned}"
        );
        let g = gaze_lr([0.0, 0.0, -1.0], [0.0, 0.0, -1.0]);
        // ch0 = 0 -> present=false (model lost the eye), raw ~0.
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
}
