//! Golden-replay regression: feed REAL recorded 120Hz sessions (the diagnostic
//! CSV recorder's output) through the post-processor and assert structural
//! fast-blink-detector properties — real blinks read fully closed, and nothing
//! false-closes at rest or on a wink's open eye. This pins the detector against
//! actual hardware data instead of synthetic feeds.

use sranibro_rs::core::eye_state::{CalibSnapshot, CalibStore, SRanipalState};
use sranibro_rs::core::types::GazeSample;

/// Replay a diag CSV from a restored calibration; returns (t_ms, raw, openness)
/// per emit frame (rows are already at emit cadence — feed each once).
fn replay(csv: &str, base: [f32; 2], depth: [f32; 2]) -> Vec<(u32, [f32; 2], [f32; 2])> {
    let snap = |b: f32, d: f32| CalibSnapshot {
        baseline: b,
        baseline_n: 5000,
        frame_count: 5000,
        blink_depth: d,
        mid_anchor: 0.5,
        learned_once: true, // restored real calibration: already learned (no re-snap)
    };
    let mut st = SRanipalState::new();
    st.restore_all(&CalibStore {
        left: snap(base[0], depth[0]),
        right: snap(base[1], depth[1]),
    });
    // Valid gaze, matching the live pipeline (the recording was made with the
    // device tracking; an invalid-gaze default would add forced closes the real
    // session never had).
    let mut g = GazeSample::default();
    g.left.gaze = [0.0, 0.0, -1.0];
    g.left.gaze_valid = true;
    g.right.gaze = [0.0, 0.0, -1.0];
    g.right.gaze_valid = true;
    let mut out = Vec::new();
    for line in csv.lines().skip(1) {
        let mut it = line.split(',');
        let t: u32 = match it.next().and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let rl: f32 = match it.next().and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let rr: f32 = match it.next().and_then(|v| v.parse().ok()) {
            Some(v) => v,
            None => continue,
        };
        let ml = [[1.0, rl, 0.0, 0.0, 0.0], [1.0, rr, 0.0, 0.0, 0.0]];
        let r = st.process_frame(ml, &g, true);
        out.push((t, [rl, rr], [r[0].openness, r[1].openness]));
    }
    out
}

/// Count separate episodes where BOTH eyes read (near-)fully closed.
fn bilateral_close_episodes(frames: &[(u32, [f32; 2], [f32; 2])]) -> usize {
    let mut n = 0;
    let mut inside = false;
    for (_, _, o) in frames {
        let closed = o[0] < 0.05 && o[1] < 0.05;
        if closed && !inside {
            n += 1;
        }
        inside = closed;
    }
    n
}

/// Count re-latch flickers per eye: openness pops 0 -> >0.35 -> back to 0 within
/// a few frames (the v1 bottom-tremor release artifact — 11 recorded instances).
fn flicker_count(frames: &[(u32, [f32; 2], [f32; 2])], eye: usize) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + 2 < frames.len() {
        if frames[i].2[eye] == 0.0 && frames[i + 1].2[eye] > 0.35 {
            // A pop off zero: does it slam back to zero within 10 frames?
            let back = frames[i + 2..frames.len().min(i + 12)]
                .iter()
                .any(|(_, _, o)| o[eye] == 0.0);
            if back {
                n += 1;
            }
        }
        i += 1;
    }
    n
}

#[test]
fn replay_healthy_session() {
    let frames = replay(
        include_str!("data/diag_healthy.csv"),
        [0.495, 0.442],
        [0.297, 0.234],
    );
    assert!(
        frames.len() > 3000,
        "fixture parsed ({} frames)",
        frames.len()
    );

    // Relaxed start (~0.9s): nothing may read closed.
    for (t, _, o) in frames.iter().take(100) {
        assert!(
            o[0] > 0.3 && o[1] > 0.3,
            "false close at rest (t={t}ms {o:?})"
        );
    }

    // The session's deliberate blinks (slow + fast) must all reach fully closed
    // — and the episode count must stay in the hand-labeled range (an UPPER
    // bound too: episode-count inflation = the re-latch flicker class v1 had).
    let closes = bilateral_close_episodes(&frames);
    assert!(
        (6..=16).contains(&closes),
        "bilateral close episodes out of range: {closes}"
    );
    // Zero mid-blink flickers (0 -> open pop -> 0 within ~80ms).
    assert_eq!(flicker_count(&frames, 0), 0, "left-eye re-latch flicker");
    assert_eq!(flicker_count(&frames, 1), 0, "right-eye re-latch flicker");

    // Wink windows: the OPEN eye must never be dragged shut.
    // Right-eye slow wink at t ~ 10.5-11.7s (left stays open) ...
    for (t, _, o) in frames
        .iter()
        .filter(|(t, _, _)| (10_500..=11_700).contains(t))
    {
        assert!(
            o[0] > 0.3,
            "left eye dragged shut during a RIGHT wink (t={t}ms open_l={})",
            o[0]
        );
    }
    // ... and left-eye slow wink at t ~ 12.4-12.95s (right stays open).
    for (t, _, o) in frames
        .iter()
        .filter(|(t, _, _)| (12_400..=12_950).contains(t))
    {
        assert!(
            o[1] > 0.3,
            "right eye dragged shut during a LEFT wink (t={t}ms open_r={})",
            o[1]
        );
    }
}

#[test]
fn replay_stale_baseline_session() {
    // Stress case: recorded while the right baseline was stuck ~0.07 high. The
    // detector's velocity evidence is baseline-independent, so blinks must still
    // read closed and rest must still read open.
    let frames = replay(
        include_str!("data/diag_stale_baseline.csv"),
        [0.506, 0.508],
        [0.262, 0.263],
    );
    assert!(
        frames.len() > 2500,
        "fixture parsed ({} frames)",
        frames.len()
    );
    for (t, _, o) in frames.iter().take(100) {
        assert!(
            o[0] > 0.3 && o[1] > 0.3,
            "false close at rest (t={t}ms {o:?})"
        );
    }
    let closes = bilateral_close_episodes(&frames);
    assert!(
        (5..=18).contains(&closes),
        "bilateral close episodes out of range: {closes}"
    );
    assert_eq!(flicker_count(&frames, 0), 0, "left-eye re-latch flicker");
    assert_eq!(flicker_count(&frames, 1), 0, "right-eye re-latch flicker");
}
