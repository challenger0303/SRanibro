
use sranibro_rs::core::eye_state::{CalibSnapshot, CalibStore, SRanipalState};
use sranibro_rs::core::types::GazeSample;

fn replay(csv: &str, base: [f32; 2], depth: [f32; 2]) -> Vec<(u32, [f32; 2], [f32; 2])> {
    let snap = |b: f32, d: f32| CalibSnapshot {
        baseline: b,
        baseline_n: 5000,
        frame_count: 5000,
        blink_depth: d,
        mid_anchor: 0.5,
        learned_once: true,
        endpoint_locked: false,
        endpoint_calibrated_unix: 0,
    };
    let mut st = SRanipalState::new();
    st.restore_all(&CalibStore {
        left: snap(base[0], depth[0]),
        right: snap(base[1], depth[1]),
    });
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

fn flicker_count(frames: &[(u32, [f32; 2], [f32; 2])], eye: usize) -> usize {
    let mut n = 0;
    let mut i = 0;
    while i + 2 < frames.len() {
        if frames[i].2[eye] == 0.0 && frames[i + 1].2[eye] > 0.35 {
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

    for (t, _, o) in frames.iter().take(100) {
        assert!(
            o[0] > 0.3 && o[1] > 0.3,
            "false close at rest (t={t}ms {o:?})"
        );
    }

    let closes = bilateral_close_episodes(&frames);
    assert!(
        (6..=16).contains(&closes),
        "bilateral close episodes out of range: {closes}"
    );
    assert_eq!(flicker_count(&frames, 0), 0, "left-eye re-latch flicker");
    assert_eq!(flicker_count(&frames, 1), 0, "right-eye re-latch flicker");

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
