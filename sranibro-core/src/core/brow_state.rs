
const HOLD_FRAMES: u16 = 15;
const DECAY_FRAMES: u16 = 45;

#[derive(Default)]
struct Eye {
    ema: f32,
    have_ema: bool,
    neutral: Option<f32>,
    last: f32,
    have_last: bool,
    blink_frames: u16,
}

pub struct BrowState {
    eyes: [Eye; 2],
    alpha: f32,
    gamma: f32,
}

impl Default for BrowState {
    fn default() -> Self {
        Self { eyes: Default::default(), alpha: 0.5, gamma: 1.0 }
    }
}

impl BrowState {
    pub fn recenter(&mut self) {
        for e in &mut self.eyes {
            e.neutral = None;
            e.have_last = false;
            e.blink_frames = 0;
        }
    }

    fn curve(&self, x: f32) -> f32 {
        let x = x.clamp(-1.0, 1.0);
        let g = self.gamma;
        if !g.is_finite() || g <= 0.0 || (g - 1.0).abs() < 1e-3 {
            x
        } else {
            x.signum() * x.abs().powf(g)
        }
    }

    pub fn process(&mut self, eye: usize, raw: f32, is_new: bool, blink: bool, recenter: bool) -> Option<f32> {
        if recenter {
            let e = &mut self.eyes[eye];
            e.neutral = None;
            e.have_last = false;
            e.blink_frames = 0;
        }
        if is_new && !blink && raw.is_finite() {
            let e = &mut self.eyes[eye];
            if e.have_ema {
                e.ema += self.alpha * (raw - e.ema);
            } else {
                e.ema = raw;
                e.have_ema = true;
            }
            if e.neutral.is_none() {
                e.neutral = Some(e.ema);
            }
        }
        let e = &mut self.eyes[eye];
        let (ema, neutral) = match (e.have_ema, e.neutral) {
            (true, Some(n)) => (e.ema, n),
            _ => return None,
        };
        if !blink {
            e.blink_frames = 0;
            let c = (ema - neutral).clamp(-1.0, 1.0);
            e.last = c;
            e.have_last = true;
            return Some(self.curve(c));
        }
        if !e.have_last {
            return Some(0.0);
        }
        let bf = e.blink_frames;
        e.blink_frames = bf.saturating_add(1);
        let held = e.last;
        let v = if bf < HOLD_FRAMES {
            held
        } else {
            let t = ((bf - HOLD_FRAMES + 1) as f32 / DECAY_FRAMES as f32).min(1.0);
            held * (1.0 - t)
        };
        Some(self.curve(v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn not_ready_until_first_open_inference() {
        let mut b = BrowState::default();
        assert!(b.process(0, 0.0, false, false, false).is_none());
        assert!(b.process(0, 0.4, true, true, false).is_none());
        assert!(b.process(0, 0.4, true, false, false).unwrap().abs() < 1e-6);
    }

    #[test]
    fn neutral_subtracts_and_recenters() {
        let mut b = BrowState::default();
        b.alpha = 1.0;
        assert!(b.process(0, 0.4, true, false, false).unwrap().abs() < 1e-6);
        assert!((b.process(0, 0.7, true, false, false).unwrap() - 0.3).abs() < 1e-6);
        assert!(b.process(0, 0.7, true, false, true).unwrap().abs() < 1e-6);
    }

    #[test]
    fn blink_holds_then_decays_and_excludes_ema() {
        let mut b = BrowState::default();
        b.alpha = 1.0;
        b.process(0, 0.0, true, false, false);
        assert!((b.process(0, 0.8, true, false, false).unwrap() - 0.8).abs() < 1e-6);
        for _ in 0..HOLD_FRAMES {
            assert!((b.process(0, 5.0, true, true, false).unwrap() - 0.8).abs() < 1e-6);
        }
        let d = b.process(0, 5.0, true, true, false).unwrap();
        assert!(d < 0.8, "decaying: {d}");
        let r = b.process(0, 0.8, true, false, false).unwrap();
        assert!((r - 0.8).abs() < 1e-6, "post-blink {r}");
    }
}
