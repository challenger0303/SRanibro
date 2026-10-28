//! Non-blocking notification sounds for calibration recordings.
//!
//! The UI publishes only state transitions. A bounded worker queue plays short,
//! distinguishable in-memory WAV earcons through the current default audio device.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cue {
    Prepare,
    Ready,
    Sampling,
    Holdout,
    Paused,
    Resumed,
    Complete,
    Saved,
    Cancelled,
    Warning,
    DiagnosticStarted,
    DiagnosticStopped,
}

impl Cue {
    fn tone(self) -> &'static [(u16, u16)] {
        match self {
            Self::Prepare | Self::Ready => &[(660, 70)],
            Self::Sampling => &[(980, 75)],
            Self::Holdout => &[(620, 65), (820, 65)],
            Self::Paused => &[(520, 90), (420, 110)],
            Self::Resumed => &[(520, 70), (720, 90)],
            Self::Complete | Self::Saved => &[(620, 80), (780, 80), (980, 120)],
            Self::Cancelled => &[(520, 90), (360, 130)],
            Self::Warning => &[(360, 150), (360, 150)],
            Self::DiagnosticStarted => &[(700, 70), (900, 100)],
            Self::DiagnosticStopped => &[(900, 70), (700, 100)],
        }
    }
}

pub struct RecordingAudio {
    shared: Arc<Shared>,
}

struct Shared {
    queue: Mutex<VecDeque<Cue>>,
    wake: Condvar,
    closed: AtomicBool,
}

impl RecordingAudio {
    pub fn new() -> Self {
        let shared = Arc::new(Shared {
            queue: Mutex::new(VecDeque::new()),
            wake: Condvar::new(),
            closed: AtomicBool::new(false),
        });
        let worker_shared = shared.clone();
        let _ = std::thread::Builder::new()
            .name("recording-audio".into())
            .spawn(move || worker(worker_shared));
        Self { shared }
    }

    /// Queue a cue without waiting on the audio device. Terminal and warning cues replace
    /// stale phase markers; duplicate sampling ticks are deliberately lossy.
    pub fn cue(&self, cue: Cue, enabled: bool) {
        if !enabled {
            return;
        }
        let Ok(mut queue) = self.shared.queue.lock() else {
            return;
        };
        if matches!(
            cue,
            Cue::Complete | Cue::Saved | Cue::Cancelled | Cue::Warning
        ) {
            queue.clear();
        } else if matches!(cue, Cue::Prepare | Cue::Ready | Cue::Holdout) {
            queue.retain(|queued| {
                !matches!(
                    queued,
                    Cue::Prepare | Cue::Ready | Cue::Sampling | Cue::Holdout
                )
            });
        } else if cue == Cue::Sampling && queue.iter().any(|queued| *queued == Cue::Sampling) {
            return;
        }
        if queue.len() >= 8 {
            queue.pop_front();
        }
        queue.push_back(cue);
        drop(queue);
        self.shared.wake.notify_one();
    }
}

impl Default for RecordingAudio {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RecordingAudio {
    fn drop(&mut self) {
        self.shared.closed.store(true, Ordering::Release);
        self.shared.wake.notify_one();
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let cue = {
            let Ok(mut queue) = shared.queue.lock() else {
                return;
            };
            while queue.is_empty() && !shared.closed.load(Ordering::Acquire) {
                let Ok(next) = shared.wake.wait(queue) else {
                    return;
                };
                queue = next;
            }
            if shared.closed.load(Ordering::Acquire) && queue.is_empty() {
                return;
            }
            queue.pop_front()
        };
        let Some(cue) = cue else {
            continue;
        };
        #[cfg(windows)]
        windows_audio::play_tones(cue.tone());
        #[cfg(not(windows))]
        let _ = cue;
    }
}

#[cfg(windows)]
mod windows_audio {
    use windows_sys::Win32::Media::Audio::{PlaySoundW, SND_MEMORY, SND_NODEFAULT, SND_SYNC};

    pub fn play_tones(pattern: &[(u16, u16)]) {
        let wav = tone_wav(pattern);
        unsafe {
            PlaySoundW(
                wav.as_ptr().cast::<u16>(),
                std::ptr::null_mut(),
                SND_MEMORY | SND_SYNC | SND_NODEFAULT,
            );
        }
    }

    fn tone_wav(pattern: &[(u16, u16)]) -> Vec<u8> {
        const RATE: u32 = 16_000;
        const AMPLITUDE: f32 = 0.16 * i16::MAX as f32;
        let gap_samples = (RATE as usize * 35) / 1000;
        let mut pcm = Vec::<i16>::new();
        for (index, &(frequency, duration_ms)) in pattern.iter().enumerate() {
            let samples = RATE as usize * duration_ms as usize / 1000;
            for n in 0..samples {
                let phase = std::f32::consts::TAU * frequency as f32 * n as f32 / RATE as f32;
                let edge = (samples / 8).max(1);
                let fade = n.min(samples.saturating_sub(1 + n)).min(edge) as f32 / edge as f32;
                pcm.push((phase.sin() * AMPLITUDE * fade.clamp(0.0, 1.0)) as i16);
            }
            if index + 1 != pattern.len() {
                pcm.extend(std::iter::repeat_n(0, gap_samples));
            }
        }
        wav_from_pcm(&pcm, RATE)
    }

    fn wav_from_pcm(pcm: &[i16], rate: u32) -> Vec<u8> {
        let data_len = (pcm.len() * 2) as u32;
        let mut out = Vec::with_capacity(44 + data_len as usize);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2).to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        for sample in pcm {
            out.extend_from_slice(&sample.to_le_bytes());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_cue_has_an_audible_earcon() {
        for cue in [
            Cue::Prepare,
            Cue::Ready,
            Cue::Sampling,
            Cue::Holdout,
            Cue::Paused,
            Cue::Resumed,
            Cue::Complete,
            Cue::Saved,
            Cue::Cancelled,
            Cue::Warning,
            Cue::DiagnosticStarted,
            Cue::DiagnosticStopped,
        ] {
            assert!(!cue.tone().is_empty());
            assert!(cue
                .tone()
                .iter()
                .all(|(frequency, duration)| *frequency > 0 && *duration > 0));
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "manual Windows audio-device smoke test"]
    fn windows_memory_wav_smoke_test() {
        windows_audio::play_tones(&[(620, 70), (820, 90)]);
    }
}
