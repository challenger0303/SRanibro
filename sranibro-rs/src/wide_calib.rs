//! Guided Dream Air/XR5 EyeWide dataset collection.
//!
//! Every run creates a separate session so train/validation can split by session rather
//! than leaking adjacent frames across a random frame split. Frames stay local under
//! `base_dir()/wide_data/sessions/`; SRanibro never uploads eye images.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Save at 30 stereo pairs/s even though XR5 supplies 120 Hz and the UI polls at 60 Hz.
/// Adjacent 60 Hz images are redundant for training; 30 Hz gives the wearer enough time
/// to settle into each instructed pose and captures more real temporal variation.
const CAPTURE_INTERVAL: Duration = Duration::from_millis(33);

#[derive(Clone, Copy, Debug)]
pub struct CaptureSpec {
    pub folder: &'static str,
    pub wide: f32,
    /// Stereo-pair count (one left + one right image per count).
    pub target: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum Phase {
    Rest {
        secs: f32,
        instruction: &'static str,
    },
    Capture {
        spec: CaptureSpec,
        instruction: &'static str,
    },
    Done,
}

macro_rules! rest {
    ($text:literal) => {
        Phase::Rest {
            secs: 3.0,
            instruction: $text,
        }
    };
}
macro_rules! cap {
    ($folder:literal, $wide:expr, $target:expr, $text:literal) => {
        Phase::Capture {
            spec: CaptureSpec {
                folder: $folder,
                wide: $wide,
                target: $target,
            },
            instruction: $text,
        }
    };
}

/// Stable poses only. Transition frames are excluded by the short REST phases.
/// Upward gaze is deliberately over-sampled because it is the strongest EyeWide confound.
pub const PHASES: &[Phase] = &[
    rest!("Relax naturally and look straight ahead"),
    cap!(
        "neutral_center",
        0.0,
        180,
        "NEUTRAL - look straight ahead, eyelids relaxed"
    ),
    rest!("Prepare a comfortable half-wide expression"),
    cap!(
        "wide_soft",
        0.5,
        150,
        "WIDE SOFT - hold a comfortable half-wide expression"
    ),
    rest!("Prepare your widest comfortable expression"),
    cap!(
        "wide_max",
        1.0,
        150,
        "WIDE MAX - hold both eyes comfortably wide"
    ),
    rest!("Relax your eyelids; next look UP without widening"),
    cap!(
        "gaze_up_neutral",
        0.0,
        240,
        "LOOK UP - keep eyelids relaxed; do not widen"
    ),
    rest!("Relax your eyelids; next look DOWN"),
    cap!(
        "gaze_down_neutral",
        0.0,
        150,
        "LOOK DOWN - keep eyelids relaxed"
    ),
    rest!("Relax your eyelids; next look LEFT"),
    cap!(
        "gaze_left_neutral",
        0.0,
        120,
        "LOOK LEFT - keep eyelids relaxed"
    ),
    rest!("Relax your eyelids; next look RIGHT"),
    cap!(
        "gaze_right_neutral",
        0.0,
        120,
        "LOOK RIGHT - keep eyelids relaxed"
    ),
    rest!("Prepare to hold wide while looking UP"),
    cap!(
        "wide_gaze_up",
        1.0,
        120,
        "WIDE + LOOK UP - keep the wide expression"
    ),
    rest!("Prepare to hold wide while looking DOWN"),
    cap!(
        "wide_gaze_down",
        1.0,
        120,
        "WIDE + LOOK DOWN - keep the wide expression"
    ),
    rest!("Relax; next blink naturally several times"),
    cap!(
        "blink_negative",
        0.0,
        180,
        "BLINK - repeat natural blinks; never widen"
    ),
    rest!("Prepare to close both eyes gently"),
    cap!(
        "closed_negative",
        0.0,
        120,
        "CLOSED - hold both eyes gently closed"
    ),
    rest!("Prepare a firm squint"),
    cap!(
        "squint_negative",
        0.0,
        150,
        "SQUINT - narrow both eyes firmly"
    ),
    rest!("Prepare a LEFT wink"),
    cap!(
        "left_wink_negative",
        0.0,
        120,
        "LEFT WINK - wink only the left eye"
    ),
    rest!("Prepare a RIGHT wink"),
    cap!(
        "right_wink_negative",
        0.0,
        120,
        "RIGHT WINK - wink only the right eye"
    ),
    Phase::Done,
];

pub fn total_target() -> u32 {
    PHASES
        .iter()
        .map(|phase| match phase {
            Phase::Capture { spec, .. } => spec.target,
            _ => 0,
        })
        .sum()
}

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Rest {
        instruction: &'static str,
        remaining: f32,
    },
    Capture {
        instruction: &'static str,
        folder: &'static str,
        captured: u32,
        target: u32,
    },
    Done {
        session: PathBuf,
    },
}

pub struct WideCalib {
    root: PathBuf,
    session: Option<PathBuf>,
    idx: Option<usize>,
    captured: u32,
    total_captured: u32,
    entered: Instant,
    last_saved: Instant,
    seq: u64,
    rows: Vec<String>,
    pub last_error: Option<String>,
}

impl Default for WideCalib {
    fn default() -> Self {
        Self::new()
    }
}

impl WideCalib {
    pub fn new() -> Self {
        Self::with_root(crate::config::base_dir().join("wide_data"))
    }

    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root,
            session: None,
            idx: None,
            captured: 0,
            total_captured: 0,
            entered: Instant::now(),
            last_saved: Instant::now() - CAPTURE_INTERVAL,
            seq: 0,
            rows: Vec::new(),
            last_error: None,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn session(&self) -> Option<&Path> {
        self.session.as_deref()
    }

    pub fn is_running(&self) -> bool {
        matches!(self.idx, Some(i) if !matches!(PHASES[i], Phase::Done))
    }

    pub fn start(&mut self) -> std::io::Result<()> {
        static RUN: AtomicU64 = AtomicU64::new(0);
        let unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let run = RUN.fetch_add(1, Ordering::Relaxed);
        let session = self.root.join("sessions").join(format!(
            "session-{unix_ms:013}-{:010}-{run:06}",
            std::process::id()
        ));
        std::fs::create_dir_all(session.join("images"))?;
        self.session = Some(session);
        self.idx = Some(0);
        self.captured = 0;
        self.total_captured = 0;
        self.entered = Instant::now();
        self.last_saved = Instant::now() - CAPTURE_INTERVAL;
        self.seq = 0;
        self.rows.clear();
        self.last_error = None;
        Ok(())
    }

    /// Delete only the active partial session. Previous completed sessions remain available
    /// for session-level validation.
    pub fn abort(&mut self) {
        if self.is_running() {
            if let Some(session) = &self.session {
                let _ = std::fs::remove_dir_all(session);
            }
        }
        self.session = None;
        self.idx = None;
        self.captured = 0;
        self.total_captured = 0;
        self.rows.clear();
        self.last_error = None;
    }

    pub fn delete_all(&mut self) -> std::io::Result<()> {
        if self.is_running() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "stop the active capture before deleting Wide data",
            ));
        }
        // Delete biometric source images/labels, but keep fitted model artifacts under
        // wide_data/models so the currently configured tracker is not broken.
        let sessions = self.root.join("sessions");
        if sessions.exists() {
            std::fs::remove_dir_all(sessions)?;
        }
        self.session = None;
        self.idx = None;
        Ok(())
    }

    pub fn tick(&mut self) {
        let Some(i) = self.idx else { return };
        if let Phase::Rest { secs, .. } = PHASES[i] {
            if self.entered.elapsed() >= Duration::from_secs_f32(secs) {
                self.advance();
            }
        }
    }

    pub fn on_frame(
        &mut self,
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
    ) -> u32 {
        let Some(i) = self.idx else { return 0 };
        let Phase::Capture { spec, .. } = PHASES[i] else {
            return 0;
        };
        if self.last_saved.elapsed() < CAPTURE_INTERVAL {
            return 0;
        }
        let (Some((lw, lh, lp)), Some((rw, rh, rp))) = (left, right) else {
            return 0;
        };
        if lw == 0
            || lh == 0
            || rw == 0
            || rh == 0
            || lp.len() < lw as usize * lh as usize
            || rp.len() < rw as usize * rh as usize
        {
            return 0;
        }
        let seq = self.seq;
        let l = self.save(spec, 'l', seq, lw, lh, lp, false);
        let r = self.save(spec, 'r', seq, rw, rh, rp, true);
        match (l, r) {
            (Ok(lrow), Ok(rrow)) => {
                self.rows.push(lrow);
                self.rows.push(rrow);
                self.seq += 1;
                self.last_saved = Instant::now();
                self.captured += 1;
                self.total_captured += 1;
                if self.captured >= spec.target {
                    self.advance();
                }
                2
            }
            (Err(e), _) | (_, Err(e)) => {
                self.last_error = Some(e.to_string());
                0
            }
        }
    }

    fn save(
        &self,
        spec: CaptureSpec,
        side: char,
        seq: u64,
        w: u32,
        h: u32,
        pixels: &[u8],
        mirror: bool,
    ) -> std::io::Result<String> {
        let session = self.session.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "Wide capture has no session")
        })?;
        let dir = session.join("images").join(spec.folder);
        std::fs::create_dir_all(&dir)?;
        let name = format!("{}_{}_{seq:08}.png", spec.folder, side);
        write_gray_png(&dir.join(&name), w, h, pixels, mirror)?;
        Ok(format!(
            "{}/{},{},{},{}",
            spec.folder, name, spec.wide, spec.folder, side
        ))
    }

    fn advance(&mut self) {
        let Some(i) = self.idx else { return };
        self.idx = Some((i + 1).min(PHASES.len() - 1));
        self.captured = 0;
        self.entered = Instant::now();
        self.last_saved = Instant::now() - CAPTURE_INTERVAL;
        if matches!(self.idx, Some(i) if matches!(PHASES[i], Phase::Done)) {
            if let Err(e) = self.flush() {
                self.last_error = Some(format!("labels.csv: {e}"));
            }
        }
    }

    fn flush(&self) -> std::io::Result<()> {
        let session = self.session.as_ref().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "Wide capture has no session")
        })?;
        let mut csv = String::from("filename,wide,phase,side\n");
        for row in &self.rows {
            csv.push_str(row);
            csv.push('\n');
        }
        std::fs::write(session.join("labels.csv"), csv)
    }

    pub fn progress(&self) -> f32 {
        self.total_captured as f32 / total_target().max(1) as f32
    }

    pub fn status(&self) -> Status {
        let Some(i) = self.idx else {
            return Status::Idle;
        };
        match PHASES[i] {
            Phase::Rest { secs, instruction } => Status::Rest {
                instruction,
                remaining: (secs - self.entered.elapsed().as_secs_f32()).max(0.0),
            },
            Phase::Capture { spec, instruction } => Status::Capture {
                instruction,
                folder: spec.folder,
                captured: self.captured,
                target: spec.target,
            },
            Phase::Done => Status::Done {
                session: self.session.clone().unwrap_or_else(|| self.root.clone()),
            },
        }
    }
}

fn write_gray_png(path: &Path, w: u32, h: u32, pixels: &[u8], mirror: bool) -> std::io::Result<()> {
    let n = w as usize * h as usize;
    if pixels.len() < n {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "grayscale frame is shorter than width*height",
        ));
    }
    let mirrored;
    let data = if mirror {
        let (w, h) = (w as usize, h as usize);
        let mut out = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                out[y * w + x] = pixels[y * w + (w - 1 - x)];
            }
        }
        mirrored = out;
        &mirrored[..]
    } else {
        &pixels[..n]
    };
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(file, w, h);
    encoder.set_color(png::ColorType::Grayscale);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .map_err(std::io::Error::other)?
        .write_image_data(data)
        .map_err(std::io::Error::other)
}

/// Research-only bridge used to prove that the XR5 audit reverses the exact
/// mirror convention and PNG bytes produced by the real capture writer. It is
/// deliberately unavailable in normal builds.
#[cfg(feature = "research-synthetic-eye-lab")]
#[doc(hidden)]
pub fn research_write_gray_png(
    path: &Path,
    w: u32,
    h: u32,
    pixels: &[u8],
    mirror: bool,
) -> std::io::Result<()> {
    write_gray_png(path, w, h, pixels, mirror)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_creates_independent_session_and_mirrors_right() {
        let root =
            std::env::temp_dir().join(format!("sranibro_wide_capture_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let mut capture = WideCalib::with_root(root.clone());
        capture.start().unwrap();
        capture.idx = Some(1); // first capture phase
        let pixels = vec![0u8, 10, 20, 30, 40, 50];
        assert_eq!(
            capture.on_frame(Some((3, 2, &pixels)), Some((3, 2, &pixels))),
            2
        );
        assert_eq!(
            capture.on_frame(Some((3, 2, &pixels)), Some((3, 2, &pixels))),
            0,
            "capture is paced at 30 stereo pairs/s"
        );
        let session = capture.session().unwrap();
        assert!(session
            .join("images/neutral_center/neutral_center_l_00000000.png")
            .is_file());
        assert!(session
            .join("images/neutral_center/neutral_center_r_00000000.png")
            .is_file());
        capture.flush().unwrap();
        let labels = std::fs::read_to_string(session.join("labels.csv")).unwrap();
        assert!(labels.starts_with("filename,wide,phase,side\n"));
        assert_eq!(labels.lines().count(), 3);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn upward_gaze_negative_is_deliberately_over_sampled() {
        let up = PHASES
            .iter()
            .find_map(|phase| match phase {
                Phase::Capture { spec, .. } if spec.folder == "gaze_up_neutral" => Some(*spec),
                _ => None,
            })
            .unwrap();
        assert_eq!(up.wide, 0.0);
        assert!(up.target >= 240);
    }
}
