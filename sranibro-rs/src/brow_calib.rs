//! B-1: eyebrow-calibration data-collection state machine + on-disk writer.
//!
//! Records training data for the eyebrow CNN in the *exact* layout the existing
//! `vr_eyebrow` Python `train.py` / `dataset.py` consumes unchanged, so a captured
//! `brow_data/` dir is drop-in for offline training (B-2, a later task):
//!
//! ```text
//! base_dir()/brow_data/
//!   images/{folder}/{folder}_{l|r}_{seq}.png   RAW full eye frame, grayscale 8-bit PNG
//!   labels.csv                                  header: filename,brow,inner,outer
//!   reference_left.png / reference_right.png    neutral reference, captured mid-NEUTRAL
//! ```
//!
//! We save the RAW, PRE-CROP full frame (not the model's 64x64 input) so training
//! preprocesses identically to inference — [`crate::ml::preprocess::brow_input`] does
//! the ROI-crop -> PIL-bilinear -> CLAHE -> z-score at train time and is left untouched.
//!
//! Convention parity with vr_eyebrow `save_calibration_frame`: the LEFT eye is saved
//! as-is; the RIGHT eye's full frame is horizontally MIRRORED before saving, so every
//! `_l_` and `_r_` image arrives in the model's left-canonical orientation. The `{seq}`
//! is a monotonic per-session counter (zero-padded) — uniqueness never relies on the
//! wall clock (two frames can share a millisecond at 120Hz).
//!
//! The phase machine is deliberately camera-agnostic: [`BrowCalib::save_frame`] takes an
//! explicit `(w, h, bytes)` grayscale frame, so the whole capture flow is unit-testable
//! with synthetic frames (see the tests at the bottom) with no live device.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Which eye a saved frame belongs to (drives the `_l_`/`_r_` filename tag and the
/// right-eye mirror-on-save).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn tag(self) -> &'static str {
        match self {
            Side::Left => "l",
            Side::Right => "r",
        }
    }
}

/// One capture step: the folder/label + the target frame count for that expression.
/// `label = (brow, inner, outer)`. Replicated verbatim from vr_eyebrow gui.py v2 protocol
/// (`_map_folder_to_targets`): inner/outer are always 0 in the brow-only protocol.
#[derive(Clone, Copy, Debug)]
pub struct CaptureSpec {
    pub folder: &'static str,
    pub brow: f32,
    pub inner: f32,
    pub outer: f32,
    pub target: u32,
}

/// A phase in the guided sequence. REST is a wall-clock pause (no capture); CAPTURE
/// records frames until its target is reached; DONE is terminal.
#[derive(Clone, Copy, Debug)]
pub enum Phase {
    /// Prepare / hold pose. `secs` = countdown; `instruction` shown big.
    Rest {
        secs: f32,
        instruction: &'static str,
    },
    /// Record `spec.target` frames. `instruction` shown big.
    Capture {
        spec: CaptureSpec,
        instruction: &'static str,
    },
    /// All phases complete.
    Done,
}

/// The full guided sequence, in order — values match vr_eyebrow gui.py `calib_states`.
/// NEUTRAL grabs a reference frame mid-way (see [`REFERENCE_AT`]).
pub const PHASES: &[Phase] = &[
    Phase::Rest {
        secs: 4.0,
        instruction: "Get ready — relax, look straight ahead",
    },
    Phase::Capture {
        spec: CaptureSpec {
            folder: "neutral",
            brow: 0.0,
            inner: 0.0,
            outer: 0.0,
            target: 1500,
        },
        instruction: "NEUTRAL — relax fully, gaze straight ahead",
    },
    Phase::Rest {
        secs: 4.0,
        instruction: "Get ready — brow UP next",
    },
    Phase::Capture {
        spec: CaptureSpec {
            folder: "brow_up_max",
            brow: 1.0,
            inner: 0.0,
            outer: 0.0,
            target: 1000,
        },
        instruction: "BROW UP MAX — surprise, lift both eyebrows high",
    },
    Phase::Rest {
        secs: 4.0,
        instruction: "Get ready — brow DOWN next",
    },
    Phase::Capture {
        spec: CaptureSpec {
            folder: "brow_down_max",
            brow: -1.0,
            inner: 0.0,
            outer: 0.0,
            target: 1000,
        },
        instruction: "BROW DOWN MAX — frown / angry, push both eyebrows down",
    },
    Phase::Rest {
        secs: 3.0,
        instruction: "Get ready — soft phases next",
    },
    Phase::Capture {
        spec: CaptureSpec {
            folder: "brow_up_soft",
            brow: 0.5,
            inner: 0.0,
            outer: 0.0,
            target: 700,
        },
        instruction: "BROW UP SOFT — half-raise, conversational lift",
    },
    Phase::Rest {
        secs: 3.0,
        instruction: "Get ready — soft frown next",
    },
    Phase::Capture {
        spec: CaptureSpec {
            folder: "brow_down_soft",
            brow: -0.5,
            inner: 0.0,
            outer: 0.0,
            target: 700,
        },
        instruction: "BROW DOWN SOFT — slight frown",
    },
    Phase::Done,
];

/// Grab the NEUTRAL reference frame once its captured count crosses this (mid-phase),
/// mirroring gui.py's "capture a reference at ~half the neutral run".
pub const REFERENCE_AT: u32 = 750;

/// Sum of all capture targets — the denominator for the overall progress bar.
pub fn total_capture_target() -> u32 {
    PHASES
        .iter()
        .map(|p| match p {
            Phase::Capture { spec, .. } => spec.target,
            _ => 0,
        })
        .sum()
}

/// Live status of the collection run, returned to the UI each frame for rendering.
#[derive(Clone, Copy, Debug)]
pub enum Status {
    /// Not started (or aborted back to idle).
    Idle,
    /// Counting down a rest phase: `remaining` seconds left.
    Rest {
        instruction: &'static str,
        remaining: f32,
    },
    /// Capturing: `captured/target` for this phase.
    Capture {
        instruction: &'static str,
        folder: &'static str,
        captured: u32,
        target: u32,
    },
    /// Finished — points the user at the saved dir.
    Done,
}

/// The eyebrow-calibration data-collection controller (owned by the egui `App`).
///
/// Drive it from the app's per-frame `update()`:
///   * [`BrowCalib::start`] on the Start button,
///   * [`BrowCalib::tick`] every frame (advances rest phases by wall clock),
///   * [`BrowCalib::on_frame`] when a NEW camera frame arrives during a capture phase,
///   * [`BrowCalib::abort`] on the Abort button (deletes the partial session dir).
pub struct BrowCalib {
    /// Root of the capture layout (`base_dir()/brow_data`).
    root: PathBuf,
    /// Index into [`PHASES`]. `None` == not running (idle).
    idx: Option<usize>,
    /// Frames captured in the CURRENT capture phase.
    captured: u32,
    /// Wall-clock instant the current phase was entered (rest countdown source).
    entered: Instant,
    /// Monotonic per-session sequence counter for unique filenames.
    seq: u64,
    /// True once the NEUTRAL reference frames have been written (one-shot).
    ref_saved: bool,
    /// Buffered CSV rows (`filename,brow,inner,outer`), flushed to labels.csv on completion.
    /// Kept in memory during the run and written once at DONE so a mid-run crash leaves no
    /// half-written CSV pointing at missing images.
    rows: Vec<String>,
    /// Last non-fatal error (surfaced in the UI), e.g. a failed PNG write.
    pub last_error: Option<String>,
}

impl BrowCalib {
    /// Create an idle controller rooted at `base_dir()/brow_data`.
    pub fn new() -> Self {
        Self::with_root(crate::config::base_dir().join("brow_data"))
    }

    /// Testable constructor: root the layout at an explicit dir (tests pass a temp dir).
    pub fn with_root(root: PathBuf) -> Self {
        Self {
            root,
            idx: None,
            captured: 0,
            entered: Instant::now(),
            seq: 0,
            ref_saved: false,
            rows: Vec::new(),
            last_error: None,
        }
    }

    /// The capture root (`.../brow_data`).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// True while a capture session is active (not idle, not done).
    pub fn is_running(&self) -> bool {
        matches!(self.idx, Some(i) if !matches!(PHASES[i], Phase::Done))
    }

    /// True once the whole sequence has completed.
    pub fn is_done(&self) -> bool {
        matches!(self.idx, Some(i) if matches!(PHASES[i], Phase::Done))
    }

    /// Frames captured so far across ALL completed + current capture phases (progress
    /// numerator). Derived from the buffered rows (2 rows per captured frame, L+R) plus the
    /// current phase's in-progress count.
    pub fn total_captured(&self) -> u32 {
        // rows holds 2 entries (L, R) per captured frame across finished phases; the current
        // phase's frames are already double-counted in rows, so derive purely from rows.
        (self.rows.len() / 2) as u32
    }

    /// Remove capture-only artifacts while preserving trained models. Historically this
    /// deleted the whole `brow_data` directory, including an active `brow.bin`, so the
    /// next app restart silently lost eyebrow tracking after a new recording.
    fn clear_capture_artifacts(&self) -> std::io::Result<()> {
        let images = self.root.join("images");
        if images.exists() {
            std::fs::remove_dir_all(images)?;
        }
        for name in [
            "labels.csv",
            "train.csv",
            "val.csv",
            "reference_left.png",
            "reference_right.png",
        ] {
            let path = self.root.join(name);
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }

    /// Begin a fresh run: wipe the previous capture only, reset state, enter phase 0.
    pub fn start(&mut self) -> std::io::Result<()> {
        // Fresh dataset each run (a re-run should not mix with stale images / labels).
        self.clear_capture_artifacts()?;
        std::fs::create_dir_all(self.root.join("images"))?;
        self.idx = Some(0);
        self.captured = 0;
        self.entered = Instant::now();
        self.seq = 0;
        self.ref_saved = false;
        self.rows.clear();
        self.last_error = None;
        Ok(())
    }

    /// Abort the run and delete the partial capture (best-effort). Trained models survive.
    pub fn abort(&mut self) {
        let _ = self.clear_capture_artifacts();
        self.idx = None;
        self.captured = 0;
        self.rows.clear();
        self.ref_saved = false;
        self.last_error = None;
    }

    /// Advance wall-clock-driven phases (REST). Call every UI frame. Capture phases advance
    /// via [`BrowCalib::on_frame`]; this is a no-op for them.
    pub fn tick(&mut self) {
        let Some(i) = self.idx else { return };
        if let Phase::Rest { secs, .. } = PHASES[i] {
            if self.entered.elapsed() >= Duration::from_secs_f32(secs) {
                self.advance();
            }
        }
    }

    /// A NEW stereo frame arrived — save both eyes if we're in a capture phase (and the
    /// frames are non-empty). Returns the number of images written (0 or 2). A `None`/empty
    /// eye skips the whole tick without advancing (so we never write a lopsided pair).
    ///
    /// `l`/`r` are `(w, h, grayscale_bytes)` at the frame's own resolution.
    pub fn on_frame(&mut self, l: Option<(u32, u32, &[u8])>, r: Option<(u32, u32, &[u8])>) -> u32 {
        let Some(i) = self.idx else { return 0 };
        let Phase::Capture { spec, .. } = PHASES[i] else {
            return 0;
        };
        let (Some((lw, lh, lpx)), Some((rw, rh, rpx))) = (l, r) else {
            return 0;
        };
        if lw == 0 || lh == 0 || rw == 0 || rh == 0 {
            return 0;
        }
        if lpx.len() < (lw as usize) * (lh as usize) || rpx.len() < (rw as usize) * (rh as usize) {
            return 0;
        }

        // NEUTRAL reference: grab once when this phase crosses REFERENCE_AT (mid-phase).
        if spec.folder == "neutral" && !self.ref_saved && self.captured >= REFERENCE_AT {
            let lref = self.root.join("reference_left.png");
            let rref = self.root.join("reference_right.png");
            // Left as-is, right mirrored (left-canonical), same as the per-image convention.
            if let Err(e) = write_gray_png(&lref, lw, lh, lpx, false) {
                self.last_error = Some(format!("reference_left.png: {e}"));
            }
            if let Err(e) = write_gray_png(&rref, rw, rh, rpx, true) {
                self.last_error = Some(format!("reference_right.png: {e}"));
            }
            self.ref_saved = true;
        }

        let seq = self.seq;
        let mut written = 0;
        // Left eye: saved as-is.
        match self.save_frame(spec, Side::Left, seq, lw, lh, lpx) {
            Ok(()) => written += 1,
            Err(e) => self.last_error = Some(e.to_string()),
        }
        // Right eye: horizontally mirrored before saving (left-canonical).
        match self.save_frame(spec, Side::Right, seq, rw, rh, rpx) {
            Ok(()) => written += 1,
            Err(e) => self.last_error = Some(e.to_string()),
        }
        // Only advance the counters if BOTH eyes wrote (keep L/R paired + labels consistent).
        if written == 2 {
            self.seq += 1;
            self.captured += 1;
            if self.captured >= spec.target {
                self.advance();
            }
        }
        written
    }

    /// Save one eye's RAW frame as grayscale PNG and buffer its labels.csv row. The right eye
    /// is mirrored on write; the CSV `filename` is relative to `images/`. This is pure I/O +
    /// bookkeeping (no phase logic), so tests can call it directly with synthetic frames.
    pub fn save_frame(
        &mut self,
        spec: CaptureSpec,
        side: Side,
        seq: u64,
        w: u32,
        h: u32,
        px: &[u8],
    ) -> std::io::Result<()> {
        let dir = self.root.join("images").join(spec.folder);
        std::fs::create_dir_all(&dir)?;
        // {folder}_{l|r}_{seq}.png, zero-padded seq (monotonic, not wall-clock).
        let fname = format!("{}_{}_{:08}.png", spec.folder, side.tag(), seq);
        let full = dir.join(&fname);
        write_gray_png(&full, w, h, px, side == Side::Right)?;
        // CSV filename is relative to images/ (e.g. "neutral/neutral_l_00000000.png").
        let rel = format!("{}/{}", spec.folder, fname);
        self.rows.push(format!(
            "{},{},{},{}",
            rel, spec.brow, spec.inner, spec.outer
        ));
        Ok(())
    }

    /// Move to the next phase; on entering DONE, flush labels.csv.
    fn advance(&mut self) {
        let Some(i) = self.idx else { return };
        let next = i + 1;
        if next >= PHASES.len() {
            self.idx = Some(PHASES.len() - 1); // clamp at DONE
        } else {
            self.idx = Some(next);
        }
        self.captured = 0;
        self.entered = Instant::now();
        if self.is_done() {
            if let Err(e) = self.flush_labels() {
                self.last_error = Some(format!("labels.csv: {e}"));
            }
        }
    }

    /// Write the accumulated rows to `labels.csv` with the train.py-compatible header.
    fn flush_labels(&self) -> std::io::Result<()> {
        let mut out = String::with_capacity(self.rows.len() * 40 + 32);
        out.push_str("filename,brow,inner,outer\n");
        for row in &self.rows {
            out.push_str(row);
            out.push('\n');
        }
        std::fs::write(self.root.join("labels.csv"), out)
    }

    /// Current status snapshot for the UI (instruction text, counters, remaining time).
    pub fn status(&self) -> Status {
        let Some(i) = self.idx else {
            return Status::Idle;
        };
        match PHASES[i] {
            Phase::Rest { secs, instruction } => {
                let remaining = (secs - self.entered.elapsed().as_secs_f32()).max(0.0);
                Status::Rest {
                    instruction,
                    remaining,
                }
            }
            Phase::Capture { spec, instruction } => Status::Capture {
                instruction,
                folder: spec.folder,
                captured: self.captured,
                target: spec.target,
            },
            Phase::Done => Status::Done,
        }
    }
}

impl Default for BrowCalib {
    fn default() -> Self {
        Self::new()
    }
}

/// Split a captured `labels.csv` into `train.csv` + `val.csv` next to it, in-place under
/// `data_dir` (the `brow_data` root). Pure Rust, no deps.
///
/// Strategy: **temporal 80/20 per (folder × side)**. Rows are grouped by
/// `(folder, side)` where `folder` is the first path segment of `filename` and `side` is
/// `l` (filename contains `_l_`) or `r` (`_r_`). Within each group rows are kept in
/// capture order — the zero-padded `{seq}` in the filename IS the temporal order, so we
/// sort by it defensively (the file is already in order, but a shuffled/edited CSV is
/// handled too). The first 80% of each group goes to train, the last 20% to val; a group
/// with fewer than 5 rows goes ENTIRELY to train (too small to spare a val holdout).
///
/// This keeps train/val from the SAME time-window of a pose out of each other (a random
/// per-row split would leak near-duplicate adjacent frames across the boundary), while
/// guaranteeing every (folder, side) is represented in train.
///
/// Returns `(train_path, val_path)` on success. Errors on a missing/malformed
/// `labels.csv` (missing header, no data rows).
pub fn split_labels(data_dir: &Path) -> std::io::Result<(PathBuf, PathBuf)> {
    let labels = data_dir.join("labels.csv");
    let text = std::fs::read_to_string(&labels)?;
    let (header, rows) = parse_labels(&text).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {e}", labels.display()),
        )
    })?;

    let (train, val) = partition_rows(rows);

    let write = |name: &str, group: &[&str]| -> std::io::Result<PathBuf> {
        let path = data_dir.join(name);
        let mut out = String::with_capacity(
            header.len() + group.iter().map(|r| r.len() + 1).sum::<usize>() + 2,
        );
        out.push_str(header);
        out.push('\n');
        for r in group {
            out.push_str(r);
            out.push('\n');
        }
        std::fs::write(&path, out)?;
        Ok(path)
    };
    let train_path = write("train.csv", &train)?;
    let val_path = write("val.csv", &val)?;
    Ok((train_path, val_path))
}

/// Parse a labels CSV into `(header_line, data_rows)`. Rows are returned as borrowed
/// slices of the input (no allocation per row). Errors on an empty file or one with no
/// data rows so the caller fails fast rather than training on nothing.
fn parse_labels(text: &str) -> Result<(&str, Vec<&str>), String> {
    let mut lines = text.lines();
    let header = lines.next().ok_or("empty labels.csv")?;
    if !header.starts_with("filename,") {
        return Err(format!(
            "unexpected header '{header}' (want 'filename,brow,inner,outer')"
        ));
    }
    let rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    if rows.is_empty() {
        return Err("labels.csv has a header but no data rows".into());
    }
    Ok((header, rows))
}

/// Group by (folder, side), sort each group by the numeric `{seq}` in the filename, then
/// take first 80% -> train / last 20% -> val (groups with <5 rows go entirely to train).
/// Returns `(train_rows, val_rows)` preserving each group's temporal order. No row is lost.
fn partition_rows(rows: Vec<&str>) -> (Vec<&str>, Vec<&str>) {
    use std::collections::BTreeMap;
    // BTreeMap keeps a deterministic (folder,side)-sorted iteration for stable output.
    let mut groups: BTreeMap<(String, char), Vec<&str>> = BTreeMap::new();
    for r in rows {
        let fname = r.split(',').next().unwrap_or("");
        let folder = fname.split(['/', '\\']).next().unwrap_or("").to_string();
        let side = if fname.contains("_r_") {
            'r'
        } else {
            // Default to 'l' for `_l_` (and anything unexpected) so no row is dropped.
            'l'
        };
        groups.entry((folder, side)).or_default().push(r);
    }
    let mut train = Vec::new();
    let mut val = Vec::new();
    for (_key, mut group) in groups {
        // Temporal order = the numeric seq in the filename (defensive re-sort).
        group.sort_by_key(|r| seq_of(r.split(',').next().unwrap_or("")));
        if group.len() < 5 {
            train.extend(group);
            continue;
        }
        // First 80% -> train, last 20% -> val (integer floor; >=1 val row for >=5).
        let cut = (group.len() * 4) / 5;
        train.extend(&group[..cut]);
        val.extend(&group[cut..]);
    }
    (train, val)
}

/// Extract the zero-padded `{seq}` from a filename like `neutral/neutral_l_00000042.png`
/// (the last `_`-delimited number before the extension). Falls back to 0 if absent, so a
/// hand-edited filename never panics the split.
fn seq_of(fname: &str) -> u64 {
    let stem = fname.rsplit(['/', '\\']).next().unwrap_or(fname);
    let stem = stem.split('.').next().unwrap_or(stem);
    stem.rsplit('_')
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0)
}

/// Encode a `w`x`h` grayscale (8-bit) buffer as a PNG at `path`. If `mirror`, horizontally
/// flip each row first (right-eye left-canonical convention). Pure-Rust `png` crate — no
/// opencv/image-crate, honoring the distribution constraint.
fn write_gray_png(path: &Path, w: u32, h: u32, px: &[u8], mirror: bool) -> std::io::Result<()> {
    let (wu, hu) = (w as usize, h as usize);
    let need = wu * hu;
    if px.len() < need {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame smaller than w*h",
        ));
    }
    // Build the row-major buffer we actually encode (mirror applied here so the PNG on disk
    // is already left-canonical for the right eye).
    let data: Vec<u8> = if mirror {
        let mut out = vec![0u8; need];
        for y in 0..hu {
            let row = y * wu;
            for x in 0..wu {
                out[row + (wu - 1 - x)] = px[row + x];
            }
        }
        out
    } else {
        px[..need].to_vec()
    };

    let file = std::fs::File::create(path)?;
    let wtr = std::io::BufWriter::new(file);
    let mut enc = png::Encoder::new(wtr, w, h);
    enc.set_color(png::ColorType::Grayscale);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc
        .write_header()
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    writer
        .write_image_data(&data)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A distinct scratch dir per test (isolated, cleaned at the end).
    fn tmp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "sranibro_browcalib_test_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn synth(w: u32, h: u32, fill: u8) -> Vec<u8> {
        vec![fill; (w * h) as usize]
    }

    #[test]
    fn phase_table_shape_and_labels() {
        // First is REST, last is DONE, and the capture labels/folders match the v2 protocol.
        assert!(matches!(PHASES[0], Phase::Rest { .. }));
        assert!(matches!(PHASES[PHASES.len() - 1], Phase::Done));
        let caps: Vec<CaptureSpec> = PHASES
            .iter()
            .filter_map(|p| match p {
                Phase::Capture { spec, .. } => Some(*spec),
                _ => None,
            })
            .collect();
        assert_eq!(caps.len(), 5, "5 capture phases");
        let by = |f: &str| caps.iter().find(|c| c.folder == f).copied().unwrap();
        assert_eq!((by("neutral").brow, by("neutral").target), (0.0, 1500));
        assert_eq!(
            (by("brow_up_max").brow, by("brow_up_max").target),
            (1.0, 1000)
        );
        assert_eq!(
            (by("brow_down_max").brow, by("brow_down_max").target),
            (-1.0, 1000)
        );
        assert_eq!(
            (by("brow_up_soft").brow, by("brow_up_soft").target),
            (0.5, 700)
        );
        assert_eq!(
            (by("brow_down_soft").brow, by("brow_down_soft").target),
            (-0.5, 700)
        );
        // inner/outer are always 0 in the brow-only protocol.
        assert!(caps.iter().all(|c| c.inner == 0.0 && c.outer == 0.0));
        assert_eq!(total_capture_target(), 1500 + 1000 + 1000 + 700 + 700);
    }

    #[test]
    fn start_creates_layout_and_enters_first_phase() {
        let mut bc = BrowCalib::with_root(tmp_root("start"));
        bc.start().unwrap();
        assert!(bc.root().join("images").is_dir());
        assert!(bc.is_running());
        assert!(!bc.is_done());
        assert!(matches!(bc.status(), Status::Rest { .. }));
        bc.abort();
    }

    #[test]
    fn save_frame_produces_correct_relative_paths() {
        let root = tmp_root("paths");
        let mut bc = BrowCalib::with_root(root.clone());
        bc.start().unwrap();
        let spec = CaptureSpec {
            folder: "neutral",
            brow: 0.0,
            inner: 0.0,
            outer: 0.0,
            target: 10,
        };
        let px = synth(8, 6, 100);
        bc.save_frame(spec, Side::Left, 3, 8, 6, &px).unwrap();
        bc.save_frame(spec, Side::Right, 3, 8, 6, &px).unwrap();
        // Files exist at the exact expected paths.
        assert!(root.join("images/neutral/neutral_l_00000003.png").is_file());
        assert!(root.join("images/neutral/neutral_r_00000003.png").is_file());
        // Buffered CSV rows use the images/-relative path + the label triple.
        assert_eq!(bc.rows.len(), 2);
        assert_eq!(bc.rows[0], "neutral/neutral_l_00000003.png,0,0,0");
        assert_eq!(bc.rows[1], "neutral/neutral_r_00000003.png,0,0,0");
        bc.abort();
    }

    #[test]
    fn on_frame_advances_and_writes_reference_at_midpoint() {
        // Drive NEUTRAL to completion with a tiny target via a hand-rolled controller so the
        // test is fast: we can't change PHASES, so use a small target by feeding frames and
        // checking the reference lands at REFERENCE_AT. To keep it quick we validate the
        // reference logic against the first capture phase using the real target boundary.
        let root = tmp_root("onframe");
        let mut bc = BrowCalib::with_root(root.clone());
        bc.start().unwrap();
        // Skip the initial REST by forcing the phase index to the first Capture (neutral).
        // (In the app, tick() advances REST by wall clock; here we jump directly.)
        bc.idx = Some(1);
        bc.captured = 0;
        assert!(matches!(
            bc.status(),
            Status::Capture {
                folder: "neutral",
                ..
            }
        ));

        let (w, h) = (10u32, 10u32);
        let px = synth(w, h, 128);
        // Feed exactly REFERENCE_AT frames: reference must NOT exist until the crossing frame.
        for _ in 0..REFERENCE_AT {
            let n = bc.on_frame(Some((w, h, &px)), Some((w, h, &px)));
            assert_eq!(n, 2);
        }
        // The frame at index REFERENCE_AT triggers the reference (captured has reached it).
        assert!(!bc.root().join("reference_left.png").exists());
        bc.on_frame(Some((w, h, &px)), Some((w, h, &px)));
        assert!(bc.root().join("reference_left.png").is_file());
        assert!(bc.root().join("reference_right.png").is_file());
        // Empty/None frames must be a no-op (no counter advance).
        let before = bc.captured;
        assert_eq!(bc.on_frame(None, Some((w, h, &px))), 0);
        assert_eq!(bc.on_frame(Some((0, 0, &[])), Some((w, h, &px))), 0);
        assert_eq!(bc.captured, before, "bad frames don't advance the counter");
        bc.abort();
    }

    #[test]
    fn completing_a_capture_phase_advances_to_next() {
        let root = tmp_root("advance");
        let mut bc = BrowCalib::with_root(root.clone());
        bc.start().unwrap();
        // Jump to the last (small) capture phase: brow_down_soft, target 700.
        let last_cap = PHASES
            .iter()
            .rposition(|p| matches!(p, Phase::Capture { .. }))
            .unwrap();
        bc.idx = Some(last_cap);
        bc.captured = 0;
        let (w, h) = (4u32, 4u32);
        let px = synth(w, h, 64);
        // Fill it to target; the next tick after target should move to DONE.
        for _ in 0..700 {
            bc.on_frame(Some((w, h, &px)), Some((w, h, &px)));
        }
        assert!(bc.is_done(), "reaching target advances to DONE");
        // DONE flushes labels.csv with the header + one row per saved image.
        let csv = std::fs::read_to_string(root.join("labels.csv")).unwrap();
        let mut lines = csv.lines();
        assert_eq!(lines.next(), Some("filename,brow,inner,outer"));
        let rows: Vec<&str> = lines.collect();
        assert_eq!(rows.len(), 700 * 2, "L+R row per captured frame");
        assert!(rows[0].starts_with("brow_down_soft/brow_down_soft_l_"));
        assert!(rows[0].ends_with(",-0.5,0,0"));
        bc.abort();
    }

    #[test]
    fn right_eye_png_is_mirrored_on_disk() {
        // Encode a left/right-asymmetric frame; the right-eye PNG must be the horizontal
        // mirror of the left-eye PNG. We decode both back with the png crate and compare.
        let root = tmp_root("mirror");
        let mut bc = BrowCalib::with_root(root.clone());
        bc.start().unwrap();
        let (w, h) = (4u32, 2u32);
        // Row pattern 0,1,2,3 — mirror is 3,2,1,0.
        let mut px = vec![0u8; (w * h) as usize];
        for y in 0..h as usize {
            for x in 0..w as usize {
                px[y * w as usize + x] = (x * 10) as u8;
            }
        }
        let spec = CaptureSpec {
            folder: "neutral",
            brow: 0.0,
            inner: 0.0,
            outer: 0.0,
            target: 10,
        };
        bc.save_frame(spec, Side::Left, 0, w, h, &px).unwrap();
        bc.save_frame(spec, Side::Right, 0, w, h, &px).unwrap();

        let decode = |p: PathBuf| {
            let dec = png::Decoder::new(std::fs::File::open(p).unwrap());
            let mut reader = dec.read_info().unwrap();
            let mut buf = vec![0u8; reader.output_buffer_size()];
            let info = reader.next_frame(&mut buf).unwrap();
            buf.truncate(info.buffer_size());
            buf
        };
        let lbuf = decode(root.join("images/neutral/neutral_l_00000000.png"));
        let rbuf = decode(root.join("images/neutral/neutral_r_00000000.png"));
        // Left is as-is.
        assert_eq!(&lbuf, &px);
        // Right is each row reversed.
        for y in 0..h as usize {
            for x in 0..w as usize {
                assert_eq!(
                    rbuf[y * w as usize + x],
                    px[y * w as usize + (w as usize - 1 - x)]
                );
            }
        }
        bc.abort();
    }

    /// Build a synthetic labels.csv body (no header) with `n` sequential frames per
    /// (folder, side): rows are `folder/folder_{side}_{seq:08}.png,brow,0,0`.
    fn synth_labels(folders: &[(&str, f32, u32)]) -> String {
        let mut s = String::from("filename,brow,inner,outer\n");
        for (folder, brow, n) in folders {
            for side in ['l', 'r'] {
                for seq in 0..*n {
                    s.push_str(&format!(
                        "{folder}/{folder}_{side}_{seq:08}.png,{brow},0,0\n"
                    ));
                }
            }
        }
        s
    }

    #[test]
    fn split_is_temporal_80_20_per_group_and_loses_no_row() {
        let root = tmp_root("split");
        std::fs::create_dir_all(&root).unwrap();
        // Two folders: one big (10/side -> 8 train + 2 val), one tiny (3/side -> all train).
        let body = synth_labels(&[("neutral", 0.0, 10), ("brow_up_soft", 0.5, 3)]);
        std::fs::write(root.join("labels.csv"), &body).unwrap();

        let (train_p, val_p) = split_labels(&root).unwrap();
        let train = std::fs::read_to_string(&train_p).unwrap();
        let val = std::fs::read_to_string(&val_p).unwrap();

        // Both files carry the exact header.
        assert!(train.starts_with("filename,brow,inner,outer\n"));
        assert!(val.starts_with("filename,brow,inner,outer\n"));
        let train_rows: Vec<&str> = train.lines().skip(1).collect();
        let val_rows: Vec<&str> = val.lines().skip(1).collect();

        // No row lost: train + val == the original data rows.
        let orig_rows: Vec<&str> = body.lines().skip(1).collect();
        assert_eq!(train_rows.len() + val_rows.len(), orig_rows.len());
        // neutral: 10/side -> cut=8, so 2 val per side = 4 val rows total; the tiny
        // brow_up_soft group (3/side < 5) contributes 0 val rows.
        assert_eq!(
            val_rows.len(),
            4,
            "only the big group yields val: {val_rows:?}"
        );
        // train = neutral (16) + all of the tiny brow_up_soft group (6) = 22.
        assert_eq!(train_rows.len(), 16 + 6);
        // Every val row is from the big group, and each is a LATE seq (>= the cut of 8).
        for r in &val_rows {
            assert!(
                r.starts_with("neutral/"),
                "val only from the >=5 group: {r}"
            );
            let seq = seq_of(r.split(',').next().unwrap());
            assert!(seq >= 8, "val holdout is the temporal tail (seq {seq})");
        }
        // The tiny group survives entirely in train (both sides present).
        assert!(train_rows.iter().any(|r| r.contains("brow_up_soft_l_")));
        assert!(train_rows.iter().any(|r| r.contains("brow_up_soft_r_")));
        // No row duplicated across the two files.
        let mut all: Vec<&str> = train_rows.iter().chain(val_rows.iter()).copied().collect();
        all.sort_unstable();
        let before = all.len();
        all.dedup();
        assert_eq!(all.len(), before, "no row appears in both train and val");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn split_rejects_missing_and_malformed_labels() {
        let root = tmp_root("split_bad");
        std::fs::create_dir_all(&root).unwrap();
        // Missing file.
        assert!(split_labels(&root).is_err(), "no labels.csv -> Err");
        // Header only, no data rows.
        std::fs::write(root.join("labels.csv"), "filename,brow,inner,outer\n").unwrap();
        assert!(split_labels(&root).is_err(), "header-only -> Err");
        // Wrong header.
        std::fs::write(root.join("labels.csv"), "x,y\na,b\n").unwrap();
        assert!(split_labels(&root).is_err(), "bad header -> Err");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn split_orders_val_after_train_even_if_csv_shuffled() {
        // A CSV whose rows are NOT in seq order must still split temporally (sort by seq).
        let root = tmp_root("split_shuf");
        std::fs::create_dir_all(&root).unwrap();
        let mut s = String::from("filename,brow,inner,outer\n");
        // Interleave seqs out of order for one (folder, side) group of 10.
        for seq in [9u32, 0, 5, 2, 7, 1, 8, 3, 6, 4] {
            s.push_str(&format!("neutral/neutral_l_{seq:08}.png,0,0,0\n"));
        }
        std::fs::write(root.join("labels.csv"), &s).unwrap();
        let (train_p, val_p) = split_labels(&root).unwrap();
        let train = std::fs::read_to_string(&train_p).unwrap();
        let val = std::fs::read_to_string(&val_p).unwrap();
        // cut = 8 -> val is seq 8 and 9 (the two latest), regardless of file order.
        let val_seqs: Vec<u64> = val
            .lines()
            .skip(1)
            .map(|r| seq_of(r.split(',').next().unwrap()))
            .collect();
        assert_eq!(val_seqs, vec![8, 9]);
        // train holds seqs 0..=7 in order.
        let train_seqs: Vec<u64> = train
            .lines()
            .skip(1)
            .map(|r| seq_of(r.split(',').next().unwrap()))
            .collect();
        assert_eq!(train_seqs, (0..=7).collect::<Vec<u64>>());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn start_and_abort_preserve_trained_model() {
        let root = tmp_root("abort");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("brow.bin"), b"keep me").unwrap();
        let mut bc = BrowCalib::with_root(root.clone());
        bc.start().unwrap();
        assert_eq!(std::fs::read(root.join("brow.bin")).unwrap(), b"keep me");
        let spec = CaptureSpec {
            folder: "neutral",
            brow: 0.0,
            inner: 0.0,
            outer: 0.0,
            target: 10,
        };
        bc.save_frame(spec, Side::Left, 0, 4, 4, &synth(4, 4, 1))
            .unwrap();
        assert!(root.exists());
        bc.abort();
        assert!(root.exists(), "model directory survives abort");
        assert_eq!(std::fs::read(root.join("brow.bin")).unwrap(), b"keep me");
        assert!(!root.join("images").exists(), "partial images are deleted");
        assert!(!bc.is_running());
        let _ = std::fs::remove_dir_all(&root);
    }
}
