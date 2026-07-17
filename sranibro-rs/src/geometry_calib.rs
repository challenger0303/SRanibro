//! Guided capture for per-user XR5 eye-model geometry fitting.
//!
//! Raw stereo frames remain in memory unless the user explicitly exports a feedback ZIP.
//! A separate holdout tail is never exposed to the search and is used only to decide
//! whether a candidate is safer than the geometry that was active when capture started.

use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::time::{Duration, Instant};

const SAMPLE_INTERVAL: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleKind {
    Neutral,
    GazeSweep,
    SlowClose,
    NaturalBlinks,
    Closed,
    HalfOpen,
    HoldoutNeutral,
    HoldoutGazeSweep,
    HoldoutSlowClose,
    HoldoutNaturalBlinks,
    HoldoutClosed,
    HoldoutHalfOpen,
}

impl SampleKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Neutral => "neutral",
            Self::GazeSweep => "gaze_sweep",
            Self::SlowClose => "slow_close",
            Self::NaturalBlinks => "natural_blinks",
            Self::Closed => "closed",
            Self::HalfOpen => "half_open",
            Self::HoldoutNeutral => "holdout_neutral",
            Self::HoldoutGazeSweep => "holdout_gaze_sweep",
            Self::HoldoutSlowClose => "holdout_slow_close",
            Self::HoldoutNaturalBlinks => "holdout_natural_blinks",
            Self::HoldoutClosed => "holdout_closed",
            Self::HoldoutHalfOpen => "holdout_half_open",
        }
    }

    pub fn is_holdout(self) -> bool {
        matches!(
            self,
            Self::HoldoutNeutral
                | Self::HoldoutGazeSweep
                | Self::HoldoutSlowClose
                | Self::HoldoutNaturalBlinks
                | Self::HoldoutClosed
                | Self::HoldoutHalfOpen
        )
    }

    pub fn family(self) -> SampleFamily {
        match self {
            Self::Neutral | Self::HoldoutNeutral => SampleFamily::Neutral,
            Self::GazeSweep | Self::HoldoutGazeSweep => SampleFamily::GazeSweep,
            Self::SlowClose | Self::HoldoutSlowClose => SampleFamily::SlowClose,
            Self::NaturalBlinks | Self::HoldoutNaturalBlinks => SampleFamily::NaturalBlinks,
            Self::Closed | Self::HoldoutClosed => SampleFamily::Closed,
            Self::HalfOpen | Self::HoldoutHalfOpen => SampleFamily::HalfOpen,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SampleFamily {
    Neutral,
    GazeSweep,
    SlowClose,
    NaturalBlinks,
    Closed,
    HalfOpen,
}

#[derive(Clone, Debug)]
pub struct GeometrySample {
    pub kind: SampleKind,
    /// Expected open fraction for the metronome-driven slow-close phases and the
    /// explicit held-half evidence. The normal fitter ignores the separate HalfOpen
    /// family; only the diagnostic audit treats its 0.5 label as an absolute target.
    pub expected_open: Option<f32>,
    pub phase_time_s: f32,
    pub left: Vec<u8>,
    pub right: Vec<u8>,
    pub left_size: (u32, u32),
    pub right_size: (u32, u32),
    /// Per-frame brightness affine captured from the live pipeline.  The fitter applies
    /// the configured deterministic filters, then this affine, before trying geometries.
    pub brightness_affine: [[f32; 2]; 2],
    /// Native Tobii openness captured only as a compliance cross-check. It never enters
    /// a geometry score or candidate selection. Reported Disable is represented as 0.
    pub native_open: [Option<f32>; 2],
    /// PHASES index, retained so repeated OPEN/HALF/CLOSED blocks can be compared.
    pub phase_index: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GeometryDataset {
    pub samples: Vec<GeometrySample>,
}

impl GeometryDataset {
    pub fn train_len(&self) -> usize {
        self.samples
            .iter()
            .filter(|sample| !sample.kind.is_holdout())
            .count()
    }

    pub fn holdout_len(&self) -> usize {
        self.samples
            .iter()
            .filter(|sample| sample.kind.is_holdout())
            .count()
    }
}

#[derive(Clone, Copy)]
enum Phase {
    Rest {
        seconds: f32,
        instruction: &'static str,
    },
    Capture {
        seconds: f32,
        kind: SampleKind,
        instruction: &'static str,
    },
    Done,
}

const PHASES: &[Phase] = &[
    Phase::Rest {
        seconds: 3.0,
        instruction: "Wear the headset normally. Look straight ahead and relax your eyelids.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Neutral,
        instruction: "OPEN 1 - look straight ahead with both eyes comfortably open and relaxed.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Next, lower both eyelids to about halfway and hold them steady.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HalfOpen,
        instruction: "HALF 1 - hold both eyelids about halfway, like relaxed sleepy eyes.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Next, close both eyes gently without squeezing.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Closed,
        instruction: "CLOSED 1 - gently close both eyes and hold; do not squeeze.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, reopen and keep your eyelids relaxed while moving only your gaze.",
    },
    Phase::Capture {
        seconds: 8.0,
        kind: SampleKind::GazeSweep,
        instruction: "GAZE SWEEP - slowly look left, right, up, and down. Do not widen or squint.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, follow three slow close/open cycles. Each half takes two seconds.",
    },
    Phase::Capture {
        seconds: 10.0,
        kind: SampleKind::SlowClose,
        instruction:
            "SLOW CLOSE - follow the on-screen target bar through three smooth close/open cycles.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Next, blink naturally five times with a relaxed open pause between blinks.",
    },
    Phase::Capture {
        seconds: 7.0,
        kind: SampleKind::NaturalBlinks,
        instruction: "NATURAL BLINKS - blink five times; fully relax open between blinks.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Repeat the static poses in reverse order. First, close gently.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Closed,
        instruction: "CLOSED 2 - gently close both eyes and hold; do not squeeze.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Again, hold both eyelids about halfway.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::HalfOpen,
        instruction: "HALF 2 - hold both eyelids about halfway and steady.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Reopen both eyes comfortably and relax.",
    },
    Phase::Capture {
        seconds: 4.0,
        kind: SampleKind::Neutral,
        instruction: "OPEN 2 - look straight ahead, comfortably open and relaxed.",
    },
    Phase::Rest {
        seconds: 2.0,
        instruction: "Untouched holdout begins. Keep both eyes naturally open.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutNeutral,
        instruction: "HOLDOUT OPEN - naturally open, looking straight ahead.",
    },
    Phase::Rest {
        seconds: 1.0,
        instruction: "Hold both eyelids about halfway again.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutHalfOpen,
        instruction: "HOLDOUT HALF - hold halfway and steady.",
    },
    Phase::Rest {
        seconds: 1.0,
        instruction: "Close both eyes gently without squeezing.",
    },
    Phase::Capture {
        seconds: 3.0,
        kind: SampleKind::HoldoutClosed,
        instruction: "HOLDOUT CLOSED - keep both eyes gently closed. Do not squeeze.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Reopen. Holdout gaze sweep next with relaxed eyelids.",
    },
    Phase::Capture {
        seconds: 5.0,
        kind: SampleKind::HoldoutGazeSweep,
        instruction: "HOLDOUT GAZE - slowly look left, right, up, and down.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "One slow close/open cycle next.",
    },
    Phase::Capture {
        seconds: 5.0,
        kind: SampleKind::HoldoutSlowClose,
        instruction: "HOLDOUT SLOW CLOSE - follow one smooth close/open cycle.",
    },
    Phase::Rest {
        seconds: 1.5,
        instruction: "Finally, blink naturally three times.",
    },
    Phase::Capture {
        seconds: 6.0,
        kind: SampleKind::HoldoutNaturalBlinks,
        instruction: "HOLDOUT BLINKS - blink three times with open pauses.",
    },
    Phase::Done,
];

#[derive(Clone, Debug)]
pub enum Status {
    Idle,
    Rest {
        instruction: &'static str,
        remaining_s: f32,
        overall: f32,
    },
    Capture {
        instruction: &'static str,
        kind: SampleKind,
        remaining_s: f32,
        phase_progress: f32,
        overall: f32,
        samples: usize,
        target_open: Option<f32>,
        stereo_stalled: bool,
    },
    Done {
        train_samples: usize,
        holdout_samples: usize,
    },
}

pub struct GeometryCapture {
    phase: Option<usize>,
    entered: Instant,
    last_sample: Instant,
    last_generation: [u64; 2],
    dataset: GeometryDataset,
    completed_seconds: f32,
    pub last_error: Option<String>,
}

impl Default for GeometryCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryCapture {
    pub fn new() -> Self {
        Self {
            phase: None,
            entered: Instant::now(),
            last_sample: Instant::now() - SAMPLE_INTERVAL,
            last_generation: [0; 2],
            dataset: GeometryDataset::default(),
            completed_seconds: 0.0,
            last_error: None,
        }
    }

    pub fn start(&mut self, generation: [u64; 2]) {
        self.phase = Some(0);
        self.entered = Instant::now();
        self.last_sample = Instant::now() - SAMPLE_INTERVAL;
        self.last_generation = generation;
        self.dataset.samples.clear();
        self.completed_seconds = 0.0;
        self.last_error = None;
    }

    pub fn abort(&mut self) {
        self.phase = None;
        self.dataset.samples.clear();
        self.completed_seconds = 0.0;
        self.last_error = None;
    }

    pub fn is_running(&self) -> bool {
        matches!(self.phase, Some(index) if !matches!(PHASES[index], Phase::Done))
    }

    pub fn is_done(&self) -> bool {
        matches!(self.phase, Some(index) if matches!(PHASES[index], Phase::Done))
    }

    pub fn tick(&mut self) {
        let Some(index) = self.phase else { return };
        let seconds = match PHASES[index] {
            Phase::Rest { seconds, .. } | Phase::Capture { seconds, .. } => seconds,
            Phase::Done => return,
        };
        if self.entered.elapsed().as_secs_f32() >= seconds {
            self.completed_seconds += seconds;
            self.phase = Some((index + 1).min(PHASES.len() - 1));
            self.entered = Instant::now();
            self.last_sample = Instant::now() - SAMPLE_INTERVAL;
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn on_frame(
        &mut self,
        generation: [u64; 2],
        left: Option<(u32, u32, &[u8])>,
        right: Option<(u32, u32, &[u8])>,
        brightness_affine: [[f32; 2]; 2],
        native_open: [Option<f32>; 2],
    ) -> bool {
        let Some(index) = self.phase else {
            return false;
        };
        let Phase::Capture { seconds, kind, .. } = PHASES[index] else {
            return false;
        };
        if generation[0] <= self.last_generation[0]
            || generation[1] <= self.last_generation[1]
            || self.last_sample.elapsed() < SAMPLE_INTERVAL
        {
            return false;
        }
        let (Some((lw, lh, left)), Some((rw, rh, right))) = (left, right) else {
            return false;
        };
        let l_len = (lw as usize).saturating_mul(lh as usize);
        let r_len = (rw as usize).saturating_mul(rh as usize);
        if lw == 0 || lh == 0 || rw == 0 || rh == 0 || left.len() < l_len || right.len() < r_len {
            self.last_error = Some("The newest stereo eye frame is incomplete.".into());
            return false;
        }

        let elapsed = self.entered.elapsed().as_secs_f32().min(seconds);
        let progress = (elapsed / seconds.max(0.001)).clamp(0.0, 1.0);
        let expected_open = match kind {
            SampleKind::SlowClose => Some(slow_target(progress, 3)),
            SampleKind::HoldoutSlowClose => Some(slow_target(progress, 1)),
            SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen => Some(0.5),
            _ => None,
        };
        self.dataset.samples.push(GeometrySample {
            kind,
            expected_open,
            phase_time_s: elapsed,
            left: left[..l_len].to_vec(),
            right: right[..r_len].to_vec(),
            left_size: (lw, lh),
            right_size: (rw, rh),
            brightness_affine,
            native_open,
            phase_index: index,
        });
        self.last_generation = generation;
        self.last_sample = Instant::now();
        self.last_error = None;
        true
    }

    pub fn take_dataset(&mut self) -> Option<GeometryDataset> {
        if !self.is_done() {
            return None;
        }
        self.phase = None;
        self.completed_seconds = 0.0;
        Some(std::mem::take(&mut self.dataset))
    }

    /// Put a completed in-memory capture back after the background fitter could not
    /// be started. This avoids making the wearer repeat the guided sequence for a
    /// transient thread/model error; no frames are written to disk.
    pub fn restore_completed_dataset(&mut self, dataset: GeometryDataset) {
        self.dataset = dataset;
        self.phase = Some(PHASES.len() - 1);
        self.entered = Instant::now();
        self.completed_seconds = total_seconds();
        self.last_error = None;
    }

    /// Export the exact completed dataset used by the geometry fitter. This is opt-in
    /// because the archive contains raw eye-camera images (biometric data).
    pub fn export_recording(&self, path: &Path, metadata: &str) -> io::Result<()> {
        if !self.is_done() || self.dataset.samples.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "finish the geometry capture before exporting it",
            ));
        }
        export_dataset_zip(path, &self.dataset, metadata)
    }

    pub fn status(&self) -> Status {
        let Some(index) = self.phase else {
            return Status::Idle;
        };
        let elapsed = self.entered.elapsed().as_secs_f32();
        match PHASES[index] {
            Phase::Rest {
                seconds,
                instruction,
            } => Status::Rest {
                instruction,
                remaining_s: (seconds - elapsed).max(0.0),
                overall: self.overall_progress(elapsed.min(seconds)),
            },
            Phase::Capture {
                seconds,
                kind,
                instruction,
            } => {
                let progress = (elapsed / seconds.max(0.001)).clamp(0.0, 1.0);
                Status::Capture {
                    instruction,
                    kind,
                    remaining_s: (seconds - elapsed).max(0.0),
                    phase_progress: progress,
                    overall: self.overall_progress(elapsed.min(seconds)),
                    samples: self.dataset.samples.len(),
                    stereo_stalled: self.last_sample.elapsed() >= Duration::from_secs(1),
                    target_open: match kind {
                        SampleKind::SlowClose => Some(slow_target(progress, 3)),
                        SampleKind::HoldoutSlowClose => Some(slow_target(progress, 1)),
                        SampleKind::HalfOpen | SampleKind::HoldoutHalfOpen => Some(0.5),
                        _ => None,
                    },
                }
            }
            Phase::Done => Status::Done {
                train_samples: self.dataset.train_len(),
                holdout_samples: self.dataset.holdout_len(),
            },
        }
    }

    fn overall_progress(&self, current_seconds: f32) -> f32 {
        ((self.completed_seconds + current_seconds) / total_seconds()).clamp(0.0, 1.0)
    }
}

fn phase_seconds(phase: Phase) -> f32 {
    match phase {
        Phase::Rest { seconds, .. } | Phase::Capture { seconds, .. } => seconds,
        Phase::Done => 0.0,
    }
}

pub fn total_seconds() -> f32 {
    PHASES.iter().copied().map(phase_seconds).sum()
}

fn slow_target(progress: f32, cycles: u32) -> f32 {
    let cycle = ((progress.clamp(0.0, 0.999_999) * cycles.max(1) as f32).fract()).clamp(0.0, 1.0);
    if cycle < 0.5 {
        1.0 - cycle * 2.0
    } else {
        (cycle - 0.5) * 2.0
    }
}

fn csv_optional(value: Option<f32>) -> String {
    value
        .filter(|value| value.is_finite())
        .map(|value| format!("{value:.6}"))
        .unwrap_or_default()
}

fn encode_gray_png(width: u32, height: u32, pixels: &[u8]) -> io::Result<Vec<u8>> {
    let need = (width as usize).saturating_mul(height as usize);
    if width == 0 || height == 0 || pixels.len() < need {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "geometry recording contains an incomplete image",
        ));
    }
    let mut encoded = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut encoded, width, height);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder
            .write_header()
            .map_err(|error| io::Error::other(error.to_string()))?;
        writer
            .write_image_data(&pixels[..need])
            .map_err(|error| io::Error::other(error.to_string()))?;
        writer
            .finish()
            .map_err(|error| io::Error::other(error.to_string()))?;
    }
    Ok(encoded)
}

fn export_dataset_zip(path: &Path, dataset: &GeometryDataset, metadata: &str) -> io::Result<()> {
    let mut csv = String::from(
        "index,kind,holdout,phase_index,phase_time_s,expected_open,left_file,right_file,left_width,left_height,right_width,right_height,left_gain,left_bias,right_gain,right_bias,native_open_left,native_open_right\n",
    );
    for (index, sample) in dataset.samples.iter().enumerate() {
        writeln!(
            csv,
            "{index},{},{},{},{:.6},{},frames/{index:06}_left.png,frames/{index:06}_right.png,{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{}",
            sample.kind.as_str(),
            sample.kind.is_holdout(),
            sample.phase_index,
            sample.phase_time_s,
            csv_optional(sample.expected_open),
            sample.left_size.0,
            sample.left_size.1,
            sample.right_size.0,
            sample.right_size.1,
            sample.brightness_affine[0][0],
            sample.brightness_affine[0][1],
            sample.brightness_affine[1][0],
            sample.brightness_affine[1][1],
            csv_optional(sample.native_open[0]),
            csv_optional(sample.native_open[1]),
        )
        .map_err(|_| io::Error::other("failed to build geometry recording manifest"))?;
    }

    const README: &str = "SRanibro XR5 geometry calibration recording\n\nThis archive contains raw grayscale eye-camera images and therefore biometric data.\nShare it only when you intend to provide debugging or model-fit feedback.\n\nframes/*.png are the exact mapped stereo frames collected by Safe Geometry Fit before crop, rotation, brightness normalization, and model inference.\nsamples.csv contains pose labels, holdout membership, timing, dimensions, per-frame brightness affine, and Tobii native openness when reported.\nmetadata.txt contains non-secret runtime settings and a pseudonymous unit identifier; it intentionally omits asset paths and the real EyeChip serial.\n";
    let mut zip = crate::diagnostics::StoredZipWriter::create(path)?;
    zip.add("README.txt", README.as_bytes())?;
    zip.add("metadata.txt", metadata.as_bytes())?;
    zip.add("samples.csv", csv.as_bytes())?;
    for (index, sample) in dataset.samples.iter().enumerate() {
        let left = encode_gray_png(sample.left_size.0, sample.left_size.1, &sample.left)?;
        zip.add(&format!("frames/{index:06}_left.png"), &left)?;
        let right = encode_gray_png(sample.right_size.0, sample.right_size.1, &sample.right)?;
        zip.add(&format!("frames/{index:06}_right.png"), &right)?;
    }
    zip.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_is_separate_from_search_families() {
        assert!(!SampleKind::Neutral.is_holdout());
        assert!(!SampleKind::HalfOpen.is_holdout());
        assert!(SampleKind::HoldoutNeutral.is_holdout());
        assert!(SampleKind::HoldoutHalfOpen.is_holdout());
        assert_eq!(SampleKind::Neutral.family(), SampleFamily::Neutral);
        assert_eq!(SampleKind::HoldoutNeutral.family(), SampleFamily::Neutral);
        assert_eq!(SampleKind::HalfOpen.family(), SampleFamily::HalfOpen);
        assert_eq!(SampleKind::HoldoutHalfOpen.family(), SampleFamily::HalfOpen);
    }

    #[test]
    fn slow_target_is_a_scale_free_close_open_metronome() {
        assert!((slow_target(0.0, 1) - 1.0).abs() < 1e-6);
        assert!(slow_target(0.5, 1) < 0.01);
        assert!(slow_target(0.999, 1) > 0.99);
        assert!(slow_target(1.0 / 6.0, 3) < 0.01);
        assert!(slow_target(1.0 / 3.0 - 0.001, 3) > 0.98);
    }

    #[test]
    fn protocol_contains_a_real_holdout_and_no_squeeze_or_wide() {
        let mut train = Vec::new();
        let mut holdout = Vec::new();
        for phase in PHASES {
            if let Phase::Capture { kind, .. } = phase {
                if kind.is_holdout() {
                    holdout.push(*kind);
                } else {
                    train.push(*kind);
                }
            }
        }
        assert_eq!(train.len(), 9);
        assert_eq!(holdout.len(), 6);
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::Neutral)
                .count(),
            2
        );
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::HalfOpen)
                .count(),
            2
        );
        assert_eq!(
            train
                .iter()
                .filter(|kind| **kind == SampleKind::Closed)
                .count(),
            2
        );
        assert!(holdout.contains(&SampleKind::HoldoutHalfOpen));
        assert!((95.0..=100.0).contains(&total_seconds()));
    }

    #[test]
    fn feedback_zip_contains_stereo_pngs_labels_and_metadata() {
        let path = std::env::temp_dir().join(format!(
            "sranibro_geometry_recording_test_{}.zip",
            std::process::id()
        ));
        let dataset = GeometryDataset {
            samples: vec![GeometrySample {
                kind: SampleKind::HalfOpen,
                expected_open: Some(0.5),
                phase_time_s: 1.25,
                left: vec![0, 64, 128, 255],
                right: vec![255, 128, 64, 0],
                left_size: (2, 2),
                right_size: (2, 2),
                brightness_affine: [[1.0, 0.0], [0.9, 0.1]],
                native_open: [Some(0.52), None],
                phase_index: 3,
            }],
        };
        export_dataset_zip(&path, &dataset, "version=test\nunit_id=unit-test\n").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        for needle in [
            b"samples.csv".as_slice(),
            b"frames/000000_left.png".as_slice(),
            b"frames/000000_right.png".as_slice(),
            b"half_open".as_slice(),
            b"unit-test".as_slice(),
            b"\x89PNG\r\n\x1a\n".as_slice(),
        ] {
            assert!(
                bytes.windows(needle.len()).any(|window| window == needle),
                "missing {:?}",
                String::from_utf8_lossy(needle)
            );
        }
        let _ = std::fs::remove_file(path);
    }
}
