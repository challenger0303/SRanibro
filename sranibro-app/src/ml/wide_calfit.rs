//! Pure-Rust per-wearer EyeWide head fitting.
//!
//! The convolution backbone comes from a generic XR5 `wide.bin`; only the small
//! regression head is fitted locally. Completed captures are split by SESSION: all
//! earlier sessions train, while the newest session is held out for validation. This
//! avoids the overly optimistic random-adjacent-frame split common in video datasets.

use std::path::{Path, PathBuf};

use crate::ml::brow_fit::{fit_head_with_validation, FitHeadReport};
use crate::ml::preprocess::{wide_input, WIDE_SIDE};
use crate::ml::wide_net::WideNet;

// Consecutive 60 Hz frames are highly redundant. Bounding the pure-Rust head fit keeps
// it interactive while an even index spread retains every capture phase.
const MAX_TRAIN_FRAMES_PER_SESSION: usize = 600;
const MAX_VALIDATION_FRAMES: usize = 800;

pub struct Dataset {
    pub features: Vec<Vec<f32>>,
    pub labels: Vec<Vec<f32>>,
    pub rows: usize,
    pub skipped: usize,
}

pub struct FitOutput {
    pub bytes: Vec<u8>,
    pub sessions: usize,
    pub train_frames: usize,
    pub val_frames: usize,
    pub train_mse: f64,
    pub val_mse: f64,
}

/// Sorted completed capture sessions. A directory is complete only when labels.csv
/// exists; aborted/partial sessions are intentionally ignored.
pub fn completed_sessions(wide_data_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let root = wide_data_dir.join("sessions");
    let entries =
        std::fs::read_dir(&root).map_err(|e| format!("Wide sessions '{}': {e}", root.display()))?;
    let mut sessions: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir() && path.join("labels.csv").is_file())
        .collect();
    sessions.sort();
    Ok(sessions)
}

pub fn load_session(session: &Path, backbone: &mut WideNet) -> Result<Dataset, String> {
    let labels_path = session.join("labels.csv");
    let text = std::fs::read_to_string(&labels_path)
        .map_err(|e| format!("labels.csv '{}': {e}", labels_path.display()))?;
    let mut lines = text.lines();
    let header = lines.next().unwrap_or("");
    if header.trim() != "filename,wide,phase,side" {
        return Err(format!(
            "unexpected Wide header '{header}' (want 'filename,wide,phase,side')"
        ));
    }

    let images_dir = session.join("images");
    let mut features = Vec::new();
    let mut labels = Vec::new();
    let mut rows = 0usize;
    let mut skipped = 0usize;
    let mut input = [0.0f32; WIDE_SIDE * WIDE_SIDE];
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let mut fields = line.split(',');
        let filename = fields.next().unwrap_or("").trim();
        let label = fields
            .next()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .ok_or_else(|| format!("row '{line}': invalid Wide label"))?;
        let image = images_dir.join(filename);
        match decode_gray8(&image) {
            Some((w, h, pixels)) => {
                // Right images are canonicalized (mirrored) while being captured.
                wide_input(&pixels, w, h, false, &mut input);
                features.push(backbone.features(&input));
                labels.push(vec![label]);
            }
            None => skipped += 1,
        }
    }
    if features.is_empty() {
        return Err(format!(
            "no usable Wide frames ({rows} rows, {skipped} skipped) under {}",
            images_dir.display()
        ));
    }
    Ok(Dataset {
        features,
        labels,
        rows,
        skipped,
    })
}

/// Fit a one-output head, holding out the newest complete session. At least two
/// separately captured sessions are required so the reported validation error measures
/// a real reseat/time boundary instead of neighboring video frames.
pub fn fit_to_bytes(
    wide_data_dir: &Path,
    backbone_bin: &Path,
    seed: u64,
) -> Result<FitOutput, String> {
    let sessions = completed_sessions(wide_data_dir)?;
    if sessions.len() < 2 {
        return Err(format!(
            "need at least 2 completed Wide sessions; found {} (capture again after reseating the headset)",
            sessions.len()
        ));
    }
    let mut backbone = WideNet::load(backbone_bin)?;
    let mut train_features = Vec::new();
    let mut train_labels = Vec::new();
    for session in &sessions[..sessions.len() - 1] {
        let ds = downsample(
            load_session(session, &mut backbone)?,
            MAX_TRAIN_FRAMES_PER_SESSION,
        );
        train_features.extend(ds.features);
        train_labels.extend(ds.labels);
    }
    let val = downsample(
        load_session(sessions.last().unwrap(), &mut backbone)?,
        MAX_VALIDATION_FRAMES,
    );
    let FitHeadReport {
        weights,
        train_mse,
        val_mse,
    } = fit_head_with_validation(
        &train_features,
        &train_labels,
        &val.features,
        &val.labels,
        1,
        seed,
    );
    if !train_mse.is_finite() || !val_mse.is_finite() {
        return Err("Wide fit produced a non-finite validation error".into());
    }
    let bytes = backbone.to_bytes_with_head(&weights)?;
    // Validate the final task header and tensor sizes before the caller writes anything.
    WideNet::from_bytes(&bytes).map_err(|e| format!("fitted Wide model validation: {e}"))?;
    Ok(FitOutput {
        bytes,
        sessions: sessions.len(),
        train_frames: train_features.len(),
        val_frames: val.features.len(),
        train_mse,
        val_mse,
    })
}

fn downsample(dataset: Dataset, max: usize) -> Dataset {
    let len = dataset.features.len();
    if len <= max || max == 0 {
        return dataset;
    }
    let indices: Vec<usize> = (0..max).map(|i| i * len / max).collect();
    Dataset {
        features: indices
            .iter()
            .map(|&index| dataset.features[index].clone())
            .collect(),
        labels: indices
            .iter()
            .map(|&index| dataset.labels[index].clone())
            .collect(),
        rows: dataset.rows,
        skipped: dataset.skipped,
    }
}

fn decode_gray8(path: &Path) -> Option<(usize, usize, Vec<u8>)> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = png::Decoder::new(file).read_info().ok()?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Grayscale || info.bit_depth != png::BitDepth::Eight {
        return None;
    }
    let (w, h) = (info.width as usize, info.height as usize);
    buf.truncate(info.buffer_size());
    if buf.len() < w * h {
        return None;
    }
    buf.truncate(w * h);
    Some((w, h, buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("sranibro_wide_calfit_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(path.join("sessions")).unwrap();
        path
    }

    #[test]
    fn only_completed_sessions_are_sorted() {
        let root = temp("sessions");
        for name in ["session-003", "session-001", "session-002"] {
            std::fs::create_dir_all(root.join("sessions").join(name)).unwrap();
        }
        std::fs::write(root.join("sessions/session-003/labels.csv"), "x").unwrap();
        std::fs::write(root.join("sessions/session-001/labels.csv"), "x").unwrap();
        let got = completed_sessions(&root).unwrap();
        let names: Vec<_> = got
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, ["session-001", "session-003"]);
        let _ = std::fs::remove_dir_all(root);
    }
}
