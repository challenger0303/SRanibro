//! Development-only positive control for the XR5 absolute-geometry initializers.
//!
//! It reads the already-authorized local `blink_negative` recordings, reverses the
//! capture writer's right-eye storage mirror, and reports only aggregate geometry.  It
//! never writes camera frames, application configuration, or model state.

use std::ffi::OsStr;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use sranibro_rs::core::types::{DespeckleParams, FlattenParams};
use sranibro_rs::geometry_discovery::{
    estimate_appearance_geometry, estimate_motion_geometry, MotionFrame,
};
use sranibro_rs::ml::preprocess;

struct GrayFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("xr5 geometry discovery failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let wide_data = parse_wide_data()?;
    let sessions_root = wide_data.join("sessions");
    let mut sessions = std::fs::read_dir(&sessions_root)
        .map_err(|error| format!("read {}: {error}", sessions_root.display()))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_ok_and(|kind| kind.is_dir()))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    sessions.sort_by(|left, right| left.file_name().cmp(&right.file_name()));
    if sessions.is_empty() {
        return Err(format!("no sessions below {}", sessions_root.display()));
    }

    let baseline = sranibro_rs::config::default_ml_geometry("pimax_xr5");
    println!("XR5 MOTION GEOMETRY POSITIVE CONTROL");
    println!("sessions={}", sessions.len());
    println!("production_config_changed=false");
    for (session_index, session) in sessions.iter().enumerate() {
        let mut left = load_label_side(session, "blink_negative", 'l')?;
        let mut right = load_label_side(session, "blink_negative", 'r')?;
        // `wide_calib` mirrors right PNGs only for storage. The live geometry capture
        // sees the original camera orientation, so reverse that exact operation here.
        for frame in &mut right {
            mirror_horizontal(
                &mut frame.pixels,
                frame.width as usize,
                frame.height as usize,
            );
        }
        preprocess_frames(&mut left);
        preprocess_frames(&mut right);
        for (cadence, left_refs, right_refs) in [
            ("saved30", frame_refs(&left, 0), frame_refs(&right, 0)),
            (
                "simulated20",
                frame_refs_20hz(&left, 0),
                frame_refs_20hz(&right, 0),
            ),
        ] {
            let estimate =
                estimate_motion_geometry(&left_refs, &right_refs, baseline, [false, true])?;
            println!(
                "session_{session_index:04} {cadence} frames L/R {}/{} confidence {:.1}% eligible={} {}",
                left_refs.len(),
                right_refs.len(),
                estimate.confidence * 100.0,
                estimate.search_eligible,
                estimate.reason
            );
            for (eye, name) in [(0usize, "L"), (1usize, "R")] {
                let value = &estimate.eyes[eye];
                let g = value.geometry;
                println!(
                    "  {name} crop {:.3}/{:.3}/{:.3}/{:.3} rot {:+.1} error {:.4} motion_pairs {} pixels {}",
                    g.crop_left,
                    g.crop_right,
                    g.crop_top,
                    g.crop_bottom,
                    g.rotate_deg,
                    value.fit_error,
                    value.descriptor.motion_pairs,
                    value.descriptor.effective_pixels
                );
                println!(
                    "    descriptor mean {:.4}/{:.4} covariance {:.4}/{:.4}/{:.4}",
                    value.descriptor.mean_px[0],
                    value.descriptor.mean_px[1],
                    value.descriptor.covariance_px2[0][0],
                    value.descriptor.covariance_px2[0][1],
                    value.descriptor.covariance_px2[1][1]
                );
            }
        }

        let mut neutral_left = load_label_side(session, "neutral_center", 'l')?;
        let mut neutral_right = load_label_side(session, "neutral_center", 'r')?;
        for frame in &mut neutral_right {
            mirror_horizontal(
                &mut frame.pixels,
                frame.width as usize,
                frame.height as usize,
            );
        }
        preprocess_frames(&mut neutral_left);
        preprocess_frames(&mut neutral_right);
        let neutral_left_refs = blocked_frame_refs(&neutral_left, 4);
        let neutral_right_refs = blocked_frame_refs(&neutral_right, 4);
        let estimate =
            estimate_appearance_geometry(&neutral_left_refs, &neutral_right_refs, baseline)?;
        println!(
            "session_{session_index:04} neutral appearance confidence {:.1}% eligible={} {}",
            estimate.confidence * 100.0,
            estimate.search_eligible,
            estimate.reason
        );
        for (eye, name) in [(0usize, "L"), (1usize, "R")] {
            let value = &estimate.eyes[eye];
            let descriptor = value.descriptor;
            let g = value.geometry;
            println!(
                "  {name} pupil {:.1}/{:.1} contrast {:.1} axis {:+.1} anisotropy {:.2} spread {:.1}px/{:.1}deg{} crop {:.3}/{:.3}/{:.3}/{:.3} rot {:+.1}",
                descriptor.pupil_center_px[0],
                descriptor.pupil_center_px[1],
                descriptor.pupil_contrast,
                descriptor.aperture_angle_deg,
                descriptor.aperture_anisotropy,
                descriptor.block_center_spread_px,
                descriptor.block_angle_spread_deg,
                if descriptor.stereo_recovered {
                    " stereo-recovered"
                } else {
                    ""
                },
                g.crop_left,
                g.crop_right,
                g.crop_top,
                g.crop_bottom,
                g.rotate_deg,
            );
        }
    }
    Ok(())
}

fn preprocess_frames(frames: &mut [GrayFrame]) {
    let despeckle = DespeckleParams::default();
    let flatten = FlattenParams::default();
    for frame in frames {
        let width = frame.width as usize;
        let height = frame.height as usize;
        let pixels = preprocess::despeckle(&frame.pixels, width, height, &despeckle);
        frame.pixels = preprocess::flatten(&pixels, width, height, &flatten);
    }
}

fn parse_wide_data() -> Result<PathBuf, String> {
    let mut args = std::env::args_os().skip(1);
    let mut wide_data = None;
    while let Some(argument) = args.next() {
        if argument == OsStr::new("--wide-data") {
            let value = args
                .next()
                .ok_or_else(|| "--wide-data requires a path".to_string())?;
            wide_data = Some(PathBuf::from(value));
        } else if argument == OsStr::new("--help") || argument == OsStr::new("-h") {
            println!("xr5-geometry-discovery --wide-data <SRanibro\\wide_data>");
            std::process::exit(0);
        } else {
            return Err(format!(
                "unknown argument {}",
                PathBuf::from(argument).display()
            ));
        }
    }
    wide_data.ok_or_else(|| "missing --wide-data <path>".into())
}

fn load_label_side(session: &Path, label: &str, side: char) -> Result<Vec<GrayFrame>, String> {
    let directory = session.join("images").join(label);
    let needle = format!("_{side}_");
    let mut paths = std::fs::read_dir(&directory)
        .map_err(|error| format!("read {}: {error}", directory.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.extension() == Some(OsStr::new("png"))
                && path
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().contains(&needle))
        })
        .collect::<Vec<_>>();
    paths.sort();
    if paths.len() < 8 {
        return Err(format!(
            "{} contains only {} {side}-eye frames",
            directory.display(),
            paths.len()
        ));
    }
    paths.iter().map(|path| decode_gray_png(path)).collect()
}

fn blocked_frame_refs<'a>(frames: &'a [GrayFrame], blocks: usize) -> Vec<MotionFrame<'a>> {
    let chunk = frames.len().div_ceil(blocks.max(1)).max(1);
    frames
        .iter()
        .enumerate()
        .map(|(index, frame)| MotionFrame {
            group: index / chunk,
            width: frame.width,
            height: frame.height,
            pixels: &frame.pixels,
        })
        .collect()
}

fn decode_gray_png(path: &Path) -> Result<GrayFrame, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let decoder = png::Decoder::new(BufReader::new(file));
    let mut reader = decoder
        .read_info()
        .map_err(|error| format!("png header {}: {error}", path.display()))?;
    let mut output = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut output)
        .map_err(|error| format!("png data {}: {error}", path.display()))?;
    if info.color_type != png::ColorType::Grayscale
        || info.bit_depth != png::BitDepth::Eight
        || info.width == 0
        || info.height == 0
    {
        return Err(format!("unsupported PNG format in {}", path.display()));
    }
    output.truncate(info.buffer_size());
    reader
        .finish()
        .map_err(|error| format!("png tail {}: {error}", path.display()))?;
    Ok(GrayFrame {
        width: info.width,
        height: info.height,
        pixels: output,
    })
}

fn mirror_horizontal(pixels: &mut [u8], width: usize, height: usize) {
    if pixels.len() < width.saturating_mul(height) {
        return;
    }
    for y in 0..height {
        let row = &mut pixels[y * width..(y + 1) * width];
        row.reverse();
    }
}

fn frame_refs<'a>(frames: &'a [GrayFrame], group: usize) -> Vec<MotionFrame<'a>> {
    frames
        .iter()
        .map(|frame| MotionFrame {
            group,
            width: frame.width,
            height: frame.height,
            pixels: &frame.pixels,
        })
        .collect()
}

fn frame_refs_20hz<'a>(frames: &'a [GrayFrame], group: usize) -> Vec<MotionFrame<'a>> {
    // Saved Wide sessions are 30 Hz; keeping two of each three frames reproduces the
    // cadence of the in-memory geometry capture closely enough to catch a rate-sensitive
    // motion descriptor before it reaches the UI.
    frames
        .iter()
        .enumerate()
        .filter(|(index, _)| index % 3 != 2)
        .map(|(_, frame)| MotionFrame {
            group,
            width: frame.width,
            height: frame.height,
            pixels: &frame.pixels,
        })
        .collect()
}
