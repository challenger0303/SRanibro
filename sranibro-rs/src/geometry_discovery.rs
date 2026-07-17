//! XR5 geometry initialization from the motion of the eyelids in raw camera frames.
//!
//! This is deliberately independent of EyeNet.  The old geometry fitter could find
//! several crops with similar open/closed scores because the network output alone does
//! not identify a unique image transform.  Here we first estimate a low-dimensional
//! blink-motion ellipse, then solve for the crop and rotation that place that ellipse in
//! the canonical region occupied by a known-good XR5 reconstruction.  EyeNet and the
//! untouched holdout remain the final safety checks in `geometry_fitrun`.

use crate::core::types::MlGeometry;

const MODEL_SIDE: f64 = 100.0;
const MIRROR_AXIS: f64 = MODEL_SIDE - 1.0;
const MIN_GROUP_FRAMES: usize = 8;
const MIN_MOTION_PAIRS: usize = 20;
const MIN_EFFECTIVE_PIXELS: usize = 64;

// Absolute appearance references measured from two independent XR5 recordings after
// undoing the capture writer's storage-only right mirror. Unlike the blink envelope,
// the pupil centre and aperture axis repeated closely across both sessions. Geometry
// below is expressed in the live, raw 200x200 camera coordinates.
const APPEARANCE_PUPIL_REFERENCE: [[f64; 2]; 2] = [[71.0, 100.0], [127.0, 99.0]];
const APPEARANCE_AXIS_REFERENCE_DEG: [f64; 2] = [12.75, -6.05];
const XR5_PRESET_CENTER_PX: [[f64; 2]; 2] = [[60.0, 100.0], [140.0, 100.0]];
const XR5_PRESET_ROTATION_DEG: [f64; 2] = [-30.0, 30.0];
const XR5_PRESET_WIDTH_PX: f64 = 120.0;
const XR5_PRESET_HEIGHT_PX: f64 = 140.0;
// The inner LED/lens stack is fixed hardware, not user geometry. Keep it out of
// appearance and motion evidence before estimating any per-user transform. The mask is
// deliberately a little stronger than the shipped 40% crop because flare extends past
// the visibly saturated emitters.
const XR5_LED_SAFE_LEFT_MAX_X: usize = 108;
const XR5_LED_SAFE_RIGHT_MIN_X: usize = 91;
const XR5_MIN_INNER_CROP: f64 = 0.35;

/// Non-reconstructable aggregate blink-motion statistics in the 100x100 EyeNet input.
///
/// These are the mean and covariance of the motion envelope produced by the validated
/// XR5 preset over two development sessions.  No camera frame, template image, user
/// identifier, or model output is embedded.  Keeping a per-eye target is intentional:
/// the XR5 optical paths and the final right-eye handedness are not identical.
const CANONICAL_MEAN: [[f64; 2]; 2] = [[64.837_784, 44.911_706], [63.288_729, 46.098_966]];
const CANONICAL_COVARIANCE: [[[f64; 2]; 2]; 2] = [
    [[247.287_843, 8.945_901], [8.945_901, 195.626_530]],
    [[291.995_741, -51.798_716], [-51.798_716, 199.309_880]],
];

#[derive(Clone, Copy)]
pub struct MotionFrame<'a> {
    /// Frames are differenced only inside one uninterrupted capture phase.
    pub group: usize,
    pub width: u32,
    pub height: u32,
    pub pixels: &'a [u8],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionDescriptor {
    pub mean_px: [f32; 2],
    pub covariance_px2: [[f32; 2]; 2],
    pub motion_pairs: usize,
    pub effective_pixels: usize,
    pub threshold: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EyeMotionGeometry {
    pub descriptor: MotionDescriptor,
    pub geometry: MlGeometry,
    pub fit_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MotionGeometryEstimate {
    pub eyes: [EyeMotionGeometry; 2],
    pub geometry: [MlGeometry; 2],
    pub confidence: f32,
    /// A seed can enter the ML search only when the raw motion evidence and binocular
    /// symmetry are plausible.  It is still never applied without the normal holdout.
    pub search_eligible: bool,
    pub reason: String,
}

/// Aggregate, non-reconstructable neutral-eye appearance measurements. The detector
/// operates on per-block median frames; no raw image is retained in this result.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AppearanceDescriptor {
    pub pupil_center_px: [f32; 2],
    pub pupil_contrast: f32,
    pub aperture_angle_deg: f32,
    pub aperture_anisotropy: f32,
    pub block_center_spread_px: f32,
    pub block_angle_spread_deg: f32,
    pub blocks: usize,
    /// One eye had an inconsistent block and was re-detected inside the physical
    /// counterpart neighbourhood predicted by the stable eye.
    pub stereo_recovered: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EyeAppearanceGeometry {
    pub descriptor: AppearanceDescriptor,
    pub geometry: MlGeometry,
}

/// Absolute XR5 crop/rotation hypothesis derived from the pupil and eye-aperture axis
/// in labelled relaxed-neutral frames. It is independent of EyeNet and of the active
/// crop/rotation; the active geometry contributes only its explicit mirror state.
#[derive(Clone, Debug, PartialEq)]
pub struct AppearanceGeometryEstimate {
    pub eyes: [EyeAppearanceGeometry; 2],
    pub geometry: [MlGeometry; 2],
    pub confidence: f32,
    pub search_eligible: bool,
    pub reason: String,
}

#[derive(Clone, Copy)]
struct Descriptor64 {
    mean: [f64; 2],
    covariance: [[f64; 2]; 2],
    motion_pairs: usize,
    effective_pixels: usize,
    threshold: f64,
}

#[derive(Clone, Copy)]
struct SolvedEye {
    geometry: MlGeometry,
    error: f64,
}

/// Estimate an absolute XR5 crop/rotation seed from temporally ordered eyelid-motion
/// frames. `left` and `right` should contain training phases only; holdout frames must
/// remain outside this function.
pub fn estimate_motion_geometry(
    left: &[MotionFrame<'_>],
    right: &[MotionFrame<'_>],
    baseline: [MlGeometry; 2],
    effective_mirrors: [bool; 2],
) -> Result<MotionGeometryEstimate, String> {
    if left.first().map(|frame| (frame.width, frame.height)) != Some((200, 200))
        || right.first().map(|frame| (frame.width, frame.height)) != Some((200, 200))
    {
        return Err("motion geometry requires XR5 stereo 200x200 frames".into());
    }
    let descriptors = [motion_descriptor(left, 0)?, motion_descriptor(right, 1)?];
    let solved = [
        solve_eye(descriptors[0], baseline[0], effective_mirrors[0], 0)?,
        solve_eye(descriptors[1], baseline[1], effective_mirrors[1], 1)?,
    ];
    let geometry = [solved[0].geometry, solved[1].geometry];

    let rotation_symmetry = (geometry[0].rotate_deg + geometry[1].rotate_deg).abs();
    let widths = geometry.map(|g| 1.0 - g.crop_left - g.crop_right);
    let heights = geometry.map(|g| 1.0 - g.crop_top - g.crop_bottom);
    let size_disagreement = (widths[0] - widths[1])
        .abs()
        .max((heights[0] - heights[1]).abs());
    let inner_disagreement = (geometry[0].crop_right - geometry[1].crop_left).abs();
    let max_error = solved[0].error.max(solved[1].error);
    let min_pairs = descriptors[0].motion_pairs.min(descriptors[1].motion_pairs);
    let min_pixels = descriptors[0]
        .effective_pixels
        .min(descriptors[1].effective_pixels);
    let descriptor_plausible = descriptors.iter().all(descriptor_is_plausible);
    let raw_x_symmetry = (descriptors[0].mean[0] + descriptors[1].mean[0] - 199.0).abs();
    let raw_y_disagreement = (descriptors[0].mean[1] - descriptors[1].mean[1]).abs();
    let traces =
        descriptors.map(|descriptor| descriptor.covariance[0][0] + descriptor.covariance[1][1]);
    let trace_ratio = traces[0].min(traces[1]) / traces[0].max(traces[1]).max(1e-6);
    let opposite_tilts = descriptors[0].covariance[0][1] * descriptors[1].covariance[0][1] <= 0.0;

    let error_confidence = (1.0 - max_error / 0.05).clamp(0.0, 1.0);
    let pair_confidence = (min_pairs as f64 / 80.0).clamp(0.0, 1.0);
    let pixel_confidence = (min_pixels as f64 / 300.0).clamp(0.0, 1.0);
    let symmetry_confidence = (1.0 - rotation_symmetry as f64 / 30.0)
        .clamp(0.0, 1.0)
        .min((1.0 - size_disagreement as f64 / 0.20).clamp(0.0, 1.0))
        .min((1.0 - inner_disagreement as f64 / 0.20).clamp(0.0, 1.0));
    let raw_descriptor_confidence = (1.0 - raw_x_symmetry / 60.0)
        .clamp(0.0, 1.0)
        .min((1.0 - raw_y_disagreement / 40.0).clamp(0.0, 1.0))
        .min(trace_ratio)
        * if opposite_tilts { 1.0 } else { 0.5 };
    let confidence = (error_confidence
        * pair_confidence.sqrt()
        * pixel_confidence.sqrt()
        * symmetry_confidence.sqrt()
        * raw_descriptor_confidence.sqrt()) as f32;
    let search_eligible = max_error <= 0.035
        && min_pairs >= MIN_MOTION_PAIRS
        && min_pixels >= MIN_EFFECTIVE_PIXELS
        && rotation_symmetry <= 20.0
        && size_disagreement <= 0.15
        && inner_disagreement <= 0.15
        && descriptor_plausible
        && raw_x_symmetry <= 45.0
        && raw_y_disagreement <= 30.0
        && trace_ratio >= 0.50
        && opposite_tilts
        && confidence >= 0.45;
    let reason = if search_eligible {
        format!(
            "motion seed ready (confidence {:.0}%, fit error L/R {:.4}/{:.4})",
            confidence * 100.0,
            solved[0].error,
            solved[1].error
        )
    } else {
        format!(
            "motion seed is diagnostic only (confidence {:.0}%, fit error L/R {:.4}/{:.4}, rotation symmetry {:.1} deg)",
            confidence * 100.0,
            solved[0].error,
            solved[1].error,
            rotation_symmetry
        )
    };

    Ok(MotionGeometryEstimate {
        eyes: std::array::from_fn(|eye| EyeMotionGeometry {
            descriptor: descriptor_to_public(descriptors[eye]),
            geometry: geometry[eye],
            fit_error: solved[eye].error as f32,
        }),
        geometry,
        confidence,
        search_eligible,
        reason,
    })
}

/// Estimate an absolute XR5 geometry seed from labelled relaxed-neutral frames.
/// Frames must retain phase indices in `group`; at least two independent OPEN blocks
/// are required so a transient gaze or glint cannot silently become geometry.
pub fn estimate_appearance_geometry(
    left: &[MotionFrame<'_>],
    right: &[MotionFrame<'_>],
    baseline: [MlGeometry; 2],
) -> Result<AppearanceGeometryEstimate, String> {
    if left.first().map(|frame| (frame.width, frame.height)) != Some((200, 200))
        || right.first().map(|frame| (frame.width, frame.height)) != Some((200, 200))
    {
        return Err("appearance geometry requires XR5 stereo 200x200 frames".into());
    }
    let mut descriptors = [
        appearance_descriptor(left, 0, None)?,
        appearance_descriptor(right, 1, None)?,
    ];
    let unstable = descriptors.map(|descriptor| descriptor.block_center_spread_px > 10.0);
    if unstable[0] != unstable[1] {
        let unstable_eye = usize::from(unstable[1]);
        let stable_eye = 1 - unstable_eye;
        let stable = descriptors[stable_eye].pupil_center_px;
        let anchor = if unstable_eye == 0 {
            [198.0 - stable[0] as f64, stable[1] as f64 + 1.0]
        } else {
            [198.0 - stable[0] as f64, stable[1] as f64 - 1.0]
        };
        let frames = if unstable_eye == 0 { left } else { right };
        if let Ok(mut recovered) = appearance_descriptor(frames, unstable_eye, Some(anchor)) {
            if recovered.block_center_spread_px < descriptors[unstable_eye].block_center_spread_px
                && recovered.pupil_contrast >= 18.0
                && recovered.aperture_anisotropy >= 2.2
            {
                recovered.stereo_recovered = true;
                descriptors[unstable_eye] = recovered;
            }
        }
    }
    let geometry = std::array::from_fn(|eye| {
        appearance_geometry_from_descriptor(descriptors[eye], baseline[eye], eye)
    });

    let min_contrast = descriptors
        .iter()
        .map(|descriptor| descriptor.pupil_contrast)
        .reduce(f32::min)
        .unwrap_or(0.0);
    let max_center_spread = descriptors
        .iter()
        .map(|descriptor| descriptor.block_center_spread_px)
        .reduce(f32::max)
        .unwrap_or(f32::INFINITY);
    let max_angle_spread = descriptors
        .iter()
        .map(|descriptor| descriptor.block_angle_spread_deg)
        .reduce(f32::max)
        .unwrap_or(f32::INFINITY);
    let min_anisotropy = descriptors
        .iter()
        .map(|descriptor| descriptor.aperture_anisotropy)
        .reduce(f32::min)
        .unwrap_or(0.0);
    let rotation_symmetry = (geometry[0].rotate_deg + geometry[1].rotate_deg).abs();
    let max_reference_shift = descriptors
        .iter()
        .enumerate()
        .map(|(eye, descriptor)| {
            let dx = descriptor.pupil_center_px[0] as f64 - APPEARANCE_PUPIL_REFERENCE[eye][0];
            let dy = descriptor.pupil_center_px[1] as f64 - APPEARANCE_PUPIL_REFERENCE[eye][1];
            dx.hypot(dy) as f32
        })
        .reduce(f32::max)
        .unwrap_or(f32::INFINITY);

    let contrast_confidence = (min_contrast / 35.0).clamp(0.0, 1.0);
    let repeat_confidence = (1.0 - max_center_spread / 12.0)
        .clamp(0.0, 1.0)
        .min((1.0 - max_angle_spread / 10.0).clamp(0.0, 1.0));
    let shape_confidence = ((min_anisotropy - 1.5) / 2.5).clamp(0.0, 1.0);
    let stereo_confidence = (1.0 - rotation_symmetry / 15.0)
        .clamp(0.0, 1.0)
        .min((1.0 - max_reference_shift / 40.0).clamp(0.0, 1.0));
    let confidence = (contrast_confidence
        * repeat_confidence.sqrt()
        * shape_confidence.sqrt()
        * stereo_confidence.sqrt())
    .clamp(0.0, 1.0);
    let search_eligible = descriptors.iter().all(|descriptor| descriptor.blocks >= 2)
        && min_contrast >= 18.0
        && max_center_spread <= 10.0
        && max_angle_spread <= 8.0
        && min_anisotropy >= 2.2
        && rotation_symmetry <= 12.0
        && max_reference_shift <= 30.0
        && confidence >= 0.45;
    let reason = if search_eligible {
        format!(
            "neutral appearance seed ready (confidence {:.0}%, pupil contrast {:.1}, block spread {:.1}px/{:.1}deg)",
            confidence * 100.0,
            min_contrast,
            max_center_spread,
            max_angle_spread
        )
    } else {
        format!(
            "neutral appearance seed is diagnostic only (confidence {:.0}%, pupil contrast {:.1}, block spread {:.1}px/{:.1}deg, rotation symmetry {:.1}deg)",
            confidence * 100.0,
            min_contrast,
            max_center_spread,
            max_angle_spread,
            rotation_symmetry
        )
    };
    Ok(AppearanceGeometryEstimate {
        eyes: std::array::from_fn(|eye| EyeAppearanceGeometry {
            descriptor: descriptors[eye],
            geometry: geometry[eye],
        }),
        geometry,
        confidence,
        search_eligible,
        reason,
    })
}

fn appearance_descriptor(
    frames: &[MotionFrame<'_>],
    eye: usize,
    stereo_anchor: Option<[f64; 2]>,
) -> Result<AppearanceDescriptor, String> {
    if frames.len() < 2 * MIN_GROUP_FRAMES {
        return Err("not enough relaxed-neutral appearance frames".into());
    }
    let width = frames[0].width as usize;
    let height = frames[0].height as usize;
    if width != 200 || height != 200 {
        return Err("neutral appearance frames are not 200x200".into());
    }
    if frames.iter().any(|frame| {
        frame.width as usize != width
            || frame.height as usize != height
            || frame.pixels.len() < width * height
    }) {
        return Err("neutral appearance frames do not share one complete image shape".into());
    }
    let mut groups = std::collections::BTreeMap::<usize, Vec<&MotionFrame<'_>>>::new();
    for frame in frames {
        groups.entry(frame.group).or_default().push(frame);
    }
    let mut blocks = Vec::new();
    for group in groups.values() {
        if group.len() < MIN_GROUP_FRAMES {
            continue;
        }
        let median = median_frame(group, width, height);
        let (center, contrast) = detect_pupil(&median, width, height, eye, stereo_anchor)?;
        let (angle, anisotropy) = aperture_axis(&median, width, height, center, eye)?;
        blocks.push((center, contrast, angle, anisotropy));
    }
    if blocks.len() < 2 {
        return Err("fewer than two independent relaxed-neutral blocks were available".into());
    }
    let center = [
        median_f64(blocks.iter().map(|block| block.0[0]).collect()),
        median_f64(blocks.iter().map(|block| block.0[1]).collect()),
    ];
    let angle = median_f64(blocks.iter().map(|block| block.2).collect());
    let center_spread = blocks
        .iter()
        .map(|block| (block.0[0] - center[0]).hypot(block.0[1] - center[1]))
        .fold(0.0f64, f64::max);
    let angle_spread = blocks
        .iter()
        .map(|block| (block.2 - angle).abs())
        .fold(0.0f64, f64::max);
    Ok(AppearanceDescriptor {
        pupil_center_px: center.map(|value| value as f32),
        pupil_contrast: blocks
            .iter()
            .map(|block| block.1)
            .fold(f64::INFINITY, f64::min) as f32,
        aperture_angle_deg: angle as f32,
        aperture_anisotropy: blocks
            .iter()
            .map(|block| block.3)
            .fold(f64::INFINITY, f64::min) as f32,
        block_center_spread_px: center_spread as f32,
        block_angle_spread_deg: angle_spread as f32,
        blocks: blocks.len(),
        stereo_recovered: false,
    })
}

fn median_frame(frames: &[&MotionFrame<'_>], width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; width * height];
    let mut scratch = Vec::with_capacity(frames.len());
    for (pixel, value) in output.iter_mut().enumerate() {
        scratch.clear();
        scratch.extend(frames.iter().map(|frame| frame.pixels[pixel]));
        scratch.sort_unstable();
        *value = scratch[quantile_index(scratch.len(), 1, 2)];
    }
    output
}

fn detect_pupil(
    image: &[u8],
    width: usize,
    height: usize,
    eye: usize,
    stereo_anchor: Option<[f64; 2]>,
) -> Result<([f64; 2], f64), String> {
    if image.len() < width * height || width != 200 || height != 200 || eye > 1 {
        return Err("pupil detector received an unsupported frame".into());
    }
    let mut values = image[..width * height].to_vec();
    values.sort_unstable();
    let global_median = values[quantile_index(values.len(), 1, 2)] as f64;
    let (x_start, x_end) = if eye == 0 {
        (30usize, 110usize)
    } else {
        (90usize, 170usize)
    };
    let mut best: Option<(f64, [f64; 2], f64)> = None;
    for cy in (65usize..=135).step_by(2) {
        for cx in (x_start..=x_end).step_by(2) {
            if stereo_anchor.is_some_and(|anchor| {
                (cx as f64 - anchor[0]).abs() > 28.0 || (cy as f64 - anchor[1]).abs() > 24.0
            }) {
                continue;
            }
            let mut inner_sum = 0.0f64;
            let mut inner_squared = 0.0f64;
            let mut inner_count = 0usize;
            let mut ring_sum = 0.0f64;
            let mut ring_count = 0usize;
            for dy in -18isize..=18 {
                for dx in -18isize..=18 {
                    let distance2 = dx * dx + dy * dy;
                    let x = (cx as isize + dx) as usize;
                    let y = (cy as isize + dy) as usize;
                    let value = image[y * width + x] as f64;
                    if distance2 <= 49 {
                        inner_sum += value;
                        inner_squared += value * value;
                        inner_count += 1;
                    } else if (144..=324).contains(&distance2) {
                        ring_sum += value;
                        ring_count += 1;
                    }
                }
            }
            if inner_count == 0 || ring_count == 0 {
                continue;
            }
            let inner_mean = inner_sum / inner_count as f64;
            let ring_mean = ring_sum / ring_count as f64;
            let inner_variance =
                (inner_squared / inner_count as f64 - inner_mean * inner_mean).max(0.0);
            let contrast = ring_mean - inner_mean;
            let score = contrast + 0.15 * (global_median - inner_mean)
                - 0.5 * (inner_variance.sqrt() - 25.0).max(0.0);
            let candidate = (score, [cx as f64, cy as f64], contrast);
            if best.as_ref().is_none_or(|current| score > current.0) {
                best = Some(candidate);
            }
        }
    }
    let Some((score, center, contrast)) = best else {
        return Err("no pupil candidate was found in the XR5 eye region".into());
    };
    if !score.is_finite() || !contrast.is_finite() {
        return Err("pupil detector produced a non-finite score".into());
    }
    Ok((center, contrast))
}

fn aperture_axis(
    image: &[u8],
    width: usize,
    height: usize,
    center: [f64; 2],
    eye: usize,
) -> Result<(f64, f64), String> {
    let mut values = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f64 - center[0]) / 48.0;
            let ny = (y as f64 - center[1]) / 30.0;
            if nx * nx + ny * ny <= 1.0 && xr5_evidence_pixel(eye, x, width) {
                values.push(image[y * width + x]);
            }
        }
    }
    if values.len() < MIN_EFFECTIVE_PIXELS {
        return Err("eye aperture region was empty".into());
    }
    values.sort_unstable();
    let threshold = values[quantile_index(values.len(), 48, 100)] as f64;
    let mut mass = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f64 - center[0]) / 48.0;
            let ny = (y as f64 - center[1]) / 30.0;
            if nx * nx + ny * ny > 1.0 || !xr5_evidence_pixel(eye, x, width) {
                continue;
            }
            let weight = (threshold - image[y * width + x] as f64).max(0.0);
            mass += weight;
            sx += weight * x as f64;
            sy += weight * y as f64;
        }
    }
    if mass <= 1e-6 {
        return Err("eye aperture contrast was empty".into());
    }
    let mean = [sx / mass, sy / mass];
    let mut a = 0.0f64;
    let mut b = 0.0f64;
    let mut d = 0.0f64;
    for y in 0..height {
        for x in 0..width {
            let nx = (x as f64 - center[0]) / 48.0;
            let ny = (y as f64 - center[1]) / 30.0;
            if nx * nx + ny * ny > 1.0 || !xr5_evidence_pixel(eye, x, width) {
                continue;
            }
            let weight = (threshold - image[y * width + x] as f64).max(0.0);
            let dx = x as f64 - mean[0];
            let dy = y as f64 - mean[1];
            a += weight * dx * dx;
            b += weight * dx * dy;
            d += weight * dy * dy;
        }
    }
    a /= mass;
    b /= mass;
    d /= mass;
    let discriminant = ((a - d).powi(2) + 4.0 * b.powi(2)).sqrt();
    let low = ((a + d - discriminant) * 0.5).max(1e-6);
    let high = ((a + d + discriminant) * 0.5).max(low);
    let angle = 0.5 * (2.0 * b).atan2(a - d).to_degrees();
    if !angle.is_finite() || !high.is_finite() {
        return Err("eye aperture axis was not finite".into());
    }
    Ok((angle, high / low))
}

fn appearance_geometry_from_descriptor(
    descriptor: AppearanceDescriptor,
    baseline: MlGeometry,
    eye: usize,
) -> MlGeometry {
    let center = [
        XR5_PRESET_CENTER_PX[eye][0] + descriptor.pupil_center_px[0] as f64
            - APPEARANCE_PUPIL_REFERENCE[eye][0],
        XR5_PRESET_CENTER_PX[eye][1] + descriptor.pupil_center_px[1] as f64
            - APPEARANCE_PUPIL_REFERENCE[eye][1],
    ];
    let half_width = XR5_PRESET_WIDTH_PX * 0.5;
    let frame_inner = XR5_MIN_INNER_CROP * 200.0;
    let x_bounds = if eye == 0 {
        [half_width, 200.0 - half_width - frame_inner]
    } else {
        [half_width + frame_inner, 200.0 - half_width]
    };
    let center = [
        center[0].clamp(x_bounds[0], x_bounds[1]),
        center[1].clamp(
            XR5_PRESET_HEIGHT_PX * 0.5,
            200.0 - XR5_PRESET_HEIGHT_PX * 0.5,
        ),
    ];
    let rotate_deg = (XR5_PRESET_ROTATION_DEG[eye]
        - (descriptor.aperture_angle_deg as f64 - APPEARANCE_AXIS_REFERENCE_DEG[eye]))
        .clamp(-45.0, 45.0);
    MlGeometry {
        crop_left: ((center[0] - XR5_PRESET_WIDTH_PX * 0.5) / 200.0) as f32,
        crop_right: ((200.0 - center[0] - XR5_PRESET_WIDTH_PX * 0.5) / 200.0) as f32,
        crop_top: ((center[1] - XR5_PRESET_HEIGHT_PX * 0.5) / 200.0) as f32,
        crop_bottom: ((200.0 - center[1] - XR5_PRESET_HEIGHT_PX * 0.5) / 200.0) as f32,
        scale_x: 1.0,
        scale_y: 1.2,
        rotate_deg: rotate_deg as f32,
        mirror_h: baseline.mirror_h,
    }
}

fn median_f64(mut values: Vec<f64>) -> f64 {
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn descriptor_is_plausible(descriptor: &Descriptor64) -> bool {
    let a = descriptor.covariance[0][0];
    let b = descriptor.covariance[0][1];
    let d = descriptor.covariance[1][1];
    let discriminant = ((a - d).powi(2) + 4.0 * b.powi(2)).sqrt();
    let low = ((a + d - discriminant) * 0.5).max(0.0).sqrt();
    let high = ((a + d + discriminant) * 0.5).max(0.0).sqrt();
    descriptor.mean[0] >= 20.0
        && descriptor.mean[0] <= 180.0
        && descriptor.mean[1] >= 35.0
        && descriptor.mean[1] <= 170.0
        && low >= 7.0
        && high <= 50.0
        && high / low.max(1e-6) <= 4.0
}

fn descriptor_to_public(value: Descriptor64) -> MotionDescriptor {
    MotionDescriptor {
        mean_px: value.mean.map(|v| v as f32),
        covariance_px2: value.covariance.map(|row| row.map(|v| v as f32)),
        motion_pairs: value.motion_pairs,
        effective_pixels: value.effective_pixels,
        threshold: value.threshold as f32,
    }
}

fn motion_descriptor(frames: &[MotionFrame<'_>], eye: usize) -> Result<Descriptor64, String> {
    if frames.len() < MIN_GROUP_FRAMES {
        return Err("not enough eyelid-motion frames".into());
    }
    let width = frames[0].width as usize;
    let height = frames[0].height as usize;
    let pixels = width.saturating_mul(height);
    if width == 0 || height == 0 || pixels == 0 {
        return Err("eyelid-motion frame dimensions are empty".into());
    }
    if frames.iter().any(|frame| {
        frame.width as usize != width
            || frame.height as usize != height
            || frame.pixels.len() < pixels
    }) {
        return Err("eyelid-motion frames do not share one complete image shape".into());
    }

    let mut groups = std::collections::BTreeMap::<usize, Vec<&MotionFrame<'_>>>::new();
    for frame in frames {
        groups.entry(frame.group).or_default().push(frame);
    }
    let mut combined = vec![0u8; pixels];
    let mut intensity_sum = vec![0u64; pixels];
    let mut intensity_frames = 0usize;
    let mut motion_pairs = 0usize;
    let mut scratch = Vec::<u8>::new();
    for group in groups.values() {
        if group.len() < MIN_GROUP_FRAMES {
            continue;
        }
        for frame in group {
            intensity_frames += 1;
            for (sum, pixel) in intensity_sum.iter_mut().zip(frame.pixels) {
                *sum += *pixel as u64;
            }
        }
        let pair_count = group.len() - 1;
        motion_pairs += pair_count;
        for (pixel, combined_pixel) in combined.iter_mut().enumerate() {
            scratch.clear();
            scratch.reserve(pair_count.saturating_sub(scratch.capacity()));
            for pair in group.windows(2) {
                scratch.push(pair[0].pixels[pixel].abs_diff(pair[1].pixels[pixel]));
            }
            scratch.sort_unstable();
            let value = scratch[quantile_index(scratch.len(), 9, 10)];
            *combined_pixel = (*combined_pixel).max(value);
        }
    }
    if motion_pairs < MIN_MOTION_PAIRS || intensity_frames == 0 {
        return Err(format!(
            "only {motion_pairs} adjacent eyelid-motion pairs were available"
        ));
    }

    let mut active = Vec::with_capacity(pixels);
    for pixel in 0..pixels {
        if !xr5_evidence_pixel(eye, pixel % width, width) {
            continue;
        }
        let mean = intensity_sum[pixel] as f64 / intensity_frames as f64;
        if mean <= 245.0 && combined[pixel] > 0 {
            active.push(combined[pixel]);
        }
    }
    if active.len() < MIN_EFFECTIVE_PIXELS {
        return Err("too few non-saturated moving pixels were detected".into());
    }
    active.sort_unstable();
    let threshold = active[quantile_index(active.len(), 4, 5)] as f64;

    let mut mass = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut effective_pixels = 0usize;
    for y in 0..height {
        for x in 0..width {
            if !xr5_evidence_pixel(eye, x, width) {
                continue;
            }
            let index = y * width + x;
            if intensity_sum[index] as f64 / intensity_frames as f64 > 245.0 {
                continue;
            }
            let weight = (combined[index] as f64 - threshold).max(0.0);
            if weight <= 0.0 {
                continue;
            }
            effective_pixels += 1;
            mass += weight;
            sx += weight * x as f64;
            sy += weight * y as f64;
        }
    }
    if effective_pixels < MIN_EFFECTIVE_PIXELS || mass <= 1e-6 {
        return Err("the eyelid-motion envelope was too sparse".into());
    }
    let mean = [sx / mass, sy / mass];
    let mut covariance = [[0.0f64; 2]; 2];
    for y in 0..height {
        for x in 0..width {
            if !xr5_evidence_pixel(eye, x, width) {
                continue;
            }
            let index = y * width + x;
            if intensity_sum[index] as f64 / intensity_frames as f64 > 245.0 {
                continue;
            }
            let weight = (combined[index] as f64 - threshold).max(0.0);
            if weight <= 0.0 {
                continue;
            }
            let dx = x as f64 - mean[0];
            let dy = y as f64 - mean[1];
            covariance[0][0] += weight * dx * dx;
            covariance[0][1] += weight * dx * dy;
            covariance[1][1] += weight * dy * dy;
        }
    }
    covariance[0][0] /= mass;
    covariance[0][1] /= mass;
    covariance[1][0] = covariance[0][1];
    covariance[1][1] /= mass;
    let determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] * covariance[1][0];
    if !determinant.is_finite() || determinant <= 4.0 {
        return Err("the eyelid-motion envelope has no stable two-dimensional extent".into());
    }
    Ok(Descriptor64 {
        mean,
        covariance,
        motion_pairs,
        effective_pixels,
        threshold,
    })
}

fn xr5_evidence_pixel(eye: usize, x: usize, width: usize) -> bool {
    if width != 200 {
        return true;
    }
    match eye {
        0 => x <= XR5_LED_SAFE_LEFT_MAX_X,
        1 => x >= XR5_LED_SAFE_RIGHT_MIN_X,
        _ => false,
    }
}

fn quantile_index(len: usize, numerator: usize, denominator: usize) -> usize {
    debug_assert!(len > 0 && denominator > 0 && numerator <= denominator);
    (len - 1).saturating_mul(numerator) / denominator
}

fn solve_eye(
    descriptor: Descriptor64,
    baseline: MlGeometry,
    mirror: bool,
    eye: usize,
) -> Result<SolvedEye, String> {
    let mut target_mean = CANONICAL_MEAN[eye];
    let mut target_covariance = CANONICAL_COVARIANCE[eye];
    if mirror {
        target_mean[0] = MIRROR_AXIS - target_mean[0];
        target_covariance[0][1] = -target_covariance[0][1];
        target_covariance[1][0] = -target_covariance[1][0];
    }
    let width = baseline_width(baseline);
    let height = baseline_height(baseline);
    let mut best: Option<SolvedEye> = None;
    for degree_step in -22..=22 {
        let degrees = degree_step as f64 * 2.0;
        for width_step in 0..=25 {
            let crop_width = 0.40 + width_step as f64 * 0.02;
            for height_step in 0..=25 {
                let crop_height = 0.40 + height_step as f64 * 0.02;
                consider_solution(
                    &mut best,
                    descriptor,
                    baseline,
                    target_mean,
                    target_covariance,
                    degrees,
                    crop_width,
                    crop_height,
                    width,
                    height,
                );
            }
        }
    }
    let coarse = best.ok_or_else(|| "no bounded motion-geometry solution exists".to_string())?;
    let coarse_width = baseline_width(coarse.geometry) as f64;
    let coarse_height = baseline_height(coarse.geometry) as f64;
    let coarse_degrees = coarse.geometry.rotate_deg as f64;
    best = None;
    for rotation_step in -6..=6 {
        let degrees = coarse_degrees + rotation_step as f64 * 0.5;
        if !(-45.0..=45.0).contains(&degrees) {
            continue;
        }
        for width_step in -8..=8 {
            let crop_width = coarse_width + width_step as f64 * 0.005;
            for height_step in -8..=8 {
                let crop_height = coarse_height + height_step as f64 * 0.005;
                if !(0.35..=0.95).contains(&crop_width) || !(0.35..=0.95).contains(&crop_height) {
                    continue;
                }
                consider_solution(
                    &mut best,
                    descriptor,
                    baseline,
                    target_mean,
                    target_covariance,
                    degrees,
                    crop_width,
                    crop_height,
                    width,
                    height,
                );
            }
        }
    }
    best.ok_or_else(|| "motion-geometry refinement left the frame bounds".into())
}

#[allow(clippy::too_many_arguments)]
fn consider_solution(
    best: &mut Option<SolvedEye>,
    descriptor: Descriptor64,
    baseline: MlGeometry,
    target_mean: [f64; 2],
    target_covariance: [[f64; 2]; 2],
    degrees: f64,
    crop_width: f64,
    crop_height: f64,
    baseline_width: f32,
    baseline_height: f32,
) {
    let frame_width = 200.0;
    let frame_height = 200.0;
    let cw = frame_width * crop_width;
    let ch = frame_height * crop_height;
    let radians = degrees.to_radians();
    let (sin, cos) = radians.sin_cos();
    let sx = finite_scale(baseline.scale_x) as f64;
    let sy = finite_scale(baseline.scale_y) as f64;
    let transform = [
        [MODEL_SIDE * sx * cos / cw, -MODEL_SIDE * sx * sin / ch],
        [MODEL_SIDE * sy * sin / cw, MODEL_SIDE * sy * cos / ch],
    ];
    let predicted = covariance_transform(transform, descriptor.covariance);
    let mut error = 0.0;
    for (row, column) in [(0, 0), (0, 1), (1, 1)] {
        let scale = target_covariance[row][column].abs().max(25.0);
        error += ((predicted[row][column] - target_covariance[row][column]) / scale).powi(2);
    }
    error /= 3.0;
    // Covariance alone has a shallow scale/aspect valley. A very weak prior selects the
    // physically ordinary solution without forcing the active crop centre or rotation.
    error += 0.006 * ((crop_width - crop_height) / 0.50).powi(2);
    error += 0.002
        * (((crop_width - baseline_width as f64) / 0.30).powi(2)
            + ((crop_height - baseline_height as f64) / 0.30).powi(2));

    let Some(inverse) = inverse_2x2(transform) else {
        return;
    };
    let target_delta = [target_mean[0] - 50.0, target_mean[1] - 50.0];
    let source_delta = matrix_vector(inverse, target_delta);
    let center = [
        descriptor.mean[0] - source_delta[0],
        descriptor.mean[1] - source_delta[1],
    ];
    if center[0] < cw * 0.5
        || center[0] > frame_width - cw * 0.5
        || center[1] < ch * 0.5
        || center[1] > frame_height - ch * 0.5
    {
        return;
    }
    let geometry = MlGeometry {
        crop_left: ((center[0] - cw * 0.5) / frame_width) as f32,
        crop_right: ((frame_width - center[0] - cw * 0.5) / frame_width) as f32,
        crop_top: ((center[1] - ch * 0.5) / frame_height) as f32,
        crop_bottom: ((frame_height - center[1] - ch * 0.5) / frame_height) as f32,
        scale_x: baseline.scale_x,
        scale_y: baseline.scale_y,
        rotate_deg: degrees as f32,
        mirror_h: baseline.mirror_h,
    };
    let candidate = SolvedEye { geometry, error };
    if best
        .as_ref()
        .is_none_or(|current| candidate.error < current.error)
    {
        *best = Some(candidate);
    }
}

fn baseline_width(geometry: MlGeometry) -> f32 {
    (1.0 - geometry.crop_left - geometry.crop_right).clamp(0.20, 1.0)
}

fn baseline_height(geometry: MlGeometry) -> f32 {
    (1.0 - geometry.crop_top - geometry.crop_bottom).clamp(0.20, 1.0)
}

fn finite_scale(value: f32) -> f32 {
    if value.is_finite() && value.abs() >= 1e-3 {
        value
    } else {
        1.0
    }
}

fn covariance_transform(a: [[f64; 2]; 2], c: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let ac = [
        [
            a[0][0] * c[0][0] + a[0][1] * c[1][0],
            a[0][0] * c[0][1] + a[0][1] * c[1][1],
        ],
        [
            a[1][0] * c[0][0] + a[1][1] * c[1][0],
            a[1][0] * c[0][1] + a[1][1] * c[1][1],
        ],
    ];
    [
        [
            ac[0][0] * a[0][0] + ac[0][1] * a[0][1],
            ac[0][0] * a[1][0] + ac[0][1] * a[1][1],
        ],
        [
            ac[1][0] * a[0][0] + ac[1][1] * a[0][1],
            ac[1][0] * a[1][0] + ac[1][1] * a[1][1],
        ],
    ]
}

fn inverse_2x2(value: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let determinant = value[0][0] * value[1][1] - value[0][1] * value[1][0];
    (determinant.abs() > 1e-12).then_some([
        [value[1][1] / determinant, -value[0][1] / determinant],
        [-value[1][0] / determinant, value[0][0] / determinant],
    ])
}

fn matrix_vector(matrix: [[f64; 2]; 2], vector: [f64; 2]) -> [f64; 2] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::default_ml_geometry;

    fn source_descriptor_for_geometry(
        geometry: MlGeometry,
        mirror: bool,
        eye: usize,
    ) -> Descriptor64 {
        let mut mean = CANONICAL_MEAN[eye];
        let mut covariance = CANONICAL_COVARIANCE[eye];
        if mirror {
            mean[0] = MIRROR_AXIS - mean[0];
            covariance[0][1] = -covariance[0][1];
            covariance[1][0] = -covariance[1][0];
        }
        let cw = 200.0 * baseline_width(geometry) as f64;
        let ch = 200.0 * baseline_height(geometry) as f64;
        let center = [
            200.0 * (geometry.crop_left as f64 + baseline_width(geometry) as f64 * 0.5),
            200.0 * (geometry.crop_top as f64 + baseline_height(geometry) as f64 * 0.5),
        ];
        let radians = (geometry.rotate_deg as f64).to_radians();
        let (sin, cos) = radians.sin_cos();
        let a = [
            [
                100.0 * geometry.scale_x as f64 * cos / cw,
                -100.0 * geometry.scale_x as f64 * sin / ch,
            ],
            [
                100.0 * geometry.scale_y as f64 * sin / cw,
                100.0 * geometry.scale_y as f64 * cos / ch,
            ],
        ];
        let inverse = inverse_2x2(a).unwrap();
        let source_mean_delta = matrix_vector(inverse, [mean[0] - 50.0, mean[1] - 50.0]);
        let source_mean = [
            center[0] + source_mean_delta[0],
            center[1] + source_mean_delta[1],
        ];
        let source_covariance = covariance_transform(inverse, covariance);
        Descriptor64 {
            mean: source_mean,
            covariance: source_covariance,
            motion_pairs: 100,
            effective_pixels: 500,
            threshold: 10.0,
        }
    }

    #[test]
    fn covariance_solver_recovers_the_validated_xr5_geometry() {
        let baseline = default_ml_geometry("pimax_xr5");
        for eye in 0..2 {
            let mirror = eye == 1;
            let descriptor = source_descriptor_for_geometry(baseline[eye], mirror, eye);
            let solved = solve_eye(descriptor, baseline[eye], mirror, eye).unwrap();
            let actual = solved.geometry;
            let expected = baseline[eye];
            assert!(
                (actual.rotate_deg - expected.rotate_deg).abs() <= 0.51,
                "{actual:?}"
            );
            assert!(
                (actual.crop_left - expected.crop_left).abs() <= 0.011,
                "{actual:?}"
            );
            assert!(
                (actual.crop_right - expected.crop_right).abs() <= 0.011,
                "{actual:?}"
            );
            assert!(
                (actual.crop_top - expected.crop_top).abs() <= 0.011,
                "{actual:?}"
            );
            assert!(
                (actual.crop_bottom - expected.crop_bottom).abs() <= 0.011,
                "{actual:?}"
            );
            assert!(solved.error < 0.01, "error={}", solved.error);
        }
    }

    #[test]
    fn absolute_solver_does_not_inherit_a_wrong_crop_or_rotation() {
        let expected = default_ml_geometry("pimax_xr5");
        for eye in 0..2 {
            let mirror = eye == 1;
            let descriptor = source_descriptor_for_geometry(expected[eye], mirror, eye);
            let mut wrong = expected[eye];
            wrong.crop_left = 0.20;
            wrong.crop_right = 0.20;
            wrong.crop_top = 0.25;
            wrong.crop_bottom = 0.15;
            wrong.rotate_deg = 0.0;
            let solved = solve_eye(descriptor, wrong, mirror, eye).unwrap();
            let actual = solved.geometry;
            assert!(
                (actual.rotate_deg - expected[eye].rotate_deg).abs() <= 1.1,
                "{actual:?}"
            );
            assert!(
                (actual.crop_left - expected[eye].crop_left).abs() <= 0.021,
                "{actual:?}"
            );
            assert!(
                (actual.crop_right - expected[eye].crop_right).abs() <= 0.021,
                "{actual:?}"
            );
            assert!(
                (actual.crop_top - expected[eye].crop_top).abs() <= 0.021,
                "{actual:?}"
            );
            assert!(
                (actual.crop_bottom - expected[eye].crop_bottom).abs() <= 0.021,
                "{actual:?}"
            );
        }
    }

    #[test]
    fn motion_descriptor_tracks_a_moving_region_not_a_static_saturated_patch() {
        let mut owned = Vec::new();
        for frame in 0..40usize {
            let mut image = vec![80u8; 64 * 64];
            for y in 10..52 {
                for x in 4..44 {
                    let dx = x as isize - 23;
                    let dy = y as isize - 30;
                    let shape = (dx * dx + 2 * dy * dy) as f32 * 0.20;
                    let pulse = if frame % 2 == 0 {
                        0.0
                    } else {
                        (130.0 - shape).max(0.0)
                    };
                    image[y * 64 + x] = (80.0 + pulse).min(220.0) as u8;
                }
            }
            for y in 0..8 {
                for x in 52..64 {
                    image[y * 64 + x] = 255;
                }
            }
            owned.push(image);
        }
        let frames: Vec<_> = owned
            .iter()
            .map(|pixels| MotionFrame {
                group: 0,
                width: 64,
                height: 64,
                pixels,
            })
            .collect();
        let descriptor = motion_descriptor(&frames, 0).unwrap();
        assert!(
            (descriptor.mean[0] - 23.0).abs() < 2.0,
            "{:?}",
            descriptor.mean
        );
        assert!(
            (descriptor.mean[1] - 30.0).abs() < 2.0,
            "{:?}",
            descriptor.mean
        );
        assert!(descriptor.effective_pixels >= MIN_EFFECTIVE_PIXELS);
        assert_eq!(descriptor.motion_pairs, 39);
    }

    #[test]
    fn xr5_motion_descriptor_ignores_the_fixed_inner_led_zone() {
        let mut owned = Vec::new();
        for frame in 0..40usize {
            let mut image = vec![80u8; 200 * 200];
            if frame % 2 == 1 {
                for y in 88..113 {
                    for x in 55..86 {
                        let spatial = ((x - 55) * 3 + (y - 88) * 2).min(150) as u8;
                        image[y * 200 + x] = 90u8.saturating_add(spatial);
                    }
                }
                // Much stronger movement in the left camera's fixed inner LED zone.
                for y in 60..141 {
                    for x in 145..191 {
                        image[y * 200 + x] = 240;
                    }
                }
            }
            owned.push(image);
        }
        let frames: Vec<_> = owned
            .iter()
            .map(|pixels| MotionFrame {
                group: 0,
                width: 200,
                height: 200,
                pixels,
            })
            .collect();
        let descriptor = motion_descriptor(&frames, 0).unwrap();
        assert!(descriptor.mean[0] < 100.0, "{:?}", descriptor.mean);
        assert!(
            (55.0..=90.0).contains(&descriptor.mean[0]),
            "{:?}",
            descriptor.mean
        );
    }

    #[test]
    fn insufficient_temporal_evidence_is_rejected() {
        let pixels = vec![0u8; 200 * 200];
        let frames = [MotionFrame {
            group: 0,
            width: 200,
            height: 200,
            pixels: &pixels,
        }];
        assert!(motion_descriptor(&frames, 0).is_err());
    }

    fn synthetic_eye(center: [f64; 2], angle_deg: f64) -> Vec<u8> {
        let mut image = vec![160u8; 200 * 200];
        let radians = angle_deg.to_radians();
        let (sin, cos) = radians.sin_cos();
        for y in 0..200 {
            for x in 0..200 {
                let dx = x as f64 - center[0];
                let dy = y as f64 - center[1];
                let major = dx * cos + dy * sin;
                let minor = -dx * sin + dy * cos;
                if (major / 45.0).powi(2) + (minor / 13.0).powi(2) <= 1.0 {
                    image[y * 200 + x] = 82;
                }
                if dx * dx + dy * dy <= 7.0f64.powi(2) {
                    image[y * 200 + x] = 28;
                }
            }
        }
        image
    }

    #[test]
    fn neutral_detector_finds_pupil_and_aperture_axis() {
        for (eye, center, angle) in [(0usize, [72.0, 103.0], 14.0), (1usize, [128.0, 98.0], -6.0)] {
            let image = synthetic_eye(center, angle);
            let (actual_center, contrast) = detect_pupil(&image, 200, 200, eye, None).unwrap();
            let (actual_angle, anisotropy) =
                aperture_axis(&image, 200, 200, actual_center, eye).unwrap();
            assert!((actual_center[0] - center[0]).abs() <= 2.0);
            assert!((actual_center[1] - center[1]).abs() <= 2.0);
            assert!(contrast >= 40.0, "contrast={contrast}");
            assert!((actual_angle - angle).abs() <= 2.0, "angle={actual_angle}");
            assert!(anisotropy >= 3.0, "anisotropy={anisotropy}");
        }
    }

    #[test]
    fn stable_eye_recovers_other_eye_from_a_reflection_false_circle() {
        let mut left_owned = Vec::new();
        let mut right_owned = Vec::new();
        let mut groups = Vec::new();
        for group in 0..2usize {
            for _ in 0..8 {
                left_owned.push(synthetic_eye([72.0, 103.0], 14.0));
                let mut right = synthetic_eye([128.0, 98.0], -6.0);
                if group == 1 {
                    let false_center = [166.0f64, 72.0f64];
                    for y in 0..200 {
                        for x in 0..200 {
                            let dx = x as f64 - false_center[0];
                            let dy = y as f64 - false_center[1];
                            let distance2 = dx * dx + dy * dy;
                            if distance2 <= 49.0 {
                                right[y * 200 + x] = 8;
                            } else if (144.0..=324.0).contains(&distance2) {
                                right[y * 200 + x] = 225;
                            }
                        }
                    }
                }
                right_owned.push(right);
                groups.push(group);
            }
        }
        let left: Vec<_> = left_owned
            .iter()
            .zip(&groups)
            .map(|(pixels, group)| MotionFrame {
                group: *group,
                width: 200,
                height: 200,
                pixels,
            })
            .collect();
        let right: Vec<_> = right_owned
            .iter()
            .zip(&groups)
            .map(|(pixels, group)| MotionFrame {
                group: *group,
                width: 200,
                height: 200,
                pixels,
            })
            .collect();
        let unanchored = appearance_descriptor(&right, 1, None).unwrap();
        assert!(unanchored.block_center_spread_px > 10.0, "{unanchored:?}");

        let estimate =
            estimate_appearance_geometry(&left, &right, default_ml_geometry("pimax_xr5")).unwrap();
        let recovered = estimate.eyes[1].descriptor;
        assert!(recovered.stereo_recovered, "{recovered:?}");
        assert!((recovered.pupil_center_px[0] - 128.0).abs() <= 2.0);
        assert!((recovered.pupil_center_px[1] - 98.0).abs() <= 2.0);
        assert!(estimate.search_eligible, "{}", estimate.reason);
    }

    #[test]
    fn appearance_seed_is_absolute_not_inherited_from_active_crop() {
        let expected = default_ml_geometry("pimax_xr5");
        for eye in 0..2 {
            let mut wrong = expected[eye];
            wrong.crop_left = 0.20;
            wrong.crop_right = 0.20;
            wrong.crop_top = 0.25;
            wrong.crop_bottom = 0.15;
            wrong.scale_y = 0.85;
            wrong.rotate_deg = 0.0;
            let descriptor = AppearanceDescriptor {
                pupil_center_px: APPEARANCE_PUPIL_REFERENCE[eye].map(|value| value as f32),
                pupil_contrast: 50.0,
                aperture_angle_deg: APPEARANCE_AXIS_REFERENCE_DEG[eye] as f32,
                aperture_anisotropy: 4.0,
                block_center_spread_px: 0.0,
                block_angle_spread_deg: 0.0,
                blocks: 2,
                stereo_recovered: false,
            };
            let actual = appearance_geometry_from_descriptor(descriptor, wrong, eye);
            assert_eq!(actual.crop_left, expected[eye].crop_left);
            assert_eq!(actual.crop_right, expected[eye].crop_right);
            assert_eq!(actual.crop_top, expected[eye].crop_top);
            assert_eq!(actual.crop_bottom, expected[eye].crop_bottom);
            assert_eq!(actual.scale_y, expected[eye].scale_y);
            assert_eq!(actual.rotate_deg, expected[eye].rotate_deg);
            assert_eq!(actual.mirror_h, wrong.mirror_h);
        }
    }
}
