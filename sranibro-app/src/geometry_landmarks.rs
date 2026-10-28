//! Candidate-independent XR5 landmark measurements for offline research.
//!
//! This module intentionally accepts only mapped raw 200x200 grayscale frames. It has
//! no access to EyeNet, geometry candidates, pose labels, or native gaze. Until quality
//! has been calibrated on independently annotated real users, every non-abstained result
//! is `DiagnosticOnly` and cannot be used to drive a production warp.

use std::collections::VecDeque;

pub const XR5_SIDE: usize = 200;
pub const LID_KNOTS: usize = 17;

const ROI_Y_MIN: usize = 34;
const ROI_Y_MAX: usize = 166;
const LEFT_X_MIN: usize = 8;
const LEFT_X_MAX: usize = 108;
const RIGHT_X_MIN: usize = 91;
const RIGHT_X_MAX: usize = 191;
const DARK_THRESHOLDS: [u8; 4] = [12, 18, 24, 30];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeSide {
    Left,
    Right,
}

impl EyeSide {
    fn x_bounds(self) -> (usize, usize) {
        match self {
            Self::Left => (LEFT_X_MIN, LEFT_X_MAX),
            Self::Right => (RIGHT_X_MIN, RIGHT_X_MAX),
        }
    }
}

#[derive(Clone, Copy)]
pub struct GrayFrame<'a> {
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl GrayFrame<'_> {
    fn valid(self) -> bool {
        self.width == XR5_SIDE
            && self.height == XR5_SIDE
            && self.pixels.len() >= self.width.saturating_mul(self.height)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PupilEllipse {
    pub center_px: [f32; 2],
    /// Semi-axis radii, largest first.
    pub radii_px: [f32; 2],
    pub angle_deg: f32,
    pub boundary_coverage: f32,
    pub contrast: f32,
    pub threshold_consensus: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LidProfile {
    pub x_px: [f32; LID_KNOTS],
    pub upper_y_px: [f32; LID_KNOTS],
    pub lower_y_px: [f32; LID_KNOTS],
    pub aperture_px: [f32; LID_KNOTS],
    pub valid: [bool; LID_KNOTS],
    pub median_aperture_px: f32,
    pub path_strength: f32,
}

impl Default for LidProfile {
    fn default() -> Self {
        Self {
            x_px: [0.0; LID_KNOTS],
            upper_y_px: [0.0; LID_KNOTS],
            lower_y_px: [0.0; LID_KNOTS],
            aperture_px: [0.0; LID_KNOTS],
            valid: [false; LID_KNOTS],
            median_aperture_px: 0.0,
            path_strength: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhotometricFeatures {
    pub p10: f32,
    pub median: f32,
    pub p90: f32,
    pub mad: f32,
    /// Least-squares gray-level change over normalized sensor X/Y in [-1, 1].
    pub horizontal_gradient: f32,
    pub vertical_gradient: f32,
    pub vertical_curvature: f32,
    pub saturation_fraction: f32,
    pub glint_fraction: f32,
    pub fixed_ir_flare_fraction: f32,
    pub pupil_minus_global: Option<f32>,
    /// Row-major 3x3 medians/P90/saturation over the safe eye evidence region. These
    /// preserve local illumination evidence without selecting or applying a correction.
    pub local_median_3x3: [f32; 9],
    pub local_p90_3x3: [f32; 9],
    pub local_saturation_3x3: [f32; 9],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbstentionReason {
    UnsupportedFrame,
    LowContrast,
    PupilAbsent,
    PupilAmbiguous,
    BorderConnectedCandidate,
    ExcessiveGlint,
    LidEvidenceInsufficient,
    CurveCrossing,
    SessionAnchorInvalid,
    /// A measurement exists, but real-image confidence has not been calibrated.
    UncalibratedRealImage,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LandmarkDecision<T> {
    Accepted {
        value: T,
        confidence_lower_bound: f32,
    },
    DiagnosticOnly {
        value: T,
        quality: f32,
        reason: AbstentionReason,
    },
    Abstained {
        quality: f32,
        reason: AbstentionReason,
    },
}

impl<T> LandmarkDecision<T> {
    /// Returns any measured value, including explicitly uncalibrated diagnostics.
    /// Production decisions must use [`Self::calibrated_value`] instead.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::Accepted { value, .. } | Self::DiagnosticOnly { value, .. } => Some(value),
            Self::Abstained { .. } => None,
        }
    }

    pub fn calibrated_value(&self) -> Option<&T> {
        match self {
            Self::Accepted { value, .. } => Some(value),
            Self::DiagnosticOnly { .. } | Self::Abstained { .. } => None,
        }
    }

    pub fn quality(&self) -> f32 {
        match self {
            Self::Accepted {
                confidence_lower_bound,
                ..
            } => *confidence_lower_bound,
            Self::DiagnosticOnly { quality, .. } | Self::Abstained { quality, .. } => *quality,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EyeLandmarkReport {
    pub pupil: LandmarkDecision<PupilEllipse>,
    pub lids: LandmarkDecision<LidProfile>,
    pub photometric: PhotometricFeatures,
    /// Conservative diagnostic proxy; this is not an absolute occlusion probability.
    pub occlusion_fraction: f32,
}

#[derive(Clone, Copy, Debug)]
struct PupilCandidate {
    ellipse: PupilEllipse,
    threshold_index: usize,
    score: f32,
}

#[derive(Clone, Debug)]
struct PupilCluster {
    members: Vec<PupilCandidate>,
    best: PupilCandidate,
}

pub fn detect_eye_landmarks(frame: GrayFrame<'_>, side: EyeSide) -> EyeLandmarkReport {
    if !frame.valid() {
        return EyeLandmarkReport {
            pupil: LandmarkDecision::Abstained {
                quality: 0.0,
                reason: AbstentionReason::UnsupportedFrame,
            },
            lids: LandmarkDecision::Abstained {
                quality: 0.0,
                reason: AbstentionReason::UnsupportedFrame,
            },
            photometric: PhotometricFeatures::default(),
            occlusion_fraction: 1.0,
        };
    }

    let integral = integral_image(frame);
    let (glint_mask, glint_fraction) = glint_mask(frame, side, &integral);
    let pupil_result = detect_pupil(frame, side, &integral, &glint_mask);
    let pupil_mask = pupil_mask(
        frame,
        pupil_result.as_ref().ok().map(|(ellipse, _)| *ellipse),
    );
    let lids = detect_lids(frame, side, &glint_mask, &pupil_mask);
    let pupil = match pupil_result {
        Ok((ellipse, quality)) => LandmarkDecision::DiagnosticOnly {
            value: ellipse,
            quality,
            reason: AbstentionReason::UncalibratedRealImage,
        },
        Err((quality, reason)) => LandmarkDecision::Abstained { quality, reason },
    };
    let lids = match lids {
        Ok((profile, quality)) => LandmarkDecision::DiagnosticOnly {
            value: profile,
            quality,
            reason: AbstentionReason::UncalibratedRealImage,
        },
        Err((quality, reason)) => LandmarkDecision::Abstained { quality, reason },
    };
    // Pupil-local photometry is invalid until pupil identity has been calibrated on
    // annotated real XR5 frames.  Global/local-plane features remain candidate-independent.
    let photometric = photometric_features(frame, side, glint_fraction, pupil.calibrated_value());
    let evidence_quality = pupil.quality().min(lids.quality());
    let occlusion_fraction = (1.0 - evidence_quality)
        .max(glint_fraction * 4.0)
        .clamp(0.0, 1.0);
    EyeLandmarkReport {
        pupil,
        lids,
        photometric,
        occlusion_fraction,
    }
}

fn evidence_pixel(side: EyeSide, x: usize, y: usize) -> bool {
    let (x_min, x_max) = side.x_bounds();
    (x_min..=x_max).contains(&x) && (ROI_Y_MIN..=ROI_Y_MAX).contains(&y)
}

fn integral_image(frame: GrayFrame<'_>) -> Vec<u64> {
    let stride = frame.width + 1;
    let mut integral = vec![0u64; stride * (frame.height + 1)];
    for y in 0..frame.height {
        let mut row_sum = 0u64;
        for x in 0..frame.width {
            row_sum += frame.pixels[y * frame.width + x] as u64;
            integral[(y + 1) * stride + x + 1] = integral[y * stride + x + 1] + row_sum;
        }
    }
    integral
}

fn window_mean(
    integral: &[u64],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> f32 {
    let x0 = x.saturating_sub(radius);
    let y0 = y.saturating_sub(radius);
    let x1 = (x + radius + 1).min(width);
    let y1 = (y + radius + 1).min(height);
    let stride = width + 1;
    let sum = integral[y1 * stride + x1] + integral[y0 * stride + x0]
        - integral[y0 * stride + x1]
        - integral[y1 * stride + x0];
    sum as f32 / ((x1 - x0) * (y1 - y0)).max(1) as f32
}

fn glint_mask(frame: GrayFrame<'_>, side: EyeSide, integral: &[u64]) -> (Vec<bool>, f32) {
    let mut histogram = [0usize; 256];
    let mut evidence = 0usize;
    for y in ROI_Y_MIN..=ROI_Y_MAX {
        for x in side.x_bounds().0..=side.x_bounds().1 {
            histogram[frame.pixels[y * frame.width + x] as usize] += 1;
            evidence += 1;
        }
    }
    let p98 = histogram_quantile(&histogram, evidence, 98, 100).max(210) as f32;
    let mut raw = vec![false; frame.width * frame.height];
    for y in ROI_Y_MIN..=ROI_Y_MAX {
        for x in side.x_bounds().0..=side.x_bounds().1 {
            let value = frame.pixels[y * frame.width + x] as f32;
            let local = window_mean(integral, frame.width, frame.height, x, y, 4);
            if value >= p98 && value - local >= 25.0 {
                raw[y * frame.width + x] = true;
            }
        }
    }
    // Two-pixel dilation keeps a bright hole from fragmenting a dark pupil component.
    let mut dilated = raw.clone();
    for y in 1..frame.height - 1 {
        for x in 1..frame.width - 1 {
            if !raw[y * frame.width + x] {
                continue;
            }
            for dy in -2isize..=2 {
                for dx in -2isize..=2 {
                    let xx = (x as isize + dx).clamp(0, frame.width as isize - 1) as usize;
                    let yy = (y as isize + dy).clamp(0, frame.height as isize - 1) as usize;
                    dilated[yy * frame.width + xx] = true;
                }
            }
        }
    }
    let count = raw
        .iter()
        .enumerate()
        .filter(|(index, marked)| {
            **marked && evidence_pixel(side, index % frame.width, index / frame.width)
        })
        .count();
    (dilated, count as f32 / evidence.max(1) as f32)
}

fn detect_pupil(
    frame: GrayFrame<'_>,
    side: EyeSide,
    integral: &[u64],
    glint: &[bool],
) -> Result<(PupilEllipse, f32), (f32, AbstentionReason)> {
    let mut likelihood = vec![0u8; frame.width * frame.height];
    for y in ROI_Y_MIN..=ROI_Y_MAX {
        for x in side.x_bounds().0..=side.x_bounds().1 {
            if glint[y * frame.width + x] {
                continue;
            }
            let local = window_mean(integral, frame.width, frame.height, x, y, 11);
            likelihood[y * frame.width + x] =
                (local - frame.pixels[y * frame.width + x] as f32).clamp(0.0, 255.0) as u8;
        }
    }

    let mut candidates = Vec::new();
    let mut saw_border = false;
    for (threshold_index, threshold) in DARK_THRESHOLDS.into_iter().enumerate() {
        let (mut level, border) = pupil_candidates_at_threshold(
            frame,
            side,
            &likelihood,
            glint,
            threshold,
            threshold_index,
        );
        saw_border |= border;
        candidates.append(&mut level);
    }
    if candidates.is_empty() {
        return Err((
            0.0,
            if saw_border {
                AbstentionReason::BorderConnectedCandidate
            } else {
                AbstentionReason::PupilAbsent
            },
        ));
    }

    candidates.sort_by(|left, right| right.score.total_cmp(&left.score));
    let mut clusters: Vec<PupilCluster> = Vec::new();
    for candidate in candidates {
        if let Some(cluster) = clusters.iter_mut().find(|cluster| {
            let a = cluster.best.ellipse;
            let b = candidate.ellipse;
            let distance = (a.center_px[0] - b.center_px[0]).hypot(a.center_px[1] - b.center_px[1]);
            let scale = (a.radii_px[0] / b.radii_px[0].max(1.0))
                .max(b.radii_px[0] / a.radii_px[0].max(1.0));
            distance <= 5.0 && scale <= 1.8
        }) {
            if candidate.score > cluster.best.score {
                cluster.best = candidate;
            }
            cluster.members.push(candidate);
        } else {
            clusters.push(PupilCluster {
                members: vec![candidate],
                best: candidate,
            });
        }
    }
    for cluster in &mut clusters {
        cluster
            .members
            .sort_by_key(|candidate| candidate.threshold_index);
        cluster
            .members
            .dedup_by_key(|candidate| candidate.threshold_index);
        cluster.best.ellipse.threshold_consensus = cluster.members.len();
    }
    clusters.retain(|cluster| cluster.members.len() >= 2);
    clusters.sort_by(|left, right| {
        let l = left.best.score + 5.0 * left.members.len() as f32;
        let r = right.best.score + 5.0 * right.members.len() as f32;
        r.total_cmp(&l)
    });
    let Some(best) = clusters.first() else {
        return Err((0.15, AbstentionReason::PupilAbsent));
    };
    if clusters.get(1).is_some_and(|second| {
        let margin = (best.best.score - second.best.score).abs();
        let distance = (best.best.ellipse.center_px[0] - second.best.ellipse.center_px[0])
            .hypot(best.best.ellipse.center_px[1] - second.best.ellipse.center_px[1]);
        margin < 6.0 && distance > 6.0
    }) {
        return Err((0.25, AbstentionReason::PupilAmbiguous));
    }
    let ellipse = best.best.ellipse;
    let consensus = (best.members.len() as f32 / DARK_THRESHOLDS.len() as f32).clamp(0.0, 1.0);
    let quality = (ellipse.contrast / 50.0)
        .clamp(0.0, 1.0)
        .min((ellipse.boundary_coverage / 0.7).clamp(0.0, 1.0))
        .min(consensus);
    if ellipse.contrast < 12.0 || ellipse.boundary_coverage < 0.25 {
        return Err((quality, AbstentionReason::LowContrast));
    }
    Ok((ellipse, quality))
}

fn pupil_candidates_at_threshold(
    frame: GrayFrame<'_>,
    side: EyeSide,
    likelihood: &[u8],
    glint: &[bool],
    threshold: u8,
    threshold_index: usize,
) -> (Vec<PupilCandidate>, bool) {
    let (x_min, x_max) = side.x_bounds();
    let mut visited = vec![false; frame.width * frame.height];
    let mut candidates = Vec::new();
    let mut saw_border = false;
    for seed_y in ROI_Y_MIN..=ROI_Y_MAX {
        for seed_x in x_min..=x_max {
            let seed = seed_y * frame.width + seed_x;
            if visited[seed] || glint[seed] || likelihood[seed] < threshold {
                continue;
            }
            let mut queue = VecDeque::from([seed]);
            visited[seed] = true;
            let mut pixels = Vec::new();
            let mut touches_border = false;
            while let Some(index) = queue.pop_front() {
                let x = index % frame.width;
                let y = index / frame.width;
                pixels.push(index);
                touches_border |= x == x_min || x == x_max || y == ROI_Y_MIN || y == ROI_Y_MAX;
                for (nx, ny) in [
                    (x.wrapping_sub(1), y),
                    (x + 1, y),
                    (x, y.wrapping_sub(1)),
                    (x, y + 1),
                ] {
                    if nx < x_min || nx > x_max || ny < ROI_Y_MIN || ny > ROI_Y_MAX {
                        continue;
                    }
                    let neighbour = ny * frame.width + nx;
                    if !visited[neighbour]
                        && !glint[neighbour]
                        && likelihood[neighbour] >= threshold
                    {
                        visited[neighbour] = true;
                        queue.push_back(neighbour);
                    }
                }
            }
            if touches_border {
                saw_border = true;
                continue;
            }
            if !(24..=900).contains(&pixels.len()) {
                continue;
            }
            if let Some(candidate) =
                candidate_from_component(frame, likelihood, glint, &pixels, threshold_index)
            {
                candidates.push(candidate);
            }
        }
    }
    (candidates, saw_border)
}

fn candidate_from_component(
    frame: GrayFrame<'_>,
    likelihood: &[u8],
    glint: &[bool],
    pixels: &[usize],
    threshold_index: usize,
) -> Option<PupilCandidate> {
    let mut weight = 0.0f64;
    let (mut sx, mut sy) = (0.0f64, 0.0f64);
    for &index in pixels {
        let w = likelihood[index].max(1) as f64;
        weight += w;
        sx += w * (index % frame.width) as f64;
        sy += w * (index / frame.width) as f64;
    }
    let center = [sx / weight, sy / weight];
    let (mut xx, mut xy, mut yy) = (0.0f64, 0.0f64, 0.0f64);
    for &index in pixels {
        let w = likelihood[index].max(1) as f64;
        let dx = (index % frame.width) as f64 - center[0];
        let dy = (index / frame.width) as f64 - center[1];
        xx += w * dx * dx;
        xy += w * dx * dy;
        yy += w * dy * dy;
    }
    xx /= weight;
    xy /= weight;
    yy /= weight;
    let discriminant = ((xx - yy).powi(2) + 4.0 * xy.powi(2)).sqrt();
    let high = ((xx + yy + discriminant) * 0.5).max(0.0);
    let low = ((xx + yy - discriminant) * 0.5).max(0.0);
    let radii = [2.0 * high.sqrt(), 2.0 * low.sqrt()];
    if !(4.0..=32.0).contains(&radii[0])
        || !(2.0..=20.0).contains(&radii[1])
        || radii[1] / radii[0].max(1.0) < 0.18
    {
        return None;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let (contrast, coverage) = ellipse_evidence(frame, glint, center, radii, angle);
    if !contrast.is_finite() || !coverage.is_finite() {
        return None;
    }
    let ellipse_area = std::f64::consts::PI * radii[0] * radii[1];
    let fill = pixels.len() as f64 / ellipse_area.max(1.0);
    if !(0.12..=2.5).contains(&fill) {
        return None;
    }
    let score =
        contrast as f32 + 18.0 * coverage as f32 - 4.0 * ((fill - 0.75).abs().min(1.0) as f32);
    Some(PupilCandidate {
        ellipse: PupilEllipse {
            center_px: [center[0] as f32, center[1] as f32],
            radii_px: [radii[0] as f32, radii[1] as f32],
            angle_deg: angle.to_degrees() as f32,
            boundary_coverage: coverage as f32,
            contrast: contrast as f32,
            threshold_consensus: 1,
        },
        threshold_index,
        score,
    })
}

fn ellipse_evidence(
    frame: GrayFrame<'_>,
    glint: &[bool],
    center: [f64; 2],
    radii: [f64; 2],
    angle: f64,
) -> (f64, f64) {
    let (sin, cos) = angle.sin_cos();
    let mut inner = Vec::new();
    let mut outer = Vec::new();
    let x0 = (center[0] - 1.7 * radii[0]).floor().max(0.0) as usize;
    let x1 = (center[0] + 1.7 * radii[0])
        .ceil()
        .min((frame.width - 1) as f64) as usize;
    let y0 = (center[1] - 1.7 * radii[0]).floor().max(0.0) as usize;
    let y1 = (center[1] + 1.7 * radii[0])
        .ceil()
        .min((frame.height - 1) as f64) as usize;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let index = y * frame.width + x;
            if glint[index] {
                continue;
            }
            let dx = x as f64 - center[0];
            let dy = y as f64 - center[1];
            let major = dx * cos + dy * sin;
            let minor = -dx * sin + dy * cos;
            let norm = (major / radii[0].max(1.0)).powi(2) + (minor / radii[1].max(1.0)).powi(2);
            if norm <= 0.65 {
                inner.push(frame.pixels[index]);
            } else if (1.25..=2.50).contains(&norm) {
                outer.push(frame.pixels[index]);
            }
        }
    }
    if inner.len() < 10 || outer.len() < 20 {
        return (0.0, 0.0);
    }
    inner.sort_unstable();
    outer.sort_unstable();
    let contrast = median_u8(&outer) as f64 - median_u8(&inner) as f64;
    let mut covered = 0usize;
    const RAYS: usize = 24;
    for ray in 0..RAYS {
        let theta = std::f64::consts::TAU * ray as f64 / RAYS as f64;
        let local = [radii[0] * theta.cos(), radii[1] * theta.sin()];
        let sample = |scale: f64| -> u8 {
            let x = center[0] + scale * (local[0] * cos - local[1] * sin);
            let y = center[1] + scale * (local[0] * sin + local[1] * cos);
            let xx = x.round().clamp(0.0, (frame.width - 1) as f64) as usize;
            let yy = y.round().clamp(0.0, (frame.height - 1) as f64) as usize;
            frame.pixels[yy * frame.width + xx]
        };
        if sample(1.30) as i16 - sample(0.70) as i16 >= 5 {
            covered += 1;
        }
    }
    (contrast, covered as f64 / RAYS as f64)
}

fn detect_lids(
    frame: GrayFrame<'_>,
    side: EyeSide,
    glint: &[bool],
    pupil_mask: &[bool],
) -> Result<(LidProfile, f32), (f32, AbstentionReason)> {
    let (x_min, x_max) = side.x_bounds();
    let margin = 7usize;
    let start = x_min + margin;
    let end = x_max.saturating_sub(margin);
    let x_positions: [f32; LID_KNOTS] = std::array::from_fn(|knot| {
        start as f32 + (end - start) as f32 * knot as f32 / (LID_KNOTS - 1) as f32
    });
    let upper = best_lid_path(frame, &x_positions, 38, 112, false, glint, pupil_mask);
    let lower = best_lid_path(frame, &x_positions, 78, 162, true, glint, pupil_mask);
    let (Some((upper_y, upper_strength)), Some((lower_y, lower_strength))) = (upper, lower) else {
        return Err((0.0, AbstentionReason::LidEvidenceInsufficient));
    };
    let mut apertures = [0.0f32; LID_KNOTS];
    for knot in 0..LID_KNOTS {
        apertures[knot] = lower_y[knot] - upper_y[knot];
    }
    if apertures.iter().any(|aperture| *aperture <= 1.0) {
        return Err((0.1, AbstentionReason::CurveCrossing));
    }
    let mut sorted = apertures;
    sorted.sort_by(f32::total_cmp);
    let median_aperture = sorted[LID_KNOTS / 2];
    let strength = upper_strength.min(lower_strength);
    let roughness = path_roughness(&upper_y).max(path_roughness(&lower_y));
    let quality = (strength / 35.0)
        .clamp(0.0, 1.0)
        .min((1.0 - roughness / 18.0).clamp(0.0, 1.0));
    if strength < 4.0 || roughness > 25.0 {
        return Err((quality, AbstentionReason::LidEvidenceInsufficient));
    }
    Ok((
        LidProfile {
            x_px: x_positions,
            upper_y_px: upper_y,
            lower_y_px: lower_y,
            aperture_px: apertures,
            valid: [true; LID_KNOTS],
            median_aperture_px: median_aperture,
            path_strength: strength,
        },
        quality,
    ))
}

fn best_lid_path(
    frame: GrayFrame<'_>,
    x_positions: &[f32; LID_KNOTS],
    y_min: usize,
    y_max: usize,
    positive_gradient: bool,
    glint: &[bool],
    pupil_mask: &[bool],
) -> Option<([f32; LID_KNOTS], f32)> {
    let height = y_max - y_min + 1;
    let mut scores = vec![0.0f32; LID_KNOTS * height];
    for (knot, &x) in x_positions.iter().enumerate() {
        let x = x.round() as usize;
        for y in y_min..=y_max {
            let mut top = 0.0;
            let mut bottom = 0.0;
            let mut count = 0.0;
            for dx in -2isize..=2 {
                let xx = (x as isize + dx).clamp(0, frame.width as isize - 1) as usize;
                let top_index = y.saturating_sub(2) * frame.width + xx;
                let bottom_index = (y + 2).min(frame.height - 1) * frame.width + xx;
                if glint[top_index]
                    || glint[bottom_index]
                    || pupil_mask[top_index]
                    || pupil_mask[bottom_index]
                {
                    continue;
                }
                top += frame.pixels[top_index] as f32;
                bottom += frame.pixels[bottom_index] as f32;
                count += 1.0;
            }
            if count > 0.0 {
                let gradient = (bottom - top) / count;
                scores[knot * height + y - y_min] = if positive_gradient {
                    gradient
                } else {
                    -gradient
                };
            } else {
                scores[knot * height + y - y_min] = -255.0;
            }
        }
    }
    let mut previous = scores[..height].to_vec();
    let mut back = vec![0usize; LID_KNOTS * height];
    for knot in 1..LID_KNOTS {
        let mut next = vec![f32::NEG_INFINITY; height];
        for y in 0..height {
            let low = y.saturating_sub(10);
            let high = (y + 10).min(height - 1);
            for prior in low..=high {
                let transition = 0.65 * y.abs_diff(prior) as f32;
                let value = previous[prior] + scores[knot * height + y] - transition;
                if value > next[y] {
                    next[y] = value;
                    back[knot * height + y] = prior;
                }
            }
        }
        previous = next;
    }
    let mut y = previous
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))?
        .0;
    let total = previous[y];
    let mut path = [0.0f32; LID_KNOTS];
    for knot in (0..LID_KNOTS).rev() {
        path[knot] = (y + y_min) as f32;
        if knot > 0 {
            y = back[knot * height + y];
        }
    }
    Some((path, total / LID_KNOTS as f32))
}

fn path_roughness(path: &[f32; LID_KNOTS]) -> f32 {
    path.windows(3)
        .map(|triple| (triple[2] - 2.0 * triple[1] + triple[0]).abs())
        .sum::<f32>()
        / (LID_KNOTS - 2) as f32
}

fn photometric_features(
    frame: GrayFrame<'_>,
    side: EyeSide,
    glint_fraction: f32,
    pupil: Option<&PupilEllipse>,
) -> PhotometricFeatures {
    let (x_min, x_max) = side.x_bounds();
    let mut histogram = [0usize; 256];
    let mut local_histogram = [[0usize; 256]; 9];
    let mut local_count = [0usize; 9];
    let mut local_saturated = [0usize; 9];
    let mut values = Vec::with_capacity((x_max - x_min + 1) * (ROI_Y_MAX - ROI_Y_MIN + 1));
    let mut sum = 0.0f64;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sy2 = 0.0f64;
    let mut xx = 0.0f64;
    let mut yy = 0.0f64;
    let mut y2y2 = 0.0f64;
    let mut count = 0usize;
    let mut saturated = 0usize;
    for y in ROI_Y_MIN..=ROI_Y_MAX {
        let yn = 2.0 * (y - ROI_Y_MIN) as f64 / (ROI_Y_MAX - ROI_Y_MIN) as f64 - 1.0;
        let y2 = yn * yn - 1.0 / 3.0;
        for x in x_min..=x_max {
            let xn = 2.0 * (x - x_min) as f64 / (x_max - x_min) as f64 - 1.0;
            let value = frame.pixels[y * frame.width + x];
            histogram[value as usize] += 1;
            let tile_x = ((x - x_min) * 3 / (x_max - x_min + 1)).min(2);
            let tile_y = ((y - ROI_Y_MIN) * 3 / (ROI_Y_MAX - ROI_Y_MIN + 1)).min(2);
            let tile = tile_y * 3 + tile_x;
            local_histogram[tile][value as usize] += 1;
            local_count[tile] += 1;
            local_saturated[tile] += usize::from(value >= 250);
            values.push(value);
            let v = value as f64;
            sum += v;
            sx += xn * v;
            sy += yn * v;
            sy2 += y2 * v;
            xx += xn * xn;
            yy += yn * yn;
            y2y2 += y2 * y2;
            saturated += usize::from(value >= 250);
            count += 1;
        }
    }
    let p10 = histogram_quantile(&histogram, count, 10, 100) as f32;
    let median = histogram_quantile(&histogram, count, 1, 2) as f32;
    let p90 = histogram_quantile(&histogram, count, 90, 100) as f32;
    let mut deviations = [0usize; 256];
    for value in values {
        deviations[(value as i16 - median as i16).unsigned_abs() as usize] += 1;
    }
    let mad = histogram_quantile(&deviations, count, 1, 2) as f32;
    let global = sum / count.max(1) as f64;
    let pupil_minus_global = pupil.map(|ellipse| {
        let radius = ellipse.radii_px[1].max(2.0) as isize;
        let cx = ellipse.center_px[0].round() as isize;
        let cy = ellipse.center_px[1].round() as isize;
        let mut local = 0.0f32;
        let mut local_count = 0usize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }
                let x = (cx + dx).clamp(0, frame.width as isize - 1) as usize;
                let y = (cy + dy).clamp(0, frame.height as isize - 1) as usize;
                local += frame.pixels[y * frame.width + x] as f32;
                local_count += 1;
            }
        }
        local / local_count.max(1) as f32 - global as f32
    });
    let mut fixed_count = 0usize;
    let mut fixed_bright = 0usize;
    for y in ROI_Y_MIN..=ROI_Y_MAX {
        for x in 0..frame.width {
            let fixed_ir_region = match side {
                EyeSide::Left => x > LEFT_X_MAX,
                EyeSide::Right => x < RIGHT_X_MIN,
            };
            if !fixed_ir_region {
                continue;
            }
            fixed_count += 1;
            fixed_bright += usize::from(frame.pixels[y * frame.width + x] >= 245);
        }
    }
    let local_median_3x3 = std::array::from_fn(|tile| {
        histogram_quantile(&local_histogram[tile], local_count[tile], 1, 2) as f32
    });
    let local_p90_3x3 = std::array::from_fn(|tile| {
        histogram_quantile(&local_histogram[tile], local_count[tile], 9, 10) as f32
    });
    let local_saturation_3x3 =
        std::array::from_fn(|tile| local_saturated[tile] as f32 / local_count[tile].max(1) as f32);
    PhotometricFeatures {
        p10,
        median,
        p90,
        mad,
        horizontal_gradient: (sx / xx.max(1.0)) as f32,
        vertical_gradient: (sy / yy.max(1.0)) as f32,
        vertical_curvature: (sy2 / y2y2.max(1.0)) as f32,
        saturation_fraction: saturated as f32 / count.max(1) as f32,
        glint_fraction,
        fixed_ir_flare_fraction: fixed_bright as f32 / fixed_count.max(1) as f32,
        pupil_minus_global,
        local_median_3x3,
        local_p90_3x3,
        local_saturation_3x3,
    }
}

fn histogram_quantile(
    histogram: &[usize; 256],
    count: usize,
    numerator: usize,
    denominator: usize,
) -> u8 {
    if count == 0 || denominator == 0 {
        return 0;
    }
    let target = (count - 1).saturating_mul(numerator) / denominator;
    let mut cumulative = 0usize;
    for (value, amount) in histogram.iter().enumerate() {
        cumulative += amount;
        if cumulative > target {
            return value as u8;
        }
    }
    255
}

fn median_u8(values: &[u8]) -> u8 {
    values[values.len() / 2]
}

fn pupil_mask(frame: GrayFrame<'_>, pupil: Option<PupilEllipse>) -> Vec<bool> {
    let mut mask = vec![false; frame.width * frame.height];
    let Some(pupil) = pupil else { return mask };
    let radius = (pupil.radii_px[0] * 1.35).ceil() as isize;
    let cx = pupil.center_px[0] as isize;
    let cy = pupil.center_px[1] as isize;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy > radius * radius {
                continue;
            }
            let x = cx + dx;
            let y = cy + dy;
            if x >= 0 && y >= 0 && x < frame.width as isize && y < frame.height as isize {
                mask[y as usize * frame.width + x as usize] = true;
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_eye(side: EyeSide, center: [f32; 2], open: f32) -> Vec<u8> {
        let mut image = vec![172u8; XR5_SIDE * XR5_SIDE];
        let angle = match side {
            EyeSide::Left => 12.0f32.to_radians(),
            EyeSide::Right => -7.0f32.to_radians(),
        };
        let (sin, cos) = angle.sin_cos();
        for y in 0..XR5_SIDE {
            for x in 0..XR5_SIDE {
                let dx = x as f32 - center[0];
                let dy = y as f32 - center[1];
                let major = dx * cos + dy * sin;
                let minor = -dx * sin + dy * cos;
                let half_height = 26.0 * open.max(0.02);
                if (major / 47.0).powi(2) + (minor / half_height).powi(2) <= 1.0 {
                    image[y * XR5_SIDE + x] = 112;
                }
                if open > 0.12 && (major / 9.0).powi(2) + (minor / 6.5).powi(2) <= 1.0 {
                    image[y * XR5_SIDE + x] = 35;
                }
            }
        }
        if open > 0.12 {
            for y in center[1] as usize - 2..=center[1] as usize + 1 {
                for x in center[0] as usize - 1..=center[0] as usize + 2 {
                    image[y * XR5_SIDE + x] = 255;
                }
            }
        }
        // Reproduce the saturated inner XR5 hardware zone. It must never become evidence.
        match side {
            EyeSide::Left => {
                for y in 45..155 {
                    for x in 150..XR5_SIDE {
                        image[y * XR5_SIDE + x] = if x % 9 < 3 { 255 } else { 8 };
                    }
                }
            }
            EyeSide::Right => {
                for y in 45..155 {
                    for x in 0..50 {
                        image[y * XR5_SIDE + x] = if x % 9 < 3 { 255 } else { 8 };
                    }
                }
            }
        }
        image
    }

    fn pupil_value(report: &EyeLandmarkReport) -> PupilEllipse {
        *report.pupil.value().expect("diagnostic pupil")
    }

    #[test]
    fn clean_synthetic_pupil_is_recovered_but_never_auto_accepted() {
        for (side, center) in [
            (EyeSide::Left, [76.0, 96.0]),
            (EyeSide::Right, [124.0, 96.0]),
        ] {
            let image = synthetic_eye(side, center, 1.0);
            let report = detect_eye_landmarks(
                GrayFrame {
                    pixels: &image,
                    width: XR5_SIDE,
                    height: XR5_SIDE,
                },
                side,
            );
            let pupil = pupil_value(&report);
            assert!((pupil.center_px[0] - center[0]).abs() <= 2.0, "{pupil:?}");
            assert!((pupil.center_px[1] - center[1]).abs() <= 2.0, "{pupil:?}");
            assert!(matches!(
                report.pupil,
                LandmarkDecision::DiagnosticOnly { .. }
            ));
            assert!(report.pupil.calibrated_value().is_none());
            assert!(report.photometric.pupil_minus_global.is_none());
        }
    }

    #[test]
    fn fixed_ir_flare_uses_only_the_explicit_inner_hardware_region() {
        let mut outer_bright = vec![100u8; XR5_SIDE * XR5_SIDE];
        for y in ROI_Y_MIN..=ROI_Y_MAX {
            for x in 0..LEFT_X_MIN {
                outer_bright[y * XR5_SIDE + x] = 255;
            }
        }
        let outer = detect_eye_landmarks(
            GrayFrame {
                pixels: &outer_bright,
                width: XR5_SIDE,
                height: XR5_SIDE,
            },
            EyeSide::Left,
        );
        assert_eq!(outer.photometric.fixed_ir_flare_fraction, 0.0);

        let mut inner_bright = vec![100u8; XR5_SIDE * XR5_SIDE];
        for y in ROI_Y_MIN..=ROI_Y_MAX {
            for x in LEFT_X_MAX + 1..XR5_SIDE {
                inner_bright[y * XR5_SIDE + x] = 255;
            }
        }
        let inner = detect_eye_landmarks(
            GrayFrame {
                pixels: &inner_bright,
                width: XR5_SIDE,
                height: XR5_SIDE,
            },
            EyeSide::Left,
        );
        assert!(inner.photometric.fixed_ir_flare_fraction > 0.99);
    }

    #[test]
    fn local_photometry_retains_a_three_by_three_brightness_pattern() {
        let mut image = vec![0u8; XR5_SIDE * XR5_SIDE];
        for y in ROI_Y_MIN..=ROI_Y_MAX {
            for x in LEFT_X_MIN..=LEFT_X_MAX {
                let tile_x = ((x - LEFT_X_MIN) * 3 / (LEFT_X_MAX - LEFT_X_MIN + 1)).min(2);
                let tile_y = ((y - ROI_Y_MIN) * 3 / (ROI_Y_MAX - ROI_Y_MIN + 1)).min(2);
                image[y * XR5_SIDE + x] = 20 + (tile_y * 3 + tile_x) as u8 * 20;
            }
        }
        let report = detect_eye_landmarks(
            GrayFrame {
                pixels: &image,
                width: XR5_SIDE,
                height: XR5_SIDE,
            },
            EyeSide::Left,
        );
        assert_eq!(
            report.photometric.local_median_3x3,
            [20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
        );
    }

    #[test]
    fn closed_eye_and_boundary_connected_dark_strip_do_not_become_pupils() {
        let mut image = synthetic_eye(EyeSide::Left, [76.0, 96.0], 0.0);
        for y in 60..145 {
            for x in 98..=LEFT_X_MAX {
                image[y * XR5_SIDE + x] = 5;
            }
        }
        let report = detect_eye_landmarks(
            GrayFrame {
                pixels: &image,
                width: XR5_SIDE,
                height: XR5_SIDE,
            },
            EyeSide::Left,
        );
        assert!(matches!(report.pupil, LandmarkDecision::Abstained { .. }));
    }

    #[test]
    fn relative_sensor_motion_is_monotonic_and_brightness_gradient_does_not_move_center() {
        let mut centers = Vec::new();
        for x in [66.0, 76.0, 86.0] {
            let mut image = synthetic_eye(EyeSide::Left, [x, 96.0], 1.0);
            for y in 0..XR5_SIDE {
                for px in 0..XR5_SIDE {
                    let bias = (18.0 * y as f32 / (XR5_SIDE - 1) as f32) as u8;
                    image[y * XR5_SIDE + px] = image[y * XR5_SIDE + px].saturating_add(bias);
                }
            }
            let report = detect_eye_landmarks(
                GrayFrame {
                    pixels: &image,
                    width: XR5_SIDE,
                    height: XR5_SIDE,
                },
                EyeSide::Left,
            );
            centers.push(pupil_value(&report).center_px[0]);
        }
        assert!(
            centers[0] < centers[1] && centers[1] < centers[2],
            "{centers:?}"
        );
        assert!((centers[1] - 76.0).abs() <= 2.0, "{centers:?}");
    }

    #[test]
    fn unsupported_shape_abstains_without_panicking() {
        let image = vec![0u8; 10 * 10];
        let report = detect_eye_landmarks(
            GrayFrame {
                pixels: &image,
                width: 10,
                height: 10,
            },
            EyeSide::Left,
        );
        assert!(matches!(
            report.pupil,
            LandmarkDecision::Abstained {
                reason: AbstentionReason::UnsupportedFrame,
                ..
            }
        ));
    }
}
