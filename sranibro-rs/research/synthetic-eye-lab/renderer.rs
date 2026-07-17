use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const SIDE: usize = 100;
pub const SUPERSAMPLE: usize = 4;
pub const MIN_PHASE0_FEATURE_PX: f32 = 1.25;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StereoPolicy {
    Duplicate,
    AnatomicalMirror,
    Independent,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PhotometricTransform {
    Identity,
    /// Add a normalized level after the deterministic camera-like u8 render.
    Offset {
        delta: f32,
    },
    /// Scale deviations around this image's own mean. Mean is preserved unless clipping occurs.
    ContrastAroundOwnMean {
        gain: f32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EyeComponents {
    pub skin_region: bool,
    pub sclera: bool,
    pub iris: bool,
    pub pupil: bool,
    pub upper_lid: bool,
    pub lower_lid: bool,
    pub canthi: bool,
}

impl EyeComponents {
    pub const fn none() -> Self {
        Self {
            skin_region: false,
            sclera: false,
            iris: false,
            pupil: false,
            upper_lid: false,
            lower_lid: false,
            canthi: false,
        }
    }

    pub const fn full() -> Self {
        Self {
            skin_region: true,
            sclera: true,
            iris: true,
            pupil: true,
            upper_lid: true,
            lower_lid: true,
            canthi: true,
        }
    }
}

/// Phase-0 stimulus. Positions and sizes are output-pixel units on the 100x100 canvas.
/// `aperture` multiplies `normal_opening_px`: 0=closed, 1=normal, >1=synthetic wide.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SyntheticEyeSpec {
    pub canvas_level: f32,
    pub skin_level: f32,
    pub sclera_level: f32,
    pub iris_level: f32,
    pub pupil_level: f32,
    pub lid_level: f32,
    pub eye_center_px: [f32; 2],
    pub eye_width_px: f32,
    pub normal_opening_px: f32,
    pub aperture: f32,
    pub lid_curve_power: f32,
    pub canthus_tilt_px: f32,
    pub iris_offset_px: [f32; 2],
    pub iris_radius_px: f32,
    pub pupil_radius_px: f32,
    pub lid_thickness_px: f32,
    pub rotation_deg: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub components: EyeComponents,
}

impl Default for SyntheticEyeSpec {
    fn default() -> Self {
        Self {
            canvas_level: 0.36,
            skin_level: 0.46,
            sclera_level: 0.78,
            iris_level: 0.28,
            pupil_level: 0.08,
            lid_level: 0.14,
            eye_center_px: [50.0, 50.0],
            eye_width_px: 72.0,
            normal_opening_px: 23.0,
            aperture: 1.0,
            lid_curve_power: 0.72,
            canthus_tilt_px: 1.5,
            iris_offset_px: [0.0, 0.5],
            iris_radius_px: 11.0,
            pupil_radius_px: 4.8,
            lid_thickness_px: 2.5,
            rotation_deg: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            components: EyeComponents::full(),
        }
    }
}

impl SyntheticEyeSpec {
    pub fn mirrored(&self) -> Self {
        let mut out = self.clone();
        out.eye_center_px[0] = SIDE as f32 - self.eye_center_px[0];
        out.iris_offset_px[0] = -self.iris_offset_px[0];
        out.canthus_tilt_px = -self.canthus_tilt_px;
        out.rotation_deg = -self.rotation_deg;
        out
    }

    pub fn validate(&self) -> bool {
        let levels = [
            self.canvas_level,
            self.skin_level,
            self.sclera_level,
            self.iris_level,
            self.pupil_level,
            self.lid_level,
        ];
        let has_minimum_eye_structure = self.components.sclera
            && self.components.iris
            && self.components.pupil
            && self.components.upper_lid
            && self.components.lower_lid;
        has_minimum_eye_structure
            && levels
                .iter()
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
            && self.eye_center_px.iter().all(|v| v.is_finite())
            && (24.0..=90.0).contains(&self.eye_width_px)
            && (4.0..=40.0).contains(&self.normal_opening_px)
            && self.aperture.is_finite()
            && (0.0..=1.5).contains(&self.aperture)
            && self.lid_curve_power.is_finite()
            && (0.35..=1.5).contains(&self.lid_curve_power)
            && self.iris_radius_px.is_finite()
            && self.pupil_radius_px.is_finite()
            && self.iris_radius_px >= 4.0
            && self.pupil_radius_px >= 2.0
            && self.pupil_radius_px < self.iris_radius_px
            && self.scale_x.is_finite()
            && self.scale_y.is_finite()
            && (0.5..=1.75).contains(&self.scale_x)
            && (0.5..=1.75).contains(&self.scale_y)
            && (!self.components.upper_lid && !self.components.lower_lid
                || self.lid_thickness_px >= MIN_PHASE0_FEATURE_PX)
            && self.eye_center_px[0] - self.eye_width_px * 0.65 >= 0.0
            && self.eye_center_px[0] + self.eye_width_px * 0.65 < SIDE as f32
            && self.eye_center_px[1] - self.eye_width_px * 0.48 >= 0.0
            && self.eye_center_px[1] + self.eye_width_px * 0.48 < SIDE as f32
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageCovariates {
    pub mean: f32,
    pub stddev: f32,
    pub edge_energy: f32,
    pub saturation_fraction: f32,
    pub visible_area_fraction: f32,
    pub measured_aperture_geometry: f32,
    pub measured_aperture_raster: f32,
    /// A declared rendered primitive touches or crosses the canonical frame boundary.
    /// Such a case remains observable, but is excluded from causal summaries because
    /// scale/rotation and implicit cropping would otherwise be confounded.
    pub frame_truncated: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RenderedEye {
    pub pixels: Vec<u8>,
    pub covariates: ImageCovariates,
    pub eye_like: bool,
}

#[derive(Clone, Debug)]
pub struct RenderedStereo {
    pub left_spec: SyntheticEyeSpec,
    pub right_spec: SyntheticEyeSpec,
    pub left: RenderedEye,
    pub right: RenderedEye,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrayMoments {
    pub mean: f64,
    pub stddev: f64,
}

/// Population moments in gray-level units, accumulated from the quantized bytes.
/// This is also the canonical check used by the two-moment solver.
pub fn gray_moments(pixels: &[u8]) -> GrayMoments {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for &pixel in pixels {
        let value = pixel as f64;
        sum += value;
        sum_sq += value * value;
    }
    let n = pixels.len().max(1) as f64;
    let mean = sum / n;
    let variance = (sum_sq / n - mean * mean).max(0.0);
    GrayMoments {
        mean,
        stddev: variance.sqrt(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PhotometricPixel {
    terms: [u64; SUPERSAMPLE * SUPERSAMPLE],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct PhotometricBin {
    pixel: PhotometricPixel,
    multiplicity: usize,
}

/// Exact fast path for searches that vary only `skin_level` and `sclera_level`.
///
/// Every output pixel retains the final material assignment of all 16 canonical
/// supersamples after renderer layer precedence. Search moments use aggregated bins;
/// the ordered coefficients reconstruct the complete byte image for the mandatory
/// independent-canonical parity check.
#[derive(Clone, Debug, PartialEq)]
pub struct PhotometricBasis {
    pixels: Vec<PhotometricPixel>,
    bins: Vec<PhotometricBin>,
}

impl PhotometricBasis {
    pub fn from_spec(spec: &SyntheticEyeSpec) -> Self {
        let mut pixels = Vec::with_capacity(SIDE * SIDE);
        let mut grouped = BTreeMap::<[u64; SUPERSAMPLE * SUPERSAMPLE], usize>::new();
        for py in 0..SIDE {
            for px in 0..SIDE {
                let mut terms = [0u64; SUPERSAMPLE * SUPERSAMPLE];
                let mut term_index = 0usize;
                for sy in 0..SUPERSAMPLE {
                    for sx in 0..SUPERSAMPLE {
                        let x = px as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32;
                        let y = py as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32;
                        let sample = sample_at(spec, x, y);
                        terms[term_index] = encode_term(sample);
                        term_index += 1;
                    }
                }
                let pixel = PhotometricPixel { terms };
                *grouped.entry(terms).or_default() += 1;
                pixels.push(pixel);
            }
        }
        let bins = grouped
            .into_iter()
            .map(|(terms, multiplicity)| PhotometricBin {
                pixel: PhotometricPixel { terms },
                multiplicity,
            })
            .collect();
        Self { pixels, bins }
    }

    pub fn predict_pixels(&self, skin_level: f32, sclera_level: f32) -> Vec<u8> {
        self.pixels
            .iter()
            .map(|pixel| predict_pixel(*pixel, skin_level, sclera_level))
            .collect()
    }

    /// Population moments in gray-level units 0..255, after exact u8 quantization.
    pub fn moments(&self, skin_level: f32, sclera_level: f32) -> GrayMoments {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut count = 0usize;
        for bin in &self.bins {
            let value = predict_pixel(bin.pixel, skin_level, sclera_level) as f64;
            let n = bin.multiplicity as f64;
            sum += value * n;
            sum_sq += value * value * n;
            count += bin.multiplicity;
        }
        let n = count.max(1) as f64;
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);
        GrayMoments {
            mean,
            stddev: variance.sqrt(),
        }
    }
}

fn predict_pixel(pixel: PhotometricPixel, skin_level: f32, sclera_level: f32) -> u8 {
    let mut sum = 0.0f32;
    for term in pixel.terms {
        sum += match term {
            TERM_SKIN => skin_level,
            TERM_SCLERA => sclera_level,
            bits => f32::from_bits(bits as u32),
        };
    }
    quantize_level(sum / (SUPERSAMPLE * SUPERSAMPLE) as f32)
}

const TERM_SKIN: u64 = 1u64 << 32;
const TERM_SCLERA: u64 = 2u64 << 32;

fn encode_term(sample: Sample) -> u64 {
    match sample.material {
        Material::Fixed => sample.level.to_bits() as u64,
        Material::Skin => TERM_SKIN,
        Material::Sclera => TERM_SCLERA,
    }
}

fn quantize_level(level: f32) -> u8 {
    (level.clamp(0.0, 1.0) * 255.0).round() as u8
}

pub fn render_stereo(
    left: &SyntheticEyeSpec,
    right: &SyntheticEyeSpec,
    policy: StereoPolicy,
) -> RenderedStereo {
    let left_spec = left.clone();
    let left_image = render(&left_spec);
    let (right_spec, right_image) = match policy {
        StereoPolicy::Duplicate => (left.clone(), left_image.clone()),
        StereoPolicy::AnatomicalMirror => {
            let mirrored_spec = left.mirrored();
            let mut mirrored_image = left_image.clone();
            for row in mirrored_image.pixels.chunks_exact_mut(SIDE) {
                row.reverse();
            }
            mirrored_image.eye_like = mirrored_spec.validate();
            (mirrored_spec, mirrored_image)
        }
        StereoPolicy::Independent => (right.clone(), render(right)),
    };
    RenderedStereo {
        left_spec,
        right_spec,
        left: left_image,
        right: right_image,
    }
}

pub fn render(spec: &SyntheticEyeSpec) -> RenderedEye {
    let mut pixels = vec![0u8; SIDE * SIDE];
    let mut visible_samples = 0usize;
    for py in 0..SIDE {
        for px in 0..SIDE {
            let mut sum = 0.0f32;
            for sy in 0..SUPERSAMPLE {
                for sx in 0..SUPERSAMPLE {
                    let x = px as f32 + (sx as f32 + 0.5) / SUPERSAMPLE as f32;
                    let y = py as f32 + (sy as f32 + 0.5) / SUPERSAMPLE as f32;
                    let sample = sample_at(spec, x, y);
                    sum += sample.level;
                    visible_samples += sample.inside_opening as usize;
                }
            }
            let level = sum / (SUPERSAMPLE * SUPERSAMPLE) as f32;
            pixels[py * SIDE + px] = quantize_level(level);
        }
    }

    let measured_aperture_geometry =
        (spec.normal_opening_px * spec.aperture * spec.scale_y) / spec.eye_width_px;
    let measured_aperture_raster = raster_center_aperture(spec);
    let mut covariates = pixel_covariates(&pixels);
    covariates.visible_area_fraction =
        visible_samples as f32 / (SIDE * SIDE * SUPERSAMPLE * SUPERSAMPLE) as f32;
    covariates.measured_aperture_geometry = measured_aperture_geometry;
    covariates.measured_aperture_raster = measured_aperture_raster;
    covariates.frame_truncated = frame_truncated(spec);
    RenderedEye {
        pixels,
        covariates,
        eye_like: spec.validate(),
    }
}

pub fn apply_photometric(rendered: &mut RenderedEye, transform: PhotometricTransform) {
    let original = rendered.covariates;
    match transform {
        PhotometricTransform::Identity => return,
        PhotometricTransform::Offset { delta } => {
            for pixel in &mut rendered.pixels {
                let level = *pixel as f32 / 255.0 + delta;
                *pixel = (level.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
        PhotometricTransform::ContrastAroundOwnMean { gain } => {
            let mean = rendered
                .pixels
                .iter()
                .map(|value| *value as f32)
                .sum::<f32>()
                / rendered.pixels.len().max(1) as f32
                / 255.0;
            for pixel in &mut rendered.pixels {
                let level = mean + (*pixel as f32 / 255.0 - mean) * gain;
                *pixel = (level.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    let mut updated = pixel_covariates(&rendered.pixels);
    updated.visible_area_fraction = original.visible_area_fraction;
    updated.measured_aperture_geometry = original.measured_aperture_geometry;
    updated.measured_aperture_raster = original.measured_aperture_raster;
    updated.frame_truncated = original.frame_truncated;
    rendered.covariates = updated;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Material {
    Fixed,
    Skin,
    Sclera,
}

#[derive(Clone, Copy)]
struct Sample {
    level: f32,
    inside_opening: bool,
    material: Material,
}

fn sample_at(spec: &SyntheticEyeSpec, x: f32, y: f32) -> Sample {
    let theta = spec.rotation_deg.to_radians();
    let cos = theta.cos();
    let sin = theta.sin();
    let dx = x - spec.eye_center_px[0];
    let dy = y - spec.eye_center_px[1];
    let local_x = (cos * dx + sin * dy) / spec.scale_x.max(0.001);
    let local_y = (-sin * dx + cos * dy) / spec.scale_y.max(0.001);
    let half_width = spec.eye_width_px * 0.5;
    let skin = (local_x / (spec.eye_width_px * 0.62)).powi(2)
        + (local_y / (spec.eye_width_px * 0.42)).powi(2)
        <= 1.0;
    let (mut level, mut material) = if spec.components.skin_region && skin {
        (spec.skin_level, Material::Skin)
    } else {
        (spec.canvas_level, Material::Fixed)
    };

    let (upper, lower, in_width) = lid_bounds(spec, local_x);
    let inside_opening = in_width && local_y > upper && local_y < lower;
    if spec.components.sclera && inside_opening {
        level = spec.sclera_level;
        material = Material::Sclera;
    }

    let iris_x = local_x - spec.iris_offset_px[0];
    let iris_y = local_y - spec.iris_offset_px[1];
    let iris_d2 = iris_x * iris_x + iris_y * iris_y;
    if spec.components.iris && inside_opening && iris_d2 <= spec.iris_radius_px.powi(2) {
        level = spec.iris_level;
        material = Material::Fixed;
    }
    if spec.components.pupil && inside_opening && iris_d2 <= spec.pupil_radius_px.powi(2) {
        level = spec.pupil_level;
        material = Material::Fixed;
    }

    let half_line = spec.lid_thickness_px * 0.5;
    if in_width
        && ((spec.components.upper_lid && (local_y - upper).abs() <= half_line)
            || (spec.components.lower_lid && (local_y - lower).abs() <= half_line))
    {
        level = spec.lid_level;
        material = Material::Fixed;
    }
    if spec.components.canthi {
        let left_y = -spec.canthus_tilt_px * 0.5;
        let right_y = spec.canthus_tilt_px * 0.5;
        let radius = spec.lid_thickness_px.max(MIN_PHASE0_FEATURE_PX);
        let dl2 = (local_x + half_width).powi(2) + (local_y - left_y).powi(2);
        let dr2 = (local_x - half_width).powi(2) + (local_y - right_y).powi(2);
        if dl2 <= radius * radius || dr2 <= radius * radius {
            level = spec.lid_level;
            material = Material::Fixed;
        }
    }
    Sample {
        level: level.clamp(0.0, 1.0),
        inside_opening,
        material,
    }
}

fn lid_bounds(spec: &SyntheticEyeSpec, local_x: f32) -> (f32, f32, bool) {
    let half_width = spec.eye_width_px * 0.5;
    let u = local_x / half_width.max(0.001);
    if u.abs() > 1.0 {
        return (0.0, 0.0, false);
    }
    let arch = (1.0 - u * u).max(0.0).sqrt().powf(spec.lid_curve_power);
    let half_gap = spec.normal_opening_px * spec.aperture * 0.5;
    let tilt = spec.canthus_tilt_px * u * 0.5;
    (-half_gap * arch + tilt, half_gap * arch + tilt, true)
}

fn raster_center_aperture(spec: &SyntheticEyeSpec) -> f32 {
    let x = spec.eye_center_px[0];
    let mut visible = 0usize;
    for py in 0..SIDE * SUPERSAMPLE {
        let y = (py as f32 + 0.5) / SUPERSAMPLE as f32;
        if sample_at(spec, x, y).inside_opening {
            visible += 1;
        }
    }
    visible as f32 / SUPERSAMPLE as f32 / spec.eye_width_px
}

fn frame_truncated(spec: &SyntheticEyeSpec) -> bool {
    // The skin-region ellipse is the broadest initial primitive. For a rotated ellipse,
    // these are the exact axis-aligned half-extents of its transformed bounding box.
    let radius_x = spec.eye_width_px * 0.62 * spec.scale_x;
    let radius_y = spec.eye_width_px * 0.42 * spec.scale_y;
    let theta = spec.rotation_deg.to_radians();
    let cos = theta.cos();
    let sin = theta.sin();
    let extent_x = ((radius_x * cos).powi(2) + (radius_y * sin).powi(2)).sqrt();
    let extent_y = ((radius_x * sin).powi(2) + (radius_y * cos).powi(2)).sqrt();
    spec.eye_center_px[0] - extent_x < 0.0
        || spec.eye_center_px[0] + extent_x > SIDE as f32
        || spec.eye_center_px[1] - extent_y < 0.0
        || spec.eye_center_px[1] + extent_y > SIDE as f32
}

fn pixel_covariates(pixels: &[u8]) -> ImageCovariates {
    let n = pixels.len().max(1) as f32;
    let mean_u8 = pixels.iter().map(|v| *v as f32).sum::<f32>() / n;
    let variance = pixels
        .iter()
        .map(|v| (*v as f32 - mean_u8).powi(2))
        .sum::<f32>()
        / n;
    let saturated = pixels.iter().filter(|v| **v == 0 || **v == 255).count();
    let mut edges = 0.0f32;
    let mut edge_count = 0usize;
    for y in 0..SIDE {
        for x in 0..SIDE {
            let here = pixels[y * SIDE + x] as f32;
            if x + 1 < SIDE {
                edges += (here - pixels[y * SIDE + x + 1] as f32).abs();
                edge_count += 1;
            }
            if y + 1 < SIDE {
                edges += (here - pixels[(y + 1) * SIDE + x] as f32).abs();
                edge_count += 1;
            }
        }
    }
    ImageCovariates {
        mean: mean_u8 / 255.0,
        stddev: variance.sqrt() / 255.0,
        edge_energy: edges / edge_count.max(1) as f32 / 255.0,
        saturation_fraction: saturated as f32 / n,
        ..ImageCovariates::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_is_deterministic() {
        let spec = SyntheticEyeSpec::default();
        assert_eq!(render(&spec), render(&spec));
    }

    #[test]
    fn requested_aperture_is_strictly_monotone_in_geometry_and_raster_measurements() {
        let mut previous_geom = -1.0;
        let mut previous_raster = -1.0;
        for step in 0..=40 {
            let mut spec = SyntheticEyeSpec::default();
            spec.aperture = 1.3 * step as f32 / 40.0;
            let rendered = render(&spec);
            assert!(rendered.covariates.measured_aperture_geometry > previous_geom);
            assert!(rendered.covariates.measured_aperture_raster > previous_raster);
            previous_geom = rendered.covariates.measured_aperture_geometry;
            previous_raster = rendered.covariates.measured_aperture_raster;
        }
    }

    #[test]
    fn zero_aperture_has_no_visible_opening() {
        let mut spec = SyntheticEyeSpec::default();
        spec.aperture = 0.0;
        let rendered = render(&spec);
        assert_eq!(rendered.covariates.measured_aperture_geometry, 0.0);
        assert_eq!(rendered.covariates.measured_aperture_raster, 0.0);
        assert_eq!(rendered.covariates.visible_area_fraction, 0.0);
    }

    #[test]
    fn anatomical_pair_is_a_pixel_mirror() {
        let mut spec = SyntheticEyeSpec::default();
        spec.iris_offset_px[0] = 4.0;
        let pair = render_stereo(&spec, &spec, StereoPolicy::AnatomicalMirror);
        for y in 0..SIDE {
            for x in 0..SIDE {
                assert_eq!(
                    pair.left.pixels[y * SIDE + x],
                    pair.right.pixels[y * SIDE + (SIDE - 1 - x)]
                );
            }
        }
    }

    #[test]
    fn independent_pair_keeps_specs_independent() {
        let left = SyntheticEyeSpec::default();
        let mut right = left.clone();
        right.aperture = 0.25;
        let pair = render_stereo(&left, &right, StereoPolicy::Independent);
        assert_eq!(pair.left_spec.aperture, 1.0);
        assert_eq!(pair.right_spec.aperture, 0.25);
        assert_ne!(pair.left.pixels, pair.right.pixels);
    }

    #[test]
    fn uniform_and_partial_ladder_patterns_are_not_labelled_eye_like() {
        let mut uniform = SyntheticEyeSpec::default();
        uniform.components = EyeComponents::none();
        assert!(!render(&uniform).eye_like);

        let mut without_lower_lid = SyntheticEyeSpec::default();
        without_lower_lid.components.lower_lid = false;
        assert!(!render(&without_lower_lid).eye_like);
        assert!(render(&SyntheticEyeSpec::default()).eye_like);
    }

    #[test]
    fn contrast_transform_preserves_mean_without_clipping() {
        let mut rendered = render(&SyntheticEyeSpec::default());
        let before = rendered.covariates.mean;
        apply_photometric(
            &mut rendered,
            PhotometricTransform::ContrastAroundOwnMean { gain: 0.8 },
        );
        assert!((rendered.covariates.mean - before).abs() < 1.0 / 255.0);
        assert_eq!(rendered.covariates.saturation_fraction, 0.0);
    }

    #[test]
    fn border_contact_is_recorded_as_a_separate_crop_confound() {
        assert!(
            !render(&SyntheticEyeSpec::default())
                .covariates
                .frame_truncated
        );
        let mut stretched = SyntheticEyeSpec::default();
        stretched.scale_x = 1.2;
        let rendered = render(&stretched);
        assert!(rendered.eye_like, "structure remains eye-like");
        assert!(rendered.covariates.frame_truncated);
    }

    #[test]
    fn photometric_basis_is_byte_identical_to_canonical_renderer() {
        for aperture in [0.2275, 0.4875, 1.0075, 1.3] {
            let mut base = SyntheticEyeSpec::default();
            base.aperture = aperture;
            let basis = PhotometricBasis::from_spec(&base);
            for (skin, sclera) in [(0.30, 0.65), (0.46, 0.78), (0.60, 0.95)] {
                let mut candidate = base.clone();
                candidate.skin_level = skin;
                candidate.sclera_level = sclera;
                assert_eq!(
                    basis.predict_pixels(skin, sclera),
                    render(&candidate).pixels,
                    "aperture={aperture} skin={skin} sclera={sclera}"
                );
            }
        }
    }

    #[test]
    fn photometric_basis_moments_are_post_quantization_population_moments() {
        let mut spec = SyntheticEyeSpec::default();
        spec.aperture = 0.73;
        let basis = PhotometricBasis::from_spec(&spec);
        let pixels = basis.predict_pixels(0.413, 0.887);
        let moments = basis.moments(0.413, 0.887);
        let mean = pixels.iter().map(|value| *value as f64).sum::<f64>() / pixels.len() as f64;
        let variance = pixels
            .iter()
            .map(|value| (*value as f64 - mean).powi(2))
            .sum::<f64>()
            / pixels.len() as f64;
        assert_eq!(moments.mean.to_bits(), mean.to_bits());
        assert!((moments.stddev - variance.sqrt()).abs() <= 1e-12);
    }
}
