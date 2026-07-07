//! Camera-frame -> ML input preprocessing.
//!
//! Matches `eye_prediction.py` device='vr4': `cv2.resize(img, (100,100))` then
//! `/255`, with the resulting image duplicated into 2 channels (the SRanipal net
//! takes a stereo pair that is the same image twice). For an exact 2x downscale,
//! cv2 INTER_LINEAR reduces to a 2x2 box average, which is what we compute.

use crate::core::types::{DespeckleParams, FlattenParams, MlGeometry};

/// Source camera frame size (per eye).
pub const SRC: usize = 200;
/// ML input size (per channel/side).
pub const DST: usize = 100;

/// 2x2 box-average downscale of a 200x200 grayscale image into a `DST*DST` f32
/// slice in [0,1]. cv2 INTER_LINEAR at exact 2x reduces to this box average.
fn downscale_into(img: &[u8], dst: &mut [f32]) {
    debug_assert!(img.len() >= SRC * SRC, "frame must be at least 200x200");
    debug_assert!(dst.len() >= DST * DST);
    for y in 0..DST {
        let sy = 2 * y;
        for x in 0..DST {
            let sx = 2 * x;
            let a = img[sy * SRC + sx] as u32;
            let b = img[sy * SRC + sx + 1] as u32;
            let c = img[(sy + 1) * SRC + sx] as u32;
            let d = img[(sy + 1) * SRC + sx + 1] as u32;
            // 2x2 box average (rounded), normalized to [0,1].
            let avg = (a + b + c + d + 2) / 4;
            dst[y * DST + x] = avg as f32 / 255.0;
        }
    }
}

/// 200x200 grayscale (`SRC*SRC` bytes) -> [2][100][100] f32 in [0,1], layout
/// [c][h][w] (channel 0 == channel 1). Returns `2*DST*DST` floats.
pub fn vr4_to_input(img: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; 2 * DST * DST];
    downscale_into(img, &mut out[..DST * DST]); // channel 0
    out.copy_within(..DST * DST, DST * DST); // channel 1 (duplicate)
    out
}

/// Resize a `w`x`h` grayscale image to `DST`x`DST` f32 [0,1] (area average). The
/// exact 2x case (200x200, VR4/StarVR) delegates to `downscale_into` so that path
/// is byte-identical to before — no ML regression. Other resolutions (e.g. Varjo)
/// use a general area average, so the model always sees its native 100x100 input.
fn resize_into(img: &[u8], w: usize, h: usize, dst: &mut [f32]) {
    if w == 2 * DST && h == 2 * DST {
        downscale_into(img, dst);
        return;
    }
    if w == 0 || h == 0 {
        for v in dst.iter_mut().take(DST * DST) {
            *v = 0.0;
        }
        return;
    }
    for dy in 0..DST {
        let sy0 = dy * h / DST;
        let sy1 = (((dy + 1) * h + DST - 1) / DST).min(h).max(sy0 + 1);
        for dx in 0..DST {
            let sx0 = dx * w / DST;
            let sx1 = (((dx + 1) * w + DST - 1) / DST).min(w).max(sx0 + 1);
            // u64 accumulators: a large source region (many px * 255) would overflow
            // u32; u64 is free here and bullet-proof.
            let mut sum = 0u64;
            let mut cnt = 0u64;
            for sy in sy0..sy1 {
                let row = sy * w;
                for sx in sx0..sx1 {
                    sum += img[row + sx] as u64;
                    cnt += 1;
                }
            }
            dst[dy * DST + dx] = if cnt > 0 { sum as f32 / cnt as f32 / 255.0 } else { 0.0 };
        }
    }
}

/// `resize_into`, then horizontally mirror each row if `flip_h` (clean L-R flip of
/// the 100x100 result; the area average is orientation-preserving).
fn resize_into_flip(img: &[u8], w: usize, h: usize, dst: &mut [f32], flip_h: bool) {
    resize_into(img, w, h, dst);
    if flip_h {
        for y in 0..DST {
            dst[y * DST..(y + 1) * DST].reverse();
        }
    }
}

/// Stereo ML input: left eye in channel 0, right eye in channel 1 — the layout the
/// real SRanipal the SRanipal eyelid model feeds (ONE forward pass over both eyes). Each
/// eye is resized from its NATIVE resolution (`*w`x`*h`, per-HMD) to the model's
/// 100x100. `flip_left`/`flip_right` apply the per-eye handedness mirror (A/B
/// experiment; the eyes are anatomical mirror images).
#[allow(clippy::too_many_arguments)]
pub fn to_input_stereo(
    left: &[u8], lw: u32, lh: u32,
    right: &[u8], rw: u32, rh: u32,
    flip_left: bool, flip_right: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; 2 * DST * DST];
    resize_into_flip(left, lw as usize, lh as usize, &mut out[..DST * DST], flip_left);
    resize_into_flip(right, rw as usize, rh as usize, &mut out[DST * DST..], flip_right);
    out
}

/// Convenience for the 200x200 (VR4/StarVR) callers + tests: stereo input at the
/// fixed source size, no flip.
pub fn vr4_to_input_stereo(left: &[u8], right: &[u8]) -> Vec<f32> {
    to_input_stereo(left, SRC as u32, SRC as u32, right, SRC as u32, SRC as u32, false, false)
}

/// 200x200 stereo input with the per-eye mirror (tests / A/B).
pub fn vr4_to_input_stereo_flip(left: &[u8], right: &[u8], flip_left: bool, flip_right: bool) -> Vec<f32> {
    to_input_stereo(left, SRC as u32, SRC as u32, right, SRC as u32, SRC as u32, flip_left, flip_right)
}

/// Bilinear sample of a `w`x`h` grayscale image at fractional `(x, y)`, edge-clamped,
/// returned in [0,1]. Out-of-frame reads clamp to the border (no wrap, no panic).
fn sample_bilinear(img: &[u8], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w == 0 || h == 0 {
        return 0.0;
    }
    let xf = x.clamp(0.0, (w - 1) as f32);
    let yf = y.clamp(0.0, (h - 1) as f32);
    let x0 = xf.floor() as usize;
    let y0 = yf.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let (tx, ty) = (xf - x0 as f32, yf - y0 as f32);
    let g = |xx: usize, yy: usize| img[yy * w + xx] as f32;
    let top = g(x0, y0) * (1.0 - tx) + g(x1, y0) * tx;
    let bot = g(x0, y1) * (1.0 - tx) + g(x1, y1) * tx;
    (top * (1.0 - ty) + bot * ty) / 255.0
}

/// Geometric warp of a `w`x`h` grayscale frame into the model's `DST`x`DST` f32 input:
/// crop a sub-window, stretch it (scale_x/y), and rotate it, sampling bilinearly. Used
/// ONLY when the geometry is non-identity (identity keeps the exact legacy resize).
/// `flip_h` mirrors the result horizontally (per-eye handedness), like the resize path.
fn warp_into(img: &[u8], w: usize, h: usize, dst: &mut [f32], g: &MlGeometry, flip_h: bool) {
    if w == 0 || h == 0 {
        for v in dst.iter_mut().take(DST * DST) {
            *v = 0.0;
        }
        return;
    }
    let (wf, hf) = (w as f32, h as f32);
    // Sanitize the geometry first: a hand-edited / corrupt config could carry NaN or inf,
    // which would otherwise turn every sampled pixel into NaN (`NaN.clamp()` stays NaN,
    // `sin_cos(inf)` is NaN) and poison the whole model input. Replace non-finite with the
    // identity value, then clamp to a sane range.
    let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
    // Crop window in source pixels (each edge clamped so the window stays non-empty).
    let cl = fin(g.crop_left, 0.0).clamp(0.0, 0.9) * wf;
    let cr = fin(g.crop_right, 0.0).clamp(0.0, 0.9) * wf;
    let ct = fin(g.crop_top, 0.0).clamp(0.0, 0.9) * hf;
    let cb = fin(g.crop_bottom, 0.0).clamp(0.0, 0.9) * hf;
    let cw = (wf - cl - cr).max(1.0);
    let ch = (hf - ct - cb).max(1.0);
    let (ccx, ccy) = (cl + cw * 0.5, ct + ch * 0.5);
    // Inverse stretch (guard near-zero scale) + inverse rotation, done in normalized
    // centered space so the rotation is isotropic (no shear) before the aspect map.
    let sx = { let v = fin(g.scale_x, 1.0); if v.abs() < 1e-3 { 1e-3 } else { v } };
    let sy = { let v = fin(g.scale_y, 1.0); if v.abs() < 1e-3 { 1e-3 } else { v } };
    let (sn, cs) = fin(g.rotate_deg, 0.0).to_radians().sin_cos();
    for dy in 0..DST {
        let v0 = ((dy as f32 + 0.5) / DST as f32 - 0.5) / sy;
        for dx in 0..DST {
            let u0 = ((dx as f32 + 0.5) / DST as f32 - 0.5) / sx;
            let ru = u0 * cs + v0 * sn;
            let rv = -u0 * sn + v0 * cs;
            let val = sample_bilinear(img, w, h, ccx + ru * cw, ccy + rv * ch);
            let ox = if flip_h { DST - 1 - dx } else { dx };
            dst[dy * DST + ox] = val;
        }
    }
}

/// Stereo ML input with PER-EYE [`MlGeometry`]. Both identity = the exact legacy
/// area-resize path (byte-identical — no ML regression for VR4/StarVR); any
/// crop/stretch/rotation switches that eye to the bilinear warp.
#[allow(clippy::too_many_arguments)]
pub fn to_input_stereo_geom(
    left: &[u8], lw: u32, lh: u32,
    right: &[u8], rw: u32, rh: u32,
    flip_left: bool, flip_right: bool,
    gl: &MlGeometry, gr: &MlGeometry,
) -> Vec<f32> {
    if gl.is_identity() && gr.is_identity() {
        return to_input_stereo(left, lw, lh, right, rw, rh, flip_left, flip_right);
    }
    let mut out = vec![0f32; 2 * DST * DST];
    warp_into(left, lw as usize, lh as usize, &mut out[..DST * DST], gl, flip_left);
    warp_into(right, rw as usize, rh as usize, &mut out[DST * DST..], gr, flip_right);
    out
}

/// Remove isolated bright specular spots (IR / glasses reflections) from a grayscale frame
/// BEFORE the ML sees it: any pixel more than `threshold` (fraction of full scale) above its
/// local mean (window radius `r`) is replaced by the window MEDIAN — so a small hot glint
/// collapses to its surroundings while large bright regions (sclera) survive. Runs on the
/// native-resolution frame; identity when disabled or when nothing is a bright outlier.
pub fn despeckle(frame: &[u8], w: usize, h: usize, p: &DespeckleParams) -> Vec<u8> {
    if !p.enabled || w == 0 || h == 0 || frame.len() < w * h {
        return frame.to_vec();
    }
    let r = (p.radius.max(1) as usize).min(8);
    let thr = (p.threshold.clamp(0.0, 1.0) * 255.0) as i32;
    // Integral image for O(1) window means.
    let stride = w + 1;
    let mut integ = vec![0u64; stride * (h + 1)];
    for y in 0..h {
        let mut rowsum = 0u64;
        for x in 0..w {
            rowsum += frame[y * w + x] as u64;
            integ[(y + 1) * stride + (x + 1)] = integ[y * stride + (x + 1)] + rowsum;
        }
    }
    let rect_sum = |x0: usize, y0: usize, x1: usize, y1: usize| -> u64 {
        integ[(y1 + 1) * stride + (x1 + 1)] + integ[y0 * stride + x0]
            - integ[y0 * stride + (x1 + 1)]
            - integ[(y1 + 1) * stride + x0]
    };
    let mut out = frame.to_vec();
    let mut buf: Vec<u8> = Vec::with_capacity((2 * r + 1) * (2 * r + 1));
    for y in 0..h {
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let y0 = y.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let y1 = (y + r).min(h - 1);
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as u64;
            let mean = (rect_sum(x0, y0, x1, y1) / area.max(1)) as i32;
            if frame[y * w + x] as i32 - mean > thr {
                buf.clear();
                for yy in y0..=y1 {
                    let row = yy * w;
                    for xx in x0..=x1 {
                        buf.push(frame[row + xx]);
                    }
                }
                buf.sort_unstable();
                out[y * w + x] = buf[buf.len() / 2];
            }
        }
    }
    out
}

/// Illumination "flatten" (flat-field): subtract the smooth (large-kernel) local-mean
/// deviation from the global mean, so a low-frequency shadow / gradient (e.g. the close-up
/// centre band) is removed while the eye's high-frequency structure survives. Identity when
/// disabled. Runs after despeckle, before brightness normalization.
pub fn flatten(frame: &[u8], w: usize, h: usize, p: &FlattenParams) -> Vec<u8> {
    if !p.enabled || w == 0 || h == 0 || frame.len() < w * h {
        return frame.to_vec();
    }
    let r = ((p.radius.clamp(0.05, 0.5) * w.min(h) as f32) as usize).max(1);
    let s = p.strength.clamp(0.0, 1.0);
    // Integral image for O(1) window means.
    let stride = w + 1;
    let mut integ = vec![0u64; stride * (h + 1)];
    for y in 0..h {
        let mut rowsum = 0u64;
        for x in 0..w {
            rowsum += frame[y * w + x] as u64;
            integ[(y + 1) * stride + (x + 1)] = integ[y * stride + (x + 1)] + rowsum;
        }
    }
    let global_mean = integ[h * stride + w] as f32 / (w * h) as f32;
    let rect_sum = |x0: usize, y0: usize, x1: usize, y1: usize| -> u64 {
        integ[(y1 + 1) * stride + (x1 + 1)] + integ[y0 * stride + x0]
            - integ[y0 * stride + (x1 + 1)]
            - integ[(y1 + 1) * stride + x0]
    };
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for x in 0..w {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let area = ((x1 - x0 + 1) * (y1 - y0 + 1)) as f32;
            let local = rect_sum(x0, y0, x1, y1) as f32 / area;
            let v = frame[y * w + x] as f32 - s * (local - global_mean);
            out[y * w + x] = v.clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// The processed `DST`x`DST` model input for ONE eye as grayscale u8 — for the
/// Calibration tab's live geometry preview (shows exactly what the model sees).
pub fn ml_input_preview(img: &[u8], w: u32, h: u32, geom: &MlGeometry, flip_h: bool) -> Vec<u8> {
    let mut f = vec![0f32; DST * DST];
    if geom.is_identity() {
        resize_into_flip(img, w as usize, h as usize, &mut f, flip_h);
    } else {
        warp_into(img, w as usize, h as usize, &mut f, geom, flip_h);
    }
    f.iter().map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8).collect()
}

/// Brow CNN input side (the model takes 64x64).
pub const BROW_SIDE: usize = 64;

/// Fixed-point precision for the PIL-compatible bilinear resampler (PIL uses 8).
const PIL_PREC: i32 = 8;

/// Precompute PIL's per-output-pixel bilinear filter taps for a 1-D `insize -> outsize`
/// resample, returned as `(xmin, coeffs)` per output pixel where `coeffs` are fixed-point
/// (`<< PIL_PREC`) integer weights over the input window `[xmin, xmin+coeffs.len())`.
///
/// Mirrors Pillow `precompute_coeffs` + `normalize_coeffs_8bpc` (src/libImaging/Resample.c):
///   * BILINEAR/triangle filter, `support = 1.0`, scaled by `filterscale = max(scale, 1)`
///     on downscale so the filter antialiases (this is exactly what `Image.resize(BILINEAR)`
///     and hence `torchvision.transforms.functional.resize` do on a PIL image).
///   * float taps normalized to sum 1, then converted to fixed point with the invariant that
///     the taps sum to exactly `1 << PIL_PREC` (largest-remainder rounding). PIL guarantees
///     this too — a flat input resizes to itself with no drift — so a constant frame is exact.
///     Non-flat inputs still see ~1 LSB of residual vs Pillow (its exact per-tap float
///     rounding is not reproducible bit-for-bit); see `brow_pp_parity` for the measurement.
///   * All intermediate math is `f64` to track PIL's C `double`.
fn pil_bilinear_coeffs(insize: usize, outsize: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = insize as f64 / outsize as f64;
    let filterscale = if scale >= 1.0 { scale } else { 1.0 };
    let support = filterscale; // triangle support 1.0 * filterscale
    let ss = 1.0 / filterscale;
    let target = 1i64 << PIL_PREC;
    let mut out = Vec::with_capacity(outsize);
    for xx in 0..outsize {
        let center = (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5) as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5) as i64;
        if xmax > insize as i64 {
            xmax = insize as i64;
        }
        let xmin = xmin as usize;
        let ksize = (xmax as usize).saturating_sub(xmin);
        let mut k = Vec::with_capacity(ksize);
        let mut ww = 0.0f64;
        for x in 0..ksize {
            // triangle filter of the scaled offset
            let arg = ((x + xmin) as f64 - center + 0.5) * ss;
            let a = arg.abs();
            let w = if a < 1.0 { 1.0 - a } else { 0.0 };
            k.push(w);
            ww += w;
        }
        if ww != 0.0 {
            for w in k.iter_mut() {
                *w /= ww;
            }
        }
        // Largest-remainder fixed-point rounding: floor every tap, then hand the deficit
        // (`target - sum(floors)`) to the taps with the biggest fractional parts. This keeps
        // the sum at exactly `target` (constants stay exact) while staying within 1 of the
        // per-tap round-to-nearest value.
        let scale_fp = target as f64;
        let raw: Vec<f64> = k.iter().map(|&w| w * scale_fp).collect();
        let mut coeffs: Vec<i32> = raw.iter().map(|&r| r.floor() as i32).collect();
        let sum: i32 = coeffs.iter().sum();
        let mut deficit = target as i32 - sum;
        if deficit != 0 && !coeffs.is_empty() {
            // Order tap indices by descending fractional remainder.
            let mut order: Vec<usize> = (0..raw.len()).collect();
            order.sort_by(|&a, &b| {
                let fa = raw[a] - raw[a].floor();
                let fb = raw[b] - raw[b].floor();
                fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
            });
            // Positive deficit: bump the largest-remainder taps up; negative: trim smallest.
            if deficit > 0 {
                for &i in order.iter() {
                    if deficit == 0 {
                        break;
                    }
                    coeffs[i] += 1;
                    deficit -= 1;
                }
            } else {
                for &i in order.iter().rev() {
                    if deficit == 0 {
                        break;
                    }
                    coeffs[i] -= 1;
                    deficit += 1;
                }
            }
        }
        out.push((xmin, coeffs));
    }
    out
}

/// PIL-compatible antialiased BILINEAR downscale of a `sw`x`sh` u8 image to `dw`x`dh` u8.
/// Separable (horizontal pass -> u8 intermediate -> vertical pass), fixed-point with a
/// `1 << (PIL_PREC-1)` rounding bias and `>> PIL_PREC` truncation, clamped to `0..=255` —
/// byte-for-byte the same structure as Pillow's `ImagingResample*_8bpc`.
fn pil_bilinear_resize(src: &[u8], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let bias = 1i64 << (PIL_PREC - 1);
    let clip = |v: i64| -> u8 {
        let v = v >> PIL_PREC;
        if v < 0 {
            0
        } else if v > 255 {
            255
        } else {
            v as u8
        }
    };
    // Horizontal pass: sw -> dw, output (sh, dw) u8.
    let hc = pil_bilinear_coeffs(sw, dw);
    let mut tmp = vec![0u8; sh * dw];
    for y in 0..sh {
        let srow = y * sw;
        let drow = y * dw;
        for (ox, (xmin, kk)) in hc.iter().enumerate() {
            let mut acc = bias;
            for (i, &w) in kk.iter().enumerate() {
                acc += src[srow + xmin + i] as i64 * w as i64;
            }
            tmp[drow + ox] = clip(acc);
        }
    }
    // Vertical pass: sh -> dh, output (dh, dw) u8.
    let vc = pil_bilinear_coeffs(sh, dh);
    let mut out = vec![0u8; dh * dw];
    for (oy, (ymin, kk)) in vc.iter().enumerate() {
        let drow = oy * dw;
        for x in 0..dw {
            let mut acc = bias;
            for (i, &w) in kk.iter().enumerate() {
                acc += tmp[(ymin + i) * dw + x] as i64 * w as i64;
            }
            out[drow + x] = clip(acc);
        }
    }
    out
}

/// OpenCV-compatible CLAHE (`cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply`) on a
/// 64x64 u8 image, returned as a new 64x64 u8. 8x8 tiles (8x8 px each), per-tile clipped
/// histogram -> CDF -> LUT, then per-pixel bilinear blend of the 4 nearest tile LUTs with
/// OpenCV's edge/corner clamping. Matches `cv2` bit-for-bit on 64x64 inputs (verified).
///
/// Mirrors OpenCV `clahe.cpp` (CLAHE_Impl): `clipLimit = max(1, round(clip * tileArea/256))`
/// with the clipped excess redistributed uniformly (`residual` spread one-per-bin with a
/// stride), then LUT = `round(cdf * 255/tileArea)`.
fn clahe_8x8(img: &[u8]) -> [u8; BROW_SIDE * BROW_SIDE] {
    const N: usize = BROW_SIDE; // 64
    const TILES: usize = 8;
    const TW: usize = N / TILES; // 8
    const TH: usize = N / TILES; // 8
    const TILE_AREA: usize = TW * TH; // 64
    const HIST: usize = 256;
    // OpenCV: clipLimit = clipLimit * tileArea / histSize, cvRound, then max(1).
    let clip_limit: i32 = {
        let v = (2.0f64 * TILE_AREA as f64 / HIST as f64).round() as i32;
        v.max(1)
    };
    let lut_scale = 255.0f64 / TILE_AREA as f64;

    // Per-tile LUTs [TILES][TILES][256]. Range loops: the tile indices also drive the
    // pixel-offset arithmetic below, so an enumerate() rewrite would be less clear.
    let mut luts = [[[0u8; HIST]; TILES]; TILES];
    #[allow(clippy::needless_range_loop)]
    for ty in 0..TILES {
        for tx in 0..TILES {
            let mut hist = [0i32; HIST];
            for yy in 0..TH {
                let row = (ty * TH + yy) * N + tx * TW;
                for xx in 0..TW {
                    hist[img[row + xx] as usize] += 1;
                }
            }
            // Clip histogram and count excess.
            if clip_limit > 0 {
                let mut excess = 0i32;
                for c in hist.iter_mut() {
                    if *c > clip_limit {
                        excess += *c - clip_limit;
                        *c = clip_limit;
                    }
                }
                // Redistribute: uniform part + residual spread with a stride (OpenCV).
                let redist = excess / HIST as i32;
                let residual = excess - redist * HIST as i32;
                if redist > 0 {
                    for c in hist.iter_mut() {
                        *c += redist;
                    }
                }
                if residual != 0 {
                    let step = (HIST as i32 / residual).max(1) as usize;
                    let mut idx = 0usize;
                    let mut r = residual;
                    while r > 0 {
                        hist[idx] += 1;
                        idx += step;
                        if idx >= HIST {
                            idx = idx - HIST + 1;
                        }
                        r -= 1;
                    }
                }
            }
            // CDF -> LUT.
            let mut cdf = 0i32;
            for v in 0..HIST {
                cdf += hist[v];
                let val = (cdf as f64 * lut_scale).round();
                luts[ty][tx][v] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

    // Per-pixel bilinear interpolation across tile-center LUTs (OpenCV edge clamp).
    let mut out = [0u8; N * N];
    for y in 0..N {
        let gy = (y as f64 - TH as f64 / 2.0) / TH as f64;
        let ty0 = gy.floor();
        let wy = gy - ty0;
        let ty0 = ty0 as i64;
        let ty0c = ty0.clamp(0, TILES as i64 - 1) as usize;
        let ty1c = (ty0 + 1).clamp(0, TILES as i64 - 1) as usize;
        for x in 0..N {
            let gx = (x as f64 - TW as f64 / 2.0) / TW as f64;
            let tx0 = gx.floor();
            let wx = gx - tx0;
            let tx0 = tx0 as i64;
            let tx0c = tx0.clamp(0, TILES as i64 - 1) as usize;
            let tx1c = (tx0 + 1).clamp(0, TILES as i64 - 1) as usize;
            let v = img[y * N + x] as usize;
            let a = luts[ty0c][tx0c][v] as f64;
            let b = luts[ty0c][tx1c][v] as f64;
            let c = luts[ty1c][tx0c][v] as f64;
            let d = luts[ty1c][tx1c][v] as f64;
            let top = a * (1.0 - wx) + b * wx;
            let bot = c * (1.0 - wx) + d * wx;
            out[y * N + x] = (top * (1.0 - wy) + bot * wy).round() as u8;
        }
    }
    out
}

/// Preprocess one eye's grayscale frame for the brow CNN (TinyBrowNet, [`crate::ml::brow_net`]).
///
/// Replicates `the eyebrow training pipeline/dataset.py`'s TRAIN-time inference preprocessing (`is_train=False`,
/// 1-channel) exactly, so the baked brow models trained there infer correctly:
///
///   1. `flip_h` (right eye): the eyebrow training pipeline horizontally flips the FULL right-eye frame at
///      capture so both eyes arrive left-canonical. We replicate that literally — the frame
///      is mirrored up front (folded into the crop read) so the ROI window is the mirror of
///      the left eye's, exactly as the reference does. Because the ROI box is horizontally
///      symmetric and every later stage is flip-equivariant (resize/CLAHE) or flip-invariant
///      (z-score), this is equivalent to flipping the final 64x64; both are verified against
///      the flipped-reference `flip200` fixture in `brow_pp_parity` and in `brow_flip_equiv`.
///   2. ROI crop (PIL `crop` semantics): box `(floor(0.15*w), 0, floor(0.85*w), floor(0.4*h))`.
///   3. Resize the ROI to 64x64 with PIL `Image.resize((64,64), BILINEAR)` — the antialiased
///      triangle filter `torchvision.transforms.functional.resize` calls on a PIL image
///      (see [`pil_bilinear_resize`]).
///   4. CLAHE on the u8 64x64: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` (see
///      [`clahe_8x8`]).
///   5. Per-frame z-score on the 0..255 scale: `(x - mean) / max(std, 1.0)` (population std).
///      NOT divided by 255 first — the `max(std,1)` clamp makes the raw scale load-bearing.
///
/// NOTE the train/infer CLAHE-order discrepancy: `onnx_inference.py`'s `preprocess_crop`
/// applies CLAHE BEFORE resize, but the model was TRAINED by `dataset.py`, which applies
/// CLAHE AFTER resize (on the 64x64). We match TRAIN (resize -> CLAHE @ 64x64).
///
/// Output is `64*64` f32 (row-major). `out: &mut [f32; 64*64]` and `flip_h` are unchanged
/// so the call sites in [`crate::pipeline`] need no edit.
pub fn brow_input(gray: &[u8], w: usize, h: usize, flip_h: bool, out: &mut [f32; BROW_SIDE * BROW_SIDE]) {
    let n = BROW_SIDE;
    if w == 0 || h == 0 || gray.len() < w * h {
        out.fill(0.0);
        return;
    }
    // Step 2: ROI crop (PIL crop box uses floor via `int()` on the float products).
    let left = ((w as f64) * 0.15) as usize;
    let right = ((w as f64) * 0.85) as usize;
    let bottom = ((h as f64) * 0.4) as usize;
    // Degenerate crop (tiny frames) — clamp to a valid non-empty window.
    let left = left.min(w.saturating_sub(1));
    let right = right.max(left + 1).min(w);
    let bottom = bottom.max(1).min(h);
    let cw = right - left;
    let ch = bottom; // top = 0
    // Copy the ROI, applying the full-frame horizontal flip up front for the right eye so
    // the crop window is the mirror of the left eye's (matches the eyebrow training pipeline capture-time flip).
    let mut crop = vec![0u8; cw * ch];
    for y in 0..ch {
        let srow = y * w;
        let drow = y * cw;
        if flip_h {
            // Mirror the full frame: source column for output x is (w-1-(left+x)).
            for x in 0..cw {
                crop[drow + x] = gray[srow + (w - 1 - (left + x))];
            }
        } else {
            crop[drow..drow + cw].copy_from_slice(&gray[srow + left..srow + left + cw]);
        }
    }
    // Step 3: PIL-compatible antialiased bilinear resize to 64x64.
    let resized = pil_bilinear_resize(&crop, cw, ch, n, n);
    // Step 4: OpenCV-compatible CLAHE on the u8 64x64.
    let clahe = clahe_8x8(&resized);
    // Step 5: per-frame z-score on the 0..255 scale (population std), `(x-mean)/max(std,1)`.
    let mut mean = 0.0f32;
    for &v in clahe.iter() {
        mean += v as f32;
    }
    mean /= (n * n) as f32;
    let mut var = 0.0f32;
    for &v in clahe.iter() {
        let d = v as f32 - mean;
        var += d * d;
    }
    var /= (n * n) as f32;
    let inv = 1.0 / var.sqrt().max(1.0);
    for i in 0..n * n {
        out[i] = (clahe[i] as f32 - mean) * inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_duplication() {
        let img = vec![128u8; SRC * SRC];
        let inp = vr4_to_input(&img);
        assert_eq!(inp.len(), 2 * DST * DST);
        // uniform 128 -> 128/255 in both channels.
        let v = 128.0 / 255.0;
        assert!((inp[0] - v).abs() < 1e-6);
        assert!((inp[DST * DST] - v).abs() < 1e-6, "channel 1 duplicates channel 0");
        assert_eq!(&inp[..DST * DST], &inp[DST * DST..]);
    }

    #[test]
    fn per_eye_flip_mirrors_only_that_channel() {
        // Left half-black / half-white frame; flipping LEFT mirrors ch0 but not ch1.
        let mut img = vec![0u8; SRC * SRC];
        for y in 0..SRC {
            for x in SRC / 2..SRC {
                img[y * SRC + x] = 255; // right half white
            }
        }
        let plain = vr4_to_input_stereo(&img, &img);
        let flip_l = vr4_to_input_stereo_flip(&img, &img, true, false);
        // ch0 (left) row reversed: leftmost output pixel now white (was black).
        assert!(plain[0] < 0.5 && flip_l[0] > 0.5, "left channel mirrored");
        // ch1 (right) unchanged.
        assert_eq!(&plain[DST * DST..], &flip_l[DST * DST..], "right channel untouched");
    }

    #[test]
    fn resize_non_200_area_average() {
        // A non-200 resolution (Varjo-like) must area-average to the model's 100x100.
        // Uniform 120 -> every dst pixel = 120/255 in both channels.
        let img = vec![120u8; 300 * 300];
        let out = to_input_stereo(&img, 300, 300, &img, 300, 300, false, false);
        let v = 120.0 / 255.0;
        assert!((out[0] - v).abs() < 1e-6, "uniform resize ch0, got {}", out[0] * 255.0);
        assert!((out[DST * DST] - v).abs() < 1e-6, "uniform resize ch1");
        // Left-black / right-white 300-wide -> dst left col dark, right col bright.
        let mut split = vec![0u8; 300 * 300];
        for y in 0..300 {
            for x in 150..300 {
                split[y * 300 + x] = 255;
            }
        }
        let o2 = to_input_stereo(&split, 300, 300, &split, 300, 300, false, false);
        assert!(o2[0] < 0.1, "left edge dark, got {}", o2[0]);
        assert!(o2[DST - 1] > 0.9, "right edge bright, got {}", o2[DST - 1]);
    }

    #[test]
    fn box_average_of_2x2() {
        // First 2x2 block = [0, 100, 200, 100] -> avg 100.
        let mut img = vec![0u8; SRC * SRC];
        img[0] = 0;
        img[1] = 100;
        img[SRC] = 200;
        img[SRC + 1] = 100;
        let inp = vr4_to_input(&img);
        assert!((inp[0] - 100.0 / 255.0).abs() < 1e-6, "got {}", inp[0] * 255.0);
    }

    /// Uniform frame -> CLAHE is identity-ish (flat histogram), z-score of a constant is 0.
    #[test]
    fn brow_uniform_is_zero() {
        let img = vec![120u8; 200 * 200];
        let mut out = [0f32; BROW_SIDE * BROW_SIDE];
        brow_input(&img, 200, 200, false, &mut out);
        // A constant frame z-scores to all-zero (std==0 -> clamp to 1, numerator 0).
        let max = out.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(max < 1e-6, "uniform frame should z-score to 0, got max {max}");
    }

    /// Degenerate sizes must not panic and must produce a finite, zero-mean-ish output.
    #[test]
    fn brow_tiny_frame_no_panic() {
        for (w, h) in [(1usize, 1usize), (2, 3), (5, 5), (64, 1)] {
            let img = vec![90u8; w * h];
            let mut out = [0f32; BROW_SIDE * BROW_SIDE];
            brow_input(&img, w, h, false, &mut out);
            assert!(out.iter().all(|v| v.is_finite()), "{w}x{h} produced non-finite");
        }
    }

    /// PIL bilinear on an exact 2x downscale of a constant is that constant (sanity).
    #[test]
    fn pil_bilinear_constant() {
        let src = vec![137u8; 128 * 128];
        let r = pil_bilinear_resize(&src, 128, 128, 64, 64);
        assert!(r.iter().all(|&v| v == 137), "constant resize drifted");
        // Non-square, non-integer ratio must also hold a constant.
        let r2 = pil_bilinear_resize(&vec![200u8; 120 * 90], 120, 90, 64, 64);
        assert!(r2.iter().all(|&v| v == 200), "non-integer-ratio constant drifted");
    }

    /// `flip_h=true` (flip the full frame before crop, as the eyebrow training pipeline's capture does) must
    /// equal preprocessing a pre-flipped frame with `flip_h=false`.
    #[test]
    fn brow_flip_equiv() {
        let (w, h) = (200usize, 150usize);
        let mut img = vec![0u8; w * h];
        // Left/right-asymmetric content so the flip is observable.
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = (((x * 7 + y * 3) % 200) as u8).wrapping_add(if x < w / 2 { 0 } else { 40 });
            }
        }
        let mut flipped = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                flipped[y * w + x] = img[y * w + (w - 1 - x)];
            }
        }
        let mut a = [0f32; BROW_SIDE * BROW_SIDE];
        let mut b = [0f32; BROW_SIDE * BROW_SIDE];
        brow_input(&img, w, h, true, &mut a); // flip inside
        brow_input(&flipped, w, h, false, &mut b); // flip outside
        let max = a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
        assert!(max < 1e-6, "flip-inside vs flip-outside differ by {max}");
    }

    /// Fixture parity against the eyebrow training pipeline/dataset.py's inference preprocessing.
    ///
    /// Skipped unless `BROW_PP_FIXTURE_DIR` points at the folder written by
    /// `tools/gen_brow_pp_fixtures.py` (so plain `cargo test` stays green). Each fixture is
    /// `<name>.raw` (u32 w, u32 h, then h*w u8) + `<name>.f32` (64x64 expected f32), listed
    /// in `manifest.json` with a `flip_h` flag.
    ///
    /// The CLAHE and z-score stages are bit-exact vs OpenCV/numpy; the residual is entirely
    /// the ~1-LSB difference between our PIL-compatible bilinear resampler and Pillow's own
    /// (a handful of u8 pixels), amplified by CLAHE's step LUT. Tolerance is set to the
    /// empirically-measured worst case; the per-image max diff is printed with --nocapture.
    #[test]
    fn brow_pp_parity() {
        let dir = match std::env::var("BROW_PP_FIXTURE_DIR") {
            Ok(d) => d,
            Err(_) => return, // no fixtures -> skip
        };
        let p = std::path::Path::new(&dir);
        let manifest = std::fs::read_to_string(p.join("manifest.json")).expect("manifest.json");
        // Minimal hand parse: pull out {"name":..,"w":..,"h":..,"flip_h":..} records.
        let mut worst_overall = 0.0f32;
        let mut count = 0usize;
        for rec in manifest.split('{').skip(1) {
            let name = json_str(rec, "name");
            if name.is_empty() {
                continue;
            }
            let flip_h = rec.contains("\"flip_h\": true") || rec.contains("\"flip_h\":true");
            let raw = std::fs::read(p.join(format!("{name}.raw"))).expect("raw fixture");
            assert!(raw.len() >= 8, "raw too short");
            let w = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as usize;
            let h = u32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]) as usize;
            let gray = &raw[8..];
            assert_eq!(gray.len(), w * h, "{name}: raw pixel count");
            let expected: Vec<f32> = std::fs::read(p.join(format!("{name}.f32")))
                .expect("f32 fixture")
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(expected.len(), BROW_SIDE * BROW_SIDE, "{name}: expected size");
            let mut got = [0f32; BROW_SIDE * BROW_SIDE];
            brow_input(gray, w, h, flip_h, &mut got);
            let max_diff = got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rms = (got
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                / (BROW_SIDE * BROW_SIDE) as f32)
                .sqrt();
            eprintln!("[brow_pp] {name:14} flip={flip_h} max={max_diff:.5} rms={rms:.6}");
            worst_overall = worst_overall.max(max_diff);
            count += 1;
            // Per-image bound: resize is ~1 LSB off in a few px, CLAHE's LUT can turn that
            // into a bounded jump. 0.75 covers the measured worst (varjo/flip); RMS stays tiny.
            assert!(
                max_diff < 0.75,
                "{name}: brow_input vs dataset.py max diff {max_diff} exceeds tolerance"
            );
        }
        assert!(count > 0, "no fixtures parsed from manifest");
        eprintln!("[brow_pp] worst over {count} fixtures = {worst_overall:.5}");
    }

    #[cfg(test)]
    fn json_str<'a>(rec: &'a str, key: &str) -> String {
        // Find `"key": "value"` inside a flat record chunk.
        let pat = format!("\"{key}\":");
        let Some(i) = rec.find(&pat) else { return String::new() };
        let rest = &rec[i + pat.len()..];
        let Some(q0) = rest.find('"') else { return String::new() };
        let after = &rest[q0 + 1..];
        let Some(q1) = after.find('"') else { return String::new() };
        after[..q1].to_string()
    }

    #[test]
    fn geom_identity_matches_legacy_resize() {
        // Identity geometry MUST be byte-identical to the legacy stereo resize, so the
        // VR4/StarVR ML input is unchanged (no regression) until the user tunes it.
        let img: Vec<u8> = (0..SRC * SRC).map(|i| (i % 256) as u8).collect();
        let legacy = vr4_to_input_stereo(&img, &img);
        let geom = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, false,
            &MlGeometry::default(), &MlGeometry::default(),
        );
        assert_eq!(legacy, geom, "identity geometry must equal the legacy resize");
    }

    #[test]
    fn geom_crop_changes_input() {
        // A non-identity crop must actually change the model input (warp path taken).
        let img: Vec<u8> = (0..SRC * SRC).map(|i| ((i * 3) % 256) as u8).collect();
        let base = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, false,
            &MlGeometry::default(), &MlGeometry::default(),
        );
        let g = MlGeometry { crop_left: 0.25, crop_top: 0.25, ..MlGeometry::default() };
        let cropped = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, false, &g, &g,
        );
        assert_ne!(base, cropped, "a crop must change the model input");
    }

    #[test]
    fn geom_preview_is_dst_sized_and_preserves_flat() {
        let img = vec![128u8; SRC * SRC];
        // Identity → flat stays flat, DST*DST long.
        let p = ml_input_preview(&img, SRC as u32, SRC as u32, &MlGeometry::default(), false);
        assert_eq!(p.len(), DST * DST);
        assert!(p.iter().all(|&v| (120..=136).contains(&v)), "flat input stays ~flat");
        // A rotation of a flat frame is still flat (interior), and still DST-sized.
        let g = MlGeometry { rotate_deg: 20.0, ..MlGeometry::default() };
        let r = ml_input_preview(&img, SRC as u32, SRC as u32, &g, false);
        assert_eq!(r.len(), DST * DST);
        let mid = r[(DST / 2) * DST + DST / 2];
        assert!((120..=136).contains(&mid), "rotated flat frame stays flat at center, got {mid}");
    }

    #[test]
    fn despeckle_removes_isolated_bright_spot() {
        let (w, h) = (40usize, 40usize);
        let mut img = vec![100u8; w * h];
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            img[(20 + dy) * w + (20 + dx)] = 255; // a small specular spot
        }
        let p = DespeckleParams { enabled: true, threshold: 0.15, radius: 3 };
        let out = despeckle(&img, w, h, &p);
        assert!(out[20 * w + 20] < 130, "bright spot collapsed toward grey, got {}", out[20 * w + 20]);
        assert_eq!(out[0], 100, "flat background untouched");
        let off = DespeckleParams { enabled: false, ..p };
        assert_eq!(despeckle(&img, w, h, &off), img, "disabled = identity");
    }

    #[test]
    fn despeckle_keeps_large_bright_region() {
        // A big bright block (like the sclera) is NOT a spot — its local mean is bright too,
        // so it must survive.
        let (w, h) = (40usize, 40usize);
        let mut img = vec![60u8; w * h];
        for y in 10..30 {
            for x in 10..30 {
                img[y * w + x] = 230;
            }
        }
        let p = DespeckleParams { enabled: true, threshold: 0.15, radius: 3 };
        let out = despeckle(&img, w, h, &p);
        assert_eq!(out[20 * w + 20], 230, "interior of a large bright region is preserved");
    }

    #[test]
    fn flatten_reduces_a_smooth_gradient() {
        // A smooth left->right ramp (a lateral illumination shadow) must be flattened toward
        // uniform, while disabled is identity.
        let (w, h) = (80usize, 80usize);
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = (40 + x * 150 / w) as u8;
            }
        }
        let p = FlattenParams { enabled: true, strength: 1.0, radius: 0.33 };
        let out = flatten(&img, w, h, &p);
        let in_span = img[40 * w + 60] as i32 - img[40 * w + 20] as i32;
        let out_span = out[40 * w + 60] as i32 - out[40 * w + 20] as i32;
        assert!(
            out_span.abs() < in_span.abs() - 20,
            "smooth gradient flattened: in {in_span} -> out {out_span}"
        );
        let off = FlattenParams { enabled: false, ..p };
        assert_eq!(flatten(&img, w, h, &off), img, "disabled = identity");
    }

    #[test]
    fn geom_nonfinite_does_not_poison_input() {
        // A corrupt config (NaN / inf geometry) must NOT produce a NaN model input.
        let img = vec![128u8; SRC * SRC];
        let g = MlGeometry {
            rotate_deg: f32::INFINITY,
            scale_x: f32::NAN,
            crop_left: f32::NAN,
            ..MlGeometry::default()
        };
        let out = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, false, &g, &g,
        );
        assert!(out.iter().all(|v| v.is_finite()), "non-finite geometry must not NaN the input");
    }
}

    #[test]
    fn despeckle_edge_case_small_frame() {
        // Test with very small frames where radius > dimensions
        let p = DespeckleParams { enabled: true, threshold: 0.15, radius: 10 };
        
        // 2x2 frame
        let img2x2 = vec![100u8; 2 * 2];
        let out = despeckle(&img2x2, 2, 2, &p);
        assert_eq!(out.len(), 4, "2x2 output length");
        
        // 1x1 frame
        let img1x1 = vec![150u8; 1 * 1];
        let out = despeckle(&img1x1, 1, 1, &p);
        assert_eq!(out.len(), 1, "1x1 output length");
        
        // 100x1 (single row)
        let img100x1 = vec![100u8; 100 * 1];
        let out = despeckle(&img100x1, 100, 1, &p);
        assert_eq!(out.len(), 100, "100x1 output length");
    }
