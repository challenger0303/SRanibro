
use crate::core::types::{DespeckleParams, FlattenParams, MlGeometry};

pub const SRC: usize = 200;
pub const DST: usize = 100;

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
            let avg = (a + b + c + d + 2) / 4;
            dst[y * DST + x] = avg as f32 / 255.0;
        }
    }
}

pub fn vr4_to_input(img: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; 2 * DST * DST];
    downscale_into(img, &mut out[..DST * DST]);
    out.copy_within(..DST * DST, DST * DST);
    out
}

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
            let mut sum = 0u64;
            let mut cnt = 0u64;
            for sy in sy0..sy1 {
                let row = sy * w;
                for sx in sx0..sx1 {
                    sum += img[row + sx] as u64;
                    cnt += 1;
                }
            }
            dst[dy * DST + dx] = if cnt > 0 {
                sum as f32 / cnt as f32 / 255.0
            } else {
                0.0
            };
        }
    }
}

fn resize_into_flip(img: &[u8], w: usize, h: usize, dst: &mut [f32], flip_h: bool) {
    resize_into(img, w, h, dst);
    if flip_h {
        for y in 0..DST {
            dst[y * DST..(y + 1) * DST].reverse();
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn to_input_stereo(
    left: &[u8],
    lw: u32,
    lh: u32,
    right: &[u8],
    rw: u32,
    rh: u32,
    flip_left: bool,
    flip_right: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; 2 * DST * DST];
    resize_into_flip(
        left,
        lw as usize,
        lh as usize,
        &mut out[..DST * DST],
        flip_left,
    );
    resize_into_flip(
        right,
        rw as usize,
        rh as usize,
        &mut out[DST * DST..],
        flip_right,
    );
    out
}

pub fn vr4_to_input_stereo(left: &[u8], right: &[u8]) -> Vec<f32> {
    to_input_stereo(
        left, SRC as u32, SRC as u32, right, SRC as u32, SRC as u32, false, false,
    )
}

pub fn vr4_to_input_stereo_flip(
    left: &[u8],
    right: &[u8],
    flip_left: bool,
    flip_right: bool,
) -> Vec<f32> {
    to_input_stereo(
        left, SRC as u32, SRC as u32, right, SRC as u32, SRC as u32, flip_left, flip_right,
    )
}

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

fn warp_into(img: &[u8], w: usize, h: usize, dst: &mut [f32], g: &MlGeometry, flip_h: bool) {
    if w == 0 || h == 0 {
        for v in dst.iter_mut().take(DST * DST) {
            *v = 0.0;
        }
        return;
    }
    let (wf, hf) = (w as f32, h as f32);
    let fin = |v: f32, d: f32| if v.is_finite() { v } else { d };
    let cl = fin(g.crop_left, 0.0).clamp(0.0, 0.9) * wf;
    let cr = fin(g.crop_right, 0.0).clamp(0.0, 0.9) * wf;
    let ct = fin(g.crop_top, 0.0).clamp(0.0, 0.9) * hf;
    let cb = fin(g.crop_bottom, 0.0).clamp(0.0, 0.9) * hf;
    let cw = (wf - cl - cr).max(1.0);
    let ch = (hf - ct - cb).max(1.0);
    let (ccx, ccy) = (cl + cw * 0.5, ct + ch * 0.5);
    let sx = {
        let v = fin(g.scale_x, 1.0);
        if v.abs() < 1e-3 {
            1e-3
        } else {
            v
        }
    };
    let sy = {
        let v = fin(g.scale_y, 1.0);
        if v.abs() < 1e-3 {
            1e-3
        } else {
            v
        }
    };
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

#[allow(clippy::too_many_arguments)]
pub fn to_input_stereo_geom(
    left: &[u8],
    lw: u32,
    lh: u32,
    right: &[u8],
    rw: u32,
    rh: u32,
    flip_left: bool,
    flip_right: bool,
    gl: &MlGeometry,
    gr: &MlGeometry,
) -> Vec<f32> {
    if gl.is_identity() && gr.is_identity() {
        return to_input_stereo(left, lw, lh, right, rw, rh, flip_left, flip_right);
    }
    let mut out = vec![0f32; 2 * DST * DST];
    warp_into(
        left,
        lw as usize,
        lh as usize,
        &mut out[..DST * DST],
        gl,
        flip_left,
    );
    warp_into(
        right,
        rw as usize,
        rh as usize,
        &mut out[DST * DST..],
        gr,
        flip_right,
    );
    out
}

pub fn despeckle(frame: &[u8], w: usize, h: usize, p: &DespeckleParams) -> Vec<u8> {
    if !p.enabled || w == 0 || h == 0 || frame.len() < w * h {
        return frame.to_vec();
    }
    let r = (p.radius.max(1) as usize).min(8);
    let thr = (p.threshold.clamp(0.0, 1.0) * 255.0) as i32;
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

pub fn flatten(frame: &[u8], w: usize, h: usize, p: &FlattenParams) -> Vec<u8> {
    if !p.enabled || w == 0 || h == 0 || frame.len() < w * h {
        return frame.to_vec();
    }
    let r = ((p.radius.clamp(0.05, 0.5) * w.min(h) as f32) as usize).max(1);
    let s = p.strength.clamp(0.0, 1.0);
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

pub fn fitted_photometric(
    frame: &[u8],
    w: usize,
    h: usize,
    geometry: &MlGeometry,
    correction: &crate::core::types::PhotometricCorrection,
    eye: usize,
) -> Vec<u8> {
    if !correction.enabled || eye >= 2 || w == 0 || h == 0 || frame.len() < w * h {
        return frame.to_vec();
    }

    let flatten_params = FlattenParams {
        enabled: correction.flatten.enabled,
        strength: if correction.flatten.strength.is_finite() {
            correction.flatten.strength.clamp(0.0, 0.75)
        } else {
            0.0
        },
        radius: if correction.flatten.radius.is_finite() {
            correction.flatten.radius.clamp(0.15, 0.5)
        } else {
            0.33
        },
    };
    let flattened = flatten_params
        .enabled
        .then(|| flatten(frame, w, h, &flatten_params));
    let source = flattened.as_deref().unwrap_or(frame);

    let [raw_gain, raw_bias] = correction.affine[eye];
    let gain = if raw_gain.is_finite() {
        raw_gain.clamp(0.70, 1.30)
    } else {
        1.0
    };
    let bias = if raw_bias.is_finite() {
        raw_bias.clamp(-30.0, 30.0)
    } else {
        0.0
    };
    let mut affine = crate::ml::brightness::apply(source, gain, bias);

    let field = correction.field[eye];
    if field.is_identity() {
        return affine;
    }
    let finite_or_zero = |value: f32, bound: f32| {
        if value.is_finite() {
            value.clamp(-bound, bound)
        } else {
            0.0
        }
    };
    let horizontal = finite_or_zero(field.horizontal, 0.12);
    let vertical = finite_or_zero(field.vertical, 0.12);
    let horizontal_curve = finite_or_zero(field.horizontal_curve, 0.08);
    let vertical_curve = finite_or_zero(field.vertical_curve, 0.08);
    if horizontal == 0.0 && vertical == 0.0 && horizontal_curve == 0.0 && vertical_curve == 0.0 {
        return affine;
    }

    let x0 = (geometry.crop_left.clamp(0.0, 0.95) * w as f32).floor() as usize;
    let x1 = ((1.0 - geometry.crop_right.clamp(0.0, 0.95)) * w as f32).ceil() as usize;
    let y0 = (geometry.crop_top.clamp(0.0, 0.95) * h as f32).floor() as usize;
    let y1 = ((1.0 - geometry.crop_bottom.clamp(0.0, 0.95)) * h as f32).ceil() as usize;
    let (x1, y1) = (x1.clamp(x0 + 1, w), y1.clamp(y0 + 1, h));
    for y in 0..h {
        let yn = if y1 - y0 <= 1 {
            0.0
        } else {
            2.0 * (y as f32 - y0 as f32) / (y1 - y0 - 1) as f32 - 1.0
        };
        for x in 0..w {
            let xn = if x1 - x0 <= 1 {
                0.0
            } else {
                2.0 * (x as f32 - x0 as f32) / (x1 - x0 - 1) as f32 - 1.0
            };
            let local_gain = (1.0
                + horizontal * xn
                + vertical * yn
                + horizontal_curve * (xn * xn - 1.0 / 3.0)
                + vertical_curve * (yn * yn - 1.0 / 3.0))
                .clamp(0.70, 1.30);
            let index = y * w + x;
            affine[index] = (affine[index] as f32 * local_gain).clamp(0.0, 255.0) as u8;
        }
    }
    affine
}

pub fn ml_input_preview(img: &[u8], w: u32, h: u32, geom: &MlGeometry, flip_h: bool) -> Vec<u8> {
    let mut f = vec![0f32; DST * DST];
    if geom.is_identity() {
        resize_into_flip(img, w as usize, h as usize, &mut f, flip_h);
    } else {
        warp_into(img, w as usize, h as usize, &mut f, geom, flip_h);
    }
    f.iter()
        .map(|&v| (v * 255.0).clamp(0.0, 255.0) as u8)
        .collect()
}

pub const BROW_SIDE: usize = 64;
pub const WIDE_SIDE: usize = BROW_SIDE;

const PIL_PREC: i32 = 8;

fn pil_bilinear_coeffs(insize: usize, outsize: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = insize as f64 / outsize as f64;
    let filterscale = if scale >= 1.0 { scale } else { 1.0 };
    let support = filterscale;
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
        let scale_fp = target as f64;
        let raw: Vec<f64> = k.iter().map(|&w| w * scale_fp).collect();
        let mut coeffs: Vec<i32> = raw.iter().map(|&r| r.floor() as i32).collect();
        let sum: i32 = coeffs.iter().sum();
        let mut deficit = target as i32 - sum;
        if deficit != 0 && !coeffs.is_empty() {
            let mut order: Vec<usize> = (0..raw.len()).collect();
            order.sort_by(|&a, &b| {
                let fa = raw[a] - raw[a].floor();
                let fb = raw[b] - raw[b].floor();
                fb.partial_cmp(&fa).unwrap_or(std::cmp::Ordering::Equal)
            });
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

fn clahe_8x8(img: &[u8]) -> [u8; BROW_SIDE * BROW_SIDE] {
    const N: usize = BROW_SIDE;
    const TILES: usize = 8;
    const TW: usize = N / TILES;
    const TH: usize = N / TILES;
    const TILE_AREA: usize = TW * TH;
    const HIST: usize = 256;
    let clip_limit: i32 = {
        let v = (2.0f64 * TILE_AREA as f64 / HIST as f64).round() as i32;
        v.max(1)
    };
    let lut_scale = 255.0f64 / TILE_AREA as f64;

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
            if clip_limit > 0 {
                let mut excess = 0i32;
                for c in hist.iter_mut() {
                    if *c > clip_limit {
                        excess += *c - clip_limit;
                        *c = clip_limit;
                    }
                }
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
            let mut cdf = 0i32;
            for v in 0..HIST {
                cdf += hist[v];
                let val = (cdf as f64 * lut_scale).round();
                luts[ty][tx][v] = val.clamp(0.0, 255.0) as u8;
            }
        }
    }

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

pub fn brow_input(
    gray: &[u8],
    w: usize,
    h: usize,
    flip_h: bool,
    out: &mut [f32; BROW_SIDE * BROW_SIDE],
) {
    tiny_eye_input(gray, w, h, flip_h, [0.15, 0.0, 0.85, 0.40], out);
}

pub fn wide_input(
    gray: &[u8],
    w: usize,
    h: usize,
    flip_h: bool,
    out: &mut [f32; BROW_SIDE * BROW_SIDE],
) {
    tiny_eye_input(gray, w, h, flip_h, [0.08, 0.08, 0.92, 0.92], out);
}

fn tiny_eye_input(
    gray: &[u8],
    w: usize,
    h: usize,
    flip_h: bool,
    crop_box: [f64; 4],
    out: &mut [f32; BROW_SIDE * BROW_SIDE],
) {
    let n = BROW_SIDE;
    if w == 0 || h == 0 || gray.len() < w * h {
        out.fill(0.0);
        return;
    }
    let left = ((w as f64) * crop_box[0]) as usize;
    let top = ((h as f64) * crop_box[1]) as usize;
    let right = ((w as f64) * crop_box[2]) as usize;
    let bottom = ((h as f64) * crop_box[3]) as usize;
    let left = left.min(w.saturating_sub(1));
    let top = top.min(h.saturating_sub(1));
    let right = right.max(left + 1).min(w);
    let bottom = bottom.max(top + 1).min(h);
    let cw = right - left;
    let ch = bottom - top;
    let mut crop = vec![0u8; cw * ch];
    for y in 0..ch {
        let srow = (top + y) * w;
        let drow = y * cw;
        if flip_h {
            for x in 0..cw {
                crop[drow + x] = gray[srow + (w - 1 - (left + x))];
            }
        } else {
            crop[drow..drow + cw].copy_from_slice(&gray[srow + left..srow + left + cw]);
        }
    }
    let resized = pil_bilinear_resize(&crop, cw, ch, n, n);
    let clahe = clahe_8x8(&resized);
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
        let v = 128.0 / 255.0;
        assert!((inp[0] - v).abs() < 1e-6);
        assert!(
            (inp[DST * DST] - v).abs() < 1e-6,
            "channel 1 duplicates channel 0"
        );
        assert_eq!(&inp[..DST * DST], &inp[DST * DST..]);
    }

    #[test]
    fn per_eye_flip_mirrors_only_that_channel() {
        let mut img = vec![0u8; SRC * SRC];
        for y in 0..SRC {
            for x in SRC / 2..SRC {
                img[y * SRC + x] = 255;
            }
        }
        let plain = vr4_to_input_stereo(&img, &img);
        let flip_l = vr4_to_input_stereo_flip(&img, &img, true, false);
        assert!(plain[0] < 0.5 && flip_l[0] > 0.5, "left channel mirrored");
        assert_eq!(
            &plain[DST * DST..],
            &flip_l[DST * DST..],
            "right channel untouched"
        );
    }

    #[test]
    fn resize_non_200_area_average() {
        let img = vec![120u8; 300 * 300];
        let out = to_input_stereo(&img, 300, 300, &img, 300, 300, false, false);
        let v = 120.0 / 255.0;
        assert!(
            (out[0] - v).abs() < 1e-6,
            "uniform resize ch0, got {}",
            out[0] * 255.0
        );
        assert!((out[DST * DST] - v).abs() < 1e-6, "uniform resize ch1");
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
        let mut img = vec![0u8; SRC * SRC];
        img[0] = 0;
        img[1] = 100;
        img[SRC] = 200;
        img[SRC + 1] = 100;
        let inp = vr4_to_input(&img);
        assert!(
            (inp[0] - 100.0 / 255.0).abs() < 1e-6,
            "got {}",
            inp[0] * 255.0
        );
    }

    #[test]
    fn brow_uniform_is_zero() {
        let img = vec![120u8; 200 * 200];
        let mut out = [0f32; BROW_SIDE * BROW_SIDE];
        brow_input(&img, 200, 200, false, &mut out);
        let max = out.iter().fold(0f32, |m, &v| m.max(v.abs()));
        assert!(
            max < 1e-6,
            "uniform frame should z-score to 0, got max {max}"
        );
    }

    #[test]
    fn brow_tiny_frame_no_panic() {
        for (w, h) in [(1usize, 1usize), (2, 3), (5, 5), (64, 1)] {
            let img = vec![90u8; w * h];
            let mut out = [0f32; BROW_SIDE * BROW_SIDE];
            brow_input(&img, w, h, false, &mut out);
            assert!(
                out.iter().all(|v| v.is_finite()),
                "{w}x{h} produced non-finite"
            );
        }
    }

    #[test]
    fn pil_bilinear_constant() {
        let src = vec![137u8; 128 * 128];
        let r = pil_bilinear_resize(&src, 128, 128, 64, 64);
        assert!(r.iter().all(|&v| v == 137), "constant resize drifted");
        let r2 = pil_bilinear_resize(&vec![200u8; 120 * 90], 120, 90, 64, 64);
        assert!(
            r2.iter().all(|&v| v == 200),
            "non-integer-ratio constant drifted"
        );
    }

    #[test]
    fn brow_flip_equiv() {
        let (w, h) = (200usize, 150usize);
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] =
                    (((x * 7 + y * 3) % 200) as u8).wrapping_add(if x < w / 2 { 0 } else { 40 });
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
        brow_input(&img, w, h, true, &mut a);
        brow_input(&flipped, w, h, false, &mut b);
        let max = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        assert!(max < 1e-6, "flip-inside vs flip-outside differ by {max}");
    }

    #[test]
    fn wide_uniform_is_zero_and_finite() {
        let image = vec![83u8; 200 * 200];
        let mut out = [f32::NAN; WIDE_SIDE * WIDE_SIDE];
        wide_input(&image, 200, 200, false, &mut out);
        assert!(out.iter().all(|value| value.is_finite()));
        assert!(out.iter().all(|value| value.abs() < 1e-6));
    }

    #[test]
    fn wide_right_eye_canonicalization_is_flip_equivalent() {
        let (w, h) = (200usize, 200usize);
        let image: Vec<u8> = (0..w * h)
            .map(|index| ((index * 17 + index / w * 9) % 251) as u8)
            .collect();
        let mut flipped = vec![0u8; image.len()];
        for y in 0..h {
            for x in 0..w {
                flipped[y * w + x] = image[y * w + (w - 1 - x)];
            }
        }
        let mut inside = [0.0f32; WIDE_SIDE * WIDE_SIDE];
        let mut outside = [0.0f32; WIDE_SIDE * WIDE_SIDE];
        wide_input(&image, w, h, true, &mut inside);
        wide_input(&flipped, w, h, false, &mut outside);
        let max = inside
            .iter()
            .zip(outside.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-6, "Wide canonical mirror differs by {max}");
    }

    #[test]
    fn wide_pp_parity() {
        let dir = match std::env::var("WIDE_PP_FIXTURE_DIR") {
            Ok(dir) => dir,
            Err(_) => return,
        };
        let root = std::path::Path::new(&dir);
        let manifest = std::fs::read_to_string(root.join("manifest.json")).unwrap();
        let mut count = 0usize;
        for record in manifest.split('{').skip(1) {
            let name = json_str(record, "name");
            if name.is_empty() {
                continue;
            }
            let flip = record.contains("\"flip_h\": true") || record.contains("\"flip_h\":true");
            let raw = std::fs::read(root.join(format!("{name}.raw"))).unwrap();
            let w = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
            let h = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
            let expected: Vec<f32> = std::fs::read(root.join(format!("{name}.f32")))
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect();
            let mut got = [0.0f32; WIDE_SIDE * WIDE_SIDE];
            wide_input(&raw[8..], w, h, flip, &mut got);
            let max = got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let rms = (got
                .iter()
                .zip(expected.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / expected.len() as f32)
                .sqrt();
            eprintln!("[wide_pp] {name} flip={flip} max={max:.5} rms={rms:.6}");
            assert!(max < 0.75, "{name}: Wide preprocessing max diff {max}");
            assert!(rms < 0.08, "{name}: Wide preprocessing RMS {rms}");
            count += 1;
        }
        assert!(count > 0, "no Wide fixtures parsed");
    }

    #[test]
    fn brow_pp_parity() {
        let dir = match std::env::var("BROW_PP_FIXTURE_DIR") {
            Ok(d) => d,
            Err(_) => return,
        };
        let p = std::path::Path::new(&dir);
        let manifest = std::fs::read_to_string(p.join("manifest.json")).expect("manifest.json");
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
            assert_eq!(
                expected.len(),
                BROW_SIDE * BROW_SIDE,
                "{name}: expected size"
            );
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
        let pat = format!("\"{key}\":");
        let Some(i) = rec.find(&pat) else {
            return String::new();
        };
        let rest = &rec[i + pat.len()..];
        let Some(q0) = rest.find('"') else {
            return String::new();
        };
        let after = &rest[q0 + 1..];
        let Some(q1) = after.find('"') else {
            return String::new();
        };
        after[..q1].to_string()
    }

    #[test]
    fn geom_identity_matches_legacy_resize() {
        let img: Vec<u8> = (0..SRC * SRC).map(|i| (i % 256) as u8).collect();
        let legacy = vr4_to_input_stereo(&img, &img);
        let geom = to_input_stereo_geom(
            &img,
            SRC as u32,
            SRC as u32,
            &img,
            SRC as u32,
            SRC as u32,
            false,
            false,
            &MlGeometry::default(),
            &MlGeometry::default(),
        );
        assert_eq!(
            legacy, geom,
            "identity geometry must equal the legacy resize"
        );
    }

    #[test]
    fn geom_crop_changes_input() {
        let img: Vec<u8> = (0..SRC * SRC).map(|i| ((i * 3) % 256) as u8).collect();
        let base = to_input_stereo_geom(
            &img,
            SRC as u32,
            SRC as u32,
            &img,
            SRC as u32,
            SRC as u32,
            false,
            false,
            &MlGeometry::default(),
            &MlGeometry::default(),
        );
        let g = MlGeometry {
            crop_left: 0.25,
            crop_top: 0.25,
            ..MlGeometry::default()
        };
        let cropped = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, false, &g, &g,
        );
        assert_ne!(base, cropped, "a crop must change the model input");
    }

    #[test]
    fn geom_preview_is_dst_sized_and_preserves_flat() {
        let img = vec![128u8; SRC * SRC];
        let p = ml_input_preview(&img, SRC as u32, SRC as u32, &MlGeometry::default(), false);
        assert_eq!(p.len(), DST * DST);
        assert!(
            p.iter().all(|&v| (120..=136).contains(&v)),
            "flat input stays ~flat"
        );
        let g = MlGeometry {
            rotate_deg: 20.0,
            ..MlGeometry::default()
        };
        let r = ml_input_preview(&img, SRC as u32, SRC as u32, &g, false);
        assert_eq!(r.len(), DST * DST);
        let mid = r[(DST / 2) * DST + DST / 2];
        assert!(
            (120..=136).contains(&mid),
            "rotated flat frame stays flat at center, got {mid}"
        );
    }

    #[test]
    fn despeckle_removes_isolated_bright_spot() {
        let (w, h) = (40usize, 40usize);
        let mut img = vec![100u8; w * h];
        for (dy, dx) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            img[(20 + dy) * w + (20 + dx)] = 255;
        }
        let p = DespeckleParams {
            enabled: true,
            threshold: 0.15,
            radius: 3,
        };
        let out = despeckle(&img, w, h, &p);
        assert!(
            out[20 * w + 20] < 130,
            "bright spot collapsed toward grey, got {}",
            out[20 * w + 20]
        );
        assert_eq!(out[0], 100, "flat background untouched");
        let off = DespeckleParams {
            enabled: false,
            ..p
        };
        assert_eq!(despeckle(&img, w, h, &off), img, "disabled = identity");
    }

    #[test]
    fn despeckle_keeps_large_bright_region() {
        let (w, h) = (40usize, 40usize);
        let mut img = vec![60u8; w * h];
        for y in 10..30 {
            for x in 10..30 {
                img[y * w + x] = 230;
            }
        }
        let p = DespeckleParams {
            enabled: true,
            threshold: 0.15,
            radius: 3,
        };
        let out = despeckle(&img, w, h, &p);
        assert_eq!(
            out[20 * w + 20],
            230,
            "interior of a large bright region is preserved"
        );
    }

    #[test]
    fn flatten_reduces_a_smooth_gradient() {
        let (w, h) = (80usize, 80usize);
        let mut img = vec![0u8; w * h];
        for y in 0..h {
            for x in 0..w {
                img[y * w + x] = (40 + x * 150 / w) as u8;
            }
        }
        let p = FlattenParams {
            enabled: true,
            strength: 1.0,
            radius: 0.33,
        };
        let out = flatten(&img, w, h, &p);
        let in_span = img[40 * w + 60] as i32 - img[40 * w + 20] as i32;
        let out_span = out[40 * w + 60] as i32 - out[40 * w + 20] as i32;
        assert!(
            out_span.abs() < in_span.abs() - 20,
            "smooth gradient flattened: in {in_span} -> out {out_span}"
        );
        let off = FlattenParams {
            enabled: false,
            ..p
        };
        assert_eq!(flatten(&img, w, h, &off), img, "disabled = identity");
    }

    #[test]
    fn fitted_photometric_disabled_is_byte_identical() {
        let frame = (0..64u8).collect::<Vec<_>>();
        let correction = crate::core::types::PhotometricCorrection::default();
        assert_eq!(
            fitted_photometric(&frame, 8, 8, &MlGeometry::default(), &correction, 0),
            frame
        );
    }

    #[test]
    fn fitted_photometric_applies_per_eye_affine_and_smooth_field() {
        let frame = vec![100u8; 9 * 9];
        let correction = crate::core::types::PhotometricCorrection {
            enabled: true,
            affine: [[1.0, 10.0], [0.8, 0.0]],
            field: [
                crate::core::types::PhotometricGainField {
                    horizontal: 0.12,
                    ..Default::default()
                },
                Default::default(),
            ],
            ..Default::default()
        };
        let left = fitted_photometric(&frame, 9, 9, &MlGeometry::default(), &correction, 0);
        let right = fitted_photometric(&frame, 9, 9, &MlGeometry::default(), &correction, 1);
        assert!(
            left[4 * 9] < left[4 * 9 + 8],
            "horizontal field must be smooth"
        );
        assert_eq!(right, vec![80u8; 9 * 9], "right eye keeps its own affine");
    }

    #[test]
    fn fitted_photometric_nonfinite_values_fall_back_safely() {
        let frame = vec![123u8; 16];
        let correction = crate::core::types::PhotometricCorrection {
            enabled: true,
            affine: [[f32::NAN, f32::INFINITY]; 2],
            field: [crate::core::types::PhotometricGainField {
                horizontal: f32::NAN,
                vertical: f32::INFINITY,
                horizontal_curve: f32::NEG_INFINITY,
                vertical_curve: f32::NAN,
            }; 2],
            ..Default::default()
        };
        assert_eq!(
            fitted_photometric(&frame, 4, 4, &MlGeometry::default(), &correction, 0),
            frame
        );
    }

    #[test]
    fn geom_nonfinite_does_not_poison_input() {
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
        assert!(
            out.iter().all(|v| v.is_finite()),
            "non-finite geometry must not NaN the input"
        );
    }

    #[test]
    fn despeckle_edge_case_small_frame() {
        let p = DespeckleParams {
            enabled: true,
            threshold: 0.15,
            radius: 10,
        };

        let img2x2 = vec![100u8; 2 * 2];
        let out = despeckle(&img2x2, 2, 2, &p);
        assert_eq!(out.len(), 4, "2x2 output length");

        let img1x1 = vec![150u8; 1 * 1];
        let out = despeckle(&img1x1, 1, 1, &p);
        assert_eq!(out.len(), 1, "1x1 output length");

        let img100x1 = vec![100u8; 100 * 1];
        let out = despeckle(&img100x1, 100, 1, &p);
        assert_eq!(out.len(), 100, "100x1 output length");
    }

    #[test]
    fn xr5_warp_matches_python_cv2_fixture() {
        let mut img = vec![0u8; SRC * SRC];
        for y in 0..SRC {
            for x in 0..SRC {
                img[y * SRC + x] = ((x + y) / 2) as u8;
            }
        }
        let geom = crate::config::default_ml_geometry("pimax_xr5");
        let out = to_input_stereo_geom(
            &img, SRC as u32, SRC as u32, &img, SRC as u32, SRC as u32, false, true, &geom[0],
            &geom[1],
        );
        let points = [
            (0, 0),
            (0, 99),
            (1, 1),
            (20, 37),
            (50, 50),
            (73, 9),
            (99, 0),
            (99, 99),
        ];
        let expected = [
            [24u8, 110, 25, 61, 80, 50, 57, 135],
            [91u8, 74, 91, 100, 120, 144, 157, 149],
        ];
        for eye in 0..2 {
            for (i, &(y, x)) in points.iter().enumerate() {
                let actual = out[eye * DST * DST + y * DST + x] * 255.0;
                assert!(
                    (actual - expected[eye][i] as f32).abs() <= 1.0,
                    "eye={eye} ({x},{y}) rust={actual:.3} cv2={}",
                    expected[eye][i]
                );
            }
        }
    }
}
