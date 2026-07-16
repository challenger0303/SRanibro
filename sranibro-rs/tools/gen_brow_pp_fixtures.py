#!/usr/bin/env python3
"""Generate golden fixtures for the brow-preprocessing parity test.

Replicates vr_eyebrow/dataset.py's INFERENCE preprocessing path (is_train=False,
1-channel) on a set of deterministic synthetic raw eye grayscale images, and dumps,
per image, the raw bytes + the expected 64x64 f32 z-scored array. The Rust test in
src/ml/preprocess.rs (fn brow_pp_parity) loads these and asserts brow_input matches.

The exact reference path (dataset.py __getitem__, is_train=False):
    Image.open(...).convert('L')          # here: synthetic uint8 (h, w)
    -> crop_roi:  box=(int(0.15*w), 0, int(0.85*w), int(0.4*h))   # PIL crop
    -> TF.resize((64,64))  == PIL Image.resize((64,64), BILINEAR) (antialiased)
    -> _pil_to_zscored_tensor:
           arr = np.array(pil, uint8)
           arr = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(arr)
           arr = arr.astype(float32)
           arr = (arr - arr.mean()) / max(arr.std(), 1.0)

The right-eye case: vr_eyebrow flips the full right-eye frame at capture time so both
eyes arrive left-canonical. We emit a "flip" fixture: the raw image is stored UNflipped
(as brow_input receives it), and the expected output is computed on the h-flipped frame
(what the model was trained on). brow_input(..., flip_h=True) must reproduce it.

Run with the vr_eyebrow CPU venv:
    <vr_eyebrow>/venv_cpu/Scripts/python.exe tools/gen_brow_pp_fixtures.py --out <dir>
Then:
    BROW_PP_FIXTURE_DIR=<dir> cargo test --lib brow_pp_parity -- --nocapture
"""
import argparse
import json
import os
import struct

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def crop_roi(pil):
    w, h = pil.size
    box = (int(w * 0.15), 0, int(w * 0.85), int(h * 0.4))
    return pil.crop(box)


def pil_to_zscored(pil_gray):
    arr = np.array(pil_gray, dtype=np.uint8)
    arr = _CLAHE.apply(arr)
    arr = arr.astype(np.float32)
    m = float(arr.mean())
    s = float(arr.std())
    return (arr - m) / max(s, 1.0)


def preprocess(raw_u8, flip_h):
    """raw_u8: (h, w) uint8 array as brow_input receives it. Returns (64,64) f32."""
    frame = raw_u8
    if flip_h:
        # vr_eyebrow flips the FULL frame at capture; both eyes then arrive
        # left-canonical. Replicate that here on the full frame BEFORE crop/resize.
        frame = np.ascontiguousarray(frame[:, ::-1])
    pil = Image.fromarray(frame, "L")
    pil = crop_roi(pil)
    pil = TF.resize(pil, (64, 64))
    return pil_to_zscored(pil)


def make_images():
    """Deterministic synthetic raw eye grayscale images. Returns list of
    (name, w, h, np.uint8 (h,w), flip_h)."""
    out = []

    def grad_noise(w, h, seed, freq=7.0):
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        base = 128.0 + 90.0 * np.sin(xx / w * freq) * np.cos(yy / h * (freq * 0.6))
        # brow-like dark horizontal band in the top ROI region
        band = 60.0 * np.exp(-((yy - h * 0.18) ** 2) / (2 * (h * 0.05) ** 2))
        rng = np.random.default_rng(seed)
        noise = rng.integers(-18, 19, size=(h, w))
        img = base - band + noise
        return np.clip(img, 0, 255).astype(np.uint8)

    # A few sizes, incl. the required 200x200 and 640x400.
    out.append(("sq200", 200, 200, grad_noise(200, 200, 11), False))
    out.append(("varjo640x400", 640, 400, grad_noise(640, 400, 22, freq=9.0), False))
    out.append(("small120x90", 120, 90, grad_noise(120, 90, 33, freq=5.0), False))
    out.append(("odd201x133", 201, 133, grad_noise(201, 133, 44, freq=8.0), False))
    # Right-eye flip exerciser: strongly left-right asymmetric so flip matters.
    asym = grad_noise(200, 200, 55, freq=6.0).astype(np.int32)
    asym[:, :100] = np.clip(asym[:, :100] - 40, 0, 255)
    asym[:, 100:] = np.clip(asym[:, 100:] + 40, 0, 255)
    out.append(("flip200", 200, 200, asym.astype(np.uint8), True))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="fixture output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = []
    for name, w, h, img, flip_h in make_images():
        assert img.shape == (h, w)
        expected = preprocess(img, flip_h).astype("<f4")
        assert expected.shape == (64, 64)
        # raw: w,h as u32 LE then h*w bytes (row-major, unflipped — as brow_input gets it)
        raw_path = os.path.join(args.out, f"{name}.raw")
        with open(raw_path, "wb") as f:
            f.write(struct.pack("<II", w, h))
            f.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        exp_path = os.path.join(args.out, f"{name}.f32")
        expected.reshape(-1).tofile(exp_path)
        manifest.append({"name": name, "w": w, "h": h, "flip_h": bool(flip_h)})

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} fixtures -> {args.out}")
    for m in manifest:
        print("  ", m)


if __name__ == "__main__":
    main()
