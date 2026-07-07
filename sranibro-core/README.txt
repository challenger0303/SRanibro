sranibro-core
=============

The HMD-agnostic eye-tracking post-processor and eyelid-model front-end from
SRanibro (https://github.com/challenger0303/SRanibro), as a standalone,
MIT-licensed Rust library.

SRanibro is a closed-source, binary-only app that bridges Tobii-based VR
headsets (Pimax Crystal / Crystal Super, StarVR One, Varjo) to VRCFaceTracking.
This crate is the part of it that is device-independent and original:
everything that turns the eyelid model's raw output into stable, calibrated
tracking. The device-access layer (camera capture, gaze decode, connection
handling) stays in the app.


WHAT'S IN HERE
--------------

core -- the post-processor (core::eye_state::SRanipalState). Feed it the per-eye
model output [[f32; 5]; 2] and a GazeSample; get back smoothed, calibrated
EyeResults. This is where the interesting work lives:

  - Band-gated per-eye baseline learning plus a "stuck-wide" breaker, so no
    expression can corrupt the auto-calibration and every bad state self-recovers,
    bounded.
  - Episode-based, persistent blink-depth calibration (slow blinks teach each
    eye's true closed point) and an L/R mid-close curve equalizer.
  - A per-eye-normalized fast-blink detector (velocity plus cumulative-fall
    evidence, rise-off-bottom release), a bilateral blink assist, and a gaze yoke.
  - Wide / squeeze with hysteresis and bilateral gating; asymmetric-EMA and
    optional Kalman smoothing.
  - Extensive unit tests, including golden-replay-style scenarios.

ml -- image preprocessing (despeckle, illumination flatten, adaptive brightness
normalization, per-eye geometry warp) and the eyelid-model inference (im2col +
GEMM CNN forward pass), plus an occlusion-sensitivity heatmap.


WHAT'S NOT IN HERE
------------------

No model weights and no proprietary assets are bundled or distributed. The
eyelid model loads at runtime from a SRanipal install you supply, from a source
you are authorized to use under its license. Camera / device access is not part
of this crate.


STATUS
------

Extracted from the app as-is; the public API is not yet stabilized. Useful as a
reference and as the algorithmic core others can build on.


LICENSE
-------

MIT -- see LICENSE.
