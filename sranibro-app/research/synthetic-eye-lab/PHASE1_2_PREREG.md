# Phase 1.2 preregistration: two-moment aperture isolation

Status: frozen before Phase 1.2 implementation or inference.

This document defines the single decisional analysis for Phase 1.2 of the
source-only synthetic EyeNet laboratory. The commit containing this document is
the preregistration commit. The implementation and every Phase 1.2 result
manifest must record that commit hash.

Phase 1.1 established strong whole-image-mean sensitivity in this synthetic
renderer family, but it did not exclude a remaining geometry response. Phase
1.2 asks a narrower question:

> Does EyeNet openness retain a positive monotone dependence on rendered eyelid
> aperture after the mean and population standard deviation of the exact 100x100
> u8 model input are matched to two semantic reference apertures?

A positive result means only that the aperture response is robust to these two
global moment controls in this renderer family. Higher image moments, spatial
photometrics, and renderer realism remain possible confounds. This is not a
claim about real-camera causality or product readiness.

## Frozen aperture universe and references

The grid is unchanged:

```text
a_i = 1.30 * i / 40, i = 0..40
```

The frozen Phase 1.2 universe is exactly the 34 original indices `7..40`
inclusive. Indices `0..6` remain outside the experiment; they may not be added
after results are seen.

Two model-independent semantic references are required:

| Meaning | Target | Nearest real index | Actual aperture |
|---|---:|---:|---:|
| half open | 0.5 | 15 | 0.4875 |
| normal open | 1.0 | 31 | 1.0075 |

Nearest-grid selection uses minimum absolute distance and a lower-index tie.
The renderer constant `normal_opening_px = 23` is unrelated to the aperture
index and must not be used as one.

## Decisional suites

All images use the canonical 100x100 renderer at 4x supersampling. The left eye
is rendered normally and the right eye is its anatomical pixel mirror.

`S3 baseline` renders every frozen aperture with the default photometric
parameters:

```text
skin = 0.46
sclera = 0.78
```

For each reference `r` in `{15, 31}`:

- `S1_r two-moment match` changes aperture normally and solves bounded skin and
  sclera levels so the canonical u8 image mean and population standard deviation
  match the canonical default render at `r`.
- `S2_r fixed-geometry replay` freezes aperture geometry at `r` and replays the
  exact solved `(skin, sclera)` sequence from `S1_r`. This is the artifact and
  compensation-trajectory control paired with `S1_r`.

The solver domain is fixed before inference:

```text
skin    in [0.30, 0.60]
sclera in [0.65, 0.95]
```

The rectangular domain preserves intensity polarity because its minimum
sclera is brighter than its maximum skin. Real recordings, EyeNet outputs, and
solver feasibility may not change these bounds.

## Canonical byte contract

The canonical renderer is the sole authority for model input bytes. For every
positive normalized pixel level it uses the existing Rust operation:

```text
(clamp(level, 0, 1) * 255).round() as u8
```

No independent rounding convention may replace this path.

The fast solver may precompute, for each output pixel, the 16 supersample final
component assignments after renderer layer precedence. A search bin contains:

- the fixed-level supersample contribution;
- the skin supersample count;
- the sclera supersample count;
- the number of pixels sharing that coefficient tuple.

Bins may accelerate moment evaluation. An ordered per-pixel coefficient table
must separately reconstruct the complete predicted 100x100 u8 byte array for
the selected candidate. That byte array must exactly match an independent
canonical render before the candidate is usable.

Mean and population standard deviation are evaluated in gray-level units
`0..255`, using f64 accumulation over the quantized u8 pixels.

## Deterministic solver

For target moments `(mean_t, std_t)`, the objective is:

```text
F(skin, sclera) = (mean - mean_t)^2 + (std - std_t)^2
```

The search is finite and deterministic:

1. Level 0 evaluates an inclusive `65 x 65` grid over both full bounds. The exact
   default pair `(0.46, 0.78)` is also an explicit candidate.
2. Let the prior axis steps be `d_skin` and `d_sclera`. For each of three
   refinement levels, evaluate a `17 x 17` grid around the previous winner:

   ```text
   skin    = best_skin    + k_skin    * d_skin    / 4
   sclera = best_sclera + k_sclera * d_sclera / 4
   k_skin, k_sclera in -8..=8 (both endpoints inclusive)
   ```

   Out-of-bound candidates are discarded. The previous winner and exact default
   pair are also candidates. After each refinement, both axis steps are divided
   by four.
3. Candidates are ordered by: lowest objective; lowest squared distance from
   defaults after normalizing each axis by its full bound width; lower skin;
   lower sclera. All comparisons use f64 total ordering.
4. The reference index must select the exact default pair. Because it has zero
   objective there and zero default distance, any failure is a renderer NO-GO.
5. Every selected point is independently canonical-rendered and byte-compared
   with the fast prediction.

The full preparation is executed twice before inference. Solved f64 bits, case
definitions, predicted hashes, and canonical hashes must be identical.

## Renderer-only gates

Renderer checks run before the model is loaded. They are divided into global
failures and per-index exclusions so no implementation judgment can alter the
denominator.

The decisional renderer predicates are pinned to the existing implementation in
the preregistration tree and may not be redefined for Phase 1.2:

- eye-like means `SyntheticEyeSpec::validate()` in `renderer.rs`;
- frame contact means `ImageCovariates::frame_truncated` as produced by
  `renderer::frame_truncated()`;
- saturation fraction means the number of canonical u8 pixels exactly equal to
  either `0` or `255`, divided by the 10,000 image pixels, as implemented by
  `renderer::pixel_covariates()`;
- finite covariates means that `mean`, `stddev`, `edge_energy`,
  `saturation_fraction`, `visible_area_fraction`,
  `measured_aperture_geometry`, and `measured_aperture_raster` are all finite,
  matching the existing Phase 1.1 renderer guard.

### Global `RENDERER_NO_GO`

The entire experiment stops without inference if any condition fails:

- repeated preparation differs in any solved f64 bit, case definition,
  predicted hash, or canonical hash;
- any selected fast-path 100x100 byte array differs from its independent
  canonical render;
- an `S1` geometry or raster aperture field differs bitwise from `S3` at the
  same index;
- an `S2` geometry or raster aperture field differs bitwise from its fixed
  reference;
- in `S3`, both named scalars `measured_aperture_geometry` and
  `measured_aperture_raster` do not each strictly increase at every consecutive
  original index in `7..40`;
- the 34 canonical `S3` image hashes are not all distinct;
- after the per-index exclusions below, an `S1_r` saturation-fraction range
  exceeds `0.005` (0.5 percentage points). With zero or one still-valid point,
  this range check passes vacuously; the final intersection gate will still
  fail;
- the frozen common valid-index intersection across `S3`, both `S1` suites, and
  both `S2` suites has fewer than 30 of 34 points or omits index 15 or 31.

There is no iterative exclusion or gate recomputation.

### Per-index exclusions

An original index is marked renderer-invalid if any paired relevant case in
`S3`, either `S1`, or either `S2` meets any condition:

- for either solved parameter, distance to the nearer bound is less than or
  equal to `0.01 * (upper - lower)`;
- an `S1` absolute mean residual or absolute standard-deviation residual exceeds
  `0.25` gray level;
- the image is not eye-like, has a nonfinite covariate, contacts the frame, or
  has saturation fraction above `0.01`;
- the solved `S1_r` skin or sclera trajectory has a discontinuous incoming
  step.

Trajectory discontinuity is computed independently for skin and sclera for
each `S1_r` over all 33 original consecutive absolute step magnitudes
`abs(p(i+1) - p(i))` before any exclusion. Exact zero magnitudes are removed. If
none remain, the trajectory passes. Otherwise the median is computed once over
the remaining absolute magnitudes; for an even count it is the arithmetic mean
of the two middle sorted values. An absolute step magnitude strictly greater
than five times that median invalidates only its higher-index endpoint. The
median is never recomputed.
`S3` has constant default parameters and therefore passes trivially. `S2`
replays the paired `S1` trajectory and is exempt from duplicate trajectory
invalidation.

The union of all per-index failures is taken once, followed by the saturation
range checks and the common-intersection global gate.

## Inference and metrics

Only the frozen common renderer-valid index set is inferred. Every case is
inferred twice in the same invocation. Every output f32 bit must match between
the two passes.

For each eye and reference, compute on the common set:

- signed Spearman correlation between original aperture index and openness,
  using average ranks for exact-value ties;
- openness span `max - min`;
- span ratio relative to the same-eye `S3` span;
- true-adjacent directional concordance.

For concordance, only original pairs `(i, i+1)` with both indices in the common
set are candidates. Gaps are never bridged. An openness delta above `+0.001` is
concordant, below `-0.001` is discordant, and absolute delta at or below `0.001`
is ineligible. All counts are reported. The eligible-pair minimum applies only
to a `GO`; flatness must remain capable of producing `NO_EVIDENCE`.

For `S1`, an all-bit-identical series has `rho = None`; otherwise exact-value
average-rank Spearman is used.

For `S2`, a span at or below `0.001` has
`rho = None_due_deadband`. Above that span, exact-value average-rank Spearman is
used. This prevents sub-deadband control jitter from becoming a perfect rank
correlation.

## Frozen classification order

Classification is a total ordered procedure:

1. `RENDERER_NO_GO`: any global renderer gate fails. No inference occurs.
2. `INCONCLUSIVE_ARTIFACT`: duplicate inference differs in any output bit.
3. `INCONCLUSIVE_INSENSITIVE`: either eye's `S3` baseline span is below `0.10`.
   No replay ratio is evaluated before this denominator floor.
4. `INCONCLUSIVE_ARTIFACT`: any same-eye, same-reference `S2` replay is not
   generically clean. Generic replay cleanliness requires both:
   - `rho` is `None_due_deadband` or `abs(rho) <= 0.30`;
   - `S2 span / S3 span <= 0.10`.
5. Apply the endpoint rubric below.

### `GO`

Every eye at both references must satisfy all conditions:

- `S1 rho >= +0.70`;
- `S1 span >= 0.05`;
- `S1 span / same-eye S3 span >= 0.20`;
- at least 20 eligible true-adjacent pairs;
- directional concordance at least `0.90`;
- the same-eye, same-reference `S2 span <= 0.25 * S1 span`, in addition to
  generic replay cleanliness.

### `NO_EVIDENCE`

Every eye at both references must satisfy all conditions:

- `S1 rho` is `None` because all outputs are bit-identical, or
  `abs(S1 rho) <= 0.30`;
- `S1 span < 0.05`;
- `S1 span / same-eye S3 span <= 0.10`;
- generic replay cleanliness.

There is no eligible-pair minimum for `NO_EVIDENCE`.

### `INCONCLUSIVE`

Every remaining valid result is `INCONCLUSIVE`. Descriptive flags do not change
the class:

- `REFERENCE_DEPENDENT` when the two references reach different provisional
  endpoints;
- `EYE_ASYMMETRIC` when left and right reach different provisional endpoints;
- `CONTROL_LARGE` when generic replay cleanliness passes but the stricter paired
  `S2 span <= 0.25 * S1 span` GO control fails for an otherwise GO-like result.

`CONTROL_LARGE` is reachable only when the relevant `S1/S3` span ratio lies in
`[0.20, 0.40)`. At ratios at or above `0.40`, the generic `0.10 * S3` control
already implies the stricter paired control.

The thresholds are deterministic unit-scale relevance anchors, not p-values:
baseline span `0.10`, matched span `0.05`, span ratios `0.20/0.10`, correlations
`0.70/0.30`, delta deadband `0.001`, and concordance `0.90`. None is derived from
the observed Phase 1.1 model statistics, and no Phase 1.1 span is used as a
denominator.

## Execution and anti-tailoring rules

- This entire document and the analysis contract are committed before Phase 1.2
  suite implementation or rendering.
- The implementation commit follows the preregistration commit. Its manifest
  records both hashes and requires a clean worktree.
- Renderer-only failures may be repaired only before inference and only through
  a new explicit preregistration-amendment commit followed by another design
  review.
- Once any Phase 1.2 model output exists, no renderer, solver, threshold,
  reference, index set, or classification rule may change. A rescue experiment
  is Phase 1.3 with a new preregistration.
- Exactly one committed-source invocation emits the synthetic classification.
- Post-hoc metrics may be labelled exploratory but may not alter the class.

Planned follow-up by class:

- `GO`: design a fresh Phase 1.3 control for higher histogram moments and spatial
  photometrics.
- `NO_EVIDENCE`: do not claim geometry sensitivity under the two-moment control;
  validate whether the renderer manipulation remains realistic before product
  conclusions.
- `INCONCLUSIVE*`: resolve only the named limitation under a new preregistration.
- `RENDERER_NO_GO`: report instrument infeasibility without loading EyeNet.

## XR5 recording audit

The authorized XR5 recordings enter only after the synthetic classification is
written. They cannot alter it, tune the renderer, fit parameters, choose bounds,
or set thresholds.

The audit uses each session separately and freezes the committed default XR5
preprocessing path:

- `default_ml_geometry("pimax_xr5")`;
- default despeckle settings;
- default disabled flattening;
- default disabled adaptive brightness normalization;
- the same stereo EyeNet adapter and fixed model used by the synthetic run.

Raw-frame mean/std and final 100x100 model-input mean/std are both reported.
The audit never reads mutable per-user geometry or brightness settings.

Within each session, phase, and side, filenames are sorted by their numeric
sequence. For `n` frames, discard `floor(0.20*n)` from each end and retain the
central remainder. All retained frames enter medians and quantiles. Starting
at the first retained frame, every 12th retained frame enters within-label
brightness/contrast association diagnostics. No p-values are computed.

The only preregistered directional annotations, evaluated per session and eye,
are:

- closed median openness below neutral-center median;
- wide-soft median openness above neutral-center median;
- wide-max median openness above neutral-center median;
- during left and right wink phases, the winked-eye median openness below the
  simultaneously retained non-winked-eye median.

Gaze-labelled strata are reported separately and never pooled into an openness
claim. Phase labels are not treated as continuous aperture ground truth.

The two available sessions come from one user and one device. The audit supports
only a transfer annotation for those recordings; it cannot support a
multi-user, multi-device, or real-camera causal claim.

## Explicit deferrals

- near-closed aperture indices `0..6`;
- histogram/CDF/LUT matching;
- block scrambling;
- Wide and Squeeze model outputs as decision variables;
- gaze modelling;
- real-recording fitting or threshold selection;
- per-eye asymmetric synthetic suites;
- multi-user or multi-device generalization.
