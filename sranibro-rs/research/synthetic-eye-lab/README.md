# SRanibro synthetic eye laboratory

This is an offline, observational research tool for the fixed SRanipal-compatible
`EyeNet`. It cannot be built unless the `research-synthetic-eye-lab` feature is
explicitly enabled. It does not read or update SRanibro configuration, change runtime
tracking, train a model, or write model bytes.

The renderer produces deterministic grayscale eyes directly in EyeNet's canonical
`[2, 100, 100]` CHW input space. It renders at 4x resolution and area-downsamples to
u8; the only tensor conversion is `u8 / 255`. The right channel is an exact horizontal
u8 mirror unless a case explicitly asks for independent stereo specifications.
PNG writing reuses the repository's pre-existing normal `png = 0.17` dependency, which
is already used by eyebrow-calibration capture; this branch did not add it to production.

## Run

Use a new, empty output directory for every run. This prevents results from two suite
versions being mixed.

```powershell
cargo run --features research-synthetic-eye-lab --bin synthetic-eye-lab --offline -- `
  --experiment phase0 `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\phase0-run

cargo run --features research-synthetic-eye-lab --bin synthetic-eye-lab --offline -- `
  --experiment milestone1 `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\milestone1-run

cargo run --features research-synthetic-eye-lab --bin synthetic-eye-lab --offline -- `
  --experiment luminance-match `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\luminance-match-run

python research\synthetic-eye-lab\plot_results.py `
  research-output\milestone1-run
```

Every Milestone 1 run repeats the preregistered Phase 0 anchor test. The additional
suites run only when all three neighboring anchors are finite, structurally eye-like,
and have raw presence at least `0.10`. The production presence gate is recorded
separately and remains the strict `presence > 0.05` comparison.

## Milestone 1 matrix

| Suite | Exact range | Cases | Interpretation |
|---|---:|---:|---|
| aperture geometry | 0.00..1.30 inclusive | 41 | only aperture changes |
| moving-edge control | 0.00..1.30 inclusive | 41 | non-eye-like diagnostic; never used for claims |
| fixed-geometry sclera level | 0.55..0.95 inclusive | 41 | intensity control; not area-matched |
| global brightness offset | -0.25..+0.25 inclusive | 11 | clipping is measured and flagged |
| contrast about image mean | 0.50..1.50 inclusive | 11 | mean-preserving before u8 clipping |
| rotation | -45..+45 degrees inclusive | 19 | canonical renderer rotation |
| stretch grid | scale X/Y in 0.80, 0.90, 1.00, 1.10, 1.20 | 25 | 2D response surface; scale X=1.20 touches the frame and is flagged/excluded |

`results.csv` retains raw floating-point values and their exact bit patterns. It also
records complete left/right specifications, measured geometry and raster aperture,
image covariates (including `frame_truncated`), classification, and the canonical tensor hash. `manifest.json`
records the exact loaded-model content hash, repository state and suite version without
copying the proprietary weights. It separately records the requested experiment and
the suites actually evaluated, so a Milestone 1 request stopped by Phase 0 is explicit.
`summary.json` withholds one-dimensional correlations when fewer than 80% of cases are
recognized, eye-like, unsaturated and free of frame contact; when fewer than five usable
points exist; or when a suite is a two-dimensional grid. PNGs are the exact u8 values
used to construct the tensor. Plotting marks excluded points and produces output
correlations separately per suite rather than pooling heterogeneous interventions.

## Phase 1.1: whole-image-mean matching

`--experiment luminance-match` is a separate source-only experiment whose contract was
frozen by read-only Gate 1 review before implementation. Before model bytes are loaded,
a renderer-only pass:

1. Uses the fixed aperture grid `a_i = 1.30*i/40`.
2. Selects index 31 (`a_ref` approximately 1.0075), the real grid point nearest 1.0.
3. Restricts sclera level to `[0.55, 0.95]`.
4. Finds the largest feasible contiguous range containing the reference whose endpoint
   mean intervals have a common intersection.
5. Solves sclera level using exactly 24 quantized-mean bisection iterations and accepts
   only error at most `0.5/255`.

Eye-likeness, finite covariates, saturation and frame contact are checked during this
renderer pass. Failure makes the luminance suite NO-GO before any EyeNet inference.
Bounds are never widened after seeing model output.

The run contains four signed comparisons:

| ID | Geometry | Photometric treatment |
|---|---|---|
| A | aperture varies normally | default sclera 0.78 |
| B | aperture varies | sclera compensates to hold whole-image mean constant |
| C | fixed at `a_ref` | replays B's exact sclera sequence |
| D | fixed at `a_ref` | sclera reproduces A's original whole-image mean trajectory |

B is not a pure-geometry experiment: aperture and compensating sclera are perfectly
coupled. A positive B response paired with negative C is evidence that geometry/visible
area overcame the deliberately opposing sclera contribution inside this renderer
family. Similar negative B/C trends support a photometric explanation; near-zero B with
negative C is cancellation and inconclusive. A and D are compared only over D's matched
points. No sign pattern is treated as proof, and whole-image mean matching does not
control bright-pixel mass, histogram, contrast, edge energy or visible sclera area.

The manifest records the complete preregistration, prior signed directions, feasible
range, solver contract and every unmatched D target. CSV rows include the target and
achieved mean, exact solver brackets, selected sclera, error and bound-proximity flag.
`summary.json` reports signed correlations, output ranges, unmatched coverage, maximum
and median mean error, and bound-pinned counts. For luminance runs, A's reported
correlation and span are restricted to the exact B/C/D matched indices.
`luminance_analysis.json` records paired A-vs-D effect sizes and range overlap, states
whether the observed B/C signs matched the preregistered rubric, and explicitly labels
any conservative post-hoc demotion caused by the absence of a preregistered residual
effect-size threshold. It also records the infeasible near-closed aperture range rather
than generalising the matched result into that untested regime.

## Limits

These results establish causality only inside this synthetic renderer family. They do
not show that EyeNet uses the same feature on real HMD camera frames, and the tool never
turns a high-scoring pattern into a production filter. Lashes, highlights, blur, noise,
translation/cropping, 200x200 XR5 preprocessing, temporal sequences and bounded
factorial search are deliberately deferred.
