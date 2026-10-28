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

cargo run --features research-synthetic-eye-lab --bin synthetic-eye-lab --offline -- `
  --experiment two-moment-match `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\two-moment-run

cargo run --release --features research-synthetic-eye-lab --bin synthetic-eye-atlas --offline -- `
  --out research-output\moment-atlas-run

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

## Phase 1.2: two-moment aperture isolation

The complete frozen contract is in
[`PHASE1_2_PREREG.md`](PHASE1_2_PREREG.md). It was committed separately before
suite implementation or Phase 1.2 model inference.

`--experiment two-moment-match` first attempts to match both whole-image mean
and population standard deviation by solving bounded skin and sclera levels for
aperture indices 7..40. It requires the same valid set to cover at least 30 of
34 points and contain both semantic reference indices 15 (half open) and 31
(normal open). Preparation is repeated twice, and the fast solver's complete
ordered u8 prediction must match an independent canonical render exactly.

The frozen dual-reference instrument is infeasible in the current renderer:
only indices 18..26 survive both references, so the required common set is 9 of
34. The executable therefore writes `RENDERER_NO_GO`, records the deterministic
exclusion reasons, and exits before reading the model file or running Phase 0.
This is an instrument-feasibility result, not evidence for or against an EyeNet
geometry response. The bounds, references, and thresholds are not relaxed to
manufacture a passing run.

The implementation retains the hypothetical GO-path analysis contract for
review and testing: duplicate bit-exact inference, same-phase baseline,
same-reference replay controls, signed Spearman, true-adjacent concordance, and
the frozen `GO` / `NO_EVIDENCE` / `INCONCLUSIVE` precedence. It is unreachable
under the current renderer gate and must not be invoked by bypassing that gate.

## Phase 1.3A: renderer moment-feasibility atlas

The frozen renderer-only contract is in
[`PHASE1_3A_PREREG.md`](PHASE1_3A_PREREG.md). The atlas is a separate binary
whose CLI has no model or recording argument and whose module tree contains no
EyeNet loader. It maps finite, nested skin/sclera candidate clouds for all 34
apertures and reports exact nearest moment pairs under the legacy,
polarity-preserving, and unrestricted mathematical domains.

Phase 1.3A has no model-facing pass threshold. Its distances and conditioning
are instrument diagnostics only. A later, separately preregistered symmetric
2x2 model study may use the sealed D0 candidate stream after the atlas artifact
is sealed. XR5/VR4 recordings remain excluded until that synthetic
classification is also sealed.

## Phase 1.3: two-target geometry study

The frozen decision and confirmation contract is in
[`PHASE1_3_PREREG.md`](PHASE1_3_PREREG.md), with the pre-inference artifact
sealing clarification in
[`PHASE1_3_AMENDMENT1.md`](PHASE1_3_AMENDMENT1.md), and the frozen complete
renderer-plan identity in
[`PHASE1_3_AMENDMENT2.md`](PHASE1_3_AMENDMENT2.md). Phase 1.3 uses the sealed D0
atlas to select two separated whole-image-mean targets for each retained
gap-four aperture pair. At each target, lower- and higher-aperture images are
matched to the same post-u8 mean and population standard deviation within
0.001 gray per component. Local moment controllability, canonical bytes,
geometry fields, mirroring, and the complete domain-separated plan hash are
gated before model bytes are read.

The global mean axis retains 29 of 30 pairs. Pair 36--40 is a frozen
renderer-only exclusion because its feasible mean range is below the 7-gray
instrument gate. Odd-index pairs form the 15-pair decision set; retained
even-index pairs form the independent 14-pair confirmation set. Both stages
rebuild the complete renderer plan twice and use the same pinned EyeNet model.

Recorded runs use the standalone binary:

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin synthetic-eye-phase13 -- decision `
  --atlas research-output\moment-atlas-49e13f0 `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\phase13-decision

cargo run --release --features research-synthetic-eye-lab `
  --bin synthetic-eye-phase13 -- confirmation `
  --atlas research-output\moment-atlas-49e13f0 `
  --model C:\path\to\00-0000.params_opencl.params `
  --decision research-output\phase13-decision `
  --decision-seal <decision-manifest-sha256> `
  --out research-output\phase13-confirmation
```

Preserve the decision manifest SHA-256 printed by the first command and pass it
unchanged as `--decision-seal`. Confirmation rejects a decision whose current
manifest no longer matches that external seal.

The primary result asks whether both within-target geometry effects are
positive while their target-dependent modulation is flat. It remains a
synthetic renderer-family result. Real XR5 or VR4 frames enter only under a
later transfer-audit preregistration.

## Phase 1.4: XR5 real-recording transfer audit

The frozen protocol is in
[`PHASE1_4_XR5_TRANSFER_PREREG.md`](PHASE1_4_XR5_TRANSFER_PREREG.md). This is a
read-only audit of every completed Dream Air / XR5 EyeWide capture below one
`wide_data/sessions` tree. It uses the fixed production XR5 geometry and the
same sealed EyeNet bytes as Phase 1.3. It does not search geometry or
brightness, fit a model, select sessions, update calibration, or write to the
recording or application configuration.

The recorded audit is deliberately split into two clean release-build commands.
The first command inventories and non-pixel-validates the private recording
tree. It cannot accept or load a model and does not decode PNG pixels:

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin synthetic-eye-xr5-transfer -- seal-input `
  --phase13-decision research-output\phase13-decision-243ab4d `
  --phase13-confirmation research-output\phase13-confirmation-243ab4d `
  --wide-data C:\path\to\wide_data `
  --out research-output\phase14-xr5-input
```

Preserve the printed input-manifest SHA-256. The second command verifies that
external seal and reconstructs the same tree before decoding any frame. It then
performs two independent decode/preprocess passes, opens the fixed model only
after those passes agree, and evaluates two fresh EyeNet instances in opposite
orders:

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin synthetic-eye-xr5-transfer -- analyze `
  --phase13-decision research-output\phase13-decision-243ab4d `
  --phase13-confirmation research-output\phase13-confirmation-243ab4d `
  --wide-data C:\path\to\wide_data `
  --input research-output\phase14-xr5-input `
  --input-seal <input-manifest-sha256> `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\phase14-xr5-analysis
```

Both commands require a clean worktree and a release executable built from the
current commit. Every trusted dependency and the private tree are rechecked
immediately before the staged artifact is published. A source, model, seal, or
recording-tree change prevents publication. Raw PNGs are never copied to the
artifacts.

This audit characterizes the existing XR5 recordings at their saved 30 Hz rate.
Those recordings come from one user/device, and the XR5 geometry was developed
with related captures, so every result is unconditionally labelled
`independence_unproven`. It cannot establish multi-user transfer, 120 Hz blink
dynamics, hour-scale stability, or a production correction. VR4 remains outside
this phase.

## XR5 absolute-geometry development positive control

`xr5-geometry-discovery` exercises the EyeNet-independent initializer in
`src/geometry_discovery.rs`. It extracts only a low-dimensional temporal motion
envelope from the existing `blink_negative` frames after fixed default despeckling and
solves for the absolute crop window and rotation that map that envelope into the
canonical XR5 EyeNet region.
The canonical target consists of aggregate mean/covariance constants only; no eye
image or reconstructable template is embedded.
All XR5 evidence first excludes the fixed inner IR-LED/lens zone. The canonical motion
statistics are defined in that masked coordinate system, and the ML search cannot
reduce the physical inner crop below the validated 40% preset. This hardware exclusion
is not fitted per user.

The same executable also evaluates the second, independent neutral-appearance
hypothesis. Per-block median `neutral_center` frames are reduced to pupil centre,
pupil contrast, eye-aperture axis, anisotropy, and repeatability. The resulting crop
and angle do not inherit the active crop/rotation. Production requires two repeatable
OPEN blocks, binocular plausibility, ordinary EyeNet training guards, and untouched
holdout improvement before this hypothesis can be offered for manual application.
If exactly one eye is inconsistent across the OPEN blocks, the detector may retry only
that eye inside the physical counterpart neighbourhood predicted by the stable eye. The
retry is marked `stereo-recovered`; it does not bypass any repeatability, binocular,
EyeNet, or holdout gate.

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin xr5-geometry-discovery -- `
  --wide-data $env:APPDATA\SRanibro\wide_data
```

This is a development positive control, not an independent transfer result: the
same developer sessions contributed to the aggregate motion and appearance references.
The normal in-app fitter derives both hypotheses from training-only frames. Either can
enter the search only after its own raw-evidence gates, and neither can be applied
without the existing EyeNet guards and untouched holdout improvement.
The command above is read-only and never changes application configuration.

## XR5 photometric replay

`xr5-photometric-replay` is a research-only counterfactual replay of an exported
Safe Geometry Fit recording. It reconstructs the recorded despeckle, flatten, and
per-frame adaptive-brightness affine first. Bounded research interventions are then
applied at one explicit seam: **after captured adaptive brightness and before
geometry**. This avoids reusing an affine learned from the original pixels on a
different pre-normalization image. The recording does not contain the complete
initial history of the stateful brightness controller, so the tool deliberately does
not claim to recompute it under each hypothesis.

That seam is also a validity gate. A replay result is evidence only for a future
implementation in which the adaptive-brightness controller remains entirely upstream
of the new intervention and its captured output is the intervention's input. Moving
the intervention before that controller, or allowing the controller to adapt to the
modified pixels, defines a different feedback system and invalidates this replay; it
must be recorded and tested again at the new seam.

At that post-normalization seam the search covers shared and per-eye affine
brightness/contrast, flattening, and smooth low-frequency gain fields for either eye.
It separately tests shared and per-eye horizontal/vertical crop translation, X/Y
scale, and rotation. A fourth, landmark-independent family tests bounded full-frame
coordinate warps: a zero-mean vertical bow and first-order radial distortion, each at
multiple signed doses and for left-only, right-only, and shared application. Gain-field
coordinates are anchored to the recorded baseline crop but evaluated smoothly over the
complete frame; warp coordinates are normalized over the complete recorded frame. Both
occur before candidate geometry, so moving a crop cannot introduce a hard correction
seam. Probe names and CSV columns use the `postnorm_` prefix, and `manifest.txt` records
the intervention seam and warp bounds. These results compare intervention classes; a
successful intervention does
not by itself prove that the original physical cause belonged to that class. The tool
never changes live or saved SRanibro configuration and the current production pipeline
does not gain a field or warp merely because a probe passes.

Exploration ranks candidates using training frames only. Untouched holdout frames
are report-only and are checked for closed/open reference preservation, slow-close
variation, saturation, monotonicity, gaze retention, and false squeeze. Static phase
labels are recovered from an anchored stable suffix rather than a fixed time trim;
invalid Closed phases and gaze phases without a nearby Neutral anchor are excluded
instead of being silently used as evidence. Exploration writes the baseline diagnostic
and stops before ranking candidates when at least two static phases are invalid or
either split has no valid Closed phase.

"Untouched" here means that code never optimizes on the holdout inside that run. Once
its values have been displayed and a person uses them to revise a candidate family,
threshold, or implementation, that holdout has been statistically consumed for those
later hypotheses. It is not a fresh sealed confirmation set. Final confirmation needs
new evidence whose candidate, thresholds, preprocessing seam, EyeNet byte fingerprint,
and decision rule were externally frozen before the archive was opened. This reuse
rule applies equally to Safe Geometry Fit, the photometric replay, and the landmark /
residual audit.

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin xr5-photometric-replay -- `
  --recording C:\path\to\extracted-recording `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\xr5-photometric-exploration
```

The former hard-coded confirmation probes are intentionally invalidated. They were
discovered when candidate fields were applied before the captured adaptive-brightness
affine; moving to the honest post-normalization seam makes them different hypotheses.
`--confirmation` therefore writes baseline diagnostics and stops with an explanation.
A new post-normalization candidate must first be selected from independent discovery
recordings and frozen in an externally sealed specification before confirmation code is
enabled again. Promotion remains separate: it will require fresh re-wear sessions,
at least 2--3 additional users/units, and fresh sealed safety evidence. Until then
every result is a diagnostic candidate, not a production transformation.

The sealed specification includes the exact EyeNet bytes. Any model-byte change (and
therefore any CRC32/byte-count fingerprint update) creates a new experimental condition
and invalidates earlier confirmation as confirmation for that condition. The old
archive may still be shown as a diagnostic replay, but it cannot promote the new model.

## XR5 landmark and residual audit

The app's separate nine-point research recorder exports versioned evidence without
running Safe Geometry Fit or changing geometry. `xr5-landmark-residual-audit` replays
the recorded baseline through the production preprocessing/inference seam and writes a
read-only audit of gaze-direction residuals and candidate-independent photometric
features. Left and right eyes remain separate, selection uses discovery-train evidence
only, and holdout/confirmation sessions are report-only.

Each repeated relaxed-open target remains visible for 2.20 seconds. After the fixed
reaction trim this leaves at least one valid one-second analysis block for every
non-centre target; the former 1.60-second phase could not do so. The archive also
records per-phase quotas (17 relaxed-open, 30 slow-close/open, and 16 natural-blink
frames). Commanded-target mode and discovery pooling are disabled unless all 56 phase
counts match the observed dataset and every required phase is complete.

Protocol v3 adds one unrecorded rehearsal before each relaxed-open, slow-close/open,
and natural-blink section in both passes. The rehearsal uses the real target and action
timing, but its frames are never sampled or exported. This reduces the first-pass
learning effect without changing the 56 labelled evidence phases or their quotas.

Every one-second analysis block must contain at least 10 frames, span at least 0.90
seconds, and contain no adjacent-frame gap greater than 0.20 seconds. Thus two dense
bursts on opposite sides of a worker/UI stall cannot masquerade as continuous evidence;
the measured 64 ms capture cadence remains comfortably inside the gate.

Real-image pupil and lid landmarks are currently `UNCALIBRATED_QA_ONLY`. They are
reported for detector quality control but cannot enter feature selection, regression,
candidate ranking, or a production decision. Schema-v3-and-later commanded targets are enabled
only when the exact recorder protocol contains every nine-point target in both train
and holdout with the required repeated phases. Model replay parity likewise requires
the supplied model's CRC32 and byte length to match the exact bytes fingerprinted when
the live EyeNet was loaded.

The stability flag is computed once from recorded raw camera pixels, protocol labels,
and the immutable capture geometry. It does not consume Tobii gaze/openness/pupil data
or EyeNet output. Likewise, all 36 global/3x3 photometric predictors are computed
directly from each raw, pre-preprocessing and pre-intervention recorded frame. EyeNet
is used only to construct the response being audited; it cannot leak into stability or
the explanatory features by construction.

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin xr5-landmark-residual-audit -- `
  --recording C:\path\to\extracted-nine-point-recording `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\xr5-landmark-residual-audit
```

Independent re-wear sessions can be appended with repeated
`--confirmation-recording <dir>` arguments and always remain frozen report-only
evidence. The 36-feature exploratory ridge is deliberately stricter: it needs at
least 41 one-second blocks per eye from at least three independent, exact schema-v3
captures with matching EyeNet fingerprints. Supply those training captures with
repeated `--discovery-recording <dir>` arguments. Only their train split is pooled;
every holdout split remains untouched evaluation evidence. Duplicate canonical paths
and byte-identical sample/frame evidence are rejected, but the operator must still
ensure each accepted discovery capture is a genuine independent headset re-wear/session.

Three re-wears of one person test within-person repeatability only; they are not
multi-user generalization evidence. A transfer claim needs the original user plus at
least 2--3 additional users/units and must include the opposite failure laterality
(for example, a right-eye failure if discovery was dominated by the left eye).

The 41-block ridge threshold is only a numerical sanity floor. A typical reachable
pool of 48 one-second blocks for 36 predictors is still statistically fragile even
with ridge regularization. Before a correction family can be frozen, a predeclared
session-wise stability gate must show consistent feature direction/ranking across
leave-one-session-out discovery analyses, every leave-one-session-out training fold
must independently clear the 41-block floor, and the frozen model must beat the
intercept on every held-out session without a laterality reversal. The current audit
reports the exploratory ridge but does not implement that promotion gate, so its
status remains `NO-GO`.

Active capture has no encoder queue that can silently drop frames: labelled stereo
frames are retained in memory and ZIP encoding starts only after capture completes.
Minimizing the window freezes both the protocol clock and the sampling gate, then
resumes without charging hidden time to a phase. Exact per-phase quotas remain the
final integrity check, so an incomplete archive is rejected rather than treated as a
short but valid session.

Optional landmark overlay images contain
biometric eye imagery and therefore require the explicit
`--emit-overlays accepted` flag. The tool never applies a warp; its output is evidence
for choosing and preregistering a later correction family.

### XR5 residual counterfactual replay

`xr5-residual-counterfactual` applies 45 isolated, bounded interventions to the same
nine-target archive and replays the exact fingerprint-matched EyeNet. The intervention
families are post-normalization affine brightness, low-frequency flattening, smooth
spatial gain fields, bounded coordinate warps, and small geometry probes. It never
changes live or saved configuration.

The score combines gaze-direction openness error, the worst instructed target,
left/right asymmetry, and phase-local slow-close fidelity. Train evidence alone orders
the diagnostic table; holdout remains report-only. Candidate selection is disabled
unless all 18 target phases in both splits have enough stable raw-image evidence and
the median train slow-close phase correlation is at least 0.50 for both eyes. A failed
gate still permits causal-class diagnosis, but every row remains non-selectable.

```powershell
cargo run --release --features research-synthetic-eye-lab `
  --bin xr5-residual-counterfactual -- `
  --recording C:\path\to\extracted-nine-point-recording `
  --model C:\path\to\00-0000.params_opencl.params `
  --out research-output\xr5-residual-counterfactual
```

## Limits

The Phase 0--1.3 interventions establish causality only inside this synthetic
renderer family; they do not show that EyeNet uses the same feature on real HMD
camera frames. Phase 1.4 separately characterizes fixed 200x200 XR5 captures and
their short recorded sequences, but it is observational and in-sample. None of
the tools turns a high-scoring pattern into a production filter. Controlled
real-image interventions for lashes, highlights, blur, noise, translation, crop,
or a bounded factorial preprocessing search remain deferred.
