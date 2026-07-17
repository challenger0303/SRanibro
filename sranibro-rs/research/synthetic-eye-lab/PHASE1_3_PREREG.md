# Phase 1.3 preregistration: two-target moment-controlled geometry study

Status: frozen before Phase 1.3 implementation, renderer preparation, or model
inference.

This document defines the primary decision and its held-out confirmation for
Phase 1.3 of the source-only synthetic EyeNet laboratory. The commit containing
this document is the preregistration commit. The implementation and every
Phase 1.3 result manifest must record that commit hash.

Phase 1.2 stopped at its renderer gate and never loaded EyeNet. Phase 1.3A then
sealed a finite atlas of the committed renderer's post-u8 whole-image mean and
population-standard-deviation surface. Phase 1.3 uses that atlas to ask:

> Across two separated common moment targets, does EyeNet openness retain a
> positive effect of a fixed four-index increase in rendered eyelid aperture?

Each geometry contrast is made between images matched to the same target mean
and population standard deviation. A positive result therefore supports an
aperture response that survives these two global controls in this synthetic
renderer family. It does not establish real-camera causality, higher-moment or
spatial invariance, model quality on an HMD, or product readiness.

## Frozen inputs

The only renderer-search input is the canonical Phase 1.3A atlas at:

```text
repository commit
49e13f0eb2b78f84a387de8a46e7309257c9304e

atlas preregistration commit
c92dbb2411c13d2f055ee7c1a67ee2b956d1e1a1

manifest.json
13d67e09915faa22f89f482ae3960703d1b610e519e5868ddcf884df0542e347

candidate_stream.bin
4eb662658c7997d37d53acc6daa8af9045e9ae2386784e39f3d37b4145313139

aperture_summaries.json
ed8a7c468c301124bd26de9c56baa6755c9b4d82b728e4cfcf39e2bc820ad2f1

pair_summaries.json
d7ce595f354b8dacd5f19c04d1a0f04f24aadad585fc0c580de29f334efd1480

canonical_checks.json
5c67d57e6321f00f7e1e6e4e8a98de5d0788670e3afb176421ecc4afb135453d

renderer source
9fdeca8c45fa6c56d7721e0a2a2d10e1b9c19799ff528ccc5d059f7031c056bd

atlas implementation source
d3cfd3a4669d30663bcbd4072e2a8584f979e9e5e5b6c789864f424dc1f1bdc8

EyeNet model bytes
bac8013e0423068924f190a1de44afd5e1dd0c7c10d1d394926e46fc1b075ded
length = 51,423,934 bytes
```

The implementation must validate every complete hash, the atlas manifest
identity, a clean atlas repository state, release profile, candidate count
`1,125,876`, pair count `1,683`, model/recording false-or-null fields, binary
magic `SRATL3A1`, schema version `1`, 28-byte little-endian record size, and D0
membership bit. Unknown versions, missing files, extra candidate records, or
identity mismatches are renderer failures. Atlas summaries are diagnostics;
`candidate_stream.bin` is the authoritative finite candidate population.

The 24-byte candidate header layout is fixed as magic bytes `[0..8)`, u32
version `[8..12)`, u32 record length `[12..16)`, and u64 record count
`[16..24)`. Each 28-byte record is aperture u8 at offset 0, membership u8 at
offset 1, zero reserved u16 at offset 2, skin f32 bits at offset 4, sclera f32
bits at offset 8, mean f64 bits at offset 12, and stddev f64 bits at offset 20.
All integers are little-endian. D0 is membership mask `0x01`; a record is
eligible when that bit is set. Aperture order, level total order, known
membership bits, reserved bytes, finite moments, and exact per-aperture and
global counts are validated while parsing.

The model path is supplied only after renderer preparation has passed. Model
bytes are read-only and must have the exact frozen byte length and SHA-256
identity above
before they may be parsed or inferred. This is the same model identity used by
the earlier sealed synthetic runs. Conclusions apply only to those exact model
bytes. The identity is repeated in the decision artifact, and confirmation must
use the identical bytes. Neither stage writes model bytes.

## Isolation from recordings and production state

The Phase 1.3 binary is feature-gated and research-only. Its decision CLI
accepts only:

```text
synthetic-eye-phase13 decision \
  --atlas <sealed-atlas-directory> \
  --model <EyeNet-params-file> \
  --out <new-directory>
```

Its confirmation CLI accepts only:

```text
synthetic-eye-phase13 confirmation \
  --atlas <sealed-atlas-directory> \
  --model <same-EyeNet-params-file> \
  --decision <sealed-decision-directory> \
  --out <new-directory>
```

Recording paths and unknown arguments are errors. The tool cannot read or
write SRanibro settings, calibration, production geometry, or production
brightness controls. XR5, VR4, and every other real recording remain unread
until both synthetic stages are sealed. Recording content, labels, moments,
and model outputs may not select an axis, target, recipe, pair, threshold, or
classification rule.

## Aperture universe and split

The aperture grid is unchanged:

```text
a_i = 1.30 * i / 40, i = 7..=40
```

The candidate factorial unit is every gap-four pair:

```text
(i, i + 4), i = 7..=36
```

There are 30 candidate pairs. A renderer-only audit of the sealed atlas, using
the boundary, conditioning, and residual rules below without reading a model,
found that one coherent mean target axis retains 29 pairs. Pair `(36,40)` has a
mean-axis feasible range of only `1.043300000000016` gray and is the single
frozen renderer exclusion. Its range must be recomputed and remain below the
`7.0` gate; it may not be restored. No further pair may be excluded.

The 29 decisional pairs are split before preparation:

- decision: the 15 pairs whose lower index is odd;
- confirmation: the 14 retained pairs whose lower index is even.

The aperture indices used by the two stages are disjoint. Pairs within a stage
form overlapping gap-four chains and are a fixed synthetic population, not
independent random samples. No p-value, confidence interval, or statistical
power claim is permitted.

## Authoritative D0 population

Only D0 candidates from the sealed stream are eligible:

```text
skin   = 0.30 + 0.30*k/128
sclera = 0.65 + 0.30*l/128
k,l in 0..=128
```

plus the exact committed default `(0.46f32, 0.78f32)`. Candidate identity is
the pair of applied f32 bits. D1 and D2 candidates are never eligible.

Let:

```text
h = 0.30 / 128
m = 2h = 0.0046875
```

An eligible recipe must have all four applied-level margins greater than or
equal to `m`:

```text
skin - 0.30
0.60 - skin
sclera - 0.65
0.95 - sclera
```

The comparisons use the f64 values of the applied f32 levels and the exact f64
constant `0.30 / 128 * 2`, without an added epsilon.

## Conditioning gate

Every recipe used to construct or render a target must have a finite central
secant at one D0 step in both level directions. Probe levels are formed by the
same committed operation used by Phase 1.3A: add or subtract `h`, cast each
probe level to f32, then evaluate post-u8 moments. The actual applied-f32 probe
distance is the derivative denominator.

For moment vector `(mean, stddev)`, the two secant columns form the 2x2 matrix
`J`. Its singular values use the fixed closed form:

```text
r = hypot(J00 + J11, J10 - J01)
s = hypot(J00 - J11, J01 + J10)
sigma_max = (r + s) / 2
sigma_min = abs(r - s) / 2
kappa = sigma_max / sigma_min
```

The probes, matrix, singular values, and kappa must be finite;
`sigma_min > 0`; and `kappa <= 20`. Range width never substitutes for this
local controllability gate.

## Common-target construction

For each gap-four pair, enumerate every ordered D0 cross-product recipe pair
`(p_low, p_high)` whose two recipes pass the boundary and conditioning gates.
Using the candidate stream's f64 moments and this exact operation order, form:

```text
T_mean = (p_low.mean + p_high.mean) * 0.5
T_std  = (p_low.std  + p_high.std)  * 0.5
```

The pair is a feasible common target only when all four endpoint residuals are
inclusive `<= 0.001` gray:

```text
abs(p_low.mean  - T_mean)
abs(p_low.std   - T_std)
abs(p_high.mean - T_mean)
abs(p_high.std  - T_std)
```

An index, range tree, or other acceleration is allowed only if a reduced-grid
test proves exact equality with exhaustive enumeration, including equality at
the threshold and all tie breaks. Equality is never pruned.

Targets with bit-identical `(T_mean, T_std)` are deduplicated. Their
representative endpoint pair is selected by this total order:

1. lower maximum of the four absolute endpoint residuals;
2. lower sum of their squared residuals;
3. lower sum of the two recipes' D0-normalized squared default distances;
4. lower `p_low.skin`, `p_low.sclera`, `p_high.skin`, then
   `p_high.sclera`, all by f32 total order.

The normalized default distance for one recipe uses the exact applied f32
defaults raised to f64:

```text
ds = (skin as f64 - 0.46f32 as f64) / 0.30f64
dc = (sclera as f64 - 0.78f32 as f64) / 0.30f64
distance = ds*ds + dc*dc
```

Every residual, distance, and square is f64. Sums are evaluated left to right
in the component order written in this document; fused or reassociated sums are
not allowed. Four-component target residuals are ordered lower mean, lower
stddev, higher mean, higher stddev. Eight-component target-pair sums evaluate
those four components for target 0 followed by target 1. Four-recipe default
distance sums are ordered target-0 lower, target-0 higher, target-1 lower,
target-1 higher.

All endpoints participating in the deduplicated feasible set are independently
canonical-rendered before target ranges are used. Fast and canonical bytes,
SHA-256, mean bits, and standard-deviation bits must agree. Every image must be
eye-like, have finite canonical covariates, avoid frame contact, and have
exactly zero saturation fraction. A parity failure is a global renderer NO-GO;
the failing candidate is not silently removed and selection is not repeated.

## One global moment axis

The global target axis is frozen as whole-image `mean`. It is shared by all 29
retained decision and confirmation pairs. Per-pair axis switching is forbidden
because it would give the target main effect a different meaning across pairs.

This choice was made from the sealed renderer cloud before implementation or
model inference. With every gate in this document, mean provides range `>= 7`
for 29 of 30 pairs; global stddev provides it for only 26. For auditability, the
implementation still computes and records feasible mean and stddev ranges for
all 30 candidate pairs. It must reproduce exactly one mean-range failure,
`(36,40)`, and no retained pair may fail.

For every retained pair on the chosen axis, let:

```text
x_min = minimum feasible target coordinate
x_max = maximum feasible target coordinate
R = x_max - x_min
q25 = x_min + 0.25 * R
q75 = x_min + 0.75 * R
```

Every retained pair must have at least two unique feasible targets and
`R >= 7.0` gray.

## Two-target selection

For a pair, enumerate all ordered distinct feasible targets `(T0, T1)` with
chosen-axis coordinates `x0 < x1` and:

```text
x1 - x0 >= 4.0 gray
x1 - x0 >= 0.25 * R
```

Select exactly one ordered target pair by this total order:

1. lower `max(abs(x0 - q25), abs(x1 - q75))`;
2. lower `abs(x0 - q25) + abs(x1 - q75)`;
3. lower absolute difference between the targets on the non-selected axis;
4. lower maximum across all eight endpoint-to-target component residuals;
5. lower sum of all eight squared component residuals;
6. lower summed D0-normalized default distance of all four endpoint recipes;
7. lower `T0` mean bits, stddev bits, endpoint recipe f32 total-order keys,
   followed by the corresponding `T1` keys.

The two target bit pairs must differ. If no ordered pair satisfies every gate,
the renderer is NO-GO. Target 0 and target 1 name coordinates on the global
axis; they are not interpreted as real-camera illumination states.

## Four independently solved cells

For each selected target and each of the lower and higher apertures, select a
cell recipe independently from the complete eligible finite D0 population at
that aperture. No continuous refinement or out-of-atlas level is allowed.
Candidates are ordered by:

1. lower maximum absolute mean/std target residual;
2. lower squared residual sum;
3. lower D0-normalized default distance;
4. lower skin, then lower sclera by f32 total order.

The representative endpoint used to create the target is an explicit member of
this search and proves that a valid solution exists. Each selected cell must
remain within inclusive `0.001` gray of its target on both moments and pass the
same boundary and conditioning gates.

Each cell is independently canonical-rendered. In addition to byte/moment
parity and the image gates above:

- its geometry and raster-aperture fields must match the canonical default
  render at the same aperture bitwise;
- all four cells in a pair must have the expected lower or higher aperture;
- the stereo input must use the left canonical image and its exact anatomical
  horizontal pixel mirror as the right image;
- channel order must be `[left, right]` in canonical CHW `[2,100,100]`;
- conversion is only `u8 / 255`, with no additional normalization.

Each stage independently derives the complete all-30 renderer plan twice before
model bytes are read. Decision then renders and infers only odd S3 and odd
factorial cases. Confirmation first validates the sealed decision, independently
re-derives the all-30 plan twice, requires its exact plan bytes and hash to match
decision, and then renders and infers only even S3 and retained even factorial
cases. Neither stage trusts an unverified cached plan.

Across each preparation repetition, global axis, feasible counts and ranges,
target bits, cell recipe bits, conditioning values, canonical bytes and hashes,
tensors and hashes, case order, and every gate result must match exactly. Any
mismatch is `RENDERER_NO_GO`.

## Factorial estimands

For one gap-four pair and one EyeNet output eye, let `Y_at` be raw openness for
aperture `a` in `{L,H}` and target `t` in `{0,1}`. Define in this exact f64
operation order after converting each raw f32 value to f64:

```text
g0 = Y_H0 - Y_L0
g1 = Y_H1 - Y_L1
pL = Y_L1 - Y_L0
pH = Y_H1 - Y_H0

G = (g0 + g1) * 0.5
P = (pL + pH) * 0.5
M = (g1 - g0) * 0.5
```

`G` is the balanced geometry effect, `P` the global-target main effect, and `M`
the geometry-effect modulation between targets. Both eyes are evaluated
independently. A left/right average is descriptive only and never enters a
gate or class.

Whole-image mean and standard deviation are matched within the frozen
tolerance. Oppositely signed cell residuals can leave up to `0.002` gray of
lower-versus-higher imbalance per component. The signed imbalance is recorded
for every contrast. Histogram shape, skew, kurtosis, edge distribution, local
blocks, and spatial structure are not controlled. The tool records available
higher-moment and spatial covariates descriptively but may not use them to
exclude pairs or change the classification.

## Baseline competence and frozen unit-scale anchors

Each stage also renders the 17 default-photometric S3 apertures of its parity:

- decision: odd indices `7..=39`;
- confirmation: even indices `8..=40`.

For each output eye separately, default S3 must satisfy:

- openness span `>= 0.10`;
- exact-value average-rank Spearman rho with aperture index `>= 0.70`;
- mean absolute retained gap-four delta `B >= 0.10 * 4 / 33`;
- strictly positive gap-four deltas above `0.001 * 4` in at least 90% of the
  stage's retained pairs: 14 of 15 for decision and 13 of 14 for confirmation.

These anchors are the Phase 1.2 unit-scale thresholds mechanically converted
from an adjacent step to a four-index contrast before Phase 1.3 inference:

```text
delta4 = 0.001 * 4        = 0.004
E4     = 0.05 * 4 / 33
B4     = 0.10 * 4 / 33
support ratio = 0.20
flat ratio    = 0.10
concordance   = 0.90, requiring ceil(0.90*n)
```

They are not derived from atlas gray-level residuals and are not p-values. The
atlas cannot convert a renderer residual in gray levels into an openness effect
threshold.

For a stage contrast vector `X` of length `n` and the same-eye stage baseline
`B`, define:

```text
POS(X):
  mean(X) >= E4
  mean(X) / B >= 0.20
  count(X > delta4) >= ceil(0.90*n)

NEG(X) = POS(-X)

FLAT(X):
  mean(abs(X)) < E4
  mean(abs(X)) / B <= 0.10
  count(abs(X) <= delta4) >= ceil(0.90*n)
```

No zero denominator is possible after baseline competence. Equality follows
the operators written above. Metrics are computed separately for each eye and
stage; eyes and stages are never pooled.

Every finite raw f32 output is converted once to f64 before any metric or
threshold comparison. S3 arrays and contrast arrays remain in ascending
aperture or lower-pair-index order. Span is f64 `max - min`. Every sum used by
`B`, a mean, an absolute mean, a covariance, or a sum of squares is a
non-fused left fold in that fixed order followed by one f64 division. Threshold
constants are evaluated as written with f64 operands, left to right; for
example `E4 = (0.05f64 * 4.0f64) / 33.0f64`. The concordance count is computed
without floating arithmetic as integer `(9*n + 9) / 10`.

Average-rank Spearman sorts raw finite f32 values by f32 total order with
original index as the secondary key. Only bit-identical f32 values share a tie
group. Ranks are one-based f64 values; a tie rank is `(first_rank + last_rank) *
0.5`. Aperture ranks follow ascending index. Rank means, covariance numerator,
and both centered sums of squares use ascending aperture order and the left
fold above. Rho is `numerator / sqrt(sum_x2 * sum_y2)`. Zero rank variance is
recorded as undefined and fails baseline competence.

## Inference integrity and recognition

After renderer preparation passes, load two fresh model instances from the
same bytes. The first instance evaluates the fixed case order and the second
evaluates its exact reverse. Every raw output f32 bit for every case must match.
Any nonfinite raw output, model hash change, case mismatch, or repeated-inference
bit mismatch is `INCONCLUSIVE_ARTIFACT`.

Presence is a contract output, not an exclusion variable. If any required S3 or
factorial cell has `(raw_presence as f64) <= 0.05f64`, the stage is
`INCONCLUSIVE_RECOGNITION`. No pair may be dropped, replaced, or reweighted.
Wide, brow, gaze, squeeze, blink, calibration, and production smoothing do not
enter the analysis. Squeeze is recorded only as an untouched contract output.

The fixed inference case order is S3 apertures in ascending index, followed by
retained factorial pairs in ascending lower index and, within each pair,
`L0, H0, L1, H1`. Canonical tensor conversion is exactly `pixel as f32 /
255.0f32`. Tensor SHA-256 uses the existing domain
`sranibro-synthetic-eye-input-f32le-v1\0` followed by every CHW f32 bit pattern
in little-endian order. The first fresh model follows fixed order; the second
fresh model follows the exact reverse order. Results are joined by the sealed
case identity, never by completion order.

## Stage classification

Classification has this fixed precedence:

1. `RENDERER_NO_GO`: an atlas identity, preparation, target, canonical, or
   renderer gate fails. Model bytes are not read.
2. `INCONCLUSIVE_ARTIFACT`: model identity, finiteness, repeat, or case-integrity
   checks fail.
3. `INCONCLUSIVE_RECOGNITION`: any required presence is `<= 0.05`.
4. `INCONCLUSIVE_INSENSITIVE`: baseline competence fails for either eye.
5. Apply the response rubric independently to both eyes.

For one eye, define its provisional response predicates by applying `POS`,
`NEG`, and `FLAT` to that eye's vectors. The response rubric is:

- `GEOMETRY_SUPPORTED`: `POS(g0)`, `POS(g1)`, `POS(G)`, and `FLAT(M)` for both
  eyes;
- `ALTERNATIVE_PHOTOMETRIC_PATH`: `FLAT(g0)`, `FLAT(g1)`, `FLAT(G)`, and
  `FLAT(M)` for both eyes, plus either `POS(pL)`, `POS(pH)`, and `POS(P)` for
  both eyes or `NEG(pL)`, `NEG(pH)`, and `NEG(P)` for both eyes;
- `NO_EVIDENCE`: `FLAT(g0)`, `FLAT(g1)`, `FLAT(pL)`, `FLAT(pH)`, `FLAT(G)`,
  `FLAT(P)`, and `FLAT(M)` for both eyes;
- `INCONCLUSIVE`: every remaining response.

A strong `P` does not invalidate `GEOMETRY_SUPPORTED` when both within-target
geometry contrasts yield a supported balanced `G` and `M` is flat. Descriptive
flags are recorded without changing the class:

- `EYE_ASYMMETRIC`: applying the rubric to each eye alone gives different
  provisional response labels;
- `TARGET_MODULATED`: `FLAT(M)` is false for either eye;
- `GEOMETRY_REVERSED`: `NEG(g0)`, `NEG(g1)`, or `NEG(G)` is true for either eye;
- `TARGET_DIRECTION_MIXED`: one eye satisfies `POS(P)` and the other `NEG(P)`;
- `PHOTOMETRIC_SENSITIVITY`: both eyes satisfy the same signed conjunction of
  `pL`, `pH`, and `P` while the joint response is `GEOMETRY_SUPPORTED`.

## Decision and confirmation sealing

Decision inference uses only odd-index cases. Its class, metrics, raw f32 bits,
model hash, atlas hashes, source identity, renderer plan, canonical hashes, and
non-manifest file inventory are atomically written to a new output directory.
Its completed manifest bytes are hashed and sealed before confirmation model
bytes may be read.

Confirmation first validates the sealed decision artifact and requires the
same atlas, implementation source, preregistration, model bytes, renderer
version, global axis, thresholds, and classification code. It independently
re-derives the all-30 renderer plan, then renders and infers only even-index
cases. Confirmation cannot tune, replace, or rewrite the decision class.

Confirmation status is an ordered total procedure:

1. If either stage has a renderer, artifact, recognition, insensitive, or
   generic `INCONCLUSIVE` status, final status is
   `CONFIRMATION_INCONCLUSIVE`.
2. Otherwise, if the two conclusive response classes are identical, final
   status is `REPLICATED`.
3. Otherwise, final status is `NOT_REPLICATED`.

`NOT_REPLICATED` and `CONFIRMATION_INCONCLUSIVE` do not erase the sealed primary
result, but no transferable synthetic conclusion may be claimed. A strong
Phase 1.3 conclusion requires `REPLICATED`.

Both stages require a clean committed checkout, Cargo release profile with
debug assertions disabled, and equality among embedded build commit, runtime
`HEAD`, compiled implementation fingerprint, and checkout implementation
fingerprint. These checks occur before preparation, before model loading, and
immediately before artifact publication.

The implementation fingerprint is SHA-256 over exact bytes, in the following
path order. For each entry it appends the UTF-8 path length as u64
little-endian, the path bytes, the file byte length as u64 little-endian, and
the exact file bytes:

```text
Cargo.toml
Cargo.lock
build.rs
research/synthetic-eye-lab/PHASE1_3_PREREG.md
research/synthetic-eye-lab/phase13.rs
research/synthetic-eye-lab/phase13_main.rs
research/synthetic-eye-lab/phase13_output.rs
research/synthetic-eye-lab/renderer.rs
research/synthetic-eye-lab/model.rs
research/synthetic-eye-lab/moments.rs
src/lib.rs
src/ml/mod.rs
src/ml/eye_net.rs
src/ml/tvm_params.rs
```

Each stage writes into a new sibling staging directory. Every non-manifest file
is flushed and assigned a raw-file SHA-256. `manifest.json` contains the sorted
relative-path/length/hash inventory and is written and flushed last as the
completion marker; it does not attempt to hash itself. The stage seal is the
raw SHA-256 of the completed manifest bytes. The staging directory is then
atomically renamed to the requested new output path. Confirmation recomputes
the decision inventory, manifest seal, source identity, plan hash, and every
declared file length before accepting it. Existing output paths, partial
directories, unlisted files, and temporary files are errors.

The renderer-plan hash uses domain
`sranibro-synthetic-eye-phase13-plan-v1\0` followed by the exact deterministic
`renderer_plan.json` bytes. The JSON contains only fixed-order structs and
arrays, never unordered maps. Both preparation repetitions must produce
bit-identical plan JSON before any artifact is published.

## Anti-tailoring rules

- This document is committed alone before Phase 1.3 implementation or model
  inference.
- The implementation follows in a separate commit and records both identities.
- Renderer-only implementation failures may be repaired only through a new,
  explicit preregistration amendment committed before any model output exists.
- Once any Phase 1.3 model output exists, no atlas, domain, boundary, kappa,
  axis, target, solver, pair, threshold, metric, or class may change.
- Decision is run exactly once from clean committed release source.
- Confirmation is run exactly once from the same clean committed release
  source after decision sealing.
- Exploratory calculations may be labelled as such but cannot alter either
  class.

## Later HMD transfer audit

Only after decision and confirmation are sealed may authorized XR5 or VR4
recordings be opened. The transfer audit must be separately preregistered. It
may test whether production preprocessing moves real model inputs through the
same brightness, contrast, recognition, and openness-response regions, but it
cannot rewrite the synthetic result.

Potential product consequences such as bounded adaptive brightness, crop and
rotation fitting, recognition-confidence fallback, or user-specific initial
geometry remain hypotheses until that real-data audit. No production setting
is changed in Phase 1.3.

## Explicit deferrals

- real-camera fitting or thresholds;
- XR5 or VR4 recording inspection;
- histogram, CDF, LUT, or block-distribution matching;
- near-closed apertures `0..6`;
- asymmetric stereo or monocular-loss cases;
- Wide, brow, gaze, blink, and calibration modelling;
- any production configuration or runtime change;
- multi-user or multi-device generalization.
