# Phase 1.3A preregistration: renderer moment-feasibility atlas

Status: frozen before atlas implementation or execution.

Phase 1.2 stopped at its renderer gate. Its common renderer-valid aperture set
was `18..=26` (9 of 34), and neither required reference survived the opposite
reference match. EyeNet was not loaded. Phase 1.2 therefore made no statement
about model geometry sensitivity.

This protocol characterizes only the finite photometric search surface of the
committed synthetic renderer. It does not change or reinterpret Phase 1.2.

## Question and interpretation limit

For each synthetic aperture, what post-u8 whole-image mean and population
standard deviation are found on three fixed finite skin/sclera candidate sets,
and how close are the finite clouds for every pair of apertures?

The atlas may show that a closer configuration was or was not found under this
frozen finite search. It cannot prove continuous-domain reachability or
impossibility, renderer realism, EyeNet sensitivity, or real-camera coverage.

## Isolation from models and recordings

The atlas is a separate feature-gated binary. Its module tree may include only
the canonical renderer and atlas-specific code. It must not import the EyeNet
model loader, experiment runner, inference output code, Phase 1.1/1.2 solvers,
or real-recording loaders.

Its CLI accepts only:

```text
synthetic-eye-atlas --out <new-or-empty-directory>
```

`--model`, recording paths, and unknown arguments are errors. The manifest must
state:

```text
model_loaded = false
model_identity_sha256 = null
phase0_evaluated = false
real_recordings_loaded = false
```

The two authorized XR5 sessions and any later VR4 recording remain outside the
atlas. Their image contents, moments, labels, and model outputs may not select a
domain, level, pair, metric, target, or threshold. Real recordings enter only
after a separate synthetic model classification has been preregistered, run,
and sealed.

## Canonical renderer

The aperture universe is fixed:

```text
i = 7..=40
a_i = 1.30 * i / 40
```

Only `skin_level` and `sclera_level` vary. Every other
`SyntheticEyeSpec` field remains at its committed default. Rendering remains
100x100 at 4x supersampling with the existing f32 sample addition order and u8
quantization:

```text
(level.clamp(0.0, 1.0) * 255.0).round() as u8
```

Mean and population standard deviation use the shared post-u8 f64
sum/sum-of-squares implementation over all 10,000 pixels. All levels are cast
to f32 exactly once before rendering. Candidate identity is the pair
`(skin_f32_bits, sclera_f32_bits)`.

## Nested finite candidate sets

The legacy grid is:

```text
G0 = {
  skin   = 0.30 + 0.30*k/128,
  sclera = 0.65 + 0.30*l/128
  | k,l in 0..=128
}
```

The unit grid is:

```text
Gunit = {
  skin = k/128,
  sclera = l/128
  | k,l in 0..=128
}
```

The exact committed default pair `(0.46f32, 0.78f32)` is denoted `Pdefault`.
The three finite domains are sets, not independently sampled rectangles:

```text
D0 = G0 union {Pdefault}
D1 = D0 union {p in Gunit | p.sclera >= p.skin}
D2 = D0 union Gunit
```

Applied-f32 duplicate pairs are removed and their domain-membership bits are
ORed. This guarantees the discrete nesting `D0 subset D1 subset D2`. D1 permits
zero contrast only on its reported polarity boundary; D2 is a mathematical
diagnostic that may contain inverted contrast. D1 and D2 do not become eligible
for a later model study automatically.

Candidate generation order is legacy grid in increasing `k,l`, then unit grid
in increasing `k,l`, then the exact default. The persisted stream is finally
ordered by aperture index, skin f32 total order, then sclera f32 total order.

## Moment-cloud representative rule

Every applied level pair is retained in the binary candidate stream. For
nearest-cloud search only, candidates with bit-identical `(mean, stddev)` within
one aperture and one domain are represented by exactly one level pair, ordered
by:

1. lowest squared distance from `(0.46, 0.78)`, after normalizing both axes by
   `0.30` in D0 and by `1.0` in D1/D2;
2. lower skin by f32 total order;
3. lower sclera by f32 total order.

This reduction cannot change the later pair ordering because it uses the same
tertiary and level tie breaks.

## Exact pair search

Only distinct unordered aperture pairs are searched:

```text
7 <= i < j <= 40
```

There are exactly 561 pairs per domain and 1,683 pair summaries. Reverse-order
records are not recomputed.

For lower-aperture candidate `p` and higher-aperture candidate `q`:

```text
d_mean = q.mean - p.mean
d_std  = q.stddev - p.stddev
L_inf  = max(abs(d_mean), abs(d_std))
L2_sq  = d_mean*d_mean + d_std*d_std
```

The winning pair is the first under this complete order:

1. lower `L_inf` by f64 total order;
2. lower `L2_sq`;
3. lower sum of the two domain-normalized default distances;
4. lower-aperture skin, then sclera, by f32 total order;
5. higher-aperture skin, then sclera, by f32 total order.

An exact deterministic two-dimensional kd-tree may accelerate the search. Tree
construction alternates mean and stddev axes by depth, sorts by split value,
other value, skin, and sclera, and chooses median `len/2`. Every node stores its
closed mean/stddev bounding box.

For query-to-box component distances `dx,dy`, a subtree may be pruned only when:

```text
max(dx,dy) > best.L_inf
```

or when the lower-bound `L_inf` is exactly equal and
`dx*dx + dy*dy > best.L2_sq`. Equality at both stages is never pruned because a
later tie break may improve. Children are visited by lower bound, with the left
child first on a complete tie.

Reduced-grid tests must compare every kd-tree result with exhaustive brute force,
including exact moment and level ties. The production implementation remains
exact, not approximate.

## Candidate midpoint and conditioning diagnostics

For every winning pair, the atlas records a symmetric candidate target with
this exact operation order:

```text
target_mean = (p.mean + q.mean) * 0.5
target_std  = (p.stddev + q.stddev) * 0.5
```

The target is descriptive. It may be unreachable by either continuous renderer
surface and is not accepted for model inference by this protocol.

A fixed-scale central secant matrix is also reported when all four probes remain
inside the continuous version of the selected domain and cast to distinct f32
bits:

```text
D0 h_skin = h_sclera = 0.30/128
D1/D2 h_skin = h_sclera = 1.0/128
```

The denominator is the difference between the actual applied f32 probe values,
not nominal `2h`. The matrix maps skin/sclera changes to mean/stddev changes.
Its determinant, two singular values, and condition number are descriptive
only. If probes are unavailable, the fields are null with a fixed reason. No
one-sided substitute and no conditioning threshold are allowed.

## Fast/canonical renderer gates

For every aperture, the ordered `PhotometricBasis` prediction must be byte-,
SHA-256-, mean-bit-, and stddev-bit-identical to an independent canonical render
at all of these fixed anchors:

- the 5x5 D0 quartile lattice from legacy-axis indices
  `{0,32,64,96,128}`;
- the 5x5 unit-square lattice from unit-axis indices
  `{0,32,64,96,128}`;
- the exact default pair.

Every level configuration selected into a pair summary is independently checked
the same way. Duplicate selected configurations may use a deterministic cache.

All canonical covariates for selected configurations are retained, including
eye-like validity, frame contact, saturation, visible area, edge energy, and
measured geometry/raster aperture. They are descriptive in this atlas and do not
silently remove candidates.

Any fast/canonical mismatch, nonfinite selected statistic, wrong aperture count,
wrong pair count, or kd-tree/brute-force test failure is an implementation
NO-GO. It may not be converted into a scientific feasibility result.

## Deterministic artifacts

The complete preparation is executed twice. The second preparation must match
the first in candidate-stream SHA-256 and exact deterministic pair-summary JSON
bytes. A mismatch is an implementation NO-GO.

`candidate_stream.bin` starts with a versioned fixed header. Each fixed-size
little-endian record contains, in order:

```text
aperture_index u8
domain_membership_mask u8   # D0=1, D1=2, D2=4
reserved_zero u16
skin_f32_bits u32
sclera_f32_bits u32
mean_f64_bits u64
stddev_f64_bits u64
```

The output directory is new or empty. The recorded run refuses a dirty or
unidentifiable Git worktree. It writes:

- `candidate_stream.bin`;
- `aperture_summaries.json`;
- `pair_summaries.json`;
- `canonical_checks.json`;
- `manifest.json`, written last.

The manifest has no wall-clock field. It records the clean implementation
commit, this preregistration commit, renderer/atlas versions, exact candidate and
pair counts, both preparation hashes, artifact hashes, CLI sanitization, all
four false/null model/recording fields, and the interpretation limit.

## Atlas outcome

There is no `GO`, `NO_EVIDENCE`, or model-facing readiness threshold in Phase
1.3A. For each pair, D0/D1/D2 signed component residuals, `L_inf`, `L2_sq`,
levels, boundary margins, canonical hashes, midpoint, and conditioning are
reported without a pass/fail label.

Domain expansion may only decrease or preserve the winning distance because the
finite candidate sets are nested. A violation is an implementation NO-GO.

The only allowed scientific statement is that the reported configurations and
distances were found on the frozen finite candidate sets. "Not found" never
means mathematically unreachable over a continuous domain.

## Handoff rule

After the clean atlas artifact is sealed, a separate Phase 1.3 preregistration
may use the D0 adjacent-pair summaries as renderer-instrument calibration. That
new document must independently freeze its pair universe, common-target solver,
component tolerances, boundary and image gates, symmetric 2x2 estimand,
disjoint decision subset, repeat-inference gate, and classification precedence.

The atlas winner may seed that solver but may not restrict it to a local
neighborhood. D1/D2 use in a model experiment requires a separate physical
justification written before inference. No model output or real recording may
feed back into the atlas or alter this protocol.
