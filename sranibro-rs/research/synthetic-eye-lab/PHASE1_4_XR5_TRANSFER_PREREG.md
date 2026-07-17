# Phase 1.4 preregistration: XR5 real-recording transfer and stability audit

Status: frozen before implementation, input sealing, PNG decode, or inference.
This document is committed by itself. The implementation must be a later
descendant commit and may not share this preregistration commit.

## Purpose and boundary

Phase 1.4 is a read-only audit of the already-recorded Dream Air / XR5 guided
EyeWide sessions. It asks how the fixed SRanipal-compatible stereo EyeNet behaves
on real XR5 images under the committed default XR5 preprocessing path.

The primary questions are:

1. whether raw EyeNet openness preserves the ordered relation
   `neutral_center < wide_soft < wide_max` for each session and eye, or instead
   shows a high-open plateau or reversal;
2. how much same-label raw openness changes with gaze, short within-phase time,
   reseating/session, image photometry, and image sharpness;
3. whether changes are common to both eyes or appear as a left/right differential;
4. whether two fresh EyeNet instances produce bit-identical outputs when the same
   cases are evaluated in opposite orders.

This is a descriptive transfer audit, not a rescue or reinterpretation of the
Phase 1.3 synthetic result. It does not fit a model, search geometry, select a
threshold, tune preprocessing, update calibration, or change production state.
No XR5 recording may be inspected for Phase 1.4 or used to select or change this
protocol before this document is committed.

The audit may begin only after the Phase 1.3 decision and confirmation artifacts
have both been sealed. The implementation must verify those artifacts and their
published seals before opening the recording root:

- decision manifest seal:
  `17291d72ab05034ea5c047225c6868ea714c6dbbebe2122beb41519dc02dab48`;
- confirmation manifest seal:
  `2e26e3d94c9ab267862869cf7fc3cc8740a15f94e4dad7cf6df10b2e254da93c`.

## Fixed model and production path

The only authorized EyeNet parameters are the same bytes used by Phase 1.3:

- byte length: `51,423,934`;
- SHA-256: `bac8013e0423068924f190a1de44afd5e1dd0c7c10d1d394926e46fc1b075ded`.

The implementation loads two fresh `EyeNet` instances from those bytes. It calls
the raw stereo forward path and records all five outputs without post-processing:
presence, left openness, right openness, left squeeze, and right squeeze. Squeeze
is reported only as a descriptive secondary output.

The fixed XR5 preprocessing is the committed production default, not the user's
mutable configuration:

- the frozen left geometry is crop `(left=0, right=0.40, top=0.15,
  bottom=0.15)`, scale `(x=1.0, y=1.20)`, rotation `-30.0` degrees;
- the frozen right geometry is crop `(left=0.40, right=0, top=0.15,
  bottom=0.15)`, scale `(x=1.0, y=1.20)`, rotation `+30.0` degrees;
- these values must equal `default_ml_geometry("pimax_xr5")` at runtime;
- `DespeckleParams::default()` (`enabled=true`, threshold `0.15`, radius `3`);
- `FlattenParams::default()` (`enabled=false`, strength `0.7`, radius `0.33`);
- adaptive brightness disabled, with identity affine `(1, 0)` for both eyes;
- no eye swap and no whole-frame mapping flip;
- geometry mirror state resolved to left `false`, right `true`;
- preprocessing order: reconstruct live orientation, despeckle, disabled flatten,
  identity brightness, per-eye geometry/mirror, contiguous CHW `[2,100,100]`.

The capture writer stores the left PNG unchanged and horizontally mirrors the
right PNG. To reproduce the live pipeline, the audit horizontally mirrors the
stored right PNG once to reconstruct the original camera orientation, then runs
the ordinary right-eye XR5 geometry and mirror. It must not feed the already
canonical right PNG through the live right mirror a second time. A source-only
test using a non-symmetric synthetic 200x200 image must prove that save-mirror,
audit-unmirror, and live preprocessing reproduce the direct live tensor bit for
bit before the real audit is accepted.

## Authorized recording population

The command accepts one `wide_data` root and includes every lexicographically
sorted child directory under `wide_data/sessions` that contains `labels.csv`.
There is no session selector and no post-hoc exclusion. At least two completed
sessions are required. Directories without `labels.csv` are inventoried as
partial captures but are not decoded or analyzed, matching the capture contract.

Every completed session must satisfy the following strict audit-input contract
before any model bytes are loaded. The 200x200 requirement is an authorization
gate for this XR5 study; the generic capture writer itself accepts other positive
frame sizes.

- CSV header exactly `filename,wide,phase,side`;
- UTF-8, LF-only CSV bytes with exactly one header, exactly 4,080 non-empty data
  rows, no blank data rows, and one final LF;
- every row has exactly four fields and a normalized relative path contained by
  that session's `images` directory;
- only ordinary directories and regular files are allowed; symbolic links,
  junctions/reparse points, and special entries are rejected, and every directory
  enumeration error is fatal;
- the only allowed entries in a completed session are `labels.csv`, `images`, the
  fourteen exact `images/<phase>` directories below, and the referenced PNGs;
  all referenced files are unique and no expected, extra, or empty directory or
  file is present;
- every image is an 8-bit grayscale PNG of exactly 200x200 pixels;
- global sequence numbers are exactly `0..2039`, with one left row followed by
  one right row for each sequence and both rows in the same phase;
- every filename has exact grammar
  `<phase>/<phase>_[lr]_[0-9]{8}.png` using forward slashes;
- the filename phase, CSV phase, side letter, sequence, label, and phase count all
  match the following frozen schedule. The CSV label is parsed as finite `f32`
  and compared numerically to `0.0`, `0.5`, or `1.0`; its source text is normally
  `0`, `0.5`, or `1` and is not compared to the table's display spelling.

| Phase | Label | Stereo pairs | Global sequence |
|---|---:|---:|---:|
| `neutral_center` | 0.0 | 180 | 0-179 |
| `wide_soft` | 0.5 | 150 | 180-329 |
| `wide_max` | 1.0 | 150 | 330-479 |
| `gaze_up_neutral` | 0.0 | 240 | 480-719 |
| `gaze_down_neutral` | 0.0 | 150 | 720-869 |
| `gaze_left_neutral` | 0.0 | 120 | 870-989 |
| `gaze_right_neutral` | 0.0 | 120 | 990-1109 |
| `wide_gaze_up` | 1.0 | 120 | 1110-1229 |
| `wide_gaze_down` | 1.0 | 120 | 1230-1349 |
| `blink_negative` | 0.0 | 180 | 1350-1529 |
| `closed_negative` | 0.0 | 120 | 1530-1649 |
| `squint_negative` | 0.0 | 150 | 1650-1799 |
| `left_wink_negative` | 0.0 | 120 | 1800-1919 |
| `right_wink_negative` | 0.0 | 120 | 1920-2039 |

Any fully enumerable schema violation produces the input-stage sealed
`INPUT_INVALID` result. Any pixel/decode or static preprocessing violation found
by `analyze` produces its reduced sealed `INPUT_INVALID` result. Neither path
loads EyeNet or emits a scientific annotation. Decode failures are never skipped.

## Input sealing and privacy

The tool has two mandatory subcommands. `seal-input` accepts the two sealed Phase
1.3 artifacts, the `wide_data` root, and a new seal destination. It rejects every
model argument, never constructs or calls EyeNet, and does not decode PNG pixels.
It constructs a deterministic inventory of every directory and regular file
below `wide_data/sessions`. Child session directory names must match
`session-[0-9]{13}-[0-9]{10}-[0-9]{6}`. They are sorted by their unmodified UTF-8
bytes. Completed directories are mapped by completed-only rank to `session_0000`,
`session_0001`, and so on; directories without `labels.csv` are mapped separately
to `partial_0000`, `partial_0001`, and so on, preserving source order within each
group. Inventory paths use those IDs, literal `/` separators, no Unicode or case
normalization, and UTF-8 byte-order sorting. Directory entries record their kind;
file entries also record byte length and SHA-256. This includes empty directories,
so an extra empty entry cannot evade the seal.

These ordinary-entry rules apply to the entire sealed tree, including partial
session directories. Any symbolic link, junction/reparse point, special entry,
non-UTF-8 name, unexpected root-level file, or enumeration error is forbidden.

`seal-input` must parse and validate every non-pixel contract: root and entry
grammar, exact tree, CSV bytes and fields, schedule, counts, paths, side pairing,
and completed-session coverage. `.` components, `..` components, backslashes,
absolute/prefixed paths, non-UTF-8 names, and path escape are forbidden. Full PNG
decode, color, and dimensions remain exclusively in `analyze`. A fully enumerable
but invalid tree is sealed with `terminal_status=INPUT_INVALID` using the same
three-file allowlist; `analyze` accepts only `terminal_status=INPUT_SEALED`. A tree
that cannot be completely enumerated or read is an untrusted hard failure and no
seal is emitted. The absolute recording root and original session directory names
are not written to the artifact.

`seal-input` publishes exactly `recording_inventory.json`, `session_plan.json`,
and `manifest.json`, with the manifest last. Its raw manifest SHA-256 is printed
as the external input seal. A valid seal has `terminal_status=INPUT_SEALED`; fewer
than two completed sessions produces `terminal_status=INPUT_INVALID` with reason
`insufficient_completed_sessions` and can never enter `analyze`.
The input manifest does not contain its own hash; its external seal exists only
outside those bytes.

`analyze` accepts the same `wide_data` root, the sealed-input directory and its
64-digit lowercase external seal, the fixed model, the two Phase 1.3 artifacts,
and a new analysis destination. It verifies the external seal over the raw input
manifest bytes before parsing that JSON, verifies the sealed artifact allowlist
and hashes, and reconstructs the current recording inventory. Any mismatch is a
hard failure before PNG decode or model loading. The inventory is recomputed
again after inference and must remain bit-identical.

Only `analyze`, after the external tree seal has been verified, fully decodes PNG
headers and image data. It performs exactly four independent decode/preprocess
passes: two preflight passes before model loading, model A's forward-order pass,
and model B's reverse-order pass. Every pass rereads each PNG, verifies those bytes
against the sealed per-file length and SHA-256 before decode, and reconstructs the
case without reusing cached pixels or tensors. Truncated or corrupt image data,
unexpected trailing image frames, non-Gray8 format, or non-200x200 dimensions
in either preflight pass produce the reduced `INPUT_INVALID` artifact before
EyeNet is loaded. After successful preflight, an inference pass that cannot
reproduce the same decoded case and hashes yields `INCONCLUSIVE_DETERMINISM`.
Decode failures are never silently skipped.

Raw PNG bytes and rendered eye images are never copied into the output artifact.
Only hashes, scalar image statistics, raw model values, and aggregate summaries
are written. The model file is also hashed before and after inference.

## Frozen frame selection

Sessions are analyzed independently. Within each session and phase, stereo pairs
are ordered by their global numeric sequence. For `n` pairs, discard
`floor(0.20*n)` pairs from each end and retain the central remainder. The trim is
applied to pairs, so left and right always use identical sequences.

All retained pairs enter transfer summaries, common/differential summaries, and
directional annotations. Starting with the first retained pair, every twelfth
retained pair enters the within-label association table. Separately, every one of
the 2,040 pairs in each completed session enters the inference and full-phase time
audit. No frame is selected or removed based on image content, presence, or model
output.

The implementation must assert the schedule-derived totals: 1,224 retained pairs
and 104 association pairs per completed session.

## Frozen scalar definitions

All aggregate calculations use `f64`. A quantile sorts finite values and applies
linear interpolation at zero-based position `p*(n-1)` (R type 7). Median is the
`p=0.5` quantile. MAD is the median absolute deviation from the median. Standard
deviation is the population value with divisor `n`. Spearman correlation is the
Pearson correlation of average ranks; it is null for fewer than three cases or
zero rank variance. No p-values are computed.

For each eye and every pair, statistics are computed on three planes: the
reconstructed native 200x200 frame, the post-despeckle native 200x200 frame, and
the final 100x100 EyeNet channel:

- mean and population standard deviation;
- type-7 `q01`, `q16`, `q50`, `q84`, and `q99`;
- black fraction (`value <= 5/255`) and saturated fraction (`value >= 250/255`);
- mean absolute neighbor gradient: the sum of all horizontal and vertical
  absolute differences divided by the number of those edges, in `[0,1]` units.

For the native despeckle step, the changed-pixel fraction and mean signed change
`post_despeckle - reconstructed_native` in `[0,1]` units are also recorded. These
definitions are diagnostics, not new preprocessing controls.

Each phase/session/eye summary reports count, median, MAD, q05, q25, q75, and q95
for presence, openness, squeeze, and every input statistic. It also reports the
median and q95 absolute first difference between consecutive retained openness
values. The shared presence output is copied to both eye rows only for tabular
symmetry; it remains one stereo output.

## Primary transfer annotation

The production presence threshold `0.05` is frozen as a recognition diagnostic.
A pair is recognized only when all five raw outputs are finite and presence is
strictly greater than `0.05`; equality is not recognized. No pair is discarded
or reweighted for failing recognition. A stratum passes the fixed 95% coverage
gate exactly when `recognized_count * 100 >= 95 * stratum_count`, using integer
arithmetic. Thus the central N/S/M requirements are respectively 103 of 108, 86
of 90, and 86 of 90 pairs. A session/eye is evaluable for the primary ordinal
annotation only when all three central strata pass. Any non-finite raw model
output anywhere in an included session makes the scientific result
`INCONCLUSIVE_ARTIFACT` with reason `nonfinite_model_output` after the two model
streams have first been checked for bit identity.

The fixed research deadband is `delta = 0.004` raw-openness units, inherited from
the already-committed Phase 1.3 unit-scale comparison rather than selected from
these recordings. For every evaluable session and eye, let `N`, `S`, and `M` be
median raw openness for `neutral_center`, `wide_soft`, and `wide_max`. The
per-session/eye categories are:

- `MONOTONE` when `S-N > delta` and `M-S > delta`;
- `PLATEAU` when `S-N > delta` and `abs(M-S) <= delta`;
- `REVERSAL` when `S-N > delta` and `M-S < -delta`;
- `SOFT_TRANSFER_NOT_SHOWN` when `S-N <= delta`, regardless of `M`.

The exact differences `S-N`, `M-S`, and `M-N` are always reported so the deadband
does not hide the raw effect size.

The audit-level high-open annotation uses this fixed precedence:

1. `HIGH_OPEN_REVERSAL_OBSERVED` if any evaluable session/eye is `REVERSAL`;
2. otherwise `HIGH_OPEN_PLATEAU_OBSERVED` if any evaluable session/eye is
   `PLATEAU`;
3. otherwise `HIGH_OPEN_MONOTONE_ALL` if every included session/eye is evaluable
   and `MONOTONE`;
4. otherwise `HIGH_OPEN_TRANSFER_NOT_ESTABLISHED`.

This annotation describes only the included recordings. It is not a claim about
all users or devices.

For each session, if the left and right central categories are both evaluable and
differ, the secondary annotation is `EYE_CATEGORY_ASYMMETRIC`; if both are
evaluable and equal it is `EYE_CATEGORY_MATCHED`; otherwise it is
`EYE_CATEGORY_NOT_EVALUABLE`. This never changes the audit-level precedence.

The following endpoint checks are reported separately and never change the
high-open annotation:

- `neutral_center - closed_negative > delta` for each eye;
- `wide_max - neutral_center > delta` for each eye;
- in `left_wink_negative`, left median below the simultaneous right median;
- in `right_wink_negative`, right median below the simultaneous left median.

Each wink check uses the median simultaneous non-wink minus wink openness and
requires it to exceed `delta`.

Endpoint, wink, and gaze contrasts always report their raw medians and
differences. A categorical annotation is emitted only when every participating
central retained stratum passes the same integer 95% recognition gate; otherwise
that annotation is `NOT_EVALUABLE`.

`blink_negative` and `squint_negative` are distribution diagnostics only.

## Gaze, time, session, and association diagnostics

Gaze-labelled strata are never pooled into an aperture claim. For each session
and eye, the tool reports median openness and input-statistic differences for:

- every `gaze_*_neutral` phase minus `neutral_center`;
- `wide_gaze_up` and `wide_gaze_down` minus `wide_max`;
- `wide_gaze_up` minus `gaze_up_neutral`, and `wide_gaze_down` minus
  `gaze_down_neutral`, to measure Wide ordering within the same instructed gaze.

It also reports the absolute gaze-neutral openness difference divided by
`abs(S-N)`, null when the session primary annotation is not evaluable or when
`abs(S-N) <= delta`. This ratio is descriptive. A
same-label gaze contrast with absolute openness change greater than `delta` gets
the annotation `POSE_OR_ORDER_SENSITIVE`; it is not called a causal pose effect
because the capture phase order is fixed.

Independently of the central transfer trim, every full phase is divided into five
contiguous equal-count blocks. All frozen phase target counts are divisible by
five; any remainder is an input-contract failure. Block medians, block-five minus
block-one differences, and the range of the five block medians are reported for
presence, raw openness, squeeze, and all three-plane input statistics. The
high-open category is also computed separately for each aligned block number
across `neutral_center`, `wide_soft`, and `wide_max`, and the artifact states
whether all five block categories agree with the central aggregate. Because each
recorded phase lasts only several seconds, this is explicitly a short-duration
drift diagnostic and cannot establish hour-scale stability.

Each block category uses the same integer 95% recognition gate independently for
its N/S/M blocks; a block failing any of the three is `NOT_EVALUABLE`. Block
agreement is `ALL_AGREE` only when all five blocks are evaluable and equal the
central aggregate, `DISAGREE` when all five are evaluable but at least one differs,
and `NOT_EVALUABLE` otherwise.

For each phase/eye, every later session is compared with the first session by
subtracting phase medians for openness and input statistics. Sessions are never
pooled for a directional claim. These comparisons confound elapsed time, headset
reseating, and expression repeatability and are labelled accordingly.

Except for the explicitly full-phase five-block time audit, every endpoint, gaze,
wink, session, association, and common/differential contrast uses the central 60%
retained pairs.

For each session/phase/eye, the every-twelfth-pair subset reports Spearman
correlations between raw openness and same-eye raw, post-despeckle, and final
mean, standard deviation, and gradient, plus despeckle changed-pixel fraction.
They are observational within-label associations, not causal photometric effects.

For every retained stereo pair, common openness is `(L+R)/2` and differential
openness is `(L-R)/2`. Per-phase median, MAD, q05, and q95 are reported. No forced
left/right coupling is evaluated or applied.

## Determinism and execution order

After validation and input sealing, the complete decode and preprocessing plan is
streamed twice independently before model loading. For every immutable
session/phase/sequence key, each pass records domain-separated SHA-256 values for
the reconstructed left/right native bytes and the final stereo tensor's little-
endian `f32` bit stream. The two passes must match. Tensors are not retained for
all sessions in memory.

Model A then reconstructs and hash-checks each of the 2,040 stereo cases per
completed session in session order, phase schedule order, then sequence order.
Fresh model B independently reconstructs and hash-checks the same cases in exact
reverse order. All five raw `f32` output bit patterns are joined by case identity
and must match. `frames.csv` records the two preprocessing-pass tensor hashes and
both model streams as five unsigned 32-bit bit patterns per case; the manifest
also records domain-separated ordered-stream digests. Thus determinism can be
checked without trusting decimal formatting or a boolean summary. Any
preprocessing or model-stream bit mismatch yields `INCONCLUSIVE_DETERMINISM`, and
no transfer annotation is published.

The analysis terminal-status precedence is fixed:

1. `INPUT_INVALID` for PNG decode/format/dimension failure or a static
   preprocessing-contract failure such as wrong frozen defaults, wrong tensor
   shape, or non-finite tensor; EyeNet is not loaded;
2. `INCONCLUSIVE_DETERMINISM` when two constructions of the same sealed case or
   the two fresh model streams differ in any bit;
3. `INCONCLUSIVE_ARTIFACT` for fixed-model identity/parse/load failure, or for
   model streams that are bit-identical but contain a non-finite output;
4. `AUDIT_COMPLETE`, accompanied by the frozen high-open annotation and all
   secondary diagnostics.

Every non-`AUDIT_COMPLETE` analysis has a null high-open annotation. A Phase 1.3
seal mismatch, external input-seal mismatch, source/build/runtime identity
mismatch, or recording-tree mismatch is untrusted input and is rejected before
the analysis destination is created. A recording tree, model file, repository,
or compiled-source identity that changes during execution also prevents artifact
publication rather than producing a scientific status.

Exactly one release-build `seal-input` invocation creates the authorized input
seal, followed by exactly one release-build `analyze` invocation for the recorded
audit. Both require a clean Git worktree, a build whose embedded commit matches
runtime `HEAD`, the committed preregistration as an ancestor, a new output
destination, the two sealed Phase 1.3 artifacts, and the `wide_data` root.
`analyze` additionally requires the sealed input and fixed model. Neither command
writes to the recording root or application configuration.

The compiled-source fingerprint covers exactly `Cargo.toml`, `Cargo.lock`,
`build.rs`, `research/synthetic-eye-lab/PHASE1_4_XR5_TRANSFER_PREREG.md`,
`research/synthetic-eye-lab/xr5_transfer_main.rs`,
`research/synthetic-eye-lab/xr5_transfer.rs`,
`research/synthetic-eye-lab/xr5_transfer_output.rs`, `src/lib.rs`,
`src/config.rs`, `src/core/types.rs`,
`src/ml/mod.rs`, `src/ml/eye_net.rs`, `src/ml/tvm_params.rs`,
`src/ml/preprocess.rs`, `src/wide_calib.rs`, and `src/pipeline.rs`. The same sorted,
domain-separated source hash is recomputed from the checkout at runtime.

## Required artifact

The sealed analysis output contains exactly:

- `frames.csv` with one row per full-session stereo pair, both raw channels/stat
  sets, frozen retained/association/block roles, both preprocessing-pass native
  and tensor hashes, and both models' five raw `u32` bit patterns;
- `phase_summaries.json`;
- `temporal_blocks.csv`;
- `associations.csv`;
- `gaze_and_session_differences.csv`;
- `interpretation.txt`;
- `manifest.json`, published last.

If analysis cannot reach `AUDIT_COMPLETE`, its exact reduced allowlist is
`diagnostic.json`, `interpretation.txt`, and `manifest.json`. It contains no
scientific annotation. A bad external input seal or recording-tree mismatch is
rejected before creating an analysis destination and therefore cannot be confused
with a trusted failure artifact.

For `INCONCLUSIVE_DETERMINISM`, `diagnostic.json` contains the ordered-stream
digests and every mismatching case key with both preprocessing hashes or both
five-value model bit patterns. For same-bit non-finite output it contains every
affected case and raw bit pattern. Other failure statuses contain their exact
pre-model reason and `model_loaded` state.

The analysis manifest records the preregistration commit,
implementation/build/runtime commit, clean-worktree state, compiled-source
fingerprint, model identity, verified Phase 1.3 seals, external input-manifest
seal and recording-tree hash, anonymized session count, exact preprocessing
constants, frame counts, determinism result, input inventory hash before and
after inference, hashes of every artifact file, scientific status, and the
ordered-stream digests. It does not contain its own hash. After the manifest is
flushed and closed, the raw manifest bytes are SHA-256 hashed and that external
analysis seal is printed only to stdout. The verifier requires the exact
stage-specific file allowlist and recomputes all non-manifest hashes and the
external manifest seal.

## Interpretation limits and forbidden changes

The available recordings are repeated sessions from one user and one XR5 device.
The committed XR5 geometry was itself discovered by evaluating labelled 200x200
XR5 captures. This run unconditionally records `independence_unproven`; it accepts
no provenance override. The scientific output is therefore an in-sample
characterization, not an independent transfer validation. A later study may
claim independence only under a separately preregistered and sealed provenance
contract. This audit can still expose a concrete reversal, category asymmetry,
pose/order association, or short-duration instability in these sessions. It
cannot establish prevalence,
multi-user generalization, an hour-scale cause, or a production correction.

The fixed capture order makes phase and elapsed-time effects inseparable. Phase
instructions are not measured physical aperture ground truth. The saved 30 Hz
subsample cannot evaluate 120 Hz dynamics or faithfully quantify brief blinks.
Presence does not prove correct openness. The fixed brightness-off path does not
represent a runtime where adaptive brightness is enabled, and this raw-model
audit does not evaluate baseline or continuous-calibration state.

After any real frame has been decoded, none of the frame selection, preprocessing,
statistics, thresholds, classifications, or output fields above may change. A
bug that can affect a scientific value requires a new explicit amendment committed
before another run; the original artifact remains immutable. Exploratory follow-up
may be added only under a new preregistration and cannot relabel this result.

The following are expressly forbidden in Phase 1.4:

- fitting EyeNet, WideNet, or any calibration head;
- crop, rotation, stretch, mirror, despeckle, flatten, or brightness search;
- reading or applying mutable per-user SRanibro settings;
- baseline, continuous-calibration, blink, Wide, gaze, or smoothing changes;
- selecting sessions or frames after seeing images or outputs;
- writing to the source recordings or production configuration.
