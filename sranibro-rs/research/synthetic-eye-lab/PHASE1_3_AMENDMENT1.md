# Phase 1.3 amendment 1: external decision seal and total confirmation

Status: frozen before any Phase 1.3 model inference.

The original Phase 1.3 preregistration remains the scientific and renderer
contract. A pre-inference implementation audit found that hashing the decision
manifest without requiring that hash at confirmation did not protect the
primary class from a later manifest-only edit. It also found that a
confirmation-side renderer failure did not reach the preregistered total
confirmation procedure.

This amendment changes only artifact sealing, confirmation input, and failure
handling. It does not change the atlas, model, renderer, aperture pairs,
targets, estimands, thresholds, response rubric, or real-recording exclusion.

## External decision seal

After atomically publishing a decision artifact, the decision command prints:

```text
Decision manifest SHA-256: <64 lowercase hexadecimal characters>
```

The hash is over the exact completed `manifest.json` bytes. It is not stored
inside that manifest because a file cannot contain its own raw hash.

The confirmation CLI is amended to require the printed value:

```text
synthetic-eye-phase13 confirmation \
  --atlas <sealed-atlas-directory> \
  --model <same-EyeNet-params-file> \
  --decision <sealed-decision-directory> \
  --decision-seal <64-hex-sha256> \
  --out <new-directory>
```

The option is required exactly once for confirmation and prohibited for
decision. Uppercase, non-hexadecimal, or non-64-character values are rejected;
the CLI normalizes no value. Before trusting any decision field, preparing the
renderer, or reading model bytes, confirmation hashes the current decision
manifest and requires exact string equality with `--decision-seal`.

A missing or mismatched external seal is an invalid input, not a new scientific
result. No confirmation artifact is written.

## Decision internal consistency

After the external seal and file inventory pass, confirmation independently
parses the inventoried decision files and requires:

- `analysis.json` stage, terminal status, response class, raw-bit arrays, and
  model/artifact state agree with the corresponding manifest fields;
- `raw_bits.json` exactly agrees with the two raw-bit arrays in
  `analysis.json`;
- `renderer_plan.json` has the manifest's domain-separated plan hash whenever
  the decision reached renderer GO;
- `stage_cases.json` count and ordered case identities agree with the manifest
  and renderer plan;
- every declared atlas, model, repository, build, preregistration, amendment,
  and implementation identity agrees with the current confirmation inputs.

Any inconsistency is an invalid/tampered decision input. No confirmation
artifact is written and no model is loaded.

## Total confirmation procedure

The original ordered confirmation status is retained and made executable:

1. A valid sealed decision whose status is renderer, artifact, recognition,
   insensitive, or generic inconclusive cannot be rescued. Confirmation writes
   a sealed metadata-only artifact with
   `confirmation_status = CONFIRMATION_INCONCLUSIVE`, records the external
   decision seal, and performs no renderer search or model inference.
2. For a conclusive decision, confirmation independently validates the atlas
   and prepares the all-30 renderer plan twice. Any confirmation-side atlas or
   renderer failure is sealed with terminal renderer status,
   `confirmation_status = CONFIRMATION_INCONCLUSIVE`, the external decision
   seal, and `model_loaded = false`.
3. After a confirmation renderer GO, any artifact, recognition, insensitive,
   or generic inconclusive confirmation result yields
   `CONFIRMATION_INCONCLUSIVE`.
4. Otherwise identical conclusive response classes yield `REPLICATED`; two
   different conclusive classes yield `NOT_REPLICATED`.

Every confirmation artifact records both the externally supplied decision seal
and the recomputed identical decision manifest SHA-256.

## Publication revalidation

For both stages, non-manifest files are written and flushed, then reopened from
the staging directory to create the sorted path/length/SHA-256 inventory. After
`manifest.json` is written and flushed last, the staging tree is walked again.
It must contain exactly the inventoried files plus `manifest.json`; every
non-manifest file is reopened and must still match its declared length and
hash. Extra files, links, special entries, byte changes, and temporary files
are errors. The staging directory and its parent are synced where the platform
supports directory sync before and after the atomic rename.

## Amended implementation fingerprint

The exact source fingerprint list is the original list with this amendment
inserted immediately after the original preregistration:

```text
Cargo.toml
Cargo.lock
build.rs
research/synthetic-eye-lab/PHASE1_3_PREREG.md
research/synthetic-eye-lab/PHASE1_3_AMENDMENT1.md
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

The length encoding, path encoding, ordering, and SHA-256 procedure remain
unchanged. Decision and confirmation manifests record both the original
preregistration commit and this amendment commit. Both must be ancestors of the
clean implementation commit.
