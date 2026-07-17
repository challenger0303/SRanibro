# Phase 1.3 amendment 2: frozen complete renderer-plan identity

Status: frozen before any Phase 1.3 model inference.

The original preregistration and amendment 1 remain the scientific, renderer,
and artifact contracts.  A final pre-inference adversarial audit found that a
metadata-only confirmation could validate the renderer-plan hash, fixed
top-level identities, pair count, and case count without proving that every
nested selected recipe, canonical hash, tensor hash, and case record was the
exact renderer-only plan produced by the frozen implementation.

This amendment closes only that artifact-validation gap.  It changes no atlas,
model, renderer, aperture pair, target, estimand, threshold, response class, or
real-recording exclusion.

## Frozen complete plan hash

Before model bytes were ever read, the committed renderer preparation was run
twice against the exact sealed Phase 1.3A atlas.  The two preparations were
bit-identical and produced the already specified domain-separated plan hash:

```text
d9312bd9434c25cab044977ffb96c3fea92fbdfa4beb12327f840f1b79552f42
```

The hash input remains exactly:

```text
sranibro-synthetic-eye-phase13-plan-v1\0 || renderer_plan.json
```

No EyeNet model or real recording was read while deriving this value.

For every renderer-GO decision artifact, including artifact, recognition,
insensitive, or generic inconclusive decisions:

1. `renderer_plan_sha256` must equal the frozen hash above;
2. recomputing the domain-separated hash over the exact inventoried
   `renderer_plan.json` bytes must equal the same frozen hash;
3. the existing manifest, plan-identity, case-order, raw-result, and model-state
   cross-checks remain mandatory.

A self-consistent plan with any changed nested recipe, selected target,
canonical image hash, tensor hash, or case record therefore remains invalid
even if its manifest and internal plan hash were rewritten together.  It is an
invalid/tampered decision input: confirmation writes no artifact and reads no
model bytes.

A renderer-NO-GO decision continues to carry a null renderer-plan hash and no
renderer-plan file.  For a conclusive decision, confirmation still prepares
the all-30 plan twice and requires exact plan bytes in addition to the frozen
hash.

Decision execution must also require its freshly prepared renderer-plan hash
to equal the frozen value before model bytes are read.

## Provenance and implementation fingerprint

Decision and confirmation manifests record this amendment commit in addition
to the original preregistration and amendment 1 commits.  All three must be
ancestors of the clean implementation commit.

The exact implementation fingerprint is amendment 1's list with this file
inserted immediately after amendment 1:

```text
Cargo.toml
Cargo.lock
build.rs
research/synthetic-eye-lab/PHASE1_3_PREREG.md
research/synthetic-eye-lab/PHASE1_3_AMENDMENT1.md
research/synthetic-eye-lab/PHASE1_3_AMENDMENT2.md
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

The fingerprint encoding, atomic publication procedure, external decision
seal, total confirmation procedure, and all scientific rules are unchanged.
