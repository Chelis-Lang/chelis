## Why

Review of the sealed pipeline core found two residual hardening gaps behind the
proof-binding seal. Implementation confirmed one is a clean fix and the other is blocked
by the identity's derivation, needing a maintainer decision.

1. **Contextual host lowering (fixed here).** Isolated lowering accepts a nonfatal lower
   rejection under `AllowHostBackend` and yields an empty host result
   (`finish_isolated_lowering`), but contextual lowering (`lower_checked_with_context`)
   had no matching arm: the same input returned a lowering error. The sole current caller
   passes `AllowHostOnly`, so the divergence is latent, but it would silently disagree
   with the isolated path if a future caller lowered context code under a selected host
   backend.

2. **Cache proof recompute (resolved via Option C; Option A tracked as a follow-up).**
   `validate_cached_library` reruns effect and linearity checks and checks that the
   decoded `TypeEnv` and `CheckedProgram` agree on a `LibraryProofId`, but it never
   recomputes that identity from the decoded source. Implementation revealed that
   recompute-from-source is not feasible without changing how the identity is derived:
   `LibraryProofId::for_library` hashes `annotated_exprs`, but the identity is bound at
   type-check time and the cached program is the post-effect, post-linearity program whose
   `annotated_exprs` differ; and a layered build's cached library is a **composed**
   program whose stored id is the extension's id (`for_library(extension_exprs, base_id)`)
   while its `exprs()` are the full base++extension concatenation - the base/extension
   split composition discards. A naive recompute would therefore reject legitimate caches.
   This change takes **Option C**: document that the stored id is a self-consistency
   check, not a source-recompute, at the owning function and in the current-state
   inventory. The canonical post-effects derivation that would let the boundary recompute
   the id (**Option A**) is tracked as the `canonicalize-library-proof-identity`
   follow-up. See the design for the two mechanisms and the options.

## What Changes

- Give contextual lowering the same `AllowHostBackend` nonfatal-rejection acceptance as
  isolated lowering, so a nonfatal lower rejection under a selected host backend yields an
  empty host result on both paths. `AllowHostOnly` keeps its stricter tensor-name guard on
  both paths.
- Extract the contextual root-binding tail into a testable
  `finish_contextual_lowering` helper (mirroring `finish_isolated_lowering`) and add
  positive and negative tests driving the arm directly.
- Document that the stored library proof identity is a self-consistency check, not a
  source-recompute (Option C), on `validate_cached_library` and in the current-state
  inventory. Do not ship a recompute that would reject legitimate caches; track the
  canonical-derivation fix (Option A) as the `canonicalize-library-proof-identity`
  follow-up change.

### Non-Goals

- Do not change language, compiler, CLI, backend, runtime, package, or generated-code
  behavior on the accepted path.
- Do not narrow the isolated `AllowHostBackend` behavior; make contextual match it.
- Do not change the `LibraryProofId` derivation, its domain separation, or the
  composition path in this change. Any recompute that requires those changes is deferred
  to the decision this change records.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`:
  - Add "Contextual host lowering matches isolated host lowering" — contextual lowering
    applies the same nonfatal-rejection host policy as isolated lowering.

## Impact

- `crates/chelis-pipeline-core/src/lower.rs` — contextual lowering gains the
  `AllowHostBackend` nonfatal arm through a shared `finish_contextual_lowering` helper;
  positive and negative tests.
- `crates/chelis-pipeline-core/src/semantic.rs` gains a doc note on
  `validate_cached_library`; `docs/investigations/compiler_pipeline_inventory.md` records
  the cache-identity residual (Option C).
- `openspec/specs/compiler-pipeline-architecture/spec.md` (delta).
- No behavior change to the cache decode paths and no `chelis-types` change here; the
  canonical-derivation fix is the `canonicalize-library-proof-identity` follow-up.

This change hardens implementation consistency only. The numbered specifications retain
authority for language, compiler, serialization, backend, runtime, and package behavior.
