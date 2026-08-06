## Why

This is the Option A follow-up to `harden-pipeline-core-cache-and-lowering`, which
documented (Option C) a residual it could not safely fix: the cache decode boundary
(`validate_cached_library`) checks that the two stored copies of a `LibraryProofId` agree,
but never recomputes the identity from the decoded source. Two mechanisms make a
source-recompute impossible at the current derivation:

1. **Pre-effect binding.** `bind_library_products` derives the identity at type-check time
   via `LibraryProofId::for_library(checked.exprs(), context)`, where `exprs()` returns
   `annotated_exprs`. Effect and linearity passes then produce the cached program, which
   *preserves* the identity but rewrites `annotated_exprs`. The stored identity therefore
   hashes different source than the cached program carries.
2. **Composition is lossy and path-dependent.** `CheckedProgram::compose` concatenates
   `library.annotated_exprs ++ new_code.annotated_exprs` but sets
   `library_proof_id = new_code.library_proof_id` (`= for_library(extension_exprs,
   base_id)`) and `context_library_proof_id = None`. The stored identity is over the
   extension's source with the base context, while `exprs()` is the concatenation and the
   recorded context is `None`. Composition discards the base/extension split the identity
   depends on. A side effect is that monolithic and layered builds assign *different*
   identities to the same whole source.

A canonical, post-effects, path-independent identity closes both, and then the cache
boundary can recompute it and reject a forged-but-self-consistent identity.

## What Changes

- Bind the `LibraryProofId` from the final semantically accepted program (after effects
  and linearity), so the stored identity hashes the exact `annotated_exprs` the cached
  program carries.
- In `CheckedProgram::compose`, recompute the composed library's identity canonically as
  `for_library(composed.exprs(), None)` (path-independent), and rebind the composed type
  environment to the same identity so `matches_checked_program` still holds. Monolithic
  and layered builds of the same whole source then carry the same identity.
- Preserve the compose-gating contract: a new-code program's `context_library_proof_id`
  still records the exact library identity it was checked against, and contextual lowering
  still rejects a lowered library with another proof identity.
- Add the cache-decode recompute: `validate_cached_library` recomputes
  `for_library(cached.exprs(), None)` and rejects a stored identity that does not match,
  for both `CompiledContext` and `StdLibContext`, without repeating type inference.
- Bump the compiler-API cache format version (the identity value changes for existing
  layered caches; a stale file is a clean miss, never a bad decode).
- Re-verify every sealed proof invariant and run a fresh red team.

### Non-Goals

- Do not change the `LibraryProofId` hash algorithm or domain separation beyond the
  derivation point and the compose canonicalization.
- Do not change public API shapes, wire schemas, diagnostics, or generated code.
- Do not change the contextual host-lowering behavior (owned by the sibling change).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `compiler-pipeline-architecture`:
  - Add "Canonical library proof identity" — the identity is derived from the final
    accepted program and is path-independent; composition recomputes it canonically.
  - Add "Cache decode recomputes the library proof identity" — the parser recomputes and
    verifies the identity from the decoded source for both cache types.

## Impact

- `crates/chelis-types` — `LibraryProofId` binding point, compose canonicalization, and
  type-env rebind; the compose-gating and `matches_checked_program` invariants.
- `crates/chelis-pipeline-core/src/semantic.rs` — `validate_cached_library` recompute;
  replaces the Option C doc note with the recompute.
- `crates/chelis-compiler-api` — cache format version bump in `context.rs` and
  `stdlib_cache.rs`; decode negative controls for a forged-but-self-consistent identity.
- A fresh red-team pass against the sealed proof mechanism.

This change hardens implementation integrity and re-opens the sealed proof invariants, so
it carries its own adversarial validation. The numbered specifications retain authority
for language, compiler, serialization, backend, runtime, and package behavior.
