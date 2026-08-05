## Context

The seal refactor made `chelis_ir::lower::LoweredLibrary` immutable, added a private
`LibraryProofId` (a SHA-256 over the canonical flat print of the accepted checked source
plus a domain-separated base identity), and gated `CheckedProgram::compose` on a
proof-identity match. Cache decode was rebuilt to rerun effect and linearity checks and
re-lower the library, then use the freshly re-lowered payload rather than the wire one.
Review confirmed the seal is sound for its stated goal — safe dependent code cannot forge
or transplant a proof through the public API — and that the pipeline-core suite (22 lib
tests, 9 compile-fail doctests) is green.

Two residual gaps remain.

1. `validate_cached_library` checks `cached_type_env.matches_checked_program(&program)`
   before and after the rerun. `matches_checked_program` requires the stored `TypeEnv`
   and `CheckedProgram` proof identities to be equal to **each other** and the declared
   type maps to match. It never recomputes the identity from the program's own source.
   Because `LibraryProofId` and the checked types derive `serde::Deserialize`, a cache
   blob can carry any identity value; if both stored copies carry the same forged value
   and the program still passes the rerun, decode accepts it. Within the trusted-local-
   disk threat model this is not exploitable (an attacker with cache write access can do
   worse, and the envelope `source_hash`/`CacheIdentity` guards the source-to-cache
   mapping), but the boundary should verify the identity it stores.

2. In `finish_isolated_lowering`, `AllowHostBackend` accepts *any* nonfatal lower
   rejection and yields an empty host result; `AllowHostOnly` accepts a nonfatal
   rejection only when there are no tensor root names. `lower_checked_with_context` has
   the `AllowHostOnly` arm and the `AllowHostBackend` *success*-empty arm (line 157) but
   no `AllowHostBackend` *nonfatal-rejection* arm, so a nonfatal rejection under
   `AllowHostBackend` returns an error there. The only contextual caller
   (`compile_new_source_in_context` in `crates/chelis-compiler-api/src/compiler.rs`)
   passes `AllowHostOnly`, so this is latent.

## Goals / Non-Goals

**Goals:**

- Contextual lowering applies the same `AllowHostBackend` nonfatal-rejection host policy
  as isolated lowering, and the two paths agree for equivalent inputs.
- The cache-proof-recompute obstacle is recorded with reproduction detail and options so a
  maintainer can decide the derivation change it needs.

**Non-Goals:**

- No `LibraryProofId` derivation change, no composition-path change, and no new on-disk
  field in this change.
- No narrowing of isolated `AllowHostBackend`; contextual is made to match it.
- No change to the trusted-disk threat model or the envelope identity guards.

## Decisions

### D1: Cache proof recompute is blocked at the current identity derivation

The intended fix — `validate_cached_library` recomputes the `LibraryProofId` from the
decoded source and rejects a mismatch — is **not** implementable without changing how the
identity is derived. Two independent mechanisms make a decoded program's own fields
insufficient to reproduce its stored identity:

1. **Pre-effect binding vs post-effect cache.** `bind_library_products` derives the id at
   type-check time via `LibraryProofId::for_library(checked.exprs(), context)`, where
   `exprs()` returns `annotated_exprs`. The pipeline then runs effect and linearity
   checks, which produce new programs that *preserve* the id but rewrite
   `annotated_exprs` (effect annotations are added). The cached `CheckedProgram` is the
   post-effect, post-linearity program, so `for_library(cached.exprs(), ...)` hashes
   different source than the stored id was computed over.
2. **Composition is lossy and path-dependent.** A layered build's cached library is a
   *composed* program: `CheckedProgram::compose` concatenates
   `library.annotated_exprs ++ new_code.annotated_exprs` but sets
   `library_proof_id = new_code.library_proof_id`
   (`= for_library(extension_exprs, base_id)`) and `context_library_proof_id = None`. So
   the stored id is over the extension's exprs with the base context, while `exprs()` is
   the full concatenation and the recorded context is `None`. The base/extension split the
   id depends on is discarded by composition, so no recompute over the composed program's
   own fields can reproduce it. (This also means the monolithic and layered paths assign
   *different* ids to the same whole source; the seal only requires the id be internally
   consistent, which it is.)

Either mechanism alone makes a naive `for_library(cached.exprs(), cached.context)`
recompute reject a legitimately produced cache. A correct recompute needs one of the
following derivation changes, each of which touches the freshly-sealed proof mechanism and
wants a maintainer decision:

- **Option A — derive the id after effects/linearity, canonically over the composed
  program.** Move the binding to the final program and, in `compose`, recompute
  `library_proof_id = for_library(composed.exprs(), None)` (a canonical, path-independent
  identity), rebinding the type-env to match. Makes the id recomputable and makes
  monolithic and layered agree, but must re-verify every compose-gating and
  contextual-lowering proof invariant and the layered acceptance oracle.
- **Option B — recompute only for `StdLibContext`.** The chelis-std sub-context is always
  monolithic (`check_prepared_library` → `build_compiled_library_context`, context
  `None`), so it is the root of trust and *could* be recomputable — but only if
  mechanism 1 is also resolved (its cached program is still post-effect). Partial, and
  leaves the composed `CompiledContext` uncovered.
- **Option C — accept the residual and document it.** The cache boundary already reruns
  effects and linearity, compares the two stored ids for agreement, re-lowers, and
  compares the payload field-by-field; the envelope `source_hash`/`CacheIdentity` guards
  the source→cache mapping. Within the trusted-local-disk threat model the un-recomputed
  id is defense-in-depth only. Document that the stored id is a self-consistency check,
  not an integrity recomputation, and close the finding.

This change takes **Option C**: it documents the residual on `validate_cached_library`
and in the current-state inventory, and ships D2. **Option A** is tracked as the
`canonicalize-library-proof-identity` follow-up change, which will move the binding
post-effects, canonicalize the composed identity, rebind the type-env, add the cache
recompute, and re-run a fresh red team against the sealed proof invariants. The
`context_decode_rejects_a_changed_lowered_payload_with_the_same_identity` negative control
already proves the payload leg is verified.

### D2: Contextual lowering gains the `AllowHostBackend` nonfatal arm

Add the missing match arm to `lower_checked_with_context` so a nonfatal lower rejection
under `AllowHostBackend` binds `(library.dag().clone()-or-empty, BTreeSet::new(),
accepted_nonfatal_rejection = true)` exactly as the isolated path does, then flows through
`RootBindingMode::AcceptedNonfatalRejection` to an empty `NamedRoots`. The two paths then
agree: under `AllowHostBackend`, a nonfatal lower rejection is an empty host result on
both isolated and contextual lowering.

*Why match isolated rather than narrow it:* `AllowHostBackend` means the facade has
committed to the host backend, which owns the output; any nonfatal lowering failure is
acceptable because nothing downstream needs the DAG. That is the permissive contract the
isolated path already encodes and the "Typed and aligned root artifacts" requirement
already states for isolated lowering; contextual lowering should not be stricter for the
same declared policy.

*Latency note:* no current caller reaches this arm (the contextual caller uses
`AllowHostOnly`), so this is a consistency and future-robustness fix, not a live bug fix.
The test drives the arm directly through `lower_checked_with_context`.

## Risks / Trade-offs

- [The contextual arm changes behavior for a future caller] → It makes contextual match
  the documented isolated policy; a parity test pins the agreement so a later divergence
  fails loudly. No current caller reaches the arm, so shipped behavior is unchanged.
- [Deferring the cache recompute leaves a defense-in-depth gap] → Within the
  trusted-local-disk threat model the stored id is already cross-checked (two stored
  copies agree, effects/linearity rerun, payload re-lowered and compared) and the
  envelope guards the source→cache mapping. The gap is that the id is not tied to the
  decoded source; closing it needs a derivation change (D1 Option A) that is out of scope
  for a latent-consistency change.

## Migration Plan

No data or install migration. Lowering gains one match arm behind a shared helper;
rollback is a revert; no persisted state changes shape. The cache decode paths are
unchanged in this change.

## Open Questions

- Resolved: the cache proof recompute takes Option C here (document the residual) and
  Option A (canonical post-effects identity) is tracked as the
  `canonicalize-library-proof-identity` follow-up change, which closes the gap for the
  composed `CompiledContext` and carries its own fresh red-team pass.
