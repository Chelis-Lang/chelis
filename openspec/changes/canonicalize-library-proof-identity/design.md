## Context

`harden-pipeline-core-cache-and-lowering` documented the cache-identity residual (its
Option C) and named this change as the Option A fix. The obstacle, verified in that
change's design D1, is that the `LibraryProofId` cannot be recomputed from a decoded
program's own source because it is bound pre-effects and, for a layered build, is the
extension's identity over the base context while the composed program's `exprs()` are the
base++extension concatenation.

The sealed proof mechanism this touches was built and red-teamed in the pipeline-core
extraction: `LibraryProofId` has private fields and no public constructor;
`CheckedProgram::compose` returns `None` unless `new_code.context_library_proof_id ==
library.library_proof_id`; contextual lowering rejects a lowered library with another
proof identity; and `TypeEnv::matches_checked_program` requires the type-env and program
identities to agree. Every one of those invariants must still hold after the derivation
change.

## Goals / Non-Goals

**Goals:**

- The library proof identity is derived from the final accepted program's source and is
  path-independent: monolithic and layered builds of the same whole source carry the same
  identity.
- The cache decode boundary recomputes the identity from the decoded source and rejects a
  forged-but-self-consistent identity, without repeating type inference.
- Every existing sealed-proof invariant still holds, proven by a fresh red team.

**Non-Goals:**

- No hash-algorithm or domain-separation change beyond the derivation point and the
  compose canonicalization.
- No change to contextual host lowering (sibling change) or to public API/wire shapes
  beyond the cache-format-version bump.

## Decisions

### D1: Bind the identity from the final accepted program

Move the `bind_library_products` identity binding out of
`build_compiled_library_context_in_session` (type-check time) to after the effect and
linearity passes complete, so the stored identity hashes the exact `annotated_exprs` the
cached program carries. The type-env and program are bound to the same identity at that
point. `check_ir_with_signature_context` still records the context library's identity into
new code's `context_library_proof_id`, read from the (now final) library identity.

*Consequence to verify:* the effect/linearity reannotation path
(`checked_program_with_effect_annotations_in_session`) currently copies the identity
through; it must instead leave it unset until the final binding, or the binding must be
idempotent at the last stage. The `with_linearity` and effect-reannotation constructors
must not carry a stale pre-effect identity.

### D2: Canonicalize the composed identity and rebind the type-env

In `CheckedProgram::compose`, after concatenating `annotated_exprs`, set
`library_proof_id = for_library(composed.annotated_exprs, None)` — a canonical identity
over the whole composed source with a top-level (`None`) context — and
`context_library_proof_id = None`. `complete_context_library_checks` then rebinds the
composed type-env to that same identity before `bind_checked_library`, so
`matches_checked_program` holds.

*Path independence falls out:* the monolithic path already derives
`for_library(whole_exprs, None)` from the final program (after D1); the layered path now
derives `for_library(composed_exprs, None)` over the same whole source. Provided the
canonical flat print of the concatenated layered source equals the monolithic whole-source
print, the two identities are equal. The byte-identical monolithic-vs-layered acceptance
oracle is the check that this holds; if the prints differ, that is a pre-existing
composition-vs-monolithic divergence to resolve, not a new one this change introduces.

*Compose-gating stays intact:* `compose` still returns `None` unless
`new_code.context_library_proof_id == library.library_proof_id`. Chained composition (an
extension checked against a composed library) records the composed library's canonical
identity as its context; a further compose then matches against that canonical identity.

### D3: Recompute at the cache boundary

`validate_cached_library` gains the recompute the sibling change could not add: derive
`for_library(cached_program.exprs(), None)` and reject a stored identity that differs,
before constructing `CheckedLibrary`. It is a hash over already-decoded source, not a
type-inference session, so the "reruns effect and linearity without repeating type
inference" contract is preserved. Both `CompiledContext` and `StdLibContext` decode
exercise it. Replace the Option C doc note on `validate_cached_library` with the recompute
and its rationale.

### D4: Cache-format-version bump and fresh red team

The identity value changes for existing layered caches, so bump the compiler-API cache
format version and the stdlib struct-format version; a stale file is a clean miss via the
existing magic/version rejection, never a bad decode. Because this re-opens the sealed
proof mechanism, a fresh local red-team subagent must: forge a self-consistent identity
that no longer matches the decoded source and confirm decode rejects it; confirm
monolithic and layered builds of the same source now carry the same identity; and confirm
compose-gating, contextual lowering proof matches, and the facade compile-fail boundaries
still hold.

## Risks / Trade-offs

- [Re-opening the sealed proof mechanism regresses a compose-gating or lowering invariant]
  → The extraction's compile-fail fixtures, the `contextual_products_reject_a_replacement_
  library` test, and the cache negative controls are the tripwires; D4 requires a fresh red
  team on top.
- [The canonical layered identity does not equal the monolithic identity] → Surfaced by
  the byte-identical monolithic-vs-layered acceptance oracle; if it fires it is a real
  composition-vs-monolithic print divergence to fix, which this change wants to know about.
- [The cache-format bump invalidates warm caches once] → Expected and cheap; a stale file
  is a clean miss, and the bump is a one-time rebuild.

## Migration Plan

Land after `harden-pipeline-core-cache-and-lowering`. The cache-format bump invalidates
existing caches once. Rollback is a revert plus a second cache-format bump. No data or
install migration.

## Open Questions

- Whether `for_library`'s canonical flat print already ignores effect annotations. If it
  does, D1's post-effects binding is a no-op on the hash and only the composition
  canonicalization (D2) is load-bearing; if it does not, D1 is required for the monolithic
  case too. Resolve by measuring the pre-effect vs post-effect print equality before
  choosing the minimal derivation change.
