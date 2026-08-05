# Tasks: canonicalize-library-proof-identity

Follow-up (Option A) to `harden-pipeline-core-cache-and-lowering`. Not started.

## 0. Measure before choosing the minimal change

- [ ] 0.1 Determine whether `LibraryProofId::for_library`'s canonical flat print differs
  before and after effect/linearity reannotation for a representative library. Record the
  result; it decides whether D1 (post-effects binding) is load-bearing or a no-op.
- [ ] 0.2 Determine whether the canonical flat print of a layered composed source equals
  the monolithic whole-source print (the D2 path-independence premise), using the existing
  monolithic-vs-layered fixtures.

## 1. Spec-first failing tests

- [ ] 1.1 `chelis-types`: a library's stored `library_proof_id` equals
  `for_library(program.exprs(), None)` for the final accepted program (fails red until D1).
- [ ] 1.2 `chelis-types`: monolithic and layered builds of the same whole source carry the
  same `library_proof_id` (fails red until D2).
- [ ] 1.3 `chelis-types`: `CheckedProgram::compose` still returns `None` for an extension
  that records a different context identity; chained composition against a composed library
  still matches (regression guard for D2).
- [ ] 1.4 `chelis-pipeline-core`: `validate_cached_library` rejects a decoded program whose
  stored identity is self-consistent across the two copies but not equal to
  `for_library(program.exprs(), None)` (fails red until D3).
- [ ] 1.5 `chelis-compiler-api`: `CompiledContext` and `StdLibContext` decode reject a
  forged-but-self-consistent identity and still round-trip a legitimately produced cache.

## 2. Derivation change (D1, D2)

- [ ] 2.1 Move the `bind_library_products` identity binding to the final accepted program;
  stop copying a pre-effect identity through effect/linearity reannotation.
- [ ] 2.2 Canonicalize the composed identity in `CheckedProgram::compose`
  (`for_library(composed.exprs(), None)`), and rebind the composed type-env in
  `complete_context_library_checks` before `bind_checked_library`.
- [ ] 2.3 Run 1.1-1.3 green; confirm the extraction compile-fail fixtures and the
  `contextual_products_reject_a_replacement_library` test still pass.

## 3. Cache recompute (D3)

- [ ] 3.1 Add the recompute to `validate_cached_library`; replace the Option C doc note.
- [ ] 3.2 Run 1.4-1.5 green through both `CompiledContext` and `StdLibContext` decode.

## 4. Cache format bump and docs (D4)

- [ ] 4.1 Bump `CACHE_MAGIC`/`CACHE_FORMAT_VERSION` and `STDLIB_CACHE_FORMAT_VERSION`;
  update the format-version rejection tests.
- [ ] 4.2 Update `docs/investigations/compiler_pipeline_inventory.md`: the cache boundary
  now recomputes the identity from source, and monolithic/layered identities agree.

## 5. Validation and oracle

- [ ] 5.1 `cargo nextest run -p chelis-types -p chelis-pipeline-core -p chelis-compiler-api`
  and the three explicit rustdoc stages green in an isolated `CARGO_TARGET_DIR`.
- [ ] 5.2 The monolithic-vs-layered acceptance oracle green (path-independence premise).
- [ ] 5.3 `.venv/bin/python scripts/compiler_pipeline_oracle.py` green.
- [ ] 5.4 `openspec validate canonicalize-library-proof-identity --strict` passes.
- [ ] 5.5 Fresh local red-team subagent: forge a self-consistent-but-wrong identity and
  confirm decode rejects it; confirm monolithic/layered identity agreement; confirm
  compose-gating, contextual lowering proof matches, and the facade compile-fail
  boundaries still hold.
