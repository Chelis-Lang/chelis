## Context

`spec/06-transformations.md` (v0.1.0-draft, authoritative) defines the DAG-to-DAG
transformations `grad` (Phase 0), `vmap` and `jit` (Phase 2, semantics specified now), the
optimization passes over RISC DAGs, the composition/commutativity rules, and a formal
reverse-mode AD semantics. This change records that content as a `transformations` capability.

## Goals / Non-Goals

**Goals:**
- Capture `grad`/`vmap`/`jit` semantics and the optimization/composition/error rules as SHALL
  requirements with positive and failure scenarios.
- Preserve the semantics-preservation invariant (`eval(optimize(G), x) = eval(G, x)`) as a
  first-class requirement.

**Non-Goals:**
- Reproducing every adjoint rule (owned by `risc-primitives`) or every algebraic-simplification
  rewrite; the requirement captures the governing algorithm and the barrier/fixpoint rules.
- Restating the full pseudocode of §7.2 `build_adjoint`; the requirement captures the
  init/seed/reverse-traversal/accumulate structure and the acyclicity guarantee.

## Decisions

- Split `grad` into several requirements (signature, `wrt`, algorithm/accumulation,
  non-differentiable handling, symbolic-dim adjoints, higher-order, static match/if, field-wise
  ADT) because each has its own distinct positive and failure behavior.
- Keep the shipped-subset boundaries (`grad(vmap(f))` rejected, ADT compiled-lane export
  rejected, host-lane AD boundary idioms) inside the relevant requirement's failure scenario
  rather than as separate requirements.

## Risks / Trade-offs

- [Phase status vs shipped subset] → `vmap`/`jit` are Phase 2 with semantics specified now, and
  parts of `grad` (ADT D2 slice, static-if pruning) are shipped slices with named limits. The
  requirements capture the specified semantics and the named rejections; the exact shipped-vs-
  future boundary is recorded in Open Questions.

## Open Questions

- `jit` is specified for forward compatibility but "remains non-executable today" per the
  composition table. Verified against code: `jit` does evaluate — it type-checks and passes
  through as a transparent no-op at eval (`crates/chelis-types/src/infer.rs:10075`,
  `chelis-cli/tests/jit_par_runtime_gap.rs`, `chelis-types/tests/jit_par_passthrough.rs`). What
  is not implemented is the shape-specialized caching/compilation, not `jit` blocking
  evaluation. The requirement records the specified caching/transparency semantics; the missing
  caching is a roadmap matter, not a spec ambiguity.
- General scalar host-lane AD (beyond the enumerated boundary idioms) is deferred to Phase 5
  (`phase5_host_scalar_ad.md`); the requirement records the supported tensor-lane path and the
  rejection of unsupported host-lane scalar AD.
