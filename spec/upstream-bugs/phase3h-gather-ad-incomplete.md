# phase3h-gather-ad-incomplete: Phase 3h marked shipped but dense recognizer and HIP sparse gather remain incomplete

**Status:** **PARTIALLY ADDRESSED**
**Filed:** 2026-05-08
**Owning phase:** Phase 3h (core numeric primitives)
**Discovered by:** Adversarial test for cross-library AD claim
(`crates/chelis-ir/tests/grad_gather_contract.rs`,
`docs/identified_gaps.md` Gap 3)

## Summary

`spec/12-roadmap.md` marks Phase 3h as **shipped** ("In progress;
`3a`, `3b`, `3b-ii`, `3c`, `3d`, `3e`, `3g`, `3h`, `3i`, and `3m`
are shipped"). Phase 3h's named scope is "core numeric primitives
such as einsum, concat / split, gather / scatter, where, cumsum,
sort, diagonal / trace, clamp."

This bug was originally filed when `gather` existed only as a
host-only forward-evaluating builtin. The remaining user-visible gap
is narrower but still open: tensor-lane Surf `gather` now lowers
directly to first-class sparse IR, but the dense §3.5 decomposition
recognizer and HIP sparse `Gather` / `ScatterAdd` kernels are still
missing.

- `crates/chelis-types/src/builtins.rs:135` — registered as a
  generic_triop in the type environment.
- `crates/chelis-ir/src/lower.rs` — tensor-lane `gather(values,
  indices, axis)` now lowers to `RiscOp::Gather` instead of the host
  runtime call.
- `crates/chelis-ir/src/host.rs` — host runtime dispatch still covers
  scatter/where/cumsum/diagonal/trace/clamp and non-tensor host paths.
- `crates/chelis-ir/src/dag.rs` — now includes first-class
  `RiscOp::Gather` and `RiscOp::ScatterAdd` implementation nodes.
- `crates/chelis-ir/src/grad.rs` — now differentiates first-class
  `Gather` into `ScatterAdd`, preserving duplicate-index
  accumulation.
- `crates/chelis-backend-c/src/emit.rs` — now emits bounded sparse C
  loops for first-class `Gather` and `ScatterAdd`.

Net: the sparse IR/C path is in place and tested for tensor-lane Surf
`gather`, including the first-class `Gather -> ScatterAdd` adjoint.
The remaining gap is no longer the ordinary C path; it is the dense
§3.5 recognizer and HIP parity.

## Why this matters

Two failure modes downstream:

1. **HIP MoE / embedding layers are not yet on a sparse backend path.**
   The natural C path for embedding lookup
   (`gather(table, token_indices, axis=0)`) now lowers to sparse IR, but
   HIP still rejects those nodes rather than compiling kernels with
   integer index tensors and duplicate-index scatter accumulation.

2. **Future risk: shipping the §3.5 lowering naively re-arms an OOM
   trap.** `spec/05-risc-primitives.md` §3.5 says
   `gather(x, idx, axis)` decomposes via `reshape + expand + mul +
   sum` (a one-hot-then-matmul pattern). If a future change wires
   that lowering without also adding a scatter-recognition
   pattern-matcher, an embedding lookup over a `[V=50K, D=1K]` table
   for `N` tokens materializes an `[N, V, D]` intermediate (~200 GB
   for typical LLM-scale shapes). The AD side is then automatically
   correct (verified empirically), but the forward path OOMs.

## Closure plan

The remaining fix is two-step:

1. **Add the dense-shape recognizer.** The
   recognizer belongs in `crates/chelis-ir/src/specialize.rs`, alongside
   the existing BLAS specialization substrate. It must replace the
   §3.5 dense shape with `RiscOp::Gather` / `RiscOp::ScatterAdd` before
   DCE and backend codegen when any producer emits the dense contract
   shape.
2. **Add HIP sparse codegen.** The C backend already has bounded sparse
   emission for the first-class nodes; HIP must gain equivalent kernels
   plus integer index tensor support, or continue to reject with a
   diagnostic that names the missing sparse kernel / duplicate-index
   scatter semantics.

The adversarial duplicate-index test
(`gather_via_section_3_5_lowering_accumulates_duplicate_indices` in
`crates/chelis-ir/tests/grad_gather_contract.rs`) now exercises the
first-class sparse `Gather` adjoint and asserts the duplicate-index
gradient is correctly accumulated. The remaining requirement is that
the dense recognizer and HIP backend route all supported user-level
programs to that path without materializing the dense one-hot product.

Acceptance must be structural, not an attempted 200 GB runtime probe:

- C build output for an embedding/MoE-shaped Surf fixture must select the
  bounded sparse gather/scatter path; HIP must do the same once its sparse
  backend support exists.
- The generated code must not allocate the dense `[N, V, D]`
  one-hot/materialized product buffer.
- A small fixture is sufficient as long as emitted-code inspection proves
  the allocation shape is sub-linear in vocabulary size and duplicate-index
  accumulation still matches the contract test.

## Why this isn't already filed

The Phase 3h plan
(`spec/design/chelis_phase3_plan.md` "Phase 3h: Core Tensor
Primitives") lists the primitives but does not enumerate
"AD adjoint shipped" or "scatter recognizer shipped" as gating
criteria. The acceptance oracle is "a pure Chelis model program
can [express the new primitives in forward computation]" — which the
host-lane `gather` satisfied. The AD and sparse-codegen gates were
implicit and slipped.

## Regression test

`crates/chelis-ir/tests/grad_gather_contract.rs` —
`gather_via_section_3_5_lowering_accumulates_duplicate_indices`.
Builds the sparse IR DAG and asserts the gradient at the embedding
table is `[3, 3, 0, 0]` for an all-zero indices stress case (3 tokens
routed to vocab 0, 0 to vocab 1). The C backend also has a bounded
emitted-code test and a compiled numerical sparse gather/scatter test.
Surf-level C structural tests now prove the host runtime call is absent;
when HIP sparse kernels land, equivalent HIP structural/numerical tests
should prove the dense `[N, V, D]` allocation is absent.
