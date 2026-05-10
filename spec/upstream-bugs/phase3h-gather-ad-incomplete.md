# phase3h-gather-ad-incomplete: Phase 3h marked shipped but Surf `gather` is not yet lowered through the sparse path

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
is narrower but still open: Surf-level `gather` does not yet lower
through the §3.5 decomposition and sparse recognizer, and HIP does not
yet have sparse `Gather` / `ScatterAdd` kernels.

- `crates/chelis-types/src/builtins.rs:135` — registered as a
  generic_triop in the type environment.
- `crates/chelis-ir/src/host.rs:4640` — host runtime dispatch for
  `gather | scatter | where | cumsum | diagonal | trace | clamp`.
- `crates/chelis-ir/src/dag.rs` — now includes first-class
  `RiscOp::Gather` and `RiscOp::ScatterAdd` implementation nodes.
- `crates/chelis-ir/src/grad.rs` — now differentiates first-class
  `Gather` into `ScatterAdd`, preserving duplicate-index
  accumulation.
- `crates/chelis-backend-c/src/emit.rs` — now emits bounded sparse C
  loops for first-class `Gather` and `ScatterAdd`.

Net: the lower-level sparse IR/C path is in place and tested, but
ordinary Surf programs that call the `gather` builtin still use the
host lane rather than the sparse IR lowering. The cross-library AD
claim is therefore still not closed for user-authored `gather`
programs, even though the first-class sparse IR node now has the
required AD and C backend behavior.

## Why this matters

Two failure modes downstream:

1. **MoE / embedding layers are not yet on the user-facing sparse AD
   path.** The natural way to express embedding lookup
   (`gather(table, token_indices, axis=0)`) and Mixture-of-Experts
   routing (`gather(expert_weights, expert_indices, axis=0)`) still
   needs Surf lowering into the sparse IR node before the shipped
   `Gather -> ScatterAdd` adjoint is available to ordinary programs.
   `Std.Nn.Embedding` is documented as the user-facing surface over
   `gather` (`spec/design/chelis_phase3_plan.md:326,1129`), so the
   Embedding layer is similarly affected until that lowering lands.

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

The remaining fix is two-step and must land together to avoid arming
the OOM trap:

1. **Wire Surf `gather` into the §3.5 lowering path.** The lowering
   may emit the dense `reshape + expand + mul + sum` contract shape
   for recognizer input, or lower directly to the first-class
   `RiscOp::Gather` node when the same structural contract is
   preserved.
2. **Add the dense-shape recognizer and HIP sparse codegen.** The
   recognizer belongs in `crates/chelis-ir/src/specialize.rs`, alongside
   the existing BLAS specialization substrate. It must replace the
   §3.5 dense shape with `RiscOp::Gather` / `RiscOp::ScatterAdd` before
   DCE and backend codegen. The C backend already has bounded sparse
   emission for the first-class nodes; HIP must gain equivalent kernels
   or continue to reject with a diagnostic that names the missing sparse
   kernel / duplicate-index scatter semantics.

The adversarial duplicate-index test
(`gather_via_section_3_5_lowering_accumulates_duplicate_indices` in
`crates/chelis-ir/tests/grad_gather_contract.rs`) now exercises the
first-class sparse `Gather` adjoint and asserts the duplicate-index
gradient is correctly accumulated. The remaining requirement is that
Surf lowering and the recognizer route user-level programs to that
path without materializing the dense one-hot product.

Acceptance must be structural, not an attempted 200 GB runtime probe:

- C and HIP build output for an embedding/MoE-shaped Surf fixture must select a
  bounded sparse gather/scatter kernel path.
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
When Surf lowering and HIP sparse kernels land, these tests must remain
green and new Surf-level structural tests should prove the dense
`[N, V, D]` allocation is absent.
