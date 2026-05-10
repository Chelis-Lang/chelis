# phase3h-gather-ad-incomplete: Phase 3h gather AD/sparse path closure notes

**Status:** **CLOSED FOR SCOPED SPARSE PATH**
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
host-only forward-evaluating builtin. The current branch closes the scoped
sparse path: tensor-lane Surf `gather` lowers directly to first-class sparse
IR, the internal dense §3.5 `OneHot + Expand + Mul + Sum` tag tree is
recognized before codegen, and C/HIP backends emit bounded sparse paths for
supported dtypes.

- `crates/chelis-types/src/builtins.rs:135` — registered as a
  generic_triop in the type environment.
- `crates/chelis-ir/src/lower.rs` — tensor-lane `gather(values,
  indices, axis)` now lowers to `RiscOp::Gather` instead of the host
  runtime call.
- `crates/chelis-ir/src/host.rs` — host runtime dispatch still covers
  scatter/where/cumsum/diagonal/trace/clamp and non-tensor host paths.
- `crates/chelis-ir/src/dag.rs` — now includes first-class
  `RiscOp::Gather` and `RiscOp::ScatterAdd` implementation nodes plus the
  internal-only `RiscOp::OneHot { vocab }` recognition anchor.
- `crates/chelis-ir/src/grad.rs` — now differentiates first-class
  `Gather` into `ScatterAdd`, preserving duplicate-index
  accumulation.
- `crates/chelis-ir/src/specialize.rs` — recognizes the internal dense
  §3.5 gather tree and replaces it with `RiscOp::Gather` before DCE/codegen.
- `crates/chelis-backend-c/src/emit.rs` — now emits bounded sparse C
  loops for first-class `Gather` and `ScatterAdd`.
- `crates/chelis-backend-hip/src/emit.rs` and `src/kernels.rs` — now emit
  f32 sparse gather/scatter-add kernels with int32/int64 indices. ScatterAdd
  uses `atomicAdd`, so f64 scatter-add remains explicitly rejected.

Net: the sparse IR/C/HIP path is in place and tested for tensor-lane Surf
`gather`, including the first-class `Gather -> ScatterAdd` adjoint. The dense
recognizer is scoped to the internal `OneHot` tag tree because arbitrary
historical const/eq one-hot encodings do not preserve the source index operand.

## Why this matters

Two failure modes downstream:

1. **Future risk: shipping an unanchored §3.5 lowering naively re-arms an OOM
   trap.** `spec/05-risc-primitives.md` §3.5 says
   `gather(x, idx, axis)` decomposes via `reshape + expand + mul +
   sum` (a one-hot-then-matmul pattern). Future producers must use
   internal `RiscOp::OneHot` so the recognizer can recover the indices and
   collapse before backend codegen.

## Closure Notes

The scoped closure is:

1. Tensor-lane Surf `gather` lowers directly to sparse `RiscOp::Gather`.
2. AD rewrites `Gather` to `ScatterAdd`, preserving duplicate-index
   accumulation.
3. The shared specialization pass recognizes the internal dense §3.5
   `OneHot + Expand + Mul + Sum` gather tree and replaces it with `Gather`
   before DCE/codegen.
4. C and HIP codegen emit sparse bounded kernels for supported dtypes.

The adversarial duplicate-index test
(`gather_via_section_3_5_lowering_accumulates_duplicate_indices` in
`crates/chelis-ir/tests/grad_gather_contract.rs`) now exercises the
first-class sparse `Gather` adjoint and asserts the duplicate-index
gradient is correctly accumulated.

Acceptance must be structural, not an attempted 200 GB runtime probe:

- C/HIP build output for an embedding/MoE-shaped sparse fixture must select the
  bounded sparse gather/scatter path.
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
routed to vocab 0, 0 to vocab 1). Additional coverage now includes
`chelis_ir::specialize` dense recognizer tests, C bounded emitted-code and
compile/run tests, HIP structural tests, and ignored HIP GPU correctness tests
for sparse gather plus duplicate-index scatter-add.
