# phase3h-gather-ad-incomplete: Phase 3h marked shipped but `gather` is host-only with no AD adjoint

**Status:** **OPEN**
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
sort, diagonal / trace, clamp." Empirically, however, `gather`
ships only as a **host-only forward-evaluating builtin**:

- `crates/chelis-types/src/builtins.rs:135` — registered as a
  generic_triop in the type environment.
- `crates/chelis-ir/src/host.rs:4640` — host runtime dispatch for
  `gather | scatter | where | cumsum | diagonal | trace | clamp`.
- `crates/chelis-ir/src/grad.rs` — **no `Gather` arm** in
  `compute_adjoints`. There is no `RiscOp::Gather` variant either,
  so the §3.5 RISC lowering path (`reshape + expand + mul + sum`) is
  not wired.

Net: `chelis check` and `chelis eval` accept programs that use
`gather` for forward computation, but `grad(f)` over a function
containing `gather` returns the diagnostic
`grad: failed to construct backward DAG (unsupported op or
verification failure)`. The cross-library AD claim — "AD flows
through ... because everything compiles to the same RISC primitive
set" — does not hold for `gather` today.

## Why this matters

Two failure modes downstream:

1. **MoE / embedding layers cannot be differentiated.** The natural
   way to express embedding lookup (`gather(table, token_indices,
   axis=0)`) and Mixture-of-Experts routing (`gather(expert_weights,
   expert_indices, axis=0)`) compiles forward but fails AD.
   `Std.Nn.Embedding` is documented as the user-facing surface over
   `gather` (`spec/design/chelis_phase3_plan.md:326,1129`), so the
   Embedding layer is similarly affected.

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

The fix is two-step and must land together to avoid arming the OOM
trap:

1. **Wire the §3.5 RISC lowering.** Either a new `RiscOp::Gather`
   variant with a hand-written adjoint (matched by the contract
   test below), or a Tier 2 desugarer that emits the
   `reshape + expand + mul + sum` chain so AD flows through existing
   adjoints.
2. **Add a scatter recognizer.** A new `detect_gather_pattern` in
   `crates/chelis-backend-c/src/blas.rs` (and the HIP analogue) that
   matches the `reshape + expand + mul + sum` shape and re-emits a
   sparse scatter-add kernel.

The adversarial duplicate-index test
(`gather_via_section_3_5_lowering_accumulates_duplicate_indices` in
`crates/chelis-ir/tests/grad_gather_contract.rs`) already builds the
post-§3.5 RISC DAG by hand and asserts the duplicate-index gradient
is correctly accumulated — the AD side will pass by construction.
The new requirement is that the recognizer pass also fire so the
forward path is bounded in memory.

## Why this isn't already filed

The Phase 3h plan
(`spec/design/chelis_phase3_plan.md` "Phase 3h: Core Tensor
Primitives") lists the primitives but does not enumerate
"AD adjoint shipped" or "scatter recognizer shipped" as gating
criteria. The acceptance oracle is "a pure Chelis model program
can [express the new primitives in forward computation]" — which
the host-only `gather` satisfies. The AD gate was implicit and
slipped.

## Regression test

`crates/chelis-ir/tests/grad_gather_contract.rs` —
`gather_via_section_3_5_lowering_accumulates_duplicate_indices`.
Builds the post-§3.5 RISC DAG and asserts the gradient at the
embedding table is `[3, 3, 0, 0]` for an all-zero indices stress
case (3 tokens routed to vocab 0, 0 to vocab 1). The test is
durable: it locks the §3.5 contract independently of whether
`gather` ever becomes a `RiscOp` variant. When closure lands, the
same test must continue to pass.
