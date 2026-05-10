# Sparse Gather / Scatter Specialization

**Status:** Active implementation contract, with only the first IR/AD slice
shipped in the current branch.

## Current Branch Status

The current branch adds first-class `RiscOp::Gather { axis }` and
`RiscOp::ScatterAdd { axis }`, evaluator support, verifier coverage, C sparse
codegen, compiler-API wire variants, and a gather AD test proving duplicate
index accumulation through `ScatterAdd`.

The following items remain open and must not be implied as complete:

- Surf / host `gather` is not yet lowered through the Section 3.5 RISC
  decomposition.
- The dense Section 3.5 tag-tree recognizer is not yet implemented in the
  shared specialization pass.
- HIP sparse gather/scatter codegen is not yet implemented; HIP compilation
  currently rejects these first-class IR nodes instead of silently emitting an
  unsupported path.
- The large embedding emitted-code oracle below is still target behavior, not a
  shipped acceptance test.

Until the recognizer ships, do not wire Surf `gather` to the dense lowering for
large embedding/MoE shapes; doing so would reintroduce the `[N,V,D]`
materialization trap this design is meant to avoid.

## Pass Order

Sparse recognition runs in the shared IR specialization substrate:

```text
AD -> closed-list no-op cleanup -> BLAS/gather/scatter recognizers
   -> cross-function callsite specialization -> DCE -> in-place fusion
   -> backend codegen
```

AD must see the ordinary RISC decomposition. Codegen must see the specialized
node after DCE has removed dense intermediates.

## Gather Contract

The Section 3.5 lowering for embedding-style gather is fixed for this workstream:

```text
Sum(axis = vocab,
  Mul(
    Expand(one_hot(indices, V), [N, V, D]),
    Expand(values,             [N, V, D])))
```

The recognizer matches exactly this tree shape. `V` may be symbolic. `N` and `D`
must be concrete in v1. The replacement is:

```text
RiscOp::Gather { axis }
inputs: values, indices
output: [N, D]
```

The backend contract is sparse indexed loading. C/HIP output must not allocate
or compute a dense `[N, V, D]` one-hot/product tensor.

## ScatterAdd Contract

The gather adjoint is represented as scatter-add, not replace scatter:

```text
RiscOp::ScatterAdd { axis }
inputs: target, indices, updates
output: same shape as target
```

Duplicate indices accumulate. This is the required behavior for embedding and
MoE gradients. Replace-mode duplicate rejection remains a host/runtime scatter
rule and is not the semantics of `ScatterAdd`.

## Acceptance

- `crates/chelis-ir/tests/grad_gather_contract.rs` continues to prove duplicate
  index accumulation.
- Current branch acceptance: the first-class `Gather` AD path lowers its
  adjoint through `ScatterAdd`, and `ScatterAdd` accumulates duplicate indices
  in the evaluator.
- Current branch acceptance: verifier and C emitted-code tests reject or avoid
  dtype-unsafe sparse paths by requiring integer indices and typed payload
  access.
- Target acceptance: a generated-code fixture for an embedding lookup with
  `V=50000`, `D=1024`, `N=128` asserts no dense `[N,V,D]` allocation appears.
- Target acceptance: C and HIP sparse paths agree with the evaluator on small
  concrete examples.
