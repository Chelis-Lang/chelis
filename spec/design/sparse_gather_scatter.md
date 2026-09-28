# Sparse Gather / Scatter Specialization

**Status:** Active implementation contract. The current branch ships the
first-class sparse IR/AD/C path, routes tensor-lane Surf `gather` directly to
that sparse node, recognizes the internal dense §3.5 `OneHot` tag tree, and
emits HIP sparse gather/scatter-add kernels for f32 payloads with i32/i64
indices.

## Current Branch Status

The current branch adds first-class `RiscOp::Gather { axis }` and
`RiscOp::ScatterAdd { axis }`, evaluator support, verifier coverage, C sparse
codegen, compiler-API wire variants, and a gather AD test proving duplicate
index accumulation through `ScatterAdd`. Tensor-lane Surf calls of the form
`gather(values, indices, axis)` now lower directly to the first-class sparse
node instead of the host runtime call.

This branch also adds internal-only `RiscOp::OneHot { vocab }`. It is a
recognition anchor for dense §3.5 gather trees, not a user-callable Surf op and
not a backend op. The specialization pass rewrites the fixed
`OneHot + Expand + Mul + Sum` tree to `RiscOp::Gather`; unmatched `OneHot`
falls back to primitive dense IR before backend codegen, and tests assert no
`OneHot` survives specialization.

The following items remain open and must not be implied as complete:

- The recognizer intentionally matches the internal `RiscOp::OneHot` tag tree,
  not arbitrary historical `const + eq + expand` encodings that no longer carry
  the original index operand.
- HIP sparse kernels are f32-payload only in v1 and require i32/i64 index
  tensors to be loaded inputs. Non-load integer index producers need broader
  integer HIP codegen before they can feed sparse kernels safely. HIP
  `ScatterAdd` uses `atomicAdd`; f64 scatter-add is rejected with an explicit
  atomic-support diagnostic.
- Replace-scatter is separate from `ScatterAdd`. Last-write-wins scatter does
  not get a sparse summary in this milestone and must not be conflated with the
  AD accumulation op.

The direct first-class lowering remains the bounded path for tensor-lane
`gather`. Any future dense lowering must use the internal `OneHot` anchor if it
wants the recognizer to preserve the index operand and collapse before codegen.

## Pass Order

Sparse recognition runs in the shared IR specialization substrate:

```text
AD -> closed-list no-op cleanup -> dense gather recognizer
   -> one_hot fallback lowering -> BLAS recognizer
   -> cross-function callsite specialization -> DCE -> in-place fusion
   -> backend codegen
```

AD must see the ordinary RISC decomposition. Codegen must see the specialized
node after DCE has removed dense intermediates. The source-level invariant is
locked by `chelis_ir::specialize::SPECIALIZATION_PIPELINE_ORDER`.

## Gather Contract

The Section 3.5 lowering for embedding-style gather is fixed for this workstream:

```text
Sum(axis = vocab,
  Mul(
    Expand(one_hot(indices, V), [N, V, D]),
    Expand(values,             [N, V, D])))
```

The recognizer matches exactly this tree shape when `one_hot` is the internal
`RiscOp::OneHot { vocab }` marker. `N` and `D` may be concrete or symbolic as
ordinary tensor dimensions; `V` is carried by the internal `vocab` field and
must agree with the values-table leading dimension. The replacement is:

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
- Current branch acceptance: generated C for first-class `Gather` and
  duplicate-index `ScatterAdd` compiles and agrees with the expected numeric
  output on a small concrete fixture.
- Current branch acceptance: a C generated-code fixture for an embedding lookup
  with `V=50000`, `D=1024`, `N=128` asserts no dense `[N,V,D]` allocation
  appears.
- Current branch acceptance: CLI dispatch coverage proves Surf tensor-lane
  `gather` no longer emits `chelis_tensor_gather()` and instead emits the
  bounded sparse C loop.
- Current branch acceptance: the shared specialization pass collapses a
  synthesized internal `OneHot + Expand + Mul + Sum` gather tree to
  `RiscOp::Gather`, and a negative unmatched `OneHot` fixture lowers to
  primitive dense IR so backend codegen never sees `OneHot`.
- Current branch acceptance: HIP sparse paths agree with the evaluator on
  small concrete examples through the ignored manual GPU gate:
  `cargo test -p chelis-backend-hip --test gpu_correctness g16_sparse -- --ignored --test-threads=1`.
