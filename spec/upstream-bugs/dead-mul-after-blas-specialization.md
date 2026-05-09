# dead-mul-after-blas-specialization: BLAS-hit matmul still allocates and computes the Tier-2 `Mul` intermediate

**Status:** **CLOSED by M1**
**Filed:** 2026-05-08
**Owning phase:** Phase 1d (BLAS specialization) / DCE pipeline
**Discovered by:** Cost-profile inspection of
`crates/chelis-cli/tests/cross_library_semantic_gap.rs`
(Test 5 — Cross-library AD semantic gap)

## Summary

M1 moved BLAS specialization into `chelis_ir::specialize` as an IR
replacement pass. The pass runs after AD and before DCE/fusion/codegen,
replacing recognized rank-2 matmul subgraphs with `RiscOp::BlasMatmul`.
DCE then removes the orphan `Mul` and `Expand` nodes. The C and HIP
emitters emit BLAS from the specialized node, so BLAS-hit direct and
inline matmul allocate only the result buffer.

Original finding: when the C backend specialized a matmul subgraph to
`cblas_sgemm` at codegen time, the Tier-2 `Mul` intermediate was still
allocated and computed at runtime.

Reproducer: build the smallest BLAS-eligible matmul and inspect
the generated C.

```chelis-surf
def f(a: tensor[8, 16, f32], b: tensor[16, 4, f32]) -> tensor[8, 4, f32] = matmul(a, b)
```

```sh
cargo run -p chelis-cli -- build f.ch --target c --output out
cat out/f.c
```

In `out/f.c`:

```c
chelis_tensor *t4 = chelis_alloc(3, (int[]){ 8, 16, 4 }, CHELIS_F32);
if (chelis_is_contiguous(t2) && chelis_is_contiguous(t3) && t2->size == t4->size && t3->size == t4->size) {
    float* restrict __out_4 = (float*)t4->data;
    const float* restrict __in_a_4 = (const float*)t2->data;
    const float* restrict __in_b_4 = (const float*)t3->data;
    #pragma omp parallel for simd
    for (int i = 0; i < t4->size; i++) {
        __out_4[i] = __in_a_4[i] * __in_b_4[i];
    }
}
// ... t5_a, t5_b setup ...
chelis_tensor *t5 = chelis_alloc(2, (int[]){ 8, 4 }, CHELIS_F32);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 8, 4, 16, 1.0f,
            t5_a->data, 16, t5_b->data, 4, 0.0f, t5->data, 4);
// t4 is never read after this point
chelis_free(t4);
```

`t4` is allocated, filled with `t2 * t3` in a fused parallel loop,
then freed without contributing to any output.

## Why this matters

The cost is cubic in matmul size. For a `[m, k] @ [k, n]` matmul,
the dead `Mul` intermediate is `m * k * n * sizeof(precision)`
bytes. Concrete examples:

| Matmul shape | Dead Mul size | Useful result size |
|---|---|---|
| 8 × 16 × 4 (f32) | 2 KiB | 128 B |
| 256 × 256 × 64 (f32) | 16 MiB | 64 KiB |
| 2048 × 2048 × 2048 (f32) | 32 GiB | 16 MiB |
| 1024 × 4096 × 1024 (f32, FFN) | 16 GiB | 4 MiB |

For the 4-head MHA + FFN block in
`examples/transformer_block.ch`, this is the dominant working-set
cost: roughly 3 MiB per `seq` token (per
`crates/chelis-cli/tests/traceability_paradox.rs` cost-profile
analysis). At `seq = 2048` the working set is ~14.6 GiB, the
majority of which is dead `Mul` intermediates that BLAS hits would
have eliminated in any normal compiler pipeline.

The compute waste is also non-trivial: the `parallel for simd`
loop runs `m * k * n` multiplies for every BLAS-hit matmul, in
parallel with the actual sgemm call.

## Why this happens

The matmul desugarer (`crates/chelis-ir/src/tier2.rs::lower_matmul`,
lines 533-606) emits the canonical Einstein form:

```
A_expanded = expand(A, [..., i, j, 1])
B_expanded = expand(B, [..., 1, j, k])
product    = mul(A_expanded, B_expanded)   // <- this node lives in the IR DAG
result     = sum(product, axis=-2)
```

BLAS detection runs at **codegen time**, inside
`crates/chelis-backend-c/src/emit.rs:1522`:

```rust
&& let Some(matmul) = crate::blas::detect_matmul_pattern(dag, NodeId(id))
```

The detector recognises the `Sum -> Mul -> (Expand, Expand)`
shape on the `Sum` node and emits `cblas_sgemm` for the result
buffer. But the `Mul` node is still in the DAG, so the codegen
emitter visits it independently and emits its own allocation +
fused parallel-for-simd kernel. There is no DCE pass between BLAS
detection and emission that could remove the now-dead `Mul`.

The optimize pass (`crates/chelis-ir/src/optimize.rs`) does have
DCE (`dead_code_eliminate`, line 118), but it runs *before*
codegen-time BLAS detection — at that point the `Mul` still has
a downstream consumer (`Sum`), so it isn't dead. Once the
emitter has decided to specialize the `Sum` to sgemm and skip the
DAG-prescribed code path for it, the `Mul` becomes dead — but
DCE has already run.

## Closure plan

Two viable approaches; either closes the gap.

### Approach A — Move BLAS detection into the optimize pass

Replace the codegen-time `detect_matmul_pattern` call with a
DAG-rewrite pass in `crates/chelis-ir/src/optimize.rs` that:

1. Walks the DAG looking for the `Sum -> Mul -> (Expand, Expand)`
   pattern.
2. Replaces the `Sum` (and its `Mul` predecessor and the `Expand`
   nodes) with a synthetic `RiscOp::BlasMatmul { m, n, k }` node,
   or marks the `Sum` with metadata that the emitter reads to skip
   the standard `Sum`/`Mul`/`Expand` codegen.
3. Runs DCE afterwards to remove the now-orphan `Mul` and `Expand`
   nodes.

Pros: surfaces BLAS specialization as a first-class IR
transformation (visible to grad / fuse / vmap passes downstream).
Cons: changes the IR vocabulary (or relies on metadata
side-channels).

### Approach B — Skip orphan-`Mul` emission at codegen time

Keep BLAS detection at codegen but extend the emitter:

1. After detecting that a `Sum` node specializes to sgemm, mark
   the `Mul` predecessor as "skip emission."
2. Mark the `Expand` predecessors as "skip emission" if no other
   live consumer reads them.
3. The `chelis_free(t4)` cleanup line should also be skipped to
   match the missing allocation.

Pros: localised change, no IR-vocabulary churn.
Cons: codegen-time skipping is fragile — easy to miss a
"sometimes the Mul is shared with another consumer" edge case.
A test would need to lock the contract.

M1 chose Approach A. Regression coverage now asserts:

- The Test 5 cost-profile assertion in
  `crates/chelis-cli/tests/cross_library_semantic_gap.rs` requires direct
  and inline_manual to produce identical 128-byte working sets (= result
  only).
- `crates/chelis-backend-c/tests/pattern_matcher_brittleness.rs` requires
  identity-cast-perturbed matmul to specialize after no-op cleanup and
  post-specialization DCE.

## Probe corpus

`crates/chelis-cli/tests/cross_library_semantic_gap.rs` already
documents this finding empirically. The current cost-profile output
shows `direct = 128 bytes` for an 8×16 @ 16×4 matmul: the dead
2048-byte `Mul` allocation is gone on BLAS-hit paths.

## Related gaps

This is the sixth gap in `docs/identified_gaps.md`. It interacts
with:

- **Gap 1 — C-backend memory planning.** Even with a memory
  planner, the dead `Mul` would still consume its buffer for the
  duration of the BLAS call (no later consumer to plan around).
  Closing this gap is more impactful than closing Gap 1 for the
  matmul-heavy workloads typical of transformers.
- **Gap 2 — pattern matcher brittleness.** Approach A above
  (moving BLAS detection into the optimize pass) is also a
  natural opportunity to add cast-elision and other algebraic
  pre-cleanup so the recogniser fires on more shapes.
- **Gap 4 — rank-2 matmul.** If the BLAS detector is generalised
  to higher rank in the same change, the dead `Mul` cost
  amplifies further (rank-3 batched matmul materialises
  `[batch, m, k, n]` instead of `[m, k, n]`) — making this gap
  even more consequential.
