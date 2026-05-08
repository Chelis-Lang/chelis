# matmul-rank2-rule-vs-einsum-shipped: `matmul` type rule rejects rank ≥ 2 even though `einsum` is the documented batched answer

**Status:** **OPEN**
**Filed:** 2026-05-08
**Owning phase:** Phase 3h / language ergonomics
**Discovered by:** Canonical heads-as-dimension MHA expressibility
test (`docs/identified_gaps.md` Gap 4)

## Summary

`spec/design/chelis_project_plan.md:457` and the Phase 3h plan name
`einsum` as the documented answer for batched matmul / contraction
patterns: *"einsum is the single highest-impact addition because it
covers matmul, batched matmul, transpose, trace, outer products,
and common contraction patterns in one primitive."* `einsum` is
registered in the type environment as a `generic_triop` builtin
(`crates/chelis-types/src/builtins.rs:135`), Phase 3h is marked
shipped in `spec/12-roadmap.md`, and the project plan's perf
boundary explicitly notes the rank-2 BLAS specialization is
intentional (`spec/design/phase1d_flattening.md:39`).

Empirically, however, the **`matmul` type rule still hard-rejects
rank ≥ 2**:

```rust
// crates/chelis-types/src/infer.rs:7052
if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
    errors.push(CheckError::new(
        CheckErrorKind::DimensionMismatch,
        format!("matmul expects rank-2 tensors, got rank {} and {}",
                lhs_dims.len(), rhs_dims.len()),
        vec![],
    ));
    return Type::Error;
}
```

A user attempting to write the canonical PyTorch heads-as-dimension
MHA form

```chelis-surf-fragment
qkv = qkv_proj(x)
qkv = reshape(qkv, [batch, seq, num_heads, 3 * head_dim])
qkv = permute(qkv, [0, 2, 1, 3])  // [batch, head, seq, dims]
q, k, v = chunk(qkv, 3, dim=-1)
scores = matmul(q, permute(k, [0, 1, 3, 2]))  // batched [batch, head, seq, seq]
```

gets blocked at the `matmul(q, permute(k, ...))` line with
`matmul expects rank-2 tensors, got rank 4 and 4`. The error
message does not mention `einsum`, so users are left without a
discoverable path forward. The corpus's published 4-head MHA
(`examples/transformer_block.ch`) works around this by manually
unrolling per head — i.e., 4 separate sets of `wq_i / wk_i / wv_i /
wo_i` and 4 explicit rank-2 matmul lines per head.

## Why this matters

1. **PyTorch / JAX users importing mental models cannot translate
   them.** The most common transformer block layout in the literature
   is `[batch, head, seq, dim]` with batched matmul broadcasting
   over leading axes. Today's Chelis requires either reshaping
   everything down to rank-2 manually or rewriting via `einsum` —
   neither is what the user expected.

2. **The compiler doesn't direct the user to the documented
   answer.** `infer.rs:7052` produces a `DimensionMismatch` error
   without a hint string. The right ergonomic answer is "use
   `einsum("...,...->...", q, k)` for batched contractions" but
   that's nowhere in the diagnostic.

3. **It blocks honest expression of `Std.Nn.Attention`.** The
   `scaled_dot_product_attention` reference shipped in
   `chelis_phase3_plan.md:871` notes it ships as concrete rank-2.
   This is a legitimate scope choice but means the `Std.Nn` surface
   doesn't currently expose a heads-as-dim MHA primitive.

## Two scope-level questions

The closure path depends on which intent the spec is committing to:

- **Option A: einsum is the canonical batched answer; matmul stays
  rank-2.** Then the gap is purely an ergonomics / diagnostic gap.
  Fix: extend the `matmul`-rank error message to point at `einsum`,
  and add an example to `packages/chelis-std/SKILL.md` showing
  batched contraction.

- **Option B: `matmul` should also accept rank ≥ 2 with leading-axis
  broadcasting.** Then this is a language-feature gap: lift the type
  rule, generalize `lower_matmul` to emit batched expand+mul+sum,
  and add a batched-GEMM specializer to the BLAS recognizer in both
  backends. Larger surface but matches PyTorch ergonomics.

Per `spec/design/chelis_canonical_reference.md:438-443` ("the
existing HIP rank-2 BLAS fast path does not yet upgrade vmapped
rank-3 matmul into a batched BLAS call"), the implicit current
intent is Option A *for the BLAS specialization* but Option B
*for type-checker acceptance*. That's not yet decided in writing.

## Closure plan

Decide between A and B, then:

- **If A:** add a hint string to the `matmul` rank-mismatch
  diagnostic pointing at `einsum`. Update
  `packages/chelis-std/SKILL.md` to include a worked
  batched-contraction example. Optionally add a rejection test that
  asserts the new hint is present.
- **If B:** lift `infer.rs:7052` to accept rank ≥ 2 with
  leading-axis broadcasting unification, extend `lower_matmul` to
  emit batched expand+mul+sum, and add batched-GEMM detection in
  `crates/chelis-backend-c/src/blas.rs` and the HIP analogue.

Either closure must update `examples/illustrative/` with a working
canonical heads-as-dimension MHA so the corpus reflects the
ergonomic answer.

## Probe corpus

`examples/illustrative/mha_two_heads_unrolled.ch` and
`examples/transformer_block.ch` — current per-head-unrolled
workaround.

`examples/illustrative/mha_single_head.ch` — single-head reference
for the rank-2 matmul shape.

The `mha_canonical_heads.ch` attempt that motivated this filing is
not in the corpus today because it does not type-check; it lives in
the planning thread for this bug filing.
