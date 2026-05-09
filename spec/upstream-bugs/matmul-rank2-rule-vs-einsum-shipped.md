# matmul-rank2-rule-vs-einsum-shipped: `matmul` type rule rejected rank ≥ 2 even though `einsum` is the documented batched answer

**Status:** **PARTIALLY CLOSED by M3** — rank ≥ 2 `matmul` now type-checks
and lowers generically; batched BLAS specialization remains open.
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

Before M3, however, the **`matmul` type rule hard-rejected rank ≥ 2**:

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

was blocked at the `matmul(q, permute(k, ...))` line with
`matmul expects rank-2 tensors, got rank 4 and 4`. M3 chose Option B:
`matmul` now accepts rank ≥ 2 with leading-axis broadcasting and lowers
through generic RISC `expand + mul + sum`. The corpus now includes
`examples/illustrative/mha_heads_as_dim.ch` as the accepted heads-as-dim
shape.

## Why this matters

1. **PyTorch / JAX users importing mental models could not translate
   them.** The most common transformer block layout in the literature
   is `[batch, head, seq, dim]` with batched matmul broadcasting
   over leading axes. M3 closes this expressibility gap.

2. **The performance fast path still has a narrower scope.** C and HIP
   specialization still target rank-2 BLAS matmul. Batched matmul stays
   correct through generic lowering until a batched BLAS recognizer lands.

3. **It used to block honest expression of `Std.Nn.Attention`.** The
   `scaled_dot_product_attention` reference shipped in
   `chelis_phase3_plan.md:871` notes it ships as concrete rank-2.
   M3 removes the type-system barrier for heads-as-dim formulations.

## Two scope-level questions

M3 resolved the language-level question in favor of Option B:

- **Option A: einsum is the canonical batched answer; matmul stays
  rank-2.** Then the gap is purely an ergonomics / diagnostic gap.
  Fix: extend the `matmul`-rank error message to point at `einsum`,
  and add an example to `packages/chelis-std/SKILL.md` showing
  batched contraction.

- **Option B: `matmul` should also accept rank ≥ 2 with leading-axis
  broadcasting.** Then this is a language-feature gap: lift the type
  rule, generalize `lower_matmul` to emit batched expand+mul+sum,
  and add a batched-GEMM specializer to the BLAS recognizer in both
  backends. M3 implemented the type rule and generic lowering pieces.
  The batched-GEMM specializer is still a performance follow-up.

Per `spec/design/chelis_canonical_reference.md:438-443` ("the
existing HIP rank-2 BLAS fast path does not yet upgrade vmapped
rank-3 matmul into a batched BLAS call"), the implicit current
intent is Option A *for the BLAS specialization* but Option B
*for type-checker acceptance*. That's not yet decided in writing.

## Remaining closure plan

Add batched-GEMM detection in `crates/chelis-backend-c/src/blas.rs` and
the HIP analogue. Static eligibility should match the existing rank-2
detector discipline: every operand/result dimension needed for loop
bounds and BLAS strides must be `Lit(n)` or `Named(_, Some(n))`; any
symbolic or polymorphic dimension falls through to the generic lowering.
The test suite should lock both the positive statically concrete case and
the negative symbolic-dimension case.

## Probe corpus

`examples/illustrative/mha_two_heads_unrolled.ch` and
`examples/transformer_block.ch` — current per-head-unrolled
workaround.

`examples/illustrative/mha_single_head.ch` — single-head reference
for the rank-2 matmul shape.

`examples/illustrative/mha_heads_as_dim.ch` — accepted M3
heads-as-dimension batched matmul corpus example.
