# In-Place Fusion

**Status:** Active implementation contract. The current branch ships fusion
metadata preservation plus C and HIP backend v1 in-place codegen for fused
elementwise nodes with a proven `reusable_input`. Broader op-level rewrites
beyond `FusedElem` remain open.

## Pass Order

In-place fusion runs after backend specialization and DCE:

```text
AD -> closed-list no-op cleanup -> BLAS/gather/scatter recognizers
   -> cross-function callsite specialization -> DCE -> in-place fusion
   -> backend codegen
```

It must not run before AD, and it must not rewrite a dense pattern before the
specialization recognizers have had a chance to replace it.

## Eligibility

An operation may write into one input buffer only when all of the following hold:

- lowering attached a `reusable_input` hint from linearity analysis
- the chosen input has no other live reader at the in-place point
- output shape, dtype, and layout match the chosen input
- the operation is in the closed v1 list

The v1 list is:

- residual add where the residual operand is linear
- softmax normalize division `x / sum(x)` where `x` is linear
- layer-norm scale chain where the normalized activation input is linear
- elementwise mask multiply where `x` is linear

Any operation not listed here opts out, even if aliasing would be legal.

## Codegen Contract

The in-place backend path aliases the output pointer to the reusable input
pointer. C/HIP codegen must keep `restrict` on non-aliased pointers and remove
it only from the aliased input/output pointer. A future broadening of the
closed list must add an emitted-code assertion for the expected alias/restrict
shape.

The shipped C v1 is intentionally narrower than the full closed-list design:
it only fires for `RiscOp::FusedElem` nodes that already carry a
`reusable_input` hint, where the hinted tensor has the same shape/dtype as the
output and no other live reader. If the reusable input is contiguous, the output
aliases that input. If the reusable input is not contiguous, C codegen falls
back to slot-backed materialization so the view cannot write through a strided
borrow. The backend keeps the ordinary out-of-place path when the hint is
absent or unsafe.

The HIP v1 mirrors the C v1 with the same alias-proof predicate
(`binder_equivalent_tensor_type`) and the same runtime contiguity guard.
HIP-specific: the kernel parameter list adopts `__restrict__` qualifiers only
on the non-aliased external inputs; the aliased external + output stay as
plain pointers because they may reference the same device memory when the
contiguity branch fires. The fall-back slot allocation path is suppressed in
device-entrypoint mode because the slot is pre-allocated by
`emit_device_slot_allocations`.

Elementwise fusion may preserve a `reusable_input` hint when every surviving
hint in the fused chain names the same external input to the fused node.
Fusion must drop the hint rather than choose between conflicting reusable
external inputs or an absorbed internal value.

## Acceptance

- A fusion test proves `reusable_input` survives fusion.
- Current branch acceptance: fusion preserves an unambiguous external
  `reusable_input` hint and drops ambiguous hints.
- Current branch acceptance: C emitted-code tests prove in-place kernels do not
  drop `restrict` from every pointer.
- Current branch acceptance: a C compile/run test verifies contiguous inputs are
  mutated in place and strided inputs fall back to a materialized output.
- Current branch acceptance: HIP emitted-code structural tests assert the
  contiguity guard, the aliased view, and the `__restrict__` discipline on the
  kernel parameter list; the GPU manual gate `gf3_fused_in_place_fan_in_gpu_matches_cpu`
  proves numerical agreement end-to-end against the unfused evaluator.

Current C acceptance commands:

```sh
cargo test -p chelis-backend-c fused_in_place -- --nocapture
```

Current HIP acceptance commands:

```sh
cargo test -p chelis-backend-hip --test fused_in_place_forall_alias
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
  cargo test -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1 gf3_fused_in_place_fan_in_gpu_matches_cpu
```

(The HIP manual gate requires the local ROCm/HIP environment per
`docs/local_hip_environment.md`.)
