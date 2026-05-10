# In-Place Fusion

**Status:** Active implementation contract. The current branch ships fusion
metadata preservation plus C backend v1 in-place codegen for fused elementwise
nodes with a proven `reusable_input`. HIP in-place codegen and broader op-level
rewrites remain open.

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
- Target acceptance: HIP emitted-code and numerical tests compare backend output
  with evaluator output on eligible and ineligible examples.

Current C acceptance commands:

```sh
cargo test -p chelis-backend-c fused_in_place -- --nocapture
```

Known target-behavior check while HIP in-place codegen is still open:

```sh
cargo test -p chelis-backend-hip target_fused_in_place_hip_restrict_shape_preserves_non_aliased_inputs -- --ignored
```

Expected target condition: the reusable input/output alias drops `restrict` only
from the aliased pointer pair, and unrelated inputs keep their restrict-qualified
fast path.
