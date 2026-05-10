# In-Place Fusion

**Status:** Active implementation contract. The current branch ships fusion
metadata preservation and target-behavior tests; broad C/HIP in-place codegen
remains open.

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

Elementwise fusion may preserve a `reusable_input` hint when every surviving
hint in the fused chain names the same external input to the fused node.
Fusion must drop the hint rather than choose between conflicting reusable
external inputs or an absorbed internal value.

## Acceptance

- A fusion test proves `reusable_input` survives fusion.
- Current branch acceptance: fusion preserves an unambiguous external
  `reusable_input` hint and drops ambiguous hints.
- Target acceptance: emitted-code tests prove in-place kernels do not drop
  `restrict` from every pointer.
- Target acceptance: C/HIP numerical tests compare backend output with evaluator
  output on eligible and ineligible examples.

Known target-behavior checks while broad C/HIP in-place codegen is still open:

```sh
cargo test -p chelis-backend-c target_fused_in_place_restrict_shape_aliases_only_reusable_input -- --ignored
cargo test -p chelis-backend-hip target_fused_in_place_hip_restrict_shape_preserves_non_aliased_inputs -- --ignored
```

Expected target condition: the reusable input/output alias drops `restrict` only
from the aliased pointer pair, and unrelated inputs keep their restrict-qualified
fast path.
