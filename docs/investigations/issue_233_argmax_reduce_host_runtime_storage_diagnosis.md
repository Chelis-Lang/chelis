# Issue #233: argmax_reduce / argmin_reduce host-runtime storage gap

## Reproducer

```chelis
make = pad_sequences([[1.0, 4.0, 2.0], [3.0, 0.5, 5.0]], 0.0)
preds = argmax_reduce(&make, 1)
refs = to_tensor([cast(1, int64), cast(2, int64)])
out = eq(preds, refs)
```

Today this fails inside `chelis eval` / the host-runtime evaluator
with:

```text
tensor comparison expects matching tensor precision
```

even though every static type label on the program is internally
consistent: `preds` is `tensor[2, int64]` (correct post-#230) and
`refs` is `tensor[2, int64]` (the int64 to_tensor literal). The
program also passes `chelis check` cleanly. The failure surfaces in
school PR #52, where `Std.Loss.Metrics.accuracy` builds an int64
prediction tensor via `argmax_reduce` and tries to match it against
an int64 labels tensor.

## Root cause

The fix for #230 (`crates/chelis-types/src/infer.rs::check_reduction_signature`,
shipped in 0.7.15) widened the **type-system** result of
`argmax_reduce` / `argmin_reduce` to `tensor[..., int64]` regardless of
input precision. That fix correctly resolved the type label on every
declared signature, downstream type-binding, and constraint surface
hydronnx and school depend on. It also explicitly **did not** touch
the host-runtime storage; the 0.7.15 CHANGELOG entry says so:

> The host-runtime / IR evaluator continues to store integer-valued
> floats internally per the Phase 3j-pre Batch 1 caveat documented on
> `RiscOp::Argmax`; the int64 type-system label is independent of the
> storage layout. The `TODO(phase3j): widen backend runtime to carry
> Int64 tensors natively` in `crates/chelis-ir/src/dag.rs` tracks the
> eventual storage widening.

That divergence is the bug. In the host-runtime, `RuntimeTensorValue`
carries a `precision: Prim` field that the comparison and conversion
primitives branch on. `crates/chelis-compiler-api/src/runtime.rs::tensor_reduce_host`
builds the result with:

```rust
Ok(RuntimeTensorValue {
    value: IrTensorValue::from_vec(out_shape, out),
    precision: tensor.precision, // <- propagates the input precision unchanged
})
```

For `Argmax` / `Argmin` the input is typically `f32`, so the produced
storage carries `precision = Prim::F32` even though every f64 cell in
`out` is an integer-valued index. The matching int64 literal built by
`to_tensor([cast(_, int64), ...])` carries `precision = Prim::Int64`
(per `list_to_tensor_data`), so `tensor_compare_value`'s precision
guard fires:

```rust
if lhs.precision != rhs.precision {
    return Err("tensor comparison expects matching tensor precision".to_string());
}
```

Same root cause for the silent-corruption paths:

- `tensor_to_list_values` (`to_list`) branches on
  `tensor.precision.is_float()` and emits `RuntimeValue::scalar_like_float`
  scalars, surfacing as `ExecutionValue::Float64` instead of
  `ExecutionValue::Int64`.
- `tensor_to_scalar` does the same on a rank-0 tensor.
- `cast(_, "int64")` on the argmax result silently succeeds (it converts
  f32 → int64) but obscures whether the original tensor was authored
  with the correct storage.

The IR-level `chelis_ir::eval::reduce_argcmp` is also still precision-
erased (it returns a `TensorValue` with no precision field), but that
is fine: the IR evaluator is the single-tensor f64 model that the
backend lane uses, and the host-runtime layer is where the `precision`
label gets attached. The fix is in the host-runtime adapter, not the
IR evaluator.

## Sibling sweep

The other reduction ops (`sum`, `max_reduce`, `min_reduce`,
`prod_reduce`) correctly propagate `tensor.precision` because they
return reduced operand values, not indices. Their output dtype is the
input dtype (modulo §5.7.1 sum widening), which is exactly what
`tensor.precision` is. No change needed for those ops, and the
existing `issue_230_argmax_argmin_runtime.rs` parity tests already
guard them — they only check the numeric output, not the precision
tag, but the type-system tests in
`issue_230_argmax_reduce_output_dtype.rs` already pin the dtype
contract.

The fix is narrow to the `Argmax` / `Argmin` arms of the result-
construction `match` in `tensor_reduce_host`.

## Fix

In `tensor_reduce_host`, branch the result `precision` on the op:
return `Prim::Int64` for `Argmax`/`Argmin`, and `tensor.precision`
(unchanged) for `Sum`/`Min`/`Max`/`Prod`. The fix is ~5 lines and
surgical to one helper.

The IR-level `reduce_argcmp` continues to operate on the precision-
erased f64 representation; the storage widening is exclusively a
host-runtime concern. `RiscOp::Argmax`'s doc comment in
`crates/chelis-ir/src/dag.rs` is updated to remove the
"whatever precision the caller assigns" claim and the TODO that #233
closes.

## Acceptance oracle

`crates/chelis-compiler-api/tests/issue_233_argmax_storage_int64.rs`
(nine tests):

- Positive: argmax / argmin axis-0 and axis-1 eq against an int64
  literal tensor produces an all-true mask without precision error.
- Positive: argmax over an f64 input still produces int64 storage.
- Positive: `to_list(argmax_reduce(...))` returns
  `ExecutionValue::Int64` per element (not `Float64`).
- Positive: the school PR #52 `Std.Loss.Metrics.accuracy` pattern
  (argmax preds vs int64 labels) runs end-to-end.
- Negative: comparing the result against a non-int64 tensor rejects
  with a precision-mismatch diagnostic that names int64 (so the test
  fails closed if the fix accidentally weakens the precision guard).

The existing `issue_230_argmax_argmin_runtime.rs` parity tests
continue to pass — they only check numeric output, which is unchanged.
