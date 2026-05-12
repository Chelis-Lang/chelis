# `eval_cast` Tensor-arm gap diagnosis

Diagnosis pass for the bundled fix on branch `fix/eval-cast-tensor-arm`.
V2-F2 of the 0.7.6 red-team v2 (PR #58).

Cross-references:

- Parallel runtime/host-lane gap closed by PR #56 (`jit`, `par`):
  `docs/investigations/jit_par_runtime_gaps_diagnosis.md`.
- Failing-test pins for this branch:
  `crates/chelis-cli/tests/eval_cast_tensor_arm_gap.rs` (commit 1 of this
  branch).

No code changes in this commit.

## Spec section (quoted, do not edit)

`spec/03-deep-syntax.md` §2.7 (Transforms table, line 319):

```
| `cast` | `(cast {} expr target-type)` | Precision cast |
```

`cast` is a first-class precision-conversion node. Spec §2.10 (Tag Count
Summary, line 353) lists `cast` in the "Transforms" category alongside
`grad`, `vmap`, `jit`, `realize`, and `copy` — all of which are
tensor-valued in the general case.

## What works today

- `chelis check`: types resolve. `cast(tensor[N, p], q)` returns
  `tensor[N, q]`. The checker accepts the program.
- `chelis build --target c`: the IR lowering and C emission paths handle
  the tensor `cast` form. The lowered DAG carries a `RiscOp::Cast {
  new_precision }` node (see
  `crates/chelis-ir/src/lower.rs::lower_cast`, line 4882) that the C
  backend emits via `emit_cast` (see
  `crates/chelis-backend-c/src/emit.rs`, line 2858).

Empirical reproduction (commit `82f287a` HEAD of branch, against
`origin/main` at `21c6386`):

```
$ cat /tmp/cast_repro.ch
result = cast(to_tensor([1.0, 2.0, 3.0]), f64)
$ chelis check /tmp/cast_repro.ch
{ "score": 1, ..., "errors": [] }
$ chelis build --target c /tmp/cast_repro.ch
Wrote ./cast_repro.c and ./cast_repro.h
...
$ chelis eval --file /tmp/cast_repro.ch
error: unsupported cast from Tensor(RuntimeTensorValue {
    value: TensorValue { data: [1.0, 2.0, 3.0], shape: [3] },
    precision: F32
})
```

## What does not work — `eval_cast` Tensor arm

File: `crates/chelis-compiler-api/src/runtime.rs`.

The host evaluator's `EvalContext::eval_cast` (line 1107) is the
dispatch point that turns a Deep `(cast {} value target-type)` form into
a `RuntimeValue`. Its match has arms for `Int`, `Float`, `Bool`, and
`String` scalars, plus a catch-all that errors with
`unsupported cast from {other:?}`:

```rust
// crates/chelis-compiler-api/src/runtime.rs lines 1119-1127
match (value, target) {
    (RuntimeValue::Int(value), "int32" | "int64") => Ok(RuntimeValue::Int(value)),
    (RuntimeValue::Int(value), "f32" | "f64") => Ok(RuntimeValue::Float(value as f64)),
    (RuntimeValue::Float(value), "f32" | "f64") => Ok(RuntimeValue::Float(value)),
    (RuntimeValue::Float(value), "int32" | "int64") => Ok(RuntimeValue::Int(value as i64)),
    (RuntimeValue::Bool(value), "bool") => Ok(RuntimeValue::Bool(value)),
    (RuntimeValue::String(value), "string") => Ok(RuntimeValue::String(value)),
    (other, _) => Err(format!("unsupported cast from {other:?}")),
}
```

No `RuntimeValue::Tensor` arm exists, so the catch-all fires for every
tensor input. The error message format (`unsupported cast from
Tensor(..)`) matches the failing-test pins in commit 1.

## Why the catch-all is reachable for a tensor

A top-level `result = cast(to_tensor([..]), <prim>)` does not get
classified as DAG-lowerable by `top_level_lowering_map` because
`to_tensor` is a host-runtime builtin (see
`expr_requires_host_runtime` and surrounding helpers in
`crates/chelis-ir/src/lower.rs`). The host runtime therefore dispatches
the `(cast ..)` form directly via `eval_list -> eval_cast`. The IR DAG
evaluator (`crates/chelis-ir/src/eval.rs`, line 877:
`RiscOp::Cast { .. } => values[&node.inputs[0]].clone()`) is never
reached on this code path because the host runtime owns the value.

This is the exact same shape as PR #56's jit/par gap: the IR layer is
correct, but the host-runtime dispatch never grew the matching arm.

## How the C backend handles tensor cast

C emission at `crates/chelis-backend-c/src/emit.rs::emit_cast` (line
2858) currently allocates a fresh output tensor and `memcpy`s the input
data. The runtime tensor stores data as `Vec<f64>` regardless of
precision (`crates/chelis-ir/src/eval.rs::TensorValue`, line 11), so the
C-backend's bit-preserving copy is consistent with that internal
representation. Strict per-precision arithmetic conversion is a
pre-existing C-backend behavior question and is not in scope for V2-F2;
V2-F2 closes the host-runtime gap so `chelis eval --file` stops
rejecting programs that `chelis check` and `chelis build` accept.

The fix commit on this branch mirrors the host-runtime conversion
semantics that the test fixtures pin: identity-on-data for casts where
both ends are floats or both ends are ints, and explicit truncation /
widening at the value level for float<->int crossings. This is what
makes the four pinned fixtures pass without claiming any new C-backend
guarantee.

## Planned fix shape

Add a `RuntimeValue::Tensor` arm to `eval_cast` in
`crates/chelis-compiler-api/src/runtime.rs` that:

1. Parses the target `Prim` from the cast metadata (same `target` string
   the scalar arms read).
2. Element-wise converts `tensor.value.data` per source / target Prim
   pair. The stored representation is `Vec<f64>`; integer targets
   truncate-toward-zero via `(x as i64) as f64`, float<->float is
   identity when storage is f64, and `f64 -> f32` round-trips through
   `(x as f32) as f64` to drop precision honestly.
3. Returns a `RuntimeTensorValue` with the updated `precision` tag and
   the converted data, preserving the input shape (Spec C8: "dims must
   not change", `crates/chelis-ir/src/verify.rs` line 422).

The four fixtures in `crates/chelis-cli/tests/eval_cast_tensor_arm_gap.rs`
are flipped from `#[ignore]` to running in the same commit.
