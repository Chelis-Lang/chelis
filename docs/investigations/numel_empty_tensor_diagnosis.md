# `numel(to_tensor([]))` returns 1 instead of 0

Diagnosis for `Runtime-EmptyTensorNumel-F1` (pre-0.7.8 release blocker).

## Symptom

```text
$ cat repro.ch
result = numel(to_tensor([]))

$ chelis eval --file repro.ch
1
```

Expected: `0`. Confirmed reproducer in
`crates/chelis-cli/tests/eval_empty_tensor_numel.rs::eval_numel_empty_tensor`,
gated `#[ignore]` with the failure pinned in commit 699148e.

The same wrong value flows through the C backend: `chelis build --target c
repro.ch && gcc ... && ./repro` prints `result = 1`. Both the host-side
evaluator and the C runtime path are wrong, by the same root cause shape.

## Where the shape is correct

`to_tensor([])` produces shape `[0]`, data `[]`. Verified by

```text
$ cat show.ch
result = to_tensor([])
$ chelis eval --file show.ch
tensor(shape=[0], data=[])
```

and by the companion fixture `eval_rank_empty_tensor` which pins
`rank(to_tensor([])) == 1`. The list-literal-of-empty path infers rank-1,
not rank-0; shape inference is not the bug.

## Failure point 1: host evaluator

`crates/chelis-compiler-api/src/runtime.rs`, around line 1805:

```rust
"numel" => {
    let tensor = expect_tensor_arg(args, 0)?;
    let numel = tensor.value.shape.iter().product::<usize>().max(1);
    Ok(RuntimeValue::Int(numel as i64))
}
```

For shape `[0]`, `[0].iter().product::<usize>() == 0`, then `.max(1) == 1`.
The clamp inflates the genuine zero back to one.

The clamp does not protect the scalar (rank-0) case either: for shape `[]`,
`[].iter().product::<usize>() == 1` already, via the empty-product identity.
The `.max(1)` is dead code for scalars and wrong for empty rank-1+ tensors.

`crates/chelis-ir/src/eval.rs::numel` (line 30) handles the scalar case
explicitly and does not clamp the non-scalar product, so the IR-level
evaluator is correct. The bug is only in the host-runtime dispatch.

## Failure point 2: C runtime

`crates/chelis-runtime/src/lib.rs`, lines 575-577 inside
`chelis_alloc_tensor`:

```rust
if tensor.size == 0 {
    tensor.size = 1;
}
```

and lines 624-626 inside `chelis_alloc_view`. Both clamp the computed
element count back to one when the shape has a zero dimension.

`chelis_tensor_numel` (line 721) then returns this clamped size to C-emitted
code. The clamp predates the present bug and was probably intended to keep
the subsequent `posix_memalign` call from receiving a zero-byte request. But
the allocator call already runs `bytes.max(1)` on line 581, so the size
clamp is not needed for allocation safety. It only corrupts the logical
element count reported by `chelis_tensor_numel`.

Other consumers of `(*tensor).size`:

* `for linear in 0..(*tensor).size { ... }` loops at lines 1997, 2044, 2102,
  2231, 2573, 3183 — a zero loop bound is the correct no-op for empty
  inputs.
* `bytes = (*tensor).size as usize * elem_size` at lines 498, 3174 — zero
  bytes is correct for empty.
* `(*out).size as usize` for buffer sizing at lines 2159, 2329, 2708, 2916
  — zero is correct and consistent with the empty data buffer.
* `chelis_tensor_numel` at line 725 — the bug surface; should return the
  genuine size, including zero.

Removing both clamps makes `size` the genuine element count everywhere.

## Spec context

The known empty-`to_tensor([])` limitation is referenced in
`spec/design/chelis_project_plan.md:611-613` and
`spec/design/chelis_phase3_plan.md:1142`. Both phrase it as a downstream
workaround the standard library uses ("zero-match masks still hit the
existing empty-`to_tensor([])` limitation"). The active spec does not pin
`numel` of an empty rank-1 tensor to either value; the fix aligns the
runtime with the empty-product convention already used by
`crates/chelis-ir/src/eval.rs::numel`, which returns `shape.iter().product()`
without clamping.

This diagnosis does not propose a spec change. The fix lands in two runtime
clamp sites and one host-evaluator clamp site; the spec remains unchanged.

## Expected fix shape

1. In `crates/chelis-compiler-api/src/runtime.rs::eval_builtin "numel"`,
   drop the `.max(1)`. Return `tensor.value.shape.iter().product::<usize>()`
   directly. Scalars (shape `[]`) still produce 1 via the empty-product
   identity; rank-1+ tensors with any zero dimension produce 0.

2. In `crates/chelis-runtime/src/lib.rs::chelis_alloc_tensor` and
   `chelis_alloc_view`, drop the `if tensor.size == 0 { tensor.size = 1; }`
   blocks. The `posix_memalign` call already runs `bytes.max(1)`; size is
   the logical element count and must stay zero for empty tensors.

Both fixes are local. No other runtime consumer of `.size` depends on the
clamp; loops, memset, and buffer-allocation paths all handle zero
correctly.

## Sibling sweep candidates

Coral red-team reports F2/F3/F4 (empty-result crashes in `filter`/`head`/
`tail`/`slice`, and the from_pairs length check on mixed-type empty frames)
trace upstream to this bug: any downstream code path that does
`numel(some_tensor) == 0 ? early_exit : work` is fooled into doing work on
a zero-length tensor. The fix at the two runtime sites is upstream of all
the downstream symptoms.

Per-primitive verification belongs in commit 3 (sibling sweep), not in
the diagnosis. If any sibling primitive turns out to have its own
independent empty-input bug (rather than being downstream of this one),
that is a separate §5 follow-on, not a scope expansion of this PR.
