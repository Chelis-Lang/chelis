# C-backend `Cast` bit-preserving memcpy diagnosis

Diagnosis pass for the bundled fix on branch `fix/cbackend-cast-elementwise`.
CBackend-CastMemcpy of the 0.7.6 red-team v2 closeout (PR #58).

Cross-references:

- Parallel runtime fix: PR #59 (`fix: eval_cast accepts RuntimeValue::Tensor (V2-F2)`),
  diagnosis at `docs/archive/investigations/eval_cast_tensor_diagnosis.md`. That
  PR closed the host-runtime `eval_cast` gap; this branch closes the
  matching C-backend `emit_cast` gap.
- Failing-test pins for this branch:
  `crates/chelis-cli/tests/cbackend_cast_memcpy.rs` (commit
  `dd61093` of this branch).

No code changes in this commit.

## Spec section (quoted, do not edit)

`spec/03-deep-syntax.md` line 319 (Transforms table):

```
| `cast` | `(cast {} expr target-type)` | Precision cast |
```

`spec/03-deep-syntax.md` line 353 (Tag Count Summary) lists `cast` in
the "Transforms" category. `crates/chelis-ir/src/verify.rs` C8 (around
line 422) pins the shape invariant: a `Cast` op may change precision
but must not change tensor dimensions; the output tensor has the same
shape as the input.

The spec phrase "Precision cast" is the load-bearing semantic: the
operation converts element values between numeric precisions. It is
not a bit-reinterpret cast.

## What works today (after PR #59)

- `chelis check` accepts `cast(tensor[N, p], q)` and types it
  `tensor[N, q]`.
- `chelis eval --file` evaluates element-wise precision conversion via
  `crates/chelis-compiler-api/src/runtime.rs::eval_cast` (PR #59), which
  dispatches through `cast_tensor_value` -> `convert_scalar_data` to
  produce per-element converted values.
- `chelis build --target c` emits, lowers, and produces a `.c` /
  `.h` pair plus a usable static lib.

## What does not work -- C-backend `emit_cast`

File: `crates/chelis-backend-c/src/emit.rs`, lines 2858--2866:

```rust
// ---- Cast ----
fn emit_cast(&mut self, id: usize, inputs: &[NodeId], ty: &TensorType) {
    // Phase 0f only supports casts to f32, validated before emission.
    let a = inputs[0].0;
    self.emit_slot_wrapper(id, ty);
    self.line(&format!(
        "memcpy(t{id}->data, t{a}->data, t{id}->size * sizeof(float));"
    ));
}
```

Three problems in one site:

1. Bit-preserving copy regardless of source precision. The source
   tensor's bytes are written into the destination tensor's buffer
   verbatim. No per-element precision conversion happens.
2. Hard-coded `sizeof(float)` for the byte count. When the *output*
   precision is f64 or int64, the destination buffer is `size * 8`
   bytes but only `size * 4` bytes are copied; the upper half is left
   zero-initialized from `chelis_alloc`'s `memset`. When the *input*
   precision is f64 or int64 and the output is f32 or int32, only the
   first 4 bytes of each 8-byte source element are read, so half the
   source data is silently dropped.
3. The comment "Phase 0f only supports casts to f32, validated before
   emission" is stale -- the validator at line 558 (cited below) now
   admits F32, F64, Int32, Int64 cast targets; the comment was written
   before that broadened.

The pre-emission validator
(`crates/chelis-backend-c/src/emit.rs` line 558) accepts:

```rust
if let RiscOp::Cast { new_precision } = node.op
    && !matches!(
        new_precision,
        Prim::F32 | Prim::F64 | Prim::Int32 | Prim::Int64
    )
{
    panic!(...);
}
```

So the four precision crossings the runtime evaluator supports
(f32 <-> f64, int32 <-> f32, and the various widenings / narrowings
between them) all reach `emit_cast` and all produce wrong output.

## Empirical reproduction

Reproduced against `origin/main` at commit `a54ca33` (the HEAD that
already includes PR #59).

Program:

```chelis
def cast_demo(x: tensor[3, f32]) -> tensor[3, f64] = {
  cast(x, f64)
}
```

Emitted kernel body (relevant lines):

```c
chelis_tensor *t1 = chelis_alloc_view(1, (int[]){ 3 }, CHELIS_F64, chelis_slot0->data);
memcpy(t1->data, t0->data, t1->size * sizeof(float));
```

Harness input `[1.5f, 2.5f, 3.5f]`. Output bytes after `cast_demo`:

| offset (byte) | hex | meaning                  |
|---------------|-----|--------------------------|
| 0..3          | `00 00 c0 3f` | `1.5f` bits          |
| 4..7          | `00 00 20 40` | `2.5f` bits          |
| 8..11         | `00 00 60 40` | `3.5f` bits          |
| 12..23        | `00 00 ..`    | zero-padded by alloc |

Reading those 24 bytes as three f64s yields
`[8.0000018998980522, 5.3360734001323986e-315, 0.0]` -- silent data
corruption.

The four fixtures in `crates/chelis-cli/tests/cbackend_cast_memcpy.rs`
record similarly concrete divergences for each conversion shape:

| Test                                  | Expected         | Observed (today)                              |
|---------------------------------------|------------------|-----------------------------------------------|
| `cbackend_cast_tensor_f32_to_f64`     | `1.5 2.5 3.5`    | `8.0000018998980522 5.3360734001323986e-315 0` |
| `cbackend_cast_tensor_f64_to_f32`     | `1.5 2.5 3.5`    | `0 1.9375 0`                                  |
| `cbackend_cast_tensor_f32_to_int32`   | `1 2 3`          | `1069547520 1075838976 1080033280`            |
| `cbackend_cast_tensor_int32_to_f32`   | `1 2 3`          | `1.40129846e-45 2.80259693e-45 4.20389539e-45` |

## How PR #59 fixes the runtime

`crates/chelis-compiler-api/src/runtime.rs` (around line 2767):

- `cast_tensor_value(tensor, target)` parses the target Prim, then
  walks the tensor's f64-storage data and applies
  `convert_scalar_data(x, src_prim, target_prim)` element-wise.
- `convert_scalar_data` (line 2789) projects ints/bools through `i64`
  storage, floats stay floats; emits each value in the target
  precision's storage convention (`(x as f32) as f64` for narrowing,
  `(x as i32) as f64` for truncating, etc.). Bool decodes via
  `!= 0.0`. Identity early-return for `src == dst`.
- Output preserves shape (Spec C8) and switches only the `precision`
  tag plus the data values.

The C backend's RiscOp::Cast represents the same operation on
contiguous typed buffers. The fix mirrors `convert_scalar_data`'s
semantics in emitted C: read one element from the source buffer at
source precision, write one element to the destination buffer at
target precision, with `(target_t)src[i]` conversion for the cross
arms and `dst[i] = src[i]` (after pointer typing) for the identity
case.

## Planned fix shape

`emit_cast` accepts the source DAG so it can fetch the *input's*
output_type (the source precision). For each `(src_prim, dst_prim)`
pair drawn from `{F32, F64, Int32, Int64}` (the set the validator
already restricts to):

1. Allocate the output slot via `emit_slot_wrapper(id, ty)` as
   today (this correctly sizes the buffer using `dtype_macro`).
2. Emit a strided element-wise loop (matching `emit_realize`'s shape)
   that:
   - reads `((src_et*)t{a}->data)[src_idx]` at the source's element
     type and stride convention,
   - writes `((dst_et*)t{id}->data)[i] = (dst_et)<read>` with a C-level
     primitive cast.
3. For identity `src_prim == dst_prim`, fall through to the same
   loop -- C's `(dst_et)src` is identity when types match, and the
   loop respects the source's strides (which a plain memcpy already
   gets wrong on non-contiguous inputs).

`bool` is excluded from cast targets by the existing validator
(`Prim::F32 | Prim::F64 | Prim::Int32 | Prim::Int64`); the fix
preserves that exclusion.

`bf16` and `int8 / int16` are not in the current cast target set; the
validator rejects them before emission with a clear panic, so no
emission code is needed for those precisions. If the validator
broadens later, the element-wise loop generalizes naturally because
`elem_type` already maps Prim -> C type for the supported precisions
(F32 -> `float`, F64 -> `double`, Int32 -> `int32_t`, Int64 -> `int64_t`).

The four fixtures in `crates/chelis-cli/tests/cbackend_cast_memcpy.rs`
are flipped from `#[ignore]` to running in the fix commit.

## Sibling sweep

Other `memcpy` call sites in `crates/chelis-backend-c/src/emit.rs`
were inspected (lines 2009 and 2111 in `emit_scatter_add` and
`emit_scatter`). Both use:

```rust
"memcpy(t{id}->data, t{id}_target->data, (size_t)t{id}->size * {target_elem_size});"
```

with `target_elem_size = format!("sizeof({})", target_et)` derived
from the target tensor's element type. The scatter validator
(`crates/chelis-backend-c/src/emit.rs` lines 627--653) requires
`updates_ty.precision == target_ty.precision == output.precision`,
so these are precision-preserving copies and are sound. Listed for
completeness; no change.

One related (but out-of-scope) bug was observed in host emission at
`crates/chelis-backend-c/src/host_emit.rs::append_tensor_reshape_helper`
line 276:

```rust
"    memcpy(out_tensor->data, input->data, (size_t)input->size * sizeof(float));"
```

The host reshape helper preserves dtype (`out_tensor` is allocated
with `input->dtype`) but the `sizeof(float)` constant only copies the
first 4 bytes of each element. For F64 / Int64 inputs the upper half
of each element is dropped. This is a separate bug class (host-side
reshape, not Cast), tracked here as a finding; the fix on this branch
does not touch it. Filing as a sibling-sweep observation rather than
expanding scope.

## Acceptance for this branch

The fix commit on this branch passes the four pinned fixtures in
`crates/chelis-cli/tests/cbackend_cast_memcpy.rs` with their `#[ignore]`
attributes removed. Each fixture compiles the emitted C with `gcc` and
runs the binary, so the assertion is on observed output bytes, not on
the emitted source text. The full workspace `cargo test --workspace`,
clippy, fmt, and `chelis lint --check` gates remain green.
