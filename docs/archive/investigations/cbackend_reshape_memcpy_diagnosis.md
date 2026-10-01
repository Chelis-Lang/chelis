# C-backend host reshape `sizeof(float)` memcpy diagnosis

Diagnosis pass for the bundled fix on branch
`fix/cbackend-reshape-elementsize`. CBackend-ReshapeMemcpy is the
second instance of the same bug class closed by PR #64
(CBackend-CastMemcpy): a host-emitted `memcpy` whose byte count is
hard-coded to `sizeof(float)` regardless of the source dtype, causing
silent data corruption for any tensor with element size greater than
four bytes (f64, int64).

Cross-references:

- Parallel C-backend fix: PR #64
  (`fix: C-backend emit_cast performs element-wise precision
  conversion`), diagnosis at
  `docs/archive/investigations/cbackend_cast_memcpy_diagnosis.md`. That PR's
  sibling-sweep section flagged the present site at
  `crates/chelis-backend-c/src/host_emit.rs::append_tensor_reshape_helper`
  line 276 and explicitly left it out of scope.
- Failing-test pins for this branch:
  `crates/chelis-cli/tests/cbackend_reshape_memcpy.rs` (commit
  `18e088a`).

No code changes in this commit.

## Spec section (quoted, do not edit)

`spec/03-deep-syntax.md` line 383 (Movement category):

```
**Movement:** `reshape`, `permute`, `expand`, `pad`, `shrink`, `stride`
```

`reshape` is a "Movement" transform: it changes the dimension layout
but does not touch the elements or their precision. The output tensor
holds exactly the same elements as the input, with the same dtype, in
row-major order. `crates/chelis-ir/src/verify.rs` (the C8 family of
shape invariants) constrains the operator analogously for the DAG
side. For the host side, the runtime contract is that
`chelis_alloc(ndim, shape, input->dtype)` allocates `size * elem_size`
bytes where `elem_size = 8` for `CHELIS_F64 | CHELIS_I64` and `4`
otherwise (`crates/chelis-runtime/src/lib.rs::chelis_alloc`,
lines 418-422).

## What works today

- `chelis check`, `chelis fmt`, and `chelis eval --file` agree that
  reshape preserves both shape (modulo the new dims) and dtype.
- The DAG-emit path (`chelis-backend-c/src/emit.rs::emit_reshape`,
  L2643) does not have this bug. It uses `chelis_alloc_view` plus
  `chelis_contiguous` for the non-contiguous case; the byte-copy
  inside `chelis_contiguous` uses a dtype-aware `elem_size`
  (`chelis-runtime/src/lib.rs` L2480-2498), so a reshape lowered
  through the DAG path is sound for every supported precision.
- The host-emit path's helper *signature* is correct: the destination
  is allocated via `chelis_alloc(ndim, shape, input->dtype)` so the
  output buffer is the right size and `out_tensor->dtype` is correct.
  Only the `memcpy` size argument is wrong.

## What does not work -- host-emit reshape helper

File: `crates/chelis-backend-c/src/host_emit.rs`, lines 234-280
(`append_tensor_reshape_helper`). Bug at line 276:

```rust
out.push(
    "    memcpy(out_tensor->data, input->data, (size_t)input->size * sizeof(float));"
        .to_string(),
);
```

The destination is the freshly allocated `out_tensor->data`. The
source is `input->data`, sized
`input->size * elem_size(input->dtype)` bytes by
`chelis_alloc` / `chelis_alloc_view`. The memcpy copies only
`size * sizeof(float)` = `size * 4` bytes regardless of the input's
dtype. Concretely:

- `CHELIS_F32` (4-byte elements): correct. `n * 4` covers the whole
  buffer. This is why the bug class went unnoticed initially.
- `CHELIS_F64` (8-byte elements): the low half of the buffer is
  copied; the high half of `out_tensor` is left zero from
  `chelis_alloc`'s `memset`. Reading the destination as `double*`
  yields the source's first `n/2` f64 elements in slots `[0..n/2]`
  and zeros in slots `[n/2..n]`. See empirical reproduction below.
- `CHELIS_I64` (8-byte elements): same shape of byte-drop as F64.
- `CHELIS_BOOL` (4-byte elements per `chelis_alloc`): correct.
- `CHELIS_I32` (4-byte elements): correct.

## Empirical reproduction

Reproduced against `origin/main` at commit `c7469d9` (the HEAD that
includes PR #64). Three fixtures in
`crates/chelis-cli/tests/cbackend_reshape_memcpy.rs` (commit
`18e088a`) compile a Chelis program through
`chelis build --target c`, patch the emitted kernel to expose the
static helper, link a custom `main.c` that drives the helper with raw
f64 / int64 / f32 input, and assert the printed destination buffer
matches the input element-wise.

Source program (the f64 case):

```chelis
module Demo
src = cast(to_tensor([1.5, 2.5, 3.5, 4.5]), f64)
result = reshape(src, [cast(2, int64), cast(2, int64)])
```

Harness input bytes (32 bytes; four 8-byte f64s): `1.5, 2.5, 3.5, 4.5`.

Observed today (with `#[ignore]` removed):

| Fixture                              | Expected                                                       | Observed (today)                                              |
|--------------------------------------|----------------------------------------------------------------|----------------------------------------------------------------|
| `cbackend_reshape_tensor_f64`        | `1.5 2.5 3.5 4.5`                                              | `1.5 2.5 0 0`                                                  |
| `cbackend_reshape_tensor_int64`      | `123456789abcdef 1122334455667788 7fedcba987654321 11223344556677` | `123456789abcdef 1122334455667788 0 0`                          |
| `cbackend_reshape_tensor_f32_control` | `1.5 2.5 3.5 4.5`                                              | `1.5 2.5 3.5 4.5` (passes today)                               |

Byte-level explanation for the f64 case. The source buffer is 32 bytes
of f64 data. The destination is allocated as `4 * sizeof(double)` = 32
bytes, zero-filled by `chelis_alloc`. The buggy memcpy copies
`4 * sizeof(float)` = 16 bytes. So bytes 0..15 of the destination hold
the f64 bit-patterns of `1.5` and `2.5`; bytes 16..31 stay zero.
Reading the destination as `double*[4]` yields
`[1.5, 2.5, 0.0, 0.0]`.

The buggy `chelis_print_tensor_stdout` helper in the same emitted file
(line 220-228 of `host_emit.rs::append_tensor_print_helper`) reads
elements as `t->data[i]` where `t->data` is typed `float*` regardless
of dtype, so it widens 4-byte chunks. For an f64 destination this
prints the float bit-patterns of the low halves of each element pair,
which by coincidence look "fine" for the small positive f64s used in
hand-authored reshape examples. That is why this bug class was not
visible through normal Chelis-level reshape testing -- two distinct
dtype-ignoring bugs cancel each other in the common case. The
adversarial harness in `cbackend_reshape_memcpy.rs` bypasses the
buggy printer by reading the destination buffer directly at the
correct C element type.

## How PR #64 fixed the parallel `emit_cast` site

`crates/chelis-backend-c/src/emit.rs::emit_cast` (post-PR-#64, around
line 2873) takes a `&Dag` so it can pull the source tensor's
`output_type`, then maps both source and target Prims to C element
types via `Self::elem_type(ty)` (a closed `match` over Prim that
returns `&'static str`). The emitted body is a strided element-wise
loop, not a memcpy:

```rust
let src_et = Self::elem_type(src_ty);
let dst_et = Self::elem_type(ty);
// ...
"(({dst_et}*)t{id}->data)[i] = ({dst_et})(({src_et}*)t{a}->data)[idx];"
```

The reshape case is structurally simpler than cast because dtype is
preserved end-to-end -- no element conversion is required. The only
fix needed is to swap the hard-coded `sizeof(float)` for the correct
per-dtype element size, derived at runtime from `input->dtype` (which
is the same value the upstream `chelis_alloc` call already passes
into the runtime to size the destination buffer).

## Planned fix shape

`append_tensor_reshape_helper` is C-source emission with no access to
a Rust `Prim` or `TensorType` -- the bytes it emits run at runtime
against a `chelis_tensor *input` whose `dtype` field is the source of
truth. The fix derives `elem_size` from `input->dtype` in emitted C:

```c
size_t elem_size = (input->dtype == CHELIS_F64 || input->dtype == CHELIS_I64)
    ? sizeof(int64_t) : sizeof(float);
memcpy(out_tensor->data, input->data, (size_t)input->size * elem_size);
```

This mirrors the dtype-to-size mapping in
`crates/chelis-runtime/src/lib.rs::chelis_alloc` lines 418-422 and
`chelis_contiguous` lines 2480-2484 verbatim. The `dtype` field is
written by `chelis_alloc` immediately before the data buffer is
allocated, so by the time `memcpy` runs both source and destination
buffers share `elem_size` bytes per element and the copy size is
`input->size * elem_size`.

`CHELIS_BOOL` and `CHELIS_I32` continue to use the 4-byte path. The
mapping is a closed set (`runtime.h` lines 12-16), so the branch is
total over the supported dtypes.

The fix commit on this branch flips the two `#[ignore]`'d fixtures in
`crates/chelis-cli/tests/cbackend_reshape_memcpy.rs` to running and
keeps the f32 control passing.

## Sibling sweep

Per the brief, the C backend was grep-swept for any other
`sizeof(float)` literal or `* 4` magic-number on memcpy or array-size
contexts. Findings:

`crates/chelis-backend-c/src/emit.rs`:

- L2861 in the doc-comment of `emit_cast`: cites the old buggy
  `memcpy(dst, src, n * sizeof(float))` shape as historical context.
  Not code.
- L2009 and L2111 in `emit_scatter_add` and `emit_scatter`: both use
  `(size_t)t{id}->size * {target_elem_size}` where `target_elem_size`
  is `format!("sizeof({})", target_et)` derived from the *target*
  tensor's element type. The scatter validator at L627-653 requires
  `updates_ty.precision == target_ty.precision == output.precision`,
  so this is a precision-preserving copy and the byte count is
  correct.
- L3919 in a test: asserts the emitted text contains
  `sizeof(double)` for an f64 scatter. Lock for the L2009/L2111 sites
  above.
- No literal `* 4` magic number in memcpy/array-size contexts.

`crates/chelis-backend-c/src/host_emit.rs`:

- L276: the bug fixed by this branch.
- L2368: `memcpy({target}->data, {target_ct}->data, (size_t){target}->size * {target_elem_size})`
  inside the host-emitted scatter path. `target_elem_size` is
  `format!("sizeof({})", target_elem_t)` derived from the validated
  target precision; precision-preserving copy. Sound.
- No other `sizeof(float)` literal in this file outside the bug.

`crates/chelis-backend-hip/src/`, `crates/chelis-backend-metal/src/`:
not in scope for this brief (sibling-sweep is "the entire C backend",
which the brief defines as `crates/chelis-backend-c/`). The
backend-hip and backend-metal trees are not swept here.

No additional bug-class sites were found. CBackend-ReshapeMemcpy is
the only remaining instance of the
`memcpy(..., n * sizeof(float))`-on-dtype-preserving-copy pattern in
the C backend. No follow-on §5 entry is needed from this sweep
beyond the one the orchestrator is filing in parallel for
CBackend-ReshapeMemcpy itself.

## Acceptance for this branch

The fix commit on this branch passes the three pinned fixtures in
`crates/chelis-cli/tests/cbackend_reshape_memcpy.rs` with the two
`#[ignore]` attributes removed. Each fixture compiles the emitted C
with `gcc` and runs the binary, so the assertion is on observed
output bytes, not on the emitted source text. The full workspace
`cargo test --workspace`, clippy, fmt, and `chelis lint --check`
gates remain green.
