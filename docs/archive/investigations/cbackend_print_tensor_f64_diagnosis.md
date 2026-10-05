# C-backend `chelis_print_tensor_stdout` f64 read diagnosis

Diagnosis pass for the bundled fix on branch
`fix/cbackend-print-tensor-f64`. CBackend-PrintTensorF64 is the third
instance of the C-backend precision-lookup bug class closed in this
branch. PR #64 (CBackend-CastMemcpy) and PR #67
(CBackend-ReshapeMemcpy) fixed the storage side of the dtype-ignoring
pattern: `emit_cast` now performs element-wise precision conversion,
and the host reshape helper's `memcpy` uses the correct
source-dtype element size. This PR fixes the read side: the print
routine emitted into every C build reads element bytes at a
hard-coded `float *` width regardless of the tensor's dtype.

The two storage-side bugs were masked in normal Chelis-level testing
because the print routine cancelled them. A reshape of an f64 tensor
copied only the low halves of each element; the print routine then
read those low halves back as `float`s and rendered them as decimals
that looked correct for hand-authored examples like
`[1.5, 2.5, 3.5, 4.5]` (whose low 32 bits as IEEE 754 are exactly
`0.0`, which the printer elides with the same `%.1f` format used for
the value-side). PR #67's diagnosis at
`docs/archive/investigations/cbackend_reshape_memcpy_diagnosis.md` lines
127-138 explicitly named this site as the masking bug:

> The buggy `chelis_print_tensor_stdout` helper in the same emitted
> file (line 220-228 of `host_emit.rs::append_tensor_print_helper`)
> reads elements as `t->data[i]` where `t->data` is typed `float*`
> regardless of dtype, so it widens 4-byte chunks. ... two distinct
> dtype-ignoring bugs cancel each other in the common case.

Cross-references:

- PR #64 (CBackend-CastMemcpy), diagnosis at
  `docs/archive/investigations/cbackend_cast_memcpy_diagnosis.md`. Fixed the
  storage side of `Cast` by emitting an element-wise loop with the
  right C types per source and target precision.
- PR #67 (CBackend-ReshapeMemcpy), diagnosis at
  `docs/archive/investigations/cbackend_reshape_memcpy_diagnosis.md`. Fixed
  the storage side of the host reshape helper by deriving
  `elem_size` from `input->dtype` at runtime.
- Failing-test pins for this branch:
  `crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` (commit
  `5396292`).

No code changes in this commit.

## What the print routine does today

File: `crates/chelis-backend-c/src/host_emit.rs`, lines 206-232,
`append_tensor_print_helper`:

```rust
out.push("static void chelis_print_tensor_stdout(const chelis_tensor* t) {".to_string());
// ...
out.push("    int64_t limit = t->size < 32 ? t->size : 32;".to_string());
out.push("    for (int64_t i = 0; i < limit; ++i) {".to_string());
out.push("        double value = t->data[i];".to_string());
// ...
```

The C type of `t->data` is `float *`, declared at
`crates/chelis-runtime/include/chelis_runtime.h` line 20:

```c
typedef struct {
    float *data;
    int shape[CHELIS_MAX_DIM];
    // ...
    int dtype;
    int size;
    int owns_data;
} chelis_tensor;
```

So `t->data[i]` is a 4-byte `float` load at stride `sizeof(float)`,
implicitly widened to `double` for the call to `printf("%.1f", ...)`.
For `CHELIS_F32` and `CHELIS_I32` tensors this is the right access
width; for `CHELIS_F64` and `CHELIS_I64` tensors the load reads only
the low half of each 8-byte element and advances by 4 bytes, so
adjacent elements interleave: element `2*k` of the print sequence is
the low half of the k-th source element, and element `2*k+1` is the
high half. The tail of the buffer past `size * 4` bytes is read as
out-of-bounds memory beyond the allocation when the storage side is
fixed (post-PR-#67); with the storage side also broken (pre-PR-#67),
the tail bytes were zeros from `chelis_alloc`'s `memset` of the
oversized buffer.

The bool dtype is also a 4-byte element (`chelis_alloc` allocates
`size * sizeof(float)` for it), so the print routine is correct
there as well.

Validated dtype set, from `chelis_runtime.h` lines 12-16:

```
CHELIS_F32  = 0   (4 bytes)
CHELIS_F64  = 1   (8 bytes)
CHELIS_I32  = 2   (4 bytes)
CHELIS_BOOL = 3   (4 bytes)
CHELIS_I64  = 4   (8 bytes)
```

This is a closed set, mirroring the runtime's `chelis_alloc`
(`crates/chelis-runtime/src/lib.rs` lines 418-422) which sizes the
buffer with the same `dtype == F64 || dtype == I64 ? 8 : 4` table.

## Empirical reproduction

Reproduced against `origin/main` at commit `0bbf8a4` (post-PR-#68,
includes both PR #67 and PR #68). Three fixtures in
`crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` (commit
`5396292`).

f64 fixture source:

```chelis
def to_f64(x: tensor[4, f32]) -> tensor[4, f64] = cast(x, f64)
src = to_tensor([1.5, 2.5, 3.5, 4.5])
result = to_f64(src)
```

`chelis eval --file` ground truth:

```
src = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])
result = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])
```

`chelis build --target c` + gcc + run observed:

```
src = tensor(shape=[4], data=[1.5, 2.5, 3.5, 4.5])
result = tensor(shape=[4], data=[0.0, 1.9375, 0.0, 2.0625])
```

Byte-level explanation. The f64 values `1.5`, `2.5`, `3.5`, `4.5`
have IEEE 754 bit patterns whose low 32 bits are all `0x00000000`
(the bits of `+0.0f`) and whose high 32 bits encode the value at a
shifted exponent. The print routine reads at stride 4:
- iteration 0: bytes 0..3 of element 0 = `0x00000000` as f32 = `0.0`
- iteration 1: bytes 4..7 of element 0 = high half of f64 `1.5` =
  `0x3FF80000` as f32 = `1.9375`
- iteration 2: bytes 8..11 of element 1 = `0x00000000` = `0.0`
- iteration 3: bytes 12..15 of element 1 = high half of f64 `2.5` =
  `0x40040000` as f32 = `2.0625`

The print loop runs `min(size, 32) == 4` iterations, so it stops
after reading 16 bytes (the first 2 source elements in two halves
each). The last 16 bytes of the f64 buffer (elements 2 and 3) are
never read.

int64 fixture source:

```chelis
def to_i64(x: tensor[4, int32]) -> tensor[4, int64] = cast(x, int64)
src = to_tensor([cast(100000, int32), cast(200000, int32),
                 cast(300000, int32), cast(400000, int32)])
result = to_i64(src)
```

Eval ground truth:

```
src = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])
result = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])
```

C-build observed:

```
src = tensor(shape=[4], data=[100000.0, 200000.0, 300000.0, 400000.0])
result = tensor(shape=[4], data=[100000.0, 0.0, 200000.0, 0.0])
```

The int64 values `100000`, `200000`, `300000`, `400000` all fit in
the low 32 bits, so the low halves of each element are
`0x000186A0`, `0x00030D40`, etc. Read as f32 those bit patterns are
small subnormals/normals that the print routine formats with the
"close to round" branch: `0x000186A0` as f32 is `8.96831e-40` which
rounds to `0.0` -- but the print routine's "is integer" test
(`fabs(value - round(value)) < 1e-9`) catches it and renders as
`0.0`. The high 32 bits of the int64 elements are all zero, so
iterations 1, 3, 5, 7 read `0.0` regardless. The interleaving means
the printer reports source element `k`'s low half at print position
`2*k` and the source's high half at `2*k+1`. For our small positive
int64s that pattern reads as `[100000, 0, 200000, 0]`. The last
two source elements (positions 4, 5, 6, 7 in the print loop) would
be read but the loop only runs `size == 4` iterations.

The f32 control passes today (4-byte read width matches f32 element
width).

## How PR #64 and PR #67 fix the parallel storage sites

PR #64 (`crates/chelis-backend-c/src/emit.rs::emit_cast`) takes a
`&Dag` so it can read the source tensor's `output_type`, then maps
both source and target Prims to C element types via
`Self::elem_type(ty)`. The emitted body is a strided element-wise
loop using the correct casts at both load and store:

```rust
let src_et = Self::elem_type(src_ty);
let dst_et = Self::elem_type(ty);
// ...
"(({dst_et}*)t{id}->data)[i] = ({dst_et})(({src_et}*)t{a}->data)[idx];"
```

This is the pattern the print routine must adopt: cast `t->data` to
the C type matching the runtime `dtype` before reading.

PR #67 (`host_emit.rs::append_tensor_reshape_helper`) derives
`elem_size` at runtime from `input->dtype`:

```rust
out.push(
    "    size_t elem_size = (input->dtype == CHELIS_F64 || input->dtype == CHELIS_I64) ? sizeof(int64_t) : sizeof(float);"
        .to_string(),
);
```

The reshape case only needs the byte count, not a typed read, so the
PR #67 fix is a switch on element size. The print case needs both the
byte stride and the C type of the load -- it must dispatch on
`t->dtype` to a typed loop body. The fix shape below mirrors PR #67's
runtime-dtype dispatch and PR #64's typed-cast pattern combined.

## Planned fix shape

`append_tensor_print_helper` emits C source with no access to a
Rust `Prim` or `TensorType` -- the bytes it emits run at runtime
against a `chelis_tensor *t` whose `dtype` field is the source of
truth. The fix dispatches on `t->dtype` inside the emitted print
loop:

```c
double value;
switch (t->dtype) {
    case CHELIS_F64:  value = ((double*)t->data)[i]; break;
    case CHELIS_I64:  value = (double)((int64_t*)t->data)[i]; break;
    case CHELIS_I32:  value = (double)((int32_t*)t->data)[i]; break;
    case CHELIS_BOOL: value = (double)((float*)t->data)[i]; break;
    case CHELIS_F32:
    default:          value = (double)((float*)t->data)[i]; break;
}
```

The dtype set is closed (`runtime.h` lines 12-16). `CHELIS_BOOL`
elements are stored as 4-byte float values (1.0 or 0.0) by
`chelis_alloc` -- see `runtime.rs` lines 418-422 -- so the f32 read
path is correct for it. The `default` arm covers any future dtype
that aliases the 4-byte float read; the explicit `CHELIS_F32` arm
locks the canonical f32 read.

The cast to `double` for `int32_t` and `int64_t` matches the
existing implicit widening: the print routine renders integer
elements through the same `%.1f` / `%.16g` path that floats use,
which is consistent with the runtime evaluator's printer
(`crates/chelis-shell/src/print.rs` and friends).

The fix commit on this branch flips the two `#[ignore]`'d fixtures
in `crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` to running
and keeps the f32 control passing.

## Sibling sweep

Per the brief, the C backend and the runtime were grep-swept for any
other hard-coded `float *` read or `t->data[...]` access that
ignores `t->dtype` and would silently corrupt elements for f64 / i64
tensors.

### `crates/chelis-backend-c/src/host_emit.rs`

- L221: the bug fixed by this branch.
- L1562, L1598, L1624, L1650: emitted host-side element-wise binary,
  ternary, and unary ops write directly into `target->data[i]` and
  read directly from `lhs->data[idx]` / `rhs->data[idx]`. These are
  the host fallback path (the DAG path uses `emit_dag` and typed
  loops); the host path implicitly assumes f32 storage and produces
  silently wrong results for f64 / i64 tensors. Same bug class. Not
  fixed here; flagged for §5 follow-on entry coordinated through the
  orchestrator.
- L1728, L1732: scalar-to-tensor seed paths write
  `tensor->data[0] = ... 1.0f` and `(float)(...)`. Same bug class;
  loses upper 4 bytes on f64 / i64. Not fixed here.
- L2308, L2425: scatter-style writes use typed buffer pointers
  (`{target}_out_data`) that are already cast upstream to the right
  C type. Sound.

### `crates/chelis-runtime/src/lib.rs`

The `chelis_tensor.data` field is declared `*mut f32` in Rust (line
31), so every `(*tensor).data.add(i)` in runtime functions performs
f32-sized pointer arithmetic and reads/writes f32 values. The
runtime currently uses this surface as if every tensor is f32. The
following functions touch `data` and would corrupt f64 / i64
tensors:

- `chelis_list_from_tensor` (L1626): reads `raw = *(*tensor).data.add(...)`
  as f32, then matches on dtype and converts. Bug for f64 / i64.
- `chelis_pad_sequences` (L1679): writes `value as f32`. Bug for f64 / i64.
- `chelis_concat`, `chelis_split`, `chelis_gather`, `chelis_scatter`,
  `chelis_where`, `chelis_cumsum`, `chelis_sort`, `chelis_diagonal`,
  `chelis_trace`, `chelis_clamp`, `chelis_einsum`, `chelis_cmplt`
  (L1772, L1815, L1869, L1884, L1897, L1952, L1970, L1992, L2022,
  L2051, L2131, L2167, L2190, L2197, L2314, L2582): all access
  `data` as `*mut f32`. Bug for f64 / i64.

This is a wide systemic surface. The runtime's `data` type is a
foundational design choice that predates the multi-precision
work; the right fix is to retype `data` to `*mut u8` (or carry a
typed pointer per dtype) and route every access through a
dtype-dispatched accessor. That is well beyond the scope of this
PR. The brief explicitly restricts this PR to the print site, with
the sibling sweep documented but unfixed and flagged for follow-on.

Findings summary:

- C-backend host_emit: 5 same-class sites beyond the print routine
  (L1562, L1598, L1624, L1650, and the pair at L1728/L1732). All
  silent-data-corruption on f64 / i64. §5 follow-on candidate.
- chelis-runtime/src/lib.rs: ~20 same-class sites; foundational
  design coupling. §5 follow-on candidate, likely larger than a
  single PR.
- HIP and Metal backends: out of scope per the brief.

No additional fixes in this commit.

### Why the bug class survives

The `chelis_tensor.data` field's C-side type (`float *`) is the
common denominator. PR #59 fixed the host-evaluator path that did
not depend on that field. PR #64 and PR #67 fixed C-backend storage
sites that wrote correctly-sized bytes but used `float *` as a
byte-pointer alias to do so. This PR fixes the read side of the
same field's misuse. The remaining sites in `host_emit.rs` and the
runtime continue to alias `float *` as a byte pointer and will keep
producing silent data corruption on multi-precision tensors until
the field's type contract is widened.

## Acceptance for this branch

The fix commit on this branch passes the three pinned fixtures in
`crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` with the two
`#[ignore]` attributes removed. Each fixture builds the source
through `chelis build --target c`, compiles the emitted C with
`gcc`, runs the binary, and asserts that stdout matches the
ground-truth `chelis eval --file` output character for character.
The full workspace `cargo test --workspace`, clippy, fmt, and
`chelis lint --check` gates remain green.
