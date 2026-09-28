# C Runtime Dtype Accessors: Migration Site Inventory and Anchor Templates

W2 PR 1 of the 0.7.8 compiler cleanup workstream. Locates the access
sites that the `TensorElement` trait closes, defines the migration
template Agent B (PRs 2-4) follows for each site, and pins the bool
and i32 routing convention that the orchestrator decision on PR #79
established.

Owning plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`,
W2 PR 1 brief.

Spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md` Contract
2 (trait surface) and Contract 3 (op enumeration).

§5 entry: `docs/archive/reports/gap_synthesis.md` `CRuntime-F32Coupling`.

PR 1 scope: introduces the trait, promotes `CHELIS_*` constants to
`pub const`, migrates two anchor ops (`chelis_tensor_to_f64` and
`chelis_fill_*`), adds a transition shim that the remaining 31+
runtime sites call so they keep compiling. Mechanical migration of
those sites is PR 2's scope; host_emit code-generation site
migration is PR 3's scope; cross-validation matrix completion is
PR 4's scope.

## Site inventory

The field `chelis_tensor.data` was statically typed `*mut f32`
before PR 1. The runtime stores tensor data in dtype-specific byte
layouts that `chelis_alloc` (`crates/chelis-runtime/src/lib.rs:471`
in the pre-PR-1 numbering, L554 post-PR-1) picks via the
`size_of::<i64>()` (for `CHELIS_F64`, `CHELIS_I64`) vs
`size_of::<f32>()` (for `CHELIS_F32`, `CHELIS_I32`, `CHELIS_BOOL`)
branch. The `*mut f32` field typing pretended every dtype was 4-byte
float; every access through `(*t).data.add(i)` indexed at f32 stride,
silently mis-decoding 8-byte storage.

### `crates/chelis-runtime/src/lib.rs` access sites

Line numbers below are post-PR-1 (after `pub const` promotion and
trait stub insertion, which shifted line numbers by ~80). The
pre-PR-1 numbers from `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md` Contract
3 still apply to `main` until commit 3 of this PR lands.

Migrated by PR 1 (anchor ops):
* L606 `chelis_tensor_to_f64` (read-side anchor): outer match on
  dtype, arms call `data_ptr_unchecked`.
* L572 `chelis_fill_i64`: body thins to `i64::fill(t, val)`.
* L564 `chelis_fill_f32`: body thins to `f32::fill(t, val)`.
* L580 `chelis_fill_f64`: body thins to `f64::fill(t, val)`.

Untouched by PR 1, must keep compiling via the transition shim
(`data_as_f32`) until Agent B migrates them in PR 2:
* L422 `chelis_zeros` (memcpy of `(*tensor).data` to `(*out).data`):
  byte-stride copy. PR 2 may keep the byte copy or migrate to a
  per-precision path; the storage-side dtype is in both operands so
  there is no decode ambiguity.
* L515 `chelis_alloc` final assignment `tensor.data = ptr as *mut
  f32;` becomes `tensor.data = ptr as *mut u8;` once the field
  changes.
* L558 `chelis_free` calls `(*t).data.cast()` to obtain `*mut
  c_void`; the cast still works from `*mut u8`. No migration needed.
* L590, L597 `chelis_scalar_tensor_from_i64` and
  `chelis_scalar_tensor_from_f64`: write a single element as `f32`.
  See "I32 storage encoding" below for the storage-truncation bug
  these surface; out of scope for PR 1.
* L1694 `chelis_flatten_nested_list` (write-side per-element): PR 2.
* L1709 `chelis_list_from_tensor` (read-side per-element): PR 2.
* L1762, L1809 `chelis_pad_sequences` and `_to` (write-side fill):
  PR 2.
* L1855 `chelis_tensor_concat` (memcpy-style per-element copy): PR 2.
* L1898 `chelis_tensor_split` (slice + copy): PR 2.
* L1952, L1967 `chelis_tensor_gather` (indexed read of f32): PR 2.
* L1980 `chelis_tensor_cmplt` (numeric compare -> bool): PR 2.
* L2035, L2053, L2055 `chelis_tensor_scatter` (indexed write): PR 2.
* L2075-L2078 `chelis_tensor_where` (cond-typed select): PR 2.
* L2105-L2106 `chelis_tensor_cumsum` (numeric prefix sum): PR 2.
* L2134, L2141-L2149 `chelis_tensor_sort` (numeric compare + indices):
  PR 2.
* L2214 `chelis_tensor_diagonal` (indexed read): PR 2.
* L2250-L2252 `chelis_tensor_trace` (diagonal sum): PR 2.
* L2271-L2282 `chelis_tensor_clamp` (numeric min/max): PR 2.
* L2397-L2408 `chelis_tensor_einsum` (numeric multiply-add): PR 2.
* L2571, L2579-L2580 `chelis_host_reshape_tensor` and
  `chelis_host_cast_tensor`: already dtype-aware byte-copy paths
  (PRs #64 and #67 fixed these directly). The pointer typing
  changes; the dtype dispatch is already correct.
* L2665 `chelis_print_tensor_stdout` host helper (read for
  tensor_to_string): PR 2.
* L2781-L2785 alignment assertions on `(*result).data`: no decode,
  just non-null + alignment check on the byte pointer. Cast to
  `*mut u8` retains the alignment check; no migration shape needed.

Total touch budget for PR 1: roughly two anchor ops (4 functions
including the `chelis_fill_*` trio) + the transition shim helper +
two trivial in-function field type adjustments at `chelis_alloc`
(`tensor.data = ptr as *mut u8;`) and call sites that previously cast
to `*mut f32` (e.g., `chelis_alloc_view` parameter). The expected
~33 sites listed in the plan and §5 entry are PRs 2-4's scope. PR 1
provides the shim so they still compile.

### `crates/chelis-backend-c/src/host_emit.rs` code-generation sites

These emit C code strings that directly index `t->data[i]`. The C
side keeps `float *data` for backward struct ABI; the emitted C
code casts at use. Migration is PR 3's scope.

* L1587 elementwise binary op (target index = lhs op rhs)
* L1623 elementwise binary `func()` call
* L1649 elementwise unary op
* L1675 elementwise unary `func()` call
* L1728 elementwise convert/store from constants
* L1732 boolean-storage write path

The PR #72 print routine (host_emit.rs:207-256) already
dtype-dispatches via `switch (t->dtype)` and reads through
`((const double*)t->data)[i]` and similar. PR 3 lifts the same
pattern to the elementwise helpers.

### HIP and Metal backends

Defensive grep across `crates/chelis-backend-hip/src/` and
`crates/chelis-backend-metal/src/` for `*mut f32`, `as *mut f32`,
`as *const f32`:

* `chelis-backend-hip`: no matches.
* `chelis-backend-metal`: no matches.

Both backends emit GPU kernel strings; the host-side host-emit
helpers they generate go through `crates/chelis-backend-c`
(target=c) or their own emitters which do not couple to `*mut f32`
at the Rust level. Out of scope for the W2 series, consistent with
the prior survey in the §5 entry.

## I32 storage encoding (out of PR 1 scope; informs the anchor
template)

The pre-PR-1 runtime stores `CHELIS_I32` tensors with 4-byte
elements where the byte pattern is the `f32` bit-encoding of the
intended integer value, not the `i32` bit pattern. Evidence:

* `chelis_scalar_tensor_from_i64` (L590): `*(*tensor).data = value
  as f32;` writes the f32 widened representation.
* `chelis_list_from_tensor` (L1709): for `CHELIS_I32` dtype, reads
  the buffer via `*(*tensor).data.add(i)` (typed `*mut f32`), then
  casts `raw as i64` to recover the int value. Truncates anything
  outside the f32 range with full precision (24-bit mantissa).
* `chelis_flatten_nested_list` (L1762): for `CHELIS_I32` leaf dtype,
  `*(*out).data.add(flat) = value as f32;`.

The trait impl `unsafe impl TensorElement for i32 { const DTYPE: c_int
= CHELIS_I32; }` is correct by the trait's contract about byte
size (i32 is 4 bytes, matching `chelis_alloc`'s 4-byte arm for
`CHELIS_I32`). But the impl's `i32::data_ptr_unchecked` cast to
`*mut i32` reads bytes as i32 -- which is the wrong decode for the
existing storage-side encoding. Using the trait at a CHELIS_I32 site
naively would expose the storage-side bug PR #72 already documented
(`cast(int32 -> int64)` produces wrong values).

**Anchor template policy for i32**: PR 1's migration of
`chelis_tensor_to_f64` routes the CHELIS_I32 arm through
`f32::data_ptr_unchecked` (NOT `i32::data_ptr_unchecked`) to match
the actual storage encoding. The same routing applies to bool. This
matches the runtime's existing decode at L1709
(`chelis_list_from_tensor`).

The `i32` trait impl exists for future use by code that genuinely
stores i32 byte patterns (e.g., new ops that allocate and fill via
`i32::fill` directly). The fix commit's
`chelis_scalar_tensor_from_i64` and other writers stay on the
f32-encoded convention.

**Recommended new §5 entry** (orchestrator files post-merge):
"int32 storage representation in C runtime: switch from 4-byte
f32-encoded to 4-byte i32-bit-pattern across writers, readers, and
all dispatch sites. Surface bug: PR #72's sibling sweep finding
that `cast(int32 -> int64)` reads f32 bit patterns through
`(int32_t*)src->data`. Scope: rewrite
`chelis_scalar_tensor_from_i64`,
`chelis_flatten_nested_list::CHELIS_I32` arm,
`chelis_list_from_tensor::CHELIS_I32` arm, every CHELIS_I32 dispatch
arm Agent B migrates in PRs 2-4."

## Migration template (Agent B follows this for every site)

For sites that read or write the data buffer with dtype known at
the call site, prefer `data_ptr_unchecked` inside an outer dtype
match. The match arm is the safety contract; the unchecked accessor
has zero runtime check cost in release builds.

```rust
unsafe {
    match (*t).dtype {
        CHELIS_F32 => {
            let p = f32::data_ptr_unchecked(t);
            for i in 0..(*t).size as isize {
                // typed f32 work
            }
        }
        CHELIS_F64 => {
            let p = f64::data_ptr_unchecked(t);
            // typed f64 work
        }
        CHELIS_I64 => {
            let p = i64::data_ptr_unchecked(t);
            // typed i64 work
        }
        // I32 and BOOL route through f32::data_ptr_unchecked while
        // the storage stays f32-encoded.  See the "I32 storage
        // encoding" section.
        CHELIS_I32 => {
            let p = f32::data_ptr_unchecked(t);
            // Read raw f32, recover i32 via `*p as i32` (or i64).
            // Write via `*p = value as f32`.
        }
        CHELIS_BOOL => {
            let p = f32::data_ptr_unchecked(t);
            // Read: compare to 0.0.
            // Write: `*p = if value { 1.0 } else { 0.0 };`
        }
        other => runtime_fail!("op_name unsupported dtype {other}"),
    }
}
```

For sites where the dtype is not yet known and the trait can also
serve as a runtime check, use the checked variant:

```rust
let p = f64::data_ptr(t)?; // Err(DtypeMismatch { expected, actual })
```

PR 1's two anchor ops use the unchecked form (the dtype is already
matched in the outer dispatch). PR 4 may use the checked form for
the cross-validation harness's negative tests.

## Anchor op A: `chelis_tensor_to_f64` (read-side)

Pre-PR-1 body:

```rust
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    *(*t).data as f64
}
```

Bug: the single read `*(*t).data as f64` reads 4 bytes as f32 and
widens. For `CHELIS_F64` or `CHELIS_I64` storage the read drops the
upper 4 bytes; for `CHELIS_I32` and `CHELIS_BOOL` (both f32-encoded)
the read is correct by historical accident.

Post-PR-1 body:

```rust
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    let t = t as *mut chelis_tensor;
    match (*t).dtype {
        CHELIS_F32 => *f32::data_ptr_unchecked(t) as f64,
        CHELIS_F64 => *f64::data_ptr_unchecked(t),
        CHELIS_I64 => *i64::data_ptr_unchecked(t) as f64,
        // I32 and BOOL storage today is 4-byte f32-encoded.
        CHELIS_I32 => *f32::data_ptr_unchecked(t) as f64,
        CHELIS_BOOL => {
            if *f32::data_ptr_unchecked(t) != 0.0 { 1.0 } else { 0.0 }
        }
        other => runtime_fail!("chelis_tensor_to_f64 unsupported dtype {other}"),
    }
}
```

The fixtures in `crates/chelis-e2e/tests/dtype_op_matrix.rs::tensor_to_f64_*`
assert every arm reads back the filled value byte-exact.

## Anchor op B: `chelis_fill_i64` (write-side validation)

Pre-PR-1 body:

```rust
pub unsafe extern "C" fn chelis_fill_i64(t: *mut chelis_tensor, val: i64) {
    let ptr = (*t).data as *mut i64;
    for i in 0..(*t).size as isize {
        *ptr.offset(i) = val;
    }
}
```

Post-PR-1 body:

```rust
pub unsafe extern "C" fn chelis_fill_i64(t: *mut chelis_tensor, val: i64) {
    i64::fill(t, val);
}
```

The trait's default `fill` expands to the same cast + loop after
inlining. The `debug_assert_eq!` in `data_ptr_unchecked` is a no-op
in release builds. Mismatched-dtype calls now trip `debug_assert`
instead of silently writing through a stale-typed pointer.

`chelis_fill_f32` and `chelis_fill_f64` get the same treatment
(`f32::fill` and `f64::fill`).

## Destructure-`..` audit on `chelis_tensor`

Per the W2 PR 1 brief's discovery-contract constraint, any
pattern-match on `chelis_tensor` that uses `..` to ignore fields
must be audited: if the struct field type changes from `*mut f32`
to `*mut u8`, existing `..` consumers might silently mis-cast a
field whose type the consumer relied on for the previous decode
shape (compare the destructure-default footgun documented in
`feedback_destructure_dot_dot_is_default_arm.md`).

Grep across all crates for pattern matches on `chelis_tensor`:

```
rg 'chelis_tensor\s*\{' crates/ --type rust
```

Result: only the struct definition itself at `crates/chelis-runtime/src/lib.rs:30`
and the constructor inside `chelis_alloc` (L477, full field
initialization, not destructure). No `..` consumers. No audit
required for this PR.

## Pre-existing bug reproducer (`int32 -> int64` cast)

`feedback_verify_bug_before_fixing.md` requires reproducing the bug
before fixing it. PR #72's diagnosis already documented the bug
(`cast(int32 -> int64)` reads f32 bit patterns); the existing
`crates/chelis-cli/tests/cbackend_print_tensor_f64.rs::cbackend_print_tensor_int64`
fixture pins the print side of the same bug class. PR 1 does NOT
fix the int32-as-f32 storage encoding (that is the scope of the
recommended new §5 entry above); it pins the trait surface that
makes the eventual fix mechanically expressible.

Specifically the PR 1 fixture
`crates/chelis-e2e/tests/dtype_op_matrix.rs::tensor_to_f64_i32`
documents the f32-encoded routing: the fixture writes `42.0f32`
through `f32::fill` (matching the storage convention) and asserts
the read returns `42.0`. A change that broke the f32-encoded
routing would fail this fixture immediately.

## Sibling sweep summary (post-fix-commit)

Per `feedback_sibling_sweep_target.md`, the fix commit grep's for
`*mut f32`, `as *mut f32`, `as *const f32`:

* `crates/chelis-runtime/`: the 33-site enumeration above. PR 1
  touches only the anchor sites listed; the remaining sites compile
  via the transition shim helper (`data_as_f32`). Discrepancies
  with the pre-PR enumeration land in the PR body.
* `crates/chelis-backend-c/`: the 6 host_emit sites + the print
  routine (already migrated). PR 3's scope.
* `crates/chelis-backend-hip/`: zero matches.
* `crates/chelis-backend-metal/`: zero matches.

No new emergent coupling found; the plan's scope estimate stands.

## References

* Plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`,
  W2 PR 1 section.
* Spec lock: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`
  Contract 2 (trait surface) and Contract 3 (op enumeration).
* §5 entry: `docs/archive/reports/gap_synthesis.md` `CRuntime-F32Coupling`.
* PR #79 orchestrator decisions: Q2 (bool routes through
  `f32::data_ptr`), Q3 (`CHELIS_*` constants `pub const`).
* Prior bug-class diagnoses: `docs/investigations/cbackend_cast_memcpy_diagnosis.md`
  (PR #64), `docs/investigations/cbackend_reshape_memcpy_diagnosis.md`
  (PR #67), `docs/investigations/cbackend_print_tensor_f64_diagnosis.md`
  (PR #72).
* Failing test harness: `crates/chelis-e2e/tests/dtype_op_matrix.rs`.
