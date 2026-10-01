# 0.7.8 Compiler Cleanup — Phase 0 Spec Lock

> Historical workstream contract for 0.7.8. Current linearity semantics belong
> to [`spec/04-type-system.md`](../../04-type-system.md); current implementation
> design belongs to [`implicit_linearity.md`](../implicit_linearity.md).

Status: Phase 0 deliverable. Pins three contracts so Wave 1 agents (W1, W2 PR 1,
W3) work against stable targets. Not a canonical spec — this is a
workstream-scoped design contract under `docs/design/`. The owning plan is
`/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`.

Closes the diagnosis ambiguity for five `docs/archive/reports/gap_synthesis.md` §5 entries:

- `Linearity-F1` (typed `ConsumeKind`)
- `Linearity-F2` (tuple-destructure linearity false negative)
- `Linearity-AliasedConsume-F1` (var-RHS aliasing tracks name not value)
- `CRuntime-F32Coupling` (C runtime data pointer is `*mut f32` regardless of dtype)
- `HostEval-ScalarFn-F1` (out of scope for this note — W3 lands surgically; no
  shared structural contract with W1/W2)

The first three feed Contract 1 (ConsumeKind shape). The fourth feeds Contract 2
(`TensorElement` accessor trait) and Contract 3 (dtype × op cross-validation
matrix). `HostEval-ScalarFn-F1` is a single-PR surgical fix in W3 and does not
require a Phase 0 contract.

## Pre-locked decisions (from the plan)

These are fixed inputs to this design note, not under review here:

1. Base branch is `main` (treated as the 0.7.8 line). Confirmed at `cc16619`
   (`release: 0.7.7 (#75)`) plus `903598d` (`Update LOC report`).
2. `Linearity-F1`/`F2`/`AliasedConsume-F1` bundle into one W1 PR with a
   warnings-first cascade for F2.
3. Tensor data accessor: Option A (`*mut u8` storage with dtype-tagged decode at
   each access site) plus the `TensorElement` trait pattern. This note specifies
   the trait surface; it does not re-evaluate trait-vs-generic-fn-vs-macro.

---

## Contract 1 — Typed `ConsumeKind`

### Shape

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConsumeKind {
    /// `let alias = x` and similar var-RHS bindings.  At the IR level
    /// `lower_let` maps `alias` to the same NodeId as `x` (the `Load`
    /// node); the value is structurally shared, not destroyed.  Later
    /// borrow-reads of `x` must succeed.  See
    /// `spec/design/implicit_linearity.md` "Copy Insertion" + "Borrows do
    /// not count as fan-out".
    Aliasing,
    /// Every other consume site: realize/drop/store, app-arg, pipe-stage,
    /// closure capture, match scrutinee, etc.  The value is gone after
    /// this point and later reads are `UseAfterConsume`.
    Structural,
}

#[derive(Debug, Clone)]
struct ConsumeSite {
    description: String,
    kind: ConsumeKind,
}
```

The `description` string stays for diagnostic rendering. `kind` carries the
discrimination axis that `read_or_error` (today at L703-728) currently extracts
by string-prefixing `"binding "` on the description.

### Why two variants is enough — including tuple-destructure

The plan asked Phase 0 to verify whether tuple-destructure consumes need a third
variant (`Destructure`) by tracing what kind of consume the `__chelis_tmp_N`
bindings produce.

`crates/chelis-surf/src/desugar.rs:1135-1156` rewrites
`let (a, b) = pair; <body>` into nested `let` chains:

```
let __chelis_tmp0 = pair in
  let __chelis_tmp1 = tuple_index(__chelis_tmp0, 0) in
    let a = __chelis_tmp1 in
      let __chelis_tmp2 = tuple_index(__chelis_tmp0, 1) in
        let b = __chelis_tmp2 in
          <body>
```

Today, `__chelis_tmp_N` bindings carry no type metadata (`destructure_pattern`
does not call `inject_type_metadata`, unlike the Var-pattern path at L1166-1171).
`expr_is_owned_linear` at L811-814 returns `false` for `(var __chelis_tmp_N)`
references because the inferred type is unknown, so the linearity check skips
them silently (`Linearity-F2`).

After W1 PR 1 threads type metadata onto the destructured components, the
references like `realize(a)` and `realize(__chelis_tmp_N)` flow through the
same paths that already exist for typed bindings:

- `consume_var_expr(generic_site)` at L356 for bare var uses,
- `app_site` at L1200 for app-arg consumes,
- `realize_site` at L1217 for `realize` calls,
- `pipe_site` at L1223 for pipe-stage consumes,
- `match scrutinee at offset N` at L575,
- `closure capture at offset N` at L519.

None of these introduce a new discrimination axis. A destructured tuple
component, once typed, is a `Structural` consume in the same sense as any
realize/app/pipe of a regular owned tensor. The only piece that needs care is
the `__chelis_tmp_N` -> destructured-component-name bind itself: that is a
`let alias = <value>` shape and is an `Aliasing` consume by Contract 1's
definition.

Conclusion: **two variants**. Reject `Destructure` as a third variant.

### Producer-site assignment table

W1 PR 1 must migrate every producer of `ConsumeSite` to set `kind` explicitly.
The eight producer call-sites in `crates/chelis-types/src/linearity.rs` map to:

| Producer site | Line | Kind |
|---|---|---|
| Var-RHS in `check_def_body` (`is_var_expr(body) && expr_is_owned_linear`) | L330-336 | `Aliasing` |
| Var-RHS in `check_let` (`is_var_expr(value) && expr_is_owned_linear`) | L482-488 | `Aliasing` |
| Closure capture in `check_fn` | L519-522 | `Structural` |
| Match scrutinee in `check_match` | L575-577 | `Structural` |
| `app_site` (function/app arg consume) | L1200-1209 | `Structural` |
| `generic_site` (bare var consume in `check_expr`) | L1211-1215 | `Structural` |
| `realize_site` | L1217-1221 | `Structural` |
| `pipe_site` (pipe-stage consume) | L1223-1231 | `Structural` |

After migration the `descriptor.starts_with("binding ")` check at L726 becomes
`matches!(site.kind, ConsumeKind::Aliasing)`. The two `"binding `{name}` at
offset N"` descriptions stay as the rendered text for `UseAfterConsume`
diagnostics but no longer carry semantic load.

### Lineage forwarding for `AliasedConsume-F1`

The typed-kind refactor alone does not close `Linearity-AliasedConsume-F1`. The
remaining gap: `LinearScope.bindings` (L39-42) is keyed only by name, so
`let y = x; realize(y); add(x, y)` consumes `y`'s entry but leaves `x` live.

Contract 1 leaves the lineage-forwarding mechanism to W1 PR 1's diagnosis (the
plan's W1.2 step). The two viable shapes are:

- Add a `BindingState::Alias { source: String }` variant alongside `Live` and
  `Consumed`. On `consume(y, ...)` walk the alias chain to the underlying source
  and consume that. Aliasing depth is bounded by source-program nesting; one
  level handles the typical `let y = x` case, multi-level handles `let z = y`.
- Or add a `LinearScope.aliases: HashMap<String, String>` map. Simpler data
  shape; `resolve_alias_chain(name)` walks until a non-alias name is found.

Either is compatible with Contract 1. W1 PR 1 picks one and pins it in
`docs/archive/investigations/linearity_typed_consumekind_diagnosis.md`.

---

## Contract 2 — `TensorElement` trait surface

### Trait definition

```rust
use libc::c_int;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeMismatch {
    pub expected: c_int,
    pub actual: c_int,
}

pub unsafe trait TensorElement: Sized + Copy {
    /// The `CHELIS_*` dtype tag for this Rust primitive.
    const DTYPE: c_int;

    /// Checked typed access.  Returns `Err` if the tensor's dtype does
    /// not match `Self::DTYPE`.  Every Phase 0a/0b accessor migration
    /// uses this path; the dtype compare is one `c_int` compare per
    /// call.
    ///
    /// # Safety
    ///
    /// `tensor` must point to a live `chelis_tensor` and remain valid
    /// for the lifetime of the returned pointer.  The returned pointer
    /// must not outlive the tensor.
    #[inline]
    unsafe fn data_ptr(tensor: *mut chelis_tensor) -> Result<*mut Self, DtypeMismatch> {
        let actual = (*tensor).dtype;
        if actual != Self::DTYPE {
            return Err(DtypeMismatch { expected: Self::DTYPE, actual });
        }
        Ok((*tensor).data as *mut Self)
    }

    /// Unchecked access for hot loops where the caller has already
    /// verified the dtype (e.g. an outer match arm that dispatched on
    /// `tensor.dtype`).  The check is the caller's contract.
    ///
    /// # Safety
    ///
    /// As `data_ptr`, plus: caller asserts `(*tensor).dtype ==
    /// Self::DTYPE`.  Passing a tensor whose dtype is something else
    /// is undefined behavior.
    #[inline]
    unsafe fn data_ptr_unchecked(tensor: *mut chelis_tensor) -> *mut Self {
        debug_assert_eq!((*tensor).dtype, Self::DTYPE);
        (*tensor).data as *mut Self
    }

    /// Element-wise fill.  Default implementation lifts the
    /// cast-then-loop pattern from `chelis_fill_i64` (L489) into the
    /// trait so every supported precision gets a fill helper for free.
    ///
    /// # Safety
    ///
    /// As `data_ptr`.
    #[inline]
    unsafe fn fill(tensor: *mut chelis_tensor, value: Self) {
        let ptr = Self::data_ptr_unchecked(tensor);
        for i in 0..(*tensor).size as isize {
            *ptr.offset(i) = value;
        }
    }
}
```

### Decisions pinned

- **Checked-by-default + opt-in unchecked.** `data_ptr` returns `Result` for
  callers that haven't yet verified the dtype. `data_ptr_unchecked` exists for
  the inner-loop case after an outer dispatch. Cost of the check is one
  `c_int` comparison; the safety-net phase admits that cost.
- **`DtypeMismatch` is `Copy + Eq`** with `expected` and `actual` `c_int`
  fields. No allocation, no string formatting at the failure site; sites that
  need a human-readable error format on demand.
- **Default `fill` is included.** It folds the existing `chelis_fill_f32`,
  `chelis_fill_f64`, and `chelis_fill_i64` bodies into one default. The three
  `chelis_fill_*` extern symbols stay because the C runtime header exports
  them; their bodies thin to `T::fill(t, val)`.
- **Trait is `unsafe`.** Implementing it asserts that `DTYPE` matches the
  byte layout that `chelis_alloc` uses for that constant. Misimplementing
  triggers silent data corruption — exactly the bug class this contract is
  closing — so the trait is unsafe and implementers must justify the dtype
  pairing.
- **Trait does not own `chelis_alloc` dispatch.** Allocation already takes a
  `dtype: c_int` argument and dispatches on `dtype == CHELIS_I64 || dtype ==
  CHELIS_F64`. Centralizing the element-size table inside `TensorElement` is
  in scope for a follow-on (W2 PR 4 or beyond); Phase 0 pins only the access
  surface.

### Impl blocks for supported precisions

Five impls land in W2 PR 1, one per dtype that exists in the runtime today:

```rust
unsafe impl TensorElement for f32  { const DTYPE: c_int = CHELIS_F32; }
unsafe impl TensorElement for f64  { const DTYPE: c_int = CHELIS_F64; }
unsafe impl TensorElement for i32  { const DTYPE: c_int = CHELIS_I32; }
unsafe impl TensorElement for i64  { const DTYPE: c_int = CHELIS_I64; }
unsafe impl TensorElement for bool { const DTYPE: c_int = CHELIS_BOOL; }
```

The `CHELIS_*` constants live at `crates/chelis-runtime/src/lib.rs:15-19`:
`CHELIS_F32 = 0`, `CHELIS_F64 = 1`, `CHELIS_I32 = 2`, `CHELIS_BOOL = 3`,
`CHELIS_I64 = 4`. These constants are currently `const` (not `pub`); W2 PR 1
must promote them to `pub const` so the trait impls can name them, or move the
trait to the same module that defines them. Recommended: promote to `pub const`
and keep the trait collocated under `chelis-runtime`.

### Discovery finding — i8 and i16 are not in the runtime today

The plan brief asked Phase 0 to specify trait impls for `f32, f64, i8, i16,
i32, i64, bool`. Grepping the runtime sources:

- `CHELIS_I8` does not exist as a constant.
- `CHELIS_I16` does not exist as a constant.
- `chelis_alloc` (L388-434) dispatches only between i64/f64 (8-byte) and
  everything else (4-byte). i8 and i16 would need new element-size arms.
- No call site, lowering pass, or test references i8 or i16 storage.

`bool` storage today writes `1.0f32` / `0.0f32` into the f32 buffer (see
`chelis_tensor_cmplt` at L1897-1901 and `chelis_pad_sequences` at
L1660-1664). The bool impl above is correct per the existing dtype tag but
sits on top of 4-byte f32-encoded storage, not 1-byte bool storage.
Re-encoding bool to 1-byte storage is a separate change and is **out of
scope** for Phase 0; the trait surface is forward-compatible with that
change because callers go through `TensorElement::data_ptr` rather than
casting the raw pointer manually.

**Open question for orchestrator decision** — see §"Open questions" below.

### Validation against `chelis_fill_i64` (`lib.rs:489`)

Current body:

```rust
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_i64(t: *mut chelis_tensor, val: i64) {
    let ptr = (*t).data as *mut i64;
    for i in 0..(*t).size as isize {
        *ptr.offset(i) = val;
    }
}
```

After migration:

```rust
#[no_mangle]
pub unsafe extern "C" fn chelis_fill_i64(t: *mut chelis_tensor, val: i64) {
    i64::fill(t, val);
}
```

The trait's default `fill` expands to the same cast + loop after inlining. The
`debug_assert_eq!` in `data_ptr_unchecked` is a no-op in release builds. No
runtime regression; the migration is bit-identical at the access site, with
the bonus that mismatched-dtype calls now trip `debug_assert` instead of
silently writing through a stale-typed pointer.

The same pattern lifts `chelis_fill_f32` and `chelis_fill_f64` to
`f32::fill(t, val)` and `f64::fill(t, val)` respectively.

### Anchor op for W2 PR 1 — `chelis_tensor_to_f64`

Recommended: `chelis_tensor_to_f64` at `lib.rs:519`. Read-side, simple,
exercised through `eval`-vs-`build c` parity tests. Current body:

```rust
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    *(*t).data as f64
}
```

The `*(*t).data as f64` reads the first f32 element of the buffer and widens.
If the tensor's dtype is f64 or i64, the read drops the upper 4 bytes.

After migration:

```rust
#[no_mangle]
pub unsafe extern "C" fn chelis_tensor_to_f64(t: *const chelis_tensor) -> f64 {
    if t.is_null() || (*t).ndim != 0 {
        runtime_fail!("chelis_tensor_to_f64 expects a rank-0 tensor");
    }
    let t = t as *mut chelis_tensor;
    match (*t).dtype {
        CHELIS_F32  => *f32::data_ptr_unchecked(t)  as f64,
        CHELIS_F64  => *f64::data_ptr_unchecked(t),
        CHELIS_I32  => *i32::data_ptr_unchecked(t)  as f64,
        CHELIS_I64  => *i64::data_ptr_unchecked(t)  as f64,
        CHELIS_BOOL => if *bool::data_ptr_unchecked(t) { 1.0 } else { 0.0 },
        other => runtime_fail!("chelis_tensor_to_f64 unsupported dtype {other}"),
    }
}
```

The outer match on `(*t).dtype` is the dispatch; each arm calls
`data_ptr_unchecked` because the dispatch is the verification. This is the
**migration template** Agent A and Agent B both follow for every site listed
in Contract 3.

For accessor sites that already sit inside a loop iterating over `(*t).size`,
the dispatch lifts above the loop:

```rust
match (*t).dtype {
    CHELIS_F32 => {
        let p = f32::data_ptr_unchecked(t);
        for i in 0..(*t).size as isize {
            // typed work on f32
        }
    }
    // other arms
}
```

Each arm holds an `&mut [T]` style typed slice (via raw pointer + size) for
the entire inner loop. Per-element dtype dispatch is forbidden by this
template.

---

## Contract 3 — dtype × op cross-validation matrix

### Matrix scope

Rows: every supported precision in the runtime today: `f32, f64, i32, i64,
bool`. (`i8`, `i16`, `bf16`, `f16` are out of scope — see open questions.)

Columns: every op that goes through tensor data accessors. Enumerated from
`crates/chelis-runtime/src/lib.rs`:

| Op | Anchor site(s) | Semantic precision domain |
|---|---|---|
| `chelis_fill_f32` / `f64` / `i64` | L482, L489, L497 | Each is precision-specific by construction |
| `chelis_scalar_tensor_from_i64` / `_f64` | L505, L512 | int + float scalar producers |
| `chelis_tensor_to_f64` (read-out) | L519 | Any numeric or bool (widens to f64) |
| `pad_sequences` / `_to` (numeric fill) | L1659-L1679, L1703-L1726 | i32, f32 (bool would need extension) |
| `list_from_tensor` (per-element read) | L1626 | Any numeric or bool (returns boxed value) |
| `concat` (memcpy-style copy) | L1772 | Any precision; output dtype matches input |
| `split` (slice + copy) | L1815 | Any precision; output dtype matches input |
| `gather` (indexed read of f32) | L1869, L1884 | Any numeric or bool data; index is i32 |
| `cmplt` (numeric compare → bool) | L1897 | Any numeric input → bool output |
| `scatter` (indexed write of f32) | L1952, L1970, L1972 | Any numeric or bool data; index is i32 |
| `where` (cond-typed select) | L1992-L1995 | Any precision data; today cond is read as f32 |
| `cumsum` (numeric prefix sum) | L2022-L2023 | Any numeric; bool is undefined |
| `sort` (numeric compare + i32 indices) | L2051-L2066 | Any numeric; bool is undefined |
| `diagonal` (indexed read) | L2131 | Any precision |
| `trace` (diagonal sum) | L2167-L2169 | Any numeric |
| `clamp` (numeric min/max) | L2190-L2199 | Any numeric; bool is undefined |
| `einsum` (numeric multiply-add) | L2314-L2325 | Any numeric; bool is undefined |
| `tensor_to_string` | L2582 | Any numeric or bool |
| `chelis_print_tensor_stdout` (in `host_emit.rs`) | L207-256 | Already dtype-dispatched (PR #72) |

Host-emit elementwise helpers (`host_emit.rs` L1563, L1599, L1623, L1649, L1728,
L1732) emit C code strings; their migration changes the generated C to use
dtype-typed accessor templates rather than `t->data[i]` raw indexing. These
are W2 PR 3's scope.

### Anchor ops for W2 PR 1's harness skeleton

Per the plan, W2 PR 1 covers one or two anchor ops with full every-precision
coverage; the remainder are W2 PR 2-4 as Agent B's mechanical migration.

Recommended anchors:

1. **`chelis_tensor_to_f64`** (L519, read-side). Single function, scalar
   output, exercises the dispatch pattern in its simplest form. The
   cross-validation test runs each of `f32 / f64 / i32 / i64 / bool` through
   the host evaluator (`eval_tensor`) and the C runtime path, asserting
   byte-exact agreement.
2. **`chelis_fill_i64`** (L489, write-side). Already takes an `i64` and is
   the lift candidate referenced in the plan. The migration validates that
   the trait's `fill` default replaces the existing body bit-identically.
   Cross-validation covers `f32::fill`, `f64::fill`, `i32::fill`, `i64::fill`,
   `bool::fill` against `chelis_alloc` of each dtype.

Negative coverage: Each anchor op gets one fixture that calls
`<T>::data_ptr(t)` on a tensor whose `dtype` is something other than `T`,
asserting `Err(DtypeMismatch { expected, actual })`.

### Harness location

Recommended: a new file `crates/chelis-e2e/tests/dtype_op_matrix.rs`.

Rationale: `eval_agreement.rs` is already large and centered on the eval-vs-C
compile/run pattern; the dtype matrix has its own row × column scaffolding and
its own "every precision through this single op" loop structure. A new file
keeps the dtype matrix discoverable as a single artifact and avoids
intermingling test concerns.

The harness must reuse the runtime-library lookup helpers from
`eval_agreement.rs` (`runtime_src_dir`, `runtime_library_path`); extracting
them to a shared helper module is permitted but not required for W2 PR 1.

### Default-gate budget

The plan's verification section requires the matrix to stay under the
~60-second `cargo test --workspace` budget or be `#[ignore]`d behind a
documented manual gate. W2 PR 1 measures the anchor-op runtime; if every-op
expansion in W2 PR 4 pushes past the budget, the matrix gates `#[ignore]`
and the manual command lands in `docs/manual_gates.md`. Decision deferred to
W2 PR 4 measurement; Phase 0 does not pre-commit to ignored.

---

## Out of scope

Recorded so Wave 1 agents do not redo this thinking:

- **`HostEval-ScalarFn-F1`** — W3 is a single-PR surgical fix. No shared
  structural contract with W1/W2.
- **`bf16` / `f16` / quantization dtypes** — no host-side precision support
  today. New impl blocks land when host precision lands.
- **`i8` / `i16` storage** — no `CHELIS_I8` / `CHELIS_I16` constants today; no
  `chelis_alloc` element-size arm; no call site. Open question to orchestrator
  on whether to include in the trait surface as forward-compatible no-op
  impls. Default recommendation: defer, file a §5 follow-on for first user
  shape that forces them.
- **bool storage layout** — bool is currently 4-byte f32-encoded. Re-encoding
  to 1-byte storage is its own change. The trait surface is
  forward-compatible: callers go through `TensorElement::data_ptr`.
- **Generic-fn vs macro vs trait** — pre-locked as trait (decision 4 of the
  plan).

## Open questions for orchestrator decision

1. **`i8` and `i16` impls.** The plan brief listed them; the runtime has no
   `CHELIS_I8` / `CHELIS_I16` constants or `chelis_alloc` arms. Three options:
   - (A) Defer: ship only `f32 / f64 / i32 / i64 / bool` in W2 PR 1; file a
     §5 follow-on for the first program shape that forces i8/i16.
     **Recommendation.**
   - (B) Skeleton-impl: add `CHELIS_I8 = 5`, `CHELIS_I16 = 6` constants and
     `chelis_alloc` arms now, plus the trait impls. No call site exercises
     them. Risk: untested code path lands.
   - (C) Treat as a separate workstream item, blocking W2 PR 4.

2. **`bool` 4-byte vs 1-byte storage.** The trait impl above says `Self = bool,
   DTYPE = CHELIS_BOOL`, but `chelis_alloc` for `CHELIS_BOOL` returns a 4-byte
   element buffer (per `chelis_alloc` at L418-422) and the runtime writes
   `1.0f32`/`0.0f32` into it. The `bool::data_ptr_unchecked` cast would
   reinterpret 4 bytes as a 1-byte `bool` — UB-adjacent at best.

   Two options:
   - (A) Treat `bool` storage as f32-encoded and route through `f32::data_ptr`
     internally. Trait does not provide a `bool` impl; bool sites stay
     dtype-dispatched on `CHELIS_BOOL` but read through the f32 typed pointer
     and compare against `0.0`. **Recommendation** — matches current runtime
     behavior, no UB.
   - (B) Re-encode bool storage to 1-byte. Out of scope for 0.7.8.

3. **`CHELIS_*` constant visibility.** Today they are `const` (not `pub`). The
   trait impl blocks need them visible. Recommended: promote to `pub const` in
   W2 PR 1; cross-crate code currently has no need to name them, so the
   visibility widening is forward-defensive only.

## Out-of-scope adjacent §5 entries

Recorded for awareness — not under Phase 0's authority:

- `IR-FirstClassFn-F1`, `IR-SelectOp-F1`, `IR-MatchLowering-F1`: architectural
  feature work; dispatch when customer shape forces.
- `Vocabulary-F1`, `Vocabulary-F2`, `SurfDecompile-PascalLowercase-F1`:
  drift-cost cleanups; dispatch when the third PR fires.

## References

- Plan: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`
- `docs/archive/reports/gap_synthesis.md` §5 — Linearity-F1/F2/AliasedConsume-F1,
  CRuntime-F32Coupling, HostEval-ScalarFn-F1
- `crates/chelis-types/src/linearity.rs` — `BindingState`, `ConsumeSite`,
  `LinearScope`, string-prefix discrimination at L726, eight producers at
  L330, L482, L519, L575, L1200, L1211, L1217, L1223
- `crates/chelis-surf/src/desugar.rs` — `destructure_pattern` (L1135-1156),
  `__chelis_tmp_N` synthesis (L1148-1153), `inject_type_metadata` (L281-294)
- `crates/chelis-runtime/src/lib.rs` — `chelis_tensor` (L29-38),
  `CHELIS_*` dtype constants (L15-19), `chelis_alloc` (L388-434),
  `chelis_fill_*` (L482, L489, L497), `chelis_tensor_to_f64` (L519),
  accessor sites enumerated in Contract 3
- `crates/chelis-backend-c/src/host_emit.rs` — already-fixed print routine
  (L207-256), elementwise helpers (L1563, L1599, L1623, L1649, L1728, L1732)
- `crates/chelis-e2e/tests/eval_agreement.rs` — existing parity harness
  pattern (runtime lookup helpers reused by the new matrix file)
- `spec/design/implicit_linearity.md` — Copy Insertion and "Borrows do not
  count as fan-out" semantics that justify `ConsumeKind::Aliasing`
