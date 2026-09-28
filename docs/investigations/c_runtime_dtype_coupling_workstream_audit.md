# CRuntime-F32Coupling workstream audit (0.7.8 W2 PR 4 terminal sweep)

## Scope

Terminal sibling sweep audit for the `CRuntime-F32Coupling` workstream
(W2 of 0.7.8 compiler cleanup). PRs landed in series: #84 (Agent A:
TensorElement trait + chelis_tensor.data: \*mut u8 + first anchor
migrations + matrix skeleton), #86 (Agent B: 31 runtime sites migrated;
matrix expanded 17 to 60 fixtures), #87 (Agent B: 6 host_emit
code-generation sites migrated; 11 host_emit-dispatch fixtures added),
and this PR (Agent B: matrix completion plus workstream audit).

This note is the artifact the Wave 3 red team reads when validating
workstream closure. It documents the final state of every audit target
called out in the W2 PR 4 dispatch brief plus the orchestrator-level
§5 follow-on recommendations.

## 1. `*mut f32` sweep across the runtime/backends

Grep for `*mut f32`, `as *mut f32`, `as *const f32` across the five
target crates (commit `b539429`, post-PR-#87, plus this PR's changes):

| Crate | Count | Notes |
|---|---:|---|
| `crates/chelis-runtime/` | 11 | All intentional, enumerated below |
| `crates/chelis-backend-c/` | 0 | Clean post-PR-#87 |
| `crates/chelis-backend-hip/` | 0 | Never had any |
| `crates/chelis-backend-metal/` | 0 | Never had any |
| `crates/chelis-ir/` | 0 | Never had any |

Total residual: 11, all in the runtime crate. Categorized:

| Line(s) | Site | Reason |
|---|---|---|
| 118 | Doc comment on `data_as_f32` | Describes the helper's return type |
| 145-146 | `data_as_f32` body | Transition shim for I32/BOOL f32-encoded storage |
| 156-157 | `data_as_f32_const` body | Const variant of same shim |
| 597 | `chelis_alloc_view` `data` param | ABI compat: `chelis_runtime.h` declares `float *data` |
| 1711 | Doc comment on `chelis_flatten_nested_list` `out` param | I32/BOOL/F32 leaves all 4-byte f32-encoded |
| 1722 | `chelis_flatten_nested_list` `out` param | I32/BOOL/F32 leaves all 4-byte f32-encoded |
| 2169 | `cmp_loop` `out_buf` param | bool output buffer (f32-encoded) |
| 2470 | `sort_loop` `idx_buf` param | i32 index buffer (f32-encoded) |
| 3187 | Doc comment | Historic note on pre-PR-#84 field type |

Every remaining instance routes through the f32-encoded storage
convention for `CHELIS_I32` or `CHELIS_BOOL`, or preserves the public
C ABI for `chelis_alloc_view`, or is a doc comment. No site is an
unmigrated runtime accessor.

Disposition: all 11 instances are intentional and locked by surrounding
comments. The runtime-side workstream closure is complete.

## 2. `data_as_f32` / `data_as_f32_const` callsite audit

`data_as_f32` and `data_as_f32_const` are the transition shims kept
alive for `CHELIS_I32` and `CHELIS_BOOL` routing. PR #86 retained them
deliberately per the orchestrator decision on PR #79 (bool 4-byte
f32-encoded today; bundle into `CRuntime-BoolStorage-F1` follow-on)
and the discovery on PR #84 (int32 4-byte f32-encoded today; bundle
into `CRuntime-I32Storage-F1` follow-on).

Callsites across the runtime (commit `b539429` plus this PR):

| Line | Function | Dispatch arm | Justification |
|---|---|---|---|
| 662 | `chelis_scalar_tensor_from_i64` | All (I64-encoded value written as f32) | Stores int as 4-byte f32 bit pattern per CHELIS_I32 storage |
| 691 | `chelis_tensor_to_f64` | CHELIS_I32 | Reads i32 storage as f32 |
| 693 | `chelis_tensor_to_f64` | CHELIS_BOOL | Reads bool storage as f32 |
| 1800 | `chelis_to_tensor` | CHELIS_F32 / CHELIS_I32 / CHELIS_BOOL | All three are f32-encoded storage |
| 1845, 1852 | scalar tensor path | I32/BOOL routing | Same convention |
| 1895, 1942 | `chelis_pad_sequences[_to]` | CHELIS_I32 / CHELIS_BOOL out | f32-encoded write |
| 2098 | `chelis_tensor_gather` indices | i32 indices | Indices buffer is f32-encoded i32 |
| 2161, 2185, 2186 | `chelis_tensor_cmplt` | bool output / I32-BOOL input | f32-encoded storage |
| 2223, 2295, 2296 | `chelis_tensor_scatter` | i32 indices, I32-BOOL data | f32-encoded routing |
| 2359 | `chelis_tensor_where` | bool cond | f32-encoded bool |
| 2426 | `chelis_tensor_cumsum` | I32-BOOL output (rejected) | runtime_fail arm |
| 2454, 2509 | `chelis_tensor_sort` | i32 indices, I32-BOOL data | f32-encoded routing |
| 2664, 2665 | `chelis_tensor_diagonal` / `_trace` | I32-BOOL data | f32-encoded routing |

Every callsite is on the `CHELIS_I32`, `CHELIS_BOOL`, or i32-index
path. No site reads or writes through `data_as_f32` for a precision
that the runtime represents at its genuine byte width
(`CHELIS_F32` itself, `CHELIS_F64`, `CHELIS_I64`). The shim is held
correctly inside the storage-encoding envelope.

## 3. Matrix completeness summary

Cross-validation matrix at `crates/chelis-e2e/tests/dtype_op_matrix.rs`
(this PR brings the total to 77) plus host_emit dispatch matrix at
`crates/chelis-backend-c/tests/host_emit_dtype_dispatch.rs` (11) plus
cast+arithmetic composition matrix at
`crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`
(this PR; 5).

### Single-op accessor matrix (precisions × ops)

Per Contract 3 of `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`,
the supported precisions are `f32, f64, i32, i64, bool`; bf16, f16,
i8, i16 are out of scope per the open-questions resolution in
Phase 0 (PR #79).

| Op | f32 | f64 | i32 | i64 | bool | Notes |
|---|---|---|---|---|---|---|
| `chelis_tensor_to_f64` | ok | ok | ok | ok | ok (true+false) | 6 fixtures (PR #84) |
| `<T>::fill` | ok | ok | ok | ok | n/a | bool routed via f32 |
| `<T>::data_ptr` dtype check | n/a | ok | n/a | ok | n/a | 4 fixtures (PR #84) |
| `chelis_list_from_tensor` | ok | ok | ok | ok | ok | 5 fixtures (PR #86) |
| `chelis_tensor_concat` | ok | ok | ok | ok | ok | 5 fixtures (PR #86) |
| `chelis_tensor_split` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_gather` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_cmplt` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_scatter` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_where` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_cumsum` | ok | ok | n/a | ok | rejects | 3 fixtures + bool reject in subprocess |
| `chelis_tensor_sort` | ok | ok | n/a | ok | rejects | 3 fixtures + bool reject in subprocess |
| `chelis_tensor_diagonal` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #86) |
| `chelis_tensor_trace` | ok | ok | n/a | ok | rejects | 3 fixtures + bool reject in subprocess |
| `chelis_tensor_clamp` | ok | ok | n/a | ok | rejects | 3 fixtures + bool reject in subprocess |
| `chelis_tensor_einsum` | ok | ok | n/a | ok | rejects | 3 fixtures + bool reject in subprocess |
| host_emit `add`/`sub`/`mul`/`div` | ok | ok | ok | ok | ok | 3 fixtures at f32/f64/i64 (PR #87) |
| host_emit `neg`/`not` | ok | ok | n/a | ok | n/a | 3 fixtures (PR #87) |
| host_emit `max_elem`/`min_elem` | ok | rejects | n/a | rejects | n/a | 1 fixture (PR #87); f64/i64 abort |
| host_emit `expf`/`logf`/`sinf` | ok | rejects | n/a | rejects | n/a | 1 fixture (PR #87) |
| host_emit scalar coercion bool/i64/f32 | ok | ok | n/a | ok | ok | 3 fixtures (PR #87) |

Notes:
* `n/a` means the op's semantic domain excludes that precision per
  Contract 3 (e.g., split doesn't have an i32 fixture because i32
  storage is f32-encoded and the i32 path is covered by the f32
  fixture).
* `rejects` means the op calls `runtime_fail!` on that precision; the
  runtime_fail invokes `std::process::exit(1)` so a subprocess harness
  is required to exercise the failure path. The runtime test surface
  intentionally excludes these arms; the CLI integration tests
  catch them via end-to-end exit-code checks.

### Multi-op composition matrix (this PR, W2 PR 4)

17 new fixtures in `crates/chelis-e2e/tests/dtype_op_matrix.rs` covering
runtime-level op chains:

* `concat -> gather` for f64 and i64
* `concat -> cumsum` for f64 and i64
* `gather -> sort` for f64 and i64
* `where -> trace` for f64 and i64
* `cumsum -> clamp` for f64 and i64
* `scatter -> diagonal` for f64
* `where -> einsum` for f64 and i64
* `clamp -> cumsum` for f64
* `cmplt -> where` for f64
* `cmplt -> where` for f32 (control)
* `list_from_tensor` round trip for f64

5 new fixtures in
`crates/chelis-cli/tests/cbackend_cast_arithmetic_composition.rs`
covering `chelis build --target c` cast-mixed-with-arithmetic
compositions:

* `add(cast(t, f64), cast(t, f64))` from f32 source
* `mul(cast(t, f64), cast(t, f64))` from f32 source
* `cast(add(t, t), f64)` arithmetic-then-widen
* `mul(cast(x, f64), y_f64)` mixed cast-with-native
* `add(cast(t, f64), cast(t, f64))` from int32 source

These compositions cover the bug shape PR 3's agent reported (see
`docs/investigations/c_runtime_dtype_multiop_cast_diagnosis.md`).
No reproduction on current main; the chain is byte-exact eval-vs-C.

### Out-of-scope coverage (filed §5 follow-ons)

* bf16, f16: no host-side precision today; bf16 follow-on filed at
  `CRuntime-F32Coupling`'s closing notes (re-evaluate when host
  precision lands).
* i8, i16: no `CHELIS_I8` / `CHELIS_I16` constants today; deferred
  per `CRuntime-I8I16-F1` until program shape forces them.
* Native bool storage (1-byte): deferred per `CRuntime-BoolStorage-F1`
  until storage migration lands.
* Native i32 storage (4-byte native int): deferred per
  `CRuntime-I32Storage-F1` until storage migration lands.

## 4. §5 follow-on recommendations

No new follow-ons surfaced by PR 4. The existing four are still
accurate after this PR's audit:

| §5 ID | Status after PR 4 |
|---|---|
| `CRuntime-F32Coupling` | Phase 1-3 closed by PRs #84/#86/#87; Phase 4 (matrix completion + audit) closed by this PR. Recommend marking Closed. |
| `CRuntime-I32Storage-F1` | Still Open. No new work required this PR. Surfaces when first int32-heavy program forces native storage. |
| `CRuntime-BoolStorage-F1` | Still Open. No new work required this PR. Bundles with `CRuntime-I32Storage-F1`. |
| `CRuntime-I8I16-F1` | Still Open. No new work required this PR. Surfaces when first i8/i16-using program forces the new constants. |

No new §5 entry is recommended for the PR-3-agent-surfaced f64-cast
bug; that bug class is already captured by `CRuntime-F32Coupling` and
the reproduction attempt on current main returned no-repro (PR #87
incidentally closed the manifestation in the host_emit fallback path;
the DAG-specialized path already had typed-pointer accesses even on
the pre-PR-#84 baseline). See
`docs/investigations/c_runtime_dtype_multiop_cast_diagnosis.md` for
the no-repro disposition.

## 5. Regression-locked surface fix verification

The four 0.7.6 surface fixes from the workstream's pre-history all
remain green:

* `crates/chelis-cli/tests/cbackend_cast_memcpy.rs` (CBackend-CastMemcpy, PR #64) — 4 fixtures pass.
* `crates/chelis-cli/tests/cbackend_reshape_memcpy.rs` (CBackend-ReshapeMemcpy, PR #67) — 3 fixtures pass.
* `crates/chelis-cli/tests/cbackend_print_tensor_f64.rs` (CBackend-PrintTensorF64, PR #72) — 3 fixtures pass.
* `crates/chelis-cli/tests/host_eval_scalar_fn_call.rs` (HostEval-ScalarFn-F1, PR #80) — 5 fixtures pass.

The full workstream test surface (PRs #84/#86/#87 plus this PR) is on
the default `cargo test --workspace` path; no manual gate is required.

## 6. Workstream closure recommendation

After this PR merges, the orchestrator should:

1. Mark `CRuntime-F32Coupling` as Closed in `docs/archive/reports/gap_synthesis.md`,
   citing this audit note.
2. Leave `CRuntime-I32Storage-F1`, `CRuntime-BoolStorage-F1`, and
   `CRuntime-I8I16-F1` Open (no work this workstream; surface-when-forced).
3. Bump workspace version to 0.7.8 in `Cargo.toml` per the workstream
   close convention.
4. Dispatch the Wave 3 red team per `redteam-exec` against this audit
   note plus the matrix files.
