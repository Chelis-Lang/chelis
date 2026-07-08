# chelis#616 — Runtime-symbolic movement-op bounds: implementation handoff

**Status:** foundation landed on branch `agent/616-runtime-movement-bounds` (PR #627,
draft). This doc is the continuation plan for the next agent. The completion oracle has
NOT flipped yet; two capability pieces remain (§4, §5).

## 1. Goal and completion oracle

Make the runtime-symbolic-window `avgpool1d` gradient work end to end. The single
authoritative oracle is:

```
crates/chelis-cli/tests/issue_368_grad_concat_windows.rs
  ::issue_368_runtime_symbolic_window_grad_is_tracked_residual
```

Today it is an expected-to-fail pin (asserts `!success()` + a fail-closed diagnostic).
When #616 lands it must flip to a PASSING finite-difference + forward-parity +
eval-vs-`chelis build --target c` oracle whose gradient is `[0.5, 0.5, 0.5, 0.5]`
(for input `[1,2,3,4]`, window 2, stride 2). Reproducer source is in that test
(`window_row` = `reshape(stride(shrink(x, [[start, extent]]), 2), [1, m])`, with
`start`/`extent`/`m` all runtime `shape()`-derived).

**SOUNDNESS IS ABSOLUTE.** A loud failure must never be traded for a silently-wrong
gradient or a mis-sized allocation. Every runtime bound that eval rejects loudly, the C
backend must reject loudly too (error-path parity, not just value-path).

## 2. The core idea (one capability, applied in two places)

Everything reduces to: **a tensor dimension whose value is a runtime rank-0 integer
scalar node (a `Shape` read or integer arithmetic over one), declared/resolved in the
eval and C backend lanes.** Arithmetic lives as ordinary DAG scalar ops
(`Add`/`Mul`/`FloorDiv`/`Neg` — already integer-capable end to end); the dim just
*references the result node*. This capability is needed in two payload positions:

- **movement bounds** (`Shrink`/`Stride`/`Pad` start/end/step) — LANDED.
- **reshape targets** (`Reshape.new_shape`) — the window count `m` and the stride
  adjoint's `m_a*step` merge extent. NOT landed. This is the remaining gap.

The uniform representation is `dag::RtDim { Lit(usize), Sym(String), Node(usize), ToEnd }`
(currently `Lit`/`Node`/`ToEnd` only — see §4 step 2 for `Sym`). `Node(i)` is an ABSOLUTE
index into the owning op's `inputs`, where `inputs[0]` is the tensor and `inputs[1..]` are
rank-0 integer bound/dim scalars. `Node(i)` reads exactly like `inputs[i]` everywhere.

## 3. What is already DONE (green; commits on the branch)

- **`RtDim` representation** (`crates/chelis-ir/src/dag.rs`): the enum + `as_lit`/
  `is_runtime`/`node_input` helpers; `Pad`/`Shrink`/`Stride` payloads carry it; `ToEnd`
  promotes the `SHRINK_TO_END` sentinel. Renamed from `Bound` in commit `cb7d0514`.
- **Lowering** (`lower.rs`): `lower_pair_bounds` / `lower_stride_bounds` / `lower_one_bound`
  build node-valued movement bounds from runtime `cast(add(...))` args instead of
  `.unwrap_or_default()` to empty (the pad/shrink/stride arms near `lower.rs:7406`).
- **Eval** (`eval.rs`): `resolve_eval_bound` / `resolve_eval_pairs` / `resolve_eval_strides`
  read each `RtDim` to a concrete extent at eval time with non-negative/integral/range
  guards; the Pad/Shrink/Stride eval arms resolve then call the existing helpers. **Forward
  eval of a runtime shrink+stride and of the full `avgpool1d` reproducer is CORRECT** (the
  eval lane resolves runtime dims via `bind_symbolic_dims`; verified `[1.5,3.5]`).
- **Verify** (`verify.rs`): variable arity (`1 + Node count`), rank-0-int bound-source
  checks (`check_bound_source`), `ToEnd`-position rules; static extent checks apply only to
  `Lit` bounds.
- **Grad (movement, mechanical + stop-gradient)** (`grad.rs`): adjoints are `RtDim`-
  compatible; the value-dependent guards STAY (node-valued = the work below). Commit
  `05fb136b` made a movement op's bound-source inputs (`inputs[1..]`) a **stop-gradient
  boundary** in `grad_dag_checked`'s differentiability pre-pass (the liveness walk
  propagates only through `inputs[0]` for Shrink/Stride/Pad), so the window-count
  `floor_div` no longer wrongly rejects the whole gradient.
- **C backend** (`backend-c/src/emit.rs`): `emit_shrink`/`emit_stride`/`emit_pad` size
  node-valued axes at runtime (`bound_c_expr` reads a bound scalar `t{n}->data[0]`;
  `runtime_dim_decl_name` declares the axis's `_anon_dim`/named dim `int name = end-start`)
  with runtime `abort` guards. Soundness edit: `rename_anonymous_dims` no longer copies
  first-input dims for node-valued movement outputs (would clobber the shrunk axis).
- **HIP/Metal**: convert `RtDim`→`usize` for the literal launch emitters, panic on a
  node-valued bound (rejected upstream). **Reject seams** (`compiler-api/src/compiler.rs`
  `reject_unsupported_hip_ops`, CLI `main.rs`) reject node-valued movement bounds loudly for
  `--target hip` (C is canonical).
- **Wire**: `WireRtDim { Lit, ToEnd, Node }` in `schema.rs`; `wire_bound` maps it.
- **issue_368 residual pin** retargeted to the current (deeper) boundary; sibling pins green.

**Where forward-C and grad still fail closed (loud):** `symbolic_occurrences`
(`dag.rs`, the `panic!` around the `!bound` arm ~line 1549) fails loud when a node-valued
movement OUTPUT dim or a runtime RESHAPE-target dim has no `Load` source. That is the
declaration gap §4 step 3 closes.

## 4. Remaining implementation — step by step

### Step 2 — `RtDim::Sym` + reshape `new_shape: Vec<RtDim>` (large, atomic, compiler-guided)

- Add `RtDim::Sym(String)` (a symbolic dim declared elsewhere, e.g. a bystander `batch`).
  This DROPS `Copy` on `RtDim` (String) — change `.copied()` → `.cloned()` on bound/stride
  slices (grep for `.iter().copied()` on `RtDim` in `vmap.rs`, `emit_*`) and any `*rtdim`.
- Change `RiscOp::Reshape { new_shape: Vec<DimInfo> }` → `Vec<RtDim>`. This is wide
  (`Reshape` is common). Do it as ONE atomic compiler-guided sweep across
  eval/verify/grad/specialize/vmap/tier2/backend-c/hip/metal/wire/tests. Conversions:
  `DimInfo::Lit(n)` → `RtDim::Lit(n)`; `DimInfo::Named(s, _)` → `RtDim::Sym(s)`; the output
  TensorType stays `Vec<DimInfo>` (convert `RtDim`→`DimInfo` for the result type).
- **Lowering the runtime reshape dim** (`lower.rs`, `extract_reshape_dim_list` ~line 9560):
  today the non-static shape-arithmetic case raises a loud lowering error
  (`is_shape_derived_arith_dim` → "refusing to lower a guessed extent"). Replace that arm:
  lower the arithmetic expression to a rank-0 int scalar node (via `lower_expr_node`),
  append it to the reshape's `inputs`, and record `RtDim::Node(slot)` for that dim. The
  static (`fold_shape_derived_static_size`) and bare-`shape(x,i)`
  (`dim_expr_from_shape_arg_with_source`) cases stay as-is (fold to `Lit`/`Sym`+shape_dep).
- Reshape eval/verify/C-emit must handle `RtDim::Node`: eval reads the scalar for the
  target dim (numel check uses it); C-emit declares the output dim from the scalar (see
  step 3); wire gets `WireRtDim` already.

### Step 3 — node-sourced symbolic-dim declaration (the missing piece)

`shape_deps` (`dag.rs`) only keeps a *Load* source alive; the actual declaration always
sources from a Load (`symbolic_bindings` → `int name = inputs[slot]->shape[axis]`,
`backend-c/src/emit.rs` ~line 989). Extend it so a symbolic output dim can be sourced from
a **scalar node**:

- `symbolic_occurrences` (`dag.rs` ~1500-1549): when a `Named(sym, None)` output dim has no
  Load source but IS the runtime output of a node-valued movement op or a `RtDim::Node`
  reshape target, record a node-source occurrence (dim name → the scalar `NodeId`) instead
  of `panic!`. (Keep the panic for genuinely sourceless symbols — soundness.)
- C backend (`emit.rs`): declare a node-sourced dim as `int name = t{node}->data[0]`
  (integer read; reuse `bound_c_expr`'s precision handling). `emit_shrink`/`emit_stride`/
  `emit_pad` already declare movement output dims from the bound; unify so reshape targets
  use the same path.
- eval (`bind_symbolic_dims`, `dag.rs`): resolve a node-sourced dim from the evaluated
  scalar node's value (the eval lane already binds symbolic dims before eval — thread the
  node value in).
- **Soundness guard (the over-unification trap):** if the checker WILDCARDS a node-valued
  movement/reshape output and unifies a whole `shrink→stride` chain to ONE sig-named symbol
  (a degenerate direct-return pattern), two nodes could declare the same symbol → wrong
  shape or a duplicate C declaration. The oracle path (movement outputs anchored by an
  intervening `reshape`) keeps each dim distinct; but detect/guard the degenerate case
  (e.g. fail loud on a duplicate node-source for one symbol) so it can never silently
  mis-size. A `chelis eval` vs `chelis build --target c` parity test catches it.

After steps 2+3: `chelis build --target c` of the forward reproducer must produce
`[1.5, 3.5]` (currently ICEs at the reshape target). Add a forward eval-vs-C parity test
(model on `crates/chelis-cli/tests/issue_558_shape_value_read.rs`).

### Step 4 — runtime grad adjoints (lift the guards)

- **Stride adjoint** (`grad.rs` ~1319-1436): the `strided_axis_size` closure panics on a
  `Named(_, None)` strided axis. Replace with a fresh `RiscOp::Shape { axis }` read on the
  forward INPUT for the source extent `n_a` (used as the trim `Shrink` bound
  `(RtDim::Lit(0), RtDim::Node(n_a))`), and make the merge reshape dim `m_a*step` a
  `RtDim::Node` computed by runtime `Mul` (m_a = the cotangent's own dim, itself runtime).
  The `SHRINK_TO_END` sentinel is INSUFFICIENT for the strided axis — the overshoot trim
  must cut to exactly `n_a`. Bystander (non-strided) symbolic axes keep the sentinel path
  (do not touch — pins `stride_adjoint_symbolic_bystander_axis_uses_sentinel_trim`,
  `prod_reduce_adjoint_symbolic_bystander_axis_uses_sentinel_slices` must stay green).
- **Shrink adjoint** (`grad.rs` ~1274): replace `dim_size(dim) - end` with
  `Add(Shape(x, axis), Neg(end))` feeding a node-valued `Pad.after`.
- Note the movement adjoints currently `.expect("node-valued ... chelis#616 M2 work")` on
  node-valued bounds — replace those `expect`s with the runtime construction above.
- **ProdReduce** reduced-axis guard (`grad.rs` ~960) is OFF the avgpool path and
  independent (needs a runtime loop / closed form) — leave it guarded; do NOT flip its pin.

### Step 5 — flip the oracle + close out

- Flip `issue_368_runtime_symbolic_window_grad_is_tracked_residual` to a passing FD +
  forward-parity + eval-vs-C oracle (`[0.5,0.5,0.5,0.5]`). Reuse the file's
  `eval_ok`/`parse_tensor`/`assert_close` harness. Flip
  `stride_adjoint_symbolic_strided_axis_fails_loud` (`grad.rs`) to a passing adjoint test.
- Add the M1 forward eval-vs-C parity + negatives file (zero-size axis, shrink overshoot,
  negative intermediate, stride overshoot — eval-vs-C ERROR parity).
- Spec sync: `spec/05-risc-primitives.md` (§2.4 Stride/Shrink, the `RtDim` representation),
  `spec/06-transformations.md` (node-valued movement + reshape dims). Update the issue_368
  pin's boundary comment.
- Gates: `python3 scripts/gate.py --local`; `cargo clippy --workspace --all-targets
  -- -D warnings`; `cargo fmt --all -- --check`; `chelis lint --check .` (§8.6, no em-dash
  in Rust string literals). Full workspace suite via CI (macOS Smoke is the authoritative
  oracle). **Run `cargo build --workspace --all-targets` locally, not just `-p <crate>
  --lib`** — the first CI red on this branch was test targets that `--lib` never compiled.

## 5. Gotchas / invariants (learned the hard way)

- **Always check `--all-targets` across the workspace**, not per-crate `--lib`. `--lib`
  skips `#[cfg(test)]` modules AND other crates' `tests/`. CI runs
  `cargo build --workspace --all-targets`.
- `RtDim::Node(i)` nodes MUST be in `inputs` (never embedded NodeId-in-payload) — DCE,
  topo-sort, `verify`'s "references earlier node" check, `shape_source_for_axis`, and wire
  all assume `inputs` is the complete edge set.
- `shape_source_for_axis` (`dag.rs`) returns `None` for node-valued movement axes (soundness
  edit already in) so a runtime output dim is not mis-sourced to the input axis.
- The forward reproducer's `if/fail` guard routes forward EVAL to the HOST lane (not DAG),
  which is why forward eval "works" while forward-C ICEs — do not mistake host-lane success
  for DAG-lane coverage. `grad` forces DAG construction, which is where the real work is.
- macOS build hygiene: concurrent heavy builds starve each other past the 600s watchdog;
  build per-crate where possible and reap orphans (`python3 scripts/reap_orphans.py`).
- `chelis-python`'s `pyo3::Bound<'py, T>` is UNRELATED to the dag type — do not rename it.
- Concrete arithmetic edges to probe (soundness): zero-size axis (`start==end`), step
  overshoot (`m*step > n`), negative intermediate, i32-vs-i64 bound precision.
