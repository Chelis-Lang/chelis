# chelis#616 — Runtime-symbolic movement-op bounds: implementation record

**Status: COMPLETE** on branch `agent/616-runtime-movement-bounds` (PR #627).
The completion oracle is green:

```
crates/chelis-cli/tests/issue_368_grad_concat_windows.rs
  ::issue_368_runtime_symbolic_window_grad_is_half_everywhere
```

a passing analytic + finite-difference + forward-parity oracle whose gradient
is `[0.5, 0.5, 0.5, 0.5]` (input `[1,2,3,4]`, window 2, stride 2). This file
was the implementation handoff plan; it now records what shipped, where the
original plan was corrected, and the residuals. The authoritative behavior
spec is `spec/05-risc-primitives.md` §2.4.1 and
`spec/06-transformations.md` §2.7.1.

**SOUNDNESS IS ABSOLUTE** held throughout: every boundary that moved was
replaced by a runtime capability with eval-vs-C value AND error parity, or
stayed loud.

## 1. What shipped (one capability, applied in two places)

A tensor dimension whose value is a runtime rank-0 integer scalar node (a
`Shape` read or integer arithmetic over one), declared and resolved in the
eval and C backend lanes. Representation: `dag::RtDim { Lit(usize),
Sym(String), Node(usize), ToEnd }`, where `Node(i)` is an ABSOLUTE index into
the owning op's `inputs` (`inputs[0]` = tensor, `inputs[1..]` = rank-0
integer scalars) and `Sym` is legal only in reshape targets.

- **Movement bounds** (`Shrink`/`Stride`/`Pad` start/end/step): lowering,
  eval resolution, verify (`check_bound_source`, variable arity, `ToEnd`
  position rules), C emission with runtime abort guards.
- **Reshape targets** (`Reshape.new_shape: Vec<RtDim>`): the shape-derived
  arithmetic arm and the term-bound-var arm (`m` as an inlined parameter)
  lower to scalar nodes; the C lane declares the dim inline behind
  negativity + numel abort guards (`chelis_alloc_view` checks nothing); the
  eval lane resolves the scalar and reports numel mismatches as clean errors.
- **Op-declared symbolic dims**: `SymbolicDimOccurrence` carries a source
  (`Load { input_label, axis } | OpDeclared { node, axis }`). An op-declared
  dim is declared inline at the owning op in C (the prologue cannot
  reference a computed `t{n}`) and bound MID-EVALUATION from actual values
  in eval; a Load-declared or earlier-declared symbol makes later sites
  runtime equality guards (the over-unification guard).
- **Runtime movement adjoints** (`grad.rs`): shrink pads with
  `shape(x, axis) - end`; pad shrinks to `before + shape(x, axis)`; stride's
  upsample cascade reads `m_a = shape(g, axis)`, merges with a runtime
  `m_a * step` extent, and trims to `(0, shape(x, axis))`. Bound scalars are
  a stop-gradient boundary (the pre-pass covers `Reshape` too). The reshape
  and Expand-restore adjoints restore Load-declared symbols as `Sym` (no new
  nodes; HIP-compatible) and any other runtime axis as an explicit `Shape`
  read.
- **`if`/`fail` in the DAG lane**: `fail(...)` lowers to a zero-`Const`
  placeholder (not the legacy bogus `Load { name: "fail" }`), conformed to
  the if's rank after both branches with a shape-dep on the sibling; the
  mask expansion and `lower_mean`'s count constants carry shape-deps that
  eval and the C anon-dim renaming resolve.
- **HIP/Metal**: reject node-valued movement bounds AND reshape targets with
  clean diagnostics naming `--target c` (compiler-api seam + CLI seam in
  lockstep; the Metal seam previously had no movement arm at all and reached
  a panic).
- **Wire**: `WireRtDim { Lit, ToEnd, Node, Sym }`; `Reshape` serializes
  `Vec<WireRtDim>`; `WIRE_DAG_SCHEMA_VERSION` bumped 2 → 3 (chelis-prove
  tripwire moved with it; `Reshape` stays in the f64-free op group).

## 2. Corrections to the original handoff plan (found during implementation)

1. The oracle's `m` reshape target is a bare `(var m)` (an inlined fn
   parameter), which never reached the arithmetic refuse-to-lower arm; a NEW
   lowering arm resolves a term var bound to a rank-0 integer node.
2. Node-sourced dims cannot be declared in the C prologue (the scalar is a
   computed tensor); declaration is inline at the owning op.
3. "Fail loud on a duplicate node-source" as written would have rejected the
   oracle (both inlined `window_row` reshapes declare `m` from the same
   scalar). Shipped: first site declares, later sites equality-guard at run
   time (both lanes).
4. "Thread the node value into `bind_symbolic_dims`" was unimplementable
   (values do not exist pre-eval); shipped as mid-evaluation binding, with
   `Const`/`Expand`/reshape-target consumers resolving through it or through
   shape-dep value shapes.
5. Checker wildcards (`*`/`""`) are not stable symbols: the op-declared
   machinery excludes them (the C lane renames anon dims to unique
   per-node names first; eval computes shapes from values). Grad-built
   restore targets use explicit `Shape` reads for them.
6. `shape_source_for_axis`'s movement pass-through was tightened to
   IDENTITY-ONLY (sentinel shrink / stride step 1 / zero pad); a non-identity
   literal axis (e.g. `stride(x, 2)`) previously traced the output symbol to
   the INPUT's Load — a latent mis-size.
7. `rename_anonymous_dims`' copy-first-input shortcut is skipped for ANY
   extent-altering movement op (not just node-bound ones); this structurally
   closed the chelis#593 heap-OOB hole (its fail-closed pins flipped to
   build-and-run oracles).
8. The eval-lane zero-size shrink rejection (`start >= end`) was not
   mirrored by the C guard (`end < start`); aligned to `end <= start`.
9. `DimInfo::Named(s, Some(n))` converts to `RtDim::Lit(n)` (not `Sym`),
   preserving eval semantics.
10. Assorted: `bind_symbolic_dims` silently dropped `shape_deps` (fixed);
    grad's pruner, eval's root liveness, and verify's dangling check now
    honor `shape_deps`; the Sum adjoint's restore Expand records the forward
    input as its shape source.

## 3. Executable oracles

- `issue_368_grad_concat_windows.rs::issue_368_runtime_symbolic_window_grad_is_half_everywhere`
  — THE completion oracle (analytic + FD + forward parity, eval lane).
- `issue_616_runtime_movement_c_parity.rs` — runtime shrink/stride forward
  eval-vs-C parity, one C binary across input lengths, the guarded
  over-unified degenerate (loud abort, never mis-sized), zero-size-axis and
  overshoot ERROR parity.
- `issue_616_runtime_reshape_c_parity.rs` — the `window_row` form: forward
  and GRADIENT eval-vs-C parity, one C binary across lengths, negative-extent
  error parity.
- `issue_513_symbolic_axis_adjoints.rs` — the formerly fail-closed pins now
  passing (symbolic-sig im2col grad = ones with a C leg; runtime numel
  mismatch as error parity; symbolic strided-axis / shrink-sub-range grads);
  `issue_291_grad_shrink_stride.rs` and the `grad.rs` unit tests likewise.
- `issue_551_grad_symbolic_concat_c_build.rs` — the chelis#593 reproducers
  as build-and-run oracles.

## 4. Residuals (tracked, loud, NOT part of this issue's oracle)

- **ProdReduce over a runtime reduced axis** stays fail-closed (needs a
  runtime loop / closed form). Pin:
  `prod_reduce_adjoint_symbolic_reduced_axis_fails_loud`.
- **Runtime (node-valued) stride STEP** has no structural adjoint (a
  runtime-extent axis insertion); loud expect in the stride adjoint.
- **Guarded (`if`/`fail`) programs through `chelis build`** route via the
  host-program lane, which (a) types a list `concat` as its element type
  (`[1, m]` instead of `[2, m]` — the mean helper's input-shape assert
  aborts loudly at run time), and (b) cannot render a wildcard-typed mask
  expansion in C (loud `symbolic_occurrences` ICE / gcc failure). Both are
  PRE-EXISTING host-lane gaps, now the boundary for guarded-program C
  builds; the runtime-window machinery itself has full C parity on
  unguarded twins. Needs its own issue.
- **Checker over-unification of movement chains** (a direct-return
  `shrink -> stride` under one sig symbol): the checker's movement typing
  passes symbolic dims through unchanged, so two different extents share a
  symbol. Guarded at run time (C abort + eval mismatch error; the eval lane
  accepts when nothing consumes the symbol — pin
  `issue_616_over_unified_movement_chain_fails_loud_not_mis_sized`). A
  precise fix is checker-side movement typing (fresh extents per
  non-identity axis).
- **`shape_source_for_axis`'s Reshape arm** still recurses positionally into
  the input (axis-naive); unsound in principle for rank-shifting reshapes
  whose downstream symbolic axes trace through it. The oracle paths avoid
  it (runtime targets return op-declared sources first); tighten when a
  reproducer appears.
- **Bound precision**: bound scalars are read through their declared integer
  type and cast to C `int`; extents beyond `int` range are out of scope
  (allocation-impossible sizes).
