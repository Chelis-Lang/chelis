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
- **`if`/`fail` in the DAG lane**: SUPERSEDED by chelis#1464. This entry
  described `fail(...)` lowering to a zero-`Const` placeholder conformed to
  the if's rank. That placeholder was selected as the branch value on the
  TAKEN path, so a transformed `fail` returned zero and exited 0 in both
  lanes, against `spec/06-transformations.md` §2.10.1 and §5.2. A direct
  `fail(...)` branch now lowers to the [05-OP-68] guarded abort instead, and
  an indirect one is a typed rejection; neither substitutes a value. The
  placeholder survives only at entry level, where the host lane owns the
  abort and discards it. `lower_mean`'s count constants still carry the
  shape-deps that eval and the C anon-dim renaming resolve.
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

## 2b. Red-team round (fresh-context subagent, executed per CLAUDE.md)

Two findings, both fixed on the branch:

1. **CRITICAL (silent mis-size)**: a multi-axis shrink mixing a runtime axis
   with a literal-bounded axis returned a wrong, unguarded C tensor while
   both lanes exited 0. `infer_shrink_app` collapsed EVERY axis to a
   wildcard when any bound was non-literal; unification filled the runtime
   axis from the sibling literal axis and the C backend baked it. Fixed by
   per-axis inference (shrink/pad/stride) plus a backend
   static-extent-vs-runtime-extent abort guard on every runtime axis
   (defense in depth against future checker imprecision). Oracles:
   `issue_616_multi_axis_runtime_shrink_matches_c` / `..._pad_...` /
   `issue_616_literal_axis_still_checked_beside_runtime_axis`.
2. **MAJOR (grad-to-C ICE)**: grad of a runtime shrink feeding a reduction
   directly had no C declaration source for the backward restore `Expand`.
   An Expand with an ANONYMOUS size and a positionally-aligned shape-dep is
   now op-declared and emitted from the dep's actual shape; real-symbol
   Form-3 broadcasts are excluded (their dep has a different axis layout).
   Oracle: `issue_616_runtime_shrink_grad_through_reduction_matches_c`.

The same machinery gave the guarded completion oracle full **gradient**
eval-vs-C parity (the compiled binary prints `[0.5, 0.5, 0.5, 0.5]`; the C
leg is part of the oracle test now).

## 3. Executable oracles

- `issue_368_grad_concat_windows.rs::issue_368_runtime_symbolic_window_grad_is_half_everywhere`
  — THE completion oracle (analytic + FD + forward parity in the eval lane,
  plus gradient eval-vs-C parity of the compiled binary).
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
- **Guarded (`if`/`fail`) FORWARD programs through `chelis build`** —
  RESOLVED by chelis#631: the checker types `concat` from the concat axis
  and the statically-counted element count (spec/04-type-system.md
  §4.5.4), anon dims are no longer dim-substitution keys, and
  fail-reaching forward bodies stay in the host lane's real `if`/`fail`
  control flow instead of the DAG lane's zero-placeholder mask form.
  Oracle: `issue_631_guarded_forward_concat_c_parity.rs` (build-and-run
  parity at two lengths plus fail-branch error parity in both lanes).
  The `grad`/`vmap` exemption is scoped to the transformed subtree: a
  forward `fail` beside a transformed call still routes the enclosing body
  through host `if`/`fail`. This routing repair deliberately left
  transformed-subtree handling unchanged; it never claimed that a taken
  internal `fail` may become a numeric placeholder. chelis#1464 owned that
  pre-existing spec divergence and has since closed it with [05-OP-68]. Oracle:
  `issue_662_forward_fail_grad_scope.rs` (taken and untaken forward siblings
  plus an untaken grad-internal routing control).
- **Checker over-unification of movement chains** — RESOLVED
  (chelis#632, in two parts). The OBSERVABLE C-lane abort on a
  direct-return `shrink -> stride` chain was the wildcard-KEYED dim
  substitution painting the sig symbol onto both movement nodes; anon
  keys are excluded and the helper root retypes positionally
  (op-declarable axes only), giving the degenerate chain full eval-vs-C
  parity (flipped pin:
  `issue_632_direct_return_movement_chain_eval_matches_c`). The checker
  side now mints a FRESH extent for every non-identity stride/pad axis
  (identity-only symbolic pass-through, mirroring
  `shape_source_for_axis`; spec/04 §4.7): the `stride(&x, 2, 2)`-keeps-
  `batch` annotation lie is gone, and the false §4.4.1 rigidity
  rejection of `sig f: tensor[n] -> tensor[u]` over `stride(x, 2)` is
  fixed (oracle: `issue_632_literal_stride_under_sig_symbols_matches_c`;
  checker pins: `chelis-types/tests/issue_632_movement_fresh_extents.rs`).
- **`shape_source_for_axis`'s Reshape arm** still recurses positionally into
  the input (axis-naive); unsound in principle for rank-shifting reshapes
  whose downstream symbolic axes trace through it. The oracle paths avoid
  it (runtime targets return op-declared sources first); tighten when a
  reproducer appears.
- **Bound precision**: bound scalars are read through their declared integer
  type and cast to C `int`; extents beyond `int` range are out of scope
  (allocation-impossible sizes).
