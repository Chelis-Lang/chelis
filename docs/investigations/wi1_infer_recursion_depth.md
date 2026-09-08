# WI-1: type-checker recursion depth and the deep-`app` SIGSEGV

Status: investigation note for the WI-1 Phase-1 close, updated for the WS-2
follow-up. The original fix is a shared `stacker::remaining_stack()` budget
guard at every self-recursive `deep::Expr` walker in
`crates/chelis-types/src/infer.rs`, plus a thread-local exhaustion flag
drained at every public check entry. The **WS-2 follow-up (now landed)** adds
`stacker::grow` at every public check entry and at the CLI `cmd_check_one`
boundary, so a legitimately deep-but-finite reef-linked program checks
end-to-end on a grown native stack instead of tripping the per-site guard or
SIGSEGV-ing in `chelis_deep`'s derived `Clone`/`Drop`/validator. This note
records why a byte budget (not a depth constant) was chosen, the
covered-or-rejected soundness funnel, the now-landed end-to-end capability and
how the depth was confirmed reef-linking-induced (finite) rather than
intrinsic/infinite, the `chelis_deep` residual the grow covers, and two
adjacent bugs found while diagnosing it.

## What the guard fixes

`chelis check src/pricing.ch` on the reef-linked Shoals pricer SIGSEGVs.
gdb pins the crash in the *type checker*: `chelis_types::infer::infer_expr`
<-> `infer_app` mutually recurse one native frame per AST level over the
very deep `app` trees the reef-linked pricer desugars into (the 15 exports
expand into deeply-nested curried-application chains). It is **not** a
lowering bug, and the recursion is **finite** -- a 512 MiB thread stack
completes the real pricer. `chelis check` runs the checker on the process
main thread (default 8 MiB on Linux), which overflows partway through. A
native stack overflow `abort()`s the process (Rust's overflow handler is
not a panic), so it cannot be converted into a diagnostic after the fact;
the only correctness-preserving option is to refuse to recurse *before* the
frame that would overflow.

The design goal is that a type checker should never SIGSEGV on input. What
the shipped guard actually proves is narrower and stated precisely below:
the checker's *own* inference recursion (`infer_expr` and its ~30 sibling
`deep::Expr` walkers) is now stack-budget-bounded with a covered-or-rejected
funnel, so the real pricer's `infer_expr` overflow is a clean located
diagnostic. It does **not** prove the general invariant for arbitrary deep
input -- the `chelis_deep` residual below still overflows -- so that general
guarantee is the deferred follow-up's job, not a property of this change.

## Why a byte budget, not a depth constant

A throwaway probe built a distinct-name left-nested `app` chain
`(app (app (app f0 f1) f2) ... fN)` and ran the checker on it on a thread
of a chosen stack size, bisecting the depth at which the process aborts
with `fatal runtime error: stack overflow`. The probe was deleted before
commit. Measured overflow depth (chain of bare `(var fN)` applications):

| build   | thread stack          | overflow depth |
|---------|-----------------------|----------------|
| debug   | 2 MiB                 | ~27            |
| debug   | 8 MiB (main thread)   | ~105           |
| debug   | 32 MiB (test worker)  | ~424           |
| release | 8 MiB (`chelis check` production) | ~1550 |
| release | 32 MiB (test worker)  | ~6090          |

The overflow depth is build- and stack-profile dependent, spanning ~27
(debug / 2 MiB) to ~6090 (release / 32 MiB) -- a >200x spread for the
*same* input. A single static depth constant is therefore a tuning
treadmill: it cannot be simultaneously "comfortably below overflow" on
every profile and "well above any legitimate nesting" a release build could
actually handle, and it drifts with per-frame size. `stacker::remaining_stack()`
measures the actual resource, so the guard stays correct under any build
profile and any thread stack without tuning -- the same mechanism rustc
uses for its own recursive passes. The shipped guard reserves a 128 KiB red
zone (`STACK_RED_ZONE_BYTES`); where `remaining_stack()` returns `None`
(platform can't report) it falls back to a conservative static depth cap
(`FALLBACK_MAX_DEPTH`), so a `None` platform is never worse than a pure
depth guard. The throwaway depth measurements above survive only as the
rationale for the fallback cap and the test stack sizes.

## Covered-or-rejected: the soundness funnel

The pipeline has ~30 self-recursive `deep::Expr` walkers, not just
`infer_expr`. Each one now opens with the shared `stack_guard!` macro (RAII
depth counter + `stack_guard_tripped`). Many of those walkers carry no
error vector and can only `return` on bail -- a bare early return would be
a *soundness* hazard, turning a should-fail into a silent green or a
partial type result. So the guard, on bail, records the first bail
(walker site + source span, first-write-wins) into a thread-local
`STACK_EXHAUSTED` flag. Every public check entry (`infer_program`,
`check_ir_with_signature_context`, `build_type_env_from_library`,
`infer_ir_program`, `check_typed_program`) opens a re-entrant
`StackExhaustionScope` and drains the flag into its error vector before the
empty-errors gate (and again after the post-gate annotation pass, which
also recurses). A stack-exhaustion bail therefore ALWAYS surfaces as a
hard, located check failure -- never swallowed into a green or partial
result. The diagnostic names the walker that bailed and the source span, so
a valid-but-deeply-nested program tells the user where the depth is.

## The residual overflow surface is in `chelis_deep`, not the checker

Guarding every `infer.rs` walker does *not* by itself make `chelis check`
SIGSEGV-proof on arbitrary deep input. Empirically (gdb-confirmed), a deep
`app` chain run through the full `check_ir_program` pipeline on a small
stack still overflows -- but now *outside* `chelis_types`, in the
`chelis_deep` core AST crate:

1. **`chelis_deep::ast::Expr::clone()`** -- the *derived* `Clone` recurses
   through the nested `List` children. A pure clone of a depth-8000 chain on
   an 8 MiB stack SIGSEGVs with no checker involved. The checker
   unavoidably clones deep `Expr`s (`build_ir_type_env`,
   `annotate_expr_with_scope`, the combined-IR map, type-expr extraction).
2. **Implicit `Drop` of a deep `Expr`** -- the derived recursive drop
   overflows when the chain is deep enough (a depth-50000 chain SIGSEGVs on
   drop). Even just dropping the input tree at end-of-check can overflow.
   `PartialEq` and the derived serde impls have the same shape.
3. **`chelis_deep::validate::validate_expr`** -- `chelis_deep`'s own
   recursive validator, invoked by the checker at `infer.rs` via
   `chelis_deep::validate::validate(exprs)`.

These live on the foundational `Expr` type. A per-site guard cannot reach a
derived `Drop`, and hand-writing non-recursive `Clone`/`Drop`/`PartialEq`
(plus explicit serde) for `Expr` is an invasive change to the core AST type
with broad blast radius (every consumer, the serde wire format, the 62-tag
Deep contract). The original `chelis_types` guard made the gdb-pinned
real-pricer crash (which overflows in `infer_expr` first) a clean located
diagnostic but did not cover this `chelis_deep` residual.

**The WS-2 follow-up covers this residual without touching `Expr`'s derived
traits:** because the whole CLI check operation now runs inside
`stacker::grow` (via `run_on_grown_stack` around `cmd_check_one`), the
derived `Clone`/`Drop` and `chelis_deep::validate` all execute on the grown
segment alongside inference. There is no per-site guard for a derived `Drop`,
but a derived `Drop` running on a 512 MiB segment does not overflow at any
realistic finite depth. So the residual is closed at the stack-budget level,
not by rewriting the AST traits. See "Capability (a)" below.

## Capability (a): full pricer checks end-to-end -- LANDED (WS-2)

The previously `#[ignore]`d repro
(`pricer_depth_chain_checks_without_depth_error_once_followup_lands` in
`crates/chelis-types/tests/infer_recursion_depth_guard.rs`) is now un-ignored
and green: a depth-4000 chain (standing in for the pricer's true nesting)
`check`s cleanly through `check_ir_program` instead of tripping the guard.

### The depth is reef-linking-induced and FINITE, not intrinsic/infinite

This was confirmed before choosing the fix (the brief's required diagnosis):

- **The reef linker concatenates, it does not re-inline.**
  `link_graph_with_package_tags` (`crates/chelis-reef/src/lib.rs`) walks every
  package/module and pushes each module's `rewrite_module_decls` output into a
  flat `LinkedModule` list. `rewrite_module_decls` renames symbols to mangled
  `internal_name(package, module, name)` references; it does **not** inline a
  callee's body into its call sites. So a program that references export `foo`
  N times yields N `(var <mangled-foo>)` leaves, not N copies of `foo`'s body.
  Linking adds **breadth** (more decls, more sibling references), not unbounded
  per-expression **depth**. The "each export re-inlined rather than referenced"
  failure mode the brief warned about does not occur.
- **The recursion is structural over a finite tree.** `infer_expr`/`infer_app`
  recurse only on `children(...)` -- strictly-smaller subtrees of a finite,
  acyclic `deep::Expr`. There is no fixed-point or self-referential `Expr`, so
  the descent depth is bounded by the deepest single desugared `app` body
  (the pricer's deepest export, which nests in the low thousands). A 512 MiB
  thread completes the real pricer (measured), confirming finite depth.

Therefore the depth is a *finite source/linking property*, and growing the
native stack is the correct fix -- it is not papering over an infinite
recursion. (The orthogonal observation that the linked trees are large for
every downstream pass is a separate efficiency matter, not a soundness one,
and is out of scope for the SIGSEGV fix.)

### The fix: `stacker::grow` at every public check entry (NOT entry `maybe_grow`)

The shipped fix runs the WHOLE check pipeline on a freshly-grown stack
segment (`with_grown_stack` -> `stacker::grow(512 MiB, ...)`), wrapped around:

- every public check entry in `crates/chelis-types/src/infer.rs`
  (`check_ir_with_signature_context` -- the funnel for `check_ir_program` and
  `check_ir_with_context` -- plus `infer_program`, `infer_ir_program`,
  `build_type_env_from_library`, `check_typed_program`), and
- the CLI `cmd_check_one` operation
  (`crates/chelis-cli/src/main.rs`, via the public `run_on_grown_stack`),
  which additionally covers the reef/deep loader, the linked-program
  `clone()`, the desugarer, the fitness structure walk, `chelis_deep::validate`,
  and the implicit drop of the deep tree -- the derived-recursive `chelis_deep`
  surface that lives outside the type checker.

**Correction to the original plan above:** the plan proposed
`stacker::maybe_grow` at the entry. That does NOT work, and the implementation
deliberately diverges. `maybe_grow` grows only when the *current* frame is
within the red zone of stack exhaustion. A check starts on a fresh, near-empty
thread stack, so at the entry `maybe_grow` sees ample headroom and never grows;
the stack only nears exhaustion many frames INTO the recursive descent, far
from the entry, with no further grow site there -- so a per-site `stack_guard!`
fires first (verified empirically: an entry `maybe_grow` left the depth-4000
repro failing in `validate_tensor_precisions` after `infer_expr` was covered).
Per-site `maybe_grow` at `infer_expr` alone also fails: it just moves the cliff
to the next recursive pass (validate / annotate), since ~30 walkers descend the
same tree. Unconditional `stacker::grow` at the boundary allocates one large
segment that ALL passes (inference, validate, annotate, clone, drop) share, so
the cliff is lifted uniformly. `grow` reserves the segment via `mmap`; on Linux
the pages are demand-zeroed, so a shallow check only commits what it touches --
the 512 MiB is reserved address space, not resident memory.

The per-site bail `stack_guard!`s shipped in the original PR remain the safety
net for input deeper than even a grown segment can hold (covered-or-rejected),
and for the `None`-platform fallback. The depth-guard test suite exercises that
net by shrinking the grown segment (test-only `set_grow_segment_bytes_for_test`
thread-local override) so a depth-4000 chain overflows it and the guard fires
with a located diagnostic -- proving the net still backs up the grow.

Post-#1019 transition note: the #908 producer switch briefly inserted a new
recursive Node/BareList-to-List normalization before those guarded walkers.
On the deliberately reduced test segment, that clone could abort before the
safety net ran. The #1023 stabilization makes the transitional normalization
an explicit heap-worklist traversal and removes its stale recursive-walker
exemption. This restores the covered-or-rejected contract; it does not make
the legacy representation bridge a permanent #908 boundary.
The regression fixture itself still owns and recursively drops its 4,000-deep
input after the borrowed check returns. Its worker therefore clears the
test-only reduced-segment override and drops that fixture inside a fresh
production-sized grown segment. Worker aborts remain fatal, and the two tests
remain active on macOS. Thus a green test proves the typed stack-budget
diagnostic rather than treating SIGBUS, a panic, or an ignored test as valid
rejection evidence.

## WI-1 guard-completeness residual (WS-5 walker scan; WS-2 red zone)

WS-5's syn-based source scan of `infer.rs` (locked by the walker-coverage
test in `crates/chelis-types/tests/`) proved the "every recursive
`deep::Expr` walker carries `stack_guard!`" invariant is INCOMPLETE: beyond
the ~27 guarded sites, six production recursive walkers descend
arbitrary-depth structures WITHOUT a guard --
`type_expr_has_tensor_prec_var` (the one clearly unbounded, over nested type
exprs), `literal_static_value`, `tensor_dim_exprs_from_type_expr`,
`tensor_precision_expr`, `tensor_dims_from_type_expr`,
`top_level_arm_is_irrefutable` -- plus three `(module ...)`-only descenders
and one cycle-guarded false positive. The coverage test allowlists these
with category-coded reasons (locks "no NEW unguarded walker" without
claiming completeness); its stale-entry assertion auto-detects when each is
later guarded.

Mitigation in place: WS-2's unconditional `stacker::grow` at the check
entries (the 512 MiB segment all passes share) means these unguarded
walkers no longer SIGSEGV at realistic depth -- the residual is a
located-diagnostic gap on a sufficiently deep input on an exhausted segment,
not a live crash. WS-2 also flagged that the 128 KiB red zone is not
provably sufficient for arbitrarily small grown segments (a single deep
`infer_expr`/validate step can exceed 128 KiB between guard checks);
harmless at the 512 MiB production size.

Follow-up (deferred): guard the six arbitrary-depth walkers (start with
`type_expr_has_tensor_prec_var`); the allowlist test names exactly which
entries to delete as each is fixed.

Scope correction (red-team checkpoint 1): "covered-or-rejected on deep
input" holds for the GUARDED `chelis-types/infer.rs` walkers, NOT for the
`chelis_deep` surface. The derived `chelis_deep::ast::Expr` Drop/Clone (a
destructor cannot carry `stack_guard!`) AND `chelis_deep::validate::validate_expr`
(`validate.rs:151`) are grow-MITIGATED by WS-2's 512 MiB segment but are NOT
covered-or-rejected: gdb confirms a RAW SIGSEGV with no located diagnostic at
`drop_in_place<Expr>` ~45k depth debug / ~140k release, and in `validate_expr`
on a deep `t-fn` sig on a small segment. Real reef-pricer depth is low
thousands (~100x margin), so not reachable in practice. Note `validate_expr`
is outside BOTH the infer.rs guard set AND WS-5's coverage scan (which only
`include_str!`s infer.rs), so it is tracked HERE explicitly. The proper close
for this surface is `stacker::maybe_grow`/per-pass guards reaching into
`chelis_deep`, on the same follow-up.

## Adjacent bug (b): `chelis build` ICE on the pricer -- ROOT IS chelis-types (WS-3)

Distinct from the checker overflow: `chelis build src/pricing.ch` hits a
deliberate ICE `panic!` at `crates/chelis-ir/src/dag.rs:1345`:
"internal compiler error: symbolic dim `*` referenced by a non-Load node
(Reshape) ... but no Load input declares it" (the Sum bucket at :1308 is the
same root in a different node).

WS-3 traced it end-to-end (instrumented DAG dumps) and the root is NOT an
IR-producing-pass bug -- it is in chelis-types. `nn = cast(shape(spots,0),
int64)` semantically IS the spots `Load`'s dim 0 (`n`), but the checker
infers the reshape operand `to_tensor(map(.., range(0, nn)))` (a
runtime-length list) as the shape-erased wildcard `*` (`Dim::Wildcard`), and
`*` wins over `nn` during inference. So `const_col`'s CALL-SITE result type
is `tensor[*, 1]` -- the link from `nn` back to the declared `n` is erased
BEFORE lowering. `vmap` reads its batch dim from that `*`; the kernel's Load
and Sum carry `*`; backend-c `rename_anonymous_dims` mints a separate
`_anon_dim_*` per node, so the Sum/Reshape references a name no Load declares
and the dag.rs guard correctly panics. Three prototyped chelis-ir fixes all
got clobbered: the call-site `*` is re-injected from scope at every IR/host
pass (`actualize_tensor_helper_types`, `remap_tensor_helper_dim_symbols`);
no chelis-ir-only change recovers `n` because the recovery source (`spots`)
is not in the kernel's scope.

The fork:
- **CLEAN (recommended):** fix chelis-types so `const_col` returns
  `tensor[n, 1]` -- preserve `n` through the `nn = shape(spots,0)` /
  `to_tensor` / `reshape` chain. Then `nn` traces to the spots `Load` in the
  IR too, the kernel batch dim is `n` (Load-declared), the guard passes.
  Single-altitude, no IR/codegen schema change.
- **FOUNDATIONAL (not recommended for one idiom):** relax the
  dag.rs:1331-1334 Load-only invariant to permit ENTRY-declared symbolic dims
  (bound at kernel entry from input metadata), have codegen emit
  `int <dim> = inputs[slot]->shape[axis]`, and stop the wildcard
  re-injection. Bigger blast radius (the guard contract, host helper
  actualization, codegen).

Secondary real bug (insufficient alone): `rename_anonymous_dims` mints a
fresh `_anon_dim` per node over output types only, desyncing the same logical
dim -- fixing it alone just turns `*` into an `_anon_dim` the guard still
correctly rejects.

Status: RESOLVED via the CLEAN fix (S2, chelis#405). The fix is a single
chelis-types change in `narrow_wildcards_with`
(`crates/chelis-types/src/infer.rs`): the #39 post-defsig wildcard
narrowing now substitutes a body `Dim::Wildcard` with a declared
`Dim::Var(v)` -- not only a `Dim::Lit` -- when `v` is bound by a *parameter*
tensor-dim position of the declared signature (the new `param_bound_dvars`
gate). For `const_col[n](spots: tensor[n, ..]) -> tensor[n, 1]` the return
dim `n` is the same dim var as the `spots` parameter axis-0, so the body's
`tensor[*, 1]` narrows to the published `tensor[n, 1]`; `n` then traces to
the spots `Load` in the lowered IR, the `vmap` kernel batch dim is
Load-declared, and the dag guard passes with no IR/codegen change.

The narrowing is gated, not broadened: a *return-only* dim var (e.g.
`arange[n](start: int32, stop: int32) -> tensor[n, int32]`, length from a
value parameter) is NOT param-bound, so its body wildcard is left as `*` --
exactly preserving the RT-39+44 soundness boundary (commit 8067c9ce) that
the literal-only restriction was protecting. The earlier broad
`Wildcard -> Dim::Var` narrowing was unsound precisely because it baked
such free return-only vars into the scheme; the `param_bound_dvars` gate is
the line between "the caller supplies this dim" (safe) and
"output-inferred / value-parameter-derived" (unsafe).

The FOUNDATIONAL fork (relaxing the dag.rs Load-only invariant) was NOT
taken; the clean single-altitude fix was sufficient.

Pinned by `crates/chelis-cli/tests/reshape_symbolic_dim_vmap_column.rs`:
`vmap_over_symbolic_column_builds_without_symbolic_dim_ice` is now
un-`#[ignore]`d and green (the build succeeds, no undeclared symbolic dim
reaches cc). The companion `..._currently_ices_at_the_vmap_kernel` test
that pinned the pre-fix ICE was deleted (it asserted a crash that no longer
happens). A narrow chelis-types unit lock
(`const_col_chain_propagates_declared_dim_to_callers` plus the
`narrow_*`/`param_bound_dvars` pure-function tests in `infer.rs`) regresses
the checker-level behavior.

The secondary `rename_anonymous_dims` per-node `_anon_dim` minting (noted
above) is moot for this idiom now that the dim is `Load`-declared rather
than `*`; it remains a latent backend-c sharpening opportunity but no
longer fires on the const_col path.
