# WI-1: type-checker recursion depth and the deep-`app` SIGSEGV

Status: investigation note for the WI-1 Phase-1 close. The shipped fix is a
shared `stacker::remaining_stack()` budget guard at every self-recursive
`deep::Expr` walker in `crates/chelis-types/src/infer.rs`, plus a
thread-local exhaustion flag drained at every public check entry. This note
records why a byte budget (not a depth constant) was chosen, the
covered-or-rejected soundness funnel, the deferred capability (full pricer
checking end-to-end), the residual overflow surface in the `chelis_deep`
core AST crate that the deferred follow-up must address, and two adjacent
bugs found while diagnosing it. The follow-up itself is not implemented
here.

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
Deep contract). The shipped `chelis_types` guard makes the gdb-pinned
real-pricer crash (which overflows in `infer_expr` first) a clean located
diagnostic; closing the `chelis_deep` residual is the deferred follow-up's
job.

## Deferred capability (a): full pricer checks end-to-end

The `#[ignore]`d repro in the test suite anchors this: the real pricer (or a
synthetic chain at the pricer's true depth) should `check` cleanly. It
cannot today (the guard fires, and the `chelis_deep` clone/drop surface
would overflow anyway), so it is `#[ignore]`d and flips green when the
follow-up lands.

The deferred fix should use **`stacker::maybe_grow`** at the check entry --
NOT a hand-rolled larger thread stack, and NOT per-site rewrites. `maybe_grow`
transparently allocates a new stack segment when the budget is low, so it
covers inference, cloning, dropping, AND `chelis_deep::validate` uniformly
with one wrapper, reusing the `stacker` dependency this PR already paid for.
A bigger fixed thread stack only moves the cliff; rewriting `Expr`'s derived
traits is large and still would not cover every future recursive consumer.
`maybe_grow` at the boundary is the profile-independent, low-marginal-cost
fix; the per-site bail guards shipped here become the safety net for the
(rare) case where even a grown stack hits its cap.

**Before implementing, diagnose the *source* of the depth.** Do not assume
the program is intrinsically that deep:

- If the depth is **intrinsic** to the pricer (its computation genuinely
  nests that deep), `maybe_grow` at the entry is the right fix.
- If the depth is **induced** by reef-linking expanding the 15 exports into
  deep `app` trees **without sharing** (each export re-inlined rather than
  referenced), the depth is a linker artifact, and the real fix is in
  linking/sharing (let the linked program reference each export once). In
  that case `maybe_grow` only treats the symptom and the trees stay
  needlessly large for every downstream pass.

Diagnose which before choosing the fix.

## Adjacent bug (b): `chelis build` ICE on the pricer

Distinct from the checker overflow: `chelis build src/pricing.ch` hits a
deliberate ICE `panic!` at `crates/chelis-ir/src/dag.rs:1345`:
"internal compiler error: symbolic dim `*` referenced by a non-Load node
(Reshape) ... but no Load input declares it." It originates from the
`const_col` / `spot_col` reshape in the pricer producing a `Reshape` whose
op-internal symbolic dim is not declared by any `Load` input. This is an
IR-producing-pass bug (a `Reshape` should carry/declare its symbolic dims
the same way a `Load` does), not a checker-recursion issue. File and fix
separately.
