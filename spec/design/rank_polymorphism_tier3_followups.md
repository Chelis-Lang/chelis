# Tier-3 Rank Polymorphism — Follow-up Plan (handoff)

**Status of the base feature.** Tier-3 *name-preserving rank polymorphism* shipped
in PR #337 (squash-merged to `main` as `c581cb8`). A single `def` reduces a
**named axis** rank-polymorphically and the checker carries the surviving named
axes through; it builds and runs on the **C backend**:

```chelis
def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
;; tensor[batch, seq, hidden] -> tensor[batch, hidden]   (one def, any rank)
```

Authoritative spec: `spec/04-type-system.md` §4.5.3 (+ revised §4.2/§4.5.1).
Design + red-team history: `spec/design/rank_polymorphism.md`.
Acceptance oracle: `crates/chelis-cli/tests/rank_poly_tier3.rs`.

Follow-up status:

- **#338 — DONE.** `chelis eval` named-axis support shipped: the host runtime
  routes named-axis work through `lower_subexpr_program` + the forward DAG
  evaluator (the `grad`/`vmap` lane), at two sites in
  `crates/chelis-compiler-api/src/runtime.rs` (def-call boundary with
  formal-typed placeholders; reduction site with a static-typed operand
  placeholder). The Tier-3 corpus now runs the eval-vs-backend agreement
  oracle at ranks 2/3/4 (non-square) plus a parity-corners suite. The red
  team also surfaced (and the same PR fixed) a lowering bug shared with the
  C backend: unary/Tier-2 elementwise arms stamped annotation-derived
  output types that stay symbolic inside a rank-poly inline, so
  `sum(exp(x), seq)`-class bodies miscomputed on BOTH lanes (backend
  garbage since #337, masked by a check-only corpus test; now executable
  and pinned). Pinned decline-not-wrong residuals: a shape-rewriting pipe
  stage and a match-pattern-bound operand
  (`pipe_rewriting_stage_then_named_reduce_is_a_pinned_gap`,
  `match_pattern_operand_is_a_pinned_gap`). See the "Eval support" and
  "Elementwise output-type fix" notes in
  `spec/design/rank_polymorphism.md`.
- **#340** — `max_reduce`/`min_reduce`/`prod_reduce`/`argmax`/`argmin` in a `..r`
  body (they route through the host scalar lane and don't compile; currently
  *rejected at check time* to keep check↔backend in sync). **The remaining
  open item.**
- **#339 — DONE.** Shipped in PR #382 (squash-merged to `main` as `ba6e8c2`),
  out of the recommended order (before #340). Named-axis expand:
  `expand(x, new, size)` inserts a trailing named axis and
  `expand(x, new, size, anchor)` inserts before a uniquely located named
  anchor; insertion inside an opaque spread, name collisions, and
  non-literal sizes are hard errors. Variadic named-axis reduction:
  `sum(x, seq, head)` for the value reductions
  (`sum`/`mean`/`max_reduce`/`min_reduce`/`prod_reduce`), desugared at
  lowering to the nested 2-arg composition; duplicate axes are a hard
  error and `argmax_reduce`/`argmin_reduce` get a targeted rejection.
  Body-Discipline admission in a `..r` body is unchanged (`sum`/`mean`
  only — that boundary still belongs to #340). Authoritative spec:
  `spec/04-type-system.md` §4.5.3; design + red-team history (including
  the two loud-not-silent collision flavors pinned as Known gaps):
  `spec/design/rank_polymorphism.md`. Acceptance:
  `crates/chelis-cli/tests/rank_poly_tier3.rs` (incl. the `variadic_*`
  suite).

---

## Recommended order: #338 (done) → #340 → #339 (shipped out of order; #340 remains)

### Why this order

1. **#338 is the only correctness/consistency gap.** A program that `check`s
   clean and `build`s+runs *errors* under `chelis eval` — breaking the repo's
   "eval agrees with the C backend" invariant. The other two are missing
   *features*, not gaps.
2. **#338 unlocks the test oracle.** The Tier-3 corpus had to *skip* the
   eval-vs-backend agreement check (the strongest oracle, used throughout the
   Tier-2 corpus) because of #338 — it verifies C-backend numerics directly
   instead. Fixing #338 restores that oracle for everything built afterward.
3. **#338 and #340 share a root.** Both are "a non-DAG lane doesn't handle
   named-axis reductions." The tensor-DAG lane already resolves named axes
   (`resolve_reduce_axis`, `crates/chelis-ir/src/lower.rs`) and supports *all*
   reduce ops; and `runtime.rs` already routes `grad`/`vmap` through
   `chelis_ir::lower::lower_subexpr_program` + the forward DAG evaluator. So the
   right #338 fix likely **subsumes #340**.
4. **#339 is purely additive** (no gap) and benefits from a reduction surface
   that is complete *and* eval-verified before new arms are added on top.

### The unifying move (spike VERDICT: lanes diverge)

The #338 spike confirmed the lanes are genuinely separate. The eval fix lives
entirely in `chelis-compiler-api/src/runtime.rs` and reuses lowering machinery
that already works; #340 is the *build host lane* (`chelis-ir/src/host.rs`
host-type inference only special-cases `sum`/`mean` with int-literal axes;
`chelis-backend-c/src/host_emit.rs` falls through to an "unsupported builtin"
stub). Nothing in the eval fix touches those. So: #338 shipped alone, and #340
is the immediate next step while the backend-lane context is hot. The
shareable artifacts are the `REDUCTION_BUILTIN_NAMES` set in runtime.rs and
the Tier-3 corpus shapes (the eval agreement oracle from #338 directly
becomes #340's acceptance oracle once max/min/prod build).

---

## #338 — eval named-axis support

**Root.** `crates/chelis-compiler-api/src/runtime.rs` evaluates call args
generically before dispatch, so `(var seq)` resolves as a *value* →
`unknown runtime name seq` (runtime.rs ~732/1070). Even past that, the reduction
handlers (~2564) do `expect_int_arg(args, 1)`. And `RuntimeTensorValue` carries
only numeric shape, no names — name→index resolution needs type info that lives
only on the compile/lowering path.

**Approach (preferred).** Route a reduction whose axis arg is a `(var name)`
through the DAG path (`lower_subexpr_program` + forward DAG evaluator) that
`grad`/`vmap` already use (see the `TransformKind` doc comment ~L22-28). The DAG
path resolves the named axis via `resolve_reduce_axis` exactly as the C backend
does. Alternative: a pre-pass that rewrites each named reduction axis to its
resolved integer index using the checker's typed nodes before eval.

**Acceptance.** `chelis eval` on the repro yields the same numerics as the C
backend; add an **eval-vs-backend agreement** test to `rank_poly_tier3.rs`
(mirror Tier-2's `*_builds_and_runs_*` oracle) at ranks 2/3/4 with **non-square**
operands.

## #340 — max/min/prod_reduce in a `..r` body

**Root.** The host scalar lane special-cases only `sum`/`mean`
(`crates/chelis-ir/src/host.rs` ~6330 host-type inference + ~6878;
`crates/chelis-backend-c/src/host_emit.rs` ~1617 "unsupported builtin"). A
rank-poly callee with max/min/prod is force-inlined into the host lane and hits
the stub (the axis name leaks as a bare C identifier). Mitigated in two layers
(defense-in-depth):

1. **Check-time** — `shape_class` (`crates/chelis-types/src/builtins.rs`) admits
   only `sum`/`mean` among the reductions as `NameTracked` (`expand` is also
   `NameTracked` since chelis#339); the other reductions are rejected at check
   time in a `..r` body
   (`rank_poly_tier3.rs::max_reduce_in_rank_poly_body_rejected`).
2. **Backend guard** — `reject_unsupported_host_lane_reductions` in the CLI
   build pipeline (`crates/chelis-cli/src/main.rs`), backed by
   `chelis_ir::host::host_program_unsupported_host_reduce` /
   `UNSUPPORTED_HOST_LANE_REDUCTIONS`, fails the build LOUDLY if any of
   `max_reduce`/`min_reduce`/`prod_reduce`/`argmax_reduce`/`argmin_reduce`
   reaches the host program, instead of emitting the silent
   `/* unsupported builtin {name} */ 0` escape. This backs up the check-time
   gate if it is ever loosened. Locks:
   `rank_poly_tier3.rs::max_reduce_rank_poly_repro_rejected_at_build`,
   `host::tests::host_program_unsupported_host_reduce_*`, with the
   `concrete_rank_max_reduce_builds_runs_and_evals` positive control proving the
   guard does not touch the working tensor-DAG path.

The variadic form (chelis#339) desugars to the 2-arg composition, so
re-admitting these ops automatically extends to their variadic calls in a `..r`
body.

**Approach (re-admission, the remaining follow-up).** Either (a) route the
rank-poly reduce through the DAG lane (likely free if #338 does this), or (b)
extend the host lane's reduce coverage to max/min/prod/argmax/argmin with
named-axis→index resolution. Then **re-admit** them as `NameTracked` in
`shape_class` (update the `shape_class` pin test), **convert**
`max_reduce_in_rank_poly_body_rejected` into a build+run test, and remove the
now-redundant backend guard (or keep it as a structural backstop).

## #339 — expand (`R+1`) + variadic reduction surface — SHIPPED

Shipped in PR #382 (`ba6e8c2`), substantially as planned below, with these
deltas surfaced by the red-team pass (full record in
`spec/design/rank_polymorphism.md`; normative rules in
`spec/04-type-system.md` §4.5.3):

- the 4-arg anchored form `expand(x, new, size, anchor)` landed alongside the
  trailing insert; both go through an `infer_expand_app` dispatcher (the
  `infer_permute_app` pattern), since the anchored arity bypasses the HM scheme
- the inserted size must be a **positive compile-time literal** (a symbolic or
  runtime size loses the name at lowering / produces a silent shape-0 tensor)
- inserted-name collisions are rejected at check time where the signature can
  see them (the introduced-name rule in `unify.rs`); two body-internal flavors
  the checker cannot see fail **loudly** at lowering and are pinned as Known
  gaps
- variadic reduction rejects duplicate axis names (never a silent dedup) and
  `argmax_reduce`/`argmin_reduce` entirely; lowering desugars to the nested
  2-arg composition so backend/eval/`grad`/`vmap` ride the single-axis lanes

The original plan, kept for the record:

- **expand (`R+1`):** `shape_class("expand") -> NameTracked`; teach
  `check_expand_signature` (`crates/chelis-types/src/infer.rs` ~13645) to insert
  a named axis at an end or adjacent to an existing named anchor (both keep
  unification unitary; inserting into an opaque spread interior has no anchor and
  must stay rejected). Lowering mirrors `resolve_reduce_axis`. Body-Discipline
  already admits `NameTracked` — no walker change.
- **variadic `sum(x, seq, head)`:** the reduction HM scheme is arity-2, so
  `sum(x, seq, head)` hits the generic arity check (see the comment at the
  `axis_exprs` loop in `check_reduction_signature`, ~13548 — the loop *already*
  handles N axes; it's just arity-gated). Add an `infer_reduction_app`
  dispatcher that bypasses the arity check (mirror `infer_stride_app` /
  `infer_permute_app`), or a list-arg form. Multi-axis works today via
  composition (`sum(sum(x, head), seq)`), so this is lower priority.

---

## Cross-cutting — do not regress these

**Soundness boundary (§4.2 / §4.5.3).** Keep unification **unitary**: a shape is
`Rank? (Name Rank?)*`, each interior split fixed by a *named* anchor; spreads
bind to the actual *named* dims (not a count). Name↔Lit / ambiguous / absent /
adjacent-spread splits are **hard-rejected, never guessed**. Body-Discipline
admits only *name-trackable* ops (elementwise + named reductions) — positional
`permute`/`reshape`/`matmul` stay rejected in a `..r` body. Any new arm (#339
expand, etc.) must preserve all of this and ship with positive+negative parity.

**Backend invariant.** The IR `DimInfo` (`crates/chelis-ir/src/dag.rs`) has no
rank variant, so no `Dim::Rank` can reach a backend by construction — a
successful lowering *is* a successful monomorphization. The
`tensor_dim_substitutions` *different-rank guard* (`lower.rs` ~901) is
load-bearing (it fixed a non-square miscompile in #337) — do not remove it.

**Process discipline (the lesson from #337's three CI fix rounds).** Run the
**exact gate commands locally before pushing**, not lib-only / targeted proxies:

```sh
python3 scripts/gate.py            # the single source of truth, or its parts:
cargo build --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings   # NOT lib-only
cargo fmt --all -- --check
cargo run -p chelis-cli --bin chelis --quiet -- lint --check .   # runs AFTER clippy; §8.6 no-em-dash applies to Rust string literals too
cargo nextest run --workspace --profile ci
```

`macOS Smoke` is the authoritative full-workspace oracle in CI; the Linux
Integration/Lint jobs can flake on disk-full (`os error 28` / `ld signal 7`).
Per `CLAUDE.md`, each of these reopens the soundness surface or the backend, so
each deserves: spec-first test stubs, a **fresh-context** red-team pass
(`redteam-exec`) and a fresh-context code review, then the full gate green.
