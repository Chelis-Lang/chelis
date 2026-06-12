# Rank Polymorphism (Tier 2: identity + erasure rank variables)

**Status:** IDENTITY TIER SHIPPED, INCLUDING BACKEND LOWERING. The identity
position (`&tensor[..r, p] -> tensor[..r, p]`) is implemented and gated by the
Body-Discipline check; the chelis#285 prerequisite is fixed and on `main`. The
call-site **rank-monomorphization** pass (the follow-up to PR #286) is shipped:
a program that uses a `..r` def through a concrete-rank caller now `build`s and
runs correctly — see Implementation Status below.

## Implementation Status (as shipped)

What landed vs. the plan below, with two deliberate divergences:

- **Surface syntax is `..r`, introduced *contextually*** (like a sig dim
  variable) — there is **no `[..r]` quantifier**. Write
  `def relu_forward(x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)`.
  The plan's `def f[..r](...)` examples below predate this decision; the
  contextual form is the shipped surface.
- **Identity tier is shipped and sound.** `..r` lexer/parser (Tier-3 adjacency
  rejected at parse time), `(d-rank {} r)` deep node, `Dim::Rank` + `Scheme.rvars`
  + a unitary rank-unification arm, the `Dim::Rank`-free monomorphization
  assertion at IR lowering, and the Body-Discipline check (an explicit
  shape-identity allowlist over every builtin; everything else is rejected in a
  `..r` body, so a missed classification can only over-reject, never open a
  §4.2 hole). A `..r` body may currently call only shape-identity *builtins* —
  calling a user-defined function from a rank-poly body is conservatively
  rejected (proving an arbitrary callee rank-safe is future work).
- **Erasure tier is deferred.** `&tensor[..r, p] -> tensor[p]` requires a
  genuine order-invariant *all-reduce-to-scalar* primitive; Chelis's `sum`/`mean`
  are axis-indexed (rank-reducing = Tier-3 rank arithmetic), so there is no
  sound erasure body to write today. Erasure ships when an all-reduce builtin
  is added; until then the Body-Discipline check rejects reductions in a `..r`
  body like any other shape-rewriting op.
- **Backend lowering of `..r` programs is SHIPPED via call-site rank
  monomorphization.** `chelis check` fully and soundly accepts/rejects rank-poly
  defs (this is the §258 value — the verb proliferation collapses and §4.2 is
  enforced at the type level). A *standalone* rank-poly def still emits nothing
  (`type_is_never_lowerable` / `type_expr_has_rank_var` skip it in both the DAG
  and host lanes, paralleling precision polymorphism). A program that *uses* a
  rank-poly def through a concrete caller now **builds**: call-site **rank
  monomorphization** substitutes the caller's concrete shape for `..r` in the
  inlined body — the rank analogue of the precision `prec_subst` path. The
  implementation is the structural twin of WS-A8 precision monomorphization in
  `crates/chelis-ir/src/lower.rs`:
  - `tensor_rank_substitutions` builds the formal-vs-actual rank-var map at a
    call site (binding a sole `(d-rank {} r)` formal slot to the actual arg's
    concrete dim vector), exactly as `tensor_prec_substitutions` does for
    precision.
  - `LowerCtx.rank_substitutions` threads that map through
    `type_from_type_expr_with_subst` / `try_extract_tensor_type_with_subst`, so
    an inlined rank-poly body resolves its `(d-rank)` slots to concrete dims.
  - the host lane (`crates/chelis-ir/src/host.rs`) gets the matching guards: a
    rank-poly sig is skipped from standalone emission (`type_expr_has_rank_var`),
    a rank-poly tensor parses to `HostType::Unknown` rather than tripping the
    tripwire, and a rank-poly callee is force-inlined at its call site
    (`callee_is_polymorphic_rank`) so no undefined symbol is emitted — every
    branch mirrors the precision-var path next to it.

  Because the IR `DimInfo` has no rank variant, no `Dim::Rank` can survive into
  a backend tensor type by construction: a successful lowering *is* a successful
  monomorphization. A standalone rank-poly def with no concrete caller is
  skipped from emission entirely (it has no usable monomorphization), so the
  genuinely-unresolvable case never reaches the backend rather than miscompiling.
  Verified end-to-end: a single `relu_forward(x: &tensor[..r, f32]) ->
  tensor[..r, f32] = relu(x)` called at concrete ranks 1, 2, 3, and 4 builds,
  compiles, and runs, with backend output equal to the evaluator output
  value-for-value (the `rank_poly_tier2` acceptance suite).

A fresh-context red-team pass informed the design; its verified findings (and
one correction to its central claim) are in §"Red-Team Findings".

**Owning issue:** chelis#258 (same-name `def` overloads with rank-distinct
sigs accepted at type-check time but call site doesn't dispatch by arg rank).

**Goal:** Let a single `def` be generic over tensor *rank* in the two
positions that are sound under spec §4.2, so that the relu/silu/gelu
activation family and the full-reduce primitives stop proliferating per-rank
verb names (`relu_forward` / `_2d` / `_3d` / `_4d`).

This is **Tier 2** of the three-tier analysis on chelis#258:

| Tier | Output-rank function | Example sig | Status |
|---|---|---|---|
| 1 | identity (`out = R`) | `&tensor[R, f32] -> tensor[R, f32]` | **SHIPPED** |
| 2 | constant / erasure (`out = []`) | `&tensor[R, f32] -> tensor[f32]` | **deferred** (no all-reduce primitive — see Implementation Status) |
| 3 | arithmetic (named-axis reduction) | `&tensor[..pre, seq, ..post, f32] -> tensor[..pre, ..post, f32]` | **SHIPPED (reduction `R−1` + expand `R+1`)** — check + C backend |

**Tier-3 Update (name-preserving, shipped for reduction).** The original plan
declared Tier-3 out of scope because the sketched `R ++ [k]` concatenation form
erases per-axis names and needs non-unitary sequence unification. The *shipped*
Tier-3 sidesteps both: a spread is **name-preserving** (it binds to the actual
named dims it covers), and a shape is the alternating form `Rank? (Name Rank?)*`
where each interior split is fixed by a **named anchor** — so unification stays
**unitary** (locate each unique named anchor in the operand; bind the spreads
between). A named-axis reduction (`def reduce_seq(x: &tensor[..pre, seq, ..post,
f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)`) reduces the named anchor and
carries the surviving named axes through; multi-axis reduction composes
single-axis reductions. The §4.2 soundness boundary is preserved by the
Body-Discipline reframe to *name-trackability* (elementwise + named reductions
admitted; positional `permute`/`reshape`/`matmul` rejected — not name-trackable
at symbolic rank). Shipped surface (spec §4.5.3): `chelis check` accepts/rejects
soundly, and call-site rank monomorphization (`tensor_rank_substitutions` /
`extract_rank_var_bindings`, with named-axis→index resolution at lowering) lets
a rank-poly reduce **build and run** on the C backend with verified numerics.
Acceptance oracle: `crates/chelis-cli/tests/rank_poly_tier3.rs`.

**Fresh-context red-team pass (PR #337).** A red team built `target/debug/chelis`
and ran adversarial programs. The checker held on every soundness probe
(Name↔Lit hard-reject, the adjacent-spread / distinct-anchor fence, the
Body-Discipline check incl. the chelis#285 sig+inline-return class and
builtin-name shadowing, and Dim::Var anchors). It found **two backend bugs that
a square-shaped corpus test masked**:

- **CRITICAL — non-square middle-anchor reduce miscompiled (FIXED).** Call-site
  specialization renamed the surviving axis to the reduced axis (`hidden→seq`),
  so `[..pre, seq, ..post]` reduces aborted at runtime whenever the reduced axis
  size differed from a surviving size. Root cause: `tensor_dim_substitutions`
  (`crates/chelis-ir/src/lower.rs`) positionally zipped a *different-rank*
  formal/actual output pair (`remap_tensor_dim_symbols` passes the rank-2 reduce
  formal against a rank-3 actual), aligning `hidden` with `seq`. Fixed by
  skipping different-rank pairs in `tensor_dim_substitutions`. The acceptance
  corpus's build+run test now uses **non-square** operands at ranks 2/3/4 so the
  mislabel would fail loudly (`named_reduce_builds_and_runs_nonsquare_at_ranks_2_3_4`).
- **HIGH — `max_reduce`/`min_reduce`/`prod_reduce` in a `..r` body don't compile
  (mitigated by restriction; chelis#340).** They route through the host scalar
  lane, which only special-cases `sum`/`mean`. Mitigation: only `sum`/`mean` are
  admitted as `NameTracked`; the others are **rejected at check time** in a `..r`
  body so a check-clean program always builds. They remain usable at concrete
  rank. Re-admission is tracked as chelis#340.

*Eval support (chelis#338, FIXED).* The tree-walking `chelis eval` interpreter
operates on nameless runtime tensors, so it cannot resolve a named axis by
itself; the fix routes named-axis work through the same
`lower_subexpr_program` + forward-DAG-eval lane that `grad`/`vmap` already
used, so name->index resolution happens in IR lowering exactly as the C
backend does it. Two routing sites in
`crates/chelis-compiler-api/src/runtime.rs`: a call to a def that *requires*
routing (its body reduces a named axis directly, or calls a rank-polymorphic
def that does) is routed at the def-call boundary with placeholders typed
from the callee's declared formal param types (mirroring how the build host
lane calls a signature-typed compiled function); a reduction app reached in
interpreted code is routed at the reduction site with the operand staged as
a placeholder typed from its static type (the checker's `{type: ...}`
annotation, the frame binding's declared type, or the top-level type-env
entry; pipes thread the piped type through Identity-class stages). The
strategy ladder is deterministic: a def-call route that does not apply or
does not lower falls back to ordinary interpretation where the reduction
site handles it, and terminal failures emit a targeted chelis#338
diagnostic, never the old `unknown runtime name` error. The Tier-3 corpus
now asserts eval-vs-backend agreement (the Tier-2 oracle) at ranks 2/3/4
with non-square operands, plus a parity-corners suite
(`named_axis_eval_parity_corners`). Pinned residuals (eval declines with a
targeted chelis#338 diagnostic while the backend builds; never wrong
numerics): a shape-rewriting pipe stage (e.g. `|> permute(1, 0) |>
sum(seq)`, `pipe_rewriting_stage_then_named_reduce_is_a_pinned_gap`); a
match-pattern-bound operand (`match_pattern_operand_is_a_pinned_gap`);
closure-captured free tensor vars and unannotated HOF lambda params share
the same decline class (untested corners, same diagnostic).

*Elementwise output-type fix (chelis#346 red team).* The unary and Tier-2
elementwise lowering arms (neg/recip/exp/log/sin/sqrt/cos/tan/atan/abs/
floor/ceil/sub/relu/sigmoid/tanh/silu/gelu/div) took their output type
from the body's `{type: ...}` annotation, whose symbolic dims survive
rank-poly inlining unsubstituted. Consequences inside a `..r` body:
`sum(exp(x), seq)` reduced the WRONG axis under the #338 eval routing,
the relu variant aborted on a DAG shape assert, and the C backend had
been emitting garbage for the same programs since #337 (masked because
the corpus elementwise test was check-only). All arms now derive output
dims from the lowered operand via `elementwise_out_ty`, the contract the
Tier-1 binary arms already used; pinned executable at
`unary_elementwise_reduce_in_rank_poly_body_builds_runs_and_evals`.

*Vmap-lane placeholder typing (chelis#351, the #346 red team's F4,
FIXED).* `vmap` over a def that calls a rank-poly named reduce checked
clean and ran correctly on the C backend, but `chelis eval` ICEd on the
dag.rs symbolic-dim guard ("symbolic dim `hidden` is referenced by a
non-Load node"): the grad/vmap transform lane (`apply_transform`)
marshalled the batched actual as a placeholder with bare `Lit` dims,
while the inlined callee body kept its formal named dims — and vmap's
rank shift (batched actual = formal rank + 1) defeats the same-rank
formal/actual remap (`tensor_dim_substitutions`' chelis#258 guard), so
the surviving name stayed unbound with no Load declaring it. The fix
adopts the chelis#338 def-call pattern in the vmap lane: the placeholder
is typed from the callee's declared formals (the vmap axis stays `Lit`;
the mapped axes carry the formal's names with runtime sizes), so the
symbolic-dim machinery binds the body's names against the placeholder
Load. `vmap(grad(f))` additionally needed the IR evaluator to resolve a
symbolic-dim-declaring Load even when it is dead under the roots' live
mask (a gradient constant in `x` never consumes the `x` Load, but its
`Expand { size: Sym(..) }` still binds from `x`'s shape); absence of
such an input stays an error, now always the dim-targeted one. The grad
lane needed no placeholder change (same-rank remap already concretizes
the dims) and is pinned. Eval-vs-backend agreement for the reproducer,
the concrete-reduce control, `grad`, and `vmap(grad(...))` is pinned at
`vmap_over_rank_poly_named_reduce_evals_and_matches_backend`, with
negative parity (a conflicting concrete dim through the vmapped callee
stays a `DimensionMismatch`, never the ICE) at
`vmap_callee_dim_conflict_stays_rejected_not_ice`.

The #371 review red team found the same ICE in a second flavor: dim-VAR
formals (`tensor[a, seq, f32]`; surf desugars single-letter dims to
`d-var`). The chelis#346 F5 decision stages d-vars as concrete `Lit`s,
which is faithful wherever the same-rank formal/actual remap
concretizes the body's names — but vmap's rank shift skips that remap,
so the body's `Named("a", None)` stayed unbound and the guard panicked
(check clean, backend correct, eval ICE: the #351 symptom exactly). In
the vmap lane only, d-vars are now staged as `Named(name, Some(size))`
so they bind through the placeholder Load exactly like d-names; the
plain-call lane keeps `Lit` staging (pinned by
`dim_var_formal_routes_and_matches_backend`). Pinned eval-vs-backend at
`vmap_over_dim_var_formal_named_reduce_evals_and_matches_backend` (two
surviving d-vars, a leading d-var, a mixed d-var/d-name formal, and
`vmap(grad(...))` with a non-constant gradient) and
`vmap_axis_one_over_rank_poly_named_reduce_evals_and_matches_backend`
(non-zero vmap axis through the same synthesis). Best-effort scope:
closures without declared param types, nested `vmap`, and unreadable
axis literals fall back to the old `Lit`-dim marshalling.

*Named-axis expand, `R+1` (chelis#339, SHIPPED).* The inverse arithmetic
direction: `expand(x, new, size)` inserts a trailing named axis and
`expand(x, new, size, anchor)` inserts immediately before an existing
named anchor (spec §4.5.3). Unification stays unitary — the insertion
point is an end of the row or a position fixed by a named anchor located
uniquely in the operand; insertion strictly inside an opaque spread has
no anchor and stays rejected (the computed output row never places the
new axis there, so a declared result demanding it fails row
unification). `shape_class("expand")` is now `NameTracked`, with the
procedural arm (`check_expand_signature`) as the real gate: positional
insert axes at symbolic rank, absent/ambiguous anchors, and inserted
names that collide with an existing axis are hard errors. The 4-arg
anchored form bypasses the arity-3 HM scheme through the
`infer_expand_app` dispatcher (the `infer_permute_app` pattern).
Lowering resolves the named insertion point against the monomorphized
operand dims (trailing → operand rank; anchored → the anchor's index,
loud error if absent) and stamps the inserted dim as
`Named(name, Some(size))` so later by-name ops in the same body can
find it. The chelis#338 eval routing treats a named-axis expand app
exactly like a named-axis reduction app (site A interception + site B
def-call routing). Acceptance lives in
`crates/chelis-cli/tests/rank_poly_tier3.rs` (insert-at-end /
insert-by-anchor positive on both lanes incl. `grad`/`vmap`, plus the
negative parity suite).

*Variadic named-axis reduction (chelis#339 Part 2, SHIPPED).*
`sum(x, seq, head)` reduces several named axes in one call, equivalent
to the documented composition and order-insensitive. The arity-2
reduction schemes are bypassed by an `infer_reduction_app` dispatcher;
`check_reduction_signature`'s named loop validates every axis
(positional integers, unknown names, ambiguity, and duplicates are hard
errors — duplicates are never silently deduplicated). Defined for the
value reductions (`sum`/`mean`/`max_reduce`/`min_reduce`/`prod_reduce`;
mean-of-means equals the joint mean under uniform weights); the
index-returning `argmax_reduce`/`argmin_reduce` get a targeted
no-variadic-form rejection. Lowering desugars the variadic app to the
nested 2-arg composition (`synth_reduction_app`), so the C backend,
eval routing, `grad`, and `vmap` all ride the existing single-axis
lanes — inheriting their behavior unchanged, including one pre-existing
gap (below). Body-Discipline admission in a `..r` body is unchanged
(`sum`/`mean` only, chelis#340). Acceptance: the `variadic_*` suite in
`crates/chelis-cli/tests/rank_poly_tier3.rs`.

*Known gaps (follow-ups):* `max`/`min`/`prod` reduce in a `..r` body
(chelis#340) remains a follow-up (single-axis and variadic alike).
Leading-end insertion into a row that *begins with a spread*
(`tensor[..rest]` with the new axis first) is not expressible — only
trailing or anchored insertion is. PRE-EXISTING (verified on main at
d786744, untouched by chelis#339): `vmap` over a def chaining TWO named
reduces to a scalar ICEs on the dag.rs symbolic-dim guard when the
vmapped operand is a top-level *binding* (`vmap(f)(y)`; an inline
literal operand and single-stage reduces are fine) — the
chelis#346/#351 annotation-dims family in a lane those fixes did not
cover; the variadic form desugars to that composition and inherits the
gap unchanged. The positional (integer-axis) reduction and expand paths
on concrete operands are unchanged.

## Why this is needed

Today (chelis 0.7.21):

- A tensor type is `Type::Tensor(Vec<Dim>, TensorPrec)` (`crates/chelis-types/src/types.rs:335`).
  Rank is the `Vec` length, fixed at the type level. `Dim` ranges over
  `Name | Var | Lit | Wildcard` — there is **no rank/shape variable**.
- Unification hard-checks rank equality (`unify.rs:361`):
  `tensor rank mismatch: N dims vs M dims`. Two ranks never unify.
- `def`s don't overload — `Env` is `HashMap<String, Scheme>` and `bind`
  overwrites (`env.rs:37`). Two same-name rank-distinct `def`s both parse;
  the second clobbers the first, so the off-rank caller fails unification.

The reduce builtins *look* rank-polymorphic but achieve it outside the type
system: their HM scheme is the loose `?a -> ?b` (`builtins.rs:302`) and the
real shape contract lives in a procedural arm keyed on the builtin name.
User `def`s get none of that machinery. Tier 2 gives the surface language a
principled, type-level way to express the two rank-mapping shapes that are
sound — identity and erasure — without the procedural escape hatch.

## Scope

**In:**

- A rank variable `R` usable in a `def`/`sig` quantifier and in tensor types,
  in exactly two positions:
  - **identity**: `R` appears as the *entire* shape of the input and the
    *entire* shape of the output (`&tensor[R, p] -> tensor[R, p]`), optionally
    with a precision change (`-> tensor[R, bool]` for comparisons/logical).
  - **erasure**: `R` appears as the entire shape of one or more inputs and
    does **not** appear in the output (`&tensor[R, p] -> tensor[p]`, i.e. a
    rank-0 result, or a fully-concrete output shape).
- Call-site instantiation of `R` to a concrete shape vector by unification.
- A body-discipline check: a `def` whose sig mentions `R` may only call ops
  whose shape semantics are **shape-identity** (output shape == input shape) or
  shape-erasing — never shape-rewriting (see §Soundness Boundary).

**Out:**

- Tier 3 rank arithmetic (`R ++ [k]`, `[k] ++ R`, `R` adjacent to concrete
  dims). The surface syntax must *reject* `R` next to other dims so the
  restriction is enforced at parse time, not discovered at unification.
- Enumerated rank overloading (multiple same-name `def`s sharing a name and
  dispatching by arg rank). That is issue#258 "option (a)" and is a distinct,
  lower-value design; this plan replaces N defs with *one* rank-poly def, so
  the duplicate-def question does not arise for the migrated surface.
- Two rank variables in a single shape, or a rank variable inside a nested
  ADT/`List` element (would reopen the §4.5.1 shape-vector-variable deferral).

## Type Representation

Keep `Type::Tensor(Vec<Dim>, TensorPrec)` to avoid churning every IR/backend/
serde site. Add one `Dim` variant:

```rust
pub enum Dim {
    Name(String),
    Var(DimVar),
    Lit(i64),
    Wildcard,
    Rank(RankVar),   // NEW: stands for an entire shape vector
}
```

with a hard structural **invariant**, enforced at construction and in
`chelis-deep/src/validate.rs`: *if a `Dim::Rank` appears in a tensor's dim
list, it is the sole element of that list.* `tensor[R, p]` is
`Type::Tensor(vec![Dim::Rank(r)], p)`. This keeps most dim-iterating code
unchanged; only unification and the monomorphization assertion care.

`Scheme` gains `rvars: Vec<RankVar>` alongside `tvars` / `dvars`
(`types.rs:427`). Erasure output is `Type::Tensor(vec![], p)` (rank-0) or a
concrete shape.

**Monomorphization invariant (parallels `TensorPrec::Var`):** after
instantiation every reachable tensor must be `Dim::Rank`-free. As shipped, this
holds *by construction* at the IR boundary: the IR `DimInfo` (`dag.rs`) has only
`Named` and `Lit` variants — there is no rank dim to represent — so call-site
rank monomorphization (`try_extract_tensor_type_with_subst` in `lower.rs`)
either expands a bound `(d-rank)` slot to the caller's concrete dims or, for the
speculative annotation read on an un-monomorphized slot, drops it (the inlined
body supplies the concrete shape downstream). A standalone rank-poly def, which
has no caller to supply the binding, is skipped from emission entirely in both
the DAG and host lanes. Thus a `Dim::Rank` can never become a backend input;
reaching the backend with one would be a monomorphization bug, exactly as a
surviving `TensorPrec::Var` (`types.rs:236`) would be.

## Unification

New arm in `unify.rs`, checked **before** the rank-equality length test:

- `Tensor([Rank(r)], p1)` vs `Tensor(dims2, p2)`: bind `r := dims2`
  (the whole vector, including a possibly-empty vector for erasure), then
  `unify_tensor_prec(p1, p2)`. Symmetric. Occurs-check `r` against `dims2`.
- `Tensor([Rank(r1)], _)` vs `Tensor([Rank(r2)], _)`: bind `r1 := r2`.

Because `R` only ever stands for the *entire* shape (never adjacent to
concrete dims), unification stays **unitary** — there is exactly one most-
general binding. This is the property Tier 3 would lose. `Subst` gains a
rank-var map; `Subst::apply` substitutes `Dim::Rank(r)` → the bound dim list.

## Surface Syntax

Proposed: a spread marker on a bare quantifier name used as a whole shape.
Candidate forms (decide in commit 1, pin in spec/02 + spec/03):

```
def relu_forward[..r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
def sum_all[..r](x: &tensor[..r, f32]) -> tensor[f32] = ...
```

The `..r` marker is mandatory and may appear **only** as the sole shape
element. `tensor[..r, k, f32]` is a **parse error** ("rank variable `..r` must
be the entire shape; `R` adjacent to concrete dimensions is Tier-3 rank
arithmetic, not supported") — this is where the Tier-2/Tier-3 boundary is
enforced syntactically.

Plumbing touched:

- `chelis-surf/src/ast.rs:252` (`Tensor(Vec<TypeExpr>, String, Span)`) — a
  `TypeExpr` rank-spread variant + parser token.
- Deep representation: a new dim node `(d-rank {} r)` joins the closed dim
  vocabulary (`d-name` / `d-var` / `d-lit`) — this is a **spec/03-deep-syntax.md
  change** (the 61-tag closed vocabulary grows by one) and
  `chelis-deep/src/validate.rs` allowlist + the `t-tensor` child rules.
- Decompiler/printer/formatter round-trip (`surf/decompile.rs:1887`,
  `deep/printer.rs:295`) must emit and re-parse `..r` identically (formatter
  round-trip invariant from CLAUDE.md).

## Soundness Boundary (the heart of Tier 2)

Spec §4.2 forbids implicit broadcasting and relies on named dims to catch
transposition/reshape bugs. A rank variable *erases the per-axis names inside
its span*, so it can only be admitted where no axis within `R` is ever
reordered or matched against a differently-named axis. Identity (`R -> R`) and
erasure (`R -> []`) satisfy that **only if the body cannot smuggle an
axis-reordering op past the opaque `R`.**

The trap, found while scoping: `relu`, `reshape`, and `permute` **all share
the identical `&tv -> tv` HM scheme** (`tensor_unop`, `builtins.rs:197`; bound
at `builtins.rs:653/693/694`). Their real shape behavior is split — `relu` is
shape-identity, but `reshape`/`permute` *rewrite* the shape via procedural
arms. Against an opaque `R`, the procedural arm cannot rewrite an unknown-rank
shape, so a body like

```
def evil[..r](x: &tensor[..r, f32]) -> tensor[..r, f32] = permute(x, [1, 0])
```

would unify `tv := tensor[..r, f32]` and return `tensor[..r, f32]` — passing
the `R -> R` check while actually transposing. With concrete ranks today the
declared `tensor[a,b]` vs actual `tensor[b,a]` mismatch is *caught*; under an
opaque `R` there are no named axes left to mismatch. **That is a real §4.2
soundness break, and the HM scheme cannot distinguish `relu` from `permute`
because both are `&tv -> tv`.**

**Required mitigation — Body Discipline check.** A `def` whose sig mentions a
rank variable may call a builtin only if that builtin is classified
**shape-identity** or **shape-erasing**, by an explicit table keyed on shape
semantics (NOT on the HM scheme). Concretely:

- *shape-identity allowlist:* elementwise unops/binops/triops whose output
  shape provably equals an input shape (`relu`, `sigmoid`, `tanh`, `exp`,
  arithmetic/comparison/logical ops over matching shapes, …).
- *shape-erasing:* full-reduce forms that consume `R` entirely to a rank-0
  result (only valid in an erasure-position body).
- *denylist (reject inside an `R`-body):* every shape-rewriting op —
  `permute`, `reshape`, `expand`, `shrink`, `stride`, `pad`, `concat`,
  `gather`, `scatter`, `matmul`, axis-indexed reductions, `conv2d`, … — and
  any user `def` not itself proven rank-safe.

The check runs during body validation (extends the §4.4 "body must type-check
for all instantiations" rule from *all dims* to *all ranks*). The good news
from the same investigation: pure elementwise pipelines need **no builtin
re-typing** — the elementwise builtins are already `&tv -> tv`, so they
propagate an opaque `R` unchanged. The work is the discipline *check*, not new
signatures.

## Red-Team Findings

A fresh-context local subagent pressure-tested the identity/erasure boundary.
Its central conclusion (there is a real §4.2 body-check soundness hole) is
**correct and main-thread-verified**; its stated *cause* and *severity* were
**wrong and are corrected here** from direct reproduction on
`target/debug/chelis`.

### CRITICAL (verified) — inline return annotation suppresses the body-vs-sig check

When a `def` has **both** (a) parameter types supplied by a separate `sig`
line (bare params) **and** (b) an inline return-type annotation `-> T` on the
def, the body is **not** validated against `T` — the annotation is trusted as
the body's type. Verified across three independent lie classes, each scoring a
clean `1` *with* the inline annotation and `< 1` (correctly rejected) *without*
it:

| Lie | body truly | declared | no inline-return | with inline-return |
|---|---|---|---|---|
| transpose (`permute(x,1,0)`) | `tensor[seq,batch]` | `tensor[batch,seq]` | 0.88 (caught) | **1 (clean — bug)** |
| rank (`sum(x,0)`) | rank-1 | rank-2 | 0.88 (caught) | **1 (clean — bug)** |
| §4.4 rigid (`def f(x,y)=y`) | `tensor[m]` | `tensor[n]` | 0.4 (caught) | **1 (clean — bug)** |

**Correction to the red team:** the trigger is **not** "the sig+bare-param
body is never checked" (the no-inline-return column proves it *is* checked —
all three are caught). The active ingredient is the **inline return
annotation** coexisting with a `sig`; the bug is the *conjunction* (sig-sourced
bare params **and** an inline `-> T`). An inline-param def with the same inline
`-> T` (no sig) is also correctly caught — so it is specifically the
sig-line + inline-return overlap where the body-vs-declared unify is skipped.

**Correction to severity:** the red team called this prevalent in the std
corpus. It is **not active** there: in `packages/chelis-std/src/nn/generate.ch`
the `sig`-bearing defs carry no inline return, and the inline-`->` defs carry no
`sig`; no executable-corpus def hits the conjunction. The hole is **latent**.
But `chelis fmt` **preserves** the redundant inline annotation rather than
stripping it (verified), so the form is reachable and passes the style gate —
filed as standalone soundness bug **chelis#285**, independent of #258.

**Why this gates Tier 2:** the identity position `def relu_forward[..r](x) ->
tensor[..r,p] = ...` is, syntactically, sig-sourced params + an inline `-> T`.
Shipping `R` on top of a body check that this exact form already skips would
ship a position whose body is unverified by construction. Tier 2 must not start
until either (i) the body-vs-declared unify runs on the sig + inline-return
form, or (ii) `R` routes through a dedicated check that does not trust the
inline annotation.

### HIGH (reasoned, consistent with scoping) — rank-preserving reorder defeats `R -> R`

`infer_permute_app` returns a fresh `Type::Tensor(reordered, prec)`
(`infer.rs:11109`) never tied back to the input's dim vars. `permute`/same-rank
`reshape` are rank-preserving, so under a *working* `R -> R` body check they
still unify (rank matches; reordered dims are fresh) — `R` carries no per-axis
identity within its span to catch the reorder. This is the §4.2 leak already
captured in §Soundness Boundary; the red team confirms the mechanism. Closes
with the Body-Discipline allowlist (Commit 4): only elementwise/structure-
preserving ops inside an `R`-span-preserving body; `permute`/`reshape`/`expand`/
`pad`/`concat`/`gather`/`shrink`/`stride` forbidden.

### HIGH (reasoned) — `R` must be the ordered *named* dim vector, not a count

Verified baseline: `add(x: tensor[batch,seq], y: tensor[seq,batch])` is rejected
today (`dimension mismatch: Name("batch") vs Name("seq")`) because dims carry
names and unify element-wise (§4.1). If `R` denoted a *rank/count*, axes within
its span become wildcards and the transposition blindness §4.2 forbids returns
(multi-arg `add2[..r](x,y)` would accept mismatched named axes — implicit
broadcast). Resolution: `R` denotes the ordered named-dim vector and must appear
identically in every tensor position of the sig. Already reflected in
§Scope/§Test Strategy; promote to an explicit representation note.

### MED (reasoned) — §4.5.1 kind-identity and expressibility fence

`R` *is* the shape-vector variable §4.5.1 deliberately deferred; "identity/
erasure only" narrows *where* it appears but not its *kind*, so the feature must
be adjudicated against §4.5.1's rationale, not asserted past it. The erasure
half (order-invariant all-reduce) is the safer one; the identity half is exactly
where §4.5.1's "mask transposition bugs" rationale bites — hence the
Body-Discipline gate. Plus: `R` must be barred syntactically from list-element
position (else `List[tensor[..r]]` defeats the rank-uniform-list guarantee) and
from adjacency to concrete dims (`tensor[..r, k]` → Tier-3 non-unitary
unification). Both already in §Scope; the fence is a parse-time rule, not a
checker afterthought.

### LOW (reasoned) — erasure restricted to order-invariant all-reduce

`tensor[..r,p] -> scalar` is sound for `sum`/`mean`/`max` (order-invariant), and
no order-*dependent* full-reduce primitive exists today, so this is a latent
ergonomic gap, not a break. Restrict the erasure tier to the named all-reduce
primitives rather than any body returning a constant shape.

### Coverage / unvalidated

Validated on the running compiler: the CRITICAL bypass (3 lie classes ×
with/without inline-return), `permute` fresh-tensor behavior, names-carry-
identity baseline, reduce name-erasure, and corpus non-use of the conjunction.
Reasoned-only (feature unimplemented): the mapping from baselines to proposed
`R`-rule accept/reject, the §4.5.1 kind argument, and the `tensor[..r,k]`
non-unitary-unification leak. A full audit of `gather`/`scatter`/`shrink`/
`stride` rank behavior under an `R`-body is deferred to Commit 4.

## Work Breakdown

```
Commit 1: Surface syntax + Deep `(d-rank {} r)` node + validate.rs vocabulary
          + spec/02 & spec/03 updates. Parser rejects R-adjacent-to-dims.
Commit 2: Type repr — Dim::Rank, Scheme.rvars, Subst rank map, Subst::apply.
Commit 3: Unification arm (pre-length-check) + occurs-check. Monomorphization
          assertion (Dim::Rank-free) in IR/backend lowering.
Commit 4: Body Discipline check — shape-semantics classification table +
          validation pass that rejects shape-rewriting ops under an R-sig.
Commit 5: Decompiler/printer/formatter round-trip for `..r`.
Commit 6: Migrate School's relu/silu/gelu family + full-reduce to single
          rank-poly defs (downstream proof); positive+negative test corpus.
```

## Test Strategy (positive + negative parity, per CLAUDE.md)

Every "works" test has its paired "fails with the right reason" test.

- [ ] `relu_forward[..r]` checks and is callable at rank 1, 2, 3, 4 with the
      same def (positive) — and the rank-2 caller no longer emits
      `tensor rank mismatch` (the chelis#258 repro turns green).
- [ ] erasure: `sum_all[..r](x) -> tensor[f32]` checks and returns rank-0 at
      every input rank.
- [ ] **negative:** `def evil[..r](x) -> tensor[..r,f32] = permute(x,[1,0])`
      is REJECTED by Body Discipline with a §4.2-citing diagnostic.
- [ ] **negative:** every denylisted op (`reshape`/`expand`/`matmul`/…) in an
      `R`-body is rejected with the op named.
- [ ] **negative:** `tensor[..r, k, f32]` is a parse error (Tier-3 boundary).
- [ ] **negative:** two rank vars in one shape, or `..r` inside a `List`
      element, rejected.
- [x] identity `R` shared across two args forces same *shape* (not just same
      rank): `add2[..r](x,y)` rejects mismatched concrete shapes — no implicit
      broadcast (§4.2). Positive: matching shapes accepted. Covered in
      `chelis-cli` (`rank_poly_tier2`): positive checks
      (`multi_arg_rank_poly_def_callable_at_ranks_1_through_4`) and a backend
      build+run with eval-vs-backend agreement
      (`multi_arg_rank_poly_builds_and_runs_at_ranks_1_through_4`); negatives
      reject a rank mismatch (`multi_arg_rank_poly_def_rank_mismatch_rejected`)
      and an equal-rank dim mismatch (`multi_arg_rank_poly_def_dim_mismatch_rejected`).
- [x] monomorphization (SHIPPED): call-site rank substitution
      (`tensor_rank_substitutions` + `LowerCtx.rank_substitutions`) resolves
      every `(d-rank)` slot of an inlined rank-poly body to the caller's
      concrete shape. No `Dim::Rank` reaches a backend by construction (the IR
      `DimInfo` has no rank variant). A rank-poly `relu_forward` called at ranks
      1–4 builds, links, runs, and matches the evaluator value-for-value; a
      standalone rank-poly def with no concrete caller is skipped from emission
      rather than miscompiled. Unit + CLI coverage in `chelis-ir`
      (`rank_poly_def_lowers_through_concrete_caller` and the
      `tensor_rank_substitutions` / `extract_rank_var_name` /
      `try_extract_tensor_type_with_subst` helper tests) and `chelis-cli`
      (`rank_poly_tier2`: `rank_poly_identity_builds_and_runs_at_ranks_1_through_4`,
      `rank_poly_composed_identity_builds_and_runs`).
- [x] decompile/format round-trip: `..r` sigs survive `chelis fmt` and
      re-parse byte-identical on the corpus. Covered in `chelis-cli`
      (`rank_poly_tier2`): `rank_poly_def_survives_fmt_round_trip` and
      `multi_arg_rank_poly_def_survives_fmt_round_trip` assert `..r` is preserved
      in every tensor position, `chelis fmt` is idempotent (byte-identical second
      pass), and the formatted text re-checks clean.
- [ ] `chelis check --json` shape/semantic invariants on the R-corpus.

## Acceptance Oracle (one per phase, per CLAUDE.md)

**Oracle:** a new nextest group `rank_poly_tier2` in `chelis-types` +
`chelis-cli` that runs the positive/negative corpus above, plus the migrated
School relu/full-reduce defs checking clean under
`cargo run -p chelis-cli --bin chelis -- check`. Phase is done when that group
is green, the chelis#258 repro returns an empty error vector, the formatter
round-trip holds, and the red team finds only minor residual issues. No part
of the oracle is manual; it joins the default `scripts/gate.py` run.

## Risks / Open Questions

- **Body Discipline completeness.** The classification table must cover every
  builtin; a missed shape-rewriting op = a silent §4.2 hole. Mitigation: drive
  the table from a single source and assert exhaustiveness (every builtin is
  classified identity/erasing/rewriting, closed-set test like the `Prim`
  enumeration lock in `types.rs:484`).
- **§4.5.1 consistency.** Tier 2 must not let `List[tensor[..r, f32]]` become
  expressible (would reopen the deferred shape-vector variable). Enforced by
  "no `R` inside ADT/List elements" (scope-out + negative test).
- **Surface syntax bikeshed** (`..r` vs `*r` vs `r...`). Decide in commit 1.
- **Interaction with the procedural reduce arms.** An erasure-position `R`
  body that calls `sum`/`mean` must agree with `check_reduction_signature`;
  confirm the arm tolerates an opaque-rank input (it currently returns
  `subst.apply(result_ty)` when the input type is still a `Var` — `infer.rs:13025`).

## Sequencing

**Prerequisite (hard):** fix the inline-annotation body-check bypass
([chelis#285](https://github.com/Chelis-Lang/chelis/issues/285), §Red-Team
Findings CRITICAL). Tier 2 is unsound to build until a `def` carrying both a
`sig` and inline annotations is checked against the most-concrete
reconciliation of the two, not the wildcard-filled synthesized defsig. This is
independent of #258 and ships first regardless of whether Tier 2 proceeds.

Otherwise independent of backend work; lives entirely in front-end
(`chelis-surf`/`chelis-deep`/`chelis-types`) plus a monomorphization assertion
at the IR boundary. Recommend pairing with, or immediately after, shipping
chelis#258 "option (b)" (reject duplicate `def` with a clear diagnostic) as a
stopgap — that stopgap is forward-compatible because a rank-poly def is a
single def and never trips the duplicate-def guard.
