# Rank Polymorphism (Tier 2: identity + erasure rank variables)

**Status:** DRAFT phase plan. Scoping only — not yet scheduled. A
fresh-context red-team pass has run; its verified findings (and one
correction to its central claim) are in §"Red-Team Findings". **Tier 2 has a
hard prerequisite: the inline-annotation body-check bypass
([chelis#285](https://github.com/Chelis-Lang/chelis/issues/285), §Red-Team
Findings CRITICAL) must be fixed first.**

**Owning issue:** chelis#258 (same-name `def` overloads with rank-distinct
sigs accepted at type-check time but call site doesn't dispatch by arg rank).

**Goal:** Let a single `def` be generic over tensor *rank* in the two
positions that are sound under spec §4.2, so that the relu/silu/gelu
activation family and the full-reduce primitives stop proliferating per-rank
verb names (`relu_forward` / `_2d` / `_3d` / `_4d`).

This is **Tier 2** of the three-tier analysis on chelis#258:

| Tier | Output-rank function | Example sig | Status |
|---|---|---|---|
| 1 | identity (`out = R`) | `&tensor[R, f32] -> tensor[R, f32]` | **this plan** |
| 2 | constant / erasure (`out = []`) | `&tensor[R, f32] -> tensor[f32]` | **this plan** |
| 3 | arithmetic (`out = R±1`, permute) | `&tensor[R ++ [k], f32] -> tensor[R, f32]` | **out of scope** |

Tier 3 (rank arithmetic, `R ++ [k]` concatenation forms) is explicitly
deferred: it needs non-unitary sequence unification and reopens the §4.2
transposition-safety tradeoff. The reduce/expand primitives that *do* change
rank are already covered by procedural inference arms
(`check_reduction_signature`, `infer.rs:13010`), so users compose them rather
than authoring rank-arithmetic defs themselves.

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
instantiation every reachable tensor must be `Dim::Rank`-free. Backends and IR
builders assert this at lowering, exactly as they already assert
`TensorPrec::Concrete(_)` (`types.rs:236`). A surviving `Dim::Rank` is a
monomorphization bug, not a backend input.

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
- [ ] identity `R` shared across two args forces same *shape* (not just same
      rank): `add2[..r](x,y)` rejects mismatched concrete shapes — no implicit
      broadcast (§4.2). Positive: matching shapes accepted.
- [ ] monomorphization: no `Dim::Rank` survives to any backend; lowering
      asserts. Negative: a synthetic unmonomorphized tensor trips the assert.
- [ ] decompile/format round-trip: `..r` sigs survive `chelis fmt` and
      re-parse byte-identical on the corpus.
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
