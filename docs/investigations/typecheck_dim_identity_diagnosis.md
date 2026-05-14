# TypeCheck dim-identity soundness gap (SR-LEAK-A / TypeCheck-FreeDimVarUnification-F1)

Status: DIAGNOSED + FIXED. The discovery-contract escalation (scope
exceeds the originally-pinned fix layer) was approved by the orchestrator
as Option 1: fix both paths in this PR. Both leak paths are closed in
PR `fix/typecheck-dim-identity`. See "Fix" below.

## Summary

A function whose declared return type and body type differ in dimension
*identity* but match in dimension *rank* type-checks with `score=1` and
then evaluates with a runtime shape mismatch. This is a HIGH severity
silent miscompilation.

Reproducer (red-team `docs/investigations/terminal_redteam_0_7_9.md`,
finding SR-LEAK-A):

```
def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = y
```

Passes `chelis check` with `score=1` today. It must fail with a
TypeMismatch: the body has type `tensor[m, f32]` (after borrow->owned
coercion) but the declared return is `tensor[n, f32]`, and `n` and `m`
are distinct, user-named dimension parameters of the def.

## Two distinct leak paths

The bug surfaces through two independent code paths. A controlled
experiment (stubbing `types_structurally_equal` to always return `false`)
proves they are independent:

### Path A: Shape A relaxed-retry (routes through `types_structurally_equal`)

`crates/chelis-types/src/infer.rs`, the relaxed-retry block around line
5085 calls `shape_a_relaxed_return`, which guards the relaxation with
`types_structurally_equal` (line 8506). That helper compares tensor
*rank* only:

```rust
(Type::Tensor(d1, p1), Type::Tensor(d2, p2)) => p1 == p2 && d1.len() == d2.len(),
```

`d1.len() == d2.len()` checks dim count, not dim identity. So a body
returning `&tensor[m, f32]` against a declared `tensor[n, f32]` passes
the structural guard, the relaxed unify then binds the free dim var, and
the program type-checks.

This path covers:
- PR #91 (0.7.8 W4-A) bare-var return (`shape_a_relaxed_return` original).
- PR #109 (0.7.9 Workstream SR) `let`/`if`/`match` tail-position bodies
  via `descend_to_tail_var`.

Both route through the same broken `types_structurally_equal` guard. PR
#91 and PR #109 are not themselves buggy; they just route more body
shapes through the broken equality check.

Experiment: with `types_structurally_equal` stubbed to `false`, the §0
reproducer correctly surfaces TypeMismatch. So the orchestrator-pinned
fix layer (tighten `types_structurally_equal` to check dim identity)
**does close Path A.**

### Path B: plain owned-tensor body (bypasses `types_structurally_equal` entirely)

```
def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y
```

This passes `chelis check` with `score=1` today, AND still passes with
`types_structurally_equal` stubbed to `false`. It never reaches the
relaxed-retry path because the *initial* `unify(&body_ty, &decl_ty)`
already succeeds.

Root cause: `unify_dim` in `crates/chelis-types/src/unify.rs` (line 425).
Two distinct dimension variables unify freely:

```rust
(Dim::Var(v), _) => bind_dvar(*v, &d2, subst),
(_, Dim::Var(v)) => bind_dvar(*v, &d1, subst),
```

`unify_dim(Dim::Var(n), Dim::Var(m))` binds `n := m` and succeeds. This
is correct for *call-site instantiation* (spec/04-type-system.md §4.5:
dim variables are instantiated by unification at call sites). It is
wrong for *def-body validation against a declared signature*: the
declared dim parameters `[n, m]` are universally quantified, so the body
must be valid for *all* instantiations. Treating them as free unifiable
variables collapses `n` and `m` into one equivalence class and accepts a
body that is only valid when `n == m`.

The existing `declared_dvars` self-pin check in `infer_fn` /
`infer_def_body_with_sig` (around lines 9794 and 9926) already encodes
the "declared dim parameters must remain polymorphic" principle, but it
only catches `Dim::Var -> Dim::Lit` (a dim var forced to a concrete
literal). It does NOT catch `Dim::Var -> Dim::Var` (two declared dim
params unified with each other), which is Path B.

## Spec grounding

spec/04-type-system.md §4.1 says `(d-var a)` "unifies with any dimension
-- binds `a` to that dimension." §4.5 frames dim-var unification as the
call-site instantiation mechanism. The spec does **not** explicitly
state the dual requirement: that within a def body checked against a
declared signature, the declared dim parameters must be treated as
rigid/skolem (distinct, non-unifiable with each other). This is a spec
ambiguity. It should be made explicit in §4.4 / §4.5 rather than papered
over. Flagged here, not edited, per the agent contract.

## Call-site inventory for `types_structurally_equal`

Only one call site: `shape_a_relaxed_return` (line 8506). It is used
only in the relaxed-retry context whose sole intended relaxation is
borrow->owned at the type's outer layer; the dim structure underneath
must match exactly. No call site legitimately wants rank-only equality.

## Sibling sweep

- `narrow_wildcards_with` (infer.rs line 4671): uses `.len()` rank gates
  but only *narrows* `Dim::Wildcard -> Dim::Lit`, preserving all other
  dims. It is not an equality predicate and not a soundness gate. Not a
  sibling instance.
- `type_expr_eq` (linearity.rs line 1530): operates on syntactic `Expr`,
  not `Type`. Not a sibling instance.
- `types_structurally_equal` is the only structural type-equality
  *soundness gate* with the rank-only shortcut.

## Escalation (resolved)

The orchestrator (§0.1) originally pinned the fix at the
`types_structurally_equal` layer. That fix closes Path A but leaves Path
B (a sibling HIGH severity hole) open. Path B is the same finding
(`TypeCheck-FreeDimVarUnification-F1`) and the red-team note for
SR-LEAK-A explicitly called it out: "the same dim-var-unification leak
fires without Shape A ... so the root cause is in HM inference, not just
`types_structurally_equal`."

Per the §4.3 discovery contract and
`feedback_escalate_structural_blockers.md`, the agent halted for an
orchestrator decision because closing Path B touches def-body validation
(a public type-system behavior), not just the surgical helper.

The orchestrator approved Option 1: fix both paths in this PR, with Path
B fixed via the surgical approach (extend the existing `declared_dvars`
post-body check), not the full skolemization rewrite.

## Fix

### Path A: `types_structurally_equal` checks dim identity

`types_structurally_equal` (infer.rs) now compares tensor dimensions for
*identity*, not just count. Two dim lists are structurally equal iff
they have the same length AND each position holds identical dims, where
two dims are identical iff:

- both are the same concrete `Dim::Name` (equal names), or
- both are the same `Dim::Lit` (equal values), or
- both are `Dim::Var` referring to the same `DimVar`.

`Dim::Wildcard` matches any dim on either side (it is the permissive
"unknown" sentinel, consistent with `unify_dim`'s treatment). Two
distinct symbolic dim variables (`n` vs `m`) are NOT identical even
though both contribute rank 1, so the Shape A relaxed-retry no longer
accepts a body whose return dim diverges from the declared return dim.
The relaxation's only intended slack -- a top-level `Ref` wrapper
difference -- is unaffected: `Ref` is still unwrapped recursively, and
the dim structure underneath must now match exactly.

### Path B: `declared_dvars` post-body rigidity check

The post-body `declared_dvars` loop in both `infer_fn` and
`infer_def_body_with_sig` previously only flagged `Dim::Var -> Dim::Lit`
(a declared dim param forced to a concrete literal). It now also flags
`Dim::Var -> Dim::Var` collapse: if two *distinct* declared dim params
resolve (via `subst.apply_dim`) to the *same* dimension after body
inference, the body unified two universally-quantified dim parameters
that must stay distinct. This is a `DimensionMismatch`. The shared
detection logic lives in a single helper, `check_declared_dvars_rigid`,
called from both inference entry points.

The invariant: two distinct declared dim parameters must not resolve to
the same dimension after body inference. A single declared dim param
appearing in multiple param positions (`def h[n](x: tensor[n], y:
tensor[n])`) is one dvar and never trips the check.

## Spec completion

`spec/04-type-system.md` §4.4 now states the dim-parameter rigidity rule
explicitly: declared dimension parameters are rigid within the def body,
and two distinct declared dim parameters do not unify with each other
during body validation. Call-site instantiation (§4.5) is unchanged --
that is where dim variables are genuinely instantiated by unification.
This is spec completion stating the correct intended contract that the
fix enforces, not editing the spec to match a shortcut.
