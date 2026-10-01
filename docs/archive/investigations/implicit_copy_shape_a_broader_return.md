# Implicit-copy Shape A broader return: tail-position coercion

Tracks the diagnosis of `Linearity-ShapeABroadReturn-F1` from
`docs/archive/reports/gap_synthesis.md`, the §5 follow-on filed against PR #91's
diagnosis note (`implicit_copy_fanout_v3_diagnosis.md`, "Sibling
sweep findings"). Fixtures live in
`crates/chelis-ir/tests/implicit_copy_shape_a_broader_return.rs`.

## Failure mode

PR #91 introduced `shape_a_relaxed_return` in
`crates/chelis-types/src/infer.rs`. The helper accepts a body whose
inferred return is `Ref(R)` and whose declared return is owned `R`,
provided the body is a bare `(fn (params...) (var x))`. The gate at
the inner-expression `get_tag(...) == Some("var")` check rejects any
other body shape. Three common Surf surfaces hit this gate:

```surf
# let-tail (block-let): desugars to (let bind body) chain.
def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = {
  y = x
  y
}

# if-tail: both branches are the same borrow-ref.
def g[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] =
  if c then x else x

# match-tail: every arm body is the same borrow-ref.
def h[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] =
  match c with {
    | Left => x
    | Right => x
  }
```

All three desugar to a body whose tail expression is a `(var x)`
reference once the structural wrappers (`(let bind body)`,
`(if cond then_e else_e)`, `(match scrutinee arm...)`) are
stripped. Today every one of them reports

```
def 'f' body doesn't match declared signature:
  body has type `(&tensor[n, f32]) -> &tensor[n, f32]`,
  declared type is `(&tensor[n, f32]) -> tensor[n, f32]`
```

because the relaxed-retry never fires on the wrapped body.

## Chosen fix

Add a private helper next to `shape_a_relaxed_return`:

```rust
fn descend_to_tail_var(expr: &deep::Expr) -> Option<&str>
```

> Since superseded. chelis#2198 renamed this helper to
> `descend_to_tail_parameter`, moved it to
> `crates/chelis-types/src/infer/app_helpers.rs`, and changed what it
> compares: sibling branches must agree on one borrowed parameter's binding
> identity, not on a repeated source name. The rest of this section records
> the PR #109 design as it was written.

The walker returns `Some(name)` when the expression is a tail-position
var-ref (after descending through `let`, `if`, and `match`), `None`
otherwise. Rules:

- `(var x)` returns `Some("x")`. This is the leaf case and matches
  the PR #91 bare-var shape.
- `(let bind body)` recurses into `body` (the third element).
- `(if cond then_e else_e)` recurses into both branches; both must
  resolve to `Some(_)` and the names must match.
- `(match scrutinee arm ...)` recurses into every arm body
  (`(arm pattern guard body)`'s third element); all arms must
  resolve to the same `Some(name)`.
- Otherwise return `None`.

The descent is type-agnostic. The surrounding logic in
`shape_a_relaxed_return` already verifies that the inferred body
type is `Fn(params, Ref(R))` and the declared type is `Fn(params, R)`
with `R` structurally equal; the descent only needs to confirm the
body has the bare-var tail-position shape the existing relaxation
treats as safe. Branches that disagree on the named variable (e.g.
`if c then x else y`) fall back to `None` so the existing
TypeMismatch still surfaces unchanged.

The descent enforces "same name across siblings" rather than just
"any var" because the existing relaxation is justified by: "the
caller's borrow lifetime already covers the parameter being
returned." When all sibling branches return the same parameter, the
lifetime extension is the same one PR #91 codifies. Allowing
heterogeneous bare-vars would extend the relaxation beyond what the
bare-var v3 fix scopes, which is out of v3 (and 0.7.9) scope.

### Integration point

`shape_a_relaxed_return` currently dispatches on
`get_tag(inner_list) == Some("var")`. Replace that check with
`descend_to_tail_var(inner).is_some()` (today
`descend_to_tail_parameter(inner, &parameter_roots)`). The rest of the helper
(structural type equality between body's unwrapped return and the
declared return; structural relaxed-type construction) is unchanged.

The caller at `crates/chelis-types/src/infer.rs::check_top_level`
(around the def-body unify) is unchanged. The relaxed-retry
machinery already exists; only the gate on the body shape needs to
broaden.

## What the descent does NOT do

- It does NOT descend through `app`, `tuple`, `cast`, or other
  expression forms. The relaxation is only safe for bare-var tail
  references whose lifetime the caller already arranged.
- It does NOT support heterogeneous return shapes (e.g. one branch
  is `(var x)`, another is `(app f x)`). Those need a richer
  coercion story that the broader implicit-copy work would have to
  design.
- It does NOT touch the catch-all `TypeMismatch` emission. When the
  descent returns `None` or the structural type-equality check
  fails, the existing diagnostic stands.

## Sibling sweep

The `check_top_level` def-body unify is the only place
`shape_a_relaxed_return` runs. Other coercion checks in the type
checker that have a similar "bare-var body only" limitation:

- `auto_borrow_call_arg_types` (`crates/chelis-types/src/infer.rs`,
  the dual at the argument position) is invoked on a
  per-argument-expression basis. It does not have the
  "body-shape-restricted" gate that Shape A has; it coerces every
  argument expression whose type is owned `T` against a parameter
  expecting `Ref(T)` regardless of the expression's structure. The
  asymmetry is intentional: the argument-side coercion only needs
  the expression's *type*, not its structure, because the borrow it
  produces lives only for the duration of the call. The return-side
  coercion is structure-restricted because the borrow's lifetime has
  to be one the caller already provided (i.e. through a `&T`
  parameter being returned through the body), which structural
  descent confirms.

No other coercion helpers were found that share Shape A's
body-shape gate. The descent is local to `shape_a_relaxed_return`.

## Out of scope

- Bodies that return a borrow constructed inside the function (e.g.
  `let y = &local in y`) — the borrow's lifetime is the function's,
  not the caller's, and the relaxation would be unsound.
- Bodies whose tail is an `app` returning a borrow (e.g.
  `def f(x: &T) -> T = some_helper(x)` where `some_helper` returns
  `&T`) — same lifetime issue and the diagnosis cite would surface
  on the helper's signature, not here.
- Heterogeneous branches that return different variables — the
  coercion would have to track each branch's lifetime independently,
  which the existing relaxation does not.

These are not closed by 0.7.9 and remain candidates for any future
broader-coercion work.
