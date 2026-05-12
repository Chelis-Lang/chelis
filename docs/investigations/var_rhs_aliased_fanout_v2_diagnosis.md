# var-RHS aliased fan-out V2 diagnosis (V2-F4)

Red-team v2 finding (PR #58), follow-on to Item 1 (PR #29).
Branch: `fix/var-rhs-aliased-fanout-v2`.
Fixtures: `crates/chelis-ir/tests/cross_statement_fanout.rs`.

## Reproducer

```chelis
x = to_tensor([1.5, 2.7, -0.3])
y = x
a = mul(y, to_tensor([2.0, 2.0, 2.0]))
b = mul(x, to_tensor([3.0, 3.0, 3.0]))
```

Note the absence of a `module` declaration. With
`--allow-style-violations` (so the style gate does not block on the
missing module decl), `chelis check` emits

```
{
  "score": 0.8,
  ...
  "errors": [{
    "kind": "UseAfterConsume",
    "message": "variable `x` was already consumed by use at offset 0; later use at offset 0 is invalid",
    "severity": 0.9
  }]
}
```

The same statements wrapped in `module Repro.V2F4 { ... }` accept
cleanly with score 1.0. The function-body equivalent
(`def f(x) = { y = x; mul(y, x); mul(x, x); ... }`) also accepts.
The bug is specific to **top-level statements outside a `module`**.

## Why PR #29's fix does not cover this

PR #29 patched `Checker::read_or_error` at
`crates/chelis-types/src/linearity.rs:621-661` so that
already-consumed names whose consume description starts with
`"binding "` are silently allowed through subsequent borrow reads.
That covers the `check_let` path
(`crates/chelis-types/src/linearity.rs:382-417`) which builds its
consume site as

```rust
ConsumeSite {
    description: format!("binding `{name}` at offset {}", value.span().offset),
}
```

The V2-F4 case routes through a different path:

1. `check_linearity` (`linearity.rs:118-144`) walks the program's
   `annotated_exprs`. For a source with no `module` wrapper each
   top-level statement is its own `(def {} name body)` node. The
   pre-declare loop (lines 126-133) iterates these defs and
   declares each name in scope with its type.
2. `check_top_level` (lines 246-263) is called for each def. For
   `(def y (var x))` the body is `(var x)`. The method's
   `!(is_var_expr(body) && var_name(body) == Some(name))` guard
   only skips the `def y = y` self-reference shape, so for
   `def y = x` it falls through and calls `check_expr(body, scope)`.
3. `check_expr` (lines 265-294) dispatches on the tag and at line
   275 calls `self.consume_var_expr(expr, scope, generic_site(expr))`.
4. `generic_site` (lines 1125-1129) yields

   ```rust
   ConsumeSite {
       description: format!("use at offset {}", expr.span().offset),
   }
   ```

   Note this description does NOT start with `"binding "`.
5. `consume_var_expr` transitions `x` from `Live` to
   `Consumed("use at offset 0")`.
6. The next top-level def `(def a (app mul (var y) (app to_tensor ...)))`
   triggers `check_app`. `mul` is in `builtin_arg_is_borrowed`
   (line 953 onward), so each var argument routes through
   `read_var_expr` -> `read_or_error`.
7. For the third top-level def `(def b (app mul (var x) ...))`,
   `read_or_error("x", ...)` reads `scope.top("x") =
   Some(BindingState::Consumed(ConsumeSite { description: "use at
   offset 0" }))`. The PR #29 tolerance (line 644)

   ```rust
   if site.description.starts_with("binding ") { return; }
   ```

   does not fire — the description prefix is `"use "` — and the
   helper falls through and pushes `UseAfterConsume`.

The two paths differ only in how the aliasing consume is described.
The `check_let` path knows the consume is a binding (it builds the
description as `"binding `{name}` at offset N"`). The
`check_top_level` -> `check_expr -> consume_var_expr` path uses
`generic_site`, which is the generic "bare variable use" description
intended for things like `def y = x` written at the top-level
without realizing this is morally a binding consume.

## Why the module-wrapped case accepts

With `module Repro.V2F4`, the desugared `annotated_exprs` contains a
single `(module {} name (def x ...) (def y (var x)) ...)` expression.
The pre-declare loop at `linearity.rs:126-133` filters on
`get_tag(list) == Some("def")` and the outer wrapper is `module`, so
**no names get pre-declared**. When the main loop reaches
`check_top_level(module)` it falls through to `check_expr`, which has
no `module` arm and so descends via the catch-all `_` arm into the
module's children. For each child `(def y (var x))` the `def` tag
also has no `check_expr` arm, so it descends into the def's
children, which includes `(var x)`. `consume_var_expr` is called
for `x`, but `scope.top("x")` returns `None` (never declared), and
the function returns silently without consuming anything.

The module case "accepts" only because the linearity checker never
actually checks anything about it. Top-level statements inside a
module wrapper currently skip the linearity check entirely. That is
a separate latent gap (no `UseAfterConsume` for `realize(x);
realize(x)` at module-top-level either; verified empirically) and is
out of scope for V2-F4.

## DAG-level behavior after the fix

Top-level `y = x` lowers to a `Load { name: "x" }` node that the
`y` binding points to (the same NodeId `x` resolves to). Subsequent
`mul(y, ...)` and `mul(x, ...)` both reach the same `Load(x)` node
via `lower_var`'s cached binding map. `mul` does not consume its
inputs at the DAG level — `op_consumes_inputs` returns true only
for `Realize | Drop | Store` (`crates/chelis-ir/src/lower.rs:497`)
— so `insert_copy_nodes_for_consuming_fanout`
(`crates/chelis-ir/src/lower.rs:434`) inserts zero Copy nodes for
this program. The structural-share invariant matches the
function-body sibling fixture
(`var_rhs_let_alias_then_borrow_use_lowers_without_explicit_copy`).

Forward + AD parity vs the explicit-`copy(x)` rewrite holds because
the explicit version differs only by an inert `RiscOp::Copy` node
whose adjoint is the identity.

## Proposed fix

Mirror the `check_let` path inside `check_top_level`. When the
def body is a `(var)` expression and the body's type is owned-linear,
call `consume_var_expr` with a `"binding `{name}` at offset N"`
description directly, instead of delegating to
`check_expr -> consume_var_expr(generic_site)`. The discrimination
matches `check_let`'s existing pattern at
`crates/chelis-types/src/linearity.rs:397-407`.

The fix is local to `check_top_level` (one new branch) and reuses
the existing `read_or_error` tolerance. No `LinearityInfo` schema
change is required. The structural-vs-aliasing discrimination
remains string-based, consistent with the existing
`consume_var_expr` / `read_or_error` pattern; a typed `ConsumeKind`
refactor is a candidate §5 follow-up (already flagged in
`var_rhs_let_fanout_diagnosis.md`).

## Sibling sweep

| Sibling                                          | Status                                                                                                                                                                                                                                                                          |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Top-level `y = x; mul(y, x); mul(x, x)` (no module) | Affected (V2-F4). Direct repro of the finding. Fixed.                                                                                                                                                                                                                            |
| Top-level `y = x; z = y; mul(z, x); mul(x, x)` (no module) | Affected. The `z = y` def chains the same path: `(def z (var y))` triggers `consume_var_expr(y, generic_site)`. After the fix the chain consumes both `x` and `y` via the binding path, and downstream borrow reads route through the PR #29 tolerance. Locked via the V2-F4 fixture's structural shape (mul-mul fan-out). |
| Function-body sibling (`def f(x) = { y = x; mul(y, x); mul(x, x); ... }`) | Already passes today. `check_let` builds a `"binding ..."` consume site, PR #29's tolerance applies. Fixture `fn_body_cross_statement_var_rhs_aliased_fan_out_lowers_without_explicit_copy` pins the behavior.                                                                                                                                                            |
| Top-level inside `module M { y = x; ... }`        | Linearity check silently skipped (pre-declare loop matches only top-level `def`, not module-wrapped). Separate latent gap; not affected by V2-F4 because the bug never fires. Out of scope; recommended §5 follow-up.                                                                                                                                                            |
| Top-level `y = realize(x); mul(x, x)` (no module) | Affected by the broader question of whether borrow reads after non-structural consumes should be allowed. The current code rejects this and the existing test `detects_use_after_consume` in `crates/chelis-types/tests/linearity.rs:21` locks the rejection. Per the spec the DAG-level Copy insertion could handle it, but V2-F4 specifically targets aliasing var-RHS bindings — not realize-then-borrow. Out of scope; flagged as a latent question. |
| Closure / match scrutinee capture of aliased name | Still rejected after the fix. The `closure capture` and `match scrutinee` sites do not start with `"binding "`, so the tolerance does not fire and the irreversible-structural rejection arm in `consume_var_expr` still applies.                                                                                                                                                            |
| Tuple/record destructure (`(a, b) = p; ...`)      | Not affected. Per the Linearity-F2 finding, surf desugars destructuring through `__chelis_tmp` intermediates whose `(var __chelis_tmp_N)` carries no type metadata. `expr_is_owned_linear` returns false, so `consume_var_expr` returns before declaring any binding-consume. Independent gap; out of scope. |

The V2-F4 fix narrowly addresses the top-level var-RHS aliased
fan-out shape. The latent module-top-level-statement gap and the
broader realize-then-borrow question are flagged for a future §5
follow-up and not addressed here.

## Out-of-scope / escalations

1. **`ConsumeKind` schema refactor**: same as the predecessor
   diagnosis. Description-string discrimination remains the
   pattern. Candidate §5 follow-up.
2. **Module-wrapped top-level statements skip linearity entirely**:
   `check_linearity`'s pre-declare loop ignores module-wrapped defs
   and `check_expr` has no `module` arm, so top-level statements
   inside `module M { ... }` are never linearity-checked. The
   V2-F4 fix does not change this behavior because (a) the
   visible-symptom shape is "check passes" not "check rejects valid
   code" and (b) fixing this requires a separate scope-shape
   decision (do module-top-level statements share a scope, or each
   sit in their own scope?). Recommended §5 follow-up.
3. **Realize-then-borrow at top level**: the current rejection is
   locked by `detects_use_after_consume`; spec analysis suggests
   Copy insertion could handle this safely at the DAG level, but
   changing it would touch the existing test surface and warrants
   its own dispatch with explicit spec alignment review. Not
   addressed here.

None of these escalations is required to fix V2-F4.
