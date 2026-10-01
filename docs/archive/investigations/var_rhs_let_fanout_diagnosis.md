# var-RHS let consume fan-out diagnosis (Item 1 corrected)

Workstream: 0.7.6 toolchain hygiene (unpublished plan).
Branch: `fix/cross-statement-fanout-v2`.
Fixtures: `crates/chelis-ir/tests/cross_statement_fanout.rs`.

## Reproducer

```chelis
module Repro.FanOut

def fanout(x: tensor[3, f32]) -> tensor[3, f32] = {
  alias = x
  mul(x, alias)
}
```

`chelis check --allow-style-violations` emits

```
"errors":[{"kind":"UseAfterConsume","message":"variable `x` was already consumed by binding `alias` at offset 0; later use at offset 0 is invalid","severity":0.9}]
```

The call-RHS sibling already accepts cleanly today on `fan-out-fix` tip
`e6b7c94`:

```chelis
def fanout(w: tensor[3, f32]) -> tensor[3, f32] = {
  a = realize(w)
  b = realize(w)
  add(a, b)
}
```

That is the plan's original Item 1 fixture 3, which a prior dispatch
verified passes. The corrected dispatch targets the var-RHS shape.

## Code path that fails

The shapes diverge after `check_let`'s `is_var_expr(value)` branch:

1. **Call-RHS** (`a = realize(w)`): `check_let` evaluates `value = realize(w)`
   as a non-var expression and recurses via `check_expr → check_realize`,
   which calls `consume_var_expr(w, scope, realize_site(...))`. The
   second statement repeats the call and `consume_var_expr` finds `w`
   already `Consumed`. It falls to the implicit-fan-out arm at
   `crates/chelis-types/src/linearity.rs:601-605`:

   ```rust
   Some(BindingState::Consumed(_)) => {
       // The implicit-linearity pass will insert a Copy for consuming
       // fan-out. Borrow-after-consume remains an error through
       // `read_or_error`.
   }
   ```

   No error, and the DAG-level
   `insert_copy_nodes_for_consuming_fanout` pass at `crates/chelis-ir/src/lower.rs:364`
   inserts one Copy because the two `Realize` nodes both consume the
   same `Load(w)` node.

2. **var-RHS** (`alias = x`): `check_let` takes the `is_var_expr(value)`
   branch at `crates/chelis-types/src/linearity.rs:397-407` and calls
   `consume_var_expr(value, scope, "binding `alias` at offset N")`.
   `x` was `Live` so this transitions it to `Consumed`. So far so good.

   The bug surfaces on the next statement. `mul(x, alias)` routes through
   `check_app`. `mul` is in `builtin_arg_is_borrowed`'s allowlist at
   `crates/chelis-types/src/linearity.rs:943`, so both its args go to the
   borrow branch at line 327-331:

   ```rust
   } else if self.arg_is_borrowed(kids.first(), builtin, index - 1, scope)
       && is_var_expr(arg)
       && self.expr_is_owned_linear(arg, scope)
   {
       self.read_var_expr(arg, scope);
   }
   ```

   `read_var_expr` calls `read_or_error` at line 621-637:

   ```rust
   fn read_or_error(&mut self, name: &str, expr: &Expr, scope: &LinearScope) {
       if let Some(BindingState::Consumed(site)) = scope.top(name) {
           self.errors.push(CheckError::new(
               CheckErrorKind::UseAfterConsume,
               ...
           ));
       }
   }
   ```

   This helper rejects every already-consumed read with no awareness of
   whether the prior consume was structural (closure capture, match
   scrutinee — irreversible) or aliasing (var-RHS let-binding,
   call-RHS let-binding — fan-out-eligible per spec).

## Exact line where the paths diverge

`Checker::consume_var_expr` (lines 580-607) classifies consume sites:

```rust
Some(BindingState::Consumed(consumed_at))
    if consumed_at.description.contains("closure capture")
        || consumed_at.description.contains("match scrutinee") =>
{
    // hard error: structural ownership consume
}
Some(BindingState::Consumed(_)) => {
    // silent: implicit-linearity pass will insert a Copy
}
```

`Checker::read_or_error` (lines 621-637) does not. Every Consumed state
errors uniformly. The path through `consume_var_expr` enjoys the
implicit-fan-out tolerance; the path through `read_or_error` does not.

For repro1 the borrow-read path (mul's auto-borrow) reaches
`read_or_error`, not `consume_var_expr`, and the inconsistent treatment
between the two helpers is the bug.

## Why the spec allows this

`spec/design/implicit_linearity.md` §"Copy Insertion":

> The compiler inserts `Copy` for source-level consuming fan-out: a value
> used in more than one non-borrow consuming position. ... Borrows do
> not count as fan-out. Multiple `&T` uses share the same source.

In repro1 `x` appears in three positions:

| Position           | Spec classification |
| ------------------ | ------------------- |
| `alias = x` (RHS)  | non-borrow consume  |
| `mul(x, ...)` arg  | borrow (mul auto-borrows) |
| `mul(..., alias)`  | borrow (mul auto-borrows) |

One non-borrow consume + two borrows. Per spec this needs no inserted
Copy: the consume is the alias binding (which is structurally a rename
at the IR level — `lower_var` returns the cached `bindings["x"]` for
`(var x)`), and the borrows share the same source. The current
linearity checker rejects the program anyway because `read_or_error`
treats the post-consume borrow reads as `UseAfterConsume`.

The matching call-RHS shape (`a = realize(w); b = realize(w)`) is
allowed because both follow-on consumes route through
`consume_var_expr`, which has the fan-out arm. `mul`'s borrow path
does not.

## DAG-level behavior after the fix

`lower_let` (`crates/chelis-ir/src/lower.rs:2500-2534`) maps `alias` to the
same `NodeId` that `x` already binds to (the `Load { name: "x" }` node).
`lower_var` at line 2565 returns the cached `bindings.get(name)` for both
`(var x)` and `(var alias)`. So `mul(x, alias)` lowers to
`Mul(Load(x), Load(x))` — a single Load with two consumers.

`insert_copy_nodes_for_consuming_fanout` only acts on
`RiscOp::Realize | Drop | Store` consumers (`op_consumes_inputs`,
`lower.rs:427`). `Mul` does not consume, so the pass inserts zero Copy
nodes. The fixture asserts `copy_count == 0` to lock this.

Forward and AD parity vs the explicit `alias = copy(x); mul(x, alias)`
rewrite holds because the explicit version differs only by an inert
`RiscOp::Copy` node whose adjoint is identity.

## Proposed fix

Mirror the `consume_var_expr` arm pattern in `read_or_error`. The helper
should silently accept reads of values consumed by non-structural sites
(var-RHS let-bindings, call-RHS let-bindings via `binding `N` at
offset`, etc.), continuing to error for `closure capture` and
`match scrutinee` consume sites.

```rust
fn read_or_error(&mut self, name: &str, expr: &Expr, scope: &LinearScope) {
    let Some(BindingState::Consumed(site)) = scope.top(name) else {
        return;
    };
    if site.description.contains("closure capture")
        || site.description.contains("match scrutinee")
    {
        self.errors.push(CheckError::new(
            CheckErrorKind::UseAfterConsume,
            ...
        ));
    }
    // Other consume sites are fan-out-eligible per
    // spec/design/implicit_linearity.md §"Copy Insertion".
}
```

The string-based discrimination matches the existing pattern in
`consume_var_expr`. A more architectural refactor (introduce a
`ConsumeKind { Structural, FanoutEligible }` enum and store it on
`ConsumeSite`) is a candidate follow-up but out of scope for this PR —
flagged below.

## Sibling sweep

Brief candidates per the dispatch boilerplate:

| Sibling                                | Status                                                                                                    |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Pattern destructuring `let (a, b) = p` | Not affected. Surf desugars to `__chelis_tmp` intermediates whose `(var __chelis_tmp_N)` carries no type metadata. `expr_is_owned_linear` returns false for the var-RHS chain, so `consume_var_expr` is never called and no consume bug fires. This is hidden by a separate quirk (incomplete type propagation on desugar-synthesized temps); it is not a sibling of repro1. |
| Tuple/record let-bindings              | Same as above — desugar-temp-chain inhibits the linearity check, so repro1's bug doesn't reach them. Probe: `(a, b) = p; alias = a; mul(a, alias)` accepts cleanly today. |
| Match-arm bindings that destructure    | `match` scrutinee consume is the irreversible-structural arm already, and arm pattern bindings introduce fresh names not coupled to outer linearity. Match scrutinee consumes route through `consume_var_expr`'s closure-capture/match-scrutinee error arm and stay rejected by design (`linearity::tests::match_consumes_tuple_scrutinee`). |
| Closure capture of var-RHS             | Already an error by design via `consume_var_expr`'s structural arm; verified by `closure_capture_consumes_outer_tensor` test in `crates/chelis-types/tests/linearity.rs:160`. Not a sibling — closure capture is irreversible. |
| Chained var-RHS aliasing               | Affected. `y = x; z = y; mul(z, x)` errors today (verified empirically). The fix above covers it because each link in the chain consumes via `consume_var_expr` with a `binding `N` at offset` site, and the borrow-reads later all route through `read_or_error`. |
| `mul(x, x)` (within-call fan-out)      | Already passes (plan's Item 1 fixture 1 control). Both args borrow; neither is a consume. |

Net sibling-sweep finding: the **only** affected shapes are direct
var-RHS let-bindings where the let-RHS value is itself a typed
owned-linear variable (i.e. `name = <typed-tensor-var>`). The
tuple-destructure quirk hides the bug from compound bindings via a
separate type-metadata gap that is **out of scope** for this PR — that
quirk is a latent bug in its own right (linearity check is silently
skipped on desugar temps), but flagging it as a sibling would require
its own fixture set and reproducers because the visible behavior is
"check passes" not "check rejects valid code". Recommended as a §5
follow-up; not filed here.

## Out-of-scope / escalations

1. **`ConsumeKind` schema refactor**: replacing the description-string
   discrimination with a typed `ConsumeKind { Structural, FanoutEligible }`
   on `ConsumeSite` would be cleaner and avoid the string-grep fragility
   shared by `consume_var_expr` and the proposed `read_or_error` fix.
   It is a `LinearityInfo`/`CheckedProgram` API change. Out of scope for
   this PR; recommended §5 follow-up.

2. **Desugar-temp type metadata propagation**: `destructure_pattern` in
   `crates/chelis-surf/src/desugar.rs:1135-1156` creates `__chelis_tmp` bindings via
   `tuple-get` whose `(var __chelis_tmp_N)` lacks type metadata. This
   causes the linearity checker to skip the chain (false-negative on
   linearity bugs in tuple-destructure programs). Independent bug, not
   blocking repro1. Recommended §5 follow-up.

Neither escalation is needed to fix repro1.
