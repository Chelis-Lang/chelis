# Pipe `grad` / `vmap(grad)` stage lowering — diagnosis

Diagnoses the failure mode for Item 2 of the 0.7.6 toolchain hygiene
workstream (`/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`).
Recommends a fix that reuses the existing non-pipe lowering.

## Bug shape (confirmed)

`crates/chelis-ir/src/lower.rs` — `lower_pipe()` (L4394–L4590) walks each
stage of `(pipe seed s1 s2 ...)` and lowers it. The resolution table at
L4567–L4585 currently is:

```text
resolve_callable_expr(stage) =>
    Some(CallableExpr::Plain(fn))      → lower_plain_callable_with_values
    Some(CallableExpr::Vmap { ... })   → lower_vmap_callable_with_nodes
    Some(CallableExpr::Grad     { ... })
    Some(CallableExpr::VmapGrad { ... }) → lower_unrepresentable("pipe stage")
    None                                 → lower_unrepresentable("pipe stage")
```

The diagnostic emitted is `"pipe stage is not supported by IR evaluation
yet; use 'chelis build --target c' instead"`. The user-facing text says
"IR evaluation" but the failure is at **IR lowering**; both the
evaluator backend and the C backend reach the lowered DAG via this
shared path. Backend choice does not avoid the gap.

## Why it currently fires

For `x |> grad(f)` Surf parses `grad(f)` as `Expr::Grad(...)` (not as
`Apply`). `desugar_pipe_stage` (`crates/chelis-surf/src/desugar.rs:511`)
only wraps `Apply(func, args)` stages in a lambda; everything else
desugars in place. So the Deep shape is the bare callable:

```
(pipe {} <seed> (grad {} (var {} f)))
```

When `f` is in `program_defs`, `resolve_callable_expr` returns
`CallableExpr::Grad { fn_expr, wrt }` and we hit the
`lower_unrepresentable` arm at L4581–L4583. Same shape for `xs |>
vmap(grad(f))` — `(vmap {} (grad {} (var {} f)) <axis>)` resolves to
`CallableExpr::VmapGrad { ... }` and hits the same arm.

## Canonical lowering

`x |> grad(f)` is semantically `grad(f)(x)` — apply the gradient
function to the seed. The non-pipe form already lowers correctly via
`lower_app` → `try_lower_callable_app` → `lower_grad_callable_app`
(`lower.rs:2813`) for `Grad`, and `lower_vmap_grad_callable_app`
(`lower.rs:3149`) for `VmapGrad`. Both helpers take `args: &[Expr]`,
lower each to a `NodeId`, then drive the grad/vmap-grad construction.

The pipe path already has the previous stage as a `LoweredValue`
(usually `Node(NodeId)`) — i.e., the argument is already lowered. The
non-pipe helpers do not currently expose a node-based entry point for
the seed; `lower_vmap_callable_app` already has a `with_nodes`
counterpart (`lower_vmap_callable_with_nodes` at L3005), but
`lower_grad_callable_app` and `lower_vmap_grad_callable_app` do not.

## Proposed fix

Extract `with_nodes` helpers from `lower_grad_callable_app` and
`lower_vmap_grad_callable_app` so the pipe path can reuse the same
construction with pre-lowered argument node IDs. The `lower_*_app`
forms become thin wrappers that lower the arg expressions and delegate.
This mirrors the existing `lower_vmap_callable_app` /
`lower_vmap_callable_with_nodes` split.

Then in `lower_pipe()` replace the two `lower_unrepresentable` arms
with calls to the new helpers, using `current.expect_node("pipe
stage")` to recover the `NodeId` from the previous stage.

This is the minimal change that reuses the canonical, already-tested
lowering. No new behaviour: just a fresh entry point into the same
construction.

## Canary test fate

`unsupported_pipe_stage_returns_diagnostic_without_panicking_public_api`
(`lower.rs:6331`) uses the fixture
`(pipe {} (lit ...) (grad {} (var {} f)))` with **no** `program_defs`.
In that state `resolve_callable_expr_inner` recurses into `(var {} f)`,
fails to find `f` in `local_callables` or `program_defs`, returns
`None`. The `"grad"` branch propagates `None`. Back in `lower_pipe`,
the `None` fallthrough at L4587 fires — **not** the `CallableExpr::Grad`
arm.

Decision: **keep the canary unchanged**. The new test file pins the
positive shape (`grad(f)` with `f` defined → pipe matches `grad(f)(x)`).
The canary continues to cover the `None`-fallthrough case (callable
that cannot be resolved still emits the diagnostic). After the fix
this fixture remains broken-by-design: a pipe whose stage names a
function not in scope is still unrepresentable.

The fix does not change the canary's expected behaviour. No spec edit;
no fixture replacement.

## Sibling lowering gaps (future workstreams)

Per §2.3 of the plan, the surrounding `lower_unrepresentable` sites in
`lower.rs` are also potential future workstreams. Recommendations,
**not** §5 entries (orchestrator decides whether to file in
`docs/gap_synthesis.md`):

| Gap                                       | Location               |
|-------------------------------------------|------------------------|
| `grad` outside function application       | `lower.rs:2821`        |
| `vmap` standalone                         | `lower.rs:3013`        |
| `vmap` with no tensor arguments           | `lower.rs:3058-3061`   |
| `vmap(grad)` with no tensor arguments     | `lower.rs:3158`        |
| `if` expressions (control flow)           | `lower.rs:4673`        |
| `par` (parallelism)                       | `lower.rs:4728`        |
| `match` (pattern matching)                | `lower.rs:4854`        |
| `jit` operator                            | `lower.rs:2447`        |

Each surfaces the same user-facing diagnostic family
(`"X is not supported by IR evaluation yet"`) and each represents a
separable scoping decision for a follow-on item.

## Sibling sweep (this PR)

The only direct sibling within Item 2's scope is the
`lower_unrepresentable` fallthrough for `None`-resolved callables
inside `lower_pipe` (L4587). It is **deliberately left in place** by
the fix: a pipe stage that names an unresolved callable should still
emit a diagnostic, not silently succeed. The canary test pins that
behaviour.
