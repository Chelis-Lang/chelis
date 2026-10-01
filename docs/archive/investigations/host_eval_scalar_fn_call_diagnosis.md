# HostEval-ScalarFn-F1 — diagnosis

`§5 ID`: HostEval-ScalarFn-F1
`Workstream`: 0.7.8 compiler cleanup, W3
`Branch`: `fix/host-eval-scalar-fn-call`
`Plan`: `/home/jeff/.claude/plans/build-up-a-plan-mossy-meteor.md`
`Phase 0 design note`: `spec/design/archive/compiler_cleanup_0_7_8_spec_lock.md`
  (notes W3 as out of scope — no shared structural contract with W1/W2)

## Bug shape

```
def go() -> f32 = 7.5
result = go()
```

`chelis eval --file` prints `tensor(shape=[], data=[0.0])` instead of
`tensor(shape=[], data=[7.5])`. The bug reproduces across every scalar
return type (f32, f64, i64, bool) for zero-arg user-def calls. A one-arg
scalar fn-call (`def go(x: f32) -> f32 = x; result = go(7.5)`) returns
the correct value, pinning the bug as zero-arg-specific. Exit code is
0; this is a silent miscompilation.

Five fixtures pin the failure mode in
`crates/chelis-cli/tests/host_eval_scalar_fn_call.rs` (W3.1 commit).

## Drop point — discovery

The plan's anchor finding pointed at `EvalContext::eval_list` in
`crates/chelis-compiler-api/src/runtime.rs` (the host evaluator). That
anchor is incorrect for this bug. The eval-time miscompilation is in
the IR lowering pipeline, not the host evaluator.

### Trace

`result = go()` desugars to Deep:

```
(def {} result (app {span: "surf:31..35"} (var {span: "surf:31..33"} go)))
```

`def go() -> f32 = 7.5` desugars to:

```
(def {} go (fn {} (params {}) (lit {type: (t-prim {} f32)} 7.5)))
```

`top_level_lowering_map` (`crates/chelis-ir/src/lower.rs:850`)
classifies `result` as lowered (its body is fully expressible in the
DAG: a zero-arg call to a `def`-bound fn returning a scalar literal).
Per `register_top_level_defs` in
`crates/chelis-compiler-api/src/runtime.rs:228-264`, when a top-level
def is classified as lowered the host runtime does not eagerly
evaluate it — it is left for the IR DAG path. So this bug never enters
the host evaluator at all.

The IR path enters `LowerCtx::lower_app` at
`crates/chelis-ir/src/lower.rs:2712`:

```rust
fn lower_app(&mut self, elems: &[Expr], app_span: Span) -> LoweredValue {
    if elems.len() < 4 {
        return LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            Self::default_type(),
            self.current_span_id.clone(),
        ));
    }
    // ...
}
```

The Deep `(app)` form's `elems` are `[Atom("app"), Map{}, func, args...]`.
For a zero-arg call `(app {meta} (var go))`, `elems.len() == 3`. The
`< 4` early-return fires before any callable resolution; `lower_app`
emits a `RiscOp::Const { value: 0.0 }` and discards both the function
expression and the body literal it would have inlined. Downstream the
DAG evaluator sees a constant-zero node and that is what gets emitted
through `chelis eval --file` as `tensor(shape=[], data=[0.0])`.

The non-zero-arg path (4+ elements) reaches `try_lower_callable_app`
at line 2744, which calls `lower_plain_callable_app` at line 2782.
That path inlines the fn body with the formal-to-actual binding loop
and returns the lowered body — which for the one-arg control yields
`tensor(shape=[], data=[7.5])` correctly. Inside
`lower_plain_callable_app` the `param_names.iter().zip(args.iter())`
loop is a no-op when `args` is empty, so if `try_lower_callable_app`
were reachable for the zero-arg case the body would lower normally.
The early-return at line 2713 is the only thing blocking the
zero-arg path from working.

### Why the plan's anchor was a near miss

The `lower_host_expr_kind` / `lower_app_host_expr` machinery in
`crates/chelis-ir/src/host.rs` is the host-side C-backend emitter. It
correctly produces a `HostExprKind::Call { function: "go", args: [],
arg_tys: [], ty }` for the same Deep input. The plan inferred from
this that the IR side was producing a well-formed call and the bug
must be in the runtime's `HostExprKind::Call` evaluation arm. But
`chelis eval --file` for a fully-lowerable result does not route
through the host-side `HostExpr` path at all — it routes through the
IR DAG path (`lower.rs`) and is evaluated by the DAG evaluator that
the C backend also uses for the forward pass. The host-side `HostExpr`
path is only invoked for ops the lowering map decided to leave on the
host (the `is_fn && !lowered_names.get(name)` branch in
`register_top_level_defs`).

The bug is reachable only through the lowering path, and only for
zero-arg `app` forms. It pre-existed the workstream and is independent
of `par` (per the §5 entry, surfaced during PR #56's "par-in-fn
returns 0.0" investigation).

## Fix shape

Replace the early-return guard's threshold from `< 4` to `< 3` so the
zero-arg `(app {meta} func)` form reaches the callable-resolution
dispatch:

```rust
fn lower_app(&mut self, elems: &[Expr], app_span: Span) -> LoweredValue {
    if elems.len() < 3 {
        return LoweredValue::Node(self.dag.add_node(
            RiscOp::Const { value: 0.0 },
            vec![],
            Self::default_type(),
            self.current_span_id.clone(),
        ));
    }
    // ...
}
```

`elems.len() == 3` corresponds to `[tag, meta, func]` with `&elems[3..]`
yielding an empty slice. `lower_builtin_app`, `try_lower_callable_app`,
and the fallback `lower func and args, return last` path all handle
empty arg slices correctly without further change (each iterates a
slice that can be empty without dropping or producing the func's
lowered value).

Single-line guard relaxation; no public API change. The four
ignored fixtures in `host_eval_scalar_fn_call.rs` flip to running and
assert exact stdout for each scalar return type.

## Sibling sweep target

Per `feedback_sibling_sweep_target.md`, the failure mode is "tag-dispatch
arity guard rejects a legitimate shape." Sweep targets:

- Other `lower_*` methods in `crates/chelis-ir/src/lower.rs` that
  guard with `elems.len() < N` and return `Const { value: 0.0 }`.
  Each guard's `N` should match the minimum legitimate arity for its
  tag, not be one too high.
- Specifically inspect `lower_tuple`, `lower_pipe`, `lower_par`,
  `lower_if`, `lower_match`, `lower_fn`, `lower_cast`, `lower_let` for
  the same off-by-one.
- The Deep grammar in `spec/03-deep-syntax.md` is the canonical source
  of minimum arity per tag.

Findings recorded in the W3 PR body.

## Confidence

- Repro pinned by four `#[ignore]` fixtures that fail on `f1aa52c`
  with exact-stdout assertions distinguishing `data=[0.0]` (bug) from
  the body literal.
- Non-bug control (one-arg scalar fn) passes today, isolating the
  failure to the zero-arg-`app` early return.
- Deep s-expression dump (`chelis deep`) confirms the `(app {meta}
  (var go))` shape that lower_app's early return rejects.

## Discovery-contract note

The plan named `EvalContext::eval_list` as the bug location; the
actual drop point is one layer up in IR lowering at `lower_app`. The
fix shape stays surgical (single-line guard relaxation in
`crates/chelis-ir/src/lower.rs`) and does not change any public type or
cross-crate API. This diagnosis surfaces the discovery for the
orchestrator's awareness; the §5 entry's stated closure path (trace
`lower_host_expr_kind` or its callees) is corrected here to point at
`lower_app` instead.
