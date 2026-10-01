# Item 2c — C-backend rejection of `grad` over piped function body

## Summary

`chelis build --target c` rejects `grad(sumsq)(theta)` when `sumsq`'s body
is written in canonical pipe form:

```
def sumsq(theta: tensor[3, f32]) -> f32 =
  mul(theta, theta) |> sum(0) |> tensor_to_scalar
def gradient(theta: tensor[3, f32]) -> tensor[3, f32] = grad(sumsq)(theta)
```

The equivalent nested-call form
`tensor_to_scalar(sum(mul(theta, theta), 0))` compiles cleanly. The
rejection surfaces at `crates/chelis-cli/src/main.rs:1571` and is gated
by `host_program_unresolved_call_sites`
(`crates/chelis-ir/src/host.rs:1314`) detecting a fallback
`Builtin { name: "call" }` in `gradient`'s host body.

The sibling-sweep brief (G12) hypothesized the gap was in the host-lane
**summarizer** — i.e. that
`derive_host_function_specialization`/`try_summarize_sparse_helper` did
not recognize pipe-lowered `Let` chains as summarizable. Empirical
tracing through `lower_compiled_program` shows the failure path is one
layer earlier: the rejection happens during **IR-level pipe lowering
inside grad's autodiff subcontext**, before the summarizer is reached.
The host-lane re-uses IR lowering, so the bug is naturally fixed where
it lives (in `lower_pipe`) and the fix carries to both `chelis eval`
and the C backend.

## Reproduction

Two fixtures, locked in `crates/chelis-cli/tests/cli.rs`:

- Control: `build_c_grad_over_named_fn_with_nested_call_body_builds`
  — non-pipe form, passes today, locks the baseline.
- Target: `build_c_grad_over_named_fn_with_pipe_body_builds`
  — pipe form, gated `#[ignore]` today, must pass after the fix.

Both fixtures assert the same numerical gradient
(`2 * theta = [2.0, 4.0, 6.0]` for `theta = [1.0, 2.0, 3.0]`).

`chelis eval --file <pipe form>` rejects with the same gap, emitting:

```
error: pipe stage is not supported by IR evaluation yet; use
       `chelis build --target c` instead at source span `surf:95..111`
```

The span `surf:95..111` covers `tensor_to_scalar` — confirming the
single failing stage is the bare-var `tensor_to_scalar` pipe stage.

## Actual code-level diagnosis

### Path through host lowering (non-pipe vs pipe `sumsq`)

Both shapes lower `sumsq`'s body identically through the **host lane**:
they reach `Builtin(tensor_to_scalar)(TensorCall(helper=0))` where
`helper=0` is the lowered DAG for `sum(mul(theta, theta), 0)`. The host
lane's pipe handler (`crates/chelis-ir/src/host.rs:3217` —
`beta_reduce_pipe_stage`) rewrites the pipe form into the equivalent
nested-app form before lowering, so the host-lane summarizer sees the
same shape in both cases.

The divergence is in the **`gradient` body**:

- Non-pipe: lowers to `TensorCall(helper=0)`. The host lane delegates
  to `try_lower_tensor_helper_call`, which calls
  `try_lower_subexpr_program` on `(app (grad sumsq) theta)`. The IR
  lowerer runs `lower_grad_callable_with_nodes`, sets up a subctx, and
  lowers `sumsq`'s nested-call body. All ops are recognized; the DAG
  is built; the tensor helper is created; the host body becomes a
  single `TensorCall`.
- Pipe: same path, but the IR lowerer fails. `lower_grad_callable_with_nodes`
  sets up a subctx and calls `subctx.lower_expr(body)` where `body` is
  the `pipe` expression. `lower_pipe` (`crates/chelis-ir/src/lower.rs:4415`)
  iterates stages and rejects the bare-`(var {} tensor_to_scalar)`
  stage. With `unrepresentable_panic_suppressed` active (the host-lane
  speculative-DAG mode), the rejection unwinds; the host lane catches
  the panic; `try_lower_tensor_helper_call` returns `None`;
  `lower_app_host_expr` falls through to the generic Builtin path and
  emits `Call { function: "call", ... }` — the unresolved-call
  fallback that `host_program_unresolved_call_sites` detects.

### Root cause: `lower_pipe`'s known-unary-builtin set is incomplete

`lower_pipe` at `crates/chelis-ir/src/lower.rs:4415` handles three
classes of pipe stage:

1. **Known unary elementwise builtin** (16 names: `neg`, `exp`, `log`,
   `sin`, `sqrt`, `cos`, `tan`, `atan`, `abs`, `floor`, `ceil`, `relu`,
   `sigmoid`, `tanh`, `silu`, `gelu`). Lowered directly to the
   corresponding `RiscOp` (lines 4438–4587).
2. **`resolve_callable_expr` resolves**: Plain function, Vmap, Grad,
   VmapGrad. Lowered via the corresponding `lower_*_callable_with_nodes`
   helper (lines 4588–4628).
3. **Everything else**: `lower_unrepresentable("pipe stage", ...)`.

`tensor_to_scalar` is a **unary** builtin (changes a 0-rank tensor to a
scalar by metadata), defined at `lower.rs:3959` as a pass-through. It
is NOT in the 16-name elementwise set, and `resolve_callable_expr`
cannot resolve it (it's not in `local_callables` or `program_defs` —
it's a primitive). So it falls into class (3) and is rejected.

`scalar_to_tensor` has the same shape (`lower.rs:3962`). Any other
bare-`var` primitive used as a unary pipe stage is also rejected by
the same code path.

### Why the host-lane sumsq body lowering succeeds

`beta_reduce_pipe_stage` at `host.rs:2947` rewrites `(pipe seed s1 s2)`
into the equivalent nested-app form: a `(var f)` stage becomes
`(app (var f) acc)`; a lambda stage gets beta-reduced. This rewrite
runs **before** `lower_app_host_expr` is invoked, so the resulting
shape is `(app (var tensor_to_scalar) (app (var sum) ...))` — which
`lower_app_host_expr` handles by routing `tensor_to_scalar` through
`lower_builtin_app`.

The host lane and IR lane have **divergent** pipe semantics. The host
lane normalizes pipes to nested-apps; the IR lane special-cases 16
elementwise unary names and falls back to `lower_unrepresentable` for
everything else, including other valid unary builtins.

### Why this divergence is wrong

Both lanes are spec-compliant Surf representations of the same program.
`spec/01-nomenclature.md` §3.6 says `x |> f` means `f(x)` regardless
of whether `f` is an elementwise primitive, a non-elementwise primitive
(`tensor_to_scalar`, `sum`, `mean`), or a user-defined fn. The IR-level
`lower_pipe` enforces a stricter substructure than the spec describes.

## Recommended fix

In `lower_pipe` (`crates/chelis-ir/src/lower.rs:4415`), after the
known-unary-builtin block and before the `resolve_callable_expr`
fallthrough, **handle bare `(var {} name)` stages by synthesizing an
`(app (var {} name) <accumulator>)` expression and routing through
`lower_app`**. This makes IR pipe lowering structurally equivalent to
the host-lane's `beta_reduce_pipe_stage` rewrite.

Concretely: when the stage is `(var {} name)` and the name is neither
a local callable nor a program def, bind the current accumulator
NodeId into a fresh local, synthesize the `app` expression referencing
that local, and call `lower_app`. The resulting DAG matches what the
nested-call form would have produced. `lower_app` already handles
builtins via the early-return at `lower.rs:2625–2638`.

### Surface

- Single helper extension in `crates/chelis-ir/src/lower.rs::lower_pipe`.
- No changes to the host-lane summarizer.
- No changes to `host_program_unresolved_call_sites`.
- No schema changes.

### Tests flipped

- `build_c_grad_over_named_fn_with_pipe_body_builds` — flip from
  `#[ignore]` to running.

### Side benefits

- `chelis eval` on the pipe form will also succeed (currently fails
  with the same `pipe stage` diagnostic; the IR-evaluator path shares
  `lower_pipe`).
- Other bare-var primitive pipe stages (`scalar_to_tensor`, `sum`,
  `mean`, etc. used unary) will lower cleanly. Today these would also
  hit `lower_unrepresentable`.

## Sibling sweep

Pattern-match gaps with the same shape as G12 — IR-level lane has a
narrower allowlist than the host-lane equivalent has:

### IR-vs-host divergent pipe handling

The host lane has a single canonical rewrite (`beta_reduce_pipe_stage`)
that handles ANY pipe stage shape by either beta-reducing a lambda or
wrapping in an outer `(app stage acc)`. The IR lane has the
fine-grained 16-name allowlist plus `resolve_callable_expr`. Once the
IR-side fix synthesizes `(app stage acc)` for unresolved bare-vars,
the two lanes converge.

**Future workstream**: collapse `lower_pipe`'s 16-name elementwise
arm into the general "synthesize `(app (var name) acc)` and dispatch
through `lower_app`" path. The elementwise arm was added before
`lower_builtin_app` covered the full primitive vocabulary; today
`lower_app` + `lower_builtin_app` handles all the same names. The
arm is a vestigial optimization that complicates the code without
shipping a measurable speedup (the synthesized-app path goes through
the same `RiscOp` constructors). **Recommendation only — orchestrator
decides whether to file a §5 entry.**

### `vmap` standalone pipe stage

`x |> vmap(f)` should work today (Item 2 landed in PR #26 closing the
`CallableExpr::Vmap` arm). However, the same root cause as Item 2c
applies if `f`'s body uses a bare-var primitive pipe stage. The Item 2c
fix in `lower_pipe` should also unblock `vmap` over pipe-bodied
functions. **Spot-check during implementation.**

### `lower_pipe` accumulator-binding hygiene

When synthesizing `(app (var name) acc)`, the accumulator NodeId must
be exposed to `lower_app` without re-lowering it (re-lowering would
duplicate the entire upstream DAG and break linearity). The
implementation must inject the NodeId via a synthetic binding lookup
or a direct `LoweredValue::Node` path. Confirm during implementation
that no double-lowering occurs.

## Out of scope here

- Re-enabling `redundant-linearity-call` autofix (Item 5 of the parent
  plan; gated on Item 1 merging — orthogonal).
- The full G1/G2 IR-level grad/vmap over function-valued parameters.
  Those require monomorphization, not a pipe-lowering fix.
- Collapsing `lower_pipe`'s 16-name allowlist into the general path.
  Recommended above as a separate workstream.

---

*Diagnosis only. No code changes in this commit.*
