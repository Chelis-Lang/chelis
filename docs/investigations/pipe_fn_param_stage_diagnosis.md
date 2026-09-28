# Pipe fn-typed-parameter stage lowering — diagnosis

Diagnoses Item 2-extended dispatch A (G9/G10) from the Item 2 sibling-sweep
findings (`docs/investigations/item2_sibling_sweep_findings.md`). The
sweep treats G9 (the code-site label) and G10 (the user-facing label) as
a single gap; this note follows that consolidation and uses "G10" for
the failure shape.

The gap: a pipe stage that resolves to a function-valued parameter
(e.g. `x |> f` inside `def double_apply(f, x) = x |> f |> f`) fires the
`pipe stage is not supported by IR evaluation yet` diagnostic from
`crates/chelis-ir/src/lower.rs:4631`, even though Surf accepts the
program and `chelis check` types it cleanly.

## Bug shape (confirmed empirically)

Verified against `6487d35` (HEAD of `worktree-agent-aced1f92373e03acd`,
i.e. `main` plus PR #26's `lower_pipe` Grad/VmapGrad arms) using a
minimal Surf fixture:

```
module Repro.PipeFnParam
def double_apply(f: &tensor[3, f32] -> tensor[3, f32], x: &tensor[3, f32])
  -> tensor[3, f32] = x |> f |> f
def relu1(x: &tensor[3, f32]) -> tensor[3, f32] = relu(x)
def test_double_apply(seed: &tensor[3, f32]) -> tensor[3, f32] =
  double_apply(relu1, seed)
```

`chelis check` passes. Both `chelis eval --file` and `chelis build`
fail with:

```
error: pipe stage is not supported by IR evaluation yet;
  use `chelis build --target c` instead at source span `surf:125..126`
```

The span is the first `f` in `x |> f |> f`. The Deep printer confirms
the body lowers to `(pipe ... (var {} x) (var {} f) (var {} f))` with
`f` declared in the enclosing `(params {} (f {type: (t-fn ...)}) ...)`.

## Why `chelis eval`/`build` both fail on lowering

`chelis-ir::lower::try_lower_program` walks every top-level def. The
classification map (`top_level_lowering_map`, `crates/chelis-ir/src/lower.rs:780`)
delegates to `def_is_lowered` (`lower.rs:1261`) which calls
`type_is_never_lowerable` (`lower.rs:942`) on the declared signature.
That helper only inspects the function's *return* type; `t-fn`
parameter types are not consulted. So `double_apply`'s signature
`(t-fn (t-fn ...) (t-ref ...) (t-tensor ...))` returns "lowerable",
and the lowering loop dispatches the body through `LowerCtx::lower_def
→ lower_expr → lower_fn → lower_pipe`.

Inside `lower_fn` (`lower.rs:4361`), each param registers via
`lower_fn_param_binding` (`lower.rs:4389`). For `f` (a `t-fn` type),
`type_from_type_expr` falls through to `default_type()` (because
`type_from_type_expr` only handles `t-prim`, `t-ref`, `t-tensor` —
`lower.rs:2056`) and a tensor-shaped `Load { name: "f" }` is created
and inserted into `self.bindings`.

Then `lower_pipe` (`lower.rs:4415`) walks each stage. For stage
`(var {} f)`:

1. The known-unary-builtin fast path (`lower.rs:4438`–`4587`) checks
   `f` against the builtin name set. `f` is not a builtin → fall
   through.
2. `resolve_callable_expr(func_expr)` calls
   `resolve_callable_expr_inner` (`lower.rs:2694`). The `Some("var")`
   arm at `lower.rs:2704`–`2717` looks up `f` in `local_callables`
   then `program_defs`. **Function-valued parameters live in
   `self.bindings`, not in either table**, so the lookup chain returns
   `None`.
3. `lower_pipe` hits the `None`-resolution fallthrough at
   `lower.rs:4631` and rejects via `lower_unrepresentable("pipe stage",
   ...)` which raises the user-facing `LowerDiagnostic`.

No suppression mechanism intercepts the panic: `lower.rs` exposes
`with_suppress_unrepresentable_panic` (`lower.rs:26`) and an
`UnrepresentableDag` payload (`lower.rs:43`), but a workspace grep
shows no callers of either today (`grep -rn 'with_suppress_unrepresentable'`
returns only the definition). The diagnostic propagates straight to
the CLI.

## Why call-site inlining is unaffected

`lower_plain_callable_app` (`lower.rs:2911`) inlines a callable
application by substituting callable arguments into `local_callables`
before lowering the body (`lower.rs:2935`–`2949`,
`self.local_callables.insert(name.clone(), callable)`). When
`test_double_apply` calls `double_apply(relu1, seed)`, the lowering
populates `local_callables["f"] = (var {} relu1)` before lowering
`double_apply`'s body. The pipe stage's resolver then finds `f` in
`local_callables`, recurses to resolve `relu1` through `program_defs`,
and dispatches the stage through `CallableExpr::Plain`. Working.

The call-site path therefore produces correct DAG semantics. The
standalone-def lowering done up-front in `try_lower_program` produces
a DAG entry for `double_apply` that no caller ever references in
practice (every caller re-inlines), but the lowering must still not
panic — that is the fix's contract.

## Chosen fix surface

Add a new `CallableExpr::Parameter { name }` variant. Two questions
the variant must answer: (a) which `(var ...)` references qualify as
"fn-typed parameter callables," and (b) what does `lower_pipe` do
with the variant?

**Detection.** Track function-typed parameter names in a new
`LowerCtx::fn_typed_params: HashSet<String>` field. Populate it from
`lower_fn` (`lower.rs:4361`–`4387`) when a param's declared type
expression has tag `t-fn` (using `chelis_deep::ast::get_tag`).
Restore/save the set across nested `lower_fn` invocations the same way
`lower_fn` already restores `self.bindings`. `lower_let` does not need
a separate save/restore because let-bound names go through
`callable_binding_expr` for actual callables and through `bindings`
for non-callables; let-binding a fn-typed parameter to a new name is
covered transitively via the existing `local_callables`
recursion in `resolve_callable_expr_inner`'s `var` arm.

**Resolution.** Extend `resolve_callable_expr_inner`'s `Some("var")`
arm at `lower.rs:2704`–`2717`: after the existing
`local_callables`/`program_defs` lookup fails to produce a body, but
*before* returning `None`, check if the name is in `fn_typed_params`.
If yes, return `Some(CallableExpr::Parameter { name })`. The
`visited`/`inlining_names` guards remain in place — a parameter
callable is not recursive (it has no body to recurse into).

**Pipe dispatch.** In `lower_pipe`, add a new arm for
`CallableExpr::Parameter`: keep `current` unchanged and continue. The
DAG genuinely cannot represent a call to a function-typed parameter
(no `RiscOp::Call` exists in the DAG vocabulary — `crates/chelis-ir/src/dag.rs:463`),
so the standalone-def DAG node for the pipe is semantically a no-op.
This matches `lower_app`'s existing "Not a recognized built-in — lower
func and args, return last" fallback at `lower.rs:2644`–`2649`, which
also produces a semantically wrong but non-panicking lowering for the
same parameter-callable shape in non-pipe position. The user-visible
correct semantics come from call-site inlining, which is unaffected.

**App-side dispatch.** `try_lower_callable_app` (`lower.rs:2652`)
returns `Some` for every `CallableExpr` variant today. For
`CallableExpr::Parameter`, return `None` so `lower_app`'s existing
fallback at `lower.rs:2644`–`2649` (which already handles this case
correctly in non-pipe shape) keeps owning the path. This avoids
introducing a new code path through the app machinery and keeps the
Parameter-variant blast radius scoped to `lower_pipe`.

**Linearity-binding helper.** `callable_binding_expr` (`lower.rs:2760`)
returns `Some(expr.clone())` whenever `resolve_callable_expr` returns
`Some`. After the fix, `Parameter` resolves as `Some(...)`, which
would cause `lower_let` (`lower.rs:2498`) and `lower_plain_callable_app`
(`lower.rs:2936`) to insert `(var f)` into `local_callables` under
the let-bound or substituted name. For let-binding (`let g = f`),
this is the desired transitive behavior — referencing `g` resolves
to `f` resolves to `Parameter`. For `lower_plain_callable_app`, this
is the *caller's* responsibility (`local_callables.insert` of the
callable expr), and the path remains correct. No change needed.

## Canary test fate

`unsupported_pipe_stage_returns_diagnostic_without_panicking_public_api`
(`lower.rs:6428`) uses the Deep fixture
`(pipe {} (lit {type: (t-prim {} f32)} 1.0) (grad {} (var {} f)))`
with no `program_defs` for `f` and no enclosing `fn` registering `f`
as a parameter. After the fix:

- The `grad` arm in `resolve_callable_expr_inner` (`lower.rs:2747`)
  recurses into `(var {} f)`.
- The `var` arm finds `f` neither in `local_callables` nor
  `program_defs`. The new fn-typed-parameter check also fails because
  no `lower_fn` invocation has registered `f` in `fn_typed_params`
  (the canary is a bare pipe expression, no surrounding fn).
- Returns `None`. The `grad` arm propagates `None`.
- `lower_pipe`'s `None` fallthrough fires `lower_unrepresentable` as
  before.

The canary continues to assert the diagnostic for truly unresolvable
callables. **No change needed to the canary.**

## Sibling sweep — `resolve_callable_expr_inner` `None` returns

Walked every site where `resolve_callable_expr_inner` returns `None`
and asked whether the `None` is correct in the
post-fix-but-pre-broader-architecture world:

| Site | Tag | None reason | Legitimate callable? | Action |
|------|-----|-------------|----------------------|--------|
| `lower.rs:2700` | not a list | Atom or Map at callable position | No | Keep `None`. |
| `lower.rs:2708` | `var` w/ non-symbol child | Malformed Deep | No | Keep `None`. |
| `lower.rs:2710` | recursion / inlining guard | Same name already inlining | No | Keep `None` (infinite-loop guard). |
| `lower.rs:2715` (post-fix new) | `var` w/ unknown name | Truly out-of-scope name | No | Keep `None`. Same fall through. |
| `lower.rs:2729` | `vmap(grad)` w/ Plain inner | Inner callable not resolvable as Plain | Sometimes — G2 territory | Keep `None` here; G2 owns the broader fix. |
| `lower.rs:2737` | `vmap` w/ wrapped inner | Inner callable resolves to Vmap/VmapGrad/Grad (no nested vmap-vmap) | Sometimes | Same — G2/G4 territory. |
| `lower.rs:2748` | `grad` w/ wrapped inner | Inner callable resolves to non-Plain | Sometimes — G1 territory | Same — G1 owns. |
| `lower.rs:2756` | unknown tag | `if`/`match`/`let`/`pipe`/etc. as callable | Pipe-position requires they be callable; today they aren't | See below — recommendation only. |

The only `None`-return that swallows a *legitimate* callable today is
the fn-typed-parameter case (G10), which this dispatch fixes.

For the broader sibling sweep: when a pipe stage is itself an
expression that produces a callable value at runtime (e.g. `x |>
(if cond then f else g)` or `x |> (let h = f in h)`), `resolve_callable_expr_inner`
returns `None` at `lower.rs:2756` because those tags aren't in the
match. These are real lowering gaps in spec terms (pipe stages can be
any callable expression per `spec/01-nomenclature.md` §3.6), but each
is a substantially larger architectural fork than G10 — they need
either a real `Call` op in the DAG or a host-lane fallback for
arbitrary control-flow-produced callables. **Recommend** (not file) a
§5 entry in `docs/archive/reports/gap_synthesis.md` for "pipe stage = arbitrary
callable expression"; orchestrator decides.

The host-lane summary G12 (`crates/chelis-ir/src/host.rs:1314`,
`host_program_unresolved_call_sites`) is a separate dispatch (the
sweep's bundle C); not touched here.

## Surface size

- One new enum variant: `CallableExpr::Parameter { name: String }`.
- One new `LowerCtx` field: `fn_typed_params: HashSet<String>`.
- Three edits in `lower.rs`:
  - `lower_fn` (or `lower_fn_param_binding`): populate
    `fn_typed_params` for `t-fn`-typed params, save/restore across
    nested fn scopes.
  - `resolve_callable_expr_inner` `Some("var")` arm: synthesize
    `Parameter` variant.
  - `lower_pipe`: new match arm for `Parameter` = identity pass.
- One edit in `try_lower_callable_app`: return `None` for the
  `Parameter` variant (defensive; lets `lower_app`'s existing fallback
  handle it).

Estimated diff: ~30 lines net. No new helpers required. No spec edit
(spec already permits any callable as pipe stage —
`spec/01-nomenclature.md` §3.6).
