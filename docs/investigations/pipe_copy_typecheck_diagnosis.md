# Pipe `copy` / `cast` type-check rejection — diagnosis

Lifted from Finding 4 of the 0.7.6 red team (PR #51): `x |> copy`
parses (Item 2b, PR #35) and `chelis fmt` round-trips, but `chelis
check` rejects with

```
TypeMismatch: copy requires tensor input, got ?<freshvar>
```

even when the pipe input is statically typed (e.g.
`def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy`). The control
case `x |> realize` passes today, so the discrimination is internal to
the type-checker's per-builtin inference arms.

The same shape of bug is recorded as `TypeCheck-PipeCast-F1` in
`docs/gap_synthesis.md` §5 for the one-arg `cast(type)` pipe-stage
form synthesized by PR #42 (Item 2b extras dispatch H). The narrower
fix option (b) noted in that entry also closes Finding 4.

## Reproduction

Sources:

```surf
def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy
def g(x: tensor[3, f32]) -> tensor[3, f32] = x |> realize
def h(x: tensor[3, f32]) -> tensor[3, f32] = x |> cast(f32)
```

`chelis check`:

* `f` (copy): `TypeMismatch: copy requires tensor input, got ?283`.
* `g` (realize): `score: 1`, no errors.
* `h` (cast(f32)): `CastNonTensor: cast requires tensor or prim type, got ?283`.

The fresh-variable index varies; the shape is identical.

The fixtures `crates/chelis-types/tests/pipe_copy_typecheck.rs` pin
all three cases.

## Pipeline trace

PR #35 made bare unary-builtin keywords parse as pipe stages by
synthesizing a unary lambda over a fresh `__chelis_pipe` parameter
(`crates/chelis-surf/src/parser.rs::synthesize_bare_unary_builtin_lambda`,
`parse_pipe_stage`). The desugarer wraps the chain in a `pipe` Deep
node:

```
def f(x: tensor[3, f32]) -> tensor[3, f32] = x |> copy
```

becomes

```
(def f (fn (params (x {type: (t-tensor (d-lit 3) (t-prim f32))}))
  (pipe (var x)
        (fn (params __chelis_pipe) (copy (var __chelis_pipe))))))
```

`infer_pipe` in `crates/chelis-types/src/infer.rs` walks each stage:

1. `current_ty = infer_expr(seed)` -> `Tensor[3, f32]` (resolved).
2. For each stage:
   1. `stage_ty = infer_expr(stage)` -- recursively infers the lambda,
      which calls `infer_fn`.
   2. `infer_fn` allocates a fresh type variable for the lambda's
      parameter `__chelis_pipe` and recurses into the body.
   3. Body is `(copy (var __chelis_pipe))`. `infer_copy` calls
      `subst.apply(&inner_ty)` and matches on the resolved form. At
      this point `__chelis_pipe` is still a fresh `TypeVar` because
      pipe-stage unification has not happened yet. The match arm

      ```
      Type::Tensor(_, _) | Type::Error => resolved,
      Type::Ref(inner) if matches!(inner.as_ref(), Type::Tensor(_, _)) => *inner,
      _ => { errors.push(...); Type::Error }
      ```

      falls into the `_` arm and pushes the rejection.
   4. The pipe loop then computes
      `expected = Fn([current_ty], ret_tv)` and unifies it with
      `stage_ty`. Unification succeeds (binding the fresh parameter
      var to `Tensor[3, f32]`), but the rejection has already been
      pushed and the body type is `Type::Error`.

`infer_realize` (same file, immediately above `Some("copy")`) simply
forwards `infer_expr(inner)` without inspecting the resolved type:

```
Some("realize") => {
    let kids = children(list);
    if let Some(inner) = kids.first() {
        infer_expr(inner, env, vg, subst, adt_reg, errors, ...)
    } else { Type::Error }
}
```

That is why `x |> realize` works: the body returns the lambda
parameter's fresh type variable, pipe-stage unification binds it, and
no per-builtin gate fires.

`infer_cast` (also in `infer.rs`) has the same shape as `copy`: it
calls `subst.apply` on the inner type and rejects anything that is
neither `Tensor` nor `Prim`. The fresh-var case takes the rejection
path.

## Discrimination

The structural difference between the rejecting arms (`copy`, `cast`,
`borrow`) and the permissive `realize` is whether the arm inspects
`subst.apply(&inner_ty)` against a closed set of resolved shapes. The
synthesized pipe-stage lambda checks its body before the pipe loop
unifies the lambda's parameter type with the upstream pipe value's
type, so any per-builtin gate that runs at body-inference time sees a
fresh type variable and fails.

`borrow` has the same arm shape as `copy` but the bare `x |> borrow`
spelling is not a spec-meaningful pipe stage and the parser does not
synthesize a lambda for it, so the bug does not surface there today.

## Closure options

### (a) Relax `copy` / `cast` arms to accept fresh type variables

`infer_copy` and `infer_cast` could pattern-match
`Type::Var(_) | Type::Tensor(...) | ...` and return the resolved type
without rejecting, letting later unification surface any real
mismatch.

This is narrower than reordering the pipe loop but introduces a
silent path: if a fresh variable that never gets bound to a `Tensor`
flows into `copy`, the rejection is deferred to unification (which
would emit a generic type-mismatch error, not the targeted "copy
requires tensor input" diagnostic). The existing `copy_rejects_scalar`
and `copy_rejects_unconstrained_generic` tests in `infer.rs` still
hold because the resolved types in those tests are concrete
`Type::Prim` or stay as fresh vars through to the unification step
that fails for unrelated structural reasons. This option preserves
the per-builtin diagnostic for the concrete-but-wrong case (scalar,
tuple, etc.) but loses the targeted message for the polymorphic-fn
case.

### (b) Pre-unify the synthesized lambda's parameter type with the upstream pipe value's type

`infer_pipe` can inspect each stage before inferring it: if the stage
is a `(fn (params v) body)` with exactly one parameter and no type
annotation, unify `v`'s fresh type with `current_ty` before recursing
into the body. The pipe loop's later `unify(stage_ty, expected)` step
becomes a no-op for the parameter (already bound) and continues to
constrain the return type.

This is local to the pipe handler, mirrors the existing pipe-stage
desugaring contract (the synthesized lambda is always called with the
upstream pipe value as its sole argument), and closes both Finding 4
and `TypeCheck-PipeCast-F1` in a single change. The per-builtin
diagnostics for `copy`, `cast`, `borrow` continue to fire for
genuinely-mistyped inputs (scalar piped into `copy`, non-tensor piped
into `cast`, etc.) because by the time the body is inferred the
parameter has been bound to a real type.

This option also matches the recommendation in the
`TypeCheck-PipeCast-F1` §5 entry of `docs/gap_synthesis.md`.

## Selected fix

Option (b). The change is local to `infer_pipe` in
`crates/chelis-types/src/infer.rs`. Approximate shape:

```rust
for stage in &kids[1..] {
    if let Some(param_var) = synthesized_pipe_stage_param(stage) {
        unify(&param_var, &current_ty, subst).ok();
    }
    let stage_ty = infer_expr(stage, ...);
    // existing unify(stage_ty, Fn([current_ty], ret_tv)) follows.
}
```

`synthesized_pipe_stage_param` is a small helper that returns the
fresh `Type::Var` for a single-parameter unannotated `fn` Deep node
(the exact shape produced by `parse_pipe_stage` and `desugar_pipe_stage`
in `crates/chelis-surf/src/{parser,desugar}.rs`). Multi-arg lambdas
and stages with annotated parameters fall through unchanged.

The fix commit:

* Implements the helper and the pre-unification step.
* Flips the two `#[ignore]` fixtures
  (`pipe_copy_with_statically_typed_tensor_typechecks`,
  `pipe_cast_with_statically_typed_tensor_typechecks`) in
  `crates/chelis-types/tests/pipe_copy_typecheck.rs` to running.
* Updates `docs/gap_synthesis.md` §5 to mark
  `TypeCheck-PipeCast-F1` closed.

The existing `copy_rejects_scalar` and
`copy_rejects_unconstrained_generic` unit tests in `infer.rs` continue
to hold: the first because the resolved type is `Type::Prim`, the
second because there is no enclosing pipe stage to pre-bind the
function's parameter, so the body still sees a fresh var and rejects
on that path.

## Anchors

* Parser: `crates/chelis-surf/src/parser.rs::parse_pipe_stage`,
  `synthesize_bare_unary_builtin_lambda`, `parse_copy`, `parse_realize`.
* Desugarer: `crates/chelis-surf/src/desugar.rs::desugar_pipe_stage`.
* Checker rejection: `crates/chelis-types/src/infer.rs`, lines ~4506
  (`realize`) and ~4523 (`copy`), and `infer_cast` near line ~9036.
* Pipe loop: `crates/chelis-types/src/infer.rs::infer_pipe`.
* Fixtures: `crates/chelis-types/tests/pipe_copy_typecheck.rs`.
* Related entries: `docs/gap_synthesis.md` §5 row
  `TypeCheck-PipeCast-F1`.
