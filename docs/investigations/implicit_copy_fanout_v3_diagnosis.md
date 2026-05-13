# Implicit-copy fan-out v3: Shapes A and B diagnosis

Two implicit-copy fan-out shapes remain after PR #29 (v1) and PR #60
(V2-F4). The fixtures pinned in
`crates/chelis-ir/tests/implicit_copy_fanout_v3.rs` reproduce both.

## Anchor note (vs original plan brief)

The plan-mossy-meteor brief named
`crates/chelis-ir/src/lower.rs::insert_copy_nodes_for_consuming_fanout`
as the anchor for both shapes. Reproduction confirmed both shapes fail
**before** lowering runs:

- Shape A: fails at typecheck (`def 'identity_dim' body doesn't match
  declared signature`). The body's inferred type is `Ref(T)` and the
  declared return is owned `T`. Unify rejects.
- Shape B: fails at the linearity check (`UseAfterConsume`). The two
  grad-app sites are treated as Structural consumes of every arg, and a
  trailing borrow-read of an already-consumed arg trips
  `read_or_error`.

Both fixes therefore live upstream of the inserter:

- Shape A is fixed in the type checker (return-position coercion
  rule).
- Shape B is fixed in the linearity checker (grad-app arg classification).

The lower-time inserter at `lower.rs:434` already handles the
fork-once-for-each-extra-consume case correctly for app-arg fan-out
when the linearity checker permits the program to reach lowering. The
v3 fixes do not change inserter behavior.

## Shape A: borrow-to-owned at return position

### Failure mode

`def identity_dim[a](x: &tensor[a, f32]) -> tensor[a, f32] = x`
desugars to:

```
(defsig identity_dim (t-fn (t-ref (t-tensor a f32)) (t-tensor a f32)))
(def identity_dim (fn (params (meta x :type (t-ref ...))) (var x)))
```

`chelis_types::infer::check_top_level` (`crates/chelis-types/src/infer.rs:4253-4302`)
inspects the def:

1. Looks up `declared_ty` from env (the defsig). Returns
   `Type::Fn([Ref(Tensor)], Tensor)`.
2. Calls `infer_expr(&body)` where body is `(fn ...)`. `infer_fn`
   (`infer.rs:8366`) infers params from the `:type` metadata, infers
   the body `(var x)` as `Ref(Tensor)`, and returns
   `Type::Fn([Ref(Tensor)], Ref(Tensor))`.
3. Calls `unify(body_ty, decl_ty, subst)`
   (`infer.rs:4287`). At the return position the unifier sees
   `Ref(Tensor) ~ Tensor` and falls through to the catch-all mismatch
   arm (`crates/chelis-types/src/unify.rs:353-356`).
4. Pushes `CheckErrorKind::TypeMismatch` with the message `def
   'identity_dim' body doesn't match declared signature`.

The existing `auto_borrow_call_arg_types`
(`infer.rs:7337`) handles the opposite coercion at argument positions:
when an arg position expects `Ref(T)` and the actual type is `T`, the
checker silently lifts the actual into `Ref(T)`. There is no mirror
helper for the return position.

### Existing v2 inserter behavior

The lower-time inserter is unaffected here because the program never
reaches lowering. Once the typecheck accepts an implicit copy at the
return position, the lowered body's bare `(var x)` resolves to a
`Load(x)` whose type is `Ref(T)`. The lowerer must materialize an
owned value at the def root; this is the same shape lowering already
handles for explicit `(copy x)`.

### Chosen fix

The auto-coerce rule lives in `unify` (or, equivalently, in
`infer_fn` post-body). The minimal change is in `unify`: when the
expected type is owned `T` and the actual type is `Ref(T)`, treat it
as success. This mirrors `auto_borrow_call_arg_types`'s structural
treatment of the opposite direction.

Adding the rule to `unify` itself is the surgical fix, but it is also
overly permissive: any `Ref(T) ~ T` constraint anywhere would silently
succeed, which could mask real borrow/owned mismatches in expression
positions. The narrower fix is at the def-body unify site
(`infer.rs:4287`): when `unify(body_ty, decl_ty)` fails, try a
relaxed unify that, at each function return position, accepts
`Ref(T) ~ T` if and only if the body is a bare `(var ...)`
reference (matching the "body is a bare reference" shape called out
in the brief). For broader bodies the existing mismatch error stands;
those cases are out of v3 scope and should be filed as a follow-on.

Lowering already handles the case: `lower_fn`'s body lowering reads
the var's load node, and the def-root machinery treats the produced
value as owned. The auto-copy at return position is a typecheck-only
relaxation; no IR-level Copy is needed because the param itself is a
`Ref(T)` load whose underlying tensor is what the caller passes by
borrow. The owned return value at the def boundary is the same tensor
without forking storage. The semantics line up: an `&T` parameter +
owned return implies the caller's borrow lifetime extends through
the call, which is what `auto_borrow_call_arg_types` already arranges
on the caller side.

### Fix location

- `crates/chelis-types/src/infer.rs`, in `check_top_level`'s def-body
  unify (lines 4286-4299): on `unify` failure, retry with a
  return-position-only relaxation that accepts `Ref(T) ~ T`.

### Sibling sweep

Other body shapes that surface the same coercion gap:

- `def f(x: &T) -> T = x` — the canonical case (Shape A literal).
- `def f(x: &T) -> T = (if ... then x else x)` — body's if/match arms
  are `Ref(T)`. Out of v3 scope; document as a candidate for a
  follow-on entry.
- `def f(x: &T) -> T = let y = x in y` — body's tail expression
  resolves to a `Ref(T)`. Out of v3 scope; same candidate.

The narrow Shape A fix (bare-`var` body) covers the hello-chelis
`dimpoly.ch::identity_dim` repro path. Broader coercion shapes are
deferred behind documentation as a §5 candidate.

## Shape B: fan-out across grad-call sites with trailing borrow

### Failure mode

The hello-chelis `linreg.ch::sgd_step` shape:

```
dw = grad(mse_loss, wrt=w)(x, y, w, b)
db = grad(mse_loss, wrt=b)(x, y, w, b)
new_w = sub(w, mul(expand(expand(lr_t, 0, 64), 1, 1), dw))
new_b = sub(b, mul(lr_t, db))
```

The two grad-app call sites pass `w` and `b` as args. The trailing
`sub(w, ...)` and `sub(b, ...)` are borrow-reads (`sub` is in
`builtin_arg_is_borrowed`).

The linearity checker (`crates/chelis-types/src/linearity.rs`) walks:

1. `dw = grad(...)(x, y, w, b)` lowers to
   `(app (grad mse_loss ...) (var x) (var y) (var w) (var b))`.
   `check_app` (`linearity.rs:590`) processes each arg via
   `arg_is_borrowed(kids.first(), builtin=None, arg_index, scope)`.
   `arg_is_borrowed` (`linearity.rs:1084`) consults the func's type:
   `grad(mse_loss, wrt=w)` has inferred type `Type::Fn(args, w_ty)`
   where `args` is `mse_loss`'s parameter list. Since `mse_loss`'s
   params are owned (no `&`), `type_expr_is_ref` returns false for
   every position. The args are routed via `consume_var_expr` with
   `app_site` (Structural).
2. `db = grad(...)(x, y, w, b)` consumes the same names again. Each
   hits the `Some(BindingState::Consumed(_))` arm at
   `linearity.rs:951-955`, which falls through silently because the
   implicit-copy pass is expected to handle this in IR.
3. `trailing = sub(w, dw)`: `sub` is in `builtin_arg_is_borrowed`, so
   `arg_is_borrowed` returns true. The checker calls
   `read_var_expr` → `read_or_error`. `read_or_error`
   (`linearity.rs:971`) sees `w`'s binding state is `Consumed` with
   `ConsumeKind::Structural`. Per the PR #29 v1 tolerance only
   `ConsumeKind::Aliasing` consumes permit a follow-up borrow-read.
   Structural consumes do not; the checker pushes `UseAfterConsume`.

The spec's "Copy Insertion" section says the compiler inserts copies
for source-level consuming fan-out. The lower-time inserter would
correctly fork `w` between the two grad calls. But the linearity
checker rejects the program before lowering, because it interprets
the grad-app site as a Structural (destroy-the-arg) consume rather
than an observational (read-the-arg) one.

### Existing v2 inserter behavior

If the program were permitted to reach lowering, the implicit-copy
pass at `lower.rs::insert_copy_nodes_for_consuming_fanout` would do
the right thing for two grad-app fan-out: it would see `w`'s NodeId
consumed twice (once per grad call) and insert one Copy. The
trailing `sub(w, ...)` is a borrow-read at the IR level
(`sub` is in `builtin_arg_is_borrowed`), and the lowered DAG shares
the same Load(w) NodeId between all reads.

The blocker is the linearity gate. Shape B's fix lifts the
classification, not the inserter.

### Chosen fix

`grad(f, ...)`-app and `vmap(f, ...)`-app call sites are observational
at the source level. The reverse-mode AD lowering at
`lower.rs::lower_grad_callable_with_nodes` (lines 2960-3059) treats the
captured args as Load nodes inside the synthesized closure; the AD
splice (`splice_dag`) reads the caller's NodeId without destroying
it. The forward and backward DAGs both read from the same Load. From
the linearity checker's perspective every arg to a grad-app should
be treated as a borrow.

The narrowest fix is in `arg_is_borrowed` (`linearity.rs:1084`): when
the func at the call site is `(grad ...)` or `(vmap ...)` or
`(vmap (grad ...))`, return true for every arg position. This matches
the existing `builtin_arg_is_borrowed` table's role for "observational"
primitives (`print`, `to_string`, `rank`, `shape`, `numel`, etc.).

The fix preserves the existing v2 inserter's behavior for normal
fan-out (the inserter still inserts copies when needed at the IR
level), and it preserves all other linearity diagnostics
(consume-after-consume on real user-fn calls, structural ownership
violations, borrow-of-borrow chains).

### Fix location

- `crates/chelis-types/src/linearity.rs`, in `arg_is_borrowed`
  (lines 1084-1098): add a tag-based clause for `grad` and `vmap`
  callees that returns true for every arg index.

### Sibling sweep

Other source-level closure-synthesizing forms in the spec vocabulary:

- `jit(f)(args)`: `jit` is a build-target marker — it lowers `f`
  through the same IR-callable path but the call site is just `f`'s
  signature. `arg_is_borrowed` should follow `f`'s param types as
  today. No change needed.
- `par(...)`: this is a parallel-evaluation block, not a function
  application. Args are evaluated as side-effecting statements
  (effects checker handles ordering). Not affected.
- `vmap(f)(args)`: same observational nature as `grad`. Same fix
  applies. Confirm by mirror reasoning: `lower.rs::lower_vmap_grad_callable_app`
  and the surrounding vmap lowering treat captured args as Load
  nodes inside the synthesized vmap closure. Linearity should mirror
  this.
- `vmap(grad(f))(args)`: chained. The `(app (vmap (grad ...)) args)`
  resolves through the same `arg_is_borrowed` path; the outermost
  callee is `(vmap ...)`. If both vmap and grad get the arg-is-borrowed
  treatment, the chain is covered.

The v3 fix covers `grad` and `vmap`. `jit` and `par` are not affected
(documented above).

## §5 candidates surfaced

These are surfaced for orchestrator consideration; agents do not file
§5 entries directly.

1. **Broader Shape A coercion**: `def f(x: &T) -> T = if c then x else y`
   and `let y = x in y` forms. Today these fail with the same
   TypeMismatch. v3 narrowly handles `body = (var x)` (bare-ref); the
   broader case needs a body-tree walk to identify return-position
   sub-expressions. Out of v3 scope.

2. **`vmap(...)`-app linearity classification**: if Shape B's fix is
   applied symmetrically (grad and vmap), this is closed. If the fix
   is applied only to grad initially, the vmap mirror should be
   filed.
