# Transforms: grad and vmap

Transforms are compiler features, not library functions or macros. They are DAG-to-DAG
rewrites: the compiler takes a function, rewrites its RISC DAG, and produces a new function.
After expansion the program contains only RISC primitives. A transform must always be
applied; a bare `grad` with no function is a parse error. Transforms compose. The
authoritative source is `spec/06-transformations.md`.

## Chelis 0.19 core-transform fence

For the 0.19 core promise, write named transforms against a direct, unshadowed
top-level function declaration, such as `grad(loss)` or `vmap(process)`.
The checker rejects aliases of a top-level function at either module or local
scope, and a local binding that shadows a top-level target, rather than
silently selecting a different callable. An inline or locally bound `vmap`
lambda remains supported when every parameter has an explicit type; an
untyped parameter is rejected, including through local aliases, because it
could be inferred from the unsliced input rather than the mapped slice.

These are current supported-fragment fences, not changes to the language
semantics in the numbered specification. The related launch rows are
[#1887](https://github.com/Chelis-Lang/chelis/issues/1887),
[#1952](https://github.com/Chelis-Lang/chelis/issues/1952), and
[#1954](https://github.com/Chelis-Lang/chelis/issues/1954), with the
local-lambda fence completed by
[#2109](https://github.com/Chelis-Lang/chelis/issues/2109).

## grad

`grad(f)` is reverse-mode differentiation. It produces a new function from `f`'s arguments to
their gradients. The output of `f` must be a scalar floating result; reduce to a scalar
first if it is not. The gradient of a single tensor parameter has the same type as that
parameter, and a multi-parameter function yields a flat tuple of gradients.

```chelis-surf-fragment
grad(loss_fn)
```

```chelis-deep-fragment
(grad {} (var {} loss_fn))
```

By default `grad(f)` differentiates with respect to every differentiable parameter. The
`wrt` argument restricts it to named parameters, and the result holds one entry per listed
parameter in the order listed. Apply the gradient function to get the values:

```chelis-surf-fragment
(dw, db) = grad(loss_fn, wrt=(w, b))(w, b)
```

`grad` returns gradients only, not the forward value alongside them. It composes with
itself for higher derivatives: `grad(grad(f))` is the second derivative.

Three control-flow/ADT slices extend the differentiated surface beyond flat
tensors (`spec/06-transformations.md` §2.10.1 has the full contract and
limits):

- A `match` whose scrutinee is a compile-time-known constructor value (a
  constructor literal, a record construction, or an ADT-typed parameter of the
  differentiated function) resolves to its taken arm at lowering time, so such
  bodies differentiate in both the eval and compiled lanes. A runtime scrutinee
  or a guarded arm is still rejected with a diagnostic naming the construct.
- `grad(f)` over an ADT argument whose fields are all float tensors returns a
  gradient with the same constructor shape, one gradient per field (eval lane).
  The ADT argument may sit alongside plain tensor arguments —
  `grad(model_forward, wrt=params)(x, params)` returns the field-wise gradient
  struct for `params`, and the default (all-argument) form returns the
  per-target tuple whose ADT slot is that struct and whose tensor slots are
  bare gradients. An argument that does not influence the output gets a shaped
  zero in its slot (a zero tensor, or a zero-filled gradient struct), so the
  tuple keeps full arity and every gradient stays in its own `out.0..out.N`
  position. Mixed types (a non-float field in any variant, even a variant
  other than the constructed one), pure enums with no fields, and compiled-lane
  ADT-param gradient exports are rejected with named diagnostics.
- An `if` whose condition const-folds at lowering time prunes to the taken
  branch, whatever its type (chelis#620) — so the eps fail-guard idiom
  (`if eps <= 0.0 then fail(...) else body`) and constructor- or list-valued
  branches differentiate when the guard's inputs are literal-rooted. A
  recursive builder whose base case prunes statically
  (`if k >= n then [] else concat([row], recurse)`) unrolls under grad, with
  a loud diagnostic at the depth cap (512 per callee, 1024 total) for chains
  static pruning cannot bound; the unrolled list value flows through list
  append and tensor `concat`. Linearity copies over params ADTs lower
  field-wise, which is what makes the curried single-argument closure
  `grad(fn (p) -> loss(p, x, y))(params)` work.

A runtime-scrutinee `match` in a differentiated body stays rejected, and so
does an ADT- or tuple-valued `if` branch under a genuinely runtime condition
— both pend the `RiscOp::Select` blend primitive (tracked in chelis#618).

## vmap

`vmap` vectorizes a per-example function over a batch dimension. It is a DAG rewrite, not a
loop: every operation is lifted to run over the added axis. The `axis` argument names the
integer position where the batch dimension is inserted, defaulting to `0`.

```chelis-surf
def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)

def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] =
  xs |> vmap(process)
```

```chelis-deep-fragment
(vmap {} (var {} process) (lit {type: (t-prim {} i32)} 0))
```

Each tensor argument of the wrapped function gains the batch dimension; non-tensor arguments
are shared across the batch. Reductions inside the wrapped function still reduce over their
original named axis, so the batch dimension passes through untouched.

## Composing transforms

`vmap(grad(f))` computes per-example gradients: each example in the batch gets its own
gradient vector, which is what per-example gradient clipping needs. This is not the same as
`grad(vmap(f))`, which would be the gradient of the sum over the batch. To differentiate a
vmapped function, reduce its result to a scalar first:

```chelis-surf-fragment
grad(fn (xs) -> sum(vmap(process)(xs), 0))
```

When you are unsure whether a particular composition is supported, write a small program and
let the compiler tell you. The supported surface is defined by the compiler and its tests.
