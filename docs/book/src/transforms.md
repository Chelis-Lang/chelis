# Transforms: grad and vmap

`grad` differentiates a function, and `vmap` applies a function across a batch.
Both are compiler transforms written as calls: supply a function to `grad` or
`vmap`, then call the resulting function with its inputs. A bare `grad` or
`vmap` is not a function value. The
[transformation specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/06-transformations.md) defines
their language semantics.

## Differentiate with `grad`

`grad(f)` computes derivatives of a scalar floating result. A floating scalar
such as `f32`, or a tensor with no dimensions such as `tensor[f32]`, can be
that result. Reduce a larger tensor to a scalar before differentiating.
`grad` returns the gradients, without the forward value.

By default, `grad(f)` selects every differentiable parameter. A single selected
parameter produces one gradient directly; multiple parameters produce a flat
tuple in parameter order. Use `wrt` to select named parameters in the order
you write them: `grad(loss, wrt=(weight, bias))` returns the gradient for
`weight` followed by the gradient for `bias`. Parameters left out of `wrt`
still supply values when you call the gradient function, but receive no
gradient in its result.

```chelis-surf
def squared(x: tensor[features, f32]) -> tensor[f32] =
  sum(mul(copy(x), x), 0i32)

def squared_grad(x: tensor[features, f32]) -> tensor[features, f32] =
  grad(squared)(x)
```

The gradient of a tensor parameter has the parameter's shape and precision.
A differentiable input that does not affect the result receives a zero
gradient of the same shape. `grad` can also be applied again to a suitable
scalar gradient function for a second derivative. For an ordered `wrt` example,
see [`examples/grad_wrt_order.ch`](https://github.com/Chelis-Lang/chelis/blob/main/examples/grad_wrt_order.ch).

## Map across a batch with `vmap`

`vmap(f)` adds a batch axis at position zero. To insert it elsewhere, write a
named axis such as `vmap(f, axis=1)`. Each tensor parameter of `f` gains that
axis. Ordinary non-tensor parameters are shared across the batch, as are
values captured by `f`. A scalar `key` parameter is different: it takes a
`tensor[batch, key]` with one key per row. A scalar key shared across rows is
rejected.

```chelis-surf
def process(x: tensor[features, f32]) -> tensor[features, f32] = relu(x)

def batch_process(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] =
  xs |> vmap(process)
```

A reduction inside `process` would reduce its row's data axis, leaving the
new batch axis intact. The runnable source is
[`examples/vmap_relu.ch`](https://github.com/Chelis-Lang/chelis/blob/main/examples/vmap_relu.ch).

## Per-example gradients

Apply `vmap` to a gradient function to get one gradient per batch row:

```chelis-surf
def loss(x: tensor[features, f32]) -> tensor[f32] =
  sum(mul(copy(x), x), 0i32)

def per_example_grad(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] =
  vmap(grad(loss))(xs)
```

This form has an evaluator test against separate row-by-row gradients. A bare
`grad(vmap(f))` does not sum the batch: if the mapped function returns a
tensor, its output must be reduced all the way to a scalar before `grad` can
differentiate it.

## Current execution limits

- A direct, unshadowed top-level function is a reliable named target for
  `grad` and `vmap`. A direct alias of a top-level function, or a local
  binding that shadows one, is rejected as a transform target even when
  `grad` has a `wrt` selector. Inline and locally bound functions have
  supported paths; this restriction does not make every local function an
  error.
- In the evaluator, gradients of ADT arguments keep the executed constructor
  and its fields. A discrete field remains in place as `unit`; a float field
  receives its gradient. A selected parameter with no differentiable float
  field is rejected. C builds currently reject an exported gradient function
  whose parameter is an ADT.
- Differentiation through a `match` works when its selected constructor is
  known during lowering. A runtime-dependent scrutinee or a guarded arm is
  rejected. Scalar `if` conditions can select a branch at runtime; a
  runtime-dependent `if` returning an ADT or tuple is still rejected.
- The evaluator can differentiate a selected local `List` argument. A C build
  currently rejects that form when it cannot reconstruct the List shape; a
  literal or resolved top-level List has a supported path.

These limits describe the current evaluator and C build. The numbered
[transformation specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/06-transformations.md) defines
the broader language rule.
