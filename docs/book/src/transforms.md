# Transforms: grad and vmap

`grad` differentiates a function, and `vmap` applies a function across a batch.
Both are compiler transforms written as calls: supply a function to `grad` or
`vmap`, then call the resulting function with its inputs. A bare `grad` or
`vmap` is not a function value.

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
def squared[n](x: tensor[n, f32]) -> tensor[f32] = x |> mul(x) |> sum(0i32)
def squared_grad[n](x: tensor[n, f32]) -> tensor[n, f32] = grad(squared)(x)
g = [1.0, 2.0, 3.0] |> to_tensor(f32) |> squared_grad
```

```text
g = tensor(shape=[3], data=[2.0, 4.0, 6.0])
```

`[n]` declares the dimension variable `n`, so both functions accept a vector
of any length. The gradient of `sum(x * x)` is `2x`. The gradient of a tensor parameter has
the parameter's shape and precision. A differentiable input that does not
affect the result receives a zero gradient of the same shape. `grad` can also
be applied again to a suitable scalar gradient function for a second
derivative.

With `wrt`, the result order follows the `wrt` list, not the parameter list:

```chelis-surf
def loss(weight: f32, bias: f32) -> f32 = ((weight * 3.0) + (bias * bias))
grads = grad(loss, wrt=(bias, weight))(2.0f32, 5.0f32)
```

```text
grads.0 = 10.0
grads.1 = 3.0
```

`grads.0` is the derivative for `bias` (`2 * 5`) and `grads.1` the derivative
for `weight` (`3`). The call still passes both arguments in parameter order.

## Softmax gradients

The softmax adjoint uses its forward output `y` and incoming cotangent `g`:
`y * (g - sum(g*y, axis))`. The reduction is finalized into the operand dtype
before subtraction. Differentiation preserves this rule for f16, bf16, f32 and
f64; it does not differentiate the stabilizing maximum used by the forward
graph. This program computes the gradient of a weighted sum of softmax
outputs with `grad` and again from the formula, where `g` is the weight
vector `w`:

```chelis-surf
module Examples.SoftmaxAdjoint
def loss(x: tensor[4, f32]) -> tensor[f32] = sum(mul(softmax(x, 0i32), to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])), 0i32)
x = to_tensor([0.1f32, 0.7f32, 1.3f32, 2.9f32])
y = softmax(x, 0i32)
w = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])
actual = grad(loss)(x)
expected = mul(y, sub(w, insert(cast(sum(mul(w, y), 0i32), f32), 0i32, 4i64)))
```

```text
x = tensor(shape=[4], data=[0.1, 0.7, 1.3, 2.9])
y = tensor(shape=[4], data=[0.04427348, 0.08067155, 0.14699313, 0.72806185])
w = tensor(shape=[4], data=[1.0, 2.0, 3.0, 4.0])
actual = tensor(shape=[4], data=[-0.1132889, -0.12575431, -0.08214614, 0.3211893])
expected = tensor(shape=[4], data=[-0.1132889, -0.12575431, -0.08214614, 0.3211893])
```

The two agree to the last bit.

## Map across a batch with `vmap`

`vmap(f)` adds a batch axis at position zero. To insert it elsewhere, write a
named axis such as `vmap(f, axis=1)`. Each tensor parameter of `f` gains that
axis. Ordinary non-tensor parameters are shared across the batch, as are
values captured by `f`. A scalar `key` parameter is different: it takes a
`tensor[batch, key]` with one key per row. A scalar key shared across rows is
rejected.

```chelis-surf
def process[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)
def batch_process[b, n](xs: tensor[b, n, f32]) -> tensor[b, n, f32] = xs |> vmap(process)
ys = [[-1.0, 2.0], [3.0, -4.0]] |> to_tensor(f32) |> batch_process
```

```text
ys = tensor(shape=[2, 2], data=[0.0, 2.0, 3.0, 0.0])
```

`process` is written for one row of `n` values; `vmap(process)` applies it to
each of the `b` rows. A reduction inside `process` would reduce its row's
data axis, leaving the new batch axis intact.

## Per-example gradients

Apply `vmap` to a gradient function to get one gradient per batch row:

```chelis-surf
def loss[n](x: tensor[n, f32]) -> tensor[f32] = x |> mul(x) |> sum(0i32)
def per_example_grad[b, n](xs: tensor[b, n, f32]) -> tensor[b, n, f32] = vmap(grad(loss))(xs)
gs = [[1.0, 2.0], [3.0, 4.0]] |> to_tensor(f32) |> per_example_grad
```

```text
gs = tensor(shape=[2, 2], data=[2.0, 4.0, 6.0, 8.0])
```

Row `i` of the result is the gradient of `loss` at row `i` of `xs`. A bare
`grad(vmap(f))` does not sum the batch: if the mapped function returns a
tensor, its output must be reduced all the way to a scalar before `grad` can
differentiate it.

## Execution limits

- A direct, unshadowed top-level function is a reliable named target for
  `grad` and `vmap`. A direct alias of a top-level function, or a local
  binding that shadows one, is rejected as a transform target even when
  `grad` has a `wrt` selector. Inline and locally bound functions have
  supported paths; this restriction does not make every local function an
  error.
- In the evaluator, gradients of ADT arguments keep the executed constructor
  and its fields. A discrete field remains in place as `unit`; a float field
  receives its gradient. A selected parameter with no differentiable float
  field is rejected. C builds reject an exported gradient function
  whose parameter is an ADT.
- Differentiation through a `match` works when its selected constructor is
  known during lowering. A runtime-dependent scrutinee or a guarded arm is
  rejected. Scalar `if` conditions can select a branch at runtime; a
  runtime-dependent `if` returning an ADT or tuple is still rejected.
- The evaluator can differentiate a selected local `List` argument. A C build
  rejects that form when it cannot reconstruct the List shape; a
  literal or resolved top-level List has a supported path.

The limits above describe evaluator and C-build behavior; the
[transformation specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/06-transformations.md)
has the complete language rules for `grad` and `vmap`.
