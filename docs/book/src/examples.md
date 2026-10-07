# Examples

Each program below is complete. Save it under the file name given, then run
`chelis eval --file NAME.ch`; the output shown is what the evaluator prints for
every top-level value. Programs must be in canonical format, so run
`chelis fmt --inplace NAME.ch` first if you retype one.

## Elementwise tensors (`hello_tensor.ch`)

```chelis-surf
module HelloTensor
def activate_scalar(x: f32) -> f32 = gelu(silu(tanh(sigmoid(relu(x)))))
def preserve_integer_scalar(x: i32) -> i32 = round(ceil(floor(x)))
def choose_scalar(a: f32, b: f32) -> f32 = a |> max_elem(b) |> min_elem(1.0f32)
def main[n]() -> tensor[n, f32] = {
  a = to_tensor([1.0, 2.0, 3.0], f32)
  b = to_tensor([4.0, 5.0, 6.0], f32)
  selected = where((a < b), a, b)
  out = add(selected, b)
  out
}
```

```text
main = tensor(shape=[3], data=[5.0, 7.0, 9.0])
```

`where(mask, a, b)` picks from `a` where the mask is true; every `a` is below
its `b`, so `out` is `a + b`. The three scalar definitions are checked but not
printed: only a zero-argument definition or a top-level value is a root the
evaluator prints. `floor`, `ceil`, and `round` are the identity on an integer,
so `preserve_integer_scalar` returns its input unchanged.

## Generic recursion over dtypes (`recursive_cast_targets.ch`)

```chelis-surf
module RecursiveCasts
def step[p: Int](x: p, n: i64) -> p = if eq(n, 0i64) then x else (x |> add(cast(1, p)) |> step(sub(n, 1i64)))
def sum_steps[p: Float](x: p, n: i64) -> p = if eq(n, 0i64) then x else (x |> add(cast(n, p)) |> sum_steps(sub(n, 1i64)))
i32 = step(1i32, 3i64)
i64 = step(9007199254740993i64, 3i64)
f32 = sum_steps(0.5f32, 3i64)
f64 = sum_steps(0.5f64, 3i64)
```

```text
i32 = 4
i64 = 9007199254740996
f32 = 6.5
f64 = 6.5
```

`[p: Int]` and `[p: Float]` bound a dtype variable to the integer or float
dtypes, and `cast(1, p)` produces a literal at whichever dtype the caller
supplies. The `i64` result is exact: 9007199254740993 is above 2^53, where an
`f64` could not represent it, and no step converts through a float.

## Structural operations (`structural.ch`)

```chelis-surf
module Structural
lhs = to_tensor([[1.0, 2.0], [3.0, 4.0]], f32)
rhs = to_tensor([[5.0, 6.0], [7.0, 8.0]], f32)
contracted = einsum("ij,jk->ik", lhs, rhs)
packed = concat([lhs, rhs], 1)
pieces = split(packed, 1, [2i64, 2i64])
table = to_tensor([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], f32)
token_ids = to_tensor([0i64, 2i64])
embed = gather(table, token_ids, 0i32)
base = to_tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], f32)
updates = to_tensor([[1.0, 1.0], [2.0, 2.0]], f32)
scattered = scatter(base, token_ids, updates, 0, "replace")
running = cumsum(embed, 1)
grid = to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], f32)
pooled = reduce_window_max(grid, [2i64, 2i64], [1i64, 1i64])
```

```text
lhs = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])
rhs = tensor(shape=[2, 2], data=[5.0, 6.0, 7.0, 8.0])
contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])
packed = tensor(shape=[2, 4], data=[1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0])
pieces = [tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0]), tensor(shape=[2, 2], data=[5.0, 6.0, 7.0, 8.0])]
table = tensor(shape=[3, 2], data=[10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
token_ids = tensor(shape=[2], data=[0, 2])
embed = tensor(shape=[2, 2], data=[10.0, 11.0, 30.0, 31.0])
base = tensor(shape=[3, 2], data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
updates = tensor(shape=[2, 2], data=[1.0, 1.0, 2.0, 2.0])
scattered = tensor(shape=[3, 2], data=[1.0, 1.0, 0.0, 0.0, 2.0, 2.0])
running = tensor(shape=[2, 2], data=[10.0, 21.0, 30.0, 61.0])
grid = tensor(shape=[3, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
pooled = tensor(shape=[2, 2], data=[5.0, 6.0, 8.0, 9.0])
```

Data is printed in row-major order. `einsum("ij,jk->ik", ...)` is the matrix
product. `concat` along axis 1 places `rhs` to the right of `lhs`, and `split`
with extents `[2, 2]` returns a `List` of the two halves. `gather` with axis 0
takes rows 0 and 2 of `table`; `scatter` with `"replace"` writes `updates`
into those rows of `base`. `cumsum` runs along axis 1 within each row.
`reduce_window_max` with a 2 by 2 window and stride 1 gives
`(3 - 2) / 1 + 1 = 2` outputs per axis. The
[runtime and standard library](stdlib.md) page gives each
operation's signature and argument rules.

## Linear regression with gradients (`linreg.ch`)

```chelis-surf
module LinReg
def predict[n, k](x: tensor[n, k, f32], w: tensor[k, 1, f32], b: tensor[1, f32]) -> tensor[n, 1, f32] = add(matmul(x, w), insert(b, 0, shape(x, 0)))
def loss[n, k](x: tensor[n, k, f32], y: tensor[n, 1, f32], w: tensor[k, 1, f32], b: tensor[1, f32]) -> tensor[f32] = {
  err = sub(predict(x, w, b), y)
  err
  |> mul(err)
  |> mean(1)
  |> mean(0)
}
x = to_tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], f32)
y = to_tensor([[3.0], [2.0], [6.0]], f32)
w = to_tensor([[2.0], [1.0]], f32)
b = to_tensor([1.0], f32)
fit = loss(copy(x), copy(y), copy(w), copy(b))
grads = grad(loss, wrt=(w, b))(x, y, w, b)
```

```text
x = tensor(shape=[3, 2], data=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
y = tensor(shape=[3, 1], data=[3.0, 2.0, 6.0])
w = tensor(shape=[2, 1], data=[2.0, 1.0])
b = tensor(shape=[1], data=[1.0])
fit = 1.3333334
grads.0 = tensor(shape=[2, 1], data=[-1.3333334, -1.3333334])
grads.1 = tensor(shape=[1], data=[-1.3333334])
```

There is no broadcasting, so `predict` widens the one-element bias to `n`
rows explicitly: `insert(b, 0, shape(x, 0))` adds axis 0 with the extent of
`x`'s first axis. The predictions are `3, 2, 4`, so only the last row has an
error (`-2`) and the mean squared error is `4 / 3`. A call to a function you
define consumes its tensor arguments, so the first call passes copies and the
second call can still use `x`, `y`, `w`, and `b`. `grad(loss, wrt=(w, b))`
returns the two gradients in the order `wrt` names them; see
[Transforms](transforms.md).

## Larger definitions

A file of definitions with no top-level value checks but prints nothing:
`chelis eval` reports `warning: input contains only def declarations; nothing
to evaluate`. Check such a file with `chelis check FILE.ch`, then add a
top-level value that calls it. The
[examples directory](https://github.com/Chelis-Lang/chelis/tree/main/examples)
in the repository has more programs in this form, including a transformer
block (`transformer_block.ch`) and batched ReLU with `vmap` (`vmap_relu.ch`).
