# Known issues

## Tensor construction helpers

`Std.Tensor.Construct` exports `stack`, `squeeze`, and `unsqueeze`, but a
call with an ordinary concrete tensor fails `chelis check`. Their signatures
place the axis between two rank spreads, `tensor[..pre, ..post, p]`, and the
checker cannot decide where `pre` ends and `post` begins in a concrete shape:

```chelis-surf-fragment
import Std.Tensor.Construct (unsqueeze)
y = unsqueeze(to_tensor([1.0, 2.0], f32), 0i32)
```

```text
`Std.Tensor.Construct.unsqueeze` argument 1; two adjacent rank spreads cannot be split against a concrete shape; the boundary between them is undetermined (outside the decidable fragment, spec/04-type-system.md §4.5.3)
```

The built-in shape operations do the same jobs with an explicit axis or
target shape:

```chelis-surf
module Recipes
row = insert(to_tensor([1.0, 2.0, 3.0], f32), 0, 1i64)
flat = reshape(to_tensor([[1.0, 2.0, 3.0]], f32), [3i64])
stacked = concat([insert(to_tensor([1.0, 2.0], f32), 0, 1i64), insert(to_tensor([3.0, 4.0], f32), 0, 1i64)], 0)
```

```text
row = tensor(shape=[1, 3], data=[1.0, 2.0, 3.0])
flat = tensor(shape=[3], data=[1.0, 2.0, 3.0])
stacked = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])
```

- In place of `unsqueeze(x, axis)`, use `insert(x, axis, 1i64)`, which adds an
  axis of extent one at a constant position.
- In place of `squeeze(x, axis)`, use `reshape(x, [...])` with the target
  extents written out as `i64` values.
- In place of `stack(xs, axis)` over a fixed set of tensors, insert the new
  axis into each and `concat` along it. A `List` whose length is known only
  at run time has no equivalent recipe.
