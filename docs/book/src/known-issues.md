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

`chelis check` reports a type error on argument 1 of `unsqueeze`, saying that
two adjacent rank spreads cannot be split against a concrete shape.

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
- In place of `stack(xs, axis)`, insert the new axis into each tensor and
  `concat` along it. `map` does this for a `List` of any length, including
  one built at run time:

```chelis-surf
def stack_rows[n](xs: List[tensor[n, f32]]) -> tensor[*, n, f32] = concat(map(fn (x) -> insert(x, 0i32, 1i64), xs), 0i32)
rows = [to_tensor([1.0, 2.0], f32), to_tensor([3.0, 4.0], f32), to_tensor([5.0, 6.0], f32)]
stacked = stack_rows(rows)
```

```text
rows = [tensor(shape=[2], data=[1.0, 2.0]), tensor(shape=[2], data=[3.0, 4.0]), tensor(shape=[2], data=[5.0, 6.0])]
stacked = tensor(shape=[3, 2], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
```

The result's first extent is `*` because it depends on the list's length.
The same program builds with `chelis build` and prints the same output.

## Module-level `dim` declarations

A file with a top-level `dim` declaration passes `chelis check`, but
`chelis eval` and `chelis build` stop with a
`lowered root count mismatch` error. A named axis
needs no declaration: write it in the signature, as in
`def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32])`, and the
checker treats `seq` as a named dimension.
