# Type System Basics

Chelis keeps tensor dimensions and precision explicit. The type checker reads the shape and
precision of every tensor from its type, so a transposition or a precision mismatch is a
compile error rather than a wrong number at runtime. This page covers the ideas you meet
first. The full surface is in the [Type System Reference](type-reference.md).

## Primitive ideas

- No implicit precision promotion. Mixed precision is an error; change it with `cast`.
- No implicit broadcasting. Shapes must match; change rank with `expand`, `reshape`, or
  `permute`.
- Named tensor dimensions are nominal. `batch` and `seq` match only by name, not by size.
- Integer literals default to `int32`, float literals to `f32`.

## Tensor types

A tensor type lists its dimensions and ends with the element type. A scalar on the device
is a tensor with no dimensions.

```chelis-surf-fragment
tensor[f32]              -- scalar
tensor[n, f32]           -- one named dimension
tensor[batch, seq, f32]  -- two named dimensions
```

The canonical Deep form names each dimension and the precision:

```chelis-deep-fragment
(t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
```

## A small typed program

```chelis-surf
def add_vec(x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

Both arguments share the named dimension `n`, so the checker requires the two inputs to
have the same length and gives the result that same length.

## Reading the Deep shape

A `def` with annotations desugars to a signature plus the function. The signature is a flat
`t-fn` whose last child is the return type.

```chelis-deep-fragment
(defsig {}
  add_vec
  (t-fn {}
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))
    (t-tensor {} (d-name {} n) (t-prim {} f32))))
```

## Dimension polymorphism

Names in the `[...]` clause before the parameters are dimension variables. A call site
binds them by unification, so one definition serves every concrete shape.

```chelis-surf
def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)
```

## Where to go next

- Named dimensions, rank polymorphism, and the no-broadcasting rule:
  [Type System Reference](type-reference.md).
- Precision rules and the accumulator parameter: [Type System Reference](type-reference.md).
- Effects in signatures: [Effects and Handlers](effects.md).
- Ownership and borrowing: [Type System Reference](type-reference.md).
