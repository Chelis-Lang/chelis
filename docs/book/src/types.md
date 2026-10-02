# Type System Basics

Chelis makes tensor dimensions and numeric precision explicit. The checker
rejects an axis order that conflicts with a declared type and an arithmetic
operation whose operands have different precision. See the
[Type System Reference](type-reference.md) for the full type surface.

## Tensor types

A tensor type lists its dimensions followed by its element type:

```chelis-surf-fragment
tensor[f32]              -- rank-zero tensor
tensor[n, f32]           -- one dimension
tensor[batch, seq, f32]  -- two named dimensions
```

A rank-zero `tensor[f32]` is distinct from an `f32` host scalar. A dimension
variable such as `n` is declared in a function's `[...]` clause. Distinct named
dimensions, such as `batch` and `seq`, do not match each other just because
they have the same extent.

```chelis-surf
def add_vec[n](x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

Both arguments must have the same dimension, and the result keeps it. Call
sites bind `n` from their inputs, so the function works with different vector
lengths.

## Shapes and precision

- Elementwise operations do not broadcast implicitly; their tensor dimension
  lists must match. Use `insert` to add an axis, `expand` to repeat a size-one
  axis, `reshape` to specify a new shape, or `permute` to reorder axes.
- Arithmetic does not promote precision implicitly. Use `cast(x, f32)` when a
  conversion is intended.
- An unsuffixed integer literal has type `i32`; an unsuffixed float literal has
  type `f32`. A suffix such as `1.0f64` selects another dtype explicitly.

An empty list supplies no element values from which to determine a tensor's
dtype. Give the list an element type before converting it:

```chelis-surf
empty_values: List[f64] = []
empty_tensor = to_tensor(empty_values)
```

Here `empty_tensor` has shape `[0]` and dtype `f64`. An unconstrained
`to_tensor([])` is rejected by the checker.

## Dtype parameters

A dtype bound restricts the dtypes a parameter accepts. A bound is either a
family or an explicit set. `Int` accepts integer dtypes; `Float` accepts
floating dtypes; `Numeric` accepts both:

```chelis-surf
def double_ints[p: Int](x: p) -> p = add(x, x)
```

An explicit set admits exactly the dtypes it lists, which is how a declaration
excludes a member its family would admit:

```chelis-surf
def widen_only[p: {f32, f64}](x: p) -> p = add(x, x)
```

For named dimensions, rank polymorphism, generic casts, ownership, the
difference between the two bound forms, and the corresponding Deep forms, see
the [Type System Reference](type-reference.md).
For effects in function types, continue to [Effects](effects.md).
