# Type System Reference

Chelis keeps tensor shape and numeric precision explicit in types. There is no implicit
broadcasting and no implicit precision promotion. Named dimensions are nominal: two
dimensions match only when their names match, so the type checker catches transposition
and shape bugs before any code runs. This page is the reference for the type surface. The
authoritative source is `spec/04-type-system.md`.

## Primitive types

The numeric primitives are:

| Type | Meaning |
|---|---|
| `f32` | 32-bit float |
| `f64` | 64-bit float |
| `bf16` | bfloat16 |
| `f16` | IEEE 754 binary16 |
| `i8` | 8-bit signed integer |
| `i16` | 16-bit signed integer |
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |
| `bool` | boolean |

There are no unsigned integer types. `string` exists as a type for parsing and host work,
and `()` is the unit type.

## Tensor types

A tensor type is written `tensor[dims..., elemtype]`. The element type is the last entry;
everything before it is a dimension. A tensor with no dimensions is a scalar on the device,
written `tensor[f32]`.

```chelis-surf-fragment
tensor[f32]                 -- scalar
tensor[n, f32]              -- one named dimension
tensor[batch, seq, f32]     -- two named dimensions
tensor[64, 64, f32]         -- two literal dimensions
```

The canonical Deep form spells out each dimension and the precision:

```chelis-deep-fragment
(t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
```

Dimensions come in three forms: a named dimension `(d-name {} batch)`, a variable
`(d-var {} a)` introduced by a function's `[...]` clause, and a literal `(d-lit {} 512)`.

## Named dimensions and polymorphism

Dimension matching is strict and element-wise. A named dimension unifies only with the same
name; `batch` does not unify with `seq`. A literal unifies only with the same literal. A
dimension variable unifies with anything and binds.

A function generic over shape uses dimension variables. There are no explicit dimension
arguments; call sites instantiate the variables by unification.

```chelis-surf
def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)
```

In a Deep signature this reads:

```chelis-deep-fragment
(defsig {}
  transpose
  (a b)
  (t-fn {}
    (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
    (t-tensor {} (d-var {} b) (d-var {} a) (t-prim {} f32))))
```

Declared dimension parameters are rigid inside the function body: the body must type-check
for every instantiation, so it cannot couple a result dimension to an unrelated input
dimension.

A wildcard dimension `*` marks a size that is not statically known, for example the axis
produced by a `concat` whose length depends on runtime data. It unifies with anything but
is never generalized. Add an explicit annotation to restore named checking.

### Dtype-family bounds

A binder in the `[...]` clause may name one dtype family, which restricts every dtype it
can be instantiated at. The families are `Float` (the four active floats), `Int` (the four
active signed integers), and `Numeric` (their union). `bool` and `string` belong to no
family.

```chelis-surf-fragment
sig arange[n, p: Int]: p -> p -> tensor[n, p]
sig linspace[n, p: Float]: p -> p -> i64 -> tensor[n, p]
```

Calling `arange` at `f32`, or `linspace` at `i32`, is a `PrecisionMismatch` naming the
required family. The bound is part of the function's type, not a check on the callee name,
so it survives aliases, wrappers, higher-order values, and imports. Two bounded variables
that unify keep the intersection of their families; `Float` and `Int` share nothing, so
identifying one with the other is an error.

A binder with no bound is still an ordinary type variable that admits any type, not only a
dtype. A bound goes on the declaration's `sig` when it has one, and on its `def` otherwise
- never on both.

### Rank polymorphism

A rank variable `..r` is a name-preserving spread over a run of dimensions, so one
definition covers tensors of every rank. Its name is listed in the declaration's complete
binder list; a spread name may not repeat in a single shape.

```chelis-surf
def relu_any_rank[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
```

Spreads can interleave with named anchors, which lets a definition reduce or insert one
named axis while preserving the rest. Reducing over a named axis:

```chelis-surf-fragment
def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] =
  sum(x, seq)
```

The body of a rank-polymorphic definition is restricted to operations whose effect on the
shape can be tracked by name: elementwise and shape-identity operations, named-axis
reductions (`sum`, `mean`, `max_reduce`, `min_reduce`, `prod_reduce`), and named-axis
`insert`. Positional rewriters such as `permute`, `reshape`, and `matmul` are rejected
inside a `..r` body, which is what preserves the named-dimension safety guarantee.

## No broadcasting

Chelis does not broadcast. Operands of an elementwise operation must have identical
dimension lists. Use `insert` to add a dimension explicitly before combining tensors of
different rank, and `expand` to broadcast an existing size-1 axis.

```chelis-surf-fragment
-- tensor[batch, hidden, f32] + tensor[hidden, f32] is a type error.
biased = add(linear, insert(b, 0, batch))
```

`insert`, `expand`, `reshape`, and `permute` are the explicit tools for changing rank and
shape.

## Precision rules

Precision never changes implicitly. Both operands of an arithmetic operation must have the
same precision; the compiler never inserts a cast for you.

```chelis-surf-fragment
-- add(tensor[d, f32], tensor[d, bf16]) is a type error.
sum = add(x, cast(y, f32))
```

`cast(e, p)` is the explicit precision change; it preserves dimensions and changes only the
element type. Integer literals default to `i32` and float literals default to `f32`.
Three things override a default: a literal suffix, a surrounding known element type, or an
explicit `cast`.

```chelis-surf-fragment
a = cast(x, bf16)        -- precision change
b = 1.0f64               -- suffix binds f64
c = cast(3000000000, i64)  -- escape hatch for out-of-i32-range literals
```

The compatibility rules: arithmetic (`add`, `mul`, `sub`, `div`) needs equal numeric
precision; comparison (`cmplt`, `eq`) takes equal numeric precision and yields `bool`;
logical operations (`and`, `or`, `not`) take `bool`; transcendental operations (`exp`,
`log`, `sin`, `cos`, `tan`, `atan`, `sqrt`) take float types only.

`matmul` and `sum` accept an optional accumulator precision, the one place mixed precision
appears. The default accumulator widens `bf16` and `f16` inputs to `f32` for numerical
stability and widens `i8` and `i16` to `i32` for overflow safety; a requested
accumulator must be at least as wide as the operands and the default.

## Function types

A function type is the arrow chain `A -> B -> C`, where the last entry is the return and
the rest are arguments. The arrow is right associative; parenthesize a function-typed
argument.

```chelis-surf-fragment
tensor[n, f32] -> tensor[n, f32] -> tensor[f32]   -- two args, scalar result
(tensor[n, f32] -> tensor[f32]) -> tensor[f32]    -- a function-typed argument
```

In Deep this is a flat `t-fn` whose last child is the return:

```chelis-deep-fragment
(t-fn {}
  (t-tensor {} (d-name {} n) (t-prim {} f32))
  (t-tensor {} (d-name {} n) (t-prim {} f32))
  (t-tensor {} (t-prim {} f32)))
```

## Aggregate types

- Tuples: `(f32, f32)` as a type, `(a, b)` as a value, projected with `.0`, `.1`.
- Algebraic data types: `type Option[a] = | None | Some { value: a }`, with positional or
  record-field payloads. Recursive types refer to themselves by name.
- Records: constructed with `Foo { x: e1, y: e2 }` (punning allowed), read with `e.field`.
- Type aliases: a `type` without variants is transparent and expanded at desugaring.
- Lists: `List[T]` holds rank-uniform elements; a list of tensors fixes one rank for every
  element. The list builtins (`len`, `index`, `append`, `concat`, `range`, `zip`,
  `enumerate`, `map`, `filter`, `fold`, and others) operate on these.

## Effects in types

A function type can carry an effect set in `eff` metadata. In Surf the effect set is the
`! { ... }` suffix on a signature or `def`.

```chelis-surf-fragment
sig predict[n]: tensor[n, f32] -> tensor[n, f32] ! { Random }
```

```chelis-deep-fragment
(t-fn {eff: (effects {} random)}
  (t-tensor {} (d-name {} n) (t-prim {} f32))
  (t-tensor {} (d-name {} n) (t-prim {} f32)))
```

Effect inference runs after type inference. A function's effect set is the union of the
effects of the operations in its body. `Random` comes from `dropout` and `uniform_like` and
the initializers that call them; it is discharged by `with seed(...)`. `IO` is inferred
from host operations such as `print` and file reads. `Resource("device")` marks a region
validated against the build target through `with device(...)`. See
[Effects and Handlers](effects.md).

## Linearity and borrowing

Tensor values are owned by default, with lightweight uniqueness rather than full ownership
and lifetimes. A consuming use makes the binding dead; the compiler inserts copies where a
value fans out and drops where an owned value goes out of scope, so you usually do not write
either by hand.

A read-only borrow is a reference type, written `&T` in a signature and `&x` at a use site.
A borrow leaves the owned binding live and is the idiomatic way to pass a tensor to a
read-only operation.

```chelis-surf-fragment
def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
```

```chelis-deep-fragment
(t-ref {} (t-tensor {} (d-name {} batch) (t-prim {} f32)))
```

Passing an owned value where a borrow is expected auto-borrows. Passing a borrow where an
owned value is expected is an error unless you write `copy(x)`, which produces a fresh owned
value from an owned or borrowed input. Borrows cannot be stored in aggregates, returned, or
captured by closures. The borrow target must be a tensor or a tensor-carrying value, where a
type is tensor-carrying when one of its fields contains a tensor, transitively through
tuples, ADTs, and references.

## Inference

Type checking is Hindley-Milner inference with tensor and dimension extensions. Annotations
are optional; the checker infers literal types, dimension variables at call sites, and
effect sets, and verifies any annotation you do supply. What you must still state: the
precision argument to `cast`, the dimension and precision parameters in a `[...]` clause, an
annotation to pin a wildcard dimension back to a name, and a suffix or cast for a literal
outside the default precision range.

The pipeline order is parse, desugar, type inference and checking, effect inference and
checking, linearity checking, then lowering.
