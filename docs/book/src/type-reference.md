# Type System Reference

Chelis records tensor shape and element dtype in types. It does not implicitly
broadcast tensors or promote numeric operands. Dimension names preserve axis
identity: distinct names do not unify, while a literal extent can satisfy a
named dimension at a call site. This page covers the type surface and its
checking rules. The [type system specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/04-type-system.md) defines the full semantics.

## Primitive types

The numeric primitive types are:

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

There are no unsigned integer types. The other primitive types are `bool`,
`string`, and `key`; `unit` is the unit type and `()` its value. Strings support
host work and cannot be tensor elements.

`key` is the non-numeric type of a random key: `key_from_seed(42i64)` makes one, and
`tensor[n, key]` holds `n` of them. A key has no arithmetic and no cast, and each key is used
at most once on every path; see [Effects](effects.md#random-keys).

## Tensor types

A tensor type is written `tensor[dimensions..., element_type]`: the last entry
is the element type and each earlier entry is a dimension. Supported elements
include numeric types, `bool`, and `key`, but each operation has its own element
type rules. A tensor with no dimensions is a rank-zero tensor, such as
`tensor[f32]`.

```chelis-surf-fragment
tensor[f32]                 -- scalar
tensor[n, f32]              -- one named dimension
tensor[batch, seq, f32]     -- two named dimensions
tensor[64, 64, f32]         -- two literal dimensions
tensor[batch, bool]         -- boolean elements
tensor[batch, key]          -- random keys
```

Dimension positions can hold a declared name such as `batch`, a variable
introduced in `[...]`, a nonnegative literal such as `512`, or the wildcard
`*` for an unknown extent. A `..r` spread stands for a run of dimensions
in a rank-polymorphic signature. See the [Deep syntax specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/03-deep-syntax.md) for their Deep forms.

## Named dimensions and polymorphism

Dimension lists match in order. `batch` does not unify with `seq`, and two
different literal extents do not unify. A named dimension can unify with a
literal extent at a call site; the literal supplies a concrete size, while the
name remains available in the function's type. A dimension variable unifies
with a compatible dimension and binds.

A function generic over shape uses dimension variables. There are no explicit dimension
arguments; call sites instantiate the variables by unification.

```chelis-surf
def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)
```

Declared dimension parameters are rigid inside the function body: the body must type-check
for every instantiation, so it cannot couple a result dimension to an unrelated input
dimension.

A wildcard dimension `*` marks a size that is not statically known, for example
an axis produced by a `concat` whose length depends on runtime data. It can
unify with another dimension but is never generalized. A declared result
dimension can restore a named shape claim, with a runtime equality check when
the size cannot be proved statically.

### Dtype bounds

A binder in the `[...]` clause may carry a dtype bound, which restricts every dtype it can
be instantiated at. A bound is written either as a family name or as an explicit dtype set.

The families are `Float` (the four active floats), `Int` (the four active signed integers),
and `Numeric` (their union). `bool` and `string` belong to no family.

```chelis-surf
def add_ints[p: Int](x: p, y: p) -> p = add(x, y)
def add_floats[p: Float](x: p, y: p) -> p = add(x, y)
def add_wide[p: {f32, f64}](x: p, y: p) -> p = add(x, y)
```

An explicit set admits exactly the dtypes it lists. The difference from a family is not
cosmetic: a family denotes whatever the active dtype set admits into it, so it widens if a
dtype is activated later, while a set never does. A set is therefore the way to exclude a
member a family would admit - which matters because a bound's literals must be valid at
*every* admissible instantiation. Under `p: Float`, `cast(1000000.0, p)` is rejected,
because `Float` admits `f16` and the literal rounds to infinity there; under
`p: {f32, f64}` the same literal is fine.

A one-member bound is written `{f64}`. There is no bare-dtype spelling: `p: f64` is a
syntax error, because a binder restricted to one dtype is a set of one rather than a type
ascription. The formatter prints a set's members in the order the active dtype set
declares them, not the order you wrote them.

Calling `add_ints` at `f32`, or `add_floats` at `i32`, is a `PrecisionMismatch` naming the
required family. The bound is part of the function's type, not a check on the callee name,
so it survives aliases, wrappers, higher-order values, and imports. Two bounded variables
that unify keep the intersection of their families; `Float` and `Int` share nothing, so
identifying one with the other is an error.

A binder with no dtype-family bound can stand for a general type, subject to
the other type rules. In particular, a function's generic type parameter
cannot be instantiated with a key-carrying type: pass `key` or
`tensor[n, key]` through an explicitly typed parameter instead. Put a
dtype-family bound on a declaration's `sig` when it has one, or on its `def`
otherwise, never both.

### Rank polymorphism

A rank variable `..r` is a name-preserving spread over a run of dimensions, so one
definition covers tensors of every rank. Its name is listed in the declaration's complete
binder list; a spread name may not repeat in a single shape.

```chelis-surf
def relu_any_rank[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
```

Spreads can surround named axes, letting a definition reduce or insert an axis
while preserving the others. To reduce a named `seq` axis:

```chelis-surf
dim seq
def reduce_seq[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)
```

Rank-polymorphic bodies admit operations whose shape effect can be tracked by
name, including elementwise operations, named-axis reductions such as `sum`
and `count`, and named-axis `insert`. Named-axis `insert` requires a
compile-time constant `i64` size. Positional rewriters such as
`permute`, `reshape`, and `matmul` are rejected inside a `..r` body.

## No broadcasting

Chelis does not broadcast. Operands of an elementwise operation must have identical
dimension lists. Use `insert` to add a dimension explicitly before combining tensors of
different rank, and `expand` to broadcast an existing size-1 axis.

```chelis-surf
def add_bias(x: tensor[2, 3, f32], bias: tensor[3, f32]) -> tensor[2, 3, f32] = add(x, insert(bias, 0i32, 2i64))
```

Calling `add(x, bias)` directly is a dimension mismatch. `insert` adds the
leading axis explicitly; `expand`, `reshape`, and `permute` provide other
explicit shape changes.

## Precision rules

Precision never changes implicitly. Both operands of an arithmetic operation must have the
same precision; the compiler never inserts a cast for you.

```chelis-surf-fragment
-- add(tensor[d, f32], tensor[d, bf16]) is a type error.
result = add(x, cast(y, f32))
```

`cast(e, p)` explicitly converts a scalar or tensor to the named dtype; a
tensor keeps its dimensions. Casts can cross numeric kinds and can target
`bool`. An integer-to-float cast may round, and a float-to-integer cast
requires a finite, integral value in range; any other value traps. To narrow
on purpose, use a named conversion: `cast_trunc` truncates a float toward
zero, `cast_saturate` clamps to the target's range, and `cast_wrap` wraps a
signed integer modulo the target width. Integer literals default to
`i32` and float literals to `f32`, subject to these exact adoption rules:

1. A suffix binds a literal to its stated dtype.
2. Unsuffixed elements of a bare bracket literal adopt the element type of
   the tensor-typed binding or declared tensor return body that makes it a
   tensor.
3. An unsuffixed scalar literal passed directly to `cast` adopts its numeric
   target dtype.

A bracket literal is a `List`. It becomes a tensor through `to_tensor`, whose
argument is an ordinary `List` that keeps each element's suffix or default,
or where its own binding or function result declares a tensor type. A tensor
parameter or a `cast` never converts a bracket literal, so a tensor argument
carries its element dtype in suffixes. A list literal and a bare scalar
passed to an ordinary function do not adopt a callee's dtype. Structural
lists such as reshape sizes therefore spell their `i64` elements explicitly.

```chelis-surf-fragment
-- explicit tensor conversion
a = cast(x, bf16)
-- suffix binds f64
b = 1.0f64
-- suffix binds i64; cast(3000000000, i64) binds the same literal the same way
c = 3000000000i64
-- the scalar binds directly at f64, not at f32 and then widened
d = cast(1.1, f64)
-- suffixed elements bind at f64; `to_tensor` keeps each element's dtype
e = to_tensor([1.1f64, 2.2f64])
```

Arithmetic operands must have the same numeric dtype and dimensions, with
each operation's own dtype domain. Ordered comparisons such as `cmplt` take
equal numeric types; `eq` and `neq` also compare booleans, strings, unit, and
two `List`, tuple, `Dict`, `Option`, or data-type values of one type,
structurally, when no part of the value is a function, key, or resource
handle. Tensor comparisons produce a boolean tensor.
Logical operations (`and`, `or`, `not`) take `bool`; transcendental operations
such as `exp`, `log`, and `sqrt` take float types.

`matmul`, `sum`, and `einsum` accept an optional final argument
`accumulator=p`, as in `sum(x, 0i32, accumulator=f64)`; any other call that
supplies it is a type error. Omitted, the compiler resolves the default from
the operand dtype. For `bf16` and
`f16`, the default accumulator is `f32`; their result returns to the
operand dtype. `sum` and `einsum` default to `i32` accumulation and an `i32`
result for `i8` and `i16` inputs; integer `matmul` is rejected. For `f32`
and signed integer reductions, an explicitly wider permitted accumulator
also widens the result. The requested accumulator must have the same numeric
kind and be no narrower than either the operands or their default.

Small-integer sums widen to `i32` (`spec/04` §5.7.1), because a total of N
values needs more bits than its elements. Stored `i8` and `i16` tensors stay
`i8` and `i16`; only the aggregate an operation returns widens:

| Operation | What it sums | Result for an `i8` or `i16` operand | Result for any other operand |
|---|---|---|---|
| `sum` | the selected axes | `i32` | the operand dtype |
| `cumsum` | every prefix along the axis | `i32` | the operand dtype |
| `trace` | the selected diagonal | `i32` | the operand dtype |
| `einsum` | every contracted label | `i32` | the operand dtype |

A declared result, or a downstream operation, that expects the operand dtype
is a type error that names this rule. Declare the result as `i32`, or narrow
it explicitly with `cast(total, i8)`, which traps `Overflow` when the total
does not fit. `sum` and `einsum` also take `accumulator=i64` for an `i64`
total. A function generic over a precision variable must bound it to
dtypes that share one result, such as `Float`, `{i32, i64}`, or `{i8, i16}`
with an `i32` result.

## Function types

A function type can be written `A -> B -> C`: `A` and `B` are the two
argument types, and `C` is the return type. Calls still supply both arguments
together as `f(a, b)`; the arrow chain does not make a function implicitly
curried. Parenthesize an argument that is itself a function type.

```chelis-surf-fragment
tensor[n, f32] -> tensor[n, f32] -> tensor[f32]   -- two args, scalar result
(tensor[n, f32] -> tensor[f32]) -> tensor[f32]    -- a function-typed argument
```

## Aggregate types

- Tuples: `(f32, f32)` as a type, `(a, b)` as a value, projected with `.0`, `.1`.
- Algebraic data types: `type Option[a] = | None | Some(a)` has a positional payload, and
  `type Shape = | Circle { radius: f32 } | Square { side: f32 }` has record-field payloads.
  Recursive types refer to themselves by name.
- Records: constructed with `Foo { x: e1, y: e2 }` (punning allowed), read with `e.field`.
- Type aliases: a `type` without variants is transparent and expanded at desugaring.
- Lists: `List[T]` holds rank-uniform elements; a list of tensors fixes one rank for every
  element. The list builtins (`len`, `index`, `append`, `concat`, `range`, `zip`,
  `enumerate`, `map`, `filter`, `fold`, and others) operate on these.

## Effects in types

A function type can declare an effect set. In Surf, write it as `! { ... }`
on a `sig` or `def`; an omitted clause leaves effects inferred, while
`! {}` declares a pure upper bound.

```chelis-surf-fragment
sig report[n]: tensor[n, f32] -> unit ! { IO }
```

Effect inference runs after type inference. Host operations such as `print`
and file reads contribute `IO`. Random draws take a `key` and contribute no
effect. `with device(...)` introduces a resource region checked against the
build target. See [Effects](effects.md) for the effect vocabulary.

## Linearity and borrowing

Tensor values are owned by default. A consuming use ends access through that
binding; read-only uses can borrow it. The compiler inserts needed copies and
drops an unused owner after its last use when it can establish that point.
Keys are different: a key-carrying value can be used at most once on a path
and cannot be copied or borrowed.

A read-only borrow is a reference type, written `&T` in a signature and `&x` at a use site.
A borrow leaves the owned binding live and is the idiomatic way to pass a tensor to a
read-only operation.

```chelis-surf-fragment
def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)
```

Passing an owned value where a borrow is expected auto-borrows. Passing a borrow where an
owned value is expected is an error unless you write `copy(x)` for a
copyable value. Borrows cannot be stored in aggregates, returned, or captured
by closures. The borrow target must be a tensor or a tensor-carrying value,
including tuples or data types with tensor fields.

## Inference

The checker infers many types and checks annotations you supply. A generic
declaration must still list its type, dimension, dtype, or rank binders in
`[...]`. A `cast` names its target dtype; a literal outside its default dtype
range needs a suffix or direct cast; and a claimed dimension may need an
annotation and a runtime equality check.

The pipeline order is parse, desugar, type inference and checking, effect inference and
checking, linearity checking, then lowering.
