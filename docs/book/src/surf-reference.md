# Surf Syntax Reference

Surf is the human-facing syntax. Every Surf program desugars to Deep, the canonical
s-expression form the compiler works on. This page is a reference for the constructs you
write in Surf. For the corresponding Deep nodes see `spec/03-deep-syntax.md`; for the full
grammar see `spec/02-surf-syntax.md`.

Fenced `chelis-surf` and `chelis-deep` blocks below are complete programs and are
validated in CI. `chelis-surf-fragment` and `chelis-deep-fragment` blocks isolate one
construct and are not standalone programs.

## Modules

One module per file. The `module` declaration is the first non-comment line. Module names
are PascalCase and dot-separated. A module `Foo.Bar` lives at `foo/bar.ch`.

```chelis-surf-fragment
module School.Nn.Linear
```

A file without a `module` line is still valid for scripts and snippets; the named form is
the convention for package source.

## Comments

```chelis-surf-fragment
-- line comment to end of line

{- block comment,
   which {- nests -} cleanly -}
```

## Function definitions

A `def` binds a name to a function. The body is a single expression, which may be a block.
Type annotations on parameters and the return are optional; the compiler infers what you
leave off. The return arrow is `->`.

```chelis-surf
def add_vec(x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

The body can be a `{ ... }` block of bindings ending in a result expression:

```chelis-surf
def twice_then_relu(x: tensor[n, f32]) -> tensor[n, f32] = {
  y = add(x, x)
  relu(y)
}
```

A function can be polymorphic over dimensions. Names in the `[...]` clause before the
parameter list are dimension or precision variables, instantiated by unification at each
call site:

```chelis-surf
def identity[a](x: tensor[a, f32]) -> tensor[a, f32] = x
```

A standalone signature with `sig` can precede a `def`. It must appear directly before the
function it describes:

```chelis-surf-fragment
sig predict:
  tensor[samples, features, f32]
  -> tensor[features, 1, f32]
  -> tensor[1, f32]
  -> tensor[samples, 1, f32]
```

## Local bindings and blocks

There is no `let` keyword. Inside a block, `name = expr` introduces a binding; bindings are
separated by a newline or `;`, and the final bare expression is the block's value. Blocks
are themselves expressions.

```chelis-surf-fragment
{
  hidden = matmul(x, w)
  biased = add(hidden, b)
  relu(biased)
}
```

A tuple pattern on the left destructures a tuple result:

```chelis-surf-fragment
(values, indices) = sort(scores, 0)
```

## Anonymous functions

A lambda is `fn (params) -> expr`. Note the body arrow is `->`, distinct from `=>` used in
match arms.

```chelis-surf-fragment
loss_fn = fn (w, b) -> mse_loss(predict(x, w, b), y)
```

## Literals

- Integers: `42`, `1_000_000`, `0xFF`, `0xCAFE_BABE`. Default type `int32`.
- Floats: `1.0`, `3.141_592_6`, `1e-5`, `3.14e10`. Default type `f32`.
- A literal can carry a precision suffix that binds it exactly: float suffixes `f32 f64
  bf16 f16` (attach to int or float tokens, e.g. `42f32`, `1.0f64`), integer suffixes `i8
  i16 i32 i64` (int tokens only). The suffix must follow the digits with no space.
- Strings: `"hello"` with escapes `\" \\ \n \t \r \0`.
- Booleans: `true`, `false`.
- Unit: `()` is both the unit value and the unit type.
- Tuples: `(a, b, c)`. `(a)` is grouping, not a tuple; there is no one-element tuple.
- Bracket literals: `[1.0, 2.0, 3.0]` builds a tensor, and bracket lists also pass list
  arguments to operators, for example the window and stride lists in
  `reduce_window_max(grid, [2, 2], [1, 1])`. A negative numeral is unary minus applied to a
  literal, so write `f(-42)` to pass a negative argument.

In a position with a known element type (a tensor-typed argument, a tensor return body, or
the first argument of `cast`), bracket-literal elements adopt that element type. So
`[1, 2, 3]` is `tensor[3, int32]` on its own, but takes the surrounding element type where
one is imposed.

## Operators

Binary and unary operators desugar to named builtin calls. Precedence from loosest to
tightest binding:

| Operators | Notes |
|---|---|
| `\|>` | pipe, left associative |
| `\|\|` | logical or |
| `&&` | logical and |
| `==` `!=` | equality, non-associative |
| `<` `>` `<=` `>=` | comparison, non-associative |
| `+` `-` | additive |
| `*` `/` `%` | multiplicative, `%` is integer modulo |
| unary `-` `!` | prefix |
| function application | |
| `.` | field and tuple access |

Equality and comparison do not chain: `a == b == c` is a parse error. There is no operator
overloading and no infix bitwise operator; use the named builtins `bitand`, `bitor`,
`bitxor`, `shl`, `shr`, and `pow` for exponentiation.

The pipe operator threads its left value as the first argument of the call on its right.
`x |> f(y)` is `f(x, y)`. When the piped value belongs in a later position, pipe into a
lambda:

```chelis-surf-fragment
hidden = matmul(x, w1)
  |> fn (z) -> add(z, b1)
  |> relu
```

## Function application

Application is `f(x, y)`. The standard call surface is positional. Two transforms take a
named argument: `grad(f, wrt=w)` selects parameters to differentiate, and `vmap(f, axis=0)`
selects the batch axis. There is no general keyword-argument surface beyond these.

## Tuples and projection

Build a tuple with commas; project a field with `.` and an integer index.

```chelis-surf-fragment
pair = (w_new, b_new)
w = pair.0
b = pair.1
```

A call result can be projected directly, which is how you read the two outputs of `sort`:

```chelis-surf-fragment
sorted_values = sort(diag, 0).0
sorted_indices = sort(diag, 0).1
```

## Control flow

`if` is an expression and `else` is mandatory:

```chelis-surf-fragment
if cond then a else b
```

`match` performs pattern matching. Arms use `=>`. Matching must be exhaustive over the
scrutinee's type; a missing variant is a compile error.

```chelis-surf
type Activation =
  | Relu
  | Sigmoid

def activate(act: Activation, x: tensor[n, f32]) -> tensor[n, f32] =
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
```

Patterns include variables, the wildcard `_`, literals, constructors with payloads,
records with field punning, tuples, and guards introduced by `if` before the `=>`:

```chelis-surf-fragment
match n with {
  | x if x > 0 => positive(x)
  | x if x < 0 => negative(x)
  | _          => zero_case
}
```

## Types and constructors

Type declarations introduce algebraic data types with `type`. Variants can be nullary,
carry positional payloads, or carry named record fields.

```chelis-surf
type Optimizer =
  | Sgd { lr: tensor[f32] }
  | Adam { lr: tensor[f32], beta1: tensor[f32], beta2: tensor[f32], eps: tensor[f32] }

def learning_rate(opt: Optimizer) -> tensor[f32] =
  match opt with {
    | Sgd { lr } => lr
    | Adam { lr, beta1, beta2, eps } => lr
  }
```

A `type` without variants (no `|`) is a transparent alias, expanded at desugaring:

```chelis-surf
type Weights = tensor[n, f32]

def keep(w: Weights) -> Weights = w
```

Records are constructed with field names (punning allowed) and read with dot access:

```chelis-surf-fragment
opt = Adam { lr, beta1, beta2, eps }
rate = opt.lr
```

For the full type-expression surface, named dimensions, precision rules, effects, and
linearity, see the [Type System Reference](type-reference.md).

## Imports and exports

```chelis-surf-fragment
import Std.Tensor.Construct (linspace, arange)
import Nautilus.LinAlg (..)
import Std.Sort
```

`import M (a, b)` brings specific names into unqualified scope, `import M (..)` brings every
exported name, and `import M` makes only qualified access (`M.name`) available. A qualified
reference works for values, constructors, and types, and is the way to disambiguate two
modules that export the same name.

With no `export` declaration, every top-level `def` and `type` is public. Once any `export`
appears, only the listed names are public. Exporting a type also exports its constructors.

```chelis-surf-fragment
export (forward, Linear)
```

## Dimensions

A module-level `dim` declares concrete named dimensions used across the file. Function-level
`[...]` parameters declare dimension variables local to one function.

```chelis-surf-fragment
dim batch, vocab_size

def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)
```

## Effects and handlers

A function's effects can be annotated with a `! { ... }` suffix on the signature or the
`def`. The handled effects are `Random` and `Resource("device")`; `IO` is inferred from
host operations such as `print`. Handlers are introduced by `with`:

```chelis-surf-fragment
with seed(42) {
  dropout(x, 0.5)
}
```

`with seed(...)` takes an integer literal and `with device("...")` takes a string literal.
See [Effects and Handlers](effects.md) for the full model.

## Transforms

`grad` and `vmap` are written as calls but are compiler transforms, not ordinary functions.
They must always be applied, and they compose.

```chelis-surf-fragment
(dw, db) = grad(loss_fn, wrt=(w, b))(w, b)

batched = vmap(process, axis=0)(xs)
```

`cast(e, p)` changes precision, and `copy(e)` produces an owned duplicate of a value. See
[Transforms](transforms.md) and the [Type System Reference](type-reference.md) for details.

## Naming conventions

- Values, parameters, dimensions, and fields are snake_case and start with a lowercase
  letter or underscore.
- Types, constructors, and module path segments are PascalCase.
- `chelis lint` enforces these from `spec/01-nomenclature.md`. Run `chelis fmt` to
  canonicalize whitespace rather than tuning it by hand.
