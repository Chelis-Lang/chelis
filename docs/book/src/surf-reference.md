# Surf syntax reference

Surf is Chelis's source syntax, stored in `.ch` files. `chelis deep FILE.ch`
prints the Deep form of a complete file, and `chelis surf FILE.dp` converts
back.

Examples here illustrate individual syntax forms. Some are complete
definitions; others depend on surrounding bindings, imports, or declarations.
For a complete file you can check and run, see the
[first program](first-program.md).

## Modules

One module per file. The `module` declaration is the first non-comment line. Module names
are PascalCase and dot-separated. In a package with `module_prefix = "Shoals"`,
`src/pricing/options.ch` declares:

```chelis-surf-fragment
module Shoals.Pricing.Options
```

A script or snippet can omit `module`. Package source must declare the name fixed by its
package prefix and path beneath the source root.

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
def add_vec[n](x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = add(x, y)
```

The body can be a `{ ... }` block of bindings ending in a result expression:

```chelis-surf
def twice_then_relu[n](x: tensor[n, f32]) -> tensor[n, f32] = {
  y = add(x, x)
  relu(y)
}
```

The `[...]` clause declares variables used in a function's types. Depending on where a
name appears, it can stand for a dimension, rank, dtype, or general type. Calls
instantiate these variables by unification:

```chelis-surf
def identity[a](x: tensor[a, f32]) -> tensor[a, f32] = x
```

A binder may carry a dtype-family bound, written after the name. The families are `Float`,
`Int`, and `Numeric`; a `sig` takes the same clause in the same position:

```chelis-surf
sig scale_ints[p: Int]: p -> p -> p
def scale_ints(x, k) = mul(x, k)
```

Direct calls are flat: write `f(x, y)`. To call a function returned by an ordinary
expression, group the callee: `(make_adder(x))(y)`. The ungrouped `f(x)(y)`
form is rejected. Compiler transforms are self-delimiting, so `grad(f)(x)`
and `vmap(f)(xs)` need no extra grouping.

Empty brackets are not decorative syntax: write `Option` and `def f(x)`, not
`Option[]` or `def f[](x)`. A constructor value such as `None` is bare;
`None()` calls the zero-argument constructor, while `Empty {}` constructs a
zero-field record. These forms have distinct meanings.

A standalone `sig` must precede a matching definition in the same module, but need not
be adjacent to it. The `scale_ints` signature above is a complete example. A `sig`
without a matching definition is rejected.

## Local bindings and blocks

There is no `let` keyword. Inside a block, `name = expr` introduces a binding; bindings are
separated by newlines, and the final bare expression is the block's value. A block needs
at least one binding and a final expression. For ordered expression sequencing, use
semicolons in `do { first; second }`. `par { ... }` is reserved syntax and is
rejected by the checker.

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

```chelis-surf
def apply_twice(f: f32 -> f32, x: f32) -> f32 = f(f(x))
scale = 3.0f32
tripled_twice = apply_twice(fn (v) -> (v * scale), 2.0f32)
```

`chelis eval --file` prints `tripled_twice = 18.0`. The lambda captures
`scale` from the enclosing scope, and its parameter type is inferred from
`apply_twice`'s signature, where `f32 -> f32` is the function type.

## Literals

- Integers: canonical decimal such as `42` and `1000000`. Default type `i32`.
- Floats: finite shortest round-trippable spellings such as `1.0`, `1e-5`, and
  `31400000000.0`. Default type `f32`. You may write a longer body that decodes
  to the same value. For example, you can transcribe a constant at published
  precision as `0.319381530f64`. `chelis fmt` prints the shortest
  spelling for it. A float literal must be finite at the type it binds at:
  `70000.0f16` and an unsuffixed `1e40` (an `f32`) are rejected because they
  round to infinity there. A cast of a finite wider value,
  `cast(70000.0f32, f16)`, is an operation and does yield infinity.
- A literal can carry a precision suffix that binds it exactly: float suffixes `f32 f64
  bf16 f16` (for example `42.0f32`, `1.0f64`), integer suffixes `i8
  i16 i32 i64` (int tokens only). The suffix must follow the digits with no space.
  A float suffix on an integer body (`42f32`) is a different literal from the
  decimal-bodied one: it binds the exact integer directly at that width instead
  of decoding a decimal, and the formatter keeps whichever you wrote.
  Literal patterns are unsuffixed because Deep patterns preserve only the raw value.
- Strings: `"hello"` with escapes `\" \\ \n \t \r \0`.
- Booleans: `true`, `false`.
- Unit: `()` is the unit value; `unit` is the unit type.
- Tuples: `(a, b, c)`. `(a)` is grouping; a one-element tuple is `(a,)`.
- Bracket literals: `[1.0, 2.0, 3.0]` builds a `List`, whatever its elements and
  wherever it stands, and bracket lists pass list arguments to operators, for example
  the window and stride lists in `reduce_window_max(grid, [2i64, 2i64], [1i64, 1i64])`.
  `to_tensor([1.0, 2.0, 3.0], f32)` builds a tensor, and so does a bracket literal whose own
  binding or function result declares a tensor type. A negative numeral is unary minus
  applied to a literal, so write `f(-42)` to pass a negative argument.

Delimited nonempty lists may carry one trailing comma (or a trailing semicolon in
`do`). The parser discards it and the formatter omits it; the comma in `(a,)`
remains because it distinguishes a one-element tuple from grouping.

The parser accepts value-preserving digit separators, hexadecimal/binary integers, every
finite decimal float body that decodes to the literal's value, and equivalent valid
Unicode escapes. `chelis fmt` prints their canonical decimal/string spelling; `fmt --check`
rejects the resulting source diff. Malformed separators, a redundant leading zero on an
integer body, invalid escapes, and semantic suffix changes remain errors.

A numeric declaration states the dtype of an unsuffixed literal it directly
contains. A direct `cast` does so when the literal's kind admits its numeric
target: `cast(1.1, f64)` is exactly `1.1f64`. A tensor declaration states the
element dtype of its bracket literal. `to_tensor` can state it with a dtype argument:
`to_tensor([1.0, 2.0], f64)`. Without that argument, every numeric literal
element needs a suffix: `to_tensor([1.0f64, 2.0f64])`. Unsuffixed numeric
literal elements inside `to_tensor`, including integers and negative values,
have no default, so `to_tensor([1, 2, 3])` is rejected.

A dtype argument checks an already-typed element; it does not convert it.
For `xs = [1.0, 2.0]`, the named list already has `f32` elements and
`to_tensor(xs)` keeps them. A tensor parameter or a `cast` never turns a
bracket literal into a tensor. Structural lists such as the window sizes
above need explicit `i64` elements. See
[precision rules](type-reference.md#precision-rules) for the full
dtype-stating rules.

`to_tensor` is reserved: no definition, parameter, local binding, pattern,
or import may reuse the name, even inside a Reef package.

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
| `*` `/` `%` | multiplicative, `%` is `mod` (integers and floats) |
| unary `-` `!` | prefix |
| function application | |
| `.` | field and tuple access |

Equality and comparison do not chain: `a == b == c` is a parse error. There is no operator
overloading and no infix bitwise or power operator. The bitwise builtins are
`bitand`, `bitor`, `bitxor`, `shl`, and `shr`. Each takes two values of one
signed-integer dtype (two scalars, or two tensors of the same shape) and
returns that dtype and shape; `bool` and float operands are type errors.
`bitand`, `bitor`, and `bitxor` act on the two's-complement bits. `shl(x, n)`
shifts left and discards bits past the width, so a count at or above the
width gives `0`. `shr(x, n)` shifts right, copying the sign bit, so a large
count gives `0` or `-1`. A negative count stops evaluation with
`shift amount must be non-negative, got -1`. None of them traps on overflow.

```chelis-surf
a = bitand(12i32, 10i32)
o = bitor(12i32, 10i32)
x = bitxor(12i32, 10i32)
l = shl(1i8, 7i8)
r = shr(-16i32, 2i32)
big = shl(1i32, 40i32)
```

```text
a = 8
o = 14
x = 6
l = -128
r = -4
big = 0
```

There is no `pow` builtin. Write an integer power as repeated `mul`, and a
float power of a positive base as `exp(mul(y, log(x)))`.

The pipe operator threads its left value as the first argument of the call on its right.
`x |> f(y)` is `f(x, y)`, and stages chain from left to right:

```chelis-surf-fragment
hidden = matmul(x, w1) |> add(b1) |> relu
```

Use pipes when left-to-right stage order helps review a linear flow. Calls,
operator sugar and pipes have equal standing. Pipes exist only in Surf: desugaring
normalizes them before literal typing, and `chelis deep` and `chelis surf` print
the resulting calls. `chelis fmt` preserves authored pipes.

Explicitly group pipes mixed with another operator or an open-ended form:
`(a * b) |> f`, `a * (b |> f)`, `(if c then a else b) |> f`, and
`fn (v) -> (v |> f)` are valid. Ungrouped equivalents are rejected.

When the piped value belongs in a later position, pipe into a lambda:

```chelis-surf-fragment
complement = p |> (fn (q) -> sub(1.0, q))
```

## Function application

Application is `f(x, y)`. Ordinary arguments are positional. These specific
forms accept a named argument:

- `grad(f, wrt=w)` selects parameters to differentiate.
- `vmap(f, axis=1)` selects a nonzero batch axis. Axis zero uses `vmap(f)`.
- `matmul`, `sum`, and `einsum` accept a final `accumulator=p`, as in
  `sum(x, 0i32, accumulator=f64)`. See
  [accumulator precision rules](type-reference.md#precision-rules).

Other calls do not accept arbitrary keyword arguments.

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

`gather` determines its result shape from the selected axis, so its i32 axis
must be a literal or an integer-cast-wrapped literal, including negative
axes. Variables, helper calls and other computed expressions are rejected
at checking, even if they return a constant. `sort` preserves the input
shape and accepts computed i32 axes; invalid runtime axes trap in both
the evaluator and compiled programs. Negative axes count from the end.
Reductions, `expand`, and `insert` require a static constant or named axis.

When projecting through nested tuples, group the inner numeric projection so
the next suffix cannot merge with it as a float:

```chelis-surf-fragment
first = (nested.0).0
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
def activate[n](act: Activation, x: tensor[n, f32]) -> tensor[n, f32] =
  match act with {
    | Relu => relu(x)
    | Sigmoid => sigmoid(x)
  }
```

Patterns include variables, the wildcard `_`, literals, constructors with payloads,
records with field punning, tuples, and guards introduced by `if` before the `=>`:

```chelis-surf
def sign_label(n: i32) -> string =
  match n with {
    | x if (x > 0) => "positive"
    | x if (x < 0) => "negative"
    | _ => "zero"
  }
labels = (sign_label(5), sign_label(-2), sign_label(0))
```

`chelis eval --file` prints `labels.0 = positive`, `labels.1 = negative`, and
`labels.2 = zero`. Arms are tried in order, and the first whose pattern
matches and whose guard is true wins.

Negative numeric patterns are written directly (`-42`, `-1.5`, or `-0.0`);
unlike expression position, pattern position has no unary-expression node.

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

A module cannot both import a name into unqualified scope and declare it:
`import Std.Scalar (max)`, or `import Std.Scalar (..)`, beside a local `def max` is an error
in every command, and the diagnostic names both. Rename the local declaration, or import the
module qualified (`import Std.Scalar`) and call `Std.Scalar.max`. Except for
reserved names, parameters and local bindings may still reuse an imported name.

A file belongs to the Reef package found by walking up from the file's own directory,
whatever directory a command runs in. A file inside a package but outside its source roots,
such as a script beside `reef.toml` or a test under `tests/`, is an entry of that package and
can import its modules, with or without a `module` line. A file outside every Reef package
imports from the compiler-bundled `chelis-std` (`Std.*`) under the same rules.
Importing any other module needs a `reef.toml` package manifest, and an import that names
no reachable module is a `chelis check` error.

With no `export` declaration, every top-level `def` and `type` is public. Once any `export`
appears, only the listed names are public. Exporting a type also exports its constructors.

```chelis-surf-fragment
export (forward, Linear)
```

## Dimensions

A module-level `dim` declares concrete named dimensions used across the file. Function-level
`[...]` parameters declare dimension variables local to one function. In
0.19.1, a file with a module-level `dim` checks but does not evaluate or build
(see [Known issues](known-issues.md)).

```chelis-surf-fragment
dim batch, vocab_size
def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = permute(x, 1, 0)
```

## Effects and device regions

A function's effects can be annotated with a `! { ... }` suffix on its `sig` or
`def`. `IO` is inferred from host operations such as `print`. Random draws take
a `key` and add no effect. Omitting the clause leaves effects inferred; `! {}`
declares a pure upper bound. The `with` form introduces a device region:

```chelis-surf
def local_region() -> i32 = with device("cpu") { 1i32 }
```

`with device("...")` takes a string literal. For a host-C build, only the exact
device name `"cpu"` is accepted; other names are rejected before output is written.
See [Effects](effects.md) for the full model.

## Transforms

`grad` and `vmap` use call syntax but are compiler transforms. Each requires a
function argument, and their results are functions that can be called or bound to
a name. Transform targets that are aliases of top-level functions can
be rejected; use a direct, unshadowed top-level function when that occurs. See
[Transforms](transforms.md) for supported target forms.

```chelis-surf-fragment
(dw, db) = grad(loss_fn, wrt=(w, b))(w, b)

batched = vmap(process)(xs)
```

`cast(e, p)` converts a scalar or tensor to the named numeric or boolean dtype;
tensor dimensions stay the same. `copy(e)` produces an owned duplicate where
copying is permitted; keys cannot be copied. See the [Type System Reference](type-reference.md)
for details.

## Naming conventions

- Function names use snake_case. Types, constructors, and module path segments use
  PascalCase. Descriptive value, parameter, dimension, and field names use snake_case;
  a single uppercase letter is also valid for a value binding or parameter.
- The parser enforces identifier roles. `chelis lint` adds these blocking
  rules for Chelis source, which `check`, `eval --file`, and `build` also run
  before their main work:

  | Rule | Requires |
  |---|---|
  | `surf-value-snake-case` | `def` names are lowercase ASCII with optional digits and underscores, no leading underscore. |
  | `surf-type-pascal-case` | `type` names are ASCII alphanumeric with a leading uppercase letter and no underscores. |
  | `module-compound-titlecase` | Module segments write abbreviations in title case (`Hamt`, never `HAMT`). |
  | `surf-test-name-prefix` | A function carrying the `Test` effect is named `test_*` or `example_*`. |
  | `surf-def-arrow-form` | A definition is written `def name(params) -> T = expr`, not with `:` before the result type. |
  | `type-suffix-policy` | A name suffix such as `_f32`, `_int`, `_bool`, or `_string` describes the element type, not the container. |
  | `prefix-namespace` | A shared function-name prefix inside a module names the module's domain or a documented sub-namespace. |
  | `reef-module-identity` | In a Reef package, each file declares one module, named from `module_prefix` and its path. |
  | `opaque-domain-construction` | An `@opaque` type is not constructed, updated, or cast outside its module. |
  | `no-em-dash-in-public-strings` | String literals contain no em dash. |

  `chelis lint --list` prints every rule with its severity, and
  `chelis lint --fix` applies the available automatic fixes. Advisory rules
  such as `opaque-escape-site` and `recursive-list-cursor` print without
  blocking.
