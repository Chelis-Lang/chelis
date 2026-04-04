# Chelis Language Specification: Deep Syntax (S-Expressions)

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

## 1. Overview

Deep is the canonical s-expression syntax for Chelis programs. Every Chelis program, regardless of whether it was written in Surf or Deep, has a unique Deep representation. The compiler operates on Deep form internally.

Deep serves as:
- The input to the type checker (Check stage).
- The output of desugaring (Desugar stage).
- A serialization format for ASTs.
- A target for machine-generated code (agents may emit Deep directly).

---

## 2. Grammar

### 2.1 Core Grammar

Every Deep expression is either an atom or a list:

```ebnf
expr     = atom | list | meta_expr
list     = '(' tag expr* ')'
tag      = SYMBOL
meta_expr = '^' meta_map expr

atom     = SYMBOL | INTEGER | FLOAT | STRING | KEYWORD | BOOL

SYMBOL   = (letter | '_') (letter | digit | '_' | '-')*
INTEGER  = '-'? digit+ | '0' ('x'|'X') hex_digit+ | '0' ('b'|'B') ('0'|'1')+
FLOAT    = '-'? digit+ '.' digit+ (('e'|'E') ('+'|'-')? digit+)?
STRING   = '"' string_char* '"'
KEYWORD  = ':' (letter | '_') (letter | digit | '_' | '-')*
BOOL     = 'true' | 'false'
```

### 2.2 Atoms

| Atom type | Examples | Description |
|-----------|----------|-------------|
| Symbol | `x`, `foo_bar`, `MyType`, `add`, `mul` | Names for variables, types, functions, tags |
| Integer | `42`, `-7`, `0xFF`, `0b1010` | Integer literals |
| Float | `3.14`, `-1e-5`, `0.001` | Floating-point literals |
| String | `"hello"`, `"line\n"` | UTF-8 string literals |
| Keyword | `:axis`, `:type`, `:pure` | Named keys (used for metadata and named arguments) |
| Bool | `true`, `false` | Boolean literals |

### 2.3 Lists

A list is a parenthesized sequence of expressions. The first element is always a symbol called the **tag**. The tag determines how the rest of the list is interpreted.

```
(def square (sig (-> f32 f32)) (fn (x) (apply mul x x)))
```

Here `def` is the tag, and the remaining three elements are the name, signature, and body.

---

## 3. Tag Vocabulary

This section defines every valid tag in Deep syntax. This is a closed set: any list whose tag is not in this vocabulary is invalid.

### 3.1 Definitions

#### `def` -- Function Definition

```
(def name sig body)
```

- `name`: Symbol. The function name.
- `sig`: A `(sig ...)` form specifying the type signature.
- `body`: An expression (typically a `(fn ...)` form).

Example:
```
(def square
  (sig (-> f32 f32))
  (fn (x) (apply mul x x)))
```

#### `let` -- Let Binding

```
(let ((name1 expr1) (name2 expr2) ...) body)
```

- The first argument is a list of binding pairs. Each pair is `(name expr)`.
- `body`: The expression in which the bindings are visible.
- Bindings are evaluated sequentially: `name2`'s expression can refer to `name1`.

Example:
```
(let ((y (apply mul x x))
      (z (apply add y 1.0)))
  z)
```

#### `type` -- ADT Definition

```
(type name (params...) ((Variant1 field_types...) (Variant2 field_types...) ...))
```

- `name`: Symbol. The type name (must start with uppercase).
- `params`: A list of type parameter symbols.
- The third argument is a list of variants. Each variant is a list: the constructor name followed by zero or more field types.

Example:
```
(type Option (a) ((Some a) (None)))
(type Shape () ((Scalar) (Vector i64) (Matrix i64 i64)))
```

#### `module` -- Module Declaration

```
(module name decl1 decl2 ...)
```

- `name`: Symbol. The module name (must start with uppercase).
- Remaining elements are declarations (def, type, let, import).

Example:
```
(module MyModule
  (import OtherModule)
  (type Foo () ((Bar) (Baz)))
  (def main (sig (-> unit i32)) (fn () 42)))
```

#### `import` -- Import Declaration

```
(import module_name)
(import module_name (name1 name2 ...))
```

- `module_name`: Symbol. The module to import.
- Optional second argument: list of specific names to import.

### 3.2 Expressions

#### `fn` -- Lambda

```
(fn (param1 param2 ...) body)
```

- First argument: list of parameter names (symbols).
- Second argument: body expression.

Parameters may carry type annotations via metadata:
```
(fn (^{:type f32} x  ^{:type f32} y) (apply add x y))
```

Example:
```
(fn (x y) (apply add x y))
```

#### `apply` -- Function Application

```
(apply f arg1 arg2 ...)
```

- `f`: The function expression.
- Remaining elements: arguments.

All function calls in Deep are explicit `apply` forms. There is no implicit application by juxtaposition (that's a Surf convenience).

Example:
```
(apply add 1 2)
(apply my_function x y z)
```

#### `if` -- Conditional

```
(if condition then_expr else_expr)
```

- Always three arguments. There is no `if` without `else` in Deep (it's an expression language).

Example:
```
(if (apply gt x 0) x (apply neg x))
```

#### `match` -- Pattern Matching

```
(match scrutinee
  (case pattern1 body1)
  (case pattern2 body2)
  ...)
```

- `scrutinee`: The expression being matched.
- Each `case` contains a pattern and a body expression.

Patterns use the following forms:
- `symbol` -- variable binding (lowercase) or constructor (uppercase with no args)
- `(ctor pat1 pat2 ...)` -- constructor pattern with sub-patterns
- `_` -- wildcard
- literal (integer, float, string, bool) -- literal pattern
- `(tuple pat1 pat2 ...)` -- tuple pattern

Example:
```
(match opt
  (case (Some x) x)
  (case None default_val))
```

#### `pipe` -- Pipeline

```
(pipe expr fn1 fn2 fn3 ...)
```

- The first argument is the initial value.
- Each subsequent argument is a function. The result of applying `fn_i` to the current value becomes the input to `fn_{i+1}`.

Semantically equivalent to nested application:
```
(pipe x f g h)  ===  (apply h (apply g (apply f x)))
```

Example:
```
(pipe x normalize relu softmax)
```

#### `:` -- Type Annotation

```
(: expr type_expr)
```

- Annotates `expr` with the type `type_expr`. This is a hint to the type checker, not a coercion.

Example:
```
(: 42 i32)
(: x (tensor (batch hidden) f32))
```

#### `tuple` -- Tuple Construction

```
(tuple expr1 expr2 ...)
```

- Constructs a tuple of two or more values.

Example:
```
(tuple 1 2 3)
(tuple (apply f x) (apply g y))
```

### 3.3 Tensor Operations

#### `tensor` -- Tensor Literal

```
(tensor shape precision data)
```

- `shape`: A list of dimension sizes (integers).
- `precision`: A precision keyword (symbol: `f32`, `f64`, etc.).
- `data`: A flat list of numeric literals, in row-major order.

Example:
```
(tensor (3) f32 (1.0 2.0 3.0))
(tensor (2 2) f64 (1.0 0.0 0.0 1.0))
```

#### `cast` -- Explicit Precision Cast

```
(cast expr precision)
```

- Converts `expr` to the specified precision. This is the only way to convert between numeric types.

Example:
```
(cast x f64)
(cast (apply add a b) bf16)
```

### 3.4 Transformations

#### `grad` -- Automatic Differentiation

```
(grad f)
```

- `f`: A function expression. Must have a scalar return type.
- Returns a new function computing the gradient of `f`.

Example:
```
(grad (fn (x) (apply mul x x)))
(def loss_grad (sig (-> (tensor (n) f32) (tensor (n) f32))) (grad loss_fn))
```

#### `vmap` -- Vectorized Map

```
(vmap f :axis n)
```

- `f`: A function expression.
- `:axis`: Keyword argument specifying which axis to vectorize over.
- `n`: Integer, the axis index.

Example:
```
(vmap normalize :axis 0)
(apply (vmap process :axis 0) batch_data)
```

#### `jit` -- JIT Compilation Marker

```
(jit f)
```

- `f`: A function expression. Returns a function with the same type and semantics, but marked for just-in-time compilation.

Example:
```
(jit (fn (x) (apply matmul w x)))
```

### 3.5 Signature and Type Expressions

#### `sig` -- Type Signature

```
(sig type_expr)
```

A wrapper used within `def` to hold the function's type signature. The `sig` form exists so that the signature is unambiguously distinguished from the body.

#### `->` -- Function Type

```
(-> arg_type return_type)
(-> arg1_type arg2_type return_type)
```

Multi-argument function types list all argument types before the return type. The last element is always the return type.

Example:
```
(-> f32 f32)                                        -- f32 -> f32
(-> (tensor (n) f32) (tensor (n) f32) f32)         -- two tensor args, scalar return
```

#### `tensor` in type position -- Tensor Type

```
(tensor dims precision)
```

When `tensor` appears inside a `sig`, `:`, or other type context, `dims` is a list of dimension names/sizes and `precision` is the element type.

Example:
```
(tensor (batch hidden) f32)     -- tensor[batch, hidden, f32]
(tensor (784) f32)              -- tensor[784, f32]
(tensor (3 3) f64)              -- tensor[3, 3, f64]
```

#### `adt` -- ADT Type Application

```
(adt Name param1 param2 ...)
```

Applies a type constructor to type arguments.

Example:
```
(adt Option f32)
(adt Result f32 String)
(adt List (tensor (n) f32))
```

---

## 4. Metadata

### 4.1 Metadata Syntax

Any Deep expression can carry metadata, written with the `^{...}` prefix:

```
^{key1 val1 key2 val2 ...} expr
```

- Keys are **keywords** (`:type`, `:effects`, `:linear`, `:source_loc`, etc.).
- Values are atoms or lists.
- Metadata does not affect semantics. It is used by the compiler for type annotations, source locations, optimization hints, and debugging information.

### 4.2 Standard Metadata Keys

| Key | Value type | Description |
|-----|-----------|-------------|
| `:type` | type expr | The inferred or annotated type of the expression |
| `:effects` | list of keywords | Effect annotations (`:pure`, `:io`, `:random`) |
| `:linear` | bool | Whether the function uses its argument exactly once |
| `:source_loc` | `(file line col)` | Original source location for error reporting |
| `:fitness` | float | Fitness score contribution of this node |
| `:doc` | string | Documentation string |

### 4.3 Example with Metadata

```
(def ^{:doc "Squares a number" :type (-> f32 f32)}
  square
  (sig (-> f32 f32))
  (fn (^{:type f32} x)
    ^{:type f32} (apply mul x x)))
```

The metadata is attached to the form that immediately follows it. In the above:
- The `def` form carries `:doc` and `:type` metadata.
- The parameter `x` carries `:type` metadata.
- The body `(apply mul x x)` carries `:type` metadata.

---

## 5. Canonical Form Rules

Two Deep programs are semantically equivalent if and only if their canonical forms are byte-identical. The canonical form is defined by these rules:

### 5.1 Whitespace

- Elements within a list are separated by a single space.
- When a list exceeds 80 characters on one line, it is broken across lines:
  - The tag and first argument stay on the same line as the opening parenthesis.
  - Subsequent arguments are each on their own line, indented by 2 spaces relative to the opening parenthesis.
- No trailing whitespace on any line.
- File ends with a single newline.

### 5.2 Metadata

- Metadata maps have keys in alphabetical order.
- Metadata is on the same line as the opening paren of the form it annotates (if it fits in 80 chars), otherwise on the preceding line with the same indentation.

### 5.3 Atoms

- Integers: decimal form, no leading zeros (except `0` itself), no underscores.
- Floats: always include a decimal point, no trailing zeros after the decimal except to keep at least one digit (e.g., `1.0` not `1.`, `0.5` not `.5`).
- Strings: minimal escaping (only `\\`, `\"`, `\n`, `\t`, `\r`, `\0`).
- Symbols: as-is (no quoting).
- Keywords: `:` prefix, no quoting.

### 5.4 Ordering

- Within a `module`, declarations appear in the order: imports, types, defs/lets (preserving source order within each group).
- Within a `let`, bindings appear in dependency order (a binding that references another comes after it).
- Within a `type`, variants appear in source order.

### 5.5 Encoding

- UTF-8, no BOM.
- Unix line endings (LF, not CRLF).

---

## 6. Desugaring: Surf to Deep

Every Surf construct has a defined, mechanical translation to Deep. The desugaring does not require type information -- it is purely syntactic.

### 6.1 Complete Desugaring Table

#### Module Declaration

```
-- Surf
module MyModule

-- Deep
(module MyModule ...)
```

#### Import

```
-- Surf
import OtherModule
import Foo(bar, baz)

-- Deep
(import OtherModule)
(import Foo (bar baz))
```

#### Type Definition

```
-- Surf
type Option a = Some a | None

-- Deep
(type Option (a) ((Some a) (None)))
```

```
-- Surf
type Shape = Scalar | Vector i64 | Matrix i64 i64

-- Deep
(type Shape () ((Scalar) (Vector i64) (Matrix i64 i64)))
```

#### Function Definition

```
-- Surf
def f(x: tensor[batch, hidden, f32]): tensor[batch, hidden, f32] = body

-- Deep
(def f
  (sig (-> (tensor (batch hidden) f32) (tensor (batch hidden) f32)))
  (fn (x) <<body>>))
```

A `def` with multiple parameters:

```
-- Surf
def add(x: f32, y: f32): f32 = x + y

-- Deep
(def add
  (sig (-> f32 f32 f32))
  (fn (x y) (apply add x y)))
```

A `def` with no type annotation (types will be inferred):

```
-- Surf
def double(x) = x + x

-- Deep
(def double
  (sig _)
  (fn (x) (apply add x x)))
```

The `_` in the `sig` indicates that the type should be inferred.

#### Let Binding (with `in`)

```
-- Surf
let x = expr1 in body

-- Deep
(let ((x <<expr1>>)) <<body>>)
```

Chained lets:

```
-- Surf
let x = a
let y = b
in x + y

-- Deep
(let ((x <<a>>)
      (y <<b>>))
  (apply add x y))
```

#### Top-Level Let (without `in`)

```
-- Surf
let pi = 3.14159

-- Deep
(def pi (sig _) 3.14159)
```

Top-level `let` without `in` desugars to a `def` with inferred type.

#### Lambda

```
-- Surf
fn (x, y) -> x + y

-- Deep
(fn (x y) (apply add x y))
```

#### Function Application

Juxtaposition in Surf becomes explicit `apply` in Deep:

```
-- Surf
f x y

-- Deep
(apply f x y)
```

Note: in Surf, `f x y` is `((f x) y)` (curried). In Deep, multi-argument application is flat: `(apply f x y)`. The desugaring collects curried applications into a single `apply`.

#### Pipe

```
-- Surf
x |> f |> g |> h

-- Deep
(pipe x f g h)
```

When a pipe argument is a partial application:

```
-- Surf
x |> f(a, _) |> g

-- Deep (the partial application becomes a lambda)
(pipe x (fn (__arg) (apply f a __arg)) g)
```

Note: Explicit partial application syntax is not in Chelis 0.1.0. The pipe always takes named functions or lambdas.

#### If/Then/Else

```
-- Surf
if cond then a else b

-- Deep
(if <<cond>> <<a>> <<b>>)
```

#### Match

```
-- Surf
match expr with
  | Some x -> x + 1
  | None   -> 0

-- Deep
(match <<expr>>
  (case (Some x) (apply add x 1))
  (case None 0))
```

#### Infix Operators

All infix operators desugar to `apply` of named functions:

```
-- Surf          -- Deep
a + b            (apply add a b)
a - b            (apply sub a b)
a * b            (apply mul a b)
a / b            (apply div a b)
a % b            (apply mod a b)
a == b           (apply eq a b)
a != b           (apply ne a b)
a < b            (apply lt a b)
a > b            (apply gt a b)
a <= b           (apply le a b)
a >= b           (apply ge a b)
a && b           (apply and a b)
a || b           (apply or a b)
-a               (apply neg a)
!a               (apply not a)
```

#### Type Annotation

```
-- Surf
expr : Type

-- Deep
(: <<expr>> <<Type>>)
```

#### Tensor Literal

```
-- Surf
tensor[1.0, 2.0, 3.0] : tensor[3, f32]

-- Deep
(: (tensor (3) f32 (1.0 2.0 3.0)) (tensor (3) f32))
```

Note: In Surf, tensor literals use `tensor[...]` for values and `tensor[..., precision]` for types. The desugaring must distinguish these by context.

#### Cast

```
-- Surf
cast(x, f64)

-- Deep
(cast x f64)
```

#### Transformations

```
-- Surf              -- Deep
grad(f)              (grad f)
vmap(f, axis=0)      (vmap f :axis 0)
vmap(f)              (vmap f :axis 0)       -- default axis is 0
jit(f)               (jit f)
```

#### Tuple

```
-- Surf
(a, b, c)

-- Deep
(tuple a b c)
```

### 6.2 Type Expression Desugaring

Type expressions in Surf desugar to type expressions in Deep:

```
-- Surf type             -- Deep type
f32                      f32
tensor[batch, hidden, f32]   (tensor (batch hidden) f32)
A -> B                   (-> A B)
A -> B -> C              (-> A (-> B C))
Option f32               (adt Option f32)
(f32, f32)               (tuple_type f32 f32)
```

Note on function types: In the `sig` of a `def`, multi-argument functions use a flat `->`:

```
-- Surf
def f(x: f32, y: f32): f32

-- Deep sig
(sig (-> f32 f32 f32))
```

This is distinct from curried function types in expression position:

```
-- Surf type annotation
f : f32 -> f32 -> f32

-- Deep
(: f (-> f32 (-> f32 f32)))
```

The `def` form's `sig` uses the flat convention for ergonomics; type annotations in expressions use the nested convention for precision.

---

## 7. Example Programs in Deep

The following are the Deep equivalents of the example programs from `02-surf-syntax.md`.

### 7.1 Hello Tensor

```
(module HelloTensor
  (def main
    (sig (-> unit (tensor (3) f32)))
    (fn ()
      (let ((a (: (tensor (3) f32 (1.0 2.0 3.0)) (tensor (3) f32)))
            (b (: (tensor (3) f32 (4.0 5.0 6.0)) (tensor (3) f32))))
        (apply add a b)))))
```

### 7.2 Linear Regression with grad

```
(module LinearRegression
  (def predict
    (sig (-> (tensor (features) f32) (tensor () f32) (tensor (features) f32) (tensor () f32)))
    (fn (w b x)
      (let ((wx (apply reduce_sum (apply mul w x))))
        (apply add wx b))))

  (def mse_loss
    (sig (-> (tensor (features) f32) (tensor () f32) (tensor (features) f32) (tensor () f32) (tensor () f32)))
    (fn (w b x y)
      (let ((pred (apply predict w b x))
            (diff (apply sub pred y)))
        (apply mul diff diff))))

  (def train_step
    (sig (-> (tensor (features) f32) (tensor () f32) (tensor (features) f32) (tensor () f32) (tensor () f32)
             (tuple_type (tensor (features) f32) (tensor () f32))))
    (fn (w b x y lr)
      (let ((loss_fn (fn (w_ b_) (apply mse_loss w_ b_ x y)))
            (grads (apply (grad loss_fn) w b))
            (dw (apply fst grads))
            (db (apply snd grads)))
        (tuple (apply sub w (apply mul lr dw))
               (apply sub b (apply mul lr db)))))))
```

### 7.3 Simple MLP

```
(module MLP
  (type Activation () ((Relu) (Tanh) (Sigmoid)))

  (def relu
    (sig (-> (tensor (n) f32) (tensor (n) f32)))
    (fn (x)
      (if (apply gt x (: (tensor (n) f32 ()) (tensor (n) f32)))
        x
        (: (tensor (n) f32 ()) (tensor (n) f32)))))

  (def linear
    (sig (-> (tensor (out_dim in_dim) f32) (tensor (out_dim) f32) (tensor (in_dim) f32) (tensor (out_dim) f32)))
    (fn (w b x)
      (apply add (apply matmul w x) b)))

  (def mlp
    (sig (-> (tensor (hidden input) f32) (tensor (hidden) f32)
             (tensor (output hidden) f32) (tensor (output) f32)
             (tensor (input) f32) (tensor (output) f32)))
    (fn (w1 b1 w2 b2 x)
      (pipe x
        (fn (__arg) (apply linear w1 b1 __arg))
        relu
        (fn (__arg) (apply linear w2 b2 __arg))))))
```

### 7.4 Pattern Matching on ADTs

```
(module ADTExample
  (type Option (a) ((Some a) (None)))
  (type Shape () ((Scalar) (Vector i64) (Matrix i64 i64)))

  (def describe_shape
    (sig (-> (adt Shape) i64))
    (fn (s)
      (match s
        (case Scalar 0)
        (case (Vector n) n)
        (case (Matrix r c) (apply mul r c)))))

  (def unwrap_or
    (sig (-> (adt Option f32) f32 f32))
    (fn (opt default)
      (match opt
        (case (Some x) x)
        (case None default))))

  (def map_option
    (sig (-> (-> a b) (adt Option a) (adt Option b)))
    (fn (f opt)
      (match opt
        (case (Some x) (apply Some (apply f x)))
        (case None None)))))
```

### 7.5 Pipe-Heavy Data Processing

```
(module Pipeline
  (def normalize
    (sig (-> (tensor (n) f32) (tensor (n) f32)))
    (fn (x)
      (let ((mean (apply div (apply reduce_sum x) (cast n f32)))
            (centered (apply sub x mean))
            (variance (apply div
                        (apply reduce_sum (apply mul centered centered))
                        (cast n f32))))
        (apply div centered (apply sqrt variance)))))

  (def softmax
    (sig (-> (tensor (n) f32) (tensor (n) f32)))
    (fn (x)
      (let ((max_x (apply reduce_max x))
            (shifted (apply sub x max_x))
            (exps (apply exp shifted)))
        (apply div exps (apply reduce_sum exps)))))

  (def process
    (sig (-> (tensor (features) f32) (tensor (features) f32)))
    (fn (x)
      (pipe x
        normalize
        (fn (v) (apply mul v (: (tensor (features) f32 (2.0)) (tensor (features) f32))))
        softmax)))

  (def batch_process
    (sig (-> (tensor (batch features) f32) (tensor (batch features) f32)))
    (fn (xs)
      (apply (vmap process :axis 0) xs)))

  (def fast_batch_process
    (sig (-> (tensor (batch features) f32) (tensor (batch features) f32)))
    (fn (xs)
      (apply (jit batch_process) xs))))
```

---

## 8. Parsing Deep

### 8.1 Reader Algorithm

Parsing Deep is straightforward because the grammar is regular (Lisp-style). The reader operates as follows:

1. Skip whitespace and comments (`;` to end-of-line).
2. If the next character is `(`, read a list: read expressions until `)`.
3. If the next character is `^`, read metadata: read `{` key-value pairs until `}`, then read the annotated expression.
4. If the next character is `"`, read a string literal.
5. If the next character is `:`, read a keyword.
6. If the next character is a digit or `-` followed by a digit, read a number.
7. Otherwise, read a symbol.

### 8.2 Comments in Deep

Deep uses semicolons for comments:

```
; This is a line comment in Deep
(def square ; defines the square function
  (sig (-> f32 f32))
  (fn (x) (apply mul x x)))
```

Note: Surf uses `--` and `{- -}`. Deep uses `;`. This is intentional -- they are different languages with different conventions.

### 8.3 Error Recovery

When parsing Deep, the reader should be lenient:
- Unbalanced parentheses: report the location and attempt to recover by assuming the missing paren.
- Unknown tags: parse the list normally and flag the unknown tag for the Check stage.
- Malformed atoms: report and skip to the next whitespace.

This leniency supports the fitness-score model: even a partially parseable Deep file should produce a useful score.

---

## 9. Invariants

The following invariants hold for all valid Deep programs:

1. **Every list has a tag.** Empty lists `()` are not valid.
2. **Tags are from the closed vocabulary** defined in Section 3.
3. **`def` always has exactly 3 children**: name, sig, body.
4. **`let` always has exactly 2 children**: a bindings list and a body.
5. **`if` always has exactly 3 children**: condition, then-branch, else-branch.
6. **`fn` always has exactly 2 children**: parameter list and body.
7. **`apply` has at least 2 children**: function and at least one argument.
8. **`match` has at least 2 children**: scrutinee and at least one case.
9. **Each `case` has exactly 2 children**: pattern and body.
10. **`grad` has exactly 1 child**: the function to differentiate.
11. **`jit` has exactly 1 child**: the function to JIT-compile.
12. **`vmap` has exactly 1 child and 1 keyword argument**: the function and `:axis`.
13. **Metadata maps have an even number of elements** (alternating keys and values).
14. **Metadata keys are keywords** (start with `:`).
15. **The canonical form is deterministic**: `canonical(parse(canonical(ast))) == canonical(ast)`.
