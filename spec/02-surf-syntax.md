# Chelis Language Specification: Surface Syntax (Surf)

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

## 1. Lexical Structure

### 1.1 Encoding

Surf source files are UTF-8 encoded. The file extension is `.ch`. A byte-order mark (BOM) at the start of the file is permitted but ignored.

### 1.2 Keywords

The following identifiers are reserved and cannot be used as value or type names:

```
def    let    in     type   match  with   fn     module
import if     then   else   grad   vmap   jit    tensor
cast   true   false  where
```

### 1.3 Identifiers

```
ident_start  = letter | '_'
ident_cont   = letter | digit | '_'
value_ident  = lower ident_cont*          -- starts with lowercase letter or _
type_ident   = upper ident_cont*          -- starts with uppercase letter
dim_ident    = lower ident_cont*          -- dimension names (same rules as value idents)
```

- **Value identifiers** (functions, variables, parameters) start with a lowercase letter or underscore: `x`, `foo_bar`, `_unused`.
- **Type identifiers** (type names, constructors) start with an uppercase letter: `Option`, `MyModel`, `Some`.
- **Dimension identifiers** are syntactically value identifiers but appear in tensor type positions: `batch`, `hidden`, `seq_len`.

Identifiers may contain Unicode letters beyond ASCII, but the first character determines the value/type distinction.

### 1.4 Literals

#### Integer Literals

```
decimal   = digit+
hex       = '0' ('x' | 'X') hex_digit+
octal     = '0' ('o' | 'O') oct_digit+
binary    = '0' ('b' | 'B') ('0' | '1')+
```

Underscores may appear between digits for readability: `1_000_000`, `0xFF_FF`. Leading underscores in the digit portion are not permitted.

Integer literals have no inherent type. Their type is inferred from context, defaulting to `i64` if unconstrained.

#### Float Literals

```
float = digit+ '.' digit+ exponent?
      | digit+ exponent
exponent = ('e' | 'E') ('+' | '-')? digit+
```

Examples: `3.14`, `1e-5`, `2.0e10`, `0.001`. Float literals default to `f64` if unconstrained.

#### String Literals

```
string = '"' string_char* '"'
string_char = any_char_except_backslash_or_quote | escape_seq
escape_seq = '\\' ('n' | 't' | 'r' | '\\' | '"' | '0')
```

Strings are UTF-8 encoded. Chelis does not have a dedicated string processing library; strings exist primarily for metadata, error messages, and interop.

#### Boolean Literals

```
boolean = 'true' | 'false'
```

### 1.5 Operators

Operators listed from **lowest to highest** precedence. All binary operators are left-associative unless noted.

| Precedence | Operator | Fixity | Description |
|------------|----------|--------|-------------|
| 1 | `\|>` | Left | Pipe (function application) |
| 2 | `\|\|` | Left | Logical OR |
| 3 | `&&` | Left | Logical AND |
| 4 | `==` `!=` | None | Equality (non-associative) |
| 5 | `<` `>` `<=` `>=` | None | Comparison (non-associative) |
| 6 | `+` `-` | Left | Addition, subtraction |
| 7 | `*` `/` `%` | Left | Multiplication, division, modulo |
| 8 | unary `-` `!` | Prefix | Negation, logical NOT |
| 9 | function application | Left | Juxtaposition: `f x` |

Non-associative operators cannot be chained: `a == b == c` is a parse error. Use explicit parentheses.

The pipe operator `|>` passes the left operand as the first argument to the right operand:
```
x |> f       -- equivalent to f(x)
x |> f |> g  -- equivalent to g(f(x))
```

### 1.6 Comments

```
line_comment  = '--' (any_char_except_newline)* newline
block_comment = '{-' (any_char | block_comment)* '-}'
```

Block comments nest: `{- outer {- inner -} still outer -}` is valid.

### 1.7 Layout Rules

Chelis supports two block styles, selectable per file:

**Indentation mode** (default): Blocks are delimited by indentation level, similar to Python and Haskell's offside rule. A block begins after `=`, `->`, `with`, `then`, `else`, or `in` when followed by a newline, and ends when the indentation returns to or below the level of the introducing keyword.

**Brace mode**: Triggered by `{` immediately after a block-introducing keyword. In brace mode, blocks are delimited by `{ }` and statements are separated by `;`. A file cannot mix modes -- the first block-introducing construct determines the mode for the entire file.

Examples (both equivalent):

```
-- Indentation mode
def f(x: f32): f32 =
  let y = x * x
  in y + 1.0

-- Brace mode
def f(x: f32): f32 = {
  let y = x * x;
  in y + 1.0
}
```

### 1.8 Significant Whitespace Details

In indentation mode:
- Tabs are forbidden. Only spaces are permitted for indentation.
- The indentation of a block's first line establishes the block's indentation level.
- Subsequent lines at the same indentation level are continuation lines of the block.
- A line indented further than the block level is a continuation of the previous expression.
- A line at or below the enclosing block's level ends the current block.
- Blank lines (containing only whitespace or comments) are ignored for layout purposes.

---

## 2. Grammar (EBNF)

This section defines the complete grammar of Surf. Terminals are in `'quotes'` or UPPER_CASE. Non-terminals are in lower_case. `*` means zero or more, `+` means one or more, `?` means optional. `|` separates alternatives. Parentheses group.

### 2.1 Top Level

```ebnf
program       = module_decl? import* decl*

module_decl   = 'module' TYPE_IDENT

import        = 'import' module_path ('(' ident_list ')')?
module_path   = TYPE_IDENT ('.' TYPE_IDENT)*
ident_list    = IDENT (',' IDENT)*
```

### 2.2 Declarations

```ebnf
decl          = type_decl
              | def_decl
              | let_decl

type_decl     = 'type' TYPE_IDENT type_params? '=' variant ('|' variant)*
type_params   = VALUE_IDENT+
variant       = TYPE_IDENT type_expr*

def_decl      = 'def' VALUE_IDENT '(' param_list? ')' (':' type_expr)? '=' expr
param_list    = param (',' param)*
param         = VALUE_IDENT (':' type_expr)?

let_decl      = 'let' VALUE_IDENT (':' type_expr)? '=' expr
```

### 2.3 Expressions

```ebnf
expr          = let_expr
              | if_expr
              | match_expr
              | fn_expr
              | pipe_expr

let_expr      = 'let' VALUE_IDENT (':' type_expr)? '=' expr 'in' expr

if_expr       = 'if' expr 'then' expr 'else' expr

match_expr    = 'match' expr 'with' match_arm+
match_arm     = '|' pattern '->' expr

fn_expr       = 'fn' '(' param_list? ')' '->' expr

pipe_expr     = or_expr ('|>' or_expr)*
```

### 2.4 Binary and Unary Expressions

```ebnf
or_expr       = and_expr ('||' and_expr)*
and_expr      = cmp_expr ('&&' cmp_expr)*
cmp_expr      = add_expr (cmp_op add_expr)?
cmp_op        = '==' | '!=' | '<' | '>' | '<=' | '>='
add_expr      = mul_expr (('+' | '-') mul_expr)*
mul_expr      = unary_expr (('*' | '/' | '%') unary_expr)*
unary_expr    = ('-' | '!') unary_expr
              | app_expr

app_expr      = primary (primary)*
```

Note: `app_expr` handles function application by juxtaposition. `f x y` parses as `((f x) y)`.

### 2.5 Primary Expressions

```ebnf
primary       = VALUE_IDENT
              | TYPE_IDENT                    -- constructor
              | literal
              | '(' expr ')'                  -- grouping
              | '(' expr ',' expr (',' expr)* ')'  -- tuple
              | tensor_expr
              | cast_expr
              | grad_expr
              | vmap_expr
              | jit_expr
              | expr ':' type_expr            -- type annotation

literal       = INTEGER | FLOAT | STRING | 'true' | 'false'

tensor_expr   = 'tensor' '[' expr_list ']'
expr_list     = expr (',' expr)*

cast_expr     = 'cast' '(' expr ',' type_expr ')'

grad_expr     = 'grad' '(' expr ')'
vmap_expr     = 'vmap' '(' expr (',' 'axis' '=' INTEGER)? ')'
jit_expr      = 'jit' '(' expr ')'
```

### 2.6 Patterns

```ebnf
pattern       = TYPE_IDENT pattern*           -- constructor pattern
              | VALUE_IDENT                    -- variable binding
              | '_'                            -- wildcard
              | literal                        -- literal pattern
              | '(' pattern ',' pattern (',' pattern)* ')'  -- tuple pattern
              | '(' pattern ')'               -- grouping
```

Patterns are checked for exhaustiveness. A non-exhaustive match is a type error (not a warning).

### 2.7 Type Expressions

```ebnf
type_expr     = type_atom ('->' type_expr)?    -- function types, right-associative

type_atom     = 'i8' | 'i16' | 'i32' | 'i64'
              | 'u8' | 'u16' | 'u32' | 'u64'
              | 'f16' | 'bf16' | 'f32' | 'f64'
              | 'bool' | 'unit'
              | tensor_type
              | TYPE_IDENT type_atom*          -- ADT application: Option f32
              | VALUE_IDENT                    -- type variable: a, b
              | '(' type_expr ')'

tensor_type   = 'tensor' '[' dim_list ',' precision ']'
dim_list      = dim (',' dim)*
dim           = VALUE_IDENT                    -- named dimension: batch
              | INTEGER                        -- concrete dimension: 784
precision     = 'f16' | 'bf16' | 'f32' | 'f64'
              | 'i8' | 'i16' | 'i32' | 'i64'
              | 'u8' | 'u16' | 'u32' | 'u64'
              | 'bool'
```

The last element in a tensor type bracket is always the precision. Everything before it is a dimension.

---

## 3. Type Syntax Details

### 3.1 Base Types

| Type | Size | Description |
|------|------|-------------|
| `i8`, `i16`, `i32`, `i64` | 8/16/32/64 bits | Signed integers |
| `u8`, `u16`, `u32`, `u64` | 8/16/32/64 bits | Unsigned integers |
| `f16` | 16 bits | IEEE 754 half-precision float |
| `bf16` | 16 bits | Brain floating point |
| `f32` | 32 bits | IEEE 754 single-precision float |
| `f64` | 64 bits | IEEE 754 double-precision float |
| `bool` | 1 bit (logically) | Boolean |
| `unit` | 0 bits (logically) | Unit type, single value `()` |

### 3.2 Tensor Types

Tensor types are written `tensor[dim1, dim2, ..., precision]`. The last element is always the precision type. All preceding elements are dimensions.

```
tensor[784, f32]                -- 1D tensor, 784 elements, f32
tensor[batch, hidden, f32]     -- 2D tensor, named dimensions, f32
tensor[3, 3, f64]              -- 2D tensor, 3x3, f64
tensor[batch, seq_len, 512, bf16]  -- 3D tensor, mixed named/concrete dims
```

Named dimensions are type-level identifiers. Two tensor types are compatible for element-wise operations when:
1. They have the same number of dimensions.
2. Corresponding dimensions have the same name (or both are concrete with the same value).
3. They have the same precision.

### 3.3 Function Types

Function types use `->`, which is right-associative:

```
f32 -> f32                          -- scalar to scalar
tensor[n, f32] -> tensor[n, f32]   -- tensor to tensor
f32 -> f32 -> f32                   -- curried two-argument function
                                     -- equivalent to f32 -> (f32 -> f32)
(f32, f32) -> f32                   -- tuple argument (not curried)
```

### 3.4 ADT Types

ADT types are applied by juxtaposition:

```
Option f32                          -- Option applied to f32
List (tensor[n, f32])               -- List of tensors (parens needed for compound arg)
Result f32 String                   -- Result applied to f32 and String
```

### 3.5 Type Variables

Type variables are lowercase identifiers used in polymorphic definitions:

```
def identity(x: a): a = x
def map(f: a -> b, xs: List a): List b = ...
```

Type variables are implicitly universally quantified at the enclosing `def`.

---

## 4. Semantics Notes

### 4.1 Evaluation Order

Chelis is **strict** (eager evaluation). Function arguments are fully evaluated before the function body executes. This is important for predictable performance in tensor computation.

### 4.2 Purity

All functions are pure by default. There are no mutable variables, no assignment operators, and no I/O primitives in the core language. Effects (I/O, randomness, statefulness) will be handled via an effect system in a future version.

### 4.3 Tensor Semantics

Tensor values are immutable. Operations like `add`, `reshape`, and `slice` produce new tensors. The compiler's optimization passes (operator fusion, in-place mutation) may reuse storage, but this is invisible to the programmer.

### 4.4 Scoping

Chelis uses lexical scoping. A `let` binding is visible in its body (the `in` clause) and nowhere else. A `def` at the top level of a module is visible throughout the module (definitions may be mutually recursive). Shadowing is permitted: a `let` binding can shadow an outer binding of the same name.

---

## 5. Example Programs

### 5.1 Hello Tensor

Create tensors and perform basic arithmetic.

```
-- hello_tensor.ch
-- Create two tensors and add them.

module HelloTensor

def main(): tensor[3, f32] =
  let a = tensor[1.0, 2.0, 3.0] : tensor[3, f32]
  let b = tensor[4.0, 5.0, 6.0] : tensor[3, f32]
  in a + b
```

Result: `tensor[5.0, 7.0, 9.0]` of type `tensor[3, f32]`.

### 5.2 Linear Regression with grad

Define a loss function and compute its gradient.

```
-- linear_regression.ch
-- Simple linear regression using grad.

module LinearRegression

def predict(w: tensor[features, f32], b: tensor[f32], x: tensor[features, f32]): tensor[f32] =
  let wx = reduce_sum(w * x)
  in wx + b

def mse_loss(
  w: tensor[features, f32],
  b: tensor[f32],
  x: tensor[features, f32],
  y: tensor[f32]
): tensor[f32] =
  let pred = predict(w, b, x)
  let diff = pred - y
  in diff * diff

-- grad computes the gradient with respect to (w, b),
-- treating x and y as fixed.
def train_step(
  w: tensor[features, f32],
  b: tensor[f32],
  x: tensor[features, f32],
  y: tensor[f32],
  lr: tensor[f32]
): (tensor[features, f32], tensor[f32]) =
  let loss_fn = fn (w_, b_) -> mse_loss(w_, b_, x, y)
  let (dw, db) = grad(loss_fn)(w, b)
  in (w - lr * dw, b - lr * db)
```

### 5.3 Simple MLP

Define a multi-layer perceptron.

```
-- mlp.ch
-- A simple two-layer MLP.

module MLP

type Activation = Relu | Tanh | Sigmoid

def relu(x: tensor[n, f32]): tensor[n, f32] =
  if x > tensor[0.0] : tensor[n, f32]
    then x
    else tensor[0.0] : tensor[n, f32]

def linear(
  w: tensor[out_dim, in_dim, f32],
  b: tensor[out_dim, f32],
  x: tensor[in_dim, f32]
): tensor[out_dim, f32] =
  matmul(w, x) + b

def mlp(
  w1: tensor[hidden, input, f32],
  b1: tensor[hidden, f32],
  w2: tensor[output, hidden, f32],
  b2: tensor[output, f32],
  x: tensor[input, f32]
): tensor[output, f32] =
  x
    |> linear(w1, b1)
    |> relu
    |> linear(w2, b2)
```

### 5.4 Pattern Matching on ADTs

```
-- adt_example.ch
-- Pattern matching with algebraic data types.

module ADTExample

type Option a = Some a | None

type Shape
  = Scalar
  | Vector i64
  | Matrix i64 i64

def describe_shape(s: Shape): i64 =
  match s with
    | Scalar     -> 0
    | Vector n   -> n
    | Matrix r c -> r * c

def unwrap_or(opt: Option f32, default: f32): f32 =
  match opt with
    | Some x -> x
    | None   -> default

def map_option(f: a -> b, opt: Option a): Option b =
  match opt with
    | Some x -> Some (f x)
    | None   -> None
```

### 5.5 Pipe-Heavy Data Processing

```
-- pipeline.ch
-- Chaining transformations with pipes.

module Pipeline

def normalize(x: tensor[n, f32]): tensor[n, f32] =
  let mean = reduce_sum(x) / cast(n, f32)
  let centered = x - mean
  let variance = reduce_sum(centered * centered) / cast(n, f32)
  in centered / sqrt(variance)

def softmax(x: tensor[n, f32]): tensor[n, f32] =
  let max_x = reduce_max(x)
  let shifted = x - max_x
  let exps = exp(shifted)
  in exps / reduce_sum(exps)

def process(x: tensor[features, f32]): tensor[features, f32] =
  x
    |> normalize
    |> fn (v) -> v * tensor[2.0] : tensor[features, f32]
    |> softmax

-- Using vmap to batch-process
def batch_process(xs: tensor[batch, features, f32]): tensor[batch, features, f32] =
  vmap(process, axis=0)(xs)

-- JIT-compile the batch version for performance
def fast_batch_process(xs: tensor[batch, features, f32]): tensor[batch, features, f32] =
  jit(batch_process)(xs)
```

---

## 6. Reserved for Future Extension

The following syntax is reserved for future versions and must not be used in valid Chelis 0.1.0 programs:

- `where` clauses for type constraints: `def f(x: a): a where a : Numeric`
- `do` notation for effects: `do { x <- read(); write(x) }`
- Record syntax: `{ field1: val1, field2: val2 }`
- List comprehensions: `[x * x | x <- xs, x > 0]`
- Operator definition: `def (|+|)(a, b) = ...`
- Type classes / traits: `trait Differentiable a where ...`
- Module-qualified names: `Module.name`

These forms are mentioned here so that implementers do not accidentally use these tokens/patterns for other purposes.
