# spec/02-surf-syntax.md — Chelis Surf Syntax Specification

**Status:** v0.3 (unified from design sprint, authoritative)

---

## 0. Notation

PEG notation. `/` is ordered choice. `*` is zero-or-more. `+` is one-or-more. `?` is optional. `!` is negative lookahead. `&` is positive lookahead. Literal strings in single quotes.

Deep desugaring shown as **⟹** with the target Deep s-expression.

### 0.1 Executable Surface Note

This syntax document records the intended Surf language shape, not only the currently
implemented evaluator/backend subset.

Important Phase 3 honesty rule:

- parser support or desugaring shape does not, by itself, mean a feature is already part
  of the practical executable language
- the remaining Phase 3 language-completeness work specifically targets the gap around
  first-class scalar/string workflows, collections, iteration, file/data loading, and
  tokenization

Until those Phase `3c` / `3d` / `3g` items land, the parser may describe surface forms
that are not yet the full self-sufficient AI-programming story.

---

## 1. Keywords

Reserved. Cannot be used as identifiers.

```
def  sig  type  dim  match  with  fn  module  property  forall
import  export  if  then  else  grad  vmap  jit  cast  macro
realize  copy  par  true  false  where
```

**Total: 26.**

Reserved for Phase 2 (parse as keywords, emit "reserved for future use" error):
```
effect  handler  perform  resume  borrow  where  do
```

---

## 2. Operator Precedence

Final. Binding power from lowest to highest:

| BP | Operators | Assoc | Deep form |
|----|-----------|-------|-----------|
| 1 | `\|>` | left | `(pipe {} ...)` |
| 2 | `\|\|` | left | `(app {} (var {} or) ...)` |
| 3 | `&&` | left | `(app {} (var {} and) ...)` |
| 4 | `==` `!=` | none | `eq` / `neq` |
| 5 | `<` `>` `<=` `>=` | none | `cmplt` / swapped `cmplt` / `lte` / `gte` |
| 6 | `+` `-` | left | `add` / `sub` |
| 7 | `*` `/` `%` | left | `mul` / `div` / `mod` |
| 8 | unary `-` `!` | prefix | `neg` / `not` |
| 9 | function application | left | `(app {} ...)` |
| 10 | `.` field access | left | `(access {} ...)` / `(tuple-get {} ...)` |

Non-associative operators (BP 4, 5) produce a parse error on chaining: `a == b == c` is rejected.

No operator overloading. No infix bitwise operators. Host-side integer bitwise work uses
named built-ins such as `bitand`, `bitor`, `bitxor`, `shl`, and `shr`. No exponentiation
operator — use `pow(x, n)` from `Std.Math`.

---

## 3. Punchlist Decisions

### P1: Module System

One module per file. `module Name` is the first non-comment line. Declarations follow until EOF — no body braces. Module names are PascalCase. No nesting within a file.

File path mapping: `module Foo.Bar` lives in `foo/bar.ch` relative to the project root.

```
module Std.Nn.Linear

import Std.Tensor (..)

def forward(x, w, b) = add(matmul(x, w), b)
```

**⟹** `(module {} std.nn.linear ...)` — module name lowercased and dot-joined in Deep.

### P2: Import / Export

| Surf form | Meaning | Deep form |
|-----------|---------|-----------|
| `import Foo.Bar (baz, qux)` | Import specific names | `(import {} foo.bar (baz qux))` |
| `import Foo.Bar (..)` | Import all exported | `(import-all {} foo.bar)` |
| `import Foo.Bar` | Qualified access only | `(import {} foo.bar ())` |

Qualified access (`Foo.Bar.baz`) is always available after any import form. Selective import additionally brings names into unqualified scope.

**Export:** Explicit. If no `export` declaration appears, all top-level `def` and `type` are public. Once any `export` appears, only listed names are public.

```
export (forward, Linear)
```

**⟹** `(export {} forward Linear)`

No re-exports in v1.

### P3: Dimension Declaration

Module-level `dim` for concrete dimensions. Function-level `[...]` brackets for polymorphic dimension parameters.

```
dim batch, vocab_size

def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] =
  permute(x, [1, 0])
```

**⟹**
```
(defdim {} batch)
(defdim {} vocab_size)
(defsig {} transpose (t-fn {} (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
                              (t-tensor {} (d-var {} b) (d-var {} a) (t-prim {} f32))))
(def {} transpose (fn {} (params x) ...))
```

**Disambiguation:** A name in a `dim` declaration or imported → concrete `d-name`. A name in a function's `[...]` → variable `d-var`. A lowercase name in a tensor type that is neither declared nor in brackets → parse error.

### P4: Type Signatures

Both inline and standalone forms. All types are optional — inference fills them in.

**Inline:**
```
def add_vecs(x: tensor[d, f32], y: tensor[d, f32]) -> tensor[d, f32] = add(x, y)
```

When a `def` has inline type annotations, the desugarer extracts a `defsig` in addition to the `def`.

**Standalone (for long signatures):**
```
sig multi_head_attn:
  tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, heads, seq, d_k, f32]
  -> tensor[batch, 1, seq, seq, f32]
  -> tensor[batch, heads, seq, d_k, f32]

def multi_head_attn(q, k, v, mask) = ...
```

**⟹** `(defsig {} multi_head_attn (t-fn {} ...arg_types... ret_type))`

`sig` must precede its corresponding `def`. Arrow chain reads as: arg₁ -> arg₂ -> ... -> return. Always flat in Deep (`t-fn` with last child as return type).

Phase 2a effect annotations are optional suffixes on either `sig` or `def`:

```text
sig predict: tensor[n, f32] -> tensor[n, f32] ! { Random }
def train(x: tensor[n, f32]) -> tensor[n, f32] ! { Random, Resource("gpu:0") } = ...
```

Surf accepts the built-in names `Diff`, `Random`, `Accum`, `IO`, and
`Resource("device")`.
The current shipped boundary-checking surface is narrower than the syntax:

- `Random` is the active user-facing boundary effect in Phase 2a
- `Resource("...")` is the active build-boundary placement annotation in Phase 2a
- `IO` is the shipped Phase 3 debugging/logging effect inferred from `print` and
  `debug`; it is allowed at the program boundary
- `Diff` is accepted as documentation / forward-compatible syntax, but `grad` remains a
  compiler capability rather than a user-handled boundary effect
- `Accum` is accepted as forward-compatible syntax but remains internal-only in the
  shipped Phase 2a subset

Omitting all types is valid: `def f(x, y) = add(x, y)`. The compiler emits a note recommending a `sig` for module-level definitions.

### P4a: Preferred Surf Style

The parser accepts both `def f(x: T): U = ...` and `def f(x: T) -> U = ...`.
Project-preferred source style is the arrow form:

```text
def f(x: tensor[batch, 784, f32]) -> tensor[batch, 128, f32] = ...
```

Style rules for human-facing Surf:

- put types on parameters rather than top-level load bindings
- use symbolic dimensions for runtime-varying axes such as `batch` and `seq`
- keep fixed architecture dimensions concrete
- omit intermediate type ascriptions when inference already determines the type
- combine short tensor operations when that improves readability

Planned Phase 3 public-style target:

- use block bindings exclusively: `x = expr` inside `{ ... }` and bare top-level
  bindings such as `result = expr`
- prefer pipe-first composition for eligible linear flows
- break long or many-stage pipes after `=` and before every `|>` using the same
  flat-first, width-threshold approach as the Deep pretty printer

Planned remaining Phase 3 language-completeness additions:

- practical scalar/string programming beyond tensor-only code
- collection literals and iteration idioms
- data-loading and tokenization helpers that remove the mandatory Python preprocessing
  step

### P5: Blocks and Sequencing

Braces define blocks. Inside blocks, bindings are sequential. Newlines and semicolons are both valid separators. The final expression is the block's value.

```
{
  x = f(a)
  y = g(x)
  z = h(y)
  combine(x, y, z)
}
```

**⟹** All sequential Surf bindings collapse into a single Deep `let` node:
```
(let {} (bind x (app {} (var {} f) (var {} a))
              y (app {} (var {} g) (var {} x))
              z (app {} (var {} h) (var {} y)))
  (app {} (var {} combine) (var {} x) (var {} y) (var {} z)))
```

Blocks are expressions: `result = { temp = f(x); g(temp) }` is valid.

A separator (newline or semicolon) is required between a binding and the next statement. Multiple separators (blank lines) are fine. Trailing semicolon after the final expression is tolerated.

There is no Surf `let ... in` expression form. Sequential bindings use blocks, and
top-level script-style bindings use bare `name = expr`.

No `where` clauses. Use blocks.

### P5a: Effect Handlers

Phase 2a adds two `with` block forms:

```text
with seed(42) {
  dropout(x, 0.5)
}

with device("gpu:0") {
  body
}
```

`with` handlers are expressions. They take exactly one argument in parentheses and a
brace-delimited block body.

Current shipped constraints:

- `with seed(...)` currently requires an explicit integer literal seed for the effect
  checker and lowering path
- `with device(...)` currently requires an explicit string literal device name
- only `seed` and `device` are valid handler names in the Phase 2a Surf parser

### P5b: Macros

Phase 2c adds top-level macro definitions:

```text
macro linear_layer(x, w, b) = add(matmul(x, w), expand(b, 0, batch))
macro relu_ref(x) = max_elem(x, 0.0)
```

Macro invocations use the ordinary call surface: `linear_layer(x, w, b)`.

Current shipped macro rules:

- resolution order is lexical blockers first, then user-defined top-level macros, then
  the standard macro prelude, then ordinary function call resolution
- a local binding named `linear_layer` or `cross_entropy` blocks macro expansion for
  that identifier
- hygiene renames only binders introduced by the macro expansion (block-binding names, `fn`
  params, pattern binders); free references in the macro body remain free and resolve
  in the caller's scope
- macro expansion runs before type checking, effect inference, linearity checking, and
  lowering

Current shipped prelude macros:

- `linear_layer(x, w, b)` -> `add(matmul(x, w), expand(b, 0, batch))`
- `residual(x, f)` -> `add(x, f(x))`
- `cross_entropy(logits, labels)` -> the standard `softmax` / `log` / `sum` / `mean`
  composition used by the current executable corpus

### P6: Records

Braces for construction. Punning allowed. Dot-chaining for access. Functional update with `with` is reserved for Phase 1 and not part of the Phase 0 parser/desugarer.

```
lr = 0.01
opt = Adam { lr, eps: 1.0e-8 }      -- punning: lr: lr
rate = opt.lr                         -- field access
chain = model.layer1.weight           -- chained access
```

**⟹**
```
Adam { lr, eps: 1.0e-8 }     ⟹  (record {} Adam (kv {} eps ...) (kv {} lr (var {} lr)))
opt.lr                        ⟹  (access {} (var {} opt) lr)
model.layer1.weight           ⟹  (access {} (access {} (var {} model) layer1) weight)
```

Record `kv` pairs are alphabetized by key in canonical Deep.

### P7: Pattern Matching

Match uses `=>` for arms (distinct from `->` in function types and lambdas).

```
match expr with {
  | Pattern => body
  | Pattern if guard => body
}
```

**Supported pattern forms:**

| Pattern | Example | Deep form |
|---------|---------|-----------|
| Variable | `x` | `(pat-var {} x)` |
| Wildcard | `_` | `(pat-wild {})` |
| Literal | `0`, `true`, `"hi"` | `(pat-lit {} value)` |
| Constructor | `Some(x)` | `(pat-ctor {} Some (pat-var {} x))` |
| Nested | `Some(Some(x))` | `(pat-ctor {} Some (pat-ctor {} Some (pat-var {} x)))` |
| Record | `Adam { lr, eps }` | `(pat-record {} Adam (kv {} lr (pat-var {} lr)) ...)` |
| Tuple | `(a, b, c)` | `(pat-tuple {} (pat-var {} a) ...)` |
| As-pattern | `x @ Some(_)` | `(pat-as {} x (pat-ctor {} Some (pat-wild {})))` |

**Guards:** `if` after pattern, before `=>`. Guard fills the guard slot in `(arm {} pattern guard body)`:

```
match n with {
  | x if x > 0 => positive(x)
  | x if x < 0 => negative(x)
  | _           => zero_case
}
```

Record patterns allow punning and ignore unmentioned fields. Field order doesn't matter.

No or-patterns in v1. Write separate arms.

Exhaustiveness required. Every variant of the scrutinee's ADT must be covered. Non-exhaustive match is a compile error.

### P8: Tuples

Construction: `(a, b, c)` with commas. `(a)` is grouping, not a tuple. No single-element tuples.

Unit: `()` is both the unit value and unit type.

Access: dot-integer syntax.

```
pair = (w_new, b_new)
w = pair.0
b = pair.1
```

**⟹**
```
(a, b)   ⟹  (tuple {} a' b')
pair.0   ⟹  (tuple-get {} (var {} pair) (lit {type: (t-prim {} int32)} 0))
()       ⟹  (lit {type: (t-unit {})} ())
```

Destructuring via bindings:
```
(w, b) = train_step(w, b, x, y, lr)
```

### P9: Transforms

Transforms use call syntax in Surf but desugar to dedicated Deep tags. The parser recognizes these keywords and emits transform nodes, not `app` nodes.

| Surf | Deep | Notes |
|------|------|-------|
| `grad(f)` | `(grad {} f')` | Returns gradients only |
| `grad(f, wrt=w)` | `(grad {wrt: ...} f' idx)` | `wrt` names one parameter of `f` |
| `grad(f, wrt=(w, b))` | `(grad {wrt: ...} f' (tuple {} idx₁ idx₂))` | Multi-parameter `wrt` preserves the written order |
| `jit(f)` | `(jit {} f')` | |
| `vmap(f, n)` | `(vmap {} f' n')` | Axis positional, defaults to 0 if omitted |
| `vmap(f)` | `(vmap {} f' (lit {type: (t-prim {} int32)} 0))` | |
| `cast(e, bf16)` | `(cast {} e' (t-prim {} bf16))` | Second arg is a type literal (special form) |
| `realize(e)` | `(realize {} e')` | |
| `copy(e)` | `(copy {} e')` | |
| `&x` | `(borrow {} (var {} x))` | Explicit read-only borrow; usually inferred at call sites |

Transforms compose naturally: `jit(grad(loss_fn))` **⟹** `(jit {} (grad {} (var {} loss_fn)))`.
When `grad` targets one differentiable parameter, the result is that gradient value.
When it targets multiple parameters, the result is a flat tuple of gradients rather than
`(value, grad)` or nested tuples.

Transforms can be called after construction when they produce a function value.
Example: `vmap(process)(xs)` parses as an ordinary application whose callee is the
transform node `vmap(process)`.

Transforms must always be applied — `g = grad` bare is a parse error.

The second argument to `cast` is a precision type literal (`f32`, `bf16`, etc.) in expression position. This is the one special form where a type appears as an argument.

### P10: Numeric Literals

Default float precision: **f32**. Default integer type: **int32**.

Underscore separators: `1_000_000`, `3.141_592_6`. Stripped during lexing.

Scientific notation: `1e-5`, `3.14e10`. Canonical Deep form: `d.dE±d`.

Hex integer literals are accepted (`0xFF`, `0xCAFE_BABE`); see the
hex-suffix interaction note below. Octal and binary literals are accepted
by the current Surf lexer (`0b...`) but their long-term spec status is
unchanged by this section.

**Literal default rule (authoritative):** an unsuffixed integer literal binds
at type `int32`; an unsuffixed float literal binds at type `f32`. The lexer
parses unsuffixed literals at i64/f64 precision so that out-of-range literals
can be diagnosed before defaulting; the desugarer/type-checker then narrows
the value to `int32` (for integer tokens) or `f32` (for float tokens) before
Deep is materialized. The narrowing is the **user-facing contract** and is
non-overridable except by:

1. an explicit literal suffix (P10a)
2. the contextual tensor-literal inference rule (P10b) when the literal
   appears inside a tensor body in a known-element-type position
3. an explicit `cast(literal, p)` around the literal expression

There is no implicit precision promotion. A bare `42` in any unannotated
position binds at `int32`, not `int64`. A bare `1.0` binds at `f32`, not
`f64`. See `spec/04-type-system.md` §5.3 for the type-system statement of
this rule.

**Negative literals:** `-42` is always parsed as unary minus applied to `42`, not as a negative literal. This resolves the `f -42` ambiguity: it's `f - 42` (infix) because `-` has lower BP than application. Use parens for negative arguments: `f(-42)`.

### P10a: Literal Suffixes

Numeric literal tokens may carry an explicit precision suffix that binds the
literal at exactly that precision, with no inference, no widening, and no
narrowing. The closed suffix set is:

| Suffix | Bound type | Example | Notes |
|---|---|---|---|
| `f32` | `f32` | `1.0f32`, `42f32`, `3.14e-2f32` | Float-typed |
| `f64` | `f64` | `1.0f64` | Float-typed |
| `bf16` | `bf16` | `1.0bf16` | Float-typed |
| `f16` | `f16` | `1.0f16` | Float-typed |
| `i8` | `int8` | `42i8` | Integer-typed |
| `i16` | `int16` | `42i16` | Integer-typed |
| `i32` | `int32` | `42i32` | Integer-typed |
| `i64` | `int64` | `42i64` | Integer-typed |

Float-typed suffixes attach to either an integer or a float literal token.
Integer-typed suffixes attach to integer literal tokens only; `1.0i8` is a
parse error.

**Adjacency rule.** A suffix is part of the literal token only if it
**immediately** follows the digit sequence with no intervening whitespace,
comment, or other character. `1.0 f32` (with whitespace) is two tokens (a
float literal followed by an identifier-position token); the literal then
binds at the §P10 default and is subject to the surrounding-position rules in
the type checker.

**Hex-literal interaction.** The lexer's hex-literal rule consumes
`[0-9a-fA-F_]*` after `0x`. Because `f` is a hex digit, a hex integer
literal cannot directly carry a float-typed suffix. `0xFFf32` lexes as the
hex digit sequence `FFf` followed by integer `32`, which is rejected as a
malformed hex literal followed by a stray integer; the diagnostic suggests
either an explicit `cast` (`cast(0xFF, f32)`) or whitespace
(`0xFF f32`). Hex integer literals MAY carry integer-typed suffixes:
`0xFFi8`, `0xFFi32`. Decimal float literals carry float suffixes without
ambiguity (`1.0f32`, `1.0e3f32`).

**Deferred and out-of-scope suffixes.**

- `f8e4m3` is deferred per `spec/04-type-system.md` §1.1.1; the suffix
  `f8e4m3` is rejected at lex time with a diagnostic pointing at §1.1.1.
- Unsigned suffixes (`u8`, `u16`, `u32`, `u64`) are out of scope per
  `spec/04-type-system.md` §1.1.2; they are rejected at lex time with a
  diagnostic pointing at §1.1.2.
- An unrecognized identifier sequence directly adjacent to a numeric literal
  (e.g. `1.0xyz`) is a parse error rather than a silently-split
  literal-then-identifier pair. The diagnostic suggests adding whitespace
  if the adjacency was unintentional.

The suffix grammar is identical in Deep canonical form (`spec/03-deep-syntax.md`
§6.4); the Surf and Deep lexers parse the same token shape.

### P10b: Contextual Tensor-Literal Inference

When a tensor literal `[e1, e2, ...]` appears in a position with a **known
element type**, the unsuffixed numeric literals in the tensor body adopt that
element type instead of the §P10 default. The closed set of "known-element-type"
positions is exactly:

1. the right-hand side of a `let`-binding whose declared type is a tensor
   type — `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]` makes the literals
   bind at `f64`
2. the corresponding argument position of a call to a function with a
   declared signature whose parameter at that position is a tensor type
3. the body expression of a function with a declared return type that is a
   tensor type, when the body is itself a tensor literal
4. the first argument of an explicit `cast(literal, p)` expression — the
   literals bind at `p`

Outside this closed set, numeric literals in a tensor body fall back to the
§P10 literal defaults: integer literals to `int32`, float literals to `f32`.
A bare `[1, 2, 3]` in an unannotated top-level binding evaluates to
`tensor[3, int32]`; a bare `[1.0, 2.0, 3.0]` evaluates to `tensor[3, f32]`.

**Mixed suffixes inside a contextual literal.** A suffixed entry inside a
contextual tensor literal is well-formed only if its suffix matches the
inferred element type. `[1.0, 2.0f64, 3.0]` in an `f32`-context is a type
error: the f64-suffixed literal at index 1 has an explicit dtype that
disagrees with the surrounding `f32` element type. The diagnostic identifies
the offending index and suggests removing the suffix.

See `spec/04-type-system.md` §5.6 for the type-system statement of the
contextual-inference rule.

### P11: Strings

Double-quoted: `"hello world"`. Escapes: `\"`, `\\`, `\n`, `\t`, `\r`, `\0`. No multiline. No interpolation. No Unicode escapes in v1.

**⟹** `(lit {type: (t-prim {} string)} "hello world")`

### P12: Whitespace and Line Continuation

Newlines are NOT significant. They are whitespace. Any expression can break across lines freely:

```
x
  |> transform_a
  |> transform_b
```

Trailing commas allowed everywhere commas appear: parameter lists, argument lists, record fields, import lists, tuples. Parser ignores trailing comma before closing delimiter.

### P13: Modulo

`%` confirmed at BP 7 (with `*`, `/`). Works on integer types only. **⟹** `(app {} (var {} mod) a' b')`

### P14: Where Clauses

Not in v1.

### P15: Type Aliases

Distinguished from ADTs by absence of `|`.

```
type Weights = tensor[hidden, hidden, f32]
type Pair[a, b] = (a, b)
```

**⟹** `(typealias {} Weights () (t-tensor {} ...))` / `(typealias {} Pair (a b) (t-tuple {} ...))`

Parser disambiguation: after `type Name =`, if next non-whitespace is `|`, it's an ADT. Otherwise alias.

Aliases are transparent — expanded during desugaring. No opaque aliases in v1.

---

## 4. Formal Grammar (PEG)

```peg
# ═══════════════════════════════════════════════════
#  PROGRAM STRUCTURE
# ═══════════════════════════════════════════════════

Program       <- S ModuleDecl S Decl* EOF
ModuleDecl    <- 'module' S ModulePath

Decl          <- ImportDecl / ExportDecl / DimDecl
               / TypeDecl / TypeAlias / SigDecl / PropertyDecl / FunDecl

# ═══════════════════════════════════════════════════
#  MODULE, IMPORT, EXPORT
# ═══════════════════════════════════════════════════

ModulePath    <- TypeIdent ('.' TypeIdent)*

ImportDecl    <- 'import' S ModulePath S ImportSpec?
ImportSpec    <- '(' S ImportNames S ')'
ImportNames   <- '..'
               / IdentOrType (S ',' S IdentOrType)* (S ',')?
IdentOrType   <- TypeIdent / Ident

ExportDecl    <- 'export' S '(' S IdentOrType
                  (S ',' S IdentOrType)* (S ',')? S ')'

# ═══════════════════════════════════════════════════
#  DIMENSIONS
# ═══════════════════════════════════════════════════

DimDecl       <- 'dim' S Ident (S ',' S Ident)* (S ',')?

# ═══════════════════════════════════════════════════
#  TYPE DECLARATIONS
# ═══════════════════════════════════════════════════

TypeDecl      <- 'type' S TypeIdent TypeParams? S '='
                  S '|'? S Variant (S '|' S Variant)*
TypeAlias     <- 'type' S TypeIdent TypeParams? S '='
                  S !('|') TypeExpr
TypeParams    <- '[' S Ident (S ',' S Ident)* (S ',')? S ']'

Variant       <- TypeIdent RecordFields?
               / TypeIdent TupleFields?
RecordFields  <- '{' S FieldDecl (S ',' S FieldDecl)*
                  (S ',')? S '}'
TupleFields   <- '(' S TypeExpr (S ',' S TypeExpr)*
                  (S ',')? S ')'
FieldDecl     <- Ident S ':' S TypeExpr

# ═══════════════════════════════════════════════════
#  TYPE SIGNATURES
# ═══════════════════════════════════════════════════

SigDecl       <- 'sig' S Ident S ':' S TypeExpr

# ═══════════════════════════════════════════════════
#  PROPERTY DECLARATIONS
# ═══════════════════════════════════════════════════

PropertyDecl  <- '@property' S Ident S 'forall' S Params
                 (S 'where' S Expr (S ',' S Expr)*)?
                 S ':' S Expr PropertyOption*
PropertyOption <- S 'with' S ('tolerance' / 'seed' / 'samples') S '=' S Expr

# ═══════════════════════════════════════════════════
#  FUNCTION DEFINITIONS
# ═══════════════════════════════════════════════════

FunDecl       <- 'def' S Ident DimParams? Params?
                  ReturnType? S '=' S Expr

DimParams     <- '[' S Ident (S ',' S Ident)* (S ',')? S ']'
Params        <- '(' S Param (S ',' S Param)* (S ',')? S ')'
Param         <- Ident (S ':' S TypeExpr)?
ReturnType    <- S ':' S TypeExpr

# ═══════════════════════════════════════════════════
#  TYPE EXPRESSIONS
# ═══════════════════════════════════════════════════

TypeExpr      <- FnType
FnType        <- TypeApp (S '->' S FnType)?

TypeApp       <- TypeAtom TypeArgs?
TypeArgs      <- '[' S TypeExprOrDim (S ',' S TypeExprOrDim)*
                  (S ',')? S ']'
TypeExprOrDim <- TypeExpr / DimExpr

TypeAtom      <- 'tensor' '[' S DimList S ',' S PrecType S ']'
               / PrecType
               / '(' S ')'
               / '(' S TypeExpr S ',' S TypeExpr
                  (S ',' S TypeExpr)* (S ',')? S ')'
               / '(' S TypeExpr S ')'
               / TypeIdent

PrecType      <- 'f32' / 'f64' / 'bf16' / 'f16'
               / 'int8' / 'int16' / 'int32' / 'int64'
               / 'bool' / 'string'
               # f8e4m3 is reserved/deferred per spec/04-type-system.md §1.1.1
               # unsigned types (u8/u16/u32/u64) are out of scope per §1.1.2

DimList       <- DimExpr (S ',' S DimExpr)*
DimExpr       <- IntLit / Ident

# ═══════════════════════════════════════════════════
#  EXPRESSIONS (Pratt parser)
# ═══════════════════════════════════════════════════

Expr          <- MatchExpr / IfExpr / FnExpr
               / PipeExpr

LetPattern    <- '(' S Ident (S ',' S Ident)+ (S ',')? S ')'
               / Ident (S ':' S TypeExpr)?

MatchExpr     <- 'match' S Expr S 'with' S '{' S MatchArms S '}'
MatchArms     <- MatchArm (S MatchArm)*
MatchArm      <- '|' S Pattern Guard? S '=>' S Expr
Guard         <- S 'if' S Expr

IfExpr        <- 'if' S Expr S 'then' S Expr S 'else' S Expr
FnExpr        <- 'fn' S Params S '->' S Expr

# ── Operator expressions ──

PipeExpr      <- OrExpr (S '|>' S OrExpr)*
OrExpr        <- AndExpr (S '||' S AndExpr)*
AndExpr       <- CmpExpr (S '&&' S CmpExpr)*
CmpExpr       <- AddExpr (S CmpOp S AddExpr)?
AddExpr       <- MulExpr (S ('+' / '-') S MulExpr)*
MulExpr       <- UnaryExpr (S ('*' / '/' / '%') S UnaryExpr)*
UnaryExpr     <- ('-' / '!') S UnaryExpr / AccessExpr

# ── Postfix ──

AccessExpr    <- AppExpr ('.' (Ident / IntLit))*
AppExpr       <- AtomExpr (S !InfixOp AtomExpr)*

# ── Atoms ──

AtomExpr      <- '(' S ')'
               / '(' S Expr S ',' S Expr
                  (S ',' S Expr)* (S ',')? S ')'
               / '(' S Expr S ')'
               / BlockExpr
               / TransformExpr
               / RecordExpr
               / Literal
               / Ident
               / TypeIdent

BlockExpr     <- '{' S BlockBody S '}'
BlockBody     <- (BlockBinding Sep)* Expr
BlockBinding  <- LetPattern S '=' S Expr
Sep           <- (S ';' S) / (S Newline S)

TransformExpr <- TransformKw S '(' S Expr
                  (S ',' S TransformArg)? (S ',')? S ')'
TransformKw   <- 'grad' / 'vmap' / 'jit' / 'realize'
               / 'cast' / 'copy'
TransformArg  <- PrecType / Expr

RecordExpr    <- TypeIdent S '{' S RecordField
                  (S ',' S RecordField)* (S ',')? S '}'
RecordField   <- Ident S ':' S Expr / Ident

# ═══════════════════════════════════════════════════
#  PATTERNS
# ═══════════════════════════════════════════════════

Pattern       <- PatAtom (S 'as' S Ident)?

PatAtom       <- '(' S Pattern (S ',' S Pattern)+
                  (S ',')? S ')'
               / '(' S Pattern S ')'
               / TypeIdent S '{' S RecordPatField
                  (S ',' S RecordPatField)* (S ',')? S '}'
               / TypeIdent PatAtom*
               / Literal
               / '_'
               / Ident

RecordPatField <- Ident S ':' S Pattern / Ident

# ═══════════════════════════════════════════════════
#  OPERATORS (for lookahead)
# ═══════════════════════════════════════════════════

CmpOp         <- '==' / '!=' / '<=' / '>=' / '<' / '>'
InfixOp       <- '|>' / '||' / '&&' / CmpOp
               / '+' / '-' / '*' / '/' / '%'

# ═══════════════════════════════════════════════════
#  LITERALS
# ═══════════════════════════════════════════════════

Literal       <- FloatLit / IntLit / BoolLit / StringLit

FloatLit      <- ('-'? Digits '.' Digits Exponent? / '-'? Digits Exponent) FloatSuffix?
Exponent      <- [eE] [+-]? Digits
IntLit        <- '-'? Digits !('.' [0-9]) ![eE] (FloatSuffix / IntSuffix)?
Digits        <- [0-9] ([0-9_]* [0-9])?
FloatSuffix   <- 'f32' / 'f64' / 'bf16' / 'f16'
IntSuffix     <- 'i8' / 'i16' / 'i32' / 'i64'
# Suffix must immediately follow the digit sequence (no whitespace, no comment).
# Closed sets: any other identifier sequence directly adjacent to a numeric
# literal (e.g. `1.0xyz`, `42u8`, `1.0f8e4m3`) is a parse error per P10a.
BoolLit       <- 'true' / 'false'
StringLit     <- '"' StringChar* '"'
StringChar    <- '\\' [nrt0"\\] / !'"' .

# ═══════════════════════════════════════════════════
#  IDENTIFIERS
# ═══════════════════════════════════════════════════

Ident         <- !Keyword [a-z_] [a-zA-Z0-9_]*
TypeIdent     <- !Keyword [A-Z] [a-zA-Z0-9]*

Keyword       <- ('def' / 'sig' / 'type' / 'dim'
               / 'match' / 'with' / 'fn' / 'module' / 'import'
               / 'export' / 'if' / 'then' / 'else' / 'grad'
               / 'vmap' / 'jit' / 'cast' / 'realize' / 'copy'
               / 'par' / 'true' / 'false' / 'tensor'
               / 'effect' / 'handler' / 'perform' / 'resume'
               / 'borrow' / 'where' / 'do'
               ) ![a-zA-Z0-9_]

# ═══════════════════════════════════════════════════
#  WHITESPACE & COMMENTS
# ═══════════════════════════════════════════════════

S             <- ([ \t\n\r] / Comment)*
Newline       <- '\n' / '\r\n' / '\r'
Comment       <- LineComment / BlockComment
LineComment   <- '--' (!Newline .)*
BlockComment  <- '{-' (BlockComment / !'-}' .)* '-}'
EOF           <- !.
```

---

## 5. Complete Desugaring Reference

Primed names (`a'`) denote recursive desugaring of subexpressions. All operators desugar to derived built-in function calls via `(app {} (var {} name) ...)`. The compiler lowers derived built-ins to RISC primitives during IR construction — the desugarer does NOT decompose them further.

### 5.1 Module Structure

```
module Foo.Bar                    ⟹  (module {} foo.bar <decls...>)
import Foo.Bar (baz, qux)        ⟹  (import {} foo.bar (baz qux))
import Foo.Bar (..)              ⟹  (import-all {} foo.bar)
import Foo.Bar                   ⟹  (import {} foo.bar ())
export (f, g, MyType)            ⟹  (export {} f g MyType)
dim batch, seq                   ⟹  (defdim {} batch) (defdim {} seq)
```

### 5.2 Declarations

```
sig f: f32 -> f32 -> f32
⟹  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))

def f(x: f32, y: f32) -> f32 = add(x, y)
⟹  (defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
    (def {} f (fn {} (params {} (x {type: (t-prim {} f32)})
                           (y {type: (t-prim {} f32)}))
                   (app {} (var {} add) (var {} x) (var {} y))))

def f(x, y) = add(x, y)
⟹  (def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))

type Option[a] = | None | Some { value: a }
⟹  (deftype {} Option (a)
      (variant {} None)
      (variant {} Some (field {} value (t-var {} a))))

type Weights = tensor[h, h, f32]
⟹  (typealias {} Weights () (t-tensor {} (d-name {} h) (d-name {} h) (t-prim {} f32)))
```

### 5.3 Expressions

```
-- Variables, literals
x                                 ⟹  (var {} x)
42                                ⟹  (lit {type: (t-prim {} int32)} 42)
3.14                              ⟹  (lit {type: (t-prim {} f32)} 3.14)
true                              ⟹  (lit {type: (t-prim {} bool)} true)
"hello"                           ⟹  (lit {type: (t-prim {} string)} "hello")
()                                ⟹  (lit {type: (t-unit {})} ())

-- Application
f(x, y)                           ⟹  (app {} (var {} f) x' y')
f x y                             ⟹  (app {} (var {} f) x' y')

-- Arithmetic (all via derived built-ins)
a + b                             ⟹  (app {} (var {} add) a' b')
a - b                             ⟹  (app {} (var {} sub) a' b')
a * b                             ⟹  (app {} (var {} mul) a' b')
a / b                             ⟹  (app {} (var {} div) a' b')
a % b                             ⟹  (app {} (var {} mod) a' b')
-a                                ⟹  (app {} (var {} neg) a')

-- Comparison
a == b                            ⟹  (app {} (var {} eq) a' b')
a != b                            ⟹  (app {} (var {} neq) a' b')
a < b                             ⟹  (app {} (var {} cmplt) a' b')
a > b                             ⟹  (app {} (var {} cmplt) b' a')
a <= b                            ⟹  (app {} (var {} lte) a' b')
a >= b                            ⟹  (app {} (var {} gte) a' b')

-- Logical
a && b                            ⟹  (app {} (var {} and) a' b')
a || b                            ⟹  (app {} (var {} or) a' b')
!a                                ⟹  (app {} (var {} not) a')

-- Pipe
x |> f |> g                       ⟹  (pipe {} x' (var {} f) (var {} g))

-- Control flow
if c then a else b                ⟹  (if {} c' a' b')
fn (x, y) -> body                 ⟹  (fn {} (params {} x y) body')

-- Block (sequential bindings)
{                                 ⟹  (let {} (bind x e1' y e2') body')
  x = e1
  y = e2
  body
}

-- Match
match e with {                    ⟹  (match {} e'
  | P1 => b1                           (arm {} P1' () b1')
  | P2 if g => b2                      (arm {} P2' g' b2'))
}

-- Tuples
(a, b, c)                         ⟹  (tuple {} a' b' c')
pair.0                            ⟹  (tuple-get {} (var {} pair) (lit {type: (t-prim {} int32)} 0))

-- Records
Foo { x: e1, y: e2 }             ⟹  (record {} Foo (kv {} x e1') (kv {} y e2'))
Foo { x, y }                      ⟹  (record {} Foo (kv {} x (var {} x)) (kv {} y (var {} y)))
e.field                           ⟹  (access {} e' field)

-- Transforms
grad(f)                           ⟹  (grad {} f')
jit(f)                            ⟹  (jit {} f')
vmap(f, n)                        ⟹  (vmap {} f' n')
vmap(f)                           ⟹  (vmap {} f' (lit {type: (t-prim {} int32)} 0))
cast(e, bf16)                     ⟹  (cast {} e' (t-prim {} bf16))
realize(e)                        ⟹  (realize {} e')
copy(e)                           ⟹  (copy {} e')
&x                                ⟹  (borrow {} (var {} x))
```

### 5.4 Type Expressions

```
f32                               ⟹  (t-prim {} f32)
bool                              ⟹  (t-prim {} bool)
string                            ⟹  (t-prim {} string)
tensor[batch, seq, f32]           ⟹  (t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
tensor[a, b, f32]  (polymorphic)  ⟹  (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
&tensor[batch, f32]               ⟹  (t-ref {} (t-tensor {} (d-name {} batch) (t-prim {} f32)))
A -> B -> C                       ⟹  (t-fn {} A' B' C')  -- flat, last is return
Option[f32]                       ⟹  (t-adt {} Option (t-prim {} f32))
(f32, f32)                        ⟹  (t-tuple {} (t-prim {} f32) (t-prim {} f32))
()                                ⟹  (t-unit {})
```

### 5.5 Patterns

```
x                                 ⟹  (pat-var {} x)
_                                 ⟹  (pat-wild {})
42                                ⟹  (pat-lit {} 42)
Some(x)                           ⟹  (pat-ctor {} Some (pat-var {} x))
Some(Some(x))                     ⟹  (pat-ctor {} Some (pat-ctor {} Some (pat-var {} x)))
Foo { x, y }                      ⟹  (pat-record {} Foo (kv {} x (pat-var {} x)) (kv {} y (pat-var {} y)))
(a, b)                            ⟹  (pat-tuple {} (pat-var {} a) (pat-var {} b))
x @ Some(_)                       ⟹  (pat-as {} x (pat-ctor {} Some (pat-wild {})))
```

---

## 6. Parser Implementation Notes

### 6.1 Juxtaposition vs Infix Disambiguation

`AppExpr` uses `!InfixOp` lookahead to stop juxtaposition when an infix operator follows. `f x + y` parses as `(f x) + y` because application (BP 9) binds tighter than `+` (BP 6).

### 6.2 Bindings

Surf has one binding surface:

1. **Top level:** bare declarations such as `result = expr`.
2. **Inside a block `{ ... }`:** sequential bindings such as `x = expr` and `(a, b) = pair`.

There is no Surf `let ... in` expression form. `let` and `in` are ordinary identifiers.

### 6.3 Negative Literals vs Unary Minus

`-42` is always unary minus applied to `42`. The literal itself is non-negative. `f -42` parses as `f - 42` (infix). Use parens for negative arguments: `f(-42)`. During constant folding, `neg(42)` collapses to a negative literal in Deep.

### 6.4 Transform Recognition

`grad`, `vmap`, `jit`, `cast`, `realize`, `copy` are keywords. In call position (`keyword(`), the parser emits a transform node. Bare usage (`g = grad`) is a parse error — transforms must always be applied.

### 6.5 TypeIdent in Expression Position

Uppercase names in expression position are ADT constructors. Nullary: `None`. Record: `Adam { lr: 0.001 }`. Tuple-variant: `Some(x)`.

### 6.6 `type` Disambiguation

After `type Name =`, the parser checks if the next non-whitespace token is `|`. If yes → ADT (`TypeDecl`). If no → alias (`TypeAlias`).

---

## 7. Deep Tag Additions

This spec requires two new Deep tags in addition to the post-sprint baseline, and it reserves one more for the deferred Phase 1 record-update surface:

| Tag | Form | Semantics |
|-----|------|-----------|
| `typealias` | `(typealias {} Name (params...) type-expr)` | Transparent type alias |
| `record-update` | `(record-update {} base-expr (kv {} field expr) ...)` | Functional record update (reserved; Phase 1) |

Additionally, `pat-tuple` is needed for tuple destructuring patterns:

| Tag | Form | Semantics |
|-----|------|-----------|
| `pat-tuple` | `(pat-tuple {} pat₁ pat₂ ...)` | Tuple pattern |

**Revised Deep tag total: 56** (baseline + `typealias` + `record-update` + `pat-tuple`).

---

## 8. Example Programs

### 8.1 Linear Regression

```
module LinReg

dim features, samples

sig predict:
  tensor[samples, features, f32]
  -> tensor[features, 1, f32]
  -> tensor[1, f32]
  -> tensor[samples, 1, f32]

def predict(
  x: tensor[samples, features, f32],
  w: tensor[features, 1, f32],
  b: tensor[1, f32]
) -> tensor[samples, 1, f32] =
  add(matmul(x, w), expand(b, 0, samples))

def mse_loss(
  y_pred: tensor[samples, 1, f32],
  y_true: tensor[samples, 1, f32]
) -> tensor[f32] = {
  diff = sub(y_pred, y_true)
  sq = mul(diff, diff)
  mean(mean(sq, 1), 0)
}

def train_step(w, b, x, y, lr) = {
  loss_fn = fn (w_, b_) -> mse_loss(predict(x, w_, b_), y)
  (dw, db) = grad(loss_fn)(w, b)
  w_new = sub(w, mul(lr, dw))
  b_new = sub(b, mul(lr, db))
  (w_new, b_new)
}
```

### 8.2 MLP with Activation ADT

```
module Mlp

dim batch, input_dim, hidden_dim, output_dim

type Activation = | ReLU | Sigmoid

def activate(act: Activation, x: tensor[batch, hidden_dim, f32]) -> tensor[batch, hidden_dim, f32] =
  match act with {
    | ReLU    => relu(x)
    | Sigmoid => sigmoid(x)
  }

def forward(w1, b1, w2, b2, act, x) = {
  h = matmul(x, w1)
    |> fn (z) -> add(z, b1)
    |> fn (z) -> activate(act, z)
  add(matmul(h, w2), b2)
}
```

### 8.3 Records and Pattern Matching

```
module Optimizer

type Optimizer =
  | Sgd { lr: f32 }
  | Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 }

default_adam =
  Adam { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1.0e-8 }

def learning_rate(opt) =
  match opt with {
    | Sgd { lr }   => lr
    | Adam { lr }  => lr
  }

def scale_lr(opt, factor) =
  match opt with {
    | Sgd { lr } =>
        Sgd { lr: mul(lr, factor) }
    | Adam { lr, beta1, beta2, eps } =>
        Adam { lr: mul(lr, factor), beta1, beta2, eps }
  }
```

### 8.4 Dimension Polymorphism

```
module Linalg

def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] =
  permute(x, [1, 0])

def dot[n](x: tensor[n, f32], y: tensor[n, f32]) -> f32 =
  sum(mul(x, y))

def normalize[d](x: tensor[d, f32]) -> tensor[d, f32] = {
  norm = sqrt(sum(mul(x, x)))
  div(x, expand(norm, 0))
}
```

### 8.5 Training Loop with Grad

```
module Train

import Std.Io (println)
import Std.Iter (fold, range)   -- stdlib helpers, not built-ins

def train(model_w, model_b, data_x, data_y, lr, epochs) = {
  step = fn (wb, i) -> {
    (w, b) = wb
    loss_fn = fn (w_, b_) -> mse_loss(predict(data_x, w_, b_), data_y)
    (dw, db) = grad(loss_fn)(w, b)
    (sub(w, mul(lr, dw)), sub(b, mul(lr, db)))
  }
  fold(step, (model_w, model_b), range(0, epochs))
}
```
