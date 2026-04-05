# spec/02-surf-syntax.md — Chelis Surf Syntax Specification

**Status:** v0.3 (unified from design sprint, authoritative)

---

## 0. Notation

PEG notation. `/` is ordered choice. `*` is zero-or-more. `+` is one-or-more. `?` is optional. `!` is negative lookahead. `&` is positive lookahead. Literal strings in single quotes.

Deep desugaring shown as **⟹** with the target Deep s-expression.

---

## 1. Keywords

Reserved. Cannot be used as identifiers.

```
def  sig  let  in  type  dim  match  with  fn  module
import  export  if  then  else  grad  vmap  jit  cast
realize  copy  par  true  false
```

**Total: 24.**

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

No operator overloading. No bitwise operators. No exponentiation operator — use `pow(x, n)` from `Std.Math`.

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

def transpose[a, b](x: tensor[a, b, f32]): tensor[b, a, f32] =
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
def add_vecs(x: tensor[d, f32], y: tensor[d, f32]): tensor[d, f32] = add(x, y)
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

Omitting all types is valid: `def f(x, y) = add(x, y)`. The compiler emits a note recommending a `sig` for module-level definitions.

### P5: Blocks and Sequencing

Braces define blocks. Inside blocks, `let` bindings are sequential — no `in` required. Newlines and semicolons are both valid separators. The final expression is the block's value.

**Two `let` forms:**

| Form | Context | Example |
|------|---------|---------|
| `let x = e in body` | Anywhere | `let x = 1 in x + 1` |
| `let x = e` (no `in`) | Inside braces only | `{ let x = 1; x + 1 }` |

```
{
  let x = f(a)
  let y = g(x)
  let z = h(y)
  combine(x, y, z)
}
```

**⟹** All sequential `let` bindings collapse into a single Deep node:
```
(let {} (bind x (app {} (var {} f) (var {} a))
              y (app {} (var {} g) (var {} x))
              z (app {} (var {} h) (var {} y)))
  (app {} (var {} combine) (var {} x) (var {} y) (var {} z)))
```

Blocks are expressions: `let result = { let temp = f(x); g(temp) }` is valid.

A separator (newline or semicolon) is required between a `let` binding and the next statement. Multiple separators (blank lines) are fine. Trailing semicolon after the final expression is tolerated.

No `where` clauses. Use `let...in` or blocks.

### P6: Records

Braces for construction. Punning allowed. Dot-chaining for access. Functional update with `with` is reserved for Phase 1 and not part of the Phase 0 parser/desugarer.

```
let lr = 0.01
let opt = Adam { lr, eps: 1.0e-8 }      -- punning: lr: lr
let rate = opt.lr                         -- field access
let chain = model.layer1.weight           -- chained access
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
let pair = (w_new, b_new)
let w = pair.0
let b = pair.1
```

**⟹**
```
(a, b)   ⟹  (tuple {} a' b')
pair.0   ⟹  (tuple-get {} (var {} pair) (lit {type: (t-prim {} int32)} 0))
()       ⟹  (lit {type: (t-unit {})} ())
```

Destructuring via `let`:
```
let (w, b) = train_step(w, b, x, y, lr)
```

### P9: Transforms

Transforms use call syntax in Surf but desugar to dedicated Deep tags. The parser recognizes these keywords and emits transform nodes, not `app` nodes.

| Surf | Deep | Notes |
|------|------|-------|
| `grad(f)` | `(grad {} f')` | Accepts named functions or inline lambdas |
| `jit(f)` | `(jit {} f')` | |
| `vmap(f, n)` | `(vmap {} f' n')` | Axis positional, defaults to 0 if omitted |
| `vmap(f)` | `(vmap {} f' (lit {type: (t-prim {} int32)} 0))` | |
| `cast(e, bf16)` | `(cast {} e' (t-prim {} bf16))` | Second arg is a type literal (special form) |
| `realize(e)` | `(realize {} e')` | |
| `copy(e)` | `(copy {} e')` | |

Transforms compose naturally: `jit(grad(loss_fn))` **⟹** `(jit {} (grad {} (var {} loss_fn)))`.

Transforms must always be applied — `let g = grad` bare is a parse error.

The second argument to `cast` is a precision type literal (`f32`, `bf16`, etc.) in expression position. This is the one special form where a type appears as an argument.

### P10: Numeric Literals

Default float precision: **f32**. Default integer type: **int32**.

Underscore separators: `1_000_000`, `3.141_592_6`. Stripped during lexing.

Scientific notation: `1e-5`, `3.14e10`. Canonical Deep form: `d.dE±d`.

No hex, octal, or binary literals. Not needed for ML workloads.

Integer range: i64 at parse time, narrowed to int32/int8 during type checking.

**Negative literals:** `-42` is always parsed as unary minus applied to `42`, not as a negative literal. This resolves the `f -42` ambiguity: it's `f - 42` (infix) because `-` has lower BP than application. Use parens for negative arguments: `f(-42)`.

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
               / TypeDecl / TypeAlias / SigDecl / FunDecl

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

PrecType      <- 'f32' / 'f64' / 'f16' / 'bf16' / 'f8e4m3'
               / 'int8' / 'int32' / 'int64' / 'bool' / 'string'

DimList       <- DimExpr (S ',' S DimExpr)*
DimExpr       <- IntLit / Ident

# ═══════════════════════════════════════════════════
#  EXPRESSIONS (Pratt parser)
# ═══════════════════════════════════════════════════

Expr          <- LetExpr / MatchExpr / IfExpr / FnExpr
               / PipeExpr

LetExpr       <- 'let' S LetPattern S '=' S Expr S LetCont
LetCont       <- 'in' S Expr
               / &(Sep LetOrExpr)

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
BlockBody     <- (LetBinding Sep)* Expr
LetBinding    <- 'let' S LetPattern S '=' S Expr
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

FloatLit      <- '-'? Digits '.' Digits Exponent?
               / '-'? Digits Exponent
Exponent      <- [eE] [+-]? Digits
IntLit        <- '-'? Digits !('.' [0-9]) ![eE]
Digits        <- [0-9] ([0-9_]* [0-9])?
BoolLit       <- 'true' / 'false'
StringLit     <- '"' StringChar* '"'
StringChar    <- '\\' [nrt0"\\] / !'"' .

# ═══════════════════════════════════════════════════
#  IDENTIFIERS
# ═══════════════════════════════════════════════════

Ident         <- !Keyword [a-z_] [a-zA-Z0-9_]*
TypeIdent     <- !Keyword [A-Z] [a-zA-Z0-9]*

Keyword       <- ('def' / 'sig' / 'let' / 'in' / 'type' / 'dim'
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

def f(x: f32, y: f32): f32 = add(x, y)
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
let x = e in body                 ⟹  (let {} (bind x e') body')
fn (x, y) -> body                 ⟹  (fn {} (params {} x y) body')

-- Block (sequential let)
{                                 ⟹  (let {} (bind x e1' y e2') body')
  let x = e1
  let y = e2
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
```

### 5.4 Type Expressions

```
f32                               ⟹  (t-prim {} f32)
bool                              ⟹  (t-prim {} bool)
string                            ⟹  (t-prim {} string)
tensor[batch, seq, f32]           ⟹  (t-tensor {} (d-name {} batch) (d-name {} seq) (t-prim {} f32))
tensor[a, b, f32]  (polymorphic)  ⟹  (t-tensor {} (d-var {} a) (d-var {} b) (t-prim {} f32))
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

### 6.2 `let` Context Sensitivity

The `let` keyword has two parsing modes:

1. **Expression level:** Parse `let P = E in body` — `in` is required.
2. **Inside a block `{ ... }`:** Parse `let P = E` — no `in`. Sequential with next statement.

The parser tracks a boolean flag for "inside block." This is the one context-sensitive rule in Surf.

### 6.3 Negative Literals vs Unary Minus

`-42` is always unary minus applied to `42`. The literal itself is non-negative. `f -42` parses as `f - 42` (infix). Use parens for negative arguments: `f(-42)`. During constant folding, `neg(42)` collapses to a negative literal in Deep.

### 6.4 Transform Recognition

`grad`, `vmap`, `jit`, `cast`, `realize`, `copy` are keywords. In call position (`keyword(`), the parser emits a transform node. Bare usage (`let g = grad`) is a parse error — transforms must always be applied.

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

def predict(x, w, b) =
  matmul(x, w) |> fn (y) -> add(y, b)

def mse_loss(y_pred, y_true) = {
  let diff = sub(y_pred, y_true)
  let sq = mul(diff, diff)
  mean(sq)
}

def train_step(w, b, x, y, lr) = {
  let loss_fn = fn (w_, b_) -> mse_loss(predict(x, w_, b_), y)
  let (dw, db) = grad(loss_fn)(w, b)
  let w_new = sub(w, mul(lr, dw))
  let b_new = sub(b, mul(lr, db))
  (w_new, b_new)
}
```

### 8.2 MLP with Activation ADT

```
module Mlp

dim batch, input_dim, hidden_dim, output_dim

type Activation = | ReLU | Sigmoid

def activate(act, x) =
  match act with {
    | ReLU    => relu(x)
    | Sigmoid => sigmoid(x)
  }

def forward(w1, b1, w2, b2, act, x) = {
  let h = matmul(x, w1)
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

def default_adam =
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

def transpose[a, b](x: tensor[a, b, f32]): tensor[b, a, f32] =
  permute(x, [1, 0])

def dot[n](x: tensor[n, f32], y: tensor[n, f32]): f32 =
  sum(mul(x, y))

def normalize[d](x: tensor[d, f32]): tensor[d, f32] = {
  let norm = sqrt(sum(mul(x, x)))
  div(x, expand(norm, 0))
}
```

### 8.5 Training Loop with Grad

```
module Train

import Std.Io (println)
import Std.Iter (fold, range)   -- stdlib helpers, not built-ins

def train(model_w, model_b, data_x, data_y, lr, epochs) = {
  let step = fn (wb, i) -> {
    let (w, b) = wb
    let loss_fn = fn (w_, b_) -> mse_loss(predict(data_x, w_, b_), data_y)
    let (dw, db) = grad(loss_fn)(w, b)
    (sub(w, mul(lr, dw)), sub(b, mul(lr, db)))
  }
  fold(step, (model_w, model_b), range(0, epochs))
}
```
