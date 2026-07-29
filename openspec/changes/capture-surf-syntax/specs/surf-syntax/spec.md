## ADDED Requirements

### Requirement: Reserved keywords

Surf SHALL reserve its keyword set so reserved words cannot be used as identifiers. The
Phase-2 reserved words (`effect`, `handler`, `perform`, `resume`, `borrow`, `where`, `do`)
SHALL parse as keywords and emit a "reserved for future use" error rather than binding.

#### Scenario: Reserved word is not an identifier

- **WHEN** source attempts `def def(x) = x`
- **THEN** parsing fails because `def` is a reserved keyword

#### Scenario: Phase-2 keyword is reserved

- **WHEN** source uses `perform` as an identifier
- **THEN** the parser reports it as reserved for future use rather than accepting it as a name

### Requirement: Operator precedence and non-associativity

Surf operators SHALL bind according to the fixed precedence table (pipe lowest through field
access highest), and the non-associative comparison/equality operators (BP 4 and 5) SHALL
reject chaining. There SHALL be no operator overloading, no infix bitwise operators, and no
exponentiation operator.

#### Scenario: Application binds tighter than addition

- **WHEN** source is `f x + y`
- **THEN** it parses as `(f x) + y` because application (BP 9) binds tighter than `+` (BP 6)

#### Scenario: Chained comparison is rejected

- **WHEN** source is `a == b == c`
- **THEN** the parser reports an error because `==` is non-associative

### Requirement: Division is float-only

The `/` operator SHALL desugar to `div`, which is float-only; applying `/` or `div` to
integer operands SHALL be a type error. Integer division SHALL use the named built-ins
`floor_div` (toward −∞) and `trunc_div` (toward zero); `%` SHALL map to `mod` on integer
types only.

#### Scenario: Float division is accepted

- **WHEN** two `f32` values are divided with `/`
- **THEN** it desugars to `div` and type-checks

#### Scenario: Integer division with `/` is a type error

- **WHEN** two integer operands are divided with `/`
- **THEN** the type checker rejects it and directs the user to `floor_div` or `trunc_div`

### Requirement: One module per file

A Surf file SHALL declare exactly one `module Name` as its first non-comment line, with
PascalCase components and no nesting within a file. `module Foo.Bar` SHALL live at
`foo/bar.ch` and desugar to `(module {} foo.bar ...)`.

#### Scenario: Module declaration desugars to lowercased path

- **WHEN** a file begins `module School.Nn.Linear`
- **THEN** it desugars to `(module {} school.nn.linear ...)`

#### Scenario: Lowercase module name is rejected

- **WHEN** a file begins `module linreg`
- **THEN** parsing fails because module name components must be PascalCase

### Requirement: Import and export forms

Surf SHALL support `import Foo.Bar (names)`, `import Foo.Bar (..)`, and qualified-only
`import Foo.Bar`, with qualified access always available after any import. Exports SHALL be
implicit (all top-level `def`/`type` public) until any `export` appears, after which only
listed names are public; exporting a `type` SHALL also export its constructors.

#### Scenario: Selective import brings names unqualified

- **WHEN** a module writes `import Std.Tensor (matmul)`
- **THEN** `matmul` is usable unqualified and `Std.Tensor.matmul` remains available qualified

#### Scenario: Non-exported name is not importable unqualified

- **WHEN** a module declares `export (forward)` and another imports a sibling `helper`
- **THEN** `helper` is not public and importing it is rejected

### Requirement: Constructor and qualified-reference scope

A bare constructor reference SHALL be in scope only when declared in the current module or
named in an import list; importing only the enclosing type SHALL NOT bring the constructor
into unqualified scope. A qualified reference whose module does not export the trailing name
SHALL be rejected with a "does not export" error, and an ambiguous unqualified reference
SHALL be rejected rather than silently bound.

#### Scenario: Qualified constructor disambiguates same-named variants

- **WHEN** two imported modules each export `Eval` and the user writes `Demo.Dropout.Eval`
- **THEN** the reference resolves to that module's constructor

#### Scenario: Out-of-scope constructor is an unknown-constructor error

- **WHEN** a match pattern names `Alpha` that is neither declared locally nor imported by name
- **THEN** `chelis check` reports `unknown constructor` rather than binding it to another module's `Alpha`

### Requirement: Dimension declarations and polymorphic dimension parameters

Module-level `dim` SHALL declare concrete dimensions (lowering to `d-name`), and a function's
`[...]` clause SHALL introduce polymorphic dimension variables (lowering to `d-var`). A
lowercase name in a tensor type that is neither declared nor bracket-bound SHALL be a parse
error.

#### Scenario: Bracket parameter is a dimension variable

- **WHEN** a function is `def transpose[a, b](x: tensor[a, b, f32]) -> tensor[b, a, f32] = ...`
- **THEN** `a` and `b` lower to `d-var`

#### Scenario: Undeclared dimension name is rejected

- **WHEN** a tensor type uses a lowercase dimension `q` that is neither `dim`-declared, imported, nor bracket-bound
- **THEN** parsing fails

### Requirement: Rank variables

A rank variable `..r` SHALL stand for a name-preserving run of dimensions, introduced
contextually without an `[..r]` quantifier. A spread name SHALL NOT repeat within one tensor
shape, and a `def` mentioning `..r` SHALL be restricted to name-trackable operations
(elementwise and named-axis reductions), never positional shape-rewriters.

#### Scenario: Rank-generic identity function

- **WHEN** a function is `def relu_forward(x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)`
- **THEN** it type-checks at every rank because `relu` is shape-identity

#### Scenario: Repeated spread name is a parse error

- **WHEN** a tensor shape uses the same spread name twice, e.g. `tensor[..r, ..r, f32]`
- **THEN** parsing fails

### Requirement: Type signatures and arrow associativity

Surf SHALL support inline (`def`) and standalone (`sig`) signatures; a `sig` SHALL precede
its `def`. The `->` arrow SHALL be right-associative and flat in Deep (`t-fn` with last child
the return type), and a function-typed argument SHALL require parentheses, preserved by the
formatter and decompiler.

#### Scenario: Inline annotations synthesize a defsig

- **WHEN** a function is `def add_vecs(x: tensor[d, f32], y: tensor[d, f32]) -> tensor[d, f32] = add(x, y)`
- **THEN** the desugarer emits a `defsig` in addition to the `def`

#### Scenario: Parenthesized function argument is distinct from curried arrows

- **WHEN** a type is written `(a -> b) -> c`
- **THEN** it is a 1-ary type whose single argument is a function, distinct from `a -> b -> c`, and the grouping parentheses are preserved

### Requirement: Effect annotations

A `sig` or `def` SHALL accept an optional `! { ... }` effect suffix drawing from the built-in
names `Diff`, `Random`, `Accum`, `IO`, and `Resource("device")`. `Random`, `IO`, and
`Resource(...)` SHALL be the active boundary effects in the shipped subset, while `Diff` and
`Accum` SHALL be accepted as forward-compatible syntax.

#### Scenario: Random effect annotation is accepted

- **WHEN** a sig is `sig predict: tensor[n, f32] -> tensor[n, f32] ! { Random }`
- **THEN** the parser accepts the effect suffix

#### Scenario: Unknown effect name is rejected

- **WHEN** an effect suffix names an effect outside the built-in set, e.g. `! { Bogus }`
- **THEN** the parser/checker rejects it

### Requirement: Contextual precision polymorphism

In a `sig`, any non-primitive lowercase name in a tensor precision slot SHALL be an implicit
`forall`-quantified type variable (`t-var`); in a `def`, a precision name SHALL be promoted
to `t-var` only if it appears in the def's `[..]` clause, otherwise it SHALL surface an
`UnsupportedTensorPrecision` diagnostic. The `[..]` clause SHALL override the case-split so a
quantified PascalCase name is a type variable.

#### Scenario: Sig precision name is an implicit type variable

- **WHEN** a sig is `sig poly_id: tensor[d, p] -> tensor[d, p]`
- **THEN** `p` is an implicitly quantified precision variable instantiated fresh per call site

#### Scenario: Unbound def precision name is rejected

- **WHEN** a def is `def f[a, b](x: tensor[3, p]) = ...` with `p` absent from `[a, b]`
- **THEN** the checker reports `UnsupportedTensorPrecision`

### Requirement: Blocks and sequencing

Braces SHALL define blocks whose bindings are sequential (newline or `;` separated) with
exactly one tail expression as the block value; all bindings SHALL collapse into a single
Deep `let`. There SHALL be no `let ... in` form and no `where` clauses; a bare non-tail
expression statement SHALL be rejected.

#### Scenario: Sequential bindings collapse to one let

- **WHEN** a block binds `x`, `y`, `z` then a tail expression
- **THEN** it desugars to a single `(let {} (bind ...) tail)` node

#### Scenario: Bare non-tail statement is rejected

- **WHEN** a block contains two top-level expressions with no binding
- **THEN** parsing fails and the user must bind it with `_ = <expr>` or move it to tail position

### Requirement: Effect handler blocks

Phase 2a SHALL provide `with seed(...)` and `with device(...)` handler expressions, each
taking exactly one parenthesized argument and a brace-delimited block body. `with seed`
SHALL require an `i64`-suffixed integer literal, `with device` a string literal, and only
`seed` and `device` SHALL be valid handler names.

#### Scenario: Seed handler with i64 literal

- **WHEN** source is `with seed(42i64) { dropout(x, 0.5) }`
- **THEN** the handler is accepted as an expression

#### Scenario: Unsuffixed seed literal is a type error

- **WHEN** source is `with seed(42) { ... }`
- **THEN** it is a type error naming the required `i64` suffix

### Requirement: Records

Surf SHALL construct records with braces (with field punning), access fields with dot
chaining, and canonicalize record `kv` pairs alphabetized by key in Deep. Functional update
with `with` SHALL be reserved for Phase 1 and not part of the Phase 0 parser.

#### Scenario: Record punning and access

- **WHEN** source is `opt = Adam { lr, eps: 1.0e-8 }` then `opt.lr`
- **THEN** `lr` puns to `lr: lr` and `opt.lr` desugars to `(access {} (var {} opt) lr)`

#### Scenario: Record fields alphabetized in Deep

- **WHEN** `Adam { lr, eps: 1.0e-8 }` is desugared
- **THEN** the `kv` pairs are emitted in alphabetical key order (`eps` before `lr`)

### Requirement: Pattern matching

`match` SHALL use `=>` arms with the supported pattern forms (variable, wildcard, literal,
constructor, nested, record, tuple, as-pattern) and optional `if` guards. Matches SHALL be
exhaustive over the scrutinee ADT, and or-patterns SHALL NOT be supported in v1.

#### Scenario: Guarded arms with distinct patterns

- **WHEN** a match uses `| x if x > 0 => ...` and `| _ => ...`
- **THEN** the guard fills the guard slot in `(arm {} pattern guard body)`

#### Scenario: Non-exhaustive match is a compile error

- **WHEN** a match over an ADT omits a variant with no wildcard
- **THEN** it is a compile error

### Requirement: Tuples and unit

Tuples SHALL be constructed with commas (`(a, b, c)`), where `(a)` is grouping not a
one-tuple, and accessed with dot-integer syntax. `()` SHALL be both the unit value and unit
type.

#### Scenario: Tuple access by index

- **WHEN** source is `pair = (w_new, b_new)` then `pair.0`
- **THEN** `pair.0` desugars to `(tuple-get {} (var {} pair) (lit ... 0))`

#### Scenario: Single-element tuple does not exist

- **WHEN** source is `(a)`
- **THEN** it parses as grouping, not a one-element tuple

### Requirement: Transforms

`grad`, `vmap`, `jit`, `cast`, `realize`, `copy`, and `&` SHALL be recognized as transform
keywords emitting dedicated Deep tags rather than `app` nodes. Transforms SHALL always be
applied; a bare transform reference SHALL be a parse error. The second argument to `cast`
SHALL be a precision type literal.

#### Scenario: Transform composition

- **WHEN** source is `jit(grad(loss_fn))`
- **THEN** it desugars to `(jit {} (grad {} (var {} loss_fn)))`

#### Scenario: Bare transform is a parse error

- **WHEN** source is `g = grad`
- **THEN** parsing fails because transforms must always be applied

### Requirement: Numeric literal defaults

An unsuffixed integer literal SHALL bind at `int32` and an unsuffixed float literal at `f32`,
with no implicit precision promotion. Underscore separators and scientific notation SHALL be
accepted, and `-42` SHALL always parse as unary minus applied to `42`.

#### Scenario: Defaults without promotion

- **WHEN** a bare `42` and a bare `1.0` appear in unannotated positions
- **THEN** `42` binds at `int32` and `1.0` binds at `f32`, never `int64`/`f64`

#### Scenario: Negative literal is unary minus

- **WHEN** source is `f -42`
- **THEN** it parses as `f - 42` (infix), and a negative argument must be written `f(-42)`

### Requirement: Literal suffixes

Numeric literals MAY carry a closed set of precision suffixes (`f32`, `f64`, `bf16`, `f16`,
`i8`, `i16`, `i32`, `i64`) that bind the literal at exactly that precision with no inference.
A suffix SHALL be part of the token only if it immediately follows the digits; integer-typed
suffixes attach to integer literals only.

#### Scenario: Suffix fixes the literal precision

- **WHEN** a literal is written `42i64`
- **THEN** it binds at `int64` with no widening or narrowing

#### Scenario: Float suffix on integer-only literal is rejected

- **WHEN** a literal is written `1.0i8`
- **THEN** it is a parse error because integer suffixes attach to integer literals only

### Requirement: Contextual tensor-literal inference

When a tensor literal appears in a known-element-type position (typed let RHS, matching call
argument, typed tensor return body, or `cast(_, p)`), its unsuffixed numeric literals SHALL
adopt that element type instead of the default. Outside that closed set, literals SHALL fall
back to the `int32`/`f32` defaults.

#### Scenario: Typed context adopts the element type

- **WHEN** source is `let xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
- **THEN** the literals bind at `f64`

#### Scenario: Suffix disagreeing with context is a type error

- **WHEN** an `f32`-context tensor literal is `[1.0, 2.0f64, 3.0]`
- **THEN** it is a type error naming the offending index because the suffix disagrees with the element type

### Requirement: Strings

String literals SHALL be double-quoted with the escapes `\"`, `\\`, `\n`, `\t`, `\r`, `\0`,
and SHALL desugar to `(lit {type: (t-prim {} string)} ...)`. Multiline strings,
interpolation, and Unicode escapes SHALL NOT be supported in v1.

#### Scenario: String literal desugars

- **WHEN** source is `"hello world"`
- **THEN** it desugars to `(lit {type: (t-prim {} string)} "hello world")`

#### Scenario: Interpolation is not supported

- **WHEN** a string attempts interpolation or a multiline body
- **THEN** it is rejected because v1 strings have no interpolation and no multiline form

### Requirement: Type aliases

A `type Name = <type>` without a leading `|` SHALL be a transparent type alias expanded at
desugaring, with an optional parameter list scoping its binders by position (precision slot →
`t-var`, dimension slot → `d-var`, unlisted dimension name → `d-name`). Opaque aliases SHALL
NOT exist in v1.

#### Scenario: Alias binder resolves by position

- **WHEN** a type is `type Matrix[p, rows] = tensor[rows, p]`
- **THEN** `p` lowers to `t-var` (precision slot) and `rows` to `d-var` (dimension slot)

#### Scenario: Unlisted dimension is a concrete name

- **WHEN** a type is `type Weights = tensor[n, f32]`
- **THEN** `n` lowers to `d-name`, not an implicit dimension binder

### Requirement: Opaque types and declared invariants

`@opaque` before an ADT `type` SHALL mark it constructible and inspectable only inside its
defining module; `@opaque` on an alias or at top level SHALL be an error. An `@invariant(binder) expr`
MAY appear between `@opaque` and `type`, recorded as Deep metadata; `@invariant` without
`@opaque` SHALL be an error.

#### Scenario: Opaque ADT records metadata

- **WHEN** an `@opaque type Probability = | Probability { value: f32 }` sits inside a module
- **THEN** it desugars with `opaque: true` metadata inside `(module {} stats.prob ...)`

#### Scenario: Opaque on an alias is a parse error

- **WHEN** `@opaque` precedes a type alias (no `|`)
- **THEN** parsing fails because `@opaque` requires an ADT declaration

#### Scenario: Invariant without opaque is an error

- **WHEN** `@invariant(p) ...` appears without `@opaque`
- **THEN** it is an error because assumption injection is unsound for a forgeable type
