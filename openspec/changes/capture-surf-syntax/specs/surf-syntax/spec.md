## ADDED Requirements

### Requirement: One canonical Surf grammar and explicit migration

Surf v0.19 SHALL define one canonical repository and producer spelling for each
grammatical construct. The normal parser MAY accept the syntax-safe aliases
enumerated by this specification when they preserve the same value and structure.
`chelis fmt` SHALL normalize those aliases to canonical source and SHALL be
idempotent on its output. Semantic, structurally ambiguous, and incompatible
legacy v0.18 aliases SHALL be accepted only by `chelis migrate surf --from 0.18`,
which SHALL emit canonical v0.19 source.
With `--inplace`, migration SHALL preflight the whole batch before replacing
any file, reject symbolic links and multiply linked files, replace each file
atomically, and restore every earlier replacement byte-for-byte if a later
replacement fails.
Migration SHALL NOT guess a replacement for an identifier newly reserved by
v0.19; it SHALL reject the input and direct the author to rename the identifier
manually before rerunning migration.
Canonical property options SHALL appear in `tolerance`, `seed`, `samples`,
then `contract` order, preserving the relative order of repeated contracts;
the parser MAY accept another order and the formatter SHALL normalize it.
Canonical Deep and Surf tooling SHALL satisfy these executable laws:

```text
desugar(resugar(deep)) = normalize_deep(deep)
format(format(surf)) = format(surf)
normalize_deep(desugar(resugar(desugar(surf))))
  = normalize_deep(desugar(surf))
```

The formatting law chooses a syntactic representative. The final law compares
normalized Deep because desugaring erases comments, whitespace, and equivalent
Surf sugar; macro-authored Surf is compared after expansion.

#### Scenario: Canonical source is a formatter fixed point

- **WHEN** canonical source is formatted twice
- **THEN** both outputs are byte-identical and parse through the canonical parser

#### Scenario: Syntax-safe alias normalizes to repository form

- **WHEN** source uses an accepted trailing separator, value-preserving numeric spelling, valid Unicode escape alias, or whitespace before a parenthesized call argument list
- **THEN** normal parsing preserves its meaning and formatting emits the canonical repository spelling

#### Scenario: In-place batch migration rolls back

- **WHEN** persisting a later file fails after an earlier file was replaced
- **THEN** the earlier file is restored byte-for-byte before migration reports failure

#### Scenario: Property delimiters make outer binary grouping redundant

- **WHEN** a property has the binary precondition `where x <= 1:`
- **THEN** the canonical formatter preserves that spelling without adding an outer grouping pair
- **AND** the canonical parser rejects the former alias `where (x <= 1):`
- **AND** the v0.18 migration parser rewrites that alias to the canonical spelling
- **AND** meaningful operand grouping such as `where (x + 1) <= y:` remains accepted

#### Scenario: Mixed property options have one output order

- **WHEN** a property interleaves `contract` options with tolerance, seed, or samples
- **THEN** formatting emits tolerance, seed, samples, and then the contracts in their original relative order

#### Scenario: Legacy alias requires migration

- **WHEN** source uses a colon result annotation or omitted nullary `()`
- **THEN** canonical parsing rejects it and the explicit v0.18 migration command rewrites it

#### Scenario: Newly reserved identifier requires an authored rename

- **WHEN** v0.18 source binds a name such as `resume` or `quote` that v0.19 reserves
- **THEN** migration rejects the input and directs the author to rename it manually rather than guessing a replacement

#### Scenario: Decorative empty forms are aliases

- **WHEN** source writes `Option[]`, `def f[](x)`, a zero-field type variant `type Empty = | Empty {}`, or `import Demo ()`
- **THEN** canonical parsing rejects it in favor of the delimiter-free form

#### Scenario: Empty effect rows carry a semantic constraint

- **WHEN** a declaration explicitly writes `! {}`
- **THEN** canonical parsing preserves the declared-pure upper bound rather than treating it as an alias for inferred effects

### Requirement: Reserved keywords

Surf SHALL reserve the 29 active lexical keywords listed by the numbered source spec so they
cannot be used as identifiers. Grammar-specific words such as `where` SHALL remain contextual.
The future words `effect`, `handler`, `perform`, `resume`, and `borrow` SHALL remain lexically
reserved without active productions. `do`, `quote`, `unquote`, and `splice` SHALL be active
keywords.

#### Scenario: Reserved word is not an identifier

- **WHEN** source attempts `def def(x) = x`
- **THEN** parsing fails because `def` is a reserved keyword

#### Scenario: Future word remains available

- **WHEN** source uses `perform` as an identifier
- **THEN** the parser accepts it as a name because no `perform` grammar is active

### Requirement: Operator precedence and non-associativity

Surf operators SHALL bind according to the fixed precedence table (pipe lowest through field
access highest), and the non-associative comparison/equality operators (BP 4 and 5) SHALL
reject chaining. There SHALL be no operator overloading, no infix bitwise operators, and no
exponentiation operator.

#### Scenario: Flat application binds tighter than addition

- **WHEN** source is `f(x) + y`
- **THEN** it parses as `f(x) + y`, while the juxtaposition alias `f x + y` is rejected

#### Scenario: Applying a returned function is explicitly grouped

- **WHEN** source is `(f(x))(y)`
- **THEN** it desugars to `(app {} (app {} f x) y)`, while ungrouped `f(x)(y)` is rejected

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

A named package Surf file SHALL declare one `module Name` as its first non-comment line, with
PascalCase components and no nesting within a file. Script/snippet source MAY omit the module
line. `module Foo.Bar` SHALL live at `foo/bar.ch` and desugar to `(module {} foo.bar ...)`.

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

A rank variable `..r` SHALL stand for a name-preserving run of dimensions. Its plain name
`r` SHALL appear in the function's complete `[...]` binder clause; `[..r]` is not a binder
spelling. A spread name SHALL NOT repeat within one tensor shape, and a `def` mentioning
`..r` SHALL be restricted to name-trackable operations
(elementwise and named-axis reductions), never positional shape-rewriters.

#### Scenario: Rank-generic identity function

- **WHEN** a function is `def relu_forward[r](x: &tensor[..r, f32]) -> tensor[..r, f32] = relu(x)`
- **THEN** it type-checks at every rank because `relu` is shape-identity

#### Scenario: Repeated spread name is a parse error

- **WHEN** a tensor shape uses the same spread name twice, e.g. `tensor[..r, ..r, f32]`
- **THEN** parsing fails

### Requirement: Type signatures and arrow associativity

Surf SHALL support inline (`def`) and standalone (`sig`) signatures; a `sig` SHALL precede
its function `def` or bare top-level value binding. Only a function `def` SHALL carry an
inline binder list. A binder-bearing signature for a non-function value SHALL therefore
remain a standalone `sig` followed by an untyped value binding; a typed value binding has no
quantifier scope. The `->` arrow SHALL be right-associative and flat in Deep (`t-fn` with
last child the return type), and a function-typed argument SHALL require parentheses,
preserved by the formatter and decompiler.

#### Scenario: Inline annotations synthesize a defsig

- **WHEN** a function is `def add_vecs[d](x: tensor[d, f32], y: tensor[d, f32]) -> tensor[d, f32] = add(x, y)`
- **THEN** the desugarer emits a `defsig` in addition to the `def`

#### Scenario: Polymorphic value signature remains standalone

- **WHEN** Deep contains `(defsig {} empty (p) (t-adt {} List (t-var {} p)))` followed by `(def {} empty (var {} Nil))`
- **THEN** canonical Surf emits `sig empty[p]: List[p]` followed by `empty = Nil`
- **AND** desugaring that Surf recovers the binder-bearing `defsig` rather than freeing `p`

#### Scenario: Parenthesized function argument is distinct from curried arrows

- **WHEN** a type is written `(a -> b) -> c`
- **THEN** it is a 1-ary type whose single argument is a function, distinct from `a -> b -> c`, and the grouping parentheses are preserved

### Requirement: Effect annotations

A `sig` or `def` SHALL accept an optional `! { ... }` effect suffix drawing from the built-in
names `Diff`, `Random`, `Accum`, `IO`, and `Resource("device")`. `Random`, `IO`, and
`Resource(...)` SHALL be the active boundary effects in the shipped subset, while `Diff` and
`Accum` SHALL be accepted as forward-compatible syntax.

#### Scenario: Random effect annotation is accepted

- **WHEN** a sig is `sig predict[n]: tensor[n, f32] -> tensor[n, f32] ! { Random }`
- **THEN** the parser accepts the effect suffix

#### Scenario: Unknown effect name is rejected

- **WHEN** an effect suffix names an effect outside the built-in set, e.g. `! { Bogus }`
- **THEN** the parser/checker rejects it

### Requirement: Explicit signature binders

Every type, dimension, and rank variable in a `sig` SHALL appear exactly once in that
signature's complete `[..]` binder list. A non-primitive name in a tensor precision slot
SHALL become `t-var` only when listed; an unlisted spelling remains a primitive request and
SHALL be rejected by the closed primitive resolver. The `[..]` clause SHALL override the
case-split so a listed PascalCase name is a type variable. A matching `def` SHALL NOT carry
a second binder list. A declaration binder SHALL remain in scope for ordinary type
positions in body-local lambda parameters, expression ascriptions, block bindings, and
nested ADT arguments. A tensor precision slot in those body-local annotations SHALL remain
a closed primitive request.

#### Scenario: Sig precision name is an explicit type variable

- **WHEN** a sig is `sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]`
- **THEN** `d` and `p` are explicitly quantified variables instantiated fresh per call site

#### Scenario: Unbound def precision name is rejected

- **WHEN** a def is `def f[a, b](x: tensor[3, p]) = ...` with `p` absent from `[a, b]`
- **THEN** the shared primitive resolver rejects `p` as unknown and suggests the nearest active dtype

#### Scenario: Body annotation keeps ordinary binder scope

- **WHEN** `def identity[p](x: p) -> p = (x: p)` uses `p` in an expression ascription
- **THEN** that `p` desugars to the declaration's `t-var`

#### Scenario: Body tensor precision remains closed

- **WHEN** `def identity[n, p](x: tensor[n, p]) = (x: tensor[n, p])` uses `p` in a body-local tensor precision slot
- **THEN** that body-local `p` remains a primitive request and is rejected by the closed primitive resolver

### Requirement: Blocks and sequencing

Braces SHALL define binding blocks whose bindings are newline-separated, with at least one
binding and exactly one tail expression; all bindings SHALL collapse into a single Deep
`let`. Semicolons and one-expression binding blocks SHALL be rejected. Direct sequential
Deep `block` nodes SHALL use `do { e1; e2 }`; parallel tasks SHALL use `par { e1; e2 }`.
Comma- and semicolon-delimited forms SHALL accept one separator before the closing
delimiter. Canonical output SHALL omit that optional trailing separator except for the
grammar-significant comma in a singleton tuple or singleton tuple pattern. Repeated or
otherwise misplaced separators SHALL be rejected.

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
chaining, and preserve record `kv` pairs in written left-to-right order in Deep. A
same-named field/value SHALL use pun syntax. Functional update SHALL use
`base with { field: value }` and preserve the same order and pun rules.
A zero-field type variant SHALL be bare, while zero-field record expressions and record
patterns SHALL retain `{}` to distinguish Deep `record`/`pat-record` from constructor
values/patterns. A record update SHALL contain at least one field.

#### Scenario: Record punning and access

- **WHEN** source is `opt = Adam { lr, eps: 1e-8 }` then `opt.lr`
- **THEN** `lr` puns to `lr: lr` and `opt.lr` desugars to `(access {} (var {} opt) lr)`

#### Scenario: Record fields preserve written order in Deep

- **WHEN** `Adam { lr, eps: 1e-8 }` is desugared
- **THEN** the `kv` pairs remain `lr` then `eps`, matching evaluation order

### Requirement: Pattern matching

`match` SHALL use `=>` arms with the supported pattern forms (variable, wildcard, literal,
constructor, nested, record, tuple, as-pattern) and optional `if` guards. Matches SHALL be
exhaustive over the scrutinee ADT, and or-patterns SHALL NOT be supported in v1.
Numeric literal patterns SHALL be unsuffixed; a leading minus SHALL decode directly into the
raw negative `pat-lit` value, including negative zero and the full `i64` minimum.

#### Scenario: Guarded arms with distinct patterns

- **WHEN** a match uses `| x if x > 0 => ...` and `| _ => ...`
- **THEN** the guard fills the guard slot in `(arm {} pattern guard body)`

#### Scenario: Non-exhaustive match is a compile error

- **WHEN** a match over an ADT omits a variant with no wildcard
- **THEN** it is a compile error

### Requirement: Tuples and unit

Tuples SHALL be constructed with commas (`(a, b, c)`), where `(a)` is grouping and `(a,)`
is a one-tuple, and accessed with dot-integer syntax. `()` SHALL be the unit value and
empty tuple pattern; `unit` SHALL be the unit type. Bare constructor values such as `None`, zero-argument
applications such as `None()`, and zero-field records such as `None {}` SHALL remain distinct;
an ordinary nullary function call remains `f()`.

#### Scenario: Tuple access by index

- **WHEN** source is `pair = (w_new, b_new)` then `pair.0`
- **THEN** `pair.0` desugars to `(tuple-get {} (var {} pair) (lit ... 0))`

#### Scenario: Single-element tuple requires a trailing comma

- **WHEN** source is `(a,)`
- **THEN** it parses as a one-element tuple, while `(a)` remains grouping

### Requirement: Transforms

`grad`, `vmap`, `jit`, `cast`, `realize`, `copy`, and `&` SHALL emit dedicated Deep tags.
`grad`, `vmap`, `jit`, and `cast` SHALL require call-like syntax; `realize` and `copy` MAY
also appear as their canonical bare unary callable pipe stages. `vmap(f)` SHALL be the sole
zero-axis spelling, and nonzero axes SHALL use `vmap(f, axis=n)`. The second argument to
`cast` SHALL be a precision type literal.

#### Scenario: Transform composition

- **WHEN** source is `jit(grad(loss_fn))`
- **THEN** it desugars to `(jit {} (grad {} (var {} loss_fn)))`

#### Scenario: Bare transform is a parse error

- **WHEN** source is `g = grad`
- **THEN** parsing fails because transforms must always be applied

### Requirement: Numeric literal defaults

An unsuffixed integer literal SHALL bind at `i32` and an unsuffixed float literal at `f32`,
with no implicit precision promotion. Canonical source SHALL equal the literal printer's
decimal spelling without separators. The normal parser SHALL also accept value-preserving
hexadecimal and binary integers, well-placed digit separators, and decimal exponent
spellings. Malformed spellings, overflow, padding that changes the accepted decimal grammar,
and non-finite values SHALL be rejected. `-42` SHALL parse as unary minus applied to `42`.

#### Scenario: Defaults without promotion

- **WHEN** a bare `42` and a bare `1.0` appear in unannotated positions
- **THEN** `42` binds at `i32` and `1.0` binds at `f32`, never `i64`/`f64`

#### Scenario: Negative literal is unary minus

- **WHEN** source passes a negative argument as `f(-42)`
- **THEN** the argument parses as unary minus applied to `42`; juxtaposition `f -42` is not a call

### Requirement: Canonical integer dtype spelling

Surf SHALL use `i8`, `i16`, `i32`, and `i64` as the only signed-integer dtype
spellings in type positions and literal suffixes. Normal ingress SHALL reject
the retired v0.18 spellings `int8`, `int16`, `int32`, and `int64`; the explicit
v0.18 migration SHALL rewrite those spellings only where they denote dtypes or
cast targets and SHALL preserve unrelated identifiers and string contents.
Retired spellings SHALL NOT be rebound as explicit type variables.

#### Scenario: Canonical source uses i64

- **WHEN** a function parameter or literal has 64-bit signed-integer precision
- **THEN** canonical Surf spells it `i64` or uses the `i64` literal suffix

#### Scenario: Retired spelling is migration-only

- **WHEN** normal Surf ingress encounters `int64` in a type position
- **THEN** it rejects the spelling and points to the explicit v0.18 migration

### Requirement: Literal suffixes

Numeric literals MAY carry a closed set of precision suffixes (`f32`, `f64`, `bf16`, `f16`,
`i8`, `i16`, `i32`, `i64`) that bind the literal at exactly that precision with no inference.
Default-type suffixes SHALL remain accepted because contextual literal adoption can make
them semantically distinct from an unsuffixed literal.
A suffix SHALL be part of the token only if it immediately follows the digits; integer-typed
suffixes attach to integer literals only. Suffixes SHALL be rejected in pattern position,
where Deep preserves only the raw literal value.

#### Scenario: Suffix fixes the literal precision

- **WHEN** a literal is written `42i64`
- **THEN** it binds at `i64` with no widening or narrowing

#### Scenario: Float suffix on integer-only literal is rejected

- **WHEN** a literal is written `1.0i8`
- **THEN** it is a parse error because integer suffixes attach to integer literals only

### Requirement: Contextual tensor-literal inference

When a tensor literal appears in a known-element-type position (typed let RHS, matching call
argument, typed tensor return body, or `cast(_, p)`), its unsuffixed numeric literals SHALL
adopt that element type instead of the default. Outside that closed set, literals SHALL fall
back to the `i32`/`f32` defaults.

#### Scenario: Typed context adopts the element type

- **WHEN** source is `xs: tensor[3, f64] = [1.0, 2.0, 3.0]`
- **THEN** the literals bind at `f64`

#### Scenario: Suffix disagreeing with context is a type error

- **WHEN** an `f32`-context tensor literal is `[1.0, 2.0f64, 3.0]`
- **THEN** it is a type error naming the offending index because the suffix disagrees with the element type

### Requirement: Strings

String literals SHALL be double-quoted with the canonical named escapes `\"`,
`\\`, `\n`, `\t`, `\r`, and `\0`, and SHALL desugar to
`(lit {type: (t-prim {} string)} ...)`. A control character without a named
escape SHALL canonically use `\u{h}` with its minimal lowercase hexadecimal scalar value.
The normal parser SHALL accept valid `\u{...}` scalar escapes with uppercase or padded
hexadecimal digits, including value-preserving aliases for printable characters and named
escapes. Raw controls, malformed or out-of-range Unicode escapes, multiline strings, and
interpolation SHALL be rejected.

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
