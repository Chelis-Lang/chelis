# Surf syntax canonicalization audit

**Snapshot:** 2026-08-01, repository head `88fe034e1c73`

**Tracking:** [chelis#1024](https://github.com/Chelis-Lang/chelis/issues/1024)

This document inventories Surf spellings that are accepted for the same parser
node, the same canonical Deep form, or the same intended operation. It records
what `chelis fmt`, `chelis surf`, the linter, and the active specifications do
today, and where Chelis has or has not selected a preferred spelling.

This is an investigation, not a new language contract. Language decisions must
land in [`spec/02-surf-syntax.md`](../../spec/02-surf-syntax.md); style decisions
and their enforcement belong in
[`spec/01-nomenclature.md`](../../spec/01-nomenclature.md). The formatter should
then be the executable layout oracle; the parser plus blocking checked-source
gate should be the acceptance oracle for canonical Surf.

## Scope and terminology

The inventory includes deliberately supported aliases and small surface sugars.
It does not attempt to identify every pair of programs that happens to compute
the same value. For example, `if true then x else y` and `x` are outside scope.

The distinctions matter:

- **Same AST** means the parser has already erased the spelling difference.
- **Same Deep** means the Surf AST differs, but desugaring erases the difference.
- **Contextual equivalence** means the forms agree only after a type-, effect-,
  or linearity-aware rule. A token-only formatter must not rewrite these.
- **Lookalike only** means the forms are not equivalent and canonicalization
  must preserve the distinction.

The status labels used below are:

- **Declared**: an active numbered spec names a preferred form.
- **Formatter-only**: `chelis fmt` chooses a form, but the active spec does not
  state that it is canonical.
- **Advisory**: a non-blocking lint or prose guideline expresses a preference.
- **Undecided**: both forms are accepted and no repository-wide choice exists.
- **Distinct**: similar-looking forms have different language meaning.

## Executive result

Chelis does not yet have one complete canonical Surf contract.

The strongest existing decision is function return syntax: `->` is declared
canonical, `chelis fmt` rewrites `:`, and the blocking
`surf-def-arrow-form` lint rejects the colon form. Many other aliases happen to
be normalized by the formatter, but that choice is not stated in the numbered
spec. Several semantically equivalent forms, notably curried versus multi-arg
application, nested calls versus pipes, `unit` versus `()`, and one-expression
blocks, remain distinct after formatting.

There are therefore three separate jobs:

1. document one accepted spelling for each Surf construct in the active specs;
2. reject non-canonical aliases in the main Surf parser or, for rules that need
   semantic context, in the blocking checked-source gate;
3. keep `chelis fmt` as an idempotent layout formatter for canonical Surf, not a
   translator between accepted dialects; and
4. require every Surf emitter, especially `chelis surf`, to resugar Deep into
   that one parseable form without changing its Deep meaning.

The target is deliberately stronger than “liberal input, canonical output.” If
`def f(x: T) -> U = e` is the language form, `def f(x: T): U = e` should be a
syntax error rather than valid input that changes spelling when formatted.
Backward compatibility, if required for a migration, should live in an explicit
legacy parser or migration command and must not remain part of canonical Surf.

## Inventory: declarations, types, and layout

| Construct | Accepted equivalent forms | Equivalence | Current output or enforcement | Current status |
|---|---|---|---|---|
| Function return annotation | `def f(x: T): U = e`; `def f(x: T) -> U = e` | Same AST | Formatter emits `->`; blocking `surf-def-arrow-form` lint rejects `:` | **Declared:** `->` is canonical ([spec/01 §3.5](../../spec/01-nomenclature.md#35-function-definition-return-type-syntax), [spec/02 P4a](../../spec/02-surf-syntax.md#p4a-preferred-surf-style)) |
| Nullary function parameters | `def f = e`; `def f() = e` | Same AST | Formatter emits `def f()` | **Formatter-only** |
| Nullary return combinations | `def f: T`; `def f -> T`; `def f(): T`; `def f() -> T` | Same AST | Formatter emits `def f() -> T`; colon lint also applies | `->` is **Declared**; `()` is **Formatter-only** |
| Inline versus standalone signature | `def f(x: A) -> B = e`; adjacent `sig f: A -> B` plus `def f(x) = e` | Same `defsig` and typed function contract when the declarations match | Formatter preserves the authored form; idiomatic decompiler combines a signature and definition, verbose decompiler separates them | **Declared choice by size:** inline is shown as normal and standalone is “for long signatures” ([spec/02 P4](../../spec/02-surf-syntax.md#p4-type-signatures)); no mechanical threshold exists |
| Function-type grouping | `A -> (B -> C)`; `A -> B -> C` | Same type by right associativity | Formatter emits `A -> B -> C` | **Declared** by the associativity rule |
| Empty type application | `Maybe`; `Maybe[]` | Same Deep `t-adt` when `Maybe` is an ADT name | Formatter emits `Maybe` | **Formatter-only** |
| Unit type | `unit`; `()` | Same Deep `t-unit` | Formatter preserves whichever AST was parsed | **Undecided.** The complete desugaring reference presents `()`, while code and examples also accept `unit` |
| Effect name casing | `Diff`/`diff`, `Random`/`random`, `Accum`/`accum`, `IO`/`io`, `Test`/`test` | Same AST | Formatter emits `Diff`, `Random`, `Accum`, `IO`, `Test` | **Formatter-only**, although the canonical-cased forms are used throughout the specs |
| Zero-field ADT variant declaration | `\| None`; `\| None()`; `\| None {}` | Same Deep zero-field `variant` (`None {}` first has a record-shaped AST) | Formatter emits `\| None` | **Formatter-only** |
| Dimension declaration grouping | `dim batch, seq`; `dim batch` plus `dim seq` | Same sequence of Deep `defdim` nodes | Formatter preserves grouping | **Undecided** |
| Ordinary block separators | `{ x = 1; y = 2; x + y; }`; newline-separated form | Same AST; final separator is discarded | Formatter emits one binding/tail per line with no semicolons | **Formatter-only.** Both separators are explicitly accepted by [spec/02 P5](../../spec/02-surf-syntax.md#p5-blocks-and-sequencing) |
| `par` separators | `par { a; b }` only; newlines may surround but do not replace `;` between tasks | Not an alias | Formatter emits `par { a; b }` | **Distinct:** do not generalize the ordinary-block rule to `par` |
| Trailing commas | Accepted in argument, parameter, type-argument, tuple, list, record, import/export, and similar delimited lists | Same AST except that `(x,)` is a singleton tuple, not grouping | Formatter removes trailing commas | **Formatter-only**; acceptance is declared by [spec/02 P12](../../spec/02-surf-syntax.md#p12-whitespace-and-line-continuation) |
| ADT and multiline layout | Variants and arrow chains can be authored on one or many lines | Same AST | Formatter emits one ADT variant per line and applies its own line breaking | **Formatter-only** |
| Line continuation in a pipe | `x \|> f \|> g` on one line or with later lines beginning `\|>` | Same AST | Formatter chooses flat form for at most three short parts, otherwise one stage per line | **Formatter-only** layout; continuation semantics are declared in spec/02 P12 |

## Inventory: expressions and patterns

| Construct | Accepted equivalent forms | Equivalence | Current output or enforcement | Current status |
|---|---|---|---|---|
| Parenthesized versus juxtaposition call | `f(x)`; `f x` | Same application shape | Formatter emits `f(x)` | **Formatter-only.** The book calls parentheses the standard form, but the numbered spec only declares both accepted |
| Multi-arg versus curried-looking call | `f(x, y)`; `f x y`; `f(x)(y)` | Same flat Deep `app`; the latter two have nested Surf `Apply` nodes before desugaring | Formatter emits `f(x, y)` for the first but `f(x)(y)` for the other two | **Undecided and not canonical today** |
| Constructor application | `Some(x)`; `Some x` | Same application | Formatter emits `Some(x)` | **Formatter-only** |
| Constructor pattern | `Some(x)`; `Some x` | Same pattern AST | Formatter emits `Some(x)` | **Formatter-only** |
| Zero-argument constructor pattern | `None`; `None()` | Same pattern AST and Deep `pat-ctor` | Formatter emits `None` | **Formatter-only** |
| Record expression pun | `Point { x, y }`; `Point { x: x, y: y }` | Same AST because the parser expands the pun | Formatter emits the explicit form | **Formatter-only** |
| Record pattern pun | `Point { x }`; `Point { x: x }` | Same AST because the parser expands the pun | Formatter emits the explicit form | **Formatter-only** |
| Record field order | `Point { x: a, y: b }`; `Point { y: b, x: a }` | Desugaring sorts fields and reaches the same Deep record; patterns are sorted too | Formatter preserves source order | **Undecided.** Any formatter sorting rule must first pin evaluation-order expectations for field values |
| Redundant grouping | `(x)`; `x` | Same AST | Formatter emits `x` | **Formatter-only** |
| One-expression block | `{ x }`; `x` | Same Deep expression; distinct Surf AST | Formatter preserves `{ x }` | **Undecided** |
| `vmap` axis spelling | `vmap(f, 1)`; `vmap(f, axis=1)` | Same AST | Formatter emits `vmap(f, axis=1)` | **Formatter-only.** The numbered spec still illustrates the positional form |
| Singleton `grad` target | `grad(f, wrt=x)`; `grad(f, wrt=(x))` | Same AST | Formatter emits `grad(f, wrt=x)` | **Formatter-only** |
| Numeric spelling | Decimal, `0x`/`0X` hex, `0b`/`0B` binary, underscores, scientific notation, and redundant decimal zeros can encode the same value | Same literal AST value and suffix | Formatter emits decimal integers and the shortest float that retains float syntax; suffixes are retained | **Formatter-only.** Hex is specified; binary is implemented but its long-term status is explicitly unresolved in spec/02 P10 |
| String escape spelling | Escaped and directly representable characters can decode to the same string value | Same literal value | Formatter re-escapes with its debug-string representation | **Formatter-only** |
| Comments | `--` line comments and nested `{- ... -}` block comments are both non-semantic | Not treated as interchangeable by tooling | Formatter deliberately preserves comment text and kind | **Intentionally not canonicalized** |

### Sugar that is equivalent only below the parser

These forms deserve an explicit style decision, but they should not be conflated
with parser aliases.

| Construct | Forms | Relationship | Current preference |
|---|---|---|---|
| Infix/prefix operator versus builtin call | `x + y` / `add(x, y)`, `-x` / `neg(x)`, `!x` / `not(x)`, and the corresponding arithmetic, comparison, and logical operations | Desugar to builtin applications; `x > y` reverses operands when becoming `cmplt(y, x)` | Formatter preserves both; **Undecided** |
| List literal versus constructors | `[x, y]`; `Cons(x, Cons(y, Nil))` | Same canonical list construction after desugaring | Formatter preserves both; **Undecided**, though bracket syntax is the user-facing form in the spec |
| Nested first-argument calls versus pipe | `relu(neg(x))`; `x \|> neg \|> relu`. Also `add(neg(x), bias)`; `x \|> neg \|> add(bias)` | Same intended operation under the first-argument insertion contract, but different Surf and Deep nodes | `prefer-pipe-operator` is **Advisory** and only proposes a rewrite when the typed-pipeline safety proof and formatter layout constraints pass. Formatter itself preserves both |
| Explicit pipe lambda versus call-stage sugar | `x \|> fn (v) -> f(v, y)`; `x \|> f(y)` | Equivalent when the carried value is the first argument | Spec/01 §3.6 prefers call-stage sugar in that case; formatter does not generally recognize and compact the lambda | **Declared preference, incompletely enforced** |
| Unary keyword pipe lambda | `x \|> fn (v) -> realize(v)` / `x \|> realize`; the same pair for `copy` | Same synthesized stage | Formatter emits bare `realize` or `copy` | **Declared/formatter canonical** |
| Cast pipe stage | `x \|> cast(f32)`; `x \|> fn (v) -> cast(v, f32)` | Parser synthesizes the latter AST for the former | Formatter currently emits `x \|> fn (__chelis_pipe) -> cast(__chelis_pipe, f32)`, exposing an internal binder | **No consistent canonical form.** This should compact back to `x \|> cast(f32)` if call-stage sugar is the policy |
| Explicit borrow versus auto-borrow | `f(x)`; `f(&x)` at a read-only use | Contextually equivalent only where the checker inserts the same borrow | New user code generally omits `&`; explicit `&` is retained for exported APIs or dense signatures. This is a **contextual guideline**, not a safe global formatter rewrite |
| Explicit versus implicit linearity operation | `copy(x)`/`drop(x)` versus checker-inserted copy/drop | Contextually equivalent only where ownership analysis inserts the same node | `redundant-linearity-call` is **Advisory**; explicit calls remain valid for real ownership forks and compatibility evidence |

## Similar forms that must remain distinct

Canonicalization must not erase these boundaries:

| Forms | Why they differ |
|---|---|
| `def x -> T = e` versus `x: T = e` | The first is a zero-argument function (`FunDef`); the second is a value binding (`LetDef`). |
| `def f = e` versus `f = e` | The first desugars with an inserted zero-argument `fn` wrapper; the second binds `e` directly, without inserting a function. This is a structural distinction and does not by itself specify eager evaluation. |
| `f` versus `f()` | A function value/reference is not a zero-argument call. |
| `(x)` versus `(x,)` | Grouping is erased; the comma constructs a singleton tuple. |
| `A -> (B -> C)` versus `(A -> B) -> C` | The former is right-associated; the latter accepts a function as its first argument. |
| Omitted effect row versus `! {}` | The AST retains `None` versus an explicitly empty set; the formatter preserves the distinction. |
| Type annotation `e: T` versus `cast(e, T)` | An annotation constrains/checks the expression; a cast performs a precision conversion. |
| Ordinary block separators versus `par` separators | Newlines can separate ordinary block bindings; `par` tasks require `;`. |
| `\|>`, `>`, and `->` | These are respectively pipeline, greater-than comparison, and function/type arrow tokens. They are not alternate spellings. |
| Bare constructor, `Ctor()`, and `Ctor {}` in expressions | Unlike a zero-field **variant declaration**, these produce reference, zero-argument application, and record nodes. They must not be collapsed without a separate semantic rule. |
| `Ctor`/`Ctor()` versus `Ctor {}` in patterns | The first pair is equivalent and formats to `Ctor`; the record-pattern form has a different Deep tag. |

## Emitter findings

The CLI already has the right high-level architecture for normal mode:
`chelis surf` decompiles Deep, reparses the emitted Surf, and runs the formatter.
That should guarantee parseability and the formatter fixed point. Two current
cases violate the stronger requirement that the result preserve the input Deep
meaning:

1. A Deep cast such as
   `(cast {} (lit {} 1.0) (t-prim {} f32))` is first emitted as
   `(1.0 as f32)`. Surf has no cast-operator form; the parser reads this as
   juxtaposition. Normal `chelis surf` therefore succeeds but prints
   `1.0(as)(f32)`, which is an application chain rather than a cast.
2. A Deep parallel expression is emitted as `par(a, b)`, while the Surf parser
   requires `par { a; b }`. Normal `chelis surf` consequently fails with
   `expected LBrace, found LParen`.

`chelis surf --verbose` bypasses the reparse/format step and exposes both
non-Surf spellings directly. These are emitter correctness bugs, not unresolved
style choices. A canonicalization project should add cast and `par` to the
decompiler round-trip corpus before relying on `chelis surf` as a canonical
emitter.

The idiomatic decompiler also reconstructs pipelines from some Deep let/call
shapes. That is consistent with the advisory pipe preference, but it means the
decompiler is already making style decisions that the formatter does not make.
Those decisions should be governed by the same single-grammar table.

## Specification and implementation drift found during the audit

The prose decisions and executable grammar are not fully synchronized:

- The formal `ReturnType` production in spec/02 still lists only `: TypeExpr`,
  although P4a and the parser accept `:` and `->` and declare `->` preferred.
- The formal pattern grammar uses `as`, while the parser, formatter, AST comment,
  and desugaring examples use `x @ Pattern`.
- The formal ADT grammar makes the first `|` appear optional, but the parser uses
  that token to distinguish an ADT from a type alias.
- P10 mentions “octal and binary,” but the lexer implements binary and hex, not
  octal; the nearby examples discuss `0b` only.
- Spec/01's enforcement prose says `prefer-pipe-operator` has no autofix pending
  a semantic proof. The current lint has an autofix guarded by the typed-pipeline
  proof described in
  [`pipe_autofix_and_bare_keyword_extras_diagnosis.md`](pipe_autofix_and_bare_keyword_extras_diagnosis.md).
- Spec/02's transform table illustrates `vmap(f, n)`, while the formatter and
  decompiler emit `vmap(f, axis=n)`.

These should be corrected in the controlling numbered specs rather than
explained by a third design document.

## Canonical Surf pipeline

The conventional term for the reverse transformation is **resugaring**. The
intended compiler boundary should be:

```text
canonical Surf source --parse--> Surf AST --desugar--> canonical Deep
canonical Deep --resugar--> Surf AST --print--> canonical Surf source
```

`chelis surf` is therefore a Deep-to-Surf resugarer (also reasonably described
as a decompiler or pretty-printer), but it should have exactly one output
grammar and be total over every well-formed canonical Deep program. Surf and
Deep are representations of the same language: every valid Deep node must have
a canonical Surf representation. A valid Deep tag with no resugaring case is a
language/spec or implementation bug, not an accepted “unrepresentable subset.”

Resugaring should construct Surf AST nodes and use the same canonical Surf
printer as every other producer. It should not maintain a second set of
handwritten source templates, emit strings such as `(x as f32)` or `par(a, b)`,
and rely on reparsing or formatting to repair them. Malformed Deep may be
rejected, but valid Deep must never fall back to `()`, a placeholder, invalid
Surf, or a meaning-changing approximation.

For every canonical Surf program and every valid canonical Deep program, the
executable laws should be:

```text
desugar(resugar(deep)) = normalize_deep(deep)
resugar(desugar(surf)) = surf
fmt(surf) = surf                 # modulo canonical layout
```

The second equality assumes `surf` already follows the canonical grammar. It is
necessarily modulo information that Deep intentionally does not carry, notably
comments, original whitespace, and source spans. If comments must survive a
Surf-to-Deep-to-Surf tool round trip, they need a separate source-preservation
channel rather than alternate Surf syntax.

`normalize_deep` must be narrowly specified. Metadata that changes typing,
effects, evaluation, validation, or lowering is language meaning: it must either
have canonical Surf syntax or be reconstructed deterministically when Surf is
desugared again. Only explicitly non-semantic or derived metadata, such as
source locations or reproducible checker annotations, may be erased or
canonicalized by the round trip. Calling semantic metadata “Deep-only” would
break the same-language invariant.

## Proposed single-grammar decisions

The following is a proposal for the normative follow-up, not a claim that the
repository has already decided it:

| Area | Proposed canonical Surf | Reason |
|---|---|---|
| Function definitions | Accept `def f(params) = e` or `def f(params) -> T = e`; require `()` for nullary functions and `->` whenever a return type is present | Makes the function/value boundary grammatical; reject colon returns and omitted nullary parentheses |
| Calls and constructors | Accept only parenthesized, flat application: `f(x, y)` and `Some(x)` | It is the book's standard form and matches flat canonical Deep `app`; reject juxtaposition and chained application aliases |
| Blocks | Accept newline-separated bindings/tail, with no semicolons or trailing separator | Gives ordinary sequencing one grammar rather than treating newline and `;` as interchangeable tokens |
| Parallel blocks | `par { a; b }` | Semicolon has semantic separator duty here; do not reuse ordinary-block layout |
| Unit | Accept only `()` | It is the spelling used by spec/02's complete value/type desugaring tables and avoids treating a builtin type word as an ordinary name |
| Effect names | Accept only `Diff`, `Random`, `Accum`, `IO`, `Test` | Matches the active specifications and current formatter; reject lowercase aliases |
| Records | Require puns (`Point { x, y }`) when field and variable names match; otherwise require `field: value` | Gives each record field one checked spelling and requires changing the formatter's current expansion behavior |
| Transform arguments | Accept named non-primary parameters, including `vmap(f, axis=n)` and `grad(f, wrt=x)` | Self-describing and already the formatter/decompiler choice; reject positional and singleton-tuple aliases |
| First-argument chains | Pipes when the typed semantic gate proves a linear first-argument chain; calls otherwise | Matches spec/01 §3.6 without changing later-argument or ownership semantics |
| Pipe stages | Call-stage sugar, including `cast(f32)`, whenever it exactly represents first-argument insertion; lambda otherwise | Prevents generated internal binders and follows the existing stage rule |
| Operators | Infix/prefix notation for the fixed operator vocabulary; named builtins when used as values or pipe stages | Keeps ordinary arithmetic readable without removing first-class builtin names |
| One-expression blocks | Reject braces around a lone expression unless a future scoped construct gives them meaning | They currently desugar away and have no declared purpose |
| Numeric literals | Decide which alternate radices and separators are intentional language constructs; for values emitted from Deep, use decimal integer and shortest valid decimal float with suffix retained | Avoids silently treating every accepted lexical representation as either an alias or an error |
| Comments | Preserve spelling and kind | Comments are authored prose, not executable syntax aliases |

Record field sorting should remain undecided until evaluation order is stated
explicitly. Auto-borrow and implicit copy/drop should remain typed lint/checker
decisions rather than token-only formatter rewrites.

## Decisions approved for the v0.19 implementation

The design discussion following this audit selected the following behavior for
the implementation tracked by chelis#1024. These decisions remain evidence
until the same rules land in the controlling numbered specifications.

- The canonical parser accepts one spelling rather than accepting aliases for
  `chelis fmt` to rewrite. Legacy v0.18 syntax is reachable only through an
  explicit `chelis migrate surf --from 0.18` path.
- Function definitions require a parameter list, including `()` for nullary
  functions, and use `->` for return types. A matching `defsig` plus `def`
  resugars as one inline typed definition; a standalone signature remains
  `sig`.
- Calls are flat and parenthesized. Ordinary blocks use newline separation and
  no semicolons; `par` remains semicolon-delimited. A one-expression ordinary
  block is rejected. Singleton tuples use `(x,)`, while unit uses `()`.
- Effects use the formatter's canonical casing. Records use puns where the
  field and variable match. Record fields preserve authored order, and that
  order becomes the specified left-to-right evaluation order.
- Transform options are named. Axis zero resugars as `vmap(f)` and other axes
  as `vmap(f, axis=n)`. Exact first-argument pipe stages use call-stage sugar;
  a lambda remains when the carried value occupies another position.
- Deep `app` and `pipe` remain distinct canonical constructs. Resugaring does
  not optimize nested calls into pipelines. At resolved ordinary builtin call
  sites the fixed operator vocabulary resugars to infix or prefix notation;
  builtin names remain values and pipe stages.
- Finite `Cons`/`Nil` chains resugar as bracket lists; open-tail `Cons` remains
  explicit. Explicit borrow and explicit copy remain distinct from the
  checker-inserted forms.
- Deep expression forms missing from Surf receive direct syntax: `block`
  becomes `do { e1; e2 }`, record update becomes
  `base with { field: value }`, and quote/unquote/splice use call-like forms.
- One canonical decimal literal spelling is emitted. Comments retain their
  authored text and kind. Macro source is compared after expansion because the
  public Deep boundary is expanded Deep.
- Narrow, validated `surf_*` metadata may preserve source distinctions that
  Deep otherwise erases: module capitalization, dimension grouping, and exact
  pipe-stage origin. Unknown `surf_*` keys are invalid. Semantic metadata must
  resugar or reconstruct; derived/debug metadata may normalize away.
- The alternate verbose Surf dialect is removed. Debug output is canonical
  Surf followed by stable, sorted comments describing non-surface Deep
  metadata.

The v0.19 cutover is atomic: grammar rejection, migration support, repository
corpus conversion, total resugaring, and emitter enforcement land together so
no accepted main-branch state depends on an unavailable migration step.

## Proposed executable contract

A follow-up change should add one table-driven single-grammar corpus. Each case
should contain its one accepted Surf form, rejected alias forms, and expected
canonical Deep. The authoritative assertions should be:

1. the canonical Surf parses and desugars to the expected Deep;
2. every former syntax-safe alias is rejected with a targeted negative test;
3. the canonical Surf is a formatter fixed point;
4. every valid Deep tag has a tested canonical Surf resugaring, with exhaustive
   matching so adding a tag makes the resugarer fail to compile until mapped;
5. resugaring the expected Deep yields the canonical Surf in normal mode;
6. reparsing and redesugaring emitted Surf reproduces the expected Deep modulo
   only the explicitly enumerated `normalize_deep` metadata rules;
7. malformed Deep is rejected, while valid Deep never uses a placeholder or
   generic meaning-changing fallback; and
8. the CLI corpus, examples, and book contain no non-canonical spelling.

Rules that require type, effect, or ownership information cannot be grammar
rejections. They should have paired positive and negative tests in the blocking
checked-source gate rather than unconditional formatter rewrites.

A practical migration sequence is:

1. amend spec/01 and spec/02 with the reviewed single-grammar table;
2. decide the release boundary and whether old source gets a separate migration
   command;
3. repair the cast and `par` Deep-to-Surf round trips;
4. add the table-driven parser/desugar/formatter/resugaring oracle;
5. reject syntax aliases in the parser and context-sensitive aliases in the
   blocking checked-source gate;
6. migrate the executable corpus, examples, and book; and
7. remove alias acceptance tests from the canonical parser, retaining them only
   for an explicitly versioned migration path if one is shipped.

## Evidence inspected

- [`spec/01-nomenclature.md`](../../spec/01-nomenclature.md), especially §§3.5,
  3.6, and 12
- [`spec/02-surf-syntax.md`](../../spec/02-surf-syntax.md), especially P4, P4a,
  P5, P9-P12, the formal grammar, and the complete desugaring reference
- [`crates/chelis-surf/src/parser.rs`](../../crates/chelis-surf/src/parser.rs)
- [`crates/chelis-surf/src/format.rs`](../../crates/chelis-surf/src/format.rs)
- [`crates/chelis-surf/src/desugar.rs`](../../crates/chelis-surf/src/desugar.rs)
- [`crates/chelis-surf/src/decompile.rs`](../../crates/chelis-surf/src/decompile.rs)
- [`crates/chelis-lint/src/rules/surf_def_arrow_form.rs`](../../crates/chelis-lint/src/rules/surf_def_arrow_form.rs)
- [`crates/chelis-lint/src/rules/prefer_pipe_operator.rs`](../../crates/chelis-lint/src/rules/prefer_pipe_operator.rs)
- [`crates/chelis-lint/src/rules/redundant_linearity_call.rs`](../../crates/chelis-lint/src/rules/redundant_linearity_call.rs)
- formatter probes for the forms listed above and the complete
  `cargo test -p chelis-surf` suite
