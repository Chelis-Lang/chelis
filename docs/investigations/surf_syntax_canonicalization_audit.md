# Surf syntax canonicalization audit

**Audit snapshot:** 2026-08-01, repository head `88fe034e1c73`

**Implementation alignment:** Surf v0.19 implementation reviewed through
`7b1100ea1c28913346abbc9f899ca0fe46e58351`

**Tracking:** [chelis#1024](https://github.com/Chelis-Lang/chelis/issues/1024)

This document inventories Surf spellings that are accepted for the same parser
node, the same canonical Deep form, or the same intended operation. It records
what `chelis fmt`, `chelis surf`, the linter, and the active specifications did
at that snapshot, and where Chelis had or had not selected a preferred spelling.

This is an investigation and implementation design, not a new language
contract. Language decisions belong in
[`spec/02-surf-syntax.md`](../../spec/02-surf-syntax.md) and
[`spec/03-deep-syntax.md`](../../spec/03-deep-syntax.md); style decisions and
their enforcement belong in
[`spec/01-nomenclature.md`](../../spec/01-nomenclature.md). The numbered-spec
edits in the implementation branch control the behavior that branch proposes,
but their presence beside the code does not prove that a choice had prior
design approval. The formatter is the executable layout oracle; the parser plus
blocking checked-source gate is the acceptance oracle for canonical Surf.

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

The inventory sections are deliberately historical: they describe the
pre-cutover grammar at the audit snapshot. The later tables compare the original
approval record with the behavior selected by the implementation. A row marked
**implementation-derived** is not retroactively approved merely because the
same branch added it to a numbered spec and implemented it.

## Executive result at the audit snapshot

Chelis does not yet have one complete canonical Surf contract.

The strongest existing decision is function return syntax: `->` is declared
canonical, `chelis fmt` rewrites `:`, and the blocking
`surf-def-arrow-form` lint rejects the colon form. Many other aliases happen to
be normalized by the formatter, but that choice is not stated in the numbered
spec. Several semantically equivalent forms, notably curried versus multi-arg
application, nested calls versus pipes, `unit` versus `()`, and one-expression
blocks, remain distinct after formatting.

There are therefore four separate jobs:

1. document one repository/output spelling for each Surf construct in the
   active specs;
2. distinguish harmless, value-preserving input spellings from semantic or
   structurally ambiguous alternatives;
3. keep `chelis fmt` idempotent while allowing it to normalize those harmless
   input spellings to repository form; and
4. require every Surf emitter, especially `chelis surf`, to resugar Deep into
   that one parseable form without changing its Deep meaning.

The reviewed target is canonical repository and producer output, with a small
set of syntax-safe aliases accepted as input. Semantic or structurally
ambiguous alternatives, such as return-type `:` or juxtaposition application,
remain errors. Compatibility that requires different semantics or a legacy
grammar lives in the explicit migration command rather than the normal parser.

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
| `Ctor`, `Ctor()`, and `Ctor {}` in expressions | These are respectively a constructor reference/value (`var`), a zero-argument application (`app`), and a zero-field record (`record`). Parsing, formatting, resugaring, and `normalize_deep` preserve all three structures. |
| `Ctor`, legacy `Ctor()`, and `Ctor {}` in patterns | A pattern has no application node: bare `Ctor` is the canonical nullary `pat-ctor`, legacy `Ctor()` migrates to it, and `Ctor {}` remains the distinct zero-field `pat-record`. |

## Emitter findings at the audit snapshot

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

## Specification and implementation drift at the audit snapshot

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
format(format(surf)) = format(surf)
normalize_deep(desugar(resugar(desugar(surf))))
  = normalize_deep(desugar(surf))
```

Formatting chooses the syntactic representative. The final equality is stated
in normalized Deep because Deep intentionally does not carry comments, original
whitespace, or every equivalent Surf sugar. If comments must survive a
Surf-to-Deep-to-Surf tool round trip, they need a separate source-preservation
channel rather than alternate Surf syntax.

`normalize_deep` must be narrowly specified. Metadata that changes typing,
effects, evaluation, validation, or lowering is language meaning: it must either
have canonical Surf syntax or be reconstructed deterministically when Surf is
desugared again. Only explicitly non-semantic or derived metadata, such as
source locations or reproducible checker annotations, may be erased or
canonicalized by the round trip. Calling semantic metadata “Deep-only” would
break the same-language invariant.

## Pre-approval single-grammar proposal (historical)

The following was the proposal taken into design review. It is retained as
audit history. A row became approved only where the following approval record
explicitly selected it; the proposal was not approved wholesale:

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

## Implementation brief and reviewed design

The design discussion following this audit selected the following behavior for
the implementation tracked by chelis#1024. The bullets immediately below are
the original approval record. They do not include decisions that first appeared
later in implementation, tests, or numbered-spec edits.

- The initial implementation brief separated the one-spelling behavior that was
  requested from conveniences that could have been retained. Review superseded
  that strict parser boundary for syntax-safe aliases: the normal parser accepts
  them, canonical producers choose one spelling, and `fmt --check` enforces the
  repository form. Legacy forms that are semantic or structurally ambiguous
  remain reachable only through an explicit `chelis migrate surf --from 0.18`
  path.
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
- Deep `app` and `pipe` remain distinct unless typed information proves that a
  nested application is exactly a value-preserving first-argument chain. A
  canonical producer promotes a proven chain to a pipeline; without that proof
  it emits calls. The stronger promotion is tracked by chelis#1171 and is not
  claimed by this foundation. At resolved ordinary builtin call sites the fixed
  operator vocabulary resugars to infix or prefix notation; builtin names
  remain values and pipe stages.
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

The implementation subsequently exposed decisions that the original record did
not settle. They were reviewed independently rather than accepted merely because
code existed. The four tables below are the approved implementation design:
**Original approval** identifies a choice already selected above, while
**Reviewed approval** identifies a choice ratified after weighing the tradeoffs
below. The controlling language rules live in the numbered specs; this section
records rationale, migration policy, and delivery scope.

The syntax choices left unsettled by the original approval have real tradeoffs:

- **Juxtaposition application:** rejecting it is a deliberate language revision,
  not a claim that every same-line use was historically ambiguous. The initial
  grammar at `d7db8860` and the first parser at `3d9baab9` accepted it;
  chelis#706 and chelis#769 demonstrate the newline-boundary hazard only. The
  decisive rule is Chelis's flat multi-argument application model with no
  implicit currying: `f x y` conventionally suggests nested application that
  Chelis does not provide. Parenthesized calls expose the real flat call
  boundary, while typed first-argument pipes provide concise dataflow without
  importing curried application semantics.
- **Nullary constructors:** collapsing a zero-argument `app` into `var` would
  give one shorter spelling, but it erases Deep structure and makes a constructor
  application unlike every other application. The approved rule preserves
  `Ctor`, `Ctor()`, and `Ctor {}` as `var`, `app`, and `record`. Pattern syntax
  remains bare because Deep patterns have `pat-ctor`, not an application node.
- **Trailing separators:** rejection gives a strict accepted-language boundary,
  but makes multiline edits noisier without protecting semantics. The reviewed
  design accepts one trailing separator in delimited families and in `par`/`do`;
  canonical output omits it except where `(x,)` denotes a singleton tuple.
- **Decorative zero arity:** omission removes redundant empty aliases and keeps
  semantic braces such as `! {}` and `Ctor {}` visibly meaningful, but loses
  uniformity with nonempty type, variant, import, export, and update forms. The
  reviewed design rejects those aliases while retaining semantic empty forms,
  including `Ctor()` as an expression application.
- **Literal, string, keyword, and property details:** canonical spelling still
  gives deterministic producer output, but the parser need not reject harmless
  representations. The reviewed design accepts value-preserving radices, digit
  separators, exponent spellings, and valid Unicode escapes while retaining
  semantic checks for overflow, non-finite values, typed patterns, raw controls,
  and invalid escapes. Property words remain contextual; the direct-form and
  future-reservation decisions are unchanged.

### Accepted and rejected syntax families

This table records the grammar selected in the numbered specs. “Accepted” is
the normal-parser boundary, while canonical producers and `fmt --check` still
select the repository spelling. The isolated v0.18 migration parser additionally
accepts legacy forms that the normal parser cannot interpret safely.

| Family | Accepted by the implementation | Rejected by the implementation | Design provenance |
|---|---|---|---|
| Definitions and signatures | `def f() = e`, `def f(x: A) -> B = e`, `sig f: A -> B` | omitted nullary `()`, return `:` | **Approved.** |
| Calls and constructors | flat `f(x, y)`, `Some(x)`, bare constructor value `None`, explicit zero-argument application `None()`, zero-field record `Ctor {}`, and grouped result application `(f(x))(y)` | juxtaposition, ungrouped `f(x)(y)`, and `Some x` | Flat calls have **original approval**; preserving the three nullary structures has **reviewed approval**. |
| Blocks and direct sequencing | ordinary `{ x = e\n tail }`; `par { a; b }`; `do { a; b }`; one optional trailing `;` in `par`/`do` | ordinary semicolons or one-expression ordinary blocks; a missing separator between multiple `par`/`do` entries or repeated trailing separators | Core block forms have **original approval**; accepting one harmless trailing separator has **reviewed approval**. |
| Delimited lists | commas separate nonempty arguments, parameters, type/dimension arguments, tuple/list/record fields, imports/exports, transform options, effects, and variant payload fields; one trailing comma is accepted; `(x,)` remains the singleton tuple/pattern exception | repeated separators or separators where the grammar has no delimited list | **Reviewed approval.** Canonical output omits the optional trailing separator except for the semantic singleton-tuple comma. |
| Decorative zero arity | `Option`, bare zero-field variant `\| Empty`, `import Demo`, nonempty `export (...)` and `base with {...}`; expression `Ctor()`, zero-field record expression/pattern `Ctor {}`, and semantic empty effect `! {}` retain delimiters because they carry structure | `Option[]`, empty quantifier lists, `\| Empty()`, variant `\| Empty {}`, `import Demo ()`, `export ()`, and `base with {}` | **Reviewed approval.** The rule removes declaration/container aliases without erasing semantic expression, record, pattern, or effect structure. |
| Unit | `()` is the singleton value; `unit` is its type | `()` in a type position or `unit` as a value | **Reviewed approval.** Separating the value and type spellings avoids overloading empty parentheses while preserving Deep `lit unit` and `t-unit`. |
| Effects and records | `Diff`/`Random`/`Accum`/`IO`/`Test`/`Resource(...)`, record puns, authored field order with left-to-right evaluation | lowercase built-in effects and redundant same-name `x: x` fields | **Approved.** |
| Transforms and pipes | `grad(f, wrt=x)`, `vmap(f)` for axis zero, `vmap(f, axis=n)` otherwise; call-stage sugar represents first-argument insertion and an explicit lambda represents later positions; a typed producer promotes a proven first-argument application chain to a pipeline | positional transform options, singleton-tuple `wrt`, explicit `axis=0`, using call-stage syntax for a later position, or promoting without the typed proof | Core forms have **original approval**; typed promotion is the controlling target and its implementation is tracked by chelis#1171. |
| Literals and literal patterns | finite decimals plus value-preserving radix, underscore, exponent, and escape spellings; exact suffix semantics; unsuffixed pattern literals, including negatives, `-0.0`, and signed minimum | malformed spellings, overflow, non-finite values, typed/suffixed literal patterns, or noncanonical negative-pattern structure | The decimal emitter has **original approval**; syntax-safe alias acceptance and the semantic rejection boundary have **reviewed approval**. |
| Strings | printable Unicode, six named escapes (`\"`, `\\`, `\n`, `\t`, `\r`, `\0`), and valid `\u{...}` escapes including uppercase and padded digits | raw controls, malformed or out-of-range Unicode escapes, and invalid multiline strings | **Reviewed approval.** Canonical output still chooses one escape form, while the parser accepts valid value-preserving aliases. |
| Keywords and properties | active reserved words include `do`, `quote`, `unquote`, and `splice`; future-reserved words remain `effect`, `handler`, `perform`, `resume`, and `borrow`; contextual words include `property`, `forall`, `where`, `opaque`, `invariant`, `wrt`, `axis`, `seed`, `device`, `tolerance`, `samples`, and `contract`; property preconditions omit redundant outer grouping and `property_preconditions`/`property_contracts` preserve order | using a reserved word as an identifier, `where (x <= 1):`, or a non-string contract option | **Reviewed approval.** It adds only the keywords required by direct Surf forms, retains prior future reservations, and keeps grammar-specific property words contextual. |

### Complete `normalize_deep` equivalences and `surf_*` metadata

| Deep input or metadata | Implemented round-trip treatment | Design provenance |
|---|---|---|
| `span`, `loc`, expanded macro `source`, inferred `effects`, `invariant_amenability` | May erase because each is informational or deterministically recomputed; no other metadata class gains this permission. | **Reviewed approval.** A closed erasure list makes round-trip equality useful without permitting semantic metadata loss. |
| Matching `type` metadata on `def`, its `fn`, and parameters beside an exact `defsig` | May erase as redundant; any disagreement is retained and fails the equivalence. | Signature/definition resugaring has **original approval**; the exact redundancy and mismatch rules have **reviewed approval**. |
| Standalone checked `def` carrying semantic `type` metadata | Materialize the equivalent adjacent `defsig`, then apply the exact redundancy rule. | **Reviewed approval.** It preserves checked type information in a Surf form rather than discarding it. |
| Empty `tuple` / `t-tuple`; empty `pat-tuple` | Normalize to unit `lit` / `t-unit`; resugar the pattern directly as `()`. | **Reviewed approval.** Surf has one unit spelling, and Deep treats these shapes as the same zero-product value. |
| Constructor `var`, zero-argument constructor `app`, and zero-field `record` | Preserve all three structures; no normalization equivalence may turn one into another. Lowercase and uppercase zero-argument applications both remain applications. | **Reviewed approval.** This is the explicit correction to the earlier contradictory implementation. |
| Negative numeric `lit` | Normalize to `neg`; full `int64` minimum is direct, while a narrower signed minimum uses `sub(neg(max), 1)`. | **Reviewed approval.** The special cases preserve exact typed values without overflow during reconstruction. |
| Integer atom with a float primitive type | Normalize to the equivalent float atom before the negative-literal rule. | **Reviewed approval.** Typed numerical equality, not the constructed atom variant, controls the canonical Surf literal. |
| Negative `pat-lit` | Resugar directly as the unsuffixed token, including `-0.0` and full `int64` minimum. | **Reviewed approval.** Patterns cannot rely on contextual suffix adoption and therefore use the one direct spelling. |
| Macro-authored Surf | Compare after expansion, because public Deep is expanded Deep. | **Original approval.** |
| `surf_path` | String only on `module`, `import`, or `import-all`; ASCII lowercase must match the path child. Default spelling may erase; valid non-default spelling is preserved; mismatch/misplacement fails. | Preserving module capitalization has **original approval**; the closed placement, value, erasure, and mismatch rules have **reviewed approval**. |
| `surf_dim_group_size` | Positive integer only on the first `defdim`; `1` may erase, larger valid groups persist, malformed/misplaced data fails. | Preserving grouping has **original approval**; the closed schema and default erasure have **reviewed approval**. |
| `surf_pipe_stage` | Exact `"call-first"` only on an `fn` child used as a non-initial `pipe` stage; retain and validate it; any other value or placement fails. | Preserving pipe-stage origin has **original approval**; the closed value and placement rules have **reviewed approval**. |
| `surf_literal_style` | `"unsuffixed"` / `"explicit"` only on `lit`; may erase after selection and is recreated by desugaring; any other value or placement fails. | **Reviewed approval.** Deterministic suffix adoption outweighs the cost of one additional validated producer key. |
| `surf_binding_type` | `"inferred"` / `"explicit"` only on the expression child of a `bind`; may erase after selection and is recreated; any other value or placement fails. | **Reviewed approval.** It prevents resugaring from inventing or deleting an authored binding annotation. |
| Unknown or malformed `surf_*`; non-finite constructed Deep floats | Reject as invalid or unrepresentable; normalization cannot hide the error. | Unknown-key rejection has **original approval**; exact key validation and non-finite rejection have **reviewed approval** as fail-closed boundaries. |

### Migration CLI, filesystem, and comment guarantees

| Surface | Implemented guarantee | Design provenance |
|---|---|---|
| Version and mode selection | Only `--from 0.18`. With neither flag, exactly one path goes to stdout without writes. `--check` accepts one or many paths, writes nothing, and lists stale paths. `--inplace` accepts one or many paths. The flags are mutually exclusive. | The versioned migration has **original approval**; modes and arity rules have **reviewed approval** as a minimal scriptable CLI. |
| Whole-batch preflight | Before any write, every input must pass legacy parsing, canonical formatter fixed-point validation, macro expansion, and normalized Surf → Deep → Surf → Deep equality. Non-finite or otherwise unrepresentable forms reject the batch. | Atomic cutover and round-trip laws have **original approval**; the fail-before-write algorithm has **reviewed approval**. |
| Literal and string migration | Rewrite genuine v0.18 forms, redundant zeros, suffix aliases, signed minima, and escapes through the canonical printer. Harmless radices, separators, exponent spellings, and valid Unicode escapes are also accepted by the normal parser and normalize through `fmt`. Preserve a default-type suffix when contextual adoption changes meaning; add `.0` before a float suffix on integer-looking digits. Reject typed patterns; emit direct unsuffixed negative patterns. | **Reviewed approval.** These rewrites preserve representable values while keeping semantic failures closed. |
| Reserved identifiers | Reject a v0.18 binding whose name is reserved by v0.19 and direct the author to rename it manually; migration never guesses a semantic replacement. | **Reviewed approval.** Identifier renaming can affect imports, exports, references, and external contracts, so it requires an authored choice. |
| Comments | Preserve text and line/block kind at representable boundaries; ambiguous interior attachment rejects the batch; comments are outside Deep equality. | Preservation has **original approval**; rejecting ambiguous attachment has **reviewed approval** because guessing would silently move documentation. |
| Eligible paths | Writable ordinary single-link files only; reject symlinks, non-files, multiple hard links, read-only, or unwritable destinations. | **Reviewed approval.** The narrower filesystem contract avoids aliasing and rollback ambiguity; linked workflows must copy to an ordinary file first. |
| Transaction | Sibling staging and backups, permission preservation, per-file atomic replacement, byte-for-byte rollback, and rollback-failure reporting. | **Reviewed approval.** Strong batch atomicity is worth the staging complexity because a partially migrated repository is not accepted. |

### Affected implementation and consumer surfaces

| Consumer | Required cutover work and completeness check | Design provenance |
|---|---|---|
| Rust Surf parser/tooling | Canonical and v0.18 parsers, shared AST-backed formatter/resugarer/decompiler, exhaustive Deep-tag handling, and round-trip corpus agree. | Parser/emitter convergence follows the **approved** architecture; exact cases inherit the decision status above. |
| tree-sitter | Grammar, highlights, generated parser, parity corpus, and 65,536-value Ryu scanner corpus match Rust; retain GCC/libstdc++ 10 and glibc 2.31 compatibility. | **Reviewed approval.** Parser parity is required; the compatibility floor prevents the external scanner from narrowing supported builders. |
| Cove | Live analysis uses the shared canonical compiler/parser path and canonical fixtures. | **Reviewed approval.** Cove is a direct syntax consumer and cannot retain a private dialect. |
| LSP | Traversal and diagnostics handle new variants and canonical-literal offsets; fixtures are canonical. | **Reviewed approval.** Editor parsing and offsets are part of the syntax cutover. |
| Compiler API | Wire/schema conversion covers record update, `do`, quote, unquote, and splice without fallback. | **Reviewed approval.** Public wire conversion must be exhaustive when Surf gains direct Deep forms. |
| CLI | `fmt` prints the canonical spelling and layout; `surf` uses shared resugaring/oracles; migration owns incompatible legacy syntax and writes; checked commands compare canonical output for style but compile the original, spanned AST. | Core command roles have **original approval**; alias normalization, exact migration modes, and authored-span diagnostics have **reviewed approval**. |
| Examples | Executable and illustrative corpora migrate without changing roles; executable examples retain canonical checks. | Corpus migration has **original approval**; retaining the executable/illustrative boundary is required delivery scope. |
| Book | Reference and migration guidance teach only the approved canonical grammar. | Book migration has **original approval**; the reviewed grammar and migration guarantees must be documented together. |

The cutover is atomic: grammar-boundary changes, migration support, repository
corpus conversion, total resugaring, and emitter enforcement land together so
no accepted main-branch state depends on an unavailable migration step. The
tables above provide the reviewed decisions needed to apply that atomicity rule.

## Executable contract

The implementation must carry one table-driven grammar corpus. Each case
contains its repository form, accepted harmless aliases, rejected semantic or
malformed alternatives, and expected canonical Deep. The authoritative assertions are:

1. the repository Surf form and every accepted alias parse and desugar to the
   expected Deep;
2. every semantic, structurally ambiguous, or malformed alternative is rejected
   with a targeted negative test;
3. the repository Surf form is a formatter fixed point and accepted aliases
   format to it;
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

The implementation sequence is:

1. amend spec/01, spec/02, and spec/03 with the approved language contracts;
2. set the canonical release boundary and isolate old source behind
   `chelis migrate surf --from 0.18`;
3. repair the cast and `par` Deep-to-Surf round trips;
4. add the table-driven parser/desugar/formatter/resugaring oracle;
5. admit syntax-safe aliases in the parser, reject semantic and ambiguous
   alternatives there, and enforce typed rules in the blocking checked-source
   gate;
6. migrate the executable corpus, examples, and book; and
7. retain parity tests for accepted aliases and keep incompatible legacy forms
   only in the explicitly versioned migration path.

## Evidence inspected

- [`spec/01-nomenclature.md`](../../spec/01-nomenclature.md), especially §§3.5,
  3.6, and 12
- [`spec/02-surf-syntax.md`](../../spec/02-surf-syntax.md), especially P4, P4a,
  P5, P9-P12, the formal grammar, and the complete desugaring reference
- [`spec/03-deep-syntax.md`](../../spec/03-deep-syntax.md), especially §§6.3.1,
  6.3.2, and 6.4
- [`crates/chelis-surf/src/parser.rs`](../../crates/chelis-surf/src/parser.rs)
- [`crates/chelis-surf/src/format.rs`](../../crates/chelis-surf/src/format.rs)
- [`crates/chelis-surf/src/desugar.rs`](../../crates/chelis-surf/src/desugar.rs)
- [`crates/chelis-surf/src/decompile.rs`](../../crates/chelis-surf/src/decompile.rs)
- [`crates/chelis-lint/src/rules/surf_def_arrow_form.rs`](../../crates/chelis-lint/src/rules/surf_def_arrow_form.rs)
- [`crates/chelis-lint/src/rules/prefer_pipe_operator.rs`](../../crates/chelis-lint/src/rules/prefer_pipe_operator.rs)
- [`crates/chelis-lint/src/rules/redundant_linearity_call.rs`](../../crates/chelis-lint/src/rules/redundant_linearity_call.rs)
- formatter probes for the forms listed above and the complete
  `cargo test -p chelis-surf` suite
