# naming-and-style

## Purpose

Define the identifier grammar, filesystem and manifest naming, Surf/Rust/Python identifier
conventions, module ladders, function-naming patterns, documentation and test naming, and
the `chelis lint` enforcement contract for the Chelis ecosystem. This capability is the
current truth of how names and style are constrained and enforced across the monorepo and
downstream shells.

**Source:** captured from [`spec/01-nomenclature.md`](../../../spec/01-nomenclature.md).

**Related capability:** [`lint-traversal-policy`](../lint-traversal-policy/spec.md) owns the
traversal contract in §12.2.

## Requirements

### Requirement: Surf identifier case-split

The Surf lexer SHALL classify identifiers by first-character case: a lowercase-leading
identifier lexes as a value identifier (`Ident`) and an uppercase-leading identifier lexes
as a type identifier (`TypeIdent`). The Surf identifier charset SHALL be
`[A-Za-z_][A-Za-z0-9_]*`, admitting no hyphens.

#### Scenario: Lowercase name is a value identifier

- **WHEN** the lexer reads `predict` in value position
- **THEN** it produces a value identifier usable for a function, value, parameter, or dimension

#### Scenario: Uppercase name is a type identifier

- **WHEN** the lexer reads `Frame` in a binding position
- **THEN** it produces a type identifier and `Frame = ...` at top level is a parse error rather than a value binding

#### Scenario: Hyphen in a Surf identifier is rejected

- **WHEN** source contains a Surf identifier such as `roll-mean`
- **THEN** the lexer rejects it because `-` is outside the Surf identifier charset

### Requirement: Single-letter uppercase value override

A single ASCII-uppercase letter SHALL be a value identifier when it appears in a position
that unambiguously binds or names a value: a top-level value-binding LHS, a function or
lambda parameter, or a block binding. The override SHALL be single-letter only; a
multi-letter PascalCase name SHALL remain a type or constructor in every position.

#### Scenario: Single uppercase letter binds a value

- **WHEN** a function is written `def payoff(S, K) = ...`
- **THEN** `S` and `K` are accepted as value parameters (finance notation) per chelis#437

#### Scenario: Multi-letter PascalCase LHS is still a parse error

- **WHEN** source contains `Foo = expr` at top level
- **THEN** it is a parse error because the single-letter override does not extend to multi-letter PascalCase names

### Requirement: Module-path lowering

A `module Foo.Bar` declaration SHALL resolve to the file path `foo/bar.ch` relative to the
package `src/`, lowercasing each PascalCase component and joining with `/`. Consequently
`.ch` filenames SHALL be lowercase or snake_case to be importable.

#### Scenario: Module path maps to a lowercase file

- **WHEN** a module is declared `module Coral.Frame`
- **THEN** the resolver looks for `coral/frame.ch` under the package `src/`

#### Scenario: Non-importable PascalCase filename

- **WHEN** a Surf source file is named `Frame.ch`
- **THEN** it cannot be imported through the module-path lowering, which only produces lowercase paths

### Requirement: Reserved keywords

Surf SHALL reserve the 28 active lowercase keywords (`def sig type dim macro match with fn
module import if then else grad vmap jit realize copy tensor cast export par do quote unquote
splice true false`) plus the five future-reserved words (`effect handler perform resume borrow`).
`property`, `forall`, `where`, `opaque`, and `invariant` SHALL remain contextual. A reserved
keyword SHALL NOT be used as a user identifier.

#### Scenario: Keyword is recognized as a keyword

- **WHEN** the lexer reads `def`
- **THEN** it produces the `def` keyword token, not a value identifier

#### Scenario: Reserved word rejected as a binder

- **WHEN** source attempts to bind a value named `match`
- **THEN** parsing fails because `match` is a reserved keyword

### Requirement: Closed Deep tag vocabulary

Deep SHALL accept only tags from the closed vocabulary hardcoded in the Deep validator.
Single-token tags SHALL be bare lowercase (e.g., `app`, `var`, `lit`, `def`) and compound
tags SHALL use hyphens as the compound separator (e.g., `t-fn`, `t-prim`, `pat-ctor`,
`d-var`). The vocabulary SHALL NOT be user-extended.

#### Scenario: Known compound tag validates

- **WHEN** a Deep node uses the tag `t-fn`
- **THEN** the validator accepts it as part of the closed vocabulary

#### Scenario: Unknown tag is rejected

- **WHEN** a Deep node uses a tag `t-widget` that is not in the closed vocabulary
- **THEN** the validator rejects it

### Requirement: Deep user-symbol charset

Any Deep `Symbol` that is not in the closed tag vocabulary SHALL satisfy the Surf identifier
charset `[A-Za-z_][A-Za-z0-9_]*`. Hyphens SHALL be admitted only for the closed compound-tag
vocabulary, never for user-defined Deep symbols.

#### Scenario: User symbol respects the Surf charset

- **WHEN** a Deep `var` names the symbol `inner_product`
- **THEN** the lint accepts it because it satisfies the Surf identifier charset

#### Scenario: Hyphenated user symbol is a violation

- **WHEN** a Deep emitter produces a non-tag `Symbol` `inner-product`
- **THEN** the §1.5 lint rule flags it as a violation of the Surf user-symbol charset

### Requirement: Formatter and backend identifier fidelity

`chelis fmt` SHALL NOT rewrite identifier case, and mixed identifier styles SHALL survive
format round-trips. The C and HIP backends SHALL emit user names as-is with no symbol
mangling or case rewriting.

#### Scenario: Formatter preserves identifier case

- **WHEN** `chelis fmt` formats a file containing a mixed-case identifier
- **THEN** the identifier's case is preserved byte-for-byte across the round-trip

#### Scenario: Formatter is not a style enforcer

- **WHEN** an identifier violates a naming convention
- **THEN** `chelis fmt` leaves it unchanged and enforcement is left to `chelis lint`, not the formatter

### Requirement: Filesystem and manifest naming conventions

The repository filesystem layer SHALL follow fixed casing rules: top-level repo directories
and Rust crate/`package.name` are kebab-case; `.rs`, `.ch`, `.dp`, and `.py` files are
lowercase/snake_case; a Reef package `name` is kebab-case while its `module_prefix` is
PascalCase; CI workflow filenames are lowercase with no separator.

#### Scenario: Conforming manifest names pass the lint

- **WHEN** a Reef package declares `name = "chelis-std"` and `module_prefix = "Std"`
- **THEN** the naming lint accepts both because the name is kebab-case and the prefix is PascalCase

#### Scenario: PascalCase Surf filename is flagged

- **WHEN** a Surf source file is named `LinAlg.ch`
- **THEN** the naming lint flags it because `.ch` files must be lowercase or snake_case

### Requirement: Surf value and type identifier conventions

Surf types and ADT constructors SHALL be PascalCase; functions and values SHALL be
snake_case except for the single-letter math-notation carve-out (chelis#437); dimensions
SHALL be snake_case or a single letter. The single-letter uppercase value carve-out SHALL
NOT extend to `def`/`sig` function names.

#### Scenario: snake_case function name is accepted

- **WHEN** a function is declared `def inner_product(a, b) = ...`
- **THEN** the `surf-value-snake-case` rule accepts it

#### Scenario: Uppercase function name is flagged

- **WHEN** a function is declared `def N(x) = ...`
- **THEN** the `surf-value-snake-case` lint flags it, because the single-letter carve-out does not cover function names

### Requirement: Def arrow return-type form

When a `def` carries an explicit return type it SHALL be written in the arrow form
`def name(params) -> T = expr`. The canonical parser SHALL reject the colon form
`def name(params) : T = expr`; the raw-source `surf-def-arrow-form` lint SHALL name the
violation, and `chelis migrate surf --from 0.18` SHALL rewrite it. `chelis fmt` SHALL NOT
serve as a dialect migration path.

#### Scenario: Arrow form passes the style gate

- **WHEN** a function is written `def loss(p: f32, q: f32) -> f32 = ...`
- **THEN** `surf-def-arrow-form` accepts it

#### Scenario: Colon return-type form requires explicit migration

- **WHEN** a function is written `def loss(p: f32, q: f32) : f32 = ...`
- **THEN** canonical parsing rejects it, `surf-def-arrow-form` names it, and the v0.18 migration command rewrites it to the arrow form

### Requirement: Pipe-first first-argument insertion

A pipe stage SHALL insert the piped value as the first argument of the stage call:
`x |> f(y, z)` SHALL mean `f(x, y, z)`, and a bare stage `x |> f` SHALL mean `f(x)`. Naming
rules that refer to a function's principal argument SHALL use this same first-argument
interpretation.

#### Scenario: Call stage inserts the piped value first

- **WHEN** source is `x |> add(bias)`
- **THEN** it desugars to `add(x, bias)`

#### Scenario: Last-argument insertion is not the semantics

- **WHEN** a later argument position is intended for the piped value
- **THEN** an explicit lambda `x |> fn (v) -> f(y, v)` is required, because `x |> f(y)` never means `f(y, x)`

### Requirement: Module ladder conventions

Module names SHALL form a ladder rooted at the shell `module_prefix` where every component
is PascalCase. Compound components SHALL use Title-case per word rather than ALL-CAPS
(`Hamt`, not `HAMT`).

#### Scenario: Title-case compound module component

- **WHEN** a module is named `Nautilus.LinAlg`
- **THEN** the naming lint accepts it because each component is PascalCase Title-case

#### Scenario: ALL-CAPS abbreviation flagged

- **WHEN** a module or constructor is named `HAMT`
- **THEN** the naming lint flags it because compounds use Title-case (`Hamt`), not ALL-CAPS

### Requirement: Function prefix and type-suffix policy

Private/helper functions SHALL carry a uniform short domain-shorthand prefix while public
functions SHALL NOT; recognized model/algorithm and math/ML prefixes SHALL be applied
uniformly across their family. Type suffixes (`_int`, `_f32`, `_vec`, `_scalar`) SHALL
describe the element type or shape of the principal argument and SHALL NOT be overloaded to
denote a container or dispatch form.

#### Scenario: Element-type suffix on the principal argument

- **WHEN** a function is named `parse_int(s: string) -> Option[i64]`
- **THEN** the naming lint accepts the `_int` suffix under the parser/converter idiom

#### Scenario: Dispatch-form suffix is flagged

- **WHEN** a function is named `is_nan_int(f: Frame, name: string)` to mean "Frame variant"
- **THEN** the naming lint flags it because `_int` conflates element type with dispatch form; the correct form is `is_nan_col`

### Requirement: Documentation filename conventions

Numbered language specs SHALL use numeric-prefix kebab-case; design files SHALL be
snake_case; per-shell narrative docs SHALL be snake_case while status reports SHALL be
SCREAMING_SNAKE_CASE; mdBook chapters under a `book/` path SHALL be kebab-case as a
deliberate exception, with `SUMMARY.md` and `README.md` exempt. New public strings SHALL
avoid em dashes under the blocking `no-em-dash-in-public-strings` rule.

#### Scenario: mdBook chapter uses kebab-case

- **WHEN** an mdBook chapter under `docs/book/src/` is named `monte-carlo.md`
- **THEN** the doc-filename lint accepts it as the deliberate kebab-case exception

#### Scenario: Em dash in a user-facing string is flagged

- **WHEN** a Surf, Deep, Rust, or Python string literal likely to reach users contains an em dash `a — b`
- **THEN** the blocking `no-em-dash-in-public-strings` rule flags it (Python docstrings excluded)

### Requirement: Test naming conventions

Surf unit tests SHALL be named `test_*` and illustrative examples `example_*`; Rust test
functions SHALL be descriptive snake_case; Insta snapshot files SHALL use the
`{context}__{section}__{test_name}.snap` double-underscore form.

#### Scenario: Surf test prefix accepted

- **WHEN** a Surf test is named `def test_softmax_sums_to_one() = ...`
- **THEN** the `surf-test-name-prefix` rule accepts it

#### Scenario: Missing test prefix flagged

- **WHEN** a Surf function carrying the `Test` effect is named `check_softmax`
- **THEN** the `surf-test-name-prefix` rule flags it because a `Test`-effect function must be `test_*` or `example_*`

### Requirement: Lint severity and style-gate contract

`chelis lint` SHALL report blocking violations as `path:line:col: rule_id (§ref): message`
and exit nonzero on any blocking violation, while advisory/warning rules SHALL never cause
`chelis lint --check` to fail. `chelis build`, `chelis check`, `chelis validate`, and
`chelis eval --file` SHALL invoke the blocking rules and `chelis fmt --check` before the
front-end pipeline. `--allow-style-violations` and `CHELIS_STYLE_GATE_DISABLE=1` SHALL bypass
only the style gate and SHALL NOT be used in CI.

#### Scenario: Blocking violation fails the check

- **WHEN** `chelis lint --check` finds a blocking naming violation
- **THEN** it prints the `path:line:col: rule_id (§ref): message` diagnostic and exits nonzero

#### Scenario: Advisory warning does not fail the check

- **WHEN** `chelis lint --check` finds only a `redundant-linearity-call` advisory
- **THEN** it prints a `warning:`/`advisory:` line and still exits zero

#### Scenario: Escape hatch bypasses only the style gate

- **WHEN** a build is run with `--allow-style-violations`
- **THEN** the style gate is bypassed with a stderr warning but parse, type, effect, validation, evaluation, and backend errors remain fatal

### Requirement: Opaque domain construction discipline

A type marked `opaque: true` SHALL be constructible directly only inside its defining
module; outside that module the blocking `opaque-domain-construction` lint SHALL reject
Surf/Deep record construction, casts targeting the marked type, and record-update of the
marked type. The type checker SHALL remain the authoritative gate via
`CheckErrorKind::OpaqueTypeViolation`.

#### Scenario: In-module construction is allowed

- **WHEN** code inside the defining module constructs a value of the opaque type
- **THEN** both the checker and the lint permit it so smart constructors can be implemented

#### Scenario: Out-of-module construction is rejected

- **WHEN** code outside the defining module constructs the opaque ADT directly or casts to it
- **THEN** the `opaque-domain-construction` lint rejects it and the checker raises `OpaqueTypeViolation`
