# deep-syntax

## Purpose

Define Deep, the primary machine interface: the universal 3-tuple node structure and
metadata contract, the span and producer-string trust boundary, the closed 62-tag
vocabulary, module identity, type-expression and dimension resolution, built-in scope,
function application and pipe semantics, canonical form and literal normalization, and
structural/arity/vocabulary validation for constructing, parsing, validating, and printing Deep.

**Source:** captured from [`spec/03-deep-syntax.md`](../../../spec/03-deep-syntax.md).

## Requirements

### Requirement: Universal 3-tuple node structure

Every Deep AST node SHALL be a 3-tuple `(tag {meta} child...)` where the tag is from the
closed vocabulary, the metadata map is always present (empty as `{}`), and children are zero
or more nodes or bare identifiers/literals. Metadata SHALL always occupy element two so a
generator never decides where to attach it.

#### Scenario: Node with empty metadata

- **WHEN** a node is written `(var {} x)`
- **THEN** it parses as tag `var`, empty metadata, and one bare-name child

#### Scenario: Missing metadata map is a parse error

- **WHEN** a node is written `(var x)` with no metadata map
- **THEN** parsing fails because the metadata map must always be present as element two

### Requirement: Metadata keys and defined vocabulary

Metadata keys SHALL use the identifier charset `[A-Za-z_][A-Za-z0-9_]*` (no hyphens) and
carry compiler-relevant annotations. The `type` key SHALL be checked rather than trusted, and
metadata fields SHALL be preserved by all spec-defined transformations and round-trip through
canonical form.

#### Scenario: Producer-specific metadata key accepted

- **WHEN** a node carries `{c_earchin_role: "..."}`
- **THEN** it is accepted because the key matches the no-hyphen identifier charset

#### Scenario: Type metadata is verified, not trusted

- **WHEN** a `lit` node carries a `type` metadata annotation that disagrees with the value
- **THEN** the checker verifies and rejects it rather than trusting the annotation

### Requirement: Span metadata is an opaque validated string

The `span` metadata value SHALL be treated as an opaque string preserved end-to-end through
parsing, lowering, optimization, and codegen. Span IDs SHALL NOT contain ASCII control
characters U+0000–U+001F except space, nor U+007F; the empty string SHALL be a valid span ID.
The parser SHALL reject forbidden characters with a diagnostic naming the byte position and
pointing at §1.1.1.

#### Scenario: Printable-Unicode span is preserved

- **WHEN** a producer emits `{span: "eq1.body"}` or a span with Greek letters
- **THEN** Chelis preserves it verbatim through the pipeline into generated `// span:` comments

#### Scenario: Control character in span is rejected

- **WHEN** a `span` value contains a newline or NUL
- **THEN** the Deep parser rejects it with a diagnostic naming the offending byte and code point

### Requirement: Synthesized span markers are reserved

A chelis-introduced node with no external source region SHALL use the reserved canonical span
form `__synthesized_<pass>__`. External producers SHALL NOT emit span IDs matching
`__synthesized_*__`, and chelis SHALL NOT mint a synthesized marker that omits the wrap.
Synthesized-marker nodes SHALL also carry forward-node spans on the `merged_spans` IR field.

#### Scenario: Tier-2 decomposition marks a synthesized span

- **WHEN** the Tier-2 RISC decomposition introduces a node whose parent had no span
- **THEN** the node's canonical span is `__synthesized_tier2__` and it carries a forward span in `merged_spans`

#### Scenario: External synthesized-shaped span is forbidden

- **WHEN** an external producer emits `{span: "__synthesized_grad__"}`
- **THEN** it violates the reserved-marker contract

### Requirement: Producer-supplied strings validated at the trust boundary

Every producer-supplied string that flows into generated source SHALL be validated at its
trust boundary: at parse time for Deep text, or at construction time for direct IR
construction. `RiscOp::Load`/`Store` names SHALL use the validating `LoadStoreName` newtype
accepting `[A-Za-z_][A-Za-z0-9_.-]*` and rejecting control bytes, whitespace, `%`, `/`,
non-ASCII, and the empty string.

#### Scenario: Tuple-flatten load name is accepted

- **WHEN** lowering constructs a load named `grads.1`
- **THEN** `LoadStoreName` accepts it because `.` is admitted for synthesized tuple-flatten names

#### Scenario: Load name with a slash is rejected at construction

- **WHEN** an IR load name contains `/` or a control byte
- **THEN** the `LoadStoreName` constructor rejects it at construction time

### Requirement: Macro boundary and provenance

LLM-facing Deep SHALL always be expanded Deep: the AST surfaced to generation, repair,
fitness, decompilation, and transforms SHALL contain only ordinary Deep nodes plus optional
`source` provenance metadata. Compiler-internal pre-expansion tags (`defmacro`,
`macro-invoke`) SHALL be outside the public vocabulary and rejected by `chelis validate --deep`.
`source` metadata SHALL be informational and ignored by all passes except error reporting.

#### Scenario: Expanded node carries provenance

- **WHEN** a macro expands to an `app` node
- **THEN** the node is a standard `app` with a `source` key that does not affect parsing, typing, or evaluation

#### Scenario: Internal macro tag is rejected

- **WHEN** raw Deep contains a `defmacro` or `macro-invoke` node
- **THEN** `chelis validate --deep` rejects it as outside the public vocabulary

### Requirement: Closed tag vocabulary

Deep SHALL accept only the closed 62-tag vocabulary; unknown tags SHALL be parse errors in
strict/canonical validation. In fitness-scoring mode an unknown tag SHALL parse as a generic
node and be penalized in the fitness score rather than rejected.

#### Scenario: Known tag validates

- **WHEN** a node uses the tag `record-update`
- **THEN** it is accepted as part of the 62-tag vocabulary

#### Scenario: Unknown tag rejected in strict mode

- **WHEN** canonical validation encounters a tag `widget`
- **THEN** it is a parse error, while fitness-scoring mode parses it as a penalized generic node

### Requirement: Module identity

A named module SHALL be opened by at most one `(module ...)` wrapper per program; re-opening a
module name SHALL be rejected as `DuplicateModule`. Hand-authored Deep SHALL NOT use the reef
linker's reserved internal-name format (`Pkg__<pkg>__<Module>__<Name>`); such a name SHALL be
rejected as `ReservedLinkerName`.

#### Scenario: Single module wrapper accepted

- **WHEN** a program has one `(module {} stats.prob ...)` wrapper
- **THEN** it is accepted

#### Scenario: Re-opened module is rejected

- **WHEN** a program opens the same module name in two `(module ...)` wrappers
- **THEN** both the validator and the checker reject it as `DuplicateModule`

### Requirement: Opaque deftype metadata and derived amenability

A `deftype` MAY carry `opaque: true` and, when opaque, a single `invariant` predicate encoded
as `(fn {} (params {} binder) body)` in metadata plus an `invariant_amenability` string. The
`invariant_amenability` SHALL be derived data, recomputed on desugar and re-verified by the
checker rather than trusted, and SHALL NOT be reconstructed by the decompiler.

#### Scenario: Opaque deftype with invariant validates

- **WHEN** a `deftype` carries `opaque: true` and an `invariant` fn with exactly one binder
- **THEN** the strict validator recurses into the metadata fn and accepts it

#### Scenario: Recorded amenability disagreeing with recomputation is rejected

- **WHEN** a hand-written `.dp` records `invariant_amenability` that disagrees with `classify_predicate`
- **THEN** the checker rejects it because amenability is re-verified against a recomputation

### Requirement: Type-expression resolution is fail-closed

Type expressions SHALL be recursively resolved before entering the checked environment, with
no dropped children, wildcard substitution, or unchecked nominal names. A `t-prim` SHALL have
exactly one primitive-vocabulary symbol child; a `t-adt` SHALL name a precollected header with
its exact arity; `t-var`/`d-var`/`d-rank` SHALL be legal only when the resolution context
supplies the binder.

#### Scenario: Well-formed function type resolves

- **WHEN** a `t-fn` has resolvable argument and return children
- **THEN** resolution accepts it and installs it in the checked environment

#### Scenario: Unknown primitive is a type error

- **WHEN** a type expression is `(t-prim {} f24)`
- **THEN** it is a type error, not an inference hole, because `f24` is outside the primitive vocabulary

### Requirement: Dimension expression shape

Every dimension tag SHALL have exactly one child: a symbol for `d-name`/`d-var`/`d-rank` or an
integer for `d-lit`. A `d-rank` name SHALL appear at most once per `t-tensor`, and unbound
`d-var`/`d-rank` names, wrong child kinds, and missing/extra children SHALL be
type-resolution errors.

#### Scenario: Literal dimension resolves

- **WHEN** a dimension is `(d-lit {} 512)`
- **THEN** it resolves as a literal size with its single integer child

#### Scenario: Repeated rank name is an error

- **WHEN** a `t-tensor` contains `(d-rank {} r)` twice
- **THEN** it is a type-resolution error because a rank name appears at most once per tensor

### Requirement: Built-in scope is not tags

RISC primitives, derived functions, and standard-library names SHALL be values referenced via
`(var {} name)` and called via `(app {} ...)`, not tags. RISC primitives and derived functions
SHALL be available without import; standard-library names SHALL require an explicit `import`.

#### Scenario: Primitive is referenced as a var

- **WHEN** Deep computes `add(a, b)`
- **THEN** it is `(app {} (var {} add) (var {} a) (var {} b))`, not an `add` tag

#### Scenario: Stdlib name requires import

- **WHEN** Deep references `println` without importing `std.io`
- **THEN** the name is unbound because standard-library names are not built-in

### Requirement: Function application is multi-argument

Deep SHALL use explicit multi-argument application rather than currying: `(app {} f x y z)`.
Supplying the wrong number of arguments to a function SHALL be a type error, not partial
application; partial application SHALL require explicit closure construction with `fn`.

#### Scenario: Full application type-checks

- **WHEN** a 3-argument function is applied to 3 arguments
- **THEN** the application type-checks

#### Scenario: Arity mismatch is a type error

- **WHEN** a 3-argument function receives 2 arguments in one `app`
- **THEN** it is a type error rather than a partially applied closure

### Requirement: Pipe is a first-class node

`pipe` SHALL be preserved in Deep as a first-class node rather than desugared to nested `app`,
with `(pipe {} e1 e2 e3)` evaluating as `(app {} e3 (app {} e2 e1))`. Each element after the
first SHALL be a function or lambda.

#### Scenario: Pipe evaluation order

- **WHEN** Deep is `(pipe {} x f g)`
- **THEN** it evaluates as `g(f(x))` while remaining a `pipe` node for round-tripping

#### Scenario: Non-function pipe stage is rejected

- **WHEN** a pipe stage after the first is not a function or lambda
- **THEN** it is rejected because every stage after the first must be callable

### Requirement: Canonical form and ordering

Deep SHALL have exactly one textual representation per program: 2-space indent, 80-column flat
threshold, structured multi-line `fn`/`let`/`bind`, no comments, and single trailing newline.
Import names and record `kv` pairs SHALL be alphabetized while module declarations, match arms,
and `let` bind pairs SHALL preserve declaration order.

#### Scenario: Record kv pairs alphabetized

- **WHEN** a record is written `(record {} Adam (kv {} lr ...) (kv {} eps ...))`
- **THEN** canonical form emits `eps` before `lr`

#### Scenario: Match arms keep declaration order

- **WHEN** a `match` has multiple arms
- **THEN** canonical form preserves their declaration order because arm order is semantically meaningful

### Requirement: Literal normalization and defaults

Canonical Deep SHALL normalize literals: integers to decimal without leading zeros, floats to
`d.d` minimum, scientific floats to `d.dE±d`. An unsuffixed integer literal SHALL bind at
`int32` and an unsuffixed float at `f32`, overridable only by a suffix, contextual
tensor-literal inference, or an explicit `cast`. The closed suffix set SHALL match Surf's.

#### Scenario: Float canonicalization

- **WHEN** a literal is written `1.` or `1e3`
- **THEN** canonical form emits `1.0` and `1.0E+3`

#### Scenario: Unsigned suffix rejected at lex time

- **WHEN** a Deep literal is written `42u8`
- **THEN** it is rejected at lex time because unsigned suffixes are out of scope per §1.1.2

### Requirement: Structural and arity validation

The parser SHALL enforce that every node is a 3-tuple with a closed-vocabulary tag and a valid
metadata map; post-parse arity validation SHALL enforce exact child counts (`if` and `arm`
have exactly 3 children; `fn` and `let` have exactly 2 with a `params`/`bind` first child;
`app` has at least 1).

#### Scenario: Correct arity validates

- **WHEN** a node is `(if {} cond then else)` with three children
- **THEN** arity validation accepts it

#### Scenario: Wrong arity is rejected

- **WHEN** a node is `(if {} cond then)` with two children
- **THEN** arity validation rejects it because `if` requires exactly three children
