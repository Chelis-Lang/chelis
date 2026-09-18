## ADDED Requirements

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

### Requirement: Metadata keys and active vocabulary

Metadata keys SHALL use the identifier charset `[A-Za-z_][A-Za-z0-9_]*` (no hyphens) and
carry compiler-relevant annotations. The `type` key SHALL be checked rather than trusted, and
metadata fields SHALL be preserved by all spec-defined transformations and round-trip through
canonical form. The `surf_*` namespace SHALL be closed to `surf_path`,
`surf_dim_group_size`, `surf_pipe_stage`, `surf_literal_style`, and
`surf_binding_type`; an unknown key in that namespace SHALL be rejected. Known keys SHALL
also be rejected outside their closed contracts: `surf_path` is a string on a module or
import whose ASCII-lowercased value equals the node's lowered path child,
`surf_dim_group_size` is a positive integer on the first grouped `defdim`,
`surf_pipe_stage` is `"call-first"` on an `fn` used as a non-initial pipe stage,
`surf_literal_style` is `"unsuffixed"` or `"explicit"` on a `lit`, and
`surf_binding_type` is `"inferred"` or `"explicit"` on a `bind` value. The latter two
markers preserve authored-versus-inferred Surf distinctions and do not change evaluation.
Round-trip normalization MAY erase those two origin markers only when their value and
placement are valid and after they select the Surf reconstruction. It MAY also erase a
`surf_path` equal to the deterministic default spelling of its lowered child and a correctly
placed `surf_dim_group_size: 1`; malformed, misplaced, and non-default markers and all other
validated `surf_*` values SHALL remain visible.
Repeatable Surf `with contract = "..."` property options SHALL be represented by ordered
`property_contracts: (tuple {} string...)` metadata on the property `def`.

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
A nested `app` whose callee is another expression SHALL remain nested and SHALL NOT be
flattened; canonical Surf SHALL group that callee explicitly.

#### Scenario: Full application type-checks

- **WHEN** a 3-argument function is applied to 3 arguments
- **THEN** the application type-checks

#### Scenario: Arity mismatch is a type error

- **WHEN** a 3-argument function receives 2 arguments in one `app`
- **THEN** it is a type error rather than a partially applied closure

### Requirement: Pipe is a first-class node

`pipe` SHALL be preserved in Deep as a first-class node rather than desugared to nested `app`,
with `(pipe {} e1 e2 e3)` evaluating as `(app {} e3 (app {} e2 e1))`. Each element after the
first SHALL be a function or lambda. A canonical Surf producer SHALL resugar a nested Deep
`app` chain as a pipeline only when typed information proves a linear first-argument chain;
without that proof it SHALL preserve application syntax. (This promotion is not fully
implemented; see chelis#1171.)

#### Scenario: Pipe evaluation order

- **WHEN** Deep is `(pipe {} x f g)`
- **THEN** it evaluates as `g(f(x))` while remaining a `pipe` node for round-tripping

#### Scenario: Non-function pipe stage is rejected

- **WHEN** a pipe stage after the first is not a function or lambda
- **THEN** it is rejected because every stage after the first must be callable

#### Scenario: Application promotion requires typed proof

- **WHEN** a Deep application chain is eligible for canonical Surf output
- **THEN** it becomes a pipeline only when the typed proof establishes value-preserving first-argument insertion

### Requirement: Canonical form and ordering

Deep SHALL have exactly one textual representation per program: 2-space indent, 80-column flat
threshold, structured multi-line `fn`/`let`/`bind`, no comments, and single trailing newline.
Import names SHALL be alphabetized while module declarations, record/record-update `kv`
pairs, match arms, and `let` bind pairs SHALL preserve declaration order. Record order is
the left-to-right evaluation order.

#### Scenario: Record kv pairs preserve written order

- **WHEN** a record is written `(record {} Adam (kv {} lr ...) (kv {} eps ...))`
- **THEN** canonical form retains `lr` before `eps`

#### Scenario: Match arms keep declaration order

- **WHEN** a `match` has multiple arms
- **THEN** canonical form preserves their declaration order because arm order is semantically meaningful

### Requirement: Literal normalization and defaults

Canonical Deep SHALL normalize literals: integers to decimal without leading zeros and finite
floats to the shortest round-trippable spelling, adding `.0` when otherwise integer-like and
using lowercase `e` only when selected by that printer. An unsuffixed integer literal SHALL
bind at `i32` and an unsuffixed float at `f32`, overridable only by a suffix, contextual
tensor-literal inference, or an explicit `cast`. The closed suffix set SHALL match Surf's.

#### Scenario: Float canonicalization

- **WHEN** a producer supplies `1.` or a longer spelling of the same finite value
- **THEN** canonical form emits the shortest representation, such as `1.0`

#### Scenario: Non-finite float is not public Deep

- **WHEN** a producer constructs a NaN or infinity float literal
- **THEN** validation/resugaring rejects it because canonical Surf has no representation

### Requirement: Canonical Deep integer dtype spelling

Canonical Deep SHALL spell signed integer primitives `i8`, `i16`, `i32`, and
`i64`. Normal Deep ingress SHALL reject the retired v0.18 `int*` spellings.
The explicit v0.18 Deep migration SHALL rewrite only the primitive symbol of a
`t-prim` node and SHALL preserve arbitrary identifiers, metadata, and strings.

#### Scenario: Canonical t-prim uses i64

- **WHEN** Deep represents the 64-bit signed-integer primitive
- **THEN** canonical printing emits `(t-prim {} i64)`

#### Scenario: Migration is AST-scoped

- **WHEN** v0.18 Deep contains both `(t-prim {} int64)` and the string `"int64"`
- **THEN** migration rewrites only the primitive symbol

### Requirement: Total canonical Surf resugaring

Every structurally valid public Deep tag SHALL have a canonical Surf AST representation.
Deep-to-Surf emitters SHALL construct that shared AST and use the canonical Surf printer;
they SHALL NOT maintain a second handwritten source dialect. Desugaring the result SHALL
recover Deep modulo only the derived metadata normalization named by the numbered source spec.
Matching `defsig`/`def` pairs SHALL collapse into one inline typed declaration when Surf can
represent the complete contract there. A binder-bearing `defsig` paired with a non-function
`def` SHALL instead remain a standalone binder-bearing `sig` followed by an untyped value
binding, because Surf value bindings have no binder-list position and SHALL NOT free the
quantified names.
That normalization SHALL materialize a checked standalone `def` type as a `defsig`, map empty
`tuple`/`t-tuple` to unit without mapping any zero-argument `app` to a `var`; uppercase
constructor `var`, zero-argument `app`, and zero-field `record` forms SHALL remain distinct
without erasing semantic type data. Multi-pair `bind` nodes SHALL resugar in
their written sequential order; empty `pat-tuple` SHALL resugar directly as `()`.
Negative Deep literals SHALL normalize to Surf's unary-minus application shape. The full
`i64` minimum SHALL use Surf's direct signed-minimum literal, while a narrower signed
minimum SHALL use a non-overflowing decomposition; float-typed integer atoms SHALL normalize
to the equivalent float atom. Negative `pat-lit` values SHALL instead resugar directly as an
unsuffixed negative pattern, including negative zero and the full `i64` minimum. Nested
application SHALL preserve its association through an explicitly grouped Surf callee. Every
Deep string SHALL use Surf's named or minimal lowercase control escape and remain
representable. Any public Deep name that cannot occupy its corresponding Surf identifier
position SHALL fail resugaring explicitly rather than be rewritten.
Surf property syntax SHALL represent user-authored properties only. A property carrying
`property_source_kind: "bridge:c-earchin"` or any `property_source_id` SHALL fail resugaring
explicitly rather than be rewritten with user provenance. `property_quantifiers` SHALL be
present and SHALL exactly match the property `fn` parameter list before resugaring. An
adjacent property `defsig`'s explicit binder list and `dtype_bounds` SHALL resugar after the
property name and SHALL survive AST serialization and public wire conversion.

#### Scenario: Direct Deep forms remain distinct

- **WHEN** Deep contains `block`, `record-update`, `quote`, `unquote`, or `splice`
- **THEN** resugaring uses `do`, `with`, `quote`, `unquote`, or `splice` respectively

#### Scenario: Polymorphic value signature remains bound

- **WHEN** a non-function `def` is paired with a `defsig` whose explicit binder list is nonempty
- **THEN** resugaring emits a standalone binder-bearing `sig` and an untyped value binding
- **AND** desugaring the result recovers the original binder ownership

#### Scenario: Unrepresentable input fails explicitly

- **WHEN** malformed Deep reaches the resugaring boundary
- **THEN** resugaring returns a structural error rather than a placeholder Surf expression

#### Scenario: Invalid surface name fails explicitly

- **WHEN** a Deep value or field name contains punctuation or uses the wrong Surf casing
- **THEN** resugaring returns an identifier error rather than emitting invalid or meaning-changing Surf

#### Scenario: Property provenance is not forged through Surf

- **WHEN** a Deep property carries bridge provenance or a producer-local source ID
- **THEN** resugaring fails explicitly rather than emitting a user-authored `@property`

#### Scenario: Property binders must agree

- **WHEN** `property_quantifiers` differs from the property `fn` parameter list
- **THEN** resugaring fails rather than emitting renamed binders and free variables

#### Scenario: Property declaration binders remain bound

- **WHEN** a property `defsig` binds `p` and its quantifier type uses `(t-var {} p)`
- **THEN** canonical Surf emits `@property name[p] forall(...)`
- **AND** desugaring and public wire conversion preserve that binder list

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
