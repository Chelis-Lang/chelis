# type-system

## Purpose

Define the Chelis type system: the checked-Deep contract, the primitive dtype set
and per-backend support matrix, tensor/function/ADT types, Hindley-Milner inference, the
named-tensor-dimension algebra with no implicit broadcasting, dimension and rank
polymorphism, opaque-type module identity and invariants, precision rules and mixed-precision
accumulators, runtime shape semantics, fitness scoring, effects, linearity,
numeric-value semantics, and checker-totality atoms.

**Source:** captured from [`spec/04-type-system.md`](../../../spec/04-type-system.md).

## Requirements

### Requirement: Checked Deep contract

The type checker SHALL be the pass that upgrades raw Deep into a `CheckedProgram` carrying
annotated Deep with `type` metadata on the returned tree. Lowering, evaluation, effect
checking, and CLI build/eval paths SHALL consume that annotated tree rather than the original
raw Deep, and effect inference SHALL run after HM type inference on the same annotated tree.

#### Scenario: Check returns an annotated program

- **WHEN** `check_ir_program` succeeds on a well-typed program
- **THEN** it returns a `CheckedProgram` whose tree carries `type` metadata consumed by downstream passes

#### Scenario: Downstream consumes the checked tree, not raw Deep

- **WHEN** effect inference runs
- **THEN** it operates on the checker's annotated tree, not the pre-check raw Deep

### Requirement: Primitive dtype set

The numeric primitive set SHALL be exactly `f32`, `f64`, `bf16`, `f16`, `int8`,
`int16`, `int32`, `int64`, `bool`, and `string`. The spellings in §1.1.1 SHALL remain
reserved non-types and SHALL be rejected by the checker. Their reservations SHALL retain the
semantic families and arithmetic widths declared there: OCP FP8 computes at `f32`; unsigned and
four-bit integers are exact at their own widths; complex arithmetic uses its `f32` or `f64`
component width; and decimal interchange values are exact base-10 at their declared scale.

#### Scenario: Primitive resolves

- **WHEN** a type expression is `(t-prim {} bf16)`
- **THEN** the checker accepts it as a dtype

#### Scenario: Reserved spellings are rejected

- **WHEN** a program uses `(t-prim {} f8e4m3)` or a `uint32` type
- **THEN** the checker rejects it, pointing at §1.1.1's reserved-name set

### Requirement: Per-backend dtype support matrix

Backend codegen SHALL honor the per-backend dtype matrix independently of the language-level
dtype contract. Metal SHALL admit `f64` only when the selected target profile and runtime device
expose native FP64 arithmetic, and SHALL admit `bf16` only when both MSL and the device expose
native bfloat capability. A missing capability SHALL be rejected consistently at the CLI gate,
IR validation, codegen entry, or runtime pipeline boundary where it becomes knowable; no lane
may substitute another dtype or software-emulated arithmetic.

#### Scenario: f32 admitted on every backend

- **WHEN** a program uses f32 with any backend target
- **THEN** codegen admits it

#### Scenario: Metal without FP64 rejects f64

- **WHEN** a program uses f64 with a Metal target profile that lacks native FP64 arithmetic
- **THEN** validation emits the native-FP64 capability diagnostic and no f64 kernel is emitted

### Requirement: Tensor and function type shape

A `t-tensor` SHALL have its precision (a numeric `t-prim` or bound precision variable) as its
last child and dimension expressions as all preceding children; a `t-fn` SHALL be flat with
its return type as the last child. Multi-argument functions SHALL be flat, not curried.

#### Scenario: Tensor precision is the last child

- **WHEN** a type is `(t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))`
- **THEN** it is a 2D `batch × hidden` f32 tensor with precision as the last child

#### Scenario: Non-precision last child is malformed

- **WHEN** a `t-tensor` ends in a dimension rather than a precision
- **THEN** type resolution rejects it because the last child must be a precision

### Requirement: Exhaustive pattern matching

The checker SHALL verify that `match` expressions cover every variant of the scrutinee's ADT;
a missing variant SHALL be a type error, not a warning. A top-level irrefutable arm (bare
variable or as-pattern with irrefutable inner) SHALL cover the match, but a variable nested
inside a constructor/record pattern SHALL NOT.

#### Scenario: Exhaustive match type-checks

- **WHEN** a match over a 2-variant ADT covers both variants
- **THEN** the checker accepts it

#### Scenario: Missing variant is a type error

- **WHEN** a match omits a variant and has no irrefutable arm
- **THEN** the checker reports a non-exhaustive-match type error

### Requirement: Opaque type module identity

Outside its defining module, each of construction, constructor reference, pattern inspection,
field access, record-update, and casting into/out of an `opaque` type SHALL be an
`OpaqueTypeViolation`; construction SHALL be allowed only inside the defining module. A
named-module wrapper SHALL be required, re-opening a module SHALL be `DuplicateModule`, and a
hand-authored name matching the reserved linker format SHALL be `ReservedLinkerName`.

#### Scenario: In-module construction allowed

- **WHEN** code inside `stats.prob` constructs a `Probability`
- **THEN** the checker accepts it so smart constructors can be defined

#### Scenario: Out-of-module construction rejected

- **WHEN** code outside `stats.prob` constructs, casts to, or field-accesses `Probability`
- **THEN** the checker reports `OpaqueTypeViolation` naming the type, defining module, and exported producers

#### Scenario: Reserved linker name in hand-authored Deep rejected

- **WHEN** a non-linker program declares a top-level name matching `Pkg__<pkg>__<Module>__<Name>`
- **THEN** the checker rejects it as `ReservedLinkerName`

### Requirement: Invariant declaration well-formedness

An `invariant` metadata key SHALL require `opaque: true`; the representation SHALL be one
record-shaped variant whose fields are all in the invariant value class; the predicate SHALL be
boolean-shaped and drawn from the whitelisted grammar; and `invariant_amenability` SHALL be
recomputed and re-verified by the checker. The checker SHALL never evaluate the invariant.

#### Scenario: Well-formed invariant is recorded

- **WHEN** an opaque `deftype` declares `@invariant(p) (p.value >= 0.0) && (p.value <= 1.0)` over an f32 field
- **THEN** the checker records it and re-verifies the amenability without evaluating the predicate

#### Scenario: Invariant over a non-value-class field is rejected

- **WHEN** an invariant references a `string` field or a symbolic-dim tensor field
- **THEN** the checker reports an `OpaqueTypeViolation` naming the field

### Requirement: Hindley-Milner inference

The checker SHALL run Algorithm W with tensor extensions: constraint generation, unification
(including dimension unification), `let`-boundary generalization, and annotation checking
against provided `type` metadata. `if` branches SHALL share one type, and a lambda SHALL
receive a flat multi-argument function type.

#### Scenario: Let generalization produces polymorphism

- **WHEN** a `let`-bound value has an unconstrained type variable
- **THEN** it is generalized so each use instantiates a fresh variable

#### Scenario: Mismatched if branches are a type error

- **WHEN** the two branches of an `if` have different types
- **THEN** unification fails with a type error

### Requirement: Named dimension matching and no broadcasting

Tensor operations SHALL require strict dimension matching: two `d-name` unify only when equal,
two `d-lit` only when equal, a `d-var` unifies with any dimension, and a `d-name` unifies with
a `d-lit` (Name↔Lit, chelis#219). Chelis SHALL NOT support implicit broadcasting; rank and
dimension changes SHALL be explicit via `expand`/`reshape`/`permute`.

#### Scenario: Matching named dimensions unify

- **WHEN** `add` is applied to two `tensor[batch, hidden, f32]` operands
- **THEN** the dimensions unify and the result is `tensor[batch, hidden, f32]`

#### Scenario: Mismatched dimensions without expand is a type error

- **WHEN** `tensor[batch, hidden, f32]` is added to `tensor[hidden, f32]` without an explicit `expand`
- **THEN** it is a type error because there is no implicit broadcasting

### Requirement: Dimension polymorphism rigidity

Declared dimension parameters SHALL be rigid within the def body: two distinct declared dim
parameters SHALL NOT unify during body validation. A return-only dim parameter SHALL be
output-inferred, and coupling it to an input dimension (input-coupled pin or collapse) SHALL
be a `DimensionMismatch`.

#### Scenario: Call-site instantiation binds dim variables

- **WHEN** `transpose[a, b]` is called on `tensor[batch, seq, f32]`
- **THEN** `a := batch` and `b := seq` and the result is `tensor[seq, batch, f32]`

#### Scenario: Rigid dim collapse in the body is a type error

- **WHEN** a def `f[n, m](x: tensor[n, f32]) -> tensor[m, f32] = x` couples `m` to `n`
- **THEN** it is a `DimensionMismatch` because `n` and `m` are distinct rigid parameters

### Requirement: Wildcard dimension and rank-uniform lists

The wildcard dimension `(d-name {} *)` SHALL unify with any dimension but SHALL NOT be
generalized. A `List[tensor[k, f32]]` SHALL require rank-uniform elements; a list literal whose
elements have different ranks SHALL be a `DimensionMismatch`. A join-origin wildcard element
SHALL NOT satisfy a declared element type that names a rigid or named dimension.

#### Scenario: Ragged concat produces a wildcard axis

- **WHEN** two tensors with genuinely mismatched concrete axes are listed for `concat`
- **THEN** that axis widens to `(d-name {} *)`

#### Scenario: Mixed-rank list elements are rejected

- **WHEN** a `List[tensor[k, f32]]` literal mixes a rank-1 and a rank-2 element
- **THEN** it is a `DimensionMismatch` naming the rank mismatch and pointing at §4.5.1

### Requirement: Name-preserving rank polymorphism

A rank variable `..r` SHALL bind to the named dims it covers, preserving per-axis names; a
named-axis reduction SHALL drop the named anchor and carry the surrounding spreads. A named
anchor SHALL occur exactly once in the operand, and a rank-poly def body SHALL be restricted to
name-trackable operations, rejecting positional shape-rewriters.

#### Scenario: Named-axis reduction at any rank

- **WHEN** `def reduce_seq(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)`
- **THEN** it reduces the named `seq` axis at any rank, computing the output shape symbolically

#### Scenario: Positional rewriter in a rank-poly body is rejected

- **WHEN** a `..r` body uses `permute` or `reshape`
- **THEN** the Body-Discipline check rejects it because positional rewriters are not name-trackable at symbolic rank

### Requirement: Runtime shape semantics

`shape(x, axis)` SHALL require a concrete non-negative integer axis and return an `int32`
runtime scalar; a negative or out-of-range axis SHALL be a `DimensionMismatch`. Every
well-typed `int32` expression SHALL be valid as an `expand` runtime size. Static positive
values MAY fold; other values SHALL be evaluated once at runtime and remain live through
lowering and optimization. A non-positive value SHALL fail before allocation or element
access.

#### Scenario: Shape-sourced expand preserves the symbolic dim

- **WHEN** `expand(b, 0, shape(x, cast(0, int32)))` is used with a declared `tensor[n, 4, f32]` return
- **THEN** the symbolic batch dim `n` is preserved through unification

#### Scenario: Parameter-derived expand size is materialized

- **WHEN** an `expand` size is a bare runtime scalar parameter `a_dim: int32`
- **THEN** every execution lane evaluates `a_dim` and uses that value as the output extent

### Requirement: No implicit precision promotion

All operands of an arithmetic operation SHALL have the same precision; mixed precision SHALL be
a type error with a cast suggestion. `cast` SHALL be the only precision change and SHALL always
be explicit. Integer literals SHALL default to `int32` and float literals to `f32`, overridable
only by suffix, contextual element type, or explicit `cast`.

#### Scenario: Same-precision arithmetic type-checks

- **WHEN** two `f32` tensors are added
- **THEN** the operation type-checks with an f32 result

#### Scenario: Mixed-precision arithmetic is a type error

- **WHEN** an f32 tensor is added to a bf16 tensor
- **THEN** it is a type error whose diagnostic suggests inserting a `cast`

### Requirement: Precision compatibility

Operations SHALL accept same-precision operands only per the compatibility table: float
division (`div`) and transcendentals SHALL be float-only; `trunc_div` SHALL be integer-only;
logical ops SHALL be bool-only. Calling a float-only op on an integer tensor SHALL be a
call-site type error citing `floor_div`/`trunc_div` (for `div`) or the float-only rule.

#### Scenario: Float transcendental type-checks

- **WHEN** `exp` is applied to an f32 tensor
- **THEN** it type-checks

#### Scenario: Transcendental on an integer tensor is a type error

- **WHEN** `log` is applied to an int32 tensor
- **THEN** it is a call-site type error because transcendentals are float-only

### Requirement: Contextual tensor-literal inference

In a known-element-type position (typed binding RHS, matching call argument, typed tensor
return body, or `cast(literal, p)`), unsuffixed numeric literals SHALL adopt that element type,
binding directly at `p` for `cast` rather than narrowing-then-widening. Outside that closed set
they SHALL fall back to the `int32`/`f32` defaults; a suffix disagreeing with the context SHALL
be a type error.

#### Scenario: cast binds the literal directly at the target

- **WHEN** a literal is `cast(1.1, f64)`
- **THEN** it binds `1.1` at f64 (`0x3ff199999999999a`), not the narrow-then-widen f32 value

#### Scenario: Disagreeing suffix in context is a type error

- **WHEN** an f32-context tensor literal is `[1.0, 2.0f64, 3.0]`
- **THEN** it is a type error naming the offending index

### Requirement: Mixed-precision accumulator

`matmul` and `reduce_sum` SHALL carry an optional accumulator-precision parameter that is the
only mixed-precision mechanism; when omitted the compiler SHALL resolve the documented default
(bf16/f16 → f32, int8/int16 → int32) before any backend is invoked. An accumulator narrower
than the operand or the default SHALL be a type error; integer `matmul` operand precisions
SHALL NOT be admitted.

#### Scenario: bf16 matmul uses an f32 accumulator by default

- **WHEN** `matmul` is applied to two bf16 tensors with no accumulator parameter
- **THEN** the IR node carries an f32 accumulator and the result is downcast to bf16

#### Scenario: Narrower accumulator is a type error

- **WHEN** `reduce_sum(x: tensor[N, int8], accumulator=int8)` requests a narrower-than-default accumulator
- **THEN** it is a type error suggesting the wider int32 default

### Requirement: Fitness scoring

Every compilation attempt SHALL produce a fitness report scoring parse (0.1), structural
validity (0.1), name resolution (0.2), and type check (0.6), summing to 0.0–1.0. When full type
checking fails the checker SHALL still infer as many sub-expressions as possible and annotate
failed nodes with error metadata plus structured repair suggestions.

#### Scenario: Partial inference annotates failed nodes

- **WHEN** a precision mismatch prevents full type checking
- **THEN** the report still annotates typed nodes and marks the failed node with error metadata and a cast suggestion

#### Scenario: Perfect success implies an empty error list

- **WHEN** the fitness score is 1.0
- **THEN** the error list is empty (a report cannot claim perfect success with errors present)

### Requirement: Effects

The checker SHALL infer a function's effect set as the union of its body's compiler-known
effects, with `Random` (from `dropout`/tensor RNG), `IO` (from `print`/`debug`/file
builtins/`process_run`), and `Resource(Device)` as boundary effects. `with seed(seed)` SHALL
handle `Random` and require an `i64`-suffixed seed; an unhandled top-level `Random` SHALL be a
check error with repair guidance while top-level `IO` SHALL be permitted.

#### Scenario: Seeded region handles Random

- **WHEN** `with seed(42i64) { dropout(x, 0.5) }` wraps the random op
- **THEN** the `Random` effect is handled and the enclosing signature is enforced on the body

#### Scenario: Unhandled top-level Random is a check error

- **WHEN** a top-level program uses `dropout` with no `with seed` handler
- **THEN** the checker reports an unhandled-`Random` error with repair guidance

### Requirement: Linearity and auto-borrow

Tensor values SHALL be owned by default with read-only calls borrowing their arguments; passing
owned `T` where `&T` is expected SHALL auto-borrow, and passing `&T` where owned `T` is expected
SHALL be a type error unless `copy(x)` is written. Borrows SHALL NOT be stored in aggregates,
returned, or captured; a borrow inner SHALL resolve to a tensor or tensor-carrying value.

#### Scenario: Owned value auto-borrows into a read-only call

- **WHEN** an owned tensor is passed where `&tensor[...]` is expected
- **THEN** it auto-borrows without an explicit `&`

#### Scenario: Borrowing a non-tensor value is a type error

- **WHEN** `&x` is applied to a scalar or non-tensor-carrying value that never resolves to a tensor
- **THEN** it is a type error, including after postponed re-check when the variable is never pinned to a tensor

### Requirement: Type-name uniqueness and builtin shadowing

Every `deftype`/`typealias` name SHALL be unique in a checked program (flat namespace),
rejecting collisions as `DuplicateDefinition`, including re-declaring a prelude type. A
top-level `def`/`sig` whose name is in the closed builtin vocabulary SHALL be rejected as
`BuiltinShadowing` before inference, in every lane that runs the checker.

#### Scenario: Duplicate type name rejected

- **WHEN** a program declares `deftype Foo` twice or `deftype Foo` plus `typealias Foo`
- **THEN** it is rejected as `DuplicateDefinition`

#### Scenario: Shadowing a builtin name rejected

- **WHEN** a top-level `def sum(...)` shadows the builtin `sum`
- **THEN** the checker rejects it as `BuiltinShadowing` before inference

### Requirement: Numeric value semantics (decided)

Every numeric op result SHALL be finalized into its declared dtype before becoming observable:
float finalization SHALL be IEEE-754 round-to-nearest-ties-to-even at the dtype width; integer
results not representable in the declared width SHALL trap rather than wrap/saturate/widen
(except named modular ops); a `bool` SHALL be exactly 0 or 1; and comparisons SHALL compare
finalized values. Named `wrap_*` ops SHALL be the explicit modular escape hatch, defined only
on integer dtypes.

#### Scenario: Correctly-rounded float result is authoritative

- **WHEN** an f64 computes `add(2^53, 1)`
- **THEN** the correctly-rounded `2^53` result is correct and SHALL NOT be "fixed"

#### Scenario: Integer overflow traps

- **WHEN** ordinary integer arithmetic produces a value outside the declared width
- **THEN** it traps with the branded overflow diagnostic rather than wrapping, and a `wrap_*` op requested on a float dtype is a type error

### Requirement: Checker totality (decided)

Every Deep tag in the closed vocabulary SHALL have an explicit checker disposition (a real
inference case or a rejection with a pushed diagnostic); an unrecognized construct SHALL
produce a diagnostic, never a silent subtree exemption. If a check completes with an empty
error vector, the typed result SHALL contain no error-typed expression, and a structurally
malformed form SHALL be rejected naming the tag and expected shape.

#### Scenario: Unknown form pushes a diagnostic

- **WHEN** the checker encounters a construct it does not recognize
- **THEN** it pushes an `UnknownForm` diagnostic rather than silently exempting the subtree

#### Scenario: Empty error vector implies no error-typed nodes

- **WHEN** a check completes with an empty error vector
- **THEN** the typed result contains no `Type::Error` node, and a malformed `(def)` with no body is rejected as `MalformedForm`
