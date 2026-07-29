## ADDED Requirements

### Requirement: RISC philosophy and no broadcasting

All tensor computation SHALL decompose into a small irreducible primitive set, with complex
behavior emerging from composition plus compiler optimization rather than a large operator
vocabulary. Operands SHALL have matching dimensions with no implicit rank extension or size
expansion; rank/size changes SHALL be explicit via `expand`.

#### Scenario: Composition expresses a complex op

- **WHEN** a standard ML operation is needed
- **THEN** it decomposes into a composition of the RISC primitives rather than a dedicated core op

#### Scenario: Broadcasting is a hard error

- **WHEN** two operands of mismatched dimension are combined without an explicit `expand`
- **THEN** it is a type error because there is no implicit broadcasting

### Requirement: Primitives are functions, not tags

RISC primitives SHALL be built-in functions in the compiler's scope, referenced via
`(var {} name)` and called via `(app {} ...)`, not syntax tags. Read-only tensor primitive
parameters SHALL be typed `&tensor[...]` with owned arguments auto-borrowing; outputs SHALL
remain owned tensors and the borrow distinction SHALL be erased before IR lowering.

#### Scenario: Primitive referenced as a var

- **WHEN** `add` is applied
- **THEN** it is `(app {} (var {} add) ...)`, not an `add` tag

#### Scenario: Owned tensor auto-borrows into a primitive

- **WHEN** an owned tensor is passed to a read-only primitive parameter
- **THEN** it auto-borrows without an explicit `&`, and `len(&xs)` is not a supported surface form

### Requirement: Two tiers

Tier-1 RISC primitives SHALL be the irreducible set the IR operates on, each with a defined AD
adjoint rule. Tier-2 derived built-ins SHALL be convenience functions the desugarer emits and
the IR lowers to Tier-1 compositions during IR construction; they SHALL exist in Deep AST only,
not in the RISC DAG.

#### Scenario: Derived builtin lowers to primitives

- **WHEN** `sub(a, b)` is lowered
- **THEN** it becomes `add(a, neg(b))` during IR construction

#### Scenario: Derived builtin is not in the RISC DAG

- **WHEN** the IR DAG is inspected after lowering
- **THEN** it contains only Tier-1 primitives, with `sub` decomposed rather than present as a node

### Requirement: Division semantics

`div` SHALL be restricted to float operands with IEEE-754 semantics; `div` (or `/`) on integer
operands SHALL be a type error citing `floor_div`/`trunc_div`. `floor_div` SHALL round toward
−∞ on integer and float operands; `trunc_div` SHALL round toward zero on integer operands only.
An integer zero divisor SHALL trap deterministically on every target via an explicit guard.

#### Scenario: Float division type-checks

- **WHEN** `div` is applied to two f32 tensors
- **THEN** it lowers to native floating `/` with IEEE-754 semantics

#### Scenario: Integer division with div is a type error

- **WHEN** `div` is applied to two integer tensors
- **THEN** it is a type error citing §2.1 and pointing at `floor_div`/`trunc_div`

#### Scenario: Integer zero divisor traps

- **WHEN** `trunc_div`, `floor_div`, or `mod` divides by an integer zero
- **THEN** it traps with the "integer division or remainder by zero" diagnostic on both eval and C, via the explicit guard rather than a hardware fault

### Requirement: Elementwise unary precision and adjoints

Elementwise unary transcendentals (`exp`, `log`, `sin`, `cos`, `tan`, `atan`, `sqrt`, `recip`)
SHALL be valid on float types only, and each SHALL carry its defined AD adjoint. `floor`,
`ceil`, and `round` SHALL be non-differentiable and `grad` SHALL reject them with a
`PiecewiseConstant` error rather than returning a zero gradient.

#### Scenario: exp adjoint differentiates

- **WHEN** `grad` differentiates through `exp(x)`
- **THEN** the adjoint is `g * exp(x)`

#### Scenario: floor is rejected by grad

- **WHEN** `grad` is applied to a function containing `floor`
- **THEN** it is rejected with an `AdRejectionReason::PiecewiseConstant` error, not a silent zero gradient

### Requirement: Reduction axis and accumulator

A reduction axis SHALL be a compile-time constant (literal or `cast(N, int32)`); a runtime axis
SHALL be rejected at the call site. Negative axes SHALL index from the end uniformly across
axis-taking primitives. `sum` SHALL carry a populated accumulator-precision field resolved to
the documented default when omitted; an explicitly narrower-than-default accumulator SHALL be a
type error.

#### Scenario: Constant axis reduction

- **WHEN** `sum(x, 1)` reduces axis 1
- **THEN** the output removes the dimension at position 1

#### Scenario: Runtime axis is rejected

- **WHEN** a reduction axis is a runtime `int32` parameter
- **THEN** it is rejected at the reduction call site naming the compile-time-constant requirement

### Requirement: Windowed reductions

`reduce_window_max/min/sum/mean` SHALL implement `Valid`-padding-only strided windowed
reductions over the trailing axes, with output extent
`floor((input_dim - window) / stride) + 1`; a non-positive output extent SHALL be a type error.
They SHALL be differentiable via `ReduceWindowGrad`; HIP codegen SHALL be deferred and rejected
before codegen with an `unsupported_feature` diagnostic.

#### Scenario: Valid-padding output extent

- **WHEN** `reduce_window_max` runs with input 8, window 2, stride 2 on an axis
- **THEN** the output extent is 4 (`floor((8-2)/2)+1`)

#### Scenario: HIP target rejects windowed reduction

- **WHEN** `chelis build --target hip` compiles a program using `reduce_window_*`
- **THEN** it is rejected at compile time with a clean `unsupported_feature` error

### Requirement: Movement ops and runtime bounds

`reshape`, `permute`, `expand`, `pad`, `shrink`, and `stride` SHALL each carry their defined AD
adjoint. Runtime (node-valued) movement bounds and reshape targets SHALL be validated at run
time in both the eval and C lanes with matching abort/error paths for negative bounds, range
overshoot, non-positive stride, and reshape numel disagreement; the HIP and Metal targets SHALL
reject them naming `--target c`.

#### Scenario: Movement adjoint is defined

- **WHEN** `grad` differentiates through `permute(x, axes)`
- **THEN** the adjoint is `permute(g, inverse_permutation)`

#### Scenario: Runtime reshape numel mismatch aborts

- **WHEN** a runtime reshape target's element product disagrees with the input
- **THEN** both the eval lane and the C backend's emitted guard abort loudly rather than allocate a wrong view

### Requirement: Memory and shape-query primitives

`const` and `load` SHALL be the pure tensor constructors and SHALL be non-differentiable
(`const` gradient zero, `load` non-differentiable). `shape(x, axis)` SHALL read the runtime
extent along a compile-time-constant axis as a rank-0 integer scalar contributing a zero
cotangent; a non-constant axis forced into DAG construction (e.g. via `grad`) SHALL fail loudly.

#### Scenario: const is non-differentiable

- **WHEN** `grad` reaches a `const` node
- **THEN** its gradient contribution is zero and it does not block AD

#### Scenario: Non-constant shape axis under grad fails loudly

- **WHEN** `grad` forces DAG construction of `shape(x, axis)` with a runtime axis
- **THEN** it fails with a clean source-located diagnostic requiring a compile-time-constant axis, not a fabricated gradient

### Requirement: Effectful primitives

`dropout` and `uniform_like` SHALL introduce the `Random` effect drawing from the active
`with seed(...)` handler. `process_run` SHALL introduce `IO`, pass its argv straight to the OS
with no shell or interpolation, report a signal-killed process as exit code `-1`, and be
eval/test-only — rejected by the C/HIP/Metal build backends with a clean diagnostic.

#### Scenario: dropout introduces Random under a seed

- **WHEN** `dropout(x, rate)` runs inside `with seed(42i64)`
- **THEN** it draws its mask from the handled seed and reuses it on the backward pass

#### Scenario: process_run rejected by a build backend

- **WHEN** a program applying `process_run` is compiled with `--target c`
- **THEN** it is rejected with a clean build error rather than a silent zero, because a compiled artifact has no host interpreter

### Requirement: Seed determinism

For a fixed compiler version and target, evaluating a `with seed(N)` program twice SHALL yield
byte-identical output, and two distinct accepted seeds SHALL yield distinct streams, in every
lane. The RNG SHALL NOT be cryptographic; streams are decorrelated only up to the SplitMix64
mixing.

#### Scenario: Same seed reproduces output

- **WHEN** a `with seed(7i64)` program is evaluated twice on the same compiler and target
- **THEN** the two outputs are byte-identical

#### Scenario: Distinct seeds diverge

- **WHEN** two accepted seeds are used for the same program
- **THEN** they yield distinct streams

### Requirement: Scatter determinism and AD policy

`Scatter` (last-write-wins) and `ScatterAdd` (commutative accumulation) SHALL be distinct
primitives. `Scatter` SHALL resolve duplicate indices by updates-tensor row-major flat order on
every backend (single-threaded on C, `<<<1,1>>>` on HIP) and SHALL structurally reject reverse-mode
AD via `AdError::NotSupported`; `ScatterAdd` SHALL have the `Gather` adjoint.

#### Scenario: ScatterAdd is differentiable

- **WHEN** `grad` differentiates through `ScatterAdd`
- **THEN** the adjoint is `Gather` and duplicate indices fan out correctly

#### Scenario: Scatter AD is rejected

- **WHEN** `grad` is applied through `Scatter`
- **THEN** it is rejected with `AdError::NotSupported` because the forward result depends on iteration order at duplicate indices

### Requirement: Host-only builtins

`tensor_scan` and the `test_*` assertion family SHALL be host-only, running inside the
`chelis test`/`chelis eval` interpreter with no compiled-lane emission. `tensor_scan` SHALL be
rejected whole-program at `chelis build --target c`/`hip`, and `grad`/`vmap` over a function
reaching it SHALL be rejected at the transform boundary (reachability-scoped).

#### Scenario: tensor_scan runs under eval

- **WHEN** `tensor_scan` builds a rank-1 tensor under `chelis eval`
- **THEN** it iterates on the host with constant worker-stack usage

#### Scenario: tensor_scan build is rejected

- **WHEN** a program calling `tensor_scan` (even in an entry-unreachable helper) is built with `--target c`
- **THEN** it is rejected at compile time with a `tensor_scan`-tagged `unsupported_feature` diagnostic

### Requirement: Standard lowerings

`matmul`, `softmax`, `cross_entropy`, `layer_norm`, `conv2d`, `embedding`, and
`multi_head_attention` SHALL lower to defined Tier-1 compositions; `matmul` SHALL carry an
accumulator parameter with the documented defaults and SHALL NOT admit integer operand
precisions. The compiler MAY recognize these patterns and emit optimized library calls.

#### Scenario: softmax lowers to a stable composition

- **WHEN** `softmax(x, axis)` is lowered
- **THEN** it desugars through max-subtraction, `exp`, `sum`, and `div` for numerical stability

#### Scenario: Integer matmul is not admitted

- **WHEN** `matmul` is applied to integer operands
- **THEN** it is a type error because the active matmul signature does not admit integer precisions

### Requirement: AD completeness and reference oracle

Every RISC primitive SHALL have a defined adjoint so `grad` can differentiate any composition;
`cmplt`, `const`, and `load` SHALL have zero gradient. The naive C reference implementations
SHALL be the correctness oracle, and GPU backends SHALL produce numerically identical results
within floating-point tolerance (1e-6 for f32, 1e-12 for f64).

#### Scenario: Composition is differentiable

- **WHEN** `grad` is applied to a composition of primitives with defined adjoints
- **THEN** it produces a gradient through the whole composition

#### Scenario: GPU matches the reference within tolerance

- **WHEN** a GPU backend evaluates a primitive against the C reference
- **THEN** the results agree within 1e-6 (f32) / 1e-12 (f64)

### Requirement: Unsupported-case response contract

When any stage encounters a case it does not support (op, builtin, dtype, kernel, construct, or
parameter shape) it SHALL respond with a diagnostic through its failure channel and SHALL NOT
substitute a value, type, dtype, kernel, or emission. The diagnostic SHALL surface at the
earliest competent stage, be branded `unsupported:`, and name what was encountered and where. A
panic reachable from source input SHALL be a defect.

#### Scenario: Unsupported dtype fails loudly

- **WHEN** a stage encounters an unsupported dtype or op
- **THEN** it emits an `unsupported:`-branded diagnostic naming the case and stage rather than substituting a value

#### Scenario: Gate is not the sole defense

- **WHEN** a pre-codegen gate for an unsupported case is missing or permissive
- **THEN** the emitter channel still rejects it, so no gate/emitter gap can ship a silent wrong binary

### Requirement: Observation and formatting contract

Every exit that renders a stored numeric value as text SHALL emit text that parses back to
exactly the stored bits at the value's own dtype width, with all exits within a lane agreeing.
Integers SHALL print exact digits, floats the shortest round-tripping string, `bool` as
`true`/`false`; a scalar SHALL render bare (never wrapped as `tensor(shape=[], ...)`); and
tensor rendering SHALL truncate after 32 elements with `, ...` (except `to_list`/wire schema).

#### Scenario: Float round-trips at its width

- **WHEN** a stored f32 value is printed
- **THEN** the shortest string that round-trips at f32 width is emitted, agreeing with every other exit in the lane

#### Scenario: Scalar renders bare, not wrapped

- **WHEN** a scalar-typed root is rendered
- **THEN** it prints as the bare scalar (`root = 0.1`), never as `tensor(shape=[], data=[0.1])`
