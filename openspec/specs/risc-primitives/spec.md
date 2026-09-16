# risc-primitives

## Purpose

Define the irreducible RISC tensor primitive set and its semantics: the RISC philosophy and
no-broadcasting rule, primitives-as-functions and borrow-typed inputs, the Tier-1 primitives
with their AD adjoints, the Tier-2 derived built-ins, division and reduction semantics,
windowed reductions, movement and memory ops, effectful primitives with seed determinism,
scatter determinism and AD policy, host-only builtins, the standard ML-op lowerings, AD
completeness and the reference oracle, and the decided unsupported-case and observation
contracts.

**Source:** captured from [`spec/05-risc-primitives.md`](../../../spec/05-risc-primitives.md).

## Requirements

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
adjoint or rejection rule. Tier-2 derived built-ins SHALL be convenience functions the desugarer
emits and the IR lowers to semantics-preserving Tier-1 compositions during IR construction; they
SHALL exist in Deep AST only, not in the RISC DAG. `sub`, `max_elem`, and `min_elem` are Tier-1
identities because an arithmetic surrogate can introduce a trap or alter selected stored bits.

#### Scenario: Subtraction stays direct

- **WHEN** `sub(a, b)` reaches RISC IR
- **THEN** it remains direct checked subtraction rather than becoming `add(a, neg(b))`

#### Scenario: Minimum stays a selection

- **WHEN** `min_elem(a, b)` reaches RISC IR
- **THEN** it remains direct selection rather than becoming `neg(max_elem(neg(a), neg(b)))`

### Requirement: Direct subtraction and extrema semantics

Checked signed-integer subtraction SHALL trap only when its exact mathematical result is
unrepresentable at the operand width. Float subtraction SHALL execute at the declared arithmetic
width. Float extrema SHALL select the first NaN in operand order with exact stored bits and SHALL
otherwise preserve the first operand on every equality, including signed-zero equality. Integer
extrema SHALL compare exactly at the stored width and use the same first-operand tie rule.

#### Scenario: Representable subtraction cannot trap at an invented negation

- **WHEN** `sub(-1i64, INT64_MIN)` executes
- **THEN** it returns `INT64_MAX` without evaluating `neg(INT64_MIN)`

#### Scenario: Extrema preserve the selected stored value

- **WHEN** two float extrema operands include a NaN or compare equal
- **THEN** the first selected operand is returned bit-for-bit and receives the whole cotangent

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

A reduction SHALL take one or more unique compile-time positional i32 axes
or one or more unique named axes, never a mixture. Negative positional axes
SHALL normalize once against original rank. Value reductions execute the
highest-original-position-first single-axis composition; `count` executes one
dedicated multi-axis bool reduction. `sum` SHALL carry a populated
accumulator-precision field resolved to the documented default when omitted;
an explicitly narrower-than-default accumulator SHALL be a type error.

#### Scenario: Constant axis reduction

- **WHEN** `sum(x, 1)` reduces axis 1
- **THEN** the output removes the dimension at position 1

#### Scenario: Runtime axis is rejected

- **WHEN** a reduction axis is a runtime `i32` parameter
- **THEN** it is rejected at the reduction call site naming the compile-time-constant requirement

### Requirement: Windowed reductions

`reduce_window_max/min/sum/mean` SHALL accept runtime `List[i64]`
`window_shape` and `strides`, validate lengths/positive values before access,
and implement the exact target-independent output-shape, arithmetic,
tie/NaN, accumulation, adjoint, and second-derivative graph of
[05-RWIN-1..2]/[05-OP-39]. Statically proven invalid inputs are type errors;
runtime invalid inputs trap `Domain` before allocation or reads.

#### Scenario: Valid-padding output extent

- **WHEN** `reduce_window_max` runs with input 8, window 2, stride 2 on an axis
- **THEN** the output extent is 4 (`floor((8-2)/2)+1`)

#### Scenario: Device targets execute windowed reduction

- **WHEN** `chelis build --target hip` compiles a program using `reduce_window_*`
- **THEN** it executes the same runtime window values, results, traps, and adjoints as eval and C

### Requirement: Movement ops and runtime bounds

`reshape`, `permute`, `expand`, `pad`, `shrink`, and `stride` SHALL each carry
their exact AD adjoint. Runtime movement bounds and reshape targets SHALL be
validated in every execution mode with matching language traps for negative
bounds, range overshoot, non-positive stride, and reshape numel disagreement.
Stride reverse mode SHALL zero-fill the original shape and route each output
cotangent to its unique forward-selected source index; runtime steps carry
zero cotangent.

#### Scenario: Movement adjoint is defined

- **WHEN** `grad` differentiates through `permute(x, axes)`
- **THEN** the adjoint is `permute(g, inverse_permutation)`

#### Scenario: Runtime reshape numel mismatch aborts

- **WHEN** a runtime reshape target's element product disagrees with the input
- **THEN** every execution mode traps before allocating a wrong view

### Requirement: Memory and shape-query primitives

`const` and `load` SHALL be the pure tensor constructors and contribute zero
cotangent. `shape(x, axis)` SHALL read the runtime extent along any literal or
computed i32 axis as a rank-0 i64 scalar contributing a zero
cotangent. Literal and computed axes SHALL remain ordinary checked runtime
values when `shape` participates in a graph constructed by `grad`.

#### Scenario: const is non-differentiable

- **WHEN** `grad` reaches a `const` node
- **THEN** its gradient contribution is zero and it does not block AD

#### Scenario: Computed shape axis under grad remains a runtime value

- **WHEN** `grad` constructs `shape(x, axis)` with a computed runtime axis
- **THEN** it preserves that axis, executes one-step normalization, and contributes exact zero cotangent

### Requirement: Effectful primitives

`dropout` and `uniform_like` SHALL introduce the `Random` effect drawing from the active
`with seed(...)` handler. `process_run` SHALL introduce `IO`, pass its argv straight to the OS
with no shell or interpolation, and report a signal-killed process as exit code `-1`.
Every legal host execution mode SHALL provide the same typed result/trap; a
device-only kernel cannot perform IO but that boundary SHALL NOT become a
whole-module or language-wide rejection.

#### Scenario: dropout introduces Random under a seed

- **WHEN** `dropout(x, rate)` runs inside `with seed(42i64)`
- **THEN** it draws its mask from the handled seed and reuses it on the backward pass

#### Scenario: process_run compiles as a host effect

- **WHEN** a program applying `process_run` is compiled with `--target c`
- **THEN** the host execution performs the exact argv call and returns `(i64,string,string)!{IO}`

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
AD via `AdError::NotSupported`; `ScatterAdd` SHALL have the `Gather` adjoint. `Gather`,
`ScatterAdd`, `Scatter`, and `ScatterElements`, including [05-OP-33]'s public C gather and scatter
callables, SHALL admit an index tensor at every active signed-integer dtype, interpreted at its
exact stored width without conversion.

#### Scenario: ScatterAdd is differentiable

- **WHEN** `grad` differentiates through `ScatterAdd`
- **THEN** the adjoint is `Gather` and duplicate indices fan out correctly

#### Scenario: Scatter AD is rejected

- **WHEN** `grad` is applied through `Scatter`
- **THEN** it is rejected with `AdError::NotSupported` because the forward result depends on iteration order at duplicate indices

#### Scenario: Narrow signed indices remain exact

- **WHEN** gather or scatter consumes an i8 or i16 index tensor
- **THEN** it interprets each stored index exactly rather than rejecting or widening the tensor

### Requirement: Host-executed builtins

`tensor_scan` and the `test_*` assertion family SHALL execute on the host in
evaluation and compiled artifacts with [05-HOST-1..3]/[05-OP-38]'s exact
signatures, effects, recurrence, equality/closeness, and AD/vmap rules. An
unreachable call SHALL neither emit a stub nor cause whole-module rejection.

#### Scenario: tensor_scan runs under eval

- **WHEN** `tensor_scan` builds a rank-1 tensor under `chelis eval`
- **THEN** it iterates on the host with constant worker-stack usage

#### Scenario: tensor_scan builds as a host operation

- **WHEN** a program calling `tensor_scan` (even in an entry-unreachable helper) is built with `--target c`
- **THEN** the reachable host call executes its exact recurrence; an unreachable helper has no effect on the artifact

### Requirement: Standard lowerings

`matmul`, `softmax`, `cross_entropy`, `layer_norm`, `conv`, `embedding`, and
`multi_head_attention` SHALL lower to defined Tier-1 compositions; `matmul` SHALL carry an
accumulator parameter with the documented defaults and SHALL NOT admit integer operand
precisions. `matmul`'s inner sum SHALL follow the canonical balanced tree, so its result
bits are target-independent; a library kernel or fusion is permissible only where it
reproduces those exact bits and traps, and a vendor-kernel accumulation order is available
only through a future named explicit opt-in, never a backend default. `relu` SHALL carry
[05-OP-43]'s dedicated adjoint (gradient exactly zero at zero, both signed zeros, and NaN)
while its forward value remains the exact `max_elem` lowering.

#### Scenario: softmax lowers to a stable composition

- **WHEN** `softmax(x, axis)` is lowered
- **THEN** it desugars through max-subtraction, `exp`, `sum`, and `div` for numerical stability

#### Scenario: Integer matmul is not admitted

- **WHEN** `matmul` is applied to integer operands
- **THEN** it is a type error because the active matmul signature does not admit integer precisions

### Requirement: AD completeness and reference oracle

Every numeric callable SHALL state exactly one of an adjoint, a zero cotangent, or a
structural `grad` rejection; `cmplt`, `const`, and `load` contribute zero. [05-OP-42]
`stop_gradient` SHALL be the differentiation barrier whose argument subgraph is outside
adjoint construction and structural rejection analysis. The pseudocode reference
implementations are illustrative, not a semantic oracle; cross-lane agreement is exact by
default, with only [05-OBS-3]'s per-operation tolerance table excepted, and blanket
per-dtype bounds are not a conforming oracle.

#### Scenario: Composition is differentiable

- **WHEN** `grad` is applied to a composition of primitives with defined adjoints
- **THEN** it produces a gradient through the whole composition

#### Scenario: Cross-lane agreement is exact outside the tolerance table

- **WHEN** two lanes evaluate an operation absent from [05-OBS-3]'s table at the same
  arithmetic width
- **THEN** their stored result bits agree exactly; a blanket 1e-6/1e-12 bound is not a
  conforming comparison

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
