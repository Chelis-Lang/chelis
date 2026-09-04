# transformations

## Purpose

Define the DAG-to-DAG transformations and optimization passes: reverse-mode automatic
differentiation (`grad`) including its algorithm, `wrt` selection, gradient payloads,
non-differentiable handling, symbolic-dimension adjoint construction, higher-order
derivatives, executed-branch `match`/`if` and recursive List/ADT gradients; vectorization (`vmap`);
just-in-time compilation (`jit`); the semantics-preserving DAG optimization passes; the
transformation composition, commutativity, and ordering rules; and the transform error
contract.

**Source:** captured from [`spec/06-transformations.md`](../../../spec/06-transformations.md).

## Requirements

### Requirement: Transformations are DAG-to-DAG functions

`grad`, `vmap`, and `jit` SHALL be functions from RISC DAGs to RISC DAGs, taking a function and
producing a new function. After `grad`/`vmap` expansion the DAG SHALL consist entirely of RISC
primitives with all transform constructs rewritten away.

#### Scenario: Transform produces a new DAG

- **WHEN** `grad(f)` is applied to a function DAG
- **THEN** it produces a new RISC DAG computing the gradient

#### Scenario: Expanded DAG has no transform nodes

- **WHEN** `grad`/`vmap` expansion completes
- **THEN** the resulting DAG contains only RISC primitives, no `grad`/`vmap` nodes

### Requirement: grad signature and gradient payload

`grad(f)` SHALL require a scalar floating output and SHALL return gradients only (not
`(value, grad)`). The gradient type SHALL mirror the argument structure: a single tensor yields
the same tensor type, a List preserves its runtime length/order, a tuple yields a tuple of
gradients, and an ADT yields the executed constructor shape with a gradient per differentiable
field and `unit` per discrete field. A multi-parameter function's payload SHALL be
a flat tuple in parameter order.

#### Scenario: Multi-parameter gradient is a flat tuple

- **WHEN** `grad(loss)` is applied to a two-tensor-parameter loss
- **THEN** the result is a flat tuple of the two parameter gradients in order

#### Scenario: Non-scalar output without a seed is rejected

- **WHEN** `grad(f)` is applied to a function returning a non-scalar tensor with no seed gradient
- **THEN** it is a `non_scalar_grad_output` error suggesting a reduction or a seed

### Requirement: wrt selection

By default `grad(f)` SHALL differentiate with respect to all differentiable parameters; the
optional `wrt` parameter SHALL restrict differentiation to the listed parameters, treating the
rest as constants. The gradient result SHALL contain entries only for the listed parameters, in
`wrt` order (one parameter returns a bare gradient, multiple return a flat tuple).

#### Scenario: wrt restricts the gradient set

- **WHEN** `grad(loss, wrt=w)` is applied to `loss(w, x, y)`
- **THEN** only `w`'s gradient is returned and `x`/`y` are treated as constants

#### Scenario: wrt on a non-differentiable parameter is a type error

- **WHEN** `wrt` names a bool or integer parameter
- **THEN** it is a `non_differentiable` error naming the parameter

### Requirement: Reverse-mode AD algorithm and accumulation

`grad` SHALL construct the backward DAG by initializing each value's recursive shape-preserving
cotangent to zero, seeding the scalar float output to exact one, traversing values in reverse
topological order under each op's exact adjoint/zero/rejection contract, and accumulating each
float leaf by the canonical adjacent-pair balanced tree over consumer-edge order. Consumer edges
SHALL be ordered lexicographically by the consumer's canonical forward node ordinal and then its
input-slot index, independently of a topological-sort implementation's tie order. The combined
forward+backward DAG SHALL remain acyclic.

#### Scenario: Multi-use node sums contributions

- **WHEN** a node is consumed by multiple downstream nodes
- **THEN** its adjoint is the sum of all consumer contributions

#### Scenario: Independent topological schedules preserve gradient bits

- **WHEN** two valid backward work lists choose different ties among unrelated forward nodes
- **THEN** both build the same canonical consumer-edge tree and produce identical gradient bits

#### Scenario: Backward DAG is acyclic

- **WHEN** the backward DAG references forward nodes in its adjoint rules
- **THEN** the combined forward+backward DAG remains acyclic because backward edges point to earlier forward nodes

### Requirement: Non-differentiable operation handling

Every numeric callable SHALL have exactly one exact adjoint, zero-cotangent,
or structural `grad` rejection contract. Comparisons and `shape` contribute
zero without blocking the graph. Piecewise-constant rounding and
float-to-integer/bool casts structurally reject with
`AdRejectionReason::PiecewiseConstant`; they never silently return zero.
Float-to-float casts use `cast(g, source_dtype)`. [05-OP-42]
`stop_gradient` SHALL be the differentiation barrier: its argument's
subgraph is outside adjoint construction and structural rejection analysis,
and the argument receives the shape-preserving exact zero cotangent.
[05-OP-43] `relu` SHALL retain its dedicated identity through AD and select
the complete incoming cotangent exactly where `0 < x`, producing exact
positive zero at both signed zeros and NaN. An
explicit `wrt` target
must contain a differentiable float leaf; mixed List/tuple/ADT targets are
legal and preserve discrete fields as `unit`.

#### Scenario: grad through internal extrema and comparison is valid

- **WHEN** `grad` is applied to a function using `MaxElem`/`MinElem`/`CmpLt`
  internally with float tensor parameters
- **THEN** it is valid: the selected operand receives the whole cotangent,
  including the first operand on equality, while the comparison contributes zero

#### Scenario: grad over a bool parameter is a type error

- **WHEN** `grad` is applied to a function whose `wrt` parameter is `bool`
- **THEN** it is a `non_differentiable` type error naming the parameter

#### Scenario: ReLU does not inherit the direct-maximum tie rule

- **WHEN** `grad` reaches `relu(x)` at `x = +0`, `x = -0`, or a NaN input
- **THEN** the input cotangent is exact positive zero, while a direct `max_elem` application keeps its first-operand-on-equality adjoint

#### Scenario: A structural rejection inside the barrier does not reject

- **WHEN** `grad` is applied to `add(x, stop_gradient(sub(round(x), x)))`
- **THEN** construction succeeds and the cotangent of `x` is exactly the
  identity path's, while bare `round` under `grad` still rejects with
  `AdRejectionReason::PiecewiseConstant`

### Requirement: Symbolic-dim adjoint construction

Adjoint construction SHALL preserve proven symbolic identities and otherwise
carry exact runtime int64 extent nodes. `ProdReduce` SHALL reverse its executed
balanced tree, runtime stride SHALL use its exact inverse sampling map, and
runtime-window reductions SHALL reverse the executed window graph. No runtime
axis, step, window, loop extent, or target shape is rejected merely because it
is unavailable at transform time. A scalar `shape()` read is AD-transparent
with a zero cotangent.

#### Scenario: Structural adjoint carries the symbolic dim

- **WHEN** a `Sum`/`Expand` adjoint is built over a symbolic axis
- **THEN** it carries the extent as a `DimExpr` without reading its runtime value

#### Scenario: Value-dependent symbolic adjoint executes exactly

- **WHEN** a `ProdReduce` adjoint or a runtime stride-step adjoint needs a value-dependent structure over a symbolic axis
- **THEN** it constructs the exact runtime reverse graph without guessing or target-specific rejection

### Requirement: Higher-order derivatives

`grad(grad(f))` SHALL compute second derivatives by applying `grad` to the backward DAG, which
is itself a valid RISC DAG. Shared nodes between the forward and backward DAG (e.g. `Exp(x)`
reused in its own adjoint) SHALL be preserved as graph references, not duplicated into a tree.

#### Scenario: Second derivative through a cubic

- **WHEN** `grad(grad(f))` is applied to `f(x) = x^3`
- **THEN** it computes `6*x`

#### Scenario: Shared nodes preserved under nested grad

- **WHEN** the second `grad` encounters a node shared by the forward and backward DAG
- **THEN** it handles the shared reference correctly rather than treating the DAG as a tree

### Requirement: Executed match and if differentiation

Inside a differentiated body, `match` and scalar `if` SHALL differentiate the
arm or branch executed by the forward program. Tags, pattern tests, guards,
and conditions carry zero cotangent; untaken code is not evaluated. Runtime
scrutinees, guarded arms, nested patterns, recursive calls, and List/tuple/ADT
branch values follow the same target-independent rule.

#### Scenario: Static arm selection lowers the taken arm

- **WHEN** a differentiated body matches on an ADT-typed parameter fixed at the `grad` boundary
- **THEN** only the taken arm is lowered and differentiated as the exact gradient

#### Scenario: Runtime-scrutinee match differentiates the executed arm

- **WHEN** a differentiated `match` has a scrutinee that lowers to a tensor node
- **THEN** only its executed arm is differentiated and the untaken arms contribute nothing

### Requirement: Field-wise ADT gradients

`grad(f)(Ctor { .. })` over an ADT argument containing a differentiable float
leaf SHALL return the executed constructor shape, exact field cotangents for
differentiable fields, explicit zero for uninfluential float fields, and
`unit` for discrete fields. Pure enums and recursively all-unit ADTs have no
differentiable leaf and are invalid only as explicit `wrt` targets. The rule
is identical in every execution mode.

#### Scenario: Field-wise gradient matches the argument structure

- **WHEN** `grad(f)` differentiates an all-float-field ADT argument
- **THEN** it returns the same constructor shape with a gradient per field, zeros for non-influencing fields

#### Scenario: Mixed-type ADT preserves discrete fields

- **WHEN** the ADT argument's type has a non-float-tensor field in any variant
- **THEN** differentiable fields receive cotangents and each discrete field remains as `unit` in position

### Requirement: vmap vectorization

`vmap(f, axis=n)` SHALL turn a single-example function into a batch function by inserting the
batch dimension at axis `n` as a DAG rewrite (not a loop), passing the batch dimension through
all operations untouched while reductions reduce the original axis. Non-tensor arguments SHALL
be broadcast (shared across the batch), and an out-of-bounds axis SHALL be `axis_out_of_bounds`.

#### Scenario: Batch dimension passes through elementwise ops

- **WHEN** the canonical default-axis form `vmap(f)` rewrites an elementwise DAG
- **THEN** the batch dimension is added and passes through without being reduced

#### Scenario: Out-of-bounds vmap axis is an error

- **WHEN** `vmap(f, axis=2)` targets a rank-1 tensor argument
- **THEN** it is an `axis_out_of_bounds` error

### Requirement: jit transparency and caching

`jit(f)` SHALL preserve the type and semantics of `f` (`jit(f)(x) = f(x)`) and act as a
compilation hint: the first call compiles and caches shape-specialized target code, matching
calls reuse it, and mismatched shapes recompile. The cache key SHALL be
`(function_identity, input_dim_names, input_dim_sizes, input_precisions)`, and `jit(jit(f))`
SHALL equal `jit(f)`.

#### Scenario: Matching shapes reuse the cached compilation

- **WHEN** a jit-compiled function is called twice with identical input shapes
- **THEN** the second call executes the cached code without recompiling

#### Scenario: Nested jit is idempotent

- **WHEN** `jit(jit(f))` is formed
- **THEN** it is equivalent to `jit(f)`

### Requirement: Optimization passes preserve semantics

Constant folding, DCE, CSE, algebraic simplification, fusion, and memory
planning SHALL preserve stored result bits, dtype, shape, effects, trap kind,
operation attribution, and trap occurrence for every input. Potentially
effectful or trapping nodes are DCE roots and never CSE candidates. Algebraic
identities require a proof under the owning numbered operation atoms; IEEE-like
spellings do not authorize zero/one, inverse-function, nested-cast, or
expand/reduction rewrites. Fusion SHALL preserve every primitive finalization
boundary and SHALL NOT cross reduction, load/store, or multi-consumer barriers.
The simplifier SHALL terminate at a fixpoint; an iteration limit is not a
semantic proof.

#### Scenario: DCE removes unconsumed nodes

- **WHEN** a total pure node's output is not consumed by any observable root
- **THEN** DCE removes it without deleting or reordering any effect or trap

#### Scenario: Fusion respects reduction barriers

- **WHEN** a reduction intervenes in an elementwise chain
- **THEN** fusion stops at the reduction rather than fusing across it

### Requirement: Transformation composition and ordering

Transformation orderings that are equivalent (`jit(grad(f)) = grad(jit(f))`, `jit(jit(f)) =
jit(f)`) SHALL be honored, while `vmap(grad(f))` and `grad(vmap(f))` SHALL NOT be equivalent.
The compiler SHALL apply `grad`/`vmap` expansion before fusion, never differentiating a fused
DAG; `grad(vmap(f))` at the source level SHALL be rejected unless reduced to a scalar first.

#### Scenario: jit is transparent to grad

- **WHEN** `grad(jit(f))` is formed
- **THEN** it is semantically equivalent to `jit(grad(f))`

#### Scenario: vmap-grad differs from grad-vmap

- **WHEN** comparing `vmap(grad(f))` and `grad(vmap(f))`
- **THEN** the former computes per-example gradients and the latter the gradient of the batch sum, and source-level `grad(vmap(f))` is rejected unless the vmapped result is first reduced to a scalar

### Requirement: Transform error contract

The transformations SHALL surface the defined errors: `non_differentiable` (non-tensor `wrt`
parameter), `no_differentiable_path` (warning, gradient identically zero),
`non_scalar_grad_output` (non-scalar output without a seed), `axis_out_of_bounds` (vmap), and
`shape_mismatch_in_jit` (recompile trigger, not an error). Each error SHALL name the offending
construct and, where applicable, a repair.

#### Scenario: No differentiable path warns

- **WHEN** no differentiable path exists from a `wrt` parameter to the output
- **THEN** a `no_differentiable_path` warning is emitted (a valid zero gradient, almost certainly a bug)

#### Scenario: jit shape mismatch recompiles rather than errors

- **WHEN** a jit-compiled function is called with shapes matching no cache entry
- **THEN** it triggers recompilation (optionally a performance warning), not an error
