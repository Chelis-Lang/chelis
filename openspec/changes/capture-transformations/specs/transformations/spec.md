## ADDED Requirements

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
the same tensor type, a tuple yields a tuple of gradients, and a differentiable ADT yields the
same constructor shape with a gradient per field. A multi-parameter function's payload SHALL be
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

`grad` SHALL construct the backward DAG by initializing adjoints to zero, seeding the scalar
output adjoint to `Const(1.0)`, traversing nodes in reverse topological order applying each
op's adjoint rule, and accumulating multi-consumer contributions by summation (the multivariate
chain rule). The combined forward+backward DAG SHALL remain acyclic.

#### Scenario: Multi-use node sums contributions

- **WHEN** a node is consumed by multiple downstream nodes
- **THEN** its adjoint is the sum of all consumer contributions

#### Scenario: Backward DAG is acyclic

- **WHEN** the backward DAG references forward nodes in its adjoint rules
- **THEN** the combined forward+backward DAG remains acyclic because backward edges point to earlier forward nodes

### Requirement: Non-differentiable operation handling

`grad` SHALL error only when a `wrt` parameter's type is non-differentiable (bool, integer,
ADT with a non-tensor field), not on non-differentiable operations in the forward pass. `CmpLt`
and `Cast` to an integer type SHALL contribute zero gradient with a warning; `Cast` to a float
type SHALL be differentiable.

#### Scenario: grad through an internal comparison is valid

- **WHEN** `grad` is applied to a function using `Max`/`CmpLt` internally with float tensor parameters
- **THEN** it is valid (zero gradient at the non-differentiable point) rather than an error

#### Scenario: grad over a bool parameter is a type error

- **WHEN** `grad` is applied to a function whose `wrt` parameter is `bool`
- **THEN** it is a `non_differentiable` type error naming the parameter

### Requirement: Symbolic-dim adjoint construction

Adjoint construction SHALL run before symbolic dimensions are bound and SHALL classify each
case: structural adjoints carry the symbolic dim as a `DimExpr` or full-axis sentinel; runtime
node-valued extents build the extent as a rank-0 integer scalar node; and value-dependent
constructions (a `ProdReduce` reduced axis, a runtime stride step) SHALL fail loudly at
construction naming the op, axis, and symbolic dim. A scalar `shape()` read SHALL be
AD-transparent with a zero cotangent.

#### Scenario: Structural adjoint carries the symbolic dim

- **WHEN** a `Sum`/`Expand` adjoint is built over a symbolic axis
- **THEN** it carries the extent as a `DimExpr` without reading its runtime value

#### Scenario: Value-dependent symbolic adjoint fails loudly

- **WHEN** a `ProdReduce` adjoint or a runtime stride-step adjoint needs a value-dependent structure over a symbolic axis
- **THEN** it fails at construction naming the op, axis, and symbolic dim rather than guessing a size

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

### Requirement: Static match and if differentiation

Inside a differentiated body, a `match` whose scrutinee is a compile-time-known constructor and
an `if` whose condition const-folds SHALL lower only the taken arm/branch, giving the exact
gradient. A runtime scrutinee, an arm guard, nested destructuring beyond field bindings, or a
runtime-condition `if` over an ADT/tuple branch SHALL be rejected loudly.

#### Scenario: Static arm selection lowers the taken arm

- **WHEN** a differentiated body matches on an ADT-typed parameter fixed at the `grad` boundary
- **THEN** only the taken arm is lowered and differentiated as the exact gradient

#### Scenario: Runtime-scrutinee match differentiation is rejected

- **WHEN** a differentiated `match` has a scrutinee that lowers to a tensor node
- **THEN** it is rejected loudly rather than differentiated

### Requirement: Field-wise ADT gradients

`grad(f)(Ctor { .. })` over an ADT argument whose fields are all float tensors/scalars SHALL
return the same constructor shape with one gradient per field, zero-filling non-influencing
fields. An ADT with a non-float field in any variant, or a pure enum with no fields, SHALL be
rejected loudly; the compiled C lane SHALL reject `grad` exports over ADT-typed parameters.

#### Scenario: Field-wise gradient matches the argument structure

- **WHEN** `grad(f)` differentiates an all-float-field ADT argument
- **THEN** it returns the same constructor shape with a gradient per field, zeros for non-influencing fields

#### Scenario: Mixed-type ADT field is rejected

- **WHEN** the ADT argument's type has a non-float-tensor field in any variant
- **THEN** it is rejected loudly naming the field, even if the constructed variant is float-clean

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

Constant folding, DCE, CSE, algebraic simplification, fusion, and memory planning SHALL be
semantics-preserving DAG rewrites: for all inputs, `eval(optimize(G), x) = eval(G, x)`. The
algebraic-simplification loop SHALL iterate to a fixpoint with a bounded iteration limit;
fusion SHALL NOT cross reduction, load/store, or multi-consumer barriers.

#### Scenario: DCE removes unconsumed nodes

- **WHEN** a node's output is not consumed by any live node
- **THEN** DCE removes it while preserving the evaluated result of the DAG

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
