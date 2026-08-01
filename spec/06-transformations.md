# Chelis Language Specification: DAG-to-DAG Transformations

## 1. Overview

Transformations are functions from DAGs to DAGs. They take a function (represented as a RISC DAG, see spec/05-risc-primitives.md) and produce a new function (a new RISC DAG). The three core transformations are:

1. **`grad`** -- Reverse-mode automatic differentiation
2. **`vmap`** -- Vectorized map over a named dimension
3. **`jit`** -- Just-in-time compilation with caching

This document also covers the optimization passes that operate on RISC DAGs (Section 5) and the rules governing how transformations compose (Section 6).

---

## 2. grad -- Reverse-Mode Automatic Differentiation

### 2.1 Signature

Given a function `f`:

```
If   f : A -> B
and  B is a scalar floating result
Then grad(f) : A -> dA
```

where `dA` is the gradient type:
- If `A` is a single tensor type `tensor[D, P]`, then `dA = tensor[D, P]` (same type).
- If `A` is a tuple `(T1, T2, ..., Tn)`, then `dA = (dT1, dT2, ..., dTn)`.
- If `A` is an ADT/record whose fields (across every variant) are all float tensors
  or float scalars, and at least one variant carries a field, then `dA` is the same
  constructor shape with a gradient per field.
- If a component of `A` is otherwise non-differentiable (e.g., `bool`, `i32`, an ADT
  with a non-tensor field in any variant, or a pure enum with no fields at all),
  then its gradient component is `unit`. An explicit `wrt` target that has no
  differentiable component is a type error.

Chelis's source-level `grad` returns gradients only, not `(value, grad)`.
For a multi-parameter function, the gradient payload is flattened:

```
If   loss : (tensor[D1, P], tensor[D2, P]) -> tensor[P]
Then grad(loss) : (tensor[D1, P], tensor[D2, P]) -> (tensor[D1, P], tensor[D2, P])
```

### 2.2 The `wrt` Parameter

By default, `grad(f)` differentiates with respect to all differentiable parameters of
`f`. The optional `wrt` parameter restricts differentiation to specific parameters:

```
grad(f, wrt=(param1, param2))
```

Parameters not in `wrt` are treated as constants (they receive no gradient). This is useful when a function has both parameters (to be optimized) and data (fixed inputs):

```
def loss(w: tensor[D, P], x: tensor[D2, P], y: tensor[P]): tensor[P] = ...

-- Differentiate only w.r.t. weights:
let dw = grad(loss, wrt=(w))(w, x, y)
```

When `wrt` is specified, the gradient result contains entries only for the listed
parameters, in the order they appear in `wrt`. One listed parameter returns one
gradient value directly; multiple listed parameters return a flat tuple.

### 2.3 Algorithm: Reverse-Mode AD

**Input:** A forward DAG `G` with nodes `[n_1, n_2, ..., n_k]` in topological order, where `n_k` is the output node.

**Output:** A backward DAG `G'` that computes both the forward output and gradients.

**Procedure:**

**Step 1 -- Initialize adjoints.**

Create an adjoint accumulator for every node in the forward DAG, initialized to zero:

```
for each node n_i in G:
    adjoint[n_i] = Const(0, shape_of(n_i), precision_of(n_i))
```

Set the output node's adjoint to the seed gradient:

```
adjoint[n_k] = Const(1.0, shape_of(n_k), precision_of(n_k))
```

If the output is a scalar tensor (0-dimensional), the seed is `Const(1.0, {}, P)`. If the output is a non-scalar tensor, the seed must be provided externally (this corresponds to a vector-Jacobian product).

**Step 2 -- Backward traversal.**

Process nodes in **reverse topological order** (from output toward inputs). For each node `n_i`:

1. Look up the adjoint rule for `n_i`'s operation in spec/05-risc-primitives.md Section 8.
2. For each input `n_j` of `n_i`, compute the adjoint contribution using the rule, passing `adjoint[n_i]` as the upstream gradient.
3. **Accumulate:** Add the contribution to `n_j`'s adjoint:

```
adjoint[n_j] = Add(adjoint[n_j], contribution_from_n_i)
```

**Step 3 -- Emit output.**

The backward DAG returns the requested gradient payload:

```
adjoint[p_1], ..., adjoint[p_m]
```

where `p_1, ..., p_m` are the parameters specified by `wrt` (or all tensor-typed Load nodes if `wrt` is not specified).

### 2.4 Gradient Accumulation (Multi-Use Nodes)

When a node `x` is consumed by multiple downstream nodes `y_1, y_2, ..., y_m`, the adjoint of `x` is the **sum** of all contributions from its consumers:

```
adjoint[x] = Add(
    adjoint_from_y_1,
    Add(adjoint_from_y_2,
        ...
        Add(adjoint_from_y_{m-1}, adjoint_from_y_m) ...))
```

This implements the multivariate chain rule: if `L = L(y_1(x), y_2(x), ..., y_m(x))`, then `dL/dx = sum_i (dL/dy_i * dy_i/dx)`.

In the DAG, this is represented by `Add` nodes that sum the contributions. The `Add` nodes are emitted during the backward traversal.

### 2.5 Worked Example

**Forward function:** `f(x) = sum(x * x)` where `x: tensor[{n=3}, f32]`

**Forward DAG:**
```
n1: Load("x")                     -- tensor[{n=3}, f32]
n2: Mul(n1, n1)                    -- tensor[{n=3}, f32]  (x * x)
n3: ReduceSum(n2, axis=n)          -- tensor[{}, f32]     (scalar sum)
```

**Backward pass:**

Initialize: `adjoint[n3] = Const(1.0, {}, f32)`

Process n3 (ReduceSum, axis=n):
- Adjoint rule: `Expand(g, axis=n, size=3)`
- `adjoint[n2] += Expand(Const(1.0), n, 3) = [1.0, 1.0, 1.0]`

Process n2 (Mul(n1, n1)):
- Adjoint rule: `(Mul(g, b), Mul(g, a))` where `a = b = n1`
- Contribution to n1 from first input: `Mul([1,1,1], n1) = n1`
- Contribution to n1 from second input: `Mul([1,1,1], n1) = n1`
- `adjoint[n1] = Add(n1, n1) = [2*x_0, 2*x_1, 2*x_2]`

**Result:** `grad(f)(x) = 2*x`

For `x = [1.0, 2.0, 3.0]`: forward output = 14.0, gradient = [2.0, 4.0, 6.0].

### 2.6 Worked Example: Multi-Layer Function

**Forward function:** `f(x) = exp(sum(x * x))` where `x: tensor[{n=2}, f32]`

**Forward DAG:**
```
n1: Load("x")                     -- tensor[{n=2}, f32]
n2: Mul(n1, n1)                    -- tensor[{n=2}, f32]
n3: ReduceSum(n2, axis=n)          -- tensor[{}, f32]
n4: Exp(n3)                        -- tensor[{}, f32]
```

**Backward pass:**

Initialize: `adjoint[n4] = Const(1.0, {}, f32)`

Process n4 (Exp(n3)):
- Rule: `Mul(g, Exp(n3))`
- `adjoint[n3] = Mul(1.0, Exp(n3)) = Exp(n3) = n4`

Process n3 (ReduceSum(n2, n)):
- Rule: `Expand(g, n, 2)`
- `adjoint[n2] = Expand(n4, n, 2) = [n4, n4]`

Process n2 (Mul(n1, n1)):
- Rule: `(Mul(g, n1), Mul(g, n1))` -- both inputs are n1, accumulate
- `adjoint[n1] = Add(Mul([n4, n4], n1), Mul([n4, n4], n1)) = 2 * n4 * n1`

**Result:** `grad(f)(x) = 2 * x * exp(sum(x*x))`

For `x = [1.0, 1.0]`: forward = exp(2) ~ 7.389, gradient = [14.778, 14.778].

### 2.7 Non-Differentiable Operations

The following operations produce zero gradients when encountered in a `grad` scope:

- **`CmpLt`**: Zero gradient to both inputs. The compiler emits a warning: "grad: CmpLt encountered; gradient is zero at this point."
- **Integer operations**: Not differentiable. If `grad` is applied to a function with integer-typed parameters, a `non_differentiable` type error is raised.
- **`Max(a, b)`**: Differentiable almost everywhere. The gradient routes to whichever input is larger (subgradient convention). Zero gradient at ties. See spec/05 Section 4.4.
- **`Cast` to integer**: `Cast(x, i32)` produces zero gradient because the floor/truncation function has zero derivative almost everywhere.

The compiler does **not** error on non-differentiable operations in the forward pass of a `grad`-ed function. It only errors if the function's **parameter types** are non-differentiable. This follows JAX's convention: `grad(relu)` is valid (ReLU uses Max internally, which has zero gradient at one point), but `grad(fn (x: bool) -> ...)` is a type error.

### 2.7.1 Symbolic Input Dimensions in Adjoint Construction

Adjoint construction runs before symbolic dimensions are bound, so an input
axis may be a `Named` dim with no concrete size. The rules split into three
classes:

- **Structural (supported).** Where the adjoint can carry the symbolic dim
  through without reading its value, it must. `Sum`/`Expand` adjoints carry
  the extent as a `DimExpr`; `Reshape`/`Permute` adjoints reuse the source
  dims; and a symbolic **bystander** axis (one the op does not touch) in the
  `Pad`, `Stride`, and `ProdReduce` adjoints uses the `SHRINK_TO_END`
  full-axis identity sentinel in its `Shrink` bounds, which
  `bind_symbolic_dims` (evaluation) and the C backend's `emit_shrink`
  resolve to the runtime extent. The `Shrink` adjoint of a full-axis
  `(0, SHRINK_TO_END)` bound is exactly zero padding on that axis. A
  `reshape` target dim written as static integer arithmetic over `shape()`
  reads of statically-sized axes folds to a literal at lowering; the
  `floor_div` / `trunc_div` / `mod` arms fold only on a non-negative
  dividend with a positive divisor, the domain where floor, truncating,
  and euclidean division agree, so the fold can never disagree with the
  runtime operator.
- **Runtime (node-valued; chelis#616).** Where the construction needs an
  extent that is only known at run time -- the strided axis of a `Stride`
  adjoint (the trim bound and the `m_a * step` merge extent), a symbolic
  axis under a concrete or runtime `Shrink` sub-range (the trailing pad
  amount `shape(x, axis) - end`), a runtime `Pad` bound (`end = before +
  shape(x, axis)`), or a `reshape` target restoring a runtime axis -- the
  adjoint builds the extent as a rank-0 integer scalar node (`Shape` reads
  plus `add`/`mul`/`neg` arithmetic, with explicit casts between integer
  precisions) referenced as a node-valued `RtDim` (see
  `spec/05-risc-primitives.md` §2.4.1). It never guesses a size. Bound
  scalars are a stop-gradient boundary: index math carries no cotangent and
  does not pull its producers (a window-count `floor_div`) into the
  differentiability check.
- **Value-dependent runtime structure.** An adjoint whose structure depends on
  a runtime extent lowers that structure explicitly. A `ProdReduce` adjoint
  computes, for each input element, the upstream cotangent times the product
  of every other element on the reduced axis; this definition does not divide
  and is correct when the input contains zeros. A runtime stride step scatters
  cotangents back into the selected positions and fills skipped positions with
  zero. Neither construction may be rejected merely because its extent or step
  is known only at run time.

Likewise at lowering, a shape()-derived arithmetic `reshape` target lowers to
a rank-0 scalar node referenced as a node-valued target extent. Every execution
lane enforces the numel invariant at run time; only a target the fold proves
negative is refused at lowering as a statically invalid program.

**Scalar `shape()` value reads.** A `shape(x, axis)`
read used as a scalar VALUE lowers to a `RiscOp::Shape { axis }` node (a
rank-0 integer extent; see `spec/05-risc-primitives.md` §2.5.1). It is
AD-transparent: the
node reads only shape metadata, so its adjoint routes a zero cotangent to the
input, and a loss whose value depends on a runtime dim (for example
`loss = sum(x) * shape(x, 0)`, whose gradient is `shape(x, 0)` at every
element) differentiates correctly in every lane. The node carries a
compile-time-constant `axis`; a `shape()` read whose axis is itself a runtime
(data- or metadata-derived) value is not representable; a DAG-forcing path
fails **loudly** with a source-located "requires a compile-time-constant
`axis`" diagnostic. Runtime movement bounds and reshape targets use the
node-valued representation from `spec/05-risc-primitives.md` §2.4.1.

### 2.8 Higher-Order Derivatives (Composition)

`grad(grad(f))` computes second derivatives. This works because the backward DAG produced by `grad(f)` is itself a valid RISC DAG, and `grad` can be applied to any valid DAG.

Example: `f(x) = x^3` (using `Mul(x, Mul(x, x))`)

- `grad(f)(x) = 3*x^2`
- `grad(grad(f))(x) = 6*x`

The second application of `grad` differentiates the gradient payload produced by the
first `grad`.

The backward DAG may share nodes with the forward DAG (e.g., `Exp(x)` reused in its own adjoint). These shared references must be preserved as-is -- the DAG is a graph, not a tree. The second `grad` application must correctly handle these shared nodes.

### 2.9 Checkpointing

For memory efficiency, `grad(f, checkpoint=true)` opts into gradient checkpointing. Instead of storing all intermediate forward values for use in the backward pass, the checkpointed version recomputes them during the backward pass.

**Semantics:** Identical to `grad(f)` -- the same function, the same gradients. The difference is operational: less memory, more compute.

During the backward traversal, when an adjoint rule needs a forward value (e.g.,
`Mul(g, Exp(x))` needs `Exp(x)`), checkpointed lowering emits a recomputation
from the forward inputs instead of referencing the cached node. The optimizer
schedules recomputations to minimize peak memory.

### 2.10 Tensor and Host Boundaries

`grad` applies to every pure differentiable function that satisfies §2.1,
whether its call originates in a tensor DAG or a host-language expression.
Both routes lower to the same RISC DAG and use the same adjoint rules:

```chelis
def jac_row[n](
  model: tensor[n, f32] -> f32 -> f32 -> f32,
  theta: tensor[n, f32], x: f32, y: f32
) -> tensor[n, f32] = {
  target = fn (theta_local: tensor[n, f32]) -> model(theta_local, x, y)
  grad(target, wrt=(theta_local))(theta)
}
```

Host runtimes apply `grad`/`vmap` by lowering the transformation application
back into the RISC DAG evaluator. That path is not a separate AD engine.

Host-list boundary idioms lower as follows:

- `to_tensor(to_list(x))`, which is the identity boundary and whose adjoint is
  the identity cotangent
- `to_tensor(map(f, to_list(x)))` and equivalent let-bound aliases, for
  concrete rank-1 `x`; the runtime lowering recurses into `f` element-by-element
  and stacks the scalar results
- `to_tensor(filter(p, to_list(x)))`, for concrete rank-1 `x`, in scalar-loss
  AD contexts; the primal predicate mask is treated as constant, selected
  positions receive cotangents, and rejected positions receive zero
- `fold(step, init, to_list(x))`, for concrete rank-1 `x`; lowering unrolls the
  recurrence so the reverse pass scans the recorded scalar trajectory through
  the ordinary structural adjoints

These rewrites deliberately use differentiable structural primitives
(`shrink`/`reshape`/`pad`/`add`) rather than `gather`/`scatter_add` for
`map`, so `grad(grad(...))` through boundary+map is supported by composition
of first-order rules. Dynamic-length list materialization is a host value
operation. Differentiating a
body that observes the selected cardinality of `filter(...)` through
`shape`/`len`/`numel` is unsupported; the supported `filter` rule is the
constant-mask cotangent path for scalar losses over the selected values.

### 2.10.1 `match` Bodies and ADT-Typed Arguments

**Static arm selection.** A `match` inside a differentiated body is
resolved at lowering time when its scrutinee is a compile-time-known
constructor value: a nullary constructor literal (`ModeA`), a positional
constructor application, a record construction (`Box { t: e }`), or an
ADT-typed parameter of the differentiated function (whose constructor shape is
fixed at the `grad` call boundary). Only the taken arm is lowered and
differentiated. This is the exact gradient, not an approximation: the
constructor tag is discrete, so perturbing tensor inputs cannot change the
taken arm. Every execution lane shares this DAG lowering.

**Static condition pruning for `if` (chelis#620).** The `if` analogue of
static arm selection: when an `if` condition const-folds at lowering time
(its lowered subgraph is scalar, closed over literals — no `Load`, no
`shape()` read, no random op — and every op is in the pure scalar
vocabulary, with `cast` truncation matching the evaluator exactly, a
zero-divisor `floor_div`/`trunc_div` refusing the fold rather than folding
a trap away, and any non-finite intermediate refusing the fold, because
the lowered comparison composition disagrees with the forward lanes' IEEE
semantics on NaN — chelis#666), ONLY the taken branch is lowered, and its
value passes
through verbatim — a tensor, a tuple, an ADT constructor, or a list. The
untaken branch is never lowered, so a `fail(...)` guard arm, an empty-list
base case, or a recursive call in the other branch cannot poison the DAG.
This is the exact gradient for the same reason as static arm selection: a
condition the fold can resolve cannot vary under input perturbation. Two
consequences follow:

- **Bounded recursion unrolling.** A recursive function in a
  differentiated body lowers by unrolling, terminated by the static
  pruning of its base case (`if k >= n then [] else ...recurse...` with
  literal-rooted bounds). Two caps backstop chains the pruning cannot
  bound — 512 active levels per callee, 1024 total — and exceeding either
  is a loud diagnostic naming the callee and the limit, never a hang. A
  recursion whose base-case condition is only known at runtime cannot
  unroll and hits the cap.
- **Static list values through `concat`.** The `Cons`/`Nil` spine an
  unrolled builder returns is consumed by list append
  (`concat(list, list)`) and by tensor `concat(list, axis)`, which emits
  the same Pad+Add cascade as the expression-level path.

Compiler-inserted linearity `copy`/`drop` over tuple and ADT values lower
structurally, one `Copy`/`Drop` per tensor leaf. A runtime-condition `if`
lowers through `RiscOp::Select`; tuple and ADT results select structurally per
leaf. The predicate is a stop-gradient boundary, and the cotangent flows only
through the selected branch. Runtime-scrutinee `match` follows the same
piecewise rule after selecting the arm by constructor tag.

**Field-wise ADT gradients.** `grad(f)(Ctor { .. })`
over an ADT argument whose fields are all float tensors or float scalars
returns a gradient with the same constructor shape, one gradient per field
(the pytree contract). A field that does not influence the output receives an
explicit zero tensor of its shape, so the gradient struct always matches the
argument's structure. For a multi-constructor sum type, the gradient
corresponds to whichever variant was constructed.

The ADT argument may appear alongside plain tensor/scalar arguments, as in
`grad(model_forward, wrt=params)(x, params)`. The
result is the per-target tuple, whose ADT slot is the field-wise gradient
struct and whose tensor slots are bare tensor gradients, exactly as the
multi-parameter tensor contract in §2.1; when `wrt` narrows to a single
target the result is that bare gradient (an ADT struct or a tensor) with no
enclosing tuple. A differentiated tensor argument that does not influence the
output receives an explicit zero tensor of its shape — mirroring the
adjoint-free ADT field above — so a multi-target tuple always keeps full
arity and every gradient stays in its own `out.0..out.N` slot; dropping the
slot would shift and mislabel every later gradient. Two pytree leaves whose
gradient is the same DAG node (e.g. `sum(add(t, y))` has adjoint `1` for both)
each keep their own root, so no tuple slot collapses.

Non-differentiable fields produce `unit` in the corresponding gradient field.
A pure enum has no continuous payload; selecting it explicitly in `wrt` is a
type error. Every execution lane preserves the resulting ADT structure.

### 2.11 Interaction With Effects

`grad` remains a compiler transform, not a user-visible effect handler.

- `Diff` is treated as a capability of the AD pipeline rather than a boundary effect
- `Accum` is an internal hook for backward-pass accumulation, not a user-facing effect
- `with seed(...)` is handled before or during lowering so seeded `dropout` enters the
  DAG as a deterministic `Dropout { rate, seed }` node
- the backward pass reuses the same seeded dropout
  mask rather than differentiating with respect to the seed

`with device(...)` is not a DAG-to-DAG transform. It is
validated on the checked Deep/build boundary: `chelis build --target c` rejects GPU
resource regions, and `chelis build --target hip` rejects CPU-only regions.

---

## 3. vmap -- Vectorized Map

### 3.1 Signature

```
If   f : tensor[D, P] -> tensor[D', P]
Then vmap(f, axis=n) : tensor[D with batch inserted at n, P]
                      -> tensor[D' with batch inserted at n, P]
```

`vmap` takes a function that operates on a single example and produces a function that
operates on a batch of examples. The new batch dimension is inserted at the requested
integer axis position. Lowering canonicalizes nonzero axes to axis 0 with
`permute`, applies the axis-0 rewrite, then permutes outputs back.

### 3.2 Semantics

Conceptually, `vmap(f, axis=0)` is equivalent to:

```
vmap(f, axis=0)(x) = stack([f(x[i]) for i in batch_dimension])
```

It is a DAG rewrite that lifts every operation over the additional batch dimension,
not a runtime loop.

### 3.3 DAG Rewrite Rules

For each node in the original DAG, the vmap transformation adds the batch dimension as follows:

| Original Op | vmapped Op |
|-------------|------------|
| `Add(a, b)` | `Add(a', b')` -- elementwise ops naturally extend |
| `Mul(a, b)` | `Mul(a', b')` -- same |
| `Neg(x)`, `Exp(x)`, etc. | `Neg(x')`, `Exp(x')` -- same |
| `ReduceSum(x, axis=d)` | `ReduceSum(x', axis=d)` -- reduce original axis, not batch |
| `ReduceMax(x, axis=d)` | `ReduceMax(x', axis=d)` -- same |
| `Reshape(x, D2)` | `Reshape(x', {batch} + D2)` -- preserve batch dim |
| `Expand(x, dim, size)` | `Expand(x', dim, size)` -- expand within each batch element |
| `Const(v, D, P)` | Batch-typed `Const(v, {batch} + D, P)` -- broadcast constant |
| `Load(buf)` | Load with batch dimension added to buffer type |

The key principle: the batch dimension passes through all operations without being touched. Elementwise ops are naturally batched. Reductions reduce over the original axis, not the batch axis. Shape operations preserve the batch dimension.

### 3.4 Type Rule

```
      G |- f : tensor[D, P] -> tensor[D', P]
      ---------------------------------------------------
      G |- vmap(f, axis=n) : tensor[D with batch inserted at n, P]
                            -> tensor[D' with batch inserted at n, P]
```

If `f` takes multiple arguments, each tensor argument gains the batch dimension:

```
      G |- f : (tensor[D1, P], tensor[D2, P]) -> tensor[D3, P]
      ---------------------------------------------------
      G |- vmap(f, axis=0) : (tensor[{batch} + D1, P], tensor[{batch} + D2, P])
                             -> tensor[{batch} + D3, P]
```

### 3.5 Composition

**vmap of vmap:** repeated application adds multiple batch axes. Nested
vectorization and stored higher-order transformation values follow the same rewrite.

**vmap of grad:**

`vmap(grad(f))` computes per-example gradients when the inner `grad(f)` function is
otherwise available.

Computes **per-example gradients**: each example in the batch gets its own independent gradient. This is useful for per-example gradient clipping or differential privacy.

The lowering path supports both the single-gradient case and flat tuple-valued
gradient payloads from multi-parameter
`grad(..., wrt=(...))`. Tuple construction and projection are resolved before the DAG
surface, so the executable DAG still carries only ordinary tensor roots.

**grad of vmap:**

`grad(vmap(f))` is well-typed only when the caller reduces the vmapped result
to a scalar floating result first, as required by §2.1.

These two are distinct concepts:
- `vmap(grad(f))` returns a batch of gradient vectors (one per example).
- `grad(fn(xs) = sum(vmap(f)(xs)))` is the batched gradient pattern once source-level
  `grad` lowering reaches that path.

### 3.6 Error Conditions

- `axis_out_of_bounds`: The integer axis is out of bounds for one of the vmapped tensor
  arguments or results.
- If `f` has non-tensor arguments, those arguments are broadcast (shared across the batch). They are not vmapped.

---

## 4. jit -- Just-In-Time Compilation

### 4.1 Signature

```
If   f : A -> B
Then jit(f) : A -> B
```

`jit` does not change the type or semantics of a function. It is a compilation hint.

### 4.2 Semantics

`jit(f)(x) = f(x)` for all `x`. The difference is operational:

1. **First call:** The DAG for `f` is compiled to target code (C, CUDA, etc.), specialized for the concrete shapes of the input arguments. The compiled code is cached.
2. **Subsequent calls:** If the input shapes match the cached version, the precompiled code is executed directly (no DAG interpretation, no compilation overhead).
3. **Shape mismatch:** If subsequent calls have different input shapes, the DAG is recompiled for the new shapes and a new cache entry is created.

### 4.3 Cache Key

The cache key for a jit-compiled function is:

```
(function_identity, input_dimension_names, input_dimension_sizes, input_precisions)
```

Two calls match the same cache entry if and only if all of these components are identical. Dimension names are part of the key because they affect the lowering (e.g., which axis to reduce over).

### 4.4 Compilation Boundary

In the RISC DAG, `jit` inserts a **compilation boundary**. The subgraph rooted at the `jit` node is treated as an opaque compiled unit by the enclosing graph. Optimizations (fusion, CSE, etc.) operate within the boundary but do not cross it.

### 4.5 Interaction with Other Transformations

- `jit(grad(f))`: Compile the gradient function. This is the most common usage pattern -- compile the training step for repeated execution.
- `grad(jit(f))`: Semantically equivalent to `jit(grad(f))` because `jit` is transparent to differentiation. The compiler may rewrite this to the more efficient form.
- `jit(vmap(f))`: Compile the vectorized function.
- `jit(jit(f))`: Equivalent to `jit(f)`. Nested jit is a no-op.

---

## 5. Optimization Passes

Optimization passes are DAG-to-DAG rewrites that preserve semantics while improving performance. They run after `grad`/`vmap` transformation but before code generation.

### 5.1 Constant Folding

**Rule:** If all inputs to a node are `Const` nodes, evaluate the operation at compile time and replace the node with a single `Const` node containing the result.

**Applies to:** All elementwise ops (unary and binary), reductions, Cast. Does not apply to Load/Store (which depend on external buffers).

**Examples:**
```
Before: Add(Const(2.0), Const(3.0))
After:  Const(5.0)

Before: Mul(Const(0.0), x)
After:  Const(0.0, shape_of(x), precision_of(x))    -- by algebraic simplification
```

### 5.2 Dead Code Elimination (DCE)

**Rule:** Remove any node whose output is not consumed by any other live node.

**Algorithm:**
1. Mark all `Store` nodes and the designated output nodes as **live**.
2. Walk backward through the DAG: for each live node, mark all its input nodes as live.
3. Remove all nodes not marked as live.

**Example:**
```
Before:
  n1 = Load("x")
  n2 = Mul(n1, n1)          -- consumed by n4
  n3 = Add(n1, Const(1.0))  -- NOT consumed by anything
  n4 = ReduceSum(n2, n)     -- output

After:
  n1 = Load("x")
  n2 = Mul(n1, n1)
  n4 = ReduceSum(n2, n)
```

DCE is particularly important after `grad`, which may produce gradient nodes for values that are not part of the `wrt` set.

### 5.3 Common Subexpression Elimination (CSE)

**Rule:** If two nodes perform the same operation on the same inputs (identical node references), merge them into one node.

**Algorithm:**
1. For each node, compute a hash: `hash(op, input_node_id_1, input_node_id_2, ..., params)`.
2. Maintain a hash map from hashes to node IDs.
3. If a new node's hash already exists in the map and the nodes are structurally identical, replace all references to the new node with the existing node.

**Example:**
```
Before:
  n1 = Load("x")
  n2 = Exp(n1)        -- first occurrence
  n3 = Exp(n1)        -- duplicate of n2
  n4 = Add(n2, n3)

After:
  n1 = Load("x")
  n2 = Exp(n1)
  n4 = Add(n2, n2)    -- n3 merged into n2
```

CSE is particularly valuable after `grad`, which often introduces duplicate subexpressions. For example, `Exp(x)` appears in both the forward pass and its own adjoint; CSE ensures it is computed only once.

### 5.4 Algebraic Simplification

**Rules (applied as pattern rewrites):**

| Pattern | Replacement | Justification |
|---------|-------------|---------------|
| `Add(x, Const(0))` | `x` | Additive identity |
| `Add(Const(0), x)` | `x` | Additive identity (commutative) |
| `Mul(x, Const(1))` | `x` | Multiplicative identity |
| `Mul(Const(1), x)` | `x` | Multiplicative identity (commutative) |
| `Mul(x, Const(0))` | `Const(0)` | Zero annihilation |
| `Mul(Const(0), x)` | `Const(0)` | Zero annihilation (commutative) |
| `Neg(Neg(x))` | `x` | Double negation |
| `Reciprocal(Reciprocal(x))` | `x` | Double reciprocal |
| `Exp(Log(x))` | `x` | Inverse functions |
| `Log(Exp(x))` | `x` | Inverse functions |
| `Reshape(Reshape(x, D1), D2)` | `Reshape(x, D2)` | Collapse reshapes |
| `Permute(Permute(x, P1), P2)` | `Permute(x, compose(P1, P2))` | Compose permutations |
| `Add(x, x)` | `Mul(Const(2), x)` | Strength reduction |
| `Cast(Cast(x, P1), P2)` | `Cast(x, P2)` | Collapse casts |
| `Expand(ReduceSum(x, d), d, s)` | `x` (if s matches) | Expand-reduce cancellation (only when safe) |

**Application strategy:** Iterate rules until a fixpoint (no more rules apply). Limit to 100 iterations to prevent pathological cases. In practice, convergence occurs within 3-5 iterations.

### 5.5 Operator Fusion

**Rule:** Merge chains of elementwise operations into a single fused kernel. Instead of writing intermediate results to memory between each operation, compute the entire chain in registers.

**Fusible pattern:** A sequence of unary and binary elementwise nodes where:
1. Every intermediate result has the same dimension set.
2. Each intermediate result is consumed by exactly one node in the chain (no fan-out to non-chain consumers).
3. No reduction or shape-changing operation intervenes.

**Example:**
```
Before (3 separate kernels, 2 intermediate memory writes):
  n1 = Neg(x)
  n2 = Exp(n1)
  n3 = Add(Const(1.0), n2)
  n4 = Reciprocal(n3)

After fusion (1 kernel, no intermediate memory):
  n_fused = FusedKernel("sigmoid", [x])
  -- internally computes: Reciprocal(Add(Const(1), Exp(Neg(x))))
```

**Fusion barriers:**
- ReduceSum, ReduceMax (change parallelism pattern)
- Load, Store (memory side effects)
- Nodes with multiple consumers outside the fusion group

Fusion is the primary optimization for GPU backends, where memory bandwidth is the bottleneck. On CPU backends, fusion improves cache utilization.

### 5.6 Memory Planning

**Purpose:** Analyze the lifetimes of intermediate tensors and schedule buffer reuse to minimize peak memory usage.

**Algorithm:**
1. Compute the liveness interval of each tensor: from its definition to its last use.
2. Build an interference graph: two tensors interfere if their liveness intervals overlap.
3. Color the interference graph: assign buffer slots to tensors such that non-interfering tensors share the same slot.
4. Emit buffer allocation and deallocation at the appropriate points.

**Interaction with grad:** The backward pass extends the liveness of many forward-pass tensors (those needed by adjoint rules). Checkpointing (Section 2.9) trades memory for compute by shortening these intervals.

---

## 6. Transformation Composition

### 6.1 Valid Compositions

| Expression | Meaning | Notes |
|-----------|---------|-------|
| `grad(f)` | Reverse-mode AD | Core operation |
| `grad(grad(f))` | Second derivatives | Nested AD |
| `grad(f, wrt=(w))` | Gradient w.r.t. specific params | Selective differentiation |
| `vmap(f, axis=a)` | Vectorize over integer axis `a` | Batch dimension inserted at `a` |
| `jit(f)` | Compile and cache | Shape-specialized |
| `jit(grad(f))` | Compile gradient function | Repeated gradient execution |
| `grad(jit(f))` | Differentiate through jit | Equivalent to `jit(grad(f))` |
| `vmap(grad(f))` | Per-example gradients | Supports flat tuple-valued gradient payloads |
| `grad(vmap(f))` | Gradient of a scalar reduction over a vectorized result | The vmapped result must first reduce to a scalar |
| `jit(vmap(grad(f)))` | Compiled per-example gradients | Composition of the three transformations |

### 6.2 Commutativity Rules

Some transformation orderings are semantically equivalent:

```
jit(grad(f))  =  grad(jit(f))       -- jit is transparent to grad
jit(vmap(f))  =  vmap(jit(f))       -- jit is transparent to vmap
jit(jit(f))   =  jit(f)             -- jit is idempotent
```

The following are **not** equivalent:

```
vmap(grad(f))  !=  grad(vmap(f))
```

The left side computes per-example gradients (a batch of gradient vectors). The right side computes the gradient of the sum over the batch (a single gradient vector equal to the sum of per-example gradients).

### 6.3 Transformation Ordering in the Compiler

The compiler applies transformations in the following order:

1. **Type checking** -- Verify the program is well-typed (spec/04).
2. **Lowering** -- Convert typed AST to RISC DAG (spec/01 pipeline).
3. **Early optimization passes** -- Apply local simplification, CSE, and DCE that do not depend on later transform expansion.
4. **`grad` expansion** -- Expand all `grad` nodes into backward DAGs.
5. **`vmap` expansion** -- Expand all `vmap` nodes into batched DAGs.
6. **Post-transform optimization passes** -- Re-run simplification, CSE, and DCE on the transformed DAG.
7. **Fusion** -- Fuse eligible post-transform DAG regions for target backends that benefit from fused kernels.
8. **`jit` boundary insertion** -- Mark compilation boundaries for jit.
9. **Code generation** -- Emit target code.

Steps 3 and 4 are the core "transformation" steps. After expansion, all grad and vmap constructs have been rewritten away, and the DAG consists entirely of RISC primitives.

For the GPU backend specifically, this means:

```text
lower -> optimize -> grad -> optimize again -> fuse -> codegen
```

Do not differentiate a fused DAG. `grad` should operate on the unfused RISC DAG, and the
resulting forward/backward graph should then be fused using the ordinary fusion pass.

---

## 7. Formal Semantics of grad

This section provides a precise formal definition of the `grad` transformation, sufficient for an unambiguous implementation.

### 7.1 Definitions

A **RISC DAG** is a tuple `(N, E, inputs, outputs)` where:
- `N` is a set of nodes, each labeled with an operation and a type.
- `E` is a set of directed edges `(src, dst, port)`, where `port` identifies which input of `dst` the edge connects to.
- `inputs` is an ordered list of Load nodes (function parameters).
- `outputs` is an ordered list of nodes whose values constitute the function's return value.

The **topological order** of a DAG is any total ordering of `N` such that for every edge `(src, dst, _)`, `src` precedes `dst`.

### 7.2 Adjoint DAG Construction

Given a forward DAG `G = (N, E, inputs, outputs)`:

```
function build_adjoint(G, wrt):
    -- Step 1: Topological sort
    topo = topological_sort(N)
    reverse_topo = reverse(topo)

    -- Step 2: Initialize adjoint map
    adj = new Map<Node, Node>
    for each node n in N:
        adj[n] = Const(0, type_of(n))

    -- Step 3: Seed the output
    assert len(outputs) == 1    -- for scalar output
    adj[outputs[0]] = Const(1.0, type_of(outputs[0]))

    -- Step 4: Backward traversal
    for each node n in reverse_topo:
        if adj[n] is Const(0):
            continue  -- skip nodes with no gradient (optimization)

        let rule = adjoint_rule(op_of(n))
        let contributions = rule(inputs_of(n), adj[n])

        for (input_node, contribution) in zip(inputs_of(n), contributions):
            adj[input_node] = Add(adj[input_node], contribution)

    -- Step 5: Collect results
    grads = [adj[p] for p in wrt]
    return (outputs[0], tuple(grads))
```

### 7.3 Adjoint Rule Interface

Each RISC primitive defines an adjoint rule with the following interface:

```
adjoint_rule(forward_inputs: [Node], upstream_grad: Node) -> [Node]
```

The function takes:
- `forward_inputs`: references to the nodes that are the inputs of the forward operation.
- `upstream_grad`: the accumulated adjoint of the forward operation's output.

It returns a list of nodes (one per forward input), each representing the adjoint contribution to that input.

**Critical requirement:** The adjoint rule may reference forward input nodes and forward output nodes. These references create edges from the backward DAG to the forward DAG, establishing the sharing that is characteristic of reverse-mode AD. The combined forward+backward DAG must remain acyclic.

### 7.4 Handling Multiple Outputs

If the forward function returns a tuple `(y1, y2, ..., yn)`, the seed gradients are a tuple of the same structure:

```
adj[output_1] = seed_1
adj[output_2] = seed_2
...
adj[output_n] = seed_n
```

For a typical loss function (scalar output), there is one output and the seed is `Const(1.0)`. For Jacobian computation, the seeds are one-hot vectors, and `grad` is called once per output element.

### 7.5 Handling Non-Differentiable Subgraphs

When the backward traversal encounters a `CmpLt` node:
1. The adjoint contributions to its inputs are `Const(0)`.
2. A warning is emitted: `"grad: non-differentiable operation CmpLt at <location>; gradient is zero."`
3. Traversal continues normally.

When the backward traversal encounters a `Cast(x, int_type)` node:
1. The adjoint contribution is `Const(0)` (integer rounding has zero derivative).
2. A warning is emitted if the cast is to an integer type.
3. `Cast(x, float_type)` has adjoint `Cast(g, original_float_type)` -- precision casts are differentiable.

### 7.6 Verification

After constructing the backward DAG, the compiler verifies:

1. **Acyclicity:** The combined forward+backward DAG is acyclic. (It always is, by construction, since backward edges point from later to earlier nodes in the forward topological order.)
2. **Type consistency:** Every new node's type is consistent with its operation and inputs.
3. **Completeness:** Every node in `wrt` has a non-trivial adjoint (or a warning is emitted if the adjoint is provably zero).

---

## 8. Error Conditions

Transformations can produce the following errors:

### 8.1 `non_differentiable`

**Trigger:** `grad` applied to a function whose `wrt` parameters include non-tensor types (bool, integer, ADT).

**Message:** `"Cannot differentiate with respect to parameter 'x' of type bool. Only tensor-typed parameters with float precision are differentiable."`

**Repair:** Suggest excluding the parameter from `wrt`, or restructuring the function.

### 8.2 `no_differentiable_path`

**Trigger:** There is no path in the DAG from any `wrt` parameter to the output that passes through differentiable operations. The gradient would be identically zero.

**Severity:** Warning, not error. The gradient is valid (it is zero), but this is almost certainly a bug.

**Message:** `"No differentiable path from parameter 'w' to the output. Gradient will be zero."`

### 8.3 `non_scalar_grad_output`

**Trigger:** `grad(f)` where `f` returns a non-scalar tensor, and no seed gradient is provided.

**Message:** `"grad requires a scalar output (0-dimensional tensor). Function returns tensor[{n=10}, f32]. Use a reduction (e.g., ReduceSum) to produce a scalar, or provide a seed gradient."`

**Repair:** Suggest wrapping the function output in a ReduceSum or computing a loss value.

### 8.4 `axis_out_of_bounds` (vmap)

**Trigger:** `vmap(f, axis=n)` where `n` is out of bounds for one of the vmapped tensor types.

**Message:** `"vmap axis 2 is out of bounds for rank 1 tensor. Choose an axis between 0 and 1."`

### 8.5 `shape_mismatch_in_jit`

**Trigger:** A jit-compiled function is called with shapes that don't match any cached compilation. This is not an error -- it triggers recompilation -- but may be logged as a performance warning if it happens frequently.

---

## 9. Pass Ordering and Convergence

### 9.1 Standard Pipeline

The standard tensor compiler pipeline, in order:

```
1. AD expansion                         -- transform grad nodes into backward DAGs
2. closed-list no-op cleanup            -- remove identity cast/reshape/permute only
3. BLAS/gather/scatter recognizers      -- replace dense patterns with explicit RISC ops
4. cross-function specialization        -- specialize call sites after AD
5. dead code elimination (DCE)          -- prune orphan dense intermediates
6. in-place fusion                      -- reuse buffers only after DCE settles liveness
7. code generation                      -- emit target code
```

This order is semantic, not only an optimization preference: AD must see the
ordinary RISC decomposition, recognizers must run before DCE/fusion/codegen, and
DCE must run after recognizers so replaced dense paths are removed. The
source-level tripwire is `chelis_ir::specialize::SPECIALIZATION_PIPELINE_ORDER`.

### 9.2 Fixpoint Convergence

Any local cleanup loop nested inside the standard pipeline is guaranteed to
terminate because:
- Each pass either reduces the number of nodes or leaves it unchanged.
- The minimum number of nodes is bounded below by the number of load/store roots
  and required output nodes.
- The iteration limit provides an absolute guarantee.

### 9.3 Correctness Requirement

Every optimization pass must preserve the semantics of the DAG. Formally:

```
For all inputs x: eval(optimize(G), x) = eval(G, x)
```

where `eval(G, x)` evaluates the DAG `G` on input `x`.
