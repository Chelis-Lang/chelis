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
- If `A` is a float scalar or tensor type `tensor[D, P]`, then `dA = A`.
- If `A` is a tuple `(T1, T2, ..., Tn)`, then `dA = (dT1, dT2, ..., dTn)`.
- If `A = List[T]` and `dT` is defined, then `dA = List[dT]`; the cotangent
  list has exactly the primal list's runtime length and positional order.
- If `A` is an ADT/record, then `dA` has the same executed constructor shape
  with `dT` for every differentiable field. A non-differentiable field has
  gradient component `unit`; a pure enum therefore has no differentiable
  payload.
- Any other non-differentiable component (`bool`, signed integer, string,
  deferred value, function, or resource) has gradient type `unit`.

The recursive definition is shape-preserving. It never drops a tuple field,
list element, or ADT field merely because its cotangent is unit, and it never
uses a backend carrier limitation to reject a language-defined cotangent.

A disconnected differentiable scalar or tensor receives exact zeros with its
actual argument's dtype and ordered shape, including empty axes and rank zero. See
[`grad_disconnected.ch`](../examples/grad_disconnected.ch).

[`grad_bitwise.ch`](../examples/grad_bitwise.ch) demonstrates exact discrete
coefficients retained in the forward graph under `grad` and `vmap`.

Source-level `grad` returns gradients only, not `(value, grad)`.
For a multi-parameter function, the gradient payload is flattened:

```
If   loss : (tensor[D1, P], tensor[D2, P]) -> tensor[P]
Then grad(loss) : (tensor[D1, P], tensor[D2, P]) -> (tensor[D1, P], tensor[D2, P])
```

### 2.2 The `wrt` Parameter

By default, `grad(f)` differentiates with respect to all differentiable parameters of
`f`. Parameter classification uses the final inferred types, including nested
components. An unresolved type variable is not a non-differentiable type and
cannot justify omitting a gradient component. Application may determine a
lambda's parameter types under [04-INF-1]; an unresolved classification at
the enclosing declaration boundary is a type error. A result annotation on
the gradient value does not supply a parameter binding site, including when
application determines the operand's function type before its parameter types.
A join with another branch's result likewise does not supply that binding site.
Result constraints retain this role through function generalization and recursive
references. Equality imposed by aggregate construction, update, or combination
is also a result constraint. These constraints remain required by ordinary type
checking; preserving their origin does not remove an equality.

The optional `wrt` parameter restricts differentiation to specific parameters:

```
grad(f, wrt=(param1, param2))
```

Parameters not in `wrt` are treated as constants (they receive no gradient). This is useful when a function has both parameters (to be optimized) and data (fixed inputs):

```
def loss(w: tensor[D, P], x: tensor[D2, P], y: tensor[P]) -> tensor[P] = ...

-- Differentiate only w.r.t. weights:
dw = grad(loss, wrt=w)(w, x, y)
```

When `wrt` is specified, the gradient result contains entries only for the listed
parameters, in the order they appear in `wrt`. One listed parameter returns one
gradient value directly; multiple listed parameters return a flat tuple. Repeated
parameter names are retained as repeated result positions. Each name denotes a
formal parameter of the target callable's immutable lexical origin, so aliases
and alias chains do not rename or reorder formals. Branches retain that origin
only when every path has the same exact lexical identity and ordered formals;
same-signature lambdas or declarations remain distinct. Tuple, ADT constructor,
and record patterns recursively project callable origins from their scrutinee
payloads. Unknown names and targets without a statically established callable
origin are rejected.

The executable [`grad_wrt_order.ch`](../examples/grad_wrt_order.ch) example
distinguishes written target order from declaration order using unequal
cotangents. [`grad_selector_provenance.ch`](../examples/illustrative/grad_selector_provenance.ch)
checks and evaluates alias-preserving constructor and record pattern projection.

### 2.3 Algorithm: Reverse-Mode AD

**Input:** A forward DAG `G` with its canonical node sequence
`[n_1, n_2, ..., n_k]`, where every input precedes its consumer and `n_k` is
the output node. A node's canonical forward ordinal is its position in this
sequence. The ordinal belongs to the input DAG; a transform never derives it
from an arbitrary topological-sort tie.

**Output:** A backward DAG `G'` that computes both the forward output and gradients.

**Procedure:**

**Step 1 -- Initialize adjoints.**

Create one cotangent accumulator for every forward value. Its type is §2.1's
recursive `dT`, not necessarily one RISC tensor node. Initialize it with the
shape-preserving zero cotangent for the executed primal value:

```
for each value v_i in G:
    adjoint[v_i] = zero_cotangent(type_of(v_i), primal_value(v_i))
```

Set the output node's adjoint to the seed gradient:

```
adjoint[n_k] = Const(1.0, shape_of(n_k), precision_of(n_k))
```

If the output is a float scalar or rank-zero float tensor, the seed is exact
one at its dtype. A vector-Jacobian product supplies a seed with exactly the
output's recursive cotangent type.

**Step 2 -- Backward traversal.**

Process nodes in **reverse topological order** (from output toward inputs). For each node `n_i`:

1. Except for the seeded output node, materialize `adjoint[n_i]` from its
   queued contributions using §2.4's canonical key order and balanced tree.
2. Look up the operation's exact adjoint, zero-cotangent, or structural
   rejection contract in its controlling numbered-spec atom.
3. For an adjoint contract, compute each input's recursive contribution from
   `adjoint[n_i]`. For a zero-cotangent contract, contribute the exact
   shape-preserving zero. For a rejection contract, stop with its named
   `AdRejectionReason`.
4. **Queue:** store the contribution for `n_j` under the consumer's canonical
   forward ordinal and the exact input-slot index. Do not incrementally build
   an addition tree during traversal.

```
contributions[n_j][(canonical_forward_ordinal(n_i), input_slot)] =
    contribution_from_n_i
```

**Step 3 -- Emit output.**

The backward DAG returns the requested gradient payload:

```
adjoint[p_1], ..., adjoint[p_m]
```

where `p_1, ..., p_m` are the parameters specified by `wrt`, or every
parameter containing at least one differentiable float leaf when `wrt` is
omitted.

### 2.4 Gradient Accumulation (Multi-Use Nodes)

When a value `x` is consumed by multiple downstream edges, order those edges
lexicographically by canonical forward node ordinal, then by input-slot index.
Repeated use by one consumer therefore remains distinct and is ordered by the
slot in that consumer's exact input list. This consumer-edge order is
independent of the work-list or topological-sort tie order used to construct
the backward graph. At every float scalar or tensor leaf, combine their contributions with the
canonical adjacent-pair balanced addition tree at that leaf's declared dtype,
beginning with an exact positive-zero base leaf. Tuple, List, and ADT
cotangents combine corresponding fields/elements recursively; List lengths and
the executed ADT constructor must match the primal, and `unit` fields remain
`unit`.

```
adjoint[x] = balanced_add(+0, contribution(edge_1), ..., contribution(edge_m))
```

This implements the multivariate chain rule: if `L = L(y_1(x), y_2(x), ..., y_m(x))`, then `dL/dx = sum_i (dL/dy_i * dy_i/dx)`.

The backward DAG represents this exact recursive tree. A backend may reassociate
only where another numbered rule explicitly permits it.

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

### 2.7 Adjoint, Zero-Cotangent, and Rejected Operations

Every numeric callable has exactly one of spec/05 §5's contracts: an exact
adjoint, an exact zero cotangent, or a structural `grad` rejection. The
operation's controlling `[05-OP-N]` atom decides which; an implementation may
not replace one class with another.

Comparisons, predicates, and `shape` use their specified zero-cotangent rules
and therefore do not block a surrounding differentiable graph. Extrema use
their exact tie/NaN subgradient rules, not a generic zero-at-ties convention.
Piecewise-constant numeric conversions and roundings whose atoms specify
`AdRejectionReason::PiecewiseConstant` reject the transformed graph even
though their forward execution is legal; they never silently return zero.
The one boundary that structural analysis does not cross is [05-OP-42]'s
`stop_gradient` barrier: its argument's subgraph is outside adjoint
construction and rejection analysis, its forward value passes through
unchanged, and its argument receives the shape-preserving exact zero
cotangent.

An explicit `wrt` target must contain at least one differentiable float leaf.
A bool, signed-integer, key, string, function, resource, or recursively
all-unit parameter is a `non_differentiable` type error. A List, tuple, or ADT with a
differentiable leaf is legal and returns §2.1's shape-preserving cotangent;
its discrete fields remain present as `unit`.
The unit cotangent does not replace a discrete primal value: string selectors,
constructor tags, and other host metadata keep their exact values and lexical
bindings while the differentiated function executes. The operation-specific
structural rejections above still apply to computations of those values.

### 2.7.1 Symbolic Input Dimensions in Adjoint Construction

Adjoint construction preserves symbolic identities where ordinary type
reasoning proves them and otherwise carries exact runtime i64 extent nodes.
`Sum`/`Expand` carry their selected extents, `Reshape`/`Permute` restore the
recorded source shape, and `Pad`/`Shrink`/`Stride` use [05-MOV-1]'s runtime
bounds and exact inverse graphs. Bound and axis scalars have zero cotangent.

Value-dependent traversal is equally part of the transform. [05-OP-14]
reverses the executed balanced `ProdReduce` tree for every runtime slice;
[05-MOV-1] maps every stride cotangent to its unique selected source index;
and [05-RWIN-2]/[05-OP-39] reverse the exact executed runtime-window graph,
including overlap accumulation and second derivatives. No symbolic or
runtime dimension, loop extent, step, window, or target shape is a structural
`grad` rejection merely because it is unavailable at transform time. Every
execution mode evaluates the same generated runtime graph and guards.

**Scalar `shape()` value reads.** A `shape(x, axis)` read used as a scalar
value is the rank-zero i64 extent operation in
`spec/05-risc-primitives.md` [05-OP-7]/[05-SHAPE-1]. It is AD-transparent:
the operation reads only shape metadata, so its adjoint routes zero cotangent
to the input and discrete i32 axis. A loss whose value depends on a runtime
dimension (for example `loss = sum(x) * shape(x, axis)`) therefore
differentiates with the actual selected runtime extent. Literal and computed
axes have one semantic operation and remain representable through every DAG
transform; a transform never guesses, freezes, or rejects the axis merely
because a backend representation once stored it as a compile-time field.

### 2.8 Higher-Order Derivatives (Composition)

`grad(grad(f))` computes second derivatives. This works because the backward DAG produced by `grad(f)` is itself a valid RISC DAG, and `grad` can be applied to any valid DAG.

Example: `f(x) = x^3` (using `Mul(x, Mul(x, x))`)

- `grad(f)(x) = 3*x^2`
- `grad(grad(f))(x) = 6*x`

The second application of `grad` differentiates the gradient payload produced by the
first `grad`.

**Implementation note:** The backward DAG may share nodes with the forward DAG (e.g., `Exp(x)` reused in its own adjoint). These shared references must be preserved as-is -- the DAG is a graph, not a tree. The second `grad` application must correctly handle these shared nodes.

### 2.9 Checkpointing

For memory efficiency, `grad(f, checkpoint=true)` opts into gradient checkpointing. Instead of storing all intermediate forward values for use in the backward pass, the checkpointed version recomputes them during the backward pass.

**Semantics:** Identical to `grad(f)` -- the same function, the same gradients. The difference is operational: less memory, more compute.

**Implementation approach:** During the backward traversal, when an adjoint rule needs a forward value (e.g., `Mul(g, Exp(x))` needs `Exp(x)`), instead of referencing the cached forward node, emit a new computation of that value from the forward inputs. The optimizer can then schedule these recomputations to minimize peak memory.

### 2.10 Recursive host values and execution modes

Every execution mode applies `grad`/`vmap` through the same semantic transform
rules. Host lists are ordinary differentiable carriers under §2.1, not a
target-specific boundary subset. In particular:

- `to_tensor(to_list(x))` for positive-rank tensors, with any trailing
  extents hidden by empty Lists supplied by the expected tensor type per
  [05-OP-57], is the identity boundary; its adjoint is the identity cotangent
  and uses the saved full forward shape
- `map(f, xs)` differentiates each executed application of `f` in list order
  and returns the same-length positional cotangent list
- `filter(p, xs)` treats the exact forward predicate mask as constant,
  routes selected output cotangents to their source positions, and gives
  rejected positions zero cotangent
- `fold(step, init, xs)` reverses the recorded recurrence trajectory through
  the ordinary structural adjoints

These rules compose through higher-order AD. A list length or selected
cardinality is discrete metadata and contributes zero cotangent, while the
values selected by the fixed forward mask retain the cotangents above.
Scalar-only and list-carrying host graphs have the same language legality as
tensor graphs; a target representation gap does not change their signatures
or adjoints.

### 2.10.1 Control flow and ADT-typed arguments

`match` differentiates the arm executed by the forward program. Constructor
tags, pattern tests, and guards are discrete and contribute zero cotangent;
the untaken arms contribute nothing and are not evaluated. This rule is
identical whether the scrutinee is known statically or only at runtime. Nested
destructuring preserves the field paths used by §2.1's recursive cotangent
type.

Scalar `if` likewise differentiates the executed branch and gives its boolean
condition zero cotangent. The untaken arm contributes exactly zero to every
cotangent outside it, row by row under `vmap`, even where a where-lowered
branch computes that arm's values and they or their derivatives are not
finite (spec/10 §3). Tensor conditions use the exact `where` adjoint.
Branch values may be scalar, tensor, List, tuple, or ADT; their cotangent keeps
the same recursive shape.

Recursive calls differentiate the finite recurrence actually executed by the
forward program and reverse that recorded call trajectory. The language
defines no fixed differentiation-depth cap. A nonterminating recursion remains
nonterminating rather than becoming a target-specific gradient rejection.

Compiler-inserted linearity `copy`/`drop` operates structurally over List,
tuple, and ADT values. `grad(f)(Ctor { .. })` returns §2.1's shape-preserving
cotangent for the executed constructor: differentiable fields receive their
exact field cotangents, uninfluential differentiable fields receive explicit
zero, and discrete fields receive unit in their original positions. A pure
enum has the same constructor with no continuous cotangent payload. For a
multi-constructor sum type, the gradient uses the constructor selected by the
primal argument.

The ADT argument may appear ALONGSIDE plain tensor/scalar arguments, as in
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

Mixed differentiable and discrete fields therefore remain legal and
well-typed. Runtime scrutinees, guarded arms, nested patterns, and compiled ADT
parameters follow the same rule in every execution mode. A missing host-ABI
carrier is a backend capability gap, not a language restriction.

### 2.11 Interaction With Effects

`grad` remains a compiler transform, not a user-visible effect handler.

- `Diff` is treated as a capability of the AD pipeline rather than a boundary effect
- `Accum` is an internal backward-pass accumulation effect and is not a
  user-handled boundary effect
- a `key` is a discrete input, like an integer: a key, its seed, and random
  source words have zero cotangent (`unit` in a structured cotangent), while
  [05-OP-8]'s bounds, [05-OP-37]'s data input, and the stdlib graphs use their
  exact pathwise adjoints at the forward draw's key and [05-OP-37] states the
  rate's contract
- the backward pass, and checkpoint recomputation under §2.9, re-read the bits
  of a key that its forward draw consumed in order to replay that draw; such a
  replay read is not a use under spec/04 [04-LIN-9]

`with device(...)` is not a DAG-to-DAG transform. It selects the declared
`Resource(Device)` region at the checked execution boundary; a target
capability gap is reported through its typed capability cell and never changes
the region's language semantics.

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
integer axis position. The implementation canonicalizes nonzero axes to axis 0 with
`permute`, applies the axis-0 rewrite, then permutes outputs back.

### 3.2 Semantics

Conceptually, the default-axis form `vmap(f)` is equivalent to:

```
vmap(f)(x) = stack([f(x[i]) for i in batch_dimension])
```

But it is **not** implemented as a loop. Instead, it is a DAG rewrite that lifts every operation to operate over the additional batch dimension.

A scalar `key` formal of `f` is mapped: its actual SHALL be a
`tensor[n, key]` holding one key per row, and row `i` of the expansion above
applies `f` to that tensor's row-`i` key, so row `i`'s draws are keyed from it
([05-RNG-1]). A scalar key actual for a key formal, or a key captured by `f`,
is a type error: every row would consume the same key (spec/04 [04-LIN-9]).

The executable [`vmap_tensor_capture.ch`](../examples/vmap_tensor_capture.ch)
demonstrates that the mapped input varies by row while one lexical tensor
capture is shared across all rows.

Fusing `vmap(grad(f))` does not remove §2.3's shape-preserving zero
cotangents or change §2.2's selected-parameter order. An absent adjoint after
lowering a complete body gives an exact positive-zero tensor of the batched
actual's shape and precision; an unresolved live callable is not evidence
of a constant body. Forward dependencies required by §5.2 remain observable.
The executable [`grad_fused_zero.ch`](../examples/grad_fused_zero.ch) demonstrates
a single batched zero cotangent; it does not demonstrate native export of
multi-result fused gradients.

### 3.3 DAG Rewrite Rules

For each node in the original DAG, the vmap transformation adds the batch dimension as follows.
Only a tensor `Load` that denotes a mapped formal receives the batch dimension
at the call boundary. A tensor `Load` that denotes a lexical capture retains
its authored type and is served once at that exact rank. When a batched
consumer needs that captured value, the rewrite inserts an explicit
`Insert(capture, axis=0, size=batch)` movement node, whose result has the batch
axis prepended. This explicit lift does not authorize implicit rank extension
or shape broadcasting in elementwise primitives.

| Original Op | vmapped Op |
|-------------|------------|
| `Add(a, b)` | `Add(a', b')` -- elementwise ops naturally extend |
| `Mul(a, b)` | `Mul(a', b')` -- same |
| `Neg(x)`, `Exp(x)`, etc. | `Neg(x')`, `Exp(x')` -- same |
| `ReduceSum(x, axis=d)` | `ReduceSum(x', axis=d)` -- reduce original axis, not batch |
| `ReduceMax(x, axis=d)` | `ReduceMax(x', axis=d)` -- same |
| `Reshape(x, D2)` | `Reshape(x', {batch} + D2)` -- preserve batch dim |
| `Expand(x, dim, size)` | `Expand(x', dim, size)` -- broadcast within each batch element |
| `Insert(x, dim, size)` | `Insert(x', dim, size)` -- insert within each batch element |
| `Const(v, D, P)` | Batch-typed `Const(v, {batch} + D, P)` -- broadcast constant |
| `Load(mapped_formal)` | Load with batch dimension added to the mapped formal type |
| `Load(key formal)` | Load of `tensor[{batch}, key]`, one key per row |
| Random draw keyed by `k` | The same draw keyed by the batched `k'`: row `b` draws with `k'[b]` (spec/10 §3.2) |
| `Load(capture)` | Exact authored capture load followed by `Insert(capture, 0, batch)` |

The key principle: the batch dimension passes through all operations without being touched. Elementwise ops are naturally batched. Reductions reduce over the original axis, not the batch axis. Shape operations preserve the batch dimension. The rank-0 subgraph that produces a bound, and the bound carrier inside a movement operation, follow §3.7: in the rewritten DAG's numbering a positional `dim` and an `InputAxis` axis shift by the inserted batch axis, and a rank-0 extent value is shared rather than batched.

For `gather` and the scatter operations, a mapped index tensor's leading
batch axes pair positionwise with the data tensor's leading batch axes. Each
paired axis occurs once in the result or update shape; only the remaining
index axes replace the selected data axis. Index bounds are checked against
that data axis within each batch element. `vmap(grad(f))` preserves this
pairing through `gather`'s scatter-add adjoint. The internal `OneHot` marker
keeps its vocabulary axis last after batching.

### 3.4 Type Rule

```
      G |- f : tensor[D, P] -> tensor[D', P]
      ---------------------------------------------------
      G |- vmap(f, axis=n) : tensor[D with batch inserted at n, P]
                            -> tensor[D' with batch inserted at n, P]
```

If `f` takes multiple arguments, each mapped tensor formal gains the batch dimension:

```
      G |- f : (tensor[D1, P], tensor[D2, P]) -> tensor[D3, P]
      ---------------------------------------------------
      G |- vmap(f) : (tensor[{batch} + D1, P], tensor[{batch} + D2, P])
                             -> tensor[{batch} + D3, P]
```

A scalar `key` formal maps to a key tensor with the batch axis alone:

```
      G |- f : (key, tensor[D, P]) -> tensor[D', P]
      ---------------------------------------------------
      G |- vmap(f) : (tensor[{batch}, key], tensor[{batch} + D, P])
                             -> tensor[{batch} + D', P]
```

A lexical tensor capture is not an additional mapped argument. In the
conceptual expansion in §3.2 it is loop-invariant, and its authored rank is
unchanged. The transformed DAG makes its use shape-equal with mapped values by
the explicit `Insert` rule in §3.3. A plain elementwise application to
different-rank tensors remains a type error under spec/04 and spec/05.

When a row function has an unknown parameter or result type, the checker
retains this type rule until the function's body and any enclosing declaration
group determine the types it transforms. An application of the mapped
function can bind an untyped row parameter from the corresponding slice of a
mapped tensor actual, under [04-INF-1]. A claimed type for the mapped result
does not determine the row function's result. Every remaining unknown
transform-relevant type is a type error at the declaration boundary.

### 3.5 Composition

**vmap of vmap:** repeated application adds multiple batch axes. Nested
vectorization and stored higher-order transform values follow the same rule as
direct `vmap(f)(args...)` applications.

**vmap of grad:**

`vmap(grad(f))` computes per-example gradients when the inner `grad(f)` function is
otherwise available.

Computes **per-example gradients**: each example in the batch gets its own independent gradient. This is useful for per-example gradient clipping or differential privacy.

The result may contain a single gradient or §2.1's recursive tuple/List/ADT
gradient payload from multi-parameter `grad(..., wrt=(...))`; vectorization
preserves that complete structure.

**grad of vmap:**

`grad(vmap(f))` follows the ordinary scalar-output rule for `grad`. A bare
vmapped tensor result is rejected by §8.3; composing an exact scalar reduction
over that result is legal and differentiates the complete vectorized graph.

These two are distinct concepts:
- `vmap(grad(f))` returns a batch of gradient vectors (one per example).
- `grad(fn(xs) = sum(vmap(f)(xs)))` is the batched gradient pattern once source-level
  `grad` receives the required scalar result.

### 3.6 Error Conditions

- `axis_out_of_bounds`: The integer axis is out of bounds for one of the vmapped tensor
  arguments or results.
- If `f` has non-tensor arguments other than scalar `key` formals, those arguments are broadcast (shared across the batch). They are not vmapped. A scalar `key` formal is always mapped (§3.2). A broadcast argument that carries a key (spec/04 §8.4.1) is a type error, because every row would use its keys (spec/04 [04-LIN-9]); keys reach the rows only as a mapped `tensor[n, key]`.

### 3.7 Runtime Extents

A rank-0 extent value that feeds a movement bound, an `expand` or `insert`
size, or a `reshape` target (a `shape()` read, an integer parameter, a cast, checked
integer arithmetic, or a user-function result over these) is not batched. It
is a non-tensor argument in the sense of §3.6: one value is shared by every
batch element, it is evaluated exactly once, and its traps and effects occur
exactly once in the order of the unbatched function. A folded tensor-axis read
(`InputAxis`, spec/05-risc-primitives.md §2.4.1) observes the axis it named in
the unbatched function: after the batch axis is inserted a literal axis shifts
by one, and a computed axis is normalized against the unbatched rank and then
shifted, so the read never selects the batch axis. A materialized `shape()`
value node (the `Node` form) shifts its axis the same way. A `shape()` read
used both as an extent and as an ordinary value is still evaluated once and
the ordinary use sees the shared rank-0 value. The movement operation itself
follows §3.3: the batch axis passes through untouched and every bound applies
within each batch element.

An extent whose value depends on the elements of a vmapped tensor argument
would vary per batch element and cannot describe one stacked result shape.
Such a program is a type error, `batch_varying_extent` (§8.6): vectorization
neither shares one element's value across the batch nor produces a ragged
result.

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

**Rule:** If all inputs to a node are `Const` nodes, evaluate the operation at compile time and replace the node with a single `Const` node containing the result. A key operation ([05-OP-69] through [05-OP-72]) is never folded: no `Const` holds a key (spec/10 §3.2), so key derivations stay symbolic.

**Applies to:** All elementwise ops (unary and binary), reductions, Cast. Does not apply to Load/Store (which depend on external buffers).

**Examples:**
```
Before: Add(Const(2.0), Const(3.0))
After:  Const(5.0)

Before: Mul(Const(0.0), x)
After:  Const(0.0, shape_of(x), precision_of(x))    -- by algebraic simplification
```

### 5.2 Dead Code Elimination (DCE)

**Rule:** Remove a node only when its result is not consumed by another live
node and removing its execution preserves every effect and trap occurrence.
Potentially effectful or trapping nodes are observable roots; purity alone does
not make a possible trap dead. A potentially trapping node traps only within
its activation (spec/10 §3): where its activation is false it computes a value
and checks nothing.

Liveness is scoped to the program the evaluation runs. An evaluation of
selected roots enters each selected root's declaration. Another declaration's
work runs only inlined into the nodes of the declaration that reaches it: a
call runs the function's body in its caller, and a reference to a top-level
value declaration whose initializer has a potentially trapping node runs that
initializer at the reference, within the reference's activation, whether or
not the reference is read (spec/03 §4.4). A value declaration with no
potentially trapping node is one set of nodes that every reference reads. A
node of a declaration the evaluation does not enter is not part of that
program, and a node whose activation is false checks nothing ([05-RNG-1]).

**Algorithm:**
1. Mark every effectful and every potentially trapping node of a declaration
   the evaluation enters, every `Store`, and every designated output as
   **live**.
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

**Rule:** If two total, deterministic, pure nodes perform the same operation on
the same typed inputs and semantic parameters, they may merge only when one
execution has the same stored result and observation order as both original
executions. Effectful, resource, IO, Test, and potentially trapping
nodes never merge. The key includes the complete dtype, shape, accumulator,
and operation identity; an implementation name is not a semantic key. A
random draw or random-key operation is a pure function of its operands
([05-RNG-1]): CSE, reordering, and DCE treat it like any other pure node, and
a draw or `split_keys` count that can trap is potentially trapping under
§5.2.

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

An algebraic rewrite is legal only when the owning numbered operation atoms
prove equality of stored result bits, dtype, shape, effects, trap kind,
operation attribution, and [04-NUM-12] trap occurrence for the complete input
domain. IEEE-looking algebra is not such a proof. In particular, none of these
patterns is unconditional:

| Pattern | Why it requires a proof |
|---------|-------------------------|
| `Add(x, 0)` / `Mul(x, 1)` | signed zero and NaN finalization can make the arithmetic observation differ from `x` |
| `Mul(x, 0)` | NaN, infinity, signed zero, and integer trap behavior defeat zero annihilation |
| `Neg(Neg(x))` | signed-min overflow and NaN finalization can be observable |
| `Reciprocal(Reciprocal(x))`, `Exp(Log(x))`, `Log(Exp(x))` | rounding, domain, zero, infinity, and NaN behavior are not inverse identities |
| nested `Cast` | the first conversion can round or trap and may not be erased |
| nested runtime `Reshape` | the inner shape guard can trap even when the outer target is valid |
| `Expand(Sum(x, ...), ...)` | reduction arithmetic and multiplicity are observable |

A dtype-specific rule may discharge those obligations for a narrower proven
domain. `Permute(Permute(x, P1), P2)` may compose only when both permutations
are already statically validated and the composed node preserves the exact
metadata transformation. Rewrites iterate to a fixpoint, but termination and
the semantic proof are both required; an iteration limit is not correctness.

### 5.5 Operator Fusion

**Rule:** Merge chains of elementwise operations into a single fused kernel. Instead of writing intermediate results to memory between each operation, compute the entire chain in registers.

Fusion preserves every primitive's declared arithmetic width and stored-value
finalization boundary. It may remove an intermediate allocation, but may not
contract operations, retain an intermediate at wider precision, reorder a
trap/effect, or change NaN and signed-zero observations unless the governing
atom explicitly permits that transformation.

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
| `grad(f)` | Reverse-mode AD | Exact adjoints from the owning op atoms |
| `grad(grad(f))` | Second derivatives | Nested AD |
| `grad(f, wrt=w)` | Gradient w.r.t. specific param | Selective differentiation |
| `vmap(f, axis=a)` | Vectorize over integer axis `a` | Batch dimension inserted at `a` |
| `jit(f)` | Compile and cache | Shape-specialized |
| `jit(grad(f))` | Compile gradient function | Same transformed graph as `grad(jit(f))` |
| `grad(jit(f))` | Differentiate through jit | Equivalent to `jit(grad(f))` |
| `vmap(grad(f))` | Per-example gradients | Preserves the complete recursive gradient payload |
| `grad(vmap(f))` | Differentiate a vectorized function | Requires the composed function to return a scalar |
| `jit(vmap(grad(f)))` | Compiled per-example gradients | Composition of the same three contracts |

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
- `N = [n_1, ..., n_k]` is the canonical ordered node sequence. Each node is
  labeled with an operation and a type, and its canonical forward node ordinal
  is its unique position in `N`.
- `E` is a set of directed edges `(src, dst, port)`, where `port` identifies which input of `dst` the edge connects to.
- `inputs` is an ordered list of Load nodes (function parameters).
- `outputs` is an ordered list of nodes whose values constitute the function's return value.

The canonical sequence is topological: for every edge `(src, dst, _)`,
`src` has a lower canonical ordinal than `dst`. An implementation may use a
different valid topological work-list internally, but canonical consumer-edge
order always uses `(ordinal(dst), port)` and therefore cannot change with that
choice.

A **contribution key** is either the distinguished `SeedKey` or a consumer-edge
key `(canonical_forward_ordinal(dst), port)`. `SeedKey` orders before every
consumer-edge key. Before accumulation, contributions are sorted by this key;
the seed is the sole contribution to the scalar output.

### 7.2 Adjoint DAG Construction

Given a forward DAG `G = (N, E, inputs, outputs)`:

```
function build_adjoint(G, wrt):
    -- Step 1: Topological sort
    topo = stable_topological_order(N, tie_break=canonical_forward_ordinal)
    reverse_topo = reverse(topo)

    -- Step 2: Initialize shape-preserving contribution lists
    contributions = new Map<Node, List<(ContributionKey, Node)>>
    for each node n in N:
        contributions[n] = []

    -- Step 3: Seed the scalar float output
    assert len(outputs) == 1    -- for scalar output
    contributions[outputs[0]].append((SeedKey, exact_one(type_of(outputs[0]))))

    -- Step 4: Backward traversal
    for each node n in reverse_topo:
        upstream = balanced_sum(
            exact_zero(cotangent_type(type_of(n))),
            values_sorted_by_key(contributions[n]))

        let rule = adjoint_rule(op_of(n))
        let input_contributions = rule(inputs_of(n), upstream)

        for (input_slot, input_node, contribution) in enumerate_inputs(
                inputs_of(n), input_contributions):
            edge_key = (canonical_forward_ordinal(n), input_slot)
            contributions[input_node].append((edge_key, contribution))

    -- Step 5: Collect the selected recursive gradient payload
    grads = [finalize_contributions(p, contributions[p]) for p in wrt]
    return pack_wrt_gradients(grads)
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

When backward traversal encounters a zero-cotangent operation such as a
comparison or `shape`, it contributes exact zero to each input named by its
atom and traversal continues. An implementation may emit a diagnostic note,
but that note does not change validity or the cotangent.

When traversal encounters an operation whose atom requires structural
rejection, construction stops with that atom's exact `AdRejectionReason`.
This includes piecewise-constant float-to-integer conversion and rounding; it
does not silently insert `Const(0)`. Float-to-float precision casts use their
declared cast adjoint. A [05-OP-42] `stop_gradient` node is a barrier:
traversal contributes the shape-preserving exact zero for its argument and
does not enter the argument's subgraph, so a structurally rejected operation
inside it does not stop construction.

A signed-integer operation whose atom assigns the `IntegerArithmeticOutput`
structural rejection passes exact zero to its operands when reached only by an
exact-zero control cotangent; this does not request its forward-only adjoint.
Traversal still visits those operands, so a structural rejection such as a
float-to-integer cast beneath the operation is reported. If any path reaches
that same operation outside the exact-zero control path, its
`IntegerArithmeticOutput` rejection applies. Other operations retain their
own atom's disposition, including a structural rejection on a zero path.

### 7.6 Verification

After constructing the backward DAG, the compiler verifies:

1. **Acyclicity:** The combined forward+backward DAG is acyclic. (It always is, by construction, since backward edges point from later to earlier nodes in the forward topological order.)
2. **Type consistency:** Every new node's type is consistent with its operation and inputs.
3. **Completeness:** Every node in `wrt` has a non-trivial adjoint (or a warning is emitted if the adjoint is provably zero).

---

## 8. Error Conditions

Transformations can produce the following errors:

### 8.1 `non_differentiable`

**Trigger:** `grad` applied to an explicit `wrt` parameter whose recursive
cotangent type contains no differentiable float leaf. Bool, integer, string,
function, resource, and all-unit ADT parameters meet this condition; an ADT,
tuple, or List containing a float scalar or tensor leaf does not.

**Message:** `"Cannot differentiate with respect to parameter 'x': its type contains no differentiable float leaf."`

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

### 8.6 `batch_varying_extent` (vmap)

**Trigger:** `vmap(f)` where a movement bound, `expand` or `insert` size, or `reshape` target inside `f` depends on the elements of a vmapped tensor argument, so its value could differ between batch elements.

**Message:** `"vmap cannot vectorize an extent that depends on batched tensor elements: the insert size at <site> reads elements of vmapped argument 'x'. Compute the extent from shape() or a scalar argument, or apply the movement outside vmap."`

**Repair:** Suggest deriving the extent from `shape()` or a scalar parameter, or moving the data-dependent movement outside `vmap`.

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
