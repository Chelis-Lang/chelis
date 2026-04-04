# Transformations

Transformations are DAG-to-DAG rewrites. They take a function (represented as a RISC DAG) and produce a new function (a new DAG). The three core transformations are `grad`, `vmap`, and `jit`. This document also covers optimization passes.

---

## 1. grad — Reverse-Mode Automatic Differentiation

### Signature

If `f : A → B`, then `grad(f) : A → (B, ∂A)` where `∂A` is a tuple of gradients with respect to each tensor-typed parameter.

`grad(f, wrt=[p1, p2])` differentiates with respect to specific parameters. Default: all tensor-typed parameters.

### Algorithm

Given a forward DAG `G` with nodes `[n₁, n₂, ..., nₖ]` in topological order:

1. **Forward pass:** Execute (or symbolically trace) the DAG to establish node values and the computation graph.

2. **Initialize:** Set `adjoint[nₖ] = 1.0` (if output is scalar) or the incoming gradient (if output is tensor). For loss functions, this is typically `Const(1.0)`.

3. **Backward pass:** For each node `nᵢ` in **reverse** topological order:
   a. Look up the adjoint rule for `nᵢ`'s operation (see spec/05, Section 7).
   b. For each input `nⱼ` of `nᵢ`, compute the adjoint contribution: `∂nⱼ_from_nᵢ = adjoint_rule(nᵢ, adjoint[nᵢ])`.
   c. **Accumulate:** `adjoint[nⱼ] += ∂nⱼ_from_nᵢ`. If `nⱼ` has multiple consumers, their contributions are summed (via `Add`).

4. **Output:** Return `(forward_output, (adjoint[p1], adjoint[p2], ...))` for the requested parameters.

### Gradient Accumulation

When a node has multiple consumers, its adjoint is the **sum** of contributions from all consumers. This is the multivariate chain rule:

```
If node x is used by nodes y₁, y₂, ..., yₘ:
  adjoint[x] = Add(adjoint_from_y₁, Add(adjoint_from_y₂, ...))
```

This is implemented by emitting `Add` nodes in the backward DAG.

### Non-Differentiable Operations

- `CmpLt`: produces zero gradients. The compiler emits a warning: "grad: CmpLt encountered; gradient is zero at this point."
- Integer operations: not differentiable. Type error if `grad` is applied to a function with integer-typed parameters.
- `Max`: differentiable almost everywhere. Gradient routes to the larger input (subgradient convention). See spec/05.

### Worked Example

**Forward function:** `f(x) = sum(x * x)` where `x: tensor[{n=3}, f32]`

**Forward DAG:**
```
n1: Load(x)                          -- tensor[{n=3}, f32]
n2: Mul(n1, n1)                      -- tensor[{n=3}, f32]  (x * x)
n3: ReduceSum(n2, axis=n)            -- tensor[{}, f32]     (scalar)
```

**Backward DAG construction:**

Step 1: `adjoint[n3] = Const(1.0, {}, f32)`

Step 2: Process n3 (ReduceSum). Adjoint rule: `Expand(grad_out, axis, size)`.
```
adjoint[n2] = Expand(adjoint[n3], n, 3)   -- tensor[{n=3}, f32] = [1,1,1]
```

Step 3: Process n2 (Mul(n1, n1)). Adjoint rule: `(Mul(g, b), Mul(g, a))`.
Since both inputs are n1, accumulate both contributions:
```
contrib_a = Mul(adjoint[n2], n1)     -- grad_out * x
contrib_b = Mul(adjoint[n2], n1)     -- grad_out * x
adjoint[n1] = Add(contrib_a, contrib_b)  -- 2 * grad_out * x = 2x
```

**Result:** `grad(f)(x) = (sum(x*x), 2*x)` ✓

For `x = [1, 2, 3]`: forward = 14, gradient = [2, 4, 6].

### Composition: Higher-Order Derivatives

`grad(grad(f))` computes second derivatives. This works because the backward DAG is itself a valid RISC DAG. Applying `grad` to the backward DAG produces a second backward DAG.

Example: `f(x) = x³` (scalar).
- `grad(f)(x) = (x³, 3x²)`
- `grad(grad(f))(x) = ((x³, 3x²), 6x)`

**Implementation note:** The backward DAG may contain nodes that reference forward values (e.g., `Exp(x)` reused in its own adjoint). These shared references are valid — the DAG is a graph, not a tree.

### Checkpointing (Phase 2)

For memory efficiency, `grad(f, checkpoint=true)` recomputes forward values during the backward pass instead of storing them. This trades compute for memory.

Not implemented in Phase 0. Specced here so the DAG structure accommodates it.

---

## 2. vmap — Vectorized Map

> **Status:** Phase 2 implementation. Semantics specified here.

### Signature

`vmap(f, axis=name) : tensor[{name} ∪ D, P] → tensor[{name} ∪ D', P]`

Where `f : tensor[D, P] → tensor[D', P]`.

### Semantics

`vmap` takes a function that operates on a single example and produces a function that operates on a batch. Conceptually:

```
vmap(f, axis=batch)(x) = stack([f(x[i]) for i in batch])
```

But implemented as a DAG rewrite, not a loop.

### DAG Rewrite

For each node in the original DAG:
- Add the `batch` dimension to its inputs and outputs.
- Elementwise ops: no change needed (they already operate per-element).
- Reductions: only reduce along the original axis, not the new batch axis.
- Shape ops: operate within each batch slice.

### Composition

```
vmap(vmap(f, "batch"), "seq")
```

Vectorizes `f` over two axes. The order of `vmap` applications determines the nesting of batch dimensions.

### Interaction with grad

```
grad(vmap(f, "batch"))    -- gradient of the batched function
vmap(grad(f), "batch")    -- batch of per-example gradients
```

These are **not** equivalent in general. The first computes the gradient of the sum over the batch. The second computes per-example gradients. Both are valid; the user chooses which they need.

---

## 3. jit — Just-In-Time Compilation

> **Status:** Phase 2 implementation. Semantics specified here.

### Signature

`jit(f) : A → B` where `f : A → B`. Semantically identical to `f`.

### Semantics

`jit(f)(x) = f(x)` for all `x`.

The difference is operational:
1. On first call, the DAG is compiled to target code (C, CUDA, etc.) specialized for the input shapes.
2. On subsequent calls with the same shapes, the compiled code is reused.
3. On calls with different shapes, recompilation occurs.

### Cache Key

The cache key is: `(function_identity, input_shapes, input_precisions)`. Dimension names are part of the shape.

### Interaction with Other Transformations

- `jit(grad(f))`: compile the gradient function. Most common usage.
- `grad(jit(f))`: differentiate through the jit boundary. Equivalent to `jit(grad(f))` since `jit` is semantically transparent.
- `jit(vmap(f))`: compile the vectorized function.

---

## 4. Optimization Passes

Optimization passes are DAG-to-DAG rewrites that preserve semantics while improving performance or reducing memory usage. They run after `grad` but before code generation.

### 4.1 Constant Folding

**Rule:** If all inputs to a node are `Const` nodes, evaluate the operation at compile time and replace with a single `Const` node.

**Example:**
```
Before: Add(Const(2.0), Const(3.0))
After:  Const(5.0)
```

**Scope:** Applies to all elementwise ops, reductions, and shape ops where all inputs are compile-time constants.

### 4.2 Dead Code Elimination (DCE)

**Rule:** Remove any node whose output is not consumed by any other node and is not a `Store` operation.

**Algorithm:**
1. Mark all `Store` nodes and the final output node as live.
2. Walk backward through the DAG, marking each input of a live node as live.
3. Remove all non-live nodes.

**Example:**
```
Before:
  n1 = Load(x)
  n2 = Mul(n1, n1)      -- used
  n3 = Add(n1, n1)      -- NOT used by anything downstream
  n4 = ReduceSum(n2)    -- output

After:
  n1 = Load(x)
  n2 = Mul(n1, n1)
  n4 = ReduceSum(n2)
```

### 4.3 Common Subexpression Elimination (CSE)

**Rule:** If two nodes have the same operation and identical inputs (same node references), merge them into one node.

**Algorithm:**
1. Hash each node by `(op, input_node_ids)`.
2. If a hash collision occurs and the nodes are structurally identical, replace all references to the second node with the first.

**Example:**
```
Before:
  n1 = Load(x)
  n2 = Exp(n1)
  n3 = Exp(n1)       -- same op, same input as n2
  n4 = Add(n2, n3)

After:
  n1 = Load(x)
  n2 = Exp(n1)
  n4 = Add(n2, n2)
```

This is particularly important after `grad`, which can introduce duplicate subexpressions (e.g., `Exp(x)` reused in both forward and backward).

### 4.4 Algebraic Simplification

**Rules:**
| Pattern | Replacement |
|---------|-------------|
| `Add(x, Const(0))` | `x` |
| `Add(Const(0), x)` | `x` |
| `Mul(x, Const(1))` | `x` |
| `Mul(Const(1), x)` | `x` |
| `Mul(x, Const(0))` | `Const(0)` |
| `Mul(Const(0), x)` | `Const(0)` |
| `Neg(Neg(x))` | `x` |
| `Reciprocal(Reciprocal(x))` | `x` |
| `Exp(Log(x))` | `x` |
| `Log(Exp(x))` | `x` |
| `Reshape(Reshape(x, D1), D2)` | `Reshape(x, D2)` |
| `Permute(Permute(x, P1), P2)` | `Permute(x, compose(P1, P2))` |
| `Add(x, x)` | `Mul(Const(2), x)` |

**Application:** Iterate until fixpoint (no more rules apply). Limit iterations to prevent infinite loops (max 100 passes).

### 4.5 Fusion (Phase 1)

> Not implemented in Phase 0. Specced for forward reference.

**Rule:** Merge chains of elementwise operations into a single fused kernel. Instead of writing intermediate results to memory, compute them in registers.

**Example:**
```
Before (3 memory round-trips):
  n1 = Exp(x)
  n2 = Add(n1, Const(1))
  n3 = Reciprocal(n2)

After fusion (1 kernel, no intermediates in memory):
  n_fused = FusedKernel([Exp, Add(_, Const(1)), Reciprocal])
```

Fusion is the primary optimization for GPU backends, where memory bandwidth is the bottleneck.

---

## 5. Pass Ordering

The standard optimization pipeline:

```
1. grad (if requested)              -- produce backward DAG
2. Algebraic simplification         -- clean up obvious patterns
3. Constant folding                 -- evaluate compile-time constants
4. CSE                              -- merge duplicate subgraphs
5. DCE                              -- remove dead nodes
6. (Repeat 2-5 until fixpoint)      -- simplification can enable more CSE/DCE
7. Fusion (Phase 1)                 -- merge elementwise chains
8. Memory planning                  -- allocate buffers, schedule reuse
9. Code generation                  -- emit target code
```

Steps 2-5 are repeated until no changes occur or a maximum iteration count (10) is reached.

---

## 6. Transformation Composition Table

| Expression | Meaning | Valid? |
|-----------|---------|--------|
| `grad(f)` | Reverse-mode AD of f | Yes (Phase 0) |
| `grad(grad(f))` | Second derivatives | Yes (Phase 0) |
| `grad(f, wrt=[w])` | Gradient w.r.t. specific param | Yes (Phase 0) |
| `vmap(f, axis=a)` | Vectorize over axis a | Phase 2 |
| `jit(f)` | Compile and cache | Phase 2 |
| `jit(grad(f))` | Compile gradient function | Phase 2 |
| `grad(jit(f))` | Same as jit(grad(f)) | Phase 2 |
| `vmap(grad(f), "b")` | Per-example gradients | Phase 2 |
| `grad(vmap(f, "b"))` | Gradient of batched function | Phase 2 |
| `jit(vmap(grad(f)))` | Compiled batched gradients | Phase 2 |

---

## 7. Error Conditions

Transformations can fail. Error types:

- **`non_differentiable_parameter`**: `grad` applied to a function with non-tensor parameters (e.g., integers). The parameter must be excluded via `wrt`.
- **`no_differentiable_path`**: No differentiable path from the specified parameters to the output. Warning, not error (gradient is zero).
- **`dimension_not_found`**: `vmap` axis name doesn't exist in the function's input type.
- **`shape_mismatch_in_jit`**: Cached compiled function called with incompatible shapes (triggers recompilation, not an error to the user).
