# Chelis Language Specification: RISC Primitive Operations

**Version:** 0.1.0-draft
**Status:** Authoritative specification draft

---

## 1. Design Philosophy

Every tensor computation in Chelis is expressed as a DAG of primitive operations. Following tinygrad's insight, the primitive set is deliberately minimal: approximately 20 operations cover all of tensor computation. Higher-level operations -- matmul, conv2d, softmax, layer_norm, attention, and so on -- are all decomposed into these primitives before optimization and code generation.

This design has three consequences:

1. **Backends are simple.** Each backend (C, CUDA, StableHLO) only needs to implement ~20 operations.
2. **Transformations are uniform.** Automatic differentiation needs only ~20 adjoint rules. Fusion needs only ~20 fusion rules.
3. **Correctness is tractable.** The entire primitive set can be formally specified and tested exhaustively.

The primitives are organized into five categories: elementwise unary, elementwise binary, reduction, shape/movement, and memory.

---

## 2. Notation

Throughout this document:

- `D` denotes a dimension list (a set of named dimensions, e.g., `{batch, hidden}`)
- `P` denotes a precision type (e.g., `f32`)
- `tensor[D, P]` is a tensor with dimensions `D` and precision `P`
- `D'` denotes a modified dimension list (result of a shape-changing operation)
- `D \ axis` denotes `D` with the named dimension `axis` removed
- `D + {axis}` denotes `D` with a new named dimension `axis` added
- `g` denotes `grad_out` (the upstream gradient / cotangent) in adjoint rules
- All operations preserve precision unless otherwise stated (Cast is the sole exception)

For numerical examples, tensors are shown as flat arrays or nested arrays in row-major order with their shape annotated.

---

## 3. Elementwise Unary Operations

These operations apply a function independently to each element of a tensor. The output has the same dimensions and precision as the input.

**General typing rule for unary elementwise ops:**

```
      G |- x : tensor[D, P]
      P is in the op's domain (see per-op notes)
      ---------------------------------
      G |- UnaryOp(x) : tensor[D, P]
```

### 3.1 Neg

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise negation. `Neg(x)[i] = -x[i]` for all indices `i`.

**Domain:** P is any numeric type (integer or float). Not valid for `bool`.

**AD adjoint:** `Neg(g)`

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [1.0, -2.0, 3.0]
Output: tensor[{n=3}, f32] = [-1.0, 2.0, -3.0]
```

### 3.2 Exp

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise exponential. `Exp(x)[i] = e^(x[i])`.

**Domain:** P is a float type (`f16`, `bf16`, `f32`, `f64`).

**AD adjoint:** `Mul(g, Exp(x))`

The adjoint reuses the forward value `Exp(x)`. Implementations should cache this value during the forward pass rather than recomputing it in the backward pass. See spec/06-transformations.md Section 1 for details on value caching vs. recomputation (checkpointing).

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [0.0, 1.0, -1.0]
Output: tensor[{n=3}, f32] = [1.0, 2.71828, 0.36788]
```

### 3.3 Log

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise natural logarithm. `Log(x)[i] = ln(x[i])`.

**Domain:** P is a float type. Input values must be positive; behavior on non-positive inputs is undefined (backends may return NaN or -inf).

**AD adjoint:** `Mul(g, Reciprocal(x))`

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [1.0, 2.71828, 0.5]
Output: tensor[{n=3}, f32] = [0.0, 1.0, -0.69315]
```

### 3.4 Sin

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise sine. `Sin(x)[i] = sin(x[i])` (radians).

**Domain:** P is a float type.

**AD adjoint:** `Mul(g, Sin(Add(x, Const(pi/2))))`

**Design decision:** `Cos` is not a separate RISC primitive. The adjoint of `Sin` requires `Cos`, which is expressed as `Sin(x + pi/2)`. This keeps the primitive set minimal at the cost of one extra Add in the backward pass. The constant `pi/2` is `1.5707963267948966` in f64 (cast to the appropriate precision). If profiling reveals this is a performance bottleneck, `Cos` may be promoted to a primitive in a future version.

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [0.0, 1.5708, 3.1416]
Output: tensor[{n=3}, f32] = [0.0, 1.0, 0.0]  (approximately)
```

### 3.5 Sqrt

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise square root. `Sqrt(x)[i] = sqrt(x[i])`.

**Domain:** P is a float type. Input values must be non-negative; behavior on negative inputs is undefined.

**AD adjoint:** `Mul(g, Mul(Const(0.5), Reciprocal(Sqrt(x))))`

Equivalently: `g / (2 * sqrt(x))`. The adjoint is undefined at `x = 0`.

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [4.0, 9.0, 16.0]
Output: tensor[{n=3}, f32] = [2.0, 3.0, 4.0]
```

### 3.6 Reciprocal

**Signature:** `tensor[D, P] -> tensor[D, P]`

**Semantics:** Element-wise reciprocal. `Reciprocal(x)[i] = 1 / x[i]`.

**Domain:** P is any numeric type (not `bool`). Input values must be non-zero; behavior on zero inputs is undefined.

**AD adjoint:** `Neg(Mul(g, Mul(Reciprocal(x), Reciprocal(x))))`

Equivalently: `-g / x^2`.

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [2.0, 4.0, 0.5]
Output: tensor[{n=3}, f32] = [0.5, 0.25, 2.0]
```

### Derived unary operations (not primitives)

These common operations are expressed using the primitives above:

- `Abs(x) = Max(x, Neg(x))` (uses binary Max, Section 4.4)
- `Square(x) = Mul(x, x)` (uses binary Mul, Section 4.2)
- `Cos(x) = Sin(Add(x, Const(pi/2)))`
- `Tanh(x) = Add(Mul(Const(2), Reciprocal(Add(Const(1), Exp(Neg(Mul(Const(2), x)))))), Neg(Const(1)))` (see Section 9.6)

---

## 4. Elementwise Binary Operations

These operations combine two tensors element-wise. Both operands must have the **same set of named dimensions** (order-independent) and the **same precision type**. See spec/04-type-system.md Section 2.2 for the dimension compatibility rules.

**General typing rule for binary elementwise ops:**

```
      G |- a : tensor[D, P]
      G |- b : tensor[D', P']
      S1 = unify_dims(D, D')       -- dimensions must match by name
      S2 = unify(P, P')            -- precision must match
      ---------------------------------
      G |- BinaryOp(a, b) : tensor[S2(S1(D)), S2(S1(P))]
```

When two operands have the same dimension names in different order, the compiler inserts a Permute on the right operand to match the left operand's order. The result takes the dimension order of the left operand.

### 4.1 Add

**Signature:** `(tensor[D, P], tensor[D, P]) -> tensor[D, P]`

**Semantics:** Element-wise addition. `Add(a, b)[i] = a[i] + b[i]`.

**AD adjoint:** `(g, g)`

Both inputs receive the upstream gradient unchanged. This is the simplest adjoint rule: addition distributes gradients to both branches.

**Numerical example:**
```
Input a: tensor[{n=3}, f32] = [1.0, 2.0, 3.0]
Input b: tensor[{n=3}, f32] = [4.0, 5.0, 6.0]
Output:  tensor[{n=3}, f32] = [5.0, 7.0, 9.0]
```

### 4.2 Mul

**Signature:** `(tensor[D, P], tensor[D, P]) -> tensor[D, P]`

**Semantics:** Element-wise multiplication. `Mul(a, b)[i] = a[i] * b[i]`.

**AD adjoint:** `(Mul(g, b), Mul(g, a))`

The gradient for each input is the upstream gradient multiplied by the other input. This requires caching both forward inputs for use in the backward pass.

**Numerical example:**
```
Input a: tensor[{n=3}, f32] = [1.0, 2.0, 3.0]
Input b: tensor[{n=3}, f32] = [4.0, 5.0, 6.0]
Output:  tensor[{n=3}, f32] = [4.0, 10.0, 18.0]
```

### 4.3 CmpLt

**Signature:** `(tensor[D, P], tensor[D, P]) -> tensor[D, bool]`

**Semantics:** Element-wise less-than comparison. `CmpLt(a, b)[i] = (a[i] < b[i])`. The result is a boolean tensor regardless of the input precision.

**Typing rule (specialized):**
```
      G |- a : tensor[D, P]
      G |- b : tensor[D', P']
      S1 = unify_dims(D, D')
      S2 = unify(P, P')
      ---------------------------------
      G |- CmpLt(a, b) : tensor[S2(S1(D)), bool]
```

Note: The output precision is always `bool`, not `P`.

**AD adjoint:** `(Const(0, D, P), Const(0, D, P))`

Comparison is not differentiable. Both inputs receive zero gradients. The compiler emits a warning (not an error) when `CmpLt` appears inside a `grad` scope: "grad: CmpLt encountered; gradient is zero at this point." This follows JAX's convention. An opt-in straight-through estimator may be supported as a future extension.

**Numerical example:**
```
Input a: tensor[{n=3}, f32] = [1.0, 5.0, 3.0]
Input b: tensor[{n=3}, f32] = [2.0, 3.0, 3.0]
Output:  tensor[{n=3}, bool] = [true, false, false]
```

**Derived comparisons (not primitives):**
- `CmpGt(a, b) = CmpLt(b, a)` -- swap operands
- `CmpLe(a, b) = Not(CmpLt(b, a))` -- where `Not(x) = CmpLt(x, Const(true))` on bool tensors
- `CmpGe(a, b) = Not(CmpLt(a, b))`
- `CmpEq(a, b) = Not(Or(CmpLt(a, b), CmpLt(b, a)))` -- neither less nor greater
- `Where(cond, a, b) = Add(Mul(Cast(cond, P), a), Mul(Cast(Not(cond), P), b))`

### 4.4 Max

**Signature:** `(tensor[D, P], tensor[D, P]) -> tensor[D, P]`

**Semantics:** Element-wise maximum. `Max(a, b)[i] = max(a[i], b[i])`.

**AD adjoint:**
```
grad_a = Mul(g, Cast(CmpLt(b, a), P))    -- g where a > b, else 0
grad_b = Mul(g, Cast(CmpLt(a, b), P))    -- g where b > a, else 0
```

When `a[i] == b[i]` (tie), both `CmpLt(b, a)` and `CmpLt(a, b)` are false, so both gradients are zero. This is a valid subgradient for `max`. The gradient is "lost" at ties, which is standard behavior (matching JAX and PyTorch).

**Numerical example:**
```
Input a: tensor[{n=3}, f32] = [1.0, 5.0, 3.0]
Input b: tensor[{n=3}, f32] = [2.0, 3.0, 4.0]
Output:  tensor[{n=3}, f32] = [2.0, 5.0, 4.0]
```

**Derived operations using Max:**
- `ReLU(x) = Max(x, Const(0, D, P))` -- single-operation ReLU
- `Clamp(x, lo, hi) = Max(Min(x, hi), lo)` where `Min(a, b) = Neg(Max(Neg(a), Neg(b)))`

---

## 5. Reduction Operations

Reduction operations collapse one named dimension of a tensor by aggregating along it. The specified axis is removed from the output dimension set.

### 5.1 ReduceSum

**Signature:** `tensor[D, P] -> tensor[D \ axis, P]` where `axis` is a named dimension in `D`

**Parameters:** `axis: DimName` -- the named dimension to reduce over.

**Semantics:** Sum all elements along the specified dimension. If `x` has dimensions `{batch, hidden}` and we reduce along `hidden`, each batch element becomes the sum of its corresponding `hidden`-length slice.

**Typing rule:**
```
      G |- x : tensor[D, P]
      axis in D
      D' = D \ {axis}
      ---------------------------------
      G |- ReduceSum(x, axis) : tensor[D', P]
```

If the reduction removes the last dimension (D' is empty), the result is a 0-dimensional scalar tensor `tensor[P]`.

**AD adjoint:** `Expand(g, axis, size_of(axis))`

The gradient is broadcast back along the reduced axis. Every element along the original axis receives the same upstream gradient, because each element contributed equally (additively) to the sum.

**Numerical example:**
```
Input:  tensor[{batch=2, hidden=3}, f32] = [[1.0, 2.0, 3.0],
                                              [4.0, 5.0, 6.0]]

ReduceSum(x, axis=hidden)
Output: tensor[{batch=2}, f32] = [6.0, 15.0]

ReduceSum(x, axis=batch)
Output: tensor[{hidden=3}, f32] = [5.0, 7.0, 9.0]
```

### 5.2 ReduceMax

**Signature:** `tensor[D, P] -> tensor[D \ axis, P]` where `axis` is a named dimension in `D`

**Parameters:** `axis: DimName`

**Semantics:** Maximum element along the specified dimension.

**Typing rule:** Same structure as ReduceSum.

**AD adjoint:** The gradient is scattered to the position(s) of the maximum element along the reduced axis:

```
adjoint(ReduceMax(x, axis), g) =
  let max_val = ReduceMax(x, axis)                            -- tensor[D', P]
  let max_expanded = Expand(max_val, axis, size_of(axis))     -- tensor[D, P]
  let mask = CmpEq(x, max_expanded)                           -- tensor[D, bool]
  let mask_float = Cast(mask, P)                               -- tensor[D, P]
  let count = ReduceSum(mask_float, axis)                      -- tensor[D', P]
  let count_expanded = Expand(count, axis, size_of(axis))      -- tensor[D, P]
  let normalized = Mul(mask_float, Reciprocal(count_expanded)) -- distribute among ties
  in Mul(Expand(g, axis, size_of(axis)), normalized)
```

If there is a unique maximum, only that position receives the gradient. If there are ties, the gradient is split equally among tied positions.

Note: `CmpEq` is not a primitive but is expressed as `Not(Or(CmpLt(x, max_expanded), CmpLt(max_expanded, x)))`. In practice, a small epsilon tolerance may be used: `CmpLt(Abs(Sub(x, max_expanded)), Const(epsilon))`.

**Numerical example:**
```
Input:  tensor[{batch=2, hidden=3}, f32] = [[1.0, 3.0, 2.0],
                                              [6.0, 4.0, 5.0]]

ReduceMax(x, axis=hidden)
Output: tensor[{batch=2}, f32] = [3.0, 6.0]
```

---

## 6. Shape and Movement Operations

These operations change the shape, layout, or extent of a tensor without computing on element values (except Pad, which introduces zeros).

### 6.1 Reshape

**Signature:** `tensor[D1, P] -> tensor[D2, P]`

**Constraint:** `product(D1) = product(D2)` -- the total number of elements must be preserved.

**Parameters:** `new_dims: Vec<(DimName, usize)>` -- the new named dimensions with sizes.

**Semantics:** Reinterpret the tensor's elements with a new shape. The flat memory layout is unchanged; only the dimension metadata changes. Elements are interpreted in row-major order.

**Typing rule:**
```
      G |- x : tensor[D1, P]
      product(D1) = product(D2)
      ---------------------------------
      G |- Reshape(x, D2) : tensor[D2, P]
```

The compiler verifies the product constraint at compile time when all dimensions are concrete. When symbolic dimensions are involved, the constraint is deferred to runtime (or recorded as a symbolic constraint for later checking).

**AD adjoint:** `Reshape(g, D1)` -- reshape the gradient back to the original shape.

**Numerical example:**
```
Input:  tensor[{n=6}, f32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

Reshape(x, {rows=2, cols=3})
Output: tensor[{rows=2, cols=3}, f32] = [[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]]

Reshape(x, {rows=3, cols=2})
Output: tensor[{rows=3, cols=2}, f32] = [[1.0, 2.0],
                                          [3.0, 4.0],
                                          [5.0, 6.0]]
```

### 6.2 Permute

**Signature:** `tensor[D, P] -> tensor[D', P]` where `D'` is a reordering of `D`

**Parameters:** `order: Vec<DimName>` -- the desired dimension order.

**Semantics:** Reorder the dimensions of a tensor (generalized transpose). Element values are unchanged; only the stride/layout metadata changes. Backends may or may not need to physically move data, depending on their memory model.

**Typing rule:**
```
      G |- x : tensor[D, P]
      D' is a permutation of D   -- same set of dimension names, different order
      ---------------------------------
      G |- Permute(x, D') : tensor[D', P]
```

**AD adjoint:** `Permute(g, D)` -- permute back to the original dimension order (inverse permutation).

**Numerical example:**
```
Input:  tensor[{rows=2, cols=3}, f32] = [[1.0, 2.0, 3.0],
                                          [4.0, 5.0, 6.0]]

Permute(x, [cols, rows])
Output: tensor[{cols=3, rows=2}, f32] = [[1.0, 4.0],
                                          [2.0, 5.0],
                                          [3.0, 6.0]]
```

**Note:** With named dimensions, Permute is less necessary than in positional systems, since elementwise operations match dimensions by name, not position. Permute is primarily needed for memory layout optimization and for operations that are sensitive to physical layout (e.g., cache-friendly access patterns).

### 6.3 Expand

**Signature:** `tensor[D, P] -> tensor[D', P]`

**Parameters:** `axis: DimName, size: usize` -- the new dimension name and its size.

**Semantics:** Broadcast a tensor by logically repeating its data along a new dimension. Two modes:

1. **Add a new dimension:** If `axis` is not in `D`, add it with the specified size. Every slice along the new dimension is identical.
2. **Expand a size-1 dimension:** If `axis` is in `D` with size 1, expand it to the specified size.

This is the **only** way to broadcast in Chelis. There is no implicit broadcasting.

**Typing rule:**
```
      G |- x : tensor[D, P]
      axis not in D (or axis in D with size 1)
      D' = D + {axis: size}  (or D with axis size changed to size)
      ---------------------------------
      G |- Expand(x, axis, size) : tensor[D', P]
```

**AD adjoint:** `ReduceSum(g, axis=axis)` -- sum over the expanded dimension.

Broadcasting in the forward pass becomes summation in the backward pass. This is because each element of the original tensor contributed to `size` elements of the expanded tensor, and the chain rule requires summing all those contributions.

**Numerical example:**
```
Input:  tensor[{hidden=3}, f32] = [1.0, 2.0, 3.0]

Expand(x, batch, 2)
Output: tensor[{batch=2, hidden=3}, f32] = [[1.0, 2.0, 3.0],
                                              [1.0, 2.0, 3.0]]
```

### 6.4 Pad

**Signature:** `tensor[D, P] -> tensor[D', P]`

**Parameters:** `padding: Vec<(DimName, usize, usize)>` -- for each axis: `(axis_name, pad_before, pad_after)`.

**Semantics:** Add zero-valued elements around the edges of a tensor along specified axes. The padded regions are filled with the additive identity (0 for numeric types, `false` for bool).

**Typing rule:**
```
      G |- x : tensor[D, P]
      For each padded axis a_i:
        D'[a_i] = D[a_i] + pad_before_i + pad_after_i
      All other dimensions unchanged
      ---------------------------------
      G |- Pad(x, padding) : tensor[D', P]
```

**AD adjoint:** `Shrink(g, inverse_spec)` -- crop away the padded regions.

The gradient in the padded (zero) regions is discarded, since those elements were not functions of the input.

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [1.0, 2.0, 3.0]

Pad(x, [(n, 1, 2)])
Output: tensor[{n=6}, f32] = [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
```

### 6.5 Shrink

**Signature:** `tensor[D, P] -> tensor[D', P]`

**Parameters:** `ranges: Vec<(DimName, usize, usize)>` -- for each axis: `(axis_name, start, end)` where the result includes indices `[start, end)`.

**Semantics:** Extract a contiguous sub-tensor (slice/crop) along specified axes.

**Typing rule:**
```
      G |- x : tensor[D, P]
      For each shrunk axis a_i:
        0 <= start_i < end_i <= D[a_i]
        D'[a_i] = end_i - start_i
      All other dimensions unchanged
      ---------------------------------
      G |- Shrink(x, ranges) : tensor[D', P]
```

**AD adjoint:** `Pad(g, inverse_spec)` -- pad the gradient with zeros in the regions that were cropped away.

**Numerical example:**
```
Input:  tensor[{n=5}, f32] = [1.0, 2.0, 3.0, 4.0, 5.0]

Shrink(x, [(n, 1, 4)])
Output: tensor[{n=3}, f32] = [2.0, 3.0, 4.0]
```

### 6.6 Stride

**Signature:** `tensor[D, P] -> tensor[D', P]`

**Parameters:** `strides: Vec<(DimName, usize)>` -- for each axis: `(axis_name, step_size)`.

**Semantics:** Take every n-th element along specified axes, starting from index 0. The dimension size is divided by the stride (floor division).

**Typing rule:**
```
      G |- x : tensor[D, P]
      For each strided axis a_i:
        D'[a_i] = floor(D[a_i] / stride_i)
      All other dimensions unchanged
      ---------------------------------
      G |- Stride(x, strides) : tensor[D', P]
```

**AD adjoint:** The adjoint inserts zeros between gradient elements ("upsampling"), then crops to the original size:

```
adjoint(Stride(x, [(axis, s)]), g) =
  -- Interleave g with (s-1) zeros along axis
  let g_reshaped = Reshape(g, ... insert unit dim ...)
  let g_padded = Pad(g_reshaped, ... (s-1) zeros after each element ...)
  let g_flat = Reshape(g_padded, ... flatten back ...)
  in Shrink(g_flat, [(axis, 0, original_size)])
```

The exact sequence depends on the stride value and original size.

**Numerical example:**
```
Input:  tensor[{n=6}, f32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

Stride(x, [(n, 2)])
Output: tensor[{n=3}, f32] = [1.0, 3.0, 5.0]

Stride(x, [(n, 3)])
Output: tensor[{n=2}, f32] = [1.0, 4.0]
```

---

## 7. Memory Operations

### 7.1 Const

**Signature:** `(value, dims, precision) -> tensor[D, P]`

**Semantics:** Create a tensor filled with a single constant value, or from a literal array of values. Constants are known at compile time and are candidates for constant folding.

**Typing rule:**
```
      value is a numeric literal compatible with P
      D, P are specified
      ---------------------------------
      G |- Const(value, D, P) : tensor[D, P]
```

**AD adjoint:** No gradient (constants have no inputs to propagate to).

**Numerical example:**
```
Const(0.0, {n=3}, f32)    -> tensor[{n=3}, f32] = [0.0, 0.0, 0.0]
Const(1.0, {r=2, c=2}, f32) -> tensor[{r=2, c=2}, f32] = [[1.0, 1.0], [1.0, 1.0]]
```

### 7.2 Load

**Signature:** `buffer_id -> tensor[D, P]`

**Semantics:** Load a tensor from a named buffer. Buffers represent external data sources: function parameters, model weights, input data. The shape and precision are determined by the buffer's declared type.

**Typing rule:**
```
      buffer_id is a valid buffer name
      D, P are the buffer's declared dimensions and precision
      ---------------------------------
      G |- Load(buffer_id) : tensor[D, P]
```

**AD adjoint:** Load nodes represent function inputs. When computing `grad(f)`, the gradient with respect to a loaded parameter is accumulated (via Add nodes in the backward DAG) and returned as part of the gradient output tuple.

### 7.3 Store

**Signature:** `(tensor[D, P], buffer_id) -> unit`

**Semantics:** Store a tensor to a named buffer. This is the mechanism for writing function outputs and intermediate results that need to persist.

**Typing rule:**
```
      G |- x : tensor[D, P]
      buffer_id is a valid buffer name with matching D, P
      ---------------------------------
      G |- Store(x, buffer_id) : unit
```

**AD adjoint:** Store is an output operation. In the backward pass, the gradient for the stored value comes from the loss function's output gradient (typically initialized to `Const(1.0)` for scalar loss).

### 7.4 Cast

**Signature:** `tensor[D, P1] -> tensor[D, P2]`

**Parameters:** `target_precision: Precision`

**Semantics:** Convert every element from precision `P1` to precision `P2`. Dimensions are preserved unchanged. This is the **only** operation that changes precision. Conversion semantics follow the rules in spec/04-type-system.md Section 3.4.

**Typing rule:**
```
      G |- x : tensor[D, P1]
      P2 is a valid precision type
      ---------------------------------
      G |- Cast(x, P2) : tensor[D, P2]
```

**AD adjoint:** `Cast(g, P1)` -- cast the gradient back to the original precision.

This is mathematically correct because cast is treated as an identity function on the real-valued computation graph; the precision change is a representational detail that the chain rule is agnostic to.

**Numerical example:**
```
Input:  tensor[{n=3}, f32] = [1.5, 2.7, 3.1]

Cast(x, i32)
Output: tensor[{n=3}, i32] = [1, 2, 3]    -- truncated toward zero

Cast(x, bf16)
Output: tensor[{n=3}, bf16] = [1.5, 2.703125, 3.09375]  -- bf16 rounding
```

---

## 8. Complete AD Adjoint Reference Table

For quick reference, here is the complete table of adjoint rules. In all rules, `g` denotes `grad_out` (the upstream gradient from downstream consumers), `a` and `b` are forward inputs, and `P` is the precision type.

| Operation | Forward | Adjoint (gradient w.r.t. each input) |
|-----------|---------|--------------------------------------|
| `Neg(x)` | `-x` | `Neg(g)` |
| `Exp(x)` | `e^x` | `Mul(g, Exp(x))` |
| `Log(x)` | `ln(x)` | `Mul(g, Reciprocal(x))` |
| `Sin(x)` | `sin(x)` | `Mul(g, Sin(Add(x, Const(pi/2))))` |
| `Sqrt(x)` | `sqrt(x)` | `Mul(g, Mul(Const(0.5), Reciprocal(Sqrt(x))))` |
| `Reciprocal(x)` | `1/x` | `Neg(Mul(g, Mul(Reciprocal(x), Reciprocal(x))))` |
| `Add(a, b)` | `a + b` | `(g, g)` |
| `Mul(a, b)` | `a * b` | `(Mul(g, b), Mul(g, a))` |
| `CmpLt(a, b)` | `a < b` | `(Const(0), Const(0))` -- non-differentiable |
| `Max(a, b)` | `max(a, b)` | `(Mul(g, Cast(CmpLt(b, a), P)),` `Mul(g, Cast(CmpLt(a, b), P)))` |
| `ReduceSum(x, axis)` | `sum` | `Expand(g, axis, size)` |
| `ReduceMax(x, axis)` | `max` | scatter `g` to max positions (Section 5.2) |
| `Reshape(x, D2)` | reshape | `Reshape(g, D1)` |
| `Permute(x, D')` | transpose | `Permute(g, D)` (inverse permutation) |
| `Expand(x, dim, size)` | broadcast | `ReduceSum(g, axis=dim)` |
| `Pad(x, spec)` | zero-pad | `Shrink(g, inverse_spec)` |
| `Shrink(x, spec)` | slice | `Pad(g, inverse_spec)` |
| `Stride(x, strides)` | subsample | upsample with zeros, trim to original |
| `Cast(x, P2)` | precision | `Cast(g, P1)` |
| `Const(v, D, P)` | constant | N/A (no inputs) |
| `Load(buf)` | load | accumulated and returned |
| `Store(x, buf)` | store | from output gradient |

**Key observations:**

1. Movement operations (Reshape, Permute, Expand, Pad, Shrink, Stride) have adjoints that are the "inverse" movement. This is a general principle: if the forward pass moves data from position A to position B, the backward pass moves gradients from position B back to position A.

2. Expand and ReduceSum are adjoint pairs. Broadcasting in the forward pass becomes summation in the backward pass, and vice versa.

3. Pad and Shrink are adjoint pairs. Padding in the forward pass becomes cropping in the backward pass.

---

## 9. Lowering Standard Operations to RISC Primitives

This section shows how common ML operations decompose into the RISC primitives. These lowerings are performed by the Lower stage of the compiler pipeline (see spec/01-nomenclature.md).

### 9.1 Matmul

`matmul(A: tensor[{m, k}, P], B: tensor[{k, n}, P]) -> tensor[{m, n}, P]`

```
-- Step 1: Reshape to 3D for broadcasting
A' = Reshape(A, {m, k, _n=1})                -- tensor[{m, k, _n=1}, P]

-- Step 2: Reshape B to 3D
B' = Reshape(B, {_m=1, k, n})                -- tensor[{_m=1, k, n}, P]

-- Step 3: Broadcast both to [m, k, n]
A'' = Expand(A', _n -> n, size_n)             -- tensor[{m, k, n}, P]
B'' = Expand(B', _m -> m, size_m)             -- tensor[{m, k, n}, P]

-- Step 4: Elementwise multiply
C = Mul(A'', B'')                             -- tensor[{m, k, n}, P]

-- Step 5: Sum over contraction dimension
result = ReduceSum(C, axis=k)                 -- tensor[{m, n}, P]
```

For batched matmul `A: tensor[{batch, m, k}, P]` and `B: tensor[{batch, k, n}, P]`, the batch dimension is carried through unchanged at every step.

**Complexity:** O(m * k * n) multiplications and additions, same as direct matmul. The decomposition into primitives does not change asymptotic complexity but enables the fusion pass to optimize memory access patterns.

### 9.2 Softmax

`softmax(x: tensor[{..d, axis}, P]) -> tensor[{..d, axis}, P]`

Numerically stable lowering (subtracts max for stability):

```
-- Step 1: Shift for numerical stability
max_val = ReduceMax(x, axis)                            -- tensor[{..d}, P]
max_expanded = Expand(max_val, axis, size_of(axis))     -- tensor[{..d, axis}, P]
shifted = Add(x, Neg(max_expanded))                     -- tensor[{..d, axis}, P]

-- Step 2: Exponentiate
exps = Exp(shifted)                                     -- tensor[{..d, axis}, P]

-- Step 3: Normalize
sum_exps = ReduceSum(exps, axis)                        -- tensor[{..d}, P]
sum_expanded = Expand(sum_exps, axis, size_of(axis))    -- tensor[{..d, axis}, P]
result = Mul(exps, Reciprocal(sum_expanded))            -- tensor[{..d, axis}, P]
```

**Note:** Without the max-subtraction (Step 1), `Exp` can overflow for large inputs. The shifted version is mathematically equivalent but numerically stable.

### 9.3 ReLU

`relu(x: tensor[D, P]) -> tensor[D, P]`

```
result = Max(x, Const(0, D, P))
```

One primitive operation. The AD adjoint follows automatically from Max's adjoint rule.

### 9.4 Sigmoid

`sigmoid(x: tensor[D, P]) -> tensor[D, P]`

```
neg_x = Neg(x)                                  -- -x
exp_neg = Exp(neg_x)                             -- e^(-x)
one_plus = Add(Const(1.0, D, P), exp_neg)        -- 1 + e^(-x)
result = Reciprocal(one_plus)                    -- 1 / (1 + e^(-x))
```

Four primitive operations. The AD adjoint follows automatically: `sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))`.

### 9.5 Tanh

`tanh(x: tensor[D, P]) -> tensor[D, P]`

Using `tanh(x) = 2 * sigmoid(2x) - 1`:

```
two_x = Mul(Const(2.0, D, P), x)                -- 2x
sig = sigmoid(two_x)                             -- sigmoid(2x), uses sigmoid lowering
two_sig = Mul(Const(2.0, D, P), sig)             -- 2 * sigmoid(2x)
result = Add(two_sig, Neg(Const(1.0, D, P)))     -- 2 * sigmoid(2x) - 1
```

### 9.6 Layer Normalization

`layer_norm(x, gamma, beta)` where:
- `x: tensor[{..batch_dims, features}, P]`
- `gamma: tensor[{features}, P]`
- `beta: tensor[{features}, P]`

```
-- Step 1: Compute mean along features
sum_x = ReduceSum(x, axis=features)                         -- tensor[{..batch}, P]
mean = Mul(sum_x, Reciprocal(Const(size_features, {..batch}, P)))  -- tensor[{..batch}, P]
mean_exp = Expand(mean, features, size_features)              -- tensor[{..batch, features}, P]

-- Step 2: Center
centered = Add(x, Neg(mean_exp))                              -- tensor[{..batch, features}, P]

-- Step 3: Compute variance
sq = Mul(centered, centered)                                   -- tensor[{..batch, features}, P]
sum_sq = ReduceSum(sq, axis=features)                          -- tensor[{..batch}, P]
var = Mul(sum_sq, Reciprocal(Const(size_features, {..batch}, P)))  -- tensor[{..batch}, P]

-- Step 4: Normalize
var_eps = Add(var, Const(1e-5, {..batch}, P))                  -- add epsilon for stability
inv_std = Reciprocal(Sqrt(var_eps))                            -- tensor[{..batch}, P]
inv_std_exp = Expand(inv_std, features, size_features)          -- tensor[{..batch, features}, P]
normalized = Mul(centered, inv_std_exp)                         -- tensor[{..batch, features}, P]

-- Step 5: Scale and shift (affine transform)
gamma_exp = Expand(gamma, ..batch_dims)                         -- tensor[{..batch, features}, P]
beta_exp = Expand(beta, ..batch_dims)                           -- tensor[{..batch, features}, P]
result = Add(Mul(normalized, gamma_exp), beta_exp)              -- tensor[{..batch, features}, P]
```

### 9.7 Conv2d (im2col approach)

`conv2d(input, kernel)` where:
- `input: tensor[{batch, in_ch, h, w}, P]`
- `kernel: tensor[{out_ch, in_ch, kh, kw}, P]`
- Output: `tensor[{batch, out_ch, out_h, out_w}, P]`

Where `out_h = h - kh + 1` (no padding) or `out_h = h` (same padding).

The im2col approach converts convolution into a matrix multiplication:

```
-- Step 1: Extract patches (im2col)
-- For each output position (oh, ow), extract the corresponding
-- (in_ch, kh, kw) patch from the input.
--
-- This is accomplished with a series of Pad, Shrink, Reshape, and Expand:
-- For each kernel offset (dy, dx) in {0..kh-1} x {0..kw-1}:
--   shifted_dy_dx = Shrink(input, [(h, dy, dy+out_h), (w, dx, dx+out_w)])
--   -- shape: tensor[{batch, in_ch, out_h, out_w}, P]
--
-- Stack all kh*kw shifted views and reshape into:
--   patches: tensor[{batch, out_h*out_w, in_ch*kh*kw}, P]

-- Step 2: Reshape kernel
kernel_matrix = Reshape(kernel, {out_ch, in_ch_kh_kw})
-- shape: tensor[{out_ch, in_ch*kh*kw}, P]

-- Step 3: Batched matmul
-- patches @ kernel^T -> tensor[{batch, out_h*out_w, out_ch}, P]
-- which uses the matmul lowering from Section 9.1
result_flat = matmul(patches, Permute(kernel_matrix, [in_ch_kh_kw, out_ch]))

-- Step 4: Reshape to output format
result = Reshape(result_flat, {batch, out_ch, out_h, out_w})
```

For strided or dilated convolutions, insert Stride operations before the Shrink steps. For padded convolutions, apply Pad to the input before extracting patches.

### 9.8 Cross-Entropy Loss

`cross_entropy(logits, targets)` where:
- `logits: tensor[{batch, classes}, P]`
- `targets: tensor[{batch}, i64]` (class indices)
- Output: `tensor[{}, P]` (scalar loss)

```
-- Step 1: Log-softmax (numerically stable)
-- Using the softmax lowering from Section 9.2, then Log:
max_val = ReduceMax(logits, axis=classes)
max_exp = Expand(max_val, classes, num_classes)
shifted = Add(logits, Neg(max_exp))
log_sum_exp = Log(ReduceSum(Exp(shifted), axis=classes))
log_probs = Add(shifted, Neg(Expand(log_sum_exp, classes, num_classes)))

-- Step 2: Gather log-probabilities for correct classes
-- This requires one-hot encoding of targets, then dot product:
--   one_hot: tensor[{batch, classes}, P] -- 1 at target index, 0 elsewhere
--   selected = ReduceSum(Mul(log_probs, one_hot), axis=classes)
-- The one-hot itself is constructed from CmpEq of target indices with a range.

-- Step 3: Negate and average
loss = Neg(Mul(
  ReduceSum(selected, axis=batch),
  Reciprocal(Const(batch_size, {}, P))
))
```

---

## 10. Fusion Rules

Elementwise operations that are adjacent in the DAG (no intervening reductions or shape changes) can be fused into a single kernel. The fusion pass identifies chains of elementwise operations and merges them.

**Fusible chains:** Sequences of unary and binary elementwise ops where every intermediate result has the same dimension set.

Examples:
- `Exp(Add(x, Neg(y)))` -- three ops, one kernel
- `Mul(Exp(x), Reciprocal(y))` -- three ops, one kernel
- `Max(x, Const(0))` -- ReLU, one kernel
- `Reciprocal(Add(Const(1), Exp(Neg(x))))` -- sigmoid, one kernel

**Fusion barriers (operations that break elementwise chains):**
- Reduction operations (ReduceSum, ReduceMax) -- change the parallelism pattern
- Shape operations (Reshape, Permute) -- may or may not break chains depending on the backend
- Memory operations (Load, Store) -- side effects require materialization

Fusion rules are applied during the Transform stage. See spec/06-transformations.md Section 4 for details.

---

## 11. DAG Invariants

A valid RISC DAG must satisfy the following invariants. The compiler verifies these after each transformation pass.

1. **Acyclicity.** The graph contains no cycles. It is a directed acyclic graph.
2. **Type consistency.** At every edge, the output type of the source node matches the expected input type of the destination node.
3. **Dimension agreement.** Binary elementwise ops require identical named dimension sets on both inputs.
4. **Precision agreement.** Binary ops require identical precision on both inputs. Cast is the only operation that changes precision.
5. **Element conservation.** Reshape preserves total element count: `product(D1) = product(D2)`.
6. **Dimension uniqueness.** No tensor type contains duplicate dimension names.
7. **Single output.** Each node produces exactly one tensor output (except Store, which produces `unit`).
8. **Reduction validity.** The axis argument of ReduceSum/ReduceMax must name a dimension that exists in the input tensor.
9. **Expand validity.** The axis argument of Expand must name a dimension that does not exist in the input tensor (or exists with size 1).

---

## 12. Interaction with Named Dimensions

All RISC primitives respect the named dimension system defined in spec/04-type-system.md Section 2. Key consequences:

1. **Elementwise ops** require that both operands have the same set of dimension names (order-independent). When the dimension order differs, the compiler inserts a Permute before the operation to align layouts.

2. **Reductions** take a dimension name (not a positional integer index) as the axis argument. This eliminates the common bug class of reducing along the wrong axis due to positional confusion.

3. **Expand** adds a named dimension. The name must not already exist in the tensor's dimension set (invariant 9).

4. **Reshape** replaces the entire dimension list. The new dimensions can have completely different names. This is the primary mechanism for "renaming" dimensions or changing the logical interpretation of tensor axes.

5. **Matmul lowering** (Section 9.1) uses dimension names to identify the contraction dimension (`k`), making the operation unambiguous regardless of the physical layout order.

6. **Error messages** from type checking reference dimension names, not positional indices, making them significantly more readable: "dimension 'hidden' does not match 'seq_len'" vs "axis 1 does not match axis 1."
