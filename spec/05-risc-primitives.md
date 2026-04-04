# RISC Primitives

The RISC DAG is the core intermediate representation of Chelis. Every tensor computation is expressed as a directed acyclic graph of the primitive operations defined in this document.

**Design philosophy:** Following tinygrad's insight, a small set of primitives is sufficient to express all tensor computations. Higher-level operations (matmul, conv2d, softmax, layer_norm, etc.) are decomposed into these primitives. This makes backends simpler (implement ~20 ops, not hundreds) and transformations uniform (AD adjoint rules for ~20 ops cover everything).

## Notation

- `D` — a set of named dimensions, e.g., `{batch, hidden}`
- `P` — a precision type: `f16`, `bf16`, `f32`, `f64`
- `tensor[D, P]` — a tensor with dimensions D and precision P
- `D \ axis` — D with the named dimension `axis` removed
- `D ∪ {axis}` — D with a new named dimension `axis` added

---

## 1. Elementwise Unary Operations

These operate independently on each element. Shape and dimensions are preserved.

### Neg

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Neg(x)[i] = -x[i]`
- **Example:** `Neg([1.0, -2.0, 3.0]) = [-1.0, 2.0, -3.0]`
- **Adjoint:** `Neg(grad_out)`

### Exp

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Exp(x)[i] = e^(x[i])`
- **Example:** `Exp([0.0, 1.0, 2.0]) = [1.0, 2.718, 7.389]`
- **Adjoint:** `Mul(grad_out, Exp(x))` (reuses forward value)

### Log

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Log(x)[i] = ln(x[i])`
- **Domain:** x[i] > 0. Behavior on non-positive inputs is undefined.
- **Example:** `Log([1.0, 2.718, 7.389]) = [0.0, 1.0, 2.0]`
- **Adjoint:** `Mul(grad_out, Reciprocal(x))`

### Sin

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Sin(x)[i] = sin(x[i])`
- **Example:** `Sin([0.0, 1.571, 3.142]) ≈ [0.0, 1.0, 0.0]`
- **Adjoint:** `Mul(grad_out, Cos(x))` where `Cos(x) = Sin(Add(x, Const(π/2)))`
- **Note:** Cos is not a separate primitive. It is expressed via `Sin` with a phase shift.

### Sqrt

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Sqrt(x)[i] = √(x[i])`
- **Domain:** x[i] ≥ 0.
- **Example:** `Sqrt([1.0, 4.0, 9.0]) = [1.0, 2.0, 3.0]`
- **Adjoint:** `Mul(grad_out, Mul(Const(0.5), Reciprocal(Sqrt(x))))`

### Reciprocal

- **Signature:** `tensor[D, P] → tensor[D, P]`
- **Semantics:** `Reciprocal(x)[i] = 1 / x[i]`
- **Domain:** x[i] ≠ 0.
- **Example:** `Reciprocal([2.0, 4.0, 0.5]) = [0.5, 0.25, 2.0]`
- **Adjoint:** `Neg(Mul(grad_out, Mul(Reciprocal(x), Reciprocal(x))))`
  - Equivalently: `-grad_out / x²`

---

## 2. Elementwise Binary Operations

Both operands must have identical named dimensions and precision.

### Add

- **Signature:** `(tensor[D, P], tensor[D, P]) → tensor[D, P]`
- **Semantics:** `Add(a, b)[i] = a[i] + b[i]`
- **Example:** `Add([1, 2, 3], [4, 5, 6]) = [5, 7, 9]`
- **Adjoint:** `(grad_out, grad_out)` — gradient flows equally to both inputs

### Mul

- **Signature:** `(tensor[D, P], tensor[D, P]) → tensor[D, P]`
- **Semantics:** `Mul(a, b)[i] = a[i] * b[i]`
- **Example:** `Mul([1, 2, 3], [4, 5, 6]) = [4, 10, 18]`
- **Adjoint:** `(Mul(grad_out, b), Mul(grad_out, a))`

### CmpLt

- **Signature:** `(tensor[D, P], tensor[D, P]) → tensor[D, bool]`
- **Semantics:** `CmpLt(a, b)[i] = a[i] < b[i]`
- **Example:** `CmpLt([1, 5, 3], [2, 4, 3]) = [true, false, false]`
- **Adjoint:** `(Const(0), Const(0))` — not differentiable. Compiler emits a warning when `grad` encounters `CmpLt`.

### Max

- **Signature:** `(tensor[D, P], tensor[D, P]) → tensor[D, P]`
- **Semantics:** `Max(a, b)[i] = max(a[i], b[i])`
- **Example:** `Max([1, 5, 3], [2, 4, 3]) = [2, 5, 3]`
- **Adjoint:** Gradient routes to whichever input was larger.
  - For input a: `Mul(grad_out, Cast(CmpLt(b, a), P))` — 1 where a > b, 0 elsewhere
  - For input b: `Mul(grad_out, Cast(CmpLt(a, b), P))` — 1 where b > a, 0 elsewhere
  - When a[i] == b[i]: gradient splits equally (each gets 0.5 * grad_out)

**Derived operations (not primitives):**
- `Sub(a, b) = Add(a, Neg(b))`
- `Div(a, b) = Mul(a, Reciprocal(b))`
- `Where(cond, a, b) = Add(Mul(Cast(cond, P), a), Mul(Cast(Not(cond), P), b))` where `Not(cond) = CmpLt(cond, Const(1))`

---

## 3. Reduction Operations

Reductions collapse one named dimension, producing a tensor with fewer dimensions.

### ReduceSum

- **Signature:** `tensor[D, P] → tensor[D \ axis, P]` where `axis ∈ D`
- **Parameters:** `axis: DimName` — which named dimension to reduce
- **Semantics:** Sum all elements along the named axis.
- **Example:** Given `x: tensor[{batch=2, features=3}, f32] = [[1,2,3],[4,5,6]]`:
  - `ReduceSum(x, axis=features)` → `tensor[{batch=2}, f32] = [6, 15]`
  - `ReduceSum(x, axis=batch)` → `tensor[{features=3}, f32] = [5, 7, 9]`
- **Adjoint:** `Expand(grad_out, axis, size)` — broadcast the gradient back along the reduced axis.

### ReduceMax

- **Signature:** `tensor[D, P] → tensor[D \ axis, P]` where `axis ∈ D`
- **Parameters:** `axis: DimName`
- **Semantics:** Maximum element along the named axis.
- **Example:** Given `x: tensor[{batch=2, features=3}, f32] = [[1,2,3],[4,5,6]]`:
  - `ReduceMax(x, axis=features)` → `[3, 6]`
- **Adjoint:** Gradient goes only to the position(s) of the maximum value. Formally:
  - Let `m = ReduceMax(x, axis)` (the forward result, expanded back)
  - `mask = Cast(CmpLt(Sub(Expand(m, axis), x), Const(epsilon)), P)` — 1 at max positions
  - `count = ReduceSum(mask, axis)` — number of max positions (for ties)
  - Adjoint: `Mul(Expand(grad_out, axis), Mul(mask, Reciprocal(Expand(count, axis))))`

---

## 4. Shape and Movement Operations

These change tensor layout without computing on element values (except Pad, which introduces zeros).

### Reshape

- **Signature:** `tensor[D1, P] → tensor[D2, P]` where `product(D1) == product(D2)`
- **Parameters:** `new_dims: Vec<(DimName, usize)>` — new named dimensions with sizes
- **Semantics:** Reinterpret the flat buffer with new dimension names and sizes. Total element count must match.
- **Example:** `Reshape(tensor[{a=6}, f32], {b=2, c=3})` → `tensor[{b=2, c=3}, f32]`
- **Adjoint:** `Reshape(grad_out, original_dims)` — reshape gradient back to original shape.

### Permute

- **Signature:** `tensor[D, P] → tensor[D', P]` where D' is a reordering of D
- **Parameters:** `order: Vec<DimName>` — the desired dimension order
- **Semantics:** Reorder dimensions (generalized transpose). Does not change element values, only strides/layout.
- **Example:** `Permute(tensor[{batch, seq, hidden}, f32], [hidden, batch, seq])` → `tensor[{hidden, batch, seq}, f32]`
- **Adjoint:** `Permute(grad_out, inverse_order)` — permute back to original order.
- **Note:** With named dimensions, Permute is less necessary than in positional systems, since operations match by name not position. Still needed for memory layout optimization.

### Expand

- **Signature:** `tensor[D, P] → tensor[D ∪ {axis}, P]` (add new dim) or `tensor[D, P] → tensor[D', P]` (expand size-1 dim)
- **Parameters:** `axis: DimName, size: usize`
- **Semantics:** Broadcast. Either:
  1. Add a new dimension of the given size (repeating the data along it), or
  2. Expand an existing size-1 dimension to the given size
- **Example:** `Expand(tensor[{features=3}, f32] = [1,2,3], axis=batch, size=2)` → `tensor[{batch=2, features=3}, f32] = [[1,2,3],[1,2,3]]`
- **Adjoint:** `ReduceSum(grad_out, axis)` — sum over the expanded axis.
- **Critical note:** This is the ONLY way to broadcast in Chelis. There is no implicit broadcasting.

### Pad

- **Signature:** `tensor[D, P] → tensor[D', P]` where D' has larger sizes on padded dims
- **Parameters:** `padding: Vec<(DimName, usize, usize)>` — (axis, pad_before, pad_after)
- **Semantics:** Add zeros around the tensor along specified axes.
- **Example:** `Pad(tensor[{n=3}, f32] = [1,2,3], [(n, 1, 1)])` → `tensor[{n=5}, f32] = [0,1,2,3,0]`
- **Adjoint:** `Shrink(grad_out, ...)` — crop away the padded regions.

### Shrink

- **Signature:** `tensor[D, P] → tensor[D', P]` where D' has smaller sizes on shrunk dims
- **Parameters:** `ranges: Vec<(DimName, usize, usize)>` — (axis, start, end) for each axis to crop
- **Semantics:** Take a contiguous slice along specified axes.
- **Example:** `Shrink(tensor[{n=5}, f32] = [0,1,2,3,0], [(n, 1, 4)])` → `tensor[{n=3}, f32] = [1,2,3]`
- **Adjoint:** `Pad(grad_out, ...)` — pad gradient with zeros in the cropped regions.

### Stride

- **Signature:** `tensor[D, P] → tensor[D', P]` where D' has smaller sizes on strided dims
- **Parameters:** `strides: Vec<(DimName, usize)>` — (axis, step_size)
- **Semantics:** Take every n-th element along specified axes.
- **Example:** `Stride(tensor[{n=6}, f32] = [0,1,2,3,4,5], [(n, 2)])` → `tensor[{n=3}, f32] = [0,2,4]`
- **Adjoint:** Insert zeros between gradient elements (upsampling), then shrink to original size.

---

## 5. Memory Operations

### Const

- **Signature:** `(value, dims, precision) → tensor[D, P]`
- **Semantics:** Create a tensor filled with the given constant value.
- **Example:** `Const(0.0, {batch=2, features=3}, f32)` → `[[0,0,0],[0,0,0]]`
- **Adjoint:** None (no inputs to propagate to).

### Load

- **Signature:** `buffer_id → tensor[D, P]`
- **Semantics:** Load a tensor from a named external buffer (function parameter, stored data).
- **Adjoint:** The gradient of a loaded input is accumulated and returned as part of the grad output.

### Store

- **Signature:** `(tensor[D, P], buffer_id) → unit`
- **Semantics:** Store a tensor to a named buffer (function output, side effect).
- **Adjoint:** Not applicable (Store is an output operation; its gradient comes from the loss).

### Cast

- **Signature:** `tensor[D, P1] → tensor[D, P2]`
- **Parameters:** `target_precision: Precision`
- **Semantics:** Convert each element to the target precision. This is the ONLY way to change precision. No implicit casts anywhere.
- **Example:** `Cast(tensor[{n=3}, f32] = [1.0, 2.5, 3.7], bf16)` → `tensor[{n=3}, bf16] = [1.0, 2.5, 3.7]` (with bf16 rounding)
- **Adjoint:** `Cast(grad_out, P1)` — cast gradient back to original precision.

---

## 6. Lowering Standard Operations to RISC Primitives

This section shows how common ML operations decompose into the primitives above. These lowerings are performed by the `chelis-ir` crate during the lowering pass.

### matmul

`matmul(A: tensor[{m, k}, P], B: tensor[{k, n}, P]) → tensor[{m, n}, P]`

Lowering:
```
a_expanded = Reshape(A, {m, k, _1=1})       -- [m, k, 1]
a_broadcast = Expand(a_expanded, n, size_n)  -- [m, k, n]
b_expanded = Reshape(B, {_1=1, k, n})        -- [1, k, n]
b_broadcast = Expand(b_expanded, m, size_m)  -- [m, k, n]
product = Mul(a_broadcast, b_broadcast)      -- [m, k, n]
result = ReduceSum(product, axis=k)          -- [m, n]
```

For batched matmul, the batch dimensions pass through unchanged.

### softmax

`softmax(x: tensor[{..., axis}, P]) → tensor[{..., axis}, P]`

Numerically stable lowering:
```
max_val = ReduceMax(x, axis)                             -- [...] 
max_expanded = Expand(max_val, axis, size_axis)          -- [..., axis]
shifted = Add(x, Neg(max_expanded))                      -- [..., axis] (subtract max for stability)
exp_shifted = Exp(shifted)                                -- [..., axis]
sum_exp = ReduceSum(exp_shifted, axis)                    -- [...]
sum_expanded = Expand(sum_exp, axis, size_axis)           -- [..., axis]
result = Mul(exp_shifted, Reciprocal(sum_expanded))       -- [..., axis]
```

### relu

`relu(x: tensor[D, P]) → tensor[D, P]`

```
result = Max(x, Const(0, D, P))
```

### sigmoid

`sigmoid(x: tensor[D, P]) → tensor[D, P]`

```
neg_x = Neg(x)
exp_neg = Exp(neg_x)
one_plus = Add(Const(1, D, P), exp_neg)
result = Reciprocal(one_plus)
```

### tanh

`tanh(x: tensor[D, P]) → tensor[D, P]`

Using `tanh(x) = 2 * sigmoid(2x) - 1`:
```
two_x = Mul(Const(2, D, P), x)
sig = sigmoid(two_x)                  -- (uses sigmoid lowering above)
result = Add(Mul(Const(2, D, P), sig), Neg(Const(1, D, P)))
```

### layer_norm

`layer_norm(x: tensor[{..., features}, P], gamma: tensor[{features}, P], beta: tensor[{features}, P]) → tensor[{..., features}, P]`

```
mean = Mul(ReduceSum(x, axis=features), Reciprocal(Const(size_features)))
mean_expanded = Expand(mean, features, size_features)
centered = Add(x, Neg(mean_expanded))

-- Variance
sq = Mul(centered, centered)
var = Mul(ReduceSum(sq, axis=features), Reciprocal(Const(size_features)))
var_expanded = Expand(var, features, size_features)

-- Normalize
inv_std = Reciprocal(Sqrt(Add(var_expanded, Const(epsilon))))
normalized = Mul(centered, inv_std)

-- Scale and shift
gamma_expanded = Expand(gamma, batch_dims...)
beta_expanded = Expand(beta, batch_dims...)
result = Add(Mul(normalized, gamma_expanded), beta_expanded)
```

### conv2d

`conv2d(input: tensor[{batch, in_channels, height, width}, P], kernel: tensor[{out_channels, in_channels, kh, kw}, P]) → tensor[{batch, out_channels, out_h, out_w}, P]`

Im2col approach (reshape convolution into matmul):
```
-- Extract patches: slide kernel-sized windows over input
-- This uses Pad (for padding), Stride, Shrink, and Reshape to build a
-- [batch, out_h, out_w, in_channels * kh * kw] tensor of flattened patches
patches = extract_patches(input, kh, kw, stride, padding)

-- Reshape kernel to [out_channels, in_channels * kh * kw]
kernel_flat = Reshape(kernel, {out_channels, in_ckk})

-- Matmul: patches @ kernel^T
-- Result: [batch, out_h, out_w, out_channels]
result = matmul(patches, Permute(kernel_flat, [in_ckk, out_channels]))
result = Permute(result, [batch, out_channels, out_h, out_w])
```

The `extract_patches` operation decomposes into a sequence of Pad, Reshape, Expand, Stride, and Shrink operations. The exact sequence depends on kernel size, stride, and padding parameters.

### cross_entropy_loss

`cross_entropy(logits: tensor[{batch, classes}, P], targets: tensor[{batch}, i64]) → tensor[{}, P]` (scalar)

```
log_softmax = Log(softmax(logits))     -- uses softmax lowering
-- one_hot encode targets, then dot with log_softmax
-- ReduceSum over classes, then mean over batch
selected = gather(log_softmax, targets, axis=classes)  -- [batch]
loss = Neg(Mul(ReduceSum(selected, axis=batch), Reciprocal(Const(batch_size))))
```

Note: `gather` itself lowers to a combination of Expand, Mul (with one-hot mask), and ReduceSum.

---

## 7. Summary: Complete Adjoint Table

| Operation | Adjoint w.r.t. input(s) |
|-----------|------------------------|
| `Neg(x)` | `Neg(g)` |
| `Exp(x)` | `Mul(g, Exp(x))` |
| `Log(x)` | `Mul(g, Reciprocal(x))` |
| `Sin(x)` | `Mul(g, Sin(Add(x, Const(π/2))))` |
| `Sqrt(x)` | `Mul(g, Mul(Const(0.5), Reciprocal(Sqrt(x))))` |
| `Reciprocal(x)` | `Neg(Mul(g, Mul(Reciprocal(x), Reciprocal(x))))` |
| `Add(a, b)` | `(g, g)` |
| `Mul(a, b)` | `(Mul(g, b), Mul(g, a))` |
| `CmpLt(a, b)` | `(0, 0)` — non-differentiable |
| `Max(a, b)` | `(Mul(g, Cast(CmpLt(b, a), P)), Mul(g, Cast(CmpLt(a, b), P)))` |
| `ReduceSum(x, axis)` | `Expand(g, axis, size)` |
| `ReduceMax(x, axis)` | scatter g to max positions (see Section 3) |
| `Reshape(x, new)` | `Reshape(g, original_dims)` |
| `Permute(x, order)` | `Permute(g, inverse_order)` |
| `Expand(x, axis, size)` | `ReduceSum(g, axis)` |
| `Pad(x, padding)` | `Shrink(g, inverse_padding)` |
| `Shrink(x, ranges)` | `Pad(g, inverse_ranges)` |
| `Stride(x, strides)` | upsample (insert zeros) then shrink |
| `Cast(x, P2)` | `Cast(g, P1)` |
| `Const(...)` | N/A (no inputs) |
| `Load(...)` | accumulated and returned |
| `Store(...)` | N/A (output operation) |

Where `g` = `grad_out` (the incoming gradient from downstream).

---

## 8. DAG Invariants

A valid RISC DAG must satisfy:

1. **Acyclicity:** No cycles. The graph is a DAG.
2. **Type consistency:** At every edge, the output type of the source node matches the expected input type of the destination node.
3. **Dimension agreement:** Binary ops require identical named dimensions on both inputs.
4. **Precision agreement:** Binary ops require identical precision on both inputs (except Cast).
5. **Conservation:** Reshape preserves total element count.
6. **Connectivity:** Every non-Const, non-Load node has at least one input. Every non-Store node has at least one consumer (after DCE).
7. **Single output type:** Each node produces exactly one tensor output (except Store, which produces unit).
