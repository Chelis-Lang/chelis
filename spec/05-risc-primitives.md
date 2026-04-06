# spec/05-risc-primitives.md — Chelis RISC Primitive Semantics

**Status:** v0.2 (post design sprint)
**Scope:** The ~12 irreducible tensor operations. Their types, semantics, AD adjoint rules, and how standard ML operations decompose into them.

---

## 1. Design Principles

### 1.1 RISC Philosophy

All tensor computation decomposes into a small set of primitive operations (from tinygrad). Complex behavior emerges from composition + compiler optimization, not from a large operator vocabulary. An AI agent learning to write Chelis only needs to master composition of these primitives.

### 1.2 No Broadcasting

All operands must have matching dimensions. No implicit rank extension, no implicit size expansion. Use `expand` explicitly. This is a hard rule. Rationale: broadcasting masks fatal dimension errors in AI-generated code. Named dimensions + no broadcasting = compile-time shape safety.

### 1.3 Not Tags, Just Functions

RISC primitives are built-in functions in the compiler's scope, not syntax tags. They are accessed via `(var {} name)` and called via `(app {} ...)`. If the primitive set changes, the syntax doesn't.

### 1.4 Two Tiers

**Tier 1: RISC Primitives** — the irreducible set. The compiler's IR operates on these. AD adjoint rules are defined for each.

**Tier 2: Derived Built-Ins** — convenience functions that the compiler lowers to RISC primitive compositions during IR construction. The desugarer emits these; the IR pass decomposes them. They exist so the Deep representation stays readable.

Together, the two tiers define everything the compiler has special knowledge of. Anything that can be expressed as a Chelis program composing these primitives — without requiring custom AD adjoints, backend fusion rules, or compiler-recognized names — belongs in the standard library (`Std.*`) or in external packages, not in the core. See `spec/design/chelis_canonical_reference.md` §8.5 for the full scope boundary taxonomy.

---

## 2. RISC Primitives (Tier 1)

### 2.1 Elementwise Binary

| Name | Signature | Semantics | AD Adjoint (∂L/∂inputs given ∂L/∂output = g) |
|---|---|---|---|
| `add` | `(tensor[D,p], tensor[D,p]) → tensor[D,p]` | Element-wise addition | `(g, g)` |
| `mul` | `(tensor[D,p], tensor[D,p]) → tensor[D,p]` | Element-wise multiplication | `(g * y, g * x)` |
| `cmplt` | `(tensor[D,p], tensor[D,p]) → tensor[D,bool]` | Element-wise less-than comparison | Non-differentiable (zero gradient) |
| `max_elem` | `(tensor[D,p], tensor[D,p]) → tensor[D,p]` | Element-wise maximum | `(g * (x >= y), g * (x < y))` — gradient flows to the max input |

**Dimension rule:** Both inputs must have identical dimension lists. Output has the same dimensions. No broadcasting.

**Precision rule:** Both inputs must have the same precision `p`. Output has the same precision. Exception: `cmplt` returns `bool` regardless of input precision.

### 2.2 Elementwise Unary

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `neg` | `(tensor[D,p]) → tensor[D,p]` | Element-wise negation: -x | `-g` |
| `exp` | `(tensor[D,p]) → tensor[D,p]` | Element-wise e^x | `g * exp(x)` |
| `log` | `(tensor[D,p]) → tensor[D,p]` | Element-wise ln(x) | `g / x` |
| `sin` | `(tensor[D,p]) → tensor[D,p]` | Element-wise sin(x) | `g * cos(x)` where `cos(x) = sin(x + π/2)` |
| `sqrt` | `(tensor[D,p]) → tensor[D,p]` | Element-wise √x | `g / (2 * sqrt(x))` |

**Precision rule:** Float types only (f32, f64, f16, bf16). Not valid on integer types (type error).

### 2.3 Reduction

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `sum` | `(tensor[d₁,...,dₙ, p], axis: int) → tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, p]` | Sum over axis k, removing that dimension | `expand(g, original_shape, axis=k)` |
| `max_reduce` | `(tensor[d₁,...,dₙ, p], axis: int) → tensor[d₁,...,d_{k-1},d_{k+1},...,dₙ, p]` | Max over axis k, removing that dimension | `g * one_hot(argmax(x, k))` — gradient flows to the max element only |

**Axis:** Zero-indexed integer. Must be a valid axis for the input rank.

**Output dimensions:** The dimension at position `axis` is removed. All other dimensions are preserved.

### 2.4 Movement

| Name | Signature | Semantics |
|---|---|---|
| `reshape` | `(tensor[D_old, p], shape) → tensor[D_new, p]` | Reinterpret memory layout. Product of dimensions must match. |
| `permute` | `(tensor[d₁,...,dₙ, p], axes) → tensor[d_{axes[0]},...,d_{axes[n-1]}, p]` | Reorder dimensions. `axes` is a permutation of 0..n-1. |
| `expand` | `(tensor[D_small, p], shape) → tensor[D_large, p]` | Broadcast a dimension of size 1 to a larger size. Does NOT copy data. |
| `pad` | `(tensor[D, p], padding, fill) → tensor[D', p]` | Add elements at boundaries. `padding` specifies (before, after) per axis. |
| `shrink` | `(tensor[D, p], bounds) → tensor[D', p]` | Slice: extract a contiguous sub-tensor. `bounds` specifies (start, end) per axis. |
| `stride` | `(tensor[D, p], strides) → tensor[D', p]` | Strided access: take every n-th element along each axis. |

**Movement AD adjoints:**

| Name | Adjoint |
|---|---|
| `reshape` | `reshape(g, original_shape)` |
| `permute` | `permute(g, inverse_permutation)` |
| `expand` | `sum(g, expanded_axes)` — collapse the expanded dimensions |
| `pad` | `shrink(g, inverse_padding)` — extract the non-padded region |
| `shrink` | `pad(g, inverse_bounds)` — pad gradient back to original size |
| `stride` | appropriate expand/scatter (implementation-specific) |

### 2.5 Memory

| Name | Signature | Semantics |
|---|---|---|
| `const` | `(value, shape...) → tensor[shape, p]` | Create a tensor filled with a constant value. Precision inferred from value or annotation. |
| `load` | `(source, shape...) → tensor[shape, p]` | Load tensor data from external source (file, memory). |

`const` and `load` are the only two ways to create tensors. All other tensors are derived from computation on existing tensors.

`const` is not differentiable (it produces a constant — gradient is zero). `load` is not differentiable.

---

## 3. Derived Built-Ins (Tier 2)

These are convenience functions emitted by the desugarer. The compiler lowers them to RISC primitive compositions during IR construction. They are NOT in the RISC DAG — they exist in Deep AST only.

### 3.1 Arithmetic

| Name | Lowering to RISC |
|---|---|
| `sub(a, b)` | `add(a, neg(b))` |
| `div(a, b)` | `mul(a, recip(b))` where `recip(x) = exp(neg(log(x)))` or specialized |

Note: `neg` is a Tier 1 RISC primitive (see §2), not listed here. `recip` is a lowering-only helper (see §3.5).

### 3.2 Comparison

| Name | Lowering to RISC |
|---|---|
| `eq(a, b)` | `neg(or(cmplt(a, b), cmplt(b, a)))` — neither less than the other |
| `neq(a, b)` | `or(cmplt(a, b), cmplt(b, a))` |
| `gt(a, b)` | `cmplt(b, a)` |
| `gte(a, b)` | `neg(cmplt(a, b))` |
| `lte(a, b)` | `neg(cmplt(b, a))` |

Note: `or(a, b)` on bools is `max_elem(a, b)`. `and(a, b)` on bools is `mul(a, b)`. `not(a)` on bools is `neg(a)` (assuming bools are 0/1).

### 3.3 Activation Functions

| Name | Lowering to RISC |
|---|---|
| `relu(x)` | `max_elem(x, const(0.0, x.shape))` |
| `sigmoid(x)` | `div(const(1.0), add(const(1.0), exp(neg(x))))` |

### 3.4 Higher-Level Operations

| Name | Lowering to RISC |
|---|---|
| `matmul(A, B)` | See §4.1 |
| `mean(x, axis)` | `div(sum(x, axis), const(dim_size))` |
| `softmax(x, axis)` | See §4.2 |
| `linear(x, w, b)` | `add(matmul(x, w), b)` (with appropriate expand on b) |
| `cross_entropy(logits, labels)` | See §4.3 |
| `min_elem(a, b)` | `neg(max_elem(neg(a), neg(b)))` |

**Current implementation note:** the type checker currently also accepts a
`normalize(x)` convenience name.
It is **not** part of the stable Tier 2 surface yet because its lowering semantics are
not specified here and there is no corresponding Phase 0e lowering rule.
Do not treat `normalize` as a stable specified built-in until this document and the IR
lowering are aligned.

### 3.5 Lowering Helpers (NOT Tier 1 or Tier 2)

The following names appear in lowering narratives (§4) as pseudocode or pattern-matched operations. They are NOT RISC primitives and NOT Tier 2 built-ins. They decompose into Tier 1 primitives:

| Helper | Decomposes to |
|---|---|
| `recip(x)` | `exp(neg(log(x)))` or backend-optimized |
| `cos(x)` | `sin(add(x, const(π/2)))` |
| `argmax(x, axis)` | comparison chain via `cmplt` + `max_elem` |
| `gather(x, idx, axis)` | one-hot encoding via `reshape`, `expand`, `mul`, `sum` |
| `im2col(x, kh, kw, ...)` | `stride`, `pad`, `reshape`, `permute` |
| `where(cond, a, b)` | `add(mul(cond, a), mul(neg(cond), b))` assuming bool 0/1 |

---

## 4. Standard Lowerings (Tier 2 → Tier 1)

### 4.1 Matrix Multiplication

```
matmul(A: tensor[..., i, j, p], B: tensor[..., j, k, p]) → tensor[..., i, k, p]
```

Lowering:
```
1. A_expanded = expand(A, [..., i, j, 1])      ;; add dimension for k
2. B_expanded = expand(B, [..., 1, j, k])      ;; add dimension for i
3. product    = mul(A_expanded, B_expanded)     ;; [..., i, j, k]
4. result     = sum(product, axis=-2)           ;; [..., i, k] — sum over j
```

This is the Einstein summation form. The compiler can recognize this pattern and emit optimized BLAS calls instead of the naive implementation.

### 4.2 Softmax

```
softmax(x: tensor[D, p], axis: int) → tensor[D, p]
```

Lowering:
```
1. m = max_reduce(x, axis)                      ;; numerical stability
2. m_expanded = expand(m, x.shape)              ;; broadcast max back
3. shifted = sub(x, m_expanded)                 ;; x - max(x)
4. e = exp(shifted)                             ;; exp(x - max(x))
5. s = sum(e, axis)                             ;; sum of exponentials
6. s_expanded = expand(s, x.shape)              ;; broadcast sum back
7. result = div(e, s_expanded)                  ;; normalize
```

### 4.3 Cross-Entropy Loss

```
cross_entropy(logits: tensor[batch, classes, p], labels: tensor[batch, int32]) → tensor[batch, p]
```

Lowering:
```
1. log_probs = log(softmax(logits, axis=-1))    ;; log-softmax
2. gathered  = gather(log_probs, labels, axis=-1) ;; select correct class
3. result    = neg(gathered)                     ;; negate
```

Note: `gather` is not a RISC primitive. It decomposes further into combinations of `reshape`, `expand`, `mul`, and `sum` using one-hot encoding. The compiler may special-case this pattern for efficiency.

### 4.4 Layer Normalization

```
layer_norm(x: tensor[..., hidden, p], gamma: tensor[hidden, p], beta: tensor[hidden, p]) → tensor[..., hidden, p]
```

Lowering:
```
1. m = mean(x, axis=-1)                         ;; mean over last dim
2. m_exp = expand(m, x.shape)
3. centered = sub(x, m_exp)
4. var = mean(mul(centered, centered), axis=-1)  ;; variance
5. var_exp = expand(var, x.shape)
6. normed = div(centered, sqrt(add(var_exp, const(eps))))  ;; normalize
7. gamma_exp = expand(gamma, x.shape)
8. beta_exp = expand(beta, x.shape)
9. result = add(mul(normed, gamma_exp), beta_exp)  ;; scale and shift
```

### 4.5 Convolution 2D

```
conv2d(input: tensor[batch, in_c, h, w, p],
       kernel: tensor[out_c, in_c, kh, kw, p],
       stride, padding) → tensor[batch, out_c, h', w', p]
```

Lowering via im2col:
```
1. cols = im2col(input, kh, kw, stride, padding)  ;; reshape input to columns
2. result = matmul(kernel_reshaped, cols)           ;; matrix multiply
3. output = reshape(result, [batch, out_c, h', w']) ;; reshape to output
```

`im2col` itself decomposes into `stride`, `pad`, `reshape`, and `permute`. The compiler can recognize this pattern and emit optimized library calls (cuDNN, MKL) instead.

### 4.6 Embedding

```
embedding(indices: tensor[batch, seq, int32], table: tensor[vocab, dim, p]) → tensor[batch, seq, dim, p]
```

Lowering via one-hot + matmul:
```
1. one_hot = ... (indices to one-hot via const + eq + expand)
2. result = matmul(one_hot, table)
```

Or via gather (which itself lowers further).

### 4.7 Multi-Head Attention

```
multi_head_attention(q, k, v: tensor[batch, heads, seq, dim, p],
                     mask: tensor[batch, 1, seq, seq, bool])
  → tensor[batch, heads, seq, dim, p]
```

Lowering:
```
1. scale = const(1.0 / sqrt(dim))
2. scores = mul(matmul(q, permute(k, [0,1,3,2])), scale)  ;; Q @ K^T / sqrt(d)
3. masked = where mask is false, replace with -inf         ;; via mul + add with mask
4. weights = softmax(scores, axis=-1)
5. output = matmul(weights, v)
```

---

## 5. AD Completeness

Every RISC primitive has a defined adjoint rule (§2). This means `grad` can differentiate through any composition of RISC primitives.

**Non-differentiable primitives:** `cmplt`, `const`, `load`. These have zero gradient. The type system (Phase 2, via the `Diff` effect) will detect when `grad` is applied to a function containing non-differentiable operations and report which operations are the problem.

**Almost-everywhere differentiable:** `max_elem` (gradient is zero at the boundary where inputs are equal), `relu` via `max_elem(x, 0)` (gradient is zero at x=0). These are valid targets for `grad` — the subgradient convention (pick one side) is standard in ML.

**Second-order derivatives:** `grad(grad(f))` works if all operations in `f` have defined second-order adjoints. For Phase 0, second-order AD is tested but not optimized.

---

## 6. Reference Implementations

For each RISC primitive, a reference implementation in pseudocode for the C backend (Phase 0f). These are the "obviously correct" naive implementations used as the test oracle.

```c
// add: element-wise
for (int i = 0; i < n; i++) out[i] = a[i] + b[i];

// mul: element-wise
for (int i = 0; i < n; i++) out[i] = a[i] * b[i];

// neg: element-wise
for (int i = 0; i < n; i++) out[i] = -a[i];

// exp: element-wise
for (int i = 0; i < n; i++) out[i] = expf(a[i]);

// log: element-wise
for (int i = 0; i < n; i++) out[i] = logf(a[i]);

// sin: element-wise
for (int i = 0; i < n; i++) out[i] = sinf(a[i]);

// sqrt: element-wise
for (int i = 0; i < n; i++) out[i] = sqrtf(a[i]);

// cmplt: element-wise
for (int i = 0; i < n; i++) out[i] = a[i] < b[i] ? 1.0f : 0.0f;

// max_elem: element-wise
for (int i = 0; i < n; i++) out[i] = a[i] > b[i] ? a[i] : b[i];

// sum: reduce over axis
// (pseudocode for axis=last, generalized via stride/shape logic)
for (int i = 0; i < outer; i++)
  for (int j = 0; j < inner; j++) {
    float acc = 0;
    for (int k = 0; k < axis_size; k++)
      acc += input[i * axis_size * inner + k * inner + j];
    output[i * inner + j] = acc;
  }

// max_reduce: reduce over axis (same loop structure, max instead of +)

// reshape: no data movement, just reinterpret shape/strides
// permute: no data movement, reorder strides
// expand: no data copy, set stride to 0 on expanded axis
// pad: copy with offset, fill padding with constant
// shrink: pointer arithmetic (sub-view, no copy)
// stride: adjust strides (sub-view, no copy)

// const: malloc + memset
// load: fread or memcpy from source
```

These C implementations are the ground truth. The GPU backend (Phase 1) must produce numerically identical results within floating-point tolerance (1e-6 for f32, 1e-12 for f64).
