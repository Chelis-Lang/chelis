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

### 1.3.1 Borrow-Typed Inputs

Read-only tensor primitive parameters are typed as `&tensor[...]` at the type-system
surface. Owned tensor arguments auto-borrow at ordinary call sites and pipe stages.
Outputs remain owned tensors. The borrow distinction is erased before IR and backend
lowering, so primitive DAG nodes and backend kernels keep their existing value model.
Consuming operations such as `realize` and explicit `drop` keep owned parameters.

### 1.4 Two Tiers

**Tier 1: RISC Primitives** — the irreducible set. The compiler's IR operates on these. AD adjoint rules are defined for each.

**Tier 2: Derived Built-Ins** — convenience functions that the compiler lowers to RISC primitive compositions during IR construction. The desugarer emits these; the IR pass decomposes them. They exist so the Deep representation stays readable.

Together, the two tiers define everything the compiler has special knowledge of. Anything that can be expressed as a Chelis program composing these primitives — without requiring custom AD adjoints, backend fusion rules, or compiler-recognized names — belongs in the standard library (`Std.*`) or in external packages, not in the core. See `spec/design/chelis_canonical_reference.md` §8.5 for the full scope boundary taxonomy.

Phase `3h` expands the practical primitive surface beyond this initial minimal set with
`einsum`, `concat` / `split`, `gather` / `scatter`, `where`, `cumsum`, `sort`,
`diagonal` / `trace`, and `clamp`. `Std.Nn.Embedding` remains the named standard-
library surface over `gather`.

For the `3h` additions, Chelis now rejects deterministic literal-driven value errors
at check time when enough information is concrete in source (for example, statically
inconsistent `einsum` extents or duplicate indices in `scatter(..., "replace")`).
When those constraints depend on runtime values instead, the evaluator and generated C
runtime reject them during execution; compiled C exits non-zero rather than aborting.

---

## 2. RISC Primitives (Tier 1)

### 2.1 Elementwise Binary

| Name | Signature | Semantics | AD Adjoint (∂L/∂inputs given ∂L/∂output = g) |
|---|---|---|---|
| `add` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise addition | `(g, g)` |
| `mul` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise multiplication | `(g * y, g * x)` |
| `div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise IEEE-754 division `a / b` | `(g / b, -g * (a/b) / b)` (= `(g/b, -g*y/b)` using `y = a/b`) |
| `cmplt` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,bool]` | Element-wise less-than comparison | Non-differentiable (zero gradient) |
| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise maximum | `(g * (x >= y), g * (x < y))` — gradient flows to the max input |

**`div` semantics.** `div(a, b)` lowers to the target's native
`/` operator.

- **Float operands** (f32, f64, f16, bf16) — IEEE-754 division.
  Corner cases follow IEEE: `1/0 = +inf`, `1/-0 = -inf`,
  `0/0 = NaN`, `1/-1 = -1`, `(any non-NaN) / -2.0` yields the
  algebraic value. A historical `mul(a, exp(neg(log(b))))`
  decomposition returned NaN for any `b ≤ 0` because `log(b)` is
  undefined there; that decomposition is not reachable from any
  Tier 2 op.

- **Integer operands** (int8, int16, int32, int64) — C/Rust
  truncating division (round toward zero). `7 / 2 == 3`,
  `-7 / 2 == -3`, `1 / 0` traps (implementation-defined per C; the
  evaluator panics, the C backend follows the platform's
  signal). This differs from torch's `true_divide` and JAX's
  default `jnp.divide`, both of which upcast integers to float
  and return float. Chelis matches the C-language convention
  because chelis-std's `Std.Decimal` arithmetic uses
  `div(int64, int64)` for scale shifts; float-only would force a
  separate `int_div` primitive without benefit.

**Dimension rule:** Both inputs must have identical dimension lists. Output has the same dimensions. No broadcasting.

**Precision rule:** Both inputs must have the same precision `p`. Output has the same precision. Exception: `cmplt` returns `bool` regardless of input precision.

### 2.2 Elementwise Unary

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `neg` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise negation: -x | `-g` |
| `recip` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise IEEE-754 reciprocal `1.0 / x` | `-g * y * y` (= `-g / x^2`, using `y = 1/x`) |
| `exp` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise e^x | `g * exp(x)` |
| `log` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise ln(x) | `g / x` |
| `sin` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise sin(x) | `g * cos(x)` where `cos(x) = sin(x + π/2)` |
| `sqrt` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise sqrt(x) | `g / (2 * sqrt(x))` |

**`recip`.** Native IEEE-754 reciprocal, used inside
`lower_sigmoid` (and any other reciprocal-shaped lowering) to
produce a single op instead of the prior `exp(neg(log(x)))` chain.
`recip(0) = +inf`, `recip(-0) = -inf`, `recip(-x) = -recip(x)` for
finite x — never NaN-from-log.

**Precision rule:** Float types only (f32, f64, f16, bf16). Not valid on integer types (type error).

### 2.3 Reduction

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `sum` | `(&tensor[d1,...,dn,p], axis: int, accumulator: prec = default(p)) -> tensor[d1,...,d{k-1},d{k+1},...,dn,acc]` | Sum over axis k, removing that dimension. `accumulator` controls the precision of the running sum and the result element type. | `expand(g, original_shape, axis=k)` (gradient flows back at the operand precision `p`; the adjoint is computed in operand precision) |
| `max_reduce` | `(&tensor[d1,...,dn,p], axis: int) -> tensor[d1,...,d{k-1},d{k+1},...,dn,p]` | Max over axis k, removing that dimension | `g * one_hot(argmax(x, k))` — gradient flows to the max element only |

**Axis:** Integer index into the input rank. Non-negative axes are
zero-indexed from the front. A negative axis indexes from the end:
`-1` is the last axis, `-2` the second-to-last, and so on (an axis `a`
with `a < 0` denotes `rank + a`). After this normalization the axis
must fall within `0..rank`; an out-of-range axis is a type error. This
from-the-end convention applies uniformly to every axis-taking
primitive — the reductions here, `softmax`, `mean`, `gather`,
`scatter`, and the movement and ordering ops — and is the convention
the formula examples below already use (`axis=-1` for the last axis).

**Output dimensions:** The dimension at position `axis` is removed. All other dimensions are preserved.

**Accumulator parameter (`sum` only).** The optional `accumulator: prec`
parameter controls the precision used for the running sum and the precision
of the output tensor. The default is the operand precision for f32/f64/i32/i64
operands, and a wider promoted type for narrower operand types. The full
table of defaults (and the rationale for each row) is the authoritative
statement in `spec/04-type-system.md` §5.7.1; this primitive doc is the
operational location of the parameter on the IR node.

In short:

- `bf16` / `f16` operands → `f32` accumulator → `f32` result
- `f32` operands → `f32` accumulator → `f32` result
- `f64` operands → `f64` accumulator → `f64` result
- `int8` / `int16` operands → `int32` accumulator → `int32` result
- `int32` operands → `int32` accumulator → `int32` result
- `int64` operands → `int64` accumulator → `int64` result

There is no implicit precision promotion: omitting the parameter resolves to
the documented default before lowering. The IR `RiscOp::ReduceSum` node
always carries a populated accumulator-precision field. Programs that
explicitly request a narrower-than-default accumulator are a type error per
§5.7.1.

`max_reduce` does not take an accumulator parameter. Max is order-preserving
and does not lose precision the way a long sum does, so the result element
type matches the operand element type.

**Reduction order (`sum` only).** `sum` evaluates the reduction with a
**stride-4 ILP cascade** — four independent accumulator lanes loaded
in round-robin (`acc[i & 3] += value[i]`), combined at the end as
`(acc0 + acc1) + (acc2 + acc3)`. This matches PyTorch's CPU
`row_sum` (`num_levels=4 ilp_factor=4`) and NumPy's pairwise sum in
the small-n regime, so f32 `sum` is bit-exact with `torch.sum(...)`
for `n ≤ 16` on the reduced axis. For `n > 16` the result may differ
from torch by up to ~1 ULP until the multi-level cascade lands as a
follow-up. The order is purely positional so the algorithm is
deterministic across runs and hosts; `#pragma omp parallel for` is
applied to the outer (output-element) loop only, never the inner
reduction.

This change is observable for floating-point operands — the prior
strict left-fold could diverge from torch by ~1 ULP at unfavorable
seeds and forced parity-oracle carve-outs in downstream harnesses
(issue Chelis-Lang/chelis#163). Integer reductions are unchanged
(integer addition is associative). The accumulator-precision rule
above is orthogonal to the reduction order: the lane type is the
accumulator type, and the final combine happens in the same
precision.

**GPU caveat.** The HIP and Metal backends keep their existing
device reduction kernels (single-accumulator per-thread + tree
combine for Metal; single-accumulator for HIP). Bit-exact GPU
parity with torch's CPU `row_sum` is out of scope for this change
— torch itself uses a different kernel (`cub::DeviceReduce`) on
GPU. CPU eval, `chelis eval`, and the C backend all match
`row_sum`; the HIP and Metal backends may differ from each other
and from CPU at the ~1 ULP level on f32.

### 2.4 Movement

| Name | Signature | Semantics |
|---|---|---|
| `reshape` | `(&tensor[D_old,p], shape) -> tensor[D_new,p]` | Reinterpret memory layout. Product of dimensions must match. |
| `permute` | `(&tensor[d1,...,dn,p], axes) -> tensor[d_axes,p]` | Reorder dimensions. `axes` is a permutation of 0..n-1. |
| `expand` | `(&tensor[D_small,p], shape) -> tensor[D_large,p]` | Broadcast a dimension of size 1 to a larger size. Does NOT copy data. |
| `pad` | `(&tensor[D,p], padding, fill) -> tensor[D',p]` | Add elements at boundaries. `padding` specifies (before, after) per axis. |
| `shrink` | `(&tensor[D,p], bounds) -> tensor[D',p]` | Slice: extract a contiguous sub-tensor. `bounds` specifies (start, end) per axis. |
| `stride` | `(&tensor[D,p], strides) -> tensor[D',p]` | Strided access: take every n-th element along each axis. |

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

`const` and `load` are the pure tensor constructors. Effectful tensor constructors such
as seeded random generation are tracked separately below. All other tensors are derived
from computation on existing tensors.

`const` is not differentiable (it produces a constant — gradient is zero). `load` is not differentiable.

### 2.6 Effectful Primitive

| Name | Signature | Semantics | AD / effect note |
|---|---|---|---|
| `dropout` | `(&tensor[D, f32], f32) -> tensor[D, f32]` | Zero elements according to a pseudorandom mask determined by the active `with seed(...)` handler and the dropout rate | Introduces `Random`. In the shipped evaluator/AD path, the mask is treated as fixed with respect to the handled seed so the backward pass reuses the same seeded dropout pattern. |
| `uniform_like` | `(&tensor[D, f32], f32, f32) -> tensor[D, f32]` | Create a tensor matching the input shape, filled from a deterministic uniform distribution under the active `with seed(...)` handler | Introduces `Random`. C backend codegen supports direct DAG lowering and generated host functions that call random stdlib/user helpers. |

Operational note: the evaluator and lowering path implement seeded `dropout`, but
`chelis build` does not yet codegen it for the `c` or `hip` backend targets.

---

## 3. Derived Built-Ins (Tier 2)

These are convenience functions emitted by the desugarer. The compiler lowers them to RISC primitive compositions during IR construction. They are NOT in the RISC DAG — they exist in Deep AST only.

### 3.1 Arithmetic

| Name | Lowering to RISC |
|---|---|
| `sub(a, b)` | `add(a, neg(b))` |

Note: `div` and `neg` are Tier 1 RISC primitives (see §2.1, §2.2),
not Tier 2 derived built-ins. `recip` is also a Tier 1 primitive
(§2.2). The historically `div(a, b) = mul(a,
exp(neg(log(b))))` lowering — which returned NaN for `b ≤ 0` — is
no longer reachable from any Tier2 op.

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
| `sigmoid(x)` | `recip(add(const(1.0), exp(neg(x))))` |

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
not specified here and there is no corresponding IR check lowering rule.
Do not treat `normalize` as a stable specified built-in until this document and the IR
lowering are aligned.

### 3.5 Lowering Helpers And Sparse Implementation Nodes

The following names appear in lowering narratives (§4) as pseudocode or
pattern-matched operations. Most decompose into Tier 1 primitives:

| Helper | Decomposes to |
|---|---|
| `cos(x)` | `sin(add(x, const(π/2)))` |
| `argmax(x, axis)` | comparison chain via `cmplt` + `max_elem` |
| `gather(x, idx, axis)` | one-hot encoding via `reshape`, `expand`, `mul`, `sum` |
| `im2col(x, kh, kw, ...)` | `stride`, `pad`, `reshape`, `permute` |
| `where(cond, a, b)` | `add(mul(cond, a), mul(neg(cond), b))` assuming bool 0/1 |

Implementation note: the compiler now also has first-class specialized sparse
IR nodes `RiscOp::Gather { axis }`, `RiscOp::ScatterAdd { axis }`, and
`RiscOp::Scatter { axis }`, with evaluator, verifier, AD, C/HIP backend, and
wire-schema support. Tensor-lane Surf `gather(values, indices, axis)` lowers
directly to `RiscOp::Gather` in the current implementation, avoiding the host
runtime call and the dense one-hot materialization. The tensor-lane Surf
builtin `scatter_replace(base, indices, updates, axis)` lowers directly to
`RiscOp::Scatter` for the last-write-wins case. The shared specialization pass
also recognizes the internal `RiscOp::OneHot { vocab } + Expand + Mul + Sum`
gather tree and collapses it before DCE/codegen. Arbitrary historical const/eq
one-hot encodings are not recognized because they do not preserve the original
index operand.

#### Replace-scatter vs scatter-add

`Scatter` and `ScatterAdd` are intentionally distinct primitives. Both
take inputs `(target, indices, updates)` with the same shape contract
(updates shape equals `target.dims[..axis] ++ indices.dims ++
target.dims[axis+1..]`) and the same precision constraints
(int32/int64 indices; target/updates/output precision identical).
They differ only in how duplicate target indices are resolved and in
their AD policies:

| Op | Duplicate-index semantics | AD adjoint |
|---|---|---|
| `ScatterAdd { axis }` | commutative accumulation (`+=`) | `Gather { axis }` — duplicate indices fan-out correctly |
| `Scatter { axis }` | last-write-wins (deterministic order rule below) | **no_grad** — fail-closed with `AdError::NotSupported` |

**Deterministic-order rule for `Scatter`:** updates-tensor row-major
(C order) flat iteration. For each `i ∈ 0..updates.size` in
ascending flat-index order, the write
`target[..., indices[idx_pos(i)], ...] = updates[i]` occurs at step
`i`. When two updates target the same cell, the write with the
larger flat index in `updates` is the final value at that cell.
Every backend (interpreter, C, HIP) must observe this rule:

- The IR evaluator (`chelis_ir::eval::scatter_replace`) iterates the
  updates tensor sequentially in row-major flat order.
- The C backend emits a single-threaded sequential loop (no
  `#pragma omp parallel for`) — parallelizing would race on
  duplicate indices and break determinism.
- The HIP backend emits a `<<<1, 1>>>` single-thread serial kernel
  for the same reason. Atomic ops do not provide ordered
  last-write-wins semantics; introducing a parallel implementation
  would require an explicit tie-breaker that picks the maximum
  flat-index writer per target cell. That optimization is
  permitted only when it preserves exactly this rule.

**AD policy for `Scatter`:** reverse-mode AD is structurally
rejected. The forward result depends on iteration order at
duplicate indices, so distributing a single output gradient across
the colliding updates would require an arbitrary policy that does
not derive from the forward semantics. The rejection is returned
through the new structured error type `chelis_ir::grad::AdError`:

```rust
AdError::NotSupported {
    op: "scatter_replace",
    reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
}
```

Downstream consumers pattern-match on the enum variant; the
rendered `Display` string is for human consumption only. Programs
that need a differentiable variant must use `ScatterAdd` (whose
adjoint is well-defined as `Gather`) or wrap `Scatter` in a
stop-gradient.

---

## 4. Standard Lowerings (Tier 2 → Tier 1)

### 4.1 Matrix Multiplication

```
matmul(A: tensor[..., i, j, p], B: tensor[..., j, k, p],
       accumulator: prec = default(p))
       → tensor[..., i, k, p]
```

Lowering:
```
1. A_expanded = expand(A, [..., i, j, 1])      ;; add dimension for k
2. B_expanded = expand(B, [..., 1, j, k])      ;; add dimension for i
3. product    = mul(A_expanded, B_expanded)     ;; [..., i, j, k]
4. result     = sum(product, axis=-2,           ;; [..., i, k] — sum over j
                    accumulator=acc)             ;; in `acc` precision
5. (optional) result = cast(result, p)          ;; downcast back to operand
                                                 ;; precision when acc != p
```

This is the Einstein summation form. The compiler can recognize this pattern
and emit optimized BLAS calls instead of the naive implementation.

**Accumulator parameter.** Like `sum` (§2.3), `matmul` carries an optional
accumulator-precision parameter. The defaults for matmul are:

- `bf16` / `f16` operands → `f32` accumulator, downcast to operand precision
- `f32` operands → `f32` accumulator (no downcast)
- `f64` operands → `f64` accumulator (no downcast)

The result precision is always the operand precision so callers see a
uniform-precision output; the wider accumulator is consumed inside the op.
There is no implicit precision promotion; omitting the parameter resolves to
the documented default before lowering. The IR `RiscOp::Matmul` node always
carries a populated accumulator-precision field. The full default table and
rationale are in `spec/04-type-system.md` §5.7.1.

Integer matmul (operands of `int8` / `int16` / `int32` / `int64`) is not
admitted in the active matmul signature; see `spec/04-type-system.md` §5.7.2
for rationale. Use `reduce_sum` over an explicit `expand`+`mul` lowering for
integer inner products.

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

Note: Surf-level `gather` is specified to decompose further into combinations of
`reshape`, `expand`, `mul`, and `sum` using one-hot encoding. The compiler may
special-case this pattern for efficiency by replacing it with the specialized
`RiscOp::Gather` / `RiscOp::ScatterAdd` sparse nodes before codegen.

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
