# spec/05-risc-primitives.md — Chelis RISC Primitive Semantics

This chapter defines the irreducible tensor operations, their types and exact
semantics, their AD adjoints or structural rejections, and the standard
decompositions built from them.

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
lowering: primitive DAG nodes and backend kernels carry values, not borrow markers.
Consuming operations such as `realize` and explicit `drop` keep owned parameters.

The same observational rule covers the read-only `List` / `Dict` queries `len` and
`index`: they auto-borrow their container argument rather than consuming it, so the
idiomatic "read a list's length / element, then reuse the list" pattern type-checks
without a `copy()`. The runtime backings (`chelis_list_len`, `chelis_list_index`) take a
`const` container pointer and never free it — `index` retains the element it returns — so
the caller still owns the container afterwards. A genuine consume of the container (an
explicit `drop`, or moving it into an owned parameter) still makes a later `len` / `index`
read a use-after-consume (chelis#527). As with tensors, the borrow is auto-applied to the
owned argument; writing the container query as `len(&xs)` is not a supported surface form.

### 1.4 Two Tiers

**Tier 1: RISC Primitives** — the irreducible set. The compiler's IR operates on these. AD adjoint rules are defined for each.

**Tier 2: Derived Built-Ins** — convenience functions whose governing atom
defines when their typed identity may lower to a RISC primitive composition.
The desugarer emits these identities; each remains intact through every named
semantic transform, including AD when required, before an IR pass decomposes
it. They exist so the Deep representation stays readable without making early
decomposition part of the language contract.

Together, the two tiers define everything the compiler has special knowledge of. Anything that can be expressed as a Chelis program composing these primitives — without requiring custom AD adjoints, backend fusion rules, or compiler-recognized names — belongs in the standard library (`Std.*`) or in external packages, not in the core. See `spec/design/chelis_canonical_reference.md` §8.5 for the full scope boundary taxonomy.

The primitive surface includes
`einsum`, `concat` / `split`, `gather` / `scatter`, `where`, `cumsum`, `sort`,
`diagonal` / `trace`, and `clamp`. `School.Nn.Embedding` (moved to the `school` library
in chelis-std) is the named library surface over `gather`.

Chelis rejects deterministic literal-driven value errors
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
| `div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise IEEE-754 division `a / b` (**float operands only**) | `(g / b, -g * (a/b) / b)` (= `(g/b, -g*y/b)` using `y = a/b`) |
| `floor_div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise floor division: `floor(a / b)`, rounding toward −∞ | Non-differentiable (piecewise constant); `grad` rejects it |
| `trunc_div` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise truncating division (round toward zero), **integer operands only** | Non-differentiable (piecewise constant); `grad` rejects it |
| `wrap_add` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular addition | Non-differentiable; `grad` rejects it |
| `wrap_sub` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular subtraction | Non-differentiable; `grad` rejects it |
| `wrap_mul` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular multiplication | Non-differentiable; `grad` rejects it |
| `cmplt` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,bool]` | Element-wise less-than comparison | Non-differentiable (zero gradient) |
| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise maximum | `(g * (x >= y), g * (x < y))` — gradient flows to the max input |

> **[05-OP-17]** `wrap_add(left, right) -> result` admits two signed-integer
> scalar operands or two signed-integer tensor operands with the same dtype
> and dimensions. It returns the same surface, dimensions, and dtype. At an
> operand width `w`, each result is the unique signed `w`-bit representative
> congruent to the exact mathematical sum modulo `2^w`. It never traps for
> overflow, has no accumulator, and is non-differentiable: `grad` rejects it.
> A mixed dtype or surface, mismatched tensor dimensions, or a float, `bool`,
> `string`, or reserved dtype spelling is a type error.
>
> **[05-OP-18]** `wrap_sub(left, right) -> result` has the signature, dtype,
> shape, rejection, accumulator, and differentiation contract of [05-OP-17],
> returning the unique signed operand-width representative congruent to the
> exact mathematical difference modulo `2^w`.
>
> **[05-OP-19]** `wrap_mul(left, right) -> result` has the signature, dtype,
> shape, rejection, accumulator, and differentiation contract of [05-OP-17],
> returning the unique signed operand-width representative congruent to the
> exact mathematical product modulo `2^w`.

**`div` semantics.** `div(a, b)`
is **restricted to float operands** (f32, f64, f16, bf16) and
lowers to the target's native floating `/` operator with IEEE-754
semantics. Corner cases follow IEEE: `1/0 = +inf`, `1/-0 = -inf`,
`0/0 = NaN`, `1/-1 = -1`, `(any non-NaN) / -2.0` yields the
algebraic value. It never decomposes through `log` and `exp`.

`div(int_tensor, int_tensor)` is a **type error** (the `/`
operator on integer operands is likewise rejected, because `/`
desugars to `div`). The diagnostic cites this section and points
at the explicit `floor_div` / `trunc_div` integer primitives below.

**`floor_div` semantics.** `floor_div(a, b)` computes
`floor(a / b)` element-wise, rounding the quotient toward −∞.

- **Integer operands** (int8, int16, int32, int64) — the result is
  the largest integer `q` with `q * b ≤ a` (for `b > 0`). It is
  realized as native truncating `/` plus a sign correction:
  `q = a / b; r = a % b; if (r != 0 && ((r < 0) != (b < 0))) q -= 1`.
  Floor and truncate agree when the operands share a sign; they
  differ on a mixed-sign exact-fraction case:
  `floor_div(-7, 2) == -4` (truncate gives `-3`),
  `floor_div(7, -2) == -4`, `floor_div(-8, 2) == -4`. Matches
  Python `//`, torch `floor_divide`, JAX `floor_divide`, numpy
  `floor_divide`.
- **Float operands** (f32, f64, f16, bf16) — `floor(a / b)` under
  IEEE division (so `floor_div(7.0, 2.0) == 3.0`,
  `floor_div(-7.0, 2.0) == -4.0`); `b == 0` follows IEEE and is not
  guarded (`floor(+inf) == +inf`).
- **Zero divisor (integer operands)** — traps with the same clean
  diagnostic as `mod` (`integer division or remainder by zero`).
  Both the evaluator and the C backend halt; the C backend emits
  the explicit `chelis_int_div_guard` zero-divisor guard (see the
  `trunc_div` note below for the portability rationale).

**`trunc_div` semantics (integer-only).** `trunc_div(a, b)`
computes the quotient rounded toward zero — the exact C/Rust
integer `/` operator — and is **valid on integer operands only**
(applying it to floats is a type error; a float quotient rounded toward
zero requires an explicit sign-aware `floor`/`ceil` composition before the
checked integer cast).
`trunc_div(7, 2) == 3`, `trunc_div(-7, 2) == -3`,
`trunc_div(7, -2) == -3`, `trunc_div(-8, 2) == -4`. This is the
exact quotient semantics chelis-std's `Std.Decimal` arithmetic
needs for scale shifts and quotient computation. `1 / 0` traps with
the `integer division or remainder by zero` diagnostic (the same
message `mod`'s zero-divisor path emits, so the two primitives are
consistent across both lanes). The C backend emits an explicit
zero-divisor guard (`chelis_int_div_guard`) before every integer
`trunc_div` / `floor_div` / `mod` rather than relying on a hardware
fault: x86 raises `SIGFPE` on integer division by zero, but AArch64
(e.g. Apple silicon) defines it to return a value and does not
fault, so a signal-dependent trap would silently compute a wrong
answer there. The explicit guard traps deterministically on every
target.

**Dimension rule:** Both inputs must have identical dimension lists. Output has the same dimensions. No broadcasting.

**Precision rule:** Both inputs must have the same precision `p`. Output has
the same precision. Exception: `cmplt` returns `bool` regardless of input
precision. Additional restrictions: `div` admits only float precisions
(integer operands are a type error citing this section); `trunc_div` and the
three `wrap_*` operations admit only signed-integer precisions (float operands
are a type error); `floor_div` admits both integer and float precisions.

**Scalar modular forms.** `wrap_add`, `wrap_sub`, and `wrap_mul` admit two
scalar operands wherever [05-OP-17..19] admit the tensor form. The result is a
scalar of the same signed-integer dtype. A scalar and a non-scalar tensor do
not broadcast.

**Scalar `max_elem`/`min_elem`.** The element-wise maximum and its §3.4
`min_elem` lowering also admit two scalar operands of the same numeric dtype
and return a scalar of that dtype. This is the rank-zero instance of the
tensor rule, not scalar/tensor broadcasting: a scalar and a non-scalar tensor
remain a dimension mismatch. The scalar forms admit the same signed-integer
and float precisions as their tensor forms and use the same adjoint rule.

### 2.2 Elementwise Unary

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `neg` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise negation: -x | `-g` |
| `recip` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise IEEE-754 reciprocal `1.0 / x` | `-g * y * y` (= `-g / x^2`, using `y = 1/x`) |
| `exp` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise e^x | `g * exp(x)` |
| `log` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise ln(x) | `g / x` |
| `sin` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise sin(x) | `g * cos(x)` |
| `cos` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise cos(x) | `-g * sin(x)` |
| `tan` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise tan(x) | `g / (cos(x) * cos(x))` (= `g / cos²(x)`) |
| `atan` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise atan(x) | `g / (1 + x * x)` |
| `sqrt` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise sqrt(x) | `g / (2 * sqrt(x))` |
| `abs` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise absolute value | For float operands, `g * sign(x)` (sign = `(x > 0) - (x < 0)`; 0 at x = 0); integer operands are forward-only |
| `floor` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise floor; identity on integer operands | Float operands are non-differentiable (piecewise constant) and `grad` rejects them; the integer identity may be erased before AD |
| `ceil` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise ceil; identity on integer operands | Float operands are non-differentiable (piecewise constant) and `grad` rejects them; the integer identity may be erased before AD |
| `round` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise round to nearest, ties to even (IEEE-754 roundTiesToEven / banker's rounding); identity on integer operands | Float operands are non-differentiable (piecewise constant) and `grad` rejects them; the integer identity may be erased before AD |
| `is_nan` | `(&tensor[D,p_float]) -> tensor[D,bool]` | True exactly for NaN | Non-differentiable predicate; contributes zero cotangent |
| `is_finite` | `(&tensor[D,p_float]) -> tensor[D,bool]` | True exactly for finite values | Non-differentiable predicate; contributes zero cotangent |
| `is_infinite` | `(&tensor[D,p_float]) -> tensor[D,bool]` | True exactly for positive or negative infinity | Non-differentiable predicate; contributes zero cotangent |

> **[05-OP-20]** `is_nan(x) -> result` admits an `f16`, `bf16`, `f32`, or
> `f64` scalar or tensor and returns `bool` on the same surface and, for a
> tensor, with the same dimensions. It reads the finalized stored value
> without conversion and returns true exactly when that value is NaN. Signed
> zeros, infinities, subnormals, and finite normals return false. It has no
> accumulator and, like `cmplt`, contributes zero cotangent when used inside a
> differentiable expression. Integer, `bool`, `string`, and reserved dtype spellings
> are type errors.
>
> **[05-OP-21]** `is_finite(x) -> result` has the signature, dtype, shape,
> no-conversion, accumulator, differentiation, and rejection contract of
> [05-OP-20]. It returns true exactly for finite stored values, including
> both signed zeros and subnormals, and false for NaN and both infinities.
>
> **[05-OP-22]** `is_infinite(x) -> result` has the signature, dtype, shape,
> no-conversion, accumulator, differentiation, and rejection contract of
> [05-OP-20]. It returns true exactly for positive or negative infinity and
> false for NaN and every finite value.

**`recip`.** Native IEEE-754 reciprocal, used inside
`lower_sigmoid` (and any other reciprocal-shaped lowering) to
produce a single op instead of the prior `exp(neg(log(x)))` chain.
`recip(0) = +inf`, `recip(-0) = -inf`, `recip(-x) = -recip(x)` for
finite x — never NaN-from-log.

**Precision rule:** `recip`, `exp`, `log`, `sin`, `cos`, `tan`, `atan`,
`sqrt`, `is_nan`, `is_finite`, and `is_infinite` admit float types only (f32,
f64, f16, bf16). The three classification operations return `bool`; the other
float operations return the operand dtype. `neg` and `abs` admit
those float types plus the signed integer types. Integer `neg` and `abs`
compute at the operand's declared width and trap on the unrepresentable
minimum-value case according to [04-NUM-9]. `floor`, `ceil`, and `round` admit
both float and signed-integer types; each is exactly the identity on an
integer operand, with no float conversion. No unary numeric primitive admits
`bool`, `string`, or the reserved `f8e4m3` dtype spelling.

**Scalar unary forms.** `recip`, `tan`, `atan`, `floor`, `ceil`, `round`,
`is_nan`, `is_finite`, and `is_infinite` admit a scalar operand wherever the
precision rule above admits the tensor form. The classification operations
return scalar `bool`; the others return the operand dtype. A scalar is the
rank-zero instance of the element-wise operation: it uses [04-NUM-8]'s
arithmetic width and is finalized once at its declared storage width. In
particular, scalar integer `floor`, `ceil`, and `round` are exact identity
operations and never convert through a float dtype.

### 2.3 Reduction

`mean` is listed with the reductions to keep its dtype, empty-axis, and
adjoint contract beside the primitives it composes. It remains Tier 2 and
lowers exactly as [05-OP-11] specifies.

For this table, `Axis+` denotes one or more unique compile-time positional
`int32` axes or one or more unique named axes under spec/04 §4.5.3. `D \ K`
denotes the input dimensions with the complete selected axis set `K` removed.

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `sum` | `(&tensor[D,p], axes: Axis+, accumulator: prec = default(p)) -> tensor[D \ K,sum_result(p,accumulator)]` | Sum over every selected axis. `accumulator` controls the running precision; `sum_result` is §5.7.1's result-precision rule. | Reverse the canonical highest-original-position-first single-axis composition at operand precision `p` |
| `count` | `(&tensor[D,bool], axes: Axis+) -> tensor[D \ K,int64]` | Count true elements over the complete selected axis set in one dedicated reduction | Non-differentiable; `grad` rejects it |
| `mean` | `(&tensor[D,p_float], axes: Axis+) -> tensor[D \ K,p_float]` | Arithmetic mean over every selected axis | Reverse the canonical highest-original-position-first single-axis composition |
| `max_reduce` | `(&tensor[D,p], axes: Axis+) -> tensor[D \ K,p]` | Maximum over every selected axis | Reverse the canonical composition; each step routes a NaN result to its first NaN or splits `g` among equal selected extrema, including infinities |
| `min_reduce` | `(&tensor[D,p], axes: Axis+) -> tensor[D \ K,p]` | Minimum over every selected axis | Reverse the canonical composition; each step routes a NaN result to its first NaN or splits `g` among equal selected extrema, including infinities |
| `prod_reduce` | `(&tensor[D,p], axes: Axis+) -> tensor[D \ K,p]` | Product over every selected axis | Reverse-mode derivative of the exact canonical composition of balanced product trees |
| `argmax_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,int64]` | Lowest axis index of the maximum or first NaN | Non-differentiable; `grad` rejects it |
| `argmin_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,int64]` | Lowest axis index of the minimum or first NaN | Non-differentiable; `grad` rejects it |

> **[05-OP-11]** `mean(x, axes...) -> result` admits a tensor operand of
> `f16`, `bf16`, `f32`, or `f64` and returns the same float dtype with the
> selected axis removed. On a non-empty axis its value is exactly the
> composition `div(sum(x, axis), divisor)`: `sum` uses its §5.7.1 default
> accumulator, order, result dtype, and finalization; `divisor` is the positive
> axis extent converted once to the sum result dtype under [04-NUM-14]; then
> `div` executes and finalizes as a separate operation at [04-NUM-8]'s declared
> width. Integer, `bool`, `string`, and reserved dtype spellings are type errors. A
> zero-length axis is a type error when statically known. If an execution-time
> extent is zero, a guard before the composition traps `Domain` as operation
> `mean` at the result dtype. The single-axis adjoint is
> `expand(g / divisor, original_shape, axis)` at the operand dtype. One or
> more positional or named axes follow spec/04 §4.5.3: the call executes
> these exact single-axis graphs in highest-original-position-first order and
> the adjoint reverses that composition. Mixed, duplicate, dynamic, absent,
> ambiguous, or out-of-range axes are type errors. `mean` has no accumulator
> parameter of its own.
>
> **[05-OP-12]** `max_reduce(x, axes...) -> result` admits every active signed
> integer and float tensor dtype and returns that same dtype with the selected
> axis removed. Values are compared without conversion at their stored dtype.
> The first NaN in increasing axis-index order is the result with its exact
> stored payload and sign bits; otherwise the
> maximum is returned, preserving the first stored representation among equal
> values. A zero-length axis is a type error when statically known. If an
> execution-time extent is zero, the operation traps `Domain` as operation
> `max_reduce` at its result dtype. There is no accumulator parameter. For
> float operands, the adjoint divides the upstream cotangent equally among all
> elements that compare equal to the selected non-NaN maximum. This includes
> equal positive or negative infinities. The tie count `k` is
> counted exactly as `int64`, converted once to the operand float dtype under
> [04-NUM-14], and each selected element receives ordinary dtype-width
> `div(g, k)`. When the forward result is NaN, its full cotangent flows to the
> first NaN selected by the forward rule and every other element receives
> zero. One or more axes use spec/04 §4.5.3's validation and
> highest-original-position-first composition of this exact graph; its
> adjoint reverses that composition. Integer operands are forward-only and
> `grad` rejects them.
>
> **[05-OP-13]** `min_reduce(x, axes...) -> result` has the same dtype,
> finalization, empty-axis, accumulator, and differentiation contract as
> [05-OP-12], replacing maximum by minimum. It returns the first NaN in
> increasing axis-index order with its exact stored payload and sign bits;
> otherwise it preserves the first stored
> representation among equal minima. For float operands, every element equal
> to the selected non-NaN minimum, including equal positive or negative
> infinities, receives the upstream cotangent divided by the number of equal
> minima.
>
> **[05-OP-14]** `prod_reduce(x, axes...) -> result` admits every active signed
> integer and float tensor dtype and returns that same dtype with the selected
> axis removed. It has no accumulator parameter: multiplication uses the
> operand's [04-NUM-8] arithmetic width (`f16` and `bf16` therefore multiply
> in `f32`). The reduction uses the canonical balanced tree: each level
> pairs adjacent values from left to right, multiplies each pair, carries an
> odd final value unchanged to the next level, and repeats until one value
> remains. Each multiplication executes at the operand arithmetic width and
> is finalized once to the operand storage dtype before it enters the next level;
> an odd carry preserves its stored bits unchanged. Integer overflow is checked
> at every multiplication. This logical tree is identical
> in every lane and governs [04-NUM-12] trap
> occurrence. A zero-length axis returns the multiplicative identity one at
> the operand dtype. For float operands, the adjoint is the reverse-mode
> derivative of that exact multiplication tree. A division-free
> prefix/suffix implementation may be used only when it preserves that tree's
> arithmetic-width operation order and result bits; consequently gradients at
> zero operands are defined. Integer operands are forward-only and `grad`
> rejects them. `bool`, `string`, reserved dtype spellings, scalar, and all other operands are
> type errors; in particular `prod_reduce` never treats `bool` as integer
> zero/one storage.
> One or more axes use spec/04 §4.5.3's validation and
> highest-original-position-first composition of this exact balanced graph;
> reverse mode differentiates the composed graph in reverse order.
>
> **[05-OP-15]** `argmax_reduce(x, axis) -> result` admits every active
> signed integer and float tensor dtype and returns `int64` indices with the
> selected axis removed. If the slice contains NaNs, it returns the lowest
> axis index containing NaN; otherwise it returns the lowest axis index whose
> stored value is maximal. Comparisons never convert through another dtype.
> A zero-length axis is a type error when statically known. If an
> execution-time extent is zero, the operation traps `Domain` as operation
> `argmax_reduce` at result dtype `int64`. It has no accumulator and is
> non-differentiable: `grad` rejects it.
>
> **[05-OP-16]** `argmin_reduce(x, axis) -> result` has the signature, dtype,
> exact-comparison, empty-axis, accumulator, and non-differentiability
> contract of [05-OP-15], returning the lowest NaN index when present and
> otherwise the lowest axis index whose stored value is minimal.
>
> **[05-OP-29]** `count(x, axes...) -> result` admits exactly a `bool` tensor operand
> and returns an `int64` tensor whose dimensions are the operand dimensions
> with every selected axis removed. Axis selection follows
> `spec/04-type-system.md` §4.5.3: a concrete-rank operand admits one or more
> unique compile-time `int32` positional axes, while the rank-polymorphic form
> admits one or more unique named axes. Missing axes, mixed positional/named
> axes, duplicate normalized positions or names, absent or ambiguous names,
> dynamic axes, and out-of-range positional axes are type errors. The dedicated
> Count node stores the normalized original-axis positions exactly once in
> strictly descending order, irrespective of source argument order; this is
> also the canonical wire order. For each result position, the selected source elements
> are visited in original row-major order and each true element contributes
> exact `1i64` while each false element contributes exact `0i64`. Their sum
> uses the canonical balanced tree below with checked `int64` addition; an
> unrepresentable result traps `Overflow` as operation `count` at dtype
> `int64`. If any selected extent is zero, the result is `0i64`. This is a
> dedicated reduction and is not a `cast` plus `sum` lowering. It is not a
> composition of nested `count` calls. Numeric, scalar `bool`, `string`, reserved dtype spellings, and all
> other operands are type errors. `count` has no accumulator. A differentiated
> graph reaching `count` is structurally rejected with
> `AdRejectionReason::IntegerIndexOutput`; it never receives a silent zero
> cotangent.
>
> **[05-OP-30]** `sum(x, axes..., accumulator = default(p)) -> result` admits
> exactly a tensor operand whose dtype `p` is an active signed integer or
> active float. `bool`, `string`, reserved dtype spellings, scalar, and all other operands are
> type errors. Axis selection and multi-axis composition follow
> `spec/04-type-system.md` §4.5.3. The selected accumulator has the same
> numeric kind as `p` and must be at least as wide as both `p` and §5.7.1's
> default; omission selects that default.
> Each source value enters the canonical balanced tree below in the
> accumulator dtype, every addition executes and finalizes there, and integer
> overflow is checked at every addition. Empty slices return exact zero in
> the accumulator dtype. The result dtype is exactly §5.7.1's
> `sum_result(p, accumulator)` rule; reduced-float lowering casts either
> admitted accumulator dtype back to operand dtype `p` after the completed
> tree.
> For float operands the adjoint expands the upstream cotangent across every
> removed axis and finalizes it at the operand dtype. Signed-integer forms are
> forward-only and `grad` rejects them. No backend may substitute a left fold,
> stride-4 cascade, library-selected tree, or bool-to-integer promotion.

**Axis:** Integer index into the input rank. Non-negative axes are
zero-indexed from the front. A negative axis indexes from the end:
`-1` is the last axis, `-2` the second-to-last, and so on (an axis `a`
with `a < 0` denotes `rank + a`). After this normalization the axis
must fall within `0..rank`; an out-of-range axis is a type error. This
from-the-end convention applies uniformly to every axis-taking
primitive — the reductions here, `softmax`, `mean`, `gather`,
`scatter`, and the movement and ordering ops — and is the convention
the formula examples below already use (`axis=-1` for the last axis).

> **[05-AXIS-1]** A reduction axis and `expand`'s insert axis SHALL be
> statically resolvable either as an integer constant (a literal or a literal
> wrapped in an integer cast) or as a named dimension of the operand. A
> runtime integer expression and an unknown dimension name are type errors at
> the call site; no lowering or backend SHALL substitute axis zero or another
> axis.

The reduction axis must resolve statically: a literal, a
`cast(N, int32)`-wrapped literal, or a named operand dimension as specified by
`spec/04-type-system.md` §4.5.3. Because the output shape is "remove the
dimension at position `axis`", the type checker cannot determine which
dimension is dropped from a runtime integer value. A reduction whose axis is
a runtime expression (for example a function-parameter `int32`) is rejected
at the reduction call site with a diagnostic naming the constant-or-named-axis
requirement, rather than leaving the output shape unresolved (chelis#259).
The same constraint and diagnostic apply to `expand`'s insert axis.

**Output dimensions:** The dimension at position `axis` is removed. All other dimensions are preserved.

**Runtime-derived operand rank.** When a reduction or `gather` consumes a
windowing or stacking result whose extent is known only at execution, its
operand rank remains determined by the result contract: a reduction operand
has result rank plus one, and a gather operand has
`result_rank - indices_rank + 1`. The removed or gathered axis is represented
as a runtime-derived symbolic dimension and resolves from the operand's
runtime shape. No compiler stage may substitute a rank-zero placeholder or axis zero.

**Accumulator parameter (`sum` only).** The optional `accumulator: prec`
parameter controls the precision used for the running sum. It also controls
the output tensor dtype except for `bf16` and `f16`, whose result returns to
the operand dtype after either admitted accumulator. The exact allowed
operand/accumulator/result matrix, defaults, and rationale are authoritative
in `spec/04-type-system.md` §5.7.1; this primitive doc is the operational
location of the parameter on the IR node.

The default rows are `bf16/f16 -> f32 -> bf16/f16`, `f32 -> f32 ->
f32`, `f64 -> f64 -> f64`, `int8/int16 -> int32 -> int32`, `int32 ->
int32 -> int32`, and `int64 -> int64 -> int64`. An explicit wider
accumulator is equally part of the signature: `bf16` and `f16` additionally
admit `f64` while returning the operand dtype; `f32` admits `f64` and returns
`f64`; and `int8`, `int16`, and `int32` admit `int64` and return `int64`.
Thus every non-reduced-float result has the selected accumulator dtype, not
merely the default or a narrow-integer special case.

There is no implicit precision promotion: omitting the parameter resolves to
the documented default before lowering. The IR `RiscOp::ReduceSum` node
always carries a populated accumulator-precision field. Programs that
explicitly request a narrower-than-default accumulator are a type error per
§5.7.1.

`mean`, `max_reduce`, `min_reduce`, `prod_reduce`, `argmax_reduce`, and
`argmin_reduce` do not take an accumulator parameter. Their exact accumulator
and result rules are the [05-OP-11..16] contracts above; in particular, the
value extrema and product return the operand dtype, while index reductions
return `int64`.

### 2.3.1 Windowed Reduction

| Name | Signature | Semantics | AD adjoint |
|---|---|---|---|
| `reduce_window_max` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int64], strides: List[int64]) -> tensor[..., d1', ..., dn', p]` | Strided windowed max over the last `n` axes | Each window splits `g` equally among equal non-NaN maxima, including infinities, or routes full `g` to its first NaN; contributions accumulate across overlapping windows |
| `reduce_window_min` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int64], strides: List[int64]) -> tensor[..., d1', ..., dn', p]` | Strided windowed min over the last `n` axes | Each window splits `g` equally among equal non-NaN minima, including infinities, or routes full `g` to its first NaN; contributions accumulate across overlapping windows |
| `reduce_window_sum` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int64], strides: List[int64]) -> tensor[..., d1', ..., dn', p]` | Strided windowed sum over the last `n` axes | Each window-source position receives the owning window's `g` (overlap-add over windows covering it) |
| `reduce_window_mean` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int64], strides: List[int64]) -> tensor[..., d1', ..., dn', p]` | Strided windowed mean over the last `n` axes | As `sum`, with each contribution scaled by `1 / window_volume` |

Within each window, `reduce_window_max` and `reduce_window_min` inspect values
in row-major window order. The first NaN is the forward result when any NaN is
present; otherwise the operation returns the selected non-NaN extreme and
preserves the first stored representation among equal extrema, including
infinities. This selection rule also defines the adjoint's first-NaN and
non-NaN tie sets.

The four names are separate builtins rather than a runtime reducer argument.
They may share one closed IR reducer identity, but the checker resolves the
name and reducer before execution.

**Padding mode: Valid only.** Output spatial extent per windowed axis is
`floor((input_dim - window) / stride) + 1`. No implicit `Same` padding is
defined; a program pads explicitly with `pad(x, ..., fill)` before calling a
windowed reduction.

**Shape contract.**

> **[05-RWIN-1]** `reduce_window_*` SHALL receive equal-length, non-empty
> `window_shape` and `strides` lists; their length SHALL NOT exceed the input
> rank; and every entry SHALL be a positive int64. These lists and symbolic
> input extents may be runtime values. A statically proved violation is a type
> error; otherwise the same violation traps `Domain` before any tensor read or
> output allocation. It never becomes an empty-list default, truncated rank,
> static-parameter signature, or backend assertion.

- `window_shape` and `strides` are int64 lists of equal length
  `n >= 1`.
- The trailing `n` axes of the input are the windowed axes. The
  leading `rank(input) - n` axes pass through unchanged.
- Each windowed entry must be a positive int64. `window_shape[i] >= 1`
  and `strides[i] >= 1`.
- The output rank equals the input rank. Leading dims match the
  input; trailing dim `i` is
  `floor((input_dims[rank - n + i] - window_shape[i]) / strides[i]) + 1`.
  When that formula is statically proved non-positive the call is a type
  error; when it becomes non-positive only at execution, it traps `Domain`
  before a tensor read or allocation. An empty window output is structurally
  meaningless under `Valid` padding.

**Primitive identity.** `reduce_window_*` is Tier 1 and carries the closed
`{reducer, window_shape, strides}` parameters. Windowed mean is windowed sum
at the declared arithmetic width followed by division by the positive window
volume at that width. It is not a Tier-2 decomposition into a materialized
window tensor.

*Accumulation precision.* A windowed reduction accumulates at the
operand's ARITHMETIC WIDTH (`spec/04-type-system.md` [04-NUM-8]) in every
lane: `f32` operands accumulate in `f32`, `f64` in `f64`, `f16`/`bf16` at
`f32`, and integers exactly at their width. `reduce_window_*` takes no
accumulator parameter, so §5.7's widening does not apply to it and there
is no other authorized widening.

Window leaves are enumerated in row-major order and combined by the canonical
adjacent-pair balanced tree from §2.3. For sum, every addition finalizes at
the declared arithmetic width and the completed root finalizes once to
operand storage. Mean applies that sum tree, divides the root by the exact
positive window volume at the arithmetic width, and finalizes once to operand
storage. Max/min use the same row-major order for their first-NaN and stored-
representation tie rules. No lane may substitute a fold or library-selected
tree.

For integer operands, a window sum that leaves the operand dtype's range
traps per [04-NUM-3], with occurrence governed by [04-NUM-12]. The global
sum's §5.7.1 widening does not apply because windowed reductions have no
accumulator parameter. A lane may not widen silently.

`reduce_window_sum` deliberately does **not** widen its RESULT precision
the way the global `sum` reduction does; the output element type matches
the operand type (which is also why the adjoint needs no `Cast` — see
**AD policy**).

**AD policy.** `reduce_window_*` is differentiable. `chelis_ir::grad`
lowers the reverse-mode adjoint to a single `RiscOp::ReduceWindowGrad`
node carrying the same `{reducer, window_shape, strides}` triple, taking
`(x, g)` (the forward input and the upstream cotangent) and returning the
input cotangent `din` (shape `S_in`). The adjoints, accumulated over the
(overlapping) windows that cover each input position, are:

- `Sum`: scatter (overlap-add) the owning window's `g` to each
  window-source position — the transpose of the windowed sum.
- `Mean`: as `Sum`, scaling each contribution by `1 / window_volume`.
- `Max` / `Min`: if the window contains NaNs, route the full `g` to the first
  NaN in row-major window order and zero to the rest. Otherwise split `g`
  equally among the `k` positions equal to the selected non-NaN extreme,
  including equal positive or negative infinities, so each
  receives `g / k`; `k` is counted as exact `int64`, converted once to the
  operand float dtype under [04-NUM-14], and ordinary dtype-width `div`
  performs the quotient. Contributions from overlapping windows add. `x` is
  read to locate the selected value and tie set.

For every input position, overlapping windows are enumerated by increasing
row-major output position. Their contributions combine through the same
adjacent-pair balanced tree at operand arithmetic width, then finalize once
to operand storage. This order governs float bits, integer trap occurrence,
and every `ReduceWindowGrad` reducer.

Second-order AD through `ReduceWindowGrad` is the derivative of the executed
first-order graph wherever that graph is differentiable; a transform may not
substitute zero merely because a backend lacks a second-order kernel.

> **[05-RWIN-2]** `ReduceWindow` and `ReduceWindowGrad` are legal tensor
> operations throughout the language. Every implementation preserves the
> reducer, window shape, strides, dtype, exact forward selection, and adjoint
> accumulation contract above. A target-support gap is not authority to change
> their signature, reject them by language design, substitute a stub, or enter
> a panic backstop.

> **[05-OP-39]** `window_reduction(arguments...) -> result` governs exactly
> `reduce_window_sum`, `reduce_window_mean`, `reduce_window_max`,
> `reduce_window_min`, and the internal `ReduceWindowGrad` identity selected by
> each float operation's adjoint. Sum, max, and min admit every active numeric
> tensor dtype; mean admits every active float tensor dtype; bool, scalar,
> string, and reserved dtype spellings are type errors. Each forward result has the
> input dtype and [05-RWIN-1]'s output dimensions. The exact valid-padding
> signature, reducer identity, window/stride validation, per-dtype arithmetic
> width, no-user-accumulator rule, canonical balanced tree, overflow/NaN/tie
> behavior, first-order adjoint, overlap accumulation, and higher-order rule
> are the complete contract immediately above. Integer forms are forward-only;
> float forms use the exact `ReduceWindowGrad` graph. No target-specific rank,
> reducer, dtype, first-order-only, host-fallback, alias, or compatibility
> identity belongs to this atom.

**Reduction order.** `sum`, `prod_reduce`, and `count` use one canonical
balanced tree. Level zero is the reduced slice in its specified positional
order. At each level, pair adjacent values from left to right, apply the
operation to every pair from left to right, carry an odd final value unchanged,
and repeat until one value remains. Empty slices return the operation identity.
Every evaluator and backend uses that logical tree; parallel scheduling may
vary only when it preserves the same result bits and trap occurrence.

For `sum`, source values enter the tree in the selected accumulator dtype and
every addition finalizes at that dtype. For `prod_reduce`, every multiplication
executes at [04-NUM-8]'s arithmetic width and finalizes once to operand storage
before the next tree level. For `count`, leaves are exact `0i64` or
`1i64` and every pair uses checked `int64` addition. These rules govern float
result bits, [04-NUM-12] integer-overflow occurrence, and reverse-mode traversal
of the `prod_reduce` tree. A left fold, stride-4 cascade, library-dependent
tree, or compatibility mode is not conforming.

### 2.4 Movement

| Name | Signature | Semantics |
|---|---|---|
| `reshape` | `(&tensor[D_old,p], shape: List<int64>) -> tensor[D_new,p]` | Reinterpret memory layout. Product of dimensions must match. |
| `permute` | `(&tensor[d1,...,dn,p], axes: int32...) -> tensor[d_axes,p]` | Reorder dimensions. `axes` is a permutation of 0..n-1, passed as one scalar argument per axis. |
| `expand` | `(&tensor[D_small,p], axis: int32, size: int64) -> tensor[D_large,p]` | Insert or set a dimension at position `axis` with width `size` (size-1 broadcast). Does NOT copy data. Named-axis and anchored forms: `spec/04-type-system.md` §4.5.3. |
| `pad` | `(&tensor[D,p], padding: List<List<int64>>, fill) -> tensor[D',p]` | Add elements at boundaries. `padding` specifies (before, after) per axis. |
| `shrink` | `(&tensor[D,p], bounds: List<List<int64>>) -> tensor[D',p]` | Slice: extract a contiguous sub-tensor. `bounds` specifies (start, end) per axis. |
| `stride` | `(&tensor[D,p], strides: int64...) -> tensor[D',p]` | Strided access: take every n-th element along each axis. |

`expand`'s positional form is the (tensor, axis, size) triop; a
two-argument list form is an arity error. The named-axis form
(`expand(x, new, size)` with a dimension name) and the four-argument
anchored form remain as `spec/04-type-system.md` §4.5.3 states them.
Where the axis is positional it is axis-domain `int32`; `size` is
extent-domain `int64` in every form, which makes the canonical broadcast
idiom `expand(b, axis, shape(x, axis))` well-typed by construction.

**Movement AD adjoints:**

| Name | Adjoint |
|---|---|
| `reshape` | `reshape(g, original_shape)` |
| `permute` | `permute(g, inverse_permutation)` |
| `expand` | `sum(g, expanded_axes)` — collapse the expanded dimensions |
| `pad` | `shrink(g, inverse_padding)` — extract the non-padded region |
| `shrink` | `pad(g, inverse_bounds)` — pad gradient back to original size |
| `stride` | [05-MOV-1]'s exact zero-filled inverse sampling map at the original shape; runtime steps have zero cotangent |

**Extent-domain and axis-domain arguments:**

> **[05-DIM-1]** Every movement and shape argument is exactly one of two
> KINDS, and the kind determines its dtype. An EXTENT-DOMAIN quantity
> measures or indexes along an axis - a dimension extent, a slice bound, a
> pad amount, a stride step - and SHALL be `int64`. An AXIS-DOMAIN quantity
> names an axis - a rank index, a permutation entry - and SHALL be `int32`.
> No movement argument is both, and within this movement and shape
> surface no lane SHALL accept one kind's dtype in the other kind's
> position.

> **[05-DIM-2]** `shape(x, axis)` SHALL return `int64`. Its result is an
> extent-domain quantity under [05-DIM-1], and it is the canonical producer
> of the values that reach every extent-domain slot. Its `axis` parameter is
> axis-domain and remains `int32`.

The split is by what the quantity can grow to, not by where it appears. An
extent-domain value scales with the data: a dimension extent, an offset into
one, or a step across one is bounded only by tensor size. An axis-domain
value is bounded by rank, which is small and statically known, so `int32` is
permanent headroom rather than a limit anyone can reach.

[05-DIM-1] scopes to movement and shape arguments. The axis parameters of
reduction and concatenation ops are governed by [05-DIM-3].

> **[05-DIM-3]** Every positional axis-domain argument to an operation SHALL
> be `int32`, including movement, shape, reduction, ordering, gathering,
> scattering, splitting, and concatenation operations. A symbolic dimension
> name used as a rank-polymorphic axis anchor is not a scalar axis-domain
> value and retains the named-dimension rules of spec/04 §4.5.3. The
> `window_shape` and `strides` arguments to every
> `reduce_window_*` operation SHALL be extent-domain `List[int64]`. No
> evaluator, lowering, runtime, binding, or compiled lane SHALL accept a
> wider or narrower substitute for either carrier.

Two things make the distinction normative rather than stylistic. First,
`spec/04-type-system.md` [04-NUM-11]: a value crosses every boundary at its
declared dtype and no stage substitutes a wider representation to
compensate, so a dtype mismatch between an extent's producer and its
consumer cannot be absorbed below the language; it has to be settled in
the signature. Second, a language whose
extent reader is narrower than its extent writer cannot round-trip its own
dimensions: an extent that can be written but not read back at the same
width has not survived the boundary crossing [04-NUM-11] requires it to
survive.

Two boundary notes. Extent-domain slots do not adopt literals: no
`spec/04-type-system.md` §5.6 position reaches a list literal or a
scalar against a remote callee signature, so an extent literal states
`int64` itself, with a suffix or an explicit `cast`
(`reshape(x, [2i64, 2i64])`, `stride(x, 2i64)`), and an unsuffixed
`int32` literal in an extent slot is a type error whose diagnostic names
the fix. §5.6 records why that adoption set stays closed. And
arithmetic on extents is ordinary program arithmetic: it computes at the
declared `int64` width like every other op ([04-NUM-8]), with no narrower
internal substitute. The width of the loop counters and addressing
expressions a backend synthesizes is governed by [04-NUM-8]'s
synthesized-arithmetic clause, not by this section.

#### 2.4.1 Runtime (node-valued) bounds and reshape targets

A movement bound (`pad` before/after, `shrink` start/end, `stride` step) and a
`reshape` target extent are each represented as a `RtDim`:

- `Lit(n)` — a compile-time-constant extent.
- `ToEnd` — the full-axis sentinel; legal only as a `shrink` end (the identity
  slice of a symbolic bystander axis).
- `Node(i)` — a **runtime** extent read from the owning node's `inputs[i]`, a
  rank-0 integer scalar (a `shape()` read or integer arithmetic over one:
  `add`/`mul`/`floor_div`/`neg`/`cast`). The index is absolute: `inputs[0]` is
  always the tensor operand and `inputs[1..]` are the bound scalars.
- `Sym(name)` — a symbolic dim declared elsewhere (e.g. a bystander `batch`);
  legal only as a `reshape` target.

Runtime bounds are validated in every execution mode with matching language
errors: a negative bound, a shrink range overshoot, a non-positive stride
step, a negative reshape target extent, and a reshape target whose element
product disagrees with the input all trap before allocation or access. The reshape numel
guard fires for ANY reshape whose output or input extents are not all static
literals — Sym-resolved targets and literal targets over runtime-sized inputs
included, not only node-valued targets — and same-shape elementwise ops guard
operand-shape agreement at equal rank whenever a non-static extent is
involved (chelis#664; rank-0 scalar operands are the backend's broadcast
idiom and are exempt). A dim whose extent
is computed by the op at run time is an *op-declared* symbolic dim bound from
the executed [05-DIM-1] int64 value; a second site
computing a different value for the same symbol traps (the
over-unification guard). The guard does not fire on a direct-return
`shrink -> stride` chain under one sig symbol: anonymous
dims are not substitution keys, so the sig symbol attaches positionally to
the final op only and each inner movement op declares its own extent. The
checker's movement typing matches: symbolic-dim
pass-through is identity-only (stride step 1 / zero pad; see
spec/04-type-system.md §4.7), so a non-identity movement axis types a
fresh runtime-guarded extent rather than repeating the input's symbol.
The guard remains the soundness floor for a genuinely claimed symbol
equality (e.g. an explicit `-> tensor[n]` over `stride(x, 2i64)`).

The movement adjoints are runtime-capable on the same representation: the
`shrink` adjoint pads with `after = shape(x, axis) - end`, the `pad` adjoint
shrinks to `end = before + shape(x, axis)`, and the `stride` adjoint's
upsample cascade reads `m_a = shape(g, axis)` and trims to
`(0, shape(x, axis))` with a runtime `m_a * step` merge extent — all as
node-valued bounds over fresh `Shape`/arithmetic scalars. For `stride`, the
adjoint allocates exact positive-zero storage at the original shape and maps
each output cotangent at index `(i0, ..., in)` to the unique input index
`(i0*s0, ..., in*sn)` selected by the forward positive steps, visiting output
indices in row-major order. No two outputs select the same input, so there is
no accumulation-order choice. Bound scalars are a **zero-cotangent boundary**:
they are discrete index math, carry exact zero cotangent, and do not pull their
producers (for example a window-count `floor_div`) into a structural
differentiability rejection.

> **[05-MOV-1]** Runtime movement bounds and reshape targets, their validation,
> and the exact adjoints above SHALL be available in every language execution
> mode for every active tensor dtype admitted by the owning movement
> operation. Eval, C, HIP, and Metal execute the same runtime values and
> traps. No lowering may erase a runtime value, substitute a literal bound,
> require host provenance, emit a statically guessed extent, or turn a backend
> implementation gap into a language restriction.

### 2.5 Memory

| Name | Signature | Semantics |
|---|---|---|
| `const` | `(value, shape...) → tensor[shape, p]` | Create a tensor filled with a constant value. Precision inferred from value or annotation. |
| `load` | `(source, shape...) → tensor[shape, p]` | Load tensor data from external source (file, memory). |

`const` and `load` are the pure tensor constructors. Effectful tensor constructors such
as seeded random generation are tracked separately below. All other tensors are derived
from computation on existing tensors.

`const` is not differentiable (it produces a constant — gradient is zero). `load` is not differentiable.

Root-scoped evaluation resolves external loads and symbolic dimensions only
for nodes that can affect the selected roots. Generic declarations from
unrelated dependency modules are dead and cannot create top-level input
requirements (chelis#991). A load that is outside the value-dependency slice
but supplies a symbolic extent to a live node remains a required shape
dependency and fails closed when its input is absent (chelis#351).

#### 2.5.1 Shape query (`shape`)

| Name | Signature | Semantics |
|---|---|---|
| `shape` | `(&tensor[d1,...,dn,p], axis: int32) -> int64` | Runtime extent of the input along `axis`, as a rank-0 integer scalar. |

The Surf `shape(tensor, axis)` builtin types this read as an `int64` scalar
per [05-DIM-2] — extent-domain out, axis-domain in.

#### Runtime extent read atom

> **[05-OP-7]** The runtime extent read (`shape(x, axis)`; C ABI
> `chelis_tensor_shape`) returns the stored extent of `x` along `axis` as
> an exact `int64` ([05-DIM-2]). The read is metadata-exact at every
> tensor dtype `p`: it performs no arithmetic and no width change on the
> stored extent, so [04-NUM-8]'s arithmetic-width table is not engaged
> and the value crosses the boundary exactly ([04-NUM-11]). Its `axis`
> operand is axis-domain `int32` ([05-DIM-1]); a negative value first
> normalizes by adding the rank exactly once, and an axis still outside
> `0..rank` is a loud error. The read has a zero-cotangent adjoint: it
> contributes exact zero to the tensor input and to the discrete axis, does
> not block differentiation of a surrounding graph, and never silently
> becomes a structural `grad` rejection. No accumulator rule applies.

Two semantic use shapes exist, and they are distinct:

- **As an extent argument** to `expand` / `reshape`, a `shape()` read is folded
  into the movement node's `DimExpr` (the output dim), not materialized as a
  value node.
- **As a scalar VALUE** (used in arithmetic, a `mean` divisor, or any other
  value position), a `shape()` read remains a dedicated typed extent operation
  whose int32 `axis` is an ordinary checked runtime value. It produces a
  rank-zero int64 scalar equal to the input's runtime extent and must never
  lower as a `Load`, missing-input default, constant-axis guess, or placeholder.

`shape` reads only the input's shape metadata, never its element values, so it
is a trivial constant with respect to those values: its reverse-mode adjoint
contributes a **zero cotangent** to the input and to the discrete axis (like
`const` / `load`, it does not block AD — a loss that reads a runtime dim
differentiates correctly, with the shape factor contributing nothing). A
symbolic input extent and a computed axis are resolved from the actual runtime
values rather than baked at codegen time.

> **[05-SHAPE-1]** A scalar `Shape` value is legal in every language execution
> mode for every active tensor element dtype and for a literal or computed
> int32 axis. Its exact int64 result and zero-cotangent rule are
> target-independent. No lowering may replace the axis or extent with a
> constant, default, stub, host-width integer, or compile-time-only signature.

A `shape()` read whose `axis` is data- or metadata-derived remains the same
operation as a literal-axis read. Using the extent as a runtime movement-op
bound or reshape target (a `shrink`/`stride`/`pad` bound or window count
derived from a `shape()` value, and the integer arithmetic feeding it) uses
§2.4.1's node-valued `RtDim` capability; it does not allocate a second shape
operation or a compile-time-only alias.

### 2.6 Effectful Primitive

| Name | Signature | Semantics | AD / effect note |
|---|---|---|---|
| `dropout` | `(&tensor[D, p_float], p_float) -> tensor[D, p_float]` | Apply [05-OP-37]'s inverted-dropout transform using the active `with seed(...)` handler and a same-dtype rate | Introduces `Random`. Its pathwise adjoint reuses the exact forward mask. |
| `uniform_like` | `(&tensor[D, p_float], p_float, p_float) -> tensor[D, p_float]` | Create a tensor matching the input shape and float dtype, filled by the deterministic affine sampler defined by [05-OP-8] under the active `with seed(...)` handler | Introduces `Random`. The template values are not observed; its adjoint is the zero cotangent. |
| `process_run` | `(String, List[String]) -> (Int64, String, String)` | Run an external program with the given argv and capture `(exit_code, stdout, stderr)`. Arguments are passed straight to the OS as argv (no shell, no interpolation), so a value in the args list cannot inject extra shell commands. A process killed by a signal reports exit code `-1`. | Introduces `Io`; it is outside AD. |

> **[05-OP-8]** `uniform_like(template, low, high) -> result` admits every
> active float template dtype `p` in spec/04 §1.1, requires `low` and `high`
> to have that same dtype `p`, and returns `tensor[D, p]` with the template's dimensions. For flat
> element index `i`, SplitMix64 over the handled seed and `i` supplies a
> 53-bit unit value `u` in `[0, 1)`. Both bounds must be finite and `low <=
> high`. At the selected arithmetic width, `high - low` must also be finite.
> These checks, including the equal-bound case, complete before the operation
> consumes a Random call ordinal; failure traps `Domain` as `uniform_like`
> and consumes none. Equal bounds are valid and produce that stored value.
> For `p = f64`, the element is the one f64 fused multiply-add
> `fma(high - low, u, low)`. For `p = f32`, it is the one f32 fused
> multiply-add `fma(high - low, round_f32(u), low)`. For `p = f16` or `bf16`,
> the stored bounds widen exactly to f32, their difference and the fused
> multiply-add execute once in f32 using `round_f32(u)`, and the result narrows
> exactly once to `p`. There is no f32 public-bound signature, default bound,
> or f64 intermediate. Ordinary final rounding may produce the stored high
> endpoint even though `u < 1`.
>
> The operation introduces `Random` and does not observe the template's
> element values. Under the fixed handled stream used by the forward pass, its
> pathwise adjoint contributes zero to the template and, in increasing
> row-major output order, contributes `g_i * (1-u_i)` to `low` and `g_i *
> u_i` to `high`, with every primitive executed at `p`'s declared arithmetic
> width and each scalar contribution combined by the canonical adjacent-pair
> balanced tree. The sampled `u_i` values are the exact forward values at that
> arithmetic width. It has no accumulator parameter.

> **[05-OP-37]** `dropout(input, rate) -> result` admits every active float
> dtype `p`, requires `input: &tensor[D,p]` and a scalar `rate: p`, and returns
> `tensor[D,p]`. The rate must be finite and satisfy `0 <= rate < 1`; validation
> completes before Random consumption, and failure traps `Domain` as `dropout`
> while consuming no call ordinal. The accepted call consumes exactly one
> ordinal, including for an empty tensor or `rate = 0`.
>
> For flat element index `i`, [05-RNG-1] supplies the same arithmetic-width
> unit value used by [05-OP-8]. The saved forward mask drops the element exactly
> when that value is less than the rate widened exactly to the arithmetic
> width. A dropped element is positive zero at `p`. A kept element computes
> the exact graph `denom = sub(1p, rate)` then `div(input[i], denom)`, with
> each named primitive finalized to `p` before its consumer under [04-NUM-8].
> For f16 and bf16, `sub` exact-widens its stored operands to f32 and narrows
> its result to `p`; `div` then exact-widens the stored `input[i]` and `denom`
> to f32 and narrows its result to `p`.
> Overflow and NaN follow [04-NUM-2]; no f32 public-rate signature, f64 funnel,
> unscaled-dropout alias, or special `rate >= 1` default exists.
>
> Under the fixed handled stream, the pathwise adjoint reuses the exact saved
> mask. A dropped input receives positive zero. A kept input receives its
> output cotangent through the exact graph `denom = sub(1p, rate)` then
> `div(g_i, denom)`. The scalar rate receives, for every kept element in
> increasing row-major order, the exact graph `denom = sub(1p, rate)`,
> `denom_sq = mul(denom, denom)`, `numerator = mul(g_i, input[i])`, then
> `div(numerator, denom_sq)`; `1p` is the exact integer one represented at `p`
> and every named primitive finalizes before its consumer under [04-NUM-8].
> Dropped elements contribute positive zero and the contributions combine by
> the canonical adjacent-pair balanced tree. The mask comparison itself has
> zero cotangent. The operation has no accumulator parameter.

> **[05-RNG-1]** Every conforming evaluation of a `with seed(N)` program
> produces byte-identical random results for the same seed, dynamic
> random-call ordinal, element index, bounds, operation, and dtype. Every lane
> that supports that operation and dtype produces the same stream and stored
> result bits; compiler version and target do not vary this result. Reinterpret the
> signed int64 seed as its uint64 two's-complement bits. For zero-based call
> ordinal `c` and flat element index `i`, the source word is
> `splitmix64(seed_bits XOR rotl64(splitmix64(c),17) XOR
> rotl64(splitmix64(i),41))`; the unit value is the exact rational formed by
> its high 53 bits divided by `2^53`. `splitmix64(x)` is the standard fixed
> map: add `0x9E3779B97F4A7C15`, xor-shift 30 and multiply by
> `0xBF58476D1CE4E5B9`, xor-shift 27 and multiply by
> `0x94D049BB133111EB`, then xor-shift 31, all modulo `2^64`.
> Each entered random primitive consumes exactly one call ordinal, even for
> an empty tensor or a later trap after Random consumption begins; validation
> that precedes Random consumption consumes none. Two different accepted
> seeds define different source streams. The RNG is deterministic, not
> cryptographic.

---

## 3. Derived Built-Ins (Tier 2)

These are convenience functions emitted by the desugarer. Their typed
identities may appear in intermediate IR until each governing atom's semantic
transforms have run; backend-facing RISC DAGs contain only the resulting
primitive compositions. A lowering table in this section defines forward
values, not permission to erase an identity before its adjoint or rejection
rule has been applied.

### 3.1 Arithmetic

| Name | Lowering to RISC |
|---|---|
| `sub(a, b)` | `add(a, neg(b))` |

Note: `div` and `neg` are Tier 1 RISC primitives (see §2.1, §2.2),
not Tier 2 derived built-ins. `recip` is also a Tier 1 primitive
(§2.2). No Tier 2 operation lowers `div` through `log` and `exp`.

### 3.2 Comparison and Logical Operations

| Name | Integer lowering | Float lowering |
|---|---|---|
| `cmplt(a, b)` | `cmplt(a, b)` | `and(not(nan), cmplt(a, b))` |
| `lt(a, b)` | `cmplt(a, b)` | `and(not(nan), cmplt(a, b))` |
| `eq(a, b)` | `not(or(lt, gt))` | `and(not(nan), not(or(lt, gt)))` |
| `neq(a, b)` | `or(lt, gt)` | `or(nan, or(lt, gt))` |
| `gt(a, b)` | `gt` | `and(not(nan), gt)` |
| `gte(a, b)` | `not(lt)` | `and(not(nan), not(lt))` |
| `lte(a, b)` | `not(gt)` | `and(not(nan), not(gt))` |

Here `lt = cmplt(a,b)`, `gt = cmplt(b,a)`, and, on floats only,
`nan = or(is_nan(a),is_nan(b))`. These lowerings define result values over
already-evaluated operands. They never reorder operand-expression evaluation:
every application evaluates `a` before `b` under `spec/03-deep-syntax.md`
§4.4, and only the value computation may read the resulting values in reverse
order.

> **[05-OP-26]** `and(left, right) -> result` admits exactly two `bool`
> scalars or two `bool` tensors with identical dimensions. It returns `bool`
> on the same surface and, for tensors, with those dimensions. Applications
> evaluate `left` and then `right` before combining their values; `and` is not
> short-circuiting. The result is true exactly when both operands are true and
> is false otherwise, applied element-wise for tensors. A mixed surface,
> mismatched tensor dimensions, or any non-`bool` operand is a type error. The
> operation is pure, performs no arithmetic or dtype conversion, has no
> accumulator, and is non-differentiable: `grad` rejects it.
>
> **[05-OP-27]** `or(left, right) -> result` has the signature, surface,
> shape, evaluation-order, rejection, purity, accumulator, and differentiation
> contract of [05-OP-26]. Its result is true exactly when either operand is
> true and is false otherwise, applied element-wise for tensors.
>
> **[05-OP-28]** `not(value) -> result` admits exactly one `bool` scalar or
> `bool` tensor and returns `bool` on the same surface and, for a tensor, with
> the same dimensions. Its result is true exactly when `value` is false and is
> false exactly when `value` is true, applied element-wise for tensors. Any
> non-`bool` operand is a type error. The operation is pure, performs no
> arithmetic or dtype conversion, has no accumulator, and is
> non-differentiable: `grad` rejects it.

> **[05-OP-36]** `comparison(left, right) -> result` governs exactly the seven
> language identities `cmplt`, `lt`, `eq`, `neq`, `gt`, `gte`, and `lte`.
> The five ordered identities `cmplt`, `lt`, `gt`, `gte`, and `lte` admit two
> active-numeric scalars of one dtype or two tensors of one active numeric
> dtype and identical dimensions. `eq` and `neq` additionally admit bool
> scalars and same-shaped bool tensors, string scalars, unit, and two `List`,
> tuple, `Dict`, `Option`, or ADT values of one static type whose reachable
> fields are recursively admitted by this equality rule. Functions and
> resource handles are not equality-comparable. Ordered comparison
> of bool, string, or a structured value is a type error.
>
> Scalar and recursive equality return one bool scalar; tensor comparisons are
> element-wise and return a bool tensor with the operand dimensions. Mixed
> surfaces, numeric dtypes, static structured types, or tensor dimensions are
> type errors. Signed integers use exact mathematical order at their stored
> width. Bool and string equality compare their exact stored values without
> Unicode normalization. Unit equals unit. Lists and tuples compare length and
> then corresponding fields in order; `Option` and ADT values compare exact
> constructors and then fields in order. Dictionaries compare key/value sets
> independent of insertion order using [05-OP-32]'s exact key equality and
> this rule recursively for values. A tensor reached as a field of a recursive
> value compares equal only when dtype, dimensions, and every row-major element
> compare equal; two equal-shaped empty tensors compare equal. This recursive
> tensor case returns one scalar predicate, while two direct tensor operands
> retain the element-wise tensor result above.
>
> On float leaves, any NaN makes `cmplt`, `lt`, `eq`, `gt`, `gte`, and `lte`
> false and makes `neq` true; otherwise ordinary IEEE numeric comparison
> applies, with signed zeros equal. `neq` is the logical complement of `eq`
> for every non-float admitted value. The tabled ordered formulas are exact
> and may use [05-OP-20] plus [05-OP-26..28] without sending bool through
> arithmetic IR. These seven operations remain distinct typed comparison
> identities through AD and other semantic transforms; logical expansion may
> occur only after the zero-cotangent adjoint is registered for the exact
> identity. Thus [05-OP-26..28]'s structural `grad` rejection does not replace
> this family's adjoint. The family performs no numeric conversion, has no
> accumulator, and contributes zero cotangent to every differentiable leaf.
> No identity has an alias, grandfathered path, deprecated spelling, or
> compatibility wrapper.

Logical operations do not alias arithmetic primitives. An implementation may
use an internal representation-specific lowering only when it preserves the
three atoms above and never admits `bool` to a numeric capability or kernel.

### 3.3 Activation Functions

| Name | Lowering to RISC |
|---|---|
| `relu(x)` | `max_elem(x, const(0.0, x.shape))` |
| `sigmoid(x)` | `recip(add(const(1.0), exp(neg(x))))` |
| `tanh(x)` | Hyperbolic tangent, equivalently `sub(mul(const(2.0), sigmoid(mul(const(2.0), x))), const(1.0))` |
| `silu(x)` | `mul(x, sigmoid(x))` |
| `gelu(x)` | The tanh approximation `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))` |

All five activation functions admit float tensors and float scalars at f16,
bf16, f32, and f64. The scalar form returns the same scalar dtype and is the
rank-zero instance of the tensor operation; non-float operands are type
errors. Each RISC primitive in the lowering computes at [04-NUM-8]'s declared
arithmetic width and finalizes to the operand's storage width before the next
primitive observes it, as required by [04-NUM-1]. The adjoint is the
derivative of the lowering above, with `relu` using §2.1's `max_elem`
subgradient convention.

### 3.4 Higher-Level Operations

| Name | Lowering to RISC |
|---|---|
| `matmul(A, B)` | See §4.1 |
| `softmax(x, axis)` | See §4.2 |
| `linear(x, w, b)` | `add(matmul(x, w), b)` (with appropriate expand on b) |
| `cross_entropy(logits, labels)` | See §4.3 |
| `min_elem(a, b)` | `neg(max_elem(neg(a), neg(b)))` |

`mean` remains a Tier 2 builtin. Its complete contract is [05-OP-11], and its
lowering is that atom's exact guarded sum-then-div composition.

`normalize` is not a Chelis builtin. An application to that undeclared name is
a type error; a program spells its intended normalization as an explicit
composition of the operations governed here.

### 3.5 Lowering Helpers And Sparse Implementation Nodes

The following names appear in lowering narratives (§4) as pseudocode or
pattern-matched operations. Most decompose into Tier 1 primitives. `cos` is a
first-class unary primitive `RiscOp::Cos` (see §2.2), alongside `tan`,
`atan`, `abs`, `floor`, and `ceil`, none of which decompose.

| Helper | Decomposes to |
|---|---|
| `argmax(x, axis)` | comparison chain via `cmplt` + `max_elem` |
| `gather(x, idx, axis)` | one-hot encoding via `reshape`, `expand`, `mul`, `sum` |
| `im2col(x, kh, kw, ...)` | `stride`, `pad`, `reshape`, `permute` |
| `where(cond, a, b)` | Element-wise selection of `a` where `cond` is true and `b` where it is false; the boolean condition is not converted to or combined through a numeric dtype |

The sparse operations lower to the first-class
IR nodes `RiscOp::Gather { axis }`, `RiscOp::ScatterAdd { axis }`,
`RiscOp::Scatter { axis }`, and `RiscOp::ScatterElements { axis }` (the
element-wise ONNX `ScatterElements`, §3.5.1). Tensor-lane Surf
`gather(values, indices, axis)` lowers directly to `RiscOp::Gather`, not to a
host-runtime call or dense one-hot materialization. The tensor-lane Surf
builtin `scatter_replace(base, indices, updates, axis)` lowers directly to
`RiscOp::Scatter` for the last-write-wins case. The shared specialization pass
also recognizes the internal `RiscOp::OneHot { vocab } + Expand + Mul + Sum`
gather tree and collapses it before DCE/codegen. Arbitrary const/eq
one-hot encodings are not recognized because they do not preserve the original
index operand.

#### Replace-scatter vs scatter-add

> **[05-SPARSE-1]** `Gather`, `ScatterAdd`, `Scatter`, and
> `ScatterElements` SHALL take int32 or int64 indices. For the three scatter
> forms, target, updates, and output SHALL have identical precision. A
> different index dtype or a precision mismatch is a type error, never an
> implicit cast.

`Scatter` and `ScatterAdd` are intentionally distinct primitives. Both
take inputs `(target, indices, updates)` with the same shape contract
(updates shape equals `target.dims[..axis] ++ indices.dims ++
target.dims[axis+1..]`) and the same precision constraints
(int32/int64 indices; target/updates/output precision identical).
They differ only in how duplicate target indices are resolved and in
their AD policies:

> **[05-SPARSE-2]** `RiscOp::OneHot` is an internal specialization marker,
> not a backend operation. Specialization SHALL consume it or lower it to
> ordinary primitive IR before backend emission. A backend boundary that
> encounters it SHALL reject the broken compiler invariant; it SHALL NOT
> emit a placeholder result.

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

#### 3.5.1 Element-wise scatter (`ScatterElements`)

`Scatter` / `ScatterAdd` above have **hyperplane** semantics: each
scattered index fans out a whole trailing hyperplane (the inverse of
`Gather`). ONNX `ScatterElements` instead writes **one element per
index**, so a third primitive `RiscOp::ScatterElements { axis }`
exists for it. (ONNX `ScatterND` is the N-D generalization and is not
covered by this primitive.)

`ScatterElements` takes inputs `(data, indices, updates)` with the
**element-wise** shape contract:

```
rank(data) == rank(indices) == rank(updates)
indices.shape == updates.shape          (NOT data.shape)
output.shape == data.shape
```

Precision constraints match the hyperplane scatters: `indices` is
`int32` or `int64`; `data`, `updates`, and `output` share one
precision. `axis` is in `0..rank(data)`; on every axis other than
`axis`, `indices.shape[d] <= data.shape[d]`.

**Semantics.** Initialize `output = data`. Then for each coordinate
`c` over `indices` (equivalently over `updates`, same shape), let
`j = indices[c]` and write

```
output[c[0], ..., c[axis-1], j, c[axis+1], ..., c[rank-1]] = updates[c]
```

`j` must satisfy `0 <= j < data.shape[axis]`. Note that the index
substitutes only the `axis` component of `c`; every other component
of the write coordinate comes from `c` directly. (In the 1-D case the
element-wise and hyperplane contracts coincide, which is why a 1-D
`scatter` already round-trips ONNX `ScatterElements`.)

**Duplicate-index semantics.** Last-write-wins under the same
deterministic order as `Scatter`: updates-tensor row-major (C order)
flat iteration. The C backend emits a single-threaded sequential loop
and the HIP backend a `<<<1, 1>>>` serial kernel, for the same
race-freedom reason given for `Scatter`.

**AD policy.** Fail-closed, identical to `Scatter`:

```rust
AdError::NotSupported {
    op: "scatter_elements",
    reason: AdRejectionReason::NonDeterministicAtDuplicateIndices,
}
```

The tensor-lane Surf builtin
`scatter_elements(data, indices, updates, axis)` lowers directly to
`RiscOp::ScatterElements`.

### 3.6 Host-Runtime Operations

The operations in this section execute in the language's host runtime. A
compiled program invokes the same typed host-runtime contract as evaluation;
host execution is not a lesser language lane and does not change legality.

> **[05-HOST-1]** A host-runtime operation SHALL preserve its complete checked
> signature, effects, exact dtype identity, evaluation order, traps, and value
> result in every language execution mode. A target implementation SHALL NOT
> substitute a stub, default value, null pointer, erased dtype, or alternate
> helper contract. Device-kernel nesting is governed by the operation's effect
> and device-boundary rules; it does not make the host operation illegal.

| Name | Signature | Semantics |
|---|---|---|
| `tensor_scan` | `(initial: T, fn: (T, int64) -> T ! E, n: int64) -> tensor[n, T] ! E` | Iteratively apply `fn(prev, i)` for `i in 0..n` and collect the `n` resulting values into a rank-1 tensor whose precision matches `T`. |

`T` must be a scalar primitive (`int8`..`int64`, `f16`..`f64`,
`bool`). The output is owned, contiguous, rank-1, and its
precision equals the dtype of `initial`. The iteration order is the
positional integer sequence `0, 1, ..., n - 1`. Callback effects `E` occur
exactly once per iteration in that order; when `n = 0`, the result is empty
and the callback is not invoked, so no callback effect occurs.

The `tensor_scan` accumulator and emitted elements remain at `T` for every
step. Each callback result is finalized once at `T` before it becomes the
next accumulator and before it is stored in the output. Integer callbacks use
checked declared-width arithmetic and propagate their trap; float callbacks
use [04-NUM-8]'s arithmetic width and storage finalization. No scalar travels
through f64 merely because the helper executes in the host runtime. This is
the same exact tagged-carrier rule as [04-NUM-11] and [05-OP-31].

`tensor_scan` runs in constant stack space with respect to `n`. On float `T`,
reverse-mode differentiation is the reverse traversal of the exact executed
recurrence: cotangents from the returned elements and later recurrence states
combine at each callback invocation in reverse iteration order, using that
callback's ordinary adjoint. The integer and bool forms are forward-only.
`vmap` maps `initial` and every mapped callback capture pointwise while the
int64 iteration index and `n` remain shared; every mapped lane executes the
same positional iteration sequence.

**Negative parity for `tensor_scan`**: a non-callable second argument,
a wrong-arity call, a negative `n`, or a callback that returns a
different dtype than the initial value's dtype are rejected with
`tensor_scan`-tagged diagnostics. A callback with an effect unavailable under
the enclosing handler is rejected by the ordinary effect rules; neither an
unreachable definition nor another definition's effects change this call's
legality.

### 3.6.1 The `test_*` assertion family

The `Test`-effect assertion builtins are exactly `test_assert`,
`test_assert_eq`, `test_assert_close_tensor`, and `test_assert_eq_tensor`.
They execute in the host runtime in every language execution mode. An
assertion evaluates its operands left to right and either returns unit or
traps `Test` with its supplied label and the operation name.

> **[05-HOST-3]** `test_assert` admits bool. `test_assert_eq` admits exactly
> [05-OP-36]'s scalar and recursive equality domain. `test_assert_eq_tensor`
> admits two same-shaped tensors of one active tensor element dtype.
> `test_assert_close_tensor` admits two same-shaped tensors and a tolerance at
> one active float dtype and follows [05-OP-35]'s exact closeness rule. The
> assertion identities are generic; dtype-named or rank-named aliases do not
> exist. An implementation SHALL NOT emit an inert assertion, default value,
> compatibility helper, or whole-module rejection based on an unreachable
> assertion.

> **[05-OP-38]** `host_numeric_builtin(arguments...) -> result` governs
> exactly these five numeric-capacity identities and signatures:
>
> | identity | exact signature |
> |---|---|
> | `tensor_scan` | `(T,((T,int64)->T!E),int64)->tensor[n,T]!E` |
> | `process_run` | `(string,List[string])->(int64,string,string)!{IO}` |
> | `test_assert_eq` | `(Q,Q,string)->unit!{Test}` |
> | `test_assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
> | `test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
>
> Here `T` is one active numeric or bool scalar type, `Q` is one static type in
> [05-OP-36]'s scalar or recursive equality domain, `p_float` is one active
> float dtype, and `p` is one active tensor element dtype. Repeated variables
> denote the same type, dtype, rank, and dimensions. `tensor_scan` has
> [05-HOST-1]'s exact-width recurrence and adjoint; `process_run` has §2.6's
> argv, exit-code, capture, `Io`, and outside-AD contract; and the assertion
> family has [05-HOST-3] and [05-OP-35]'s equality, own-width closeness,
> left-to-right evaluation, zero-cotangent, and `Test` behavior. None has a
> numeric accumulator other than `tensor_scan`'s explicitly typed recurrence
> state. No dtype-named, rank-named, evaluator-only, legacy, or compatibility
> identity is part of this atom.

### 3.6.2 Sequence-padding builders

> **[05-OP-9]** `pad_sequences(sequences: List[List[T]], pad: T) ->
> tensor[len(sequences), width, T]` admits every active tensor element dtype
> `T` in spec/04 §1.1, including `bool`; `string` and every reserved dtype are
> type errors.
> `width` is the greatest source-row length, or
> zero when the outer list is empty. Result element `(r, c)` is
> `sequences[r][c]` when `c < len(sequences[r])`, and `pad` otherwise.
> Every source and padding element is moved at its declared dtype `T` with
> no arithmetic, widening, narrowing, or other rounding. The operation is
> differentiable when `T` is a float dtype. Given output cotangent `g`, the
> source cotangent preserves the outer and inner runtime List shapes exactly,
> and source element `(r, c)` receives `g[r, c]`. The `pad` cotangent is the
> sum of `g[r, c]` over padded result cells in increasing row-major `(r, c)`
> order, combined at dtype `T` by the canonical adjacent-pair balanced tree
> with an exact positive-zero base leaf. When there are no padded cells that
> cotangent is positive zero. For integer or bool `T`, the operation is
> forward-only and its source and `pad` components are non-differentiable
> under spec/06 §2.1. The operation has no public accumulator parameter.

> **[05-OP-10]** `pad_sequences_to(sequences: List[List[T]], width: int64,
> pad: T) -> tensor[len(sequences), width, T]` has the same dtype,
> element-movement, float-domain differentiation, and no-public-accumulator
> rules as [05-OP-9]. `width` SHALL be non-negative. Result element `(r, c)` for
> `0 <= c < width` is `sequences[r][c]` when that source element exists,
> and `pad` otherwise; source elements at index `width` or beyond do not
> appear in the result. For float `T`, given output cotangent `g`, a source
> element `(r, c)` receives `g[r, c]` when `c < width` and exact positive zero
> otherwise, so truncated source cells remain present in the nested List
> cotangent. The `pad` cotangent uses [05-OP-9]'s exact traversal, arithmetic,
> tree, and positive-zero rule over the padded result cells. `width` is
> non-differentiable.

### 3.6.3 Canonical value-to-string conversion

> **[05-OP-25]** `to_string(value) -> result` borrows exactly one value
> without consuming it and returns `string`. It admits unit; an active
> numeric, `bool`, or `string` scalar; a tensor whose element dtype is active;
> or a `List`, tuple, `Dict`, `Option`, or ADT whose reachable values are
> recursively admitted by this rule. Functions and resource
> handles are type errors. A `string`
> scalar returns `value` byte-for-byte unchanged. Numeric and boolean scalar
> and tensor elements render under [05-OBS-1..5] and §8.1. A rank-`r > 0`
> tensor with dimensions `[d0, ..., d_(r-1)]` and `N` elements renders as
> `tensor(shape=[d0, ..., d_(r-1)], data=[...])`, visiting elements in
> row-major order: all `N` elements when `N <= 32`, otherwise the first 32
> followed by [05-OBS-5]'s marked `, ...` cut. A rank-zero tensor renders as
> its bare element under [05-OBS-4]. A `List` renders as `[` followed by every
> element's recursive rendering in source order, separated by `, `, and then
> `]`; `[]` is the empty-list rendering. A List boundary never truncates or
> elides elements, although a tensor nested within it retains [05-OBS-5]
> tensor truncation. Unit renders `()`. A tuple renders `()`, `(x,)`, or
> `(x, y, ...)` for zero, one, or multiple fields. A dictionary renders `{}`
> or `{key: value, ...}` in [05-OP-32]'s canonical key order. `None` renders `None` and
> `Some(x)` renders with that constructor spelling. An ADT renders its exact
> constructor name alone when it has no fields and as `Ctor(x, y, ...)`
> otherwise. Each nested value uses this same rule. String elements are inserted verbatim, without quoting or
> escaping: this is a non-injective display form, not a serialization. Every
> lane produces byte-identical text for the same admitted stored value. The
> operation is pure, performs no arithmetic or dtype conversion, is
> non-differentiable (`grad` rejects it), and has no accumulator.

### 3.7 Host-Lane Data I/O Numeric Operations

The sole public JSON value family is `io/json::Json` and its `JsonNull`,
`JsonBool`, `JsonInt`, `JsonFloat`, `JsonString`, `JsonArray`, and
`JsonObject` constructors from [05-OP-34]. Its operations are the exact
`io/json::*` definitions in [05-OP-35]. A second prelude `Json`, `JInt`/`JNum`
constructors, `j*` helpers, or same-named builtin aliases do not exist. CSV is
an untyped text table `List[Dict[string,string]]`; its `parse_csv`/`to_csv` and
`csv_*` accessors never reuse JSON constructors as a cell type. These
operations and `round_to` execute in the host runtime in every language
execution mode. None is a device tensor primitive. Unless a numbered atom
states an adjoint, the data-I/O operation is outside AD and has no accumulator.

The atoms below are the normative numeric authority for the family, in
the sense `spec/design/capability_table.md` §New numeric ops requires: a
callable in these families has exactly the numeric behavior its
governing atom states, and a numeric behavior no atom governs does not
ship. `io/json::Json`'s numeric capacity (`JsonInt(int64)` beside
`JsonFloat(f64)`) is decided by [05-OP-2] and its exact ADT identity by
[05-OP-34].

#### Host-effect execution atom

> **[05-HOST-2]** JSON and CSV operations, `round_to`, and `process_run` are
> legal host-runtime operations in every language execution mode. Pure parsing,
> projection, serialization, and rounding retain their stated purity; file and
> process operations retain their declared `Io` effect and observable order.
> A compiled host execution SHALL produce the same typed result or language
> trap as evaluation. A device-only kernel may not perform `Io`, but that
> effect-boundary fact SHALL NOT be represented as a language-wide rejection,
> inert stub, default value, or evaluator-only signature.

#### Decimal rounding atom

> **[05-OP-1]** `round_to(x, places) -> r` performs decimal rounding at
> a digit boundary: `r` is the decimal number with `places` fractional
> digits nearest to the EXACT binary value of `x`, ties resolved to the
> even final digit (IEEE 754 roundTiesToEven at a decimal boundary; the
> semantics of Python's `round`), finalized ONCE to the operand's own
> storage width. The supported operand dtypes and result dtypes are:
>
> | operand dtype | computation | result dtype |
> |---|---|---|
> | `f64` | exact decimal rounding of the exact binary value, one final rounding to f64 | `f64` |
> | `f32` | exact decimal rounding of the exact binary value, one final rounding to f32 | `f32` |
> | `f16` | exact decimal rounding of the exact binary value, one final rounding to f16 | `f16` |
> | `bf16` | exact decimal rounding of the exact binary value, one final rounding to bf16 | `bf16` |
> | integer, bool, tensor | type error | — |
>
> No lane may compute an operand at any width other than the operand's
> own ([04-NUM-8]; the decimal rounding itself is exact, so the single
> finalization is the only rounding). `places` is a scalar at any active
> signed-integer dtype and its complete mathematical integer domain is
> admitted. For `places >= 0`, the exact binary rational is rounded to the
> nearest multiple of `10^(-places)`; for `places < 0`, it is rounded to the
> nearest multiple of `10^(-places)` to the left of the decimal point. Ties
> select the multiple whose integer coefficient is even. Powers, quotient,
> and remainder are mathematical intermediates and do not acquire a host-width
> overflow limit. A sufficiently large positive `places` is the identity once
> the quantum is finer than the exact binary rational's finite decimal
> expansion. A finite result of zero preserves the operand's sign, including
> for a sufficiently negative `places`; finalization then follows [04-NUM-2]. A non-finite operand
> passes through with its stored bits unchanged. `round_to` is piecewise
> constant: a differentiated graph containing it is structurally rejected
> with `AdRejectionReason::PiecewiseConstant`, rather than receiving a silent
> zero cotangent. It has no accumulator.

#### Numeric ingestion atom

> **[05-OP-2]** Ingestion preserves the source format's numeric
> distinctions ([04-NUM-11]; `spec/design/dtype_semantics.md` §C6 "type
> the boundary"). A JSON number token containing `.`, `e`, or `E` SHALL
> ingest as `JsonFloat` carrying the correctly-rounded f64 of the token; any
> other number token SHALL ingest as `JsonInt` carrying its exact int64
> value. An integer-form token outside int64 range is a loud `Overflow`
> error; punctuation never selects a lossy float fallback for an integer.
> A float-form token whose f64 image is non-finite is a loud error. CSV cells are TEXT at
> parse time (no inferred numeric type); numeric meaning is assigned
> only by an accessor, under the JSON number grammar with surrounding
> ASCII space/tab tolerated: float accessors accept the full grammar,
> integer accessors accept only its integer subset (a `.`/`e`/`E`
> production refuses loudly, naming the float accessor), and an
> integer cell outside int64 range is a loud Overflow-kind error, never
> an f64 fallback. An empty or non-conforming cell is a loud error in
> every numeric accessor — no NaN, no default, no skip.

#### Exact read atom

> **[05-OP-3]** `io/json::json_int` returns the stored `JsonInt` int64
> exactly and returns `None` for every other variant. `io/json::json_float`
> returns a stored `JsonFloat` f64 exactly; on `JsonInt` it performs the named
> lossy int64-to-f64 widening (exact for magnitudes at or below 2^53), and on
> every other variant it returns `None`. It never truncates or rounds a float
> into an integer. The exact numeric CSV identities and argument order are:
>
> | identity | exact signature |
> |---|---|
> | `csv_int` | `(List[Dict[string,string]], int64, string) -> int64` |
> | `csv_ints` | `(List[Dict[string,string]], string) -> List[int64]` |
> | `csv_f64` | `(List[Dict[string,string]], int64, string) -> f64` |
> | `csv_f64s` | `(List[Dict[string,string]], string) -> List[f64]` |
> | `csv_nrows` | `(List[Dict[string,string]]) -> int64` |
>
> The scalar accessors take `(table, zero_based_row, column)` in that order.
> The plural accessors take `(table, column)` and return exactly one value per
> source row in source order, so the result List has the table's runtime
> length. CSV integer reads (`csv_int`, `csv_ints`) parse the selected text
> cell under [05-OP-2]'s exact integer grammar and return int64; CSV float
> reads (`csv_f64`, `csv_f64s`) parse the full finite f64 grammar. Structural
> counts (`csv_nrows`) return the outer List length as exact int64. A missing
> column, out-of-range row,
> empty/nonconforming cell, or wrong numeric grammar is a loud error naming the
> exact accessor; no JSON variant or default cell is fabricated. These text
> parsers are structurally rejected inside `grad`; they have no cotangent and
> may not be replaced by a silent zero. They have no accumulator.

#### Exact construction atom

> **[05-OP-4]** `JsonFloat(value)` accepts exactly f64 and
> `JsonInt(value)` accepts exactly int64; every other operand width is a type
> error naming an explicit checked cast. No construction path widens or narrows a
> numeric value: the constructed `io/json::Json` document feeds the byte-exact
> serialization channel of [05-OP-5], and a silent f32-to-f64 widening
> would serialize the f32 literal's image (`0.1f32` as
> `0.10000000149011612`) rather than the value the program stated.

#### Numeric serialization atom

> **[05-OP-5]** `io/json::to_json` emits a stored `JsonInt` int64
> as its exact decimal digits with no fractional part and no float
> round-trip, and a stored f64 through the [05-OBS-1] shortest-
> round-trip channel (`format_element` at `f64`; the §8.1 grammar —
> every finite emission parses back to the identical f64 and is a valid
> JSON number token). A non-finite `JsonFloat` is a loud serialization error.
> Equal documents serialize to identical bytes. `to_csv` accepts only the
> text-table type `List[Dict[string,string]]`; it applies the CSV quoting and
> row-order rules without inferring, preserving, or serializing a numeric cell
> type. Numeric source values enter CSV only through explicit `to_string`.

#### Exact public scalar and container boundaries

> **[05-OP-31]** `scalar_carrier(value) -> result` governs exactly the ten
> final public C callables in this table. The table is their canonical public
> identity; a differently named, typed-by-name, or raw-dtype successor is a
> different callable and has no authority from this atom.
>
> | callable | exact C signature |
> |---|---|
> | dtype storage size | `int64_t chelis_dtype_size(chelis_dtype dtype)` |
> | scalar validation/construction | `chelis_scalar chelis_scalar_from_bits(chelis_dtype dtype, uint64_t bits)` |
> | value boxing | `chelis_value chelis_value_from_scalar(chelis_scalar value)` |
> | value extraction | `chelis_scalar chelis_value_as_scalar(chelis_value value)` |
> | rank-zero tensor construction | `chelis_tensor *chelis_scalar_tensor(chelis_scalar value)` |
> | rank-zero tensor extraction | `chelis_scalar chelis_tensor_to_scalar(const chelis_tensor *tensor)` |
> | tensor fill | `void chelis_fill_scalar(chelis_tensor *tensor, chelis_scalar value)` |
> | scalar rendering | `chelis_string chelis_string_from_scalar(chelis_scalar value)` |
> | scalar parsing | `chelis_option_scalar chelis_parse_scalar(chelis_string text, chelis_dtype dtype)` |
> | exact dictionary scalar lookup | `chelis_option_scalar chelis_dict_get_scalar(const chelis_dict *dict, chelis_value key, chelis_dtype dtype)` |
>
> The final public declarations are exact:
>
> `typedef uint8_t chelis_dtype;`
>
> `enum { CHELIS_DTYPE_F32 = 0, CHELIS_DTYPE_F64 = 1, CHELIS_DTYPE_I32 = 2, CHELIS_DTYPE_BOOL = 3, CHELIS_DTYPE_I64 = 4, CHELIS_DTYPE_BF16 = 5, CHELIS_DTYPE_F16 = 6, CHELIS_DTYPE_I8 = 7, CHELIS_DTYPE_I16 = 8 };`
>
> `typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;`
>
> `typedef struct { uint8_t is_some; uint8_t reserved[7]; chelis_scalar value; } chelis_option_scalar;`
>
> `typedef uint8_t chelis_value_tag;`
>
> `enum { CHELIS_VALUE_UNIT = 0, CHELIS_VALUE_SCALAR = 1, CHELIS_VALUE_STRING = 2, CHELIS_VALUE_TENSOR = 3, CHELIS_VALUE_LIST = 4, CHELIS_VALUE_TUPLE = 5, CHELIS_VALUE_DICT = 6, CHELIS_VALUE_ADT = 7 };`
>
> `typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;`
>
> `typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;`
>
> `typedef struct { uint8_t is_some; uint8_t reserved[7]; chelis_value value; } chelis_option_value;`
>
> `typedef struct { void *data; const int64_t *shape; const int64_t *strides; int64_t size; int64_t byte_capacity; int32_t rank; chelis_dtype dtype; uint8_t owns_data; uint8_t reserved[2]; } chelis_tensor;`
>
> `typedef struct { chelis_value key; chelis_value value; } chelis_dict_entry;`
>
> `chelis_dtype` has the closed active IDs `F32=0`, `F64=1`, `I32=2`,
> `Bool=3`, `I64=4`, `Bf16=5`, `F16=6`, `I8=7`, and `I16=8`. In that
> order, the low 32/64/32/8/64/16/16/8/16 bits of `chelis_scalar.bits` are
> the exact stored image and all unused high bits are zero. A bool payload is
> exactly `0` or `1`. Float construction and transport preserve NaN payload
> and signed-zero bits. Every consumer validates both the foreign dtype value
> and this canonical bit shape before sizing, allocation, storage access, or
> observation. An unknown dtype, nonzero unused bit, malformed bool, or
> dtype mismatch traps `Domain` at that boundary; no operation repairs or
> reinterprets it.
>
> Every `reserved` byte is zero. Each option discriminant is exactly zero or
> one; a `None` carrier has an all-zero value field, and `Some` validates its
> value before crossing the boundary. A `chelis_value` tag is exactly one of
> the eight constants above. Unit has a null, otherwise-zero payload; Scalar
> embeds the complete canonical `chelis_scalar`; every heap tag carries a
> non-null owned handle in `payload.handle` and zero bytes in the remainder of
> the union. The natural C ABI of these fixed-width field declarations is the
> ABI; no feature macro, build mode, or typedef substitution may change their
> order, widths, or signedness.
>
> Every public `chelis_tensor` is contiguous row-major storage. It has rank in
> `0..=INT32_MAX`; rank zero has null `shape` and `strides` pointers, while a
> positive rank has non-null pointers to exactly `rank` int64 entries that the
> runtime owns for the carrier lifetime. Every extent is nonnegative and the
> strides are the exact checked products of the following extents in element
> units. `size` is the checked product of the extents, with the rank-zero
> empty product equal to one. There is no rank-eight limit.
> `byte_capacity` is nonnegative and at least the checked product
> `size * chelis_dtype_size(dtype)`. A zero-size tensor has null `data` and
> zero `byte_capacity`; a nonempty tensor has a non-null pointer aligned for
> its validated dtype. `owns_data` is exactly zero or one and both reserved
> bytes are zero. When ownership is one, the runtime owns the complete
> `byte_capacity`-byte allocation beginning at `data` and releases it exactly
> once; when ownership is zero, it never releases that storage. An internal
> noncontiguous view is materialized before it crosses this public carrier.
> `chelis_alloc_view` copies the supplied shape into runtime-owned metadata and
> derives the canonical strides. Its caller must provide a readable and writable
> allocation of the declared capacity that remains live for the view's
> lifetime; the runtime validates the declared metadata and bounds but cannot
> prove a foreign allocation's lifetime or physical size.
> Each `chelis_dict_entry` contains two independently canonical
> `chelis_value` carriers in key-then-value order.
>
> `chelis_dtype_size` returns the exact byte width of the validated stored
> representation. Tensor extraction requires a rank-zero tensor with exactly
> one element. Fill requires the scalar dtype to equal the tensor dtype and
> writes the exact scalar bits to every element. Neither operation converts
> through `double` or another dtype. The family does not convert through
> `double` at any other edge. Rendering follows [05-OBS-1..2] at the
> scalar's own dtype; a stored NaN renders as `NaN` and therefore preserves
> its class but not its payload through text. Parsing ignores surrounding
> ASCII space and tab. Signed-decimal integer text is exactly an optional
> `+` or `-` followed by one or more ASCII digits; leading zeros are admitted,
> while a bare sign, internal whitespace, or non-ASCII digit is malformed.
> The mathematical integer must fit the requested signed-integer dtype. A
> finite float token is an optional `+` or `-`, followed by either one or more
> ASCII digits with an optional `.` and zero or more following digits, or `.`
> followed by one or more ASCII digits, and then an optional exponent
> `[eE][+-]?[0-9]+`. Leading and trailing fractional zeros are admitted; a
> bare sign, bare point, internal whitespace, non-ASCII digit, hex form, or
> suffix is malformed. Its exact decimal value rounds once at the requested
> float width under [04-NUM-2]; finite overflow returns `None` and underflow
> rounds normally, including to signed zero. Thus parsing accepts a strict
> decimal superset of §8.1's canonical output spellings. Bool text is exactly
> `true` or `false`. The only accepted NaN spelling is exactly
> `NaN`; its canonical quiet-NaN images are f16 `0x7e00`, bf16 `0x7fc0`,
> f32 `0x7fc00000`, and f64 `0x7ff8000000000000`. The exact spellings `inf`
> and `-inf` produce respectively f16 `0x7c00`/`0xfc00`, bf16
> `0x7f80`/`0xff80`, f32 `0x7f800000`/`0xff800000`, and f64
> `0x7ff0000000000000`/`0xfff0000000000000`. Every other non-finite
> spelling is malformed. Malformed or finite out-of-range text returns `None`.
> Dictionary lookup admits only [05-OP-32]'s key domain and returns `None`
> only when the key is absent; a present non-scalar value or a scalar whose
> dtype differs from the requested dtype traps `Domain`.
>
> This operation family has no alias, wrapper, or deprecated spelling for
> dtype-named value constructors/extractors, scalar-tensor constructors, fills,
> parsers, options, string formatters, `chelis_format_shortest`, or public
> reduced-float buffer-conversion helpers. This carrier family has no
> accumulator and is outside AD.
>
> **[05-OP-32]** `shape_index(container, parameters...) -> result` governs
> exactly the container, extent, index, byte-read, and recursive-observation
> callable identities below. The signature is part of each identity.
>
> | callable | exact C signature |
> |---|---|
> | string length | `int64_t chelis_string_len(chelis_string value)` |
> | string slice | `chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)` |
> | list length | `int64_t chelis_list_len(const chelis_list *list)` |
> | list construction | `chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len)` |
> | list index | `chelis_value chelis_list_index(const chelis_list *list, int64_t index)` |
> | list take | `chelis_list *chelis_list_take(const chelis_list *list, int64_t count)` |
> | list drop | `chelis_list *chelis_list_drop(const chelis_list *list, int64_t count)` |
> | list chunk | `chelis_list *chelis_list_chunk(const chelis_list *list, int64_t size)` |
> | integer range | `chelis_list *chelis_range_i64(int64_t start, int64_t end)` |
> | list enumerate | `chelis_list *chelis_list_enumerate(const chelis_list *list)` |
> | tuple length | `int64_t chelis_tuple_len(const chelis_tuple *tuple)` |
> | tuple construction | `chelis_tuple *chelis_tuple_from_values(const chelis_value *items, int64_t len)` |
> | tuple index | `chelis_value chelis_tuple_get(const chelis_tuple *tuple, int64_t index)` |
> | ADT construction | `chelis_adt *chelis_adt_construct(chelis_string ctor, const chelis_value *fields, int64_t len)` |
> | ADT field count | `int64_t chelis_adt_field_count(const chelis_adt *adt)` |
> | ADT field index | `chelis_value chelis_adt_get_field(const chelis_adt *adt, int64_t index)` |
> | dictionary length | `int64_t chelis_dict_len(const chelis_dict *dict)` |
> | dictionary construction | `chelis_dict *chelis_dict_from_pairs(const chelis_list *pairs)` |
> | dictionary membership | `bool chelis_dict_contains(const chelis_dict *dict, chelis_value key)` |
> | dictionary lookup | `chelis_option_value chelis_dict_get(const chelis_dict *dict, chelis_value key)` |
> | dictionary removal | `chelis_dict *chelis_dict_remove(const chelis_dict *dict, chelis_value key)` |
> | dictionary insertion | `chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value)` |
> | dictionary merge | `chelis_dict *chelis_dict_merge(const chelis_dict *left, const chelis_dict *right)` |
> | byte-file read | `chelis_list *chelis_read_bytes(chelis_string path)` |
> | mapped byte read | `chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len)` |
> | mapped byte length | `int64_t chelis_mmap_len(const chelis_mapped_file *mapped)` |
> | list print | `void chelis_print_list(const chelis_list *list)` |
> | tuple print | `void chelis_print_tuple(const chelis_tuple *tuple)` |
> | dictionary print | `void chelis_print_dict(const chelis_dict *dict)` |
> | ADT print | `void chelis_print_adt(const chelis_adt *adt)` |
>
> All lengths, indices, offsets, sizes, and returned counts are exact `int64`.
> Negative lengths, indices, offsets, and counts trap `Domain`; an indexed
> read outside its container and a positive-length foreign buffer with a null
> pointer also trap `Domain`. Every result-length and allocation arithmetic
> traps `Overflow` before allocation or access. `list_take` truncates at the
> end, `list_drop` past the end returns empty, and `list_chunk` requires a
> positive size and permits only its final chunk to be shorter.
> `chelis_range_i64(start, end)` is the half-open increasing sequence and is
> empty when `end <= start`; enumeration pairs each value with its exact
> zero-based int64 index. String indices count Unicode scalar values, not
> encoded bytes. A slice whose nonnegative start is at or beyond the scalar
> length is empty; otherwise it takes at most `len` scalars and truncates at
> the end.
>
> Dictionary keys are exactly `string`, `bool`, or a scalar of any active
> signed-integer dtype. Equality includes the key kind and integer dtype and
> then compares exact stored values, so no integer width is coerced. Float keys
> are rejected because NaN non-reflexivity and signed-zero numeric equality do
> not define the stable equivalence relation a dictionary requires.
> `dict_from_pairs` and `dict_merge` process entries from left to right; a
> later duplicate replaces the value at the existing insertion position,
> and a new key appends. `dict_insert` uses the same rule, `dict_remove` of an
> absent key is unchanged, and `dict_get` returns `None` only for absence.
> Recursive dictionary observation is canonical rather than insertion-ordered:
> bool keys order `false` before `true`, integer keys order by exact
> mathematical value within their one static key dtype, and string keys order
> lexicographically by Unicode scalar value. A dictionary has one static key
> type, so no cross-kind or cross-width ordering is defined.
> `read_bytes` and `mmap_read` return int64 elements in `0..=255`; mapped
> reads also require `offset + length` to lie within the mapped file.
> Each of `chelis_print_list`, `chelis_print_tuple`, `chelis_print_dict`, and
> `chelis_print_adt` writes exactly [05-OP-25]'s complete recursive rendering
> of its argument followed by one byte `\n` to standard output. It adds no
> label, prefix, extra space, truncation beyond the nested tensor rule, or
> additional newline. A short or failed write traps `IO`; bytes the operating
> system accepted before that failure remain an ordinary prior `Io` effect.
> Successful return means every required byte was written. Recursive printing
> applies [05-OBS-1..5] at each stored scalar's own dtype and never widens a
> value for observation. These boundary operations are outside AD and have no
> accumulator.
>
> **[05-OP-33]** `runtime_tensor(value, parameters...) -> result` governs
> exactly the twenty-three final public C callable identities below. These
> signatures are canonical: axes and rank are `int32_t`; extents, sizes,
> offsets, counts, and element counts are `int64_t`; dtype arguments are
> `chelis_dtype`; and an untyped, string-mode, or dtype-named successor has no
> authority from this atom.
>
> | callable | exact C signature |
> |---|---|
> | owned allocation | `chelis_tensor *chelis_alloc(int32_t rank, const int64_t *shape, chelis_dtype dtype)` |
> | borrowed view | `chelis_tensor *chelis_alloc_view(int32_t rank, const int64_t *shape, chelis_dtype dtype, void *data, int64_t byte_capacity)` |
> | rank | `int32_t chelis_tensor_rank(const chelis_tensor *tensor)` |
> | extent | `int64_t chelis_tensor_shape(const chelis_tensor *tensor, int32_t axis)` |
> | element count | `int64_t chelis_tensor_numel(const chelis_tensor *tensor)` |
> | contiguous copy | `chelis_tensor *chelis_contiguous(const chelis_tensor *tensor)` |
> | typed list ingress | `chelis_tensor *chelis_tensor_from_values(const chelis_list *list, chelis_dtype dtype)` |
> | row-major element egress | `chelis_list *chelis_tensor_elements(const chelis_tensor *tensor)` |
> | inferred-width padding | `chelis_tensor *chelis_pad_sequences(const chelis_list *sequences, chelis_scalar pad_value)` |
> | fixed-width padding | `chelis_tensor *chelis_pad_sequences_to(const chelis_list *sequences, int64_t width, chelis_scalar pad_value)` |
> | concatenate | `chelis_tensor *chelis_tensor_concat(const chelis_list *parts, int32_t axis)` |
> | split | `chelis_list *chelis_tensor_split(const chelis_tensor *tensor, int32_t axis, const chelis_list *sizes)` |
> | gather | `chelis_tensor *chelis_tensor_gather(const chelis_tensor *tensor, const chelis_tensor *indices, int32_t axis)` |
> | replace scatter | `chelis_tensor *chelis_tensor_scatter_replace(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
> | additive scatter | `chelis_tensor *chelis_tensor_scatter_add(const chelis_tensor *base, const chelis_tensor *indices, const chelis_tensor *updates, int32_t axis)` |
> | comparison | `chelis_tensor *chelis_tensor_cmplt(const chelis_tensor *left, const chelis_tensor *right)` |
> | selection | `chelis_tensor *chelis_tensor_where(const chelis_tensor *condition, const chelis_tensor *then_tensor, const chelis_tensor *else_tensor)` |
> | inclusive prefix sum | `chelis_tensor *chelis_tensor_cumsum(const chelis_tensor *tensor, int32_t axis)` |
> | stable sort | `chelis_tuple *chelis_tensor_sort(const chelis_tensor *tensor, int32_t axis)` |
> | diagonal | `chelis_tensor *chelis_tensor_diagonal(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
> | trace | `chelis_tensor *chelis_tensor_trace(const chelis_tensor *tensor, int32_t axis1, int32_t axis2)` |
> | clamp | `chelis_tensor *chelis_tensor_clamp(const chelis_tensor *tensor, const chelis_tensor *lower, const chelis_tensor *upper)` |
> | contraction | `chelis_tensor *chelis_tensor_einsum(chelis_string equation, const chelis_tensor *left, const chelis_tensor *right, chelis_dtype accumulator)` |
>
> This atom's selection rule also governs exactly the language builtin
> `where(condition, then, else)` with signature
> `(&tensor[D,bool], &tensor[D,p], &tensor[D,p]) -> tensor[D,p]` and its
> exact public C counterpart `chelis_tensor_where` tabled above. All four
> tensors have identical dimensions, both branches and the result have the
> same active element dtype `p`, and selection copies the chosen stored bits
> without numeric conversion. On float branches the adjoint routes each
> cotangent to the selected branch and exact zero to the other; the condition
> has no cotangent. Signed-integer and bool branches are forward-only. The
> operation has no accumulator.
>
> Every entry validates every observable input-tensor invariant from
> [05-OP-31], including dtype, shape, canonical strides, size, capacity,
> alignment, and ownership metadata, before reading data. The rank is nonnegative
> and representable as int32; every extent
> is nonnegative; a positive-rank shape pointer is non-null; and a nonempty
> borrowed view has a non-null, representation-aligned data pointer. Shape
> products, byte counts, offsets, output extents, and allocation sizes use
> checked arithmetic. A malformed carrier or invalid axis traps `Domain`; an
> unrepresentable count, extent, offset, or allocation size traps `Overflow`
> before allocation or element access. Each language operation follows its
> own axis atom: [05-AXIS-1] governs the static reduction/expand family, while
> [05-OP-7]/[05-SHAPE-1] admits a computed int32 axis for `shape`. C-family
> axis parameters are runtime int32 values. Every signed axis accepted by this C family first
> applies §2.3's one-step negative normalization; an axis still out of range
> then traps `Domain`.
>
> Allocation returns owned, contiguous, row-major, zero-filled storage at the
> requested representation. A zero extent means zero elements, never one
> synthetic element. A view is non-owning contiguous storage whose declared
> capacity covers its checked byte size, and releasing
> it never releases the caller's data. `contiguous` preserves every element's
> exact stored bits in row-major order. Tensor ingress accepts a rectangular
> nested list whose scalar leaves all have exactly the requested dtype; it
> neither infers nor converts that dtype. `chelis_tensor_elements` boxes every
> element as its exact scalar in row-major order for every rank. Rank zero
> therefore returns a one-element list, and any zero extent returns an empty
> list. The operation deliberately exposes elements rather than shape; callers
> that need dimensions use `chelis_tensor_rank` and `chelis_tensor_shape`.
> There is no rank-specialized or recursively nested list-egress alias.
> Padding follows [05-OP-9..10] exactly.
>
> `concat` requires a nonempty list of tensors with one common rank and dtype and equal
> non-concatenated extents and copies parts in list order. `split` requires
> nonnegative int64 sizes whose checked sum equals the selected extent and
> returns consecutive pieces in size-list order. Both preserve element bits;
> on floats their adjoints are respectively split by source extents and
> concat in source order. Integer and bool forms are forward-only.
>
> `gather` admits an index tensor of any active signed-integer dtype and every active tensor element dtype.
> Its result shape is `data[..axis] ++ indices.shape ++ data[axis+1..]`;
> indices are zero-based and in range. It preserves selected bits. Its float
> adjoint scatter-adds cotangents into an all-zero input-shaped tensor. Each
> destination's canonical balanced accumulation tree begins with an exact
> positive-zero base leaf, not an omitted initializer; contributions follow
> in increasing output row-major flat-index order.
>
> `chelis_tensor_scatter_replace` and `chelis_tensor_scatter_add` have the
> inverse gather shape contract: `updates.shape` is
> `base[..axis] ++ indices.shape ++ base[axis+1..]`, while base, updates, and
> result have one dtype. Replace admits every active dtype, including bool,
> and is [05-SPARSE-1..2]'s row-major
> last-write-wins operation and structurally rejects `grad`. Add starts each
> destination's leaf sequence with the base value, admits exactly active
> signed-integer and float dtypes, and follows the base by targeting
> updates in increasing row-major flat-index order, and combines those leaves
> with the canonical balanced tree; integer overflow is checked at each node.
> Its float adjoint passes the cotangent to base and gathers it to each update.
> Scatter indices have any active signed-integer dtype, are interpreted at
> their exact stored mathematical values, are zero-based, and must lie in the
> selected base-axis extent. Any negative or out-of-range index traps `Domain`
> before any write. Indices have no cotangent. There is no public string
> scatter mode and no int32/int64-only dispatch exception.
>
> `cmplt` requires identical shapes and identical active signed-integer or
> float dtypes, compares stored values without conversion, and returns native
> bool storage; comparison with NaN is false and no cotangent exists. `where`
> follows the exact language-and-C selection contract above.
>
> `cumsum` admits signed-integer and float tensors and returns
> `sum_result(p, default(p))` at the input shape. In increasing axis order an
> exact-zero accumulator adds each input at §5.7.1's default sum-accumulator
> dtype. Every prefix is finalized to the result dtype for output without
> narrowing the continuing accumulator. Integer overflow is checked at each
> addition. A zero-length axis returns an empty tensor. The float adjoint is
> the inclusive decreasing-axis scan at the same accumulator dtype, emitting
> each suffix at the operand dtype. Integers are forward-only and bool is a
> type error.
>
> `sort` admits signed-integer and float tensors and returns `(values,
> indices)`, with values at the input shape/dtype and exact int64 indices at
> the same shape. Each axis slice is stable ascending: NaNs follow all
> non-NaNs; non-NaNs use numeric order; signed zeros compare equal; and equal
> values and NaNs retain increasing source-axis order. Stored representations
> and NaN payloads therefore remain stable. On floats the values adjoint uses
> the inverse exact permutation; indices have no cotangent. Integers are
> forward-only and bool is a type error.
>
> `diagonal` admits every active dtype including bool, requires distinct axes,
> keeps source axis order with the second
> axis removed, and replaces the retained first axis extent with the smaller
> selected extent. It reads equal coordinates on both axes and preserves bits.
> Its float adjoint scatters to those diagonal cells. `trace` is that diagonal
> followed by [05-OP-30]'s canonical balanced tree and default accumulator;
> it inherits sum's result dtype, empty identity, and overflow rule and removes
> both axes. Its float adjoint expands the cotangent on the selected diagonal.
>
> `clamp` admits signed-integer and float tensors. Each bound has the input
> dtype and is rank zero or input-shaped. It rejects a NaN bound or `lower >
> upper` at the first row-major position. A NaN input is preserved. Otherwise
> the result is lower for `x < lower`, upper for `x > upper`, and the exact
> stored input at inclusive bounds. The float adjoint follows that executed
> branch; cotangents to a rank-zero bound combine in input row-major order by
> the balanced tree. Integers are forward-only and bool is a type error.
>
> `einsum` accepts exactly the grammar `[a-z]*,[a-z]*->[a-z]*`, with no
> whitespace, ellipsis, implicit output, or third operand. Label counts equal
> ranks. A repeated input label denotes a diagonal and requires equal extents;
> shared labels have one extent; each output label occurs in an input and
> exactly once in the output. Output labels set output axis order. Non-output
> labels reduce in first-occurrence order scanning left then right, with the
> last such label varying fastest.
>
> Both operands have one active signed-integer or float dtype `p`.
> `accumulator` is a resolved accumulator dtype of the same numeric kind as
> `p` and at least §5.7.1's default.
> Operand elements convert exactly to it, each pair multiplies and finalizes
> there, and products in the specified assignment order combine through the
> adjacent-pair balanced tree. An empty reduction is exact zero. The result is
> `sum_result(p, accumulator)`; integer overflow is checked at every multiply
> and add. Output positions and tree nodes evaluate left to right, fixing the
> first trap. The float adjoint is the reverse derivative of this exact
> multiply-and-balanced-add graph; contributions to one operand cell follow
> forward output then reduction order and combine through the same tree.
> Integers are forward-only and bool is a type error.
>
> No operation in this family converts a stored element through `double`,
> treats bool as numeric storage, silently changes an axis width, or supplies a
> compatibility alias. Except for the stated cumsum, trace, and einsum rules,
> the family has no user-selectable accumulator.
>
> **[05-OP-34]** `numeric_adt(fields...) -> value` governs exactly these five
> exported stdlib ADT identities and no structurally similar successor:
>
> | identity | exact variants and fields |
> |---|---|
> | `io/json::Json` | `JsonNull | JsonBool(bool) | JsonInt(int64) | JsonFloat(f64) | JsonString(string) | JsonArray(List[Json]) | JsonObject(Dict[string,Json])` |
> | `decimal::Decimal` | `Decimal { coefficient: int64, scale: int64 }` |
> | `time::Date` | `Date { year: int64, month: int64, day: int64 }` |
> | `time::Duration` | `Duration { days: int64, hours: int64, minutes: int64, seconds: int64 }` |
> | `tokenizer::Tokenizer` | `BpeTokenizer(Dict[string,int64], Dict[string,int64], Dict[int64,string], int64)` |
>
> Every field crosses at its declared dtype and stored bits, without
> arithmetic, conversion, or float funnel. An ordinary public ADT constructor
> accepts every representable declared field tuple; validation and
> normalization belong to named stdlib functions, not a module-name
> special-case in generic ADT construction. A public signature is numeric when
> any reachable field of an admitted ADT, tuple, `List`, `Dict`, `Option`, or
> tensor is an active numeric primitive. The reachability computation expands
> nominal ADT definitions recursively to a fixed point and visits a recursive
> type cycle once; scanning only primitives spelled directly in the signature
> is nonconforming. A precision type variable in a tensor element or linked
> scalar position is numeric even though its instantiated primitive is not
> written in the signature. Every exported definition whose parameter or result reaches
> a numeric field binds to [05-OP-35] or another exact numeric operation atom.
> [05-OP-2] governs `io/json::Json`'s exact
> integer/float source distinction. Decimal, date, duration, vocabulary,
> merge-rank, inverse-vocabulary, and unknown-token invariants are checked by
> the named [05-OP-35] operations before use. There is no second prelude JSON
> identity or constructor registry. Under spec/06 §2.1 and §2.10.1, an
> ordinary constructor and the executed matching arm preserve the recursive
> cotangent shape: differentiable float fields receive their corresponding
> field cotangents, while non-differentiable fields carry `unit`. Thus, for
> example, `JsonFloat(x)` followed by an executed `JsonFloat(y)` match routes
> the cotangent of `y` to `x`; integer-only ADTs naturally have only `unit`
> field cotangents. The constructors have no accumulator.
>
> **[05-OP-35]** `stdlib_numeric_def(arguments...) -> result` governs exactly
> the eighty-three final exported stdlib numeric definitions in this table. A
> signature and effect set are part of the identity. Only the exact table
> identities exist: no effectless, wildcard-result, or otherwise weakened alias
> is part of the language.
>
> | # | identity | exact final signature |
> |---:|---|---|
> | 1 | `contracts::normal_cdf` | `(p_float)->p_float` |
> | 2 | `contracts::normal_cdf_contract_samples` | `()->int64` |
> | 3 | `contracts::normal_cdf_contract_seed` | `()->int64` |
> | 4 | `contracts::standard_contract_tolerance` | `()->f32` |
> | 5 | `decimal::decimal` | `(string)->Decimal` |
> | 6 | `decimal::decimal_add` | `(Decimal,Decimal)->Decimal` |
> | 7 | `decimal::decimal_div` | `(Decimal,Decimal,int64,RoundingMode)->Decimal` |
> | 8 | `decimal::decimal_eq` | `(Decimal,Decimal)->bool` |
> | 9 | `decimal::decimal_from_int` | `(int64)->Decimal` |
> | 10 | `decimal::decimal_gt` | `(Decimal,Decimal)->bool` |
> | 11 | `decimal::decimal_gte` | `(Decimal,Decimal)->bool` |
> | 12 | `decimal::decimal_lt` | `(Decimal,Decimal)->bool` |
> | 13 | `decimal::decimal_lte` | `(Decimal,Decimal)->bool` |
> | 14 | `decimal::decimal_mul` | `(Decimal,Decimal)->Decimal` |
> | 15 | `decimal::decimal_sub` | `(Decimal,Decimal)->Decimal` |
> | 16 | `decimal::decimal_to_float` | `(Decimal)->f64` |
> | 17 | `decimal::decimal_to_string` | `(Decimal)->string` |
> | 18 | `decimal::try_decimal` | `(string)->Option[Decimal]` |
> | 19 | `index::drop_list` | `(List[T],int64)->List[T]` |
> | 20 | `index::list_index` | `(List[T],int64)->T` |
> | 21 | `index::take_list` | `(List[T],int64)->List[T]` |
> | 22 | `init/kaiming::kaiming_normal` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
> | 23 | `init/kaiming::kaiming_uniform` | `(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}` |
> | 24 | `init/random::normal_like` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
> | 25 | `init/xavierext::trunc_normal` | `(&tensor[..r,p_float],p_float,p_float,p_float,p_float)->tensor[..r,p_float]!{Random}` |
> | 26 | `init/xavierext::xavier_normal` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
> | 27 | `init/xavierext::xavier_uniform` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}` |
> | 28 | `io/json::json_array` | `(Option[Json])->Option[List[Json]]` |
> | 29 | `io/json::json_bool` | `(Option[Json])->Option[bool]` |
> | 30 | `io/json::json_float` | `(Option[Json])->Option[f64]` |
> | 31 | `io/json::json_get` | `(Json,string)->Option[Json]` |
> | 32 | `io/json::json_int` | `(Option[Json])->Option[int64]` |
> | 33 | `io/json::json_is_null` | `(Option[Json])->bool` |
> | 34 | `io/json::json_object` | `(Option[Json])->Option[Dict[string,Json]]` |
> | 35 | `io/json::json_string` | `(Option[Json])->Option[string]` |
> | 36 | `io/json::load_json` | `(string)->Json!{IO}` |
> | 37 | `io/json::parse_json` | `(string)->Json` |
> | 38 | `io/json::to_json` | `(Json)->string` |
> | 39 | `io/json::try_load_json` | `(string)->Option[Json]!{IO}` |
> | 40 | `io/json::try_parse_json` | `(string)->Option[Json]` |
> | 41 | `io/json::try_to_json` | `(Json)->Option[string]` |
> | 42 | `io/json::try_write_json` | `(string,Json)->Option[unit]!{IO}` |
> | 43 | `io/json::write_json` | `(string,Json)->unit!{IO}` |
> | 44 | `io::mmap_size` | `(string)->int64!{IO}` |
> | 45 | `io::read_head_bytes` | `(string,int64)->List[int64]!{IO}` |
> | 46 | `process::run` | `(string,List[string])->(int64,string,string)!{IO}` |
> | 47 | `process::run_chelis` | `(List[string])->(int64,string,string)!{IO}` |
> | 48 | `scalar::abs` | `(p_numeric)->p_numeric` |
> | 49 | `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |
> | 50 | `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |
> | 51 | `sort::sort` | `(&tensor[..r,p_numeric],int32)->(tensor[..r,p_numeric],tensor[..r,int64])` |
> | 52 | `tensor/construct::arange` | `(p_int,p_int)->tensor[n,p_int]` |
> | 53 | `tensor/construct::linspace` | `(p_float,p_float,int64)->tensor[n,p_float]` |
> | 54 | `tensor/construct::squeeze` | `(&tensor[..pre,1,..post,p],int32)->tensor[..pre,..post,p]` |
> | 55 | `tensor/construct::stack` | `(List[tensor[..pre,..post,p]],int32)->tensor[..pre,rows,..post,p]` |
> | 56 | `tensor/construct::unsqueeze` | `(&tensor[..pre,..post,p],int32)->tensor[..pre,1,..post,p]` |
> | 57 | `tensor/mask::where_indices` | `(&tensor[..r,bool])->tensor[hits,int64]` |
> | 58 | `test::assert_close` | `(p_float,p_float,p_float,string)->unit!{Test}` |
> | 59 | `test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
> | 60 | `test::assert_eq` | `(Q,Q,string)->unit!{Test}` |
> | 61 | `test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
> | 62 | `test::assert_shape` | `(&tensor[..r,p],List[int64],string)->unit!{Test}` |
> | 63 | `time::add_days` | `(Date,int64)->Date` |
> | 64 | `time::date` | `(int64,int64,int64)->Date` |
> | 65 | `time::date_gt` | `(Date,Date)->bool` |
> | 66 | `time::date_gte` | `(Date,Date)->bool` |
> | 67 | `time::date_lt` | `(Date,Date)->bool` |
> | 68 | `time::date_lte` | `(Date,Date)->bool` |
> | 69 | `time::date_to_string` | `(Date)->string` |
> | 70 | `time::day_of_week` | `(Date)->DayOfWeek` |
> | 71 | `time::day_of_week_name` | `(Date)->string` |
> | 72 | `time::day_of_year` | `(Date)->int64` |
> | 73 | `time::days_between` | `(Date,Date)->int64` |
> | 74 | `time::duration` | `(int64,int64,int64,int64)->Duration` |
> | 75 | `time::is_leap_year` | `(int64)->bool` |
> | 76 | `time::parse_date` | `(string)->Option[Date]` |
> | 77 | `time::sub_days` | `(Date,int64)->Date` |
> | 78 | `time::try_date` | `(int64,int64,int64)->Option[Date]` |
> | 79 | `tokenizer::batch_encode` | `(Tokenizer,List[string],int64,int64)->tensor[batch,seq,int64]` |
> | 80 | `tokenizer::decode` | `(Tokenizer,List[int64])->string` |
> | 81 | `tokenizer::encode` | `(Tokenizer,string)->List[int64]` |
> | 82 | `tokenizer::load_tokenizer` | `(string)->Tokenizer!{IO}` |
> | 83 | `tokenizer::try_load_tokenizer` | `(string)->Option[Tokenizer]!{IO}` |
>
> `init/xavier::sample` is not a language operation and must not be exported.
> It has no semantics, registration, alias, or stub disposition; a final
> stdlib containing that identity violates this exact manifest.
>
> Every primitive-width intermediate in a graph whose contract names a dtype
> executes and finalizes at [04-NUM-8]'s declared width; integer primitive
> arithmetic is checked and a composed trap propagates at its first specified
> operation. Decimal rational and calendar ordinal computations explicitly
> named as mathematical below use an exact internal domain; only their named
> int64 input and final-representation boundaries can trap `Overflow`, and no
> host integer width becomes observable. JSON access follows [05-OP-2..5]: an
> integer-form token outside int64 traps `Overflow` and never becomes
> `JsonFloat`; `json_int`
> refuses `JsonFloat`, while `json_float` performs [05-OP-3]'s named
> int64-to-f64 widening and returns a stored f64 unchanged. Index wrappers
> follow [05-OP-32], sort wrappers follow [05-OP-33], and no tensor
> constructor infers or casts an element dtype.
> For a differentiable element type, `list_index(xs,i)` returns an input
> cotangent list of the primal length with the output cotangent at exact index
> `i` and zero cotangents elsewhere. `take_list(xs,n)` returns the output
> cotangents followed by zeros for every untaken source position;
> `drop_list(xs,n)` returns zeros for every dropped position followed by the
> output cotangents. The exact forward truncation rules determine the split,
> and every list cotangent preserves the primal runtime length and order under
> spec/06 §2.1. Non-differentiable element types are forward-only.
>
> The contracts constants are exactly `8192i64`, `3235848230i64`, and the f32
> image of `1e-10`. `standard_contract_tolerance` is deliberately the fixed f32
> tolerance policy of the named standard-contract property corpus; it is not
> an arithmetic operand, default dtype, or restriction on `normal_cdf`, whose
> semantic result remains at its input dtype. A caller defining another
> property policy states its own tolerance at that property's value dtype.
> `normal_cdf(+inf)` is exact `1p`, `normal_cdf(-inf)` is exact `0p`, and a NaN
> input returns [04-NUM-2]'s canonical NaN at `p_float`. Every finite input uses
> this exact same-`p_float` composition:
> `0.5 * (1 - erf_approx(-x * 0.7071067811865475))`, where for
> `a=abs(x)`, `erf_approx(x)` is `x * 1.1283791670955126` when `a < 1e-5` and
> otherwise uses `t=1/(1+0.3275911*a)`,
> `p=t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))`,
> and `sign(x)*(1-p*exp(-a*a))`. Each decimal literal rounds directly to
> `p_float`, and each primitive finalizes to `p_float` before its consumer.
> The `exp` inside this compound graph is correctly rounded to `p_float`; it does
> not inherit standalone `exp`'s [05-OBS-3] lane tolerance. Consequently the
> complete callable has [05-OBS-3]'s default zero-ULP cross-lane bound.
> Its adjoint is the derivative of that exact finite graph, not a substituted
> library CDF. The infinities have zero cotangent and NaN propagates the
> canonical NaN cotangent; no non-finite input traps `Domain`.
>
> A `Decimal { coefficient, scale }` denotes the exact rational
> `coefficient / 10^scale` and is valid for stdlib arithmetic exactly when
> `scale >= 0`. A nonzero canonical value has no trailing base-10 zero in its
> coefficient while `scale > 0`; canonical zero is
> `Decimal { coefficient: 0i64, scale: 0i64 }`. Every Decimal-taking operation
> validates `scale` before arithmetic and normalizes its result. The accepted
> decimal grammar is ASCII space/tab/CR/LF around one optional `+` or `-` and
> either one or more digits with an optional `.` and zero or more following
> digits, or `.` followed by one or more digits. Exponents and internal
> whitespace are invalid. `try_decimal` returns `None` for invalid syntax;
> `decimal` traps `Domain`. Syntactically valid digits and their scale are
> interpreted in exact arithmetic and normalized before either representation
> check. Both forms trap `Overflow`, rather than returning `None`, exactly when
> the resulting canonical coefficient or scale has no int64 representation;
> removable trailing zeros do not cause `Overflow`.
>
> `decimal_from_int(v)` is the canonical
> `Decimal { coefficient: v, scale: 0i64 }`. Addition, subtraction,
> multiplication, equality, and ordering use exact mathematical rationals;
> no alignment power, product, or comparison converts through float.
> Arithmetic returns the unique canonical representable pair and traps
> `Overflow` when no such int64 coefficient/scale pair exists. The five
> comparison callables are the ordinary exact-rational `=`, `<`, `<=`, `>`,
> and `>=` relations named by their suffixes. `decimal_div` rejects a zero
> divisor or negative result scale. It rounds the exact rational to the
> requested base-10 scale: `RoundDown` truncates the magnitude toward zero,
> `RoundUp` increases a non-integral magnitude away from zero,
> `RoundHalfUp` rounds an exact half away from zero, and `RoundHalfEven`
> rounds a half to an even magnitude. The rounded result is normalized and
> traps `Overflow` when its canonical pair is unrepresentable.
>
> `decimal_to_float` correctly rounds the exact decimal rational once to f64;
> it is the named lossy boundary when that rational has no exact f64 image.
> `decimal_to_string` emits the unique canonical non-exponent form: optional
> `-`, at least one integer digit, and a fractional point followed by exactly
> `scale` digits only when the normalized scale is nonzero. It emits zero as
> `0` and never emits `+`, leading integer zeros, or trailing fractional
> zeros. Decimal operations are outside AD and have no accumulator.
>
> JSON parsing accepts exactly one complete RFC 8259 value encoded as valid
> Unicode scalar text, including the standard escape grammar and valid UTF-16
> surrogate pairs in `\\u` escapes. It rejects unescaped controls, invalid or
> lone surrogates, trailing tokens, and malformed numbers. A leading U+FEFF
> byte-order mark is not RFC 8259 whitespace and is rejected. Every finitely
> nested valid document is in the language; there is no fixed semantic nesting
> depth such as 512. A repeated object key replaces the earlier value. Numeric
> tokens follow [05-OP-2].
> `try_parse_json` returns `None` exactly for invalid JSON text;
> `parse_json` traps `Domain` for the same input. `try_load_json` returns
> `None` for a missing file, invalid UTF-8, or invalid JSON and otherwise
> returns the parsed value; permission, device, and other read failures trap
> `IO`. `load_json` traps `IO` for a missing or unreadable file and `Domain`
> for invalid UTF-8 or JSON.
>
> JSON accessors are exact variant projections. `json_get` returns the value
> at a string key only for `JsonObject`; each `json_string`, `json_int`,
> `json_float`, `json_bool`, `json_array`, and `json_object` returns `Some`
> only for its named variant, except [05-OP-3]'s explicit int64-to-f64 case.
> `json_is_null` is true exactly for `Some(JsonNull)`. Every other shape,
> including `None`, returns `None` or false without fabricating a payload.
> Serialization follows [05-OP-5], uses compact JSON punctuation, and escapes
> every required control. Object serialization orders members by increasing
> Unicode scalar-value key sequence, recursively, so equal documents have
> identical bytes. `try_to_json` returns `None` exactly when a reachable float
> is non-finite; `to_json` traps `Domain`. `write_json` and `try_write_json`
> validate the complete document before opening or truncating the destination.
> The try form returns `None` for non-finite content; both forms propagate a
> filesystem write failure as `IO` and otherwise write exactly the bytes of
> `to_json`.
>
> In this family `p` ranges over all active tensor element dtypes,
> `p_numeric` over all active numeric dtypes, `p_int` over all active signed
> integers, `p_float` over all four active floats, and `Q` over one static type
> in [05-OP-36]'s scalar or recursive equality domain (direct tensor arguments
> use `assert_eq_tensor`). Every repeated variable denotes one
> common static type. All random parameters
> are finite and are validated before consuming Random.
> Kaiming requires finite `fan_in > 0`. Xavier computes
> `add(fan_in, fan_out)` at `p_float` before consuming Random; that computed
> denominator must be finite and strictly positive. An overflowed infinite sum
> is a `Domain` failure, not a zero scale. `normal_like` requires `std >= 0`
> and invokes [05-OP-8] twice,
> first with the direct `p_float` images of decimal bounds `1e-7, 1.0` to
> obtain `u1` and then with the direct `p_float` images of `0.0, 1.0` to
> obtain `u2`; [05-OP-8] owns their exact values. The internal 53-bit unit in [05-OP-8] is half-open, but ordinary
> final rounding can make either stored result equal its stored high bound;
> a rounded result equals the stored upper endpoint for some source words.
> No stricter range is assumed by this graph. Its
> exact graph is `two_pi = round_p(2*pi)`,
> `cos_term = cos(mul(two_pi, u2))`,
> `radius = sqrt(mul(-2p, log(u1)))`, `z = mul(radius, cos_term)`, then
> `add(mean, mul(std, z))`. Every named assignment and primitive finalizes to
> `p_float` before its consumer under [04-NUM-8]; a sine phase-shift
> substitution is not conforming. `round_p(2*pi)` is the correctly rounded
> image of the mathematical constant at `p_float`. The `log` and `cos` inside
> this compound graph are correctly rounded at `p_float`'s arithmetic width
> before ordinary finalization to `p_float`; they do not inherit the
> standalone primitive tolerance. Together with [05-RNG-1], the complete
> random callable has zero-ULP cross-lane difference for a supported dtype.
> `trunc_normal` additionally requires `a <= b` and clips that normal result
> to inclusive `[a,b]`; it is not rejection sampling. Kaiming and Xavier use
> respectively `sqrt(div(2p, fan_in))` or
> `sqrt(div(2p, add(fan_in, fan_out)))` as normal scale. Their uniform
> bounds replace `2p` with `6p` under the same divisions; a uniform
> `u` maps by the exact `p_float` graph
> `mul(sub(mul(2p, u), 1p), bound)`. Here `Np` means the exact integer `N`
> represented at `p_float`. Violations trap `Domain`.
>
> With the handled Random stream fixed to the forward execution, every random
> stdlib callable has the pathwise adjoint of its exact graph above; source
> units and mask comparisons contribute zero cotangent. Template element values
> are unobserved and receive a same-shaped zero cotangent. `normal_like`
> combines per-element `g_i` contributions to `mean` and `g_i * z_i`
> contributions to `std` in increasing row-major order through separate
> canonical adjacent-pair balanced trees. `trunc_normal` uses the executed
> clipping branch: `raw < a` routes the whole cotangent to `a`, `raw > b`
> routes it to `b`, and the inclusive `a <= raw <= b` branch differentiates
> the exact normal graph; equality therefore stays on the raw branch.
> Kaiming and Xavier differentiate their exact authored scale/bound graphs into
> `fan_in` and `fan_out`. Uniform bounds differentiate through their exact
> [05-OP-8] affine graph. Every broadcast scalar contribution is enumerated in
> increasing row-major output order and combined by the canonical balanced
> tree. Each named primitive and adjoint primitive finalizes at `p_float`
> before its consumer.
>
> `scalar::abs` follows the unary abs rule at its active signed-integer or
> float dtype, including zero derivative at float zero and checked overflow at
> the signed minimum. Signed-integer scalar operations are forward-only.
> Scalar min/max return the first NaN with its exact stored payload and sign
> bits when either operand is NaN; otherwise
> they select the numerical extreme and preserve the first operand on every
> equality, including signed-zero equality. Their float adjoints route the
> whole cotangent to that selected operand. The sort wrappers inherit
> [05-OP-33] without another ordering rule.
>
> `arange(start,stop)` admits one active signed-integer dtype `p_int` for both
> endpoints and returns the increasing half-open same-dtype sequence. Its
> length and every step are checked in exact mathematical integers; an
> unrepresentable length or element traps `Overflow`. `linspace` requires finite endpoints and
> int64 `count >= 1`; count one returns `[start]`, while a larger count includes the
> exact stored start and stop. For interior position `i`, the exact
> mathematical rational `i/(count-1)` is rounded once to `p_float`, then the
> declared `p_float` composition `start + (stop-start)*weight` executes in that
> order. Squeeze removes the selected singleton dimension. Unsqueeze inserts a
> singleton dimension and stack inserts the input-list length at the selected
> position; all three are rank-polymorphic, bit-preserving reshape/concat
> operations. Squeeze normalizes a negative axis by adding the input rank once
> and then requires `0 <= axis < rank`; its selected extent must be one.
> Unsqueeze and stack normalize a negative insertion axis by adding the result
> rank once and then require `0 <= axis <= input_rank`.
> Stack rejects an empty list or any inconsistent input dimension list or
> dtype. `where_indices`
> returns the increasing row-major int64 flat indices of true elements. The
> float `squeeze` and `unsqueeze` adjoints are the reverse reshape graph.
> The float `stack` adjoint slices the output cotangent along the inserted axis
> in increasing input-list order and returns a `List` of tensor cotangents with
> exactly the input list's length, shapes, and dtype. Integer and bool stack are
> forward-only. For
> `linspace` with count one, start receives the sole output cotangent and stop
> receives exact zero. For a larger count, the start and stop contributions
> use the exact interpolation weights above, are each enumerated by increasing
> output index, and combine through separate canonical adjacent-pair balanced
> trees. Integer and bool outputs have no cotangent.
>
> Test tolerances have the same active float dtype as the values, are finite,
> and are nonnegative. NaN is unequal to every value,
> including itself; equal signed infinities are close, opposite or finite/
> infinite pairs are not; signed zeros compare equal. `assert_close_tensor`
> admits exactly one common active float dtype `p` and equal shapes. Its
> comparison executes at `p`'s [04-NUM-8] arithmetic width: f16/bf16/f32
> values compare in f32 and f64 values compare in f64, with the stored
> same-dtype tolerance converted exactly to that arithmetic width. It never converts either tensor
> through f64. Each finite element pair computes `abs(actual - expected)` at
> that width and is close exactly when that difference is less than or equal
> to the converted tolerance; zero tolerance uses numeric equality. The first row-major
> mismatch is reported. Scalar `assert_close` applies the same own-width rule
> to any active float dtype. `assert_eq` uses [05-OP-36] equality for one
> common scalar or recursively comparable `Q`: float NaN is unequal, equal
> signed infinities and signed zeros are equal, and every container/ADT field
> follows the exact recursive rule. Direct tensor arguments use
> `assert_eq_tensor`, which requires equal shapes and one common active element
> dtype. Float elements use the same numeric-equality rule as scalar
> `assert_eq`; signed-integer and bool elements use exact equality. It reports
> the first unequal row-major element. `assert_shape` requires its expected list
> to contain only nonnegative int64 extents and compares its length and every
> entry to the tensor's complete shape in axis order; a negative expected
> extent is a `Domain` failure. Exact tensor assertions report the first row-major
> mismatch. A failed assertion is a branded `Test` failure with
> its label, never a panic, inert stub, or default result.
>
> `mmap_size(path)` returns the exact nonnegative byte length as int64 and
> traps `Overflow` when the platform length has no int64 image.
> `read_head_bytes(path, count)` requires `count >= 0` and returns the first
> `min(count, file_length)` bytes as exact int64 values in `0..=255`.
> Each call returns the length or byte prefix observed by one successful host
> filesystem operation. Any error actually reported by open, stat, or read
> propagates as `IO`; concurrent mutation may yield a valid observed snapshot
> or that reported error. Neither operation substitutes zero or an empty value
> for an error or promises detection of a mutation that the host did not report.
>
> Date operations use the proleptic Gregorian calendar with astronomical
> int64 years. `try_date` returns `None` for an invalid month/day and otherwise
> the exact fields; `date` traps `Domain` for the same invalid input. Every
> operation accepting `Date` validates it first. With `ordinal` and
> `from_ordinal` denoting the exact proleptic-Gregorian bijection,
> `add_days(d,n) = from_ordinal(ordinal(d) + n)`,
> `sub_days(d,n) = from_ordinal(ordinal(d) - n)`, and
> `days_between(lhs,rhs) = ordinal(rhs) - ordinal(lhs)`. This mathematical
> ordinal arithmetic traps `Overflow` only if its final Date year or int64
> day difference is unrepresentable; `sub_days` treats
> `int64::MIN` mathematically rather than negating it at width. `duration`
> forms the exact mathematical component total and applies Euclidean division
> by 86400: `days` is the floor quotient and the nonnegative remainder supplies
> hours in `0..23` and minutes/seconds in `0..59`. It traps `Overflow` exactly
> when that final normalized `days` field has no int64 representation. Valid
> date comparisons are lexicographic on `(year, month, day)`.
> `day_of_year` is one-based; `day_of_week` fixes `1970-01-01` as Thursday
> and returns Monday through Sunday in calendar order, while
> `day_of_week_name` returns the corresponding lowercase ASCII name.
>
> Date text has one canonical form. Years `0000` through `9999` use exactly
> four digits. A negative year uses `-` followed by exactly
> `max(4, digits(|year|))` decimal digits, where `|year|` is the exact
> mathematical magnitude rather than an int64 `abs`, with leading zeros to that width;
> a year above 9999 uses `+` plus its unpadded decimal digits.
> Month and day are two digits, separated as `year-MM-DD`.
> `date_to_string` emits that form for a valid Date and traps `Domain` for an
> invalid raw constructor value. `parse_date` accepts only that canonical form
> and returns `None` for a syntax error, an int64-unrepresentable year, or an
> invalid calendar date.
>
> A tokenizer operation first validates that vocabulary IDs are injective,
> the inverse is exact, `unk_id` exists, merge ranks are unique nonnegative
> integers, every merge key denotes one unambiguous token pair, and no token
> pair occurs at more than one merge rank. Encoding
> starts from Unicode scalar-value tokens, repeatedly selects the lowest merge
> rank and then the leftmost pair, and replaces every nonoverlapping selected
> pair from left to right with the concatenation of its two token strings.
> When no merge remains, `encode` maps each final token through `vocab`, using
> `unk_id` exactly when the token is absent. `decode` maps each ID through the
> inverse vocabulary, using the unknown-token string exactly when the ID is
> absent, and concatenates the resulting token strings with no separator.
> `batch_encode` requires `max_length >= 0`, truncates on the
> right, and right-pads with the exact int64 `pad_value`.
> `try_load_tokenizer` accepts a JSON object whose `model` is an object with
> exact string `type: "BPE"`, a `vocab` object of unique string keys to unique
> `JsonInt` IDs, a `merges` array of strings each containing exactly two
> nonempty space-free tokens separated by one ASCII space, and an `unk_token`
> string present in `vocab`. Merge array position is the unique nonnegative
> rank. Missing files, malformed JSON, schema mismatch, or an invariant
> violation return `None`; other read failures trap `IO`. `load_tokenizer`
> uses the same schema and traps `Domain` instead of returning `None` after a
> successful read.
>
> IO and process functions introduce the tabled `IO` effect. Process calls
> pass the executable and argument vector directly without invoking a shell,
> capture stdout/stderr and every ordinary nonzero exit, encode a signal exit
> as status `-1`, and trap on spawn failure or non-UTF-8 captured text. `run`
> resolves a name without a directory separator through the call-time `PATH`
> and otherwise uses the supplied path. Every host-evaluator context carries
> an optional absolute, executable `chelis_executable` capability.
> `run_chelis` invokes exactly that path, without a `PATH` lookup; a CLI
> context configures its own executable, while an embedding or test harness
> must configure the intended toolchain explicitly. An absent, relative, or
> non-executable capability traps `IO` with
> `run_chelis: no valid Chelis executable configured`. Both
> inherit the current working directory and environment. They are `Io`
> operations under [05-HOST-2] in every language execution mode. IO and process
> operations are outside AD. Pure constructors, tokenizers, time values, and
> decimal values have no cotangent unless their governing atom explicitly
> defines one. Comparison predicates and assertions contribute zero cotangent
> to differentiable leaves; an assertion's `Test` effect is preserved.
> Collection operations, random initializers, shape constructors, and sort
> wrappers use their explicit adjoint or forward-only rule above; no blanket
> host-family rule overrides a float adjoint. No callable derives authority
> from its implementation body or age.

---

### 3.8 Named Lossy Cast Forms

[04-NUM-14] makes the default `cast` a CHECKED cast: a fractional or
non-finite float cast to an integer target traps `Domain`. That default
does not change. The named forms below are the explicit, auditable escape
hatches a program opts into when a lossy conversion is the intent. This
section defines the complete named ladder. Each form is a distinct operation,
not a mode parameter to `cast`.

> **[05-OP-6]** `cast_trunc(source, target)` is the explicit truncating narrowing cast
> from a float source dtype to an integer target dtype. For a **finite** source value it
> yields the integer part truncated toward zero (the value with its fractional part
> discarded), finalized at the target width; if that truncated integer is outside the
> target range it traps `overflow` (never wraps or saturates). A **non-finite** source
> (`NaN`, `±inf`) traps `Domain` — truncation of a non-finite value has no integer
> meaning. On any source/target pair that is not float→integer, `cast_trunc` is a type
> error (use `cast` / [04-NUM-14]); it never widens, never rounds, and never applies to
> `bool`. Semantics are identical on scalar and tensor surfaces and identical across the
> eval and compiled lanes at the declared widths of [04-NUM-8]. `cast_trunc` is
> **non-differentiable**: its adjoint is zero almost everywhere (the map is piecewise
> constant), so it carries the `no_grad` rule — a gradient goal through it is a clean
> error, never a silent zero that masks a modeling bug (same discipline as `argmax`).

`cast_trunc(x, T)` agrees with `cast(x, T)` exactly when `x` is already
finite and integral and in range (both yield the same integer); it differs
only by *defining* the fractional case as truncation where `cast` traps
`Domain`. `cast_trunc` has no accumulator.

> **[05-OP-23]** `cast_saturate(source, target) -> result` admits an active
> signed-integer or float source dtype and a signed-integer target dtype on a
> scalar or tensor surface. It preserves the source surface and tensor
> dimensions and returns the target dtype. It reads the source exactly at its
> stored dtype. A finite float is truncated toward zero, then the resulting
> mathematical integer is clamped to the target's inclusive range; an integer
> source is clamped directly. Negative infinity returns the target minimum,
> positive infinity the target maximum, and NaN traps `Domain` as operation
> `cast_saturate` at the target dtype. It never traps `Overflow`, wraps, or
> converts through another numeric dtype. `bool`, `string`, reserved dtype spellings, and
> non-integer targets are type errors. It has no accumulator and is
> non-differentiable: `grad` rejects it.
>
> **[05-OP-24]** `cast_wrap(source, target) -> result` admits an active
> signed-integer source and signed-integer target on a scalar or tensor
> surface. It preserves the source surface and tensor dimensions and returns
> the unique signed target-width representative congruent to the exact stored
> source modulo `2^target_width`. It never traps for overflow, saturates, or
> converts through a float dtype. Float, `bool`, `string`, reserved dtype spellings, and
> non-integer targets are type errors. It has no accumulator and is
> non-differentiable: `grad` rejects it.

There is no `cast_round` operation. A program that wants rounding followed by
checked conversion spells `cast(round(source), target)`, so the rounding and
checked-cast boundaries remain independently observable.

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

Every numeric callable states one of three contracts: an adjoint, a zero
cotangent, or a structural `grad` rejection. No operation acquires an adjoint
from a backend fallback.

**Zero-cotangent predicates and sources.** Each exact [05-OP-36] identity —
`cmplt`, `lt`, `eq`, `neq`, `gt`, `gte`, and `lte` — together with `is_nan`,
`is_finite`, `is_infinite`, `const`, `load`, and `shape` contributes zero
cotangent. This permits a predicate or metadata read to participate in a
differentiable guard without pretending that the predicate itself has a useful
derivative.

**Structural rejections.** On float operands, `floor`, `ceil`, and `round` are
piecewise constant and `grad` rejects them with an
`AdRejectionReason::PiecewiseConstant` error rather than silently returning a
zero gradient. [05-OP-1] `round_to` and [04-NUM-14] float-to-integer or
float-to-bool default casts use that same structural reason. Float-to-float
default casts use [04-NUM-14]'s exact backward cast. `cast_trunc`,
`cast_saturate`, `cast_wrap`, `and`, `or`, `not`,
`count`, `argmax_reduce`, and `argmin_reduce` likewise reject `grad` under their atoms.
The `wrap_*` operations and integer reduction/unary forms are forward-only
because integer values do not carry cotangents. Integer `floor`, `ceil`, and
`round` are exact identities and may be erased before AD. The `Diff` effect
reports a non-differentiable operation before execution.

**Almost-everywhere differentiable:** `max_elem` (gradient is zero at the boundary where inputs are equal), `relu` via `max_elem(x, 0)` (gradient is zero at x=0). These are valid targets for `grad` — the subgradient convention (pick one side) is standard in ML.

**Second-order derivatives:** `grad(grad(f))` is valid exactly when every
operation reached by `f` has the required second-order adjoint.

---

## 6. Reference Implementations

For each RISC primitive, the following pseudocode gives the direct reference
implementation used as the C-backend semantic oracle.

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
// `acc_t` is the accumulator type resolved per §2.3 / spec/04 §5.7.1,
// NOT unconditionally `float`. `reduce_balanced` pairs adjacent values,
// carries an odd tail unchanged, and repeats exactly as §2.3 specifies.
for (int i = 0; i < outer; i++)
  for (int j = 0; j < inner; j++) {
    acc_t level[axis_size];
    for (int k = 0; k < axis_size; k++)
      level[k] = input[i * axis_size * inner + k * inner + j];
    output[i * inner + j] = reduce_balanced(level, axis_size, add, 0);
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

These C implementations are illustrative reference shapes, not the semantic
authority. Where one of them and a normative section disagree, the normative
section wins and the pseudocode has a bug: §2.3 owns reduction order and
accumulator type, `spec/04-type-system.md` [04-NUM-8] owns the arithmetic
width every `acc` and every temporary is computed at, and §5.7.1 owns the
accumulator defaults.

The GPU backend must satisfy [05-OBS-3]'s per-operation, per-arithmetic-width
agreement table. Blanket `1e-6` (f32) / `1e-12` (f64) bounds are not a
conforming cross-lane oracle.

---

## 7. The Unsupported-Case Response Contract

This section governs how an unsupported case is reported. It does not decide
which operations, dtypes, targets, or parameter shapes are supported; those
decisions belong to their owning numbered-spec atoms. The compiler
architecture that enforces this contract is specified in
`spec/design/loud_unsupported.md`.

> **[05-UNS-1]** When any stage encounters a case it does not support -
> an op, builtin, dtype, kernel, construct, or parameter shape - it
> SHALL respond with a diagnostic through its failure channel. No stage
> SHALL substitute a value, type, dtype, kernel, or emission.

For closed compiler/runtime vocabularies, "SHALL NOT substitute" is enforced
by construction: raw symbols and numeric IDs are converted through a
`Result` decoder into a no-`Unknown`, no-`Default` enum at the first boundary;
all semantic matches are exhaustive and wildcard-free. An unknown runtime
dtype ID must fail before sizing or buffer access. A source-text scanner or
count allowlist is supporting evidence only and cannot satisfy [05-UNS-1].

> **[05-UNS-2]** The diagnostic SHALL surface at the earliest competent
> stage - the checker for type-answerable questions, the build for
> target- or lowering-decided ones, the runtime only for genuinely
> dynamic conditions - and SHALL be branded with the `unsupported:`
> prefix, name what was encountered and at which stage, and carry a
> span where one exists and the supported alternative where one exists.

> **[05-UNS-3]** A panic or assertion failure reachable from `.ch` or
> `.dp` source input is a defect; internal panics are reserved for
> compiler invariants whose upstream guarantee is named at the panic
> site.

> **[05-UNS-4]** A pre-codegen gate MAY make an unsupported diagnostic
> earlier or more specific; it SHALL NOT be the sole defense against an
> unsupported case reaching emission, and a gate/emitter disagreement is
> a defect in the gate.

> **[05-UNS-5]** An unsupported diagnostic SHALL carry the authority for its
> disposition. A language-rejected case cites the normative atom that decides
> it. A legal operation unavailable on the selected target identifies the
> exact typed capability cell. Those categories SHALL be distinguishable at
> the diagnostic surface, and a rejection carrying neither authority is a
> defect. Project scheduling or issue metadata may be associated with a
> capability cell outside this normative contract; it is not semantic
> authority and does not alter legality.

> **[05-UNS-6]** The machine-facing kind of a diagnostic is drawn from
> a closed vocabulary with stable spellings; the build surface's
> spelling for this contract's rejections is `unsupported_feature`.
> A machine consumer SHALL be able to distinguish an unsupported-case
> rejection from an internal compiler error by kind alone. Producing
> this kind for anything other than a typed unsupported rejection, or
> a different kind for one, is a defect.

---

## 8. Observation And Formatting Contract

> **[05-OBS-1]** Every exit that renders a stored numeric value as text -
> `print`, `to_string`, `to_list`, diagnostics, the wire schema's rendering - SHALL
> emit text that parses back to exactly the stored bits at the value's
> own dtype width, except that every NaN payload renders as the exact spelling
> `NaN` and parses to §3.7's canonical quiet-NaN image at that dtype. Thus NaN
> text round-trips at the class level and deliberately loses payload bits.
> All exits within a lane SHALL agree with each other and with this rule.

> **[05-OBS-2]** Integer dtypes SHALL print as integers with all digits
> exact; floats SHALL print the shortest string that round-trips at
> their own width; `bool` SHALL print `true`/`false` at every exit; the
> number grammar (digit selection, exponent form, special-value
> spellings) SHALL be identical across lanes and is pinned in §8.1.

> **[05-OBS-3]** Cross-lane VALUE differences are permitted only for the
> ops listed in the per-op tolerance table below, within the listed bound;
> `sqrt`
> SHALL be correctly rounded (bound zero, per chelis#719). Formatting
> differences are never within tolerance.

The following table is normative. Implementations SHALL mirror it through a
closed operation identity derived from the operation being compared; callers
MUST NOT supply an unchecked textual identity. The executable mirror is
tripwire-checked byte-for-byte against this block.

<!-- BEGIN GENERATED OBSERVATION TOLERANCE TABLE -->
| operation | maximum cross-lane value difference | authority |
|---|---:|---|
| `atan` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `cos` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `exp` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `log` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `sin` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `sqrt` | 0 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
| `tan` | 1 ULP at [04-NUM-8]'s arithmetic width | [05-OBS-3] |
<!-- END GENERATED OBSERVATION TOLERANCE TABLE -->

Operations absent from the table have a zero-ULP bound. In particular,
add, subtract, multiply, divide, comparisons, reductions, and compound
builtins are exact-by-default; a new or misspelled operation identity
cannot inherit a float fallback. A table row is eligible only when both
lanes compute at [04-NUM-8]'s declared arithmetic width. A lane pair that
does not meet that precondition is a failed comparison and may not be
laundered through the table. For f16 and bf16, which compute at f32 and
round once into narrower storage, differing final strings do not by
themselves prove a distance at the f32 arithmetic width: a conforming
oracle SHALL compare the lanes' pre-final f32 bit patterns and verify
that each rounds to its observed f16/bf16 result. Missing or inconsistent
pre-final evidence is a failed comparison, never an existential
rounding-bin acceptance. Non-finite and signed-zero mismatches are never
toleranced. Byte-different strings which denote identical stored bits are
formatting violations under [05-OBS-2], not value differences.

This section is the table's single normative address. The table covers only
implementation variance at a single width - one lane's libm or SLEEF or
vForce against another's - and never a structural precision mismatch.

> **[05-OBS-4]** A scalar-typed value SHALL render as the bare scalar at
> every exit in both lanes, including as a top-level labeled root
> (`root = 0.1`, never `root = tensor(shape=[], data=[0.1])`). A rank-0
> tensor renders as its single element, bare: the
> `tensor(shape=[], data=[..])` wrapper is not an exit form. An
> implementation MAY realize scalar bindings through rank-0 tensors
> internally; that realization SHALL NOT leak into the observation
> channel.

> **[05-OBS-5]** Every exit in both lanes SHALL truncate tensor element
> rendering after 32 elements, marking the cut with `, ...` inside the
> `data=[..]` brackets. `to_list` and the wire schema never truncate:
> full-element fidelity is theirs.

> **[05-OBS-6]** Every root SHALL render with a `name = value` label at
> every exit in both lanes. The bare-when-single form is removed. Render
> order is manifest entry order. A lane that cannot produce a root it
> owes SHALL emit [05-UNS-1] naming that root, the lane, and the reason.

### 8.1 The Number Grammar

The grammar is Rust `{:?}` (`Debug`) float formatting, normatively
(`faithful_observation.md` §C1.3; `Display` is NOT this grammar - it
never emits e-notation):

- shortest round-trip digits at the value's own width;
- decimal form exactly when the RENDERED magnitude - the value the
  chosen shortest digits denote - is zero or satisfies
  `1e-4 <= |v| < 1e16` (the normative threshold constants
  `DECIMAL_LOWER_BOUND` / `DECIMAL_UPPER_BOUND`).
  The rule follows the digits actually printed, not the stored
  magnitude: when a width's ulp straddles a threshold, the shortest
  rendering can sit on the other side of it - the bf16 whose image is
  9.9921e15 renders `1e16` (e-notation), and the f32 whose image is
  9.9999997e-5 renders `0.0001` (decimal). This is rustc's observed
  `{:?}` behavior, which compares against the constants at the value's
  own width - equivalent to the rendered-magnitude rule at every
  representable boundary;
- decimal renderings of integral values keep one fractional digit
  (`2048.0`, never `2048`);
- e-notation is `<mantissa>e<exp>`: lowercase `e`, no `+`, no zero
  padding (`1e-7`, `9.999999980506448e19`);
- specials spell `inf` / `-inf` / `NaN` (NaN round-trips at the class
  level: exit text carries no payload); `-0.0` prints with its sign;
- `f16`/`bf16` print the shortest decimal whose parse-back (`strtod` to
  f64, then one correctly-rounded narrowing to the half width - safe by
  [04-NUM-1]'s single-rounding argument) yields the stored bits,
  for all bit patterns; their decimal/e-notation decision applies the
  same rendered-magnitude rule to the chosen digits, and a same-length
  candidate tie breaks to the numerically closest, then the even mantissa;
- integer dtypes print exact base-10 digits (i64 formatting, never
  through double).

**Tag-vs-bits disagreements print the bits:** when an integer- or
bool-tagged tensor slot stores a value outside the tag's value set, the
element renders the stored value without truncating it into a well-formed
lie. Rendering never repairs, rounds, or rejects stored values.
