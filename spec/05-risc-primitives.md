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
Outputs remain owned tensors. Source borrow syntax and primitive-DAG borrow markers
are erased before backend emission: primitive nodes and kernels carry values, not
source markers. The resolved disposition of every use is not erased; ownership
lowering first records explicit borrow, move, clone, and terminal `Drop` obligations
in the verified ownership representation consumed by every backend. Consuming
operations such as `realize` and explicit `drop` keep owned parameters.

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

Chelis rejects deterministic literal-driven value errors at check time when enough
information is concrete in source (for example, statically inconsistent `einsum`
extents or out-of-bounds scatter indices). When those constraints depend on runtime
values instead, the evaluator and generated C runtime reject them during execution;
compiled C exits non-zero rather than aborting. Duplicate replace-scatter indices are
not errors: they follow §3.5's deterministic last-write-wins rule.

### 1.5 Builtin semantic identities

The [builtin identity registry](registry/builtin_semantic_identities.md)
is incorporated by reference into each numbered operation atom named in its
Atom column. Each atom incorporates exactly the rows that name it, including
its rejected domain/case identities. An identity is the exact triple of
domain, canonical operation, and builtin-owned case; the rows have no
semantic ordinals.

Each atom supplies its signatures, admitted and rejected domains, results,
failures, differentiation rules, and any accumulator or traversal order.
The registry supplies identity membership only: it adds no behavior,
default, alias, or backend-support disposition. Arithmetic and storage
widths remain governed by [04-NUM-8].

---

## 2. RISC Primitives (Tier 1)

### 2.1 Elementwise Binary

| Name | Signature | Semantics | AD Adjoint (∂L/∂inputs given ∂L/∂output = g) |
|---|---|---|---|
| `add` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise addition | `(g, g)` |
| `sub` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise direct subtraction | `(g, -g)` |
| `mul` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise multiplication | `(g * y, g * x)` |
| `div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise IEEE-754 division `a / b` (**float operands only**) | `(g / b, -g * (a/b) / b)` (= `(g/b, -g*y/b)` using `y = a/b`) |
| `floor_div` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise floor division: `floor(a / b)`, rounding toward −∞ | Non-differentiable (piecewise constant); `grad` rejects it |
| `trunc_div` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise truncating division (round toward zero), **integer operands only** | Non-differentiable (piecewise constant); `grad` rejects it |
| `wrap_add` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular addition | Non-differentiable; `grad` rejects it |
| `wrap_sub` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular subtraction | Non-differentiable; `grad` rejects it |
| `wrap_mul` | `(&tensor[D,p_int], &tensor[D,p_int]) -> tensor[D,p_int]` | Element-wise modular multiplication | Non-differentiable; `grad` rejects it |
| `cmplt` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,bool]` | Element-wise less-than comparison | Non-differentiable (zero gradient) |
| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise maximum | Whole `g` flows to the operand selected by [05-OP-40] |
| `min_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise minimum | Whole `g` flows to the operand selected by [05-OP-40] |

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

**Scalar `max_elem`/`min_elem`.** The element-wise extrema also admit two
scalar operands of the same numeric dtype
and return a scalar of that dtype. This is the rank-zero instance of the
tensor rule, not scalar/tensor broadcasting: a scalar and a non-scalar tensor
remain a dimension mismatch. The scalar forms admit the same signed-integer
and float precisions as their tensor forms and use the same adjoint rule.

> **[05-OP-40]** `max_elem(left, right) -> result` and
> `min_elem(left, right) -> result` each admit two values of one active
> signed-integer or float dtype on the same scalar surface or on tensor
> surfaces with identical dimensions. The result has that same surface,
> dimensions, and dtype. Each operand is read at its exact stored width; the
> operation performs selection, not arithmetic or numeric conversion. On
> floats, it returns the first NaN in operand order when either operand is NaN,
> preserving that value's exact stored bits, including payload and sign.
> Otherwise it returns the numerical maximum or minimum respectively and
> returns the first operand on every equality, preserving its exact stored bits,
> including signed-zero equality. Signed integers are compared exactly at their declared
> width and likewise preserve the first operand on equality. For floats, the
> adjoint routes the whole cotangent to the selected operand and exact zero to
> the other operand; this is the internal `ExtremaAdjoint` contract.
> `relu` is a distinct Tier-2 identity whose adjoint is
> [05-OP-43]'s zero-at-zero rule, not this selection rule's tie behavior.
> Signed-integer forms are forward-only and `grad`
> rejects them. Both operations have no accumulator. `min_elem` is a direct
> selection identity and never lowers through arithmetic negation. `bool`,
> `string`, reserved dtype spellings, mixed dtypes or surfaces, and mismatched
> tensor dimensions are type errors.

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
> `insert(g / divisor, axis, original_extent)` at the operand dtype. One or
> more positional or named axes follow spec/04 §4.5.3: the call executes
> these exact single-axis graphs in highest-original-position-first order and
> the adjoint reverses that composition. Mixed, duplicate, dynamic, absent,
> ambiguous, or out-of-range axes are type errors. `mean` has no accumulator
> parameter of its own.

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

> **[05-OP-13]** `min_reduce(x, axes...) -> result` has the same dtype,
> finalization, empty-axis, accumulator, and differentiation contract as
> [05-OP-12], replacing maximum by minimum. It returns the first NaN in
> increasing axis-index order with its exact stored payload and sign bits;
> otherwise it preserves the first stored
> representation among equal minima. For float operands, every element equal
> to the selected non-NaN minimum, including equal positive or negative
> infinities, receives the upstream cotangent divided by the number of equal
> minima.

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

> **[05-OP-15]** `argmax_reduce(x, axis) -> result` admits every active
> signed integer and float tensor dtype and returns `int64` indices with the
> selected axis removed. If the slice contains NaNs, it returns the lowest
> axis index containing NaN; otherwise it returns the lowest axis index whose
> stored value is maximal. Comparisons never convert through another dtype.
> A zero-length axis is a type error when statically known. If an
> execution-time extent is zero, the operation traps `Domain` as operation
> `argmax_reduce` at result dtype `int64`. It has no accumulator and is
> non-differentiable: `grad` rejects it.

> **[05-OP-16]** `argmin_reduce(x, axis) -> result` has the signature, dtype,
> exact-comparison, empty-axis, accumulator, and non-differentiability
> contract of [05-OP-15], returning the lowest NaN index when present and
> otherwise the lowest axis index whose stored value is minimal.

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
> `AdRejectionReason::IntegerReductionOutput`; it never receives a silent zero
> cotangent.

*(Not fully implemented; chelis#1287 owns the evaluator and C delivery,
chelis#1291 the HIP and Metal kernels.)*

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

> **[05-AXIS-1]** A reduction axis, `expand`'s broadcast axis, and `insert`'s
> new-axis position SHALL be
> statically resolvable either as an integer constant (a literal or a literal
> wrapped in an integer cast) or as a named dimension of the operand. A
> runtime integer expression and an unknown dimension name are type errors at
> the call site; no lowering or backend SHALL substitute axis zero or another
> axis. `insert`'s named-axis form (`spec/04-type-system.md` §4.5.3) names the
> dimension it creates, which is by construction not a dimension of the
> operand; that name SHALL be statically resolvable in the same sense and is a
> type error when it already names an operand dimension. Its optional anchor
> is a named dimension of the operand and follows the operand rule above.

The reduction axis must resolve statically: a literal, a
`cast(N, int32)`-wrapped literal, or a named operand dimension as specified by
`spec/04-type-system.md` §4.5.3. Because the output shape is "remove the
dimension at position `axis`", the type checker cannot determine which
dimension is dropped from a runtime integer value. A reduction whose axis is
a runtime expression (for example a function-parameter `int32`) is rejected
at the reduction call site with a diagnostic naming the constant-or-named-axis
requirement, rather than leaving the output shape unresolved (chelis#259).
The same constraint and diagnostic apply to `expand`'s broadcast axis and
to `insert`'s new-axis position.

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
| `expand` | `(&tensor[D,p], axis: int32, size: int64) -> tensor[D',p]` | Set the size-1 dimension at position `axis` to width `size`. Rank is unchanged and the operand's extent at `axis` is 1. Does NOT copy data. |
| `insert` | `(&tensor[D,p], axis: int32, size: int64) -> tensor[D_plus,p]` | Insert a new dimension of width `size` at position `axis`, producing rank `rank(x) + 1`. Does NOT copy data. Named-axis and anchored forms: `spec/04-type-system.md` §4.5.3. |
| `pad` | `(&tensor[D,p], padding: List<List<int64>>, fill) -> tensor[D',p]` | Add elements at boundaries. `padding` specifies (before, after) per axis. |
| `shrink` | `(&tensor[D,p], bounds: List<List<int64>>) -> tensor[D',p]` | Slice: extract a contiguous sub-tensor. `bounds` specifies (start, end) per axis. |
| `stride` | `(&tensor[D,p], strides: int64...) -> tensor[D',p]` | Strided access: take every n-th element along each axis. |

`expand` and `insert` each take the (tensor, axis, size) triop
positionally; a two-argument list form is an arity error. `expand` sets an
existing size-1 axis and leaves the rank alone; `insert` adds an axis and
raises the rank by one. The named-axis form (`insert(x, new, size)` with a
dimension name) and the four-argument anchored form belong to `insert` and
remain as `spec/04-type-system.md` §4.5.3 states them. Where the axis is
positional it is axis-domain `int32`; `size` is extent-domain `int64` in
every form, which makes the canonical broadcast idiom
`insert(b, axis, shape(x, axis))` well-typed by construction.

**Movement AD adjoints:**

| Name | Adjoint |
|---|---|
| `reshape` | `reshape(g, original_shape)` |
| `permute` | `permute(g, inverse_permutation)` |
| `expand` | `insert(sum(g, axis), axis, 1i64)` — sum over the broadcast axis, then restore its extent-1 slot so the adjoint keeps the operand's rank |
| `insert` | `sum(g, axis)` — collapse the inserted dimension |
| `pad` | `shrink(g, inverse_padding)` — extract the non-padded region |
| `shrink` | `pad(g, inverse_bounds)` — pad gradient back to original size |
| `stride` | [05-MOV-1]'s exact zero-filled inverse sampling map at the original shape; runtime steps have zero cotangent |

> **[05-OP-65]** `axis_movement(arguments...) -> result` governs exactly
> `permute`, `expand`, and `insert`, with the signatures, dimension mapping,
> and adjoints in §2.4. Their positional axes and permutation entries are
> `int32`; sizes are `int64`, under [05-DIM-1..3]. `permute` requires a
> permutation of the operand's axes. `expand` preserves rank and requires an
> extent-1 operand axis; `insert` adds one axis, including at the trailing
> position. Named-axis forms retain spec/04 §4.5.3's distinct contract.
> Runtime extents and their claims obey §2.4.1 and [05-MOV-1].
>
> Each operation preserves every admitted operand dtype and its stored element
> representations, with no numeric conversion or forward element arithmetic.
> Axis and size parameters are discrete zero-cotangent boundaries. Float
> cotangents follow the movement adjoints above; any adjoint reduction uses
> [05-OP-30]'s exact accumulator and finalization rules. Integer and bool
> payloads are forward-only. The forward operations have no accumulator
> parameter. The shared `Expand` wire representation preserves the two source
> operations' rank and axis contracts under spec/10 §3.4; sharing that
> representation does not merge their source signatures.

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

A movement bound (`pad` before/after, `shrink` start/end, `stride` step), an
`expand` or `insert` size, and a `reshape` target extent are each represented
as a `RtDim`:

- `Lit(n)` — a compile-time-constant extent.
- `ToEnd` — the full-axis sentinel; legal only as a `shrink` end (the identity
  slice of a symbolic bystander axis), and well formed only when the start
  paired with it is `Lit(0)`. A `ToEnd` end over any other start is a
  malformed bound: every stage that validates a bound rejects it rather than
  resolving it to a slice.
- `Node(i)` — a **runtime** extent read from the owning node's `inputs[i]`, a
  rank-0 integer scalar. The index is absolute: `inputs[0]` is always the
  tensor operand, and `inputs[1..]` are the bound scalars, any rank-0 `int32`
  axis scalar a node-valued `InputAxis` axis names, and any shape-only tensor
  operand an `InputAxis` names.
- `Sym(name)` — a symbolic dim declared elsewhere (e.g. a bystander `batch`);
  legal only as a `reshape` target.
- `InputAxis(t, a)` — the extent of an earlier tensor node's axis, read
  directly from that tensor's shape metadata: `t` is an absolute index into
  the owning node's `inputs` naming a tensor operand, and `a` is a normalized
  `int32` axis literal in `0..rank(t)` (spec/04-type-system.md §4.7.1
  normalizes a negative literal statically) or an absolute input index naming
  a rank-0 `int32` scalar (a computed axis under [05-OP-7]). Legal only as an
  `expand` or `insert` size or a `reshape` target. It is the folded
  extent-argument form of §2.5.1 for a direct `shape(x, axis)` extent
  argument and, in an `expand` or `insert` size, for an in-scope dimension
  binder instantiated by a tensor axis; a
  `reshape` target that restates such a binder is `Sym`, and the same read
  bound to a `pad`, `shrink`, or `stride` position is the rank-0 `Node` form.
  The read carries no identity: whether the resulting axis keeps the source
  dimension's name is decided by ordinary type reasoning
  (spec/04-type-system.md §4.7.3), and an unproved identity is a fresh extent
  under an equality guard.

`reshape` admits `Lit`, `Node`, `InputAxis`, and `Sym`; `expand` and `insert`
admit `Lit`, `Node`, and `InputAxis`; `pad`, `shrink`, and `stride` admit
`Lit` and `Node`, plus `ToEnd` for a `shrink` end.

`expand` sets the extent at `axis` and is well formed only when the operand's
extent at `axis` is 1 (the size-1 broadcast of §2.4's table); the operation is
a claim that the operand's extent at `axis` is 1. A literal operand extent at
`axis` other than 1 is a type error. A symbolic or runtime operand extent at
`axis` other than 1 fails that claim's runtime extent guard and traps
`Domain`, placed and rendered per `spec/04-type-system.md` §4.7 and
[04-NUM-9].

Runtime bounds are validated in every execution mode with matching language
errors: a negative bound, a shrink range overshoot, a non-positive stride
step, a negative reshape target extent, and a reshape target whose element
product disagrees with the input all trap before allocation or access. The reshape numel
guard fires for ANY reshape whose output or input extents are not all static
literals — named-dimension targets read as `InputAxis` and literal targets
over runtime-sized inputs included, not only node-valued targets — and
same-shape elementwise ops guard
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

> **[05-MOV-1]** Runtime movement bounds, `expand` and `insert` sizes, and
> reshape targets, their validation, and the exact adjoints above SHALL be
> available in every
> language execution mode for every active tensor dtype admitted by the owning movement
> operation. Eval, C, HIP, and Metal execute the same runtime values and
> traps. No lowering may erase a runtime value, substitute a literal bound,
> require host provenance, emit a statically guessed extent, or turn a backend
> implementation gap into a language restriction.

*(Not fully implemented; chelis#1277, chelis#1298.)*

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

- **As an extent argument** to `expand` / `insert` / `reshape`, a `shape()`
  read is folded
  into the movement node's `InputAxis` carrier (§2.4.1), not materialized as
  a value node.
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

*(Not fully implemented; chelis#1298.)*

A `shape()` read whose `axis` is data- or metadata-derived remains the same
operation as a literal-axis read. Using the extent as a runtime movement-op
bound or as a reshape target computed by integer arithmetic over one (a
`shrink`/`stride`/`pad` bound or window count derived from a `shape()` value,
and the integer arithmetic feeding it) uses §2.4.1's node-valued `RtDim`
capability; it does not allocate a second shape operation or a
compile-time-only alias.

#### Python full-shape metadata

> **[05-OP-45]** `python_tensor_shape(tensor) -> extents` governs exactly
> the registered binding identities in
> `spec/registry/python_tensor_metadata.md`, incorporated by reference.
> It borrows a validated tensor descriptor and returns all extents in axis
> order as exact nonnegative `int64` values, exposed as Python integers.
> The result length equals the descriptor's dynamic int32 rank, including
> an empty vector for rank zero. It reads no tensor elements and preserves
> every extent at every active element dtype and admitted device under
> [04-NUM-11]. No host-width conversion, fixed-rank truncation, fabricated
> extent, or unchecked foreign descriptor is admitted. Descriptor validity
> and lifetime are established before the wrapper is constructed, as
> specified in spec/11 §1.2; observing shape does not authorize allocation
> or prove equality of capacities. This Python metadata operation is
> non-differentiable and has no arithmetic accumulator. It does not alter
> the language `shape(x, axis)` operation or its [05-OP-7] zero-cotangent rule.

(The exact binding shape carrier is not fully implemented; see chelis#1288.)

### 2.6 Effectful Primitive

| Name | Signature | Semantics | AD / effect note |
|---|---|---|---|
| `dropout` | `(&tensor[D, p_float], p_float) -> tensor[D, p_float]` | Apply [05-OP-37]'s inverted-dropout transform using the active `with seed(...)` handler and a same-dtype rate | Introduces `Random`. Its pathwise adjoint reuses the exact forward mask. |
| `uniform_like` | `(&tensor[D, p_float], p_float, p_float) -> tensor[D, p_float]` | Create a tensor matching the input shape and float dtype, filled by the deterministic affine sampler defined by [05-OP-8] under the active `with seed(...)` handler | Introduces `Random`. The template values are not observed; its adjoint is the zero cotangent. |
| `process_run` | `(String, List[String]) -> (Int64, String, String)` | Run an external program with the given argv and capture `(exit_code, stdout, stderr)`. Arguments are passed straight to the OS as argv (no shell, no interpolation), so a value in the args list cannot inject extra shell commands. A process killed by a signal reports exit code `-1`. | Introduces `IO`; it is outside AD. |

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

`sub` is a Tier 1 primitive governed by [05-OP-41], not a derived
`add`/`neg` composition. `div` and `neg` are likewise Tier 1 RISC primitives
(see §2.1 and §2.2), and `recip` is Tier 1 (§2.2). No Tier 2 arithmetic
operation introduces an intermediate numeric operation that the source
program did not request.

> **[05-OP-41]** `sub(left, right) -> result` admits two values of one active
> signed-integer or float dtype on the same scalar surface or on tensor
> surfaces with identical dimensions, returning that same surface,
> dimensions, and dtype. At a signed-integer width `w`, direct checked
> subtraction computes the exact mathematical difference and returns its
> signed `w`-bit representation when representable; otherwise it traps
> `Overflow` as operation `sub` at the operand dtype. It never lowers through
> `neg`, so an unrepresentable intermediate negation cannot replace the exact
> subtraction's own result or trap. On floats, subtraction executes at
> [04-NUM-8]'s declared arithmetic width and finalizes once to the operand
> storage dtype. The float adjoint is `(g, neg(g))` at that dtype.
> Signed-integer forms are forward-only and `grad` rejects them. The operation
> has no accumulator. `bool`, `string`, reserved dtype spellings, mixed dtypes
> or surfaces, and mismatched tensor dimensions are type errors.

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

> **[05-OP-27]** `or(left, right) -> result` has the signature, surface,
> shape, evaluation-order, rejection, purity, accumulator, and differentiation
> contract of [05-OP-26]. Its result is true exactly when either operand is
> true and is false otherwise, applied element-wise for tensors.

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
derivative of the lowering above for `sigmoid`, `tanh`, `silu`, and `gelu`;
`relu` instead carries its own adjoint under [05-OP-43] and survives AD as an
intact Tier-2 identity.

> **[05-OP-43]** `relu(x) -> result` admits every active float dtype on a
> scalar or tensor surface and returns that same surface, dimensions, and
> dtype. Its forward value is exactly the §3.3 lowering
> `max_elem(x, const(0.0))` under [05-OP-40], so a NaN input propagates with
> its exact stored payload and sign bits and `relu(-0.0)` returns `-0.0` by
> the first-operand equality rule. Its adjoint is its own, not [05-OP-40]'s:
> the input cotangent is `g` exactly where `cmplt(0, x)` is true and exact
> positive zero otherwise, including at `x = 0`, at both signed zeros, and at
> a NaN input. The `relu` identity remains intact through AD and every other
> semantic transform; only after its adjoint or zero rule has been applied
> may it decompose to the lowering, so the adjoint attaches to the identity
> rather than to `max_elem`'s tie rule. Non-float operands are type errors.
> The operation has no accumulator. The internal `ReluAdjoint` identity has
> this cotangent contract.

### 3.4 Higher-Level Operations

| Name | Lowering to RISC |
|---|---|
| `matmul(A, B)` | See §4.1 |
| `softmax(x, axis)` | See §4.2 |
| `linear(x, w, b)` | `add(matmul(x, w), b)` (with appropriate expand on b) |
| `cross_entropy(logits, labels)` | See §4.3 |

`min_elem` is a Tier 1 primitive governed by [05-OP-40], not a higher-level
arithmetic lowering.

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
| `gather(x, idx, axis)` | Direct `Gather` selection under §3.5.1 and [05-OP-52] |
| Window matrix extraction | The rank-generic pad/gather/reshape graph in §4.5; this is a lowering step, not a callable |
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
> `ScatterElements` SHALL take an index tensor of any active signed-integer dtype.
> The index is interpreted at its exact stored width; no index dtype is widened,
> narrowed, or otherwise converted. For the three scatter forms, target,
> updates, and output SHALL have identical precision. A non-integer index dtype
> or a precision mismatch is a type error, never an implicit cast. The public C
> gather and scatter callables in [05-OP-33] have this same complete index-dtype
> domain; they have no int32/int64-only exception.

`Scatter` and `ScatterAdd` are intentionally distinct primitives. Both
take inputs `(target, indices, updates)` with the same shape contract
(updates shape equals `target.dims[..axis] ++ indices.dims ++
target.dims[axis+1..]`) and the same precision constraints
(any one active signed-integer index dtype; target/updates/output precision
identical).
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

Precision constraints match the hyperplane scatters: `indices` has any one
active signed-integer dtype; `data`, `updates`, and `output` share one
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

> **[05-OP-66]** `indexed_tensor(arguments...) -> result` governs exactly
> `Gather`, `ScatterAdd`, `Scatter`, and `ScatterElements`, with §3.5's
> hyperplane or element-wise signatures, output dimensions, bounds, duplicate
> handling, and adjoint rules. Every positional axis is `int32` under
> [05-DIM-3]. Indices retain any active signed-integer dtype at its exact
> stored width under [05-SPARSE-1]; they are never silently widened or narrowed.
> Gathering and replacement admit every active tensor element dtype, including
> bool, and copy selected stored payload representations without conversion.
> `ScatterAdd` admits active signed-integer and float payloads; bool is a type
> error under [04-NUM-4]. Each result preserves the admitted payload dtype.
> Every `ScatterAdd` addition executes at [04-NUM-8]'s declared arithmetic
> width and finalizes to that payload's storage dtype; integer overflow follows
> [04-NUM-3/12]. It has no user-selected or implicit wider accumulator.
>
> On float payloads, gathering routes cotangents through `ScatterAdd` into
> an input-shaped zero base. `ScatterAdd` passes the cotangent to its base and
> routes update cotangents through `Gather`. Index and axis arguments have
> zero cotangent. Integer and bool payloads are forward-only; `Scatter` and
> `ScatterElements` retain §3.5's structural AD rejection. Wire axis parameters
> preserve these exact operation identities; serialization supplies no
> alternative operation, dtype, or accumulator rule.

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
| `tensor_scan` | `(initial: T, fn: (T, int64) -> T ! E, n: int64) -> tensor[n, ..state_shape(T), element(T)] ! E` | Iteratively apply `fn(prev, i)` for `i in 0..n` and stack the `n` resulting states along a new leading axis. |

`T` is either an active tensor-element scalar primitive or a tensor of any
rank at one active element dtype, including bool. `state_shape(T)` is empty
for a scalar and is the full tensor shape otherwise; `element(T)` is that
scalar dtype or tensor precision. These are signature relations, not new
builtins. The callback preserves the exact state type, dtype, rank, and
dimensions. The output is owned and contiguous, with shape
`[n] ++ state_shape(T)` and precision `element(T)`. Scalar and rank-zero
tensor states both yield rank-one output but remain distinct callback types.
The iteration order is the
positional integer sequence `0, 1, ..., n - 1`. Callback effects `E` occur
exactly once per iteration in that order; when `n = 0`, the result is empty
and the callback is not invoked, so no callback effect occurs. The empty
result retains every initial-state extent, including zero trailing extents;
no callback invocation is needed to discover its shape. Size and extent
arithmetic is checked int64 arithmetic. A zero-sized tensor state still
invokes the callback exactly n times.

The `tensor_scan` accumulator and emitted elements remain at `T` for every
step. Each callback result is finalized once at `T` before it becomes the
next accumulator and before it is stored in the output. Integer callbacks use
checked declared-width arithmetic and propagate their trap; float callbacks
use [04-NUM-8]'s arithmetic width and storage finalization. No scalar travels
through f64 merely because the helper executes in the host runtime. This is
the same exact tagged-carrier rule as [04-NUM-11] and [05-OP-31].

`tensor_scan` runs in constant stack space with respect to `n`. On float state elements,
reverse-mode differentiation is the reverse traversal of the exact executed
recurrence: cotangents from the returned elements and later recurrence states
combine at each callback invocation in reverse iteration order, using that
callback's ordinary adjoint. The integer and bool forms are forward-only.
`vmap` maps `initial` and every mapped callback capture pointwise while the
int64 iteration index and `n` remain shared; every mapped lane executes the
same positional iteration sequence.

**Negative parity for `tensor_scan`**: a non-callable second argument,
a wrong-arity call, a negative `n`, or a callback that returns a
different state type, dtype, rank, or shape are rejected with
`tensor_scan`-tagged diagnostics. A callback with an effect unavailable under
the enclosing handler is rejected by the ordinary effect rules; neither an
unreachable definition nor another definition's effects change this call's
legality.

> **[05-HOST-4]** `list_dir(path: string) -> List[string]` returns one element
> per directory entry, each the entry's own name rather than a path, and
> retains its declared `IO` effect. The current-directory and parent-directory
> links are not entries. The result is ordered by the byte sequence of the
> entry name the host reports, which for a name that is valid UTF-8 is
> lexicographically by Unicode scalar value. That order is fixed on the host's
> names before any conversion to `string`. Conversion SHALL be strict: a
> name not representable as UTF-8 fails the complete call with an `IO` trap
> tagged `list_dir`; no replacement characters, skipped entries, or partial
> successful list may substitute for that failure. The offending entry is
> the first invalid name in the host-name order above. Valid names retain
> their exact UTF-8 bytes, without Unicode normalization. A directory with
> no entries yields the empty `List`.
>
> The conversion diagnostic SHALL be
> `IO trap in list_dir: directory <directory>, entry <entry>: name is not valid UTF-8`.
> Both placeholders are reversible escaped host-byte representations,
> delimited by `b"` and `"`. Tab, carriage return, newline, backslash,
> single quote, and double quote use `\t`, `\r`, `\n`, `\\`, `\'`, and
> `\"`, respectively; other printable ASCII bytes are literal, and every
> remaining byte uses `\xhh` with two lowercase hexadecimal digits.
> On Unix the bytes are the host's filename bytes. String-valued path APIs
> cannot directly spell a name that is not representable as UTF-8; this
> contract does not introduce a byte-preserving path or listing API.
> The operation is outside AD: it has no adjoint, no cotangent, and no
> accumulator.

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
> | `tensor_scan` | `(T,((T,int64)->T!E),int64)->tensor[n,..state_shape(T),element(T)]!E` |
> | `process_run` | `(string,List[string])->(int64,string,string)!{IO}` |
> | `test_assert_eq` | `(Q,Q,string)->unit!{Test}` |
> | `test_assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |
> | `test_assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |
>
> Here `T` is a scalar or tensor state with the exact `state_shape(T)` and
> `element(T)` relations in section 3.6; its shape and dtype are invariant
> across every callback application. `Q` is one static type in
> [05-OP-36]'s scalar or recursive equality domain, `p_float` is one active
> float dtype, and `p` is one active tensor element dtype. Repeated variables
> denote the same type, dtype, rank, and dimensions. `tensor_scan` has
> [05-HOST-1]'s exact-width recurrence and adjoint; `process_run` has §2.6's
> argv, exit-code, capture, `IO`, and outside-AD contract; and the assertion
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
> without consuming it and returns `string`. It admits exactly an active
> numeric, `bool`, or `string` scalar; a tensor whose element dtype is one of
> the nine active tensor element dtypes in spec/04 §1.1; or a `List` whose
> reachable elements are recursively admitted by this rule. Unit, tuples,
> `Dict`, `Option`, ADTs, functions, resource handles, and
> deferred values are type errors. A `string`
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
> tensor truncation. Each nested List element uses this same rule. String
> elements are inserted verbatim, without quoting or
> escaping: this is a non-injective display form, not a serialization. Every
> lane produces byte-identical text for the same admitted stored value. The
> operation is pure, performs no arithmetic or dtype conversion, is
> non-differentiable (`grad` rejects it), and has no accumulator.

### 3.7 Host-Lane Data I/O Numeric Operations

The sole public JSON value family is `io/json::Json` and its `JsonNull`,
`JsonBool`, `JsonInt`, `JsonBigInt`, `JsonFloat`, `JsonString`, `JsonArray`,
and `JsonObject` constructors from [05-OP-34]. Its operations are the exact
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
ship. `io/json::Json`'s numeric capacity (`JsonInt(int64)` and
`JsonBigInt(string)` beside `JsonFloat(f64)`) is decided by [05-OP-2] and its
exact ADT identity by [05-OP-34].

#### Host-effect execution atom

> **[05-HOST-2]** JSON and CSV operations, `round_to`, and `process_run` are
> legal host-runtime operations in every language execution mode. Pure parsing,
> projection, serialization, and rounding retain their stated purity; file and
> process operations retain their declared `IO` effect and observable order.
> A compiled host execution SHALL produce the same typed result or language
> trap as evaluation. A device-only kernel may not perform `IO`, but that
> effect-boundary fact SHALL NOT be represented as a language-wide rejection,
> inert stub, default value, or evaluator-only signature.

*(Not fully implemented; chelis#1297 owns compiled host execution.)*

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
> | integer, bool, tensor | type error | n/a |
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
> value. An integer-form token outside int64 range SHALL ingest as
> `JsonBigInt` carrying the token's exact decimal spelling; ingestion never
> selects a lossy float image for an integer-form token.
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
> exactly and returns `None` for every other variant, including `JsonBigInt`.
> `io/json::json_bigint` returns a stored `JsonBigInt`'s exact decimal string
> and returns `None` for every other variant, including an in-range
> `JsonInt`; converting that string to a numeric dtype is a caller decision
> through an explicit parse, never an implicit widening. `io/json::json_float`
> returns a stored `JsonFloat` f64 exactly; on `JsonInt` it performs the named
> lossy int64-to-f64 widening (exact for magnitudes at or below 2^53), and on
> every other variant, including `JsonBigInt`, it returns `None`. It never
> truncates or rounds a float
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
> error naming an explicit checked cast. `JsonBigInt(value)` accepts exactly
> `string`; the canonical-form check is [05-OP-5]'s serialization rule, not a
> constructor special case. No construction path widens or narrows a
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
> JSON number token). A stored `JsonBigInt` emits its stored string verbatim
> as the number token after validating that it is the canonical integer form
> - an optional `-` followed by a nonzero leading digit and decimal digits -
> denoting a value outside int64 range; any other stored string is a loud
> serialization error, so a serialized document reparses to the identical
> variant structure. A non-finite `JsonFloat` is a loud serialization error.
> Equal documents serialize to identical bytes. `to_csv` accepts only the
> text-table type `List[Dict[string,string]]`; it applies the CSV quoting and
> row-order rules without inferring, preserving, or serializing a numeric cell
> type. Numeric source values enter CSV only through explicit `to_string`.
>
> `to_csv(table: List[Dict[string,string]]) -> string` returns the serialized
> text table. Text serialization is non-differentiable and outside AD; it
> has no accumulator.

#### Exact public scalar and container boundaries

> **[05-OP-31]** `scalar_carrier(value) -> result` governs exactly the ten
> final public C callables enumerated in the normative registry
> `spec/registry/c_scalar_carrier.md`, which this atom incorporates by
> reference. That registry is their canonical public
> identity; a differently named, typed-by-name, or raw-dtype successor is a
> different callable and has no authority from this atom.
>
> The final public declarations are exact:
>
> `typedef uint8_t chelis_dtype;`
>
> `enum { CHELIS_DTYPE_F32 = 0, CHELIS_DTYPE_F64 = 1, CHELIS_DTYPE_I32 = 2, CHELIS_DTYPE_BOOL = 3, CHELIS_DTYPE_I64 = 4, CHELIS_DTYPE_BF16 = 5, CHELIS_DTYPE_F16 = 6, CHELIS_DTYPE_I8 = 7, CHELIS_DTYPE_I16 = 8 };`
>
> `typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;`
>
> `typedef uint8_t chelis_value_tag;`
>
> `enum { CHELIS_VALUE_UNIT = 0, CHELIS_VALUE_SCALAR = 1, CHELIS_VALUE_STRING = 2, CHELIS_VALUE_TENSOR = 3, CHELIS_VALUE_LIST = 4, CHELIS_VALUE_TUPLE = 5, CHELIS_VALUE_DICT = 6, CHELIS_VALUE_ADT = 7, CHELIS_VALUE_OPTION = 8, CHELIS_VALUE_MAPPED_FILE = 9 };`
>
> `typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;`
>
> `typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;`
>
> `typedef struct { const void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_read_view;`
>
> `typedef struct { void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_write_view;`
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
> Every `reserved` byte is zero. An optional scalar result is one owned
> [05-OP-44] option node whose `Some` child is a validated scalar-tagged
> value; there is no by-value option carrier. A `chelis_value` tag is exactly
> one of the ten constants above. Unit has a null, otherwise-zero payload; Scalar
> embeds the complete canonical `chelis_scalar`; every heap tag carries a
> non-null owned handle in `payload.handle` and zero bytes in the remainder of
> the union, and that handle is exactly one [05-OP-44] owner. The natural C
> ABI of these fixed-width field declarations is the ABI; no feature macro,
> build mode, or typedef substitution may change their order, widths, or
> signedness. `chelis_tensor` is [05-OP-44]'s opaque descriptor handle and
> has no public field.
>
> Every public tensor descriptor observes as contiguous row-major storage
> through a view. It has rank in `0..=INT32_MAX` read through
> `chelis_tensor_rank` and exactly `rank` nonnegative int64 extents read
> through `chelis_tensor_shape`; a rank-zero descriptor has no extents. Its
> element count is the checked product of the extents, with the rank-zero
> empty product equal to one. There is no rank-eight limit. A read view or
> write view of a descriptor has `count` equal to that element count, `dtype`
> equal to the descriptor's validated dtype, and zero `reserved` bytes; its
> byte size is the checked product `count * chelis_dtype_size(dtype)`. A view
> with zero `count` has null `data`; a nonempty view has a non-null pointer
> aligned for its validated dtype. A read view is valid only while an owner
> of its descriptor is live and until that descriptor is passed to
> `chelis_tensor_begin_write`, whichever comes first. A successful begin
> invalidates every read view previously returned for that descriptor;
> dereferencing such a stale view violates the caller precondition. A write
> view is valid only while its exclusive guard is live; nothing is retained,
> released, or freed through a view pointer.
> An internal noncontiguous view is materialized before it crosses a public
> view. Foreign storage enters only through [05-OP-44]'s entry borrow, which
> validates the declared metadata and bounds but cannot prove a foreign
> allocation's lifetime or physical size.
> Each `chelis_dict_entry` contains two independently canonical
> `chelis_value` carriers in key-then-value order.
>
> `chelis_dtype_size` returns the exact byte width of the validated stored
> representation. Tensor extraction requires a rank-zero tensor with exactly
> one element. Fill requires the scalar dtype to equal the tensor dtype and
> writes the exact scalar bits to every element through [05-OP-44]'s
> exclusive write guard. Neither operation converts through `double` or
> another dtype. The family does not convert through `double` at any other
> edge. Rendering follows [05-OBS-1..2] at the
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
> callable identities enumerated in the normative registry
> `spec/registry/c_container_boundary.md`, which this atom incorporates by
> reference. The signature is part of each identity.
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
> absent key is unchanged, and `dict_get` returns one owned [05-OP-44] option
> node that is `None` only for absence.
> Recursive dictionary observation is canonical rather than insertion-ordered:
> bool keys order `false` before `true`, integer keys order by exact
> mathematical value within their one static key dtype, and string keys order
> lexicographically by Unicode scalar value. A dictionary has one static key
> type, so no cross-kind or cross-width ordering is defined.
> Recursive observation uses one byte grammar `R(value)` over every validated
> `chelis_value` variant. Unit renders as `()`. Scalars, strings, tensors, and
> Lists use [05-OP-25]'s exact scalar, raw-string, tensor, bracket, separator,
> traversal, and tensor-truncation rules, except that `R` applies recursively
> to each List element and therefore admits every variant defined here. A tuple
> renders as `()` when it has no fields, `(R(v),)` when it has one, and as `(`
> followed by its fields in index order separated by `, ` and then `)` when it
> has two or more. A dictionary renders entries in the canonical key order above as
> `{` followed by `R(key): R(value)` pairs separated by `, ` and then `}`;
> `{}` is empty. An ADT renders its exact stored constructor-name bytes followed by `(`,
> its fields in index order rendered by `R` and separated by `, `, and then
> `)`; a zero-field constructor therefore renders as `Ctor()`. An option node
> renders as `None` when it owns no child and otherwise as `Some(` followed by
> `R` of its child and `)`, the exact spelling of the ADT rule for those two
> constructors. A mapped file renders as `<mapped-file:` followed by its exact
> int64 byte length in decimal digits and then `>`. No structure
> inserts quoting or escaping. The grammar is deliberately non-injective:
> unit and an empty tuple both render `()`, and string bytes are unquoted.
> `read_bytes` and `mmap_read` return int64 elements in `0..=255`; mapped
> reads also require `offset + length` to lie within the mapped file.
> Each of `chelis_print_list`, `chelis_print_tuple`, `chelis_print_dict`, and
> `chelis_print_adt` writes exactly `R` of its argument followed by one byte
> `\n` to standard output. It adds no
> label, prefix, extra space, truncation beyond the nested tensor rule, or
> additional newline. A short or failed write traps `IO`; bytes the operating
> system accepted before that failure remain an ordinary prior `IO` effect.
> Successful return means every required byte was written. Recursive printing
> applies [05-OBS-1..5] at each stored scalar's own dtype and never widens a
> value for observation. These boundary operations are outside AD and have no
> accumulator.
>
> **[05-OP-33]** `runtime_tensor(value, parameters...) -> result` governs
> exactly the thirty-two final public C callable identities enumerated in
> the normative registry `spec/registry/c_tensor_runtime.md`, which this atom
> incorporates by reference. These
> signatures are canonical: unboxed axes and rank are `int32_t`; extents, sizes,
> offsets, counts, and element counts are `int64_t`; dtype arguments are
> `chelis_dtype`; tensor arguments and results are [05-OP-44]'s opaque
> `chelis_tensor` handles, and every tensor result is a new owner; and an
> untyped, string-mode, or dtype-named successor has no authority from this
> atom.
>
> This atom's selection rule also governs exactly the language builtin
> `where(condition, then, else)` with signature
> `(&tensor[D,bool], &tensor[D,p], &tensor[D,p]) -> tensor[D,p]` and its
> exact public C counterpart `chelis_tensor_where` in that registry. All four
> tensors have identical dimensions, both branches and the result have the
> same active element dtype `p`, and selection copies the chosen stored bits
> without numeric conversion. On float branches the adjoint routes each
> cotangent to the selected branch and exact zero to the other; the condition
> has no cotangent. Signed-integer and bool branches are forward-only. The
> operation has no accumulator.
>
> Every entry validates every observable input-descriptor invariant from
> [05-OP-31] and [05-OP-44], including dtype, shape, element count, capacity,
> alignment, live-owner state, and write-guard state, before reading data. The
> rank is nonnegative and representable as int32; every extent
> is nonnegative; a positive-rank shape pointer passed to allocation or entry
> borrow is non-null; and a nonempty entry borrow has a non-null,
> representation-aligned data pointer. Shape
> products, byte counts, offsets, output extents, and allocation sizes use
> checked arithmetic. A malformed carrier or invalid axis traps `Domain`; an
> unrepresentable count, extent, offset, or allocation size traps `Overflow`
> before allocation or element access. Each language operation follows its
> own axis atom: [05-AXIS-1] governs the reduction, `expand`, and `insert`
> family, while
> [05-OP-7]/[05-SHAPE-1] admits a computed int32 axis for `shape`. C-family
> unboxed axis parameters are runtime int32 values. Every signed axis accepted by this C family first
> applies §2.3's one-step negative normalization; an axis still out of range
> then traps `Domain`.
>
> `chelis_tensor_alloc_like` takes a validated input tensor and a canonical
> all-zero tagged scalar exemplar selecting any admitted output representation.
> It returns independent owned, contiguous, zero-filled storage with exactly the
> input shape. It checks output bytes and target allocation projection at that
> representation before allocation. It reads metadata only, so an active input
> write guard is allowed; it neither copies payload nor changes input ownership.
> Null input and malformed or nonzero exemplars trap `Domain`; unrepresentable
> output metadata traps `Overflow`. Numeric metadata failures name `alloc_like`
> at `int64` under [04-NUM-9]. Rank zero retains one element and any zero
> extent retains zero elements. This allocation operation has no cotangent or
> accumulator.
>
> Allocation returns owned, contiguous, row-major, zero-filled storage at the
> requested representation. A zero extent means zero elements, never one
> synthetic element. Foreign storage enters only through [05-OP-44]'s entry
> borrow; no callable in this family constructs a non-owning view, adopts
> caller bytes, or frees storage. `contiguous` preserves every element's
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
> `chelis_tensor_stride` projects the checked contiguous suffix stride at the
> normalized axis; `chelis_tensor_byte_count` projects the checked logical byte
> count at the tensor's declared representation, excluding spare storage capacity.
> Both return exact int64 metadata. These observations, like rank, shape, and
> element count, remain valid during an active write guard and access no elements.
> `chelis_tensor_check_reshape` takes rank and every target extent as exact tagged
> int64 scalars; the decoded rank must fit nonnegative int32. It validates the
> target rank, extents, contiguous
> suffix strides, element count, logical byte count, and target allocation domain,
> and requires the target count to equal the input count. It reads only metadata,
> is valid during an active write guard, and changes no metadata, ownership, or
> payload. Positive rank requires a non-null shape pointer. An empty target has
> zero elements but still requires representable suffix strides; rank zero has
> one element. Malformed metadata or unequal counts trap `Domain`; unrepresentable
> products, strides, byte counts, or allocation projections trap `Overflow`.
>
> `chelis_tensor_elementwise_index_step` validates input and domain tensor metadata
> and returns exact int64 zero for a rank-zero input, or one when the input shape
> is identical to the domain shape. Any other shape pairing traps `Domain`, even
> when the element counts agree or are zero. Input and domain dtypes may differ.
> `chelis_tensor_elementwise_index_step_for_shape` applies the same rule to an
> iteration domain supplied as exact tagged int64 rank and extents. The decoded
> rank must fit nonnegative int32; positive rank requires a non-null shape pointer.
> It validates every extent and the exact zero-aware element product before
> selecting the scalar or identity step. An iteration domain requires neither
> storage byte counts nor contiguous suffix strides. Negative rank or extents,
> malformed tagged scalars, and invalid handles trap `Domain`; unrepresentable
> rank, element count, or scratch allocation projections trap `Overflow`.
> Both step operations observe only checked metadata, remain valid during active
> write guards, and change no metadata, ownership, or payload. A caller validates
> the original input before repurposing its storage, uses the checked domain count
> as its loop bound, and maps a domain index `i` to the input index `i * step`.
> These internal scalar projections do not introduce language-level broadcasting.
>
> `chelis_tensor_unravel_index` converts an exact tagged int64 linear index into
> rank-many canonical int64 scalars in row-major axis order.
> `chelis_tensor_flat_index` converts rank-many exact tagged int64 coordinates
> into an exact int64 linear index. Both use the tensor's checked shape, count,
> and strides. The linear index must be in `[0, count)` and each coordinate in
> `[0, extent)`; an empty tensor admits no index, while rank zero admits exactly
> linear index zero and an empty coordinate tuple. Positive rank requires a
> non-null coordinate pointer to rank-many scalars, writable for unraveling.
> The caller supplies the complete array and preserves the tensor's borrowed
> storage; unraveling writes only the coordinate array. Invalid indices, null
> required pointers, and noncanonical or non-int64 scalar carriers trap `Domain`;
> unrepresentable coordinate-buffer projections or offsets trap `Overflow`.
>
> `chelis_tensor_check_permute` and `chelis_tensor_check_expand` validate a target
> shape supplied as exact tagged int64 rank and extents, including checked count,
> contiguous strides, logical bytes at the input representation, and target
> allocation projection. Rank must fit nonnegative int32; positive rank requires
> a non-null shape pointer. Permutation additionally takes rank-many exact tagged int64 axes
> (non-null at positive rank), normalizes each negative axis once, requires a
> bijection of the input axes, and requires each target extent to equal its
> selected input extent. Input and target ranks are equal. Expansion with equal
> ranks replaces a unit input axis; expansion with target rank one greater inserts
> an axis. The expansion axis normalizes against the target rank, its target
> extent is any nonnegative int64, and every other extent equals its corresponding
> input extent. No other rank relationship is admitted. Invalid axis, rank, or
> extent relationships trap `Domain`, including on empty targets; unrepresentable
> target metadata or scratch projections trap `Overflow`. Generated callers
> perform these checks before allocation or repurpose, using the same target shape
> at submission. These four operations access metadata only, remain valid during
> an active write guard, change no tensor metadata, payload, or ownership, and
> have no cotangent or accumulator. They preserve [05-MOV-1]'s movement semantics.
>
> `chelis_tensor_pad_shape`, `chelis_tensor_shrink_shape`, and
> `chelis_tensor_stride_shape` derive a complete checked target shape from the
> input metadata and rank-many exact tagged int64 bounds. The supplied tagged
> int64 rank must equal the input rank. Padding requires nonnegative before/after
> amounts and computes `input + before + after` with checked addition. Shrinking
> requires `0 <= start <= end <= input` and computes `end - start`; equal endpoints
> describe an empty axis. Striding requires positive steps and computes the exact
> ceiling quotient without an overflowing intermediate sum. Each validates the
> complete target count, suffix strides, representation bytes, and target allocation
> domain before writing rank-many canonical int64 scalars to the caller's output
> array. Positive rank requires non-null bound and output arrays. Rank zero admits
> empty arrays and retains one element. Empty targets do not bypass bound checks.
> Callers preserve operation extent claims and require the computed shape to equal
> the shape submitted for allocation or repurpose, before either occurs.
>
> `chelis_tensor_affine_index` takes rank-many exact tagged int64 coordinates,
> nonnegative offsets, and positive steps. It computes each coordinate as
> `coordinate * step + offset` with checked arithmetic, requires both the original
> coordinate to be nonnegative and the resulting coordinate to be inside the
> tensor's extent, and returns the checked row-major int64 index. Positive rank
> requires all three arrays; rank zero admits empty arrays and returns zero.
> Empty tensors admit no index. These four operations access metadata only, remain
> valid during a write guard, and change no tensor metadata, payload, or ownership.
> They have no cotangent or accumulator. Malformed carriers, pointers, ranks, bounds,
> or indices trap `Domain`; unrepresentable metadata or arithmetic traps `Overflow`.
> Numeric failures use [04-NUM-9]'s canonical line at `int64`, with operation `pad`,
> `shrink`, `stride`, or `affine_index`, respectively.
>
> `chelis_tensor_reduction_plan` snapshots checked tensor metadata;
> `chelis_shape_reduction_plan` constructs the corresponding metadata-only virtual
> domain from exact tagged int64 rank and extents. Both take a nonempty list of
> distinct original axes in strictly descending order after one-step negative
> normalization, an exact tagged scalar whose
> dtype selects the result representation, and a closed `chelis_reduction_op`
> diagnostic identity (`SUM`, `COUNT`, `MAX`, `MIN`, `PROD`, `ARGMAX`, or `ARGMIN`).
> The result shape removes the selected axes and checks count, suffix strides,
> representation bytes, and target allocation projection before returning an opaque
> independently owned plan. Virtual input domains owe checked counts but no storage
> bytes or unused storage strides. Positive array lengths require complete non-null
> arrays; scalar carriers must be canonical. `chelis_reduction_extent` observes a
> result axis and `chelis_reduction_count` observes the selected row-major leaf count.
> An empty result has no reachable group and returns leaf count zero without
> evaluating an irrelevant selected-axis product. An empty selected domain with a
> nonempty result has zero leaves. `chelis_reduction_index` maps checked result-group
> and leaf positions to the original row-major input index; either empty domain
> admits no index. `chelis_reduction_check_target` requires exact result shape,
> including every axis of an empty result. `chelis_reduction_check_scratch` checks
> leaf-count bytes and target projection at its exact tagged exemplar's dtype before
> scratch or result allocation. `chelis_reduction_plan_release` consumes the plan.
> The caller supplies a live plan of the correct kind, releases it exactly once,
> and permits concurrent observations only while it remains live. Plans borrow no
> tensor storage and perform no payload or ownership action on their input tensor;
> construction and observation remain valid during tensor write guards. Invalid
> domains trap `Domain`, unrepresentable arithmetic traps `Overflow`, and failures
> retain [04-NUM-9]'s canonical primitive identity at int64. These metadata operations
> have no cotangent or arithmetic accumulator and do not select a reduction algorithm.
>
> `chelis_tensor_sparse_plan` snapshots checked base, index, and (for scatter)
> update metadata before output allocation or reuse. Its exact tagged int64 axis
> normalizes once against the base rank. The closed `chelis_sparse_op` identifies
> gather, scatter-add, replace-scatter, or element-wise scatter; canonical numeric
> failure identities are respectively `gather`, `scatter`, `scatter_replace`, and
> `scatter_elements`. Index tensors have an active signed-integer dtype. Scatter
> updates have the base dtype and the exact section 3.5 shape, including every
> dimension of an empty tensor. Gather supplies no update tensor. Hyperplane
> iteration replaces the base axis with the complete index shape; element-wise
> iteration has the index shape and validates every non-scattered bound.
> Counts, strides, representation bytes, and target projection are checked before
> the independently owned opaque plan is returned.
>
> `chelis_sparse_extent` observes the result shape, using one-step axis
> normalization; `chelis_sparse_count` observes the checked iteration count.
> `chelis_sparse_index_slot` maps an in-range row-major iteration position to its
> index-tensor slot. `chelis_sparse_data_index` maps that position and its exact
> tagged int64 selected index to the base-tensor position, requiring the selected
> index in `[0, base.shape[axis])`. Empty iteration domains admit no index. The
> mapping preserves section 3.5's row-major update order and selects no arithmetic
> algorithm. `chelis_sparse_check_target` requires the exact result rank and shape,
> transported as canonical tagged int64 scalars, before allocation or reuse;
> positive array lengths require complete non-null arrays.
> `chelis_sparse_plan_release` consumes the live plan exactly once. Plans retain
> no tensor payload or ownership and remain valid after source release or repurpose;
> observations may run concurrently while the plan remains live. Malformed carriers
> and invalid domains trap `Domain`; unrepresentable metadata or offsets trap
> `Overflow`, retaining [04-NUM-9]'s canonical primitive identity at int64. These
> metadata operations have no cotangent or arithmetic accumulator.
>
> `chelis_tensor_matmul_plan` snapshots two checked row-major matrix operands
> and an exact tagged exemplar selecting the result representation. Both operands
> have rank at least two, equal leading batch shapes, equal payload dtype, and
> equal contraction extents. The result retains those batch axes followed by the
> left row and right column extents. Its count, strides, representation bytes, and
> allocation projection are checked before returning an independently owned opaque
> plan. Batch broadcasting is explicit and precedes this boundary.
> `chelis_matmul_extent` observes a result axis with one-step normalization;
> `chelis_matmul_dimension` observes the closed `ROWS`, `COLUMNS`, or `REDUCTION`
> dimension. `chelis_matmul_batch_count` gives the checked number of matrix groups,
> zero when the result is empty. `chelis_matmul_matrix_count` selects the checked
> per-matrix count for closed part `LEFT`, `RIGHT`, or `RESULT`.
> `chelis_matmul_index` maps an in-range batch and per-matrix element to that
> part's checked complete tensor index. Empty domains admit no index.
>
> `chelis_matmul_check_target` requires the exact tagged int64 result rank and
> shape before allocation or reuse; positive array lengths require complete non-null
> arrays. `chelis_matmul_check_scratch` checks a selected matrix count's bytes and
> allocation projection at an exact tagged exemplar's representation.
> `chelis_matmul_check_vendor` checks all dimensions against a positive exact tagged
> int64 maximum supplied from the selected vendor argument type before conversion.
> A computation with no matrix calls (empty output or zero contraction extent)
> requires no vendor projection. Empty outputs owe no unused per-matrix products
> or scratch. A zero contraction with a nonempty result retains its zero identity.
> These metadata utilities select no arithmetic algorithm and confer no authority
> to substitute vendor GEMM for section 4.1's exact primitive contraction.
> `chelis_matmul_plan_release` consumes the live plan exactly once. Plans retain
> no tensor storage and remain valid after source release or repurpose, including
> concurrent observations while live. Invalid carriers, shapes, selectors, or
> indices trap `Domain`; unrepresentable arithmetic or target projections trap
> `Overflow`, with [04-NUM-9]'s canonical `matmul` identity at int64. There is no
> cotangent or arithmetic accumulator for these metadata operations.
>
> `chelis_tensor_check_literal` validates a complete result shape and literal
> element count, both carried as exact tagged int64 values, against a canonical
> zero exemplar of the literal's element dtype. Shape rank, extents, strides,
> count, representation bytes, and target allocation projection must be valid;
> the literal count must be nonnegative and equal the checked result element
> count. Rank zero requires one literal element; an empty shape domain requires
> zero. Positive rank requires a complete non-null shape array. Validation occurs
> before destination allocation or reuse and before reading the literal buffer.
> All active storage dtypes are admitted without payload conversion. Malformed
> metadata or count mismatch traps `Domain`; unrepresentable metadata traps
> `Overflow`, with canonical `const` identity at int64. This metadata check has
> no cotangent or accumulator.
>
> `chelis_tensor_write_literal` takes a live tensor write guard, an exact tagged
> int64 count, and that many complete, stable, nonoverlapping `chelis_scalar`
> values. Zero count permits a null value pointer. The count must equal the
> destination's checked element count. Every source carrier must be canonical and
> have the destination dtype; all carriers and the complete source array's target
> byte projection are validated before the first destination write. Each stored
> element bit is preserved at its declared representation width, with no dtype
> conversion or arithmetic accumulator. Metadata or carrier mismatch traps
> `Domain`, and an unrepresentable source array traps `Overflow`, with canonical
> `const` identity at int64. The write guard remains live and exclusively owns the
> destination throughout; the operation neither ends it nor retains the source.
> This constant-storage ingress has no cotangent.
>
> `chelis_tensor_permute_plan`, `chelis_tensor_expand_plan`, and
> `chelis_tensor_affine_plan` snapshot complete source and result metadata for
> the exact row-major movements in [05-MOV-1], independently of tensor payload.
> Permutation takes a tagged int64 rank and complete array of tagged int64 axes;
> it requires a bijection after one-step negative axis normalization. Expansion
> takes a tagged int64 axis and extent plus the closed `EXPAND` or `INSERT` form:
> `EXPAND` replaces a unit axis without changing rank, and `INSERT` adds an axis.
> Negative axes normalize once against the result rank. Affine movement takes a
> tagged int64 rank and complete tagged int64 bound arrays with the closed `PAD`,
> `SHRINK`, or `STRIDE` form. Padding's arrays are before/after; shrinking's are
> start/end; stride's first array is positive steps and its second pointer is
> unused. Rank must equal source rank. All positive array lengths require complete
> non-null arrays. The same checked shape, bound, dtype, count, stride, byte, and
> target projection rules apply as the corresponding movement checks above.
> The opaque plan validates its own projection storage before allocation.
>
> `chelis_movement_extent` observes the closed `SOURCE` or `RESULT` shape at a
> tagged int64 axis with one-step negative normalization. `chelis_movement_count`
> returns the source count for padding and result count for other forms.
> `chelis_movement_index` maps an in-range tagged int64 position in that domain
> to the padded destination index or other forms' source index. Each projection
> uses checked exact coordinate/stride arithmetic without per-index scratch;
> empty domains admit no index. `chelis_movement_check_target` requires a complete
> tagged result rank and shape to match the plan before allocation or reuse.
> `chelis_movement_plan_release` consumes the live plan exactly once. The plan
> retains no tensor payload or ownership, remains valid during source writes and
> after source release or repurpose, and admits concurrent observations while live.
> Malformed carriers, selectors or domains trap `Domain`; unrepresentable metadata
> or projections trap `Overflow`, retaining the selected canonical `permute`,
> `expand`, `insert`, `pad`, `shrink`, or `stride` identity at int64. Invalid
> operation selectors and null plans use `movement`. These metadata operations have no
> cotangent or arithmetic accumulator and do not alter payload arithmetic.
>
> `chelis_tensor_window_plan` snapshots the complete input and valid-padding
> result metadata from [05-RWIN-1]'s positive trailing window and stride lists,
> transported as equal-length arrays of exact tagged int64 values. The tagged
> length is nonzero and does not exceed input rank; each input spatial extent
> admits its window. Positive array lengths require complete non-null arrays.
> Leading extents pass through unchanged, including zero extents. The independently
> owned opaque plan checks result extents, rank, count, strides, bytes, and target
> projection before return. It retains no source payload or ownership and remains
> observable during a source write guard and after source release or repurpose.
>
> `chelis_window_extent` observes an axis of the closed `SOURCE` or `RESULT`
> shape, with one-step negative normalization. `chelis_window_count` returns the
> checked row-major window leaf count; an empty result returns zero without
> evaluating an unused window product. `chelis_window_index` maps an in-range
> result position and window leaf to the original input index, using checked
> coordinate/stride arithmetic; either empty domain admits no index.
> `chelis_window_check_tensor` requires a tensor's exact shape and dtype to match
> the selected side. `chelis_window_check_target` requires a submitted rank and
> complete shape, transported as canonical tagged int64 scalars, to match that
> side before allocation or reuse. `chelis_window_plan_release` consumes the live
> plan once; concurrent observations require the plan to remain live.
>
> The closed `chelis_window_op` selects the canonical diagnostic identity
> `reduce_window_sum`, `reduce_window_mean`, `reduce_window_max`,
> `reduce_window_min`, or `reduce_window_grad`. Invalid domains and malformed
> carriers trap `Domain`; unrepresentable metadata or offsets trap `Overflow`,
> using [04-NUM-9]'s selected identity at int64. These metadata operations have
> no cotangent or arithmetic accumulator and do not select or alter OP39's
> arithmetic algorithm or dtype admission. An invalid operation selector or null
> plan traps `Domain` with identity `reduce_window`.
>
> `chelis_tensor_reshape` accepts a live, flat `List<int64>` of target extents
> and an idle tensor of any active element dtype. It applies the same checked
> reshape validation before allocating result storage, returns an independent
> owner with the target shape, and preserves every stored element bit in row-major
> order. The input is unchanged; no shape-list element is inferred or converted.
> On floats the adjoint reshapes the cotangent to the input shape; integer and bool
> forms are forward-only. Metadata observations and validation have no cotangent,
> shape arguments have no cotangent, and none of these operations has an accumulator.
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
> and follows [05-SPARSE-1] plus §3.5's row-major last-write-wins rule; it
> structurally rejects `grad`. Add starts each
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
> **[05-OP-44]** `heap_lifetime(handle, parameters...) -> result` governs
> exactly the heap-handle, strong-owner, tagged-value conversion, option-node,
> entry-borrow, and guarded-access callable identities enumerated in the
> normative registry `spec/registry/c_heap_lifetime.md`, which this atom
> incorporates by reference. The signature is part of each identity; a
> differently named, unguarded, field-reading, or ownership-ambiguous
> successor has no authority from this atom. (The opaque carrier and the
> lifetime callables are not yet fully implemented; see chelis#1286.)
>
> The final public declarations are exact:
>
> `typedef struct { void *handle; } chelis_string;`
>
> `typedef struct chelis_tensor chelis_tensor;`
>
> `typedef struct chelis_tensor_write chelis_tensor_write;`
>
> `typedef struct chelis_list chelis_list;`
>
> `typedef struct chelis_tuple chelis_tuple;`
>
> `typedef struct chelis_dict chelis_dict;`
>
> `typedef struct chelis_adt chelis_adt;`
>
> `typedef struct chelis_option chelis_option;`
>
> `typedef struct chelis_mapped_file chelis_mapped_file;`
>
> The heap-kind universe is closed and exact: `String`, `Tensor`,
> `TensorStorage`, `List`, `Tuple`, `Dict`, `Adt`, `Option`, and `MappedFile`.
> Every heap allocation carries exactly one of those kinds and exactly one
> strong-owner count, and every kind has exactly one finalizer.
> `TensorStorage` is private: it is reached only through a `Tensor` descriptor
> and has no public handle, value tag, or callable. Each of the eight public
> kinds has exactly one public ownership carrier declared above, exactly one
> `chelis_value` tag (`CHELIS_VALUE_STRING`, `CHELIS_VALUE_TENSOR`,
> `CHELIS_VALUE_LIST`, `CHELIS_VALUE_TUPLE`, `CHELIS_VALUE_DICT`,
> `CHELIS_VALUE_ADT`, `CHELIS_VALUE_OPTION`, and `CHELIS_VALUE_MAPPED_FILE`),
> one retain callable, one release callable, one take conversion into a
> value, one take conversion out of a value, and one borrow conversion out of
> a value. The registry enumerates exactly those identities plus the option,
> entry-borrow, view, and guard callables below; there is no kind-generic
> handle, untagged payload, owner flag, or second representation of any kind.
> Tensor, List, tuple, dictionary, ADT, option, and mapped-file carriers are
> pointers to incomplete C types. `chelis_string` is the one fixed by-value
> wrapper; its `handle` field preserves that wrapper's ABI, while the object it
> points to is opaque and callers never read, compare, or write through it.
>
> A live handle is one logical owner under [04-LIN-3]. Retain creates one
> additional owner by a checked relaxed increment; a count that would exceed
> the representable owner range traps `Overflow`. Release consumes one owner;
> the final release synchronizes with release/acquire ordering before running
> the kind's finalizer exactly once, and the finalizer releases each stored
> child exactly once before freeing its own allocation once. Every handle or
> guard argument must be backed by the live owner that the call consumes or
> borrows. A retain, release, take, borrow, view, or guard operation on a null
> handle, or on a live handle whose kind disagrees with the callable or with
> the value tag, traps `Domain`; no operation repairs, ignores, or reinterprets
> it. A take, final release, or guard end consumes that caller-held handle or
> guard. Reusing its stale pointer afterward violates the live-handle
> precondition; because the allocation may already be freed, the ABI does not
> promise a diagnostic or retain a tombstone to recognize that invalid C use.
>
> Every heap-tagged `chelis_value` holds exactly one owner of its handle;
> there is no non-owning value. `chelis_value_clone` creates one additional
> owner for a heap tag and copies a unit or scalar payload;
> `chelis_value_release` consumes a heap tag's owner and consumes nothing for
> unit or scalar. A take conversion into a value moves the caller's owner into
> the value. A take conversion out of a value moves the value's owner to the
> returned handle and ends the value. A borrow conversion out of a value
> returns the handle without creating or consuming an owner and is valid only
> while an owner of that value is live. A conversion whose tag does not match
> its named kind traps `Domain`.
>
> Strings, option nodes, Lists, tuples, dictionaries, ADTs, and mapped files
> are immutable after construction. A constructor clones each borrowed child
> exactly once; a by-value accessor, including `chelis_option_unwrap` and
> [05-OP-32]'s index, field, and lookup callables, clones the stored child
> exactly once; a finalizer releases each stored child exactly once. Because a
> stored child set never changes, no value contains itself and no heap graph
> has a cycle, so strong ownership is complete and no collector exists.
>
> Every target-representable option, including an option of a scalar, of a
> mapped file, or of another option, is one option node: `chelis_option_none`
> owns no child and `chelis_option_some` owns exactly one tagged child.
> `chelis_option_is_some` reads the discriminant, and `chelis_option_unwrap`
> of a `None` node traps `Domain`. There is no by-value,
> discriminant-plus-payload, or scalar-special option carrier. A mapped file is
> a resource whose
> bytes are read only through [05-OP-32]'s mapped-read callables;
> `chelis_mapped_file_retain` and `chelis_mapped_file_release` follow the
> owner rule above, and `CHELIS_VALUE_MAPPED_FILE` is its only tagged
> representation.
>
> A tensor handle is a descriptor that retains exactly one storage allocation
> for its whole lifetime; a descriptor's finalizer releases its storage once,
> and the storage finalizer frees the bytes once. Descriptors that share one
> storage are views. `chelis_tensor_retain` and `chelis_tensor_release` are the
> only public tensor lifetime operations, and no public callable frees,
> adopts, or transfers storage bytes. `chelis_tensor_read_view` returns
> [05-OP-31]'s read view of a descriptor's contiguous row-major elements with
> the owner-and-write-begin validity bound defined there.
> `chelis_tensor_begin_write` succeeds only when the descriptor has exactly
> one live owner, its storage has exactly one live descriptor, the storage is
> runtime-owned, and no guard is active on it; it returns the one exclusive
> non-owning guard embedded in that descriptor. A successful begin invalidates
> every read view previously returned for that descriptor before it activates
> the guard. Dereferencing one afterward violates the caller precondition; the
> runtime does not promise to diagnose that stale pointer. The guard borrows,
> but neither consumes nor clones, the descriptor's existing owner for the
> guard lifetime; it allocates no guard object. Every other begin, read view,
> retain, clone, or release of that descriptor traps `Domain` until
> `chelis_tensor_end_write` consumes and deactivates the guard without freeing
> an allocation or consuming the descriptor owner. `chelis_tensor_write_view` borrows its `const` guard and
> is valid only while that guard is live; an ended guard has no view. Fill
> under [05-OP-31] and every other public
> element mutation takes the guard, never the descriptor. The runtime performs these
> checks itself on the live counts and write state; a compiler's reuse proof
> never replaces them.
>
> `chelis_tensor_repurpose` is the one descriptor-metadata mutation. Its rank
> and each shape extent arrive as exact `int64` `chelis_scalar` values; any
> other scalar dtype or a nonzero reserved byte traps `Domain`. It
> succeeds only when the descriptor has exactly one live owner, its storage
> has exactly one live descriptor, the storage is runtime-owned, and no write
> guard is active. It validates `rank` and `shape` by the same rules as
> `chelis_alloc`, using the descriptor's existing dtype, and requires the new
> checked byte size to equal the storage allocation's byte capacity exactly.
> It then replaces the rank, shape, canonical row-major strides, and element
> count while preserving the descriptor, storage, dtype, stored bits, and all
> owner counts. Success invalidates every earlier read view for the descriptor;
> a failed uniqueness, provenance, active-guard, or exact-capacity condition
> traps `Domain`, and invalid metadata arithmetic traps `Overflow`, without
> adopting, freeing, transferring, or reallocating storage bytes. This
> operation is outside AD and has no adjoint or accumulator.
>
> An entry borrow, following [04-LIN-7], is a descriptor over storage the
> caller owns: `chelis_tensor_entry_borrow` validates the declared rank,
> extents, dtype, alignment, and capacity exactly as [05-OP-33] validates an
> owned allocation, admits a null `data` pointer only for a zero element
> count, requires a nonnegative `byte_capacity` at least the checked byte size,
> and produces storage that is never runtime-owned. Releasing it never
> releases the caller's bytes; `chelis_tensor_begin_write` on it traps
> `Domain`; and no address comparison, retain count, or later invocation makes
> that storage runtime-owned. An owned result crossing that boundary always
> has runtime-owned storage. The runtime validates the declared metadata and
> bounds but cannot prove a foreign allocation's lifetime or physical size;
> those remain the caller's preconditions.
>
> This family carries no numeric payload of its own: a read or write view
> exposes the exact stored bits at the descriptor's validated dtype and never
> converts, and `chelis_tensor_entry_borrow` accepts only a validated
> `chelis_dtype`. It has no alias, wrapper, deprecated spelling, field-level
> access, owner flag, or free-style path; it has no accumulator and is outside
> AD.
>
> **[05-OP-34]** `numeric_adt(fields...) -> value` governs exactly the five
> exported stdlib ADT identities enumerated in the normative registry
> `spec/registry/stdlib_adt_identities.md`, which this atom incorporates by
> reference, and no structurally similar successor.
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
> integer/float source distinction. `JsonBigInt` carries the exact decimal
> spelling of an integer-form source token outside int64 range ([05-OP-2]);
> it is source-faithful text, never a float funnel, and its string field
> compares and renders byte-exactly. Decimal, date, duration, vocabulary,
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
> the eighty-four final exported stdlib numeric definitions enumerated in the
> normative registry `spec/registry/stdlib_numeric_manifest.md`, which this
> atom incorporates by reference. A
> signature and effect set are part of the identity. Only the exact registry
> identities exist: no effectless, wildcard-result, or otherwise weakened alias
> is part of the language.
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
> integer-form token outside int64 ingests as `JsonBigInt` and never becomes
> `JsonFloat`; `json_int`
> refuses `JsonFloat` and `JsonBigInt`, `json_bigint` is [05-OP-3]'s exact
> big-integer projection, while `json_float` performs [05-OP-3]'s named
> int64-to-f64 widening, refuses `JsonBigInt`, and returns a stored f64
> unchanged. Index wrappers
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
> `json_bigint`, `json_float`, `json_bool`, `json_array`, and `json_object`
> returns `Some`
> only for its named variant, except [05-OP-3]'s explicit int64-to-f64 case.
> `json_is_null` is true exactly for `Some(JsonNull)`. Every other shape,
> including `None`, returns `None` or false without fabricating a payload.
> Serialization follows [05-OP-5], uses compact JSON punctuation, and escapes
> every required control. Object serialization orders members by increasing
> Unicode scalar-value key sequence, recursively, so equal documents have
> identical bytes. `try_to_json` returns `None` exactly when a reachable float
> is non-finite or a reachable `JsonBigInt` violates [05-OP-5]'s canonical
> out-of-range integer form; `to_json` traps `Domain` for the same documents.
> `write_json` and `try_write_json`
> validate the complete document before opening or truncating the destination.
> The try form returns `None` for that same invalid content; both forms
> propagate a
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
> IO and process functions introduce their registry-declared `IO` effect.
> Process calls
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
> inherit the current working directory and environment. They are `IO`
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

### 3.9 Differentiation Barrier

`stop_gradient` is a Tier-2 identity whose entire meaning lives at the
transform layer: forward it is the identity, and under `grad` it is the
boundary the transform does not cross.

> **[05-OP-42]** `stop_gradient(value) -> result` admits exactly one value of
> any checked type and returns that value unchanged: the same type,
> dimensions, stored bits, and ownership. The argument expression evaluates
> exactly once, in its ordinary order, with its ordinary effects and traps;
> the operation itself is pure and adds none. Under differentiation the
> operation is a barrier: the transform SHALL NOT traverse the argument's
> subgraph for adjoint construction or for structural differentiability
> analysis, so an operation whose atom structurally rejects `grad` does not
> reject a graph that reaches it only through this barrier. The operation's
> cotangent contract is spec/06 §2.1's shape-preserving exact zero for its
> argument. Nested differentiation treats the barrier identically at every
> order, and `vmap` maps the identity pointwise. The identity remains intact
> through every semantic transform and only then erases to its argument. It
> is a named operation, never a mode, annotation, or effect, and it has no
> accumulator.

The barrier is what makes gradient surgery expressible while the
structural-rejection discipline stays intact: the straight-through estimator
`add(x, stop_gradient(sub(round(x), x)))` differentiates as the identity
path even though bare `round` under `grad` remains a structural
`AdRejectionReason::PiecewiseConstant` rejection.

*(Not fully implemented; chelis#1312.)*

---

### 3.10 Exact builtin operation contracts

#### Exact arithmetic

> **[05-OP-64]** Signature: `add(x,y)`, `mul(x,y)`, `div(x,y)`, `floor_div(x,y)`,
> `trunc_div(x,y)`, and `mod(x,y)` take two same-dtype numeric scalars or
> two same-shaped, same-dtype tensors and return that surface and dtype.
>
> Domain: `add` and `mul` admit all active signed integers and floats; `div`
> admits floats only; `floor_div` admits signed integers and floats;
> `trunc_div` and `mod` admit signed integers only. Bool, string, mixed
> dtypes, implicit broadcasting, and reserved dtypes are type errors.
> Arithmetic and storage finalization use [04-NUM-8]'s declared widths.
>
> Result: Addition and multiplication compute their ordinary arithmetic
> result. Float division is IEEE division, including signed zero,
> infinities, and NaNs. Integer floor division rounds the exact quotient
> toward negative infinity; truncating division rounds toward zero. Integer
> `mod` has the dividend's sign and equals `x - trunc_div(x,y)*y`
> mathematically, without introducing intermediate overflow. Float floor
> division is the own-width IEEE quotient followed by floor.
>
> Failure: Integer zero divisors trap Domain. Unrepresentable signed
> arithmetic and signed minimum divided by -1 trap Overflow. Float
> exceptional results follow [04-NUM-2]; no integer computation passes
> through a float. Static failures are diagnosed when concrete, and runtime
> failures use [04-NUM-9].
>
> Adjoint: For floats, add sends `(g,g)`, multiply sends `(g*y,g*x)`, and
> divide sends `(g/y,-g*(x/y)/y)`, evaluated at the declared width. Integer
> operations and the piecewise-constant floor, truncation, and remainder
> operations structurally reject differentiation.
>
> Accumulator: None. Each primitive finalizes its own result; an algebraic
> rewrite may not introduce another trap or change a float operation order.

#### Unary arithmetic

> **[05-OP-46]** Signature: `neg(x)`, `recip(x)`, `exp(x)`, `log(x)`, `sin(x)`, `sqrt(x)`,
> `cos(x)`, `tan(x)`, `atan(x)`, `abs(x)`, `floor(x)`, `ceil(x)`, and
> `round(x)` preserve the scalar or tensor shape and dtype of one operand.
>
> Domain: Negation, absolute value, floor, ceil, and round admit active
> signed integers and floats. The other operations admit active floats only.
> Their computation widths and storage rounding are [04-NUM-8]'s; bool,
> strings, reserved dtypes, and integer transcendental operands are type
> errors.
>
> Result: Each operation computes its named mathematical operation under the
> IEEE exceptional-value and finalization rules. Reciprocal is direct
> division of same-dtype one by x. On floats, round selects the nearest
> integer with ties to even; floor and ceil round toward negative and
> positive infinity. Floor, ceil, and round are exact identities on
> integers. Signed-zero and non-finite behavior follow the primitive's IEEE
> operation, without a lossy ingress conversion.
>
> Failure: Negating or taking absolute value of the signed minimum traps
> Overflow. Float domain and exceptional values follow [04-NUM-2], with no
> host-language panic substituted for a numeric trap.
>
> Adjoint: For differentiable float inputs: neg gives -g; reciprocal gives
> -g*y*y using the forward y=1/x; exp gives g*exp(x); log gives g/x; sin
> gives g*cos(x); sqrt gives g/(2*sqrt(x)); cos gives -g*sin(x); tan gives
> g/(cos(x)*cos(x)); atan gives g/(1+x*x); abs gives g*sign(x), where
> sign(x)=(x>0)-(x<0), including zero for x=0 and NaN. Integer
> differentiated inputs and float floor/ceil/round structurally reject
> differentiation; integer rounding identities may be erased before AD.
> Constants and intermediate arithmetic remain at the operand dtype.
>
> Accumulator: None; no operation widens through an unrelated dtype.

#### Bitwise arithmetic

> **[05-OP-47]** Signature: `bitand(x,y)`, `bitor(x,y)`, `bitxor(x,y)`, `shl(x,y)`, and
> `shr(x,y)` take two same-dtype signed-integer scalars or same-shaped
> tensors.
>
> Domain: Only the active signed-integer widths are admitted. Bool, float,
> string, mixed precision, and shape broadcasting are type errors. Shifts
> use the right operand as the shift count at its declared width.
>
> Result: The result has the input dtype and shape. And/or/xor operate on
> the exact w-bit two's-complement representation. Left shift discards bits
> beyond w; right shift is arithmetic, extending the sign bit. As
> [04-NUM-13] requires, nonnegative counts at or above w yield zero for
> `shl`; `shr` yields zero for nonnegative x and -1 for negative x. Counts
> are never implicitly masked.
>
> Failure: A negative shift count traps with
> `shift amount must be non-negative, got N` per [04-NUM-13]. Bitwise results
> preserve exactly w bits and do not trap merely because their signed
> interpretation differs from an unbounded arithmetic result.
>
> Adjoint: These discrete operations structurally reject differentiation.
>
> Accumulator: None.

#### Activation compositions

> **[05-OP-48]** Signature: `sigmoid(x)`, `tanh(x)`, `silu(x)`, and `gelu(x)` preserve one
> float scalar or tensor's shape and dtype; `softmax(x,axis)` takes a float
> tensor and an axis-domain int32 and returns the same tensor type.
>
> Domain: All active float dtypes are admitted at [04-NUM-8]'s widths.
> Integer/bool/string operands are type errors. Softmax's axis obeys
> [05-DIM-3], including negative-axis normalization and runtime validation.
>
> Result: The pointwise lowerings are the formulas in section 3.3: sigmoid
> is 1/(1+exp(-x)), tanh is the hyperbolic tangent, silu is x*sigmoid(x),
> and gelu is the stated tanh approximation. Softmax uses section 4.2's
> max-shifted exponentials divided by their axis sum. Constants, each
> primitive intermediate, and results retain the operand dtype; the formulas
> do not license an f64 evaluation funnel.
>
> Failure: Invalid axes and shape obligations fail loudly. IEEE exceptional
> values propagate through the stated primitive graph; no clipping, default
> distribution, or hidden epsilon repairs a non-finite input.
>
> Adjoint: Differentiate the exact stated composition at its own width.
> Softmax gives y*(g-sum(g*y,axis)), preserving the input shape;
> contributions use the owning reduction's order. No zero-adjoint shortcut
> is allowed.
>
> Accumulator: Softmax reductions use [05-OP-30]'s declared accumulator and
> canonical balanced tree. Pointwise activations have no accumulator.

#### Movement identities

> **[05-OP-49]** Signature: `reshape(x,shape)`, `permute(x,axes)`, `expand(x,axis,size)`,
> `insert(x,axis,size)`, `pad(x,padding)`, `shrink(x,bounds)`, and
> `stride(x,steps)` have section 2.4's tensor movement signatures, including
> its named-axis and anchored insert forms.
>
> Domain: Every active tensor element dtype is admitted without conversion.
> Axis positions/permutations are int32; extents, bounds, padding, and steps
> are int64. Runtime arguments and named dimensions obey [05-DIM-1..3] and
> spec/04 section 4.7; kernel capacity cannot narrow these domains.
>
> Result: Reshape preserves row-major element sequence and total element
> count. Permute reorders axes. Expand repeats a size-one axis; insert
> creates and repeats a new axis. Pad inserts dtype-exact zero cells, shrink
> selects the stated half-open bounds, and stride follows [05-MOV-1]'s
> signed sampling map. All copied elements retain exact stored bits,
> including NaN payloads and signed zeros.
>
> Failure: Invalid axes, inconsistent element counts, illegal
> extent/bound/step values, non-unit expand sources, and malformed runtime
> extent forms fail the exact guards in section 2.4 and spec/04 section 4.7.
> Extent arithmetic overflows trap Overflow. No clipped shape or default
> dimension is substituted.
>
> Adjoint: For float elements: reshape restores the original shape, permute
> applies the inverse permutation, expand sums the repeated axis and
> restores its size-one slot, insert sums the inserted axis, pad shrinks to
> the source, shrink pads exact zeros, and stride scatters to its original
> sampling positions. Shape arguments have zero cotangent. Integer/bool data
> follows spec/06's structural rejection rules.
>
> Accumulator: Forward movement has no numeric accumulator. Repeated float
> cotangents use [05-OP-30]'s canonical reduction contract.

#### Shape and scalar boundaries

> **[05-OP-50]** Signature: `rank(x)` and `numel(x)` borrow a tensor and return int32 and
> int64 respectively; `scalar_to_tensor(x)` maps one active tensor-element
> scalar to a rank-zero tensor; `tensor_to_scalar(x)` borrows a rank-zero
> tensor and returns its element scalar.
>
> Domain: All active tensor element dtypes are admitted. No conversion
> crosses either scalar/tensor boundary. Tensor-to-scalar requires rank
> zero, rather than an arbitrary one-element shape. Rank and element-count
> observations do not inspect element values.
>
> Result: Rank is the number of dimensions and numel is their exact
> mathematical product, with rank-zero product one. The boundary conversions
> preserve the dtype and stored bits exactly.
>
> Failure: An unrepresentable int32 rank or int64 element count traps
> Overflow. Non-tensor observations, string scalar construction, and
> nonzero-rank extraction are type errors; there is no rank-erasing
> fallback.
>
> Adjoint: Shape observations have zero cotangent. Float scalar/tensor
> conversion has the inverse shape conversion as adjoint; integer/bool
> differentiated data is structurally rejected.
>
> Accumulator: Only exact checked int64 metadata multiplication for numel; no floating accumulator.

#### Contractions and normalization graphs

> **[05-OP-51]** Signature: `matmul(a,b)` uses section 4.1's batched matrix signature;
> `einsum(equation,a,b)` takes a string equation and two tensors;
> `conv(input,kernel,strides:List[int64],padding:List[(int64,int64)])` uses section 4.5's
> layout; `layer_norm(x,gamma,beta,epsilon:p)` uses section 4.4's trailing-axis
> normalization and affine parameters.
>
> Domain: Matmul, conv, and layer_norm operands share one active float
> dtype; einsum admits one active signed-integer or float dtype, with its
> exact equation grammar, result dtype, accumulator, and contraction graph
> from [05-OP-33]. All contracted extents, batch dimensions, layouts, and
> affine shapes satisfy their exact section 4 rules. Contraction dimensions
> are equal, not broadcast by convenience. Extent/axis parameters retain
> [05-DIM-3]'s kinds.
>
> Result: The output shape and primitive graph are the section 4
> definitions. Layer normalization's epsilon is an explicit scalar of the
> operand dtype, consumed at its stored value without an implicit default;
> the trailing hidden extent is positive. Conv computes cross-correlation
> without flipping the kernel, for every positive spatial rank r. Input and
> kernel have rank r+2; the first two axes are batch/input-channel and
> output-channel/input-channel respectively. Spatial axis j in the input
> corresponds explicitly to spatial axis j in the kernel and result.
> Strides and padding each contain exactly r entries: each stride is positive,
> and each padding pair gives nonnegative (low,high) extents. Padding inserts
> exact dtype-zero cells. Each kernel extent is positive and fits its padded
> input extent. Output extent j is
> floor((input[j]+low[j]+high[j]-kernel[j])/strides[j])+1,
> computed in checked int64 arithmetic. Products visit
> (input-channel,kernel-axis-0,...,kernel-axis-r-1) in row-major order for each
> output before the contraction tree. A zero input-channel extent is an empty
> contraction with dtype-zero result; zero batch or output-channel extents
> produce empty tensors without changing the spatial shape obligations.
> No scalar metadata broadcast or rank-named convolution alias is admitted.
> Einsum's equation
> explicitly chooses labels, contractions, diagonals, and output order; it
> cannot invent a missing extent. Multiplication, sums, constants,
> normalization epsilon, and affine results use [04-NUM-8]'s operand and
> explicitly declared accumulator widths, never an unrequested f64 graph.
>
> Failure: Malformed equations, repeated-label inconsistencies, incompatible
> contraction/batch/affine dimensions, invalid stride/padding, and metadata
> overflow fail loudly. Literal-provable errors fail at check time;
> runtime-dependent obligations retain guards. Backend limits are
> implementation dispositions, not reduced language signatures.
>
> Adjoint: Matmul and einsum contract the upstream cotangent with the other
> operand along the complementary axes. Convolution reverses the exact
> gather/multiply/reduce graph. Layer normalization differentiates its
> mean/variance and affine composition, including the epsilon operand. Every path preserves shape and
> applies spec/06's accumulation order; no identity receives an invented
> zero adjoint.
>
> Accumulator: Matmul and conv use spec/04 section 5.7.1's contraction
> accumulator; einsum uses [05-OP-33]'s multiply-and-balanced-add graph.
> Layer normalization reductions use [05-OP-30]. Every default, explicit
> accumulator, finalization, and operation order is preserved by the section
> 4 graph.

#### Sparse tensor identities

> **[05-OP-52]** Signature: `gather(values,indices,axis)`,
> `scatter(base,indices,updates,axis,mode)`,
> `scatter_replace(base,indices,updates,axis)`, and
> `scatter_elements(base,indices,updates,axis)` use section 3.5's exact
> index/result shapes; the internal scatter-add identity is the add-mode
> instance of scatter.
>
> Domain: Indices use any active signed-integer dtype at their exact stored
> width. Tensor data and updates share one active element dtype; axis is
> int32. The selected scatter mode is explicitly add or replace. Bool data
> may be selected/replaced but cannot enter numeric addition.
>
> Result: Gather selects the indexed source cells. Add scatter accumulates
> at matching destinations; replace scatter uses the last update in
> row-major update order. Scatter-elements uses its elementwise index shape
> and the same deterministic replacement order. All pure selections retain
> stored bits without conversion, including wide integer indices.
>
> Failure: Noninteger indices, dtype/shape mismatches, unknown modes, and
> out-of-bounds indices fail loudly under [05-SPARSE-1] and section 3.5. No
> clipped index, skipped update, or atomic race supplies a result.
>
> Adjoint: Float gather scatters cotangents to source positions; float
> scatter-add gathers cotangents for updates and preserves the base
> cotangent. Replace scatter and scatter-elements structurally reject
> differentiation as section 3.5 requires. Indices and axes are
> non-differentiable.
>
> Accumulator: Additive data/cotangent collisions use the exact same-dtype
> accumulation and order specified in section 3.5 and spec/06. Replacement
> has no numeric accumulator.

#### Tensor ordering and selection

> **[05-OP-53]** Signature: `where(condition,a,b)` uses a bool condition and same-shaped
> same-dtype branches; `cumsum(x,axis)` preserves x's shape with the default
> sum result dtype, while `sort(x,axis)` returns (values, int64 indices) at
> that shape; `diagonal(x,axis1,axis2)` and `trace(x,axis1,axis2)` select
> and reduce the paired axes; `clamp(x,low,high)` preserves its numeric
> operand shape; `split(x,axis,sizes)` returns a List of tensor slices with
> int64 sizes.
>
> Domain: Axis arguments are int32. Where, diagonal, and split admit every
> active tensor element dtype by exact selection. Cumsum, sort, trace, and
> clamp admit the arithmetic dtype domains and parameter shapes specified
> for their corresponding tensor operations in [05-OP-33]. No parameter or
> element is silently narrowed.
>
> Result: Where selects stored bits directly from the chosen branch without
> converting bool to numeric. Cumsum returns inclusive axis-prefix sums;
> sort orders each axis slice using [05-OP-33]'s NaN/tie rule; diagonal uses
> that atom's axis ordering, and trace sums the diagonal. Clamp follows the
> atom's exact lower/upper selection rule. Split preserves source order,
> dtype and bits; its sizes partition the entire selected extent.
>
> Failure: Invalid axes, rank/shape/dtype mismatches, inconsistent or
> negative split sizes, invalid clamp bounds, and checked integer arithmetic
> overflow fail with the owning operation's diagnostic. Metadata errors
> cannot be hidden by an empty tensor.
>
> Adjoint: Where routes g only to the selected branch. Cumsum is the reverse
> inclusive sum, diagonal scatters g into an otherwise zero tensor, trace
> broadcasts g onto the diagonal, split concatenates the slice cotangents,
> and clamp routes only through the selected differentiable operand. Sort
> uses the saved permutation under [05-OP-33]'s tie rule. Discrete
> data/control arguments follow the same atom's structural rejection/zero
> rules.
>
> Accumulator: Cumsum and trace use the accumulator and operation order of
> [05-OP-33]; selection and split have no accumulator. Cotangent collisions
> obey spec/06.

#### List structure and counts

> **[05-OP-54]** Signature: `len(xs)` accepts List[T] or Dict[K,V] and returns int64;
> `index(xs,i)` takes List[T] and int64 and returns T; `append(xs,x)` and
> `concat(xs,ys)` return List[T]; `take(xs,n)` and `drop(xs,n)` take int64
> counts; `chunk(xs,n)` returns List[List[T]]; `range(start,end)` takes
> int64 endpoints and returns List[int64]. Separately, `drop(value)`
> consumes one value and returns unit.
>
> Domain: List elements retain their exact type, recursively, with no
> numeric conversion. Element/count/index quantities are int64. The two
> concat forms are disjoint: List concatenation takes a List second
> argument; tensor concatenation takes an int32 axis and is governed by
> [05-OP-62]. The one-argument drop is explicit lifetime consumption, not a
> List count default.
>
> Result: Length is the exact collection size. Index selects a zero-based
> element. Append and concat preserve order. Take/drop retain/remove up to n
> leading elements; a count above length yields the whole/empty List. Chunk
> uses consecutive groups of size n, retaining a shorter final group. Range
> is the half-open ascending integer interval and is empty when end <=
> start. Drop(value) performs the ordinary linear release.
>
> Failure: Negative index/count, out-of-bounds index, nonpositive chunk
> size, wrong arity/type, and unrepresentable int64 counts fail loudly.
> Clamping take/drop at length is their stated semantics and never applies
> to index.
>
> Adjoint: Float-containing List selection/concatenation/chunking routes
> cotangents through the exact element positions under spec/06 section
> 2.10.1; omitted elements receive recursive zero cotangents. Counts/indices
> have zero cotangent; range is structurally non-differentiable. Explicit
> lifetime drop follows spec/06's structural drop rule.
>
> Accumulator: No forward numeric accumulator. Repeated cotangent
> destinations use spec/06's ordered own-width combination.

#### Higher-order List identities

> **[05-OP-55]** Signature: `map(f,xs)` maps T->U over List[T]; `filter(f,xs)` and
> `partition(f,xs)` use T->bool; `fold(f,init,xs)` and `scan(f,init,xs)` use
> (A,T)->A; `flat_map(f,xs)` uses T->List[U]; `flatten(xss)` takes
> List[List[T]]; `zip(xs,ys)` returns List[(T,U)]; `enumerate(xs)` returns
> List[(int64,T)]. Callback effects E are preserved in the result function's
> effects.
>
> Domain: Element and accumulator types may be recursive checked types, with
> exact declared numeric dtypes. Callback arity, parameter, result and
> effect constraints must hold for every invocation. No dtype funnel or
> callback-result default is allowed.
>
> Result: Callbacks execute exactly once per visited element in source
> order; an empty List executes none. Filter preserves retained order;
> partition returns retained and rejected Lists in original order. Fold
> returns the final state; scan returns the successive post-step states.
> Flat-map flattens each callback result in order. Flatten removes one List
> layer. Zip stops at the shorter length. Enumerate uses zero-based int64
> indices.
>
> Failure: Callback traps propagate at their actual invocation; incompatible
> callback/result types or unavailable effects are type errors. An
> index/count beyond int64 traps Overflow, never wraps. No callback may be
> elided merely because its result is unused.
>
> Adjoint: Use spec/06 section 2.10.1's exact executed List/callback graph:
> map differentiates each invocation; fold/scan reverse the recorded
> recurrence; flat-map/flatten/zip route the corresponding elements. Filter
> and partition hold the exact forward predicate mask constant and route
> each output cotangent to its source position, with zero for omitted
> positions, under spec/06 section 2.10; the mask itself has zero cotangent.
> Enumeration indices have zero cotangent.
>
> Accumulator: Fold and scan retain A at each step; each numeric callback
> result finalizes at its own declared dtype before becoming the next state.
> Cotangent combination follows spec/06's order.

#### Dictionary identities

> **[05-OP-56]** Signature: `dict_of(entries)` takes List[(K,V)]; `dict_get(dict,key)`
> returns Option[V]; `dict_contains(dict,key)` returns bool;
> `dict_remove(dict,key)`, `dict_insert(dict,key,value)`, and
> `dict_merge(left,right)` return Dict[K,V]; `dict_keys`, `dict_values`, and
> `dict_entries` return List[K], List[V], and List[(K,V)].
>
> Domain: Keys share one static type: string, bool, or any active
> signed-integer scalar dtype; values share one checked type V. Float and
> aggregate keys are type errors. Numeric keys/values retain exact dtype and
> stored bits; recursive values are not erased. Key equivalence includes the
> exact kind/dtype and stored value, as in [05-OP-32].
>
> Result: Lookup returns Some for a present key and None for absence;
> contains reports membership. Insert replaces an equal existing key's
> value; remove of an absent key leaves the dictionary unchanged. In dict_of
> and merge, later/right entries win equal-key conflicts. Keys, values, and
> entries enumerate in [05-OP-32]'s canonical observation order (bool false
> before true, integer mathematical order, string Unicode scalar order),
> preserving key/value pairing. Updates retain insertion positions
> internally as that atom specifies; observation never exposes insertion
> order.
>
> Failure: Inconsistent key/value types and non-admitted key equality are
> type errors. Missing lookup is the explicit Option result, not an
> exception or numeric default. No hash-table iteration order may leak into
> observation.
>
> Adjoint: Dictionary construction, updates, and selection obey spec/06's
> recursive value-cotangent and discrete-key rules. A discrete key is never
> differentiated through a lookup decision; an unsupported differentiability
> shape rejects structurally rather than returning an invented zero.
>
> Accumulator: No forward numeric accumulator. Shared value cotangents
> combine at their declared widths under spec/06.

#### List/tensor conversion

> **[05-OP-57]** Signature: `to_tensor(xs)` takes a rectangular, recursively nested List
> with one active scalar tensor-element leaf dtype T. A nesting depth r
> yields a rank-r tensor. `to_list(x)` borrows a tensor of any positive rank r
> and returns r nested Lists with scalar leaf dtype T.
>
> Domain: All active tensor element dtypes, including bool, are admitted
> without conversion. The recursive shape relation is `shape(scalar) = []`
> and `shape([v0, ..., vn-1]) = [n] ++ s` when every child has the same
> shape s and leaf dtype T. This admits arbitrary List nesting, including
> spec/04 section 4.5.1's rectangular nested construction; it has no
> rank-two exception or maximum nesting depth. A List's element type
> determines an empty result's dtype. Inner extents that an empty outer
> List cannot establish remain explicit shape obligations; no default f32
> or invented trailing extent is permitted. Strings and mixed leaf dtypes
> are type errors.
>
> Result: `to_tensor` concatenates leaves in recursive source order into
> the tensor's row-major storage and preserves the full recursive shape,
> dtype, and each element's stored bits. `to_list` recursively partitions
> the row-major elements by every successive axis, preserving axis order,
> all observable lengths, dtype, and stored bits. A zero extent gives an
> empty List at that level; trailing extents below an empty List are not
> encoded as invented values. `to_tensor(to_list(x))` has x's exact value
> and shape when those unobservable extents are supplied by the expected
> tensor type; otherwise they remain explicit shape obligations. Its
> identity adjoint follows spec/06 section 2.10 and retains the saved shape.
>
> Failure: Inconsistent child shapes reject, statically when known and at
> runtime otherwise. Unresolved required extents and unrepresentable size
> arithmetic fail loudly. Invalid leaf types, mixed leaf dtypes, and
> scalar or rank-zero `to_list` operands are type errors; use [05-OP-50]'s
> explicit scalar conversion for rank zero.
>
> Adjoint: For float T, `to_tensor` reconstructs the saved source List
> nesting and routes each corresponding element cotangent. `to_list` builds
> the original full tensor shape from the nested element cotangents,
> using the saved forward shape even when empty Lists hide trailing extents.
> Integer/bool
> differentiated data structurally rejects under spec/06.
>
> Accumulator: None.

#### String identities

> **[05-OP-58]** Signature: `string_len(s)` returns int64; `string_concat(a,b)` returns
> string; `string_slice(s,start,length)` takes int64 offsets and returns
> string; `string_contains`, `string_starts_with`, and `string_ends_with`
> take two strings and return bool; `string_trim(s)` returns string.
>
> Domain: String length and slicing count Unicode scalar values, not UTF-8
> bytes or grapheme clusters. Strings are not normalized; offset/count
> quantities are exact int64. The operations are pure.
>
> Result: Concatenation preserves both strings' bytes. Slice selects up to
> length scalar values starting at start, returning empty at or beyond the
> end. Contains/prefix/suffix compare the supplied literal strings,
> including the empty string. Trim removes leading and trailing Unicode
> White_Space characters and preserves the interior bytes.
>
> Failure: Negative slice start/length is a domain error; unrepresentable
> int64 lengths trap Overflow. Wrong types/arity are type errors, not
> stringification or parsing fallbacks.
>
> Adjoint: These string operations structurally reject differentiation; no numeric cotangent is fabricated.
>
> Accumulator: None.

#### Explicit text parsers

> **[05-OP-59]** Signature: `to_int(text: string)->Option[int64]` and `to_float(text:
> string)->Option[f64]` explicitly select their result dtype.
>
> Domain: Both parsers consume the whole string after trimming surrounding
> Unicode whitespace. The integer grammar is an optional sign followed by
> one or more decimal digits. The float grammar is ASCII
> `[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?`, or an optional sign
> and case-insensitive `inf`, `infinity`, or `nan`. No collection or
> non-string argument is implicitly rendered.
>
> Result: A valid in-range integer yields its exact int64. A valid float
> yields the correctly rounded f64, including IEEE overflow to infinity,
> underflow with signed zero, and an explicitly parsed exceptional value.
> The Option variants retain the chosen dtype. These named parsers do not
> replace JSON/CSV's distinct [05-OP-2] grammar and refusal rules.
>
> Failure: Malformed text and out-of-range int64 yield None. Parsing never
> returns a numeric zero for failure and never widens an integer result
> through f64. Wrong input types are type errors.
>
> Adjoint: Text parsing is structurally non-differentiable.
>
> Accumulator: None.

#### Host observation and file boundaries

> **[05-OP-60]** Signature: `print(value)->unit!{IO}`, `debug(value)->value!{IO}`,
> `fail(message:string)->T`, and
> `test_assert(condition:bool,label:string)->unit!{Test}` are the
> observation/control identities. `read_file(path)->string`,
> `write_file(path,text)->unit`, `read_lines(path)->List[string]`,
> `read_bytes(path)->List[int64]`, `file_exists(path)->bool`,
> `list_dir(path)->List[string]`, and `mmap_file(path)->MappedFile` take
> string paths with IO effects.
> `mmap_read(mapped,offset,length)->List[int64]` and
> `mmap_len(mapped)->int64` borrow the mapped handle and take exact int64
> offsets/counts.
>
> Domain: All argument/result types and effects are exact. Observation
> retains the recursive rendering domain of spec/05 section 8 and
> [05-HOST-1]; fail has no returning execution. Mapped/file bytes are
> integers in 0..255, not float payloads. Test assertions obey [05-HOST-3].
>
> Result: Print emits the observation and debug returns the original value
> after observing it. Fail raises its supplied failure; test_assert succeeds
> only for true. Text reads return decoded text, lines preserve file order
> without their line endings, byte reads preserve every byte. Write replaces
> the target file's contents. List_dir obeys [05-HOST-4]'s exact host-name
> ordering. Mmap_len is the exact byte length; mmap_read selects exactly
> length bytes starting at offset.
>
> Failure: OS/read/decode/write failures propagate loudly with the operation
> and path. File_exists returns false for absence. Negative mapped bounds or
> an offset-plus-length beyond the mapping fail; checked offset-plus-length
> overflow traps Overflow. Offset at the end is valid only for zero length.
> Invalid assertion traps Test with its label. No stub, default value, or
> unused-result optimization may erase an effect.
>
> Adjoint: IO, failure, and testing operations are outside AD under the
> effect rules. No executable effect is replaced by a zero-cotangent path.
>
> Accumulator: None; all returned byte counts and offsets are exact checked
> int64.

#### CSV text structure

> **[05-OP-61]** Signature: `parse_csv(text:string)->List[Dict[string,string]]`;
> `csv_str(table,row:int64,column:string)->string`;
> `csv_strs(table,column:string)->List[string]`; and
> `csv_cols(table)->List[string]` operate on the text-table representation.
>
> Domain: CSV fields remain strings without numeric inference. Parsing uses
> the header as column names, comma separation, double-quoted fields with
> doubled quote escaping, and quoted delimiters/newlines. Numeric accessors
> are separately governed by [05-OP-3].
>
> Result: Parsing preserves record order and field text. Scalar lookup
> selects a zero-based row/column; plural lookup returns one cell per row in
> order. Column enumeration uses [05-OP-56]'s canonical string-key order.
> Empty tables remain empty rather than acquiring a synthetic numeric row.
>
> Failure: Malformed CSV, missing columns, out-of-range rows, and
> inconsistent table structure fail loudly; there is no default empty cell,
> row skipping, or inferred JSON constructor. Wrong types/arity are type
> errors.
>
> Adjoint: CSV parsing and text access structurally reject differentiation.
>
> Accumulator: None.

#### Tensor concatenation

> **[05-OP-62]** Signature: `concat(parts:List[tensor[..r,p]],axis:int32)->tensor[..r,p]`
> concatenates a nonempty List of equal-rank tensors along one existing
> axis; the List/List overload is [05-OP-54].
>
> Domain: All active tensor element dtypes are admitted. Elements share p
> and every non-concatenated extent. The selected axis may have different
> extents; rank is positive. The exact inferred result follows spec/04
> section 4.5.4, including ragged direct List literals and guarded runtime
> extents.
>
> Result: The selected result extent is the checked int64 sum of source
> extents. Other extents are preserved. Source tensors contribute in List
> order; stored element bits and dtype are unchanged.
>
> Failure: Empty tensor lists, invalid axes, rank/precision/non-axis extent
> mismatches, and metadata overflow fail loudly. A joined unknown extent
> never licenses retaining the first element's extent without a guard.
>
> Adjoint: For float data, split the upstream cotangent at the exact source
> boundaries, preserving the input List's recursive shape. Axis values have
> zero cotangent; integer/bool differentiated data structurally rejects.
>
> Accumulator: Only checked int64 metadata addition; no forward numeric element accumulator.

#### Checked cast identity

> **[05-OP-63]** Signature: `cast(value,target_dtype)` returns the same scalar or tensor
> shape with the explicitly named target dtype.
>
> Domain: The source/target product is exactly [04-NUM-14]'s checked cast
> domain over active dtypes, with the type/literal spelling and admission
> rules of spec/04 section 5.2. No backend or host carrier narrows the
> source or target domain.
>
> Result: Read the exact source stored value and finalize it directly into
> the target dtype under [04-NUM-14], including same-dtype identity,
> signed-zero, NaN, integer exactness, and float rounding rules. No
> intermediate float image may change an integer conversion.
>
> Failure: Out-of-range conversions trap Overflow; fractional
> float-to-integer conversion traps Domain. Unsupported source/target kinds
> are type errors. Named truncating/saturating/wrapping conversions retain
> their separate identities and cannot act as implicit fallbacks.
>
> Adjoint: Float-to-float casts use the cast adjoint specified by spec/06
> and [04-NUM-14]; discrete source/target casts structurally reject
> differentiation.
>
> Accumulator: None.

## 4. Standard Lowerings (Tier 2 → Tier 1)

### 4.1 Matrix Multiplication

```
matmul(A: tensor[..., i, j, p], B: tensor[..., j, k, p],
       accumulator: prec = default(p))
       → tensor[..., i, k, p]
```

First align the leading batch axes explicitly: prepend missing axes with
`insert`, then use `expand` only on an existing extent of one whose target
extent differs. Incompatible non-unit extents are rejected. Let `b` be the
number of aligned batch axes, and `acc` the resolved accumulator precision.
The graph uses zero-based axis indices (int32) and int64 extents:

```
1. A_expanded = insert(A, b+2, k)             ;; [..., i, j, k]
2. B_expanded = insert(B, b, i)               ;; [..., i, j, k]
3. product = mul(A_expanded, B_expanded)
4. accumulated = sum(product, axis=b+1, accumulator=acc) ;; [..., i, k, acc]
5. result = cast(accumulated, p)             ;; identity when acc == p
```

Step 4 denotes the IR reduction, whose output dtype is its accumulator;
step 5 restores the declared matrix result dtype. `insert` introduces and
replicates a new axis; `expand` only broadcasts an existing unit axis.

This is the Einstein summation form. The lowering's step-4 `sum` is
[05-OP-30]'s canonical balanced tree, so `matmul`'s result bits are
target-independent like every other operation absent from [05-OBS-3]'s
tolerance table. A vendor GEMM library or fused kernel may implement it only
where it reproduces those exact bits and traps; a target-selected
accumulation order would be observable and is therefore never available
implicitly. Vendor-kernel matmul, if wanted, requires a future named,
explicit call-site opt-in authored as its own atom on [04-NUM-8]'s opt-in
pattern - never a backend default, build flag, or global mode
(chelis#1315 owns that design). `einsum` is likewise exact under
[05-OP-33]'s pinned contraction tree; neither operation is a compatibility
spelling for the other, and a lowering from one to the other is legal only
where it preserves the owning atom's bits and traps.

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
for rationale. Use `reduce_sum` over an explicit `insert`+`mul` lowering for
integer inner products.

### 4.2 Softmax

```
softmax(x: tensor[D, p], axis: int) → tensor[D, p]
```

Normalize the selected axis to its nonnegative index `a`; let
`extent = shape(x,a)`. Restore a reduced axis by inserting it at that same
position, including when it is not the trailing axis:

```
1. m = max_reduce(x, a)
2. m_expanded = insert(m, a, extent)
3. shifted = sub(x, m_expanded)
4. e = exp(shifted)
5. s = sum(e, a)
6. s_expanded = insert(s, a, extent)
7. result = div(e, s_expanded)
```

### 4.3 Cross-Entropy Loss

```
cross_entropy(logits: tensor[batch, classes, p], labels: tensor[batch, i_signed]) → tensor[batch, p]
```

Lowering:
```
1. log_probs = log(softmax(logits, 1))
2. selected = gather(log_probs, labels, 1)    ;; [batch, batch]
3. paired = diagonal(selected, 0, 1)          ;; [batch]: log_probs[b, labels[b]]
4. result = neg(paired)
```

`i_signed` is any active signed-integer dtype. Labels must be in `[0,classes)` and the classes extent must be positive.
The diagonal couples each label to its own batch row. A fused selection may
avoid materializing the intermediate matrix only while preserving these
exact values, bounds failures, and adjoints. `gather` has [05-OP-52]'s direct
selection contract; one-hot arithmetic is not an equivalent definition for
unselected nonfinite values.

### 4.4 Layer Normalization

```
layer_norm(x: tensor[..., hidden, p], gamma: tensor[hidden, p],
           beta: tensor[hidden, p], epsilon: p) → tensor[..., hidden, p]
```

`epsilon` is an explicit float scalar of dtype `p`, with no implicit value.
Its stored value participates in the graph under the ordinary float rules,
including nonfinite values. Let `a` be the input's trailing axis and
`hidden = shape(x,a)`, which must be positive.

```
1. m = mean(x, a)
2. m_exp = insert(m, a, hidden)
3. centered = sub(x, m_exp)
4. var = mean(mul(centered, centered), a)
5. var_exp = insert(var, a, hidden)
6. normed = div(centered, sqrt(add(var_exp, epsilon_exp)))
7. result = add(mul(normed, gamma_exp), beta_exp)
```

Here `gamma_exp` and `beta_exp` are constructed by inserting each input
leading axis before the affine parameter's trailing axis, in input order,
with its exact `shape(x,axis)` extent. Construct `epsilon_exp` from
`scalar_to_tensor(epsilon)` by inserting all input axes in order with those
same extents. All three operands then have exactly `x`'s shape; no implicit
scalar or tensor broadcasting is part of the recipe. Differentiation
reverses this graph for the input, both affine parameters, and epsilon.

### 4.5 Convolution

```
conv(input: tensor[batch, in_c, s0, ..., s(r-1), p],
     kernel: tensor[out_c, in_c, k0, ..., k(r-1), p],
     strides: List[int64], padding: List[(int64, int64)])
  → tensor[batch, out_c, o0, ..., o(r-1), p]
```

Here r is any positive integer and p is any active float dtype. The ellipses
denote r explicit corresponding spatial axes, not an inferred permutation.
For a different authored layout, use an explicit `permute` into this layout
and an explicit `permute` of the result. Equal input-channel extents are
contracted; neither channels nor spatial metadata broadcast implicitly.

[05-OP-51] gives the per-axis extent formula and failure rules. For example,
input spatial extents `[5,7]`, kernel extents `[3,2]`, strides `[2,1]`, and
padding `[(0,1),(2,0)]` yield output spatial extents `[2,8]`. The same rule
applies to one, three, and higher spatial ranks. A statically known rank
determines both metadata lengths and the number of result extents. A generic
instantiation must discharge that relation; spec/04 section 4.5.3's rank
spreads do not authorize positional rewriting of unknown axis identities.

The defining graph pads the input, gathers each output window, and flattens
each window in `(input-channel,kernel-axis-0,...,kernel-axis-r-1)` order.
Flatten the kernel in that same order, apply section 4.1's matrix contraction
with its resolved default accumulator, then reshape and permute into the
declared output layout. There is no separately selected convolution
accumulator. Strides, padding, and axis correspondence are discrete metadata
with zero cotangents; the input and kernel adjoints reverse this exact graph.

An implementation may specialize this graph for a spatial rank or target
library only while preserving its full shape, dtype, accumulation, and
adjoint contract. Specialization does not create a public rank-named builtin.

### 4.6 Embedding

```
embedding(indices: tensor[..index_axes, i], table: tensor[vocab, ..entry_axes, p])
  → tensor[..index_axes, ..entry_axes, p]
```

Here `i` is any active signed-integer dtype and `p` is any active tensor
element dtype. The index axes and entry axes are independent rank spreads.
The graph is exactly `gather(table, indices, 0)`: each index selects one
table entry without arithmetic on its stored values. Out-of-range indices
fail under [05-OP-52]. The float-table adjoint scatters and accumulates
cotangents under that atom; indices have zero cotangent. A one-hot matrix
contraction is not equivalent: zero times an unselected NaN or infinity can
change the selected result.

### 4.7 Multi-Head Attention

```
multi_head_attention(q: tensor[..batch, queries, key_dim, p],
                     k: tensor[..batch, keys, key_dim, p],
                     v: tensor[..batch, keys, value_dim, p],
                     mask: tensor[..batch, queries, keys, bool], scale: p)
  → tensor[..batch, queries, value_dim, p]
```

The leading axes are explicit matching batch axes, including any authored
head axes; query count, key count, and value width need not coincide.
`scale` is an explicit scalar of the shared active float dtype. A caller
choosing inverse-square-root scaling supplies that value at its declared
dtype. The mask has the score shape; a shared mask is aligned explicitly
with `insert` or unit-axis `expand` before this composition.

```
1. k_transposed = permute(k, ...batch_axes, last_axis, second_last_axis)
2. raw_scores = matmul(q, k_transposed)
3. scores = mul(raw_scores, scale_exp)
4. masked = where(mask, scores, negative_infinity_exp)
5. weights = softmax(masked, last_axis)
6. output = matmul(weights, v)
```

The permutation supplies every axis as a positional int32 argument,
exchanging only the last two. `scale_exp` and `negative_infinity_exp` insert
all score axes into rank-zero tensors containing the supplied scale and
dtype negative infinity.
`where` selects values with a bool condition under [05-OP-53]; it performs
no mask-to-number conversion or multiply/add masking. A fully masked row
follows the ordinary nonfinite softmax graph; it does not acquire a default
zero result.

---

## 5. AD Completeness

Every numeric callable states one of three contracts: an adjoint, a zero
cotangent, or a structural `grad` rejection. No operation acquires an adjoint
from a backend fallback.

**Zero-cotangent predicates and sources.** Each exact [05-OP-36] identity:
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

**Almost-everywhere differentiable:** `max_elem` routes the whole cotangent to
the first operand when the inputs are equal; that first-operand tie rule is
the language's exact subgradient convention for the selection identities.
`relu` carries [05-OP-43]'s own adjoint instead: its gradient is exactly zero
at `x = 0`, matching the ecosystem's relu convention, while direct `max_elem`
applications keep the selection rule. Both are valid targets for `grad`.

**Differentiation barrier:** [05-OP-42] `stop_gradient` returns its
argument's exact value and contributes the shape-preserving zero cotangent;
its argument's subgraph is outside adjoint construction and structural
rejection analysis, so a structurally rejected operation reached only through
the barrier does not reject the surrounding graph.

**Second-order derivatives:** `grad(grad(f))` is valid exactly when every
operation reached by `f` has the required second-order adjoint.

---

## 6. Reference Implementations

For each RISC primitive, the following pseudocode gives illustrative
implementation shapes for the C backend; it is not a semantic oracle. Each
primitive's governing numbered rule or operation atom remains authoritative.

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

// max_elem: first NaN, then the numerical maximum, preserving lhs on equality
for (int i = 0; i < n; i++) out[i] = select_max_first(a[i], b[i]);

// min_elem: first NaN, then the numerical minimum, preserving lhs on equality
for (int i = 0; i < n; i++) out[i] = select_min_first(a[i], b[i]);

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

### 8.2 The Root Manifest

> **[05-OBS-7]** For a selected target, every top-level value binding and
> every effect-free zero-parameter definition with a value result SHALL
> contribute an owed root. Applying an effectful declaration merely to observe
> it would change its execution count and SHALL NOT occur. Parameterized
> definitions are callable entries, not observation
> roots until evaluation explicitly selects one and supplies all of its runtime
> inputs; that concrete call result becomes an owed root. A zero-parameter
> generic definition whose result retains an unresolved type, dimension, or
> rank parameter likewise remains a callable entry until a concrete call
> instantiates its result. The manifest SHALL list owed roots in source
> declaration order.

> **[05-OBS-8]** A tuple-valued root SHALL expand recursively into dotted
> positional names in depth-first order (`result.0`, `result.1.0`, ...). A
> fixed-product user ADT-valued root SHALL expand only when its constructor is
> statically fixed:
> record fields use their declared field names, and positional fields use a
> declared field name when present or their zero-based position otherwise. If
> the constructor is not statically fixed, the ADT remains one bare root.
> Variable-sized and opaque container ADTs such as `List`, `Dict`, `Option`,
> and `MappedFile` are not fixed products and remain bare roots.

> **[05-OBS-9]** The root manifest SHALL carry the selected target and SHALL
> assign each entry its Tensor or Host lane and exactly the free runtime tensor
> inputs in that root's reachable closure. Selecting one root SHALL NOT make a
> sibling root's input live. Dotted descendants inherit their originating
> root's lane and input closure.

> **[05-OBS-10]** Lane assignment SHALL be target-aware. A root whose declared
> dtype is outside the selected target's Tensor capability SHALL route through
> the Host lane without changing dtype; it SHALL NOT be narrowed, dropped, or
> admitted to the Tensor lane because another target supports that dtype.

> **[05-OBS-11]** A non-empty root manifest SHALL require an executable
> observation entry point; an empty manifest SHALL produce an object without
> one. Every successful eval result and build artifact SHALL realize every
> selected manifest entry in manifest order. If an assigned lane cannot
> produce an owed root, the whole operation SHALL fail through [05-UNS-1]
> before returning a partial result or artifact, naming the root, lane, and
> failure reason.

(Metal observation-entry generation is not fully implemented; see
[chelis#912](https://github.com/Chelis-Lang/chelis/issues/912).)
