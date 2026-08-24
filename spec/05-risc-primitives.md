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

**Tier 2: Derived Built-Ins** — convenience functions that the compiler lowers to RISC primitive compositions during IR construction. The desugarer emits these; the IR pass decomposes them. They exist so the Deep representation stays readable.

Together, the two tiers define everything the compiler has special knowledge of. Anything that can be expressed as a Chelis program composing these primitives — without requiring custom AD adjoints, backend fusion rules, or compiler-recognized names — belongs in the standard library (`Std.*`) or in external packages, not in the core. See `spec/design/chelis_canonical_reference.md` §8.5 for the full scope boundary taxonomy.

Phase `3h` expands the practical primitive surface beyond this initial minimal set with
`einsum`, `concat` / `split`, `gather` / `scatter`, `where`, `cumsum`, `sort`,
`diagonal` / `trace`, and `clamp`. `School.Nn.Embedding` (moved to the `school` library
in chelis-std 0.4.0) remains the named library surface over `gather`.

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
> `string`, or deferred operand is a type error.
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

*(Not fully implemented; see chelis#753.)*

**`div` semantics (float-only since chelis#178).** `div(a, b)`
is **restricted to float operands** (f32, f64, f16, bf16) and
lowers to the target's native floating `/` operator with IEEE-754
semantics. Corner cases follow IEEE: `1/0 = +inf`, `1/-0 = -inf`,
`0/0 = NaN`, `1/-1 = -1`, `(any non-NaN) / -2.0` yields the
algebraic value. A historical `mul(a, exp(neg(log(b))))`
decomposition returned NaN for any `b ≤ 0` because `log(b)` is
undefined there; that decomposition is not reachable from any
Tier 2 op.

`div(int_tensor, int_tensor)` is a **type error** (the `/`
operator on integer operands is likewise rejected, because `/`
desugars to `div`). The diagnostic cites this section and points
at `floor_div` / `trunc_div`. This is a deliberate breaking change
from the pre-chelis#178 behavior, where `div` on integer operands
performed C/Rust truncating division. The single-op-two-semantics
overload (`div(7, 2) == 3` for ints, `== 3.5` for floats) was a
footgun and matched none of torch / JAX / numpy: their default
`divide` upcasts integers to float, and their integer division op
is `floor_divide` (round toward −∞), which the old C-truncating
behavior also did not match. Integer division now has two explicit,
named primitives below.

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
> differentiable expression. Integer, `bool`, `string`, and deferred operands
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

*(Not fully implemented; see chelis#965.)*

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
`bool`, `string`, or the deferred `f8e4m3` dtype.

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

| Name | Signature | Semantics | AD Adjoint |
|---|---|---|---|
| `sum` | `(&tensor[d1,...,dn,p], axis: int32, accumulator: prec = default(p)) -> tensor[d1,...,d{k-1},d{k+1},...,dn,sum_result(p,accumulator)]` | Sum over axis k, removing that dimension. `accumulator` controls the running precision; `sum_result` is §5.7.1's result-precision rule. | `expand(g, original_shape, axis=k)` (gradient flows back at the operand precision `p`; the adjoint is computed in operand precision) |
| `mean` | `(&tensor[d1,...,dn,p_float], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,p_float]` | Arithmetic mean over axis k | `expand(g / axis_extent, original_shape, axis=k)` |
| `max_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,p]` | Maximum over axis k | Route a NaN result to its first NaN; otherwise split `g` equally among every element equal to the selected maximum, including infinities |
| `min_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,p]` | Minimum over axis k | Route a NaN result to its first NaN; otherwise split `g` equally among every element equal to the selected minimum, including infinities |
| `prod_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,p]` | Product over axis k | Reverse-mode derivative of the exact stride-4 product tree (mathematically, `g` times the product of every other element) |
| `argmax_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,int64]` | Lowest axis index of the maximum or first NaN | Non-differentiable; `grad` rejects it |
| `argmin_reduce` | `(&tensor[d1,...,dn,p], axis: int32) -> tensor[d1,...,d{k-1},d{k+1},...,dn,int64]` | Lowest axis index of the minimum or first NaN | Non-differentiable; `grad` rejects it |

> **[05-OP-11]** `mean(x, axis) -> result` admits a tensor operand of
> `f16`, `bf16`, `f32`, or `f64` and returns the same float dtype with the
> selected axis removed. On a non-empty axis its value is exactly the
> composition `div(sum(x, axis), divisor)`: `sum` uses its §5.7.1 default
> accumulator, order, result dtype, and finalization; `divisor` is the positive
> axis extent converted once to the sum result dtype under [04-NUM-14]; then
> `div` executes and finalizes as a separate operation at [04-NUM-8]'s declared
> width. Integer, `bool`, `string`, and deferred operands are type errors. A
> zero-length axis is a type error when statically known. If an execution-time
> extent is zero, a guard before the composition traps `Domain` as operation
> `mean` at the result dtype. The adjoint is
> `expand(g / divisor, original_shape, axis)` at the operand dtype. `mean` has
> no accumulator parameter of its own.
>
> **[05-OP-12]** `max_reduce(x, axis) -> result` admits every active signed
> integer and float tensor dtype and returns that same dtype with the selected
> axis removed. Values are compared without conversion at their stored dtype.
> The first NaN in increasing axis-index order is the result; otherwise the
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
> zero. Integer operands are forward-only and `grad` rejects them.
>
> **[05-OP-13]** `min_reduce(x, axis) -> result` has the same dtype,
> finalization, empty-axis, accumulator, and differentiation contract as
> [05-OP-12], replacing maximum by minimum. It returns the first NaN in
> increasing axis-index order; otherwise it preserves the first stored
> representation among equal minima. For float operands, every element equal
> to the selected non-NaN minimum, including equal positive or negative
> infinities, receives the upstream cotangent divided by the number of equal
> minima.
>
> **[05-OP-14]** `prod_reduce(x, axis) -> result` admits every active signed
> integer and float tensor dtype and returns that same dtype with the selected
> axis removed. It has no accumulator parameter: multiplication uses the
> operand's [04-NUM-8] arithmetic width (`f16` and `bf16` therefore multiply
> in `f32`). The reduction uses four product lanes initialized to one; axis
> element `i` updates lane `i mod 4`, and the lanes combine as
> `(p0 * p1) * (p2 * p3)`. Each update and combine is finalized at the
> arithmetic width, integer overflow is checked at every multiplication, and
> the completed product is finalized once to the operand storage dtype. This
> logical tree is identical in every lane and governs [04-NUM-12] trap
> occurrence. A zero-length axis returns the multiplicative identity one at
> the operand dtype. For float operands, the adjoint is the reverse-mode
> derivative of that exact multiplication tree. A division-free
> prefix/suffix implementation may be used only when it preserves that tree's
> arithmetic-width operation order and result bits; consequently gradients at
> zero operands are defined. Integer operands are forward-only and `grad`
> rejects them.
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
runtime shape. No phase may substitute a rank-zero placeholder or axis zero.

**Accumulator parameter (`sum` only).** The optional `accumulator: prec`
parameter controls the precision used for the running sum and the precision
of the output tensor. The default is the operand precision for f32/f64/i32/i64
operands, and a wider promoted type for narrower operand types. The full
table of defaults (and the rationale for each row) is the authoritative
statement in `spec/04-type-system.md` §5.7.1; this primitive doc is the
operational location of the parameter on the IR node.

In short:

- `bf16` / `f16` operands → `f32` accumulator → `bf16` / `f16` result
- `f32` operands → `f32` accumulator → `f32` result
- `f64` operands → `f64` accumulator → `f64` result
- `int8` / `int16` operands → `int32` accumulator → `int32` result
- `int32` operands → `int32` accumulator → `int32` result
- `int64` operands → `int64` accumulator → `int64` result

The `f32` accumulator for `bf16`/`f16` is consumed inside the op and downcast
on output, so the caller sees a uniform-precision result tensor. Only the
narrow integer rows widen their result, and they do so for overflow safety.

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

*(Not fully implemented; see chelis#1281.)*

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
> rank; and every entry SHALL be a positive int64. A violation is a type
> error, never an empty-list default, truncated rank, or backend assertion.

- `window_shape` and `strides` are int64 lists of equal length
  `n >= 1`.
- The trailing `n` axes of the input are the windowed axes. The
  leading `rank(input) - n` axes pass through unchanged.
- Each windowed entry must be a positive int64. `window_shape[i] >= 1`
  and `strides[i] >= 1`.
- The output rank equals the input rank. Leading dims match the
  input; trailing dim `i` is
  `floor((input_dims[rank - n + i] - window_shape[i]) / strides[i]) + 1`.
  When that formula yields a non-positive value the call is a type
  error (an empty window output is structurally meaningless under
  `Valid` padding).

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

Second-order AD through `ReduceWindowGrad` is not defined.

> **[05-RWIN-2]** The C target SHALL implement `ReduceWindow` and
> `ReduceWindowGrad`. The HIP target SHALL reject either node before codegen
> with an `unsupported_feature` diagnostic that directs the caller to the C
> target; it SHALL NOT emit a stub kernel or enter a panic backstop.

**Reduction order.** `sum` evaluates each reduced slice with a
stride-4 cascade: four accumulators initialized to zero receive axis element
`i` in lane `i mod 4`, and the lanes combine as
`(acc0 + acc1) + (acc2 + acc3)`. Each update and combine executes in the
selected accumulator dtype. The order is positional and deterministic. Every
evaluator and backend uses that logical operation tree; parallel scheduling
may vary only when it preserves the same result bits and trap occurrence.

`prod_reduce` uses one cross-lane order: four accumulators initialized to one
receive axis element `i` in lane `i mod 4`, then combine as
`(p0 * p1) * (p2 * p3)`. Every evaluator and backend uses that logical order;
parallel scheduling may vary only when it preserves the same operation tree.
Each multiplication executes at [04-NUM-8]'s arithmetic width, and the final
result is finalized once to the operand storage dtype. This order governs
both float result bits and [04-NUM-12] integer-overflow trap occurrence.

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
| `stride` | appropriate expand/scatter (implementation-specific) |

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

*(Not fully implemented; chelis#1112 owns the dim-surface gaps.)*

#### 2.4.1 Runtime (node-valued) bounds and reshape targets (chelis#616)

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

Runtime bounds are validated at run time in BOTH mandatory lanes with matching
error paths: the eval lane raises a clean error and the C backend emits an
abort guard for a negative bound, a shrink range overshoot, a non-positive
stride step, a negative reshape target extent, and a reshape target whose
element product disagrees with the input (`chelis_alloc_view` itself performs
no numel check, so the emitted guard is the only defense). The reshape numel
guard fires for ANY reshape whose output or input extents are not all static
literals — Sym-resolved targets and literal targets over runtime-sized inputs
included, not only node-valued targets — and same-shape elementwise ops guard
operand-shape agreement at equal rank whenever a non-static extent is
involved (chelis#664; rank-0 scalar operands are the backend's broadcast
idiom and are exempt). A dim whose extent
is computed by the op at run time is an *op-declared* symbolic dim: the C
backend declares it inline at the owning op (`int64_t name = <extent>;`,
the [05-DIM-1] extent carrier) and the
evaluator binds it from the actual value mid-evaluation; a second site
computing a different value for the same symbol aborts/errs loudly (the
over-unification guard). Since chelis#631/#632 the guard no longer fires on
a direct-return `shrink -> stride` chain under one sig symbol — anonymous
dims are not substitution keys, so the sig symbol attaches positionally to
the FINAL op only and each inner movement op declares its own extent (full
eval-vs-C parity). The checker's movement typing matches: symbolic-dim
pass-through is identity-only (stride step 1 / zero pad; see
spec/04-type-system.md §4.7), so a non-identity movement axis types a
fresh runtime-guarded extent rather than repeating the input's symbol.
The guard remains the soundness floor for a genuinely CLAIMED symbol
equality (e.g. an explicit `-> tensor[n]` over `stride(x, 2i64)`) and for
any future checker imprecision.

The movement adjoints are runtime-capable on the same representation: the
`shrink` adjoint pads with `after = shape(x, axis) - end`, the `pad` adjoint
shrinks to `end = before + shape(x, axis)`, and the `stride` adjoint's
upsample cascade reads `m_a = shape(g, axis)` and trims to
`(0, shape(x, axis))` with a runtime `m_a * step` merge extent — all as
node-valued bounds over fresh `Shape`/arithmetic scalars. Bound scalars are a
**stop-gradient boundary**: they are index math, carry no cotangent, and do
not pull their producers (e.g. a window-count `floor_div`) into the
differentiability check. A runtime (node-valued) stride STEP has no
structural adjoint yet and fails loud.

> **[05-MOV-1]** Runtime movement bounds and reshape targets SHALL be
> available on the eval and C lanes. The HIP and Metal targets SHALL reject
> them with a clean diagnostic naming the C target; they SHALL NOT erase the
> runtime value, substitute a literal bound, or emit a device kernel with a
> statically guessed extent.

This is the target disposition of the runtime-bound representation delivered
under chelis#616.

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
> operand is axis-domain `int32` ([05-DIM-1]); a negative or
> out-of-range axis is a loud error. The read is non-differentiable: it
> produces no adjoint and contributes no gradient. No accumulator rule
> applies.

Two lowering shapes
exist, and they are distinct:

- **As an extent argument** to `expand` / `reshape`, a `shape()` read is folded
  into the movement node's `DimExpr` (the output dim), not materialized as a
  value node. This is the pre-existing size-recovery path (chelis#318/#369).
- **As a scalar VALUE** (used in arithmetic, a `mean` divisor, or any other
  value position), a `shape()` read lowers to a dedicated `RiscOp::Shape { axis }`
  node — a rank-0 integer scalar equal to the input's runtime extent along a
  compile-time-constant `axis`. Before chelis#513 there was no such node and a
  scalar shape read fell through to a bogus `Load { name: "shape" }` placeholder,
  which was silently wrong in the eval lane (resolved to the missing-input
  default) and a hard missing-input error in the C backend.

`shape` reads only the input's shape metadata, never its element values, so it
is a trivial constant with respect to those values: its reverse-mode adjoint
contributes a **zero cotangent** to the input (like `const` / `load`, it does
not block AD — a loss that reads a runtime dim differentiates correctly, with
the shape factor contributing nothing). The C backend emits the read directly
(`t{input}->shape[axis]`), so a symbolic input axis is resolved from the actual
runtime input tensor rather than baked at codegen time. Under `--target hip` a
`grad` export host-falls-back to the C emitter (the scalar read is a host-side
metadata op); a `Shape` node reaching the HIP device-kernel path is rejected
loudly (`reject_unsupported_hip_ops`) and the Metal lane rejects it via its
emit-time `unsupported`-op arm. eval and C are the mandatory lanes.

> **[05-SHAPE-1]** A scalar `Shape` value SHALL be implemented by evaluation
> and the C target. A `Shape` node that reaches the HIP device-kernel path
> SHALL be rejected before codegen with an `unsupported_feature` diagnostic
> that directs the caller to the C target; it SHALL NOT be replaced by a
> constant, a default extent, or a stub kernel.

A `shape()` read whose `axis` is not a compile-time literal (a data- or
metadata-derived runtime axis) is not DAG-representable, because `RiscOp::Shape`
carries a compile-time `axis`. The **forward host evaluator** still resolves
such a read at runtime. Any path that forces DAG construction — notably
`grad` — fails **loud** with a clean, source-located lowering diagnostic
(`shape(tensor, axis)` requires a compile-time-constant `axis`, citing
chelis#616), rather than the pre-fix silent `Load { name: "shape" }`
fabrication (which produced a wrong/fabricated gradient in the eval lane and a
missing-input error in the C backend). Using the extent as a runtime
**movement-op bound** or **reshape target** (a `shrink`/`stride`/`pad` bound
or window count derived from a `shape()` value, and the integer arithmetic
feeding it) is the chelis#616 node-valued `RtDim` capability built on this
node; see §2.4.1.

### 2.6 Effectful Primitive

| Name | Signature | Semantics | AD / effect note |
|---|---|---|---|
| `dropout` | `(&tensor[D, f32], f32) -> tensor[D, f32]` | Zero elements according to a pseudorandom mask determined by the active `with seed(...)` handler and the dropout rate | Introduces `Random`. In the shipped evaluator/AD path, the mask is treated as fixed with respect to the handled seed so the backward pass reuses the same seeded dropout pattern. |
| `uniform_like` | `(&tensor[D, p], f32, f32) -> tensor[D, p]` | Create a tensor matching the input shape and float dtype `p`, filled by the deterministic affine sampler defined by [05-OP-8] under the active `with seed(...)` handler | Introduces `Random`. The template values are not observed; its adjoint is the zero cotangent. |
| `process_run` | `(String, List[String]) -> (Int64, String, String)` | Run an external program with the given argv and capture `(exit_code, stdout, stderr)`. Arguments are passed straight to the OS as argv (no shell, no interpolation), so a value in the args list cannot inject extra shell commands. A process killed by a signal reports exit code `-1`. | Introduces `Io`. Eval/test-only: implemented by the IR evaluator (`chelis eval` / `chelis test`); rejected by the C/HIP/Metal build backends with a clean diagnostic rather than a silent fallthrough. |

Operational note: the evaluator and lowering path implement seeded `dropout`, but
`chelis build` does not yet codegen it for the `c` or `hip` backend targets.

Operational note: `process_run` is an eval/test-only subprocess-exec primitive
(Hull subprocess support). It carries the `Io` effect and runs under the IR
evaluator. The compiled backends (`c`, `hip`, `metal`) deliberately reject any
program that applies `process_run` because a compiled artifact has no host
interpreter to reach the subprocess-exec path; the rejection is a build error,
not a silent zero. Full backend support (host-side `host_emit` lowering plus a
sandboxed runtime exec helper) is tracked in Chelis-Lang/chelis#267.

> **[05-OP-8]** `uniform_like(template, low, high) -> result` admits every
> active float template dtype `p` in spec/04 §1.1, requires `low` and `high`
> to be `f32`, and returns `tensor[D, p]` with the template's dimensions. For flat
> element index `i`, SplitMix64 over the handled seed and `i` supplies a
> 53-bit unit value `u` in `[0, 1)`. For `p = f64`, the element is the one
> `f64` fused multiply-add `fma(high_f64 - low_f64, u, low_f64)`, where the
> two bounds are widened exactly from their stored `f32` values. For
> `p = f32`, the element is the one `f32` fused multiply-add
> `fma(high - low, f32(u), low)`. For `p = f16` or `bf16`, that same `f32`
> result is rounded exactly once to `p`. The operation introduces `Random`,
> does not observe the template's element values, and contributes a zero
> cotangent to the template. It has no accumulator parameter.

#### Seed determinism atom

Transitional blockquote authority per `spec/design/spec_provenance.md` §C1,
matching the §7/§8 atoms of this file. The block remains normative until it is
selected for fixture-proven migration through the pinned Buoy shell-side
integration in chelis#733 Phase 1; no semantic revision is embedded here.

> **[05-RNG-1]** For a fixed compiler version and target, evaluating a
> `with seed(N)` program twice SHALL yield byte-identical output, and two
> distinct accepted seeds SHALL yield distinct streams, in every lane. The
> RNG is not cryptographic: streams are decorrelated only up to the
> SplitMix64 mixing - in particular the per-call counter and per-element
> index enter the hash symmetrically.

*(Measured true with no exceptions in the eval and compiled-C host lanes -
byte-identical run-to-run and across separate programs, distinct seeds
diverge - by the chelis#735 cross-lane seed sweep and its re-sweep. This atom
is the unconditional per-lane determinism-at-rest guarantee only. Cross-lane
stream identity - eval and the compiled-C host lane producing the same draw
sequence - is a separate, still-parked atom, held behind chelis#731 Phase 1's
out-of-range/unsuffixed seed-literal diagnostic; the GPU/kernel lanes are the
chelis#736 follow-on. See chelis#735 for the sweep evidence and the decided
contract.)*

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

### 3.2 Comparison and Logical Operations

| Name | Lowering to RISC |
|---|---|
| `eq(a, b)` | `not(or(cmplt(a, b), cmplt(b, a)))` — neither less than the other |
| `neq(a, b)` | `or(cmplt(a, b), cmplt(b, a))` |
| `gt(a, b)` | `cmplt(b, a)` |
| `gte(a, b)` | `not(cmplt(a, b))` |
| `lte(a, b)` | `not(cmplt(b, a))` |

These lowerings define result values over already-evaluated operands. They
never reorder the evaluation of the operand expressions themselves:
`gt(a, b)` evaluates `a` before `b` like every application
(`spec/03-deep-syntax.md` §4.4), and only the value computation reads the
operands in `cmplt(b, a)` order.

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

Logical operations do not alias arithmetic primitives. An implementation may
use an internal representation-specific lowering only when it preserves the
three atoms above and never admits `bool` to a numeric capability or kernel.
(Not fully implemented; see chelis#1284.)

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

**Current implementation note:** the type checker currently also accepts a
`normalize(x)` convenience name.
It is **not** part of the stable Tier 2 surface yet because its lowering semantics are
not specified here and there is no corresponding IR check lowering rule.
Do not treat `normalize` as a stable specified built-in until this document and the IR
lowering are aligned.

### 3.5 Lowering Helpers And Sparse Implementation Nodes

The following names appear in lowering narratives (§4) as pseudocode or
pattern-matched operations. Most decompose into Tier 1 primitives.
(`cos` was formerly listed here as `sin(add(x, const(π/2)))`; it is now a
first-class unary primitive `RiscOp::Cos` — see §2.2 — alongside `tan`,
`atan`, `abs`, `floor`, and `ceil`, none of which decompose.)

| Helper | Decomposes to |
|---|---|
| `argmax(x, axis)` | comparison chain via `cmplt` + `max_elem` |
| `gather(x, idx, axis)` | one-hot encoding via `reshape`, `expand`, `mul`, `sum` |
| `im2col(x, kh, kw, ...)` | `stride`, `pad`, `reshape`, `permute` |
| `where(cond, a, b)` | Element-wise selection of `a` where `cond` is true and `b` where it is false; the boolean condition is not converted to or combined through a numeric dtype |

Implementation note: the compiler now also has first-class specialized sparse
IR nodes `RiscOp::Gather { axis }`, `RiscOp::ScatterAdd { axis }`,
`RiscOp::Scatter { axis }`, and `RiscOp::ScatterElements { axis }` (the
element-wise ONNX `ScatterElements`, §3.5.1), with evaluator, verifier, AD,
C/HIP backend, and wire-schema support. Tensor-lane Surf `gather(values, indices, axis)` lowers
directly to `RiscOp::Gather` in the current implementation, avoiding the host
runtime call and the dense one-hot materialization. The tensor-lane Surf
builtin `scatter_replace(base, indices, updates, axis)` lowers directly to
`RiscOp::Scatter` for the last-write-wins case. The shared specialization pass
also recognizes the internal `RiscOp::OneHot { vocab } + Expand + Mul + Sum`
gather tree and collapses it before DCE/codegen. Arbitrary historical const/eq
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

### 3.6 Host-Runtime Builders

The following helper is **host-runtime only**. It runs inside the
`chelis test` / `chelis eval` interpreter and produces a tensor without
going through a Surf `List` intermediate. It is not in the RISC DAG
and has no AD adjoint; differentiable code must build its accumulator
state through the tensor-lane primitives in §2.

> **[05-HOST-1]** A host-runtime builder SHALL be rejected when a compiled
> target is requested. It SHALL NOT be lowered to a target stub, default
> value, or null pointer. The diagnostic SHALL direct the caller to the host
> evaluator or to an equivalent composition of tensor-lane primitives.

| Name | Signature | Semantics |
|---|---|---|
| `tensor_scan` | `(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]` | Iteratively apply `fn(prev, i)` for `i in 0..n` and collect the `n` resulting values into a rank-1 tensor whose precision matches `T`. |

`T` must be a scalar primitive (`int8`..`int64`, `f16`..`f64`,
`bool`). The output is owned, contiguous, rank-1, and its
precision equals the dtype of `initial`. The iteration order is the
positional integer sequence `0, 1, ..., n - 1`.

Precision caveat: the host-runtime interpreter stores every scalar — the
running accumulator included, not only the emitted tensor elements — as
an `f64` (`crates/chelis-compiler-api/src/runtime.rs`
`ScalarBits::as_f64`), so a `T = int64` accumulator is exact only up to
2^53; integer magnitudes beyond that lose their low bits, matching
IEEE-754 double semantics and the behavior of every other host-runtime
tensor builder. This is not specific to `tensor_scan`. Because the
*accumulator itself* is f64-backed, the loss is not confined to the
final stored elements: if `fn` drives the accumulator above 2^53 at any
step, that step rounds and every subsequent step folds the rounded value
forward, so a scan whose values transiently exceed 2^53 is wrong even
where the final element lands back inside the exact range. Concretely,
three `+1` steps from 2^53 yield `[2^53, 2^53, 2^53]` rather than
`[2^53+1, 2^53+2, 2^53+3]`, because 2^53+1 is unrepresentable and the
rounded accumulator carries forward. The init-style use cases that
motivate the helper (LCG-driven Glorot weights bounded by the modulus,
positional/index sequences, learned-schedule precompute) all stay within
2^53 at every step, so the caveat is documented rather than guarded.

`tensor_scan` exists because the host-runtime interpreter has no
tail-call optimization: right-recursive Surf list builds of more than
~10000 elements overflow the worker stack (Chelis-Lang/chelis#257),
and the chunked / fold workaround patterns hit an O(n²) `concat`
wall well below the 30k–40k-element regime that init-style use cases
(LCG-driven Glorot weights, positional embedding precompute, learned
schedule precompute) need. `tensor_scan` runs the loop on the host
in Rust, so the worker stack is constant in `n`.

The builtin is **not** wired into `chelis build` for the `c` or `hip`
backend target. A program that calls `tensor_scan` at top-level, inside
a higher-order callback body (`map`/`fold`/`filter`/`scan`/`partition`/
`flat_map`), or inside any top-level function — *whether or not that
function is reachable from the build entry* — is rejected at compile
time with a `tensor_scan`-tagged `unsupported_feature` diagnostic that
points back to this section. The rejection is enforced in
`crates/chelis-compiler-api/src/compiler.rs::reject_host_only_builtins`,
which walks every top-level binding value, every function body, and
every inline callback body, so the C/HIP emitters never see a
`tensor_scan` call; previously the C host emitter silently produced
`__binding_0_value = /* unsupported builtin tensor_scan */ 0` and the
compiled program returned garbage. Programs that need a compiled scan
over a tensor must compose `expand` + the tensor-lane primitives
directly.

This build-time walk is intentionally **whole-program**, in contrast to
the *reachability-scoped* AD/`vmap` rejection below. The asymmetry is
deliberate and tracks each backend's emission scope: the C/HIP host
emitter (`chelis_backend_c::host_emit`) emits *every* top-level function
unconditionally with no dead-code pruning, so a `tensor_scan` call inside
an otherwise-unreferenced helper still reaches the emitter and would
produce the silent stub above. Narrowing the build guard to the entry's
reachable call graph while the emitter still emits the whole program
would let that broken stub ship in a build the user believes succeeded.
The AD/`vmap` guard can scope to the transform target because the AD
lowering only ever touches that target's subgraph. If backend
dead-function pruning is added later, the build guard can be narrowed to
the emitted set in lockstep.

A future Tier 1 primitive can replace this host-only helper once the
RISC DAG admits higher-order tensor primitives. Until that lands,
`tensor_scan` is the recommended path for building per-index tensor
data at `chelis test` / `chelis eval` time without paying the
right-recursive list cost.

**Negative parity for `tensor_scan`**: a non-callable second argument,
a wrong-arity call, a negative `n`, or a callback that returns a
different dtype than the initial value's dtype are rejected with
`tensor_scan`-tagged diagnostics (the first three at type-check
time, the dtype-mismatch as a belt-and-suspenders runtime guard).
A `chelis build --target c` or `--target hip` of a program that
calls `tensor_scan` is rejected at compile time, and `grad(...)`
/ `vmap(...)` over a function whose body reaches `tensor_scan` is
rejected at the host-runtime transform boundary with a tagged error
referencing this section (the diagnostic verb is transform-specific:
`grad` reports it cannot *differentiate through* the builtin, `vmap`
that it cannot *vectorize over* it). The AD-boundary rejection is
*reachability*-scoped: it fires only when `tensor_scan` is reachable
from the transform target (the applied function and the def bodies it
calls), so an unrelated top-level binding that happens to call
`tensor_scan` does not falsely block a differentiable transform. The
acceptance tests in
`crates/chelis-compiler-api/tests/issue_257_tensor_scan_host_runtime.rs`
pin each of these — the reachability-scoping case, separate `grad` and
`vmap` rejections, the higher-order-callback build rejection, and the
whole-program build rejection of a `tensor_scan` call in an
entry-unreachable helper — alongside the positive 8/20000/40000-element
cases.

### 3.6.1 The `test_*` assertion family (host-only)

The `Test`-effect assertion builtins — `test_assert`, `test_assert_eq_f32`,
`test_assert_eq_int`, `test_assert_eq_bool`, `test_assert_eq_string`,
`test_assert_close_tensor`, and `test_assert_eq_tensor_int64` — are
**host-only**. They run inside the `chelis test` / `chelis eval` interpreter,
where an assertion evaluates its condition and aborts the run with a branded
label on failure. They have **no compiled-lane emission arm**.

Their build rejection is enforced *differently* from `tensor_scan`'s. They
are **not** in the whole-program pre-codegen host-only gate
(`reject_host_only_builtins`; `HOST_ONLY_BUILTINS = ["tensor_scan"]`).
Instead the rejection is the C host emitter's unsupported-builtin catch-all
(`chelis_backend_c::host_emit`): a `test_*` call in a function the emitter
emits is a build error with a branded `unsupported` diagnostic (chelis#703
class), never a silently-inert assertion. That makes it **liveness-scoped** —
the opposite of `tensor_scan`'s whole-program gate. A `test_*` in a function
reachable from the build entry, or in *any* function of an entry-less
object-mode module such as `Std.Test` itself (where every function is live),
is rejected at `chelis build`; a `test_*` in an entry-unreachable helper is
emitted as an abort stub and the build succeeds (the stub aborts only if
reached). Before the loud-unsupported sweep the emitter compiled every such
call to a `/* unsupported builtin test_assert */ 0` stub, so a compiled test
asserted nothing; that silent stub is gone.

Consequently the `Test`-effect wrappers in `Std.Test` (`assert_true`,
`assert_eq`, `assert_close`, `assert_shape`, `fail`, … — each a thin
`test_assert*` call) are eval/check-only: `chelis check` and `chelis test`
accept them, and `chelis build` of the module rejects them (it is
object-mode, so every wrapper is live). The compiled-lane arm — a compiled
binary that can *fail its own assertions* — requires real C assertion helpers
(compare + branded abort on mismatch) and is tracked by chelis#796.
(Separately, promoting `test_*` into the pre-codegen host-only gate would only
make the build reject them *whole-program* like `tensor_scan`, not make them
assertable — a rejection-cleanliness change, not the compiled-lane arm.)
Until the compiled-lane helpers land, assertions are an eval-lane contract.

### 3.6.2 Sequence-padding builders

> **[05-OP-9]** `pad_sequences(sequences: List[List[T]], pad: T) ->
> tensor[len(sequences), width, T]` admits exactly the active numeric
> primitive dtypes `T` in spec/04 §1.1, with arithmetic width governed by
> [04-NUM-8]; `bool`, `string`, and every reserved dtype are type errors.
> `width` is the greatest source-row length, or
> zero when the outer list is empty. Result element `(r, c)` is
> `sequences[r][c]` when `c < len(sequences[r])`, and `pad` otherwise.
> Every source and padding element is moved at its declared dtype `T` with
> no arithmetic, widening, narrowing, or other rounding. The operation is
> non-differentiable because its host `List` input carries no adjoint, and
> it has no accumulator rule.

> **[05-OP-10]** `pad_sequences_to(sequences: List[List[T]], width: int64,
> pad: T) -> tensor[len(sequences), width, T]` has the same dtype,
> element-movement, non-differentiability, and no-accumulator rules as
> [05-OP-9]. `width` SHALL be non-negative. Result element `(r, c)` for
> `0 <= c < width` is `sequences[r][c]` when that source element exists,
> and `pad` otherwise; source elements at index `width` or beyond do not
> appear in the result.

### 3.6.3 Canonical value-to-string conversion

> **[05-OP-25]** `to_string(value) -> result` borrows exactly one value
> without consuming it and returns `string`. It admits exactly an active
> numeric, `bool`, or `string` scalar; a tensor whose element dtype is an
> active numeric dtype or `bool`; or `List[T]` when `T` is recursively
> admitted by this rule. Every other value type is a type error. A `string`
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
> tensor truncation. String elements are inserted verbatim, without quoting or
> escaping: this is a non-injective display form, not a serialization. Every
> lane produces byte-identical text for the same admitted stored value. The
> operation is pure, performs no arithmetic or dtype conversion, is
> non-differentiable (`grad` rejects it), and has no accumulator.

*(Not fully implemented; see chelis#1282 and chelis#1059.)*

### 3.7 Host-Lane Data I/O Numeric Operations (Eval-Only; chelis#890 / chelis#903)

The JSON and CSV I/O builtin families (`parse_json`/`to_json`, the
`json_*` accessors, the `j*` constructors, `round_to`, `parse_csv`/
`to_csv`, and the `csv_*` accessors; surface inventory in
`docs/CHELIS_SURFACE.md` §3.8–§3.9) are **host-runtime only**: they run
under `chelis eval` / `chelis test`, and every compiled target rejects a
program that reaches one, whole-program, through the §7 `Unsupported`
channel (`chelis_ir::host::EVAL_ONLY_HOST_BUILTINS`). None of these
operations is in the RISC DAG and none carries an AD adjoint — the whole
family is **non-differentiable**, and no operation in it accumulates, so
no accumulator rule applies anywhere in this section unless an atom
states one.

The atoms below are the normative numeric authority for the family, in
the sense `spec/design/capability_table.md` §New numeric ops requires: a
callable in these families has exactly the numeric behavior its
governing atom states, and a numeric behavior no atom governs does not
ship. The prelude `Json` ADT's numeric capacity (`JInt int64` beside
`JNum f64`) is decided by [05-OP-2].

#### Compiled-target rejection atom

The rejection rule this section's opening paragraph states, as the
family's citable authority. [05-HOST-1] (§3.6) governs only the
host-runtime tensor BUILDERS; this atom governs the eval-only data-I/O
family, whose members have no tensor-lane composition equivalent.

> **[05-HOST-2]** An eval-only data-I/O builtin (this section's JSON and
> CSV families, `round_to`, and `process_run`; the
> `chelis_ir::host::EVAL_ONLY_HOST_BUILTINS` roster) SHALL be rejected
> when a compiled target is requested, whole-program, through the §7
> `Unsupported` channel. It SHALL NOT be lowered to a stub or default
> value. The diagnostic SHALL direct the caller to the host evaluator
> (`chelis eval` / `chelis test`).

(chelis#1184 tracks a `chelis build` / `chelis check` divergence on this rule.)

#### Decimal rounding atom

Transitional blockquote authority per `spec/design/spec_provenance.md`
§C1, matching the §7/§8 atoms of this file.

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
> | `f16`, `bf16` | **unsupported**: loud rejection at check and eval | — |
> | integer, bool, tensor | type error | — |
>
> No lane may compute an operand at any width other than the operand's
> own ([04-NUM-8]; the decimal rounding itself is exact, so the single
> finalization is the only rounding). `places` is a value-domain integer
> accepted at any integer storage width; values outside `0..=100` are a
> loud error (negative `places` has no defined semantics). A non-finite
> operand passes through unchanged. `round_to` is non-differentiable and
> has no accumulator.

#### Numeric ingestion atom

> **[05-OP-2]** Ingestion preserves the source format's numeric
> distinctions ([04-NUM-11]; `spec/design/dtype_semantics.md` §C6 "type
> the boundary"). A JSON number token containing `.`, `e`, or `E` SHALL
> ingest as `JNum` carrying the correctly-rounded f64 of the token; any
> other number token SHALL ingest as `JInt` carrying its exact int64
> value. An integer token outside int64 range falls back to the
> correctly-rounded `JNum` — the ONE sanctioned lossy ingestion case,
> which every surface documenting the family SHALL name — and a token
> whose f64 image is non-finite is a loud error. CSV cells are TEXT at
> parse time (no inferred numeric type); numeric meaning is assigned
> only by an accessor, under the JSON number grammar with surrounding
> ASCII space/tab tolerated: float accessors accept the full grammar,
> integer accessors accept only its integer subset (a `.`/`e`/`E`
> production refuses loudly, naming the float accessor), and an
> integer cell outside int64 range is a loud Overflow-kind error, never
> an f64 fallback. An empty or non-conforming cell is a loud error in
> every numeric accessor — no NaN, no default, no skip.

#### Exact read atom

> **[05-OP-3]** An integer read (`json_int`, `json_ints`, `csv_int`,
> `csv_ints`) returns the stored int64 EXACTLY and SHALL refuse a float
> value, naming the corresponding float accessor — never truncating,
> never rounding. A float read (`json_f64`, `json_f64s`, `csv_f64`,
> `csv_f64s`) returns a stored f64 exactly; applied to a stored int64 it
> performs the NAMED lossy int64-to-f64 widening (exact for magnitudes
> at or below 2^53), which every surface documenting the family SHALL
> state together with `json_int`/`csv_int` as the exact alternative.
> Structural counts (`csv_nrows`) are exact int64. Every missing path,
> missing column, out-of-range row, or type mismatch is a loud error.

#### Exact construction atom

> **[05-OP-4]** `jnum` accepts exactly `f64` and `jint` exactly `int64`;
> every other operand width is a loud error at check and at eval, naming
> the suffix/cast remedy. No construction path widens or narrows a
> numeric value: the constructed document feeds the byte-exact
> serialization channel of [05-OP-5], and a silent f32-to-f64 widening
> would serialize the f32 literal's image (`0.1f32` as
> `0.10000000149011612`) rather than the value the program stated.

#### Numeric serialization atom

> **[05-OP-5]** Serialization (`to_json`, `to_csv`) emits a stored int64
> as its exact decimal digits with no fractional part and no float
> round-trip, and a stored f64 through the [05-OBS-1] shortest-
> round-trip channel (`format_element` at `f64`; the §8.1 grammar —
> every finite emission parses back to the identical f64 and is a valid
> JSON number token). A non-finite f64 is a loud serialization error in
> both formats. Equal documents serialize to identical bytes.

*(Implemented by `crates/chelis-compiler-api/src/runtime/json.rs` and
`runtime/csv.rs`, with the checker contracts in
`crates/chelis-types/src/infer/app_hostio.rs`; the acceptance surface is
`crates/chelis-cli/tests/json_io.rs` / `csv_io.rs` plus the runtime unit
and pipeline suites. The f32 lane of [05-OP-1] is pinned by a
width-divergence test — `2.0025f32` rounds to `2.003` while `2.0025f64`
rounds to `2.002` — so a widen-to-f64 implementation cannot pass. The
non-ASCII-emission, escape, duplicate-key, and depth rules of the JSON
text layer are documented at the surface (`docs/CHELIS_SURFACE.md`
§3.8–§3.9); they are text-layer behavior, not numeric semantics, and are
deliberately not frozen here.)*

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
> converts through another numeric dtype. `bool`, `string`, deferred, and
> non-integer targets are type errors. It has no accumulator and is
> non-differentiable: `grad` rejects it.
>
> **[05-OP-24]** `cast_wrap(source, target) -> result` admits an active
> signed-integer source and signed-integer target on a scalar or tensor
> surface. It preserves the source surface and tensor dimensions and returns
> the unique signed target-width representative congruent to the exact stored
> source modulo `2^target_width`. It never traps for overflow, saturates, or
> converts through a float dtype. Float, `bool`, `string`, deferred, and
> non-integer targets are type errors. It has no accumulator and is
> non-differentiable: `grad` rejects it.

There is no `cast_round` operation. A program that wants rounding followed by
checked conversion spells `cast(round(source), target)`, so the rounding and
checked-cast boundaries remain independently observable.

*(Not fully implemented; see chelis#759.)*

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

**Zero-cotangent predicates and sources.** `cmplt`, `is_nan`, `is_finite`,
`is_infinite`, `const`, and `load` contribute zero cotangent. This permits a
predicate to participate in a differentiable guard without pretending that
the predicate itself has a useful derivative.

**Structural rejections.** On float operands, `floor`, `ceil`, and `round` are
piecewise constant and `grad` rejects them with an
`AdRejectionReason::PiecewiseConstant` error rather than silently returning a
zero gradient. `cast_trunc`, `cast_saturate`, `cast_wrap`, `and`, `or`, `not`,
`argmax_reduce`, and `argmin_reduce` likewise reject `grad` under their atoms.
The `wrap_*` operations and integer reduction/unary forms are forward-only
because integer values do not carry cotangents. Integer `floor`, `ceil`, and
`round` are exact identities and may be erased before AD. The `Diff` effect
reports a non-differentiable operation before execution.

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
// `acc_t` is the accumulator type resolved per §2.3 / spec/04 §5.7.1,
// NOT unconditionally `float`. The four lanes are the stride-4 ILP
// cascade §2.3 pins; a single-accumulator left-fold is NOT conforming
// and produces different f32 bits.
for (int i = 0; i < outer; i++)
  for (int j = 0; j < inner; j++) {
    acc_t acc[4] = {0, 0, 0, 0};
    for (int k = 0; k < axis_size; k++)
      acc[k & 3] += input[i * axis_size * inner + k * inner + j];
    output[i * inner + j] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
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
accumulator defaults. (This paragraph formerly read "These C implementations
are the ground truth", which had already gone stale against §2.3's cascade;
corrected 2026-07-28.)

The GPU backend must satisfy [05-OBS-3]'s per-operation, per-arithmetic-width
agreement table. Blanket `1e-6` (f32) / `1e-12` (f64) bounds are not a
conforming cross-lane oracle.

---

## 7. The Unsupported-Case Response Contract

This section governs how an unsupported case is reported. It does not decide
which operations, dtypes, targets, or parameter shapes are supported; those
decisions belong to their owning numbered-spec atoms. The implementation and
delivery design is `spec/design/loud_unsupported.md`.

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

> **[05-UNS-5]** An unsupported diagnostic SHALL carry the authority
> for its rejection: a deliberately unsupported case cites the spec
> atom that decides it, and a not-yet-implemented case cites its
> tracking issue. The two SHALL be distinguishable at the diagnostic
> surface, and a rejection carrying neither citation is a defect.

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
> own dtype width, and all exits within a lane SHALL agree with each
> other and with the stored bits.

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
