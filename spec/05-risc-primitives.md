# spec/05-risc-primitives.md — Chelis RISC Primitive Semantics

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
lowering, so primitive DAG nodes and backend kernels use the same value model.
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

The primitive surface includes
`einsum`, `concat` / `split`, `gather` / `scatter`, `where`, `cumsum`, `sort`,
`diagonal` / `trace`, and `clamp`. `School.Nn.Embedding` is the named library
surface over `gather`.

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
| `cmplt` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,bool]` | Element-wise less-than comparison | Non-differentiable (zero gradient) |
| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | Element-wise maximum | `(g * (x >= y), g * (x < y))` — gradient flows to the max input |

**`div` semantics.** `div(a, b)`
is **restricted to float operands** (f32, f64, f16, bf16) and
lowers to the target's native floating `/` operator with IEEE-754
semantics. Corner cases follow IEEE: `1/0 = +inf`, `1/-0 = -inf`,
`0/0 = NaN`, `1/-1 = -1`, `(any non-NaN) / -2.0` yields the
algebraic value.

`div(int_tensor, int_tensor)` is a **type error** (the `/`
operator on integer operands is likewise rejected, because `/`
desugars to `div`). The diagnostic cites this section and points
at `floor_div` / `trunc_div`. Integer division uses the two explicit,
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
(applying it to floats is a type error; use `floor_div` plus
`floor`, or a `cast`, for a float truncating quotient).
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

**Precision rule:** Both inputs must have the same precision `p`. Output has the same precision. Exception: `cmplt` returns `bool` regardless of input precision. Additional restrictions: `div` admits only float precisions (integer operands are a type error citing this section); `trunc_div` admits only integer precisions (float operands are a type error); `floor_div` admits both integer and float precisions.

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
| `abs` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise absolute value | `g * sign(x)` (sign = `(x > 0) - (x < 0)`; 0 at x = 0) |
| `floor` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise floor | non-differentiable (piecewise constant); `grad` rejects it |
| `ceil` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise ceil | non-differentiable (piecewise constant); `grad` rejects it |
| `round` | `(&tensor[D,p]) -> tensor[D,p]` | Element-wise round to nearest, ties to even (IEEE-754 roundTiesToEven / banker's rounding) | non-differentiable (piecewise constant); `grad` rejects it |

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

The reduction axis must be a compile-time constant (a literal, or a
`cast(N, int32)`-wrapped literal). Because the output shape is "remove
the dimension at position `axis`", the type checker cannot determine
which dimension is dropped from a runtime axis value. A reduction whose
axis is a runtime expression (for example a function-parameter `int32`)
is rejected at the reduction call site with a diagnostic naming the
compile-time-constant requirement, rather than leaving the output shape
unresolved (chelis#259). The same constraint and diagnostic apply to
`expand`'s insert axis.

**Output dimensions:** The dimension at position `axis` is removed. All other dimensions are preserved.

**Runtime-derived operand rank (chelis#320).** When a reduction
(`max_reduce`) or `gather` is applied to a windowing/stacking
intermediate whose IR node lowered without a static tensor type (a
rank-0 placeholder), the lowering recovers the operand's rank from the
ascribed result type — for a reduction the operand rank is the result
rank plus one; for `gather` it is `result_rank - indices_rank + 1` —
and re-inserts the reduced/gathered axis as a runtime-derived symbolic
dim. This lets `grad` differentiate a windowed reduce/gather (the
pooling/im2col pattern) instead of raising "axis out of range for an
operand of rank 0"; the symbolic axis resolves from the operand's
runtime shape at evaluation time.

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
narrow integer rows widen their result, for overflow safety.

There is no implicit precision promotion: omitting the parameter resolves to
the documented default before lowering. The IR `RiscOp::ReduceSum` node
always carries a populated accumulator-precision field. Programs that
explicitly request a narrower-than-default accumulator are a type error per
§5.7.1.

`max_reduce` does not take an accumulator parameter. Max is order-preserving
and does not lose precision the way a long sum does, so the result element
type matches the operand element type.

### 2.3.1 Windowed Reduction

| Name | Signature | Semantics | AD adjoint |
|---|---|---|---|
| `reduce_window_max` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int32], strides: List[int32]) -> tensor[..., d1', ..., dn', p]` | Strided windowed max over the last `n` axes | Subgradient: each window's `g` flows to every position equal to that window's max (ties distribute, as `max_reduce`); accumulated over overlapping windows |
| `reduce_window_min` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int32], strides: List[int32]) -> tensor[..., d1', ..., dn', p]` | Strided windowed min over the last `n` axes | Subgradient: each window's `g` flows to every position equal to that window's min (ties distribute, as `min_reduce`); accumulated over overlapping windows |
| `reduce_window_sum` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int32], strides: List[int32]) -> tensor[..., d1', ..., dn', p]` | Strided windowed sum over the last `n` axes | Each window-source position receives the owning window's `g` (overlap-add over windows covering it) |
| `reduce_window_mean` | `(&tensor[..., d1, ..., dn, p], window_shape: List[int32], strides: List[int32]) -> tensor[..., d1', ..., dn', p]` | Strided windowed mean over the last `n` axes | As `sum`, with each contribution scaled by `1 / window_volume` |

**Four primitives, one IR family.** The public surface follows the same
pattern as `max_reduce`, `min_reduce`, `prod_reduce`, `argmax_reduce`, and
`argmin_reduce`: reducer choice is encoded in the function name, not a runtime
argument. The IR carries a single
`RiscOp::ReduceWindow { reducer, window_shape, strides }` node whose
`reducer` field selects `Max` / `Min` / `Sum` / `Mean`; the four Surf
builtins differ only in which `ReduceWindowKind` they emit.

**Padding mode: Valid only.** Output spatial extent per windowed axis is
`floor((input_dim - window) / stride) + 1`. `Same`-padding (with
`ceil(input_dim / stride)` output and zero / `-inf` fill at the
boundary) is not part of these primitives; programs that need that
behavior should pad explicitly with `pad(x, ..., fill)` before
calling `reduce_window_*`.

**Shape contract.**

- `window_shape` and `strides` are int32 lists of equal length
  `n >= 1`.
- The trailing `n` axes of the input are the windowed axes. The
  leading `rank(input) - n` axes pass through unchanged.
- Each windowed entry must be a positive int32. `window_shape[i] >= 1`
  and `strides[i] >= 1`.
- The output rank equals the input rank. Leading dims match the
  input; trailing dim `i` is
  `floor((input_dims[rank - n + i] - window_shape[i]) / strides[i]) + 1`.
  When that formula yields a non-positive value the call is a type
  error (an empty window output is structurally meaningless under
  `Valid` padding).

**Lowering.** The IR `RiscOp::ReduceWindow` carries the full
`{reducer, window_shape, strides}` triple. Every execution lane lowers it
as a direct windowed loop nest; `Mean` is windowed `Sum` divided by the window
volume, computed inline rather than as a separate `Div` op. There is no Tier-2
to Tier-1 decomposition: `reduce_window_*` is a Tier-1 primitive in its
own right. The Surf `reduce_window_*` names are the public surface;
the IR node and backends share the single `ReduceWindow` lowering
path.

*Accumulation precision.* A windowed reduction accumulates at the
operand's ARITHMETIC WIDTH (`spec/04-type-system.md` [04-NUM-8]) in every
lane: `f32` operands accumulate in `f32`, `f64` in `f64`, `f16`/`bf16` at
`f32`, and integers exactly at their width. `reduce_window_*` takes no
accumulator parameter, so §5.7's widening does not apply to it and there
is no other authorized widening.

For integer operands, a window sum that leaves the operand dtype's
range traps per [04-NUM-3] - with occurrence governed by [04-NUM-12] -
where the GLOBAL reduction's §5.7.1 default would have widened, because
there is no parameter to request a wider window accumulator. If windowed
reductions over narrow integers become a real need, the resolution is
authoring an accumulator parameter for `reduce_window_*` on §5.7's
pattern, never silent widening.

*(Not fully implemented; see chelis#729.)*

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
- `Max` / `Min`: route each window's `g` to every position equal to that
  window's extreme — the windowed generalization of the `max_reduce` /
  `min_reduce` `eq`-mask subgradient, so ties distribute the full `g`
  (not a `1/k` share). `x` is read to locate the extreme.

Like the forward op, `ReduceWindowGrad` lowers directly in every execution
lane. Overlapping windows scatter-add into shared `din` positions. Second-order
AD through the adjoint itself is not defined.

**Output-dim formula.** For `stride > 1`,
`(input_dim - window + 1) / stride` under-counts (e.g.
`input=8, window=2, stride=2` gives `3` instead of the correct `4`
non-overlapping windows at positions `0, 2, 4, 6`). The normative formula
is `floor((input_dim - window) / stride) + 1`.

**Runtime-derived extents.** Windowed output extents are computed from the
concrete runtime input shape using the normative formula above. Ahead-of-time
lowering must preserve that dimension expression and allocate the computed
output extent; it must never reuse the input extent or silently mis-size the
result. A backend that cannot represent the expression rejects the program
under [05-UNS-1..6] without narrowing the primitive's dtype or shape contract.

**Reduction order (`sum` only).** `sum` evaluates the reduction with a
**stride-4 ILP cascade** — four independent accumulator lanes loaded
in round-robin (`acc[i & 3] += value[i]`), combined at the end as
`(acc0 + acc1) + (acc2 + acc3)`. This matches PyTorch's CPU
`row_sum` (`num_levels=4 ilp_factor=4`), so f32 `sum` is bit-exact
with `torch.sum(...)` for `n ≤ 16` on the reduced axis. NumPy's
`sum` uses a divide-and-conquer pairwise tree with 128-element
blocks — structurally different from the stride-4 cascade — so the
two coincide only by accident on specific inputs; chelis `sum` is
**not** in general bit-exact with `numpy.sum`. For `n > 16` the
result may differ from torch by up to ~1 ULP. The order is purely positional so the
algorithm is deterministic across runs and hosts; `#pragma omp
parallel for` is applied to the outer (output-element) loop only,
never the inner reduction.

Integer reductions are order-independent in VALUE when they complete, but
under [04-NUM-3]'s traps,
whether an intermediate leaves the accumulator's range is
order-dependent at range edges - `spec/04-type-system.md` [04-NUM-12]
defines trap occurrence relative to each lane's documented order,
including this cascade. The accumulator-precision rule
above is orthogonal to the reduction order: the lane type is the
accumulator type, and the final combine happens in the same
precision.

**Device reduction order.** A device backend may use a documented parallel tree
order. Cross-lane float differences are permitted only by [05-OBS-3]'s per-op
tolerance table; integer trap occurrence follows [04-NUM-12].

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
backend declares it inline at the owning op (`int name = <extent>;`) and the
evaluator binds it from the actual value mid-evaluation; a second site
computing a different value for the same symbol aborts/errs loudly (the
over-unification guard). Anonymous dims are not substitution keys, so a
signature symbol attaches positionally to the final op and each inner movement
op declares its own extent. The checker's movement typing matches: symbolic-dim
pass-through is identity-only (stride step 1 / zero pad; see
spec/04-type-system.md §4.7), so a non-identity movement axis types a
fresh runtime-guarded extent rather than repeating the input's symbol.
The guard remains the soundness floor for a genuinely CLAIMED symbol
equality (e.g. an explicit `-> tensor[n]` over `stride(x, 2)`) and for
checker imprecision.

The movement adjoints are runtime-capable on the same representation: the
`shrink` adjoint pads with `after = shape(x, axis) - end`, the `pad` adjoint
shrinks to `end = before + shape(x, axis)`, and the `stride` adjoint's
upsample cascade reads `m_a = shape(g, axis)` and trims to
`(0, shape(x, axis))` with a runtime `m_a * step` merge extent — all as
node-valued bounds over fresh `Shape`/arithmetic scalars. Bound scalars are a
**stop-gradient boundary**: they are index math, carry no cotangent, and do
not pull their producers (e.g. a window-count `floor_div`) into the
differentiability check. A runtime (node-valued) stride step uses the same
scatter-to-selected-positions adjoint as a static step and fills skipped
positions with zero.

Runtime movement bounds and reshape targets have identical semantics in every
execution lane. A backend unable to represent them rejects the program under
[05-UNS-1..6].

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
| `shape` | `(&tensor[d1,...,dn,p], axis: int32) -> int` | Runtime extent of the input along `axis`, as a rank-0 integer scalar. |

The Surf `shape(tensor, axis)` builtin types this read as an `int32` scalar.
Two lowering shapes exist, and they are distinct:

- **As an extent argument** to `expand` / `reshape`, a `shape()` read is folded
  into the movement node's runtime extent expression (the output dim), not
  materialized as a value node.
- **As a scalar VALUE** (used in arithmetic, a `mean` divisor, or any other
  value position), a `shape()` read lowers to a dedicated `RiscOp::Shape { axis }`
  node — a rank-0 integer scalar equal to the input's runtime extent along a
  compile-time-constant `axis`.

`shape` reads only the input's shape metadata, never its element values, so it
is a trivial constant with respect to those values: its reverse-mode adjoint
contributes a **zero cotangent** to the input (like `const` / `load`, it does
not block AD — a loss that reads a runtime dim differentiates correctly, with
the shape factor contributing nothing). Every backend resolves a symbolic input
axis from the actual runtime tensor metadata rather than baking it at codegen
time. Device builds execute this host-side metadata operation at the host boundary.

A `shape()` read whose `axis` is not a compile-time literal (a data- or
metadata-derived runtime axis) is not DAG-representable, because `RiscOp::Shape`
carries a compile-time `axis`. A path that forces DAG construction — notably
`grad` — fails **loudly** with a clean, source-located diagnostic that
`shape(tensor, axis)` requires a compile-time-constant `axis`. Using the extent as a runtime
**movement-op bound** or **reshape target** (a `shrink`/`stride`/`pad` bound
or window count derived from a `shape()` value, and the integer arithmetic
feeding it) uses the node-valued `RtDim` representation in §2.4.1.

### 2.6 Effectful Primitive

| Name | Signature | Semantics | AD / effect note |
|---|---|---|---|
| `dropout` | `(&tensor[D, f32], f32) -> tensor[D, f32]` | Zero elements according to a pseudorandom mask determined by the active `with seed(...)` handler and the dropout rate | Introduces `Random`. The mask is fixed with respect to the handled seed, so the backward pass reuses the same seeded pattern. |
| `uniform_like` | `(&tensor[D, f32], f32, f32) -> tensor[D, f32]` | Create a tensor matching the input shape, filled from a deterministic uniform distribution under the active `with seed(...)` handler | Introduces `Random`. |
| `process_run` | `(String, List[String]) -> (Int64, String, String)` | Run an external program with the given argv and capture `(exit_code, stdout, stderr)`. Arguments are passed straight to the OS as argv (no shell, no interpolation), so a value in the args list cannot inject extra shell commands. A process killed by a signal reports exit code `-1`. | Introduces `Io`. Compiled artifacts execute it through a sandboxed host-runtime boundary. |

#### Seed determinism atom

> **[05-RNG-1]** For a fixed compiler version and target, evaluating a
> `with seed(N)` program twice SHALL yield byte-identical output, and two
> distinct accepted seeds SHALL yield distinct streams, in every lane. The
> RNG is not cryptographic: streams are decorrelated only up to the
> SplitMix64 mixing - in particular the per-call counter and per-element
> index enter the hash symmetrically.

---

## 3. Derived Built-Ins (Tier 2)

These are convenience functions emitted by the desugarer. The compiler lowers them to RISC primitive compositions during IR construction. They are NOT in the RISC DAG — they exist in Deep AST only.

### 3.1 Arithmetic

| Name | Lowering to RISC |
|---|---|
| `sub(a, b)` | `add(a, neg(b))` |

Note: `div`, `neg`, and `recip` are Tier 1 RISC primitives (see §2.1, §2.2),
not Tier 2 derived built-ins.

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
| `mean(x, axis)` | `div(sum(x, axis), divisor)` where `divisor = const(dim_size)` for a concrete-extent axis, or the runtime count `sum(const(1.0, x.shape), axis)` when the reduced axis is a runtime-derived (`Named(_, None)`) extent (chelis#320) |
| `softmax(x, axis)` | See §4.2 |
| `linear(x, w, b)` | `add(matmul(x, w), b)` (with appropriate expand on b) |
| `cross_entropy(logits, labels)` | See §4.3 |
| `min_elem(a, b)` | `neg(max_elem(neg(a), neg(b)))` |

### 3.5 Lowering Helpers And Sparse Nodes

The following names appear in lowering narratives (§4) as pseudocode or
pattern-matched operations. Most decompose into Tier 1 primitives.
`cos` is a first-class unary primitive `RiscOp::Cos` — see §2.2 — alongside `tan`,
`atan`, `abs`, `floor`, and `ceil`, none of which decompose.

| Helper | Decomposes to |
|---|---|
| `argmax(x, axis)` | comparison chain via `cmplt` + `max_elem` |
| `gather(x, idx, axis)` | one-hot encoding via `reshape`, `expand`, `mul`, `sum` |
| `im2col(x, kh, kw, ...)` | `stride`, `pad`, `reshape`, `permute` |
| `where(cond, a, b)` | `add(mul(cond, a), mul(neg(cond), b))` assuming bool 0/1 |

Sparse operations use the distinct IR nodes `RiscOp::Gather { axis }`,
`RiscOp::ScatterAdd { axis }`, `RiscOp::Scatter { axis }`, and
`RiscOp::ScatterElements { axis }` (the element-wise ONNX operation in
§3.5.1). Every evaluator, verifier, AD pass, backend, and wire representation
must preserve their typed semantics. Tensor-lane `gather` and
`scatter_replace` lower directly to their corresponding nodes. A
specialization pass may recognize and replace the internal
`OneHot + Expand + Mul + Sum` gather composition only when it preserves the
original index operand.

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
| `Scatter { axis }` | last-write-wins (deterministic order rule below) | deterministic last-writer adjoint defined below |

**Deterministic-order rule for `Scatter`:** updates-tensor row-major
(C order) flat iteration. For each `i ∈ 0..updates.size` in
ascending flat-index order, the write
`target[..., indices[idx_pos(i)], ...] = updates[i]` occurs at step
`i`. When two updates target the same cell, the write with the
larger flat index in `updates` is the final value at that cell.
Every backend must observe this rule. Sequential execution is conforming; a
parallel realization must use a deterministic tie-breaker that selects the
maximum flat-index writer for each target cell.

**AD policy for `Scatter`:** let `g` be the output cotangent. For each output
cell written by one or more updates, the target cotangent is zero; every other
target cell receives `g` unchanged. An update receives the cotangent at its
destination exactly when it is the maximum flat-index writer for that cell;
every shadowed update receives zero. Indices are non-differentiable. Thus the
adjoint uses the same deterministic winner relation as the forward operation
and never distributes one cotangent among colliding updates.

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
flat iteration. A parallel implementation must select the maximum flat-index
writer for each destination.

**AD policy.** The adjoint is the element-wise form of the `Scatter` rule.
Written data cells receive zero cotangent and untouched data cells receive the
output cotangent. Each destination's maximum flat-index update receives that
destination's cotangent; shadowed updates receive zero. Indices are
non-differentiable.

The tensor-lane Surf builtin
`scatter_elements(data, indices, updates, axis)` lowers directly to
`RiscOp::ScatterElements`.

### 3.6 Host-Runtime Builders

The following helper is **host-runtime only**. It runs inside the
`chelis test` / `chelis eval` interpreter and produces a tensor without
going through a Surf `List` intermediate. It is not in the RISC DAG
and has no AD adjoint; differentiable code must build its accumulator
state through the tensor-lane primitives in §2.

| Name | Signature | Semantics |
|---|---|---|
| `tensor_scan` | `(initial: T, fn: (T, int64) -> T, n: int64) -> tensor[n, T]` | Iteratively apply `fn(prev, i)` for `i in 0..n` and collect the `n` resulting values into a rank-1 tensor whose precision matches `T`. |

`T` must be a scalar primitive (`int8`..`int64`, `f16`..`f64`,
`bool`). The output is owned, contiguous, rank-1, and its
precision equals the dtype of `initial`. The iteration order is the
positional integer sequence `0, 1, ..., n - 1`.

The accumulator and every emitted element obey the declared dtype's value and
arithmetic-width semantics from `spec/04-type-system.md` §9; an implementation
must not route them through `f64` or another untagged carrier. The host loop uses
constant stack space in `n`. Compiled artifacts execute the helper through their
host-runtime boundary.

A non-callable second argument, a wrong-arity call, a negative `n`, or a
callback that returns a different dtype than `initial` is rejected with a
`tensor_scan`-tagged diagnostic. `tensor_scan` has no AD adjoint and is not
vectorizable; `grad` and `vmap` reject it when it is reachable from the
transformation target. Unrelated definitions do not block a transformation.

### 3.6.1 The `test_*` Assertion Family

The `Test`-effect assertion builtins — `test_assert`, `test_assert_eq_f32`,
`test_assert_eq_int`, `test_assert_eq_bool`, `test_assert_eq_string`,
`test_assert_close_tensor`, and `test_assert_eq_tensor_int64` — are
`Test`-effect host operations. Every execution lane evaluates the assertion
and terminates the run with its branded label on failure. Device builds keep
the assertion at the host-runtime boundary. An assertion may never compile to
a no-op, zero value, or unreachable stub.

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
admitted in the matmul signature; see `spec/04-type-system.md` §5.7.2
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

**Non-differentiable primitives:** `cmplt`, `const`, `load` have zero gradient.
`floor`, `ceil`, and `round` are piecewise constant and `grad` rejects them with an
`AdRejectionReason::PiecewiseConstant` error rather than silently returning a zero
gradient. The type system detects through the `Diff` capability when `grad` is
applied to a function containing non-differentiable operations and report which
operations are the problem.

**Almost-everywhere differentiable:** `max_elem` (gradient is zero at the boundary where inputs are equal), `relu` via `max_elem(x, 0)` (gradient is zero at x=0). These are valid targets for `grad` — the subgradient convention (pick one side) is standard in ML.

**Second-order derivatives:** `grad(grad(f))` works if all operations in `f` have defined second-order adjoints.

---

## 6. Reference Implementations

For each RISC primitive, this section gives illustrative C-like pseudocode.

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
accumulator defaults.

Cross-lane numeric agreement is governed exclusively by [05-OBS-3].

---

## 7. The Unsupported-Case Response Contract

The HostType contract consumes checked type metadata, preserves named
polymorphism, inference identity, bottom, and exact dtype, and permits only a
resolved `ConcreteHostType` plus an authoritative target capability decision
to produce `HostAbiType`. Codegen accepts only the ABI vocabulary. This
contract owns failure representation; grounded dtype semantics and Table A/B
policy remain owned by `spec/04-type-system.md`. A target selects only exact
ABI representations; an accidentally widened value is not a supported cell.
Typed C callback parameters and direct statically-known callback arguments
cross a private callback-declarator path; general function values do not.
Function results, stored function values, and dynamically selected callables
return `Unsupported` before emission, and no function type maps to `void *`,
zero, or a raw call target.

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

*(Not fully implemented; see chelis#959.)*

> **[05-UNS-6]** The machine-facing kind of a diagnostic is drawn from
> a closed vocabulary with stable spellings; the build surface's
> spelling for this contract's rejections is `unsupported_feature`.
> A machine consumer SHALL be able to distinguish an unsupported-case
> rejection from an internal compiler error by kind alone. Producing
> this kind for anything other than a typed unsupported rejection, or
> a different kind for one, is a defect.

*(Not fully implemented; see chelis#959.)*

---

## 8. Observation And Formatting Contract

> **[05-OBS-1]** Every exit that renders a stored numeric value as text -
> `print`, `to_list`, diagnostics, the wire schema's rendering - SHALL
> emit text that parses back to exactly the stored bits at the value's
> own dtype width, and all exits within a lane SHALL agree with each
> other and with the stored bits.

*(Not fully implemented; see chelis#729 and chelis#864.)*

> **[05-OBS-2]** Integer dtypes SHALL print as integers with all digits
> exact; floats SHALL print the shortest string that round-trips at
> their own width; `bool` SHALL print `true`/`false` at every exit; the
> number grammar (digit selection, exponent form, special-value
> spellings) SHALL be identical across lanes and is pinned in §8.1.

*(Not fully implemented; see chelis#865.)*

> **[05-OBS-3]** Cross-lane VALUE differences are permitted only when the
> operation's normative definition states an explicit tolerance, and only
> within that bound. An operation with no stated tolerance has bound zero.
> `sqrt` SHALL be correctly rounded and has bound zero. Formatting
> differences are never within tolerance.

> **[05-OBS-4]** A scalar-typed value SHALL render as the bare scalar at
> every exit in both lanes, including as a top-level labeled root
> (`root = 0.1`, never `root = tensor(shape=[], data=[0.1])`). A rank-0
> tensor renders as its single element, bare: the
> `tensor(shape=[], data=[..])` wrapper is not an exit form. An
> implementation MAY realize scalar bindings through rank-0 tensors
> internally; that realization SHALL NOT leak into the observation
> channel.

*(Not fully implemented; see chelis#729.)*

> **[05-OBS-5]** Every exit in both lanes SHALL truncate tensor element
> rendering after 32 elements, marking the cut with `, ...` inside the
> `data=[..]` brackets. `to_list` and the wire schema never truncate:
> full-element fidelity is theirs.

> **[05-OBS-6]** Every root SHALL render with a `name = value` label at
> every exit in both lanes. The bare-when-single form is removed. Render
> order is manifest entry order. A lane that cannot produce a root it
> owes SHALL emit [05-UNS-1] naming that root, the lane, and the reason.

*(Not fully implemented; see chelis#1023.)*

### 8.1 The Number Grammar

The grammar is Rust `{:?}` (`Debug`) float formatting, normatively; `Display`
is not this grammar because it never emits e-notation:

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
  [04-NUM-1]'s single-rounding argument) yields the stored bits; their
  decimal/e-notation decision applies the same rendered-magnitude rule
  to the chosen digits, and a same-length candidate tie breaks to the
  numerically closest, then the even mantissa;
- integer dtypes print exact base-10 digits (i64 formatting, never
  through double).
