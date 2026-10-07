# Runtime and standard library

Chelis has compiler built-ins for tensor, scalar, collection, and host operations.
The compiler also bundles `chelis-std`, whose modules use the `Std` prefix. In a
Reef package or a standalone `.ch` file, import the names you need, for example
`import Std.Sort (sort)`; there is no separate standard-library install. The `Std` source modules and the
native runtime archive emitted by `chelis build` are different parts of the
runtime.

## Conventions for built-in operations

Built-ins are called by name without an import. These rules apply to every
tensor operation below:

- **No broadcasting, no promotion.** Operands of an element-wise operation have
  one dtype and identical dimensions. A scalar beside a tensor is a type error;
  give the scalar the tensor's shape with `insert(scalar_to_tensor(c), 0i32, shape(xs, 0i32))`.
  `add(1i32, 2i64)` is a type error; write the `cast` yourself.
- **Axes are `i32`, extents are `i64`.** An axis argument is an `i32` constant
  such as `1i32`; a negative axis counts from the end, so `-1i32` is the last
  axis. Sizes, bounds, padding, strides, and window shapes are `i64`, and a
  list of them is written with suffixes: `[3i64, 2i64]`. An unsuffixed list
  such as `[3, 2]` is a type error that names the fix.
- **Reduction axes are static.** A reduction, `expand`, or `insert` axis must be
  a literal, a `cast(N, i32)` literal, or a named dimension of the operand. A
  function parameter holding an axis is rejected at the call.
- **Built-ins borrow.** Passing a tensor to a built-in leaves it usable. Passing
  it to your own function with an owned parameter `x: tensor[...]` consumes it;
  declare `x: &tensor[...]` and call with `&x`, or pass `copy(x)`.
- **Failures.** A violation visible in the source is a type error from
  `chelis check`. One that depends on run-time values traps with the
  operation and dtype, as in `numeric trap: domain in gather at i64`. Integer
  arithmetic is checked: `add(127i8, 1i8)` traps `overflow in add at i8`.

```chelis-surf
module TensorSummary
values = to_tensor([1.0f32, 2.0f32, 3.0f32])
total = sum(values, 0i32)
```

`chelis eval --file summary.ch` prints:

```text
values = tensor(shape=[3], data=[1.0, 2.0, 3.0])
total = 6.0
```

### Arithmetic and comparison

Each operation accepts two scalars of one dtype or two same-shaped tensors of
one dtype, and returns that shape.

| Operation | Accepted dtypes | Result and failure behavior |
|---|---|---|
| `add`, `sub`, `mul` | signed integers, floats | Integer overflow traps. |
| `div(a, b)` | floats only | IEEE division: `1/0` is `inf`, `0/0` is NaN. Integer operands are a type error; use `floor_div` or `trunc_div`. |
| `floor_div(a, b)` | signed integers, floats | Rounds toward negative infinity: `floor_div(-7i32, 2i32)` is `-4`. An integer zero divisor traps `division by zero`. |
| `trunc_div(a, b)` | signed integers | Rounds toward zero: `trunc_div(-7i32, 2i32)` is `-3`. Zero divisor traps. |
| `mod(a, b)` | signed integers, floats | Remainder with the dividend's sign: `mod(-7i32, 2i32)` is `-1`; float `mod` is C `fmod`. |
| `max_elem`, `min_elem` | signed integers, floats | Element-wise extremum. A NaN operand wins (the first NaN); on a tie the first operand is returned. |
| `cmplt` or `lt`, `gt`, `gte`, `lte` | signed integers, floats | `bool` result. Any comparison with NaN is false. |
| `eq`, `neq` | numbers, `bool`, `string`, and `List`, tuple, `Dict`, `Option`, or data-type values | `bool` result, element-wise for tensors. NaN is unequal to everything; `neq` is true for NaN. |
| `and`, `or`, `not` | `bool` | Both operands are evaluated; there is no short-circuit. |

Unary operations keep the operand's shape and dtype:

- `neg` and `abs` accept signed integers and floats; integer `neg` and `abs` trap
  on the minimum value, such as `-128i8`.
- `recip`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tan`, `atan`, `erf`, and `erfc`
  accept floats only; `sqrt(4i32)` is a type error. Results follow IEEE:
  `recip(0.0)` is `inf` and `log` of a negative number is NaN.
- `floor`, `ceil`, and `round` accept floats and signed integers. On integers
  they are the identity. `round` breaks ties to even.
- `is_nan`, `is_finite`, and `is_infinite` accept floats and return `bool`.
- `relu`, `sigmoid`, `tanh`, `silu`, `gelu`, and `gelu_tanh` are float
  activations on scalars or tensors. `gelu` is the exact `x * Phi(x)` with `Phi`
  the standard normal CDF; `gelu_tanh` is the tanh approximation that GPT-2-style
  models use. `relu(-0.0)` is `-0.0`, and NaN propagates.
- `standard_normal_cdf(x)` is `Phi`, built from the correctly rounded `erfc`;
  `standard_normal_cdf(0.0f64)` is `0.5`. `Std.Contracts.normal_cdf` calls it.

### Reductions

| Signature | Contract |
|---|---|
| `sum(x, axes..., accumulator=p)` | Signed integers and floats. Removes every selected axis. `accumulator` sets the running dtype; see below. An empty axis gives zero. |
| `mean(x, axes...)` | Floats only. Sum, then one division by the axis extent. A zero-length axis traps `domain`. |
| `max_reduce(x, axes...)`, `min_reduce(x, axes...)` | Signed integers and floats; result keeps the dtype. The first NaN wins. A zero-length axis traps `domain`. |
| `prod_reduce(x, axes...)` | Signed integers and floats; result keeps the dtype. Integer overflow traps. An empty axis gives one. |
| `count(mask, axes...)` | `bool` tensor in, `i64` tensor out: the number of true elements. |
| `argmax_reduce(x, axis)`, `argmin_reduce(x, axis)` | One axis only. Returns `i64` indices: the lowest index of the extremum, or of the first NaN. |
| `softmax(x, axis)` | Float tensor; keeps the shape. Computed as `exp(x - max)` divided by its sum along `axis`. |

The result dimensions are the operand dimensions with the selected axes
removed; reducing every axis gives a rank-zero tensor, printed as a scalar.

Reduce several axes in one call by naming them. Named axes are the dimension
names in the operand's type, so the function must declare them, as `total`
does below. With positional axes, pass one axis per call and chain the calls.

`sum` takes an optional `accumulator=` dtype of the same kind as the operand and
at least as wide as the default. The defaults are `f32` for `f16`, `bf16`, and
`f32`; `f64` for `f64`; `i32` for `i8`, `i16`, and `i32`; and `i64` for `i64`.
The result has the accumulator dtype, except that `f16` and `bf16` results
return to the operand dtype. So `sum` over `i8` returns `i32`, and
`sum(x, 0i32, accumulator=f64)` over `f32` returns `f64`. A narrower accumulator
is a type error. The other reductions take no accumulator.

### Windowed reductions

`reduce_window_sum`, `reduce_window_mean`, `reduce_window_max`, and
`reduce_window_min` all have the signature
`(x, window_shape: List[i64], strides: List[i64])`.

- The two lists have the same length `n`, with `1 <= n <= rank(x)`. Every entry
  is a positive `i64`.
- The window covers the last `n` axes; leading axes pass through. The rank is
  unchanged.
- Padding is "valid" only. Windowed axis `i` has output extent
  `floor((extent - window[i]) / stride[i]) + 1`. A window larger than the axis is
  a type error when the extents are known and a `domain` trap otherwise. Pad
  explicitly with `pad` first if you want "same" output size.
- Sum, max, and min accept signed integers and floats; mean accepts floats. The
  result keeps the input dtype, and sums do not widen: an integer window sum that
  leaves the dtype's range traps. `f16` and `bf16` accumulate in `f32`.
- `chelis build` compiles windowed reductions on `f32` tensors and stops with an
  `unsupported` error naming the dtype for any other; cast to `f32` first.
  `chelis eval` and `chelis test` run every dtype above.

```chelis-surf
module Reductions
def total(x: &tensor[rows, cols, f32]) -> tensor[f32] = sum(x, rows, cols)
x = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])
row_sums = sum(x, 1i32)
col_means = mean(x, 0i32)
wide = sum(x, -1i32, accumulator=f64)
grand = total(&x)
best = argmax_reduce(x, 1i32)
pooled = reduce_window_max(x, [2i64, 2i64], [1i64, 1i64])
```

```text
x = tensor(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
row_sums = tensor(shape=[2], data=[6.0, 15.0])
col_means = tensor(shape=[3], data=[2.5, 3.5, 4.5])
wide = tensor(shape=[2], data=[6.0, 15.0])
grand = 21.0
best = tensor(shape=[2], data=[2, 2])
pooled = tensor(shape=[1, 2], data=[5.0, 6.0])
```

### Shape operations

None of these converts an element; every operation accepts every tensor dtype,
including `bool`.

| Signature | Contract |
|---|---|
| `reshape(x, shape: List[i64])` | Same row-major element sequence, new shape. The element counts must match. |
| `permute(x, a0, a1, ...)` | One `i32` argument per axis, forming a permutation of `0..rank`. Output axis `k` is input axis `a_k`. |
| `insert(x, axis, size: i64)` | Adds a new axis of extent `size` at `axis` (`0..rank`, so the end is allowed) and repeats the data along it. Rank grows by one. |
| `expand(x, axis, size: i64)` | Repeats an existing axis of extent one to `size`. Rank is unchanged. An axis whose extent is not one is an error. |
| `pad(x, padding: List[List[i64]], fill)` | One `[before, after]` pair per axis, each nonnegative. `fill` has the tensor's dtype. |
| `shrink(x, bounds: List[List[i64]])` | One `[start, end]` pair per axis, half-open, with `0 <= start <= end <= extent`. |
| `stride(x, s0, s1, ...)` | One positive `i64` step per axis; keeps every `s`-th element starting at index 0. |
| `concat(parts: List[tensor], axis)` | A nonempty list of tensors with one rank and dtype and equal extents off `axis`. Parts appear in list order. |
| `split(x, axis, sizes: List[i64])` | Returns a `List` of tensors. Sizes are nonnegative and must sum to the extent of `axis`; otherwise it traps `domain`. |

To align shapes for an element-wise operation, use `insert` to add an axis and
`expand` to widen an extent-one axis; there is no implicit broadcasting to do it
for you. The idiom `insert(b, axis, shape(x, axis))` repeats `b` to match `x`.

### Indexing

| Signature | Contract |
|---|---|
| `gather(x, indices, axis)` | Result shape is `x` dims before `axis`, then `indices` dims, then `x` dims after `axis`. `axis` must be an `i32` literal. |
| `scatter(base, indices, updates, axis, mode)` | Writes `updates` into a copy of `base`. `updates` has the shape `gather` would return for the same `indices`. `mode` is `"replace"` (for a repeated index, the update latest in row-major order wins) or `"add"` (repeated indices accumulate). Other strings are a type error. `"add"` rejects `bool`. |
| `scatter_elements(base, indices, updates, axis)` | One element per index: `indices`, `updates`, and `base` have equal rank, `indices` and `updates` have equal shape, and each index replaces only the `axis` coordinate. Last write wins. |

Indices may be any signed integer dtype and are zero-based. A negative or
out-of-range index traps before any write, with a message such as
`gather index 3 out of bounds at axis 1 of extent 3`; negative indices do not
count from the end. `grad` rejects `"replace"` scatter and `scatter_elements`,
because the result at a repeated index depends on write order; use `"add"` on a
differentiated path.

```chelis-surf
module Shapes
x = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])
flat = reshape(x, [6i64])
swapped = permute(x, 1i32, 0i32)
tiled = insert(x, 0i32, 2i64)
padded = pad(x, [[0i64, 0i64], [1i64, 1i64]], 0.0f32)
middle = shrink(x, [[0i64, 2i64], [1i64, 2i64]])
joined = concat([copy(x), copy(x)], 0i32)
parts = split(x, 1i32, [1i64, 2i64])
picked = gather(x, to_tensor([2i64, 0i64]), 1i32)
added = scatter(x, to_tensor([0i64, 0i64]), to_tensor([[10.0f32, 20.0f32], [30.0f32, 40.0f32]]), 1i32, "add")
```

```text
x = tensor(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
flat = tensor(shape=[6], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
swapped = tensor(shape=[3, 2], data=[1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
tiled = tensor(shape=[2, 2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
padded = tensor(shape=[2, 5], data=[0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0])
middle = tensor(shape=[2, 1], data=[2.0, 5.0])
joined = tensor(shape=[4, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
parts = [tensor(shape=[2, 1], data=[1.0, 4.0]), tensor(shape=[2, 2], data=[2.0, 3.0, 5.0, 6.0])]
picked = tensor(shape=[2, 2], data=[3.0, 1.0, 6.0, 4.0])
added = tensor(shape=[2, 3], data=[31.0, 2.0, 3.0, 74.0, 5.0, 6.0])
```

The list literal in `concat` takes ownership of its elements, so the example
passes `copy(x)` to keep `x` usable.

### Contraction, selection, and ordering

| Signature | Contract |
|---|---|
| `matmul(a, b, accumulator=p)` | Floats only. `a: [..., i, j]` and `b: [..., j, k]` give `[..., i, k]`. Leading batch axes must be equal, have extent one, or be absent on one side; `matmul(insert(x, 0i32, 4i64), w)` multiplies each of 4 matrices by `w`. Integer operands are a type error; use `einsum`. The result keeps the operand dtype. |
| `einsum(equation, a, b, accumulator=p)` | Exactly two operands. The equation matches `[a-z]*,[a-z]*->[a-z]*` with no spaces, no `...`, and an explicit output. Each operand has one label per axis; a label repeated in one operand takes its diagonal; labels absent from the output are summed. Signed integers and floats; the result follows `sum`'s accumulator rule. |
| `where(mask, a, b)` | `bool` mask, branches of one dtype and the mask's shape. Takes `a` where the mask is true. |
| `clamp(x, low, high)` | Signed integers and floats. `low` and `high` are tensors of `x`'s dtype, rank zero or `x`'s shape: write `scalar_to_tensor(2.0f32)`, not `2.0f32`. A NaN bound or `low > high` traps; a NaN input stays NaN. |
| `cumsum(x, axis)` | Inclusive running sum along `axis`. Signed integers and floats; the result dtype follows `sum`'s default accumulator, so `i8` input gives `i32`. |
| `sort(x, axis)` | Returns `(values, indices)` with `i64` indices. Stable ascending; NaNs go last, and `-0.0` equals `0.0`. The axis may be computed at run time. |
| `diagonal(x, axis1, axis2)` | Elements with equal coordinates on the two distinct axes. The second axis is removed; the first takes the smaller extent. |
| `trace(x, axis1, axis2)` | Sum of that diagonal, removing both axes, with `sum`'s accumulator rule. |

```chelis-surf
module Select
x = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])
w = to_tensor([[1.0f32, 0.0f32], [0.0f32, 1.0f32], [1.0f32, 1.0f32]])
product = matmul(x, w)
gram = einsum("ij,kj->ik", x, x)
bounded = clamp(x, scalar_to_tensor(2.0f32), scalar_to_tensor(5.0f32))
running = cumsum(x, 1i32)
ordered = sort(to_tensor([3.0f32, -1.0f32, 2.0f32]), 0i32)
diag = trace(x, 0i32, 1i32)
```

```text
x = tensor(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
w = tensor(shape=[3, 2], data=[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
product = tensor(shape=[2, 2], data=[4.0, 5.0, 10.0, 11.0])
gram = tensor(shape=[2, 2], data=[14.0, 32.0, 32.0, 77.0])
bounded = tensor(shape=[2, 3], data=[2.0, 2.0, 3.0, 4.0, 5.0, 5.0])
running = tensor(shape=[2, 3], data=[1.0, 3.0, 6.0, 4.0, 9.0, 15.0])
ordered.0 = tensor(shape=[3], data=[-1.0, 2.0, 3.0])
ordered.1 = tensor(shape=[3], data=[1, 2, 0])
diag = 6.0
```

### Construction, conversion, and inspection

| Signature | Contract |
|---|---|
| `to_tensor(xs)`, `to_tensor(xs, T)` | A rectangular nested `List` becomes a tensor whose rank is the nesting depth. Unsuffixed literals need the dtype argument: `to_tensor([1, 2], i16)`; `to_tensor([1, 2])` is an error. Ragged rows are an error. |
| `to_list(x)` | A rank-one tensor becomes a `List`. Other ranks are a type error; `reshape` to rank one first. |
| `pad_sequences(rows, pad)` | Ragged `List[List[T]]` to a `[len(rows), width]` tensor, where `width` is the longest row; short rows are filled with `pad`. |
| `pad_sequences_to(rows, width: i64, pad)` | Same with a fixed nonnegative `width`; longer rows are truncated. |
| `cast(x, T)` | Explicit dtype change, same shape. A float with a fractional part cast to an integer traps `domain`; a value outside the target range traps `overflow`. The named `cast_trunc`, `cast_saturate`, and `cast_wrap` choose other policies; see the [type reference](type-reference.md). |
| `scalar_to_tensor(v)`, `tensor_to_scalar(t)` | Between a scalar and a rank-zero tensor. `tensor_to_scalar` requires rank zero, not just one element. |
| `shape(x, axis)` | `i64` extent of `axis`. The axis must be nonnegative here. |
| `rank(x)`, `numel(x)` | `i32` rank and `i64` element count. |
| `copy(x)`, `realize(x)` | `copy` makes an owned duplicate, so you can pass a value somewhere that consumes it and keep the original. `realize` materializes an intermediate result. |

### Explicit randomness

Random draws are pure functions of a `key`. The same key gives the same numbers
in `chelis eval`, `chelis test`, and compiled C. Each key can be used once:
using it a second time is a compile error, so split a key before making two
draws.

| Signature | Contract |
|---|---|
| `key_from_seed(seed: i64) -> key` | The key whose bits are the seed's bits. |
| `split_key(k) -> (key, key)` | Two independent child keys; consumes `k`. |
| `split_keys(k, n: i64) -> tensor[n, key]` | `n` child keys; `n` must be nonnegative. |
| `fold_in(k, n: i64) -> key` | The child key for any `i64` `n`, negative included. |
| `uniform_like(k, x, low, high)` | A tensor with `x`'s shape and float dtype, each element uniform in `[low, high)`. `x`'s values are not read. `low` and `high` have `x`'s dtype, are finite, and satisfy `low <= high`; otherwise it traps `domain` before drawing. Rounding can produce exactly `high`. |
| `dropout(k, x, rate)` | Inverted dropout: each element is zeroed with probability `rate` and the rest are divided by `1 - rate`. `rate` has `x`'s float dtype and satisfies `0 <= rate < 1`; `rate = 1` and NaN trap `domain`. |

Neither draw carries an effect, so a function that uses them stays pure.

```chelis-surf
module Rng
def noisy(seed: i64, x: tensor[4, f32]) -> (tensor[4, f32], tensor[4, f32]) = {
  (k1, k2) = split_key(key_from_seed(seed))
  (uniform_like(k1, x, -1.0f32, 1.0f32), dropout(k2, x, 0.5f32))
}
out = noisy(42i64, to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))
```

```text
out.0 = tensor(shape=[4], data=[0.7780236, -0.40719092, -0.26147556, 0.94049156])
out.1 = tensor(shape=[4], data=[2.0, 4.0, 0.0, 0.0])
```

## `Std` modules

### Tensor and list helpers

| Function | Contract |
|---|---|
| `Std.Tensor.Construct.arange(start, stop)` | Both endpoints have one signed-integer dtype, which the result keeps. Half-open and ascending: `arange(-2i64, 3i64)` is `[-2, -1, 0, 1, 2]`. `stop <= start` gives an empty tensor. |
| `Std.Tensor.Construct.linspace(start, stop, count: i64)` | Float endpoints of one dtype, `count >= 1`. Count one returns `[start]`; a larger count includes both endpoints exactly, and `stop < start` descends. Each interior weight `i / (count - 1)` is rounded once to the endpoint dtype, then `start + (stop - start) * weight` is evaluated at that dtype. A non-finite endpoint or `count < 1` fails with `linspace: ...`. |
| `Std.Tensor.Mask.where_indices(mask)` | A rank-one `bool` tensor in; the increasing `i64` indices of true elements out. For a higher-rank mask, `reshape` it to rank one first; the indices are then row-major flat positions. |
| `Std.Sort.sort(x, axis)` | The built-in `sort` through an import. |
| `Std.Scan.scan_list(step, initial, values)` | `step(acc, item)` applied left to right. Returns one accumulator per item, not including `initial`: summing `[1, 2, 3]` from `0` gives `[1, 3, 6]`. |
| `Std.Index.list_index(xs, i: i64)` | Zero-based element. A negative or out-of-range index fails: `index 2 out of bounds for list of len 2`. |
| `Std.Index.take_list(xs, n)`, `skip_list(xs, n)` | The first `n` elements, or all but the first `n`. A count past the end truncates; a negative count fails. |
| `Std.Scalar.max(a, b)`, `min(a, b)` | Numeric scalars of one dtype. The first NaN wins; on a tie, including `-0.0` against `0.0`, the first operand is returned, and the gradient flows to it. |
| `Std.Scalar.abs(x)` | Numeric scalar. Returns `+0.0` for both zeros and `+inf` for `-inf`; the derivative is zero at both zeros and NaN. Integer `abs` of the minimum value traps. |
| `Std.Text.join(parts, sep)` | Joins a `List[string]` with `sep`; an empty list gives `""`. |
| `Std.Contracts` | `normal_cdf(x)` over any float dtype, plus the contract names and settings used by numerical property tests. |

`Std.Tensor.Construct` also exports `stack`, `squeeze`, and `unsqueeze`. Their
signatures place the axis between two rank spreads, and a call on a tensor of
concrete shape fails `chelis check`. Use `insert`, `reshape`, and `concat`
instead; [known issues](known-issues.md#tensor-construction-helpers)
has the replacement for each.

```chelis-surf
module Helpers
import Std.Tensor.Construct (arange, linspace)
import Std.Tensor.Mask (where_indices)
import Std.Scan (scan_list)
import Std.Index (take_list)
steps = arange(-2i64, 3i64)
grid = linspace(0.0f64, 1.0f64, 5i64)
hits = where_indices(to_tensor([false, true, false, true]))
running = scan_list(fn (acc: i64, x: i64) -> add(acc, x), 0i64, [1i64, 2i64, 3i64])
first_two = take_list([7i64, 8i64, 9i64], 2i64)
```

```text
steps = tensor(shape=[5], data=[-2, -1, 0, 1, 2])
grid = tensor(shape=[5], data=[0.0, 0.25, 0.5, 0.75, 1.0])
hits = tensor(shape=[2], data=[1, 3])
running = [1, 3, 6]
first_two = [7, 8]
```

### Files

Every function in `Std.Io` carries the `IO` effect and takes a `string` path,
relative to the working directory. A failed operating-system call stops the
program with the operation, path, and OS error, such as
``read_file failed for `missing.txt`: No such file or directory (os error 2)``.

| Signature | Contract |
|---|---|
| `read_text(path) -> string` | The whole file as UTF-8 text. |
| `write_text(path, contents) -> unit` | Creates the file or replaces its contents. The directory must exist. |
| `read_trimmed_lines(path) -> List[string]` | Lines without line endings, each trimmed of surrounding whitespace, with empty lines dropped. |
| `read_head_bytes(path, width: i64) -> List[i64]` | The first `width` bytes as integers `0..255`; a file shorter than `width` returns all of its bytes. A negative width fails. |
| `exists(path) -> bool` | False for a missing path; never fails for absence. |
| `list(path) -> List[string]` | Entry names (not paths) of a directory, sorted by name bytes, without `.` and `..`. A non-directory fails. |
| `mmap_size(path) -> i64` | The file's size in bytes. |

`Std.Io.Parquet` and `Std.Io.Safetensors` contain no readers or writers:
each of their functions fails with a message naming itself. Use CSV or JSON for
tabular and structured data.

### CSV

`Std.Io.Csv` represents a table as `List[Dict[string, string]]`, one dictionary
per row keyed by the header. Fields stay strings; convert them yourself.

| Signature | Contract |
|---|---|
| `read_csv(path)` | Reads a file. The first nonempty line is the header; empty lines are skipped. Fields are comma-separated; a field in double quotes may contain commas, and `""` inside quotes is one quote. Whitespace is kept. Fails with `read_csv failed for <path>` when the file is missing, a quote is unterminated, or a row's field count differs from the header's. |
| `try_read_csv(path)` | The same, returning `None` instead of failing. |
| `to_csv(rows)`, `try_to_csv(rows)` | Renders text: a header line, one line per row, and a final newline. Column order is the first row's key order. Fields containing a comma or quote are quoted. Fails, or returns `None`, when a row's keys differ from the first row's or a field or header contains CR or LF; the message names the first bad row. |
| `write_csv(path, rows)`, `try_write_csv(path, rows)` | `to_csv` written to a file. |

Headers are not checked for duplicates. With a repeated header, the row
dictionary holds one key, carrying the last such column's value, so a later
`to_csv` writes fewer columns than the file had. The reader works line by line,
so a CR or LF inside a quoted field cannot be read; the writer rejects such
values rather than writing a file it cannot read back.

With `parts.csv` containing a quoted comma:

```text
name,qty
bolt,4
"nut, hex",12
```

```chelis-surf
module CsvDemo
import Std.Io.Csv (read_csv, to_csv)
rows = read_csv("parts.csv")
text = to_csv(rows)
```

```text
rows = [dict(name: bolt, qty: 4), dict(name: nut, hex, qty: 12)]
text = name,qty
bolt,4
"nut, hex",12
```

### JSON

`Std.Io.Json` exports `Json` and its constructors, including `JsonInt(i64)`,
`JsonBigInt(string)` for integer text outside `i64`, and
`JsonFloat(f64, string)` for a number token with a fraction or exponent: the
token's correctly rounded `f64` and its exact text.
Use `parse_json(text)` or `try_parse_json(text)` for text,
`load_json(path)` or `try_load_json(path)` for files, and `json_get` and the
`json_*` accessors to inspect values. `json_int` returns only a stored
`JsonInt`; `json_float` returns a stored `JsonFloat`'s `f64`, also accepts a
`JsonInt` through an explicit `i64`-to-`f64` conversion, and does not accept
`JsonBigInt`. Match `JsonFloat(_, text)` and pass `text` to `decimal` in
`Std.Decimal` to get the exact value the producer wrote. `to_json` and
`try_to_json` render text; `write_json` and `try_write_json` write files.
A `JsonFloat` serializes as its text, so `1E5` and `19.950` keep their
spelling, and documents that spell one number differently are unequal. Build
one from a computed value with `JsonFloat(x, to_string(x))`. A `JsonFloat`
whose text is not such a token or does not round to its finite `f64`, and
invalid `JsonBigInt` text, cannot be serialized.

Parsing, accessors, and text rendering in the CSV and JSON modules are pure.
Their file operations carry `IO`. All of them run under `chelis eval`,
`chelis test`, and compiled C.

### Tests and processes

`Std.Test` assertions carry the `Test` effect, so they belong in `def test_*()`
functions, which `chelis test` runs. Every assertion takes a `label: string`
last, and a failure reports it.

| Signature | Contract |
|---|---|
| `assert_true(cond, label)`, `assert_false(cond, label)` | `cond: bool`. |
| `assert_eq(actual, expected, label)` | Two values of one type: scalars, strings, or `List`, tuple, `Dict`, `Option`, or data-type values compared field by field. NaN is unequal to itself. Not for tensors. |
| `assert_eq_tensor(actual, expected, label)` | Two tensors of one shape and dtype, compared element by element. |
| `assert_close(actual, expected, tol, label)` | Float scalars of one dtype. Passes when `abs(actual - expected) <= tol`. |
| `assert_close_tensor(actual, expected, tol, label)` | The same rule for every element of two same-shaped float tensors; reports the first failing row-major index. |
| `assert_shape(t, expected: List[i64], label)` | The tensor's rank and every extent equal `expected`. A negative entry fails. |
| `fail(msg)` | Fails the test with `msg`. |

The tolerance is absolute, has the values' dtype, and must be finite and
nonnegative; zero means exact numeric equality. `f16`, `bf16`, and `f32` compare
in `f32` arithmetic, `f64` in `f64`. NaN is never close to anything, an infinity
is close only to the same signed infinity, and `-0.0` equals `0.0`. The builtin
`test_assert(cond, label)` is the primitive under `assert_true`.

In a Reef package with module prefix `Demo`, `tests/asserts.ch`:

```chelis-surf
module Demo.Tests.Asserts
import Std.Test (assert_true, assert_eq, assert_close, assert_close_tensor, assert_shape)
def test_scalars() -> unit ! { Test } = {
  _ = assert_true(gt(3i64, 2i64), "three exceeds two")
  _ = assert_eq([1i64, 2i64], [1i64, 2i64], "lists match")
  assert_close(add(0.1f64, 0.2f64), 0.3f64, 1e-12f64, "float sum")
}
def test_tensors() -> unit ! { Test } = {
  x = to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32]])
  _ = assert_shape(x, [2i64, 2i64], "square")
  assert_close_tensor(x, to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.5f32]]), 0.1f32, "values")
}
```

`chelis test` prints the following and exits `1`:

```text
tests/asserts.ch
  test_scalars .................. PASS
  test_tensors .................. FAIL (assert_close_tensor (values): at index 3 expected 4.5, got 4.0, tol 0.1)

1 passed, 1 failed
```

Assertions also compile: in a program built with `chelis build`, a failed
assertion prints `assert failed: <label>` and the program exits `1`.

`Std.Process.run(cmd, args)` runs the program `cmd` with the argument list
`args: List[string]` and returns `(exit_code: i64, stdout: string, stderr: string)`.
No shell is involved, so arguments are never split or expanded. A process killed
by a signal reports exit code `-1`, and output that is not valid UTF-8 fails the
call. `run_chelis(args)` runs the `chelis` command. Both carry `IO` and run
under `chelis eval`, `chelis test`, and compiled C.

### Dates and times

`Std.Datetime` provides civil dates (`Date`), times of day (`Time`), civil
datetimes (`DateTime`), instants on the POSIX timescale (`Instant`), fixed
offsets (`Offset`), instants with their written offset (`OffsetDateTime`),
exact durations (`Duration`), calendar periods (`Period`), and the column types
`Dates[n]` and `Instants[n]`. Years run from -9999 through 9999. Each of these
types is opaque: obtain a value from a validating producer such as `date(y, m, d)`,
`parse_date(text)`, `instant_from_unix(s, ns)`, `duration(s, ns)`, or
`period(months, days)`, never from a record literal.

Where a library would choose silently, the caller states the policy:
`date_add_months(d, n, ClampToMonthEnd)` or `RejectInvalidDay` for 31 January
plus one month, and a `Rounding` from `Std.Rounding` for `instant_to_unix_count`,
`duration_to_count`, and `instant_round_to`, such as
`instant_to_unix_count(i, Milliseconds, RoundTowardNegative)`; `RejectInexact`
fails instead of rounding. `duration_to_seconds_f64` is the one other conversion
that drops precision, rounding to nearest. Text forms follow one RFC 3339-based
profile (`2026-10-01T09:30:00-04:00`, `PT3661S`, `P14M3D`). A failure reports
`<function>: domain: <detail>` for an input that denotes no value, or
`<function>: overflow: <detail>` when arithmetic or a count conversion leaves
its type's range. Each `try_` form other than the masked column forms returns
`None` where its twin fails `domain`. `eq` and
`neq` compare values. Dates, times, datetimes, instants, and durations also
have `*_lt`, `*_lte`, `*_gt`, and `*_gte` functions that order them; offsets,
offset datetimes, and periods have no order. The module is pure and runs under
`chelis eval`, `chelis test`, and generated C, including a program that
projects `.0` from a masked column producer such as `try_dates_from_epoch_days`.

`Std.Datetime.Clock` reads the host clocks. It is a separate module, so
nothing in `Std.Datetime` reads a clock; any code that reads one, through this
module or the underlying builtins, carries `IO`. `clock_now()`
returns the wall-clock `Instant`. `monotonic_now()` returns a `MonotonicInstant`
from a clock that never runs backwards, and `monotonic_until(a, b)` is the exact
`Duration` from `a` to `b`. A `MonotonicInstant` has an unspecified origin, so
no other datetime function takes one and it does not convert to an `Instant`. Both reads carry
`IO`, so every caller carries it, declared or inferred. A failed read reports
`clock_wall_read: io: <detail>` or `clock_monotonic_read: io: <detail>`, naming
the underlying read. The clocks run under `chelis eval`, `chelis test`, and
compiled C, which share one definition of each read.

### Business days

`Std.Datetime.Business` builds a `BusinessCalendar` from a `Weekmask` record
naming the business weekdays, a holiday list, and a horizon:
`business_calendar(w, holidays, valid_from, valid_until)`. There is no built-in
weekend. A calendar answers only from the days of its horizon. A date outside
it, or a query whose answer the days beyond the horizon could change, fails
`domain` instead of degrading to a weekends-only answer; the one exception is
a count's end, which may be the day after the horizon. `is_business_day`
tests a date. `business_day_roll` takes `Unadjusted`, `Following`,
`Preceding`, `ModifiedFollowing`, or `ModifiedPreceding`. `business_day_offset`
rolls a non-business start as its `NonBusinessStart` policy says
(`RejectNonBusinessStart`, `RollStartForward`, or `RollStartBackward`) and
then moves whole business days. `business_day_count` counts `[begin, end)` and
is negative for a reversed range. `business_in_all` and `business_in_any`
combine two calendars over their common horizon. The constructor, the
queries, and the combinations each have a `try_` form, and the `dates_*` forms
apply the queries to `Dates[n]` columns.

### Date and instant columns

`Std.Datetime.Columns` applies `Std.Datetime` to whole columns. `dates_year`,
`dates_month`, `dates_day`, `dates_weekday_iso_number`, and `dates_day_of_year`
read every date of a `Dates[n]` into a `tensor[n, i64]`. `dates_from_ymd` builds
a column from year, month, and day tensors. `dates_add_days` and
`dates_add_months` take a tensor of counts, one per date, and
`dates_add_months` takes a `DayOverflow` policy. `dates_days_until` and the
`dates_lt` family compare two columns element by element. `try_parse_dates`
reads a `List[string]` of dates, and `dates_to_strings` writes them.

For instants, `instants_from_unix_count` and `instants_to_unix_count` convert
counts of a `TimeUnit`; the second takes a `Rounding`.
`instants_add_duration` adds a `Durations[n]` column, which `instants_until`
produces and `durations(seconds, nanoseconds)` builds. `instants_round_to`
buckets instants by one increment, and `instants_to_dates_at` reads their dates
at an offset. `instants_seconds_since_f64(is, origin)` gives a numerical time
axis, equal bit for bit to `duration_to_seconds_f64` of each difference.

Each callable agrees with its scalar function at every element. A call fails
when any element would fail, and the message names the lowest such element
(`dates_add_days: overflow: element 3: ...`). Columns of different lengths fail
`domain`. The masked forms `try_dates_from_ymd`, `try_durations`, and
`try_parse_dates` return the column with a `tensor[n, bool]` mask that is false
where the scalar function would fail. Those positions hold 1970-01-01 or a
zero duration. Every callable consumes its columns and borrows its tensors.

### Exact decimals

`Std.Decimal` provides `Decimal`, an exact base-10 number: a coefficient of at
most 38 digits over a power of ten with at most 38 fractional digits, such as
`12.5` or `-0.000001`. Every value has one canonical form, so `eq` and `neq`
compare the numbers themselves, and `decimal_lt`, `decimal_lte`, `decimal_gt`,
and `decimal_gte` order them. The type is opaque: obtain a value from
`decimal(text)`, which accepts an RFC 8259 number such as `"19.99"` or
`"1e-3"`, from `decimal_from_i64(n)`, or from `decimal_from_f64(x, places,
mode)`, never from a record literal.

`decimal_add`, `decimal_sub`, and `decimal_mul` are exact and fail rather than
round when the result does not fit. `decimal_round`, `decimal_div`,
`decimal_to_i64`, and `decimal_from_f64` take a `Rounding` from `Std.Rounding`,
and all but `decimal_to_i64` also take a number of fractional places:
`decimal_round(x, 2, RoundTiesToEven)`, `decimal_div(a, b, 10, RoundTiesToEven)`,
`decimal_to_i64(x, RoundTowardZero)`, and `decimal_from_f64(x, 2,
RoundTiesToEven)`, which rounds the float's exact binary value; `RejectInexact`
fails instead of rounding. `decimal_to_f64`, `decimal_to_f32`,
`decimal_to_f16`, and `decimal_to_bf16` are the other conversions that drop
precision: each rounds the exact value once, to nearest with ties to even, so
`decimal_to_f16(x)` can differ from casting `decimal_to_f32(x)` to `f16`, which
rounds twice. A magnitude of 65520 or more is past f16's range and becomes the
infinity of its sign. `decimal_to_string` writes the canonical text without an
exponent, and `decimal_to_fixed_string(x, n)` writes exactly `n` fractional
digits and never rounds. A failure reports `<function>: domain: <detail>` for an argument that
denotes no value of its domain, including text or a float whose value lies
outside the decimal range, or `<function>: overflow: <detail>` when the exact
result of arithmetic lies outside that range or `decimal_to_i64`'s integer lies
outside i64; each `try_` form returns `None` where its twin fails `domain`, and
`try_decimal_to_i64` also where its twin fails `overflow`. The module is
pure and runs under `chelis eval`, `chelis test`, and generated C.

### Time zones

`Std.Datetime.Zone` represents time zone rules as values. `time_zone_from_tzif(name,
bytes)` reads a TZif file (RFC 9636, versions 2 to 4) whose bytes the program
supplies: the standard library holds no time zone database and never reads the
host's. `time_zone_fixed(o)` and `time_zone_utc()` build fixed zones. A `Zoned` is
an instant in a zone: `zoned(i, tz)` pairs them, and `zoned_from_local(dt, tz,
policy)` resolves a local reading, where the `Disambiguation` (`EarlierInstant`,
`LaterInstant`, `CompatibleInstant`, or `RejectNonUniqueLocal`) decides a reading
that a daylight saving change skips or repeats. `zoned_add_duration` moves along
the instant line and `zoned_add_period` moves the wall clock, keeping the value
unchanged for a zero period. `zoned_to_string` writes RFC 9557 text such as
`2026-10-01T09:30:00-04:00[America/New_York]`, with an offset whose seconds are
nonzero written `±HH:MM:SS` as the text profile allows;
`parse_zoned_text` reads it into a plain `ZonedText` record, and
`zoned_from_text(zt, tz, policy)` resolves that against a zone the caller
obtained, with an `OffsetConflict` policy for a written offset the zone does not
use there. A written `Z`, `z` or `-00:00` means a UTC time whose local offset is
unknown (RFC 9557): its record has no offset, and every policy resolves it to that
UTC instant. A zone whose TZif footer is empty has no offsets from its last
transition on, and a call that needs one there fails `domain`.
