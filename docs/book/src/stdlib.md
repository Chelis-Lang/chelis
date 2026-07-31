# Runtime and Standard Library

This page is a reference for the operations you call when writing Chelis: the tensor
primitives that are built into the language, and the `chelis-std` library that ships with
the compiler under the `Std` module prefix. The primitives are documented in
`spec/05-risc-primitives.md`; the library source lives under `packages/chelis-std/src/`.

`chelis-std` is the runtime. It is bundled with the compiler the way `core` is bundled with
Rust, so every program can use it without a Reef install. Each file is its own
`module Std.<...>` with its own export list.

## Tensor primitives

These are built into the compiler and called as plain names. They take and return tensors,
respect named dimensions exactly, and never broadcast or promote precision implicitly.

### Elementwise

Binary, operating on two tensors with identical dimensions and precision:

- `add`, `mul`, `sub`, `div` arithmetic.
- `max_elem`, `min_elem` element-wise maximum and minimum.
- `cmplt`, `eq`, `neq`, `gt`, `gte`, `lte` comparisons, returning a `bool` tensor.
- `and`, `or`, `not` on `bool` tensors.

Unary, operating on float tensors and preserving the shape:

- `neg`, `recip`, `abs`.
- `exp`, `log`, `sqrt`.
- `sin`, `cos`, `tan`, `atan`.
- `floor`, `ceil` (these are not differentiable; `grad` rejects them).
- `relu`, `sigmoid` activations.

### Reductions

Reductions take an explicit integer axis and remove that axis from the result.

- `sum(x, axis)` with an optional accumulator precision, `mean(x, axis)`.
- `max_reduce(x, axis)`, `min_reduce(x, axis)`, `prod_reduce(x, axis)`.
- `argmax_reduce(x, axis)`, `argmin_reduce(x, axis)` returning index tensors.
- `softmax(x, axis)` numerically stable softmax.

Windowed reductions slide a window over the tensor:

- `reduce_window_max`, `reduce_window_min`, `reduce_window_sum`, `reduce_window_mean`, each
  taking the input, a window shape, and strides. The C target is the supported codegen path
  for these.

```chelis-surf-fragment
windowed_max = reduce_window_max(pool_grid, [2, 2], [1, 1])
windowed_mean = reduce_window_mean(pool_grid, [2, 2], [2, 2])
```

### Structural and shape operations

- `reshape(x, shape)` reinterprets the layout; the product of dimensions must match.
- `permute(x, axes)` reorders dimensions by a permutation.
- `expand(x, axis, size)` adds a dimension explicitly. This is the tool that replaces
  broadcasting.
- `pad(x, padding, fill)`, `shrink(x, bounds)` add or slice boundary elements.
- `concat(tensors, axis)`, `split(x, axis, sizes)` join and divide along an axis.
- `gather(table, indices, axis)`, `scatter(base, indices, updates, axis, mode)` index and
  update. `scatter` modes are `"replace"` (last write wins) and `"add"` (differentiable
  accumulation).
- `where(mask, a, b)` selects by a `bool` mask, `clamp(x, low, high)` clips to bounds.
- `cumsum(x, axis)` cumulative sum.
- `einsum("ij,jk->ik", a, b)` contraction by subscript, covering matmul, batched matmul,
  transpose, trace, and outer products.
- `sort(x, axis)` returns a `(values, indices)` tuple.
- `diagonal(x, axis1, axis2)`, `trace(x, axis1, axis2)`.
- `matmul(a, b)` matrix multiply with an optional accumulator precision, for rank two and
  above.

```chelis-surf-fragment
contracted = einsum("ij,jk->ik", lhs, rhs)
packed = concat([lhs, rhs], 1)
embed = gather(table, token_ids, 0)
selected = where(mask, embed, zeros)
clipped = clamp(running, floor15, ceil30)
```

### Construction, precision, and ownership

- `to_tensor(list)`, `to_list(tensor)` bridge lists and tensors. `pad_sequences(list, fill)`
  builds a rectangular tensor from ragged rows.
- `cast(x, precision)` changes precision.
- `shape(t, axis)` returns a runtime `int32` scalar for an axis length.
- `copy(x)` produces a fresh owned value; `realize(x)` materializes an intermediate.

### Randomness

`dropout(x, rate)` and `uniform_like(x, low, high)` carry the `Random` effect, discharged by
`with seed(...)`. The `Std.Init` modules build their initializers on these.

## Standard library modules

### Tensor helpers

`Std.Tensor.Construct` builds and reshapes tensors:

- `linspace(start, stop, count)`, `arange(start, stop)`.
- `stack(xs)` concatenates a list of rows along a new leading axis.
- `squeeze(x)` removes a size-1 dimension, `unsqueeze(x)` inserts one.

`Std.Tensor.Mask`:

- `where_indices(mask)` returns the `int64` indices where a `bool` mask is true.

### Initializers

`Std.Init.Random`:

- `normal_like(template, mean, std)` draws a normal tensor shaped like the template.

`Std.Init.Kaiming`:

- `kaiming_uniform(template, fan_in)`, `kaiming_normal(template, fan_in)`.

`Std.Init.XavierExt`:

- `xavier_uniform(template, fan_in, fan_out)`, `xavier_normal(template, fan_in, fan_out)`.
- `trunc_normal(template, mean, std, a, b)` draws a normal tensor clipped to `[a, b]`.

All initializers carry the `Random` effect.

### Sorting and scanning

`Std.Sort`:

- `sort_1d(values, axis)`, `sort_2d(values, axis)`, each returning a `(values, indices)`
  tuple.

`Std.Scan`:

- `scan_list(step, initial, values)` returns the running accumulator at each step.

`Std.Index`:

- `list_index(values, idx)`, `take_list(values, count)`, `drop_list(values, count)`.

### Decimal and time

`Std.Decimal` is fixed-point decimal, a coefficient and a scale. Construction with
`decimal(text)` or `decimal_from_int(value)`; arithmetic with `decimal_add`, `decimal_sub`,
`decimal_mul`, and `decimal_div(lhs, rhs, result_scale, mode)`; comparison with
`decimal_eq`, `decimal_lt`, and the rest; conversion with `decimal_to_float` and
`decimal_to_string`. Rounding modes are `round_half_up`, `round_half_even`, `round_down`,
and `round_up`.

`Std.Time` is calendar dates on the proleptic Gregorian calendar. Construction with
`date(year, month, day)`; arithmetic with `add_days`, `sub_days`, `days_between`; comparison
with `date_lt` and friends; `day_of_week`, `day_of_year`, `is_leap_year`; and
`date_to_string` and `parse_date` for ISO `YYYY-MM-DD` text.

### Input and output

`Std.Io` wraps the host file operations:

- `read_text(path)`, `write_text(path, contents)`, `read_trimmed_lines(path)`,
  `read_head_bytes(path, width)`, `exists(path)`, `list(path)`, `mmap_size(path)`.

`Std.Io.Csv`:

- `read_csv(path)` returns a list of header-keyed dictionaries, `try_read_csv(path)` returns
  the optional form.
- `to_csv(rows)` renders that same shape back to CSV text (header from the first
  row's key order, minimal quoting with doubled embedded quotes, LF line endings,
  trailing newline; zero rows render as the empty string; a record that would
  render as a blank line — a single empty value or header — renders as a quoted
  empty field `""` so `read_csv`'s blank-line filter cannot drop it). Fields or
  headers containing CR or LF are **rejected** — the line-based reader cannot
  round-trip them (chelis#954 tracks the whole-file reader that lifts this), so
  the writer refuses rather than emit output its own reader mangles. `try_to_csv` returns `None` instead of failing (mismatched row key
  sets, CR/LF content); `write_csv(path, rows)` writes the rendered text and
  names the path and first offending row on failure; `try_write_csv` is its
  `Option` twin.

`Std.Io.Json` parses JSON into a `Json` value (`JsonNull`, `JsonBool`, `JsonInt`,
`JsonFloat`, `JsonString`, `JsonArray`, `JsonObject`; the constructors are
exported, so documents can be built directly):

- `load_json(path)`, `parse_json(text)` and their `try_` variants.
- `json_get`, and the typed accessors `json_string`, `json_int`, `json_float`, `json_bool`,
  `json_array`, `json_object`, plus `json_is_null`.
- `to_json(value)` renders a `Json` value compactly (object keys in dictionary
  insertion order, f64 via `to_string`'s shortest-round-trip form — a claim
  scoped to **f64 specifically**: narrower floats route through a compiled-lane
  `to_string` funnel that PR #863 fixes and that fix is not on `main` at this
  writing, so only the f64 path is version-independent — escapes for
  `\" \\ \n \t \r`). Non-finite numbers have no JSON representation: `to_json`
  fails on them and `try_to_json` returns `None`. `write_json(path, value)`
  writes the rendered text and names the path on failure; `try_write_json` is
  its `Option` twin. Control characters outside the escaped set pass through
  unescaped — RFC 8259-invalid output for such input — because chelis has no
  `char_code` primitive to emit `\u00XX`; the parser's `decode_escape` likewise
  rejects `\uXXXX` input. Both halves of that asymmetry need a character-level
  primitive and are tracked as chelis#953.

These IO modules carry the `IO` effect and run on the evaluator and host paths.

### Tokenization

`Std.Tokenizer` is a byte-pair tokenizer that loads a Hugging-Face-style `tokenizer.json`:

- `load_tokenizer(path)` and `try_load_tokenizer(path)`.
- `encode(tokenizer, text)`, `decode(tokenizer, ids)`.
- `batch_encode(tokenizer, texts, max_length, pad_value)` returns a padded `int64` tensor.

### Testing

`Std.Test` provides assertion helpers for `def test_*()` functions discovered by
`chelis test`. They carry the `Test` effect:

- `assert_true`, `assert_false`, `assert_eq`, `assert_eq_int`, `assert_eq_bool`,
  `assert_eq_string`, `assert_close`.
- `assert_close_tensor`, `assert_eq_tensor_int64`, `assert_shape` for tensors.
- `fail(msg)`.

## Process execution

`Std.Process` runs external programs through the host:

- `run(cmd, args)` returns `(exit_code, stdout, stderr)`.
- `run_chelis(args)` runs the Chelis binary itself.

These run on the evaluator and test paths.
