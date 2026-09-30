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
windowed_max = reduce_window_max(pool_grid, [2i64, 2i64], [1i64, 1i64])
windowed_mean = reduce_window_mean(pool_grid, [2i64, 2i64], [2i64, 2i64])
```

### Structural and shape operations

- `reshape(x, shape)` reinterprets the layout; the product of dimensions must match.
- `permute(x, axes)` reorders dimensions by a permutation.
- `insert(x, axis, size)` adds a dimension explicitly, raising the rank by one.
  This is the tool that replaces broadcasting.
- `expand(x, axis, size)` broadcasts an existing size-1 dimension to `size`,
  leaving the rank alone.
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
- `shape(t, axis)` returns a runtime `i64` scalar for an axis length; the
  axis argument itself is `i32`.
- `copy(x)` produces a fresh owned value; `realize(x)` materializes an intermediate.

### Randomness

`dropout(k, x, rate)` and `uniform_like(k, x, low, high)` draw from the key `k` they are
given and consume it; they carry no effect. `key_from_seed(seed)` makes a root key from an
`i64`, `split_key(k)` returns two keys, `split_keys(k, n)` returns a `tensor[n, key]`, and
`fold_in(k, n)` derives the key of an integer. The `Std.Init` modules build their
initializers on these.

## Standard library modules

### Tensor helpers

`Std.Tensor.Construct` builds and reshapes tensors:

- `linspace(start, stop, count)` uses float endpoints and an `i64` count;
  `arange(start, stop)` uses signed-integer endpoints.
- `stack(xs, axis)` concatenates tensors along a new axis.
- `squeeze(x, axis)` removes a size-1 dimension; `unsqueeze(x, axis)` inserts
  one.

The three rank-changing exports are specified but their public signatures do
not yet type-check for concrete tensor callers; see [chelis#1416](https://github.com/Chelis-Lang/chelis/issues/1416).
The checker does not yet enforce the endpoint dtype families
([chelis#1417](https://github.com/Chelis-Lang/chelis/issues/1417)), and C builds
do not yet actualize the helpers' generic cast targets
([chelis#1418](https://github.com/Chelis-Lang/chelis/issues/1418)).

`Std.Tensor.Mask`:

- `where_indices(mask)` returns the `i64` indices where a `bool` mask is true.

### Initializers

`Std.Init.Random`:

- `normal_like(k, template, mean, std)` draws a normal tensor shaped like the template. It
  splits `k` and draws its two Box-Muller uniforms from the halves.

`Std.Init.Kaiming`:

- `kaiming_uniform(k, template, fan_in)`, `kaiming_normal(k, template, fan_in)`.

`Std.Init.XavierExt`:

- `xavier_uniform(k, template, fan_in, fan_out)`, `xavier_normal(k, template, fan_in, fan_out)`.
- `trunc_normal(k, template, mean, std, a, b)` draws a normal tensor clipped to `[a, b]`.

Every initializer takes its key first, consumes it, and carries no effect. The same key
gives the same tensor, so `kaiming_uniform(key_from_seed(7i64), w, 4.0)` is reproducible;
initialising two tensors takes two keys, for example the halves of one `split_key`. A draw
in a conditional branch that does not run is not evaluated.

`normal_like`, `kaiming_uniform`, `xavier_uniform` and `trunc_normal` are each written in
two layers inside their module: a draw from the key, and a pure layer that turns the draws
into the result. `normal_like(k, template, mean, std)` is
`normal_like_given(units, template, mean, std)` applied to `normal_like_sample(k, template)`;
the two uniform initializers apply `kaiming_uniform_given` and `xavier_uniform_given` to a
unit `uniform_like` draw, and `trunc_normal` applies `trunc_normal_given` to a
`normal_like` draw. `kaiming_normal` and `xavier_normal` call `normal_like` directly. The
layers are private to their modules; the exported function taking the key first is the
public surface.

### Sorting and scanning

`Std.Sort`:

- `sort(values, axis)` accepts a numeric tensor of any rank and returns
  `(sorted_values, indices)`, with both tensors preserving the input shape and
  the indices using `i64`.

`Std.Scan`:

- `scan_list(step, initial, values)` returns the running accumulator at each step.

`Std.Index`:

- `list_index(values, idx)`, `take_list(values, count)`, `skip_list(values, count)`.
  Their adjoints preserve the input List's runtime length and positions: index
  routes the cotangent to the selected element, while take/skip fill excluded
  positions with zeros. Negative indices/counts fail; take/skip counts beyond
  the length retain their ordinary truncation behavior. Scalar, tensor, empty,
  nested, and multiple-List targets use the same rule, including runtime
  selectors/counts reused elsewhere in the differentiated body and selection
  composed through wrappers. These public `grad(...)` calls run in both the
  evaluator and generated-C programs.

### Decimal and time

`Std.Decimal` is fixed-point decimal, a coefficient and a scale. Construction with
`decimal(text)` or `decimal_from_int(value)`; arithmetic with `decimal_add`, `decimal_sub`,
`decimal_mul`, and `decimal_div(lhs, rhs, result_scale, mode)`; comparison with
`decimal_eq`, `decimal_lt`, and the rest; conversion with `decimal_to_float` and
`decimal_to_string`. Rounding modes are `round_half_up`, `round_half_even`, `round_down`,
and `round_up`.

`Std.Time` callables currently raise an error citing #2779. Their intended
proleptic Gregorian API includes `date(year, month, day)`, `add_days`,
`sub_days`, `days_between`, date comparisons, `day_of_week`, `day_of_year`,
`is_leap_year`, `date_to_string`, and `parse_date`. Use of these operations
requires an exact implementation of [05-OP-35].

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
`JsonBigInt`, `JsonFloat`, `JsonString`, `JsonArray`, `JsonObject`; the constructors
are exported, so documents can be built directly). Integer-form tokens outside
the `i64` range retain their exact spelling as `JsonBigInt`; float-form tokens whose f64
image is non-finite are rejected:

- `load_json(path)`, `parse_json(text)` and their `try_` variants.
- `json_get`, and the typed accessors `json_string`, `json_int`, `json_bigint`,
  `json_float`, `json_bool`, `json_array`, `json_object`, plus `json_is_null`.
- `to_json(value)` renders a `Json` value compactly (object keys recursively
  sorted by increasing Unicode scalar-value sequence before escaping, f64 via
  `to_string`'s shortest-round-trip form — a claim made for **f64
  specifically**, the dtype `JsonFloat` carries — and RFC 8259 escaping for
  quotes, backslashes, and every control character). Equal object mappings
  therefore produce the same bytes regardless of insertion history.
  Non-finite numbers have no JSON representation: `to_json`
  fails on them and `try_to_json` returns `None`. `write_json(path, value)`
  writes the rendered text and names the path on failure; `try_write_json` is
  its `Option` twin. The parser decodes all four-hex-digit `\uXXXX` escapes,
  combines valid UTF-16 surrogate pairs into one Unicode scalar, and rejects
  malformed or unpaired surrogate sequences.

These IO modules carry the `IO` effect and run in **both lanes**: under
`chelis eval`/`chelis test` and inside compiled `chelis build` programs alike.
`Std.Io.Json` is the sole public JSON value surface. `Std.Io.Csv` is distinct
from the eval-only prelude CSV builtins (`parse_csv`/`to_csv`, chelis#903),
which `chelis build` rejects whole-program; reef package name-rewriting keeps
the shared CSV names apart in both lanes.

### Tokenization

`Std.Tokenizer` is a byte-pair tokenizer that loads a Hugging-Face-style `tokenizer.json`:

- `load_tokenizer(path)` and `try_load_tokenizer(path)`.
- `encode(tokenizer, text)`, `decode(tokenizer, ids)`.
- `batch_encode(tokenizer, texts, max_length, pad_value)` returns a padded `i64` tensor.

### Testing

`Std.Test` provides assertion helpers for `def test_*()` functions discovered by
`chelis test`. They carry the `Test` effect:

- `assert_true`, `assert_false`, generic `assert_eq`, and `assert_close`.
- `assert_close_tensor`, generic `assert_eq_tensor`, and `assert_shape` for tensors.
- `fail(msg)`.

## Process execution

`Std.Process` runs external programs through the host:

- `run(cmd, args)` returns `(exit_code, stdout, stderr)`.
- `run_chelis(args)` runs the Chelis binary itself.

These run on the evaluator and test paths.
