# Runtime and Standard Library

Chelis has compiler built-ins for tensor, scalar, collection, and host operations.
The compiler also bundles `chelis-std`, whose modules use the `Std` prefix. In a
Reef package or a standalone `.ch` file, import the names you need, for example
`import Std.Sort (sort)`; there is no separate standard-library install. The `Std` source modules and the
native runtime archive emitted by `chelis build` are different parts of the
runtime.

This page covers commonly used names and which of them run. The
[operation specification](https://github.com/Chelis-Lang/chelis/blob/main/spec/05-risc-primitives.md) defines the full
signatures, failure rules, and differentiation behavior. The module sources
are in `packages/chelis-std/src/`.

## Built-in operations

Built-ins are called by name without an import. Tensor operations preserve the
dimension and dtype rules of their signatures: there is no implicit
broadcasting or precision promotion. These operations also have scalar,
`List`, or `key` forms where stated below.

```chelis-surf
module TensorSummary
values = to_tensor([1.0f32, 2.0f32, 3.0f32])
total = sum(values, 0i32)
```

Save this as `summary.ch` and run `chelis eval --file summary.ch` to see the
tensor and its sum.

### Arithmetic and comparison

- `add`, `sub`, and `mul` accept matching signed-integer or float scalars, or
  same-shaped tensors of one dtype. `div` accepts floats only. `floor_div`
  accepts numeric values; `trunc_div` and `mod` accept signed integers.
- `max_elem` and `min_elem` select element-wise extrema. `cmplt` or `lt`,
  `eq`, `neq`, `gt`, `gte`, and `lte` compare values and return `bool` values
  or tensors. `and`, `or`, and `not` operate on booleans.
- `neg` and `abs` accept signed integers and floats. `recip`, `exp`, `log`,
  `sqrt`, `sin`, `cos`, `tan`, and `atan` accept floats. `floor`, `ceil`, and
  `round` also accept integers, for which they are identities.
- `relu`, `sigmoid`, `tanh`, `silu`, and `gelu` are float activations.

### Reductions and windows

`sum`, `mean`, `max_reduce`, `min_reduce`, and `prod_reduce` remove one or more
selected axes. An axis must resolve from a constant or a named dimension when
the program is checked; an arbitrary runtime integer axis is not accepted.
`mean` requires floats. `sum` has an optional accumulator dtype, which can
affect its result dtype. `count` counts true `bool` elements and returns
an `i64` tensor; `argmax_reduce` and `argmin_reduce` each take one axis
and return `i64` index tensors. `softmax` takes a float tensor and
preserves its shape.

`reduce_window_sum`, `reduce_window_mean`, `reduce_window_max`, and
`reduce_window_min` take a tensor, a window shape, and strides. `mean` requires
floats. The evaluator supports these operations; generated C supports selected
window shapes and dtypes. HIP and Metal builds reject unsupported window
operations rather than running them on the CPU without notice.

### Shapes, indexing, and construction

- `reshape` preserves the element sequence while changing shape; `permute`
  reorders axes. `insert` adds a dimension, while `expand` repeats an existing
  dimension of size one without changing rank. Use them to align shapes
  explicitly. `pad` adds boundary elements; `shrink` slices a region.
- `concat` joins tensors along an axis; `split` divides one tensor. `gather`
  selects indexed values; `scatter` updates them. `scatter` supports
  `"replace"` (the last write wins for a repeated index) and `"add"`.
- `where` selects between same-dtype tensor branches using a boolean mask;
  `clamp` bounds values; `cumsum` computes cumulative sums. `einsum` uses an
  equation and two tensors for contractions. `matmul` multiplies matrices,
  including supported batched forms.
- `sort` returns values and `i64` indices. `diagonal` selects paired axes;
  `trace` reduces a diagonal.
- `to_tensor` converts a rectangular nested list to a tensor; `to_list`
  converts a tensor to a list. `pad_sequences` makes a rectangular tensor from
  ragged rows. `cast` changes a numeric dtype explicitly. `shape(t, axis)`
  returns an `i64` extent for an `i32` axis; `rank` and `numel` report rank
  and element count. `copy` creates an owned value, and `realize`
  materializes an intermediate.

### Explicit randomness

`key_from_seed(i64)` creates a key. `split_key`, `split_keys`, and `fold_in`
derive keys. `dropout(k, x, rate)` and `uniform_like(k, x, low, high)` consume
the given key and contribute no effect. Split a key before making two draws;
the same key cannot be consumed twice.

## `Std` modules

### Tensor and general helpers

- `Std.Tensor.Construct` provides `arange(start, stop)` for signed-integer
  endpoints and `linspace(start, stop, count)` for float endpoints and an
  `i64` count. These functions run for supported inputs in evaluation and
  generated C. Imported calls do not enforce every endpoint dtype family
  restriction, so use the stated types. Its `stack`, `squeeze`, and
  `unsqueeze` names are exported, but concrete tensor calls do not
  type-check; do not depend on them in a runnable program.
- `Std.Tensor.Mask.where_indices(mask)` returns the increasing flat `i64`
  indices of true elements. `Std.Sort.sort` exposes the built-in tensor sort
  through a module import.
- `Std.Scan.scan_list(step, initial, values)` returns running accumulator
  values. `Std.Index` provides `list_index`, `take_list`, and `skip_list`;
  negative indices or counts fail, while take/skip counts beyond the list
  length truncate. Differentiation follows the
  [standard-library operation rules](https://github.com/Chelis-Lang/chelis/blob/main/spec/05-risc-primitives.md).
- `Std.Scalar` exports numeric `max`, `min`, and `abs` for scalars.
  `Std.Text.join(parts, sep)` joins strings. `Std.Contracts` provides
  `normal_cdf`, contract names, and settings for numerical tests.

### Files, CSV, and JSON

`Std.Io` provides `read_text`, `write_text`, `read_trimmed_lines`,
`read_head_bytes`, `exists`, `list`, and `mmap_size`. These file operations
carry the `IO` effect.

`Std.Io.Csv` reads header-keyed `List[Dict[string, string]]` rows with
`read_csv(path)` or `try_read_csv(path)`. `to_csv(rows)` and
`try_to_csv(rows)` render text; `write_csv(path, rows)` and
`try_write_csv(path, rows)` write it. The `try_` forms return `None` for
parsing or serialization failures. CSV fields remain strings. Its reader is
line-based, so CR or LF inside a header or field cannot round-trip; its writer
rejects those values rather than emitting unreadable CSV.

`Std.Io.Json` exports `Json` and its constructors, including `JsonInt(i64)`,
`JsonBigInt(string)` for integer text outside `i64`, and `JsonFloat(f64)`.
Use `parse_json(text)` or `try_parse_json(text)` for text,
`load_json(path)` or `try_load_json(path)` for files, and `json_get` and the
`json_*` accessors to inspect values. `json_int` returns only a stored
`JsonInt`; `json_float` also accepts a `JsonInt` through an explicit
`i64`-to-`f64` conversion and does not accept `JsonBigInt`. `to_json` and
`try_to_json` render text; `write_json` and `try_write_json` write files.
Non-finite `JsonFloat` values and invalid `JsonBigInt` text cannot be
serialized.

Parsing, accessors, and text rendering in the CSV and JSON modules are pure.
Their file operations carry `IO`. The file operations above run
under `chelis eval` and in supported generated C host programs.

### Tests and processes

`Std.Test` provides `assert_true`, `assert_false`, `assert_eq`,
`assert_close`, `assert_close_tensor`, `assert_eq_tensor`, `assert_shape`,
and `fail` for `def test_*()` functions run by `chelis test`. Assertions carry
the `Test` effect. Generated builds reject these assertion calls.

`Std.Process.run(cmd, args)` passes an argument list to an external program
and returns `(exit_code, stdout, stderr)`. `run_chelis(args)` invokes the
`chelis` command. Both carry `IO` and run during evaluation and
testing. The language specifies compiled host process execution too, but
`chelis build` rejects these calls.

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
`chelis eval`, `chelis test`, and generated C. `chelis build` refuses a program
that projects `.0` from a masked column producer such as
`try_dates_from_epoch_days`
([#2883](https://github.com/Chelis-Lang/chelis/issues/2883)); destructure the
pair instead.

`Std.Datetime.Clock` reads the host clocks. It is a separate module, so
nothing in `Std.Datetime` reads a clock; any code that reads one, through this
module or the underlying builtins, carries `IO`. `clock_now()`
returns the wall-clock `Instant`. `monotonic_now()` returns a `MonotonicInstant`
from a clock that never runs backwards, and `monotonic_until(a, b)` is the exact
`Duration` from `a` to `b`. A `MonotonicInstant` has an unspecified origin, so
no other datetime function takes one and it does not convert to an `Instant`. Both reads carry
`IO`, so every caller carries it, declared or inferred. A failed read reports
`clock_wall_read: io: <detail>` or `clock_monotonic_read: io: <detail>`, naming
the underlying read. The clocks run under `chelis eval` and `chelis test`;
`chelis build` rejects them.

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
fails instead of rounding. `decimal_to_f64` and `decimal_to_f32` are the one
other conversion that drops precision, rounding once to nearest with ties to
even. `decimal_to_string` writes the canonical text without an exponent, and
`decimal_to_fixed_string(x, n)` writes exactly `n` fractional digits and never
rounds. A failure reports `<function>: domain: <detail>` for an argument that
denotes no value of its domain, including text or a float whose value lies
outside the decimal range, or `<function>: overflow: <detail>` when the exact
result of arithmetic lies outside that range or `decimal_to_i64`'s integer lies
outside i64; each `try_` form returns `None` where its twin fails `domain`, and
`try_decimal_to_i64` also where its twin fails `overflow`. The module is
pure and runs under `chelis eval`, `chelis test`, and generated C.

`Std.Datetime.Zone` makes time zone rules values. `time_zone_from_tzif(name,
bytes)` reads a TZif file (RFC 9636, versions 2 to 4) whose bytes the program
supplies: the standard library holds no time zone database and never reads the
host's. `time_zone_fixed(o)` and `time_zone_utc()` build fixed zones. A `Zoned` is
an instant in a zone: `zoned(i, tz)` pairs them, and `zoned_from_local(dt, tz,
policy)` resolves a local reading, where the `Disambiguation` (`EarlierInstant`,
`LaterInstant`, `CompatibleInstant`, or `RejectNonUniqueLocal`) decides a reading
that a daylight saving change skips or repeats. `zoned_add_duration` moves along
the instant line and `zoned_add_period` moves the wall clock. `zoned_to_string`
writes RFC 9557 text such as `2026-10-01T09:30:00-04:00[America/New_York]`, with an
offset whose seconds are nonzero written `±HH:MM:SS` as the text profile allows;
`parse_zoned_text` reads it into a plain `ZonedText` record, and
`zoned_from_text(zt, tz, policy)` resolves that against a zone the caller
obtained, with an `OffsetConflict` policy for a written offset the zone does not
use there. A written `Z`, `z` or `-00:00` means a UTC time whose local offset is
unknown (RFC 9557): its record has no offset, and every policy resolves it to that
UTC instant. A zone whose TZif footer is empty has no offsets from its last
transition on, and a call that needs one there fails `domain`.

### Exported but unavailable

`Std.Io.Parquet` and `Std.Io.Safetensors` export names whose calls fail. Use
the modules above for runnable programs.
