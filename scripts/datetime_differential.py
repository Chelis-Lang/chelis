#!/usr/bin/env python3
"""Differential oracle for `Std.Datetime` (chelis#2859, chelis#2861): eval and C against the reference.

The harness generates Chelis programs that call `Std.Datetime`, runs each one
through `chelis eval` and through `chelis build --target c` plus the native
compiler, and compares each lane's complete output with the values that
`datetime_reference.py` computes independently. A lane agreeing with the other
lane is not enough: each must equal the reference.

Three kinds of observation are generated:

- A grid is one binding that loops in Chelis over small input lists and
  ranges and prints a flat `List[i64]` (or `List[string]`) of results; the
  reference computes the same rows in the same order. Opaque values are
  observed through their exported accessors and an `Option` as a presence
  flag followed by its value. Inputs come from ranges, short literal lists,
  and an in-language linear congruential generator that the reference mirrors,
  because long list literals are slow for `chelis build` and overflow its
  stack.
- A value is one binding whose printed value the reference predicts, used
  where a grid does not fit (strings, floats compared bit for bit, columns).
- A failure is a program whose single binding must fail. Both lanes must exit
  nonzero with the same §5 message, `<function>: <kind>: <detail>`, whose
  function and kind equal the reference's and whose detail is nonempty. This
  also shows that no primitive numeric trap escapes (§5): a trap would carry a
  primitive's message instead. Repeated paths are sampled; every grid point is
  still observed through the `try_` forms.

Every printed line must belong to a binding of the program: a stray line, such
as a library constant printed by one lane, is a disagreement.

Profiles:

- `ci`: the range edges, seeded samples across the whole range, a stride
  through 1900-2100, every year's calendar queries and Easter, every policy
  branch and the text corpus on both lanes, plus every day from 1900-01-01
  through 2100-12-31 in compiled C. (`chelis eval` takes about 14 ms per day
  row, so the full 1900-2100 sweep on eval belongs to the manual profile.)
  Its column corpus calls every `Std.Datetime.Columns` callable on both lanes
  over seeded columns that include the range edges, under every `TimeUnit`,
  `Rounding` mode and `DayOverflow` policy; column bindings run in programs of
  their own, and each column failure is a named program of its own.
- `canary`: five unchanged `ci` programs on both lanes (`CANARY_PROGRAMS`):
  the first and last 400 days of the range, one column value per family of
  `COLUMN_FAMILIES`, a domain failure and an overflow failure. It runs on every pull request; `ci` runs nightly.
- `exhaustive`: the manual gate of `docs/manual_gates.md`. Every one of the
  7 304 484 days in compiled C (field round trip, weekday, day of year, ISO
  week and its inverse, text round trip), `date_period_until`'s defining
  property on 200 000 seeded pairs in C, and every day 1900-2100 on eval.

The supported entry point is `crates/chelis-cli/tests/std_datetime_oracle.rs`,
which supplies the freshly built binary, a published standard library and the
strict reference toolchain (`chelis_backend_c::toolchain`).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import itertools
import random
from pathlib import Path
import re
import struct
import sys
import tempfile

from datetime_lanes import LaneResult, Runner, Toolchain
import datetime_reference as ref
from datetime_reference import DatetimeError, I64_MAX, I64_MIN

SEED = 2859
SPAN = ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY + 1
FIRST_1900 = ref.days_from_civil(1900, 1, 1)
LAST_2100 = ref.days_from_civil(2100, 12, 31)
PASS_MARKER = "STD DATETIME ORACLE: PASS"
FAILURES_PER_PATH = 2
COLUMN_CHUNK = 25
BOTH = ("eval", "c")
C_ONLY = ("c",)
EVAL_ONLY = ("eval",)
NANO = ref.NANOS_PER_SECOND
DAY_NANOS = ref.NANOS_PER_DAY

TYPES = (
    "Date", "Time", "DateTime", "Instant", "Offset", "OffsetDateTime", "Duration", "Period",
    "Weekday", "DayOverflow", "TimeUnit", "Dates", "Instants",
)
CONSTRUCTORS = ref.WEEKDAYS + ref.DAY_OVERFLOW + tuple(ref.TIME_UNIT_NANOS)
COLUMN_TYPES = ("Durations",)
COLUMN_FUNCTIONS = (
    "durations", "try_durations", "durations_seconds", "durations_nanoseconds",
    "dates_year", "dates_month", "dates_day", "dates_weekday_iso_number", "dates_day_of_year",
    "dates_from_ymd", "try_dates_from_ymd", "dates_add_days", "dates_add_months", "dates_days_until",
    "dates_lt", "dates_lte", "dates_gt", "dates_gte", "try_parse_dates", "dates_to_strings",
    "instants_from_unix_count", "instants_to_unix_count", "instants_add_duration", "instants_until",
    "instants_round_to", "instants_to_dates_at", "instants_seconds_since_f64",
    "instants_lt", "instants_lte", "instants_gt", "instants_gte",
)
# `COLUMN_FUNCTIONS` by family: the canary's column program calls at least
# one callable of each.
COLUMN_FAMILIES = {
    "durations": ("durations", "try_durations", "durations_seconds", "durations_nanoseconds"),
    "date fields": ("dates_year", "dates_month", "dates_day", "dates_weekday_iso_number", "dates_day_of_year"),
    "date construction and text": ("dates_from_ymd", "try_dates_from_ymd", "try_parse_dates", "dates_to_strings"),
    "date arithmetic": ("dates_add_days", "dates_add_months", "dates_days_until"),
    "date order": ("dates_lt", "dates_lte", "dates_gt", "dates_gte"),
    "instant counts": ("instants_from_unix_count", "instants_to_unix_count"),
    "instant arithmetic": ("instants_add_duration", "instants_until"),
    "instant rounding": ("instants_round_to",),
    "instant conversion": ("instants_to_dates_at", "instants_seconds_since_f64"),
    "instant order": ("instants_lt", "instants_lte", "instants_gt", "instants_gte"),
}
FUNCTIONS = (
    "is_leap_year", "days_in_year", "days_in_month", "weekday_iso_number",
    "weekday_from_iso_number", "try_weekday_from_iso_number", "weekday_name",
    "date", "try_date", "date_year", "date_month", "date_day", "date_epoch_day",
    "date_from_epoch_day", "try_date_from_epoch_day", "date_weekday", "date_day_of_year",
    "date_iso_week", "date_from_iso_week", "try_date_from_iso_week", "date_add_days",
    "date_days_until", "date_add_months", "try_date_add_months", "date_add_period",
    "try_date_add_period", "date_period_until", "date_lt", "date_lte", "date_gt", "date_gte",
    "date_to_string", "parse_date", "try_parse_date",
    "nth_weekday_in_month", "last_weekday_in_month", "weekday_on_or_after",
    "weekday_on_or_before", "easter_sunday_gregorian", "easter_sunday_orthodox",
    "time", "try_time", "time_hour", "time_minute", "time_second", "time_nanosecond",
    "time_nanosecond_of_day", "time_from_nanosecond_of_day", "try_time_from_nanosecond_of_day",
    "time_add_duration", "time_until", "time_lt", "time_lte", "time_gt", "time_gte",
    "time_to_string", "parse_time", "try_parse_time",
    "datetime", "datetime_date", "datetime_time", "datetime_add_duration",
    "datetime_add_period", "try_datetime_add_period", "datetime_until",
    "datetime_lt", "datetime_lte", "datetime_gt", "datetime_gte",
    "datetime_to_string", "parse_datetime", "try_parse_datetime",
    "offset_from_seconds", "try_offset_from_seconds", "offset_seconds",
    "offset_to_string", "parse_offset", "try_parse_offset",
    "instant_from_unix", "try_instant_from_unix", "instant_unix_second", "instant_nanosecond",
    "instant_from_unix_count", "try_instant_from_unix_count", "instant_to_unix_count",
    "try_instant_to_unix_count", "instant_add_duration", "instant_until", "instant_round_to",
    "instant_lt", "instant_lte", "instant_gt", "instant_gte", "instant_to_datetime_at",
    "datetime_to_instant_at", "instant_to_string", "parse_instant", "try_parse_instant",
    "offset_datetime", "offset_datetime_instant", "offset_datetime_offset",
    "offset_datetime_local", "offset_datetime_to_string", "parse_offset_datetime",
    "try_parse_offset_datetime",
    "duration", "duration_from_count", "duration_second", "duration_nanosecond",
    "duration_to_count", "try_duration_to_count", "duration_to_seconds_f64", "duration_add",
    "duration_sub", "duration_negate", "duration_mul", "duration_lt", "duration_lte",
    "duration_gt", "duration_gte", "duration_to_string", "parse_duration", "try_parse_duration",
    "period", "try_period", "period_months", "period_days", "period_negate", "period_mul",
    "period_to_string", "parse_period", "try_parse_period",
    "dates_from_epoch_days", "try_dates_from_epoch_days", "dates_epoch_days",
    "instants_from_unix", "try_instants_from_unix", "instants_unix_seconds", "instants_nanoseconds",
)

# Encoders print opaque values through accessors; `row_*` helpers return the
# flat `List[i64]` rows that grids concatenate. `@NAME@` tokens are filled in
# by `prelude()`.
PRELUDE = """\
def b2i(x: bool) -> i64 = if x then 1i64 else 0i64

def lcg(x: i64) -> i64 = mod(x * 1103515245i64 + 12345i64, 2147483648i64)

def mix(k: i64) -> i64 = lcg(lcg(k + @SEED@i64))

def sample_day(k: i64) -> i64 = @MIN@ + mod(mix(k), @SPAN@i64)

def sample_year(k: i64) -> i64 = (-9998i64) + mod(mix(k), 19997i64)

def near_day(k: i64) -> i64 = @INNER_MIN@ + mod(mix(k), @INNER_SPAN@i64)

def p2a(x: (i64, i64)) -> i64 = {
  (a, b) = x
  a
}

def p2b(x: (i64, i64)) -> i64 = {
  (a, b) = x
  b
}

def p3a(x: (i64, i64, i64)) -> i64 = {
  (a, b, c) = x
  a
}

def p3b(x: (i64, i64, i64)) -> i64 = {
  (a, b, c) = x
  b
}

def p3c(x: (i64, i64, i64)) -> i64 = {
  (a, b, c) = x
  c
}

def p4a(x: (i64, i64, i64, i64)) -> i64 = {
  (a, b, c, d) = x
  a
}

def p4b(x: (i64, i64, i64, i64)) -> i64 = {
  (a, b, c, d) = x
  b
}

def p4c(x: (i64, i64, i64, i64)) -> i64 = {
  (a, b, c, d) = x
  c
}

def p4d(x: (i64, i64, i64, i64)) -> i64 = {
  (a, b, c, d) = x
  d
}

def dur2(x: (i64, i64)) -> Duration = duration(p2a(x), p2b(x))

def inst2(x: (i64, i64)) -> Instant = instant_from_unix(p2a(x), p2b(x))

def dt2(x: (i64, i64)) -> DateTime = datetime(date_from_epoch_day(p2a(x)), time_from_nanosecond_of_day(p2b(x)))

def per2(x: (i64, i64)) -> Period = period(p2a(x), p2b(x))

def row_dt(x: DateTime) -> List[i64] = [date_epoch_day(datetime_date(x)), time_nanosecond_of_day(datetime_time(x))]

def row_instant(x: Instant) -> List[i64] = [instant_unix_second(x), instant_nanosecond(x)]

def row_duration(x: Duration) -> List[i64] = [duration_second(x), duration_nanosecond(x)]

def row_period(x: Period) -> List[i64] = [period_months(x), period_days(x)]

def row_odt(x: OffsetDateTime) -> List[i64] = [instant_unix_second(offset_datetime_instant(x)), instant_nanosecond(offset_datetime_instant(x)), offset_seconds(offset_datetime_offset(x))]

def row_carry(x: (i64, Time)) -> List[i64] = {
  (carry, t) = x
  [carry, time_nanosecond_of_day(t)]
}

def row_opt_i64(o: Option[i64]) -> List[i64] = match o with {
  | Some(v) => [1i64, v]
  | None => [0i64, 0i64]
}

def row_opt_date(o: Option[Date]) -> List[i64] = match o with {
  | Some(v) => [1i64, date_epoch_day(v)]
  | None => [0i64, 0i64]
}

def row_opt_time(o: Option[Time]) -> List[i64] = match o with {
  | Some(v) => [1i64, time_nanosecond_of_day(v)]
  | None => [0i64, 0i64]
}

def row_opt_weekday(o: Option[Weekday]) -> List[i64] = match o with {
  | Some(v) => [1i64, weekday_iso_number(v)]
  | None => [0i64, 0i64]
}

def row_opt_offset(o: Option[Offset]) -> List[i64] = match o with {
  | Some(v) => [1i64, offset_seconds(v)]
  | None => [0i64, 0i64]
}

def row_opt_dt(o: Option[DateTime]) -> List[i64] = match o with {
  | Some(v) => [1i64, date_epoch_day(datetime_date(v)), time_nanosecond_of_day(datetime_time(v))]
  | None => [0i64, 0i64, 0i64]
}

def row_opt_instant(o: Option[Instant]) -> List[i64] = match o with {
  | Some(v) => [1i64, instant_unix_second(v), instant_nanosecond(v)]
  | None => [0i64, 0i64, 0i64]
}

def row_opt_duration(o: Option[Duration]) -> List[i64] = match o with {
  | Some(v) => [1i64, duration_second(v), duration_nanosecond(v)]
  | None => [0i64, 0i64, 0i64]
}

def row_opt_period(o: Option[Period]) -> List[i64] = match o with {
  | Some(v) => [1i64, period_months(v), period_days(v)]
  | None => [0i64, 0i64, 0i64]
}

def row_opt_odt(o: Option[OffsetDateTime]) -> List[i64] = match o with {
  | Some(v) => [1i64, instant_unix_second(offset_datetime_instant(v)), instant_nanosecond(offset_datetime_instant(v)), offset_seconds(offset_datetime_offset(v))]
  | None => [0i64, 0i64, 0i64, 0i64]
}

def day_row(n: i64) -> List[i64] = {
  d = date_from_epoch_day(n)
  y = date_year(d)
  m = date_month(d)
  dd = date_day(d)
  w = date_weekday(d)
  (iy, iw) = date_iso_week(d)
  [y, m, dd, date_epoch_day(date(y, m, dd)), weekday_iso_number(w), date_day_of_year(d), iy, iw, date_epoch_day(date_from_iso_week(iy, iw, w)), date_epoch_day(parse_date(date_to_string(d)))]
}

def try_column_days[n](x: (Dates[n], tensor[n, bool])) -> (List[i64], List[bool]) = {
  (ds, mask) = x
  (to_list(dates_epoch_days(ds)), to_list(mask))
}

def try_column_seconds[n](x: (Instants[n], tensor[n, bool])) -> (List[i64], List[bool]) = {
  (xs, mask) = x
  (to_list(instants_unix_seconds(xs)), to_list(mask))
}

def try_column_nanos[n](x: (Instants[n], tensor[n, bool])) -> List[i64] = {
  (xs, mask) = x
  to_list(instants_nanoseconds(xs))
}

def instants_row[n](xs: Instants[n]) -> (List[i64], List[i64]) = {
  seconds = instants_unix_seconds(xs)
  (to_list(seconds), to_list(instants_nanoseconds(xs)))
}

def durations_row[n](ds: Durations[n]) -> (List[i64], List[i64]) = {
  seconds = durations_seconds(ds)
  (to_list(seconds), to_list(durations_nanoseconds(ds)))
}

def try_column_durations[n](x: (Durations[n], tensor[n, bool])) -> ((List[i64], List[i64]), List[bool]) = {
  (ds, mask) = x
  (durations_row(ds), to_list(mask))
}

def span[k](count: i64) -> tensor[k, i64] = to_tensor(range(0i64, count))

def year_row(y: i64) -> List[i64] = [days_in_year(y), b2i(is_leap_year(y)), date_epoch_day(easter_sunday_gregorian(y)), date_epoch_day(easter_sunday_orthodox(y))]

def period_row(a: i64, b: i64) -> List[i64] = {
  p = date_period_until(date_from_epoch_day(a), date_from_epoch_day(b))
  [period_months(p), period_days(p), date_epoch_day(date_add_period(date_from_epoch_day(a), p, ClampToMonthEnd))]
}
"""


def prelude() -> str:
    return (PRELUDE.replace("@SEED@", str(SEED)).replace("@MIN@", lit(ref.MIN_EPOCH_DAY))
            .replace("@SPAN@", str(SPAN)).replace("@INNER_MIN@", lit(ref.MIN_EPOCH_DAY + 2000))
            .replace("@INNER_SPAN@", str(SPAN - 4000)))


def prelude_defs() -> dict[str, str]:
    """The prelude's definitions by name, in source order."""
    defs: dict[str, str] = {}
    for block in prelude().split("\n\ndef "):
        block = block if block.startswith("def ") else "def " + block
        defs[re.match(r"def (\w+)", block).group(1)] = block.strip() + "\n"
    return defs


def prelude_for(body: str) -> str:
    """Only the prelude definitions `body` reaches: each one costs `chelis build` time."""
    defs = prelude_defs()
    needed: set[str] = set()
    pending = [body]
    while pending:
        text = pending.pop()
        for name, block in defs.items():
            if name not in needed and re.search(rf"\b{name}\(", text):
                needed.add(name)
                pending.append(block)
    return "\n".join(block for name, block in defs.items() if name in needed)


# Python mirrors of the prelude's input generators. They only choose inputs;
# expected values always come from `datetime_reference`.
def lcg(x: int) -> int:
    return (x * 1103515245 + 12345) % 2147483648


def mix(k: int) -> int:
    return lcg(lcg(k + SEED))


def sample_day(k: int) -> int:
    return ref.MIN_EPOCH_DAY + mix(k) % SPAN


def sample_year(k: int) -> int:
    return -9998 + mix(k) % 19997


def near_day(k: int) -> int:
    return ref.MIN_EPOCH_DAY + 2000 + mix(k) % (SPAN - 4000)


# ---------------------------------------------------------------------------
# Chelis source builders.


def lit(n: int) -> str:
    if not ref.fits_i64(n):
        raise ValueError(f"{n} does not fit in i64")
    if n == I64_MIN:
        return "(-9223372036854775807i64 - 1i64)"
    return f"({n}i64)" if n < 0 else f"{n}i64"


def text_lit(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def list_lit(values: Sequence[int]) -> str:
    return "[" + ", ".join(lit(v) for v in values) + "]"


def tuple_lit(values: Sequence[int]) -> str:
    return "(" + ", ".join(lit(v) for v in values) + ")"


def date_src(epoch_day: int) -> str:
    return f"date_from_epoch_day({lit(epoch_day)})"


def time_src(nod: int) -> str:
    return f"time_from_nanosecond_of_day({lit(nod)})"


def dt_src(dt: tuple[int, int]) -> str:
    return f"datetime({date_src(dt[0])}, {time_src(dt[1])})"


def instant_src(i: tuple[int, int]) -> str:
    return f"instant_from_unix({lit(i[0])}, {lit(i[1])})"


def offset_src(seconds: int) -> str:
    return f"offset_from_seconds({lit(seconds)})"


def duration_src(d: tuple[int, int]) -> str:
    return f"duration({lit(d[0])}, {lit(d[1])})"


def period_src(p: tuple[int, int]) -> str:
    return f"period({lit(p[0])}, {lit(p[1])})"


def weekday_src(iso: int) -> str:
    return ref.WEEKDAYS[iso - 1]


# ---------------------------------------------------------------------------
# Rendering: the reference value printed the way both lanes print it.


def render(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(render(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(render(v) for v in value) + "]"
    raise TypeError(f"cannot render {value!r}")


def try_value(compute):
    """A `try_` form: None exactly where the reference fails with `domain`."""
    try:
        return compute()
    except DatetimeError as error:
        if error.kind == "domain":
            return None
        raise


def opt_row(compute, width: int) -> list[int]:
    """`[present, fields...]` for a `try_` form, as the prelude's `row_opt_*` prints it."""
    value = try_value(compute)
    if value is None:
        return [0] * (width + 1)
    return [1, *flatten(value)]


def flatten(value: object) -> list[int]:
    if isinstance(value, bool):
        return [int(value)]
    if isinstance(value, int):
        return [value]
    if isinstance(value, tuple):
        return [x for part in value for x in flatten(part)]
    raise TypeError(value)


def expect_error(compute) -> DatetimeError:
    try:
        compute()
    except DatetimeError as error:
        return error
    raise AssertionError("the reference was expected to fail")


def succeeds(compute) -> bool:
    try:
        compute()
    except DatetimeError:
        return False
    return True


@dataclass(frozen=True)
class Value:
    """A binding whose printed value the reference predicts."""

    name: str
    expr: str
    expected: object
    float_bits: bool = False
    lanes: tuple[str, ...] = BOTH

    def resolve(self) -> object:
        """Grids carry a zero-argument function; compute it on demand."""
        return self.expected() if callable(self.expected) else self.expected


@dataclass(frozen=True)
class Failure:
    """A program whose binding must fail with `<function>: <kind>: <detail>`."""

    name: str
    expr: str
    function: str
    kind: str


@dataclass(frozen=True)
class Axis:
    var: str
    type: str
    source: str
    values: tuple


def int_axis(var: str, values: Sequence[int]) -> Axis:
    return Axis(var, "i64", list_lit(values), tuple(values))


def range_axis(var: str, low: int, high: int) -> Axis:
    return Axis(var, "i64", f"range({lit(low)}, {lit(high)})", tuple(range(low, high)))


def tuple_axis(var: str, values: Sequence[tuple[int, ...]]) -> Axis:
    arity = len(values[0])
    kind = "(" + ", ".join(["i64"] * arity) + ")"
    return Axis(var, kind, "[" + ", ".join(tuple_lit(v) for v in values) + "]", tuple(values))


def ctor_axis(var: str, kind: str, names: Sequence[str]) -> Axis:
    return Axis(var, kind, "[" + ", ".join(names) + "]", tuple(names))


def text_axis(var: str, values: Sequence[str]) -> Axis:
    return Axis(var, "string", "[" + ", ".join(text_lit(v) for v in values) + "]", tuple(values))


class DayCoverage:
    """The days a corpus puts through `day_row`, kept as ranges so 7.3 million days stay small."""

    def __init__(self) -> None:
        self.ranges: list[tuple[int, int]] = []
        self.singles: set[int] = set()

    def add_range(self, low: int, high: int) -> None:
        self.ranges.append((low, high))

    def update(self, days) -> None:
        self.singles.update(days)

    def merged(self) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for low, high in sorted(self.ranges):
            if merged and low <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], high))
            else:
                merged.append((low, high))
        return merged

    def __contains__(self, day: int) -> bool:
        return day in self.singles or any(low <= day < high for low, high in self.ranges)

    def __len__(self) -> int:
        merged = self.merged()
        inside = sum(1 for day in self.singles if any(low <= day < high for low, high in merged))
        return sum(high - low for low, high in merged) + len(self.singles) - inside

    def covers(self, low: int, high: int) -> bool:
        return any(a <= low and high <= b for a, b in self.merged())


@dataclass
class Corpus:
    values: list[Value] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    bulk: list[Value] = field(default_factory=list)
    days: DayCoverage = field(default_factory=DayCoverage)
    columns: list[Value] = field(default_factory=list)

    def value(self, label: str, expr: str, expected: object, float_bits: bool = False) -> None:
        self.values.append(Value(f"v{len(self.values):05d}_{label}", expr, expected, float_bits))

    def fails(self, label: str, expr: str, error: DatetimeError) -> None:
        self.failures.append(Failure(f"f{len(self.failures):04d}_{label}", expr, error.function, error.kind))

    def outcome(self, label: str, expr: str, compute) -> None:
        """Record a trapping call as a value or a failure, whichever the reference says."""
        try:
            result = compute()
        except DatetimeError as error:
            self.fails(label, expr, error)
            return
        self.value(label, expr, result)

    def grid(self, label: str, axes: Sequence[Axis], row_src: str, row_py: Callable[..., list],
             lanes: tuple[str, ...] = BOTH, strings: bool = False) -> None:
        """One binding looping over the product of `axes` in Chelis, mirrored row by row."""
        expr = row_src
        for depth, axis in enumerate(reversed(axes)):
            combinator = "map" if strings and depth == 0 else "flat_map"
            expr = f"{combinator}(fn ({axis.var}: {axis.type}) -> {expr}, {axis.source})"
        frozen_axes = tuple(axes)

        def expected() -> list:
            rows: list = []
            for combo in itertools.product(*(axis.values for axis in frozen_axes)):
                rows.extend([row_py(*combo)] if strings else row_py(*combo))
            return rows

        name = f"b{len(self.bulk):03d}_{re.sub(r'[^A-Za-z0-9_]', '_', label)}"
        self.bulk.append(Value(name, expr, expected, lanes=lanes))


# ---------------------------------------------------------------------------
# Corpus: days and periods.


def day_row(n: int) -> list[int]:
    year, month, day = ref.civil_from_days(n)
    weekday = ref.weekday_of(n)
    iso_year, iso_week = ref.iso_week(n)
    return [
        year, month, day, ref.date(year, month, day), weekday, ref.day_of_year(n), iso_year, iso_week,
        ref.date_from_iso_week(iso_year, iso_week, weekday), ref.parse_date(ref.date_to_string(n)),
    ]


def period_row(a: int, b: int) -> list[int]:
    months, days = ref.date_period_until(a, b)
    return [months, days, ref.date_add_period(a, (months, days), "ClampToMonthEnd")]


def day_range(corpus: Corpus, label: str, low: int, high: int, lanes: tuple[str, ...]) -> None:
    corpus.days.add_range(low, high)
    corpus.grid(label, [range_axis("n", low, high)], "day_row(n)", day_row, lanes)


def day_samples(corpus: Corpus) -> None:
    day_range(corpus, "days_first_400", ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY + 400, BOTH)
    day_range(corpus, "days_last_400", ref.MAX_EPOCH_DAY - 399, ref.MAX_EPOCH_DAY + 1, BOTH)
    count = 2000
    corpus.days.update(sample_day(k) for k in range(count))
    corpus.grid("days_sampled", [range_axis("k", 0, count)], "day_row(sample_day(k))", lambda k: day_row(sample_day(k)))
    stride = 73
    corpus.days.update(range(FIRST_1900, LAST_2100 + 1, stride))
    corpus.grid("days_stride_1900_2100", [range_axis("k", 0, (LAST_2100 - FIRST_1900) // stride + 1)],
                f"day_row({lit(FIRST_1900)} + k * {lit(stride)})", lambda k: day_row(FIRST_1900 + k * stride))
    # Each window runs from 22 December of a sampled year across New Year.
    windows = [(j, t) for j in range(40) for t in range(18)]
    corpus.days.update(ref.days_from_civil(sample_year(j + 50_000), 12, 22) + t for j, t in windows)
    corpus.grid("days_year_ends", [range_axis("j", 0, 40), range_axis("t", 0, 18)],
                "day_row(date_epoch_day(date(sample_year(j + 50000i64), 12i64, 22i64)) + t)",
                lambda j, t: day_row(ref.days_from_civil(sample_year(j + 50_000), 12, 22) + t))
    corpus.grid("text_sampled", [range_axis("k", 0, count)], "date_to_string(date_from_epoch_day(sample_day(k + 60000i64)))",
                lambda k: ref.date_to_string(sample_day(k + 60_000)), strings=True)
    day_range(corpus, "days_1900_2100", FIRST_1900, LAST_2100 + 1, C_ONLY)


def period_pairs(corpus: Corpus, label: str, first: int, count: int, lanes: tuple[str, ...]) -> None:
    corpus.grid(f"{label}_far", [range_axis("k", first, first + count)],
                "period_row(sample_day(k + 100000i64), sample_day(k + 200000i64))",
                lambda k: period_row(sample_day(k + 100_000), sample_day(k + 200_000)), lanes)
    corpus.grid(f"{label}_near", [range_axis("k", first, first + count)],
                "period_row(near_day(k + 300000i64), near_day(k + 300000i64) + mod(mix(k + 400000i64), 1601i64) - 800i64)",
                lambda k: period_row(near_day(k + 300_000), near_day(k + 300_000) + mix(k + 400_000) % 1601 - 800), lanes)


MONTH_ENDS = [ref.days_from_civil(*ymd) for ymd in (
    (2024, 1, 31), (2024, 2, 29), (2023, 1, 31), (2024, 3, 31), (2024, 5, 31), (2024, 8, 30),
    (2000, 2, 29), (1900, 1, 29), (-1, 12, 31), (0, 2, 29), (-4713, 11, 30), (5000, 7, 31),
)]
EDGE_STARTS = [ref.days_from_civil(*ymd) for ymd in ((ref.MIN_YEAR, 1, 31), (ref.MIN_YEAR, 2, 28), (ref.MAX_YEAR, 12, 31), (ref.MAX_YEAR, 11, 30))]


def periods_and_dates(corpus: Corpus) -> None:
    period_pairs(corpus, "period_until", 0, 1000, BOTH)
    corpus.grid("period_until_month_ends", [int_axis("a", MONTH_ENDS + EDGE_STARTS), int_axis("b", MONTH_ENDS + EDGE_STARTS)],
                "period_row(a, b)", period_row)


# ---------------------------------------------------------------------------
# Corpus: calendar queries and date construction.

QUERY_YEARS = [I64_MIN, I64_MIN + 1, -10_000, -9_999, -400, -100, -4, -1, 0, 1, 1900, 2000, 2023, 2024,
               9_999, 10_000, I64_MAX - 1, I64_MAX]


def calendar_queries(corpus: Corpus) -> None:
    quarter = (ref.MAX_YEAR - ref.MIN_YEAR + 1) // 4 + 1
    for low in range(ref.MIN_YEAR, ref.MAX_YEAR + 1, quarter):
        high = min(low + quarter, ref.MAX_YEAR + 1)
        corpus.grid(f"every_year_from_{low - ref.MIN_YEAR}", [range_axis("y", low, high)], "year_row(y)",
                    lambda y: [ref.days_in_year(y), int(ref.is_leap_year(y)), ref.easter_sunday_gregorian(y), ref.easter_sunday_orthodox(y)])
    corpus.grid("leap_extremes", [int_axis("y", QUERY_YEARS)], "[days_in_year(y), b2i(is_leap_year(y))]",
                lambda y: [ref.days_in_year(y), int(ref.is_leap_year(y))])
    corpus.grid("days_in_month", [int_axis("y", QUERY_YEARS), range_axis("m", 1, 13)], "[days_in_month(y, m)]",
                lambda y, m: [ref.days_in_month(y, m)])
    for month in (0, 13, I64_MIN):
        corpus.fails("days_in_month", f"days_in_month(2024i64, {lit(month)})", expect_error(lambda: ref.days_in_month(2024, month)))
    corpus.grid("weekday_numbers", [range_axis("k", 1, 8)], "[weekday_iso_number(weekday_from_iso_number(k))]", lambda k: [k])
    numbers = [0, 1, 7, 8, -1, I64_MIN, I64_MAX]
    corpus.grid("try_weekday", [int_axis("k", numbers)], "row_opt_weekday(try_weekday_from_iso_number(k))",
                lambda k: opt_row(lambda: ref.weekday_from_iso_number(k), 1))
    for number in (0, 8, I64_MIN):
        corpus.fails("weekday_from", f"weekday_iso_number(weekday_from_iso_number({lit(number)}))",
                     expect_error(lambda: ref.weekday_from_iso_number(number)))
    for iso in range(1, 8):
        corpus.value("weekday_name", f"weekday_name({weekday_src(iso)})", ref.weekday_name(iso))
        corpus.value("weekday_number", f"weekday_iso_number({weekday_src(iso)})", iso)


DATE_FIELDS = [
    (2024, 2, 29), (2023, 2, 29), (2024, 2, 30), (2024, 4, 31), (2024, 13, 1), (2024, 0, 1),
    (2024, 1, 0), (2024, 1, 32), (0, 2, 29), (-1, 2, 29), (-4, 2, 29), (-100, 2, 29), (-400, 2, 29),
    (ref.MIN_YEAR, 1, 1), (ref.MAX_YEAR, 12, 31), (ref.MIN_YEAR - 1, 12, 31), (ref.MAX_YEAR + 1, 1, 1),
    (I64_MIN, 1, 1), (I64_MAX, 1, 1), (2024, I64_MIN, 1), (2024, 1, I64_MAX), (2024, 1, -1),
]
EPOCH_CANDIDATES = [ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, ref.MIN_EPOCH_DAY - 1, ref.MAX_EPOCH_DAY + 1, 0, I64_MIN, I64_MAX]
ISO_CASES = [(2020, 53, 4), (2021, 53, 1), (2021, 52, 7), (2026, 0, 1), (2026, 54, 1), (9_999, 52, 5), (9_999, 52, 6),
             (-9_999, 1, 1), (-10_000, 52, 7), (10_000, 1, 1), (-10_001, 1, 1), (10_001, 1, 1), (I64_MIN, 1, 1),
             (I64_MAX, 1, 1), (2026, I64_MAX, 1), (2026, I64_MIN, 1)]


def date_construction(corpus: Corpus) -> None:
    corpus.grid("try_date", [tuple_axis("x", DATE_FIELDS)], "row_opt_date(try_date(p3a(x), p3b(x), p3c(x)))",
                lambda x: opt_row(lambda: ref.date(*x), 1))
    valid = [x for x in DATE_FIELDS if succeeds(lambda: ref.date(*x))]
    corpus.grid("date", [tuple_axis("x", valid)], "[date_epoch_day(date(p3a(x), p3b(x), p3c(x)))]", lambda x: [ref.date(*x)])
    for x in DATE_FIELDS:
        if x not in valid:
            corpus.fails("date", f"date_epoch_day(date({lit(x[0])}, {lit(x[1])}, {lit(x[2])}))", expect_error(lambda: ref.date(*x)))
    corpus.grid("try_from_epoch", [int_axis("n", EPOCH_CANDIDATES)], "row_opt_date(try_date_from_epoch_day(n))",
                lambda n: opt_row(lambda: ref.date_from_epoch_day(n), 1))
    for n in EPOCH_CANDIDATES:
        corpus.outcome("from_epoch", f"date_epoch_day(date_from_epoch_day({lit(n)}))", lambda: ref.date_from_epoch_day(n))
    corpus.grid("try_from_iso", [tuple_axis("x", ISO_CASES)],
                "row_opt_date(try_date_from_iso_week(p3a(x), p3b(x), weekday_from_iso_number(p3c(x))))",
                lambda x: opt_row(lambda: ref.date_from_iso_week(*x), 1))
    for x in ISO_CASES:
        corpus.outcome("from_iso", f"date_epoch_day(date_from_iso_week({lit(x[0])}, {lit(x[1])}, {weekday_src(x[2])}))",
                       lambda: ref.date_from_iso_week(*x))


# ---------------------------------------------------------------------------
# Corpus: date arithmetic and holiday helpers.

EDGE_DAYS = [ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, 0, -1]
DAY_STEPS = [0, 1, -1, 7, -7, ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY - ref.MAX_EPOCH_DAY,
             I64_MAX, I64_MIN, I64_MAX - 1, I64_MIN + 1]
SMALL_PERIODS = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (13, 45), (-13, -45), (12, 0), (0, -366), (1, 30),
                 (-1, -30), (240, 0), (-1200, 0)]
EXTREME_PERIODS = [(I64_MAX, 0), (I64_MIN, 0), (0, I64_MAX), (0, I64_MIN), (240_000, 0), (-240_000, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]


def date_arithmetic(corpus: Corpus) -> None:
    for start in EDGE_DAYS:
        for step in DAY_STEPS:
            corpus.outcome("add_days", f"date_epoch_day(date_add_days({date_src(start)}, {lit(step)}))",
                           lambda: ref.date_add_days(start, step))
    corpus.grid("days_until_compare", [int_axis("a", EDGE_DAYS + MONTH_ENDS), int_axis("b", EDGE_DAYS + MONTH_ENDS)],
                "[date_days_until(date_from_epoch_day(a), date_from_epoch_day(b)), b2i(date_lt(date_from_epoch_day(a), date_from_epoch_day(b))), "
                "b2i(date_lte(date_from_epoch_day(a), date_from_epoch_day(b))), b2i(date_gt(date_from_epoch_day(a), date_from_epoch_day(b))), "
                "b2i(date_gte(date_from_epoch_day(a), date_from_epoch_day(b)))]",
                lambda a, b: [b - a, int(a < b), int(a <= b), int(a > b), int(a >= b)])

    def months_row(s: int, n: int) -> list[int]:
        return (opt_row(lambda: ref.date_add_months(s, n, "ClampToMonthEnd"), 1)
                + opt_row(lambda: ref.date_add_months(s, n, "RejectInvalidDay"), 1)
                + [ref.date_add_months(s, n, "ClampToMonthEnd")])

    months_src = ("concat(concat(row_opt_date(try_date_add_months(date_from_epoch_day(s), n, ClampToMonthEnd)), "
                  "row_opt_date(try_date_add_months(date_from_epoch_day(s), n, RejectInvalidDay))), "
                  "[date_epoch_day(date_add_months(date_from_epoch_day(s), n, ClampToMonthEnd))])")
    corpus.grid("add_months", [int_axis("s", MONTH_ENDS), range_axis("n", -25, 26)], months_src, months_row)
    corpus.grid("add_months_far", [int_axis("s", MONTH_ENDS), int_axis("n", [1200, -1200, 12 * 4000, -12 * 4000, 12 * 3000 + 7])],
                months_src, months_row)
    corpus.grid("add_months_low_edge", [int_axis("s", EDGE_STARTS[:2]), int_axis("n", [0, 1, 11, 12, 25, 12 * 19_998 + 10])],
                months_src, months_row)
    corpus.grid("add_months_high_edge", [int_axis("s", EDGE_STARTS[2:]), int_axis("n", [0, -1, -11, -12, -25, -12 * 19_998 - 10])],
                months_src, months_row)
    for start, months in ((EDGE_STARTS[2], 1), (EDGE_STARTS[0], -1), (0, I64_MAX), (0, I64_MIN), (EDGE_STARTS[3], 12 * 19_999),
                          (MONTH_ENDS[0], 1)):
        for policy in ref.DAY_OVERFLOW:
            args = f"{date_src(start)}, {lit(months)}, {policy}"
            corpus.outcome("months_trap", f"date_epoch_day(date_add_months({args}))", lambda: ref.date_add_months(start, months, policy))
            try:
                try_value(lambda: ref.date_add_months(start, months, policy))
            except DatetimeError as error:
                corpus.fails("months_try_overflow", f"row_opt_date(try_date_add_months({args}))",
                             DatetimeError("try_date_add_months", error.kind, error.detail))

    def period_row_py(s: int, p: tuple[int, int]) -> list[int]:
        return (opt_row(lambda: ref.date_add_period(s, p, "ClampToMonthEnd"), 1)
                + opt_row(lambda: ref.date_add_period(s, p, "RejectInvalidDay"), 1)
                + [ref.date_add_period(s, p, "ClampToMonthEnd")])

    corpus.grid("add_period", [int_axis("s", MONTH_ENDS), tuple_axis("p", SMALL_PERIODS)],
                "concat(concat(row_opt_date(try_date_add_period(date_from_epoch_day(s), per2(p), ClampToMonthEnd)), "
                "row_opt_date(try_date_add_period(date_from_epoch_day(s), per2(p), RejectInvalidDay))), "
                "[date_epoch_day(date_add_period(date_from_epoch_day(s), per2(p), ClampToMonthEnd))])", period_row_py)
    for start in (EDGE_STARTS[2], EDGE_STARTS[0], MONTH_ENDS[0]):
        for p in EXTREME_PERIODS:
            for policy in ref.DAY_OVERFLOW:
                args = f"{date_src(start)}, {period_src(p)}, {policy}"
                corpus.outcome("period_trap", f"date_epoch_day(date_add_period({args}))", lambda: ref.date_add_period(start, p, policy))
                try:
                    corpus.value("period_try", f"row_opt_date(try_date_add_period({args}))",
                                 opt_row(lambda: ref.date_add_period(start, p, policy), 1))
                except DatetimeError as error:
                    corpus.fails("period_try_overflow", f"row_opt_date(try_date_add_period({args}))",
                                 DatetimeError("try_date_add_period", error.kind, error.detail))


HOLIDAY_YEARS = [2024, 2023, 2000, 1900, 0, -1, ref.MIN_YEAR, ref.MAX_YEAR, 1583, -4713]
HOLIDAY_MONTHS = [1, 2, 5, 11, 12]


def holiday_helpers(corpus: Corpus) -> None:
    corpus.grid("nth_weekday", [int_axis("y", HOLIDAY_YEARS), int_axis("m", HOLIDAY_MONTHS), range_axis("w", 1, 8), range_axis("n", 1, 7)],
                "row_opt_date(nth_weekday_in_month(y, m, weekday_from_iso_number(w), n))",
                lambda y, m, w, n: opt_row(lambda: ref.nth_weekday_in_month(y, m, w, n), 1))
    corpus.grid("nth_weekday_huge", [int_axis("y", HOLIDAY_YEARS[:3]), range_axis("w", 1, 8)],
                f"row_opt_date(nth_weekday_in_month(y, 2i64, weekday_from_iso_number(w), {lit(I64_MAX)}))",
                lambda y, w: opt_row(lambda: ref.nth_weekday_in_month(y, 2, w, I64_MAX), 1))
    corpus.grid("last_weekday", [int_axis("y", HOLIDAY_YEARS), int_axis("m", HOLIDAY_MONTHS), range_axis("w", 1, 8)],
                "[date_epoch_day(last_weekday_in_month(y, m, weekday_from_iso_number(w)))]",
                lambda y, m, w: [ref.last_weekday_in_month(y, m, w)])
    for year, month, n in ((2024, 11, 0), (2024, 11, -1), (2024, 11, I64_MIN), (2024, 0, 1), (2024, 13, 1),
                           (ref.MIN_YEAR - 1, 1, 1), (ref.MAX_YEAR + 1, 1, 1), (I64_MAX, 1, 1)):
        corpus.fails("nth_weekday", f"row_opt_date(nth_weekday_in_month({lit(year)}, {lit(month)}, Thursday, {lit(n)}))",
                     expect_error(lambda: ref.nth_weekday_in_month(year, month, 4, n)))
    for year, month in ((2024, 0), (2024, 13), (ref.MIN_YEAR - 1, 12), (ref.MAX_YEAR + 1, 1), (I64_MIN, 1)):
        corpus.fails("last_weekday", f"date_epoch_day(last_weekday_in_month({lit(year)}, {lit(month)}, Monday))",
                     expect_error(lambda: ref.last_weekday_in_month(year, month, 1)))
    corpus.grid("weekday_rolls", [range_axis("d", 19_000, 19_014), range_axis("w", 1, 8)],
                "[date_epoch_day(weekday_on_or_after(date_from_epoch_day(d), weekday_from_iso_number(w))), "
                "date_epoch_day(weekday_on_or_before(date_from_epoch_day(d), weekday_from_iso_number(w)))]",
                lambda d, w: [ref.weekday_on_or_after(d, w), ref.weekday_on_or_before(d, w)])
    corpus.grid("weekday_rolls_inward", [range_axis("k", 0, 7), range_axis("w", 1, 8)],
                f"[date_epoch_day(weekday_on_or_after(date_from_epoch_day({lit(ref.MIN_EPOCH_DAY)} + k), weekday_from_iso_number(w))), "
                f"date_epoch_day(weekday_on_or_before(date_from_epoch_day({lit(ref.MAX_EPOCH_DAY)} - k), weekday_from_iso_number(w)))]",
                lambda k, w: [ref.weekday_on_or_after(ref.MIN_EPOCH_DAY + k, w), ref.weekday_on_or_before(ref.MAX_EPOCH_DAY - k, w)])
    for k in range(7):
        for w in range(1, 8):
            day = ref.MAX_EPOCH_DAY - k
            corpus.outcome("on_or_after_edge", f"date_epoch_day(weekday_on_or_after({date_src(day)}, {weekday_src(w)}))",
                           lambda: ref.weekday_on_or_after(day, w))
            day = ref.MIN_EPOCH_DAY + k
            corpus.outcome("on_or_before_edge", f"date_epoch_day(weekday_on_or_before({date_src(day)}, {weekday_src(w)}))",
                           lambda: ref.weekday_on_or_before(day, w))
    for year in (ref.MIN_YEAR - 1, ref.MAX_YEAR + 1, I64_MIN, I64_MAX):
        corpus.fails("easter_g", f"date_epoch_day(easter_sunday_gregorian({lit(year)}))", expect_error(lambda: ref.easter_sunday_gregorian(year)))
        corpus.fails("easter_o", f"date_epoch_day(easter_sunday_orthodox({lit(year)}))", expect_error(lambda: ref.easter_sunday_orthodox(year)))


# ---------------------------------------------------------------------------
# Corpus: time, datetime, offsets and instants.

EXTREME_DURATIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, NANO - 1), (86_399, NANO - 1), (86_400, 0),
                     (-86_400, 0), (I64_MAX, NANO - 1), (I64_MIN, 0), (I64_MIN, 1), (I64_MAX, 0),
                     (3_000_000_000, 5), (-3_000_000_000, 5)]
SAFE_DURATIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, NANO - 1), (86_399, NANO - 1), (-86_400, 0),
                  (3_000_000_000, 5), (-3_000_000_000, 5), (2**61, 0), (-(2**61), 999)]
TIMES = [0, DAY_NANOS - 1, 45_296_789_000_001, 43_200 * NANO]
DTS = [(ref.MIN_EPOCH_DAY, 0), (ref.MAX_EPOCH_DAY, DAY_NANOS - 1), (0, 0), (-1, DAY_NANOS - 1), (19_631, 34_200 * NANO)]
INTERIOR_DTS = [(0, 0), (-1, DAY_NANOS - 1), (19_631, 34_200 * NANO)]
TIME_FIELDS = [(0, 0, 0, 0), (23, 59, 59, NANO - 1), (12, 30, 15, 500), (24, 0, 0, 0), (23, 60, 0, 0),
               (23, 59, 60, 0), (0, 0, 0, NANO), (0, 0, 0, -1), (-1, 0, 0, 0), (I64_MAX, 0, 0, 0), (0, I64_MIN, 0, 0)]
NODS = [0, DAY_NANOS - 1, DAY_NANOS, -1, 45_296_789_000_001, I64_MAX, I64_MIN]


def time_and_datetime(corpus: Corpus) -> None:
    corpus.grid("try_time", [tuple_axis("x", TIME_FIELDS)], "row_opt_time(try_time(p4a(x), p4b(x), p4c(x), p4d(x)))",
                lambda x: opt_row(lambda: ref.time(*x), 1))
    for x in TIME_FIELDS:
        corpus.outcome("time", f"time_nanosecond_of_day(time({', '.join(lit(v) for v in x)}))", lambda: ref.time(*x))
    corpus.grid("try_time_nod", [int_axis("n", NODS)], "row_opt_time(try_time_from_nanosecond_of_day(n))",
                lambda n: opt_row(lambda: ref.time_from_nanosecond_of_day(n), 1))
    for n in NODS:
        corpus.outcome("time_nod", f"time_nanosecond_of_day(time_from_nanosecond_of_day({lit(n)}))", lambda: ref.time_from_nanosecond_of_day(n))
    corpus.grid("time_fields", [int_axis("t", TIMES)],
                "[time_hour(time_from_nanosecond_of_day(t)), time_minute(time_from_nanosecond_of_day(t)), "
                "time_second(time_from_nanosecond_of_day(t)), time_nanosecond(time_from_nanosecond_of_day(t))]",
                lambda t: list(ref.time_fields(t)))
    corpus.grid("time_add", [int_axis("t", TIMES), tuple_axis("d", EXTREME_DURATIONS)],
                "row_carry(time_add_duration(time_from_nanosecond_of_day(t), dur2(d)))",
                lambda t, d: list(ref.time_add_duration(t, d)))
    corpus.grid("time_pairs", [int_axis("a", TIMES), int_axis("b", TIMES)],
                "concat(row_duration(time_until(time_from_nanosecond_of_day(a), time_from_nanosecond_of_day(b))), "
                "[b2i(time_lt(time_from_nanosecond_of_day(a), time_from_nanosecond_of_day(b))), b2i(time_lte(time_from_nanosecond_of_day(a), time_from_nanosecond_of_day(b))), "
                "b2i(time_gt(time_from_nanosecond_of_day(a), time_from_nanosecond_of_day(b))), b2i(time_gte(time_from_nanosecond_of_day(a), time_from_nanosecond_of_day(b)))])",
                lambda a, b: [*ref.time_until(a, b), int(a < b), int(a <= b), int(a > b), int(a >= b)])
    corpus.grid("datetime_parts", [tuple_axis("x", DTS)], "row_dt(dt2(x))", lambda x: list(x))
    corpus.grid("datetime_add", [tuple_axis("x", INTERIOR_DTS), tuple_axis("d", SAFE_DURATIONS[:9])],
                "row_dt(datetime_add_duration(dt2(x), dur2(d)))", lambda x, d: list(ref.datetime_add_duration(x, d)))
    for x in DTS:
        for d in EXTREME_DURATIONS:
            corpus.outcome("datetime_add_edge", f"row_dt(datetime_add_duration({dt_src(x)}, {duration_src(d)}))",
                           lambda: list(ref.datetime_add_duration(x, d)))

    def dt_period_row(x: tuple[int, int], p: tuple[int, int]) -> list[int]:
        return (opt_row(lambda: ref.datetime_add_period(x, p, "ClampToMonthEnd"), 2)
                + opt_row(lambda: ref.datetime_add_period(x, p, "RejectInvalidDay"), 2)
                + list(ref.datetime_add_period(x, p, "ClampToMonthEnd")))

    jan31 = (ref.days_from_civil(2023, 1, 31), 3_600 * NANO)
    corpus.grid("datetime_period", [tuple_axis("x", INTERIOR_DTS + [jan31]), tuple_axis("p", SMALL_PERIODS)],
                "concat(concat(row_opt_dt(try_datetime_add_period(dt2(x), per2(p), ClampToMonthEnd)), "
                "row_opt_dt(try_datetime_add_period(dt2(x), per2(p), RejectInvalidDay))), "
                "row_dt(datetime_add_period(dt2(x), per2(p), ClampToMonthEnd)))", dt_period_row)
    corpus.fails("dt_period_reject", f"row_dt(datetime_add_period({dt_src(jan31)}, {period_src((1, 0))}, RejectInvalidDay))",
                 expect_error(lambda: ref.datetime_add_period(jan31, (1, 0), "RejectInvalidDay")))
    for x in DTS[:2]:
        for p in ((1, 0), (-1, 0), (0, 1), (0, -1), (I64_MAX, 0)):
            for policy in ref.DAY_OVERFLOW:
                args = f"{dt_src(x)}, {period_src(p)}, {policy}"
                corpus.outcome("dt_period_edge", f"row_dt(datetime_add_period({args}))", lambda: list(ref.datetime_add_period(x, p, policy)))
                try:
                    corpus.value("dt_period_try", f"row_opt_dt(try_datetime_add_period({args}))",
                                 opt_row(lambda: ref.datetime_add_period(x, p, policy), 2))
                except DatetimeError as error:
                    corpus.fails("dt_period_try_overflow", f"row_opt_dt(try_datetime_add_period({args}))",
                                 DatetimeError("try_datetime_add_period", error.kind, error.detail))
    corpus.grid("datetime_pairs", [tuple_axis("a", DTS), tuple_axis("b", DTS)],
                "concat(row_duration(datetime_until(dt2(a), dt2(b))), [b2i(datetime_lt(dt2(a), dt2(b))), b2i(datetime_lte(dt2(a), dt2(b))), "
                "b2i(datetime_gt(dt2(a), dt2(b))), b2i(datetime_gte(dt2(a), dt2(b)))])",
                lambda a, b: [*ref.datetime_until(a, b), *(int(f(ref.civil_nanos(a), ref.civil_nanos(b))) for f in COMPARATORS)])


COMPARATORS = (lambda a, b: a < b, lambda a, b: a <= b, lambda a, b: a > b, lambda a, b: a >= b)
OFFSET_CANDIDATES = [0, 1, -1, 86_399, -86_399, 86_400, -86_400, 19_800, -16_200, I64_MAX, I64_MIN]
OFFSETS = [0, 86_399, -86_399, 3_600, -1]
INSTANTS = [(ref.INSTANT_MIN_SECOND, 0), (ref.INSTANT_MAX_SECOND, NANO - 1), (0, 0), (-1, NANO - 1),
            (9_223_372_036, 854_775_807), (9_223_372_036, 854_775_808), (-9_223_372_037, 145_224_192),
            (-9_223_372_037, 145_224_191), (1_727_789_400, 123_456_789), (-1_000_000_000, 500_000_000)]
INTERIOR_INSTANTS = [(0, 0), (-1, NANO - 1), (1_727_789_400, 123_456_789), (-1_000_000_000, 500_000_000), (86_399, 999_999_999), (-86_401, 1),
                     (2, 500_000_000), (-3, 500_000_000), (7_200, 0)]
INSTANT_CANDIDATES = [(ref.INSTANT_MIN_SECOND - 1, 0), (ref.INSTANT_MAX_SECOND + 1, 0), (0, NANO), (0, -1), (I64_MIN, 0), (I64_MAX, 0)] + INSTANTS
INCREMENTS = [(0, 1), (1, 0), (0, 500_000_000), (1, 500_000_000), (3_600, 0), (86_400, 0), (0, 3_125)]
BAD_INCREMENTS = [(7, 0), (0, 0), (-1, 0), (172_800, 0), (0, 7), (I64_MAX, 0)]
UNITS = tuple(ref.TIME_UNIT_NANOS)


def offsets_and_instants(corpus: Corpus) -> None:
    corpus.grid("try_offset", [int_axis("s", OFFSET_CANDIDATES)], "row_opt_offset(try_offset_from_seconds(s))",
                lambda s: opt_row(lambda: ref.offset_from_seconds(s), 1))
    for s in OFFSET_CANDIDATES:
        corpus.outcome("offset", f"offset_seconds(offset_from_seconds({lit(s)}))", lambda: ref.offset_from_seconds(s))
    corpus.grid("try_instant", [tuple_axis("x", INSTANT_CANDIDATES)], "row_opt_instant(try_instant_from_unix(p2a(x), p2b(x)))",
                lambda x: opt_row(lambda: ref.instant_from_unix(*x), 2))
    for x in INSTANT_CANDIDATES:
        corpus.outcome("instant", f"row_instant(instant_from_unix({lit(x[0])}, {lit(x[1])}))", lambda: list(ref.instant_from_unix(*x)))
    for unit, per in ref.TIME_UNIT_NANOS.items():
        high = ((ref.INSTANT_MAX_SECOND + 1) * NANO - 1) // per
        low = -((-ref.INSTANT_MIN_SECOND * NANO) // per)
        counts = [c for c in (0, 1, -1, high, high + 1, low, low - 1, I64_MAX, I64_MIN) if ref.fits_i64(c)]
        corpus.grid(f"try_from_count_{unit}", [int_axis("c", counts)], f"row_opt_instant(try_instant_from_unix_count(c, {unit}))",
                    lambda c, unit=unit: opt_row(lambda: ref.instant_from_unix_count(c, unit), 2))
        for c in counts:
            corpus.outcome("from_count", f"row_instant(instant_from_unix_count({lit(c)}, {unit}))",
                           lambda: list(ref.instant_from_unix_count(c, unit)))
    in_range = [i for i in INSTANTS if ref.in_instant_range(i[0])]
    corpus.grid("try_to_count", [tuple_axis("i", in_range), ctor_axis("u", "TimeUnit", UNITS), ctor_axis("r", "Rounding", ref.ROUNDING)],
                "row_opt_i64(try_instant_to_unix_count(inst2(i), u, r))",
                lambda i, u, r: (lambda v: [0, 0] if v is None else [1, v])(ref.try_instant_to_unix_count(i, u, r)))
    corpus.grid("to_count", [tuple_axis("i", INTERIOR_INSTANTS), ctor_axis("u", "TimeUnit", UNITS), ctor_axis("r", "Rounding", ref.ROUNDING_TOTAL)],
                "[instant_to_unix_count(inst2(i), u, r)]", lambda i, u, r: [ref.instant_to_unix_count(i, u, r)])
    for i in in_range:
        for rounding in ref.ROUNDING:
            corpus.outcome("to_count_nanos", f"instant_to_unix_count({instant_src(i)}, Nanoseconds, {rounding})",
                           lambda: ref.instant_to_unix_count(i, "Nanoseconds", rounding))
    for i in INTERIOR_INSTANTS:
        for unit in UNITS:
            corpus.outcome("to_count_exact", f"instant_to_unix_count({instant_src(i)}, {unit}, RejectInexact)",
                           lambda: ref.instant_to_unix_count(i, unit, "RejectInexact"))
        for inc in INCREMENTS:
            corpus.outcome("round_to_exact", f"row_instant(instant_round_to({instant_src(i)}, {duration_src(inc)}, RejectInexact))",
                           lambda: list(ref.instant_round_to(i, inc, "RejectInexact")))
    corpus.grid("instant_add", [tuple_axis("i", INTERIOR_INSTANTS), tuple_axis("d", SAFE_DURATIONS[:9])],
                "row_instant(instant_add_duration(inst2(i), dur2(d)))", lambda i, d: list(ref.instant_add_duration(i, d)))
    for i in in_range:
        for d in EXTREME_DURATIONS:
            corpus.outcome("instant_add_edge", f"row_instant(instant_add_duration({instant_src(i)}, {duration_src(d)}))",
                           lambda: list(ref.instant_add_duration(i, d)))
    corpus.grid("instant_pairs", [tuple_axis("a", in_range), tuple_axis("b", in_range)],
                "concat(row_duration(instant_until(inst2(a), inst2(b))), [b2i(instant_lt(inst2(a), inst2(b))), b2i(instant_lte(inst2(a), inst2(b))), "
                "b2i(instant_gt(inst2(a), inst2(b))), b2i(instant_gte(inst2(a), inst2(b)))])",
                lambda a, b: [*ref.instant_until(a, b), *(int(f(ref.nanos_of(a), ref.nanos_of(b))) for f in COMPARATORS)])
    corpus.grid("round_to", [tuple_axis("i", INTERIOR_INSTANTS), tuple_axis("inc", INCREMENTS), ctor_axis("r", "Rounding", ref.ROUNDING_TOTAL)],
                "row_instant(instant_round_to(inst2(i), dur2(inc), r))", lambda i, inc, r: list(ref.instant_round_to(i, inc, r)))
    for i in [(0, 0), (ref.INSTANT_MAX_SECOND, 1), (ref.INSTANT_MIN_SECOND, 1)]:
        for inc in BAD_INCREMENTS + [(86_400, 0), (0, 1)]:
            for rounding in ref.ROUNDING:
                corpus.outcome("round_to_edge", f"row_instant(instant_round_to({instant_src(i)}, {duration_src(inc)}, {rounding}))",
                               lambda: list(ref.instant_round_to(i, inc, rounding)))
    corpus.grid("to_datetime_at", [tuple_axis("i", in_range), int_axis("o", OFFSETS)],
                "concat(concat(row_dt(instant_to_datetime_at(inst2(i), offset_from_seconds(o))), "
                "row_odt(offset_datetime(inst2(i), offset_from_seconds(o)))), "
                "row_dt(offset_datetime_local(offset_datetime(inst2(i), offset_from_seconds(o)))))",
                lambda i, o: [*ref.instant_to_datetime_at(i, o), *i, o, *ref.instant_to_datetime_at(i, o)])
    for x in DTS + [(ref.MIN_EPOCH_DAY, 86_399 * NANO), (ref.MAX_EPOCH_DAY, 0)]:
        for o in (0, 86_399, -86_399, 1, -1):
            corpus.outcome("to_instant_at", f"row_instant(datetime_to_instant_at({dt_src(x)}, {offset_src(o)}))",
                           lambda: list(ref.datetime_to_instant_at(x, o)))


# ---------------------------------------------------------------------------
# Corpus: durations and periods.

RAW_DURATIONS = [(0, 0), (1, -1), (-1, 1), (0, -1_500_000_000), (0, I64_MAX), (0, I64_MIN), (I64_MAX, NANO - 1),
                 (I64_MAX, NANO), (I64_MIN, 0), (I64_MIN, -1), (I64_MIN, I64_MAX), (I64_MAX, I64_MIN), (5, 3 * NANO)]
COUNT_DURATIONS = EXTREME_DURATIONS + [(-2, 500_000_000), (2, 500_000_000), (-3, 500_000_000), (0, 1_500_000)]
F64_DURATIONS = EXTREME_DURATIONS + [(1_727_789_400, 123_456_789), (2**53, 1), (2**53 - 1, NANO - 1), (-(2**53), 999_999_999),
                                     (-1, 999_999_000), (-1, 999_000_000), (-2, 1)]
MULTIPLIERS = [0, 1, -1, 2, -2, 1_000_000_007, I64_MAX, I64_MIN]
PERIOD_CANDIDATES = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1), (I64_MIN, 0), (0, I64_MIN),
                     (I64_MAX, I64_MAX), (I64_MIN, I64_MIN), (I64_MIN, 1), (I64_MAX, -1)]


def durations_and_periods(corpus: Corpus) -> None:
    corpus.grid("duration", [tuple_axis("x", [x for x in RAW_DURATIONS if succeeds(lambda: ref.duration(*x))])],
                "row_duration(duration(p2a(x), p2b(x)))", lambda x: list(ref.duration(*x)))
    for x in RAW_DURATIONS:
        if not succeeds(lambda: ref.duration(*x)):
            corpus.fails("duration", f"row_duration(duration({lit(x[0])}, {lit(x[1])}))", expect_error(lambda: ref.duration(*x)))
    counts = [0, 1, -1, I64_MAX, I64_MIN, 2_562_047_788_015_215, 2_562_047_788_015_216]
    for unit in UNITS:
        ok = [c for c in counts if succeeds(lambda: ref.duration_from_count(c, unit))]
        corpus.grid(f"from_count_{unit}", [int_axis("c", ok)], f"row_duration(duration_from_count(c, {unit}))",
                    lambda c, unit=unit: list(ref.duration_from_count(c, unit)))
        for c in counts:
            if c not in ok:
                corpus.fails("duration_from_count", f"row_duration(duration_from_count({lit(c)}, {unit}))",
                             expect_error(lambda: ref.duration_from_count(c, unit)))
    corpus.grid("try_duration_to_count", [tuple_axis("d", COUNT_DURATIONS), ctor_axis("u", "TimeUnit", UNITS), ctor_axis("r", "Rounding", ref.ROUNDING)],
                "row_opt_i64(try_duration_to_count(dur2(d), u, r))",
                lambda d, u, r: (lambda v: [0, 0] if v is None else [1, v])(ref.try_duration_to_count(d, u, r)))
    # Inexact and too large at once: the reference rounds first, so RejectInexact's
    # `domain` comes before the fit check's `overflow`.
    both = (I64_MAX, NANO - 1)
    corpus.fails("named_inexact_and_too_large", f"duration_to_count({duration_src(both)}, Milliseconds, RejectInexact)",
                 expect_error(lambda: ref.duration_to_count(both, "Milliseconds", "RejectInexact")))
    for d in COUNT_DURATIONS:
        for unit in UNITS:
            corpus.outcome("duration_count_exact", f"duration_to_count({duration_src(d)}, {unit}, RejectInexact)",
                           lambda: ref.duration_to_count(d, unit, "RejectInexact"))
        for unit in ("Nanoseconds", "Microseconds"):
            for rounding in ref.ROUNDING:
                corpus.outcome("duration_to_count", f"duration_to_count({duration_src(d)}, {unit}, {rounding})",
                               lambda: ref.duration_to_count(d, unit, rounding))
    for d in F64_DURATIONS:
        corpus.value("seconds_f64", f"duration_to_seconds_f64({duration_src(d)})", ref.duration_to_seconds_f64(d), float_bits=True)
    corpus.grid("duration_negate", [tuple_axis("d", [d for d in F64_DURATIONS if succeeds(lambda: ref.duration_negate(d))])],
                "row_duration(duration_negate(dur2(d)))", lambda d: list(ref.duration_negate(d)))
    corpus.fails("duration_negate", f"row_duration(duration_negate({duration_src((I64_MIN, 0))}))",
                 expect_error(lambda: ref.duration_negate((I64_MIN, 0))))
    pairs = [(a, b) for a in EXTREME_DURATIONS for b in EXTREME_DURATIONS]
    corpus.grid("duration_add", [tuple_axis("x", [(*a, *b) for a, b in pairs if succeeds(lambda: ref.duration_add(a, b))])],
                "row_duration(duration_add(duration(p4a(x), p4b(x)), duration(p4c(x), p4d(x))))",
                lambda x: list(ref.duration_add(x[:2], x[2:])))
    corpus.grid("duration_sub", [tuple_axis("x", [(*a, *b) for a, b in pairs if succeeds(lambda: ref.duration_sub(a, b))])],
                "row_duration(duration_sub(duration(p4a(x), p4b(x)), duration(p4c(x), p4d(x))))",
                lambda x: list(ref.duration_sub(x[:2], x[2:])))
    for a, b in pairs:
        if not succeeds(lambda: ref.duration_add(a, b)):
            corpus.fails("duration_add", f"row_duration(duration_add({duration_src(a)}, {duration_src(b)}))", expect_error(lambda: ref.duration_add(a, b)))
        if not succeeds(lambda: ref.duration_sub(a, b)):
            corpus.fails("duration_sub", f"row_duration(duration_sub({duration_src(a)}, {duration_src(b)}))", expect_error(lambda: ref.duration_sub(a, b)))
    corpus.grid("duration_compare", [tuple_axis("a", EXTREME_DURATIONS), tuple_axis("b", EXTREME_DURATIONS)],
                "[b2i(duration_lt(dur2(a), dur2(b))), b2i(duration_lte(dur2(a), dur2(b))), b2i(duration_gt(dur2(a), dur2(b))), b2i(duration_gte(dur2(a), dur2(b)))]",
                lambda a, b: [int(f(ref.nanos_of(a), ref.nanos_of(b))) for f in COMPARATORS])
    products = [(*d, k) for d in F64_DURATIONS for k in MULTIPLIERS]
    corpus.grid("duration_mul", [tuple_axis("x", [x for x in products if succeeds(lambda: ref.duration_mul(x[:2], x[2]))])],
                "row_duration(duration_mul(duration(p3a(x), p3b(x)), p3c(x)))", lambda x: list(ref.duration_mul(x[:2], x[2])))
    for x in products:
        if not succeeds(lambda: ref.duration_mul(x[:2], x[2])):
            corpus.fails("duration_mul", f"row_duration(duration_mul({duration_src(x[:2])}, {lit(x[2])}))",
                         expect_error(lambda: ref.duration_mul(x[:2], x[2])))

    corpus.grid("try_period", [tuple_axis("x", PERIOD_CANDIDATES)], "row_opt_period(try_period(p2a(x), p2b(x)))",
                lambda x: opt_row(lambda: ref.period(*x), 2))
    valid = [p for p in PERIOD_CANDIDATES if succeeds(lambda: ref.period(*p))]
    corpus.grid("period", [tuple_axis("x", valid)], "row_period(per2(x))", lambda x: list(x))
    for p in PERIOD_CANDIDATES:
        if p not in valid:
            corpus.fails("period", f"row_period(period({lit(p[0])}, {lit(p[1])}))", expect_error(lambda: ref.period(*p)))
    corpus.grid("period_negate", [tuple_axis("x", [p for p in valid if succeeds(lambda: ref.period_negate(p))])],
                "row_period(period_negate(per2(x)))", lambda x: list(ref.period_negate(x)))
    for p in valid:
        if not succeeds(lambda: ref.period_negate(p)):
            corpus.fails("period_negate", f"row_period(period_negate({period_src(p)}))", expect_error(lambda: ref.period_negate(p)))
    scaled = [(*p, k) for p in valid for k in (0, 1, -1, 3, I64_MAX, I64_MIN)]
    corpus.grid("period_mul", [tuple_axis("x", [x for x in scaled if succeeds(lambda: ref.period_mul(x[:2], x[2]))])],
                "row_period(period_mul(period(p3a(x), p3b(x)), p3c(x)))", lambda x: list(ref.period_mul(x[:2], x[2])))
    for x in scaled:
        if not succeeds(lambda: ref.period_mul(x[:2], x[2])):
            corpus.fails("period_mul", f"row_period(period_mul({period_src(x[:2])}, {lit(x[2])}))", expect_error(lambda: ref.period_mul(x[:2], x[2])))


# ---------------------------------------------------------------------------
# Corpus: the §8.8 text profile. Accepted texts must parse to the reference
# value, format to the canonical spelling, and re-parse to the same value;
# rejected texts make the `try_` parser return None and the trapping parser
# fail `domain`.

VALID_TEXT = {
    "date": ["2024-02-29", "0000-01-01", "-000001-12-31", "+002024-01-01", "+000000-06-15", "-009999-01-01",
             "9999-12-31", "1970-01-01", "+009999-12-31", "-000400-02-29", "0001-01-01"],
    "time": ["00:00:00", "23:59:59", "12:34:56.7", "12:34:56.45", "12:34:56.100", "12:34:56.1234",
             "12:34:56.12345", "12:34:56.123456", "12:34:56.1234567", "12:34:56.12345678",
             "12:34:56.123456789", "12:34:56.000000001", "12:34:56.0", "12:34:56.000000000"],
    "datetime": ["2024-02-29T12:00:00", "2024-02-29t12:00:00", "2024-02-29 12:00:00.5",
                 "-009999-01-01T00:00:00", "9999-12-31T23:59:59.999999999", "+000000-01-01T00:00:00.010",
                 "9999-12-31T23:59:59"],
    "offset": ["Z", "z", "+00:00", "-00:00", "+05:30", "-05:30", "+23:59", "-23:59:59", "+00:00:01",
               "+14:00", "-12:00", "+05:30:00", "-00:00:00"],
    "instant": ["1970-01-01T00:00:00Z", "2024-02-29T12:00:00.5+05:30", "2026-10-01T09:30:00-04:00",
                "1969-12-31T23:59:59.999999999z", "-009999-01-01T23:59:59Z", "-009999-01-01T00:00:00-23:59:59",
                "9999-12-31T00:00:00Z", "9999-12-31T23:59:59.999999999+23:59:59", "2024-02-29 12:00:00+00:00"],
    "offset_datetime": ["2026-10-01T09:30:00-04:00", "2026-10-01T13:30:00Z", "2026-10-01T13:30:00+00:00",
                        "2026-10-01T13:30:00-00:00", "-009999-01-01T00:00:00-23:59:59", "9999-12-31T23:59:59+23:59:59",
                        "2024-02-29t23:00:00.25+05:45:30"],
    "duration": ["PT0S", "-PT0S", "PT1H", "PT1M", "PT1S", "PT1H1M1S", "PT3661S", "PT0.5S", "-PT0.5S",
                 "PT1.000000001S", "PT90M", "PT25H", "PT1H30M", "PT00001S", "PT0.100S",
                 "PT9223372036854775807.999999999S", "-PT9223372036854775808S", "-PT2562047788015215H30M8S",
                 "PT2562047788015215H30M7.999999999S", "-PT9223372036854775807.999999999S", "-PT0.000000001S"],
    "period": ["P0D", "-P0D", "P1Y", "P1M", "P1W", "P1D", "P1Y2M3W4D", "P14M3D", "-P14M3D", "P0Y0M0W0D",
               "P9223372036854775807M", "-P9223372036854775808M", "-P768614336404564650Y8M",
               "P1317624576693539401W", "-P1317624576693539401W1D", "P007D"],
}
INVALID_TEXT = {
    "date": ["2024-02-30", "2023-02-29", "2024-13-01", "2024-00-10", "2024-01-00", "2024-1-01", "24-01-01",
             "-000000-01-01", "+010000-01-01", "-010000-01-01", "+12345-01-01", "20240-01-01", "2024/01/01",
             " 2024-01-01", "2024-01-01 ", "", "2024-01-01T00:00:00", "+2024-01-01", "-0001-01-01",
             "２024-01-01", "2024-02-3a"],
    "time": ["24:00:00", "23:60:00", "23:59:60", "12:00", "12:00:00.", "12:00:00.1234567890", "1:00:00",
             "12:00:00Z", "12:00:00,5", "-1:00:00", ""],
    "datetime": ["2024-02-29X12:00:00", "2024-02-29T24:00:00", "2024-02-29", "2024-02-29T12:00:00Z",
                 "2024-02-29  12:00:00", "2024-02-29T12:00:60"],
    "offset": ["+24:00", "-24:00", "+05:60", "+05:30:60", "+0530", "05:30", "+5:30", "UTC", "", "Z ",
               "+05:30:0", "−05:00", "+05:30:00.5"],
    "instant": ["1970-01-01T00:00:00", "1970-01-01T00:00:00+24:00", "1970-01-01T00:00:60Z",
                "-009999-01-01T00:00:00Z", "9999-12-31T23:59:59Z", "-009999-01-01T23:59:58.999999999Z",
                "9999-12-31T00:00:00.000000001-00:00:01", "1970-01-01T00:00:00.0000000001Z"],
    "offset_datetime": ["2024-01-01T00:00:00+05:30:60", "2024-01-01T00:00:00", "-009999-01-01T00:00:00+00:00:01",
                        "2024-01-01T00:00:00 Z"],
    "duration": ["PT", "P1D", "PT1D", "pt1s", "PT1.5H", "PT1.S", "PT.5S", "PT1S1M", "PT1H1H", "+PT1S",
                 "PT1.1234567890S", "PT9223372036854775808S", "-PT9223372036854775808.000000001S", "PT-1S",
                 "P1DT1H", "PT1s", "", "-", "--PT1S", "PT1H ", "PT2562047788015216H"],
    "period": ["P", "P1H", "PT1H", "P1DT1H", "P1D1Y", "P1.5D", "+P1D", "P-1D", "p1d", "P9223372036854775808M",
               "-P9223372036854775809M", "P768614336404564650Y8M", "P1Y1Y", "", "P1317624576693539402W"],
}
# The texts §17 names, plus each parser's first and last rejection, run through
# the trapping parser on both lanes; every rejection runs through `try_`.
NAMED_REJECTIONS = {("time", "23:59:60"), ("date", "-000000-01-01"), ("time", "12:00:00.1234567890"),
                    ("instant", "9999-12-31T23:59:59Z"), ("duration", "PT9223372036854775808S"),
                    ("period", "P768614336404564650Y8M")}
TEXT_KINDS = {
    # kind: (row helper for the value, its width, option row helper)
    "date": ("[date_epoch_day({})]", 1, "row_opt_date"),
    "time": ("[time_nanosecond_of_day({})]", 1, "row_opt_time"),
    "datetime": ("row_dt({})", 2, "row_opt_dt"),
    "offset": ("[offset_seconds({})]", 1, "row_opt_offset"),
    "instant": ("row_instant({})", 2, "row_opt_instant"),
    "offset_datetime": ("row_odt({})", 3, "row_opt_odt"),
    "duration": ("row_duration({})", 2, "row_opt_duration"),
    "period": ("row_period({})", 2, "row_opt_period"),
}
TO_STRING = {
    "date": "date_to_string", "time": "time_to_string", "datetime": "datetime_to_string", "offset": "offset_to_string",
    "instant": "instant_to_string", "offset_datetime": "offset_datetime_to_string",
    "duration": "duration_to_string", "period": "period_to_string",
}


def text_corpus(corpus: Corpus) -> None:
    for kind, texts in VALID_TEXT.items():
        row, width, option = TEXT_KINDS[kind]
        parse, render_text = ref.PARSERS[kind]
        parser = f"parse_{kind}"
        corpus.grid(f"text_parse_{kind}", [text_axis("s", texts)], f"{option}(try_{parser}(s))",
                    lambda s, parse=parse, width=width: opt_row(lambda: parse(s), width))
        corpus.grid(f"text_canonical_{kind}", [text_axis("s", texts)], f"{TO_STRING[kind]}({parser}(s))",
                    lambda s, parse=parse, render_text=render_text: render_text(parse(s)), strings=True)
        corpus.grid(f"text_reparse_{kind}", [text_axis("s", texts)], row.format(f"{parser}({TO_STRING[kind]}({parser}(s)))"),
                    lambda s, parse=parse, render_text=render_text: flatten(parse(render_text(parse(s)))))
    for kind, texts in INVALID_TEXT.items():
        row, width, option = TEXT_KINDS[kind]
        parse, _ = ref.PARSERS[kind]
        parser = f"parse_{kind}"
        corpus.grid(f"text_reject_{kind}", [text_axis("s", texts)], f"{option}(try_{parser}(s))",
                    lambda s, parse=parse, width=width: opt_row(lambda: parse(s), width))
        for index, text in enumerate(texts):
            label = "named_reject" if (kind, text) in NAMED_REJECTIONS else f"reject_{kind}"
            if label == "named_reject" or index in (0, len(texts) - 1):
                corpus.fails(label, row.format(f"{parser}({text_lit(text)})"), expect_error(lambda: parse(text)))


def formatting(corpus: Corpus) -> None:
    in_range = [i for i in INSTANTS if ref.in_instant_range(i[0])]
    corpus.grid("instant_text", [tuple_axis("i", in_range)], "instant_to_string(inst2(i))", lambda i: ref.instant_to_string(i), strings=True)
    corpus.grid("instant_reparse", [tuple_axis("i", in_range)], "row_instant(parse_instant(instant_to_string(inst2(i))))", lambda i: list(i))
    corpus.grid("odt_text", [tuple_axis("i", in_range), int_axis("o", OFFSETS + [19_800])],
                "offset_datetime_to_string(offset_datetime(inst2(i), offset_from_seconds(o)))",
                lambda i, o: ref.offset_datetime_to_string((i, o)), strings=True)
    corpus.grid("odt_reparse", [tuple_axis("i", in_range), int_axis("o", OFFSETS + [19_800])],
                "row_odt(parse_offset_datetime(offset_datetime_to_string(offset_datetime(inst2(i), offset_from_seconds(o)))))",
                lambda i, o: [*i, o])
    offsets = [0, 1, -1, 86_399, -86_399, 3_600, -3_660, 19_800]
    corpus.grid("offset_text", [int_axis("o", offsets)], "offset_to_string(offset_from_seconds(o))", ref.offset_to_string, strings=True)
    corpus.grid("offset_reparse", [int_axis("o", offsets)], "[offset_seconds(parse_offset(offset_to_string(offset_from_seconds(o))))]", lambda o: [o])
    durations = EXTREME_DURATIONS + [(0, 500_000_000), (-1, 500_000_000), (3_661, 0), (-1, NANO - 1)]
    corpus.grid("duration_text", [tuple_axis("d", durations)], "duration_to_string(dur2(d))", ref.duration_to_string, strings=True)
    corpus.grid("duration_reparse", [tuple_axis("d", durations)], "row_duration(parse_duration(duration_to_string(dur2(d))))", lambda d: list(d))
    periods = [(0, 0), (14, 3), (-14, -3), (0, -1), (I64_MIN, 0), (0, I64_MIN), (I64_MAX, I64_MAX), (I64_MIN, I64_MIN)]
    corpus.grid("period_text", [tuple_axis("p", periods)], "period_to_string(per2(p))", ref.period_to_string, strings=True)
    corpus.grid("period_reparse", [tuple_axis("p", periods)], "row_period(parse_period(period_to_string(per2(p))))", lambda p: list(p))
    dts = DTS + [(0, 1), (-1, 10)]
    corpus.grid("datetime_text", [tuple_axis("x", dts)], "datetime_to_string(dt2(x))", ref.datetime_to_string, strings=True)
    corpus.grid("datetime_reparse", [tuple_axis("x", dts)], "row_dt(parse_datetime(datetime_to_string(dt2(x))))", lambda x: list(x))
    nods = [mix(k) * 40_234_567 % DAY_NANOS // 10 ** (k % 10) * 10 ** (k % 10) for k in range(40)]
    corpus.grid("time_text", [int_axis("t", nods)], "time_to_string(time_from_nanosecond_of_day(t))", ref.time_to_string, strings=True)
    corpus.grid("time_reparse", [int_axis("t", nods)], "[time_nanosecond_of_day(parse_time(time_to_string(time_from_nanosecond_of_day(t))))]", lambda t: [t])


def columns(corpus: Corpus) -> None:
    """§8.7: the S1 column types round-trip, and validate like their scalar producers."""
    days = [0, ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, -1, 19_782]
    tensor = f"to_tensor({list_lit(days)})"
    corpus.value("dates_column", f"to_list(dates_epoch_days(dates_from_epoch_days({tensor})))", days)
    mixed = [5, ref.MAX_EPOCH_DAY + 1, -7, I64_MIN]
    mixed_tensor = f"to_tensor({list_lit(mixed)})"
    corpus.value("try_dates_column", f"try_column_days(try_dates_from_epoch_days({mixed_tensor}))", ([5, 0, -7, 0], [True, False, True, False]))
    corpus.fails("dates_column", f"to_list(dates_epoch_days(dates_from_epoch_days({mixed_tensor})))",
                 expect_error(lambda: [ref.date_from_epoch_day(n, "dates_from_epoch_days") for n in mixed]))
    seconds = [0, ref.INSTANT_MIN_SECOND, ref.INSTANT_MAX_SECOND, -1]
    nanos = [0, 1, NANO - 1, 500]
    column = f"instants_from_unix(to_tensor({list_lit(seconds)}), to_tensor({list_lit(nanos)}))"
    corpus.value("instants_column", f"(to_list(instants_unix_seconds({column})), to_list(instants_nanoseconds({column})))", (seconds, nanos))
    bad_seconds = [1, ref.INSTANT_MAX_SECOND + 1, 2, 3]
    bad_nanos = [7, 0, NANO, 9]
    attempt = f"try_instants_from_unix(to_tensor({list_lit(bad_seconds)}), to_tensor({list_lit(bad_nanos)}))"
    corpus.value("try_instants_seconds", f"try_column_seconds({attempt})", ([1, 0, 0, 3], [True, False, False, True]))
    corpus.value("try_instants_nanos", f"try_column_nanos({attempt})", [7, 0, 0, 9])
    corpus.fails("instants_column", f"to_list(instants_unix_seconds(instants_from_unix(to_tensor({list_lit(bad_seconds)}), to_tensor({list_lit(bad_nanos)}))))",
                 expect_error(lambda: [ref.instant_from_unix(s, n, "instants_from_unix") for s, n in zip(bad_seconds, bad_nanos)]))


# ---------------------------------------------------------------------------
# Corpus: `Std.Datetime.Columns` (§10, stage S3). Columns are short literal
# lists: random elements across the whole range, the range edges, ties for
# every rounding mode, and month ends for both DayOverflow policies. Every
# failure names its own column callable, and a column with two failing
# elements must report the lower one; the reference names the element.

COLUMN_SEED = SEED + 2
INCREMENTS = ((0, 1), (0, 1_000), (1, 0), (900, 0), (28_800, 0), (86_400, 0))


def tensor_lit(values: Sequence[int]) -> str:
    return f"to_tensor({list_lit(values)})"


def dates_col(days: Sequence[int]) -> str:
    return f"dates_from_epoch_days({tensor_lit(days)})"


def instants_col(instants: Sequence[tuple[int, int]]) -> str:
    return f"instants_from_unix({tensor_lit([i[0] for i in instants])}, {tensor_lit([i[1] for i in instants])})"


def durations_col(ds: Sequence[tuple[int, int]]) -> str:
    return f"durations({tensor_lit([d[0] for d in ds])}, {tensor_lit([d[1] for d in ds])})"


def instants_rows(instants: Sequence[tuple[int, int]]) -> tuple[list[int], list[int]]:
    return [i[0] for i in instants], [i[1] for i in instants]


def column_inputs(rng: random.Random) -> dict[str, list]:
    days = [ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY + 1, -1, 0, 1, ref.MAX_EPOCH_DAY - 1, ref.MAX_EPOCH_DAY]
    days += [rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY) for _ in range(13)]
    month_ends = []
    for _ in range(12):
        year, month = rng.randint(ref.MIN_YEAR, ref.MAX_YEAR), rng.randint(1, 12)
        month_ends.append(ref.date(year, month, ref.days_in_month(year, month)))
    month_ends += [ref.date(2024, 1, 31), ref.date(2023, 1, 29), ref.date(2024, 3, 30)]
    edge_instants = [(ref.INSTANT_MIN_SECOND, 0), (ref.INSTANT_MIN_SECOND + 1, 999_999_999), (-1, 999_999_999), (0, 0),
                     (0, 1), (ref.INSTANT_MAX_SECOND - 1, 0), (ref.INSTANT_MAX_SECOND, 999_999_999)]
    instants = edge_instants + [(rng.randint(ref.INSTANT_MIN_SECOND, ref.INSTANT_MAX_SECOND), rng.randrange(NANO))
                                for _ in range(10)]
    # Exact ties between two multiples of each unit and increment, on both sides of the epoch.
    ties = [(-2, 500_000_000), (2, 500_000_000), (-5_400, 0), (5_400, 0), (-90, 0), (30, 0), (0, 500), (-1, 999_999_500),
            (0, 500_000), (-1, 999_500_000), (450, 0), (-450, 0), (14_400, 0), (-14_400, 0), (43_200, 0)]
    near_epoch = [(rng.randint(-9_000_000_000, 9_000_000_000), rng.randrange(NANO)) for _ in range(10)]
    return {"days": days, "month_ends": month_ends, "instants": instants, "ties": ties, "near_epoch": near_epoch}


def column_fields(corpus: Corpus, inputs: dict[str, list]) -> None:
    days = inputs["days"]
    column = dates_col(days)
    for name in ("dates_year", "dates_month", "dates_day", "dates_weekday_iso_number", "dates_day_of_year"):
        corpus.value(f"col_{name}", f"to_list({name}({column}))", getattr(ref, name)(days))
    corpus.value("col_dates_to_strings", f"dates_to_strings({column})", ref.dates_to_strings(days))
    fields = [ref.civil_from_days(day) for day in days]
    years, months, mdays = ([f[k] for f in fields] for k in range(3))
    corpus.value("col_dates_from_ymd", f"to_list(dates_epoch_days(dates_from_ymd({tensor_lit(years)}, {tensor_lit(months)}, {tensor_lit(mdays)})))",
                 ref.dates_from_ymd(years, months, mdays))
    bad = [(2024, 2, 29), (2023, 2, 29), (10_000, 1, 1), (-10_000, 12, 31), (2024, 0, 1), (2024, 13, 1), (2024, 4, 31),
           (2024, 1, 0), (I64_MAX, I64_MIN, I64_MAX), (-9999, 1, 1), (9999, 12, 31), (-44, 2, 29), (-100, 2, 29)]
    ys, ms, ds = ([b[k] for b in bad] for k in range(3))
    corpus.value("col_try_dates_from_ymd", f"try_column_days(try_dates_from_ymd({tensor_lit(ys)}, {tensor_lit(ms)}, {tensor_lit(ds)}))",
                 ref.try_dates_from_ymd(ys, ms, ds))
    two_bad = ys[:2] + [2024, 2024, 2023], ms[:2] + [1, 13, 2], ds[:2] + [1, 1, 29]
    corpus.fails("named_col_dates_from_ymd", f"to_list(dates_epoch_days(dates_from_ymd({', '.join(tensor_lit(v) for v in two_bad)})))",
                 expect_error(lambda: ref.dates_from_ymd(*two_bad)))
    texts = [ref.date_to_string(day) for day in days] + list(VALID_TEXT["date"]) + list(INVALID_TEXT["date"])
    corpus.value("col_try_parse_dates", f"try_column_days(try_parse_dates([{', '.join(text_lit(t) for t in texts)}]))",
                 ref.try_parse_dates(texts))


def column_date_arithmetic(corpus: Corpus, inputs: dict[str, list], rng: random.Random) -> None:
    days = inputs["days"]
    counts = [rng.randint(ref.MIN_EPOCH_DAY - day, ref.MAX_EPOCH_DAY - day) for day in days]
    counts[0], counts[-1] = ref.MAX_EPOCH_DAY - days[0], 0
    corpus.value("col_dates_add_days", f"to_list(dates_epoch_days(dates_add_days({dates_col(days)}, {tensor_lit(counts)})))",
                 ref.dates_add_days(days, counts))
    for label, shift in (("end", [0, 1, 0, 1]), ("start", [0, -1, 0, -1]), ("max", [0, I64_MAX, 0, 0]), ("min", [0, 0, I64_MIN, 0])):
        edge = [0, ref.MAX_EPOCH_DAY if label == "end" else ref.MIN_EPOCH_DAY, 5, ref.MAX_EPOCH_DAY if label == "end" else ref.MIN_EPOCH_DAY]
        corpus.fails(f"named_col_add_days_{label}", f"to_list(dates_epoch_days(dates_add_days({dates_col(edge)}, {tensor_lit(shift)})))",
                     expect_error(lambda edge=edge, shift=shift: ref.dates_add_days(edge, shift)))
    ends = inputs["month_ends"] + days
    months = []
    for day in ends:
        total = 12 * ref.civil_from_days(day)[0] + ref.civil_from_days(day)[1] - 1
        months.append(rng.randint(max(-119_988 - total, -30_000), min(119_999 - total, 30_000)))
    corpus.value("col_add_months_clamp", f"to_list(dates_epoch_days(dates_add_months({dates_col(ends)}, {tensor_lit(months)}, ClampToMonthEnd)))",
                 ref.dates_add_months(ends, months, "ClampToMonthEnd"))
    kept = [day for day in days if ref.civil_from_days(day)[2] <= 28]
    kept_months = [rng.randint(-12, 12) if ref.MIN_EPOCH_DAY + 400 < day < ref.MAX_EPOCH_DAY - 400 else 0 for day in kept]
    corpus.value("col_add_months_reject", f"to_list(dates_epoch_days(dates_add_months({dates_col(kept)}, {tensor_lit(kept_months)}, RejectInvalidDay)))",
                 ref.dates_add_months(kept, kept_months, "RejectInvalidDay"))
    january_31 = ref.date(2024, 1, 31)
    december_1 = ref.date(9999, 12, 1)
    for label, edge, shift, policy in (
        ("reject", [0, january_31, january_31], [1, 1, 2], "RejectInvalidDay"),
        ("domain_first", [0, january_31, december_1], [0, 1, 1], "RejectInvalidDay"),
        ("overflow_first", [december_1, january_31], [1, 1], "RejectInvalidDay"),
        ("range_before_day", [january_31], [I64_MAX], "RejectInvalidDay"),
        ("clamp_min", [0, 0], [0, I64_MIN], "ClampToMonthEnd"),
    ):
        corpus.fails(f"named_col_add_months_{label}",
                     f"to_list(dates_epoch_days(dates_add_months({dates_col(edge)}, {tensor_lit(shift)}, {policy})))",
                     expect_error(lambda edge=edge, shift=shift, policy=policy: ref.dates_add_months(edge, shift, policy)))
    other = [day if k % 3 == 0 else rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY) for k, day in enumerate(days)]
    corpus.value("col_dates_days_until", f"to_list(dates_days_until({dates_col(days)}, {dates_col(other)}))", ref.dates_days_until(days, other))
    for name in ref.DATE_ORDER:
        corpus.value(f"col_{name}", f"to_list({name}({dates_col(days)}, {dates_col(other)}))", ref.dates_order(name, days, other))


def column_durations(corpus: Corpus, rng: random.Random) -> None:
    pairs = [(rng.randint(-10**15, 10**15), rng.randint(-3 * NANO, 3 * NANO)) for _ in range(10)]
    pairs += [(I64_MAX, NANO - 1), (I64_MIN, 0), (I64_MIN + 1, -NANO), (-1, 999_999_999), (0, -1)]
    seconds, nanos = [p[0] for p in pairs], [p[1] for p in pairs]
    corpus.value("col_durations", f"durations_row(durations({tensor_lit(seconds)}, {tensor_lit(nanos)}))",
                 tuple(list(x) for x in zip(*ref.durations(seconds, nanos))))
    mixed = pairs[:3] + [(I64_MAX, NANO), (I64_MIN, -1), (I64_MAX, I64_MAX), (I64_MIN, I64_MIN)] + pairs[3:5]
    ms, mn = [p[0] for p in mixed], [p[1] for p in mixed]
    values, valid = ref.try_durations(ms, mn)
    corpus.value("col_try_durations", f"try_column_durations(try_durations({tensor_lit(ms)}, {tensor_lit(mn)}))",
                 (([v[0] for v in values], [v[1] for v in values]), valid))
    corpus.fails("named_col_durations", f"durations_row(durations({tensor_lit(ms)}, {tensor_lit(mn)}))",
                 expect_error(lambda: ref.durations(ms, mn)))


def column_instants(corpus: Corpus, inputs: dict[str, list], rng: random.Random) -> None:
    instants = inputs["instants"]
    for unit, per in ref.TIME_UNIT_NANOS.items():
        low = -((-ref.INSTANT_MIN_SECOND * NANO) // per)
        high = (ref.INSTANT_MAX_SECOND * NANO + NANO - 1) // per
        counts = [max(low, I64_MIN), min(high, I64_MAX), 0, -1, 1]
        counts += [rng.randint(max(low, I64_MIN), min(high, I64_MAX)) for _ in range(6)]
        corpus.value(f"col_from_count_{unit}", f"instants_row(instants_from_unix_count({tensor_lit(counts)}, {unit}))",
                     instants_rows(ref.instants_from_unix_count(counts, unit)))
        if high < I64_MAX:  # every i64 count of nanoseconds lies inside the instant range
            past = [0, high + 1, low - 1, I64_MAX]
            corpus.fails(f"named_col_from_count_{unit}", f"instants_row(instants_from_unix_count({tensor_lit(past)}, {unit}))",
                         expect_error(lambda past=past, unit=unit: ref.instants_from_unix_count(past, unit)))
    near = inputs["ties"] + inputs["near_epoch"]
    for unit in ref.TIME_UNIT_NANOS:
        column = near if unit == "Nanoseconds" else near + instants
        for rounding in ref.ROUNDING_TOTAL:
            corpus.value(f"col_to_count_{unit}_{rounding}", f"to_list(instants_to_unix_count({instants_col(column)}, {unit}, {rounding}))",
                         ref.instants_to_unix_count(column, unit, rounding))
        exact = [(s - s % (ref.TIME_UNIT_NANOS[unit] // NANO), 0) if ref.TIME_UNIT_NANOS[unit] >= NANO
                 else (s, ns - ns % ref.TIME_UNIT_NANOS[unit]) for s, ns in (near if unit == "Nanoseconds" else near + instants[2:5])]
        corpus.value(f"col_to_count_{unit}_exact", f"to_list(instants_to_unix_count({instants_col(exact)}, {unit}, RejectInexact))",
                     ref.instants_to_unix_count(exact, unit, "RejectInexact"))
        if unit != "Nanoseconds":
            inexact = exact[:2] + [(1, 1), (3, 7)]
            corpus.fails(f"named_col_to_count_inexact_{unit}", f"to_list(instants_to_unix_count({instants_col(inexact)}, {unit}, RejectInexact))",
                         expect_error(lambda inexact=inexact, unit=unit: ref.instants_to_unix_count(inexact, unit, "RejectInexact")))
    for label, column in (("max", [(0, 0), (ref.INSTANT_MAX_SECOND, 0)]), ("min", [(0, 0), (ref.INSTANT_MIN_SECOND, 0), (1, 1)])):
        corpus.fails(f"named_col_to_count_range_{label}", f"to_list(instants_to_unix_count({instants_col(column)}, Nanoseconds, RoundTiesToEven))",
                     expect_error(lambda column=column: ref.instants_to_unix_count(column, "Nanoseconds", "RoundTiesToEven")))
    shifts = []
    for second, nano in instants:
        total = rng.randint(ref.INSTANT_MIN_SECOND * NANO - second * NANO - nano, (ref.INSTANT_MAX_SECOND + 1) * NANO - 1 - second * NANO - nano)
        shifts.append(divmod(total, NANO))
    corpus.value("col_add_duration", f"instants_row(instants_add_duration({instants_col(instants)}, {durations_col(shifts)}))",
                 instants_rows(ref.instants_add_duration(instants, shifts)))
    for label, column, shift in (("end", [(0, 0), (ref.INSTANT_MAX_SECOND, 999_999_999)], [(1, 0), (0, 1)]),
                                 ("start", [(5, 0), (ref.INSTANT_MIN_SECOND, 0), (ref.INSTANT_MIN_SECOND, 0)], [(0, 0), (-1, 999_999_999), (I64_MIN, 0)]),
                                 ("far", [(0, 0)], [(I64_MAX, 999_999_999)])):
        corpus.fails(f"named_col_add_duration_{label}", f"instants_row(instants_add_duration({instants_col(column)}, {durations_col(shift)}))",
                     expect_error(lambda column=column, shift=shift: ref.instants_add_duration(column, shift)))
    reversed_instants = list(reversed(instants))
    corpus.value("col_until", f"durations_row(instants_until({instants_col(instants)}, {instants_col(reversed_instants)}))",
                 tuple(list(x) for x in zip(*ref.instants_until(instants, reversed_instants))))
    for name in ref.INSTANT_ORDER:
        twin = [(s, ns) if k % 2 else (s, min(ns + 1, NANO - 1)) for k, (s, ns) in enumerate(reversed_instants)]
        column = instants[:3] + twin[3:]
        corpus.value(f"col_{name}", f"to_list({name}({instants_col(instants)}, {instants_col(column)}))", ref.instants_order(name, instants, column))
    for offset in (-86_399, -3_600, 0, 19_800, 86_399):
        corpus.value(f"col_dates_at_{'west' if offset < 0 else 'east'}_{abs(offset)}", f"to_list(dates_epoch_days(instants_to_dates_at({instants_col(instants)}, {offset_src(offset)})))",
                     ref.instants_to_dates_at(instants, offset))
    for origin in ((0, 0), (ref.INSTANT_MIN_SECOND, 0), (ref.INSTANT_MAX_SECOND, 999_999_999), (-1, 999_999_999), instants[9]):
        corpus.values.append(Value(f"v{len(corpus.values):05d}_col_seconds_since",
                                   f"to_list(instants_seconds_since_f64({instants_col(instants + inputs['ties'])}, {instant_src(origin)}))",
                                   ref.instants_seconds_since_f64(instants + inputs["ties"], origin), float_bits=True))


def column_rounding(corpus: Corpus, inputs: dict[str, list]) -> None:
    column = inputs["ties"] + inputs["near_epoch"] + inputs["instants"][2:5]
    for increment in INCREMENTS:
        for rounding in ref.ROUNDING_TOTAL:
            corpus.value(f"col_round_{increment[0]}_{increment[1]}_{rounding}",
                         f"instants_row(instants_round_to({instants_col(column)}, {duration_src(increment)}, {rounding}))",
                         instants_rows(ref.instants_round_to(column, increment, rounding)))
        step = ref.nanos_of(increment)
        multiples = [ref.split_nanos(ref.nanos_of(i) - ref.nanos_of(i) % step) for i in column]
        corpus.value(f"col_round_{increment[0]}_{increment[1]}_exact",
                     f"instants_row(instants_round_to({instants_col(multiples)}, {duration_src(increment)}, RejectInexact))",
                     instants_rows(ref.instants_round_to(multiples, increment, "RejectInexact")))
        if step > 1:
            inexact = multiples[:3] + [(0, 1), (5, 3)]
            corpus.fails(f"named_col_round_inexact_{increment[0]}_{increment[1]}",
                         f"instants_row(instants_round_to({instants_col(inexact)}, {duration_src(increment)}, RejectInexact))",
                         expect_error(lambda inexact=inexact, increment=increment: ref.instants_round_to(inexact, increment, "RejectInexact")))
    for label, increment in (("zero", (0, 0)), ("negative", (-1, 0)), ("seven", (7, 0)), ("two_days", (172_800, 0))):
        corpus.fails(f"named_col_round_increment_{label}", f"instants_row(instants_round_to(instants_from_unix(span(0i64), span(0i64)), {duration_src(increment)}, RoundTiesToEven))",
                     expect_error(lambda increment=increment: ref.instants_round_to([], increment, "RoundTiesToEven")))
    for label, edge, increment, rounding in (("up", [(0, 0), (ref.INSTANT_MAX_SECOND, 999_999_999)], (1, 0), "RoundTowardPositive"),
                                             ("down", [(ref.INSTANT_MIN_SECOND, 0)], (2, 0), "RoundTowardNegative"),
                                             ("day", [(ref.INSTANT_MAX_SECOND, 1)], (86_400, 0), "RoundAwayFromZero")):
        corpus.fails(f"named_col_round_range_{label}", f"instants_row(instants_round_to({instants_col(edge)}, {duration_src(increment)}, {rounding}))",
                     expect_error(lambda edge=edge, increment=increment, rounding=rounding: ref.instants_round_to(edge, increment, rounding)))


def column_lengths(corpus: Corpus) -> None:
    """Columns whose lengths only the running program knows: the call rejects the pair."""
    two, three = "dates_from_epoch_days(span(2i64))", "dates_from_epoch_days(span(3i64))"
    stamps_two, stamps_three = "instants_from_unix(span(2i64), span(2i64))", "instants_from_unix(span(3i64), span(3i64))"
    cases = (
        ("dates_add_days", f"to_list(dates_epoch_days(dates_add_days({two}, span(3i64))))"),
        ("dates_add_months", f"to_list(dates_epoch_days(dates_add_months({three}, span(2i64), ClampToMonthEnd)))"),
        ("dates_days_until", f"to_list(dates_days_until({two}, {three}))"),
        ("dates_lte", f"to_list(dates_lte({three}, {two}))"),
        ("instants_add_duration", f"instants_row(instants_add_duration({stamps_two}, durations(span(3i64), span(3i64))))"),
        ("instants_until", f"durations_row(instants_until({stamps_three}, {stamps_two}))"),
        ("instants_gt", f"to_list(instants_gt({stamps_two}, {stamps_three}))"),
    )
    for function, expr in cases:
        corpus.fails(f"named_col_lengths_{function}", expr, ref.domain(function, "arguments have different lengths"))


def column_corpus(corpus: Corpus) -> None:
    """§10: every column callable against its scalar twin, element by element."""
    rng = random.Random(COLUMN_SEED)
    inputs = column_inputs(rng)
    columns = Corpus()
    column_fields(columns, inputs)
    column_date_arithmetic(columns, inputs, rng)
    column_durations(columns, rng)
    column_instants(columns, inputs, rng)
    column_rounding(columns, inputs)
    column_lengths(columns)
    corpus.columns += [Value("c" + value.name, value.expr, value.expected, value.float_bits) for value in columns.values]
    corpus.failures += [Failure("c" + failure.name, failure.expr, failure.function, failure.kind) for failure in columns.failures]


def build_ci_corpus() -> Corpus:
    corpus = Corpus()
    day_samples(corpus)
    periods_and_dates(corpus)
    calendar_queries(corpus)
    date_construction(corpus)
    date_arithmetic(corpus)
    holiday_helpers(corpus)
    time_and_datetime(corpus)
    offsets_and_instants(corpus)
    durations_and_periods(corpus)
    text_corpus(corpus)
    formatting(corpus)
    columns(corpus)
    column_corpus(corpus)
    return corpus


def build_exhaustive_corpus(chunk_days: int = 250_000) -> Corpus:
    corpus = Corpus()
    start = ref.MIN_EPOCH_DAY
    while start <= ref.MAX_EPOCH_DAY:
        stop = min(start + chunk_days, ref.MAX_EPOCH_DAY + 1)
        day_range(corpus, f"days_{start}", start, stop, C_ONLY)
        start = stop
    for first in range(0, 100_000, 20_000):
        period_pairs(corpus, f"period_until_{first}", first, 20_000, C_ONLY)
    for year in range(1900, 2101, 10):
        low = ref.days_from_civil(year, 1, 1)
        high = ref.days_from_civil(min(year + 10, 2101), 1, 1)
        corpus.grid(f"eval_days_{year}", [range_axis("n", low, high)], "day_row(n)", day_row, EVAL_ONLY)
    if len(corpus.days) != SPAN:
        raise AssertionError("the exhaustive profile must cover every day of the range")
    return corpus


# ---------------------------------------------------------------------------
# Running programs on the lanes.


@dataclass(frozen=True)
class Program:
    name: str
    source: str
    bindings: tuple[Value, ...] = ()
    failure: Failure | None = None
    lanes: tuple[str, ...] = BOTH


def program_source(body: list[str]) -> str:
    imports = ", ".join(TYPES + CONSTRUCTORS + FUNCTIONS)
    columns = ", ".join(COLUMN_TYPES + COLUMN_FUNCTIONS)
    rounding = ", ".join(("Rounding",) + ref.ROUNDING)
    text = "\n".join(body)
    return ("module Demo.Main\n\n" + f"import Std.Datetime ({imports})\n" + f"import Std.Datetime.Columns ({columns})\n"
            + f"import Std.Rounding ({rounding})\n\n" + prelude_for(text) + "\n" + text + "\n")


def representative_failures(failures: list[Failure], per_path: int) -> list[Failure]:
    """Every named failure, and up to `per_path` evenly spaced cases per label, function and kind.

    A failure needs a program of its own on each lane, so repeated paths are
    sampled; the `try_` forms still observe every grid point.
    """
    groups: dict[tuple[str, str, str], list[Failure]] = {}
    for failure in failures:
        label = failure.name.split("_", 1)[1]
        groups.setdefault((label, failure.function, failure.kind), []).append(failure)
    chosen = []
    for (label, _, _), members in groups.items():
        if label.startswith("named_") or len(members) <= per_path:
            chosen += members
        else:
            step = (len(members) - 1) / (per_path - 1)
            chosen += [members[round(k * step)] for k in range(per_path)]
    return sorted(chosen, key=lambda f: f.name)


def calls(expr: str, functions: Sequence[str]) -> bool:
    return any(re.search(rf"\b{function}\(", expr) for function in functions)


def column_family_values(columns: list[Value]) -> list[Value]:
    """The first column value that calls each family of `COLUMN_FAMILIES` no
    earlier choice calls, in corpus order."""
    chosen: list[Value] = []
    covered: set[str] = set()
    for value in columns:
        new = {family for family, functions in COLUMN_FAMILIES.items() if family not in covered and calls(value.expr, functions)}
        if new:
            chosen.append(value)
            covered |= {family for family, functions in COLUMN_FAMILIES.items() if calls(value.expr, functions)}
    missing = set(COLUMN_FAMILIES) - covered
    if missing:
        raise AssertionError(f"no column value calls the families {sorted(missing)}")
    return chosen


def make_programs(corpus: Corpus, chunk: int) -> list[Program]:
    programs = []
    for value in corpus.bulk:
        programs.append(Program(value.name, program_source([f"{value.name} = [{value.expr}]"]), (value,), lanes=value.lanes))
    for start in range(0, len(corpus.values), chunk):
        group = tuple(corpus.values[start:start + chunk])
        body = [f"{v.name} = [{v.expr}]" for v in group]
        programs.append(Program(f"values_{start:05d}", program_source(body), group))
    # Column literals are long, so their bindings get smaller programs. One
    # value per column family comes first, in a program the canary keeps.
    families = column_family_values(corpus.columns) if corpus.columns else []
    programs.append(Program("columns_families", program_source([f"{v.name} = [{v.expr}]" for v in families]), tuple(families)))
    rest = [value for value in corpus.columns if value not in families]
    for start in range(0, len(rest), COLUMN_CHUNK):
        group = tuple(rest[start:start + COLUMN_CHUNK])
        programs.append(Program(f"columns_{start:05d}", program_source([f"{v.name} = [{v.expr}]" for v in group]), group))
    for failure in representative_failures(corpus.failures, FAILURES_PER_PATH):
        programs.append(Program(failure.name, program_source([f"{failure.name} = [{failure.expr}]"]), failure=failure))
    return programs


# The `ci` programs the canary keeps: the first and last 400 days of the
# range, one value per `Std.Datetime.Columns` family, a domain failure, and
# an overflow failure. Each compiled C program builds the standard library,
# so the canary keeps to about one program per harness job.
CANARY_PROGRAMS = (
    "b000_days_first_400",
    "b001_days_last_400",
    "columns_families",
    "f0021_date",
    "f0064_months_trap",
)


def canary_programs(programs: list[Program]) -> list[Program]:
    """The `CANARY_PROGRAMS` of the `ci` programs, each unchanged; a missing name fails."""
    by_name = {program.name: program for program in programs}
    missing = [name for name in CANARY_PROGRAMS if name not in by_name]
    if missing:
        raise AssertionError(f"canary programs missing from the ci profile: {missing}")
    return [by_name[name] for name in CANARY_PROGRAMS]


def parse_bindings(stdout: str) -> tuple[dict[str, str], list[str]]:
    """Printed bindings by name, and every line that is not a `name = value` binding."""
    found: dict[str, str] = {}
    stray = []
    for line in stdout.splitlines():
        name, sep, value = line.partition(" = ")
        if sep and name not in found:
            found[name] = value
        else:
            stray.append(line)
    return found, stray


def float_bits(text: str) -> int | None:
    try:
        return struct.unpack("<Q", struct.pack("<d", float(text)))[0]
    except ValueError:
        return None


def compare_value(binding: Value, printed: str | None) -> str | None:
    if printed is None:
        return "no output line"
    expected = binding.resolve()
    if binding.float_bits and isinstance(expected, list):
        inner = printed[2:-2] if printed.startswith("[[") and printed.endswith("]]") else None
        got = [float_bits(x.strip()) for x in inner.split(",")] if inner else []
        want = [struct.unpack("<Q", struct.pack("<d", x))[0] for x in expected]
        if got != want:
            return f"expected {render([expected])} (bits {[hex(b) for b in want]}), printed {printed}"
        return None
    if binding.float_bits:
        assert isinstance(expected, float)
        inner = printed[1:-1] if printed.startswith("[") and printed.endswith("]") else None
        bits = struct.unpack("<Q", struct.pack("<d", expected))[0]
        if inner is None or float_bits(inner) != bits:
            return f"expected [{render(expected)}] (bits {bits:#018x}), printed {printed}"
        return None
    expected_text = render([expected])
    if printed != expected_text:
        return first_difference(expected, expected_text, printed)
    return None


def first_difference(expected: object, expected_text: str, printed: str) -> str:
    """Name the first differing element of a grid's flat list."""
    if isinstance(expected, list) and printed.startswith("[[") and expected_text.startswith("[["):
        want = [x.strip() for x in expected_text[2:-2].split(",")]
        got = [x.strip() for x in printed[2:-2].split(",")]
        for index, (w, g) in enumerate(zip(want, got)):
            if w != g:
                return f"element {index}: expected {w}, printed {g} ({len(want)} elements expected, {len(got)} printed)"
        if len(want) != len(got):
            return f"{len(want)} elements expected, {len(got)} printed"
    return f"expected {expected_text[:600]}, printed {printed[:600]}"


def failure_message(result: LaneResult) -> str | None:
    for line in result.stderr.splitlines():
        text = line.strip()
        if text.startswith("error: "):
            text = text[len("error: "):]
        name, sep, rest = text.partition(": ")
        kind, sep2, _ = rest.partition(": ")
        if sep and sep2 and kind in ("domain", "overflow") and name.replace("_", "").isalnum():
            return text
    return None


@dataclass
class Report:
    checked: int = 0
    failures_checked: int = 0
    problems: list[str] = field(default_factory=list)


def check_program(program: Program, results: list[LaneResult], report: Report) -> None:
    if program.failure is not None:
        failure = program.failure
        messages = {}
        for result in results:
            message = failure_message(result)
            if result.status == 0 or message is None:
                report.problems.append(
                    f"{failure.name} [{result.lane}/{result.stage}]: expected failure `{failure.function}: {failure.kind}: ...`, "
                    f"got status {result.status}; stdout {result.stdout.strip()[:300]!r}; stderr {result.stderr.strip()[:300]!r}\n"
                    f"    expression: {failure.expr}")
                continue
            messages[result.lane] = message
            name, _, rest = message.partition(": ")
            kind, _, detail = rest.partition(": ")
            if (name, kind) != (failure.function, failure.kind) or not detail.strip():
                report.problems.append(
                    f"{failure.name} [{result.lane}]: expected `{failure.function}: {failure.kind}: <detail>`, got `{message}`\n"
                    f"    expression: {failure.expr}")
        if len(set(messages.values())) > 1:
            report.problems.append(f"{failure.name}: lanes disagree on the message: {messages}")
        report.failures_checked += 1
        return
    for result in results:
        if result.status != 0:
            report.problems.append(
                f"{program.name} [{result.lane}/{result.stage}]: status {result.status}; stderr {result.stderr.strip()[:2000]}")
            continue
        printed, stray = parse_bindings(result.stdout)
        expected_names = {binding.name for binding in program.bindings}
        stray += [f"{name} = {value[:200]}" for name, value in printed.items() if name not in expected_names]
        if stray:
            report.problems.append(f"{program.name} [{result.lane}]: unexpected output lines: {stray[:10]}")
        for binding in program.bindings:
            problem = compare_value(binding, printed.get(binding.name))
            if problem:
                report.problems.append(f"{binding.name} [{result.lane}]: {problem}\n    expression: {binding.expr[:400]}")
    report.checked += len(program.bindings)


def run_all(runner: Runner, programs: list[Program], lanes: list[str], jobs: int, log) -> Report:
    report = Report()

    def run_one(program: Program) -> tuple[Program, list[LaneResult]]:
        results = []
        for lane in lanes:
            if lane not in program.lanes:
                continue
            results.append(runner.lane(program.name, program.source, lane))
        return program, results

    done = 0
    timings: list[tuple[float, str]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for program, results in pool.map(run_one, programs):
            check_program(program, results, report)
            timings += [(result.seconds, f"{program.name} [{result.lane}]") for result in results]
            done += 1
            if done % 25 == 0 or done == len(programs):
                log(f"{done}/{len(programs)} programs, {len(report.problems)} problems")
    timings.sort(reverse=True)
    lane_total = {lane: sum(t for t, name in timings if name.endswith(f"[{lane}]")) for lane in lanes}
    log("lane seconds: " + ", ".join(f"{lane} {total:.0f}" for lane, total in lane_total.items())
        + "; slowest: " + ", ".join(f"{name} {t:.0f}s" for t, name in timings[:8]))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chelis", type=Path, required=True, help="the chelis binary under test")
    parser.add_argument("--reef-home", type=Path, required=True, help="a reef home with chelis-std published")
    parser.add_argument("--toolchain-json", help="strict reference toolchain: {compiler, compile_flags, link_flags}")
    parser.add_argument("--profile", choices=("canary", "ci", "exhaustive"), required=True)
    parser.add_argument("--lanes", default="eval,c")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=150, help="value bindings per program")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per lane process")
    parser.add_argument("--work", type=Path, help="keep generated programs here instead of a temporary directory")
    parser.add_argument("--list", action="store_true", help="print the corpus size and exit")
    parser.add_argument("--only", help="diagnosis: run only programs whose name matches this regular expression")
    args = parser.parse_args(argv)

    lanes = args.lanes.split(",")
    if not lanes or any(lane not in BOTH for lane in lanes):
        parser.error("--lanes is a comma-separated subset of eval,c")
    toolchain = None
    if "c" in lanes:
        if not args.toolchain_json:
            parser.error("the c lane needs --toolchain-json")
        toolchain = Toolchain.from_json(args.toolchain_json)

    ref.check_range_constants()
    corpus = build_exhaustive_corpus() if args.profile == "exhaustive" else build_ci_corpus()
    programs = make_programs(corpus, args.chunk)
    if args.profile == "canary":
        programs = canary_programs(programs)
    if args.only:
        programs = [program for program in programs if re.search(args.only, program.name)]
    programs = [program for program in programs if set(program.lanes) & set(lanes)]
    if args.profile == "canary":
        summary = (f"canary: {sum(len(p.bindings) for p in programs)} bindings, "
                   f"{sum(p.failure is not None for p in programs)} failure cases, {len(programs)} programs")
    else:
        summary = (f"{len(corpus.days)} days in day rows, {len(corpus.bulk)} grids, {len(corpus.values)} values, "
                   f"{len(corpus.columns)} column values, "
                   f"{len(corpus.failures)} failure cases, {len(programs)} programs")
    print(f"corpus: {summary}", flush=True)
    if args.list:
        for program in programs:
            print(f"  {program.name} [{'+'.join(program.lanes)}]")
        return 0

    def log(message: str) -> None:
        print(message, flush=True)

    with tempfile.TemporaryDirectory(prefix="datetime-oracle-") as scratch:
        work = args.work or Path(scratch)
        runner = Runner(args.chelis.resolve(), args.reef_home.resolve(), work, toolchain, args.timeout, "datetime-oracle")
        report = run_all(runner, programs, lanes, args.jobs, log)

    for problem in report.problems[:200]:
        print(f"DISAGREEMENT {problem}")
    if report.problems:
        print(f"STD DATETIME ORACLE: FAIL ({len(report.problems)} disagreements; {summary}; lanes {'+'.join(lanes)})")
        return 1
    print(f"{PASS_MARKER} ({report.checked} observations and {report.failures_checked} failure cases "
          f"on lanes {'+'.join(lanes)}; {summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
