#!/usr/bin/env python3
"""Differential oracle for `Std.Datetime` S1 (chelis#2859): eval and C against the reference.

The harness generates Chelis programs that call `Std.Datetime`, runs each one
through `chelis eval` and through `chelis build --target c` plus the native
compiler, and compares both lanes' complete output with the values that
`datetime_reference.py` computes independently. A lane agreeing with the other
lane is not enough: each must equal the reference.

Two kinds of case are generated:

- A value observation is a top-level binding whose printed value the reference
  predicts exactly. Opaque values are observed through their exported
  accessors, an `Option` through a `(present, value)` pair, and every binding
  is wrapped in a one-element list so tuples print on one line.
- A failure case is a program whose single binding must fail. Both lanes must
  exit nonzero with the same §5 message, `<function>: <kind>: <detail>`, whose
  function and kind equal the reference's and whose detail is nonempty. This
  also proves that no primitive numeric trap escapes (§5): a trap would carry
  a primitive's message instead.

Profiles:

- `ci`: the range edges, every day from 1900-01-01 through 2100-12-31, seeded
  random samples across the whole range, every policy branch, every year's
  Easter, and the text corpus (valid and invalid), on both lanes.
- `exhaustive`: the manual gate of `docs/manual_gates.md`. Every one of the
  7 304 484 days in compiled C (field round trip, weekday, day of year, ISO
  week and its inverse, text round trip), and `date_period_until`'s defining
  property on seeded pairs. Pass `--lanes c`.

The supported entry point is `crates/chelis-cli/tests/std_datetime_oracle.rs`,
which supplies the freshly built binary, a published standard library and the
strict reference toolchain (`chelis_backend_c::toolchain`).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import tempfile

import datetime_reference as ref
from datetime_reference import DatetimeError, I64_MAX, I64_MIN

SEED = 2859
FIRST_1900 = ref.days_from_civil(1900, 1, 1)
LAST_2100 = ref.days_from_civil(2100, 12, 31)
PASS_MARKER = "STD DATETIME ORACLE: PASS"
FAILURES_PER_PATH = 3

TYPES = (
    "Date", "Time", "DateTime", "Instant", "Offset", "OffsetDateTime", "Duration", "Period",
    "Weekday", "DayOverflow", "TimeRounding", "TimeUnit",
)
CONSTRUCTORS = ref.WEEKDAYS + ref.DAY_OVERFLOW + ref.TIME_ROUNDING + tuple(ref.TIME_UNIT_NANOS)
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
)

# Encoders turn opaque values into printable integers; `opt_*` turns an Option
# into `(present, value)` with a fixed placeholder when absent.
PRELUDE = """\
def b2i(x: bool) -> i64 = if x then 1i64 else 0i64

def enc_dt(x: DateTime) -> (i64, i64) = (date_epoch_day(datetime_date(x)), time_nanosecond_of_day(datetime_time(x)))

def enc_instant(x: Instant) -> (i64, i64) = (instant_unix_second(x), instant_nanosecond(x))

def enc_duration(x: Duration) -> (i64, i64) = (duration_second(x), duration_nanosecond(x))

def enc_period(x: Period) -> (i64, i64) = (period_months(x), period_days(x))

def enc_odt(x: OffsetDateTime) -> ((i64, i64), i64) = (enc_instant(offset_datetime_instant(x)), offset_seconds(offset_datetime_offset(x)))

def enc_carry(x: (i64, Time)) -> (i64, i64) = {
  (carry, t) = x
  (carry, time_nanosecond_of_day(t))
}

def opt_i64(o: Option[i64]) -> (bool, i64) = match o with {
  | Some(v) => (true, v)
  | None => (false, 0i64)
}

def opt_date(o: Option[Date]) -> (bool, i64) = match o with {
  | Some(v) => (true, date_epoch_day(v))
  | None => (false, 0i64)
}

def opt_time(o: Option[Time]) -> (bool, i64) = match o with {
  | Some(v) => (true, time_nanosecond_of_day(v))
  | None => (false, 0i64)
}

def opt_weekday(o: Option[Weekday]) -> (bool, i64) = match o with {
  | Some(v) => (true, weekday_iso_number(v))
  | None => (false, 0i64)
}

def opt_offset(o: Option[Offset]) -> (bool, i64) = match o with {
  | Some(v) => (true, offset_seconds(v))
  | None => (false, 0i64)
}

def opt_dt(o: Option[DateTime]) -> (bool, (i64, i64)) = match o with {
  | Some(v) => (true, enc_dt(v))
  | None => (false, (0i64, 0i64))
}

def opt_instant(o: Option[Instant]) -> (bool, (i64, i64)) = match o with {
  | Some(v) => (true, enc_instant(v))
  | None => (false, (0i64, 0i64))
}

def opt_duration(o: Option[Duration]) -> (bool, (i64, i64)) = match o with {
  | Some(v) => (true, enc_duration(v))
  | None => (false, (0i64, 0i64))
}

def opt_period(o: Option[Period]) -> (bool, (i64, i64)) = match o with {
  | Some(v) => (true, enc_period(v))
  | None => (false, (0i64, 0i64))
}

def opt_odt(o: Option[OffsetDateTime]) -> (bool, ((i64, i64), i64)) = match o with {
  | Some(v) => (true, enc_odt(v))
  | None => (false, ((0i64, 0i64), 0i64))
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

def year_row(y: i64) -> List[i64] = [days_in_year(y), b2i(is_leap_year(y)), date_epoch_day(easter_sunday_gregorian(y)), date_epoch_day(easter_sunday_orthodox(y))]

def period_property(pair: (i64, i64)) -> List[i64] = {
  (a, b) = pair
  p = date_period_until(date_from_epoch_day(a), date_from_epoch_day(b))
  [period_months(p), period_days(p), date_epoch_day(date_add_period(date_from_epoch_day(a), p, ClampToMonthEnd))]
}
"""


# ---------------------------------------------------------------------------
# Chelis source builders. Every builder returns Surf text for an expression.


def lit(n: int) -> str:
    if not ref.fits_i64(n):
        raise ValueError(f"{n} does not fit in i64")
    if n == I64_MIN:
        return "(-9223372036854775807i64 - 1i64)"
    return f"({n}i64)" if n < 0 else f"{n}i64"


def text_lit(s: str) -> str:
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def list_lit(values: list[int]) -> str:
    return "[" + ", ".join(lit(v) for v in values) + "]"


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


def opt(value: object | None, placeholder: object) -> tuple[bool, object]:
    return (False, placeholder) if value is None else (True, value)


ZERO_PAIR = (0, 0)


@dataclass(frozen=True)
class Value:
    """A binding whose printed value the reference predicts."""

    name: str
    expr: str
    expected: object
    float_bits: bool = False

    def resolve(self) -> object:
        """Bulk observations carry a zero-argument function; compute it on demand."""
        return self.expected() if callable(self.expected) else self.expected


@dataclass(frozen=True)
class Failure:
    """A program whose binding must fail with `<function>: <kind>: <detail>`."""

    name: str
    expr: str
    function: str
    kind: str


@dataclass
class Corpus:
    values: list[Value] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)
    bulk: list[Value] = field(default_factory=list)
    days: set[int] = field(default_factory=set)

    def value(self, label: str, expr: str, expected: object, float_bits: bool = False) -> None:
        self.values.append(Value(f"v{len(self.values):05d}_{label}", expr, expected, float_bits))

    def fails(self, label: str, expr: str, error: DatetimeError) -> None:
        self.failures.append(Failure(f"f{len(self.failures):04d}_{label}", expr, error.function, error.kind))

    def outcome(self, label: str, expr: str, compute, encode=lambda v: v) -> None:
        """Record a trapping call as a value or a failure, whichever the reference says."""
        try:
            result = compute()
        except DatetimeError as error:
            self.fails(label, expr, error)
            return
        self.value(label, expr, encode(result))


def expect_error(compute) -> DatetimeError:
    try:
        compute()
    except DatetimeError as error:
        return error
    raise AssertionError("the reference was expected to fail")


def try_value(compute):
    """A `try_` form: None exactly where the reference fails with `domain`."""
    try:
        return compute()
    except DatetimeError as error:
        if error.kind == "domain":
            return None
        raise


# ---------------------------------------------------------------------------
# Corpus construction.


def interesting_days(rng: random.Random) -> list[int]:
    days = set(range(ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY + 400))
    days |= set(range(ref.MAX_EPOCH_DAY - 399, ref.MAX_EPOCH_DAY + 1))
    days |= {rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY) for _ in range(4000)}
    for year in rng.sample(range(ref.MIN_YEAR + 1, ref.MAX_YEAR), 150) + [-1, 0, 1, 1582, 1969, 1970]:
        start = ref.days_from_civil(year, 12, 22)
        days |= set(range(start, start + 18))
        march = ref.days_from_civil(year, 2, 27)
        days |= set(range(march, march + 4))
    return sorted(days)


def day_row(n: int) -> list[int]:
    year, month, day = ref.civil_from_days(n)
    weekday = ref.weekday_of(n)
    iso_year, iso_week = ref.iso_week(n)
    return [
        year, month, day, ref.date(year, month, day), weekday, ref.day_of_year(n), iso_year, iso_week,
        ref.date_from_iso_week(iso_year, iso_week, weekday), ref.parse_date(ref.date_to_string(n)),
    ]


def year_row(year: int) -> list[int]:
    return [
        ref.days_in_year(year), int(ref.is_leap_year(year)),
        ref.easter_sunday_gregorian(year), ref.easter_sunday_orthodox(year),
    ]


def add_bulk_days(corpus: Corpus, label: str, days: list[int], contiguous: bool) -> None:
    corpus.days.update(days)
    if contiguous:
        source = f"range({lit(days[0])}, {lit(days[-1] + 1)})"
    else:
        source = list_lit(days)
    corpus.bulk.append(Value(
        f"b{len(corpus.bulk):03d}_{label}",
        f"flat_map(fn (n: i64) -> day_row(n), {source})",
        lambda: [x for n in days for x in day_row(n)],
    ))


def add_bulk_strings(corpus: Corpus, label: str, days: list[int]) -> None:
    corpus.bulk.append(Value(
        f"b{len(corpus.bulk):03d}_{label}",
        f"join(map(fn (n: i64) -> date_to_string(date_from_epoch_day(n)), {list_lit(days)}), \" \")",
        " ".join(ref.date_to_string(n) for n in days),
    ))


def add_period_property(corpus: Corpus, label: str, pairs: list[tuple[int, int]]) -> None:
    source = "[" + ", ".join(f"({lit(a)}, {lit(b)})" for a, b in pairs) + "]"

    def expected() -> list[int]:
        rows = []
        for a, b in pairs:
            months, days = ref.date_period_until(a, b)
            rows += [months, days, ref.date_add_period(a, (months, days), "ClampToMonthEnd")]
        return rows

    corpus.bulk.append(Value(
        f"b{len(corpus.bulk):03d}_{label}",
        f"flat_map(fn (pair: (i64, i64)) -> period_property(pair), {source})",
        expected,
    ))


def calendar_queries(corpus: Corpus) -> None:
    years = list(range(ref.MIN_YEAR, ref.MAX_YEAR + 1))
    corpus.bulk.append(Value(
        f"b{len(corpus.bulk):03d}_every_year",
        f"flat_map(fn (y: i64) -> year_row(y), range({lit(ref.MIN_YEAR)}, {lit(ref.MAX_YEAR + 1)}))",
        [x for y in years for x in year_row(y)],
    ))
    for year in (I64_MIN, I64_MIN + 1, -10_000, 10_000, I64_MAX - 1, I64_MAX, -400, -100, -4, 0):
        corpus.value("leap", f"b2i(is_leap_year({lit(year)}))", int(ref.is_leap_year(year)))
        corpus.value("year_days", f"days_in_year({lit(year)})", ref.days_in_year(year))
        for month in range(1, 13):
            corpus.value("month_days", f"days_in_month({lit(year)}, {lit(month)})", ref.days_in_month(year, month))
    for month in (0, 13, -1, I64_MIN, I64_MAX):
        corpus.fails("month_days", f"days_in_month(2024i64, {lit(month)})", expect_error(lambda: ref.days_in_month(2024, month)))
    for iso in range(1, 8):
        corpus.value("weekday_number", f"weekday_iso_number({weekday_src(iso)})", iso)
        corpus.value("weekday_name", f"weekday_name({weekday_src(iso)})", ref.weekday_name(iso))
    for number in (0, 1, 7, 8, -1, I64_MIN, I64_MAX):
        corpus.value("try_weekday", f"opt_weekday(try_weekday_from_iso_number({lit(number)}))",
                     opt(try_value(lambda: ref.weekday_from_iso_number(number)), 0))
        corpus.outcome("weekday_from", f"weekday_iso_number(weekday_from_iso_number({lit(number)}))",
                       lambda: ref.weekday_from_iso_number(number))


def date_construction(corpus: Corpus) -> None:
    fields = [
        (2024, 2, 29), (2023, 2, 29), (2024, 2, 30), (2024, 4, 31), (2024, 13, 1), (2024, 0, 1),
        (2024, 1, 0), (2024, 1, 32), (0, 2, 29), (-1, 2, 29), (-4, 2, 29), (-100, 2, 29), (-400, 2, 29),
        (ref.MIN_YEAR, 1, 1), (ref.MAX_YEAR, 12, 31), (ref.MIN_YEAR - 1, 12, 31), (ref.MAX_YEAR + 1, 1, 1),
        (I64_MIN, 1, 1), (I64_MAX, 1, 1), (2024, I64_MIN, 1), (2024, 1, I64_MAX), (2024, 1, -1),
    ]
    for year, month, day in fields:
        args = f"{lit(year)}, {lit(month)}, {lit(day)}"
        corpus.value("try_date", f"opt_date(try_date({args}))", opt(try_value(lambda: ref.date(year, month, day)), 0))
        corpus.outcome("date", f"date_epoch_day(date({args}))", lambda: ref.date(year, month, day))
    for n in (ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, ref.MIN_EPOCH_DAY - 1, ref.MAX_EPOCH_DAY + 1, 0, I64_MIN, I64_MAX):
        corpus.value("try_from_epoch", f"opt_date(try_date_from_epoch_day({lit(n)}))",
                     opt(try_value(lambda: ref.date_from_epoch_day(n)), 0))
        corpus.outcome("from_epoch", f"date_epoch_day(date_from_epoch_day({lit(n)}))", lambda: ref.date_from_epoch_day(n))
    iso_cases = [(2020, 53, 4), (2021, 53, 1), (2021, 52, 7), (2026, 0, 1), (2026, 54, 1),
                 (-10_000, 52, 7), (-10_000, 53, 7), (10_000, 1, 1), (10_000, 1, 7), (-10_001, 1, 1),
                 (10_001, 1, 1), (I64_MIN, 1, 1), (I64_MAX, 1, 1), (2026, I64_MAX, 1), (2026, I64_MIN, 1)]
    for iso_year, week, weekday in iso_cases:
        args = f"{lit(iso_year)}, {lit(week)}, {weekday_src(weekday)}"
        corpus.value("try_iso", f"opt_date(try_date_from_iso_week({args}))",
                     opt(try_value(lambda: ref.date_from_iso_week(iso_year, week, weekday)), 0))
        corpus.outcome("from_iso", f"date_epoch_day(date_from_iso_week({args}))",
                       lambda: ref.date_from_iso_week(iso_year, week, weekday))


def date_arithmetic(corpus: Corpus, rng: random.Random) -> None:
    edges = [ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, 0, -1]
    steps = [0, 1, -1, 7, -7, ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY, ref.MIN_EPOCH_DAY - ref.MAX_EPOCH_DAY,
             I64_MAX, I64_MIN, I64_MAX - 1, I64_MIN + 1]
    for start in edges:
        for n in steps:
            corpus.outcome("add_days", f"date_epoch_day(date_add_days({date_src(start)}, {lit(n)}))",
                           lambda: ref.date_add_days(start, n))
    for a in edges:
        for b in edges:
            corpus.value("days_until", f"date_days_until({date_src(a)}, {date_src(b)})", ref.date_days_until(a, b))
            corpus.value("compare", f"[date_lt({date_src(a)}, {date_src(b)}), date_lte({date_src(a)}, {date_src(b)}), "
                         f"date_gt({date_src(a)}, {date_src(b)}), date_gte({date_src(a)}, {date_src(b)})]",
                         [a < b, a <= b, a > b, a >= b])

    starts = [ref.days_from_civil(y, m, d) for y, m, d in (
        (2024, 1, 31), (2024, 2, 29), (2023, 1, 31), (2024, 3, 31), (2024, 5, 31), (2024, 8, 30),
        (2000, 2, 29), (1900, 1, 29), (-1, 12, 31), (0, 2, 29), (ref.MIN_YEAR, 1, 31), (ref.MAX_YEAR, 12, 31),
        (ref.MAX_YEAR, 11, 30), (ref.MIN_YEAR, 2, 28),
    )]
    months = list(range(-25, 26)) + [1200, -1200, 12 * 19_998, -12 * 19_998, 12 * 19_999, -12 * 19_999,
                                     I64_MAX, I64_MIN, I64_MAX - 11, I64_MIN + 11]
    for start in starts:
        for n in months:
            for policy in ref.DAY_OVERFLOW:
                args = f"{date_src(start)}, {lit(n)}, {policy}"
                ordinary = False
                try:
                    expected = opt(try_value(lambda: ref.date_add_months(start, n, policy)), 0)
                except DatetimeError as error:
                    corpus.fails("try_add_months", f"opt_date(try_date_add_months({args}))",
                                 DatetimeError("try_date_add_months", error.kind, error.detail))
                else:
                    corpus.value("try_add_months", f"opt_date(try_date_add_months({args}))", expected)
                    ordinary = abs(n) <= 25 and expected[0]
                if ordinary:
                    continue  # an ordinary success is covered once, by the try form
                corpus.outcome("add_months", f"date_epoch_day(date_add_months({args}))",
                               lambda: ref.date_add_months(start, n, policy))

    periods = [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (13, 45), (-13, -45), (12, 0), (0, -366), (1, 30),
               (-1, -30), (I64_MAX, 0), (I64_MIN, 0), (0, I64_MAX), (0, I64_MIN), (240_000, 0), (-240_000, 0)]
    for start in starts:
        for p in periods:
            for policy in ref.DAY_OVERFLOW:
                args = f"{date_src(start)}, {period_src(p)}, {policy}"
                try:
                    expected = opt(try_value(lambda: ref.date_add_period(start, p, policy)), 0)
                except DatetimeError as error:
                    corpus.fails("try_add_period", f"opt_date(try_date_add_period({args}))",
                                 DatetimeError("try_date_add_period", error.kind, error.detail))
                else:
                    corpus.value("try_add_period", f"opt_date(try_date_add_period({args}))", expected)
                corpus.outcome("add_period", f"date_epoch_day(date_add_period({args}))",
                               lambda: ref.date_add_period(start, p, policy))

    pairs = [(rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY), rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY))
             for _ in range(1500)]
    near = []
    for _ in range(1500):
        a = rng.randint(ref.MIN_EPOCH_DAY + 1000, ref.MAX_EPOCH_DAY - 1000)
        near.append((a, a + rng.randint(-800, 800)))
    month_ends = [s for s in starts]
    structured = [(a, b) for a in month_ends for b in month_ends]
    structured += [(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY), (ref.MAX_EPOCH_DAY, ref.MIN_EPOCH_DAY)]
    add_period_property(corpus, "period_until", pairs + near + structured)


def holiday_helpers(corpus: Corpus, rng: random.Random) -> None:
    years = [2024, 2023, 2000, 1900, 0, -1, ref.MIN_YEAR, ref.MAX_YEAR] + rng.sample(range(ref.MIN_YEAR, ref.MAX_YEAR + 1), 6)
    for year in years:
        for month in (1, 2, 5, 11, 12):
            for weekday in range(1, 8):
                for n in (1, 2, 4, 5, 6, I64_MAX):
                    args = f"{lit(year)}, {lit(month)}, {weekday_src(weekday)}, {lit(n)}"
                    corpus.value("nth_weekday", f"opt_date(nth_weekday_in_month({args}))",
                                 opt(ref.nth_weekday_in_month(year, month, weekday, n), 0))
                corpus.value("last_weekday", f"date_epoch_day(last_weekday_in_month({lit(year)}, {lit(month)}, {weekday_src(weekday)}))",
                             ref.last_weekday_in_month(year, month, weekday))
    for year, month, n in ((2024, 11, 0), (2024, 11, -1), (2024, 11, I64_MIN), (2024, 0, 1), (2024, 13, 1),
                           (ref.MIN_YEAR - 1, 1, 1), (ref.MAX_YEAR + 1, 1, 1), (I64_MAX, 1, 1)):
        args = f"{lit(year)}, {lit(month)}, Thursday, {lit(n)}"
        corpus.fails("nth_weekday", f"opt_date(nth_weekday_in_month({args}))",
                     expect_error(lambda: ref.nth_weekday_in_month(year, month, 4, n)))
    for year, month in ((2024, 0), (2024, 13), (ref.MIN_YEAR - 1, 12), (ref.MAX_YEAR + 1, 1), (I64_MIN, 1)):
        corpus.fails("last_weekday", f"date_epoch_day(last_weekday_in_month({lit(year)}, {lit(month)}, Monday))",
                     expect_error(lambda: ref.last_weekday_in_month(year, month, 1)))
    days = [ref.MIN_EPOCH_DAY + k for k in range(7)] + [ref.MAX_EPOCH_DAY - k for k in range(7)] + [0, 19_000]
    for day in days:
        for weekday in range(1, 8):
            corpus.outcome("on_or_after", f"date_epoch_day(weekday_on_or_after({date_src(day)}, {weekday_src(weekday)}))",
                           lambda: ref.weekday_on_or_after(day, weekday))
            corpus.outcome("on_or_before", f"date_epoch_day(weekday_on_or_before({date_src(day)}, {weekday_src(weekday)}))",
                           lambda: ref.weekday_on_or_before(day, weekday))
    for year in (ref.MIN_YEAR - 1, ref.MAX_YEAR + 1, I64_MIN, I64_MAX):
        corpus.fails("easter_g", f"date_epoch_day(easter_sunday_gregorian({lit(year)}))",
                     expect_error(lambda: ref.easter_sunday_gregorian(year)))
        corpus.fails("easter_o", f"date_epoch_day(easter_sunday_orthodox({lit(year)}))",
                     expect_error(lambda: ref.easter_sunday_orthodox(year)))


NANO = ref.NANOS_PER_SECOND
DAY_NANOS = ref.NANOS_PER_DAY
EXTREME_DURATIONS = [(0, 0), (1, 0), (-1, 0), (0, 1), (-1, NANO - 1), (86_399, NANO - 1), (86_400, 0),
                     (-86_400, 0), (I64_MAX, NANO - 1), (I64_MIN, 0), (I64_MIN, 1), (I64_MAX, 0),
                     (3_000_000_000, 5), (-3_000_000_000, 5)]


def time_and_datetime(corpus: Corpus) -> None:
    for fields in ((0, 0, 0, 0), (23, 59, 59, NANO - 1), (12, 30, 15, 500), (24, 0, 0, 0), (23, 60, 0, 0),
                   (23, 59, 60, 0), (0, 0, 0, NANO), (0, 0, 0, -1), (-1, 0, 0, 0), (I64_MAX, 0, 0, 0),
                   (0, I64_MIN, 0, 0)):
        args = ", ".join(lit(x) for x in fields)
        corpus.value("try_time", f"opt_time(try_time({args}))", opt(try_value(lambda: ref.time(*fields)), 0))
        corpus.outcome("time", f"time_nanosecond_of_day(time({args}))", lambda: ref.time(*fields))
    for nod in (0, DAY_NANOS - 1, DAY_NANOS, -1, 45_296_789_000_001, I64_MAX, I64_MIN):
        corpus.value("try_time_nod", f"opt_time(try_time_from_nanosecond_of_day({lit(nod)}))",
                     opt(try_value(lambda: ref.time_from_nanosecond_of_day(nod)), 0))
        corpus.outcome("time_nod", f"time_nanosecond_of_day(time_from_nanosecond_of_day({lit(nod)}))",
                       lambda: ref.time_from_nanosecond_of_day(nod))
    times = [0, DAY_NANOS - 1, 45_296_789_000_001, 43_200 * NANO]
    for t in times:
        corpus.value("time_fields", f"[time_hour({time_src(t)}), time_minute({time_src(t)}), "
                     f"time_second({time_src(t)}), time_nanosecond({time_src(t)})]", list(ref.time_fields(t)))
        corpus.value("time_text", time_to_text_src(t), ref.time_to_string(t))
        for d in EXTREME_DURATIONS:
            corpus.value("time_add", f"enc_carry(time_add_duration({time_src(t)}, {duration_src(d)}))",
                         ref.time_add_duration(t, d))
        for u in times:
            corpus.value("time_until", f"enc_duration(time_until({time_src(t)}, {time_src(u)}))", ref.time_until(t, u))
            corpus.value("time_compare", f"[time_lt({time_src(t)}, {time_src(u)}), time_lte({time_src(t)}, {time_src(u)}), "
                         f"time_gt({time_src(t)}, {time_src(u)}), time_gte({time_src(t)}, {time_src(u)})]",
                         [t < u, t <= u, t > u, t >= u])

    dts = [(ref.MIN_EPOCH_DAY, 0), (ref.MAX_EPOCH_DAY, DAY_NANOS - 1), (0, 0), (-1, DAY_NANOS - 1), (19_631, 34_200 * NANO)]
    for dt in dts:
        corpus.value("datetime_parts", f"enc_dt({dt_src(dt)})", dt)
        for d in EXTREME_DURATIONS + [(1, 0), (-1, 0), (0, NANO - 1)]:
            corpus.outcome("dt_add", f"enc_dt(datetime_add_duration({dt_src(dt)}, {duration_src(d)}))",
                           lambda: ref.datetime_add_duration(dt, d))
        for p in ((1, 0), (-1, 0), (0, 1), (0, -1), (I64_MAX, 0), (12, 0)):
            for policy in ref.DAY_OVERFLOW:
                args = f"{dt_src(dt)}, {period_src(p)}, {policy}"
                try:
                    expected = opt(try_value(lambda: ref.datetime_add_period(dt, p, policy)), ZERO_PAIR)
                except DatetimeError as error:
                    corpus.fails("try_dt_period", f"opt_dt(try_datetime_add_period({args}))",
                                 DatetimeError("try_datetime_add_period", error.kind, error.detail))
                else:
                    corpus.value("try_dt_period", f"opt_dt(try_datetime_add_period({args}))", expected)
                corpus.outcome("dt_period", f"enc_dt(datetime_add_period({args}))",
                               lambda: ref.datetime_add_period(dt, p, policy))
        for other in dts:
            corpus.value("dt_until", f"enc_duration(datetime_until({dt_src(dt)}, {dt_src(other)}))",
                         ref.datetime_until(dt, other))
            a, b = ref.civil_nanos(dt), ref.civil_nanos(other)
            corpus.value("dt_compare", f"[datetime_lt({dt_src(dt)}, {dt_src(other)}), datetime_lte({dt_src(dt)}, {dt_src(other)}), "
                         f"datetime_gt({dt_src(dt)}, {dt_src(other)}), datetime_gte({dt_src(dt)}, {dt_src(other)})]",
                         [a < b, a <= b, a > b, a >= b])
    # Jan 31 + 1 month under RejectInvalidDay fails `domain` through datetime_add_period.
    jan31 = (ref.days_from_civil(2023, 1, 31), 0)
    corpus.fails("dt_period_reject", f"enc_dt(datetime_add_period({dt_src(jan31)}, {period_src((1, 0))}, RejectInvalidDay))",
                 expect_error(lambda: ref.datetime_add_period(jan31, (1, 0), "RejectInvalidDay")))


def time_to_text_src(t: int) -> str:
    return f"time_to_string({time_src(t)})"


INSTANT_EDGES = [(ref.INSTANT_MIN_SECOND, 0), (ref.INSTANT_MAX_SECOND, NANO - 1), (0, 0), (-1, NANO - 1),
                 (9_223_372_036, 854_775_807), (9_223_372_036, 854_775_808), (-9_223_372_037, 145_224_192),
                 (-9_223_372_037, 145_224_191), (1_727_789_400, 123_456_789), (-1_000_000_000, 500_000_000)]


def offsets_and_instants(corpus: Corpus) -> None:
    for seconds in (0, 1, -1, 86_399, -86_399, 86_400, -86_400, 19_800, -16_200, I64_MAX, I64_MIN):
        corpus.value("try_offset", f"opt_offset(try_offset_from_seconds({lit(seconds)}))",
                     opt(try_value(lambda: ref.offset_from_seconds(seconds)), 0))
        corpus.outcome("offset", f"offset_seconds(offset_from_seconds({lit(seconds)}))", lambda: ref.offset_from_seconds(seconds))
    candidates = [(ref.INSTANT_MIN_SECOND - 1, 0), (ref.INSTANT_MAX_SECOND + 1, 0), (0, NANO), (0, -1),
                  (I64_MIN, 0), (I64_MAX, 0)] + INSTANT_EDGES
    for second, nano in candidates:
        args = f"{lit(second)}, {lit(nano)}"
        corpus.value("try_instant", f"opt_instant(try_instant_from_unix({args}))",
                     opt(try_value(lambda: ref.instant_from_unix(second, nano)), ZERO_PAIR))
        corpus.outcome("instant", f"enc_instant(instant_from_unix({args}))", lambda: ref.instant_from_unix(second, nano))
    for unit, per in ref.TIME_UNIT_NANOS.items():
        high = ((ref.INSTANT_MAX_SECOND + 1) * NANO - 1) // per
        low = -((-ref.INSTANT_MIN_SECOND * NANO) // per)
        for count in (0, 1, -1, high, high + 1, low, low - 1, I64_MAX, I64_MIN):
            if not ref.fits_i64(count):
                continue
            corpus.value("try_from_count", f"opt_instant(try_instant_from_unix_count({lit(count)}, {unit}))",
                         opt(try_value(lambda: ref.instant_from_unix_count(count, unit)), ZERO_PAIR))
            corpus.outcome("from_count", f"enc_instant(instant_from_unix_count({lit(count)}, {unit}))",
                           lambda: ref.instant_from_unix_count(count, unit))
        for i in INSTANT_EDGES:
            if not ref.in_instant_range(i[0]):
                continue
            for rounding in ref.TIME_ROUNDING:
                args = f"{instant_src(i)}, {unit}, {rounding}"
                corpus.value("try_to_count", f"opt_i64(try_instant_to_unix_count({args}))",
                             opt(ref.try_instant_to_unix_count(i, unit, rounding), 0))
                corpus.outcome("to_count", f"instant_to_unix_count({args})",
                               lambda: ref.instant_to_unix_count(i, unit, rounding))
    for i in INSTANT_EDGES:
        for d in EXTREME_DURATIONS:
            corpus.outcome("instant_add", f"enc_instant(instant_add_duration({instant_src(i)}, {duration_src(d)}))",
                           lambda: ref.instant_add_duration(i, d))
        for other in INSTANT_EDGES:
            corpus.value("instant_until", f"enc_duration(instant_until({instant_src(i)}, {instant_src(other)}))",
                         ref.instant_until(i, other))
            a, b = ref.nanos_of(i), ref.nanos_of(other)
            corpus.value("instant_compare", f"[instant_lt({instant_src(i)}, {instant_src(other)}), instant_lte({instant_src(i)}, {instant_src(other)}), "
                         f"instant_gt({instant_src(i)}, {instant_src(other)}), instant_gte({instant_src(i)}, {instant_src(other)})]",
                         [a < b, a <= b, a > b, a >= b])
        for increment in ((0, 1), (1, 0), (0, 500_000_000), (1, 500_000_000), (3_600, 0), (86_400, 0),
                          (7, 0), (0, 0), (-1, 0), (172_800, 0), (0, 7), (I64_MAX, 0)):
            for rounding in ref.TIME_ROUNDING:
                corpus.outcome("round_to", f"enc_instant(instant_round_to({instant_src(i)}, {duration_src(increment)}, {rounding}))",
                               lambda: ref.instant_round_to(i, increment, rounding))
        for seconds in (0, 86_399, -86_399, 3_600):
            corpus.value("to_datetime_at", f"enc_dt(instant_to_datetime_at({instant_src(i)}, {offset_src(seconds)}))",
                         ref.instant_to_datetime_at(i, seconds))
            corpus.value("odt", f"enc_odt(offset_datetime({instant_src(i)}, {offset_src(seconds)}))", (i, seconds))
            corpus.value("odt_local", f"enc_dt(offset_datetime_local(offset_datetime({instant_src(i)}, {offset_src(seconds)})))",
                         ref.instant_to_datetime_at(i, seconds))
    for dt in ((ref.MIN_EPOCH_DAY, 0), (ref.MAX_EPOCH_DAY, DAY_NANOS - 1), (0, 0), (ref.MIN_EPOCH_DAY, 86_399 * NANO),
               (ref.MAX_EPOCH_DAY, 0)):
        for seconds in (0, 86_399, -86_399, 1, -1):
            corpus.outcome("to_instant_at", f"enc_instant(datetime_to_instant_at({dt_src(dt)}, {offset_src(seconds)}))",
                           lambda: ref.datetime_to_instant_at(dt, seconds))


def durations_and_periods(corpus: Corpus) -> None:
    raw = [(0, 0), (1, -1), (-1, 1), (0, -1_500_000_000), (0, I64_MAX), (0, I64_MIN), (I64_MAX, NANO - 1),
           (I64_MAX, NANO), (I64_MIN, 0), (I64_MIN, -1), (I64_MIN, I64_MAX), (I64_MAX, I64_MIN), (5, 3 * NANO)]
    for second, nano in raw:
        corpus.outcome("duration", f"enc_duration(duration({lit(second)}, {lit(nano)}))", lambda: ref.duration(second, nano))
    for unit in ref.TIME_UNIT_NANOS:
        for count in (0, 1, -1, I64_MAX, I64_MIN, 2_562_047_788_015_215, 2_562_047_788_015_216):
            corpus.outcome("from_count", f"enc_duration(duration_from_count({lit(count)}, {unit}))",
                           lambda: ref.duration_from_count(count, unit))
        for d in EXTREME_DURATIONS + [(-2, 500_000_000), (2, 500_000_000), (-3, 500_000_000), (0, 1_500_000)]:
            for rounding in ref.TIME_ROUNDING:
                args = f"{duration_src(d)}, {unit}, {rounding}"
                corpus.value("try_dur_count", f"opt_i64(try_duration_to_count({args}))",
                             opt(ref.try_duration_to_count(d, unit, rounding), 0))
                corpus.outcome("dur_count", f"duration_to_count({args})", lambda: ref.duration_to_count(d, unit, rounding))
    for d in EXTREME_DURATIONS + [(1_727_789_400, 123_456_789), (2**53, 1), (2**53 - 1, NANO - 1), (-(2**53), 999_999_999),
                                  (-1, 999_999_000), (-1, 999_000_000), (-2, 1)]:
        corpus.value("seconds_f64", f"duration_to_seconds_f64({duration_src(d)})", ref.duration_to_seconds_f64(d), float_bits=True)
        corpus.outcome("negate", f"enc_duration(duration_negate({duration_src(d)}))", lambda: ref.duration_negate(d))
        for other in EXTREME_DURATIONS:
            corpus.outcome("dur_add", f"enc_duration(duration_add({duration_src(d)}, {duration_src(other)}))",
                           lambda: ref.duration_add(d, other))
            corpus.outcome("dur_sub", f"enc_duration(duration_sub({duration_src(d)}, {duration_src(other)}))",
                           lambda: ref.duration_sub(d, other))
            a, b = ref.nanos_of(d), ref.nanos_of(other)
            corpus.value("dur_compare", f"[duration_lt({duration_src(d)}, {duration_src(other)}), duration_lte({duration_src(d)}, {duration_src(other)}), "
                         f"duration_gt({duration_src(d)}, {duration_src(other)}), duration_gte({duration_src(d)}, {duration_src(other)})]",
                         [a < b, a <= b, a > b, a >= b])
        for k in (0, 1, -1, 2, -2, 1_000_000_007, I64_MAX, I64_MIN):
            corpus.outcome("dur_mul", f"enc_duration(duration_mul({duration_src(d)}, {lit(k)}))", lambda: ref.duration_mul(d, k))

    for months, days in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1), (I64_MIN, 0), (0, I64_MIN),
                         (I64_MAX, I64_MAX), (I64_MIN, I64_MIN), (I64_MIN, 1), (I64_MAX, -1)):
        args = f"{lit(months)}, {lit(days)}"
        corpus.value("try_period", f"opt_period(try_period({args}))", opt(try_value(lambda: ref.period(months, days)), ZERO_PAIR))
        try:
            p = ref.period(months, days)
        except DatetimeError as error:
            corpus.fails("period", f"enc_period(period({args}))", error)
            continue
        corpus.value("period", f"enc_period(period({args}))", p)
        corpus.outcome("period_negate", f"enc_period(period_negate({period_src(p)}))", lambda: ref.period_negate(p))
        for k in (0, 1, -1, 3, I64_MAX, I64_MIN):
            corpus.outcome("period_mul", f"enc_period(period_mul({period_src(p)}, {lit(k)}))", lambda: ref.period_mul(p, k))


# §8.8 text corpus: (kind, text) pairs. Accepted texts must parse to the
# reference value and format to the canonical spelling; rejected texts make
# the `try_` parser return None and the trapping parser fail `domain`.
VALID_TEXT = {
    "date": ["2024-02-29", "0000-01-01", "-000001-12-31", "+002024-01-01", "+000000-06-15", "-009999-01-01",
             "9999-12-31", "1970-01-01", "+009999-12-31", "-000400-02-29", "0001-01-01"],
    "time": ["00:00:00", "23:59:59", "12:34:56.7", "12:34:56.45", "12:34:56.100", "12:34:56.1234",
             "12:34:56.12345", "12:34:56.123456", "12:34:56.1234567", "12:34:56.12345678",
             "12:34:56.123456789", "12:34:56.000000001", "12:34:56.0", "12:34:56.000000000"],
    "datetime": ["2024-02-29T12:00:00", "2024-02-29t12:00:00", "2024-02-29 12:00:00.5",
                 "-009999-01-01T00:00:00", "9999-12-31T23:59:59.999999999", "+000000-01-01T00:00:00.010"],
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
                 "PT2562047788015215H30M7.999999999S", "-PT9223372036854775807.999999999S"],
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
PARSE_ENCODER = {
    "date": ("opt_date", "parse_date", 0),
    "time": ("opt_time", "parse_time", 0),
    "datetime": ("opt_dt", "parse_datetime", ZERO_PAIR),
    "offset": ("opt_offset", "parse_offset", 0),
    "instant": ("opt_instant", "parse_instant", ZERO_PAIR),
    "offset_datetime": ("opt_odt", "parse_offset_datetime", (ZERO_PAIR, 0)),
    "duration": ("opt_duration", "parse_duration", ZERO_PAIR),
    "period": ("opt_period", "parse_period", ZERO_PAIR),
}
ENCODE_PLAIN = {
    "date": "date_epoch_day", "time": "time_nanosecond_of_day", "datetime": "enc_dt", "offset": "offset_seconds",
    "instant": "enc_instant", "offset_datetime": "enc_odt", "duration": "enc_duration", "period": "enc_period",
}
TO_STRING = {
    "date": "date_to_string", "time": "time_to_string", "datetime": "datetime_to_string", "offset": "offset_to_string",
    "instant": "instant_to_string", "offset_datetime": "offset_datetime_to_string",
    "duration": "duration_to_string", "period": "period_to_string",
}


def text_corpus(corpus: Corpus) -> None:
    for kind, texts in VALID_TEXT.items():
        option, parser, placeholder = PARSE_ENCODER[kind]
        parse, render_text = ref.PARSERS[kind]
        for text in texts:
            value = parse(text)
            corpus.value(f"parse_{kind}", f"{option}(try_{parser}({text_lit(text)}))", (True, value))
            corpus.value(f"canon_{kind}", f"{TO_STRING[kind]}({parser}({text_lit(text)}))", render_text(value))
            canonical = render_text(value)
            corpus.value(f"reparse_{kind}", f"{ENCODE_PLAIN[kind]}({parser}({text_lit(canonical)}))", value)
    for kind, texts in INVALID_TEXT.items():
        option, parser, placeholder = PARSE_ENCODER[kind]
        parse, _ = ref.PARSERS[kind]
        for index, text in enumerate(texts):
            error = expect_error(lambda: parse(text))
            corpus.value(f"reject_{kind}", f"{option}(try_{parser}({text_lit(text)}))", (False, placeholder))
            # Every rejected text also runs through the trapping parser, so each
            # rejection's message is checked on both lanes.
            corpus.fails(f"reject_{kind}", f"{ENCODE_PLAIN[kind]}({parser}({text_lit(text)}))", error)


def formatting(corpus: Corpus, rng: random.Random) -> None:
    for n in [ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY, ref.days_from_civil(0, 1, 1), ref.days_from_civil(-1, 12, 31)]:
        corpus.value("date_text", f"date_to_string({date_src(n)})", ref.date_to_string(n))
    for i in INSTANT_EDGES:
        if ref.in_instant_range(i[0]):
            corpus.value("instant_text", f"instant_to_string({instant_src(i)})", ref.instant_to_string(i))
            corpus.value("instant_reparse", f"enc_instant(parse_instant(instant_to_string({instant_src(i)})))", i)
            for seconds in (0, 86_399, -86_399, 19_800, -1):
                odt = f"offset_datetime({instant_src(i)}, {offset_src(seconds)})"
                corpus.value("odt_text", f"offset_datetime_to_string({odt})", ref.offset_datetime_to_string((i, seconds)))
                corpus.value("odt_reparse", f"enc_odt(parse_offset_datetime(offset_datetime_to_string({odt})))", (i, seconds))
    for seconds in (0, 1, -1, 86_399, -86_399, 3_600, -3_660, 19_800):
        corpus.value("offset_text", f"offset_to_string({offset_src(seconds)})", ref.offset_to_string(seconds))
        corpus.value("offset_reparse", f"offset_seconds(parse_offset(offset_to_string({offset_src(seconds)})))", seconds)
    for d in EXTREME_DURATIONS + [(0, 500_000_000), (-1, 500_000_000), (3_661, 0)]:
        corpus.value("duration_text", f"duration_to_string({duration_src(d)})", ref.duration_to_string(d))
        corpus.value("duration_reparse", f"enc_duration(parse_duration(duration_to_string({duration_src(d)})))", d)
    for p in ((0, 0), (14, 3), (-14, -3), (0, -1), (I64_MIN, 0), (0, I64_MIN), (I64_MAX, I64_MAX), (I64_MIN, I64_MIN)):
        corpus.value("period_text", f"period_to_string({period_src(p)})", ref.period_to_string(p))
        corpus.value("period_reparse", f"enc_period(parse_period(period_to_string({period_src(p)})))", p)
    for dt in ((ref.MIN_EPOCH_DAY, 0), (ref.MAX_EPOCH_DAY, DAY_NANOS - 1), (0, 1), (-1, 10)):
        corpus.value("dt_text", f"datetime_to_string({dt_src(dt)})", ref.datetime_to_string(dt))
        corpus.value("dt_reparse", f"enc_dt(parse_datetime(datetime_to_string({dt_src(dt)})))", dt)
    for _ in range(60):
        t = rng.randrange(DAY_NANOS)
        t -= t % 10 ** rng.randint(0, 9)
        corpus.value("time_text", f"time_to_string({time_src(t)})", ref.time_to_string(t))
        corpus.value("time_reparse", f"time_nanosecond_of_day(parse_time(time_to_string({time_src(t)})))", t)


def build_ci_corpus() -> Corpus:
    rng = random.Random(SEED)
    corpus = Corpus()
    add_bulk_days(corpus, "days_1900_2100", list(range(FIRST_1900, LAST_2100 + 1)), contiguous=True)
    sample = interesting_days(rng)
    add_bulk_days(corpus, "days_edges_and_samples", sample, contiguous=False)
    add_bulk_strings(corpus, "text_edges_and_samples", sample)
    calendar_queries(corpus)
    date_construction(corpus)
    date_arithmetic(corpus, rng)
    holiday_helpers(corpus, rng)
    time_and_datetime(corpus)
    offsets_and_instants(corpus)
    durations_and_periods(corpus)
    text_corpus(corpus)
    formatting(corpus, rng)
    return corpus


def build_exhaustive_corpus(chunk_days: int) -> Corpus:
    corpus = Corpus()
    start = ref.MIN_EPOCH_DAY
    while start <= ref.MAX_EPOCH_DAY:
        stop = min(start + chunk_days, ref.MAX_EPOCH_DAY + 1)
        add_bulk_days(corpus, f"days_{start}", list(range(start, stop)), contiguous=True)
        start = stop
    rng = random.Random(SEED)
    pairs = [(rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY), rng.randint(ref.MIN_EPOCH_DAY, ref.MAX_EPOCH_DAY))
             for _ in range(100_000)]
    for _ in range(100_000):
        a = rng.randint(ref.MIN_EPOCH_DAY + 2000, ref.MAX_EPOCH_DAY - 2000)
        pairs.append((a, a + rng.randint(-1500, 1500)))
    for index in range(0, len(pairs), 20_000):
        add_period_property(corpus, f"period_until_{index}", pairs[index:index + 20_000])
    if len(corpus.days) != ref.MAX_EPOCH_DAY - ref.MIN_EPOCH_DAY + 1:
        raise AssertionError("the exhaustive profile must cover every day of the range")
    return corpus


# ---------------------------------------------------------------------------
# Running programs on the lanes.


@dataclass(frozen=True)
class Toolchain:
    compiler: str
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]


@dataclass(frozen=True)
class Program:
    name: str
    source: str
    bindings: tuple[Value, ...] = ()
    failure: Failure | None = None


@dataclass
class LaneResult:
    lane: str
    status: int | None
    stdout: str
    stderr: str
    stage: str


def program_source(body: list[str]) -> str:
    imports = ", ".join(TYPES + CONSTRUCTORS + FUNCTIONS)
    return "module Demo.Main\n\n" + f"import Std.Datetime ({imports})\n" + "import Std.Text (join)\n\n" + PRELUDE + "\n" + "\n".join(body) + "\n"


def representative_failures(failures: list[Failure], per_path: int) -> list[Failure]:
    """Every rejected text, and up to `per_path` evenly spaced cases per label and kind.

    A failure needs a program of its own on each lane, so the grid's repeated
    paths are sampled; the `try_` forms still observe every grid point.
    """
    groups: dict[tuple[str, str, str], list[Failure]] = {}
    for failure in failures:
        label = failure.name.split("_", 1)[1]
        groups.setdefault((label, failure.function, failure.kind), []).append(failure)
    chosen = []
    for (label, _, _), members in groups.items():
        if label.startswith("reject_") or len(members) <= per_path:
            chosen += members
        else:
            step = (len(members) - 1) / (per_path - 1)
            chosen += [members[round(k * step)] for k in range(per_path)]
    return sorted(chosen, key=lambda f: f.name)


def make_programs(corpus: Corpus, chunk: int) -> list[Program]:
    programs = []
    for value in corpus.bulk:
        programs.append(Program(value.name, program_source([f"{value.name} = [{value.expr}]"]), (value,)))
    for start in range(0, len(corpus.values), chunk):
        group = tuple(corpus.values[start:start + chunk])
        body = [f"{v.name} = [{v.expr}]" for v in group]
        programs.append(Program(f"values_{start:05d}", program_source(body), group))
    for failure in representative_failures(corpus.failures, FAILURES_PER_PATH):
        programs.append(Program(failure.name, program_source([f"{failure.name} = [{failure.expr}]"]), failure=failure))
    return programs


class Runner:
    def __init__(self, chelis: Path, reef_home: Path, app_template: Path, work: Path, toolchain: Toolchain | None, timeout: int):
        self.chelis = chelis
        self.reef_home = reef_home
        self.app_template = app_template
        self.work = work
        self.toolchain = toolchain
        self.timeout = timeout

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"CHELIS_REEF_HOME": str(self.reef_home), "CHELIS_STYLE_GATE_DISABLE": "1", "OMP_NUM_THREADS": "1"})
        return env

    def app(self, program: Program, lane: str) -> Path:
        app = self.work / lane / program.name
        if app.exists():
            shutil.rmtree(app)
        (app / "src").mkdir(parents=True)
        shutil.copy(self.app_template / "reef.toml", app / "reef.toml")
        (app / "src" / "main.ch").write_text(program.source, encoding="utf-8")
        return app

    def run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(argv, cwd=cwd, env=self.env(), capture_output=True, text=True, timeout=self.timeout, check=False)

    def eval_lane(self, program: Program) -> LaneResult:
        app = self.app(program, "eval")
        done = self.run([str(self.chelis), "eval", "--file", "src/main.ch"], app)
        return LaneResult("eval", done.returncode, done.stdout, done.stderr, "eval")

    def c_lane(self, program: Program) -> LaneResult:
        assert self.toolchain is not None
        app = self.app(program, "c")
        build = self.run([str(self.chelis), "build", "src/main.ch", "--target", "c", "--output", "out"], app)
        if build.returncode != 0:
            return LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        link = self.run([self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", "out/main.c",
                         "out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"], app)
        if link.returncode != 0:
            return LaneResult("c", link.returncode, link.stdout, link.stderr, "link")
        done = self.run([str(app / "out" / "case")], app)
        return LaneResult("c", done.returncode, done.stdout, done.stderr, "run")


def parse_bindings(stdout: str) -> dict[str, str]:
    found = {}
    for line in stdout.splitlines():
        name, sep, value = line.partition(" = ")
        if sep:
            found[name] = value
    return found


def float_bits(text: str) -> int | None:
    try:
        return struct.unpack("<Q", struct.pack("<d", float(text)))[0]
    except ValueError:
        return None


def compare_value(binding: Value, printed: str | None) -> str | None:
    if printed is None:
        return "no output line"
    expected = binding.resolve()
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
    """Name the first differing element of a bulk list or a space-joined text list."""
    if isinstance(expected, (list, str)) and printed.startswith("[[") == expected_text.startswith("[["):
        separator = "," if isinstance(expected, list) else " "
        want = [x.strip() for x in expected_text.strip("[]").split(separator)]
        got = [x.strip() for x in printed.strip("[]").split(separator)]
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
        kind, sep2, detail = rest.partition(": ")
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
        printed = parse_bindings(result.stdout)
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
            try:
                results.append(runner.eval_lane(program) if lane == "eval" else runner.c_lane(program))
            except subprocess.TimeoutExpired as error:
                results.append(LaneResult(lane, None, "", f"timed out after {error.timeout} s", "timeout"))
        return program, results

    done = 0
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for program, results in pool.map(run_one, programs):
            check_program(program, results, report)
            done += 1
            if done % 50 == 0 or done == len(programs):
                log(f"{done}/{len(programs)} programs, {len(report.problems)} problems")
    return report


def write_app_template(chelis: Path, root: Path) -> Path:
    version = subprocess.run([str(chelis), "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
    template = root / "template"
    template.mkdir(parents=True, exist_ok=True)
    (template / "reef.toml").write_text(
        'schema = "1"\n\n[package]\nname = "datetime-oracle"\nversion = "0.1.0"\n'
        f'compiler = "={version}"\nmodule_prefix = "Demo"\n\n[dependencies]\nchelis-std = {{ version = "0.4.0" }}\n',
        encoding="utf-8")
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chelis", type=Path, required=True, help="the chelis binary under test")
    parser.add_argument("--reef-home", type=Path, required=True, help="a reef home with chelis-std published")
    parser.add_argument("--toolchain-json", help="strict reference toolchain: {compiler, compile_flags, link_flags}")
    parser.add_argument("--profile", choices=("ci", "exhaustive"), required=True)
    parser.add_argument("--lanes", default="eval,c")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=400, help="value bindings per program")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per lane process")
    parser.add_argument("--work", type=Path, help="keep generated programs here instead of a temporary directory")
    parser.add_argument("--list", action="store_true", help="print the corpus size and exit")
    args = parser.parse_args(argv)

    lanes = args.lanes.split(",")
    if not lanes or any(lane not in ("eval", "c") for lane in lanes):
        parser.error("--lanes is a comma-separated subset of eval,c")
    toolchain = None
    if "c" in lanes:
        if not args.toolchain_json:
            parser.error("the c lane needs --toolchain-json")
        spec = json.loads(args.toolchain_json)
        toolchain = Toolchain(spec["compiler"], tuple(spec["compile_flags"]), tuple(spec["link_flags"]))

    ref.check_range_constants()
    corpus = build_ci_corpus() if args.profile == "ci" else build_exhaustive_corpus(chunk_days=250_000)
    programs = make_programs(corpus, args.chunk)
    summary = (f"{len(corpus.days)} days in day rows, {len(corpus.values)} values, {len(corpus.bulk)} bulk observations, "
               f"{len(corpus.failures)} failure cases, {len(programs)} programs")
    print(f"corpus: {summary}", flush=True)
    if args.list:
        return 0

    def log(message: str) -> None:
        print(message, flush=True)

    with tempfile.TemporaryDirectory(prefix="datetime-oracle-") as scratch:
        work = args.work or Path(scratch)
        template = write_app_template(args.chelis, work)
        runner = Runner(args.chelis.resolve(), args.reef_home.resolve(), template, work, toolchain, args.timeout)
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
