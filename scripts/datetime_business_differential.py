#!/usr/bin/env python3
"""Differential oracle for `Std.Datetime.Business` (chelis#2860): eval and C against the reference.

The harness generates Chelis programs over the seeded calendar corpus of
`datetime_business_reference.py`, runs each through `chelis eval` and
through `chelis build --target c` plus the native compiler, and compares
each lane's output with the values the reference computes from the
definitions in [05-OP-73]. Agreement between the lanes is not enough: each
must equal the reference.

- A calendar program observes, for each query day, `try_is_business_day`,
  every roll, and every start with each offset through the `try_` forms,
  then the counts of its query pairs, then the vectorized forms over the
  elements whose scalar twin succeeds. An absent `Option` prints as
  `ABSENT`, outside every epoch day and count. Drawn holidays, query days
  and query pairs come from a linear congruential generator that the
  program and the reference both run, so programs stay small.
- Compiled C observes every query of a calendar, grouped into few programs
  because each build compiles the standard library; `chelis eval` observes
  a subset of each calendar's days, offsets and pairs, one calendar per
  program.
- A combination program observes `try_business_in_all` and
  `try_business_in_any` of two calendars: presence, weekmask, holidays, and
  horizon.
- A failure program has one binding that must fail. Both lanes must exit
  nonzero with the same `<function>: domain: <detail>` message, whose
  function equals the expected one, and a vectorized failure must name the
  element the reference names.

Profiles:

- `full`: the edge calendars and `--calendars` seeded ones, the
  combinations, and every failure path. It runs nightly.
- `canary`: both range-edge calendars on both lanes and one vectorized
  failure; every observation is one `full` also makes.
  It runs on every pull request.

The reference itself is checked against NumPy by
`test_datetime_business_reference.py`. The supported entry point is
`crates/chelis-cli/tests/std_datetime_business_oracle.rs`, which supplies
the freshly built binary, a published standard library and the strict
reference toolchain.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import datetime_business_reference as ref

SEED = 2860
PASS_MARKER = "STD DATETIME BUSINESS ORACLE: PASS"
ABSENT = -999_999_999
BOTH = ("eval", "c")

IMPORTS = (
    "import Std.Datetime (Date, Dates, date_from_epoch_day, date_epoch_day, dates_from_epoch_days, dates_epoch_days)\n"
    "import Std.Datetime.Business (Weekmask, BusinessCalendar, BusinessDayRoll, NonBusinessStart, "
    + ", ".join(ref.ROLLS + ref.STARTS)
    + ", business_calendar, try_business_calendar, business_calendar_weekmask, business_calendar_holidays, "
    "business_calendar_valid_from, business_calendar_valid_until, is_business_day, try_is_business_day, "
    "business_day_roll, try_business_day_roll, business_day_offset, try_business_day_offset, business_day_count, "
    "try_business_day_count, business_in_all, try_business_in_all, business_in_any, try_business_in_any, "
    "dates_is_business_day, dates_business_day_roll, dates_business_day_offset, dates_business_day_count)\n"
)

PRELUDE = f"""\
def b2i(x: bool) -> i64 = if x then 1i64 else 0i64
def flag_or(o: Option[bool]) -> i64 =
  match o with {{
    | Some(x) => b2i(x)
    | None => {ABSENT}i64
  }}
def day_or(o: Option[Date]) -> i64 =
  match o with {{
    | Some(d) => date_epoch_day(d)
    | None => {ABSENT}i64
  }}
def count_or(o: Option[i64]) -> i64 =
  match o with {{
    | Some(n) => n
    | None => {ABSENT}i64
  }}
def day_row(cal: BusinessCalendar, x: i64, offsets: List[i64]) -> List[i64] = {{
  d = date_from_epoch_day(x)
  rolls = map(fn (r: BusinessDayRoll) -> day_or(try_business_day_roll(cal, d, r)), [{", ".join(ref.ROLLS)}])
  moves = flatten(map(fn (n: i64) -> map(fn (s: NonBusinessStart) -> day_or(try_business_day_offset(cal, d, n, s)), [{", ".join(ref.STARTS)}]), offsets))
  concat(concat([flag_or(try_is_business_day(cal, d))], rolls), moves)
}}
def count_row(cal: BusinessCalendar, pair: (i64, i64)) -> i64 = count_or(try_business_day_count(cal, date_from_epoch_day(pair.0), date_from_epoch_day(pair.1)))
def calendar_row(o: Option[BusinessCalendar]) -> List[i64] =
  match o with {{
    | Some(c) => {{
      w = business_calendar_weekmask(c)
      bits = map(fn (b: bool) -> b2i(b), [w.monday, w.tuesday, w.wednesday, w.thursday, w.friday, w.saturday, w.sunday])
      bounds = [date_epoch_day(business_calendar_valid_from(c)), date_epoch_day(business_calendar_valid_until(c))]
      concat(concat(concat([1i64], bits), bounds), map(fn (d: Date) -> date_epoch_day(d), business_calendar_holidays(c)))
    }}
    | None => [0i64]
  }}
def column[n](days: List[i64]) -> Dates[n] = dates_from_epoch_days(to_tensor(days))
def lcg(x: i64) -> i64 = mod(add(mul(x, 1103515245i64), 12345i64), 2147483648i64)
def mix(k: i64, seed: i64) -> i64 = lcg(lcg(add(k, seed)))
def holidays_of(f: i64, u: i64, density: i64, seed: i64) -> List[Date] = {{
  picked = filter(fn (i: i64) -> lt(mod(mix(i, seed), 1000i64), density), range(0i64, add(sub(u, f), 1i64)))
  map(fn (i: i64) -> date_from_epoch_day(add(f, i)), concat(picked, take(picked, 3i64)))
}}
def query_days(f: i64, u: i64, seed: i64, limit: i64, extra: List[i64]) -> List[i64] = concat(map(fn (k: i64) -> add(f, mod(mix(k, seed), add(sub(u, f), 1i64))), range(0i64, limit)), extra)
def query_pairs(days: List[i64], beyond: List[i64], seed: i64, count: i64, fixed: List[(i64, i64)]) -> List[(i64, i64)] = {{
  ends = concat(days, beyond)
  drawn = map(fn (k: i64) -> (index(ends, mod(mix(k, add(seed, 1i64)), len(ends))), index(ends, mod(mix(k, add(seed, 2i64)), len(ends)))), range(0i64, count))
  concat(drawn, fixed)
}}
def present(o: Option[Date]) -> bool =
  match o with {{
    | Some(_) => true
    | None => false
  }}
def counted(o: Option[i64]) -> bool =
  match o with {{
    | Some(_) => true
    | None => false
  }}
def none_listed(seed: i64) -> List[i64] = take([0i64], 0i64)
def roll_column(cal: BusinessCalendar, days: List[i64], r: BusinessDayRoll) -> List[i64] = {{
  good = filter(fn (x: i64) -> present(try_business_day_roll(cal, date_from_epoch_day(x), r)), days)
  if eq(len(good), 0i64) then none_listed(0i64) else to_list(dates_epoch_days(dates_business_day_roll(cal, column(good), r)))
}}
def offset_column(cal: BusinessCalendar, days: List[i64], offsets: List[i64], s: NonBusinessStart) -> List[i64] = {{
  pairs = flatten(map(fn (x: i64) -> map(fn (n: i64) -> (x, n), offsets), days))
  good = filter(fn (p: (i64, i64)) -> present(try_business_day_offset(cal, date_from_epoch_day(p.0), p.1, s)), pairs)
  if eq(len(good), 0i64) then none_listed(0i64) else to_list(dates_epoch_days(dates_business_day_offset(cal, column(map(fn (p: (i64, i64)) -> p.0, good)), to_tensor(map(fn (p: (i64, i64)) -> p.1, good)), s)))
}}
def count_column(cal: BusinessCalendar, pairs: List[(i64, i64)]) -> List[i64] = {{
  good = filter(fn (p: (i64, i64)) -> counted(try_business_day_count(cal, date_from_epoch_day(p.0), date_from_epoch_day(p.1))), pairs)
  if eq(len(good), 0i64) then none_listed(0i64) else to_list(dates_business_day_count(cal, column(map(fn (p: (i64, i64)) -> p.0, good)), column(map(fn (p: (i64, i64)) -> p.1, good))))
}}
def flag_column(cal: BusinessCalendar, days: List[i64]) -> List[i64] = if eq(len(days), 0i64) then none_listed(0i64) else map(fn (b: bool) -> b2i(b), to_list(dates_is_business_day(cal, column(days))))
"""


def lit(n: int) -> str:
    if n == ref.I64_MIN:
        return "sub(-9223372036854775807i64, 1i64)"
    return f"({n}i64)" if n < 0 else f"{n}i64"


def list_lit(values) -> str:
    """A `List[i64]` expression: an empty one spells its element type, and a long
    one concatenates short literals, because `chelis eval` overflows its stack
    on a few hundred elements in one literal (#906)."""
    if not values:
        return "take([0i64], 0i64)"
    chunks = ["[" + ", ".join(lit(v) for v in values[k:k + 64]) + "]" for k in range(0, len(values), 64)]
    expression = chunks[0]
    for chunk in chunks[1:]:
        expression = f"concat({expression}, {chunk})"
    return expression


def weekmask_src(mask) -> str:
    fields = ", ".join(f"{name}: {'true' if bit else 'false'}" for name, bit in zip(ref.WEEKDAYS, mask))
    return f"Weekmask {{ {fields} }}"


def calendar_src(spec: ref.CalendarSpec) -> str:
    if spec.density is None:
        holidays = f"map(fn (d: i64) -> date_from_epoch_day(d), {list_lit(list(spec.holidays))})"
    else:
        holidays = f"holidays_of({lit(spec.valid_from)}, {lit(spec.valid_until)}, {spec.density}i64, {spec.seed}i64)"
    return (f"business_calendar({weekmask_src(spec.weekmask)}, {holidays}, "
            f"date_from_epoch_day({lit(spec.valid_from)}), date_from_epoch_day({lit(spec.valid_until)}))")


def render(values) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def absent(value) -> int:
    return ABSENT if value is None else value


@dataclass(frozen=True)
class Program:
    name: str
    body: str
    expected: dict[str, str] = field(default_factory=dict)
    failure: tuple[str, int | None] | None = None
    lanes: tuple[str, ...] = BOTH


# `chelis eval` interprets every query, so its programs observe a subset of
# each calendar's queries: fewer days and the offsets that reach both ends of
# the horizon and both i64 extremes. Compiled C observes all of them.
EVAL_DAYS = 8
EVAL_PAIRS = 10
C_DAYS = 32
C_PAIRS = 30
C_GROUP = 13


def program_source(body: str) -> str:
    return "module Demo.Main\n" + IMPORTS + PRELUDE + body + "\n"


def pairs_lit(pairs) -> str:
    if not pairs:
        return "take([(0i64, 0i64)], 0i64)"
    return "[" + ", ".join(f"({lit(a)}, {lit(b)})" for a, b in pairs) + "]"


def calendar_program(index: int, spec: ref.CalendarSpec, lane: str) -> Program:
    """One calendar's observations: scalar `try_` forms per query, then the vectorized forms."""
    cal = spec.calendar()
    seed = SEED + 11 * index
    limit = EVAL_DAYS if lane == "eval" else C_DAYS
    pair_count = EVAL_PAIRS if lane == "eval" else C_PAIRS
    queries = ref.calendar_queries(seed, cal, limit, pair_count)
    offsets = list(queries.offsets)
    if lane == "eval":
        budget = len(cal.business_days())
        offsets = [0, 1, -1, 17, budget, ref.I64_MIN]
    days = list(queries.days)
    extra = days[limit:]
    beyond = [d for d in (cal.valid_until + 2,) if d <= ref.MAX_EPOCH_DAY]
    fixed = list(queries.pairs[pair_count:])
    days_src = f"query_days({lit(spec.valid_from)}, {lit(spec.valid_until)}, {seed}i64, {limit}i64, {list_lit(extra)})"
    pairs_src = f"query_pairs(days, {list_lit(beyond)}, {seed}i64, {pair_count}i64, {pairs_lit(fixed)})"
    rows: list[int] = []
    for day in days:
        rows.append(absent(None if not cal.inside(day) else int(cal.business(day))))
        rows += [absent(ref.roll(cal, day, name)) for name in ref.ROLLS]
        for n in offsets:
            rows += [absent(ref.offset(cal, day, n, start)) for start in ref.STARTS]
    rows += [absent(ref.count(cal, a, b)) for a, b in queries.pairs]
    inside = [d for d in days if cal.inside(d)]
    vector_offsets = offsets[:7]
    vectors: list[int] = [int(cal.business(d)) for d in inside]
    for name in ref.ROLLS:
        vectors += [r for r in (ref.roll(cal, d, name) for d in inside) if r is not None]
    for start in ref.STARTS:
        vectors += [o for o in (ref.offset(cal, d, n, start) for d in inside for n in vector_offsets) if o is not None]
    vectors += [c for c in (ref.count(cal, a, b) for a, b in queries.pairs) if c is not None]
    rolls = ", ".join(f"roll_column(cal, inside, {name})" for name in ref.ROLLS)
    starts = ", ".join(f"offset_column(cal, inside, {list_lit(vector_offsets)}, {start})" for start in ref.STARTS)
    lines = [
        f"def observe_{index:03d}(seed: i64) -> List[i64] = {{",
        f"  cal = {calendar_src(spec)}",
        f"  days = {days_src}",
        f"  rows = flatten(map(fn (x: i64) -> day_row(cal, x, {list_lit(offsets)}), days))",
        f"  counts = map(fn (p: (i64, i64)) -> count_row(cal, p), {pairs_src})",
        "  concat(rows, counts)",
        "}",
        f"observed_{index:03d} = observe_{index:03d}(0i64)",
        f"def vectors_{index:03d}(seed: i64) -> List[i64] = {{",
        f"  cal = {calendar_src(spec)}",
        f"  days = {days_src}",
        f"  inside = filter(fn (x: i64) -> and(gte(x, {lit(spec.valid_from)}), lte(x, {lit(spec.valid_until)})), days)",
        f"  flatten([flag_column(cal, inside), {rolls}, {starts}, count_column(cal, {pairs_src})])",
        "}",
        f"vectorized_{index:03d} = vectors_{index:03d}(0i64)",
    ]
    expected = {f"observed_{index:03d}": render(rows), f"vectorized_{index:03d}": render(vectors)}
    return Program(f"calendar_{index:03d}_{lane}", "\n".join(lines), expected, lanes=(lane,))


def combination_program(index: int, a: ref.CalendarSpec, b: ref.CalendarSpec) -> Program:
    def row(cal):
        if cal is None:
            return [0]
        return [1] + [int(x) for x in cal.weekmask] + [cal.valid_from, cal.valid_until] + list(cal.holidays)
    left, right = a.calendar(), b.calendar()
    lines = [
        f"def combined_{index:03d}(seed: i64) -> List[i64] = {{",
        f"  a = {calendar_src(a)}",
        f"  b = {calendar_src(b)}",
        "  flatten([calendar_row(try_business_in_all(a, b)), calendar_row(try_business_in_any(a, b)), "
        "calendar_row(try_business_in_all(b, a)), calendar_row(try_business_in_any(b, a))])",
        "}",
        f"combination_{index:03d} = combined_{index:03d}(0i64)",
    ]
    rows = (row(ref.combine(left, right, True)) + row(ref.combine(left, right, False))
            + row(ref.combine(right, left, True)) + row(ref.combine(right, left, False)))
    return Program(f"combination_{index:03d}", "\n".join(lines), {f"combination_{index:03d}": render(rows)})


def failure_programs() -> list[Program]:
    week = ref.WEEKDAYS_ONLY
    year = ref.CalendarSpec(week, 20_454, 20_818, (20_454, 20_472))  # 2026 with 2026-01-01 and 2026-01-19
    cal = calendar_src(year)
    later = calendar_src(ref.CalendarSpec(week, 20_819, 21_183))
    weekend = calendar_src(ref.CalendarSpec((False,) * 5 + (True, True), 20_454, 20_818))
    no_holidays = f"map(fn (d: i64) -> date_from_epoch_day(d), {list_lit([])})"
    days = [20_456, 20_455, 20_819, 20_454]
    rolled = [20_456, 20_455, 20_454]
    expected_roll = ref.first_failure([ref.roll(year.calendar(), d, "Preceding") for d in rolled])
    cases = [
        ("business_calendar", f"business_calendar({weekmask_src((False,) * 7)}, {no_holidays}, date_from_epoch_day(0i64), date_from_epoch_day(1i64))", None),
        ("business_calendar", f"business_calendar({weekmask_src(week)}, {no_holidays}, date_from_epoch_day(1i64), date_from_epoch_day(0i64))", None),
        ("business_calendar", calendar_src(ref.CalendarSpec(week, 0, 4, (5,))), None),
        ("is_business_day", f"is_business_day({cal}, date_from_epoch_day(20819i64))", None),
        ("business_day_roll", f"business_day_roll({cal}, date_from_epoch_day(20454i64), Preceding)", None),
        ("business_day_roll", f"business_day_roll({cal}, date_from_epoch_day(20453i64), Unadjusted)", None),
        ("business_day_offset", f"business_day_offset({cal}, date_from_epoch_day(20456i64), 1i64, RejectNonBusinessStart)", None),
        ("business_day_offset", f"business_day_offset({cal}, date_from_epoch_day(20818i64), 9223372036854775807i64, RollStartForward)", None),
        ("business_day_count", f"business_day_count({cal}, date_from_epoch_day(20454i64), date_from_epoch_day(20820i64))", None),
        ("business_in_all", f"business_in_all({cal}, {later})", None),
        ("business_in_all", f"business_in_all({cal}, {weekend})", None),
        ("business_in_any", f"business_in_any({cal}, {later})", None),
        ("dates_is_business_day", f"dates_is_business_day({cal}, column({list_lit(days)}))", 2),
        ("dates_business_day_roll", f"dates_business_day_roll({cal}, column({list_lit(rolled)}), Preceding)", expected_roll),
        ("dates_business_day_offset", f"dates_business_day_offset({cal}, column([20455i64, 20456i64]), to_tensor([1i64, 1i64]), RejectNonBusinessStart)", 1),
        ("dates_business_day_count", f"dates_business_day_count({cal}, column([20454i64, 20454i64]), column([20819i64, 20820i64]))", 1),
    ]
    return [Program(f"failure_{index:02d}_{function}", f"failing = {expression}", failure=(function, element))
            for index, (function, expression, element) in enumerate(cases)]


def calendar_programs(specs: list[ref.CalendarSpec]) -> list[Program]:
    programs = []
    for lane in BOTH:
        # Each C program pays for compiling the standard library, so compiled C
        # gets few programs; eval pays per query, so it gets many small ones.
        group = C_GROUP if lane == "c" else 1
        singles = [calendar_program(i, spec, lane) for i, spec in enumerate(specs)]
        for start in range(0, len(singles), group):
            members = singles[start:start + group]
            programs.append(Program(f"calendars_{start:03d}_{lane}", "\n".join(m.body for m in members),
                                    {k: v for m in members for k, v in m.expected.items()}, lanes=(lane,)))
    return programs


def combination_programs(count: int) -> list[Program]:
    rng = ref.random.Random(SEED + 1)
    programs = []
    for index in range(count):
        valid_from = rng.randint(-50_000, 50_000)
        pair = []
        for side in range(2):
            start = valid_from + (rng.randint(-60, 60) if side else 0)
            weekmask = tuple(rng.random() < 0.7 for _ in range(6)) + (True,)
            pair.append(ref.CalendarSpec(weekmask, start, start + rng.choice([30, 90, 200]) - 1, density=100,
                                         seed=SEED + 101 * index + side))
        programs.append(combination_program(index, *pair))
    return programs


def build_programs(random_calendars: int) -> list[Program]:
    specs = ref.build_calendars(SEED, random_calendars)
    return calendar_programs(specs) + combination_programs(max(4, random_calendars // 4)) + failure_programs()


# The failure the canary keeps: a vectorized roll that must name its failing
# element. The calendar programs already reach both i64 extremes through the
# `try_` offsets.
CANARY_FAILURES = ("failure_13_dates_business_day_roll",)


def build_canary_programs() -> list[Program]:
    """The per-pull-request subset of the full corpus.

    Both range-edge calendars on both lanes and the `CANARY_FAILURES`; every
    observation is one the full corpus also makes. Each compiled C program
    builds the standard library, so the canary keeps to two of them.
    """
    failures = {program.name: program for program in failure_programs()}
    missing = [name for name in CANARY_FAILURES if name not in failures]
    if missing:
        raise AssertionError(f"canary failures missing from the full corpus: {missing}")
    return calendar_programs(ref.edge_calendars()[:2]) + [failures[name] for name in CANARY_FAILURES]


@dataclass(frozen=True)
class Toolchain:
    compiler: str
    compile_flags: tuple[str, ...]
    link_flags: tuple[str, ...]


@dataclass
class LaneResult:
    lane: str
    status: int | None
    stdout: str
    stderr: str
    stage: str
    seconds: float = 0.0


class Runner:
    def __init__(self, chelis: Path, reef_home: Path, template: Path, work: Path, toolchain: Toolchain | None, timeout: int):
        self.chelis, self.reef_home, self.template, self.work = chelis, reef_home, template, work
        self.toolchain, self.timeout = toolchain, timeout

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update({"CHELIS_REEF_HOME": str(self.reef_home), "CHELIS_STYLE_GATE_DISABLE": "1", "OMP_NUM_THREADS": "1"})
        return env

    def app(self, program: Program, lane: str) -> Path:
        app = self.work / lane / program.name
        if app.exists():
            shutil.rmtree(app)
        (app / "src").mkdir(parents=True)
        shutil.copy(self.template / "reef.toml", app / "reef.toml")
        (app / "src" / "main.ch").write_text(program_source(program.body), encoding="utf-8")
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
        build = self.run([str(self.chelis), "build", "--emit-c", "src/main.ch", "--target", "c", "--output", "out"], app)
        if build.returncode != 0:
            return LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        link = self.run([self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", "out/main.c",
                         "out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"], app)
        if link.returncode != 0:
            return LaneResult("c", link.returncode, link.stdout, link.stderr, "link")
        done = self.run([str(app / "out" / "case")], app)
        return LaneResult("c", done.returncode, done.stdout, done.stderr, "run")


def parse_bindings(stdout: str) -> tuple[dict[str, str], list[str]]:
    found: dict[str, str] = {}
    stray = []
    for line in stdout.splitlines():
        name, sep, value = line.partition(" = ")
        if sep and name not in found:
            found[name] = value
        else:
            stray.append(line)
    return found, stray


def failure_message(result: LaneResult) -> str | None:
    for line in result.stderr.splitlines():
        text = line.strip()
        if text.startswith("error: "):
            text = text[len("error: "):]
        name, sep, rest = text.partition(": ")
        kind, sep2, _ = rest.partition(": ")
        if sep and sep2 and kind == "domain" and name.replace("_", "").isalnum():
            return text
    return None


def check_program(program: Program, results: list[LaneResult], problems: list[str]) -> int:
    if program.failure is not None:
        function, element = program.failure
        messages = {}
        for result in results:
            message = failure_message(result)
            if result.status == 0 or message is None:
                problems.append(f"{program.name} [{result.lane}/{result.stage}]: expected `{function}: domain: ...`, got "
                                f"status {result.status}; stdout {result.stdout.strip()[:300]!r}; stderr {result.stderr.strip()[:300]!r}")
                continue
            messages[result.lane] = message
            name, _, rest = message.partition(": ")
            _, _, detail = rest.partition(": ")
            if name != function or not detail.strip():
                problems.append(f"{program.name} [{result.lane}]: expected `{function}: domain: <detail>`, got `{message}`")
            if element is not None and not detail.startswith(f"element {element}: "):
                problems.append(f"{program.name} [{result.lane}]: expected element {element}, got `{message}`")
        if len(set(messages.values())) > 1:
            problems.append(f"{program.name}: lanes disagree on the message: {messages}")
        return 1
    for result in results:
        if result.status != 0:
            problems.append(f"{program.name} [{result.lane}/{result.stage}]: status {result.status}; stderr {result.stderr.strip()[:2000]}")
            continue
        printed, stray = parse_bindings(result.stdout)
        stray += [f"{name} = {value[:200]}" for name, value in printed.items() if name not in program.expected]
        if stray:
            problems.append(f"{program.name} [{result.lane}]: unexpected output lines: {stray[:10]}")
        for name, want in program.expected.items():
            got = printed.get(name)
            if got != want:
                problems.append(f"{program.name}.{name} [{result.lane}]: {first_difference(want, got)}")
    return sum(len(want.split(",")) for want in program.expected.values())


def first_difference(want: str, got: str | None) -> str:
    if got is None:
        return "no output line"
    w, g = want.strip("[]").split(", "), got.strip("[]").split(", ")
    for index, (x, y) in enumerate(zip(w, g)):
        if x != y:
            return f"element {index}: expected {x}, printed {y} ({len(w)} expected, {len(g)} printed)"
    return f"{len(w)} elements expected, {len(g)} printed"


def write_app_template(chelis: Path, root: Path) -> Path:
    version = subprocess.run([str(chelis), "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
    template = root / "template"
    template.mkdir(parents=True, exist_ok=True)
    (template / "reef.toml").write_text(
        'schema = "1"\n\n[package]\nname = "business-oracle"\nversion = "0.1.0"\n'
        f'compiler = "={version}"\nmodule_prefix = "Demo"\n\n[dependencies]\nchelis-std = {{ version = "0.4.0" }}\n',
        encoding="utf-8")
    return template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chelis", type=Path, required=True, help="the chelis binary under test")
    parser.add_argument("--reef-home", type=Path, required=True, help="a reef home with chelis-std published")
    parser.add_argument("--toolchain-json", help="strict reference toolchain: {compiler, compile_flags, link_flags}")
    parser.add_argument("--profile", choices=("full", "canary"), required=True)
    parser.add_argument("--lanes", default="eval,c")
    parser.add_argument("--calendars", type=int, default=16, help="full profile: random calendars beyond the edge calendars")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per lane process")
    parser.add_argument("--work", type=Path, help="keep generated programs here instead of a temporary directory")
    args = parser.parse_args(argv)

    lanes = args.lanes.split(",")
    if not lanes or any(lane not in BOTH for lane in lanes):
        parser.error("--lanes is a comma-separated subset of eval,c")
    toolchain = None
    if "c" in lanes:
        if not args.toolchain_json:
            parser.error("the c lane needs --toolchain-json")
        spec = json.loads(args.toolchain_json)
        toolchain = Toolchain(spec["compiler"], tuple(spec["compile_flags"]), tuple(spec["link_flags"]))

    if args.profile == "full":
        corpus, calendars = build_programs(args.calendars), args.calendars + len(ref.edge_calendars())
    else:
        corpus, calendars = build_canary_programs(), 2
    programs = [p for p in corpus if set(p.lanes) & set(lanes)]
    summary = f"{len(programs)} programs over {calendars} calendars"
    print(f"corpus: {summary}", flush=True)
    problems: list[str] = []
    observations = 0
    with tempfile.TemporaryDirectory(prefix="business-oracle-") as scratch:
        work = args.work or Path(scratch)
        runner = Runner(args.chelis.resolve(), args.reef_home.resolve(), write_app_template(args.chelis, work), work,
                        toolchain, args.timeout)

        def run_one(program: Program) -> tuple[Program, list[LaneResult]]:
            results = []
            for lane in (lane for lane in lanes if lane in program.lanes):
                started = time.monotonic()
                try:
                    result = runner.eval_lane(program) if lane == "eval" else runner.c_lane(program)
                except subprocess.TimeoutExpired as error:
                    result = LaneResult(lane, None, "", f"timed out after {error.timeout} s", "timeout")
                result.seconds = time.monotonic() - started
                results.append(result)
            return program, results

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for done, (program, results) in enumerate(pool.map(run_one, programs), start=1):
                observations += check_program(program, results, problems)
                if done % 10 == 0 or done == len(programs):
                    print(f"{done}/{len(programs)} programs, {len(problems)} problems", flush=True)

    for problem in problems[:200]:
        print(f"DISAGREEMENT {problem}")
    if problems:
        print(f"STD DATETIME BUSINESS ORACLE: FAIL ({len(problems)} disagreements; {summary}; lanes {'+'.join(lanes)})")
        return 1
    print(f"{PASS_MARKER} ({observations} observations on lanes {'+'.join(lanes)}; {summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
