#!/usr/bin/env python3
"""Differential oracle for `Std.Datetime.Zone` (chelis#2862): eval and C against Python's `zoneinfo`.

For every real zone under `crates/chelis-cli/tests/fixtures/tzif/zones/`, the
harness loads the file with `zoneinfo.ZoneInfo.from_file` and generates one
Chelis program that reads the same bytes with `read_bytes`, builds the zone
with `time_zone_from_tzif`, and prints four bindings:

- `offsets`: `time_zone_offset_at` at the instants just before and at every
  offset change the file lists, at every change of the footer rule in a set of
  years from 1970 to 9999, and at seeded random instants;
- `resolutions`: for the local readings at the edges and middle of every gap
  and fold found above, `try_zoned_from_local` under each `Disambiguation`,
  as a presence flag and a unix second;
- `texts`: `zoned_to_string` at sampled instants, joined with `|`;
- `round_trips`: whether `zoned_from_text(parse_zoned_text(zoned_to_string(z)),
  tz, policy)` returns `z` under both `UseWrittenOffset` and
  `RejectOffsetMismatch` (the first holds for every `z` by [05-OP-73]; the
  second need not, and is compared at these sampled instants only);
- `utc_texts`: at the same instants, each written as its UTC date and time
  with `Z`, `z` or `-00:00` (RFC 9557's unknown local offset) and an
  elective or critical annotation, `zoned_from_text` under every
  `OffsetConflict` policy, compared with `zoneinfo`'s reading of that instant;
- for `UTC` only, a separate program checks that the file's zone equals
  `time_zone_utc()`.

The expected values come from `zoneinfo` alone: offsets from `astimezone`,
and the policies from PEP 495's `fold` (in a fold, `fold=0` is the earlier
instant; in a gap, `fold=0` reads the local time at the offset before the
transition, which is the later instant). Python's `datetime` covers years 1
through 9999, so every observation lies there. Each lane must print exactly
these bindings and equal the expectation; agreeing with the other lane is not
enough.

Profiles: `full` runs every fixture zone and runs nightly; `canary` runs
`CANARY_ZONES`, a zone with both a gap and a fold and `UTC`, with the
`time_zone_utc()` check, on every pull request.

The supported entry point is `crates/chelis-cli/tests/std_datetime_zone.rs`,
which supplies the freshly built `chelis`, a published `chelis-std`, and the
strict reference toolchain.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import time
import zoneinfo

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "crates/chelis-cli/tests/fixtures/tzif"
PASS_MARKER = "STD DATETIME ZONE ORACLE: PASS"
SEED = 2862
BOTH = ("eval", "c")
POLICIES = ("EarlierInstant", "LaterInstant", "CompatibleInstant", "RejectNonUniqueLocal")
EPOCH = datetime.datetime(1970, 1, 1)
UTC = datetime.timezone.utc
# Instants whose local reading under every offset is a Python datetime.
LOW = int((datetime.datetime(1, 1, 3) - EPOCH).total_seconds())
HIGH = int((datetime.datetime(9999, 12, 29) - EPOCH).total_seconds())
FOOTER_YEARS = (1970, 2026, 2027, 2037, 2038, 2100, 2400, 5000, 9998)
RANDOM_INSTANTS = 40

PRELUDE = """\
def parse_ints(text: string) -> List[i64] = {
  folded = fold(fn (acc: (List[i64], i64, bool, bool), idx: i64) -> if text |> string_slice(idx, 1i64) |> eq(",") then (append(acc.0, if acc.2 then neg(acc.1) else acc.1), 0i64, false, false) else if text |> string_slice(idx, 1i64) |> eq("-") then (acc.0, acc.1, true, acc.3) else (acc.0, acc.1 |> mul(10i64) |> add(sub(char_code(string_slice(text, idx, 1i64)), 48i64)), acc.2, true), ([], 0i64, false, false), range(0i64, string_len(text)))
  if folded.3 then append(folded.0, if folded.2 then neg(folded.1) else folded.1) else folded.0
}
def offset_rows(tz: TimeZone, seconds: List[i64]) -> List[i64] = map(fn (s: i64) -> offset_seconds(time_zone_offset_at(tz, instant_from_unix(s, 0i64))), seconds)
def resolution_row(tz: TimeZone, civil: i64, policy: Disambiguation) -> List[i64] = {
  local = datetime(date_from_epoch_day(floor_div(civil, 86400i64)), time_from_nanosecond_of_day(mul(mod(add(mod(civil, 86400i64), 86400i64), 86400i64), 1000000000i64)))
  match try_zoned_from_local(local, tz, policy) with {
    | Some(z) => [1i64, instant_unix_second(zoned_instant(z))]
    | None => [0i64, 0i64]
  }
}
def resolution_rows(tz: TimeZone, civils: List[i64]) -> List[i64] = fold(fn (acc: List[i64], civil: i64) -> acc |> concat(resolution_row(tz, civil, EarlierInstant)) |> concat(resolution_row(tz, civil, LaterInstant)) |> concat(resolution_row(tz, civil, CompatibleInstant)) |> concat(resolution_row(tz, civil, RejectNonUniqueLocal)), [], civils)
def text_rows(tz: TimeZone, seconds: List[i64]) -> string = fold(fn (acc: string, s: i64) -> if eq(acc, "") then zoned_to_string(zoned(instant_from_unix(s, 0i64), tz)) else acc |> string_concat("|") |> string_concat(zoned_to_string(zoned(instant_from_unix(s, 0i64), tz))), "", seconds)
def utc_text_row(tz: TimeZone, text: string) -> string =
  zoned_to_string(zoned_from_text(parse_zoned_text(text), tz, UseWrittenOffset))
  |> string_concat("|")
  |> string_concat(zoned_to_string(zoned_from_text(parse_zoned_text(text), tz, UseZoneRules)))
  |> string_concat("|")
  |> string_concat(zoned_to_string(zoned_from_text(parse_zoned_text(text), tz, RejectOffsetMismatch)))
def utc_text_rows(tz: TimeZone, texts: List[string]) -> string = fold(fn (acc: string, text: string) -> if eq(acc, "") then utc_text_row(tz, text) else acc |> string_concat("|") |> string_concat(utc_text_row(tz, text)), "", texts)
def round_trip_rows(tz: TimeZone, seconds: List[i64]) -> List[i64] = map(fn (s: i64) -> if and(eq(zoned_from_text(parse_zoned_text(zoned_to_string(zoned(instant_from_unix(s, 0i64), tz))), tz, UseWrittenOffset), zoned(instant_from_unix(s, 0i64), tz)), eq(zoned_from_text(parse_zoned_text(zoned_to_string(zoned(instant_from_unix(s, 0i64), tz))), tz, RejectOffsetMismatch), zoned(instant_from_unix(s, 0i64), tz))) then 1i64 else 0i64, seconds)
"""

IMPORTS = """\
import Std.Datetime (datetime, date_from_epoch_day, time_from_nanosecond_of_day, instant_from_unix, instant_unix_second, offset_seconds)
import Std.Datetime.Zone (TimeZone, Disambiguation, EarlierInstant, LaterInstant, CompatibleInstant, RejectNonUniqueLocal, UseWrittenOffset, UseZoneRules, RejectOffsetMismatch, time_zone_from_tzif, time_zone_utc, time_zone_offset_at, try_zoned_from_local, zoned, zoned_instant, zoned_to_string, parse_zoned_text, zoned_from_text)
"""


def utc_offset(zone: zoneinfo.ZoneInfo, second: int) -> int:
    moment = datetime.datetime.fromtimestamp(second, UTC).astimezone(zone)
    offset = moment.utcoffset()
    assert offset is not None
    return int(offset.total_seconds())


def changes(zone: zoneinfo.ZoneInfo, path: Path) -> list[tuple[int, int, int]]:
    """Every offset change (instant, offset before, offset after) the file
    lists inside Python's range, then every change of the footer rule in
    `FOOTER_YEARS`, found by scanning each year by day and bisecting."""
    found: dict[int, tuple[int, int, int]] = {}
    for second in listed_transitions(path.read_bytes()):
        if LOW < second < HIGH:
            before, after = utc_offset(zone, second - 1), utc_offset(zone, second)
            if before != after:
                found[second] = (second, before, after)
    for year in FOOTER_YEARS:
        start = int((datetime.datetime(year, 1, 1) - EPOCH).total_seconds())
        for day in range(366):
            low, high = start + day * 86400, start + (day + 1) * 86400
            if high >= HIGH or utc_offset(zone, low) == utc_offset(zone, high):
                continue
            while high - low > 1:
                middle = (low + high) // 2
                if utc_offset(zone, middle) == utc_offset(zone, low):
                    low = middle
                else:
                    high = middle
            found.setdefault(high, (high, utc_offset(zone, low), utc_offset(zone, high)))
    return [found[key] for key in sorted(found)]


def listed_transitions(data: bytes) -> list[int]:
    """The version 2+ transition times of a TZif file."""
    def counts(at: int) -> list[int]:
        return [int.from_bytes(data[at + 20 + 4 * k:at + 24 + 4 * k], "big") for k in range(6)]
    isut, isstd, leap, timecnt, typecnt, charcnt = counts(0)
    second = 44 + timecnt * 5 + typecnt * 6 + charcnt + leap * 8 + isstd + isut
    timecnt = counts(second)[3]
    start = second + 44
    return [int.from_bytes(data[start + 8 * k:start + 8 * k + 8], "big", signed=True) for k in range(timecnt)]


def local_readings(change: tuple[int, int, int]) -> list[int]:
    """Civil seconds at the edges and middle of the gap or fold a change
    makes, and one second outside each edge."""
    second, before, after = change
    low, high = second + min(before, after), second + max(before, after)
    return sorted({low - 1, low, (low + high) // 2, high - 1, high})


def expected_resolution(zone: zoneinfo.ZoneInfo, civil: int) -> list[int]:
    naive = EPOCH + datetime.timedelta(seconds=civil)
    first = int(naive.replace(tzinfo=zone, fold=0).timestamp())
    second = int(naive.replace(tzinfo=zone, fold=1).timestamp())
    exists = datetime.datetime.fromtimestamp(first, zone).replace(tzinfo=None) == naive
    if first == second:
        by_policy = {policy: first for policy in POLICIES[:3]}
        unique = True
    elif exists:
        # A fold: fold=0 is the earlier occurrence.
        by_policy = {"EarlierInstant": first, "LaterInstant": second, "CompatibleInstant": first}
        unique = False
    else:
        # A gap: fold=0 reads at the offset before the transition, the later instant.
        by_policy = {"EarlierInstant": second, "LaterInstant": first, "CompatibleInstant": first}
        unique = False
    row = []
    for policy in POLICIES:
        if policy == "RejectNonUniqueLocal":
            row += [1, first] if unique else [0, 0]
        else:
            row += [1, by_policy[policy]]
    return row


def expected_text(zone: zoneinfo.ZoneInfo, name: str, second: int) -> str:
    return datetime.datetime.fromtimestamp(second, UTC).astimezone(zone).isoformat() + f"[{name}]"


def utc_text(name: str, second: int, index: int) -> str:
    """The instant's UTC date and time with an unknown-offset spelling and an
    annotation, varied by `index` over `Z`, `z` and `-00:00` and over
    elective and critical."""
    written = datetime.datetime.fromtimestamp(second, UTC).replace(tzinfo=None).isoformat()
    spelling = ("Z", "z", "-00:00")[index % 3]
    flag = "!" if (index // 3) % 2 else ""
    return f"{written}{spelling}[{flag}{name}]"


def strings_literal(values: list[str]) -> str:
    return "[" + ", ".join(json.dumps(value) for value in values) + "]"


def ints_literal(values: list[int]) -> str:
    return '"' + ",".join(str(value) for value in values) + '"'


@dataclass(frozen=True)
class Program:
    name: str
    source: str
    expected: dict[str, str]
    lanes: tuple[str, ...] = BOTH


def zone_program(name: str) -> Program:
    path = FIXTURE_DIR / f"zones/{name}.tzif"
    with path.open("rb") as handle:
        zone = zoneinfo.ZoneInfo.from_file(handle, key=name)
    found = changes(zone, path)
    rng = random.Random(f"{SEED}:{name}")
    instants = sorted({s for change in found for s in (change[0] - 1, change[0])} | {rng.randrange(LOW, HIGH) for _ in range(RANDOM_INSTANTS)})
    civils = sorted({civil for change in found for civil in local_readings(change)})
    samples = sorted({rng.randrange(LOW, HIGH) for _ in range(RANDOM_INSTANTS)} | {change[0] for change in found[:: max(1, len(found) // 20)]})
    construct = f'time_zone_from_tzif("{name}", read_bytes("{path.as_posix()}"))'
    body = [
        f"offsets = offset_rows({construct}, parse_ints({ints_literal(instants)}))",
        f"resolutions = resolution_rows({construct}, parse_ints({ints_literal(civils)}))",
        f"texts = text_rows({construct}, parse_ints({ints_literal(samples)}))",
        f"round_trips = round_trip_rows({construct}, parse_ints({ints_literal(samples)}))",
        f"utc_texts = utc_text_rows({construct}, {strings_literal([utc_text(name, s, k) for k, s in enumerate(samples)])})",
    ]
    expected = {
        "offsets": render([utc_offset(zone, s) for s in instants]),
        "resolutions": render([value for civil in civils for value in expected_resolution(zone, civil)]),
        "texts": "|".join(expected_text(zone, name, s) for s in samples),
        "round_trips": render([1 for _ in samples]),
        "utc_texts": "|".join(expected_text(zone, name, s) for s in samples for _ in range(3)),
    }
    source = "module Demo.Main\n" + IMPORTS + PRELUDE + "\n".join(body) + "\n"
    return Program(name.replace("/", "_"), source, expected)


def utc_program() -> Program:
    """`time_zone_utc()` equals tzdata's UTC file."""
    path = FIXTURE_DIR / "zones/UTC.tzif"
    body = f'utc_equal = eq(time_zone_from_tzif("UTC", read_bytes("{path.as_posix()}")), time_zone_utc())\n'
    return Program("UTC_nullary", "module Demo.Main\n" + IMPORTS + PRELUDE + body, {"utc_equal": "true"})


# The zones the canary keeps: Asia/Kolkata lists both a gap and a fold among
# few changes, and UTC lists none.
CANARY_ZONES = ("Asia/Kolkata", "UTC")


def canary_zones(zones: list[str]) -> list[str]:
    """`CANARY_ZONES`, each a fixture zone, with a gap and a fold among them."""
    missing = [name for name in CANARY_ZONES if name not in zones]
    if missing:
        raise AssertionError(f"canary zones missing from the fixture: {missing}")
    found = []
    for name in CANARY_ZONES:
        path = FIXTURE_DIR / f"zones/{name}.tzif"
        with path.open("rb") as handle:
            found += changes(zoneinfo.ZoneInfo.from_file(handle, key=name), path)
    if not any(after > before for _, before, after in found) or not any(after < before for _, before, after in found):
        raise AssertionError("the canary zones must include a gap and a fold")
    return list(CANARY_ZONES)


def render(values: list[int]) -> str:
    return "[" + ", ".join(str(value) for value in values) + "]"


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
    def __init__(self, chelis: Path, reef_home: Path, work: Path, toolchain: Toolchain | None, timeout: int):
        self.chelis = chelis
        self.reef_home = reef_home
        self.work = work
        self.toolchain = toolchain
        self.timeout = timeout
        version = subprocess.run([str(chelis), "--version"], capture_output=True, text=True, check=True).stdout.split()[1]
        self.manifest = ('schema = "1"\n\n[package]\nname = "datetime-zone-oracle"\nversion = "0.1.0"\n'
                         f'compiler = "={version}"\nmodule_prefix = "Demo"\n\n[dependencies]\nchelis-std = {{ version = "0.4.0" }}\n')

    def app(self, program: Program, lane: str) -> Path:
        app = self.work / lane / program.name
        if app.exists():
            shutil.rmtree(app)
        (app / "src").mkdir(parents=True)
        (app / "reef.toml").write_text(self.manifest, encoding="utf-8")
        (app / "src" / "main.ch").write_text(program.source, encoding="utf-8")
        return app

    def run(self, argv: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.update({"CHELIS_REEF_HOME": str(self.reef_home), "CHELIS_STYLE_GATE_DISABLE": "1", "OMP_NUM_THREADS": "1"})
        return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True, timeout=self.timeout, check=False)

    def lane(self, program: Program, lane: str) -> LaneResult:
        app = self.app(program, lane)
        if lane == "eval":
            done = self.run([str(self.chelis), "eval", "--file", "src/main.ch"], app)
            return LaneResult("eval", done.returncode, done.stdout, done.stderr, "eval")
        assert self.toolchain is not None
        build = self.run([str(self.chelis), "build", "src/main.ch", "--target", "c", "--emit-c", "--output", "out"], app)
        if build.returncode != 0:
            return LaneResult("c", build.returncode, build.stdout, build.stderr, "build")
        link = self.run([self.toolchain.compiler, *self.toolchain.compile_flags, "-Iout", "out/main.c",
                         "out/libchelis_runtime.a", *self.toolchain.link_flags, "-o", "out/case"], app)
        if link.returncode != 0:
            return LaneResult("c", link.returncode, link.stdout, link.stderr, "link")
        done = self.run([str(app / "out" / "case")], app)
        return LaneResult("c", done.returncode, done.stdout, done.stderr, "run")


def first_difference(expected: str, printed: str) -> str:
    want, got = expected.strip("[]").split(", "), printed.strip("[]").split(", ")
    if expected.startswith("[") and printed.startswith("["):
        for index, (w, g) in enumerate(zip(want, got)):
            if w != g:
                return f"element {index}: expected {w}, printed {g} ({len(want)} expected, {len(got)} printed)"
        if len(want) != len(got):
            return f"{len(want)} elements expected, {len(got)} printed"
    for index, (w, g) in enumerate(zip(expected, printed)):
        if w != g:
            return f"character {index}: expected {expected[max(0, index - 80):index + 80]!r}, printed {printed[max(0, index - 80):index + 80]!r}"
    return f"expected {len(expected)} characters, printed {len(printed)}"


@dataclass
class Report:
    observations: int = 0
    problems: list[str] = field(default_factory=list)


def check(program: Program, result: LaneResult, report: Report) -> None:
    if result.status != 0:
        report.problems.append(f"{program.name} [{result.lane}/{result.stage}]: status {result.status}; stderr {result.stderr.strip()[:2000]}")
        return
    printed: dict[str, str] = {}
    stray = []
    for line in result.stdout.splitlines():
        name, sep, value = line.partition(" = ")
        if sep and name in program.expected and name not in printed:
            printed[name] = value
        else:
            stray.append(line[:200])
    if stray:
        report.problems.append(f"{program.name} [{result.lane}]: unexpected output lines: {stray[:5]}")
    for name, expected in program.expected.items():
        if name not in printed:
            report.problems.append(f"{program.name}.{name} [{result.lane}]: no output line")
        elif printed[name] != expected:
            report.problems.append(f"{program.name}.{name} [{result.lane}]: {first_difference(expected, printed[name])}")
        else:
            report.observations += expected.count(", ") + expected.count("|") + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chelis", type=Path, required=True, help="the chelis binary under test")
    parser.add_argument("--reef-home", type=Path, required=True, help="a reef home with chelis-std published")
    parser.add_argument("--toolchain-json", help="strict reference toolchain: {compiler, compile_flags, link_flags}")
    parser.add_argument("--profile", choices=("full", "canary"), required=True)
    parser.add_argument("--lanes", default="eval,c")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per lane process")
    parser.add_argument("--work", type=Path, help="keep generated programs here instead of a temporary directory")
    parser.add_argument("--only", help="diagnosis: run only the zones whose name contains this text")
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

    zones = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))["zones"]
    if args.profile == "canary":
        zones = canary_zones(zones)
    programs = [zone_program(name) for name in zones if not args.only or args.only in name]
    if not args.only or args.only in "UTC_nullary":
        programs.append(utc_program())
    print(f"corpus: {len(programs)} programs, {sum(p.source.count(',') for p in programs)} inputs", flush=True)
    report = Report()
    with tempfile.TemporaryDirectory(prefix="datetime-zone-oracle-") as scratch:
        work = args.work or Path(scratch)
        runner = Runner(args.chelis.resolve(), args.reef_home.resolve(), work, toolchain, args.timeout)

        def run_one(job: tuple[Program, str]) -> tuple[Program, LaneResult]:
            program, lane = job
            started = time.monotonic()
            try:
                result = runner.lane(program, lane)
            except subprocess.TimeoutExpired as error:
                result = LaneResult(lane, None, "", f"timed out after {error.timeout} s", "timeout")
            result.seconds = time.monotonic() - started
            return program, result

        jobs = [(program, lane) for program in programs for lane in lanes if lane in program.lanes]
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for program, result in pool.map(run_one, jobs):
                check(program, result, report)
                print(f"{program.name} [{result.lane}] {result.seconds:.0f}s", flush=True)

    for problem in report.problems[:100]:
        print(f"DISAGREEMENT {problem}")
    summary = f"{len(zones) if not args.only else len(programs)} zones, {report.observations} observations, lanes {'+'.join(lanes)}"
    if report.problems:
        print(f"STD DATETIME ZONE ORACLE: FAIL ({len(report.problems)} disagreements; {summary})")
        return 1
    print(f"{PASS_MARKER} ({summary})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
