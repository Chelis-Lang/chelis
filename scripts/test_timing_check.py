#!/usr/bin/env python3
"""Test-timing budget check.

Parses the JUnit XML that `cargo nextest run --profile ci` writes
(`target/nextest/ci/junit.xml`) and flags integration tests that have
grown too slow:

  - a test that is NEW relative to the committed baseline AND runs
    longer than `absolute_ceiling` seconds, or
  - a test that REGRESSED past `tolerance` x its baseline time.

Thresholds are config, never hardcoded:

  - `scripts/test_timing_config.json` holds `tolerance` (a multiplier)
    and `absolute_ceiling` (seconds).
  - `scripts/test_timing_baseline.json` maps `binary::test` -> seconds.

The baseline is hand-curated and explicitly regenerated, NOT
auto-updated on every merge: auto-regen would launder a real
regression into the baseline. Regeneration is one command:

    python3 scripts/test_timing_check.py --update-baseline

CI runs this as an informational, non-failing step today (the CI step
uses `continue-on-error: true`); it is promotable to blocking later.

Usage:
    python3 scripts/test_timing_check.py
    python3 scripts/test_timing_check.py --junit path/to/junit.xml
    python3 scripts/test_timing_check.py --update-baseline

Exit codes:
    0  no test over budget (or --update-baseline succeeded)
    1  one or more tests over budget
    2  usage / IO error (missing or malformed JUnit XML, bad config)
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JUNIT = REPO_ROOT / "target" / "nextest" / "ci" / "junit.xml"
CONFIG_PATH = REPO_ROOT / "scripts" / "test_timing_config.json"
BASELINE_PATH = REPO_ROOT / "scripts" / "test_timing_baseline.json"


class TimingError(Exception):
    """Raised for usage / IO / parse errors; mapped to exit code 2."""


def load_config(path: Path = CONFIG_PATH) -> tuple[float, float]:
    """Return `(tolerance, absolute_ceiling)` from the config JSON."""
    if not path.is_file():
        raise TimingError(f"missing timing config: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TimingError(f"malformed timing config {path}: {exc}") from exc
    try:
        tolerance = float(data["tolerance"])
        absolute_ceiling = float(data["absolute_ceiling"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TimingError(
            f"timing config {path} must have numeric `tolerance` and "
            f"`absolute_ceiling`: {exc}"
        ) from exc
    if tolerance < 1.0:
        raise TimingError(
            f"timing config `tolerance` must be >= 1.0, got {tolerance}"
        )
    if absolute_ceiling <= 0.0:
        raise TimingError(
            f"timing config `absolute_ceiling` must be > 0, got "
            f"{absolute_ceiling}"
        )
    return tolerance, absolute_ceiling


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, float]:
    """Return the `binary::test -> seconds` baseline map."""
    if not path.is_file():
        raise TimingError(f"missing timing baseline: {path}")
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise TimingError(f"malformed timing baseline {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TimingError(
            f"timing baseline {path} must be a JSON object of "
            f"`binary::test` -> seconds"
        )
    out: dict[str, float] = {}
    for key, value in data.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise TimingError(
                f"timing baseline {path}: entry {key!r} is not numeric: "
                f"{exc}"
            ) from exc
    return out


def parse_junit(path: Path) -> dict[str, float]:
    """Parse a nextest JUnit XML file into a `binary::test -> seconds`
    map.

    nextest emits `<testcase classname="<binary>" name="<test>"
    time="<seconds>">`. The composite key `classname::name` matches the
    baseline key shape.
    """
    if not path.is_file():
        raise TimingError(
            f"JUnit XML not found: {path}. Run `cargo nextest run "
            f"--workspace --profile ci` (or `python3 scripts/gate.py "
            f"integration`) first."
        )
    try:
        tree = ET.parse(path)
    except ET.ParseError as exc:
        raise TimingError(f"malformed JUnit XML {path}: {exc}") from exc
    root = tree.getroot()
    timings: dict[str, float] = {}
    testcases = list(root.iter("testcase"))
    if not testcases:
        raise TimingError(
            f"JUnit XML {path} has no <testcase> elements; expected "
            f"nextest per-test timing output"
        )
    for case in testcases:
        classname = case.get("classname")
        name = case.get("name")
        time_attr = case.get("time")
        if classname is None or name is None or time_attr is None:
            raise TimingError(
                f"JUnit XML {path}: a <testcase> is missing "
                f"classname/name/time attributes"
            )
        try:
            seconds = float(time_attr)
        except ValueError as exc:
            raise TimingError(
                f"JUnit XML {path}: <testcase {classname}::{name}> has "
                f"non-numeric time={time_attr!r}: {exc}"
            ) from exc
        key = f"{classname}::{name}"
        # If a test name collides (parameterized reruns), keep the
        # slowest observation; the budget cares about worst case.
        timings[key] = max(timings.get(key, 0.0), seconds)
    return timings


class Flag:
    """A single over-budget finding."""

    NEW_OVER_CEILING = "new-over-absolute-ceiling"
    REGRESSED = "regressed-past-tolerance"

    def __init__(self, key: str, kind: str, observed: float, limit: float):
        self.key = key
        self.kind = kind
        self.observed = observed
        self.limit = limit

    def render(self) -> str:
        if self.kind == Flag.NEW_OVER_CEILING:
            return (
                f"  {self.key}: {self.observed:.2f}s -- NEW test over the "
                f"{self.limit:.2f}s absolute ceiling"
            )
        return (
            f"  {self.key}: {self.observed:.2f}s -- regressed past "
            f"{self.limit:.2f}s budget (tolerance x baseline)"
        )


def evaluate(
    timings: dict[str, float],
    baseline: dict[str, float],
    tolerance: float,
    absolute_ceiling: float,
) -> list[Flag]:
    """Return the list of over-budget findings, sorted slowest first."""
    flags: list[Flag] = []
    for key, observed in timings.items():
        if key in baseline:
            budget = baseline[key] * tolerance
            if observed > budget:
                flags.append(Flag(key, Flag.REGRESSED, observed, budget))
        else:
            # New test (not in baseline): only flag if it is also over
            # the absolute ceiling. A fast new test is fine and gets
            # picked up at the next explicit baseline regeneration.
            if observed > absolute_ceiling:
                flags.append(
                    Flag(key, Flag.NEW_OVER_CEILING, observed, absolute_ceiling)
                )
    flags.sort(key=lambda f: f.observed, reverse=True)
    return flags


def print_report(
    flags: list[Flag], timings: dict[str, float], tolerance: float
) -> None:
    total = len(timings)
    if not flags:
        print(
            f"test-timing budget: OK ({total} tests checked, "
            f"tolerance {tolerance}x, none over budget)"
        )
        return
    print(
        f"test-timing budget: {len(flags)} test(s) over budget "
        f"({total} checked, tolerance {tolerance}x):"
    )
    for flag in flags:
        print(flag.render())
    print(
        "If a regression is intentional (e.g. the typecheck-cache "
        "workstream shifts the baseline), regenerate explicitly: "
        "python3 scripts/test_timing_check.py --update-baseline"
    )


def update_baseline(
    junit_path: Path, baseline_path: Path = BASELINE_PATH
) -> int:
    """Regenerate the baseline file from a fresh JUnit run.

    Hand-curated / explicitly-invoked: this is never called from CI on
    merge. Auto-regen would launder regressions into the baseline.
    """
    timings = parse_junit(junit_path)
    ordered = {key: round(timings[key], 3) for key in sorted(timings)}
    baseline_path.write_text(json.dumps(ordered, indent=2) + "\n")
    try:
        shown = baseline_path.relative_to(REPO_ROOT)
    except ValueError:
        shown = baseline_path
    print(
        f"test-timing baseline regenerated: {len(ordered)} tests written "
        f"to {shown}"
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Flag integration tests that regressed past their committed "
            "timing budget."
        ),
    )
    p.add_argument(
        "--junit",
        type=Path,
        default=DEFAULT_JUNIT,
        help=(
            "Path to the nextest JUnit XML. Default: "
            "target/nextest/ci/junit.xml"
        ),
    )
    p.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "Regenerate scripts/test_timing_baseline.json from the JUnit "
            "XML and exit. The baseline is hand-curated; this is the one "
            "documented regeneration command."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        if args.update_baseline:
            return update_baseline(args.junit)
        tolerance, absolute_ceiling = load_config()
        baseline = load_baseline()
        timings = parse_junit(args.junit)
    except TimingError as exc:
        print(f"test-timing budget: error: {exc}", file=sys.stderr)
        return 2
    flags = evaluate(timings, baseline, tolerance, absolute_ceiling)
    print_report(flags, timings, tolerance)
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
