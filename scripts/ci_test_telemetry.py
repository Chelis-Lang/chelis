#!/usr/bin/env python3
"""Merge nextest JUnit shards and publish exact CI timing telemetry.

Each CI test process writes one JUnit document. This helper makes the shard
boundary explicit: every expected report is named on the command line,
missing or malformed reports fail loudly, and hash partitions may be required
to be disjoint before their XML is handed to the timing diagnostics.

Cross-lane reports may retain overlap because the workspace, dtype,
generalization, and macOS lanes intentionally exercise some of the same test
identities. Their machine report counts both observations and unique tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TelemetryError(RuntimeError):
    """A missing, malformed, ambiguous, or overlapping telemetry input."""


@dataclass(frozen=True)
class Observation:
    shard: str
    test_id: str
    classname: str
    name: str
    seconds: float


@dataclass(frozen=True)
class Shard:
    label: str
    path: Path
    observations: tuple[Observation, ...]

    @property
    def timings(self) -> dict[str, float]:
        return {
            observation.test_id: observation.seconds
            for observation in self.observations
        }


@dataclass(frozen=True)
class Report:
    shards: tuple[Shard, ...]
    observations: tuple[Observation, ...]
    overlaps: dict[str, tuple[str, ...]]

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def unique_test_count(self) -> int:
        return len({observation.test_id for observation in self.observations})

    @property
    def slowest(self) -> tuple[Observation, ...]:
        return tuple(
            sorted(
                self.observations,
                key=lambda observation: (
                    -observation.seconds,
                    observation.test_id,
                    observation.shard,
                ),
            )
        )


def _validate_label(label: str) -> None:
    if _SAFE_LABEL.fullmatch(label) is None:
        raise TelemetryError(
            f"invalid shard label {label!r}; use letters, digits, dot, dash, or underscore"
        )


def parse_shard(label: str, path: Path) -> Shard:
    """Parse one nonempty nextest JUnit report with unique test identities."""
    _validate_label(label)
    if not path.is_file():
        raise TelemetryError(f"JUnit report for {label} not found: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        raise TelemetryError(
            f"malformed JUnit report for {label} at {path}: {error}"
        ) from error
    except OSError as error:
        raise TelemetryError(
            f"cannot read JUnit report for {label} at {path}: {error}"
        ) from error

    cases = list(root.iter("testcase"))
    if not cases:
        raise TelemetryError(f"JUnit report for {label} has no testcases: {path}")
    observations: list[Observation] = []
    seen: set[str] = set()
    for case in cases:
        classname = case.get("classname")
        name = case.get("name")
        raw_seconds = case.get("time")
        if classname is None or name is None or raw_seconds is None:
            raise TelemetryError(
                f"JUnit report for {label} has a testcase without classname, name, or time"
            )
        try:
            seconds = float(raw_seconds)
        except ValueError as error:
            raise TelemetryError(
                f"JUnit report for {label} has nonnumeric time {raw_seconds!r} "
                f"for {classname}::{name}"
            ) from error
        if not math.isfinite(seconds) or seconds < 0.0:
            raise TelemetryError(
                f"JUnit report for {label} has invalid time {raw_seconds!r} "
                f"for {classname}::{name}"
            )
        test_id = f"{classname}::{name}"
        if test_id in seen:
            raise TelemetryError(
                f"JUnit report for {label} contains duplicate test identity {test_id}"
            )
        seen.add(test_id)
        observations.append(
            Observation(label, test_id, classname, name, seconds)
        )
    return Shard(label, path, tuple(observations))


def build_report(
    shards: tuple[Shard, ...], *, require_disjoint: bool
) -> Report:
    """Combine named shards, optionally enforcing an exact partition."""
    if not shards:
        raise TelemetryError("at least one JUnit shard is required")
    labels = [shard.label for shard in shards]
    if len(set(labels)) != len(labels):
        duplicates = sorted(label for label in set(labels) if labels.count(label) > 1)
        raise TelemetryError(f"duplicate shard labels: {', '.join(duplicates)}")

    owners: dict[str, list[str]] = {}
    observations = tuple(
        observation
        for shard in shards
        for observation in shard.observations
    )
    for observation in observations:
        owners.setdefault(observation.test_id, []).append(observation.shard)
    overlaps = {
        test_id: tuple(test_owners)
        for test_id, test_owners in owners.items()
        if len(test_owners) > 1
    }
    if require_disjoint and overlaps:
        sample_id = sorted(overlaps)[0]
        raise TelemetryError(
            f"disjoint shard overlap for {sample_id}: "
            + ", ".join(overlaps[sample_id])
        )
    return Report(shards, observations, overlaps)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_merged_junit(report: Report, path: Path) -> None:
    """Write a timing-check-compatible union for disjoint partitions."""
    if report.overlaps:
        raise TelemetryError("cannot write merged JUnit while test identities overlap")
    root = ET.Element("testsuites")
    for shard in sorted(report.shards, key=lambda item: item.label):
        suite = ET.SubElement(
            root,
            "testsuite",
            {
                "name": shard.label,
                "tests": str(len(shard.observations)),
            },
        )
        for observation in sorted(
            shard.observations, key=lambda item: item.test_id
        ):
            ET.SubElement(
                suite,
                "testcase",
                {
                    "classname": observation.classname,
                    "name": observation.name,
                    # Preserve the parsed float exactly across the merged
                    # artifact. Rounding here can move a value just above the
                    # configured timing threshold onto the allowed boundary.
                    "time": repr(observation.seconds),
                },
            )
    _ensure_parent(path)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _report_payload(report: Report) -> dict[str, object]:
    return {
        "schema_version": 1,
        "observation_count": report.observation_count,
        "unique_test_count": report.unique_test_count,
        "overlap_count": len(report.overlaps),
        "shards": [
            {
                "label": shard.label,
                "test_count": len(shard.observations),
            }
            for shard in sorted(report.shards, key=lambda item: item.label)
        ],
        "overlaps": {
            test_id: sorted(owners)
            for test_id, owners in sorted(report.overlaps.items())
        },
        "slowest": [
            {
                "shard": observation.shard,
                "test_id": observation.test_id,
                "seconds": round(observation.seconds, 6),
            }
            for observation in report.slowest[:50]
        ],
    }


def write_json_report(report: Report, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(
        json.dumps(_report_payload(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown_summary(report: Report) -> str:
    lines = [
        "## CI test timing telemetry",
        "",
        f"- Shards: {len(report.shards)}",
        f"- Test observations: {report.observation_count}",
        f"- Unique test identities: {report.unique_test_count}",
        f"- Cross-shard overlaps: {len(report.overlaps)}",
        "",
        "| Seconds | Shard | Test |",
        "| ---: | --- | --- |",
    ]
    for observation in report.slowest[:20]:
        lines.append(
            f"| {observation.seconds:.3f}s | `{observation.shard}` | "
            f"`{observation.test_id}` |"
        )
    return "\n".join(lines) + "\n"


def write_markdown_summary(report: Report, path: Path) -> None:
    _ensure_parent(path)
    path.write_text(render_markdown_summary(report), encoding="utf-8")


def append_github_summary(report: Report, path: Path) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(render_markdown_summary(report))


def _shard_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard",
        action="append",
        type=_shard_spec,
        required=True,
        metavar="LABEL=PATH",
        help="Expected named nextest JUnit input; repeat once per shard.",
    )
    parser.add_argument(
        "--require-disjoint",
        action="store_true",
        help="Fail if any test identity appears in more than one input.",
    )
    parser.add_argument("--merged-junit", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--github-summary", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        labels = [label for label, _path in args.shard]
        if len(set(labels)) != len(labels):
            duplicates = sorted(
                label for label in set(labels) if labels.count(label) > 1
            )
            raise TelemetryError(
                f"duplicate shard labels: {', '.join(duplicates)}"
            )
        shards = tuple(parse_shard(label, path) for label, path in args.shard)
        report = build_report(shards, require_disjoint=args.require_disjoint)
        if args.merged_junit is not None:
            write_merged_junit(report, args.merged_junit)
        if args.json is not None:
            write_json_report(report, args.json)
        if args.summary is not None:
            write_markdown_summary(report, args.summary)
        if args.github_summary is not None:
            append_github_summary(report, args.github_summary)
    except (OSError, TelemetryError) as error:
        print(f"CI test telemetry: error: {error}", file=sys.stderr)
        return 2
    print(
        "CI test telemetry: PASS "
        f"({report.observation_count} observations, "
        f"{report.unique_test_count} unique tests, "
        f"{len(report.overlaps)} overlaps)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
