#!/usr/bin/env python3
"""Measure Devenv reload, activation, task, and persistent-payload costs."""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = "CHELIS_DEVENV_BENCHMARK_JSON="


@dataclass(frozen=True)
class Measurement:
    wall_s: float
    user_s: float
    system_s: float


def summarize(measurements: list[Measurement]) -> dict[str, float]:
    if not measurements:
        raise ValueError("at least one measurement is required")
    return {
        "wall_median_s": statistics.median(item.wall_s for item in measurements),
        "user_median_s": statistics.median(item.user_s for item in measurements),
        "system_median_s": statistics.median(item.system_s for item in measurements),
    }


def normalize_payload(payload: list[str]) -> list[str]:
    """Accept the conventional optional ``--`` before the payload command."""

    if payload[:1] == ["--"]:
        return payload[1:]
    return payload


def measure(command: list[str], environment: dict[str, str] | None = None) -> Measurement:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.monotonic()
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    wall = time.monotonic() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return Measurement(
        wall_s=wall,
        user_s=after.ru_utime - before.ru_utime,
        system_s=after.ru_stime - before.ru_stime,
    )


def inside_session(samples: int, payload: list[str]) -> int:
    measurements = [measure(payload) for _ in range(samples)]
    print(SENTINEL + json.dumps([asdict(item) for item in measurements]))
    return 0


def repeated(
    samples: int,
    command: list[str],
    environment: dict[str, str] | None = None,
) -> list[Measurement]:
    return [measure(command, environment) for _ in range(samples)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--inside-session", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("payload", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    payload = normalize_payload(arguments.payload) or [sys.executable, "-c", "pass"]
    if arguments.inside_session:
        return inside_session(arguments.samples, payload)
    if arguments.samples < 3:
        parser.error("--samples must be at least 3")

    base = ["devenv", "shell"]
    ordinary = repeated(arguments.samples, [*base, "--", *payload])
    no_reload = repeated(
        arguments.samples,
        [*base, "--no-reload", "--", *payload],
    )
    skip_environment = dict(os.environ)
    skip_environment["DEVENV_SKIP_TASKS"] = "1"
    activation = repeated(
        arguments.samples,
        [*base, "--no-reload", "--", *payload],
        skip_environment,
    )

    persistent_command = [
        *base,
        "--no-reload",
        "--",
        ".devenv/state/venv/bin/python",
        str(Path(__file__).relative_to(REPO_ROOT)),
        "--inside-session",
        "--samples",
        str(arguments.samples),
        *payload,
    ]
    completed = subprocess.run(
        persistent_command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    sentinel_line = next(
        line for line in completed.stdout.splitlines() if line.startswith(SENTINEL)
    )
    persistent = [
        Measurement(**item)
        for item in json.loads(sentinel_line.removeprefix(SENTINEL))
    ]

    summaries = {
        "ordinary": summarize(ordinary),
        "no_reload": summarize(no_reload),
        "skip_tasks_no_reload": summarize(activation),
        "persistent_payload": summarize(persistent),
    }
    wall = {name: values["wall_median_s"] for name, values in summaries.items()}
    decomposition = {
        "reload_and_evaluation_s": max(0.0, wall["ordinary"] - wall["no_reload"]),
        "enter_shell_tasks_s": max(
            0.0, wall["no_reload"] - wall["skip_tasks_no_reload"]
        ),
        "activation_s": max(
            0.0, wall["skip_tasks_no_reload"] - wall["persistent_payload"]
        ),
        "payload_s": wall["persistent_payload"],
    }
    print(json.dumps({"samples": summaries, "decomposition": decomposition}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
