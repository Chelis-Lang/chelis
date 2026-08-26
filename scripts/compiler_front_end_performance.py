#!/usr/bin/env python3
"""Authoritative structural and executable oracle for chelis#1205.

Acceptance is one successful run of every focused command followed by the
exact marker ``compiler front-end performance oracle: PASS``. The Rust tests
measure deterministic work rather than wall time, so shared-runner CPU
contention cannot turn the complexity gate into noise.
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUSED_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        sys.executable,
        "-m",
        "unittest",
        "scripts/test_front_end_performance_fixtures.py",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-effects",
        "-E",
        "test(issue_1205_effect_clone_work_is_linear)",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-ir",
        "-E",
        "test(issue_1205_host_lowering_work_is_linear) | test(issue_1205_preflight_preserves_static_to_tensor_literals) | test(issue_1205_preflight_does_not_confuse_shadowed_operands_with_calls) | test(issue_1205_preflight_tracks_callable_shadowing_and_aliases)",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "test",
        "-p",
        "chelis-cli",
        "--test",
        "issue_1205_front_end_performance",
        "--",
        "--ignored",
        "--test-threads=1",
    ),
)


class OracleFailure(RuntimeError):
    """A focused acceptance command failed."""


def command_text(command: Sequence[str]) -> str:
    return shlex.join(command)


def run_oracle(
    commands: Sequence[Sequence[str]] = FOCUSED_COMMANDS,
    *,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> None:
    env = dict(os.environ if environment is None else environment)
    env["CARGO_HUSKY_DONT_INSTALL_HOOKS"] = "1"
    for command in commands:
        rendered = command_text(command)
        print(f"compiler front-end performance oracle: RUN {rendered}", flush=True)
        result = runner(list(command), cwd=REPO_ROOT, env=env, check=False)
        if result.returncode != 0:
            raise OracleFailure(f"`{rendered}` failed with exit {result.returncode}")
    print("compiler front-end performance oracle: PASS", flush=True)


def main() -> int:
    try:
        run_oracle()
    except OracleFailure as error:
        print(f"compiler front-end performance oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
