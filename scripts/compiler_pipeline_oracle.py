#!/usr/bin/env python3
"""Authoritative oracle for canonical compiler-pipeline ownership and parity."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUSED_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--test",
        "type_analysis_outcome",
        "-E",
        "all()",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "-E",
        "test(analysis_uses_one_type_session)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "pipeline_contract",
        "--test",
        "fragment_parity",
        "--test",
        "compiled_context",
        "--test",
        "redteam_typecheck_cache",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "-E",
        "test(source_arch)",
    ),
    ("cargo", "test", "-p", "chelis-compiler-api", "--doc"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "stdlib_typecheck_cache_oracle",
        "--test",
        "issue_207_check_exit_code_invariant",
        "--test",
        "check_exits_zero_with_errors_contract",
        "--ignore-default-filter",
    ),
    ("cargo", "nextest", "run", "-p", "chelis-e2e", "--test", "pipeline"),
    ("cargo", "check", "-p", "chelis-e2e", "--bin", "check_snippet"),
)


class OracleFailure(RuntimeError):
    """A canonical pipeline acceptance command failed."""


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def cargo_bin_directory() -> Path | None:
    cargo = shutil.which("cargo")
    if cargo is not None:
        return Path(cargo).resolve().parent
    toolchains = Path.home() / ".rustup" / "toolchains"
    candidates = sorted(toolchains.glob("stable-*/bin/cargo"))
    return candidates[0].parent if candidates else None


def oracle_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cargo_directory = cargo_bin_directory()
    path_entries = []
    if cargo_directory is not None:
        path_entries.append(str(cargo_directory))
    cargo_home_bin = Path.home() / ".cargo" / "bin"
    if cargo_home_bin.is_dir():
        path_entries.append(str(cargo_home_bin))
    if path_entries:
        environment["PATH"] = os.pathsep.join(
            [*path_entries, environment.get("PATH", "")]
        )
    environment.setdefault("PYO3_PYTHON", sys.executable)
    environment.setdefault(
        "CARGO_TARGET_DIR", str(REPO_ROOT / "target" / "agents" / "compiler-pipeline-oracle")
    )
    return environment


def run_oracle(
    commands: Sequence[Sequence[str]] = FOCUSED_COMMANDS,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: dict[str, str] | None = None,
) -> None:
    env = oracle_environment() if environment is None else environment
    for command in commands:
        print(f"+ {command_text(command)}", flush=True)
        completed = runner(command, cwd=REPO_ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise OracleFailure(
                f"command failed with exit {completed.returncode}: {command_text(command)}"
            )
    print("compiler pipeline oracle: PASS", flush=True)


def main() -> int:
    try:
        run_oracle()
    except OracleFailure as error:
        print(f"compiler pipeline oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
