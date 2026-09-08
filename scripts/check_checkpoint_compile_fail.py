#!/usr/bin/env python3
"""Verify that diagnostic iteration rejects a raw offset at compile time."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = (
    REPO_ROOT
    / "crates"
    / "chelis-types"
    / "tests"
    / "compile_fail"
    / "checkpoint_raw_offset"
    / "Cargo.toml"
)
COMMAND = (
    "cargo",
    "check",
    "--quiet",
    "--locked",
    "--manifest-path",
    str(MANIFEST),
)
REQUIRED_DIAGNOSTICS = (
    "mismatched types",
    "expected `DiagnosticCheckpoint`",
    "found `usize`",
)


class CheckpointCompileFailure(RuntimeError):
    """The compile-fail fixture did not fail for the required type error."""


def compile_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cargo = shutil.which("cargo")
    path_entries = []
    if cargo is not None:
        path_entries.append(str(Path(cargo).resolve().parent))
    else:
        candidates = sorted((Path.home() / ".rustup" / "toolchains").glob("stable-*/bin"))
        if candidates:
            path_entries.append(str(candidates[0]))
    cargo_home_bin = Path.home() / ".cargo" / "bin"
    if cargo_home_bin.is_dir():
        path_entries.append(str(cargo_home_bin))
    environment["PATH"] = os.pathsep.join(
        [*path_entries, environment.get("PATH", "")]
    )
    return environment


def validate(
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: dict[str, str] | None = None,
) -> None:
    env = compile_environment() if environment is None else environment
    completed = runner(
        COMMAND,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        raise CheckpointCompileFailure("the raw checkpoint offset compiled")
    diagnostic = f"{completed.stdout}\n{completed.stderr}"
    missing = [text for text in REQUIRED_DIAGNOSTICS if text not in diagnostic]
    if missing:
        raise CheckpointCompileFailure(
            f"the compiler output lacks the required text: {', '.join(missing)}"
        )
    print("diagnostic checkpoint compile-fail: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except CheckpointCompileFailure as error:
        print(f"diagnostic checkpoint compile-fail: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
