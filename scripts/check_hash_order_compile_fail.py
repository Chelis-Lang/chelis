#!/usr/bin/env python3
"""Verify that deferred-shape stores expose no raw hash iteration API."""

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
    / "hash_order_raw_access"
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
    "no method named `keys`",
    "no method named `iter_mut`",
)


class HashOrderCompileFailure(RuntimeError):
    """The compile-fail fixture did not fail for both private store APIs."""


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
    completed = runner(
        COMMAND,
        cwd=REPO_ROOT,
        env=compile_environment() if environment is None else environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        raise HashOrderCompileFailure("raw deferred-store iteration compiled")
    diagnostic = f"{completed.stdout}\n{completed.stderr}"
    missing = [text for text in REQUIRED_DIAGNOSTICS if text not in diagnostic]
    if missing:
        raise HashOrderCompileFailure(
            f"the compiler output lacks the required text: {', '.join(missing)}"
        )
    print("hash-order deferred-store compile-fail: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except HashOrderCompileFailure as error:
        print(f"hash-order deferred-store compile-fail: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
