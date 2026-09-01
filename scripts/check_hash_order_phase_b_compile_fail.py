#!/usr/bin/env python3
"""Lock the Phase B disallowed-type and order-escape compile failures."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "crates" / "chelis-unord" / "tests" / "compile_fail"


@dataclass(frozen=True)
class Fixture:
    name: str
    command: tuple[str, ...]
    required_diagnostics: tuple[str, ...]


FIXTURES = (
    Fixture(
        "disallowed hash type spellings",
        (
            "cargo",
            "clippy",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(FIXTURE_ROOT / "disallowed_hash_types" / "Cargo.toml"),
            "--",
            "-D",
            "warnings",
        ),
        (
            "use of a disallowed type `std::collections::HashMap`",
            "use of a disallowed type `std::collections::HashSet`",
            "type ForbiddenAlias",
            "collections_alias::HashMap",
            "RenamedMap",
            "hash_map::HashMap",
        ),
    ),
    Fixture(
        "order escape API",
        (
            "cargo",
            "check",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(FIXTURE_ROOT / "order_escape" / "Cargo.toml"),
            "--bin",
            "hash-order-escape-fixture",
        ),
        (
            "no method named `iter`",
            "no method named `keys`",
            "no method named `values`",
            "is not an iterator",
        ),
    ),
    Fixture(
        "callback and projection order escape API",
        (
            "cargo",
            "check",
            "--quiet",
            "--locked",
            "--manifest-path",
            str(FIXTURE_ROOT / "order_escape" / "Cargo.toml"),
            "--bin",
            "fn_mut",
        ),
        (
            "no method named `all`",
            "no method named `any`",
            "no method named `for_each_mut`",
            "no method named `retain`",
            "no method named `map_values`",
            "no method named `to_sorted_by_key`",
            "no method named `into_sorted_by_key`",
        ),
    ),
)


class PhaseBCompileFailure(RuntimeError):
    """A Phase B compile-fail fixture compiled or failed for the wrong reason."""


def compile_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cargo = shutil.which("cargo")
    path_entries = []
    if cargo is not None:
        path_entries.append(str(Path(cargo).resolve().parent))
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
    fixtures: Sequence[Fixture] = FIXTURES,
    environment: dict[str, str] | None = None,
) -> None:
    for fixture in fixtures:
        completed = runner(
            fixture.command,
            cwd=REPO_ROOT,
            env=compile_environment() if environment is None else environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            raise PhaseBCompileFailure(f"{fixture.name} unexpectedly compiled")
        diagnostic = f"{completed.stdout}\n{completed.stderr}"
        missing = [
            text for text in fixture.required_diagnostics if text not in diagnostic
        ]
        if missing:
            raise PhaseBCompileFailure(
                f"{fixture.name} lacks diagnostics: {', '.join(missing)}"
            )
    print("hash-order Phase B compile-fail: PASS", flush=True)


def main() -> int:
    try:
        validate()
    except PhaseBCompileFailure as error:
        print(f"hash-order Phase B compile-fail: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
