#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_ARGUMENTS = ("--self-test", "--merge-bound", "--base")
VALIDATE_ARGUMENTS = ("validate", "--all", "--strict", "--no-interactive")
VALIDATION_FINDING = 1
OPERATIONAL_FAILURE = 2
VALIDATION_TIMEOUT_SECONDS = 300
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class CheckError(RuntimeError):
    pass


@dataclass(frozen=True)
class BaseRevision:
    value: str

    @classmethod
    def parse(cls, value: str) -> BaseRevision:
        if value != "origin/main" and FULL_SHA.fullmatch(value) is None:
            raise CheckError("base must be origin/main or a lowercase full SHA")
        return cls(value)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Invocation:
    openspec: Path
    base: BaseRevision


def parse_invocation(
    arguments: Sequence[str], environment: Mapping[str, str]
) -> Invocation:
    values = tuple(arguments)
    if len(values) != 4 or values[:3] != ACTION_ARGUMENTS:
        raise CheckError(
            "expected fixed action arguments: --self-test --merge-bound --base <revision>"
        )

    executable = environment.get("OPENSPEC_BIN", "")
    if not executable or "\0" in executable:
        raise CheckError("OPENSPEC_BIN must name the action-local executable")

    return Invocation(
        openspec=Path(executable),
        base=BaseRevision.parse(values[3]),
    )


def build_command(invocation: Invocation) -> tuple[str, ...]:
    return (str(invocation.openspec), *VALIDATE_ARGUMENTS)


def run_validation(
    invocation: Invocation,
    runner: Callable[..., object] = subprocess.run,
) -> int:
    try:
        completed = runner(
            build_command(invocation),
            cwd=REPO_ROOT,
            check=False,
            timeout=VALIDATION_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"check_openspec: OpenSpec execution failed: {error}", file=sys.stderr)
        return OPERATIONAL_FAILURE

    returncode = getattr(completed, "returncode", None)
    if returncode == 0:
        return 0
    if returncode == VALIDATION_FINDING:
        return VALIDATION_FINDING

    print(
        f"check_openspec: OpenSpec returned operational exit code {returncode}",
        file=sys.stderr,
    )
    return OPERATIONAL_FAILURE


def main(
    arguments: Sequence[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> int:
    try:
        invocation = parse_invocation(
            sys.argv[1:] if arguments is None else arguments,
            os.environ if environment is None else environment,
        )
    except CheckError as error:
        print(f"check_openspec: {error}", file=sys.stderr)
        return OPERATIONAL_FAILURE
    return run_validation(invocation)


if __name__ == "__main__":
    sys.exit(main())
