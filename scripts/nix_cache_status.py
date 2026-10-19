#!/usr/bin/env python3
"""Report whether the chelis-lang cache holds the Nix packages of this revision.

The first job of `nix-packages.yml` runs this script. It evaluates the output
paths of `packages.<system>.chelis`, `chelis-runtime`, `chelis-unwrapped`, and
`chelisup` for each supported system. Evaluation disables import from derivation, so one host gives
the paths of every system without a builder for that system.

Then the script asks the cache for each path with
`nix path-info --store https://chelis-lang.cachix.org <path>`. A path that the
cache does not hold is missing. Any other failure of that command is an error,
because the script cannot know the state of the path.

With `--github-output PATH`, the script appends `complete=true` or
`complete=false` to that file. On a scheduled run, the native jobs skip when
the value is `true`, because an earlier run built and pushed the same
packages. Manual dispatches and releases run the native jobs in all cases.

The script uses only the Python standard library, so a hosted runner can run
it with its own `python3` before any project environment exists.

Exit codes: 0 the report is complete (the cache holds all paths or some are
missing), 1 an evaluation or cache query failed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

# The packages that `scripts/publish_nix_cache.py` pushes for each system.
from publish_nix_cache import CACHE_NAME, PACKAGES

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_URL = f"https://{CACHE_NAME}.cachix.org"
# The text that `nix path-info` prints when the store does not hold a path.
MISSING_MARKER = "is not valid"
STORE_PREFIX = "/nix/store/"


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str]], Completed]


class StatusError(RuntimeError):
    """A Nix command failed, or its output had an unknown shape."""


def run_command(argv: Sequence[str]) -> Completed:
    completed = subprocess.run(
        list(argv),
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return Completed(completed.returncode, completed.stdout, completed.stderr)


def package_expression() -> str:
    names = " ".join(f'"{name}"' for name in PACKAGES)
    return (
        "packages: builtins.mapAttrs "
        f"(system: set: map (name: set.${{name}}.outPath) [ {names} ]) packages"
    )


def package_paths(run: Runner) -> dict[str, list[str]]:
    """The output paths of the four packages, by supported system."""
    argv = [
        "nix",
        "eval",
        "--json",
        "--option",
        "allow-import-from-derivation",
        "false",
        ".#packages",
        "--apply",
        package_expression(),
    ]
    completed = run(argv)
    if completed.returncode != 0:
        raise StatusError(
            f"`{' '.join(argv[:7])}` failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise StatusError(f"package evaluation printed no JSON: {error}") from error
    if not isinstance(value, dict) or not value:
        raise StatusError("package evaluation gave no supported system")
    paths: dict[str, list[str]] = {}
    for system, outputs in sorted(value.items()):
        if (
            not isinstance(outputs, list)
            or len(outputs) != len(PACKAGES)
            or not all(
                isinstance(path, str) and path.startswith(STORE_PREFIX)
                for path in outputs
            )
        ):
            raise StatusError(f"package evaluation gave unknown paths for {system}")
        paths[system] = outputs
    return paths


def cache_holds(path: str, run: Runner) -> bool:
    argv = ["nix", "path-info", "--store", CACHE_URL, path]
    completed = run(argv)
    if completed.returncode == 0:
        return True
    if MISSING_MARKER in completed.stderr:
        return False
    raise StatusError(
        f"`{' '.join(argv)}` failed with exit code {completed.returncode}: "
        f"{completed.stderr.strip()}"
    )


def main(argv: Sequence[str] | None = None, run: Runner = run_command) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--github-output",
        type=Path,
        help="append complete=true or complete=false to this file",
    )
    args = parser.parse_args(argv)
    try:
        paths = package_paths(run)
        missing = [
            (system, path)
            for system, outputs in paths.items()
            for path in outputs
            if not cache_holds(path, run)
        ]
    except StatusError as error:
        print(f"nix_cache_status: {error}", file=sys.stderr)
        return 1
    total = sum(len(outputs) for outputs in paths.values())
    for system, path in missing:
        print(f"nix_cache_status: {CACHE_URL} does not hold {path} ({system})")
    complete = not missing
    print(
        f"nix_cache_status: {CACHE_URL} holds {total - len(missing)} of {total} "
        f"package path(s) for {', '.join(paths)}"
    )
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"complete={'true' if complete else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
