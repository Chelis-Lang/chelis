#!/usr/bin/env python3
"""Report whether the cache holds this revision's packages and activated shell.

The native cache-status jobs of `nix-packages.yml` run this script. It evaluates
`chelis`, `chelis-runtime`, `chelis-unwrapped`, and `chelisup` for every supported
system, plus the activated base shell for its native system. Publication and
this report share the shell resolver. Evaluation disables import from derivation.

Then the script asks the cache for each path with
`nix path-info --store https://chelis-lang.cachix.org <path>`. A path that the
cache does not hold is missing. Any other failure of that command is an error,
because the script cannot know the state of the path.

With `--github-output PATH`, the script appends `complete=true` or
`complete=false` to that file. On a scheduled run, the native jobs skip when
the value is `true` on every supported system. Manual dispatches and releases
run the native jobs in all cases.

The script uses only the Python standard library, Nix, and the pinned Devenv
binary. It evaluates the shell without activation or source builds.

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

# Publication and this report must resolve the same activated shell.
from publish_nix_cache import (
    CACHE_NAME, PACKAGES, PublishError, activated_shell_path, current_system,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_URL = f"https://{CACHE_NAME}.cachix.org"
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
    if f"error: path '{path}' is not valid" in completed.stderr:
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

    def checked_run(command: Sequence[str], stdin: str | None = None) -> str:
        completed = run(command)
        if completed.returncode:
            raise StatusError(
                f"`{' '.join(command)}` failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip()}"
            )
        return completed.stdout

    try:
        paths = package_paths(run)
        system = current_system(checked_run)
        if system not in paths:
            raise StatusError(f"the native system {system} has no package outputs")
        paths[system].append(activated_shell_path(system, checked_run))
        missing = [
            (system, path)
            for system, outputs in paths.items()
            for path in outputs
            if not cache_holds(path, run)
        ]
    except (StatusError, PublishError) as error:
        print(f"nix_cache_status: {error}", file=sys.stderr)
        return 1
    total = sum(len(outputs) for outputs in paths.values())
    for system, path in missing:
        print(f"nix_cache_status: {CACHE_URL} does not hold {path} ({system})")
    complete = not missing
    print(
        f"nix_cache_status: {CACHE_URL} holds {total - len(missing)} of {total} "
        f"package and shell path(s) for {', '.join(paths)}"
    )
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"complete={'true' if complete else 'false'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
