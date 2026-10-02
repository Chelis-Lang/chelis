#!/usr/bin/env python3
"""Refuse committed copies of the generated chelis-std runtime.

The chelis-std archive and shell a binary embeds are packed from
`packages/chelis-std` by `crates/chelis-std-bundle/build.rs` while the compiler
builds. Nothing generated from them belongs in the repository, so this check
refuses three kinds of tracked file:

- anything under `packages/chelis-std/dist/`, where `chelis reef build
  packages/chelis-std` writes the pair;
- anything under `crates/chelis-std-bundle/dist/`, where the pair used to be
  embedded from;
- a `reef.lock` that records a `chelis-std` dependency with source kind
  `bundled`. Every lock reef writes records the runtime by the hashes of the
  binary that wrote it, so such a lock goes stale with any std edit.

The default mode reads every tracked path and its staged bytes from the index
and exits 1 when it finds one. `--staged --warn` checks only the paths staged
for the next commit and always exits 0; the commit-msg hook runs it that way,
and CI is the refusal.

Usage:

    <managed-python> scripts/check_std_bundle_untracked.py
    <managed-python> scripts/check_std_bundle_untracked.py --staged --warn

Acceptance is exit 0 with the final line ``std bundle tracking: PASS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = (
    "packages/chelis-std/dist/",
    "crates/chelis-std-bundle/dist/",
)
LOCK_FILE_NAME = "reef.lock"
RUNTIME_PACKAGE = "chelis-std"
PRODUCER = "crates/chelis-std-bundle/build.rs"


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str

    def render(self) -> str:
        return f"{self.path}: {self.reason}"


def _git(repo: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
    )
    return completed.stdout


def tracked_paths(repo: Path) -> list[str]:
    """Every path in the index."""
    output = _git(repo, "ls-files", "-z")
    return [path for path in output.decode("utf-8").split("\0") if path]


def staged_paths(repo: Path) -> list[str]:
    """Paths the next commit adds, copies, modifies, or renames to."""
    output = _git(
        repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"
    )
    return [path for path in output.decode("utf-8").split("\0") if path]


def staged_bytes(repo: Path, path: str) -> bytes:
    """The bytes the index holds for `path`."""
    return _git(repo, "show", f":{path}")


def is_lock(path: str) -> bool:
    return path == LOCK_FILE_NAME or path.endswith("/" + LOCK_FILE_NAME)


def lock_violation(path: str, contents: bytes) -> Violation | None:
    """Why a tracked lock may not be committed, or None when it may."""
    try:
        lock = tomllib.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        return Violation(
            path,
            f"cannot be read as a reef lock ({error}), so it cannot be shown "
            "to record no bundled chelis-std",
        )
    dependencies = lock.get("dependencies", [])
    if not isinstance(dependencies, list):
        return None
    for dependency in dependencies:
        if not isinstance(dependency, dict):
            continue
        source = dependency.get("source")
        if (
            dependency.get("name") == RUNTIME_PACKAGE
            and isinstance(source, dict)
            and source.get("kind") == "bundled"
        ):
            return Violation(
                path,
                "records the bundled chelis-std by the hashes of the binary "
                "that wrote it; reef writes this lock again when it is absent",
            )
    return None


def violations(repo: Path, paths: Sequence[str]) -> list[Violation]:
    """The tracked generated files among `paths`, read from the index."""
    found = []
    for path in sorted(paths):
        if any(path.startswith(prefix) for prefix in GENERATED_DIRS):
            found.append(
                Violation(
                    path,
                    f"is generated: {PRODUCER} packs the chelis-std runtime "
                    "while the compiler builds",
                )
            )
        elif is_lock(path):
            violation = lock_violation(path, staged_bytes(repo, path))
            if violation is not None:
                found.append(violation)
    return found


def main(argv: Sequence[str], *, out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--staged",
        action="store_true",
        help="check only the paths staged for the next commit",
    )
    parser.add_argument(
        "--warn",
        action="store_true",
        help="report findings on stderr and exit 0",
    )
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    paths = staged_paths(args.repo) if args.staged else tracked_paths(args.repo)
    found = violations(args.repo, paths)
    if not found:
        print("std bundle tracking: PASS", file=out)
        return 0
    label = "warning" if args.warn else "error"
    print(
        f"std bundle tracking: {label}: generated chelis-std files are tracked.",
        file=err,
    )
    for violation in found:
        print(f"  {violation.render()}", file=err)
    print(
        f"The runtime is built from packages/chelis-std by {PRODUCER}; nothing "
        "generated from it is committed. Untrack each file with "
        "`git rm --cached <path>` (the ignore rules keep it out afterwards).",
        file=err,
    )
    if args.warn:
        return 0
    print("std bundle tracking: FAIL", file=out)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
