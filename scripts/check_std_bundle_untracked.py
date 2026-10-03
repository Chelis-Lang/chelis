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
  `bundled` or `local_registry`. Every lock reef writes records the runtime by
  the hashes of the binary that wrote it, so a `bundled` entry goes stale with
  any std edit; reef never loads chelis-std from a local registry, so it treats
  a `local_registry` entry, at any version, as stale and resolves the runtime
  again.

It reads every tracked path and its bytes from the index and exits 1 when it
finds one. `gate.py --fast` and CI's lint-and-unit stage run it. With
`--tree DIR` it reads every file below DIR instead, by its path relative to
DIR, and a symlink by its target; the pre-commit hook runs it that way on a
copy of the staged files. With `--untrack` it removes each refused file from
the index with `git rm --cached`, leaving the working tree alone; the ignore
rules keep the files out afterwards.

Usage:

    <managed-python> scripts/check_std_bundle_untracked.py
    <managed-python> scripts/check_std_bundle_untracked.py --tree DIR
    <managed-python> scripts/check_std_bundle_untracked.py --untrack

Acceptance is exit 0 with the final line ``std bundle tracking: PASS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Callable, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATED_DIRS = (
    "packages/chelis-std/dist/",
    "crates/chelis-std-bundle/dist/",
)
LOCK_FILE_NAME = "reef.lock"
RUNTIME_PACKAGE = "chelis-std"
# The chelis-std source kinds a committed lock cannot keep current: reef
# serves the runtime only from the binary, and treats a `local_registry`
# chelis-std entry at any version as stale.
RUNTIME_SOURCE_KINDS = {
    "bundled": "records the bundled chelis-std by the hashes of the binary that "
    "wrote it",
    "local_registry": "records chelis-std from a local registry, which reef never "
    "loads it from, so it treats the entry as stale",
}
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


def staged_bytes(repo: Path, path: str) -> bytes:
    """The bytes the index holds for `path`."""
    return _git(repo, "show", f":{path}")


def tree_paths(root: Path) -> list[str]:
    """Every file and symlink below `root`, relative to it."""
    paths = []
    for directory, subdirectories, files in os.walk(root):
        for name in files + subdirectories:
            path = Path(directory) / name
            if path.is_symlink() or path.is_file():
                paths.append(path.relative_to(root).as_posix())
    return paths


def tree_bytes(root: Path, path: str) -> bytes:
    """The bytes below `root` for `path`; a symlink's are its target, as in
    the index."""
    file = root / path
    if file.is_symlink():
        return os.readlink(file).encode("utf-8")
    return file.read_bytes()


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
        if dependency.get("name") != RUNTIME_PACKAGE or not isinstance(source, dict):
            continue
        description = RUNTIME_SOURCE_KINDS.get(source.get("kind"))
        if description is not None:
            return Violation(
                path, f"{description}; reef writes this lock again when it is absent"
            )
    return None


def violations(paths: Sequence[str], read: Callable[[str], bytes]) -> list[Violation]:
    """The generated files among `paths`, whose bytes `read` returns."""
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
            violation = lock_violation(path, read(path))
            if violation is not None:
                found.append(violation)
    return found


def main(argv: Sequence[str], *, out: TextIO = sys.stdout, err: TextIO = sys.stderr) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    source.add_argument(
        "--tree",
        type=Path,
        metavar="DIR",
        help="check every file below DIR, as committed, instead of the index",
    )
    parser.add_argument(
        "--untrack",
        action="store_true",
        help="remove every refused file from the index with git rm --cached",
    )
    args = parser.parse_args(argv)
    if args.untrack and args.tree is not None:
        parser.error("--untrack acts on the index; it cannot take --tree")

    if args.tree is not None:
        if not args.tree.is_dir():
            parser.error(f"--tree {args.tree} is not a directory")
        root = args.tree
        found = violations(tree_paths(root), lambda path: tree_bytes(root, path))
    else:
        repo = args.repo
        found = violations(tracked_paths(repo), lambda path: staged_bytes(repo, path))
    if found and args.untrack:
        paths = [violation.path for violation in found]
        _git(args.repo, "rm", "--cached", "--quiet", "--", *paths)
        for path in paths:
            print(f"std bundle tracking: untracked {path}", file=out)
        found = violations(
            tracked_paths(args.repo), lambda path: staged_bytes(args.repo, path)
        )
    if not found:
        print("std bundle tracking: PASS", file=out)
        return 0
    print(
        "std bundle tracking: error: generated chelis-std files are tracked.",
        file=err,
    )
    for violation in found:
        print(f"  {violation.render()}", file=err)
    print(
        f"The runtime is built from packages/chelis-std by {PRODUCER}; nothing "
        "generated from it is committed. `scripts/check_std_bundle_untracked.py "
        "--untrack` removes these files from the index (`git rm --cached`), and "
        "the ignore rules keep them out afterwards.",
        file=err,
    )
    print("std bundle tracking: FAIL", file=out)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
