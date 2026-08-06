#!/usr/bin/env python3
"""Regenerate or freshness-check the committed Cargo.nix crate2nix graph.

The Nix package path imports a committed ``Cargo.nix`` instead of generating the
graph through import-from-derivation. This worker regenerates that file with the
pinned crate2nix and, in ``--check`` mode, fails when the committed bytes drift
from a fresh generation.

It assumes ``crate2nix`` and ``cargo`` are already on ``PATH``; the
``regenerate-crate2nix`` flake app supplies both from the pinned crate2nix input
and the pinned Rust toolchain, so ``nix run .#regenerate-crate2nix`` is
self-contained.
"""

from __future__ import annotations

import argparse
import difflib
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

# The one canonical generation command. It must stay identical to the command
# recorded in nix/packages.nix (the committed-graph comment) and to the feature
# selection the freshness gate documents. crate2nix embeds this command in the
# Cargo.nix banner, so a change here changes the committed bytes.
CANONICAL_ARGS: tuple[str, ...] = (
    "generate",
    "--no-default-features",
    "--features",
    "chelis-cli/smt",
)

CARGO_NIX = "Cargo.nix"

Runner = Callable[[Sequence[str], Path], "subprocess.CompletedProcess[bytes]"]


def _run(argv: Sequence[str], cwd: Path) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(list(argv), cwd=cwd, check=False)


def regenerate(repo_root: Path, run: Runner = _run) -> None:
    """Overwrite ``repo_root/Cargo.nix`` with a fresh crate2nix graph."""
    completed = run(["crate2nix", *CANONICAL_ARGS], repo_root)
    if completed.returncode != 0:
        raise SystemExit(
            f"crate2nix {' '.join(CANONICAL_ARGS)} failed with "
            f"exit code {completed.returncode}"
        )


def check(repo_root: Path, run: Runner = _run) -> int:
    """Return 0 when the committed Cargo.nix matches a fresh generation.

    Regenerates in place, compares, and always restores the committed bytes so
    the working tree is left unchanged whether or not it drifted.
    """
    target = repo_root / CARGO_NIX
    if not target.is_file():
        print(f"error: {CARGO_NIX} is not committed at {target}", file=sys.stderr)
        return 1
    committed = target.read_bytes()
    try:
        regenerate(repo_root, run)
        fresh = target.read_bytes()
    finally:
        target.write_bytes(committed)
    if fresh == committed:
        return 0
    committed_lines = committed.decode("utf-8", "replace").splitlines(keepends=True)
    fresh_lines = fresh.decode("utf-8", "replace").splitlines(keepends=True)
    diff = difflib.unified_diff(
        committed_lines,
        fresh_lines,
        fromfile=f"committed/{CARGO_NIX}",
        tofile=f"regenerated/{CARGO_NIX}",
        n=1,
    )
    sys.stderr.write(
        f"error: {CARGO_NIX} is stale. Run `nix run .#regenerate-crate2nix` and "
        "commit the result.\n"
    )
    for line in diff:
        sys.stderr.write(line if line.endswith("\n") else line + "\n")
    return 1


def main(argv: Sequence[str] | None = None, run: Runner = _run) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed Cargo.nix drifts, leaving the tree unchanged",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="workspace root that owns Cargo.toml, Cargo.lock, and Cargo.nix",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    if not (repo_root / "Cargo.toml").is_file():
        print(f"error: no Cargo.toml under {repo_root}", file=sys.stderr)
        return 2
    if args.check:
        return check(repo_root, run)
    regenerate(repo_root, run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
