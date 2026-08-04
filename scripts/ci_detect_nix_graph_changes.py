#!/usr/bin/env python3
"""Decide whether a change requires exact Cargo.nix regeneration."""

from __future__ import annotations

import os
import sys
from pathlib import PurePosixPath


GRAPH_EXACT_PATHS = frozenset(
    {
        ".cargo/config",
        ".cargo/config.toml",
        "Cargo.lock",
        "Cargo.nix",
        "crate-hashes.json",
        "crate2nix.json",
        "devenv.lock",
        "devenv.yaml",
        "devenv/rust-workspace.nix",
        "flake.lock",
        "flake.nix",
        "nix/crate2nix-regeneration.nix",
        "nix/source.nix",
        "nix/workspace.nix",
        "rust-toolchain.toml",
        "scripts/check_crate2nix_sync.py",
        "scripts/ci_detect_nix_graph_changes.py",
    }
)


def is_graph_input(path: str) -> bool:
    """Return true when one path can change the generated Cargo graph."""
    normalized = path.strip().strip('"')
    if not normalized:
        return False
    posix = PurePosixPath(normalized)
    return normalized in GRAPH_EXACT_PATHS or posix.name == "Cargo.toml"


def requires_regeneration(paths: list[str]) -> bool:
    """Select regeneration for graph inputs or an unusable empty path set."""
    cleaned = [path.strip().strip('"') for path in paths if path.strip()]
    if not cleaned:
        return True
    return any(is_graph_input(path) for path in cleaned)


def _emit(required: bool) -> None:
    line = f"cargo_graph_changed={'true' if required else 'false'}"
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(line + "\n")
    print(line)


def main() -> int:
    _emit(requires_regeneration(sys.stdin.read().splitlines()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
