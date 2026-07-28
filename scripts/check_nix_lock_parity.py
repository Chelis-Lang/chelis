#!/usr/bin/env python3
"""Compare shared Nixpkgs and Rust overlay revisions in both lock files."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLAKE_LOCK = REPO_ROOT / "flake.lock"
DEFAULT_DEVENV_LOCK = REPO_ROOT / "devenv.lock"


class LockParseError(ValueError):
    """A lock file does not contain the required typed node data."""


@dataclass(frozen=True)
class LockedNode:
    file_label: str
    node_name: str
    revision: str

    @property
    def label(self) -> str:
        return f"{self.file_label}:{self.node_name}"


@dataclass(frozen=True)
class ParsedLock:
    path: Path
    nodes: Mapping[str, LockedNode]

    def node(self, name: str) -> LockedNode:
        return self.nodes[name]


@dataclass(frozen=True)
class SharedInput:
    name: str
    flake_node: LockedNode
    devenv_node: LockedNode

    @property
    def matches(self) -> bool:
        return self.flake_node.revision == self.devenv_node.revision


def _object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LockParseError(f"{location} must be a JSON object")
    return value


def parse_lock(path: Path, node_names: tuple[str, ...]) -> ParsedLock:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LockParseError(f"cannot read {path}: {error}") from error

    document = _object(raw, str(path))
    nodes = _object(document.get("nodes"), f"{path}:nodes")
    parsed: dict[str, LockedNode] = {}
    for node_name in node_names:
        node = _object(nodes.get(node_name), f"{path}:nodes.{node_name}")
        locked = _object(node.get("locked"), f"{path}:nodes.{node_name}.locked")
        revision = locked.get("rev")
        if not isinstance(revision, str) or not revision:
            raise LockParseError(
                f"{path}:nodes.{node_name}.locked.rev must be a nonempty string"
            )
        parsed[node_name] = LockedNode(path.name, node_name, revision)

    return ParsedLock(path=path, nodes=MappingProxyType(parsed))


def shared_inputs(flake: ParsedLock, devenv: ParsedLock) -> tuple[SharedInput, ...]:
    return (
        SharedInput("nixpkgs", flake.node("nixpkgs"), devenv.node("nixpkgs-src")),
        SharedInput(
            "rust-overlay",
            flake.node("rust-overlay"),
            devenv.node("rust-overlay"),
        ),
    )


def check_parity(flake_path: Path, devenv_path: Path) -> tuple[SharedInput, ...]:
    flake = parse_lock(flake_path, ("nixpkgs", "rust-overlay"))
    devenv = parse_lock(devenv_path, ("nixpkgs-src", "rust-overlay"))
    return shared_inputs(flake, devenv)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flake-lock", type=Path, default=DEFAULT_FLAKE_LOCK)
    parser.add_argument("--devenv-lock", type=Path, default=DEFAULT_DEVENV_LOCK)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        inputs = check_parity(args.flake_lock, args.devenv_lock)
    except LockParseError as error:
        print(f"invalid lock file: {error}", file=sys.stderr)
        return 2

    mismatches = tuple(shared for shared in inputs if not shared.matches)
    if mismatches:
        for shared in mismatches:
            print(f"{shared.name} revision mismatch:", file=sys.stderr)
            print(
                f"  {shared.flake_node.label} = {shared.flake_node.revision}",
                file=sys.stderr,
            )
            print(
                f"  {shared.devenv_node.label} = {shared.devenv_node.revision}",
                file=sys.stderr,
            )
        return 1

    for shared in inputs:
        print(
            f"{shared.name}: {shared.flake_node.revision} "
            f"({shared.flake_node.label} = {shared.devenv_node.label})"
        )
    print("matching shared input revisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
