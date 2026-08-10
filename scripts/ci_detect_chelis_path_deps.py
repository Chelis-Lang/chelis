#!/usr/bin/env python3
"""Classify downstream Cargo manifests by Chelis source-path use."""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SKIP_DIRECTORIES = frozenset({".git", "target"})
CARGO_CONFIG_NAMES = frozenset({"config", "config.toml"})


class ChelisPathDependencyError(ValueError):
    """Report invalid classifier input at its parse boundary."""


@dataclass(frozen=True)
class ChelisPathDependency:
    """One Cargo path value that resolves into the Chelis source root."""

    manifest: Path
    dependency_path: str
    resolved_path: Path


def _parse_directory(raw: str, label: str) -> Path:
    path = Path(raw).resolve()
    if not path.is_dir():
        raise ChelisPathDependencyError(f"{label} is not a directory: {path}")
    return path


def _is_skipped(path: Path, shell_root: Path) -> bool:
    relative = path.relative_to(shell_root)
    return any(part in SKIP_DIRECTORIES for part in relative.parts)


def _manifest_paths(shell_root: Path) -> tuple[Path, ...]:
    manifests = [
        path.resolve()
        for path in shell_root.rglob("Cargo.toml")
        if not _is_skipped(path, shell_root)
    ]
    return tuple(sorted(manifests))


def _cargo_config_paths(shell_root: Path) -> tuple[Path, ...]:
    configs = [
        path.resolve()
        for path in shell_root.rglob("*")
        if path.is_file()
        and path.name in CARGO_CONFIG_NAMES
        and path.parent.name == ".cargo"
        and not _is_skipped(path, shell_root)
    ]
    return tuple(sorted(configs))


def _dependency_tables(document: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    table_names = ("dependencies", "dev-dependencies", "build-dependencies")
    tables: list[dict[str, Any]] = []

    def append_named_tables(parent: Any) -> None:
        if not isinstance(parent, dict):
            return
        for name in table_names:
            table = parent.get(name)
            if isinstance(table, dict):
                tables.append(table)

    append_named_tables(document)
    workspace = document.get("workspace")
    append_named_tables(workspace)

    targets = document.get("target")
    if isinstance(targets, dict):
        for target in targets.values():
            append_named_tables(target)

    patches = document.get("patch")
    if isinstance(patches, dict):
        for patch_table in patches.values():
            if isinstance(patch_table, dict):
                tables.append(patch_table)

    replacement = document.get("replace")
    if isinstance(replacement, dict):
        tables.append(replacement)

    return tuple(tables)


def _dependency_path_values(document: dict[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    for table in _dependency_tables(document):
        for dependency in table.values():
            if not isinstance(dependency, dict):
                continue
            path_value = dependency.get("path")
            if isinstance(path_value, str):
                found.append(path_value)
    return tuple(found)


def _config_path_values(document: dict[str, Any]) -> tuple[str, ...]:
    found = list(_dependency_path_values(document))
    path_overrides = document.get("paths")
    if isinstance(path_overrides, list):
        found.extend(path for path in path_overrides if isinstance(path, str))
    return tuple(found)


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ChelisPathDependencyError(f"cannot parse {path}: {error}") from error


def _is_chelis_source_path(resolved: Path, chelis_root: Path) -> bool:
    return resolved == chelis_root or resolved.is_relative_to(chelis_root)


def find_chelis_path_dependencies(
    shell_root: Path, chelis_root: Path
) -> tuple[ChelisPathDependency, ...]:
    """Return all Cargo path values that resolve inside Chelis source."""
    shell_root = shell_root.resolve()
    chelis_source = chelis_root.resolve()
    found: list[ChelisPathDependency] = []

    for manifest in _manifest_paths(shell_root):
        try:
            document = _read_toml(manifest)
        except ChelisPathDependencyError:
            if manifest == shell_root / "Cargo.toml":
                raise
            print(
                f"warning: ignored invalid nested Cargo manifest: {manifest}",
                file=sys.stderr,
            )
            continue

        for dependency_path in _dependency_path_values(document):
            resolved = (manifest.parent / dependency_path).resolve()
            if _is_chelis_source_path(resolved, chelis_source):
                found.append(
                    ChelisPathDependency(
                        manifest=manifest,
                        dependency_path=dependency_path,
                        resolved_path=resolved,
                    )
                )

    for config in _cargo_config_paths(shell_root):
        document = _read_toml(config)
        config_root = config.parent.parent
        for dependency_path in _config_path_values(document):
            resolved = (config_root / dependency_path).resolve()
            if _is_chelis_source_path(resolved, chelis_source):
                found.append(
                    ChelisPathDependency(
                        manifest=config,
                        dependency_path=dependency_path,
                        resolved_path=resolved,
                    )
                )

    return tuple(found)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify downstream Cargo manifests by Chelis source-path use."
    )
    parser.add_argument("shell_root")
    parser.add_argument("chelis_root")
    parser.add_argument("--expect", choices=("present", "absent"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        shell_root = _parse_directory(args.shell_root, "shell root")
        chelis_root = Path(args.chelis_root).resolve()
        found = find_chelis_path_dependencies(shell_root, chelis_root)
    except ChelisPathDependencyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    for dependency in found:
        print(
            f"{dependency.manifest}: {dependency.dependency_path} -> "
            f"{dependency.resolved_path}"
        )

    present = bool(found)
    expected_present = args.expect == "present"
    if present != expected_present:
        actual = "present" if present else "absent"
        print(
            f"error: expected Chelis source dependencies to be {args.expect}, "
            f"but they are {actual}",
            file=sys.stderr,
        )
        return 1

    print(f"Chelis source dependencies: {args.expect}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
