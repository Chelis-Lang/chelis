#!/usr/bin/env python3
"""Fail closed when Pyright/Pylance can escape Chelis's Python source graph."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "pyrightconfig.json"
REQUIRED_EXCLUDES = frozenset(
    {
        "**/.devenv",
        "**/.git",
        "**/.venv",
        "**/__pycache__",
        "**/node_modules",
        "**/target",
    }
)


def tracked_python_files() -> tuple[PurePosixPath, ...]:
    completed = subprocess.run(
        ("git", "ls-files", "-z", "--", "*.py"),
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return tuple(
        PurePosixPath(raw.decode("utf-8"))
        for raw in completed.stdout.split(b"\0")
        if raw
    )


def _is_beneath(path: PurePosixPath, root: PurePosixPath) -> bool:
    return path == root or root in path.parents


def validate_scope(config: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    raw_includes = config.get("include")
    raw_excludes = config.get("exclude")
    if not isinstance(raw_includes, list) or not all(
        isinstance(item, str) for item in raw_includes
    ):
        return ["pyrightconfig.json include must be a list of relative paths"]
    if not isinstance(raw_excludes, list) or not all(
        isinstance(item, str) for item in raw_excludes
    ):
        return ["pyrightconfig.json exclude must be a list of glob strings"]

    missing_excludes = sorted(REQUIRED_EXCLUDES.difference(raw_excludes))
    if missing_excludes:
        failures.append(f"missing generated-tree exclusions: {missing_excludes}")

    include_paths: list[PurePosixPath] = []
    for spelling in raw_includes:
        relative = PurePosixPath(spelling)
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"include roots must stay relative to the repository: {spelling}")
            continue
        candidate = REPO_ROOT.joinpath(*relative.parts)
        if not candidate.is_dir():
            failures.append(f"include root is not a repository directory: {spelling}")
            continue
        cursor = REPO_ROOT
        escaped = False
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                failures.append(f"include root crosses a directory symlink: {spelling}")
                escaped = True
                break
        if escaped:
            continue
        for directory, names, _files in os.walk(candidate, followlinks=False):
            for name in names:
                child = Path(directory) / name
                if child.is_symlink():
                    failures.append(
                        f"included Python tree contains a directory symlink: "
                        f"{child.relative_to(REPO_ROOT)}"
                    )
        include_paths.append(relative)

    for tracked in tracked_python_files():
        if not any(_is_beneath(tracked, root) for root in include_paths):
            failures.append(f"tracked Python file is outside the analysis roots: {tracked}")
    return failures


def main() -> int:
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"pyright scope: cannot read {CONFIG_PATH}: {error}", file=sys.stderr)
        return 1
    if not isinstance(config, dict):
        print("pyright scope: configuration root must be an object", file=sys.stderr)
        return 1
    failures = validate_scope(config)
    if failures:
        for failure in failures:
            print(f"pyright scope: {failure}", file=sys.stderr)
        return 1
    print(
        "pyright scope: PASS "
        f"({len(config['include'])} roots, {len(tracked_python_files())} tracked files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
