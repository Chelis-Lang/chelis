#!/usr/bin/env python3
"""Build once per Linux feature configuration and verify run-local test archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ENV = "CHELIS_TEST_ARCHIVE"
CONFIGURATIONS = {"workspace": [], "generalization": ["--features", "chelis-types/generalize-sweep-oracle"]}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(configuration: str) -> dict:
    return {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "configuration": configuration, "system": platform.system(),
            "machine": platform.machine(), "workspace": str(ROOT),
            "nextest": subprocess.check_output(["cargo", "nextest", "--version"], text=True).strip()}


def check_identity(expected: dict, actual: dict) -> None:
    if expected != actual:
        raise ValueError(f"Archive identity mismatch: expected {expected}, found {actual}")


def verify(path: Path, configuration: str) -> None:
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    check_identity(identity(configuration), manifest["identity"])
    if digest(path) != manifest["sha256"]:
        raise ValueError("Archive checksum mismatch")


def from_environment(environment=None) -> Path | None:
    environment = os.environ if environment is None else environment
    if ARCHIVE_ENV not in environment:
        return None
    value = environment[ARCHIVE_ENV]
    if not value:
        raise ValueError(f"{ARCHIVE_ENV} must name an archive")
    path = Path(value).resolve()
    verify(path, "workspace")
    return path


def reuse_command(command, path: Path) -> list[str]:
    """Translate only the package/target selectors used by the support oracles.

    Unknown options reject instead of silently applying a different build. The
    existing profile, filter and hash partition still decide which tests run.
    """
    if list(command[:3]) not in (["cargo", "nextest", "run"], ["cargo", "nextest", "list"]):
        raise ValueError("Archive reuse requires cargo nextest run/list")
    result = list(command[:3])
    packages, binaries, filters = [], [], []
    tokens = iter(command[3:])
    for token in tokens:
        if token == "--workspace":
            continue
        if token in ("-p", "--test"):
            name = next(tokens)
            if re.fullmatch(r"[A-Za-z0-9_-]+", name) is None:
                raise ValueError(f"Unsupported selector: {name}")
            (packages if token == "-p" else binaries).append(name)
        elif token == "--lib":
            filters.append("kind(lib)")
        elif token == "-E":
            filters.append(next(tokens))
        elif token in ("--profile", "--partition", "--message-format", "--run-ignored", "--test-threads"):
            result.extend([token, next(tokens)])
        elif token in ("--no-fail-fast", "--ignore-default-filter"):
            result.append(token)
        else:
            raise ValueError(f"Unsupported archive option: {token}")
    if packages:
        filters.append(" | ".join(f"package(={name})" for name in packages))
    if binaries:
        filters.append("(" + " | ".join(f"binary(={name})" for name in binaries) + ") & kind(test)")
    if filters:
        expression = filters[0] if len(filters) == 1 else " & ".join(f"({item})" for item in filters)
        result.extend(["-E", expression])
    return result + ["--archive-file", str(path), "--extract-to", str(ROOT), "--extract-overwrite"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "verify"))
    parser.add_argument("--configuration", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--archive-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.operation == "verify":
            verify(args.archive_file, args.configuration)
        else:
            args.archive_file.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cargo", "nextest", "archive", "--workspace",
                            *CONFIGURATIONS[args.configuration], "--archive-file", str(args.archive_file)],
                           cwd=ROOT, check=True)
            manifest = {"identity": identity(args.configuration), "sha256": digest(args.archive_file)}
            args.archive_file.with_suffix(args.archive_file.suffix + ".json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"CI archive failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
