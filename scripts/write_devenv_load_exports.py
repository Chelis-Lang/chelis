#!/usr/bin/env python3
"""Atomically publish Devenv's generated shell-export payload."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path


def publish(destination: Path, payload: str) -> None:
    """Replace *destination* with one complete, executable UTF-8 payload."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_staged = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    staged = Path(raw_staged)
    try:
        os.fchmod(descriptor, 0o700)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    publish(arguments.destination, os.environ["DEVENV_TASK_ENV"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
