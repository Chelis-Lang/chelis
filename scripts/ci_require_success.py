#!/usr/bin/env python3
"""Fail unless every named upstream CI job completed successfully."""

from __future__ import annotations

import sys
from collections.abc import Sequence


def parse_results(arguments: Sequence[str]) -> list[tuple[str, str]]:
    """Parse ``NAME=RESULT`` arguments without accepting unnamed results."""

    parsed: list[tuple[str, str]] = []
    for argument in arguments:
        name, separator, result = argument.partition("=")
        if not separator or not name or not result:
            raise ValueError(f"expected NAME=RESULT, got {argument!r}")
        parsed.append((name, result))
    if not parsed:
        raise ValueError("expected at least one NAME=RESULT argument")
    return parsed


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        results = parse_results(sys.argv[1:] if arguments is None else arguments)
    except ValueError as error:
        print(f"CI result aggregation error: {error}", file=sys.stderr)
        return 2

    failed = [(name, result) for name, result in results if result != "success"]
    stream = sys.stderr if failed else sys.stdout
    for name, result in results:
        print(f"{name}: {result}", file=stream)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
