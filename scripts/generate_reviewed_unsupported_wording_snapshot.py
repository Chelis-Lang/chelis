#!/usr/bin/env python3
"""Generate the centralized reviewed unsupported-wording expectation.

This tier-0 artifact is not production-derived: tier 0 cannot execute the Rust
renderer, and parsing or re-rendering Rust source would create the prohibited
second prose implementation. The exact CLI tests compare production stderr to
this one canonical reviewed row, so the generated reviewed snapshot centralizes
the two chelis#1918 consumers without claiming that Python derived the prose
from production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "crates/chelis-cli/tests/snapshots"
    / "issue_1870__reviewed__unsupported_wording.snap"
)
SNAPSHOT_PROVENANCE = "centralized-reviewed-expectation"


def reviewed_expectation_rows() -> dict[str, str]:
    """One canonical reviewed row shared by multiple exact-output tests."""
    return {
        "annotated_concat_softmax_c": (
            "error: unsupported: builtin `softmax` on `chelis build` host emission "
            "(codegen:c); deliberate [04-TOT-2]: the checked builtin vocabulary and "
            "C expression vocabulary disagree; no fallback expression is permitted\n"
        ),
    }


def rendered_snapshot() -> bytes:
    return (
        json.dumps(reviewed_expectation_rows(), indent=2, sort_keys=True) + "\n"
    ).encode()


def run(output: Path = DEFAULT_OUTPUT, *, check: bool) -> int:
    expected = rendered_snapshot()
    if check:
        return 0 if output.is_file() and output.read_bytes() == expected else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(expected)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    return run(args.output, check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
