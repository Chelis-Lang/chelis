#!/usr/bin/env python3
"""Reject commit messages that contain AI tool authorship markers.

It also warns, without refusing the commit, when the commit stages a generated
chelis-std file (`check_std_bundle_untracked.py`); CI refuses those.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TextIO


PROHIBITED_PATTERNS = (
    re.compile(
        r"(?:co-)?authored-by:.*"
        r"(?:claude|anthropic|kiro|openai|codex|chatgpt|copilot)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:co-)?authored-by:.*\b(?:ai|llm|assistant)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"generated (?:with|by).*"
        r"(?:claude|anthropic|kiro|openai|codex|chatgpt|copilot|"
        r"gpt-[0-9]+|\bai\b|\bllm\b)",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class CommitMessage:
    path: Path
    text: str

    @classmethod
    def parse(cls, argv: Sequence[str]) -> CommitMessage:
        if len(argv) != 1:
            raise ValueError("the hook requires one commit message file")
        path = Path(argv[0])
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ValueError(
                f"cannot read commit message file {path}: {error}"
            ) from error
        return cls(path=path, text=text)


def find_prohibited_marker(message: str) -> str | None:
    """Return the first prohibited pattern that matches the message."""
    for pattern in PROHIBITED_PATTERNS:
        if pattern.search(message) is not None:
            return pattern.pattern
    return None


def warn_on_staged_generated_std_files(stderr: TextIO) -> None:
    """Report staged generated chelis-std files on `stderr`.

    Advisory only: a missing guard, or a git failure, skips the warning rather
    than blocking the commit, because CI is the refusal.
    """
    try:
        import check_std_bundle_untracked as guard
    except ImportError:
        return
    try:
        toplevel = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        guard.main(
            ["--staged", "--warn", "--repo", toplevel], out=io.StringIO(), err=stderr
        )
    except (OSError, subprocess.CalledProcessError):
        return


def main(argv: Sequence[str], *, stderr: TextIO = sys.stderr) -> int:
    try:
        message = CommitMessage.parse(argv)
    except ValueError as error:
        print(f"ERROR: {error}", file=stderr)
        return 2

    warn_on_staged_generated_std_files(stderr)

    matched_pattern = find_prohibited_marker(message.text)
    if matched_pattern is None:
        return 0

    print(
        "ERROR: commit message contains a prohibited AI authorship marker.",
        file=stderr,
    )
    print(f"Matched pattern: {matched_pattern}", file=stderr)
    print("Remove AI tool attribution lines from the commit message.", file=stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
