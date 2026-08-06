#!/usr/bin/env python3
"""Reject false portability claims for chelis-pipeline-core."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = REPO_ROOT / "docs" / "investigations" / "pipeline_core_std_blockers.md"
REQUIRED_STATEMENT = "`chelis-pipeline-core` requires `std`."
REQUIRED_SECTIONS = (
    "Collections and allocation",
    "Global state",
    "Stack support",
    "Panic behavior",
    "Operating-system use",
    "Dependency features",
)
REQUIRED_CRATES = (
    "`chelis-deep`",
    "`chelis-types`",
    "`chelis-effects`",
    "`chelis-ir`",
)


class DocumentationBoundaryError(RuntimeError):
    """The blocker inventory states an invalid portability contract."""


@dataclass(frozen=True)
class InventoryDocument:
    text: str
    lines: tuple[str, ...]


def parse_text(text: str) -> InventoryDocument:
    if not isinstance(text, str):
        raise DocumentationBoundaryError("the blocker inventory is not text")
    return InventoryDocument(text=text, lines=tuple(text.splitlines()))


def false_no_std_claim(line: str) -> bool:
    normalized = " ".join(
        line.lower()
        .replace("`", "")
        .replace("#![no_std]", "no_std")
        .replace("#![no-std]", "no_std")
        .replace("no-std", "no_std")
        .split()
    )

    # Affirmative std-independence: the crate is claimed to not need std.
    independence_claims = (
        "does not require std",
        "does not need std",
        "no longer requires std",
        "no longer needs std",
        "works without std",
        "work without std",
        "runs without std",
        "run without std",
    )
    if any(claim in normalized for claim in independence_claims):
        return True

    if "no_std" not in normalized:
        return False

    # A negated no_std capability, or a future-target frame, is not a current
    # claim. This is checked before any affirmative pattern so "supports std
    # only, not no_std" and "does not support no_std" are never flagged, and so
    # a blocker line that says what no_std would require is not a claim.
    negation_or_future_patterns = (
        r"\bnot\s+(?:a\s+|an\s+|yet\s+)?no_std\b",
        r"\b(?:does not|do not|cannot|can not|is not|are not|will not|won't|wont|no longer)\b[^.]{0,25}\bno_std\b",
        r"\bno_std\b[^.]{0,25}\b(?:is|are)\s+not\b",
        r"\bno_std\b[^.]{0,25}\bnot\s+(?:yet\s+)?(?:supported|available|possible|current)\b",
        r"\bfuture\s+(?:target|capability|goal|work|non-goal|milestone)\b",
        r"\bnot\s+(?:a\s+|an\s+)?current\b",
        # Hypothetical framing: a line that says what no_std *would* require is
        # discussing a blocker, not asserting a current capability.
        r"\bno_std\b[^.]{0,20}\b(?:would|could)\b",
        r"\b(?:would|could)\b[^.]{0,20}\bno_std\b",
        r"\bno_std\b[^.]{0,20}\bout of scope\b",
    )
    if any(re.search(pattern, normalized) for pattern in negation_or_future_patterns):
        return False

    # An affirmative current no_std capability claim about the crate. The
    # vocabulary is deliberately broad (support verbs, adjective forms, and
    # environment phrasings) so a real claim does not slip through on a common
    # wording such as "portable to no_std" or "works in a no_std environment".
    affirmative_patterns = (
        r"\b(?:supports?|provides?|enables?|offers?|gain(?:s|ed)?|add(?:s|ed)?)\b"
        r"[^.]{0,20}\bno_std\b",
        r"\bno_std\b[- ]?(?:compatible|compatibility|ready|capable|support|supported)\b",
        r"\b(?:is|are)\s+no_std\b",
        r"\b(?:portable\s+(?:to|on)|works?\s+(?:in|on|under)|runs?\s+(?:in|on|under)"
        r"|compiles?\s+(?:as|for|to|under)|builds?\s+(?:as|for|under)|targets?)\b"
        r"[^.]{0,25}\bno_std\b",
    )
    return any(re.search(pattern, normalized) for pattern in affirmative_patterns)


def validate_document(document: InventoryDocument) -> None:
    for line in document.lines:
        if false_no_std_claim(line):
            raise DocumentationBoundaryError(
                f"false current portability claim: {line.strip()}"
            )

    if REQUIRED_STATEMENT not in document.text:
        raise DocumentationBoundaryError(
            f"blocker inventory lacks `{REQUIRED_STATEMENT}`"
        )
    for section in REQUIRED_SECTIONS:
        if section not in document.text:
            raise DocumentationBoundaryError(
                f"blocker inventory lacks class `{section}`"
            )
    for crate in REQUIRED_CRATES:
        if crate not in document.text:
            raise DocumentationBoundaryError(
                f"blocker inventory lacks transitive crate {crate}"
            )


def validate_text(text: str) -> None:
    validate_document(parse_text(text))


def validate_repository() -> None:
    try:
        text = INVENTORY_PATH.read_text(encoding="utf-8")
    except OSError as error:
        raise DocumentationBoundaryError(
            f"cannot read {INVENTORY_PATH}: {error}"
        ) from error
    validate_text(text)


def main() -> int:
    try:
        validate_repository()
    except DocumentationBoundaryError as error:
        print(f"pipeline core documentation guard: FAIL: {error}", file=sys.stderr)
        return 1
    print("pipeline core documentation guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
