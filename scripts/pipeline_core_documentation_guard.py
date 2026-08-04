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
        line.lower().replace("`", "").replace("#![no_std]", "no_std").split()
    )
    absence_claims = (
        "does not require std",
        "no longer requires std",
        "works without std",
        "work without std",
        "runs without std",
        "run without std",
    )
    if any(claim in normalized for claim in absence_claims):
        return True
    if "no_std" not in normalized:
        return False
    if re.search(r"\b(?:does not|cannot|is not)\b.{0,40}\bno_std\b", normalized):
        return False
    positive_patterns = (
        r"\b(?:supports?|provides?|enables?|has|uses?)\b.{0,40}\bno_std\b",
        r"\bno_std\b.{0,24}\b(?:compatible|ready|supported|capable)\b",
        r"\bis\s+no_std\b",
    )
    return any(re.search(pattern, normalized) for pattern in positive_patterns)


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
