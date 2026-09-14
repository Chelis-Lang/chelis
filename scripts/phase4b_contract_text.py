"""Shared Phase 4B contract extraction; no inventory or execution authority."""
from __future__ import annotations

import re

ATOM_START = re.compile(
    r"^> \*\*\[(\d{2}-[A-Z]+-\d+)\]\*\*", re.MULTILINE
)


class OracleError(RuntimeError):
    """The frozen Phase 4B contract is incomplete or internally inconsistent."""


def strict_atom_block(text: str, atom: str) -> str:
    starts = [match for match in ATOM_START.finditer(text) if match.group(1) == atom]
    if len(starts) != 1:
        raise OracleError(
            f"frozen normative atom {atom} must occur exactly once, got {len(starts)}"
        )
    start = starts[0].start()
    end = start
    for line in text[start:].splitlines(keepends=True):
        if end > start and ATOM_START.match(line):
            break
        if not line.startswith(">"):
            break
        end += len(line)
    return text[start:end]


def normalize_frozen_block(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in normalized.split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def frozen_region(text: str, start: str, end: str, label: str) -> str:
    start_count = text.count(start)
    end_count = text.count(end)
    if start_count != 1 or end_count != 1:
        raise OracleError(
            f"frozen {label} markers must occur exactly once "
            f"(start={start_count}, end={end_count})"
        )
    start_index = text.index(start)
    end_index = text.index(end)
    if end_index <= start_index:
        raise OracleError(f"frozen {label} end marker precedes its start")
    return text[start_index:end_index]
