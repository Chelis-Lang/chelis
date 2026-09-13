#!/usr/bin/env python3
"""Generate the [05-UNS-5] atom and issue membership artifacts.

The atom input is derived from normative blockquote definitions in numbered
specs. The issue manifest is derived from exact literal
``unimplemented_rejection!`` citations in production crate ``src`` trees.
Its rows are separately validated against GitHub by
``validate_rejection_issue_manifest.py``.

Run with the uv-managed interpreter:

    .venv/bin/python scripts/generate_rejection_registries.py --check
    .venv/bin/python scripts/generate_rejection_registries.py --write
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ATOM = re.compile(r"^> \*\*(\[[0-9]{2}-[A-Z]+-[1-9][0-9]*\])\*\*", re.MULTILINE)
NUMBERED_SPEC = re.compile(r"^(?:0[0-9]|1[0-2])-[^/]+\.md$")
EXACT_ISSUE_LITERAL = re.compile(r"[1-9][0-9]*(?:_[0-9]+)*")
MANIFEST_REL = Path("spec/design/loud_unsupported_issue_manifest.json")
OUTPUT_REL = Path("crates/chelis-types/src/rejection_registry_generated.rs")
PRODUCTION_SOURCE_EXCLUSIONS = frozenset(
    {Path("crates/chelis-types/src/unsupported.rs")}
)


class RegistryError(ValueError):
    """A source registry is malformed or ambiguous."""


@dataclass(frozen=True)
class IssueCitation:
    """One exact production issue citation."""

    path: Path
    line: int
    number: int


def discover_atoms(spec_dir: Path) -> list[str]:
    """Return every normative atom declared by the numbered specs."""
    atoms: list[str] = []
    for path in sorted(spec_dir.glob("*.md")):
        if NUMBERED_SPEC.fullmatch(path.name):
            atoms.extend(ATOM.findall(path.read_text()))
    duplicates = sorted({atom for atom in atoms if atoms.count(atom) > 1})
    if duplicates:
        raise RegistryError(f"duplicate normative atoms: {duplicates}")
    return sorted(atoms)


def discover_production_sources(root: Path) -> list[Path]:
    """Return the exact Rust production surface that may cite issue authority.

    Production is every ``*.rs`` file recursively below ``crates/*/src``.
    Crate tests, examples, benches, build scripts, scripts, and generated
    build output are outside that root. The macro implementation itself is
    excluded because it defines the construction edge rather than citing an
    implementation owner.
    """
    sources: list[Path] = []
    for source_root in sorted((root / "crates").glob("*/src")):
        if not source_root.is_dir():
            continue
        for path in sorted(source_root.rglob("*.rs")):
            relative = path.relative_to(root)
            if relative in PRODUCTION_SOURCE_EXCLUSIONS:
                continue
            sources.append(path)
    return sources


def _skip_block_comment(source: str, index: int, path: Path) -> int:
    depth = 1
    cursor = index + 2
    while cursor < len(source):
        if source.startswith("/*", cursor):
            depth += 1
            cursor += 2
        elif source.startswith("*/", cursor):
            depth -= 1
            cursor += 2
            if depth == 0:
                return cursor
        else:
            cursor += 1
    line = source.count("\n", 0, index) + 1
    raise RegistryError(f"{path}:{line}: unterminated block comment")


def _raw_string_end(source: str, index: int) -> int | None:
    cursor = index
    for prefix in ("br", "cr", "r"):
        if source.startswith(prefix, cursor):
            cursor += len(prefix)
            break
    else:
        return None
    hashes_start = cursor
    while cursor < len(source) and source[cursor] == "#":
        cursor += 1
    if cursor >= len(source) or source[cursor] != '"':
        return None
    hashes = source[hashes_start:cursor]
    closing = '"' + hashes
    end = source.find(closing, cursor + 1)
    return len(source) if end < 0 else end + len(closing)


def _quoted_string_end(source: str, index: int) -> int | None:
    cursor = index
    if cursor < len(source) and source[cursor] in {"b", "c"}:
        cursor += 1
    if cursor >= len(source) or source[cursor] != '"':
        return None
    cursor += 1
    while cursor < len(source):
        if source[cursor] == "\\":
            cursor += 2
        elif source[cursor] == '"':
            return cursor + 1
        else:
            cursor += 1
    return len(source)


def _character_literal_end(source: str, index: int) -> int | None:
    """Return the end of an ordinary or byte character literal.

    A failed match is left to the ordinary token scan so Rust lifetimes such
    as ``'static`` are not consumed as character literals.
    """
    cursor = index
    if source.startswith("b'", cursor):
        cursor += 1
    if cursor >= len(source) or source[cursor] != "'":
        return None
    cursor += 1
    if cursor >= len(source):
        return None

    if source[cursor] == "\\":
        cursor += 1
        if cursor >= len(source):
            return None
        escape = source[cursor]
        if escape == "x":
            cursor += 3
        elif escape == "u" and source.startswith("u{", cursor):
            closing = source.find("}", cursor + 2)
            if closing < 0:
                return None
            cursor = closing + 1
        else:
            cursor += 1
    else:
        if source[cursor] in {"'", "\n", "\r"}:
            return None
        cursor += 1

    if cursor >= len(source) or source[cursor] != "'":
        return None
    return cursor + 1


def _skip_trivia(source: str, index: int, path: Path) -> int:
    cursor = index
    while cursor < len(source):
        if source[cursor].isspace():
            cursor += 1
        elif source.startswith("//", cursor):
            newline = source.find("\n", cursor + 2)
            cursor = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", cursor):
            cursor = _skip_block_comment(source, cursor, path)
        else:
            break
    return cursor


def _citation_error(path: Path, source: str, index: int) -> RegistryError:
    line = source.count("\n", 0, index) + 1
    return RegistryError(
        f"{path}:{line}: unimplemented_rejection! issue must be an "
        "unsuffixed positive decimal integer literal followed by a comma"
    )


def _is_identifier(source: str, index: int, identifier: str) -> bool:
    end = index + len(identifier)
    return (
        source.startswith(identifier, index)
        and (end >= len(source) or not (source[end].isalnum() or source[end] == "_"))
    )


def _alias_error(path: Path, source: str, index: int) -> RegistryError:
    line = source.count("\n", 0, index) + 1
    return RegistryError(
        f"{path}:{line}: production unimplemented_rejection imports and "
        "reexports must retain its exact spelling; aliases are forbidden"
    )


def parse_issue_citations(path: Path, source: str) -> list[IssueCitation]:
    """Parse exact-spelled literal citations outside comments and literals."""
    citations: list[IssueCitation] = []
    cursor = 0
    while cursor < len(source):
        if source.startswith("//", cursor):
            newline = source.find("\n", cursor + 2)
            cursor = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", cursor):
            cursor = _skip_block_comment(source, cursor, path)
            continue
        raw_end = _raw_string_end(source, cursor)
        if raw_end is not None:
            cursor = raw_end
            continue
        character_end = _character_literal_end(source, cursor)
        if character_end is not None:
            cursor = character_end
            continue
        quoted_end = _quoted_string_end(source, cursor)
        if quoted_end is not None:
            cursor = quoted_end
            continue
        if source[cursor].isalpha() or source[cursor] == "_":
            identifier_start = cursor
            cursor += 1
            while cursor < len(source) and (
                source[cursor].isalnum() or source[cursor] == "_"
            ):
                cursor += 1
            if source[identifier_start:cursor] != "unimplemented_rejection":
                continue

            bang = _skip_trivia(source, cursor, path)
            if bang >= len(source) or source[bang] != "!":
                if _is_identifier(source, bang, "as"):
                    raise _alias_error(path, source, identifier_start)
                continue
            opening = _skip_trivia(source, bang + 1, path)
            if opening >= len(source) or source[opening] != "(":
                raise _citation_error(path, source, identifier_start)
            argument = _skip_trivia(source, opening + 1, path)
            literal_end = argument
            while literal_end < len(source) and (
                source[literal_end].isdigit() or source[literal_end] == "_"
            ):
                literal_end += 1
            literal = source[argument:literal_end]
            after_literal = _skip_trivia(source, literal_end, path)
            if (
                EXACT_ISSUE_LITERAL.fullmatch(literal) is None
                or after_literal >= len(source)
                or source[after_literal] != ","
            ):
                raise _citation_error(path, source, identifier_start)
            number = int(literal.replace("_", ""))
            if number > 0xFFFF_FFFF:
                raise _citation_error(path, source, identifier_start)
            citations.append(
                IssueCitation(
                    path=path,
                    line=source.count("\n", 0, identifier_start) + 1,
                    number=number,
                )
            )
            cursor = after_literal + 1
            continue
        cursor += 1
    return citations


def discover_issue_citations(root: Path) -> list[IssueCitation]:
    """Return every exact production ``unimplemented_rejection!`` citation."""
    citations: list[IssueCitation] = []
    for path in discover_production_sources(root):
        citations.extend(
            parse_issue_citations(
                path.relative_to(root), path.read_text(encoding="utf-8")
            )
        )
    return citations


def derive_issue_numbers(citations: list[IssueCitation]) -> list[int]:
    """Collapse repeated sites into the sorted issue membership registry."""
    return sorted({citation.number for citation in citations})


def load_issue_manifest(path: Path) -> list[int]:
    """Parse the checked-in open-issue manifest with strict shape checks."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read issue manifest: {error}") from error
    if payload.get("schema") != 1 or not isinstance(payload.get("issues"), list):
        raise RegistryError("issue manifest must have schema 1 and an issues list")

    numbers: list[int] = []
    for index, row in enumerate(payload["issues"]):
        if not isinstance(row, dict):
            raise RegistryError(f"issue row {index} is not an object")
        if set(row) != {"number", "kind", "state"}:
            raise RegistryError(f"issue row {index} has unknown or missing fields")
        number = row["number"]
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise RegistryError(f"issue row {index} has invalid number {number!r}")
        if row["kind"] != "issue" or row["state"] != "open":
            raise RegistryError(
                f"issue row {index} must record kind=issue and state=open"
            )
        numbers.append(number)
    if numbers != sorted(set(numbers)):
        raise RegistryError("issue rows must be sorted and unique")
    return numbers


def manifest_derivation_problems(
    manifest_issues: list[int], source_issues: list[int]
) -> list[str]:
    """Report both directions of drift between source citations and rows."""
    problems: list[str] = []
    missing = sorted(set(source_issues) - set(manifest_issues))
    stale = sorted(set(manifest_issues) - set(source_issues))
    if missing:
        problems.append(f"issue manifest is missing source-cited rows: {missing}")
    if stale:
        problems.append(f"issue manifest contains uncited stale rows: {stale}")
    return problems


def render_issue_manifest(issues: list[int]) -> str:
    """Render the checked-in, source-derived issue liveness input."""
    lines = ['{', '  "schema": 1,', '  "issues": [']
    for index, number in enumerate(issues):
        comma = "," if index + 1 < len(issues) else ""
        lines.append(
            f'    {{ "number": {number}, "kind": "issue", "state": "open" }}{comma}'
        )
    lines.extend(["  ]", "}", ""])
    return "\n".join(lines)


def render_registry(atoms: list[str], issues: list[int]) -> str:
    """Render the dependency-free Rust membership artifact."""
    lines = [
        "// @generated by scripts/generate_rejection_registries.py; do not edit.",
        "",
        "#[rustfmt::skip]",
        "pub(crate) const REGISTERED_SPEC_ATOMS: &[&str] = &[",
    ]
    lines.extend(f'    "{atom}",' for atom in atoms)
    lines.extend(
        [
            "];",
            "",
            "#[rustfmt::skip]",
            "pub(crate) const REGISTERED_OPEN_ISSUES: &[u32] = &[",
        ]
    )
    lines.extend(f"    {number}," for number in issues)
    lines.extend(["];", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    citations = discover_issue_citations(root)
    issues = derive_issue_numbers(citations)
    rendered_manifest = render_issue_manifest(issues)
    rendered_registry = render_registry(discover_atoms(root / "spec"), issues)
    manifest = root / MANIFEST_REL
    output = root / OUTPUT_REL
    if args.write:
        manifest.write_text(rendered_manifest)
        output.write_text(rendered_registry)
        print(f"wrote {manifest.relative_to(root)}")
        print(f"wrote {output.relative_to(root)}")
        return 0
    manifest_issues = load_issue_manifest(manifest)
    problems = manifest_derivation_problems(manifest_issues, issues)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(
            "source-derived rejection issue manifest is stale; run "
            ".venv/bin/python scripts/generate_rejection_registries.py --write",
            file=sys.stderr,
        )
        return 1
    if manifest.read_text() != rendered_manifest:
        print(
            "source-derived rejection issue manifest is not byte-identical; run "
            ".venv/bin/python scripts/generate_rejection_registries.py --write",
            file=sys.stderr,
        )
        return 1
    try:
        current = output.read_text()
    except OSError as error:
        print(f"generated rejection registry missing: {error}", file=sys.stderr)
        return 1
    if current != rendered_registry:
        print(
            "generated rejection registry is stale; run "
            ".venv/bin/python scripts/generate_rejection_registries.py --write",
            file=sys.stderr,
        )
        return 1
    print("REJECTION REGISTRIES: BYTE AGREEMENT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
