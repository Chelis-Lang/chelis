#!/usr/bin/env python3
"""Generate the [05-UNS-5] atom and issue membership tables.

The atom input is derived from normative blockquote definitions in numbered
specs. The issue input is the checked-in manifest whose rows are separately
validated against GitHub by ``validate_rejection_issue_manifest.py``.

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
UNIMPLEMENTED_LITERAL = re.compile(
    r"\b(?P<name>unimplemented_rejection)\s*!\s*\(\s*([0-9][0-9_]*)",
    re.MULTILINE,
)
UNIMPLEMENTED_NAME = re.compile(r"\bunimplemented_rejection\b")
RAW_STRING_START = re.compile(r"(?:br|r)(?P<hashes>#{0,255})\"")
MANIFEST_REL = Path("spec/design/loud_unsupported_issue_manifest.json")
OUTPUT_REL = Path("crates/chelis-types/src/rejection_registry_generated.rs")


class RegistryError(ValueError):
    """A source registry is malformed or ambiguous."""


@dataclass(frozen=True, order=True)
class AuthoritySite:
    """One executable unimplemented-authority construction site."""

    path: str
    line: int


def _blank(masked: list[str], source: str, start: int, end: int) -> None:
    """Blank a non-code span while preserving offsets and line numbers."""
    for index in range(start, end):
        if source[index] != "\n":
            masked[index] = " "


def _mask_rust_non_code(source: str) -> str:
    """Mask Rust comments and strings without changing source positions."""
    masked = list(source)
    index = 0
    while index < len(source):
        if source.startswith("//", index):
            end = source.find("\n", index + 2)
            if end < 0:
                end = len(source)
            _blank(masked, source, index, end)
            index = end
            continue
        if source.startswith("/*", index):
            start = index
            index += 2
            depth = 1
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            _blank(masked, source, start, index)
            continue

        raw = RAW_STRING_START.match(source, index)
        if raw:
            start = index
            hashes = raw.group("hashes")
            index = raw.end()
            terminator = f'\"{hashes}'
            end = source.find(terminator, index)
            index = len(source) if end < 0 else end + len(terminator)
            _blank(masked, source, start, index)
            continue

        if source[index] == '"':
            start = index
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index = min(index + 2, len(source))
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            _blank(masked, source, start, index)
            continue
        index += 1
    return "".join(masked)


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


def discover_issue_authorities(root: Path) -> dict[int, list[AuthoritySite]]:
    """Discover every production ``unimplemented_rejection!`` literal.

    Tests and fixtures are deliberately excluded: only executable production
    constructors keep an implementation issue live. The checked-in manifest is
    rendered from this map, so a constructor addition/removal cannot drift from
    its liveness evidence.
    """
    authorities: dict[int, list[AuthoritySite]] = {}
    crates = root / "crates"
    paths: list[Path] = []
    for crate in sorted(crates.glob("*")):
        if not crate.is_dir():
            continue
        src = crate / "src"
        if src.is_dir():
            paths.extend(src.rglob("*.rs"))
        build = crate / "build.rs"
        if build.is_file():
            paths.append(build)

    for path in sorted(set(paths)):
        relative_path = path.relative_to(root)
        if "tests" in relative_path.parts:
            continue
        if relative_path.as_posix() == "crates/chelis-types/src/unsupported.rs":
            # This is the macro definition and private builder owner, not a
            # construction site. Its shape is locked by the boundary checker.
            continue
        source = path.read_text(encoding="utf-8")
        code = _mask_rust_non_code(source)
        relative = relative_path.as_posix()
        matches = list(UNIMPLEMENTED_LITERAL.finditer(code))
        canonical_names = {match.start("name") for match in matches}
        for name in UNIMPLEMENTED_NAME.finditer(code):
            if name.start() not in canonical_names:
                line = source.count("\n", 0, name.start()) + 1
                raise RegistryError(
                    f"{relative}:{line}: noncanonical unimplemented_rejection "
                    "spelling; macro aliases and re-exports are forbidden so "
                    "the source-derived authority inventory stays complete"
                )
        for match in matches:
            number = int(match.group(2).replace("_", ""))
            line = source.count("\n", 0, match.start()) + 1
            authorities.setdefault(number, []).append(AuthoritySite(relative, line))
    return {
        number: sorted(set(sites))
        for number, sites in sorted(authorities.items())
    }


def render_issue_manifest(
    authorities: dict[int, list[AuthoritySite]],
) -> str:
    """Render the source-derived issue manifest with reviewable sites."""
    payload = {
        "schema": 2,
        "issues": [
            {
                "number": number,
                "kind": "issue",
                "state": "open",
                "sites": [
                    {"path": site.path, "line": site.line}
                    for site in sorted(sites)
                ],
            }
            for number, sites in sorted(authorities.items())
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def load_issue_manifest(path: Path) -> list[int]:
    """Parse the checked-in open-issue manifest with strict shape checks."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RegistryError(f"cannot read issue manifest: {error}") from error
    if payload.get("schema") != 2 or not isinstance(payload.get("issues"), list):
        raise RegistryError("issue manifest must have schema 2 and an issues list")

    numbers: list[int] = []
    for index, row in enumerate(payload["issues"]):
        if not isinstance(row, dict):
            raise RegistryError(f"issue row {index} is not an object")
        if set(row) != {"number", "kind", "state", "sites"}:
            raise RegistryError(f"issue row {index} has unknown or missing fields")
        number = row["number"]
        if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
            raise RegistryError(f"issue row {index} has invalid number {number!r}")
        if row["kind"] != "issue" or row["state"] != "open":
            raise RegistryError(
                f"issue row {index} must record kind=issue and state=open"
            )
        sites = row["sites"]
        if not isinstance(sites, list) or not sites:
            raise RegistryError(f"issue row {index} must have construction sites")
        site_keys: list[tuple[str, int]] = []
        for site_index, site in enumerate(sites):
            if not isinstance(site, dict) or set(site) != {"path", "line"}:
                raise RegistryError(
                    f"issue row {index} site {site_index} has invalid shape"
                )
            site_path = site["path"]
            line = site["line"]
            if (
                not isinstance(site_path, str)
                or not site_path.startswith("crates/")
                or not isinstance(line, int)
                or isinstance(line, bool)
                or line <= 0
            ):
                raise RegistryError(
                    f"issue row {index} site {site_index} is invalid"
                )
            site_keys.append((site_path, line))
        if site_keys != sorted(set(site_keys)):
            raise RegistryError(f"issue row {index} sites must be sorted and unique")
        numbers.append(number)
    if numbers != sorted(set(numbers)):
        raise RegistryError("issue rows must be sorted and unique")
    return numbers


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
    authorities = discover_issue_authorities(root)
    manifest_rendered = render_issue_manifest(authorities)
    issues = sorted(authorities)
    registry_rendered = render_registry(discover_atoms(root / "spec"), issues)
    manifest = root / MANIFEST_REL
    output = root / OUTPUT_REL
    if args.write:
        manifest.write_text(manifest_rendered)
        output.write_text(registry_rendered)
        print(f"wrote {manifest.relative_to(root)}")
        print(f"wrote {output.relative_to(root)}")
        return 0
    stale: list[str] = []
    for path, expected in (
        (manifest, manifest_rendered),
        (output, registry_rendered),
    ):
        try:
            current = path.read_text()
        except OSError as error:
            print(f"generated rejection artifact missing: {error}", file=sys.stderr)
            return 1
        if current != expected:
            stale.append(path.relative_to(root).as_posix())
    if stale:
        print(
            "generated rejection artifacts are stale "
            f"({', '.join(stale)}); run "
            ".venv/bin/python scripts/generate_rejection_registries.py --write",
            file=sys.stderr,
        )
        return 1
    print("REJECTION REGISTRIES: BYTE AGREEMENT PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
