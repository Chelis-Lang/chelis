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
import tomllib
from dataclasses import dataclass
from pathlib import Path

ATOM = re.compile(r"^> \*\*(\[[0-9]{2}-[A-Z]+-[1-9][0-9]*\])\*\*", re.MULTILINE)
NUMBERED_SPEC = re.compile(r"^(?:0[0-9]|1[0-2])-[^/]+\.md$")
UNIMPLEMENTED_LITERAL = re.compile(
    r"\b(?P<name>unimplemented_rejection)\s*!\s*\(\s*([0-9][0-9_]*)",
    re.MULTILINE,
)
UNIMPLEMENTED_NAME = re.compile(r"\bunimplemented_rejection\b")
PRODUCTION_RUST_INCLUDE = re.compile(r"\binclude\s*!\s*\(")
BUILD_SCRIPT_MODULE = re.compile(
    r"\bmod\b\s+[^;{}\s][^;{}]*;"
)
ATTRIBUTE_START = re.compile(r"#\s*\[")
PATH_META = re.compile(r"^\s*(?:r#)?path\b\s*=")
CFG_ATTR_META = re.compile(r"^\s*(?:r#)?cfg_attr\b\s*\(")
UNIMPLEMENTED_DEFINITION = re.compile(
    r"\bmacro_rules\s*!\s*unimplemented_rejection\s*\{"
)
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

        if source[index] == "'":
            start = index
            cursor = index + 1
            if cursor < len(source) and source[cursor] == "\\":
                cursor += 1
                if cursor < len(source) and source[cursor] == "u":
                    if cursor + 1 < len(source) and source[cursor + 1] == "{":
                        closing_brace = source.find("}", cursor + 2)
                        cursor = len(source) if closing_brace < 0 else closing_brace + 1
                    else:
                        cursor += 1
                elif cursor < len(source) and source[cursor] == "x":
                    cursor = min(cursor + 3, len(source))
                else:
                    cursor = min(cursor + 1, len(source))
            else:
                cursor = min(cursor + 1, len(source))
            if cursor < len(source) and source[cursor] == "'":
                index = cursor + 1
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


def _matching_delimiter(code: str, opening: int) -> int | None:
    """Return the close paired with ``code[opening]`` in masked Rust."""
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}
    opener = code[opening]
    if opener not in pairs:
        raise ValueError(f"not an opening delimiter: {opener!r}")
    stack = [opener]
    for index in range(opening + 1, len(code)):
        token = code[index]
        if token in pairs:
            stack.append(token)
        elif token in closing:
            if not stack or stack[-1] != closing[token]:
                return None
            stack.pop()
            if not stack:
                return index
    return None


def _split_top_level_commas(meta: str) -> list[str]:
    """Split Rust meta items without splitting nested predicate arguments."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}
    for index, token in enumerate(meta):
        if token in pairs:
            stack.append(token)
        elif token in closing:
            if stack and stack[-1] == closing[token]:
                stack.pop()
        elif token == "," and not stack:
            parts.append(meta[start:index])
            start = index + 1
    parts.append(meta[start:])
    return parts


def _meta_contains_path_attribute(meta: str) -> bool:
    """Recognize built-in ``path`` meta, including nested ``cfg_attr``."""
    if PATH_META.match(meta):
        return True
    cfg_attr = CFG_ATTR_META.match(meta)
    if cfg_attr is None:
        return False
    opening = cfg_attr.end() - 1
    closing = _matching_delimiter(meta, opening)
    if closing is None:
        return False
    arguments = _split_top_level_commas(meta[opening + 1 : closing])
    return any(_meta_contains_path_attribute(item) for item in arguments[1:])


def _path_attribute_offset(code: str) -> int | None:
    """Return the first built-in Rust module ``path`` attribute offset."""
    for attribute in ATTRIBUTE_START.finditer(code):
        opening = code.find("[", attribute.start(), attribute.end())
        closing = _matching_delimiter(code, opening)
        if closing is None:
            continue
        if _meta_contains_path_attribute(code[opening + 1 : closing]):
            return attribute.start()
    return None


def _mask_owner_macro_definition(code: str, relative: str) -> str:
    """Mask only the canonical macro definition, never its whole owner file."""
    if relative != "crates/chelis-types/src/unsupported.rs":
        return code
    definitions = list(UNIMPLEMENTED_DEFINITION.finditer(code))
    if len(definitions) != 1:
        raise RegistryError(
            f"{relative}: expected exactly one canonical "
            "unimplemented_rejection macro definition"
        )
    definition = definitions[0]
    opening = definition.end() - 1
    closing = _matching_delimiter(code, opening)
    if closing is None:
        raise RegistryError(f"{relative}: unterminated unimplemented_rejection macro")
    masked = list(code)
    for index in range(definition.start(), closing + 1):
        if masked[index] != "\n":
            masked[index] = " "
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


def _load_toml(path: Path, label: str) -> dict:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
        raise RegistryError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RegistryError(f"{label} {path} is not a TOML table")
    return payload


def _workspace_members(root: Path) -> list[Path]:
    """Resolve every workspace member declared by the root Cargo manifest."""
    manifest_path = root / "Cargo.toml"
    if not manifest_path.is_file():
        raise RegistryError(f"missing workspace manifest: {manifest_path}")
    payload = _load_toml(manifest_path, "workspace manifest")
    workspace = payload.get("workspace")
    if not isinstance(workspace, dict):
        raise RegistryError("root Cargo.toml has no workspace table")
    patterns = workspace.get("members")
    if not isinstance(patterns, list) or not patterns:
        raise RegistryError("workspace members must be a non-empty list")

    members: set[Path] = set()
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern:
            raise RegistryError(f"invalid workspace member pattern: {pattern!r}")
        relative = Path(pattern)
        if relative.is_absolute() or ".." in relative.parts:
            raise RegistryError(f"workspace member escapes repository: {pattern}")
        matches = sorted(path for path in root.glob(pattern) if path.is_dir())
        if not matches:
            raise RegistryError(f"workspace member pattern matches nothing: {pattern}")
        members.update(matches)
    return sorted(members)


def _member_path(root: Path, member: Path, raw: str, label: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RegistryError(f"{label} must stay within its workspace member: {raw}")
    candidate = member / relative
    try:
        candidate.resolve().relative_to(member.resolve())
    except ValueError as error:
        raise RegistryError(f"{label} escapes workspace member: {raw}") from error
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise RegistryError(f"{label} escapes repository: {raw}") from error
    cursor = candidate
    while cursor != member:
        if cursor.is_symlink():
            raise RegistryError(
                f"{label} uses a source symlink: {candidate.relative_to(root)}"
            )
        cursor = cursor.parent
    return candidate


def _target_paths(member: Path, manifest: dict) -> list[Path]:
    """Return explicit lib/bin/example source paths from one package manifest."""
    paths: list[Path] = []
    tables: list[tuple[str, object]] = [
        ("lib", manifest.get("lib")),
        ("bin", manifest.get("bin", [])),
        ("example", manifest.get("example", [])),
    ]
    for kind, raw_tables in tables:
        if raw_tables is None:
            continue
        rows = [raw_tables] if kind == "lib" else raw_tables
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RegistryError(f"{member}/Cargo.toml has malformed {kind} targets")
        for row in rows:
            raw_path = row.get("path")
            if raw_path is None:
                continue
            if not isinstance(raw_path, str) or not raw_path:
                raise RegistryError(
                    f"{member}/Cargo.toml has invalid {kind} target path"
                )
            paths.append(Path(raw_path))
    return paths


def _inventory_tree(root: Path, source_root: Path, paths: set[Path]) -> None:
    """Add Rust files under one lexical root, rejecting symlink edges."""
    if source_root.is_symlink():
        raise RegistryError(
            f"source symlink is forbidden: {source_root.relative_to(root)}"
        )
    if not source_root.is_dir():
        return
    for entry in source_root.rglob("*"):
        if entry.is_symlink():
            raise RegistryError(
                f"source symlink is forbidden: {entry.relative_to(root)}"
            )
    paths.update(source_root.rglob("*.rs"))


def _workspace_rust_paths(root: Path) -> tuple[list[Path], set[Path]]:
    """Derive lexical production roots from workspace package manifests."""
    paths: set[Path] = set()
    build_scripts: set[Path] = set()
    for member in _workspace_members(root):
        if member.is_symlink():
            raise RegistryError(
                f"workspace member symlink is forbidden: {member.relative_to(root)}"
            )
        manifest_path = member / "Cargo.toml"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise RegistryError(
                f"missing ordinary member manifest: {manifest_path.relative_to(root)}"
            )
        manifest = _load_toml(manifest_path, "member manifest")
        package = manifest.get("package")
        if not isinstance(package, dict):
            raise RegistryError(f"{manifest_path} has no package table")

        _inventory_tree(root, member / "src", paths)
        _inventory_tree(root, member / "examples", paths)
        for raw_target in _target_paths(member, manifest):
            target = _member_path(root, member, raw_target.as_posix(), "target path")
            if target.is_symlink() or not target.is_file():
                raise RegistryError(
                    f"missing ordinary target source: {target.relative_to(root)}"
                )
            _inventory_tree(root, target.parent, paths)

        build_setting = package.get("build")
        if build_setting is False:
            build = None
        elif build_setting is None:
            default_build = member / "build.rs"
            build = default_build if default_build.is_file() else None
        elif isinstance(build_setting, str) and build_setting:
            build = _member_path(root, member, build_setting, "build script")
        else:
            raise RegistryError(f"{manifest_path} has invalid package.build")
        if build is not None:
            if build.is_symlink() or not build.is_file():
                raise RegistryError(
                    f"missing ordinary build script: {build.relative_to(root)}"
                )
            paths.add(build)
            build_scripts.add(build)
    return sorted(paths), build_scripts


def discover_issue_authorities(root: Path) -> dict[int, list[AuthoritySite]]:
    """Discover every production ``unimplemented_rejection!`` literal.

    Dedicated test trees and fixtures are deliberately excluded. Co-located
    ``cfg(test)`` code is conservatively counted because this lexical inventory
    does not evaluate Rust configurations. The checked-in manifest is rendered
    from this map, so a production constructor addition/removal cannot drift
    from its liveness evidence.
    """
    authorities: dict[int, list[AuthoritySite]] = {}
    paths, build_scripts = _workspace_rust_paths(root)

    for path in sorted(set(paths)):
        relative_path = path.relative_to(root)
        source = path.read_text(encoding="utf-8")
        code = _mask_rust_non_code(source)
        relative = relative_path.as_posix()
        code = _mask_owner_macro_definition(code, relative)
        build_module = BUILD_SCRIPT_MODULE.search(code)
        if path in build_scripts and build_module is not None:
            line = source.count("\n", 0, build_module.start()) + 1
            raise RegistryError(
                f"{relative}:{line}: build-script module edge is forbidden "
                "because helper modules fall outside the source-derived "
                "authority inventory; keep helper code inline or extend the "
                "inventory before adding this edge"
            )
        include = PRODUCTION_RUST_INCLUDE.search(code)
        if include is not None:
            line = source.count("\n", 0, include.start()) + 1
            raise RegistryError(
                f"{relative}:{line}: production include! is forbidden because "
                "the source-derived authority inventory does not follow include "
                "edges; move the Rust source into an ordinary module or extend "
                "the inventory before using this edge"
            )
        path_attribute = _path_attribute_offset(code)
        if path_attribute is not None:
            line = source.count("\n", 0, path_attribute) + 1
            raise RegistryError(
                f"{relative}:{line}: production path attribute is forbidden "
                "because the source-derived authority inventory does not follow "
                "out-of-tree module edges; use the ordinary module layout or "
                "extend the inventory before using this attribute"
            )
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
