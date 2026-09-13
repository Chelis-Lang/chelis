#!/usr/bin/env python3
"""Generate the [05-UNS-5] atom and issue membership artifacts.

The atom input is derived from normative blockquote definitions in numbered
specs. The issue manifest is derived from exact literal
``unimplemented_rejection!`` citations in parser-confirmed production crate
module graphs. Its rows are separately validated against GitHub by
``validate_rejection_issue_manifest.py``.

Run with the uv-managed interpreter:

    .venv/bin/python scripts/generate_rejection_registries.py --check
    .venv/bin/python scripts/generate_rejection_registries.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from check_configuration_closure import parse_dep_info

ATOM = re.compile(r"^> \*\*(\[[0-9]{2}-[A-Z]+-[1-9][0-9]*\])\*\*", re.MULTILINE)
NUMBERED_SPEC = re.compile(r"^(?:0[0-9]|1[0-2])-[^/]+\.md$")
EXACT_ISSUE_LITERAL = re.compile(r"[0-9][0-9_]*")
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


@dataclass(frozen=True)
class ProductionSource:
    """One parser-confirmed source in a production crate target graph."""

    path: Path
    source: str


@dataclass(frozen=True)
class ProductionWorkspace:
    """Cargo-owned package and target identities for production scanning."""

    package_ids: frozenset[str]
    target_roots: tuple[str, ...]


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


PRODUCTION_TARGET_KINDS = frozenset(
    {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro", "bin"}
)


def production_workspace_from_metadata(
    root: Path, metadata: object
) -> ProductionWorkspace:
    """Select repository-local workspace packages and production target roots."""
    if not isinstance(metadata, dict):
        raise RegistryError("cargo metadata output is not an object")
    packages = metadata.get("packages")
    members = metadata.get("workspace_members")
    if not isinstance(packages, list) or not isinstance(members, list):
        raise RegistryError("cargo metadata lacks packages or workspace_members")
    member_ids = set(members)
    package_ids: set[str] = set()
    roots: set[str] = set()
    canonical_root = root.resolve()
    for package in packages:
        if not isinstance(package, dict):
            raise RegistryError("cargo metadata has an invalid package")
        package_id = package.get("id")
        if package_id not in member_ids:
            continue
        manifest = package.get("manifest_path")
        if not isinstance(package_id, str) or not isinstance(manifest, str):
            raise RegistryError("workspace package has invalid id or manifest_path")
        try:
            Path(manifest).resolve().relative_to(canonical_root)
        except ValueError:
            continue
        package_ids.add(package_id)
        targets = package.get("targets")
        if not isinstance(targets, list):
            raise RegistryError("workspace package has invalid targets")
        for target in targets:
            if not isinstance(target, dict):
                raise RegistryError("workspace package has an invalid target")
            kinds = target.get("kind")
            source = target.get("src_path")
            if (
                not isinstance(kinds, list)
                or not all(isinstance(kind, str) for kind in kinds)
                or not isinstance(source, str)
            ):
                raise RegistryError("workspace target has invalid kind or src_path")
            if not PRODUCTION_TARGET_KINDS.intersection(kinds):
                continue
            path = Path(source).resolve()
            try:
                relative = path.relative_to(canonical_root)
            except ValueError as error:
                raise RegistryError(
                    f"production target root resolves outside repository: {source}"
                ) from error
            roots.add(relative.as_posix())
    if not package_ids:
        raise RegistryError("cargo metadata names no repository-local workspace packages")
    if not roots:
        raise RegistryError("cargo metadata names no production workspace target roots")
    return ProductionWorkspace(
        package_ids=frozenset(package_ids),
        target_roots=tuple(sorted(roots)),
    )


def production_target_roots_from_metadata(
    root: Path, metadata: object
) -> list[str]:
    """Compatibility wrapper for the Cargo-owned production target roots."""
    return list(production_workspace_from_metadata(root, metadata).target_roots)


def discover_production_workspace(root: Path) -> ProductionWorkspace:
    """Ask Cargo for repository-local package and production-target identities."""
    completed = subprocess.run(
        ("cargo", "metadata", "--no-deps", "--format-version", "1"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryError(
            f"cargo metadata failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RegistryError(f"cargo metadata returned invalid JSON: {error}") from error
    return production_workspace_from_metadata(root, metadata)


def discover_production_target_roots(root: Path) -> list[str]:
    """Ask Cargo for the actual workspace library and binary roots."""
    return list(discover_production_workspace(root).target_roots)


def compiler_source_closure_problems(
    parser_sources: set[str], compiler_sources: set[str]
) -> list[str]:
    """Name compiler-read production files absent from the parser graph."""
    excluded = {path.as_posix() for path in PRODUCTION_SOURCE_EXCLUSIONS}
    return sorted(compiler_sources - parser_sources - excluded)


def production_dep_info_files(
    messages: list[object], workspace: ProductionWorkspace
) -> set[Path]:
    """Select dep-info for non-test artifacts in the shared Cargo package set."""
    dep_info_files: set[Path] = set()
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("reason") != "compiler-artifact"
            or message.get("package_id") not in workspace.package_ids
        ):
            continue
        profile = message.get("profile")
        target = message.get("target")
        filenames = message.get("filenames")
        if (
            not isinstance(profile, dict)
            or profile.get("test") is not False
            or not isinstance(target, dict)
            or not isinstance(target.get("kind"), list)
            or not PRODUCTION_TARGET_KINDS.intersection(target["kind"])
            or not isinstance(filenames, list)
        ):
            continue
        candidates: set[Path] = set()
        for filename in filenames:
            if not isinstance(filename, str):
                continue
            artifact = Path(filename)
            stem = artifact.stem.removeprefix("lib")
            candidate = artifact.with_name(f"{stem}.d")
            if candidate.is_file():
                candidates.add(candidate)
        if not candidates:
            raise RegistryError(
                f"cargo artifact `{target.get('name', '?')}` has no readable dep-info"
            )
        dep_info_files.update(candidates)
    return dep_info_files


def discover_compiler_production_sources(
    root: Path, workspace: ProductionWorkspace | None = None
) -> set[str]:
    """Ask rustc dep-info which files default production targets read.

    This is the exact mechanism owned by ``check_configuration_closure.py``.
    Cargo JSON selects only non-test artifacts for the same repository-local
    package identity set that owns the structural target roots; their matching
    dep-info files independently supply the compiler-read source closure.
    """
    workspace = workspace or discover_production_workspace(root)
    canonical_root = root.resolve()

    environment = dict(os.environ)
    environment["PYO3_PYTHON"] = sys.executable
    completed = subprocess.run(
        (
            "cargo",
            "check",
            "--workspace",
            "--lib",
            "--bins",
            "--message-format=json",
        ),
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryError(
            f"cargo production source inventory failed with exit "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )

    messages: list[object] = []
    for line in completed.stdout.splitlines():
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    dep_info_files = production_dep_info_files(messages, workspace)
    if not dep_info_files:
        raise RegistryError("cargo reported no production workspace dep-info")

    sources: set[str] = set()
    for dep_info in dep_info_files:
        for raw_path in parse_dep_info(dep_info):
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = root / candidate
            try:
                relative = candidate.resolve().relative_to(canonical_root)
            except (OSError, ValueError):
                continue
            if relative.suffix == ".rs":
                sources.add(relative.as_posix())
    return sources


def discover_production_sources(
    root: Path, workspace: ProductionWorkspace | None = None
) -> list[ProductionSource]:
    """Return parser-confirmed sources reachable from production Cargo targets.

    Cargo owns target-root discovery. The existing ``syn`` inventory owner
    follows external and inline module edges, excludes items that cannot exist
    with ``cfg(test)`` false, follows literal ``#[path]`` edges inside the
    repository, and fails closed on ambiguous or unsupported module wiring.
    """
    workspace = workspace or discover_production_workspace(root)
    completed = subprocess.run(
        (
            "cargo",
            "run",
            "--quiet",
            "-p",
            "chelis-repr-inventory",
            "--bin",
            "rejection_source_inventory",
            "--",
            "--repo",
            str(root),
        ),
        cwd=root,
        input=json.dumps(workspace.target_roots),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RegistryError(
            f"production source inventory failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RegistryError(
            f"production source inventory returned invalid JSON: {error}"
        ) from error
    if not isinstance(payload, list):
        raise RegistryError("production source inventory is not a list")
    sources: list[ProductionSource] = []
    for index, row in enumerate(payload):
        if (
            not isinstance(row, dict)
            or set(row) != {"path", "source"}
            or not isinstance(row["path"], str)
            or not isinstance(row["source"], str)
        ):
            raise RegistryError(f"production source row {index} is malformed")
        path = Path(row["path"])
        if path in PRODUCTION_SOURCE_EXCLUSIONS:
            continue
        sources.append(ProductionSource(path=path, source=row["source"]))
    if [source.path.as_posix() for source in sources] != sorted(
        source.path.as_posix() for source in sources
    ):
        raise RegistryError("production source inventory is not sorted")
    return sources


def verify_compiler_source_closure(
    root: Path,
    sources: list[ProductionSource],
    workspace: ProductionWorkspace | None = None,
) -> None:
    """Require the parser graph to contain every rustc-read production file."""
    parser_paths = {source.path.as_posix() for source in sources}
    missing = compiler_source_closure_problems(
        parser_paths, discover_compiler_production_sources(root, workspace)
    )
    if missing:
        raise RegistryError(
            "rustc dep-info found production sources absent from the parser "
            f"module graph: {missing}"
        )


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
            if number == 0 or number > 0xFFFF_FFFF:
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
    return parse_production_issue_citations(discover_production_sources(root))


def parse_production_issue_citations(
    sources: list[ProductionSource],
) -> list[IssueCitation]:
    """Parse every citation from one already-derived production graph."""
    citations: list[IssueCitation] = []
    for source in sources:
        citations.extend(parse_issue_citations(source.path, source.source))
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
    workspace = discover_production_workspace(root)
    sources = discover_production_sources(root, workspace)
    citations = parse_production_issue_citations(sources)
    issues = derive_issue_numbers(citations)
    rendered_manifest = render_issue_manifest(issues)
    rendered_registry = render_registry(discover_atoms(root / "spec"), issues)
    manifest = root / MANIFEST_REL
    output = root / OUTPUT_REL
    if args.write:
        manifest.write_text(rendered_manifest)
        output.write_text(rendered_registry)
        verify_compiler_source_closure(root, sources, workspace)
        print(f"wrote {manifest.relative_to(root)}")
        print(f"wrote {output.relative_to(root)}")
        return 0
    verify_compiler_source_closure(root, sources, workspace)
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
