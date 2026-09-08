#!/usr/bin/env python3
"""Enforce the dependency boundary for chelis-pipeline-core."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from typing import Callable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_PACKAGE = "chelis-pipeline-core"
CORE_MANIFEST = REPO_ROOT / "crates" / CORE_PACKAGE / "Cargo.toml"
APPROVED_DIRECT_DEPENDENCIES = frozenset(
    {"chelis-deep", "chelis-types", "chelis-effects", "chelis-ir", "chelis-unord"}
)
APPROVED_WORKSPACE_CLOSURE = frozenset(
    {CORE_PACKAGE, *APPROVED_DIRECT_DEPENDENCIES, "chelis-pred", "chelis-vocab"}
)
DENIED_PACKAGES = frozenset(
    {
        "chelis-compiler-api",
        "chelis-reef",
        "chelis-surf",
        "chelis-macros",
        "chelis-shell",
        "chelis-std-bundle",
    }
)
DENIED_PREFIXES = ("chelis-backend-",)


class DependencyBoundaryError(RuntimeError):
    """The core dependency graph crosses its approved boundary."""


@dataclass(frozen=True)
class ManifestDependencies:
    names: frozenset[str]


@dataclass(frozen=True)
class DependencyGraph:
    edges: Mapping[str, tuple[str, ...]]
    workspace_packages: frozenset[str]

    def path_to_denied(self, start: str) -> tuple[str, ...] | None:
        return self.path_to(start, is_denied)

    def path_to_unapproved_workspace(self, start: str) -> tuple[str, ...] | None:
        return self.path_to(
            start,
            lambda package: package in self.workspace_packages
            and package not in APPROVED_WORKSPACE_CLOSURE,
        )

    def path_to(
        self, start: str, matches: Callable[[str], bool]
    ) -> tuple[str, ...] | None:
        queue = deque([(start, (start,))])
        visited = {start}
        while queue:
            current, path = queue.popleft()
            for dependency in self.edges.get(current, ()):
                next_path = (*path, dependency)
                if dependency != start and matches(dependency):
                    return next_path
                if dependency not in visited:
                    visited.add(dependency)
                    queue.append((dependency, next_path))
        return None


def is_denied(package: str) -> bool:
    return package in DENIED_PACKAGES or package.startswith(DENIED_PREFIXES)


def dependency_name(visible_name: str, value: object) -> str:
    if isinstance(value, dict):
        package = value.get("package")
        if isinstance(package, str):
            return package
    return visible_name


def parse_manifest(text: str) -> ManifestDependencies:
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise DependencyBoundaryError(f"core manifest is not valid TOML: {error}") from error

    names: set[str] = set()
    dependency_tables: list[object] = [
        document.get("dependencies", {}),
        document.get("build-dependencies", {}),
    ]
    target_tables = document.get("target", {})
    if isinstance(target_tables, dict):
        for target in target_tables.values():
            if isinstance(target, dict):
                dependency_tables.extend(
                    (
                        target.get("dependencies", {}),
                        target.get("build-dependencies", {}),
                    )
                )

    for table in dependency_tables:
        if not isinstance(table, dict):
            raise DependencyBoundaryError("a production dependency table is not a TOML table")
        names.update(dependency_name(name, value) for name, value in table.items())
    return ManifestDependencies(frozenset(names))


def parse_fixture_graph(raw: Mapping[str, Sequence[str]]) -> DependencyGraph:
    edges: dict[str, tuple[str, ...]] = {}
    for package, dependencies in raw.items():
        if not isinstance(package, str) or isinstance(dependencies, (str, bytes)):
            raise DependencyBoundaryError("the dependency graph fixture has an invalid row")
        if not all(isinstance(dependency, str) for dependency in dependencies):
            raise DependencyBoundaryError(
                f"the dependency graph fixture has an invalid edge from `{package}`"
            )
        edges[package] = tuple(dependencies)
    return DependencyGraph(edges, frozenset(edges))


def parse_metadata(raw: object) -> DependencyGraph:
    if not isinstance(raw, dict):
        raise DependencyBoundaryError("cargo metadata did not return an object")
    packages = raw.get("packages")
    workspace_members = raw.get("workspace_members")
    resolve = raw.get("resolve")
    if (
        not isinstance(packages, list)
        or not isinstance(workspace_members, list)
        or not isinstance(resolve, dict)
    ):
        raise DependencyBoundaryError(
            "cargo metadata lacks packages, workspace members, or a resolved graph"
        )
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list):
        raise DependencyBoundaryError("cargo metadata lacks resolved graph nodes")

    names_by_id: dict[str, str] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise DependencyBoundaryError("cargo metadata contains an invalid package")
        package_id = package.get("id")
        name = package.get("name")
        if not isinstance(package_id, str) or not isinstance(name, str):
            raise DependencyBoundaryError("cargo metadata contains an invalid package identity")
        names_by_id[package_id] = name

    workspace_packages: set[str] = set()
    for package_id in workspace_members:
        if not isinstance(package_id, str) or package_id not in names_by_id:
            raise DependencyBoundaryError(
                "cargo metadata contains an unknown workspace member"
            )
        workspace_packages.add(names_by_id[package_id])

    edges: dict[str, set[str]] = {}
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise DependencyBoundaryError("cargo metadata contains an invalid graph node")
        package_id = node["id"]
        package_name = names_by_id.get(package_id)
        if package_name is None:
            raise DependencyBoundaryError(
                f"cargo metadata has no package for graph node `{package_id}`"
            )
        dependencies = node.get("deps")
        if not isinstance(dependencies, list):
            raise DependencyBoundaryError(
                f"cargo metadata has invalid dependency details for `{package_name}`"
            )
        resolved_names = edges.setdefault(package_name, set())
        for dependency in dependencies:
            if not isinstance(dependency, dict):
                raise DependencyBoundaryError(
                    f"cargo metadata has an invalid dependency from `{package_name}`"
                )
            dependency_id = dependency.get("pkg")
            kinds = dependency.get("dep_kinds")
            if not isinstance(dependency_id, str) or dependency_id not in names_by_id:
                raise DependencyBoundaryError(
                    f"cargo metadata has an unknown dependency from `{package_name}`"
                )
            if not isinstance(kinds, list) or not kinds:
                raise DependencyBoundaryError(
                    f"cargo metadata lacks dependency kinds from `{package_name}`"
                )
            production_edge = False
            for kind in kinds:
                if not isinstance(kind, dict):
                    raise DependencyBoundaryError(
                        f"cargo metadata has an invalid dependency kind from `{package_name}`"
                    )
                value = kind.get("kind")
                if value is not None and not isinstance(value, str):
                    raise DependencyBoundaryError(
                        f"cargo metadata has an invalid dependency kind from `{package_name}`"
                    )
                if value != "dev":
                    production_edge = True
            if production_edge:
                resolved_names.add(names_by_id[dependency_id])

    return DependencyGraph(
        {package: tuple(sorted(dependencies)) for package, dependencies in edges.items()},
        frozenset(workspace_packages),
    )


def validate_manifest(manifest: ManifestDependencies) -> None:
    for dependency in sorted(manifest.names - APPROVED_DIRECT_DEPENDENCIES):
        if is_denied(dependency):
            raise DependencyBoundaryError(
                f"core has forbidden direct dependency `{dependency}`"
            )
        raise DependencyBoundaryError(f"core has unknown dependency `{dependency}`")

    missing = sorted(APPROVED_DIRECT_DEPENDENCIES - manifest.names)
    if missing:
        raise DependencyBoundaryError(
            "core manifest lacks approved dependencies: " + ", ".join(missing)
        )


def validate_graph(graph: DependencyGraph) -> None:
    if CORE_PACKAGE not in graph.edges:
        raise DependencyBoundaryError(
            f"resolved graph does not contain `{CORE_PACKAGE}`"
        )
    path = graph.path_to_denied(CORE_PACKAGE)
    if path is not None:
        raise DependencyBoundaryError(
            "forbidden dependency path: " + " -> ".join(path)
        )
    path = graph.path_to_unapproved_workspace(CORE_PACKAGE)
    if path is not None:
        raise DependencyBoundaryError(
            "unapproved workspace dependency path: " + " -> ".join(path)
        )


def validate_fixture(
    manifest_text: str, graph_rows: Mapping[str, Sequence[str]]
) -> None:
    manifest = parse_manifest(manifest_text)
    graph = parse_fixture_graph(graph_rows)
    validate_manifest(manifest)
    validate_graph(graph)


def cargo_metadata() -> DependencyGraph:
    completed = subprocess.run(
        ("cargo", "metadata", "--format-version", "1", "--locked"),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise DependencyBoundaryError(
            f"cargo metadata failed with exit {completed.returncode}: {detail}"
        )
    try:
        raw = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise DependencyBoundaryError(f"cargo metadata returned invalid JSON: {error}") from error
    return parse_metadata(raw)


def validate_repository() -> None:
    try:
        manifest_text = CORE_MANIFEST.read_text(encoding="utf-8")
    except OSError as error:
        raise DependencyBoundaryError(f"cannot read {CORE_MANIFEST}: {error}") from error
    validate_manifest(parse_manifest(manifest_text))
    validate_graph(cargo_metadata())


def main() -> int:
    try:
        validate_repository()
    except DependencyBoundaryError as error:
        print(f"pipeline core dependency guard: FAIL: {error}", file=sys.stderr)
        return 1
    print("pipeline core dependency guard: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
