#!/usr/bin/env python3
"""Rebuild exact shell dependency tags with the ecosystem canary's HEAD compiler.

Published predecessor CHB envelopes are deliberately not accepted by current
Chelis. The drift canary still uses release tags as immutable source identities,
but checks those tags out, repins their throwaway manifests to HEAD, applies the
supported v0.18-to-v0.19 source migration, builds with auto-fetch disabled,
verifies the resulting current-format artifacts, and installs those exact bytes
into the local Reef registry.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tomllib

import drift_repin_compiler


ECOSYSTEM_PACKAGES = frozenset(
    {"nautilus", "coral", "shoals", "school", "octant", "c-earchin"}
)
_SPEC_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9_.-]+/(?P<name>[A-Za-z0-9_.-]+))"
    r"@(?P<tag>v(?P<version>\d+\.\d+\.\d+))$"
)
_DEPENDENCY_LINE_RE = re.compile(
    r"^(?P<prefix>[ \t]*(?P<name>[A-Za-z0-9_.-]+)[ \t]*=[ \t]*\{)"
    r"(?P<body>[^}\n]*)(?P<suffix>\}[ \t]*(?:#.*)?)$"
)
_VERSION_FIELD_RE = re.compile(
    r'(?P<prefix>\bversion[ \t]*=[ \t]*")'
    r"(?P<version>\d+\.\d+\.\d+)"
    r'(?P<suffix>")'
)


@dataclass(frozen=True)
class DependencySpec:
    repo: str
    name: str
    tag: str
    version: str


@dataclass(frozen=True)
class Package:
    name: str
    version: str
    dependencies: frozenset[str]
    source_roots: tuple[str, ...]


def parse_dependency_spec(encoded: str) -> DependencySpec:
    match = _SPEC_RE.fullmatch(encoded)
    if match is None:
        raise ValueError(
            f"invalid dependency source `{encoded}`; expected ORG/REPO@vX.Y.Z"
        )
    return DependencySpec(
        repo=match.group("repo"),
        name=match.group("name"),
        tag=match.group("tag"),
        version=match.group("version"),
    )


def read_package(manifest_path: Path) -> Package:
    data = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    package = data.get("package")
    if not isinstance(package, dict):
        raise ValueError(f"{manifest_path}: missing [package] table")
    name = package.get("name")
    version = package.get("version")
    if not isinstance(name, str) or not isinstance(version, str):
        raise ValueError(f"{manifest_path}: package name/version must be strings")
    dependencies = data.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise ValueError(f"{manifest_path}: [dependencies] must be a table")
    additional_sources = package.get("additional_sources", [])
    if not isinstance(additional_sources, list) or not all(
        isinstance(source, str) for source in additional_sources
    ):
        raise ValueError(
            f"{manifest_path}: package.additional_sources must be strings"
        )
    return Package(
        name=name,
        version=version,
        dependencies=frozenset(str(dependency) for dependency in dependencies),
        source_roots=("src", *additional_sources),
    )


def _rewrite_dependency_versions(
    manifest_text: str,
    selected_versions: dict[str, str],
) -> str:
    parsed = tomllib.loads(manifest_text)
    dependencies = parsed.get("dependencies", {})
    if not isinstance(dependencies, dict):
        raise ValueError("[dependencies] must be a table")
    for dependency in dependencies:
        if dependency in ECOSYSTEM_PACKAGES and dependency not in selected_versions:
            raise ValueError(
                f"ecosystem dependency `{dependency}` is not selected by this "
                "canary leg"
            )
    selected_direct = set(dependencies).intersection(selected_versions)
    path_dependencies = {
        dependency
        for dependency in selected_direct
        if isinstance(dependencies[dependency], dict)
        and isinstance(dependencies[dependency].get("path"), str)
    }
    version_dependencies = selected_direct - path_dependencies
    for dependency in version_dependencies:
        value = dependencies[dependency]
        if not isinstance(value, dict) or not isinstance(value.get("version"), str):
            raise ValueError(
                f"dependency `{dependency}` must carry an inline version field "
                "or an explicit path"
            )

    section = ""
    rewritten: list[str] = []
    changed_dependencies: set[str] = set()
    for line in manifest_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
        if section != "[dependencies]":
            rewritten.append(line)
            continue
        line_without_newline = line.rstrip("\r\n")
        newline = line[len(line_without_newline) :]
        match = _DEPENDENCY_LINE_RE.fullmatch(line_without_newline)
        if match is None or match.group("name") not in selected_versions:
            rewritten.append(line)
            continue
        dependency = match.group("name")
        if dependency in path_dependencies:
            rewritten.append(line)
            continue
        body, replacements = _VERSION_FIELD_RE.subn(
            lambda version_match: (
                f'{version_match.group("prefix")}'
                f"{selected_versions[dependency]}"
                f'{version_match.group("suffix")}'
            ),
            match.group("body"),
        )
        if replacements != 1:
            raise ValueError(
                f"dependency `{dependency}` must carry one inline version field"
            )
        rewritten.append(
            f'{match.group("prefix")}{body}{match.group("suffix")}{newline}'
        )
        changed_dependencies.add(dependency)

    if changed_dependencies != version_dependencies:
        missing = sorted(version_dependencies - changed_dependencies)
        raise ValueError(
            "could not rewrite selected dependency declarations: "
            + ", ".join(missing)
        )
    result = "".join(rewritten)
    reparsed = tomllib.loads(result).get("dependencies", {})
    for dependency in version_dependencies:
        value = reparsed[dependency]
        if not isinstance(value, dict) or value.get("version") != selected_versions[
            dependency
        ]:
            raise ValueError(
                f"dependency `{dependency}` did not retain the selected version"
            )
    return result


def rewrite_manifest(
    manifest_text: str,
    compiler_version: str,
    selected_versions: dict[str, str],
) -> str:
    repinned = drift_repin_compiler.repin(manifest_text, compiler_version)
    return _rewrite_dependency_versions(repinned, selected_versions)


def _rewrite_tree(
    root: Path,
    compiler_version: str,
    selected_versions: dict[str, str],
) -> None:
    repinned, _ = drift_repin_compiler.repin_tree(root, compiler_version)
    if repinned == 0:
        raise ValueError(f"{root}: no compiler pin was repinned")
    for manifest_path in sorted(root.rglob("reef.toml")):
        if not manifest_path.is_file():
            continue
        original = manifest_path.read_text(encoding="utf-8")
        rewritten = _rewrite_dependency_versions(original, selected_versions)
        manifest_path.write_text(rewritten, encoding="utf-8")


def _dependency_order(
    packages: dict[str, tuple[DependencySpec, Path, Package]],
) -> list[str]:
    selected = set(packages)
    remaining = set(packages)
    built: set[str] = set()
    order: list[str] = []
    while remaining:
        ready = sorted(
            name
            for name in remaining
            if packages[name][2].dependencies.intersection(selected) <= built
        )
        if not ready:
            raise ValueError(
                "selected dependency graph contains a cycle: "
                + ", ".join(sorted(remaining))
            )
        for name in ready:
            order.append(name)
            built.add(name)
            remaining.remove(name)
    return order


def _clone_repo(repo: str, tag: str, destination: Path) -> None:
    subprocess.run(
        [
            "gh",
            "repo",
            "clone",
            repo,
            str(destination),
            "--",
            "--branch",
            tag,
            "--depth",
            "1",
            "--single-branch",
        ],
        check=True,
    )


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _source_paths(package_root: Path, package: Package) -> list[Path]:
    return sorted(
        path
        for source_root in package.source_roots
        for path in (package_root / source_root).rglob("*.ch")
        if path.is_file()
    )


def prepare_dependencies(
    *,
    workspace: Path,
    shell: Path,
    compiler_version: str,
    encoded_specs: Sequence[str],
    clone: Callable[[str, str, Path], None] = _clone_repo,
    run: Callable[[list[str]], None] = _run,
    chelis: str = "chelis",
) -> None:
    specs = [parse_dependency_spec(encoded) for encoded in encoded_specs]
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("dependency source package names must be unique")
    selected_versions = {spec.name: spec.version for spec in specs}

    packages_root = workspace / "packages"
    packages_root.mkdir(parents=True, exist_ok=False)
    packages: dict[str, tuple[DependencySpec, Path, Package]] = {}
    for spec in specs:
        package_root = packages_root / spec.name
        clone(spec.repo, spec.tag, package_root)
        _rewrite_tree(package_root, compiler_version, selected_versions)
        package = read_package(package_root / "reef.toml")
        if package.name != spec.name:
            raise ValueError(
                f"{spec.repo}@{spec.tag}: manifest package is `{package.name}`, "
                f"expected `{spec.name}`"
            )
        if package.version != spec.version:
            raise ValueError(
                f"{spec.repo} tag `{spec.tag}` does not match manifest version "
                f"`{package.version}`"
            )
        packages[spec.name] = (spec, package_root, package)

    _rewrite_tree(shell, compiler_version, selected_versions)

    for name in _dependency_order(packages):
        _, package_root, package = packages[name]
        source_paths = _source_paths(package_root, package)
        if source_paths:
            run(
                [
                    chelis,
                    "migrate",
                    "surf",
                    "--from",
                    "0.18",
                    "--inplace",
                    *(str(path) for path in source_paths),
                ]
            )
        run([chelis, "reef", "build", "--no-auto-fetch", str(package_root)])
        shell_artifact = package_root / "dist" / f"{name}-{package.version}.chb"
        archive_artifact = (
            package_root / "dist" / f"{name}-{package.version}.tar.zst"
        )
        if not shell_artifact.is_file() or not archive_artifact.is_file():
            raise ValueError(
                f"{name}: build did not produce the expected CHB/archive pair"
            )
        run(
            [
                chelis,
                "reef",
                "verify-artifact",
                "--archive",
                str(archive_artifact),
                "--shell",
                str(shell_artifact),
            ]
        )
        run(
            [
                chelis,
                "reef",
                "install",
                "--from-monorepo",
                str(workspace),
                name,
            ]
        )


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--shell", required=True, type=Path)
    parser.add_argument("--compiler-version", required=True)
    parser.add_argument("dependencies", nargs="+")
    args = parser.parse_args(argv[1:])
    try:
        prepare_dependencies(
            workspace=args.workspace,
            shell=args.shell,
            compiler_version=args.compiler_version,
            encoded_specs=args.dependencies,
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"dependency preparation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"prepared {len(args.dependencies)} exact source dependency bundle(s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
