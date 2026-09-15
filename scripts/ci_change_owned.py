#!/usr/bin/env python3
"""Plan and execute package-qualified change-owned integration tests.

The planner classifies every changed path, compares exact base/candidate Cargo
metadata, and emits two disjoint four-shard selections:

* ``change-owned`` is required and contains added/directly modified eligible
  integration targets.
* ``package-expansion`` is informational and contains the remaining eligible
  targets in directly changed or explicitly selected packages.

Every persisted plan and shard receipt is content-digested. Reports reject
missing, duplicate, mismatched, excluded, uncovered, or unsuccessful required
execution. The informational report records the same defects without returning
failure.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

if __package__:
    from .ci_detect_docs_only import is_docs_only
else:
    from ci_detect_docs_only import is_docs_only


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 2
PLAN_VERSION = 2
RECEIPT_VERSION = 1
STANDING_COVERAGE_VERSION = 1
SHARDS = tuple(range(4))
LANE_KEYS = {
    "change-owned": "change_owned",
    "package-expansion": "package_expansion",
}
IDENTIFIER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
ISSUE = re.compile(r"chelis#[1-9][0-9]*\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
TEST_FUNCTION = re.compile(
    r"#\s*\[\s*(?:[A-Za-z0-9_:]+\s*)?test(?:\s*\([^]]*\))?\s*\]"
    r"(?:\s*#\s*\[[^]]*\])*\s*(?:pub(?:\([^)]*\))?\s+)?"
    r"(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
OWNER_FIELDS = {"workflow", "job", "cadence", "reason", "tracking_issue"}
SIDECAR_NAMES = ("commands.json", "timings.json", "test-list.json", "junit.xml")
SOFT_BUDGET_SECONDS = 15 * 60
EXPANSION_EXECUTION_SECONDS = 16 * 60
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
STANDING_EXECUTION = {
    "profile": "ci-fast",
    "ignore_default_filter": True,
    "run_ignored": "default",
    "no_fail_fast": True,
}


@dataclass(frozen=True, order=True)
class Identity:
    package: str
    target: str

    @property
    def canonical(self) -> str:
        return f"{self.package}::{self.target}"

    @classmethod
    def parse(cls, value: str) -> Identity:
        parts = value.split("::")
        if len(parts) != 2 or not all(IDENTIFIER.fullmatch(part) for part in parts):
            raise ValueError(f"invalid package-qualified target identity: {value!r}")
        return cls(*parts)


@dataclass(frozen=True, order=True)
class TestIdentity:
    package: str
    target: str
    test: str

    @property
    def target_identity(self) -> Identity:
        return Identity(self.package, self.target)

    @property
    def canonical(self) -> str:
        return f"{self.package}::{self.target}::{self.test}"

    @classmethod
    def parse(cls, value: str) -> TestIdentity:
        parts = value.split("::", 2)
        if (
            len(parts) != 3
            or not all(IDENTIFIER.fullmatch(part) for part in parts[:2])
            or not parts[2]
            or any(character in parts[2] for character in "\0\r\n")
        ):
            raise ValueError(f"invalid exact test identity: {value!r}")
        return cls(*parts)


@dataclass(frozen=True)
class Owner:
    workflow: str
    job: str
    cadence: str
    reason: str
    tracking_issue: str

    def as_dict(self) -> dict[str, str]:
        return {
            "workflow": self.workflow,
            "job": self.job,
            "cadence": self.cadence,
            "reason": self.reason,
            "tracking_issue": self.tracking_issue,
        }


@dataclass(frozen=True)
class PathRule:
    prefix: str
    disposition: str
    packages: tuple[str, ...] = ()
    owner: Owner | None = None

    def matches(self, path: str) -> bool:
        if self.prefix.endswith("/"):
            return path.startswith(self.prefix)
        return path == self.prefix


@dataclass(frozen=True)
class Config:
    version: int
    standing_targets: tuple[Identity, ...]
    manual_only_targets: Mapping[Identity, Owner]
    target_exclusions: Mapping[Identity, Owner]
    test_exclusions: Mapping[TestIdentity, Owner]
    path_rules: tuple[PathRule, ...]


@dataclass(frozen=True)
class TargetInfo:
    identity: Identity
    src_path: str
    required_features: tuple[str, ...]


@dataclass(frozen=True)
class PackageInfo:
    name: str
    root: str


@dataclass(frozen=True)
class ChangeRecord:
    status: str
    path: str
    old_path: str | None = None


def _nonempty_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value


def _identifier(row: Mapping[str, Any], key: str) -> str:
    value = _nonempty_string(row, key)
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {key}: {value!r}")
    return value


def _owner(row: Mapping[str, Any]) -> Owner:
    values = {key: _nonempty_string(row, key) for key in OWNER_FIELDS}
    if not values["workflow"].endswith((".yml", ".yaml")):
        raise ValueError("owner workflow must name a workflow YAML file")
    if not ISSUE.fullmatch(values["tracking_issue"]):
        raise ValueError(
            "owner tracking_issue must be chelis#N: "
            f"{values['tracking_issue']!r}"
        )
    return Owner(**values)


def _rows(data: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = data.get(key, [])
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{key} must be an array of tables")
    return value


def _valid_prefix(prefix: str) -> bool:
    if (
        prefix.startswith("/")
        or "\\" in prefix
        or prefix in {"", ".", "./"}
        or "//" in prefix
    ):
        return False
    parts = PurePosixPath(prefix.rstrip("/")).parts
    return bool(parts) and all(part not in {"", ".", ".."} for part in parts)


def _rules_overlap(left: str, right: str) -> bool:
    if left == right:
        return True
    return (left.endswith("/") and right.startswith(left)) or (
        right.endswith("/") and left.startswith(right)
    )


def read_config(path: Path) -> Config:
    """Read the strict version-2 five-row-kind configuration."""
    data = tomllib.loads(path.read_text())
    allowed = {
        "version",
        "standing_target",
        "manual_only_target",
        "target_exclusion",
        "test_exclusion",
        "path_rule",
    }
    if set(data) - allowed:
        raise ValueError(f"unknown configuration keys: {sorted(set(data) - allowed)}")
    if type(data.get("version")) is not int or data["version"] != SCHEMA_VERSION:
        raise ValueError(f"configuration requires version = {SCHEMA_VERSION}")

    standing: list[Identity] = []
    for row in _rows(data, "standing_target"):
        if set(row) != {"package", "name"}:
            raise ValueError("standing_target must name exactly package and name")
        standing.append(Identity(_identifier(row, "package"), _identifier(row, "name")))
    if not standing:
        raise ValueError("configuration requires at least one standing_target")
    if len(set(standing)) != len(standing):
        raise ValueError("duplicate standing_target identity")

    manual_only_targets: dict[Identity, Owner] = {}
    for row in _rows(data, "manual_only_target"):
        if set(row) != {"package", "name"} | OWNER_FIELDS:
            raise ValueError(
                "manual_only_target requires package, name, and exact owner fields"
            )
        identity = Identity(_identifier(row, "package"), _identifier(row, "name"))
        if identity in manual_only_targets:
            raise ValueError(f"duplicate manual_only_target: {identity.canonical}")
        manual_only_targets[identity] = _owner(row)

    target_exclusions: dict[Identity, Owner] = {}
    for row in _rows(data, "target_exclusion"):
        if set(row) != {"package", "name"} | OWNER_FIELDS:
            raise ValueError(
                "target_exclusion requires package, name, and exact owner fields"
            )
        identity = Identity(_identifier(row, "package"), _identifier(row, "name"))
        if identity in target_exclusions:
            raise ValueError(f"duplicate target_exclusion: {identity.canonical}")
        target_exclusions[identity] = _owner(row)

    test_exclusions: dict[TestIdentity, Owner] = {}
    for row in _rows(data, "test_exclusion"):
        if set(row) != {"package", "target", "name"} | OWNER_FIELDS:
            raise ValueError(
                "test_exclusion requires package, target, name, and exact owner fields"
            )
        identity = TestIdentity(
            _identifier(row, "package"),
            _identifier(row, "target"),
            _identifier(row, "name"),
        )
        if identity in test_exclusions:
            raise ValueError(f"duplicate test_exclusion: {identity.canonical}")
        test_exclusions[identity] = _owner(row)

    if set(standing) & set(target_exclusions):
        overlap = sorted(item.canonical for item in set(standing) & set(target_exclusions))
        raise ValueError(f"standing targets cannot be target exclusions: {overlap}")
    manual_conflicts = sorted(
        item.canonical
        for item in set(manual_only_targets)
        & (set(standing) | set(target_exclusions))
    )
    if manual_conflicts:
        raise ValueError(
            "manual-only targets cannot be standing targets or target exclusions: "
            f"{manual_conflicts}"
        )
    contradictory_tests = sorted(
        identity.canonical
        for identity in test_exclusions
        if identity.target_identity in target_exclusions
    )
    if contradictory_tests:
        raise ValueError(
            f"test exclusions cannot sit under target exclusions: {contradictory_tests}"
        )
    manual_test_exclusions = sorted(
        identity.canonical
        for identity in test_exclusions
        if identity.target_identity in manual_only_targets
    )
    if manual_test_exclusions:
        raise ValueError(
            "manual-only targets must execute their complete ignored suite; "
            f"test exclusions are forbidden: {manual_test_exclusions}"
        )

    path_rules: list[PathRule] = []
    for row in _rows(data, "path_rule"):
        disposition = row.get("disposition")
        if disposition == "packages":
            if set(row) != {"prefix", "disposition", "packages"}:
                raise ValueError(
                    "package path_rule requires exactly prefix, disposition, packages"
                )
            raw_packages = row["packages"]
            if (
                not isinstance(raw_packages, list)
                or not raw_packages
                or any(
                    not isinstance(package, str)
                    or not IDENTIFIER.fullmatch(package)
                    for package in raw_packages
                )
                or len(set(raw_packages)) != len(raw_packages)
            ):
                raise ValueError("path_rule packages must be unique exact package names")
            packages = tuple(raw_packages)
            owner = None
        elif disposition == "owner":
            if set(row) != {"prefix", "disposition"} | OWNER_FIELDS:
                raise ValueError(
                    "owner path_rule requires prefix, disposition, and owner fields"
                )
            packages = ()
            owner = _owner(row)
        else:
            raise ValueError(
                "path_rule disposition must be packages or owner"
            )
        prefix = _nonempty_string(row, "prefix")
        if not _valid_prefix(prefix):
            raise ValueError(f"invalid path_rule prefix: {prefix!r}")
        for existing in path_rules:
            if _rules_overlap(existing.prefix, prefix):
                raise ValueError(
                    f"ambiguous path rules overlap: {existing.prefix!r}, {prefix!r}"
                )
        path_rules.append(PathRule(prefix, disposition, packages, owner))

    return Config(
        SCHEMA_VERSION,
        tuple(standing),
        manual_only_targets,
        target_exclusions,
        test_exclusions,
        tuple(path_rules),
    )


def _workspace_packages(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    members = set(metadata["workspace_members"])
    packages = [package for package in metadata["packages"] if package["id"] in members]
    if not members or len(packages) != len(members):
        raise ValueError("Cargo metadata does not enumerate the complete workspace")
    names = [package["name"] for package in packages]
    if len(set(names)) != len(names):
        raise ValueError("Cargo metadata contains duplicate workspace package names")
    return packages


def _relative(path: str, root: str) -> str:
    try:
        relative = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError as error:
        raise ValueError(f"metadata path escapes workspace: {path}") from error
    return relative.as_posix()


def default_features(package: Mapping[str, Any]) -> set[str]:
    """Return the package-local named feature closure enabled by default."""
    features = package.get("features", {})
    if not isinstance(features, dict):
        raise ValueError(f"malformed feature map for {package.get('name')}")
    enabled: set[str] = set()
    pending = ["default"]
    while pending:
        feature = pending.pop()
        if feature in enabled:
            continue
        enabled.add(feature)
        for successor in features.get(feature, []):
            if not isinstance(successor, str):
                raise ValueError(f"malformed feature edge in {package.get('name')}")
            if successor in features:
                pending.append(successor)
    return enabled


def integration_targets(metadata: Mapping[str, Any]) -> dict[Identity, TargetInfo]:
    root = metadata["workspace_root"]
    result: dict[Identity, TargetInfo] = {}
    for package in _workspace_packages(metadata):
        enabled = default_features(package)
        for target in package["targets"]:
            if "test" not in target.get("kind", []):
                continue
            required = tuple(target.get("required-features", []))
            if not set(required) <= enabled:
                continue
            identity = Identity(package["name"], target["name"])
            if identity in result:
                raise ValueError(f"duplicate Cargo target identity: {identity.canonical}")
            result[identity] = TargetInfo(
                identity,
                _relative(target["src_path"], root),
                required,
            )
    return result


def all_integration_targets(metadata: Mapping[str, Any]) -> dict[Identity, TargetInfo]:
    root = metadata["workspace_root"]
    result: dict[Identity, TargetInfo] = {}
    for package in _workspace_packages(metadata):
        for target in package["targets"]:
            if "test" not in target.get("kind", []):
                continue
            identity = Identity(package["name"], target["name"])
            if identity in result:
                raise ValueError(f"duplicate Cargo target identity: {identity.canonical}")
            result[identity] = TargetInfo(
                identity,
                _relative(target["src_path"], root),
                tuple(target.get("required-features", [])),
            )
    return result


def package_infos(metadata: Mapping[str, Any]) -> tuple[PackageInfo, ...]:
    root = metadata["workspace_root"]
    result = []
    for package in _workspace_packages(metadata):
        manifest = _relative(package["manifest_path"], root)
        result.append(PackageInfo(package["name"], str(PurePosixPath(manifest).parent)))
    return tuple(sorted(result, key=lambda item: item.name))


def test_functions(source: str) -> set[str]:
    return set(TEST_FUNCTION.findall(source))


def validate_config(
    config: Config,
    metadata: Mapping[str, Any],
    tracked_paths: set[str],
    source_reader: Callable[[str], str],
) -> None:
    """Reject stale package, target, test, and path-rule authority."""
    packages = {package.name for package in package_infos(metadata)}
    eligible = integration_targets(metadata)
    all_targets = all_integration_targets(metadata)

    for identity in config.standing_targets:
        if identity.package not in packages:
            raise ValueError(f"stale standing package: {identity.package}")
        if identity not in all_targets:
            raise ValueError(f"stale standing target: {identity.canonical}")
        if identity not in eligible:
            raise ValueError(
                f"standing target is not default-feature eligible: {identity.canonical}"
            )
    for identity in config.manual_only_targets:
        if identity.package not in packages:
            raise ValueError(f"stale manual-only package: {identity.package}")
        if identity not in all_targets:
            raise ValueError(f"stale manual-only target: {identity.canonical}")
        if identity not in eligible:
            raise ValueError(
                "manual-only target is not default-feature eligible: "
                f"{identity.canonical}"
            )
    for identity in config.target_exclusions:
        if identity.package not in packages:
            raise ValueError(f"stale exclusion package: {identity.package}")
        if identity not in all_targets:
            raise ValueError(f"stale target exclusion: {identity.canonical}")
        if identity not in eligible:
            raise ValueError(
                f"target exclusion is not default-feature eligible: {identity.canonical}"
            )
    for identity in config.test_exclusions:
        target = all_targets.get(identity.target_identity)
        if identity.package not in packages:
            raise ValueError(f"stale test-exclusion package: {identity.package}")
        if target is None:
            raise ValueError(
                f"stale test-exclusion target: {identity.target_identity.canonical}"
            )
        if identity.target_identity not in eligible:
            raise ValueError(
                f"test-exclusion target is not default-feature eligible: "
                f"{identity.target_identity.canonical}"
            )
        try:
            source = source_reader(target.src_path)
        except (OSError, KeyError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"cannot read source for test exclusion: {identity.canonical}"
            ) from error
        if identity.test not in test_functions(source):
            raise ValueError(f"stale exact test exclusion: {identity.canonical}")

    for rule in config.path_rules:
        stale_packages = sorted(set(rule.packages) - packages)
        if stale_packages:
            raise ValueError(
                f"path rule {rule.prefix!r} names stale packages: {stale_packages}"
            )
        if not any(rule.matches(path) for path in tracked_paths):
            raise ValueError(f"stale path rule matches no tracked path: {rule.prefix}")


def parse_name_status_z(raw: bytes) -> list[ChangeRecord]:
    """Parse ``git diff --name-status -z`` without losing rename pairs."""
    if not raw or not raw.endswith(b"\0"):
        raise ValueError("NUL-delimited git diff is empty or unterminated")
    fields = raw[:-1].split(b"\0")
    result: list[ChangeRecord] = []
    index = 0
    while index < len(fields):
        try:
            status = fields[index].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("git diff status is not ASCII") from error
        index += 1
        kind = status[:1]
        if kind == "C":
            raise ValueError("copy status is not supported; planner expects --find-renames")
        valid_rename = re.fullmatch(r"R(?:100|[0-9]{1,2})", status)
        if status not in {"A", "D", "M", "T"} and valid_rename is None:
            raise ValueError(f"unsupported git diff status: {status!r}")
        if index >= len(fields):
            raise ValueError(f"missing path after git diff status {status}")
        try:
            first = fields[index].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("changed path is not UTF-8") from error
        index += 1
        if not first:
            raise ValueError("changed path is empty")
        if kind == "R":
            if index >= len(fields):
                raise ValueError(f"missing destination for git diff status {status}")
            try:
                second = fields[index].decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("renamed path is not UTF-8") from error
            index += 1
            if not second:
                raise ValueError("renamed destination is empty")
            result.append(ChangeRecord(status, second, first))
        else:
            result.append(ChangeRecord(status, first))
    return result


def diff_paths(records: Sequence[ChangeRecord]) -> set[str]:
    result = {record.path for record in records}
    result.update(record.old_path for record in records if record.old_path is not None)
    return result


def _changed_path_sides(
    records: Sequence[ChangeRecord],
) -> Iterator[tuple[str, str, str]]:
    """Yield ``(path, side, status)`` in diff order."""
    for record in records:
        kind = record.status[0]
        if record.old_path is not None:
            yield record.old_path, "base", record.status
            yield record.path, "candidate", record.status
        elif kind == "D":
            yield record.path, "base", record.status
        else:
            yield record.path, "candidate", record.status


def _matching_packages(path: str, packages: Sequence[PackageInfo]) -> list[str]:
    matches = []
    for package in packages:
        if package.root == ".":
            continue
        if path == package.root or path.startswith(package.root + "/"):
            matches.append(package.name)
    return matches


def static_path_classification(
    path: str,
    packages: Sequence[PackageInfo],
    config: Config,
) -> tuple[str, list[str], list[PathRule]]:
    """Return the planner's non-target disposition for one changed path."""
    package_matches = sorted(set(_matching_packages(path, packages)))
    matching_rules = [rule for rule in config.path_rules if rule.matches(path)]
    if len(package_matches) > 1:
        return "ambiguous_package", package_matches, matching_rules
    if package_matches:
        return "package", package_matches, matching_rules
    if not matching_rules and is_docs_only([path]):
        return "docs", package_matches, matching_rules
    if len(matching_rules) == 1:
        return "rule", package_matches, matching_rules
    return (
        "ambiguous_rule" if matching_rules else "unclassified",
        package_matches,
        matching_rules,
    )


def _target_at_path(
    path: str, targets: Mapping[Identity, TargetInfo]
) -> list[Identity]:
    return sorted(info.identity for info in targets.values() if info.src_path == path)


def _owner_dict(owner: Owner) -> dict[str, str]:
    return owner.as_dict()


def _owned_target_rows(
    owners: Mapping[Identity, Owner],
) -> list[dict[str, Any]]:
    return [
        {"identity": identity.canonical, "owner": _owner_dict(owner)}
        for identity, owner in sorted(owners.items())
    ]


def _exclusion_rows(config: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_rows = [
        {"identity": identity.canonical, "owner": _owner_dict(owner)}
        for identity, owner in sorted(config.target_exclusions.items())
    ]
    test_rows = [
        {"identity": identity.canonical, "owner": _owner_dict(owner)}
        for identity, owner in sorted(config.test_exclusions.items())
    ]
    return target_rows, test_rows


def config_digest(config: Config) -> str:
    """Digest the normalized execution-relevant ownership configuration."""
    path_rules = []
    for rule in sorted(config.path_rules, key=lambda row: row.prefix):
        row: dict[str, Any] = {
            "prefix": rule.prefix,
            "disposition": rule.disposition,
        }
        if rule.disposition == "packages":
            row["packages"] = sorted(rule.packages)
        elif rule.owner is not None:
            row["owner"] = _owner_dict(rule.owner)
        path_rules.append(row)
    payload = {
        "version": config.version,
        "standing_targets": sorted(
            identity.canonical for identity in config.standing_targets
        ),
        "manual_only_targets": _owned_target_rows(config.manual_only_targets),
        "target_exclusions": _owned_target_rows(config.target_exclusions),
        "test_exclusions": [
            {"identity": identity.canonical, "owner": _owner_dict(owner)}
            for identity, owner in sorted(config.test_exclusions.items())
        ],
        "path_rules": path_rules,
    }
    return sha256_bytes(canonical_json(payload))


def make_plan(
    *,
    mode: str,
    base_sha: str,
    candidate_sha: str,
    records: Sequence[ChangeRecord],
    base_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    config: Config,
    tracked_paths: set[str],
    source_reader: Callable[[str], str],
    event_pr_head: str | None = None,
    base_tracked_paths: set[str] | None = None,
) -> dict[str, Any]:
    if mode not in {"pull_request", "push", "targeted_rebase"}:
        raise ValueError(f"unsupported planning mode: {mode}")
    validate_config(
        config,
        candidate_metadata,
        tracked_paths | (base_tracked_paths or set()),
        source_reader,
    )

    base_all = all_integration_targets(base_metadata)
    candidate_all = all_integration_targets(candidate_metadata)
    candidate_eligible = integration_targets(candidate_metadata)
    base_packages = package_infos(base_metadata)
    candidate_packages = package_infos(candidate_metadata)
    candidate_package_names = {package.name for package in candidate_packages}

    selected_packages: set[str] = set()
    change_owned: set[Identity] = set()
    dispositions: list[dict[str, Any]] = []
    target_dispositions: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for identity in sorted(set(candidate_all) - set(base_all)):
        info = candidate_all[identity]
        row: dict[str, Any] = {
            "identity": identity.canonical,
            "kind": "integration_target_added",
            "src_path": info.src_path,
        }
        if identity in config.target_exclusions:
            row["kind"] = "integration_target_added_excluded"
            row["owner"] = _owner_dict(config.target_exclusions[identity])
        elif identity not in candidate_eligible:
            row["kind"] = "integration_target_added_ineligible"
            row["required_features"] = list(info.required_features)
        else:
            change_owned.add(identity)
            selected_packages.add(identity.package)
            if identity in config.manual_only_targets:
                row["execution_mode"] = "ignored-only"
        target_dispositions.append(row)
    for identity in sorted(set(base_all) - set(candidate_all)):
        target_dispositions.append(
            {
                "identity": identity.canonical,
                "kind": "integration_target_deleted",
                "src_path": base_all[identity].src_path,
            }
        )

    for path, side, status in _changed_path_sides(records):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        target_space = base_all if side == "base" else candidate_all
        targets = _target_at_path(path, target_space)
        if len(targets) > 1:
            raise ValueError(f"ambiguous integration target source path: {path}")
        if targets:
            identity = targets[0]
            base_info = base_all.get(identity)
            candidate_info = candidate_all.get(identity)
            moved = (
                base_info is not None
                and candidate_info is not None
                and base_info.src_path != candidate_info.src_path
            )
            deleted = side == "base" and (
                identity not in candidate_all or moved or status.startswith("R")
            )
            added = side == "candidate" and (
                identity not in base_all or moved or status.startswith("R")
            )
            if deleted:
                dispositions.append(
                    {
                        "path": path,
                        "status": status,
                        "kind": "integration_target_deleted",
                        "identity": identity.canonical,
                    }
                )
                continue
            if identity in config.target_exclusions:
                dispositions.append(
                    {
                        "path": path,
                        "status": status,
                        "kind": "integration_target_excluded",
                        "identity": identity.canonical,
                        "owner": _owner_dict(config.target_exclusions[identity]),
                    }
                )
                continue
            if identity not in candidate_eligible:
                dispositions.append(
                    {
                        "path": path,
                        "status": status,
                        "kind": "integration_target_ineligible",
                        "identity": identity.canonical,
                        "required_features": list(
                            candidate_all[identity].required_features
                        ),
                    }
                )
                continue
            change_owned.add(identity)
            if identity.package in candidate_package_names:
                selected_packages.add(identity.package)
            dispositions.append(
                {
                    "path": path,
                    "status": status,
                    "kind": (
                        "integration_target_added"
                        if added
                        else "integration_target_directly_modified"
                    ),
                    "identity": identity.canonical,
                    **(
                        {"execution_mode": "ignored-only"}
                        if identity in config.manual_only_targets
                        else {}
                    ),
                }
            )
            continue

        classification, package_matches, matching_rules = static_path_classification(
            path,
            (*base_packages, *candidate_packages),
            config,
        )
        if classification == "ambiguous_package":
            raise ValueError(
                f"ambiguous package path {path!r}: matches {package_matches}"
            )
        if classification == "package":
            package = package_matches[0]
            if package in candidate_package_names:
                selected_packages.add(package)
                kind = "workspace_package"
            else:
                kind = "workspace_package_deleted"
            dispositions.append(
                {
                    "path": path,
                    "status": status,
                    "kind": kind,
                    "packages": [package],
                }
            )
            continue

        if classification == "docs":
            dispositions.append({"path": path, "status": status, "kind": "docs_only"})
            continue
        if classification != "rule":
            qualifier = (
                "ambiguous" if classification == "ambiguous_rule" else "unclassified"
            )
            raise ValueError(f"{qualifier} changed path: {path}")
        rule = matching_rules[0]
        disposition: dict[str, Any] = {
            "path": path,
            "status": status,
            "kind": f"path_rule_{rule.disposition}",
            "rule": rule.prefix,
        }
        if rule.disposition == "packages":
            selected_packages.update(rule.packages)
            disposition["packages"] = list(rule.packages)
        elif rule.disposition == "owner":
            if rule.owner is None:
                raise ValueError(f"owner path rule has no owner: {rule.prefix}")
            disposition["owner"] = _owner_dict(rule.owner)
        dispositions.append(disposition)

    expansion = {
        identity
        for identity in candidate_eligible
        if identity.package in selected_packages
        and identity not in change_owned
        and identity not in config.target_exclusions
    }
    if change_owned & expansion:
        raise ValueError("change-owned and package-expansion selections overlap")
    if mode == "targeted_rebase":
        change_owned |= expansion
        expansion = set()
        standing_coverage_reuse: set[Identity] = set()
        change_owned_execution = change_owned
    else:
        standing_coverage_reuse = change_owned & set(config.standing_targets)
        change_owned_execution = change_owned - standing_coverage_reuse

    target_exclusions, test_exclusions = _exclusion_rows(config)
    plan: dict[str, Any] = {
        "version": PLAN_VERSION,
        "mode": mode,
        "base_sha": base_sha,
        "candidate_sha": candidate_sha,
        "event_pr_head": event_pr_head,
        "config_digest": config_digest(config),
        "changed_records": [
            {
                "status": record.status,
                "path": record.path,
                **({"old_path": record.old_path} if record.old_path else {}),
            }
            for record in records
        ],
        "path_dispositions": dispositions,
        "target_dispositions": target_dispositions,
        "selected_packages": sorted(selected_packages),
        "eligible_targets": sorted(identity.canonical for identity in candidate_eligible),
        "change_owned": sorted(identity.canonical for identity in change_owned),
        "package_expansion": sorted(identity.canonical for identity in expansion),
        "standing_targets": sorted(
            identity.canonical for identity in config.standing_targets
        ),
        "standing_coverage_reuse": sorted(
            identity.canonical for identity in standing_coverage_reuse
        ),
        "manual_only_targets": _owned_target_rows(config.manual_only_targets),
        "target_exclusions": target_exclusions,
        "test_exclusions": test_exclusions,
        "shards": {
            "change_owned": shard_map(change_owned_execution),
            "package_expansion": shard_map(expansion),
        },
    }
    attach_plan_digest(plan)
    return plan


def canonical_json(data: Any) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest_without(data: Mapping[str, Any], key: str) -> str:
    payload = dict(data)
    payload.pop(key, None)
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def attach_plan_digest(plan: dict[str, Any]) -> None:
    plan["plan_digest"] = _digest_without(plan, "plan_digest")


def _identity_list(data: Mapping[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"plan {key} must be a list of identities")
    for item in value:
        Identity.parse(item)
    if len(value) != len(set(value)):
        raise ValueError(f"plan {key} contains duplicate identities")
    return value


def _validate_owner_mapping(owner: Any) -> None:
    if not isinstance(owner, dict) or set(owner) != OWNER_FIELDS:
        raise ValueError("owner mapping has the wrong fields")
    _owner(owner)


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    expected_keys = {
        "version",
        "mode",
        "base_sha",
        "candidate_sha",
        "event_pr_head",
        "config_digest",
        "changed_records",
        "path_dispositions",
        "target_dispositions",
        "selected_packages",
        "eligible_targets",
        "change_owned",
        "package_expansion",
        "standing_targets",
        "standing_coverage_reuse",
        "manual_only_targets",
        "target_exclusions",
        "test_exclusions",
        "shards",
        "plan_digest",
    }
    if set(plan) != expected_keys:
        raise ValueError(
            f"plan schema keys mismatch: "
            f"missing={sorted(expected_keys - set(plan))}, "
            f"extra={sorted(set(plan) - expected_keys)}"
        )
    if type(plan.get("version")) is not int or plan["version"] != PLAN_VERSION:
        raise ValueError(f"plan version must be {PLAN_VERSION}")
    if plan.get("mode") not in {
        "pull_request",
        "push",
        "targeted_rebase",
    }:
        raise ValueError(f"invalid plan mode: {plan.get('mode')!r}")
    for key in ("base_sha", "candidate_sha"):
        if not isinstance(plan.get(key), str) or not SHA.fullmatch(plan[key]):
            raise ValueError(f"plan {key} must be a full commit SHA")
    if not isinstance(plan.get("config_digest"), str) or not DIGEST.fullmatch(
        plan["config_digest"]
    ):
        raise ValueError("plan config_digest must be a SHA-256 digest")
    event_head = plan.get("event_pr_head")
    if plan["mode"] in {"pull_request", "targeted_rebase"}:
        if not isinstance(event_head, str) or not SHA.fullmatch(event_head):
            raise ValueError(
                f"{plan['mode']} plan requires a full event_pr_head"
            )
    elif event_head is not None:
        raise ValueError("push plan event_pr_head must be null")
    for key in ("changed_records", "path_dispositions", "target_dispositions"):
        if not isinstance(plan.get(key), list):
            raise ValueError(f"plan {key} must be a list")
    packages = plan.get("selected_packages")
    if (
        not isinstance(packages, list)
        or any(
            not isinstance(package, str) or not IDENTIFIER.fullmatch(package)
            for package in packages
        )
        or len(packages) != len(set(packages))
    ):
        raise ValueError("plan selected_packages must be unique package names")
    eligible = set(_identity_list(plan, "eligible_targets"))
    change_owned = set(_identity_list(plan, "change_owned"))
    expansion = set(_identity_list(plan, "package_expansion"))
    standing = set(_identity_list(plan, "standing_targets"))
    standing_reuse = set(_identity_list(plan, "standing_coverage_reuse"))
    if change_owned & expansion:
        raise ValueError("plan lanes overlap")
    if not (change_owned | expansion) <= eligible:
        raise ValueError("plan lane contains an ineligible target")
    expected_standing_reuse = (
        set()
        if plan["mode"] == "targeted_rebase"
        else change_owned & standing
    )
    if standing_reuse != expected_standing_reuse:
        raise ValueError(
            "plan standing coverage reuse must exactly match "
            "change-owned standing targets"
        )
    for key, parser in (
        ("manual_only_targets", Identity.parse),
        ("target_exclusions", Identity.parse),
        ("test_exclusions", TestIdentity.parse),
    ):
        rows = plan.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"plan {key} must be a list")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"identity", "owner"}:
                raise ValueError(f"plan {key} row has the wrong shape")
            parser(row["identity"])
            if row["identity"] in seen:
                raise ValueError(f"plan {key} contains duplicate identities")
            seen.add(row["identity"])
            _validate_owner_mapping(row["owner"])
    manual_only = {
        row["identity"] for row in plan["manual_only_targets"]
    }
    if not manual_only <= eligible:
        raise ValueError("plan manual-only targets must be eligible")
    target_exclusions = {
        row["identity"] for row in plan["target_exclusions"]
    }
    if manual_only & (standing | target_exclusions):
        raise ValueError(
            "plan manual-only targets conflict with standing or excluded targets"
        )
    test_exclusion_targets = {
        TestIdentity.parse(row["identity"]).target_identity.canonical
        for row in plan["test_exclusions"]
    }
    if manual_only & test_exclusion_targets:
        raise ValueError("plan manual-only targets cannot contain test exclusions")
    shards = plan.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(LANE_KEYS.values()):
        raise ValueError("plan shards must name both lanes")
    for lane_key in LANE_KEYS.values():
        lane_shards = shards[lane_key]
        if not isinstance(lane_shards, dict) or set(lane_shards) != {
            str(shard) for shard in SHARDS
        }:
            raise ValueError(f"plan {lane_key} must contain four exact shards")
        flattened: list[str] = []
        for shard in SHARDS:
            rows = lane_shards[str(shard)]
            if not isinstance(rows, list):
                raise ValueError(f"plan {lane_key} shard {shard} must be a list")
            for canonical in rows:
                identity = Identity.parse(canonical)
                if shard_for(identity) != shard:
                    raise ValueError(
                        f"plan {lane_key} identity is in the wrong shard: {canonical}"
                    )
            flattened.extend(rows)
        expected_lane = (
            change_owned - standing_reuse
            if lane_key == "change_owned"
            else expansion
        )
        if sorted(flattened) != sorted(expected_lane):
            raise ValueError(f"plan {lane_key} shards do not exactly cover the lane")
    if not isinstance(plan.get("plan_digest"), str) or not DIGEST.fullmatch(
        plan["plan_digest"]
    ):
        raise ValueError("plan_digest must be a SHA-256 digest")


def verify_plan_digest(plan: Mapping[str, Any]) -> None:
    _validate_plan_shape(plan)
    expected = _digest_without(plan, "plan_digest")
    if plan.get("plan_digest") != expected:
        raise ValueError(
            f"plan digest mismatch: expected {expected}, got {plan.get('plan_digest')}"
        )


def attach_receipt_digest(receipt: dict[str, Any]) -> None:
    receipt["receipt_digest"] = _digest_without(receipt, "receipt_digest")


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    expected_keys = {
        "version",
        "lane",
        "shard",
        "plan_digest",
        "selected_targets",
        "executed_targets",
        "selected_tests",
        "executed_tests",
        "commands_file",
        "timings_file",
        "test_list_file",
        "junit_file",
        "started_at",
        "finished_at",
        "elapsed_seconds",
        "soft_budget_seconds",
        "soft_budget_exceeded",
        "sidecars",
        "success",
        "failures",
        "receipt_digest",
    }
    if set(receipt) != expected_keys:
        raise ValueError(
            f"receipt schema keys mismatch: "
            f"missing={sorted(expected_keys - set(receipt))}, "
            f"extra={sorted(set(receipt) - expected_keys)}"
        )
    if (
        type(receipt.get("version")) is not int
        or receipt["version"] != RECEIPT_VERSION
    ):
        raise ValueError(f"receipt version must be {RECEIPT_VERSION}")
    lane = receipt.get("lane")
    if lane not in LANE_KEYS:
        raise ValueError(f"invalid receipt lane: {lane!r}")
    shard = receipt.get("shard")
    if type(shard) is not int or shard not in SHARDS:
        raise ValueError(f"invalid receipt shard: {shard!r}")
    if not isinstance(receipt.get("plan_digest"), str) or not DIGEST.fullmatch(
        receipt["plan_digest"]
    ):
        raise ValueError("receipt plan_digest must be a SHA-256 digest")
    for key in ("selected_targets", "executed_targets"):
        _identity_list(receipt, key)
    for key in ("selected_tests", "executed_tests"):
        rows = receipt.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
            raise ValueError(f"receipt {key} must be a list")
        for row in rows:
            TestIdentity.parse(row)
        if len(rows) != len(set(rows)):
            raise ValueError(f"receipt {key} contains duplicates")
    files = {
        "commands_file": "commands.json",
        "timings_file": "timings.json",
        "test_list_file": "test-list.json",
        "junit_file": "junit.xml",
    }
    for key, expected in files.items():
        if receipt.get(key) != expected:
            raise ValueError(f"receipt {key} must be {expected!r}")
    for key in ("started_at", "finished_at"):
        if not isinstance(receipt.get(key), str) or not receipt[key]:
            raise ValueError(f"receipt {key} must be a timestamp")
    elapsed = receipt.get("elapsed_seconds")
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or elapsed < 0:
        raise ValueError("receipt elapsed_seconds must be nonnegative")
    expected_budget = SOFT_BUDGET_SECONDS if lane == "package-expansion" else None
    if receipt.get("soft_budget_seconds") != expected_budget:
        raise ValueError("receipt soft_budget_seconds does not match its lane")
    exceeded = receipt.get("soft_budget_exceeded")
    if type(exceeded) is not bool or exceeded != (
        expected_budget is not None and elapsed > expected_budget
    ):
        raise ValueError("receipt soft_budget_exceeded is inconsistent")
    sidecars = receipt.get("sidecars")
    if not isinstance(sidecars, dict) or set(sidecars) != set(SIDECAR_NAMES):
        raise ValueError("receipt sidecars must bind every exact sidecar")
    if any(not isinstance(value, str) or not DIGEST.fullmatch(value) for value in sidecars.values()):
        raise ValueError("receipt sidecar digest is malformed")
    if type(receipt.get("success")) is not bool:
        raise ValueError("receipt success must be boolean")
    failures = receipt.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise ValueError("receipt failures must be nonempty strings")
    if receipt["success"] == bool(failures):
        raise ValueError("receipt success and failures disagree")
    if not isinstance(receipt.get("receipt_digest"), str) or not DIGEST.fullmatch(
        receipt["receipt_digest"]
    ):
        raise ValueError("receipt_digest must be a SHA-256 digest")


def verify_receipt_digest(receipt: Mapping[str, Any]) -> None:
    _validate_receipt_shape(receipt)
    expected = _digest_without(receipt, "receipt_digest")
    if receipt.get("receipt_digest") != expected:
        raise ValueError(
            "receipt digest mismatch: "
            f"expected {expected}, got {receipt.get('receipt_digest')}"
        )


def attach_standing_coverage_digest(coverage: dict[str, Any]) -> None:
    coverage["coverage_digest"] = _digest_without(
        coverage, "coverage_digest"
    )


def _validate_standing_coverage_shape(coverage: Mapping[str, Any]) -> None:
    expected_keys = {
        "version",
        "candidate_sha",
        "config_digest",
        "execution",
        "selected_targets",
        "executed_targets",
        "selected_tests",
        "executed_tests",
        "success",
        "failures",
        "coverage_digest",
    }
    if set(coverage) != expected_keys:
        raise ValueError(
            "standing coverage schema keys mismatch: "
            f"missing={sorted(expected_keys - set(coverage))}, "
            f"extra={sorted(set(coverage) - expected_keys)}"
        )
    if (
        type(coverage.get("version")) is not int
        or coverage["version"] != STANDING_COVERAGE_VERSION
    ):
        raise ValueError(
            f"standing coverage version must be {STANDING_COVERAGE_VERSION}"
        )
    if not isinstance(coverage.get("candidate_sha"), str) or not SHA.fullmatch(
        coverage["candidate_sha"]
    ):
        raise ValueError("standing coverage candidate_sha must be a full SHA")
    if not isinstance(coverage.get("config_digest"), str) or not DIGEST.fullmatch(
        coverage["config_digest"]
    ):
        raise ValueError(
            "standing coverage config_digest must be a SHA-256 digest"
        )
    execution = coverage.get("execution")
    if not isinstance(execution, dict) or execution != STANDING_EXECUTION:
        raise ValueError(
            "standing coverage execution configuration mismatch: "
            f"expected={STANDING_EXECUTION}, got={execution}"
        )
    selected_targets = set(_identity_list(coverage, "selected_targets"))
    executed_targets = set(_identity_list(coverage, "executed_targets"))
    for key in ("selected_tests", "executed_tests"):
        rows = coverage.get(key)
        if not isinstance(rows, list) or any(
            not isinstance(row, str) for row in rows
        ):
            raise ValueError(f"standing coverage {key} must be a list")
        parsed = [TestIdentity.parse(row) for row in rows]
        if len(rows) != len(set(rows)):
            raise ValueError(f"standing coverage {key} contains duplicates")
        unknown = sorted(
            identity.canonical
            for identity in parsed
            if identity.target_identity.canonical not in selected_targets
        )
        if unknown:
            raise ValueError(
                f"standing coverage {key} names unselected targets: {unknown}"
            )
    if not executed_targets <= selected_targets:
        raise ValueError(
            "standing coverage executed targets must be selected targets"
        )
    if type(coverage.get("success")) is not bool:
        raise ValueError("standing coverage success must be boolean")
    failures = coverage.get("failures")
    if not isinstance(failures, list) or any(
        not isinstance(failure, str) or not failure for failure in failures
    ):
        raise ValueError(
            "standing coverage failures must be nonempty strings"
        )
    if coverage["success"] == bool(failures):
        raise ValueError("standing coverage success and failures disagree")
    if not isinstance(coverage.get("coverage_digest"), str) or not DIGEST.fullmatch(
        coverage["coverage_digest"]
    ):
        raise ValueError("coverage_digest must be a SHA-256 digest")


def verify_standing_coverage_digest(coverage: Mapping[str, Any]) -> None:
    _validate_standing_coverage_shape(coverage)
    expected = _digest_without(coverage, "coverage_digest")
    if coverage.get("coverage_digest") != expected:
        raise ValueError(
            "standing coverage digest mismatch: "
            f"expected {expected}, got {coverage.get('coverage_digest')}"
        )


def validate_standing_coverage(
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any] | None,
) -> list[str]:
    """Return exact reused targets after validating a complete ci-fast record."""
    verify_plan_digest(plan)
    reused = sorted(plan["standing_coverage_reuse"])
    if not reused:
        return []
    if coverage is None:
        raise ValueError(
            "standing coverage receipt is required for reused targets"
        )
    verify_standing_coverage_digest(coverage)
    if coverage["candidate_sha"] != plan["candidate_sha"]:
        raise ValueError(
            "standing coverage candidate mismatch: "
            f"expected={plan['candidate_sha']}, "
            f"got={coverage['candidate_sha']}"
        )
    if coverage["config_digest"] != plan["config_digest"]:
        raise ValueError(
            "standing coverage configuration mismatch: "
            f"expected={plan['config_digest']}, "
            f"got={coverage['config_digest']}"
        )
    expected_targets = sorted(plan["standing_targets"])
    if sorted(coverage["selected_targets"]) != expected_targets:
        raise ValueError(
            "standing coverage selected targets mismatch: "
            f"expected={expected_targets}, "
            f"got={sorted(coverage['selected_targets'])}"
        )
    if sorted(coverage["executed_targets"]) != expected_targets:
        raise ValueError(
            "standing coverage executed targets mismatch: "
            f"expected={expected_targets}, "
            f"got={sorted(coverage['executed_targets'])}"
        )
    if coverage["success"] is not True:
        raise ValueError(
            f"standing coverage did not succeed: {coverage['failures']}"
        )
    selected_tests = sorted(coverage["selected_tests"])
    executed_tests = sorted(coverage["executed_tests"])
    if selected_tests != executed_tests:
        raise ValueError(
            "standing coverage selected test coverage mismatch: "
            f"selected={selected_tests}, executed={executed_tests}"
        )
    for target in expected_targets:
        prefix = f"{target}::"
        if not any(test.startswith(prefix) for test in selected_tests):
            raise ValueError(
                f"standing coverage has no complete test results for {target}"
            )
    return reused


def shard_for(identity: Identity) -> int:
    digest = hashlib.sha256(identity.canonical.encode()).digest()
    return int.from_bytes(digest, "big") % len(SHARDS)


def shard_map(identities: Iterable[Identity]) -> dict[str, list[str]]:
    result = {str(shard): [] for shard in SHARDS}
    for identity in sorted(set(identities)):
        result[str(shard_for(identity))].append(identity.canonical)
    return result


def git_output(repo: Path, args: Sequence[str]) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _commit(repo: Path, value: str) -> str:
    resolved = git_output(repo, ["rev-parse", f"{value}^{{commit}}"]).decode().strip()
    if not SHA.fullmatch(resolved):
        raise ValueError(f"git did not resolve a full commit SHA: {value!r}")
    return resolved


def resolve_pr_commits(
    repo: Path, candidate: str, event_pr_head: str
) -> tuple[str, str]:
    fields = git_output(repo, ["rev-list", "--parents", "-n", "1", candidate])
    parts = fields.decode().strip().split()
    if len(parts) != 3:
        raise ValueError("pull-request candidate must have exactly two parents")
    _, base, head = parts
    if head != event_pr_head:
        raise ValueError(
            f"candidate HEAD^2 {head} does not equal event pull-request head "
            f"{event_pr_head}"
        )
    return base, parts[0]


def _metadata_in(repo: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout
    )


def metadata_at(repo: Path, commit: str) -> dict[str, Any]:
    head = _commit(repo, "HEAD")
    if commit == head:
        return _metadata_in(repo)
    with tempfile.TemporaryDirectory(prefix="chelis-ci-metadata-") as tmp:
        worktree = Path(tmp) / "tree"
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            return _metadata_in(worktree)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )


def tracked_paths_at(repo: Path, commit: str) -> set[str]:
    raw = git_output(repo, ["ls-tree", "-r", "--name-only", "-z", commit])
    if raw and not raw.endswith(b"\0"):
        raise ValueError("git ls-tree returned unterminated paths")
    return {
        field.decode("utf-8")
        for field in raw.rstrip(b"\0").split(b"\0")
        if field
    }


def source_at(repo: Path, commit: str, path: str) -> str:
    return git_output(repo, ["show", f"{commit}:{path}"]).decode("utf-8")


def diff_at(repo: Path, base: str, candidate: str) -> list[ChangeRecord]:
    return parse_name_status_z(
        git_output(
            repo,
            [
                "diff",
                "--name-status",
                "-z",
                "--find-renames",
                base,
                candidate,
            ],
        )
    )


def generate_plan(
    repo: Path,
    *,
    event_name: str,
    pr_head: str,
    before: str,
    after: str,
    config_path: Path,
) -> dict[str, Any]:
    head = _commit(repo, "HEAD")
    if event_name == "pull_request":
        if not pr_head:
            raise ValueError("pull_request planning requires --pr-head")
        pr_head = _commit(repo, pr_head)
        base, candidate = resolve_pr_commits(repo, head, pr_head)
        mode = "pull_request"
        event_pr_head: str | None = pr_head
    elif event_name == "targeted_rebase":
        if not pr_head or not before:
            raise ValueError(
                "targeted_rebase planning requires --pr-head and --before"
            )
        pr_head = _commit(repo, pr_head)
        _, candidate = resolve_pr_commits(repo, head, pr_head)
        base = _commit(repo, before)
        mode = "targeted_rebase"
        event_pr_head = pr_head
    elif event_name == "push":
        if not before or not after:
            raise ValueError("push planning requires --before and --after")
        base, candidate = _commit(repo, before), _commit(repo, after)
        if candidate != head:
            raise ValueError(
                f"push --after {candidate} must equal checked-out HEAD {head}"
            )
        mode = "push"
        event_pr_head = None
    else:
        raise ValueError(f"unsupported --event-name: {event_name!r}")

    config = read_config(config_path)
    base_metadata = metadata_at(repo, base)
    candidate_metadata = metadata_at(repo, candidate)
    records = diff_at(repo, base, candidate)
    tracked = tracked_paths_at(repo, candidate)
    base_tracked = tracked_paths_at(repo, base)
    return make_plan(
        mode=mode,
        base_sha=base,
        candidate_sha=candidate,
        event_pr_head=event_pr_head,
        records=records,
        base_metadata=base_metadata,
        candidate_metadata=candidate_metadata,
        config=config,
        tracked_paths=tracked,
        base_tracked_paths=base_tracked,
        source_reader=lambda path: source_at(repo, candidate, path),
    )


def _test_exclusions_from_plan(
    plan: Mapping[str, Any],
) -> dict[TestIdentity, dict[str, str]]:
    result = {}
    for row in plan["test_exclusions"]:
        identity = TestIdentity.parse(row["identity"])
        result[identity] = row["owner"]
    return result


def target_command(
    identity: Identity,
    exclusions: Mapping[TestIdentity, Any],
    *,
    list_only: bool,
    manual_only: bool = False,
) -> list[str]:
    action = "list" if list_only else "run"
    command = [
        "cargo",
        "nextest",
        action,
        "-p",
        identity.package,
        "--test",
        identity.target,
        "--locked",
        "--profile",
        "ci-full",
        "--ignore-default-filter",
    ]
    if manual_only:
        command.extend(["--run-ignored", "all"])
    relevant = sorted(
        exclusion.test
        for exclusion in exclusions
        if exclusion.target_identity == identity
    )
    if relevant:
        alternatives = " + ".join(
            f"test(/^{re.escape(test)}$/)" for test in relevant
        )
        command.extend(["-E", f"not ({alternatives})"])
    if list_only:
        command.extend(["--message-format", "json"])
    else:
        command.append("--no-fail-fast")
    return command


def execution_groups(
    plan: Mapping[str, Any],
    *,
    lane: str,
    selected: Sequence[str],
) -> list[tuple[Identity, ...]]:
    """Batch ordinary manual-expansion targets by package."""
    identities = [Identity.parse(canonical) for canonical in selected]
    if lane != "package-expansion":
        return [(identity,) for identity in identities]
    manual_only = {
        row["identity"] for row in plan["manual_only_targets"]
    }
    excluded_targets = {
        TestIdentity.parse(row["identity"]).target_identity
        for row in plan["test_exclusions"]
    }
    groups: list[list[Identity]] = []
    ordinary_by_package: dict[str, int] = {}
    for identity in identities:
        special = (
            identity.canonical in manual_only
            or identity in excluded_targets
        )
        if special:
            groups.append([identity])
            continue
        index = ordinary_by_package.get(identity.package)
        if index is None:
            ordinary_by_package[identity.package] = len(groups)
            groups.append([identity])
        else:
            groups[index].append(identity)
    return [tuple(group) for group in groups]


def target_group_command(
    identities: Sequence[Identity],
    exclusions: Mapping[TestIdentity, Any],
    *,
    list_only: bool,
    manual_only: bool = False,
) -> list[str]:
    if not identities:
        raise ValueError("target command group must not be empty")
    if len(identities) == 1:
        return target_command(
            identities[0],
            exclusions,
            list_only=list_only,
            manual_only=manual_only,
        )
    packages = {identity.package for identity in identities}
    if len(packages) != 1:
        raise ValueError("target command group must name one exact package")
    if manual_only:
        raise ValueError("manual-only targets must execute in singleton groups")
    if any(
        exclusion.target_identity in identities for exclusion in exclusions
    ):
        raise ValueError(
            "targets with exact test exclusions must execute in singleton groups"
        )
    action = "list" if list_only else "run"
    command = [
        "cargo",
        "nextest",
        action,
        "-p",
        identities[0].package,
    ]
    for identity in identities:
        command.extend(["--test", identity.target])
    command.extend(
        [
            "--locked",
            "--profile",
            "ci-full",
            "--ignore-default-filter",
        ]
    )
    if list_only:
        command.extend(["--message-format", "json"])
    else:
        command.append("--no-fail-fast")
    return command


def _listing_tests_for_group(
    data: Mapping[str, Any],
    identities: Sequence[Identity],
    exclusions: Mapping[TestIdentity, Any],
    *,
    manual_only: bool = False,
) -> list[str]:
    suites = data.get("rust-suites")
    if not isinstance(suites, dict):
        raise ValueError("nextest listing has no rust-suites object")
    expected = {identity.canonical for identity in identities}
    if set(suites) != expected:
        raise ValueError(
            "package-scoped listing mismatch for target group: "
            f"expected={sorted(expected)}, got={sorted(suites)}"
        )
    selected = []
    for identity in identities:
        selected.extend(
            _listing_tests(
                {"rust-suites": {identity.canonical: suites[identity.canonical]}},
                identity,
                exclusions,
                manual_only=manual_only,
            )
        )
    if len(selected) != len(set(selected)):
        raise ValueError("target group listing contains duplicate tests")
    return sorted(selected)


def _junit_tests_for_group(
    path: Path,
    identities: Sequence[Identity],
) -> list[str]:
    expected = {identity.canonical for identity in identities}
    tests = []
    observed_targets = set()
    for case in ET.parse(path).iter("testcase"):
        classname = case.get("classname")
        name = case.get("name")
        if classname not in expected:
            raise ValueError(
                f"JUnit testcase has unexpected target {classname!r}: {path}"
            )
        if not name:
            raise ValueError(f"JUnit testcase has no name: {path}")
        require_executed_junit_case(case, path)
        observed_targets.add(classname)
        tests.append(f"{classname}::{name}")
    if len(tests) != len(set(tests)):
        raise ValueError(f"JUnit contains duplicate test results: {path}")
    if not observed_targets <= expected:
        raise ValueError(f"JUnit target mismatch: {path}")
    return sorted(tests)


def _listing_tests(
    data: Mapping[str, Any],
    identity: Identity,
    exclusions: Mapping[TestIdentity, Any],
    *,
    manual_only: bool = False,
) -> list[str]:
    suites = data.get("rust-suites")
    if not isinstance(suites, dict):
        raise ValueError("nextest listing has no rust-suites object")
    expected = identity.canonical
    if set(suites) != {expected}:
        raise ValueError(
            f"package-scoped listing mismatch for {expected}: {sorted(suites)}"
        )
    tests = suites[expected].get("testcases")
    if not isinstance(tests, dict):
        raise ValueError(f"nextest listing has no testcases for {expected}")
    selected = []
    nonmatching: set[str] = set()
    active: set[str] = set()
    for name, info in tests.items():
        if not isinstance(info, dict):
            raise ValueError(f"malformed testcase listing for {expected}::{name}")
        ignored = info.get("ignored")
        if type(ignored) is not bool:
            raise ValueError(
                f"malformed ignored flag for {expected}::{name}: {ignored!r}"
            )
        if manual_only and not ignored:
            active.add(name)
        if ignored and not manual_only:
            continue
        filter_match = info.get("filter-match", {})
        if not isinstance(filter_match, dict):
            raise ValueError(
                f"malformed filter match for {expected}::{name}: "
                f"{filter_match!r}"
            )
        status = filter_match.get("status")
        if status == "matches":
            selected.append(f"{expected}::{name}")
        elif status == "mismatch":
            nonmatching.add(name)
        else:
            raise ValueError(
                f"malformed filter status for {expected}::{name}: {status!r}"
            )
    if manual_only and active:
        raise ValueError(
            f"manual-only target {expected} has active tests: {sorted(active)}"
        )
    configured_nonmatching = {
        exclusion.test
        for exclusion in exclusions
        if exclusion.target_identity == identity
    }
    if nonmatching != configured_nonmatching:
        raise ValueError(
            f"nonmatching active tests differ from exact exclusions for {expected}: "
            f"configured={sorted(configured_nonmatching)}, "
            f"observed={sorted(nonmatching)}"
        )
    if not selected:
        raise ValueError(f"selected integration has no active tests: {expected}")
    return sorted(selected)


def _junit_tests(path: Path, identity: Identity) -> list[str]:
    tree = ET.parse(path)
    tests = []
    for case in tree.iter("testcase"):
        name = case.get("name")
        if not name:
            raise ValueError(f"JUnit testcase has no name: {path}")
        require_executed_junit_case(case, path)
        tests.append(f"{identity.canonical}::{name}")
    return sorted(tests)


def require_executed_junit_case(case: ET.Element, path: Path) -> None:
    """Reject a selected test that nextest reported as skipped."""
    if case.find("skipped") is not None:
        classname = case.get("classname")
        name = case.get("name")
        identity = "::".join(
            value for value in (classname, name) if value
        ) or "<unknown>"
        raise ValueError(
            f"JUnit testcase was skipped rather than executed: "
            f"{identity} ({path})"
        )


def _write_junit(path: Path, suite_documents: Sequence[Path]) -> None:
    root = ET.Element("testsuites")
    for document in suite_documents:
        parsed = ET.parse(document).getroot()
        if parsed.tag == "testsuite":
            root.append(parsed)
        elif parsed.tag == "testsuites":
            root.extend(list(parsed))
        else:
            raise ValueError(f"unexpected JUnit root {parsed.tag!r}: {document}")
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    with path.open("a") as handle:
        handle.write("\n")


def run_command(
    command: Sequence[str], *, timeout: float | None = None, **kwargs: Any
) -> subprocess.CompletedProcess[str]:
    """Bound the optional worker's command tree, including inherited pipes."""
    if timeout is None:
        return subprocess.run(command, **kwargs)
    if os.name != "posix":
        raise ValueError("bounded expansion execution requires POSIX process groups")
    check = kwargs.pop("check", False)
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException as error:
            # Killing only Cargo can leave rustc/nextest/tests holding the pipes
            # open, so communicate() would outlive the deadline indefinitely.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = process.communicate()
            if isinstance(error, subprocess.TimeoutExpired):
                raise subprocess.TimeoutExpired(
                    command, timeout, output=stdout, stderr=stderr
                ) from None
            raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    if check:
        result.check_returncode()
    return result


def execute_shard(
    plan: Mapping[str, Any],
    *,
    lane: str,
    shard: int,
    output: Path,
    repo: Path = ROOT,
    runner: Callable[..., subprocess.CompletedProcess[str]] = run_command,
) -> dict[str, Any]:
    verify_plan_digest(plan)
    checked_out = _commit(repo, "HEAD")
    if checked_out != plan["candidate_sha"]:
        raise ValueError(
            f"checked-out HEAD {checked_out} does not equal plan candidate "
            f"{plan['candidate_sha']}"
        )
    if lane not in LANE_KEYS:
        raise ValueError(f"unsupported lane: {lane}")
    if shard not in SHARDS:
        raise ValueError(f"shard must be one of {SHARDS}")
    shard_started_at = _utc_timestamp()
    shard_started = time.monotonic()
    deadline = (
        shard_started + EXPANSION_EXECUTION_SECONDS
        if lane == "package-expansion" else None
    )
    deadline_exhausted = False
    lane_key = LANE_KEYS[lane]
    selected = list(plan["shards"][lane_key][str(shard)])
    output.mkdir(parents=True, exist_ok=True)
    junit_output = output / "junit.xml"
    timing_output = output / "timings.json"
    command_output = output / "commands.json"
    listing_output = output / "test-list.json"
    receipt_output = output / "receipt.json"
    scratch = output / ".junit"
    scratch.mkdir(exist_ok=True)

    commands: list[dict[str, Any]] = []
    target_timings: dict[str, dict[str, Any]] = {}
    product_timing: dict[str, Any] | None = None
    selected_tests: list[str] = []
    executed_tests: list[str] = []
    executed_targets: list[str] = []
    failures: list[str] = []

    def run(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal deadline_exhausted
        if deadline is None:
            return runner(command, **kwargs)
        remaining = deadline - time.monotonic()
        try:
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, 0, output="", stderr="")
            return runner(command, timeout=remaining, **kwargs)
        except subprocess.TimeoutExpired as error:
            deadline_exhausted = True
            failures.append(
                "package-expansion execution deadline exhausted; "
                "unfinished selected coverage remains unsuccessful"
            )
            # TimeoutExpired can contain bytes even for a text-mode subprocess.
            def text_output(value: str | bytes | None) -> str:
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return value or ""

            raise subprocess.CalledProcessError(
                124, command, output=text_output(error.stdout),
                stderr=text_output(error.stderr),
            ) from error

    suite_documents: list[Path] = []
    exclusions = _test_exclusions_from_plan(plan)
    manual_only_targets = {
        row["identity"] for row in plan["manual_only_targets"]
    }
    cargo_target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not cargo_target.is_absolute():
        cargo_target = repo / cargo_target
    produced_junit = cargo_target / "nextest/ci-full/junit.xml"

    build_succeeded = True
    if selected:
        build_command = [
            "cargo",
            "build",
            "--workspace",
            "--lib",
            "--bins",
            "--locked",
        ]
        build_started_at = _utc_timestamp()
        build_started = time.monotonic()
        try:
            completed = run(
                build_command,
                cwd=repo,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            commands.append(
                {
                    "identity": None,
                    "kind": "workspace-products",
                    "argv": build_command,
                    "started_at": build_started_at,
                    "finished_at": _utc_timestamp(),
                    "returncode": completed.returncode,
                    "stdout": completed.stdout or "",
                    "stderr": completed.stderr or "",
                }
            )
        except subprocess.CalledProcessError as error:
            build_succeeded = False
            failures.append(
                f"workspace product build failed with {error.returncode}"
            )
            commands.append(
                {
                    "identity": None,
                    "kind": "workspace-products",
                    "argv": build_command,
                    "started_at": build_started_at,
                    "finished_at": _utc_timestamp(),
                    "returncode": error.returncode,
                    "stdout": error.stdout or "",
                    "stderr": error.stderr or "",
                }
            )
        product_timing = {
            "started_at": build_started_at,
            "finished_at": commands[-1]["finished_at"],
            "elapsed_seconds": round(time.monotonic() - build_started, 3),
            "success": build_succeeded,
        }

    if build_succeeded:
        groups = execution_groups(plan, lane=lane, selected=selected)
        for group_index, identities in enumerate(groups):
            if deadline_exhausted:
                break
            canonicals = [identity.canonical for identity in identities]
            label = ", ".join(canonicals)
            manual_only = (
                len(identities) == 1
                and identities[0].canonical in manual_only_targets
            )
            list_command = target_group_command(
                identities,
                exclusions,
                list_only=True,
                manual_only=manual_only,
            )
            list_started_at = _utc_timestamp()
            started = time.monotonic()
            try:
                listed = run(
                    list_command,
                    cwd=repo,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                listed_tests = _listing_tests_for_group(
                    json.loads(listed.stdout),
                    identities,
                    exclusions,
                    manual_only=manual_only,
                )
                selected_tests.extend(listed_tests)
                commands.append(
                    {
                        "identity": (
                            canonicals[0] if len(canonicals) == 1 else None
                        ),
                        "identities": canonicals,
                        "kind": "list",
                        "argv": list_command,
                        "started_at": list_started_at,
                        "finished_at": _utc_timestamp(),
                        "returncode": listed.returncode,
                        "stdout": listed.stdout,
                        "stderr": listed.stderr or "",
                    }
                )
            except (
                TypeError,
                ValueError,
                json.JSONDecodeError,
                subprocess.CalledProcessError,
            ) as error:
                failures.append(f"{label}: list failed: {error}")
                commands.append(
                    {
                        "identity": (
                            canonicals[0] if len(canonicals) == 1 else None
                        ),
                        "identities": canonicals,
                        "kind": "list",
                        "argv": list_command,
                        "started_at": list_started_at,
                        "finished_at": _utc_timestamp(),
                        "returncode": getattr(error, "returncode", 1),
                        "stdout": getattr(error, "stdout", "") or "",
                        "stderr": getattr(error, "stderr", "") or str(error),
                    }
                )
                for canonical in canonicals:
                    target_timings[canonical] = {
                        "command_group": canonicals,
                        "list_started_at": list_started_at,
                        "list_finished_at": commands[-1]["finished_at"],
                        "list_seconds": round(time.monotonic() - started, 3),
                        "run_started_at": None,
                        "run_finished_at": None,
                        "run_seconds": 0.0,
                    }
                continue
            list_seconds = round(time.monotonic() - started, 3)
            list_finished_at = commands[-1]["finished_at"]

            run_command = target_group_command(
                identities,
                exclusions,
                list_only=False,
                manual_only=manual_only,
            )
            produced_junit.unlink(missing_ok=True)
            run_started_at = _utc_timestamp()
            started = time.monotonic()
            try:
                completed = run(
                    run_command,
                    cwd=repo,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                commands.append(
                    {
                        "identity": (
                            canonicals[0] if len(canonicals) == 1 else None
                        ),
                        "identities": canonicals,
                        "kind": "run",
                        "argv": run_command,
                        "started_at": run_started_at,
                        "finished_at": _utc_timestamp(),
                        "returncode": completed.returncode,
                        "stdout": completed.stdout or "",
                        "stderr": completed.stderr or "",
                    }
                )
            except subprocess.CalledProcessError as error:
                failures.append(
                    f"{label}: test run failed with {error.returncode}"
                )
                commands.append(
                    {
                        "identity": (
                            canonicals[0] if len(canonicals) == 1 else None
                        ),
                        "identities": canonicals,
                        "kind": "run",
                        "argv": run_command,
                        "started_at": run_started_at,
                        "finished_at": _utc_timestamp(),
                        "returncode": error.returncode,
                        "stdout": error.stdout or "",
                        "stderr": error.stderr or "",
                    }
                )
            run_seconds = round(time.monotonic() - started, 3)
            for canonical in canonicals:
                target_timings[canonical] = {
                    "command_group": canonicals,
                    "list_started_at": list_started_at,
                    "list_finished_at": list_finished_at,
                    "list_seconds": list_seconds,
                    "run_started_at": run_started_at,
                    "run_finished_at": commands[-1]["finished_at"],
                    "run_seconds": run_seconds,
                }
            if produced_junit.is_file():
                target_junit = scratch / f"group-{group_index}.xml"
                shutil.copyfile(produced_junit, target_junit)
                try:
                    group_executed_tests = (
                        _junit_tests(target_junit, identities[0])
                        if len(identities) == 1
                        else _junit_tests_for_group(
                            target_junit,
                            identities,
                        )
                    )
                except (ValueError, ET.ParseError) as error:
                    failures.append(f"{label}: malformed JUnit: {error}")
                else:
                    executed_tests.extend(group_executed_tests)
                    for canonical in canonicals:
                        prefix = f"{canonical}::"
                        target_selected = sorted(
                            test
                            for test in listed_tests
                            if test.startswith(prefix)
                        )
                        target_executed = sorted(
                            test
                            for test in group_executed_tests
                            if test.startswith(prefix)
                        )
                        if target_selected == target_executed:
                            executed_targets.append(canonical)
                        else:
                            failures.append(
                                f"{canonical}: incomplete test results: "
                                f"selected={target_selected}, "
                                f"executed={target_executed}"
                            )
                    # A deadline may interrupt nextest's XML write. Keep that
                    # failure, but merge only validated suites so receipt
                    # finalization still preserves earlier completed evidence.
                    suite_documents.append(target_junit)
            else:
                failures.append(f"{label}: test run produced no JUnit")

    _write_junit(junit_output, suite_documents)
    shard_finished_at = _utc_timestamp()
    elapsed_seconds = round(time.monotonic() - shard_started, 3)
    soft_budget_seconds = (
        SOFT_BUDGET_SECONDS if lane == "package-expansion" else None
    )
    soft_budget_exceeded = (
        soft_budget_seconds is not None and elapsed_seconds > soft_budget_seconds
    )
    command_output.write_bytes(canonical_json(commands))
    timing_output.write_bytes(
        canonical_json(
            {
                "started_at": shard_started_at,
                "finished_at": shard_finished_at,
                "elapsed_seconds": elapsed_seconds,
                "workspace_products": product_timing,
                "targets": target_timings,
            }
        )
    )
    listing_output.write_bytes(
        canonical_json(
            {
                "selected_targets": selected,
                "selected_tests": sorted(selected_tests),
                "executed_targets": sorted(executed_targets),
                "executed_tests": sorted(executed_tests),
            }
        )
    )
    sidecars = {
        name: sha256_file(output / name)
        for name in SIDECAR_NAMES
    }
    receipt: dict[str, Any] = {
        "version": RECEIPT_VERSION,
        "lane": lane,
        "shard": shard,
        "plan_digest": plan["plan_digest"],
        "selected_targets": selected,
        "executed_targets": sorted(executed_targets),
        "selected_tests": sorted(selected_tests),
        "executed_tests": sorted(executed_tests),
        "commands_file": command_output.name,
        "timings_file": timing_output.name,
        "test_list_file": listing_output.name,
        "junit_file": junit_output.name,
        "started_at": shard_started_at,
        "finished_at": shard_finished_at,
        "elapsed_seconds": elapsed_seconds,
        "soft_budget_seconds": soft_budget_seconds,
        "soft_budget_exceeded": soft_budget_exceeded,
        "sidecars": sidecars,
        "success": not failures,
        "failures": failures,
    }
    attach_receipt_digest(receipt)
    receipt_output.write_bytes(canonical_json(receipt))
    shutil.rmtree(scratch)
    return receipt


def prepare_shard(
    plan: Mapping[str, Any],
    *,
    lane: str,
    shard: int,
    output: Path,
    github_output: Path,
    repo: Path = ROOT,
) -> dict[str, Any] | None:
    """Write an empty receipt or tell the hosted worker to prepare execution."""
    verify_plan_digest(plan)
    if lane not in LANE_KEYS:
        raise ValueError(f"unsupported lane: {lane}")
    if shard not in SHARDS:
        raise ValueError(f"shard must be one of {SHARDS}")
    selected = plan["shards"][LANE_KEYS[lane]][str(shard)]
    has_targets = bool(selected)
    github_output.parent.mkdir(parents=True, exist_ok=True)
    github_output.write_text(
        f"has_targets={'true' if has_targets else 'false'}\n"
    )
    if has_targets:
        return None
    return execute_shard(
        plan,
        lane=lane,
        shard=shard,
        output=output,
        repo=repo,
    )


def _report_findings(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    lane: str,
) -> list[str]:
    verify_plan_digest(plan)
    lane_key = LANE_KEYS[lane]
    findings: list[str] = []
    by_shard: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        try:
            verify_receipt_digest(receipt)
        except ValueError as error:
            findings.append(str(error))
            continue
        if receipt.get("lane") != lane:
            findings.append(
                f"receipt lane mismatch: expected {lane}, got {receipt.get('lane')}"
            )
        shard = receipt.get("shard")
        if type(shard) is not int or shard not in SHARDS:
            findings.append(f"invalid receipt shard: {shard!r}")
            continue
        if shard in by_shard:
            findings.append(f"duplicate receipt shard: {shard}")
        else:
            by_shard[shard] = receipt
        if receipt.get("plan_digest") != plan["plan_digest"]:
            findings.append(f"receipt shard {shard} plan digest mismatch")
    missing = sorted(set(SHARDS) - set(by_shard))
    if missing:
        findings.append(f"missing receipt shards: {missing}")

    all_selected_targets: list[str] = []
    all_executed_targets: list[str] = []
    all_selected_tests: list[str] = []
    all_executed_tests: list[str] = []
    for shard, receipt in sorted(by_shard.items()):
        expected = plan["shards"][lane_key][str(shard)]
        selected = receipt.get("selected_targets")
        if selected != expected:
            findings.append(
                f"shard {shard} selected targets mismatch: "
                f"expected={expected}, got={selected}"
            )
        if not isinstance(selected, list):
            selected = []
        executed = receipt.get("executed_targets")
        if not isinstance(executed, list):
            findings.append(f"shard {shard} executed_targets is not a list")
            executed = []
        selected_tests = receipt.get("selected_tests")
        executed_tests = receipt.get("executed_tests")
        if not isinstance(selected_tests, list) or not isinstance(executed_tests, list):
            findings.append(f"shard {shard} test lists are malformed")
            selected_tests, executed_tests = [], []
        all_selected_targets.extend(selected)
        all_executed_targets.extend(executed)
        all_selected_tests.extend(selected_tests)
        all_executed_tests.extend(executed_tests)
        if receipt.get("success") is not True:
            findings.append(
                f"shard {shard} did not succeed: {receipt.get('failures', [])}"
            )
        if lane == "package-expansion" and receipt["soft_budget_exceeded"]:
            findings.append(
                f"shard {shard} exceeded the {SOFT_BUDGET_SECONDS}s soft budget: "
                f"{receipt['elapsed_seconds']}s"
            )

    expected_targets = sorted(
        identity
        for rows in plan["shards"][lane_key].values()
        for identity in rows
    )
    if sorted(all_selected_targets) != expected_targets:
        findings.append(
            "selected target coverage mismatch: "
            f"expected={expected_targets}, got={sorted(all_selected_targets)}"
        )
    if sorted(all_executed_targets) != expected_targets:
        findings.append(
            "executed target coverage mismatch: "
            f"expected={expected_targets}, got={sorted(all_executed_targets)}"
        )
    if len(all_executed_targets) != len(set(all_executed_targets)):
        findings.append("duplicate target execution")
    if len(all_executed_tests) != len(set(all_executed_tests)):
        findings.append("duplicate test execution")
    if sorted(all_selected_tests) != sorted(all_executed_tests):
        findings.append(
            "selected test coverage mismatch: "
            f"selected={sorted(all_selected_tests)}, "
            f"executed={sorted(all_executed_tests)}"
        )

    excluded_targets = {
        row["identity"] for row in plan.get("target_exclusions", [])
    }
    forbidden_targets = sorted(excluded_targets & set(all_executed_targets))
    if forbidden_targets:
        findings.append(f"executed target exclusions: {forbidden_targets}")
    excluded_tests = {row["identity"] for row in plan.get("test_exclusions", [])}
    forbidden_tests = sorted(excluded_tests & set(all_executed_tests))
    if forbidden_tests:
        findings.append(f"executed test exclusions: {forbidden_tests}")
    return findings


def validate_change_owned_report(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    standing_coverage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    findings = _report_findings(plan, receipts, "change-owned")
    reused_targets = validate_standing_coverage(plan, standing_coverage)
    if findings:
        raise ValueError("; ".join(findings))
    return {
        "version": 1,
        "lane": "change-owned",
        "required": True,
        "success": True,
        "observed_success": True,
        "plan_digest": plan["plan_digest"],
        "covered_targets": sorted(plan["change_owned"]),
        "standing_reused_targets": reused_targets,
        "failures": [],
    }


def summarize_package_expansion(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    findings = _report_findings(plan, receipts, "package-expansion")
    return {
        "version": 1,
        "lane": "package-expansion",
        "required": False,
        "success": not findings,
        "observed_success": not findings,
        "plan_digest": plan["plan_digest"],
        "covered_targets": sorted(plan["package_expansion"]),
        "standing_reused_targets": [],
        "failures": findings,
    }


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def load_receipts(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("receipt.json"))
    receipts = []
    for path in paths:
        receipt = load_json(path)
        verify_receipt_digest(receipt)
        for name, expected in receipt["sidecars"].items():
            sidecar = path.parent / name
            if not sidecar.is_file():
                raise ValueError(f"missing receipt sidecar: {sidecar}")
            observed = sha256_file(sidecar)
            if observed != expected:
                raise ValueError(
                    f"sidecar digest mismatch for {sidecar}: "
                    f"expected {expected}, got {observed}"
                )
        receipts.append(receipt)
    return receipts


def _write_report_files(output: Path, report: Mapping[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_bytes(canonical_json(report))
    lines = [
        f"# {report['lane']} report",
        "",
        f"- Required: {str(report['required']).lower()}",
        f"- Observed success: {str(report['observed_success']).lower()}",
        f"- Plan digest: `{report['plan_digest']}`",
        f"- Covered targets: {len(report['covered_targets'])}",
        f"- Reused standing targets: "
        f"{len(report.get('standing_reused_targets', []))}",
        f"- Findings: {len(report['failures'])}",
    ]
    for finding in report["failures"]:
        lines.append(f"  - {finding}")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="generate the exact event plan")
    plan.add_argument("--event-name", required=True)
    plan.add_argument("--pr-head", default="")
    plan.add_argument("--before", default="")
    plan.add_argument("--after", default="")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--config",
        type=Path,
        default=ROOT / ".config/ci-test-targets.toml",
    )

    run_shard = subparsers.add_parser("run-shard", help="execute one plan shard")
    run_shard.add_argument("--plan", type=Path, required=True)
    run_shard.add_argument("--lane", choices=tuple(LANE_KEYS), required=True)
    run_shard.add_argument("--shard", type=int, choices=SHARDS, required=True)
    run_shard.add_argument("--output", type=Path, required=True)

    prepare_shard_parser = subparsers.add_parser(
        "prepare-shard",
        help="emit empty evidence or select a shard for hosted preparation",
    )
    prepare_shard_parser.add_argument("--plan", type=Path, required=True)
    prepare_shard_parser.add_argument(
        "--lane", choices=tuple(LANE_KEYS), required=True
    )
    prepare_shard_parser.add_argument(
        "--shard", type=int, choices=SHARDS, required=True
    )
    prepare_shard_parser.add_argument("--output", type=Path, required=True)
    prepare_shard_parser.add_argument(
        "--github-output", type=Path, required=True
    )

    report = subparsers.add_parser("report", help="validate or summarize receipts")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--lane", choices=tuple(LANE_KEYS), required=True)
    report.add_argument("--receipts-root", type=Path, required=True)
    report.add_argument("--standing-coverage", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--required", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = generate_plan(
            ROOT,
            event_name=args.event_name,
            pr_head=args.pr_head,
            before=args.before,
            after=args.after,
            config_path=args.config,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(result))
        print(
            "CHANGE-OWNED PLAN: PASS: "
            f"{len(result['change_owned'])} required, "
            f"{len(result['package_expansion'])} informational"
        )
        return 0
    if args.command == "run-shard":
        plan = load_json(args.plan)
        receipt = execute_shard(
            plan,
            lane=args.lane,
            shard=args.shard,
            output=args.output,
        )
        print(
            f"{args.lane.upper()} SHARD {args.shard}: "
            f"{'PASS' if receipt['success'] else 'FAIL'}"
        )
        return 0
    if args.command == "prepare-shard":
        plan = load_json(args.plan)
        receipt = prepare_shard(
            plan,
            lane=args.lane,
            shard=args.shard,
            output=args.output,
            github_output=args.github_output,
        )
        disposition = "EMPTY" if receipt is not None else "SELECTED"
        print(f"{args.lane.upper()} SHARD {args.shard}: {disposition}")
        return 0

    if args.lane == "change-owned" and not args.required:
        raise ValueError("change-owned report requires --required")
    if args.lane == "package-expansion" and args.required:
        raise ValueError("package-expansion report must remain informational")
    plan = load_json(args.plan)
    if args.lane == "change-owned":
        try:
            receipts = load_receipts(args.receipts_root)
            standing_coverage = (
                load_json(args.standing_coverage)
                if args.standing_coverage is not None
                else None
            )
            result = validate_change_owned_report(
                plan,
                receipts,
                standing_coverage,
            )
        except (
            ValueError,
            KeyError,
            OSError,
            json.JSONDecodeError,
            ET.ParseError,
        ) as error:
            result = {
                "version": 1,
                "lane": "change-owned",
                "required": True,
                "success": False,
                "observed_success": False,
                "plan_digest": plan.get("plan_digest"),
                "covered_targets": [],
                "standing_reused_targets": [],
                "failures": [str(error)],
            }
            _write_report_files(args.output, result)
            print(f"CHANGE-OWNED REPORT: FAIL: {error}", file=sys.stderr)
            return 1
    else:
        try:
            receipts = load_receipts(args.receipts_root)
            result = summarize_package_expansion(plan, receipts)
        except (
            ValueError,
            KeyError,
            OSError,
            json.JSONDecodeError,
            ET.ParseError,
        ) as error:
            result = {
                "version": 1,
                "lane": "package-expansion",
                "required": False,
                "success": False,
                "observed_success": False,
                "plan_digest": plan.get("plan_digest"),
                "covered_targets": [],
                "standing_reused_targets": [],
                "failures": [f"informational report validation failed: {error}"],
            }
    _write_report_files(args.output, result)
    print(
        f"{args.lane.upper()} REPORT: "
        f"{'PASS' if result['observed_success'] else 'RECORDED FAILURES'}"
    )
    return 0 if result["observed_success"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        ET.ParseError,
    ) as error:
        print(f"CHANGE-OWNED CI: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
