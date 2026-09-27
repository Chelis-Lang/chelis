#!/usr/bin/env python3
"""Plan and execute package-qualified change-owned integration tests.

The planner classifies every changed path, compares exact base/candidate Cargo
metadata, and emits two disjoint four-shard selections:

* ``change-owned`` is required and contains added/directly modified eligible
  integration targets plus every eligible target in packages selected by a
  reviewed required-package migration rule.
* ``package-expansion`` is informational and contains the remaining eligible
  targets in directly changed or explicitly selected packages.

Every persisted plan and shard receipt is content-digested. Reports reject
missing, duplicate, mismatched, excluded, uncovered, or unsuccessful required
execution.

The informational package-expansion report answers a different question, so it
classifies rather than repeating the required lane's verdict. Each observed test
failure is differenced against a recorded default-branch failure baseline and
reported as *introduced* or *inherited*, and each selected target with no
execution evidence is reported as *unrun*. Those three counts are disjoint and
none absorbs another: a target the lane could not reach is never reported as
inherited. The report is clean when nothing was introduced. Inherited failures
and unrun coverage are reported as the numbers they are, because a lane whose
verdict is decided by the state of the default branch cannot say anything about
the candidate. A missing baseline, or one the candidate's own history does not
contain, fails the report loudly instead of differencing against the wrong
tree.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
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
SCHEMA_VERSION = 3
PLAN_VERSION = 5
RECEIPT_VERSION = 2
STANDING_COVERAGE_VERSION = 1
DURATION_BASELINE_VERSION = 1
DURATION_BASELINE_PATH = ROOT / ".config/ci-change-owned-durations.json"
DEFAULT_DURATION_MILLISECONDS = 30_000
CHANGE_OWNED_SHARD_ALGORITHM = "duration-lpt-v1"
PACKAGE_EXPANSION_SHARD_ALGORITHM = "duration-lpt-v1"
PACKAGE_EXPANSION_COMPATIBILITY_ALGORITHM = "sha256-modulo-v1"
PACKAGE_EXPANSION_EXECUTION_KIND = "package_expansion_duration_plan"
PACKAGE_EXPANSION_GROUP_TARGET_LIMIT = 16
PACKAGE_EXPANSION_GROUP_ESTIMATED_MILLISECONDS = 300_000
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
MANUAL_GATE_FIELDS = {"package", "name", "manual_gates", "reason", "tracking_issue"}
# A manual_gate_target row cites rows of this one table by their first cell.
MANUAL_GATES_PATH = "docs/manual_gates.md"
MANUAL_GATES_SECTION = "## Ignored tests (wired manual gates)"
MANUAL_GATES_HEADER = (
    "Test",
    "Crate",
    "Manual command",
    "Prerequisite",
    "Owning phase",
)
MANUAL_GATE_STATUS = "manual gate, not executed in PR CI"
SIDECAR_NAMES = ("commands.json", "timings.json", "test-list.json", "junit.xml")
# The informational lane's deadline is a backstop against a hung command, not
# a schedule. Sized at 80 minutes it cuts none of the 38 dispatches measured
# under the current planner, whose longest shard projects to a median of 1580s
# and a maximum of 4410s; 45 minutes would still cut six of them and 60 would
# still cut one. It is deliberately not sized to the plan's per-shard estimate,
# which is a longest-processing-time balancing weight rather than a predicted
# duration and runs a median 3.73x over on shards that finish.
#
# `timeout-minutes` on the worker job in `pr-package-expansion.yml` is the
# limit that actually binds, and this one sits ten minutes under it. That gap
# is consumed by job setup rather than by finalization: checkout, apt, the
# toolchain, uv, the cache restore and nextest take a measured median of 118s
# and a maximum of 211s across the last 30 dispatches, and they run before the
# executor's clock starts, while merging and digesting even a 98 MB JUnit takes
# under two seconds. Raising this constant alone would move the failure from
# "partial report written" to "no report at all", so the two move together.
EXPANSION_EXECUTION_SECONDS = 80 * 60
# Reporting only since chelis#2248 removed its finding: an early warning that
# a shard came close to the wall. Derived rather than set independently, so the
# pair cannot drift apart; the bounds the tests enforce keep the warning window
# from degenerating at either end.
#
# Ten minutes rather than the permitted floor, for a reason the bounds cannot
# express. `soft_budget_exceeded` is computed once from total elapsed, so a
# shard the deadline cut always warns whatever the margin; the flag only
# discriminates among shards that finished, separating "finished inside the
# last margin seconds" from "finished comfortably". Elapsed time advances in
# whole commands, so a window narrower than one command is jumped over rather
# than landed in, and the flag decays into a synonym for `not success`. The
# longest single command observed to date is about 593s, the
# `runtime_extent_claim_preparation` run that caused chelis#2251's cut, so 600s
# clears it by seven seconds. That is thin, and one slower command would make
# it thinner; the floor is deliberately not anchored to that measurement,
# because a maximum committed into the repository goes stale the first time a
# slower target lands.
EXPANSION_SOFT_BUDGET_MARGIN_SECONDS = 10 * 60
SOFT_BUDGET_SECONDS = (
    EXPANSION_EXECUTION_SECONDS - EXPANSION_SOFT_BUDGET_MARGIN_SECONDS
)
EXPANSION_REPORT_VERSION = 2
FAILURE_BASELINE_VERSION = 1
# The report runs from the candidate checkout, so a new flag in its own
# invocation is rejected outright by a candidate whose base predates this
# change. A conventional path costs that candidate nothing: its parser never
# sees the argument, and it reports exactly as it does today.
DEFAULT_FAILURE_BASELINE = PurePosixPath(
    "target/integration-change/failure-baseline/baseline.json"
)
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
STANDING_EXECUTION = {
    "profile": "ci-fast",
    "ignore_default_filter": True,
    "run_ignored": "default",
    "no_fail_fast": True,
}
TARGETED_REBASE_REUSED_OWNER_JOBS = frozenset({("ci.yml", "ci-fast")})
TARGETED_REBASE_SUPPORTED_OWNER_JOBS = frozenset(
    {
        ("ci.yml", "docs"),
        ("ci.yml", "lint-rust"),
        ("ci.yml", "script-unit"),
        ("conformance.yml", "conformance"),
    }
)


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
class DurationBaseline:
    default_milliseconds: int
    targets: Mapping[Identity, int]
    digest: str


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
class ManualGate:
    entries: tuple[str, ...]
    reason: str
    tracking_issue: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "manual_gates": list(self.entries),
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
class RequiredPackageRule:
    prefix: str
    packages: tuple[str, ...]
    reason: str
    tracking_issue: str

    def matches(self, path: str) -> bool:
        if self.prefix.endswith("/"):
            return path.startswith(self.prefix)
        return path == self.prefix


@dataclass(frozen=True)
class Config:
    version: int
    standing_targets: tuple[Identity, ...]
    manual_only_targets: Mapping[Identity, Owner]
    manual_gate_targets: Mapping[Identity, ManualGate]
    target_exclusions: Mapping[Identity, Owner]
    test_exclusions: Mapping[TestIdentity, Owner]
    required_package_rules: tuple[RequiredPackageRule, ...]
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
    """Read the strict version-3 seven-row-kind configuration."""
    data = tomllib.loads(path.read_text())
    allowed = {
        "version",
        "standing_target",
        "manual_only_target",
        "manual_gate_target",
        "target_exclusion",
        "test_exclusion",
        "required_package_rule",
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

    manual_gate_targets: dict[Identity, ManualGate] = {}
    for row in _rows(data, "manual_gate_target"):
        if set(row) != MANUAL_GATE_FIELDS:
            raise ValueError(
                "manual_gate_target requires exactly package, name, "
                "manual_gates, reason, and tracking_issue"
            )
        identity = Identity(_identifier(row, "package"), _identifier(row, "name"))
        if identity in manual_gate_targets:
            raise ValueError(f"duplicate manual_gate_target: {identity.canonical}")
        entries = row["manual_gates"]
        if (
            not isinstance(entries, list)
            or not entries
            or any(
                not isinstance(entry, str) or not entry or entry != entry.strip()
                for entry in entries
            )
            or len(set(entries)) != len(entries)
        ):
            raise ValueError(
                "manual_gate_target manual_gates must be unique exact "
                f"{MANUAL_GATES_PATH} row identifiers"
            )
        tracking_issue = _nonempty_string(row, "tracking_issue")
        if not ISSUE.fullmatch(tracking_issue):
            raise ValueError(
                "manual_gate_target tracking_issue must be chelis#N: "
                f"{tracking_issue!r}"
            )
        manual_gate_targets[identity] = ManualGate(
            tuple(entries),
            _nonempty_string(row, "reason"),
            tracking_issue,
        )

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
    gate_conflicts = sorted(
        item.canonical
        for item in set(manual_gate_targets)
        & (set(standing) | set(target_exclusions) | set(manual_only_targets))
    )
    if gate_conflicts:
        raise ValueError(
            "manual-gate targets cannot be standing, manual-only, or excluded "
            f"targets: {gate_conflicts}"
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
    gate_test_exclusions = sorted(
        identity.canonical
        for identity in test_exclusions
        if identity.target_identity in manual_gate_targets
    )
    if gate_test_exclusions:
        raise ValueError(
            "manual-gate targets must list their complete ignored suite; "
            f"test exclusions are forbidden: {gate_test_exclusions}"
        )

    required_package_rules: list[RequiredPackageRule] = []
    for row in _rows(data, "required_package_rule"):
        if set(row) != {
            "prefix",
            "packages",
            "reason",
            "tracking_issue",
        }:
            raise ValueError(
                "required_package_rule requires prefix, packages, reason, "
                "and tracking_issue"
            )
        prefix = _nonempty_string(row, "prefix")
        if not _valid_prefix(prefix):
            raise ValueError(
                f"invalid required_package_rule prefix: {prefix!r}"
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
            raise ValueError(
                "required_package_rule packages must be unique exact package names"
            )
        reason = _nonempty_string(row, "reason")
        tracking_issue = _nonempty_string(row, "tracking_issue")
        if not ISSUE.fullmatch(tracking_issue):
            raise ValueError(
                "required_package_rule tracking_issue must be chelis#N: "
                f"{tracking_issue!r}"
            )
        for existing in required_package_rules:
            if _rules_overlap(existing.prefix, prefix):
                raise ValueError(
                    "ambiguous required package rules overlap: "
                    f"{existing.prefix!r}, {prefix!r}"
                )
        required_package_rules.append(
            RequiredPackageRule(
                prefix,
                tuple(raw_packages),
                reason,
                tracking_issue,
            )
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
        manual_gate_targets,
        target_exclusions,
        test_exclusions,
        tuple(required_package_rules),
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


def workspace_reverse_dependency_closure(
    metadata: Mapping[str, Any], seeds: Iterable[str]
) -> set[str]:
    """Include every workspace package whose build can consume a seed package."""
    packages = _workspace_packages(metadata)
    names = {package["name"] for package in packages}
    closure = set(seeds)
    unknown = sorted(closure - names)
    if unknown:
        raise ValueError(f"unknown workspace package seeds: {unknown}")
    dependencies: dict[str, set[str]] = {}
    for package in packages:
        rows = package.get("dependencies", [])
        if not isinstance(rows, list):
            raise ValueError(
                f"malformed dependency list for {package.get('name')}"
            )
        direct: set[str] = set()
        for dependency in rows:
            if not isinstance(dependency, Mapping):
                raise ValueError(
                    f"malformed dependency row for {package.get('name')}"
                )
            name = dependency.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"malformed dependency name for {package.get('name')}"
                )
            if name in names:
                direct.add(name)
        dependencies[package["name"]] = direct
    changed = True
    while changed:
        changed = False
        for package, direct in dependencies.items():
            if package not in closure and direct & closure:
                closure.add(package)
                changed = True
    return closure


def targeted_rebase_frontier(
    paths: Sequence[str],
    *,
    base_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    config: Config,
) -> dict[str, object]:
    """Classify the exact synthetic-candidate delta into executable owners."""
    base_targets = all_integration_targets(base_metadata)
    candidate_targets = all_integration_targets(candidate_metadata)
    base_packages = package_infos(base_metadata)
    candidate_packages = package_infos(candidate_metadata)
    candidate_names = {package.name for package in candidate_packages}
    seeds: set[str] = set()
    owner_jobs: set[tuple[str, str]] = set()
    unsafe_paths: set[str] = set()

    for path in paths:
        targets = {
            identity
            for identity in (
                *_target_at_path(path, base_targets),
                *_target_at_path(path, candidate_targets),
            )
        }
        if len(targets) > 1:
            unsafe_paths.add(path)
            continue
        if targets:
            if any(identity in config.target_exclusions for identity in targets):
                unsafe_paths.add(path)
                continue
            package = next(iter(targets)).package
            if package not in candidate_names:
                unsafe_paths.add(path)
            else:
                seeds.add(package)
            continue

        classification, package_matches, matching_rules = static_path_classification(
            path, (*base_packages, *candidate_packages), config
        )
        if classification == "package":
            package = package_matches[0]
            if package not in candidate_names:
                unsafe_paths.add(path)
            else:
                seeds.add(package)
            continue
        if classification == "docs":
            continue
        if classification != "rule":
            unsafe_paths.add(path)
            continue
        rule = matching_rules[0]
        if rule.disposition == "packages":
            if not set(rule.packages) <= candidate_names:
                unsafe_paths.add(path)
            else:
                seeds.update(rule.packages)
            continue
        if rule.owner is None:
            unsafe_paths.add(path)
            continue
        owner = (rule.owner.workflow, rule.owner.job)
        if owner not in TARGETED_REBASE_SUPPORTED_OWNER_JOBS:
            unsafe_paths.add(path)
            continue
        owner_jobs.add(owner)

    packages = (
        workspace_reverse_dependency_closure(candidate_metadata, seeds)
        if seeds
        else set()
    )
    return {
        "packages": sorted(packages),
        "owner_jobs": [
            {"workflow": workflow, "job": job}
            for workflow, job in sorted(owner_jobs)
        ],
        "unsafe_paths": sorted(unsafe_paths),
    }


def test_functions(source: str) -> set[str]:
    return set(TEST_FUNCTION.findall(source))


def _manual_gate_cells(line: str) -> list[str]:
    stripped = line.strip()
    cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", stripped)[1:-1]]
    if not stripped.endswith("|") or len(cells) != len(MANUAL_GATES_HEADER):
        raise ValueError(f"malformed {MANUAL_GATES_PATH} table row: {line!r}")
    return cells


def manual_gate_entries(document: str) -> dict[str, list[str]]:
    """Map each wired manual-gate row identifier to its command cells.

    The identifier is the row's first cell with its code-span backticks
    removed. A repeated identifier keeps every command, so a citation of it is
    rejected as ambiguous rather than resolved to one of them.
    """
    lines = document.splitlines()
    if lines.count(MANUAL_GATES_SECTION) != 1:
        raise ValueError(
            f"{MANUAL_GATES_PATH} must contain exactly one "
            f"{MANUAL_GATES_SECTION!r} section"
        )
    rows = []
    for line in lines[lines.index(MANUAL_GATES_SECTION) + 1 :]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            rows.append(_manual_gate_cells(line))
    if (
        len(rows) < 2
        or tuple(rows[0]) != MANUAL_GATES_HEADER
        or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in rows[1])
    ):
        raise ValueError(
            f"{MANUAL_GATES_PATH} {MANUAL_GATES_SECTION!r} has no "
            f"{' | '.join(MANUAL_GATES_HEADER)} table"
        )
    entries: dict[str, list[str]] = {}
    for cells in rows[2:]:
        entries.setdefault(cells[0].replace("`", "").strip(), []).append(cells[2])
    return entries


# A manual-gate command cell is one code span of shell words joined by these
# operators. A segment is classified only when every word is plain and is an
# argument the runner below knows; anything else is rejected, never skipped.
COMMAND_SEPARATORS = {"&&", "||", ";", "|"}
PLAIN_WORD = re.compile(r"[A-Za-z0-9_.,/:=-]+")
ENV_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
# The value-taking options each runner accepts before ``--``, by role, and the
# harness options and flags after it. A bare word is a test-name filter.
CARGO_TEST_RUNNERS: dict[tuple[str, ...], dict[str, str]] = {
    ("cargo", "test"): {"-p": "package", "--package": "package", "--test": "test"},
    ("cargo", "nextest", "run"): {
        "-p": "package",
        "--package": "package",
        "--test": "test",
        "--run-ignored": "run-ignored",
    },
}
HARNESS_OPTIONS = {"--test-threads": "setting"}
HARNESS_FLAGS = {"--ignored", "--exact", "--nocapture"}


@dataclass(frozen=True)
class CargoTestRun:
    packages: tuple[str, ...]
    tests: tuple[str, ...]
    ignored: bool
    filtered: bool


def _arguments(
    args: list[str],
    valued: Mapping[str, str],
    flags: frozenset[str] | set[str] = frozenset(),
) -> list[tuple[str, str]]:
    """Return ``(role, value)`` for each argument, rejecting any other spelling."""
    parsed = []
    index = 0
    while index < len(args):
        arg = args[index]
        name, equals, value = arg.partition("=")
        if equals and name.startswith("--") and name in valued:
            parsed.append((valued[name], value))
            index += 1
        elif (
            arg in valued
            and index + 1 < len(args)
            and not args[index + 1].startswith("-")
        ):
            parsed.append((valued[arg], args[index + 1]))
            index += 2
        elif arg in flags:
            parsed.append((arg, ""))
            index += 1
        elif not arg.startswith("-"):
            parsed.append(("filter", arg))
            index += 1
        else:
            raise ValueError(f"unrecognized argument {arg!r}")
    return parsed


def _cargo_test_run(segment: list[str]) -> CargoTestRun:
    """Classify one command segment as a Cargo test run, or reject it."""
    for word in segment:
        if not PLAIN_WORD.fullmatch(word):
            raise ValueError(f"unrecognized word {word!r}")
    while segment and ENV_ASSIGNMENT.fullmatch(segment[0]):
        segment = segment[1:]
    runner = next(
        (
            runner
            for runner in CARGO_TEST_RUNNERS
            if tuple(segment[: len(runner)]) == runner
        ),
        None,
    )
    if runner is None:
        raise ValueError("not `cargo test` or `cargo nextest run`")
    args = segment[len(runner) :]
    split = args.index("--") if "--" in args else len(args)
    arguments = _arguments(args[:split], CARGO_TEST_RUNNERS[runner]) + _arguments(
        args[split + 1 :], HARNESS_OPTIONS, HARNESS_FLAGS
    )
    run_ignored = [value for role, value in arguments if role == "run-ignored"]
    if not set(run_ignored) <= {"all", "only"}:
        raise ValueError(f"unrecognized --run-ignored value in {run_ignored}")
    return CargoTestRun(
        packages=tuple(value for role, value in arguments if role == "package"),
        tests=tuple(value for role, value in arguments if role == "test"),
        ignored=bool(run_ignored) or ("--ignored", "") in arguments,
        filtered=any(role == "filter" for role, _ in arguments),
    )


def _command_segments(command: str) -> list[list[str]]:
    """Split one code-span command cell into its operator-separated words."""
    span = re.fullmatch(r"`([^`]+)`", command)
    if span is None:
        raise ValueError(f"manual gate command is not one code span: {command!r}")
    try:
        words = shlex.split(span.group(1))
    except ValueError as error:
        raise ValueError(f"manual gate command does not parse: {command!r}") from error
    segments: list[list[str]] = [[]]
    for word in words:
        if word in COMMAND_SEPARATORS:
            segments.append([])
        else:
            segments[-1].append(word)
    return segments


def _classify(name: str, command: str, segments: list[list[str]]) -> list[CargoTestRun]:
    """Classify every given segment of a wired entry's command, or reject it.

    Quoting is rejected so that the classified words are the words the shell
    passes to Cargo.
    """
    if segments and re.search(r"[\\'\"]", command):
        raise ValueError(
            f"{MANUAL_GATES_PATH} entry {name!r} quotes or escapes a word: {command}"
        )
    runs = []
    for segment in segments:
        try:
            runs.append(_cargo_test_run(segment))
        except ValueError as error:
            raise ValueError(
                f"{MANUAL_GATES_PATH} entry {name!r} has a segment that is not a "
                f"recognized Cargo test run ({error}): {command}"
            ) from error
    return runs


def _names(target: str, text: str) -> bool:
    """Whether ``text`` names ``target`` as a whole word."""
    return re.search(rf"(?<![\w-]){re.escape(target)}(?![\w-])", text) is not None


def validate_manual_gate_entries(
    targets: Mapping[Identity, ManualGate],
    document: str,
) -> None:
    """Bind every manual-gate row to exactly the wired entries that run it.

    Each cited entry must exist once, and every segment of its command must be
    a Cargo test run of exactly the row's package and target with its ignored
    tests; at least one cited entry must run them with no name filter. An
    uncited entry that runs the target makes the row stale, and a segment of
    any entry that names the target but cannot be classified is rejected.
    """
    entries = manual_gate_entries(document)
    for identity, gate in sorted(targets.items()):
        exact = ((identity.package,), (identity.target,))
        whole_suite = False
        for name in gate.entries:
            commands = entries.get(name, [])
            if len(commands) != 1:
                raise ValueError(
                    f"manual-gate target {identity.canonical} cites {name!r}, "
                    f"which names {len(commands)} rows of {MANUAL_GATES_PATH} "
                    f"{MANUAL_GATES_SECTION!r}; exactly one is required"
                )
            runs = _classify(name, commands[0], _command_segments(commands[0]))
            if any(
                (run.packages, run.tests) != exact or not run.ignored for run in runs
            ):
                raise ValueError(
                    f"{MANUAL_GATES_PATH} entry {name!r} does not run exactly "
                    f"{identity.canonical} with its ignored tests: {commands[0]}"
                )
            whole_suite = whole_suite or any(not run.filtered for run in runs)
        if not whole_suite:
            raise ValueError(
                f"manual-gate target {identity.canonical} cites no "
                f"{MANUAL_GATES_PATH} entry that runs its whole ignored suite "
                "without a name filter"
            )
        uncited = []
        for name, commands in sorted(entries.items()):
            if name in gate.entries:
                continue
            for command in commands:
                # A command that does not name the target is left alone, so an
                # unrelated row the parser cannot read fails no plan.
                try:
                    segments = _command_segments(command)
                except ValueError:
                    if _names(identity.target, command):
                        raise
                    continue
                named = [
                    segment
                    for segment in segments
                    if any(_names(identity.target, word) for word in segment)
                ]
                if any(
                    identity.package in run.packages and identity.target in run.tests
                    for run in _classify(name, command, named)
                ):
                    uncited.append(name)
        if uncited:
            raise ValueError(
                f"manual-gate target {identity.canonical} does not cite "
                f"{MANUAL_GATES_PATH} entries that run it: {sorted(set(uncited))}"
            )


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
    for identity in config.manual_gate_targets:
        if identity.package not in packages:
            raise ValueError(f"stale manual-gate package: {identity.package}")
        if identity not in all_targets:
            raise ValueError(f"stale manual-gate target: {identity.canonical}")
    if config.manual_gate_targets:
        try:
            document = source_reader(MANUAL_GATES_PATH)
        except (OSError, KeyError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"cannot read {MANUAL_GATES_PATH} for manual-gate targets"
            ) from error
        validate_manual_gate_entries(config.manual_gate_targets, document)
    for identity in config.target_exclusions:
        if identity.package not in packages:
            raise ValueError(f"stale exclusion package: {identity.package}")
        if identity not in all_targets:
            raise ValueError(f"stale target exclusion: {identity.canonical}")
    for identity in config.test_exclusions:
        target = all_targets.get(identity.target_identity)
        if identity.package not in packages:
            raise ValueError(f"stale test-exclusion package: {identity.package}")
        if target is None:
            raise ValueError(
                f"stale test-exclusion target: {identity.target_identity.canonical}"
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
    for rule in config.required_package_rules:
        stale_packages = sorted(set(rule.packages) - packages)
        if stale_packages:
            raise ValueError(
                f"required package rule {rule.prefix!r} names stale packages: "
                f"{stale_packages}"
            )
        if not any(rule.matches(path) for path in tracked_paths):
            raise ValueError(
                "stale required package rule matches no tracked path: "
                f"{rule.prefix}"
            )


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
        rename_score = status[1:]
        valid_rename = (
            kind == "R"
            and re.fullmatch(r"[0-9]{1,3}", rename_score) is not None
            and int(rename_score) <= 100
        )
        if status not in {"A", "D", "M", "T"} and not valid_rename:
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


def refused_path_message(classification: str, path: str) -> str:
    """The one spelling of a refusal, shared by the planner and `--fast`.

    A developer who hits this locally and a developer who reads a failed
    `Plan Changed Integration Tests` log should be reading the same sentence,
    which is only true if there is one of it.
    """
    qualifier = "ambiguous" if classification != "unclassified" else "unclassified"
    return f"{qualifier} changed path: {path}"


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


def targeted_rebase_preflight_path_classification(
    path: str,
    *,
    base_metadata: Mapping[str, Any],
    candidate_metadata: Mapping[str, Any],
    config: Config,
) -> str:
    """Classify one path against exact targets and targeted-lane owner execution."""
    base_targets = _target_at_path(path, all_integration_targets(base_metadata))
    candidate_targets = _target_at_path(
        path, all_integration_targets(candidate_metadata)
    )
    if len(base_targets) > 1 or len(candidate_targets) > 1:
        return "ambiguous_integration_target"
    if base_targets or candidate_targets:
        return "integration_target"
    classification, _, matching_rules = static_path_classification(
        path,
        (*package_infos(base_metadata), *package_infos(candidate_metadata)),
        config,
    )
    if classification == "rule":
        rule = matching_rules[0]
        if (
            rule.disposition == "owner"
            and rule.owner is not None
            and (rule.owner.workflow, rule.owner.job)
            in TARGETED_REBASE_REUSED_OWNER_JOBS
        ):
            return "reused_standing_owner"
    return classification


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


def _manual_gate_rows(
    targets: Mapping[Identity, ManualGate],
) -> list[dict[str, Any]]:
    return [
        {"identity": identity.canonical, **gate.as_dict()}
        for identity, gate in sorted(targets.items())
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
        "manual_gate_targets": _manual_gate_rows(config.manual_gate_targets),
        "target_exclusions": _owned_target_rows(config.target_exclusions),
        "test_exclusions": [
            {"identity": identity.canonical, "owner": _owner_dict(owner)}
            for identity, owner in sorted(config.test_exclusions.items())
        ],
        "required_package_rules": [
            {
                "prefix": rule.prefix,
                "packages": sorted(rule.packages),
                "reason": rule.reason,
                "tracking_issue": rule.tracking_issue,
            }
            for rule in sorted(
                config.required_package_rules,
                key=lambda row: row.prefix,
            )
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
    targeted_packages: Sequence[str] = (),
    duration_baseline: DurationBaseline | None = None,
) -> dict[str, Any]:
    if mode not in {"pull_request", "push", "targeted_rebase"}:
        raise ValueError(f"unsupported planning mode: {mode}")
    if duration_baseline is None:
        duration_baseline = DurationBaseline(
            default_milliseconds=DEFAULT_DURATION_MILLISECONDS,
            targets={},
            digest="0" * 64,
        )
    validate_config(
        config,
        candidate_metadata,
        tracked_paths | (base_tracked_paths or set()),
        source_reader,
    )

    base_all = all_integration_targets(base_metadata)
    candidate_all = all_integration_targets(candidate_metadata)
    candidate_eligible = candidate_all
    base_packages = package_infos(base_metadata)
    candidate_packages = package_infos(candidate_metadata)
    candidate_package_names = {package.name for package in candidate_packages}

    trusted_targeted_packages = set(targeted_packages)
    if len(trusted_targeted_packages) != len(targeted_packages):
        raise ValueError("targeted package frontier contains duplicates")
    if any(not IDENTIFIER.fullmatch(package) for package in targeted_packages):
        raise ValueError("targeted package frontier contains an invalid package")
    if mode != "targeted_rebase" and trusted_targeted_packages:
        raise ValueError(
            "only targeted_rebase planning accepts a targeted package frontier"
        )
    unknown_targeted_packages = sorted(
        trusted_targeted_packages - candidate_package_names
    )
    if unknown_targeted_packages:
        raise ValueError(
            "targeted package frontier contains unknown workspace packages: "
            f"{unknown_targeted_packages}"
        )

    selected_packages: set[str] = set(trusted_targeted_packages)
    required_packages: set[str] = set()
    change_owned: set[Identity] = set()
    dispositions: list[dict[str, Any]] = []
    target_dispositions: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    required_rows_by_path: dict[str, list[dict[str, object]]] = {}

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
        else:
            change_owned.add(identity)
            selected_packages.add(identity.package)
            if identity in config.manual_only_targets:
                row["execution_mode"] = "ignored-only"
            elif identity in config.manual_gate_targets:
                row["execution_mode"] = "manual-gate"
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
        required_rows = []
        for rule in config.required_package_rules:
            if not rule.matches(path):
                continue
            required_packages.update(rule.packages)
            selected_packages.update(rule.packages)
            required_rows.append(
                {
                    "rule": rule.prefix,
                    "packages": list(rule.packages),
                    "reason": rule.reason,
                    "tracking_issue": rule.tracking_issue,
                }
            )
        if required_rows:
            required_rows_by_path[path] = required_rows
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
                        else {"execution_mode": "manual-gate"}
                        if identity in config.manual_gate_targets
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
        if (
            classification == "unclassified"
            and side == "base"
            and base_tracked_paths is not None
            and path in base_tracked_paths
            and path not in tracked_paths
        ):
            # A removed shared input has no candidate execution owner. Record
            # its retirement, rather than retaining a dead rule or classifying
            # the removal as prose. Package/test removals were handled above;
            # ordinary CI and protected-contract removal checks still apply.
            dispositions.append(
                {"path": path, "status": status, "kind": "shared_path_deleted"}
            )
            continue
        if classification != "rule":
            raise ValueError(refused_path_message(classification, path))
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

    for disposition in dispositions:
        required_rows = required_rows_by_path.get(disposition["path"])
        if required_rows:
            disposition["required_package_rules"] = required_rows

    change_owned.update(
        identity
        for identity in candidate_eligible
        if identity.package in required_packages
        and identity not in config.target_exclusions
    )
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
        selected_packages = workspace_reverse_dependency_closure(
            candidate_metadata, selected_packages
        )
        expansion = {
            identity
            for identity in candidate_eligible
            if identity.package in selected_packages
            and identity not in change_owned
            and identity not in config.target_exclusions
        }
        change_owned |= expansion
        expansion = set()
        standing_coverage_reuse: set[Identity] = set()
        change_owned_execution = change_owned
    else:
        standing_coverage_reuse = change_owned & set(config.standing_targets)
        change_owned_execution = change_owned - standing_coverage_reuse

    target_exclusions, test_exclusions = _exclusion_rows(config)
    change_owned_shards, change_owned_planning = change_owned_shard_plan(
        change_owned_execution,
        duration_baseline,
    )
    expansion_shards, expansion_planning = package_expansion_shard_plan(
        expansion,
        duration_baseline,
    )
    target_dispositions.append(
        package_expansion_execution_disposition(
            expansion_shards,
            expansion_planning,
        )
    )
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
        "target_features": {
            identity.canonical: sorted(info.required_features)
            for identity, info in sorted(candidate_eligible.items())
        },
        "change_owned": sorted(identity.canonical for identity in change_owned),
        "package_expansion": sorted(identity.canonical for identity in expansion),
        "standing_targets": sorted(
            identity.canonical for identity in config.standing_targets
        ),
        "standing_coverage_reuse": sorted(
            identity.canonical for identity in standing_coverage_reuse
        ),
        "manual_only_targets": _owned_target_rows(config.manual_only_targets),
        "manual_gate_targets": _manual_gate_rows(config.manual_gate_targets),
        "target_exclusions": target_exclusions,
        "test_exclusions": test_exclusions,
        "shard_planning": {
            "change_owned": change_owned_planning,
            "package_expansion": {
                "algorithm": PACKAGE_EXPANSION_COMPATIBILITY_ALGORITHM,
            },
        },
        "shards": {
            "change_owned": change_owned_shards,
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


def _strict_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    try:
        data = json.loads(
            path.read_text(),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value in {path}: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed JSON {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def load_duration_baseline(path: Path = DURATION_BASELINE_PATH) -> DurationBaseline:
    data = _strict_json_object(path)
    expected_keys = {
        "version",
        "default_milliseconds",
        "sources",
        "targets",
    }
    if set(data) != expected_keys:
        raise ValueError(
            "duration baseline schema keys mismatch: "
            f"missing={sorted(expected_keys - set(data))}, "
            f"extra={sorted(set(data) - expected_keys)}"
        )
    if (
        type(data.get("version")) is not int
        or data["version"] != DURATION_BASELINE_VERSION
    ):
        raise ValueError(
            f"duration baseline version must be {DURATION_BASELINE_VERSION}"
        )
    default_milliseconds = _positive_int(
        data.get("default_milliseconds"),
        "duration baseline default_milliseconds",
    )
    if default_milliseconds != DEFAULT_DURATION_MILLISECONDS:
        raise ValueError(
            "duration baseline default_milliseconds must be "
            f"{DEFAULT_DURATION_MILLISECONDS}"
        )
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("duration baseline requires at least one source")
    source_keys: list[tuple[str, str]] = []
    for row in sources:
        if not isinstance(row, dict) or set(row) != {
            "candidate_sha",
            "plan_digest",
        }:
            raise ValueError("duration baseline source has the wrong shape")
        candidate_sha = row["candidate_sha"]
        plan_digest = row["plan_digest"]
        if not isinstance(candidate_sha, str) or not SHA.fullmatch(candidate_sha):
            raise ValueError("duration baseline source candidate_sha is malformed")
        if not isinstance(plan_digest, str) or not DIGEST.fullmatch(plan_digest):
            raise ValueError("duration baseline source plan_digest is malformed")
        source_keys.append((candidate_sha, plan_digest))
    if source_keys != sorted(set(source_keys)):
        raise ValueError("duration baseline sources must be unique and sorted")

    raw_targets = data.get("targets")
    if not isinstance(raw_targets, dict) or not raw_targets:
        raise ValueError("duration baseline requires at least one target")
    targets: dict[Identity, int] = {}
    for canonical, row in raw_targets.items():
        if not isinstance(canonical, str):
            raise ValueError("duration baseline target identity must be a string")
        identity = Identity.parse(canonical)
        if not isinstance(row, dict) or set(row) != {"milliseconds", "samples"}:
            raise ValueError(
                f"duration baseline target has the wrong shape: {canonical}"
            )
        milliseconds = _positive_int(
            row["milliseconds"],
            f"duration baseline {canonical} milliseconds",
        )
        _positive_int(
            row["samples"],
            f"duration baseline {canonical} samples",
        )
        targets[identity] = milliseconds
    return DurationBaseline(
        default_milliseconds=default_milliseconds,
        targets=targets,
        digest=sha256_file(path),
    )


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


def package_expansion_execution_disposition(
    shards: Mapping[str, list[str]],
    planning: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "kind": PACKAGE_EXPANSION_EXECUTION_KIND,
        "planning": dict(planning),
        "shards": {key: list(rows) for key, rows in shards.items()},
    }


def package_expansion_execution(
    plan: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = [
        row
        for row in plan["target_dispositions"]
        if isinstance(row, dict)
        and row.get("kind") == PACKAGE_EXPANSION_EXECUTION_KIND
    ]
    if len(rows) != 1:
        raise ValueError(
            "plan requires one package-expansion duration execution disposition"
        )
    execution = rows[0]
    if set(execution) != {"kind", "planning", "shards"}:
        raise ValueError(
            "plan package-expansion execution disposition has the wrong shape"
        )
    return execution


def _validated_duration_planning(
    planning: Any,
    expected_identities: set[str],
    *,
    algorithm: str,
    lane_label: str,
) -> dict[str, list[str]]:
    expected_keys = {
        "algorithm",
        "baseline_digest",
        "default_milliseconds",
        "weights_milliseconds",
        "estimated_milliseconds",
    }
    if not isinstance(planning, dict) or set(planning) != expected_keys:
        raise ValueError(f"plan {lane_label} shard planning has the wrong shape")
    if planning["algorithm"] != algorithm:
        raise ValueError(f"plan {lane_label} shard algorithm is unsupported")
    baseline_digest = planning["baseline_digest"]
    if (
        not isinstance(baseline_digest, str)
        or not DIGEST.fullmatch(baseline_digest)
    ):
        raise ValueError(
            f"plan {lane_label} duration baseline digest is malformed"
        )
    _positive_int(
        planning["default_milliseconds"],
        f"plan {lane_label} duration default_milliseconds",
    )
    raw_weights = planning["weights_milliseconds"]
    if not isinstance(raw_weights, dict):
        raise ValueError(
            f"plan {lane_label} duration weights must be an identity mapping"
        )
    if set(raw_weights) != expected_identities:
        raise ValueError(
            f"plan {lane_label} duration weights must exactly cover "
            "its execution targets"
        )
    weights: dict[Identity, int] = {}
    for canonical, milliseconds in raw_weights.items():
        weights[Identity.parse(canonical)] = _positive_int(
            milliseconds,
            f"plan {lane_label} duration weight for {canonical}",
        )
    shards, expected_estimates = duration_shard_map(weights)
    estimates = planning["estimated_milliseconds"]
    if (
        not isinstance(estimates, dict)
        or set(estimates) != {str(shard) for shard in SHARDS}
        or any(
            type(value) is not int or value < 0
            for value in estimates.values()
        )
    ):
        raise ValueError(
            f"plan {lane_label} duration estimates must contain "
            "four nonnegative totals"
        )
    if estimates != expected_estimates:
        raise ValueError(
            f"plan {lane_label} duration estimates do not match selected weights"
        )
    return shards


def _validated_shards(
    value: Any,
    expected_identities: set[str],
    *,
    lane_label: str,
) -> dict[str, list[str]]:
    if not isinstance(value, dict) or set(value) != {
        str(shard) for shard in SHARDS
    }:
        raise ValueError(f"plan {lane_label} must contain four exact shards")
    flattened: list[str] = []
    for shard in SHARDS:
        rows = value[str(shard)]
        if not isinstance(rows, list):
            raise ValueError(f"plan {lane_label} shard {shard} must be a list")
        for canonical in rows:
            Identity.parse(canonical)
        flattened.extend(rows)
    if sorted(flattened) != sorted(expected_identities):
        raise ValueError(f"plan {lane_label} shards do not exactly cover the lane")
    return value


def execution_shards(
    plan: Mapping[str, Any],
    lane: str,
) -> Mapping[str, list[str]]:
    if lane == "package-expansion":
        return package_expansion_execution(plan)["shards"]
    return plan["shards"][LANE_KEYS[lane]]


def execution_planning(
    plan: Mapping[str, Any],
    lane: str,
) -> Mapping[str, Any]:
    if lane == "package-expansion":
        return package_expansion_execution(plan)["planning"]
    return plan["shard_planning"][LANE_KEYS[lane]]


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
        "target_features",
        "change_owned",
        "package_expansion",
        "standing_targets",
        "standing_coverage_reuse",
        "manual_only_targets",
        "manual_gate_targets",
        "target_exclusions",
        "test_exclusions",
        "shard_planning",
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
    required_packages: set[str] = set()
    seen_required_rules: set[tuple[str, str]] = set()
    for disposition in plan["path_dispositions"]:
        if not isinstance(disposition, dict):
            raise ValueError("plan path_dispositions rows must be objects")
        rows = disposition.get("required_package_rules", [])
        if not isinstance(rows, list):
            raise ValueError("plan required_package_rules must be a list")
        path = disposition.get("path")
        if rows and not isinstance(path, str):
            raise ValueError(
                "plan required_package_rules require a disposition path"
            )
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "rule",
                "packages",
                "reason",
                "tracking_issue",
            }:
                raise ValueError(
                    "plan required_package_rules row has the wrong shape"
                )
            rule = row["rule"]
            row_packages = row["packages"]
            reason = row["reason"]
            tracking_issue = row["tracking_issue"]
            if not isinstance(rule, str) or not _valid_prefix(rule):
                raise ValueError(
                    "plan required_package_rules requires a valid rule prefix"
                )
            if (
                rule.endswith("/")
                and not path.startswith(rule)
                or not rule.endswith("/")
                and path != rule
            ):
                raise ValueError(
                    "plan required_package_rules rule does not match its path"
                )
            if (
                not isinstance(row_packages, list)
                or not row_packages
                or any(
                    not isinstance(package, str)
                    or not IDENTIFIER.fullmatch(package)
                    for package in row_packages
                )
                or len(row_packages) != len(set(row_packages))
            ):
                raise ValueError(
                    "plan required_package_rules packages must be unique "
                    "package names"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    "plan required_package_rules requires a reason"
                )
            if (
                not isinstance(tracking_issue, str)
                or not ISSUE.fullmatch(tracking_issue)
            ):
                raise ValueError(
                    "plan required_package_rules requires chelis#N tracking"
                )
            identity = (path, rule)
            if identity in seen_required_rules:
                raise ValueError(
                    "plan required_package_rules contains a duplicate rule"
                )
            seen_required_rules.add(identity)
            required_packages.update(row_packages)
    if not required_packages <= set(packages):
        raise ValueError("plan required packages must be selected packages")
    eligible = set(_identity_list(plan, "eligible_targets"))
    raw_target_features = plan.get("target_features")
    if not isinstance(raw_target_features, dict):
        raise ValueError("plan target_features must be an identity mapping")
    if set(raw_target_features) != eligible:
        raise ValueError(
            "plan target_features must exactly cover eligible targets"
        )
    for canonical, features in raw_target_features.items():
        Identity.parse(canonical)
        if (
            not isinstance(features, list)
            or any(
                not isinstance(feature, str)
                or not IDENTIFIER.fullmatch(feature)
                for feature in features
            )
            or features != sorted(set(features))
        ):
            raise ValueError(
                f"plan target_features for {canonical} must be sorted unique "
                "feature names"
            )
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
    required_targets = {
        identity
        for identity in eligible
        if Identity.parse(identity).package in required_packages
        and identity not in target_exclusions
    }
    if not required_targets <= change_owned:
        raise ValueError(
            "plan required package targets must be change-owned"
        )
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
    rows = plan.get("manual_gate_targets")
    if not isinstance(rows, list):
        raise ValueError("plan manual_gate_targets must be a list")
    manual_gates: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "identity",
            "manual_gates",
            "reason",
            "tracking_issue",
        }:
            raise ValueError("plan manual_gate_targets row has the wrong shape")
        Identity.parse(row["identity"])
        if row["identity"] in manual_gates:
            raise ValueError("plan manual_gate_targets contains duplicate identities")
        manual_gates.add(row["identity"])
        entries = row["manual_gates"]
        if (
            not isinstance(entries, list)
            or not entries
            or any(not isinstance(entry, str) or not entry for entry in entries)
            or len(set(entries)) != len(entries)
            or not isinstance(row["reason"], str)
            or not row["reason"].strip()
            or not isinstance(row["tracking_issue"], str)
            or not ISSUE.fullmatch(row["tracking_issue"])
        ):
            raise ValueError("plan manual_gate_targets row has malformed fields")
    if not manual_gates <= eligible:
        raise ValueError("plan manual-gate targets must be eligible")
    if manual_gates & (
        standing | target_exclusions | manual_only | test_exclusion_targets
    ):
        raise ValueError(
            "plan manual-gate targets conflict with standing, excluded, "
            "manual-only, or test-excluded targets"
        )

    expected_change_owned = change_owned - standing_reuse
    shard_planning = plan.get("shard_planning")
    if not isinstance(shard_planning, dict) or set(shard_planning) != {
        "change_owned",
        "package_expansion",
    }:
        raise ValueError("plan shard_planning must name both lanes")
    expected_change_owned_shards = _validated_duration_planning(
        shard_planning["change_owned"],
        expected_change_owned,
        algorithm=CHANGE_OWNED_SHARD_ALGORITHM,
        lane_label="change-owned",
    )
    if shard_planning["package_expansion"] != {
        "algorithm": PACKAGE_EXPANSION_COMPATIBILITY_ALGORITHM,
    }:
        raise ValueError(
            "plan package-expansion compatibility algorithm is unsupported"
        )
    expansion_execution = package_expansion_execution(plan)
    expected_expansion_shards = _validated_duration_planning(
        expansion_execution["planning"],
        expansion,
        algorithm=PACKAGE_EXPANSION_SHARD_ALGORITHM,
        lane_label="package-expansion",
    )

    shards = plan.get("shards")
    if not isinstance(shards, dict) or set(shards) != set(LANE_KEYS.values()):
        raise ValueError("plan shards must name both lanes")
    change_owned_shards = _validated_shards(
        shards["change_owned"],
        expected_change_owned,
        lane_label="change-owned",
    )
    if change_owned_shards != expected_change_owned_shards:
        raise ValueError(
            "plan change-owned shard assignment does not match "
            "change-owned duration planning"
        )
    compatibility_shards = _validated_shards(
        shards["package_expansion"],
        expansion,
        lane_label="package-expansion compatibility",
    )
    if compatibility_shards != shard_map(
        Identity.parse(canonical) for canonical in expansion
    ):
        raise ValueError(
            "plan package-expansion compatibility assignment does not match "
            "trusted v3 hashing"
        )
    expansion_shards = _validated_shards(
        expansion_execution["shards"],
        expansion,
        lane_label="package-expansion execution",
    )
    if expansion_shards != expected_expansion_shards:
        raise ValueError(
            "plan package-expansion shard assignment does not match "
            "package-expansion duration planning"
        )
    if not isinstance(plan.get("plan_digest"), str) or not DIGEST.fullmatch(
        plan["plan_digest"]
    ):
        raise ValueError("plan_digest must be a SHA-256 digest")


def verify_plan_duration_baseline(
    plan: Mapping[str, Any],
    duration_baseline: DurationBaseline,
) -> None:
    for lane_key, selected in (
        (
            "change_owned",
            sorted(
                set(plan["change_owned"])
                - set(plan["standing_coverage_reuse"])
            ),
        ),
        ("package_expansion", sorted(plan["package_expansion"])),
    ):
        lane_label = lane_key.replace("_", "-")
        planning = (
            package_expansion_execution(plan)["planning"]
            if lane_key == "package_expansion"
            else plan["shard_planning"][lane_key]
        )
        if planning["baseline_digest"] != duration_baseline.digest:
            raise ValueError(
                f"plan {lane_label} duration baseline digest does not match "
                "the checked-out baseline"
            )
        if (
            planning["default_milliseconds"]
            != duration_baseline.default_milliseconds
        ):
            raise ValueError(
                f"plan {lane_label} duration fallback does not match "
                "the checked-out baseline"
            )
        expected_weights = {
            canonical: duration_baseline.targets.get(
                Identity.parse(canonical),
                duration_baseline.default_milliseconds,
            )
            for canonical in selected
        }
        if planning["weights_milliseconds"] != expected_weights:
            raise ValueError(
                f"plan {lane_label} duration weights do not match "
                "the checked-out baseline"
            )


def verify_plan_digest(
    plan: Mapping[str, Any],
    *,
    duration_baseline: DurationBaseline | None = None,
) -> None:
    _validate_plan_shape(plan)
    expected = _digest_without(plan, "plan_digest")
    if plan.get("plan_digest") != expected:
        raise ValueError(
            f"plan digest mismatch: expected {expected}, got {plan.get('plan_digest')}"
        )
    if duration_baseline is not None:
        verify_plan_duration_baseline(plan, duration_baseline)


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
        "manual_gate_tests",
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
    for key in ("selected_tests", "executed_tests", "manual_gate_tests"):
        rows = receipt.get(key)
        if not isinstance(rows, list) or any(not isinstance(row, str) for row in rows):
            raise ValueError(f"receipt {key} must be a list")
        for row in rows:
            TestIdentity.parse(row)
        if len(rows) != len(set(rows)):
            raise ValueError(f"receipt {key} contains duplicates")
    # A manual-gate test is listed and never run, so it is neither selected
    # nor executed, and only the required lane lists one.
    gate_tests = receipt["manual_gate_tests"]
    gate_targets = {
        TestIdentity.parse(row).target_identity.canonical for row in gate_tests
    }
    if gate_tests and lane != "change-owned":
        raise ValueError("receipt manual_gate_tests belong only to change-owned")
    if (
        not gate_targets <= set(receipt["selected_targets"])
        or gate_targets & set(receipt["executed_targets"])
        or set(gate_tests)
        & (set(receipt["selected_tests"]) | set(receipt["executed_tests"]))
    ):
        raise ValueError(
            "receipt manual_gate_tests must be listed, unexecuted tests of "
            "selected targets"
        )
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
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError("receipt elapsed_seconds must be finite and nonnegative")
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


def duration_shard_map(
    weights: Mapping[Identity, int],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    for identity, milliseconds in weights.items():
        _positive_int(
            milliseconds,
            f"duration weight for {identity.canonical}",
        )
    result = {str(shard): [] for shard in SHARDS}
    estimates = {str(shard): 0 for shard in SHARDS}
    counts = {str(shard): 0 for shard in SHARDS}
    ordered = sorted(
        weights,
        key=lambda identity: (-weights[identity], identity.canonical),
    )
    for identity in ordered:
        shard_key = min(
            estimates,
            key=lambda key: (estimates[key], counts[key], int(key)),
        )
        result[shard_key].append(identity.canonical)
        estimates[shard_key] += weights[identity]
        counts[shard_key] += 1
    for rows in result.values():
        rows.sort()
    return result, estimates


def duration_shard_plan(
    identities: Iterable[Identity],
    baseline: DurationBaseline,
    *,
    algorithm: str,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    selected = sorted(set(identities))
    weights = {
        identity: baseline.targets.get(
            identity,
            baseline.default_milliseconds,
        )
        for identity in selected
    }
    shards, estimates = duration_shard_map(weights)
    planning = {
        "algorithm": algorithm,
        "baseline_digest": baseline.digest,
        "default_milliseconds": baseline.default_milliseconds,
        "weights_milliseconds": {
            identity.canonical: weights[identity] for identity in selected
        },
        "estimated_milliseconds": estimates,
    }
    return shards, planning


def change_owned_shard_plan(
    identities: Iterable[Identity],
    baseline: DurationBaseline,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    return duration_shard_plan(
        identities,
        baseline,
        algorithm=CHANGE_OWNED_SHARD_ALGORITHM,
    )


def package_expansion_shard_plan(
    identities: Iterable[Identity],
    baseline: DurationBaseline,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    return duration_shard_plan(
        identities,
        baseline,
        algorithm=PACKAGE_EXPANSION_SHARD_ALGORITHM,
    )


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
    duration_baseline_path: Path = DURATION_BASELINE_PATH,
    targeted_packages: Sequence[str] = (),
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
        if not pr_head or not before or not targeted_packages:
            raise ValueError(
                "targeted_rebase planning requires --pr-head, --before, "
                "and --targeted-packages"
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
    if mode != "targeted_rebase" and targeted_packages:
        raise ValueError(
            "--targeted-packages is valid only for targeted_rebase planning"
        )

    config = read_config(config_path)
    duration_baseline = load_duration_baseline(duration_baseline_path)
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
        targeted_packages=targeted_packages,
        duration_baseline=duration_baseline,
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
    required_features: Sequence[str] = (),
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
    ]
    features = sorted(set(required_features))
    if features:
        command.extend(["--features", ",".join(features)])
    command.extend(
        [
            "--locked",
            "--profile",
            "ci-full",
            "--ignore-default-filter",
        ]
    )
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
    """Batch ordinary expansion targets into bounded package-scoped chunks."""
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
    planning = execution_planning(plan, lane)
    weights = planning["weights_milliseconds"]
    default_milliseconds = planning["default_milliseconds"]
    groups: list[list[Identity]] = []
    group_estimates: list[int] = []
    ordinary_by_package: dict[str, int] = {}
    for identity in identities:
        special = (
            identity.canonical in manual_only
            or identity in excluded_targets
        )
        if special:
            groups.append([identity])
            group_estimates.append(
                weights.get(identity.canonical, default_milliseconds)
            )
            continue
        estimate = weights.get(identity.canonical, default_milliseconds)
        index = ordinary_by_package.get(identity.package)
        if (
            index is None
            or len(groups[index]) >= PACKAGE_EXPANSION_GROUP_TARGET_LIMIT
            or (
                groups[index]
                and group_estimates[index] + estimate
                > PACKAGE_EXPANSION_GROUP_ESTIMATED_MILLISECONDS
            )
        ):
            ordinary_by_package[identity.package] = len(groups)
            groups.append([identity])
            group_estimates.append(estimate)
        else:
            groups[index].append(identity)
            group_estimates[index] += estimate
    return [tuple(group) for group in groups]


def target_group_command(
    identities: Sequence[Identity],
    exclusions: Mapping[TestIdentity, Any],
    *,
    list_only: bool,
    manual_only: bool = False,
    required_features: Sequence[str] = (),
) -> list[str]:
    if not identities:
        raise ValueError("target command group must not be empty")
    if len(identities) == 1:
        return target_command(
            identities[0],
            exclusions,
            list_only=list_only,
            manual_only=manual_only,
            required_features=required_features,
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
    features = sorted(set(required_features))
    if features:
        command.extend(["--features", ",".join(features)])
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
    allow_empty: bool = False,
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
                allow_empty=allow_empty,
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
    allow_empty: bool = False,
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
            f"ignored-only target {expected} has active tests: {sorted(active)}"
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
    if not selected and not allow_empty:
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
    duration_baseline: DurationBaseline | None = None,
) -> dict[str, Any]:
    if duration_baseline is None:
        duration_baseline = load_duration_baseline()
    verify_plan_digest(plan, duration_baseline=duration_baseline)
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
    selected = list(execution_shards(plan, lane)[str(shard)])
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
    not_applicable_targets: list[str] = []
    manual_gate_tests: list[str] = []
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
    target_features = {
        Identity.parse(canonical): tuple(features)
        for canonical, features in plan["target_features"].items()
    }
    manual_only_targets = {
        row["identity"] for row in plan["manual_only_targets"]
    }
    # The required lane lists a manual-gate target with its ignored tests and
    # stops there: the gate's prerequisites exist only where its documented
    # command is run by hand.
    manual_gate_targets = (
        {row["identity"] for row in plan["manual_gate_targets"]}
        if lane == "change-owned"
        else set()
    )
    cargo_target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not cargo_target.is_absolute():
        cargo_target = repo / cargo_target
    produced_junit = cargo_target / "nextest/ci-full/junit.xml"
    test_env = os.environ.copy()

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
            runtime_archive = cargo_target / "debug/libchelis_runtime.a"
            if not runtime_archive.is_file():
                build_succeeded = False
                failures.append(
                    "workspace product build did not produce the exact-head "
                    f"runtime archive: {runtime_archive}"
                )
                product_timing["success"] = False
            else:
                test_env["CHELIS_RUNTIME_LIB"] = str(runtime_archive)

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
            manual_gate = (
                len(identities) == 1
                and identities[0].canonical in manual_gate_targets
            )
            ignored_only = manual_only or manual_gate
            required_features = sorted(
                {
                    feature
                    for identity in identities
                    for feature in target_features[identity]
                }
            )
            list_command = target_group_command(
                identities,
                exclusions,
                list_only=True,
                manual_only=ignored_only,
                required_features=required_features,
            )
            list_started_at = _utc_timestamp()
            started = time.monotonic()
            try:
                listed = run(
                    list_command,
                    cwd=repo,
                    check=True,
                    env=test_env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                listed_tests = _listing_tests_for_group(
                    json.loads(listed.stdout),
                    identities,
                    exclusions,
                    manual_only=ignored_only,
                    allow_empty=(
                        lane == "package-expansion" and not ignored_only
                    ),
                )
                group_not_applicable = [
                    canonical
                    for canonical in canonicals
                    if not any(
                        test.startswith(f"{canonical}::")
                        for test in listed_tests
                    )
                ]
                not_applicable_targets.extend(group_not_applicable)
                executed_targets.extend(group_not_applicable)
                (manual_gate_tests if manual_gate else selected_tests).extend(
                    listed_tests
                )
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
                        "not_applicable_targets": group_not_applicable,
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
            if manual_gate or not listed_tests:
                for canonical in canonicals:
                    target_timings[canonical] = {
                        "command_group": canonicals,
                        "list_started_at": list_started_at,
                        "list_finished_at": list_finished_at,
                        "list_seconds": list_seconds,
                        "run_started_at": None,
                        "run_finished_at": None,
                        "run_seconds": 0.0,
                    }
                continue

            run_command = target_group_command(
                identities,
                exclusions,
                list_only=False,
                manual_only=manual_only,
                required_features=required_features,
            )
            produced_junit.unlink(missing_ok=True)
            run_started_at = _utc_timestamp()
            started = time.monotonic()
            try:
                completed = run(
                    run_command,
                    cwd=repo,
                    check=True,
                    env=test_env,
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
                            if canonical not in executed_targets:
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
                "not_applicable_targets": sorted(not_applicable_targets),
                "manual_gate_tests": sorted(manual_gate_tests),
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
        "manual_gate_tests": sorted(manual_gate_tests),
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
    duration_baseline: DurationBaseline | None = None,
) -> dict[str, Any] | None:
    """Write an empty receipt or tell the hosted worker to prepare execution."""
    if duration_baseline is None:
        duration_baseline = load_duration_baseline()
    verify_plan_digest(plan, duration_baseline=duration_baseline)
    if lane not in LANE_KEYS:
        raise ValueError(f"unsupported lane: {lane}")
    if shard not in SHARDS:
        raise ValueError(f"shard must be one of {SHARDS}")
    selected = execution_shards(plan, lane)[str(shard)]
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
        duration_baseline=duration_baseline,
    )


@dataclass(frozen=True)
class FailureBaseline:
    """Recorded default-branch test outcomes the expansion differences against."""

    workflow: str
    run_id: str
    run_url: str
    head_sha: str
    created_at: str
    observed: frozenset[str]
    failed: frozenset[str]

    def provenance(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "run_id": self.run_id,
            "run_url": self.run_url,
            "head_sha": self.head_sha,
            "created_at": self.created_at,
            "observed_cases": len(self.observed),
            "failing_cases": len(self.failed),
        }


def junit_case_outcomes(path: Path) -> tuple[set[str], set[str]]:
    """Return the executed and failing ``package::target::test`` identities."""
    observed: set[str] = set()
    failed: set[str] = set()
    for case in ET.parse(path).iter("testcase"):
        classname = case.get("classname")
        name = case.get("name")
        if not classname or not name:
            raise ValueError(f"JUnit testcase has no identity: {path}")
        if case.find("skipped") is not None:
            continue
        identity = f"{classname}::{name}"
        observed.add(identity)
        if case.find("failure") is not None or case.find("error") is not None:
            failed.add(identity)
    return observed, failed


def load_failure_baseline(path: Path) -> FailureBaseline:
    """Read a recorded default-branch baseline, rejecting an unusable one."""
    manifest = load_json(path)
    expected_keys = {
        "version",
        "workflow",
        "run_id",
        "run_url",
        "head_sha",
        "created_at",
        "documents",
    }
    if set(manifest) != expected_keys:
        raise ValueError(
            f"failure baseline manifest keys mismatch: "
            f"missing={sorted(expected_keys - set(manifest))}, "
            f"extra={sorted(set(manifest) - expected_keys)}"
        )
    if manifest["version"] != FAILURE_BASELINE_VERSION:
        raise ValueError(
            f"failure baseline manifest version must be {FAILURE_BASELINE_VERSION}"
        )
    for key in ("workflow", "run_id", "run_url", "created_at"):
        if not isinstance(manifest[key], str) or not manifest[key]:
            raise ValueError(f"failure baseline manifest {key} must be a string")
    head_sha = manifest["head_sha"]
    if not isinstance(head_sha, str) or not SHA.fullmatch(head_sha):
        raise ValueError("failure baseline manifest head_sha must be a commit")
    documents = manifest["documents"]
    if (
        not isinstance(documents, list)
        or not documents
        or any(not isinstance(row, str) or not row for row in documents)
    ):
        raise ValueError("failure baseline manifest documents must be named")
    observed: set[str] = set()
    failed: set[str] = set()
    for relative in documents:
        parts = PurePosixPath(relative).parts
        if PurePosixPath(relative).is_absolute() or ".." in parts:
            raise ValueError(f"failure baseline document escapes its root: {relative}")
        document = path.parent / PurePosixPath(relative)
        if not document.is_file():
            raise ValueError(f"missing failure baseline document: {document}")
        document_observed, document_failed = junit_case_outcomes(document)
        observed |= document_observed
        failed |= document_failed
    if not observed:
        raise ValueError("failure baseline recorded no executed test cases")
    return FailureBaseline(
        workflow=manifest["workflow"],
        run_id=manifest["run_id"],
        run_url=manifest["run_url"],
        head_sha=head_sha,
        created_at=manifest["created_at"],
        observed=frozenset(observed),
        failed=frozenset(failed),
    )


def _ancestor_distance(repo: Path, ancestor: str, descendant: str) -> int | None:
    """Commits between two reachable commits, or None when unrelated."""
    for commit in (ancestor, descendant):
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if resolved.returncode != 0:
            raise ValueError(f"commit is not present in this clone: {commit}")
    if ancestor == descendant:
        return 0
    contained = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if contained.returncode == 1:
        return None
    if contained.returncode != 0:
        raise ValueError(
            f"could not compare {ancestor} against {descendant}: "
            f"{contained.stderr.strip()}"
        )
    counted = subprocess.run(
        ["git", "rev-list", "--count", f"{ancestor}..{descendant}"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return int(counted.stdout.strip())


def baseline_provenance(
    plan: Mapping[str, Any],
    baseline: FailureBaseline,
    repo: Path,
    *,
    distance: Callable[[Path, str, str], int | None] | None = None,
) -> dict[str, Any]:
    """Bind a baseline to the candidate's own history, or refuse it.

    A baseline the candidate's merge base does not contain describes a
    different tree, and differencing against it would report a failure the
    candidate introduced as one it inherited. That is refused outright.

    Distance behind the base is annotated rather than refused, but it is not
    harmless in one direction only. A test green at the baseline and broken by
    the candidate is correctly introduced, and a test broken on the default
    branch after the baseline is over-reported as introduced, which is the safe
    error. The unsafe case is a test that was failing at the baseline, was
    fixed on the default branch since, and is broken again by the candidate:
    its identity is still in ``baseline.failed``, so it reports as inherited.
    Twelve identities moved that way between two nightlies five days apart, so
    the window is real rather than theoretical. The producer therefore selects
    the newest baseline the candidate's base contains, which makes the window
    the commits between that baseline and the base and no larger, and the
    report states the distance so a reader can size what is left.
    """
    base_sha = plan["base_sha"]
    measure = distance or _ancestor_distance
    behind = measure(repo, baseline.head_sha, base_sha)
    if behind is None:
        raise ValueError(
            f"failure baseline commit {baseline.head_sha} is not an ancestor of "
            f"the candidate base {base_sha}; differencing against it could "
            f"report an introduced failure as inherited"
        )
    provenance = baseline.provenance()
    provenance["candidate_base_sha"] = base_sha
    provenance["commits_behind_candidate_base"] = behind
    return provenance


def classify_expansion_failures(
    documents: Sequence[tuple[Mapping[str, Any], Path | None]],
    baseline: FailureBaseline,
) -> dict[str, Any]:
    """Split observed failures into introduced and inherited, and count unrun.

    ``introduced`` carries the baseline's own verdict per row. A failure the
    baseline never executed is introduced rather than inherited, because an
    absent verdict is not evidence of prior breakage; the row says ``absent``
    so a reader is not told the baseline confirmed anything.
    """
    observed_failures: set[str] = set()
    unrun_targets: set[str] = set()
    unrun_tests: set[str] = set()
    for receipt, junit in documents:
        selected = receipt.get("selected_targets")
        executed = receipt.get("executed_targets")
        if isinstance(selected, list) and isinstance(executed, list):
            unrun_targets |= set(selected) - set(executed)
        selected_tests = receipt.get("selected_tests")
        executed_tests = receipt.get("executed_tests")
        if isinstance(selected_tests, list) and isinstance(executed_tests, list):
            unrun_tests |= set(selected_tests) - set(executed_tests)
        if junit is not None and junit.is_file():
            _, failed = junit_case_outcomes(junit)
            observed_failures |= failed
    introduced = sorted(observed_failures - baseline.failed)
    return {
        "introduced": [
            {
                "test": test,
                "baseline": "passed" if test in baseline.observed else "absent",
            }
            for test in introduced
        ],
        "inherited": sorted(observed_failures & baseline.failed),
        "unrun_targets": sorted(unrun_targets),
        "unrun_tests": sorted(unrun_tests),
    }


def fork_point(repo: Path, base: str) -> str:
    """The merge base of `base` and HEAD, the commit the branch diffs from.

    The branch's own change is the diff from here, not from `base`'s tip. A
    branch behind `base` would otherwise be charged with every path `base`
    changed since the fork (chelis#2480), which CI never charges it with: the
    synthetic merge it plans already contains that work on its first parent.
    A missing ref or unrelated histories are refused rather than replaced by
    some other diff.
    """
    found = subprocess.run(
        ["git", "merge-base", base, "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if found.returncode == 1 and not found.stderr.strip():
        raise ValueError(
            f"{base} and HEAD share no history, so there is no merge base "
            f"to diff the branch from"
        )
    if found.returncode != 0:
        raise ValueError(
            f"could not find the merge base of {base} and HEAD "
            f"({found.stderr.strip()}); fetch {base} and rerun"
        )
    return found.stdout.strip()


def working_tree_changed_paths(repo: Path = ROOT, base: str = "origin/main") -> list[str]:
    """The changed set the local classification reads, derived live.

    Derived here rather than handed in, so the set is taken when the check
    runs. `--fast` regenerates before it checks, and a set captured before
    those writers cannot see a file they created.

    Four deliberate differences from a naive `git diff --name-only` plus
    `git status`:

    * the diff starts at the merge base with `base`, not at its tip, so a
      branch behind `base` carries only its own paths; see `fork_point`.
    * `--find-renames` and both sides of a rename, matching `diff_at`, which
      is what CI classifies. A bare `--name-only` collapses a rename to its
      destination, so moving an unrouted file into a package root reads clean
      locally while the planner refuses the source path.
    * committed and unstaged work, because both reach the candidate.
    * **not** untracked entries. CI never sees them, nothing can route a
      scratch file or a directory, and including them produces a local
      failure with no hosted counterpart whose printed remedy is to add a
      junk row to a reviewed manifest.
    """
    paths: set[str] = set()
    for revisions in ([fork_point(repo, base), "HEAD"], ["HEAD"]):
        # `parse_name_status_z` rejects an empty diff, because a candidate
        # always has one. A working tree legitimately may not, so emptiness is
        # handled here rather than by loosening the parser CI shares.
        raw = git_output(
            repo, ["diff", "--name-status", "-z", "--find-renames", *revisions]
        )
        if not raw:
            continue
        for record in parse_name_status_z(raw):
            paths.add(record.path)
            if record.old_path:
                paths.add(record.old_path)
    return sorted(paths)


def classify_changed_paths(
    paths: Sequence[str],
    *,
    repo: Path = ROOT,
    config: Config | None = None,
    packages: Sequence[PackageInfo] | None = None,
    base: str | None = None,
) -> list[tuple[str, str]]:
    """Every path the planner would refuse, as (path, disposition) pairs.

    This is the planner's own `static_path_classification` and nothing else.
    The `plan` subcommand cannot answer the question locally: in
    `pull_request` mode it requires a two-parent synthetic merge and exits
    before classifying anything, so a new tracked file could pass a clean
    `--fast` and a green contract suite and still fail `Plan Changed
    Integration Tests` on the first unclassified path (chelis#2250).

    Every offending path is returned rather than the first. CI stops at the
    first, which is how chelis#2248's repair could have left a second one
    behind; locally there is no reason to make a developer find them one push
    at a time.

    With `base`, an unrouted path that is gone from the working tree is a
    retirement when the merge base with `base` tracked it: that is the tree
    `working_tree_changed_paths` diffs from, so a file the branch deleted and
    `base` has since deleted too is still recognized.
    """
    if not paths:
        # Nothing to classify, so do not pay for `cargo metadata`.
        return []
    if config is None:
        config = read_config(repo / ".config/ci-test-targets.toml")
    if packages is None:
        try:
            packages = package_infos(_metadata_in(repo))
        except FileNotFoundError as error:
            # This is the gate's first check, so it is the one that reports a
            # missing toolchain. A bare errno there reads as a defect in the
            # check rather than an absent cargo.
            raise ValueError(
                f"classifying changed paths needs cargo on PATH to read the "
                f"workspace package roots: {error}"
            ) from error
    refused = []
    base_paths = (
        tracked_paths_at(repo, fork_point(repo, base)) if base is not None else set()
    )
    for path in sorted(set(paths)):
        disposition, _, _ = static_path_classification(path, packages, config)
        if (
            disposition == "unclassified"
            and path in base_paths
            and not (repo / path).exists()
            and not (repo / path).is_symlink()
        ):
            continue
        if disposition in {"unclassified", "ambiguous_rule", "ambiguous_package"}:
            refused.append((path, disposition))
    return refused


def _report_findings(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    lane: str,
    duration_baseline: DurationBaseline | None = None,
    *,
    classifies_coverage: bool = False,
) -> list[str]:
    """Structural findings shared by both lanes.

    ``classifies_coverage`` belongs to a caller that reports incomplete
    execution as its own count. It suppresses exactly the rows that count would
    otherwise duplicate, and nothing that describes a defect in the receipts
    themselves.
    """
    if duration_baseline is None:
        duration_baseline = load_duration_baseline()
    verify_plan_digest(plan, duration_baseline=duration_baseline)
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
    all_manual_gate_tests: list[str] = []
    for shard, receipt in sorted(by_shard.items()):
        expected = execution_shards(plan, lane)[str(shard)]
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
        all_manual_gate_tests.extend(receipt["manual_gate_tests"])
        # The informational lane reports an unsuccessful shard through the
        # introduced/inherited/unrun counts, which are derived from the shard's
        # own JUnit and target lists rather than from its prose.
        # `summarize_package_expansion` still fails loudly on a shard whose
        # recorded failures none of those three counts can account for.
        if not classifies_coverage and receipt.get("success") is not True:
            findings.append(
                f"shard {shard} did not succeed: {receipt.get('failures', [])}"
            )

    expected_targets = sorted(
        identity
        for rows in execution_shards(plan, lane).values()
        for identity in rows
    )
    # A manual-gate target is listed, never executed: it leaves the executed
    # set, and its listed ignored tests must account for exactly those rows.
    manual_gates = (
        {row["identity"] for row in plan["manual_gate_targets"]}
        & set(expected_targets)
        if lane == "change-owned"
        else set()
    )
    expected_executed = sorted(set(expected_targets) - manual_gates)
    if sorted(all_selected_targets) != expected_targets:
        findings.append(
            "selected target coverage mismatch: "
            f"expected={expected_targets}, got={sorted(all_selected_targets)}"
        )
    if not classifies_coverage and sorted(all_executed_targets) != expected_executed:
        findings.append(
            "executed target coverage mismatch: "
            f"expected={expected_executed}, got={sorted(all_executed_targets)}"
        )
    listed_gates = {
        TestIdentity.parse(test).target_identity.canonical
        for test in all_manual_gate_tests
    }
    if listed_gates != manual_gates:
        findings.append(
            "manual-gate listing mismatch: "
            f"expected={sorted(manual_gates)}, listed={sorted(listed_gates)}"
        )
    if len(all_manual_gate_tests) != len(set(all_manual_gate_tests)):
        findings.append("duplicate manual-gate test listing")
    if len(all_executed_targets) != len(set(all_executed_targets)):
        findings.append("duplicate target execution")
    if len(all_executed_tests) != len(set(all_executed_tests)):
        findings.append("duplicate test execution")
    if not classifies_coverage and sorted(all_selected_tests) != sorted(
        all_executed_tests
    ):
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
    *,
    duration_baseline: DurationBaseline | None = None,
) -> dict[str, Any]:
    findings = _report_findings(
        plan,
        receipts,
        "change-owned",
        duration_baseline,
    )
    reused_targets = validate_standing_coverage(plan, standing_coverage)
    if findings:
        raise ValueError("; ".join(findings))
    listed = [test for receipt in receipts for test in receipt["manual_gate_tests"]]
    manual_gates = [
        {
            **row,
            "status": MANUAL_GATE_STATUS,
            "listed_tests": sorted(
                test for test in listed if test.startswith(f"{row['identity']}::")
            ),
        }
        for row in plan["manual_gate_targets"]
        if row["identity"] in plan["change_owned"]
    ]
    return {
        "version": 1,
        "lane": "change-owned",
        "required": True,
        "success": True,
        "observed_success": True,
        "plan_digest": plan["plan_digest"],
        "covered_targets": sorted(
            set(plan["change_owned"]) - {row["identity"] for row in manual_gates}
        ),
        "manual_gate_targets": manual_gates,
        "standing_reused_targets": reused_targets,
        "shard_durations": shard_durations(
            plan,
            receipts,
            lane="change-owned",
        ),
        "failures": [],
    }


def shard_durations(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    lane: str,
) -> list[dict[str, int | None]]:
    estimates = execution_planning(plan, lane)["estimated_milliseconds"]
    actual: dict[int, int] = {}
    for receipt in receipts:
        if (
            receipt.get("lane") != lane
            or receipt.get("plan_digest") != plan.get("plan_digest")
        ):
            continue
        shard = receipt.get("shard")
        elapsed = receipt.get("elapsed_seconds")
        if (
            type(shard) is int
            and shard in SHARDS
            and not isinstance(elapsed, bool)
            and isinstance(elapsed, (int, float))
            and math.isfinite(elapsed)
            and elapsed >= 0
        ):
            actual[shard] = math.ceil(float(elapsed) * 1000)
    return [
        {
            "shard": shard,
            "estimated_milliseconds": estimates[str(shard)],
            "actual_milliseconds": actual.get(shard),
        }
        for shard in SHARDS
    ]


def _unclassified_shard_findings(
    documents: Sequence[tuple[Mapping[str, Any], Path | None]],
    classification: Mapping[str, Any],
) -> list[str]:
    """Refuse to drop a shard defect the three counts cannot account for.

    Every failure `execute_shard` records implies either a failing test in the
    shard's JUnit or a selected target with no execution evidence. That
    enumeration is an argument about the executor, not a guarantee, so a shard
    that reports failure while contributing to neither count is reported rather
    than assumed benign.
    """
    accounted = {
        row["test"] for row in classification["introduced"]
    } | set(classification["inherited"])
    unrun = set(classification["unrun_targets"])
    findings: list[str] = []
    for receipt, _ in documents:
        if receipt.get("success") is not True:
            shard = receipt.get("shard")
            selected = receipt.get("selected_targets")
            selected = selected if isinstance(selected, list) else []
            prefixes = tuple(f"{canonical}::" for canonical in selected)
            explained = unrun & set(selected) or any(
                test.startswith(prefixes) for test in accounted
            )
            if not explained:
                findings.append(
                    f"shard {shard} recorded failures that no introduced, "
                    f"inherited or unrun row accounts for: "
                    f"{receipt.get('failures', [])}"
                )
    return findings


def summarize_package_expansion(
    plan: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    duration_baseline: DurationBaseline | None = None,
    junit_documents: Mapping[int, Path] | None = None,
    failure_baseline: FailureBaseline | None = None,
    baseline_unavailable: str | None = None,
    repo: Path = ROOT,
    ancestor_distance: Callable[[Path, str, str], int | None] | None = None,
) -> dict[str, Any]:
    findings = _report_findings(
        plan,
        receipts,
        "package-expansion",
        duration_baseline,
        classifies_coverage=True,
    )
    junit_documents = junit_documents or {}
    documents: list[tuple[Mapping[str, Any], Path | None]] = [
        (receipt, junit_documents.get(receipt.get("shard")))
        for receipt in receipts
    ]
    classification: dict[str, Any] | None = None
    if failure_baseline is None:
        reason = baseline_unavailable or "no failure baseline was supplied"
        findings.append(
            f"no usable default-branch failure baseline ({reason}), so no "
            f"observed failure can be reported as introduced or inherited"
        )
    else:
        try:
            provenance = baseline_provenance(
                plan,
                failure_baseline,
                repo,
                distance=ancestor_distance,
            )
        except ValueError as error:
            findings.append(
                f"no usable default-branch failure baseline ({error}), so no "
                f"observed failure can be reported as introduced or inherited"
            )
        else:
            classification = classify_expansion_failures(
                documents,
                failure_baseline,
            )
            classification["baseline"] = provenance
            findings.extend(
                _unclassified_shard_findings(documents, classification)
            )
    introduced = classification["introduced"] if classification else []
    observed_success = not findings and not introduced
    return {
        "version": EXPANSION_REPORT_VERSION,
        "lane": "package-expansion",
        "required": False,
        "success": observed_success,
        "observed_success": observed_success,
        "plan_digest": plan["plan_digest"],
        "covered_targets": sorted(plan["package_expansion"]),
        "standing_reused_targets": [],
        "shard_durations": shard_durations(
            plan,
            receipts,
            lane="package-expansion",
        ),
        "failure_classification": classification,
        "failures": findings,
    }


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _load_receipt_at(path: Path) -> dict[str, Any]:
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
    return receipt


def load_receipt_documents(root: Path) -> list[tuple[dict[str, Any], Path]]:
    """Each verified receipt beside the JUnit its own sidecar digest binds."""
    documents = []
    for path in sorted(root.rglob("receipt.json")):
        receipt = _load_receipt_at(path)
        documents.append((receipt, path.parent / receipt["junit_file"]))
    return documents


def load_receipts(root: Path) -> list[dict[str, Any]]:
    return [receipt for receipt, _ in load_receipt_documents(root)]


def _verify_duration_sample_plan(plan: Mapping[str, Any]) -> None:
    version = plan.get("version")
    target_dispositions = plan.get("target_dispositions")
    has_package_execution = isinstance(target_dispositions, list) and any(
        isinstance(row, dict)
        and row.get("kind") == PACKAGE_EXPANSION_EXECUTION_KIND
        for row in target_dispositions
    )
    if version == PLAN_VERSION and has_package_execution:
        verify_plan_digest(plan)
        return
    if version not in {2, 3}:
        raise ValueError(f"unsupported duration sample plan version: {version!r}")
    candidate_sha = plan.get("candidate_sha")
    if not isinstance(candidate_sha, str) or not SHA.fullmatch(candidate_sha):
        raise ValueError("duration sample plan candidate_sha is malformed")
    digest = plan.get("plan_digest")
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise ValueError("duration sample plan digest is malformed")
    if digest != _digest_without(plan, "plan_digest"):
        raise ValueError("duration sample plan digest mismatch")
    change_owned = set(_identity_list(plan, "change_owned"))
    standing_reuse = set(_identity_list(plan, "standing_coverage_reuse"))
    expected = change_owned - standing_reuse
    shards = plan.get("shards")
    if not isinstance(shards, dict):
        raise ValueError("duration sample plan shards are malformed")
    lane_shards = shards.get("change_owned")
    if not isinstance(lane_shards, dict) or set(lane_shards) != {
        str(shard) for shard in SHARDS
    }:
        raise ValueError("duration sample plan requires four change-owned shards")
    flattened: list[str] = []
    for shard in SHARDS:
        rows = lane_shards[str(shard)]
        if not isinstance(rows, list):
            raise ValueError("duration sample shard target list is malformed")
        for canonical in rows:
            Identity.parse(canonical)
        flattened.extend(rows)
    if len(flattened) != len(set(flattened)):
        raise ValueError("duration sample plan contains duplicate shard targets")
    if set(flattened) != expected:
        raise ValueError(
            "duration sample plan shards do not exactly cover change-owned execution"
        )


def _duration_seconds(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative number")
    return float(value)


def build_duration_baseline(
    samples: Sequence[tuple[Path, Path]],
) -> dict[str, Any]:
    if not samples:
        raise ValueError("duration baseline generation requires at least one sample")
    sources: list[dict[str, str]] = []
    observations: dict[Identity, list[int]] = {}
    seen_sources: set[tuple[str, str]] = set()

    for plan_path, receipts_root in samples:
        plan = _strict_json_object(plan_path)
        _verify_duration_sample_plan(plan)
        source_key = (plan["candidate_sha"], plan["plan_digest"])
        if source_key in seen_sources:
            raise ValueError(
                "duplicate duration sample source: "
                f"{source_key[0]} {source_key[1]}"
            )
        seen_sources.add(source_key)
        sources.append(
            {
                "candidate_sha": source_key[0],
                "plan_digest": source_key[1],
            }
        )

        receipt_paths = sorted(receipts_root.rglob("receipt.json"))
        if len(receipt_paths) != len(SHARDS):
            raise ValueError(
                "duration sample requires exactly four change-owned receipts"
            )
        by_shard: dict[int, tuple[Path, dict[str, Any]]] = {}
        for receipt_path in receipt_paths:
            receipt = _load_receipt_at(receipt_path)
            if receipt["lane"] != "change-owned":
                raise ValueError("duration sample contains a non-change-owned receipt")
            if receipt["plan_digest"] != plan["plan_digest"]:
                raise ValueError("duration sample receipt plan digest mismatch")
            shard = receipt["shard"]
            if shard in by_shard:
                raise ValueError(f"duplicate duration sample shard: {shard}")
            by_shard[shard] = (receipt_path, receipt)
        if set(by_shard) != set(SHARDS):
            raise ValueError("duration sample is missing a change-owned shard")

        for shard in SHARDS:
            receipt_path, receipt = by_shard[shard]
            expected = plan["shards"]["change_owned"][str(shard)]
            if receipt["selected_targets"] != expected:
                raise ValueError(
                    f"duration sample shard {shard} selected targets mismatch"
                )
            timings = _strict_json_object(receipt_path.parent / "timings.json")
            if set(timings) != {
                "started_at",
                "finished_at",
                "elapsed_seconds",
                "workspace_products",
                "targets",
            }:
                raise ValueError(
                    f"duration sample shard {shard} timings have the wrong shape"
                )
            _duration_seconds(
                timings["elapsed_seconds"],
                f"duration sample shard {shard} elapsed_seconds",
            )
            raw_targets = timings["targets"]
            if not isinstance(raw_targets, dict):
                raise ValueError("duration sample targets must be an object")
            unexpected = sorted(set(raw_targets) - set(receipt["selected_targets"]))
            if unexpected:
                raise ValueError(
                    "duration sample timings contain unselected targets: "
                    f"{unexpected}"
                )
            executed = set(receipt["executed_targets"])
            for canonical, row in raw_targets.items():
                identity = Identity.parse(canonical)
                if not isinstance(row, dict) or set(row) != {
                    "command_group",
                    "list_started_at",
                    "list_finished_at",
                    "list_seconds",
                    "run_started_at",
                    "run_finished_at",
                    "run_seconds",
                }:
                    raise ValueError(
                        f"duration sample timing has the wrong shape: {canonical}"
                    )
                if row["command_group"] != [canonical]:
                    raise ValueError(
                        "change-owned duration samples require singleton commands: "
                        f"{canonical}"
                    )
                list_seconds = _duration_seconds(
                    row["list_seconds"],
                    f"duration sample {canonical} list_seconds",
                )
                run_seconds = _duration_seconds(
                    row["run_seconds"],
                    f"duration sample {canonical} run_seconds",
                )
                if canonical not in executed:
                    continue
                milliseconds = max(
                    1,
                    math.ceil((list_seconds + run_seconds) * 1000),
                )
                observations.setdefault(identity, []).append(milliseconds)

    if not observations:
        raise ValueError("duration samples contain no completed target observations")
    return {
        "version": DURATION_BASELINE_VERSION,
        "default_milliseconds": DEFAULT_DURATION_MILLISECONDS,
        "sources": sorted(
            sources,
            key=lambda row: (row["candidate_sha"], row["plan_digest"]),
        ),
        "targets": {
            identity.canonical: {
                "milliseconds": max(values),
                "samples": len(values),
            }
            for identity, values in sorted(observations.items())
        },
    }


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
    if report["lane"] == "change-owned":
        manual_gates = report.get("manual_gate_targets", [])
        lines.append(f"- Manual gates, not executed in PR CI: {len(manual_gates)}")
        for row in manual_gates:
            entries = ", ".join(f"`{entry}`" for entry in row["manual_gates"])
            lines.append(
                f"  - `{row['identity']}`: {row['status']}. Ignored tests "
                f"listed: {len(row['listed_tests'])}, run: 0. Run by hand: "
                f"{MANUAL_GATES_PATH} {entries} ({row['tracking_issue']})"
            )
    lines.extend(_classification_counts(report))
    for row in report.get("shard_durations", []):
        weight = row["estimated_milliseconds"] / 1000
        actual = row["actual_milliseconds"]
        actual_text = "unavailable" if actual is None else f"{actual / 1000:.3f}s"
        lines.append(
            f"- Shard {row['shard']}: took {actual_text} "
            f"(balancing weight {weight:.3f}s)"
        )
    for finding in report["failures"]:
        lines.append(f"  - {finding}")
    lines.extend(_classification_lines(report))
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def _classification_counts(report: Mapping[str, Any]) -> list[str]:
    """The three disjoint counts, beside the header the reader sees first."""
    if report.get("lane") != "package-expansion":
        return []
    classification = report.get("failure_classification")
    if classification is None:
        return ["- Failure classification: unavailable, see below"]
    return [
        f"- Introduced failures: {len(classification['introduced'])}",
        f"- Inherited failures: {len(classification['inherited'])}",
        f"- Unrun targets: {len(classification['unrun_targets'])}",
    ]


def _classification_lines(report: Mapping[str, Any]) -> list[str]:
    """Render the three disjoint counts, or say why there are none."""
    if report.get("lane") != "package-expansion":
        return []
    classification = report.get("failure_classification")
    if classification is None:
        return [
            "",
            "## Failure classification",
            "",
            "Unavailable: no usable default-branch baseline, so this run "
            "reports no introduced or inherited count. Absence of a count is "
            "not a clean result.",
        ]
    baseline = classification["baseline"]
    introduced = classification["introduced"]
    absent = [row["test"] for row in introduced if row["baseline"] == "absent"]
    lines = [
        "",
        "## Failure classification",
        "",
        f"{len(introduced)} introduced, {len(classification['inherited'])} "
        f"inherited, {len(classification['unrun_targets'])} targets unrun "
        f"({len(classification['unrun_tests'])} selected tests never "
        f"executed).",
        "",
        f"Baseline: `{baseline['workflow']}` run {baseline['run_id']} at "
        f"`{baseline['head_sha'][:9]}`, recorded {baseline['created_at']}, "
        f"{baseline['commits_behind_candidate_base']} commits behind this "
        f"candidate's base `{baseline['candidate_base_sha'][:9]}`. It executed "
        f"{baseline['observed_cases']} cases, {baseline['failing_cases']} of "
        f"them failing.",
        "",
        f"Run link: {baseline['run_url']}",
        "",
        "Inherited and introduced are the baseline's verdict at its own "
        "commit, not at this candidate's base. Over that distance a test that "
        "went red on the default branch is reported here as introduced, which "
        "over-reports; and a test that was failing at the baseline, was fixed "
        "on the default branch since, and is broken again by this candidate is "
        "reported as inherited, which under-reports. The producer takes the "
        "newest baseline this candidate's base contains, so that distance is "
        "as small as a recorded baseline allows.",
    ]
    if absent:
        lines.append(
            f"{len(absent)} introduced row(s) are absent from the baseline "
            f"rather than green in it, so the baseline confirms nothing about "
            f"them."
        )
    if introduced:
        lines.extend(["", "### Introduced", ""])
        lines.extend(
            f"- `{row['test']}` (baseline: {row['baseline']})"
            for row in introduced
        )
    if classification["unrun_targets"]:
        lines.extend(["", "### Unrun targets", ""])
        lines.extend(
            f"- `{identity}`" for identity in classification["unrun_targets"]
        )
    if classification["inherited"]:
        lines.extend(["", "### Inherited", ""])
        lines.extend(f"- `{test}`" for test in classification["inherited"])
    lines.extend(
        [
            "",
            "Each shard's elapsed time is reported without a pass or fail "
            "verdict. The per-shard weight is a longest-processing-time "
            "balancing input derived from serial per-target measurements, "
            "while the executor runs up to sixteen targets of one package in a "
            "single command, so the weight overstates a completed shard and "
            "understates one the deadline cut. Comparing it against a budget "
            "carries no information, which is why no soft-budget finding "
            "exists.",
        ]
    )
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="generate the exact event plan")
    plan.add_argument("--event-name", required=True)
    plan.add_argument("--pr-head", default="")
    plan.add_argument("--before", default="")
    plan.add_argument("--after", default="")
    plan.add_argument(
        "--targeted-packages",
        default="",
        help="trusted comma-separated package frontier for targeted_rebase",
    )
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument(
        "--config",
        type=Path,
        default=ROOT / ".config/ci-test-targets.toml",
    )
    plan.add_argument(
        "--duration-baseline",
        type=Path,
        default=DURATION_BASELINE_PATH,
    )

    duration_baseline = subparsers.add_parser(
        "build-duration-baseline",
        help="build a reviewed change-owned target-duration baseline",
    )
    duration_baseline.add_argument(
        "--sample",
        action="append",
        nargs=2,
        metavar=("PLAN", "RECEIPTS_ROOT"),
        type=Path,
        required=True,
        help="authenticated plan and its four change-owned receipt directories",
    )
    duration_baseline.add_argument("--output", type=Path, required=True)

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

    classify = subparsers.add_parser(
        "classify-paths",
        help="reject any changed path the planner would refuse",
    )
    classify.add_argument(
        "paths",
        nargs="*",
        help="repo-relative changed paths; none means nothing to classify",
    )
    classify.add_argument(
        "--from-git",
        action="store_true",
        help=(
            "derive the changed set from the working tree when the check "
            "runs, rather than taking it on the command line"
        ),
    )
    classify.add_argument(
        "--base",
        default="origin/main",
        help="with --from-git, diff the branch from its merge base with this ref",
    )
    classify.add_argument(
        "--config",
        type=Path,
        default=ROOT / ".config/ci-test-targets.toml",
    )

    report = subparsers.add_parser("report", help="validate or summarize receipts")
    report.add_argument("--plan", type=Path, required=True)
    report.add_argument("--lane", choices=tuple(LANE_KEYS), required=True)
    report.add_argument("--receipts-root", type=Path, required=True)
    report.add_argument("--standing-coverage", type=Path)
    report.add_argument(
        "--failure-baseline",
        type=Path,
        help=(
            "recorded default-branch failure baseline manifest, which the "
            "package-expansion lane classifies every observed failure "
            "against; the conventional path its producing step writes is "
            "read when this is omitted"
        ),
    )
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--required", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        targeted_packages = (
            tuple(args.targeted_packages.split(","))
            if args.targeted_packages
            else ()
        )
        result = generate_plan(
            ROOT,
            event_name=args.event_name,
            pr_head=args.pr_head,
            before=args.before,
            after=args.after,
            config_path=args.config,
            duration_baseline_path=args.duration_baseline,
            targeted_packages=targeted_packages,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(result))
        print(
            "CHANGE-OWNED PLAN: PASS: "
            f"{len(result['change_owned'])} required, "
            f"{len(result['package_expansion'])} informational"
        )
        return 0
    if args.command == "build-duration-baseline":
        result = build_duration_baseline(
            [(plan, receipts) for plan, receipts in args.sample]
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(canonical_json(result))
        print(
            "CHANGE-OWNED DURATION BASELINE: PASS: "
            f"{len(result['targets'])} targets from "
            f"{len(result['sources'])} samples"
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
    if args.command == "classify-paths":
        if args.from_git and args.paths:
            raise ValueError("--from-git derives the set; pass no paths with it")
        paths = (
            working_tree_changed_paths(base=args.base)
            if args.from_git
            else list(args.paths)
        )
        refused = classify_changed_paths(
            paths,
            config=read_config(args.config),
            base=args.base if args.from_git else None,
        )
        for path, disposition in refused:
            print(
                f"CHANGE-OWNED CI: FAIL: "
                f"{refused_path_message(disposition, path)}",
                file=sys.stderr,
            )
        if refused:
            print(
                f"CLASSIFY PATHS: {len(refused)} path(s) that no rule routes. "
                f"Add a [[path_rule]] row in .config/ci-test-targets.toml, or "
                f"move the file under a package root.",
                file=sys.stderr,
            )
            return 1
        print(f"CLASSIFY PATHS: PASS ({len(paths)} path(s))")
        return 0
    if args.lane == "change-owned" and not args.required:
        raise ValueError("change-owned report requires --required")
    if args.lane == "package-expansion" and args.required:
        raise ValueError("package-expansion report must remain informational")
    if args.lane == "change-owned" and args.failure_baseline is not None:
        # An argparse default would make the exact conventional path the one
        # spelling this fail-closed lane accepts, so the default is resolved
        # in the informational branch instead and absence stays absence here.
        raise ValueError(
            "change-owned report is fail-closed and takes no failure baseline"
        )
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
                "manual_gate_targets": [],
                "standing_reused_targets": [],
                "failures": [str(error)],
            }
            _write_report_files(args.output, result)
            print(f"CHANGE-OWNED REPORT: FAIL: {error}", file=sys.stderr)
            return 1
    else:
        try:
            documents = load_receipt_documents(args.receipts_root)
            failure_baseline = None
            baseline_unavailable = None
            manifest = (
                args.failure_baseline
                if args.failure_baseline is not None
                else Path(DEFAULT_FAILURE_BASELINE)
            )
            if not manifest.is_file():
                baseline_unavailable = (
                    f"the baseline manifest is absent at {manifest}; "
                    f"its producing step did not leave one"
                )
            else:
                failure_baseline = load_failure_baseline(manifest)
            result = summarize_package_expansion(
                plan,
                [receipt for receipt, _ in documents],
                junit_documents={
                    receipt["shard"]: junit for receipt, junit in documents
                },
                failure_baseline=failure_baseline,
                baseline_unavailable=baseline_unavailable,
            )
        except (
            ValueError,
            KeyError,
            OSError,
            json.JSONDecodeError,
            ET.ParseError,
        ) as error:
            result = {
                "version": EXPANSION_REPORT_VERSION,
                "lane": "package-expansion",
                "required": False,
                "success": False,
                "observed_success": False,
                "plan_digest": plan.get("plan_digest"),
                "covered_targets": [],
                "standing_reused_targets": [],
                "failure_classification": None,
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
