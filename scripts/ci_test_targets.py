#!/usr/bin/env python3
"""Build only the reviewed PR test targets, retaining every default-feature unit.

Cargo selectors bound compilation; nextest filters alone do not. Full workspace
coverage lives in heavy-e2e.yml. The binary-ID receipt uses nextest's stable API:
https://nexte.st/rustdoc/nextest_metadata/struct.RustBinaryId.html
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

if __package__:
    from . import ci_change_owned
else:
    import ci_change_owned

ROOT = Path(__file__).resolve().parents[1]
LIB_KINDS = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}
STANDING_EXECUTION = dict(ci_change_owned.STANDING_EXECUTION)


def read_targets(path: Path) -> list[tuple[str, str]]:
    config = ci_change_owned.read_config(path)
    return [
        (identity.package, identity.target)
        for identity in config.standing_targets
    ]


def packages(metadata: dict) -> list[dict]:
    members = set(metadata["workspace_members"])
    result = [p for p in metadata["packages"] if p["id"] in members]
    if len(result) != len(members) or not result:
        raise ValueError("metadata does not enumerate the workspace")
    return result


def default_features(package: dict) -> set[str]:
    features = package.get("features", {})
    enabled, pending = set(), ["default"]
    while pending:
        feature = pending.pop()
        if feature in enabled:
            continue
        enabled.add(feature)
        pending.extend(features.get(feature, []))
    return enabled


def cargo_selections(metadata: dict, selected: list[tuple[str, str]]) -> list[list[str]]:
    """Select units once, isolating shared test names by their exact package.

    A target whose required features are not all default-enabled builds in its
    own group with exactly those features, shared only with targets of the same
    package and feature set, so no other unit is compiled with them.
    """
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("integration selection is empty or contains duplicate identities")
    by_name: dict[str, list[tuple[str, dict]]] = {}
    workspace = {package["name"]: package for package in packages(metadata)}
    for package in packages(metadata):
        for target in package["targets"]:
            if "test" in target["kind"]:
                by_name.setdefault(target["name"], []).append((package["name"], target))
    args = ["--workspace", "--lib", "--bins"]
    scoped: dict[str, list[str]] = {}
    gated: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for package, name in selected:
        matches = by_name.get(name, [])
        exact = [target for owner, target in matches if owner == package]
        if len(exact) != 1:
            raise ValueError(f"missing or duplicate integration target: {package}::{name}")
        required = tuple(sorted(set(exact[0].get("required-features", []))))
        undeclared = [
            feature for feature in required
            if feature not in workspace[package].get("features", {})
        ]
        if undeclared:
            raise ValueError(
                f"integration target requires undeclared features {undeclared}: {package}::{name}"
            )
        if not set(required) <= default_features(workspace[package]):
            gated.setdefault(
                (package, required), ["-p", package, "--features", ",".join(required)]
            ).extend(["--test", name])
        elif len(matches) == 1:
            args.extend(["--test", name])
        else:
            scoped.setdefault(package, ["-p", package]).extend(["--test", name])
    return [
        args,
        *(scoped[package] for package in sorted(scoped)),
        *(gated[key] for key in sorted(gated)),
    ]


def merge_listings(documents: list[dict]) -> dict:
    suites = {}
    for document in documents:
        incoming = document["rust-suites"]
        duplicates = set(suites) & set(incoming)
        if duplicates:
            raise ValueError(f"duplicate compiled test targets: {sorted(duplicates)}")
        suites.update(incoming)
    return {"rust-suites": suites,
            "test-count": sum(len(suite["testcases"]) for suite in suites.values())}


def validate_listing(
    data: dict,
    metadata: dict,
    selected: list[tuple[str, str]],
) -> list[str]:
    cargo_selections(metadata, selected)
    integrations = {f"{package}::{name}" for package, name in selected}
    expected = set(integrations)
    for package in packages(metadata):
        for target in package["targets"]:
            if not set(target.get("required-features", [])) <= default_features(package):
                continue
            kinds = set(target["kind"])
            if kinds & LIB_KINDS:
                expected.add(package["name"])
            elif "bin" in kinds:
                expected.add(f"{package['name']}::bin/{target['name']}")
    suites = data["rust-suites"]
    if set(suites) != expected:
        raise ValueError(f"compiled test target mismatch: missing={sorted(expected - set(suites))}, extra={sorted(set(suites) - expected)}")
    selected_tests = []
    for binary, suite in suites.items():
        tests = suite["testcases"]
        active = [info for info in tests.values() if not info["ignored"]]
        if binary in integrations and not active:
            raise ValueError(f"selected integration has no non-ignored tests: {binary}")
        if any(info["filter-match"]["status"] != "matches" for info in active):
            raise ValueError(f"fast lane silently filters a non-ignored test: {binary}")
        if binary in integrations:
            selected_tests.extend(
                f"{binary}::{name}"
                for name, info in tests.items()
                if not info["ignored"]
            )
    if len(selected_tests) != len(set(selected_tests)):
        raise ValueError("fast lane listing contains duplicate integration tests")
    return sorted(selected_tests)


def junit_integration_tests(
    path: Path,
    selected: list[tuple[str, str]],
) -> list[str]:
    selected_targets = {f"{package}::{target}" for package, target in selected}
    tests = []
    for case in ET.parse(path).iter("testcase"):
        classname = case.get("classname")
        if classname not in selected_targets:
            continue
        name = case.get("name")
        if not name:
            raise ValueError(
                f"standing JUnit testcase has no name for {classname}"
            )
        ci_change_owned.require_executed_junit_case(case, path)
        tests.append(f"{classname}::{name}")
    if len(tests) != len(set(tests)):
        raise ValueError("standing JUnit contains duplicate integration tests")
    return sorted(tests)


def run(root: Path = ROOT, *, candidate_sha: str | None = None) -> None:
    config = ci_change_owned.read_config(
        root / ".config/ci-test-targets.toml"
    )
    selected = [
        (identity.package, identity.target)
        for identity in config.standing_targets
    ]
    if candidate_sha is None:
        candidate_sha = ci_change_owned._commit(root, "HEAD")
    elif not ci_change_owned.SHA.fullmatch(candidate_sha):
        raise ValueError("candidate_sha must be a full lowercase commit SHA")
    metadata = json.loads(subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=root, check=True, stdout=subprocess.PIPE, text=True,
    ).stdout)
    selections = cargo_selections(metadata, selected)
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target.is_absolute():
        target = root / target
    receipts = target / "ci-fast"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "selection.json").write_text(json.dumps({"integrations": selected, "cargo_selections": selections}, indent=2) + "\n")
    timing = {}
    commands = []
    listings = []
    suite_documents = []
    selected_tests: list[str] = []
    executed_tests: list[str] = []
    caught: Exception | None = None
    junit = target / "nextest/ci-fast/junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    junit.unlink(missing_ok=True)

    def invoke(label, command, *, capture=False):
        commands.append({"label": label, "command": command})
        print("+ " + " ".join(command), flush=True)
        started = time.monotonic()
        try:
            return subprocess.run(command, cwd=root, check=True, text=True,
                                  stdout=subprocess.PIPE if capture else None)
        finally:
            timing[label] = round(time.monotonic() - started, 3)

    try:
        invoke("build_products", ["cargo", "build", "--workspace", "--lib", "--bins", "--locked"])
        for index, args in enumerate(selections):
            result = invoke(f"compile_and_list_tests_{index}", [
                "cargo", "nextest", "list", *args, "--locked", "--profile", "ci-fast",
                "--ignore-default-filter", "--message-format", "json",
            ], capture=True)
            (receipts / f"test-list-{index}.json").write_text(result.stdout)
            listings.append(json.loads(result.stdout))
        combined = merge_listings(listings)
        (receipts / "test-list.json").write_text(json.dumps(combined, indent=2) + "\n")
        selected_tests = validate_listing(combined, metadata, selected)
        for index, args in enumerate(selections):
            # nextest overwrites its profile's JUnit path on each invocation.
            # Remove the previous receipt so a missing write cannot look fresh.
            junit.unlink(missing_ok=True)
            try:
                invoke(f"run_tests_{index}", [
                    "cargo", "nextest", "run", *args, "--locked", "--profile", "ci-fast",
                    "--ignore-default-filter", "--no-fail-fast",
                ])
            finally:
                if junit.is_file():
                    group_junit = receipts / f"junit-{index}.xml"
                    shutil.copyfile(junit, group_junit)
                    suite_documents.append(group_junit)
                    executed_tests.extend(
                        junit_integration_tests(group_junit, selected)
                    )
            if not junit.is_file():
                raise ValueError(f"missing JUnit for command group {index}")
    except Exception as error:
        caught = error
    finally:
        (receipts / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        (receipts / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        try:
            ci_change_owned._write_junit(junit, suite_documents)
        except (ValueError, ET.ParseError) as error:
            if caught is None:
                caught = error
        expected_targets = sorted(
            f"{package}::{name}" for package, name in selected
        )
        executed_tests = sorted(set(executed_tests))
        executed_targets = []
        for identity in expected_targets:
            prefix = f"{identity}::"
            selected_for_target = {
                test for test in selected_tests if test.startswith(prefix)
            }
            executed_for_target = {
                test for test in executed_tests if test.startswith(prefix)
            }
            if selected_for_target and selected_for_target == executed_for_target:
                executed_targets.append(identity)
        failures = []
        if caught is not None:
            failures.append(str(caught))
        if sorted(selected_tests) != executed_tests:
            failures.append(
                "standing integration test results are incomplete: "
                f"selected={sorted(selected_tests)}, "
                f"executed={executed_tests}"
            )
        if executed_targets != expected_targets:
            failures.append(
                "standing integration target results are incomplete: "
                f"selected={expected_targets}, executed={executed_targets}"
            )
        coverage = {
            "version": ci_change_owned.STANDING_COVERAGE_VERSION,
            "candidate_sha": candidate_sha,
            "config_digest": ci_change_owned.config_digest(config),
            "execution": dict(STANDING_EXECUTION),
            "selected_targets": expected_targets,
            "executed_targets": executed_targets,
            "selected_tests": sorted(selected_tests),
            "executed_tests": executed_tests,
            "success": not failures,
            "failures": failures,
        }
        ci_change_owned.attach_standing_coverage_digest(coverage)
        (receipts / "coverage.json").write_bytes(
            ci_change_owned.canonical_json(coverage)
        )
    if caught is not None:
        raise caught
    if failures:
        raise ValueError("; ".join(failures))


if __name__ == "__main__":
    try:
        run()
    except (ValueError, KeyError, OSError, ET.ParseError, subprocess.CalledProcessError) as error:
        print(f"CI FAST: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
    print("CI FAST: PASS")
