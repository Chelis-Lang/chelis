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
    """Select units once, isolating shared test names by their exact package."""
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
    for package, name in selected:
        matches = by_name.get(name, [])
        exact = [target for owner, target in matches if owner == package]
        if len(exact) != 1:
            raise ValueError(f"missing or duplicate integration target: {package}::{name}")
        if not set(exact[0].get("required-features", [])) <= default_features(workspace[package]):
            raise ValueError(f"fast integration must use default features: {package}::{name}")
        if len(matches) == 1:
            args.extend(["--test", name])
        else:
            scoped.setdefault(package, ["-p", package]).extend(["--test", name])
    return [args, *(scoped[package] for package in sorted(scoped))]


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


def validate_listing(data: dict, metadata: dict, selected: list[tuple[str, str]]) -> None:
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
    for binary, suite in suites.items():
        tests = suite["testcases"]
        active = [info for info in tests.values() if not info["ignored"]]
        if binary in integrations and not active:
            raise ValueError(f"selected integration has no non-ignored tests: {binary}")
        if any(info["filter-match"]["status"] != "matches" for info in active):
            raise ValueError(f"fast lane silently filters a non-ignored test: {binary}")


def run(root: Path = ROOT) -> None:
    selected = read_targets(root / ".config/ci-test-targets.toml")
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
        validate_listing(combined, metadata, selected)
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
            if not junit.is_file():
                raise ValueError(f"missing JUnit for command group {index}")
    finally:
        (receipts / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        (receipts / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")
        ci_change_owned._write_junit(junit, suite_documents)


if __name__ == "__main__":
    try:
        run()
    except (ValueError, KeyError, OSError, ET.ParseError, subprocess.CalledProcessError) as error:
        print(f"CI FAST: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
    print("CI FAST: PASS")
