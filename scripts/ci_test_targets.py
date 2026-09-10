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
import re
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]
LIB_KINDS = {"lib", "rlib", "dylib", "cdylib", "staticlib", "proc-macro"}


def read_targets(path: Path) -> list[tuple[str, str]]:
    data = tomllib.loads(path.read_text())
    if set(data) != {"version", "target"} or type(data["version"]) is not int or data["version"] != 1:
        raise ValueError("selection requires version = 1 and target rows")
    rows = data["target"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection must contain at least one integration target")
    result = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"package", "name"}:
            raise ValueError("each target must name exactly its package and name")
        for value in row.values():
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", value):
                raise ValueError(f"invalid package/target identity: {value!r}")
        result.append((row["package"], row["name"]))
    return result


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


def cargo_args(metadata: dict, selected: list[tuple[str, str]]) -> list[str]:
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("integration selection is empty or contains duplicate identities")
    by_name: dict[str, list[tuple[str, dict]]] = {}
    for package in packages(metadata):
        for target in package["targets"]:
            if "test" in target["kind"]:
                by_name.setdefault(target["name"], []).append((package["name"], target))
    args = ["--workspace", "--lib", "--bins"]
    for package, name in selected:
        matches = by_name.get(name, [])
        if len(matches) != 1 or matches[0][0] != package:
            raise ValueError(f"missing or ambiguous integration target: {package}::{name}")
        if matches[0][1].get("required-features"):
            raise ValueError(f"fast integration must use default features: {package}::{name}")
        args.extend(["--test", name])
    return args


def validate_listing(data: dict, metadata: dict, selected: list[tuple[str, str]]) -> None:
    cargo_args(metadata, selected)
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
    args = cargo_args(metadata, selected)
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    if not target.is_absolute():
        target = root / target
    receipts = target / "ci-fast"
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "selection.json").write_text(json.dumps({"integrations": selected, "cargo_args": args}, indent=2) + "\n")
    timing = {}
    commands = [
        ["cargo", "build", "--workspace", "--lib", "--bins", "--locked"],
        ["cargo", "nextest", "list", *args, "--locked", "--profile", "ci-fast", "--ignore-default-filter", "--message-format", "json"],
        ["cargo", "nextest", "run", *args, "--locked", "--profile", "ci-fast", "--ignore-default-filter", "--no-fail-fast"],
    ]
    try:
        for label, command in zip(("build_products", "compile_and_list_tests", "run_tests"), commands):
            print("+ " + " ".join(command), flush=True)
            started = time.monotonic()
            try:
                result = subprocess.run(command, cwd=root, check=True, text=True,
                                        stdout=subprocess.PIPE if label == "compile_and_list_tests" else None)
                if label == "compile_and_list_tests":
                    (receipts / "test-list.json").write_text(result.stdout)
                    validate_listing(json.loads(result.stdout), metadata, selected)
            finally:
                timing[label] = round(time.monotonic() - started, 3)
    finally:
        (receipts / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")


if __name__ == "__main__":
    try:
        run()
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"CI FAST: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
    print("CI FAST: PASS")
