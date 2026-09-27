#!/usr/bin/env python3
"""Authorize whole-target Fast Tests reuse for manual package expansion.

Run this from the default-branch checkout, never from the PR checkout. The
candidate's plan only names possible overlap; a successful exact-candidate
Fast job, complete receipt and JUnit, matching configuration, and the actual
Cargo metadata must independently justify every omitted target. A missing or
unfinished Fast job writes an explicit fallback and leaves expansion runnable.
"""
from __future__ import annotations

import argparse
import ast
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from . import ci_change_owned as owned
else:
    import ci_change_owned as owned

VERSION = 1
WORKFLOW = ".github/workflows/ci.yml"
FAST_JOB = "Fast Tests (Linux)"
FAST_RECEIPTS = "ci-fast-receipts"
FAST_JUNIT = "junit-linux-fast"


def _digest(document: Mapping[str, Any]) -> str:
    return owned._digest_without(document, "digest")


def _seal(document: dict[str, Any]) -> dict[str, Any]:
    document["digest"] = _digest(document)
    return document


def fallback_manifest(plan: Mapping[str, Any], shard: int, reason: str) -> dict[str, Any]:
    return _seal({
        "version": VERSION,
        "candidate_sha": plan["candidate_sha"],
        "plan_digest": plan["plan_digest"],
        "shard": shard,
        "status": "fallback",
        "reason": reason,
        "source": None,
        "fast_tests": [],
        "reused_targets": [],
        "reused_tests": [],
    })


def candidate_supports_reuse(candidate_root: Path) -> bool:
    """Keep the default-branch workflow usable for older open PR heads."""
    module = candidate_root / "scripts/ci_change_owned.py"
    helper = candidate_root / "scripts/ci_expansion_fast_reuse.py"
    if not module.is_file() or not helper.is_file():
        return False
    try:
        tree = ast.parse(module.read_text())
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "RECEIPT_VERSION"
                for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 3
        for node in tree.body
    )


def _profile_settings(path: Path, profile: str) -> dict[str, Any]:
    document = tomllib.loads(path.read_text())
    profiles = document.get("profile")
    if not isinstance(profiles, dict) or profile not in profiles:
        raise ValueError(f"missing Nextest profile {profile}")
    settings = copy.deepcopy(profiles[profile])
    junit = settings.pop("junit", None)
    if junit != {"path": "junit.xml"}:
        raise ValueError(f"unexpected {profile} JUnit configuration")
    return settings


def require_equivalent_profiles(path: Path) -> None:
    if _profile_settings(path, "ci-fast") != _profile_settings(path, "ci-full"):
        raise ValueError("Fast and expansion Nextest profile settings differ")


def _junit_integration_tests(junit: bytes, targets: set[str]) -> list[str]:
    try:
        root = ET.fromstring(junit)
    except ET.ParseError as error:
        raise ValueError(f"Fast JUnit is malformed: {error}") from error
    if root.tag not in {"testsuite", "testsuites"}:
        raise ValueError("Fast JUnit root is not a test suite")
    tests = []
    for case in root.iter("testcase"):
        target = case.get("classname")
        if target not in targets:
            continue
        name = case.get("name")
        if not name or case.find("skipped") is not None:
            raise ValueError(f"Fast JUnit has missing or skipped case in {target}")
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"Fast JUnit has a failing case in {target}")
        tests.append(f"{target}::{name}")
    if len(tests) != len(set(tests)):
        raise ValueError("Fast JUnit repeats an integration test identity")
    return sorted(tests)


def _actual_target_features(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for package in metadata["packages"]:
        for target in package["targets"]:
            if "test" in target["kind"]:
                identity = f"{package['name']}::{target['name']}"
                if identity in result:
                    raise ValueError(f"duplicate Cargo target {identity}")
                result[identity] = sorted(target.get("required-features", []))
    return result


def authorized_targets(
    plan: Mapping[str, Any],
    coverage: Mapping[str, Any],
    junit: bytes,
    metadata: Mapping[str, Any],
    *,
    shard_targets: Sequence[str],
    groups: Sequence[Sequence[str]],
) -> dict[str, list[str]]:
    """Return complete default-feature targets safe to omit in this shard."""
    owned.verify_standing_coverage_digest(coverage)
    if coverage["candidate_sha"] != plan["candidate_sha"]:
        raise ValueError("Fast coverage is for a stale synthetic candidate")
    if coverage["config_digest"] != plan["config_digest"]:
        raise ValueError("Fast coverage configuration differs from the plan")
    if coverage["execution"] != owned.STANDING_EXECUTION:
        raise ValueError("Fast test mode differs from the reviewed mode")
    standing = sorted(plan["standing_targets"])
    if sorted(coverage["selected_targets"]) != standing:
        raise ValueError("Fast selected target inventory is incomplete")
    if sorted(coverage["executed_targets"]) != standing:
        raise ValueError("Fast executed target inventory is incomplete")
    if coverage["success"] is not True or coverage["failures"]:
        raise ValueError("Fast coverage did not succeed")
    selected_tests = sorted(coverage["selected_tests"])
    if selected_tests != sorted(coverage["executed_tests"]):
        raise ValueError("Fast selected test results are incomplete")
    if selected_tests != _junit_integration_tests(junit, set(standing)):
        raise ValueError("Fast JUnit does not prove every selected test ran")
    if any(not any(test.startswith(f"{target}::") for test in selected_tests)
           for target in standing):
        raise ValueError("Fast receipt has an empty standing target")

    actual_features = _actual_target_features(metadata)
    for identity in shard_targets:
        if actual_features.get(identity) != plan["target_features"].get(identity):
            raise ValueError(f"Cargo feature set differs from plan for {identity}")
    if set(shard_targets) - set(plan["package_expansion"]):
        raise ValueError("shard requests a target outside package expansion")
    if {target for group in groups for target in group} != set(shard_targets):
        raise ValueError("execution groups do not cover exactly this shard")

    excluded = {row["identity"] for row in plan["target_exclusions"]}
    excluded |= {owned.TestIdentity.parse(row["identity"]).target_identity.canonical
                 for row in plan["test_exclusions"]}
    manual = {row["identity"] for row in plan["manual_only_targets"]}
    reusable: set[str] = set()
    for group in groups:
        # A grouped Nextest command enables the union of the targets' extra
        # features. With any extra feature, even its default-feature sibling
        # can have a different set of tests, so the whole group runs.
        if any(plan["target_features"].get(target) for target in group):
            continue
        for target in group:
            if target in standing and target not in excluded | manual:
                reusable.add(target)
    fast_tests = sorted(test for test in selected_tests
                        if owned.TestIdentity.parse(test).target_identity.canonical
                        in set(shard_targets))
    return {
        "fast_tests": fast_tests,
        "reused_targets": sorted(reusable),
        "reused_tests": sorted(test for test in fast_tests
                               if owned.TestIdentity.parse(test).target_identity.canonical
                               in reusable),
    }


def _gh_json(endpoint: str) -> dict[str, Any]:
    completed = subprocess.run(["gh", "api", endpoint], capture_output=True)
    if completed.returncode != 0:
        raise ValueError(f"GitHub API failed: {completed.stderr.decode(errors='replace').strip()}")
    result = json.loads(completed.stdout)
    if not isinstance(result, dict):
        raise ValueError("GitHub API returned no object")
    return result


def _gh_archive(repository: str, artifact_id: int) -> bytes:
    completed = subprocess.run(
        ["gh", "api", f"repos/{repository}/actions/artifacts/{artifact_id}/zip"],
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValueError(f"artifact {artifact_id} download failed")
    return completed.stdout


def _archive_member(archive: bytes, name: str) -> bytes:
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as opened:
            if name not in opened.namelist():
                raise ValueError(f"missing artifact member {name}")
            info = opened.getinfo(name)
            if info.file_size > 20_000_000:
                raise ValueError(f"artifact member {name} is too large")
            return opened.read(info)
    except zipfile.BadZipFile as error:
        raise ValueError("artifact is not a valid zip") from error


def _source_evidence(
    repository: str,
    candidate_sha: str,
    *,
    api: Callable[[str], dict[str, Any]],
    download: Callable[[str, int], bytes],
    pinned_run: int | None = None,
) -> tuple[dict[str, int], dict[str, Any], bytes] | None:
    endpoint = f"repos/{repository}/actions/workflows/ci.yml/runs?head_sha={candidate_sha}&event=pull_request&per_page=100"
    runs = api(endpoint).get("workflow_runs", [])
    if not isinstance(runs, list):
        raise ValueError("CI run listing is malformed")
    for run in sorted(runs, key=lambda row: row.get("id", 0), reverse=True):
        run_id = run.get("id")
        if pinned_run is not None and run_id != pinned_run:
            continue
        if (run.get("head_sha") != candidate_sha
                or run.get("path") != WORKFLOW
                or run.get("event") != "pull_request"
                or run.get("status") != "completed"
                or (run.get("head_repository") or {}).get("full_name") != repository):
            continue
        jobs = api(f"repos/{repository}/actions/runs/{run_id}/jobs?per_page=100").get("jobs", [])
        successful = [job for job in jobs if job.get("name") == FAST_JOB
                      and job.get("conclusion") == "success"
                      and job.get("head_sha") == candidate_sha]
        if len(successful) != 1:
            continue
        artifacts = api(f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100").get("artifacts", [])
        by_name = {name: [item for item in artifacts
                          if item.get("name") == name and item.get("expired") is False
                          and (item.get("workflow_run") or {}).get("id") == run_id
                          and (item.get("workflow_run") or {}).get("head_sha") == candidate_sha]
                   for name in (FAST_RECEIPTS, FAST_JUNIT)}
        if any(len(rows) != 1 for rows in by_name.values()):
            continue
        receipt_id = by_name[FAST_RECEIPTS][0]["id"]
        junit_id = by_name[FAST_JUNIT][0]["id"]
        coverage = json.loads(_archive_member(
            download(repository, receipt_id), "ci-fast/coverage.json"
        ))
        junit = _archive_member(download(repository, junit_id), "junit.xml")
        source = {"run_id": run_id, "job_id": successful[0]["id"],
                  "receipts_artifact_id": receipt_id, "junit_artifact_id": junit_id}
        return source, coverage, junit
    return None


def _cargo_metadata(candidate_root: Path) -> dict[str, Any]:
    completed = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=candidate_root, capture_output=True, text=True, check=True,
    )
    return json.loads(completed.stdout)


def authorize(
    plan: Mapping[str, Any], shard: int, candidate_root: Path, repository: str,
    *, api: Callable[[str], dict[str, Any]] = _gh_json,
    download: Callable[[str, int], bytes] = _gh_archive,
    pinned_run: int | None = None,
) -> dict[str, Any]:
    if shard not in owned.SHARDS:
        raise ValueError("invalid expansion shard")
    if owned._commit(candidate_root, "HEAD") != plan["candidate_sha"]:
        raise ValueError("candidate checkout differs from expansion plan")
    try:
        owned.verify_plan_digest(plan)
        selected = owned.execution_shards(plan, "package-expansion")[str(shard)]
        if not selected:
            return fallback_manifest(plan, shard, "empty expansion shard")
        config = owned.read_config(candidate_root / ".config/ci-test-targets.toml")
        if owned.config_digest(config) != plan["config_digest"]:
            raise ValueError("candidate configuration differs from expansion plan")
        if sorted(identity.canonical for identity in config.standing_targets) != sorted(plan["standing_targets"]):
            raise ValueError("candidate standing inventory differs from plan")
        require_equivalent_profiles(candidate_root / ".config/nextest.toml")
        evidence = _source_evidence(
            repository, plan["candidate_sha"], api=api, download=download,
            pinned_run=pinned_run,
        )
        if evidence is None:
            return fallback_manifest(plan, shard, "no completed exact-candidate Fast job with both artifacts")
        source, coverage, junit = evidence
        groups = [[identity.canonical for identity in group]
                  for group in owned.execution_groups(plan, lane="package-expansion", selected=selected)]
        result = authorized_targets(
            plan, coverage, junit, _cargo_metadata(candidate_root),
            shard_targets=selected, groups=groups,
        )
        return _seal({
            "version": VERSION,
            "candidate_sha": plan["candidate_sha"],
            "plan_digest": plan["plan_digest"],
            "shard": shard,
            "status": "fast_evidence",
            "reason": "complete successful exact-candidate Fast job",
            "source": source,
            **result,
        })
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError,
            json.JSONDecodeError) as error:
        return fallback_manifest(plan, shard, str(error))


def validate_manifest(plan: Mapping[str, Any], manifest: Mapping[str, Any], shard: int) -> None:
    expected = {"version", "candidate_sha", "plan_digest", "shard", "status",
                "reason", "source", "fast_tests", "reused_targets", "reused_tests", "digest"}
    if set(manifest) != expected or manifest.get("version") != VERSION:
        raise ValueError("Fast reuse manifest schema mismatch")
    if manifest.get("digest") != _digest(manifest):
        raise ValueError("Fast reuse manifest digest mismatch")
    if (manifest["candidate_sha"] != plan["candidate_sha"]
            or manifest["plan_digest"] != plan["plan_digest"]
            or manifest["shard"] != shard):
        raise ValueError("Fast reuse manifest candidate, plan, or shard mismatch")
    selected = set(owned.execution_shards(plan, "package-expansion")[str(shard)])
    reused = manifest["reused_targets"]
    if not isinstance(reused, list) or reused != sorted(set(reused)) or not set(reused) <= selected:
        raise ValueError("Fast reuse target selection is invalid")
    for key in ("fast_tests", "reused_tests"):
        tests = manifest[key]
        if not isinstance(tests, list) or tests != sorted(set(tests)):
            raise ValueError(f"Fast reuse {key} must be sorted unique")
        if any(owned.TestIdentity.parse(test).target_identity.canonical not in selected
               for test in tests):
            raise ValueError(f"Fast reuse {key} names another shard")
    if (not set(manifest["reused_tests"]) <= set(manifest["fast_tests"])
            or {owned.TestIdentity.parse(test).target_identity.canonical
                for test in manifest["reused_tests"]} != set(reused)):
        raise ValueError("Fast reuse tests do not exactly cover reused targets")
    if manifest["status"] == "fallback":
        if manifest["source"] is not None or reused or manifest["reused_tests"] or manifest["fast_tests"]:
            raise ValueError("fallback cannot authorize omission")
    elif manifest["status"] != "fast_evidence" or not isinstance(manifest["source"], dict):
        raise ValueError("Fast reuse source status is invalid")


def verify_authorizations(
    plan: Mapping[str, Any], candidate_root: Path, repository: str,
    manifests_root: Path, receipts_root: Path,
    *, api: Callable[[str], dict[str, Any]] = _gh_json,
    download: Callable[[str, int], bytes] = _gh_archive,
) -> None:
    """Default-branch check of all omissions and their exact source evidence."""
    owned.verify_plan_digest(plan)
    documents = owned.load_receipt_documents(receipts_root)
    by_shard = {receipt["shard"]: receipt for receipt, _ in documents}
    if set(by_shard) != set(owned.SHARDS) or len(documents) != len(owned.SHARDS):
        raise ValueError("Fast reuse verification needs four exact shard receipts")
    for shard in owned.SHARDS:
        manifest = json.loads((manifests_root / f"fast-reuse-{shard}.json").read_text())
        validate_manifest(plan, manifest, shard)
        receipt = by_shard[shard]
        expected_digest = manifest["digest"] if receipt["selected_targets"] else None
        if (receipt["reused_targets"] != manifest["reused_targets"]
                or receipt["reused_tests"] != manifest["reused_tests"]
                or receipt["fast_reuse_digest"] != expected_digest):
            raise ValueError(f"shard {shard} reuse receipt differs from trusted manifest")
        if manifest["status"] == "fast_evidence":
            expected = authorize(
                plan, shard, candidate_root, repository, api=api,
                download=download, pinned_run=manifest["source"]["run_id"],
            )
            if expected != manifest:
                raise ValueError(f"shard {shard} Fast reuse source is not reproducible")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("supports", "authorize", "verify"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--repository")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifests-root", type=Path)
    parser.add_argument("--receipts-root", type=Path)
    args = parser.parse_args(argv)
    if args.command == "supports":
        if args.github_output is None:
            parser.error("supports requires --github-output")
        supported = candidate_supports_reuse(args.candidate_root)
        args.github_output.parent.mkdir(parents=True, exist_ok=True)
        with args.github_output.open("a") as handle:
            handle.write(f"supports_reuse={'true' if supported else 'false'}\n")
        print(f"FAST REUSE CANDIDATE SUPPORT: {supported}")
        return 0
    if args.plan is None or args.repository is None:
        parser.error("authorize and verify require --plan and --repository")
    plan = owned.load_json(args.plan)
    if args.command == "authorize":
        if args.shard not in owned.SHARDS or args.output is None:
            parser.error("authorize requires --shard and --output")
        result = authorize(plan, args.shard, args.candidate_root, args.repository)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(owned.canonical_json(result))
        print(f"FAST REUSE {args.shard}: {result['status']}: "
              f"{len(result['reused_targets'])} targets, {len(result['reused_tests'])} tests; "
              f"{result['reason']}")
    else:
        if args.manifests_root is None or args.receipts_root is None:
            parser.error("verify requires --manifests-root and --receipts-root")
        verify_authorizations(plan, args.candidate_root, args.repository,
                              args.manifests_root, args.receipts_root)
        print("FAST REUSE VERIFY: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
        print(f"FAST REUSE: FAIL: {error}", file=sys.stderr)
        sys.exit(1)
