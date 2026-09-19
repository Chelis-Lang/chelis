#!/usr/bin/env python3
"""Record a default-branch test-failure baseline for the expansion report.

The informational package-expansion lane reports whether a candidate
*introduced* a failure. That question needs a second observation: the same
tests, run on the default branch, recently enough that the candidate's own
merge base contains the commit they ran on. The nightly full-workspace job in
``heavy-e2e.yml`` already produces exactly that evidence and uploads it as
JUnit, so this script selects one of those runs and writes a manifest naming
the downloaded documents and the run they came from.

Selection is deliberately narrow. Only completed runs of the named workflow on
the named branch qualify, newest first, and a run qualifies only when every
expected JUnit artifact is present and unexpired. A run that is still in
progress, or that lost an artifact to retention, is skipped rather than
partially used, because a baseline missing a shard reports that shard's
inherited failures as introduced. When no run in the search window qualifies
the script fails, and the report that consumes the manifest fails with it: a
report that quietly skips the comparison would call every failure clean.

The workflow conclusion is not a selection criterion. The nightly is routinely
red, and a red nightly is precisely the evidence this baseline exists to carry.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable, Sequence
from typing import Any

BASELINE_VERSION = 1
DEFAULT_WORKFLOW = "heavy-e2e.yml"
DEFAULT_BRANCH = "main"
DEFAULT_ARTIFACTS = (
    "junit-linux-full-1",
    "junit-linux-full-2",
    "junit-linux-full-3",
    "junit-linux-full-4",
)
DEFAULT_SEARCH_RUNS = 10

Runner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def list_candidate_runs(
    repository: str,
    workflow: str,
    branch: str,
    limit: int,
    runner: Runner,
) -> list[dict[str, Any]]:
    payload = runner(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/workflows/{workflow}/runs"
            f"?branch={branch}&status=completed&per_page={limit}",
        ]
    )
    document = json.loads(payload)
    runs = document.get("workflow_runs")
    if not isinstance(runs, list):
        raise ValueError(f"unexpected workflow run listing for {workflow}")
    rows = []
    for run in runs:
        if run.get("head_branch") != branch:
            continue
        rows.append(
            {
                "run_id": str(run["id"]),
                "run_url": run["html_url"],
                "head_sha": run["head_sha"],
                "created_at": run["created_at"],
            }
        )
    return rows


def run_has_artifacts(
    repository: str,
    run_id: str,
    expected: Sequence[str],
    runner: Runner,
) -> bool:
    payload = runner(
        [
            "gh",
            "api",
            "--paginate",
            f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100",
        ]
    )
    available: set[str] = set()
    for chunk in _json_documents(payload):
        for artifact in chunk.get("artifacts", []):
            if not artifact.get("expired", False):
                available.add(artifact.get("name", ""))
    return set(expected) <= available


def _json_documents(payload: str) -> list[dict[str, Any]]:
    """Decode one or more concatenated JSON objects from a paginated reply."""
    decoder = json.JSONDecoder()
    documents = []
    index = 0
    while index < len(payload):
        while index < len(payload) and payload[index].isspace():
            index += 1
        if index >= len(payload):
            break
        document, index = decoder.raw_decode(payload, index)
        if not isinstance(document, dict):
            raise ValueError("unexpected artifact listing payload")
        documents.append(document)
    return documents


def download_artifacts(
    repository: str,
    run_id: str,
    expected: Sequence[str],
    destination: Path,
    runner: Runner,
) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    command = ["gh", "run", "download", run_id, "--repo", repository, "--dir", str(destination)]
    for name in expected:
        command.extend(["--name", name])
    runner(command)
    documents = []
    for name in expected:
        document = destination / name / "junit.xml"
        if not document.is_file():
            raise ValueError(f"baseline artifact {name} has no junit.xml")
        documents.append(f"{name}/junit.xml")
    return documents


def prepare(
    *,
    repository: str,
    workflow: str,
    branch: str,
    artifacts: Sequence[str],
    search_runs: int,
    output: Path,
    runner: Runner = _run,
) -> dict[str, Any]:
    candidates = list_candidate_runs(
        repository,
        workflow,
        branch,
        search_runs,
        runner,
    )
    if not candidates:
        raise ValueError(
            f"no completed {workflow} run on {branch} to use as a baseline"
        )
    for candidate in candidates:
        if not run_has_artifacts(
            repository,
            candidate["run_id"],
            artifacts,
            runner,
        ):
            continue
        documents = download_artifacts(
            repository,
            candidate["run_id"],
            artifacts,
            output,
            runner,
        )
        manifest = {
            "version": BASELINE_VERSION,
            "workflow": workflow,
            "run_id": candidate["run_id"],
            "run_url": candidate["run_url"],
            "head_sha": candidate["head_sha"],
            "created_at": candidate["created_at"],
            "documents": documents,
        }
        output.mkdir(parents=True, exist_ok=True)
        (output / "baseline.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return manifest
    raise ValueError(
        f"none of the {len(candidates)} most recent completed {workflow} runs "
        f"on {branch} retains every baseline artifact {list(artifacts)}; "
        f"without a complete baseline the expansion report cannot tell an "
        f"introduced failure from an inherited one"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="owner/name of the repository holding the baseline runs",
    )
    parser.add_argument("--workflow", default=DEFAULT_WORKFLOW)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument(
        "--artifact",
        action="append",
        dest="artifacts",
        help="expected JUnit artifact name; repeatable",
    )
    parser.add_argument("--search-runs", type=int, default=DEFAULT_SEARCH_RUNS)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.repository:
        raise ValueError("--repository is required outside a GitHub Actions run")
    if args.search_runs < 1:
        raise ValueError("--search-runs must be positive")
    manifest = prepare(
        repository=args.repository,
        workflow=args.workflow,
        branch=args.branch,
        artifacts=tuple(args.artifacts or DEFAULT_ARTIFACTS),
        search_runs=args.search_runs,
        output=args.output,
    )
    print(
        f"FAILURE BASELINE: {manifest['workflow']} run {manifest['run_id']} "
        f"at {manifest['head_sha']} ({manifest['created_at']})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"FAILURE BASELINE: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
