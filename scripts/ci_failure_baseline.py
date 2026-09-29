#!/usr/bin/env python3
"""Record a default-branch test-failure baseline for the expansion report.

The informational package-expansion lane reports whether a candidate
*introduced* a failure. That question needs a second observation: the same
tests, run on the default branch, recently enough that the candidate's own
merge base contains the commit they ran on. The nightly full-workspace job in
``heavy-e2e.yml`` already produces exactly that evidence and uploads it as
JUnit, so this script selects one of those runs and writes a manifest naming
the downloaded documents and the run they came from.

Selection is deliberately narrow. Only completed schedule and full-scope
workflow_dispatch runs of the named workflow on the named branch qualify,
newest first. Every expected JUnit artifact must be present and unexpired.
Dispatches also need an exact run/head-bound full-scope receipt. An incomplete
run is skipped because a missing shard reports inherited failures as
introduced. When no run qualifies, the consuming report fails with it.

Newest is not the same as usable. The consumer refuses a baseline the
candidate's merge base does not contain, and a pull request's merge ref is not
recomputed as the default branch advances, so the newest nightly is routinely
ahead of an older candidate's base. Given ``--plan``, this script reads that
base and takes the newest qualifying run the base actually contains. That both
finds a usable baseline where the newest one is not, and shrinks the window in
which a test fixed on the default branch since the baseline can be re-broken by
the candidate and still read as inherited.

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
import tempfile
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
DISPATCH_SCOPE_ARTIFACT = "linux-extended-dispatch-scope"
DISPATCH_SCOPE_FILENAME = "scope.json"
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
        if not isinstance(run, dict):
            continue
        if run.get("head_branch") != branch:
            continue
        if run.get("event") not in {"schedule", "workflow_dispatch"}:
            continue
        rows.append(
            {
                "run_id": str(run["id"]),
                "run_url": run["html_url"],
                "head_sha": run["head_sha"],
                "created_at": run["created_at"],
                "event": run["event"],
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
    matches: dict[str, list[dict[str, Any]]] = {
        name: [] for name in expected
    }
    for chunk in _json_documents(payload):
        artifacts = chunk.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"unexpected artifact listing for run {run_id}")
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                raise ValueError(f"unexpected artifact row for run {run_id}")
            name = artifact.get("name")
            if isinstance(name, str) and name in matches:
                matches[name].append(artifact)
    return all(
        len(rows) == 1 and rows[0].get("expired") is False
        for rows in matches.values()
    )


def dispatch_is_full_scope(
    repository: str,
    candidate: dict[str, Any],
    runner: Runner,
) -> bool:
    """Check the dispatch receipt against the API's exact run and head."""
    with tempfile.TemporaryDirectory() as tmp:
        destination = Path(tmp)
        runner([
            "gh", "run", "download", candidate["run_id"],
            "--repo", repository, "--dir", str(destination),
            "--name", DISPATCH_SCOPE_ARTIFACT,
        ])
        document = destination / DISPATCH_SCOPE_FILENAME
        if not document.is_file():
            return False
        try:
            receipt = json.loads(document.read_text())
        except (OSError, json.JSONDecodeError):
            return False
    if not isinstance(receipt, dict) or type(receipt.get("version")) is not int:
        return False
    return receipt == {
        "version": 1,
        "event": "workflow_dispatch",
        "scope": "all",
        "run_id": candidate["run_id"],
        "head_sha": candidate["head_sha"],
    }


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


def _resolves(repo: Path, value: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{value}^{{commit}}"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        ).returncode
        == 0
    )


def base_contains(repo: Path, commit: str, base_sha: str) -> bool:
    """Whether the candidate's own merge base contains this baseline commit.

    The two commits are not symmetric and their failure modes must not be. A
    baseline commit this clone cannot resolve is genuinely not contained, so
    it is skipped and the next run is tried. A *base* this clone cannot
    resolve disqualifies every run alike, and the caller would then report
    that none retains its artifacts, which is a false reason that sends a
    reader to look at retention. That is raised instead, which puts this
    function and the consumer's `_ancestor_distance` loud at the same end.
    """
    if not _resolves(repo, base_sha):
        raise ValueError(
            f"candidate base {base_sha} is not present in {repo}, so no "
            f"baseline can be tested for containment"
        )
    if not _resolves(repo, commit):
        return False
    if commit == base_sha:
        return True
    contained = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, base_sha],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return contained.returncode == 0


def plan_base_sha(plan: Path) -> str:
    document = json.loads(plan.read_text())
    base_sha = document.get("base_sha")
    if not isinstance(base_sha, str) or not base_sha:
        raise ValueError(f"plan has no base_sha: {plan}")
    return base_sha


def prepare(
    *,
    repository: str,
    workflow: str,
    branch: str,
    artifacts: Sequence[str],
    search_runs: int,
    output: Path,
    runner: Runner = _run,
    base_sha: str | None = None,
    repo: Path | None = None,
    contains: Callable[[Path, str, str], bool] = base_contains,
) -> dict[str, Any]:
    if workflow == DEFAULT_WORKFLOW and tuple(artifacts) != DEFAULT_ARTIFACTS:
        raise ValueError(
            f"{DEFAULT_WORKFLOW} baseline requires all four JUnit artifacts"
        )
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
    skipped_ahead = 0
    for candidate in candidates:
        if base_sha is not None and not contains(
            repo or Path("."),
            candidate["head_sha"],
            base_sha,
        ):
            skipped_ahead += 1
            continue
        expected = (
            (*artifacts, DISPATCH_SCOPE_ARTIFACT)
            if candidate["event"] == "workflow_dispatch"
            else artifacts
        )
        if not run_has_artifacts(
            repository,
            candidate["run_id"],
            expected,
            runner,
        ):
            continue
        if candidate["event"] == "workflow_dispatch" and not dispatch_is_full_scope(
            repository, candidate, runner,
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
    ahead = (
        f" ({skipped_ahead} of them are not contained in the candidate base "
        f"{base_sha})"
        if skipped_ahead
        else ""
    )
    raise ValueError(
        f"none of the {len(candidates)} most recent completed {workflow} runs "
        f"on {branch}{ahead} retains every baseline artifact "
        f"{list(artifacts)} and, for a dispatch, a matching full-scope receipt; "
        f"without a complete baseline the expansion report "
        f"cannot tell an introduced failure from an inherited one"
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
    parser.add_argument(
        "--plan",
        type=Path,
        help=(
            "candidate plan whose base_sha the baseline must be contained in; "
            "without it the newest qualifying run is taken, which the report "
            "may then refuse"
        ),
    )
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path("."),
        help="checkout whose history decides containment",
    )
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
        base_sha=plan_base_sha(args.plan) if args.plan is not None else None,
        repo=args.repo_path,
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
