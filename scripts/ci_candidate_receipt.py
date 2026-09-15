#!/usr/bin/env python3
"""Issue a default-branch-owned receipt for one fully green PR candidate."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
from typing import Any
from zipfile import BadZipFile, ZipFile

if __package__:
    from . import ci_candidate_identity as identity
    from . import ci_contract_paths
else:
    import ci_candidate_identity as identity
    import ci_contract_paths


SCHEMA = "chelis-ci-candidate-receipt/v1"
IDENTITY_FILENAME = "candidate-identity.json"
PAGE_SIZE = 100
PASSING_CONCLUSIONS = frozenset({"success", "skipped"})
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")


class ReceiptError(RuntimeError):
    """Trusted collection found malformed or contradictory evidence."""


class ReceiptNotReady(RuntimeError):
    """The candidate has not yet accumulated all reusable evidence."""


@dataclass(frozen=True)
class WorkflowContract:
    contexts: tuple[str, ...]
    identity_artifact: str | None = None


WORKFLOWS = {
    "ci.yml": WorkflowContract(
        contexts=(
            "Lint and Unit Tests (Linux)",
            "Integration Tests (Linux)",
            "Backend Sanitizers",
            "SMT Feature Build (Linux)",
            "Docs",
            "No AI authorship markers",
        ),
        identity_artifact="candidate-identity-ci",
    ),
    "conformance.yml": WorkflowContract(
        contexts=("Hull Conformance Gate (Linux)",),
        identity_artifact="candidate-identity-hull",
    ),
    "changelog.yml": WorkflowContract(contexts=("Changelog",)),
    "pr-contract-acknowledgements.yml": WorkflowContract(
        contexts=("PR Contract Acknowledgements",)
    ),
    "pr-base-retarget.yml": WorkflowContract(
        contexts=("PR Base Retarget Validation",)
    ),
}
CONTEXT_WORKFLOW = {
    context: workflow_file
    for workflow_file, contract in WORKFLOWS.items()
    for context in contract.contexts
}
TRIGGER_PATHS = {
    f".github/workflows/{workflow_file}" for workflow_file in WORKFLOWS
}

Api = Callable[[str], object]
ArtifactDownloader = Callable[[int], bytes]
CommitFetcher = Callable[[str], None]


def reuse_eligibility(paths: Sequence[str]) -> tuple[bool, list[str]]:
    return ci_contract_paths.reuse_eligibility(paths)


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReceiptError(f"{label} was not an object")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReceiptError(f"{label} must be a positive integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptError(f"{label} must be a nonempty string")
    return value


def _full_sha(value: object, label: str) -> str:
    candidate = _string(value, label)
    if not SHA.fullmatch(candidate):
        raise ReceiptError(f"{label} must be a lowercase 40-character SHA")
    return candidate


def _paged(api: Api, endpoint: str, key: str) -> list[Mapping[str, Any]]:
    collected: list[Mapping[str, Any]] = []
    for page in range(1, 21):
        separator = "&" if "?" in endpoint else "?"
        payload = _mapping(
            api(f"{endpoint}{separator}per_page={PAGE_SIZE}&page={page}"),
            f"{key} response",
        )
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise ReceiptError(f"{key} response had no list")
        collected.extend(
            _mapping(entry, f"{key} entry") for entry in entries
        )
        if len(entries) < PAGE_SIZE:
            return collected
    raise ReceiptError(f"{key} response did not end within 20 pages")


def _run_pr_number(run: Mapping[str, Any]) -> int:
    pulls = run.get("pull_requests")
    if not isinstance(pulls, list):
        raise ReceiptError("workflow run had no pull_requests list")
    numbers = {
        item.get("number")
        for item in pulls
        if isinstance(item, Mapping) and type(item.get("number")) is int
    }
    if len(numbers) != 1:
        raise ReceiptError("workflow run did not identify exactly one pull request")
    return _positive_integer(numbers.pop(), "workflow run pull request number")


def _trigger_run(
    api: Api, repository: str, trigger_run_id: int
) -> Mapping[str, Any]:
    run = _mapping(
        api(f"repos/{repository}/actions/runs/{trigger_run_id}"),
        "trigger workflow run",
    )
    if _positive_integer(run.get("id"), "trigger run id") != trigger_run_id:
        raise ReceiptError("trigger workflow run id did not match the event")
    path = _string(run.get("path"), "trigger workflow path")
    if path not in TRIGGER_PATHS:
        raise ReceiptError(f"unexpected trigger workflow {path!r}")
    if run.get("event") != "pull_request":
        raise ReceiptError("trigger workflow was not a pull_request run")
    if run.get("status") != "completed":
        raise ReceiptNotReady("trigger workflow has not completed")
    head_repository = _mapping(
        run.get("head_repository"), "trigger head_repository"
    )
    if head_repository.get("full_name") != repository:
        raise ReceiptError("trigger workflow ran from another repository")
    return run


def _current_pull(
    api: Api,
    repository: str,
    pr_number: int,
    expected_head_sha: str,
) -> tuple[Mapping[str, Any], str]:
    pull = _mapping(
        api(f"repos/{repository}/pulls/{pr_number}"),
        "pull request",
    )
    if pull.get("state") != "open":
        raise ReceiptNotReady("pull request is no longer open")
    head = _mapping(pull.get("head"), "pull request head")
    if head.get("sha") != expected_head_sha:
        raise ReceiptNotReady("pull request head advanced after the workflow run")
    head_repository = _mapping(
        head.get("repo"), "pull request head repository"
    )
    if head_repository.get("full_name") != repository:
        raise ReceiptError("pull request head belongs to another repository")
    base = _mapping(pull.get("base"), "pull request base")
    base_ref = _string(base.get("ref"), "pull request base ref")
    if not REF.fullmatch(base_ref) or ".." in base_ref or base_ref.endswith("/"):
        raise ReceiptError("pull request base ref is not an ordinary branch name")
    base_repository = _mapping(
        base.get("repo"), "pull request base repository"
    )
    if base_repository.get("full_name") != repository:
        raise ReceiptError("pull request base belongs to another repository")
    return pull, base_ref


def _required_checks(
    api: Api, repository: str, base_ref: str
) -> dict[str, int]:
    branch = _mapping(
        api(f"repos/{repository}/branches/{base_ref}"),
        "base branch",
    )
    if branch.get("protected") is not True:
        raise ReceiptError(f"base branch {base_ref} is not protected")
    protection = _mapping(branch.get("protection"), "branch protection")
    status_checks = _mapping(
        protection.get("required_status_checks"), "required status checks"
    )
    entries = status_checks.get("checks")
    if not isinstance(entries, list) or not entries:
        raise ReceiptError("base branch declares no required check entries")
    required: dict[str, int] = {}
    for entry in entries:
        check = _mapping(entry, "required check")
        context = _string(check.get("context"), "required check context")
        app_id = _positive_integer(check.get("app_id"), f"{context} app_id")
        if context in required:
            raise ReceiptError(f"required context {context!r} is duplicated")
        if context not in CONTEXT_WORKFLOW:
            raise ReceiptError(
                f"required context {context!r} has no receipt workflow mapping"
            )
        required[context] = app_id
    return required


def _latest_check_runs(
    api: Api, repository: str, head_sha: str
) -> dict[str, Mapping[str, Any]]:
    runs = _paged(
        api,
        f"repos/{repository}/commits/{head_sha}/check-runs",
        "check_runs",
    )
    observed: dict[str, Mapping[str, Any]] = {}
    for run in runs:
        name = _string(run.get("name"), "check run name")
        current = observed.get(name)
        order = (
            str(run.get("started_at") or ""),
            _positive_integer(run.get("id"), f"{name} check run id"),
        )
        if current is None:
            observed[name] = run
            continue
        current_order = (
            str(current.get("started_at") or ""),
            _positive_integer(current.get("id"), f"{name} check run id"),
        )
        if order > current_order:
            observed[name] = run
    return observed


def _required_job_provenance(
    *,
    api: Api,
    repository: str,
    pr_number: int,
    head_sha: str,
    context: str,
    check_run: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    workflow_file = CONTEXT_WORKFLOW[context]
    check_run_id = _positive_integer(
        check_run.get("id"), f"{context} check run id"
    )
    job = _mapping(
        api(f"repos/{repository}/actions/jobs/{check_run_id}"),
        f"{context} workflow job",
    )
    if _positive_integer(job.get("id"), f"{context} job id") != check_run_id:
        raise ReceiptError(f"{context}: Actions returned another workflow job")
    if job.get("name") != context:
        raise ReceiptError(
            f"{context}: expected workflow job name, observed {job.get('name')!r}"
        )
    if job.get("head_sha") != head_sha:
        raise ReceiptError(
            f"{context}: expected workflow job ran on another head"
        )
    if (
        job.get("status") != check_run.get("status")
        or job.get("conclusion") != check_run.get("conclusion")
    ):
        raise ReceiptError(
            f"{context}: check run and expected workflow job disagree"
        )
    run_id = _positive_integer(job.get("run_id"), f"{context} run id")
    job_attempt = _positive_integer(
        job.get("run_attempt"), f"{context} run attempt"
    )
    run = _mapping(
        api(f"repos/{repository}/actions/runs/{run_id}"),
        f"{context} workflow run",
    )
    if _positive_integer(run.get("id"), f"{context} run id") != run_id:
        raise ReceiptError(f"{context}: Actions returned another workflow run")
    current_attempt = _positive_integer(
        run.get("run_attempt"), f"{context} current run attempt"
    )
    if job_attempt > current_attempt:
        raise ReceiptError(
            f"{context}: job attempt exceeds the workflow run attempt"
        )
    expected_path = f".github/workflows/{workflow_file}"
    if run.get("path") != expected_path:
        raise ReceiptError(
            f"{context}: run path was {run.get('path')!r}, "
            f"expected {expected_path!r}"
        )
    allowed_events = {"pull_request"}
    if workflow_file == "pr-base-retarget.yml":
        allowed_events.add("pull_request_target")
    if run.get("event") not in allowed_events:
        raise ReceiptError(
            f"{context}: workflow event {run.get('event')!r} is not allowed"
        )
    head_repository = _mapping(
        run.get("head_repository"), f"{context} head_repository"
    )
    if head_repository.get("full_name") != repository:
        raise ReceiptError(f"{context}: workflow ran from another repository")
    if _run_pr_number(run) != pr_number:
        raise ReceiptError(f"{context}: workflow ran for another pull request")
    exact_run = dict(run)
    exact_run["run_attempt"] = job_attempt
    return job, exact_run


def _required_evidence(
    *,
    api: Api,
    repository: str,
    pr_number: int,
    head_sha: str,
    required: Mapping[str, int],
    observed: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, object]], dict[str, Mapping[str, Any]]]:
    evidence: list[dict[str, object]] = []
    workflows: dict[str, Mapping[str, Any]] = {}
    for context in sorted(required):
        check_run = observed.get(context)
        if check_run is None:
            raise ReceiptNotReady(
                f"{context}: no check run was reported for this head"
            )
        status = check_run.get("status")
        conclusion = check_run.get("conclusion")
        if status != "completed" or conclusion not in PASSING_CONCLUSIONS:
            raise ReceiptNotReady(
                f"{context}: latest check is {conclusion or status or 'unknown'}"
            )
        application = _mapping(
            check_run.get("app"), f"{context} check app"
        )
        app_id = _positive_integer(application.get("id"), f"{context} app id")
        if app_id != required[context]:
            raise ReceiptError(
                f"{context}: app {app_id} reported the check, "
                f"but protection requires app {required[context]}"
            )
        workflow_file = CONTEXT_WORKFLOW[context]
        _, workflow_run = _required_job_provenance(
            api=api,
            repository=repository,
            pr_number=pr_number,
            head_sha=head_sha,
            context=context,
            check_run=check_run,
        )
        check_run_id = _positive_integer(
            check_run.get("id"), f"{context} check run id"
        )
        previous = workflows.get(workflow_file)
        if previous is not None and (
            previous.get("id") != workflow_run.get("id")
            or previous.get("run_attempt") != workflow_run.get("run_attempt")
        ):
            raise ReceiptError(
                f"{workflow_file}: required contexts came from different runs"
            )
        workflows[workflow_file] = workflow_run
        evidence.append(
            {
                "context": context,
                "app_id": app_id,
                "conclusion": conclusion,
                "check_run_id": check_run_id,
                "workflow_file": workflow_file,
                "workflow_run_id": _positive_integer(
                    workflow_run.get("id"), f"{workflow_file} run id"
                ),
                "workflow_run_attempt": _positive_integer(
                    workflow_run.get("run_attempt"),
                    f"{workflow_file} run attempt",
                ),
            }
        )
    missing = set(WORKFLOWS) - set(workflows)
    if missing:
        raise ReceiptError(
            f"required check set did not cover workflows: {sorted(missing)}"
        )
    return evidence, workflows


def _identity_archive(
    archive_bytes: bytes, workflow_file: str
) -> Mapping[str, Any]:
    try:
        with ZipFile(BytesIO(archive_bytes)) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if len(members) != 1:
                raise ReceiptError(
                    f"{workflow_file} identity archive must contain exactly one file"
                )
            member = members[0]
            if member.filename != IDENTITY_FILENAME:
                raise ReceiptError(
                    f"{workflow_file} identity archive must contain "
                    f"{IDENTITY_FILENAME}"
                )
            raw = archive.read(member)
    except BadZipFile as error:
        raise ReceiptError(
            f"{workflow_file} identity artifact was not a zip archive"
        ) from error
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(
            f"{workflow_file} identity artifact was not readable JSON"
        ) from error
    return _mapping(payload, f"{workflow_file} candidate identity")


def _download_identity(
    *,
    api: Api,
    download_artifact: ArtifactDownloader,
    repository: str,
    workflow_file: str,
    run: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract = WORKFLOWS[workflow_file]
    if contract.identity_artifact is None:
        raise ReceiptError(f"{workflow_file} has no candidate identity artifact")
    run_id = _positive_integer(run.get("id"), f"{workflow_file} run id")
    artifacts = _paged(
        api,
        f"repos/{repository}/actions/runs/{run_id}/artifacts",
        "artifacts",
    )
    matches = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == contract.identity_artifact
        and artifact.get("expired") is not True
    ]
    if not matches:
        raise ReceiptNotReady(
            f"{workflow_file}: candidate identity artifact is unavailable"
        )
    if len(matches) != 1:
        raise ReceiptError(
            f"{workflow_file}: duplicate candidate identity artifacts"
        )
    artifact_id = _positive_integer(
        matches[0].get("id"), f"{workflow_file} artifact id"
    )
    return _identity_archive(download_artifact(artifact_id), workflow_file)


def _has_commit(repository_path: Path, sha: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repository_path,
        capture_output=True,
    )
    return completed.returncode == 0


def _validated_identity(
    payload: Mapping[str, Any],
    *,
    repository_path: Path,
    repository: str,
    workflow_file: str,
    workflow_run: Mapping[str, Any],
    pr_number: int,
    head_sha: str,
    base_ref: str,
    fetch_commit: CommitFetcher | None,
) -> dict[str, object]:
    candidate_sha = _full_sha(
        payload.get("candidate_sha"), f"{workflow_file} candidate_sha"
    )
    base_sha = _full_sha(payload.get("base_sha"), f"{workflow_file} base_sha")
    for sha in (candidate_sha, base_sha, head_sha):
        if not _has_commit(repository_path, sha):
            if fetch_commit is None:
                raise ReceiptError(f"commit {sha} is unavailable locally")
            fetch_commit(sha)
        if not _has_commit(repository_path, sha):
            raise ReceiptError(f"commit {sha} remained unavailable after fetch")
    try:
        expected = identity.build_identity(
            repository_path=repository_path,
            repository=repository,
            workflow_file=workflow_file,
            run_id=_positive_integer(
                workflow_run.get("id"), f"{workflow_file} run id"
            ),
            run_attempt=_positive_integer(
                workflow_run.get("run_attempt"), f"{workflow_file} run attempt"
            ),
            pr_number=pr_number,
            head_sha=head_sha,
            base_ref=base_ref,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
        )
    except identity.IdentityError as error:
        raise ReceiptError(
            f"{workflow_file} candidate identity failed trusted recomputation: "
            f"{error}"
        ) from error
    if set(payload) != set(expected):
        raise ReceiptError(
            f"{workflow_file} candidate identity has unexpected fields"
        )
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ReceiptError(
                f"{workflow_file} candidate identity {key} does not match "
                "trusted recomputation"
            )
    return expected


def collect_receipt(
    *,
    repository: str,
    trigger_run_id: int,
    receipt_run_id: int,
    repository_path: Path,
    api: Api,
    download_artifact: ArtifactDownloader,
    fetch_commit: CommitFetcher | None = None,
) -> dict[str, object]:
    if not REPOSITORY.fullmatch(repository):
        raise ReceiptError("repository must be an owner/name identifier")
    trigger_run_id = _positive_integer(trigger_run_id, "trigger_run_id")
    receipt_run_id = _positive_integer(receipt_run_id, "receipt_run_id")
    repository_path = Path(repository_path)
    if not repository_path.is_dir():
        raise ReceiptError("repository_path must be an existing directory")

    trigger = _trigger_run(api, repository, trigger_run_id)
    head_sha = _full_sha(trigger.get("head_sha"), "trigger head_sha")
    pr_number = _run_pr_number(trigger)
    _, base_ref = _current_pull(
        api, repository, pr_number, head_sha
    )
    required = _required_checks(api, repository, base_ref)

    observed = _latest_check_runs(api, repository, head_sha)
    required_evidence, workflow_runs = _required_evidence(
        api=api,
        repository=repository,
        pr_number=pr_number,
        head_sha=head_sha,
        required=required,
        observed=observed,
    )

    identities = {
        workflow_file: _validated_identity(
            _download_identity(
                api=api,
                download_artifact=download_artifact,
                repository=repository,
                workflow_file=workflow_file,
                run=workflow_runs[workflow_file],
            ),
            repository_path=repository_path,
            repository=repository,
            workflow_file=workflow_file,
            workflow_run=workflow_runs[workflow_file],
            pr_number=pr_number,
            head_sha=head_sha,
            base_ref=base_ref,
            fetch_commit=fetch_commit,
        )
        for workflow_file in ("ci.yml", "conformance.yml")
    }
    shared_keys = (
        "repository",
        "event",
        "pr_number",
        "head_sha",
        "base_ref",
        "base_sha",
        "patch_base_sha",
        "candidate_sha",
        "candidate_parents",
        "patch_id",
        "patch_digest",
        "changed_paths",
        "changed_paths_sha256",
    )
    ci_identity = identities["ci.yml"]
    hull_identity = identities["conformance.yml"]
    for key in shared_keys:
        if ci_identity[key] != hull_identity[key]:
            raise ReceiptError(
                f"CI and Hull candidate identity disagree on {key}"
            )
    reuse_eligible, reuse_blockers = reuse_eligibility(
        ci_identity["changed_paths"]
    )

    return {
        "schema": SCHEMA,
        "repository": repository,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "base_sha": ci_identity["base_sha"],
        "patch_base_sha": ci_identity["patch_base_sha"],
        "candidate_sha": ci_identity["candidate_sha"],
        "candidate_parents": ci_identity["candidate_parents"],
        "patch_id": ci_identity["patch_id"],
        "patch_digest": ci_identity["patch_digest"],
        "changed_paths": ci_identity["changed_paths"],
        "changed_paths_sha256": ci_identity["changed_paths_sha256"],
        "reuse_eligible": reuse_eligible,
        "reuse_blockers": reuse_blockers,
        "required_checks": required_evidence,
        "workflow_runs": {
            workflow_file: {
                "id": _positive_integer(
                    run.get("id"), f"{workflow_file} run id"
                ),
                "attempt": _positive_integer(
                    run.get("run_attempt"), f"{workflow_file} run attempt"
                ),
                "conclusion": run.get("conclusion"),
            }
            for workflow_file, run in workflow_runs.items()
        },
        "source_trigger_run_id": trigger_run_id,
        "receipt_workflow_run_id": receipt_run_id,
    }


def write_receipt(payload: Mapping[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _gh_api_json(endpoint: str) -> object:
    completed = subprocess.run(
        ["gh", "api", endpoint],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ReceiptError(f"gh api failed: {message or 'no output'}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReceiptError(f"gh api returned unreadable JSON: {error}") from error


def _gh_download(repository: str, artifact_id: int) -> bytes:
    completed = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{repository}/actions/artifacts/{artifact_id}/zip",
        ],
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise ReceiptError(
            f"artifact {artifact_id} download failed: {message or 'no output'}"
        )
    return completed.stdout


def _git_fetch(repository_path: Path, sha: str) -> None:
    completed = subprocess.run(
        ["git", "fetch", "--no-tags", "origin", sha],
        cwd=repository_path,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ReceiptError(
            f"could not fetch commit {sha}: {message or 'no output'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--trigger-run-id", type=int, required=True)
    parser.add_argument("--receipt-run-id", type=int, required=True)
    parser.add_argument("--repository-path", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        result = collect_receipt(
            repository=arguments.repository,
            trigger_run_id=arguments.trigger_run_id,
            receipt_run_id=arguments.receipt_run_id,
            repository_path=arguments.repository_path,
            api=_gh_api_json,
            download_artifact=lambda artifact_id: _gh_download(
                arguments.repository, artifact_id
            ),
            fetch_commit=lambda sha: _git_fetch(arguments.repository_path, sha),
        )
    except ReceiptNotReady as error:
        print(f"candidate receipt not issued: {error}")
        return 0
    except (ReceiptError, identity.IdentityError) as error:
        print(f"candidate receipt collection failed: {error}")
        return 2
    write_receipt(result, arguments.output)
    print(
        f"issued candidate receipt for PR #{result['pr_number']} "
        f"head {result['head_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
