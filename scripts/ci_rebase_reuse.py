#!/usr/bin/env python3
"""Select a fail-closed incremental CI lane for a trusted PR rebase."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from zipfile import BadZipFile, ZipFile

if __package__:
    from . import ci_candidate_identity as identity
    from . import ci_candidate_receipt as candidate_receipt
    from . import ci_change_owned
    from .ci_detect_docs_only import is_docs_only
else:
    import ci_candidate_identity as identity
    import ci_candidate_receipt as candidate_receipt
    import ci_change_owned
    from ci_detect_docs_only import is_docs_only


SCHEMA = "chelis-ci-rebase-reuse/v1"
RECEIPT_SCHEMA = "chelis-ci-candidate-receipt/v1"
RECEIPT_PREFIX = "pr-candidate-receipt-"
RECEIPT_WORKFLOW = ".github/workflows/pr-candidate-receipt.yml"
RECEIPT_FILENAME = "receipt.json"
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
PASSING_CONCLUSIONS = frozenset({"success"})
TRUSTED_ROOT = Path(__file__).resolve().parents[1]


class ReuseError(RuntimeError):
    """Current event or trusted evidence is malformed."""


Api = Callable[[str], object]
ArtifactDownloader = Callable[[int], bytes]


@dataclass(frozen=True)
class PriorReceipt:
    payload: Mapping[str, Any]
    artifact_id: int
    run_id: int
    created_at: str


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReuseError(f"{label} was not an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReuseError(f"{label} must be a nonempty string")
    return value


def _positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReuseError(f"{label} must be a positive integer")
    return value


def _sha(value: object, label: str) -> str:
    candidate = _string(value, label)
    if not SHA.fullmatch(candidate):
        raise ReuseError(f"{label} must be a lowercase 40-character SHA")
    return candidate


def _timestamp(value: object, label: str) -> datetime:
    raw = _string(value, label)
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReuseError(f"{label} must be an ISO-8601 timestamp") from error


def _git(
    repository_path: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_path,
        input=input_bytes,
        capture_output=True,
        env=None if env is None else {**os.environ, **env},
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.decode(errors="replace").strip()
        raise ReuseError(
            f"git {' '.join(arguments)} failed: {detail or 'no error output'}"
        )
    return completed


def _commit(repository_path: Path, value: str) -> str:
    resolved = _git(
        repository_path, "rev-parse", f"{value}^{{commit}}"
    ).stdout.decode().strip()
    return _sha(resolved, f"resolved commit {value}")


def _ensure_commit(repository_path: Path, value: str) -> None:
    present = _git(
        repository_path,
        "cat-file",
        "-e",
        f"{value}^{{commit}}",
        check=False,
    )
    if present.returncode == 0:
        return
    _git(repository_path, "fetch", "--no-tags", "origin", value)
    _commit(repository_path, value)


def _is_ancestor(repository_path: Path, ancestor: str, descendant: str) -> bool:
    completed = _git(
        repository_path,
        "merge-base",
        "--is-ancestor",
        ancestor,
        descendant,
        check=False,
    )
    if completed.returncode == 0:
        return True
    if completed.returncode == 1:
        return False
    detail = completed.stderr.decode(errors="replace").strip()
    raise ReuseError(
        f"cannot compare {ancestor} with {descendant}: {detail or 'no output'}"
    )


def _merge_base(repository_path: Path, left: str, right: str) -> str:
    resolved = _git(repository_path, "merge-base", left, right).stdout.decode().strip()
    return _sha(resolved, f"merge base of {left} and {right}")


def _changed_paths(
    repository_path: Path, before_sha: str, after_sha: str
) -> list[str]:
    raw = _git(
        repository_path,
        "diff",
        "--name-only",
        "--no-renames",
        "-z",
        before_sha,
        after_sha,
    ).stdout
    paths = sorted(item.decode("utf-8") for item in raw.split(b"\0") if item)
    if len(paths) != len(set(paths)):
        raise ReuseError("rebase delta contains duplicate changed paths")
    return paths


def _interaction_frontier(
    repository_path: Path,
    *,
    prior_candidate: str,
    current_candidate: str,
    delta_paths: Sequence[str],
) -> dict[str, object]:
    config = ci_change_owned.read_config(
        TRUSTED_ROOT / ".config/ci-test-targets.toml"
    )
    base_metadata = ci_change_owned.metadata_at(
        repository_path, prior_candidate
    )
    candidate_metadata = ci_change_owned.metadata_at(
        repository_path, current_candidate
    )
    return ci_change_owned.targeted_rebase_frontier(
        delta_paths,
        base_metadata=base_metadata,
        candidate_metadata=candidate_metadata,
        config=config,
    )


def _frontier_flags(
    packages: Sequence[str], owner_jobs: Sequence[Mapping[str, str]]
) -> dict[str, bool]:
    package_set = set(packages)
    owners = {
        (owner.get("workflow"), owner.get("job"))
        for owner in owner_jobs
    }
    return {
        "run_rust": bool(package_set) or ("ci.yml", "lint-rust") in owners,
        "run_script_unit": (
            ("ci.yml", "script-unit") in owners
            or "chelis-python" in package_set
        ),
        "run_integration": bool(package_set),
        "run_smt": "chelis-prove" in package_set,
        "run_backend": "chelis-backend-c" in package_set,
        "run_diagnostic": bool(
            {"chelis-types", "chelis-compiler-api"} & package_set
        ),
        "run_hull": (
            ("conformance.yml", "conformance") in owners
            or bool({"chelis-cli", "chelis-conformance"} & package_set)
        ),
    }


def _candidate_parents(repository_path: Path, candidate_sha: str) -> list[str]:
    raw = _git(
        repository_path,
        "show",
        "-s",
        "--format=%P",
        candidate_sha,
    ).stdout.decode().strip()
    parents = raw.split()
    if len(parents) != 2 or any(not SHA.fullmatch(parent) for parent in parents):
        raise ReuseError("current synthetic candidate must have exactly two parents")
    return parents


def _full(
    reason: str,
    *,
    before_sha: str | None = None,
    head_sha: str | None = None,
    base_sha: str | None = None,
    delta_paths: Sequence[str] = (),
    prior_receipt_run_id: int | None = None,
    ci_contract_changed: bool = False,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "lane": "full",
        "reason": reason,
        "before_sha": before_sha,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "delta_paths": list(delta_paths),
        "prior_receipt_run_id": prior_receipt_run_id,
        "ci_contract_changed": ci_contract_changed,
    }


def _ordinary(
    reason: str,
    *,
    before_sha: str | None = None,
    head_sha: str | None = None,
    base_sha: str | None = None,
    delta_paths: Sequence[str] = (),
    ci_contract_changed: bool = False,
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "lane": "ordinary",
        "reason": reason,
        "before_sha": before_sha,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "delta_paths": list(delta_paths),
        "prior_receipt_run_id": None,
        "ci_contract_changed": ci_contract_changed,
    }


def _is_linear_content_update(
    repository_path: Path, before_sha: str, head_sha: str
) -> bool:
    if not _is_ancestor(repository_path, before_sha, head_sha):
        return False
    rows = _git(
        repository_path,
        "rev-list",
        "--parents",
        f"{before_sha}..{head_sha}",
    ).stdout.decode().splitlines()
    return bool(rows) and all(len(row.split()) == 2 for row in rows)


def _timeline_events(timeline: Sequence[object]) -> list[Mapping[str, Any]]:
    events: list[Mapping[str, Any]] = []
    for item in timeline:
        if isinstance(item, Mapping):
            events.append(item)
            continue
        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            events.extend(
                _mapping(event, "pull request timeline event")
                for event in item
            )
            continue
        raise ReuseError("pull request timeline contains an invalid page")
    return events


def _retargeted_after(
    timeline: Sequence[object], receipt_created_at: str
) -> bool:
    boundary = _timestamp(receipt_created_at, "receipt workflow created_at")
    for event in _timeline_events(timeline):
        if event.get("event") != "base_ref_changed":
            continue
        if _timestamp(event.get("created_at"), "base retarget created_at") >= boundary:
            return True
    return False


def evaluate_rebase(
    *,
    repository_path: Path,
    repository: str,
    event: Mapping[str, Any],
    current_pr: Mapping[str, Any],
    candidate_sha: str,
    prior_receipt: Mapping[str, Any] | None,
    receipt_created_at: str | None,
    timeline: Sequence[object],
) -> dict[str, object]:
    """Return ``ordinary``, ``full``, ``docs``, or ``targeted``."""
    repository_path = Path(repository_path)
    if not repository_path.is_dir():
        raise ReuseError("repository_path must be an existing directory")
    if not REPOSITORY.fullmatch(repository):
        raise ReuseError("repository must be an owner/name identifier")
    if event.get("action") != "synchronize":
        return _ordinary("event is not a pull-request synchronize")
    before_sha = _sha(event.get("before"), "event before")
    head_sha = _sha(event.get("after"), "event after")
    pr_number = _positive_integer(current_pr.get("number"), "pull request number")
    current_head = _mapping(current_pr.get("head"), "current pull request head")
    current_base = _mapping(current_pr.get("base"), "current pull request base")
    if _sha(current_head.get("sha"), "current pull request head sha") != head_sha:
        return _full(
            "current pull request head no longer matches the event",
            before_sha=before_sha,
            head_sha=head_sha,
        )
    _sha(current_base.get("sha"), "current pull request base sha")
    base_ref = _string(current_base.get("ref"), "current pull request base ref")
    for side, data in (("head", current_head), ("base", current_base)):
        source = _mapping(data.get("repo"), f"current pull request {side} repo")
        if source.get("full_name") != repository:
            return _full(
                f"pull request {side} belongs to another repository",
                before_sha=before_sha,
                head_sha=head_sha,
            )
    candidate_sha = _sha(candidate_sha, "current synthetic candidate sha")
    for commit in (before_sha, head_sha, candidate_sha):
        _ensure_commit(repository_path, commit)
    parents = _candidate_parents(repository_path, candidate_sha)
    if parents[1] != head_sha:
        return _full(
            "current synthetic candidate head parent does not match the PR event",
            before_sha=before_sha,
            head_sha=head_sha,
        )
    base_sha = parents[0]
    _ensure_commit(repository_path, base_sha)
    delta_paths = _changed_paths(repository_path, before_sha, head_sha)
    if not delta_paths:
        return _full(
            "synchronize event changed no candidate-tree paths",
            before_sha=before_sha,
            head_sha=head_sha,
            base_sha=base_sha,
        )
    delta_contract_safe, _ = candidate_receipt.reuse_eligibility(delta_paths)
    delta_contract_changed = not delta_contract_safe
    if _is_linear_content_update(repository_path, before_sha, head_sha):
        return _ordinary(
            "ordinary linear content update",
            before_sha=before_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            delta_paths=delta_paths,
            ci_contract_changed=delta_contract_changed,
        )

    def fallback(
        reason: str,
        *,
        prior_receipt_run_id: int | None = None,
    ) -> dict[str, object]:
        return _full(
            reason,
            before_sha=before_sha,
            head_sha=head_sha,
            base_sha=base_sha,
            delta_paths=delta_paths,
            prior_receipt_run_id=prior_receipt_run_id,
            ci_contract_changed=delta_contract_changed,
        )

    if prior_receipt is None or receipt_created_at is None:
        return fallback(
            "prior trusted receipt unavailable",
        )
    prior_run_id = _positive_integer(
        prior_receipt.get("receipt_workflow_run_id"),
        "prior receipt workflow run id",
    )
    if prior_receipt.get("schema") != RECEIPT_SCHEMA:
        return fallback(
            "prior trusted receipt has an unsupported schema",
            prior_receipt_run_id=prior_run_id,
        )
    if (
        prior_receipt.get("repository") != repository
        or prior_receipt.get("pr_number") != pr_number
        or prior_receipt.get("head_sha") != before_sha
    ):
        return fallback(
            "prior trusted receipt does not bind the previous PR head",
            prior_receipt_run_id=prior_run_id,
        )
    if prior_receipt.get("reuse_eligible") is not True:
        return fallback(
            "prior trusted receipt is not reusable",
            prior_receipt_run_id=prior_run_id,
        )
    if prior_receipt.get("base_ref") != base_ref:
        return fallback(
            "target branch changed since the prior receipt",
            prior_receipt_run_id=prior_run_id,
        )
    if _retargeted_after(timeline, receipt_created_at):
        return fallback(
            "pull request was retargeted after the prior receipt",
            prior_receipt_run_id=prior_run_id,
        )

    prior_base = _sha(prior_receipt.get("base_sha"), "prior receipt base sha")
    prior_head = _sha(prior_receipt.get("head_sha"), "prior receipt head sha")
    prior_candidate = _sha(
        prior_receipt.get("candidate_sha"), "prior receipt candidate sha"
    )
    for commit in (
        before_sha,
        head_sha,
        base_sha,
        prior_base,
        prior_head,
        prior_candidate,
        candidate_sha,
    ):
        _ensure_commit(repository_path, commit)

    rebase_base = _merge_base(repository_path, base_sha, head_sha)
    if prior_base == rebase_base or not _is_ancestor(
        repository_path, prior_base, rebase_base
    ):
        return fallback(
            "update is not a forward base-changing rebase",
            prior_receipt_run_id=prior_run_id,
        )
    if not _is_ancestor(repository_path, rebase_base, base_sha):
        return fallback(
            "rebased base is not contained by the current target",
            prior_receipt_run_id=prior_run_id,
        )

    delta_paths = _changed_paths(repository_path, prior_candidate, candidate_sha)
    if not delta_paths:
        return fallback(
            "trusted synthetic candidate tree did not change",
            prior_receipt_run_id=prior_run_id,
        )
    delta_contract_safe, _ = candidate_receipt.reuse_eligibility(delta_paths)
    delta_contract_changed = not delta_contract_safe
    current = identity.build_identity(
        repository_path=repository_path,
        repository=repository,
        workflow_file="ci.yml",
        run_id=1,
        run_attempt=1,
        pr_number=pr_number,
        head_sha=head_sha,
        base_ref=base_ref,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
    )
    if delta_contract_changed:
        return fallback(
            "rebase changed CI policy; full CI follows contract preflight",
            prior_receipt_run_id=prior_run_id,
        )

    prior_paths = prior_receipt.get("changed_paths")
    if (
        not isinstance(prior_paths, list)
        or not prior_paths
        or any(not isinstance(path, str) or not path for path in prior_paths)
        or len(prior_paths) != len(set(prior_paths))
    ):
        return fallback(
            "prior trusted receipt has malformed changed paths",
            prior_receipt_run_id=prior_run_id,
        )
    try:
        frontier = _interaction_frontier(
            repository_path,
            prior_candidate=prior_candidate,
            current_candidate=candidate_sha,
            delta_paths=delta_paths,
        )
    except (OSError, ReuseError, ValueError, subprocess.CalledProcessError):
        return fallback(
            "trusted test mapping could not classify the rebase delta",
            prior_receipt_run_id=prior_run_id,
        )
    unsafe_paths = frontier["unsafe_paths"]
    if unsafe_paths:
        return fallback(
            "rebase delta has unmapped, ambiguous, or unsupported owner paths",
            prior_receipt_run_id=prior_run_id,
        )
    base_delta_paths = _changed_paths(repository_path, prior_base, base_sha)
    overlap_paths = sorted(
        set(base_delta_paths)
        & (set(prior_paths) | set(current["changed_paths"]))
    )
    exact_patch_fields = (
        "patch_id",
        "patch_digest",
        "changed_paths",
        "changed_paths_sha256",
    )
    patch_identity_unchanged = all(
        current[key] == prior_receipt.get(key) for key in exact_patch_fields
    )
    prior_patch_docs_only = is_docs_only(prior_paths)
    current_patch_docs_only = is_docs_only(current["changed_paths"])
    lane = "docs" if is_docs_only(delta_paths) else "targeted"
    reason = (
        "trusted docs-only synthetic-candidate delta"
        if lane == "docs"
        else "trusted interaction-frontier rebase"
    )
    frontier_packages = frontier["packages"] if lane == "targeted" else []
    frontier_owner_jobs = frontier["owner_jobs"] if lane == "targeted" else []
    flags = _frontier_flags(frontier_packages, frontier_owner_jobs)
    return {
        "schema": SCHEMA,
        "lane": lane,
        "reason": reason,
        "before_sha": before_sha,
        "head_sha": head_sha,
        "base_sha": base_sha,
        "rebase_base_sha": rebase_base,
        "validation_base_sha": prior_candidate,
        "delta_paths": delta_paths,
        "base_delta_paths": base_delta_paths,
        "overlap_paths": overlap_paths,
        "prior_patch_docs_only": prior_patch_docs_only,
        "current_patch_docs_only": current_patch_docs_only,
        "frontier_packages": frontier_packages,
        "frontier_owner_jobs": frontier_owner_jobs,
        "patch_identity_unchanged": patch_identity_unchanged,
        "standing_review_required": bool(
            overlap_paths or not patch_identity_unchanged
        ),
        "prior_receipt_run_id": prior_run_id,
        "ci_contract_changed": False,
        **flags,
    }


def _paged(api: Api, endpoint: str, key: str) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for page in range(1, 21):
        separator = "&" if "?" in endpoint else "?"
        payload = _mapping(
            api(f"{endpoint}{separator}per_page=100&page={page}"),
            f"{key} response",
        )
        rows = payload.get(key)
        if not isinstance(rows, list):
            raise ReuseError(f"{key} response had no list")
        result.extend(_mapping(row, f"{key} row") for row in rows)
        if len(rows) < 100:
            return result
    raise ReuseError(f"{key} response exceeded 20 pages")


def _receipt_archive(raw: bytes) -> Mapping[str, Any]:
    try:
        with ZipFile(BytesIO(raw)) as archive:
            files = [item for item in archive.infolist() if not item.is_dir()]
            if len(files) != 1 or files[0].filename != RECEIPT_FILENAME:
                raise ReuseError(
                    f"receipt archive must contain exactly {RECEIPT_FILENAME}"
                )
            payload = json.loads(archive.read(files[0]))
    except (BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReuseError("receipt artifact was not a readable zip JSON") from error
    return _mapping(payload, "candidate receipt")


def find_prior_receipt(
    *,
    repository: str,
    before_sha: str,
    api: Api,
    download_artifact: ArtifactDownloader,
) -> PriorReceipt | None:
    before_sha = _sha(before_sha, "previous head sha")
    repository_info = _mapping(api(f"repos/{repository}"), "repository")
    repository_id = _positive_integer(repository_info.get("id"), "repository id")
    default_branch = _string(
        repository_info.get("default_branch"), "repository default branch"
    )
    name = RECEIPT_PREFIX + before_sha
    artifacts = _paged(
        api,
        f"repos/{repository}/actions/artifacts?name={name}",
        "artifacts",
    )
    candidates = sorted(
        (
            artifact
            for artifact in artifacts
            if artifact.get("name") == name and artifact.get("expired") is not True
        ),
        key=lambda artifact: _positive_integer(
            artifact.get("id"), "receipt artifact id"
        ),
        reverse=True,
    )
    for artifact in candidates:
        artifact_id = _positive_integer(
            artifact.get("id"), "receipt artifact id"
        )
        workflow = _mapping(
            artifact.get("workflow_run"), "receipt artifact workflow run"
        )
        run_id = _positive_integer(
            workflow.get("id"), "receipt artifact workflow run id"
        )
        if workflow.get("repository_id") != repository_id:
            continue
        run = _mapping(
            api(f"repos/{repository}/actions/runs/{run_id}"),
            "receipt workflow run",
        )
        if (
            run.get("path") != RECEIPT_WORKFLOW
            or run.get("event") != "workflow_run"
            or run.get("conclusion") not in PASSING_CONCLUSIONS
            or run.get("head_branch") != default_branch
        ):
            continue
        head_repository = _mapping(
            run.get("head_repository"), "receipt workflow head repository"
        )
        if head_repository.get("full_name") != repository:
            continue
        payload = _receipt_archive(download_artifact(artifact_id))
        if (
            payload.get("repository") != repository
            or payload.get("head_sha") != before_sha
            or payload.get("receipt_workflow_run_id") != run_id
        ):
            continue
        created_at = _string(run.get("created_at"), "receipt workflow created_at")
        _timestamp(created_at, "receipt workflow created_at")
        return PriorReceipt(payload, artifact_id, run_id, created_at)
    return None


def _gh_api_json(endpoint: str) -> object:
    completed = subprocess.run(
        ["gh", "api", endpoint],
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise ReuseError(f"GitHub API request failed: {detail or endpoint}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReuseError("GitHub API response was not JSON") from error


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
        detail = completed.stderr.decode(errors="replace").strip()
        raise ReuseError(
            f"candidate receipt download failed: {detail or artifact_id}"
        )
    return completed.stdout


def write_decision(
    decision: Mapping[str, Any],
    *,
    output: Path,
    github_output: Path | None = None,
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    if github_output is not None:
        reason = _string(decision.get("reason"), "decision reason")
        contract_changed = decision.get("ci_contract_changed")
        if type(contract_changed) is not bool:
            raise ReuseError("decision ci_contract_changed must be a boolean")
        if any(character in reason for character in "\r\n"):
            raise ReuseError("decision reason cannot contain a newline")
        flags = {
            key: decision.get(key, False)
            for key in (
                "run_rust",
                "run_script_unit",
                "run_integration",
                "run_smt",
                "run_backend",
                "run_diagnostic",
                "run_hull",
            )
        }
        if any(type(value) is not bool for value in flags.values()):
            raise ReuseError("decision frontier flags must be booleans")
        packages = decision.get("frontier_packages", [])
        if (
            not isinstance(packages, list)
            or any(
                not isinstance(package, str)
                or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", package) is None
                for package in packages
            )
            or len(packages) != len(set(packages))
        ):
            raise ReuseError("decision frontier packages are malformed")
        Path(github_output).write_text(
            "\n".join(
                [
                    f"rebase_lane={decision['lane']}",
                    f"rebase_reason={reason}",
                    "rebase_before="
                    + str(
                        decision.get("validation_base_sha")
                        or decision.get("before_sha")
                        or ""
                    ),
                    "rebase_contract_changed="
                    + ("true" if contract_changed else "false"),
                    "rebase_packages=" + ",".join(packages),
                    *[
                        f"rebase_{key}={'true' if value else 'false'}"
                        for key, value in flags.items()
                    ],
                ]
            )
            + "\n"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repository-path", type=Path, required=True)
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--current-pr", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--github-output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        event = json.loads(arguments.event.read_text())
        current_pr = json.loads(arguments.current_pr.read_text())
        timeline = json.loads(arguments.timeline.read_text())
        if not isinstance(event, dict) or not isinstance(current_pr, dict):
            raise ReuseError("event and current pull request must be JSON objects")
        if not isinstance(timeline, list):
            raise ReuseError("pull request timeline must be a JSON list")
        before = event.get("before")
        prior = (
            find_prior_receipt(
                repository=arguments.repository,
                before_sha=_sha(before, "event before"),
                api=_gh_api_json,
                download_artifact=lambda artifact_id: _gh_download(
                    arguments.repository, artifact_id
                ),
            )
            if event.get("action") == "synchronize"
            else None
        )
        decision = evaluate_rebase(
            repository_path=arguments.repository_path,
            repository=arguments.repository,
            event=event,
            current_pr=current_pr,
            candidate_sha=arguments.candidate_sha,
            prior_receipt=None if prior is None else prior.payload,
            receipt_created_at=None if prior is None else prior.created_at,
            timeline=timeline,
        )
    except (OSError, json.JSONDecodeError, ReuseError, identity.IdentityError) as error:
        print(f"REBASE REUSE: FAIL: {error}", file=sys.stderr)
        return 2
    write_decision(
        decision,
        output=arguments.output,
        github_output=arguments.github_output,
    )
    print(f"REBASE REUSE: {decision['lane'].upper()}: {decision['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
