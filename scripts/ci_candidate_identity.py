#!/usr/bin/env python3
"""Describe the exact synthetic candidate tested by one PR workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Sequence


SCHEMA = "chelis-ci-candidate-identity/v1"
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
WORKFLOW_FILE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml\Z")
REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*\Z")
INDEX_LINE = re.compile(
    rb"index [0-9a-f]+\.\.[0-9a-f]+(?P<mode> [0-7]{6})?(?P<newline>\n)?\Z"
)


class IdentityError(ValueError):
    """The requested identity is malformed or not the checked-out candidate."""


def _positive_integer(label: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise IdentityError(f"{label} must be a positive integer")
    return value


def _full_sha(label: str, value: str) -> str:
    if not SHA.fullmatch(value):
        raise IdentityError(f"{label} must be a lowercase 40-character SHA")
    return value


def _run(
    repository_path: Path,
    arguments: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    completed = subprocess.run(
        list(arguments),
        cwd=repository_path,
        input=input_bytes,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise IdentityError(
            f"{' '.join(arguments)} failed: {message or 'no error output'}"
        )
    return completed.stdout


def _git(repository_path: Path, *arguments: str) -> str:
    return _run(repository_path, ["git", *arguments]).decode().strip()


def _commit_parents(repository_path: Path, sha: str) -> list[str]:
    raw = _run(repository_path, ["git", "cat-file", "commit", sha])
    parents: list[str] = []
    for line in raw.splitlines():
        if not line:
            break
        if not line.startswith(b"parent "):
            continue
        try:
            parent = line.removeprefix(b"parent ").decode("ascii")
        except UnicodeDecodeError as error:
            raise IdentityError("candidate parent was not an ASCII SHA") from error
        parents.append(_full_sha("candidate parent", parent))
    return parents


def _changed_paths(
    repository_path: Path, base_sha: str, head_sha: str
) -> list[str]:
    raw = _run(
        repository_path,
        ["git", "diff", "--name-only", "-z", base_sha, head_sha],
    )
    paths = [
        item.decode("utf-8")
        for item in raw.split(b"\0")
        if item
    ]
    if not paths:
        raise IdentityError("the pull request patch has no changed paths")
    if len(paths) != len(set(paths)):
        raise IdentityError("the pull request patch repeats a changed path")
    return sorted(paths)


def _changed_paths_digest(paths: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _patch_id(repository_path: Path, base_sha: str, head_sha: str) -> str:
    patch = _run(
        repository_path,
        ["git", "diff", "--full-index", "--binary", base_sha, head_sha],
    )
    output = _run(
        repository_path,
        ["git", "patch-id", "--stable"],
        input_bytes=patch,
    ).decode()
    rows = [line.split() for line in output.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) < 1 or not SHA.fullmatch(rows[0][0]):
        raise IdentityError("git patch-id did not produce one stable patch identity")
    return rows[0][0]


def _patch_digest(repository_path: Path, base_sha: str, head_sha: str) -> str:
    patch = _run(
        repository_path,
        [
            "git",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--no-renames",
            "--binary",
            "--full-index",
            "--unified=0",
            base_sha,
            head_sha,
        ],
    )
    canonical = bytearray()
    for line in patch.splitlines(keepends=True):
        index = INDEX_LINE.fullmatch(line)
        if index is not None:
            canonical.extend(b"index <objects>")
            canonical.extend(index.group("mode") or b"")
            canonical.extend(index.group("newline") or b"")
        elif line.startswith(b"@@ "):
            canonical.extend(b"@@")
            if line.endswith(b"\n"):
                canonical.extend(b"\n")
        else:
            canonical.extend(line)
    if not canonical:
        raise IdentityError("the pull request patch produced no exact diff")
    return hashlib.sha256(canonical).hexdigest()


def build_identity(
    *,
    repository_path: Path,
    repository: str,
    workflow_file: str,
    run_id: int,
    run_attempt: int,
    pr_number: int,
    head_sha: str,
    base_ref: str,
    base_sha: str,
    candidate_sha: str,
) -> dict[str, object]:
    repository_path = Path(repository_path)
    if not repository_path.is_dir():
        raise IdentityError("repository_path must be an existing directory")
    if not REPOSITORY.fullmatch(repository):
        raise IdentityError("repository must be an owner/name identifier")
    if not WORKFLOW_FILE.fullmatch(workflow_file):
        raise IdentityError("workflow_file must be a bare workflow file name")
    if not REF.fullmatch(base_ref) or ".." in base_ref or base_ref.endswith("/"):
        raise IdentityError("base_ref must be an ordinary branch name")
    run_id = _positive_integer("run_id", run_id)
    run_attempt = _positive_integer("run_attempt", run_attempt)
    pr_number = _positive_integer("pr_number", pr_number)
    head_sha = _full_sha("head_sha", head_sha)
    base_sha = _full_sha("base_sha", base_sha)
    candidate_sha = _full_sha("candidate_sha", candidate_sha)

    resolved_candidate = _git(
        repository_path, "rev-parse", f"{candidate_sha}^{{commit}}"
    )
    if resolved_candidate != candidate_sha:
        raise IdentityError("candidate_sha did not resolve to itself")
    parents = _commit_parents(repository_path, candidate_sha)
    expected_parents = [base_sha, head_sha]
    if parents != expected_parents:
        raise IdentityError(
            "candidate parents do not match the event base and pull request head: "
            f"expected {expected_parents}, observed {parents}"
        )

    patch_base_sha = _git(
        repository_path, "merge-base", base_sha, head_sha
    )
    _full_sha("patch_base_sha", patch_base_sha)
    paths = _changed_paths(repository_path, patch_base_sha, head_sha)
    return {
        "schema": SCHEMA,
        "repository": repository,
        "event": "pull_request",
        "workflow_file": workflow_file,
        "workflow_run_id": run_id,
        "workflow_run_attempt": run_attempt,
        "pr_number": pr_number,
        "head_sha": head_sha,
        "base_ref": base_ref,
        "base_sha": base_sha,
        "patch_base_sha": patch_base_sha,
        "candidate_sha": candidate_sha,
        "candidate_parents": expected_parents,
        "patch_id": _patch_id(repository_path, patch_base_sha, head_sha),
        "patch_digest": _patch_digest(
            repository_path, patch_base_sha, head_sha
        ),
        "changed_paths": paths,
        "changed_paths_sha256": _changed_paths_digest(paths),
    }


def write_identity(payload: dict[str, object], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-path", type=Path, default=Path("."))
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-file", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--run-attempt", type=int, required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    candidate_sha = arguments.candidate_sha
    if candidate_sha is None:
        candidate_sha = _git(arguments.repository_path, "rev-parse", "HEAD")
    try:
        payload = build_identity(
            repository_path=arguments.repository_path,
            repository=arguments.repository,
            workflow_file=arguments.workflow_file,
            run_id=arguments.run_id,
            run_attempt=arguments.run_attempt,
            pr_number=arguments.pr_number,
            head_sha=arguments.head_sha,
            base_ref=arguments.base_ref,
            base_sha=arguments.base_sha,
            candidate_sha=candidate_sha,
        )
        write_identity(payload, arguments.output)
    except IdentityError as error:
        parser.error(str(error))
    print(
        f"recorded {arguments.workflow_file} candidate "
        f"{payload['candidate_sha']} for PR #{payload['pr_number']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
