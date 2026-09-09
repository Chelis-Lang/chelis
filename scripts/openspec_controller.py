#!/usr/bin/env python3
"""Turn an ordinary branch push into an internal OpenSpec pull request.

A contributor pushes a branch. Nothing else. This controller classifies the
pushed commit, and if every changed path is an OpenSpec document it opens --
or reuses -- one internal pull request for it. The merge worker then takes
over.

THE TRUST PROBLEM, AND THE SHAPE OF THE ANSWER

A `push` event runs the workflow file from the pushed branch. That workflow
is written by whoever pushed, so it must never hold a token, and nothing it
says can be believed. The signal workflow therefore declares no permissions
and does no work: it exists only so that a `workflow_run` event fires.

`workflow_run` runs the DEFAULT BRANCH's copy of the controller workflow.
That is the trust boundary. This script runs there, and re-derives every
fact it acts on from the API:

- which event produced the run (must be `push`),
- which repository the branch is on (must be this one),
- which branch and which commit,
- whether that branch still points at that commit,
- and whether the diff is confined to OpenSpec documents, decided by
  default-branch code over `git diff --raw` metadata.

A rewritten signal workflow can refuse to run, which stops autoland for that
push. That is its entire influence. It cannot grant anything, because the
controller reads nothing it produced: no artifacts, no outputs, not even its
conclusion.

WHY THE PULL REQUEST NEEDS ITS OWN CREDENTIAL

A pull request created with `GITHUB_TOKEN` does not raise a `pull_request`
event, so it does not start the workflows whose jobs are this repository's
required status checks. Measured against the repository's own workflow
files: `conformance.yml` runs only on `push: [main]` and `pull_request`,
and `changelog.yml` on the default branch has only `pull_request`, so
`Hull Conformance Gate (Linux)` and `Changelog` can never appear. Since
protection requires all nine contexts, such a pull request could never go
green.

So the create call -- and only the create call -- uses a short-lived
installation token, which the workflow mints from the GitHub App this
repository already configures. It is scoped to this repository and to
`Pull requests: write` with `Contents: read` -- the second is what lets the
endpoint resolve the head and base refs -- and it is revoked when the job
ends; nothing is stored. Reads keep the ambient token, and the workflow grants that token
`pull-requests: read` rather than `write`, so the default credential cannot
open a pull request even by mistake.

There is no fallback. If the App is not configured, or its installation
does not grant pull-request write on this repository, this reports the push
blocked and writes nothing. Falling back would produce a pull request that
can never merge and would hide the misconfiguration behind something that
looks like it worked.

Exit status is `0` when a pull request is open for the commit, `1` when the
push was refused, and `2` for an operational failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

QUEUED = 0
BLOCKED = 1
FAILED = 2

API_TIMEOUT_SECONDS = 120
CLASSIFY_TIMEOUT_SECONDS = 300

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
# An ordinary branch name. Restrictive on purpose: this value is
# interpolated into an API path, and a leading `-` would also read as an
# option to `gh`.
BRANCH_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

CLASSIFIER = "openspec_acceptance.py"

# The workflow FILE whose runs this controller reacts to. The `workflows:`
# filter in the controller workflow matches on a run's display NAME, and a
# branch can add a second file declaring the same name. The name is not
# provenance; the path is.
SIGNAL_WORKFLOW = ".github/workflows/openspec-autoland-signal.yml"

# An ordinary branch name, matching `openspec_merge.require_branch_ref`.
# This value is interpolated into an API path and passed to `gh`, where a
# leading `-` would read as an option.
MAX_BRANCH_LENGTH = 255

# The environment variable carrying the credential that opens the pull
# request. The workflow mints it per run from the GitHub App this repository
# already configures, scoped to this repository and to pull-request write
# alone, and the action revokes it when the job ends. Nothing is stored.
#
# There is deliberately no fallback to `GITHUB_TOKEN`: a pull request opened
# with it raises no `pull_request` event, so the required checks never run
# and the pull request can never go green. Falling back would produce
# exactly the stalled pull request this design exists to avoid, and would
# hide the misconfiguration behind something that looks like it worked.
SUBMISSION_TOKEN_ENV = "OPENSPEC_SUBMISSION_TOKEN"

MISSING_TOKEN_ADVICE = f"""no submission credential reached this step, so no pull
            request was opened. The workflow mints one per run from the
            `chelis-openspec` GitHub App, which exists for this mechanism
            alone. Check, in order:
              1. `vars.OPENSPEC_APP_ID` and `secrets.OPENSPEC_APP_PRIVATE_KEY`
                 are set for this repository. These are NOT the shared
                 `CI_APP_*` credentials: that App's installation grants no
                 pull-request write, and widening it would give every
                 workflow holding its key the ability to open pull requests.
              2. The App's installation covers this repository and grants
                 "Pull requests: write" and "Contents: read" on it. Those
                 are the two `POST /repos/{{owner}}/{{repo}}/pulls` needs --
                 the second to resolve the head and base refs -- and the
                 mint step requests nothing else.
              3. The mint step ran before this one and produced a token.
                 A mint failure is reported by that step, not this one:
                 HTTP 422 "The permissions requested are not granted to
                 this installation" means item 2 is unsatisfied.
            The credential must not be GITHUB_TOKEN: a pull request opened
            with that token starts none of the required checks. `{SUBMISSION_TOKEN_ENV}`
            is the variable this step reads."""

# The contexts branch protection required for `main` when this was measured.
# Used ONLY by the test that proves the credential prerequisite is real; the
# merge worker discovers the live set from the API and never reads this.
REQUIRED_CONTEXTS_FOR_EVIDENCE = (
    "Lint and Unit Tests (Linux)",
    "Integration Tests (Linux)",
    "macOS Smoke",
    "Backend Sanitizers",
    "SMT Feature Build (Linux)",
    "Docs",
    "No AI authorship markers",
    "Hull Conformance Gate (Linux)",
    "Changelog",
)

PULL_BODY = """Automated OpenSpec document submission.

This pull request was opened by `openspec-autoland` from a push to
`{branch}` at `{head}`. Every changed path is an OpenSpec document; the
boundary was decided by default-branch code over the diff metadata, not by
anything the branch supplied.

Structural validation proves schema validity only. It does not prove that
the wording is correct, wanted, or consistent with `spec/**`. Where an
OpenSpec artifact contradicts an owning authority, the owning authority
controls and the artifact is corrected.
"""


class ControllerError(RuntimeError):
    """An operational failure: the controller could not run its own steps."""


class Refused(RuntimeError):
    """The push is not eligible; nothing was written."""


@dataclass(frozen=True)
class PushSignal:
    """The identity of one push, as the API reports it."""

    branch: str
    head: str
    repository: str


def _run(
    command: Sequence[str],
    repository_path: Path,
    runner: Callable | None,
    timeout: int = API_TIMEOUT_SECONDS,
    token: str | None = None,
):
    """Run one command, optionally under a specific credential.

    `token` replaces `GH_TOKEN` for this call only. Every other call keeps
    the ambient token, so the submission credential is never present in a
    process that is merely reading.
    """
    execute = runner or subprocess.run
    environment = {**os.environ, "GH_TOKEN": token} if token else dict(os.environ)
    try:
        return execute(
            list(command),
            cwd=str(repository_path),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=environment,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise ControllerError(f"cannot run {command[0]}: {error}") from error


def api_json(
    path: str, repository_path: Path, runner: Callable | None
) -> object:
    """Read one API endpoint, or raise."""
    completed = _run(["gh", "api", path], repository_path, runner)
    if getattr(completed, "returncode", None) != 0:
        message = (
            getattr(completed, "stderr", "") or getattr(completed, "stdout", "") or ""
        ).strip()
        raise ControllerError(f"gh api {path} failed: {message or 'no output'}")
    try:
        return json.loads(getattr(completed, "stdout", "") or "")
    except json.JSONDecodeError as error:
        raise ControllerError(f"unreadable API response for {path}") from error


def require_text(payload: dict, field: str) -> str:
    """A string field, or an operational failure.

    A payload whose shape is wrong is a broken API contract, not a push to
    refuse, and reporting it as a refusal would hide it.
    """
    value = payload.get(field)
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ControllerError(f"{field} is not a string: {type(value).__name__}")
    return value


def require_branch(name: str) -> str:
    """Reject every branch spelling that is not an ordinary ref."""
    if not BRANCH_REF.fullmatch(name):
        raise Refused(f"the branch name is not an ordinary ref: {name!r}")
    if ".." in name or name.endswith("/") or len(name) > MAX_BRANCH_LENGTH:
        raise Refused(f"the branch name is not an ordinary ref: {name!r}")
    return name


def read_signal(
    repository: str,
    run_id: int,
    default_branch: str,
    repository_path: Path,
    runner: Callable | None,
) -> PushSignal:
    """Derive the push identity from the API and refuse anything else.

    Nothing here reads the signal run's outputs, artifacts, or conclusion.
    A run that failed, or that uploaded a lie, still yields exactly the same
    three facts, because those come from GitHub's record of the event.
    """
    payload = api_json(
        f"repos/{repository}/actions/runs/{run_id}", repository_path, runner
    )
    if not isinstance(payload, dict):
        raise ControllerError("workflow run response was not an object")

    event = require_text(payload, "event")
    if event != "push":
        raise Refused(
            f"the signal run was a {event or 'unknown'} event, not a push"
        )

    path = require_text(payload, "path")
    if path != SIGNAL_WORKFLOW:
        raise Refused(
            f"the run came from workflow file {path or 'unknown'}, not from "
            f"{SIGNAL_WORKFLOW}"
        )

    head_repository_payload = payload.get("head_repository")
    if head_repository_payload is not None and not isinstance(
        head_repository_payload, dict
    ):
        raise ControllerError("head_repository is not an object")
    head_repository = (head_repository_payload or {}).get("full_name")
    if head_repository != repository:
        raise Refused(
            f"the branch is on {head_repository or 'an unknown repository'}, "
            f"not on {repository}"
        )

    branch = require_branch(require_text(payload, "head_branch"))
    if branch == default_branch:
        raise Refused(
            f"the push was to the default branch ({default_branch}); autoland "
            "acts only on other branches"
        )

    head = require_text(payload, "head_sha")
    if not FULL_SHA.fullmatch(head):
        raise Refused(f"the run does not name a commit: {head!r}")

    return PushSignal(branch=branch, head=head, repository=repository)


def branch_head(
    repository: str, branch: str, repository_path: Path, runner: Callable | None
) -> str | None:
    """The commit a branch points at now, or `None` if it is gone."""
    completed = _run(
        ["gh", "api", f"repos/{repository}/branches/{branch}"],
        repository_path,
        runner,
    )
    if getattr(completed, "returncode", None) != 0:
        return None
    try:
        payload = json.loads(getattr(completed, "stdout", "") or "")
    except json.JSONDecodeError as error:
        raise ControllerError(f"unreadable branch response for {branch}") from error
    return str(((payload or {}).get("commit") or {}).get("sha") or "") or None


def require_current(signal: PushSignal, repository_path: Path, runner) -> None:
    """Refuse unless the branch still points at the commit we were told about.

    Events arrive late, out of order, and more than once. Acting on a
    superseded commit would open a pull request for something nobody is
    waiting on, and the newer push has its own signal run that will handle
    itself.
    """
    current = branch_head(signal.repository, signal.branch, repository_path, runner)
    if current is None:
        raise Refused(
            f"branch {signal.branch} no longer exists, so there is nothing to open"
        )
    if current != signal.head:
        raise Refused(
            f"branch {signal.branch} moved to {current} since this push at "
            f"{signal.head}; the newer push decides"
        )


def classify(
    signal: PushSignal,
    base: str,
    repository_path: Path,
    runner: Callable | None,
) -> str:
    """Run the default-branch classifier over the exact pushed commit."""
    script = str(Path(__file__).resolve().parent / CLASSIFIER)
    completed = _run(
        [
            sys.executable,
            script,
            "--repo",
            str(repository_path),
            "--base",
            base,
            "--head",
            signal.head,
            "--require-identical-governance",
            "--require-auto",
        ],
        repository_path,
        runner,
        timeout=CLASSIFY_TIMEOUT_SECONDS,
    )
    output = (
        (getattr(completed, "stdout", "") or "")
        + (getattr(completed, "stderr", "") or "")
    ).strip()
    code = getattr(completed, "returncode", None)
    if code == 0:
        return output
    if code == 1:
        raise Refused(f"this push is not an OpenSpec document change\n{output}")
    raise ControllerError(f"the classifier failed with exit code {code}\n{output}")


def find_existing(
    repository: str, branch: str, repository_path: Path, runner: Callable | None
) -> dict | None:
    """An open pull request already carrying this branch, if there is one."""
    completed = _run(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repository,
            "--head",
            branch,
            "--state",
            "open",
            "--limit",
            "1",
            "--json",
            "number,url,headRefOid",
        ],
        repository_path,
        runner,
    )
    if getattr(completed, "returncode", None) != 0:
        raise ControllerError("cannot list existing pull requests")
    text = (getattr(completed, "stdout", "") or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ControllerError("unreadable pull request list") from error
    return payload[0] if payload else None


def create_pull_request(
    signal: PushSignal,
    base_branch: str,
    submission_token: str,
    repository_path: Path,
    runner: Callable | None,
) -> str:
    """Open the internal pull request and return its URL.

    This is the ONLY call that uses the submission credential, and it is
    the only write this controller performs. The REST endpoint is used
    rather than `gh pr create` because `POST /repos/{owner}/{repo}/pulls`
    requires exactly one fine-grained permission -- `Pull requests: write`
    -- while the CLI additionally reads repository and branch metadata.
    Fewer permissions to grant, and a scope that can be stated exactly.
    """
    title = f"docs(openspec): {signal.branch}"
    completed = _run(
        [
            "gh",
            "api",
            "--method",
            "POST",
            f"repos/{signal.repository}/pulls",
            "-f",
            f"title={title}",
            "-f",
            f"head={signal.branch}",
            "-f",
            f"base={base_branch}",
            "-f",
            f"body={PULL_BODY.format(branch=signal.branch, head=signal.head)}",
        ],
        repository_path,
        runner,
        token=submission_token,
    )
    if getattr(completed, "returncode", None) != 0:
        # BOTH streams. `gh api` puts its one-line status on stderr and the
        # response body -- which carries the `errors` array naming the field
        # GitHub objected to -- on stdout. Reporting only the status reduces
        # a 422 to "Validation Failed" and throws away the reason, which
        # then costs a round trip to a hosted runner to rediscover.
        parts = [
            (getattr(completed, "stderr", "") or "").strip(),
            (getattr(completed, "stdout", "") or "").strip(),
        ]
        message = " ".join(part for part in parts if part) or "no output"
        raise ControllerError(f"cannot open the pull request: {message}")
    try:
        payload = json.loads(getattr(completed, "stdout", "") or "")
    except json.JSONDecodeError as error:
        raise ControllerError("unreadable pull request response") from error
    return str((payload or {}).get("html_url") or "")


def run(
    repository: str,
    run_id: int,
    default_branch: str = "main",
    repository_path: Path = Path("."),
    runner: Callable | None = None,
    submission_token: str | None = None,
    default_token: str | None = None,
) -> int:
    """Classify one push and open or reuse its pull request."""
    print("openspec-autoland controller")
    # Absolute from here down, and this is load-bearing rather than tidy.
    # The workflow passes `base`, the directory it checked the default
    # branch out to. Every command below runs with `cwd` set to this path,
    # and the classifier is additionally TOLD the path with `--repo`. While
    # it stayed relative, the child resolved it a second time against the
    # working directory it had already been given and looked for
    # `base/base`, which is how the first hosted run failed:
    # `cannot run Git: [Errno 2] No such file or directory: 'base'`.
    repository_path = Path(repository_path).resolve()
    try:
        signal = read_signal(
            repository, run_id, default_branch, repository_path, runner
        )
        print(f"  push      {signal.branch} at {signal.head[:7]}")
        require_current(signal, repository_path, runner)

        base = branch_head(repository, default_branch, repository_path, runner)
        if base is None or not FULL_SHA.fullmatch(base):
            raise ControllerError(
                f"cannot read the head of the default branch ({default_branch})"
            )

        verdict = classify(signal, base, repository_path, runner)
        print(f"  classify  {verdict.splitlines()[0] if verdict else 'auto'}")

        existing = find_existing(repository, signal.branch, repository_path, runner)
        if existing is not None:
            url = str(existing.get("url") or "")
            print(f"  QUEUED    reused pull request #{existing.get('number')}  {url}")
            return QUEUED

        # The credential is checked here, not at the top: a push that is not
        # a document change must report the boundary, not a secret it never
        # needed. This is still before any write.
        if not (submission_token or "").strip():
            raise Refused(MISSING_TOKEN_ADVICE)

        url = create_pull_request(
            signal, default_branch, submission_token, repository_path, runner
        )
        print(f"  QUEUED    opened  {url}")
        return QUEUED
    except Refused as error:
        print(f"  BLOCKED   {error}")
        return BLOCKED
    except ControllerError as error:
        print(f"  FAILED    {error}", file=sys.stderr)
        return FAILED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Open an internal pull request for a pushed OpenSpec branch."
    )
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument(
        "--run-id", required=True, type=int, help="the signal workflow run id"
    )
    parser.add_argument("--default-branch", default="main")
    parser.add_argument("--repository-path", default=".")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        print("openspec_controller: no GitHub token for reads", file=sys.stderr)
        return FAILED
    # Read deliberately without a fallback. An absent secret must surface as
    # a blocked, actionable report, never as a read token quietly opening a
    # pull request that can never merge.
    return run(
        repository=arguments.repository,
        run_id=arguments.run_id,
        default_branch=arguments.default_branch,
        repository_path=Path(arguments.repository_path),
        submission_token=os.environ.get(SUBMISSION_TOKEN_ENV),
    )


if __name__ == "__main__":
    sys.exit(main())
