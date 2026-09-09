#!/usr/bin/env python3
"""Merge one OpenSpec document pull request, or refuse and say why.

This is the only component that holds a write token. It runs from the base
revision under `pull_request_target`, never imports or executes anything
from the pull request, and merges exactly one commit.

WHAT IT DOES NOT DO

- It never enables auto-merge. Auto-merge is a standing grant on a mutable
  branch: it survives later pushes, so a decision made on one commit would
  authorize every commit after it.
- It never uses an administrative or protection-bypassing option.
- It never submits a review or an approval.
- It changes no repository setting.

Branch protection remains the final arbiter. This worker checks the same
conditions in advance so that it does not ask for a merge that should be
refused, but the API call it makes is an ordinary merge that protection is
free to reject. If protection refuses -- including for an approval rule
added later -- the worker reports the pull request as blocked and exits.

WHAT A VERDICT AUTHORIZES

A boundary verdict is computed over `base.sha...head.sha` for one base
BRANCH. It authorizes that triple, not a pull request number. A squash
merge lands `merge-base(base, head)...head`, so retargeting the base after
the verdict, or rewinding the base branch, changes what would land without
changing anything the classifier looked at. `pull_request_target` does not
fire on `edited`, so a retarget produces no new verdict either. The worker
is therefore given the authorized base ref and base sha and refuses any
base it did not decide about: a different ref outright, and a same-ref sha
that is not `ahead` of or `identical` to the authorized one.

WHY THE REST MERGE ENDPOINT RATHER THAN `gh pr merge`

`gh pr merge` documents a fallback: "If required checks have not yet
passed, auto-merge will be enabled." That fallback is exactly the standing
grant this design refuses, and it would be reached by a race rather than by
choice. `PUT /repos/{owner}/{repo}/pulls/{n}/merge` has no such path. It
takes `sha`, which binds the merge to one commit exactly as
`--match-head-commit` does, and returns a documented status for every
outcome: `409` when the head moved, `405` when protection says no.

SIX MEASURED FACTS THIS ENCODES

1. Check runs attach to the pull request HEAD sha, not the test-merge sha,
   so binding the merge to the head sha binds it to the commit that was
   actually tested.
2. One context appears many times on a single commit. `Changelog` was
   observed with three `success` runs and one `cancelled` run. Only the
   latest run for a context counts; `all()` and `any()` are both wrong.
3. The legacy combined-status endpoint reports `state: "pending"` with zero
   statuses for commits whose checks all passed, so it cannot be read as
   the answer. It is consulted only for contexts that post real statuses.
4. `GET .../branches/{branch}/protection` requires the `Administration`
   permission. A workflow `permissions:` block has no `administration` key,
   so GITHUB_TOKEN can never hold it and that endpoint is unreachable from
   Actions. `GET .../branches/{branch}` needs only `Contents: read` and
   carries the same `protection.required_status_checks.checks` array,
   app ids included. It is the primary source; the protection endpoint is a
   fallback for a caller that does hold admin.
5. A commit status carries no app id, and anyone with write access can post
   one. A context that protection pins to an app is therefore satisfiable
   only by a check run from that app -- never by a status, and never by a
   result whose app is unknown.
6. A check-run NAME is not provenance. Any Actions workflow can publish a
   job with any name. The strict validation requirement is pinned to a
   successful `pull_request` run of one workflow FILE on the exact head
   sha, which the governance-identity check proves is the base's file.

Exit status is `0` when the pull request was merged, `1` when it was
refused or is blocked, and `2` for an operational failure.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

MERGED = 0
REFUSED = 1
OPERATIONAL_FAILURE = 2

DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_POLL_SECONDS = 30
API_TIMEOUT_SECONDS = 120
PAGE_SIZE = 100
MAX_PAGES = 20

# `mergeable` is null while GitHub computes the test merge in the
# background. That is "not answered yet", not "no", so a single undecided
# read must not refuse a healthy pull request. The retries are bounded and
# an undecided flag after all of them still refuses.
MERGEABLE_ATTEMPTS = 4
MERGEABLE_POLL_SECONDS = 5

MERGE_METHOD = "squash"

# The conclusions branch protection accepts for a required check.
#
# `skipped` is here because it is the normal state for the exact class of
# pull request this worker exists for, not as a relaxation. `ci.yml` skips
# its heavy jobs on a docs-only diff through a job-level `if` (chelis#419),
# and a skipped job reports its context as skipped. Measured on real
# docs-only pull request #1634, which merged: six of the nine required
# contexts were `skipped`. Refusing them would refuse every eligible pull
# request while protection was willing to merge it.
#
# `neutral` is deliberately NOT here. Nothing in this repository produces
# it, so accepting it would widen the rule on speculation; refusing costs a
# human one click, and protection remains the final arbiter either way.
COMPLETED = "completed"
SUCCESS = "success"
PASSING_CONCLUSIONS = frozenset({SUCCESS, "skipped"})

# The event whose runs may satisfy a required workflow. The validation
# workflow declares `on: pull_request` only, so a run under any other event
# did not come from the file this worker is pinning.
WORKFLOW_EVENT = "pull_request"

# `compare/{a}...{b}` statuses that mean `b` only moved forward from `a`.
FORWARD_STATUSES = frozenset({"ahead", "identical"})

FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
# An ordinary branch name, restrictive on purpose: this value is
# interpolated into an API path, and a leading `-` would also read as an
# option to `gh`.
BRANCH_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
# A workflow file NAME, never a path. The Actions API takes the bare file
# name, so a value containing a separator is a mistake or an escape.
WORKFLOW_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.ya?ml$")

# What to tell a maintainer when the wait expires. `gh workflow run` is
# deliberately absent: `openspec-autoland.yml` declares no
# `workflow_dispatch` trigger, so that command fails outright.
RETRY_ADVICE = (
    "Re-run this decision after the checks finish by producing a new\n"
    "`pull_request_target` event for the pull request:\n"
    "  git commit --allow-empty -m 'retry autoland' && git push\n"
    "  # or: gh pr ready <n>, or close and reopen the pull request"
)


class MergeError(RuntimeError):
    """An operational failure, or an API call that did not succeed."""


@dataclass(frozen=True)
class RequiredCheck:
    """One required context, with its owning app.

    `must_run` separates two different requirements that happen to share a
    mechanism. Branch protection's own contexts may conclude `skipped` --
    that is by design on a docs-only diff, and protection accepts it. A
    context this worker adds itself with `--require-context` is the strict
    OpenSpec validation, and a skipped validation proves nothing about the
    documents about to be merged, so it must have actually run.
    """

    context: str
    app_id: int | None
    must_run: bool = False


@dataclass(frozen=True)
class CheckResult:
    """The latest observed result for one context."""

    context: str
    app_id: int | None
    status: str
    conclusion: str | None
    order: tuple

    @property
    def green(self) -> bool:
        return (
            self.status == COMPLETED and self.conclusion in PASSING_CONCLUSIONS
        )

    def describe(self) -> str:
        if self.status != COMPLETED:
            return f"{self.context}: {self.status}"
        return f"{self.context}: {self.conclusion or 'no conclusion'}"


@dataclass(frozen=True)
class PullRequest:
    number: int
    state: str
    draft: bool
    merged: bool
    mergeable: bool | None
    mergeable_state: str
    head_sha: str
    head_repository: str | None
    base_sha: str
    base_ref: str

    @classmethod
    def from_payload(cls, payload: dict) -> PullRequest:
        head = payload.get("head") or {}
        base = payload.get("base") or {}
        repository = head.get("repo") or {}
        return cls(
            number=int(payload.get("number", 0)),
            state=str(payload.get("state", "")),
            draft=bool(payload.get("draft", False)),
            merged=bool(payload.get("merged", False)),
            mergeable=payload.get("mergeable"),
            mergeable_state=str(payload.get("mergeable_state", "")),
            head_sha=str(head.get("sha", "")),
            head_repository=repository.get("full_name"),
            base_sha=str(base.get("sha", "")),
            base_ref=str(base.get("ref", "")),
        )


def gh_api(arguments: Sequence[str]) -> str:
    """Run `gh api` and return stdout, raising on any failure."""
    try:
        completed = subprocess.run(
            ["gh", "api", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=API_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise MergeError(f"cannot run gh: {error}") from error
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise MergeError(f"gh api failed: {message or 'no output'}")
    return completed.stdout


def _load(api: Callable[[list[str]], str], arguments: list[str]) -> object:
    try:
        return json.loads(api(arguments))
    except json.JSONDecodeError as error:
        raise MergeError(f"unreadable API response: {error}") from error


def read_pull_request(
    repository: str, number: int, api: Callable[[list[str]], str]
) -> PullRequest:
    payload = _load(api, [f"repos/{repository}/pulls/{number}"])
    if not isinstance(payload, dict):
        raise MergeError("pull request response was not an object")
    return PullRequest.from_payload(payload)


def _required_from_status_checks(payload: object) -> tuple[RequiredCheck, ...]:
    """Parse one `required_status_checks` object.

    Returns an empty tuple when the object declares nothing, so the caller
    can decide whether to try another source. An entry it cannot read
    raises instead: a half-understood requirement is not a requirement.
    """
    if not isinstance(payload, dict):
        return ()
    entries = payload.get("checks")
    if isinstance(entries, list) and entries:
        checks: list[RequiredCheck] = []
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("context"):
                raise MergeError("a required check entry is unreadable")
            app = entry.get("app_id")
            checks.append(
                RequiredCheck(
                    context=str(entry["context"]),
                    app_id=int(app) if isinstance(app, int) else None,
                )
            )
        return tuple(checks)
    contexts = payload.get("contexts")
    if isinstance(contexts, list) and contexts:
        for name in contexts:
            if not isinstance(name, str) or not name:
                raise MergeError("a required context entry is unreadable")
        return tuple(RequiredCheck(context=name, app_id=None) for name in contexts)
    return ()


def require_branch_ref(value: str) -> str:
    """Refuse a base ref that is not an ordinary branch name."""
    if not BRANCH_REF.fullmatch(value) or ".." in value or value.endswith("/"):
        raise MergeError(f"the base ref {value!r} is not an ordinary branch name")
    return value


def read_required_checks(
    repository: str, base_ref: str, api: Callable[[list[str]], str]
) -> tuple[RequiredCheck, ...]:
    """Discover the contexts branch protection actually requires.

    Two sources, in order of what an Actions token can actually reach:

    1. `GET /repos/{repo}/branches/{branch}` -- `Contents: read`. Verified
       live on this repository: its `protection.required_status_checks`
       carries the same nine `{context, app_id}` entries the dedicated
       endpoint returns.
    2. `GET /repos/{repo}/branches/{branch}/protection` --
       `Administration: read`, which a workflow `permissions:` block cannot
       grant. Kept for a caller running with an admin credential.

    An empty result is never returned. A branch with no required checks, or
    protection this worker cannot read, means nothing was proven about the
    commit, which is the opposite of permission to merge it.
    """
    require_branch_ref(base_ref)
    problems: list[str] = []

    try:
        payload = _load(api, [f"repos/{repository}/branches/{base_ref}"])
        if not isinstance(payload, dict):
            raise MergeError("branch response was not an object")
        if payload.get("protected") is not True:
            # An unprotected branch has no protection to read anywhere, so
            # this is the answer rather than a reason to try again.
            raise MergeError(
                f"branch {base_ref} is not protected, so no check was ever "
                "required on it; refusing rather than merging into it"
            )
        found = _required_from_status_checks(
            (payload.get("protection") or {}).get("required_status_checks")
        )
        if found:
            return found
        problems.append(
            f"the branch endpoint reports no required status checks on {base_ref}"
        )
    except MergeError as error:
        if "is not protected" in str(error):
            raise
        problems.append(f"the branch endpoint could not answer ({error})")

    try:
        payload = _load(api, [f"repos/{repository}/branches/{base_ref}/protection"])
        if not isinstance(payload, dict):
            raise MergeError("branch protection response was not an object")
        found = _required_from_status_checks(payload.get("required_status_checks"))
        if found:
            return found
        problems.append(
            f"the protection endpoint reports no required status checks on {base_ref}"
        )
    except MergeError as error:
        problems.append(f"the protection endpoint could not answer ({error})")

    raise MergeError(
        f"could not discover the checks branch protection requires on {base_ref}; "
        "refusing rather than treating an unknown set as a pass. "
        "GET /repos/{owner}/{repo}/branches/{branch} needs Contents: read, "
        "which GITHUB_TOKEN can hold; "
        "GET /repos/{owner}/{repo}/branches/{branch}/protection needs "
        "Administration: read, which it cannot, because the workflow "
        "`permissions:` block has no `administration` key. "
        + "; ".join(problems)
    )


def _paged(
    api: Callable[[list[str]], str], repository: str, suffix: str, key: str
) -> list[dict]:
    """Read every page of a list endpoint, or raise rather than truncate.

    Stopping at `MAX_PAGES` and returning what was read would hide the
    newest result for a context behind a page boundary, which turns a
    superseded `success` into the answer. A truncated read is an error.
    """
    collected: list[dict] = []
    for page in range(1, MAX_PAGES + 1):
        separator = "&" if "?" in suffix else "?"
        payload = _load(
            api,
            [f"repos/{repository}/{suffix}{separator}per_page={PAGE_SIZE}&page={page}"],
        )
        if not isinstance(payload, dict):
            raise MergeError(f"{key} response was not an object")
        entries = payload.get(key)
        if not isinstance(entries, list):
            raise MergeError(f"{key} response had no list")
        collected.extend(entry for entry in entries if isinstance(entry, dict))
        if len(entries) < PAGE_SIZE:
            return collected
    raise MergeError(
        f"{key} for {suffix} did not end within {MAX_PAGES} pages; refusing "
        "rather than deciding on a truncated read"
    )


def read_check_state(
    repository: str, sha: str, api: Callable[[list[str]], str]
) -> dict[str, CheckResult]:
    """The latest result for every context reported on `sha`.

    Check runs win over commit statuses when both exist for a context,
    because Actions reports through check runs here. Within each source the
    most recent entry wins, which is what branch protection itself uses.
    """
    if not FULL_SHA.fullmatch(sha):
        raise MergeError(f"{sha!r} is not a full commit sha")
    observed: dict[str, CheckResult] = {}

    def offer(candidate: CheckResult) -> None:
        current = observed.get(candidate.context)
        if current is None or candidate.order > current.order:
            observed[candidate.context] = candidate

    for run_payload in _paged(
        api, repository, f"commits/{sha}/check-runs", "check_runs"
    ):
        application = run_payload.get("app") or {}
        offer(
            CheckResult(
                context=str(run_payload.get("name", "")),
                app_id=application.get("id"),
                status=str(run_payload.get("status", "")),
                conclusion=run_payload.get("conclusion"),
                order=(
                    1,
                    str(run_payload.get("started_at") or ""),
                    int(run_payload.get("id") or 0),
                ),
            )
        )

    # Commit statuses are consulted for contexts that post them. The
    # endpoint's own `state` field is ignored: it reports "pending" with an
    # empty list on commits whose check runs all passed. `app_id` stays
    # None because a status has no app -- see `failing_checks`.
    for status in _paged(api, repository, f"commits/{sha}/status", "statuses"):
        state = str(status.get("state", ""))
        offer(
            CheckResult(
                context=str(status.get("context", "")),
                app_id=None,
                status=COMPLETED,
                conclusion=SUCCESS if state == SUCCESS else state,
                order=(0, str(status.get("created_at") or ""), 0),
            )
        )
    return observed


def failing_checks(
    required: Sequence[RequiredCheck], observed: dict[str, CheckResult]
) -> tuple[str, ...]:
    """Every required context that is not a completed success.

    When protection pins a context to an app, only that app satisfies it.
    An unknown app does not, and a commit status -- which has no app and
    which anyone with write access can post -- never does. GitHub applies
    the same rule; a worker that applied a weaker one would ask for merges
    protection then refuses, and would report a forged status as evidence.
    """
    failures: list[str] = []
    for check in required:
        result = observed.get(check.context)
        if result is None:
            failures.append(f"{check.context}: no result reported for this commit")
            continue
        if check.app_id is not None and result.app_id != check.app_id:
            reporter = (
                f"app {result.app_id}"
                if result.app_id is not None
                else "no app (a commit status, which anyone with write access can post)"
            )
            failures.append(
                f"{check.context}: reported by {reporter}, "
                f"but protection requires app {check.app_id}"
            )
            continue
        if check.must_run and result.conclusion != SUCCESS:
            failures.append(
                f"{result.describe()} (this context must run, not be skipped)"
            )
        elif not result.green:
            failures.append(result.describe())
    return tuple(failures)


def read_workflow_runs(
    repository: str, workflow_file: str, sha: str, api: Callable[[list[str]], str]
) -> list[dict]:
    """Every run of one workflow FILE recorded against one commit."""
    if not WORKFLOW_FILE.fullmatch(workflow_file):
        raise MergeError(
            f"{workflow_file!r} is not a workflow file name; the Actions API "
            "takes a bare file name, never a path"
        )
    if not FULL_SHA.fullmatch(sha):
        raise MergeError(f"{sha!r} is not a full commit sha")
    payload = _load(
        api,
        [
            f"repos/{repository}/actions/workflows/{workflow_file}/runs"
            f"?head_sha={sha}&per_page={PAGE_SIZE}"
        ],
    )
    if not isinstance(payload, dict):
        raise MergeError("workflow runs response was not an object")
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise MergeError("workflow runs response had no run list")
    return [entry for entry in runs if isinstance(entry, dict)]


def workflow_failures(
    repository: str, workflow_file: str, sha: str, api: Callable[[list[str]], str]
) -> tuple[str, ...]:
    """Refusal reasons for the required workflow, empty when it is green.

    This is the provenance pin behind the strict validation requirement. A
    check-run name proves nothing -- any workflow can publish any job name,
    and any collaborator can post a commit status with any context. A run
    of this FILE, on this sha, under `pull_request`, is the base branch's
    workflow reporting on the commit being merged, because the
    governance-identity check proved the head's `.github` tree matches the
    base's.

    The newest run decides, by run id and then by attempt: a re-run keeps
    its id and increments the attempt, a fresh run gets a larger id.
    """
    latest: tuple[tuple[int, int], dict] | None = None
    for entry in read_workflow_runs(repository, workflow_file, sha, api):
        if str(entry.get("head_sha", "")) != sha:
            continue
        if str(entry.get("event", "")) != WORKFLOW_EVENT:
            continue
        order = (int(entry.get("id") or 0), int(entry.get("run_attempt") or 0))
        if latest is None or order > latest[0]:
            latest = (order, entry)
    if latest is None:
        return (
            f"{workflow_file}: no completed {WORKFLOW_EVENT} run for this commit",
        )
    entry = latest[1]
    status = str(entry.get("status", ""))
    conclusion = entry.get("conclusion")
    if status != COMPLETED or conclusion != SUCCESS:
        return (
            f"{workflow_file}: the latest run is "
            f"{conclusion or status or 'in an unknown state'}",
        )
    return ()


def read_compare_status(
    repository: str, base: str, head: str, api: Callable[[list[str]], str]
) -> str:
    """`ahead`, `behind`, `identical`, or `diverged` for `base...head`."""
    for value in (base, head):
        if not FULL_SHA.fullmatch(value):
            raise MergeError(f"{value!r} is not a full commit sha")
    payload = _load(api, [f"repos/{repository}/compare/{base}...{head}"])
    if not isinstance(payload, dict):
        raise MergeError("compare response was not an object")
    return str(payload.get("status", ""))


def identity_refusals(
    pull: PullRequest,
    repository: str,
    expected_head: str,
    expected_base_ref: str,
) -> tuple[str, ...]:
    """Reasons that waiting will never fix.

    These are checked on every poll, so a pull request that is closed,
    converted to a draft, retargeted, or pushed to during the wait stops
    this run instead of being discovered at merge time.
    """
    refusals: list[str] = []
    if pull.state != "open":
        refusals.append(f"the pull request is {pull.state or 'in an unknown state'}")
    if pull.merged:
        refusals.append("the pull request is already merged")
    if pull.draft:
        refusals.append("the pull request is a draft")
    if pull.head_repository != repository:
        refusals.append(
            f"the head is on {pull.head_repository or 'an unknown repository'}, "
            "not a branch of this repository"
        )
    if pull.head_sha != expected_head:
        refusals.append(
            f"the head moved to {pull.head_sha or 'nothing'}; "
            f"this run decided about {expected_head}"
        )
    if pull.base_ref != expected_base_ref:
        refusals.append(
            f"the base branch is now {pull.base_ref or 'unknown'}; this run "
            f"decided about {expected_base_ref}, and a squash merge onto a "
            "different base lands a different diff"
        )
    return tuple(refusals)


def base_movement_refusals(
    repository: str,
    pull: PullRequest,
    expected_base_sha: str,
    api: Callable[[list[str]], str],
) -> tuple[str, ...]:
    """Refuse a base branch that did not simply move forward.

    The boundary verdict covers `merge-base(base, head)...head`. The base
    branch advancing does not change that set, so an ordinary busy `main`
    must not refuse every submission. A base that was rewound or rewritten
    does change it: commits the classifier never judged become part of what
    the squash merge would land.
    """
    if pull.base_sha == expected_base_sha:
        return ()
    status = read_compare_status(repository, expected_base_sha, pull.base_sha, api)
    if status in FORWARD_STATUSES:
        return ()
    return (
        f"the base branch is at {pull.base_sha or 'nothing'}, which is "
        f"{status or 'not comparable'} relative to the authorized base "
        f"{expected_base_sha}; only a base that moved forward keeps the "
        "classified diff the diff that would land",
    )


# `mergeable_state` values that still permit a merge. `behind` is mergeable
# because this branch does not require branches to be up to date, and
# `unstable` means a non-required check failed. Everything else -- `dirty`
# (conflicts), `blocked`, `draft`, `unknown`, and any value GitHub adds
# later -- refuses.
MERGEABLE_STATES = frozenset({"clean", "unstable", "behind"})


def eligibility_refusals(
    pull: PullRequest,
    repository: str,
    expected_head: str,
    expected_base_ref: str,
) -> tuple[str, ...]:
    """Every reason this pull request must not be merged right now."""
    refusals = list(
        identity_refusals(pull, repository, expected_head, expected_base_ref)
    )
    if pull.mergeable is not True:
        refusals.append(
            "GitHub does not report the pull request as mergeable "
            f"(mergeable={pull.mergeable}, state={pull.mergeable_state or 'unknown'})"
        )
    elif pull.mergeable_state not in MERGEABLE_STATES:
        refusals.append(
            f"the merge state is {pull.mergeable_state or 'unknown'}, "
            "which is not a state this worker merges from"
        )
    return tuple(refusals)


def merge_pull_request(
    repository: str, number: int, sha: str, api: Callable[[list[str]], str]
) -> str:
    """Merge exactly `sha`, or raise.

    `sha` binds the request to one commit: GitHub refuses with `409` if the
    head has moved, so a push that races this call cannot be merged by it.
    """
    payload = _load(
        api,
        [
            "--method",
            "PUT",
            f"repos/{repository}/pulls/{number}/merge",
            "-f",
            f"sha={sha}",
            "-f",
            f"merge_method={MERGE_METHOD}",
        ],
    )
    if not isinstance(payload, dict) or not payload.get("merged"):
        raise MergeError(f"the merge request did not report success: {payload}")
    return str(payload.get("sha") or "")


def run(
    repository: str,
    number: int,
    expected_head: str,
    expected_base_ref: str,
    expected_base_sha: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    extra_contexts: Sequence[str] = (),
    required_workflow: str | None = None,
    api: Callable[[list[str]], str] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Wait for the required checks on one commit, then merge it."""
    call = api or gh_api

    def blocking(pull: PullRequest) -> tuple[str, ...]:
        """Every reason to stop that is not "a check is still running"."""
        refusals = identity_refusals(
            pull, repository, expected_head, expected_base_ref
        )
        if refusals:
            return refusals
        return base_movement_refusals(repository, pull, expected_base_sha, call)

    try:
        pull = read_pull_request(repository, number, call)
        refusals = blocking(pull)
        if refusals:
            for refusal in refusals:
                print(f"refused: {refusal}")
            return REFUSED

        required = list(read_required_checks(repository, expected_base_ref, call))
        # The OpenSpec strict validation context is required in addition to
        # branch protection's own set. Protection cannot express it, because
        # making it a required check would block every pull request that
        # does not touch `openspec/` (chelis#419).
        #
        # The app is taken from protection's own set when protection agrees
        # on one, so a forged commit status cannot answer for the context.
        # This is a redundancy, not the guarantee: `required_workflow` below
        # is what actually pins the result to a trusted producer.
        application = {check.app_id for check in required if check.app_id is not None}
        extra_app = application.pop() if len(application) == 1 else None
        required.extend(
            RequiredCheck(context=name, app_id=extra_app, must_run=True)
            for name in extra_contexts
        )
        print(f"required contexts ({len(required)}):")
        for check in required:
            print(f"  - {check.context}")
        if required_workflow:
            print(f"required workflow run: {required_workflow} on {expected_head}")

        deadline = clock() + timeout_seconds
        failures: tuple[str, ...] = ()
        while True:
            observed = read_check_state(repository, expected_head, call)
            failures = failing_checks(required, observed)
            if required_workflow:
                failures += workflow_failures(
                    repository, required_workflow, expected_head, call
                )
            if not failures:
                break
            if clock() >= deadline:
                print(f"not merged: {len(failures)} required result(s) not green")
                for failure in failures:
                    print(f"  - {failure}")
                print(RETRY_ADVICE)
                return REFUSED
            sleeper(poll_seconds)
            # Re-read the pull request each round: a push, a close, a
            # retarget, or a conversion to draft during the wait must stop
            # this run rather than be discovered only at merge time.
            # Mergeability is not checked here, because it is legitimately
            # unsettled while checks are still running.
            pull = read_pull_request(repository, number, call)
            refusals = blocking(pull)
            if refusals:
                for refusal in refusals:
                    print(f"refused: {refusal}")
                return REFUSED

        # Final read immediately before the merge. The `sha` binding below
        # is the real guard against a race; this makes the refusal legible
        # instead of surfacing as a bare 409.
        for attempt in range(MERGEABLE_ATTEMPTS):
            pull = read_pull_request(repository, number, call)
            refusals = blocking(pull) or eligibility_refusals(
                pull, repository, expected_head, expected_base_ref
            )
            if not refusals:
                break
            if pull.mergeable is None and attempt + 1 < MERGEABLE_ATTEMPTS:
                # Not "no": GitHub has not finished computing the test
                # merge. Ask again a bounded number of times.
                sleeper(MERGEABLE_POLL_SECONDS)
                continue
            for refusal in refusals:
                print(f"refused: {refusal}")
            return REFUSED

        try:
            merged_sha = merge_pull_request(repository, number, expected_head, call)
        except MergeError as error:
            print(f"blocked: the merge was refused: {error}")
            print(
                "Branch protection is the final arbiter. Nothing here bypasses "
                "it; resolve the reported condition and run again."
            )
            return REFUSED
        print(f"merged {expected_head} as {merged_sha}")
        return MERGED
    except MergeError as error:
        print(f"openspec_merge: {error}", file=sys.stderr)
        return OPERATIONAL_FAILURE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge one validated OpenSpec document pull request."
    )
    parser.add_argument("--repository", required=True, help="owner/name")
    parser.add_argument("--number", required=True, type=int, help="pull request number")
    parser.add_argument("--head", required=True, help="the exact head SHA to merge")
    parser.add_argument(
        "--base-ref",
        required=True,
        help="the base branch the boundary verdict was computed against",
    )
    parser.add_argument(
        "--base-sha",
        required=True,
        help="the base SHA the boundary verdict was computed against",
    )
    parser.add_argument(
        "--require-context",
        action="append",
        default=[],
        help="a context required in addition to branch protection's set",
    )
    parser.add_argument(
        "--require-workflow",
        default=None,
        help=(
            "a workflow file name that must have a successful pull_request "
            "run on the exact head SHA"
        ),
    )
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS
    )
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if not os.environ.get("GH_TOKEN") and not os.environ.get("GITHUB_TOKEN"):
        print("openspec_merge: no GitHub token in the environment", file=sys.stderr)
        return OPERATIONAL_FAILURE
    for name, value in (("--head", arguments.head), ("--base-sha", arguments.base_sha)):
        if not FULL_SHA.fullmatch(value):
            print(
                f"openspec_merge: {name} must be a lowercase full SHA",
                file=sys.stderr,
            )
            return OPERATIONAL_FAILURE
    try:
        require_branch_ref(arguments.base_ref)
        if arguments.require_workflow and not WORKFLOW_FILE.fullmatch(
            arguments.require_workflow
        ):
            raise MergeError("--require-workflow must be a workflow file name")
    except MergeError as error:
        print(f"openspec_merge: {error}", file=sys.stderr)
        return OPERATIONAL_FAILURE
    return run(
        repository=arguments.repository,
        number=arguments.number,
        expected_head=arguments.head,
        expected_base_ref=arguments.base_ref,
        expected_base_sha=arguments.base_sha,
        timeout_seconds=arguments.timeout_seconds,
        poll_seconds=arguments.poll_seconds,
        extra_contexts=tuple(arguments.require_context),
        required_workflow=arguments.require_workflow,
    )


if __name__ == "__main__":
    sys.exit(main())
