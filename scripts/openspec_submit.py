#!/usr/bin/env python3
"""Submit an OpenSpec document change with one command.

An OpenSpec change is prose. It has no build, no test suite, and nothing a
reviewer can run. Asking a human to approve one is a queue, not a control.
So the maintainer authorized automated acceptance for OpenSpec document
paths, and this command is the front door to it:

    openspec-submit                 # submit, then wait for the outcome
    openspec-submit --dry-run       # print the plan; write nothing
    openspec-submit --no-wait       # return a queued pull request URL
    openspec-submit --watch 1650    # resume waiting on an existing one

Outside Devenv: `python3 scripts/openspec_submit.py` with a managed Python.

The command prints one line per stage -- prepare, validate, submit, wait --
and one final outcome. The outcome vocabulary is deliberate: "opened a pull
request" describes the transport, not the result, so it is never the last
word. A default run waits for the merge to happen or to stop, and says
which.

    ACCEPTED  merged, with the merge commit
    QUEUED    the pull request exists; the outcome is not known yet
    BLOCKED   something stopped it, named, with the remedy
    FAILED    the command could not run its own steps

Exit status: `0` accepted or queued on request, `1` blocked, `2` failed,
`3` still waiting when the timeout expired.

Blocked is never reported as "human review required". The five ways a
submission stops -- a non-document path, a stale branch, an unresolved
merge, a validation finding, a failed check -- need five different actions,
so each says which one it is and what to do about it. A validation finding
in particular is a validation result, not an approval question.

The pull request is merged by the `openspec-autoland` workflow, which waits
for the required checks on that exact commit and then performs one ordinary
merge bound to it. Auto-merge is never used: it is a standing grant on a
mutable branch, so a decision made on one commit would authorize every
commit after it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Outcomes, in the vocabulary the command prints. "A pull request was
# created" is deliberately not one of them: it says what happened to the
# transport, not whether the change was accepted.
ACCEPTED = 0
BLOCKED = 1
FAILED = 2
TIMED_OUT = 3

# Retained names for callers that still speak the old vocabulary.
SUCCESS = ACCEPTED
REVIEW_REQUIRED = BLOCKED
OPERATIONAL_FAILURE = FAILED

DEFAULT_WAIT_SECONDS = 3600
DEFAULT_POLL_SECONDS = 20
VALIDATION_CONTEXT = "OpenSpec Strict Validation"
MAX_DETAIL_WIDTH = 150
REQUIRED_TOOLS = ("git", "gh")

BASE_BRANCH = "main"
BASE_REVISION = "origin/main"
BRANCH_PREFIX = "openspec/"
COMMAND_TIMEOUT_SECONDS = 600
SLUG_CHARACTERS = re.compile(r"[^a-z0-9-]+")

_MODULE_PATH = Path(__file__).resolve().parent / "openspec_acceptance.py"
_SPEC = importlib.util.spec_from_file_location("openspec_acceptance", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load the acceptance module: {_MODULE_PATH}")
acceptance = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = acceptance
_SPEC.loader.exec_module(acceptance)


class SubmitError(RuntimeError):
    """An operational failure: the command could not run its own steps."""


class ReviewRequired(RuntimeError):
    """The submission cannot proceed, and a person must act.

    `stage` names where it stopped and `remedy` is the exact next command
    or edit. A single undifferentiated "human review required" told the
    contributor nothing about which of five very different problems they
    had hit.
    """

    def __init__(
        self,
        summary: str,
        *,
        stage: str = "prepare",
        remedy: str = "",
        detail: Sequence[str] = (),
    ) -> None:
        super().__init__(summary)
        self.stage = stage
        self.remedy = remedy
        self.detail = tuple(detail)


class Reporter:
    """Prints one aligned line per stage, then one outcome line."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout

    def stage(self, name: str, message: str) -> None:
        print(f"  {name:<9} {message}", file=self.stream)

    def detail(self, message: str) -> None:
        # One finding can run to several hundred characters. The first
        # clause identifies it; the rest is guidance the tool already
        # printed in full when it was run directly.
        text = " ".join(str(message).split())
        if len(text) > MAX_DETAIL_WIDTH:
            text = text[: MAX_DETAIL_WIDTH - 1] + "…"
        print(f"            {text}", file=self.stream)

    def outcome(self, label: str, message: str) -> None:
        print(f"  {label:<9} {message}", file=self.stream)


@dataclass(frozen=True)
class Plan:
    """Everything the submission will do, decided before it does any of it."""

    repository: Path
    paths: tuple[str, ...]
    branch: str
    title: str
    body: str

    def describe(self) -> str:
        listing = "\n".join(f"  {path}" for path in self.paths)
        return (
            f"branch: {self.branch}\n"
            f"title:  {self.title}\n"
            f"base:   {BASE_BRANCH}\n"
            f"paths:\n{listing}"
        )


def run(
    command: Sequence[str],
    repository: Path,
    runner=None,
    check: bool = True,
) -> str:
    """Run one command in the repository and return its stdout."""
    execute = runner or subprocess.run
    try:
        completed = execute(
            list(command),
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SubmitError(f"cannot run {command[0]}: {error}") from error
    stdout = getattr(completed, "stdout", "") or ""
    if check and getattr(completed, "returncode", None) != 0:
        stderr = (getattr(completed, "stderr", "") or "").strip()
        raise SubmitError(
            f"{' '.join(command)} failed: {stderr or stdout.strip() or 'no output'}"
        )
    return stdout


def repository_root(start: Path, runner=None) -> Path:
    """Resolve the working repository root."""
    output = run(["git", "rev-parse", "--show-toplevel"], start, runner).strip()
    if not output:
        raise SubmitError("not inside a Git repository")
    return Path(output)


def untracked_paths(repository: Path, runner=None) -> tuple[str, ...]:
    """Untracked files at OpenSpec document paths.

    A new OpenSpec change is entirely untracked until it is staged, so the
    boundary check has to see these paths before anything is committed --
    in particular their filesystem modes, which is what catches a document
    spelled as a symlink or carrying the executable bit.

    Untracked files OUTSIDE the document set are deliberately excluded, and
    the reason is structural rather than a relaxation. An untracked file
    reaches a commit only through an explicit `git add`, and `submit` adds
    exactly the classified paths as `:(literal)` pathspecs, then commits
    with `--only` naming the same list. So a stray can never be in the
    commit CI classifies, and refusing on it reports a difference that does
    not exist. It is not hypothetical: the maintainer's working tree holds
    37 untracked, unignored non-document paths (`result`, `.work/`, build
    output), and classifying them refused every document submission.

    Tracked changes are NOT filtered this way. The index is a commit input,
    and a staged code path must still stop the submission.
    """
    output = run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        repository,
        runner,
    )
    return tuple(
        sorted(
            part
            for part in output.split("\0")
            if part and acceptance.document_class(part) is not None
        )
    )


def untracked_records(repository: Path, paths: Sequence[str]) -> str:
    """Render untracked files as `git diff --raw` additions.

    The recorded mode comes from the filesystem, so a symlink, a directory
    entry, or an executable bit reaches the same classifier that judges
    tracked changes instead of getting its own weaker check.
    """
    parts: list[str] = []
    for path in paths:
        absolute = repository / path
        if absolute.is_symlink():
            mode = "120000"
        elif absolute.is_dir():
            mode = "040000"
        elif os.access(absolute, os.X_OK):
            mode = "100755"
        else:
            mode = "100644"
        parts.append(f":000000 {mode} {'0' * 7} {'0' * 7} A\0{path}\0")
    return "".join(parts)


def merge_base(repository: Path, runner=None) -> str:
    """The commit this branch started from.

    Comparing against the tip of `origin/main` would also report every
    commit `main` gained since the branch started. Those are other
    people's changes, and reporting them refuses a document submission for
    code the contributor never wrote.
    """
    output = run(
        ["git", "merge-base", BASE_REVISION, "HEAD"], repository, runner
    ).strip()
    if not acceptance.valid_revision(output):
        raise SubmitError(f"the merge base is not a commit: {output!r}")
    return output


def head_revision(repository: Path, runner=None) -> str:
    """The commit this branch is on."""
    output = run(["git", "rev-parse", "HEAD"], repository, runner).strip()
    if not acceptance.valid_revision(output):
        raise SubmitError(f"HEAD is not a commit: {output!r}")
    return output


def require_identical_governance(repository: Path, runner=None) -> None:
    """Refuse a branch whose governance content differs from the base.

    The hosted classifier applies the same rule, so skipping it here would
    push a branch that CI immediately refuses. The comparison is against
    the base tip rather than the merge base, because staleness is exactly
    what it is looking for: a branch that predates a change to `.github`
    or `devenv.nix` carries the old one.
    """
    differences = acceptance.governance_differences(
        repository, BASE_REVISION, head_revision(repository, runner), runner
    )
    if differences:
        raise ReviewRequired(
            f"your branch is behind {BASE_REVISION} on {len(differences)} "
            "governance path(s)",
            stage="prepare",
            remedy=f"git fetch origin && git rebase origin/{BASE_BRANCH}",
            detail=differences,
        )


def require_no_conflicts(repository: Path, runner=None) -> None:
    """Refuse while the index holds an unresolved merge.

    A conflicted path is reported as an ordinary modification by
    `git diff`, so the classifier cannot see the conflict. Refusing here
    keeps a half-merged tree from being submitted as a document change.
    """
    if run(["git", "ls-files", "--unmerged", "-z"], repository, runner).strip("\0"):
        raise ReviewRequired(
            "the working tree has an unresolved merge",
            stage="prepare",
            remedy="Resolve the conflict and commit, then run openspec-submit again.",
        )


def worktree_diff(repository: Path, base: str, runner=None) -> str:
    """Raw diff of everything this submission could commit.

    Three streams, because a commit can pick up content from any of them:
    the working tree, the index, and untracked files. Classifying only the
    working tree would miss a path staged with one content and left
    unchanged on disk, which is a real way to smuggle a file past a
    worktree-only check.
    """
    try:
        tracked = acceptance.read_raw_diff(repository, base, None, runner)
    except acceptance.BoundaryError as error:
        raise SubmitError(str(error)) from error
    staged = run(
        [
            "git",
            "diff",
            "--raw",
            "-z",
            "-M",
            "--no-color",
            "--cached",
            base,
            "--",
        ],
        repository,
        runner,
    )
    return (
        tracked
        + staged
        + untracked_records(repository, untracked_paths(repository, runner))
    )


def derive_slug(paths: Sequence[str]) -> str:
    """Name the branch after what changed, with a content-stable suffix."""
    changes = sorted(
        {
            path.split("/")[2]
            for path in paths
            if path.startswith("openspec/changes/") and len(path.split("/")) > 3
        }
        - {"archive"}
    )
    capabilities = sorted(
        {
            path.split("/")[2]
            for path in paths
            if path.startswith("openspec/specs/") and len(path.split("/")) > 3
        }
    )
    if len(changes) == 1:
        stem = changes[0]
    elif not changes and len(capabilities) == 1:
        stem = f"spec-{capabilities[0]}"
    else:
        stem = "documents"
    stem = SLUG_CHARACTERS.sub("-", stem.lower()).strip("-") or "documents"
    digest = hashlib.sha256("\n".join(sorted(paths)).encode("utf-8")).hexdigest()
    return f"{BRANCH_PREFIX}{stem[:48]}-{digest[:8]}"


def derive_title(paths: Sequence[str], slug: str) -> str:
    """A title that says which documents moved."""
    stem = slug[len(BRANCH_PREFIX) : -9]
    specs = sum(1 for path in paths if path.startswith("openspec/specs/"))
    if stem == "documents":
        return f"docs(openspec): update {len(paths)} OpenSpec documents"
    if specs and stem.startswith("spec-"):
        return f"docs(openspec): update the {stem[5:]} capability specification"
    return f"docs(openspec): update the {stem} change"


def build_body(paths: Sequence[str], verdict) -> str:
    """State what the automation proved, and what it did not."""
    listing = "\n".join(f"- `{path}`" for path in paths)
    return (
        "Automated OpenSpec document submission.\n\n"
        "## Changed documents\n\n"
        f"{listing}\n\n"
        "## Acceptance basis\n\n"
        f"- Boundary verdict: `{verdict.name}` "
        f"({verdict.reason}).\n"
        "- `openspec validate --all --strict --no-interactive` passed locally.\n"
        "- Every changed path is an OpenSpec document, so no code, workflow, "
        "script, or acceptance-policy input is in this change.\n\n"
        "## What this does not prove\n\n"
        "Structural validation proves schema validity only. It does not "
        "prove that the wording is correct, wanted, or consistent with "
        "`spec/**`. Where an OpenSpec artifact contradicts an owning "
        "authority, the owning authority controls and the artifact is "
        "corrected.\n"
    )


def validate_openspec(repository: Path, runner=None) -> None:
    """Run strict structural validation with the OpenSpec CLI."""
    executable = os.environ.get("OPENSPEC_BIN") or shutil.which("openspec")
    if not executable:
        raise SubmitError(
            "the OpenSpec CLI is missing; enter Devenv or install OpenSpec 1.6.0"
        )
    execute = runner or subprocess.run
    try:
        completed = execute(
            [executable, "validate", "--all", "--strict", "--no-interactive"],
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SubmitError(f"cannot run OpenSpec: {error}") from error
    returncode = getattr(completed, "returncode", None)
    if returncode == 0:
        return
    output = (getattr(completed, "stdout", "") or "") + (
        getattr(completed, "stderr", "") or ""
    )
    if returncode == 1:
        lines = [line.strip() for line in output.splitlines()]
        # Prefer the error lines: the `✗` lines are per-item headers that
        # repeat what the error already says.
        findings = [line for line in lines if "[ERROR]" in line]
        if not findings:
            findings = [line for line in lines if line.startswith("✗")]
        summary = (
            f"strict validation reported {len(findings)} finding(s)"
            if findings
            else "strict validation reported findings"
        )
        raise ReviewRequired(
            summary,
            stage="validate",
            remedy="Fix the findings, then run openspec-submit again.",
            detail=findings[:8] or [output.strip()[:400]],
        )
    raise SubmitError(f"OpenSpec exited with operational code {returncode}")


def build_plan(repository: Path, paths: Sequence[str], verdict) -> Plan:
    slug = derive_slug(paths)
    return Plan(
        repository=repository,
        paths=tuple(paths),
        branch=slug,
        title=derive_title(paths, slug),
        body=build_body(paths, verdict),
    )


def literal(path: str) -> str:
    """Pass `path` to Git as a filename, never as a glob.

    A file named `openspec/specs/*.md` is a valid path and an ordinary
    pathspec would expand it across the tree, committing files the
    classifier never saw.
    """
    return f":(literal){path}"


def submit(plan: Plan, runner=None) -> str:
    """Branch, commit, push, and open the pull request; return its URL.

    It stops there. It does not request auto-merge, which is a standing
    grant on a mutable branch: GitHub keeps auto-merge enabled across later
    pushes by anyone with write access, so a verdict computed on one commit
    would authorize every commit after it. Any withdrawal we could write
    races the merge it is trying to prevent.
    """
    repository = plan.repository
    pathspecs = [literal(path) for path in plan.paths]
    run(["git", "checkout", "-b", plan.branch], repository, runner)
    # `add` registers untracked documents; `commit --only` then commits
    # exactly these paths and nothing else in the index. Without `--only`,
    # any unrelated staged change would ride along into a pull request
    # that the classifier approved without ever seeing it.
    run(["git", "add", "--", *pathspecs], repository, runner)
    run(
        [
            "git",
            "commit",
            "--only",
            "-m",
            plan.title,
            "-m",
            plan.body,
            "--",
            *pathspecs,
        ],
        repository,
        runner,
    )
    run(
        ["git", "push", "--set-upstream", "origin", plan.branch],
        repository,
        runner,
    )
    return run(
        [
            "gh",
            "pr",
            "create",
            "--base",
            BASE_BRANCH,
            "--head",
            plan.branch,
            "--title",
            plan.title,
            "--body",
            plan.body,
        ],
        repository,
        runner,
    ).strip().splitlines()[-1].strip()


def preflight(repository: Path, need_remote: bool, runner=None) -> None:
    """Check the tools and credentials before anything is written.

    A submission that discovers a missing `gh` after it has already made a
    branch, a commit, and a push leaves the contributor to clean up. All of
    it is knowable first, so all of it is checked first.
    """
    missing = [tool for tool in REQUIRED_TOOLS if shutil.which(tool) is None]
    if not (os.environ.get("OPENSPEC_BIN") or shutil.which("openspec")):
        missing.append("openspec")
    if missing:
        raise SubmitError(
            f"missing required tool(s): {', '.join(missing)}. "
            "Enter Devenv, or install them and try again."
        )
    if not need_remote:
        return
    execute = runner or subprocess.run
    try:
        completed = execute(
            ["gh", "auth", "status"],
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise SubmitError(f"cannot run gh: {error}") from error
    if getattr(completed, "returncode", None) != 0:
        raise SubmitError(
            "the GitHub CLI is not authenticated. Run `gh auth login`, then "
            "run openspec-submit again."
        )


def gh_json(command: Sequence[str], repository: Path, runner=None):
    """Run a `gh` command that prints JSON and parse it."""
    output = run(list(command), repository, runner).strip()
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError as error:
        raise SubmitError(f"unreadable gh response: {error}") from error


@dataclass(frozen=True)
class Submission:
    """The transport identity of one submission."""

    number: int
    url: str
    head: str


def find_existing(branch: str, repository: Path, runner=None) -> Submission | None:
    """An earlier submission of this exact content, if there is one.

    The branch name is derived from the changed paths, so re-running the
    command for unchanged content finds its own previous pull request
    instead of opening a second one.
    """
    payload = gh_json(
        [
            "gh",
            "pr",
            "list",
            "--head",
            branch,
            "--state",
            "all",
            "--limit",
            "1",
            "--json",
            "number,state,url,headRefOid",
        ],
        repository,
        runner,
    )
    if not payload:
        return None
    entry = payload[0]
    return Submission(
        number=int(entry.get("number", 0)),
        url=str(entry.get("url", "")),
        head=str(entry.get("headRefOid", "")),
    )


def failing_checks(repository: Path, head: str, runner=None) -> tuple[str, ...]:
    """Named checks that have concluded unsuccessfully on `head`.

    This is a report, not a gate: the merge worker decides. Naming the
    failing check here is what turns "it never merged" into something the
    contributor can act on.
    """
    payload = gh_json(
        ["gh", "api", f"repos/{{owner}}/{{repo}}/commits/{head}/check-runs"],
        repository,
        runner,
    )
    runs = (payload or {}).get("check_runs") or []
    failed = []
    for entry in runs:
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "completed":
            continue
        if entry.get("conclusion") in (None, "success", "skipped", "neutral"):
            continue
        failed.append(f"{entry.get('name', '?')}: {entry.get('conclusion')}")
    return tuple(sorted(set(failed)))


def wait_for_outcome(
    number: int,
    head: str,
    repository: Path,
    reporter: Reporter,
    timeout_seconds: int,
    poll_seconds: int,
    runner=None,
    sleeper=time.sleep,
    clock=time.monotonic,
) -> tuple[int, str, str]:
    """Wait for the pull request to merge or stop; return the outcome.

    Returns `(exit status, label, message)`. The pull request identity is
    fixed: this waits for one number and one head commit, and says so if
    either stops matching.
    """
    deadline = clock() + timeout_seconds
    url = ""
    while True:
        payload = gh_json(
            [
                "gh",
                "pr",
                "view",
                str(number),
                "--json",
                "state,url,headRefOid,mergeCommit",
            ],
            repository,
            runner,
        )
        payload = payload or {}
        url = str(payload.get("url") or url)
        state = str(payload.get("state") or "").upper()
        observed_head = str(payload.get("headRefOid") or "")
        merge_commit = (payload.get("mergeCommit") or {}).get("oid") or ""

        if state == "MERGED":
            return ACCEPTED, "ACCEPTED", f"merged as {merge_commit[:7]}  {url}"
        if state == "CLOSED":
            return (
                BLOCKED,
                "BLOCKED",
                f"the pull request was closed without merging  {url}",
            )
        if observed_head and observed_head != head:
            return (
                BLOCKED,
                "BLOCKED",
                "the head commit changed while waiting, so this decision no "
                f"longer covers it  {url}",
            )
        failures = failing_checks(repository, head, runner)
        if failures:
            reporter.outcome("BLOCKED", f"a required check failed  {url}")
            for failure in failures[:5]:
                reporter.detail(failure)
            hint = (
                "Fix the finding and push again."
                if any(VALIDATION_CONTEXT in f for f in failures)
                else "Fix the failure and push again; that restarts the decision."
            )
            reporter.detail(hint)
            return BLOCKED, "", ""
        if clock() >= deadline:
            return (
                TIMED_OUT,
                "QUEUED",
                f"still waiting when the timeout expired  {url}",
            )
        sleeper(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="openspec-submit",
        description=(
            "Submit an OpenSpec document change and wait for it to land. "
            "Only OpenSpec document paths are eligible; anything else stops "
            "with the reason and the remedy."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan and write nothing, locally or remotely",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="return as soon as the pull request exists, reporting it as queued",
    )
    parser.add_argument(
        "--watch",
        type=int,
        metavar="NUMBER",
        help="wait on an existing submission instead of creating one",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=DEFAULT_WAIT_SECONDS,
        metavar="SECONDS",
        help=f"how long to wait for the outcome (default {DEFAULT_WAIT_SECONDS})",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip the read-only `git fetch origin main` and use the local base",
    )
    return parser


def _report_stop(reporter: Reporter, label: str, error) -> None:
    reporter.outcome(label, str(error))
    for line in getattr(error, "detail", ()):
        reporter.detail(line)
    remedy = getattr(error, "remedy", "")
    if remedy:
        reporter.detail(remedy)


def main(argv: Sequence[str] | None = None, runner=None) -> int:
    arguments = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    reporter = Reporter()
    print("openspec-submit")
    try:
        repository = repository_root(Path.cwd(), runner)
        preflight(repository, need_remote=not arguments.dry_run, runner=runner)

        if arguments.watch is not None:
            head = head_revision(repository, runner)
            reporter.stage("wait", f"pull request #{arguments.watch} at {head[:7]}")
            status, label, message = wait_for_outcome(
                arguments.watch,
                head,
                repository,
                reporter,
                arguments.wait_timeout,
                DEFAULT_POLL_SECONDS,
                runner,
            )
            if label:
                reporter.outcome(label, message)
            return status

        if not arguments.no_fetch:
            run(["git", "fetch", "--quiet", "origin", BASE_BRANCH], repository, runner)

        require_no_conflicts(repository, runner)
        require_identical_governance(repository, runner)
        base = merge_base(repository, runner)
        verdict = acceptance.classify_stream(worktree_diff(repository, base, runner))
        if not verdict.paths:
            raise ReviewRequired(
                "there is nothing to submit",
                stage="prepare",
                remedy="Edit an OpenSpec document first.",
            )
        if not verdict.auto:
            raise ReviewRequired(
                "this is not an OpenSpec document change",
                stage="prepare",
                remedy=(
                    "Move the non-document paths to their own pull request, "
                    "which follows the ordinary review path."
                ),
                detail=verdict.reasons[:5],
            )

        paths = sorted(dict.fromkeys(verdict.paths))
        plan = build_plan(repository, paths, verdict)
        reporter.stage(
            "prepare",
            f"{len(paths)} document(s) -> {plan.branch}",
        )
        for path in paths:
            reporter.detail(path)

        validate_openspec(repository, runner)
        reporter.stage("validate", "openspec validate --all --strict: passed")

        if arguments.dry_run:
            fetched = "" if arguments.no_fetch else " (fetched origin/main, read-only)"
            reporter.outcome("DRY RUN", f"nothing was written{fetched}")
            reporter.detail(f"would push {plan.branch} and open a pull request")
            return ACCEPTED

        head = head_revision(repository, runner)
        existing = find_existing(plan.branch, repository, runner)
        if existing is not None:
            reporter.stage(
                "submit", f"already submitted as #{existing.number}  {existing.url}"
            )
            if arguments.no_wait:
                reporter.outcome("QUEUED", f"already open  {existing.url}")
                return ACCEPTED
            number, url, head = existing.number, existing.url, existing.head or head
        else:
            url = submit(plan, runner)
            number = int(url.rstrip("/").rsplit("/", 1)[-1] or 0)
            head = head_revision(repository, runner)
            reporter.stage("submit", f"opened #{number}  {url}")

        if arguments.no_wait:
            reporter.outcome("QUEUED", f"waiting for the required checks  {url}")
            reporter.detail(f"openspec-submit --watch {number}")
            return ACCEPTED

        reporter.stage("wait", f"required checks on {head[:7]}")
        status, label, message = wait_for_outcome(
            number,
            head,
            repository,
            reporter,
            arguments.wait_timeout,
            DEFAULT_POLL_SECONDS,
            runner,
        )
        if label:
            reporter.outcome(label, message)
        if status == TIMED_OUT:
            reporter.detail(f"openspec-submit --watch {number}")
        return status
    except ReviewRequired as error:
        _report_stop(reporter, "BLOCKED", error)
        return BLOCKED
    except SubmitError as error:
        _report_stop(reporter, "FAILED", error)
        return FAILED


if __name__ == "__main__":
    sys.exit(main())
