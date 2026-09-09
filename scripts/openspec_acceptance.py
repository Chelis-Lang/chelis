#!/usr/bin/env python3
"""Decide whether a change may land without human review.

The maintainer authorized automated acceptance for OpenSpec **document**
changes, including normative `openspec/specs/**` text. Code, tooling, and
policy changes keep the ordinary review and test path. This module is the
single trusted decision point for that split.

Trust model
-----------

This script is the only thing standing between "a pull request touched a
file" and "a machine merged it". Two properties make that safe:

1. **It runs trusted code on trusted data.** The workflow runs this file
   from the pull request's *base* revision, never from the head, and feeds
   it `git diff --raw` metadata (statuses, modes, paths) rather than file
   contents. A pull request cannot edit the classifier that judges it.
2. **It fails closed.** Every unexpected input -- an empty diff, an
   unparseable record, a combined merge diff, a symlink, a submodule
   pointer, an executable bit, a rename out of the document tree, an
   unknown status letter -- returns "review required". A false review
   costs a human a click; a false acceptance merges unreviewed code.

The path allowlist deliberately excludes `openspec/config.yaml` and every
path under `.github/`, `scripts/`, and `spec/`. Those inputs decide *how*
acceptance itself behaves, so a change to one of them can never be judged
by the version of the policy it is trying to replace.

Structural validation runs elsewhere and answers a different question.
A pull request that this module accepts is confined to OpenSpec document
paths; it is not thereby correct, wanted, or consistent with `spec/**`.

Usage:
    python3 scripts/openspec_acceptance.py --base <sha> --head <sha>
    python3 scripts/openspec_acceptance.py --base origin/main   # worktree
    git diff --raw -z -M <base> <head> \\
        | python3 scripts/openspec_acceptance.py --raw-stdin

Writes `verdict=auto|review` and `reason_count=<integer>` to
`$GITHUB_OUTPUT` when that variable is set, and always prints the verdict
and its escaped reasons to stdout. No repository-derived text reaches the
step output: see `_emit`. Exit status is `0` for a completed
classification, `2` for an operational failure, and `1` only when
`--require-auto` is given and the verdict is `review`.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CLASSIFIED = 0
REVIEW_REQUIRED = 1
OPERATIONAL_FAILURE = 2

GIT_TIMEOUT_SECONDS = 120
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
RAW_HEADER = re.compile(
    r"^:(?P<src_mode>[0-7]{6}) (?P<dst_mode>[0-7]{6}) "
    r"(?P<src_sha>[0-9a-f]+) (?P<dst_sha>[0-9a-f]+) "
    r"(?P<status>[A-Z])(?P<score>[0-9]{0,3})$"
)

# A tracked regular file. Every other mode is refused: 100755 grants the
# executable bit, 120000 is a symlink whose target is content the document
# rules never inspected, 160000 is a submodule pointer, and 040000 is a
# tree. None of those spellings is a document.
REGULAR_MODE = "100644"
ABSENT_MODE = "000000"

SPEC_ROOT = "openspec/specs/"
CHANGE_ROOT = "openspec/changes/"
PROJECT_DOC = "openspec/project.md"
DOCUMENT_SUFFIX = ".md"
CHANGE_SCAFFOLD = ".openspec.yaml"

# Inputs that decide acceptance policy itself. Listing them is redundant --
# they already fail the allowlist -- but the named reason is better than
# "outside the boundary" when a contributor edits one by accident.
POLICY_PATHS = frozenset(
    {
        "openspec/config.yaml",
        ".github/workflows/openspec-autoland.yml",
        "scripts/openspec_acceptance.py",
        "scripts/openspec_submit.py",
        "scripts/check_openspec.py",
    }
)

# Paths whose content decides how acceptance itself behaves. These are
# compared by Git object identity between head and base, not by diff: a
# diff starts at a merge base, so a head that merely predates a change to
# `.github/` shows no difference there while carrying an older workflow.
# Object identity answers the question that actually matters.
#
# The set covers what CI runs and what it runs *inside*. Identical
# `scripts/` content proves little on its own, because every real CI step
# executes through `devenv-retry --profile ci shell`, so the Devenv, Nix,
# and toolchain inputs decide which interpreter that identical code runs
# under. Cargo manifests and lockfiles are deliberately excluded: they
# select library versions rather than steering CI execution, and they
# change often enough that including them would refuse most branches.
GOVERNANCE_PATHS: tuple[str, ...] = (
    ".github",
    "scripts",
    "openspec/config.yaml",
    "devenv.nix",
    "devenv.yaml",
    "devenv.lock",
    "devenv",
    "flake.nix",
    "flake.lock",
    "nix",
    ".cargo",
    "rust-toolchain.toml",
    ".gitattributes",
    ".gitmodules",
)

SPEC_DOCUMENT = "spec"
CHANGE_DOCUMENT = "change"
PROJECT_DOCUMENT = "project"

# How many reasons a summary shows. The verdict never depends on this.
REASON_LIMIT = 10
MAX_DISPLAY_PATH = 200


class BoundaryError(RuntimeError):
    """The diff could not be read as a set of ordinary file changes."""


@dataclass(frozen=True)
class DiffRecord:
    """One `git diff --raw` record: what happened to which path."""

    status: str
    src_mode: str
    dst_mode: str
    paths: tuple[str, ...]

    @property
    def source(self) -> str:
        return self.paths[0]

    @property
    def target(self) -> str:
        return self.paths[-1]


@dataclass(frozen=True)
class Verdict:
    """The acceptance decision plus every reason it was not automatic."""

    auto: bool
    reasons: tuple[str, ...]
    paths: tuple[str, ...]

    @property
    def name(self) -> str:
        return "auto" if self.auto else "review"

    @property
    def reason(self) -> str:
        """A bounded one-line summary fit for a log line or an error.

        A change set that misses the boundary by a whole branch produces
        thousands of reasons. Reporting all of them buries the first one,
        which is the one a reader acts on. The full list stays in
        `reasons` for any caller that wants it.
        """
        if self.auto:
            return "every changed path is an OpenSpec document"
        shown = "; ".join(self.reasons[:REASON_LIMIT])
        remaining = len(self.reasons) - REASON_LIMIT
        if remaining > 0:
            return f"{shown}; and {remaining} more"
        return shown


def display(path: str) -> str:
    """Render a path safely for a human-readable reason.

    A Git filename may contain a newline. A reason built from a raw path
    is copied into logs, job summaries, and error text, so every control
    character is escaped before it leaves this module.
    """
    escaped = path.encode("unicode_escape").decode("ascii")
    if len(escaped) > MAX_DISPLAY_PATH:
        return escaped[:MAX_DISPLAY_PATH] + "..."
    return escaped


def is_safe_path(path: str) -> bool:
    """True when `path` is an ordinary relative repository path.

    Refuses absolute paths, backslash separators, control characters, and
    any `.`/`..`/empty component, so no allowlist rule can be satisfied by
    a spelling that resolves somewhere else.
    """
    if not path or path != path.strip():
        return False
    if path.startswith("/") or "\\" in path:
        return False
    if any(character < " " or character == "\x7f" for character in path):
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def document_class(path: str) -> str | None:
    """Name the document kind for `path`, or `None` when it is not one."""
    if not is_safe_path(path) or path in POLICY_PATHS:
        return None
    if path == PROJECT_DOC:
        return PROJECT_DOCUMENT
    if path.startswith(SPEC_ROOT):
        return SPEC_DOCUMENT if path.endswith(DOCUMENT_SUFFIX) else None
    if path.startswith(CHANGE_ROOT):
        name = path.rsplit("/", 1)[-1]
        if path.endswith(DOCUMENT_SUFFIX) or name == CHANGE_SCAFFOLD:
            return CHANGE_DOCUMENT
        return None
    return None


def parse_raw_diff(stream: str) -> tuple[DiffRecord, ...]:
    """Parse `git diff --raw -z` output into records.

    Raises `BoundaryError` rather than skipping anything it cannot read:
    a record this parser does not understand is exactly the case where
    guessing is unsafe.
    """
    fields = stream.split("\0")
    if fields and fields[-1] == "":
        fields.pop()

    records: list[DiffRecord] = []
    index = 0
    while index < len(fields):
        header = fields[index]
        if header.startswith("::"):
            raise BoundaryError(
                "combined merge diff records cannot be classified per path"
            )
        match = RAW_HEADER.match(header)
        if match is None:
            raise BoundaryError(f"unreadable diff record header: {header!r}")
        status = match.group("status")
        wanted = 2 if status in ("R", "C") else 1
        if index + wanted >= len(fields):
            raise BoundaryError(f"diff record is missing a path: {header!r}")
        paths = tuple(fields[index + 1 : index + 1 + wanted])
        if any(not path for path in paths):
            raise BoundaryError(f"diff record has an empty path: {header!r}")
        records.append(
            DiffRecord(
                status=status,
                src_mode=match.group("src_mode"),
                dst_mode=match.group("dst_mode"),
                paths=paths,
            )
        )
        index += 1 + wanted
    return tuple(records)


def _mode_reasons(record: DiffRecord) -> list[str]:
    """Reject every file mode that is not a plain tracked regular file."""
    reasons: list[str] = []
    expected = {
        "A": (ABSENT_MODE, REGULAR_MODE),
        "M": (REGULAR_MODE, REGULAR_MODE),
        "D": (REGULAR_MODE, ABSENT_MODE),
        "R": (REGULAR_MODE, REGULAR_MODE),
    }[record.status]
    if (record.src_mode, record.dst_mode) != expected:
        reasons.append(
            f"{display(record.target)}: file mode "
            f"{record.src_mode}->{record.dst_mode} "
            f"is not a regular-file {record.status} change"
        )
    return reasons


def _record_reasons(record: DiffRecord) -> list[str]:
    """Every reason this single record blocks automatic acceptance."""
    if record.status not in ("A", "M", "D", "R"):
        return [
            f"{display(record.target)}: change status {record.status} is "
            "outside the add, modify, delete, and rename set"
        ]

    reasons = _mode_reasons(record)
    for path in record.paths:
        if path in POLICY_PATHS:
            reasons.append(f"{display(path)}: changes acceptance policy itself")
        elif document_class(path) is None:
            reasons.append(
                f"{display(path)}: outside the OpenSpec document boundary"
            )

    source_class = document_class(record.source)
    target_class = document_class(record.target)
    if record.status == "D" and source_class in (SPEC_DOCUMENT, PROJECT_DOCUMENT):
        reasons.append(
            f"{display(record.source)}: deletes a normative OpenSpec document"
        )
    if (
        record.status == "R"
        and source_class == SPEC_DOCUMENT
        and target_class != SPEC_DOCUMENT
    ):
        reasons.append(
            f"{display(record.source)}: moves a capability specification out "
            f"of {SPEC_ROOT}"
        )
    # A rename away from a normative slot removes the document from that
    # slot as completely as a deletion does, and a deletion here already
    # routes to review. Without this, `openspec/project.md` could be moved
    # to any other document path unreviewed, because both ends are
    # documents and no other rule objects.
    if (
        record.status == "R"
        and source_class == PROJECT_DOCUMENT
        and record.target != PROJECT_DOC
    ):
        reasons.append(
            f"{display(record.source)}: moves a normative OpenSpec document "
            "out of its slot"
        )
    return reasons


def classify(records: Sequence[DiffRecord]) -> Verdict:
    """Decide whether `records` may land without human review."""
    paths = tuple(path for record in records for path in record.paths)
    if not records:
        return Verdict(
            auto=False,
            reasons=("the change set is empty, so no boundary was proven",),
            paths=paths,
        )
    reasons: list[str] = []
    for record in records:
        reasons.extend(_record_reasons(record))
    # Preserve first-seen order while removing duplicates so one bad path
    # named by two records is reported once.
    unique = tuple(dict.fromkeys(reasons))
    return Verdict(auto=not unique, reasons=unique, paths=paths)


def classify_stream(stream: str) -> Verdict:
    """Classify raw diff text, turning a parse failure into a verdict."""
    try:
        records = parse_raw_diff(stream)
    except BoundaryError as error:
        return Verdict(auto=False, reasons=(str(error),), paths=())
    return classify(records)


def valid_revision(value: str) -> bool:
    """Accept only the two revision spellings the callers actually use."""
    return value == "origin/main" or FULL_SHA.fullmatch(value) is not None


def read_raw_diff(
    repository: Path,
    base: str,
    head: str | None,
    runner=None,
) -> str:
    """Run `git diff --raw` in `repository` and return its output.

    With `head` omitted the comparison ends at the working tree, which is
    what the local submit command needs before anything is committed.
    """
    if not valid_revision(base):
        raise BoundaryError("base must be origin/main or a lowercase full SHA")
    if head is not None and not valid_revision(head):
        raise BoundaryError("head must be origin/main or a lowercase full SHA")
    git = os.environ.get("GIT_BIN") or "git"
    # `base...head` asks for the changes on `head` since the merge base.
    # A plain two-dot range would also report every commit `main` gained
    # after the branch started, which are changes the pull request did not
    # make. Measured on this repository across 30 recent merged pull
    # requests: `base.sha` was strictly AHEAD of the merge base in 13 of
    # them, and two-dot would have attributed up to 136 extra paths to one
    # of those changes. (`base.sha` cannot be behind the merge base --
    # `merge-base(base, head)` is an ancestor-or-equal of `base` by
    # construction -- so "ahead" is the only direction that can arise.)
    revisions = f"{base}...{head}" if head is not None else base
    # The trailing `--` keeps a revision from being read as a filename when
    # a real path shares its spelling.
    command = [git, "diff", "--raw", "-z", "-M", "--no-color", revisions, "--"]
    execute = runner or subprocess.run
    try:
        completed = execute(
            command,
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise BoundaryError(f"cannot run Git: {error}") from error
    if getattr(completed, "returncode", None) != 0:
        stderr = (getattr(completed, "stderr", "") or "").strip()
        raise BoundaryError(f"Git diff failed: {stderr or 'unknown error'}")
    return getattr(completed, "stdout", "") or ""


def _object_id(
    repository: Path, revision: str, path: str, execute
) -> str | None:
    """Resolve one Git object id, or `None` when it cannot be read."""
    git = os.environ.get("GIT_BIN") or "git"
    try:
        completed = execute(
            [git, "rev-parse", "--verify", "--quiet", f"{revision}:{path}"],
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    if getattr(completed, "returncode", None) != 0:
        return None
    value = (getattr(completed, "stdout", "") or "").strip()
    return value or None


def governance_differences(
    repository: Path,
    base: str,
    head: str,
    runner=None,
) -> tuple[str, ...]:
    """Governance paths whose content differs between `base` and `head`.

    An empty result means the head carries byte-identical workflow, script,
    and OpenSpec configuration content. That is what makes a result
    produced by a head-side workflow meaningful: the workflow that produced
    it is the base's workflow.

    An object that cannot be read counts as a difference. Failing to prove
    sameness is not the same as proving it.
    """
    for revision in (base, head):
        if not valid_revision(revision):
            raise BoundaryError(
                "governance revisions must be origin/main or a lowercase full SHA"
            )
    execute = runner or subprocess.run
    differences: list[str] = []
    resolved = 0
    for path in GOVERNANCE_PATHS:
        base_id = _object_id(repository, base, path, execute)
        head_id = _object_id(repository, head, path, execute)
        resolved += base_id is not None
        if base_id is None and head_id is None:
            # Absent on both sides is the same content: nothing.
            continue
        if base_id != head_id:
            differences.append(path)
    if not resolved:
        # Every lookup failed, so nothing was compared. That is a broken
        # Git invocation, not a clean bill of health.
        raise BoundaryError(
            "no governance path could be read; the comparison proved nothing"
        )
    return tuple(differences)


def _emit(verdict: Verdict) -> None:
    """Report the verdict without letting a filename become a step output.

    `$GITHUB_OUTPUT` is a `key=value` file, and a Git filename may contain
    a newline. Writing a reason built from a path would let a pull request
    append its own `verdict=auto` line, which the runner resolves
    last-wins. So only a closed vocabulary and an integer are written
    there. The human-readable reasons go to stdout, escaped.
    """
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"verdict={verdict.name}\n")
            handle.write(f"reason_count={len(verdict.reasons)}\n")
    print(f"verdict={verdict.name}")
    print(f"reason={verdict.reason}")
    if not verdict.auto:
        for reason in verdict.reasons[:REASON_LIMIT]:
            print(f"  - {reason}")
        remaining = len(verdict.reasons) - REASON_LIMIT
        if remaining > 0:
            print(f"  - and {remaining} more")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a diff as an OpenSpec document change that may land "
            "automatically, or as a change that requires human review."
        )
    )
    parser.add_argument("--repo", default=".", help="repository to inspect")
    parser.add_argument("--base", help="base revision: origin/main or a full SHA")
    parser.add_argument("--head", help="head revision; omit to use the worktree")
    parser.add_argument(
        "--raw-stdin",
        action="store_true",
        help="read `git diff --raw -z` output from standard input",
    )
    parser.add_argument(
        "--require-auto",
        action="store_true",
        help="exit 1 when the verdict is review",
    )
    parser.add_argument(
        "--require-identical-governance",
        action="store_true",
        help=(
            "also verify that the head carries byte-identical workflow, "
            "script, and OpenSpec configuration content"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    if arguments.require_identical_governance and not (
        arguments.base and arguments.head
    ):
        print(
            "openspec_acceptance: --require-identical-governance needs "
            "--base and --head",
            file=sys.stderr,
        )
        return OPERATIONAL_FAILURE

    if arguments.raw_stdin:
        if arguments.base or arguments.head:
            print(
                "openspec_acceptance: --raw-stdin takes no revision",
                file=sys.stderr,
            )
            return OPERATIONAL_FAILURE
        verdict = classify_stream(sys.stdin.read())
    else:
        if not arguments.base:
            print("openspec_acceptance: --base is required", file=sys.stderr)
            return OPERATIONAL_FAILURE
        try:
            stream = read_raw_diff(
                Path(arguments.repo), arguments.base, arguments.head
            )
            verdict = classify_stream(stream)
            if arguments.require_identical_governance:
                differences = governance_differences(
                    Path(arguments.repo), arguments.base, arguments.head
                )
                if differences:
                    verdict = Verdict(
                        auto=False,
                        reasons=verdict.reasons
                        + tuple(
                            f"{path}: governance content differs from the base"
                            for path in differences
                        ),
                        paths=verdict.paths,
                    )
        except BoundaryError as error:
            print(f"openspec_acceptance: {error}", file=sys.stderr)
            return OPERATIONAL_FAILURE
    _emit(verdict)
    if arguments.require_auto and not verdict.auto:
        return REVIEW_REQUIRED
    return CLASSIFIED


if __name__ == "__main__":
    sys.exit(main())
