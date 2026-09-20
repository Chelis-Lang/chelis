#!/usr/bin/env python3
"""Establish the candidate clone and assert what its verifiers depend on.

Three verifiers run against this clone: the docs-only detector, the
candidate-lifecycle classifier and the immutable-identity recorder. Each
used to inherit whatever shape the fetches written for the previous one
happened to leave, and two defects came from exactly that. chelis#2228: the
base backstop's `--depth=1` grafted the target snapshot, so the identity
step's merge base vanished. chelis#2234: the head deepen's boundary grafted
the target tip, so the classifier demoted every force-push to an
unclassifiable rewrite. Both were a verifier depending on execution order.

So this module owns the shape instead of leaving it to order. It performs
the fetches the job needs, then asserts one written property:

    every commit the verifiers read is present, and every pair of them they
    compare has a merge base this clone can actually walk to.

A graft is what breaks that property, and only a deepening fetch removes a
graft: a depth-less fetch of a commit already present is a no-op. So when
the cheap shape does not satisfy the assertion the escalation is
`--unshallow`, and when that does not either the failure says which pair
could not be resolved and whether the clone is still shallow, rather than
leaving a later verifier to report a missing merge base as somebody's
mistake.

The assertion is only worth having if nothing undoes it afterwards, and an
invariant asserted once and then violated reads as a guarantee it no longer
gives. A fetch can only add history, with one exception: a depth-limited
fetch adds a graft. This step therefore owns every depth-limited fetch in
the job, and `scripts/test_ci_establish_candidate_clone.py` fails if another
step acquires one.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence


SHA = re.compile(r"[0-9a-f]{40}\Z")


class CloneError(ValueError):
    """The clone cannot be established, or cannot be shown to be usable."""


def _run(
    repository_path: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_path,
        check=check,
        text=True,
        capture_output=True,
    )


def _git(repository_path: Path, *arguments: str) -> str:
    try:
        return _run(repository_path, *arguments).stdout.strip()
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        raise CloneError(
            f"git {' '.join(arguments)} failed: "
            f"{detail or f'git printed no reason (exit {error.returncode})'}"
        ) from error


def _full_sha(label: str, value: str) -> str:
    if not SHA.fullmatch(value):
        raise CloneError(f"{label} must be a lowercase 40-character SHA")
    return value


def is_shallow(repository_path: Path) -> bool:
    return _git(repository_path, "rev-parse", "--is-shallow-repository") == "true"


def _reachable(repository_path: Path, sha: str) -> int:
    completed = _run(repository_path, "rev-list", "--count", sha, check=False)
    if completed.returncode != 0:
        return 0
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return 0


def _would_deepen(repository_path: Path, head: str, depth: int) -> bool:
    """True when fetching `head` at `depth` adds history rather than cutting it."""

    if not is_shallow(repository_path):
        return False
    return _reachable(repository_path, head) < depth


def first_parent(repository_path: Path, sha: str = "HEAD") -> str | None:
    """Read the first parent from the raw commit header.

    A shallow graft hides a commit's parents from `rev-parse HEAD^1` while
    leaving the commit object untouched, so `cat-file` is the only reading
    that survives the very condition this module exists to detect.
    """

    raw = _git(repository_path, "cat-file", "commit", sha)
    for line in raw.splitlines():
        if not line:
            break
        if line.startswith("parent "):
            return _full_sha("first parent", line.removeprefix("parent ").strip())
    return None


def _present(repository_path: Path, sha: str) -> bool:
    return (
        _run(repository_path, "cat-file", "-e", f"{sha}^{{commit}}", check=False)
    ).returncode == 0


def _resolves(repository_path: Path, left: str, right: str) -> bool:
    return (
        _run(repository_path, "merge-base", left, right, check=False)
    ).returncode == 0


def unmet_requirements(
    repository_path: Path,
    *,
    commits: dict[str, str],
    pairs: Sequence[tuple[str, str]],
) -> list[str]:
    """Return one line per part of the invariant this clone does not meet."""

    absent = {
        label for label, sha in commits.items()
        if not _present(repository_path, sha)
    }
    unmet = [f"{label} {commits[label]} is not in the clone" for label in sorted(absent)]
    for left, right in pairs:
        if left not in commits or right not in commits:
            continue
        # Skip a pair only when one of *its own* endpoints is missing. An
        # earlier version matched label names as substrings of the lines
        # already collected, so an absent "pre-push head" suppressed every
        # pair whose text happened to contain "head". That could not cause
        # a false pass, since the absence itself is reported, but it
        # truncated the diagnosis to the first thing that went wrong.
        if left in absent or right in absent:
            continue
        if not _resolves(repository_path, commits[left], commits[right]):
            unmet.append(
                f"{left} {commits[left]} and {right} {commits[right]} have no "
                "merge base this clone can walk to"
            )
    return unmet


def _fetch(
    repository_path: Path, *arguments: str, label: str, notes: list[str]
) -> None:
    completed = _run(repository_path, "fetch", *arguments, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        notes.append(f"{label} failed: {detail.splitlines()[-1] if detail else '?'}")


def establish(
    repository_path: Path,
    *,
    base: str,
    head: str,
    before: str | None,
    target_tip: str | None,
    commits: int,
    remote: str = "origin",
) -> tuple[dict[str, str], list[str]]:
    """Fetch what the verifiers read, then make the invariant hold.

    Returns the commits it established and the notes worth printing. Raises
    `CloneError` when the invariant still does not hold after deepening.
    """

    repository_path = Path(repository_path)
    notes: list[str] = []
    wanted = {"base": _full_sha("base", base), "head": _full_sha("head", head)}
    if before is not None:
        wanted["pre-push head"] = _full_sha("pre-push head", before)
    if target_tip is not None:
        wanted["target tip"] = _full_sha("target tip", target_tip)

    # The one depth-limited fetch in this job, and the reason this module
    # owns the invariant: its boundary is what grafts a commit. It runs
    # only when it would actually deepen. Against a complete clone it
    # would *create* the graft this module exists to remove; against a
    # shallow clone that already reaches further than the requested depth
    # it would SHORTEN it, dropping reachable commits and adding a second
    # graft. Neither is a cheaper way to reach the same place. Today the
    # candidate checkout is depth one so only the first case is reachable,
    # but the rule is written for the depth, not for today's depth.
    if commits > 0 and _would_deepen(repository_path, head, commits + 1):
        _fetch(
            repository_path,
            f"--depth={commits + 1}",
            remote,
            head,
            label="head deepen",
            notes=notes,
        )
    parent = first_parent(repository_path)
    if parent is not None:
        wanted["candidate first parent"] = parent
    for sha in dict.fromkeys(wanted.values()):
        if sha != head:
            _fetch(
                repository_path,
                "--no-tags",
                remote,
                sha,
                label=f"fetch {sha}",
                notes=notes,
            )

    pairs = [
        ("base", "head"),
        ("candidate first parent", "head"),
        ("pre-push head", "base"),
        ("pre-push head", "head"),
        ("pre-push head", "target tip"),
        ("head", "target tip"),
    ]
    unmet = unmet_requirements(repository_path, commits=wanted, pairs=pairs)
    if not unmet:
        return wanted, notes

    # Only a deepening fetch removes a graft; a depth-less fetch of a commit
    # already present is a no-op, which is why chelis#2230's backstop could
    # not reach chelis#2234's graft.
    notes.append(
        "the cheap clone shape did not satisfy the invariant, deepening: "
        + "; ".join(unmet)
    )
    if is_shallow(repository_path):
        _fetch(
            repository_path,
            "--unshallow",
            remote,
            *dict.fromkeys(wanted.values()),
            label="unshallow",
            notes=notes,
        )
    unmet = unmet_requirements(repository_path, commits=wanted, pairs=pairs)
    if unmet:
        shape = "shallow" if is_shallow(repository_path) else "complete"
        raise CloneError(
            "the candidate clone does not support the verifiers that read it, "
            f"and it is {shape} after deepening: " + "; ".join(unmet)
        )
    return wanted, notes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-path", type=Path, default=Path("."))
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--before")
    parser.add_argument("--target-tip")
    parser.add_argument("--commits", type=int, default=0)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--github-output", type=Path)
    return parser


def _write_output(path: Path | None, lines: Iterable[str]) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        established, notes = establish(
            arguments.repository_path,
            base=arguments.base,
            head=arguments.head,
            before=arguments.before,
            target_tip=arguments.target_tip,
            commits=arguments.commits,
            remote=arguments.remote,
        )
    except CloneError as error:
        for note in getattr(error, "notes", []):
            print(note, file=sys.stderr)
        print(f"CANDIDATE CLONE: FAIL: {error}", file=sys.stderr)
        _write_output(arguments.github_output, ["clone_invariant=failed"])
        return 1
    for note in notes:
        print(note)
    parent = established.get("candidate first parent", "")
    _write_output(
        arguments.github_output,
        ["clone_invariant=established", f"first_parent={parent}"],
    )
    print(
        "CANDIDATE CLONE: PASS: "
        + ", ".join(f"{label} {sha[:9]}" for label, sha in established.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
