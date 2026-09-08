#!/usr/bin/env python3
"""Answer "is this worktree free right now?" with evidence, not a claim.

Background (chelis#1568): every red-team brief in the 2026-09-02 fleet run
asserted "the worktree is clean, the target is free" by transcribing the
author's last status report. That is a claim about the past, and five
collisions followed. In the worst one a reviewer entered a worktree where the
author's `scripts/gate.py --local` had been running for five minutes, wrote a
probe file, checked out an old revision over the working tree, and the gate
run then failed at stage 6 of 24 on a compile error the reviewer had caused.

Three signals that would have prevented it already existed and nobody
consulted them: the advisory gate lease, a scoped `reap_orphans.py` listing,
and the worktree's own `target/gate-reports/`. This script reads all three,
plus the git state, and prints one block a brief pastes INSTEAD of an
assertion. The block carries the instant it was produced and the command that
produced it, so a stale paste is visibly stale.

Usage:
    .venv/bin/python scripts/worktree_status.py                # this checkout
    .venv/bin/python scripts/worktree_status.py --path <other worktree>
    .venv/bin/python scripts/worktree_status.py --json
    .venv/bin/python scripts/worktree_status.py --quiet        # verdict only

Exit codes, which are the machine channel:

    0  FREE       no owner, and the tree is clean
    1  BUSY       positive evidence that something owns this checkout now
    2  UNKNOWN    no positive busy signal AND an evidence source failed
    3  NOT CLEAN  no owner, but the tree has uncommitted or untracked work

Precedence is BUSY > UNKNOWN > NOT CLEAN > FREE. The rule behind it: positive
evidence is never downgraded by a failure somewhere else, because a broken
`ps` does not make a lease sidecar less true; absence of evidence always is,
because a probe that could not read `ps` has not established that nobody owns
the checkout and must never print FREE.

What this can and cannot prove
------------------------------
A `--local` or full gate run is proven by the advisory lease, which is an
`fcntl.flock` the kernel releases only when the holder's last descriptor
closes. That is authoritative. A `--fast` run and a `--no-lease` run take no
lease, so they are caught instead by the process scan: `gate.py` spawns every
child with `cwd=repo_root`, so `reap_orphans.match_repo_processes` sees the
cargo/rustc/nextest children even when it does not recognise the Python parent
(this script adds its own matcher for that parent).

There is deliberately no claim that a gate run is in flight based on
`target/gate-reports/`. `gate.py` writes a summary from exactly one place, its
`main`'s `finally` block, and names the file from the run's END timestamp, so
that directory holds finished runs only and has no start marker of any kind.
The newest report is printed under an explicit "last finished run" label and
is never evidence about now.

Read-only discipline
--------------------
The reviewer who "wrote a probe file" is the failure this script exists to
prevent, so it writes nothing anywhere: no temp file, no report, no index
refresh. Every git call is one of `READ_ONLY_GIT_SUBCOMMANDS` and carries
`--no-optional-locks`, which matters for a measured reason. On this repository's
worktree (3,618 tracked files), with at least one tracked file holding stale
stat data in the index, a plain `git status --porcelain` rewrote the index
file while the `--no-optional-locks` form left its mtime untouched; when the
index stat data was already fresh, neither form rewrote it. So the flag is not
cosmetic under exactly the conditions a probe meets after somebody's formatter
pass, and it costs nothing when the index is settled.

Why there is no content re-hashing here
---------------------------------------
An earlier design proposed comparing `git ls-files -s` blob ids against
`git hash-object --stdin-paths` to tell a stat-dirty path from a real
modification. That is unnecessary: git already does the content comparison
itself, and a file rewritten with identical bytes is reported clean by
`git status --porcelain` with and without `--no-optional-locks`, and after a
same-second racy write, while a real edit is reported ` M`. Re-deriving that
would also be a WORSE oracle, because any future clean/smudge filter would
make an unfiltered hash disagree with the index blob and invent a
modification. The stat-dirty count below is therefore reported as what it is:
a note about index freshness, never part of the dirty verdict.

Per repo policy this is Python and uv-managed like every other script. It is
NOT in the bootstrap-free pair (`reap_orphans.py`, `preflight_exec_probe.py`):
those two run before a working project environment exists, whereas this one is
run by an agent that already has a managed interpreter and is asking about a
different directory. Tests: `scripts/test_worktree_status.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

# `gate.py` owns the lease location, the lease probe, the holder rendering and
# the report directory; `reap_orphans.py` owns the process layer. Both are
# imported rather than reimplemented: a second copy of the lease path rule
# would silently probe a different file than the gate takes, and a second copy
# of the path-boundary matcher is exactly how a sibling checkout such as
# `<repo>-165` starts matching. `gate.py` is safe to import because its uv
# re-exec (`ensure_managed_runtime`) is called only under `__main__`; loading
# it costs 17 ms and runs no subprocess.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import gate  # noqa: E402
import reap_orphans as reap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

# 2 added `probed_path` and `degraded` to the payload. Nothing consumes this
# yet, but a version that does not move when the shape does is worse than no
# version at all.
SCHEMA_VERSION = 2

EXIT_FREE = 0
EXIT_BUSY = 1
EXIT_UNKNOWN = 2
EXIT_NOT_CLEAN = 3

# Every signal that can make this worktree BUSY. The verdict tags each reason
# with the signal that produced it, so the test suite can require one true
# positive AND one near miss per signal. Near-miss coverage previously existed
# only where a reviewer had already attacked, which is how a signal ships with
# a boundary nobody has probed.
SIGNAL_GATE_LEASE = "gate-lease"
SIGNAL_GATE_PROCESS = "gate-process"
SIGNAL_BUILD_PROCESS = "build-process"
SIGNAL_GIT_OPERATION = "git-operation"
BUSY_SIGNALS = (
    SIGNAL_GATE_LEASE,
    SIGNAL_GATE_PROCESS,
    SIGNAL_BUILD_PROCESS,
    SIGNAL_GIT_OPERATION,
)

VERDICT_FREE = "FREE"
VERDICT_BUSY = "BUSY"
VERDICT_UNKNOWN = "UNKNOWN"
VERDICT_NOT_CLEAN = "NOT CLEAN"

EXIT_FOR_VERDICT = {
    VERDICT_FREE: EXIT_FREE,
    VERDICT_BUSY: EXIT_BUSY,
    VERDICT_UNKNOWN: EXIT_UNKNOWN,
    VERDICT_NOT_CLEAN: EXIT_NOT_CLEAN,
}

# The complete set of git subcommands this script may run. Frozen on purpose:
# the whole point of the tool is that probing a worktree must not change it,
# and `scripts/test_worktree_status.py` asserts every argv against this set.
READ_ONLY_GIT_SUBCOMMANDS = frozenset({"rev-parse", "status", "ls-files"})

# Never acceptable in an argv, even as an argument to an allowed subcommand.
FORBIDDEN_GIT_TOKENS = frozenset(
    {
        "add",
        "checkout",
        "clean",
        "commit",
        "fetch",
        "gc",
        "prune",
        "reset",
        "restore",
        "stash",
        "switch",
        "update-index",
        "update-ref",
        "write-tree",
    }
)

NO_OPTIONAL_LOCKS = "--no-optional-locks"

GIT_TIMEOUT_SECONDS = 60.0

# Porcelain v1 two-character codes that mean an unmerged path.
UNMERGED_CODES = frozenset({"DD", "AU", "UD", "UA", "DU", "AA", "UU"})

# Files and directories in the resolved git dir that mean a git operation is
# part-way through. Each genuinely persists until its operation finishes or is
# aborted, so its presence supports the sentence beside it. That is the
# property `index.lock` lacks, which is why it is handled separately below.
GIT_OPERATION_MARKERS = (
    ("MERGE_HEAD", "a merge is in progress"),
    ("CHERRY_PICK_HEAD", "a cherry-pick is in progress"),
    ("REVERT_HEAD", "a revert is in progress"),
    ("BISECT_LOG", "a bisect is in progress"),
    ("rebase-merge", "a rebase is in progress"),
    ("rebase-apply", "a rebase or am is in progress"),
)

# `index.lock` is deliberately NOT in that table. Its existence does not prove
# a git command is writing: git removes it on a normal exit and on SIGINT, but
# a SIGKILLed git leaves it behind forever, and nothing kernel-backed holds it
# the way `flock` holds the gate lease. Reading existence as liveness made a
# free worktree report BUSY permanently, which is the same substitution this
# file made elsewhere: a cheap observable standing in for the property the
# verdict depends on. It is reported as what it is and does not drive the
# verdict. The other markers stay, because a rebase or a merge genuinely
# persists until it is finished or aborted.
INDEX_LOCK_NAME = "index.lock"
INDEX_LOCK_NOTE = (
    "a stale lock file is present, so a git command may have crashed; git "
    "operations here will fail until it is removed"
)

# A gate run is a Python interpreter whose SCRIPT ARGUMENT is this worktree's
# gate. `reap_orphans` cannot see it: `BUILD_TOOL_NAMES` has no Python entry
# and the interpreter does not live under `target/`. Deliberately NOT fixed by
# adding `python` to that set, which would put every gate run into the
# reaper's SIGKILL list.
GATE_SCRIPT_RELATIVE = "scripts/gate.py"

PYTHON_BASENAME_PREFIX = "python"

# Python options that consume the following token, so what follows them is an
# option argument rather than the script. `-c` and `-m` are absent on purpose:
# both mean there is NO script argument at all, and every later token belongs
# to the command or the module, so they end the search rather than skip one.
PYTHON_OPTIONS_WITH_ARGUMENT = frozenset({"-W", "-X", "--check-hash-based-pycs"})
PYTHON_OPTIONS_WITHOUT_A_SCRIPT = frozenset({"-c", "-m"})


def script_argument(command: str) -> str | None:
    """The script a Python command line runs, or None if it runs no script.

    Why this is not a substring test. Asking whether a command line CONTAINS
    `scripts/gate.py` answers a different question from whether the process IS
    a gate run, and that gap is where several defects in this file have lived.
    A shell that names the script, an editor open on it, a linter invoked on
    it, and `python3 -c '...' scripts/gate.py` all contain the string and none
    is a gate run. Narrowing the substring cannot close that, because
    containment is not the property; the token's POSITION is.

    `ps -o command=` flattens argv into one string, so this splits on
    whitespace and walks it the way Python's own launcher does: skip an
    option, consume the argument of an option that takes one, stop outright at
    `-c` or `-m` because neither has a script, and return the first remaining
    token.

    This walk is a MODEL of Python's option grammar and models are incomplete.
    Python accepts a value-taking short option clustered onto others, so
    `-uX importtime` is one token this exact-spelling table does not know, and
    the walk then returns the option's value as the script. A path containing
    whitespace defeats the split the same way. Chasing the grammar is
    unbounded: attachment, clustering, `=` forms, and whatever a future
    release adds. So the caller does not trust a negative answer from this
    function on its own; see `match_gate_processes`, which degrades to UNKNOWN
    rather than to FREE when the walk disagrees with a line that names the
    gate. That converts an unbounded parsing problem into a bounded one.
    """
    tokens = command.split()
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in PYTHON_OPTIONS_WITHOUT_A_SCRIPT:
            return None
        if token in PYTHON_OPTIONS_WITH_ARGUMENT:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return None


class GitQueryError(RuntimeError):
    """A git query failed, was not runnable, or timed out."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_read_only(args: Sequence[str]) -> None:
    """Refuse to run anything outside the frozen read-only set. This is a
    belt-and-braces guard on top of the test: a future edit that adds a
    mutating call fails here at run time rather than quietly rewriting a
    worktree somebody else is using."""
    subcommand = next((arg for arg in args if not arg.startswith("-")), None)
    if subcommand not in READ_ONLY_GIT_SUBCOMMANDS:
        raise GitQueryError(
            f"refusing to run `git {subcommand}`: not in the read-only set "
            f"{sorted(READ_ONLY_GIT_SUBCOMMANDS)}"
        )
    forbidden = sorted(set(args) & FORBIDDEN_GIT_TOKENS)
    if forbidden:
        raise GitQueryError(
            f"refusing to run a git command carrying {forbidden}"
        )


def git_query(
    args: Sequence[str],
    *,
    worktree: Path,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    allow_optional_locks: bool = False,
) -> str:
    """Run one read-only git query in `worktree` and return its stdout.

    `--no-optional-locks` is passed on every call, not just the ones that
    could refresh the index, so there is a single rule to state and to test.
    A git old enough to reject the flag is handled by one retry without it;
    the flag has existed since git 2.15 (2017), so that path is insurance
    rather than an expected case.
    """
    _check_read_only(args)
    prefix = [] if allow_optional_locks else [NO_OPTIONAL_LOCKS]
    argv = ["git", "-C", str(worktree), *prefix, *args]
    try:
        result = runner(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitQueryError(f"`{' '.join(argv)}` could not run: {exc}") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if not allow_optional_locks and _rejects_no_optional_locks(stderr):
            return git_query(
                args,
                worktree=worktree,
                runner=runner,
                allow_optional_locks=True,
            )
        raise GitQueryError(
            f"`{' '.join(argv)}` exited {result.returncode}: "
            f"{stderr or '(no stderr)'}"
        )
    return result.stdout


def _rejects_no_optional_locks(stderr: str) -> bool:
    # Git names the offending flag WITHOUT its leading dashes: "error: unknown
    # option `no-optional-locks'". Matching on the dashed spelling would never
    # fire, so the bare name is what is looked for.
    lowered = stderr.lower()
    return NO_OPTIONAL_LOCKS.lstrip("-") in lowered and (
        "unknown option" in lowered or "unknown switch" in lowered
    )


Query = Callable[..., str]


def git_facts(worktree: Path, *, query: Query = git_query) -> dict:
    """HEAD, branch, and the resolved git directories.

    A linked worktree keeps `.git` as a FILE, so the git directory must come
    from `rev-parse --absolute-git-dir` rather than from `<worktree>/.git`;
    the in-progress-operation markers below are read from that resolved path.
    """
    out = query(
        [
            "rev-parse",
            "--absolute-git-dir",
            "--git-common-dir",
            "--show-toplevel",
            # Order matters: `--abbrev-ref` is a modifier that applies to
            # every later revision argument, so the raw HEAD must come first
            # or both lines come back as the branch name.
            "HEAD",
            "--abbrev-ref",
            "HEAD",
        ],
        worktree=worktree,
    )
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if len(lines) != 5:
        # `git rev-parse` separates its answers with newlines, and a path may
        # contain one, so a path with an embedded newline yields more lines
        # than values and `lines[:5]` would silently misalign. Since the
        # re-anchor below feeds `toplevel` into every later query, a misaligned
        # parse would fabricate a worktree rather than merely misprint one.
        # There is no way to tell the two apart from this output, so say so.
        raise GitQueryError(
            f"`git rev-parse` returned {len(lines)} lines, expected exactly 5, "
            f"so its values cannot be told apart: {lines!r}"
        )
    git_dir, common_dir, toplevel, head, branch = lines
    common = Path(common_dir)
    if not common.is_absolute():
        # `--git-common-dir` is reported relative to the directory git RAN in,
        # which is the `-C` path, not relative to `--absolute-git-dir`. A
        # primary checkout probed at its root answers a bare `.git`, so
        # anchoring to the git dir would build `<root>/.git/.git`, a path that
        # does not exist, and every primary checkout would then be labelled
        # linked.
        common = (worktree / common).resolve()
    return {
        "git_dir": git_dir,
        "git_common_dir": str(common),
        "toplevel": toplevel,
        "branch": None if branch == "HEAD" else branch,
        "detached": branch == "HEAD",
        "head": head,
        "linked": Path(git_dir).resolve() != common.resolve(),
    }


def parse_porcelain_z(text: str) -> list[tuple[str, str]]:
    """Parse `git status --porcelain=v1 -z` into (XY, path) pairs.

    The `-z` form is NUL-terminated with raw (unquoted) paths, and a rename or
    copy entry spends TWO NUL-terminated fields: `XY dest\\0orig\\0`. The
    origin path is consumed and dropped; the destination is what a reviewer
    needs to see.
    """
    fields = text.split("\0")
    entries: list[tuple[str, str]] = []
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if len(field) < 4:
            continue
        code = field[:2]
        path = field[3:]
        if code[0] in ("R", "C"):
            index += 1  # the origin path is its own NUL-terminated field
        entries.append((code, path))
    return entries


def classify_entries(entries: Sequence[tuple[str, str]]) -> dict:
    """Split porcelain entries into the categories a reviewer acts on."""
    modified: list[str] = []
    staged: list[str] = []
    unmerged: list[str] = []
    untracked: list[str] = []
    for code, path in entries:
        if code == "??":
            untracked.append(path)
            continue
        if code in UNMERGED_CODES:
            unmerged.append(path)
            continue
        if code[0] not in (" ", "?"):
            staged.append(path)
        if code[1] not in (" ", "?"):
            modified.append(path)
    return {
        "modified": sorted(modified),
        "staged": sorted(staged),
        "unmerged": sorted(unmerged),
        "untracked": sorted(untracked),
    }


def parse_ls_files_debug(text: str) -> dict[str, dict]:
    """Parse `git ls-files -z --debug` into `{path: {mtime_ns, size}}`.

    The `-z` form emits a raw NUL-terminated path followed by an indented
    block of index stat fields, so splitting on NUL gives, for every chunk
    after the first, the previous entry's stat block plus the next path on
    the final line.
    """
    chunks = text.split("\0")
    if not chunks:
        return {}
    entries: dict[str, dict] = {}
    path = chunks[0]
    for chunk in chunks[1:]:
        block, _, next_path = chunk.rpartition("\n")
        fields = _parse_debug_block(block)
        if path and fields is not None:
            entries[path] = fields
        path = next_path
    return entries


def _parse_debug_block(block: str) -> dict | None:
    mtime_ns: int | None = None
    size: int | None = None
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("mtime:"):
            value = stripped.split(":", 1)[1].strip()
            seconds, _, nanoseconds = value.partition(":")
            try:
                mtime_ns = int(seconds) * 1_000_000_000 + int(nanoseconds or 0)
            except ValueError:
                return None
        elif stripped.startswith("size:"):
            value = stripped.split(":", 1)[1].strip().split()[0]
            try:
                size = int(value)
            except (ValueError, IndexError):
                return None
    if mtime_ns is None or size is None:
        return None
    return {"mtime_ns": mtime_ns, "size": size}


def stale_index_paths(
    worktree: Path,
    *,
    query: Query = git_query,
    stat: Callable[[Path], os.stat_result] = os.lstat,
) -> list[str]:
    """Tracked paths whose recorded index stat data no longer matches the file.

    This is NOT a dirtiness signal: git's own status already re-reads the
    content of such a path and reports it clean when the bytes match, which is
    the measured finding in the module docstring. It is reported separately so
    a reader who has been told "the index is stat-dirty" can see the number and
    see that it did not move the verdict.

    Only mtime and size are compared, which is git's own `core.checkStat =
    minimal` subset. A path whose inode or ownership changed but whose mtime
    and size did not will be counted fresh here while git would re-hash it.
    That is a false negative on an informational line, and it is preferred to
    guessing which stat fields the probed repository's `core.checkStat`
    actually honours.
    """
    entries = parse_ls_files_debug(
        query(["ls-files", "-z", "--debug"], worktree=worktree)
    )
    stale: list[str] = []
    for path, fields in entries.items():
        try:
            info = stat(worktree / path)
        except OSError:
            # Deleted or unreadable: `git status` owns that verdict, and it
            # has already reported the path. Not an index-freshness question.
            continue
        recorded_size = fields["size"]
        # Git zeroes the size of a racily-clean entry as a marker that it must
        # re-check the content, so a zero recorded size against a non-empty
        # file is staleness, not a size mismatch to compare literally.
        if recorded_size == 0 and info.st_size != 0:
            stale.append(path)
            continue
        if info.st_size != recorded_size or info.st_mtime_ns != fields["mtime_ns"]:
            stale.append(path)
    return sorted(stale)


def dirty_state(
    worktree: Path,
    *,
    query: Query = git_query,
    stat: Callable[[Path], os.stat_result] = os.lstat,
) -> tuple[dict, list[dict]]:
    """The tracked/untracked/unmerged split, plus the index-freshness note.

    Returns the state and a list of DEGRADED sources. A failure to read index
    freshness does not make the dirty verdict unknown, because the dirty
    verdict does not depend on it.
    """
    degraded: list[dict] = []
    entries = parse_porcelain_z(
        query(
            ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
            worktree=worktree,
        )
    )
    state = classify_entries(entries)
    try:
        state["index_stale"] = stale_index_paths(worktree, query=query, stat=stat)
    except GitQueryError as exc:
        state["index_stale"] = []
        degraded.append({"source": "index-freshness", "error": str(exc)})
    state["clean"] = not (
        state["modified"] or state["staged"] or state["unmerged"] or state["untracked"]
    )
    return state, degraded


def git_operations_in_progress(
    git_dir: Path, *, exists: Callable[[Path], bool] = Path.exists
) -> list[dict]:
    """Half-finished git operations, read from the RESOLVED git directory.

    `index.lock` is not here; `stale_index_lock` reports it separately,
    because its existence does not prove anything is running.
    """
    found: list[dict] = []
    for name, description in GIT_OPERATION_MARKERS:
        if exists(git_dir / name):
            found.append({"marker": name, "description": description})
    return found


def stale_index_lock(
    git_dir: Path, *, exists: Callable[[Path], bool] = Path.exists
) -> bool:
    """Whether a git index lock file is present.

    Reported, never counted as busy. Git removes this on a normal exit and on
    SIGINT, but a SIGKILLed git leaves it behind forever and nothing
    kernel-backed holds it, so existence supports "a git command may have
    crashed" and not "a git command is writing right now". Reading it as
    liveness made a free worktree report BUSY permanently.
    """
    return exists(git_dir / INDEX_LOCK_NAME)


def lease_state(
    environ: Mapping[str, str],
    worktree: Path,
    *,
    peek: Callable[[Path], dict | None] = gate.GateLease.peek,
    current_holder: Callable[[Path], dict | None] = gate.GateLease.current_holder,
) -> tuple[dict, list[dict]]:
    """Who holds the workstation-wide advisory gate lease, if anyone.

    The lease is a single workstation-wide slot, so "held" alone says nothing
    about THIS worktree; the sidecar's `worktree` field is what makes it
    local evidence, and that distinction is the one the collisions turned on.
    """
    path = gate.lease_dir(dict(environ)) / gate.LEASE_FILE_NAME
    state: dict = {
        "path": str(path),
        "held": False,
        "holder": None,
        "held_by_this_worktree": False,
        "held_elsewhere": False,
        "stale_sidecar": False,
    }
    unknown: list[dict] = []
    try:
        holder = peek(path)
    except OSError as exc:
        unknown.append({"source": "gate-lease", "error": str(exc)})
        return state, unknown
    if holder is None:
        # Free. A sidecar may still name a holder that died; the kernel
        # released the lock, so the sidecar is descriptive, never
        # authoritative, and gate.py overwrites it on the next acquire.
        try:
            stale = current_holder(path)
        except OSError:
            stale = None
        if stale:
            state["stale_sidecar"] = True
            state["holder"] = stale
        return state, unknown
    state["held"] = True
    if not holder:
        # Held, but the sidecar could not be read, so the holder's identity is
        # unknown. This cannot be called BUSY for this worktree (it may well
        # be another one), and it must not be called free either, so it is
        # recorded as a failed identity source and the verdict degrades to
        # UNKNOWN unless some other signal proves BUSY.
        unknown.append(
            {
                "source": "gate-lease-identity",
                "error": "the lease is held but its sidecar is unreadable, so "
                "the holding worktree cannot be identified",
            }
        )
        return state, unknown
    state["holder"] = holder
    recorded = holder.get("worktree")
    if not recorded:
        unknown.append(
            {
                "source": "gate-lease-identity",
                "error": "the lease sidecar names no worktree",
            }
        )
        return state, unknown
    try:
        same = Path(recorded).resolve() == worktree.resolve()
    except OSError:
        same = str(recorded) == str(worktree)
    state["held_by_this_worktree"] = same
    state["held_elsewhere"] = not same
    return state, unknown


def _is_undecided(
    proc: reap.ProcInfo,
    script: str,
    worktree: Path,
    expected: Path,
    cwd_lookup: Callable[[int], str | None],
) -> bool:
    """Whether a non-match might be the option grammar rather than the truth.

    Only reached when the walk DID return a script token that is not this
    worktree's gate. A walk that returned nothing is a definite answer, since
    `-c` and `-m` mean Python runs no script at all, so a linter or a `-c`
    one-liner naming the gate stays a clean FREE rather than becoming noise.

    That carve-out is an assumption, but not the same kind as the one that
    failed. The failed assumption governed TOKEN CONSUMPTION: getting `-uX`
    wrong shifted which token was read as the script, so the walk returned a
    real token that was the wrong one, and a wrong answer that looks like a
    right one is indistinguishable from truth downstream, which is how it
    produced a silent FREE. This one governs SEARCH TERMINATION, and for these
    two options the termination is guaranteed by Python's semantics rather
    than by any model of them: if Python parses either, there is no script
    argument, so returning nothing is not a guess. The clustered and attached
    spellings, `-uc` and `-mmodule`, are not recognised as terminators at all;
    they fall through the generic skip branch and land in `undecided`. So this
    carve-out's failure mode is UNKNOWN, never FREE.

    The substring test below is used deliberately, and only here. Everywhere
    else in this file containment was the wrong tool because it was asked to
    PROVE identity; here it is asked whether identity is still POSSIBLE after
    a structural match failed, and for that question over-answering costs an
    UNKNOWN rather than a wrong BUSY.
    """
    if Path(script).is_absolute():
        # An absolute token that resolved elsewhere is a definite other
        # script, unless the line ALSO names this gate absolutely, which means
        # the token picked was more likely an option's value.
        #
        # KNOWN GAP, deliberately not closed. This compares a canonical
        # `expected` against a raw command line, so a process naming the gate
        # through a non-canonical path, a symlinked prefix such as `/var`
        # against `/private/var` on macOS, is not recognised here and the
        # verdict is FREE rather than UNKNOWN. Reaching it needs whitespace in
        # the checkout path AND a non-canonical spelling together, since
        # without whitespace the walk never lands in this branch.
        #
        # A previous attempt compared a second spelling of the WORKTREE, which
        # cannot help: `collect` re-anchors to `--show-toplevel`, which git
        # always reports physically, so both spellings of the worktree are the
        # same string and the clause could never fire. It was removed rather
        # than kept, because a comparison that cannot fire reads as coverage
        # and is worse than an absent one. Recovering the path the process
        # actually named would mean resolving runs of space-joined command
        # tokens; a reviewer measured that at about nine lines inside this
        # branch, bounded to runs of six tokens, breaking nothing else. So the
        # reason this is open is not cost. It is that the defect needs
        # whitespace and a non-canonical spelling together, and this branch had
        # already taken two repairs that each introduced a defect, so the next
        # change to it should be made deliberately rather than in passing.
        #
        # The command line is the ONLY input to this program whose spelling it
        # does not control; every path it compares against is canonical by
        # construction. `chelis#1568` records this as residual, with the same
        # identity-versus-spelling root as `chelis#1584`.
        return reap.command_mentions_path(proc.command, str(expected))
    cwd = cwd_lookup(proc.pid)
    if cwd is None:
        return GATE_SCRIPT_RELATIVE in proc.command
    if not reap.path_is_under(cwd, worktree):
        # A relative invocation resolved against another checkout's working
        # directory is that checkout's gate, decisively not this one's.
        return False
    return GATE_SCRIPT_RELATIVE in proc.command or reap.command_mentions_path(
        proc.command, str(expected)
    )


def match_gate_processes(
    procs: Sequence[reap.ProcInfo],
    worktree: Path,
    cwd_lookup: Callable[[int], str | None] = reap.proc_cwd,
) -> tuple[list[reap.ProcInfo], list[reap.ProcInfo]]:
    """Live gate runs scoped to this worktree, and the ones it cannot decide.

    Returns `(matched, undecided)`. A match is a Python interpreter whose
    SCRIPT ARGUMENT resolves to this worktree's `scripts/gate.py`; both halves
    are structural, and `script_argument` explains why the token's position is
    the property rather than its presence.

    `undecided` is the fail-safe, and it is the half that matters. When the
    walk returned a script that is not this gate, but the command line still
    names this gate, the walk has probably met a spelling its model of
    Python's option grammar does not cover, so absence of a match is not
    evidence of absence. The caller records it as a failed source and the
    verdict degrades to UNKNOWN. Reporting FREE while a gate runs is the
    collision this tool exists to prevent, so every gap in the model has to
    land on the safe side of it, including the gaps nobody has found yet.
    """
    expected = (worktree / GATE_SCRIPT_RELATIVE).resolve()
    own_pid = os.getpid()
    matched: list[reap.ProcInfo] = []
    undecided: list[reap.ProcInfo] = []
    for proc in procs:
        if proc.pid == own_pid:
            continue
        if not proc.basename.startswith(PYTHON_BASENAME_PREFIX):
            continue
        script = script_argument(proc.command)
        if script is None:
            continue
        candidate = Path(script)
        resolved: Path | None = None
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = None
        else:
            cwd = cwd_lookup(proc.pid)
            if cwd is not None:
                try:
                    resolved = (Path(cwd) / candidate).resolve()
                except OSError:
                    resolved = None
        if resolved is not None and resolved == expected:
            matched.append(proc)
        elif _is_undecided(proc, script, worktree, expected, cwd_lookup):
            undecided.append(proc)
    return matched, undecided


def _describe_proc(proc: reap.ProcInfo, orphaned: bool = False) -> dict:
    command = proc.command
    return {
        "pid": proc.pid,
        "ppid": proc.ppid,
        "etime": proc.etime,
        "command": command if len(command) <= 160 else command[:157] + "...",
        "orphaned": orphaned,
    }


def process_state(
    worktree: Path,
    *,
    snapshot: Callable[[], list[reap.ProcInfo]] = reap.ps_snapshot,
    cwd_lookup: Callable[[int], str | None] = reap.proc_cwd,
) -> tuple[dict, list[dict]]:
    """Build/test processes and gate processes scoped to this worktree."""
    state: dict = {"matched": [], "gate_processes": [], "orphaned": 0}
    unknown: list[dict] = []
    try:
        procs = snapshot()
    except RuntimeError as exc:
        unknown.append({"source": "processes", "error": str(exc)})
        return state, unknown
    matched = reap.match_repo_processes(procs, worktree, cwd_lookup)
    orphaned = reap.classify_orphans(matched, procs)
    gate_procs, undecided = match_gate_processes(procs, worktree, cwd_lookup)
    for proc in undecided:
        unknown.append(
            {
                "source": "gate-process-identity",
                "error": (
                    f"pid {proc.pid} is a Python process whose command line "
                    "names this worktree's gate, but its script argument could "
                    "not be read from the flattened command line, so whether a "
                    f"gate is running here cannot be decided: {proc.command[:120]}"
                ),
            }
        )
    matched_pids = {proc.pid for proc in matched}
    state["matched"] = [
        _describe_proc(proc, proc.pid in orphaned) for proc in matched
    ]
    state["gate_processes"] = [
        _describe_proc(proc, proc.pid in orphaned)
        for proc in gate_procs
        if proc.pid not in matched_pids
    ]
    state["orphaned"] = len(orphaned)
    return state, unknown


def last_gate_report(
    worktree: Path, environ: Mapping[str, str]
) -> tuple[dict | None, list[dict]]:
    """The newest finished gate run's summary, or None.

    HISTORY ONLY. `gate.py` writes a summary from its `main`'s `finally`
    block and names the file from the run's end timestamp, so nothing here can
    say whether a run is in flight. Every caller must label it accordingly,
    and a failure to read it is DEGRADED rather than unknown: history that
    cannot speak about now must not be able to withhold FREE.
    """
    degraded: list[dict] = []
    directory = gate.report_directory(dict(environ), worktree)
    try:
        reports = sorted(
            path for path in directory.glob("*.json") if path.is_file()
        )
    except OSError as exc:
        degraded.append({"source": "gate-reports", "error": str(exc)})
        return None, degraded
    if not reports:
        return None, degraded
    newest = reports[-1]
    try:
        payload = json.loads(newest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        degraded.append(
            {"source": "gate-reports", "error": f"{newest.name}: {exc}"}
        )
        return None, degraded
    if not isinstance(payload, dict):
        degraded.append(
            {"source": "gate-reports", "error": f"{newest.name}: not an object"}
        )
        return None, degraded
    return {
        "file": newest.name,
        "mode": payload.get("mode"),
        "termination": payload.get("termination"),
        "exit_code": payload.get("exit_code"),
        "seconds": payload.get("seconds"),
        "ended_at": payload.get("ended_at"),
        "head": (payload.get("git") or {}).get("head"),
    }, degraded


def busy_signals(state: dict) -> list[tuple[str, str]]:
    """Every busy signal that fired, as `(signal, reason)` pairs.

    Split out from `verdict` so the suite can assert exactly WHICH signal a
    given input fires, which is what makes a near-miss test meaningful: a test
    that only checks the verdict cannot tell "the right signal fired" from
    "a different signal happened to fire too".
    """
    fired: list[tuple[str, str]] = []
    lease = state["lease"]
    if lease.get("held_by_this_worktree"):
        fired.append(
            (
                SIGNAL_GATE_LEASE,
                "the gate lease is held by this worktree: "
                + gate.describe_holder(lease.get("holder")),
            )
        )
    processes = state["processes"]
    for proc in processes.get("gate_processes", []):
        fired.append(
            (SIGNAL_GATE_PROCESS, f"a gate process is running here: pid {proc['pid']}")
        )
    matched = processes.get("matched", [])
    if matched:
        fired.append(
            (
                SIGNAL_BUILD_PROCESS,
                f"{len(matched)} build/test process(es) scoped to this checkout, "
                f"newest pid {matched[0]['pid']}",
            )
        )
    for operation in state.get("git_operations", []):
        fired.append(
            (
                SIGNAL_GIT_OPERATION,
                f"{operation['description']} ({operation['marker']})",
            )
        )
    return fired


def verdict(state: dict) -> tuple[str, list[str]]:
    """The verdict and the reasons behind it.

    BUSY > UNKNOWN > NOT CLEAN > FREE. Positive evidence is never downgraded
    by a failure elsewhere; absence of evidence always is.
    """
    fired = busy_signals(state)
    if fired:
        return VERDICT_BUSY, [reason for _signal, reason in fired]
    if state.get("unknown"):
        # Only sources that could have hidden an owner or a modification
        # withhold FREE. A source that speaks about the past or about index
        # freshness is recorded in `degraded` instead, because history that
        # cannot speak about now must not be able to pin the verdict at
        # UNKNOWN forever: a SIGKILLed gate can leave a truncated report
        # behind, and nothing would ever clean it up.
        return VERDICT_UNKNOWN, [
            f"{item['source']}: {item['error']}" for item in state["unknown"]
        ]
    dirty = state["dirty"]
    if not dirty["clean"]:
        counts = [
            f"{len(dirty[key])} {key}"
            for key in ("modified", "staged", "unmerged", "untracked")
            if dirty[key]
        ]
        return VERDICT_NOT_CLEAN, [", ".join(counts)]
    return VERDICT_FREE, []


def collect(
    worktree: Path,
    *,
    environ: Mapping[str, str],
    now: Callable[[], datetime] = _utc_now,
    query: Query = git_query,
    snapshot: Callable[[], list[reap.ProcInfo]] = reap.ps_snapshot,
    cwd_lookup: Callable[[int], str | None] = reap.proc_cwd,
    stat: Callable[[Path], os.stat_result] = os.lstat,
    exists: Callable[[Path], bool] = Path.exists,
) -> dict:
    """Gather every signal. Writes nothing, anywhere."""
    unknown: list[dict] = []
    degraded: list[dict] = []
    probed_path = worktree
    state: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(now()),
        "worktree": str(worktree),
        "probed_path": str(probed_path),
        "command": (
            f".venv/bin/python scripts/worktree_status.py --path {worktree}"
        ),
        "git": None,
        "degraded": degraded,
        "dirty": {
            "modified": [],
            "staged": [],
            "unmerged": [],
            "untracked": [],
            "index_stale": [],
            "clean": True,
        },
        "git_operations": [],
        "stale_index_lock": False,
        "lease": {},
        "processes": {"matched": [], "gate_processes": [], "orphaned": 0},
        "last_gate_report": None,
        "unknown": unknown,
    }

    try:
        facts = git_facts(worktree, query=query)
        state["git"] = facts
        # Re-anchor to the repository root. Everything below is scoped by path
        # -- the lease sidecar's worktree, the process matcher, the report
        # directory, and the index paths, which git reports relative to the
        # root. Probing `<root>/subdir` without this reports FREE while a gate
        # runs at the root, which is the exact collision this tool exists to
        # prevent, so the caller's convenience path is resolved rather than
        # trusted.
        worktree = Path(facts["toplevel"])
        state["worktree"] = str(worktree)
    except GitQueryError as exc:
        unknown.append({"source": "git", "error": str(exc)})

    if state["git"] is not None:
        try:
            dirty, dirty_degraded = dirty_state(worktree, query=query, stat=stat)
            state["dirty"] = dirty
            degraded.extend(dirty_degraded)
        except GitQueryError as exc:
            unknown.append({"source": "git-status", "error": str(exc)})
        git_dir = Path(state["git"]["git_dir"])
        state["git_operations"] = git_operations_in_progress(git_dir, exists=exists)
        state["stale_index_lock"] = stale_index_lock(git_dir, exists=exists)

    lease, lease_unknown = lease_state(environ, worktree)
    state["lease"] = lease
    unknown.extend(lease_unknown)

    processes, process_unknown = process_state(
        worktree, snapshot=snapshot, cwd_lookup=cwd_lookup
    )
    state["processes"] = processes
    unknown.extend(process_unknown)

    report, report_degraded = last_gate_report(worktree, environ)
    state["last_gate_report"] = report
    degraded.extend(report_degraded)

    name, reasons = verdict(state)
    state["verdict"] = name
    state["reasons"] = reasons
    state["exit_code"] = EXIT_FOR_VERDICT[name]
    return state


def _sample(paths: Sequence[str], limit: int = 3) -> str:
    if not paths:
        return ""
    shown = ", ".join(paths[:limit])
    if len(paths) > limit:
        shown += f", and {len(paths) - limit} more"
    return f" [{shown}]"


def render_human(state: dict) -> str:
    """The paste-ready block. It names the instant and the command that
    produced it, so a brief that transcribes a stale one shows its age."""
    lines = [f"VERDICT: {state['verdict']}"]
    for reason in state.get("reasons", []):
        lines.append(f"  because: {reason}")

    git = state.get("git")
    if git is None:
        lines.append(f"worktree: {state['worktree']}  (git facts unavailable)")
        lines.append("head:     unknown")
    else:
        kind = "linked" if git["linked"] else "main"
        branch = git["branch"] or "detached HEAD"
        lines.append(f"worktree: {state['worktree']}  ({kind}, {branch})")
        if state.get("probed_path") and state["probed_path"] != state["worktree"]:
            lines.append(
                f"  note:   resolved from the probed path {state['probed_path']}"
            )
        lines.append(f"head:     {git['head']}")

    dirty = state["dirty"]
    counts = (
        f"{len(dirty['modified'])} modified, {len(dirty['staged'])} staged, "
        f"{len(dirty['untracked'])} untracked, {len(dirty['unmerged'])} unmerged"
    )
    status = "clean" if dirty["clean"] else "DIRTY"
    stale = len(dirty.get("index_stale", []))
    note = f"; {stale} stat-dirty only, not counted as dirty" if stale else ""
    lines.append(f"tree:     {status} ({counts}{note})")
    for key in ("modified", "staged", "unmerged", "untracked"):
        if dirty[key]:
            lines.append(f"  {key}:{_sample(dirty[key])}")

    lease = state["lease"]
    if lease.get("held_by_this_worktree"):
        where = "HELD BY THIS WORKTREE"
    elif lease.get("held_elsewhere"):
        where = "held by another worktree"
    elif lease.get("held"):
        where = "held, holder unidentified"
    else:
        where = "free"
    lines.append(f"lease:    {lease.get('path', '?')}  {where}")
    if lease.get("holder") and not lease.get("stale_sidecar"):
        lines.append(f"  holder: {gate.describe_holder(lease['holder'])}")
    if lease.get("stale_sidecar"):
        lines.append(
            "  note:   a sidecar names a holder that is gone; the kernel "
            "released the lock, so the file is descriptive only"
        )
    if lease.get("held_elsewhere"):
        lines.append(
            "  WARN:   a heavyweight run started here will queue behind that "
            "holder"
        )

    processes = state["processes"]
    running = processes["matched"] + processes["gate_processes"]
    if running:
        lines.append(f"processes: {len(running)} scoped to this checkout")
        for proc in running[:5]:
            flag = " ORPHAN" if proc["orphaned"] else ""
            lines.append(
                f"  pid {proc['pid']:>7} ppid {proc['ppid']:>7} "
                f"{proc['etime']:>11}{flag}  {proc['command']}"
            )
        if len(running) > 5:
            lines.append(f"  and {len(running) - 5} more")
    else:
        lines.append("processes: none scoped to this checkout")

    report = state.get("last_gate_report")
    if report is None:
        lines.append("last run: no gate report in this worktree")
    else:
        lines.append(
            f"last run: gate {report.get('mode', '?')} "
            f"{str(report.get('termination', '?')).upper()}, "
            f"exit {report.get('exit_code', '?')}, "
            f"{report.get('seconds', '?')} s, ended {report.get('ended_at', '?')}"
            "  [history, not now]"
        )

    operations = state.get("git_operations", [])
    if operations:
        for operation in operations:
            lines.append(f"git op:   {operation['description']}")
    else:
        lines.append("git op:   none")
    if state.get("stale_index_lock"):
        lines.append(f"index:    {INDEX_LOCK_NOTE}")

    if state.get("unknown"):
        for item in state["unknown"]:
            lines.append(f"unknown:  {item['source']}: {item['error']}")
    for item in state.get("degraded", []):
        lines.append(f"degraded: {item['source']}: {item['error']} (verdict unaffected)")

    lines.append(f"probed:   {state['generated_at']} by {state['command']}")
    return "\n".join(lines)


def render_json(state: dict) -> str:
    return json.dumps(state, indent=2, sort_keys=True)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report whether a worktree is free right now, with the evidence: "
            "the advisory gate lease, this checkout's build/test and gate "
            "processes, the git state, and the last finished gate run. "
            "Writes nothing. Exit 0 free, 1 busy, 2 cannot tell, 3 not clean."
        ),
    )
    parser.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help=(
            "The worktree to probe (default: this script's own checkout). "
            "Scoping is per-checkout, as in reap_orphans.py."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the machine payload instead of the human block.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the verdict line.",
    )
    args = parser.parse_args(list(argv))
    if args.json and args.quiet:
        parser.error("--json and --quiet are mutually exclusive")
    return args


def main(
    argv: Sequence[str],
    *,
    environ: Mapping[str, str] | None = None,
    output_stream=None,
    **collect_kwargs,
) -> int:
    args = parse_args(argv)
    stream = sys.stdout if output_stream is None else output_stream
    environment = dict(os.environ if environ is None else environ)
    worktree = Path(args.path).resolve() if args.path else REPO_ROOT
    state = collect(worktree, environ=environment, **collect_kwargs)
    if args.json:
        print(render_json(state), file=stream)
    elif args.quiet:
        print(f"VERDICT: {state['verdict']}", file=stream)
    else:
        print(render_human(state), file=stream)
    return int(state["exit_code"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
