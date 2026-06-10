#!/usr/bin/env python3
"""List (and optionally reap) orphaned build/test processes for this repo.

Background: local validation throughput collapses when build/test
processes outlive their owning agent session (issue #348). An orphaned
`cargo nextest run` keeps burning CPU and can hold the cargo target-dir
lock across sessions; the resulting contention starves unrelated test
runs (measured 2026-06-10: the 25-test `rank_poly_tier3` suite took
2,434s under contention vs 24s on a quiet machine). This script is the
pre-build hygiene step the agent contract points at.

What it considers:

  - processes whose executable basename is one of `cargo`, `rustc`,
    `cargo-nextest`, or `chelis` AND whose command line or working
    directory references this repo checkout (path-boundary matched, so
    a sibling checkout like `<repo>-165` never matches);
  - processes whose EXECUTABLE lives in this repo's `target/`
    directory (nextest-spawned test binaries have arbitrary names but
    run from `target/`).

An installed `chelis` running elsewhere (e.g. a different project's
workload) is deliberately NOT matched when its command line does not
mention this repo and its cwd is outside it.

Known limitations (review the dry-run listing before `--kill`):

  - ppid==1 cannot distinguish an abandoned build from a DELIBERATELY
    detached one (`nohup cargo build` you are still tailing) - both
    classify ORPHANED;
  - an installed `chelis` working on another project but LAUNCHED from
    a shell cwd'd into this repo is matched, and killed if detached;
  - scoping is per-checkout: run from a worktree, the script does not
    see the main checkout's orphans (and vice versa) - run it from the
    checkout whose `target/` you are about to use.

A matched process is classified ORPHANED when:

  - its parent PID is 1 (re-parented to init/launchd after the owning
    shell died), or
  - its parent PID is no longer alive (gone from the `ps` snapshot), or
  - its parent is itself a matched build process that is orphaned
    (e.g. a `rustc` child of an orphaned `cargo`).

A build process whose ancestor chain still includes a live shell or
agent session is NOT orphaned and is never killed.

Usage:
    python3 scripts/reap_orphans.py            # dry-run listing (default)
    python3 scripts/reap_orphans.py --kill     # TERM orphans, then KILL
                                               # survivors after a grace
                                               # period (default 5s)
    python3 scripts/reap_orphans.py --kill --grace 10

Exit code is 0 unless the process snapshot itself could not be taken.
Per repo policy this is Python (stdlib only: `ps`/`lsof` via
subprocess, no psutil). Tests: `scripts/test_reap_orphans.py`.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

# Executable basenames that count as build/test tool processes.
BUILD_TOOL_NAMES = frozenset({"cargo", "rustc", "cargo-nextest", "chelis"})

# Default seconds to wait between SIGTERM and SIGKILL.
DEFAULT_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    ppid: int
    etime: str
    command: str

    @property
    def basename(self) -> str:
        """Basename of argv[0]. `ps -o command=` prints the full command
        line, so take the first whitespace token."""
        argv0 = self.command.split()[0] if self.command.split() else ""
        return os.path.basename(argv0)


def parse_ps_output(text: str) -> list[ProcInfo]:
    """Parse `ps -axww -o pid=,ppid=,etime=,command=` output. Each line:
    two right-aligned integers, an elapsed-time token, then the command
    line (which may contain spaces). Malformed lines are skipped."""
    procs: list[ProcInfo] = []
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid_s, ppid_s, etime, command = parts
        try:
            pid = int(pid_s)
            ppid = int(ppid_s)
        except ValueError:
            continue
        procs.append(ProcInfo(pid=pid, ppid=ppid, etime=etime, command=command))
    return procs


def ps_snapshot() -> list[ProcInfo]:
    """Take a live process snapshot via `ps`. Raises RuntimeError if
    `ps` cannot be run (the only condition that fails the script)."""
    try:
        result = subprocess.run(
            ["ps", "-axww", "-o", "pid=,ppid=,etime=,command="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"could not take a ps snapshot: {exc}") from exc
    return parse_ps_output(result.stdout)


def proc_cwd(pid: int) -> str | None:
    """Best-effort working-directory lookup: /proc on Linux, `lsof` on
    macOS. Returns None when the cwd cannot be determined (permission,
    process exited, lsof unavailable)."""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        pass
    try:
        result = subprocess.run(
            ["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("n"):
            return line[1:]
    return None


def _path_is_under(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _command_mentions_path(command: str, path_str: str) -> bool:
    """Path-boundary-aware containment check. A raw substring test would
    let a sibling checkout match (`<repo>-165` contains `<repo>`), and
    this script sends SIGKILL, so `path_str` counts only when followed
    by a path separator, whitespace, a quote, or end-of-string."""
    start = 0
    while True:
        idx = command.find(path_str, start)
        if idx == -1:
            return False
        end = idx + len(path_str)
        if end == len(command) or command[end] in "/ \t\"'=:,;":
            return True
        start = idx + 1


def current_command(pid: int) -> str | None:
    """The pid's command line right now, formatted identically to the
    snapshot's command field, or None if the process is gone. Used to
    re-verify identity immediately before signaling (PID reuse guard)."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-ww", "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    command = result.stdout.strip()
    return command or None


def match_repo_processes(
    procs: list[ProcInfo],
    repo_root: Path,
    cwd_lookup: Callable[[int], str | None] = proc_cwd,
) -> list[ProcInfo]:
    """Return the subset of `procs` that belong to this repo's build/test
    activity. `cwd_lookup` is injectable for tests.

    Matching rules (see module docstring): build-tool basenames scoped
    to this repo by command line or cwd, plus repo `target/` binaries by
    command line. The cwd lookup runs only for build-tool processes
    whose command line does not already mention the repo, so the
    expensive per-pid `lsof` is rare."""
    repo_str = str(repo_root)
    target_str = str(repo_root / "target")
    own_pid = os.getpid()
    matched: list[ProcInfo] = []
    for proc in procs:
        if proc.pid == own_pid:
            continue
        is_tool = proc.basename in BUILD_TOOL_NAMES
        mentions_repo = _command_mentions_path(proc.command, repo_str)
        if is_tool and mentions_repo:
            matched.append(proc)
            continue
        # nextest-spawned test binaries: arbitrary basenames, but their
        # EXECUTABLE lives in this repo's target/ directory. Keyed on
        # the first command token so a `tail -f .../target/...` or a
        # manual `gcc .../target/.../out.c` is never matched.
        executable = proc.command.split(None, 1)[0] if proc.command else ""
        if executable.startswith(target_str + os.sep):
            matched.append(proc)
            continue
        if is_tool:
            cwd = cwd_lookup(proc.pid)
            if cwd is not None and _path_is_under(cwd, repo_root):
                matched.append(proc)
    return matched


def classify_orphans(
    matched: list[ProcInfo], snapshot: list[ProcInfo]
) -> set[int]:
    """Return the PIDs of matched processes that look orphaned.

    Orphaned: ppid == 1, parent gone from the snapshot, or parent is a
    matched build process that is itself orphaned (transitive, so a
    rustc under an orphaned cargo is reaped with it)."""
    live_pids = {proc.pid for proc in snapshot}
    matched_by_pid = {proc.pid: proc for proc in matched}
    orphaned: set[int] = set()

    def is_orphaned(proc: ProcInfo, seen: frozenset[int]) -> bool:
        if proc.pid in orphaned:
            return True
        if proc.pid in seen:
            return False  # ppid cycle; be conservative
        if proc.ppid == 1 or proc.ppid not in live_pids:
            return True
        parent = matched_by_pid.get(proc.ppid)
        if parent is not None:
            return is_orphaned(parent, seen | {proc.pid})
        return False

    for proc in matched:
        if is_orphaned(proc, frozenset()):
            orphaned.add(proc.pid)
    return orphaned


def terminate(
    pids: list[int],
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    kill_fn: Callable[[int, int], None] = os.kill,
    sleep_fn: Callable[[float], None] = time.sleep,
    expected: dict[int, str] | None = None,
    command_lookup: Callable[[int], str | None] = current_command,
) -> list[int]:
    """SIGTERM every pid, wait `grace_seconds`, then SIGKILL survivors.
    Returns the pids that needed SIGKILL. `kill_fn`/`sleep_fn` are
    injectable for tests; no real signals are sent in the test suite.

    `expected` maps pid -> the command line recorded when the pid was
    classified. When provided, each pid's identity is re-verified via
    `command_lookup` immediately before EVERY signal (TERM and the
    post-grace KILL): a pid whose command no longer matches has exited
    and been reused, and is skipped rather than signaled. The snapshot
    can be seconds stale and PID churn is highest exactly in the
    contention scenarios this script targets."""

    def send(pid: int, sig: int) -> bool:
        """True if the signal was delivered (process still exists)."""
        try:
            kill_fn(pid, sig)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            print(f"reap_orphans: no permission to signal pid {pid}", file=sys.stderr)
            return False

    def identity_holds(pid: int) -> bool:
        if expected is None:
            return True
        live = command_lookup(pid)
        if live is not None and live == expected.get(pid):
            return True
        print(
            f"reap_orphans: pid {pid} no longer matches its snapshot; skipping",
            file=sys.stderr,
        )
        return False

    termed = [pid for pid in pids if identity_holds(pid) and send(pid, signal.SIGTERM)]
    if not termed:
        return []
    sleep_fn(grace_seconds)
    killed: list[int] = []
    for pid in termed:
        # Re-verify identity, then probe with signal 0; if still alive,
        # escalate to SIGKILL.
        if identity_holds(pid) and send(pid, 0) and send(pid, signal.SIGKILL):
            killed.append(pid)
    return killed


def format_listing(matched: list[ProcInfo], orphaned: set[int]) -> str:
    """Human-readable table: one line per matched process, orphans
    marked. Stable order: orphans first, then by pid."""
    if not matched:
        return "reap_orphans: no repo build/test processes found."
    rows = sorted(matched, key=lambda p: (p.pid not in orphaned, p.pid))
    lines = [f"{'STATE':<8} {'PID':>7} {'PPID':>7} {'ELAPSED':>11} COMMAND"]
    for proc in rows:
        state = "ORPHAN" if proc.pid in orphaned else "owned"
        command = proc.command if len(proc.command) <= 120 else proc.command[:117] + "..."
        lines.append(
            f"{state:<8} {proc.pid:>7} {proc.ppid:>7} {proc.etime:>11} {command}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "List this repo's cargo/rustc/cargo-nextest/chelis processes, "
            "mark the ones that look orphaned, and with --kill reap the "
            "orphans (TERM, then KILL after a grace period). Default is a "
            "dry-run listing."
        ),
    )
    p.add_argument(
        "--kill",
        action="store_true",
        help="Terminate the orphaned processes (SIGTERM, then SIGKILL).",
    )
    p.add_argument(
        "--grace",
        type=float,
        default=DEFAULT_GRACE_SECONDS,
        metavar="SECONDS",
        help=(
            "Seconds to wait between SIGTERM and SIGKILL "
            f"(default {DEFAULT_GRACE_SECONDS:g})."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        snapshot = ps_snapshot()
    except RuntimeError as exc:
        print(f"reap_orphans: {exc}", file=sys.stderr)
        return 1
    matched = match_repo_processes(snapshot, REPO_ROOT)
    orphaned = classify_orphans(matched, snapshot)
    print(format_listing(matched, orphaned))
    if not args.kill:
        if orphaned:
            print(
                f"reap_orphans: {len(orphaned)} orphan(s) found; "
                "re-run with --kill to reap them."
            )
        return 0
    if not orphaned:
        print("reap_orphans: nothing to kill.")
        return 0
    pids = sorted(orphaned)
    print(f"reap_orphans: sending SIGTERM to {pids} (grace {args.grace:g}s)")
    expected = {proc.pid: proc.command for proc in matched if proc.pid in orphaned}
    killed = terminate(pids, grace_seconds=args.grace, expected=expected)
    if killed:
        print(f"reap_orphans: SIGKILLed survivors {killed}")
    print(f"reap_orphans: done; {len(pids)} orphan(s) reaped.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
