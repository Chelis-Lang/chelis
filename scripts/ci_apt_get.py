#!/usr/bin/env python3
"""Run `apt-get update` + `apt-get install` with retries on transient
network flakes, for the Linux CI jobs.

Several CI jobs install C/SMT build prerequisites with a bare
`apt-get update && apt-get install -y ...`. On the GitHub-hosted runners
this intermittently fails mid-download with a transient network error --
most often `deb.debian.org` (or `azure.archive.ubuntu.com`) closing the
connection:

    Connection failed [IP: ...] Connection reset by peer
    E: Failed to fetch http://deb.debian.org/... Connection reset by peer
    E: Unable to fetch some archives ...

This is purely environmental: the same job passes on a re-run with no code
change, so it reads like a flaky failure but is a mirror/CDN hiccup. A bare
single-shot `apt-get` has no recovery, so one reset reds the whole job (and,
for a required check, blocks the PR until someone manually re-runs).

This helper wraps the update+install in a bounded retry loop with backoff so
a transient reset self-heals within the step instead of failing the job. It
is intentionally NARROW: it only retries the apt-get pair, it does not retry
the cargo build or any test, and a genuine non-transient failure (an
unknown package, a held broken dep) still surfaces after the attempts are
exhausted -- the retries cannot turn a real packaging error green, only ride
out a flaky download.

Each individual `apt-get` invocation runs under a per-command WALL-CLOCK
TIMEOUT. A connection *reset* returns a non-zero exit that the retry loop
already handles, but a stalled mirror that accepts the TCP socket and then
stops sending data produces no exit at all: the bare `subprocess.run` would
block forever and hang the whole job until GitHub's 6-hour runner cap kills
it (observed: a 2h20m wedge on the "Install C dependencies" step). The
timeout kills a hung attempt and feeds it back into the retry loop as a
failed attempt, so a stall self-heals on the next attempt instead of
wedging the job. A normal update+install for the C prerequisites finishes in
well under a minute, so the default timeout is pure headroom for a working
mirror and only ever bites a genuine hang.

Usage (single-line `run:` step, mirroring the ci_free_disk.py /
ci_setup_uv_python.py convention so the gate's no-hand-inlined-command lock
is satisfied and the step stays Python, not shell, per repo policy):

    python3 scripts/ci_apt_get.py gcc libopenblas-dev libasan8 libubsan1

By default it runs `apt-get` under `sudo`; pass `--no-sudo` for container
jobs (e.g. the debian:11 glibc-2.31 lane) that already run as root and have
no `sudo`. `--no-install-recommends` is forwarded to the install when
passed. Everything after a literal `--` (or every bare token) is treated as
a package name.

Per repo policy this is Python, not a shell script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# apt-get's own documented "transient failure" exit code is 100, but a
# connection reset can also surface as a generic non-zero. We retry on ANY
# non-zero from the update/install pair: the loop is bounded and a genuine
# packaging error simply fails again on every attempt and then propagates,
# so retrying it costs only a few wasted seconds, never a false green.
DEFAULT_ATTEMPTS = 4

# Seconds to sleep before each retry (not before the first attempt). Short,
# linear backoff: a CDN reset clears in seconds, and a long sleep would just
# burn runner minutes on a genuinely-broken mirror that will fail anyway.
BACKOFF_SECONDS = 5

# Wall-clock ceiling for a single `apt-get update` or `apt-get install`
# invocation. A working update+install of the C prerequisites finishes in
# well under a minute on the hosted runners, so 300s is generous headroom for
# a slow-but-alive mirror while still bounding a genuine hang (a stalled
# connection that never returns) to a few minutes instead of the 6-hour
# runner cap. Pass 0 (or a negative value) via --timeout to disable.
PER_COMMAND_TIMEOUT_SECONDS = 300

# Conventional shell exit code for "killed by a timeout" (matches coreutils
# `timeout(1)`). The value is non-zero, so the retry loop treats a timed-out
# attempt exactly like any other transient failure.
TIMEOUT_EXIT_CODE = 124


def _run(cmd: list[str], *, timeout: float | None = None) -> int:
    """Run `cmd`, inheriting stdio so apt's output lands in the CI log.
    Returns the exit code (or 1 if it could not even be spawned, or
    TIMEOUT_EXIT_CODE if it exceeded `timeout` and was killed)."""
    try:
        return subprocess.run(cmd, check=False, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        # subprocess.run has already sent SIGKILL and reaped the child. Report
        # it as a failed attempt so the retry loop rides it out rather than
        # inheriting the hang. This is the whole point of the timeout: a
        # mirror that accepts the socket and then stops sending data yields no
        # exit code, so without this the bare run would block until the
        # 6-hour runner cap.
        print(
            f"ci_apt_get: {' '.join(cmd)} exceeded {timeout:g}s and was killed "
            f"(likely a stalled mirror connection); treating as a failed attempt.",
            file=sys.stderr,
        )
        return TIMEOUT_EXIT_CODE
    except OSError as exc:
        print(f"ci_apt_get: could not run {cmd!r}: {exc}", file=sys.stderr)
        return 1


def _attempt(
    packages: list[str],
    *,
    sudo: bool,
    no_install_recommends: bool,
    timeout: float | None,
) -> int:
    """One full update+install attempt. Returns the first non-zero exit of
    the pair, or 0 if both succeed."""
    prefix = ["sudo"] if sudo else []
    update = prefix + ["apt-get", "update"]
    install = prefix + ["apt-get", "install", "-y"]
    if no_install_recommends:
        install.append("--no-install-recommends")
    install += packages

    rc = _run(update, timeout=timeout)
    if rc != 0:
        return rc
    return _run(install, timeout=timeout)


def apt_get(
    packages: list[str],
    *,
    sudo: bool = True,
    no_install_recommends: bool = False,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    timeout: float | None = PER_COMMAND_TIMEOUT_SECONDS,
    sleep=time.sleep,
) -> int:
    """Run the update+install pair, retrying up to `attempts` times on any
    non-zero exit (a reset, a spawn failure, or a per-command timeout).
    Returns 0 on success or the last attempt's exit code.

    `timeout` bounds each individual `apt-get` invocation in wall-clock
    seconds; None disables it. `sleep` is injectable so the unit tests don't
    actually wait."""
    if not packages:
        print("ci_apt_get: no packages requested; nothing to do.")
        return 0

    last_rc = 1
    for i in range(1, attempts + 1):
        print(f"ci_apt_get: attempt {i}/{attempts}: installing {' '.join(packages)}")
        last_rc = _attempt(
            packages,
            sudo=sudo,
            no_install_recommends=no_install_recommends,
            timeout=timeout,
        )
        if last_rc == 0:
            return 0
        if i < attempts:
            print(
                f"ci_apt_get: apt-get exited {last_rc} "
                f"(likely a transient mirror/network reset); "
                f"retrying in {backoff_seconds:g}s",
                file=sys.stderr,
            )
            sleep(backoff_seconds)

    print(
        f"ci_apt_get: apt-get still failing after {attempts} attempts "
        f"(exit {last_rc}); giving up.",
        file=sys.stderr,
    )
    return last_rc


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="apt-get update+install with retries on transient flakes."
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="run apt-get directly (for container jobs already running as root).",
    )
    parser.add_argument(
        "--no-install-recommends",
        action="store_true",
        help="forward --no-install-recommends to apt-get install.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"max update+install attempts (default {DEFAULT_ATTEMPTS}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=PER_COMMAND_TIMEOUT_SECONDS,
        help=(
            "per-command wall-clock timeout in seconds "
            f"(default {PER_COMMAND_TIMEOUT_SECONDS}); a hung apt-get is killed "
            "and retried. Pass 0 to disable."
        ),
    )
    parser.add_argument(
        "packages",
        nargs="+",
        help="package names to install.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    # A non-positive --timeout disables the ceiling (subprocess.run treats
    # None as "no timeout").
    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    return apt_get(
        args.packages,
        sudo=not args.no_sudo,
        no_install_recommends=args.no_install_recommends,
        attempts=args.attempts,
        timeout=timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
