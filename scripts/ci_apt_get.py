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


def _run(cmd: list[str]) -> int:
    """Run `cmd`, inheriting stdio so apt's output lands in the CI log.
    Returns the exit code (or 1 if it could not even be spawned)."""
    try:
        return subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        print(f"ci_apt_get: could not run {cmd!r}: {exc}", file=sys.stderr)
        return 1


def _attempt(packages: list[str], *, sudo: bool, no_install_recommends: bool) -> int:
    """One full update+install attempt. Returns the first non-zero exit of
    the pair, or 0 if both succeed."""
    prefix = ["sudo"] if sudo else []
    update = prefix + ["apt-get", "update"]
    install = prefix + ["apt-get", "install", "-y"]
    if no_install_recommends:
        install.append("--no-install-recommends")
    install += packages

    rc = _run(update)
    if rc != 0:
        return rc
    return _run(install)


def apt_get(
    packages: list[str],
    *,
    sudo: bool = True,
    no_install_recommends: bool = False,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    sleep=time.sleep,
) -> int:
    """Run the update+install pair, retrying up to `attempts` times on any
    non-zero exit. Returns 0 on success or the last attempt's exit code.

    `sleep` is injectable so the unit tests don't actually wait."""
    if not packages:
        print("ci_apt_get: no packages requested; nothing to do.")
        return 0

    last_rc = 1
    for i in range(1, attempts + 1):
        print(f"ci_apt_get: attempt {i}/{attempts}: installing {' '.join(packages)}")
        last_rc = _attempt(
            packages, sudo=sudo, no_install_recommends=no_install_recommends
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
        "packages",
        nargs="+",
        help="package names to install.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    return apt_get(
        args.packages,
        sudo=not args.no_sudo,
        no_install_recommends=args.no_install_recommends,
        attempts=args.attempts,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
