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
also has an explicit GitHub-hosted Ubuntu mirror policy: select the official
HTTPS archives before apt starts, so a live but slow Azure transfer cannot
consume every outer timeout without selecting another mirror. The runner
source lists, package selections and authentication remain intact.

For the Debian Bullseye/glibc-2.31 lanes, ``--debian-bullseye-snapshot``
replaces every moving apt source with one immutable snapshot before the first
update. That prevents the mirror race
where a freshly downloaded index names a package version that its selected
CDN edge no longer serves (a persistent 404 which retries cannot heal).
APT selects package versions from the signed snapshot indexes.
A private worker requests a SHA-256 URI plan and uses Debian Archive only
as a verified byte carrier. It publishes verified files atomically, then
calls APT with downloads disabled. The worker and all its children share
one install timeout.

The retry loop is intentionally NARROW: it only retries the apt-get pair, it
does not retry the cargo build or any test, and a genuine non-transient
failure (an unknown package, a held broken dep) still surfaces after the
attempts are exhausted -- the retries cannot turn a real packaging error
green, only ride out a flaky download.

Each update and install phase runs under a per-command WALL-CLOCK
TIMEOUT. A connection *reset* returns a non-zero exit that the retry loop
already handles, but a stalled mirror that accepts the TCP socket and then
stops sending data produces no exit at all: an unbounded `apt-get` would
block forever and hang the whole job until GitHub's 6-hour runner cap kills
it (observed: a 2h20m wedge on the "Install C dependencies" step). The
timeout kills a hung attempt and feeds it back into the retry loop as a
failed attempt, so a stall self-heals on the next attempt instead of
wedging the job. A normal update+install for the C prerequisites finishes in
well under a minute, so the default timeout is pure headroom for a working
mirror and only ever bites a genuine hang.

The timeout is enforced by the coreutils `timeout(1)` binary, inserted
AFTER any `sudo` (`sudo timeout --kill-after=Ns Ms apt-get ...`), NOT by
Python's `subprocess.run(timeout=)`. That distinction is essential: Python's
timeout SIGKILLs the direct child, which is `sudo`, and Ubuntu's default
`Defaults use_pty` runs `apt-get` in a separate pty session, so killing
`sudo` leaves `apt-get` ORPHANED and still holding `/var/lib/apt/lists/lock`
-- every retry then dies instantly on "Could not get lock ... held by
process N (apt-get)" and the step fails anyway. Placing `timeout` as the
direct parent of `apt-get`, inside the same session, means the kill actually
reaches `apt-get`; the kernel releases its flock on death (even a SIGKILL),
so the next retry can re-acquire the lock. `--kill-after` escalates SIGTERM
to SIGKILL if apt-get does not exit on the polite signal. If the `timeout`
binary is somehow absent the command fails fast and non-zero (127 under
`sudo`, or 1 on the `--no-sudo` path where the missing binary surfaces as an
OSError) and is retried -- apt-get never launches unbounded, so the
no-6h-hang invariant still holds either way.

Scope: the ceiling defends against a stalled DOWNLOAD (the realistic and
observed failure -- the wedge that motivated this was on `apt-get update`,
which only fetches indices). A kill that landed mid-`dpkg` unpack/configure
would not auto-recover: the next `apt-get install` could report "dpkg was
interrupted, you must manually run 'dpkg --configure -a'". That window is
seconds wide for these few small packages, so it is knowingly out of scope;
if it ever bites, add a `dpkg --configure -a` recovery before the retry.

Usage (single-line `run:` step, mirroring the ci_free_disk.py /
ci_setup_uv_python.py convention so the gate's no-hand-inlined-command lock
is satisfied and the step stays Python, not shell, per repo policy):

    python3 scripts/ci_apt_get.py gcc libopenblas-dev libasan8 libubsan1

By default it runs `apt-get` under `sudo`; pass `--no-sudo` for container
jobs (e.g. the Bullseye glibc-2.31 lane) that already run as root and have no
`sudo`. `--no-install-recommends` is forwarded to the install when passed.
Everything after a literal `--` (or every bare token) is treated as a package
name.

Per repo policy this is Python, not a shell script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, build_opener

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
# invocation, enforced by the coreutils `timeout(1)` binary. A working
# update+install of the C prerequisites finishes in well under a minute on
# the hosted runners, so 300s is generous headroom for a slow-but-alive
# mirror while still bounding a genuine hang (a stalled connection that never
# returns) to a few minutes instead of the 6-hour runner cap. Pass 0 (or a
# negative value) via --timeout to disable.
PER_COMMAND_TIMEOUT_SECONDS = 300

# After the timeout fires, `timeout` sends SIGTERM; if apt-get has not exited
# within this grace period it escalates to SIGKILL. apt-get normally releases
# its lock and exits on SIGTERM; the SIGKILL backstop guarantees termination
# (the kernel releases the flock on death) so a wedged apt-get cannot outlive
# its ceiling and poison the retries.
KILL_AFTER_SECONDS = 30

# Exit codes that read as "the wall-clock ceiling stopped the command":
# 124 = coreutils `timeout(1)` timed the command out (killed by the requested
# signal); 137 = 128+9, the command was SIGKILLed -- usually our --kill-after
# escalation when apt-get ignored SIGTERM. Note 137 is NOT unambiguous: any
# external SIGKILL (e.g. the kernel OOM killer) also yields 137, so the
# "timed out" log line below is a best-effort label, not a proof. It does not
# matter for control flow: the retry loop fires on ANY non-zero exit, so these
# are special-cased purely to print a more accurate diagnostic.
TIMEOUT_EXIT_CODES = (124, 137)

# A timestamp after Bullseye's final LTS updates but before these jobs were
# repaired. snapshot.debian.org resolves this timestamp to the latest import
# at or before it and keeps both the indices and referenced package pool
# immutable. Update this deliberately, with the workflow contract test and a
# successful hosted glibc-2.31 run as evidence.
BULLSEYE_SNAPSHOT_TIMESTAMP = "20260901T000000Z"

_BULLSEYE_WORKER_FLAG = "--_bullseye-install-worker"


@dataclass(frozen=True)
class _ArchivePackage:
    uri: str
    filename: str
    size: int
    sha256: str


def _archive_uri(uri: str) -> str:
    """Map only pinned snapshot package paths to the untrusted byte carrier."""
    url = urlsplit(uri)
    if url.query or url.fragment:
        raise ValueError(f"unexpected package URI: {uri!r}")
    for archive in ("debian", "debian-security"):
        prefix = (
            "https://snapshot-cloudflare.debian.org/archive/"
            f"{archive}/{BULLSEYE_SNAPSHOT_TIMESTAMP}/"
        )
        if not uri.startswith(prefix):
            continue
        relative = uri[len(prefix):]
        decoded = unquote(relative, errors="strict")
        if (
            not re.fullmatch(r"pool/(?:[A-Za-z0-9+_.:~\-]+/)*[A-Za-z0-9+_.:~\-]+\.deb", decoded)
            or any(part in (".", "..") for part in decoded.split("/"))
        ):
            raise ValueError(f"unsafe package path: {uri!r}")
        return f"https://archive.debian.org/{archive}/{relative}"
    raise ValueError(f"unrecognized snapshot package origin: {uri!r}")


def _parse_archive_plan(output: str) -> list[_ArchivePackage]:
    """Accept only APT URI records with exact sizes and SHA-256 hashes."""
    packages = []
    names = set()
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("'"):
            continue
        record = re.fullmatch(
            r"'([^'\s]+)'\s+(\S+)\s+([0-9]+)\s+SHA256:([0-9a-fA-F]{64})",
            line,
        )
        if record is None:
            raise ValueError(f"invalid SHA-256 package record: {line!r}")
        uri, name, size, digest = record.groups()
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9+_.:%~\-]*\.deb", name)
            or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9+_.:~\-]*\.deb",
                unquote(name, errors="strict"),
            )
            or name in names
        ):
            raise ValueError(f"unsafe or duplicate APT cache name: {name!r}")
        if int(size) <= 0:
            raise ValueError(f"invalid package size: {size!r}")
        names.add(name)
        packages.append(_ArchivePackage(_archive_uri(uri), name, int(size), digest.lower()))
    return packages


class _RejectArchiveRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        fp.close()
        raise ValueError(f"Archive carrier redirect is not permitted: {newurl!r}")


def _cache_archive(package: _ArchivePackage, archives: Path) -> None:
    """Publish a complete verified package, never a partial destination."""
    destination = archives / package.filename
    if destination.is_symlink():
        raise ValueError(f"APT cache destination is a symbolic link: {destination}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".ci-apt-", suffix=".partial", dir=archives, delete=False
        ) as target:
            temporary = Path(target.name)
            digest = hashlib.sha256()
            size = 0
            with build_opener(_RejectArchiveRedirects()).open(package.uri) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > package.size:
                        raise ValueError(f"Archive package exceeds expected size: {package.filename}")
                    digest.update(chunk)
                    target.write(chunk)
            if size != package.size or digest.hexdigest() != package.sha256:
                raise ValueError(f"Archive package size or SHA-256 mismatch: {package.filename}")
            target.flush()
            os.fsync(target.fileno())
            os.fchmod(target.fileno(), 0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _bullseye_install(
    packages: list[str],
    *,
    no_install_recommends: bool,
    archives: Path,
) -> int:
    """Resolve from authenticated snapshot metadata, then verify carrier bytes.

    Only the preceding complete signed update establishes metadata trust.
    URI output and no-download mode do not authenticate repository metadata.
    The final APT install retains package validation without a bypass.
    The outer coreutils timeout controls this worker and all its children.
    The parent owns one private cache for all attempts and removes it afterward.
    """
    install = [
        "apt-get", "install", "-y", "-o", "Acquire::ForceHash=SHA256",
        "-o", f"Dir::Cache::archives={archives}",
    ]
    if no_install_recommends:
        install.append("--no-install-recommends")
    try:
        plan = subprocess.run(
            install + ["--print-uris", "--", *packages],
            check=False,
            stdout=subprocess.PIPE,
            text=True,
        )
        print(plan.stdout, end="", flush=True)
        if plan.returncode != 0:
            return plan.returncode
        selected = _parse_archive_plan(plan.stdout)
        for package in selected:
            if (archives / package.filename).is_symlink():
                raise ValueError(f"APT cache destination is a symbolic link: {package.filename}")
        archives.mkdir(parents=True, exist_ok=True)
        for package in selected:
            _cache_archive(package, archives)
    except (OSError, ValueError, HTTPException) as exc:
        print(f"ci_apt_get: Bullseye carrier failed: {exc}", file=sys.stderr)
        return 1
    return _run(install + ["--no-download", "--", *packages])


def bullseye_snapshot_sources() -> str:
    """Return the complete immutable apt source set for the Bullseye lanes."""
    main = (
        "https://snapshot-cloudflare.debian.org/archive/debian/"
        f"{BULLSEYE_SNAPSHOT_TIMESTAMP}/"
    )
    security = (
        "https://snapshot-cloudflare.debian.org/archive/debian-security/"
        f"{BULLSEYE_SNAPSHOT_TIMESTAMP}/"
    )
    option = "[check-valid-until=no]"
    return (
        f"deb {option} {main} bullseye main\n"
        f"deb {option} {main} bullseye-updates main\n"
        f"deb {option} {security} bullseye-security main\n"
    )


def configure_bullseye_snapshot(apt_root: Path = Path("/etc/apt")) -> None:
    """Replace moving Debian sources with the pinned Bullseye snapshot.

    Both traditional ``*.list`` files and deb822 ``*.sources`` fragments are
    removed so a future base-image source-layout change cannot silently
    reintroduce a moving mirror beside the pinned source. The directory must
    already be an apt configuration root; refusing to create an arbitrary
    path makes a wrong container fail loudly.
    """
    if not apt_root.is_dir():
        raise FileNotFoundError(
            f"apt configuration directory does not exist: {apt_root}"
        )

    fragments = apt_root / "sources.list.d"
    if fragments.is_dir():
        for pattern in ("*.list", "*.sources"):
            for source in fragments.glob(pattern):
                source.unlink()

    (apt_root / "sources.list").write_text(
        bullseye_snapshot_sources(), encoding="utf-8"
    )
    print(
        "ci_apt_get: configured immutable Debian Bullseye snapshot "
        f"{BULLSEYE_SNAPSHOT_TIMESTAMP}"
    )


def configure_hosted_ubuntu_mirrors(apt_root: Path = Path("/etc/apt")) -> None:
    """Select the official HTTPS archives in the runner's existing mirror list.

    Run as root, before apt update/install. GitHub's Ubuntu images refer to
    this list from their ordinary source configuration. A missing list is a
    runner-contract change, so fail rather than create an unused policy file.
    """
    mirrors = apt_root / "apt-mirrors.txt"
    if not mirrors.is_file():
        raise FileNotFoundError(f"GitHub-hosted Ubuntu mirror list is missing: {mirrors}")
    mirrors.write_text(
        "https://archive.ubuntu.com/ubuntu/\tpriority:1\n"
        "https://security.ubuntu.com/ubuntu/\tpriority:2\n",
        encoding="utf-8",
    )
    print("ci_apt_get: selected official HTTPS Ubuntu archive mirrors", flush=True)


def _run(cmd: list[str]) -> int:
    """Run `cmd`, inheriting stdio so apt's output lands in the CI log.
    Returns the exit code (or 1 if it could not even be spawned).

    The wall-clock ceiling is NOT enforced here via subprocess.run(timeout=):
    that would SIGKILL our direct child (`sudo`) and orphan the apt-get it
    spawned, leaving the apt lock held. The ceiling is baked into `cmd`
    itself as a coreutils `timeout` wrapper (see _timeout_wrapper)."""
    try:
        return subprocess.run(cmd, check=False).returncode
    except OSError as exc:
        print(f"ci_apt_get: could not run {cmd!r}: {exc}", file=sys.stderr)
        return 1


def _timeout_wrapper(timeout: float | None) -> list[str]:
    """The coreutils `timeout` prefix tokens for one apt-get command, or []
    when the ceiling is disabled. Placed AFTER any `sudo` so `timeout` is
    apt-get's direct parent and the kill actually reaches apt-get.

    `--foreground` is deliberately NOT passed: without it `timeout` puts the
    command in its own process group and signals the WHOLE group, so apt-get's
    `/usr/lib/apt/methods/*` fetcher children are cleaned up with it. That is
    what we want in a non-interactive `run:` step (there is no TTY to hand
    back); `--foreground` would leave those children un-timed-out."""
    if timeout is None:
        return []
    return ["timeout", f"--kill-after={KILL_AFTER_SECONDS:g}s", f"{timeout:g}s"]


def _attempt(
    packages: list[str],
    *,
    sudo: bool,
    no_install_recommends: bool,
    timeout: float | None,
    archives: Path | None,
) -> int:
    """One full update+install attempt. Returns the first non-zero exit of
    the pair, or 0 if both succeed."""
    # `sudo` first (privilege), then `timeout` (so it parents apt-get), then
    # the apt-get command itself.
    prefix = (["sudo"] if sudo else []) + _timeout_wrapper(timeout)
    update = prefix + ["apt-get", "update"]
    if archives is not None:
        update += [
            "-o", "APT::Update::Error-Mode=any",
            "-o", "Acquire::AllowInsecureRepositories=false",
            "-o", "Acquire::AllowDowngradeToInsecureRepositories=false",
        ]
        install = prefix + [
            sys.executable, str(Path(__file__).resolve()),
            _BULLSEYE_WORKER_FLAG, str(archives),
            "--timeout", f"{timeout:g}" if timeout is not None else "0",
        ]
    else:
        install = prefix + ["apt-get", "install", "-y"]
    if no_install_recommends:
        install.append("--no-install-recommends")
    install += ["--", *packages] if archives is not None else packages

    rc = _run(update)
    if rc != 0:
        return rc
    return _run(install)


def apt_get(
    packages: list[str],
    *,
    sudo: bool = True,
    no_install_recommends: bool = False,
    debian_bullseye_snapshot: bool = False,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    timeout: float | None = PER_COMMAND_TIMEOUT_SECONDS,
    sleep=time.sleep,
) -> int:
    """Run the update+install pair, retrying up to `attempts` times on any
    non-zero exit (a reset, a spawn failure, or a per-command timeout).
    Returns 0 on success or the last attempt's exit code.

    `timeout` bounds each update and install phase in wall-clock seconds.
    The Bullseye install phase includes its plan and byte transfer.
    None disables the ceiling. Tests inject `sleep` to avoid a delay."""
    if not packages:
        print("ci_apt_get: no packages requested; nothing to do.")
        return 0

    cache = (
        tempfile.TemporaryDirectory(prefix="ci-apt-archives-")
        if debian_bullseye_snapshot else nullcontext(None)
    )
    with cache as directory:
        archives = Path(directory) if directory is not None else None
        last_rc = 1
        for i in range(1, attempts + 1):
            print(f"ci_apt_get: attempt {i}/{attempts}: installing {' '.join(packages)}")
            last_rc = _attempt(
                packages,
                sudo=sudo,
                no_install_recommends=no_install_recommends,
                timeout=timeout,
                archives=archives,
            )
            if last_rc == 0:
                return 0
            if i < attempts:
                if last_rc == 124:
                    cause = "timed out at the wall-clock ceiling"
                elif last_rc == 137:
                    cause = "SIGKILLed (--kill-after, or an external signal such as OOM)"
                else:
                    cause = "likely a transient mirror/network reset"
                print(
                    f"ci_apt_get: apt-get exited {last_rc} "
                    f"({cause}); retrying in {backoff_seconds:g}s",
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
        _BULLSEYE_WORKER_FLAG, type=Path, help=argparse.SUPPRESS
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
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--github-hosted-ubuntu-mirrors",
        action="store_true",
        help="select official HTTPS Ubuntu archives in the hosted runner mirror list (run as root).",
    )
    sources.add_argument(
        "--debian-bullseye-snapshot",
        action="store_true",
        help=(
            "replace all apt sources with the repository-pinned immutable "
            "Debian Bullseye snapshot before updating"
        ),
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


def _wait_for_timeout_kill(_signum, _frame) -> None:
    """Keep the timeout monitor alive until its process-group kill."""
    while True:
        signal.pause()


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if args._bullseye_install_worker:
        # Retain the monitor until its group-wide kill escalation.
        # Child executables retain their normal SIGTERM behavior.
        if args.timeout and args.timeout > 0:
            signal.signal(signal.SIGTERM, _wait_for_timeout_kill)
        return _bullseye_install(
            args.packages, no_install_recommends=args.no_install_recommends,
            archives=args._bullseye_install_worker,
        )
    if args.github_hosted_ubuntu_mirrors:
        configure_hosted_ubuntu_mirrors()
    if args.debian_bullseye_snapshot:
        configure_bullseye_snapshot()
    # A non-positive --timeout disables the ceiling: _timeout_wrapper returns
    # no `timeout` prefix and apt-get runs unbounded.
    timeout = args.timeout if args.timeout and args.timeout > 0 else None
    return apt_get(
        args.packages,
        sudo=not args.no_sudo,
        no_install_recommends=args.no_install_recommends,
        debian_bullseye_snapshot=args.debian_bullseye_snapshot,
        attempts=args.attempts,
        timeout=timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
