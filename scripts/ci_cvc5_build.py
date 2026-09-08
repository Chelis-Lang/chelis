#!/usr/bin/env python3
"""Run a cvc5-linked release build, retrying ONLY a transient source fetch.

`release.yml` builds cvc5 cold from source in all three release jobs. That
is deliberate and must stay that way: `docs/smt_build_setup.md` records that
the shipped binary is an INDEPENDENT, license-safe proof of the exact recipe
(WI-11: `ENABLE_GPL=OFF`, `USE_CLN=OFF`) rather than trusting an asset
produced by `build-cvc5.yml`, and `ci.yml`'s darwin SMT lane names the
release job "the authoritative cold-build proof for the SHIPPED artifact".
The per-PR SMT lanes link the durable prebuilt via `ci_cvc5_cache.py`, so
they never exercise this path at all. This wrapper does not weaken any of
that -- it never links a prebuilt and never skips a compile.

What it does fix (chelis#1004): the cold build begins by pulling cvc5 and
its dependencies over the network, and a transient GitHub refusal there
fails the release job. Because `publish-release` has `needs:` on all three
build jobs, ONE such failure skips the publish and leaves a pushed tag with
no GitHub Release. The v0.18.0 release (run 30673030685) needed three
attempts for exactly this reason, failing at two different fetch sites with
no change to the tree in between:

    error: downloading '.../cvc5-deps/blob/main/gmp-6.3.0.tar.bz2?raw=true' failed
            The requested URL returned error: 403
    thread 'main' panicked at cvc5-sys-0.3.1/build.rs:231:5: cvc5 build failed

    thread 'main' panicked at cvc5-sys-0.3.1/build.rs:276:5:
    git clone of cvc5 tag cvc5-1.3.1 failed

Both Linux jobs built cvc5 from source successfully in that same run, so
this was runner-side network trouble, not a recipe problem.

The retry is SIGNATURE-GATED, not blind, and that distinction is the whole
point. "Treat a `release.yml` cvc5 failure as real" stays true because a
compile error, a CMake configure error, or a link error is NOT retried: it
fails on the first attempt exactly as it does today. Only a failure whose
output carried a transient-fetch signature is retried, and even then only a
bounded number of times before the exit code propagates. A retry loop that
fired on any non-zero exit would be able to turn a genuinely broken cvc5
build green on a flaky-compile day, which is precisely the property the
cold-build proof exists to deny.

Usage (single-line `run:` step, mirroring the `ci_apt_get.py` /
`ci_free_disk.py` convention so the step stays Python, not shell, per repo
policy):

    python3 scripts/ci_cvc5_build.py -- cargo build --release -p chelis-cli --features smt

The command is passed through verbatim rather than hardcoded here. That is
deliberate on two counts: this wrapper stays a generic
retry-the-cvc5-fetch harness rather than a second place where the release
build recipe is written down, and `scripts/test_release_workflow_pyo3_isolation.py`
parses `release.yml` for `cargo build -p <name>` to decide which crates to
vet for pyo3. Hiding the real command inside this script would make that
parser find nothing, and its `chelis-cli` check would then SKIP itself
rather than fail -- a live guard silently retired by a refactor.

Per repo policy this is Python, not a shell script.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

# Bounded attempts. Three is enough to ride out the observed failure mode
# (a fetch refusal that clears within minutes) without spending a whole
# runner-hour on a GitHub-side outage that is not going to clear. On
# exhaustion the last exit code propagates and the release job fails, which
# is the correct outcome: a sustained outage is not something CI should
# paper over.
DEFAULT_ATTEMPTS = 3

# Seconds to sleep before each retry (not before the first attempt). Longer
# than ci_apt_get.py's 5s because the observed refusals were HTTP 403 from
# github.com rather than a CDN connection reset: a rate-limit style refusal
# wants a pause long enough for a token bucket to refill, not an immediate
# hammer that just consumes another attempt.
BACKOFF_SECONDS = 30

# Substrings (matched case-insensitively) that mark a failure as a transient
# SOURCE FETCH failure rather than a real build failure. Keep this list
# narrow: every entry is a licence to retry, and a signature that can also
# be emitted by a genuine compile failure would silently re-open the hole
# this script is written to keep closed.
#
# The first four were observed verbatim in run 30673030685; the rest are the
# standard curl/git transport failures that reach the same two fetch sites
# in cvc5-sys's build.rs (231 = CMake ExternalProject downloads, 276 = the
# cvc5 source clone).
TRANSIENT_SIGNATURES: tuple[str, ...] = (
    "the requested url returned error: 403",
    "the requested url returned error: 429",
    "each download failed!",
    "git clone of cvc5 tag",
    "error: downloading '",
    "the requested url returned error: 5",
    "could not resolve host",
    "connection reset by peer",
    "connection timed out",
    "failed to connect to github.com",
    "ssl connect error",
    "unexpected disconnect while reading sideband packet",
    "the remote end hung up unexpectedly",
    "early eof",
    "rpc failed",
)


def classify_line(line: str) -> bool:
    """True if `line` carries a transient-fetch signature."""
    lowered = line.lower()
    return any(sig in lowered for sig in TRANSIENT_SIGNATURES)


def run_streaming(cmd: list[str], *, echo=None) -> tuple[int, bool]:
    """Run `cmd`, streaming merged stdout/stderr to our stdout as it arrives.

    Returns `(exit_code, saw_transient)`. Output is streamed rather than
    captured-then-printed so the CI log stays live during a ~22 minute cvc5
    compile and so a hung build still shows where it stopped. Lines are
    classified as they stream, so nothing is buffered for the whole run.

    `echo` is injectable for the tests; it defaults to writing through to
    stdout with a flush, because a GitHub runner log that is missing the
    build's own output is not worth retrying against.
    """
    if echo is None:

        def echo(text: str) -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

    saw_transient = False
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except OSError as exc:
        # A command we cannot even spawn is a configuration error, not a
        # network flake: report it as non-transient so it fails immediately.
        print(f"ci_cvc5_build: could not run {cmd!r}: {exc}", file=sys.stderr)
        return 1, False

    assert proc.stdout is not None
    for line in proc.stdout:
        echo(line)
        if classify_line(line):
            saw_transient = True
    return proc.wait(), saw_transient


def build(
    cmd: list[str],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    backoff_seconds: float = BACKOFF_SECONDS,
    sleep=time.sleep,
    runner=run_streaming,
) -> int:
    """Run `cmd`, retrying only when the failing attempt showed a transient
    fetch signature. Returns 0 on success, else the last attempt's code.

    `sleep` and `runner` are injectable so the unit tests neither wait nor
    spawn a real build.
    """
    if not cmd:
        print("ci_cvc5_build: no command given; nothing to do.", file=sys.stderr)
        return 2

    last_rc = 0
    for attempt in range(1, attempts + 1):
        if attempt > 1:
            print(
                f"ci_cvc5_build: transient cvc5 fetch failure; "
                f"retrying in {backoff_seconds:g}s "
                f"(attempt {attempt}/{attempts})",
                flush=True,
            )
            sleep(backoff_seconds)
        rc, saw_transient = runner(cmd)
        if rc == 0:
            if attempt > 1:
                print(f"ci_cvc5_build: succeeded on attempt {attempt}/{attempts}")
            return 0
        last_rc = rc
        if not saw_transient:
            # The cold-build proof: a real cvc5 failure must fail now, with
            # no retry to muddy whether the recipe actually builds.
            print(
                f"ci_cvc5_build: build failed (exit {rc}) with no transient "
                "fetch signature. NOT retrying -- this is a real build "
                "failure, per docs/smt_build_setup.md.",
                file=sys.stderr,
            )
            return rc
        if attempt == attempts:
            print(
                f"ci_cvc5_build: transient fetch failure persisted across "
                f"{attempts} attempts; giving up (exit {rc}).",
                file=sys.stderr,
            )
    return last_rc


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run a cvc5-linked build, retrying only a transient source-fetch "
            "failure. A real build failure is never retried."
        ),
    )
    p.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help=f"maximum attempts (default {DEFAULT_ATTEMPTS})",
    )
    p.add_argument(
        "--backoff-seconds",
        type=float,
        default=BACKOFF_SECONDS,
        help=f"sleep before each retry (default {BACKOFF_SECONDS})",
    )
    p.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="the build command, after a literal `--`",
    )
    args = p.parse_args(argv)
    # argparse.REMAINDER keeps the leading `--` when it is the first token.
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.attempts < 1:
        print("ci_cvc5_build: --attempts must be >= 1", file=sys.stderr)
        return 2
    return build(
        args.command,
        attempts=args.attempts,
        backoff_seconds=args.backoff_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
