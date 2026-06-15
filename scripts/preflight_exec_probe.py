#!/usr/bin/env python3
"""Preflight probe for the macOS first-exec (syspolicyd) wedge.

Background: on macOS, `syspolicyd` can enter an error loop
(`Unable to initialize qtn_proc: 3` / `dispatch_mig_server returned
268435459`) that makes freshly linked binaries hang in `_dyld_start` on
their first exec — for minutes, or indefinitely under parallel
first-exec bursts. Every `cargo build` produces fresh binaries, so the
wedge poisons exactly the edit-build-test loop while already-assessed
binaries keep running fine. The full runbook is
`docs/local_macos_environment.md`.

This probe answers one question cheaply: "will a freshly linked binary
exec right now?" It writes a trivial C file into a private temp dir,
compiles it with `cc`, and execs the result under a timeout.

Exit codes:
    0   first exec completed promptly; prints `exec ok (N ms)`
    1   first exec timed out; first-exec assessment appears wedged
    2   the probe could not run (`cc` missing, compile failed, probe
        not executable, or probe exited non-zero) — an environment
        problem, not a wedge verdict
    3   first exec completed but took longer than `--warn-ms`
        (default 2000): assessment is admitting binaries slowly — the
        silent degradation variant (chelis#356). Multi-binary test
        runs will crawl even though single execs eventually succeed.

A passing probe is a point-in-time result, NOT a session clearance:
degradation is volume-induced and can begin minutes later once a mass
of freshly built binaries hits assessment (observed: probe at 377 ms,
stalls within the hour under a workspace build burst — chelis#356).

Usage:
    python3 scripts/preflight_exec_probe.py                # default 15s timeout
    python3 scripts/preflight_exec_probe.py --timeout 60   # patient variant

The script is safe to run from any cwd: it does all work inside a
`tempfile.TemporaryDirectory` and always cleans it up.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

RUNBOOK = "docs/local_macos_environment.md"
TEMP_DIR_PREFIX = "chelis-preflight-exec-probe-"
DEFAULT_TIMEOUT_SECONDS = 15.0
COMPILE_TIMEOUT_SECONDS = 120.0

EXIT_OK = 0
EXIT_WEDGED = 1
EXIT_ENV = 2
EXIT_SLOW = 3

DEFAULT_WARN_MS = 2000.0

# The probe payload: the smallest program whose successful exit proves
# the binary was assessed and ran.
PROBE_SOURCE = "int main(void) { return 0; }\n"


class CompileError(RuntimeError):
    """The probe binary could not be produced (cc missing/failed)."""


def compile_probe(workdir: Path, source: str = PROBE_SOURCE) -> Path:
    """Compile `source` with `cc` inside `workdir` and return the path
    to the freshly linked binary. Raises CompileError when `cc` is
    unavailable or the compile fails."""
    c_file = workdir / "probe.c"
    binary = workdir / "probe"
    c_file.write_text(source)
    command = ["cc", "-o", str(binary), str(c_file)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=COMPILE_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise CompileError("cc not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise CompileError(
            f"cc did not finish within {COMPILE_TIMEOUT_SECONDS:g}s"
        ) from exc
    if result.returncode != 0:
        raise CompileError(
            f"cc failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    return binary


def exec_probe(binary: Path, timeout_seconds: float) -> float | None:
    """First-exec `binary` under `timeout_seconds`. Returns the elapsed
    wall-clock milliseconds on success, or None when the exec timed out
    (the wedge classification). Raises RuntimeError when the probe
    binary exits non-zero — the payload is `return 0`, so any other
    exit means the probe itself is broken, not the machine wedged."""
    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(binary)],
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    except OSError as err:
        # Exec refused outright (e.g. noexec TMPDIR, permissions): an
        # environment problem, not a wedge verdict.
        raise RuntimeError(f"probe binary could not be executed: {err}") from err
    elapsed_ms = (time.monotonic() - start) * 1000.0
    if result.returncode != 0:
        raise RuntimeError(
            f"probe binary exited {result.returncode}; expected 0"
        )
    return elapsed_ms


def positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"timeout must be > 0, got {text!r}"
        )
    return value


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compile and first-exec a trivial binary to detect the macOS "
            "syspolicyd first-exec wedge. Exit 0 = exec ok, 1 = wedged, "
            "2 = probe could not run, 3 = exec ok but slow (assessment "
            f"degrading). Runbook: {RUNBOOK}."
        ),
    )
    p.add_argument(
        "--timeout",
        type=positive_float,
        default=DEFAULT_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "How long to wait for the first exec before classifying the "
            "machine as wedged (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--warn-ms",
        type=positive_float,
        default=DEFAULT_WARN_MS,
        metavar="MS",
        help=(
            "First-exec latency above which the machine is classified as "
            "slowly admitting (exit 3), even though the exec succeeded "
            "(default: %(default)s). A healthy first exec is well under "
            "one second."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory(prefix=TEMP_DIR_PREFIX) as tmp:
        try:
            # Look up PROBE_SOURCE at call time (not via the def-time
            # default) so tests can substitute the payload.
            binary = compile_probe(Path(tmp), source=PROBE_SOURCE)
        except CompileError as exc:
            print(f"preflight_exec_probe: cannot run probe: {exc}", file=sys.stderr)
            return EXIT_ENV
        try:
            elapsed_ms = exec_probe(binary, args.timeout)
        except RuntimeError as exc:
            print(f"preflight_exec_probe: {exc}", file=sys.stderr)
            return EXIT_ENV
    if elapsed_ms is None:
        print(
            (
                f"preflight_exec_probe: WARNING: first exec of a freshly "
                f"linked binary did not complete within {args.timeout:g}s. "
                f"First-exec assessment appears wedged (macOS syspolicyd / "
                f"Gatekeeper); local exec results will be unreliable. "
                f"See {RUNBOOK} for diagnosis and mitigations."
            ),
            file=sys.stderr,
        )
        return EXIT_WEDGED
    if elapsed_ms > args.warn_ms:
        print(
            (
                f"preflight_exec_probe: WARNING: first exec succeeded but "
                f"took {elapsed_ms:.0f} ms (threshold {args.warn_ms:g} ms). "
                f"First-exec assessment is admitting binaries slowly (the "
                f"silent degradation variant, chelis#356); multi-binary "
                f"test runs will crawl. See {RUNBOOK}."
            ),
            file=sys.stderr,
        )
        return EXIT_SLOW
    print(f"exec ok ({elapsed_ms:.0f} ms)")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
