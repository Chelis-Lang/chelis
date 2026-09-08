#!/usr/bin/env python3
"""Reclaim disk on GitHub-hosted Ubuntu CI runners before the Rust build.

The `lint-rust` and `workspace-tests-shard` Linux workers compile or test the
full Cargo workspace. The debug `target/` plus all build artifacts pushes
the runner past its disk ceiling intermittently, surfacing as
`No space left on device (os error 28)` during an in-test `gcc` build or
`/usr/bin/ld: final link failed: No space left on device`
(`ld terminated with signal 7 [Bus error]`) while linking the `chelis`
binary. Those failures land on whatever crate happens to be building at the
moment (often unrelated `chelis-backend-c` f16 tests), so they read like a
code regression but are purely environmental. The macOS workspace workers
(more disk) run the same partitioned `cargo nextest run --workspace` suite.

This mirrors the `Free Disk Space (Ubuntu)` step the neoteny repo runs in
its CI: it removes large pre-installed SDK toolchains that the chelis
Rust/C workspace never links against (~25-30 GB: the Android SDK, .NET,
GHC/Haskell, Swift, PowerShell, Boost, the JVMs, and the CodeQL bundle)
and prunes any cached Docker images.

Invoked from `.github/workflows/ci.yml` as `python3 scripts/ci_free_disk.py`,
mirroring the existing `python3 scripts/ci_setup_uv_python.py` convention so
the gate jobs keep single-line `run:` steps (the `scripts/test_gate.py`
parity lock disallows multi-line `run:` blocks and hand-inlined cargo/chelis
commands in those jobs). Per repo policy this is Python, not a shell script.

Best-effort and non-fatal by design: a missing path or a failed remove must
never fail CI, and the script is a no-op off Linux (so running it locally on
macOS does nothing). Only `/opt/hostedtoolcache/CodeQL` is removed from the
tool cache — not the whole `$AGENT_TOOLSDIRECTORY` — so the Rust toolchain,
`cargo-nextest`, and `uv` installs that follow are left intact.
"""

from __future__ import annotations

import concurrent.futures
import shutil
import subprocess
import sys

# Large pre-installed toolchains on the ubuntu-latest GitHub runner image
# that the chelis workspace never uses. Removing them reclaims ~25-30 GB.
# Targeted subdirectories only (not all of /opt/hostedtoolcache) so the
# toolchain/tool installs that run after this step are unaffected.
PURGE_PATHS: tuple[str, ...] = (
    "/usr/share/dotnet",
    "/usr/local/lib/android",
    "/opt/ghc",
    "/usr/local/.ghcup",
    "/usr/share/swift",
    "/usr/local/share/powershell",
    "/usr/local/share/boost",
    "/usr/lib/jvm",
    "/opt/hostedtoolcache/CodeQL",
)


def _run(cmd: list[str]) -> int:
    """Run `cmd`, never raising. Returns its exit code (or 1 if it could
    not be spawned). Output is inherited so it shows up in the CI log."""
    try:
        return subprocess.run(cmd, check=False).returncode
    except OSError as exc:  # e.g. the binary is missing on this platform
        print(f"ci_free_disk: could not run {cmd!r}: {exc}")
        return 1


def _df() -> None:
    """Print `df -h /` for before/after observability; ignore failures."""
    try:
        out = subprocess.run(
            ["df", "-h", "/"], check=False, capture_output=True, text=True
        ).stdout
        print(out.rstrip())
    except OSError:
        print("ci_free_disk: df unavailable")


def free_disk_space() -> int:
    """Reclaim runner disk. Always returns 0 — this step must never fail
    CI, since it is a pure best-effort optimization."""
    if not sys.platform.startswith("linux"):
        print(f"ci_free_disk: non-Linux platform {sys.platform!r}; nothing to do.")
        return 0

    print("ci_free_disk: disk usage before cleanup:")
    _df()

    # `sudo` is required: these live under root-owned system dirs on the
    # runner. A missing path or a non-zero exit is fine.
    #
    # The removals are disjoint directory trees, and the step is dominated
    # by unlink syscalls rather than CPU, so running them one at a time
    # left the runner blocked on I/O for the whole step: measured 0.8-3.5
    # min per job across the five Linux jobs that call this. Dispatching
    # them together overlaps that wait. Threads (not processes) are the
    # right tool because every worker blocks inside `subprocess.run`,
    # which releases the GIL. The docker prune joins the same batch: it is
    # independent of every path above.
    commands = [["sudo", "rm", "-rf", path] for path in PURGE_PATHS]
    if shutil.which("docker") is not None:
        commands.append(["sudo", "docker", "image", "prune", "--all", "--force"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(commands)) as pool:
        # `map` yields in submission order, so the log stays deterministic
        # even though the removals do not finish in that order.
        codes = list(pool.map(_run, commands))

    for cmd, rc in zip(commands, codes):
        print(f"ci_free_disk: {' '.join(cmd[1:])} -> exit {rc}")

    print("ci_free_disk: disk usage after cleanup:")
    _df()
    return 0


if __name__ == "__main__":
    raise SystemExit(free_disk_space())
