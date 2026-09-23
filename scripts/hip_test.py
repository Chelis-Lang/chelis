#!/usr/bin/env python3
"""Run Chelis HIP tests with the reconciled hipBLAS env.

The systemd `~/.config/environment.d/hip.conf` only sets the `-isystem` half
of `HIPCC_COMPILE_FLAGS_APPEND` and `HSA_OVERRIDE_GFX_VERSION=11.0.0`. That is
enough for non-hipBLAS HIP tests but causes a process-exit SIGSEGV (empty
stdout/stderr) on any test that links hipBLAS, because the linker resolves
`libhipblas` against a mismatched stack.

For hipBLAS gates the authoritative env (per `docs/local_hip_environment.md`
§3) is:

    HSA_OVERRIDE_GFX_VERSION=11.5.1
    LD_LIBRARY_PATH=<wheel core>/lib:<wheel gfx1151>/lib
    HIPCC_COMPILE_FLAGS_APPEND="-isystem <wheel core>/include
                                -L<wheel gfx1151>/lib"

This script verifies the wheel paths exist on disk, prints the resolved env,
and execs `cargo test` with the remaining args. Use:

    scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- \
        --ignored --test-threads=1

    scripts/hip_test.py -p chelis-cli --test cross_library_semantic_gap_hip_gpu \
        -- --ignored --test-threads=1

Pass `--print-env` to print the env block instead of running cargo (useful for
`eval $(scripts/hip_test.py --print-env)` in a shell).
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

WHEEL_CORE = Path.home() / ".local/lib/python3.12/site-packages/_rocm_sdk_core"
WHEEL_GFX = Path.home() / ".local/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151"

HSA_OVERRIDE_GFX_VERSION = "11.5.1"


def verify_wheel_paths() -> list[str]:
    """Return a list of human-readable diagnostics for any missing wheel paths."""
    diagnostics: list[str] = []
    required = [
        (WHEEL_CORE / "include", "wheel core include dir"),
        (WHEEL_CORE / "lib", "wheel core lib dir"),
        (WHEEL_GFX / "lib", "wheel gfx1151 lib dir"),
        (WHEEL_GFX / "lib" / "libhipblas.so", "wheel gfx1151 libhipblas.so symlink"),
    ]
    for path, label in required:
        if not path.exists():
            diagnostics.append(f"{label} missing at {path}")
    return diagnostics


def build_env() -> dict[str, str]:
    core_lib = str(WHEEL_CORE / "lib")
    gfx_lib = str(WHEEL_GFX / "lib")
    core_include = str(WHEEL_CORE / "include")

    # Compose LD_LIBRARY_PATH preserving any existing entries.
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [core_lib, gfx_lib]
    if ld_existing:
        ld_parts.append(ld_existing)
    ld_library_path = ":".join(ld_parts)

    # HIPCC flags: always include -isystem; for hipBLAS also include -L.
    hipcc_flags = f"-isystem {core_include} -L{gfx_lib}"

    return {
        "HSA_OVERRIDE_GFX_VERSION": HSA_OVERRIDE_GFX_VERSION,
        "LD_LIBRARY_PATH": ld_library_path,
        "HIPCC_COMPILE_FLAGS_APPEND": hipcc_flags,
    }


USAGE = """\
usage: scripts/hip_test.py [--print-env | --help | --cargo-subcommand <cmd>] <cargo args>...

  --print-env              Print shell-eval-able export statements and exit.
  --cargo-subcommand CMD   Cargo subcommand (default: test).
  --help, -h               Show this help.

Everything else is forwarded verbatim to `cargo <subcommand>`. Example:

  scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- \\
      --ignored --test-threads=1
"""


def main() -> int:
    argv = sys.argv[1:]
    print_env = False
    cargo_subcommand = "test"
    cargo_args: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--help", "-h"):
            print(USAGE)
            return 0
        if a == "--print-env":
            print_env = True
            i += 1
            continue
        if a == "--cargo-subcommand":
            if i + 1 >= len(argv):
                print("error: --cargo-subcommand requires an argument", file=sys.stderr)
                return 2
            cargo_subcommand = argv[i + 1]
            i += 2
            continue
        # Anything else, including leading-dash flags, goes to cargo.
        cargo_args = argv[i:]
        break

    diagnostics = verify_wheel_paths()
    if diagnostics:
        for line in diagnostics:
            print(f"error: {line}", file=sys.stderr)
        print(
            "\nThis script assumes the ROCm wheels documented in "
            "docs/local_hip_environment.md are installed. If the wheels were "
            "relocated or upgraded, update the WHEEL_CORE/WHEEL_GFX paths in "
            "this script and the doc.",
            file=sys.stderr,
        )
        return 2

    overrides = build_env()

    if print_env:
        for k, v in overrides.items():
            print(f"export {k}={shlex.quote(v)}")
        return 0

    if not cargo_args:
        print(USAGE, file=sys.stderr)
        print(
            "\nerror: no cargo args provided",
            file=sys.stderr,
        )
        return 2

    print("scripts/hip_test.py setting hipBLAS env:", file=sys.stderr)
    for k, v in overrides.items():
        print(f"  {k}={v}", file=sys.stderr)
    print("", file=sys.stderr)

    env = os.environ.copy()
    env.update(overrides)

    argv_full = ["cargo", cargo_subcommand, *cargo_args]
    print(f"+ {shlex.join(argv_full)}", file=sys.stderr)
    os.execvpe("cargo", argv_full, env)
    # execvpe does not return on success.
    return 1


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        sys.exit(main())
