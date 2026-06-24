#!/usr/bin/env python3
"""Run Chelis `z3`-feature tests against a PREBUILT libz3.

The `z3` cargo feature (WI-12 / WS-5) builds `z3` = 0.20 / `z3-sys` = 0.11 with
`default-features = false`, so z3-sys does NOT vendor/build Z3 from source (no
cmake). z3-sys links a prebuilt libz3 via `#[link(name = "z3")]`; its build.rs
adds a link-search path from the `Z3_LIBRARY_PATH_OVERRIDE` env var, and the
runtime needs that same dir on `LD_LIBRARY_PATH`.

On a CI runner the prebuilt libz3 is the apt `libz3-dev` package (already on the
default link/loader paths, so the override is a no-op there). On this
workstation the prebuilt libz3 ships inside the python `z3` package's
site-packages, which is NOT on the default paths; this wrapper finds it and sets
both env vars, then execs the cargo command with `--features z3` injected.

Usage (defaults to `cargo nextest run -p chelis-prove --features z3`):

    scripts/z3_test.py
    scripts/z3_test.py --test cross_engine_oracle
    scripts/z3_test.py --features "smt z3" --test cross_engine_oracle
    scripts/z3_test.py --cargo-subcommand "clippy" -p chelis-prove --features z3 -- -D warnings

Pass `--print-env` to print shell-eval-able export statements instead of running
cargo (useful for `eval $(scripts/z3_test.py --print-env)`).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

# Candidate locations for a prebuilt libz3, in priority order. The python `z3`
# package ships `libz3.so` under `<site-packages>/z3/lib`; the apt `libz3-dev`
# package installs to the default loader path (so an empty override is fine).
def candidate_lib_dirs() -> list[Path]:
    dirs: list[Path] = []
    # 1. An explicit override always wins.
    override = os.environ.get("Z3_LIBRARY_PATH_OVERRIDE")
    if override:
        dirs.append(Path(override))
    # 2. The python z3 package's bundled libz3 (this workstation). Ask any
    #    interpreter that can `import z3` where the package lives (so the python
    #    minor version is not hardcoded).
    for py in (sys.executable, "python3"):
        try:
            out = subprocess.run(
                [py, "-c", "import z3, os; print(os.path.dirname(z3.__file__))"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0:
            pkg = Path(out.stdout.strip())
            if pkg:
                dirs.append(pkg / "lib")
    # 3. The user-site python z3 install, even when no interpreter on PATH has
    #    it importable (the workstation ships it under ~/.local for a python
    #    minor that is not the default `python3`). Glob the user site-packages.
    for site in sorted((Path.home() / ".local/lib").glob("python3.*/site-packages/z3/lib")):
        dirs.append(site)
    return dirs


def dir_has_linkable_libz3(d: Path) -> bool:
    """Whether `d` holds an UNVERSIONED `libz3.so`. z3-sys links `-lz3`, which
    needs that exact symlink, not only a versioned `libz3.so.4.15` -- so a dir
    with only the versioned file is NOT a usable link-search target."""
    return (d / "libz3.so").exists()


def find_libz3_dir() -> Path | None:
    """The first candidate dir that actually contains a linkable `libz3.so`."""
    for d in candidate_lib_dirs():
        if dir_has_linkable_libz3(d):
            return d
    return None


def system_libz3_available() -> bool:
    """Whether the system loader can already find libz3 (apt libz3-dev on CI)."""
    try:
        out = subprocess.run(
            ["ldconfig", "-p"], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return out.returncode == 0 and "libz3.so" in out.stdout


def build_env(lib_dir: Path | None) -> dict[str, str]:
    """Compose the Z3 link env. With a discovered `lib_dir`, set both the
    build-time link-search override and the runtime loader path; with none
    (the system-libz3 case) return an empty override (apt libz3-dev is on the
    default paths)."""
    if lib_dir is None:
        return {}
    ld_existing = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [str(lib_dir)]
    if ld_existing:
        ld_parts.append(ld_existing)
    return {
        "Z3_LIBRARY_PATH_OVERRIDE": str(lib_dir),
        "LD_LIBRARY_PATH": ":".join(ld_parts),
    }


USAGE = """\
usage: scripts/z3_test.py [--print-env | --help | --cargo-subcommand <cmd>] <cargo args>...

  --print-env              Print shell-eval-able export statements and exit.
  --cargo-subcommand CMD   Cargo subcommand (default: "nextest run").
  --help, -h               Show this help.

With no cargo args the default is:
  cargo nextest run -p chelis-prove --features z3

Everything after the flags is forwarded verbatim to cargo. Example:

  scripts/z3_test.py --features "smt z3" --test cross_engine_oracle
"""

DEFAULT_ARGS = ["-p", "chelis-prove", "--features", "z3"]


def main() -> int:
    argv = sys.argv[1:]
    print_env = False
    cargo_subcommand = "nextest run"
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
        cargo_args = argv[i:]
        break

    lib_dir = find_libz3_dir()
    if lib_dir is None and not system_libz3_available():
        print(
            "error: no prebuilt libz3.so found. Install the python `z3` package "
            "(this workstation) or the apt `libz3-dev` package (CI), or set "
            "Z3_LIBRARY_PATH_OVERRIDE to the directory containing libz3.so. See "
            "docs/local_z3_environment.md.",
            file=sys.stderr,
        )
        return 2

    overrides = build_env(lib_dir)

    if print_env:
        for k, v in overrides.items():
            print(f"export {k}={shlex.quote(v)}")
        return 0

    if not cargo_args:
        cargo_args = DEFAULT_ARGS

    if overrides:
        print("scripts/z3_test.py setting z3 link env:", file=sys.stderr)
        for k, v in overrides.items():
            print(f"  {k}={v}", file=sys.stderr)
        print("", file=sys.stderr)
    else:
        print(
            "scripts/z3_test.py: using system libz3 (no override needed)",
            file=sys.stderr,
        )

    env = os.environ.copy()
    env.update(overrides)

    argv_full = ["cargo", *shlex.split(cargo_subcommand), *cargo_args]
    print(f"+ {shlex.join(argv_full)}", file=sys.stderr)
    os.execvpe("cargo", argv_full, env)
    # execvpe does not return on success.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
