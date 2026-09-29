#!/usr/bin/env python3
"""Run Chelis tests that link against a prebuilt Z3 library.

The helper finds the platform's link library, adds its directory to Cargo's
link and runtime search paths, and forwards the remaining arguments to Cargo.
"""

from __future__ import annotations

import glob
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

LINK_LIBRARIES = {
    "Linux": ("libz3.so",),
    "Darwin": ("libz3.dylib",),
    "Windows": ("libz3.lib", "liblibz3.dll.a"),
}
LOADER_PATHS = {
    "Linux": ("LD_LIBRARY_PATH", ":"),
    "Darwin": ("DYLD_LIBRARY_PATH", ":"),
    "Windows": ("PATH", ";"),
}
SEARCH_PATHS = {
    "Linux": ("LD_LIBRARY_PATH", "LIBRARY_PATH"),
    "Darwin": ("DYLD_LIBRARY_PATH", "LIBRARY_PATH"),
    "Windows": ("LIB", "PATH"),
}


def candidate_lib_dirs(system: str | None = None) -> list[Path]:
    """Return likely directories for the host platform's linkable Z3 library."""
    system = system or platform.system()
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            dirs.append(path)

    override = os.environ.get("Z3_LIBRARY_PATH_OVERRIDE")
    if override:
        add(Path(override))

    # Include paths already configured for the host linker or runtime loader.
    path_separator = ";" if system == "Windows" else ":"
    for variable in SEARCH_PATHS.get(system, ()):
        for value in os.environ.get(variable, "").split(path_separator):
            if value:
                add(Path(value))

    # The Python package carries the matching shared library under z3/lib.
    interpreters = [sys.executable]
    fallback_python = shutil.which("python3") or shutil.which("python")
    if fallback_python and fallback_python not in interpreters:
        interpreters.append(fallback_python)
    for python in interpreters:
        try:
            result = subprocess.run(
                [
                    python,
                    "-c",
                    "import z3, os; print(os.path.dirname(z3.__file__))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            add(Path(result.stdout.strip()) / "lib")

    # Find user-site installs belonging to Python versions not on PATH.
    home = Path.home()
    if system == "Linux":
        patterns = [home / ".local/lib/python3.*/site-packages/z3/lib"]
    elif system == "Darwin":
        patterns = [home / "Library/Python/3.*/lib/python/site-packages/z3/lib"]
    elif system == "Windows":
        appdata = os.environ.get("APPDATA")
        patterns = (
            [Path(appdata) / "Python/Python*/site-packages/z3/lib"] if appdata else []
        )
    else:
        patterns = []
    for pattern in patterns:
        for match in sorted(glob.glob(str(pattern))):
            add(Path(match))

    # pkg-config discovers distribution and package-manager install prefixes.
    try:
        pkg_config = subprocess.run(
            ["pkg-config", "--variable=libdir", "z3"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pkg_config = None
    if pkg_config and pkg_config.returncode == 0 and pkg_config.stdout.strip():
        add(Path(pkg_config.stdout.strip()))

    # Homebrew keeps a stable opt path on Apple Silicon and Intel Macs.
    if system == "Darwin":
        brew = shutil.which("brew")
        if brew:
            try:
                prefix = subprocess.run(
                    [brew, "--prefix", "z3"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                prefix = None
            if prefix and prefix.returncode == 0 and prefix.stdout.strip():
                add(Path(prefix.stdout.strip()) / "lib")
        add(Path("/opt/homebrew/opt/z3/lib"))
        add(Path("/usr/local/opt/z3/lib"))
        add(Path("/usr/local/lib"))
        add(Path("/opt/homebrew/lib"))
        add(Path("/usr/lib"))
    elif system == "Linux":
        add(Path("/usr/local/lib"))
        add(Path("/usr/lib"))
        add(Path("/usr/lib64"))
        add(Path("/lib"))
        add(Path("/lib64"))
        for pattern in ("/usr/lib/*-linux-gnu", "/lib/*-linux-gnu"):
            for match in sorted(glob.glob(pattern)):
                add(Path(match))
    return dirs


def dir_has_linkable_libz3(d: Path, system: str | None = None) -> bool:
    """Whether `d` contains a link-library filename used by this platform."""
    names = LINK_LIBRARIES.get(system or platform.system(), ())
    return any((d / name).exists() for name in names)


def find_libz3_dir(system: str | None = None) -> Path | None:
    """Return the first candidate containing a linkable Z3 library."""
    system = system or platform.system()
    for directory in candidate_lib_dirs(system):
        if dir_has_linkable_libz3(directory, system):
            return directory
    return None


def build_env(lib_dir: Path | None, system: str | None = None) -> dict[str, str]:
    """Set the linker override and the host's runtime library search path."""
    if lib_dir is None:
        return {}
    system = system or platform.system()
    loader = LOADER_PATHS.get(system)
    if loader is None:
        return {"Z3_LIBRARY_PATH_OVERRIDE": str(lib_dir)}
    variable, separator = loader
    paths = [str(lib_dir)]
    existing = os.environ.get(variable, "")
    if existing:
        paths.append(existing)
    return {
        "Z3_LIBRARY_PATH_OVERRIDE": str(lib_dir),
        variable: separator.join(paths),
    }


USAGE = """\
usage: scripts/z3_test.py [--print-env | --help | --cargo-subcommand <cmd>] <cargo args>...

  --print-env              Print shell export statements and exit.
  --cargo-subcommand CMD   Cargo subcommand (default: "nextest run").
  --help, -h               Show this help.

With no cargo args the default is:
  cargo nextest run -p chelis-prove --features z3

Everything after the flags is forwarded to Cargo. Example:
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
        arg = argv[i]
        if arg in ("--help", "-h"):
            print(USAGE)
            return 0
        if arg == "--print-env":
            print_env = True
            i += 1
            continue
        if arg == "--cargo-subcommand":
            if i + 1 >= len(argv):
                print("error: --cargo-subcommand requires an argument", file=sys.stderr)
                return 2
            cargo_subcommand = argv[i + 1]
            i += 2
            continue
        cargo_args = argv[i:]
        break

    system = platform.system()
    lib_dir = find_libz3_dir(system)
    if lib_dir is None:
        expected = ", ".join(LINK_LIBRARIES.get(system, ()))
        print(
            f"error: no prebuilt Z3 link library found for {system} "
            f"(expected {expected or 'a supported platform library'}). Install "
            "the Z3 development library or set Z3_LIBRARY_PATH_OVERRIDE to its "
            "directory. See docs/local_z3_environment.md.",
            file=sys.stderr,
        )
        return 2

    overrides = build_env(lib_dir, system)
    if print_env:
        for key, value in overrides.items():
            print(f"export {key}={shlex.quote(value)}")
        return 0

    if not cargo_args:
        cargo_args = DEFAULT_ARGS

    if overrides:
        print("scripts/z3_test.py setting Z3 library paths:", file=sys.stderr)
        for key, value in overrides.items():
            print(f"  {key}={value}", file=sys.stderr)
        print("", file=sys.stderr)
    else:
        print(
            "scripts/z3_test.py: using the system Z3 library",
            file=sys.stderr,
        )

    env = os.environ.copy()
    env.update(overrides)
    cargo_argv = ["cargo", *shlex.split(cargo_subcommand), *cargo_args]
    print(f"+ {shlex.join(cargo_argv)}", file=sys.stderr)
    os.execvpe("cargo", cargo_argv, env)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
