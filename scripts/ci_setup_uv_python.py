"""Set up the uv-managed Python 3.11 for GitHub Actions cargo jobs.

Creates `.venv/` (using uv, which must already be installed), then writes
`LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) into
`$GITHUB_ENV` so subsequent `cargo` invocations can both **build**
`chelis-python` (via `PYO3_PYTHON=.venv/bin/python` from
`.cargo/config.toml`) **and** run its test binaries (which dynamically
load `libpython3.11.so` / `libpython3.11.dylib` from the uv interpreter's
`sysconfig.LIBDIR`).

Background:
- Pre-PYO3 fix, pyo3-build-config auto-detected the runner's system
  Python and linked the chelis-python test binaries against
  `/usr/lib/.../libpython3.X.so`. That lib was already in the dynamic
  linker's default search path (ld.so.cache on Linux), so the test
  binaries could dlopen libpython at runtime without any rpath setup.
- After `.cargo/config.toml` points `PYO3_PYTHON` at
  `.venv/bin/python`, the test binaries link against the uv-managed
  `libpython3.11.so.1.0`, which lives at
  `~/.local/share/uv/python/cpython-3.11.X-...-none/lib/`. That path
  is not in any default linker search list, so the binaries fail at
  load time with `cannot open shared object file: No such file or
  directory`. This helper computes that lib dir from the venv's
  sysconfig and writes it into `$GITHUB_ENV` so it's exported for the
  rest of the job. PR #153's notes flagged this exact runtime-rpath
  gap as "separate from the link-step fix"; this is the fix.

Local-dev developers do not need this script. On a local workstation,
`cargo test -p chelis-python` would hit the same runtime-rpath issue,
but that test was already excluded from the local gate (per PR #153's
commit message) and the relevant manual phase 3b oracle is invoked
separately from a shell where developers can `export
LD_LIBRARY_PATH` themselves. Promoting that to a workspace-wide setup
is out of scope here.

Usage (CI only):
    python3 scripts/ci_setup_uv_python.py

Exits non-zero on:
- uv not installed,
- venv creation failure,
- failure to query `sysconfig.LIBDIR`,
- unsupported runner OS (only Linux and Darwin are wired).
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def runner_libpath_var() -> str:
    """Return the platform's runtime shared-library search-path env var."""
    system = platform.system()
    if system == "Linux":
        return "LD_LIBRARY_PATH"
    if system == "Darwin":
        return "DYLD_LIBRARY_PATH"
    raise RuntimeError(
        f"Unsupported platform {system!r}: only Linux and Darwin are wired here"
    )


def create_venv(python_version: str) -> Path:
    """Create `.venv/` via uv. Returns the path to `.venv/bin/python`."""
    subprocess.run(
        ["uv", "venv", "--python", python_version],
        cwd=REPO_ROOT,
        check=True,
    )
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"`uv venv --python {python_version}` did not create {venv_python}"
        )
    return venv_python


def libdir_for(venv_python: Path) -> str:
    """Query the venv interpreter's `sysconfig.LIBDIR`."""
    result = subprocess.run(
        [str(venv_python), "-c", "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"],
        capture_output=True,
        text=True,
        check=True,
    )
    libdir = result.stdout.strip()
    if not libdir:
        raise RuntimeError(
            f"`{venv_python} -c 'sysconfig.get_config_var(LIBDIR)'` returned empty"
        )
    return libdir


def append_to_github_env(var: str, value: str) -> None:
    """Append `var=<value>:<existing>` to `$GITHUB_ENV`.

    Preserves any existing value of `var` already exported to the runner.
    """
    github_env_path = os.environ.get("GITHUB_ENV")
    if not github_env_path:
        print(
            "GITHUB_ENV not set; skipping env export (assuming local dry-run).",
            file=sys.stderr,
        )
        return
    existing = os.environ.get(var, "")
    new_value = f"{value}:{existing}" if existing else value
    with open(github_env_path, "a", encoding="utf-8") as fh:
        fh.write(f"{var}={new_value}\n")
    print(f"{var}={new_value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default="3.11",
        help="Python version pin passed to `uv venv --python` (default: 3.11)",
    )
    args = parser.parse_args()

    var = runner_libpath_var()
    venv_python = create_venv(args.python)
    libdir = libdir_for(venv_python)
    append_to_github_env(var, libdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
