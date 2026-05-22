"""Set up the uv-managed Python for GitHub Actions cargo jobs.

Creates `.venv/` (using uv, which must already be installed), then writes
`LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) into
`$GITHUB_ENV` so subsequent `cargo` invocations can both **build**
`chelis-python` (via `PYO3_PYTHON=.venv/bin/python` from
`.cargo/config.toml`) **and** run its test binaries (which dynamically
load `libpython.so` / `libpython.dylib` from the uv interpreter's
`sysconfig.LIBDIR`).

The Python version is read from `py/pyproject.toml`'s `requires-python`
constraint (single source of truth). It can be overridden with
`--python <version>`.

Background:
- Pre-PYO3 fix, pyo3-build-config auto-detected the runner's system
  Python and linked the chelis-python test binaries against
  `/usr/lib/.../libpython3.X.so`. That lib was already in the dynamic
  linker's default search path (ld.so.cache on Linux), so the test
  binaries could dlopen libpython at runtime without any rpath setup.
- After `.cargo/config.toml` points `PYO3_PYTHON` at
  `.venv/bin/python`, the test binaries link against the uv-managed
  `libpython3.X.so.1.0`, which lives at
  `~/.local/share/uv/python/cpython-3.X.Y-...-none/lib/`. That path
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
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "py" / "pyproject.toml"


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


def pinned_python_version(pyproject_path: Path = PYPROJECT_PATH) -> str:
    """Return the Python version pinned by `py/pyproject.toml`'s requires-python.

    Parses `requires-python = ">=3.11"` (or similar) and returns the bare
    version string (e.g. "3.11"). This is the single source of truth for
    the project's Python pin; the README, `.cargo/config.toml` comment,
    and CI invocations all reference the same value.

    Raises RuntimeError if pyproject.toml is missing or doesn't pin a
    version we can parse.
    """
    if not pyproject_path.exists():
        raise RuntimeError(
            f"{pyproject_path} not found; cannot determine pinned Python version"
        )
    text = pyproject_path.read_text(encoding="utf-8")
    # Match `requires-python = ">=3.11"` (quote style and spacing flexible).
    match = re.search(
        r'requires-python\s*=\s*["\'][>=~^]*\s*(\d+\.\d+(?:\.\d+)?)',
        text,
    )
    if not match:
        raise RuntimeError(
            f"could not parse `requires-python` from {pyproject_path}"
        )
    return match.group(1)


def create_venv(python_version: str) -> Path:
    """Create `.venv/` via uv. Returns the path to `.venv/bin/python`.

    Raises with a clear actionable diagnostic on common failure modes:
    - uv binary missing from PATH (FileNotFoundError → SystemExit with hint)
    - uv venv command failed (CalledProcessError → SystemExit with hint)
    """
    try:
        subprocess.run(
            ["uv", "venv", "--python", python_version],
            cwd=REPO_ROOT,
            check=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "`uv` not found on PATH. The `Install uv` step "
            "(astral-sh/setup-uv) probably failed or was reordered after "
            "this one. See README.md for local-dev setup."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"`uv venv --python {python_version}` exited with code "
            f"{exc.returncode}. See uv output above for the underlying error."
        ) from exc
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


def python_abi_version(venv_python: Path) -> str:
    """Query the venv interpreter's ABI version string (e.g. '3.11')."""
    result = subprocess.run(
        [
            str(venv_python),
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    abi = result.stdout.strip()
    if not abi:
        raise RuntimeError(f"`{venv_python}` returned empty version_info")
    return abi


def ensure_link_symlink(libdir: Path, abi: str) -> None:
    """Ensure `libpython<abi>.{so,dylib}` exists in `libdir` for `-lpython<abi>`.

    uv-managed Python on Linux ships `libpython3.X.so.1.0` (with SONAME
    suffix) but not always the unversioned `libpython3.X.so` symlink.
    GNU ld and rust-lld resolve `-lpython3.X` to `libpython3.X.so` (or
    `.a`), not to the SONAME-suffixed file, so the link step fails with
    `unable to find library -lpython3.X`. This helper creates the
    missing symlink. macOS uses `.dylib` and uv already ships
    `libpython3.X.dylib` unversioned, so the symlink check is a no-op
    there but kept symmetric in case a future uv build changes that.
    """
    if platform.system() == "Linux":
        link_name = libdir / f"libpython{abi}.so"
        target = libdir / f"libpython{abi}.so.1.0"
    elif platform.system() == "Darwin":
        link_name = libdir / f"libpython{abi}.dylib"
        # uv on macOS ships the unversioned dylib already; nothing to do.
        if link_name.exists():
            return
        target = libdir / f"libpython{abi}.dylib"
    else:
        return

    if link_name.exists() or link_name.is_symlink():
        return  # Already there (or a real file).
    if not target.exists():
        # If neither the unversioned link nor the SONAME-suffixed file
        # exists, something else is wrong with the uv install — surface
        # it loudly rather than silently leaving a broken link.
        raise RuntimeError(
            f"expected `{target}` to exist in uv-managed Python libdir; "
            f"`{link_name}` cannot be auto-symlinked. Check the uv install."
        )
    link_name.symlink_to(target.name)
    print(f"linked {link_name} -> {target.name}")


def append_to_github_env(var: str, value: str) -> None:
    """Append `var=<value>:<existing>` to `$GITHUB_ENV`, deduplicating.

    Preserves any existing value of `var` already exported to the runner.
    Deduplicates colon-separated path entries (preserving first-occurrence
    order) so repeated invocations within the same job don't grow the
    value linearly.
    """
    github_env_path = os.environ.get("GITHUB_ENV")
    if not github_env_path:
        print(
            "GITHUB_ENV not set; skipping env export (assuming local dry-run).",
            file=sys.stderr,
        )
        return
    existing = os.environ.get(var, "")
    combined = f"{value}:{existing}" if existing else value
    deduped = ":".join(
        dict.fromkeys(part for part in combined.split(":") if part)
    )
    with open(github_env_path, "a", encoding="utf-8") as fh:
        fh.write(f"{var}={deduped}\n")
    print(f"{var}={deduped}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=None,
        help=(
            "Python version pin passed to `uv venv --python`. "
            "Default: read from py/pyproject.toml's requires-python."
        ),
    )
    args = parser.parse_args()
    python_version = args.python or pinned_python_version()

    var = runner_libpath_var()
    venv_python = create_venv(python_version)
    libdir = libdir_for(venv_python)
    abi = python_abi_version(venv_python)
    # Ensure the linker can resolve `-lpython<abi>` — uv on Linux ships
    # the SONAME-suffixed `.so.1.0` but not the unversioned `.so`.
    ensure_link_symlink(Path(libdir), abi)
    append_to_github_env(var, libdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
