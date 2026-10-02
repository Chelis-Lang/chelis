"""Set up the uv-managed Python for GitHub Actions cargo jobs.

Creates `.venv/` (using uv, which must already be installed), then writes
`LD_LIBRARY_PATH` (Linux) or `DYLD_LIBRARY_PATH` (macOS) into
`$GITHUB_ENV` so subsequent `cargo` invocations can both **build**
`chelis-python` (via `PYO3_PYTHON=.venv/bin/python` from
`.cargo/config.toml`) **and** run its test binaries (which dynamically
load `libpython.so` / `libpython.dylib` from the uv interpreter's
`sysconfig.LIBDIR`).

It also writes `PYO3_ENVIRONMENT_SIGNATURE`, the interpreter's full version
and lib dir, so a restored cargo target whose PyO3 build configuration came
from a different interpreter is reconfigured instead of reused. PyO3 reruns
its build script when that variable changes (PyO3/pyo3#2724) but not when
the interpreter behind an unchanged `PYO3_PYTHON` path changes. Without it,
a uv release that moves the patch version (3.11.16 to 3.11.17, chelis#2895)
leaves the cached `-L` pointing at the previous install, which a fresh
runner does not have, and linking `chelis-python` fails.

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
PYO3_SIGNATURE_VAR = "PYO3_ENVIRONMENT_SIGNATURE"


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


def interpreter_signature(venv_python: Path, libdir: str) -> str:
    """Return a one-line identity of the interpreter behind `venv_python`.

    Combines `sys.version` (patch release, build date, compiler) with the
    `sysconfig.LIBDIR` that PyO3 links against, so any interpreter change
    that could alter PyO3's build configuration changes the signature.
    Whitespace runs collapse to one space because `$GITHUB_ENV` takes one
    line per variable.
    """
    result = subprocess.run(
        [str(venv_python), "-c", "import sys; print(sys.version)"],
        capture_output=True,
        text=True,
        check=True,
    )
    version = " ".join(result.stdout.split())
    if not version:
        raise RuntimeError(f"`{venv_python}` returned empty sys.version")
    return f"{version} {libdir}"


def _list_libdir(libdir: Path) -> str:
    """Render a one-line snapshot of `libdir` for error diagnostics."""
    try:
        entries = sorted(p.name for p in libdir.iterdir())
        return ", ".join(entries) if entries else "<empty>"
    except OSError as exc:
        return f"<unreadable: {exc}>"


def discover_python_libdirs(primary_libdir: str, abi: str) -> list[Path]:
    """Return every plausible lib/ dir that contains a libpython<abi> file.

    The primary libdir from `sysconfig.LIBDIR` is the authoritative
    answer for the venv's interpreter, but pyo3-build-config can end up
    invoking a different copy of uv's Python install whose sys-paths
    differ. Searches the documented uv install roots (default
    `~/.local/share/uv/python/`, plus the `UV_PYTHON_INSTALL_DIR`
    override) for any `cpython-<abi>*` install and includes its
    `lib/` subdir if libpython files are present there.

    Returns a de-duplicated list with `primary_libdir` first so the
    caller links it before any siblings.
    """
    results: list[Path] = []
    seen: set[Path] = set()

    def maybe_add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        if not resolved.is_dir():
            return
        # Only include lib dirs that actually contain a libpython<abi>
        # file (versioned or not). Skip unrelated dirs.
        suffix = ".so" if platform.system() == "Linux" else ".dylib"
        if not any(resolved.glob(f"libpython{abi}{suffix}*")):
            return
        seen.add(resolved)
        results.append(p)

    maybe_add(Path(primary_libdir))

    candidate_roots: list[Path] = []
    if "UV_PYTHON_INSTALL_DIR" in os.environ:
        candidate_roots.append(Path(os.environ["UV_PYTHON_INSTALL_DIR"]))
    candidate_roots.append(Path.home() / ".local" / "share" / "uv" / "python")
    for root in candidate_roots:
        if not root.is_dir():
            continue
        for install in sorted(root.glob(f"cpython-{abi}*")):
            maybe_add(install / "lib")

    return results


def ensure_link_symlink(libdir: Path, abi: str) -> None:
    """Ensure `libpython<abi>.{so,dylib}` exists in `libdir`.

    The linker resolves `-lpython<abi>` to `libpython<abi>.so` on Linux
    or `libpython<abi>.dylib` on macOS. Versioned `.so.<X>` files don't
    match. uv-managed Python on Linux ships the versioned file but not
    always the unversioned link; on macOS uv ships the unversioned
    dylib already.

    Strategy:
    1. If the unversioned name already resolves to a real file, no-op.
    2. Otherwise glob for `libpython<abi>{suffix}*` candidates in
       `libdir`, pick the one with the shortest name (prefers
       `.so.1.0` over `.so.1.0.X.Y`), and symlink the unversioned name
       to it.
    3. If no candidate is found, raise with a directory listing so
       failure diagnostics are actionable in CI logs.
    """
    system = platform.system()
    if system == "Linux":
        suffix = ".so"
    elif system == "Darwin":
        suffix = ".dylib"
    else:
        return

    link_name = libdir / f"libpython{abi}{suffix}"
    # If a real file (or working symlink) already resolves, we're done.
    # `.is_symlink() and .exists()` catches working symlinks; `.is_file()`
    # catches actual files. Broken symlinks fall through to be replaced.
    if link_name.is_file() or (link_name.is_symlink() and link_name.exists()):
        print(f"libpython link already present: {link_name}")
        return

    # Glob for versioned siblings.
    pattern = f"libpython{abi}{suffix}*"
    candidates = sorted(
        (p for p in libdir.glob(pattern) if p != link_name),
        key=lambda p: len(p.name),
    )
    if not candidates:
        raise RuntimeError(
            f"no libpython{abi} files found in {libdir} matching {pattern!r}. "
            f"Directory contents: {_list_libdir(libdir)}. "
            f"Check the uv install or pin a different Python build."
        )
    target = candidates[0]
    # Remove any stale broken symlink at link_name before re-creating.
    if link_name.is_symlink():
        link_name.unlink()
    link_name.symlink_to(target.name)
    print(f"linked {link_name} -> {target.name}")


def ensure_default_uv_root_mirror(libdir: str, abi: str) -> None:
    """Create a libpython symlink at the default uv install root.

    Workaround for the case where pyo3-build-config emits a `-L` flag
    pointing at `~/.local/share/uv/python/cpython-<install>/lib/` even
    when uv staged the actual install elsewhere (via
    `$UV_PYTHON_INSTALL_DIR` or via a stale path cached from a prior
    build). The path can also persist in cached pyo3-build-config
    output that Swatinem/rust-cache restores between runs.

    Strategy:
    1. Find the basename of the actual install directory by walking up
       from `libdir` (`libdir/../` should be `cpython-<abi>-...-...`).
    2. Mirror that basename under `~/.local/share/uv/python/`.
    3. Symlink `libpython<abi>.{so,dylib}` there to the actual lib
       file by absolute path.

    Idempotent. No-op if the basename doesn't look like a uv install
    (e.g., libdir is a system Python). No-op if the default root
    already has a working entry.
    """
    suffix = ".so" if platform.system() == "Linux" else (
        ".dylib" if platform.system() == "Darwin" else None
    )
    if suffix is None:
        return

    libdir_path = Path(libdir)
    actual_install = libdir_path.parent  # e.g. .../uv-python-dir/cpython-3.11.15-...
    if not actual_install.name.startswith(f"cpython-{abi}"):
        # Not a uv-style install layout; nothing to mirror.
        return

    # Find the actual libpython file inside libdir_path.
    pattern = f"libpython{abi}{suffix}*"
    candidates = sorted(
        (p for p in libdir_path.glob(pattern) if p.is_file()),
        key=lambda p: len(p.name),
    )
    if not candidates:
        return
    real_lib = candidates[0].resolve()

    default_root = Path.home() / ".local" / "share" / "uv" / "python"
    mirror_libdir = default_root / actual_install.name / "lib"
    mirror_libdir.mkdir(parents=True, exist_ok=True)
    mirror_link = mirror_libdir / f"libpython{abi}{suffix}"
    if mirror_link.is_file() or (mirror_link.is_symlink() and mirror_link.exists()):
        print(f"default-root mirror already present: {mirror_link}")
        return
    if mirror_link.is_symlink():
        mirror_link.unlink()  # stale broken link
    mirror_link.symlink_to(real_lib)  # absolute target
    print(f"mirrored {mirror_link} -> {real_lib}")


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


def set_github_env(var: str, value: str) -> None:
    """Write `var=<value>` to `$GITHUB_ENV`, replacing any inherited value.

    Unlike `append_to_github_env`, the value is opaque rather than a path
    list: it is never split on `:` or merged with the existing value.
    """
    if "\n" in value or "\r" in value:
        raise ValueError(f"{var} value must be one line: {value!r}")
    github_env_path = os.environ.get("GITHUB_ENV")
    if not github_env_path:
        print(
            "GITHUB_ENV not set; skipping env export (assuming local dry-run).",
            file=sys.stderr,
        )
        return
    with open(github_env_path, "a", encoding="utf-8") as fh:
        fh.write(f"{var}={value}\n")
    print(f"{var}={value}")


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
    # Ensure the linker can resolve `-lpython<abi>`. The venv's
    # interpreter reports `sysconfig.LIBDIR` for its actual install
    # location, but uv may stage Python installs at multiple roots
    # (`$UV_PYTHON_INSTALL_DIR` and `~/.local/share/uv/python/`) and
    # pyo3-build-config can end up querying either. Symlink the
    # unversioned name at every install root we can find so the
    # linker resolves regardless of which path pyo3 picked.
    install_libdirs = discover_python_libdirs(libdir, abi)
    # If pyo3 keeps emitting `-L ~/.local/share/uv/python/<install>/lib`
    # even when uv installed Python elsewhere (cached path from a prior
    # build, or pyo3's own path canonicalization through sys.base_prefix),
    # create a mirror dir at the default uv root with a symlink to the
    # real libpython. Cheap and safe — if no install_libdirs are found,
    # this no-ops; if pyo3 doesn't look there, it's an unused symlink.
    ensure_default_uv_root_mirror(libdir, abi)
    linked_any = False
    for d in install_libdirs:
        try:
            ensure_link_symlink(d, abi)
            linked_any = True
        except RuntimeError as exc:
            # A specific install root might be a stale cache without a
            # libpython; non-fatal unless NONE of the roots had one.
            print(f"warning: skipping {d}: {exc}", file=sys.stderr)
    if not linked_any:
        raise RuntimeError(
            "could not link libpython at any uv install root; "
            "checked: " + ", ".join(str(p) for p in install_libdirs)
        )
    append_to_github_env(var, libdir)
    set_github_env(PYO3_SIGNATURE_VAR, interpreter_signature(venv_python, libdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
