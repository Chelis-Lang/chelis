"""Create CI's owned uv venv and bind native builds to that interpreter.

Python's version pin comes from py/pyproject.toml. Only the selected
interpreter's LIBDIR can supply libpython; another install is not a fallback.
The runtime loader path and PyO3's environment signature are exported before
Cargo cache restoration. PyO3 0.24.2 tracks PYO3_ENVIRONMENT_SIGNATURE even
when the lexical .venv/bin/python path stays fixed across interpreter updates.
Local gate children use the same signature query without creating a venv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


# Query in the selected venv, with ambient Python imports disabled. Resolving
# executable paths contributes identity only; PYO3_PYTHON stays lexical.
INTERPRETER_BUILD_PROBE = """
import json, os, struct, sys, sysconfig
print(json.dumps({
    "executable": sys.executable,
    "resolved_executable": os.path.realpath(sys.executable),
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "implementation": sys.implementation.name, "version": sys.version,
    "version_info": list(sys.version_info[:3]),
    "pointer_width": struct.calcsize("P") * 8,
    "libdir": sysconfig.get_config_var("LIBDIR"),
    "ldlibrary": sysconfig.get_config_var("LDLIBRARY"),
    "soabi": sysconfig.get_config_var("SOABI"),
    "build_flags": {name: sysconfig.get_config_var(name) for name in
                    ("Py_DEBUG", "Py_REF_DEBUG", "Py_TRACE_REFS", "COUNT_ALLOCS", "Py_ENABLE_SHARED", "Py_GIL_DISABLED")},
}))
"""


def reject_discovery_overrides(environ: dict[str, str]) -> None:
    """Native project builds must query the selected interpreter."""
    conflicts = sorted(name for name in environ if name in {
        "PYO3_CONFIG_FILE", "PYO3_NO_PYTHON",
    } or name.startswith("PYO3_CROSS"))
    if conflicts:
        raise RuntimeError(
            "Python discovery overrides conflict with the selected interpreter: "
            + ", ".join(conflicts)
        )


def pyo3_environment_signature(
    interpreter: Path, *, environ: dict[str, str] | None = None,
) -> str:
    """Bind Cargo's PyO3 configuration to the actual interpreter build.

    PyO3's supported PYO3_ENVIRONMENT_SIGNATURE invalidation closes stale
    build metadata behind an unchanged venv path. Do not hash ambient state,
    timestamps or cache contents: identical interpreter inputs stay identical.
    """
    environment = dict(os.environ if environ is None else environ)
    reject_discovery_overrides(environment)
    try:
        result = subprocess.run(
            [str(interpreter), "-I", "-c", INTERPRETER_BUILD_PROBE],
            capture_output=True, text=True, check=True, timeout=30,
            env=environment,
        )
        identity = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError("could not query interpreter build identity") from exc
    expected = {
        "executable", "resolved_executable", "prefix", "base_prefix",
        "implementation", "version", "version_info", "pointer_width",
        "libdir", "ldlibrary", "soabi", "build_flags",
    }
    valid = isinstance(identity, dict) and set(identity) == expected
    if valid:
        strings = expected - {"version_info", "pointer_width", "build_flags"}
        valid = all(isinstance(identity[name], str) and identity[name] for name in strings)
        valid = valid and all(Path(identity[name]).is_absolute() for name in (
            "executable", "resolved_executable", "prefix", "base_prefix", "libdir",
        ))
        version = identity["version_info"]
        valid = valid and isinstance(version, list) and len(version) == 3
        valid = valid and all(type(part) is int and part >= 0 for part in version)
        valid = valid and tuple(version) >= (3, 11, 0)
        valid = valid and type(identity["pointer_width"]) is int and identity["pointer_width"] in (32, 64)
        flags = identity["build_flags"]
        valid = valid and isinstance(flags, dict) and set(flags) == {
            "Py_DEBUG", "Py_REF_DEBUG", "Py_TRACE_REFS", "COUNT_ALLOCS", "Py_ENABLE_SHARED", "Py_GIL_DISABLED",
        }
        valid = valid and all(value is None or type(value) is int for value in flags.values())
    if not valid:
        raise RuntimeError("invalid interpreter build identity")
    identity["selected_executable"] = str(interpreter.absolute())
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "chelis-pyo3-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _list_libdir(libdir: Path) -> str:
    """Render a one-line snapshot of `libdir` for error diagnostics."""
    try:
        entries = sorted(p.name for p in libdir.iterdir())
        return ", ".join(entries) if entries else "<empty>"
    except OSError as exc:
        return f"<unreadable: {exc}>"


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
        (p for p in libdir.glob(pattern) if p != link_name and p.is_file()),
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

    reject_discovery_overrides(dict(os.environ))
    var = runner_libpath_var()
    venv_python = create_venv(python_version)
    libdir = libdir_for(venv_python)
    abi = python_abi_version(venv_python)
    # The current interpreter supplies the library and the cache identity.
    # A different install cannot repair configuration that Cargo must rebuild.
    ensure_link_symlink(Path(libdir), abi)
    signature = pyo3_environment_signature(venv_python)
    append_to_github_env(var, libdir)
    if github_env := os.environ.get("GITHUB_ENV"):
        with Path(github_env).open("a", encoding="utf-8") as stream:
            stream.write(f"PYO3_PYTHON={venv_python}\n")
            stream.write(f"PYO3_ENVIRONMENT_SIGNATURE={signature}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
