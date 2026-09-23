"""Task-owned Cargo launchers for repository Python build entrypoints.

This module only arranges process environments. The build driver owns all
compiler observation, bootstrap, and runtime-identity policy.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import tempfile


def cargo_environment(environ, *, directory: Path, python: Path):
    """Install the shared driver using an explicit Cargo and interpreter.

    The caller selects/validates its Python and target before entering here.
    Preserve a rustup proxy's filename: resolving its symlink to ``rustup``
    changes argv[0] dispatch and no longer invokes Cargo.
    """
    environment = dict(environ)
    configured = environment.get(
        "CHELIS_IDENTITY_REAL_CARGO", environment.get("CARGO", "cargo")
    )
    cargo = shutil.which(configured, path=environment.get("PATH", os.defpath))
    if cargo is None:
        raise ValueError(
            "runtime identity requires an executable Cargo; install the pinned "
            "toolchain or set CHELIS_IDENTITY_REAL_CARGO to its Cargo executable"
        )
    real_cargo = Path(os.path.abspath(cargo))
    from runtime_identity_build import install_cargo_launcher

    launcher = install_cargo_launcher(
        directory, real_cargo=real_cargo, python=python
    )
    environment["CHELIS_IDENTITY_REAL_CARGO"] = str(real_cargo)
    environment.setdefault("CHELIS_IDENTITY_PROVENANCE", "source-worktree")
    environment["PATH"] = os.pathsep.join(
        (str(launcher.parent), environment.get("PATH", os.defpath))
    )
    environment["CARGO"] = str(launcher)
    return environment


@contextmanager
def observed_cargo_environment(environ, *, python: Path, required=False):
    """Keep one launcher alive for a complete child-process tree.

    A command that does not use Cargo remains runnable without a Rust toolchain.
    If it later attempts a build, normal executable discovery fails; an explicit
    but unusable Cargo override always fails here instead of falling back.
    """
    if (not required and "CHELIS_IDENTITY_REAL_CARGO" not in environ
            and "CARGO" not in environ
            and shutil.which("cargo", path=environ.get("PATH", os.defpath)) is None):
        yield dict(environ)
        return
    with tempfile.TemporaryDirectory(prefix="chelis-observed-cargo-") as directory:
        yield cargo_environment(environ, directory=Path(directory), python=python)


@contextmanager
def observed_cargo():
    """Execution-only boundary for direct repository build/oracle scripts."""
    original = dict(os.environ)
    configured_python = original.get("PYO3_PYTHON")
    python = Path(configured_python or sys.executable)
    if not python.is_absolute():
        python = Path(__file__).resolve().parent.parent / python
    if configured_python is not None and (
        not configured_python or not python.is_file() or not os.access(python, os.X_OK)
    ):
        raise ValueError(
            f"PYO3_PYTHON must name an executable Python 3.11+ interpreter: {python}"
        )
    with observed_cargo_environment(original, python=python) as environment:
        changed = {name: value for name, value in environment.items()
                   if original.get(name) != value}
        os.environ.update(changed)
        try:
            yield
        finally:
            for name in changed:
                if name in original:
                    os.environ[name] = original[name]
                else:
                    os.environ.pop(name, None)
