"""Install the shared Cargo observer for subsequent GitHub Actions steps.

Run with uv's managed Python 3.11 after selecting the job's Rust toolchain.
The launcher remains on PATH for Cargo invoked by scripts and test binaries;
this adapter does not build anything or alter Cargo's target/cache settings.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile

from runtime_identity_build import install_cargo_launcher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provenance",
        required=True,
        choices=("source-worktree", "sealed-distribution"),
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11):
        parser.error("run with uv run --managed-python --python 3.11 --no-project python")

    github_path = Path(os.environ["GITHUB_PATH"])
    github_env = Path(os.environ["GITHUB_ENV"])
    # Retain an earlier explicit selection if setup is repeated. Do not resolve
    # Cargo's symlink: rustup uses the invoked executable name to select Cargo.
    selected = os.environ.get("CHELIS_IDENTITY_REAL_CARGO") or shutil.which("cargo")
    if not selected:
        parser.error("select the real Rust toolchain before installing the observer")
    real_cargo = Path(selected).absolute()
    python = Path(sys.executable).absolute()
    directory = Path(tempfile.mkdtemp(prefix="chelis-cargo-", dir=os.environ["RUNNER_TEMP"]))
    launcher = install_cargo_launcher(directory, real_cargo=real_cargo, python=python)
    exports = {
        "CHELIS_IDENTITY_REAL_CARGO": str(real_cargo),
        "CHELIS_IDENTITY_PROVENANCE": args.provenance,
    }
    with github_env.open("a", encoding="utf-8") as stream:
        for name, value in exports.items():
            stream.write(f"{name}={value}\n")
    with github_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{launcher.parent}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
