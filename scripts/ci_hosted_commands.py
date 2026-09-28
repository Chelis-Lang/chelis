#!/usr/bin/env python3
"""Publish the hosted twins of the Devenv project commands.

Job steps run under `chelis-ci-shell run` and call `chelis-gate` whichever
runner executes them. The self-hosted runner gets both from its activated
Devenv profile; GitHub-hosted runners keep main's toolchain, so this helper
installs two plain shims after `scripts/ci_setup_uv_python.py` created the
uv-managed `.venv`, puts that venv and the shims first on the runner path,
and names the venv interpreter as `PYO3_PYTHON` for repository commands.
"""
from __future__ import annotations

import os
from pathlib import Path
import stat
import sys


ROOT = Path(__file__).resolve().parents[1]
SHELL = """#!/usr/bin/env bash
# Hosted twin of the Devenv `chelis-ci-shell`: run an Actions command file
# with the checkout's uv-managed interpreter first on PATH.
set -euo pipefail
if [ "${1:-}" != run ] || [ "$#" -ne 2 ]; then
  echo "chelis-ci-shell: usage: chelis-ci-shell run <command-file>" >&2
  exit 64
fi
exec bash --noprofile --norc -e -o pipefail "$2"
"""
GATE = """#!/usr/bin/env bash
# Hosted twin of the Devenv `chelis-gate`: the gate with the venv interpreter.
set -euo pipefail
exec {interpreter} {gate} "$@"
"""


def publish(root: Path, commands: Path, github_env: Path, github_path: Path) -> None:
    interpreter = root / ".venv" / "bin" / "python"
    if not interpreter.is_file():
        raise RuntimeError(f"uv-managed interpreter is missing: {interpreter}")
    commands.mkdir(parents=True, exist_ok=True)
    gate = GATE.format(interpreter=interpreter, gate=root / "scripts" / "gate.py")
    for name, body in (("chelis-ci-shell", SHELL), ("chelis-gate", gate)):
        shim = commands / name
        shim.write_text(body, encoding="utf-8")
        shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    with github_env.open("a", encoding="utf-8") as stream:
        stream.write(f"PYO3_PYTHON={interpreter}\n")
    # The runner prepends entries in reverse order; the venv ends up first.
    with github_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{commands}\n{interpreter.parent}\n")


def main() -> int:
    publish(
        ROOT,
        Path(os.environ["RUNNER_TEMP"]) / "chelis-hosted-commands",
        Path(os.environ["GITHUB_ENV"]),
        Path(os.environ["GITHUB_PATH"]),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
