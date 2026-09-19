#!/usr/bin/env python3
"""Acceptance oracle for callable-origin `grad` selector identity."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMMANDS = (
    (
        "surf selector identity",
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-surf",
        "--test",
        "issue_1955_grad_selector_identity",
        "--no-fail-fast",
    ),
    (
        "compiler preparation propagation",
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "issue_1955_grad_selector_pipeline",
        "--no-fail-fast",
    ),
    (
        "CLI selector diagnostics",
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "issue_1955_grad_selector_identity",
        "--no-fail-fast",
    ),
)


def main() -> int:
    expected_python = (ROOT / ".venv" / "bin" / "python").resolve()
    if pathlib.Path(sys.executable).resolve() != expected_python:
        print(
            "run this oracle with the worktree-managed interpreter: "
            ".venv/bin/python scripts/grad_selector_identity_oracle.py",
            file=sys.stderr,
        )
        return 2

    environment = os.environ.copy()
    environment.setdefault("CARGO_TARGET_DIR", str(ROOT / "target" / "agents" / "1955"))
    environment.setdefault("PYO3_PYTHON", sys.executable)

    for label, *command in COMMANDS:
        print(f"== {label}: {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if completed.returncode != 0:
            print(f"GRAD SELECTOR IDENTITY ORACLE: FAIL ({label})", file=sys.stderr)
            return completed.returncode

    source = (ROOT / "crates" / "chelis-surf" / "src" / "desugar.rs").read_text()
    forbidden = "(0..wrt.len())"
    if forbidden in source:
        print(
            f"GRAD SELECTOR IDENTITY ORACLE: FAIL (found fallback {forbidden!r})",
            file=sys.stderr,
        )
        return 1

    print("GRAD SELECTOR IDENTITY ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
