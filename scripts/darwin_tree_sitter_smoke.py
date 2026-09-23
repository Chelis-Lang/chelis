#!/usr/bin/env python3
"""Run the clean Darwin tree-sitter compiler and parser acceptance legs."""

from __future__ import annotations

import os
import platform
import subprocess
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MISMATCH_MARKERS = (
    "supplying the --target arm64-apple-macosx != arm64-apple-darwin",
    "cc-wrapper is currently not designed with multi-target compilers",
)


def target_mismatch_diagnostics(output: str) -> list[str]:
    return [
        line
        for line in output.splitlines()
        if all(marker in line for marker in MISMATCH_MARKERS)
    ]


def run_and_echo(command: list[str], environment: dict[str, str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Darwin tree-sitter acceptance command failed: {' '.join(command)}"
        )
    return completed.stdout


def main() -> int:
    if platform.system() != "Darwin":
        print("Darwin tree-sitter smoke: SKIP (non-Darwin host)")
        return 0
    nix_cc = Path(os.environ.get("NIX_CC", ""))
    for variable, command in (
        ("CC_aarch64_apple_darwin", "cc"),
        ("CXX_aarch64_apple_darwin", "c++"),
    ):
        value = os.environ.get(variable, "")
        expected = nix_cc / "bin" / command
        if not nix_cc.is_dir() or Path(value) != expected or not expected.is_file():
            raise RuntimeError(f"{variable} is not the managed Nix compiler: {value}")
    if os.environ.get("CRATE_CC_NO_DEFAULTS") != "1":
        raise RuntimeError("CRATE_CC_NO_DEFAULTS must suppress the redundant native target alias")

    state = Path(os.environ["DEVENV_STATE"])
    acceptance_root = state / "acceptance-targets"
    acceptance_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="tree-sitter-",
        dir=acceptance_root,
    ) as raw_target:
        environment = dict(os.environ)
        environment["CARGO_TARGET_DIR"] = raw_target
        build_output = run_and_echo(
            ["cargo", "check", "-p", "tree-sitter-chelis"],
            environment,
        )
        diagnostics = target_mismatch_diagnostics(build_output)
        if diagnostics:
            raise RuntimeError(
                "Darwin cc-wrapper target mismatch returned:\n" + "\n".join(diagnostics)
            )
        run_and_echo(
            [
                "cargo",
                "nextest",
                "run",
                "-p",
                "tree-sitter-chelis",
                "--no-fail-fast",
            ],
            environment,
        )

    print("Darwin tree-sitter smoke: PASS")
    return 0


if __name__ == "__main__":
    from observed_cargo import observed_cargo
    with observed_cargo():
        raise SystemExit(main())
