#!/usr/bin/env python3
"""Declare the activated project runtime and completed native flake outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


def nix_output(*arguments: str) -> str:
    return subprocess.check_output(["nix", *arguments], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", action="store_true", help="Run only after the complete native flake check succeeds")
    arguments = parser.parse_args()
    roots = [os.path.realpath(os.environ["DEVENV_PROFILE"])]
    # Environment-only inputs are not dependencies of Devenv's package profile.
    # Keep the SMT solver's archive/header tree reachable from the shared cache.
    if cvc5_dir := os.environ.get("CVC5_DIR"):
        roots.append(os.path.realpath(cvc5_dir))
    if arguments.checks:
        system = nix_output("eval", "--impure", "--raw", "--expr", "builtins.currentSystem")
        checks = json.loads(nix_output(
            "eval", "--json", f".#checks.{system}", "--apply",
            "checks: builtins.mapAttrs (_: check: check.outPath) checks",
        ))
        roots.extend(checks.values())
        cvc5 = nix_output("eval", "--raw", f".#legacyPackages.{system}.cvc5-dir.outPath")
        # Substituted checks need not realize their build-only inputs. Publish
        # cvc5 when present; never build an unused toolchain merely to cache it.
        if Path(cvc5).exists():
            roots.append(cvc5)
    with Path(os.environ["BUILD_CACHE_ROOTS_PATH"]).open("a", encoding="utf-8") as manifest:
        manifest.write("".join(path + "\n" for path in dict.fromkeys(roots)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
