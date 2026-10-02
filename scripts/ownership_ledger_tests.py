#!/usr/bin/env python3
"""Run one package's ownership-ledger integration targets.

A ledger target is an integration test whose Cargo ``required-features`` names
``ownership-ledger``. Cargo metadata is the only list of them: adding a ledger
target to a package below edits its ``Cargo.toml`` and nothing else. The gate
and the macOS nightly job both run this script, so neither repeats the list.

Usage::

    python3 scripts/ownership_ledger_tests.py chelis-compiler-api
    python3 scripts/ownership_ledger_tests.py chelis-cli --print

The script prints the derived ``cargo nextest`` command and replaces itself
with it. It uses only the standard library and runs on the system Python a
hosted runner ships.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
LEDGER_FEATURE = "ownership-ledger"
# Every package that has ledger targets, in run order. The chelis-cli command
# rebuilds <target>/debug/chelis against the instrumented runtime, so it runs
# after every command that trusts the binary built before it. A ledger target
# in any other package fails until it is placed here and in the gate.
LEDGER_PACKAGES = ("chelis-compiler-api", "chelis-cli")


class LedgerTargetError(ValueError):
    """Cargo metadata does not describe a runnable ledger target set."""


def ledger_targets(metadata: Mapping[str, Any]) -> dict[str, list[str]]:
    """Map each package in ``LEDGER_PACKAGES`` to its sorted ledger targets."""
    members = set(metadata["workspace_members"])
    found: dict[str, list[str]] = {}
    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        for target in package["targets"]:
            features = target.get("required-features", [])
            if LEDGER_FEATURE not in features:
                continue
            identity = f"{package['name']}::{target['name']}"
            if target.get("kind") != ["test"]:
                raise LedgerTargetError(
                    f"{identity} requires {LEDGER_FEATURE} but is not an "
                    "integration test target"
                )
            if features != [LEDGER_FEATURE]:
                raise LedgerTargetError(
                    f"{identity} requires {features}; the ledger command "
                    f"activates only {LEDGER_FEATURE}"
                )
            found.setdefault(package["name"], []).append(target["name"])
    unplaced = sorted(set(found) - set(LEDGER_PACKAGES))
    if unplaced:
        raise LedgerTargetError(
            f"{LEDGER_FEATURE} targets in packages no ledger command runs: "
            f"{unplaced}"
        )
    empty = [package for package in LEDGER_PACKAGES if package not in found]
    if empty:
        raise LedgerTargetError(
            f"no {LEDGER_FEATURE} targets in ledger packages: {empty}"
        )
    return {package: sorted(found[package]) for package in LEDGER_PACKAGES}


def ledger_command(package: str, metadata: Mapping[str, Any]) -> list[str]:
    """The exact ``cargo nextest`` command for one ledger package."""
    if package not in LEDGER_PACKAGES:
        raise LedgerTargetError(f"{package} is not a ledger package")
    command = [
        "cargo", "nextest", "run", "-p", package, "--features", LEDGER_FEATURE,
    ]
    for target in ledger_targets(metadata)[package]:
        command.extend(["--test", target])
    return command


def cargo_metadata(root: Path = REPO_ROOT) -> dict[str, Any]:
    completed = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1", "--locked"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("package", choices=LEDGER_PACKAGES)
    parser.add_argument(
        "--print",
        action="store_true",
        help="print the derived command without running it",
    )
    arguments = parser.parse_args(argv)
    try:
        command = ledger_command(arguments.package, cargo_metadata())
    except (LedgerTargetError, subprocess.CalledProcessError) as error:
        print(f"ownership-ledger tests: {error}", file=sys.stderr)
        return 1
    print(shlex.join(command), flush=True)
    if arguments.print:
        return 0
    os.chdir(REPO_ROOT)
    os.execvp(command[0], command)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
