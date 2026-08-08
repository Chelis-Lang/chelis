#!/usr/bin/env python3
"""Assert the off-Nix consumption environment's glibc matches the contract.

The release derivation records the portable binary's glibc floor in
nix/contracts.nix (`linuxReleaseGlibcFloor`) beside the glibc version of
the consumption environment (`linuxReleaseConsumptionGlibc`). This script
runs inside the digest-pinned consumption container and asserts that the
container's actual glibc equals the recorded consumption value, so the
image pin and the contract cannot drift apart silently.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def contract_value(contracts_text: str, key: str) -> str:
    match = re.search(rf'{re.escape(key)} = "(\d+\.\d+)"', contracts_text)
    if match is None:
        raise SystemExit(f"nix/contracts.nix does not record {key}")
    return match.group(1)


def host_glibc_version() -> str:
    first_line = subprocess.check_output(["ldd", "--version"], text=True).splitlines()[0]
    match = re.search(r"(\d+\.\d+)\s*$", first_line)
    if match is None:
        raise SystemExit(f"cannot parse the glibc version from: {first_line!r}")
    return match.group(1)


def main() -> int:
    contracts = Path("nix/contracts.nix").read_text(encoding="utf-8")
    expected = contract_value(contracts, "linuxReleaseConsumptionGlibc")
    floor = contract_value(contracts, "linuxReleaseGlibcFloor")
    actual = host_glibc_version()
    print(f"consumption glibc: {actual} (contract: {expected}, binary floor: {floor})")
    if actual != expected:
        print(
            f"verify_consumption_glibc: container glibc {actual} differs from "
            f"the recorded linuxReleaseConsumptionGlibc {expected}; update the "
            "container digest and the contract together",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
