#!/usr/bin/env python3
"""Authoritative chelis#729 Phase 3 acceptance oracle.

Phase 3 inherits the complete Phase 2 numeric contract and the generated
observation contract, then turns on the compiled-C dtype rows.  Acceptance is
exit 0 with the final line ``DTYPE PHASE 3 ORACLE: PASS``.

Usage:

    .venv/bin/python scripts/dtype_phase3_oracle.py
"""

from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import sys

import dtype_oracle_manifest as manifest
import faithful_observation_phase3_oracle as faithful


REPO_ROOT = Path(__file__).resolve().parents[1]


def oracle_legs(python: str) -> tuple[manifest.OracleLeg, ...]:
    """Return the frozen Phase 3 command manifest in execution order."""
    return manifest.phase3_legacy_legs(python)


def _target_root() -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR", "target"))
    return target if target.is_absolute() else REPO_ROOT / target


def run_oracle(python: str = sys.executable) -> None:
    violations = faithful.preflight_violations()
    if violations:
        raise SystemExit(
            "DTYPE PHASE 3 ORACLE: FAIL: " + "; ".join(violations)
        )

    output_dir = _target_root() / "dtype-phase3"
    output_dir.mkdir(parents=True, exist_ok=True)
    ownership_path = output_dir / "phase-ownership.json"
    runtime_receipt_path = output_dir / "observation-runtime-receipts.tsv"
    runtime_receipt_path.unlink(missing_ok=True)
    manifest.write_ownership_receipt(ownership_path, python)
    print(f"phase ownership receipt: {ownership_path}", flush=True)

    nextest_command = manifest.flattened_nextest_command(python)
    env = os.environ.copy()
    env["CHELIS_PHASE3_RECEIPT_PATH"] = str(runtime_receipt_path)
    print(
        f"[1/4] flattened Phase 0-3 nextest union: "
        f"{shlex.join(nextest_command)}",
        flush=True,
    )
    try:
        subprocess.run(
            nextest_command,
            cwd=REPO_ROOT,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "DTYPE PHASE 3 ORACLE: FAIL: flattened nextest union "
            f"(exit {error.returncode})"
        ) from error

    runtime_receipts = (
        runtime_receipt_path.read_text(encoding="utf-8")
        if runtime_receipt_path.exists()
        else ""
    )
    receipt_violations = faithful.receipt_violations(runtime_receipts)
    if receipt_violations:
        raise SystemExit(
            "DTYPE PHASE 3 ORACLE: FAIL: "
            + "; ".join(receipt_violations)
        )

    non_test_legs = manifest.non_test_legs(python)
    for index, leg in enumerate(non_test_legs, start=2):
        print(
            f"[{index}/4] {leg.name}: {shlex.join(leg.argv)}",
            flush=True,
        )
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"DTYPE PHASE 3 ORACLE: FAIL: {leg.name} "
                f"(exit {error.returncode})"
            ) from error
    print("DTYPE PHASE 3 ORACLE: PASS")


if __name__ == "__main__":
    run_oracle()
