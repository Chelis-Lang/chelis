#!/usr/bin/env python3
"""Post-build verification that a release `chelis` binary discharges
property obligations through cvc5 (the `smt` feature), not the
solver-free fuzz fallback.

chelis#422 (WS-4): release.yml builds `chelis-cli --features smt` so the
shipped binary uses cvc5. The engines are inert in a binary built
without the feature, and that regression is invisible from the outside
unless something asserts the discharge path. This script is that
assertion: it runs `chelis prove --json --tier auto` on a small,
self-contained property whose producer obligation routes to cvc5, then
checks the emitted NDJSON for the cvc5 discharge signal.

The probe is a `@opaque` type carrying a unit-interval `@invariant` plus
a producer (`scale_half`) whose obligation -- "given `u.value` in
[0,1], `u.value * 0.5` stays in [0,1]" -- is a forall over a continuous
real domain that only Tier B (cvc5) certifies; the fuzz tier cannot.

Pass condition (smt-enabled binary):
  * at least one emitted record has `proof_tier == "smt"` AND
    `discharge_tier.engine == "cvc5"`, and
  * no record is the "requires the smt-enabled build" warning the
    default (feature-less) binary emits for the same input.

Fail condition (feature-less binary, i.e. the regression we guard):
  * the smt-disabled warning record is present, and/or
  * no cvc5 / smt discharge record is emitted.

Usage:
  verify_release_smt.py <path-to-chelis-binary>

Exit codes:
  0  binary discharges via cvc5 (smt feature live)
  2  binary not found
  3  `chelis prove` did not run cleanly
  4  no cvc5/smt discharge record found (feature inert -- the regression)
  5  smt-disabled warning emitted (feature inert -- the regression)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


# A producer obligation that routes to Tier B (cvc5): proving
# `u.value * 0.5` stays in the unit interval for every `u.value` in
# [0,1] is a forall over a continuous domain, not a finite fuzz check.
PROBE_PROGRAM = """\
module Smt.Probe
export (make, scale_half)
@opaque
@invariant(u) ((u.value >= 0.0) && (u.value <= 1.0))
type Unit =
  | Unit { value: f32 }
def make(x: f32) -> Option[Unit] = if ((x >= 0.0) && (x <= 1.0)) then Some(Unit { value: x }) else None
def scale_half(u: Unit) -> Unit = Unit { value: (u.value * 0.5) }
"""

# Substring of the structured warning the feature-less binary emits on
# the producer-obligation path; its presence means cvc5 is not linked.
SMT_DISABLED_MARKER = "requires the smt-enabled build"


def run_prove(chelis: Path, src: Path) -> str:
    """Run `chelis prove --json --tier auto <src>` and return stdout.

    `prove` exits nonzero only on a property *failure*; an all-pass run
    exits 0. We capture stdout either way and inspect the records, so a
    nonzero exit with parseable records is not itself fatal here -- but
    a crash (no stdout, or unparseable) is.
    """
    cmd = [str(chelis), "prove", "--json", "--tier", "auto", str(src)]
    print("+ " + " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        print("verify_release_smt: `chelis prove` timed out", file=sys.stderr)
        sys.exit(3)
    if proc.stderr:
        # Forward stderr (warnings, diagnostics) for the CI log.
        print(proc.stderr, file=sys.stderr, end="")
    return proc.stdout


def parse_records(stdout: str) -> list[dict]:
    records: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # `prove` may interleave a non-JSON human warning line even
            # under --json; skip non-JSON lines rather than fail parsing.
            continue
    return records


def record_is_cvc5_smt(record: dict) -> bool:
    """True if `record` is an obligation discharged by cvc5 at the SMT tier."""
    if record.get("proof_tier") != "smt":
        return False
    for assumption in record.get("assumptions", []):
        tier = assumption.get("discharge_tier", {})
        if tier.get("engine") == "cvc5" and tier.get("guarantee") == "smt":
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chelis",
        type=Path,
        help="path to the `chelis` binary (e.g. ./target/release/chelis)",
    )
    args = parser.parse_args()

    chelis = args.chelis.resolve()
    if not chelis.exists():
        print(
            f"verify_release_smt: chelis binary not found at {chelis}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="chelis_smt_verify_") as tmp:
        src = Path(tmp) / "probe.ch"
        src.write_text(PROBE_PROGRAM)
        stdout = run_prove(chelis, src)

    print(stdout, end="" if stdout.endswith("\n") else "\n")

    if not stdout.strip():
        print(
            "verify_release_smt: `chelis prove` produced no output",
            file=sys.stderr,
        )
        return 3

    if SMT_DISABLED_MARKER in stdout:
        print(
            "verify_release_smt: FAIL -- the binary reports that producer "
            "obligation verification requires the smt-enabled build. The "
            "release binary was built WITHOUT --features smt; cvc5 is inert "
            "and `prove` silently degrades to the solver-free fuzz path "
            "(chelis#422 regression).",
            file=sys.stderr,
        )
        return 5

    records = parse_records(stdout)
    if any(record_is_cvc5_smt(r) for r in records):
        print(
            "verify_release_smt: OK -- producer obligation discharged via "
            "cvc5 (proof_tier=smt, engine=cvc5). The smt feature is live in "
            "this binary."
        )
        return 0

    print(
        "verify_release_smt: FAIL -- no obligation was discharged via cvc5 "
        "(expected a record with proof_tier=\"smt\" and "
        "discharge_tier.engine=\"cvc5\"). cvc5 appears inert; the release "
        "binary may have been built without --features smt (chelis#422 "
        "regression).",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())
