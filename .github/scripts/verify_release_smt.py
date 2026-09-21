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
  * at least one emitted record has `proof_tier == "smt"` AND a
    `discharge_tier` with both `engine == "cvc5"` AND
    `guarantee == "smt"` (all three are required by
    `record_is_cvc5_smt`; the `guarantee` field pins the discharge to
    the SMT guarantee level, not just the cvc5 engine name), and
  * no record is the "requires the smt-enabled build" warning the
    default (feature-less) binary emits for the same input.

Fail condition (feature-less binary, i.e. the regression we guard):
  * the smt-disabled warning record is present, and/or
  * no cvc5 / smt discharge record is emitted.

Usage:
  verify_release_smt.py <path-to-chelis-binary>
  verify_release_smt.py --tarball <path-to-staged-release.tar.gz>
  verify_release_smt.py --runner <pinned-runtime-executable> <path-to-chelis-binary>

The `--tarball` form unpacks a staged release `.tar.gz` and verifies the
EXTRACTED (stripped) `bin/chelis` -- the most faithful chelis#422 guard,
since it checks the exact artifact shipped to users rather than a
pre-staging binary. It also catches a staging bug that copies the wrong
binary into the tarball.

Exit codes:
  0  binary discharges via cvc5 (smt feature live)
  2  binary (or tarball, or bin/chelis inside it) not found
  3  `chelis prove` did not run cleanly
  4  no cvc5/smt discharge record found (feature inert -- the regression)
  5  smt-disabled warning emitted (feature inert -- the regression)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tarfile
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


def run_prove(chelis: Path, src: Path, *, runner: Path | None = None) -> str:
    """Run `chelis prove --json --tier auto <src>` and return stdout.

    `prove` exits nonzero only on a property *failure*; an all-pass run
    exits 0. We capture stdout either way and inspect the records, so a
    nonzero exit with parseable records is not itself fatal here -- but
    a crash (no stdout, or unparseable) is.
    """
    cmd = [str(chelis), "prove", "--json", "--tier", "auto", str(src)]
    if runner is not None:
        cmd.insert(0, str(runner))
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


def verify_binary(chelis: Path, *, runner: Path | None = None) -> int:
    """Run the cvc5-discharge probe against `chelis` and return an exit code.

    See the module docstring for the exit-code meanings (0 pass; 3 no
    output; 4 no cvc5/smt record; 5 smt-disabled warning). Callers
    resolve/existence-check the path; this assumes `chelis` is runnable.
    """
    with tempfile.TemporaryDirectory(prefix="chelis_smt_verify_") as tmp:
        src = Path(tmp) / "probe.ch"
        src.write_text(PROBE_PROGRAM)
        stdout = run_prove(chelis, src, runner=runner)

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


def find_chelis_in_tree(root: Path) -> Path | None:
    """Locate the `bin/chelis` binary under an extracted release staging
    tree. The release tarball stages it at `<staging>/bin/chelis`."""
    matches = sorted(root.glob("*/bin/chelis"))
    if matches:
        return matches[0]
    # Fall back to any `chelis` file (defensive; tarball layout is fixed).
    any_chelis = sorted(p for p in root.rglob("chelis") if p.is_file())
    return any_chelis[0] if any_chelis else None


def extract_and_verify_tarball(tarball: Path, *, runner: Path | None = None) -> int:
    """Unpack a staged release `.tar.gz` and verify the EXTRACTED
    (stripped) `bin/chelis` discharges via cvc5 -- the most faithful
    chelis#422 guard (verify the artifact actually shipped, not a
    pre-staging proxy). Returns 2 if the tarball or its binary is
    missing, else delegates to `verify_binary`."""
    if not tarball.exists():
        print(
            f"verify_release_smt: tarball not found at {tarball}",
            file=sys.stderr,
        )
        return 2
    with tempfile.TemporaryDirectory(prefix="chelis_smt_tarball_") as tmp:
        root = Path(tmp)
        with tarfile.open(tarball, "r:gz") as tf:
            tf.extractall(root)
        chelis = find_chelis_in_tree(root)
        if chelis is None:
            print(
                f"verify_release_smt: no bin/chelis found inside {tarball}",
                file=sys.stderr,
            )
            return 2
        chelis.chmod(0o755)
        print(f"verify_release_smt: verifying extracted artifact {chelis}")
        return verify_binary(chelis.resolve(), runner=runner)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "chelis",
        type=Path,
        nargs="?",
        help="path to the `chelis` binary (e.g. ./target/release/chelis)",
    )
    group.add_argument(
        "--tarball",
        type=Path,
        help=(
            "path to a staged release .tar.gz; the EXTRACTED bin/chelis is "
            "verified (the most faithful chelis#422 guard)"
        ),
    )
    parser.add_argument(
        "--runner",
        type=Path,
        help="execute the binary through this pinned runtime executable (no shell parsing)",
    )
    args = parser.parse_args()
    runner = args.runner.resolve() if args.runner is not None else None

    if args.tarball is not None:
        return extract_and_verify_tarball(args.tarball.resolve(), runner=runner)

    chelis = args.chelis.resolve()
    if not chelis.exists():
        print(
            f"verify_release_smt: chelis binary not found at {chelis}",
            file=sys.stderr,
        )
        return 2
    return verify_binary(chelis, runner=runner)


if __name__ == "__main__":
    sys.exit(main())
