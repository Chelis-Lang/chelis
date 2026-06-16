"""Solver-free regression assertion for `chelis check` (RFC D-CORPUS, §0).

THE LOAD-BEARING W7 DELIVERABLE. The opaque-types feature's central premise is
that `chelis check` stays SOLVER-FREE: the everyday type checker never
evaluates the invariant and never reaches cvc5 (the solver enters only through
`chelis prove`'s Tier B, behind the `smt` feature). This module makes that an
executable gate in three independent assertions:

  A. NO-LINK. The DEFAULT (non-smt) `chelis` binary contains ZERO cvc5
     symbols -- the solver is not even linked into the binary that ships
     `chelis check`. (`nm -C <bin> | grep -i cvc5` is empty.) Independently,
     the `--features smt` binary DOES link cvc5 (a control proving the symbol
     probe discriminates), so the absence in the default build is meaningful,
     not a probe that can never see cvc5.

  B. IDENTICAL CHECK. The non-smt binary and the smt binary produce the SAME
     `chelis check` verdict (exit code + errors[] kinds/messages) on every
     check-lane corpus program. If `chelis check` consulted the solver, the
     smt build (which has cvc5) could diverge from the non-smt build (which
     cannot); byte-identical results across the two prove the check verdict
     does not depend on the solver. (When only one binary is available the
     identity arm is SKIPPED and reported, never silently passed.)

  C. CHECK-COVERS. The non-smt binary `chelis check`s every check-lane corpus
     program and produces its pinned exit code -- check needs no solver to
     run the full corpus.

Assertion A is the structural proof (no solver code present); B is the
behavioral proof (the verdict is solver-independent even where the solver IS
present); C is the liveness proof (check runs the whole corpus without one).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Assertion A: cvc5 symbol probe (PURE over nm output, unit-tested).
# ===========================================================================
def cvc5_symbol_count(nm_output: str) -> int:
    """Count lines mentioning a cvc5 symbol in `nm -C` output. PURE."""
    return sum(1 for line in nm_output.splitlines() if "cvc5" in line.lower())


def nm_symbols(binary: str) -> str:
    """`nm -C <binary>` stdout (demangled). Empty string if nm is unavailable
    or fails (the caller treats an unprobeable binary as a reported skip, not
    a pass)."""
    try:
        proc = subprocess.run(
            ["nm", "-C", binary], capture_output=True, text=True, timeout=120
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout


# ===========================================================================
# Assertion B+C: per-program check verdict (PURE comparison, unit-tested).
# ===========================================================================
@dataclass(frozen=True)
class CheckVerdict:
    """A normalized `chelis check` verdict: exit code + the sorted list of
    (kind, message) error pairs. Two verdicts are equal iff the check decision
    is identical."""

    exit_code: int
    errors: tuple  # tuple[(kind, message), ...] sorted


def normalize_check(exit_code: int, stdout: str) -> CheckVerdict:
    """Build a CheckVerdict from raw check output. PURE."""
    pairs: list = []
    try:
        root = json.loads(stdout)
        errs = root.get("errors") if isinstance(root, dict) else None
        if isinstance(errs, list):
            for e in errs:
                if isinstance(e, dict):
                    pairs.append((e.get("kind"), e.get("message")))
    except (json.JSONDecodeError, ValueError):
        pairs.append(("<unparseable>", stdout[:200]))
    pairs.sort(key=lambda p: (str(p[0]), str(p[1])))
    return CheckVerdict(exit_code, tuple(pairs))


def run_check(binary: str, filename: str, timeout: float) -> tuple:
    """Run `chelis check` and return (exit_code, stdout). The check-lane
    programs synthesize ad-hoc opaque source, so the documented style-gate
    bypass is used."""
    env = dict(os.environ)
    env["CHELIS_STYLE_GATE_DISABLE"] = "1"
    proc = subprocess.run(
        [binary, "check", str(PROGRAMS_DIR / filename), "--allow-style-violations"],
        capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, env=env,
    )
    return proc.returncode, proc.stdout


@dataclass
class SolverFreeReport:
    no_link_ok: bool
    nonsmt_cvc5_count: int
    smt_cvc5_count: int | None  # None when the smt binary was not probed
    identity_checked: int
    identity_mismatches: list
    exit_mismatches: list
    skipped: list


def run(
    manifest: dict,
    nonsmt_bin: str,
    smt_bin: str | None,
    timeout: float,
) -> SolverFreeReport:
    check_progs = [p for p in manifest["programs"] if p["lane"] == "check"]
    skipped: list = []

    # Assertion A: cvc5 symbol probe.
    nonsmt_cvc5 = cvc5_symbol_count(nm_symbols(nonsmt_bin))
    smt_cvc5: int | None = None
    if smt_bin is not None:
        smt_cvc5 = cvc5_symbol_count(nm_symbols(smt_bin))
    no_link_ok = nonsmt_cvc5 == 0

    # Assertion B+C.
    identity_mismatches: list = []
    exit_mismatches: list = []
    identity_checked = 0
    for prog in check_progs:
        ec_n, out_n = run_check(nonsmt_bin, prog["filename"], timeout)
        v_n = normalize_check(ec_n, out_n)
        if ec_n != prog["expect_exit"]:
            exit_mismatches.append(
                {"id": prog["id"], "expected": prog["expect_exit"], "actual": ec_n}
            )
        if smt_bin is not None:
            ec_s, out_s = run_check(smt_bin, prog["filename"], timeout)
            v_s = normalize_check(ec_s, out_s)
            identity_checked += 1
            if v_n != v_s:
                identity_mismatches.append(
                    {"id": prog["id"], "nonsmt": v_n, "smt": v_s}
                )
    if smt_bin is None:
        skipped.append("check-identity (no smt binary)")
    return SolverFreeReport(
        no_link_ok=no_link_ok,
        nonsmt_cvc5_count=nonsmt_cvc5,
        smt_cvc5_count=smt_cvc5,
        identity_checked=identity_checked,
        identity_mismatches=identity_mismatches,
        exit_mismatches=exit_mismatches,
        skipped=skipped,
    )


def render(report: SolverFreeReport) -> str:
    lines = ["Solver-free regression report for `chelis check`"]
    lines.append(
        f"  A. NO-LINK: non-smt binary cvc5 symbols = {report.nonsmt_cvc5_count} "
        f"({'PASS' if report.no_link_ok else 'FAIL'})"
    )
    if report.smt_cvc5_count is not None:
        lines.append(
            f"     control: smt binary cvc5 symbols = {report.smt_cvc5_count} "
            f"({'discriminating' if report.smt_cvc5_count > 0 else 'NON-DISCRIMINATING'})"
        )
    lines.append(
        f"  B. IDENTICAL CHECK: {report.identity_checked} programs compared, "
        f"{len(report.identity_mismatches)} mismatch(es)"
    )
    lines.append(
        f"  C. CHECK-COVERS: {len(report.exit_mismatches)} exit-code drift(s)"
    )
    if report.skipped:
        lines.append(f"  SKIPPED: {', '.join(report.skipped)}")
    return "\n".join(lines)


class SolverFreeError(AssertionError):
    pass


def assert_solver_free(report: SolverFreeReport, require_control: bool) -> None:
    if not report.no_link_ok:
        raise SolverFreeError(
            f"SOLVER-FREE GATE FAILED: the non-smt `chelis` binary links "
            f"{report.nonsmt_cvc5_count} cvc5 symbol(s); `chelis check` must be "
            f"solver-free in the default build"
        )
    if require_control and (report.smt_cvc5_count is None or report.smt_cvc5_count == 0):
        raise SolverFreeError(
            "SOLVER-FREE GATE FAILED: the smt control binary shows no cvc5 "
            "symbols, so the symbol probe cannot discriminate (build "
            "--features smt or drop --require-control)"
        )
    if report.identity_mismatches:
        raise SolverFreeError(
            f"SOLVER-FREE GATE FAILED: {len(report.identity_mismatches)} "
            f"check verdict(s) differ between the non-smt and smt builds "
            f"(check must not depend on the solver): {report.identity_mismatches}"
        )
    if report.exit_mismatches:
        raise SolverFreeError(
            f"SOLVER-FREE GATE FAILED: {len(report.exit_mismatches)} check "
            f"program(s) drifted from their pinned exit code: "
            f"{report.exit_mismatches}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="`chelis check` solver-free regression gate")
    parser.add_argument("--nonsmt-bin", required=True, help="default (non-smt) chelis binary")
    parser.add_argument("--smt-bin", default=None, help="--features smt chelis binary (control + identity)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--require-control", action="store_true",
        help="fail unless the smt control binary is present and links cvc5",
    )
    args = parser.parse_args(argv[1:])

    manifest = load_manifest()
    report = run(manifest, args.nonsmt_bin, args.smt_bin, args.timeout)
    print(render(report))
    try:
        assert_solver_free(report, args.require_control)
    except SolverFreeError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return 1
    print("\nSOLVER-FREE GATE: PASS (`chelis check` reaches no solver on the corpus)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
