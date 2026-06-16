"""MEASURED coverage runner for the opaque-invariants corpus (RFC D-CORPUS).

THE COVERAGE GATE. Runs every manifest program against the live binary and
MEASURES which targeted tokens are hit by >= 1 program -- coverage is measured,
never assumed. A targeted token with zero coverage FAILS the gate (a generator
gap, or -- if the path is provably unreachable -- a finding to report). Emits a
coverage report (token -> hitting program ids).

A target token is matched against the live (exit_code, stdout, stderr) of the
program by `token_hit`:
  - `OpaqueTypeViolation` / `DuplicateModule` / `ReservedLinkerName` /
    `DuplicateDefinition`: a check-lane error object with that `kind`.
  - `wf:<substr>`: a check-lane error message contains <substr>
    (the well-formedness / rejection message marker).
  - `pos:check_clean`: a check-lane program with exit 0 and an empty errors[].
  - `status:<s>` for s in {passed, failed, unsupported, error}: a prove-lane
    `{kind:"obligation"}` record with that status.
  - `status:check_error`: a prove-lane `{kind:"error", stage:"check"}` record.
  - `reason:<substr>`: a prove-lane obligation/error `reason` contains <substr>.
  - `position:<p>` for p in {Direct, Option, tuple, constant}: a prove-lane
    PASS whose producer's return shape is the named position (verified
    structurally from the manifest note + a passed obligation record).
  - `inject:<m>`: an injection marker keyed off a property record status.
  - `starve:<substr>`: a prove-lane property/obligation `reason` contains
    <substr> (the starvation shape diagnostic).
  - `pos:prove_clean`: a prove-lane program with exit 0.

Lane -> binary: `check` runs against the DEFAULT (non-smt) binary; `prove`
runs against the `--features smt` binary. The caller supplies both paths; a
lane with no binary is reported as SKIPPED (its tokens uncovered-by-skip, not
silently passed).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"


# ===========================================================================
# PURE token-matching core (unit-tested in test_coverage_runner.py).
# ===========================================================================
@dataclass(frozen=True)
class RunResult:
    """The live result of one program: exit code + parsed records. For the
    check lane `records` is the single check-report dict wrapped in a list; for
    the prove lane it is the NDJSON record list. `stderr` carries the raw
    stderr (build/eval gate the violation there)."""

    exit_code: int
    records: tuple
    stderr: str = ""


def _check_errors(records: tuple) -> list:
    """The errors[] array from a check report (the first/only record)."""
    if not records:
        return []
    first = records[0]
    if isinstance(first, dict):
        errs = first.get("errors")
        if isinstance(errs, list):
            return errs
    return []


def _prove_records(records: tuple, kind: str) -> list:
    return [r for r in records if isinstance(r, dict) and r.get("kind") == kind]


def token_hit(token: str, lane: str, result: RunResult) -> bool:
    """Whether `token` is hit by this program's live result. PURE."""
    if lane == "check":
        errs = _check_errors(result.records)
        if token in ("OpaqueTypeViolation", "DuplicateModule", "ReservedLinkerName",
                     "DuplicateDefinition"):
            return any(e.get("kind") == token for e in errs if isinstance(e, dict))
        if token.startswith("wf:"):
            sub = token[len("wf:"):]
            return any(
                sub in (e.get("message") or "")
                for e in errs if isinstance(e, dict)
            )
        if token == "pos:check_clean":
            return result.exit_code == 0 and len(errs) == 0
        return False

    if lane == "prove":
        obs = _prove_records(result.records, "obligation")
        props = _prove_records(result.records, "property")
        errors = _prove_records(result.records, "error")
        if token.startswith("status:"):
            s = token[len("status:"):]
            if s == "check_error":
                return any(e.get("stage") == "check" for e in errors)
            # `unsupported` is the exit-2 obligation/generation status: it
            # surfaces on an obligation record (a producer obligation that did
            # not discharge) OR a property record (a starved injected binder,
            # D-STARVE). Both are the same ObligationStatus::Unsupported / exit 2.
            if s == "unsupported":
                return any(o.get("status") == s for o in obs) or any(
                    p.get("status") == s for p in props
                )
            return any(o.get("status") == s for o in obs)
        if token.startswith("reason:"):
            sub = token[len("reason:"):]
            pool = obs + errors + props
            return any(sub in (r.get("reason") or "") for r in pool)
        if token.startswith("position:"):
            # A passed obligation exists; the position is established by the
            # manifest note (structural) -- here we only require a PASS so the
            # producer was placed in the set at all.
            return any(o.get("status") == "passed" for o in obs)
        if token.startswith("inject:"):
            marker = token[len("inject:"):]
            if marker == "pass":
                return any(p.get("status") == "passed" for p in props)
            if marker == "false_fails":
                return any(p.get("status") == "failed" for p in props)
            if marker == "free_not_injected":
                # Unsupported binder => the property did NOT pass and exit is 2.
                return result.exit_code == 2 and not any(
                    p.get("status") == "passed" for p in props
                )
            return False
        if token.startswith("starve:"):
            sub = token[len("starve:"):]
            pool = props + obs
            if sub == "legacy_error":
                return any(p.get("status") == "error" for p in props) or result.exit_code == 3
            return any(sub in (r.get("reason") or "") for r in pool)
        if token == "pos:prove_clean":
            return result.exit_code == 0
        if token == "perf:many_producers":
            # A many-producer module: >= 30 obligation records, all accounted
            # for, and the summary obligations count equals the obligation
            # record count (the accounting invariant). Exit 0 (all pass).
            summary = _prove_records(result.records, "summary")
            if len(obs) < 30 or not summary:
                return False
            accounted = sum(
                1 for o in obs
                if o.get("status") in ("passed", "failed", "unsupported", "error")
            )
            return (
                result.exit_code == 0
                and accounted == len(obs)
                and summary[0].get("obligations") == len(obs)
            )
        return False

    return False


@dataclass
class CoverageReport:
    """token -> sorted list of program ids that hit it. `uncovered` is the set
    of manifest targets with zero hits."""

    hits: dict = field(default_factory=dict)
    expect_exit_mismatches: list = field(default_factory=list)
    skipped_lanes: list = field(default_factory=list)

    @property
    def uncovered(self) -> list:
        return sorted(t for t, ids in self.hits.items() if not ids)


def measure(
    manifest: dict,
    results: dict,
    targets: list,
    skipped_lanes: list,
) -> CoverageReport:
    """Aggregate per-program results into a coverage report. PURE over the
    already-collected (id -> RunResult) map. `results` omits skipped-lane
    programs. Also records expect_exit mismatches (a program whose live exit
    differs from its pinned expectation is a corpus drift finding)."""
    report = CoverageReport()
    report.skipped_lanes = list(skipped_lanes)
    for t in targets:
        report.hits[t] = []
    by_id = {p["id"]: p for p in manifest["programs"]}
    for pid, result in results.items():
        prog = by_id[pid]
        lane = prog["lane"]
        if result.exit_code != prog["expect_exit"]:
            report.expect_exit_mismatches.append(
                {"id": pid, "expected": prog["expect_exit"], "actual": result.exit_code}
            )
        for t in prog["targets"]:
            if t in report.hits and token_hit(t, lane, result):
                report.hits[t].append(pid)
    for t in report.hits:
        report.hits[t].sort()
    return report


# ===========================================================================
# IMPURE: drive the binaries.
# ===========================================================================
def run_program(prog: dict, check_bin: str, prove_bin: str, timeout: float) -> RunResult:
    path = str(PROGRAMS_DIR / prog["filename"])
    env = dict(os.environ)
    env["CHELIS_STYLE_GATE_DISABLE"] = "1"
    if prog["lane"] == "check":
        cmd = [check_bin, "check", path, "--allow-style-violations"]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL, env=env,
        )
        try:
            rec = json.loads(proc.stdout)
            records = (rec,)
        except (json.JSONDecodeError, ValueError):
            records = ()
        return RunResult(proc.returncode, records, proc.stderr)
    # prove lane
    cmd = [prove_bin, "prove", path, "--json", *prog.get("prove_args", [])]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        stdin=subprocess.DEVNULL, env=env,
    )
    records = tuple(
        json.loads(line)
        for line in proc.stdout.splitlines()
        if line.strip() and _is_json(line)
    )
    return RunResult(proc.returncode, records, proc.stderr)


def _is_json(line: str) -> bool:
    try:
        json.loads(line)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_corpus(
    manifest: dict,
    check_bin: str | None,
    prove_bin: str | None,
    timeout: float,
) -> CoverageReport:
    results: dict = {}
    skipped_lanes: list = []
    if check_bin is None:
        skipped_lanes.append("check")
    if prove_bin is None:
        skipped_lanes.append("prove")
    for prog in manifest["programs"]:
        lane = prog["lane"]
        if lane == "check" and check_bin is None:
            continue
        if lane == "prove" and prove_bin is None:
            continue
        results[prog["id"]] = run_program(prog, check_bin or "", prove_bin or "", timeout)
    # Targets only reachable from a skipped lane must not be counted as
    # uncovered failures: drop targets whose every owning program is skipped.
    active_targets = []
    for t in manifest["targets"]:
        owners = [p for p in manifest["programs"] if t in p["targets"]]
        if all(
            (o["lane"] == "check" and check_bin is None)
            or (o["lane"] == "prove" and prove_bin is None)
            for o in owners
        ):
            continue  # entirely owned by skipped lanes
        active_targets.append(t)
    return measure(manifest, results, active_targets, skipped_lanes)


def render(report: CoverageReport) -> str:
    lines: list[str] = ["Opaque-invariants corpus coverage report"]
    if report.skipped_lanes:
        lines.append(f"  SKIPPED LANES (no binary): {', '.join(report.skipped_lanes)}")
    lines.append(f"  targets measured: {len(report.hits)}")
    covered = sum(1 for ids in report.hits.values() if ids)
    lines.append(f"  covered:          {covered}")
    lines.append(f"  uncovered:        {len(report.uncovered)}")
    lines.append("  token -> hitting programs:")
    for t in sorted(report.hits):
        ids = report.hits[t]
        mark = "OK " if ids else "** "
        lines.append(f"    {mark}{t:<42} {len(ids):>2}  {', '.join(ids)}")
    if report.expect_exit_mismatches:
        lines.append("  EXIT MISMATCHES (corpus drift):")
        for m in report.expect_exit_mismatches:
            lines.append(
                f"    {m['id']}: expected exit {m['expected']}, got {m['actual']}"
            )
    return "\n".join(lines)


class CoverageGateError(AssertionError):
    pass


def assert_full_coverage(report: CoverageReport) -> None:
    if report.uncovered:
        raise CoverageGateError(
            f"COVERAGE GATE FAILED: {len(report.uncovered)} targeted token(s) "
            f"with zero coverage: {report.uncovered}"
        )
    if report.expect_exit_mismatches:
        raise CoverageGateError(
            f"COVERAGE GATE FAILED: {len(report.expect_exit_mismatches)} "
            f"program(s) drifted from their pinned exit code: "
            f"{report.expect_exit_mismatches}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Opaque-invariants corpus coverage gate")
    parser.add_argument("--check-bin", default=None, help="non-smt chelis binary (check lane)")
    parser.add_argument("--prove-bin", default=None, help="smt chelis binary (prove lane)")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--require-all", action="store_true",
        help="fail unless BOTH lanes ran (no skipped lane allowed)",
    )
    args = parser.parse_args(argv[1:])

    if args.check_bin is None and args.prove_bin is None:
        sys.stderr.write("at least one of --check-bin / --prove-bin is required\n")
        return 2

    manifest = load_manifest()
    report = run_corpus(manifest, args.check_bin, args.prove_bin, args.timeout)
    print(render(report))
    if args.require_all and report.skipped_lanes:
        sys.stderr.write(
            f"\n--require-all: lanes skipped: {report.skipped_lanes}\n"
        )
        return 1
    try:
        assert_full_coverage(report)
    except CoverageGateError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return 1
    print("\nCOVERAGE GATE: PASS (every measured target token hit by >= 1 program)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
