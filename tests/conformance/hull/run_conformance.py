"""Hull Phase 5b conformance runner (chelis-side, CI orchestration).

THE GATE. Streams the frozen, version-stamped Hull conformance corpus, runs the
LIVE chelis binary per program, and asserts the acceptance oracle: ZERO
CompilerUnsound, ZERO unexplained Disagree, ZERO EvalDisagree, EvalAgree >= floor.

DIVISION OF SEMANTICS (authoritative_verdict_pinning). The only semantic content
-- what type a program has, accept vs reject, the reference scalar -- is FROZEN in
the corpus as canonical strings Hull itself emitted. This runner does ONLY a
mechanical re-render (wire_to_canonical) + plain string equality + the
`differential_check`-mirroring outcome table. There is NO Python type lattice and
NO re-implemented types_equal: the runner cannot silently re-decide semantics.

The pure core (derive_compiler_outcome, classify_program, derive_eval_outcome,
assert_acceptance_oracle, classify_corpus) is unit-tested in
test_run_conformance.py over synthetic (verdict-record, exit_code, stdout) tuples.
The subprocess-driving path (run_program, run_corpus) is exercised by the corpus
run itself.

Usage (the uv-managed interpreter, per the no-shell scripting policy):
    .venv/bin/python tests/conformance/hull/run_conformance.py \
        --chelis-bin /abs/path/to/chelis [--check-n N --eval-n M] [--strict] \
        [--allow-version-skew]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Local import (this script's directory is the package dir).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wire_canonical import WireNormalizationError, wire_to_canonical  # noqa: E402

HERE = Path(__file__).resolve().parent
PROGRAMS_DIR = HERE / "programs"
MANIFEST_PATH = HERE / "manifest.json"
VERDICTS_PATH = HERE / "verdicts.jsonl"

# The seven check-lane buckets + four eval-lane buckets, reusing the EXACT shape
# from Hull's run_differential_suite.py so the report format is identical across
# repos.
CHECK_BUCKETS: tuple[str, ...] = (
    "agree", "agree_reject", "disagree", "too_conservative", "unsound",
    "reference_gap", "crash",
)
EVAL_BUCKETS: tuple[str, ...] = ("agree", "disagree", "ref_not_value", "compiler_crash")

# Float comparison tolerance for the eval lane (the f32 tolerance Hull's campaign
# pinned: run_differential_suite.ch tol = 0.01).
EVAL_TOL = 0.01


# ============================================================================
# COMPILER-VERDICT DERIVATION. Mirrors classify_compiler_check (check.ch 138):
# the crash-vs-reject gate that keeps a crash out of the agreement table.
# ============================================================================
@dataclass(frozen=True)
class CompilerVerdict:
    """The runner's view of what the live compiler decided. `kind` is one of
    'accept' | 'reject' | 'crash'. On accept, `canonical` is the
    wire_to_canonical string (or None when the type is un-representable, mirroring
    che_wire_type_to_type returning None); `accepts_fact` is the weaker
    acceptance fact (exit 0 + empty errors), used by the accept-aware refinement.
    On crash, `reason` explains."""

    kind: str
    canonical: str | None = None
    accepts_fact: bool = False
    reason: str = ""


def derive_compiler_check_verdict(exit_code: int, stdout: str) -> CompilerVerdict:
    """Derive a CompilerVerdict from the raw (exit_code, stdout) of `chelis check
    --show-inferred`. Mirrors classify_compiler_check + read_compiler_accept:
      - accept iff exit 0 AND valid JSON AND errors[] empty.
      - reject iff exit 2 AND valid JSON AND errors[] non-empty.
      - crash otherwise (exit not in {0,2}, unparseable stdout, exit/errors
        mismatch, missing fields).
    On accept, the canonical type string is wire_to_canonical of the FIRST
    inferred signature (a bare top-level def yields exactly one). A value def has
    an EMPTY inferred_signatures[] -- that is a genuine accept whose type cannot
    be string-compared, so canonical=None with accepts_fact=True (the accept-aware
    path)."""
    if exit_code not in (0, 2):
        return CompilerVerdict("crash", reason=f"unexpected exit code {exit_code}")
    try:
        root = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return CompilerVerdict("crash", reason=f"stdout is not valid JSON at exit {exit_code}")
    if not isinstance(root, dict):
        return CompilerVerdict("crash", reason="check JSON is not an object")
    errors = root.get("errors")
    if not isinstance(errors, list):
        return CompilerVerdict("crash", reason="check JSON has no errors[] array")
    has_errors = len(errors) > 0
    accepts_fact = exit_code == 0 and not has_errors
    if accepts_fact:
        return _read_compiler_accept(root)
    if exit_code == 2 and has_errors:
        return CompilerVerdict("reject")
    return CompilerVerdict(
        "crash",
        reason=f"exit / errors mismatch: exit {exit_code} "
        + ("with errors" if has_errors else "without errors"),
    )


def _read_compiler_accept(root: dict) -> CompilerVerdict:
    """Build an accept CompilerVerdict by reading the type + effect row from the
    FIRST inferred_signatures[] entry. Mirrors read_compiler_accept: an empty
    inferred_signatures[] (a non-function value def) is a genuine accept whose
    type is un-representable -> canonical=None, accepts_fact=True. A wire type
    wire_to_canonical maps to None (var / f64 / etc.) is the same accept-aware
    case."""
    sigs = root.get("inferred_signatures")
    if not isinstance(sigs, list) or len(sigs) == 0:
        # Genuine accept, no inferred signature (value def). Accept-aware path.
        return CompilerVerdict("accept", canonical=None, accepts_fact=True)
    first = sigs[0]
    if not isinstance(first, dict):
        return CompilerVerdict("crash", reason="inferred signature is not an object")
    try:
        canonical = wire_to_canonical(
            first.get("display_signature_structured"), first.get("effect_row")
        )
    except WireNormalizationError as exc:
        return CompilerVerdict("crash", reason=f"wire normalization failed: {exc}")
    return CompilerVerdict("accept", canonical=canonical, accepts_fact=True)


# ============================================================================
# THE OUTCOME TABLE. Mirrors differential_check (check.ch 111) +
# differential_check_accept_aware (check.ch 466), re-expressed over the pinned
# Hull canonical STRING instead of a live Hull Type. The only semantic input is
# the frozen string; the runner makes no type decision of its own.
# ============================================================================
def classify_check_program(
    record: dict, exit_code: int, stdout: str
) -> tuple[str, str]:
    """Classify ONE check/reject-lane program. Returns (bucket, detail) where
    bucket is one of CHECK_BUCKETS and detail is a human reason (used for the
    findings list on a failing bucket).

    `record` is the frozen verdict record; it carries `hull_verdict`
    ('accept'|'reject'), `hull_type_canonical` (the pinned string or None on
    reject), `hull_effects_canonical`, and `gap_family`. The decision table:
      crash                              -> crash
      hull reject & comp reject          -> agree_reject
      hull reject & comp accept & GapNone-> unsound (FAIL)
      hull reject & comp accept & gap!=  -> reference_gap (pass)
      hull accept & comp reject          -> too_conservative (flag)
      hull accept & comp accept & strings equal (or accept-aware) -> agree
      hull accept & comp accept & strings differ                  -> disagree (FAIL)
    """
    verdict = derive_compiler_check_verdict(exit_code, stdout)
    hull_verdict = record["hull_verdict"]
    gap_family = record.get("gap_family", "GapNone")

    if verdict.kind == "crash":
        return "crash", verdict.reason
    if verdict.kind == "reject":
        if hull_verdict == "reject":
            return "agree_reject", ""
        # hull accept & comp reject: sound but Hull-incomplete.
        return "too_conservative", "Hull accepts, compiler rejects"
    # verdict.kind == "accept"
    if hull_verdict == "reject":
        if gap_family in ("GapAdtFragment", "GapBuiltinNameShadow"):
            return "reference_gap", gap_family
        live = verdict.canonical if verdict.canonical is not None else "<accept-aware>"
        return "unsound", (
            f"Hull rejects (gap=GapNone) but compiler accepts with type {live}"
        )
    # hull accept & comp accept.
    hull_canonical = record.get("hull_type_canonical")
    # ACCEPT-AWARE refinement (differential_check_accept_aware): when the compiler
    # accepted but the type is un-representable (canonical None, e.g. a value def
    # with empty inferred_signatures[] or an unsolved type var) AND Hull also
    # accepts, both accept -> Agree on the acceptance fact. Soundness preserved:
    # this only fires when Hull ACCEPTS, so it can never hide a CompilerUnsound.
    if verdict.canonical is None:
        if verdict.accepts_fact:
            return "agree", "accept-aware (type un-representable, both accept)"
        return "crash", "compiler accept with no comparable type"
    # Both have a comparable string -> plain string equality is the decision.
    if verdict.canonical == hull_canonical:
        return "agree", ""
    return "disagree", (
        f"type mismatch: hull {hull_canonical!r} vs compiler {verdict.canonical!r}"
    )


# ============================================================================
# EVAL-LANE OUTCOME. Mirrors compiler_eval_scalar + read_root_scalar +
# differential_eval (check.ch 528-704).
# ============================================================================
def _read_root_scalar(root_value: object) -> float | None:
    """The f32 view of one EvaluatedRoot's value, mirroring read_root_scalar.
    float64/float32 read directly; an explicit JSON null is a NON-FINITE render
    -> NaN sentinel (Some(NaN), NOT None). int64 reads as float; bool -> 1.0/0.0;
    tensor reads data[0]. Other value types -> None."""
    if not isinstance(root_value, dict):
        return None
    ty = root_value.get("type")
    val = root_value.get("value")
    if ty in ("float64", "float32"):
        if val is None:
            return math.nan  # the compiler-null non-finite sentinel
        if isinstance(val, (int, float)):
            return float(val)
        return None
    if ty == "int64":
        if isinstance(val, int):
            return float(val)
        return None
    if ty == "bool":
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        return None
    if ty == "tensor":
        if isinstance(val, dict):
            data = val.get("data")
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
                return float(data[0])
        return None
    return None


def compiler_eval_scalar(exit_code: int, stdout: str) -> float | None:
    """The compiler's scalar view from `chelis eval --json`, mirroring
    compiler_eval_scalar. Reads the first root's value. exit != 0, non-JSON, no
    roots[] -> None (EvalCompilerCrash on the runner side)."""
    if exit_code != 0:
        return None
    try:
        root = json.loads(stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(root, dict):
        return None
    roots = root.get("roots")
    if not isinstance(roots, list) or len(roots) == 0:
        return None
    first = roots[0]
    if not isinstance(first, dict):
        return None
    return _read_root_scalar(first.get("value"))


def _is_finite(x: float) -> bool:
    return math.isfinite(x)


def derive_eval_outcome(
    reference_scalar: float | None, compiler_scalar: float | None, tol: float
) -> tuple[str, str]:
    """PURE eval-agreement classifier, mirroring differential_eval (check.ch 696).
    Returns (bucket, detail) over EVAL_BUCKETS. NON-FINITE reconciliation:
    compiler-null (NaN sentinel) AND Hull-non-finite -> agree; a finite value is
    never close to a non-finite one. NaN never matches under tolerance."""
    if reference_scalar is None:
        return "ref_not_value", ""
    if compiler_scalar is None:
        return "compiler_crash", "compiler eval produced no parseable scalar root"
    c = compiler_scalar
    r = reference_scalar
    # compiler reported a non-finite (NaN sentinel) AND Hull also non-finite.
    if math.isnan(c) and not _is_finite(r):
        return "agree", ""
    # f32_close: NaN never matches; abs diff within tol.
    if math.isnan(r) or math.isnan(c):
        return "disagree", f"reference {r} vs compiler {c}"
    if abs(r - c) <= tol:
        return "agree", ""
    return "disagree", f"reference {r} vs compiler {c}"


def classify_eval_program(
    record: dict, exit_code: int, stdout: str, tol: float = EVAL_TOL
) -> tuple[str, str]:
    """Classify ONE eval-lane program. The reference scalar is FROZEN in the
    record (hull_eval_value); the compiler scalar is read live."""
    ref = record.get("hull_eval_value")
    reference_scalar = None if ref is None else float(ref)
    compiler_scalar = compiler_eval_scalar(exit_code, stdout)
    return derive_eval_outcome(reference_scalar, compiler_scalar, tol)


# ============================================================================
# THE PER-PROGRAM CLASSIFIER (pure over a record + a live result).
# ============================================================================
@dataclass(frozen=True)
class ProgramResult:
    """One classified program: the record id, lane, bucket, detail, and program
    path (for a reproducible finding)."""

    id: str
    lane: str
    bucket: str
    detail: str
    program_path: str


def classify_program(record: dict, exit_code: int, stdout: str) -> ProgramResult:
    """Classify one corpus program from its frozen record + a live (exit, stdout).
    PURE: no subprocess. The lane selects the check vs eval table."""
    lane = record["lane"]
    if lane in ("check", "reject"):
        bucket, detail = classify_check_program(record, exit_code, stdout)
    elif lane == "eval":
        bucket, detail = classify_eval_program(record, exit_code, stdout)
    else:
        raise ValueError(f"unknown lane: {lane!r}")
    return ProgramResult(
        id=record["id"],
        lane=lane,
        bucket=bucket,
        detail=detail,
        program_path=record.get("program_path", ""),
    )


# ============================================================================
# TABULATION + ACCEPTANCE ORACLE.
# ============================================================================
@dataclass
class CorpusResult:
    check: dict[str, int] = field(default_factory=lambda: {b: 0 for b in CHECK_BUCKETS})
    eval: dict[str, int] = field(default_factory=lambda: {b: 0 for b in EVAL_BUCKETS})
    unsound_programs: list[ProgramResult] = field(default_factory=list)
    disagree_programs: list[ProgramResult] = field(default_factory=list)
    too_conservative_programs: list[ProgramResult] = field(default_factory=list)
    eval_disagree_programs: list[ProgramResult] = field(default_factory=list)
    crash_programs: list[ProgramResult] = field(default_factory=list)

    @property
    def check_total(self) -> int:
        return sum(self.check.values())

    @property
    def eval_total(self) -> int:
        return sum(self.eval.values())


def classify_corpus(results: list[ProgramResult]) -> CorpusResult:
    """Aggregate classified programs into bucket tallies + findings lists. PURE
    over the already-classified ProgramResult list."""
    out = CorpusResult()
    for r in results:
        if r.lane in ("check", "reject"):
            out.check[r.bucket] += 1
            if r.bucket == "unsound":
                out.unsound_programs.append(r)
            elif r.bucket == "disagree":
                out.disagree_programs.append(r)
            elif r.bucket == "too_conservative":
                out.too_conservative_programs.append(r)
            elif r.bucket == "crash":
                out.crash_programs.append(r)
        elif r.lane == "eval":
            out.eval[r.bucket] += 1
            if r.bucket == "disagree":
                out.eval_disagree_programs.append(r)
            elif r.bucket == "compiler_crash":
                out.crash_programs.append(r)
    return out


def tabulate(result: CorpusResult) -> str:
    """Render the outcome distribution, reusing Hull's report shape."""
    lines: list[str] = []
    lines.append("Hull Phase 5b conformance gate result")
    lines.append(f"  check programs:   {result.check_total}")
    lines.append(f"  eval programs:    {result.eval_total}")
    lines.append("  check outcomes:")
    for b in CHECK_BUCKETS:
        lines.append(f"    {b:<18} {result.check[b]}")
    lines.append("  eval outcomes:")
    for b in EVAL_BUCKETS:
        lines.append(f"    {b:<18} {result.eval[b]}")
    lines.append(f"  CompilerUnsound:  {result.check['unsound']}")
    lines.append(f"  Disagree (check): {result.check['disagree']}")
    lines.append(f"  TooConservative:  {result.check['too_conservative']}")
    lines.append(f"  EvalDisagree:     {result.eval['disagree']}")
    lines.append(f"  EvalAgree:        {result.eval['agree']}")
    return "\n".join(lines)


class AcceptanceOracleError(AssertionError):
    """Raised when the acceptance oracle is violated."""


def assert_acceptance_oracle(
    result: CorpusResult,
    eval_agree_floor: int,
    disagree_allowlist: list[str] | None = None,
    strict: bool = False,
) -> None:
    """The acceptance oracle, mirroring assert_acceptance_oracle (Hull). Raises on:
      - any CompilerUnsound (the release blocker, never allowed),
      - any UNEXPLAINED Disagree (allowlist empty by default, keyed by id),
      - any EvalDisagree,
      - EvalAgree below the floor,
      - any check/eval crash (a hung/garbled compiler is a finding, not a pass),
      - in --strict mode, any CompilerTooConservative (sound but Hull-incomplete;
        a WARNING by default).
    PURE over the aggregated result; no I/O."""
    allow = set(disagree_allowlist or [])
    if result.check["unsound"] != 0:
        progs = [(p.id, p.program_path, p.detail) for p in result.unsound_programs]
        raise AcceptanceOracleError(
            f"GATE FAILED: {result.check['unsound']} CompilerUnsound finding(s): {progs}"
        )
    unexplained = [p for p in result.disagree_programs if p.id not in allow]
    if unexplained:
        progs = [(p.id, p.program_path, p.detail) for p in unexplained]
        raise AcceptanceOracleError(
            f"GATE FAILED: {len(unexplained)} unexplained Disagree finding(s): {progs}"
        )
    if result.eval["disagree"] != 0:
        progs = [(p.id, p.program_path, p.detail) for p in result.eval_disagree_programs]
        raise AcceptanceOracleError(
            f"GATE FAILED: {result.eval['disagree']} EvalDisagree finding(s): {progs}"
        )
    crash_count = result.check["crash"] + result.eval["compiler_crash"]
    if crash_count != 0:
        progs = [(p.id, p.program_path, p.detail) for p in result.crash_programs]
        raise AcceptanceOracleError(
            f"GATE FAILED: {crash_count} CompilerCrash finding(s): {progs}"
        )
    if result.eval["agree"] < eval_agree_floor:
        raise AcceptanceOracleError(
            f"GATE FAILED: EvalAgree {result.eval['agree']} < floor {eval_agree_floor}"
        )
    if strict and result.check["too_conservative"] != 0:
        progs = [(p.id, p.program_path, p.detail) for p in result.too_conservative_programs]
        raise AcceptanceOracleError(
            f"GATE FAILED (--strict): {result.check['too_conservative']} "
            f"CompilerTooConservative finding(s): {progs}"
        )


# ============================================================================
# THE SUBPROCESS-DRIVING PATH (impure; exercised by the corpus run).
# ============================================================================
def run_program(
    chelis_bin: str,
    record: dict,
    timeout: float,
    simulate_unsound: str | None = None,
) -> ProgramResult:
    """Run ONE corpus program against the live binary and classify it. A
    hung/killed subprocess surfaces as a crash on that program, never a silent
    hang (per-call timeout). check/reject lanes use `chelis check
    --show-inferred --allow-style-violations`; the eval lane uses `chelis eval
    --file --json --allow-style-violations`. JSON is parsed from STDOUT (the
    style-gate warning goes to STDERR).

    --simulate-unsound <id> END-TO-END TEETH PROOF (design point D): when this
    record is the simulated id, the live binary is NOT run; instead a SYNTHETIC
    forced-accept JSON (a plausible function type) is fed to the classifier,
    end-to-end-proving the FULL runner (parse + classify + oracle) FAILS on a
    soundness regression -- the e2e analogue of the unit-level
    test_injected_unsound_fails."""
    lane = record["lane"]
    if simulate_unsound is not None and record["id"] == simulate_unsound:
        forced_accept = json.dumps(
            {
                "errors": [],
                "inferred_signatures": [
                    {
                        "display_signature_structured": {
                            "kind": "fn",
                            "args": [{"kind": "prim", "name": "f32"}],
                            "ret": {"kind": "prim", "name": "f32"},
                        },
                        "effect_row": [],
                    }
                ],
            }
        )
        return classify_program(record, 0, forced_accept)
    program_path = str(PROGRAMS_DIR / record["program_path"])
    if lane == "eval":
        cmd = [chelis_bin, "eval", "--file", program_path, "--json", "--allow-style-violations"]
    else:
        cmd = [chelis_bin, "check", program_path, "--show-inferred", "--allow-style-violations"]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL
        )
        return classify_program(record, proc.returncode, proc.stdout)
    except subprocess.TimeoutExpired:
        return ProgramResult(
            id=record["id"], lane=lane, bucket=("compiler_crash" if lane == "eval" else "crash"),
            detail=f"timeout after {timeout}s", program_path=record.get("program_path", ""),
        )


def run_corpus(
    chelis_bin: str,
    records: list[dict],
    timeout: float,
    max_workers: int | None,
    simulate_unsound: str | None = None,
) -> CorpusResult:
    """Fan the per-program subprocess calls across cores with a per-call timeout.
    The only semantic content (the verdict) is frozen in the corpus, so
    Python-calls-compiler is fine here (no Hull process, no process_run
    boundary)."""
    workers = max_workers or max(2, (os.cpu_count() or 4))
    results: list[ProgramResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(run_program, chelis_bin, rec, timeout, simulate_unsound)
            for rec in records
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())
    return classify_corpus(results)


# ============================================================================
# CORPUS LOADING + PROVENANCE.
# ============================================================================
def load_manifest() -> dict:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_verdicts() -> list[dict]:
    records: list[dict] = []
    with open(VERDICTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def assert_version_pinned(manifest: dict, live_version: str, allow_skew: bool) -> None:
    """A corpus pinned to a different compiler is a stale-corpus error, not a
    pass -- unless --allow-version-skew (the bump PR)."""
    pinned = manifest.get("chelis_version_pinned")
    live = live_version.strip().split()[-1] if live_version else ""
    if pinned != live and not allow_skew:
        raise AcceptanceOracleError(
            f"STALE CORPUS: manifest pins chelis {pinned!r} but live binary is "
            f"{live!r}. Refresh the corpus or pass --allow-version-skew."
        )


def live_chelis_version(chelis_bin: str) -> str:
    proc = subprocess.run(
        [chelis_bin, "--version"], capture_output=True, text=True, timeout=30
    )
    return proc.stdout.strip()


def select_slice(records: list[dict], check_n: int | None, eval_n: int | None) -> list[dict]:
    """Select a deterministic fixed slice by lane: the first check_n check/reject
    records and the first eval_n eval records (the corpus is ordered, so the same
    pinned slice is drawn each run -- a PR's pass/fail is reproducible)."""
    check_recs = [r for r in records if r["lane"] in ("check", "reject")]
    eval_recs = [r for r in records if r["lane"] == "eval"]
    if check_n is not None:
        check_recs = check_recs[:check_n]
    if eval_n is not None:
        eval_recs = eval_recs[:eval_n]
    return check_recs + eval_recs


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Hull Phase 5b conformance gate")
    parser.add_argument("--chelis-bin", required=True, help="ABSOLUTE path to the chelis binary")
    parser.add_argument("--check-n", type=int, default=None, help="cap check/reject programs")
    parser.add_argument("--eval-n", type=int, default=None, help="cap eval programs")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-program subprocess timeout (s)")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="promote CompilerTooConservative to a failure")
    parser.add_argument("--allow-version-skew", action="store_true")
    parser.add_argument("--eval-agree-floor", type=int, default=None)
    parser.add_argument(
        "--simulate-unsound", default=None,
        help="end-to-end teeth proof: force a synthetic accept for this record id "
        "(a reject sentinel) so the FULL runner + oracle must FAIL",
    )
    args = parser.parse_args(argv[1:])

    if not os.path.isabs(args.chelis_bin):
        sys.stderr.write(f"--chelis-bin must be an ABSOLUTE path, got {args.chelis_bin!r}\n")
        return 2

    manifest = load_manifest()
    try:
        assert_version_pinned(manifest, live_chelis_version(args.chelis_bin), args.allow_version_skew)
    except AcceptanceOracleError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return 1

    records = select_slice(load_verdicts(), args.check_n, args.eval_n)
    result = run_corpus(
        args.chelis_bin, records, args.timeout, args.max_workers,
        simulate_unsound=args.simulate_unsound,
    )
    print(tabulate(result))

    eval_floor = (
        args.eval_agree_floor
        if args.eval_agree_floor is not None
        else max(1, result.eval_total - result.eval["ref_not_value"])
    )
    try:
        assert_acceptance_oracle(
            result, eval_floor, disagree_allowlist=[], strict=args.strict
        )
    except AcceptanceOracleError as exc:
        sys.stderr.write(f"\n{exc}\n")
        return 1
    # A CompilerTooConservative is sound but Hull-incomplete: a WARNING by default.
    if result.check["too_conservative"]:
        for p in result.too_conservative_programs:
            sys.stderr.write(
                f"WARNING: CompilerTooConservative {p.id} ({p.program_path}): {p.detail}\n"
            )
    print("\nACCEPTANCE ORACLE: PASS (zero CompilerUnsound, zero unexplained Disagree, zero EvalDisagree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
