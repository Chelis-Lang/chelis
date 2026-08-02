#!/usr/bin/env python3
"""Authoritative chelis#732 Phase 3 / chelis#687 agreement oracle.

The oracle binds the machine tolerance table, its numbered-spec rendering,
the shared comparator, and all three current consumers into one executable
acceptance command. It rejects silent corpus shrinkage before running:

* `parity.rs` must use the exact branch of the shared comparator;
* `eval_agreement.rs` must contain no f64 parse/epsilon fallback and must
  mark the current evaluator as [04-NUM-8]-nonconforming under chelis#897;
* the rejected-cell corpus must use the same exact comparator entrypoint;
* the only ignored test in these suites is the declared system-CBLAS
  prerequisite. A value-divergence ignore must be added to this executable
  ledger and to the owning design status rather than silently appearing.

Usage:

    .venv/bin/python scripts/faithful_observation_phase3_oracle.py

Acceptance is exit 0 with the final line ``PHASE 3 ORACLE: PASS``. A host C
toolchain and cargo-nextest are required. Set ``CARGO_TARGET_DIR`` when another
agent or session may build concurrently.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
PARITY_SOURCE = Path("crates/chelis-cli/tests/parity.rs")
EVAL_AGREEMENT_SOURCE = Path("crates/chelis-e2e/tests/eval_agreement.rs")
REJECTED_SOURCE = Path("crates/chelis-cli/tests/issue_687_rejected_cells_corpus.rs")

PHASE3_SOURCES = (PARITY_SOURCE, EVAL_AGREEMENT_SOURCE, REJECTED_SOURCE)

ALLOWED_IGNORES = {
    "parity_transformer_block_library_only": (
        "requires system cblas.h (install openblas-devel / libopenblas-dev)"
    ),
}

REQUIRED_TESTS = {
    PARITY_SOURCE: {
        "parity_dict_foundation",
        "parity_constraint_directed_risk_guards_library_only",
        "parity_iter_foundation",
        "parity_list_foundation",
        "parity_scalar_string_foundation",
        "parity_tensor_structural_ops",
        "parity_hello_tensor_library_only",
        "parity_induction_bond_library_only",
        "parity_linreg_library_only",
        "parity_mnist_library_only",
        "parity_transformer_block_library_only",
        "parity_vmap_relu_library_only",
        "parity_opaque_invariants_library_only",
        "parity_opaque_invariants_simplex_library_only",
        "parity_rank_poly_borrow_library_only",
        "parity_corpus_is_complete",
        "parity_comparator_accepts_byte_identical_tensor_lines",
        "parity_comparator_reports_sub_tolerance_float_drift",
        "parity_comparator_rejects_value_divergence",
        "parity_comparator_byte_equal_for_non_tensor",
        "parity_comparator_rejects_non_tensor_diff",
    },
    EVAL_AGREEMENT_SOURCE: {
        "agreement_operation_identity_is_derived_from_ir",
        "agreement_compiled_observation_reaches_comparator",
        "agreement_width_nonconformance_is_behavioral",
        "agreement_add",
        "agreement_mul",
        "agreement_neg",
        "agreement_relu",
        "agreement_exp",
        "agreement_log",
        "agreement_sin",
        "agreement_sqrt_is_exact",
        "agreement_cos",
        "agreement_tan",
        "agreement_atan",
        "agreement_bf16_add",
        "agreement_f16_add",
        "agreement_bf16_reduce_sum_matches_eval_exactly",
    },
    REJECTED_SOURCE: {
        "rejected_cells_fail_the_build_with_their_pinned_diagnostics",
        "metal_rank2_abort_stub_names_itself_in_the_emission",
        "runtime_rejected_cells_abort_with_their_pinned_diagnostics",
    },
}

REQUIRED_EVAL_RECEIPTS = {
    "operation-identity-canary",
    "compiled-observation-canary",
    "width-nonconformance-canary",
    "add(3,4)",
    "mul(5,6)",
    "neg(7)",
    "relu(-2)",
    "relu(3)",
    "exp(0)",
    "log(1)",
    "sin(0)",
    "sqrt(4)",
    "cos(0)",
    "tan(0)",
    "atan(0)",
    "bf16 add(1.5, 2.5)",
    "f16 add(1.5, 2.5)",
    "bf16 reduce_sum(0.25 x 8)",
}

SUITE_COMMANDS = (
    (
        "agreement policy and spec tripwire",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-types",
            "--test",
            "agreement_tolerance",
        ),
    ),
    (
        "example parity and rejected-cell corpus",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-cli",
            "--test",
            "parity",
            "--test",
            "issue_687_rejected_cells_corpus",
        ),
    ),
    (
        "DAG evaluator versus compiled C agreement",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-e2e",
            "--test",
            "eval_agreement",
        ),
    ),
)


def shipped_sources() -> dict[Path, str]:
    return {
        path: (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in PHASE3_SOURCES
    }


def test_declarations(source: str) -> dict[str, str]:
    tests: dict[str, str] = {}
    pattern = re.compile(
        r"(?P<attrs>(?:#\[[^\]]+\]\s*)+)fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    for match in pattern.finditer(source):
        attrs = match.group("attrs")
        if "#[test]" not in attrs:
            continue
        tests[match.group("name")] = attrs.strip()
    return tests


def ignored_tests(source: str) -> dict[str, str]:
    tests: dict[str, str] = {}
    for name, attrs in test_declarations(source).items():
        if "ignore" not in attrs:
            continue
        reason = re.search(r'#\[ignore\s*=\s*"(?P<reason>[^"]*)"\]', attrs)
        tests[name] = reason.group("reason") if reason else ""
    return tests


def source_violations(sources: Mapping[Path, str] | None = None) -> list[str]:
    sources = shipped_sources() if sources is None else sources
    observed: dict[str, str] = {}
    declarations: dict[Path, dict[str, str]] = {}
    for path in PHASE3_SOURCES:
        declarations[path] = test_declarations(sources[path])
        for name, reason in ignored_tests(sources[path]).items():
            if name in observed:
                return [f"duplicate ignored Phase 3 test name: {name}"]
            observed[name] = reason

    violations: list[str] = []
    for path, required in REQUIRED_TESTS.items():
        for name in sorted(required - set(declarations[path])):
            violations.append(f"{path} is missing required Phase 3 corpus row {name}")
    for path, tests in declarations.items():
        for name, attrs in tests.items():
            expected = "#[test]"
            if name in ALLOWED_IGNORES:
                expected += f'\n#[ignore = "{ALLOWED_IGNORES[name]}"]'
            if attrs != expected:
                violations.append(
                    f"{path} test {name} has conditional or undeclared attributes: {attrs!r}"
                )
    for name in sorted(set(observed) - set(ALLOWED_IGNORES)):
        violations.append(
            f"undeclared ignored Phase 3 test {name}: value divergences must be "
            "issue-linked and added to the authoritative ledger"
        )
    for name in sorted(set(ALLOWED_IGNORES) - set(observed)):
        violations.append(f"stale Phase 3 ignore ledger row: {name}")
    for name in sorted(set(observed) & set(ALLOWED_IGNORES)):
        if observed[name] != ALLOWED_IGNORES[name]:
            violations.append(
                f"{name} ignore reason drifted: expected "
                f"{ALLOWED_IGNORES[name]!r}, got {observed[name]!r}"
            )
    return violations


def comparator_violations(sources: Mapping[Path, str] | None = None) -> list[str]:
    sources = shipped_sources() if sources is None else sources
    required = {
        PARITY_SOURCE: ("compare_exact_observations",),
        EVAL_AGREEMENT_SOURCE: (
            "compare_rendered_elements",
            "chelis_format_shortest",
            "agreement_op_for_risc",
            "agreement_compiled_observation_reaches_comparator",
            "agreement_width_nonconformance_is_behavioral",
            "ArithmeticWidthStatus::Nonconforming { issue: 897 }",
        ),
        REJECTED_SOURCE: ("compare_exact_observations",),
    }
    violations: list[str] = []
    for path, needles in required.items():
        for needle in needles:
            if needle not in sources[path]:
                label = "chelis#897 width guard" if "Nonconforming" in needle else needle
                violations.append(f"{path} is missing shared comparator obligation {label}")

    forbidden = (
        "fn eval_last(dag: &Dag) -> f64",
        "fn parse_c_output(output: &str) -> f64",
        "fn assert_close(a: f64, b: f64, tol: f64",
        'printf("%.6f"',
        'printf("%.8f',
    )
    eval_source = sources[EVAL_AGREEMENT_SOURCE]
    for needle in forbidden:
        if needle in eval_source:
            violations.append(
                f"{EVAL_AGREEMENT_SOURCE} restored forbidden f64/tolerance path {needle!r}"
            )
    return violations


def receipt_violations(receipt_text: str) -> list[str]:
    observed: list[str] = []
    malformed: list[str] = []
    for line in receipt_text.splitlines():
        case, separator, detail = line.partition("\t")
        if not separator or not case or not detail:
            malformed.append(line)
            continue
        observed.append(case)

    violations = [f"malformed Phase 3 runtime receipt: {line!r}" for line in malformed]
    observed_set = set(observed)
    for case in sorted(REQUIRED_EVAL_RECEIPTS - observed_set):
        violations.append(f"missing Phase 3 runtime receipt: {case}")
    for case in sorted(observed_set - REQUIRED_EVAL_RECEIPTS):
        violations.append(f"undeclared Phase 3 runtime receipt: {case}")
    for case in sorted({case for case in observed if observed.count(case) != 1}):
        violations.append(f"Phase 3 runtime receipt must occur exactly once: {case}")
    return violations


def run_command(
    label: str,
    command: tuple[str, ...],
    *,
    env: Mapping[str, str] | None = None,
) -> bool:
    print(f"\n== {label} ==", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False, env=env)
    if completed.returncode != 0:
        print(
            f"PHASE 3 ORACLE: FAIL ({label} exited {completed.returncode})",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    violations = source_violations() + comparator_violations()
    if violations:
        for violation in violations:
            print(f"PHASE 3 ORACLE: FAIL: {violation}", file=sys.stderr)
        return 1

    if shutil.which("cc") is None:
        print("PHASE 3 ORACLE: FAIL: a host C compiler (`cc`) is required", file=sys.stderr)
        return 1
    if shutil.which("cargo") is None:
        print("PHASE 3 ORACLE: FAIL: cargo is required", file=sys.stderr)
        return 1

    for label, command in SUITE_COMMANDS[:-1]:
        if not run_command(label, command):
            return 1

    label, command = SUITE_COMMANDS[-1]
    with tempfile.TemporaryDirectory(prefix="chelis-phase3-receipts-") as temp_dir:
        receipt_path = Path(temp_dir) / "eval-agreement.tsv"
        env = os.environ.copy()
        env["CHELIS_PHASE3_RECEIPT_PATH"] = str(receipt_path)
        if not run_command(label, command, env=env):
            return 1
        receipt_text = receipt_path.read_text(encoding="utf-8") if receipt_path.exists() else ""
        violations = receipt_violations(receipt_text)
        if violations:
            for violation in violations:
                print(f"PHASE 3 ORACLE: FAIL: {violation}", file=sys.stderr)
            return 1

    print("\nPHASE 3 ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
