#!/usr/bin/env python3
"""Authoritative chelis#732 Phase 3 / chelis#687 agreement oracle.

The oracle binds the machine tolerance table, its numbered-spec rendering,
the shared comparator, and all three current consumers into one executable
acceptance command. It rejects silent corpus shrinkage before running:

* `parity.rs` must use the exact branch of the shared comparator;
* `eval_agreement.rs` must contain no f64 parse/epsilon fallback and must
  mark the current evaluator as [04-NUM-8]-nonconforming under chelis#897;
* the rejected-cell corpus must use the same exact comparator entrypoint;
* every required executable row must retain its designated call, execution
  mode and checked-result role in the standing Rust body audit;
* both of `eval_agreement.rs`'s comparison legs are guarded by a behavioral
  canary that runs the shipped helper, because the checks in this file can
  only see that the comparator is NAMED, never that it is INVOKED: a shared
  assertion helper neutered into a no-op deletes its leg from every row at
  once and edits no protected definition (chelis#1104);
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
        "parity_checked_reshape",
        "parity_checked_sparse_axes",
        "parity_checked_window_geometry",
        "parity_dict_foundation",
        "parity_count_bool_axes",
        "parity_explicit_normalization",
        "parity_generic_explicit_shape",
        "parity_hash_order_determinism",
        "parity_constraint_directed_risk_guards_library_only",
        "parity_iter_foundation",
        "parity_list_foundation",
        "parity_literal_extent_claim",
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
        "parity_recursive_generic",
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
        "agreement_expected_value_reaches_comparator",
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
        "metal_rank2_gap_rejects_without_an_artifact",
        "runtime_rejected_cells_abort_with_their_pinned_diagnostics",
    },
}


def required_body_contract() -> dict:
    """Export reviewed call roles from the one required-identity inventory.

    Closed direct-result forms supplement call adoption, not runtime reach.
    Executable comparator/receipt controls and changed-test review remain.
    """
    canaries = {
        "agreement_operation_identity_is_derived_from_ir": "agreement_op_for_risc",
        "agreement_compiled_observation_reaches_comparator": "compare_lanes_with",
        "agreement_expected_value_reaches_comparator": "assert_expected",
        "agreement_width_nonconformance_is_behavioral": "compare_rendered_elements",
    }
    unary_rows = {
        "agreement_atan", "agreement_cos", "agreement_log", "agreement_sin",
        "agreement_sqrt_is_exact", "agreement_tan",
    }
    # These are reviewed library-only dispositions, not inferred from names:
    # hello_tensor and opaque_invariants_simplex retain old suffixes but execute.
    library_rows = {
        "parity_constraint_directed_risk_guards_library_only",
        "parity_induction_bond_library_only", "parity_linreg_library_only",
        "parity_mnist_library_only", "parity_opaque_invariants_library_only",
        "parity_rank_poly_borrow_library_only", "parity_transformer_block_library_only",
        "parity_vmap_relu_library_only",
    }
    if not library_rows <= REQUIRED_TESTS.get(PARITY_SOURCE, set()):
        raise ValueError("stale reviewed library-only body role")
    if not (set(canaries) | unary_rows) <= REQUIRED_TESTS.get(EVAL_AGREEMENT_SOURCE, set()):
        raise ValueError("stale reviewed evaluator body role")
    sources = []
    for path, names in REQUIRED_TESTS.items():
        tests = []
        for name in sorted(names):
            run_parity = None
            if path == PARITY_SOURCE:
                if name == "parity_corpus_is_complete":
                    calls = ["parity_corpus::validate"]
                elif name.startswith("parity_comparator_"):
                    calls = ["assert_parity"]
                else:
                    calls = ["drive_parity"]
                    run_parity = name not in library_rows
            elif path == EVAL_AGREEMENT_SOURCE:
                if name in canaries:
                    calls = [canaries[name], "record_phase3_receipt"]
                elif name in unary_rows:
                    calls = ["assert_unary_transcendental"]
                else:
                    calls = ["assert_agrees", "assert_expected"]
            elif path == REJECTED_SOURCE:
                calls = ["compare_exact_observations"]
            else:
                raise ValueError(f"no reviewed body-call role for {path}::{name}")
            result_kind = None
            if path == REJECTED_SOURCE or name == "parity_corpus_is_complete":
                result_kind = "success"
            elif path == PARITY_SOURCE and name.startswith("parity_comparator_"):
                result_kind = "assert_ok" if name in {
                    "parity_comparator_accepts_byte_identical_tensor_lines",
                    "parity_comparator_byte_equal_for_non_tensor",
                } else "assert_err"
            elif name in {"agreement_compiled_observation_reaches_comparator",
                          "agreement_width_nonconformance_is_behavioral"}:
                result_kind = "expect_err"
            elif name == "agreement_operation_identity_is_derived_from_ir":
                result_kind = "assert_eq"
            result = {"call": calls[0], "kind": result_kind} if result_kind else None
            tests.append({"name": name, "calls": calls, "run_parity": run_parity, "result": result})
        sources.append({"path": str(path), "tests": tests})
    return {"schema_version": 2, "sources": sources}


REQUIRED_EVAL_RECEIPTS = {
    "operation-identity-canary",
    "compiled-observation-canary",
    "expected-value-canary",
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
            "parity_corpus_contract",
            "--test",
            "phase3_body_contract",
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


TEST_DECLARATION = re.compile(
    r"(?P<attrs>(?:#\[[^\]]+\]\s*)+)fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)
RAW_STRING = re.compile(r'(?:br|cr|r)(?P<hashes>#{0,255})"')
CHAR_LITERAL = re.compile(r"(?:b)?'(?:\\(?:u\{[0-9A-Fa-f_]+\}|x[0-9A-Fa-f]{2}|[nrt\\0'\"])|[^'\\\n])'")


def rust_code_mask(source: str) -> str:
    """Blank comments/literals without changing source offsets or line numbers.

    This is lexical source discovery, not Rust compilation or cfg evaluation.
    Lifetimes and labels remain code; only complete character literals are blanked.
    """
    code = list(source)
    index = 0
    while index < len(source):
        start = index
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline
        elif source.startswith("/*", index):
            depth = 1
            index += 2
            while index < len(source) and depth:
                if source.startswith("/*", index):
                    depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError("unterminated Rust block comment")
        elif raw := RAW_STRING.match(source, index):
            delimiter = '"' + raw.group("hashes")
            end = source.find(delimiter, raw.end())
            if end == -1:
                raise ValueError("unterminated Rust raw string")
            index = end + len(delimiter)
        elif source[index] == '"' or source.startswith(('b"', 'c"'), index):
            index += 1 if source[index] == '"' else 2
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("unterminated Rust string")
        elif character := CHAR_LITERAL.match(source, index):
            index = character.end()
        else:
            index += 1
            continue
        for offset in range(start, index):
            if source[offset] != "\n":
                code[offset] = " "
    return "".join(code)


def test_declaration_matches(source: str) -> list[re.Match[str]]:
    return [match for match in TEST_DECLARATION.finditer(rust_code_mask(source))
            if "#[test]" in match.group("attrs")]


def test_declarations(source: str) -> dict[str, str]:
    return {match.group("name"): source[match.start("attrs"):match.end("attrs")].strip()
            for match in test_declaration_matches(source)}


def ignored_tests(source: str) -> dict[str, str]:
    tests: dict[str, str] = {}
    for name, attrs in test_declarations(source).items():
        if "ignore" not in attrs:
            continue
        reason = re.search(r'#\[ignore\s*=\s*"(?P<reason>[^"]*)"\]', attrs)
        tests[name] = reason.group("reason") if reason else ""
    return tests


def _matching_rust_brace(source: str, opening: int, *, masked: bool = False) -> int:
    """Return the closing brace for a Rust block, ignoring literal/comment braces."""
    code = source if masked else rust_code_mask(source)
    if code[opening] != "{":
        raise ValueError("opening offset does not point at a code brace")
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("unterminated Rust test body")


def test_definition_spans(source: str) -> dict[str, tuple[int, int, int, int]]:
    """Map each test to definition and body spans.

    Values are ``(definition_start, definition_end, body_start, body_end)``;
    end offsets are exclusive and index the original, unmodified source.
    """
    spans: dict[str, tuple[int, int, int, int]] = {}
    code = rust_code_mask(source)
    for match in TEST_DECLARATION.finditer(code):
        if "#[test]" not in match.group("attrs"):
            continue
        opening = code.find("{", match.end())
        if opening == -1:
            raise ValueError(f"test {match.group('name')} has no body")
        closing = _matching_rust_brace(code, opening, masked=True)
        spans[match.group("name")] = (match.start(), closing + 1, opening, closing + 1)
    return spans


def replace_test_body(source: str, name: str, replacement: str) -> str:
    """Test helper: replace one Rust test body while retaining its attributes/signature."""
    try:
        _, _, body_start, body_end = test_definition_spans(source)[name]
    except KeyError as error:
        raise ValueError(f"unknown Rust test {name}") from error
    return source[:body_start] + replacement + source[body_end:]


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
            "chelis_string_from_scalar",
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


def preflight_violations() -> list[str]:
    """Return every static-contract or required-tool failure."""
    violations = (
        source_violations()
        + comparator_violations()
    )
    if shutil.which("cc") is None:
        violations.append("a host C compiler (`cc`) is required")
    if shutil.which("cargo") is None:
        violations.append("cargo is required")
    return violations


def main() -> int:
    violations = preflight_violations()
    if violations:
        for violation in violations:
            print(f"PHASE 3 ORACLE: FAIL: {violation}", file=sys.stderr)
        return 1

    for index, (label, command) in enumerate(SUITE_COMMANDS):
        if index != len(SUITE_COMMANDS) - 1:
            if not run_command(label, command):
                return 1
            continue

        with tempfile.TemporaryDirectory(prefix="chelis-phase3-receipts-") as temp_dir:
            receipt_path = Path(temp_dir) / "eval-agreement.tsv"
            env = os.environ.copy()
            env["CHELIS_PHASE3_RECEIPT_PATH"] = str(receipt_path)
            if not run_command(label, command, env=env):
                return 1
            receipt_text = (
                receipt_path.read_text(encoding="utf-8")
                if receipt_path.exists()
                else ""
            )
            violations = receipt_violations(receipt_text)
            if violations:
                for violation in violations:
                    print(f"PHASE 3 ORACLE: FAIL: {violation}", file=sys.stderr)
                return 1

    print("\nPHASE 3 ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
