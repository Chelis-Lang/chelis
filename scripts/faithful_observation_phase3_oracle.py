#!/usr/bin/env python3
"""Authoritative chelis#732 Phase 3 / chelis#687 agreement oracle.

The oracle binds the machine tolerance table, its numbered-spec rendering,
the shared comparator, and all three current consumers into one executable
acceptance command. It rejects silent corpus shrinkage before running:

* `parity.rs` must use the exact branch of the shared comparator;
* `eval_agreement.rs` must contain no f64 parse/epsilon fallback and must
  mark the current evaluator as [04-NUM-8]-nonconforming under chelis#897;
* the rejected-cell corpus must use the same exact comparator entrypoint;
* every frozen executable row must retain its independently reviewed
  definition digest, so producer-authored receipts cannot replace behavior;
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

import hashlib
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

# Guard artifact: exact definitions for every frozen Phase 3 executable row.
# A digest changes only when the owning design's corpus is intentionally revised
# and the replacement behavior has independent review evidence. Editing this map
# merely to accept a changed test is not a repair.
REQUIRED_TEST_DEFINITION_SHA256: dict[Path, dict[str, str]] = {
    PARITY_SOURCE: {
        "parity_comparator_accepts_byte_identical_tensor_lines": "9224411d844dc758300d5880b424edd89deea5eb9dfa9d12a34ba7257a58e38f",
        "parity_comparator_byte_equal_for_non_tensor": "06e91c12393496a46c96a53e0e6f65cf8840b74615c09a6cd4b29511e4a7d8eb",
        "parity_comparator_rejects_non_tensor_diff": "22b39df165d7350e80d467ca484cdb4d85407aa6645ebebe1ba0d35e721a88a5",
        "parity_comparator_rejects_value_divergence": "06b51ffeb117c26b55c7901d64d3525a22855bec70f20246d263d9b5fc041f89",
        "parity_comparator_reports_sub_tolerance_float_drift": "40d029638fe1b70c1611adab72eeb31c1befed97d74c5400f5aae82f8c86aafe",
        "parity_constraint_directed_risk_guards_library_only": "ac6933d790a89ff00d7658e1260d61614ccc2547d9a91a67a0aa98918e33ca32",
        "parity_corpus_is_complete": "af5a2013e2b241bf9dbb51764f907a1e6426fbf31f6ed4db1a85367de5a29116",
        "parity_dict_foundation": "1bfd21bf0d78c9f36869908852a963037e0f13e36d5f9bc73b77131ff9d2970f",
        "parity_hello_tensor_library_only": "c3291d95cd21db6230fdc7ec47627d294b95a47e51efe4b97d4cf5f1080bd86b",
        "parity_induction_bond_library_only": "3e83f0cf929583a8df5a8fa05c62e9a712826fbe899b244766fdcad47b741389",
        "parity_iter_foundation": "c99d74a439e006c29748429c3877941460cd3ed18a0a98fd16caf81fb510c84f",
        "parity_linreg_library_only": "041271517605b7fa97a616c9fbe37d97d30bf0a6740c419e7dd91182736d97e4",
        "parity_list_foundation": "5400fe48566a947d9970a2231ea00b2d573abe32fcf3576b3ef7c8dfece02f14",
        "parity_mnist_library_only": "f46a10e016c52751f1072770cce71c39e8322d5ce8b19f1c81879d53fbc833f4",
        "parity_opaque_invariants_library_only": "f2a0340b7b1d509b2d06ad84eb11ff7f237015ee90555ea9c1c7609c1b3d25f5",
        "parity_opaque_invariants_simplex_library_only": "1a3bcf81033223eeca2cb93b272540965be43941ebc79e915664314f5f22b3cb",
        "parity_rank_poly_borrow_library_only": "4325ec047118bf72a0a4518d371dbaaa7597b861946a13aac1bd4e2c155f642c",
        "parity_scalar_string_foundation": "30ec444cbbea6be840b3c603121b3d4d35b1c8eec7311c6808377cf4b6883372",
        "parity_tensor_structural_ops": "02b454423039f3ab3b3c99005481cef1cb744fc9ee81af8848f3638d0e793286",
        "parity_transformer_block_library_only": "1219362b0fe28ff5efacbe52249f4c151c646fabdc6557fbb04de5b691bcd06d",
        "parity_vmap_relu_library_only": "0c3450ba322a3254bed1eb9b1474abf660661453af72c4b74d1084b68f0260b1",
    },
    EVAL_AGREEMENT_SOURCE: {
        "agreement_add": "aa21bd3c8867d8703241ccec34d42e5fc4c38cd86b331d9cc7dfac7dd051645a",
        "agreement_atan": "80c1d1edd7d7857c6392470b80ec650e7db64c0d790781746ed92520c4ed9b85",
        "agreement_bf16_add": "681477e6d9c0a38477f4649d165361fee6709066c14d7e583f22ba98eb8c9f16",
        "agreement_bf16_reduce_sum_matches_eval_exactly": "24ef73d63f4c561841b70164b0e8ff0c5db7f54047d25419eebe37602eb31403",
        "agreement_compiled_observation_reaches_comparator": "343b1887715aa2566c09c40d0a8f7897e3bf71e4b5b999bb64d823fb4d79d0f0",
        "agreement_cos": "bb39c151b0be1d95a7c7e75c4e949e6e4c1da4c335f7abe119e088fa5d866d04",
        "agreement_exp": "3d982a6201d599802368ba9e8919dd5d27a1807fd66ce9beec77f4f1e027482f",
        "agreement_f16_add": "76e15994659af4bd3643e57276ea9f21005135ee26a1772264ceb33debea9b3e",
        "agreement_log": "84ca6c7ba3b27a0718b42f40d15dd1c786648e33562b63a9ecddfaa02416c709",
        "agreement_mul": "99d1f365f623572885e27af44137c37da1a6fd3e34c4df7a01a121c2ed7dce7b",
        "agreement_neg": "3445ff9b0d11d539bd03aa7c76b888a30929705be701d9e7b5e0058800eec364",
        "agreement_operation_identity_is_derived_from_ir": "b35dd2f9eac4362f5c38639c8a882b38872d5400d2399d8963cfe2427340dec6",
        "agreement_relu": "ff45e597ad9d21fe2df350a937bea44cded67f5c3348271be368db77bb8f6679",
        "agreement_sin": "7d1c26bc002402b089c6c035eb756dedd70de05f06c2253ad362462d8243a170",
        "agreement_sqrt_is_exact": "63ee422b92eef85a5635892c57282dbd9cec0154bd3d79ae4c57ca1744f0ac6a",
        "agreement_tan": "33480e20e37c50cbd1ba8ae7864b77f860831c1ecce8577dcf22b28640e0116e",
        "agreement_width_nonconformance_is_behavioral": "8b99a54287fe3517ab80544f77a25eadda35a9d84ec85f0687e15fa4910feb86",
    },
    REJECTED_SOURCE: {
        "metal_rank2_abort_stub_names_itself_in_the_emission": "24edf2a745fc9665b0a2d9497040752cecf77cfdac153990d4c7dcfe0ed81f74",
        "rejected_cells_fail_the_build_with_their_pinned_diagnostics": "c1751e1db4ab78e882ed73710c3a4367c724fb223f718ccb800430c40a1ebd12",
        "runtime_rejected_cells_abort_with_their_pinned_diagnostics": "d37afb668293c790033ab1f469f7535ff780a19a4c68cc1e5b88092013ec2a60",
    },
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


def _matching_rust_brace(source: str, opening: int) -> int:
    """Return the closing brace for a Rust block, ignoring literal/comment braces."""
    if source[opening] != "{":
        raise ValueError("opening offset does not point at a brace")
    depth = 0
    index = opening
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline == -1 else newline + 1
            continue
        if source.startswith("/*", index):
            comment_depth = 1
            index += 2
            while index < len(source) and comment_depth:
                if source.startswith("/*", index):
                    comment_depth += 1
                    index += 2
                elif source.startswith("*/", index):
                    comment_depth -= 1
                    index += 2
                else:
                    index += 1
            if comment_depth:
                raise ValueError("unterminated Rust block comment")
            continue

        raw = re.match(r'(?:br|r)(?P<hashes>#{0,255})"', source[index:])
        if raw:
            delimiter = '"' + raw.group("hashes")
            end = source.find(delimiter, index + raw.end())
            if end == -1:
                raise ValueError("unterminated Rust raw string")
            index = end + len(delimiter)
            continue

        quote_offset = 1 if source.startswith(('b"', "b'"), index) else 0
        quote = source[index + quote_offset] if index + quote_offset < len(source) else ""
        if quote == '"':
            index += quote_offset + 1
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
            continue
        if quote == "'":
            # Treat a quote as a character literal only when it closes before a
            # newline; lifetime syntax contains no braces and can be skipped as text.
            cursor = index + quote_offset + 1
            escaped = False
            closing = -1
            while cursor < len(source) and source[cursor] != "\n":
                if not escaped and source[cursor] == "'":
                    closing = cursor
                    break
                if not escaped and source[cursor] == "\\":
                    escaped = True
                else:
                    escaped = False
                cursor += 1
            if closing != -1:
                index = closing + 1
                continue

        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("unterminated Rust test body")


def test_definition_spans(source: str) -> dict[str, tuple[int, int, int, int]]:
    """Map each test to definition and body spans.

    Values are ``(definition_start, definition_end, body_start, body_end)``;
    end offsets are exclusive.
    """
    spans: dict[str, tuple[int, int, int, int]] = {}
    pattern = re.compile(
        r"(?P<attrs>(?:#\[[^\]]+\]\s*)+)fn\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\(",
        re.MULTILINE,
    )
    for match in pattern.finditer(source):
        if "#[test]" not in match.group("attrs"):
            continue
        opening = source.find("{", match.end())
        if opening == -1:
            raise ValueError(f"test {match.group('name')} has no body")
        closing = _matching_rust_brace(source, opening)
        spans[match.group("name")] = (match.start(), closing + 1, opening, closing + 1)
    return spans


def replace_test_body(source: str, name: str, replacement: str) -> str:
    """Test helper: replace one Rust test body while retaining its attributes/signature."""
    try:
        _, _, body_start, body_end = test_definition_spans(source)[name]
    except KeyError as error:
        raise ValueError(f"unknown Rust test {name}") from error
    return source[:body_start] + replacement + source[body_end:]


def definition_digest_violations(
    sources: Mapping[Path, str] | None = None,
) -> list[str]:
    sources = shipped_sources() if sources is None else sources
    violations: list[str] = []
    for path, required in REQUIRED_TESTS.items():
        spans = test_definition_spans(sources[path])
        expected_digests = REQUIRED_TEST_DEFINITION_SHA256.get(path, {})
        for name in sorted(required):
            if name not in spans:
                continue
            start, end, _, _ = spans[name]
            observed = hashlib.sha256(sources[path][start:end].encode()).hexdigest()
            expected = expected_digests.get(name)
            if expected != observed:
                violations.append(
                    f"{path} test {name} changed its frozen Phase 3 definition "
                    f"(expected {expected or 'no digest'}, got {observed}); restore the "
                    "reviewed behavior or revise the guard with independent evidence"
                )
    return violations


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
    violations = (
        source_violations()
        + comparator_violations()
        + definition_digest_violations()
    )
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
