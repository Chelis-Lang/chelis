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
* both of `eval_agreement.rs`'s comparison legs are guarded by a behavioral
  canary that runs the shipped helper, because the checks in this file can
  only see that the comparator is NAMED, never that it is INVOKED: a shared
  assertion helper neutered into a no-op deletes its leg from every row at
  once and edits no frozen definition (chelis#1104);
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

# Guard artifact: exact definitions for every frozen Phase 3 executable row.
# PR #1692 adds checked_sparse_axes with independent stored-bit/index/shape
# controls. The corpus guard changes only by adding its executable filename;
# deletion, empty-body, library-only, and corpus-removal mutations remain red.
# A digest changes only when the owning design's corpus is intentionally revised
# and the replacement behavior has independent review evidence. Editing this map
# merely to accept a changed test is not a repair.
# chelis#1158 added parity_recursive_generic to the frozen corpus. The 22-case
# recursive_generic_monomorphization suite supplies its independent evidence.
# chelis#1287 adds parity_count_bool_axes and count_bool_axes.ch together; the
# executable parity row itself plus the dedicated Count oracle are the
# independent evidence for extending this guard.
# chelis#1338 adds hash_order_determinism.ch and its eval/C parity row together.
# The Phase A oracle invokes that exact row and the 24-process acceptance and
# rejection matrix independently, so extending the frozen corpus cannot replace
# the behavior evidence that justified it.
# chelis#1247/#1258 add parity_kinded_nominal_dimensions and its executable
# example. The dedicated issue_1247_integer_type_application suite supplies
# independent check/test/Surf/eval/C-backend evidence for the corpus change.
# chelis#1621 adds generic_explicit_shape.ch and its executable parity row.
# The independently reviewed issue_1621_scalar_surface_cli suite pins the
# migration example's exact eval/C output and rejects implicit scalar mixing.
# chelis#889/#893 add checked_reshape.ch and its executable parity row.
# Independent runtime and generated host/DAG UBSan controls cover exact stored
# bits, empty domains, metadata rejection, and owned reshape independence.
# chelis#1294 adds explicit_normalization.ch and its executable parity row.
# The independently reviewed issue_1294_normalize suite pins its exact eval
# result and the undeclared-name rejection; the parity row also runs generated C.

# chelis#1377 adds literal_extent_claim.ch and its executable parity row.
# runtime_extent_claim_preparation independently asserts the example's declared
# signature, exact shape/value, and the required failure with five input elements.
# Its 40-call matrix separately exercises export, binding, and inlined roots.
REQUIRED_TEST_DEFINITION_SHA256: dict[Path, dict[str, str]] = {
    PARITY_SOURCE: {
        "parity_checked_reshape": "3f2defb3802726dac732a24ee5a9815433c8a16679f60a8ae337268305119424",
        "parity_comparator_accepts_byte_identical_tensor_lines": "9224411d844dc758300d5880b424edd89deea5eb9dfa9d12a34ba7257a58e38f",
        "parity_comparator_byte_equal_for_non_tensor": "06e91c12393496a46c96a53e0e6f65cf8840b74615c09a6cd4b29511e4a7d8eb",
        "parity_comparator_rejects_non_tensor_diff": "22b39df165d7350e80d467ca484cdb4d85407aa6645ebebe1ba0d35e721a88a5",
        "parity_comparator_rejects_value_divergence": "06b51ffeb117c26b55c7901d64d3525a22855bec70f20246d263d9b5fc041f89",
        "parity_comparator_reports_sub_tolerance_float_drift": "40d029638fe1b70c1611adab72eeb31c1befed97d74c5400f5aae82f8c86aafe",
        "parity_constraint_directed_risk_guards_library_only": "ac6933d790a89ff00d7658e1260d61614ccc2547d9a91a67a0aa98918e33ca32",
        "parity_count_bool_axes": "66e82bb4aeedafabc5d77eefeec25cb2728085becf2fdf44c335631fde750ba9",
        "parity_checked_sparse_axes": "69b1b926e2294ef2dcf704f68d218f0c692788c943aa3f6a88ed06e331e107b0",
        "parity_checked_window_geometry": "9df8502bf07ccbbd5731c79596d328130cddedede4217f7af585063404cc64dc",
        "parity_corpus_is_complete": "6603a110d51e7c5629d9d6e1026f2ca6ecfa4e05d565aa2df32caa91180e41ee",
        "parity_explicit_normalization": "d09c17ffa744ee21214877d59476ce58441480e4f6f8a29d6eb5edf3ad1417cb",
        "parity_dict_foundation": "1bfd21bf0d78c9f36869908852a963037e0f13e36d5f9bc73b77131ff9d2970f",
        "parity_generic_explicit_shape": "72ebff1fb9ca21ef52e6622c724f90e7f24f9054c8bff0b5c582be8582007f73",
        "parity_hash_order_determinism": "148c637280b238c9a119e22196960703873e291f1f0df59323c33ddb96b47170",
        # chelis#912 applies [05-OBS-7] uniformly: hello_tensor's pure
        # nullary `main` and opaque_invariants_simplex's top-level `eps`
        # are now owed manifest roots. Their definitions changed only from
        # object-only parity to executable stdout parity; the dedicated
        # root-boundary suite independently locks both the root selection
        # rule and ordered C realization.
        "parity_hello_tensor_library_only": "4a872b0b09a5f589ab8c31e80f9815f2da83b9093f870b9899a186396349f10f",
        "parity_induction_bond_library_only": "3e83f0cf929583a8df5a8fa05c62e9a712826fbe899b244766fdcad47b741389",
        "parity_iter_foundation": "c99d74a439e006c29748429c3877941460cd3ed18a0a98fd16caf81fb510c84f",
        "parity_linreg_library_only": "041271517605b7fa97a616c9fbe37d97d30bf0a6740c419e7dd91182736d97e4",
        "parity_list_foundation": "5400fe48566a947d9970a2231ea00b2d573abe32fcf3576b3ef7c8dfece02f14",
        "parity_literal_extent_claim": "9458c0cd6591887b81d4e4c6334846f6cd37dc5cd59b4a396a8bd9c7d9684592",
        "parity_mnist_library_only": "f46a10e016c52751f1072770cce71c39e8322d5ce8b19f1c81879d53fbc833f4",
        "parity_opaque_invariants_library_only": "f2a0340b7b1d509b2d06ad84eb11ff7f237015ee90555ea9c1c7609c1b3d25f5",
        "parity_opaque_invariants_simplex_library_only": "92ed0ac36b8cfedfad49a86707e83f230c7edcf0484e29951c0a16c1d9c2e865",
        "parity_rank_poly_borrow_library_only": "4325ec047118bf72a0a4518d371dbaaa7597b861946a13aac1bd4e2c155f642c",
        "parity_recursive_generic": "9bdd2c3be82db69c4fd11d54f6a301e17655f397b6fad90bbdf923a373e198d6",
        "parity_scalar_string_foundation": "30ec444cbbea6be840b3c603121b3d4d35b1c8eec7311c6808377cf4b6883372",
        "parity_tensor_structural_ops": "02b454423039f3ab3b3c99005481cef1cb744fc9ee81af8848f3638d0e793286",
        "parity_transformer_block_library_only": "1219362b0fe28ff5efacbe52249f4c151c646fabdc6557fbb04de5b691bcd06d",
        "parity_vmap_relu_library_only": "0c3450ba322a3254bed1eb9b1474abf660661453af72c4b74d1084b68f0260b1",
    },
    # The nine f32/f16/bf16 rows below were re-frozen when chelis#732 Phase 3
    # rebased onto chelis#729 Phase 1/2 (PRs #1049, #1054). Evidence for the
    # revision: #1054 deleted the untyped `RiscOp::Const { value: f64 }`
    # constructor, so the previous definitions cannot compile at all - this is
    # a forced spelling migration, not a corpus revision. Each row's diff is
    # exactly `RiscOp::Const { value: X }` -> `RiscOp::synth_const(P, X)` where
    # `P` is the same dtype the node already declared and `X` the same literal.
    # No expected string, comparator entrypoint, label, or assertion changed,
    # and all rows pass with their verbatim expected renderings.
    # `agreement_expected_value_reaches_comparator` was added by chelis#1104 as
    # the expected-value leg's usage canary. It is a new row, not a revision:
    # no existing digest moved with it, and its own freeze is secondary - the
    # canary fails behaviorally when `assert_expected` stops consulting the
    # comparator, so tampering with this digest does not buy a green run.

    EVAL_AGREEMENT_SOURCE: {
        "agreement_add": "4db22a4302714b8be688532417daa7776374646094e70c2671875f5564e30960",
        "agreement_atan": "80c1d1edd7d7857c6392470b80ec650e7db64c0d790781746ed92520c4ed9b85",
        "agreement_bf16_add": "ad2029f00557c4759e3c90d2461a223fade971fefd9dcb666569e4b2b9b0974d",
        "agreement_bf16_reduce_sum_matches_eval_exactly": "681adb9f8f561b673798b7e29c35494fa42e62ffbef8757a25a63412fd7c043d",
        "agreement_compiled_observation_reaches_comparator": "bbfed1ebbe05619f0ccb6fd73b1e6356d47444f2e7e4511f492a967f277c1d90",
        "agreement_cos": "bb39c151b0be1d95a7c7e75c4e949e6e4c1da4c335f7abe119e088fa5d866d04",
        "agreement_exp": "fb328fbf479c5e174e3ca7fccc4f2df3404b826771c39618ce841715412a3f12",
        "agreement_expected_value_reaches_comparator": "44e94c88c0ad3978ae6dd3e68bcdd307fdc7b10aa73eb29b73ec1b5705fe5fc2",
        "agreement_f16_add": "8ecc6efe5e052e32850786174495f249776d35676ffffbcced66e308c6ce3c7c",
        "agreement_log": "84ca6c7ba3b27a0718b42f40d15dd1c786648e33562b63a9ecddfaa02416c709",
        "agreement_mul": "ba21fed3999506c32eb163bcdaaa10a135c744008f7b908e5f7fc98c0338fc97",
        "agreement_neg": "71fb3152676dbfbbd06487b493cc2201e3730163b7cceff01801b930bc003a30",
        "agreement_operation_identity_is_derived_from_ir": "b35dd2f9eac4362f5c38639c8a882b38872d5400d2399d8963cfe2427340dec6",
        # Chelis#1313 intentionally replaces only this row's generic
        # `MaxElem(x, synth_const(0))` spelling with the dedicated
        # `RiscOp::Relu(x)` identity. Its negative and positive cases retain
        # the same labels, comparator calls, and exact expected outputs. The
        # independent dtype ReLU oracle proves the dedicated identity's exact
        # per-width semantic/bit contract and evaluator/C agreement (with HIP
        # and Metal structural coverage), while this row preserves both
        # negative-input and positive-input parity.
        "agreement_relu": "d8bfd8d952bcac571f5e7f8b028bf12b367c642b329c9ad22ecc3854727c1659",
        "agreement_sin": "7d1c26bc002402b089c6c035eb756dedd70de05f06c2253ad362462d8243a170",
        "agreement_sqrt_is_exact": "63ee422b92eef85a5635892c57282dbd9cec0154bd3d79ae4c57ca1744f0ac6a",
        "agreement_tan": "33480e20e37c50cbd1ba8ae7864b77f860831c1ecce8577dcf22b28640e0116e",
        "agreement_width_nonconformance_is_behavioral": "8b99a54287fe3517ab80544f77a25eadda35a9d84ec85f0687e15fa4910feb86",
    },
    REJECTED_SOURCE: {
        "metal_rank2_gap_rejects_without_an_artifact": "bfc6fefc2678cea4e76a9bf9935e00e23b7dd1af4a8c98ecfe982cca3f7ff18e",
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
        + definition_digest_violations()
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
