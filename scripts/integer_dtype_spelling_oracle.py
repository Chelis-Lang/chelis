#!/usr/bin/env python3
"""Authoritative acceptance oracle for chelis#1592.

The claim is deliberately bounded: Chelis source and canonical Deep use
``i8``/``i16``/``i32``/``i64``; normal Surf and Deep ingress reject the retired
``int*`` spellings; the explicit v0.18 migrations preserve program structure;
and the existing versioned JSON, WireDag, cache, and ABI vocabulary remains
``int*``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED = re.compile(r"(?<![A-Za-z0-9_])(int8|int16|int32|int64)(?![A-Za-z0-9_])")

TEST_COMMANDS: tuple[tuple[str, ...], ...] = (
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-surf",
        "--test",
        "issue_1587_short_integer_alias",
        "--test",
        "issue_1592_integer_dtype_spelling",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-ir",
        "--test",
        "issue_1948_same_shape_result_claim_sources",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--test",
        "issue_1537_ingress_pass_set_parity",
        "--test",
        "issue_1592_integer_dtype_spelling",
        "--test",
        "unresolved_operand_census",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "issue_1587_short_integer_alias",
        "--test",
        "issue_1592_integer_dtype_spelling",
        "--test",
        "issue_1853_build_check_diagnostics",
        "--test",
        "issue_1948_same_shape_result_claim",
        "--no-fail-fast",
    ),
    (
        ".venv/bin/python",
        "-m",
        "unittest",
        "-v",
        "tests.conformance.hull.test_corpus_integrity",
        "tests.conformance.hull.test_wire_canonical",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "execution_wire_v3",
        "--test",
        "wire_dag_vocabulary",
        "--test",
        "cache_wire_compatibility",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-tide",
        "--test",
        "api",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-prove",
        "--lib",
        "--no-fail-fast",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-runtime",
        "--test",
        "exact_tagged_c_header",
        "--test",
        "exact_tagged_c_abi",
        "--test",
        "runtime_dtype_generated_header",
        "--no-fail-fast",
    ),
)


def _code_without_strings_or_comments(text: str, *, suffix: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    surf_block_depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        pair = text[index : index + 2]
        if surf_block_depth:
            if suffix == ".ch" and pair == "{-":
                surf_block_depth += 1
                output.extend((" ", " "))
                index += 2
                continue
            if suffix == ".ch" and pair == "-}":
                surf_block_depth -= 1
                output.extend((" ", " "))
                index += 2
                continue
            output.append("\n" if char == "\n" else " ")
            index += 1
            continue
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            output.append(" ")
            index += 1
            continue
        if char == '"':
            in_string = True
            output.append(" ")
            index += 1
            continue
        if suffix == ".ch" and pair == "{-":
            surf_block_depth = 1
            output.extend((" ", " "))
            index += 2
            continue
        if pair in {"//", "--"} or (suffix == ".dp" and char == ";"):
            while index < len(text) and text[index] != "\n":
                output.append(" ")
                index += 1
            continue
        output.append(char)
        index += 1
    return "".join(output)


def retired_spelling_hits(paths: Iterable[Path]) -> list[tuple[Path, int, str]]:
    hits: list[tuple[Path, int, str]] = []
    for path in paths:
        code_text = _code_without_strings_or_comments(
            path.read_text(encoding="utf-8"),
            suffix=path.suffix,
        )
        for line_number, code in enumerate(code_text.splitlines(), start=1):
            hits.extend(
                (path, line_number, match.group(1)) for match in RETIRED.finditer(code)
            )
    return hits


def tracked_language_sources() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.ch", "*.dp"],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [
        REPO_ROOT / relative.decode("utf-8")
        for relative in result.stdout.split(b"\0")
        if relative
    ]


def _require(path: str, fragments: tuple[str, ...]) -> list[str]:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    return [f"{path}: missing {fragment!r}" for fragment in fragments if fragment not in text]


def _forbid(path: str, fragments: tuple[str, ...]) -> list[str]:
    text = (REPO_ROOT / path).read_text(encoding="utf-8")
    return [f"{path}: retained {fragment!r}" for fragment in fragments if fragment in text]


def boundary_contract_errors() -> list[str]:
    errors: list[str] = []
    errors.extend(
        _require(
            "crates/chelis-types/src/types.rs",
            (
                'Prim::Int8 => "i8"',
                'Prim::Int64 => "i64"',
                'Prim::Int8 => "int8"',
                'Prim::Int64 => "int64"',
                '"int64" => Some(Prim::Int64)',
            ),
        )
    )
    errors.extend(
        _require(
            "spec/04-type-system.md",
            (
                "(t-prim {} i8)",
                "(t-prim {} i64)",
                "The retired v0.18 spellings `int8`, `int16`, `int32`, and `int64`",
            ),
        )
    )
    errors.extend(
        _require(
            "spec/10-serialization.md",
            (
                "| `int64`, `int32`, `int16`, `int8` |",
                "language spelling",
                "interchange spelling",
            ),
        )
    )
    errors.extend(
        _require(
            "spec/design/chelis_hull_design_spec.md",
            ("exact-int64 extents, dynamic-int32 rank",),
        )
    )
    errors.extend(
        _require(
            "crates/chelis-vocab/src/lib.rs",
            ('Self::I32 => "int32"', 'Self::I64 => "int64"'),
        )
    )
    errors.extend(
        _require(
            "crates/chelis-tide/tests/api.rs",
            ('["value"]["dtype"],\n        "int64"',),
        )
    )
    errors.extend(
        _require(
            "crates/chelis-backend-hip/tests/device_entry_execution.rs",
            ('Some("flat-index-int32")',),
        )
    )
    errors.extend(
        _require(
            "crates/chelis-backend-metal/tests/gpu_correctness.rs",
            ("one int64 output", "the int64 output"),
        )
    )
    errors.extend(
        _require(
            "crates/chelis-runtime/tests/tensor_repurpose.rs",
            ("non-int64 rank", "non-int64 shape extent"),
        )
    )
    errors.extend(
        _require(
            "scripts/nautilus_local_gate.py",
            (
                "range(cast(0, i64),",
                "cast(0, i64))",
                "j: i64",
            ),
        )
    )
    errors.extend(
        _require(
            "editors/vscode/syntaxes/chelis.tmLanguage.json",
            ("|i8|i16|i32|i64|bool|string|unit)",),
        )
    )
    errors.extend(
        _forbid(
            "editors/vscode/syntaxes/chelis.tmLanguage.json",
            ("|int8|", "|int16|", "|int32|", "|int64|"),
        )
    )
    errors.extend(
        _require(
            "grammars/tree-sitter-chelis-surf/queries/highlights.scm",
            ("|i8|i16|i32|i64|bool|string)",),
        )
    )
    errors.extend(
        _forbid(
            "grammars/tree-sitter-chelis-surf/queries/highlights.scm",
            ("|int8|", "|int16|", "|int32|", "|int64|"),
        )
    )
    errors.extend(
        _forbid(
            "docs/CHELIS_SURFACE.md",
            ("int8", "int16", "int32", "int64"),
        )
    )
    errors.extend(
        _forbid(
            "tests/conformance/hull/build_corpus.py",
            ("(t-prim {} int8)", "(t-prim {} int16)", "(t-prim {} int32)", "(t-prim {} int64)"),
        )
    )
    errors.extend(
        _forbid(
            "tests/conformance/hull/known_conservative.json",
            ("(t-prim {} int8)", "(t-prim {} int16)", "(t-prim {} int32)", "(t-prim {} int64)"),
        )
    )
    errors.extend(
        _forbid(
            "crates/chelis-cli/tests/issue_1853_build_check_diagnostics.rs",
            ("def f() -> int32", "def g() -> int32"),
        )
    )
    errors.extend(
        _forbid(
            "crates/chelis-compiler-api/src/schema.rs",
            ("(`f32`, `int64`, `bool`, ...)",),
        )
    )
    errors.extend(
        _forbid(
            "crates/chelis-types/tests/fixtures/unresolved_operand_census.json",
            ("unifies the slot with int64",),
        )
    )
    errors.extend(
        _forbid(
            "grammars/tree-sitter-chelis-surf/src/scanner.cc",
            ("`int64::max`",),
        )
    )
    errors.extend(
        _forbid(
            "bindings/python/tests/manual_reef_context.py",
            ("`with seed(...)` int64 rule",),
        )
    )
    errors.extend(
        _forbid(
            "crates/chelis-python/tests/manual_reef_context.rs",
            ("`with seed(...)` int64 rule",),
        )
    )
    errors.extend(
        _forbid(
            "spec/design/remediation_roadmap.md",
            ("`JsonInt(int64)`", "`cast(3.5, int32)`"),
        )
    )
    return errors


def run_command(command: tuple[str, ...]) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="run the corpus and boundary checks without the Rust test commands",
    )
    args = parser.parse_args()

    errors = boundary_contract_errors()
    for path, line, spelling in retired_spelling_hits(tracked_language_sources()):
        errors.append(
            f"{path.relative_to(REPO_ROOT)}:{line}: retired dtype spelling `{spelling}`"
        )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    if not args.scan_only:
        for command in TEST_COMMANDS:
            run_command(command)

    print("INTEGER DTYPE SPELLING ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
