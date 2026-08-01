#!/usr/bin/env python3
"""Unrepresentable-domain oracle for chelis#908.

This oracle verifies the CLI-observable and carrier-internal surfaces of the
unrepresentable-domain contract, including the chelis#731 Phase 3 successor
obligations introduced by the gated `Node` representation:

1. Bare `:keyword` in expression position → parse error (exit 2, non-empty
   errors array).
2. A bare Name in a RuntimeExpr position → stamp error through `chelis check`.
3. Valid Deep programs with Names at structural positions (binder slots,
   metadata map keys) score exactly 1.0 — the over-application control
   ensuring that rejecting bare names at RuntimeExpr slots does not
   accidentally break programs that legitimately use names at structural
   positions.
4. The stamp pass, recursive successor-carrier validator, public Node gate,
   and compiler API stamped-ingress suites all execute and pass.

Usage:

    .venv/bin/python scripts/unrepresentable_domain_oracle.py

Acceptance is exit 0 with the final line ``ORACLE: PASS``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Fixture programs ─────────────────────────────────────────────────

# Programs that MUST fail at parse time (bare keyword in expression slot).
KEYWORD_IN_EXPR_FIXTURES: list[tuple[str, str]] = [
    ("bare keyword as def body", "(def {} f :bad)"),
    ("bare keyword inside fn body", "(def {} f (fn {} (params {} x) :oops))"),
    ("bare keyword as app argument", "(def {} f (app {} (var {} g) :arg))"),
]

BARE_NAME_IN_EXPR_FIXTURES: list[tuple[str, str]] = [
    ("bare name as def body", "(def {} f unwrapped_name)"),
    (
        "bare name inside fn body",
        "(def {} f (fn {} (params {} x) unwrapped_name))",
    ),
    (
        "bare name as app argument",
        "(def {} f (app {} (var {} g) unwrapped_name))",
    ),
]

# Valid programs with Names at structural positions that MUST score 1.0.
# These exercise the over-application control: the domain restriction on
# RuntimeExpr must not accidentally reject names where they belong
# (binder slots, params children, module names, etc.).
SCORE_ONE_CONTROL_FIXTURES: list[tuple[str, str]] = [
    ("simple literal def", "(def {} f (lit {} 42))"),
    ("fn with named params", "(def {} f (fn {} (params {} x) (var {} x)))"),
    (
        "fn with multiple params applied",
        "(def {} my_add (fn {} (params {} a b) (app {} (var {} add) (var {} a) (var {} b))))",
    ),
    (
        "nested fn",
        "(def {} outer (fn {} (params {} x) (fn {} (params {} y) (var {} x))))",
    ),
]

# ── Successor-carrier integration tests (Rust-side oracle) ───────────

SUCCESSOR_NEXTEST_COMMAND: tuple[str, ...] = (
    "cargo",
    "nextest",
    "run",
    "-p",
    "chelis-deep",
    "--test",
    "stamp_to_typed",
    "--test",
    "phase3_successor_validation",
    "-p",
    "chelis-compiler-api",
    "--test",
    "phase3_stamped_ingress",
)


# ── Helpers ──────────────────────────────────────────────────────────


class OracleFailure(RuntimeError):
    """A failed oracle obligation."""


def chelis_check_command() -> tuple[str, ...]:
    """Return the cargo command for `chelis check`.

    Uses --allow-style-violations because these are synthetic oracle
    fixtures that may not pass `chelis fmt --check`. The oracle tests
    semantic behavior, not formatting.
    """
    return (
        "cargo",
        "run",
        "-p",
        "chelis-cli",
        "--bin",
        "chelis",
        "--quiet",
        "--",
        "check",
        "--allow-style-violations",
    )


def run_chelis_check(fixture_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `chelis check` on a fixture file and return the completed process."""
    cmd = chelis_check_command() + (str(fixture_path),)
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def parse_check_json(stdout: str) -> dict:
    """Parse the JSON output from `chelis check`."""
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise OracleFailure(f"Failed to parse chelis check JSON: {e}\nOutput: {stdout}")


def write_fixture(content: str, suffix: str = ".dp") -> Path:
    """Write fixture content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="oracle_")
    os.write(fd, content.encode())
    os.close(fd)
    return Path(path)


# ── Obligations ──────────────────────────────────────────────────────


def check_keyword_in_expr_rejected() -> None:
    """Obligation 1: bare :keyword in expression position → parse error."""
    print("── Obligation 1: bare :keyword → parse error ──")
    for name, source in KEYWORD_IN_EXPR_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode == 0:
                raise OracleFailure(
                    f"[{name}] expected non-zero exit, got 0.\n"
                    f"Source: {source}\nStdout: {result.stdout}"
                )
            # Parse the JSON to verify errors array is non-empty.
            report = parse_check_json(result.stdout)
            errors = report.get("errors", [])
            if not errors:
                raise OracleFailure(
                    f"[{name}] exit was non-zero but errors array is empty.\n"
                    f"Source: {source}\nReport: {report}"
                )
            # Verify the error mentions keyword.
            error_text = json.dumps(errors)
            if "keyword" not in error_text.lower():
                raise OracleFailure(
                    f"[{name}] error does not mention 'keyword'.\n"
                    f"Errors: {errors}"
                )
            print(f"  PASS: {name}")
        finally:
            fixture.unlink(missing_ok=True)


def check_bare_name_in_expr_rejected() -> None:
    """Obligation 2: bare Name in RuntimeExpr position → stamp error."""
    print("── Obligation 2: bare RuntimeExpr Name → stamp error ──")
    for name, source in BARE_NAME_IN_EXPR_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode == 0:
                raise OracleFailure(
                    f"[{name}] expected non-zero exit, got 0.\n"
                    f"Source: {source}\nStdout: {result.stdout}"
                )
            report = parse_check_json(result.stdout)
            errors = report.get("errors", [])
            if not errors:
                raise OracleFailure(
                    f"[{name}] exit was non-zero but errors array is empty.\n"
                    f"Source: {source}\nReport: {report}"
                )
            error_text = json.dumps(errors).lower()
            if "bare name" not in error_text and "stamp error" not in error_text:
                raise OracleFailure(
                    f"[{name}] rejection was not the stamped-name invariant.\n"
                    f"Errors: {errors}"
                )
            print(f"  PASS: {name}")
        finally:
            fixture.unlink(missing_ok=True)


def check_score_one_controls() -> None:
    """Obligation 3: valid programs with Names at structural slots score 1.0."""
    print("── Obligation 3: structural-name programs score 1.0 ──")
    for name, source in SCORE_ONE_CONTROL_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode != 0:
                raise OracleFailure(
                    f"[{name}] expected exit 0, got {result.returncode}.\n"
                    f"Source: {source}\nStderr: {result.stderr}\n"
                    f"Stdout: {result.stdout}"
                )
            report = parse_check_json(result.stdout)
            score = report.get("score")
            if score != 1 and score != 1.0:
                raise OracleFailure(
                    f"[{name}] expected score 1.0, got {score}.\n"
                    f"Source: {source}\nReport: {report}"
                )
            errors = report.get("errors", [])
            if errors:
                raise OracleFailure(
                    f"[{name}] expected empty errors, got: {errors}\n"
                    f"Source: {source}"
                )
            print(f"  PASS: {name} (score={score})")
        finally:
            fixture.unlink(missing_ok=True)


def check_successor_integration_tests() -> None:
    """Obligation 4: all stamped-carrier ingress suites exist and pass."""
    print("── Obligation 4: stamped successor integration suites green ──")
    cmd = SUCCESSOR_NEXTEST_COMMAND
    print(f"  + {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        raise OracleFailure(
            f"Stamped successor integration tests failed (exit {result.returncode}).\n"
            f"Stderr: {result.stderr}\nStdout: {result.stdout}"
        )
    combined = result.stdout + result.stderr
    print("  PASS: stamped successor integration suites green")
    # Print a summary line from nextest if available.
    for line in combined.splitlines():
        if "pass" in line.lower() and ("test" in line.lower() or "run" in line.lower()):
            print(f"    {line.strip()}")
            break


# ── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    """Run all oracle obligations. Returns 0 on full pass, 1 on failure."""
    print("Unrepresentable-domain oracle (chelis#908)")
    print("=" * 60)
    print()
    try:
        check_keyword_in_expr_rejected()
        print()
        check_bare_name_in_expr_rejected()
        print()
        check_score_one_controls()
        print()
        check_successor_integration_tests()
        print()
        print("=" * 60)
        print("ORACLE: PASS")
        return 0
    except OracleFailure as e:
        print(f"\nORACLE: FAIL\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
