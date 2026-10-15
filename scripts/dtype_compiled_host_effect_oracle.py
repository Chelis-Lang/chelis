#!/usr/bin/env python3
"""Compiled host-effect acceptance oracle for chelis#1297.

[05-HOST-1], [05-HOST-2] and [05-HOST-3] make `tensor_scan`, `process_run`,
the clock reads, `round_to`, the CSV builtins and the `test_*` assertions
legal host-runtime operations in every execution mode. This oracle proves
compiled C produces the same typed result or language trap, in the same
effect order, as `chelis eval`.

Parity rule: two lanes agree when they print the same stdout, exit with the
same status, and report the same failure body. The one presentation
difference ignored is eval's `error: ` prefix on a runtime failure.

The structural leg forbids the evaluator-only and host-only build rosters,
the compiled-target rejection text, lossy process decoding, and
dtype-named assertion aliases, and requires each operation's one runtime
definition to be called by both lanes. The behavioural legs build and run
exact positive and negative programs on both lanes. The self-test leg
proves each structural mutation fails the oracle. Tensor-scan conversion and stack
identities are checked inside their lowering branch; state checks are checked
inside the callback owner. Typed identities elsewhere cannot supply evidence.

Acceptance is exit 0 with the final line
``DTYPE COMPILED HOST EFFECT ORACLE: PASS``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess
import sys

if __package__:
    from .faithful_observation_phase3_oracle import _matching_rust_brace, rust_code_mask
else:
    from faithful_observation_phase3_oracle import _matching_rust_brace, rust_code_mask


REPO_ROOT = Path(__file__).resolve().parents[1]


class OracleFailure(RuntimeError):
    """A compiled host-effect contract is absent."""


@dataclass(frozen=True)
class SourceContract:
    name: str
    path: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    owner: str | None = None


@dataclass(frozen=True)
class OracleLeg:
    name: str
    argv: tuple[str, ...]


# Each host-runtime operation: its runtime module, the evaluator's call into
# it, and the C export the host emitter calls.
SHARED_DEFINITIONS = (
    ("clock reads", "host_clock", "crates/chelis-compiler-api/src/runtime/system_adapter.rs",
     "chelis_runtime::host_clock::read_wall_clock()", ("chelis_clock_wall_read()", "chelis_clock_monotonic_read()")),
    ("process_run", "host_process", "crates/chelis-compiler-api/src/runtime/eval.rs",
     "chelis_runtime::host_process::decode_process_output(", ("chelis_process_run(",)),
    ("round_to", "host_round", "crates/chelis-compiler-api/src/runtime/eval.rs",
     "use chelis_runtime::host_round::{FloatLayout, round_to_bits};", ("chelis_round_to(",)),
    ("CSV", "host_csv", "crates/chelis-compiler-api/src/runtime/csv.rs",
     "host_csv::parse_csv_text(text)", ("chelis_{name}({call_args})", "chelis_csv_nrows(")),
    ("assertions", "host_assert", "crates/chelis-compiler-api/src/runtime/eval.rs",
     "chelis_runtime::host_assert::assert_eq_message(",
     ("chelis_test_assert_fail(", "chelis_test_assert_eq(", "chelis_test_assert_eq_tensor(",
      "chelis_test_assert_close_tensor(")),
)

HOST_EMIT = "crates/chelis-backend-c/src/host_emit.rs"
TENSOR_SCAN_OWNER = 'if active_compiler_name == Some("tensor_scan") && kids.len() == 4'
TENSOR_SCAN_CALLBACK_OWNER = "fn checked_tensor_scan_callback("


def rust_owner_span(source: str, owner: str) -> tuple[int, int]:
    """Select one real Rust owner block; comments/literals cannot name an owner."""
    try:
        code = rust_code_mask(source)
        keyword = owner.split()[0]
        matches = [match for match in re.finditer(re.escape(owner), source)
                   if code.startswith(keyword, match.start())]
        if len(matches) != 1:
            raise OracleFailure(f"tensor_scan owner {owner!r}: expected exactly one code owner")
        opening = code.find("{", matches[0].end())
        between = code[matches[0].end():opening] if opening != -1 else ""
        valid_header = (
            not between.strip() if owner == TENSOR_SCAN_OWNER else
            re.fullmatch(r"[^;={}]*\)\s*->\s*HostCallback\s*", between) is not None
            and re.search(r"\bfn\b", between) is None
        )
        if opening == -1 or not valid_header:
            raise OracleFailure(f"tensor_scan owner {owner!r}: missing body")
        closing = _matching_rust_brace(code, opening, masked=True)
        return opening + 1, closing
    except ValueError as error:
        raise OracleFailure(f"tensor_scan owner {owner!r}: malformed Rust: {error}") from error


def has_rust_expression(source: str, expression: str) -> bool:
    """Literal spelling and executable tokens must agree at the same offsets."""
    code = rust_code_mask(source)
    expected = rust_code_mask(expression)
    return any(code[match.start():match.end()] == expected
               for match in re.finditer(re.escape(expression), source))


def source_contracts() -> tuple[SourceContract, ...]:
    contracts = [
        SourceContract(
            "no evaluator-only build roster",
            "crates/chelis-ir/src/host.rs",
            (),
            ("EVAL_ONLY_HOST_BUILTINS", "fn find_eval_only_host_builtin"),
        ),
        SourceContract(
            "no evaluator-only drop or gate in the CLI",
            "crates/chelis-cli/src/main.rs",
            (),
            ("drop_unreachable_eval_only_defs", "reject_eval_only_builtins", "reject_host_only_builtins"),
        ),
        SourceContract(
            "no host-only or evaluator-only gate in the compiler API",
            "crates/chelis-compiler-api/src/compiler.rs",
            (),
            (
                "HOST_ONLY_BUILTINS",
                "fn reject_eval_only_builtins",
                "fn reject_host_only_builtins",
                "the host interpreter's eval/test lanes only",
            ),
        ),
        SourceContract(
            "no compiled-target rejection constructor",
            "crates/chelis-types/src/unsupported.rs",
            (),
            ("fn compiled_host_only_builtin",),
        ),
        SourceContract(
            "strict process captures",
            "crates/chelis-runtime/src/host_process.rs",
            ("String::from_utf8(stdout)", "String::from_utf8(stderr)", "exit_status.map_or(-1_i64, i64::from)"),
            ("from_utf8_lossy",),
        ),
        SourceContract(
            "numeric traps exit with eval's status",
            "crates/chelis-runtime/include/chelis_runtime.h",
            ("static inline void chelis_flush_and_exit_trap(void) {\n    (void)fflush(stdout);\n    (void)fflush(stderr);\n    exit(1);\n}",),
            ("chelis_flush_and_abort",),
        ),
        SourceContract(
            "tensor_scan composes the list scan at the state dtype",
            "crates/chelis-ir/src/host.rs",
            (
                'chelis_abi::failure::negative_length_prefix("tensor_scan")',
                'name: "to_tensor".into(),',
                "name: TENSOR_SCAN_STACK.into(),",
                "checked_tensor_scan_callback(callback, &state_ty, template, next, checked)",
                "let state_ty = host_expr_type(&init_expr);",
                "let states_ty = HostTypeTerm::List(Box::new(state_ty.clone()));",
            ),
            owner=TENSOR_SCAN_OWNER,
        ),
        SourceContract(
            "tensor_scan callback checks the actual state against the initial type",
            "crates/chelis-ir/src/host.rs",
            (
                "name: TENSOR_SCAN_STATE.into(),",
                "HostExprKind::Var(next.clone(), ret_ty.clone())",
                "HostExprKind::Var(template.to_string(), state_ty.clone())",
            ),
            owner=TENSOR_SCAN_CALLBACK_OWNER,
        ),
        SourceContract(
            "tensor states are checked per application and stack against the initial state",
            HOST_EMIT,
            ("chelis_tensor_scan_check_state(", "chelis_tensor_scan_stack("),
        ),
        SourceContract(
            "assertions are emitted, never stubbed",
            HOST_EMIT,
            ('"test_assert" => {', '"test_assert_eq" => {', '"test_assert_eq_tensor" => {', '"test_assert_close_tensor" => {'),
        ),
    ]
    for name, module, eval_path, eval_call, exports in SHARED_DEFINITIONS:
        contracts.append(SourceContract(
            f"{name}: one runtime definition",
            f"crates/chelis-runtime/src/{module}.rs",
            ("pub fn",),
        ))
        contracts.append(SourceContract(f"{name}: evaluator calls the runtime", eval_path, (eval_call,)))
        contracts.append(SourceContract(f"{name}: compiled C calls the runtime", HOST_EMIT, exports))
    return tuple(contracts)


# Assertion identities are generic ([05-HOST-3]); a dtype- or rank-named
# spelling is a forbidden alias.
ALIAS = re.compile(r'"(test_)?assert_[a-z_]*_(?:f|i|bf)(?:8|16|32|64)\b|"(test_)?assert_[a-z_]*_rank\d')


def validate_source_contracts(repo_root: Path = REPO_ROOT) -> None:
    for contract in source_contracts():
        path = repo_root / contract.path
        if not path.exists():
            raise OracleFailure(f"{contract.name}: missing {contract.path}")
        source = path.read_text()
        if contract.owner is not None:
            start, end = rust_owner_span(source, contract.owner)
            source = source[start:end]
        for required in contract.required:
            if not (has_rust_expression(source, required) if contract.owner else required in source):
                raise OracleFailure(f"{contract.name}: missing {required!r} in {contract.path}")
        for forbidden in contract.forbidden:
            if forbidden in source:
                raise OracleFailure(f"{contract.name}: forbidden {forbidden!r} in {contract.path}")
    builtins = (repo_root / "crates/chelis-types/src/builtins.rs").read_text()
    match = ALIAS.search(builtins)
    if match:
        raise OracleFailure(f"assertion identities are generic: alias {match.group(0)!r}")


def oracle_legs(python: str) -> tuple[OracleLeg, ...]:
    cli_tests = (
        "compiled_host_effects",
        "process_run_builtin",
        "host_clock_builtins",
        "std_datetime_clock",
        "issue_1170_rejection_scope",
    )
    return (
        OracleLeg("oracle self-tests and structural mutations", (python, "-B", "scripts/test_dtype_compiled_host_effect_oracle.py")),
        OracleLeg(
            "shared runtime definitions",
            ("cargo", "nextest", "run", "-p", "chelis-runtime", "--lib", "-E", "test(/host_(clock|process|round|csv|assert)::/)"),
        ),
        OracleLeg(
            "eval and compiled C parity for every operation",
            ("cargo", "nextest", "run", "-p", "chelis-cli", *(arg for test in cli_tests for arg in ("--test", test))),
        ),
        OracleLeg(
            "compiler API builds tensor_scan for C and HIP host code",
            ("cargo", "nextest", "run", "-p", "chelis-compiler-api", "--test", "issue_257_tensor_scan_host_runtime"),
        ),
        OracleLeg(
            "numeric surface registrations",
            ("cargo", "nextest", "run", "-p", "chelis-cli", "--test", "capacity_census_tripwire", "-E", "test(capacity_census_matches_public_surface)"),
        ),
    )


def run_oracle(python: str = sys.executable, structural_only: bool = False) -> None:
    validate_source_contracts()
    if structural_only:
        print("DTYPE COMPILED HOST EFFECT ORACLE: STRUCTURAL PASS")
        return
    legs = oracle_legs(python)
    for index, leg in enumerate(legs, start=1):
        print(f"[{index}/{len(legs)}] {leg.name}: {shlex.join(leg.argv)}", flush=True)
        try:
            subprocess.run(leg.argv, cwd=REPO_ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"DTYPE COMPILED HOST EFFECT ORACLE: FAIL: {leg.name} (exit {error.returncode})"
            ) from error
    print("DTYPE COMPILED HOST EFFECT ORACLE: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help="check the source contracts without building or running programs",
    )
    args = parser.parse_args()
    try:
        run_oracle(structural_only=args.structural_only)
    except OracleFailure as error:
        print(f"DTYPE COMPILED HOST EFFECT ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
