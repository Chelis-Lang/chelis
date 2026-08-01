#!/usr/bin/env python3
"""Executable C2.2 oracle for the sealed diagnostic-kind pipeline.

This is a focused Phase 3 component oracle, not the full Phase 3 completion
oracle. It runs the closed-vocabulary, producer/wire, and compile-fail suites,
then plants each same-crate privacy bypass in a non-chokepoint compiler-api
module and one fully rendered vocabulary addition. Every mutation must make
the workspace check fail, and each owner file is restored byte-for-byte in a
``finally`` block.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
MUTATION_SOURCE = Path("crates/chelis-compiler-api/src/context.rs")
VOCAB_SOURCE = Path("crates/chelis-vocab/src/lib.rs")
MUTATION_ANCHOR = "\n#[cfg(test)]"

FOCUSED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("cargo", "nextest", "run", "-p", "chelis-vocab"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "diagnostic_kind_pipeline",
        "--test",
        "pr799_host_resolution_result",
    ),
    ("cargo", "test", "-p", "chelis-compiler-api", "--doc"),
)

WORKSPACE_CHECK: tuple[str, ...] = (
    "cargo",
    "check",
    "--workspace",
    "--all-targets",
)

# Exclude test targets for the vocabulary mutation. Its own exact-list test is
# expected to reject an unratified addition too, but this leg specifically
# proves that the compiler-api producer projection cannot silently inherit a
# newly ratified and fully rendered kind.
VOCABULARY_CHECK: tuple[str, ...] = (
    "cargo",
    "check",
    "--workspace",
    "--lib",
)


class OracleFailure(RuntimeError):
    """A failed C2.2 oracle obligation."""


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def oracle_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "CARGO_TARGET_DIR",
        str(Path(tempfile.gettempdir()) / "chelis-diagnostic-kind-oracle-target"),
    )
    return env


def _insert_before_test_module(source: str, mutation: str) -> str:
    count = source.count(MUTATION_ANCHOR)
    if count != 1:
        raise OracleFailure(
            f"diagnostic privacy mutation anchor drifted: expected 1, found {count}"
        )
    return source.replace(MUTATION_ANCHOR, f"\n{mutation}{MUTATION_ANCHOR}", 1)


def mutate_diagnostic_literal(source: str) -> str:
    mutation = """
#[allow(dead_code)]
fn diagnostic_kind_oracle_literal() -> crate::schema::Diagnostic {
    crate::schema::Diagnostic {
        kind: "unsupported_feature".to_owned(),
        message: "forged".to_owned(),
        severity: 1.0,
        expected: None,
        got: None,
        suggestions: Vec::new(),
        span: None,
        deep_path: None,
    }
}
"""
    return _insert_before_test_module(source, mutation)


def mutate_diagnostic_kind(source: str) -> str:
    mutation = """
#[allow(dead_code)]
fn diagnostic_kind_oracle_mutation(mut diagnostic: crate::schema::Diagnostic) {
    diagnostic.kind = "compile_error".to_owned();
}
"""
    return _insert_before_test_module(source, mutation)


def mutate_diagnostic_vocabulary(source: str) -> str:
    replacements = (
        (
            "    CheckOther,\n}\n\nimpl DiagnosticKind",
            "    CheckOther,\n    Phase3OracleKind,\n}\n\nimpl DiagnosticKind",
        ),
        (
            "    pub const ALL: [Self; 47] = [",
            "    pub const ALL: [Self; 48] = [",
        ),
        (
            "        Self::CheckOther,\n    ];",
            "        Self::CheckOther,\n        Self::Phase3OracleKind,\n    ];",
        ),
        (
            '            Self::CheckOther => "Other",\n',
            '            Self::CheckOther => "Other",\n'
            '            Self::Phase3OracleKind => "phase3_oracle_kind",\n',
        ),
    )
    mutated = source
    for old, new in replacements:
        count = mutated.count(old)
        if count != 1:
            raise OracleFailure(
                "diagnostic vocabulary mutation anchor drifted: "
                f"expected 1, found {count}: {old!r}"
            )
        mutated = mutated.replace(old, new, 1)
    return mutated


@contextmanager
def temporary_mutation(path: Path, mutate: Callable[[str], str]) -> Iterator[None]:
    original = path.read_bytes()
    path.write_text(mutate(original.decode("utf-8")), encoding="utf-8")
    try:
        yield
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            raise OracleFailure(f"failed to restore controlled mutation: {path}")


def run_success(command: Sequence[str], env: dict[str, str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise OracleFailure(
            f"focused command failed with exit {completed.returncode}: "
            f"{command_text(command)}"
        )


def assert_mutation_sources_clean(env: dict[str, str]) -> None:
    completed = subprocess.run(
        (
            "git",
            "status",
            "--porcelain",
            "--",
            str(MUTATION_SOURCE),
            str(VOCAB_SOURCE),
        ),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise OracleFailure("could not inspect diagnostic mutation source status")
    if completed.stdout.strip():
        raise OracleFailure(
            "refusing to mutate dirty diagnostic owner source"
        )


def run_privacy_mutation(
    mutate: Callable[[str], str],
    expected_error: str,
    label: str,
    env: dict[str, str],
) -> None:
    path = REPO_ROOT / MUTATION_SOURCE
    print(f"+ plant {label}; expect {command_text(WORKSPACE_CHECK)} to fail", flush=True)
    with temporary_mutation(path, mutate):
        completed = subprocess.run(
            WORKSPACE_CHECK,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0:
            raise OracleFailure(f"{label} compiled; diagnostic.kind is writable")
        if expected_error not in output or "field `kind`" not in output:
            raise OracleFailure(
                f"{label} failed for the wrong reason; expected {expected_error} and the "
                f"private kind field, got:\n{output[-4000:]}"
            )


def run_vocabulary_mutation(env: dict[str, str]) -> None:
    path = REPO_ROOT / VOCAB_SOURCE
    print(
        "+ add a fully rendered DiagnosticKind; expect "
        f"{command_text(VOCABULARY_CHECK)} to fail",
        flush=True,
    )
    with temporary_mutation(path, mutate_diagnostic_vocabulary):
        completed = subprocess.run(
            VOCABULARY_CHECK,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        if completed.returncode == 0:
            raise OracleFailure(
                "a new DiagnosticKind compiled without a GeneralKind disposition"
            )
        if "error[E0004]" not in output or "src/schema.rs" not in output:
            raise OracleFailure(
                "DiagnosticKind mutation failed for the wrong reason; expected the "
                f"compiler-api projection to go non-exhaustive, got:\n{output[-4000:]}"
            )


def main() -> int:
    env = oracle_environment()
    try:
        for command in FOCUSED_COMMANDS:
            run_success(command, env)
        assert_mutation_sources_clean(env)
        run_privacy_mutation(
            mutate_diagnostic_literal,
            "error[E0451]",
            "out-of-schema Diagnostic literal",
            env,
        )
        run_vocabulary_mutation(env)
        run_privacy_mutation(
            mutate_diagnostic_kind,
            "error[E0616]",
            "post-construction Diagnostic.kind mutation",
            env,
        )
    except OracleFailure as error:
        print(f"C2.2 DIAGNOSTIC KIND ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    print("C2.2 DIAGNOSTIC KIND ORACLE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
