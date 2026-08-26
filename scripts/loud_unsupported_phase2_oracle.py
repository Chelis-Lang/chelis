#!/usr/bin/env python3
"""Authoritative Phase 2 oracle for the loud-unsupported contract.

This is intentionally one runner rather than a prose conjunction. It executes
the focused vocabulary/runtime/host-state/ABI/emission/public-API surfaces,
the two privacy compile-fail doctests, a structural endpoint scan, and a
controlled vocabulary mutation that adds one fully-decodable variant to BOTH
closed vocabularies (``EffectKind`` and ``RuntimeDType``) in a single
workspace check, requiring non-exhaustive compile errors at each vocabulary's
independent consumers, plus a controlled ``HostAbiType`` mutation (the
backend's crate-private typed ABI vocabulary, section C6.3) that must go red
at the ABI owner's and the emitter's exhaustive matches. The endpoint scan rejects the former Unit and
raw-generic-field escape hatches in addition to the legacy host-type and
expression-emission endpoints. The mutation touches only the vocabulary
owner, refuses to run over a dirty owner file, and restores the original bytes
in a ``finally`` block.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterator, Sequence
from contextlib import contextmanager


REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB_SOURCE = Path("crates/chelis-vocab/src/lib.rs")
HOST_ABI_SOURCE = Path("crates/chelis-backend-c/src/host_abi.rs")

FOCUSED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("cargo", "nextest", "run", "-p", "chelis-vocab"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-runtime",
        "--test",
        "runtime_dtype_generated_header",
        "--test",
        "runtime_dtype_invalid_ffi",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-ir",
        "--test",
        "host_type_failure_states",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-backend-c",
        "-E",
        "test(emitted_expr::tests) | test(host_abi_tests)",
    ),
    ("cargo", "test", "-p", "chelis-backend-c", "--doc"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "pr799_host_resolution_result",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "closed_vocabulary_architecture",
        "--test",
        "int_width_lane_matrix",
        "--test",
        "issue_734_tostring_placeholder",
    ),
)


class OracleFailure(RuntimeError):
    """A failed Phase 2 oracle obligation."""


def oracle_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "CARGO_TARGET_DIR",
        str(Path(tempfile.gettempdir()) / "chelis-loud-unsupported-phase2-target"),
    )
    return env


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def run_success(command: Sequence[str], *, env: dict[str, str]) -> None:
    print(f"+ {command_text(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise OracleFailure(
            f"focused command failed with exit {completed.returncode}: "
            f"{command_text(command)}"
        )


def mutate_effect_kind(source: str) -> str:
    """Add one fully-decodable owner variant so downstream matches go red."""

    replacements = (
        (
            "pub enum EffectKind {\n    Random,\n    Resource,\n}",
            "pub enum EffectKind {\n    Random,\n    Resource,\n"
            "    Phase2OracleMutation,\n}",
        ),
        (
            "pub const ALL: [Self; 2] = [Self::Random, Self::Resource];",
            "pub const ALL: [Self; 3] = [\n"
            "        Self::Random,\n"
            "        Self::Resource,\n"
            "        Self::Phase2OracleMutation,\n"
            "    ];",
        ),
        (
            '            Self::Resource => "resource",\n',
            '            Self::Resource => "resource",\n'
            '            Self::Phase2OracleMutation => "phase2-oracle-mutation",\n',
        ),
        (
            '            EffectKindInput::Symbol("resource") => Ok(Self::Resource),\n',
            '            EffectKindInput::Symbol("resource") => Ok(Self::Resource),\n'
            '            EffectKindInput::Symbol("phase2-oracle-mutation") => {\n'
            "                Ok(Self::Phase2OracleMutation)\n"
            "            }\n",
        ),
    )
    return apply_anchored_replacements(source, replacements, "EffectKind")


def mutate_runtime_dtype(source: str) -> str:
    """Add one fully-decodable dtype variant so downstream matches go red."""

    replacements = (
        (
            "pub enum RuntimeDType {\n"
            "    F32 = 0,\n"
            "    F64 = 1,\n"
            "    I32 = 2,\n"
            "    Bool = 3,\n"
            "    I64 = 4,\n"
            "    Bf16 = 5,\n"
            "    F16 = 6,\n"
            "    I8 = 7,\n"
            "    I16 = 8,\n"
            "}",
            "pub enum RuntimeDType {\n"
            "    F32 = 0,\n"
            "    F64 = 1,\n"
            "    I32 = 2,\n"
            "    Bool = 3,\n"
            "    I64 = 4,\n"
            "    Bf16 = 5,\n"
            "    F16 = 6,\n"
            "    I8 = 7,\n"
            "    I16 = 8,\n"
            "    Phase2OracleDType = 9,\n"
            "}",
        ),
        (
            "    pub const ALL: [Self; 9] = [\n        Self::F32,\n",
            "    pub const ALL: [Self; 10] = [\n        Self::F32,\n",
        ),
        (
            "        Self::I16,\n    ];",
            "        Self::I16,\n        Self::Phase2OracleDType,\n    ];",
        ),
        (
            '            Self::I16 => "int16",\n',
            '            Self::I16 => "int16",\n'
            '            Self::Phase2OracleDType => "phase2-oracle-dtype",\n',
        ),
        (
            '            Self::I16 => "CHELIS_DTYPE_I16",\n',
            '            Self::I16 => "CHELIS_DTYPE_I16",\n'
            '            Self::Phase2OracleDType => "CHELIS_DTYPE_PHASE2_ORACLE",\n',
        ),
        (
            "            Self::I16 => Repr::TwosComplement16,\n",
            "            Self::I16 => Repr::TwosComplement16,\n"
            "            Self::Phase2OracleDType => Repr::Ieee754Binary32,\n",
        ),
        (
            "            8 => Ok(Self::I16),\n",
            "            8 => Ok(Self::I16),\n"
            "            9 => Ok(Self::Phase2OracleDType),\n",
        ),
    )
    return apply_anchored_replacements(source, replacements, "RuntimeDType")


def apply_anchored_replacements(
    source: str, replacements: tuple[tuple[str, str], ...], owner: str
) -> str:
    mutated = source
    for old, new in replacements:
        count = mutated.count(old)
        if count != 1:
            raise OracleFailure(
                f"{owner} owner shape drifted: expected exactly one mutation "
                f"anchor, found {count}: {old!r}"
            )
        mutated = mutated.replace(old, new, 1)
    return mutated


@contextmanager
def temporary_vocab_mutation(path: Path) -> Iterator[None]:
    original = path.read_bytes()
    mutated_text = mutate_runtime_dtype(mutate_effect_kind(original.decode("utf-8")))
    path.write_bytes(mutated_text.encode("utf-8"))
    try:
        yield
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            raise OracleFailure(f"failed to restore controlled mutation: {path}")


def mutate_host_abi_type(source: str) -> str:
    """Add one ABI variant so every exhaustive HostAbiType consumer goes red.

    HostAbiType is the crate-private section C6.3 typed-state boundary in the
    C backend (chelis#730 Phase 2). A new variant with no arms anywhere must
    be a compile-error work-list at the owner's ``c_type_name`` matches and
    the emitter's boxing/unboxing/print consumers, never a silent fall-through.

    Anchor on the enum declaration rather than a particular adjacent variant:
    later dtype phases are expected to refine this closed vocabulary, and the
    mutation oracle must remain live across those sanctioned changes.
    """

    replacements = (
        (
            "pub(crate) enum HostAbiType {\n",
            "pub(crate) enum HostAbiType {\n    Phase2OracleAbi,\n",
        ),
    )
    return apply_anchored_replacements(source, replacements, "HostAbiType")


@contextmanager
def temporary_host_abi_mutation(path: Path) -> Iterator[None]:
    original = path.read_bytes()
    mutated_text = mutate_host_abi_type(original.decode("utf-8"))
    path.write_bytes(mutated_text.encode("utf-8"))
    try:
        yield
    finally:
        path.write_bytes(original)
        if path.read_bytes() != original:
            raise OracleFailure(f"failed to restore controlled mutation: {path}")


def assert_vocab_source_clean(env: dict[str, str]) -> None:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--", str(VOCAB_SOURCE)),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise OracleFailure(f"could not inspect vocabulary source: {completed.stderr}")
    if completed.stdout.strip():
        raise OracleFailure(
            "controlled mutation requires a clean vocabulary owner; commit or "
            f"stash changes to {VOCAB_SOURCE}"
        )


def run_vocab_mutation(env: dict[str, str]) -> None:
    assert_vocab_source_clean(env)
    source_path = REPO_ROOT / VOCAB_SOURCE
    command = ("cargo", "check", "--workspace", "--all-targets", "--keep-going")
    print(
        f"+ controlled EffectKind + RuntimeDType mutation: {command_text(command)}",
        flush=True,
    )
    with temporary_vocab_mutation(source_path):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout
        if completed.returncode == 0:
            raise OracleFailure("added closed-vocabulary variants compiled successfully")
        required_evidence = (
            "non-exhaustive patterns",
            # EffectKind: the two independently-checkable consumer roots.
            "crates/chelis-surf/src/decompile.rs",
            "crates/chelis-types/src/infer/expr.rs",
            # RuntimeDType: every runtime FFI dtype boundary matches
            # exhaustively; an added dtype must go red there before any
            # sizing, allocation, or element-access decision exists for it.
            "crates/chelis-runtime/src/lib.rs",
        )
        missing = [needle for needle in required_evidence if needle not in output]
        if missing:
            tail = "\n".join(output.splitlines()[-80:])
            raise OracleFailure(
                "vocabulary mutation failed without the required exhaustive "
                f"consumer evidence {missing}:\n{tail}"
            )
    print("  mutation produced the expected downstream compile failures", flush=True)


def assert_host_abi_source_clean(env: dict[str, str]) -> None:
    completed = subprocess.run(
        ("git", "status", "--porcelain", "--", str(HOST_ABI_SOURCE)),
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise OracleFailure(f"could not inspect host ABI source: {completed.stderr}")
    if completed.stdout.strip():
        raise OracleFailure(
            "controlled mutation requires a clean ABI owner; commit or stash "
            f"changes to {HOST_ABI_SOURCE}"
        )


def run_host_abi_mutation(env: dict[str, str]) -> None:
    assert_host_abi_source_clean(env)
    source_path = REPO_ROOT / HOST_ABI_SOURCE
    command = (
        "cargo",
        "check",
        "-p",
        "chelis-backend-c",
        "--all-targets",
        "--keep-going",
    )
    print(
        f"+ controlled HostAbiType mutation: {command_text(command)}",
        flush=True,
    )
    with temporary_host_abi_mutation(source_path):
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = completed.stdout
        if completed.returncode == 0:
            raise OracleFailure("an added HostAbiType variant compiled successfully")
        required_evidence = (
            "non-exhaustive patterns",
            # The ABI owner's own exhaustive matches (c_type_name and the
            # Option-inner spelling table).
            "crates/chelis-backend-c/src/host_abi.rs",
            # The emitter's exhaustive consumers (boxing, unboxing, print,
            # declaration paths).
            "crates/chelis-backend-c/src/host_emit.rs",
        )
        missing = [needle for needle in required_evidence if needle not in output]
        if missing:
            tail = "\n".join(output.splitlines()[-80:])
            raise OracleFailure(
                "HostAbiType mutation failed without the required exhaustive "
                f"consumer evidence {missing}:\n{tail}"
            )
    print("  mutation produced the expected downstream compile failures", flush=True)


def endpoint_violations(root: Path) -> list[str]:
    """Return structural endpoint regressions found in production sources."""

    zero_occurrence = (
        (Path("crates/chelis-ir/src/host.rs"), "HostType::Unknown"),
        (Path("crates/chelis-ir/src/host.rs"), "pub fn lower_compiled_program"),
        (Path("crates/chelis-compiler-api/src/compiler.rs"), "::host::lower_compiled_program"),
        (Path("crates/chelis-backend-c/src/emitted_expr.rs"), "EmittedExpr::raw"),
        (Path("crates/chelis-backend-c/src/host_emit.rs"), "EmittedExpr::raw"),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "_ => HostExpr::new(HostExprKind::Unit)",
        ),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "let placeholder = HostExpr::new(HostExprKind::Unit)",
        ),
        (Path("crates/chelis-ir/src/host.rs"), "fn lookup_adt_ctor("),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "fn lookup_adt_ctor_details(",
        ),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "fn lookup_adt_ctor_details_for_type(",
        ),
        (Path("crates/chelis-backend-c/src/host_abi.rs"), '"void*"'),
        (Path("crates/chelis-backend-c/src/host_abi.rs"), '"void *"'),
    )
    required = (
        (
            Path("crates/chelis-backend-c/src/host_abi.rs"),
            "pub(crate) enum HostAbiType",
        ),
        (
            Path("crates/chelis-backend-c/src/host_emit.rs"),
            'require_same_abi_type(ty, &HostType::Unit, "unit expression")?;',
        ),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "pub fn try_lower_compiled_program",
        ),
        (Path("crates/chelis-ir/src/host.rs"), "fn lower_host_expr("),
        (Path("crates/chelis-ir/src/host.rs"), "fn lower_host_expr_kind("),
        (
            Path("crates/chelis-ir/src/host.rs"),
            ") -> Result<HostExpr, crate::lower::LowerDiagnostic> {",
        ),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "struct GenericAdtConstructor",
        ),
        (Path("crates/chelis-ir/src/host.rs"), "struct GenericAdtField"),
        (
            Path("crates/chelis-ir/src/host.rs"),
            "struct InstantiatedAdtConstructor",
        ),
        (
            Path("crates/chelis-ir/src/host.rs"),
            ") -> Result<InstantiatedAdtConstructor, AdtInstantiationError> {",
        ),
    )
    violations: list[str] = []
    cache: dict[Path, str] = {}

    def read(relative: Path) -> str | None:
        if relative in cache:
            return cache[relative]
        path = root / relative
        try:
            cache[relative] = path.read_text(encoding="utf-8")
        except OSError as error:
            violations.append(f"missing endpoint source {relative}: {error}")
            return None
        return cache[relative]

    for relative, forbidden in zero_occurrence:
        source = read(relative)
        if source is not None and forbidden in source:
            violations.append(f"{relative}: forbidden endpoint token {forbidden!r}")
    for relative, marker in required:
        source = read(relative)
        if source is not None and marker not in source:
            violations.append(f"{relative}: missing structural marker {marker!r}")
    return violations


def run_endpoint_scan() -> None:
    violations = endpoint_violations(REPO_ROOT)
    if violations:
        raise OracleFailure("Phase 2 endpoint scan failed:\n" + "\n".join(violations))
    print("+ zero-occurrence and typed-endpoint scan: pass", flush=True)


def main() -> int:
    env = oracle_environment()
    try:
        for command in FOCUSED_COMMANDS:
            run_success(command, env=env)
        run_endpoint_scan()
        run_vocab_mutation(env)
        run_host_abi_mutation(env)
    except OracleFailure as error:
        print(f"PHASE 2 ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    print("PHASE 2 ORACLE: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
