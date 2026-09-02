#!/usr/bin/env python3
"""Unrepresentable-domain oracle for chelis#908.

The authoritative completion oracle for the unrepresentable-domain contract.
Every obligation below runs compiled code: the CLI obligations drive a built
`chelis` binary over real `.dp` fixtures, and the suite obligations execute
compiled test binaries through `cargo nextest`. Nothing here is mocked; the
unit tests in `test_unrepresentable_domain_oracle.py` patch the runners to
exercise this script's own decision logic and are evidence about the script,
never a substitute for running it.

Obligations:

1. Bare `:keyword` in expression position → parse error (exit 2, non-empty
   errors array).
2. Valid Deep programs with Names at structural positions (binder slots,
   metadata map keys) score exactly 1.0 — the over-application control
   ensuring that rejecting bare names at RuntimeExpr slots does not
   accidentally break programs that legitimately use names at structural
   positions.
3. `spec/03-deep-syntax.md` [03-PROG-1]: a top-level form is a `module`
   wrapper or a declaration. Every other top-level form is rejected by the
   ingress boundary, identified per [03-PROG-2] by its head symbol or by one
   of the nine syntactic classes that rule fixes, and text yielding no
   top-level form at all is the [03-PROG-3] rejection. The admissible
   spellings still score 1.0.
4. `chelis validate --deep` reaches the same program-level verdict as
   `chelis check`, in the same words. That command runs its own Pest grammar,
   which admits only `node+`, so the stamped ingress has to decide top-level
   acceptance before it or a headless form dies as `expected program` with
   its class unnamed.
5. The stamp pass and successor-carrier validation suites cover the headline
   criterion — bare Name at a RuntimeExpr slot produces a StampError — and
   the chelis#731 Phase 3 obligation that the gated `Node` carrier refuses to
   construct a raw closed-vocabulary tag below itself, in metadata or in a
   child.
6. chelis#1088: the compiler-API embedding surface consumes the same stamped
   carrier. Its parity suite drives one accept/reject corpus through every
   public Deep text door and structurally forbids reopening the weaker
   ingress anywhere in the workspace's production sources.
7. `spec/03-deep-syntax.md` [03-ROLE-2] (chelis#885): a bare identifier at
   an expression position (def body, fn body, application argument, bind
   RHS) is an ingress rejection that identifies the offending name AND
   names the `(var {} ...)` remediation spelling. This is the continuous
   form of the chelis#885 oracle; obligation 2's structural-name controls
   are its over-application guard.

Usage:

    .venv/bin/python scripts/unrepresentable_domain_oracle.py

Acceptance is exit 0 with the final line ``ORACLE: PASS``.

Wiring: `scripts/gate.py`'s `integration` stage and its `--local` pre-push
subset. Hosted CI runs that stage in the `workspace-tests-shard` matrix,
which feeds the stable `Workspace Tests (Linux)` aggregate on every pull
request that is not docs-only. The stage choice is not incidental:
obligations 4 and 5 run `cargo nextest`, which the `lint-rust` worker
deliberately does not install, so the oracle would fail there with
`no such command: nextest`. `scripts/test_gate.py` locks both memberships
and the pairing between the oracle's stage and a worker that installs
cargo-nextest.

Binary handoff (chelis#1322): when a gate command list already builds
`chelis` before it reaches this oracle, `scripts/gate.py` names the built
path in ``CHELIS_ORACLE_BINARY`` and this script skips its own
``cargo build``. See `resolve_chelis_binary`.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Fixture programs ─────────────────────────────────────────────────

# Programs that MUST fail at parse time (bare keyword in expression slot).
KEYWORD_IN_EXPR_FIXTURES: list[tuple[str, str]] = [
    ("bare keyword as def body", "(def {} f :bad)"),
    ("bare keyword inside fn body", "(def {} f (fn {} (params {} x) :oops))"),
    ("bare keyword as app argument", "(def {} f (app {} (var {} g) :arg))"),
]

# Programs that MUST fail at ingress: a bare identifier at an expression
# position (`spec/03-deep-syntax.md` [03-ROLE-2], chelis#885). Every fixture
# names the identifier `oops` so obligation 7 can assert the remediation
# spelling `use `(var {} oops)`` exactly.
NAME_IN_EXPR_FIXTURES: list[tuple[str, str]] = [
    ("bare name as def body", "(def {} f oops)"),
    ("bare name inside fn body", "(def {} f (fn {} (params {} x) oops))"),
    ("bare name as app argument", "(def {} f (app {} (var {} g) oops))"),
    (
        "bare name as bind RHS",
        "(def {} f (fn {} (params {} x) (let {} (bind {} y oops) (var {} y))))",
    ),
]

# The [03-ROLE-2] identification and remediation the obligation-7 diagnostic
# must carry (`StampErrorKind::NameAtExprSlot`'s Display).
NAME_IN_EXPR_IDENTIFICATION = "bare name"
NAME_IN_EXPR_REMEDIATION = "use `(var {} oops)`"

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
    # chelis#885 over-application control: every structural position the
    # corpus guard `structural_symbol_positions_still_score_one` names —
    # record head, kv key, access field, export names, deftype / defsig /
    # def names — must keep scoring exactly 1.0 while obligation 7 rejects
    # bare names at expression positions. The program is that corpus
    # member's, verbatim.
    (
        "record head, kv key, access field, export, deftype names",
        "(module {} stats.prob "
        "(export {} probability prob_value) "
        "(deftype {opaque: true} Probability () "
        "(variant {} Probability (field {} value (t-prim {} f32)))) "
        "(defsig {} probability (t-fn {} (t-prim {} f32) (t-adt {} Probability))) "
        "(def {} probability (fn {} (params {} (x {type: (t-prim {} f32)})) "
        "(record {} Probability (kv {} value (var {} x))))) "
        "(defsig {} prob_value (t-fn {} (t-adt {} Probability) (t-prim {} f32))) "
        "(def {} prob_value (fn {} (params {} (p {type: (t-adt {} Probability)})) "
        "(access {} (var {} p) value))))",
    ),
]

# ── [03-PROG-1] top-level form fixtures ──────────────────────────────

# `spec/03-deep-syntax.md` [03-PROG-1] enumerates the admissible top-level
# forms. Every other top-level form is rejected, and [03-PROG-2] requires the
# rejection to identify the offending form: by its head symbol when it has
# one, and otherwise by one of the nine syntactic classes that rule fixes.
# All nine appear below, so this table walks the whole closed set rather than
# the headed half of it.
TOP_LEVEL_REJECTED_FIXTURES: list[tuple[str, str, str]] = [
    # (name, source, the identification the diagnostic must carry)
    # Headed forms: identified by the backtick-quoted head symbol.
    (
        "top-level fn expression",
        "(fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x))",
        "`fn`",
    ),
    ("top-level var expression", "(var {} x)", "`var`"),
    ("top-level application", "(app {} (var {} f) (var {} x))", "`app`"),
    ("top-level type expression", "(t-prim {} f32)", "`t-prim`"),
    ("top-level pattern", "(pat-var {} x)", "`pat-var`"),
    # `variant` and `field` table under §2.2 but are structural children.
    (
        "top-level variant",
        "(variant {} Some (field {} value (t-prim {} f32)))",
        "`variant`",
    ),
    ("top-level helper", "(params {} (x {type: (t-prim {} f32)}))", "`params`"),
    ("top-level unknown tag", "(future-form {} value)", "`future-form`"),
    # Headless forms: the closed [03-PROG-2] class set, all nine.
    ("bare identifier", "some_name", "a bare identifier"),
    ("bare integer literal", "42", "a bare integer literal"),
    ("bare float literal", "1.5", "a bare float literal"),
    ("bare string literal", '"text"', "a bare string literal"),
    ("bare boolean literal", "true", "a bare boolean literal"),
    ("empty list", "()", "an empty list"),
    (
        "list without a tag symbol",
        "((var {} f) (var {} x))",
        "a list without a tag symbol",
    ),
    ("metadata map", "{key: 1}", "a metadata map"),
    (
        "metadata-annotated form",
        '^{:surf_literal_style "explicit"} (var {} x)',
        "a metadata-annotated form",
    ),
    # [03-PROG-3]: text yielding no top-level form at all.
    ("empty text", "", "empty program"),
    ("whitespace only", "   \n\t\n", "empty program"),
    ("comments only", "; a comment\n", "empty program"),
    ("comments and whitespace", "\n; a\n\n; b\n\n", "empty program"),
]

# [03-PROG-3]'s self-identification for text that yields no top-level form.
EMPTY_PROGRAM_IDENTIFICATION = "empty program"

# The exact class spellings [03-PROG-2] fixes. The oracle asserts the set is
# fully covered above, so a class added to the rule without a fixture here is
# a loud failure rather than silent under-enforcement.
TOP_LEVEL_HEADLESS_CLASSES: tuple[str, ...] = (
    "a bare identifier",
    "a bare integer literal",
    "a bare float literal",
    "a bare string literal",
    "a bare boolean literal",
    "an empty list",
    "a list without a tag symbol",
    "a metadata map",
    "a metadata-annotated form",
)

# The admissible spellings: a `module` wrapper, bare declarations, and a mix.
TOP_LEVEL_ACCEPTED_FIXTURES: list[tuple[str, str]] = [
    ("module wrapper", "(module {} m (def {} f (lit {} 1)))"),
    ("bare declaration", "(def {} f (lit {} 1))"),
    (
        "bare declaration pair",
        "(defsig {} f (t-fn {eff: (effects {})} (t-prim {} int32)))\n"
        "(def {} f (fn {} (params {}) (lit {type: (t-prim {} int32)} 1)))",
    ),
    (
        "module wrapper beside a bare declaration",
        "(module {} m (def {} inner (lit {} 1)))\n(def {} outer (lit {} 2))",
    ),
    ("type declaration", "(deftype {} Color () (variant {} Red))"),
]

# ── Compiled Rust obligations ────────────────────────────────────────

STAMP_NEXTEST_COMMAND: tuple[str, ...] = (
    "cargo",
    "nextest",
    "run",
    "-p",
    "chelis-deep",
    "--test",
    "stamp_to_typed",
    "--test",
    "phase3_successor_validation",
)

# chelis#1088. The compiler-API embedding surface is not reachable from the
# `chelis` CLI's own `.dp` path, so its behavioral coverage runs as a
# compiled test binary: the parity table over every public Deep text door,
# and the structural guard that no production source in the workspace
# reopens the weaker ingress.
COMPILER_API_INGRESS_NEXTEST_COMMAND: tuple[str, ...] = (
    "cargo",
    "nextest",
    "run",
    "-p",
    "chelis-compiler-api",
    "--test",
    "phase3_stamped_ingress",
)


# ── Helpers ──────────────────────────────────────────────────────────


class OracleFailure(RuntimeError):
    """A failed oracle obligation."""


class OracleBinaryError(RuntimeError):
    """An explicitly handed-over `chelis` binary that cannot be used.

    Distinct from `OracleFailure`: nothing about the unrepresentable-domain
    contract has been observed yet. The harness was pointed at a binary that
    is not there, so the run is abandoned before any obligation reports.
    """


# `scripts/gate.py` names the `chelis` it already built in this variable when
# the command list it is running provably builds that binary before reaching
# this oracle. Unset means "build your own", which is what a standalone run
# and hosted CI's `workspace-tests-shard` workers both do.
ORACLE_BINARY_ENV = "CHELIS_ORACLE_BINARY"
STYLE_GATE_DISABLE_ENV = "CHELIS_STYLE_GATE_DISABLE"


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


def target_directory() -> Path:
    """The cargo target directory this run writes to."""
    override = os.environ.get("CARGO_TARGET_DIR")
    return Path(override) if override else REPO_ROOT / "target"


_RESOLVED_CHELIS_BINARY: Path | None = None
_BINARY_RESOLUTION_ATTEMPTED = False


_REMEDIATION = (
    "The explicit setting will not be replaced by a build. Point it at a "
    "`chelis` binary, or unset it entirely so the oracle builds its own."
)


def handed_over_binary() -> Path | None:
    """The `chelis` a caller handed over, validated, or None if none was.

    Absent means no handoff, and the caller stays on the build-it-yourself
    path this oracle has always taken. PRESENT is an explicit override and
    is therefore authoritative: if it does not name an executable file,
    that is an `OracleBinaryError`, never a quiet fall back to
    `cargo build`. Falling back would reintroduce the exact cost the
    handoff removes while still reporting `ORACLE: PASS`, so the caller
    would never learn the handoff had stopped working. This mirrors how
    `scripts/gate.py` treats an explicit `PYO3_PYTHON`, and is deliberately
    not the silent candidate chain `scripts/nautilus_local_gate.py` uses
    for `CHELIS_BIN`.

    Present-but-empty is therefore a failure too, not a soft off switch.
    `export CHELIS_ORACLE_BINARY=` reads as a caller who meant to hand
    something over; treating it as "unset" would let an ambient empty value
    disable the handoff with no notice on either side. `scripts/gate.py`
    applies the same rule when it decides whether a value is the caller's.
    """
    if ORACLE_BINARY_ENV not in os.environ:
        return None
    configured = os.environ[ORACLE_BINARY_ENV].strip()
    if not configured:
        raise OracleBinaryError(
            f"{ORACLE_BINARY_ENV} is set to an empty value. An empty "
            "handoff is not an off switch: it would silently disable the "
            f"handoff instead of naming a binary. {_REMEDIATION}"
        )
    candidate = Path(configured)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if candidate.is_dir():
        raise OracleBinaryError(
            f"{ORACLE_BINARY_ENV} is set to {configured!r}, but "
            f"{candidate} is a directory, not a `chelis` binary. "
            f"{_REMEDIATION}"
        )
    if not candidate.is_file():
        raise OracleBinaryError(
            f"{ORACLE_BINARY_ENV} is set to {configured!r}, but no file "
            f"exists at {candidate}. {_REMEDIATION}"
        )
    if not os.access(candidate, os.X_OK):
        raise OracleBinaryError(
            f"{ORACLE_BINARY_ENV} is set to {configured!r}, but {candidate} "
            f"is not executable. {_REMEDIATION}"
        )
    return candidate


def resolve_chelis_binary() -> Path | None:
    """Return the `chelis` to drive the CLI obligations with, or None.

    The CLI obligations run more than twenty fixtures. Going through
    `cargo run` for each pays cargo's dependency resolution twenty times,
    which is most of this oracle's wall clock and none of its coverage.
    Building once and invoking the produced binary is the same compiled
    behavioral path, and is what keeps the oracle affordable as a per-PR
    gate stage.

    When a caller has already built that binary it can say so in
    ``CHELIS_ORACLE_BINARY`` (chelis#1322), and this function skips the
    build entirely. `scripts/gate.py` does that for any command list whose
    earlier commands provably build `chelis`; a bad path raises rather than
    falling back, so the resolution state is left untouched and a second
    call raises again instead of quietly reaching cargo.

    Otherwise the build runs as before. A failure there is not an oracle
    failure: the caller falls back to `cargo run`, which reports the build
    problem in context.
    """
    global _RESOLVED_CHELIS_BINARY, _BINARY_RESOLUTION_ATTEMPTED
    if _BINARY_RESOLUTION_ATTEMPTED:
        return _RESOLVED_CHELIS_BINARY
    handed_over = handed_over_binary()
    _BINARY_RESOLUTION_ATTEMPTED = True
    if handed_over is not None:
        _RESOLVED_CHELIS_BINARY = handed_over
        return _RESOLVED_CHELIS_BINARY
    build = subprocess.run(
        ("cargo", "build", "-p", "chelis-cli", "--bin", "chelis", "--quiet"),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if build.returncode != 0:
        return None
    candidate = target_directory() / "debug" / "chelis"
    if candidate.is_file():
        _RESOLVED_CHELIS_BINARY = candidate
    return _RESOLVED_CHELIS_BINARY


def run_chelis_check(fixture_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `chelis check` on a fixture file and return the completed process."""
    binary = resolve_chelis_binary()
    if binary is not None:
        cmd: tuple[str, ...] = (
            str(binary),
            "check",
            "--allow-style-violations",
            str(fixture_path),
        )
    else:
        cmd = chelis_check_command() + (str(fixture_path),)
    return run_bounded_child(
        f"chelis check fixture {fixture_path.name}",
        cmd,
        timeout=60,
        env={**os.environ, STYLE_GATE_DISABLE_ENV: "1"},
    )


def run_bounded_child(
    obligation: str,
    command: Sequence[str],
    *,
    timeout: float,
    cwd: Path = REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one oracle child, reaping its process group on timeout.

    The failure is deliberately field-oriented: retained gate transcripts must
    identify which obligation stalled, which child was alive, and whether the
    termination itself completed. A timeout is never reinterpreted as a
    semantic rejection.
    """

    started = time.monotonic()
    try:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=os.name == "posix",
        )
    except OSError as error:
        raise OracleFailure(
            "\n".join(
                (
                    f"obligation={obligation}",
                    f"command={shlex.join(command)}",
                    "state=spawn_failed",
                    f"error={error}",
                )
            )
        ) from error

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        termination = "SIGKILL" if os.name == "posix" else "kill"
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            termination = "already_exited"
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise OracleFailure(
                "\n".join(
                    (
                        f"obligation={obligation}",
                        f"command={shlex.join(command)}",
                        f"pid={process.pid}",
                        "state=termination_failed",
                        f"elapsed={time.monotonic() - started:.3f}s",
                        f"timeout={timeout:.3f}s",
                        f"termination={termination}",
                    )
                )
            ) from error
        raise OracleFailure(
            "\n".join(
                (
                    f"obligation={obligation}",
                    f"command={shlex.join(command)}",
                    f"pid={process.pid}",
                    "state=timed_out",
                    f"elapsed={time.monotonic() - started:.3f}s",
                    f"timeout={timeout:.3f}s",
                    f"termination={termination}",
                    f"stdout={stdout.rstrip()}",
                    f"stderr={stderr.rstrip()}",
                )
            )
        )

    return subprocess.CompletedProcess(
        args=tuple(command),
        returncode=process.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def chelis_validate_command() -> tuple[str, ...]:
    """The cargo command for `chelis validate --deep`.

    The fallback spelling, used only when the build-once binary resolution
    fails. Mirrors `chelis_check_command`, including
    `--allow-style-violations`: these are synthetic fixtures that need not be
    canonically formatted, and the obligation tests the program-level verdict,
    not formatting.
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
        "validate",
        "--deep",
        "--allow-style-violations",
    )


def run_chelis_validate(fixture_path: Path) -> subprocess.CompletedProcess[str]:
    """Run `chelis validate --deep` on a fixture and return the process."""
    binary = resolve_chelis_binary()
    if binary is not None:
        cmd: tuple[str, ...] = (
            str(binary),
            "validate",
            "--deep",
            "--allow-style-violations",
            str(fixture_path),
        )
    else:
        cmd = chelis_validate_command() + (str(fixture_path),)
    return run_bounded_child(
        f"chelis validate fixture {fixture_path.name}",
        cmd,
        timeout=60,
        env={**os.environ, STYLE_GATE_DISABLE_ENV: "1"},
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


def check_name_in_expr_rejected() -> None:
    """Obligation 7: bare identifier in expression position → ingress error.

    `spec/03-deep-syntax.md` [03-ROLE-2] (chelis#885): the rejection
    identifies the offending name and names the `(var {} ...)` remediation
    spelling (`StampErrorKind::NameAtExprSlot`'s Display). Obligation 2's
    structural-name fixtures are the over-application control.
    """
    print("── Obligation 7: bare name at expression position → ingress error ──")
    for name, source in NAME_IN_EXPR_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode == 0:
                raise OracleFailure(
                    f"[{name}] [03-ROLE-2] requires an ingress rejection, got "
                    f"exit 0.\nSource: {source}\nStdout: {result.stdout}"
                )
            report = parse_check_json(result.stdout)
            errors = report.get("errors", [])
            if not errors:
                raise OracleFailure(
                    f"[{name}] exit was non-zero but errors array is empty.\n"
                    f"Source: {source}\nReport: {report}"
                )
            error_text = json.dumps(errors)
            if NAME_IN_EXPR_IDENTIFICATION not in error_text:
                raise OracleFailure(
                    f"[{name}] [03-ROLE-2] requires the diagnostic to identify "
                    f"the offending form as a {NAME_IN_EXPR_IDENTIFICATION}.\n"
                    f"Errors: {errors}"
                )
            if NAME_IN_EXPR_REMEDIATION not in error_text:
                raise OracleFailure(
                    f"[{name}] [03-ROLE-2] requires the diagnostic to name the "
                    f"remediation spelling ({NAME_IN_EXPR_REMEDIATION}).\n"
                    f"Errors: {errors}"
                )
            print(f"  PASS: {name}")
        finally:
            fixture.unlink(missing_ok=True)


def check_score_one_controls() -> None:
    """Obligation 2: valid programs with Names at structural slots score 1.0."""
    print("── Obligation 2: structural-name programs score 1.0 ──")
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


def check_top_level_form_rule() -> None:
    """Obligation 3: `spec/03-deep-syntax.md` [03-PROG-1] and [03-PROG-2].

    A top-level form is a `module` wrapper or a declaration. Every other
    top-level form is rejected at the ingress boundary, and the rejection
    names the offending head.
    """
    print("── Obligation 3: [03-PROG-1] top-level form rule ──")
    covered = {identification for _, _, identification in TOP_LEVEL_REJECTED_FIXTURES}
    missing = [cls for cls in TOP_LEVEL_HEADLESS_CLASSES if cls not in covered]
    if missing:
        raise OracleFailure(
            "[03-PROG-2] classes with no coverage fixture: " + ", ".join(missing)
        )

    for name, source, identification in TOP_LEVEL_REJECTED_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode == 0:
                raise OracleFailure(
                    f"[{name}] [03-PROG-1] requires a rejection, got exit 0.\n"
                    f"Source: {source}\nStdout: {result.stdout}"
                )
            report = parse_check_json(result.stdout)
            errors = report.get("errors", [])
            if not errors:
                raise OracleFailure(
                    f"[{name}] exit was non-zero but errors array is empty.\n"
                    f"Source: {source}\nReport: {report}"
                )
            error_text = json.dumps(errors)
            # [03-PROG-2]: the rejection identifies the offending form.
            if identification not in error_text:
                raise OracleFailure(
                    f"[{name}] [03-PROG-2] requires the diagnostic to identify "
                    f"the offending form as {identification}.\nErrors: {errors}"
                )
            # [03-PROG-2] forbids substituting a placeholder.
            if "<" in error_text or ">" in error_text:
                raise OracleFailure(
                    f"[{name}] [03-PROG-2] forbids a placeholder identification."
                    f"\nErrors: {errors}"
                )
            print(f"  PASS: {name} (identified as {identification})")
        finally:
            fixture.unlink(missing_ok=True)

    for name, source in TOP_LEVEL_ACCEPTED_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_check(fixture)
            if result.returncode != 0:
                raise OracleFailure(
                    f"[{name}] [03-PROG-1] admits this top-level form; got exit "
                    f"{result.returncode}.\nSource: {source}\n"
                    f"Stderr: {result.stderr}\nStdout: {result.stdout}"
                )
            report = parse_check_json(result.stdout)
            score = report.get("score")
            if score != 1 and score != 1.0:
                raise OracleFailure(
                    f"[{name}] expected score 1.0, got {score}.\n"
                    f"Source: {source}\nReport: {report}"
                )
            print(f"  PASS: {name} (score={score})")
        finally:
            fixture.unlink(missing_ok=True)


def check_validate_agrees_with_check() -> None:
    """Obligation 4: `validate --deep` reaches the same program-level verdict.

    `chelis validate --deep` runs its own Pest grammar, which admits only
    `node+`. Before the stamped ingress ran first, a headless top-level form
    died there as `expected program` and the reader never learned its
    [03-PROG-2] class. This obligation drives the same corpus obligation 3
    drives, through the other public command, behind the same real binary.
    """
    print("── Obligation 4: `validate --deep` shares the program-level verdict ──")
    for name, source, identification in TOP_LEVEL_REJECTED_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_validate(fixture)
            if result.returncode == 0:
                raise OracleFailure(
                    f"[{name}] `validate --deep` must reject what `check` "
                    f"rejects, got exit 0.\nSource: {source}\n"
                    f"Stdout: {result.stdout}"
                )
            rendered = result.stdout + result.stderr
            if identification not in rendered:
                raise OracleFailure(
                    f"[{name}] [03-PROG-2] requires `validate --deep` to "
                    f"identify the offending form as {identification}.\n"
                    f"Output: {rendered}"
                )
            if "<" in rendered or ">" in rendered:
                raise OracleFailure(
                    f"[{name}] [03-PROG-2] forbids a placeholder "
                    f"identification.\nOutput: {rendered}"
                )
            print(f"  PASS: {name} (identified as {identification})")
        finally:
            fixture.unlink(missing_ok=True)

    for name, source in TOP_LEVEL_ACCEPTED_FIXTURES:
        fixture = write_fixture(source)
        try:
            result = run_chelis_validate(fixture)
            if result.returncode != 0:
                raise OracleFailure(
                    f"[{name}] [03-PROG-1] admits this top-level form; "
                    f"`validate --deep` gave exit {result.returncode}.\n"
                    f"Source: {source}\nStderr: {result.stderr}"
                )
            print(f"  PASS: {name} (validated)")
        finally:
            fixture.unlink(missing_ok=True)


def run_compiled_suite(label: str, cmd: tuple[str, ...], timeout: int = 300) -> None:
    """Execute a compiled test binary through nextest and require green."""
    print(f"  + {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise OracleFailure(
            f"{label} failed (exit {result.returncode}).\n"
            f"Stderr: {result.stderr}\nStdout: {result.stdout}"
        )
    combined = result.stdout + result.stderr
    print(f"  PASS: {label} green")
    for line in combined.splitlines():
        if "Summary" in line and "tests run" in line:
            print(f"    {line.strip()}")
            break


def check_stamp_pass_integration_tests() -> None:
    """Obligation 5: the stamp pass and successor-carrier suites are green.

    The headline criterion (Name at RuntimeExpr slot → StampError) is
    enforced by the compiled integration tests in
    `crates/chelis-deep/tests/stamp_to_typed.rs`.
    """
    print("── Obligation 5: stamp pass integration tests green ──")
    run_compiled_suite("stamp_to_typed + phase3_successor_validation", STAMP_NEXTEST_COMMAND)


def check_compiler_api_ingress() -> None:
    """Obligation 6: the compiler-API embedding surface shares the carrier.

    chelis#1088. `crates/chelis-compiler-api/tests/phase3_stamped_ingress.rs`
    drives one accept/reject corpus through every public Deep text door and
    carries the structural guard forbidding a weaker ingress in any of the
    workspace's production sources.
    """
    print("── Obligation 6: compiler-API stamped ingress parity ──")
    run_compiled_suite(
        "phase3_stamped_ingress",
        COMPILER_API_INGRESS_NEXTEST_COMMAND,
    )


# ── Main ─────────────────────────────────────────────────────────────


OBLIGATIONS = (
    check_keyword_in_expr_rejected,
    check_score_one_controls,
    check_top_level_form_rule,
    check_validate_agrees_with_check,
    check_stamp_pass_integration_tests,
    check_compiler_api_ingress,
    check_name_in_expr_rejected,
)


def main() -> int:
    """Run all oracle obligations. Returns 0 on full pass, 1 on failure."""
    print("Unrepresentable-domain oracle (chelis#908)")
    print("=" * 60)
    print()
    print(
        "Every obligation runs compiled code: the CLI obligations drive the\n"
        "built `chelis` binary over real .dp fixtures, and the suite\n"
        "obligations execute compiled test binaries through cargo nextest.\n"
    )

    # Resolve the CLI binary up front. A bad handoff is a harness fault, not
    # an obligation result, so it must not be reported as one; failing here
    # also means no obligation runs against a binary nobody chose.
    try:
        binary = resolve_chelis_binary()
    except OracleBinaryError as e:
        print(f"\nORACLE: FAIL\n{e}", file=sys.stderr)
        return 2
    if binary is None:
        print("chelis binary: falling back to `cargo run` per fixture\n")
    elif ORACLE_BINARY_ENV in os.environ:
        # Resolution succeeded and the variable was present, so it named
        # this binary: a present-but-unusable value would have raised.
        print(f"chelis binary: {binary} (handed over by the caller)\n")
    else:
        print(f"chelis binary: {binary} (built by this run)\n")

    try:
        for index, obligation in enumerate(OBLIGATIONS):
            if index:
                print()
            obligation()
        print()
        print("=" * 60)
        print("ORACLE: PASS")
        return 0
    except OracleFailure as e:
        print(f"\nORACLE: FAIL\n{e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
