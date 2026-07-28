#!/usr/bin/env python3
"""Authoritative Phase 2 oracle for the faithful-observation contract.

`spec/design/faithful_observation.md` Phase 2 states its oracle as a prose
conjunction ("the #723 and #716 rows green and un-ignored; the round-trip
harness green on every exit in both lanes; the interim byte-decode lock
retired"). Prose is not runnable, and the repo's completion standard
requires one authoritative command per phase, so this script IS that
command. It also closes the specific assurance hole PR #863's exact-head
red team found (F3): the default harness run reports "30 passed, 3
skipped" and nothing anywhere asserts what the three skips ARE, that they
still fail for their declared reasons, or that none of them has silently
gone green.

The obligations, in execution order:

1. **The un-ignore rows** (Phase 2 oracle bullet 1). chelis#723's and
   chelis#716's cells must be PRESENT, must carry no `#[ignore]`, and must
   pass. Structural plus executed: a green run of a test that had been
   quietly re-ignored would otherwise satisfy the suite.
2. **The retired interim locks** (bullet 3) must be absent from the tree.
3. **The round-trip harness green set** (bullet 2) must pass.
4. **The known-red ledger** must hold EXACTLY. `KNOWN_RED_CELLS` below
   names every `#[ignore]`d cell in the harness with its owning issue.
   The oracle requires set equality against the source (an undeclared
   ignore is a silently narrowed corpus; a stale ledger row is a lie), and
   then RUNS each cell:
   - still red on its declared assertion  -> obligation met;
   - red for some other reason            -> FAIL (the cell stopped
     testing what it claims to test);
   - **GREEN**                            -> FAIL, loudly, naming the
     issue. This is the chelis#729 handoff signal: the upstream value or
     capacity repair landed, so the cell must be un-ignored and its ledger
     row deleted in that change set. Without this leg an upstream fix
     leaves a permanently-skipped test behind and nobody learns.
5. **The formatter byte locks** (`chelis_format_shortest` == the Rust
   `format_element`, exhaustive over both half formats) must pass.
6. **The corpus-exclusion lists** must match their declared contents. The
   harness excludes rows for chelis#751 (C ingress) and chelis#717 (the
   eval f64 to_list tag); both are documented at their definitions, both
   are legitimate, and both are also exactly how a corpus gets narrowed
   until "green" means nothing. Widening either list is a deliberate act
   that must edit this ledger too.
7. **The §B2.4 no-third-formatter tripwire** must run green with a
   PRODUCTION allowlist of zero: the only permitted `c-format-narrowing`
   row is the cfg(test) fixture-string one.

Scope, stated rather than assumed (the harness's own no-silent-caps rule):
the set-equality obligation in (4) covers the OBSERVATION HARNESS only -
that is this plan's instrument and the file whose header claims exit
coverage. The sibling matrix files carry `#[ignore]`d cells owned by
chelis#682/#714/#717/#724/#729, which are not this plan's to inventory;
for those files the oracle asserts only that the three Phase 2 oracle rows
named in (1) are un-ignored and green.

Usage:

    .venv/bin/python scripts/faithful_observation_phase2_oracle.py

Acceptance is exit 0 with the final line ``PHASE 2 ORACLE: PASS``. Needs a
host C toolchain: most obligations build, link, and run generated C.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]

HARNESS_SOURCE = Path("crates/chelis-cli/tests/observation_roundtrip_harness.rs")
NARROW_MATRIX_SOURCE = Path("crates/chelis-cli/tests/narrow_dtype_matrix.rs")
REDUCTION_MATRIX_SOURCE = Path("crates/chelis-cli/tests/reduction_and_bitwise_matrix.rs")
TRIPWIRE_SOURCE = Path("crates/chelis-cli/tests/loud_unsupported_tripwire.rs")


class RedCell:
    """One declared known-red harness cell and how it must fail."""

    def __init__(self, name: str, issue: str, fragment: str, owner: str) -> None:
        self.name = name
        # The issue that owns the REPAIR (never this plan: rendering is not
        # the fix site for any of these).
        self.issue = issue
        # A substring of the declared assertion's failure message. Present
        # in the run output => the cell is red for the reason it advertises.
        self.fragment = fragment
        self.owner = owner


# The complete `#[ignore]` inventory of the observation harness. Every row
# is a VALUE or CAPACITY defect upstream of the rendering contract; each
# leaves this table by being repaired upstream and un-ignored, never by
# being weakened here.
KNOWN_RED_CELLS: tuple[RedCell, ...] = (
    RedCell(
        name="eval_int64_scalar_root_above_2p53_renders_exact",
        issue="chelis#684",
        fragment="the labeled root must carry the exact stored int64",
        owner="chelis#729 value layer (rank-0 realization collapses the value)",
    ),
    RedCell(
        name="eval_f64_cast_tensor_root_renders_stored_width",
        issue="chelis#864",
        fragment="the labeled root must render the same stored bits as print",
        owner="chelis#729 value layer (chelis#717 stale precision tag)",
    ),
    RedCell(
        name="c_boxed_f32_renders_at_own_width",
        issue="chelis#865",
        fragment="boxed f32 elements must render shortest at their own width",
        owner="chelis#729/#686 capacity family (the untagged f64 value box)",
    ),
)

# Phase 2 oracle bullet 1: red-to-green only by un-ignoring (§B2.3).
UNIGNORED_ROWS: tuple[tuple[Path, str, str], ...] = (
    (REDUCTION_MATRIX_SOURCE, "c_int64_tensor_print_is_exact_above_2p53", "chelis#723"),
    (NARROW_MATRIX_SOURCE, "c_print_of_f16_tensor_prints_f16_values", "chelis#716"),
    (NARROW_MATRIX_SOURCE, "c_to_list_of_f16_tensor_works", "chelis#716"),
)

# Phase 2 oracle bullet 3: retired per their own in-file instructions,
# replaced by the direct print rows above.
RETIRED_LOCKS: tuple[str, ...] = (
    "c_dag_kernels_compute_correct_f16_bits_despite_print",
    "c_f16_tensor_print_aborts_with_dtype_id_instead_of_misreading",
)

# Declared corpus exclusions (obligation 6). Both are documented at their
# definitions with the owning issue; this table is the mechanical guard
# against either growing silently.
DECLARED_EXCLUSIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "C_LANE_EXCLUDED",
        "chelis#751 (C constant-emission ingress; rows are unconstructible, not red)",
        ("f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max"),
    ),
    (
        "EVAL_F64_LIST_EXCLUDED",
        "chelis#717 (eval's to_list narrows through the stale F32 tag)",
        (
            "f64-max",
            "f64-min-subnormal",
            "f64-min-normal",
            "f64-tenth",
            "f64-17-digit",
            "f64-2p53",
            "f64-2p53-plus-2",
            "f64-audit-e19",
        ),
    ),
)

# The §C2.3 cross-lane byte-identity corpus floor. This lock is what makes
# "the two lanes render identically" a fact rather than an assertion, so it
# must not shrink: growing it is ordinary progress, dropping a row silently
# re-narrows the very claim Phase 2 freezes. The set is a FLOOR (extra
# labels are fine); each entry is a program label inside
# `cross_lane_stdout_is_byte_identical_where_bits_agree`.
CROSS_LANE_CORPUS_FLOOR: tuple[str, ...] = (
    "int8-exits",
    "int64-exits",
    "bool-exits",
    "f32-dyadic-exits",
    "f16-dyadic-print",
    "f64-scalar-17-digit",
    "f64-scalar-specials",
    "rank0-reduction-root",
)

# Obligation 7: the only permitted c-format-narrowing allowlist row. The
# two production rows (host_emit.rs = 4, chelis-runtime lib.rs = 1) were
# deleted at Phase 2 as the Phase 0 handoff promised.
PERMITTED_FORMAT_NARROWING_PATHS: frozenset[str] = frozenset(
    {"crates/chelis-backend-c/src/lib.rs"}
)

GREEN_SUITES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "the round-trip harness green set",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-cli",
            "--test",
            "observation_roundtrip_harness",
        ),
    ),
    (
        "the chelis#716/#723 un-ignored oracle rows",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-cli",
            "-E",
            " + ".join(f"test(={name})" for _, name, _ in UNIGNORED_ROWS),
        ),
    ),
    (
        "the chelis_format_shortest byte locks",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-runtime",
            "--test",
            "format_shortest_matches_reference",
            "--test",
            "format_shortest_invalid_width",
        ),
    ),
    (
        "the §B2.4 no-third-formatter tripwire",
        (
            "cargo",
            "nextest",
            "run",
            "-p",
            "chelis-cli",
            "--test",
            "loud_unsupported_tripwire",
        ),
    ),
)


class OracleFailure(RuntimeError):
    """A failed Phase 2 oracle obligation."""


# ---------------------------------------------------------------------------
# Source structure (pure; unit-tested in test_faithful_observation_phase2_oracle.py)
# ---------------------------------------------------------------------------

_IGNORE_START = re.compile(r"^\s*#\[ignore\b")
_FN_DEF = re.compile(r"^\s*fn\s+([A-Za-z0-9_]+)\s*\(")


def ignored_cells(source: str) -> dict[str, str]:
    """Map each `#[ignore]`d test function to its ignore-attribute text.

    Handles the repo's multi-line ignore reasons (Rust string continuations
    spanning several lines before the closing `]`).
    """

    cells: dict[str, str] = {}
    pending: str | None = None
    buffer: list[str] = []
    open_attribute = False
    for line in source.splitlines():
        if open_attribute:
            buffer.append(line.strip())
            if line.rstrip().endswith("]"):
                open_attribute = False
                pending = " ".join(buffer)
                buffer = []
            continue
        if _IGNORE_START.match(line):
            buffer = [line.strip()]
            if line.rstrip().endswith("]"):
                pending = " ".join(buffer)
                buffer = []
            else:
                open_attribute = True
            continue
        match = _FN_DEF.match(line)
        if match is not None and pending is not None:
            cells[match.group(1)] = pending
            pending = None
    return cells


def defines_test(source: str, name: str) -> bool:
    return re.search(r"^\s*fn\s+" + re.escape(name) + r"\s*\(", source, re.M) is not None


def ledger_violations(source: str, cells: Sequence[RedCell]) -> list[str]:
    """Require the harness's ignore inventory to equal the declared ledger."""

    found = ignored_cells(source)
    declared = {cell.name: cell for cell in cells}
    violations: list[str] = []
    for name in sorted(set(found) - set(declared)):
        violations.append(
            f"{name}: `#[ignore]`d in the harness but absent from KNOWN_RED_CELLS. "
            "An undeclared skip is a silently narrowed corpus - add the row with "
            "its owning issue, or un-ignore the cell."
        )
    for name in sorted(set(declared) - set(found)):
        violations.append(
            f"{name}: declared in KNOWN_RED_CELLS but not `#[ignore]`d in the "
            "harness. If the cell was repaired and un-ignored, delete its ledger "
            "row in the same change set."
        )
    for name in sorted(set(declared) & set(found)):
        issue = declared[name].issue
        if issue not in found[name]:
            violations.append(
                f"{name}: the ignore text does not cite its owning issue {issue}; "
                "every known-red cell must name the repair it waits on."
            )
    return violations


def unignored_violations(sources: dict[Path, str]) -> list[str]:
    violations: list[str] = []
    for relative, name, issue in UNIGNORED_ROWS:
        source = sources.get(relative)
        if source is None:
            violations.append(f"{relative}: unreadable")
            continue
        if not defines_test(source, name):
            violations.append(
                f"{relative}: the {issue} oracle row {name} is gone; Phase 2 "
                "requires it present and green."
            )
            continue
        if name in ignored_cells(source):
            violations.append(
                f"{relative}: the {issue} oracle row {name} is `#[ignore]`d. "
                "Phase 2's oracle requires it un-ignored (§B2.3: red-to-green "
                "only by un-ignoring)."
            )
    return violations


def retired_lock_violations(sources: dict[Path, str]) -> list[str]:
    violations: list[str] = []
    for relative, source in sources.items():
        for name in RETIRED_LOCKS:
            if defines_test(source, name):
                violations.append(
                    f"{relative}: the interim lock {name} is retired at Phase 2 "
                    "(replaced by the direct print row) but still defined."
                )
    return violations


def exclusion_list(source: str, const_name: str) -> list[str] | None:
    match = re.search(
        r"const\s+" + re.escape(const_name) + r"\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
        source,
        re.S,
    )
    if match is None:
        return None
    return re.findall(r'"([^"]*)"', match.group(1))


def exclusion_violations(source: str) -> list[str]:
    violations: list[str] = []
    for const_name, owner, declared in DECLARED_EXCLUSIONS:
        found = exclusion_list(source, const_name)
        if found is None:
            violations.append(
                f"{const_name}: not found in the harness; the corpus-exclusion "
                "guard cannot run."
            )
            continue
        if list(found) != list(declared):
            violations.append(
                f"{const_name} [{owner}] changed: declared {list(declared)}, "
                f"found {list(found)}. Excluding a row removes it from the "
                "corpus entirely - update this ledger deliberately, with the "
                "issue that owns the exclusion."
            )
    return violations


def cross_lane_corpus_labels(source: str) -> list[str] | None:
    """The program labels inside the §C2.3 cross-lane byte-identity lock."""

    match = re.search(
        r"fn cross_lane_stdout_is_byte_identical_where_bits_agree\(\)(.*?)\n\}",
        source,
        re.S,
    )
    if match is None:
        return None
    body = match.group(1)
    programs = re.search(r"let programs:[^=]*=\s*&\[(.*?)\n    \];", body, re.S)
    if programs is None:
        return None
    return re.findall(r'\(\s*"([a-z0-9\-]+)"\s*,', programs.group(1))


def cross_lane_corpus_violations(source: str) -> list[str]:
    labels = cross_lane_corpus_labels(source)
    if labels is None:
        return [
            "cross_lane_stdout_is_byte_identical_where_bits_agree: the §C2.3 "
            "byte-identity lock or its program list could not be parsed; the "
            "corpus floor cannot be checked."
        ]
    missing = [label for label in CROSS_LANE_CORPUS_FLOOR if label not in labels]
    if missing:
        return [
            "cross_lane_stdout_is_byte_identical_where_bits_agree: the §C2.3 "
            f"corpus lost {missing}. Phase 2 freezes cross-lane byte equality; "
            "the corpus may grow but never shrink."
        ]
    return []


def format_narrowing_allowlist(source: str) -> list[tuple[str, int]]:
    rows = re.findall(
        r"Pat::CFormatNarrowing\s*,\s*\"([^\"]+)\"\s*,\s*(\d+)\s*,",
        source,
    )
    return [(path, int(count)) for path, count in rows]


def format_narrowing_violations(source: str) -> list[str]:
    violations: list[str] = []
    for path, count in format_narrowing_allowlist(source):
        if path not in PERMITTED_FORMAT_NARROWING_PATHS:
            violations.append(
                f"{path}: a c-format-narrowing allowlist row of {count} outside "
                "the permitted cfg(test) fixture row. Phase 2 deleted every "
                "PRODUCTION row; a new one is a third formatter (§B2.4, a "
                "review-blocking finding)."
            )
    return violations


def classify_red_run(cell: RedCell, returncode: int, output: str) -> str | None:
    """Decide whether a known-red cell met its obligation."""

    if returncode == 0:
        return (
            f"{cell.name}: PASSED. {cell.issue} appears to be REPAIRED "
            f"({cell.owner}). Un-ignore the cell on its original assertion and "
            "delete its KNOWN_RED_CELLS row in the same change set - a repaired "
            "defect must not leave a permanently skipped test behind."
        )
    tail = output[-2000:]
    if "running 1 test" not in output:
        return (
            f"{cell.name}: the run did not execute exactly one test (filter or "
            f"harness drift). Output:\n{tail}"
        )
    if cell.fragment not in output:
        return (
            f"{cell.name}: red, but NOT on its declared assertion (expected the "
            f"message fragment {cell.fragment!r}). The cell has stopped testing "
            f"what its ledger row claims. Output:\n{tail}"
        )
    return None


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def oracle_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault(
        "CARGO_TARGET_DIR",
        str(Path(tempfile.gettempdir()) / "chelis-faithful-observation-phase2-target"),
    )
    return env


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def read_sources() -> dict[Path, str]:
    sources: dict[Path, str] = {}
    for relative in (
        HARNESS_SOURCE,
        NARROW_MATRIX_SOURCE,
        REDUCTION_MATRIX_SOURCE,
        TRIPWIRE_SOURCE,
    ):
        try:
            sources[relative] = (REPO_ROOT / relative).read_text(encoding="utf-8")
        except OSError as error:
            raise OracleFailure(f"unreadable oracle source {relative}: {error}") from error
    return sources


def run_structural_scan(sources: dict[Path, str]) -> None:
    violations: list[str] = []
    violations.extend(unignored_violations(sources))
    violations.extend(retired_lock_violations(sources))
    violations.extend(ledger_violations(sources[HARNESS_SOURCE], KNOWN_RED_CELLS))
    violations.extend(exclusion_violations(sources[HARNESS_SOURCE]))
    violations.extend(cross_lane_corpus_violations(sources[HARNESS_SOURCE]))
    violations.extend(format_narrowing_violations(sources[TRIPWIRE_SOURCE]))
    if violations:
        raise OracleFailure("structural scan failed:\n" + "\n".join(violations))
    print(
        "+ structural scan: un-ignored oracle rows present, interim locks "
        f"retired, {len(KNOWN_RED_CELLS)} known-red cells declared and cited, "
        "corpus exclusions unchanged, the §C2.3 cross-lane corpus floor "
        f"({len(CROSS_LANE_CORPUS_FLOOR)} programs) intact, production "
        "format-narrowing allowlist empty",
        flush=True,
    )


def run_green_suites(env: dict[str, str]) -> None:
    for label, command in GREEN_SUITES:
        print(f"+ {command_text(command)}", flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise OracleFailure(
                f"{label} failed with exit {completed.returncode}: "
                f"{command_text(command)}"
            )


def run_known_red_cells(env: dict[str, str]) -> None:
    violations: list[str] = []
    for cell in KNOWN_RED_CELLS:
        command = (
            "cargo",
            "test",
            "-p",
            "chelis-cli",
            "--test",
            "observation_roundtrip_harness",
            "--",
            "--ignored",
            "--exact",
            cell.name,
        )
        print(f"+ {command_text(command)}", flush=True)
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        output = completed.stdout + completed.stderr
        violation = classify_red_run(cell, completed.returncode, output)
        if violation is None:
            print(
                f"  known-red as declared: {cell.name} ({cell.issue}, "
                f"repair owned by {cell.owner})",
                flush=True,
            )
        else:
            violations.append(violation)
    if violations:
        raise OracleFailure("known-red ledger failed:\n" + "\n".join(violations))


def main() -> int:
    env = oracle_environment()
    try:
        sources = read_sources()
        run_structural_scan(sources)
        run_green_suites(env)
        run_known_red_cells(env)
    except OracleFailure as error:
        print(f"PHASE 2 ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    print("PHASE 2 ORACLE: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
