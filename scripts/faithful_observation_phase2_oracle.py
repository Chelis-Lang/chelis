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
6. **The corpus-exclusion lists** carry the full §B2.9 three-legged
   treatment. The harness excludes rows for chelis#751 (C ingress) and
   chelis#717 (the eval f64 to_list tag); both are documented at their
   definitions, both are legitimate, and both are also exactly how a
   corpus gets narrowed until "green" means nothing. Leg 1: each list
   must EQUAL its `DECLARED_EXCLUSIONS` row - changing a list in either
   direction edits this ledger too (the pre-2026-07-30 check stopped
   here, so a stale list overstating breakage stayed green). Legs 2-3:
   each list's declared, NON-ignored exclusion probe must exist in the
   harness, iterate the list constant itself, and carry the
   shrink-protocol message - the probes run in the default suite (and in
   obligation 3's harness leg), re-executing every excluded behavior and
   failing the moment an upstream repair lands. Their first execution
   shrank `EVAL_F64_LIST_EXCLUDED` by the text-coincident `f64-tenth`.
7. **The §B2.4 no-third-formatter tripwire** must run green, and every
   hosted class's baseline paths must stay inside its per-class
   permitted set (`FORMAT_CLASS_TABLE`): `c-format-narrowing`'s
   PRODUCTION allowlist is zero (the only permitted row is the
   cfg(test) fixture-string one), and the Rust classes'
   (`rust-format-narrowing`, `rust-debug-numeric-format`) permitted
   sets are the frozen annotated non-exit carriers. A new baseline path
   requires editing BOTH the tripwire and this oracle - the
   two-instrument interlock.
8. **Doc-citation parity** (§B2.8). Every tripwire `Pat` whose `doc()`
   cites `faithful_observation.md` must be a `FORMAT_CLASS_TABLE` key,
   and vice versa: a hosted detector this oracle cannot see, or a
   coverage row for a detector that no longer exists, is structural
   failure - the 2026-07-30 review's finding one layer up.
9. **The §B2 rule-instrument manifest** (§B2.8). The design doc's §B2
   item list must equal `B2_RULE_INSTRUMENTS` (numbers and titles), and
   every named instrument must exist (an oracle function here, a green
   suite label, or an explicit justified review-rule entry). A §B2 rule
   cannot land without a deliberate instrument decision.

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
DESIGN_DOC = Path("spec/design/faithful_observation.md")


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

# Declared corpus exclusions (obligation 6, §B2.9 three-legged). Both are
# documented at their definitions with the owning issue. Leg 1 is the
# equality check against `labels` (either direction). Legs 2-3 are the
# named `probes`: NON-ignored harness tests that iterate the list constant
# itself, re-execute each excluded behavior on its declared fingerprint,
# and fail with the shrink-protocol message when a repair lands. A list
# may not exist here without at least one probe (§B2.9: a boundary that
# cannot be re-executed may not exist).
#
# 2026-07-30: `f64-tenth` left EVAL_F64_LIST_EXCLUDED on the probes' first
# execution - its F32-narrowed image renders `0.1`, text-coincident with
# the original f64, so the text assertion the list guards passes for it.
# The stale row is precisely what the pre-probe equality-only check could
# never catch.
DECLARED_EXCLUSIONS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "C_LANE_EXCLUDED",
        "chelis#751 (C constant-emission ingress; rows are unconstructible, not red)",
        (
            "c_lane_excluded_labels_still_fail_at_ingress",
            "c_lane_excluded_neg_zero_still_drops_the_sign",
        ),
        ("f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max"),
    ),
    (
        "EVAL_F64_LIST_EXCLUDED",
        "chelis#717 (eval's to_list narrows through the stale F32 tag)",
        ("eval_f64_list_excluded_rows_still_narrow_through_the_f32_tag",),
        (
            "f64-max",
            "f64-min-subnormal",
            "f64-min-normal",
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
# labels are fine).
#
# Each row binds its label to TOKENS ITS PROGRAM MUST CONTAIN. A label
# alone was not enough (PR #863 R2, LOW): a body could be weakened or
# swapped for a trivial program while the floor still reported intact, so
# the guard would certify a corpus that no longer covers what its names
# claim. The tokens are the dtype the row exists for plus the construct
# that makes it an EXIT, which is the least a replacement would have to
# preserve to still be the row it is named after.
CROSS_LANE_CORPUS_FLOOR: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("int8-exits", ("int8",)),
    ("int64-exits", ("int64",)),
    ("bool-exits", ("bool", "to_tensor")),
    ("f32-dyadic-exits", ("f32", "to_tensor")),
    ("f16-dyadic-print", ("f16", "print", "to_list")),
    ("f64-scalar-17-digit", ("f64", "print")),
    ("f64-scalar-specials", ("f64", "div")),
    ("rank0-reduction-root", ("f32", "sum", "print")),
)

# The lock's own body must keep comparing FULL stdout across both lanes.
# A corpus of the right size proves nothing if the assertion under it
# stops being byte equality.
CROSS_LANE_REQUIRED_ASSERTIONS: tuple[str, ...] = (
    "eval_stdout",
    "c_stdout",
    "assert_eq!",
)

RUNTIME_SOURCE = Path("crates/chelis-runtime/src/lib.rs")
RUNTIME_HEADER = Path("crates/chelis-runtime/include/chelis_runtime.h")
# The retired-export scan reads the WHOLE crate, not just `lib.rs`: an
# export re-added in any module is the same public exit returning, and a
# scan scoped to one file would miss it (`chelis_format_shortest` itself
# lives in `format_shortest.rs`, which is how this gap surfaced).
RUNTIME_SRC_DIR = Path("crates/chelis-runtime/src")

# The observation lane's DECODE table: which pointer view each dtype's
# element arm in `tensor_elem_to_string` is allowed to read through.
#
# chelis#732 Phase 2 made FORMATTING canonical (one `format_element`, the
# C side generated from it) and left DECODING - `bits <- read(bytes,
# dtype)` - as hand-written arms. That is how a native two's-complement
# int32 buffer came to be read through an f32 view at this exit while
# `to_list` and the generated print helper read it correctly: one
# formatter, N decoders. `format_element(prim, ElementRef)` and
# `chelis_format_shortest(value, dtype, buf, cap)` both receive an
# ALREADY-DECODED element, so neither can catch it.
#
# ENFORCEMENT RUNG (docs/agent_quality_architecture.md, chelis#740): this
# table is a TRIPWIRE TEST, the third rung, and the standing rule is to
# push every rule as far up the ladder as it can go. The justification for
# not taking a higher rung HERE: the compile-error rung for this class is
# chelis#893's `seal-tensor-data-pointer` then `type-tensor-element-access`
# (make the untyped `chelis_tensor.data` pointer unreachable, then make the
# forced accessor impossible to get wrong), which rewrites ~46 call sites
# across the runtime and belongs in that issue's change set, not in a
# rendering PR that overlaps chelis#894 in five files. chelis#894 supplies
# the complement by making REPRESENTATION rather than width the ABI
# primitive - width does not determine representation (`Ieee754Binary32`
# and `TwosComplement32` are both 4 bytes and are not interchangeable,
# which is exactly why a width-based ABI probe caught bool and was blind
# to int32).
#
# So this table is INTERIM and scoped to one function. **Delete it when
# chelis#893's seal lands** - a tripwire that outlives its compile-time
# replacement is the "it's documented" floor wearing a test's clothes.
#
# Related, and deliberately not duplicated here: chelis#730 §C6.2 names
# these same decode sites (including "tensor formatting") as a normative
# consumer row, and chelis#895 is the general mechanism for binding such a
# row to an executable check or an explicit deferral. This table is the
# per-arm decode check for chelis#732's exit only; §C6.2's other consumers
# (cmplt, where, scatter-add, cumsum, trace, clamp, einsum) are chelis#894's
# and are NOT covered by it.
OBSERVATION_DECODE_TABLE: tuple[tuple[str, str, str], ...] = (
    ("F32", "f32::data_ptr_unchecked", "typed accessor"),
    ("F64", "f64::data_ptr_unchecked", "typed accessor"),
    ("I64", "i64::data_ptr_unchecked", "typed accessor"),
    ("I32", "i32::data_ptr_unchecked", "typed accessor"),
    ("I16", "i16::data_ptr_unchecked", "typed accessor"),
    ("I8", "i8::data_ptr_unchecked", "typed accessor"),
    (
        "Bool",
        "data_as_f32_const",
        "DECLARED EXCEPTION: bool storage IS f32-encoded today "
        "(`read_index_slot` keeps `F32 | Bool` on the same view). "
        "chelis#894 migrates it to `Repr::Bool8`; this arm moves WITH "
        "that change - a 1-byte read against 4-byte writers is a "
        "misdecode in the other direction",
    ),
    (
        "Bf16",
        "*const u16",
        "raw 2-byte bit read: `half::bf16` has no `TensorElement` impl, "
        "so the bits are decoded explicitly via `from_bits`",
    ),
    (
        "F16",
        "*const u16",
        "raw 2-byte bit read: `half::f16` has no `TensorElement` impl, "
        "so the bits are decoded explicitly via `from_bits`",
    ),
)

# Obligation 7: the only permitted c-format-narrowing allowlist row. The
# two production rows (host_emit.rs = 4, chelis-runtime lib.rs = 1) were
# deleted at Phase 2 as the Phase 0 handoff promised.
PERMITTED_FORMAT_NARROWING_PATHS: frozenset[str] = frozenset(
    {"crates/chelis-backend-c/src/lib.rs"}
)

# Obligation 7, the Rust precision class (2026-07-30): every permitted row
# is an annotated NON-EXIT carrier (test assertion messages, tooling
# eprintln reports, UX displays). A new PATH here means a precision spec
# appeared in a file that never had one - review it as a possible third
# formatter before extending either side.
PERMITTED_RUST_FORMAT_NARROWING_PATHS: frozenset[str] = frozenset(
    {
        "crates/chelis-backend-c/src/lib.rs",
        "crates/chelis-backend-hip/src/emit.rs",
        "crates/chelis-cli/src/main.rs",
        "crates/chelis-cove/src/live.rs",
        "crates/chelis-e2e/src/bench.rs",
        "crates/chelis-e2e/src/bin/train_mnist.rs",
        "crates/chelis-ir/src/grad.rs",
        "crates/chelis-prove/src/bin/certify_erf_envelope.rs",
        "crates/chelis-prove/src/bin/certify_special_fn_envelope.rs",
        "crates/chelis-prove/src/opaque.rs",
    }
)

# Obligation 7, the Rust Debug class (2026-07-30): the first two rows ARE
# the sanctioned formatters (`{:?}` is the normative grammar's definition
# there); the compiler-api rows are the declared derived-Debug residue
# carriers per faithful_observation.md §B2.4.
PERMITTED_RUST_DEBUG_FORMAT_PATHS: frozenset[str] = frozenset(
    {
        "crates/chelis-types/src/observation.rs",
        "crates/chelis-runtime/src/format_shortest.rs",
        "crates/chelis-compiler-api/src/runtime/host_ops.rs",
        "crates/chelis-compiler-api/src/runtime/eval.rs",
        "crates/chelis-compiler-api/src/runtime/invariant.rs",
        "crates/chelis-compiler-api/src/runtime/named_axis.rs",
        "crates/chelis-compiler-api/src/runtime/tests.rs",
        "crates/chelis-compiler-api/src/runtime/transforms.rs",
    }
)

# The no-third-formatter classes and their permitted baseline path sets.
# §B2.8's doc-citation parity leg requires this table to equal the set of
# tripwire patterns whose `doc()` cites faithful_observation.md, so a
# hosted detector this oracle cannot see is unrepresentable.
FORMAT_CLASS_TABLE: tuple[tuple[str, str, frozenset[str]], ...] = (
    ("CFormatNarrowing", "c-format-narrowing", PERMITTED_FORMAT_NARROWING_PATHS),
    (
        "RustFormatNarrowing",
        "rust-format-narrowing",
        PERMITTED_RUST_FORMAT_NARROWING_PATHS,
    ),
    (
        "RustDebugNumericFormat",
        "rust-debug-numeric-format",
        PERMITTED_RUST_DEBUG_FORMAT_PATHS,
    ),
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
    for const_name, owner, _probes, declared in DECLARED_EXCLUSIONS:
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


def test_fn_body(source: str, name: str) -> str | None:
    """The body of a top-level `fn name()` (through its closing `\\n}`)."""

    match = re.search(r"fn " + re.escape(name) + r"\(\)(.*?)\n\}", source, re.S)
    return None if match is None else match.group(1)


def exclusion_probe_violations(source: str) -> list[str]:
    """§B2.9 legs 2-3, structurally: each exclusion's probes exist, are
    NOT `#[ignore]`d (an ignored probe is a disabled re-execution leg),
    iterate the list constant itself (a probe rewritten over hard-coded
    labels loses inventory parity silently), and carry the
    shrink-protocol message naming this ledger."""

    ignored = ignored_cells(source)
    violations: list[str] = []
    for const_name, owner, probes, _declared in DECLARED_EXCLUSIONS:
        if not probes:
            violations.append(
                f"{const_name}: no exclusion probe declared. §B2.9: a declared "
                "boundary that is not re-executed may not exist."
            )
            continue
        for probe in probes:
            if not defines_test(source, probe):
                violations.append(
                    f"{const_name} [{owner}]: exclusion probe {probe} is not "
                    "defined in the harness. The re-execution leg is gone; "
                    "restore the probe or delete the exclusion with its rows."
                )
                continue
            if probe in ignored:
                violations.append(
                    f"{const_name}: exclusion probe {probe} is `#[ignore]`d - "
                    "a disabled re-execution leg. Probes run in the DEFAULT "
                    "suite; if the probe is failing, that is the §B2.9 signal "
                    "working (shrink the list or file the new defect), not a "
                    "flake to silence."
                )
                continue
            body = test_fn_body(source, probe)
            if body is None:
                violations.append(
                    f"{const_name}: exclusion probe {probe}'s body could not "
                    "be extracted; the probe-shape checks cannot run."
                )
                continue
            if const_name not in body:
                violations.append(
                    f"{const_name}: probe {probe} no longer references the "
                    "list constant. Probes iterate the exclusion const itself "
                    "so shrinking the list shrinks the probe in the same edit "
                    "- hard-coded labels drift."
                )
            if "DECLARED_EXCLUSIONS" not in body:
                violations.append(
                    f"{const_name}: probe {probe} no longer carries the "
                    "shrink-protocol message naming DECLARED_EXCLUSIONS; the "
                    "gone-green failure must say what to do."
                )
    return violations


def cross_lane_lock_body(source: str) -> str | None:
    match = re.search(
        r"fn cross_lane_stdout_is_byte_identical_where_bits_agree\(\)(.*?)\n\}",
        source,
        re.S,
    )
    return None if match is None else match.group(1)


def cross_lane_corpus_programs(source: str) -> dict[str, str] | None:
    """Map each program label to its program text inside the §C2.3 lock."""

    body = cross_lane_lock_body(source)
    if body is None:
        return None
    programs = re.search(r"let programs:[^=]*=\s*&\[(.*?)\n    \];", body, re.S)
    if programs is None:
        return None
    return split_tuple_entries(programs.group(1))


def split_tuple_entries(text: str) -> dict[str, str]:
    """Split a Rust array of `("label", program)` tuples by paren depth.

    A regex split cannot do this: `("int8-exits", int_table_program("int8",
    ...))` contains an INNER `("int8",` that matches the same shape, so a
    pattern scan cuts each entry in half and leaves the label sitting alone
    with no program to check. Depth tracking (with string literals skipped,
    since the programs are Rust strings full of parentheses) gives the real
    top-level entries.
    """

    entries: dict[str, str] = {}
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "(":
            if depth == 0:
                start = index
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0 and start is not None:
                entry = text[start : index + 1]
                label = re.search(r'"([a-z0-9\-]+)"', entry)
                if label is not None:
                    entries[label.group(1)] = entry
                start = None
    return entries


def cross_lane_corpus_labels(source: str) -> list[str] | None:
    entries = cross_lane_corpus_programs(source)
    return None if entries is None else list(entries)


def cross_lane_corpus_violations(source: str) -> list[str]:
    entries = cross_lane_corpus_programs(source)
    if entries is None:
        return [
            "cross_lane_stdout_is_byte_identical_where_bits_agree: the §C2.3 "
            "byte-identity lock or its program list could not be parsed; the "
            "corpus floor cannot be checked."
        ]
    violations: list[str] = []
    missing = [label for label, _ in CROSS_LANE_CORPUS_FLOOR if label not in entries]
    if missing:
        violations.append(
            "cross_lane_stdout_is_byte_identical_where_bits_agree: the §C2.3 "
            f"corpus lost {missing}. Phase 2 freezes cross-lane byte equality; "
            "the corpus may grow but never shrink."
        )
    # A label that survives while its program is gutted is the same loss
    # wearing the right name.
    for label, required in CROSS_LANE_CORPUS_FLOOR:
        program = entries.get(label)
        if program is None:
            continue
        # Check the PROGRAM, not the label: `bool-exits` contains "bool",
        # so a token scan over the whole entry is satisfied by the name it
        # is supposed to be validating.
        program = program.replace(f'"{label}"', "", 1)
        absent = [token for token in required if token not in program]
        if absent:
            violations.append(
                f"cross_lane_stdout_is_byte_identical_where_bits_agree: the "
                f"`{label}` program no longer mentions {absent}. The label is "
                "intact but the body no longer covers what it is named for - "
                "a corpus row may be strengthened, never hollowed out."
            )
    body = cross_lane_lock_body(source) or ""
    lacking = [token for token in CROSS_LANE_REQUIRED_ASSERTIONS if token not in body]
    if lacking:
        violations.append(
            "cross_lane_stdout_is_byte_identical_where_bits_agree: the lock no "
            f"longer references {lacking}. §C2.3 is FULL-stdout byte equality "
            "across both lanes; a corpus of the right size proves nothing if "
            "the assertion under it weakens."
        )
    return violations


_LINE_COMMENT = re.compile(r"//.*?$", re.M)
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)

# Every typed accessor a decode arm could reach for. The arm's CODE must
# contain its own and no other: a decode that reads through a foreign
# width is the defect, whether it arrives by drift or by a comment that
# says the right thing while the body does the wrong one.
_TYPED_ACCESSORS: tuple[str, ...] = (
    "f32::data_ptr_unchecked",
    "f64::data_ptr_unchecked",
    "i8::data_ptr_unchecked",
    "i16::data_ptr_unchecked",
    "i32::data_ptr_unchecked",
    "i64::data_ptr_unchecked",
)


def strip_comments(text: str) -> str:
    """Rust source with `//` and `/* */` comments removed.

    The decode table is a source-text guard, so it must scan CODE. A
    red-team probe put `i16::data_ptr_unchecked` in a comment while the
    body read through `i32::data_ptr_unchecked`, and the substring check
    that preceded this passed the whole oracle - a wrong reader certified
    green. Comments are stripped before any arm is inspected.
    """

    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", text))


def observation_decode_arms(source: str) -> dict[str, str] | None:
    """Map each dtype arm of `tensor_elem_to_string` to its arm body."""

    match = re.search(
        r"unsafe fn tensor_elem_to_string\([^)]*\)[^{]*\{(.*?)\n\}", source, re.S
    )
    if match is None:
        return None
    body = match.group(1)
    arms: dict[str, str] = {}
    starts = [(m.group(1), m.start()) for m in re.finditer(r"RuntimeDType::(\w+) =>", body)]
    for index, (name, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(body)
        arms[name] = body[start:end]
    return arms


def observation_decode_violations(source: str) -> list[str]:
    arms = observation_decode_arms(source)
    if arms is None:
        return [
            "tensor_elem_to_string: not found in the runtime; the observation "
            "decode table cannot be checked."
        ]
    violations: list[str] = []
    declared = {name for name, _, _ in OBSERVATION_DECODE_TABLE}
    for name in sorted(set(arms) - declared):
        violations.append(
            f"tensor_elem_to_string: dtype arm {name} has no row in "
            "OBSERVATION_DECODE_TABLE. Every observation decode is declared - "
            "add the row naming the pointer view and why."
        )
    for name, expected, why in OBSERVATION_DECODE_TABLE:
        arm = arms.get(name)
        if arm is None:
            violations.append(
                f"tensor_elem_to_string: declared dtype arm {name} is missing "
                "from the match."
            )
            continue
        # CODE only - a comment naming the right accessor while the body
        # reads through another is exactly the wrong-reader case.
        code = strip_comments(arm)
        if expected not in code:
            violations.append(
                f"tensor_elem_to_string: the {name} arm no longer decodes "
                f"through `{expected}` ({why}). A changed pointer view at an "
                "observation exit is a misdecode until proven otherwise - this "
                "is the chelis#894 int32 shape, where native storage was read "
                "through an f32 view while sibling exits read it correctly."
            )
        # Presence is not enough: the arm must use its OWN accessor and no
        # other. Without this, a body reading a foreign width passes as
        # long as the declared token appears somewhere.
        foreign = [
            accessor
            for accessor in _TYPED_ACCESSORS
            if accessor != expected and accessor in code
        ]
        if foreign:
            violations.append(
                f"tensor_elem_to_string: the {name} arm reads through "
                f"{foreign} as well as (or instead of) its declared "
                f"`{expected}`. An arm decodes at its OWN dtype's width and no "
                "other; a foreign accessor here renders one dtype's bytes as "
                "another's, which is the whole defect class."
            )
    # The f32 view is the exact mechanism of the int32 defect class; only
    # the declared exception may reach for it.
    for name, arm in arms.items():
        if "data_as_f32_const" in strip_comments(arm) and name != "Bool":
            violations.append(
                f"tensor_elem_to_string: the {name} arm reaches for the "
                "untyped f32 view. Only the declared Bool exception may, and "
                "only until chelis#894's `Repr::Bool8` migration."
            )
    return violations


RETIRED_EXPORTS: tuple[str, ...] = ("chelis_print_f32",)


def rust_exported_symbols(source: str) -> set[str]:
    """`#[no_mangle]` `extern "C"` function names, comments excluded."""

    code = strip_comments(source)
    return set(
        re.findall(
            r'#\[no_mangle\]\s*(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+([A-Za-z0-9_]+)',
            code,
        )
    )


def header_declared_functions(header: str) -> set[str]:
    """Function names DECLARED in the C header, comments excluded."""

    code = strip_comments(header)
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_ \*]*\b([A-Za-z0-9_]+)\s*\([^;{]*\)\s*;", code))


def dead_export_violations(source: str, header: str) -> list[str]:
    """A retired public exit must be gone from the ABI, not just the text.

    Substring presence was the first cut and PR #863 R2 flagged it (LOW):
    it cannot tell a live declaration from a comment explaining the
    removal - a false positive on documentation and a false negative on a
    declaration spelled differently. Both surfaces are parsed instead.
    """

    exported = rust_exported_symbols(source)
    declared = header_declared_functions(header)
    violations: list[str] = []
    for name in RETIRED_EXPORTS:
        if name in exported:
            violations.append(
                f"chelis-runtime: `{name}` is exported again. It was a public "
                "`#[no_mangle]` tensor print with zero emitters - an "
                "observation exit no program could reach, which is how its "
                "int32 misdecode stayed uncensused. A public exit owes exit "
                "coverage or does not exist (§B2.7)."
            )
        if name in declared:
            violations.append(
                f"chelis_runtime.h: `{name}` is declared again; the published "
                "C ABI must not re-export a retired exit."
            )
    return violations


def format_narrowing_allowlist(
    source: str, variant: str = "CFormatNarrowing"
) -> list[tuple[str, int]]:
    rows = re.findall(
        r"Pat::" + re.escape(variant) + r"\s*,\s*\"([^\"]+)\"\s*,\s*(\d+)\s*,",
        source,
    )
    return [(path, int(count)) for path, count in rows]


def format_narrowing_violations(source: str) -> list[str]:
    violations: list[str] = []
    for variant, class_id, permitted in FORMAT_CLASS_TABLE:
        for path, count in format_narrowing_allowlist(source, variant):
            if path not in permitted:
                violations.append(
                    f"{path}: a {class_id} allowlist row of {count} outside "
                    "the permitted set. A new baseline path is a possible "
                    "third formatter (§B2.4, a review-blocking finding); if "
                    "review clears it as a non-exit carrier, extend the "
                    "permitted set here in the same change set - the "
                    "two-instrument interlock is deliberate."
                )
    return violations


_DOC_ARM = re.compile(
    r"((?:Pat::\w+\s*\|\s*)*Pat::\w+|_)\s*=>\s*\{?\s*\"([^\"]+)\"",
)


def tripwire_doc_arms(source: str) -> list[tuple[tuple[str, ...], str]] | None:
    """Parse `fn doc()`'s match arms into (variant names, citation) pairs.

    Returns None when the function cannot be located; a located-but-empty
    parse is reported by the caller as non-triviality failure (a silently
    empty parse would otherwise pass every check).
    """

    match = re.search(r"fn doc\(self\) -> &'static str \{(.*?)\n    \}", source, re.S)
    if match is None:
        return None
    arms: list[tuple[tuple[str, ...], str]] = []
    for patterns, citation in _DOC_ARM.findall(match.group(1)):
        names = tuple(re.findall(r"Pat::(\w+)", patterns))
        arms.append((names, citation))
    return arms


def doc_citation_violations(source: str) -> list[str]:
    """§B2.8's doc-citation parity: tripwire patterns citing this plan's
    doc and FORMAT_CLASS_TABLE must be the same set, both ways."""

    arms = tripwire_doc_arms(source)
    if arms is None:
        return [
            "loud_unsupported_tripwire.rs: fn doc() could not be located; the "
            "doc-citation parity check cannot run."
        ]
    if not arms or not any("loud_unsupported.md" in cite for _, cite in arms):
        return [
            "loud_unsupported_tripwire.rs: fn doc() parsed to nothing "
            "recognizable (no loud_unsupported.md arm). The parser has gone "
            "non-trivial-blind; fix it before trusting parity."
        ]
    cited = {
        name
        for names, citation in arms
        if "faithful_observation.md" in citation
        for name in names
    }
    covered = {variant for variant, _, _ in FORMAT_CLASS_TABLE}
    violations: list[str] = []
    for name in sorted(cited - covered):
        violations.append(
            f"Pat::{name} cites faithful_observation.md in doc() but has no "
            "FORMAT_CLASS_TABLE row - a hosted detector this oracle cannot "
            "see. Add its permitted-path row here in the same change set "
            "(§B2.8)."
        )
    for name in sorted(covered - cited):
        violations.append(
            f"FORMAT_CLASS_TABLE covers Pat::{name} but the tripwire's doc() "
            "no longer cites faithful_observation.md for it - a stale "
            "coverage row or a detector that changed owner. Reconcile both "
            "sides deliberately."
        )
    return violations


# §B2.8's rule-instrument manifest: one row per §B2 item of the design
# doc - (item number, a fragment of the item's bold title, instruments).
# An instrument is an oracle function name (checked against this module),
# `suite:<label>` naming a GREEN_SUITES entry, or `review-rule: <why>` -
# an explicit, justified decision that the rule's enforcement is human
# review. The point is not that every rule gets a mechanical check; it is
# that NO rule gets to exist without the decision being recorded and
# tripwired against the doc drifting away from it.
B2_RULE_INSTRUMENTS: tuple[tuple[int, str, tuple[str, ...]], ...] = (
    (
        1,
        "one-time migration carve-out",
        (
            "review-rule: the migration completed at Phases 1-2; its proof "
            "was the round-trip harness surviving both §B2.1 PRs unchanged",
        ),
    ),
    (
        2,
        "Bits before text",
        (
            "review-rule: bit-level companions are corpus content, checked "
            "where the migration touched expectations",
        ),
    ),
    (
        3,
        "Red-to-green only by un-ignoring",
        ("ledger_violations", "classify_red_run"),
    ),
    (
        4,
        "No third formatter",
        (
            "format_narrowing_violations",
            "doc_citation_violations",
            "suite:the §B2.4 no-third-formatter tripwire",
        ),
    ),
    (
        5,
        "Discoveries fork",
        (
            "review-rule: filing and censusing are process acts; the "
            "append-only census lives on the tracking issue",
        ),
    ),
    (
        6,
        "No untyped decode at an exit",
        ("observation_decode_violations",),
    ),
    (
        7,
        "A public exit owes exit coverage",
        ("dead_export_violations",),
    ),
    (
        8,
        "Detector-scope parity",
        ("b2_manifest_violations", "doc_citation_violations"),
    ),
    (
        9,
        "Three-legged boundaries",
        ("exclusion_violations", "exclusion_probe_violations"),
    ),
)


def b2_rule_items(doc: str) -> list[tuple[int, str]] | None:
    """The §B2 section's numbered items as (number, bold title) pairs."""

    match = re.search(r"^## B2\..*?$(.*?)^## ", doc, re.S | re.M)
    if match is None:
        return None
    return [
        (int(number), title)
        for number, title in re.findall(r"^(\d+)\. \*\*(.+?)\*\*", match.group(1), re.M)
    ]


def b2_manifest_violations(doc: str) -> list[str]:
    items = b2_rule_items(doc)
    if items is None:
        return [
            "spec/design/faithful_observation.md: the §B2 section could not "
            "be located; the rule-instrument manifest cannot run."
        ]
    if len(items) < 7:
        return [
            f"spec/design/faithful_observation.md: the §B2 parse found only "
            f"{len(items)} items where at least 7 are known to exist - the "
            "parser has gone non-trivial-blind; fix it before trusting the "
            "manifest."
        ]
    violations: list[str] = []
    titles = dict(items)
    manifest_numbers = {number for number, _, _ in B2_RULE_INSTRUMENTS}
    for number in sorted(set(titles) - manifest_numbers):
        violations.append(
            f"§B2.{number} ({titles[number]!r}) has no B2_RULE_INSTRUMENTS "
            "row. A §B2 rule lands only with a deliberate instrument decision "
            "(§B2.8) - name its check, its suite, or an explicit justified "
            "review-rule entry, in this same change set."
        )
    for number in sorted(manifest_numbers - set(titles)):
        violations.append(
            f"B2_RULE_INSTRUMENTS declares §B2.{number}, which the doc no "
            "longer has. Delete the manifest row with the rule, never before."
        )
    suite_labels = {label for label, _ in GREEN_SUITES}
    for number, fragment, instruments in B2_RULE_INSTRUMENTS:
        title = titles.get(number)
        if title is not None and fragment not in title:
            violations.append(
                f"§B2.{number}'s title {title!r} no longer contains its "
                f"manifest fragment {fragment!r} - the rule moved or was "
                "rewritten; re-bind the manifest deliberately."
            )
        for instrument in instruments:
            if instrument.startswith("review-rule"):
                if len(instrument.removeprefix("review-rule:").strip()) < 10:
                    violations.append(
                        f"§B2.{number}: a review-rule instrument needs its "
                        "justification spelled out, not a bare tag."
                    )
                continue
            if instrument.startswith("suite:"):
                label = instrument.removeprefix("suite:")
                if label not in suite_labels:
                    violations.append(
                        f"§B2.{number}: instrument {instrument!r} names no "
                        "GREEN_SUITES entry."
                    )
                continue
            if not callable(globals().get(instrument)):
                violations.append(
                    f"§B2.{number}: instrument `{instrument}` is not a "
                    "function in this oracle - the manifest names something "
                    "that cannot run."
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


def read_runtime_crate_sources(root: Path) -> str:
    """Every `.rs` file in the runtime crate, concatenated."""

    files = sorted((root / RUNTIME_SRC_DIR).rglob("*.rs"))
    if not files:
        raise OracleFailure(f"no runtime sources under {RUNTIME_SRC_DIR}")
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def read_sources() -> dict[Path, str]:
    sources: dict[Path, str] = {}
    for relative in (
        HARNESS_SOURCE,
        NARROW_MATRIX_SOURCE,
        REDUCTION_MATRIX_SOURCE,
        TRIPWIRE_SOURCE,
        RUNTIME_SOURCE,
        RUNTIME_HEADER,
        DESIGN_DOC,
    ):
        try:
            sources[relative] = (REPO_ROOT / relative).read_text(encoding="utf-8")
        except OSError as error:
            raise OracleFailure(f"unreadable oracle source {relative}: {error}") from error
    sources[RUNTIME_SRC_DIR] = read_runtime_crate_sources(REPO_ROOT)
    return sources


def run_structural_scan(sources: dict[Path, str]) -> None:
    violations: list[str] = []
    violations.extend(unignored_violations(sources))
    violations.extend(retired_lock_violations(sources))
    violations.extend(ledger_violations(sources[HARNESS_SOURCE], KNOWN_RED_CELLS))
    violations.extend(exclusion_violations(sources[HARNESS_SOURCE]))
    violations.extend(exclusion_probe_violations(sources[HARNESS_SOURCE]))
    violations.extend(cross_lane_corpus_violations(sources[HARNESS_SOURCE]))
    violations.extend(observation_decode_violations(sources[RUNTIME_SOURCE]))
    violations.extend(
        dead_export_violations(sources[RUNTIME_SRC_DIR], sources[RUNTIME_HEADER])
    )
    violations.extend(format_narrowing_violations(sources[TRIPWIRE_SOURCE]))
    violations.extend(doc_citation_violations(sources[TRIPWIRE_SOURCE]))
    violations.extend(b2_manifest_violations(sources[DESIGN_DOC]))
    if violations:
        raise OracleFailure("structural scan failed:\n" + "\n".join(violations))
    probe_count = sum(len(probes) for _, _, probes, _ in DECLARED_EXCLUSIONS)
    print(
        "+ structural scan: un-ignored oracle rows present, interim locks "
        f"retired, {len(KNOWN_RED_CELLS)} known-red cells declared and cited, "
        f"corpus exclusions unchanged with {probe_count} non-ignored "
        "re-execution probes bound, the §C2.3 cross-lane corpus floor "
        f"({len(CROSS_LANE_CORPUS_FLOOR)} programs) intact, all "
        f"{len(OBSERVATION_DECODE_TABLE)} observation decode arms on their "
        f"declared pointer views, no zero-emitter public exit, all "
        f"{len(FORMAT_CLASS_TABLE)} no-third-formatter classes inside their "
        "permitted sets with doc-citation parity, and the §B2 "
        f"rule-instrument manifest bound ({len(B2_RULE_INSTRUMENTS)} rules)",
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
