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
   Hardened per PR #962's red team (round 1 F2, round 2 M1): each probe
   must carry exactly the unconditional `#[test]` attribute (a
   `cfg_attr` ignore or a cfg gate fails the scan, and unrecognized
   attribute shapes fail closed); the oracle RUNS each probe
   individually, requiring exactly one executed test plus an ORDERED
   receipt equal to the probe's declared sequence (multiplicity and
   order compared); and - because receipts are probe-authored text a
   probe could forge - `run_exclusion_ground_truth` re-executes every
   excluded behavior ITSELF: the oracle writes the per-label programs,
   runs eval / the C emitter, and re-derives each exclusion's
   fingerprint (the chelis#717 F32 narrowing, the chelis#751
   bare-integer-literal emission, the signless -0.0 spelling) without
   trusting any probe output. That oracle-owned leg is the
   independently observable re-execution; the probes remain the
   continuous CI leg.
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
   item list must equal `B2_RULE_INSTRUMENTS` (numbers and titles),
   every named instrument must exist, and - checked at the END of the
   run - every callable instrument must have produced a RUNTIME
   invocation receipt AND a centrally consumed-result receipt (the
   `@instrument` decorator records execution; `consume_findings` records
   only after extending a verdict sink; suite legs record both on
   success). Source-text scanning was demonstrated to certify an
   `if False:` branch, and entry-only receipts were demonstrated to
   certify a non-empty result discarded by its caller (PR #962 round-2
   M2 and exact-head F2). Review-rule entries need a substantive
   justification. The manifest binds names, execution, and verdict
   consumption - per-instrument mutation tests in
   test_faithful_observation_phase2_oracle.py are for, one per failure
   mode.

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

import functools
import math
import os
from pathlib import Path
import re
import shlex
import struct
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
# equality check against the row labels (either direction). Legs 2-3 come
# from TWO mechanisms with different trust models (PR #962 round-2 M1):
#
# - the `probes`: NON-ignored harness tests, each with its declared
#   ordered receipt sequence. Receipts are probe-authored text - they
#   bind ordering and coverage of the CI leg but a probe could forge
#   them, so they are never the independence evidence;
# - the `rows`: per-label ground truth (dtype, the constructing Chelis
#   element expression, the exact intended value) that
#   `run_exclusion_ground_truth` re-executes ITSELF - the oracle builds
#   the programs, runs the toolchain, compiles/runs executable C rows,
#   compares intended bits, and re-derives each exclusion's fingerprint
#   without trusting any probe output. That leg is the
#   independently observable re-execution §B2.9 requires.
#
# A list may not exist here without probes and rows (§B2.9: a boundary
# that cannot be re-executed may not exist).
#
# 2026-07-30: `f64-tenth` left EVAL_F64_LIST_EXCLUDED on the probes' first
# execution - its F32-narrowed image renders `0.1`, text-coincident with
# the original f64, so the text assertion the list guards passes for it.
# The stale row is precisely what the pre-probe equality-only check could
# never catch.
DECLARED_EXCLUSIONS: tuple[
    tuple[
        str,
        str,
        tuple[tuple[str, tuple[str, ...]], ...],
        tuple[tuple[str, str, str, str], ...],
    ],
    ...,
] = (
    (
        "C_LANE_EXCLUDED",
        "chelis#751 (C constant-emission ingress; rows are unconstructible, not red)",
        (
            (
                "c_lane_excluded_labels_still_fail_at_ingress",
                ("f64-max", "f64-audit-e19", "f32-max"),
            ),
            (
                "c_lane_excluded_neg_zero_still_drops_the_sign",
                ("f64-neg-zero",),
            ),
        ),
        (
            ("f64-neg-zero", "f64", "cast(-0.0, f64)", "-0.0"),
            (
                "f64-max",
                "f64",
                "cast(1.7976931348623157e308, f64)",
                "1.7976931348623157e308",
            ),
            (
                "f64-audit-e19",
                "f64",
                "cast(9.999999980506448e19, f64)",
                "9.999999980506448e19",
            ),
            ("f32-max", "f32", "3.4028234663852886e38", "3.4028234663852886e38"),
        ),
    ),
    (
        "EVAL_F64_LIST_EXCLUDED",
        "chelis#717 (eval's to_list narrows through the stale F32 tag)",
        (
            (
                "eval_f64_list_excluded_rows_still_narrow_through_the_f32_tag",
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
        ),
        (
            (
                "f64-max",
                "f64",
                "cast(1.7976931348623157e308, f64)",
                "1.7976931348623157e308",
            ),
            ("f64-min-subnormal", "f64", "cast(5e-324, f64)", "5e-324"),
            (
                "f64-min-normal",
                "f64",
                "cast(2.2250738585072014e-308, f64)",
                "2.2250738585072014e-308",
            ),
            (
                "f64-17-digit",
                "f64",
                "cast(0.30000000000000004, f64)",
                "0.30000000000000004",
            ),
            ("f64-2p53", "f64", "cast(9007199254740992.0, f64)", "9007199254740992.0"),
            (
                "f64-2p53-plus-2",
                "f64",
                "cast(9007199254740994.0, f64)",
                "9007199254740994.0",
            ),
            (
                "f64-audit-e19",
                "f64",
                "cast(9.999999980506448e19, f64)",
                "9.999999980506448e19",
            ),
        ),
    ),
)


def exclusion_labels(rows: Sequence[tuple[str, str, str, str]]) -> tuple[str, ...]:
    return tuple(label for label, _dt, _elem, _value in rows)

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
        "crates/chelis-compiler-api/src/context.rs",
        "crates/chelis-cove/src/live.rs",
        "crates/chelis-deep/src/ast.rs",
        "crates/chelis-e2e/src/bench.rs",
        "crates/chelis-e2e/src/bin/train_mnist.rs",
        "crates/chelis-ir/src/grad.rs",
        "crates/chelis-ir/src/lower.rs",
        "crates/chelis-prove/src/bin/certify_erf_envelope.rs",
        "crates/chelis-prove/src/bin/certify_special_fn_envelope.rs",
        "crates/chelis-prove/src/opaque.rs",
        "crates/chelis-runtime/src/format_shortest.rs",
        "crates/chelis-types/src/infer.rs",
        "crates/chelis-types/src/observation.rs",
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


# Runtime entry and result-consumption receipts. Entry alone is insufficient:
# exact-head F2 demonstrated a detector returning a violation that its caller
# discarded. The final check requires both sets.
INVOKED_INSTRUMENTS: set[str] = set()
CONSUMED_INSTRUMENTS: set[str] = set()


def instrument(fn):
    """Record fn's execution in INVOKED_INSTRUMENTS when it runs."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        INVOKED_INSTRUMENTS.add(fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


def consume_findings(sink: list[str], fn, *args) -> None:
    """Run one list-valued instrument and connect every result to verdict.

    The consumed receipt is written only after this helper extends the
    caller's verdict sink. Calling a decorated detector directly can prove
    entry, but cannot forge result consumption.
    """

    findings = fn(*args)
    sink.extend(findings)
    CONSUMED_INSTRUMENTS.add(fn.__name__)


def consume_optional_finding(sink: list[str], fn, *args) -> None:
    """Consume an instrument returning either one violation or ``None``."""

    finding = fn(*args)
    if finding is not None:
        sink.append(finding)
    CONSUMED_INSTRUMENTS.add(fn.__name__)


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def f32_bits(value: float) -> int:
    """Bits of value narrowed to f32, saturating to the infinities the
    way Rust `as f32` does (struct.pack raises on out-of-range instead)."""

    try:
        packed = struct.pack("<f", value)
    except OverflowError:
        packed = struct.pack("<f", math.inf if value > 0 else -math.inf)
    return struct.unpack("<I", packed)[0]


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


@instrument
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


@instrument
def exclusion_violations(source: str) -> list[str]:
    violations: list[str] = []
    for const_name, owner, _probes, rows in DECLARED_EXCLUSIONS:
        declared = exclusion_labels(rows)
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


def probe_attributes(source: str, name: str) -> list[str] | None:
    """The attribute lines bound to `fn name(`, outermost first.

    Walks upward through single-line attributes and comments. A
    multi-line attribute (its closing `)]` on its own line) stops the
    walk early, which FAILS CLOSED: the collected list then cannot equal
    the required `["#[test]"]`, so an attribute shape this parser cannot
    vouch for is a violation, never a pass. None when the fn is absent.
    """

    lines = source.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"\s*fn\s+" + re.escape(name) + r"\s*\(", line) is None:
            continue
        attrs: list[str] = []
        walk = index - 1
        while walk >= 0:
            stripped = lines[walk].strip()
            if stripped.startswith("#["):
                attrs.append(stripped)
            elif stripped.startswith("//"):
                pass
            else:
                break
            walk -= 1
        attrs.reverse()
        return attrs
    return None


@instrument
def probe_attribute_violations(source: str) -> list[str]:
    """PR #962 red-team F2: a probe must be an unconditional `#[test]` -
    exactly that attribute and nothing else. `#[cfg_attr(all(), ignore)]`
    disables a probe without matching the ignore scan; a `#[cfg(...)]`
    compiles it away; any unrecognized or multi-line attribute shape
    fails closed."""

    violations: list[str] = []
    for const_name, _owner, probes, _rows in DECLARED_EXCLUSIONS:
        for probe, _expected in probes:
            attrs = probe_attributes(source, probe)
            if attrs is None:
                continue  # absence is exclusion_probe_violations' finding
            if attrs != ["#[test]"]:
                violations.append(
                    f"{const_name}: probe {probe} must carry exactly the "
                    f"unconditional `#[test]` attribute; found {attrs!r}. "
                    "Conditional or additional attributes (cfg, cfg_attr, "
                    "ignore) can disable the re-execution leg without "
                    "tripping the ignore scan."
                )
    return violations


@instrument
def exclusion_probe_violations(source: str) -> list[str]:
    """§B2.9 CI-leg structure: each exclusion's probes exist, are NOT
    `#[ignore]`d (an ignored probe is a disabled leg), iterate the list
    constant itself, carry the shrink-protocol message naming this
    ledger, and between them are assigned every declared label. These
    are source-shape checks on the probe-authored leg; the independent
    re-execution evidence is `run_exclusion_ground_truth`."""

    ignored = ignored_cells(source)
    violations: list[str] = []
    for const_name, owner, probes, rows in DECLARED_EXCLUSIONS:
        if not probes:
            violations.append(
                f"{const_name}: no exclusion probe declared. §B2.9: a declared "
                "boundary that is not re-executed may not exist."
            )
            continue
        assigned = [label for _probe, expected in probes for label in expected]
        if sorted(assigned) != sorted(exclusion_labels(rows)):
            violations.append(
                f"{const_name}: the probes' declared receipt sequences cover "
                f"{sorted(set(assigned))} but the ledger declares "
                f"{sorted(exclusion_labels(rows))}. Every label is assigned to "
                "exactly one probe's expected receipt."
            )
        for probe, _expected in probes:
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


@instrument
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


@instrument
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


@instrument
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


@instrument
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
        (
            "exclusion_violations",
            "exclusion_probe_violations",
            "probe_attribute_violations",
            "classify_probe_outputs",
            "eval_exclusion_ground_truth_violations",
            "c_exclusion_ground_truth_violations",
        ),
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


@instrument
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
        for entry in instruments:
            if entry.startswith("review-rule"):
                if len(entry.removeprefix("review-rule:").strip()) < 30:
                    violations.append(
                        f"§B2.{number}: a review-rule instrument needs its "
                        "justification spelled out, not a bare tag (PR #962 "
                        "red-team F5.2)."
                    )
                continue
            if entry.startswith("suite:"):
                label = entry.removeprefix("suite:")
                if label not in suite_labels:
                    violations.append(
                        f"§B2.{number}: instrument {entry!r} names no "
                        "GREEN_SUITES entry."
                    )
                continue
            if not callable(globals().get(entry)):
                violations.append(
                    f"§B2.{number}: instrument `{entry}` is not a "
                    "function in this oracle - the manifest names something "
                    "that cannot run."
                )
    return violations


def instrument_result_violations(
    invoked: set[str], consumed: set[str]
) -> list[str]:
    """Require both runtime entry and a centrally consumed detector result.

    A receipt written on function entry proves only that code ran. PR #962's
    exact-head review demonstrated a non-empty detector result discarded by
    its caller while every entry receipt remained green. Callable instruments
    therefore owe two independent facts: the decorator's runtime receipt and
    a receipt written only by ``consume_findings`` after the result reaches a
    verdict sink. Suite receipts are both invoked and consumed on success.
    """

    violations: list[str] = []
    for number, _fragment, instruments in B2_RULE_INSTRUMENTS:
        for entry in instruments:
            if entry.startswith("review-rule"):
                continue
            if entry not in invoked:
                violations.append(
                    f"§B2.{number}: instrument `{entry}` produced no runtime "
                    "invocation receipt during this oracle run - it is named "
                    "but never executed (manifest theater)."
                )
            elif entry not in consumed:
                violations.append(
                    f"§B2.{number}: instrument `{entry}` executed, but its "
                    "result was not consumed by the oracle verdict "
                    "(discarded-result detector theater)."
                )
    return violations


def instrument_invocation_violations(invoked: set[str]) -> list[str]:
    """Compatibility unit seam: treat every supplied invocation as consumed."""

    return instrument_result_violations(invoked, invoked)


_RECEIPT = re.compile(r"^exclusion probe (\w+) visited: (.*)$", re.M)


def probe_receipts(output: str) -> dict[str, list[str]]:
    """Parse `exclusion probe <CONST> visited: <labels>` receipt lines
    into ORDERED label lists, concatenated in line order. Order and
    multiplicity are preserved (PR #962 round-2 M1: the set-union form
    accepted `a a b b` for the declaration `["a", "b"]`)."""

    receipts: dict[str, list[str]] = {}
    for const_name, labels in _RECEIPT.findall(output):
        receipts.setdefault(const_name, []).extend(labels.split())
    return receipts


@instrument
def classify_probe_outputs(
    const_name: str,
    probe_runs: Sequence[tuple[str, Sequence[str], int, str]],
) -> list[str]:
    """The CI-leg receipt check: each probe run must have executed
    exactly one test, passed, and printed EXACTLY its declared ordered
    receipt sequence - order and multiplicity compared, so shrunken,
    duplicated, or reordered receipts fail. The receipt is still
    probe-authored text; a probe that forges the expected sequence while
    skipping rows passes THIS check and is caught by
    `run_exclusion_ground_truth`, which re-derives every label's
    fingerprint without trusting probe output (PR #962 round-2 M1)."""

    violations: list[str] = []
    for probe, expected, returncode, output in probe_runs:
        if returncode != 0:
            violations.append(
                f"{const_name}: probe {probe} FAILED (exit {returncode}). "
                "If the output carries a GOOD NEWS message, follow its "
                "shrink protocol; otherwise a probe leg broke. Output "
                f"tail:\n{output[-2000:]}"
            )
            continue
        if "running 1 test" not in output:
            violations.append(
                f"{const_name}: the {probe} run did not execute exactly "
                "one test (filter or harness drift) - a zero-test result "
                "is not a green probe."
            )
            continue
        receipt = probe_receipts(output).get(const_name, [])
        if receipt != list(expected):
            violations.append(
                f"{const_name}: probe {probe}'s ordered receipt "
                f"{receipt} does not equal its declared sequence "
                f"{list(expected)}. A shrunken, duplicated, or reordered "
                "receipt means the probe stopped visiting what the "
                "ledger declares."
            )
    return violations


@instrument
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
    checks = (
        (unignored_violations, (sources,)),
        (retired_lock_violations, (sources,)),
        (ledger_violations, (sources[HARNESS_SOURCE], KNOWN_RED_CELLS)),
        (exclusion_violations, (sources[HARNESS_SOURCE],)),
        (exclusion_probe_violations, (sources[HARNESS_SOURCE],)),
        (probe_attribute_violations, (sources[HARNESS_SOURCE],)),
        (cross_lane_corpus_violations, (sources[HARNESS_SOURCE],)),
        (observation_decode_violations, (sources[RUNTIME_SOURCE],)),
        (
            dead_export_violations,
            (sources[RUNTIME_SRC_DIR], sources[RUNTIME_HEADER]),
        ),
        (format_narrowing_violations, (sources[TRIPWIRE_SOURCE],)),
        (doc_citation_violations, (sources[TRIPWIRE_SOURCE],)),
        (b2_manifest_violations, (sources[DESIGN_DOC],)),
    )
    for check, args in checks:
        consume_findings(violations, check, *args)
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
        # The suite's runtime invocation receipt for the manifest.
        INVOKED_INSTRUMENTS.add(f"suite:{label}")
        CONSUMED_INSTRUMENTS.add(f"suite:{label}")


def run_exclusion_probes(env: dict[str, str]) -> None:
    violations: list[str] = []
    for const_name, _owner, probes, rows in DECLARED_EXCLUSIONS:
        probe_runs: list[tuple[str, Sequence[str], int, str]] = []
        for probe, expected in probes:
            command = (
                "cargo",
                "test",
                "-p",
                "chelis-cli",
                "--test",
                "observation_roundtrip_harness",
                "--",
                "--exact",
                probe,
                "--nocapture",
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
            probe_runs.append(
                (probe, expected, completed.returncode, completed.stdout + completed.stderr)
            )
        consume_findings(violations, classify_probe_outputs, const_name, probe_runs)
        if not violations:
            print(
                f"  probe receipts exact: {const_name} "
                f"({len(exclusion_labels(rows))} labels across {len(probes)} probe(s))",
                flush=True,
            )
    if violations:
        raise OracleFailure("exclusion probe receipts failed:\n" + "\n".join(violations))


def c_code_without_comments_or_literals(c_source: str) -> str:
    """Mask C comments plus string/character literals, preserving positions."""

    out = list(c_source)
    index = 0
    state = "code"
    while index < len(c_source):
        char = c_source[index]
        following = c_source[index + 1] if index + 1 < len(c_source) else ""
        if state == "code":
            if char == "/" and following == "/":
                out[index] = out[index + 1] = " "
                index += 2
                state = "line-comment"
                continue
            if char == "/" and following == "*":
                out[index] = out[index + 1] = " "
                index += 2
                state = "block-comment"
                continue
            if char == '"':
                out[index] = " "
                index += 1
                state = "string"
                continue
            if char == "'":
                out[index] = " "
                index += 1
                state = "char"
                continue
        elif state == "line-comment":
            if char == "\n":
                state = "code"
            else:
                out[index] = " "
            index += 1
            continue
        elif state == "block-comment":
            out[index] = " "
            if char == "*" and following == "/":
                out[index + 1] = " "
                index += 2
                state = "code"
                continue
            index += 1
            continue
        else:
            out[index] = " "
            if char == "\\" and following:
                out[index + 1] = " "
                index += 2
                continue
            delimiter = '"' if state == "string" else "'"
            if char == delimiter:
                state = "code"
            index += 1
            continue
        index += 1
    return "".join(out)


def c_has_bare_giant_integer_literal(c_source: str) -> bool:
    """Python replica of the harness's chelis#751 fingerprint: a digit
    run longer than i64::MAX's 19 digits, not followed by a decimal
    point or exponent - no legitimate C integer constant has that shape."""

    code = c_code_without_comments_or_literals(c_source)
    index = 0
    length = len(code)
    digits = set("0123456789")
    while index < length:
        if code[index] in digits:
            start = index
            while index < length and code[index] in digits:
                index += 1
            following = code[index] if index < length else ""
            if index - start > 19 and following not in (".", "e", "E"):
                return True
        else:
            index += 1
    return False


@instrument
def eval_exclusion_ground_truth_violations(
    texts: Sequence[str], rows: Sequence[tuple[str, str, str, str]]
) -> list[str]:
    """The chelis#717 fingerprint, re-derived by the ORACLE from a
    to_list render it produced itself: each excluded element's text must
    parse to the F32-narrowing of the intended value (any other shape is
    a different defect) and must NOT round-trip at f64 (when it does,
    the repair landed and the exclusion must shrink). No probe output is
    trusted anywhere in this leg (PR #962 round-2 M1)."""

    violations: list[str] = []
    if len(texts) != len(rows):
        return [
            f"eval ground truth: expected {len(rows)} rendered elements, "
            f"got {len(texts)} - the driver program and the ledger drifted."
        ]
    for (label, _dt, _elem, value_text), text in zip(rows, texts):
        value = float(value_text)
        try:
            rendered = float(text)
        except ValueError:
            violations.append(
                f"eval ground truth [{label}]: `{text}` is not a float - a "
                "different defect; file it per B2.5."
            )
            continue
        if f32_bits(rendered) != f32_bits(value):
            violations.append(
                f"eval ground truth [{label}]: `{text}` is not the F32-tag "
                "narrowing of the intended value - the declared chelis#717 "
                "fingerprint no longer matches; file the new defect per B2.5 "
                "before touching the exclusion."
            )
        elif f64_bits(rendered) == f64_bits(value):
            violations.append(
                f"eval ground truth [{label}]: the to_list render now "
                "round-trips at f64 - the chelis#717/[#729] repair landed (or "
                "the row went text-coincident). Remove the label from "
                "EVAL_F64_LIST_EXCLUDED and DECLARED_EXCLUSIONS in one change "
                "set."
            )
    return violations


class CExclusionGroundTruth:
    """One oracle-owned C emission plus its independently observed behavior."""

    def __init__(
        self, label: str, build_ok: bool, c_source: str, native_status: str
    ) -> None:
        self.label = label
        self.build_ok = build_ok
        self.c_source = c_source
        self.native_status = native_status


@instrument
def c_exclusion_ground_truth_violations(
    entries: Sequence[CExclusionGroundTruth],
) -> list[str]:
    """Judge C exclusions from source reason AND independently run behavior."""

    violations: list[str] = []
    for entry in entries:
        label = entry.label
        if not entry.build_ok:
            violations.append(
                f"C ground truth [{label}]: `chelis build` failed - the "
                "exclusion declares a successful build with defective "
                "emission (chelis#751); a build-time rejection is different "
                "behavior. Re-adjudicate per B2.5."
            )
            continue
        if entry.native_status == "exact":
            violations.append(
                f"C ground truth [{label}]: generated C compiles, runs, and "
                "renders the exact intended bits - the chelis#751 repair "
                "landed. Shrink C_LANE_EXCLUDED and DECLARED_EXCLUSIONS in "
                "one change set."
            )
            continue
        if label == "f64-neg-zero":
            if entry.native_status != "wrong-bits":
                violations.append(
                    "C ground truth [f64-neg-zero]: native execution did not "
                    "reproduce the declared dropped-sign behavior; this is a "
                    "different defect or incomplete repair. Re-adjudicate "
                    "under B2.5."
                )
            continue
        fingerprint = c_has_bare_giant_integer_literal(entry.c_source)
        if entry.native_status not in ("compile-failed", "wrong-bits", "run-failed"):
            violations.append(
                f"C ground truth [{label}]: unknown native observation "
                f"{entry.native_status!r}; the executable boundary is not "
                "proved."
            )
        elif not fingerprint:
            violations.append(
                f"C ground truth [{label}]: the generated C no longer "
                "carries the chelis#751 bare-integer-literal fingerprint, but "
                "native execution is not exact. The old defect is gone and a "
                "different defect remains; file it under B2.5 before "
                "touching the exclusion."
            )
    return violations


def run_chelis(args: Sequence[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    command = (
        "cargo",
        "run",
        "-p",
        "chelis-cli",
        "--bin",
        "chelis",
        "--quiet",
        "--",
        *args,
    )
    child_env = dict(env)
    child_env["CHELIS_STYLE_GATE_DISABLE"] = "1"
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=child_env,
        check=False,
        capture_output=True,
        text=True,
    )


def c_rendered_values(stdout: str) -> list[str]:
    """Numeric element spellings from generated tensor/list exit lines."""

    values: list[str] = []
    for line in stdout.splitlines():
        data = re.search(r"data=\[([^\]]*)\]", line)
        if data is not None:
            values.extend(
                item.strip() for item in data.group(1).split(",") if item.strip()
            )
            continue
        payload = line.strip()
        if " = [" in payload:
            payload = payload.split(" = ", 1)[1]
        if payload.startswith("[") and payload.endswith("]"):
            values.extend(
                item.strip()
                for item in payload.removeprefix("[").removesuffix("]").split(",")
                if item.strip()
            )
    return values


def run_generated_c_ground_truth(
    build_stdout: str,
    out_dir: Path,
    dtype: str,
    intended_text: str,
    env: dict[str, str],
) -> str:
    """Compile/run the CLI's emitted command and compare rendered bits."""

    compile_line = next(
        (
            line.removeprefix("Compile: ")
            for line in build_stdout.splitlines()
            if line.startswith("Compile: ")
        ),
        None,
    )
    if compile_line is None:
        return "run-failed"
    command = shlex.split(compile_line)
    compiled = subprocess.run(
        command,
        cwd=out_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if compiled.returncode != 0:
        return "compile-failed"
    try:
        output_index = command.index("-o") + 1
        binary = Path(command[output_index])
    except (ValueError, IndexError):
        return "run-failed"
    if not binary.is_absolute():
        binary = out_dir / binary
    ran = subprocess.run(
        (str(binary),),
        cwd=out_dir,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if ran.returncode != 0:
        return "run-failed"
    rendered = c_rendered_values(ran.stdout)
    if not rendered:
        return "run-failed"
    intended = float(intended_text)
    intended_bits = f32_bits(intended) if dtype == "f32" else f64_bits(intended)
    try:
        rendered_bits = [
            f32_bits(float(text)) if dtype == "f32" else f64_bits(float(text))
            for text in rendered
        ]
    except ValueError:
        return "wrong-bits"
    return (
        "exact"
        if all(bits == intended_bits for bits in rendered_bits)
        else "wrong-bits"
    )


def run_exclusion_ground_truth(env: dict[str, str]) -> None:
    """§B2.9 leg 2's INDEPENDENT half: the oracle re-executes every
    excluded behavior itself - it writes the programs, runs the
    toolchain, drives native C behavior, and re-derives each fingerprint
    - so a forged probe receipt changes nothing here."""

    violations: list[str] = []
    exclusions = {name: rows for name, _o, _p, rows in DECLARED_EXCLUSIONS}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        eval_rows = exclusions["EVAL_F64_LIST_EXCLUDED"]
        elems = ", ".join(elem for _l, _d, elem, _v in eval_rows)
        program = (
            "module M.Main\n"
            f"def mk() -> tensor[{len(eval_rows)}, f64] = to_tensor([{elems}])\n"
            "lroot = to_list(mk())\n"
        )
        source = tmp_path / "eval_ground_truth.ch"
        source.write_text(program, encoding="utf-8")
        print("+ oracle-driven eval re-execution of EVAL_F64_LIST_EXCLUDED", flush=True)
        completed = run_chelis(("eval", "--file", str(source)), env)
        if completed.returncode != 0:
            violations.append(
                f"eval ground truth: `chelis eval` failed:\n{completed.stderr[-2000:]}"
            )
        else:
            # A single-root program renders its list BARE in eval; a
            # multi-root program labels it `lroot = [...]` (the chelis#862
            # root-labeling behavior). Accept both.
            line = next(
                (
                    stripped.removeprefix("lroot = ")
                    for stripped in (
                        l.strip() for l in completed.stdout.splitlines()
                    )
                    if stripped.startswith("lroot = [") or stripped.startswith("[")
                ),
                None,
            )
            if line is None:
                violations.append(
                    "eval ground truth: no to_list render in the driver "
                    f"output:\n{completed.stdout[-2000:]}"
                )
            else:
                texts = [
                    t.strip()
                    for t in line.removeprefix("[").rstrip("]").split(",")
                    if t.strip()
                ]
                consume_findings(
                    violations,
                    eval_exclusion_ground_truth_violations,
                    texts,
                    eval_rows,
                )

        entries: list[CExclusionGroundTruth] = []
        for label, dt, elem, value_text in exclusions["C_LANE_EXCLUDED"]:
            name = f"gt_{label.replace('-', '_')}"
            program = (
                "module M.Main\n"
                f"def mk() -> tensor[1, {dt}] = to_tensor([{elem}])\n"
                "shown = print(mk())\n"
                "listed = print(to_list(mk()))\n"
                "troot = mk()\n"
                "lroot = to_list(mk())\n"
            )
            source = tmp_path / f"{name}.ch"
            out_dir = tmp_path / f"{name}-out"
            source.write_text(program, encoding="utf-8")
            print(f"+ oracle-driven C emission re-execution of {label}", flush=True)
            completed = run_chelis(
                ("build", str(source), "--target", "c", "--output", str(out_dir)), env
            )
            c_file = out_dir / f"{name}.c"
            c_source = (
                c_file.read_text(encoding="utf-8") if c_file.is_file() else ""
            )
            native_status = (
                run_generated_c_ground_truth(
                    completed.stdout, out_dir, dt, value_text, env
                )
                if completed.returncode == 0
                else "not-run"
            )
            entries.append(
                CExclusionGroundTruth(
                    label, completed.returncode == 0, c_source, native_status
                )
            )
        consume_findings(violations, c_exclusion_ground_truth_violations, entries)

    if violations:
        raise OracleFailure(
            "independent exclusion re-execution failed:\n" + "\n".join(violations)
        )
    print(
        "+ independent exclusion re-execution: every declared label's "
        "fingerprint and executable C behavior re-derived by the oracle itself",
        flush=True,
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
        before = len(violations)
        consume_optional_finding(
            violations, classify_red_run, cell, completed.returncode, output
        )
        if len(violations) == before:
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
        run_exclusion_probes(env)
        run_exclusion_ground_truth(env)
        run_known_red_cells(env)
        receipts = instrument_result_violations(
            INVOKED_INSTRUMENTS, CONSUMED_INSTRUMENTS
        )
        if receipts:
            raise OracleFailure(
                "instrument result receipts failed:\n" + "\n".join(receipts)
            )
        print(
            "+ runtime result receipts: all manifest instruments executed "
            "and their results reached the verdict "
            f"({len(CONSUMED_INSTRUMENTS)} consumed receipts)",
            flush=True,
        )
    except OracleFailure as error:
        print(f"PHASE 2 ORACLE: FAIL: {error}", file=sys.stderr)
        return 1
    print("PHASE 2 ORACLE: PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
