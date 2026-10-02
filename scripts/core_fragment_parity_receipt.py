#!/usr/bin/env python3
"""Core-fragment eval/C parity receipt for chelis#2102.

The authoritative completion oracle for chelis#1362 guarantee 2 -- "`eval` and
`build --target c` agree on observations and traps" -- over the pinned downstream
corpora. The owning design document is
`spec/design/core_fragment_parity_corpus.md`; every rule this script enforces is
cited there against its normative authority in `spec/05-risc-primitives.md` §8.

Acceptance is exit 0 with a final `RECEIPT: PASS` line.

What it does, per required case:

1. Materializes the case into an isolated directory. Isolation is required, not
   tidiness: a corpus `.ch` file sits inside a reef package whose `compiler =`
   pin is checked by `chelis build`, so building in place fails on the pin
   rather than on the program.
2. Canonically formats the isolated copy with `chelis fmt --inplace`. The
   receipt never passes `--allow-style-violations`: a case that cannot survive
   canonical formatting is a recorded failure, not a bypassed gate.
3. Runs `chelis eval --file k.ch --target c`, capturing stdout, stderr and exit
   status. `--target c` is the documented flag for cross-lane comparison: it
   manifests the eval lane under C backend constraints.
4. Runs `chelis build --target c`, executes the compile command the build
   printed (optionally under a named flag override), and runs the binary,
   capturing stdout, stderr and exit status.
5. Compares the two lanes' complete stdout byte for byte when a value is
   expected, and their complete stderr and exit status when a trap is expected.

The comparison predicate is byte equality, which is
`chelis_types::agreement::compare_exact_observations` verbatim -- that function's
body is `reference.as_bytes() == candidate.as_bytes()`. No tolerance is applied:
[05-OBS-3] permits one only when both lanes compute at [04-NUM-8]'s declared
arithmetic width, and chelis#897 records that the evaluator does not, so the
atom's own text forbids laundering a float mismatch through the table.

`scripts/test_core_fragment_parity_receipt.py` exercises this script's decision
logic with the subprocess layer patched, and locks the comparator-equivalence
claim above against the Rust source. Those tests are evidence about this script
and never a substitute for running it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# spec/design/core_fragment_parity_corpus.md §6.3. Closed on purpose: an open
# list silently absorbs the file nobody classified, which is the vacuous pass
# this receipt exists to prevent.
CLOSED_EXCLUSION_REASONS = frozenset(
    {
        "library-only",
        "prove-only",
        "parse-rejected",
        "retired-syntax",
        "nondeterministic",
        "c-unsupported",
        "unmeasurable-by-probe",
        "outcome-undetermined",
    }
)

KNOWN_CORPORA = frozenset({"c-note", "sonar", "voyage"})

EXPECTED_OUTCOMES = frozenset({"value", "trap"})

# §5.2: [05-OBS-5]'s truncation marker, emitted inside `data=[..]` when a
# tensor rendering cuts after 32 elements.
TRUNCATION_MARKER = ", ..."

# The build prints the command that finishes the build on its own line.
COMPILE_LINE_RE = re.compile(r"^Compile: (.+)$", re.MULTILINE)

# Named compile profiles. `emitted` runs exactly what the build printed, which
# is what a user following the tool's instructions gets. `no-fp-contract` is
# chelis#2782's candidate repair, kept available so one receipt run can report
# under both without editing the script.
COMPILE_PROFILES = ("emitted", "no-fp-contract")


class ReceiptError(Exception):
    """A receipt-level failure: the run cannot produce a trustworthy verdict."""


class ManifestError(ReceiptError):
    """The manifest itself is malformed. Never a case verdict."""


# --------------------------------------------------------------------------
# Verdict vocabulary
# --------------------------------------------------------------------------

# Ordered most severe first. `lane-split` is the worst outcome the receipt can
# report: one lane produced a value and the other trapped, which is the exact
# divergence #1362 guarantee 2 forbids.
VERDICT_LANE_SPLIT = "lane-split"
VERDICT_TRAP_REASON_MISMATCH = "trap-reason-mismatch"
VERDICT_TRAP_STATUS_MISMATCH = "trap-status-mismatch"
VERDICT_OBSERVATION_MISMATCH = "observation-mismatch"
VERDICT_MISSING_DECLARED_ROOT = "missing-declared-root"
VERDICT_WRONG_OUTCOME = "wrong-outcome"
VERDICT_HARNESS_FAILURE = "harness-failure"
VERDICT_EXPECTED_FAILING = "expected-failing"
VERDICT_UNEXPECTED_PASS = "unexpected-pass"
VERDICT_AGREE = "agree"

UNTRACKED_FAILING_VERDICTS = (
    VERDICT_LANE_SPLIT,
    VERDICT_TRAP_REASON_MISMATCH,
    VERDICT_TRAP_STATUS_MISMATCH,
    VERDICT_OBSERVATION_MISMATCH,
    VERDICT_MISSING_DECLARED_ROOT,
    VERDICT_WRONG_OUTCOME,
    VERDICT_HARNESS_FAILURE,
)

# A tracked divergence still fails the receipt. chelis#1362's ship rule is "no
# `demo-path` row remains open", and this manifest is `demo-path`'s only
# authority, so a receipt that passed while a known divergence was open would
# contradict the rule it exists to serve. `known_divergence` buys diagnosis --
# separating a regression from a tracked defect -- not a pass.
FAILING_VERDICTS = UNTRACKED_FAILING_VERDICTS + (VERDICT_EXPECTED_FAILING,)

# Exit codes. 0 is the only acceptance.
#
# 2 means "the run could not start, so no verdict exists". A malformed manifest
# and an unusable invocation are the same thing to a reader, and argparse also
# exits 2 on a usage error, so the code is documented at that granularity
# rather than pretending to distinguish them.
EXIT_PASS = 0
EXIT_UNTRACKED_FAILURE = 1
EXIT_COULD_NOT_START = 2
EXIT_TRACKED_FAILURE_ONLY = 3


@dataclass(frozen=True)
class LaneResult:
    """One lane's complete observation."""

    returncode: int
    stdout: bytes
    stderr: bytes

    @property
    def trapped(self) -> bool:
        return self.returncode != 0


@dataclass
class CaseRow:
    case_id: str
    corpus: str
    source: dict
    expected: str
    roots: list[str]
    truncating_roots: list[str] = field(default_factory=list)
    notes: str = ""
    known_divergence: dict | None = None


@dataclass
class ExclusionRow:
    corpus: str
    path: str
    reason: str
    notes: str = ""


@dataclass
class Manifest:
    manifest_version: int
    corpora: dict  # corpus -> {"repo": str, "rev": str, "expected_case_count": int}
    cases: list[CaseRow]
    exclusions: list[ExclusionRow]


@dataclass
class CaseVerdict:
    case_id: str
    corpus: str
    verdict: str
    detail: str = ""
    truncated: bool = False
    comparable: bool = False
    commands: list[str] = field(default_factory=list)
    eval_returncode: int | None = None
    compiled_returncode: int | None = None
    # Which stage of the compiled lane produced its result: `build`, `compile`
    # or `run`. A trap case whose compiled lane failed at `build` and whose eval
    # lane failed at check time are comparable refusals; one that failed at
    # `run` is a different kind of event, and a reader needs to be able to tell
    # them apart rather than infer it from the diagnostic text.
    compiled_stage: str | None = None


# --------------------------------------------------------------------------
# Manifest loading and validation (§6)
# --------------------------------------------------------------------------


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _validate_source(case_id: str, source: object) -> dict:
    _require(
        isinstance(source, dict),
        f"case {case_id}: `source` must be an object",
    )
    assert isinstance(source, dict)  # narrowed above
    kind = source.get("kind")
    _require(
        kind in ("committed", "derived"),
        f"case {case_id}: source.kind must be 'committed' or 'derived', got {kind!r}",
    )
    if kind == "committed":
        required = ("repo", "rev", "path")
    else:
        required = ("repo", "rev", "generator", "task", "index")
    for key in required:
        _require(
            key in source,
            f"case {case_id}: {kind} source is missing required field `{key}`",
        )
    return source


def parse_manifest(data: object) -> Manifest:
    """Validate a decoded manifest document and return it typed.

    Every rule here is a §6 requirement. A malformed manifest raises rather
    than degrading: a receipt run against an unvalidated manifest would report
    a verdict whose scope nobody can state.
    """
    _require(isinstance(data, dict), "manifest must be a JSON object")
    assert isinstance(data, dict)

    version = data.get("manifest_version")
    _require(
        isinstance(version, int) and not isinstance(version, bool),
        "manifest_version must be an integer",
    )
    assert isinstance(version, int)

    raw_corpora = data.get("corpora")
    _require(isinstance(raw_corpora, dict), "manifest must carry a `corpora` object")
    assert isinstance(raw_corpora, dict)
    for name, entry in raw_corpora.items():
        _require(
            name in KNOWN_CORPORA,
            f"unknown corpus {name!r}; known corpora are {sorted(KNOWN_CORPORA)}",
        )
        _require(isinstance(entry, dict), f"corpus {name}: entry must be an object")
        for key in ("repo", "rev", "expected_case_count"):
            _require(
                key in entry,
                f"corpus {name}: missing required field `{key}`",
            )
        _require(
            isinstance(entry["expected_case_count"], int)
            and not isinstance(entry["expected_case_count"], bool),
            f"corpus {name}: expected_case_count must be an integer",
        )
        _require(
            bool(entry["rev"]) and entry["rev"] != "unpinned",
            f"corpus {name}: rev must be an exact pinned revision",
        )

    raw_cases = data.get("cases")
    _require(isinstance(raw_cases, list), "manifest must carry a `cases` array")
    assert isinstance(raw_cases, list)

    cases: list[CaseRow] = []
    seen_ids: set[str] = set()
    for raw in raw_cases:
        _require(isinstance(raw, dict), "each case must be an object")
        assert isinstance(raw, dict)
        case_id = raw.get("case_id")
        _require(
            isinstance(case_id, str) and bool(case_id),
            "each case must carry a non-empty `case_id`",
        )
        assert isinstance(case_id, str)
        _require(
            case_id not in seen_ids,
            f"duplicate case_id {case_id!r}: identities are unique and never reused",
        )
        seen_ids.add(case_id)

        corpus = raw.get("corpus")
        _require(
            corpus in KNOWN_CORPORA,
            f"case {case_id}: corpus must be one of {sorted(KNOWN_CORPORA)}",
        )
        _require(
            corpus in raw_corpora,
            f"case {case_id}: corpus {corpus!r} has no pinned revision in `corpora`",
        )

        # §6.2: `expected` is required and has no default. A case whose outcome
        # nobody has decided is an `outcome-undetermined` exclusion, not a case.
        expected = raw.get("expected")
        _require(
            expected in EXPECTED_OUTCOMES,
            f"case {case_id}: `expected` must be 'value' or 'trap' "
            f"(got {expected!r}); an undecided case belongs in `exclusions` "
            f"with reason 'outcome-undetermined'",
        )

        roots = raw.get("roots")
        _require(
            isinstance(roots, list) and all(isinstance(r, str) for r in roots),
            f"case {case_id}: `roots` must be an array of strings",
        )
        assert isinstance(roots, list)
        # §5.2's blackout guard reads `roots`, so an empty list on a `value`
        # case silently disarms it: two agreeing empty streams would pass with
        # nothing to be missing. A value case owing no root is incoherent
        # anyway -- that is what the `library-only` exclusion is for.
        _require(
            expected != "value" or bool([r for r in roots if r.strip()]),
            f"case {case_id}: an `expected: \"value\"` case must declare at "
            f"least one root; a case that owes no observation belongs in "
            f"`exclusions` with reason 'library-only'",
        )

        truncating = raw.get("truncating_roots", [])
        _require(
            isinstance(truncating, list)
            and all(isinstance(r, str) for r in truncating),
            f"case {case_id}: `truncating_roots` must be an array of strings",
        )
        assert isinstance(truncating, list)
        unknown_truncating = sorted(set(truncating) - set(roots))
        _require(
            not unknown_truncating,
            f"case {case_id}: truncating_roots names roots the case does not "
            f"own: {unknown_truncating}",
        )

        known_divergence = raw.get("known_divergence")
        if known_divergence is not None:
            _require(
                isinstance(known_divergence, dict),
                f"case {case_id}: known_divergence must be an object",
            )
            assert isinstance(known_divergence, dict)
            issue = known_divergence.get("issue")
            _require(
                isinstance(issue, int) and not isinstance(issue, bool) and issue > 0,
                f"case {case_id}: known_divergence requires a positive integer "
                f"`issue`; an untracked divergence may not be pre-accepted",
            )

        cases.append(
            CaseRow(
                case_id=case_id,
                corpus=str(corpus),
                source=_validate_source(case_id, raw.get("source")),
                expected=str(expected),
                roots=[str(r) for r in roots],
                truncating_roots=[str(r) for r in truncating],
                notes=str(raw.get("notes", "")),
                known_divergence=known_divergence,
            )
        )

    raw_exclusions = data.get("exclusions")
    _require(
        isinstance(raw_exclusions, list),
        "manifest must carry an `exclusions` array (it may be empty)",
    )
    assert isinstance(raw_exclusions, list)

    exclusions: list[ExclusionRow] = []
    seen_paths: set[tuple[str, str]] = set()
    for raw in raw_exclusions:
        _require(isinstance(raw, dict), "each exclusion must be an object")
        assert isinstance(raw, dict)
        corpus = raw.get("corpus")
        _require(
            corpus in KNOWN_CORPORA,
            f"exclusion: corpus must be one of {sorted(KNOWN_CORPORA)}",
        )
        path = raw.get("path")
        _require(
            isinstance(path, str) and bool(path),
            "each exclusion must carry a non-empty `path`",
        )
        assert isinstance(path, str)
        key = (str(corpus), path)
        _require(
            key not in seen_paths,
            f"duplicate exclusion for {corpus}:{path}",
        )
        seen_paths.add(key)
        reason = raw.get("reason")
        # §6.3: an unrecognised reason is a receipt failure, not a warning.
        _require(
            reason in CLOSED_EXCLUSION_REASONS,
            f"exclusion {corpus}:{path}: reason {reason!r} is not in the closed "
            f"set {sorted(CLOSED_EXCLUSION_REASONS)}",
        )
        exclusions.append(
            ExclusionRow(
                corpus=str(corpus),
                path=path,
                reason=str(reason),
                notes=str(raw.get("notes", "")),
            )
        )

    return Manifest(
        manifest_version=version,
        corpora={str(k): dict(v) for k, v in raw_corpora.items()},
        cases=cases,
        exclusions=exclusions,
    )


def load_manifest(path: Path) -> Manifest:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{path}: not valid JSON: {exc}") from exc
    return parse_manifest(raw)


# --------------------------------------------------------------------------
# Comparison and classification (§2, §3)
# --------------------------------------------------------------------------


def streams_agree(reference: bytes, candidate: bytes) -> bool:
    """Byte equality: `compare_exact_observations`'s predicate, verbatim.

    See that function in `crates/chelis-types/src/agreement.rs`. No tolerance
    branch exists here by design (§3.2).
    """
    return reference == candidate


def classify_case(
    case: CaseRow,
    eval_result: LaneResult,
    compiled_result: LaneResult,
) -> tuple[str, str]:
    """Decide one case's verdict from the two lanes' complete observations.

    Returns `(verdict, detail)`. The ordering of the branches is the severity
    ordering in `FAILING_VERDICTS`: a lane split is decided before any
    same-shape comparison, because "one lane trapped and the other did not" is
    a stronger statement than "their bytes differ".
    """
    eval_trapped = eval_result.trapped
    compiled_trapped = compiled_result.trapped

    if eval_trapped != compiled_trapped:
        trapping = "eval" if eval_trapped else "compiled"
        returning = "compiled" if eval_trapped else "eval"
        return (
            VERDICT_LANE_SPLIT,
            f"{trapping} lane trapped (exit "
            f"{eval_result.returncode if eval_trapped else compiled_result.returncode})"
            f" while the {returning} lane returned a value",
        )

    if case.expected == "value":
        if eval_trapped:
            # Both trapped, but the case owes a value. That is a real failure
            # and it is NOT a parity failure: the lanes agree with each other
            # and disagree with the manifest.
            return (
                VERDICT_WRONG_OUTCOME,
                "case expects a value; both lanes trapped "
                f"(eval exit {eval_result.returncode}, "
                f"compiled exit {compiled_result.returncode})",
            )
        if not streams_agree(eval_result.stdout, compiled_result.stdout):
            return (
                VERDICT_OBSERVATION_MISMATCH,
                _first_difference(eval_result.stdout, compiled_result.stdout),
            )
        # Two agreeing streams are not evidence unless they carry the
        # observation the manifest says the case owes. Without this, a
        # regression that silences root rendering in BOTH lanes reports whole-
        # corpus agreement and exits 0 -- and flips every `known_divergence`
        # row to `unexpected-pass`, which §6.2 reads as "drop the `demo-path`
        # tag". That is the vacuous pass §5 exists to make impossible, and
        # `roots` was already declared per case and never read.
        missing = missing_roots(case, eval_result.stdout)
        if missing:
            return (
                VERDICT_MISSING_DECLARED_ROOT,
                f"both lanes agree, but the case declares root(s) "
                f"{missing} that appear in neither lane's observation; "
                f"agreement over an absent observation is not evidence",
            )
        return (VERDICT_AGREE, "")

    # case.expected == "trap"
    if not eval_trapped:
        return (
            VERDICT_WRONG_OUTCOME,
            "case expects a trap; both lanes returned a value",
        )
    if eval_result.returncode != compiled_result.returncode:
        return (
            VERDICT_TRAP_STATUS_MISMATCH,
            f"eval exit {eval_result.returncode} vs compiled exit "
            f"{compiled_result.returncode}",
        )
    if not streams_agree(eval_result.stderr, compiled_result.stderr):
        return (
            VERDICT_TRAP_REASON_MISMATCH,
            _first_difference(eval_result.stderr, compiled_result.stderr),
        )
    return (VERDICT_AGREE, "")


def apply_known_divergence(case: CaseRow, verdict: str, detail: str) -> tuple[str, str]:
    """Re-label a verdict against the case's recorded known divergence (§6.2).

    A `known_divergence` row is still run and still compared. A divergence
    becomes `expected-failing`; agreement becomes `unexpected-pass`, which is
    the signal to drop the issue's `demo-path` tag rather than something to
    swallow.
    """
    if case.known_divergence is None:
        return (verdict, detail)
    issue = case.known_divergence["issue"]
    if verdict == VERDICT_MISSING_DECLARED_ROOT:
        # A silenced observation channel is never a tracked divergence. Without
        # this, a blackout on a `known_divergence` row is relabelled
        # `expected-failing` and the row that was supposed to stay visible is
        # the one that hides the blackout.
        return (verdict, detail)
    if verdict in UNTRACKED_FAILING_VERDICTS:
        return (
            VERDICT_EXPECTED_FAILING,
            f"chelis#{issue}: {verdict}: {detail}",
        )
    if verdict == VERDICT_AGREE:
        return (
            VERDICT_UNEXPECTED_PASS,
            f"chelis#{issue} is recorded as a known divergence for this case, "
            f"but the lanes now agree; re-check the issue and drop its "
            f"`demo-path` tag if the repair has landed",
        )
    return (verdict, detail)


def _first_difference(reference: bytes, candidate: bytes) -> str:
    """Name the first differing byte offset and show a bounded window.

    A whole-stream dump of two long outputs is unreadable and a bare "differs"
    is unactionable; the offset plus a window is the smallest useful form.
    """
    limit = min(len(reference), len(candidate))
    offset = limit
    for index in range(limit):
        if reference[index] != candidate[index]:
            offset = index
            break
    window_start = max(0, offset - 40)
    window_end = offset + 40

    def window(data: bytes) -> str:
        return data[window_start:window_end].decode("utf-8", errors="replace")

    if offset == limit and len(reference) != len(candidate):
        return (
            f"streams share their first {limit} bytes but differ in length "
            f"(eval {len(reference)}, compiled {len(candidate)}); "
            f"eval tail {reference[limit:limit + 80]!r}, "
            f"compiled tail {candidate[limit:limit + 80]!r}"
        )
    return (
        f"first difference at byte {offset}; "
        f"eval {window(reference)!r} vs compiled {window(candidate)!r}"
    )


def missing_roots(case: CaseRow, stdout: bytes) -> list[str]:
    """Declared roots that do not appear as a `name = ` label in `stdout`.

    [05-OBS-6]: every root renders with a `name = value` label at every exit in
    both lanes. A declared root absent from the rendering means the case is not
    observing what the manifest says it observes.
    """
    rendered = stdout.decode("utf-8", errors="replace").splitlines()
    labels = {
        line.split(" = ", 1)[0]
        for line in rendered
        if " = " in line and not line.startswith((" ", "\t"))
    }
    # [05-OBS-8]: a tuple-valued root expands recursively into dotted
    # positional names (`result.0`, `result.1.0`), and a fixed-product ADT root
    # into its declared field names, so the bare root name never appears. A
    # dotted descendant therefore satisfies its declared ancestor. Requiring
    # the exact label would turn a byte-exactly-agreeing tuple case red, and
    # the correct spelling is not even knowable in advance: the same atom says
    # an ADT whose constructor is not statically fixed "remains one bare root".
    def observed(root: str) -> bool:
        return root in labels or any(
            label.startswith(f"{root}.") for label in labels
        )

    # Named limit (§6.2): a base root is satisfied by ANY ONE dotted
    # descendant, so a regression silencing `result.1` while `result.0` still
    # renders passes this check. Closing it needs the full expansion declared,
    # which [05-OBS-8] makes unknowable in advance for a non-fixed constructor.
    return [root for root in case.roots if not observed(root)]


def observation_truncated(stdout: bytes) -> bool:
    """Whether this observation carried [05-OBS-5]'s truncation marker (§5.2)."""
    return TRUNCATION_MARKER.encode("utf-8") in stdout


# --------------------------------------------------------------------------
# Non-vacuity (§5)
# --------------------------------------------------------------------------


def non_vacuity_failures(
    manifest: Manifest,
    discovered_paths: dict[str, set[str]],
    verdicts: Sequence[CaseVerdict],
) -> list[str]:
    """Return every §5.1 structural non-vacuity failure, named.

    `discovered_paths` maps corpus name to the set of corpus-relative `.ch`
    paths found on disk at the pinned revision. A corpus absent from the
    mapping was not walked and is not checked for rule 3; that absence is
    itself reported by the caller.
    """
    failures: list[str] = []

    required = manifest.cases
    if not required:
        failures.append(
            "rule 1: the manifest declares zero required cases; a receipt over "
            "an empty case set proves nothing"
        )

    verdict_ids = {verdict.case_id for verdict in verdicts}
    missing = sorted(case.case_id for case in required if case.case_id not in verdict_ids)
    if missing:
        failures.append(
            f"rule 2: {len(missing)} required case(s) produced no verdict: {missing[:10]}"
        )

    # Rule 3: a discovered file that is neither a case nor a recorded exclusion
    # is unclassified. That is a build failure, never an implicit skip.
    for corpus, paths in sorted(discovered_paths.items()):
        classified = {
            case.source["path"]
            for case in manifest.cases
            if case.corpus == corpus and case.source.get("kind") == "committed"
        }
        classified |= {
            exclusion.path
            for exclusion in manifest.exclusions
            if exclusion.corpus == corpus
        }
        unclassified = sorted(paths - classified)
        if unclassified:
            failures.append(
                f"rule 3: {corpus}: {len(unclassified)} discovered file(s) are "
                f"neither a case nor a recorded exclusion: {unclassified[:10]}"
            )

    comparable = sum(1 for verdict in verdicts if verdict.comparable)
    if required and comparable == 0:
        failures.append(
            "rule 4: zero required cases produced a comparable observation; "
            "the run is vacuous"
        )

    # Rule 5: per-corpus counts must match the manifest's declared expectation.
    for corpus, entry in sorted(manifest.corpora.items()):
        actual = sum(1 for case in manifest.cases if case.corpus == corpus)
        expected = entry["expected_case_count"]
        if actual != expected:
            failures.append(
                f"rule 5: {corpus}: manifest declares expected_case_count "
                f"{expected} but carries {actual} case row(s)"
            )

    return failures


def observation_coverage(verdicts: Sequence[CaseVerdict]) -> dict:
    """§5.2's two figures: agreement, and agreement with no truncated root."""
    agreeing = [
        verdict
        for verdict in verdicts
        if verdict.verdict in (VERDICT_AGREE, VERDICT_UNEXPECTED_PASS)
    ]
    untruncated = [verdict for verdict in agreeing if not verdict.truncated]
    return {
        "agreed": len(agreeing),
        "agreed_with_no_truncated_root": len(untruncated),
        "agreed_relying_on_truncated_rendering": len(agreeing) - len(untruncated),
    }


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


@dataclass
class Runner:
    """The subprocess layer, isolated so the unit tests can replace it."""

    chelis: Path
    timeout_seconds: int = 300

    def run(self, argv: Sequence[str], cwd: Path) -> LaneResult:
        try:
            completed = subprocess.run(
                list(argv),
                cwd=str(cwd),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LaneResult(
                returncode=124,
                stdout=b"",
                stderr=f"harness: timed out after {self.timeout_seconds}s\n".encode(),
            )
        except OSError as exc:
            return LaneResult(
                returncode=127,
                stdout=b"",
                stderr=f"harness: could not execute {argv[0]!r}: {exc}\n".encode(),
            )
        return LaneResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_shell(self, command: str, cwd: Path) -> LaneResult:
        """Run the compile command the build printed, as printed.

        The build emits one shell line; running it through a shell is what a
        user following the instructions does, and the receipt records it
        verbatim.
        """
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LaneResult(
                returncode=124,
                stdout=b"",
                stderr=f"harness: compile timed out after {self.timeout_seconds}s\n".encode(),
            )
        return LaneResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def apply_compile_profile(command: str, profile: str) -> str:
    """Rewrite the emitted compile command for a named profile (§4.3).

    `emitted` is the identity. `no-fp-contract` inserts `-ffp-contract=off`,
    which is chelis#2782's candidate repair; the flag is appended rather than
    substituted so the profile does not depend on `-O2` being present.
    """
    if profile == "emitted":
        return command
    if profile == "no-fp-contract":
        if "-ffp-contract" in command:
            return command
        parts = command.split()
        # Insert immediately after the compiler name so the flag cannot land
        # after a `--` or inside a linker argument list.
        return " ".join(parts[:1] + ["-ffp-contract=off"] + parts[1:])
    raise ReceiptError(
        f"unknown compile profile {profile!r}; known profiles are {list(COMPILE_PROFILES)}"
    )


def extract_compile_command(build_output: bytes) -> str:
    text = build_output.decode("utf-8", errors="replace")
    matches = COMPILE_LINE_RE.findall(text)
    if not matches:
        raise ReceiptError(
            "build emitted no `Compile:` line; the compiled lane cannot be built"
        )
    if len(matches) > 1:
        raise ReceiptError(
            f"build emitted {len(matches)} `Compile:` lines; the receipt needs "
            f"exactly one to run: {matches}"
        )
    return matches[0].strip()


def materialize_case(case: CaseRow, corpus_root: Path, workdir: Path) -> Path:
    """Copy a case into `workdir` as `k.ch`, returning that path.

    Isolation is load-bearing: a corpus `.ch` file sits inside a reef package
    whose `compiler =` pin `chelis build` checks, so an in-place build fails on
    the pin rather than on the program.
    """
    kind = case.source.get("kind")
    if kind == "committed":
        source_path = corpus_root / case.source["path"]
        if not source_path.is_file():
            raise ReceiptError(
                f"case {case.case_id}: {source_path} does not exist at the pinned "
                f"revision"
            )
        # Preserve the suffix: discovery walks `.ch` and `.dp`, and a `.dp`
        # program handed over as `k.ch` is parsed as Surf and dies in `fmt`.
        target = workdir / f"k{source_path.suffix}"
        shutil.copyfile(source_path, target)
        return target
    if kind == "derived":
        # A derived case is produced by the pinned generator, not read from the
        # corpus tree. The capture directory is `<corpus_root>/<task>/<index>`.
        capture = corpus_root / str(case.source["task"]) / str(case.source["index"])
        program = capture / "k.ch"
        if not program.is_file():
            raise ReceiptError(
                f"case {case.case_id}: derived capture {program} is absent; run "
                f"the pinned generator ({case.source['generator']}) first -- this "
                f"receipt consumes captures and does not generate them"
            )
        shutil.copyfile(program, workdir / "k.ch")
        inputs = capture / "inputs"
        if inputs.is_dir():
            shutil.copytree(inputs, workdir / "inputs")
        return workdir / "k.ch"
    raise ReceiptError(f"case {case.case_id}: unknown source kind {kind!r}")


def execute_case(
    case: CaseRow,
    corpus_root: Path,
    runner: Runner,
    compile_profile: str,
    keep_dir: Path | None = None,
) -> CaseVerdict:
    """Run one case through both lanes and classify it."""
    commands: list[str] = []
    compiled_stage: str | None = None
    # A case_id carries the corpus-relative path and therefore separators; a
    # tempdir prefix cannot.
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", case.case_id)[-60:]
    workdir_context = (
        tempfile.TemporaryDirectory(prefix=f"parity-{safe_id}-")
        if keep_dir is None
        else None
    )
    workdir = Path(workdir_context.name) if workdir_context else keep_dir
    assert workdir is not None
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        try:
            program = materialize_case(case, corpus_root, workdir)
        except ReceiptError as exc:
            return CaseVerdict(
                case_id=case.case_id,
                corpus=case.corpus,
                verdict=VERDICT_HARNESS_FAILURE,
                detail=str(exc),
                commands=commands,
            )

        chelis = str(runner.chelis)

        # Canonical formatting, never `--allow-style-violations` (§ docstring).
        fmt_argv = [chelis, "fmt", "--inplace", program.name]
        commands.append(" ".join(fmt_argv))
        fmt_result = runner.run(fmt_argv, workdir)
        if fmt_result.trapped:
            return CaseVerdict(
                case_id=case.case_id,
                corpus=case.corpus,
                verdict=VERDICT_HARNESS_FAILURE,
                detail=(
                    "`chelis fmt --inplace` failed, so neither lane ran under "
                    "the style gate this receipt requires: "
                    + fmt_result.stderr.decode("utf-8", errors="replace").strip()
                ),
                commands=commands,
            )

        eval_argv = [chelis, "eval", "--file", program.name, "--target", "c"]
        commands.append(" ".join(eval_argv))
        eval_result = runner.run(eval_argv, workdir)

        build_argv = [chelis, "build", "--emit-c", "--target", "c", "-o", "out", program.name]
        commands.append(" ".join(build_argv))
        build_result = runner.run(build_argv, workdir)

        if build_result.trapped:
            # The compiled lane refused the program. Whether that is a parity
            # failure depends on the eval lane, so it is classified, not
            # special-cased: a build refusal is the compiled lane trapping.
            compiled_result = build_result
            compiled_stage = "build"
        else:
            try:
                compile_command = apply_compile_profile(
                    extract_compile_command(build_result.stdout + build_result.stderr),
                    compile_profile,
                )
            except ReceiptError as exc:
                return CaseVerdict(
                    case_id=case.case_id,
                    corpus=case.corpus,
                    verdict=VERDICT_HARNESS_FAILURE,
                    detail=str(exc),
                    commands=commands,
                    eval_returncode=eval_result.returncode,
                )
            commands.append(compile_command)
            compile_result = runner.run_shell(compile_command, workdir)
            if compile_result.trapped:
                # The emitted C did not compile. That is the compiled lane
                # failing, not the harness: "eval returns a value and the C the
                # build emitted does not compile" is exactly a #1362
                # guarantee-2 divergence, and attributing it to the harness
                # would hide it. The receipt cannot tell an invalid artifact
                # from a broken local toolchain, so it reports the loud
                # direction: a toolchain that cannot compile anything makes
                # every case lane-split, which is obvious rather than silent.
                compiled_result = compile_result
                compiled_stage = "compile"
            else:
                binary = workdir / "out" / "k"
                commands.append(str(Path("out") / "k"))
                compiled_result = runner.run([str(binary)], workdir)
                compiled_stage = "run"

        verdict, detail = classify_case(case, eval_result, compiled_result)
        verdict, detail = apply_known_divergence(case, verdict, detail)

        return CaseVerdict(
            case_id=case.case_id,
            corpus=case.corpus,
            verdict=verdict,
            detail=detail,
            truncated=observation_truncated(eval_result.stdout)
            or observation_truncated(compiled_result.stdout),
            comparable=True,
            commands=commands,
            eval_returncode=eval_result.returncode,
            compiled_returncode=compiled_result.returncode,
            compiled_stage=compiled_stage,
        )
    finally:
        if workdir_context is not None:
            workdir_context.cleanup()


# --------------------------------------------------------------------------
# Pins (§4)
# --------------------------------------------------------------------------


def _git_revision(path: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def probe_staging_receipt(chelis: Path, runner: Runner) -> dict | None:
    """Read the compiler's own staging receipt by performing one real build.

    `chelis_runtime_bundle` writes `chelis_runtime.receipt.json` beside every
    staged runtime, carrying `mode` (`sealed` or `development`), the runtime
    archive's sha256, and every published header's digest. That is the
    product's declared channel for the build mode -- the crate's own comment
    says "Staging receipts record the mode" -- so the receipt reads it rather
    than inferring the mode from the binary's bytes.

    Recorded because an unsealed development build re-checks its source
    checkout for runtime-bundle freshness on every `build`, so a concurrent
    writer can change its behaviour mid-run while its hash stays constant. A
    hash alone does not pin an unsealed build (§4.2).
    """
    with tempfile.TemporaryDirectory(prefix="parity-pin-probe-") as tmp:
        work = Path(tmp)
        (work / "k.ch").write_text("probe = 1\n", encoding="utf-8")
        result = runner.run(
            [str(chelis), "build", "--emit-c", "--target", "c", "-o", "out", "k.ch"], work
        )
        if result.trapped:
            return None
        receipt_path = work / "out" / "chelis_runtime.receipt.json"
        if not receipt_path.is_file():
            return None
        try:
            return json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None


def _sealed_from_staging(staging: dict | None) -> bool | None:
    """`True`/`False` from the staging receipt's mode; `None` if unreadable."""
    if staging is None:
        return None
    mode = staging.get("mode")
    if mode == "sealed":
        return True
    if mode == "development":
        return False
    return None


def collect_pins(
    chelis: Path,
    manifest: Manifest,
    corpus_roots: dict[str, Path],
    compile_profile: str,
    runner: Runner,
) -> dict:
    version = runner.run([str(chelis), "--version"], REPO_ROOT)
    staging = probe_staging_receipt(chelis, runner)
    pins = {
        "compiler": {
            "path": str(chelis),
            # `--chelis` is required, so the binary is always named explicitly.
            # Recorded anyway: §4.2 requires the receipt to state how it found
            # the binary, and the bare `chelis` shim resolves elsewhere.
            "resolution": "explicit --chelis path",
            "version_string": version.stdout.decode("utf-8", errors="replace").strip(),
            "sha256": hashlib.sha256(chelis.read_bytes()).hexdigest()
            if chelis.is_file()
            else None,
            "sealed_runtime": _sealed_from_staging(staging),
            # NOT the compiler's provenance. This is the revision of the
            # checkout the receipt ran FROM, which is the harness's identity.
            # It was previously called `repo_revision` under §4.2's "the git
            # revision ... for the compiler under test", which made it a field
            # that lies whenever `--chelis` points outside this checkout -- an
            # installed toolchain still recorded this worktree's HEAD. The
            # compiler's own identity is its `version_string`, `sha256`, and
            # the staging receipt's `chelis_version` and archive digest.
            "harness_repo_revision": _git_revision(REPO_ROOT),
            "staging_receipt": staging,
        },
        "manifest_version": manifest.manifest_version,
        "compile_profile": compile_profile,
        "corpora": {},
        "platform": {
            "sys_platform": sys.platform,
            "uname": list(os.uname()) if hasattr(os, "uname") else None,
        },
    }
    for corpus, entry in sorted(manifest.corpora.items()):
        root = corpus_roots.get(corpus)
        pins["corpora"][corpus] = {
            "repo": entry["repo"],
            "declared_rev": entry["rev"],
            "root": str(root) if root else None,
            "observed_rev": _git_revision(root) if root else None,
        }
    return pins


def pin_failures(pins: dict) -> list[str]:
    """Every pin whose declared and observed value disagree, or is unstated."""
    failures: list[str] = []
    compiler = pins["compiler"]
    if not compiler.get("version_string"):
        failures.append(
            "the compiler under test produced no `--version` output; the "
            "receipt cannot state which binary it ran"
        )
    # §4.2: only a positively-read `sealed` mode clears this. `None` means the
    # probe could not tell, and an unpinnable compiler is not evidence -- the
    # installed release toolchains write no staging receipt at all, so treating
    # `None` as acceptable made the gate inert for exactly the compiler class
    # it exists for.
    sealed = compiler.get("sealed_runtime")
    if sealed is not True:
        unknown = sealed is None
        failures.append(
            (
                "the receipt could not read the compiler's build mode from its "
                "staging receipt (`chelis_runtime.receipt.json`), so the build "
                "cannot be pinned"
                if unknown
                else "the compiler under test is not a sealed-runtime build"
            )
            + ": an unsealed build re-checks its source checkout on every "
            "`build`, so a concurrent writer can change its behaviour mid-run "
            "while its hash stays constant (§4.2). Build the compiler under "
            "test with `--features sealed-runtime`"
        )
    for corpus, entry in sorted(pins["corpora"].items()):
        if entry.get("root") is None:
            failures.append(
                f"corpus {corpus}: no checkout supplied, so its cases cannot run"
            )
            continue
        observed = entry.get("observed_rev")
        declared = entry.get("declared_rev")
        if observed is None:
            failures.append(
                f"corpus {corpus}: {entry['root']} is not a git checkout, so its "
                f"revision cannot be pinned"
            )
        elif observed != declared:
            failures.append(
                f"corpus {corpus}: manifest pins {declared} but the checkout at "
                f"{entry['root']} is at {observed}"
            )
    return failures


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


# Both Chelis source extensions `chelis eval --file` accepts. Globbing `.ch`
# alone made §5.3's "every file discovered in a pinned corpus" false: a pinned
# corpus carries `.dp` programs the eval lane reads.
DISCOVERED_SUFFIXES = (".ch", ".dp")


def discover_committed_paths(corpus_root: Path) -> set[str]:
    """Every Chelis source in a committed corpus, as corpus-relative POSIX paths."""
    return {
        str(path.relative_to(corpus_root).as_posix())
        for suffix in DISCOVERED_SUFFIXES
        for path in corpus_root.rglob(f"*{suffix}")
        if ".git" not in path.parts
    }


# --------------------------------------------------------------------------
# Receipt assembly and CLI
# --------------------------------------------------------------------------


def build_receipt(
    manifest: Manifest,
    pins: dict,
    verdicts: Sequence[CaseVerdict],
    non_vacuity: Sequence[str],
    pin_problems: Sequence[str],
) -> dict:
    by_verdict: dict[str, int] = {}
    for verdict in verdicts:
        by_verdict[verdict.verdict] = by_verdict.get(verdict.verdict, 0) + 1
    return {
        "receipt_kind": "core-fragment-eval-c-parity",
        "owning_issue": 2102,
        "manifest_version": manifest.manifest_version,
        "pins": pins,
        "verdict_counts": by_verdict,
        "observation_coverage": observation_coverage(verdicts),
        "non_vacuity_failures": list(non_vacuity),
        "pin_failures": list(pin_problems),
        "failing_stage_counts": failing_stage_counts(verdicts),
        "exclusions": [
            {
                "corpus": exclusion.corpus,
                "path": exclusion.path,
                "reason": exclusion.reason,
                "notes": exclusion.notes,
            }
            for exclusion in manifest.exclusions
        ],
        "cases": [
            {
                "case_id": verdict.case_id,
                "corpus": verdict.corpus,
                "verdict": verdict.verdict,
                "detail": verdict.detail,
                "truncated": verdict.truncated,
                "comparable": verdict.comparable,
                "eval_returncode": verdict.eval_returncode,
                "compiled_returncode": verdict.compiled_returncode,
                "compiled_stage": verdict.compiled_stage,
                "commands": verdict.commands,
            }
            for verdict in verdicts
        ],
    }


def failing_stage_counts(verdicts: Sequence[CaseVerdict]) -> dict:
    """How many failing cases failed at each compiled-lane stage.

    §7 files a defect per reported divergence, and a broken local C toolchain
    produces one lane split per case, so a reader needs to see where the
    failures sit before filing anything. This reports the distribution the
    verdicts already carry and draws no conclusion from it: an earlier version
    asserted "every failure is at the compile stage" as a boolean and was wrong
    in both directions -- inert on the shipped manifest, whose tracked rows fail
    at `build`, and affirmative on a single genuine invalid-C defect beside
    twelve agreeing cases, where blaming the toolchain is exactly the wrong
    advice.
    """
    counts: dict[str, int] = {}
    for verdict in verdicts:
        if verdict.verdict not in FAILING_VERDICTS:
            continue
        stage = verdict.compiled_stage or "unreached"
        counts[stage] = counts.get(stage, 0) + 1
    return counts


def receipt_passes(receipt: dict) -> bool:
    if receipt["non_vacuity_failures"] or receipt["pin_failures"]:
        return False
    counts = receipt["verdict_counts"]
    return not any(counts.get(verdict) for verdict in FAILING_VERDICTS)


def receipt_exit_code(receipt: dict) -> int:
    """0 on acceptance; 3 when every failure is tracked; 1 otherwise.

    The distinction is for a reader deciding what the run means, never a way to
    pass: only 0 is acceptance (§8).
    """
    if receipt_passes(receipt):
        return EXIT_PASS
    counts = receipt["verdict_counts"]
    untracked = any(counts.get(verdict) for verdict in UNTRACKED_FAILING_VERDICTS)
    if untracked or receipt["non_vacuity_failures"] or receipt["pin_failures"]:
        return EXIT_UNTRACKED_FAILURE
    return EXIT_TRACKED_FAILURE_ONLY


def _summarize(receipt: dict, stream) -> None:
    counts = receipt["verdict_counts"]
    coverage = receipt["observation_coverage"]
    print("", file=stream)
    print("core-fragment eval/C parity receipt", file=stream)
    print(f"  manifest version : {receipt['manifest_version']}", file=stream)
    print(
        f"  compiler         : {receipt['pins']['compiler']['version_string']} "
        f"({receipt['pins']['compiler']['path']})",
        file=stream,
    )
    print(f"  compile profile  : {receipt['pins']['compile_profile']}", file=stream)
    for corpus, entry in sorted(receipt["pins"]["corpora"].items()):
        print(
            f"  corpus {corpus:<8} : {entry['repo']} @ {entry['declared_rev']}",
            file=stream,
        )
    print("", file=stream)
    for verdict in sorted(counts):
        print(f"  {verdict:<26} {counts[verdict]}", file=stream)
    print("", file=stream)
    print(
        f"  agreed                     {coverage['agreed']}\n"
        f"  agreed, no truncated root  {coverage['agreed_with_no_truncated_root']}\n"
        f"  agreed via truncation only {coverage['agreed_relying_on_truncated_rendering']}",
        file=stream,
    )
    for failure in receipt["pin_failures"]:
        print(f"  PIN FAILURE: {failure}", file=stream)
    for failure in receipt["non_vacuity_failures"]:
        print(f"  VACUOUS: {failure}", file=stream)
    for case in receipt["cases"]:
        if case["verdict"] in FAILING_VERDICTS:
            print(
                f"  FAIL {case['case_id']}: {case['verdict']}: {case['detail']}",
                file=stream,
            )
    stages = receipt.get("failing_stage_counts") or {}
    if stages:
        # Data, not a conclusion (§7 step 1 does the reasoning). A run whose
        # failures are concentrated at `compile` may be a broken local C
        # toolchain rather than N divergences; one `compile` failure beside
        # agreeing cases is more likely a real invalid-C defect.
        rendered = ", ".join(f"{stage}={count}" for stage, count in sorted(stages.items()))
        print(f"  failing cases by compiled-lane stage: {rendered}", file=stream)
    print("", file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the chelis#2102 core-fragment eval/C parity receipt. "
            "Acceptance is exit 0 with a final `RECEIPT: PASS` line."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "tests" / "corpus" / "core_fragment_parity" / "manifest.json",
    )
    parser.add_argument(
        "--chelis",
        type=Path,
        required=True,
        help=(
            "the compiler binary under test, by explicit path. Never the bare "
            "`chelis` shim: outside a reef package its pin order resolves to an "
            "unrelated toolchain version"
        ),
    )
    parser.add_argument(
        "--corpus",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="a pinned corpus checkout, as --corpus <corpus>=/path/to/checkout",
    )
    parser.add_argument(
        "--compile-profile",
        choices=COMPILE_PROFILES,
        default="emitted",
        help=(
            "which compile command to run. `emitted` runs exactly what the "
            "build printed; `no-fp-contract` appends -ffp-contract=off "
            "(chelis#2782's candidate repair)"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="run only these case ids (diagnosis; never acceptance evidence)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        manifest = load_manifest(args.manifest)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("RECEIPT: FAIL", file=sys.stdout)
        return EXIT_COULD_NOT_START

    corpus_roots: dict[str, Path] = {}
    for spec in args.corpus:
        name, _, path = spec.partition("=")
        if not path:
            print(
                f"error: --corpus expects NAME=PATH, got {spec!r}",
                file=sys.stderr,
            )
            return EXIT_COULD_NOT_START
        if name not in KNOWN_CORPORA:
            print(
                f"error: unknown corpus {name!r}; known corpora are "
                f"{sorted(KNOWN_CORPORA)}",
                file=sys.stderr,
            )
            return EXIT_COULD_NOT_START
        corpus_roots[name] = Path(path).resolve()

    runner = Runner(chelis=args.chelis.resolve(), timeout_seconds=args.timeout)
    pins = collect_pins(
        args.chelis.resolve(), manifest, corpus_roots, args.compile_profile, runner
    )
    pin_problems = pin_failures(pins)

    selected = set(args.case)
    cases = [
        case
        for case in manifest.cases
        if (not selected or case.case_id in selected)
        and case.corpus in corpus_roots
    ]

    verdicts: list[CaseVerdict] = []
    for case in cases:
        verdicts.append(
            execute_case(
                case,
                corpus_roots[case.corpus],
                runner,
                args.compile_profile,
            )
        )
        last = verdicts[-1]
        marker = "ok  " if last.verdict == VERDICT_AGREE else "FAIL"
        print(f"{marker} {last.case_id} {last.verdict}", flush=True)

    discovered = {
        corpus: discover_committed_paths(root)
        for corpus, root in sorted(corpus_roots.items())
    }
    non_vacuity = non_vacuity_failures(manifest, discovered, verdicts)
    if selected:
        non_vacuity.append(
            "--case was used, so this run is diagnosis over a subset and is "
            "never acceptance evidence"
        )

    receipt = build_receipt(manifest, pins, verdicts, non_vacuity, pin_problems)

    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _summarize(receipt, sys.stdout)
    print(f"receipt written to {receipt_path}")

    code = receipt_exit_code(receipt)
    if code == EXIT_PASS:
        print("RECEIPT: PASS")
        return code
    if code == EXIT_TRACKED_FAILURE_ONLY:
        print(
            "RECEIPT: FAIL (every failure is a tracked known divergence; "
            "chelis#1362's ship rule still blocks while they are open)"
        )
        return code
    print("RECEIPT: FAIL")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
