#!/usr/bin/env python3
"""Unit tests for scripts/core_fragment_parity_receipt.py (chelis#2102).

These tests exercise the receipt's own decision logic with the subprocess layer
replaced. They are evidence about the script and never a substitute for running
it: the acceptance oracle is the receipt itself, per
`spec/design/core_fragment_parity_corpus.md` §8.

Every positive row below has its negative partner, per `AGENTS.md`
§"Negative Test Parity". The obligations come from the design document:

- §3.1 the comparison predicate is byte equality, and is the same predicate as
  `chelis_types::agreement::compare_exact_observations`. `ComparatorEquivalence`
  locks that against the Rust source rather than asserting it in prose.
- §5.1 the five structural non-vacuity rules, each with a passing and a failing
  case.
- §5.2 truncation-aware observation accounting.
- §6.2 every required manifest field, each with a row that omits or corrupts it.
- §6.3 the closed exclusion-reason set.
- §2.3 trap parity: status and reason, separately.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core_fragment_parity_receipt as receipt

REPO_ROOT = Path(__file__).resolve().parents[1]


def minimal_manifest() -> dict:
    """A valid two-case manifest: one value case, one trap case."""
    return {
        "manifest_version": 1,
        "corpora": {
            "c-note": {
                "repo": "Chelis-Lang/c-note",
                "rev": "0" * 40,
                "expected_case_count": 2,
            }
        },
        "cases": [
            {
                "case_id": "c-note/square",
                "corpus": "c-note",
                "source": {
                    "kind": "committed",
                    "repo": "Chelis-Lang/c-note",
                    "rev": "0" * 40,
                    "path": "a.ch",
                },
                "expected": "value",
                "roots": ["result"],
            },
            {
                "case_id": "c-note/refuses",
                "corpus": "c-note",
                "source": {
                    "kind": "committed",
                    "repo": "Chelis-Lang/c-note",
                    "rev": "0" * 40,
                    "path": "b.ch",
                },
                "expected": "trap",
                "roots": ["result"],
            },
        ],
        "exclusions": [
            {"corpus": "c-note", "path": "lib.ch", "reason": "library-only"}
        ],
    }


def case_row(**overrides) -> receipt.CaseRow:
    base = {
        "case_id": "x",
        "corpus": "c-note",
        "source": {
            "kind": "committed",
            "repo": "Chelis-Lang/c-note",
            "rev": "0" * 40,
            "path": "a.ch",
        },
        "expected": "value",
        "roots": ["result"],
    }
    base.update(overrides)
    return receipt.CaseRow(**base)


def lane(returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> receipt.LaneResult:
    return receipt.LaneResult(returncode=returncode, stdout=stdout, stderr=stderr)


# ---------------------------------------------------------------------------
# §3.1 -- the comparator-equivalence lock
# ---------------------------------------------------------------------------


class ComparatorEquivalence(unittest.TestCase):
    """Lock the claim that byte equality IS the production comparator.

    The receipt is a Python script and cannot call
    `chelis_types::agreement::compare_exact_observations`. It claims instead
    that byte equality is the same predicate. That claim is true only while
    that function's body remains a byte comparison, so it is checked against
    the Rust source rather than trusted.
    """

    SOURCE = REPO_ROOT / "crates" / "chelis-types" / "src" / "agreement.rs"

    # The whole function, whitespace-normalised. An exact match is the only
    # form of this check that holds: a keyword blacklist ("ulp", "tolerance",
    # …) is evadable by a tolerance branch that happens to use none of those
    # words -- `} else if close_enough(a, b) { Ok(ByteExact) }`, or a
    # delegation to the crate's own `compare_rendered_elements`, both slip
    # straight through one. Round 1 demonstrated exactly that.
    EXPECTED_BODY = (
        "pub fn compare_exact_observations( context: &str, reference: &str, "
        "candidate: &str, ) -> Result<AgreementOutcome, AgreementError> { "
        "if reference.as_bytes() == candidate.as_bytes() { "
        "Ok(AgreementOutcome::ByteExact) } else { "
        "Err(AgreementError::ExactMismatch { context: context.to_string(), "
        "reference: reference.to_string(), candidate: candidate.to_string(), "
        "}) } }"
    )

    @classmethod
    def comparator_body(cls) -> str | None:
        text = cls.SOURCE.read_text(encoding="utf-8")
        match = re.search(
            r"pub fn compare_exact_observations\((?:.|\n)*?\n\}", text
        )
        return " ".join(match.group(0).split()) if match else None

    def test_production_comparator_is_still_byte_exact(self):
        body = self.comparator_body()
        self.assertIsNotNone(
            body,
            "compare_exact_observations is no longer in agreement.rs; the "
            "receipt's equivalence claim in §3.1 has to be re-earned",
        )
        self.assertEqual(
            body,
            self.EXPECTED_BODY,
            "compare_exact_observations is no longer the exact byte comparison "
            "the receipt's §3.1 equivalence claim rests on. Any change here -- "
            "including one that adds a branch without using the word "
            "'tolerance' -- means the Python byte comparison may no longer be "
            "the same predicate. Re-read §3.1 and re-earn the claim rather "
            "than updating this constant to match.",
        )

    def test_the_lock_rejects_an_added_branch_that_names_no_tolerance_word(self):
        # The negative control for the check above, and the exact hole round 1
        # found in its predecessor: a tolerance branch mentioning none of
        # "ulp"/"tolerance"/"abs("/"epsilon" must still be rejected.
        evasive = self.EXPECTED_BODY.replace(
            "} else { Err(AgreementError::ExactMismatch",
            "} else if close_enough(reference, candidate) { "
            "Ok(AgreementOutcome::ByteExact) } else { "
            "Err(AgreementError::ExactMismatch",
        )
        self.assertNotEqual(evasive, self.EXPECTED_BODY)
        for word in ("ulp", "tolerance", "abs(", "epsilon"):
            self.assertNotIn(
                word,
                evasive.lower(),
                "the evasive mutation must name no blacklisted word, or it "
                "does not exercise the hole it is a control for",
            )

    def test_the_lock_rejects_a_reformatted_but_semantically_changed_body(self):
        # Whitespace normalisation must not launder a real change.
        changed = self.EXPECTED_BODY.replace(
            "reference.as_bytes() == candidate.as_bytes()",
            "reference.trim() == candidate.trim()",
        )
        self.assertNotEqual(changed, self.EXPECTED_BODY)

    def test_streams_agree_matches_byte_equality(self):
        self.assertTrue(receipt.streams_agree(b"result = 49.0\n", b"result = 49.0\n"))
        self.assertFalse(receipt.streams_agree(b"result = 49.0\n", b"result = 49.1\n"))
        # A formatting-only difference is a difference, per [05-OBS-2]: text
        # denoting the same bits in another spelling is never tolerated.
        self.assertFalse(receipt.streams_agree(b"result = 49.0\n", b"result = 4.9e1\n"))
        # Trailing whitespace is a difference. This is the whole point of
        # comparing complete streams rather than parsed values.
        self.assertFalse(receipt.streams_agree(b"result = 49.0\n", b"result = 49.0 \n"))


# ---------------------------------------------------------------------------
# §6.2 / §6.3 -- manifest validation, every field with its negative row
# ---------------------------------------------------------------------------


class ManifestValidation(unittest.TestCase):
    def test_minimal_manifest_is_valid(self):
        parsed = receipt.parse_manifest(minimal_manifest())
        self.assertEqual(parsed.manifest_version, 1)
        self.assertEqual(len(parsed.cases), 2)
        self.assertEqual(len(parsed.exclusions), 1)

    def assert_rejects(self, mutate, expected_fragment: str):
        data = minimal_manifest()
        mutate(data)
        with self.assertRaises(receipt.ManifestError) as ctx:
            receipt.parse_manifest(data)
        self.assertIn(expected_fragment, str(ctx.exception))

    def test_missing_manifest_version_rejected(self):
        self.assert_rejects(lambda d: d.pop("manifest_version"), "manifest_version")

    def test_non_integer_manifest_version_rejected(self):
        self.assert_rejects(
            lambda d: d.__setitem__("manifest_version", "1"), "manifest_version"
        )

    def test_boolean_manifest_version_rejected(self):
        # `True` is an int in Python; the validator must not accept it.
        self.assert_rejects(
            lambda d: d.__setitem__("manifest_version", True), "manifest_version"
        )

    def test_missing_expected_outcome_rejected(self):
        # §6.2: `expected` has no default. This is the row that keeps a
        # deliberately-failing probe from counting as parity evidence.
        self.assert_rejects(
            lambda d: d["cases"][0].pop("expected"), "outcome-undetermined"
        )

    def test_unknown_expected_outcome_rejected(self):
        self.assert_rejects(
            lambda d: d["cases"][0].__setitem__("expected", "maybe"), "expected"
        )

    def test_duplicate_case_id_rejected(self):
        self.assert_rejects(
            lambda d: d["cases"][1].__setitem__("case_id", d["cases"][0]["case_id"]),
            "duplicate case_id",
        )

    def test_unknown_corpus_rejected(self):
        self.assert_rejects(
            lambda d: d["cases"][0].__setitem__("corpus", "nautilus"), "corpus"
        )

    def test_corpus_without_pinned_revision_rejected(self):
        def mutate(data):
            data["corpora"]["sonar"] = {
                "repo": "Chelis-Lang/sonar",
                "rev": "unpinned",
                "expected_case_count": 0,
            }

        self.assert_rejects(mutate, "exact pinned revision")

    def test_empty_corpus_revision_rejected(self):
        self.assert_rejects(
            lambda d: d["corpora"]["c-note"].__setitem__("rev", ""),
            "exact pinned revision",
        )

    def test_committed_source_without_path_rejected(self):
        self.assert_rejects(
            lambda d: d["cases"][0]["source"].pop("path"), "missing required field"
        )

    def test_derived_source_requires_generator_and_task(self):
        def mutate(data):
            data["corpora"]["voyage"] = {
                "repo": "Chelis-Lang/Voyage",
                "rev": "1" * 40,
                "expected_case_count": 1,
            }
            data["cases"].append(
                {
                    "case_id": "voyage/1-0",
                    "corpus": "voyage",
                    "source": {
                        "kind": "derived",
                        "repo": "Chelis-Lang/Voyage",
                        "rev": "1" * 40,
                        # `generator`, `task` and `index` all omitted
                    },
                    "expected": "value",
                    "roots": ["result"],
                }
            )

        self.assert_rejects(mutate, "missing required field")

    def test_derived_source_with_all_fields_is_valid(self):
        data = minimal_manifest()
        data["corpora"]["voyage"] = {
            "repo": "Chelis-Lang/Voyage",
            "rev": "1" * 40,
            "expected_case_count": 1,
        }
        data["cases"].append(
            {
                "case_id": "voyage/1-0",
                "corpus": "voyage",
                "source": {
                    "kind": "derived",
                    "repo": "Chelis-Lang/Voyage",
                    "rev": "1" * 40,
                    "generator": "probes/qcb-compiled/driver.py interp",
                    "task": 1,
                    "index": 0,
                },
                "expected": "value",
                "roots": ["result"],
            }
        )
        parsed = receipt.parse_manifest(data)
        self.assertEqual(len(parsed.cases), 3)

    def test_unknown_source_kind_rejected(self):
        self.assert_rejects(
            lambda d: d["cases"][0]["source"].__setitem__("kind", "inline"),
            "source.kind",
        )

    def test_unknown_exclusion_reason_rejected(self):
        # §6.3: an unrecognised reason is a receipt failure, not a warning.
        self.assert_rejects(
            lambda d: d["exclusions"][0].__setitem__("reason", "looked-weird"),
            "closed set",
        )

    def test_every_closed_reason_is_accepted(self):
        for reason in sorted(receipt.CLOSED_EXCLUSION_REASONS):
            data = minimal_manifest()
            data["exclusions"][0]["reason"] = reason
            parsed = receipt.parse_manifest(data)
            self.assertEqual(parsed.exclusions[0].reason, reason)

    def test_parse_rejected_and_retired_syntax_are_distinct_reasons(self):
        # §6.3 keeps them separate on purpose: the first is a property of the
        # program, the second of the pin. Collapsing them hides migration debt.
        self.assertIn("parse-rejected", receipt.CLOSED_EXCLUSION_REASONS)
        self.assertIn("retired-syntax", receipt.CLOSED_EXCLUSION_REASONS)

    def test_duplicate_exclusion_path_rejected(self):
        def mutate(data):
            data["exclusions"].append(dict(data["exclusions"][0]))

        self.assert_rejects(mutate, "duplicate exclusion")

    def test_known_divergence_requires_an_issue_number(self):
        self.assert_rejects(
            lambda d: d["cases"][0].__setitem__("known_divergence", {}),
            "positive integer",
        )

    def test_known_divergence_rejects_a_non_positive_issue(self):
        self.assert_rejects(
            lambda d: d["cases"][0].__setitem__("known_divergence", {"issue": 0}),
            "positive integer",
        )

    def test_known_divergence_with_an_issue_is_valid(self):
        data = minimal_manifest()
        data["cases"][0]["known_divergence"] = {"issue": 2782}
        parsed = receipt.parse_manifest(data)
        self.assertEqual(parsed.cases[0].known_divergence, {"issue": 2782})

    def test_truncating_roots_must_name_owned_roots(self):
        self.assert_rejects(
            lambda d: d["cases"][0].__setitem__("truncating_roots", ["not_a_root"]),
            "does not",
        )

    def test_truncating_roots_naming_an_owned_root_is_valid(self):
        data = minimal_manifest()
        data["cases"][0]["truncating_roots"] = ["result"]
        parsed = receipt.parse_manifest(data)
        self.assertEqual(parsed.cases[0].truncating_roots, ["result"])

    def test_missing_exclusions_array_rejected(self):
        self.assert_rejects(lambda d: d.pop("exclusions"), "exclusions")

    def test_empty_exclusions_array_is_valid(self):
        data = minimal_manifest()
        data["exclusions"] = []
        self.assertEqual(receipt.parse_manifest(data).exclusions, [])

    def test_invalid_json_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(receipt.ManifestError) as ctx:
                receipt.load_manifest(path)
            self.assertIn("not valid JSON", str(ctx.exception))


# ---------------------------------------------------------------------------
# §2 / §3 -- classification, every verdict with its negative partner
# ---------------------------------------------------------------------------


class Classification(unittest.TestCase):
    def test_value_case_agreeing_bytes_passes(self):
        verdict, detail = receipt.classify_case(
            case_row(),
            lane(stdout=b"result = 49.0\n"),
            lane(stdout=b"result = 49.0\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_AGREE)
        self.assertEqual(detail, "")

    def test_value_case_differing_bytes_is_an_observation_mismatch(self):
        verdict, detail = receipt.classify_case(
            case_row(),
            lane(stdout=b"result = 49.0\n"),
            lane(stdout=b"result = 49.000001\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_OBSERVATION_MISMATCH)
        self.assertIn("first difference at byte", detail)

    def test_eval_trapping_while_compiled_returns_is_a_lane_split(self):
        verdict, detail = receipt.classify_case(
            case_row(),
            lane(returncode=1, stderr=b"error: nope\n"),
            lane(stdout=b"result = 49.0\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_LANE_SPLIT)
        self.assertIn("eval lane trapped", detail)

    def test_compiled_trapping_while_eval_returns_is_a_lane_split(self):
        verdict, detail = receipt.classify_case(
            case_row(),
            lane(stdout=b"result = 49.0\n"),
            lane(returncode=134, stderr=b"abort\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_LANE_SPLIT)
        self.assertIn("compiled lane trapped", detail)

    def test_lane_split_outranks_a_byte_difference(self):
        # Severity ordering (§ classify_case docstring): "one lane trapped and
        # the other did not" is a stronger statement than "their bytes differ",
        # so a case that is both must report the split.
        verdict, _ = receipt.classify_case(
            case_row(),
            lane(returncode=1, stdout=b"partial\n"),
            lane(returncode=0, stdout=b"different\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_LANE_SPLIT)

    def test_value_case_where_both_lanes_trap_is_a_wrong_outcome_not_parity(self):
        # The lanes AGREE with each other and disagree with the manifest. That
        # distinction matters: reporting it as a parity failure would blame the
        # compiler for a stale manifest row.
        verdict, detail = receipt.classify_case(
            case_row(expected="value"),
            lane(returncode=1, stderr=b"error: x\n"),
            lane(returncode=1, stderr=b"error: x\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_WRONG_OUTCOME)
        self.assertIn("expects a value", detail)

    def test_trap_case_with_identical_status_and_stderr_passes(self):
        verdict, detail = receipt.classify_case(
            case_row(expected="trap"),
            lane(returncode=1, stderr=b"error: domain trap\n"),
            lane(returncode=1, stderr=b"error: domain trap\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_AGREE)
        self.assertEqual(detail, "")

    def test_trap_case_with_differing_status_is_a_status_mismatch(self):
        verdict, detail = receipt.classify_case(
            case_row(expected="trap"),
            lane(returncode=1, stderr=b"error: domain trap\n"),
            lane(returncode=134, stderr=b"error: domain trap\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_TRAP_STATUS_MISMATCH)
        self.assertIn("134", detail)

    def test_trap_case_with_differing_reason_is_a_reason_mismatch(self):
        # [05-OBS-6]: a lane that cannot produce a root it owes emits a
        # diagnostic naming the root, the lane and the reason. Two lanes
        # trapping for different reasons is a divergence, not agreement.
        verdict, detail = receipt.classify_case(
            case_row(expected="trap"),
            lane(returncode=1, stderr=b"error: divide by zero\n"),
            lane(returncode=1, stderr=b"error: index out of bounds\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_TRAP_REASON_MISMATCH)
        self.assertIn("first difference at byte", detail)

    def test_trap_case_where_both_lanes_succeed_is_a_wrong_outcome(self):
        verdict, detail = receipt.classify_case(
            case_row(expected="trap"),
            lane(stdout=b"result = 1.0\n"),
            lane(stdout=b"result = 1.0\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_WRONG_OUTCOME)
        self.assertIn("expects a trap", detail)

    def test_value_case_ignores_stderr_differences(self):
        # Warnings on stderr are not observations. A value case is decided on
        # stdout; folding stderr in would make every warning a divergence.
        verdict, _ = receipt.classify_case(
            case_row(expected="value"),
            lane(stdout=b"result = 1.0\n", stderr=b"warning: something\n"),
            lane(stdout=b"result = 1.0\n", stderr=b""),
        )
        self.assertEqual(verdict, receipt.VERDICT_AGREE)


class DeclaredRootPresence(unittest.TestCase):
    """§5's third clause: agreement over an absent observation is not evidence.

    Round 1's uncovered vacuity route. `roots` was validated at parse time and
    never read again, so two agreeing empty streams reported whole-corpus
    agreement, exited 0, and flipped every `known_divergence` row to
    `unexpected-pass` -- which §6.2 reads as "drop the `demo-path` tag".
    """

    def test_agreement_with_every_declared_root_present_passes(self):
        out = b"price = 1.0\ndelta = 2.0\n"
        verdict, detail = receipt.classify_case(
            case_row(roots=["price", "delta"]), lane(stdout=out), lane(stdout=out)
        )
        self.assertEqual(verdict, receipt.VERDICT_AGREE)
        self.assertEqual(detail, "")

    def test_two_agreeing_empty_streams_are_not_agreement(self):
        verdict, detail = receipt.classify_case(
            case_row(roots=["result"]), lane(stdout=b""), lane(stdout=b"")
        )
        self.assertEqual(verdict, receipt.VERDICT_MISSING_DECLARED_ROOT)
        self.assertIn("result", detail)

    def test_one_missing_root_among_several_is_caught(self):
        out = b"price = 1.0\n"
        verdict, detail = receipt.classify_case(
            case_row(roots=["price", "delta"]), lane(stdout=out), lane(stdout=out)
        )
        self.assertEqual(verdict, receipt.VERDICT_MISSING_DECLARED_ROOT)
        self.assertIn("delta", detail)
        self.assertNotIn("'price'", detail)

    def test_a_missing_root_is_not_absorbed_by_a_known_divergence(self):
        # A silenced observation channel is never a tracked divergence. Asserting
        # only its membership in UNTRACKED_FAILING_VERDICTS was not enough -- that
        # is exactly what makes a verdict RELABELLABLE, so a blackout on a
        # `known_divergence` row was being reported as `expected-failing` and the
        # row meant to stay visible was hiding the blackout.
        verdict, detail = receipt.apply_known_divergence(
            case_row(known_divergence={"issue": 2379}),
            receipt.VERDICT_MISSING_DECLARED_ROOT,
            "both lanes agree, but the case declares root(s) ['delta']",
        )
        self.assertEqual(verdict, receipt.VERDICT_MISSING_DECLARED_ROOT)
        self.assertNotIn("chelis#2379", detail)

    def test_other_failures_on_a_known_row_are_still_absorbed(self):
        # The negative partner: the exemption is specific to a missing root.
        verdict, detail = receipt.apply_known_divergence(
            case_row(known_divergence={"issue": 2379}),
            receipt.VERDICT_LANE_SPLIT,
            "compiled lane trapped",
        )
        self.assertEqual(verdict, receipt.VERDICT_EXPECTED_FAILING)
        self.assertIn("chelis#2379", detail)

    def test_a_missing_root_still_fails_the_receipt(self):
        self.assertIn(
            receipt.VERDICT_MISSING_DECLARED_ROOT, receipt.UNTRACKED_FAILING_VERDICTS
        )
        self.assertIn(receipt.VERDICT_MISSING_DECLARED_ROOT, receipt.FAILING_VERDICTS)

    def test_a_trap_case_owes_no_rendered_root(self):
        # The negative partner: the check must not fire on a trap case, whose
        # lanes legitimately render nothing.
        verdict, _ = receipt.classify_case(
            case_row(expected="trap", roots=["result"]),
            lane(returncode=1, stderr=b"error: x\n"),
            lane(returncode=1, stderr=b"error: x\n"),
        )
        self.assertEqual(verdict, receipt.VERDICT_AGREE)

    def test_a_declared_root_is_matched_by_label_not_substring(self):
        # `delta` must not be satisfied by `call_delta = …`; a label is the
        # whole name before ` = ` at line start.
        out = b"call_delta = 2.0\n"
        verdict, _ = receipt.classify_case(
            case_row(roots=["delta"]), lane(stdout=out), lane(stdout=out)
        )
        self.assertEqual(verdict, receipt.VERDICT_MISSING_DECLARED_ROOT)

    def test_missing_roots_helper_reports_exactly_the_absent_names(self):
        self.assertEqual(
            receipt.missing_roots(case_row(roots=["a", "b"]), b"a = 1\n"), ["b"]
        )
        self.assertEqual(
            receipt.missing_roots(case_row(roots=["a"]), b"a = 1\n"), []
        )


class KnownDivergenceRelabelling(unittest.TestCase):
    def test_divergence_on_a_known_row_becomes_expected_failing(self):
        verdict, detail = receipt.apply_known_divergence(
            case_row(known_divergence={"issue": 2782}),
            receipt.VERDICT_OBSERVATION_MISMATCH,
            "first difference at byte 12",
        )
        self.assertEqual(verdict, receipt.VERDICT_EXPECTED_FAILING)
        self.assertIn("chelis#2782", detail)

    def test_agreement_on_a_known_row_becomes_unexpected_pass(self):
        verdict, detail = receipt.apply_known_divergence(
            case_row(known_divergence={"issue": 2782}),
            receipt.VERDICT_AGREE,
            "",
        )
        self.assertEqual(verdict, receipt.VERDICT_UNEXPECTED_PASS)
        self.assertIn("demo-path", detail)

    def test_a_case_without_a_known_row_is_untouched(self):
        verdict, detail = receipt.apply_known_divergence(
            case_row(), receipt.VERDICT_OBSERVATION_MISMATCH, "detail"
        )
        self.assertEqual(verdict, receipt.VERDICT_OBSERVATION_MISMATCH)
        self.assertEqual(detail, "detail")

    def test_a_tracked_divergence_still_fails_the_receipt(self):
        # #1362's ship rule is "no `demo-path` row remains open", and this
        # manifest is `demo-path`'s only authority, so a tracked divergence
        # must not buy a pass. It buys a distinguishable exit code.
        self.assertIn(receipt.VERDICT_EXPECTED_FAILING, receipt.FAILING_VERDICTS)
        self.assertNotIn(
            receipt.VERDICT_EXPECTED_FAILING, receipt.UNTRACKED_FAILING_VERDICTS
        )

    def test_an_unexpected_pass_is_not_a_failure(self):
        self.assertNotIn(receipt.VERDICT_UNEXPECTED_PASS, receipt.FAILING_VERDICTS)


class DifferenceReporting(unittest.TestCase):
    def test_names_the_first_differing_offset(self):
        detail = receipt._first_difference(b"result = 1.0\n", b"result = 2.0\n")
        self.assertIn("byte 9", detail)

    def test_reports_a_pure_length_difference_distinctly(self):
        # A truncated stream shares every byte it has. "first difference at
        # byte N" would be misleading, so this case is reported as a length
        # difference with both tails.
        detail = receipt._first_difference(b"a = 1.0\n", b"a = 1.0\nb = 2.0\n")
        self.assertIn("differ in length", detail)
        self.assertIn("b = 2.0", detail)

    def test_identical_streams_are_never_passed_to_the_reporter(self):
        # Guard the contract rather than the output: the reporter is only
        # reached from a failing branch.
        self.assertTrue(receipt.streams_agree(b"same", b"same"))


# ---------------------------------------------------------------------------
# §5.2 -- truncation-aware observation accounting
# ---------------------------------------------------------------------------


class TruncationAccounting(unittest.TestCase):
    def test_marker_in_output_is_detected(self):
        self.assertTrue(
            receipt.observation_truncated(
                b"t = tensor(shape=[100], data=[1.0, 2.0, ...])\n"
            )
        )

    def test_untruncated_output_is_not_flagged(self):
        self.assertFalse(
            receipt.observation_truncated(b"t = tensor(shape=[2], data=[1.0, 2.0])\n")
        )

    def test_scalar_output_is_not_flagged(self):
        self.assertFalse(receipt.observation_truncated(b"result = 49.0\n"))

    def test_coverage_separates_truncated_from_untruncated_agreement(self):
        verdicts = [
            receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, truncated=False),
            receipt.CaseVerdict("b", "c-note", receipt.VERDICT_AGREE, truncated=True),
            receipt.CaseVerdict("c", "c-note", receipt.VERDICT_AGREE, truncated=True),
            receipt.CaseVerdict(
                "d", "c-note", receipt.VERDICT_OBSERVATION_MISMATCH, truncated=False
            ),
        ]
        coverage = receipt.observation_coverage(verdicts)
        self.assertEqual(coverage["agreed"], 3)
        self.assertEqual(coverage["agreed_with_no_truncated_root"], 1)
        self.assertEqual(coverage["agreed_relying_on_truncated_rendering"], 2)

    def test_an_all_truncated_corpus_reports_zero_untruncated_agreement(self):
        # The negative partner: a corpus whose agreement rests entirely on
        # truncated renderings must not report full observation coverage.
        verdicts = [
            receipt.CaseVerdict(str(i), "c-note", receipt.VERDICT_AGREE, truncated=True)
            for i in range(5)
        ]
        coverage = receipt.observation_coverage(verdicts)
        self.assertEqual(coverage["agreed"], 5)
        self.assertEqual(coverage["agreed_with_no_truncated_root"], 0)


# ---------------------------------------------------------------------------
# §5.1 -- the five structural non-vacuity rules
# ---------------------------------------------------------------------------


class NonVacuity(unittest.TestCase):
    def manifest(self) -> receipt.Manifest:
        return receipt.parse_manifest(minimal_manifest())

    def passing_verdicts(self) -> list[receipt.CaseVerdict]:
        return [
            receipt.CaseVerdict(
                "c-note/square", "c-note", receipt.VERDICT_AGREE, comparable=True
            ),
            receipt.CaseVerdict(
                "c-note/refuses", "c-note", receipt.VERDICT_AGREE, comparable=True
            ),
        ]

    def test_a_complete_run_has_no_failures(self):
        failures = receipt.non_vacuity_failures(
            self.manifest(),
            {"c-note": {"a.ch", "b.ch", "lib.ch"}},
            self.passing_verdicts(),
        )
        self.assertEqual(failures, [])

    def test_rule_1_zero_required_cases_fails(self):
        data = minimal_manifest()
        data["cases"] = []
        data["corpora"]["c-note"]["expected_case_count"] = 0
        failures = receipt.non_vacuity_failures(
            receipt.parse_manifest(data), {"c-note": set()}, []
        )
        self.assertTrue(any("rule 1" in f for f in failures))

    def test_rule_2_a_required_case_with_no_verdict_fails(self):
        failures = receipt.non_vacuity_failures(
            self.manifest(),
            {"c-note": {"a.ch", "b.ch", "lib.ch"}},
            self.passing_verdicts()[:1],
        )
        self.assertTrue(any("rule 2" in f for f in failures))
        self.assertTrue(any("c-note/refuses" in f for f in failures))

    def test_rule_3_an_unclassified_discovered_file_fails(self):
        # This is the rule that makes the manifest per-case rather than a
        # directory glob: a probe nobody classified cannot slip in.
        failures = receipt.non_vacuity_failures(
            self.manifest(),
            {"c-note": {"a.ch", "b.ch", "lib.ch", "probes/p01.ch"}},
            self.passing_verdicts(),
        )
        self.assertTrue(any("rule 3" in f for f in failures))
        self.assertTrue(any("probes/p01.ch" in f for f in failures))

    def test_rule_3_an_excluded_file_is_classified(self):
        # The negative partner: recording the file as an exclusion clears it.
        data = minimal_manifest()
        data["exclusions"].append(
            {"corpus": "c-note", "path": "probes/p01.ch", "reason": "prove-only"}
        )
        failures = receipt.non_vacuity_failures(
            receipt.parse_manifest(data),
            {"c-note": {"a.ch", "b.ch", "lib.ch", "probes/p01.ch"}},
            self.passing_verdicts(),
        )
        self.assertEqual(failures, [])

    def test_rule_4_zero_comparable_observations_fails(self):
        verdicts = [
            receipt.CaseVerdict(
                "c-note/square",
                "c-note",
                receipt.VERDICT_HARNESS_FAILURE,
                comparable=False,
            ),
            receipt.CaseVerdict(
                "c-note/refuses",
                "c-note",
                receipt.VERDICT_HARNESS_FAILURE,
                comparable=False,
            ),
        ]
        failures = receipt.non_vacuity_failures(
            self.manifest(), {"c-note": {"a.ch", "b.ch", "lib.ch"}}, verdicts
        )
        self.assertTrue(any("rule 4" in f for f in failures))

    def test_rule_5_a_count_mismatch_fails(self):
        data = minimal_manifest()
        data["corpora"]["c-note"]["expected_case_count"] = 7
        failures = receipt.non_vacuity_failures(
            receipt.parse_manifest(data),
            {"c-note": {"a.ch", "b.ch", "lib.ch"}},
            self.passing_verdicts(),
        )
        self.assertTrue(any("rule 5" in f for f in failures))
        self.assertTrue(any("7" in f for f in failures))

    def test_a_corpus_not_walked_is_not_checked_for_rule_3(self):
        # Absence of a corpus from the discovery map is reported by the caller
        # as a pin failure, not silently turned into a rule-3 violation.
        failures = receipt.non_vacuity_failures(
            self.manifest(), {}, self.passing_verdicts()
        )
        self.assertFalse(any("rule 3" in f for f in failures))


# ---------------------------------------------------------------------------
# §4 -- pins
# ---------------------------------------------------------------------------


class PinChecking(unittest.TestCase):
    def pins(self, **overrides) -> dict:
        base = {
            "compiler": {
                "path": "/tmp/chelis",
                "version_string": "chelis 0.19.0",
                "sealed_runtime": True,
                "sha256": "a" * 64,
            },
            "corpora": {
                "c-note": {
                    "repo": "Chelis-Lang/c-note",
                    "declared_rev": "0" * 40,
                    "root": "/tmp/c-note",
                    "observed_rev": "0" * 40,
                }
            },
        }
        base.update(overrides)
        return base

    def test_matching_pins_have_no_failures(self):
        self.assertEqual(receipt.pin_failures(self.pins()), [])

    def test_a_revision_mismatch_fails(self):
        pins = self.pins()
        pins["corpora"]["c-note"]["observed_rev"] = "1" * 40
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("manifest pins" in f for f in failures))

    def test_an_unsealed_runtime_build_fails(self):
        # §4.2: an unsealed build re-checks its source checkout on every
        # `build`, so a concurrent writer can change its behaviour mid-run
        # while its hash stays constant. A hash alone does not pin it.
        pins = self.pins()
        pins["compiler"]["sealed_runtime"] = False
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("sealed-runtime" in f for f in failures))

    def test_an_unknown_seal_state_fails(self):
        # `None` means the probe could not tell, and an unpinnable compiler is
        # not evidence. Round 1 found this pinned open: the installed release
        # toolchains write no staging receipt at all, so `None` is the common
        # case, and accepting it made the gate inert for exactly the
        # release-candidate compiler class §4.2 exists for.
        pins = self.pins()
        pins["compiler"]["sealed_runtime"] = None
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("could not read" in f for f in failures))

    def test_only_a_positively_read_sealed_mode_clears_the_pin_check(self):
        # The three-way control: True clears, False fails, None fails.
        for value, should_clear in ((True, True), (False, False), (None, False)):
            pins = self.pins()
            pins["compiler"]["sealed_runtime"] = value
            failures = receipt.pin_failures(pins)
            self.assertEqual(
                failures == [],
                should_clear,
                f"sealed_runtime={value!r} must "
                f"{'clear' if should_clear else 'fail'} the pin check",
            )

    def test_a_missing_corpus_checkout_fails(self):
        pins = self.pins()
        pins["corpora"]["c-note"]["root"] = None
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("no checkout supplied" in f for f in failures))

    def test_a_non_git_corpus_checkout_fails(self):
        pins = self.pins()
        pins["corpora"]["c-note"]["observed_rev"] = None
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("not a git checkout" in f for f in failures))

    def test_a_compiler_with_no_version_output_fails(self):
        pins = self.pins()
        pins["compiler"]["version_string"] = ""
        failures = receipt.pin_failures(pins)
        self.assertTrue(any("--version" in f for f in failures))


# ---------------------------------------------------------------------------
# §4.3 -- compile profiles and the compile line
# ---------------------------------------------------------------------------


class StagingReceiptMode(unittest.TestCase):
    """The build mode comes from the compiler's own staging receipt (§4.2)."""

    def test_sealed_mode_reads_true(self):
        self.assertIs(receipt._sealed_from_staging({"mode": "sealed"}), True)

    def test_development_mode_reads_false(self):
        self.assertIs(receipt._sealed_from_staging({"mode": "development"}), False)

    def test_an_absent_receipt_is_unknown_not_sealed(self):
        # The negative that matters: a build that wrote no receipt must not
        # read as sealed, or the §4.2 pin check silently stops firing.
        self.assertIsNone(receipt._sealed_from_staging(None))

    def test_an_unrecognised_mode_is_unknown_not_sealed(self):
        self.assertIsNone(receipt._sealed_from_staging({"mode": "hermetic"}))

    def test_a_receipt_without_a_mode_field_is_unknown(self):
        self.assertIsNone(receipt._sealed_from_staging({"schema": "x"}))

    def test_probe_returns_none_when_the_build_refuses(self):
        class RefusingRunner(receipt.Runner):
            def run(self, argv, cwd):
                return lane(1, stderr=b"error: nope\n")

        self.assertIsNone(
            receipt.probe_staging_receipt(
                Path("/tmp/chelis"), RefusingRunner(chelis=Path("/tmp/chelis"))
            )
        )

    def test_probe_reads_the_receipt_a_successful_build_wrote(self):
        class StagingRunner(receipt.Runner):
            def run(self, argv, cwd):
                out = Path(cwd) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "chelis_runtime.receipt.json").write_text(
                    json.dumps({"mode": "sealed", "archive_sha256": "ab"}),
                    encoding="utf-8",
                )
                return lane(0)

        staging = receipt.probe_staging_receipt(
            Path("/tmp/chelis"), StagingRunner(chelis=Path("/tmp/chelis"))
        )
        self.assertEqual(staging, {"mode": "sealed", "archive_sha256": "ab"})
        self.assertIs(receipt._sealed_from_staging(staging), True)

    def test_probe_returns_none_when_the_receipt_is_malformed(self):
        class BadRunner(receipt.Runner):
            def run(self, argv, cwd):
                out = Path(cwd) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "chelis_runtime.receipt.json").write_text(
                    "{not json", encoding="utf-8"
                )
                return lane(0)

        self.assertIsNone(
            receipt.probe_staging_receipt(
                Path("/tmp/chelis"), BadRunner(chelis=Path("/tmp/chelis"))
            )
        )


class RecordedProvenance(unittest.TestCase):
    """§4.2's compiler fields must not include one that can lie (round 1 F5)."""

    def test_the_harness_revision_is_named_as_the_harness_revision(self):
        # It used to be `repo_revision` under "the git revision for the
        # compiler under test", so pointing --chelis at an installed toolchain
        # recorded this worktree's HEAD as the compiler's provenance.
        source = Path(receipt.__file__).read_text(encoding="utf-8")
        self.assertIn('"harness_repo_revision"', source)
        self.assertNotIn('"repo_revision"', source)

    def test_pin_failures_never_consults_the_harness_revision(self):
        # The negative half: the field is recorded, not used as compiler
        # evidence, so removing it cannot change a verdict.
        pins = {
            "compiler": {
                "version_string": "chelis 0.19.0",
                "sealed_runtime": True,
                "harness_repo_revision": None,
            },
            "corpora": {},
        }
        self.assertEqual(receipt.pin_failures(pins), [])


class CompileCommand(unittest.TestCase):
    EMITTED = (
        "Wrote out/k.c and out/k.h\n"
        "Compile: clang -O2 -march=native out/k.c out/libchelis_runtime.a "
        "-lm -framework Accelerate -o out/k\n"
    )

    def test_extracts_the_single_compile_line(self):
        command = receipt.extract_compile_command(self.EMITTED.encode())
        self.assertTrue(command.startswith("clang -O2 -march=native"))
        self.assertNotIn("Compile:", command)

    def test_no_compile_line_is_a_receipt_error(self):
        with self.assertRaises(receipt.ReceiptError) as ctx:
            receipt.extract_compile_command(b"Wrote out/k.c\n")
        self.assertIn("no `Compile:` line", str(ctx.exception))

    def test_two_compile_lines_is_a_receipt_error(self):
        # Ambiguity is refused rather than resolved by picking one. Guessing
        # here would silently compile the wrong artifact.
        doubled = self.EMITTED + "Compile: gcc -O2 x.c -o x\n"
        with self.assertRaises(receipt.ReceiptError) as ctx:
            receipt.extract_compile_command(doubled.encode())
        self.assertIn("2 `Compile:` lines", str(ctx.exception))

    def test_emitted_profile_is_the_identity(self):
        command = "clang -O2 -march=native k.c -o k"
        self.assertEqual(receipt.apply_compile_profile(command, "emitted"), command)

    def test_no_fp_contract_profile_inserts_the_flag_after_the_compiler(self):
        # chelis#2782's candidate repair. Inserting after the compiler name
        # keeps the flag out of a linker argument list.
        rewritten = receipt.apply_compile_profile(
            "clang -O2 -march=native k.c -o k", "no-fp-contract"
        )
        self.assertEqual(
            rewritten, "clang -ffp-contract=off -O2 -march=native k.c -o k"
        )

    def test_no_fp_contract_profile_is_idempotent(self):
        already = "clang -ffp-contract=off -O2 k.c -o k"
        self.assertEqual(
            receipt.apply_compile_profile(already, "no-fp-contract"), already
        )

    def test_the_profile_does_not_depend_on_o2_being_present(self):
        # The flag is inserted positionally, not substituted into `-O2`, so a
        # build that stops printing `-O2` does not silently lose the override.
        rewritten = receipt.apply_compile_profile("gcc k.c -o k", "no-fp-contract")
        self.assertEqual(rewritten, "gcc -ffp-contract=off k.c -o k")

    def test_an_unknown_profile_is_refused(self):
        with self.assertRaises(receipt.ReceiptError):
            receipt.apply_compile_profile("gcc k.c -o k", "fast-and-loose")

    def test_every_declared_profile_is_implemented(self):
        for profile in receipt.COMPILE_PROFILES:
            receipt.apply_compile_profile("gcc k.c -o k", profile)


# ---------------------------------------------------------------------------
# Receipt assembly and the pass/fail decision
# ---------------------------------------------------------------------------


class ReceiptDecision(unittest.TestCase):
    def receipt_for(self, verdicts, non_vacuity=(), pin_problems=()) -> dict:
        return receipt.build_receipt(
            receipt.parse_manifest(minimal_manifest()),
            {
                "compiler": {"version_string": "chelis 0.19.0", "path": "/tmp/chelis"},
                "compile_profile": "emitted",
                "corpora": {},
            },
            verdicts,
            non_vacuity,
            pin_problems,
        )

    def test_all_agreeing_passes(self):
        built = self.receipt_for(
            [
                receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True),
                receipt.CaseVerdict("b", "c-note", receipt.VERDICT_AGREE, comparable=True),
            ]
        )
        self.assertTrue(receipt.receipt_passes(built))

    def test_any_failing_verdict_fails(self):
        for verdict in receipt.FAILING_VERDICTS:
            built = self.receipt_for(
                [
                    receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True),
                    receipt.CaseVerdict("b", "c-note", verdict, comparable=True),
                ]
            )
            self.assertFalse(
                receipt.receipt_passes(built),
                f"{verdict} must not pass the receipt",
            )

    def test_a_non_vacuity_failure_fails_even_with_every_case_agreeing(self):
        # The important one: a green case set over a vacuous discovery is
        # exactly what #2102 exists to reject.
        built = self.receipt_for(
            [receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True)],
            non_vacuity=["rule 3: c-note: 1 discovered file(s) unclassified"],
        )
        self.assertFalse(receipt.receipt_passes(built))

    def test_a_pin_failure_fails_even_with_every_case_agreeing(self):
        built = self.receipt_for(
            [receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True)],
            pin_problems=["corpus c-note: manifest pins X but the checkout is at Y"],
        )
        self.assertFalse(receipt.receipt_passes(built))

    def test_expected_failing_fails_the_receipt_with_its_own_exit_code(self):
        built = self.receipt_for(
            [
                receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True),
                receipt.CaseVerdict(
                    "b", "c-note", receipt.VERDICT_EXPECTED_FAILING, comparable=True
                ),
            ]
        )
        self.assertFalse(receipt.receipt_passes(built))
        self.assertEqual(
            receipt.receipt_exit_code(built), receipt.EXIT_TRACKED_FAILURE_ONLY
        )
        self.assertEqual(built["verdict_counts"][receipt.VERDICT_EXPECTED_FAILING], 1)

    def test_an_untracked_failure_beside_a_tracked_one_reports_exit_1(self):
        # The negative partner: a new divergence must not be masked by the
        # presence of tracked ones.
        built = self.receipt_for(
            [
                receipt.CaseVerdict(
                    "a", "c-note", receipt.VERDICT_EXPECTED_FAILING, comparable=True
                ),
                receipt.CaseVerdict(
                    "b", "c-note", receipt.VERDICT_OBSERVATION_MISMATCH, comparable=True
                ),
            ]
        )
        self.assertEqual(
            receipt.receipt_exit_code(built), receipt.EXIT_UNTRACKED_FAILURE
        )

    def test_a_clean_run_reports_exit_0(self):
        built = self.receipt_for(
            [receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True)]
        )
        self.assertEqual(receipt.receipt_exit_code(built), receipt.EXIT_PASS)

    def test_a_vacuous_run_reports_exit_1_not_3(self):
        # A vacuous run is never "only tracked failures", however clean the
        # case verdicts look.
        built = self.receipt_for(
            [receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True)],
            non_vacuity=["rule 4: zero comparable observations"],
        )
        self.assertEqual(
            receipt.receipt_exit_code(built), receipt.EXIT_UNTRACKED_FAILURE
        )

    def test_the_receipt_records_every_command_verbatim(self):
        built = self.receipt_for(
            [
                receipt.CaseVerdict(
                    "a",
                    "c-note",
                    receipt.VERDICT_AGREE,
                    comparable=True,
                    commands=["chelis fmt --inplace k.ch", "clang -O2 k.c -o k"],
                )
            ]
        )
        self.assertEqual(
            built["cases"][0]["commands"],
            ["chelis fmt --inplace k.ch", "clang -O2 k.c -o k"],
        )

    def test_the_receipt_records_the_exclusion_ledger(self):
        built = self.receipt_for([])
        self.assertEqual(built["exclusions"][0]["reason"], "library-only")

    def test_the_receipt_is_json_serialisable(self):
        built = self.receipt_for(
            [receipt.CaseVerdict("a", "c-note", receipt.VERDICT_AGREE, comparable=True)]
        )
        json.loads(json.dumps(built))


# ---------------------------------------------------------------------------
# End-to-end over the patched subprocess layer
# ---------------------------------------------------------------------------


class FakeRunner(receipt.Runner):
    """A Runner whose four command shapes are scripted, not executed."""

    def __init__(self, eval_result, build_stdout, binary_result, fmt_ok=True):
        super().__init__(chelis=Path("/tmp/chelis"))
        self.eval_result = eval_result
        self.build_stdout = build_stdout
        self.binary_result = binary_result
        self.fmt_ok = fmt_ok
        self.shell_commands: list[str] = []

    def run(self, argv, cwd):
        argv = list(argv)
        if "fmt" in argv:
            return lane(0 if self.fmt_ok else 1, stderr=b"not formattable\n")
        if "eval" in argv:
            return self.eval_result
        if "build" in argv:
            return lane(0, stdout=self.build_stdout)
        return self.binary_result

    def run_shell(self, command, cwd):
        self.shell_commands.append(command)
        return lane(0)


BUILD_STDOUT = b"Compile: clang -O2 out/k.c -o out/k\n"


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self._tmp.name)
        (self.corpus / "a.ch").write_text("result = 1\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_an_agreeing_case_runs_through_both_lanes(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 1\n"),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_AGREE)
        self.assertTrue(verdict.comparable)
        self.assertIn("clang -O2 out/k.c -o out/k", runner.shell_commands)

    def test_a_diverging_case_is_reported(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 2\n"),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_OBSERVATION_MISMATCH)

    def test_the_no_fp_contract_profile_reaches_the_shell(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 1\n"),
        )
        receipt.execute_case(case_row(), self.corpus, runner, "no-fp-contract")
        self.assertIn("-ffp-contract=off", runner.shell_commands[0])

    def test_a_formatting_failure_is_a_harness_failure_not_a_divergence(self):
        # The receipt never passes --allow-style-violations. A case that cannot
        # survive canonical formatting is recorded as such, and crucially is
        # NOT reported as a lane disagreement.
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 1\n"),
            fmt_ok=False,
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_HARNESS_FAILURE)
        self.assertFalse(verdict.comparable)

    def test_a_missing_committed_case_is_a_harness_failure(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b""),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(),
        )
        missing = case_row(
            source={
                "kind": "committed",
                "repo": "Chelis-Lang/c-note",
                "rev": "0" * 40,
                "path": "absent.ch",
            }
        )
        verdict = receipt.execute_case(missing, self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_HARNESS_FAILURE)
        self.assertIn("does not exist at the pinned revision", verdict.detail)

    def test_an_absent_derived_capture_names_its_generator(self):
        # §9: this receipt consumes Voyage captures and does not generate them.
        # The failure has to say so, or a reader will look for a committed file
        # that never existed.
        runner = FakeRunner(
            eval_result=lane(), build_stdout=BUILD_STDOUT, binary_result=lane()
        )
        derived = case_row(
            corpus="voyage",
            source={
                "kind": "derived",
                "repo": "Chelis-Lang/Voyage",
                "rev": "1" * 40,
                "generator": "probes/qcb-compiled/driver.py interp",
                "task": 1,
                "index": 0,
            },
        )
        verdict = receipt.execute_case(derived, self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_HARNESS_FAILURE)
        self.assertIn("driver.py interp", verdict.detail)
        self.assertIn("does not generate them", verdict.detail)

    def test_a_build_refusal_is_classified_as_the_compiled_lane_trapping(self):
        # A `build` refusal is not a harness failure: it is the compiled lane
        # trapping, and whether that is a parity failure depends on eval.
        class RefusingRunner(FakeRunner):
            def run(self, argv, cwd):
                argv = list(argv)
                if "build" in argv:
                    return lane(1, stderr=b"error: inconsistent live owners\n")
                return super().run(argv, cwd)

        runner = RefusingRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=b"",
            binary_result=lane(),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_LANE_SPLIT)
        self.assertIn("compiled lane trapped", verdict.detail)

    def test_both_lanes_refusing_a_trap_case_agrees(self):
        class RefusingRunner(FakeRunner):
            def run(self, argv, cwd):
                argv = list(argv)
                if "build" in argv or "eval" in argv:
                    return lane(1, stderr=b"error: same reason\n")
                return super().run(argv, cwd)

        runner = RefusingRunner(
            eval_result=lane(1, stderr=b"error: same reason\n"),
            build_stdout=b"",
            binary_result=lane(),
        )
        verdict = receipt.execute_case(
            case_row(expected="trap"), self.corpus, runner, "emitted"
        )
        self.assertEqual(verdict.verdict, receipt.VERDICT_AGREE)

    def test_a_case_id_carrying_path_separators_runs(self):
        # A real case_id is `<corpus>/<corpus-relative path>`, so it contains
        # separators. A tempdir prefix cannot, and the first live run died on
        # exactly that; this row keeps it dead.
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 1\n"),
        )
        verdict = receipt.execute_case(
            case_row(case_id="c-note/docs/qa_evidence/ground_truth/bs_pricing.ch"),
            self.corpus,
            runner,
            "emitted",
        )
        self.assertEqual(verdict.verdict, receipt.VERDICT_AGREE)

    def test_a_run_stage_result_is_recorded_as_run(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=b"result = 1\n"),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.compiled_stage, "run")

    def test_a_build_refusal_is_recorded_as_the_build_stage(self):
        # A build refusal and a runtime trap are different events. Without the
        # stage a reader has to infer which one happened from diagnostic text.
        class RefusingRunner(FakeRunner):
            def run(self, argv, cwd):
                argv = list(argv)
                if "build" in argv:
                    return lane(1, stderr=b"error: refused\n")
                return super().run(argv, cwd)

        runner = RefusingRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=b"",
            binary_result=lane(),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.compiled_stage, "build")

    def test_a_compile_failure_is_the_compiled_lane_trapping_at_compile(self):
        # Round 1: this used to short-circuit to `harness-failure` with no
        # stage, so "eval returns a value and the emitted C does not compile" --
        # a #1362 guarantee-2 divergence -- was attributed to the harness.
        class BadCompiler(FakeRunner):
            def run_shell(self, command, cwd):
                self.shell_commands.append(command)
                return lane(1, stderr=b"error: use of undeclared identifier\n")

        runner = BadCompiler(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(),
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertEqual(verdict.verdict, receipt.VERDICT_LANE_SPLIT)
        self.assertEqual(verdict.compiled_stage, "compile")
        self.assertTrue(verdict.comparable)

    def test_every_documented_stage_value_is_reachable(self):
        # §10 documents `build`, `compile` and `run`. Round 1 found "compile"
        # appeared nowhere in the runner, so the documented set was wrong.
        source = (
            Path(receipt.__file__).read_text(encoding="utf-8")
        )
        for stage in ("build", "compile", "run"):
            self.assertIn(
                f'compiled_stage = "{stage}"',
                source,
                f"stage {stage!r} is documented but never assigned",
            )

    def test_a_harness_failure_records_no_stage(self):
        runner = FakeRunner(
            eval_result=lane(stdout=b"result = 1\n"),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(),
            fmt_ok=False,
        )
        verdict = receipt.execute_case(case_row(), self.corpus, runner, "emitted")
        self.assertIsNone(verdict.compiled_stage)

    def test_truncated_output_is_recorded_on_the_verdict(self):
        truncated = b"t = tensor(shape=[100], data=[1.0, ...])\n"
        runner = FakeRunner(
            eval_result=lane(stdout=truncated),
            build_stdout=BUILD_STDOUT,
            binary_result=lane(stdout=truncated),
        )
        verdict = receipt.execute_case(
            case_row(roots=["t"]), self.corpus, runner, "emitted"
        )
        self.assertEqual(verdict.verdict, receipt.VERDICT_AGREE)
        self.assertTrue(verdict.truncated)


class Discovery(unittest.TestCase):
    def test_finds_nested_ch_files_as_posix_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "fixtures" / "deep").mkdir(parents=True)
            (root / "a.ch").write_text("", encoding="utf-8")
            (root / "fixtures" / "deep" / "b.ch").write_text("", encoding="utf-8")
            (root / "notes.md").write_text("", encoding="utf-8")
            self.assertEqual(
                receipt.discover_committed_paths(root),
                {"a.ch", "fixtures/deep/b.ch"},
            )

    def test_discovers_dp_programs_as_well_as_ch(self):
        # Round 1: globbing `.ch` alone made §5.3's "every file discovered in a
        # pinned corpus" false. `chelis eval --file` also reads `.dp`, and Sonar
        # carries two such programs at the pinned revision.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.ch").write_text("", encoding="utf-8")
            (root / "b.dp").write_text("", encoding="utf-8")
            (root / "notes.md").write_text("", encoding="utf-8")
            (root / "data.json").write_text("", encoding="utf-8")
            self.assertEqual(
                receipt.discover_committed_paths(root), {"a.ch", "b.dp"}
            )

    def test_the_discovered_suffix_set_is_exactly_the_two_source_forms(self):
        self.assertEqual(set(receipt.DISCOVERED_SUFFIXES), {".ch", ".dp"})

    def test_skips_the_git_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()
            (root / ".git" / "hook.ch").write_text("", encoding="utf-8")
            self.assertEqual(receipt.discover_committed_paths(root), set())


class ShippedManifest(unittest.TestCase):
    """The manifest committed in this repository must satisfy its own schema."""

    PATH = REPO_ROOT / "tests" / "corpus" / "core_fragment_parity" / "manifest.json"

    def test_the_shipped_manifest_validates(self):
        parsed = receipt.load_manifest(self.PATH)
        self.assertGreaterEqual(parsed.manifest_version, 1)

    def test_the_shipped_manifest_declares_accurate_case_counts(self):
        # Rule 5 against the real file, so a hand-edited row cannot drift from
        # the declared count without a test failing.
        parsed = receipt.load_manifest(self.PATH)
        for corpus, entry in parsed.corpora.items():
            actual = sum(1 for case in parsed.cases if case.corpus == corpus)
            self.assertEqual(
                actual,
                entry["expected_case_count"],
                f"{corpus}: expected_case_count is stale",
            )

    def test_every_known_divergence_cites_an_issue(self):
        parsed = receipt.load_manifest(self.PATH)
        for case in parsed.cases:
            if case.known_divergence is not None:
                self.assertIsInstance(case.known_divergence["issue"], int)


if __name__ == "__main__":
    unittest.main()
