"""Unit tests for the coverage runner's PURE core (no binary needed).

Run with the uv-managed interpreter:
    .venv/bin/python -m unittest -v \
        tests.corpus.opaque_invariants.test_coverage_runner
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage_runner as cr  # noqa: E402


def check_result(exit_code, errors):
    return cr.RunResult(exit_code, ({"errors": errors},))


def prove_result(exit_code, records):
    return cr.RunResult(exit_code, tuple(records))


class CheckTokenTests(unittest.TestCase):
    def test_kind_token_hits_on_matching_error_kind(self):
        r = check_result(2, [{"kind": "OpaqueTypeViolation", "message": "x"}])
        self.assertTrue(cr.token_hit("OpaqueTypeViolation", "check", r))
        self.assertFalse(cr.token_hit("DuplicateModule", "check", r))

    def test_wf_token_hits_on_message_substring(self):
        r = check_result(2, [{"kind": "OpaqueTypeViolation",
                              "message": "field access of opaque type"}])
        self.assertTrue(cr.token_hit("wf:field access", "check", r))
        self.assertFalse(cr.token_hit("wf:record update", "check", r))

    def test_pos_check_clean_requires_exit_zero_and_empty_errors(self):
        self.assertTrue(cr.token_hit("pos:check_clean", "check", check_result(0, [])))
        self.assertFalse(
            cr.token_hit("pos:check_clean", "check", check_result(0, [{"kind": "X"}]))
        )
        self.assertFalse(cr.token_hit("pos:check_clean", "check", check_result(2, [])))


class ProveTokenTests(unittest.TestCase):
    def test_obligation_status_tokens(self):
        obs = [{"kind": "obligation", "status": "passed"}]
        self.assertTrue(cr.token_hit("status:passed", "prove", prove_result(0, obs)))
        self.assertFalse(cr.token_hit("status:failed", "prove", prove_result(0, obs)))

    def test_unsupported_matches_obligation_or_property(self):
        ob = prove_result(2, [{"kind": "obligation", "status": "unsupported"}])
        prop = prove_result(2, [{"kind": "property", "status": "unsupported"}])
        self.assertTrue(cr.token_hit("status:unsupported", "prove", ob))
        self.assertTrue(cr.token_hit("status:unsupported", "prove", prop))

    def test_check_error_token_matches_check_stage_error(self):
        r = prove_result(3, [{"kind": "error", "stage": "check", "reason": "x"}])
        self.assertTrue(cr.token_hit("status:check_error", "prove", r))

    def test_reason_token_substring(self):
        r = prove_result(3, [{"kind": "obligation", "status": "error",
                             "reason": "producer `many` ... List ..."}])
        self.assertTrue(cr.token_hit("reason:List", "prove", r))
        self.assertTrue(cr.token_hit("reason:many", "prove", r))
        self.assertFalse(cr.token_hit("reason:Wrapper", "prove", r))

    def test_inject_free_not_injected_requires_no_pass_and_exit_2(self):
        r = prove_result(2, [{"kind": "property", "status": "unsupported"}])
        self.assertTrue(cr.token_hit("inject:free_not_injected", "prove", r))
        passing = prove_result(0, [{"kind": "property", "status": "passed"}])
        self.assertFalse(cr.token_hit("inject:free_not_injected", "prove", passing))

    def test_perf_many_producers_requires_accounting(self):
        obs = [{"kind": "obligation", "status": "passed"} for _ in range(40)]
        summary = [{"kind": "summary", "obligations": 40}]
        good = prove_result(0, obs + summary)
        self.assertTrue(cr.token_hit("perf:many_producers", "prove", good))
        # Too few producers: not the perf case.
        few = prove_result(0, obs[:5] + [{"kind": "summary", "obligations": 5}])
        self.assertFalse(cr.token_hit("perf:many_producers", "prove", few))
        # Summary count mismatch: accounting invariant broken.
        mismatch = prove_result(0, obs + [{"kind": "summary", "obligations": 39}])
        self.assertFalse(cr.token_hit("perf:many_producers", "prove", mismatch))

    def test_starve_token_substring_and_legacy_error(self):
        starved = prove_result(2, [{"kind": "property", "status": "unsupported",
                                   "reason": "generator starvation ... equality-atoms ... Tier B"}])
        self.assertTrue(cr.token_hit("starve:equality-atoms", "prove", starved))
        self.assertTrue(cr.token_hit("starve:Tier B", "prove", starved))
        legacy = prove_result(3, [{"kind": "property", "status": "error"}])
        self.assertTrue(cr.token_hit("starve:legacy_error", "prove", legacy))


def _manifest():
    return {
        "programs": [
            {"id": "a", "lane": "check", "targets": ["OpaqueTypeViolation"],
             "expect_exit": 2},
            {"id": "b", "lane": "prove", "targets": ["status:passed"],
             "expect_exit": 0},
        ],
        "targets": ["OpaqueTypeViolation", "status:passed"],
    }


class AggregationTests(unittest.TestCase):
    def test_measure_reports_full_coverage_when_all_hit(self):
        m = _manifest()
        results = {
            "a": check_result(2, [{"kind": "OpaqueTypeViolation", "message": "x"}]),
            "b": prove_result(0, [{"kind": "obligation", "status": "passed"}]),
        }
        report = cr.measure(m, results, m["targets"], [])
        self.assertEqual(report.uncovered, [])
        self.assertFalse(report.expect_exit_mismatches)
        cr.assert_full_coverage(report)  # does not raise

    def test_measure_detects_uncovered_target(self):
        m = _manifest()
        results = {
            "a": check_result(2, []),  # no OpaqueTypeViolation -> uncovered
            "b": prove_result(0, [{"kind": "obligation", "status": "passed"}]),
        }
        report = cr.measure(m, results, m["targets"], [])
        self.assertIn("OpaqueTypeViolation", report.uncovered)
        with self.assertRaises(cr.CoverageGateError):
            cr.assert_full_coverage(report)

    def test_measure_detects_exit_mismatch(self):
        m = _manifest()
        results = {
            "a": check_result(0, [{"kind": "OpaqueTypeViolation", "message": "x"}]),
            "b": prove_result(0, [{"kind": "obligation", "status": "passed"}]),
        }
        report = cr.measure(m, results, m["targets"], [])
        self.assertTrue(report.expect_exit_mismatches)
        with self.assertRaises(cr.CoverageGateError):
            cr.assert_full_coverage(report)


if __name__ == "__main__":
    unittest.main()
