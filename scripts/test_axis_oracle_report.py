"""The report must preserve uncertainty and require explicit survivor review."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axis_oracle_report as report


class ScoringTests(unittest.TestCase):
    def row(self):
        return {"id": "a", "classification": "real_gap", "witness": {"rank": 1},
                "verus_credit": "caught", "verus": {"status": "proof_failure", "wall_seconds": 1},
                "tests": [{"suite": "axis", "status": "pass", "wall_seconds": 1}],
                "kani": {"admission": {"status": "timeout", "wall_seconds": 300}}}

    def test_timeout_is_not_a_kani_miss_or_catch(self):
        row = report.summarize(self.row(), {})
        self.assertEqual(row["kani"], "inconclusive")
        self.assertFalse(row["verus_only_confirmed"])

    def test_unbounded_only_requires_conclusive_other_oracles_and_witness(self):
        mutant = self.row()
        mutant["kani"]["admission"]["status"] = "pass"
        self.assertTrue(report.summarize(mutant, {})["verus_only_confirmed"])
        mutant["witness"] = None
        mutant["classification"] = "needs_manual_classification"
        with self.assertRaises(ValueError):
            report.summarize(mutant, {})

    def test_equivalence_requires_reason_and_cannot_override_witness(self):
        mutant = self.row()
        with self.assertRaises(ValueError):
            report.summarize(mutant, {"a": {"classification": "equivalent", "reason": "overwritten fill"}})
        mutant["witness"] = None
        mutant["classification"] = "needs_manual_classification"
        mutant["verus_credit"] = "unwitnessed_proof_failure"
        row = report.summarize(mutant, {"a": {"classification": "equivalent", "reason": "overwritten fill"}})
        self.assertEqual(row["verus"], "unwitnessed_proof_failure")

    def test_integration_failure_is_distinct_from_contract_detection(self):
        mutant = self.row()
        mutant["tests"][0]["status"] = "test_failure"
        self.assertEqual(report.summarize(mutant, {})["tests"], "caught")


if __name__ == "__main__":
    unittest.main()
