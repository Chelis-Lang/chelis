"""Lock source preservation and honest oracle scoring before the campaign."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axis_oracle_comparison as comparison

SOURCE = Path(__file__).resolve().parents[1] / "crates/chelis-axis-core/src/verified.rs"


class MutationTests(unittest.TestCase):
    def test_inventory_is_deterministic_and_only_changes_exec_tokens(self):
        source = SOURCE.read_text()
        mutants = comparison.mutants(source)
        self.assertGreater(len(mutants), 0)
        self.assertEqual(mutants, comparison.mutants(source))
        self.assertEqual(len({m["id"] for m in mutants}), len(mutants))
        allowed = comparison.exec_ranges(source)
        for mutant in mutants:
            self.assertTrue(any(a <= mutant["start"] < mutant["end"] <= b for a, b in allowed.values()))
            changed = comparison.apply_mutant(source, mutant)
            self.assertEqual(changed[:mutant["start"]], source[:mutant["start"]])
            self.assertEqual(changed[mutant["start"] + len(mutant["replacement"]):], source[mutant["end"]:])

    def test_invariants_comments_and_strings_are_preserved(self):
        source = SOURCE.read_text()
        source = source.replace('let mut i = 0;', '// false != true\n        let label = "false < true";\n        let mut i = 0;', 1)
        for mutant in comparison.mutants(source):
            original = source[mutant["start"]:mutant["end"]]
            masked = comparison.executable_mask(source)
            self.assertEqual(masked[mutant["start"]:mutant["end"]], original)

    def test_changed_baseline_cannot_receive_stale_patch(self):
        source = SOURCE.read_text()
        mutant = comparison.mutants(source)[0]
        with self.assertRaises(ValueError):
            comparison.apply_mutant(source + "\n", mutant)

    def test_call_graph_selects_only_reachable_harnesses(self):
        graph = comparison.harness_mapping(SOURCE.read_text())
        self.assertEqual(graph["is_permutation"], ["admission", "inverse"])
        self.assertEqual(graph["checked_inverse"], ["inverse"])
        self.assertEqual(graph["normalize_axis"], ["normalization"])
        self.assertEqual(graph["reduction_survivors"], ["survivors"])


class VerdictTests(unittest.TestCase):
    def test_cached_logs_reject_missing_or_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oracle.log"
            path.write_text("passed")
            receipt = {"log": str(path), "log_sha256": comparison.digest("passed")}
            comparison.validate_receipts([receipt])
            path.write_text("changed")
            with self.assertRaises(ValueError):
                comparison.validate_receipts([receipt])
            path.unlink()
            with self.assertRaises(ValueError):
                comparison.validate_receipts([receipt])

    def test_proof_failure_requires_witness_for_credit(self):
        self.assertEqual(comparison.verus_credit("proof_failure", None), "unwitnessed_proof_failure")
        self.assertEqual(comparison.verus_credit("proof_failure", {"rank": 1}), "caught")
        self.assertEqual(comparison.verus_credit("timeout", {"rank": 1}), "timeout")

    def test_parser_failure_is_not_a_proof_failure(self):
        self.assertEqual(comparison.classify("verus", 1, "error: expected token", False), "tool_error")
        self.assertEqual(comparison.classify("verus", 1, "postcondition not satisfied", False), "proof_failure")

    def test_unwind_failure_is_inconclusive(self):
        self.assertEqual(comparison.classify("kani", 1, "unwinding assertion: FAILURE", False), "unwind_failure")
        self.assertEqual(comparison.classify("kani", 1, "assertion failed VERIFICATION:- FAILED", False), "assertion_failure")

    def test_timeout_never_becomes_a_catch(self):
        self.assertEqual(comparison.classify("tests", 1, "test result: FAILED", True), "timeout")

    def test_multiline_unwind_and_solver_timeout_are_inconclusive(self):
        self.assertEqual(comparison.classify("kani", 1, "Description: unwinding assertion loop 0\nStatus: FAILURE\nVERIFICATION:- FAILED", False), "unwind_failure")
        self.assertEqual(comparison.classify("kani", 1, "CBMC timed out", False), "timeout")

    def test_mutant_loop_selects_disagreement_and_unwitnessed_proof_failure(self):
        base = {"verus_credit": "caught", "kani": {"admission": {"status": "assertion_failure"}}}
        self.assertFalse(comparison.selected_for_vermilion(base))
        self.assertTrue(comparison.selected_for_vermilion(base | {"kani": {"admission": {"status": "pass"}}}))
        self.assertTrue(comparison.selected_for_vermilion(base | {"verus_credit": "unwitnessed_proof_failure"}))
        self.assertTrue(comparison.selected_for_vermilion(base | {"verus_credit": "pass", "witness": {"rank": 1}}))


if __name__ == "__main__":
    unittest.main()
