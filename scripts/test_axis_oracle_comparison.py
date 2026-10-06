"""Lock source preservation and honest oracle scoring before the campaign."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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
    def test_command_receipt_matches_completed_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = comparison.execute([sys.executable, "-c", "print('oracle output')"], root,
                                         os.environ.copy(), root / "oracle.log", 5)
            self.assertEqual(receipt["exit_code"], 0)
            self.assertFalse(receipt["timeout"])
            comparison.validate_receipts(receipt)

    def test_interrupted_command_is_reaped_before_restoring_the_kernel(self):
        children = []
        spawn = subprocess.Popen
        def tracked(*args, **kwargs):
            child = spawn(*args, **kwargs)
            children.append(child)
            return child
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(comparison.subprocess, "Popen", side_effect=tracked), \
                 patch.object(comparison, "group_rss", side_effect=InterruptedError("stop campaign")):
                with self.assertRaises(InterruptedError):
                    comparison.execute([sys.executable, "-c", "import time; time.sleep(60)"], root,
                                       os.environ.copy(), root / "oracle.log", 5)
            try:
                self.assertIsNotNone(children[0].poll())
            finally:
                if children[0].poll() is None:
                    children[0].kill()
                    children[0].wait()

    def test_environment_preserves_explicit_python_even_when_invalid(self):
        for python in (sys.executable, "/missing/explicit-python"):
            original = {"PYO3_PYTHON": python, "RUSTFLAGS": "unrelated flags"}
            environment = comparison.execution_environment(original)
            self.assertEqual(environment["PYO3_PYTHON"], python)
            self.assertNotIn("RUSTFLAGS", environment)
            self.assertIn("RUSTFLAGS", original)

    def test_environment_defaults_to_the_managed_running_interpreter(self):
        self.assertEqual(comparison.execution_environment({})["PYO3_PYTHON"], sys.executable)

    def test_vermilion_status_requires_this_case_and_distinguishes_refusals(self):
        source = "examples/current/verified.rs"
        receipt = {"timeout": False, "exit_code": 0}
        structural = {"phase": "lean", "exit": 0, "source": source}
        self.assertEqual(comparison.vermilion_status(receipt, structural, source, False), "pass")
        failed = receipt | {"exit_code": 1}
        self.assertEqual(comparison.vermilion_status(failed, structural, source, False), "proof_failure")
        self.assertEqual(comparison.vermilion_status(failed, structural, source, True), "tool_error")
        self.assertEqual(comparison.vermilion_status(failed, structural | {"source": "previous.rs"}, source, False), "tool_error")
        self.assertEqual(comparison.vermilion_status(failed, structural | {"phase": "lowering"}, source, False), "tool_error")
        self.assertEqual(comparison.vermilion_status(receipt | {"timeout": True}, structural, source, False), "timeout")

    def test_resumed_vermilion_twin_keeps_exactly_one_helper(self):
        source = "namespace verified.is_permutation\nend verified.is_permutation\n"
        helper = "-- vrml:user:begin\nprivate theorem helper : True := by trivial\n-- vrml:user:end\n"
        once = comparison.insert_tactic_helper(source, helper)
        self.assertEqual(comparison.insert_tactic_helper(once, helper), once)
        self.assertEqual(once.count(helper), 1)
        with self.assertRaisesRegex(ValueError, "namespace"):
            comparison.insert_tactic_helper("namespace wrong\n", helper)
        with self.assertRaisesRegex(ValueError, "helper"):
            comparison.insert_tactic_helper(once.replace("by trivial", "by sorry"), helper)
        with self.assertRaisesRegex(ValueError, "helper"):
            comparison.insert_tactic_helper(helper + once, helper)

    def test_cost_and_mutant_settings_cannot_mix_different_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "bound.json"
            path.write_text(json.dumps({"bound": 4, "unwind_by_harness": {"admission": 7}}))
            args = argparse.Namespace(output=root)
            self.assertEqual(comparison.frozen_settings(args)["bound"], 4)
            self.assertEqual(comparison.frozen_settings(args)["bound"], 4)
            path.write_text(json.dumps({"bound": 5, "unwind_by_harness": {"admission": 8}}))
            with self.assertRaises(ValueError):
                comparison.frozen_settings(args)

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
        self.assertEqual(comparison.classify("verus", 1, "error: precondition not met: index in bounds for this access", False), "proof_failure")

    def test_unwind_failure_is_inconclusive(self):
        self.assertEqual(comparison.classify("kani", 1, "unwinding assertion: FAILURE", False), "unwind_failure")
        self.assertEqual(comparison.classify("kani", 1, "assertion failed VERIFICATION:- FAILED", False), "assertion_failure")

    def test_reached_unsupported_construct_is_not_contract_detection(self):
        self.assertEqual(comparison.classify("kani", 1, "Failed Checks: reached unsupported construct\nVERIFICATION:- FAILED", False), "tool_error")
        self.assertEqual(comparison.classify("kani", 1, "warning: unsupported constructs\nFailed Checks: assertion failed\nVERIFICATION:- FAILED", False), "assertion_failure")

    def test_timeout_never_becomes_a_catch(self):
        self.assertEqual(comparison.classify("tests", 1, "test result: FAILED", True), "timeout")

    def test_multiline_unwind_and_solver_timeout_are_inconclusive(self):
        self.assertEqual(comparison.classify("kani", 1, "Description: unwinding assertion loop 0\nStatus: FAILURE\nVERIFICATION:- FAILED", False), "unwind_failure")
        self.assertEqual(comparison.classify("kani", 1, "CBMC timed out", False), "timeout")

    def test_actual_kani_check_blocks_put_status_before_description(self):
        block = ('Check 182: verified::is_permutation.unwind.0\n'
                 '\t - Status: FAILURE\n\t - Description: "unwinding assertion loop 0"\n'
                 'VERIFICATION:- FAILED\n')
        self.assertEqual(comparison.classify("kani", 1, block, False), "unwind_failure")
        self.assertEqual(comparison.classify("kani", 1, block.replace("FAILURE", "SUCCESS"), False), "assertion_failure")
        unsupported = block.replace("verified::is_permutation.unwind.0", "panic.unsupported_construct.1").replace(
            "unwinding assertion loop 0", "call to foreign Rust function is not currently supported by Kani")
        self.assertEqual(comparison.classify("kani", 1, unsupported, False), "tool_error")
        self.assertEqual(comparison.classify("kani", 1, unsupported.replace("FAILURE", "SUCCESS"), False), "assertion_failure")

    def test_mutant_loop_selects_disagreement_and_unwitnessed_proof_failure(self):
        base = {"verus_credit": "caught", "kani": {"admission": {"status": "assertion_failure"}}}
        self.assertFalse(comparison.selected_for_vermilion(base))
        self.assertTrue(comparison.selected_for_vermilion(base | {"kani": {"admission": {"status": "pass"}}}))
        self.assertTrue(comparison.selected_for_vermilion(base | {"verus_credit": "unwitnessed_proof_failure"}))
        self.assertTrue(comparison.selected_for_vermilion(base | {"verus_credit": "pass", "witness": {"rank": 1}}))


if __name__ == "__main__":
    unittest.main()
