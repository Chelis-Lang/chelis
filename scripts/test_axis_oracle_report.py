"""The report must preserve uncertainty and require explicit survivor review."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axis_oracle_report as report


class ScoringTests(unittest.TestCase):
    def test_recorded_kani_unwind_failure_is_not_detection_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "kani.log"
            path.write_text('Check 1: verified::is_permutation.unwind.0\n'
                            '\t - Status: FAILURE\n\t - Description: "unwinding assertion loop 0"\n'
                            'VERIFICATION:- FAILED\n')
            mutant = self.row()
            mutant["kani"]["admission"].update(status="assertion_failure", exit_code=1, timeout=False, log=str(path))
            analyzed = report.adjudicate_kani(mutant)
            self.assertEqual(analyzed["kani"]["admission"]["status"], "unwind_failure")
            self.assertEqual(analyzed["kani_status_recorded"]["admission"], "assertion_failure")
            self.assertEqual(report.summarize(analyzed, {})["kani"], "inconclusive")
            self.assertEqual(mutant["kani"]["admission"]["status"], "assertion_failure")
            mutant["kani"]["admission"]["status"] = "pass"
            with self.assertRaisesRegex(ValueError, "unexplained Kani"):
                report.adjudicate_kani(mutant)

    def test_portable_logs_resolve_by_content_without_changing_original_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "logs"
            store.mkdir()
            content = b"error: postcondition not satisfied\n"
            sha = hashlib.sha256(content).hexdigest()
            (store / f"{sha}.log").write_bytes(content)
            original = {"trials": [{"log": "/unavailable/original.log", "log_sha256": sha}]}
            path = root / "receipts.json"
            path.write_text(json.dumps(original))
            loaded = report.load_receipts(path, store)
            report.validate_receipts(loaded)
            self.assertEqual(loaded["trials"][0]["log"], str(store / f"{sha}.log"))
            self.assertEqual(json.loads(path.read_text()), original)

    def test_portable_logs_reject_corruption_and_non_digest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sha = hashlib.sha256(b"original").hexdigest()
            (root / f"{sha}.log").write_text("changed")
            path = root / "receipts.json"
            path.write_text(json.dumps({"log": "/original", "log_sha256": sha}))
            with self.assertRaisesRegex(ValueError, "changed or missing"):
                report.validate_receipts(report.load_receipts(path, root))
            path.write_text(json.dumps({"log": "/original", "log_sha256": "../../outside"}))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                report.load_receipts(path, root)

    def test_recorded_precondition_rejection_is_transparently_reclassified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verus.log"
            path.write_text("error: precondition not met: index in bounds for this access\n")
            mutant = self.row()
            mutant["verus"] = {"status": "tool_error", "wall_seconds": 1, "exit_code": 1,
                               "timeout": False, "log": str(path)}
            mutant["verus_credit"] = "tool_error"
            analyzed = report.adjudicate_verus(mutant)
            self.assertEqual(analyzed["verus_status_recorded"], "tool_error")
            self.assertEqual(analyzed["verus"]["status"], "proof_failure")
            self.assertEqual(analyzed["verus_credit"], "caught")
            self.assertEqual(mutant["verus"]["status"], "tool_error")
            mutant["witness"] = None
            self.assertEqual(report.adjudicate_verus(mutant)["verus_credit"], "unwitnessed_proof_failure")

    def test_adjudication_does_not_promote_a_parser_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "verus.log"
            path.write_text("error: expected token\n")
            mutant = self.row()
            mutant["verus"] = {"status": "tool_error", "wall_seconds": 1, "exit_code": 1,
                               "timeout": False, "log": str(path)}
            mutant["verus_credit"] = "tool_error"
            self.assertEqual(report.adjudicate_verus(mutant)["verus_credit"], "tool_error")

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

    def test_persisted_credit_cannot_bypass_the_witness_rule(self):
        mutant = self.row()
        mutant["witness"] = None
        with self.assertRaises(ValueError):
            report.summarize(mutant, {})

    def export_fixture(self, root: Path) -> dict:
        log = root / "oracle.log"
        log.write_text("Verification Time: 0.25s\n")
        receipt = {"status": "pass", "wall_seconds": 1, "peak_group_rss_bytes": 100, "timeout": False, "exit_code": 0,
                   "log": str(log), "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()}
        identity = {"id": "a", "function": "checked_inverse", "line": 151,
                    "original": "0", "replacement": "1"}
        result = dict(identity, classification="needs_manual_classification", witness=None,
                      verus_credit="pass", verus=receipt, build=receipt,
                      tests=[dict(receipt, suite=suite) for suite in ("axis", "types", "ir")],
                      kani={"inverse": dict(receipt, bound=5, unwind=8)},
                      unaffected_harnesses=["admission", "normalization", "survivors"])
        data = {"results.json": [result], "mutants.json": [identity],
                "manual.json": {"a": {"classification": "equivalent", "reason": "fill overwritten"}},
                "calibration.json": {"inverse": {"largest_completed_bound": 5, "ceiling_reached": False,
                                                  "trials": [dict(receipt, bound=5, unwind=8)]}},
                "bound.json": {"bound": 5, "runtime_abi_rank_ceiling": 2147483647,
                               "unwind_by_harness": {"admission": 8, "normalization": 32, "survivors": 8, "inverse": 8}},
                "manifest.json": {"harness_mapping": {"checked_inverse": ["inverse"]}},
                "vermilion.json": {"baseline": {"checked": dict(receipt, timeout=False, exit_code=0),
                                               "status": "pass", "credit": "pass", "obligations": 75,
                                               "refused_functions": [],
                                               "structural_verdict": {"phase": "lean", "exit": 0,
                                                                      "source": "examples/chelis-campaign-baseline/verified.rs"}}},
                "costs.json": {f"{oracle}-{phase}": [receipt] for oracle in ("tests", "kani", "verus")
                               for phase in ("cold", "warm")}}
        for filename, value in data.items():
            (root / filename).write_text(json.dumps(value))
        return result

    def export(self, root):
        with patch.object(sys, "argv", ["report", str(root), "--manual", str(root / "manual.json"),
                                        "--output", str(root / "tables")]):
            report.main()

    def test_complete_export_preserves_equivalence_and_real_rank_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.export_fixture(root)
            self.export(root)
            summary = json.loads((root / "tables/summary.json").read_text())
            self.assertEqual(summary["equivalent"], 1)
            self.assertEqual(summary["real_gaps"], 0)
            self.assertEqual(summary["vermilion"]["selected_mutants"], 0)
            self.assertIn("2147483647,False", (root / "tables/costs.csv").read_text())
            self.assertIn("kani_bound_covers_abi_ceiling", (root / "tables/costs.csv").read_text())
            self.assertIn("vermilion_seconds", (root / "tables/mutants.csv").read_text())
            self.assertIn("0.25", (root / "tables/bounds.csv").read_text())

    def test_export_rejects_location_changed_after_frozen_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.export_fixture(root)
            result["line"] += 1
            (root / "results.json").write_text(json.dumps([result]))
            with self.assertRaisesRegex(ValueError, "location or operator"):
                self.export(root)

    def test_export_rejects_unrun_or_differently_bounded_oracles(self):
        for omission in ("suite", "harness", "bound", "unwind"):
            with self.subTest(omission=omission), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result = self.export_fixture(root)
                if omission == "suite":
                    result["tests"].pop()
                elif omission == "harness":
                    result["kani"].clear()
                else:
                    result["kani"]["inverse"][omission] += 1
                (root / "results.json").write_text(json.dumps([result]))
                with self.assertRaises(ValueError):
                    self.export(root)

    def test_export_requires_every_selected_vermilion_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.export_fixture(root)
            proof: dict = dict(result["verus"], status="proof_failure")
            log = root / "proof-failure.log"
            log.write_text("error: postcondition not satisfied\n")
            proof.update(exit_code=1, log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest())
            result["verus"] = proof
            result["verus_credit"] = "unwitnessed_proof_failure"
            (root / "results.json").write_text(json.dumps([result]))
            with self.assertRaisesRegex(ValueError, "selected mutant checks are incomplete"):
                self.export(root)

    def test_export_rejects_a_stale_structural_vermilion_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.export_fixture(root)
            path = root / "vermilion.json"
            value = json.loads(path.read_text())
            value["baseline"]["structural_verdict"]["source"] = "previous/verified.rs"
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "current case evidence"):
                self.export(root)

    def test_vermilion_requires_explicit_lowering_refusal_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.export_fixture(root)
            path = root / "vermilion.json"
            rows = json.loads(path.read_text())
            del rows["baseline"]["refused_functions"]
            with self.assertRaisesRegex(ValueError, "lowering evidence"):
                report.audit_vermilion_cases(rows, None)
            case = root / "cases/chelis-campaign-baseline"
            (case / "generated").mkdir(parents=True)
            source = b"the frozen kernel"
            (case / "verified.rs").write_bytes(source)
            generated = {"rust_file": "examples/chelis-campaign-baseline/verified.rs", "mode": "per-file",
                         "functions": [{"function": f"verified.{name}"} for name in
                                       ("is_permutation", "normalize_axis", "reduction_survivors", "checked_inverse")]}
            (case / "generated/verified.json").write_text(json.dumps(generated))
            rows["baseline"]["source_sha256"] = hashlib.sha256(source).hexdigest()
            report.audit_vermilion_cases(rows, root / "cases")
            self.assertEqual(rows["baseline"]["refused_functions"], [])
            generated["functions"].pop()
            (case / "generated/verified.json").write_text(json.dumps(generated))
            with self.assertRaisesRegex(ValueError, "complete kernel"):
                report.audit_vermilion_cases(rows, root / "cases")
            (case / "verified.rs").write_text("wrong mutant")
            with self.assertRaisesRegex(ValueError, "source differs"):
                report.audit_vermilion_cases(rows, root / "cases")


if __name__ == "__main__":
    unittest.main()
