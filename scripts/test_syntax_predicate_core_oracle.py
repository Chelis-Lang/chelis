"""Failure controls for syntax parity and resource evidence."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from no_std_core_support import OracleFailure
import syntax_predicate_core_oracle as oracle
from syntax_predicate_core_oracle import CASES, OPERATIONS, compare_snapshots


def evidence() -> dict:
    semantic = {
        name: {
            "canonical": name,
            "deep": name,
            "json": f'["{name}"]',
            "binary": [1, 2, 3],
            "recovered": name,
        }
        for name in CASES
    }
    semantic.update({
        "predicate": {"free_vars": ["beta", "alpha"], "rejection": "outside grammar"},
        "errors": ["syntax", "suffix", "bare name", "declaration"],
        "span": {"offset": 2, "len": 5},
        "path_error": "mismatch",
    })
    return {
        "semantic": semantic,
        "costs": {
            case: {
                operation: {"iterations": 8, "allocations": 8,
                            "allocated_bytes": 800, "nanoseconds": 1000}
                for operation in OPERATIONS
            }
            for case in CASES
        },
    }


class SyntaxEvidenceTests(unittest.TestCase):
    def test_wire_mismatch_rejects_other_matching_evidence(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["semantic"]["identity"]["json"] = '["different"]'
        with self.assertRaisesRegex(OracleFailure, "wire parity differs"):
            compare_snapshots(baseline, candidate)

    def test_two_empty_semantic_records_do_not_prove_parity(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        baseline["semantic"] = {}
        candidate["semantic"] = {}
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax semantic"):
            compare_snapshots(baseline, candidate)

    def test_missing_case_rejects_other_measurements(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        del candidate["costs"]["tensor"]
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax cost cases"):
            compare_snapshots(baseline, candidate)

    def test_missing_operation_rejects_other_measurements(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        del candidate["costs"]["macro"]["expand"]
        with self.assertRaisesRegex(OracleFailure, "incomplete syntax cost operations"):
            compare_snapshots(baseline, candidate)

    def test_allocation_growth_cannot_hide_in_another_operation(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["macro"]["expand"]["allocations"] += 1
        candidate["costs"]["macro"]["parse"]["allocations"] -= 1
        with self.assertRaisesRegex(OracleFailure, "allocations regression: macro/expand"):
            compare_snapshots(baseline, candidate)

    def test_byte_growth_rejects_equal_allocation_counts(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["tensor"]["resugar"]["allocated_bytes"] += 1
        with self.assertRaisesRegex(OracleFailure, "allocated_bytes regression: tensor/resugar"):
            compare_snapshots(baseline, candidate)

    def test_reduced_iteration_count_does_not_prove_lower_cost(self) -> None:
        baseline = evidence()
        candidate = deepcopy(baseline)
        candidate["costs"]["identity"]["parse"]["iterations"] = 4
        with self.assertRaisesRegex(OracleFailure, "unequal syntax cost work"):
            compare_snapshots(baseline, candidate)

    def test_invalid_measurements_cannot_prove_acceptance(self) -> None:
        for value in (-1, "8", True, None):
            with self.subTest(value=value):
                baseline = evidence()
                candidate = deepcopy(baseline)
                candidate["costs"]["integer-float"]["format"]["allocations"] = value
                with self.assertRaisesRegex(OracleFailure, "invalid syntax cost measurement"):
                    compare_snapshots(baseline, candidate)

    def test_symmetric_source_field_loss_rejects_matching_snapshots(self) -> None:
        for case in CASES:
            for field in ("canonical", "deep", "json", "binary", "recovered"):
                with self.subTest(case=case, field=field):
                    baseline = evidence()
                    del baseline["semantic"][case][field]
                    with self.assertRaisesRegex(OracleFailure, "source fields"):
                        compare_snapshots(baseline, deepcopy(baseline))

    def test_empty_source_values_do_not_prove_parity(self) -> None:
        for field, value in (("canonical", ""), ("json", None), ("binary", []),
                             ("binary", [256]), ("binary", [True])):
            with self.subTest(field=field, value=value):
                baseline = evidence()
                baseline["semantic"]["identity"][field] = value
                with self.assertRaises(OracleFailure):
                    compare_snapshots(baseline, deepcopy(baseline))

    def test_symmetric_predicate_field_loss_rejects_matching_snapshots(self) -> None:
        for field in ("free_vars", "rejection"):
            with self.subTest(field=field):
                baseline = evidence()
                del baseline["semantic"]["predicate"][field]
                with self.assertRaisesRegex(OracleFailure, "predicate"):
                    compare_snapshots(baseline, deepcopy(baseline))

    def test_missing_diagnostic_rejects_matching_snapshots(self) -> None:
        baseline = evidence()
        baseline["semantic"]["errors"].pop()
        with self.assertRaisesRegex(OracleFailure, "diagnostic"):
            compare_snapshots(baseline, deepcopy(baseline))

    def test_symmetric_span_field_loss_rejects_matching_snapshots(self) -> None:
        for field in ("offset", "len"):
            with self.subTest(field=field):
                baseline = evidence()
                del baseline["semantic"]["span"][field]
                with self.assertRaisesRegex(OracleFailure, "span"):
                    compare_snapshots(baseline, deepcopy(baseline))

    def test_empty_nested_values_cannot_prove_parity(self) -> None:
        for key, value in (("predicate", {}), ("errors", []), ("span", {}),
                           ("path_error", "")):
            with self.subTest(key=key):
                baseline = evidence()
                baseline["semantic"][key] = value
                with self.assertRaises(OracleFailure):
                    compare_snapshots(baseline, deepcopy(baseline))


class BaselineHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="syntax-history-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.upstream = self.directory / "upstream"
        self.upstream.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Syntax Oracle Test")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("config", "commit.gpgsign", "false")
        (self.upstream / "value.txt").write_text("baseline", encoding="utf-8")
        self.git("add", "value.txt")
        self.git("commit", "-qm", "baseline")
        self.baseline = self.git("rev-parse", "HEAD")
        (self.upstream / "value.txt").write_text("candidate", encoding="utf-8")
        self.git("commit", "-qam", "candidate")
        self.shallow = self.directory / "shallow"
        self.git("clone", "--quiet", "--depth", "1", self.upstream.as_uri(),
                 str(self.shallow))

    def git(self, *arguments: str, cwd: Path | None = None) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=cwd or self.upstream,
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def test_shallow_checkout_fetches_the_exact_baseline(self) -> None:
        self.assertEqual(self.git("rev-list", "--count", "HEAD", cwd=self.shallow), "1")
        with patch.multiple(oracle, ROOT=self.shallow, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=self.upstream.as_uri()):
            baseline = oracle._checkout_baseline(self.directory / "baseline")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=baseline), self.baseline)
        self.assertEqual((baseline / "value.txt").read_text(encoding="utf-8"), "baseline")

    def test_cached_baseline_needs_no_remote_access(self) -> None:
        with patch.multiple(oracle, ROOT=self.upstream, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=str(self.directory / "absent")):
            baseline = oracle._checkout_baseline(self.directory / "baseline")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=baseline), self.baseline)
        self.assertEqual((baseline / "value.txt").read_text(encoding="utf-8"), "baseline")

    def test_missing_baseline_cannot_use_the_candidate_as_a_fallback(self) -> None:
        with patch.multiple(oracle, ROOT=self.shallow, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=str(self.directory / "absent")):
            with self.assertRaisesRegex(OracleFailure, "git fetch"):
                oracle._checkout_baseline(self.directory / "baseline")


if __name__ == "__main__":
    unittest.main()
