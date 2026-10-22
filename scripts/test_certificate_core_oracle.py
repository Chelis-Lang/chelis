#!/usr/bin/env python3
"""Failure controls for frozen certificate parity and exact evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import certificate_core_oracle as oracle
from no_std_core_support import OracleFailure


def proof(certificate: dict) -> dict:
    return {
        "result": "Proved", "soundness": "Exact", "qualifiers": ["CertificateBearing"],
        "evidence": {"engine": "clarabel_sos", "backend": {
            "outcome": "proved", "strict": False, "certificate": deepcopy(certificate),
        }},
    }


def nonproof(*, repair: bool = False) -> dict:
    return {
        "result": "Unknown", "soundness": "Untrusted", "qualifiers": [],
        "evidence": {"engine": "clarabel_sos", "backend": {
            "outcome": "unknown",
            "reason": "certificate does not match the canonical interval layout" if repair else "no certificate",
            "detail": {"cert_matches_goal_layout": False} if repair else {},
        }},
    }


def evidence(*, candidate: bool) -> dict:
    certificates = oracle.expected_certificates()
    errors = {name: nonproof() for name in oracle.ERROR_CASES}
    for name, value in errors.items():
        value["evidence"]["backend"]["detail"] = {"cert_error": name}
    return {
        "arithmetic": oracle.expected_arithmetic(),
        "host": {
            "canonical": {name: proof(certificates[name]) for name in oracle.CANONICAL_CASES},
            "binding": {name: nonproof() for name in oracle.BINDING_CASES},
            "errors": errors,
            "production": {
                **{name: proof(certificates[name]) for name in oracle.PRODUCTION_PROOFS},
                **{name: nonproof() for name in oracle.PRODUCTION_NONPROOFS},
            },
            "layout_repair": {
                name: {
                    "arithmetic": "Ok(())",
                    "admission": nonproof(repair=True) if candidate else proof(certificates["repair_" + name]),
                }
                for name in oracle.REPAIR_CASES
            },
        },
    }


class CertificateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = evidence(candidate=False)
        self.candidate = evidence(candidate=True)

    def compare(self) -> None:
        oracle.compare_snapshots(self.baseline, self.candidate)

    def test_matching_empty_snapshots_cannot_prove_parity(self) -> None:
        with self.assertRaisesRegex(OracleFailure, "incomplete snapshot"):
            oracle.compare_snapshots({}, {})

    def test_matching_empty_arithmetic_cannot_prove_parity(self) -> None:
        for value in (None, {}, {"identities": {}, "matrices": {}, "repairs": {}}):
            with self.subTest(value=value):
                self.baseline["arithmetic"] = deepcopy(value)
                self.candidate["arithmetic"] = deepcopy(value)
                with self.assertRaisesRegex(OracleFailure, "exact arithmetic evidence"):
                    self.compare()

    def test_each_arithmetic_section_is_mandatory_on_both_sides(self) -> None:
        for side in (False, True):
            for name in ("identities", "matrices", "repairs"):
                with self.subTest(candidate=side, section=name):
                    baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                    del (candidate if side else baseline)["arithmetic"][name]
                    with self.assertRaisesRegex(OracleFailure, "exact arithmetic evidence"):
                        oracle.compare_snapshots(baseline, candidate)

    def test_matching_wrong_exact_values_cannot_prove_parity(self) -> None:
        for section, field, replacement in (
            ("identities", "huge_coefficients", ["1/1", "1/1"]),
            ("identities", "evaluation", "9/1"),
            ("matrices", "tiny_pivot", "0/1"),
            ("matrices", "errors", ["Err(NotPsd { block: 0 })"]),
            ("repairs", "pivots", [[1, 1], [0, 0]]),
            ("repairs", "free", []),
            ("repairs", "solution", ["3/1", "1/1", "0/1"]),
            ("repairs", "inconsistent", "Ok([])"),
            ("repairs", "shape_assertions", [1, 1, 1, 1]),
        ):
            with self.subTest(section=section, field=field):
                baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                for record in (baseline, candidate):
                    record["arithmetic"][section][field] = deepcopy(replacement)
                with self.assertRaisesRegex(OracleFailure, "exact arithmetic evidence"):
                    oracle.compare_snapshots(baseline, candidate)

    def test_each_host_section_is_mandatory_on_both_sides(self) -> None:
        for side in (False, True):
            for section in self.baseline["host"]:
                with self.subTest(candidate=side, section=section):
                    baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                    del (candidate if side else baseline)["host"][section]
                    with self.assertRaisesRegex(OracleFailure, "incomplete host"):
                        oracle.compare_snapshots(baseline, candidate)

    def test_each_host_case_is_mandatory_on_both_sides(self) -> None:
        for side in (False, True):
            for section, cases in self.baseline["host"].items():
                for name in cases:
                    with self.subTest(candidate=side, section=section, case=name):
                        baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                        del (candidate if side else baseline)["host"][section][name]
                        with self.assertRaisesRegex(OracleFailure, "incomplete"):
                            oracle.compare_snapshots(baseline, candidate)

    def test_extra_case_cannot_expand_the_admission_exception(self) -> None:
        for record in (self.baseline, self.candidate):
            record["host"]["layout_repair"]["other"] = deepcopy(record["host"]["layout_repair"]["negative"])
        with self.assertRaisesRegex(OracleFailure, "incomplete layout_repair"):
            self.compare()

    def test_canonical_block_order_cannot_change(self) -> None:
        blocks = self.candidate["host"]["canonical"]["even"]["evidence"]["backend"]["certificate"]["blocks"]
        blocks.reverse()
        with self.assertRaisesRegex(OracleFailure, "canonical certificate evidence"):
            self.compare()

    def test_matching_missing_certificate_fields_cannot_prove_parity(self) -> None:
        for field in ("poly", "lo", "hi", "blocks"):
            with self.subTest(field=field):
                baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                for record in (baseline, candidate):
                    del record["host"]["canonical"]["even"]["evidence"]["backend"]["certificate"][field]
                with self.assertRaisesRegex(OracleFailure, "incomplete certificate"):
                    oracle.compare_snapshots(baseline, candidate)

    def test_matching_empty_exact_values_cannot_prove_parity(self) -> None:
        for field, replacement in (("poly", []), ("lo", ""), ("blocks", [])):
            with self.subTest(field=field):
                baseline, candidate = evidence(candidate=False), evidence(candidate=True)
                for record in (baseline, candidate):
                    record["host"]["canonical"]["constant"]["evidence"]["backend"]["certificate"][field] = replacement
                with self.assertRaises(OracleFailure):
                    oracle.compare_snapshots(baseline, candidate)

    def test_invalid_rational_text_cannot_prove_parity(self) -> None:
        for value in ("1", "1.0", "2/2", "1/0", "1/-2", "", None, 1):
            with self.subTest(value=value):
                with self.assertRaises(OracleFailure):
                    oracle.validate_rational(value)

    def test_missing_gram_entries_or_multiplier_cannot_prove_parity(self) -> None:
        for field, replacement in (("gram", [["1/1"], ["0/1"]]), ("gram", []),
                                   ("multiplier", []), ("multiplier", ["1/1", "0/1"])):
            with self.subTest(field=field, value=replacement):
                certificate = oracle.expected_certificates()["even"]
                certificate["blocks"][0][field] = replacement
                with self.assertRaises(OracleFailure):
                    oracle.validate_certificate(certificate)

    def test_public_error_label_change_cannot_hide_in_an_unknown(self) -> None:
        value = self.candidate["host"]["errors"]["not_symmetric"]
        value["evidence"]["backend"]["detail"]["cert_error"] = "not_psd"
        with self.assertRaisesRegex(OracleFailure, "public error label"):
            self.compare()

    def test_binding_rejection_evidence_must_match_the_base(self) -> None:
        self.candidate["host"]["binding"]["strict_same"]["evidence"]["backend"]["reason"] = "new rejection"
        with self.assertRaisesRegex(OracleFailure, "binding values, outcomes, or exact evidence differ"):
            self.compare()

    def test_production_exact_evidence_must_match_the_base(self) -> None:
        certificate = self.candidate["host"]["production"]["even"]["evidence"]["backend"]["certificate"]
        certificate["blocks"][0]["gram"][0][0] = "1000000000000000000000000000000000001/1000000000000000000000000000000000000"
        with self.assertRaisesRegex(OracleFailure, "production values, outcomes, or exact evidence differ"):
            self.compare()

    def test_production_proof_cannot_regress_to_unknown(self) -> None:
        self.candidate["host"]["production"]["constant"] = nonproof()
        with self.assertRaisesRegex(OracleFailure, "host authority"):
            self.compare()

    def test_production_nonproof_cannot_gain_authority(self) -> None:
        self.candidate["host"]["production"]["strict"] = proof(oracle.expected_certificates()["even"])
        with self.assertRaisesRegex(OracleFailure, "host authority"):
            self.compare()

    def test_nonproof_cannot_retain_a_qualifier(self) -> None:
        self.candidate["host"]["binding"]["wrong_goal"]["qualifiers"] = ["CertificateBearing"]
        with self.assertRaisesRegex(OracleFailure, "host authority"):
            self.compare()

    def test_engine_attribution_is_mandatory(self) -> None:
        for value in ({}, {"engine": "other", "backend": {}}):
            with self.subTest(value=value):
                record = nonproof()
                record["evidence"] = value
                with self.assertRaises(OracleFailure):
                    oracle.validate_discharge(record, proved=False)

    def test_baseline_layout_admission_must_demonstrate_the_original_defect(self) -> None:
        self.baseline["host"]["layout_repair"]["negative"]["admission"] = nonproof(repair=True)
        with self.assertRaisesRegex(OracleFailure, "host authority"):
            self.compare()

    def test_original_layout_admission_requires_exact_certificate_evidence(self) -> None:
        value = self.baseline["host"]["layout_repair"]["negative"]["admission"]
        value["evidence"]["backend"]["certificate"]["poly"] = []
        with self.assertRaisesRegex(OracleFailure, "original layout admission evidence"):
            self.compare()

    def test_candidate_layout_admission_must_reject_the_original_defect(self) -> None:
        self.candidate["host"]["layout_repair"]["negative"] = deepcopy(self.baseline["host"]["layout_repair"]["negative"])
        with self.assertRaisesRegex(OracleFailure, "host authority"):
            self.compare()

    def test_layout_repair_cannot_change_arithmetic(self) -> None:
        self.candidate["host"]["layout_repair"]["negative"]["arithmetic"] = "Err(PolynomialMismatch)"
        with self.assertRaisesRegex(OracleFailure, "changed arithmetic"):
            self.compare()

    def test_an_unrelated_rejection_cannot_substitute_for_the_layout_repair(self) -> None:
        self.candidate["host"]["layout_repair"]["negative"]["admission"] = nonproof()
        with self.assertRaisesRegex(OracleFailure, "designed repair evidence"):
            self.compare()

    def test_layout_repair_requires_the_exact_detail(self) -> None:
        backend = self.candidate["host"]["layout_repair"]["negative"]["admission"]["evidence"]["backend"]
        backend["detail"] = {}
        with self.assertRaisesRegex(OracleFailure, "designed repair evidence"):
            self.compare()


class StructuredRecordTests(unittest.TestCase):
    def test_one_record_preserves_exact_values(self) -> None:
        record = {"exact": "123456789012345678901234567890/7"}
        self.assertEqual(oracle.parse_record("other output\nSNAPSHOT " + json.dumps(record), "SNAPSHOT "), record)

    def test_missing_repeated_or_invalid_records_cannot_prove_parity(self) -> None:
        for output in ("", "SNAPSHOT {}\nSNAPSHOT {}", "SNAPSHOT {", "SNAPSHOT []",
                       'SNAPSHOT {"x": 1, "x": 2}', 'SNAPSHOT {"x": {"v": 1, "v": 2}}'):
            with self.subTest(output=output):
                with self.assertRaises(OracleFailure):
                    oracle.parse_record(output, "SNAPSHOT ")


class BaselineHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="certificate-history-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.upstream = self.directory / "upstream"
        self.upstream.mkdir()
        self.git("init", "-q")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Certificate Oracle Test")
        self.git("config", "core.hooksPath", "/dev/null")
        self.git("config", "commit.gpgsign", "false")
        for name in oracle.BASELINE_FILES:
            path = self.upstream / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("baseline\n", encoding="utf-8")
        (self.upstream / "value.txt").write_text("baseline", encoding="utf-8")
        self.git("add", ".")
        self.git("commit", "-qm", "baseline")
        self.baseline = self.git("rev-parse", "HEAD")
        (self.upstream / "value.txt").write_text("candidate", encoding="utf-8")
        self.git("commit", "-qam", "candidate")
        self.shallow = self.directory / "shallow"
        self.git("clone", "--quiet", "--depth", "1", self.upstream.as_uri(), str(self.shallow))

    def git(self, *arguments: str, cwd: Path | None = None) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=cwd or self.upstream,
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    def test_shallow_clone_fetches_the_exact_pinned_commit(self) -> None:
        self.assertEqual(self.git("rev-list", "--count", "HEAD", cwd=self.shallow), "1")
        with patch.multiple(oracle, ROOT=self.shallow, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=self.upstream.as_uri()):
            baseline = oracle.checkout_baseline(self.directory / "baseline")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=baseline), self.baseline)
        self.assertEqual((baseline / "value.txt").read_text(encoding="utf-8"), "baseline")

    def test_cached_commit_needs_no_remote_access(self) -> None:
        with patch.multiple(oracle, ROOT=self.upstream, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=str(self.directory / "absent")):
            baseline = oracle.checkout_baseline(self.directory / "baseline")
        self.assertEqual(self.git("rev-parse", "HEAD", cwd=baseline), self.baseline)
        self.assertEqual((baseline / "value.txt").read_text(encoding="utf-8"), "baseline")

    def test_complete_clone_fails_for_an_absent_pinned_commit(self) -> None:
        with patch.multiple(oracle, ROOT=self.upstream, BASELINE="0" * 40,
                            BASELINE_REPOSITORY=str(self.directory / "absent")):
            with self.assertRaisesRegex(OracleFailure, "absent from a complete clone"):
                oracle.checkout_baseline(self.directory / "baseline")

    def test_failed_fetch_cannot_substitute_candidate_evidence(self) -> None:
        with patch.multiple(oracle, ROOT=self.shallow, BASELINE=self.baseline,
                            BASELINE_REPOSITORY=str(self.directory / "absent")):
            with self.assertRaisesRegex(OracleFailure, "git fetch"):
                oracle.checkout_baseline(self.directory / "baseline")

    def test_missing_original_source_cannot_prove_baseline_parity(self) -> None:
        with patch.multiple(oracle, ROOT=self.upstream, BASELINE=self.baseline,
                            BASELINE_FILES=(*oracle.BASELINE_FILES, "absent-original.rs")):
            with self.assertRaisesRegex(OracleFailure, "missing original baseline evidence"):
                oracle.checkout_baseline(self.directory / "baseline")


MANDATORY_TESTS = (
    "CertificateEvidenceTests.test_matching_empty_snapshots_cannot_prove_parity",
    "CertificateEvidenceTests.test_matching_empty_arithmetic_cannot_prove_parity",
    "CertificateEvidenceTests.test_each_arithmetic_section_is_mandatory_on_both_sides",
    "CertificateEvidenceTests.test_matching_wrong_exact_values_cannot_prove_parity",
    "CertificateEvidenceTests.test_each_host_section_is_mandatory_on_both_sides",
    "CertificateEvidenceTests.test_each_host_case_is_mandatory_on_both_sides",
    "CertificateEvidenceTests.test_extra_case_cannot_expand_the_admission_exception",
    "CertificateEvidenceTests.test_canonical_block_order_cannot_change",
    "CertificateEvidenceTests.test_matching_missing_certificate_fields_cannot_prove_parity",
    "CertificateEvidenceTests.test_matching_empty_exact_values_cannot_prove_parity",
    "CertificateEvidenceTests.test_invalid_rational_text_cannot_prove_parity",
    "CertificateEvidenceTests.test_missing_gram_entries_or_multiplier_cannot_prove_parity",
    "CertificateEvidenceTests.test_public_error_label_change_cannot_hide_in_an_unknown",
    "CertificateEvidenceTests.test_binding_rejection_evidence_must_match_the_base",
    "CertificateEvidenceTests.test_production_exact_evidence_must_match_the_base",
    "CertificateEvidenceTests.test_production_proof_cannot_regress_to_unknown",
    "CertificateEvidenceTests.test_production_nonproof_cannot_gain_authority",
    "CertificateEvidenceTests.test_nonproof_cannot_retain_a_qualifier",
    "CertificateEvidenceTests.test_engine_attribution_is_mandatory",
    "CertificateEvidenceTests.test_baseline_layout_admission_must_demonstrate_the_original_defect",
    "CertificateEvidenceTests.test_original_layout_admission_requires_exact_certificate_evidence",
    "CertificateEvidenceTests.test_candidate_layout_admission_must_reject_the_original_defect",
    "CertificateEvidenceTests.test_layout_repair_cannot_change_arithmetic",
    "CertificateEvidenceTests.test_an_unrelated_rejection_cannot_substitute_for_the_layout_repair",
    "CertificateEvidenceTests.test_layout_repair_requires_the_exact_detail",
    "StructuredRecordTests.test_one_record_preserves_exact_values",
    "StructuredRecordTests.test_missing_repeated_or_invalid_records_cannot_prove_parity",
    "BaselineHistoryTests.test_shallow_clone_fetches_the_exact_pinned_commit",
    "BaselineHistoryTests.test_cached_commit_needs_no_remote_access",
    "BaselineHistoryTests.test_complete_clone_fails_for_an_absent_pinned_commit",
    "BaselineHistoryTests.test_failed_fetch_cannot_substitute_candidate_evidence",
    "BaselineHistoryTests.test_missing_original_source_cannot_prove_baseline_parity",
)


class EvidenceResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed: set[str] = set()

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.add(test.id().removeprefix("__main__."))


def main() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2, resultclass=EvidenceResult).run(suite)
    if not result.wasSuccessful() or result.passed != set(MANDATORY_TESTS) or result.skipped:
        return 1
    print("CERTIFICATE_ORACLE_TESTS " + json.dumps({"passed": sorted(result.passed), "skipped": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
