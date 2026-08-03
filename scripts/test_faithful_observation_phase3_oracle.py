#!/usr/bin/env python3

import unittest

import faithful_observation_phase3_oracle as oracle


class IgnoreInventoryTests(unittest.TestCase):
    def test_shipped_phase3_suites_have_only_the_declared_environment_skip(self) -> None:
        self.assertEqual(oracle.source_violations(), [])

    def test_new_value_ignore_is_rejected_even_when_it_cites_an_issue(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] += '''
#[test]
#[ignore = "chelis#897 value mismatch"]
fn hidden_value_row() {}
'''
        violations = oracle.source_violations(sources)
        self.assertTrue(any("hidden_value_row" in item for item in violations), violations)

    def test_environment_skip_must_keep_its_exact_reason(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = sources[oracle.PARITY_SOURCE].replace(
            oracle.ALLOWED_IGNORES["parity_transformer_block_library_only"],
            "some other prerequisite",
        )
        violations = oracle.source_violations(sources)
        self.assertTrue(
            any("parity_transformer_block_library_only" in item for item in violations),
            violations,
        )

    def test_a_conditionally_ignored_test_fails_closed(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace(
            "#[test]\nfn agreement_exp()",
            "#[test]\n#[cfg_attr(any(), ignore)]\nfn agreement_exp()",
        )
        violations = oracle.source_violations(sources)
        self.assertTrue(any("agreement_exp" in item for item in violations), violations)

    def test_deleting_a_required_corpus_row_is_a_violation(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace("fn agreement_sqrt_is_exact()", "fn deleted_sqrt_row()")
        violations = oracle.source_violations(sources)
        self.assertTrue(any("agreement_sqrt_is_exact" in item for item in violations), violations)

    def test_deleting_the_expected_value_canary_is_a_violation(self) -> None:
        """chelis#1104: the usage guard for the verbatim leg cannot quietly leave."""
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace(
            "fn agreement_expected_value_reaches_comparator()",
            "fn deleted_expected_value_canary()",
        )
        violations = oracle.source_violations(sources)
        self.assertTrue(
            any("agreement_expected_value_reaches_comparator" in item for item in violations),
            violations,
        )

    def test_empty_required_body_is_caught_by_missing_runtime_receipt(self) -> None:
        receipts = "\n".join(
            f"{case}\tbehavior executed"
            for case in sorted(oracle.REQUIRED_EVAL_RECEIPTS - {"sqrt(4)"})
        )
        violations = oracle.receipt_violations(receipts)
        self.assertEqual(violations, ["missing Phase 3 runtime receipt: sqrt(4)"])

    def test_duplicate_or_fabricated_runtime_receipts_fail_closed(self) -> None:
        receipts = [
            f"{case}\tbehavior executed" for case in sorted(oracle.REQUIRED_EVAL_RECEIPTS)
        ]
        receipts.extend(
            [
                "compiled-observation-canary\tduplicate",
                "fabricated-case\tundeclared",
            ]
        )
        violations = oracle.receipt_violations("\n".join(receipts))
        self.assertIn(
            "Phase 3 runtime receipt must occur exactly once: compiled-observation-canary",
            violations,
        )
        self.assertIn("undeclared Phase 3 runtime receipt: fabricated-case", violations)

    def test_forged_eval_receipts_cannot_replace_required_behavior(self) -> None:
        sources = oracle.shipped_sources()
        source = sources[oracle.EVAL_AGREEMENT_SOURCE]
        for name, case in (
            ("agreement_operation_identity_is_derived_from_ir", "operation-identity-canary"),
            ("agreement_compiled_observation_reaches_comparator", "compiled-observation-canary"),
            ("agreement_expected_value_reaches_comparator", "expected-value-canary"),
            ("agreement_width_nonconformance_is_behavioral", "width-nonconformance-canary"),
            ("agreement_sqrt_is_exact", "sqrt(4)"),
        ):
            source = oracle.replace_test_body(
                source,
                name,
                f'{{ record_phase3_receipt("{case}", "forged"); }}',
            )
        sources[oracle.EVAL_AGREEMENT_SOURCE] = source

        violations = oracle.definition_digest_violations(sources)

        for name in (
            "agreement_operation_identity_is_derived_from_ir",
            "agreement_compiled_observation_reaches_comparator",
            "agreement_expected_value_reaches_comparator",
            "agreement_width_nonconformance_is_behavioral",
            "agreement_sqrt_is_exact",
        ):
            self.assertTrue(any(name in item for item in violations), violations)

    def test_empty_parity_and_rejected_drivers_fail_the_definition_ratchet(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
            sources[oracle.PARITY_SOURCE],
            "parity_tensor_structural_ops",
            "{}",
        )
        sources[oracle.REJECTED_SOURCE] = oracle.replace_test_body(
            sources[oracle.REJECTED_SOURCE],
            "rejected_cells_fail_the_build_with_their_pinned_diagnostics",
            "{}",
        )

        violations = oracle.definition_digest_violations(sources)

        self.assertTrue(any("parity_tensor_structural_ops" in item for item in violations))
        self.assertTrue(
            any(
                "rejected_cells_fail_the_build_with_their_pinned_diagnostics" in item
                for item in violations
            )
        )


class ComparatorAdoptionTests(unittest.TestCase):
    def test_removed_f64_oracle_spellings_stay_absent(self) -> None:
        sources = oracle.shipped_sources()
        self.assertEqual(oracle.comparator_violations(sources), [])

    def test_a_returning_blanket_tolerance_is_rejected(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] += "\nfn assert_close(a: f64, b: f64, tol: f64) {}\n"
        violations = oracle.comparator_violations(sources)
        self.assertTrue(any("assert_close" in item for item in violations), violations)

    def test_dropping_width_nonconformance_is_rejected(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace("ArithmeticWidthStatus::Nonconforming { issue: 897 }", "WIDTH_OK")
        violations = oracle.comparator_violations(sources)
        self.assertTrue(any("chelis#897" in item for item in violations), violations)


if __name__ == "__main__":
    unittest.main()
