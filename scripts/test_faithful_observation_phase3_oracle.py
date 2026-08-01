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
