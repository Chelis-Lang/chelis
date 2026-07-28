#!/usr/bin/env python3

import unittest

import faithful_observation_phase2_oracle as oracle


HARNESS_FIXTURE = """
#[test]
#[ignore = "chelis#684 ([#729] value layer): the rank-0 f64 realization \\
            collapses int64 scalar roots above 2^53."]
fn eval_int64_scalar_root_above_2p53_renders_exact() {
}

#[test]
#[ignore = "chelis#864 (chelis#717 stale-tag family): eval's labeled root."]
fn eval_f64_cast_tensor_root_renders_stored_width() {
}

#[test]
#[ignore = "chelis#865 ([#729]/[#686] capacity family): the untagged f64 box."]
fn c_boxed_f32_renders_at_own_width() {
}

#[test]
fn cross_lane_stdout_is_byte_identical_where_bits_agree() {
}

const C_LANE_EXCLUDED: &[&str] = &["f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max"];

const EVAL_F64_LIST_EXCLUDED: &[&str] = &[
    "f64-max",
    "f64-min-subnormal",
    "f64-min-normal",
    "f64-tenth",
    "f64-17-digit",
    "f64-2p53",
    "f64-2p53-plus-2",
    "f64-audit-e19",
];
"""


class IgnoreInventoryTests(unittest.TestCase):
    def test_multiline_and_single_line_ignores_are_both_attributed(self) -> None:
        cells = oracle.ignored_cells(HARNESS_FIXTURE)
        self.assertEqual(
            set(cells),
            {
                "eval_int64_scalar_root_above_2p53_renders_exact",
                "eval_f64_cast_tensor_root_renders_stored_width",
                "c_boxed_f32_renders_at_own_width",
            },
        )
        self.assertIn(
            "chelis#684",
            cells["eval_int64_scalar_root_above_2p53_renders_exact"],
            "the multi-line ignore reason must be captured whole",
        )

    def test_unignored_test_is_not_in_the_inventory(self) -> None:
        cells = oracle.ignored_cells(HARNESS_FIXTURE)
        self.assertNotIn("cross_lane_stdout_is_byte_identical_where_bits_agree", cells)
        self.assertTrue(
            oracle.defines_test(
                HARNESS_FIXTURE, "cross_lane_stdout_is_byte_identical_where_bits_agree"
            )
        )


class LedgerTests(unittest.TestCase):
    def test_the_shipped_ledger_matches_the_shipped_harness(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HARNESS_SOURCE).read_text(encoding="utf-8")
        self.assertEqual(oracle.ledger_violations(source, oracle.KNOWN_RED_CELLS), [])

    def test_fixture_matches_the_shipped_ledger(self) -> None:
        self.assertEqual(
            oracle.ledger_violations(HARNESS_FIXTURE, oracle.KNOWN_RED_CELLS), []
        )

    def test_an_undeclared_ignore_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE + (
            '\n#[test]\n#[ignore = "no ledger row"]\nfn sneaky_new_skip() {\n}\n'
        )
        violations = oracle.ledger_violations(source, oracle.KNOWN_RED_CELLS)
        self.assertTrue(
            any("sneaky_new_skip" in violation for violation in violations), violations
        )

    def test_a_stale_ledger_row_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE.replace(
            '#[ignore = "chelis#865 ([#729]/[#686] capacity family): the untagged f64 box."]\n',
            "",
        )
        violations = oracle.ledger_violations(source, oracle.KNOWN_RED_CELLS)
        self.assertTrue(
            any("c_boxed_f32_renders_at_own_width" in v for v in violations), violations
        )

    def test_an_ignore_that_drops_its_issue_citation_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE.replace("chelis#864", "some other reason")
        violations = oracle.ledger_violations(source, oracle.KNOWN_RED_CELLS)
        self.assertTrue(
            any("chelis#864" in violation for violation in violations), violations
        )

    def test_every_ledger_row_names_a_repair_owner_outside_this_plan(self) -> None:
        for cell in oracle.KNOWN_RED_CELLS:
            self.assertTrue(cell.owner, cell.name)
            self.assertIn(
                "chelis#729",
                cell.owner,
                "every known-red cell is a chelis#729-family value/capacity "
                "repair; rendering is not the fix site",
            )


class RedRunClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = oracle.KNOWN_RED_CELLS[0]

    def test_red_on_the_declared_assertion_meets_the_obligation(self) -> None:
        output = f"running 1 test\n{self.cell.fragment}\ntest result: FAILED"
        self.assertIsNone(oracle.classify_red_run(self.cell, 101, output))

    def test_a_green_cell_fails_the_oracle_and_names_the_issue(self) -> None:
        violation = oracle.classify_red_run(self.cell, 0, "running 1 test\nok")
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertIn(self.cell.issue, violation)
        self.assertIn("Un-ignore", violation)

    def test_red_for_an_undeclared_reason_fails_the_oracle(self) -> None:
        output = "running 1 test\npanicked at 'chelis eval: No such file'"
        violation = oracle.classify_red_run(self.cell, 101, output)
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertIn("NOT on its declared assertion", violation)

    def test_a_run_that_executed_no_test_fails_the_oracle(self) -> None:
        violation = oracle.classify_red_run(self.cell, 101, "error: no bin target")
        self.assertIsNotNone(violation)
        assert violation is not None
        self.assertIn("did not execute exactly one test", violation)


class ExclusionTests(unittest.TestCase):
    def test_the_shipped_exclusions_match_the_declared_ledger(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HARNESS_SOURCE).read_text(encoding="utf-8")
        self.assertEqual(oracle.exclusion_violations(source), [])

    def test_fixture_exclusions_match(self) -> None:
        self.assertEqual(oracle.exclusion_violations(HARNESS_FIXTURE), [])

    def test_a_widened_exclusion_list_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE.replace(
            '&["f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max"]',
            '&["f64-neg-zero", "f64-max", "f64-audit-e19", "f32-max", "f32-tenth"]',
        )
        violations = oracle.exclusion_violations(source)
        self.assertTrue(
            any("C_LANE_EXCLUDED" in violation for violation in violations), violations
        )

    def test_a_missing_exclusion_const_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE.replace("const C_LANE_EXCLUDED", "const RENAMED_AWAY")
        violations = oracle.exclusion_violations(source)
        self.assertTrue(
            any("cannot run" in violation for violation in violations), violations
        )


class UnignoredAndRetiredTests(unittest.TestCase):
    def _shipped_sources(self) -> dict:
        return {
            relative: (oracle.REPO_ROOT / relative).read_text(encoding="utf-8")
            for relative in (
                oracle.HARNESS_SOURCE,
                oracle.NARROW_MATRIX_SOURCE,
                oracle.REDUCTION_MATRIX_SOURCE,
                oracle.TRIPWIRE_SOURCE,
            )
        }

    def test_the_shipped_oracle_rows_are_present_and_unignored(self) -> None:
        self.assertEqual(oracle.unignored_violations(self._shipped_sources()), [])

    def test_the_retired_locks_are_absent_from_the_shipped_tree(self) -> None:
        self.assertEqual(oracle.retired_lock_violations(self._shipped_sources()), [])

    def test_a_reignored_oracle_row_is_a_violation(self) -> None:
        sources = self._shipped_sources()
        sources[oracle.REDUCTION_MATRIX_SOURCE] = sources[
            oracle.REDUCTION_MATRIX_SOURCE
        ].replace(
            "fn c_int64_tensor_print_is_exact_above_2p53() {",
            '#[ignore = "quietly re-ignored"]\nfn c_int64_tensor_print_is_exact_above_2p53() {',
        )
        violations = oracle.unignored_violations(sources)
        self.assertTrue(
            any("c_int64_tensor_print_is_exact_above_2p53" in v for v in violations),
            violations,
        )

    def test_a_deleted_oracle_row_is_a_violation(self) -> None:
        sources = self._shipped_sources()
        sources[oracle.NARROW_MATRIX_SOURCE] = sources[
            oracle.NARROW_MATRIX_SOURCE
        ].replace("fn c_to_list_of_f16_tensor_works(", "fn removed_row(")
        violations = oracle.unignored_violations(sources)
        self.assertTrue(
            any("c_to_list_of_f16_tensor_works" in v for v in violations), violations
        )

    def test_a_returning_interim_lock_is_a_violation(self) -> None:
        sources = self._shipped_sources()
        sources[oracle.NARROW_MATRIX_SOURCE] += (
            "\n#[test]\nfn c_f16_tensor_print_aborts_with_dtype_id_instead_of_misreading() {\n}\n"
        )
        violations = oracle.retired_lock_violations(sources)
        self.assertTrue(any("retired at Phase 2" in v for v in violations), violations)


class CrossLaneCorpusTests(unittest.TestCase):
    def _harness(self) -> str:
        return (oracle.REPO_ROOT / oracle.HARNESS_SOURCE).read_text(encoding="utf-8")

    def test_the_shipped_corpus_meets_its_declared_floor(self) -> None:
        labels = oracle.cross_lane_corpus_labels(self._harness())
        self.assertIsNotNone(labels)
        assert labels is not None
        for declared in oracle.CROSS_LANE_CORPUS_FLOOR:
            self.assertIn(declared, labels)
        self.assertEqual(oracle.cross_lane_corpus_violations(self._harness()), [])

    def test_a_shrunk_corpus_is_a_violation(self) -> None:
        source = self._harness().replace('("int64-exits"', '("int64-exits-renamed"', 1)
        violations = oracle.cross_lane_corpus_violations(source)
        self.assertTrue(
            any("int64-exits" in violation for violation in violations), violations
        )

    def test_a_grown_corpus_is_allowed(self) -> None:
        source = self._harness().replace(
            '("int8-exits", int_table_program("int8", INT_ROWS[0].1)),',
            '("int8-exits", int_table_program("int8", INT_ROWS[0].1)),\n'
            '        ("int16-exits", int_table_program("int16", INT_ROWS[1].1)),',
            1,
        )
        self.assertEqual(oracle.cross_lane_corpus_violations(source), [])

    def test_a_deleted_lock_is_a_violation(self) -> None:
        violations = oracle.cross_lane_corpus_violations("fn unrelated() {}\n")
        self.assertTrue(
            any("could not be parsed" in violation for violation in violations),
            violations,
        )


class FormatNarrowingTests(unittest.TestCase):
    def test_the_shipped_production_allowlist_is_empty(self) -> None:
        source = (oracle.REPO_ROOT / oracle.TRIPWIRE_SOURCE).read_text(encoding="utf-8")
        rows = oracle.format_narrowing_allowlist(source)
        self.assertTrue(rows, "the cfg(test) fixture row must still be parsed")
        self.assertEqual(oracle.format_narrowing_violations(source), [])

    def test_a_returning_production_row_is_a_violation(self) -> None:
        source = (
            "(\n"
            "    Pat::CFormatNarrowing,\n"
            '    "crates/chelis-backend-c/src/host_emit.rs",\n'
            "    4,\n"
            '    "regressed",\n'
            "),\n"
        )
        violations = oracle.format_narrowing_violations(source)
        self.assertTrue(any("third formatter" in v for v in violations), violations)


if __name__ == "__main__":
    unittest.main()
