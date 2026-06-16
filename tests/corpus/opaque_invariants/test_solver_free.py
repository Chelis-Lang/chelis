"""Unit tests for the solver-free assertion's PURE core (no binary needed).

Run with the uv-managed interpreter:
    .venv/bin/python -m unittest -v \
        tests.corpus.opaque_invariants.test_solver_free
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_free as sf  # noqa: E402


class SymbolCountTests(unittest.TestCase):
    def test_zero_for_clean_output(self):
        nm = "0000 T main\n0000 T che_check_program\n0000 T parse_surf\n"
        self.assertEqual(sf.cvc5_symbol_count(nm), 0)

    def test_detects_cvc5_lines(self):
        nm = (
            "0000 T main\n"
            "0000 T cvc5::api::Solver::Solver()\n"
            "0000 T cvc5::api::Term::getKind()\n"
        )
        self.assertEqual(sf.cvc5_symbol_count(nm), 2)

    def test_case_insensitive(self):
        self.assertEqual(sf.cvc5_symbol_count("0000 T CVC5_init\n"), 1)


class VerdictTests(unittest.TestCase):
    def test_normalize_extracts_sorted_kind_message_pairs(self):
        stdout = (
            '{"errors": [{"kind": "B", "message": "two"}, '
            '{"kind": "A", "message": "one"}]}'
        )
        v = sf.normalize_check(2, stdout)
        self.assertEqual(v.exit_code, 2)
        self.assertEqual(v.errors, (("A", "one"), ("B", "two")))

    def test_identical_outputs_compare_equal(self):
        a = sf.normalize_check(2, '{"errors": [{"kind": "K", "message": "m"}]}')
        b = sf.normalize_check(2, '{"errors": [{"kind": "K", "message": "m"}]}')
        self.assertEqual(a, b)

    def test_different_exit_codes_differ(self):
        a = sf.normalize_check(0, '{"errors": []}')
        b = sf.normalize_check(2, '{"errors": []}')
        self.assertNotEqual(a, b)

    def test_different_error_kinds_differ(self):
        a = sf.normalize_check(2, '{"errors": [{"kind": "X", "message": "m"}]}')
        b = sf.normalize_check(2, '{"errors": [{"kind": "Y", "message": "m"}]}')
        self.assertNotEqual(a, b)

    def test_unparseable_stdout_is_a_distinct_verdict(self):
        v = sf.normalize_check(2, "not json")
        self.assertEqual(v.errors[0][0], "<unparseable>")


def _report(**kw):
    base = dict(
        no_link_ok=True,
        nonsmt_cvc5_count=0,
        smt_cvc5_count=32858,
        identity_checked=24,
        identity_mismatches=[],
        exit_mismatches=[],
        skipped=[],
    )
    base.update(kw)
    return sf.SolverFreeReport(**base)


class GateTests(unittest.TestCase):
    def test_clean_report_passes(self):
        sf.assert_solver_free(_report(), require_control=True)  # no raise

    def test_nonsmt_with_cvc5_fails_no_link(self):
        with self.assertRaises(sf.SolverFreeError):
            sf.assert_solver_free(_report(no_link_ok=False, nonsmt_cvc5_count=5),
                                  require_control=False)

    def test_identity_mismatch_fails(self):
        with self.assertRaises(sf.SolverFreeError):
            sf.assert_solver_free(_report(identity_mismatches=[{"id": "x"}]),
                                  require_control=True)

    def test_exit_mismatch_fails(self):
        with self.assertRaises(sf.SolverFreeError):
            sf.assert_solver_free(
                _report(exit_mismatches=[{"id": "x", "expected": 2, "actual": 0}]),
                require_control=True,
            )

    def test_require_control_fails_when_smt_absent_but_passes_without(self):
        with self.assertRaises(sf.SolverFreeError):
            sf.assert_solver_free(_report(smt_cvc5_count=None), require_control=True)
        sf.assert_solver_free(_report(smt_cvc5_count=None), require_control=False)

    def test_require_control_fails_on_nondiscriminating_probe(self):
        with self.assertRaises(sf.SolverFreeError):
            sf.assert_solver_free(_report(smt_cvc5_count=0), require_control=True)


if __name__ == "__main__":
    unittest.main()
