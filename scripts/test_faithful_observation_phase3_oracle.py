#!/usr/bin/env python3

import unittest
from unittest import mock

import faithful_observation_phase3_oracle as oracle


class RustDeclarationTests(unittest.TestCase):
    definition = '#[test]\nfn protected() { assert!(true); }'

    def test_comments_and_literals_cannot_supply_test_declarations(self):
        for hidden in [
            '/* ' + self.definition + ' */',
            '/* outer /* inner */ ' + self.definition + ' */',
            '// #[test]\n// fn protected() {}',
            'const TEXT: &str = r###"' + self.definition + '"###;',
            'const TEXT: &[u8] = br#"' + self.definition + '"#;',
            'const TEXT: &CStr = cr#"' + self.definition + '"#;',
            'const TEXT: &str = "' + self.definition + '";',
        ]:
            with self.subTest(hidden=hidden):
                self.assertEqual(oracle.test_declarations(hidden), {})
                self.assertEqual(oracle.test_definition_spans(hidden), {})
                real = hidden + '\n' + self.definition
                spans = oracle.test_definition_spans(real)
                self.assertEqual(set(spans), {'protected'})
                start, end, _, _ = spans['protected']
                self.assertEqual(real[start:end], self.definition)

    def test_real_attributes_keep_their_literal_ignore_reason(self):
        source = '#[test]\n#[ignore = "requires device"]\nfn protected() {}'
        self.assertEqual(oracle.ignored_tests(source), {'protected': 'requires device'})
        self.assertEqual(oracle.ignored_tests('/* ' + source + ' */'), {})

    def test_definition_boundaries_ignore_comment_literal_and_lifetime_braces(self):
        body = '''{
    let _ = r##"} #[test] fn fake() {"##;
    let _ = br#"}"#;
    let _ = cr#"}"#;
    let _ = "\\\"}";
    let _ = b'}'; let _ = '\\u{7d}';
    /* outer { /* inner } */ } */
    'outer: { let _: &'static str = "}"; break 'outer; }
}'''
        source = '#[test] /* separator { */ fn protected() ' + body
        start, end, opening, closing = oracle.test_definition_spans(source)['protected']
        self.assertEqual(source[start:end], source)
        self.assertEqual(source[opening:closing], body)
        self.assertEqual(oracle.replace_test_body(source, 'protected', '{}'), source[:opening] + '{}')

    def test_unterminated_comment_or_string_cannot_hide_source(self):
        for suffix in ['/*', '"', 'r#"', 'br##"', 'cr#"']:
            with self.subTest(suffix=suffix), self.assertRaises(ValueError):
                oracle.test_definition_spans(self.definition + '\n' + suffix)

    def test_commenting_each_required_definition_is_a_missing_row(self):
        original = oracle.shipped_sources()
        for path, required in oracle.REQUIRED_TESTS.items():
            spans = oracle.test_definition_spans(original[path])
            for name in required:
                with self.subTest(path=path, name=name):
                    start, end, _, _ = spans[name]
                    sources = dict(original)
                    source = sources[path]
                    sources[path] = source[:start] + '/*\n' + source[start:end] + '\n*/' + source[end:]
                    self.assertTrue(any(name in item and 'missing required' in item
                                        for item in oracle.source_violations(sources)))


class BodyRoleTests(unittest.TestCase):
    def test_export_covers_exactly_the_current_required_identities(self):
        contract = oracle.required_body_contract()
        self.assertEqual(contract["schema_version"], 2)
        observed = {
            oracle.Path(row["path"]): {test["name"] for test in row["tests"]}
            for row in contract["sources"]
        }
        self.assertEqual(observed, oracle.REQUIRED_TESTS)
        for row in contract["sources"]:
            self.assertEqual(len(row["tests"]), len(observed[oracle.Path(row["path"])]))
            self.assertTrue(all(test["calls"] for test in row["tests"]))

    def test_historical_library_suffix_does_not_disable_execution(self):
        source = next(row for row in oracle.required_body_contract()["sources"]
                      if row["path"] == str(oracle.PARITY_SOURCE))
        by_name = {row["name"]: row for row in source["tests"]}
        for name in ("parity_hello_tensor_library_only", "parity_opaque_invariants_simplex_library_only"):
            self.assertIs(by_name[name]["run_parity"], True)
        self.assertIs(by_name["parity_induction_bond_library_only"]["run_parity"], False)

    def test_stale_reviewed_role_or_unknown_source_fails(self):
        required = {path: set(names) for path, names in oracle.REQUIRED_TESTS.items()}
        required[oracle.PARITY_SOURCE].remove("parity_induction_bond_library_only")
        with mock.patch.object(oracle, "REQUIRED_TESTS", required), self.assertRaises(ValueError):
            oracle.required_body_contract()
        required = dict(oracle.REQUIRED_TESTS)
        required[oracle.Path("unknown.rs")] = {"new_test"}
        with mock.patch.object(oracle, "REQUIRED_TESTS", required), self.assertRaises(ValueError):
            oracle.required_body_contract()

    def test_result_roles_cover_the_reviewed_result_returning_entrypoints(self):
        results = {test["name"]: test["result"]
                   for source in oracle.required_body_contract()["sources"]
                   for test in source["tests"] if test["result"] is not None}
        self.assertEqual(len(results), 12)
        self.assertEqual(results["parity_corpus_is_complete"],
                         {"call": "parity_corpus::validate", "kind": "success"})
        for name in ("parity_comparator_accepts_byte_identical_tensor_lines",
                     "parity_comparator_byte_equal_for_non_tensor"):
            self.assertEqual(results[name], {"call": "assert_parity", "kind": "assert_ok"})
        self.assertEqual(results["agreement_compiled_observation_reaches_comparator"],
                         {"call": "compare_lanes_with", "kind": "expect_err"})
        self.assertEqual(results["agreement_width_nonconformance_is_behavioral"],
                         {"call": "compare_rendered_elements", "kind": "expect_err"})
        self.assertEqual(results["agreement_operation_identity_is_derived_from_ir"],
                         {"call": "agreement_op_for_risc", "kind": "assert_eq"})


class RunnerTests(unittest.TestCase):
    def _run_main_with(self, fake_run):
        with (
            mock.patch.object(oracle, "source_violations", return_value=[]),
            mock.patch.object(oracle, "comparator_violations", return_value=[]),
            mock.patch.object(oracle.shutil, "which", return_value="/usr/bin/tool"),
            mock.patch.object(oracle, "run_command", side_effect=fake_run),
            mock.patch.object(oracle, "receipt_violations", return_value=[]),
        ):
            return oracle.main()

    def test_main_executes_every_declared_suite_command(self) -> None:
        calls = []

        def fake_run(label, command, *, env=None):
            calls.append((label, command, env))
            return True

        self.assertEqual(self._run_main_with(fake_run), 0)
        self.assertEqual(
            [(label, command) for label, command, _env in calls],
            list(oracle.SUITE_COMMANDS),
        )
        self.assertTrue(all(env is None for _label, _command, env in calls[:-1]))
        receipt_env = calls[-1][2]
        self.assertIsNotNone(receipt_env)
        self.assertTrue(receipt_env["CHELIS_PHASE3_RECEIPT_PATH"])

    def test_main_stops_at_the_first_failed_suite_command(self) -> None:
        calls = []
        failed_command = oracle.SUITE_COMMANDS[1]

        def fake_run(label, command, *, env=None):
            calls.append((label, command))
            return (label, command) != failed_command

        self.assertEqual(self._run_main_with(fake_run), 1)
        self.assertEqual(calls, list(oracle.SUITE_COMMANDS[:2]))

    def test_preflight_reports_missing_tools_without_running_tests(self) -> None:
        with (
            mock.patch.object(oracle, "source_violations", return_value=[]),
            mock.patch.object(oracle, "comparator_violations", return_value=[]),
            mock.patch.object(
                oracle.shutil,
                "which",
                side_effect=lambda name: None if name == "cc" else "/tool",
            ),
        ):
            self.assertEqual(
                oracle.preflight_violations(),
                ["a host C compiler (`cc`) is required"],
            )


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

    # Definition mutations now execute through phase3_body_contract's Rust AST
    # audit: every empty/removed row and parity mode, the five forged receipts,
    # and the discarded completeness result. Receipt and source controls above
    # remain Python tests because those are this runner's own obligations.


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

    def test_restoring_the_retired_formatter_in_the_agreement_harness_is_rejected(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace("chelis_string_from_scalar", "chelis_format_shortest")
        violations = oracle.comparator_violations(sources)
        self.assertTrue(any("chelis_string_from_scalar" in item for item in violations), violations)


if __name__ == "__main__":
    unittest.main()
