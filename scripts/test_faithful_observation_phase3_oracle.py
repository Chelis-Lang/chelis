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


class RunnerTests(unittest.TestCase):
    def _run_main_with(self, fake_run):
        with (
            mock.patch.object(oracle, "source_violations", return_value=[]),
            mock.patch.object(oracle, "comparator_violations", return_value=[]),
            mock.patch.object(
                oracle, "definition_digest_violations", return_value=[]
            ),
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
                oracle, "definition_digest_violations", return_value=[]
            ),
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
        sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
            sources[oracle.PARITY_SOURCE],
            "parity_count_bool_axes",
            "{}",
        )
        sources[oracle.REJECTED_SOURCE] = oracle.replace_test_body(
            sources[oracle.REJECTED_SOURCE],
            "rejected_cells_fail_the_build_with_their_pinned_diagnostics",
            "{}",
        )

        violations = oracle.definition_digest_violations(sources)

        self.assertTrue(any("parity_tensor_structural_ops" in item for item in violations))
        self.assertTrue(any("parity_count_bool_axes" in item for item in violations))
        self.assertTrue(
            any(
                "rejected_cells_fail_the_build_with_their_pinned_diagnostics" in item
                for item in violations
            )
        )

    def test_checked_reshape_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_checked_reshape"
        self.assertIn(name, oracle.REQUIRED_TESTS[oracle.PARITY_SOURCE])
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("checked_reshape.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(
                    any(name in item for item in oracle.definition_digest_violations(sources))
                )

        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        sources[oracle.PARITY_SOURCE] = source.replace(
            f"fn {name}()", "fn deleted_checked_reshape_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))
        sources[oracle.PARITY_SOURCE] = source.replace('        "checked_reshape.ch",\n', "")
        self.assertTrue(
            any(
                "parity_corpus_is_complete" in item
                for item in oracle.definition_digest_violations(sources)
            )
        )

    def test_literal_extent_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_literal_extent_claim"
        self.assertIn(name, oracle.REQUIRED_TESTS[oracle.PARITY_SOURCE])
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("literal_extent_claim.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(
                    any(name in item for item in oracle.definition_digest_violations(sources))
                )

        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        sources[oracle.PARITY_SOURCE] = source.replace(
            f"fn {name}()", "fn deleted_literal_extent_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))
        sources[oracle.PARITY_SOURCE] = source.replace('        "literal_extent_claim.ch",\n', "")
        self.assertTrue(
            any(
                "parity_corpus_is_complete" in item
                for item in oracle.definition_digest_violations(sources)
            )
        )

    def test_generic_shape_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_generic_explicit_shape"
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("generic_explicit_shape.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(
                    any(name in item for item in oracle.definition_digest_violations(sources))
                )
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = sources[oracle.PARITY_SOURCE].replace(
            f"fn {name}()", "fn deleted_generic_shape_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))

    def test_window_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_checked_window_geometry"
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("checked_window_geometry.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(any(name in item for item in oracle.definition_digest_violations(sources)))
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = sources[oracle.PARITY_SOURCE].replace(
            f"fn {name}()", "fn deleted_window_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))

    def test_window_cannot_leave_the_frozen_corpus_inventory(self) -> None:
        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        _, _, start, end = oracle.test_definition_spans(source)["parity_corpus_is_complete"]
        body = source[start:end].replace('        "checked_window_geometry.ch",\n', "")
        self.assertNotEqual(body, source[start:end])
        sources[oracle.PARITY_SOURCE] = source[:start] + body + source[end:]
        self.assertTrue(any("parity_corpus_is_complete" in item for item in oracle.definition_digest_violations(sources)))

    def test_sparse_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_checked_sparse_axes"
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("checked_sparse_axes.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(any(name in item for item in oracle.definition_digest_violations(sources)))
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = sources[oracle.PARITY_SOURCE].replace(
            f"fn {name}()", "fn deleted_sparse_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))

    def test_sparse_cannot_leave_the_frozen_corpus_inventory(self) -> None:
        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        _, _, start, end = oracle.test_definition_spans(source)["parity_corpus_is_complete"]
        body = source[start:end].replace('        "checked_sparse_axes.ch",\n', "")
        self.assertNotEqual(body, source[start:end])
        sources[oracle.PARITY_SOURCE] = source[:start] + body + source[end:]
        self.assertTrue(any("parity_corpus_is_complete" in item for item in oracle.definition_digest_violations(sources)))

    def test_normalization_parity_remains_a_required_executable_row(self) -> None:
        name = "parity_explicit_normalization"
        for replacement in (
            "{}",
            '{ drive_parity(&examples_root().join("explicit_normalization.ch"), false); }',
        ):
            with self.subTest(replacement=replacement):
                sources = oracle.shipped_sources()
                sources[oracle.PARITY_SOURCE] = oracle.replace_test_body(
                    sources[oracle.PARITY_SOURCE], name, replacement
                )
                self.assertTrue(
                    any(name in item for item in oracle.definition_digest_violations(sources))
                )
        sources = oracle.shipped_sources()
        sources[oracle.PARITY_SOURCE] = sources[oracle.PARITY_SOURCE].replace(
            f"fn {name}()", "fn deleted_normalization_row()"
        )
        self.assertTrue(any(name in item for item in oracle.source_violations(sources)))

    def test_normalization_cannot_leave_the_frozen_corpus_inventory(self) -> None:
        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        _, _, start, end = oracle.test_definition_spans(source)["parity_corpus_is_complete"]
        body = source[start:end].replace('        "explicit_normalization.ch",\n', "")
        self.assertNotEqual(body, source[start:end])
        sources[oracle.PARITY_SOURCE] = source[:start] + body + source[end:]
        self.assertTrue(
            any(
                "parity_corpus_is_complete" in item
                for item in oracle.definition_digest_violations(sources)
            )
        )

    def test_generic_shape_cannot_leave_the_frozen_corpus_inventory(self) -> None:
        sources = oracle.shipped_sources()
        source = sources[oracle.PARITY_SOURCE]
        _, _, start, end = oracle.test_definition_spans(source)["parity_corpus_is_complete"]
        body = source[start:end].replace('        "generic_explicit_shape.ch",\n', "")
        self.assertNotEqual(body, source[start:end])
        sources[oracle.PARITY_SOURCE] = source[:start] + body + source[end:]
        self.assertTrue(
            any(
                "parity_corpus_is_complete" in item
                for item in oracle.definition_digest_violations(sources)
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

    def test_restoring_the_retired_formatter_in_the_agreement_harness_is_rejected(self) -> None:
        sources = oracle.shipped_sources()
        sources[oracle.EVAL_AGREEMENT_SOURCE] = sources[
            oracle.EVAL_AGREEMENT_SOURCE
        ].replace("chelis_string_from_scalar", "chelis_format_shortest")
        violations = oracle.comparator_violations(sources)
        self.assertTrue(any("chelis_string_from_scalar" in item for item in violations), violations)


if __name__ == "__main__":
    unittest.main()
