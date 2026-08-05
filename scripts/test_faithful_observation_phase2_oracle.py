#!/usr/bin/env python3

import unittest
from unittest import mock

import faithful_observation_phase2_oracle as oracle


HARNESS_FIXTURE = """
#[test]
fn c_boxed_f32_renders_at_own_width() {
}

#[test]
fn c_suffixed_f32_literal_widens_from_its_stored_width() {
}

#[test]
fn cross_lane_stdout_is_byte_identical_where_bits_agree() {
}

"""

PROBE_LEDGER = (
    (
        "TEST_EXCLUDED",
        "chelis#test",
        (("test_exclusion_probe", ("hidden",)),),
        (("hidden", "f64", "cast(1.0, f64)", "1.0"),),
    ),
)

PROBE_FIXTURE = """
const TEST_EXCLUDED: &[&str] = &["hidden"];

#[test]
fn test_exclusion_probe() {
    for _label in TEST_EXCLUDED {}
    let _shrink_protocol = "DECLARED_EXCLUSIONS";
    println!("exclusion probe TEST_EXCLUDED visited: hidden");
}
"""

FIXTURE_CELL = oracle.RedCell(
    name="c_boxed_f32_renders_at_own_width",
    issue="chelis#865",
    fragment="boxed f32 elements must render shortest at their own width",
    owner="chelis#729/#686 capacity family",
)

# Parser-only sample for `ignored_cells`. It is deliberately NOT checked
# against the shipped ledger: its job is to keep both attribute spellings
# covered. The multi-line leg used to come from the chelis#684 row, which
# was un-ignored when chelis#729 repaired scalar-root storage (chelis#1078).
IGNORE_PARSER_FIXTURE = """
#[test]
#[ignore = "chelis#111 (some family): a reason long enough to wrap across \\
            two source lines."]
fn multi_line_ignored_cell() {
}

#[test]
#[ignore = "chelis#222 (another family): a single-line reason."]
fn single_line_ignored_cell() {
}

#[test]
fn cross_lane_stdout_is_byte_identical_where_bits_agree() {
}
"""


class IgnoreInventoryTests(unittest.TestCase):
    def test_multiline_and_single_line_ignores_are_both_attributed(self) -> None:
        cells = oracle.ignored_cells(IGNORE_PARSER_FIXTURE)
        self.assertEqual(
            set(cells),
            {"multi_line_ignored_cell", "single_line_ignored_cell"},
        )
        self.assertIn(
            "chelis#111",
            cells["multi_line_ignored_cell"],
            "the multi-line ignore reason must be captured whole",
        )
        self.assertIn(
            "two source lines",
            cells["multi_line_ignored_cell"],
            "the continuation line must be captured too",
        )
        self.assertIn("chelis#222", cells["single_line_ignored_cell"])

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
        violations = oracle.ledger_violations(HARNESS_FIXTURE, (FIXTURE_CELL,))
        self.assertTrue(
            any("c_boxed_f32_renders_at_own_width" in v for v in violations), violations
        )

    def test_an_ignore_that_drops_its_issue_citation_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE.replace(
            "#[test]\nfn c_boxed_f32_renders_at_own_width()",
            '#[test]\n#[ignore = "some other reason"]\nfn c_boxed_f32_renders_at_own_width()',
        )
        violations = oracle.ledger_violations(source, (FIXTURE_CELL,))
        self.assertTrue(
            any("chelis#865" in violation for violation in violations), violations
        )

    def test_repaired_boxed_f32_cell_left_the_red_ledger(self) -> None:
        self.assertEqual(oracle.KNOWN_RED_CELLS, ())
        self.assertIn(
            (oracle.HARNESS_SOURCE, "c_boxed_f32_renders_at_own_width", "chelis#865"),
            oracle.UNIGNORED_ROWS,
        )
        self.assertIn(
            (
                oracle.HARNESS_SOURCE,
                "c_suffixed_f32_literal_widens_from_its_stored_width",
                "chelis#1110",
            ),
            oracle.UNIGNORED_ROWS,
        )


class RedRunClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = FIXTURE_CELL

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
        source = HARNESS_FIXTURE + '\nconst C_LANE_EXCLUDED: &[&str] = &["f32-tenth"];\n'
        violations = oracle.exclusion_violations(source)
        self.assertTrue(
            any("C_LANE_EXCLUDED" in violation for violation in violations), violations
        )

    def test_no_exclusion_const_is_required_when_the_ledger_is_empty(self) -> None:
        self.assertEqual(oracle.DECLARED_EXCLUSIONS, ())
        self.assertEqual(oracle.exclusion_violations(HARNESS_FIXTURE), [])

    def test_an_undeclared_exclusion_const_is_a_violation(self) -> None:
        source = HARNESS_FIXTURE + '\nconst NEW_ROWS_EXCLUDED: &[&str] = &["hidden"];\n'
        violations = oracle.exclusion_violations(source)
        self.assertTrue(
            any("NEW_ROWS_EXCLUDED" in violation and "not declared" in violation for violation in violations),
            violations,
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

    def test_a_conditionally_reignored_repaired_row_is_a_violation(self) -> None:
        sources = self._shipped_sources()
        sources[oracle.HARNESS_SOURCE] = sources[oracle.HARNESS_SOURCE].replace(
            "#[test]\nfn eval_f64_cast_tensor_root_renders_stored_width() {",
            "#[test]\n"
            '#[cfg_attr(not(any()), ignore = "silently re-ignore chelis#864")]\n'
            "fn eval_f64_cast_tensor_root_renders_stored_width() {",
        )

        violations = oracle.unignored_violations(sources)

        self.assertTrue(
            any("eval_f64_cast_tensor_root_renders_stored_width" in v for v in violations),
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
        for declared, _tokens in oracle.CROSS_LANE_CORPUS_FLOOR:
            self.assertIn(declared, labels)
        self.assertEqual(oracle.cross_lane_corpus_violations(self._harness()), [])

    def test_a_hollowed_out_program_is_a_violation_even_with_its_label(self) -> None:
        """R2 LOW: a label alone did not bind its body."""
        entries = oracle.cross_lane_corpus_programs(self._harness())
        assert entries is not None
        source = self._harness().replace(
            entries["bool-exits"],
            '("bool-exits", exits_program("tensor[1, int8]", "to_tensor([1])")),\n        ',
            1,
        )
        violations = oracle.cross_lane_corpus_violations(source)
        self.assertTrue(
            any("bool-exits" in v and "hollowed out" in v for v in violations), violations
        )

    def test_a_weakened_assertion_in_the_lock_is_a_violation(self) -> None:
        source = self._harness().replace(
            "fn cross_lane_stdout_is_byte_identical_where_bits_agree() {",
            "fn cross_lane_stdout_is_byte_identical_where_bits_agree() {\n    return;",
            1,
        )
        # Drop the comparison entirely from the lock body.
        body = oracle.cross_lane_lock_body(source)
        assert body is not None
        source = source.replace(body, body.replace("assert_eq!", "let _unused ="))
        violations = oracle.cross_lane_corpus_violations(source)
        self.assertTrue(any("assert_eq!" in v for v in violations), violations)

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


class ObservationDecodeTableTests(unittest.TestCase):
    def _runtime(self) -> str:
        return (oracle.REPO_ROOT / oracle.RUNTIME_SOURCE).read_text(encoding="utf-8")

    def test_the_shipped_arms_match_the_declared_decode_table(self) -> None:
        self.assertEqual(oracle.observation_decode_violations(self._runtime()), [])

    def test_every_runtime_dtype_arm_is_declared(self) -> None:
        arms = oracle.observation_decode_arms(self._runtime())
        self.assertIsNotNone(arms)
        assert arms is not None
        declared = {name for name, _, _ in oracle.OBSERVATION_DECODE_TABLE}
        self.assertEqual(set(arms), declared)

    def test_an_int_arm_on_the_f32_view_is_a_violation(self) -> None:
        source = self._runtime().replace(
            "RuntimeDType::I32 => (*i32::data_ptr_unchecked(tm).add(i)).to_string(),",
            "RuntimeDType::I32 => (*data_as_f32_const(t).add(i) as i64).to_string(),",
            1,
        )
        violations = oracle.observation_decode_violations(source)
        self.assertTrue(any("I32" in v for v in violations), violations)
        self.assertTrue(
            any("untyped f32 view" in v for v in violations),
            "the f32 view is the defect mechanism and must be named",
        )

    def test_a_swapped_typed_accessor_is_a_violation(self) -> None:
        source = self._runtime().replace(
            "RuntimeDType::I16 => (*i16::data_ptr_unchecked(tm).add(i)).to_string(),",
            "RuntimeDType::I16 => (*i32::data_ptr_unchecked(tm).add(i)).to_string(),",
            1,
        )
        violations = oracle.observation_decode_violations(source)
        self.assertTrue(any("I16" in v for v in violations), violations)

    def test_bool_is_the_only_declared_f32_view_exception(self) -> None:
        exceptions = [
            name
            for name, view, _ in oracle.OBSERVATION_DECODE_TABLE
            if view == "data_as_f32_const"
        ]
        self.assertEqual(exceptions, ["Bool"])
        why = next(
            why
            for name, _, why in oracle.OBSERVATION_DECODE_TABLE
            if name == "Bool"
        )
        self.assertIn("chelis#894", why, "the exception must name its retirement")

    def test_a_deleted_decoder_is_a_violation(self) -> None:
        violations = oracle.observation_decode_violations("fn unrelated() {}\n")
        self.assertTrue(any("cannot be checked" in v for v in violations), violations)

    # The R2 red team put the expected accessor in a COMMENT while the body
    # read through a foreign one; the substring check that preceded this
    # passed the whole oracle. Each variant below must now be caught.
    def test_line_comment_decoy_is_a_violation(self) -> None:
        source = self._runtime().replace(
            "RuntimeDType::I16 => (*i16::data_ptr_unchecked(tm).add(i)).to_string(),",
            "RuntimeDType::I16 => {\n"
            "            // decoy: i16::data_ptr_unchecked\n"
            "            (*i32::data_ptr_unchecked(tm).add(i)).to_string()\n"
            "        }",
            1,
        )
        violations = oracle.observation_decode_violations(source)
        self.assertTrue(any("I16" in v for v in violations), violations)

    def test_block_comment_decoy_is_a_violation(self) -> None:
        source = self._runtime().replace(
            "RuntimeDType::I16 => (*i16::data_ptr_unchecked(tm).add(i)).to_string(),",
            "RuntimeDType::I16 => { /* i16::data_ptr_unchecked */ "
            "(*i64::data_ptr_unchecked(tm).add(i)).to_string() }",
            1,
        )
        violations = oracle.observation_decode_violations(source)
        self.assertTrue(any("I16" in v for v in violations), violations)

    def test_a_foreign_accessor_beside_the_declared_one_is_a_violation(self) -> None:
        source = self._runtime().replace(
            "RuntimeDType::I32 => (*i32::data_ptr_unchecked(tm).add(i)).to_string(),",
            "RuntimeDType::I32 => { let _ = i32::data_ptr_unchecked(tm); "
            "(*i64::data_ptr_unchecked(tm).add(i)).to_string() }",
            1,
        )
        violations = oracle.observation_decode_violations(source)
        self.assertTrue(
            any("reads through" in v and "I32" in v for v in violations), violations
        )

    def test_strip_comments_removes_both_comment_forms(self) -> None:
        self.assertNotIn("hidden", oracle.strip_comments("code // hidden\nmore"))
        self.assertNotIn("hidden", oracle.strip_comments("code /* hidden */ more"))
        self.assertIn("code", oracle.strip_comments("code // hidden"))


class DeadExportTests(unittest.TestCase):
    def test_the_shipped_tree_has_no_zero_emitter_print_export(self) -> None:
        source = (oracle.REPO_ROOT / oracle.RUNTIME_SOURCE).read_text(encoding="utf-8")
        header = (oracle.REPO_ROOT / oracle.RUNTIME_HEADER).read_text(encoding="utf-8")
        self.assertEqual(oracle.dead_export_violations(source, header), [])

    def test_a_returning_export_is_a_violation_in_both_surfaces(self) -> None:
        violations = oracle.dead_export_violations(
            "#[no_mangle]\n"
            'pub unsafe extern "C" fn chelis_print_f32(t: *const chelis_tensor) {}\n',
            "void chelis_print_f32(const chelis_tensor *t);\n",
        )
        self.assertEqual(len(violations), 2, violations)
        self.assertTrue(any("zero emitters" in v for v in violations), violations)
        self.assertTrue(any("published C ABI" in v for v in violations), violations)

    # R2 LOW: substring presence could not tell a live declaration from a
    # comment about the removal.
    def test_a_comment_naming_the_retired_export_is_not_a_violation(self) -> None:
        violations = oracle.dead_export_violations(
            "// chelis_print_f32 was removed at Phase 2; see the census.\n",
            "/* chelis_print_f32 was removed; do not re-declare it. */\n",
        )
        self.assertEqual(violations, [])

    def test_the_shipped_lib_mentions_the_retired_name_in_prose_only(self) -> None:
        source = (oracle.REPO_ROOT / oracle.RUNTIME_SOURCE).read_text(encoding="utf-8")
        self.assertIn(
            "chelis_print_f32", source, "the removal note should still be readable"
        )
        self.assertNotIn("chelis_print_f32", oracle.rust_exported_symbols(source))

    def test_the_parsed_surfaces_are_non_trivial(self) -> None:
        """A parser that silently matched nothing would pass every check."""
        crate = oracle.read_runtime_crate_sources(oracle.REPO_ROOT)
        header = (oracle.REPO_ROOT / oracle.RUNTIME_HEADER).read_text(encoding="utf-8")
        exported = oracle.rust_exported_symbols(crate)
        declared = oracle.header_declared_functions(header)
        self.assertGreater(len(exported), 50, "no-mangle exports failed to parse")
        self.assertGreater(len(declared), 50, "header declarations failed to parse")
        self.assertIn("chelis_format_shortest", exported)
        self.assertIn("chelis_format_shortest", declared)

    def test_the_scan_covers_the_whole_crate_not_only_lib_rs(self) -> None:
        """`chelis_format_shortest` lives in a sibling module, not lib.rs.

        A retired export re-added in any module is the same public exit
        returning; scoping the scan to one file would miss it.
        """
        lib_only = (oracle.REPO_ROOT / oracle.RUNTIME_SOURCE).read_text(encoding="utf-8")
        crate = oracle.read_runtime_crate_sources(oracle.REPO_ROOT)
        self.assertNotIn("chelis_format_shortest", oracle.rust_exported_symbols(lib_only))
        self.assertIn("chelis_format_shortest", oracle.rust_exported_symbols(crate))

    def test_the_shipped_crate_and_header_are_clean(self) -> None:
        crate = oracle.read_runtime_crate_sources(oracle.REPO_ROOT)
        header = (oracle.REPO_ROOT / oracle.RUNTIME_HEADER).read_text(encoding="utf-8")
        self.assertEqual(oracle.dead_export_violations(crate, header), [])


class FormatNarrowingTests(unittest.TestCase):
    def _tripwire(self) -> str:
        return (oracle.REPO_ROOT / oracle.TRIPWIRE_SOURCE).read_text(encoding="utf-8")

    def test_the_shipped_production_allowlist_is_empty(self) -> None:
        source = self._tripwire()
        rows = oracle.format_narrowing_allowlist(source)
        self.assertTrue(rows, "the cfg(test) fixture row must still be parsed")
        self.assertEqual(oracle.format_narrowing_violations(source), [])

    def test_every_declared_class_parses_baseline_rows(self) -> None:
        """A silently-empty parse for a class would vacuously pass it."""
        source = self._tripwire()
        for variant, class_id, _permitted in oracle.FORMAT_CLASS_TABLE:
            rows = oracle.format_narrowing_allowlist(source, variant)
            self.assertTrue(rows, f"{class_id} ({variant}) parsed no baseline rows")

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

    def test_an_unpermitted_rust_class_row_is_a_violation_per_class(self) -> None:
        for variant, class_id in (
            ("RustFormatNarrowing", "rust-format-narrowing"),
            ("RustDebugNumericFormat", "rust-debug-numeric-format"),
        ):
            source = (
                "(\n"
                f"    Pat::{variant},\n"
                '    "crates/chelis-backend-c/src/host_emit.rs",\n'
                "    1,\n"
                '    "planted",\n'
                "),\n"
            )
            violations = oracle.format_narrowing_violations(source)
            self.assertTrue(
                any(class_id in v and "host_emit.rs" in v for v in violations),
                (class_id, violations),
            )

    def test_a_permitted_row_growing_is_the_tripwires_business_not_ours(self) -> None:
        """The oracle guards PATHS; counts are the tripwire's exact-count
        ratchet. A permitted path at any count is not an oracle violation."""
        source = (
            "(\n"
            "    Pat::RustDebugNumericFormat,\n"
            '    "crates/chelis-types/src/observation.rs",\n'
            "    999,\n"
            '    "grown",\n'
            "),\n"
        )
        self.assertEqual(oracle.format_narrowing_violations(source), [])


class DocCitationParityTests(unittest.TestCase):
    def _tripwire(self) -> str:
        return (oracle.REPO_ROOT / oracle.TRIPWIRE_SOURCE).read_text(encoding="utf-8")

    def test_the_shipped_tripwire_has_citation_parity(self) -> None:
        self.assertEqual(oracle.doc_citation_violations(self._tripwire()), [])

    def test_the_shipped_doc_fn_parses_non_trivially(self) -> None:
        arms = oracle.tripwire_doc_arms(self._tripwire())
        self.assertIsNotNone(arms)
        assert arms is not None
        cited = {
            name
            for names, citation in arms
            if "faithful_observation.md" in citation
            for name in names
        }
        self.assertEqual(
            cited, {variant for variant, _, _ in oracle.FORMAT_CLASS_TABLE}
        )

    def test_a_hosted_detector_without_a_coverage_row_is_a_violation(self) -> None:
        source = self._tripwire().replace(
            '_ => "spec/design/loud_unsupported.md B2.5",',
            'Pat::SneakyFormatter => "spec/design/faithful_observation.md B2.4",\n'
            '            _ => "spec/design/loud_unsupported.md B2.5",',
            1,
        )
        violations = oracle.doc_citation_violations(source)
        self.assertTrue(
            any("SneakyFormatter" in v and "cannot see" in v for v in violations),
            violations,
        )

    def test_a_stale_coverage_row_is_a_violation(self) -> None:
        source = self._tripwire().replace(
            "Pat::CFormatNarrowing | Pat::RustFormatNarrowing | "
            "Pat::RustDebugNumericFormat => {",
            "Pat::CFormatNarrowing | Pat::RustFormatNarrowing => {",
            1,
        )
        violations = oracle.doc_citation_violations(source)
        self.assertTrue(
            any("RustDebugNumericFormat" in v for v in violations), violations
        )

    def test_an_unparseable_doc_fn_is_reported_not_vacuous(self) -> None:
        violations = oracle.doc_citation_violations("fn unrelated() {}\n")
        self.assertTrue(
            any("could not be located" in v for v in violations), violations
        )


class ExclusionProbeTests(unittest.TestCase):
    def test_the_shipped_empty_ledger_needs_no_probes(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HARNESS_SOURCE).read_text(encoding="utf-8")
        self.assertEqual(oracle.DECLARED_EXCLUSIONS, ())
        self.assertEqual(oracle.exclusion_probe_violations(source), [])

    def test_a_declared_probe_satisfies_all_legs(self) -> None:
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            self.assertEqual(oracle.exclusion_probe_violations(PROBE_FIXTURE), [])

    def test_a_deleted_probe_is_a_violation(self) -> None:
        source = PROBE_FIXTURE.replace("fn test_exclusion_probe(", "fn renamed_probe(")
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.exclusion_probe_violations(source)
        self.assertTrue(
            any("not defined" in v and "TEST_EXCLUDED" in v for v in violations),
            violations,
        )

    def test_an_ignored_probe_is_a_disabled_leg_violation(self) -> None:
        source = PROBE_FIXTURE.replace(
            "#[test]\nfn test_exclusion_probe() {",
            '#[test]\n#[ignore = "silenced"]\nfn test_exclusion_probe() {',
            1,
        )
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.exclusion_probe_violations(source)
        self.assertTrue(
            any("disabled re-execution leg" in v for v in violations), violations
        )

    def test_a_probe_dropping_the_const_reference_is_a_violation(self) -> None:
        body = oracle.test_fn_body(PROBE_FIXTURE, "test_exclusion_probe")
        assert body is not None
        source = PROBE_FIXTURE.replace(body, body.replace("TEST_EXCLUDED", "LOCAL_LABELS"))
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.exclusion_probe_violations(source)
        self.assertTrue(
            any("no longer references the list constant" in v for v in violations),
            violations,
        )

    def test_a_probe_dropping_the_shrink_protocol_is_a_violation(self) -> None:
        body = oracle.test_fn_body(PROBE_FIXTURE, "test_exclusion_probe")
        assert body is not None
        source = PROBE_FIXTURE.replace(body, body.replace("DECLARED_EXCLUSIONS", "SOME_LEDGER"))
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.exclusion_probe_violations(source)
        self.assertTrue(
            any("shrink-protocol" in v for v in violations), violations
        )


class ProbeAttributeTests(unittest.TestCase):
    def test_a_declared_probe_is_an_unconditional_test(self) -> None:
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            self.assertEqual(oracle.probe_attribute_violations(PROBE_FIXTURE), [])
        self.assertEqual(
            oracle.probe_attributes(PROBE_FIXTURE, "test_exclusion_probe"), ["#[test]"]
        )

    def test_a_cfg_attr_ignore_is_a_violation(self) -> None:
        source = PROBE_FIXTURE.replace(
            "#[test]\nfn test_exclusion_probe() {",
            "#[test]\n#[cfg_attr(all(), ignore)]\nfn test_exclusion_probe() {",
            1,
        )
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.probe_attribute_violations(source)
        self.assertTrue(
            any("cfg_attr" in v or "exactly the" in v for v in violations), violations
        )

    def test_a_missing_test_attribute_is_a_violation(self) -> None:
        source = PROBE_FIXTURE.replace("#[test]\nfn test_exclusion_probe() {", "fn test_exclusion_probe() {")
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.probe_attribute_violations(source)
        self.assertTrue(
            any("test_exclusion_probe" in v for v in violations),
            violations,
        )

    def test_a_multi_line_attribute_fails_closed(self) -> None:
        source = PROBE_FIXTURE.replace(
            "#[test]\nfn test_exclusion_probe() {",
            "#[test]\n#[cfg_attr(\n    all(),\n    ignore\n)]\n"
            "fn test_exclusion_probe() {",
            1,
        )
        with mock.patch.object(oracle, "DECLARED_EXCLUSIONS", PROBE_LEDGER):
            violations = oracle.probe_attribute_violations(source)
        self.assertTrue(
            any("test_exclusion_probe" in v for v in violations),
            "an attribute shape the parser cannot vouch for must be a "
            f"violation, never a pass; got {violations}",
        )


class ProbeReceiptTests(unittest.TestCase):
    def test_receipt_lines_parse_into_ordered_lists(self) -> None:
        output = (
            "running 1 test\n"
            "exclusion probe C_LANE_EXCLUDED visited: f64-max f32-max\n"
            "exclusion probe C_LANE_EXCLUDED visited: f64-neg-zero\n"
            "ok\n"
        )
        self.assertEqual(
            oracle.probe_receipts(output),
            {"C_LANE_EXCLUDED": ["f64-max", "f32-max", "f64-neg-zero"]},
        )

    def test_exact_ordered_receipts_pass(self) -> None:
        runs = [
            ("trio", ("a", "b"), 0, "running 1 test\nexclusion probe C visited: a b\nok"),
            ("solo", ("c",), 0, "running 1 test\nexclusion probe C visited: c\nok"),
        ]
        self.assertEqual(oracle.classify_probe_outputs("C", runs), [])

    def test_a_shrunken_receipt_is_a_violation(self) -> None:
        runs = [("trio", ("a", "b"), 0, "running 1 test\nexclusion probe C visited: a\nok")]
        violations = oracle.classify_probe_outputs("C", runs)
        self.assertTrue(
            any("ordered receipt" in v for v in violations),
            f"a probe that stops iterating the ledger must fail; got {violations}",
        )

    def test_a_duplicated_receipt_is_a_violation(self) -> None:
        """PR #962 round-2 M1: the set-union form accepted `a a b b` for
        the declaration ["a", "b"]. Multiplicity now fails."""
        runs = [
            ("trio", ("a", "b"), 0, "running 1 test\nexclusion probe C visited: a a b b\nok")
        ]
        violations = oracle.classify_probe_outputs("C", runs)
        self.assertTrue(any("ordered receipt" in v for v in violations), violations)

    def test_a_reordered_receipt_is_a_violation(self) -> None:
        runs = [
            ("trio", ("a", "b"), 0, "running 1 test\nexclusion probe C visited: b a\nok")
        ]
        violations = oracle.classify_probe_outputs("C", runs)
        self.assertTrue(any("ordered receipt" in v for v in violations), violations)

    def test_a_failing_probe_is_reported_with_its_output(self) -> None:
        runs = [("trio", ("a",), 101, "running 1 test\nGOOD NEWS: ...\nFAILED")]
        violations = oracle.classify_probe_outputs("C", runs)
        self.assertTrue(any("FAILED (exit 101)" in v for v in violations), violations)

    def test_a_zero_test_run_is_a_violation(self) -> None:
        runs = [("trio", ("a",), 0, "running 0 tests\nok")]
        violations = oracle.classify_probe_outputs("C", runs)
        self.assertTrue(
            any("did not execute exactly one test" in v for v in violations), violations
        )


class GroundTruthTests(unittest.TestCase):
    """PR #962 round-2 M1's required regression: the strict-subset probe
    with a forged full receipt passes the receipt check by construction
    (receipts are probe-authored), so the catch is the oracle's OWN
    re-execution - these tests pin its classifiers on real current
    behavior and on both failure directions."""

    def test_c_fingerprints_confirm_and_flip(self) -> None:
        giant = "chelis_fill(t, 179769313486231570000000000000000000000000.0);"
        bare = giant.replace(".0", "")
        self.assertFalse(oracle.c_has_bare_giant_integer_literal(giant))
        self.assertTrue(oracle.c_has_bare_giant_integer_literal(bare))
        entries = [
            oracle.CExclusionGroundTruth("f64-max", True, bare, "wrong-bits")
        ]
        self.assertEqual(oracle.c_exclusion_ground_truth_violations(entries), [])
        violations = oracle.c_exclusion_ground_truth_violations(
            [oracle.CExclusionGroundTruth("f64-max", True, giant, "exact")]
        )
        self.assertTrue(any("repair landed" in v for v in violations), violations)
        violations = oracle.c_exclusion_ground_truth_violations(
            [oracle.CExclusionGroundTruth("f64-max", False, "", "not-run")]
        )
        self.assertTrue(any("build" in v.lower() for v in violations), violations)

    def test_c_fingerprint_ignores_comments_and_strings(self) -> None:
        giant = "12345678901234567890"
        self.assertTrue(oracle.c_has_bare_giant_integer_literal(f"double x = {giant};"))
        self.assertFalse(
            oracle.c_has_bare_giant_integer_literal(
                f'double x = 1e20; // stale {giant}\n'
                f'const char *s = "{giant}"; char c = \'0\'; /* stale {giant} */'
            )
        )

    def test_exact_native_behavior_forces_shrink_even_with_a_fingerprint(self) -> None:
        violations = oracle.c_exclusion_ground_truth_violations(
            [
                oracle.CExclusionGroundTruth(
                    "f64-max",
                    True,
                    "double x = 12345678901234567890;",
                    "exact",
                )
            ]
        )
        self.assertTrue(any("repair landed" in v for v in violations), violations)

    def test_neg_zero_emission_legs(self) -> None:
        self.assertEqual(
            oracle.c_exclusion_ground_truth_violations(
                [
                    oracle.CExclusionGroundTruth(
                        "f64-neg-zero", True, "double c = -0;", "wrong-bits"
                    )
                ]
            ),
            [],
        )
        violations = oracle.c_exclusion_ground_truth_violations(
            [
                oracle.CExclusionGroundTruth(
                    "f64-neg-zero", True, "double c = -0.0;", "exact"
                )
            ]
        )
        self.assertTrue(any("repair landed" in v for v in violations), violations)


class VerdictSinkTests(unittest.TestCase):
    """PR #962 round-4 F2: sink identity is asserted and the raise is
    helper-owned, so a scratch list can neither mint a consumed receipt
    nor swallow a failure."""

    def test_a_scratch_list_cannot_mint_a_consumed_receipt(self) -> None:
        oracle.CONSUMED_INSTRUMENTS.discard("classify_probe_outputs")
        with self.assertRaisesRegex(oracle.OracleFailure, "scratch-list"):
            with oracle.verdict_sink("test leg"):
                oracle.consume_findings([], oracle.classify_probe_outputs, "C", [])
        self.assertNotIn("classify_probe_outputs", oracle.CONSUMED_INSTRUMENTS)

    def test_the_sink_context_owns_the_raise(self) -> None:
        def mandatory(_arg: object) -> list[str]:
            return ["RT4 mandatory violation"]

        with self.assertRaisesRegex(oracle.OracleFailure, "RT4 mandatory violation"):
            with oracle.verdict_sink("test leg") as sink:
                oracle.consume_findings(sink, mandatory, None)
                # A caller "forgetting" to raise changes nothing: the
                # context raises from the same list on exit.

    def test_an_empty_sink_exits_clean_and_records_consumption(self) -> None:
        def clean(_arg: object) -> list[str]:
            return []

        with oracle.verdict_sink("test leg") as sink:
            oracle.consume_findings(sink, clean, None)
        self.assertIn("clean", oracle.CONSUMED_INSTRUMENTS)

    def test_nested_sinks_are_rejected(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "nested"):
            with oracle.verdict_sink("outer"):
                with oracle.verdict_sink("inner"):
                    pass

    def test_consume_outside_any_sink_is_a_forgery(self) -> None:
        def clean(_arg: object) -> list[str]:
            return []

        with self.assertRaisesRegex(oracle.OracleFailure, "scratch-list"):
            oracle.consume_findings([], clean, None)


class GroundTruthDriverTests(unittest.TestCase):
    """PR #962 round-4 F4: the executable-C driver's helpers are covered
    directly, and plumbing failures raise instead of degrading to a
    native status the classifier accepts as 'still broken'."""

    def test_compile_command_parses_from_build_output(self) -> None:
        command = oracle.parse_compile_command(
            "Wrote p.c\nCompile: cc p.c runtime.c -o p_bin -lm\nDone\n"
        )
        self.assertEqual(command[0], "cc")
        self.assertIn("-o", command)

    def test_a_missing_compile_line_is_loud_plumbing(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "no `Compile:` line"):
            oracle.parse_compile_command("Wrote p.c\nDone\n")

    def test_a_compile_line_without_output_is_loud_plumbing(self) -> None:
        with self.assertRaisesRegex(oracle.OracleFailure, "names no `-o`"):
            oracle.parse_compile_command("Compile: cc p.c runtime.c -lm\n")

    def test_rendered_bit_classification_both_directions(self) -> None:
        self.assertEqual(
            oracle.classify_rendered_bits(["0.1", "0.1"], "f64", "0.1"), "exact"
        )
        self.assertEqual(
            oracle.classify_rendered_bits(["0.1", "0.2"], "f64", "0.1"),
            "wrong-bits",
        )
        self.assertEqual(
            oracle.classify_rendered_bits(["garbage"], "f64", "0.1"), "wrong-bits"
        )
        # The f32 comparison reconciles an f64-image spelling against the
        # narrowed width.
        self.assertEqual(
            oracle.classify_rendered_bits(
                ["0.10000000149011612"], "f32", "0.1"
            ),
            "exact",
        )

    def test_rendered_values_parse_every_exit_shape(self) -> None:
        stdout = (
            "tensor(shape=[1], data=[5e-324])\n"
            "[5e-324]\n"
            "lroot = [5e-324]\n"
            "shown = ()\n"
        )
        self.assertEqual(
            oracle.c_rendered_values(stdout), ["5e-324", "5e-324", "5e-324"]
        )
        self.assertEqual(oracle.c_rendered_values("no renders here\n"), [])


class KnownRedCellRunTests(unittest.TestCase):
    """PR #962 round-4 F3: the known-red leg's failure paths are executed
    by the unit suite through an injected runner - the gone-green branch
    shipped a NameError because nothing ran it."""

    class _Completed:
        def __init__(self, returncode: int, stdout: str) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def test_empty_ledger_has_an_empty_real_run_set(self) -> None:
        with mock.patch.object(
            oracle, "classify_red_run", wraps=oracle.classify_red_run
        ) as classifier:
            self.assertEqual(oracle.known_red_run_violations((), ()), [])
        classifier.assert_not_called()

    def test_nonempty_ledger_cannot_omit_its_real_run(self) -> None:
        violations = oracle.known_red_run_violations((FIXTURE_CELL,), ())
        self.assertTrue(
            any(
                "did not execute its declared run set" in violation
                for violation in violations
            ),
            violations,
        )

    def test_default_ledger_is_resolved_at_call_time(self) -> None:
        runner = mock.Mock(
            return_value=self._Completed(
                101,
                f"running 1 test\n{FIXTURE_CELL.fragment}\nFAILED",
            )
        )
        with mock.patch.object(
            oracle, "KNOWN_RED_CELLS", (FIXTURE_CELL,)
        ):
            oracle.run_known_red_cells({}, runner=runner)
        runner.assert_called_once()

    def test_classifier_is_a_unit_tested_helper_not_a_manifest_instrument(
        self,
    ) -> None:
        previous_invoked = set(oracle.INVOKED_INSTRUMENTS)
        try:
            oracle.INVOKED_INSTRUMENTS.discard("classify_red_run")
            self.assertIsNone(
                oracle.classify_red_run(
                    FIXTURE_CELL,
                    101,
                    f"running 1 test\n{FIXTURE_CELL.fragment}\nFAILED",
                )
            )
            self.assertNotIn(
                "classify_red_run", oracle.INVOKED_INSTRUMENTS
            )
        finally:
            oracle.INVOKED_INSTRUMENTS.clear()
            oracle.INVOKED_INSTRUMENTS.update(previous_invoked)

    def test_all_cells_red_on_their_fragments_passes(self) -> None:
        def runner(_command, **_kwargs):
            name = _command[-1]
            cell = FIXTURE_CELL
            self.assertEqual(cell.name, name)
            return self._Completed(101, f"running 1 test\n{cell.fragment}\nFAILED")

        oracle.run_known_red_cells({}, runner=runner, cells=(FIXTURE_CELL,))

    def test_a_gone_green_cell_raises_the_shrink_protocol(self) -> None:
        def runner(_command, **_kwargs):
            return self._Completed(0, "running 1 test\nok")

        with self.assertRaisesRegex(oracle.OracleFailure, "Un-ignore"):
            oracle.run_known_red_cells({}, runner=runner, cells=(FIXTURE_CELL,))

    def test_a_wrong_reason_cell_raises_with_the_fragment(self) -> None:
        def runner(_command, **_kwargs):
            return self._Completed(101, "running 1 test\nsome unrelated panic")

        with self.assertRaisesRegex(oracle.OracleFailure, "NOT on its declared"):
            oracle.run_known_red_cells({}, runner=runner, cells=(FIXTURE_CELL,))

    def test_shipped_empty_ledger_top_level_path_passes(self) -> None:
        """The phase oracle remains executable after the final red cell retires.

        Run the shipped top-level path with its real structural, exclusion, and
        known-red receipt plumbing. Only the expensive Cargo suites are replaced,
        and their fake records the same success receipts as ``run_green_suites``.
        This is the transition chelis#1118 made when it emptied the ledger.
        """

        previous_invoked = set(oracle.INVOKED_INSTRUMENTS)
        previous_consumed = set(oracle.CONSUMED_INSTRUMENTS)

        def record_green_suites(_env) -> None:
            for label, _command in oracle.GREEN_SUITES:
                receipt = f"suite:{label}"
                oracle.INVOKED_INSTRUMENTS.add(receipt)
                oracle.CONSUMED_INSTRUMENTS.add(receipt)

        try:
            oracle.INVOKED_INSTRUMENTS.clear()
            oracle.CONSUMED_INSTRUMENTS.clear()
            with (
                mock.patch.object(oracle, "KNOWN_RED_CELLS", ()),
                mock.patch.object(
                    oracle,
                    "run_green_suites",
                    side_effect=record_green_suites,
                ),
            ):
                self.assertEqual(oracle.main(), 0)
        finally:
            oracle.INVOKED_INSTRUMENTS.clear()
            oracle.INVOKED_INSTRUMENTS.update(previous_invoked)
            oracle.CONSUMED_INSTRUMENTS.clear()
            oracle.CONSUMED_INSTRUMENTS.update(previous_consumed)


class RuleManifestTests(unittest.TestCase):
    def _doc(self) -> str:
        return (oracle.REPO_ROOT / oracle.DESIGN_DOC).read_text(encoding="utf-8")

    def test_the_shipped_doc_matches_the_manifest(self) -> None:
        self.assertEqual(oracle.b2_manifest_violations(self._doc()), [])

    def test_bounded_migration_rule_records_both_contract_handoffs(self) -> None:
        row = next(row for row in oracle.B2_RULE_INSTRUMENTS if row[0] == 1)
        _number, fragment, instruments = row
        rationale = " ".join(instruments)
        self.assertEqual(fragment, "bounded migration carve-outs")
        self.assertIn("v0.18.1", rationale)
        self.assertIn("chelis#1023", rationale)

    def test_the_shipped_doc_parses_non_trivially(self) -> None:
        items = oracle.b2_rule_items(self._doc())
        self.assertIsNotNone(items)
        assert items is not None
        self.assertGreaterEqual(len(items), 9)
        self.assertTrue(any("No third formatter" in title for _, title in items))

    def test_a_new_rule_without_a_manifest_row_is_a_violation(self) -> None:
        doc = self._doc().replace(
            "## B3. How to pick up a phase",
            "10. **A brand new rule.** With no instrument decision.\n\n"
            "## B3. How to pick up a phase",
            1,
        )
        violations = oracle.b2_manifest_violations(doc)
        self.assertTrue(
            any("B2.10" in v and "instrument decision" in v for v in violations),
            violations,
        )

    def test_a_retitled_rule_is_a_violation(self) -> None:
        doc = self._doc().replace("**No third formatter.**", "**No second formatter.**")
        violations = oracle.b2_manifest_violations(doc)
        self.assertTrue(
            any("manifest fragment" in v for v in violations), violations
        )

    def test_a_manifest_instrument_that_cannot_run_is_a_violation(self) -> None:
        original = oracle.B2_RULE_INSTRUMENTS
        oracle.B2_RULE_INSTRUMENTS = original + (
            (9, "Three-legged boundaries", ("nonexistent_check",)),
        )
        try:
            violations = oracle.b2_manifest_violations(self._doc())
        finally:
            oracle.B2_RULE_INSTRUMENTS = original
        self.assertTrue(
            any("nonexistent_check" in v and "cannot run" in v for v in violations),
            violations,
        )

    def test_a_missing_runtime_receipt_is_a_violation(self) -> None:
        """PR #962 round-2 M2's required regression: 'invoked' means a
        RUNTIME receipt, so an instrument referenced only from an
        `if False:` branch (which records nothing) fails at end of run -
        no source-text scan is consulted at all."""
        invoked = {
            entry
            for _n, _f, instruments in oracle.B2_RULE_INSTRUMENTS
            for entry in instruments
            if not entry.startswith("review-rule")
        }
        self.assertEqual(oracle.instrument_invocation_violations(invoked), [])
        invoked.discard("known_red_run_violations")
        violations = oracle.instrument_invocation_violations(invoked)
        self.assertTrue(
            any(
                "known_red_run_violations" in v
                and "runtime invocation receipt" in v
                for v in violations
            ),
            violations,
        )

    def test_the_instrument_decorator_writes_runtime_receipts(self) -> None:
        oracle.INVOKED_INSTRUMENTS.discard("classify_probe_outputs")
        oracle.classify_probe_outputs("C", [])
        self.assertIn("classify_probe_outputs", oracle.INVOKED_INSTRUMENTS)

    def test_an_invoked_but_unconsumed_result_is_a_violation(self) -> None:
        invoked = {
            entry
            for _n, _f, instruments in oracle.B2_RULE_INSTRUMENTS
            for entry in instruments
            if not entry.startswith("review-rule")
        }
        consumed = set(invoked)
        consumed.discard("b2_manifest_violations")
        violations = oracle.instrument_result_violations(invoked, consumed)
        self.assertTrue(
            any(
                "b2_manifest_violations" in v and "result was not consumed" in v
                for v in violations
            ),
            violations,
        )

    def test_suite_receipts_are_manifest_checkable(self) -> None:
        """Suite instruments are satisfied only by their runtime receipt
        (recorded by run_green_suites on success), never by existence."""
        invoked = {
            entry
            for _n, _f, instruments in oracle.B2_RULE_INSTRUMENTS
            for entry in instruments
            if not entry.startswith("review-rule") and not entry.startswith("suite:")
        }
        violations = oracle.instrument_invocation_violations(invoked)
        self.assertTrue(
            any("suite:" in v for v in violations),
            f"a suite instrument without its runtime receipt must fail; got {violations}",
        )

    def test_a_bare_review_rule_tag_is_a_violation(self) -> None:
        original = oracle.B2_RULE_INSTRUMENTS
        oracle.B2_RULE_INSTRUMENTS = original + (
            (9, "Three-legged boundaries", ("review-rule: trust me",)),
        )
        try:
            violations = oracle.b2_manifest_violations(self._doc())
        finally:
            oracle.B2_RULE_INSTRUMENTS = original
        self.assertTrue(
            any("justification" in v for v in violations), violations
        )

    def test_an_unlocatable_b2_section_is_reported_not_vacuous(self) -> None:
        violations = oracle.b2_manifest_violations("# Some other doc\n")
        self.assertTrue(
            any("could not be located" in v for v in violations), violations
        )


if __name__ == "__main__":
    unittest.main()
