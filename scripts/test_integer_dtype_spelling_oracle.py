"""Unit contracts for the chelis#1592 integer dtype spelling oracle."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import integer_dtype_spelling_oracle as oracle  # noqa: E402


class CorpusScanTests(unittest.TestCase):
    def test_retired_spelling_is_rejected_in_language_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.ch"
            source.write_text("def legacy(x: int64) -> int64 = x\n", encoding="utf-8")
            self.assertEqual(
                oracle.retired_spelling_hits([source]),
                [(source, 1, "int64"), (source, 1, "int64")],
            )

    def test_identifiers_containing_the_text_are_not_dtype_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "names.ch"
            source.write_text(
                "int64_parser = 1i64\nlabel = \"int64\"\n",
                encoding="utf-8",
            )
            self.assertEqual(oracle.retired_spelling_hits([source]), [])

    def test_surf_and_deep_line_comments_are_not_dtype_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "comments.ch"
            source.write_text(
                "-- retired int64 spelling\nvalue = 1i64 // old int64 note\n",
                encoding="utf-8",
            )
            self.assertEqual(oracle.retired_spelling_hits([source]), [])

    def test_nested_surf_block_comments_are_not_dtype_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "comments.ch"
            source.write_text(
                "{- retired int64 {- nested int32 -}\n"
                "   external int16 vocabulary -}\n"
                "value: i64 = 1i64\n",
                encoding="utf-8",
            )
            self.assertEqual(oracle.retired_spelling_hits([source]), [])

    def test_surf_code_after_a_block_comment_is_still_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.ch"
            source.write_text(
                "{- external int64 vocabulary -} value: int32 = 1i32\n",
                encoding="utf-8",
            )
            self.assertEqual(
                oracle.retired_spelling_hits([source]),
                [(source, 1, "int32")],
            )

    def test_deep_semicolon_comments_are_not_dtype_spellings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "comments.dp"
            source.write_text(
                "; retired int64 spelling\n"
                "(def {} x (lit {type: (t-prim {} i64)} 1)) ; old int32 note\n",
                encoding="utf-8",
            )
            self.assertEqual(oracle.retired_spelling_hits([source]), [])

    def test_deep_code_after_semicolon_comment_line_is_still_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "legacy.dp"
            source.write_text(
                "; external int64 vocabulary\n"
                "(def {} x (lit {type: (t-prim {} int16)} 1))\n",
                encoding="utf-8",
            )
            self.assertEqual(
                oracle.retired_spelling_hits([source]),
                [(source, 2, "int16")],
            )


class ContractTests(unittest.TestCase):
    def test_repository_boundary_contract_is_current(self) -> None:
        self.assertEqual(oracle.boundary_contract_errors(), [])

    def test_test_plan_covers_both_ingresses_migration_and_wire_compatibility(self) -> None:
        plan = "\n".join(" ".join(command) for command in oracle.TEST_COMMANDS)
        for required in (
            "issue_1592_integer_dtype_spelling",
            "issue_1587_short_integer_alias",
            "issue_1948_same_shape_result_claim",
            "issue_1948_same_shape_result_claim_sources",
            "issue_1537_ingress_pass_set_parity",
            "issue_1853_build_check_diagnostics",
            "unresolved_operand_census",
            "tests.conformance.hull.test_corpus_integrity",
            "tests.conformance.hull.test_wire_canonical",
            "execution_wire_v3",
            "wire_dag_vocabulary",
            "cache_wire_compatibility",
            "chelis-tide --test api",
            "chelis-prove --lib",
            "exact_tagged_c_header",
            "exact_tagged_c_abi",
            "runtime_dtype_generated_header",
        ):
            self.assertIn(required, plan)

    def test_boundary_contract_has_no_mutable_disposition_allowlist(self) -> None:
        source = Path(oracle.__file__).read_text(encoding="utf-8")
        self.assertNotIn("ALLOWED_RETIRED", source)
        self.assertNotIn("retired_allowlist", source)


if __name__ == "__main__":
    unittest.main()
