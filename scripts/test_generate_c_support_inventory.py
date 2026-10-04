"""The published C support inventory follows the source, not copied names."""
from __future__ import annotations

from pathlib import Path
import unittest

if __package__:
    from . import generate_c_support_inventory as inventory
else:
    import generate_c_support_inventory as inventory


class SourceRosterTests(unittest.TestCase):
    def test_reads_literal_rows_and_ignores_comment_decoys(self):
        source = '''/* const ITEMS: &[&str] = &["fake"]; */
        const ITEMS: &[&str] = &["one", // comment
        "two",];'''
        self.assertEqual(inventory.literal_roster(source, "ITEMS"), ("one", "two"))

    def test_missing_ambiguous_or_computed_rosters_fail(self):
        for source in [
            'const OTHER: &[&str] = &["one"];',
            'const ITEMS: &[&str] = &["one"]; const ITEMS: &[&str] = &["two"];',
            'const ITEMS: &[&str] = OTHER;',
            'const ITEMS: &[&str] = &[concat!("one", "two")];',
            'const ITEMS: &[&str] = &["one", "one"];',
        ]:
            with self.subTest(source=source), self.assertRaises(ValueError):
                inventory.literal_roster(source, "ITEMS")


class PublishedInventoryTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_snapshot_matches_source(self):
        expected = inventory.render(self.root)
        self.assertEqual((self.root / inventory.OUTPUT).read_text(), expected)

    def test_host_runtime_operations_have_no_build_gate(self):
        text = inventory.render(self.root)
        for name in ["tensor_scan", "process_run", "round_to"]:
            self.assertIn(f"`{name}`", text)
            self.assertNotIn(f"| `{name}` |", text)
        self.assertNotIn("Rejected", text)

    def test_test_assertions_are_compiled_host_operations(self):
        text = inventory.render(self.root)
        self.assertIn("[05-HOST-3]", text)
        for name in ["test_assert", "test_assert_eq", "test_assert_close_tensor", "test_assert_eq_tensor"]:
            self.assertIn(f"`{name}`", text)
            self.assertNotIn(f"| `{name}` |", text)

if __name__ == "__main__":
    unittest.main()
