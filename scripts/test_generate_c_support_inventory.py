"""The published C exclusions follow executable rosters, not copied names."""
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

    def test_new_exclusion_needs_reviewed_route_metadata(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            inventory.route_metadata("future_builtin")

    def test_compiled_host_operations_need_no_route(self):
        for name in ["process_run", "clock_wall_read", "round_to", "parse_csv", "csv_f64s"]:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "metadata"):
                inventory.route_metadata(name)


class PublishedInventoryTests(unittest.TestCase):
    root = Path(__file__).resolve().parents[1]

    def test_snapshot_matches_source(self):
        expected = inventory.render(self.root)
        self.assertEqual((self.root / inventory.OUTPUT).read_text(), expected)

    def test_each_source_row_has_one_route_and_both_lane_verdicts(self):
        rows = inventory.exclusions(self.root)
        names = [row[0] for row in rows]
        self.assertEqual(len(names), len(set(names)))
        for name, scope, alternative in rows:
            with self.subTest(name=name):
                self.assertTrue(scope)
                self.assertTrue(alternative)
                line = next(line for line in inventory.render(self.root).splitlines()
                            if line.startswith(f"| `{name}` |"))
                self.assertIn("builtin", line)
                self.assertIn("Available", line)
                self.assertIn("Rejected", line)

    def test_kernel_routing_roster_is_not_published_as_exclusions(self):
        names = {row[0] for row in inventory.exclusions(self.root)}
        for name in ["print", "string_concat", "process_run", "round_to", "parse_csv", "csv_int"]:
            self.assertNotIn(name, names)

    def test_test_assertions_are_live_scoped_implementation_gaps(self):
        text = inventory.render(self.root)
        self.assertIn("[05-HOST-3]", text)
        self.assertIn("implementation gap", text)
        for name in ["test_assert", "test_assert_eq", "test_assert_close_tensor", "test_assert_eq_tensor"]:
            self.assertIn(f"| `{name}` | builtin | Available | Rejected where live", text)


if __name__ == "__main__":
    unittest.main()
