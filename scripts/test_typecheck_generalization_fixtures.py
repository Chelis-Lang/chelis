#!/usr/bin/env python3
"""Tests for the chelis#1207 binding-count fixture generator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import typecheck_generalization_fixtures as fixtures


class RenderTests(unittest.TestCase):
    def test_exact_small_snapshots_cover_every_shape(self) -> None:
        generated = fixtures.build_fixtures([2], 2, [1, 2])
        by_name = {fixture.filename: fixture.source for fixture in generated}

        self.assertEqual(
            by_name["flat_n2.ch"],
            """module TypecheckGeneralization.FlatN2
def flat_0(x: f32) -> f32 = mul(x, cast(1.0, f32))
def flat_1(x: f32) -> f32 = mul(x, cast(2.0, f32))
out = flat_0(1.0)
""",
        )
        self.assertEqual(
            by_name["shallow_n2.ch"],
            """module TypecheckGeneralization.ShallowN2
def shared_base(x: f32) -> f32 = mul(x, cast(2.0, f32))
def shallow_0(x: f32) -> f32 = add(x, cast(1.0, f32)) |> shared_base
def shallow_1(x: f32) -> f32 = add(x, cast(2.0, f32)) |> shared_base
out = shallow_0(1.0)
""",
        )
        self.assertEqual(
            by_name["lets_n2.ch"],
            """module TypecheckGeneralization.LetsN2
def main(x: f32) -> f32 = {
  value_0 = add(x, cast(1.0, f32))
  value_1 = add(value_0, cast(2.0, f32))
  value_1
}
out = main(1.0)
""",
        )
        self.assertEqual(
            by_name["letsind_n2.ch"],
            """module TypecheckGeneralization.LetsindN2
def main(x: f32) -> f32 = {
  value_0 = add(x, cast(1.0, f32))
  value_1 = add(x, cast(2.0, f32))
  value_1
}
out = main(1.0)
""",
        )
        self.assertEqual(
            by_name["split_n2_k1.ch"],
            """module TypecheckGeneralization.SplitN2K1
def split_0(x: f32) -> f32 = {
  value_0 = add(x, cast(1.0, f32))
  value_1 = add(value_0, cast(2.0, f32))
  value_1
}
out = split_0(1.0)
""",
        )
        self.assertEqual(
            by_name["split_n2_k2.ch"],
            """module TypecheckGeneralization.SplitN2K2
def split_0(x: f32) -> f32 = {
  value_0 = add(x, cast(1.0, f32))
  value_0
}
def split_1(x: f32) -> f32 = {
  value_1 = add(x, cast(2.0, f32))
  value_1
}
out = split_0(1.0)
""",
        )

    def test_stable_file_order(self) -> None:
        generated = fixtures.build_fixtures([2, 3], 6, [1, 2, 3])
        self.assertEqual(
            [fixture.filename for fixture in generated],
            [
                "flat_n2.ch",
                "flat_n3.ch",
                "shallow_n2.ch",
                "shallow_n3.ch",
                "lets_n2.ch",
                "lets_n3.ch",
                "letsind_n2.ch",
                "letsind_n3.ch",
                "split_n6_k1.ch",
                "split_n6_k2.ch",
                "split_n6_k3.ch",
            ],
        )

    def test_requested_binding_counts_and_split_allocation(self) -> None:
        generated = fixtures.build_fixtures([4], 12, [3])
        by_name = {fixture.filename: fixture.source for fixture in generated}
        self.assertEqual(by_name["flat_n4.ch"].count("def flat_"), 4)
        self.assertEqual(by_name["shallow_n4.ch"].count("def shallow_"), 4)
        self.assertEqual(by_name["lets_n4.ch"].count(" = add("), 4)
        self.assertEqual(by_name["letsind_n4.ch"].count(" = add("), 4)
        split = by_name["split_n12_k3.ch"]
        self.assertEqual(split.count("def split_"), 3)
        self.assertEqual(split.count(" = add("), 12)
        for definition in range(3):
            start = definition * 4
            section = split.split(f"def split_{definition}", 1)[1]
            if definition < 2:
                section = section.split(f"def split_{definition + 1}", 1)[0]
            self.assertEqual(section.count(" = add("), 4)
            self.assertIn(f"value_{start}", section)


class ValidationTests(unittest.TestCase):
    def test_rejects_nonpositive_and_duplicate_values(self) -> None:
        invalid_calls = (
            ([0], 2, [1], "bindings values must be positive"),
            ([1, 1], 2, [1], "bindings values must be unique"),
            ([1], 0, [1], "split total must be positive"),
            ([1], 2, [0], "split counts values must be positive"),
            ([1], 2, [1, 1], "split counts values must be unique"),
        )
        for binding_counts, total, split_counts, message in invalid_calls:
            with self.subTest(message=message), self.assertRaisesRegex(
                fixtures.FixtureError, message
            ):
                fixtures.build_fixtures(binding_counts, total, split_counts)

    def test_rejects_indivisible_split(self) -> None:
        with self.assertRaisesRegex(fixtures.FixtureError, "not divisible"):
            fixtures.build_fixtures([2], 10, [3])

    def test_refuses_to_overwrite_any_target(self) -> None:
        generated = fixtures.build_fixtures([1], 1, [1])
        with tempfile.TemporaryDirectory() as raw:
            output_dir = Path(raw)
            existing = output_dir / generated[0].filename
            existing.write_text("keep me\n", encoding="utf-8")
            with self.assertRaisesRegex(fixtures.FixtureError, "refusing to overwrite"):
                fixtures.write_fixtures(output_dir, generated)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse((output_dir / generated[1].filename).exists())

    def test_writes_every_fixture_to_a_new_directory(self) -> None:
        generated = fixtures.build_fixtures([1], 2, [1, 2])
        with tempfile.TemporaryDirectory() as raw:
            output_dir = Path(raw) / "fixtures"
            paths = fixtures.write_fixtures(output_dir, generated)
            self.assertEqual(
                [path.name for path in paths],
                [fixture.filename for fixture in generated],
            )
            self.assertEqual(
                [path.read_text(encoding="utf-8") for path in paths],
                [fixture.source for fixture in generated],
            )


if __name__ == "__main__":
    unittest.main()
