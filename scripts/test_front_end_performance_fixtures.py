#!/usr/bin/env python3
"""Tests for the chelis#1205 nested/flat performance corpus."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).with_name("front_end_performance_fixtures.py")
    spec = importlib.util.spec_from_file_location("front_end_performance_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fixtures = load_module()


class RenderTests(unittest.TestCase):
    def test_exact_small_nested_and_flat_snapshots(self) -> None:
        generated = fixtures.build_fixtures([2], width=8)
        by_name = {fixture.filename: fixture.source for fixture in generated}
        self.assertEqual(
            by_name["nested_n2.ch"],
            """module FrontEndPerformance.NestedN2
def bc(c: f32) -> tensor[8, f32] = reshape(insert(to_tensor([c]), 0, 8i64), [8i64])
def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = if gte(i, 5i64) then s else st(mul(add(mul(add(s, bc(cast(1.0, f32))), bc(cast(0.5, f32))), bc(cast(1.0, f32))), bc(cast(0.5, f32))), add(i, 1i64))
r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)
""",
        )
        self.assertEqual(
            by_name["flat_n2.ch"],
            """module FrontEndPerformance.FlatN2
def bc(c: f32) -> tensor[8, f32] = reshape(insert(to_tensor([c]), 0, 8i64), [8i64])
def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = if gte(i, 5i64) then s else {
  t0 = mul(add(s, bc(cast(1.0, f32))), bc(cast(0.5, f32)))
  t1 = mul(add(t0, bc(cast(1.0, f32))), bc(cast(0.5, f32)))
  st(t1, add(i, 1i64))
}
r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)
""",
        )

    def test_stable_order_and_exact_operation_counts(self) -> None:
        generated = fixtures.build_fixtures([20, 40], width=20_000)
        self.assertEqual(
            [fixture.filename for fixture in generated],
            ["nested_n20.ch", "flat_n20.ch", "nested_n40.ch", "flat_n40.ch"],
        )
        for fixture, expected in zip(generated, [20, 20, 40, 40], strict=True):
            self.assertEqual(fixture.source.count("mul(add("), expected)
            self.assertIn("tensor[20000, f32]", fixture.source)


class ValidationTests(unittest.TestCase):
    def test_rejects_empty_duplicate_nonpositive_counts_and_width(self) -> None:
        invalid = (
            ([], 8, "at least one"),
            ([0], 8, "positive"),
            ([2, 2], 8, "unique"),
            ([2], 0, "width must be positive"),
        )
        for counts, width, message in invalid:
            with self.subTest(message=message), self.assertRaisesRegex(
                fixtures.FixtureError, message
            ):
                fixtures.build_fixtures(counts, width=width)

    def test_refuses_to_overwrite_existing_fixture(self) -> None:
        generated = fixtures.build_fixtures([1], width=8)
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw)
            existing = output / generated[0].filename
            existing.write_text("keep\n", encoding="utf-8")
            with self.assertRaisesRegex(fixtures.FixtureError, "refusing to overwrite"):
                fixtures.write_fixtures(output, generated)
            self.assertEqual(existing.read_text(encoding="utf-8"), "keep\n")
            self.assertFalse((output / generated[1].filename).exists())


if __name__ == "__main__":
    unittest.main()
