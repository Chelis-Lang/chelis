"""Tests for the wire_to_canonical transcriber.

TWO layers:
  1. SYNTHETIC unit tests over hand-built wire blobs, positive + negative parity:
     every WireInferredType kind che_wire_type_to_type handles (prim / fn / ref /
     tensor / adt / tuple / unit) plus the un-representable cases (var / error /
     f64 / zero-arg fn) that must return None, and the effect injection / sorting.
  2. THE GOLDEN GUARD (test_golden_wire_blobs_reproduce_hull_string): feeds the
     FROZEN wire blobs Hull's export recorded (golden_wire.json) to
     wire_to_canonical and asserts it reproduces the Hull string. If the wire
     schema or normalization drifts, this fails LOUD rather than the runner
     silently mis-transcribing.

Run with the uv-managed interpreter:
    .venv/bin/python -m unittest -v tests.conformance.hull.test_wire_canonical
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wire_canonical import WireNormalizationError, wire_to_canonical  # noqa: E402

GOLDEN_WIRE_PATH = Path(__file__).resolve().parent / "golden_wire.json"

PRIM_F32 = {"kind": "prim", "name": "f32"}
PRIM_INT32 = {"kind": "prim", "name": "int32"}
PRIM_INT64 = {"kind": "prim", "name": "int64"}
PRIM_BOOL = {"kind": "prim", "name": "bool"}


class PrimTests(unittest.TestCase):
    def test_f32(self):
        self.assertEqual(wire_to_canonical(PRIM_F32, []), "(t-prim {} f32)")

    def test_int32_maps_to_int64(self):
        # che_wire_prim: int32 maps to TInt64 (Hull models the int family as int64).
        self.assertEqual(wire_to_canonical(PRIM_INT32, []), "(t-prim {} int64)")

    def test_int64(self):
        self.assertEqual(wire_to_canonical(PRIM_INT64, []), "(t-prim {} int64)")

    def test_bool(self):
        self.assertEqual(wire_to_canonical(PRIM_BOOL, []), "(t-prim {} bool)")

    def test_string(self):
        self.assertEqual(
            wire_to_canonical({"kind": "prim", "name": "string"}, []), "(t-prim {} string)"
        )

    def test_f64_is_none(self):
        # f64 has no Hull scalar type -> None (a sound mismatch).
        self.assertIsNone(wire_to_canonical({"kind": "prim", "name": "f64"}, []))


class FnTests(unittest.TestCase):
    def test_single_arg_fn(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        self.assertEqual(
            wire_to_canonical(fn, []), "(t-fn {} (t-prim {} f32) (t-prim {} f32))"
        )

    def test_multi_arg_fn_curries_right(self):
        fn = {"kind": "fn", "args": [PRIM_F32, PRIM_INT32], "ret": PRIM_F32}
        # (f32, int32) -> f32 curries to f32 -> (int64 -> f32). int32 -> int64.
        self.assertEqual(
            wire_to_canonical(fn, []),
            "(t-fn {} (t-prim {} f32) (t-fn {} (t-prim {} int64) (t-prim {} f32)))",
        )

    def test_zero_arg_fn_is_none(self):
        fn = {"kind": "fn", "args": [], "ret": PRIM_F32}
        self.assertIsNone(wire_to_canonical(fn, []))

    def test_fn_with_unrepresentable_ret_is_none(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": {"kind": "prim", "name": "f64"}}
        self.assertIsNone(wire_to_canonical(fn, []))

    def test_fn_with_unrepresentable_arg_is_none(self):
        fn = {"kind": "fn", "args": [{"kind": "var", "id": 0}], "ret": PRIM_F32}
        self.assertIsNone(wire_to_canonical(fn, []))


class EffectInjectionTests(unittest.TestCase):
    def test_single_effect_injected_into_outer_arrow(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        out = wire_to_canonical(fn, [{"kind": "io"}])
        self.assertEqual(
            out, "(t-fn {eff: (effects {} io)} (t-prim {} f32) (t-prim {} f32))"
        )

    def test_effects_sorted_for_set_ordering(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        # io + accum + random -> sorted as accum io random (set ordering).
        out = wire_to_canonical(
            fn, [{"kind": "io"}, {"kind": "accum"}, {"kind": "random"}]
        )
        self.assertEqual(
            out,
            "(t-fn {eff: (effects {} accum io random)} (t-prim {} f32) (t-prim {} f32))",
        )

    def test_effect_only_outermost_arrow_carries_it(self):
        # A 2-arg fn: only the OUTERMOST arrow gets the effect; the inner is {}.
        fn = {"kind": "fn", "args": [PRIM_F32, PRIM_F32], "ret": PRIM_F32}
        out = wire_to_canonical(fn, [{"kind": "io"}])
        self.assertEqual(
            out,
            "(t-fn {eff: (effects {} io)} (t-prim {} f32) (t-fn {} (t-prim {} f32) (t-prim {} f32)))",
        )

    def test_resource_effect_rendered_with_device(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        out = wire_to_canonical(fn, [{"kind": "resource", "device": "gpu:0"}])
        self.assertEqual(
            out,
            '(t-fn {eff: (effects {} (resource {} "gpu:0"))} (t-prim {} f32) (t-prim {} f32))',
        )

    def test_unknown_effect_kind_raises(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        with self.assertRaises(WireNormalizationError):
            wire_to_canonical(fn, [{"kind": "bogus_effect"}])

    def test_resource_missing_device_raises(self):
        fn = {"kind": "fn", "args": [PRIM_F32], "ret": PRIM_F32}
        with self.assertRaises(WireNormalizationError):
            wire_to_canonical(fn, [{"kind": "resource"}])


class RefEraseTests(unittest.TestCase):
    def test_ref_erases_to_inner(self):
        # ref{inner} (auto-borrow &T) erases to the inner type.
        ref = {"kind": "ref", "inner": PRIM_F32}
        self.assertEqual(wire_to_canonical(ref, []), "(t-prim {} f32)")

    def test_fn_with_ref_arg(self):
        fn = {"kind": "fn", "args": [{"kind": "ref", "inner": PRIM_F32}], "ret": PRIM_F32}
        self.assertEqual(
            wire_to_canonical(fn, []), "(t-fn {} (t-prim {} f32) (t-prim {} f32))"
        )


class TensorTests(unittest.TestCase):
    def test_tensor_named_dims(self):
        # The live wire precision is an object {kind:concrete, name}; lit dims
        # carry `size`.
        t = {
            "kind": "tensor",
            "precision": {"kind": "concrete", "name": "f32"},
            "dims": [{"kind": "name", "name": "batch"}, {"kind": "lit", "size": 4}],
        }
        self.assertEqual(
            wire_to_canonical(t, []),
            "(t-tensor {} (d-name {} batch) (d-lit {} 4) (t-prim {} f32))",
        )

    def test_tensor_scalar_zero_dims_double_space(self):
        # A zero-dim (scalar) tensor renders with a DOUBLE space, matching
        # unparse_type's join_space([]) == "" then the " " separator.
        t = {"kind": "tensor", "precision": {"kind": "concrete", "name": "f32"}, "dims": []}
        self.assertEqual(wire_to_canonical(t, []), "(t-tensor {}  (t-prim {} f32))")

    def test_tensor_int_precision_maps_to_int64(self):
        t = {
            "kind": "tensor",
            "precision": {"kind": "concrete", "name": "int32"},
            "dims": [{"kind": "lit", "size": 2}],
        }
        self.assertEqual(
            wire_to_canonical(t, []), "(t-tensor {} (d-lit {} 2) (t-prim {} int64))"
        )

    def test_tensor_bare_string_precision_fallback(self):
        # An older blob with a bare-string precision still transcribes.
        t = {"kind": "tensor", "precision": "f32", "dims": [{"kind": "lit", "value": 2}]}
        self.assertEqual(
            wire_to_canonical(t, []), "(t-tensor {} (d-lit {} 2) (t-prim {} f32))"
        )

    def test_tensor_bad_precision_is_none(self):
        t = {
            "kind": "tensor",
            "precision": {"kind": "concrete", "name": "f64"},
            "dims": [{"kind": "lit", "size": 2}],
        }
        self.assertIsNone(wire_to_canonical(t, []))


class TupleUnitAdtTests(unittest.TestCase):
    def test_tuple(self):
        t = {"kind": "tuple", "items": [PRIM_F32, PRIM_BOOL]}
        self.assertEqual(
            wire_to_canonical(t, []), "(t-tuple {} (t-prim {} f32) (t-prim {} bool))"
        )

    def test_unit_is_empty_tuple(self):
        self.assertEqual(wire_to_canonical({"kind": "unit"}, []), "(t-tuple {} )")

    def test_adt_nullary(self):
        t = {"kind": "adt", "name": "Color", "args": []}
        self.assertEqual(wire_to_canonical(t, []), "(t-adt {} Color)")

    def test_adt_with_args(self):
        t = {"kind": "adt", "name": "Box", "args": [PRIM_F32]}
        self.assertEqual(wire_to_canonical(t, []), "(t-adt {} Box (t-prim {} f32))")


class UnrepresentableTests(unittest.TestCase):
    def test_var_is_none(self):
        self.assertIsNone(wire_to_canonical({"kind": "var", "id": 3}, []))

    def test_error_is_none(self):
        self.assertIsNone(wire_to_canonical({"kind": "error"}, []))

    def test_non_dict_is_none(self):
        self.assertIsNone(wire_to_canonical("not a dict", []))
        self.assertIsNone(wire_to_canonical(None, []))


class GoldenWireTests(unittest.TestCase):
    def test_golden_wire_blobs_reproduce_hull_string(self):
        """THE DRIFT GUARD. For each frozen (live wire blob, expected Hull
        canonical string) recorded by Hull's export, assert wire_to_canonical
        reproduces the Hull string exactly. A wire-schema or normalization drift
        fails this LOUD."""
        if not GOLDEN_WIRE_PATH.is_file():
            self.skipTest("golden_wire.json not yet generated")
        golden = json.loads(GOLDEN_WIRE_PATH.read_text())
        self.assertGreater(len(golden), 0, "golden_wire.json must carry fixtures")
        for entry in golden:
            wire = entry["wire"]
            expected = entry["expected_canonical"]
            actual = wire_to_canonical(
                wire["display_signature_structured"], wire["effect_row"]
            )
            self.assertEqual(
                actual,
                expected,
                f"wire_to_canonical drift on {entry['id']}: "
                f"expected {expected!r}, got {actual!r}",
            )


if __name__ == "__main__":
    unittest.main()
