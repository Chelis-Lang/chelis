"""The Python binding decodes the tagged diagnostic span carrier (chelis#1395).

`spec/04-type-system.md` [04-FIT-17] forbids the serializer inventing an extent
it did not measure, so the wire distinguishes

    {"span": "range", "offset": N, "len": M}   -- a measured range
    {"span": "point", "offset": N}             -- a coordinate, no extent

An earlier revision of the binding read `span["len"]` unconditionally, so every
`chelis.check()` whose program produced a check error with a coordinate raised
`KeyError: 'len'` instead of returning a result. That is the ordinary case --
an unbound variable, a type mismatch.

Nothing caught it. `bindings/python/tests` had two discoverable files, neither
touching the diagnostic decoder, and the only test calling `chelis.check` used
a clean program whose `errors` array is empty. This file exists so the decoder
is covered by `unittest discover -p 'test_*.py'`, which is what CI runs.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest


class _ChelisError(Exception):
    pass


def _load_bindings_module():
    python_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(python_root))

    native = types.ModuleType("chelis._native")
    native.ChelisError = _ChelisError
    sys.modules["chelis._native"] = native

    if "safetensors.numpy" not in sys.modules:
        safetensors = types.ModuleType("safetensors")
        safetensors_numpy = types.ModuleType("safetensors.numpy")
        safetensors_numpy.load_file = lambda _path: {}
        safetensors_numpy.save_file = lambda _tensors, _path: None
        safetensors.numpy = safetensors_numpy
        sys.modules["safetensors"] = safetensors
        sys.modules["safetensors.numpy"] = safetensors_numpy

    return importlib.import_module("chelis")


chelis = _load_bindings_module()


def _report(span: dict | None) -> dict:
    """A minimal check report carrying one diagnostic with `span`."""
    diagnostic = {
        "kind": "UnboundVariable",
        "message": "unbound variable: nope",
        "severity": 0.6,
        "suggestions": ["Check spelling of 'nope'"],
    }
    if span is not None:
        diagnostic["span"] = span
    return {
        "score": 0.8,
        "components": {"parse": 1.0, "structure": 1.0, "names": 0.0, "types": 1.0},
        "typed_nodes": 3,
        "untyped_nodes": 1,
        "total_nodes": 4,
        "unresolved_names": ["nope"],
        "errors": [diagnostic],
    }


class DiagnosticSpanTests(unittest.TestCase):
    def test_a_point_decodes_with_no_extent(self) -> None:
        """The case that raised `KeyError: 'len'`."""
        result = chelis._check_result(_report({"span": "point", "offset": 30}))
        self.assertEqual(result.errors[0].span, (30, None))

    def test_a_measured_range_keeps_its_extent(self) -> None:
        result = chelis._check_result(
            _report({"span": "range", "offset": 30, "len": 4})
        )
        self.assertEqual(result.errors[0].span, (30, 4))

    def test_a_measured_empty_range_is_not_an_absent_extent(self) -> None:
        """[04-FIT-17]: a consumer must be able to tell these apart."""
        measured = chelis._check_result(
            _report({"span": "range", "offset": 30, "len": 0})
        )
        absent = chelis._check_result(_report({"span": "point", "offset": 30}))
        self.assertEqual(measured.errors[0].span, (30, 0))
        self.assertEqual(absent.errors[0].span, (30, None))
        self.assertNotEqual(measured.errors[0].span, absent.errors[0].span)

    def test_no_location_at_all_decodes_as_none(self) -> None:
        result = chelis._check_result(_report(None))
        self.assertIsNone(result.errors[0].span)

    def test_an_untagged_or_unknown_shape_is_rejected(self) -> None:
        """Fails closed.

        Probing for a `len` key instead of reading the tag would decode a
        malformed `range` that lost its extent as though it were a point --
        turning a transport fault into a plausible value, which is the class
        of defect the tagged carrier exists to prevent.
        """
        for span in ({"offset": 30}, {"span": "interval", "offset": 30, "len": 4}):
            with self.subTest(span=span):
                with self.assertRaises(ValueError):
                    chelis._check_result(_report(span))


if __name__ == "__main__":
    unittest.main()
