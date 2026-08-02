from __future__ import annotations

import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


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


class DtypeIngressTests(unittest.TestCase):
    def test_losslessly_widenable_unsigned_arrays_get_exact_wire_dtypes(self) -> None:
        cases = [
            (np.array([0, 255], dtype=np.uint8), "int16", [0, 255]),
            (np.array([0, 65535], dtype=np.uint16), "int32", [0, 65535]),
            (np.array([0, 2**32 - 1], dtype=np.uint32), "int64", [0, 2**32 - 1]),
        ]
        for array, dtype, values in cases:
            with self.subTest(source=str(array.dtype)):
                payload = chelis._tensor_value_payload(array)
                self.assertEqual(payload["data"], {"dtype": dtype, "values": values})

    def test_uint64_never_falls_through_an_f64_funnel(self) -> None:
        value = np.array([2**63 + 1], dtype=np.uint64)
        with self.assertRaises(chelis.ChelisError) as caught:
            chelis._tensor_value_payload(value)
        message = str(caught.exception)
        self.assertIn("uint64", message)
        self.assertIn("astype(np.int64)", message)
        self.assertIn("astype(np.float64)", message)

    def test_unmapped_float_width_is_rejected_loudly(self) -> None:
        value = np.array([1.0], dtype=np.longdouble)
        if value.dtype.itemsize == np.dtype(np.float64).itemsize:
            self.skipTest("this platform aliases longdouble to float64")
        with self.assertRaises(chelis.ChelisError) as caught:
            chelis._tensor_value_payload(value)
        self.assertIn(str(value.dtype), str(caught.exception))

    def test_numpy_array_reaches_dtype_rejection_without_dlpack(self) -> None:
        value = np.array([1.0 + 2.0j], dtype=np.complex128)
        with mock.patch.object(
            np,
            "from_dlpack",
            side_effect=AssertionError("NumPy arrays must bypass DLPack"),
        ):
            with self.assertRaises(chelis.ChelisError) as caught:
                chelis._tensor_value_payload(value)
        self.assertIn("complex128", str(caught.exception))

    def test_decoded_tensor_value_requires_an_explicit_dtype(self) -> None:
        with self.assertRaises(TypeError):
            chelis.TensorValue(shape=(1,), data=(1,))

    def test_execution_wire_accepts_every_exact_numeric_scalar_tag(self) -> None:
        cases = [
            ("int8", -8),
            ("int16", -16),
            ("int32", -32),
            ("int64", 2**53 + 1),
            ("float16", 1.5),
            ("bfloat16", 1.5),
            ("float32", 0.25),
            ("float64", 1e100),
        ]
        for wire_type, value in cases:
            with self.subTest(wire_type=wire_type):
                self.assertEqual(
                    chelis._execution_value({"type": wire_type, "value": value}),
                    value,
                )


if __name__ == "__main__":
    unittest.main()
