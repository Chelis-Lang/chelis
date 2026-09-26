"""The public facade's exact execution v3 transport (spec/10 §3.2, spec/11)."""
from __future__ import annotations

import copy
import json
import unittest
from unittest import mock

import numpy as np
from ml_dtypes import bfloat16

from test_dtype_ingress import chelis


FLOATS = {"f16": (np.dtype(np.float16), 2), "bf16": (np.dtype(bfloat16), 2),
          "f32": (np.dtype(np.float32), 4), "f64": (np.dtype(np.float64), 8)}


def scalar(dtype, bits):
    return {"type": "scalar", "value": {"dtype": dtype, "bits": bits}}


def envelope(value):
    return {"schema_version": 4, "roots": [{"node_id": 0, "name": "x", "value": value}], "transcript": []}


class ExecutionWireTests(unittest.TestCase):
    def test_all_f16_and_bf16_stored_patterns_survive_scalar_and_tensor_transport(self):
        words = np.arange(65536, dtype=np.uint16)
        for tag in ("f16", "bf16"):
            dtype, width = FLOATS[tag]
            expected = [f"{int(word):04x}" for word in words]
            source = words.view(dtype)
            wire = chelis._tensor_value_payload(source)
            self.assertEqual(wire["shape"], [65536])
            self.assertEqual(set(wire["data"]), {"dtype", "bits"})
            self.assertEqual(wire["data"]["dtype"], tag)
            self.assertTrue(wire["data"]["bits"] == expected, f"{tag} stored bits changed")
            decoded = chelis._tensor_value(wire)
            self.assertTrue(all(type(value) is dtype.type for value in decoded.data))
            restored = np.asarray(decoded.data, dtype=dtype)
            np.testing.assert_array_equal(restored.view(np.uint16), words)
            self.assertTrue(chelis._tensor_value_payload(restored) == wire, f"{tag} re-encoded bits changed")
            # Scalar extraction is its own transport edge: never .item()/float().
            for word, bits in enumerate(expected):
                value = chelis._execution_value(scalar(tag, bits))
                self.assertIs(type(value), dtype.type)
                self.assertEqual(int(value.view(np.uint16)), word)

    def test_f32_f64_bits_survive_strides_endianness_and_scalar_shape(self):
        for tag, words in {
            "f32": [0, 0x80000000, 1, 0x007fffff, 0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00042, 0xff800042],
            "f64": [0, 0x8000000000000000, 1, 0x000fffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000,
                    0xfff0000000000000, 0x7ff8000000000042, 0xfff0000000000042],
        }.items():
            dtype, width = FLOATS[tag]
            for endian in ("<", ">"):
                unsigned = np.dtype(f"{endian}u{width}")
                source = np.array(words, dtype=unsigned).view(dtype.newbyteorder(endian))
                for array in (source, source[::-2], source[:1].reshape(())):
                    expected = [f"{int(word):0{width*2}x}" for word in array.view(unsigned).reshape(-1)]
                    wire = chelis._tensor_value_payload(array)
                    self.assertEqual(wire["data"], {"dtype": tag, "bits": expected})
                    self.assertEqual(wire["shape"], list(array.shape))
                    decoded = chelis._tensor_value(wire)
                    got = np.asarray(decoded.data, dtype=dtype).view(f"u{width}")
                    self.assertEqual(got.tolist(), [int(bits, 16) for bits in expected])
                    for bits in expected:
                        self.assertEqual(int(chelis._execution_value(scalar(tag, bits)).view(f"u{width}")), int(bits, 16))

    def test_bad_float_and_integer_carriers_fail_without_coercion(self):
        for value in [1.0, None, "0", "8000000", "800000000", "0X80000000", "7FC00000", " 80000000", "8000000g"]:
            with self.subTest(bits=value), self.assertRaises(ValueError):
                chelis._execution_value(scalar("f32", value))
        for carrier in [{"dtype": "f32", "value": 1.0}, {"dtype": "f32", "bits": "00000000", "value": 0},
                        {"dtype": "f128", "bits": "0"*32}, {"dtype": "bool", "value": True},
                        {"dtype": "int8", "value": 128}, {"dtype": "int8", "value": -129},
                        {"dtype": "int64", "value": True}, {"dtype": "int64", "value": 1.0},
                        {"dtype": "int64", "value": "1"}, {"dtype": "int64", "value": 1, "bits": "1"}]:
            with self.subTest(carrier=carrier), self.assertRaises(ValueError):
                chelis._execution_value({"type": "scalar", "value": carrier})
        for old in ("float16", "bfloat16", "float32", "float64", "int8", "int64"):
            with self.assertRaises(ValueError):
                chelis._execution_value({"type": old, "value": 1})

    def test_integer_extrema_and_booleans_stay_exact(self):
        for tag, width in (("int8", 8), ("int16", 16), ("int32", 32), ("int64", 64)):
            values = [-(1 << (width-1)), (1 << (width-1))-1]
            wire = {"shape": [2], "data": {"dtype": tag, "values": values}}
            self.assertEqual(chelis._tensor_value(wire).data, tuple(values))
            for value in values:
                decoded = chelis._execution_value({"type": "scalar", "value": {"dtype": tag, "value": value}})
                self.assertIs(type(decoded), np.dtype(tag).type)
                self.assertEqual(decoded, value)
        self.assertEqual(chelis._tensor_value({"shape": [2], "data": {"dtype": "bool", "values": [False, True]}}).data, (False, True))
        with self.assertRaises(ValueError):
            chelis._execution_value({"type": "bool", "value": 1})

    def test_shape_and_payload_are_validated_before_storage_allocation(self):
        good = {"shape": [1], "data": {"dtype": "f32", "bits": ["00000000"]}}
        for shape in ([True], [1.0], [-1], [2**63], [2], [2**62, 8]):
            bad = dict(good, shape=shape)
            with self.subTest(shape=shape), mock.patch.object(np, "array", side_effect=AssertionError("premature allocation")):
                with self.assertRaises(ValueError):
                    chelis._tensor_value(bad)
        for shape in ([0], [2**62, 2**62, 0], [0, 2**62, 2**62]):
            self.assertEqual(chelis._tensor_value({"shape": shape, "data": {"dtype": "f32", "bits": []}}).data, ())
        self.assertEqual(chelis._tensor_value({"shape": [], "data": {"dtype": "int64", "values": [2**53+1]}}).data, (2**53+1,))
        for data in ({"dtype": "f32", "values": [1.0]}, {"dtype": "f32", "bits": ["00000000"], "extra": 1},
                     {"dtype": "bool", "values": [1]}, {"dtype": "int8", "values": [128]}):
            with self.assertRaises(ValueError):
                chelis._tensor_value(dict(good, data=data))

    def test_facade_eval_checks_exact_stamp_before_visiting_values(self):
        for version in (None, 1, 2, 3, 5, True, 4.0, "4"):
            payload = envelope({"invalid": "must not decode"})
            if version is None:
                del payload["schema_version"]
            else:
                payload["schema_version"] = version
            with mock.patch.object(chelis._native, "eval_json", return_value=json.dumps(payload), create=True), \
                 mock.patch.object(chelis, "_execution_value", side_effect=AssertionError("values decoded before version")):
                with self.assertRaisesRegex(ValueError, "schema_version"):
                    chelis.eval("x = 1")
        with mock.patch.object(chelis._native, "eval_json", return_value=json.dumps(envelope(scalar("f32", "80000000"))), create=True):
            value = chelis.eval("x = 1").roots[0].value
            self.assertEqual(int(value.view(np.uint32)), 0x80000000)
        unnamed = envelope({"type": "unit"})
        del unnamed["roots"][0]["name"]
        self.assertIsNone(chelis._eval_result(unnamed).roots[0].name)
        unnamed["roots"][0]["node_id"] = -1
        with self.assertRaises(ValueError):
            chelis._eval_result(unnamed)

    def test_eval_accepts_the_native_envelopes_empty_transcript_omission(self):
        payload = envelope({"type": "unit"})
        del payload["transcript"]
        payload["manifest"] = {"target": "Eval", "entries": [], "requires_main": False}
        self.assertEqual(chelis._eval_result(payload).transcript, ())
        payload["transcript"] = None
        with self.assertRaises(ValueError):
            chelis._eval_result(payload)

    def test_nested_values_and_bad_aggregate_members(self):
        value = {"type": "adt", "ctor": "Example", "fields": [{"type": "tuple", "value": [
            {"type": "list", "value": [scalar("bf16", "7f82")]},
            {"type": "dict", "entries": [{"key": {"type": "string", "value": "id"},
                                           "value": {"type": "scalar", "value": {"dtype": "int64", "value": 2**53+1}}}]}]}]}
        decoded = chelis._execution_value(value)
        self.assertEqual(decoded.ctor, "Example")
        self.assertEqual(int(decoded.fields[0][0][0].view(np.uint16)), 0x7f82)
        self.assertEqual(decoded.fields[0][1], {"id": 2**53+1})
        for bad in ({"type": "string", "value": 1}, {"type": "unit", "value": None},
                    {"type": "list", "value": "abc"}, {"type": "tuple", "value": [], "extra": 1},
                    {"type": "adt", "ctor": 1, "fields": []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                chelis._execution_value(bad)

    def test_duplicate_json_fields_and_non_json_constants_are_rejected(self):
        for raw in ('{"schema_version":4,"schema_version":3,"roots":[],"transcript":[]}',
                    '{"schema_version":4,"roots":[],"transcript":[NaN]}'):
            with mock.patch.object(chelis._native, "eval_json", return_value=raw, create=True):
                with self.assertRaises(ValueError):
                    chelis.eval("x = 1")


if __name__ == "__main__":
    unittest.main()


class KeyExecutionValueTests(unittest.TestCase):
    """chelis#2413: keys cross the facade only as spec/10 §3.2 execution values."""

    ROWS = ["25ea33e61c10576f", "707124fbecd5f054", "823936153a565205"]

    def test_scalar_key_and_key_tensor_round_trip(self):
        key = chelis._execution_value({"type": "key", "bits": "0000000000000007"})
        self.assertEqual(key, chelis.Key("0000000000000007"))
        wire = {"shape": [3], "data": {"dtype": "key", "bits": self.ROWS}}
        tensor = chelis._execution_value({"type": "tensor", "value": wire})
        self.assertEqual(tensor.dtype, "key")
        self.assertEqual(tensor.data, tuple(chelis.Key(bits) for bits in self.ROWS))
        # Printed, then fed back: the binding payload is the printed storage.
        self.assertEqual(chelis._tensor_value_payload(tensor), wire)

    def test_key_decoders_are_strict(self):
        for bits in ("000000000000007", "00000000000000007", "00000000000000AB", 7):
            with self.assertRaises(ValueError):
                chelis._execution_value({"type": "key", "bits": bits})
            with self.assertRaises(ValueError):
                chelis._tensor_value({"shape": [1], "data": {"dtype": "key", "bits": [bits]}})
        for payload in (
            {"type": "key", "bits": "0000000000000007", "extra": 1},
            {"type": "key", "value": "0000000000000007"},
            {"type": "scalar", "value": {"dtype": "key", "bits": "0000000000000007"}},
        ):
            with self.assertRaises(ValueError):
                chelis._execution_value(payload)
        for storage in (
            {"dtype": "key", "values": [7]},
            {"dtype": "key", "bits": ["0000000000000007"], "values": [7]},
        ):
            with self.assertRaises(ValueError):
                chelis._tensor_value({"shape": [1], "data": storage})

    def test_a_scalar_key_binding_says_what_to_pass_instead(self):
        with self.assertRaises(chelis.ChelisError) as caught:
            chelis._tensor_value_payload(chelis.Key("0000000000000007"))
        message = str(caught.exception)
        self.assertIn("scalar `chelis.Key` is not a binding", message)
        self.assertIn("key_from_seed", message)
        self.assertIn("tensor[n, key]", message)
        self.assertNotIn("astype", message)

    def test_a_key_is_never_a_numpy_value(self):
        with self.assertRaises(ValueError):
            chelis.Key(7)
        with self.assertRaises(ChelisErrorOrValueError):
            chelis._tensor_value_payload(np.array([7], dtype=np.uint64))


ChelisErrorOrValueError = (ValueError, chelis.ChelisError)
