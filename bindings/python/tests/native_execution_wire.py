"""Executed by the Rust test with the actual registered PyO3 module."""
import json

import numpy as np
from ml_dtypes import bfloat16

import chelis


def run(reference_json):
    references = json.loads(reference_json)
    results = []
    for wire in references:
        decoded = chelis._tensor_value(wire)
        dtype = bfloat16 if decoded.dtype == "bf16" else {
            "f16": np.float16, "f32": np.float32, "f64": np.float64,
        }.get(decoded.dtype, decoded.dtype)
        array = np.asarray(decoded.data, dtype=dtype).reshape(decoded.shape)
        assert chelis._tensor_value_payload(array) == wire, decoded.dtype
        dims = ", ".join(str(dim) for dim in decoded.shape)
        language_dtype = (
            decoded.dtype.replace("int", "i", 1)
            if decoded.dtype.startswith("int")
            else decoded.dtype
        )
        signature = f"{dims}, {language_dtype}" if dims else language_dtype
        result = chelis.eval(f"x = (x : tensor[{signature}])\n", {"x": array})
        root = next(root for root in result.roots if root.name == "x")
        assert isinstance(root.value, chelis.TensorValue)
        observed = np.asarray(root.value.data, dtype=dtype).reshape(root.value.shape)
        results.append(chelis._tensor_value_payload(observed))

    scalar_cases = [("int8", np.int8), ("int16", np.int16), ("int32", np.int32),
                    ("int64", np.int64), ("f16", np.float16), ("bf16", bfloat16),
                    ("f32", np.float32), ("f64", np.float64)]
    for dtype, python_type in scalar_cases:
        literal = f"7{dtype}" if dtype.startswith("int") else f"-0.0{dtype}"
        # Integer literal suffixes use the canonical i8/i16/i32/i64 spelling.
        literal = literal.replace("int", "i")
        result = chelis.eval(f"x = {literal}\n")
        value = next(root.value for root in result.roots if root.name == "x")
        assert type(value) is python_type, (dtype, type(value))
        if dtype.startswith("int"):
            assert value == 7
        else:
            width = np.dtype(python_type).itemsize
            assert int(value.view(f"u{width}")) == 1 << (width * 8 - 1)

    invalid = [
        {"shape": [1], "data": {"dtype": "f32", "values": [0.0]}},
        {"shape": [1], "data": {"dtype": "f32", "bits": ["7FC00001"]}},
        {"shape": [2], "data": {"dtype": "f32", "bits": ["00000000"]}},
        {"shape": [1], "data": {"dtype": "int8", "values": [128]}},
        {"shape": [-1], "data": {"dtype": "f32", "bits": []}},
    ]
    for wire in invalid:
        try:
            chelis._native.eval_json("x = (x : tensor[1, f32])\n", json.dumps({"x": wire}))
        except ValueError as error:
            assert "invalid bindings json" in str(error), str(error)
        else:
            raise AssertionError(f"native decoder accepted {wire}")
    return json.dumps(results)
