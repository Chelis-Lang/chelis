#!/usr/bin/env python3
"""Unit tests for the macOS Metal smoke's output-writeback contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "smoke_macos_metal.py"
SPEC = importlib.util.spec_from_file_location("smoke_macos_metal", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
smoke_macos_metal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke_macos_metal)


def guarded_writeback(prefix: str, index: int) -> str:
    """A minimal emitted Metal output writeback using the opaque ABI."""

    return f"""\
outputs[{index}] = chelis_alloc(1, shape_{index}, CHELIS_DTYPE_F32);
chelis_tensor_write *{prefix}_guard_{index} = chelis_tensor_begin_write(outputs[{index}]);
chelis_write_view {prefix}_view_{index} = chelis_tensor_write_view({prefix}_guard_{index});
chelis_metal_device_to_host({prefix}_view_{index}.data, d_t0, 32u);
chelis_tensor_end_write({prefix}_guard_{index});
"""


class GuardedOutputWritebackTests(unittest.TestCase):
    def test_store_and_root_writebacks_are_accepted(self) -> None:
        source = guarded_writeback("store", 0) + guarded_writeback("root", 2)
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source),
            {0, 2},
        )

    def test_incomplete_or_misdirected_writeback_is_rejected(self) -> None:
        complete = guarded_writeback("store", 0)
        mutations = {
            "missing allocation": complete.replace(
                "outputs[0] = chelis_alloc(1, shape_0, CHELIS_DTYPE_F32);\n", ""
            ),
            "missing begin": complete.replace(
                "chelis_tensor_write *store_guard_0 = "
                "chelis_tensor_begin_write(outputs[0]);\n",
                "",
            ),
            "missing view": complete.replace(
                "chelis_write_view store_view_0 = "
                "chelis_tensor_write_view(store_guard_0);\n",
                "",
            ),
            "missing transfer": complete.replace(
                "chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);\n", ""
            ),
            "missing end": complete.replace(
                "chelis_tensor_end_write(store_guard_0);\n", ""
            ),
            "end before transfer": complete.replace(
                "chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);\n"
                "chelis_tensor_end_write(store_guard_0);\n",
                "chelis_tensor_end_write(store_guard_0);\n"
                "chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);\n",
            ),
            "mismatched output": complete.replace(
                "chelis_tensor_begin_write(outputs[0])",
                "chelis_tensor_begin_write(outputs[1])",
            ),
            "old direct field": (
                "outputs[0] = chelis_alloc(1, shape_0, CHELIS_DTYPE_F32);\n"
                "chelis_metal_device_to_host(outputs[0]->data, d_t0, 32u);\n"
            ),
        }
        for label, source in mutations.items():
            with self.subTest(label=label):
                self.assertEqual(
                    smoke_macos_metal.guarded_output_writeback_indices(source),
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
