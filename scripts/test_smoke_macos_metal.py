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


def generated_entry(body: str, name: str = "simple_add") -> str:
    """Wrap statements in the exact generated tensor entry signature."""

    return f"""\
extern "C" void {name}(chelis_tensor **inputs, int n_in,
                        chelis_tensor **outputs, int n_out) {{
{body}}}
"""


class GuardedOutputWritebackTests(unittest.TestCase):
    def test_store_and_root_writebacks_are_accepted(self) -> None:
        source = generated_entry(
            guarded_writeback("store", 0) + guarded_writeback("root", 2)
        )
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            {0, 2},
        )

    def test_scoped_shape_allocation_is_accepted(self) -> None:
        source = generated_entry(
            "{ int64_t shape_store_0[1] = { (int64_t)8 };\n"
            "  outputs[0] = chelis_alloc(1, shape_store_0, CHELIS_DTYPE_F32); }\n"
            "chelis_tensor_write *store_guard_0 = "
            "chelis_tensor_begin_write(outputs[0]);\n"
            "chelis_write_view store_view_0 = "
            "chelis_tensor_write_view(store_guard_0);\n"
            "chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);\n"
            "chelis_tensor_end_write(store_guard_0);\n"
        )
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            {0},
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
            "intervening statement": complete.replace(
                "chelis_write_view store_view_0 =",
                "observe_unrelated_state();\nchelis_write_view store_view_0 =",
            ),
            "old direct field": (
                "outputs[0] = chelis_alloc(1, shape_0, CHELIS_DTYPE_F32);\n"
                "chelis_metal_device_to_host(outputs[0]->data, d_t0, 32u);\n"
            ),
        }
        for label, source in mutations.items():
            with self.subTest(label=label):
                self.assertEqual(
                    smoke_macos_metal.guarded_output_writeback_indices(
                        generated_entry(source), "simple_add"
                    ),
                    set(),
                )

    def test_operations_split_across_unrelated_functions_are_rejected(self) -> None:
        stitched = r'''
void alloc_only(chelis_tensor **outputs) {
    outputs[0] = chelis_alloc(1, shape_0, CHELIS_DTYPE_F32);
}
void begin_only(chelis_tensor **outputs) {
    chelis_tensor_write *store_guard_0 = chelis_tensor_begin_write(outputs[0]);
}
void view_only(chelis_tensor_write *store_guard_0) {
    chelis_write_view store_view_0 = chelis_tensor_write_view(store_guard_0);
}
void transfer_only(chelis_write_view store_view_0) {
    chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);
}
void end_only(chelis_tensor_write *store_guard_0) {
    chelis_tensor_end_write(store_guard_0);
}
'''
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(stitched, "simple_add"),
            set(),
        )

    def test_operations_split_across_generated_entries_are_rejected(self) -> None:
        complete = guarded_writeback("store", 0)
        split_at = complete.index("chelis_tensor_write")
        source = generated_entry(complete[:split_at], "allocation_only")
        source += generated_entry(complete[split_at:], "lease_only")
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "allocation_only"),
            set(),
        )

    def test_comment_only_writeback_is_rejected(self) -> None:
        source = generated_entry("/*\n" + guarded_writeback("store", 0) + "*/\n")
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            set(),
        )

    def test_commented_entry_signature_cannot_authorize_a_helper(self) -> None:
        forged = r'''
/* Documented generated signature:
extern "C" void documented(chelis_tensor **inputs, int n_in,
                           chelis_tensor **outputs, int n_out) {
*/
static void unrelated_helper(chelis_tensor **outputs) {
    outputs[0] = chelis_alloc(1, shape_0, CHELIS_DTYPE_F32);
    chelis_tensor_write *store_guard_0 = chelis_tensor_begin_write(outputs[0]);
    chelis_write_view store_view_0 = chelis_tensor_write_view(store_guard_0);
    chelis_metal_device_to_host(store_view_0.data, d_t0, 32u);
    chelis_tensor_end_write(store_guard_0);
}
'''
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(forged, "documented"),
            set(),
        )

    def test_preprocessor_disabled_writeback_is_rejected(self) -> None:
        writeback = guarded_writeback("store", 0)
        forged = r'''
#if 0
extern "C" void disabled_proof(chelis_tensor **inputs, int n_in,
                                chelis_tensor **outputs, int n_out) {
''' + writeback + r'''}
#endif
extern "C" void simple_add(chelis_tensor **inputs, int n_in,
                           chelis_tensor **outputs, int n_out) {
    (void)inputs; (void)n_in; (void)outputs; (void)n_out;
}
'''
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(forged, "simple_add"),
            set(),
        )

    def test_embedded_msl_conditionals_do_not_hide_host_writeback(self) -> None:
        source = r'''
static NSString *const kernel_src = @R"MSL(
#if __METAL_VERSION__ >= 320
kernel void conditional_kernel() {}
#endif
)MSL";
'''
        source += generated_entry(guarded_writeback("root", 0))
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            {0},
        )

    def test_writeback_for_an_unrelated_entry_is_rejected(self) -> None:
        writeback = guarded_writeback("store", 0)
        source = generated_entry(writeback, "unrelated_proof")
        source += generated_entry(
            "    (void)inputs; (void)n_in; (void)outputs; (void)n_out;\n",
            "simple_add",
        )
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            set(),
        )

    def test_multiple_entries_keep_writeback_indices_separate(self) -> None:
        source = generated_entry(guarded_writeback("root", 3), "unrelated")
        source += generated_entry(guarded_writeback("store", 0), "simple_add")
        self.assertEqual(
            smoke_macos_metal.guarded_output_writeback_indices(source, "simple_add"),
            {0},
        )


if __name__ == "__main__":
    unittest.main()
