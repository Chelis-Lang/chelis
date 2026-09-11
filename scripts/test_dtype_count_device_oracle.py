#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import dtype_count_device_oracle as oracle


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_structural_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs()],
            [
                "dedicated HIP Count kernel",
                "dedicated Metal Count kernel",
                "compiler API device Count artifacts",
                "CLI device Count artifacts and example parity",
            ],
        )

    def test_manifest_covers_direct_host_and_example_entry_paths(self) -> None:
        joined = "\n".join(
            oracle.command_text(leg.argv) for leg in oracle.oracle_legs()
        )
        for required in (
            "chelis-backend-hip",
            "chelis-backend-metal",
            "hip_count_codegen",
            "metal_count_codegen",
            "issue_1291_count_device_helpers_api",
            "issue_1291_count_device_helpers_cli",
            "issue_1291_count_device_entry",
            "parity_count_bool_device_entry_library_only",
            "parity_corpus_is_complete",
        ):
            self.assertIn(required, joined)


class MutationTests(unittest.TestCase):
    def test_hip_dispatch_mutation_aliases_count_to_sum(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HIP_EMIT_SOURCE).read_text(
            encoding="utf-8"
        )
        mutated = oracle.mutate_hip_kernel_dispatch(source)
        self.assertNotEqual(mutated, source)
        self.assertIn('Some(format!("kernel_sum_{}", node.id.0))', mutated)

    def test_metal_dispatch_mutation_aliases_count_to_sum(self) -> None:
        source = (oracle.REPO_ROOT / oracle.METAL_EMIT_SOURCE).read_text(
            encoding="utf-8"
        )
        mutated = oracle.mutate_metal_kernel_dispatch(source)
        self.assertNotEqual(mutated, source)
        self.assertIn('format!("k_reduce_sum_{}", node.id.0)', mutated)

    def test_host_selection_mutation_restores_c_host_fallback(self) -> None:
        for relative in (oracle.HIP_LIB_SOURCE, oracle.METAL_LIB_SOURCE):
            with self.subTest(relative=relative):
                source = (oracle.REPO_ROOT / relative).read_text(encoding="utf-8")
                mutated = oracle.mutate_host_helper_selection(source)
                self.assertNotEqual(mutated, source)
                self.assertIn(".any(|_node| false)", mutated)

    def test_mutations_refuse_drifted_or_duplicated_anchors(self) -> None:
        cases = (
            (
                oracle.HIP_EMIT_SOURCE,
                oracle.mutate_hip_kernel_dispatch,
                'Some(format!("kernel_count_{}", node.id.0))',
            ),
            (
                oracle.METAL_EMIT_SOURCE,
                oracle.mutate_metal_kernel_dispatch,
                'format!("k_count_{}", node.id.0)',
            ),
            (
                oracle.HIP_LIB_SOURCE,
                oracle.mutate_host_helper_selection,
                ".any(|node| matches!(node.op, chelis_ir::dag::RiscOp::Count { .. }))",
            ),
        )
        for relative, mutate, anchor in cases:
            with self.subTest(relative=relative):
                source = (oracle.REPO_ROOT / relative).read_text(encoding="utf-8")
                with self.assertRaisesRegex(oracle.OracleFailure, "anchor drifted"):
                    mutate(source.replace(anchor, "anchor_removed", 1))
                with self.assertRaisesRegex(oracle.OracleFailure, "anchor drifted"):
                    mutate(source + "\n" + anchor + "\n")

    def test_temporary_mutation_restores_original_bytes_after_failure(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HIP_EMIT_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "emit.rs"
            path.write_bytes(source)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_mutation(
                    path, oracle.mutate_hip_kernel_dispatch
                ):
                    self.assertIn(
                        "kernel_sum_",
                        path.read_text(encoding="utf-8"),
                    )
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), source)


if __name__ == "__main__":
    unittest.main()
