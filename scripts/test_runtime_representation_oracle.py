#!/usr/bin/env python3

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import runtime_representation_oracle as oracle


class InventoryContractTests(unittest.TestCase):
    def test_inventory_covers_every_phase0_source_class(self) -> None:
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        self.assertEqual(
            {row.kind for row in rows},
            {
                "backend-element-spelling",
                "descriptor-field",
                "direct-data-access",
                "dtype-contract",
                "fixed-rank-metadata",
                "load-store-template",
                "narrow-metadata",
                "normalized-key-arithmetic",
                "raw-element-pointer",
                "width-arithmetic",
            },
        )

    def test_inventory_identities_do_not_use_mutable_line_numbers(self) -> None:
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        self.assertTrue(rows)
        for row in rows:
            self.assertNotRegex(row.identity, r":\d+(?::|$)")
            self.assertIn(row.path, row.identity)
            self.assertIn(row.owner, row.identity)

    def test_every_row_has_one_deletion_phase(self) -> None:
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        self.assertTrue(all(row.deletion_phase in {1, 2, 3, 4} for row in rows))

    def test_descriptor_detection_uses_field_shape_not_a_name_allowlist(self) -> None:
        source = """
struct NewlyHandwrittenMirror {
    data: *mut u8,
    shape: *const i64,
    strides: *const i64,
    size: i64,
    rank: i32,
    dtype: u8,
}
typedef struct { void *data; const int64_t *shape; const int64_t *strides;
    int64_t size; int32_t rank; uint8_t dtype; } second_c_abi_list;
"""
        self.assertEqual(
            oracle.descriptor_owner_names(source),
            {"NewlyHandwrittenMirror", "second_c_abi_list"},
        )

    def test_committed_debt_is_exact_and_shrink_only(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        oracle.validate_baseline(baseline, rows)

    def test_foundation_digest_rejects_an_edited_row(self) -> None:
        baseline = oracle.load_baseline()
        baseline["foundation_rows"][0]["owner"] = "forged_owner"
        with self.assertRaisesRegex(oracle.OracleFailure, "foundation digest"):
            oracle.validate_baseline(baseline, oracle.inventory_rows(oracle.REPO_ROOT))

    def test_active_debt_cannot_add_an_identity(self) -> None:
        baseline = oracle.load_baseline()
        baseline["active_debt"].append("forged:new-row")
        with self.assertRaisesRegex(oracle.OracleFailure, "outside the frozen foundation"):
            oracle.validate_baseline(baseline, oracle.inventory_rows(oracle.REPO_ROOT))

    def test_active_debt_cannot_retain_a_stale_identity(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        with mock.patch.object(oracle, "inventory_rows", return_value=rows[1:]):
            with self.assertRaisesRegex(oracle.OracleFailure, "stale active debt"):
                oracle.validate_phase0_inventory()


class MutationContractTests(unittest.TestCase):
    def test_direct_access_mutation_creates_one_unclassified_identity(self) -> None:
        baseline = oracle.load_baseline()
        with oracle.temporary_mutation(
            oracle.REPO_ROOT / oracle.DIRECT_ACCESS_MUTATION_SOURCE,
            oracle.mutate_direct_data_access,
        ):
            rows = oracle.inventory_rows(oracle.REPO_ROOT)
            with self.assertRaisesRegex(oracle.OracleFailure, "unclassified inventory hit"):
                oracle.validate_baseline(baseline, rows)

    def test_incomplete_dtype_mutation_creates_one_unclassified_identity(self) -> None:
        baseline = oracle.load_baseline()
        with oracle.temporary_mutation(
            oracle.REPO_ROOT / oracle.DTYPE_MUTATION_SOURCE,
            oracle.mutate_incomplete_dtype,
        ):
            rows = oracle.inventory_rows(oracle.REPO_ROOT)
            with self.assertRaisesRegex(oracle.OracleFailure, "unclassified inventory hit"):
                oracle.validate_baseline(baseline, rows)

    def test_temporary_mutation_restores_original_bytes_after_failure(self) -> None:
        original = (oracle.REPO_ROOT / oracle.DIRECT_ACCESS_MUTATION_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "owner.rs"
            path.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_mutation(path, oracle.mutate_direct_data_access):
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), original)


class PhaseZeroManifestTests(unittest.TestCase):
    def test_release_reproducers_and_landed_receipts_are_named(self) -> None:
        names = [leg.name for leg in oracle.phase0_legs()]
        self.assertEqual(
            names,
            [
                "capacity collision release reproducer",
                "count byte zero and foreign metadata release reproducers",
                "landed representation receipts",
            ],
        )
        for leg in oracle.phase0_legs():
            self.assertIn("--release", leg.argv, leg.name)

    def test_hardware_manifest_cannot_misreport_ignored_tests_as_executed(self) -> None:
        manifest = oracle.hardware_probe_manifest()
        self.assertEqual({row["lane"] for row in manifest}, {"hip", "metal"})
        for row in manifest:
            self.assertEqual(row["status"], "manual-required")
            self.assertIn("--ignored", row["command"])
            self.assertIn("--test-threads=1", row["command"])

    def test_phase2_phase3_host_descriptor_ownership_is_unambiguous(self) -> None:
        oracle.validate_design_sequencing()

    def test_phase_index_names_the_authoritative_continuous_command(self) -> None:
        phase_index = (oracle.REPO_ROOT / "docs/phase_oracles.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "scripts/runtime_representation_oracle.py --phase 0",
            phase_index,
        )
        self.assertIn("`spec/design/runtime_representation.md`", phase_index)

    @mock.patch.object(oracle, "_run_leg")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = oracle.OracleFailure("capacity collision failed with exit 9")
        with self.assertRaisesRegex(oracle.OracleFailure, "capacity collision"):
            oracle.run_phase0(run_mutations=False)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
