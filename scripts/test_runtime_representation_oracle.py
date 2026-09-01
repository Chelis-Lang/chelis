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

    def test_inventory_covers_the_handwritten_hip_runtime_descriptor(self) -> None:
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        hip_runtime_rows = [
            row
            for row in rows
            if row.path
            == "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h"
        ]
        self.assertTrue(hip_runtime_rows)
        self.assertIn("descriptor-field", {row.kind for row in hip_runtime_rows})
        self.assertIn("fixed-rank-metadata", {row.kind for row in hip_runtime_rows})
        self.assertIn("narrow-metadata", {row.kind for row in hip_runtime_rows})

    def test_inventory_covers_the_metal_runtime_support_header(self) -> None:
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        metal_runtime_rows = [
            row
            for row in rows
            if row.path
            == "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h"
        ]
        self.assertTrue(metal_runtime_rows)
        self.assertIn("width-arithmetic", {row.kind for row in metal_runtime_rows})

    def test_every_tracked_backend_source_or_runtime_directory_is_scoped(self) -> None:
        completed = oracle.subprocess.run(
            ("git", "ls-files", "-z"),
            cwd=oracle.REPO_ROOT,
            check=True,
            capture_output=True,
        )
        expected = {
            raw.decode("utf-8")
            for raw in completed.stdout.split(b"\0")
            if raw
            and oracle.re.match(
                rb"crates/chelis-backend-[^/]+/(?:src|runtime|include)/",
                raw,
            )
            and oracle.PurePosixPath(raw.decode("utf-8")).suffix
            in oracle.SOURCE_SUFFIXES
        }
        observed = {
            path.as_posix() for path in oracle._tracked_source_paths(oracle.REPO_ROOT)
        }
        self.assertTrue(expected)
        self.assertEqual(expected - observed, set())

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

    def test_shared_c_parser_covers_the_reported_spelling_variants(self) -> None:
        source = """
extern void const_after(float const *payload);
extern void array_parameter(float payload[]);
static float const_after_cast(void *payload) {
    return *((float const *)payload);
}
extern void multiline(const volatile double * restrict payload);
static size_t qualified_sizeof(void) { return sizeof(const float); }
"""
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            relative = Path("crates/chelis-runtime/include/fixture.h")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding="utf-8")
            rows = oracle.c_surface_inventory_rows(root, (relative,))
        self.assertEqual(
            sum(kind == "raw-element-pointer" for kind, *_ in rows),
            4,
        )
        self.assertEqual(
            sum(kind == "width-arithmetic" for kind, *_ in rows),
            1,
        )

    def test_shared_c_parser_fails_closed_for_unknown_arithmetic_type(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            relative = Path("crates/chelis-runtime/include/fixture.h")
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_text(
                "extern void unknown(_Float16 *payload);\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                oracle.OracleFailure,
                "fail-closed C-surface parser rejected",
            ):
                oracle.c_surface_inventory_rows(root, (relative,))

    def test_committed_debt_is_exact_and_shrink_only(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(oracle.REPO_ROOT)
        oracle.validate_baseline(baseline, rows)

    def test_foundation_digest_rejects_an_edited_row(self) -> None:
        baseline = oracle.load_baseline()
        baseline["foundation_rows"][0]["identity"] += "-forged"
        with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
            oracle.validate_baseline(baseline, oracle.inventory_rows(oracle.REPO_ROOT))

    def test_freeze_digest_rejects_an_edited_coverage_manifest(self) -> None:
        baseline = oracle.load_baseline()
        altered = baseline["coverage_manifest"]["source_inventory"]["mutations"]
        altered.pop()
        with mock.patch.object(
            oracle,
            "coverage_manifest",
            return_value=baseline["coverage_manifest"],
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
                oracle.validate_baseline(
                    baseline,
                    oracle.inventory_rows(oracle.REPO_ROOT),
                )

    def test_c_family_suffix_set_covers_all_supported_source_forms(self) -> None:
        self.assertEqual(
            oracle.SOURCE_SUFFIXES,
            {
                ".c",
                ".cc",
                ".cpp",
                ".cu",
                ".cuh",
                ".cxx",
                ".h",
                ".h++",
                ".hh",
                ".hip",
                ".hpp",
                ".hxx",
                ".m",
                ".metal",
                ".mm",
                ".rs",
            },
        )

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
    def test_implementation_digest_is_independent_of_import_name(self) -> None:
        def local_mutation(source: str) -> str:
            return source + " controlled"

        original_module = local_mutation.__module__
        original_digest = oracle._mutation_implementation_sha256(local_mutation)
        try:
            local_mutation.__module__ = "alternate_runtime_oracle_import"
            renamed_digest = oracle._mutation_implementation_sha256(local_mutation)
        finally:
            local_mutation.__module__ = original_module

        self.assertEqual(original_digest, renamed_digest)

    def test_manifest_binds_exact_mutation_semantics_and_command(self) -> None:
        manifest = oracle.mutation_manifest(oracle.phase0_mutation_probes())
        self.assertTrue(manifest)
        for row in manifest:
            self.assertEqual(
                set(row),
                {
                    "command",
                    "expected_failure",
                    "expected_kind",
                    "implementation_sha256",
                    "path",
                    "witness_id",
                },
            )
            self.assertRegex(row["implementation_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                row["command"],
                "uv run --managed-python --python 3.11 --no-project python "
                "scripts/runtime_representation_oracle.py --phase 0",
            )
            self.assertEqual(
                set(row["expected_failure"]),
                {"code", "reason_prefix"},
            )

    def test_mutation_body_change_moves_the_freeze_digest(self) -> None:
        probes = oracle.phase0_mutation_probes()
        baseline_manifest = oracle.coverage_manifest(probes)
        replacement = oracle.MutationProbe(
            witness_id=probes[0].witness_id,
            expected_kind=probes[0].expected_kind,
            path=probes[0].path,
            mutate=oracle.mutate_free_standing_c_element_pointer,
            expected_failure=probes[0].expected_failure,
        )
        changed = (replacement, *probes[1:])
        changed_manifest = oracle.coverage_manifest(changed)
        self.assertNotEqual(baseline_manifest, changed_manifest)
        rows = [row.to_baseline_dict() for row in oracle.inventory_rows(oracle.REPO_ROOT)]
        self.assertNotEqual(
            oracle._freeze_digest(rows, baseline_manifest),
            oracle._freeze_digest(rows, changed_manifest),
        )

    def test_every_classifier_has_one_controlled_mutation(self) -> None:
        inventory_kinds = {
            row.kind for row in oracle.inventory_rows(oracle.REPO_ROOT)
        }
        probes = oracle.phase0_mutation_probes()
        self.assertEqual({probe.expected_kind for probe in probes}, inventory_kinds)
        self.assertEqual(
            len(probes),
            len({probe.witness_id for probe in probes}),
        )

    def test_c_pointer_sizeof_and_metal_descriptor_edges_have_mutations(self) -> None:
        probes = {
            (probe.expected_kind, probe.path.as_posix(), probe.mutate.__name__)
            for probe in oracle.phase0_mutation_probes()
        }
        self.assertIn(
            (
                "raw-element-pointer",
                "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
                "mutate_free_standing_c_element_pointer",
            ),
            probes,
        )
        self.assertIn(
            (
                "width-arithmetic",
                "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
                "mutate_c_sizeof_width_authority",
            ),
            probes,
        )
        self.assertIn(
            (
                "descriptor-field",
                "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
                "mutate_descriptor_field",
            ),
            probes,
        )

    def test_structural_c_parser_edges_have_controlled_mutations(self) -> None:
        names = {probe.mutate.__name__ for probe in oracle.phase0_mutation_probes()}
        self.assertTrue(
            {
                "mutate_c_const_after_element_pointer",
                "mutate_c_array_parameter",
                "mutate_c_const_after_pointer_cast",
                "mutate_c_qualified_sizeof_width_authority",
                "mutate_c_typedef_alias_pointer",
                "mutate_c_macro_alias_pointer",
                "mutate_c_unknown_arithmetic_pointer",
                "mutate_c_redefined_alias_pointer",
                "mutate_cxx_reference_and_template",
                "mutate_c_declaration_relocation",
                "mutate_c_atomic_element_pointer",
                "mutate_cxx_rvalue_reference",
                "mutate_c_complete_declarator_shapes",
                "mutate_c_pointer_return",
                "mutate_objc_pointer_return",
                "mutate_rust_dynamic_c_pointer",
                "mutate_rust_positional_and_macro_rules_pointer",
                "mutate_rust_split_c_pointer",
                "mutate_rust_unconstrained_c_source",
                "mutate_rust_stringify_pointer",
                "mutate_direct_data_access_after_test_module",
            }
            <= names
        )

    def test_every_classifier_mutation_is_rejected_for_its_intended_reason(self) -> None:
        baseline = oracle.load_baseline()
        for probe in oracle.phase0_mutation_probes():
            with self.subTest(kind=probe.expected_kind):
                path = oracle.REPO_ROOT / probe.path
                original = path.read_bytes()
                with oracle.temporary_mutation(path, probe.mutate):
                    try:
                        oracle.validate_baseline(
                            baseline,
                            oracle.inventory_rows(oracle.REPO_ROOT),
                        )
                    except oracle.OracleFailure as error:
                        message = str(error)
                        self.assertIn(probe.expected_error, message)
                        if probe.expected_error == "unclassified inventory hit":
                            self.assertIn(f"kind={probe.expected_kind}|", message)
                    else:
                        self.fail(f"{probe.mutate.__name__} was silently accepted")
                self.assertEqual(path.read_bytes(), original)

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
                "fail-closed C-surface parser contract",
                "capacity collision release reproducer",
                "count byte zero and foreign metadata release reproducers",
                "landed representation receipts",
            ],
        )
        parser_leg, *release_legs = oracle.phase0_legs()
        self.assertEqual(parser_leg.argv[-2:], ("--test", "c_surface"))
        for leg in release_legs:
            self.assertIn("--release", leg.argv, leg.name)
        capacity_leg = release_legs[0]
        self.assertIn("--test", capacity_leg.argv)
        self.assertIn("issue_888_capacity_collision", capacity_leg.argv)

    def test_hardware_manifest_cannot_misreport_ignored_tests_as_executed(self) -> None:
        manifest = oracle.hardware_probe_manifest()
        self.assertEqual({row["lane"] for row in manifest}, {"hip", "metal"})
        for row in manifest:
            self.assertEqual(row["status"], "manual-required")
            self.assertIn("--ignored", row["command"])
            self.assertIn("--test-threads=1", row["command"])

    def test_phase2_phase3_host_descriptor_ownership_is_unambiguous(self) -> None:
        oracle.validate_design_sequencing()

    def test_phase2_phase3_guard_rejects_the_old_host_ownership(self) -> None:
        current = oracle.DESIGN_PATH.read_text(encoding="utf-8")
        contradiction = (
            "\nPhase 2 installs the generated host descriptor and delivers C3 "
            "completely before Phase 3.\n"
        )
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "runtime_representation.md"
            path.write_text(current + contradiction, encoding="utf-8")
            with mock.patch.object(oracle, "DESIGN_PATH", path):
                with self.assertRaisesRegex(
                    oracle.OracleFailure, "contradictory Phase 2/3 ownership"
                ):
                    oracle.validate_design_sequencing()

    def test_design_status_records_phase0_as_implemented(self) -> None:
        source = oracle.DESIGN_PATH.read_text(encoding="utf-8")
        self.assertIn("Phase 0 is implemented", source)
        self.assertNotIn("no phase is implemented", source)

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
