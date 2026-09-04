"""Unit contracts for the chelis#893 Phase 0 acceptance oracle.

These cover the parts of the oracle that are not proved by running it: that the
frozen source list still equals its roots, that the freeze digest actually binds
what it claims to, that every classifier has a controlled mutation, and that a
mutation cannot be weakened without moving the freeze.

The seam classification rules themselves are proved in the scanner's own suite
(`cargo nextest run -p chelis-repr-inventory --test inventory`), which is where
they belong: they are Rust and C parsing decisions, not Python ones.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import runtime_representation_oracle as oracle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


class SourceUniverseTests(unittest.TestCase):
    """The completeness claim is over a file list, so the list must be honest."""

    def test_the_frozen_source_list_equals_its_roots(self) -> None:
        self.assertEqual(
            sorted(oracle.INVENTORY_SOURCES),
            sorted(oracle._inventory_candidates(REPO_ROOT)),
            "INVENTORY_SOURCES has drifted from the tracked contents of its roots",
        )

    def test_every_registered_source_exists_exactly_once(self) -> None:
        self.assertEqual(
            len(oracle.INVENTORY_SOURCES),
            len(set(oracle.INVENTORY_SOURCES)),
            "duplicate entry in the frozen source list",
        )
        for relative in oracle.INVENTORY_SOURCES:
            self.assertTrue(
                (REPO_ROOT / relative).is_file(), f"registered source missing: {relative}"
            )

    def test_an_unregistered_file_under_a_root_fails_closed(self) -> None:
        with mock.patch.object(
            oracle,
            "_inventory_candidates",
            return_value=(*oracle.INVENTORY_SOURCES, "crates/chelis-ir/src/invented.rs"),
        ):
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle._assert_source_list_current(REPO_ROOT)
        self.assertEqual(caught.exception.code, oracle.SOURCE_LIST_FAILURE.code)
        self.assertIn("invented.rs", str(caught.exception))
        self.assertIn("INVENTORY_SOURCES", str(caught.exception))

    def test_a_departed_registered_file_fails_closed(self) -> None:
        with mock.patch.object(
            oracle,
            "_inventory_candidates",
            return_value=tuple(oracle.INVENTORY_SOURCES[1:]),
        ):
            with self.assertRaises(oracle.OracleFailure) as caught:
                oracle._assert_source_list_current(REPO_ROOT)
        self.assertEqual(caught.exception.code, oracle.SOURCE_LIST_FAILURE.code)

    def test_candidates_come_from_disk_not_the_git_index(self) -> None:
        # cargo compiles what is on disk, so an unstaged file is production
        # source. Enumerating the index instead would hide it.
        source = Path(oracle.__file__).read_text(encoding="utf-8")
        candidates = source[source.index("def _inventory_candidates") :]
        candidates = candidates[: candidates.index("\ndef ", 1)]
        self.assertIn("root.glob(pattern)", candidates)
        self.assertNotIn("ls-files", candidates)

    def test_the_universe_holds_the_handwritten_device_headers(self) -> None:
        for header in (
            "crates/chelis-backend-hip/runtime/chelis_hip_runtime.h",
            "crates/chelis-backend-metal/runtime/chelis_metal_runtime.h",
        ):
            self.assertIn(header, oracle.INVENTORY_SOURCES)


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = oracle.load_baseline()
        self.rows = oracle.inventory_rows(REPO_ROOT)

    def test_committed_debt_is_exact_and_shrink_only(self) -> None:
        oracle.validate_baseline(self.baseline, self.rows)

    def test_every_identity_is_kind_path_owner_only(self) -> None:
        # No line number, no source offset, and no hash of the exact bytes:
        # reformatting must not move the freeze.
        for row in self.baseline["foundation_rows"]:
            identity = str(row["identity"])
            self.assertRegex(identity, r"^kind=[^|]+\|path=[^|]+\|owner=.+$")
            self.assertNotIn("occurrence=", identity)
            self.assertNotIn("signature=", identity)

    def test_every_row_has_a_phase_the_design_maps(self) -> None:
        for row in self.baseline["foundation_rows"]:
            self.assertIn(row["deletion_phase"], (1, 2, 3, 4))
        for kind, phase in oracle.DELETION_PHASE_BY_KIND.items():
            self.assertIn(phase, (1, 2, 3, 4), kind)

    def test_the_sample_is_evidence_and_never_identity(self) -> None:
        # Samples live only in the active-debt list, which is outside the
        # digest, so rewording a line cannot move the freeze.
        for row in self.baseline["active_debt"]:
            self.assertEqual(set(row), {"identity", "sample"})
        for row in self.baseline["foundation_rows"]:
            self.assertEqual(set(row), {"identity", "deletion_phase"})

    def test_foundation_digest_rejects_an_edited_row(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        row = mutated["foundation_rows"][0]
        row["deletion_phase"] = 1 if row["deletion_phase"] == 4 else 4
        with self.assertRaisesRegex(oracle.OracleFailure, "freeze digest"):
            oracle.validate_baseline(mutated, self.rows)

    def test_freeze_digest_rejects_an_edited_coverage_manifest(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["coverage_manifest"]["acceptance"] = "ALWAYS PASS"
        with self.assertRaises(oracle.OracleFailure):
            oracle.validate_baseline(mutated, self.rows)

    def test_active_debt_cannot_add_an_identity(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        mutated["active_debt"].append(
            {"identity": "kind=direct-data-access|path=invented.rs|owner=f", "sample": ""}
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "outside the frozen foundation"):
            oracle.validate_baseline(mutated, self.rows)

    def test_active_debt_cannot_retain_a_stale_identity(self) -> None:
        mutated = json.loads(json.dumps(self.baseline))
        keep = mutated["foundation_rows"][0]["identity"]
        with self.assertRaisesRegex(oracle.OracleFailure, "stale active debt"):
            oracle.validate_baseline(
                mutated, tuple(row for row in self.rows if row.identity != keep)
            )

    def test_an_unclassified_hit_names_its_exact_identities(self) -> None:
        extra = oracle.InventoryRow(
            kind="direct-data-access",
            path="crates/chelis-ir/src/dag.rs",
            owner="runtime_representation_phase0_unit_probe",
            deletion_phase=3,
        )
        with self.assertRaises(oracle.OracleFailure) as caught:
            oracle.validate_baseline(self.baseline, (*self.rows, extra))
        self.assertEqual(caught.exception.code, oracle.UNCLASSIFIED_FAILURE.code)
        self.assertIn(extra.identity, caught.exception.details)


class MutationContractTests(unittest.TestCase):
    def test_every_seam_kind_in_the_ledger_has_a_mutation(self) -> None:
        baseline = oracle.load_baseline()
        kinds = {
            str(row["identity"]).split("|", 1)[0].removeprefix("kind=")
            for row in baseline["foundation_rows"]
        }
        covered = {probe.expected_kind for probe in oracle.phase0_mutation_probes()}
        self.assertEqual(
            kinds - covered,
            set(),
            "a seam class in the frozen ledger has no controlled mutation",
        )

    def test_witness_ids_are_unique(self) -> None:
        ids = [probe.witness_id for probe in oracle.phase0_mutation_probes()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_mutation_targets_a_registered_source(self) -> None:
        for probe in oracle.phase0_mutation_probes():
            if probe.expected_failure is oracle.SOURCE_LIST_FAILURE:
                # This one deliberately creates an UNregistered file.
                self.assertNotIn(probe.path.as_posix(), oracle.INVENTORY_SOURCES)
                continue
            self.assertIn(probe.path.as_posix(), oracle.INVENTORY_SOURCES)

    def test_manifest_binds_exact_mutation_semantics_and_command(self) -> None:
        for entry in oracle.mutation_manifest(oracle.phase0_mutation_probes()):
            self.assertEqual(
                set(entry),
                {
                    "witness_id",
                    "expected_kind",
                    "path",
                    "implementation_sha256",
                    "expected_failure",
                    "expected_owners",
                    "command",
                },
            )
            self.assertEqual(entry["command"], oracle.PHASE0_COMMAND)
            self.assertRegex(str(entry["implementation_sha256"]), r"^[0-9a-f]{64}$")

    def test_mutation_body_change_moves_the_freeze_digest(self) -> None:
        probes = oracle.phase0_mutation_probes()
        before = oracle.coverage_manifest(probes)

        def weakened(source: str) -> str:
            return source

        replaced = (
            oracle.MutationProbe(
                witness_id=probes[0].witness_id,
                expected_kind=probes[0].expected_kind,
                path=probes[0].path,
                mutate=weakened,
                expected_failure=probes[0].expected_failure,
            ),
            *probes[1:],
        )
        self.assertNotEqual(before, oracle.coverage_manifest(replaced))

    def test_implementation_digest_is_independent_of_import_name(self) -> None:
        digest = oracle._mutation_implementation_sha256(oracle.mutate_direct_data_access)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(
            digest,
            oracle._mutation_implementation_sha256(oracle.mutate_direct_data_access),
        )

    def test_temporary_mutation_restores_original_bytes_after_failure(self) -> None:
        path = REPO_ROOT / "crates/chelis-runtime/src/decimal_parse.rs"
        original = path.read_bytes()
        with self.assertRaises(RuntimeError):
            with oracle.temporary_mutation(path, oracle.mutate_direct_data_access):
                raise RuntimeError("simulated failure inside the mutation window")
        self.assertEqual(path.read_bytes(), original)

    def test_temporary_mutation_refuses_a_dirty_source(self) -> None:
        # The dirty check lives inside temporary_mutation, so no caller, unit
        # tests included, can plant a mutation over uncommitted work.
        path = REPO_ROOT / "crates/chelis-runtime/src/decimal_parse.rs"
        with mock.patch.object(
            oracle.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=(), returncode=0, stdout=" M x\n", stderr=""),
        ):
            with self.assertRaisesRegex(oracle.OracleFailure, "dirty detector source"):
                with oracle.temporary_mutation(path, lambda source: source):
                    pass

    def test_a_created_probe_file_is_removed_not_left_behind(self) -> None:
        path = REPO_ROOT / "crates/chelis-ir/src/runtime_representation_unit_probe.rs"
        self.assertFalse(path.exists())
        with oracle.temporary_mutation(path, lambda _: "// probe\n"):
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())


class ManifestTests(unittest.TestCase):
    def test_release_reproducers_and_landed_receipts_are_named(self) -> None:
        commands = [
            entry["command"] for entry in oracle.coverage_manifest()["release_reproducers"]
        ]
        self.assertTrue(
            any("issue_888_capacity_collision" in command for command in commands),
            commands,
        )
        self.assertTrue(any("exact_tagged_c_abi" in command for command in commands), commands)
        self.assertTrue(
            any("chelis-repr-inventory" in command for command in commands), commands
        )
        # The reproducers that hide in a debug profile must run in release.
        for command in commands:
            if "chelis-runtime" in command or "chelis-ir" in command:
                self.assertIn("--release", command)

    def test_hardware_manifest_cannot_misreport_ignored_tests_as_executed(self) -> None:
        for probe in oracle.hardware_probe_manifest():
            self.assertEqual(probe["status"], "manual-required")
            self.assertIn("--ignored", probe["command"])

    def test_the_manifest_records_the_closed_universe_rule(self) -> None:
        universe = oracle.coverage_manifest()["source_inventory"]["universe"]
        self.assertEqual(universe["registered_sources"], len(oracle.INVENTORY_SOURCES))
        self.assertIn("must equal", str(universe["closure_rule"]))

    def test_no_libclang_or_configuration_enumeration_remains(self) -> None:
        # The front end is a `clang` subprocess reading one fixed configuration
        # per header; a libclang binding or a preprocessor-configuration
        # product is the earlier, undischargeable design and must not return.
        source = Path(oracle.__file__).read_text(encoding="utf-8").lower()
        for banned in (
            "libclang",
            "clang.cindex",
            "preprocessor configuration",
            "cartesian",
        ):
            self.assertNotIn(banned, source)

    def test_phase_index_names_the_authoritative_continuous_command(self) -> None:
        index = (REPO_ROOT / "docs/phase_oracles.md").read_text(encoding="utf-8")
        self.assertIn("scripts/runtime_representation_oracle.py --phase 0", index)
        self.assertIn("RUNTIME REPRESENTATION PHASE 0: PASS", Path(oracle.__file__).read_text())

    @mock.patch("runtime_representation_oracle.subprocess.run")
    def test_runner_stops_at_the_first_failed_leg(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess(args=(), returncode=1)
        with mock.patch.object(oracle, "validate_phase0_inventory"):
            with self.assertRaisesRegex(oracle.OracleFailure, "failed with exit 1"):
                oracle.run_phase0(run_mutations=False)
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()


class RedTeamRegressionTests(unittest.TestCase):
    """Regressions from the first red-team round on the rebuilt architecture."""

    def test_a_subdirectory_of_a_root_is_a_candidate(self) -> None:
        # Single-level globs did not see a file cargo compiles from a
        # subdirectory, so the closure check could be evaded by placing a seam
        # one directory down.
        for root in oracle.INVENTORY_ROOTS:
            # A root is recursive, or it names a crate's one build script
            # exactly; a single-level directory glob is the evadable shape.
            self.assertTrue(
                "**" in root or root.endswith("/build.rs"),
                f"inventory root is neither recursive nor a build script: {root}",
            )

    def test_a_stale_active_debt_sample_fails(self) -> None:
        baseline = oracle.load_baseline()
        rows = oracle.inventory_rows(REPO_ROOT)
        mutated = json.loads(json.dumps(baseline))
        mutated["active_debt"][0]["sample"] = "THIS SAMPLE IS A LIE"
        with self.assertRaisesRegex(oracle.OracleFailure, "sample is stale"):
            oracle.validate_baseline(mutated, rows)

    def test_the_c_front_end_has_witnesses_for_every_mis_modelled_form(self) -> None:
        # Each of these is a declaration form that carried a seam past the
        # hand-written token walk in one red-team round. The compiler-backed
        # reader closes them by construction; the witnesses keep it that way.
        witnesses = {probe.witness_id for probe in oracle.phase0_mutation_probes()}
        for expected in (
            "phase0.mutate_c_public_element_pointer_export",
            "phase0.mutate_c_body_direct_data_access",
            "phase0.mutate_c_extern_element_data",
            "phase0.mutate_c_non_descriptor_struct_field",
            "phase0.mutate_c_tagged_struct_field",
            "phase0.mutate_c_union_field",
            "phase0.mutate_c_macro_typed_carrier",
            "phase0.mutate_c_multi_declarator_data",
            "phase0.mutate_c_enum_width",
            "phase0.mutate_objc_element_pointer_parameter",
            "phase0.mutate_unknown_c_arithmetic_spelling",
            "phase0.mutate_c_carrier_in_simd_arm",
            "phase0.mutate_c_undeclared_conditional",
            "phase0.mutate_objc_method_carrier",
            "phase0.mutate_c_unclassified_cast_spelling",
            "phase0.mutate_c_pointer_to_element_array",
            "phase0.mutate_c_include_outside_universe",
            "phase0.mutate_c_include_in_dead_arm",
            "phase0.mutate_objc_block_parameter",
            "phase0.mutate_c_sizeof_in_array_bound",
            "phase0.mutate_rust_cast_turbofish",
            "phase0.mutate_c_int8_element_pointer",
            "phase0.mutate_c_elifdef_arm",
        ):
            self.assertIn(expected, witnesses)

    def test_a_multi_declarator_witness_demands_every_owner(self) -> None:
        probes = {probe.witness_id: probe for probe in oracle.phase0_mutation_probes()}
        pair = probes["phase0.mutate_c_multi_declarator_data"]
        self.assertEqual(len(pair.expected_owners), 2)
        manifest = {
            entry["witness_id"]: entry
            for entry in oracle.coverage_manifest()["source_inventory"]["mutations"]
        }
        self.assertEqual(
            manifest["phase0.mutate_c_multi_declarator_data"]["expected_owners"],
            list(pair.expected_owners),
        )

    def test_build_scripts_are_inside_the_universe(self) -> None:
        # A build script is compiled by cargo like any other source, so a root
        # that cannot see one is a closure hole.
        self.assertIn("crates/chelis-backend-c/build.rs", oracle.INVENTORY_SOURCES)
        self.assertTrue(
            any(root.endswith("build.rs") for root in oracle.INVENTORY_ROOTS),
            oracle.INVENTORY_ROOTS,
        )

    def test_the_closure_check_has_a_subdirectory_witness(self) -> None:
        paths = {probe.path.as_posix() for probe in oracle.phase0_mutation_probes()}
        self.assertTrue(
            any(path.count("/") > 3 and path.endswith("mod.rs") for path in paths),
            f"no subdirectory closure witness among {sorted(paths)}",
        )

    def test_the_docstring_source_counts_match_the_frozen_list(self) -> None:
        rust = sum(1 for path in oracle.INVENTORY_SOURCES if path.endswith(".rs"))
        headers = len(oracle.INVENTORY_SOURCES) - rust
        source = Path(oracle.__file__).read_text(encoding="utf-8")
        self.assertIn("Fifty-nine are Rust and seven are C or Objective-C headers", source)
        self.assertEqual((rust, headers), (59, 7))
