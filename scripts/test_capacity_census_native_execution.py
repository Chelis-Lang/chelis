"""Spec11/OP45 execution evidence must come from the fixed live native matrix.

These pure controls test evidence rejection, not native boundary execution.
The 37 actual Rust cases remain the separate executable acceptance matrix.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from capacity_census_graph import GraphError
import capacity_census_native_execution as execution


class MatrixContractTests(unittest.TestCase):
    def test_fixed_matrix_preserves_every_exact_spec_case_and_all_fixture_instances(self):
        actual = tuple((group.name, group.kind, name, count, group.fixture_kind)
                       for group in execution.GROUPS
                       for name, count in zip(group.selected, group.instances, strict=True))
        self.assertEqual(actual, EXPECTED_MATRIX)
        self.assertEqual(len(actual), 37)
        self.assertEqual(sum(row[3] for row in actual), 49)

    def test_public_collection_accepts_no_supplied_selection_or_execution(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(execution.collect_native_execution).parameters),
                         ("root", "target"))
        with self.assertRaises(TypeError):
            execution.CheckedNativeExecution(packet={"passed": 37})

    def test_worker_environment_is_explicit_without_mutating_parent_or_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = root / "runtime"
            group = root / "group"
            hostile = {
                "CHELIS_RUNTIME_DIR": "/foreign/runtime", "CHELIS_CC": "/foreign/compiler",
                "CHELIS_TEST_CC": "/foreign/test-compiler", "CHELIS_HIPCC": "/foreign/hipcc",
                "CHELIS_DEVICE_OWNER_TEST_WORKER": "1", "CHELIS_NATIVE_ENV_TEST_WORKER": "1",
                "CHELIS_NATIVE_EXECUTION_CAPTURE": "/foreign/capture", "PYTHONPATH": "/foreign/python",
        }
            with mock.patch.dict(os.environ, hostile):
                before = dict(os.environ)
                child = execution._worker_environment(runtime, group, execution.GROUPS[0])
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(child["CHELIS_RUNTIME_DIR"], str(runtime))
                self.assertEqual(child["CHELIS_NATIVE_EXECUTION_CAPTURE"], str(group))
                self.assertEqual(child["CHELIS_NATIVE_EXECUTION_SUITE"], execution.GROUPS[0].name)
                self.assertEqual(child["CARGO_BUILD_JOBS"], "1")
                for name in ("CHELIS_CC", "CHELIS_TEST_CC", "CHELIS_HIPCC",
                             "CHELIS_DEVICE_OWNER_TEST_WORKER", "CHELIS_NATIVE_ENV_TEST_WORKER"):
                    self.assertNotIn(name, child)
                self.assertNotEqual(child["PYTHONPATH"], hostile["PYTHONPATH"])


class RuntimeArtifactTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.target = self.root / "target/owned"
        self.target.mkdir(parents=True)
        self.source = self.root / "crates/chelis-runtime/src/lib.rs"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("// exact source identity\n")
        self.archive = self.target / "debug/libchelis_runtime.a"
        self.archive.parent.mkdir()
        self.archive.write_bytes(b"actual Cargo output stand-in for this pure selector test")
        self.row = {"reason": "compiler-artifact", "package_id": "path+file://runtime#chelis-runtime@0.18.6",
                    "target": {"name": "chelis_runtime", "kind": ["staticlib", "rlib"],
                               "crate_types": ["staticlib", "rlib"], "src_path": str(self.source)},
                    "profile": {"test": False}, "filenames": [str(self.archive)], "executable": None}

    def test_selects_only_the_actual_exact_runtime_staticlib_record(self):
        self.assertEqual(execution._runtime_artifact(self.root, self.target, [self.row]), self.archive)

    def test_missing_duplicate_test_or_foreign_runtime_artifact_is_rejected(self):
        variants = [[], [self.row, self.row], [dict(self.row, profile={"test": True})],
                    [dict(self.row, target={**self.row["target"], "name": "other"})],
                    [dict(self.row, target={**self.row["target"], "src_path": "/foreign/lib.rs"})],
                    [dict(self.row, filenames=[str(self.root / "foreign.a")])]]
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(GraphError):
                execution._runtime_artifact(self.root, self.target, rows)

    def test_symlink_cannot_redirect_runtime_selection(self):
        target = self.archive.with_name("actual.a")
        self.archive.rename(target)
        self.archive.symlink_to(target)
        with self.assertRaises(GraphError):
            execution._runtime_artifact(self.root, self.target, [self.row])


class RetainedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.group = execution.GROUPS[0]
        self.case = self.root / "fixtures" / self.group.selected[0] / "instance-1"
        self.case.mkdir(parents=True)
        self.runtime = b"current runtime bytes"
        self.runtime_digest = hashlib.sha256(self.runtime).hexdigest()
        self.files = {
            "program.ch": b"def main(x: tensor[1,f32]) -> tensor[1,f32] = copy(x)\n",
            "compiled/program.c": b"/* generated fixture source */",
            "compiled/chelis_runtime.h": b"/* current embedded header */",
            "compiled/program.so": b"actual library stand-in for pure record validation",
            "compiled/program.json": b'{"abi_version":2,"target":"c"}',
            "compiled/libchelis_runtime.a": self.runtime,
            }
        original = self.root / "deleted-original"
        self.files["loaded-model.json"] = json.dumps({
            "path": str(original / "program.so"), "root": str(original),
            "snapshot": "compiled", "library": "compiled/program.so",
            "files": [{"original": str(original / Path(name).name), "captured": name,
                       "sha256": hashlib.sha256(content).hexdigest()}
                      for name, content in sorted(self.files.items()) if name.startswith("compiled/")],
        }).encode()
        for name, content in self.files.items():
            path = self.case / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.packet = {"schema": 1, "suite": self.group.name, "test": self.group.selected[0],
                       "completed": True, "library": str(self.case / "compiled/program.so"),
                       "files": [{"path": name, "sha256": hashlib.sha256(content).hexdigest()}
                                 for name, content in sorted(self.files.items())]}

    def validate(self):
        return execution._validate_instance(self.case, self.group, self.group.selected[0],
                                            self.packet, self.runtime_digest)

    def test_complete_retained_generated_artifacts_bind_actual_model_and_runtime(self):
        self.assertTrue(self.validate())

    def test_missing_changed_or_extra_artifact_cannot_preserve_evidence(self):
        for name, content in self.files.items():
            with self.subTest(name=name):
                path = self.case / name
                path.unlink()
                with self.assertRaises(GraphError): self.validate()
                path.write_bytes(content + b"changed")
                with self.assertRaises(GraphError): self.validate()
                path.write_bytes(content)
        (self.case / "unrecorded.c").write_bytes(b"extra")
        with self.assertRaises(GraphError): self.validate()

    def test_staged_archive_must_match_current_cargo_bytes_even_if_its_record_is_rehashed(self):
        path = self.case / "compiled/libchelis_runtime.a"
        path.write_bytes(b"foreign runtime")
        for row in self.packet["files"]:
            if row["path"] == "compiled/libchelis_runtime.a":
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        model_path = self.case / "loaded-model.json"
        model = json.loads(model_path.read_text())
        for row in model["files"]:
            if row["captured"] == "compiled/libchelis_runtime.a":
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        model_path.write_text(json.dumps(model))
        for row in self.packet["files"]:
            if row["path"] == "loaded-model.json":
                row["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        with self.assertRaises(GraphError): self.validate()

    def test_one_fixture_cannot_stand_in_for_the_other_selected_cases(self):
        (self.case / "completion.json").write_text(json.dumps(self.packet))
        with self.assertRaises(GraphError):
            execution._validate_captures(self.root, self.group, self.runtime_digest,
                                         self.root / "actual-test-binary")

    def test_original_model_artifacts_may_disappear_without_invalidating_snapshot(self):
        self.assertFalse((self.root / "deleted-original").exists())
        self.assertTrue(self.validate())

    def test_wrong_owner_incomplete_duplicate_or_external_file_record_is_rejected(self):
        original = self.packet
        changes = [{"completed": False}, {"suite": "another"}, {"test": "another"},
                   {"library": "/foreign/program.so"},
                   {"files": original["files"] + [original["files"][0]]},
                   {"files": [{"path": "../escape", "sha256": "0" * 64}]}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(GraphError):
                self.packet = {**original, **change}
                self.validate()
        self.packet = original

    def test_same_named_library_outside_actual_model_path_is_rejected(self):
        path = self.case / "loaded-model.json"
        record = json.loads(path.read_text())
        record["path"] = str(self.root / "different/program.so")
        path.write_text(json.dumps(record))
        for row in self.packet["files"]:
            if row["path"] == "loaded-model.json":
                row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(GraphError): self.validate()

    def test_symlink_is_not_a_retained_artifact_even_when_hash_matches(self):
        path = self.case / "compiled/program.so"
        other = self.root / "library-copy"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
        with self.assertRaises(GraphError): self.validate()


EXPECTED_MATRIX = (
    ('native_tensor_boundary', 'test', 'rank_zero_is_one_element_and_rejects_rank_one_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'rank_one_roundtrips_and_rejects_rank_two_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'rank_eight_roundtrips_and_rejects_rank_nine_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'rank_nine_roundtrips_and_rejects_rank_ten_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'zero_extent_preserves_empty_shape_and_rejects_nonempty_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'empty_large_extent_shape_is_exact_without_payload_allocation', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'f64_copy_preserves_stored_bits_and_rejects_f32_input', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'dlpack_cpu_accepts_none_stream_and_rejects_integer_streams', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'dlpack_keywords_are_keyword_only_and_validate_version_and_device_shapes', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'dlpack_copy_false_shares_storage_and_copy_true_never_returns_legacy_capsule', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'dlpack_consumer_keeps_storage_alive_after_tensor_and_model_drop', 1, 'generated-c-current-runtime'),
    ('native_tensor_boundary', 'test', 'callable_v2_writer_and_loader_reject_v1_before_opening_a_library', 1, 'generated-c-current-runtime'),
    ('native_output_validation', 'test', 'actual_single_output_is_validated_before_native_tensor_construction', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'actual_named_outputs_preserve_dictionary_behavior_and_release_every_owner', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'invalid_actual_descriptors_reject_and_release_all_returned_handles', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'capsule_consumption_and_unconsumed_deletion_each_keep_exactly_one_storage_owner', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'returned_alias_retains_actual_input_owner_for_entire_consumer_lifetime', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'consumed_capsule_can_finalize_on_a_foreign_thread_without_the_python_gil', 1, 'host-foreign-abi'),
    ('native_output_validation', 'test', 'versioned_capsule_preserves_descriptor_flags_and_consumption_lifetime', 1, 'host-foreign-abi'),
    ('native_device_validation', 'test', 'dynamic_device_owners_preserve_rank_device_and_input_lifetime', 4, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'invalid_storage_offset_capacity_and_context_never_reach_import_or_entry', 7, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'malformed_dynamic_outputs_release_every_returned_owner', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'empty_device_view_checks_storage_bounds_without_requiring_a_payload_pointer', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'device_dlpack_export_synchronizes_and_rejects_invalid_requests_without_a_capsule', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'device_owner_duplicate_outputs_reject_without_double_finalization', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'device_owner_returned_input_borrow_rejects_without_double_finalization', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'device_owner_duplicate_bad_descriptor_cleanup_releases_once', 1, 'simulated-device-foreign-abi'),
    ('native_device_validation', 'test', 'device_owner_distinct_named_outputs_keep_independent_lifetimes', 1, 'simulated-device-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'repeated_input_names_bind_once_across_inputs_and_returned_outputs', 1, 'host-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'returned_named_extent_must_equal_the_admitted_input_binding', 1, 'host-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'repeated_axes_bind_zero_as_an_extent_and_reject_unequal_values', 1, 'host-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'wildcard_axes_are_independent_with_only_their_own_literal_constraints', 3, 'host-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'explicit_named_constraints_are_consistent_before_execution', 2, 'host-foreign-abi'),
    ('native_manifest_dimensions', 'test', 'malformed_unspecified_and_unbound_named_output_claims_never_execute', 3, 'host-foreign-abi'),
    ('native_python_environment', 'test', 'selected_python_packages_load_with_clean_ambient_environment', 0, 'interpreter-prerequisite'),
    ('native_python_environment', 'test', 'missing_selected_interpreter_is_a_prerequisite_error_without_fallback', 0, 'interpreter-prerequisite'),
    ('chelis_python', 'lib', 'native_dlpack_owner_tests::mismatched_dlpack_request_rejects_and_releases_only_its_retained_owner', 1, 'generated-c-current-runtime'),
)


if __name__ == "__main__":
    unittest.main()
