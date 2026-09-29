"""Spec11/OP45 execution evidence must come from the fixed live native matrix.

These pure controls test evidence rejection, not native boundary execution.
The 37 actual Rust cases remain the separate executable acceptance matrix.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from capacity_census_graph import GraphError
import capacity_census_native_execution as execution


def replace_bytes(path: Path, content: bytes) -> None:
    """Give `path` new bytes on a fresh inode, never through the old one.

    Without reflinks Kache restores Cargo outputs as hardlinks to its read-only
    store blobs: writing through one fails, and forcing it writable would
    corrupt the shared blob.
    """
    path.unlink(missing_ok=True)
    path.write_bytes(content)


def receipt_bytes(archive_sha256: str, archive: str = "libchelis_runtime.a") -> bytes:
    """A staging receipt shaped like the one chelis-runtime-bundle writes."""
    return json.dumps({"schema": "chelis-runtime-staging/1", "archive": archive,
                       "archive_sha256": archive_sha256, "headers": {"chelis_runtime.h": "h" * 64},
                       "mode": "development", "chelis_version": "0.0.0"}).encode()


class MatrixContractTests(unittest.TestCase):
    def test_authority_identity_ignores_run_paths_but_binds_the_executed_contract(self):
        packet = {
            "schema": 1,
            "source": {
                "head": "current-head",
                "common_source_sha256": "a" * 64,
                "files": [{"path": "source.rs", "kind": "file", "sha256": "b" * 64}],
            },
            "selected": ["suite::test"],
            "executed": [{"id": "suite::test", "outcome": "passed"}],
            "binaries": [
                {"path": f"/run-a/binary-{index}", "sha256": chr(99 + index) * 64}
                for index in range(len(execution.GROUPS))
            ],
            "interpreter": {
                "path": "/run-a/python",
                "sha256": "i" * 64,
                "prefix": "/run-a/venv",
                "version": "Python test version",
            },
            "runtime": {"sha256": "r" * 64},
            "captures": [
                {
                    "directory": "/run-a/capture",
                    "test": "test",
                    "fixture_kind": "host-foreign-abi",
                    "library": {"path": "/run-a/model.so", "sha256": "l" * 64},
                    "original_library": "/run-a/original/model.so",
                    "runtime_sha256": "r" * 64,
                    "completion_sha256": "m" * 64,
                }
            ],
            "evidence": [{"path": "run-a.log", "sha256": "n" * 64}],
            "limits": {"device": "simulated", "generated_c_link": "fixed"},
        }
        expected = execution.execution_identity(packet)
        another_run = copy.deepcopy(packet)
        for row in another_run["binaries"]:
            row["path"] = row["path"].replace("/run-a/", "/run-b/")
        another_run["interpreter"]["path"] = "/run-b/python"
        another_run["interpreter"]["prefix"] = "/run-b/venv"
        another_run["captures"][0].update(
            directory="/run-b/capture",
            original_library="/run-b/original/model.so",
            completion_sha256="z" * 64,
        )
        another_run["captures"][0]["library"] = {
            "path": "/run-b/model.so",
            "sha256": "y" * 64,
        }
        another_run["evidence"] = [{"path": "run-b.log", "sha256": "x" * 64}]
        another_run["source"]["head"] = "committed-head"
        another_run["source"]["files"].append(
            {
                "path": "spec/design/capacity_census_bindings.json",
                "kind": "file",
                "sha256": "w" * 64,
            }
        )
        self.assertEqual(execution.execution_identity(another_run), expected)

        mutations = (
            (
                "source",
                lambda value: value["source"].update(common_source_sha256="0" * 64),
            ),
            ("selection", lambda value: value["selected"].append("suite::other")),
            ("outcome", lambda value: value["executed"][0].update(outcome="failed")),
            ("binary", lambda value: value["binaries"][0].update(sha256="0" * 64)),
            ("interpreter", lambda value: value["interpreter"].update(sha256="0" * 64)),
            ("runtime", lambda value: value["runtime"].update(sha256="0" * 64)),
            ("capture", lambda value: value["captures"][0].update(test="other")),
            ("limits", lambda value: value["limits"].update(device="hardware")),
        )
        for label, mutate in mutations:
            changed = copy.deepcopy(packet)
            mutate(changed)
            with self.subTest(label=label):
                self.assertNotEqual(execution.execution_identity(changed), expected)

    def test_fixed_worker_imports_current_helpers_with_safe_path_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            environment = execution._worker_environment(root / "capture", execution.GROUPS[0])
            self.assertEqual(environment["PYTHONSAFEPATH"], "1")
            result = subprocess.run([sys.executable, str(Path(execution.__file__).resolve())],
                                    cwd=root, env=environment, capture_output=True, check=False)
            self.assertEqual(result.returncode, 1)
            self.assertIn(b"native execution is invoked by its fixed collection API", result.stderr)
            self.assertNotIn(b"ModuleNotFoundError", result.stderr)

    def test_fixed_matrix_preserves_every_exact_spec_case_and_all_fixture_instances(self):
        actual = tuple((group.name, group.kind, name, count, group.fixture_kind)
                       for group in execution.GROUPS
                       for name, count in zip(group.selected, group.instances, strict=True))
        self.assertEqual(actual, EXPECTED_MATRIX)
        self.assertEqual(len(actual), 37)
        self.assertEqual(sum(row[3] for row in actual), 50)

    def test_public_collection_accepts_no_supplied_selection_or_execution(self):
        import inspect
        self.assertEqual(tuple(inspect.signature(execution.collect_native_execution).parameters),
                         ("root", "target"))
        with self.assertRaises(TypeError):
            execution.CheckedNativeExecution(packet={"passed": 37})

    def test_native_worker_packet_keeps_exact_lifecycle_and_rejects_wire_probe(self):
        from capacity_census_wire_runner import TestExecution

        root = Path(execution.__file__).resolve().parent.parent
        target = root / "target" / "native-worker-packet-probe"
        directory = target / "capture"
        group = execution.GROUPS[0]
        selected = tuple(sorted(group.selected))
        result = TestExecution(("/bin/native-test", "--exact", "case"),
                               selected, selected, "a" * 64)

        def emitted_packet(worker_result):
            environment = {
                "CHELIS_NATIVE_EXECUTION_CAPTURE": str(directory),
                "CHELIS_NATIVE_EXECUTION_SUITE": group.name,
            }
            with mock.patch.dict(os.environ, environment), \
                 mock.patch.object(execution, "build_and_run_rust_test", return_value=worker_result) as build, \
                 mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                execution._collect_worker(root, target, directory, group.name)
            build.assert_called_once_with(
                root, target, "chelis-python", group.name, group.selected,
                kind=group.kind, log_prefix=directory / "test",
            )
            return json.loads(output.getvalue())

        self.assertEqual(emitted_packet(result), {
            "command": ["/bin/native-test", "--exact", "case"],
            "selected": list(selected),
            "executed": list(selected),
            "output_sha256": "a" * 64,
        })
        selected_probe = TestExecution(
            result.command, result.selected, result.executed, result.output_sha256,
            "/owned/wire-target", "b" * 64,
        )
        with self.assertRaisesRegex(GraphError, "native worker cannot carry a selected wire probe"):
            emitted_packet(selected_probe)

    def test_worker_environment_is_explicit_without_mutating_parent_or_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            group = root / "group"
            hostile = {
                "CHELIS_RUNTIME_DIR": "/foreign/runtime", "CHELIS_CC": "/foreign/compiler",
                "CHELIS_TEST_CC": "/foreign/test-compiler", "CHELIS_HIPCC": "/foreign/hipcc",
                "CHELIS_DEVICE_OWNER_TEST_WORKER": "1", "CHELIS_NATIVE_ENV_TEST_WORKER": "1",
                "CHELIS_NATIVE_EXECUTION_CAPTURE": "/foreign/capture", "PYTHONPATH": "/foreign/python",
            }
            with mock.patch.dict(os.environ, hostile):
                before = dict(os.environ)
                child = execution._worker_environment(group, execution.GROUPS[0])
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(child["CHELIS_NATIVE_EXECUTION_CAPTURE"], str(group))
                self.assertEqual(child["CHELIS_NATIVE_EXECUTION_SUITE"], execution.GROUPS[0].name)
                self.assertEqual(child["CARGO_BUILD_JOBS"], "1")
                for name in ("CHELIS_RUNTIME_DIR", "CHELIS_CC", "CHELIS_TEST_CC", "CHELIS_HIPCC",
                             "CHELIS_DEVICE_OWNER_TEST_WORKER", "CHELIS_NATIVE_ENV_TEST_WORKER"):
                    self.assertNotIn(name, child)
                self.assertNotEqual(child["PYTHONPATH"], hostile["PYTHONPATH"])


class SourceIdentityTests(unittest.TestCase):
    def test_symlinked_header_referent_bytes_are_bound_even_without_a_tracked_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "external-header.h"
            target.write_bytes(b"first declaration")
            (root / "runtime.h").symlink_to("external-header.h")
            def snapshot():
                with mock.patch.object(execution.subprocess, "check_output",
                                       side_effect=[b"current-head\n", b"runtime.h\0"]), \
                     mock.patch.object(execution, "source_identity", return_value="common-source"):
                    return execution._source_packet(root)
            first = snapshot()
            target.write_bytes(b"changed declaration")
            self.assertNotEqual(snapshot(), first)

    def test_tracked_directory_symlink_and_actual_header_bytes_are_both_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "agent-skills").mkdir()
            (root / "skills").symlink_to("agent-skills", target_is_directory=True)
            header = root / "runtime.h"
            header.write_bytes(b"current header")
            def snapshot():
                with mock.patch.object(execution.subprocess, "check_output",
                                       side_effect=[b"current-head\n", b"runtime.h\0skills\0"]), \
                     mock.patch.object(execution, "source_identity", return_value="common-source"):
                    return execution._source_packet(root)
            first = snapshot()
            self.assertEqual(first["files"][1]["kind"], "symlink")
            self.assertEqual(first["files"][1]["sha256"], hashlib.sha256(b"agent-skills").hexdigest())
            header.write_bytes(b"changed header")
            self.assertNotEqual(snapshot(), first)
            (root / "other-skills").mkdir()
            (root / "skills").unlink()
            (root / "skills").symlink_to("other-skills", target_is_directory=True)
            self.assertNotEqual(snapshot()["files"][1], first["files"][1])

    def test_unsupported_tracked_directory_is_a_typed_source_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "directory").mkdir()
            with mock.patch.object(execution.subprocess, "check_output",
                                   side_effect=[b"current-head\n", b"directory\0"]), \
                 self.assertRaises(GraphError):
                execution._source_packet(root)


class RetainedCaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.group = execution.GROUPS[0]
        self.case = self.root / "fixtures" / self.group.selected[0] / "instance-1"
        self.case.mkdir(parents=True)
        self.runtime = b"current runtime bytes"
        self.runtime_digest = hashlib.sha256(self.runtime).hexdigest()
        source = b"def main(x: tensor[1,f32]) -> tensor[1,f32] = copy(x)\n"
        library = b"actual library stand-in for pure record validation"
        self.manifest = {"abi_version": 2, "target": "c", "source_path": str(self.case / "program.ch"),
                         "source_hash": hashlib.sha256(source).hexdigest(),
                         "runtime_sha256": self.runtime_digest,
                         "library_sha256": hashlib.sha256(library).hexdigest()}
        self.files = {
            "program.ch": source,
            "compiled/program.c": b"/* generated fixture source */",
            "compiled/chelis_runtime.h": b"/* current embedded header */",
            "compiled/program.so": library,
            "compiled/program.json": json.dumps(self.manifest).encode(),
            "compiled/libchelis_runtime.a": self.runtime,
            "compiled/chelis_runtime.receipt.json": receipt_bytes(self.runtime_digest),
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
        return execution._validate_instance(self.case, self.group, self.group.selected[0], self.packet)

    def rewrite(self, name, content):
        """Replace or remove one snapshot file and rehash every record of it, as a forger would."""
        path = self.case / name
        path.unlink(missing_ok=True)
        model_path = self.case / "loaded-model.json"
        model = json.loads(model_path.read_text())
        model["files"] = [row for row in model["files"] if row["captured"] != name]
        files = [row for row in self.packet["files"] if row["path"] not in (name, "loaded-model.json")]
        if content is not None:
            path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            model["files"].append({"original": str(Path(model["root"]) / Path(name).name),
                                   "captured": name, "sha256": digest})
            files.append({"path": name, "sha256": digest})
        model_path.write_text(json.dumps(model))
        files.append({"path": "loaded-model.json", "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()})
        self.packet = {**self.packet, "files": sorted(files, key=lambda row: row["path"])}

    def rewrite_manifest(self, **changes):
        manifest = {key: value for key, value in {**self.manifest, **changes}.items() if value is not None}
        self.rewrite("compiled/program.json", json.dumps(manifest).encode())

    def test_complete_retained_generated_artifacts_bind_actual_model_and_runtime(self):
        self.assertEqual(self.validate()["runtime_sha256"], self.runtime_digest)

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

    def test_staged_archive_must_match_its_receipt_and_manifest_even_if_its_record_is_rehashed(self):
        self.rewrite("compiled/libchelis_runtime.a", b"foreign runtime")
        with self.assertRaises(GraphError): self.validate()

    def test_generated_manifest_must_name_the_runtime_it_staged(self):
        other = hashlib.sha256(b"another carried runtime").hexdigest()
        self.rewrite_manifest(runtime_sha256=other)
        with self.assertRaises(GraphError): self.validate()
        # A foreign manifest stages nothing, so only the run-wide shared digest binds it.
        library = self.case / "compiled/program.so"
        self.assertEqual(execution._bound_runtime(library, "host-foreign-abi"), other)

    def test_receipt_must_exist_and_record_the_staged_archive(self):
        original = self.files["compiled/chelis_runtime.receipt.json"]
        variants = [None, b"[]", receipt_bytes(self.runtime_digest, archive="other.a"),
                    receipt_bytes(hashlib.sha256(b"another carried runtime").hexdigest())]
        for content in variants:
            with self.subTest(content=content):
                self.rewrite("compiled/chelis_runtime.receipt.json", content)
                with self.assertRaises(GraphError): self.validate()
                self.rewrite("compiled/chelis_runtime.receipt.json", original)
        self.assertTrue(self.validate())

    def test_every_manifest_must_carry_both_digests_and_name_its_library(self):
        library = self.case / "compiled/program.so"
        variants = [{"runtime_sha256": None}, {"library_sha256": None},
                    {"library_sha256": hashlib.sha256(b"another library").hexdigest()}]
        for change in variants:
            with self.subTest(change=change):
                self.rewrite_manifest(**change)
                with self.assertRaises(GraphError): self.validate()
                with self.assertRaises(GraphError): execution._bound_runtime(library, "host-foreign-abi")
        self.rewrite_manifest()
        self.assertTrue(self.validate())

    def test_consistently_restaged_runtime_is_bound_by_its_own_digest(self):
        runtime = b"another carried runtime"
        digest = hashlib.sha256(runtime).hexdigest()
        self.rewrite("compiled/libchelis_runtime.a", runtime)
        self.rewrite("compiled/chelis_runtime.receipt.json", receipt_bytes(digest))
        self.rewrite_manifest(runtime_sha256=digest)
        self.assertEqual(self.validate()["runtime_sha256"], digest)

    def test_captures_must_share_one_carried_runtime(self):
        other = hashlib.sha256(b"another carried runtime").hexdigest()
        shared = [{"runtime_sha256": self.runtime_digest}] * 2
        self.assertEqual(execution._carried_runtime(shared), self.runtime_digest)
        for captures in ([], shared + [{"runtime_sha256": other}]):
            with self.subTest(captures=captures), self.assertRaises(GraphError):
                execution._carried_runtime(captures)

    def test_one_fixture_cannot_stand_in_for_the_other_selected_cases(self):
        (self.case / "completion.json").write_text(json.dumps(self.packet))
        with self.assertRaises(GraphError):
            execution._validate_captures(self.root, self.group, self.root / "actual-test-binary")

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


class NativeExecutionIntegration(unittest.TestCase):
    """Current collection is the only input to these acceptance controls."""
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent.parent
        requested = os.environ.get("CHELIS_NATIVE_EXECUTION_TARGET")
        if not requested:
            raise RuntimeError("native execution integration requires an explicit owned target")
        cls.witness = execution.collect_native_execution(root, Path(requested))

    def test_current_fixed_corpus_binds_37_outcomes_and_all_50_actual_fixture_instances(self):
        packet = self.witness.validate()
        self.assertEqual(len(packet["selected"]), 37)
        self.assertEqual(len(packet["executed"]), 37)
        self.assertTrue(all(row["outcome"] == "passed" for row in packet["executed"]))
        self.assertEqual(len(packet["captures"]), 50)
        self.assertEqual(len(packet["binaries"]), 6)
        self.assertEqual(json.loads((self.witness.directory / "report.json").read_text()), packet)
        generated = [row for row in packet["captures"] if row["fixture_kind"] == "generated-c-current-runtime"]
        self.assertEqual(len(generated), 13)
        self.assertTrue(all(not Path(row["original_library"]).exists() for row in generated),
                        "recording must preserve default model-owned TempDir deletion")

    def test_current_receipt_rejects_corrupted_missing_binary_runtime_fixture_and_transcripts(self):
        packet = self.witness.validate()
        first = Path(packet["captures"][0]["directory"])
        library = Path(packet["captures"][0]["library"]["path"])
        candidates = [Path(packet["binaries"][0]["path"]), library.with_name("libchelis_runtime.a"),
                      library.with_name("chelis_runtime.receipt.json"), library, library.with_suffix(".json"),
                      first / "loaded-model.json", first / "completion.json",
                      self.witness.directory / "groups/native_tensor_boundary/test.stdout.log",
                      self.witness.directory / "report.json"]
        for path in candidates:
            with self.subTest(path=path):
                original = path.read_bytes()
                status = path.stat()
                try:
                    replace_bytes(path, original + b"changed")
                    with self.assertRaises(GraphError): self.witness.validate()
                    path.unlink()
                    with self.assertRaises(GraphError): self.witness.validate()
                finally:
                    replace_bytes(path, original)
                    path.chmod(status.st_mode)
                    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
        with mock.patch.object(execution, "_source_packet", return_value={"head": "changed"}):
            with self.assertRaises(GraphError): self.witness.validate()
        self.witness.validate()


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
    ('native_device_validation', 'test', 'dynamic_device_owners_preserve_rank_device_and_input_lifetime', 5, 'simulated-device-foreign-abi'),
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
