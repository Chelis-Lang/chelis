"""Live registration packets select obligations; they cannot issue authority."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

from capacity_census_graph import GraphError
from capacity_census_native_registration import CheckedNativeRegistrations, validate_native_registration


def packet(source):
    slots = (
        ("CompiledModel", "NativeCompiledModel", "__call__", "method", "wrapper_descriptor"),
        ("NativeTensor", "NativeTensor", "__dlpack__", "method", "method_descriptor"),
        ("NativeTensor", "NativeTensor", "__dlpack_device__", "method", "method_descriptor"),
        ("NativeTensor", "NativeTensor", "shape", "getter", "getset_descriptor"),
    )
    return {
        "source_path": "crates/chelis-python/src/lib.rs",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "classes": {"CompiledModel": "chelis_python::NativeCompiledModel", "NativeTensor": "chelis_python::NativeTensor"},
        "registrations": [dict(owner=rust, python_name=name, rust_name=name, kind=kind, line=offset, column=1)
                          for offset, (_, rust, name, kind, _) in enumerate(slots, 1)],
        "descriptors": [dict(owner=owner, rust_owner="chelis_python::" + rust, python_name=name,
                             kind=kind, descriptor=descriptor) for owner, rust, name, kind, descriptor in slots],
    }


class NativeRegistrationControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "crates/chelis-python/src/lib.rs"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"compiled registrar source\n")
        self.packet = packet(self.source.read_bytes())

    def test_complete_registration_yields_only_exact_owner_obligations(self):
        roots = validate_native_registration(self.root, self.packet)
        self.assertEqual(len(roots), 4)
        self.assertEqual(roots["chelis_python::CompiledModel::__call__"],
                         "chelis_python::NativeCompiledModel::__call__#method")
        self.assertEqual(roots["chelis_python::NativeTensor::shape"],
                         "chelis_python::NativeTensor::shape#getter")

    def test_supplied_registration_packet_cannot_construct_compiled_evidence(self):
        for arguments in ({}, {"packet": self.packet}, {"root": self.root, "source_sha256": "0" * 64}):
            with self.assertRaises(TypeError):
                CheckedNativeRegistrations(**arguments)

    def test_missing_extra_duplicate_or_changed_exposures_are_rejected(self):
        for mutation in ("missing", "extra", "duplicate", "foreign", "wrong-kind", "wrong-implementation",
                         "descriptor-kind", "descriptor-owner", "descriptor-missing", "descriptor-duplicate",
                         "classes", "position", "bool-position", "unknown-field", "path", "source"):
            changed = copy.deepcopy(self.packet)
            if mutation == "missing": changed["registrations"].pop()
            elif mutation == "extra": changed["registrations"].append({**changed["registrations"][0], "python_name": "other"})
            elif mutation == "duplicate": changed["registrations"].append(changed["registrations"][0])
            elif mutation == "foreign": changed["registrations"][0]["owner"] = "ForeignTensor"
            elif mutation == "wrong-kind": changed["registrations"][0]["kind"] = "staticmethod"
            elif mutation == "wrong-implementation": changed["registrations"][0]["rust_name"] = "helper"
            elif mutation == "descriptor-kind": changed["descriptors"][0]["descriptor"] = "method_descriptor"
            elif mutation == "descriptor-owner": changed["descriptors"][0]["rust_owner"] = "other::NativeCompiledModel"
            elif mutation == "descriptor-missing": changed["descriptors"].pop()
            elif mutation == "descriptor-duplicate": changed["descriptors"].append(changed["descriptors"][0])
            elif mutation == "classes": changed["classes"]["CompiledModel"] = "other::NativeCompiledModel"
            elif mutation == "position": changed["registrations"][0]["column"] = 0
            elif mutation == "bool-position": changed["registrations"][0]["line"] = True
            elif mutation == "unknown-field": changed["authority"] = "TaggedTransport"
            elif mutation == "path": changed["source_path"] = "elsewhere.rs"
            else: changed["source_sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                validate_native_registration(self.root, changed)

    def test_changed_current_source_invalidates_registration(self):
        validate_native_registration(self.root, self.packet)
        self.source.write_bytes(b"changed registrar source\n")
        with self.assertRaisesRegex(GraphError, "source"):
            validate_native_registration(self.root, self.packet)

    def test_malformed_packets_fail_as_obligation_errors(self):
        for changed in (None, [], {}, {**self.packet, "registrations": None},
                        {**self.packet, "descriptors": [None] * 4},
                        {**self.packet, "registrations": [None] * 4}):
            with self.subTest(packet=changed), self.assertRaises(GraphError):
                validate_native_registration(self.root, changed)


if __name__ == "__main__":
    unittest.main()
