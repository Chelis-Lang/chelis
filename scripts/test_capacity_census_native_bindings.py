"""Native boundary obligations from spec/11 §§1.1–1.4 and dtype C6.

These fixtures test obligation checking. Only the execution factory can issue
authority for a registered binding; a fixture is never such a receipt.
"""

import copy
import unittest

from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_native_bindings import (
    NATIVE_OUTPUTS,
    require_native_slots,
    native_output_adapter,
    require_private_owner,
)
from test_capacity_census_graph import Artifact, primitive, reference


def fixture():
    artifact = Artifact("chelis_python")
    artifact.external(10, "pyo3::err::PyResult")
    artifact.external(11, "alloc::vec::Vec")
    for item_id, identity in enumerate(NATIVE_OUTPUTS.values(), 20):
        artifact.external(item_id, identity)
    return artifact


def private_owner_fixture():
    artifact = Artifact("chelis_python")
    artifact.add(2, "native_tensor", {"module": {"items": [20], "is_stripped": False}},
                 path=["chelis_python", "native_tensor"], visibility="crate")
    field = artifact.field("metadata", reference(30))
    artifact.doc["index"][str(field)]["visibility"] = {
        "restricted": {"parent": 2, "path": "::native_tensor"},
    }
    artifact.external(30, "chelis_abi::metadata::CheckedTensorMetadata")
    artifact.struct(20, "ValidatedTensor", [field], public=False)
    artifact.doc["paths"]["20"]["path"] = ["chelis_python", "native_tensor", "ValidatedTensor"]
    artifact.doc["index"]["20"]["span"] = {
        "filename": "crates/chelis-python/src/native_tensor.rs",
    }
    return artifact, field


class NativeBoundaryObligations(unittest.TestCase):
    def test_every_registered_boundary_requires_all_and_only_its_slots(self):
        slots = {
            "chelis_python::CompiledModel::__call__": ("self", "py", "args", "kwargs"),
            "chelis_python::NativeTensor::__dlpack__":
                ("self", "py", "stream", "max_version", "dl_device", "copy"),
            "chelis_python::NativeTensor::__dlpack_device__": ("self",),
            "chelis_python::NativeTensor::shape": ("self",),
        }
        for owner, labels in slots.items():
            inputs = [(label, primitive("bool")) for label in labels]
            require_native_slots(owner, inputs)
            for changed in (inputs[:-1], inputs + inputs[:1],
                            inputs + [("extra_number", primitive("f64"))],
                            inputs + [("extra_json", primitive("str"))]):
                with self.subTest(owner=owner, changed=changed), self.assertRaises(GraphError):
                    require_native_slots(owner, changed)
        with self.assertRaises(GraphError):
            require_native_slots("chelis_python::Arbitrary::tagged", [("self", None)])

    def test_transport_returns_require_the_exact_concrete_adapter(self):
        artifact = fixture()
        graph = RustdocGraph([artifact.doc])
        for item_id, (owner, adapter) in enumerate(NATIVE_OUTPUTS.items(), 20):
            wrapped = owner != "chelis_python::NativeTensor::__dlpack_device__"
            output = reference(10, reference(item_id)) if wrapped else reference(item_id)
            self.assertEqual(native_output_adapter(graph, owner, output), adapter)
            for wrong in (primitive("f64"), reference(11, primitive("i64")),
                          reference(10, reference(item_id), primitive("f64")),
                          {"tuple": [output, primitive("f64")]},
                          reference(item_id, primitive("f64")),
                          reference(10, reference(999))):
                with self.subTest(owner=owner, wrong=wrong), self.assertRaises(GraphError):
                    native_output_adapter(graph, owner, wrong)

    def test_display_names_and_foreign_wrappers_do_not_establish_identity(self):
        artifact = fixture()
        owner = "chelis_python::CompiledModel::__call__"
        expected = reference(10, reference(20))
        self.assertEqual(native_output_adapter(RustdocGraph([artifact.doc]), owner, expected),
                         NATIVE_OUTPUTS[owner])
        for change in ("foreign_crate", "foreign_module", "arbitrary_tagged_enum"):
            modified = copy.deepcopy(artifact.doc)
            path = modified["paths"]["20"]["path"]
            if change == "foreign_crate":
                path[0] = "impostor"
            elif change == "foreign_module":
                path[1] = "unvalidated"
            else:
                path[-1] = "TaggedNumber"
            with self.subTest(change=change), self.assertRaises(GraphError):
                native_output_adapter(RustdocGraph([modified]), owner, expected)

    def test_shape_is_an_exact_numeric_operation_instead_of_a_transport(self):
        artifact = fixture()
        graph = RustdocGraph([artifact.doc])
        owner = "chelis_python::NativeTensor::shape"
        self.assertIsNone(native_output_adapter(graph, owner, reference(11, primitive("i64"))))
        for wrong in (reference(11, primitive("usize")), reference(11, primitive("i32")),
                      reference(11, primitive("f64")), reference(10, reference(11, primitive("i64"))),
                      reference(20), primitive("i64")):
            with self.subTest(wrong=wrong), self.assertRaises(GraphError):
                native_output_adapter(graph, owner, wrong)

    def test_private_owner_requires_defining_module_and_complete_field_graph(self):
        artifact, field = private_owner_fixture()
        identity = "chelis_python::native_tensor::ValidatedTensor"
        source = "crates/chelis-python/src/native_tensor.rs"
        self.assertEqual(require_private_owner(RustdocGraph([artifact.doc]), identity, source),
                         (("metadata", reference(30)),))
        for mutation in ("public", "crate", "wrong_parent", "wrong_path", "stripped",
                         "missing_field", "detached", "moved", "generic", "tuple", "alias"):
            modified = copy.deepcopy(artifact.doc)
            item = modified["index"]["20"]
            body = item["inner"]["struct"]
            private_field = modified["index"][str(field)]
            if mutation in ("public", "crate"):
                private_field["visibility"] = mutation
            elif mutation == "wrong_parent":
                private_field["visibility"]["restricted"]["parent"] = 0
            elif mutation == "wrong_path":
                private_field["visibility"]["restricted"]["path"] = "::another"
            elif mutation == "stripped":
                body["kind"]["plain"]["has_stripped_fields"] = True
            elif mutation == "missing_field":
                del modified["index"][str(field)]
            elif mutation == "detached":
                modified["index"]["2"]["inner"]["module"]["items"] = []
            elif mutation == "moved":
                item["span"]["filename"] = "elsewhere.rs"
            elif mutation == "generic":
                body["generics"]["params"] = [{"name": "T", "kind": {"type": {}}}]
            elif mutation == "tuple":
                body["kind"] = {"tuple": [field]}
            else:
                item["inner"] = {"type_alias": {"type": reference(30), "generics": {"params": []}}}
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                require_private_owner(RustdocGraph([modified]), identity, source)


if __name__ == "__main__":
    unittest.main()
