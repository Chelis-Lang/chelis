"""Binding exposure controls for spec/11 §1 and dtype_semantics.md §C6."""

import copy
import unittest

from capacity_census_bindings import discover_bindings
from test_capacity_census_graph import Artifact, primitive, reference


def fixture(output=None, inputs=()):
    a = Artifact("chelis_python")
    a.add(1, "probe", {"function": {"sig": {"inputs": list(inputs), "output": output}}},
          path=["chelis_python", "probe"])
    return a


def exposure(a, **kwargs):
    return discover_bindings([a.doc], ["probe"], [], {}, **kwargs)[0]


class BindingExposure(unittest.TestCase):
    def test_float_return_and_parameter_are_both_visible(self):
        for a in (fixture(primitive("f64")), fixture(None, [("value", primitive("f64"))])):
            result = exposure(a)
            self.assertIsNone(result["problem"])
            self.assertIn("float-carrier", result["flags"])

    def test_bool_return_is_nonnumeric_but_integer_return_is_not(self):
        self.assertEqual(exposure(fixture(primitive("bool")))["flags"], [])
        self.assertEqual(exposure(fixture(primitive("i64")))["flags"], ["numeric-return"])

    def test_return_containers_tuples_and_aliases_reach_private_numeric_fields(self):
        a = fixture(reference(10, reference(11)))
        a.external(10, "alloc::vec::Vec")
        field = a.field("value", {"tuple": [primitive("bool"), primitive("f64")]})
        a.struct(12, "Hidden", [field], public=False)
        a.add(11, "Alias", {"type_alias": {"type": reference(12), "generics": {"params": []}}},
              path=["chelis_python", "Alias"])
        result = exposure(a)
        self.assertIsNone(result["problem"])
        self.assertIn("float-carrier", result["flags"])
        self.assertIn("chelis_python::Hidden.value", str(result["leaves"]))

    def test_same_flag_private_width_change_changes_exposure_identity(self):
        a = fixture(reference(10))
        field = a.field("value", primitive("f32"))
        a.struct(10, "Hidden", [field], public=False)
        before = exposure(a)
        a.doc["index"][str(field)]["inner"]["struct_field"] = primitive("f64")
        after = exposure(a)
        self.assertEqual(before["flags"], after["flags"])
        self.assertNotEqual(before["identity"], after["identity"])

    def test_missing_definition_unknown_primitive_and_open_generic_fail_closed(self):
        for ty in (reference(404), primitive("f128"), {"generic": "T"}, {"impl_trait": []}):
            self.assertIsNotNone(exposure(fixture(ty))["problem"])

    def test_dynamic_python_values_require_an_actual_payload_contract(self):
        for identity in ("pyo3::types::any::PyAny", "pyo3::types::dict::PyDict",
                         "pyo3::types::tuple::PyTuple", "pyo3::instance::PyObject"):
            a = fixture(reference(10))
            a.external(10, identity)
            self.assertIn("dynamic Python payload", exposure(a)["problem"])

    def test_untyped_text_output_cannot_hide_compiler_json(self):
        a = fixture(reference(10))
        a.external(10, "alloc::string::String")
        self.assertIn("text result requires", exposure(a)["problem"])

    def test_gil_token_is_not_a_python_value(self):
        a = fixture(primitive("bool"), [("py", reference(10))])
        a.external(10, "pyo3::marker::Python")
        self.assertEqual(exposure(a)["flags"], [])
        a.doc["paths"]["10"]["path"] = ["impostor", "Python"]
        self.assertIsNotNone(exposure(a)["problem"])

    def test_python_result_analyzes_success_type(self):
        a = fixture(reference(10, primitive("f64")))
        a.external(10, "pyo3::err::PyResult")
        self.assertIn("float-carrier", exposure(a)["flags"])
        a.doc["index"]["1"]["inner"]["function"]["sig"]["output"] = reference(10)
        self.assertIn("arity", exposure(a)["problem"])

    def test_registered_class_is_opaque_but_each_getter_is_discovered(self):
        a = fixture(reference(10))
        a.struct(10, "NativeModel", [a.field("private_storage", primitive("f64"))])
        a.external(40, "pyo3::pyclass::PyClass")
        a.add(41, None, {"impl": {"trait": {"id": 40}, "items": []}})
        a.add(42, None, {"impl": {"trait": None, "items": [43]}})
        a.add(43, "score", {"function": {"sig": {"inputs": [], "output": primitive("f64")}}})
        a.doc["index"]["10"]["inner"]["struct"]["impls"] += [41, 42]
        classes = {"Model": "chelis_python::NativeModel"}
        rows = discover_bindings([a.doc], ["probe"], ["Model::score"], classes)
        self.assertEqual(rows[0]["flags"], [])
        self.assertIn("float-carrier", rows[1]["flags"])
        with self.assertRaisesRegex(Exception, "constructor provenance"):
            discover_bindings([a.doc], [], ["Model::__new__"], classes)
        a.doc["index"]["10"]["inner"]["struct"]["impls"].remove(41)
        with self.assertRaisesRegex(Exception, "PyClass"):
            discover_bindings([a.doc], ["probe"], [], classes)

    def test_constructor_requires_provenance_even_with_an_inherent_new_helper(self):
        a = fixture()
        a.struct(10, "Handle", [])
        a.external(40, "pyo3::pyclass::PyClass")
        a.add(41, None, {"impl": {"trait": {"id": 40}, "items": []}})
        a.add(42, None, {"impl": {"trait": None, "items": [43, 44]}})
        a.add(43, "new", {"function": {"sig": {"inputs": [], "output": {"generic": "Self"}}}})
        a.add(44, "create", {"function": {"sig": {
            "inputs": [("dtype", primitive("i32"))], "output": {"generic": "Self"}}}})
        a.doc["index"]["10"]["inner"]["struct"]["impls"] += [41, 42]
        classes = {"Handle": "chelis_python::Handle"}
        # An explicitly named method has an exact Rust identity. A Python
        # constructor slot does not reveal which of these functions owns it.
        result = discover_bindings([a.doc], [], ["Handle::create"], classes)[0]
        self.assertEqual(result["flags"], ["raw-dtype-int"])
        with self.assertRaisesRegex(Exception, "constructor provenance"):
            discover_bindings([a.doc], [], ["Handle::__new__"], classes)

    def test_duplicate_or_missing_registered_callables_fail(self):
        a = fixture(primitive("bool"))
        for names in (["probe", "probe"], ["missing"]):
            with self.assertRaisesRegex(Exception, "duplicate|missing"):
                discover_bindings([a.doc], names, [], {})

    def test_alias_cycle_fails_and_nominal_recursion_terminates(self):
        a = fixture(reference(10))
        a.struct(10, "Tree", [a.field("value", primitive("i64")), a.field("next", reference(10))])
        self.assertEqual(exposure(a)["flags"], ["numeric-return"])
        a.doc["index"]["10"]["inner"] = {"type_alias": {"type": reference(10)}}
        self.assertIn("alias cycle", exposure(a)["problem"])

    def test_imports_require_the_defining_artifact(self):
        a = fixture(reference(10))
        a.external(10, "producer::Value")
        producer = Artifact("producer")
        producer.struct(1, "Value", [producer.field("score", primitive("f64"))])
        self.assertIn("missing defining artifact", exposure(a)["problem"])
        result = discover_bindings([a.doc, producer.doc], ["probe"], [], {})[0]
        self.assertIn("float-carrier", result["flags"])

    def test_source_json_requires_its_exact_sealed_result_and_numeric_changes_are_visible(self):
        a = fixture(reference(10, reference(11)))
        a.struct(10, "SourceJson", [a.field("value", {"generic": "T"})], params=("T",),
                 path=["chelis_python", "source_json", "SourceJson"])
        a.struct(11, "DecompileResult", [a.field("surf_text", primitive("str"))],
                 path=["chelis_compiler_api", "schema", "DecompileResult"])
        self.assertEqual(exposure(a)["flags"], [])
        a.doc["index"]["11"]["inner"]["struct"]["kind"]["plain"]["fields"].append(a.field("score", primitive("f64")))
        self.assertIn("float-carrier", exposure(a)["flags"])
        a.doc["paths"]["11"]["path"][-1] = "UnregisteredPayload"
        self.assertIn("sealed source result", exposure(a)["problem"])

    def test_source_json_custom_codec_or_extra_wrapper_field_is_rejected(self):
        a = fixture(reference(10, reference(11)))
        a.struct(10, "SourceJson", [a.field("value", {"generic": "T"})], params=("T",),
                 path=["chelis_python", "source_json", "SourceJson"])
        a.struct(11, "ValidateResult", [a.field("valid", primitive("bool"))],
                 path=["chelis_compiler_api", "schema", "ValidateResult"])
        b = copy.deepcopy(a)
        b.doc["index"]["11"]["inner"]["struct"]["impls"] = []
        self.assertIn("codec", exposure(b)["problem"])
        b.external(20, "pyo3::err::PyResult")
        b.doc["index"]["1"]["inner"]["function"]["sig"]["output"] = reference(20, reference(10, reference(11)))
        self.assertIn("codec", exposure(b)["problem"])
        a.doc["index"]["10"]["inner"]["struct"]["kind"]["plain"]["fields"].append(a.field("extra", primitive("f64")))
        self.assertIn("wrapper", exposure(a)["problem"])

    def test_source_json_authority_is_confined_to_its_subtree(self):
        a = fixture()
        a.struct(10, "SourceJson", [a.field("value", {"generic": "T"})], params=("T",),
                 path=["chelis_python", "source_json", "SourceJson"])
        a.struct(11, "ValidateResult", [a.field("mode", primitive("str"))],
                 path=["chelis_compiler_api", "schema", "ValidateResult"])
        a.external(12, "alloc::string::String")
        a.external(13, "pyo3::err::PyResult")
        a.external(14, "alloc::vec::Vec")
        a.add(15, "TextAlias", {"type_alias": {"type": reference(12)}},
              path=["chelis_python", "TextAlias"])
        a.struct(16, "Envelope", [a.field("value", {"generic": "T"})], params=("T",))
        a.struct(17, "Recursive", [a.field("value", {"generic": "T"}),
                                   a.field("next", reference(17, {"generic": "T"}))], params=("T",))
        source = reference(10, reference(11))

        def result(ty):
            a.doc["index"]["1"]["inner"]["function"]["sig"]["output"] = reference(13, ty)
            return exposure(a)

        for good in (source, {"tuple": [source, primitive("bool")]},
                     reference(14, source), reference(16, source), reference(17, source)):
            with self.subTest(good=good):
                self.assertIsNone(result(good)["problem"])
        for sibling in (reference(12), reference(14, reference(12)), reference(15),
                        reference(16, reference(12)), reference(17, reference(12)), reference(11)):
            for members in ([source, sibling], [sibling, source]):
                with self.subTest(members=members):
                    self.assertIn("text result requires", result({"tuple": members})["problem"])

        # A sibling's ordinary Rust shape does not require a JSON codec. Move
        # that same type into the payload and its missing codec must fail.
        no_codec = a.struct(18, "Plain", [a.field("valid", primitive("bool"))])
        a.doc["index"][str(no_codec)]["inner"]["struct"]["impls"] = []
        self.assertIsNone(result({"tuple": [source, reference(18)]})["problem"])
        a.doc["index"]["11"]["inner"]["struct"]["kind"]["plain"]["fields"].append(
            a.field("nested", reference(18)))
        self.assertIn("codec", result(source)["problem"])


if __name__ == "__main__":
    unittest.main()
