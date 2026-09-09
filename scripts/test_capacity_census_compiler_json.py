"""Obligation-validator controls; only the actual execution factory issues authority."""
import copy
from pathlib import Path
import unittest

from capacity_census_compiler_json import (
    API, INPUT_ROLES, MODULE, NATIVE_CASES, OUTPUTS, SOURCE, VerifiedCompilerJsonBindings,
    adapter_for_slot, adapter_payload, require_parameter_slots, validate_conversion_calls,
)
from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_wire_runner import validate_libtest_execution
from test_capacity_census_graph import Artifact, primitive, reference
from test_capacity_census_wire_invocation_owners import nominal


def definition(name):
    crate, path = name.split("::", 1)
    return {"crate": crate, "path": "::" + path, "stable_crate_id": crate}


def conversion_fixture():
    traits = [definition("pyo3::conversion::" + name) for name in ("IntoPyObject", "FromPyObject")]
    def call(adapter, payload, output):
        return {
            "caller": {"definition": {"item_name": "into_pyobject" if output else "extract_bound"},
                       "implementation": {"trait": traits[0 if output else 1],
                                          "self_type": {"shape": nominal(MODULE + adapter)}}},
            "callee": definition("serde_json::ser::to_string" if output else "serde_json::de::from_str"),
            "payloads": [{"shape": payload}], "serializers": [], "local_callee": False,
        }
    return {
        "format": 2, "scope": "compiler-json", "crate": {
            "crate": "chelis_python", "item_name": "chelis_python", "path": "", "def_id": "0:0",
        },
        "bodies": ["compiled"], "errors": [], "conversion_traits": traits,
        "deserialize_trait": definition("serde_core::de::Deserialize"),
        "codec_calls": [], "schema_calls": [], "dynamic_returns": [],
        "calls": [call(adapter, nominal(API + result), True) for adapter, result in OUTPUTS.values()],
        "reader_calls": [call("EvalBindingsJson", nominal("alloc::collections::btree::map::BTreeMap",
                              nominal("alloc::string::String"), nominal(API + "TensorValue"),
                              nominal("alloc::alloc::Global")), False)],
    }


def graph_fixture():
    a, api = Artifact("chelis_python"), Artifact("chelis_compiler_api")
    a.external(10, "pyo3::err::PyResult")
    a.external(11, "alloc::collections::btree::map::BTreeMap")
    a.external(12, "alloc::string::String")
    for index, (adapter, result) in enumerate(list(OUTPUTS.values()) + [("EvalBindingsJson", "TensorValue")], 20):
        api.struct(index, result, [api.field("number", primitive("f64"))])
        api.doc["paths"][str(index)]["path"] = (API + result).split("::")
        a.external(index + 100, API + result)
        payload = reference(index + 100)
        if adapter == "EvalBindingsJson":
            payload = reference(11, reference(12), payload)
        field = a.field("value", payload)
        a.doc["index"][str(field)]["visibility"] = "default"
        a.struct(index, adapter, [field], public=False)
        a.doc["paths"][str(index)]["path"] = (MODULE + adapter).split("::")
        a.doc["index"][str(index)]["span"] = {"filename": SOURCE}
    return a, api


class CompilerJsonAuthority(unittest.TestCase):
    def test_construction_controls_require_exact_compiler_success_or_failure(self):
        from capacity_census_compiler_json_construction import check_construction_outcome, construction_sources
        cases = construction_sources(Path(__file__).resolve().parent.parent)
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({name for name, _, _ in cases}), 22)
        self.assertEqual(sum(error is None for _, _, error in cases), 5)
        check_construction_outcome(None, 0, [], True)
        error = {"level": "error", "code": {"code": "E0308"}}
        check_construction_outcome("E0308", 1, [error], False)
        for expected, status, diagnostics, exists in (
            (None, 0, [], False), (None, 1, [error], False),
            ("E0308", 0, [], True), ("E0451", 1, [error], False),
            ("E0308", 1, [], False), ("E0308", 1, [error, error], False),
            ("E0308", 1, [error], True),
        ):
            with self.assertRaises(GraphError):
                check_construction_outcome(expected, status, diagnostics, exists)

    def test_actual_registration_conversion_and_wire_execution_issue_authority(self):
        # Pure validators admit obligations. The live binding gate executes the
        # actual factory as the positive leg; these cannot create its witness.
        for kwargs in ({}, {"source_sha256": "f" * 64, "ownership": validate_conversion_calls(conversion_fixture())}):
            with self.assertRaises(TypeError):
                VerifiedCompilerJsonBindings(**kwargs)

    def test_each_adapter_binds_exact_root_and_direction(self):
        a, api = graph_fixture()
        graph = RustdocGraph([a.doc, api.doc])
        for index, (owner, (adapter, _)) in enumerate(OUTPUTS.items(), 20):
            slots = [(label, None) for label in INPUT_ROLES[owner]]
            require_parameter_slots(owner, slots)
            for changed in (slots[:-1], slots + [("arbitrary_json", None)], slots + slots[:1]):
                with self.assertRaises(GraphError):
                    require_parameter_slots(owner, changed)
            self.assertEqual(adapter_for_slot(graph, owner, "output", "$return", reference(10, reference(index))), MODULE + adapter)
            for ty in (reference(12), reference(10, reference(24)), primitive("f64"), {"tuple": [reference(index), primitive("f64")]}):
                with self.subTest(owner=owner, ty=ty), self.assertRaises(GraphError):
                    adapter_for_slot(graph, owner, "output", "$return", ty)
            with self.assertRaises(GraphError):
                adapter_for_slot(graph, owner, "input", "source", reference(index))
            text = {"borrowed_ref": {"type": primitive("str"), "is_mutable": False, "lifetime": None}}
            self.assertIsNone(adapter_for_slot(graph, owner, "input", "source", text))
            for label, ty in (("arbitrary_json", text), ("source", primitive("f64")), ("source", reference(12))):
                with self.assertRaises(GraphError):
                    adapter_for_slot(graph, owner, "input", label, ty)
        self.assertEqual(adapter_for_slot(graph, "chelis_python::eval_json", "input", "bindings_json", reference(24)), MODULE + "EvalBindingsJson")
        for ty in (reference(12), reference(20), primitive("str")):
            with self.assertRaises(GraphError):
                adapter_for_slot(graph, "chelis_python::eval_json", "input", "bindings_json", ty)

    def test_compiled_conversion_ownership_cannot_be_a_named_helper(self):
        raw = conversion_fixture()
        self.assertEqual(len(validate_conversion_calls(raw)), 5)
        for field, wrong in (("crate", "chelis_compiler_api"), ("item_name", "impostor"),
                             ("path", "::compiler_json"), ("def_id", "1:0")):
            changed = copy.deepcopy(raw)
            changed["crate"][field] = wrong
            with self.subTest(root_field=field), self.assertRaises(GraphError):
                validate_conversion_calls(changed)
        for mutation in ("helper", "trait", "root", "duplicate", "missing", "callee", "extra"):
            changed = copy.deepcopy(raw)
            call = changed["calls"][0]
            if mutation == "helper": call["caller"]["implementation"] = None
            elif mutation == "trait": call["caller"]["implementation"]["trait"] = definition("impostor::IntoPyObject")
            elif mutation == "root": call["payloads"][0]["shape"] = nominal(API + "CompileResult")
            elif mutation == "duplicate": changed["calls"].append(copy.deepcopy(call))
            elif mutation == "missing": changed["calls"].pop()
            elif mutation == "callee": call["callee"] = definition("local::to_string")
            else: call["caller"]["implementation"]["self_type"]["shape"] = nominal(MODULE + "OtherJson")
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                validate_conversion_calls(changed)

    def test_eval_reader_ownership_requires_deserialization_into_the_tensor_map(self):
        raw = conversion_fixture()
        self.assertEqual(len(validate_conversion_calls(raw)), 5)
        for mutation in ("missing", "serialize", "root", "map-value", "helper", "callee"):
            changed = copy.deepcopy(raw)
            call = changed["reader_calls"][0]
            if mutation == "missing": changed["reader_calls"] = []
            elif mutation == "serialize": changed["deserialize_trait"] = definition("serde_core::ser::Serialize")
            elif mutation == "root": call["payloads"][0]["shape"] = nominal(API + "TensorValue")
            elif mutation == "map-value": call["payloads"][0]["shape"]["arguments"][1] = nominal("serde_json::value::Value")
            elif mutation == "helper": call["caller"]["definition"]["item_name"] = "decode"
            else: call["callee"] = definition("serde_json::de::from_slice")
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                validate_conversion_calls(changed)

    def test_compiler_json_authority_stays_inside_its_adapter_subtree(self):
        for index, (adapter, _) in enumerate(list(OUTPUTS.values()) + [("EvalBindingsJson", "TensorValue")], 20):
            a, api = graph_fixture()
            self.assertTrue(adapter_payload(RustdocGraph([a.doc, api.doc]), MODULE + adapter).numeric_leaves)
            for mutation in ("field", "public", "extra", "moved", "generic"):
                changed = copy.deepcopy(a.doc)
                item = changed["index"][str(index)]
                body = item["inner"]["struct"]
                field = changed["index"][str(body["kind"]["plain"]["fields"][0])]
                if mutation == "field": field["inner"]["struct_field"] = reference(12)
                elif mutation == "public": field["visibility"] = "public"
                elif mutation == "extra": body["kind"]["plain"]["fields"].append(body["kind"]["plain"]["fields"][0])
                elif mutation == "moved": item["span"]["filename"] = "elsewhere.rs"
                else: body["generics"]["params"] = [{"name": "T", "kind": {"type": {"bounds": [], "default": None, "is_synthetic": False}}}]
                with self.subTest(adapter=adapter, mutation=mutation), self.assertRaises(GraphError):
                    adapter_payload(RustdocGraph([changed, api.doc]), MODULE + adapter)

    def test_stale_or_supplied_receipts_cannot_issue_binding_authority(self):
        from capacity_census_bindings import discover_bindings
        from test_capacity_census_bindings import fixture, receipt
        a = fixture(primitive("bool"))
        for substitute in ({"authority": "TaggedTransport"}, object(), conversion_fixture()):
            with self.assertRaises(GraphError):
                discover_bindings([a.doc], ["probe"], [], {}, provenance=receipt(a, ["probe"], [], {}), compiler_json=substitute)
        forged = object.__new__(VerifiedCompilerJsonBindings)
        object.__setattr__(forged, "root", Path(__file__).resolve().parent.parent)
        object.__setattr__(forged, "source_sha256", "0" * 64)
        with self.assertRaisesRegex(GraphError, "stale"):
            forged.validate()

    def test_all_selected_native_cases_must_execute_without_skips(self):
        events = [{"type": "suite", "event": "started", "test_count": len(NATIVE_CASES)}]
        for name in NATIVE_CASES:
            events += [{"type": "test", "event": "started", "name": name}, {"type": "test", "event": "ok", "name": name}]
        events.append({"type": "suite", "event": "ok", "passed": len(NATIVE_CASES), "failed": 0, "ignored": 0, "measured": 0, "filtered_out": 0})
        self.assertEqual(validate_libtest_execution(NATIVE_CASES, events), tuple(sorted(NATIVE_CASES)))
        for mutation in ("missing", "skipped", "failed", "zero", "duplicate"):
            changed = copy.deepcopy(events)
            if mutation == "missing": del changed[2]
            elif mutation in ("skipped", "failed"): changed[2]["event"] = mutation
            elif mutation == "zero": changed[0]["test_count"] = 0
            else: changed.insert(3, copy.deepcopy(changed[2]))
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                validate_libtest_execution(NATIVE_CASES, changed)


if __name__ == "__main__":
    unittest.main()
