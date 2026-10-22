"""Schema metadata ownership names an owner; it never approves a payload."""

import copy
import unittest

from capacity_census_wire_schema_publication import resolve_schema_owners


def definition(name, number, crate="chelis_compiler_api", item_name=None):
    return {"def_id": f"0:{number}", "def_path_hash": f"hash-{crate}-{number}",
            "stable_crate_id": "compiler" if crate == "chelis_compiler_api" else crate,
            "crate": crate, "path": name, "item_name": item_name}


ROOT = definition("", 0)
SCHEMA = definition("::JsonSchema", 1, "schemars")
OWNER = definition("::schema::Diagnostic", 2)
METHOD = definition("::schema::an_impl::json_schema", 3)
CLOSURE = definition("::schema::an_impl::json_schema::closure", 4)
SHAPE = definition("::schema::execution::shape_schema", 5)
SCHEMA_TYPE = definition("::schema::Schema", 6, "schemars")
CHECK_REPORT = definition("::check_report::<impl>::to_report_json", 7, item_name="to_report_json")
CHECK_RESULT = definition("::schema::CheckResult", 8)
GRAPH = {"chelis_compiler_api::schema::Diagnostic", "chelis_compiler_api::schema::TensorValue"}


def caller(identity, *, implementation=None, ancestors=()):
    return {"definition": identity, "implementation": implementation,
            "ancestors": [*ancestors, ROOT], "substitutions": "[]"}


def record(owner, nominal=SCHEMA_TYPE):
    return {"caller": owner, "type": {"text": "display is not identity", "nominal": nominal},
            "source": {"span": "fixture.rs:1", "expansion": "compiler expansion"}}


def fixture():
    method = caller(METHOD, implementation={"trait": SCHEMA, "self_type": {"nominal": OWNER}})
    closure = caller(CLOSURE, ancestors=(METHOD,))
    return {"schema_trait": SCHEMA, "crate": ROOT, "bodies": [METHOD, CLOSURE],
            "calls": [], "codec_calls": [], "schema_calls": [],
            "dynamic_returns": [record(method), record(closure)], "errors": []}


PROJECTIONS = {
    "HexBits": "chelis_numeric_core::dtype_semantics::wire_codec::HexBits",
    "KeyBitsSchema": "chelis_numeric_core::dtype_semantics::wire_codec::KeyBits",
    "ScalarWire": "chelis_numeric_core::dtype_semantics::wire_codec::ScalarWire",
    "StorageWire": "chelis_numeric_core::dtype_semantics::wire_codec::StorageWire",
    "ScalarValueSchema": "chelis_numeric_core::dtype_semantics::ScalarValue",
    "TensorStorageSchema": "chelis_numeric_core::dtype_semantics::TensorStorage",
}
HELPERS = {
    "execution_storage_schema": "chelis_compiler_api::schema::TensorValue",
    "key_bits_schema": "chelis_numeric_core::dtype_semantics::wire_codec::KeyBits",
}


def projection_fixture(name):
    projection = definition("::schema::numeric::" + name, 20)
    method = definition(
        "::schema::numeric::{impl#0}::json_schema", 21, item_name="json_schema"
    )
    implementation = {"trait": SCHEMA, "self_type": {"nominal": projection}}
    value = {**fixture(), "bodies": [method],
             "dynamic_returns": [record(caller(method, implementation=implementation))]}
    return value, method


def key_variant_fixture():
    helper = definition("::schema::numeric::is_key_variant", 60)
    closure = definition("::schema::numeric::is_key_variant::{closure#0}", 61)
    option = definition("::option::Option", 62, "core")
    row = record(caller(closure, ancestors=(helper,)), nominal=option)
    row["type"]["shape"] = {
        "tag": "nominal", "definition": option,
        "arguments": [{"tag": "reference", "mutable": False, "inner": {
            "tag": "nominal", "definition": SCHEMA_TYPE, "arguments": [],
        }}],
    }
    return {**fixture(), "bodies": [helper, closure], "dynamic_returns": [row]}, closure

class SchemaPublicationControls(unittest.TestCase):
    def resolve(self, value):
        return resolve_schema_owners(value, GRAPH)

    def test_closure_joins_its_exact_compiler_parent_without_parsing_impl_names(self):
        value = fixture()
        expected = "chelis_compiler_api::schema::Diagnostic"
        self.assertEqual(self.resolve(value), {METHOD["def_path_hash"]: expected, CLOSURE["def_path_hash"]: expected})
        broken = copy.deepcopy(value)
        broken["dynamic_returns"][1]["caller"]["ancestors"] = [ROOT]
        with self.assertRaisesRegex(ValueError, "owner"):
            self.resolve(broken)

    def test_check_report_publisher_requires_its_compiled_inherent_receiver(self):
        publisher = caller(
            CHECK_REPORT,
            implementation={"trait": None, "self_type": {"nominal": CHECK_RESULT}},
        )
        value = fixture()
        value["bodies"].append(CHECK_REPORT)
        value["calls"].append({"caller": publisher})
        self.assertEqual(
            self.resolve(value)[CHECK_REPORT["def_path_hash"]],
            "chelis_compiler_api::schema::CheckResult",
        )
        for mutation in ("item", "receiver", "receiver_crate", "trait", "crate"):
            changed = copy.deepcopy(value)
            report = changed["calls"][0]["caller"]
            if mutation == "item":
                report["definition"]["item_name"] = "to_string_pretty"
            elif mutation == "receiver":
                report["implementation"]["self_type"]["nominal"]["path"] = "::schema::Other"
            elif mutation == "receiver_crate":
                report["implementation"]["self_type"]["nominal"]["stable_crate_id"] = "foreign"
            elif mutation == "trait":
                report["implementation"]["trait"] = SCHEMA
            else:
                report["definition"]["stable_crate_id"] = "foreign"
            with self.subTest(mutation=mutation):
                try:
                    owners = self.resolve(changed)
                except ValueError:
                    continue
                self.assertNotIn(CHECK_REPORT["def_path_hash"], owners)

    def test_wrong_trait_and_missing_graph_owner_are_rejected(self):
        for mutate in (
            lambda v: v["dynamic_returns"][0]["caller"]["implementation"]["trait"].update(crate="lookalike"),
            lambda v: v["dynamic_returns"][0]["caller"]["implementation"].update(trait=None),
            lambda v: v["dynamic_returns"][0]["caller"]["implementation"]["self_type"]["nominal"].update(path="::unowned::Diagnostic"),
        ):
            value = copy.deepcopy(fixture())
            mutate(value)
            with self.assertRaises(ValueError):
                self.resolve(value)

    def test_missing_conflicting_or_cyclic_ancestry_is_rejected(self):
        for parents in ([ROOT, METHOD], [METHOD, METHOD, ROOT], [CLOSURE, METHOD, ROOT]):
            value = copy.deepcopy(fixture())
            value["dynamic_returns"][1]["caller"]["ancestors"] = parents
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                self.resolve(value)
        value = copy.deepcopy(fixture())
        value["dynamic_returns"][1]["caller"]["ancestors"][0] = {
            **value["dynamic_returns"][1]["caller"]["ancestors"][0], "def_id": "0:999"}
        with self.assertRaises(ValueError):
            self.resolve(value)

    def test_generated_local_marker_inherits_only_a_verified_containing_owner(self):
        value = fixture()
        marker = definition("::schema::an_impl::json_schema::marker", 8)
        marker_method = definition("::schema::an_impl::json_schema::marker_impl::json_schema", 9)
        value["bodies"].append(marker_method)
        value["dynamic_returns"].append(record(caller(marker_method, ancestors=(METHOD,),
            implementation={"trait": SCHEMA, "self_type": {"nominal": marker}})))
        self.assertEqual(self.resolve(value)[marker_method["def_path_hash"]], "chelis_compiler_api::schema::Diagnostic")
        value["dynamic_returns"][-1]["caller"]["ancestors"] = [ROOT]
        with self.assertRaisesRegex(ValueError, "owner"):
            self.resolve(value)

    def test_schema_calls_and_scoped_serde_calls_keep_their_numeric_payload(self):
        value = fixture()
        call = {"caller": value["dynamic_returns"][1]["caller"], "payloads": [{"text": "f64"}]}
        value["calls"].append(call)
        value["schema_calls"].append({"caller": value["dynamic_returns"][0]["caller"]})
        before = copy.deepcopy(value)
        self.assertIn(CLOSURE["def_path_hash"], self.resolve(value))
        self.assertEqual(value, before)
        other = definition("::unowned", 10)
        value["bodies"].append(other)
        value["schema_calls"].append({"caller": caller(other)})
        with self.assertRaisesRegex(ValueError, "owner"):
            self.resolve(value)

    def test_binary_calls_are_left_for_the_binary_owner_discharge(self):
        value = fixture()
        binary = definition("::cache_envelope::save", 11)
        value["bodies"].append(binary)
        value["calls"].append({"caller": caller(binary)})
        self.assertNotIn(binary["def_path_hash"], self.resolve(value))

    def test_shape_helper_requires_exact_path_schema_type_and_tensor_owner(self):
        value = {**fixture(), "bodies": [SHAPE], "dynamic_returns": [record(caller(SHAPE))]}
        self.assertEqual(self.resolve(value), {SHAPE["def_path_hash"]: "chelis_compiler_api::schema::TensorValue"})
        for mutation in ("path", "type", "graph"):
            changed = copy.deepcopy(value)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                if mutation == "path":
                    changed["dynamic_returns"][0]["caller"]["definition"]["path"] += "_copy"
                if mutation == "type":
                    changed["dynamic_returns"][0]["type"]["nominal"]["path"] = "::Value"
                resolve_schema_owners(changed, set() if mutation == "graph" else GRAPH)

    def test_empty_or_unresolved_evidence_cannot_be_an_owner_proof(self):
        for value in ({}, {**fixture(), "errors": ["unresolved dispatch"]}, {**fixture(), "schema_trait": None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.resolve(value)

    def test_shape_helper_joins_schema_calls_with_its_required_return_observation(self):
        value = {**fixture(), "bodies": [SHAPE], "schema_calls": [{"caller": caller(SHAPE)}],
                 "dynamic_returns": [record(caller(SHAPE))]}
        self.assertEqual(self.resolve(value), {SHAPE["def_path_hash"]: "chelis_compiler_api::schema::TensorValue"})
        value["dynamic_returns"] = []
        with self.assertRaisesRegex(ValueError, "return"):
            self.resolve(value)

    def test_shape_helper_validates_every_return_observation(self):
        value = {**fixture(), "bodies": [SHAPE],
                 "dynamic_returns": [record(caller(SHAPE)), copy.deepcopy(record(caller(SHAPE)))]}
        self.assertEqual(self.resolve(value), {SHAPE["def_path_hash"]: "chelis_compiler_api::schema::TensorValue"})
        for mutation in ("nominal", "crate"):
            changed = copy.deepcopy(value)
            nominal = changed["dynamic_returns"][1]["type"]["nominal"]
            nominal["path" if mutation == "nominal" else "stable_crate_id"] = "::Value" if mutation == "nominal" else "foreign"
            for order, observations in enumerate((changed["dynamic_returns"], list(reversed(changed["dynamic_returns"])))):
                with self.subTest(mutation=mutation, order=order), self.assertRaises(ValueError):
                    self.resolve({**changed, "dynamic_returns": observations})

    def test_numeric_projections_name_only_their_exact_codec_graph_owners(self):
        for name, owner in PROJECTIONS.items():
            value, method = projection_fixture(name)
            with self.subTest(projection=name):
                self.assertEqual(
                    resolve_schema_owners(value, {owner}),
                    {method["def_path_hash"]: owner},
                )
                child = definition("::schema::numeric::closure", 22)
                value["bodies"].append(child)
                value["dynamic_returns"].append(record(caller(child, ancestors=(method,))))
                self.assertEqual(
                    resolve_schema_owners(value, {owner})[child["def_path_hash"]], owner
                )

    def test_projection_schema_id_dispatch_keeps_its_verified_implementation_owner(self):
        value, method = projection_fixture("ScalarWire")
        method["path"] = "::schema::numeric::{impl#0}::schema_id"
        method["item_name"] = "schema_id"
        value["schema_calls"] = [{"caller": value["dynamic_returns"][0]["caller"]}]
        value["dynamic_returns"] = []
        owner = PROJECTIONS["ScalarWire"]
        self.assertEqual(
            resolve_schema_owners(value, {owner}), {method["def_path_hash"]: owner}
        )
        value["schema_calls"][0]["caller"]["implementation"] = None
        with self.assertRaises(ValueError):
            resolve_schema_owners(value, {owner})

    def test_numeric_projections_reject_altered_trait_receiver_crate_and_return(self):
        for name, owner in PROJECTIONS.items():
            for mutation in ("name", "trait", "receiver_crate", "caller_crate",
                             "return", "return_crate", "graph", "missing_return"):
                value, _ = projection_fixture(name)
                row = value["dynamic_returns"][0]
                current = row["caller"]
                if mutation == "name":
                    current["implementation"]["self_type"]["nominal"]["path"] += "Copy"
                elif mutation == "trait":
                    current["implementation"]["trait"] = definition("::JsonSchema", 1, "other")
                elif mutation == "receiver_crate":
                    current["implementation"]["self_type"]["nominal"]["stable_crate_id"] = "other"
                elif mutation == "caller_crate":
                    current["definition"]["stable_crate_id"] = "other"
                elif mutation == "return":
                    row["type"]["nominal"] = definition("::Value", 30, "serde_json")
                elif mutation == "return_crate":
                    row["type"]["nominal"] = {**SCHEMA_TYPE, "stable_crate_id": "other"}
                elif mutation == "missing_return":
                    value["schema_calls"] = [{"caller": current}]
                    value["dynamic_returns"] = []
                with self.subTest(projection=name, mutation=mutation), self.assertRaises(ValueError):
                    resolve_schema_owners(value, set() if mutation == "graph" else {owner})

    def test_numeric_schema_helpers_need_exact_local_identity_return_and_owner(self):
        for name, owner in HELPERS.items():
            helper = definition("::schema::numeric::" + name, 40)
            value = {**fixture(), "bodies": [helper],
                     "schema_calls": [{"caller": caller(helper)}],
                     "dynamic_returns": [record(caller(helper))]}
            with self.subTest(helper=name):
                self.assertEqual(
                    resolve_schema_owners(value, {owner}), {helper["def_path_hash"]: owner}
                )
            for mutation in ("name", "crate", "return", "return_crate", "graph", "missing_return"):
                changed = copy.deepcopy(value)
                if mutation in ("name", "crate"):
                    field = "path" if mutation == "name" else "stable_crate_id"
                    for row in changed["schema_calls"] + changed["dynamic_returns"]:
                        row["caller"]["definition"][field] += "Copy"
                    changed["bodies"][0][field] += "Copy"
                elif mutation == "return":
                    changed["dynamic_returns"][0]["type"]["nominal"]["path"] = "::Value"
                elif mutation == "return_crate":
                    changed["dynamic_returns"][0]["type"]["nominal"]["stable_crate_id"] = "other"
                elif mutation == "missing_return":
                    changed["dynamic_returns"] = []
                with self.subTest(helper=name, mutation=mutation), self.assertRaises(ValueError):
                    resolve_schema_owners(changed, set() if mutation == "graph" else {owner})

    def test_numeric_schema_edges_check_each_return_without_changing_payloads(self):
        cases = []
        for name, owner in PROJECTIONS.items():
            value, _ = projection_fixture(name)
            cases.append((name, owner, value))
        for name, owner in HELPERS.items():
            helper = definition("::schema::numeric::" + name, 50)
            value = {**fixture(), "bodies": [helper],
                     "dynamic_returns": [record(caller(helper))]}
            cases.append((name, owner, value))
        for name, owner, value in cases:
            value["dynamic_returns"].append(copy.deepcopy(value["dynamic_returns"][0]))
            before = copy.deepcopy(value)
            with self.subTest(edge=name):
                resolve_schema_owners(value, {owner})
                self.assertEqual(value, before)
            value["dynamic_returns"][1]["type"]["nominal"] = None
            for observations in (value["dynamic_returns"], list(reversed(value["dynamic_returns"]))):
                with self.subTest(edge=name), self.assertRaises(ValueError):
                    resolve_schema_owners({**value, "dynamic_returns": observations}, {owner})

    def test_key_variant_child_needs_its_exact_helper_and_optional_schema_reference(self):
        value, child = key_variant_fixture()
        owner = PROJECTIONS["TensorStorageSchema"]
        self.assertEqual(
            resolve_schema_owners(value, {owner}), {child["def_path_hash"]: owner}
        )
        for mutation in ("child", "parent", "crate", "parent_crate", "parent_body",
                         "implementation", "option", "mutable", "mutable_type", "schema", "schema_crate",
                         "arguments", "missing_shape", "graph", "missing_return"):
            changed = copy.deepcopy(value)
            row = changed["dynamic_returns"][0]
            current = row["caller"]
            shape = row["type"]["shape"]
            if mutation == "child":
                current["definition"]["path"] += "Copy"
            elif mutation == "parent":
                current["ancestors"][0]["path"] += "Copy"
            elif mutation == "crate":
                current["definition"]["stable_crate_id"] = "other"
            elif mutation == "parent_crate":
                current["ancestors"][0]["stable_crate_id"] = "other"
            elif mutation == "parent_body":
                changed["bodies"] = changed["bodies"][1:]
            elif mutation == "implementation":
                current["implementation"] = {"trait": None}
            elif mutation == "option":
                shape["definition"]["path"] = "::result::Result"
            elif mutation == "mutable":
                shape["arguments"][0]["mutable"] = True
            elif mutation == "mutable_type":
                shape["arguments"][0]["mutable"] = 0
            elif mutation in ("schema", "schema_crate"):
                field = "path" if mutation == "schema" else "stable_crate_id"
                shape["arguments"][0]["inner"]["definition"] = {
                    **SCHEMA_TYPE, field: "::Value" if mutation == "schema" else "other"
                }
            elif mutation == "arguments":
                shape["arguments"].append(copy.deepcopy(shape["arguments"][0]))
            elif mutation == "missing_shape":
                del row["type"]["shape"]
            elif mutation == "missing_return":
                changed["schema_calls"] = [{"caller": current}]
                changed["dynamic_returns"] = []
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                resolve_schema_owners(changed, set() if mutation == "graph" else {owner})


if __name__ == "__main__":
    unittest.main()
