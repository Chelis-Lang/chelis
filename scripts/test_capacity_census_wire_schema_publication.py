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


if __name__ == "__main__":
    unittest.main()
