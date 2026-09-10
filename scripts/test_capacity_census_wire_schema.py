"""Paired controls for actual fixed-number and execution schema adapters."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from capacity_census_graph import GraphError

ROOT = Path(__file__).resolve().parent.parent

VOCABULARY = [
    {"name": "f64", "width": 8, "kind": "float"},
    {"name": "int64", "width": 8, "kind": "integer"},
    {"name": "bool", "width": 1, "kind": "bool"},
]
ORDER = tuple(d["name"] for d in VOCABULARY)


class SchemaCases(unittest.TestCase):
    def test_required_span_decoder_is_confined_to_its_exact_optional_text_field(self):
        from capacity_census_wire_schema import _SchemaShapeGraph
        from test_capacity_census_graph import Artifact, primitive, reference

        artifact = Artifact("chelis_compiler_api")
        artifact.external(40, "core::option::Option")
        artifact.external(41, "alloc::string::String")
        field = artifact.field(
            "span_id", reference(40, reference(41)),
            attrs=('#[serde(deserialize_with = "require_explicit_span")]',),
        )
        artifact.struct(1, "WireDagNode", [field])
        artifact.doc["paths"]["1"]["path"] = [
            "chelis_compiler_api", "schema", "WireDagNode"
        ]
        graph = _SchemaShapeGraph([artifact.doc], VOCABULARY).discover_exports(
            "chelis_compiler_api"
        )
        self.assertFalse(graph.numeric_leaves)
        for mutation in ("owner", "field", "helper", "numeric", "container", "default"):
            changed = copy.deepcopy(artifact.doc)
            item = changed["index"][str(field)]
            if mutation == "owner":
                changed["paths"]["1"]["path"][-1] = "Unrelated"
            elif mutation == "field":
                item["name"] = "arbitrary_data"
            elif mutation == "helper":
                item["attrs"] = [{"other": '#[serde(deserialize_with = "custom")]'}]
            elif mutation == "numeric":
                item["inner"]["struct_field"] = reference(40, primitive("f64"))
            elif mutation == "container":
                item["inner"]["struct_field"] = reference(41)
            else:
                item["attrs"].append({"other": "#[serde(default)]"})
            with self.assertRaisesRegex(GraphError, "required span decoder"):
                _SchemaShapeGraph([changed], VOCABULARY).discover_exports(
                    "chelis_compiler_api"
                )

    def test_source_admission_selection_keeps_each_decided_field_obligation(self):
        from capacity_census_wire_materialization import (
            materialization_cases,
            source_field_evidence,
        )

        cases = {c.identity: c for c in materialization_cases()}
        for field, pairs in source_field_evidence().items():
            for accepted, rejected in pairs:
                self.assertIsNotNone(cases[accepted].expected, field)
                self.assertIsNone(cases[rejected].expected, field)

    def test_artifact_manifest_and_discriminator_have_independent_codec_pairs(self):
        from capacity_census_wire_artifact import artifact_cases

        cases = artifact_cases()
        for carrier in ("ArtifactAbiVersion", "CompiledArtifactManifest"):
            for codec in ("json", "construct"):
                selected = [
                    c for c in cases if c.carrier == carrier and c.codec == codec
                ]
                self.assertTrue(any(c.expected is not None for c in selected))
                self.assertTrue(any(c.expected is None for c in selected))
        self.assertTrue(
            any(c.rejection_contains == "artifact ABI version" for c in cases)
        )

    def test_boolean_default_does_not_admit_a_numeric_default_function(self):
        from capacity_census_wire_schema import _SchemaShapeGraph
        from test_capacity_census_graph import Artifact, primitive

        artifact = Artifact()
        field = artifact.field("check", primitive("bool"))
        artifact.doc["index"][str(field)]["attrs"] = [
            {"other": '#[serde(default = "default_true")]'}
        ]
        artifact.struct(1, "Request", [field])
        self.assertFalse(
            _SchemaShapeGraph([artifact.doc], VOCABULARY)
            .discover_exports("fixture")
            .numeric_leaves
        )
        for ty, predicate in (("i64", "default_true"), ("bool", "custom_default")):
            artifact.doc["index"][str(field)]["inner"]["struct_field"] = primitive(ty)
            artifact.doc["index"][str(field)]["attrs"] = [
                {"other": f'#[serde(default = "{predicate}")]'}
            ]
            with self.assertRaisesRegex(GraphError, "unsupported.*default"):
                _SchemaShapeGraph([artifact.doc], VOCABULARY).discover_exports(
                    "fixture"
                )

    def test_map_omission_follows_both_types_and_rejects_wrong_container(self):
        from capacity_census_wire_schema import _SchemaShapeGraph
        from test_capacity_census_graph import Artifact, primitive, reference

        artifact = Artifact()
        artifact.external(40, "alloc::collections::btree::map::BTreeMap")
        field = artifact.field(
            "scores", reference(40, primitive("u64"), primitive("f64"))
        )
        artifact.doc["index"][str(field)]["attrs"] = [
            {"other": '#[serde(default, skip_serializing_if = "BTreeMap::is_empty")]'}
        ]
        artifact.struct(1, "Map", [field])
        graph = _SchemaShapeGraph([artifact.doc], VOCABULARY).discover_exports(
            "fixture"
        )
        self.assertEqual(
            {leaf.primitive for leaf in graph.numeric_leaves}, {"u64", "f64"}
        )
        artifact.doc["index"][str(field)]["inner"]["struct_field"] = primitive("f64")
        with self.assertRaisesRegex(GraphError, "type-mismatched serde omission"):
            _SchemaShapeGraph([artifact.doc], VOCABULARY).discover_exports("fixture")

    def test_every_number_adapter_executes_json_binary_and_carrier_admission(self):
        from capacity_census_wire_schema import schema_cases

        cases = schema_cases(VOCABULARY, ORDER)
        self.assertEqual(len(cases), len({case.identity for case in cases}))
        for carrier in (
            "UnitInterval",
            "SourceFloat",
            "SourceInteger",
            "NonnegativeCount",
            "NonnegativeExtent",
        ):
            for codec in ("json", "binary", "scalar"):
                selected = [
                    case
                    for case in cases
                    if case.carrier == carrier and case.codec == codec
                ]
                self.assertTrue(
                    any(case.expected is not None for case in selected),
                    (carrier, codec),
                )
                self.assertTrue(
                    any(case.expected is None for case in selected), (carrier, codec)
                )
        source_zero = next(
            case for case in cases if case.identity == "SourceFloat/json/-0.0"
        )
        self.assertEqual(source_zero.expected["elements"], ["8000000000000000"])
        count = next(
            case
            for case in cases
            if case.identity == "NonnegativeCount/json/9007199254740993"
        )
        self.assertEqual(count.expected["elements"], [9007199254740993])

    def test_result_reference_maps_have_producer_consumer_ownership_pairs(self):
        from capacity_census_wire_envelopes import result_reference_cases

        cases = result_reference_cases()
        for carrier in ("LowerResult", "GradResult"):
            for codec in ("json", "construct"):
                selected = [
                    c for c in cases if c.carrier == carrier and c.codec == codec
                ]
                self.assertTrue(any(c.expected is not None for c in selected))
                self.assertTrue(any(c.expected is None for c in selected))

    def test_source_coordinates_preserve_point_empty_absent_and_checked_access(self):
        from capacity_census_wire_sources import source_cases

        cases = source_cases()
        for carrier in ("Span", "DiagnosticLocation"):
            selected = [c for c in cases if c.carrier == carrier]
            self.assertTrue(any(c.expected is not None for c in selected))
            self.assertTrue(any(c.expected is None for c in selected))
        self.assertTrue(
            any(c.codec == "slice" and c.rejection_contains == "UTF-8" for c in cases)
        )
        self.assertTrue(
            any(
                c.identity.endswith("/absent") and c.expected["extent"] is None
                for c in cases
            )
        )

    def test_envelope_matrix_requires_versions_before_values_and_exact_dispatch(self):
        from capacity_census_wire_envelopes import envelope_cases

        cases = envelope_cases()
        for carrier in ("EvalResult", "WireApiEnvelope<EvalResult>", "WireBatchResult"):
            selected = [c for c in cases if c.carrier == carrier]
            self.assertTrue(any(c.expected is not None for c in selected))
            self.assertTrue(any(c.expected is None for c in selected))
        self.assertTrue(any(c.rejection_contains == "schema_version" for c in cases))
        self.assertTrue(any("duplicate-bits" in c.identity for c in cases))

    def test_reports_execute_producer_consumer_and_owned_reference_checks(self):
        from capacity_census_wire_schema import report_cases

        cases = report_cases()
        for carrier in ("CheckResult", "OrderedInferredParameters"):
            for codec in ("json", "construct"):
                selected = [
                    c for c in cases if c.carrier == carrier and c.codec == codec
                ]
                self.assertTrue(any(c.expected is not None for c in selected))
                self.assertTrue(any(c.expected is None for c in selected))
        self.assertTrue(
            any(c.rejection_contains == "owning list position" for c in cases)
        )

    def test_tensor_and_numeric_subset_have_positive_negative_pairs(self):
        from capacity_census_wire_schema import schema_cases

        cases = schema_cases(VOCABULARY, ORDER)
        for carrier in ("NumericScalar", "TensorValue"):
            selected = [case for case in cases if case.carrier == carrier]
            self.assertTrue(any(case.expected is not None for case in selected))
            self.assertTrue(any(case.expected is None for case in selected))

    def test_runtime_reference_owner_matrix_covers_all_admission_entry_points(self):
        from capacity_census_wire_envelopes import dag_cases

        cases = {c.identity: c for c in dag_cases()}
        for codec in ("json", "construct", "admit"):
            for owner in ("expand", "reshape", "pad", "shrink", "stride"):
                prefix = f"WireDag/{codec}/owner-{owner}-"
                self.assertIsNotNone(cases[prefix + "node"].expected)
                self.assertIsNone(cases[prefix + "node-zero-slot"].expected)
                axis = cases[prefix + "input-axis"]
                self.assertEqual(
                    axis.expected is not None, owner in {"expand", "reshape"}
                )
            self.assertIsNotNone(cases[f"WireDag/{codec}/owner-shrink-to-end"].expected)
            full_axis = cases[f"WireDag/{codec}/owner-shrink-to-end"]
            self.assertEqual(
                full_axis.expected["nodes"][1]["op"]["bounds"][0][0],
                {"bound": "lit", "value": 0},
            )
            self.assertIsNone(
                cases[f"WireDag/{codec}/owner-shrink-to-end-start-one"].expected
            )
        for codec in ("json", "admit"):
            self.assertIsNone(cases[f"WireDag/{codec}/duplicate-bits"].expected)
        self.assertNotIn("WireDag/construct/duplicate-bits", cases)

    def test_literal_witness_matrix_covers_domains_provenance_and_mandatory_fields(self):
        from capacity_census_wire_envelopes import dag_cases

        cases = {case.identity: case for case in dag_cases()}
        for codec in ("json", "construct", "admit"):
            prefix = f"WireDag/{codec}/"
            good = cases[prefix + "extent-witness-owned"]
            self.assertEqual(
                good.expected["nodes"][1]["op"]["requirements"], [4, 4, 9]
            )
            self.assertEqual(good.expected["nodes"][2]["shape_deps"], [1])
            for name in (
                "shape-dep-self",
                "shape-dep-large",
                "shape-dep-negative",
                "shape-dep-float",
                "witness-negative-requirement",
                "witness-float-requirement",
                "witness-axis-out-of-range",
                "witness-no-input",
                "witness-extra-input",
                "witness-wrong-output",
                "witness-ranked-output",
                "missing-witness-parameter",
                "missing-witness-axis",
                "missing-witness-requirements",
                "missing-node-shape_deps",
                "missing-node-span_id",
                "missing-node-merged_spans",
            ):
                self.assertIsNone(cases[prefix + name].expected, name)


class ActualSchemaCodec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_schema import verify_schema_codecs

        cls.receipt = verify_schema_codecs(
            ROOT, ROOT / "target/agents/wire-codec-rustdoc"
        )
        report = ROOT / "target/coordination/schema-codec-execution.json"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(cls.receipt.execution_report(), indent=2) + "\n")
        cls.documents = [
            json.loads(cls.receipt.canonical.document),
            json.loads(cls.receipt.document),
            *(json.loads(d) for d in cls.receipt.imported_documents),
        ]

    def test_structural_roles_require_current_shape_and_actual_admission_observations(
        self,
    ):
        from capacity_census_wire_schema import SchemaWireGraph
        from capacity_census_wire_structural import (
            structural_contracts,
            structural_evidence,
        )

        graph = SchemaWireGraph(self.documents, self.receipt).schema_graph()
        self.assertEqual(self.receipt.structural, structural_contracts())
        required = {
            identity
            for pairs in structural_evidence().values()
            for pair in pairs
            for identity in pair
        }
        self.assertEqual(set(self.receipt.structural_executions), required)
        self.assertLessEqual(required, {o.identity for o in self.receipt.outcomes})
        self.assertEqual(self.receipt.graph_identity, graph.identity)
        report = self.receipt.execution_report()
        self.assertEqual(report["structural_executions"], sorted(required))
        self.assertEqual(
            {row["field"] for row in report["structural_roles"]},
            {c.field for c in structural_contracts()},
        )

    def test_classification_plan_is_bijective_with_the_executed_numeric_graph(self):
        from dataclasses import replace
        from capacity_census_graph import Leaf
        from capacity_census_wire_adapters import CanonicalWireGraph
        from capacity_census_wire_authority import classification_plan
        from capacity_census_wire_schema import SchemaWireGraph

        graph = SchemaWireGraph(self.documents, self.receipt).schema_graph()
        canonical = CanonicalWireGraph(
            self.documents, self.receipt.canonical
        ).canonical_graph()
        plan = classification_plan(
            graph, canonical, self.receipt.operations, self.receipt.structural
        )
        self.assertEqual(plan, self.receipt.classifications)
        self.assertEqual({c.leaf for c in plan}, set(graph.numeric_leaves))
        self.assertEqual(len(plan), len(graph.numeric_leaves))
        for arbitrary in (
            Leaf("chelis_compiler_api::schema::Metadata::Value.value", "f64"),
            Leaf("chelis_compiler_api::schema::DiagnosticSpan::Point.score", "f64"),
        ):
            with self.assertRaisesRegex(GraphError, "authority.*bijection"):
                classification_plan(
                    replace(graph, numeric_leaves=graph.numeric_leaves + (arbitrary,)),
                    canonical,
                    self.receipt.operations,
                    self.receipt.structural,
                )
        with self.assertRaisesRegex(GraphError, "authority.*bijection"):
            classification_plan(
                graph, canonical, self.receipt.operations[1:], self.receipt.structural
            )
        self.assertEqual(
            {
                c["leaf"]["path"]
                for c in self.receipt.execution_report()["classification_plan"]
            },
            {leaf.path for leaf in graph.numeric_leaves},
        )

    def test_all_nominal_serializers_are_graph_roots_or_explicit_binary_obligations(
        self,
    ):
        from capacity_census_wire_publication import COMPILER_BINARY_OWNERS
        from capacity_census_wire_schema import SchemaWireGraph

        engine = SchemaWireGraph(self.documents, self.receipt)
        publication = engine.publication_graph()
        nominal = engine.serialization_definitions("chelis_compiler_api")
        self.assertEqual({d.identity for d in publication.declarations}, set(nominal))
        reached = {d.identity for d in publication.graph.definitions}
        self.assertEqual(set(nominal) - reached, set(COMPILER_BINARY_OWNERS))
        self.assertEqual(self.receipt.declarations, publication.declarations)
        report = self.receipt.execution_report()
        self.assertEqual(
            {d["identity"] for d in report["serialization_definitions"]}, set(nominal)
        )

    def test_actual_source_execution_is_required_by_the_schema_receipt(self):
        from capacity_census_wire_materialization import materialization_cases

        report = self.receipt.execution_report()
        selected = {case.identity for case in materialization_cases()}
        observed = {row["identity"] for row in report["cases"]}
        self.assertTrue(selected <= observed)
        self.assertTrue(self.receipt.materialization_executions)
        self.assertTrue(set(self.receipt.materialization_executions) <= selected)
        self.assertIn(
            "SourceProgram/eval-deep/f32-no-double-round",
            self.receipt.materialization_executions,
        )

    def test_local_declaration_closure_binds_actual_compiler_provenance(self):
        report = self.receipt.execution_report()["local_declarations"]
        self.assertEqual(len(report["expanded_identity"]), 64)
        self.assertEqual(len(report["probe_sha256"]), 64)
        self.assertGreater(report["module_nominals"], 0)
        self.assertGreater(report["serde_derives"], 0)
        self.assertIn("serde_derive@", report["serde_package"])
        self.assertGreater(report["schema_derives"], 0)
        self.assertIn("schemars_derive@", report["schema_package"])

    def test_actual_consumers_have_framework_execution_evidence(self):
        report = self.receipt.execution_report()
        executions = report["consumer_executions"]
        self.assertEqual(len(executions), 2)
        expected = [
            "actual_facade_and_native_preserve_exact_values_and_reject_malformed_carriers"
        ]
        self.assertEqual(list(executions[0]["selected"]), expected)
        self.assertEqual(list(executions[0]["executed"]), expected)
        self.assertEqual(len(executions[0]["output_sha256"]), 64)
        report_tests = [
            "report_document_producer_and_consumer_preserve_typed_values",
            "report_document_producer_rejects_inconsistent_counts",
        ]
        self.assertEqual(list(executions[1]["selected"]), report_tests)
        self.assertEqual(list(executions[1]["executed"]), report_tests)
        self.assertEqual(len(executions[1]["output_sha256"]), 64)
        self.assertNotIn("Python facade execution", report["remaining"])

    def test_numeric_parameters_bind_the_actual_graph_and_operation_authority(self):
        from capacity_census_wire_operations import (
            operation_contracts,
            validate_numeric_operation_fields,
        )
        from capacity_census_wire_schema import SchemaWireGraph

        graph = SchemaWireGraph(self.documents, self.receipt).schema_graph()
        self.assertEqual(self.receipt.operations, operation_contracts())
        self.assertEqual(
            self.receipt.operations,
            validate_numeric_operation_fields(
                graph,
                self.receipt.operations,
                (ROOT / "spec/05-risc-primitives.md").read_text(),
            ),
        )
        report = self.receipt.execution_report()
        self.assertEqual(
            {row["field"] for row in report["numeric_operations"]},
            {row.field for row in self.receipt.operations},
        )

    def test_actual_fixed_number_and_execution_graph_requires_executed_domains(self):
        from capacity_census_wire_schema import SchemaWireGraph

        graph = SchemaWireGraph(self.documents, self.receipt).schema_graph()
        names = {leaf.path: leaf.primitive for leaf in graph.numeric_leaves}
        for name, primitive in (
            ("UnitInterval", "f64"),
            ("SourceFloat", "f64"),
            ("SourceInteger", "i64"),
            ("NonnegativeCount", "i64"),
            ("NonnegativeExtent", "i64"),
        ):
            self.assertEqual(
                names[f"chelis_compiler_api::schema::numbers::{name}.$number"],
                primitive,
            )
        self.assertEqual(
            names["chelis_compiler_api::schema::execution::TensorWire.shape"], "i64"
        )
        self.assertEqual(
            names["chelis_compiler_api::schema::WireInferredParameter.index"], "u64"
        )
        self.assertTrue(
            any(d.identity.endswith("::reports::ReportWire") for d in graph.definitions)
        )
        self.assertEqual(self.receipt.canonical.profile, "compiler-api")
        self.assertTrue(self.receipt.outcomes)
        self.assertTrue(
            all(
                row.executed and row.outcome == "passed"
                for row in self.receipt.outcomes
            )
        )

    def test_codec_relocation_preserves_shape_but_invalidates_saved_artifact_proof(self):
        from capacity_census_wire_schema import SchemaWireGraph, _SchemaShapeGraph

        documents = copy.deepcopy(self.documents)
        api = documents[1]
        moved = 0
        for item in api["index"].values():
            span = item.get("span")
            if span and span["filename"] == "crates/chelis-compiler-api/src/schema.rs":
                span["begin"][0] += 6
                span["end"][0] += 6
                moved += 1
        self.assertGreater(moved, 0)
        vocabulary = json.loads(self.receipt.canonical.vocabulary)
        original = _SchemaShapeGraph(self.documents, vocabulary).publication_graph()
        relocated = _SchemaShapeGraph(documents, vocabulary).publication_graph()
        self.assertEqual(original.graph.numeric_leaves, relocated.graph.numeric_leaves)
        self.assertEqual(original.identity, relocated.identity)
        # Matching structural identity never makes an old execution witness
        # authoritative for altered rustdoc, including relocated source spans.
        with self.assertRaisesRegex(GraphError, "does not bind this artifact"):
            SchemaWireGraph(documents, self.receipt)

    def test_complete_public_exports_include_metadata_requests_and_templates(self):
        from capacity_census_wire_schema import SchemaWireGraph

        publication = SchemaWireGraph(self.documents, self.receipt).publication_graph()
        candidates = {c.export: c for c in publication.candidates}
        self.assertIn("chelis_compiler_api::compiler::ExecutionDim", candidates)
        self.assertIn("chelis_compiler_api::schema::ChangeSignatureRequest", candidates)
        self.assertIn(
            "chelis_compiler_api::schema::CompiledArtifactManifest", candidates
        )
        self.assertEqual(
            candidates["chelis_compiler_api::ContextHash"].owner, "context-cache"
        )
        self.assertEqual(
            candidates["chelis_compiler_api::schema::ApiEnvelope"].parameters, ("T",)
        )
        definitions = {d.identity for d in publication.graph.definitions}
        self.assertIn("chelis_compiler_api::schema::ApiEnvelope", definitions)
        self.assertIn("chelis_compiler_api::schema::BatchRequest", definitions)
        self.assertEqual(self.receipt.publication_identity, publication.identity)
        with self.assertRaisesRegex(GraphError, "complete defining artifact set"):
            SchemaWireGraph(self.documents[:2], self.receipt)

    def test_actual_python_artifact_has_no_unowned_nominal_serde_protocol(self):
        from capacity_census_graph import RustdocGraph
        from capacity_census_wire_publication import require_shared_binding_protocols

        graph = RustdocGraph(self.documents)
        self.assertIn("chelis_python", graph.documents)
        require_shared_binding_protocols(graph)
        self.assertFalse(graph.serialization_definitions("chelis_python"))

    def test_fixed_adapter_cannot_promote_an_unrelated_field(self):
        from capacity_census_wire_schema import _SchemaShapeGraph

        documents = copy.deepcopy(self.documents)
        api = documents[1]
        source_field = next(
            i
            for i in api["index"].values()
            if i.get("name") == "value"
            and "SourceFloat" in json.dumps(i.get("inner", {}))
        )
        request = next(
            i
            for i in api["index"].values()
            if i.get("name") == "CheckRequest" and "struct" in i["inner"]
        )
        request["inner"]["struct"]["kind"]["plain"]["fields"].append(source_field["id"])
        with self.assertRaisesRegex(
            GraphError, "unregistered fixed-number field.*CheckRequest.value"
        ):
            _SchemaShapeGraph(
                documents, json.loads(self.receipt.canonical.vocabulary)
            ).schema_graph()

    def test_private_field_carrier_dtype_and_codec_identity_mutations_reject(self):
        from capacity_census_wire_schema import SchemaWireGraph, _SchemaShapeGraph

        for mutation in (
            "public-field",
            "bare-f64",
            "unknown-serde",
            "missing-decoder",
            "wrong-codec-source",
            "wrong-shape-width",
            "wrong-parameter-width",
            "public-parameter-list",
            "missing-report-mirror",
            "wrong-envelope-version-width",
            "missing-envelope-mirror",
            "wrong-response-discriminator",
            "removed-batch-kind",
            "weak-source-discriminator",
            "wrong-artifact-version-width",
            "wrong-artifact-version-vocabulary",
            "missing-artifact-version-header",
            "wrong-artifact-metadata-mirror",
        ):
            with self.subTest(mutation=mutation):
                documents = copy.deepcopy(self.documents)
                api = documents[1]
                number = next(
                    i for i in api["index"].values() if i.get("name") == "UnitInterval"
                )
                field = api["index"][str(number["inner"]["struct"]["kind"]["tuple"][0])]
                if mutation.startswith("wrong-artifact-version"):
                    version = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "ArtifactAbiVersion"
                    )
                    if mutation.endswith("width"):
                        version["attrs"] = [
                            {"other": '#[serde(try_from = "u64", into = "u64")]'}
                        ]
                    else:
                        api["index"][str(version["inner"]["enum"]["variants"][0])][
                            "name"
                        ] = "V2"
                elif mutation in {
                    "missing-artifact-version-header",
                    "wrong-artifact-metadata-mirror",
                }:
                    name = (
                        "ArtifactAbiHeader"
                        if mutation.startswith("missing")
                        else "CompiledArtifactManifestFields"
                    )
                    helper = next(
                        i for i in api["index"].values() if i.get("name") == name
                    )
                    field_id = helper["inner"]["struct"]["kind"]["plain"]["fields"][0]
                    api["index"][str(field_id)]["inner"]["struct_field"] = {
                        "primitive": "u32"
                    }
                elif mutation == "wrong-envelope-version-width":
                    header = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "VersionHeader"
                    )
                    member = api["index"][
                        str(header["inner"]["struct"]["kind"]["plain"]["fields"][0])
                    ]
                    member["inner"]["struct_field"]["resolved_path"]["args"][
                        "angle_bracketed"
                    ]["args"][0]["type"] = {"primitive": "u64"}
                elif mutation == "missing-envelope-mirror":
                    mirror = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "ExecutionFieldsRef"
                    )
                    del api["index"][str(mirror["id"])]
                elif mutation == "wrong-response-discriminator":
                    header = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "ResponseHeader"
                    )
                    member = api["index"][
                        str(header["inner"]["struct"]["kind"]["plain"]["fields"][0])
                    ]
                    member["inner"]["struct_field"] = {"primitive": "u8"}
                elif mutation == "removed-batch-kind":
                    header = next(
                        i for i in api["index"].values() if i.get("name") == "BatchKind"
                    )
                    header["inner"]["enum"]["variants"].pop()
                elif mutation == "weak-source-discriminator":
                    location = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "DiagnosticSpan"
                    )
                    location["attrs"] = [
                        {"other": '#[serde(tag = "span", rename_all = "snake_case")]'}
                    ]
                elif mutation == "wrong-parameter-width":
                    parameter = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "WireInferredParameter"
                    )
                    index = api["index"][
                        str(parameter["inner"]["struct"]["kind"]["plain"]["fields"][0])
                    ]
                    index["inner"]["struct_field"] = {"primitive": "usize"}
                elif mutation == "public-parameter-list":
                    ordered = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "OrderedInferredParameters"
                    )
                    api["index"][str(ordered["inner"]["struct"]["kind"]["tuple"][0])][
                        "visibility"
                    ] = "public"
                elif mutation == "missing-report-mirror":
                    mirror = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "ReportWire"
                    )
                    del api["index"][str(mirror["id"])]
                elif mutation == "public-field":
                    field["visibility"] = "public"
                elif mutation == "bare-f64":
                    field["inner"]["struct_field"] = {"primitive": "f64"}
                elif mutation == "unknown-serde":
                    number["attrs"] = [{"other": "#[serde(transparent)]"}]
                elif mutation == "wrong-shape-width":
                    tensor = next(
                        i
                        for i in api["index"].values()
                        if i.get("name") == "TensorWire"
                    )
                    shape = api["index"][
                        str(tensor["inner"]["struct"]["kind"]["plain"]["fields"][0])
                    ]
                    shape["inner"]["struct_field"]["resolved_path"]["args"][
                        "angle_bracketed"
                    ]["args"][0]["type"] = {"primitive": "u64"}
                else:
                    for impl_id in number["inner"]["struct"]["impls"]:
                        item = api["index"][str(impl_id)]
                        trait = item.get("inner", {}).get("impl", {}).get("trait")
                        if trait and trait["path"] == "Deserialize":
                            if mutation == "missing-decoder":
                                number["inner"]["struct"]["impls"].remove(impl_id)
                            else:
                                item["span"]["filename"] = "unverified.rs"
                            break
                with self.assertRaises(GraphError):
                    SchemaWireGraph(documents, self.receipt).schema_graph()
                with self.assertRaises(GraphError):
                    _SchemaShapeGraph(
                        documents, json.loads(self.receipt.canonical.vocabulary)
                    ).schema_graph()


if __name__ == "__main__":
    unittest.main()
