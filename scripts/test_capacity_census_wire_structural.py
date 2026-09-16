"""Structural wire roles are closed contracts, never numeric-name heuristics."""

from dataclasses import replace
import unittest

from capacity_census_graph import Definition, DiscoveredGraph, Edge, GraphError, Leaf


def fixture(contracts):
    definitions = tuple(
        Definition(
            c.field.rsplit(".", 1)[0],
            "struct",
            (),
            (),
            (),
            (Edge(c.field, c.type, ()),),
            "serde-derived",
        )
        for c in contracts
    )
    leaves = tuple(Leaf(c.field, c.primitive) for c in contracts)
    return DiscoveredGraph((), definitions, leaves, "fixture", 1)


class StructuralRoles(unittest.TestCase):
    def test_source_carriers_keep_their_closed_point_and_measured_range_shapes(self):
        from capacity_census_wire_structural import validate_source_carriers

        schema = "chelis_compiler_api::schema::"
        span = Definition(
            schema + "Span",
            "struct",
            (),
            (("deny_unknown_fields", True),),
            (("", (), ("plain", (("offset", ()), ("len", ())))),),
            tuple(
                Edge(schema + "Span." + f, ("primitive", "u64"), ())
                for f in ("offset", "len")
            ),
            "serde-derived",
        )
        diagnostic = Definition(
            schema + "DiagnosticSpan",
            "enum",
            (),
            (
                ("deny_unknown_fields", True),
                ("rename_all", "snake_case"),
                ("tag", "span"),
            ),
            (
                ("Range", (), ("struct", (("offset", ()), ("len", ())))),
                ("Point", (), ("struct", (("offset", ()),))),
            ),
            tuple(
                Edge(schema + "DiagnosticSpan::" + f, ("primitive", "u64"), ())
                for f in ("Range.offset", "Range.len", "Point.offset")
            ),
            "serde-derived",
        )
        graph = DiscoveredGraph((), (span, diagnostic), (), "fixture", 1)
        validate_source_carriers(graph)
        for changed in (
            replace(diagnostic, serde=(("tag", "metadata"),)),
            replace(
                diagnostic,
                layout=diagnostic.layout + (("Arbitrary", (), ("unit", ())),),
            ),
            replace(
                diagnostic,
                layout=(("Point", (), ("struct", (("offset", ()), ("len", ())))),),
            ),
            replace(diagnostic, codec="custom"),
            replace(diagnostic, parameters=("T",)),
        ):
            with self.assertRaisesRegex(GraphError, "source carrier"):
                validate_source_carriers(replace(graph, definitions=(span, changed)))
        with self.assertRaisesRegex(GraphError, "source carrier"):
            validate_source_carriers(
                replace(
                    graph, definitions=(replace(span, edges=span.edges[:1]), diagnostic)
                )
            )

    def test_all_decided_roles_require_the_exact_field_width_and_container(self):
        from capacity_census_wire_structural import (
            structural_contracts,
            validate_structural_fields,
        )

        contracts = structural_contracts()
        graph = fixture(contracts)
        self.assertEqual(validate_structural_fields(graph), contracts)
        for index, contract in enumerate(contracts):
            with self.subTest(field=contract.field):
                definition = graph.definitions[index]
                edge = definition.edges[0]
                for changed in (
                    replace(edge, type=("primitive", "f64")),
                    replace(edge, type=("container", "alloc::vec::Vec", (edge.type,))),
                    replace(edge, path=edge.path + "_renamed"),
                    replace(edge, serde=(("rename", "arbitrary"),)),
                ):
                    definitions = list(graph.definitions)
                    definitions[index] = replace(definition, edges=(changed,))
                    with self.assertRaisesRegex(GraphError, "structural field"):
                        validate_structural_fields(
                            replace(graph, definitions=tuple(definitions))
                        )

    def test_a_tag_or_reference_name_does_not_grant_a_structural_role(self):
        from capacity_census_wire_structural import structural_contracts

        by_field = {c.field: c for c in structural_contracts()}
        source = "chelis_compiler_api::schema::DiagnosticSpan::Point.offset"
        self.assertEqual(by_field[source].role, "source-byte-coordinate")
        self.assertEqual(by_field[source].primitive, "u64")
        local = (
            "chelis_compiler_api::schema::"
            "WireExtentWitnessSite::LocalAscriptionClaim.ascription_id"
        )
        self.assertEqual(by_field[local].role, "local-ascription-identity")
        self.assertEqual(by_field[local].primitive, "u64")
        for arbitrary in (
            "chelis_compiler_api::schema::Metadata::Value.value",
            "chelis_compiler_api::schema::DiagnosticSpan::Point.value",
            "chelis_compiler_api::schema::WireRtDim::InputAxis.extent",
            "chelis_compiler_api::schema::CheckResult.score",
        ):
            self.assertNotIn(arbitrary, by_field)

    def test_private_encoder_decoder_slots_share_their_public_contract(self):
        from capacity_census_wire_structural import structural_contracts

        by_field = {c.field: c for c in structural_contracts()}
        schema = "chelis_compiler_api::schema::"
        for helper in ("WireDagFields", "WireDagFieldsRef"):
            contract = by_field[schema + helper + ".roots"]
            self.assertEqual(contract.owner_field, schema + "WireDag.roots")
            self.assertEqual(contract.role, "dag-root")
        header = by_field[schema + "envelopes::VersionHeader.schema_version"]
        self.assertEqual(header.role, "execution-version-header")
        self.assertEqual(header.type[0], "container")

    def test_a_duplicate_or_missing_edge_is_not_a_second_admission_path(self):
        from capacity_census_wire_structural import (
            structural_contracts,
            validate_structural_fields,
        )

        graph = fixture(structural_contracts())
        with self.assertRaisesRegex(GraphError, "duplicate.*field"):
            validate_structural_fields(
                replace(graph, definitions=graph.definitions + graph.definitions[:1])
            )
        with self.assertRaisesRegex(GraphError, "structural field"):
            validate_structural_fields(
                replace(graph, definitions=graph.definitions[1:])
            )

    def test_every_role_requires_its_selected_acceptance_and_rejection_executions(self):
        from capacity_census_wire_adapters import ExecutionOutcome
        from capacity_census_wire_artifact import artifact_cases
        from capacity_census_wire_envelopes import (
            dag_cases,
            envelope_cases,
            metadata_reference_cases,
            result_reference_cases,
        )
        from capacity_census_wire_schema import report_cases
        from capacity_census_wire_sources import source_cases
        from capacity_census_wire_structural import validate_structural_execution

        cases = (
            source_cases()
            + dag_cases()
            + envelope_cases()
            + metadata_reference_cases()
            + result_reference_cases()
            + report_cases()
            + artifact_cases()
        )
        # Synthetic execution records test reconciliation only; they never mint
        # VerifiedSchemaCodecs or provide live codec/transport authority.
        outcomes = tuple(
            ExecutionOutcome(c.identity, True, True, "passed", "a" * 64) for c in cases
        )
        selected = validate_structural_execution(cases, outcomes)
        self.assertTrue(selected)
        for identity in selected:
            with self.subTest(identity=identity):
                missing = tuple(o for o in outcomes if o.identity != identity)
                with self.assertRaisesRegex(GraphError, "structural execution"):
                    validate_structural_execution(cases, missing)
                skipped = tuple(
                    replace(o, executed=False) if o.identity == identity else o
                    for o in outcomes
                )
                with self.assertRaisesRegex(GraphError, "structural execution"):
                    validate_structural_execution(cases, skipped)
        changed = [
            replace(c, expected={"invented": True})
            if c.identity == "Span/json/negative-offset"
            else c
            for c in cases
        ]
        with self.assertRaisesRegex(GraphError, "structural.*rejection"):
            validate_structural_execution(changed, outcomes)


if __name__ == "__main__":
    unittest.main()
