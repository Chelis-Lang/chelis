"""Exact operation authority for numeric wire parameters, C6 and spec/05."""

from dataclasses import replace
from pathlib import Path
import unittest

from capacity_census_graph import Definition, DiscoveredGraph, Edge, GraphError

ROOT = Path(__file__).resolve().parent.parent


class OperationFields(unittest.TestCase):
    def test_each_registered_parameter_has_an_existing_exact_operation_atom(self):
        from capacity_census_wire_operations import (
            operation_contracts,
            validate_operation_atoms,
        )

        contracts = operation_contracts()
        spec = (ROOT / "spec/05-risc-primitives.md").read_text()
        validate_operation_atoms(contracts, spec)
        self.assertEqual(len(contracts), len({c.field for c in contracts}))
        for contract in contracts:
            with self.subTest(field=contract.field):
                changed = spec.replace(
                    f"> **{contract.atom}**", f"See {contract.atom}", 1
                )
                with self.assertRaisesRegex(
                    GraphError, "normative operation definition"
                ):
                    validate_operation_atoms((contract,), changed)

    def test_descriptor_type_and_selected_atom_are_both_checked(self):
        from capacity_census_wire_operations import (
            operation_contracts,
            validate_numeric_operation_fields,
        )

        contracts = operation_contracts()
        spec = (ROOT / "spec/05-risc-primitives.md").read_text()
        definitions = tuple(
            Definition(
                c.field.rsplit(".", 1)[0],
                "enum",
                (),
                (),
                (),
                (Edge(c.field, c.type, ()),),
                "serde-derived",
            )
            for c in contracts
        )
        graph = DiscoveredGraph((), definitions, (), "fixture", 1)
        admitted = validate_numeric_operation_fields(graph, contracts, spec)
        self.assertEqual({c.field for c in admitted}, {c.field for c in contracts})
        first = definitions[0]
        bad = replace(
            first, edges=(replace(first.edges[0], type=("primitive", "f64")),)
        )
        with self.assertRaisesRegex(
            GraphError, "numeric operation field shape changed"
        ):
            validate_numeric_operation_fields(
                replace(graph, definitions=(bad, *definitions[1:])), contracts, spec
            )
        with self.assertRaisesRegex(GraphError, "duplicate numeric operation field"):
            validate_numeric_operation_fields(graph, (*contracts, contracts[0]), spec)
        with self.assertRaisesRegex(GraphError, "authority anchor"):
            validate_numeric_operation_fields(
                graph, (replace(contracts[0], anchor="invented operation"),), spec
            )


if __name__ == "__main__":
    unittest.main()
