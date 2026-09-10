"""C6 fixed-number membership belongs to the field's decided domain."""

from dataclasses import replace
import unittest

from capacity_census_graph import Definition, DiscoveredGraph, Edge, GraphError


def definition(identity, *edges, kind="struct", parameters=()):
    return Definition(identity, kind, parameters, (), (), tuple(edges), "serde-derived")


def graph(*definitions):
    return DiscoveredGraph((), tuple(definitions), (), "fixture", 1)


class FixedFieldRoles(unittest.TestCase):
    def test_field_contract_accepts_its_carrier_and_rejects_unrelated_tagged_data(self):
        from capacity_census_wire_fixed_roles import (
            FixedFieldContract,
            validate_fixed_number_fields,
        )

        scalar = ("reference", "fixture::SourceFloat", ())
        contract = FixedFieldContract(
            "fixture::Literal.value", "fixture::SourceFloat", scalar, "source-float"
        )
        good = definition("fixture::Literal", Edge(contract.field, scalar, ()))
        validate_fixed_number_fields(graph(good), (contract,))
        arbitrary = definition(
            "fixture::Metadata", Edge("fixture::Metadata::Value.data", scalar, ())
        )
        with self.assertRaisesRegex(
            GraphError, "unregistered fixed-number field.*Metadata"
        ):
            validate_fixed_number_fields(graph(good, arbitrary), (contract,))
        with self.assertRaisesRegex(GraphError, "missing fixed-number field"):
            validate_fixed_number_fields(graph(arbitrary), (contract,))

    def test_aliases_are_transparent_but_generic_wrappers_cannot_hide_a_new_use(self):
        from capacity_census_wire_fixed_roles import (
            FixedFieldContract,
            validate_fixed_number_fields,
        )

        scalar = ("reference", "fixture::SourceFloat", ())
        optional = ("container", "core::option::Option", (scalar,))
        contract = FixedFieldContract(
            "fixture::Literal.value", "fixture::SourceFloat", optional, "source-float"
        )
        alias = definition(
            "fixture::Alias",
            Edge("fixture::Alias.$alias", optional, ()),
            kind="type_alias",
        )
        good = definition(
            "fixture::Literal",
            Edge(contract.field, ("reference", "fixture::Alias", ()), ()),
        )
        validate_fixed_number_fields(graph(alias, good), (contract,))
        hidden = definition(
            "fixture::Extra",
            Edge(
                "fixture::Extra.data", ("reference", "fixture::Generic", (scalar,)), ()
            ),
        )
        with self.assertRaisesRegex(
            GraphError, "unregistered fixed-number field.*Extra"
        ):
            validate_fixed_number_fields(graph(alias, good, hidden), (contract,))
        changed = replace(good, edges=(Edge(contract.field, scalar, ()),))
        with self.assertRaisesRegex(GraphError, "fixed-number field shape changed"):
            validate_fixed_number_fields(graph(alias, changed), (contract,))
        published_alias = replace(
            graph(alias, good),
            roots=(
                ("fixture::Published", ("reference", "fixture::Generic", (scalar,))),
            ),
        )
        with self.assertRaisesRegex(
            GraphError, "unregistered fixed-number field.*Published"
        ):
            validate_fixed_number_fields(published_alias, (contract,))

    def test_duplicate_roles_or_changed_carrier_do_not_supply_authority(self):
        from capacity_census_wire_fixed_roles import (
            FixedFieldContract,
            validate_fixed_number_fields,
        )

        scalar = ("reference", "fixture::SourceFloat", ())
        contract = FixedFieldContract(
            "fixture::Literal.value", "fixture::SourceFloat", scalar, "source-float"
        )
        good = definition("fixture::Literal", Edge(contract.field, scalar, ()))
        with self.assertRaisesRegex(GraphError, "duplicate fixed-number field"):
            validate_fixed_number_fields(graph(good), (contract, contract))
        bad = replace(good, edges=(Edge(contract.field, ("primitive", "f64"), ()),))
        with self.assertRaisesRegex(GraphError, "missing fixed-number field"):
            validate_fixed_number_fields(graph(bad), (contract,))


if __name__ == "__main__":
    unittest.main()
