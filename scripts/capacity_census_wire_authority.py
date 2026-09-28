"""Bijection from a verified compiler wire graph to its final classifications.

The functions here prepare a classification plan. Only the schema verifier,
after actual codec and admission execution, may bind that plan into its private
receipt. Neither an editable baseline nor a static transport descriptor is an
input to this classifier.
"""

from dataclasses import dataclass

from capacity_census_graph import GraphError, Leaf


@dataclass(frozen=True)
class LeafClassification:
    leaf: Leaf
    authority: str
    contract: str


def reconcile_leaves(leaves, candidates):
    expected = set(leaves)
    if len(expected) != len(leaves):
        raise GraphError(
            "wire authority bijection contains duplicate discovered leaves"
        )
    matches = {}
    for candidate in candidates:
        if candidate.authority not in {"TaggedTransport", "NumericOperation"}:
            raise GraphError(f"nonfinal numeric wire authority {candidate.authority}")
        if candidate.leaf in matches:
            raise GraphError(
                f"wire authority bijection has multiple matches: {candidate.leaf}"
            )
        matches[candidate.leaf] = candidate
    if expected != matches.keys():
        missing = sorted(expected - matches.keys())
        surplus = sorted(matches.keys() - expected)
        raise GraphError(
            f"wire authority bijection mismatch; unclassified={missing}, absent={surplus}"
        )
    return tuple(matches[leaf] for leaf in sorted(expected))


def classification_plan(graph, canonical_graph, operations, structural):
    """Consume already checked canonical/schema shapes, never infer by a tag."""
    from capacity_census_wire_fixed_roles import (
        compiler_fixed_field_contracts,
        validate_fixed_number_fields,
    )
    from capacity_census_wire_schema import _FIXED, _TENSOR_WIRE
    from capacity_census_wire_structural import (
        validate_source_carriers,
        validate_structural_fields,
    )

    definitions = {d.identity: d for d in graph.definitions}
    fields = {e.path: e for d in graph.definitions for e in d.edges}
    candidates = []
    validate_source_carriers(graph)
    # Canonical membership comes from the separate exact stored-carrier graph,
    # including both private JSON and binary mirrors. Mere prefix resemblance
    # or an arbitrary application's tagged enum is never a candidate.
    for definition in canonical_graph.definitions:
        if definitions.get(definition.identity) != definition:
            raise GraphError("canonical carrier definition differs between wire graphs")
    for leaf in canonical_graph.numeric_leaves:
        candidates.append(
            LeafClassification(
                leaf, "TaggedTransport", "stored-value/" + leaf.primitive
            )
        )
    fixed = compiler_fixed_field_contracts()
    validate_fixed_number_fields(graph, fixed)
    for carrier, primitive in _FIXED.items():
        field = carrier + ".$number"
        edge = fields.get(field)
        definition = definitions.get(carrier)
        if (
            edge is None
            or edge.type != ("primitive", primitive)
            or edge.serde
            or definition is None
            or not definition.codec.startswith("schema-codec:")
        ):
            raise GraphError(f"fixed-number codec shape is unverified: {field}")
        roles = sorted({c.role for c in fixed if c.carrier == carrier})
        if not roles:
            raise GraphError(f"fixed-number codec has no decided field role: {carrier}")
        candidates.append(
            LeafClassification(
                Leaf(field, primitive),
                "TaggedTransport",
                "fixed-number/" + ",".join(roles),
            )
        )
    tensor_shape = _TENSOR_WIRE + ".shape"
    edge = fields.get(tensor_shape)
    if (
        edge is None
        or edge.type != ("container", "alloc::vec::Vec", (("primitive", "i64"),))
        or edge.serde
    ):
        raise GraphError("checked tensor shape representation changed")
    candidates.append(
        LeafClassification(
            Leaf(tensor_shape, "i64"), "TaggedTransport", "checked-tensor-shape"
        )
    )
    if structural != validate_structural_fields(graph):
        raise GraphError("structural wire contracts differ from checked field shapes")
    for contract in structural:
        candidates.append(
            LeafClassification(
                Leaf(contract.field, contract.primitive),
                "TaggedTransport",
                contract.role,
            )
        )
    for operation in operations:
        leaves = tuple(
            leaf for leaf in graph.numeric_leaves if leaf.path == operation.field
        )
        if len(leaves) != 1:
            raise GraphError(
                f"numeric operation field has no unique numeric leaf: {operation.field}"
            )
        candidates.append(
            LeafClassification(leaves[0], "NumericOperation", operation.atom)
        )
    return reconcile_leaves(graph.numeric_leaves, candidates)
