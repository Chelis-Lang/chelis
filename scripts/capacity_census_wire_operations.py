"""Exact semantic registration of axis parameters and handled random seeds.

These registrations supply operation semantics, not transport membership. Codec,
owner/rank admission, and execution evidence are independent requirements.
"""

from dataclasses import dataclass
import re

from capacity_census_graph import GraphError


@dataclass(frozen=True)
class OperationContract:
    field: str
    type: tuple
    atom: str
    anchor: str


def operation_contracts():
    schema = "chelis_compiler_api::schema::"
    rows = (
        ("Argmax.axis", 15, "argmax_reduce(x, axis)", False),
        ("Argmin.axis", 16, "argmin_reduce(x, axis)", False),
        ("Count.axes", 29, "count(x, axes...)", True),
        ("Expand.axis", 65, "axis_movement(arguments...)", False),
        ("Gather.axis", 66, "indexed_tensor(arguments...)", False),
        ("MaxReduce.axis", 12, "max_reduce(x, axes...)", False),
        ("MinReduce.axis", 13, "min_reduce(x, axes...)", False),
        ("Permute.axes", 65, "axis_movement(arguments...)", True),
        ("ProdReduce.axis", 14, "prod_reduce(x, axes...)", False),
        ("Scatter.axis", 66, "indexed_tensor(arguments...)", False),
        ("ScatterAdd.axis", 66, "indexed_tensor(arguments...)", False),
        ("ScatterElements.axis", 66, "indexed_tensor(arguments...)", False),
        ("Shape.axis", 7, "axis-domain `i32`", False),
        ("Sum.axis", 30, "sum(x, axes...", False),
    )
    result = []
    for field, number, anchor, many in rows:
        ty = ("primitive", "i32")
        if many:
            ty = ("container", "alloc::vec::Vec", (ty,))
        result.append(
            OperationContract(
                schema + "WireRiscOp::" + field, ty, f"[05-OP-{number}]", anchor
            )
        )
    result.append(
        OperationContract(
            schema + "WireRtAxis::Lit.value",
            ("primitive", "i32"),
            "[05-OP-7]",
            "axis-domain `i32`",
        )
    )
    for owner, number in (("UniformLike", 8), ("Dropout", 37)):
        result.append(
            OperationContract(
                schema + "WireRiscOp::" + owner + ".seed",
                ("primitive", "u64"),
                f"[05-OP-{number}]",
                "handled seed" if number == 8 else "[05-RNG-1]",
            )
        )
    return tuple(sorted(result, key=lambda c: c.field))


def validate_operation_atoms(contracts, spec):
    definitions = {}
    current = None
    for line in spec.splitlines():
        match = re.match(r"^> \*\*(\[05-OP-[0-9]+\])\*\*", line)
        if match:
            current = match[1]
            if current in definitions:
                raise GraphError(f"duplicate normative operation definition {current}")
            definitions[current] = line
        elif current and line.startswith(">") and not line.startswith("> **["):
            definitions[current] += "\n" + line
        else:
            current = None
    for contract in contracts:
        if (
            not re.fullmatch(r"\[05-OP-[0-9]+\]", contract.atom)
            or contract.atom not in definitions
        ):
            raise GraphError(
                f"missing normative operation definition {contract.atom} for {contract.field}"
            )
        if not contract.anchor or contract.anchor not in definitions[contract.atom]:
            raise GraphError(
                f"unrelated operation authority anchor for {contract.field}: {contract.atom}"
            )


def validate_numeric_operation_fields(graph, contracts, spec):
    if len(contracts) != len({c.field for c in contracts}):
        raise GraphError("duplicate numeric operation field registration")
    validate_operation_atoms(contracts, spec)
    fields = {e.path: e.type for d in graph.definitions for e in d.edges}
    for contract in contracts:
        if fields.get(contract.field) != contract.type:
            raise GraphError(f"numeric operation field shape changed: {contract.field}")
    return tuple(contracts)
