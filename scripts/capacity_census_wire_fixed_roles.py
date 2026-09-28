"""Field-specific use of fixed JSON-number adapters under spec/10 §§3.3–3.5.

The adapter's private scalar proves its representation. These checks additionally
bind its use to a decided source, report, count, or extent field. They are shape
checks consumed by the execution-backed verifier, not authority witnesses.
"""

from dataclasses import dataclass

from capacity_census_graph import GraphError


@dataclass(frozen=True)
class FixedFieldContract:
    field: str
    carrier: str
    type: tuple
    role: str


def validate_fixed_number_fields(graph, contracts):
    expected = {c.field: c for c in contracts}
    if len(expected) != len(contracts):
        raise GraphError("duplicate fixed-number field contract")
    carriers = {c.carrier for c in contracts}
    definitions = {d.identity: d for d in graph.definitions}

    def substitute(ty, arguments):
        if ty[0] == "generic":
            if ty[1] not in arguments:
                raise GraphError("unresolved fixed-number alias parameter")
            return arguments[ty[1]]
        return tuple(
            substitute(x, arguments)
            if isinstance(x, tuple) and x and isinstance(x[0], str)
            else tuple(substitute(y, arguments) for y in x)
            if isinstance(x, tuple)
            else x
            for x in ty
        )

    def normalize(ty, active=frozenset()):
        if ty[0] == "reference":
            definition = definitions.get(ty[1])
            if definition and definition.kind == "type_alias":
                if ty[1] in active or len(definition.edges) != 1:
                    raise GraphError("unresolved fixed-number alias")
                arguments = dict(zip(definition.parameters, ty[2], strict=True))
                return normalize(
                    substitute(definition.edges[0].type, arguments), active | {ty[1]}
                )
            return ("reference", ty[1], tuple(normalize(arg, active) for arg in ty[2]))
        if ty[0] == "container":
            return ("container", ty[1], tuple(normalize(arg, active) for arg in ty[2]))
        if ty[0] == "tuple":
            return ("tuple", tuple(normalize(arg, active) for arg in ty[1]))
        if ty[0] in {"slice", "array", "borrow"}:
            return (*ty[:-1], normalize(ty[-1], active))
        return ty

    def used(ty):
        if ty[0] in {"reference", "container"}:
            if ty[0] == "reference" and ty[1] in carriers:
                yield ty[1]
            children = ty[2]
        elif ty[0] == "tuple":
            children = ty[1]
        elif ty[0] in {"slice", "array", "borrow"}:
            children = (ty[-1],)
        else:
            children = ()
        for child in children:
            yield from used(child)

    actual = {}
    for name, root_type in graph.roots:
        ty = normalize(root_type)
        found = set(used(ty))
        if found and not (ty[0] == "reference" and ty[1] in carriers and not ty[2]):
            actual[name + ".$root"] = (ty, found)
    for definition in graph.definitions:
        if definition.kind == "type_alias":
            continue
        for edge in definition.edges:
            # A generic template is checked at the use supplying its arguments;
            # concrete fields in that same template remain independently visible.
            ty = normalize(edge.type)
            found = set(used(ty))
            if found:
                actual[edge.path] = (ty, found)
    missing = set(expected) - actual.keys()
    if missing:
        raise GraphError(f"missing fixed-number field: {sorted(missing)}")
    extra = actual.keys() - expected.keys()
    if extra:
        raise GraphError(f"unregistered fixed-number field: {sorted(extra)}")
    for field, contract in expected.items():
        ty, found = actual[field]
        if ty != contract.type or found != {contract.carrier}:
            raise GraphError(f"fixed-number field shape changed: {field}")


def compiler_fixed_field_contracts():
    schema = "chelis_compiler_api::schema::"
    contracts = []

    def add(field, carrier, role, container=None):
        identity = schema + "numbers::" + carrier
        ty = ("reference", identity, ())
        if container:
            ty = ("container", container, (ty,))
        contracts.append(FixedFieldContract(field, identity, ty, role))

    for owner in ("CheckResult", "WireCheckResult", "reports::ReportWire"):
        add(schema + owner + ".score", "UnitInterval", "report-score")
        for name in ("typed_nodes", "untyped_nodes", "total_nodes"):
            add(schema + owner + "." + name, "NonnegativeCount", "report-count")
    for owner in ("Diagnostic", "WireDiagnostic"):
        add(schema + owner + ".severity", "UnitInterval", "diagnostic-severity")
    for name in ("parse", "structure", "names", "types"):
        add(schema + "FitnessComponents." + name, "UnitInterval", "fitness-component")
    for field in (
        "ChangeSignatureResult.rewritten_calls",
        "RenameResult.renamed_references",
    ):
        add(schema + field, "NonnegativeCount", "edit-count")
    option = "core::option::Option"
    add(
        schema + "CompileResult.peak_device_bytes_estimate",
        "NonnegativeCount",
        "byte-estimate",
        option,
    )
    for field in (
        "WireDeepAtom::Int.value",
        "WireLiteral::Int.value",
        "WireLiteral::TypedInt.value",
        "WireSurfExpr::TupleGet.index",
    ):
        add(schema + field, "SourceInteger", "source-integer")
    add(schema + "WireSurfExpr::Vmap.axis", "SourceInteger", "source-integer", option)
    for field in (
        "WireDeepAtom::Float.value",
        "WireLiteral::Float.value",
        "WireLiteral::TypedFloat.value",
    ):
        add(schema + field, "SourceFloat", "source-float")
    add(
        "chelis_compiler_api::compiler::ExecutionDim.size",
        "NonnegativeExtent",
        "extent",
        option,
    )
    add(schema + "WireDimInfo::Named.size", "NonnegativeExtent", "extent", option)
    for field in (
        "WireDimExpr::Concrete.value",
        "WireDimInfo::Lit.size",
        "WireInferredDim::Lit.size",
        "WireRiscOp::OneHot.vocab",
        "WireRtDim::Lit.value",
    ):
        add(schema + field, "NonnegativeExtent", "extent")
    for owner in ("ReduceWindow", "ReduceWindowGrad"):
        for field in ("window_shape", "strides"):
            add(
                schema + "WireRiscOp::" + owner + "." + field,
                "NonnegativeExtent",
                "window-extent",
                "alloc::vec::Vec",
            )
    add(
        schema + "WireRiscOp::ExtentWitness.requirements",
        "NonnegativeExtent",
        "literal-witness-requirement",
        "alloc::vec::Vec",
    )
    add(
        schema + "WireRiscOp::OrderedAdjointSum.groups",
        "NonnegativeCount",
        "adjoint-contribution-group-count",
        "alloc::vec::Vec",
    )
    return tuple(sorted(contracts, key=lambda c: c.field))
