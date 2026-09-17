"""Exact structural roles under spec/10 §§3, 3.3–3.4 and spec/11 §1.

This catalog associates discovered fields with their decided owner. It cannot
grant transport authority: the schema verifier must also execute the selected
codec and owner-admission controls. In particular, no spelling or enclosing tag
turns an unmatched numeric leaf into a structural field.
"""

from dataclasses import dataclass

from capacity_census_graph import GraphError


@dataclass(frozen=True)
class StructuralFieldContract:
    field: str
    type: tuple
    primitive: str
    role: str
    owner_field: str


def structural_contracts():
    schema = "chelis_compiler_api::schema::"
    u64 = ("primitive", "u64")
    u32 = ("primitive", "u32")
    vector = ("container", "alloc::vec::Vec", (u64,))
    mapping = (
        "container",
        "alloc::collections::btree::map::BTreeMap",
        (("atomic", "alloc::string::String"), u64),
    )
    contracts = []

    def add(field, role, ty=u64, owner=None, primitive="u64"):
        contracts.append(
            StructuralFieldContract(
                schema + field, ty, primitive, role, schema + (owner or field)
            )
        )

    for field in (
        "Span.offset",
        "Span.len",
        "DiagnosticSpan::Point.offset",
        "DiagnosticSpan::Range.offset",
        "DiagnosticSpan::Range.len",
    ):
        add(field, "source-byte-coordinate")
    add("EvaluatedRoot.node_id", "opaque-evaluated-node")
    add("WireDagNode.id", "dag-position")
    add("WireDagNode.inputs", "earlier-dag-node", vector)
    add("WireDagNode.shape_deps", "earlier-shape-dependency", vector)
    add(
        "WireExtentWitnessSite::LocalAscriptionClaim.ascription_id",
        "local-ascription-identity",
    )
    for owner in ("WireDag", "WireDagFields", "WireDagFieldsRef"):
        ty = (
            ("borrow", False, "'a", ("slice", u64)) if owner.endswith("Ref") else vector
        )
        add(owner + ".roots", "dag-root", ty, "WireDag.roots")
        add(
            owner + ".schema_version",
            "dag-version",
            u32,
            "WireDag.schema_version",
            "u32",
        )
    for owner in ("LowerResult", "envelopes::LowerFields", "envelopes::LowerFieldsRef"):
        ty = ("borrow", False, "'a", mapping) if owner.endswith("Ref") else mapping
        add(owner + ".named_roots", "lower-result-node", ty, "LowerResult.named_roots")
    for owner in ("GradResult", "envelopes::GradFields", "envelopes::GradFieldsRef"):
        for field in ("output_node", "grad_nodes_by_name", "forward_nodes_by_name"):
            ty = u64 if field == "output_node" else mapping
            if owner.endswith("Ref") and field != "output_node":
                ty = ("borrow", False, "'a", ty)
            add(owner + "." + field, "grad-result-node", ty, "GradResult." + field)
    add("WireFusedInput::External.index", "fused-external-input")
    add("WireFusedInput::PreviousStep.index", "earlier-fused-step")
    add("WireRtDim::InputAxis.tensor", "owning-input-axis-slot")
    add("WireRtDim::Node.input", "owning-input-extent-slot")
    for owner, variant, role in (
        ("WireInferredType", "Var", "inferred-type-variable"),
        ("WireInferredPrecision", "Var", "inferred-precision-variable"),
        ("WireInferredDim", "Var", "inferred-dimension-variable"),
        ("WireInferredDim", "Rank", "inferred-rank-variable"),
    ):
        add(owner + "::" + variant + ".id", role, u32, primitive="u32")
    add("WireInferredParameter.index", "inferred-parameter-position")
    for owner in (
        "EvalResult",
        "envelopes::ExecutionFields",
        "envelopes::ExecutionFieldsRef",
    ):
        add(
            owner + ".schema_version",
            "execution-version",
            u32,
            "EvalResult.schema_version",
            "u32",
        )
    add(
        "envelopes::VersionHeader.schema_version",
        "execution-version-header",
        ("container", "core::option::Option", (u32,)),
        "EvalResult.schema_version",
        "u32",
    )
    add(
        "artifact::ArtifactAbiVersion.$version",
        "artifact-abi-version",
        u32,
        "CompiledArtifactManifest.abi_version",
        "u32",
    )
    return tuple(sorted(contracts, key=lambda c: c.field))


def validate_structural_fields(graph):
    fields = {}
    for definition in graph.definitions:
        for edge in definition.edges:
            if edge.path in fields:
                raise GraphError(f"duplicate discovered field {edge.path}")
            fields[edge.path] = edge
    contracts = structural_contracts()
    if len(contracts) != len({c.field for c in contracts}):
        raise GraphError("duplicate structural field contract")
    for contract in contracts:
        edge = fields.get(contract.field)
        if edge is None or edge.type != contract.type or edge.serde:
            raise GraphError(f"structural field shape changed: {contract.field}")
    return contracts


def validate_source_carriers(graph):
    """Recognize the decided source shapes, including their nonnumeric tags."""
    schema = "chelis_compiler_api::schema::"
    definitions = {d.identity: d for d in graph.definitions}
    for name, kind, serde, layout, fields in (
        (
            "Span",
            "struct",
            {"deny_unknown_fields": True},
            (("", (), ("plain", (("offset", ()), ("len", ())))),),
            (".offset", ".len"),
        ),
        (
            "DiagnosticSpan",
            "enum",
            {"deny_unknown_fields": True, "rename_all": "snake_case", "tag": "span"},
            (
                ("Point", (), ("struct", (("offset", ()),))),
                ("Range", (), ("struct", (("offset", ()), ("len", ())))),
            ),
            ("::Point.offset", "::Range.offset", "::Range.len"),
        ),
    ):
        identity = schema + name
        definition = definitions.get(identity)
        if (
            definition is None
            or definition.kind != kind
            or definition.parameters
            or definition.codec != "serde-derived"
            or dict(definition.serde) != serde
            or tuple(sorted(definition.layout)) != layout
            or {e.path for e in definition.edges}
            != {identity + field for field in fields}
            or any(e.type != ("primitive", "u64") or e.serde for e in definition.edges)
        ):
            raise GraphError(
                f"source carrier has changed its decided shape: {identity}"
            )


def structural_evidence():
    """Per-role accepted/rejected observations, independent of any baseline."""

    def pairs(carrier, accepted, rejected, codecs=None):
        if codecs is None:
            codecs = (
                ("json", "construct", "admit")
                if carrier == "WireDag"
                else ("json", "construct")
            )
        return tuple(
            (f"{carrier}/{codec}/{accepted}", f"{carrier}/{codec}/{bad}")
            for codec in codecs
            for bad in rejected
        )

    evidence = {
        "source-byte-coordinate": (
            *pairs("Span", "range-0-0", ("negative-offset",), ("json",)),
            *pairs(
                "Span",
                "utf8-character",
                ("split-start", "split-end", "endpoint-overflow", "out-of-bounds"),
                ("slice",),
            ),
            *pairs(
                "DiagnosticLocation",
                "point",
                ("point-fabricated-extent",),
                ("json",),
            ),
            *pairs(
                "DiagnosticLocation",
                "empty-range",
                ("range-missing-extent",),
                ("json",),
            ),
        ),
        "opaque-evaluated-node": pairs(
            "EvaluatedRoot", "opaque-18446744073709551615", ("opaque--1",), ("json",)
        ),
        "dag-position": pairs("WireDag", "owned-reference", ("id-position",)),
        "earlier-dag-node": pairs(
            "WireDag", "owned-reference", ("self-input", "large-input")
        ),
        "dag-root": pairs("WireDag", "owned-reference", ("root-owner",)),
        "dag-version": pairs(
            "WireDag",
            "empty",
            (
                "version-None",
                "version-10",
                "version-11",
                "version-12",
                "version-13",
                "version-15",
            ),
        ),
        "earlier-shape-dependency": pairs(
            "WireDag",
            "extent-witness-owned",
            (
                "shape-dep-self",
                "shape-dep-large",
                "shape-dep-negative",
                "shape-dep-float",
            ),
        ),
        "local-ascription-identity": pairs(
            "WireDag",
            "local-ascription-owned",
            ("local-ascription-id-negative", "local-ascription-id-float"),
        ),
        "lower-result-node": pairs("LowerResult", "owned", ("named_roots-1",)),
        "grad-result-node": pairs(
            "GradResult",
            "owned",
            ("output_node-1", "grad_nodes_by_name-1", "forward_nodes_by_name-1"),
        ),
        "fused-external-input": pairs(
            "WireDag", "fused-owned", ("fused-0-external-1",)
        ),
        "earlier-fused-step": pairs(
            "WireDag",
            "fused-owned",
            ("fused-0-previous_step-0", "fused-1-previous_step-1"),
        ),
        "owning-input-axis-slot": pairs(
            "WireDag", "input-axis-owned", ("input-axis-slot-0", "input-axis-slot-2")
        ),
        "owning-input-extent-slot": pairs(
            "WireDag",
            "extent-node-owned",
            ("extent-node-source-int320", "extent-node-source-int641"),
        ),
        "inferred-parameter-position": pairs(
            "OrderedInferredParameters",
            "indices-(0, 1, 2)",
            ("indices-(1,)", "indices-(0, 0)"),
        ),
        "execution-version": (
            *pairs(
                "EvalResult",
                "empty",
                ("version-first-None", "version-first-2", "version-first-4"),
                ("json",),
            ),
            *pairs(
                "EvalResult",
                "producer-version-3",
                ("producer-version-2", "producer-version-4"),
                ("construct",),
            ),
        ),
        "execution-version-header": pairs(
            "EvalResult",
            "reordered",
            (
                "version-last-None",
                "version-last-2",
                "version-last-4",
                "duplicate-version",
            ),
            ("json",),
        ),
        "artifact-abi-version": pairs("ArtifactAbiVersion", "2", ("0", "1", "3")),
    }
    for carrier, tag, role in (
        ("WireInferredType", "var", "inferred-type-variable"),
        ("WireInferredPrecision", "var", "inferred-precision-variable"),
        ("WireInferredDim", "var", "inferred-dimension-variable"),
        ("WireInferredDim", "rank", "inferred-rank-variable"),
    ):
        evidence[role] = pairs(
            carrier, tag + "-4294967295", (tag + "-4294967296", tag + "-0.0"), ("json",)
        )
    evidence["owning-input-axis-slot"] += (
        *pairs(
            "WireDag",
            "owner-expand-input-axis",
            (
                "owner-pad-input-axis",
                "owner-shrink-input-axis",
                "owner-stride-input-axis",
                "owner-expand-input-axis-extra-input",
            ),
        ),
        *pairs(
            "WireDag",
            "owner-reshape-input-axis",
            ("owner-reshape-input-axis-extra-input",),
        ),
    )
    for owner in ("expand", "reshape", "pad", "shrink", "stride"):
        evidence["owning-input-extent-slot"] += pairs(
            "WireDag",
            f"owner-{owner}-node",
            (f"owner-{owner}-node-zero-slot", f"owner-{owner}-node-extra-input"),
        )
    return evidence


def validate_structural_execution(cases, outcomes):
    """Reconcile actual selected observations before a verifier mints evidence."""
    evidence = structural_evidence()
    if set(evidence) != {c.role for c in structural_contracts()}:
        raise GraphError("structural execution roles differ from the field contracts")
    selected = {c.identity: c for c in cases}
    executed = {o.identity: o for o in outcomes}
    if len(selected) != len(cases) or len(executed) != len(outcomes):
        raise GraphError("duplicate structural execution identity")
    required = set()
    for role, pairs in evidence.items():
        if not pairs:
            raise GraphError(f"empty structural execution for {role}")
        for accepted, rejected in pairs:
            for identity, positive in ((accepted, True), (rejected, False)):
                case = selected.get(identity)
                outcome = executed.get(identity)
                if (
                    case is None
                    or outcome is None
                    or not outcome.selected
                    or not outcome.executed
                    or outcome.outcome != "passed"
                    or len(outcome.observation_sha256) != 64
                ):
                    raise GraphError(
                        f"missing or failed structural execution {identity}"
                    )
                if (case.expected is not None) != positive:
                    raise GraphError(
                        f"structural acceptance/rejection changed for {identity}"
                    )
                required.add(identity)
    return tuple(sorted(required))
