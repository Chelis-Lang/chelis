"""Reconcile module-level serde definitions with their serialization owner.

This discovers graph inputs, not authority. The binary-owner records are retained
obligations for the separate cache codecs; they are never numeric exemptions. A
cache reached through a wire field remains in that wire's serialized graph.
Every concrete nominal definition is a graph root even if private or not
referenced by a public type. Function-local definitions and publication entry
points still require source/codec evidence beyond rustdoc's item inventory.
"""

from dataclasses import dataclass
import hashlib
import json

from capacity_census_graph import DiscoveredGraph, GraphError, RustdocGraph


@dataclass(frozen=True)
class SerializationCandidate:
    export: str
    identity: str
    owner: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class SerializationDefinition:
    identity: str
    owner: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class PublishedGraph:
    candidates: tuple[SerializationCandidate, ...]
    declarations: tuple[SerializationDefinition, ...]
    graph: DiscoveredGraph
    identity: str


@dataclass(frozen=True)
class PublicationProtocol:
    """One explicit producer/consumer route over an already-discovered DTO."""

    identity: str
    producer: str
    producer_owner: str
    consumer: str


# This is not a new wire kind. `to_report_json` publishes the already
# discovered CheckResult report through the existing WireCheckResult consumer.
CHECK_REPORT_PROTOCOL = PublicationProtocol(
    identity="compiler-check-report-json",
    producer="chelis_compiler_api::schema::CheckResult::to_report_json",
    producer_owner="chelis_compiler_api::schema::CheckResult",
    consumer="chelis_compiler_api::schema::WireCheckResult",
)


def require_check_report_protocol(definitions: dict[str, int]) -> PublicationProtocol:
    """Bind the report publisher to its existing concrete DTO endpoints."""

    missing = [
        identity
        for identity in (
            CHECK_REPORT_PROTOCOL.producer_owner,
            CHECK_REPORT_PROTOCOL.consumer,
        )
        if definitions.get(identity) != 0
    ]
    if missing:
        raise GraphError(
            "missing concrete check-report protocol endpoint: " + ", ".join(missing)
        )
    return CHECK_REPORT_PROTOCOL


def discover_published_graph(
    graph: RustdocGraph, crate: str, *, binary_owners: dict[str, str]
) -> PublishedGraph:
    candidates = []
    declarations = []
    roots = {}
    templates = set()
    seen_binary = set()

    def consider(ty):
        owner, item_id = ty["graph_export"]
        _, _, identity = graph._resolve(owner, item_id)
        item = graph._item(owner, item_id)
        kinds = {"struct", "enum", "type_alias"} & item["inner"].keys()
        if len(kinds) != 1:
            raise GraphError(f"unsupported serialization candidate {identity}")
        parameters = graph._parameters(item["inner"][next(iter(kinds))])
        codec_owner = binary_owners.get(identity, "wire")
        if identity in binary_owners:
            if not codec_owner or codec_owner == "wire" or parameters:
                raise GraphError(f"invalid exact binary ownership for {identity}")
            seen_binary.add(identity)
        elif parameters:
            templates.add(identity)
        else:
            roots[identity] = (owner, identity, ty)
        return identity, codec_owner, parameters

    for export, ty in sorted(graph.serialization_candidates(crate).items()):
        identity, codec_owner, parameters = consider(ty)
        candidates.append(
            SerializationCandidate(export, identity, codec_owner, parameters)
        )
    for identity, ty in sorted(graph.serialization_definitions(crate).items()):
        resolved, codec_owner, parameters = consider(ty)
        if resolved != identity:
            raise GraphError("serialized definition identity changed during resolution")
        declarations.append(SerializationDefinition(identity, codec_owner, parameters))
    missing = set(binary_owners) - seen_binary
    if missing:
        raise GraphError(f"absent binary candidate: {sorted(missing)}")
    discovered = graph._discover([roots[name] for name in sorted(roots)])
    missing_templates = templates - {d.identity for d in discovered.definitions}
    if missing_templates:
        raise GraphError(
            f"uninstantiated public/private serialization template: {sorted(missing_templates)}"
        )
    if not roots:
        raise GraphError("no concrete serialization roots")
    identity = hashlib.sha256(
        json.dumps(
            {
                "candidates": [
                    (c.export, c.identity, c.owner, c.parameters) for c in candidates
                ],
                "declarations": [
                    (d.identity, d.owner, d.parameters) for d in declarations
                ],
                "graph": discovered.identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return PublishedGraph(tuple(candidates), tuple(declarations), discovered, identity)


COMPILER_BINARY_OWNERS = {
    "chelis_compiler_api::cache_envelope::Envelope": "shared-cache-envelope",
    "chelis_compiler_api::context::CacheEnvelope": "context-cache",
    "chelis_compiler_api::context::ContextHash": "context-cache",
    "chelis_compiler_api::context::CacheIdentity": "context-cache",
    "chelis_compiler_api::context::CompiledContext": "context-cache",
    "chelis_compiler_api::context::CompiledContextWire": "context-cache",
    "chelis_compiler_api::stdlib_cache::StdLibContext": "stdlib-cache",
    "chelis_compiler_api::stdlib_cache::StdLibContextWire": "stdlib-cache",
    "chelis_compiler_api::library_cache::LibraryContext": "library-cache",
    "chelis_compiler_api::library_cache::LibraryContextWire": "library-cache",
}


def require_shared_binding_protocols(graph: RustdocGraph) -> None:
    """A binding-owned nominal codec needs shared-schema discovery first."""
    local = graph.serialization_definitions("chelis_python")
    if local:
        raise GraphError(
            "binding-local serialized protocol requires a shared wire owner: "
            + ", ".join(sorted(local))
        )
