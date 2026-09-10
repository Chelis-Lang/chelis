"""Name schema metadata owners using compiler identity and ancestry.

This pure helper returns obligations, never a capacity or codec witness. Its
caller must already have bound the driver process/Cargo provenance and the local
declaration proof. The latter is essential for a generated local schema marker:
only that preceding proof establishes that its local declaration came from the
real derive machinery. An ancestry join does not establish macro provenance.

Every dynamic return and schema dispatch must acquire an exact graph owner.
Ordinary serialization calls acquire one only when their compiler ancestry lies
inside a verified JsonSchema implementation. Their payloads remain untouched;
for example an f64 serialized inside JsonSchema still needs independent numeric
admission and gets no permission from this mapping.
"""

from collections.abc import Collection

from capacity_census_wire_publication import CHECK_REPORT_PROTOCOL

_DEFINITION_KEYS = ("def_id", "def_path_hash", "stable_crate_id", "crate", "path")
_SHAPE_HELPER = "chelis_compiler_api::schema::execution::shape_schema"
_TENSOR_OWNER = "chelis_compiler_api::schema::TensorValue"


def check_report_publisher_owner(caller: dict, crate: dict) -> str | None:
    """Return the CheckResult owner only for the compiled inherent publisher.

    The method name alone is not authority: rustc must report the local crate,
    its stable crate identity, an inherent implementation, and the exact
    CheckResult nominal receiver. Formatter types do not participate in this
    ownership decision; the call's actual Serialize payload is checked later.
    """

    if not isinstance(caller, dict):
        raise ValueError("missing compiler report publisher")
    root = _definition(crate)
    definition = caller.get("definition")
    if not isinstance(definition, dict):
        raise ValueError("missing compiler report publisher definition")
    current = _definition(definition)
    if definition.get("item_name") != "to_report_json":
        return None
    if current[2:4] != root[2:4]:
        return None
    implementation = caller.get("implementation")
    if not isinstance(implementation, dict) or implementation.get("trait") is not None:
        return None
    try:
        nominal = implementation["self_type"]["nominal"]
        receiver_definition = _definition(nominal)
        receiver = _canonical(nominal)
    except (KeyError, TypeError, ValueError):
        return None
    if receiver_definition[2:4] != root[2:4]:
        return None
    if receiver != CHECK_REPORT_PROTOCOL.producer_owner:
        return None
    return receiver


def _definition(value):
    if not isinstance(value, dict) or any(not isinstance(value.get(k), str) for k in _DEFINITION_KEYS):
        raise ValueError("missing compiler definition identity")
    if not value["def_id"] or not value["def_path_hash"] or not value["stable_crate_id"]:
        raise ValueError("empty compiler definition identity")
    return tuple(value[k] for k in _DEFINITION_KEYS)


def _canonical(value):
    _definition(value)
    if value["path"] and not value["path"].startswith("::"):
        raise ValueError("noncanonical compiler definition path")
    return value["crate"] + value["path"]


def resolve_schema_owners(evidence: dict, graph_definition_identities: Collection[str]) -> dict[str, str]:
    """Return exact caller DefPathHash -> nominal graph definition identity.

    Unknown schema owners, missing parents, and conflicting ancestry are errors.
    The dedicated shape_schema edge names TensorValue only after both its exact
    path and its concrete schemars::schema::Schema return identity are checked.
    The caller still binds that edge to TensorValue's verified adapter.
    """
    required = {"schema_trait", "crate", "bodies", "calls", "codec_calls", "schema_calls", "dynamic_returns", "errors"}
    if not isinstance(evidence, dict) or not required <= evidence.keys():
        raise ValueError("missing compiler schema ownership evidence")
    if evidence["errors"]:
        raise ValueError("unresolved compiler publication evidence")
    schema = evidence["schema_trait"]
    if _canonical(schema) != "schemars::JsonSchema":
        raise ValueError("wrong defining schema trait")
    schema_identity = _definition(schema)
    root_identity = _definition(evidence["crate"])
    graph = set(graph_definition_identities)
    if any(not isinstance(identity, str) or "::" not in identity for identity in graph):
        raise ValueError("invalid graph definition identity")
    definitions, bodies, callers, records = {}, set(), {}, {}

    def intern(definition):
        identity = _definition(definition)
        key = definition["def_path_hash"]
        if key in definitions and definitions[key] != identity:
            raise ValueError("conflicting compiler definition identity")
        definitions[key] = identity
        return key

    root_hash = intern(evidence["crate"])
    for body in evidence["bodies"]:
        key = intern(body)
        if key in bodies:
            raise ValueError("duplicate compiler body")
        bodies.add(key)

    must_resolve, serde_callers, report_publishers = set(), set(), set()
    for section in ("schema_calls", "dynamic_returns", "calls", "codec_calls"):
        if not isinstance(evidence[section], list):
            raise ValueError("invalid schema call section")
        for record in evidence[section]:
            try:
                caller = record["caller"]
                key = intern(caller["definition"])
                ancestors = tuple(intern(parent) for parent in caller["ancestors"])
            except (KeyError, TypeError) as error:
                raise ValueError("missing compiler caller/ancestry") from error
            if key not in bodies:
                raise ValueError("caller absent from compiler body census")
            if not ancestors or ancestors[-1] != root_hash or len(set((key, *ancestors))) != len(ancestors) + 1:
                raise ValueError("missing, cyclic or unordered compiler ancestry")
            if key in callers and callers[key] != ancestors:
                raise ValueError("conflicting compiler ancestry")
            callers[key] = ancestors
            records.setdefault(key, []).append(record)
            if section in ("schema_calls", "dynamic_returns"):
                must_resolve.add(key)
            else:
                serde_callers.add(key)
            if section == "calls" and check_report_publisher_owner(
                caller, evidence["crate"]
            ) is not None:
                report_publishers.add(key)
    # Every observed ancestor body must agree with the suffix claimed by its
    # child. The source compiler supplied the remaining (non-body) item links.
    for ancestors in callers.values():
        for index, ancestor in enumerate(ancestors):
            if ancestor in callers and callers[ancestor] != ancestors[index + 1:]:
                raise ValueError("broken compiler parent linkage")

    anchors, schema_scopes = {}, set()
    for key, observations in records.items():
        owners = set()
        for record in observations:
            implementation = record["caller"].get("implementation")
            if implementation is None or implementation.get("trait") is None:
                continue
            trait = implementation["trait"]
            if _definition(trait) != schema_identity:
                continue
            schema_scopes.add(key)
            try:
                nominal = implementation["self_type"]["nominal"]
                canonical = _canonical(nominal)
            except (KeyError, TypeError) as error:
                raise ValueError("schema implementation has no nominal owner") from error
            if canonical in graph:
                owners.add(canonical)
        if len(owners) > 1:
            raise ValueError("conflicting schema implementation owners")
        if owners:
            anchors[key] = owners.pop()

    def owner_of(key):
        if key in report_publishers:
            return CHECK_REPORT_PROTOCOL.producer_owner
        if key in anchors:
            return anchors[key]
        for parent in callers[key]:
            if parent in anchors:
                return anchors[parent]
        # This is an exact additional owner edge, not a module/prefix rule.
        observations = records[key]
        if _canonical(observations[0]["caller"]["definition"]) == _SHAPE_HELPER:
            returns = [observation["type"] for observation in observations if "type" in observation]
            if not returns:
                raise ValueError("shape_schema has no concrete schema return observation")
            for result in returns:
                nominal = result.get("nominal") if isinstance(result, dict) else None
                if nominal is None or _canonical(nominal) != "schemars::schema::Schema":
                    raise ValueError("shape_schema has the wrong concrete schema return")
                if nominal["stable_crate_id"] != schema["stable_crate_id"]:
                    raise ValueError("shape_schema returns a foreign schema crate")
            if _TENSOR_OWNER not in graph:
                raise ValueError("shape_schema has no TensorValue graph owner")
            return _TENSOR_OWNER
        raise ValueError("unresolved schema publication owner: " + key)

    result = {}
    for key in sorted(must_resolve | schema_scopes | report_publishers):
        result[key] = owner_of(key)
    for key in sorted(serde_callers - result.keys()):
        if any(parent in schema_scopes or parent in anchors for parent in callers[key]):
            result[key] = owner_of(key)
    return result
