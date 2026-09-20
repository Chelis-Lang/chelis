"""Discharge compiled serde calls against executed wire and cache owners.

The JSON payload graph and the native binary codecs are separate domains.
Naming a cache owner never classifies its numeric descendants as wire metadata.
The binary use contracts below retain key/comparison inputs as explicit native
codec obligations while the cache proof checks the changed shared value codecs.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from capacity_census_graph import GraphError
from capacity_census_typed import NUMERIC_PRIMITIVES
from capacity_census_wire_adapters import canonical, source_identity
from capacity_census_wire_publication import (
    CHECK_REPORT_PROTOCOL,
    COMPILER_BINARY_OWNERS,
    require_check_report_protocol,
)


def identity(definition):
    if not isinstance(definition, dict):
        raise GraphError("missing defining compiler identity")
    crate, path = definition.get("crate"), definition.get("path")
    if (
        not isinstance(crate, str)
        or not crate
        or not isinstance(path, str)
        or not path.startswith("::")
    ):
        raise GraphError("invalid defining compiler identity")
    return crate + path


def _definition_key(value):
    """Keep complete same-compilation identity when joining codec instances."""
    fields = ("def_id", "def_path_hash", "stable_crate_id", "crate", "path", "item_name")
    if (
        not isinstance(value, dict)
        or set(value) != set(fields)
        or any(not isinstance(value[k], str) or not value[k] for k in fields[:4])
        or not isinstance(value["path"], str)
        or (value["path"] and not value["path"].startswith("::"))
        or (value["item_name"] is not None and not isinstance(value["item_name"], str))
    ):
        raise GraphError("incomplete defining compiler identity for codec specialization")
    return tuple(value[k] for k in fields)


def _shape_key(shape, *, parameters=False, complete_identity=False):
    """Canonicalize compiler type terms; display strings are never authority."""

    def child(value):
        return _shape_key(value, parameters=parameters, complete_identity=complete_identity)

    if not isinstance(shape, dict):
        raise GraphError("missing compiler payload shape")
    tag = shape.get("tag")
    if tag == "primitive" and set(shape) == {"tag", "name"}:
        if shape["name"] in NUMERIC_PRIMITIVES | {"bool", "char", "str"}:
            return (tag, shape["name"])
    elif tag == "nominal" and set(shape) == {"tag", "definition", "arguments"}:
        if isinstance(shape["arguments"], list):
            return (
                tag,
                (_definition_key if complete_identity else identity)(shape["definition"]),
                tuple(child(a) for a in shape["arguments"]),
            )
    elif tag == "reference" and set(shape) == {"tag", "mutable", "inner"}:
        if type(shape["mutable"]) is bool:
            return (tag, shape["mutable"], child(shape["inner"]))
    elif tag == "slice" and set(shape) == {"tag", "element"}:
        return (tag, child(shape["element"]))
    elif tag == "array" and set(shape) == {"tag", "element", "length"}:
        if type(shape["length"]) is int and 0 <= shape["length"] <= (1 << 64) - 1:
            return (tag, shape["length"], child(shape["element"]))
    elif (
        tag == "tuple"
        and set(shape) == {"tag", "elements"}
        and isinstance(shape["elements"], list)
    ):
        return (tag, tuple(child(a) for a in shape["elements"]))
    elif (
        parameters
        and tag == "parameter"
        and set(shape) == {"tag", "index"}
        and type(shape["index"]) is int
        and shape["index"] >= 0
    ):
        return (tag, shape["index"])
    raise GraphError(f"unsupported or open compiled payload shape: {tag}")


def shape_key(shape):
    return _shape_key(shape)


def _nominal(name, *args):
    return ("nominal", name, tuple(args))


_GLOBAL = _nominal("alloc::alloc::Global")
_RANDOM_STATE = _nominal("std::hash::random::RandomState")
_CONTAINERS = {
    "alloc::vec::Vec": (1, (_GLOBAL,)),
    "alloc::boxed::Box": (1, (_GLOBAL,)),
    "alloc::collections::btree::map::BTreeMap": (2, (_GLOBAL,)),
    "std::collections::hash::map::HashMap": (2, (_RANDOM_STATE, _GLOBAL)),
    "core::option::Option": (1, ()),
    "core::result::Result": (2, ()),
}


def require_wire_payload(shape, definitions):
    """Allow only nonnumeric primitives or a fully discovered codec graph.

    A nominal's arguments are checked independently, so Response<f64> cannot
    borrow the registration of Response<NumericScalar>.
    """

    def visit(key):
        tag = key[0]
        if tag == "primitive":
            if key[1] in {"bool", "char", "str"}:
                return
            raise GraphError("bare numeric publication has no tagged carrier")
        if tag == "reference":
            return visit(key[2])
        if tag in {"slice", "array"}:
            return visit(key[-1])
        if tag == "tuple":
            for child in key[1]:
                visit(child)
            return
        _, name, arguments = key
        if name == "alloc::string::String" and not arguments:
            return
        if name in _CONTAINERS:
            arity, defaults = _CONTAINERS[name]
            if len(arguments) != arity + len(defaults) or arguments[arity:] != defaults:
                raise GraphError("unrecognized concrete serialization container")
            children = arguments[:arity]
        elif name in definitions and name not in COMPILER_BINARY_OWNERS:
            if len(arguments) != definitions[name]:
                raise GraphError(
                    "payload generic arguments differ from the discovered declaration"
                )
            children = arguments
        else:
            raise GraphError(f"unowned serialization payload: {name}")
        for child in children:
            visit(child)

    visit(shape_key(shape))


_API = "chelis_compiler_api::"
_NODE = _nominal("chelis_ir::dag::NodeId")
_TYPE = _nominal("chelis_ir::dag::TensorType")
_DEEP = _nominal("chelis_deep::ast::Expr")
_LOCAL_TENSOR_ASCRIPTION = _nominal(
    "chelis_types::infer::checked::CheckedLocalTensorAscription"
)
_BOOL = ("primitive", "bool")
_CHECK_REPORT = _nominal(CHECK_REPORT_PROTOCOL.producer_owner)
_CHECK_REPORT_SERIALIZE = "serde_core::ser::Serialize::serialize"


def _entries(value):
    pair = (
        "tuple",
        (("reference", False, ("primitive", "str")), ("reference", False, value)),
    )
    return _nominal("alloc::vec::Vec", pair, _GLOBAL)


# Exact production uses, not an allowlist of every serializable cache type.
# A different caller, payload, container, width or codec requires its own proof.
_BINARY_CALLS = {
    _API + "cache_envelope::save": (
        "shared-cache-envelope",
        {
            _nominal(_API + "cache_envelope::Envelope"),
            _nominal(_API + "library_cache::LibraryContext"),
            _nominal(_API + "stdlib_cache::StdLibContext"),
        },
    ),
    _API + "context::CompiledContext::envelope_bytes": (
        "context-cache",
        {
            _nominal(_API + "context::CompiledContext"),
            _nominal(_API + "context::CacheEnvelope"),
        },
    ),
    _API + "stdlib_cache::visit_stdlib_cache_key_inputs": (
        "native-stdlib-key-material",
        {("slice", _nominal("chelis_surf::ast::Decl"))},
    ),
    _API + "library_cache::visit_library_cache_key_inputs": (
        "native-library-key-material",
        {("slice", _nominal("chelis_surf::ast::Decl"))},
    ),
    _API + "library_cache::expanded_deep_digest": (
        "native-source-key-material",
        {("slice", _DEEP)},
    ),
    _API + "cache_envelope::sorted_map_bytes": (
        "native-lowered-payload-comparison",
        {_entries(_NODE)},
    ),
    _API + "cache_envelope::sorted_btree_map_bytes": (
        "native-lowered-payload-comparison",
        {_entries(t) for t in (_BOOL, _DEEP, _TYPE)},
    ),
    _API + "cache_envelope::lowered_library_payload_matches": (
        "native-lowered-payload-comparison",
        {
            # Comparison-only native codec uses. The cache publication proof
            # owns compatibility; neither payload acquires wire authority.
            _nominal("chelis_ir::dag::Dag"),
            ("slice", _LOCAL_TENSOR_ASCRIPTION),
        },
    ),
}


def binary_call_owner(caller, payloads):
    contract = _BINARY_CALLS.get(caller)
    if (
        contract is None
        or len(payloads) != 1
        or shape_key(payloads[0]) not in contract[1]
    ):
        raise GraphError(f"unowned binary serialization use: {caller}")
    return contract[0]


def _caller_identity(caller):
    implementation = caller.get("implementation")
    if implementation is not None and implementation.get("trait") is None:
        owner = identity(implementation["self_type"]["nominal"])
        name = caller["definition"].get("item_name")
        if not isinstance(name, str) or not name or "::" in name:
            raise GraphError("missing compiler inherent method identity")
        return owner + "::" + name
    return identity(caller["definition"])


def _local_binary_owner(call):
    caller, callee = _caller_identity(call["caller"]), identity(call["callee"])
    payloads = tuple(shape_key(p["shape"]) for p in call["payloads"])
    # CachePayload's Serialize bound also appears on its load edge. The cache
    # proof executes both operations for these two sealed payload owners.
    if callee in {_API + "cache_envelope::save", _API + "cache_envelope::load"}:
        expected = {
            _API + "library_cache::load_or_build_library_context": _nominal(
                _API + "library_cache::LibraryContext"
            ),
            _API + "stdlib_cache::load_or_build_stdlib_context": _nominal(
                _API + "stdlib_cache::StdLibContext"
            ),
        }
        if payloads == (expected.get(caller),):
            return "shared-cache-envelope"
    if caller == _API + "cache_envelope::lowered_library_payload_matches":
        expected = {
            _API + "cache_envelope::sorted_map_bytes": {_NODE},
            _API + "cache_envelope::sorted_btree_map_bytes": {_BOOL, _DEEP, _TYPE},
        }
        if len(payloads) == 1 and payloads[0] in expected.get(callee, set()):
            return "native-lowered-payload-comparison"
    raise GraphError(f"unowned local generic serialization edge: {caller} -> {callee}")


def check_report_call_owner(call, crate) -> str:
    """Replay the exact compiled CheckResult publisher and payload edge.

    This intentionally knows neither `ReportFormatter` nor a serde_json helper
    name. Those are renderer implementation details. The compiler must instead
    identify the inherent CheckResult publisher and the exact Serialize payload;
    the report's byte and consumer controls own renderer behavior.
    """

    from capacity_census_wire_schema_publication import check_report_publisher_owner

    try:
        publisher = check_report_publisher_owner(call["caller"], crate)
        callee = identity(call["callee"])
        payloads = tuple(shape_key(payload["shape"]) for payload in call["payloads"])
    except (KeyError, TypeError, ValueError) as error:
        raise GraphError("malformed check-report publication edge") from error
    if publisher != CHECK_REPORT_PROTOCOL.producer_owner:
        raise GraphError("unowned check-report publication caller")
    if call.get("local_callee") is not False or callee != _CHECK_REPORT_SERIALIZE:
        raise GraphError("unowned check-report serialization callee")
    if payloads != (_CHECK_REPORT,):
        raise GraphError("check-report publication has the wrong payload")
    return "check-report/" + publisher


def require_check_report_call(calls, crate) -> dict:
    """Require one compiled publication edge for the registered report route."""

    if not isinstance(calls, list):
        raise GraphError("missing check-report publication calls")
    from capacity_census_wire_schema_publication import check_report_publisher_owner

    publishers = [
        call
        for call in calls
        if isinstance(call, dict)
        and check_report_publisher_owner(call.get("caller"), crate)
        == CHECK_REPORT_PROTOCOL.producer_owner
    ]
    if len(publishers) != 1:
        raise GraphError("missing or duplicate compiled check-report publication edge")
    check_report_call_owner(publishers[0], crate)
    return publishers[0]



def _codec_site(row, serialize_trait):
    """Match a compiler-classified codec call site, never its surrounding name."""
    try:
        caller = row["caller"]
        implementation = caller["implementation"]
        trait = _definition_key(implementation["trait"])
        if trait != _definition_key(serialize_trait):
            raise GraphError("codec specialization has the wrong Serialize trait")
        receiver = implementation["self_type"]
        if receiver["shape"].get("tag") != "nominal" or (
            _definition_key(receiver["nominal"])
            != _definition_key(receiver["shape"]["definition"])
        ):
            raise GraphError("codec specialization has inconsistent nominal ownership")
        source = row["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"span", "expansion"}
            or any(not isinstance(value, str) or not value for value in source.values())
            or type(row["local_callee"]) is not bool
            or not isinstance(caller["ancestors"], list)
            or not caller["ancestors"]
        ):
            raise GraphError("missing compiler codec call-site provenance")
        return (
            _definition_key(caller["definition"]),
            tuple(_definition_key(parent) for parent in caller["ancestors"]),
            trait,
            _definition_key(row["callee"]),
            source["span"],
            source["expansion"],
            row["local_callee"],
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise GraphError("malformed compiler codec specialization") from error


def _codec_terms(row, *, parameters):
    try:
        values = [row["caller"]["implementation"]["self_type"]]
        for section in ("payloads", "serializers"):
            if not isinstance(row[section], list):
                raise GraphError("missing compiler codec payload or serializer obligations")
            values.extend(row[section])
        if not row["payloads"] and not row["serializers"]:
            raise GraphError("codec call has no payload or serializer obligation")
        return (
            len(row["payloads"]),
            len(row["serializers"]),
            tuple(
                _shape_key(value["shape"], parameters=parameters, complete_identity=True)
                for value in values
            ),
        )
    except (KeyError, TypeError) as error:
        raise GraphError("malformed compiler codec specialization terms") from error


def _unify_codec_terms(template, concrete, bindings):
    if isinstance(template, tuple):
        if template and template[0] == "parameter":
            index = template[1]
            if index in bindings:
                return bindings[index] == concrete
            bindings[index] = concrete
            return True
        return (
            isinstance(concrete, tuple)
            and len(template) == len(concrete)
            and all(_unify_codec_terms(a, b, bindings) for a, b in zip(template, concrete))
        )
    return type(template) is type(concrete) and template == concrete


def codec_specialization_owner(call, templates, serialize_trait, definitions):
    """Discharge a concrete instance of a previously verified generic codec call.

    The driver separates calls using the enclosing serializer parameter from
    independent publications. Only those `codec_calls` are templates here. A
    single consistent indexed substitution must match the caller's self type,
    every payload and every serializer at the exact observed call site. Several
    derive calls may share a span; exactly one full structural match is required.
    Arguments/results may contain unresolved associated-type projections and are
    not substituted: the compiler's explicit trait payload/serializer obligations
    above are the admission surface.
    """
    site = _codec_site(call, serialize_trait)
    concrete = _codec_terms(call, parameters=False)
    matches = []
    for template in templates:
        if _codec_site(template, serialize_trait) != site:
            continue
        bindings = {}
        if (
            _unify_codec_terms(_codec_terms(template, parameters=True), concrete, bindings)
            and bindings
        ):
            matches.append(template)
    if len(matches) != 1:
        raise GraphError("missing or ambiguous parameterized compiler codec call template")
    receiver = call["caller"]["implementation"]["self_type"]
    require_wire_payload(receiver["shape"], definitions)
    for payload in call["payloads"]:
        require_wire_payload(payload["shape"], definitions)
    return "wire-codec/" + identity(receiver["nominal"])


def discharge_invocations(raw, definitions):
    from capacity_census_wire_schema_publication import (
        check_report_publisher_owner,
        resolve_schema_owners,
    )

    if raw.get("errors") or any(
        not raw.get(section)
        for section in ("bodies", "calls", "codec_calls", "dynamic_returns")
    ):
        raise GraphError("missing or unresolved compiled publication obligations")
    schema_owners = resolve_schema_owners(raw, definitions)
    require_check_report_protocol(definitions)
    report_call = require_check_report_call(raw["calls"], raw["crate"])
    ownership = []

    def record(section, row, owner):
        ownership.append(
            {
                "section": section,
                "record_sha256": hashlib.sha256(canonical(row).encode()).hexdigest(),
                "owner": owner,
            }
        )

    for row in raw["codec_calls"]:
        implementation = row["caller"].get("implementation")
        if (
            implementation is None
            or implementation.get("trait") != raw["serialize_trait"]
        ):
            raise GraphError(
                "generic codec call has no actual Serialize implementation owner"
            )
        owner = identity(implementation["self_type"]["nominal"])
        if owner not in definitions:
            raise GraphError(
                f"codec implementation is absent from the discovered graph: {owner}"
            )
        record("codec_calls", row, COMPILER_BINARY_OWNERS.get(owner, owner))
    for section in ("dynamic_returns", "schema_calls"):
        for row in raw[section]:
            key = row["caller"]["definition"]["def_path_hash"]
            if key not in schema_owners:
                raise GraphError("unowned dynamic JSON or generic schema publication")
            record(section, row, "schema-metadata/" + schema_owners[key])
    for row in raw["calls"]:
        callee = identity(row["callee"])
        caller = _caller_identity(row["caller"])
        report_publisher = check_report_publisher_owner(row["caller"], raw["crate"])
        if report_publisher is not None:
            if row is not report_call:
                raise GraphError("duplicate compiled check-report publication edge")
            key = row["caller"]["definition"]["def_path_hash"]
            if schema_owners.get(key) != report_publisher:
                raise GraphError("unowned check-report publisher identity")
            owner = check_report_call_owner(row, raw["crate"])
        elif row["local_callee"]:
            owner = _local_binary_owner(row)
        elif callee == "bincode::serialize":
            owner = binary_call_owner(
                caller, tuple(p["shape"] for p in row["payloads"])
            )
        elif (
            row["caller"].get("implementation") is not None
            and row["caller"]["implementation"].get("trait") == raw["serialize_trait"]
        ):
            owner = codec_specialization_owner(
                row, raw["codec_calls"], raw["serialize_trait"], definitions
            )
        else:
            key = row["caller"]["definition"]["def_path_hash"]
            if key not in schema_owners or not row["payloads"] or row["serializers"]:
                raise GraphError(
                    f"unowned concrete JSON publication: {caller} -> {callee}"
                )
            # Trait-selected calls have already identified the actual payload,
            # including aliases/macros/custom entrypoints. Schema scope does
            # not authorize an arbitrary number or a dynamic Value payload.
            for payload in row["payloads"]:
                require_wire_payload(payload["shape"], definitions)
            owner = "schema-metadata/" + schema_owners[key]
        record("calls", row, owner)
    return tuple(ownership)


@dataclass(frozen=True, init=False, slots=True)
class VerifiedInvocations:
    root: str
    source_sha256: str
    driver: str
    driver_build: str
    files: tuple
    report: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("publication requires actual compiled invocation verification")

    def validate(self):
        from capacity_census_wire_calls import _current_driver_receipt

        if source_identity(Path(self.root)) != self.source_sha256:
            raise GraphError("compiled publication source changed")
        if canonical(_current_driver_receipt(Path(self.driver))) != self.driver_build:
            raise GraphError("compiled publication driver build evidence changed")
        for filename, digest in self.files:
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                raise GraphError("compiled publication artifact changed")

    def execution_report(self):
        self.validate()
        return json.loads(self.report)


def verify_invocation_ownership(root, target, schema, caches):
    from capacity_census_wire_calls import build_driver, collect_library

    caches.validate()
    before = source_identity(root)
    documents = [
        json.loads(schema.canonical.document),
        json.loads(schema.document),
        *(json.loads(d) for d in schema.imported_documents),
    ]
    schema.validate(documents)
    # These include the exact binary obligations as well as every nominal JSON
    # root. Actual graph/role discovery and the local derive proof ran first.
    definitions = {d.identity: len(d.parameters) for d in schema.declarations}
    if not set(COMPILER_BINARY_OWNERS) <= definitions.keys():
        raise GraphError("missing actual binary publication owners")
    driver = build_driver(root, target / "wire-invocations")
    collected = collect_library(root, target, driver)
    ownership = discharge_invocations(collected["evidence"], definitions)
    files = [(str(driver), collected["binary_sha256"])]
    files.extend((row["artifact"], row["sha256"]) for row in collected["provenance"])
    aggregate_identity = hashlib.sha256(
        canonical(
            {
                "evidence_identity": collected["identity"],
                "driver_build": collected["driver_build"],
                "provenance": collected["provenance"],
                "ownership": ownership,
                "source_sha256": before,
            }
        ).encode()
    ).hexdigest()
    report = {
        "scope": "compiler-api default library Serialize/Serializer calls and dynamic JSON returns",
        "identity": aggregate_identity,
        "ownership": ownership,
        "compiled": collected,
        "native_binary_scope": "exact key/comparison uses and shared numeric-codec cache compatibility; native AST/IR fields acquire no wire authority",
    }
    if source_identity(root) != before:
        raise GraphError("publication source changed during compilation")
    witness = object.__new__(VerifiedInvocations)
    for name, value in {
        "root": str(root),
        "source_sha256": before,
        "driver": str(driver),
        "driver_build": canonical(collected["driver_build"]),
        "files": tuple(files),
        "report": canonical(report),
    }.items():
        object.__setattr__(witness, name, value)
    witness.validate()
    return witness
