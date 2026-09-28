"""Reusable, fail-closed typed graph machinery for the C6 wire migration.

This module is supporting infrastructure. The live census deliberately does not
call it until the atomic wire migration supplies all authority and codec tests.
The fixed point describes declaration leaves, not infinitely expanded value
paths: a recursive field contributes once for each reachable primitive width.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from capacity_census_typed import CensusError, NUMERIC_PRIMITIVES


class GraphError(CensusError):
    """The supplied artifacts do not establish a complete supported graph."""


# Exact standard-library identities, never suffix/name tests. Additional external
# types require their defining artifact or an explicitly implemented adapter.
_CONTAINERS = {
    "core::option::Option": 1,
    "core::result::Result": 2,
    "alloc::vec::Vec": 1,
    "alloc::boxed::Box": 1,
    "alloc::collections::btree::map::BTreeMap": 2,
    "std::collections::hash::map::HashMap": 2,
}
_ATOMIC_TYPES = {"alloc::string::String"}
_NONNUMERIC = {"bool", "char", "str"}
_TYPE_KINDS = {"struct", "enum", "type_alias"}
_NON_TYPE_EXPORTS = {
    "function",
    "constant",
    "static",
    "trait",
    "trait_alias",
    "macro",
    "proc_attribute",
    "proc_derive",
    "variant",
}
_SERDE_VALUES = {"tag", "content", "rename", "rename_all"}
_SERDE_FLAGS = {"deny_unknown_fields"}
_RENAME_RULES = {
    "lowercase",
    "UPPERCASE",
    "PascalCase",
    "camelCase",
    "snake_case",
    "SCREAMING_SNAKE_CASE",
    "kebab-case",
    "SCREAMING-KEBAB-CASE",
}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _attributes(item: dict) -> tuple[str, ...]:
    result = []
    for value in item.get("attrs", []):
        if isinstance(value, str):
            result.append(
                "#[automatically_derived]"
                if value == "automatically_derived"
                else value
            )
        elif (
            isinstance(value, dict)
            and set(value) == {"other"}
            and isinstance(value["other"], str)
        ):
            result.append(value["other"])
        elif isinstance(value, dict) and "other" not in value:
            # rustdoc's structured repr/doc/deprecated attributes do not encode
            # serde options. Serde helper attributes use the `other` variant.
            result.append(_json(value))
        else:
            raise GraphError("unsupported rustdoc attribute shape")
    return tuple(result)


def _serde(
    item: dict, *, flags: set[str] = _SERDE_FLAGS, values: set[str] = _SERDE_VALUES
) -> tuple[tuple[str, str | bool], ...]:
    options: dict[str, str | bool] = {}
    for attr in _attributes(item):
        if not re.match(r"#\[\s*serde\b", attr):
            continue
        match = re.fullmatch(r"#\[\s*serde\s*\((.*)\)\s*\]", attr, re.S)
        if not match:
            raise GraphError(f"unsupported serde attribute: {attr}")
        remaining = match[1].strip()
        while remaining:
            part = re.match(
                r'([a-z_]+)(?:\s*=\s*("(?:[^"\\]|\\.)*"))?\s*(,|$)', remaining, re.S
            )
            if not part:
                raise GraphError(f"unsupported serde option syntax: {attr}")
            key, raw, _ = part.groups()
            if key in options:
                raise GraphError(f"duplicate serde option {key}")
            if key in flags and raw is None:
                value: str | bool = True
            elif key in values and raw is not None:
                value = json.loads(raw)
                if key == "rename_all" and value not in _RENAME_RULES:
                    raise GraphError(f"unsupported serde rename rule {value}")
            else:
                raise GraphError(
                    f"unsupported serde option {key}; an explicit codec adapter is required"
                )
            options[key] = value
            remaining = remaining[part.end() :].strip()
    return tuple(sorted(options.items()))


@dataclass(frozen=True, order=True)
class Leaf:
    path: str
    primitive: str


@dataclass(frozen=True)
class Edge:
    path: str
    type: tuple
    serde: tuple


@dataclass(frozen=True)
class Definition:
    identity: str
    kind: str
    parameters: tuple[str, ...]
    serde: tuple
    layout: tuple
    edges: tuple[Edge, ...]
    codec: str
    parameter_kinds: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiscoveredGraph:
    """An immutable artifact-derived shape, including nonnumeric companions."""

    roots: tuple
    definitions: tuple[Definition, ...]
    numeric_leaves: tuple[Leaf, ...]
    identity: str
    iterations: int


class RustdocGraph:
    """Resolve a bundle of rustdoc-v60 defining artifacts.

    Supported generics are explicit type parameters and nondefault usize const
    parameters with decimal literals or declared-parameter substitutions.
    Computed consts, associated/opaque types and custom serializers require
    explicit support; none is treated as an empty/nonnumeric node. Arrays
    and borrowed containers are supported. The finite summary loses multiplicity
    and recursive path length, but preserves every declaration leaf and width.
    """

    def __init__(self, documents: Iterable[dict]):
        self.documents: dict[str, dict] = {}
        self.locations: dict[str, tuple[str, str]] = {}
        self.names: dict[tuple[str, str], str] = {}
        for document in documents:
            if document.get("format_version") != 60:
                raise GraphError("unsupported rustdoc format version; expected 60")
            root = document.get("index", {}).get(str(document.get("root")))
            if not root or "module" not in root.get("inner", {}):
                raise GraphError("missing rustdoc crate root")
            name = root.get("name")
            if not isinstance(name, str) or name in self.documents:
                raise GraphError("missing or duplicate defining crate artifact")
            self.documents[name] = document
            for item_id, path in document.get("paths", {}).items():
                if path.get("crate_id") != root.get("crate_id"):
                    continue
                identity = "::".join(path.get("path", []))
                if not identity or identity in self.locations:
                    raise GraphError(f"duplicate or empty defining identity {identity}")
                self.locations[identity] = (name, str(item_id))
                self.names[(name, str(item_id))] = identity
        if not self.documents:
            raise GraphError("no defining artifacts supplied")
        self.definitions: dict[str, Definition | None] = {}

    def _item(self, crate: str, item_id: str | int) -> dict:
        try:
            return self.documents[crate]["index"][str(item_id)]
        except KeyError as error:
            raise GraphError(
                f"missing definition {crate}:{item_id} in defining artifact"
            ) from error

    def _resolve(self, crate: str, item_id: str | int) -> tuple[str, str, str]:
        key = (crate, str(item_id))
        if key in self.names:
            identity = self.names[key]
            self._item(crate, item_id)
            return crate, str(item_id), identity
        record = self.documents[crate].get("paths", {}).get(str(item_id))
        if not record or not record.get("path"):
            raise GraphError(f"unresolved type identity {crate}:{item_id}")
        identity = "::".join(record["path"])
        if identity in _CONTAINERS or identity in _ATOMIC_TYPES:
            return "", "", identity
        location = self.locations.get(identity)
        if location is None:
            raise GraphError(f"missing defining artifact for imported type {identity}")
        self._item(*location)
        return *location, identity

    def public_exports(self, crate: str) -> dict[str, dict]:
        """Discover public type exports, including explicit and glob reexports.

        This is type-root discovery, not evidence that all exports serialize.
        The live integration must select its published serialization roots and
        require serializer/decoder evidence for every admitted carrier.
        """
        exports: dict[str, dict] = {}

        def visit(owner: str, item_id: str | int, prefix: str, ancestors: frozenset):
            key = (owner, str(item_id))
            if key in ancestors:
                raise GraphError("unsupported cyclic module export")
            item = self._item(owner, item_id)
            module = item.get("inner", {}).get("module")
            if module is None or module.get("is_stripped"):
                raise GraphError("missing or stripped public module")
            for child_id in module["items"]:
                child = self._item(owner, child_id)
                if child.get("visibility") != "public":
                    continue
                inner = child["inner"]
                if "use" in inner:
                    use = inner["use"]
                    if use.get("id") is None:
                        raise GraphError("unresolved public reexport")
                    record = self.documents[owner].get("paths", {}).get(str(use["id"]))
                    if record and record.get("kind") in _NON_TYPE_EXPORTS:
                        continue
                    target_crate, target_id, _ = self._resolve(owner, use["id"])
                    target = self._item(target_crate, target_id)
                    export_name = use["name"]
                    if use.get("is_glob"):
                        visit(target_crate, target_id, prefix, ancestors | {key})
                        continue
                    child = target
                    child_id = use["id"]
                    inner = child["inner"]
                else:
                    export_name = child.get("name")
                    target_crate, target_id = owner, child_id
                if "module" in inner:
                    visit(
                        target_crate,
                        target_id,
                        f"{prefix}::{export_name}",
                        ancestors | {key},
                    )
                elif _TYPE_KINDS & inner.keys():
                    name = f"{prefix}::{export_name}"
                    # Keep the exporting artifact's resolvable ID. A cross-crate
                    # glob export is represented by its defining identity below.
                    value = {"graph_export": (target_crate, str(target_id))}
                    if name in exports and exports[name] != value:
                        raise GraphError(f"ambiguous public type export {name}")
                    exports[name] = value

        if crate not in self.documents:
            raise GraphError(f"missing crate artifact {crate}")
        visit(crate, self.documents[crate]["root"], crate, frozenset())
        return exports

    def serialization_candidates(self, crate: str) -> dict[str, dict]:
        """Public serde types and aliases, across every exported module.

        Publication is a separate ownership obligation. This inventory neither
        selects JSON roots nor exempts cache types. New candidates require an
        explicit owner; unsupported custom codecs remain candidates and fail
        graph discovery until an execution-backed adapter exists.
        """
        candidates = {}
        for name, ty in self.public_exports(crate).items():
            owner, item_id = ty["graph_export"]
            item = self._item(owner, item_id)
            kind = next(iter(_TYPE_KINDS & item["inner"].keys()))
            if kind == "type_alias":
                # An alias can expose a primitive/container with serde through
                # its underlying type; absence of nominal impls is not a skip.
                candidates[name] = ty
                continue
            if self._has_serde(owner, item["inner"][kind]):
                candidates[name] = ty
        return candidates

    def _has_serde(self, crate: str, body: dict) -> bool:
        identities = {
            f"{module}::{name}"
            for module, name in (
                ("serde_core::ser", "Serialize"),
                ("serde_core::de", "Deserialize"),
                ("serde::ser", "Serialize"),
                ("serde::de", "Deserialize"),
                ("serde", "Serialize"),
                ("serde", "Deserialize"),
            )
        }
        found = False
        for impl_id in body.get("impls", []):
            impl = self._item(crate, impl_id).get("inner", {}).get("impl", {})
            trait = impl.get("trait")
            if not trait:
                continue
            record = self.documents[crate].get("paths", {}).get(str(trait.get("id")))
            if record is None:
                raise GraphError("unresolved serde candidate trait identity")
            found |= "::".join(record.get("path", [])) in identities
        return found

    def serialization_definitions(self, crate: str) -> dict[str, dict]:
        """Nominal serde definitions, including private module declarations.

        Unlike public candidates, aliases do not define a local serde codec.
        Function-local types absent from rustdoc still require source-entry
        discovery; this inventory does not assert their absence.
        """
        if crate not in self.documents:
            raise GraphError(f"missing crate artifact {crate}")
        result = {}
        for (owner, item_id), identity in sorted(self.names.items()):
            if owner != crate:
                continue
            item = self._item(owner, item_id)
            kinds = {"struct", "enum"} & item["inner"].keys()
            if kinds and self._has_serde(owner, item["inner"][next(iter(kinds))]):
                result[identity] = {"graph_export": (owner, item_id)}
        return result

    def discover_exports(self, crate: str) -> DiscoveredGraph:
        exports = self.public_exports(crate)
        if not exports:
            raise GraphError("no public type roots discovered")
        return self._discover(
            [(crate, name, ty) for name, ty in sorted(exports.items())]
        )

    def discover(self, crate: str, root_type: dict) -> DiscoveredGraph:
        return self._discover([(crate, "$root", root_type)])

    @staticmethod
    def _const(expression: str, parameters: dict[str, str]) -> tuple:
        if re.fullmatch(r"[0-9]+", expression):
            return ("const", str(int(expression)))
        if parameters.get(expression) == "usize":
            return ("const_generic", expression)
        raise GraphError(f"unresolved or unsupported const expression {expression}")

    def _type(self, crate: str, ty: Any, parameters: dict[str, str]) -> tuple:
        if not isinstance(ty, dict) or len(ty) != 1:
            raise GraphError("unsupported rustdoc type shape")
        kind, body = next(iter(ty.items()))
        if kind == "graph_export":
            return self._type(
                body[0], {"resolved_path": {"id": body[1], "args": None}}, parameters
            )
        if kind == "primitive":
            if body not in NUMERIC_PRIMITIVES | _NONNUMERIC:
                raise GraphError(f"unknown primitive {body}")
            return ("primitive", body)
        if kind == "generic":
            if parameters.get(body) != "type":
                raise GraphError(f"unresolved generic {body}")
            return ("generic", body)
        if kind == "resolved_path":
            args_shape = body.get("args")
            if args_shape is None:
                raw_args = []
            elif set(args_shape) == {"angle_bracketed"} and not args_shape[
                "angle_bracketed"
            ].get("constraints"):
                raw_args = args_shape["angle_bracketed"]["args"]
            else:
                raise GraphError(
                    "unsupported associated or parenthesized generic arguments"
                )
            args = []
            for arg in raw_args:
                if set(arg) == {"type"}:
                    args.append(self._type(crate, arg["type"], parameters))
                elif set(arg) == {"const"}:
                    args.append(self._const(arg["const"]["expr"], parameters))
                elif set(arg) != {"lifetime"}:
                    raise GraphError("unsupported generic argument")
            defining_crate, item_id, identity = self._resolve(crate, body["id"])
            if identity in _ATOMIC_TYPES:
                if args:
                    raise GraphError(f"unexpected generic arguments for {identity}")
                return ("atomic", identity)
            if identity in _CONTAINERS:
                if len(args) != _CONTAINERS[identity] or any(
                    arg[0] in {"const", "const_generic"} for arg in args
                ):
                    raise GraphError(
                        f"wrong generic arity or argument kind for {identity}"
                    )
                return ("container", identity, tuple(args))
            self._definition(defining_crate, item_id, identity)
            # During recursion the declaration itself may still be assembling.
            declaration = self._item(defining_crate, item_id)
            inner = next(v for k, v in declaration["inner"].items() if k in _TYPE_KINDS)
            expected = self._parameters(inner)
            if len(args) != len(expected):
                raise GraphError(
                    f"unresolved or excess generic arguments for {identity}"
                )
            for parameter_kind, argument in zip(
                self._parameter_kinds(inner), args, strict=True
            ):
                is_const = argument[0] in {"const", "const_generic"}
                if is_const != (parameter_kind == "usize"):
                    raise GraphError(f"wrong generic argument kind for {identity}")
            return ("reference", identity, tuple(args))
        if kind == "tuple":
            return ("tuple", tuple(self._type(crate, t, parameters) for t in body))
        if kind == "slice":
            return ("slice", self._type(crate, body, parameters))
        if kind == "array":
            length = self._const(str(body["len"]), parameters)
            return ("array", length, self._type(crate, body["type"], parameters))
        if kind == "borrowed_ref":
            return (
                "borrow",
                body.get("is_mutable", False),
                body.get("lifetime"),
                self._type(crate, body["type"], parameters),
            )
        raise GraphError(f"unsupported rustdoc type {kind}")

    @staticmethod
    def _parameter_kinds(inner: dict) -> tuple[str, ...]:
        kinds = []
        for parameter in inner.get("generics", {}).get("params", []):
            kind = parameter["kind"]
            if "lifetime" in kind:
                continue
            if set(kind) == {"type"} and kind["type"].get("default") is None:
                kinds.append("type")
            elif (
                set(kind) == {"const"}
                and kind["const"].get("default") is None
                and kind["const"].get("type") == {"primitive": "usize"}
            ):
                kinds.append("usize")
            else:
                raise GraphError("unsupported generic parameter or default")
        return tuple(kinds)

    @classmethod
    def _parameters(cls, inner: dict) -> tuple[str, ...]:
        cls._parameter_kinds(inner)
        names = tuple(
            parameter["name"]
            for parameter in inner.get("generics", {}).get("params", [])
            if "lifetime" not in parameter["kind"]
        )
        if len(set(names)) != len(names):
            raise GraphError("duplicate generic parameter")
        return names

    def _codec(self, crate: str, inner: dict) -> str:
        found = set()
        for impl_id in inner.get("impls", []):
            item = self._item(crate, impl_id)
            trait = item.get("inner", {}).get("impl", {}).get("trait")
            if not trait:
                continue
            record = self.documents[crate].get("paths", {}).get(str(trait.get("id")))
            actual = "::".join(record.get("path", [])) if record else ""
            paths = {
                f"{owner}::{module}::{name}": name
                for owner in ("serde", "serde_core")
                for module, name in (("ser", "Serialize"), ("de", "Deserialize"))
            }
            paths.update(
                {"serde::Serialize": "Serialize", "serde::Deserialize": "Deserialize"}
            )
            path = paths.get(actual)
            if path is None:
                continue
            if "#[automatically_derived]" not in _attributes(item):
                raise GraphError(
                    "custom serializer/decoder requires an execution-backed shape adapter"
                )
            found.add(path)
        return {
            frozenset(): "unproven",
            frozenset({"Serialize"}): "serde-derived-serialize",
            frozenset({"Deserialize"}): "serde-derived-deserialize",
            frozenset({"Serialize", "Deserialize"}): "serde-derived",
        }[frozenset(found)]

    def _serde(self, item: dict):
        return _serde(item)

    def _validate_field_serde(self, options: tuple, ty: tuple):
        """Additional adapters may validate supported field omission rules."""

    def _definition(self, crate: str, item_id: str, identity: str):
        if identity in self.definitions:
            return
        item = self._item(crate, item_id)
        kinds = _TYPE_KINDS & item.get("inner", {}).keys()
        if len(kinds) != 1:
            raise GraphError(f"unsupported nominal definition {identity}")
        kind = next(iter(kinds))
        inner = item["inner"][kind]
        parameters = self._parameters(inner)
        parameter_kinds = self._parameter_kinds(inner)
        context = dict(zip(parameters, parameter_kinds, strict=True))
        serde = self._serde(item)
        codec = self._codec(crate, inner)
        self.definitions[identity] = None
        edges = []
        layout = []

        def fields(shape, prefix):
            if shape == "unit" or shape == "plain":
                return ("unit", ())
            if not isinstance(shape, dict) or len(shape) != 1:
                raise GraphError(f"unsupported aggregate shape in {identity}")
            shape_kind, value = next(iter(shape.items()))
            if shape_kind in {"plain", "struct"}:
                if value.get("has_stripped_fields"):
                    raise GraphError(f"stripped fields in {identity}")
                ids = value["fields"]
            elif shape_kind == "tuple":
                ids = value
            else:
                raise GraphError(f"unsupported aggregate shape in {identity}")
            members = []
            for position, field_id in enumerate(ids):
                if field_id is None:
                    raise GraphError(f"stripped tuple field in {identity}")
                field = self._item(crate, field_id)
                name = field.get("name") or f"${position}"
                if shape_kind == "tuple":
                    name = f"${position}"
                field_serde = self._serde(field)
                if "struct_field" not in field.get("inner", {}):
                    raise GraphError(f"invalid field in {identity}")
                ty = self._type(crate, field["inner"]["struct_field"], context)
                self._validate_field_serde(field_serde, ty)
                edge = Edge(f"{prefix}.{name}", ty, field_serde)
                if edge.path in {e.path for e in edges}:
                    raise GraphError(f"duplicate field {edge.path}")
                edges.append(edge)
                members.append((name, field_serde))
            return (shape_kind, tuple(members))

        if kind == "type_alias":
            edges.append(
                Edge(
                    identity + ".$alias",
                    self._type(crate, inner["type"], context),
                    (),
                )
            )
        elif kind == "struct":
            layout.append(("", (), fields(inner["kind"], identity)))
        else:
            if inner.get("has_stripped_variants"):
                raise GraphError(f"stripped variants in {identity}")
            seen = set()
            for variant_id in inner["variants"]:
                variant = self._item(crate, variant_id)
                name = variant.get("name")
                if not name or name in seen:
                    raise GraphError(f"missing or duplicate variant in {identity}")
                seen.add(name)
                layout.append(
                    (
                        name,
                        self._serde(variant),
                        fields(
                            variant["inner"]["variant"]["kind"], f"{identity}::{name}"
                        ),
                    )
                )
        self.definitions[identity] = Definition(
            identity,
            kind,
            parameters,
            serde,
            tuple(layout),
            tuple(edges),
            codec,
            parameter_kinds,
        )

    def _evaluate(
        self, expr: tuple, origin: str, summaries: dict[str, set[Leaf]]
    ) -> set[Leaf]:
        kind = expr[0]
        if kind == "primitive":
            return {Leaf(origin, expr[1])} if expr[1] in NUMERIC_PRIMITIVES else set()
        if kind == "generic":
            return {Leaf(origin, "$" + expr[1])}
        if kind in {"atomic", "const", "const_generic"}:
            return set()
        if kind == "reference":
            declaration = self.definitions[expr[1]]
            assert declaration is not None
            arguments = dict(zip(declaration.parameters, expr[2], strict=True))
            result = set()
            for leaf in summaries[expr[1]]:
                if leaf.primitive.startswith("$"):
                    result.update(
                        self._evaluate(
                            arguments[leaf.primitive[1:]], leaf.path, summaries
                        )
                    )
                else:
                    result.add(leaf)
            return result
        children = (
            expr[2]
            if kind == "container"
            else expr[1]
            if kind == "tuple"
            else (expr[-1],)
        )
        return set().union(
            *(self._evaluate(child, origin, summaries) for child in children)
        )

    def _alias_cycles(self):
        def references(expr):
            if expr[0] == "reference":
                yield expr[1]
                children = expr[2]
            elif expr[0] == "container":
                children = expr[2]
            elif expr[0] == "tuple":
                children = expr[1]
            elif expr[0] in {"slice", "array", "borrow"}:
                children = (expr[-1],)
            else:
                children = ()
            for child in children:
                yield from references(child)

        complete = set()

        def visit(identity, active):
            if identity in active:
                raise GraphError(f"unresolvable alias cycle at {identity}")
            if identity in complete:
                return
            declaration = self.definitions[identity]
            assert declaration is not None
            if declaration.kind != "type_alias":
                return
            for edge in declaration.edges:
                for child in references(edge.type):
                    visit(child, active | {identity})
            complete.add(identity)

        for identity in self.definitions:
            visit(identity, set())

    def _discover(self, roots) -> DiscoveredGraph:
        self.definitions = {}
        parsed = tuple((name, self._type(crate, ty, {})) for crate, name, ty in roots)
        self._alias_cycles()
        summaries: dict[str, set[Leaf]] = {name: set() for name in self.definitions}
        iterations = 0
        while True:
            iterations += 1
            changed = False
            for name, definition in self.definitions.items():
                assert definition is not None
                discovered = set().union(
                    *(
                        self._evaluate(edge.type, edge.path, summaries)
                        for edge in definition.edges
                    )
                )
                if not discovered <= summaries[name]:
                    summaries[name].update(discovered)
                    changed = True
            if not changed:
                break
        leaves = set().union(
            *(self._evaluate(ty, name, summaries) for name, ty in parsed)
        )
        if any(leaf.primitive.startswith("$") for leaf in leaves):
            raise GraphError("unresolved generic survived root substitution")
        definitions = tuple(self.definitions[key] for key in sorted(self.definitions))
        shape = (
            parsed,
            [
                (
                    d.identity,
                    d.kind,
                    d.parameters,
                    d.parameter_kinds,
                    d.serde,
                    d.layout,
                    [(e.path, e.type, e.serde) for e in d.edges],
                    d.codec,
                )
                for d in definitions
            ],
        )
        return DiscoveredGraph(
            parsed,
            definitions,
            tuple(sorted(leaves)),
            hashlib.sha256(_json(shape).encode()).hexdigest(),
            iterations,
        )


class Role(Enum):
    SOURCE_OFFSET = "source-offset"
    SOURCE_EXTENT = "source-extent"
    INPUT_REFERENCE = "input-reference"


@dataclass(frozen=True)
class FieldRole:
    path: str
    role: Role


@dataclass(frozen=True)
class CarrierContract:
    """Reviewed expected identity plus closed roles; never a discovery source.

    Supplying a current digest only passes the exact-shape check. The verifier
    independently requires the known carrier identity, structure and widths.
    This slice supports source-coordinate and input-slot roles only. Numeric
    payload codecs and runtime admission evidence remain integration work.
    """

    carrier: str
    graph_identity: str
    fields: tuple[FieldRole, ...]


@dataclass(frozen=True, init=False, slots=True)
class VerifiedTransport:
    """Only verify_transport constructs these; descriptors are not witnesses."""

    _leaf: Leaf
    _role: Role
    _identity: str

    def __new__(cls):
        raise TypeError("transport witnesses are constructed only by the verifier")

    @property
    def leaf(self):
        return self._leaf

    @property
    def role(self):
        return self._role

    @property
    def graph_identity(self):
        return self._identity


_SPAN = "chelis_compiler_api::schema::DiagnosticSpan"
_DIM = "chelis_compiler_api::schema::WireRtDim"
_AXIS = "chelis_compiler_api::schema::WireRtAxis"


def _validate_role(graph: DiscoveredGraph, contract: CarrierContract, field: FieldRole):
    definitions = {d.identity: d for d in graph.definitions}
    declaration = definitions.get(contract.carrier)
    if declaration is None or declaration.codec != "serde-derived":
        raise GraphError("carrier lacks derived serializer and decoder evidence")
    if declaration.kind != "enum" or declaration.parameters:
        raise GraphError("carrier role requires its exact closed enum structure")
    if contract.carrier == _SPAN:
        expected = {
            _SPAN + "::Point.offset": Role.SOURCE_OFFSET,
            _SPAN + "::Range.offset": Role.SOURCE_OFFSET,
            _SPAN + "::Range.len": Role.SOURCE_EXTENT,
        }
        if field.role != expected.get(field.path):
            raise GraphError("source role does not govern this exact field")
        if dict(declaration.serde) != {"rename_all": "snake_case", "tag": "span"}:
            raise GraphError(
                "source carrier lacks its exact Point/Range serde discriminator"
            )
        if {edge.path for edge in declaration.edges} != set(expected):
            raise GraphError("source carrier has changed fields")
        if any(
            edge.type != ("primitive", "u64") or edge.serde
            for edge in declaration.edges
        ):
            raise GraphError("source coordinates require exact nonnegative u64 fields")
        expected_layout = (
            ("Point", (), ("struct", (("offset", ()),))),
            ("Range", (), ("struct", (("offset", ()), ("len", ())))),
        )
        if tuple(sorted(declaration.layout)) != expected_layout:
            raise GraphError("source carrier has changed variant structure")
    elif contract.carrier == _DIM:
        if (
            field.path != _DIM + "::InputAxis.tensor"
            or field.role != Role.INPUT_REFERENCE
        ):
            raise GraphError("input-reference role does not govern this exact field")
        if dict(declaration.serde) != {"rename_all": "snake_case", "tag": "bound"}:
            raise GraphError("input reference lacks its owning variant discriminator")
        fields = {
            e.path: e
            for e in declaration.edges
            if e.path.startswith(_DIM + "::InputAxis.")
        }
        if set(fields) != {_DIM + "::InputAxis.tensor", _DIM + "::InputAxis.axis"}:
            raise GraphError("input reference has changed sibling structure")
        slot, axis = (
            fields[_DIM + "::InputAxis.tensor"],
            fields[_DIM + "::InputAxis.axis"],
        )
        variant = next((v for v in declaration.layout if v[0] == "InputAxis"), None)
        if variant != ("InputAxis", (), ("struct", (("tensor", ()), ("axis", ())))):
            raise GraphError("input reference has changed variant structure")
        if (
            slot.type != ("primitive", "u64")
            or slot.serde
            or axis.serde
            or axis.type != ("reference", _AXIS, ())
        ):
            raise GraphError(
                "input reference requires u64 slot and separate exact axis carrier"
            )
        axis_definition = definitions.get(_AXIS)
        if (
            axis_definition is None
            or axis_definition.codec != "serde-derived"
            or axis_definition.kind != "enum"
        ):
            raise GraphError("missing exact axis carrier")
        if (
            dict(axis_definition.serde) != {"tag": "axis", "rename_all": "snake_case"}
            or len(axis_definition.edges) != 1
        ):
            raise GraphError("axis carrier has changed discriminator or fields")
        if axis_definition.layout != (("Lit", (), ("struct", (("value", ()),))),):
            raise GraphError("axis carrier has changed variant structure")
        edge = axis_definition.edges[0]
        if (
            edge.path != _AXIS + "::Lit.value"
            or edge.type != ("primitive", "i32")
            or edge.serde
        ):
            raise GraphError(
                "reference axis requires its separate int32 numeric-operation role"
            )
    else:
        raise GraphError("no closed semantic role governs this carrier identity")


def verify_transport(
    graph: DiscoveredGraph, leaf: Leaf, contracts: Iterable[CarrierContract]
) -> VerifiedTransport:
    if leaf not in graph.numeric_leaves:
        raise GraphError("leaf is absent from the current discovered artifact")
    matches = []
    for contract in contracts:
        if contract.graph_identity != graph.identity:
            continue
        for field in contract.fields:
            if field.path == leaf.path:
                _validate_role(graph, contract, field)
                matches.append(field)
    if len(matches) != 1:
        raise GraphError(
            f"{'zero' if not matches else 'multiple'} matching transport contracts for {leaf.path}"
        )
    witness = object.__new__(VerifiedTransport)
    object.__setattr__(witness, "_leaf", leaf)
    object.__setattr__(witness, "_role", matches[0].role)
    object.__setattr__(witness, "_identity", graph.identity)
    return witness
