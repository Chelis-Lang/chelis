"""Registered Python value exposure, separate from serialized Rust ownership.

An opaque pyclass is a Python handle. Its registered getters and methods are
separate roots; walking its private allocation internals would describe a
different surface. Dynamic Python objects require a payload adapter and fail
closed here until the owning tensor/CompilerJson migration supplies one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_typed import FLOAT_PRIMITIVES, INTEGER_PRIMITIVES, canonical_type

_PYCLASS = "pyo3::pyclass::PyClass"
_SOURCE_JSON = "chelis_python::source_json::SourceJson"
_SOURCE_RESULTS = {
    "chelis_compiler_api::schema::DecompileResult",
    "chelis_compiler_api::schema::ValidateResult",
}
# spec/11 §1 defines these observations as names/path/closed vocabulary.
# All other string outputs require a typed payload adapter, including the four
# legacy compiler JSON callables. String spelling alone is never admission.
_TEXT_OUTPUT_DOMAINS = {
    "chelis_python::CompiledModel::input_names": "input-names",
    "chelis_python::CompiledModel::output_names": "output-names",
    "chelis_python::CompiledModel::path": "artifact-path",
    "chelis_python::CompiledModel::target": "compile-target-vocabulary",
    "chelis_python::NativeTensor::dtype": "dtype-vocabulary",
}


_DYNAMIC = {
    "pyo3::types::any::PyAny",
    "pyo3::types::dict::PyDict",
    "pyo3::types::tuple::PyTuple",
    "pyo3::instance::PyObject",
    "pyo3::instance::Py",
    "pyo3::instance::Bound",
}


def payload_obligations(graph):
    """Summarize text and codec obligations separately for each adapter subtree.

    A nominal type can occur both inside SourceJson and beside it. Keep those
    contexts distinct, including when a generic argument is substituted. The
    finite summaries terminate for recursive nominal types without expanding
    their potentially unbounded instantiations.
    """
    definitions = {definition.identity: definition for definition in graph.definitions}
    summaries = {(name, source_json): set() for name in definitions for source_json in (False, True)}

    def evaluate(expr, source_json):
        kind = expr[0]
        if kind in {"primitive", "atomic"}:
            text = expr in (("atomic", "alloc::string::String"), ("primitive", "str"))
            return {("text",)} if text and not source_json else set()
        if kind == "generic":
            return {("parameter", expr[1], source_json)}
        if kind == "reference":
            declaration = definitions[expr[1]]
            arguments = dict(zip(declaration.parameters, expr[2], strict=True))
            result = set()
            if source_json and declaration.kind != "type_alias" and declaration.codec != "serde-derived":
                result.add(("codec",))
            for obligation in summaries[(expr[1], source_json)]:
                if obligation[0] == "parameter":
                    result.update(evaluate(arguments[obligation[1]], obligation[2]))
                else:
                    result.add(obligation)
            return result
        if kind == "container":
            children = expr[2]
            source_json = source_json or expr[1] == _SOURCE_JSON
        elif kind == "tuple":
            children = expr[1]
        else:
            children = (expr[-1],)
        return set().union(*(evaluate(child, source_json) for child in children))

    while True:
        changed = False
        for (name, source_json), known in summaries.items():
            obligations = set().union(*(evaluate(edge.type, source_json) for edge in definitions[name].edges))
            if not obligations <= known:
                known.update(obligations)
                changed = True
        if not changed:
            break
    return set().union(*(evaluate(expr, False) for _, expr in graph.roots))


class BindingGraph(RustdocGraph):
    def __init__(self, documents, classes):
        super().__init__(documents)
        self.classes = classes
        if len(set(classes.values())) != len(classes):
            raise GraphError("duplicate registered Rust class identity")
        for identity in classes.values():
            location = self.locations.get(identity)
            if location is None:
                raise GraphError(f"missing registered PyClass {identity}")
            item = self._item(*location)
            inner = item.get("inner", {}).get("struct")
            if inner is None:
                raise GraphError(f"registered PyClass is not a struct: {identity}")
            traits = []
            for impl_id in inner.get("impls", []):
                impl = self._item(location[0], impl_id)["inner"].get("impl", {})
                trait = impl.get("trait")
                if trait:
                    record = self.documents[location[0]].get("paths", {}).get(str(trait["id"]), {})
                    traits.append("::".join(record.get("path", [])))
            if _PYCLASS not in traits:
                raise GraphError(f"registered identity lacks exact PyClass implementation: {identity}")

    def _type(self, crate, ty, parameters):
        if isinstance(ty, dict) and set(ty) == {"resolved_path"}:
            path = ty["resolved_path"]
            record = self.documents[crate].get("paths", {}).get(str(path["id"]))
            identity = "::".join(record.get("path", [])) if record else ""
            if identity in _DYNAMIC:
                raise GraphError(f"dynamic Python payload requires a producer/consumer contract: {identity}")
            if identity in set(self.classes.values()) | {"pyo3::marker::Python", "pyo3::err::PyResult", _SOURCE_JSON}:
                shape = path.get("args")
                if shape is None:
                    raw = []
                elif set(shape) == {"angle_bracketed"} and not shape["angle_bracketed"].get("constraints"):
                    raw = shape["angle_bracketed"]["args"]
                else:
                    raise GraphError("unsupported binding adapter arguments")
                if any(set(arg) not in ({"type"}, {"lifetime"}) for arg in raw):
                    raise GraphError("unsupported binding adapter argument")
                args = [arg["type"] for arg in raw if "type" in arg]
                expected = 1 if identity in {"pyo3::err::PyResult", _SOURCE_JSON} else 0
                if len(args) != expected:
                    raise GraphError(f"wrong binding adapter arity for {identity}")
                if not expected:
                    return ("atomic", identity)
                if identity == _SOURCE_JSON:
                    location = self.locations.get(identity)
                    if not location:
                        raise GraphError("missing source JSON wrapper definition")
                    declaration = self._item(*location)["inner"].get("struct", {})
                    fields = declaration.get("kind", {}).get("plain", {})
                    ids = fields.get("fields", [])
                    if fields.get("has_stripped_fields") or len(ids) != 1:
                        raise GraphError("source JSON wrapper has changed fields")
                    field = self._item(location[0], ids[0])
                    if field.get("name") != "value" or field.get("inner") != {"struct_field": {"generic": "T"}}:
                        raise GraphError("source JSON wrapper must retain its typed result")
                    root = args[0].get("resolved_path", {})
                    target = self.documents[crate].get("paths", {}).get(str(root.get("id")), {})
                    if "::".join(target.get("path", [])) not in _SOURCE_RESULTS:
                        raise GraphError("source JSON requires an exact sealed source result")
                return ("container", identity, (self._type(crate, args[0], parameters),))
        return super()._type(crate, ty, parameters)

    def exposure(self, ty, owner=None):
        if owner:
            crate, item_id = self.locations[owner]

            def replace_self(value):
                if value == {"generic": "Self"}:
                    return {"resolved_path": {"id": item_id, "args": None}}
                if isinstance(value, dict):
                    return {key: replace_self(member) for key, member in value.items()}
                if isinstance(value, list):
                    return [replace_self(member) for member in value]
                return value

            ty = replace_self(ty)
        return self.discover("chelis_python", ty if ty is not None else {"tuple": []})


def registration_roots(graph, functions, methods, classes, provenance):
    """Join syn-derived registrar roots to live names and exact rustdoc spans.

    Metadata is produced from the compile-time source used by the live census.
    It is never inferred from a public name or an inherent helper's spelling.
    Unsupported registration forms are rejected by the source producer.
    """
    if not isinstance(provenance, dict) or set(provenance) != {
        "source_path", "source_sha256", "registrations"
    }:
        raise GraphError("missing or malformed registration provenance")
    source_path = provenance["source_path"]
    try:
        source = Path(source_path).read_bytes()
    except (OSError, TypeError) as error:
        raise GraphError(f"unreadable registration provenance source: {error}") from error
    if hashlib.sha256(source).hexdigest() != provenance["source_sha256"]:
        raise GraphError("stale registration provenance source bytes")
    if len(set(functions)) != len(functions) or len(set(methods)) != len(methods):
        raise GraphError("duplicate registered callable")
    expected = set(functions) | set(methods)
    seen = {}
    roots = []
    inverse = {identity: name for name, identity in classes.items()}
    for row in provenance["registrations"]:
        if set(row) != {"owner", "python_name", "rust_name", "kind", "line", "column"}:
            raise GraphError("malformed registration provenance row")
        if row["kind"] not in {"function", "method", "getter", "setter", "constructor", "classmethod", "staticmethod"}:
            raise GraphError("unsupported registration provenance kind")
        if row["owner"] is None:
            if row["kind"] != "function":
                raise GraphError("function registration provenance has a method kind")
            public = row["python_name"]
            owner = None
            identity = "chelis_python::" + row["rust_name"]
            location = graph.locations.get(identity)
            found = [] if location is None else [graph._item(*location)]
        else:
            if row["kind"] == "function":
                raise GraphError("method registration provenance has a function kind")
            owner = "chelis_python::" + row["owner"]
            if owner not in inverse:
                raise GraphError("registration provenance owner is not an exact registered class")
            public = inverse[owner] + "::" + row["python_name"]
            identity = owner + "::" + row["rust_name"]
            crate, item_id = graph.locations[owner]
            item = graph._item(crate, item_id)
            found = []
            for impl_id in item["inner"]["struct"].get("impls", []):
                impl = graph._item(crate, impl_id)["inner"].get("impl", {})
                if impl.get("trait") is not None:
                    continue
                found += [graph._item(crate, method_id) for method_id in impl.get("items", [])
                          if graph._item(crate, method_id).get("name") == row["rust_name"]
                          and "function" in graph._item(crate, method_id).get("inner", {})]
        if public not in expected:
            raise GraphError(f"registration provenance has no live exposure: {public}")
        kinds = seen.setdefault(public, set())
        if row["kind"] in kinds or (kinds and kinds | {row["kind"]} != {"getter", "setter"}):
            raise GraphError(f"ambiguous registration provenance for {public}")
        kinds.add(row["kind"])
        if len(found) != 1:
            raise GraphError(f"missing or duplicate rustdoc registration provenance: {identity}")
        item = found[0]
        span = item.get("span") or {}
        position = [row["line"], row["column"]]
        if (span.get("filename") != source_path
                or not all(type(coordinate) is int and coordinate > 0 for coordinate in position)
                or not span.get("begin", [0, 0]) <= position < span.get("end", [0, 0])):
            raise GraphError(f"rustdoc registration provenance span mismatch: {identity}")
        kind = "binding-pyfunction" if owner is None else "binding-pymethod"
        proof = ("registration", provenance["source_sha256"], identity, row["kind"], public,
                 row["line"], row["column"])
        roots.append((kind, "chelis_python::" + public, item, owner, proof))
    if set(seen) != expected:
        raise GraphError(f"missing registration provenance: {sorted(expected - set(seen))}")
    return roots


def discover_bindings(documents, functions, methods, classes, *, provenance=None, compiler_json=None):
    """Analyze exactly registered implementations; retain unresolved legacy types.

    The caller must reject a problem on every final/new row. Only the Rust
    tripwire owns the exact unchanged frozen remainder. This function neither
    knows that cohort nor grants nonnumeric or transport authority.
    """
    graph = BindingGraph(documents, classes)
    if compiler_json is not None:
        from capacity_census_compiler_json import OUTPUTS, VerifiedCompilerJsonBindings, require_parameter_slots
        if type(compiler_json) is not VerifiedCompilerJsonBindings:
            raise GraphError("compiler JSON requires the actual execution witness")
        compiler_json.validate()
    roots = registration_roots(graph, functions, methods, classes, provenance)
    results = []
    for kind, name, item, owner, proof in roots:
        signature = item.get("inner", {}).get("function", {}).get("sig")
        if signature is None:
            raise GraphError(f"missing typed signature for {name}")
        inputs = signature.get("inputs", [])
        output = signature.get("output")
        rendered = ", ".join(f"{label}: {canonical_type(ty)}" for label, ty in inputs)
        result = {"kind": kind, "id": f"{name}({rendered}) -> {canonical_type(output)}",
                  "implementation": f"{proof[2]}#{proof[3]}",
                  "flags": [], "leaves": [], "identity": None, "problem": None}
        identities, flags, leaves = [proof], set(), []
        adapted = set()
        try:
            if compiler_json is not None and name in OUTPUTS:
                require_parameter_slots(name, inputs)
            for direction, label, ty in [("input", label, ty) for label, ty in inputs] + [("output", "$return", output)]:
                discovered = compiler_json.exposure(name, direction, label, ty, proof) if compiler_json is not None else None
                if discovered is not None:
                    adapted.add((direction, label))
                    identities.append(("compiler-json", compiler_json.wire.graph_identity,
                                       tuple((adapter, direction) for adapter, direction, _ in compiler_json.ownership)))
                    # The scoped wire projection has its own exact text/codec
                    # contracts. It cannot discharge a sibling Python subtree.
                    obligations = set()
                else:
                    discovered = graph.exposure(ty, owner)
                    obligations = payload_obligations(discovered)
                    if compiler_json is not None and name in OUTPUTS and discovered.numeric_leaves:
                        raise GraphError("numeric sibling outside the exact compiler JSON adapter")
                if ("codec",) in obligations:
                    raise GraphError("source JSON result lacks a proven derived codec")
                if direction == "output" and ("text",) in obligations:
                    domain = _TEXT_OUTPUT_DOMAINS.get(name)
                    if domain is None:
                        raise GraphError("untyped Python text result requires an actual payload contract")
                    identities.append(("text-domain", domain))
                identities.append((direction, label, discovered.identity))
                primitives = {leaf.primitive for leaf in discovered.numeric_leaves}
                leaves += [(direction, label, leaf.path, leaf.primitive) for leaf in discovered.numeric_leaves]
                if primitives & FLOAT_PRIMITIVES:
                    flags.add("float-carrier")
                if direction == "output" and primitives:
                    flags.add("numeric-return")
                elif "dtype" in label.lower() and primitives & INTEGER_PRIMITIVES:
                    flags.add("raw-dtype-int")
                elif primitives:
                    flags.add("numeric-param")
            result["identity"] = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
            result["flags"], result["leaves"] = sorted(flags), leaves
            if adapted:
                expected = {("output", "$return")}
                if name == "chelis_python::eval_json":
                    expected.add(("input", "bindings_json"))
                if adapted != expected:
                    raise GraphError("compiler JSON binding omitted an exact payload direction")
                result["authority"] = "TaggedTransport"
                result["contract"] = "compiler-json/" + name
        except GraphError as error:
            result["problem"] = str(error)
        results.append(result)
    return sorted(results, key=lambda row: (row["kind"], row["id"]))
