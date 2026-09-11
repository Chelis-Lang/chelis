"""Native binding obligations; structural checks alone issue no authority.

The execution factory must additionally prove actual registration, construction,
validation and conversion ownership, current artifacts, and boundary execution.
"""

from capacity_census_compiler_json import _path
from capacity_census_graph import GraphError

NATIVE_OUTPUTS = {
    "chelis_python::CompiledModel::__call__":
        "chelis_python::native_tensor::CompiledTensorResults",
    "chelis_python::NativeTensor::__dlpack__": "chelis_python::dlpack::DLPackCapsule",
    "chelis_python::NativeTensor::__dlpack_device__": "chelis_python::dlpack::DLPackDevice",
}

_NATIVE_INPUTS = {
    "chelis_python::CompiledModel::__call__": ("self", "py", "args", "kwargs"),
    "chelis_python::NativeTensor::__dlpack__":
        ("self", "stream", "max_version", "dl_device", "copy"),
    "chelis_python::NativeTensor::__dlpack_device__": ("self",),
    "chelis_python::NativeTensor::shape": ("self",),
}


def _require(condition, message):
    if not condition:
        raise GraphError(message)


def require_native_slots(owner, inputs):
    """Select every slot before proving any dynamic request/payload subtree."""
    _require(owner in _NATIVE_INPUTS, "unowned native binding")
    labels = [label for label, _ in inputs]
    _require(len(labels) == len(set(labels)) and set(labels) == set(_NATIVE_INPUTS[owner]),
             "native binding input slots changed or duplicated")


def native_input_role(graph, owner, label, ty):
    """Describe a slot's obligation without treating a decoded request as proof."""
    _require(owner in _NATIVE_INPUTS and label in _NATIVE_INPUTS[owner],
             "unowned native input slot")

    def reference(value):
        _require(isinstance(value, dict) and set(value) == {"borrowed_ref"},
                 "native input requires its exact shared reference")
        body = value["borrowed_ref"]
        _require(body.get("is_mutable") is False and "type" in body,
                 "native input cannot expose a mutable owner")
        return body["type"]

    def nominal(value, expected, arity):
        name, arguments = _path(graph, "chelis_python", value)
        _require(name == expected and len(arguments) == arity,
                 "native input differs from its exact role")
        return arguments

    if label == "self":
        _require(reference(ty) == {"generic": "Self"}, "changed native receiver")
        return "validated-receiver"
    if label == "py":
        nominal(ty, "pyo3::marker::Python", 0)
        return "gil"
    if label in {"args", "kwargs"}:
        if label == "kwargs":
            ty, = nominal(ty, "core::option::Option", 1)
        payload, = nominal(reference(ty), "pyo3::instance::Bound", 1)
        nominal(payload, "pyo3::types::" + ("tuple::PyTuple" if label == "args" else "dict::PyDict"), 0)
        return "untrusted-tensor-" + label
    payload, = nominal(ty, "core::option::Option", 1)
    if label == "copy":
        _require(payload == {"primitive": "bool"}, "copy request must remain optional bool")
        return "untrusted-copy-request"
    request = {"stream": "DLPackStreamRequest", "max_version": "DLPackVersionRequest",
               "dl_device": "DLPackDeviceRequest"}[label]
    nominal(payload, "chelis_python::dlpack::" + request, 0)
    return "untrusted-" + label + "-request"


def native_output_adapter(graph, owner, output):
    """Return an adapter obligation; None leaves shape under exact OP45 authority."""
    _require(owner in _NATIVE_INPUTS, "unowned native binding")
    name, args = _path(graph, "chelis_python", output)
    if owner == "chelis_python::NativeTensor::shape":
        _require(name == "alloc::vec::Vec" and args == [{"primitive": "i64"}],
                 "native shape requires the exact int64 observation")
        return None
    if owner != "chelis_python::NativeTensor::__dlpack_device__":
        _require(name == "pyo3::err::PyResult" and len(args) == 1,
                 "native conversion requires its exact fallible result")
        name, args = _path(graph, "chelis_python", args[0])
    _require(name == NATIVE_OUTPUTS[owner] and not args,
             "native output differs from its concrete validated adapter")
    return name


def _private_owner_context(graph, identity, source):
    location = graph.locations.get(identity)
    _require(location is not None, "missing defining native owner")
    item = graph._item(*location)
    _require(item.get("span", {}).get("filename") == source,
             "native owner moved from its compiled source")
    module_identity = identity.rsplit("::", 1)[0]
    module_location = graph.locations.get(module_identity)
    _require(module_location is not None and module_location[0] == location[0],
             "missing defining native module")
    module = graph._item(*module_location)
    module_body = module.get("inner", {}).get("module", {})
    _require(not module_body.get("is_stripped") and item["id"] in module_body.get("items", []),
             "native owner is outside its defining module")
    module_path = module_identity.partition("::")[2]
    is_root = module["id"] == graph.documents[location[0]]["root"]
    _require(is_root == (module_identity == location[0]),
             "native owner module differs from the defining crate root")
    if is_root:
        _require(module_body.get("is_crate") is True, "native root lacks compiler crate identity")
        # Rustdoc canonicalizes root-private visibility to `crate`: the root
        # already contains every local module. Constructor ownership therefore
        # still requires the complete local-crate MIR census, not this spelling.
        private = "crate"
    else:
        _require(not module_body.get("is_crate"), "nested native module claims crate identity")
        private = {"restricted": {"parent": module["id"], "path": "::" + module_path}}
    return location, item, private


def _private_fields(graph, location, fields, private, *, positional=False):
    _require(isinstance(fields, list) and fields
             and all(field is not None for field in fields)
             and len(fields) == len(set(map(str, fields))),
             "native owner fields are missing, stripped or duplicated")
    result = []
    for index, field_id in enumerate(fields):
        field = graph._item(location[0], field_id)
        _require(field.get("visibility") == private
                 and set(field.get("inner", {})) == {"struct_field"}
                 and isinstance(field.get("name"), str),
                 "native owner field must remain private to its defining module")
        if positional:
            _require(field["name"] == str(index), "native tuple field position changed")
        result.append((field["name"], field["inner"]["struct_field"]))
    _require(len({name for name, _ in result}) == len(result), "duplicate native field name")
    return tuple(result)


def require_private_owner(graph, identity, source):
    """Require complete private named fields; construction and field types are separate."""
    location, item, private = _private_owner_context(graph, identity, source)
    body = item.get("inner", {}).get("struct")
    _require(isinstance(body, dict), "native owner must be a concrete struct")
    _require(not body.get("generics", {}).get("params"), "native owner cannot be generic")
    plain = body.get("kind", {}).get("plain")
    _require(isinstance(plain, dict) and not plain.get("has_stripped_fields"),
             "native owner requires a complete named-field graph")
    return _private_fields(graph, location, plain.get("fields"), private)


def require_private_tuple_owner(graph, identity, source):
    """Expose every private tuple field; wrapping a payload grants no authority."""
    location, item, private = _private_owner_context(graph, identity, source)
    body = item.get("inner", {}).get("struct")
    _require(isinstance(body, dict) and not body.get("generics", {}).get("params"),
             "native tuple owner must be a concrete struct")
    kind = body.get("kind", {})
    _require(isinstance(kind, dict) and set(kind) == {"tuple"},
             "native owner requires its exact tuple structure")
    return tuple(ty for _, ty in _private_fields(
        graph, location, kind["tuple"], private, positional=True,
    ))


def require_private_enum_owner(graph, identity, source):
    """Expose a private owner's complete unit/tuple choices without classifying payloads.

    Rust enum fields inherit visibility from the enum. Their rustdoc `default`
    spelling is accepted only after checking that exact enclosing definition's
    privacy and module membership. Numeric payloads remain visible obligations.
    """
    location, item, private = _private_owner_context(graph, identity, source)
    _require(item.get("visibility") == private,
             "native enum owner must remain private to its defining module")
    body = item.get("inner", {}).get("enum")
    _require(isinstance(body, dict) and not body.get("has_stripped_variants")
             and not body.get("generics", {}).get("params"),
             "native enum owner requires every concrete variant")
    variants = body.get("variants")
    _require(isinstance(variants, list) and variants
             and all(variant is not None for variant in variants)
             and len(variants) == len(set(map(str, variants))),
             "native enum variants are missing, stripped or duplicated")
    result = []
    for variant_id in variants:
        item = graph._item(location[0], variant_id)
        _require(item.get("visibility") == "default"
                 and set(item.get("inner", {})) == {"variant"}
                 and isinstance(item.get("name"), str),
                 "native variant must inherit the exact private enum owner")
        variant = item["inner"]["variant"]
        _require(variant.get("discriminant") is None,
                 "native owner choice cannot introduce an explicit numeric discriminant")
        kind = variant.get("kind")
        if kind == "plain":
            fields = ()
        else:
            _require(isinstance(kind, dict) and set(kind) == {"tuple"},
                     "native owner choice requires its exact unit or tuple structure")
            fields = tuple(ty for _, ty in _private_fields(
                graph, location, kind["tuple"], "default", positional=True,
            ))
        result.append((item["name"], fields))
    _require(len({name for name, _ in result}) == len(result),
             "native enum variant names are duplicated")
    return tuple(result)


def require_tensor_owner_choices(graph):
    """Bind the closed owner choice to retained handles, not arbitrary numeric tags.

    Handle fields, all constructor sites, deletion and current execution still
    require their own proofs. These edges alone cannot issue transport authority.
    """
    choices = require_private_enum_owner(
        graph, "chelis_python::TensorOwner", "crates/chelis-python/src/lib.rs",
    )
    _require(tuple(name for name, _ in choices) == ("Cpu", "Gpu"),
             "native storage owner choices changed")
    for name, fields in choices:
        _require(len(fields) == 1, "native storage owner requires one retained handle")
        container, arguments = _path(graph, "chelis_python", fields[0])
        _require(container == "alloc::sync::Arc" and len(arguments) == 1,
                 "native storage owner must retain its exact shared handle")
        handle, arguments = _path(graph, "chelis_python", arguments[0])
        _require(handle == f"chelis_python::{name}TensorHandle" and not arguments,
                 "native storage owner has an unvalidated payload")
    return tuple(name for name, _ in choices)


def require_native_fields(graph, identity):
    """Validate an exact adapter's private field edges, without issuing authority."""
    def nominal(name, *arguments):
        return ("nominal", name, tuple(arguments))

    tensor = nominal("chelis_python::native_tensor::ValidatedTensor")
    contracts = {
        "chelis_python::NativeTensor": ("lib", (("tensor", tensor),)),
        "chelis_python::native_tensor::ValidatedTensor": (
            "native_tensor", (("inner", nominal("alloc::sync::Arc",
                nominal("chelis_python::native_tensor::ValidatedTensorInner"))),),
        ),
        "chelis_python::native_tensor::CompiledTensorResults": (
            "native_tensor", (("tensors", nominal("alloc::vec::Vec",
                ("tuple", (nominal("alloc::string::String"), tensor)))),),
        ),
        "chelis_python::dlpack::DLPackDevice": ("dlpack", (("tensor", tensor),)),
        "chelis_python::dlpack::DLPackCapsule": (
            "dlpack", (("request", nominal("chelis_python::dlpack::DLPackRequest")),),
        ),
        "chelis_python::dlpack::DLPackRequest": (
            "dlpack", (("tensor", tensor),
                       ("abi", nominal("chelis_python::dlpack::ExportAbi"))),
        ),
    }
    _require(identity in contracts, "unowned native adapter field contract")
    module, expected = contracts[identity]
    fields = require_private_owner(graph, identity, f"crates/chelis-python/src/{module}.rs")

    def shape(ty):
        if isinstance(ty, dict) and set(ty) == {"tuple"}:
            return ("tuple", tuple(shape(element) for element in ty["tuple"]))
        name, arguments = _path(graph, "chelis_python", ty)
        return nominal(name, *(shape(argument) for argument in arguments))

    actual = tuple((name, shape(ty)) for name, ty in fields)
    _require(actual == expected, "native adapter private field edges changed")
    return actual


def require_export_abi_choices(graph):
    """Check the private request's closed protocol choice, never numeric authority."""
    identity = "chelis_python::dlpack::ExportAbi"
    location = graph.locations.get(identity)
    module_location = graph.locations.get("chelis_python::dlpack")
    _require(location is not None and module_location is not None
             and location[0] == module_location[0], "missing defining DLPack ABI enum")
    item = graph._item(*location)
    module = graph._item(*module_location).get("inner", {}).get("module", {})
    _require(not module.get("is_stripped") and item["id"] in module.get("items", []),
             "DLPack ABI enum is outside its defining module")
    _require(item.get("span", {}).get("filename") == "crates/chelis-python/src/dlpack.rs",
             "DLPack ABI enum moved from its compiled source")
    body = item.get("inner", {}).get("enum")
    _require(isinstance(body, dict) and not body.get("has_stripped_variants")
             and not body.get("generics", {}).get("params"),
             "DLPack ABI requires its complete concrete enum")
    variant_ids = body.get("variants", [])
    _require(len(variant_ids) == 2 and len(set(map(str, variant_ids))) == 2,
             "DLPack ABI choices changed or duplicated")
    names = []
    for variant_id in variant_ids:
        variant = graph._item(location[0], variant_id)
        _require(set(variant.get("inner", {})) == {"variant"}
                 and variant["inner"]["variant"].get("kind") == "plain",
                 "DLPack ABI choice cannot carry an arbitrary payload")
        names.append(variant.get("name"))
    _require(names == ["Legacy", "Versioned"], "DLPack ABI choices changed")
    return tuple(names)
