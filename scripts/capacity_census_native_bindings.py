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


def require_private_owner(graph, identity, source):
    """Require a complete private struct; its field contracts are checked separately.

    This does not decide that fields, constructors, or arbitrary tagged enums
    implement a transport. It establishes which defining module the compiled
    construction census must audit, without trusting a display path for privacy.
    """
    location = graph.locations.get(identity)
    _require(location is not None, "missing defining native owner")
    item = graph._item(*location)
    body = item.get("inner", {}).get("struct")
    _require(isinstance(body, dict), "native owner must be a concrete struct")
    _require(not body.get("generics", {}).get("params"), "native owner cannot be generic")
    _require(item.get("span", {}).get("filename") == source,
             "native owner moved from its compiled source")
    plain = body.get("kind", {}).get("plain")
    _require(isinstance(plain, dict) and not plain.get("has_stripped_fields"),
             "native owner requires a complete named-field graph")
    fields = plain.get("fields", [])
    _require(fields and len(fields) == len(set(map(str, fields))),
             "native owner fields are missing or duplicated")
    module_identity = identity.rsplit("::", 1)[0]
    module_location = graph.locations.get(module_identity)
    _require(module_location is not None and module_location[0] == location[0],
             "missing defining native module")
    module = graph._item(*module_location)
    module_body = module.get("inner", {}).get("module", {})
    _require(not module_body.get("is_stripped") and item["id"] in module_body.get("items", []),
             "native owner is outside its defining module")
    module_path = module_identity.partition("::")[2]
    private = {"restricted": {"parent": module["id"], "path": "::" + module_path}}
    result = []
    for field_id in fields:
        field = graph._item(location[0], field_id)
        _require(field.get("visibility") == private
                 and set(field.get("inner", {})) == {"struct_field"}
                 and isinstance(field.get("name"), str),
                 "native owner field must remain private to its defining module")
        result.append((field["name"], field["inner"]["struct_field"]))
    _require(len({name for name, _ in result}) == len(result), "duplicate native field name")
    return tuple(result)
