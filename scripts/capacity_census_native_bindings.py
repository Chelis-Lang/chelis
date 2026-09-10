"""Native binding obligations; structural checks alone issue no authority.

The execution factory must additionally prove actual registration, construction,
validation and conversion ownership, current artifacts, and boundary execution.
"""

NATIVE_OUTPUTS = {
    "chelis_python::CompiledModel::__call__":
        "chelis_python::native_tensor::CompiledTensorResults",
    "chelis_python::NativeTensor::__dlpack__": "chelis_python::dlpack::DLPackCapsule",
    "chelis_python::NativeTensor::__dlpack_device__": "chelis_python::dlpack::DLPackDevice",
}


def require_native_slots(owner, inputs):
    raise NotImplementedError("exact native parameter obligations")


def native_output_adapter(graph, owner, output):
    raise NotImplementedError("exact native output obligations")


def require_private_owner(graph, identity, source):
    raise NotImplementedError("defining private construction owner")
