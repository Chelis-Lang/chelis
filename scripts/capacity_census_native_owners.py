"""Resolve four native ownership obligations without issuing authority.

The fixed policy comes from spec/11's validated wrapper design. It joins
checked live registration to current private Rustdoc and format-4 MIR records.
Observed constructors never select or expand policy. Lexical ownership does
not prove validation, lifetimes, execution, or numeric transport authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_native_arguments import ArgumentSlot, argument_forwarding_problems
from capacity_census_native_bindings import (
    native_input_role,
    native_output_adapter,
    require_export_abi_choices,
    require_native_fields,
    require_native_slots,
    require_tensor_owner_choices,
)
from capacity_census_native_construction import constructor_scope_ownership
from capacity_census_native_flow import (
    AdapterFlowObligation,
    CallStage,
    ConstructorOwnership,
    DefinitionIdentity,
    NativeReceiver,
    native_flow_problems,
)
from capacity_census_native_registration import CheckedNativeRegistrations
from capacity_census_bindings import registration_roots


@dataclass(frozen=True)
class NativeEntryObligation:
    public: str
    registered_implementation: str
    registration_proof: tuple
    implementation: DefinitionIdentity
    input_roles: tuple[tuple[str, str], ...]
    output_adapter: str | None
    stages: tuple[CallStage, ...]
    constructors: tuple[ConstructorOwnership, ...]
    receiver: NativeReceiver
    keyword_slots: tuple[ArgumentSlot, ...] = ()


@dataclass(frozen=True)
class NativeOwnershipReport:
    """Bounded observations with deliberately no authority/permission API."""

    entries: tuple[NativeEntryObligation, ...]
    private_fields: tuple[tuple[str, tuple], ...]
    export_abi: tuple[str, ...]
    tensor_owner_choices: tuple[str, ...]
    constructor_scopes: tuple[ConstructorOwnership, ...]


_REGISTERED = (
    ("chelis_python::CompiledModel::__call__",
     "chelis_python::NativeCompiledModel::__call__#method",
     "::NativeCompiledModel", "__call__"),
    ("chelis_python::NativeTensor::shape",
     "chelis_python::NativeTensor::shape#getter", "::NativeTensor", "shape"),
    ("chelis_python::NativeTensor::__dlpack_device__",
     "chelis_python::NativeTensor::__dlpack_device__#method",
     "::NativeTensor", "__dlpack_device__"),
    ("chelis_python::NativeTensor::__dlpack__",
     "chelis_python::NativeTensor::__dlpack__#method",
     "::NativeTensor", "__dlpack__"),
)

_PRIVATE_FIELDS = (
    "chelis_python::NativeTensor",
    "chelis_python::native_tensor::ValidatedTensor",
    "chelis_python::native_tensor::CompiledTensorResults",
    "chelis_python::dlpack::DLPackDevice",
    "chelis_python::dlpack::DLPackCapsule",
    "chelis_python::dlpack::DLPackRequest",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError(message)


def _identity(record: Any) -> DefinitionIdentity:
    try:
        return DefinitionIdentity.from_record(record)
    except (KeyError, TypeError, ValueError) as error:
        raise GraphError(f"malformed native definition identity: {error}") from error


def _identity_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if {"stable_crate_id", "def_id", "def_path_hash"} <= value.keys():
            yield value
        for child in value.values():
            yield from _identity_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identity_records(child)


def _definition(raw: dict, crate: str, path: str) -> DefinitionIdentity:
    found = {
        _identity(record)
        for record in _identity_records(raw)
        if record.get("crate") == crate and record.get("path") == path
    }
    _require(len(found) == 1, f"missing or ambiguous exact definition {crate}{path}")
    return found.pop()


def _associated(raw: dict, self_path: str, name: str,
                trait: tuple[str, str] | None = None) -> DefinitionIdentity:
    """Resolve by exact self and trait identities, never impl ordinals."""
    expected_owner = _definition(raw, "chelis_python", self_path)
    expected_trait = None if trait is None else _definition(raw, *trait)
    found = set()
    for body in raw["bodies"]:
        implementation = body.get("implementation")
        definition = body.get("definition", {})
        if (body.get("kind") != "AssocFn"
                or definition.get("crate") != "chelis_python"
                or definition.get("item_name") != name
                or not isinstance(implementation, dict)):
            continue
        owner = implementation.get("self_type", {}).get("nominal")
        if not isinstance(owner, dict):
            continue
        if (_identity(owner) != expected_owner
                or owner.get("crate") != "chelis_python"
                or owner.get("path") != self_path):
            continue
        actual_trait = implementation.get("trait")
        if trait is None:
            if actual_trait is not None:
                continue
        else:
            if not isinstance(actual_trait, dict):
                continue
            if (_identity(actual_trait) != expected_trait or (
                actual_trait.get("crate"), actual_trait.get("path")
            ) != trait):
                continue
        found.add(_identity(definition))
    label = f"{self_path}::{name}"
    _require(len(found) == 1, f"missing or ambiguous governing method {label}")
    return found.pop()


def _function(raw: dict, path: str) -> DefinitionIdentity:
    found = {
        _identity(body["definition"])
        for body in raw["bodies"]
        if body.get("kind") == "Fn"
        and body.get("implementation") is None
        and body.get("definition", {}).get("crate") == "chelis_python"
        and body["definition"].get("path") == path
    }
    _require(len(found) == 1, f"missing or ambiguous governing function {path}")
    return found.pop()


def _registered_rustdoc(
    graph: RustdocGraph, registrations: CheckedNativeRegistrations
) -> dict[str, tuple[dict, str | None, tuple]]:
    """Join the sealed live packet to exact Rustdoc items and source spans."""
    packet = registrations.packet
    provenance = {
        key: packet[key] for key in ("source_path", "source_sha256", "registrations")
    }
    methods = tuple(public.removeprefix("chelis_python::") for public, *_ in _REGISTERED)
    rows = registration_roots(
        graph, (), methods, packet["classes"], provenance
    )
    result = {}
    for kind, public, item, owner, proof in rows:
        _require(kind == "binding-pymethod" and public not in result,
                 "registered Rustdoc root changed or duplicated")
        function = item.get("inner", {}).get("function")
        _require(isinstance(function, dict)
                 and item.get("visibility") == "crate"
                 and item.get("span", {}).get("filename") == packet["source_path"]
                 and function.get("has_body") is True
                 and not function.get("generics", {}).get("params")
                 and not function.get("generics", {}).get("where_predicates"),
                 f"registered Rustdoc method changed ownership {public}")
        result[public] = (function["sig"], owner, proof)
    _require(set(result) == {public for public, *_ in _REGISTERED},
             "registered Rustdoc roots differ from the live four")
    return result


def _entry_signature(graph: RustdocGraph, public: str, signature: dict):
    inputs = signature.get("inputs")
    _require(isinstance(inputs, list), f"registered method lacks inputs {public}")
    require_native_slots(public, inputs)
    roles = tuple((label, native_input_role(graph, public, label, ty)) for label, ty in inputs)
    return roles, native_output_adapter(graph, public, signature.get("output"))


def _result_success(raw: dict) -> DefinitionIdentity:
    found = {
        _identity(row["variant_definition"])
        for row in raw["aggregates"]
        if row.get("variant") == "Ok"
        and row.get("variant_definition", {}).get("crate") == "core"
        and row["variant_definition"].get("path") == "::result::Result::Ok"
    }
    _require(len(found) == 1, "missing or ambiguous exact core Result::Ok constructor")
    return found.pop()


def _result_controls(raw: dict, entry: DefinitionIdentity) -> tuple[CallStage, ...]:
    result = []
    for crate, path in (
        ("core", "::ops::try_trait::Try::branch"),
        ("core", "::ops::try_trait::FromResidual::from_residual"),
    ):
        matches = []
        for call in raw["calls"]:
            if _identity(call.get("caller", {}).get("definition")) != entry:
                continue
            callee = call.get("callee")
            if not isinstance(callee, dict):
                continue
            declared = callee.get("definition", {})
            resolved = callee.get("resolved", {}).get("definition")
            if declared.get("crate") == crate and declared.get("path") == path:
                matches.append(CallStage(_identity(declared), _identity(resolved)))
        _require(len(matches) == 1,
                 f"missing or ambiguous exact Result control {crate}{path}")
        result.extend(matches)
    return tuple(result)


def _check_flow(raw: dict, name: str, obligation: AdapterFlowObligation) -> None:
    problems = native_flow_problems(raw, obligation)
    _require(not problems, f"{name} ownership flow failed: {'; '.join(problems)}")


def _registered_implementation(
    raw: dict, public: str, owner: str, name: str
) -> DefinitionIdentity:
    try:
        return _associated(raw, owner, name)
    except GraphError as error:
        raise GraphError(
            f"registered implementation changed for {public}: {error}"
        ) from error


def _constructor_scope(
    raw: dict, carrier: DefinitionIdentity, roots: set[DefinitionIdentity]
) -> ConstructorOwnership:
    try:
        return constructor_scope_ownership(raw, carrier, roots)
    except ValueError as error:
        raise GraphError(f"native constructor obligation failed: {error}") from error


def resolve_native_ownership(
    graph: RustdocGraph,
    evidence: dict,
    registrations: CheckedNativeRegistrations,
) -> NativeOwnershipReport:
    """Resolve the fixed report from current registration, Rustdoc, and MIR.

    A registration table, saved descriptor, or observed constructor set is not
    accepted as policy. The returned value records obligations only.
    """
    _require(type(graph) is RustdocGraph, "native ownership needs exact Rustdoc graph")
    _require(type(registrations) is CheckedNativeRegistrations,
             "native ownership needs checked native registrations")
    roots = registrations.validate()
    expected = {public: rust for public, rust, _, _ in _REGISTERED}
    _require(roots == expected, "checked native registrations changed")
    _require(isinstance(evidence, dict)
             and evidence.get("format") == 4
             and evidence.get("scope") == "native-bindings"
             and evidence.get("errors") == []
             and all(isinstance(evidence.get(key), list) for key in (
                 "bodies", "calls", "aggregates", "constructor_uses", "flows"
             )), "native ownership needs complete format-4 compiler evidence")
    crate = evidence.get("crate")
    _require(isinstance(crate, dict)
             and crate.get("crate") == "chelis_python"
             and crate.get("item_name") == "chelis_python"
             and crate.get("path") == ""
             and crate.get("def_id") == "0:0",
             "native ownership evidence is not the chelis_python crate root")

    rustdoc = _registered_rustdoc(graph, registrations)
    registered = {}
    signatures = {}
    for public, _, owner, name in _REGISTERED:
        signature, rustdoc_owner, _ = rustdoc[public]
        _require(rustdoc_owner == "chelis_python" + owner,
                 f"registered Rustdoc owner changed {public}")
        signatures[public] = _entry_signature(graph, public, signature)
        registered[public] = _registered_implementation(
            evidence, public, owner, name
        )

    model = _definition(evidence, "chelis_python", "::NativeCompiledModel")
    native_tensor = _definition(evidence, "chelis_python", "::NativeTensor")
    validated = _definition(evidence, "chelis_python", "::native_tensor::ValidatedTensor")
    inputs = _definition(evidence, "chelis_python", "::native_tensor::CompiledInputs")
    raw_outputs = _definition(evidence, "chelis_python", "::native_tensor::RawOutputs")
    results = _definition(evidence, "chelis_python", "::native_tensor::CompiledTensorResults")
    request = _definition(evidence, "chelis_python", "::dlpack::DLPackRequest")
    device = _definition(evidence, "chelis_python", "::dlpack::DLPackDevice")
    capsule = _definition(evidence, "chelis_python", "::dlpack::DLPackCapsule")

    admit = _associated(evidence, "::native_tensor::CompiledInputs", "admit")
    execute = _function(evidence, "::native_tensor::execute_checked")
    adopt_outputs = _associated(
        evidence, "::native_tensor::CompiledTensorResults", "adopt_outputs"
    )
    validated_shape = _associated(evidence, "::native_tensor::ValidatedTensor", "shape")
    from_device = _associated(evidence, "::dlpack::DLPackDevice", "from_validated")
    validate = _associated(evidence, "::dlpack::DLPackRequest", "validate")
    export = _associated(evidence, "::native_tensor::ValidatedTensor", "export")
    from_capsule = _associated(evidence, "::dlpack::DLPackCapsule", "from_validated")
    success = _result_success(evidence)

    call_constructors = (
        ConstructorOwnership(inputs, frozenset({admit}), frozenset({admit})),
        ConstructorOwnership(raw_outputs, frozenset({execute}), frozenset({execute})),
        ConstructorOwnership(results, frozenset({adopt_outputs}), frozenset({adopt_outputs})),
    )
    call_stages = tuple(CallStage(value, value) for value in (admit, execute, adopt_outputs))
    call_receiver = NativeReceiver(1, model)
    _check_flow(evidence, "compiled call", AdapterFlowObligation(
        entry=registered[_REGISTERED[0][0]], stages=call_stages,
        constructors=call_constructors,
        native_types=frozenset({model, inputs, raw_outputs, results}),
        success_variant=success,
        control_calls=_result_controls(evidence, registered[_REGISTERED[0][0]]),
        receiver=call_receiver,
    ))

    shape_stages = (CallStage(validated_shape, validated_shape),)
    shape_receiver = NativeReceiver(1, native_tensor)
    _check_flow(evidence, "shape", AdapterFlowObligation(
        entry=registered[_REGISTERED[1][0]], stages=shape_stages,
        constructors=(), native_types=frozenset({native_tensor, validated}),
        receiver=shape_receiver,
    ))

    device_constructors = (
        ConstructorOwnership(device, frozenset({from_device}), frozenset({from_device})),
    )
    device_stages = (CallStage(from_device, from_device),)
    device_receiver = NativeReceiver(1, native_tensor)
    _check_flow(evidence, "device", AdapterFlowObligation(
        entry=registered[_REGISTERED[2][0]], stages=device_stages,
        constructors=device_constructors,
        native_types=frozenset({native_tensor, validated, device}),
        receiver=device_receiver,
    ))

    export_constructors = (
        ConstructorOwnership(request, frozenset({validate}), frozenset({validate})),
        ConstructorOwnership(capsule, frozenset({from_capsule}), frozenset({from_capsule})),
    )
    export_stages = tuple(CallStage(value, value) for value in (validate, export))
    export_receiver = NativeReceiver(1, native_tensor)
    _check_flow(evidence, "DLPack export", AdapterFlowObligation(
        entry=registered[_REGISTERED[3][0]], stages=export_stages,
        constructors=export_constructors,
        native_types=frozenset({native_tensor, validated, request, capsule}),
        success_variant=success,
        control_calls=_result_controls(evidence, registered[_REGISTERED[3][0]]),
        receiver=export_receiver,
    ))
    keyword_slots = tuple(
        ArgumentSlot(local=index + 1, argument=index) for index in range(1, 5)
    )
    keyword_problems = argument_forwarding_problems(
        evidence, registered[_REGISTERED[3][0]], CallStage(validate, validate), keyword_slots
    )
    _require(not keyword_problems,
             "DLPack keyword forwarding failed: " + "; ".join(keyword_problems))

    private_fields = tuple(
        (identity, require_native_fields(graph, identity)) for identity in _PRIVATE_FIELDS
    )
    export_abi = require_export_abi_choices(graph)
    tensor_choices = require_tensor_owner_choices(graph)

    adopt = _associated(evidence, "::native_tensor::ValidatedTensor", "adopt")
    validated_clone = _associated(
        evidence, "::native_tensor::ValidatedTensor", "clone",
        ("core", "::clone::Clone"),
    )
    tensor_owner_clone = _associated(
        evidence, "::TensorOwner", "clone", ("core", "::clone::Clone")
    )
    results_into_python = _associated(
        evidence, "::native_tensor::CompiledTensorResults", "into_pyobject",
        ("pyo3", "::conversion::IntoPyObject"),
    )
    capsule_into_python = _associated(
        evidence, "::dlpack::DLPackCapsule", "into_pyobject",
        ("pyo3", "::conversion::IntoPyObject"),
    )
    gpu_input = _function(evidence, "::gpu_input_tensor")
    policies = (
        (native_tensor, {results_into_python}),
        (validated, {adopt, validated_clone}),
        (results, {adopt_outputs}),
        (device, {from_device}),
        (capsule, {from_capsule}),
        (request, {validate}),
        (_definition(evidence, "chelis_python", "::native_tensor::ValidatedTensorInner"),
         {adopt}),
        (_definition(evidence, "chelis_python", "::native_tensor::ForeignDataPointer"),
         {adopt}),
        (_definition(evidence, "chelis_python", "::CpuTensorHandle"), {execute}),
        (_definition(evidence, "chelis_python", "::GpuTensorHandle"),
         {execute, gpu_input}),
        (_definition(evidence, "chelis_python", "::TensorOwner"),
         {execute, tensor_owner_clone}),
        (_definition(evidence, "chelis_python", "::dlpack::Context"),
         {capsule_into_python}),
    )
    constructor_scopes = tuple(
        _constructor_scope(evidence, carrier, roots)
        for carrier, roots in policies
    )

    entry_values = (
        (call_stages, call_constructors, call_receiver, ()),
        (shape_stages, (), shape_receiver, ()),
        (device_stages, device_constructors, device_receiver, ()),
        (export_stages, export_constructors, export_receiver, keyword_slots),
    )
    entries = tuple(
        NativeEntryObligation(
            public=public,
            registered_implementation=roots[public],
            registration_proof=rustdoc[public][2],
            implementation=registered[public],
            input_roles=signatures[public][0],
            output_adapter=signatures[public][1],
            stages=entry_values[index][0],
            constructors=entry_values[index][1],
            receiver=entry_values[index][2],
            keyword_slots=entry_values[index][3],
        )
        for index, (public, rust, _, _) in enumerate(_REGISTERED)
    )
    return NativeOwnershipReport(
        entries=entries,
        private_fields=private_fields,
        export_abi=export_abi,
        tensor_owner_choices=tensor_choices,
        constructor_scopes=constructor_scopes,
    )
