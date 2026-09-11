"""Issue final authority for the four fixed native Python binding contracts.

Ownership, registration, Rustdoc shape, MIR flow, and boundary execution are
all necessary and individually insufficient. Only ``verify_native_bindings``
can construct the sealed witness that joins them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from capacity_census_graph import GraphError, Leaf, RustdocGraph
from capacity_census_native_execution import (
    CheckedNativeExecution,
    collect_native_execution,
)
from capacity_census_native_owners import (
    NativeOwnershipReport,
    resolve_native_ownership,
)
from capacity_census_native_registration import (
    CheckedNativeRegistrations,
    collect_native_registration,
)
from capacity_census_wire_adapters import canonical, source_identity


NATIVE_CLASSIFICATIONS = {
    "chelis_python::CompiledModel::__call__": (
        "TaggedTransport",
        "native/compiled-tensor-call",
        ("float-carrier", "numeric-param", "numeric-return"),
    ),
    "chelis_python::NativeTensor::__dlpack__": (
        "TaggedTransport",
        "native/dlpack-capsule",
        ("float-carrier", "numeric-param", "numeric-return"),
    ),
    "chelis_python::NativeTensor::__dlpack_device__": (
        "TaggedTransport",
        "native/dlpack-device",
        ("numeric-return",),
    ),
    "chelis_python::NativeTensor::shape": (
        "NumericOperation",
        "[05-OP-45]",
        ("numeric-return",),
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphError(message)


@dataclass(frozen=True)
class NativeProjection:
    identity: str
    numeric_leaves: tuple[Leaf, ...]


def _slot_leaves(name: str, direction: str, label: str) -> tuple[Leaf, ...]:
    path = f"{name}:{direction}:{label}"
    if name == "chelis_python::CompiledModel::__call__":
        if direction == "output" or label in {"args", "kwargs"}:
            return (Leaf(path, "f64"),)
    elif name == "chelis_python::NativeTensor::__dlpack__":
        if direction == "output":
            return (Leaf(path, "f64"),)
        if label in {"stream", "max_version", "dl_device"}:
            return (Leaf(path, "i64"),)
    elif name == "chelis_python::NativeTensor::__dlpack_device__":
        if direction == "output":
            return (Leaf(path, "i32"),)
    elif name == "chelis_python::NativeTensor::shape" and direction == "output":
        return (Leaf(path, "i64"),)
    return ()


@dataclass(frozen=True, init=False, slots=True)
class VerifiedNativeBindings:
    root: Path
    source_sha256: str
    graph: RustdocGraph
    ownership: NativeOwnershipReport
    registrations: CheckedNativeRegistrations
    compiler_evidence: dict
    execution: CheckedNativeExecution

    def __new__(cls, *args, **kwargs):
        raise TypeError("native binding authority requires actual current execution")

    def validate(self) -> None:
        _require(source_identity(self.root) == self.source_sha256,
                 "stale native binding source")
        _require(type(self.graph) is RustdocGraph,
                 "native authority requires the exact Rustdoc graph")
        _require(type(self.registrations) is CheckedNativeRegistrations,
                 "native authority requires checked registration")
        _require(type(self.execution) is CheckedNativeExecution,
                 "native authority requires checked boundary execution")
        self.registrations.validate()
        self.execution.validate()
        current = resolve_native_ownership(
            self.graph, self.compiler_evidence["evidence"], self.registrations
        )
        _require(current == self.ownership, "native ownership obligations changed")
        _require(
            {entry.public for entry in current.entries}
            == set(NATIVE_CLASSIFICATIONS),
            "native authority entry set changed",
        )

    def classification(self, name: str) -> tuple[str, str, tuple[str, ...]]:
        self.validate()
        _require(name in NATIVE_CLASSIFICATIONS, "unowned native binding")
        return NATIVE_CLASSIFICATIONS[name]

    def exposure(self, name: str, direction: str, label: str, proof: tuple) -> NativeProjection:
        self.validate()
        entries = [entry for entry in self.ownership.entries if entry.public == name]
        _require(len(entries) == 1, "missing native ownership entry")
        entry = entries[0]
        _require(proof == entry.registration_proof,
                 "native exposure differs from its registered Rust owner")
        if direction == "input":
            roles = dict(entry.input_roles)
            _require(label in roles, "native input slot changed")
            role = roles[label]
        else:
            _require(direction == "output" and label == "$return",
                     "unknown native binding direction")
            role = entry.output_adapter or "[05-OP-45]"
        payload = (
            name,
            direction,
            label,
            role,
            entry.registered_implementation,
            entry.registration_proof,
            self.registrations.packet_sha256,
            self.execution.identity_sha256,
        )
        identity = hashlib.sha256(canonical(payload).encode()).hexdigest()
        return NativeProjection(identity, _slot_leaves(name, direction, label))

    def execution_report(self) -> dict:
        self.validate()
        return {
            "source_sha256": self.source_sha256,
            "registration_packet_sha256": self.registrations.packet_sha256,
            "compiler_identity": self.compiler_evidence["driver_build"]["identity"],
            "ownership": [
                {
                    "public": entry.public,
                    "implementation": entry.registered_implementation,
                    "input_roles": list(entry.input_roles),
                    "output_adapter": entry.output_adapter,
                }
                for entry in self.ownership.entries
            ],
            "execution": {
                "packet_sha256": self.execution.packet_sha256,
                "identity_sha256": self.execution.identity_sha256,
                "selected": len(self.execution.packet["selected"]),
                "captures": len(self.execution.packet["captures"]),
                "binaries": len(self.execution.packet["binaries"]),
            },
        }


def verify_native_bindings(root: Path, target: Path) -> VerifiedNativeBindings:
    """Collect every current native witness; accept no saved evidence input."""

    from capacity_census_typed import generate_rustdoc_json
    from capacity_census_wire_calls import build_driver, collect_library

    root, target = root.resolve(), target.resolve()
    _require(target.is_relative_to(root / "target"),
             "native authority target must belong to this worktree")
    before = source_identity(root)
    registrations = collect_native_registration(root, target)
    documents = [
        generate_rustdoc_json(
            root=root,
            package=package,
            crate_name=crate,
            target_dir=target,
        )
        for package, crate in (
            ("chelis-python", "chelis_python"),
            ("chelis-abi", "chelis_abi"),
        )
    ]
    graph = RustdocGraph(documents)
    driver = build_driver(root, target / "native-authority-driver")
    compiler_evidence = collect_library(
        root, target, driver, scope="native-bindings"
    )
    ownership = resolve_native_ownership(
        graph, compiler_evidence["evidence"], registrations
    )
    execution = collect_native_execution(root, target)
    _require(source_identity(root) == before,
             "source changed during native binding verification")
    witness = object.__new__(VerifiedNativeBindings)
    for key, value in {
        "root": root,
        "source_sha256": before,
        "graph": graph,
        "ownership": ownership,
        "registrations": registrations,
        "compiler_evidence": compiler_evidence,
        "execution": execution,
    }.items():
        object.__setattr__(witness, key, value)
    witness.validate()
    return witness
