"""Build and inspect the four actual native Python descriptors.

This sealed collection binds current compiled registration only. Numeric or
transport authority additionally requires ownership and boundary execution.
"""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import _target_lease, canonical, source_identity


SOURCE = "crates/chelis-python/src/lib.rs"
CLASSES = {"CompiledModel": "chelis_python::NativeCompiledModel", "NativeTensor": "chelis_python::NativeTensor"}
SLOTS = (
    ("CompiledModel", "NativeCompiledModel", "__call__", "method", "wrapper_descriptor"),
    ("NativeTensor", "NativeTensor", "__dlpack__", "method", "method_descriptor"),
    ("NativeTensor", "NativeTensor", "__dlpack_device__", "method", "method_descriptor"),
    ("NativeTensor", "NativeTensor", "shape", "getter", "getset_descriptor"),
)


def _require(condition, message):
    if not condition:
        raise GraphError(message)


def _digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError) as error:
        raise GraphError("unreadable native registration artifact: " + str(error)) from error


def validate_native_registration(root, packet):
    """Select exact owner obligations from a packet, without creating a witness."""
    _require(isinstance(packet, dict) and set(packet) == {
        "source_path", "source_sha256", "registrations", "classes", "descriptors",
    }, "missing or malformed native registration packet")
    _require(packet["source_path"] == SOURCE and packet["source_sha256"] == _digest(Path(root) / SOURCE),
             "native registration source changed")
    _require(packet["classes"] == CLASSES, "native compiled class identities changed")
    registrations, descriptors = packet["registrations"], packet["descriptors"]
    _require(isinstance(registrations, list) and isinstance(descriptors, list)
             and len(registrations) == len(descriptors) == len(SLOTS),
             "native registration requires exactly four exposures")
    for row in registrations:
        _require(isinstance(row, dict) and set(row) == {
            "owner", "python_name", "rust_name", "kind", "line", "column",
        }, "malformed native source registration")
        _require(all(type(row[key]) is int and row[key] > 0 for key in ("line", "column")),
                 "native source registration lacks exact coordinates")
    for row in descriptors:
        _require(isinstance(row, dict) and set(row) == {
            "owner", "rust_owner", "python_name", "kind", "descriptor",
        }, "malformed native live descriptor")
    roots = {}
    for public, owner, name, kind, descriptor in SLOTS:
        rows = [row for row in registrations if row["owner"] == owner and row["python_name"] == name]
        _require(len(rows) == 1 and rows[0]["rust_name"] == name and rows[0]["kind"] == kind,
                 "native source registration changed or duplicated")
        expected = dict(owner=public, rust_owner="chelis_python::" + owner, python_name=name,
                        kind=kind, descriptor=descriptor)
        _require(descriptors.count(expected) == 1, "native live descriptor changed or duplicated")
        roots[f"chelis_python::{public}::{name}"] = f"chelis_python::{owner}::{name}#{kind}"
    return roots


@dataclass(frozen=True, init=False, slots=True)
class CheckedNativeRegistrations:
    root: Path
    source_sha256: str
    packet: dict
    packet_sha256: str
    binary: tuple
    processes: tuple

    def __new__(cls, *args, **kwargs):
        raise TypeError("native registration requires the actual current compiled probe")

    def validate(self):
        _require(source_identity(self.root) == self.source_sha256, "stale native registration source")
        _require(hashlib.sha256(canonical(self.packet).encode()).hexdigest() == self.packet_sha256,
                 "native registration packet changed")
        _require(_digest(self.binary[0]) == self.binary[1], "native registration binary changed")
        for process in self.processes:
            for stream in ("stdout", "stderr"):
                artifact = process[stream]
                _require(_digest(artifact["path"]) == artifact["sha256"],
                         "native registration transcript changed")
        return validate_native_registration(self.root, self.packet)


def collect_native_registration(root: Path, target: Path):
    """Build and execute one fixed probe; no supplied artifact or packet input."""
    from capacity_census_wire_calls import record_process

    root, target = root.resolve(), target.resolve()
    _require(target.is_relative_to(root / "target"), "native registration target must belong to this worktree")
    source_sha256 = source_identity(root)
    source = root / "crates/chelis-python/examples/native_binding_probe.rs"
    environment = {**os.environ, "CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "1",
                   "PYO3_PYTHON": sys.executable, "VIRTUAL_ENV": sys.prefix}
    command = ["cargo", "build", "--locked", "-p", "chelis-python", "--example", "native_binding_probe", "--message-format=json"]
    with _target_lease(target):
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
        build = record_process(target / "native-registration-processes/build", command, result)
        _require(result.returncode == 0, "native registration probe build failed: " + result.stderr)
        try:
            records = [json.loads(line) for line in result.stdout.splitlines()]
        except json.JSONDecodeError as error:
            raise GraphError("native registration Cargo output is malformed") from error
        candidates = [Path(row["executable"]).resolve() for row in records
                      if row.get("reason") == "compiler-artifact" and row.get("executable")
                      and row.get("target", {}).get("name") == "native_binding_probe"
                      and row["target"].get("kind") == ["example"]
                      and row["target"].get("src_path") == str(source)]
        _require(len(candidates) == 1 and candidates[0].is_relative_to(target),
                 "missing exact compiled native registration probe")
        binary = (str(candidates[0]), _digest(candidates[0]))
        result = subprocess.run([binary[0]], cwd=root, env=environment, capture_output=True, text=True)
        execution = record_process(target / "native-registration-processes/execution", [binary[0]], result)
        _require(result.returncode == 0 and _digest(binary[0]) == binary[1],
                 "native registration probe failed or changed: " + result.stderr)
        try:
            packet = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise GraphError("native registration probe output is malformed") from error
    validate_native_registration(root, packet)
    witness = object.__new__(CheckedNativeRegistrations)
    for key, value in dict(root=root, source_sha256=source_sha256, packet=packet,
                           packet_sha256=hashlib.sha256(canonical(packet).encode()).hexdigest(),
                           binary=binary, processes=(build, execution)).items():
        object.__setattr__(witness, key, value)
    witness.validate()
    return witness
