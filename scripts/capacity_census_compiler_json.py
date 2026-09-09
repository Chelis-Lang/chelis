"""Execute the five concrete Python JSON adapter contracts from spec/11 §1.1.

The contract table selects obligations. Only current compiler ownership, live
PyO3 registration, native execution and the private wire verifier discharge
them. This does not classify the four deferred native tensor bindings.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import _target_lease, canonical, source_identity
from capacity_census_wire_invocation_owners import identity, shape_key
from capacity_census_wire_runner import build_and_run_rust_test, run_python_tests

MODULE = "chelis_python::compiler_json::"
SOURCE = "crates/chelis-python/src/compiler_json.rs"
API = "chelis_compiler_api::schema::"
OUTPUTS = {
    "chelis_python::check_json": ("CheckJson", "CheckResult"),
    "chelis_python::compile_json": ("CompileJson", "CompileResult"),
    "chelis_python::desugar_json": ("DesugarJson", "DesugarResult"),
    "chelis_python::eval_json": ("EvalJson", "EvalResult"),
}
INPUT_ROLES = {
    "chelis_python::check_json": {"py": "gil", "source": "text", "source_kind": "text"},
    "chelis_python::compile_json": {"py": "gil", "source": "text", "target": "text", "source_kind": "text", "entry_name": "optional-name"},
    "chelis_python::desugar_json": {"py": "gil", "source": "text"},
    "chelis_python::eval_json": {"py": "gil", "source": "text", "bindings_json": "tensor-bindings", "source_kind": "text", "project_root": "optional-path"},
}
NATIVE_CASES = (
    "native_check_json_preserves_success_and_failure_reports",
    "native_compile_json_preserves_its_typed_result_contract",
    "native_desugar_json_preserves_source_numbers_and_rejects_invalid_source",
    "native_eval_json_preserves_exact_execution_values",
    "native_eval_bindings_reject_invalid_payloads_before_either_route",
    "native_context_eval_json_uses_the_same_execution_codec",
    "compiler_json_conversion_rejects_invalid_typed_execution_envelopes",
    "compiler_json_construction_accepts_only_its_exact_result_type",
)
PYTHON_CASES = tuple("test_capacity_census_compiler_json.CompilerJsonAuthority." + name for name in (
    "test_actual_registration_conversion_and_wire_execution_issue_authority",
    "test_each_adapter_binds_exact_root_and_direction",
    "test_compiled_conversion_ownership_cannot_be_a_named_helper",
    "test_eval_reader_ownership_requires_deserialization_into_the_tensor_map",
    "test_compiler_json_authority_stays_inside_its_adapter_subtree",
    "test_stale_or_supplied_receipts_cannot_issue_binding_authority",
    "test_all_selected_native_cases_must_execute_without_skips",
    "test_construction_controls_require_exact_compiler_success_or_failure",
))


def _require(condition, message):
    if not condition:
        raise GraphError(message)


def _nominal(name, *arguments):
    return ("nominal", name, tuple(arguments))


MAP = _nominal(
    "alloc::collections::btree::map::BTreeMap",
    _nominal("alloc::string::String"),
    _nominal(API + "TensorValue"),
    _nominal("alloc::alloc::Global"),
)


def validate_conversion_calls(raw):
    """Check compiler-derived obligations; a successful check is not a witness."""
    _require(
        raw.get("format") == 2 and raw.get("scope") == "compiler-json"
        and identity(raw["crate"]) == "chelis_python::"
        and not raw.get("errors") and raw.get("bodies"),
        "missing actual compiler JSON ownership scope",
    )
    traits = {identity(trait): trait for trait in raw["conversion_traits"]}
    _require(set(traits) == {"pyo3::conversion::IntoPyObject", "pyo3::conversion::FromPyObject"},
             "missing defining Python conversion traits")
    _require(identity(raw["deserialize_trait"]) == "serde_core::de::Deserialize",
             "compiler JSON ingress requires its defining Deserialize trait")
    _require(not raw["codec_calls"] and not raw["schema_calls"] and not raw["dynamic_returns"],
             "compiler JSON adapter has additional or dynamic codec obligations")
    expected = {MODULE + adapter: API + result for adapter, result in OUTPUTS.values()}
    seen = set()
    records = []
    for direction, calls in (("output", raw["calls"]), ("input", raw["reader_calls"])):
        for call in calls:
            implementation = call["caller"].get("implementation")
            _require(isinstance(implementation, dict), "conversion is not an actual trait implementation")
            trait = "IntoPyObject" if direction == "output" else "FromPyObject"
            _require(implementation.get("trait") == traits["pyo3::conversion::" + trait],
                     "codec belongs to a helper instead of the actual conversion implementation")
            owner = shape_key(implementation["self_type"]["shape"])
            _require(owner[0] == "nominal" and not owner[2], "conversion owner must be concrete")
            name = owner[1]
            _require(name not in seen, "duplicate conversion codec owner")
            seen.add(name)
            if direction == "output":
                _require(name in expected, "unowned Python JSON output adapter")
                payload = _nominal(expected[name])
                callee, method = "serde_json::ser::to_string", "into_pyobject"
                _require(not call["serializers"] and not call["local_callee"],
                         "output adapter must use the exact existing serializer")
            else:
                _require(name == MODULE + "EvalBindingsJson", "unowned Python JSON input adapter")
                payload = MAP
                callee, method = "serde_json::de::from_str", "extract_bound"
            _require(call["caller"]["definition"].get("item_name") == method,
                     "wrong Python conversion method")
            _require(identity(call["callee"]) == callee, "wrong defining JSON codec")
            _require(tuple(shape_key(p["shape"]) for p in call["payloads"]) == (payload,),
                     "compiled codec payload differs from its exact root/direction")
            records.append((name, direction, hashlib.sha256(canonical(call).encode()).hexdigest()))
    _require(seen == set(expected) | {MODULE + "EvalBindingsJson"}, "missing concrete conversion owner")
    return tuple(sorted(records))


def _path(graph, crate, ty):
    _require(isinstance(ty, dict) and set(ty) == {"resolved_path"}, "exact nominal binding adapter required")
    path = ty["resolved_path"]
    record = graph.documents[crate].get("paths", {}).get(str(path.get("id")), {})
    name = "::".join(record.get("path", []))
    _require(name, "unresolved binding adapter identity")
    args = path.get("args")
    if args is None:
        return name, []
    _require(set(args) == {"angle_bracketed"} and not args["angle_bracketed"].get("constraints"),
             "unsupported binding adapter arguments")
    values = args["angle_bracketed"]["args"]
    _require(all(set(arg) in ({"type"}, {"lifetime"}) for arg in values), "open binding adapter argument")
    return name, [arg["type"] for arg in values if "type" in arg]


def adapter_for_slot(graph, callable_name, direction, label, ty):
    """Select exact obligations by registered owner, direction and typed slot."""
    if callable_name not in OUTPUTS:
        return None
    if direction == "output":
        name, args = _path(graph, "chelis_python", ty)
        _require(name == "pyo3::err::PyResult" and len(args) == 1, "compiler JSON return must retain its typed result")
        name, args = _path(graph, "chelis_python", args[0])
        expected = MODULE + OUTPUTS[callable_name][0]
        _require(name == expected and not args, "wrong compiler JSON output root")
        return name
    if callable_name == "chelis_python::eval_json" and label == "bindings_json":
        name, args = _path(graph, "chelis_python", ty)
        _require(name == MODULE + "EvalBindingsJson" and not args, "eval input must retain its exact tensor map")
        return name
    # These spec/11 inputs are source text, closed vocabulary, a name or a
    # filesystem path. The transport cannot authorize an added numeric/text
    # payload input or a changed type outside its exact adapter subtree.
    role = INPUT_ROLES[callable_name].get(label)
    _require(direction == "input" and role is not None, "unowned compiler JSON input slot")
    def text_input(value):
        return (isinstance(value, dict) and set(value) == {"borrowed_ref"}
                and value["borrowed_ref"].get("is_mutable") is False
                and value["borrowed_ref"].get("type") == {"primitive": "str"})
    if role == "text":
        valid = text_input(ty)
    else:
        name, args = _path(graph, "chelis_python", ty)
        if role == "gil":
            valid = name == "pyo3::marker::Python" and not args
        else:
            valid = name == "core::option::Option" and len(args) == 1
            if valid and role == "optional-name":
                name, nested = _path(graph, "chelis_python", args[0])
                valid = name == "alloc::string::String" and not nested
            elif valid:
                valid = text_input(args[0])
    _require(valid, "changed non-payload compiler JSON input role")
    return None


def require_parameter_slots(callable_name, inputs):
    labels = [label for label, _ in inputs]
    _require(len(labels) == len(set(labels)) and set(labels) == set(INPUT_ROLES[callable_name]),
             "compiler JSON input slots changed or duplicated")


def adapter_payload(graph, adapter):
    _require(adapter in {MODULE + item[0] for item in OUTPUTS.values()} | {MODULE + "EvalBindingsJson"},
             "unknown compiler JSON adapter")
    location = graph.locations.get(adapter)
    _require(location is not None, "missing defining compiler JSON adapter")
    item = graph._item(*location)
    body = item.get("inner", {}).get("struct", {})
    _require(not body.get("generics", {}).get("params"), "compiler JSON adapter cannot be generic")
    fields = body.get("kind", {}).get("plain", {})
    ids = fields.get("fields", [])
    _require(not fields.get("has_stripped_fields") and len(ids) == 1, "compiler JSON adapter changed fields")
    field = graph._item(location[0], ids[0])
    _require(field.get("name") == "value" and field.get("visibility") == "default"
             and set(field.get("inner", {})) == {"struct_field"}, "compiler JSON must retain one private typed value")
    _require(item.get("span", {}).get("filename") == SOURCE, "compiler JSON adapter moved from its compiled owner")
    ty = field["inner"]["struct_field"]
    projected = graph.discover(location[0], ty)
    root = projected.roots[0][1]
    if adapter == MODULE + "EvalBindingsJson":
        expected = ("container", "alloc::collections::btree::map::BTreeMap", (
            ("atomic", "alloc::string::String"), ("reference", API + "TensorValue", ())))
    else:
        result = next(result for name, result in OUTPUTS.values() if MODULE + name == adapter)
        expected = ("reference", API + result, ())
    _require(root == expected, "adapter field differs from its exact typed codec root")
    return projected


def _native_registration(root, target):
    from capacity_census_wire_calls import record_process

    source = root / "crates/chelis-python/examples/compiler_json_probe.rs"
    environment = {**os.environ, "CARGO_TARGET_DIR": str(target), "CARGO_BUILD_JOBS": "1",
                   "PYO3_PYTHON": sys.executable, "VIRTUAL_ENV": sys.prefix}
    command = ["cargo", "build", "--locked", "-p", "chelis-python", "--example", "compiler_json_probe", "--message-format=json"]
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True, text=True)
    build_log = record_process(target / "compiler-json-processes/registration-build", command, result)
    _require(result.returncode == 0, "compiler JSON probe build failed: " + result.stderr)
    candidates = [Path(item["executable"]).resolve() for line in result.stdout.splitlines()
                  if (item := json.loads(line)).get("reason") == "compiler-artifact"
                  and item.get("target", {}).get("name") == "compiler_json_probe"
                  and item["target"].get("kind") == ["example"]
                  and item["target"].get("src_path") == str(source)
                  and item.get("executable")]
    _require(len(candidates) == 1 and candidates[0].is_relative_to(target), "missing exact compiled probe artifact")
    binary = candidates[0]
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    result = subprocess.run([str(binary)], cwd=root, env=environment, capture_output=True, text=True)
    execution_log = record_process(target / "compiler-json-processes/registration-execution", [str(binary)], result)
    _require(result.returncode == 0 and hashlib.sha256(binary.read_bytes()).hexdigest() == before,
             "native registration probe failed or changed: " + result.stderr)
    return json.loads(result.stdout), (str(binary), before), (build_log, execution_log)


@dataclass(frozen=True, init=False, slots=True)
class VerifiedCompilerJsonBindings:
    root: Path
    source_sha256: str
    wire: object
    graph: object
    ownership: tuple
    registration: dict
    execution: object
    compiler_evidence: dict
    binary: tuple
    native_binary: tuple
    driver: str
    construction: tuple
    controls: object
    mir_controls: tuple
    processes: tuple
    control_packet: tuple

    def __new__(cls, *args, **kwargs):
        raise TypeError("CompilerJson authority requires actual current execution")

    def validate(self):
        from capacity_census_compiler_json_construction import construction_sources
        from capacity_census_compiler_json_controls import mir_control_sources
        from capacity_census_wire_calls import _current_driver_receipt

        _require(source_identity(self.root) == self.source_sha256, "stale compiler JSON source")
        self.wire.validate()
        for path, digest in (self.binary, self.native_binary, self.control_packet):
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, "stale native compiler JSON artifact")
        _require(_current_driver_receipt(Path(self.driver)) == self.compiler_evidence["driver_build"],
                 "compiled conversion driver changed")
        _require(tuple(sorted(self.execution.selected)) == tuple(sorted(NATIVE_CASES))
                 and self.execution.selected == self.execution.executed, "incomplete native compiler JSON execution")
        _require(validate_conversion_calls(self.compiler_evidence["evidence"]) == self.ownership,
                 "changed compiled conversion ownership")
        _require(tuple(sorted(self.controls.selected)) == tuple(sorted(PYTHON_CASES))
                 and self.controls.selected == self.controls.executed, "incomplete compiler JSON verifier controls")
        expected = [(name, hashlib.sha256(source.encode()).hexdigest(), error)
                    for name, source, error in construction_sources(self.root)]
        _require([(row["name"], row["source_sha256"], row["expected_error"]) for row in self.construction] == expected,
                 "construction controls changed or missing")
        expected_mir = [(name, hashlib.sha256(source.encode()).hexdigest(), error)
                        for name, source, error in mir_control_sources(self.root)]
        _require([(row["name"], row["source_sha256"], row["expected_error"]) for row in self.mir_controls] == expected_mir,
                 "compiled conversion controls changed or missing")
        for entry in self.compiler_evidence["provenance"] + self.compiler_evidence["fixture_externs"]:
            _require(hashlib.sha256(Path(entry["artifact"]).read_bytes()).hexdigest() == entry["sha256"],
                     "changed defining conversion codec artifact")
        processes = (*self.processes, self.compiler_evidence["process"],
                     *(row["process"] for row in self.construction), *(row["process"] for row in self.mir_controls))
        for process in processes:
            for stream in ("stdout", "stderr"):
                entry = process[stream]
                _require(hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"],
                         "compiler JSON execution transcript changed or missing")

    def exposure(self, name, direction, label, ty, proof):
        adapter = adapter_for_slot(self.graph, name, direction, label, ty)
        if adapter is None:
            return None
        self.validate()
        expected = [row for row in self.registration["registrations"] if "chelis_python::" + row["python_name"] == name]
        _require(len(expected) == 1, "missing actual registered compiler JSON function")
        row = expected[0]
        _require(proof == ("registration", self.registration["source_sha256"],
                          "chelis_python::" + row["rust_name"], "function", row["python_name"], row["line"], row["column"]),
                 "compiled Python exposure differs from registered conversion owner")
        projected = adapter_payload(self.graph, adapter)
        classified = {item.leaf: item for item in self.wire.schema.classifications}
        _require(projected.numeric_leaves and all(leaf in classified for leaf in projected.numeric_leaves),
                 "compiler JSON root contains an unverified numeric leaf")
        return projected

    def execution_report(self):
        self.validate()
        return {"source_sha256": self.source_sha256, "wire_graph_identity": self.wire.graph_identity,
                "ownership": self.ownership, "registration": self.registration,
                "execution": asdict(self.execution), "compiler": self.compiler_evidence,
                "native_registration_binary": self.binary, "native_test_binary": self.native_binary,
                "construction": self.construction, "verifier_controls": asdict(self.controls),
                "mir_controls": self.mir_controls, "processes": self.processes,
                "control_packet": self.control_packet}


def verify_compiler_json_bindings(root: Path, target: Path):
    from capacity_census_wire_calls import build_driver, collect_library
    from capacity_census_wire_schema import SchemaWireGraph
    from capacity_census_wire_verifier import verify_wire_census
    from capacity_census_compiler_json_construction import compile_construction_controls
    from capacity_census_compiler_json_controls import verify_mir_controls

    root, target = root.resolve(), target.resolve()
    _require(target.is_relative_to(root / "target"), "compiler JSON target must belong to this worktree")
    before = source_identity(root)
    with _target_lease(root / "target/.compiler-json-sequence"):
        process_directory = target / "compiler-json-processes"
        controls = run_python_tests(root, PYTHON_CASES, log_prefix=process_directory / "verifier-controls")
        control_packet = (str(process_directory / "verifier-controls.execution.json"), controls.output_sha256)
        # Finish ordinary consumers before wire/cache evidence binds dependency
        # artifacts. Compiled invocation collection retains its own namespace.
        with _target_lease(target):
            execution = build_and_run_rust_test(root, target, "chelis-python", "compiler_json_payloads", NATIVE_CASES,
                                              log_prefix=process_directory / "native-tests")
            native_binary = (execution.command[0], hashlib.sha256(Path(execution.command[0]).read_bytes()).hexdigest())
            registration, binary, processes = _native_registration(root, target)
            processes += tuple(json.loads((process_directory / (name + ".json")).read_text())
                               for name in ("native-tests-build", "native-tests", "verifier-controls"))
        wire = verify_wire_census(root, target)
        documents = [json.loads(wire.schema.canonical.document), json.loads(wire.schema.document),
                     *(json.loads(item) for item in wire.schema.imported_documents)]
        graph = SchemaWireGraph(documents, wire.schema)
        graph.publication_graph()
        driver = build_driver(root, target / "compiler-json-driver")
        evidence = collect_library(root, target, driver, scope="compiler-json")
        ownership = validate_conversion_calls(evidence["evidence"])
        construction = compile_construction_controls(root, target, evidence)
        mir_controls = verify_mir_controls(root, target, driver, evidence)
        _require(source_identity(root) == before, "source changed during compiler JSON verification")
        witness = object.__new__(VerifiedCompilerJsonBindings)
        for key, value in dict(root=root, source_sha256=before, wire=wire, graph=graph, ownership=ownership,
                               registration=registration, execution=execution, compiler_evidence=evidence,
                               binary=binary, native_binary=native_binary, driver=str(driver),
                               construction=construction, controls=controls, mir_controls=mir_controls,
                               processes=processes, control_packet=control_packet).items():
            object.__setattr__(witness, key, value)
        witness.validate()
        return witness
