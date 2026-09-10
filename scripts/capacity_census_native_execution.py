"""Current execution of the fixed Spec11/OP45 native boundary corpus.

Device cases use simulated SDK/foreign ABI fixtures. This bounded witness
does not establish GPU execution, owner-flow authority or final binding closure.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from capacity_census_graph import GraphError
from capacity_census_wire_adapters import _target_lease, canonical, source_identity
from capacity_census_wire_calls import record_process
from capacity_census_wire_runner import (
    _managed_python_environment, build_and_run_rust_test, select_test_binary,
    validate_libtest_execution,
)


@dataclass(frozen=True)
class Group:
    name: str
    kind: str
    selected: tuple[str, ...]
    instances: tuple[int, ...]
    fixture_kind: str
    children: tuple[str, ...] = ()


GROUPS = (
    Group(
        'native_tensor_boundary', 'test',
        (
            'rank_zero_is_one_element_and_rejects_rank_one_input',
            'rank_one_roundtrips_and_rejects_rank_two_input',
            'rank_eight_roundtrips_and_rejects_rank_nine_input',
            'rank_nine_roundtrips_and_rejects_rank_ten_input',
            'zero_extent_preserves_empty_shape_and_rejects_nonempty_input',
            'empty_large_extent_shape_is_exact_without_payload_allocation',
            'f64_copy_preserves_stored_bits_and_rejects_f32_input',
            'dlpack_cpu_accepts_none_stream_and_rejects_integer_streams',
            'dlpack_keywords_are_keyword_only_and_validate_version_and_device_shapes',
            'dlpack_copy_false_shares_storage_and_copy_true_never_returns_legacy_capsule',
            'dlpack_consumer_keeps_storage_alive_after_tensor_and_model_drop',
            'callable_v2_writer_and_loader_reject_v1_before_opening_a_library',
        ),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        'generated-c-current-runtime',
        (),
    ),
    Group(
        'native_output_validation', 'test',
        (
            'actual_single_output_is_validated_before_native_tensor_construction',
            'actual_named_outputs_preserve_dictionary_behavior_and_release_every_owner',
            'invalid_actual_descriptors_reject_and_release_all_returned_handles',
            'capsule_consumption_and_unconsumed_deletion_each_keep_exactly_one_storage_owner',
            'returned_alias_retains_actual_input_owner_for_entire_consumer_lifetime',
            'consumed_capsule_can_finalize_on_a_foreign_thread_without_the_python_gil',
            'versioned_capsule_preserves_descriptor_flags_and_consumption_lifetime',
        ),
        (1, 1, 1, 1, 1, 1, 1),
        'host-foreign-abi',
        (),
    ),
    Group(
        'native_device_validation', 'test',
        (
            'dynamic_device_owners_preserve_rank_device_and_input_lifetime',
            'invalid_storage_offset_capacity_and_context_never_reach_import_or_entry',
            'malformed_dynamic_outputs_release_every_returned_owner',
            'empty_device_view_checks_storage_bounds_without_requiring_a_payload_pointer',
            'device_dlpack_export_synchronizes_and_rejects_invalid_requests_without_a_capsule',
            'device_owner_duplicate_outputs_reject_without_double_finalization',
            'device_owner_returned_input_borrow_rejects_without_double_finalization',
            'device_owner_duplicate_bad_descriptor_cleanup_releases_once',
            'device_owner_distinct_named_outputs_keep_independent_lifetimes',
        ),
        (4, 7, 1, 1, 1, 1, 1, 1, 1),
        'simulated-device-foreign-abi',
        ('device_owner_duplicate_outputs_reject_without_double_finalization', 'device_owner_returned_input_borrow_rejects_without_double_finalization', 'device_owner_duplicate_bad_descriptor_cleanup_releases_once'),
    ),
    Group(
        'native_manifest_dimensions', 'test',
        (
            'repeated_input_names_bind_once_across_inputs_and_returned_outputs',
            'returned_named_extent_must_equal_the_admitted_input_binding',
            'repeated_axes_bind_zero_as_an_extent_and_reject_unequal_values',
            'wildcard_axes_are_independent_with_only_their_own_literal_constraints',
            'explicit_named_constraints_are_consistent_before_execution',
            'malformed_unspecified_and_unbound_named_output_claims_never_execute',
        ),
        (1, 1, 1, 3, 2, 3),
        'host-foreign-abi',
        (),
    ),
    Group(
        'native_python_environment', 'test',
        (
            'selected_python_packages_load_with_clean_ambient_environment',
            'missing_selected_interpreter_is_a_prerequisite_error_without_fallback',
        ),
        (0, 0),
        'interpreter-prerequisite',
        ('selected_python_packages_load_with_clean_ambient_environment',),
    ),
    Group(
        'chelis_python', 'lib',
        (
            'native_dlpack_owner_tests::mismatched_dlpack_request_rejects_and_releases_only_its_retained_owner',
        ),
        (1,),
        'generated-c-current-runtime',
        (),
    ),
)


def _require(condition, message):
    if not condition:
        raise GraphError(message)


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate native execution field")
        result[key] = value
    return result


def _json_bytes(content):
    try:
        return json.loads(content, object_pairs_hook=_unique_fields)
    except (ValueError, TypeError) as error:
        raise GraphError("malformed native execution JSON") from error


def _read_json(path):
    try:
        return _json_bytes(Path(path).read_bytes())
    except OSError as error:
        raise GraphError("missing native execution JSON: " + str(path)) from error


def _digest(path):
    try:
        _require(Path(path).is_file() and not Path(path).is_symlink(), "missing or symlinked native artifact")
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except (OSError, TypeError) as error:
        raise GraphError("unreadable native execution artifact") from error


def _inside(path, root):
    try:
        path, root = Path(path), Path(root).resolve()
    except (TypeError, ValueError, OSError) as error:
        raise GraphError("invalid native artifact path") from error
    _require(path.is_absolute() and path.resolve().is_relative_to(root), "native artifact escaped its owned directory")
    for component in (path, *path.parents):
        _require(not component.is_symlink(), "native artifact path contains a symlink")
        if component == root:
            break
    return path


def _files(root):
    records = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "native capture contains a symlink")
        if path.is_dir():
            continue
        _require(path.is_file(), "native capture contains a nonregular artifact")
        records.append({"path": str(path.relative_to(root)), "sha256": _digest(path)})
    return records


def _source_packet(root):
    commands = (("git", "rev-parse", "HEAD"), ("git", "ls-files", "-z"))
    head, paths = (subprocess.check_output(command, cwd=root) for command in commands)
    records = []
    # Bind all tracked inputs, including actual C/HIP headers, fixture sources,
    # private helpers and package/compiler data outside the Rust-only closure.
    for raw in sorted(paths.split(b"\0")):
        if raw:
            relative = raw.decode()
            path = root / relative
            link = {}
            if path.is_symlink():
                target = os.readlink(path)
                kind, content = "symlink", os.fsencode(target)
                link["target"] = target
                if path.is_file():
                    link["referent_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                _require(path.is_file(), "missing or unsupported tracked native source input: " + relative)
                kind, content = "file", path.read_bytes()
            records.append({"path": relative, "kind": kind, "sha256": hashlib.sha256(content).hexdigest(), **link})
    return {"head": head.decode().strip(), "common_source_sha256": source_identity(root), "files": records}


def _runtime_artifact(root, target, records):
    source = root / "crates/chelis-runtime/src/lib.rs"
    _require(isinstance(records, list), "malformed Cargo runtime artifact stream")
    rows = [row for row in records if isinstance(row, dict)
            and row.get("reason") == "compiler-artifact"
            and isinstance(row.get("target"), dict) and isinstance(row.get("profile"), dict)
            and row.get("target", {}).get("name") == "chelis_runtime"
            and row["target"].get("src_path") == str(source.resolve())
            and row["target"].get("kind") == ["staticlib", "rlib"]
            and row["target"].get("crate_types") == ["staticlib", "rlib"]
            and row.get("profile", {}).get("test") is False]
    _require(len(rows) == 1, "Cargo did not emit one exact current runtime artifact")
    names = rows[0].get("filenames")
    _require(isinstance(names, list) and all(isinstance(name, str) for name in names),
             "malformed Cargo runtime filenames")
    archives = [Path(name) for name in names if name.endswith(".a")]
    _require(len(archives) == 1, "Cargo runtime staticlib is missing or ambiguous")
    archive = _inside(archives[0], target)
    _digest(archive)
    return archive


def _worker_environment(runtime, directory, group):
    environment = _managed_python_environment()
    for name in ("CHELIS_RUNTIME_DIR", "CHELIS_CC", "CHELIS_TEST_CC", "CHELIS_HIPCC",
                 "CHELIS_NATIVE_EXECUTION_CAPTURE", "CHELIS_NATIVE_EXECUTION_SUITE",
                 "CHELIS_DEVICE_OWNER_TEST_WORKER", "CHELIS_NATIVE_ENV_TEST_WORKER"):
        environment.pop(name, None)
    environment.update(CHELIS_RUNTIME_DIR=str(runtime), CARGO_BUILD_JOBS="1",
                       CHELIS_NATIVE_EXECUTION_CAPTURE=str(directory),
                       CHELIS_NATIVE_EXECUTION_SUITE=group.name,
                       CARGO_HUSKY_DONT_INSTALL_HOOKS="1")
    return environment


def _validate_instance(directory, group, test, packet, runtime_digest):
    _require(isinstance(packet, dict) and set(packet) == {
        "schema", "suite", "test", "completed", "library", "files",
    }, "malformed native fixture completion")
    _require(type(packet["schema"]) is int and packet["schema"] == 1
             and packet["suite"] == group.name and packet["test"] == test
             and packet["completed"] is True, "native fixture completion belongs to another or unfinished case")
    files = packet["files"]
    _require(isinstance(files, list) and files, "native fixture retained no artifacts")
    actual = [row for row in _files(directory) if row["path"] != "completion.json"]
    _require(files == actual, "native fixture artifacts are missing, duplicated or changed")
    model = _read_json(directory / "loaded-model.json")
    _require(isinstance(model, dict) and set(model) == {"path", "root", "snapshot", "library", "files"},
             "native capture lacks actual model path provenance")
    _require(all(isinstance(model[key], str) for key in ("path", "root", "snapshot", "library")),
             "malformed native model snapshot paths")
    _require(all(not Path(model[key]).is_absolute() and ".." not in Path(model[key]).parts
                 for key in ("snapshot", "library")), "native snapshot path is not relative")
    original, original_root = Path(model["path"]), Path(model["root"])
    _require(original.is_absolute() and original.parent == original_root,
             "native model path differs from its actual original directory")
    snapshot = _inside(directory / model["snapshot"], directory)
    _require(snapshot != directory and snapshot.is_dir(), "native model snapshot must be a retained subdirectory")
    library = _inside(directory / model["library"], snapshot)
    _require(str(library) == packet["library"] and library.name == original.name,
             "native captured library differs from the actual loaded model")
    _digest(library)
    entries = model["files"]
    _require(isinstance(entries, list) and entries, "native model snapshot has no original artifacts")
    captured = []
    for entry in entries:
        _require(isinstance(entry, dict) and set(entry) == {"original", "captured", "sha256"},
                 "malformed native snapshot provenance")
        _require(all(isinstance(value, str) for value in entry.values())
                 and not Path(entry["captured"]).is_absolute() and ".." not in Path(entry["captured"]).parts,
                 "malformed native snapshot artifact paths")
        destination = _inside(directory / entry["captured"], snapshot)
        origin = original_root / destination.relative_to(snapshot)
        _require(str(origin) == entry["original"] and _digest(destination) == entry["sha256"],
                 "native artifact snapshot changed or refers to a different original")
        captured.append(str(destination.relative_to(snapshot)))
    _require(sorted(captured) == [row["path"] for row in _files(snapshot)],
             "native snapshot contains missing, duplicate or extra original-file records")
    _require(str(library.relative_to(snapshot)) in captured, "actual loaded library was not captured")
    manifest = library.with_suffix(".json")
    _require(manifest.is_file(), "actual loaded library manifest was not retained")
    decoded = _read_json(manifest)
    _require(isinstance(decoded, dict) and type(decoded.get("abi_version")) is int
             and decoded["abi_version"] == 2, "native loaded manifest is not callable ABI 2")
    _require(any(path.endswith(".c") for path in captured), "native fixture source was not retained")
    if group.fixture_kind == "generated-c-current-runtime":
        _require(decoded.get("target") == "c" and any(path.endswith(".h") for path in captured),
                 "generated C source/header provenance is incomplete")
        _require(any(row["path"].endswith(".ch") for row in files), "authored compiled source was not retained")
        _require(isinstance(decoded.get("source_path"), str) and isinstance(decoded.get("source_hash"), str),
                 "generated manifest omitted authored source provenance")
        authored = _inside(Path(decoded["source_path"]), directory)
        _require(authored.suffix == ".ch" and _digest(authored) == decoded["source_hash"],
                 "generated manifest differs from its actual authored source")
        _require(_digest(snapshot / "libchelis_runtime.a") == runtime_digest,
                 "generated library staged a different runtime archive")
    else:
        expected = "hip" if group.fixture_kind == "simulated-device-foreign-abi" else "c"
        _require(decoded.get("target") == expected, "foreign fixture target changed")
        process = _validate_process(directory / "compiler.json", directory)
        _require(process["returncode"] == 0 and process["command"][0] == "cc",
                 "foreign fixture compiler failed or changed")
        command = process["command"]
        _require(command.count("-o") == 1 and command.index("-o") + 1 < len(command)
                 and command[command.index("-o") + 1] == str(original),
                 "foreign compile output differs from actual loaded model")
        _require(any(str(original_root / Path(path).name) in command for path in captured if path.endswith(".c")),
                 "foreign compiler did not receive the retained fixture source")
        _require(not any(name.endswith(".a") for name in command),
                 "foreign fixture unexpectedly links an archive")
    return {"directory": str(directory), "test": test, "fixture_kind": group.fixture_kind,
            "library": {"path": str(library), "sha256": _digest(library)},
            "original_library": str(original), "completion_sha256": _digest(directory / "completion.json")
            if (directory / "completion.json").is_file() else None}


def _validate_process(path, root):
    process = _read_json(path)
    _require(isinstance(process, dict) and isinstance(process.get("command"), list)
             and process["command"] and all(isinstance(value, str) for value in process["command"])
             and type(process.get("returncode")) is int, "malformed actual native process record")
    for stream in ("stdout", "stderr"):
        row = process.get(stream)
        _require(isinstance(row, dict) and set(row) == {"path", "sha256"}, "missing native process transcript")
        artifact = _inside(Path(row["path"]), root)
        _require(_digest(artifact) == row["sha256"], "native process transcript changed")
    return process


def _validate_captures(directory, group, runtime_digest, binary):
    fixtures = directory / "fixtures"
    expected = {name: count for name, count in zip(group.selected, group.instances, strict=True) if count}
    actual = {path.name for path in fixtures.iterdir()} if fixtures.exists() else set()
    _require(actual == set(expected), "native fixture completion omitted or added a selected case")
    results = []
    for test, count in expected.items():
        cases = sorted((fixtures / test).iterdir())
        _require(len(cases) == count, "native fixture instance count differs")
        for instance in cases:
            _inside(instance, fixtures)
            _require(instance.is_dir(), "native fixture instance is not a directory")
            results.append(_validate_instance(instance, group, test,
                                             _read_json(instance / "completion.json"), runtime_digest))
    children = directory / "children"
    actual_children = {path.name for path in children.iterdir()} if children.exists() else set()
    _require(actual_children == set(group.children), "native subprocess case association differs")
    for test in group.children:
        process = _validate_process(children / test / "worker.json", directory)
        _require(process["returncode"] == 0 and process["command"] == [
            str(binary), "--exact", test, "--nocapture",
        ], "native subprocess did not run its actual associated case")
    return results


def _collect_worker(root, target, directory, name):
    groups = [group for group in GROUPS if group.name == name]
    _require(len(groups) == 1, "unknown fixed native execution group")
    group = groups[0]
    _require(os.environ.get("CHELIS_NATIVE_EXECUTION_CAPTURE") == str(directory)
             and os.environ.get("CHELIS_NATIVE_EXECUTION_SUITE") == name,
             "native worker requires its framework-owned environment")
    result = build_and_run_rust_test(root, target, "chelis-python", group.name,
                                     group.selected, kind=group.kind, log_prefix=directory / "test")
    print(canonical(asdict(result)))


@dataclass(frozen=True, init=False, slots=True)
class CheckedNativeExecution:
    root: Path
    directory: Path
    packet: dict
    packet_sha256: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("native execution requires the actual fixed current test corpus")

    def validate(self):
        _require(hashlib.sha256(canonical(self.packet).encode()).hexdigest() == self.packet_sha256,
                 "native execution packet changed")
        _require(_source_packet(self.root) == self.packet["source"], "native execution source or head changed")
        _require(_read_json(self.directory / "report.json") == self.packet, "native execution report changed")
        _require([row for row in _files(self.directory) if row["path"] != "report.json"] == self.packet["evidence"],
                 "native execution evidence changed or disappeared")
        for binary in self.packet["binaries"]:
            _require(_digest(binary["path"]) == binary["sha256"], "native execution binary changed")
        interpreter = self.packet["interpreter"]
        _require(_digest(interpreter["path"]) == interpreter["sha256"], "native execution interpreter changed")
        runtime = self.packet["runtime"]
        _require(_digest(runtime["cargo_artifact"]) == _digest(runtime["isolated_archive"]) == runtime["sha256"],
                 "native current runtime artifact changed")
        expected = tuple(f"{group.name}::{name}" for group in GROUPS for name in sorted(group.selected))
        _require(tuple(self.packet["selected"]) == expected
                 and self.packet["executed"] == [{"id": name, "outcome": "passed"} for name in expected],
                 "native execution no longer covers the exact selected corpus")
        return self.packet


def collect_native_execution(root: Path, target: Path):
    """Build current runtime and execute the exact corpus; accept no evidence inputs."""
    root, target = root.resolve(), target.resolve()
    _require(target.is_relative_to(root / "target"), "native execution target must belong to this worktree")
    _require(Path(sys.prefix).resolve() == (root / ".venv").resolve(),
             "native execution requires this worktree's owned interpreter")
    source = _source_packet(root)
    environment = _managed_python_environment()
    environment.update(CARGO_TARGET_DIR=str(target), CARGO_BUILD_JOBS="1", CARGO_HUSKY_DONT_INSTALL_HOOKS="1")
    with _target_lease(target):
        runs = target / "native-execution"
        runs.mkdir(exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=runs))
        command = ["cargo", "build", "--locked", "-p", "chelis-runtime", "--lib", "--message-format=json"]
        result = subprocess.run(command, cwd=root, env=environment, capture_output=True, check=False)
        record_process(directory / "runtime-build", command, result)
        _require(result.returncode == 0, "current native runtime build failed: " + result.stderr.decode(errors="replace")[-6000:])
        records = [_json_bytes(line) for line in result.stdout.splitlines()]
        runtime = _runtime_artifact(root, target, records)
        runtime_digest = _digest(runtime)
        isolated = directory / "runtime"
        isolated.mkdir()
        archive = isolated / "libchelis_runtime.a"
        shutil.copyfile(runtime, archive)
        _require(_digest(archive) == runtime_digest, "native runtime isolation changed archive bytes")
        selected, executed, binaries, captures = [], [], [], []
        worker = root / "scripts/capacity_census_native_execution.py"
        _require(worker.resolve() == Path(__file__).resolve(), "native execution worker differs from current source")
        for group in GROUPS:
            group_dir = directory / "groups" / group.name
            group_dir.mkdir(parents=True)
            command = [sys.executable, str(worker), "--worker", str(root), str(target), str(group_dir), group.name]
            child_environment = _worker_environment(isolated, group_dir, group)
            child_environment["CARGO_TARGET_DIR"] = str(target)
            result = subprocess.run(command, cwd=root, env=child_environment, capture_output=True, check=False)
            record_process(group_dir / "worker", command, result)
            _require(result.returncode == 0, "native execution worker failed: " + result.stderr.decode(errors="replace")[-6000:])
            build = _validate_process(group_dir / "test-build.json", group_dir)
            _require(build["returncode"] == 0, "native test build failed")
            build_records = [_json_bytes(line) for line in Path(build["stdout"]["path"]).read_bytes().splitlines()]
            test_source = root / "crates/chelis-python" / ("src/lib.rs" if group.kind == "lib" else f"tests/{group.name}.rs")
            binary = select_test_binary(target, test_source, group.name, build_records, kind=group.kind)
            process = _validate_process(group_dir / "test.json", group_dir)
            expected_command = [str(binary), "-Zunstable-options", "--format=json", "--exact", "--test-threads=1", *sorted(group.selected)]
            _require(process["returncode"] == 0 and process["command"] == expected_command,
                     "native test process did not execute the fixed exact selection")
            output = Path(process["stdout"]["path"]).read_bytes()
            actual = validate_libtest_execution(group.selected, [_json_bytes(line) for line in output.splitlines()])
            worker_packet = _json_bytes(result.stdout)
            _require(worker_packet == dict(command=expected_command, selected=sorted(group.selected),
                                            executed=list(actual), output_sha256=hashlib.sha256(output).hexdigest()),
                     "native worker result differs from its actual lifecycle")
            names = [f"{group.name}::{name}" for name in actual]
            selected.extend(names)
            executed.extend({"id": name, "outcome": "passed"} for name in names)
            binaries.append({"path": str(binary), "sha256": _digest(binary)})
            captures.extend(_validate_captures(group_dir, group, runtime_digest, binary))
            _require(_digest(runtime) == _digest(archive) == runtime_digest, "runtime changed during native execution")
        _require(_source_packet(root) == source, "native execution source changed during collection")
        _require(len(selected) == 37 and len(captures) == 49, "native execution matrix is incomplete")
        interpreter = Path(sys.executable).resolve()
        packet = dict(schema=1, source=source, selected=selected, executed=executed, binaries=binaries,
                      interpreter=dict(path=str(interpreter), sha256=_digest(interpreter), prefix=sys.prefix, version=sys.version),
                      runtime=dict(cargo_artifact=str(runtime), isolated_archive=str(archive), sha256=runtime_digest),
                      captures=captures, evidence=_files(directory),
                      limits=dict(device="simulated SDK and foreign ABI; no GPU execution",
                                  generated_c_link="fixed current source path and actual staged archive; internal C child argv not separately observed"))
    witness = object.__new__(CheckedNativeExecution)
    with (directory / "report.json").open("x") as output:
        output.write(canonical(packet) + "\n")
    for key, value in dict(root=root, directory=directory, packet=packet,
                           packet_sha256=hashlib.sha256(canonical(packet).encode()).hexdigest()).items():
        object.__setattr__(witness, key, value)
    witness.validate()
    return witness


if __name__ == "__main__":
    try:
        _require(len(sys.argv) == 6 and sys.argv[1] == "--worker", "native execution is invoked by its fixed collection API")
        _collect_worker(Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve(),
                        Path(sys.argv[4]).resolve(), sys.argv[5])
    except (GraphError, OSError, ValueError, TypeError) as error:
        print("native execution failed: " + str(error), file=sys.stderr)
        raise SystemExit(1) from error
