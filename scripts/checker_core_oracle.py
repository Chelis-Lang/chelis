#!/usr/bin/env python3
"""Compare portable checker behavior with its fixed native source."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
import tomllib
import xml.etree.ElementTree as ET

import no_std_core_support as support
from no_std_core_support import ROOT, TARGET, OracleFailure

BASELINE = "7e6c0aa438c635889fd78c819ea7efffa86a9409"
REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
CRATE = "chelis-checker-core"
FIXTURES = ROOT / "scripts/fixtures/checker_core"
ARTIFACTS = ROOT / "target/agents/checker-core"
FORBIDDEN = (
    "chelis-types", "chelis-effects", "chelis-surf", "chelis-syntax-host",
    "chelis-compiler-api", "chelis-pipeline-core", "chelis-ir", "chelis-dag-core",
    "chelis-runtime", "chelis-prove", "chelis-stdlib", "chelis-cli", "chelis-reef",
    "chelis-backend-c", "chelis-backend-hip", "chelis-backend-metal",
    "stacker", "psm", "tokio", "rayon", "parking_lot", "getrandom", "pyo3",
)
HOST_TARGETS = (
    "env_clone_allocations", "let_chain_inference_cost", "tuple_nesting_cost",
    "substitution_trial_cost", "infer_library_context_depth_guard", "context",
    "contextual_tensor_literal", "linearity_with_context", "signature_inference",
    "metadata_ingress", "typed_metadata_api", "rt800_checked_program_construction",
    "rt800_annotation_owner_stamps", "rt800_binder_scope_isolation",
    "rt800_append_only_cycle_resolution", "rt800_session_location_conformance",
    "issue_2110_local_ascription_obligations", "issue_1537_ingress_pass_set_parity",
    "opaque_types", "stack_guard_coverage", "rejection_authority",
    "rt800_checker_session_source_contract", "type_core_host_boundary", "type_analysis_outcome",
    "expand_insert_dispatch_family", "builtin_atom_discovery", "issue_1125_annotation_reader_parity",
    "issue_1923_pipe_fold_placement", "issue_1125_e5b_linearity",
)
CACHE_TARGETS = (
    "disk_cache", "redteam_typecheck_cache", "cache_wire_compatibility",
    "cache_load_cancellation_2617", "hash_order_cache_bytes",
    "issue_2110_local_ascription_cache", "issue_1875_callable_label_caches",
    "issue_3744_handoff_from_verified_cache", "pipeline_contract",
    "stdlib_cache_key_runtime_identity", "bundled_runtime_cache_determinism_2616",
)
LANES = {
    "full-host": ("-p", "chelis-types", *sum((("--test", name) for name in HOST_TARGETS), ())),
    "checker-unit": ("-p", "chelis-types", "--lib", "--features", "generalize-sweep-oracle"),
    "cache": ("-p", "chelis-compiler-api", *sum((("--test", name) for name in CACHE_TARGETS), ())),
    "cache-unit": ("-p", "chelis-compiler-api", "--lib", "-E",
        "test(cache) | test(handoff) | test(cancel) | test(context::) | test(layered::artifact_outcome_tests::)"),
    "abi": ("-p", "chelis-abi", "--all-targets"),
}


def digest(path: Path) -> dict:
    body = path.read_bytes()
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def decode(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise OracleFailure(f"Duplicate JSON evidence field: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise OracleFailure(f"Nonfinite JSON evidence value: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def load(path: Path):
    return decode(path.read_text(encoding="utf-8"))


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def fixture_identity() -> dict:
    names = ("corpus.json", "schema.json", "host.rs", "diagnostics.rs", "abi.rs",
        "immutable_source.json", "historical_gaps.json")
    return {name: digest(FIXTURES / name) for name in names}


class Evidence:
    """Retain command output, consumer inputs, locks, costs, and identities."""
    def __init__(self, directory: Path, scope: str):
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise OracleFailure("Evidence destination must be empty.")
        self.directory = directory
        self.scope = scope
        self.receipts: list[dict] = []
        self.script_identity = digest(Path(__file__))
        self.original = support._execute
        if (getattr(self.original, "__module__", None) != support.__name__
                or getattr(self.original, "__name__", None) != "_execute"):
            raise OracleFailure("A mocked command runner cannot establish acceptance.")

    def save(self, name: str, value) -> None:
        (self.directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def execute(self, command, *, cwd=ROOT, env=None, live_stderr=False):
        if live_stderr:
            raise OracleFailure("Raw stderr must remain in the evidence.")
        index = len(self.receipts)
        executable = shutil.which(str(command[0]), path=(env or support._environment()).get("PATH"))
        if executable is None:
            raise OracleFailure(f"Command executable is absent: {command[0]}")
        tool = Path(executable).resolve()
        started = time.monotonic_ns()
        result = self.original(command, cwd=cwd, env=env, live_stderr=False)
        files = {}
        for stream in ("stdout", "stderr"):
            path = self.directory / f"command-{index:05d}.{stream}"
            path.write_text(getattr(result, stream) or "", encoding="utf-8")
            files[stream] = {"path": path.name, **digest(path)}
        consumer = Path(cwd).name.startswith("chelis-no-std-consumer-")
        if consumer:
            for name in ("Cargo.toml", "Cargo.lock", "src/main.rs", "src/lib.rs"):
                source = Path(cwd) / name
                if source.is_file():
                    path = self.directory / f"command-{index:05d}.{name.replace('/', '_')}"
                    path.write_bytes(source.read_bytes())
                    files[name] = {"path": path.name, **digest(path)}
        if tuple(command[:3]) == ("cargo", "nextest", "run") and "--config-file" in command:
            configuration = tomllib.loads(Path(command[command.index("--config-file") + 1]).read_text())
            profile = configuration["profile"][command[command.index("--profile") + 1]]
            report = Path(profile["junit"]["path"])
            if report.is_file():
                path = self.directory / f"command-{index:05d}.junit.xml"
                path.write_bytes(report.read_bytes())
                files["junit"] = {"path": path.name, **digest(path)}
        self.receipts.append({"identity": index, "command": list(map(str, command)),
            "cwd": str(Path(cwd).resolve()), "exit": result.returncode,
            "elapsed_ns": time.monotonic_ns() - started, "consumer": consumer, "files": files,
            "tool": {"path": str(tool), **digest(tool)}})
        return result

    def accept_negative(self, expected: list[str], diagnostic: str) -> None:
        receipt = self.receipts[-1]
        if receipt["exit"] == 0 or not expected or any(text not in diagnostic for text in expected):
            raise OracleFailure("Negative control lacks its intended compiler error.")
        receipt["expected_failure"] = {"fragments": expected, "diagnostic": diagnostic}

    def finish(self, status: str) -> None:
        self.save("receipts.json", {"status": status, "scope": self.scope,
            "platform": sys.platform, "machine": platform.machine(),
            "python": sys.executable, "python_version": platform.python_version(),
            "script": self.script_identity, "invocation": sys.argv,
            "command_count": len(self.receipts), "receipts": self.receipts})
        if status == "passed":
            validate_receipts(self.directory)
            validate_artifacts(self.directory, self.scope)
        self.save("artifact-index.json", {str(path.relative_to(self.directory)): digest(path)
            for path in sorted(self.directory.rglob("*"))
            if path.is_file() and path.name != "artifact-index.json"})


def validate_artifacts(directory: Path, scope: str) -> None:
    required = {"frozen-inputs.json", "baseline.json", "receipts.json"}
    if scope == "implementation":
        required.update(("provenance.json", "candidate-host.json", "candidate-diagnostics.json",
            "candidate-abi.json", "current-tests.json", "cost-observations.json",
            "cost-source-contracts.json", "target-graphs.json", "negative-graphs.json",
            "authority-controls.json", "prior-core-obligations.json", "ci-ownership.json",
            "compiled-artifacts.json", "candidate-fixtures.json", "test-migrations.json"))
    elif scope != "baseline-capture":
        raise OracleFailure("The acceptance evidence scope is unknown.")
    missing = sorted(name for name in required if not (directory / name).is_file())
    if missing:
        raise OracleFailure(f"Mandatory acceptance artifacts are absent: {missing}")
    previous = load(directory / "baseline.json")
    frozen = load(directory / "frozen-inputs.json")
    if (previous["revision"] != BASELINE or frozen["revision"] != BASELINE
            or previous["fixtures"] != frozen["fixtures"] or previous["fixtures"] != fixture_identity()
            or set(previous["tests"]) != set(LANES)):
        raise OracleFailure("Frozen source, fixture, or execution inventory evidence differs.")
    for kind in ("host", "diagnostics", "abi"):
        validate_snapshot(previous[kind], kind)
    cost_observations(previous["tests"])
    for lane in previous["tests"].values():
        if not lane["selected"] or lane["selected"] != lane["executed"]:
            raise OracleFailure("Baseline execution evidence is incomplete.")
    if scope == "implementation":
        current = load(directory / "current-tests.json")
        if set(current) != set(LANES) | {"portable-unit"}:
            raise OracleFailure("Candidate execution lane evidence differs.")
        cost_observations(current)
        if not load(directory / "compiled-artifacts.json"):
            raise OracleFailure("Compiled target artifact evidence is empty.")


def validate_receipts(directory: Path) -> list[dict]:
    try:
        document = load(directory / "receipts.json")
        required = {"status", "scope", "platform", "machine", "python", "python_version",
            "script", "invocation", "command_count", "receipts"}
        if set(document) != required or set(document["script"]) != {"bytes", "sha256"}:
            raise OracleFailure("Source, tool, or invocation provenance is incomplete.")
        receipts = document["receipts"]
        if (document["status"] != "passed" or not receipts
                or document["command_count"] != len(receipts)
                or [item["identity"] for item in receipts] != list(range(len(receipts)))):
            raise OracleFailure("Missing or duplicate command identity.")
        paths = set()
        for item in receipts:
            if (not item["command"] or type(item["exit"]) is not int
                    or type(item["elapsed_ns"]) is not int or item["elapsed_ns"] < 0):
                raise OracleFailure("Invalid command receipt.")
            tool = item["tool"]
            if not Path(tool["path"]).is_absolute() or digest(Path(tool["path"])) != {
                    key: tool[key] for key in ("bytes", "sha256")}:
                raise OracleFailure("Command tool identity is absent or differs.")
            if item["exit"] != 0:
                failure = item.get("expected_failure")
                missing_source = item["command"] == ["git", "cat-file", "-e", f"{BASELINE}^{{commit}}"]
                if not missing_source and (not failure or not failure["fragments"]
                        or any(text not in failure["diagnostic"] for text in failure["fragments"])):
                    raise OracleFailure("Failed command lacks intended negative evidence.")
            required = {"stdout", "stderr"}
            if item["exit"] == 0 and item["command"][:3] == ["cargo", "nextest", "run"]:
                required.add("junit")
            if item["consumer"]:
                required.update(("Cargo.toml", "Cargo.lock"))
                if not {"src/main.rs", "src/lib.rs"}.intersection(item["files"]):
                    raise OracleFailure("Independent consumer source is absent.")
            if not required <= set(item["files"]):
                raise OracleFailure("Raw output or consumer lock is absent.")
            for record in item["files"].values():
                path = directory / record["path"]
                if record["path"] in paths or path.parent != directory:
                    raise OracleFailure("Duplicate or foreign raw evidence path.")
                paths.add(record["path"])
                if digest(path) != {key: record[key] for key in ("bytes", "sha256")}:
                    raise OracleFailure("Raw evidence differs.")
            if "expected_failure" in item:
                if item["exit"] == 0 or not item["consumer"] or "src/lib.rs" not in item["files"]:
                    raise OracleFailure("A negative receipt lacks its actual Rust consumer.")
                source = (directory / item["files"]["src/lib.rs"]["path"]).read_text()
                stdout = (directory / item["files"]["stdout"]["path"]).read_text()
                failure = item["expected_failure"]
                diagnostic = negative_diagnostics(stdout, Path(item["cwd"]), source, failure["fragments"])
                if diagnostic != failure["diagnostic"]:
                    raise OracleFailure("A negative receipt differs from its raw compiler evidence.")
        return receipts
    except (OSError, KeyError, ValueError, TypeError) as error:
        raise OracleFailure(f"Invalid execution evidence: {error}") from error


def selected_tests(metadata: dict) -> set[str]:
    selected = support._selected_test_names(metadata)
    actual = [f"{suite['package-name']}::{suite['binary-name']}${name}"
        for suite in metadata["rust-suites"].values()
        for name, item in suite["testcases"].items()
        if item["filter-match"]["status"] == "matches"]
    if len(actual) != len(set(actual)):
        raise OracleFailure("Duplicate selected test identity.")
    return selected


def check_execution(stdout: str, selected: set[str]) -> set[str]:
    completed = Counter()
    started = Counter()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = decode(line)
            if record["type"] != "test":
                continue
            if record["event"] == "started":
                started[record["name"]] += 1
                continue
            if record["event"] == "ignored" and record["name"] not in selected:
                continue
            if record["event"] != "ok":
                raise OracleFailure(f"Mandatory test did not pass: {record['name']}")
            completed[record["name"]] += 1
        except (KeyError, ValueError, TypeError) as error:
            raise OracleFailure(f"Invalid test execution record: {error}") from error
    if (set(started) != selected or set(completed) != selected
            or any(value != 1 for value in (*started.values(), *completed.values()))):
        raise OracleFailure("Missing, duplicate, or substituted test identity.")
    return set(completed)


def junit_evidence(raw: str, metadata: dict) -> dict:
    root = ET.fromstring(raw)
    selected = selected_tests(metadata)
    suites = {identity: f"{suite['package-name']}::{suite['binary-name']}"
        for identity, suite in metadata["rust-suites"].items()}
    identities = Counter()
    observations = {}
    for suite in root.iter("testsuite"):
        binary = suites.get(suite.attrib["name"])
        if binary is None:
            raise OracleFailure("A JUnit suite lacks its selected binary identity.")
        for case in suite.findall("testcase"):
            identity = binary + "$" + case.attrib["name"]
            if any(case.find(name) is not None for name in (
                    "failure", "error", "skipped", "flakyFailure", "flakyError", "rerunFailure", "rerunError")):
                raise OracleFailure("A JUnit record is failed, skipped, or retried.")
            if "time" not in case.attrib:
                raise OracleFailure("A test timing observation is absent.")
            identities[identity] += 1
            observations[identity] = {"attributes": dict(case.attrib),
                "stdout": case.findtext("system-out", ""), "stderr": case.findtext("system-err", "")}
    if set(identities) != selected or any(count != 1 for count in identities.values()):
        raise OracleFailure("JUnit and selected test identities differ.")
    return {"raw_xml": raw, "observations": observations}


def tests(arguments: tuple[str, ...], workspace: Path) -> dict:
    environment = support._environment()
    environment["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"
    with tempfile.TemporaryDirectory(prefix="checker-nextest-") as temporary:
        directory = Path(temporary)
        report = directory / "results.xml"
        configuration = (workspace / ".config/nextest.toml").read_text()
        configuration += ('\n[profile.checker-core-oracle]\ninherits = "default"\n'
            '[profile.checker-core-oracle.junit]\n'
            f'path = {json.dumps(str(report.resolve()))}\n'
            'store-success-output = true\nstore-failure-output = true\n')
        config = directory / "nextest.toml"
        config.write_text(configuration)
        flags = ("--config-file", str(config), "--profile", "checker-core-oracle", "--ignore-default-filter")
        inventory = support.run(("cargo", "nextest", "list", "--locked", *arguments, *flags,
            "--run-ignored", "all", "--message-format", "json"), cwd=workspace, env=environment)
        metadata = decode(inventory.stdout)
        selected = selected_tests(metadata)
        result = support._execute(("cargo", "nextest", "run", "--locked", *arguments, *flags,
            "--retries", "0", "--no-fail-fast", "--message-format", "libtest-json-plus"),
            cwd=workspace, env=environment)
        if result.returncode:
            raise OracleFailure("Mandatory nextest command failed.")
        return {"inventory": metadata, "selected": sorted(selected),
            "executed": sorted(check_execution(result.stdout, selected)),
            "configuration": configuration, "junit": junit_evidence(report.read_text(), metadata)}


def dependency(crate: str, workspace: Path, *, alias=None, defaults=True) -> str:
    return (f'{alias or crate} = {{ package = {json.dumps(crate)}, '
        f'path = {json.dumps(str((workspace / "crates" / crate).resolve()))}, '
        f'default-features = {str(defaults).lower()} }}\n')


def checkout_baseline(destination: Path) -> Path:
    destination = destination.resolve()
    support.run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT.resolve()), str(destination)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=destination)
    if present.returncode:
        support.run(("git", "fetch", "--quiet", "--no-tags", "--depth=1", REPOSITORY, BASELINE), cwd=destination)
    support.run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=destination)
    if support.run(("git", "rev-parse", "HEAD"), cwd=destination).stdout.strip() != BASELINE:
        raise OracleFailure("Immutable source identity differs.")
    expected = load(FIXTURES / "immutable_source.json")
    if expected["revision"] != BASELINE:
        raise OracleFailure("Frozen source revision differs.")
    for name, identity in expected["files"].items():
        if digest(destination / name) != identity:
            raise OracleFailure(f"Immutable source file differs: {name}")
    if support.run(("git", "status", "--porcelain=v1", "--untracked-files=normal"), cwd=destination).stdout.strip():
        raise OracleFailure("The immutable source checkout is not clean.")
    return destination


def snapshot(workspace: Path, kind: str) -> dict:
    source = (FIXTURES / f"{kind}.rs").read_text()
    source += '\nconst CORPUS_JSON: &str = r####"' + (FIXTURES / "corpus.json").read_text() + '"####;\n'
    dependencies = 'bincode = "=1.3.3"\nserde_json = { version = "1", features = ["float_roundtrip"] }\n'
    owner = "chelis-types" if workspace != ROOT else CRATE
    # One canonical source path prevents duplicate native links.
    if kind == "host":
        dependencies += dependency(owner, workspace, alias="checker-data")
        crates = ("chelis-deep", "chelis-syntax-host", "chelis-surf", "chelis-macros", "chelis-effects")
        crate = "chelis-types"
        prefix = "CHECKER_HOST_SNAPSHOT "
    elif kind == "diagnostics":
        dependencies += dependency("chelis-effects-core", workspace, alias="effect-data")
        crates = ()
        crate = "chelis-compiler-api"
        prefix = "CHECKER_DIAGNOSTIC_SNAPSHOT "
    else:
        crates = ("chelis-vocab",)
        crate = "chelis-abi"
        prefix = "CHECKER_ABI_SNAPSHOT "
    dependencies += "".join(dependency(name, workspace) for name in crates)
    # The primary dependency also needs its host spelling in the baseline.
    if kind == "host" and workspace != ROOT:
        source = source.replace("use checker_data::InferResult;", "use chelis_types::InferResult;")
        dependencies = dependencies.replace(dependency(owner, workspace, alias="checker-data"), "")
    result = support.consumer((workspace / "crates" / crate).resolve(), source,
        default_features=True, dependencies=dependencies)
    records = [line.removeprefix(prefix) for line in result.stdout.splitlines() if line.startswith(prefix)]
    if len(records) != 1:
        raise OracleFailure(f"Missing or duplicate {kind} snapshot.")
    value = decode(records[0])
    validate_snapshot(value, kind)
    return value


def validate_snapshot(value: dict, kind="host") -> None:
    schema = load(FIXTURES / "schema.json")
    corpus = load(FIXTURES / "corpus.json")
    if kind == "abi":
        if set(value) != set(schema["abi_fields"]) or not value["descriptors"]:
            raise OracleFailure("Descriptor evidence is incomplete.")
        return
    if kind == "diagnostics":
        if (set(value) != {"conversions", "outcomes"} or len(value["conversions"]) != 16
                or set(value["outcomes"]) != {item["id"] for item in corpus["cases"]}):
            raise OracleFailure("Compiler diagnostic inventory differs.")
        return
    def fields(record, name):
        if not isinstance(record, dict) or set(record) != set(schema[name]):
            raise OracleFailure(f"Semantic field inventory differs: {name}")
    def report(record):
        fields(record, "report_fields")
        for error in record["errors"]:
            fields(error, "error_fields")
    def checked(record):
        fields(record, "checked_fields")
        fields(record["payload"], "checked_payload_fields")
        if not record["bytes"] or not record["tree"]:
            raise OracleFailure("Complete checked bytes or owner tree is absent.")
        if not 0 <= record["typed_nodes"] <= record["total_nodes"]:
            raise OracleFailure("Invalid inference counters.")
    def outcome(record):
        if record.get("status") == "accepted":
            if set(record) != {"status", "checked", "effects"}:
                raise OracleFailure("Accepted result fields differ.")
            checked(record["checked"])
        elif record.get("status") == "rejected" and set(record) == {"status", "report"}:
            report(record["report"])
            if not record["report"]["errors"]:
                raise OracleFailure("Rejection lacks diagnostics.")
        else:
            raise OracleFailure("Checker result evidence differs.")
    fields(value, "snapshot_fields")
    if set(value["cases"]) != {item["id"] for item in corpus["cases"]}:
        raise OracleFailure("Frozen case inventory differs.")
    for record in value["cases"].values():
        if record.get("status") == "admission-rejected":
            if set(record) != {"status", "error"} or not record["error"]:
                raise OracleFailure("Admission rejection lacks evidence.")
            continue
        fields(record, "case_fields")
        report(record["inference"])
        report(record["ir_inference"])
        outcome(record["ir"])
        outcome(record["typed"])
    checked(value["library"])
    checked(value["layer"])
    for name in ("context", "layered_context"):
        fields(value[name], "context_fields")
        if not value[name]["bytes"] or not value[name]["payload"] or not value[name]["matches_program"]:
            raise OracleFailure("Library context evidence is absent.")
    if set(value["extensions"]) != {item["id"] for item in corpus["extensions"]}:
        raise OracleFailure("Frozen extension inventory differs.")
    for record in value["extensions"].values():
        fields(record, "extension_fields")
        for name in ("contextual", "restored", "layered", "restored_layered", "monolithic"):
            outcome(record[name])
        if record["composition"] is not None:
            checked(record["composition"])
        if record["foreign_composition"]:
            raise OracleFailure("Foreign checked composition was accepted.")
    if len(value["isolation"]) != 8:
        raise OracleFailure("Concurrent context evidence is absent.")
    for record in value["isolation"]:
        outcome(record)


def compare_snapshots(previous: dict, current: dict, kind="host") -> None:
    validate_snapshot(previous, kind)
    validate_snapshot(current, kind)
    if canonical(previous) != canonical(current):
        raise OracleFailure(f"Complete {kind} semantic fields or bytes differ.")


def cost_observations(results: dict) -> dict:
    required = {
        "env_clone_allocations$snapshots_bound_allocations_and_preserve_lexical_isolation_and_serialization":
            "512-signature snapshot allocated",
        "let_chain_inference_cost$let_chain_inference_holds_near_linear_memory":
            "peak inference bytes at k = 256, 512, 1024:",
        "tuple_nesting_cost$checking_a_literal_tuple_chain_allocates_linearly":
            "literal tuple chain: check bytes at depth",
        "tuple_nesting_cost$checking_a_variable_tuple_chain_allocates_linearly":
            "variable tuple chain: check bytes at depth",
        "substitution_trial_cost$checking_annotated_results_allocates_linearly":
            "annotated results: check bytes at n = 128, 256, 512:",
        "substitution_trial_cost$checking_binder_suspended_calls_allocates_linearly":
            "binder-suspended calls: check bytes at n = 128, 256, 512:",
    }
    observations = results["full-host"]["junit"]["observations"]
    records = {}
    for suffix, marker in required.items():
        identity = "chelis-types::" + suffix
        if identity not in observations:
            raise OracleFailure(f"A full-host cost observation is absent: {identity}")
        record = observations[identity]
        if marker not in record["stdout"] + record["stderr"]:
            raise OracleFailure(f"A successful cost test lacks its measured output: {identity}")
        records[identity] = record
    return records


def capture_baseline(evidence: Evidence) -> dict:
    fixtures = fixture_identity()
    evidence.save("frozen-inputs.json", {"revision": BASELINE, "fixtures": fixtures,
        "schema": load(FIXTURES / "schema.json"), "corpus": load(FIXTURES / "corpus.json")})
    with tempfile.TemporaryDirectory(prefix="checker-base-") as temporary:
        workspace = checkout_baseline(Path(temporary) / "source")
        previous = {"revision": BASELINE, "fixtures": fixtures,
            "host": snapshot(workspace, "host"), "diagnostics": snapshot(workspace, "diagnostics"),
            "abi": snapshot(workspace, "abi"),
            "timing": load(workspace / "scripts/test_timing_baseline.json"),
            "tests": {name: tests(arguments, workspace) for name, arguments in LANES.items()},
            "historical_gaps": load(FIXTURES / "historical_gaps.json")}
        previous["cost_observations"] = cost_observations(previous["tests"])
        evidence.save("baseline.json", previous)
        if (support.run(("git", "status", "--porcelain=v1", "--untracked-files=normal"), cwd=workspace).stdout.strip()
                or support.run(("git", "rev-parse", "HEAD"), cwd=workspace).stdout.strip() != BASELINE):
            raise OracleFailure("The immutable source changed during capture.")
    if fixture_identity() != fixtures:
        raise OracleFailure("Frozen corpus or schema changed during capture.")
    return previous


def negative_diagnostics(stdout: str, directory: Path, source: str, expected: list[str]) -> str:
    diagnostic = support._fixture_diagnostics(stdout, directory, source)
    fixture = (directory / "src/lib.rs").resolve()
    allowed = {text for text in expected if len(text) == 5 and text[0] == "E" and text[1:].isdigit()}
    for line in stdout.splitlines():
        record = decode(line)
        if record.get("reason") != "compiler-message" or record["message"]["level"] != "error":
            continue
        if Path(record["target"]["src_path"]).resolve() != fixture:
            raise OracleFailure("A negative consumer failed outside its intended source.")
        code = record["message"].get("code")
        private_field = "private" in expected and "private" in record["message"]["message"]
        if code and code["code"] not in allowed and not (code["code"] == "E0451" and private_field):
            raise OracleFailure(f"A negative consumer failed with an unrelated Rust error: {code['code']}")
        if not code and not private_field:
            raise OracleFailure("A negative consumer lacks its intended private-field boundary.")
    if not expected or any(fragment not in diagnostic for fragment in expected):
        raise OracleFailure("A negative consumer lacks its intended compiler error.")
    return diagnostic


def rejected_consumer(evidence: Evidence, source: str, expected: list[str], *,
        features=(), dependencies="") -> None:
    with support._package(CRATE, source, executable=False, features=features,
            default_features=False, dependencies=dependencies) as directory:
        result = support._execute(("cargo", "check", "--lib", "--message-format", "json"), cwd=directory)
        if result.returncode == 0:
            raise OracleFailure("An authority misuse consumer compiled.")
        diagnostic = negative_diagnostics(result.stdout, directory, source, expected)
        evidence.accept_negative(expected, diagnostic)


def authority_controls(evidence: Evidence) -> set[str]:
    controls = load(FIXTURES / "authority.json")
    required = {"witness", "witness-vector", "sink-fields", "checkpoint", "checkpoint-index", "completion-fields",
        "completion-default", "completion-deserialize", "checked-fields", "facts-adoption",
        "completion-adoption", "facts-completion", "context-admission"}
    if set(controls) != required:
        raise OracleFailure("The authority consumer inventory differs.")
    records = {}
    for name, control in controls.items():
        dependencies = 'bincode = "=1.3.3"\n'
        if control.get("host"):
            dependencies += dependency("chelis-types", ROOT)
        rejected_consumer(evidence, control["source"], control["expected"],
            features=control.get("features", ()), dependencies=dependencies)
        records["authority::" + name] = {"receipt": evidence.receipts[-1]["identity"],
            "source_sha256": hashlib.sha256(control["source"].encode()).hexdigest(),
            "expected": control["expected"]}
    rejected_consumer(evidence, (FIXTURES / "pipeline_forgery.rs").read_text(),
        ["E0599", "checked", "SemanticRejection", "E0308", "NamedRoots", "ForwardNodeIndex",
         "CheckedLibrary", "CheckedProgram", "E0615", "dag", "library_proof_id"],
        dependencies=dependency("chelis-compiler-api", ROOT))
    records["authority::pipeline-forgery"] = {"receipt": evidence.receipts[-1]["identity"],
        "source": digest(FIXTURES / "pipeline_forgery.rs")}
    evidence.save("authority-controls.json", records)
    return set(records)


def portable_consumers(evidence: Evidence) -> None:
    manifest = tomllib.loads((ROOT / "crates" / CRATE / "Cargo.toml").read_text())
    features = manifest.get("features", {})
    if features.get("default", []) or "std" in features or "serde" not in features:
        raise OracleFailure("Checker features violate the portable contract.")
    source = (FIXTURES / "core.rs").read_text()
    dependencies = "".join(dependency(name, ROOT, defaults=False) for name in (
        "chelis-deep", "chelis-type-core", "chelis-stack-services", "chelis-abi", "chelis-vocab"))
    graphs = {}
    for name, defaults, selected in (
            ("default", True, ()), ("no-default", False, ()), ("serde", False, ("serde",))):
        graphs[name] = sorted(support.portable(CRATE, "#![no_std]\n" + source,
            default_features=defaults, features=selected, forbidden=FORBIDDEN, dependencies=dependencies))
    abi = '''#![no_std]
pub fn descriptor() {
    let value = chelis_abi::metadata::ShapeMetadata::contiguous(&[2, 3], chelis_vocab::RuntimeDType::F32).unwrap();
    assert_eq!(value.elements().get(), 6);
    assert!(chelis_abi::metadata::ShapeMetadata::contiguous(&[0, -1], chelis_vocab::RuntimeDType::F32).is_err());
}
'''
    graphs["abi"] = sorted(support.portable("chelis-abi", abi, forbidden=FORBIDDEN,
        dependencies=dependency("chelis-vocab", ROOT, defaults=False)))
    evidence.save("target-graphs.json", graphs)
    independent = support.consumer(CRATE, source + "\n" + (FIXTURES / "host_core.rs").read_text(),
        dependencies=dependencies)
    if independent.stdout.splitlines().count("CHECKER_CORE_CONSUMER accepted") != 1:
        raise OracleFailure("Independent checker execution evidence is absent or duplicate.")
    with support._package(CRATE, "#![no_std]\n#![no_main]\n" + source
            + "\n" + (FIXTURES / "target.rs").read_text(), executable=True,
            features=(), default_features=False, dependencies=dependencies) as directory:
        manifest_path = directory / "Cargo.toml"
        manifest_path.write_text(manifest_path.read_text() + '\n[profile.dev]\npanic = "abort"\n')
        support.run(("cargo", "rustc", "--bin", "chelis-no-std-consumer",
            "--target", TARGET, "--", "--emit=obj"), cwd=directory)
    serde_source = source + '''
fn main() {
    let facts = exercise();
    let bytes = bincode::serialize(&facts).unwrap();
    let decoded: chelis_checker_core::CheckFacts = bincode::deserialize(&bytes).unwrap();
    assert_eq!(bytes, bincode::serialize(&decoded).unwrap());
    let stack = CallerStack;
    let resources = Resources::new(&stack);
    let library = parse("(def {} library_id (fn {} (params {} x) (var {} x)))");
    let context = chelis_checker_core::build_type_env_from_library(&library, &resources).unwrap();
    let context_bytes = bincode::serialize(&context).unwrap();
    let restored: chelis_checker_core::ContextSnapshot = bincode::deserialize(&context_bytes).unwrap();
    assert_eq!(context_bytes, bincode::serialize(&restored).unwrap());
    let extension = parse("(def {} use_id (app {} (var {} library_id) (lit {type: (t-prim {} i32)} 1)))");
    let direct = chelis_checker_core::check_ir_with_context(&context, &extension, &resources).unwrap();
    let continued = chelis_checker_core::check_ir_with_context(&restored, &extension, &resources).unwrap();
    assert_eq!(bincode::serialize(direct.facts()).unwrap(), bincode::serialize(continued.facts()).unwrap());
}
'''
    support.consumer(CRATE, serde_source, features=("serde",),
        dependencies=dependencies + 'bincode = "=1.3.3"\n')
    rejected_consumer(evidence,
        "pub fn disabled(value: &chelis_checker_core::CheckFacts) { let _ = bincode::serialize(value); }\n",
        ["E0277", "Serialize", "CheckFacts"], dependencies='bincode = "=1.3.3"\n')


def forbidden_graph_controls(evidence: Evidence) -> None:
    controls = (
        ("hosted-inference", dependency("chelis-types", ROOT), "forbidden target dependency", "chelis-types"),
        ("native-stack", dependency("chelis-syntax-host", ROOT), "forbidden target dependency", "chelis-syntax-host"),
        ("runtime", dependency("chelis-runtime", ROOT), "forbidden target dependency", "chelis-runtime"),
        ("backend", dependency("chelis-backend-c", ROOT), "forbidden target dependency", "chelis-backend-c"),
        ("dependency-std", 'serde = { version = "1", default-features = false, features = ["std"] }\n',
         "target std feature", "serde"),
    )
    outcomes = {}
    for name, dependencies, expected, owner in controls:
        with support._package(CRATE, "#![no_std]\n", executable=False, features=(),
                default_features=False, dependencies=dependencies) as directory:
            result = support.run(("cargo", "metadata", "--format-version", "1",
                "--filter-platform", TARGET), cwd=directory)
            graph = decode(result.stdout)
            try:
                support._normal_packages(graph, FORBIDDEN)
            except OracleFailure as error:
                if expected not in str(error) or owner not in str(error):
                    raise OracleFailure(f"The {name} graph failed at a different boundary: {error}") from error
                outcomes[name] = {"graph": graph, "boundary": str(error)}
            else:
                raise OracleFailure(f"The {name} negative graph did not reject.")
    evidence.save("negative-graphs.json", outcomes)


def reconcile_inventories(previous: dict, current: dict, migrations: dict, *,
        external: frozenset[str] | set[str] = frozenset()) -> dict[str, str]:
    def identities(lanes):
        result = []
        for lane in lanes.values():
            if lane["selected"] != lane["executed"]:
                raise OracleFailure("Selected and executed test identities differ.")
            result.extend(lane["executed"])
        if len(result) != len(set(result)):
            raise OracleFailure("A mandatory test executed in more than one checker lane.")
        return set(result)
    before = identities(previous)
    after = identities(current)
    renamed = migrations["renamed"]
    retired = migrations["retired"]
    new_tests = migrations.get("new_tests", [])
    if (len(new_tests) != len(set(new_tests)) or not set(new_tests) <= after
            or set(new_tests).intersection(before | set(renamed.values()))):
        raise OracleFailure("A new mandatory behavior identity is missing, duplicate, or substituted.")
    if set(renamed).intersection(retired) or (set(renamed) | set(retired)) - before:
        raise OracleFailure("A test migration lacks a unique baseline identity.")
    exact_source_assertions = {
        "context$adt_reader_has_no_local_optional_carrier_adapter",
        "builtin_atom_discovery$every_declared_case_is_a_live_enum_member_and_every_member_is_declared",
        "issue_1125_annotation_reader_parity$root_node_dispatch_names_every_deep_tag",
        "issue_1923_pipe_fold_placement$the_public_checker_entries_are_the_ones_this_file_covers",
        "issue_1125_e5b_linearity$linearity_reader_has_only_role_total_carrier_matches",
        "issue_1125_e5b_linearity$carrier_totality_ratchet_is_signature_and_visibility_independent",
        "expand_insert_dispatch_family$vmap_movement_dispatch_families_cover_expand_and_insert",
        "expand_insert_dispatch_family$each_dispatch_family_operation_is_mutation_live",
        "expand_insert_dispatch_family$unrelated_literals_cannot_authorize_a_dispatch_family",
        "chelis_types$static_seed::tests::static_seed_reader_has_no_local_optional_carrier_adapter",
        "chelis_types$builtins::tests::doc_surface_section_4_mirrors_builtin_names",
    }
    for identity, record in retired.items():
        permitted = identity.startswith("chelis-types::chelis_types$source_arch::tests::") or identity.startswith((
            "chelis-types::stack_guard_coverage$", "chelis-types::rt800_checker_session_source_contract$"))
        permitted = permitted or identity.removeprefix("chelis-types::") in exact_source_assertions
        if not permitted:
            raise OracleFailure("Only incidental source assertions can retire.")
        if (set(record) != {"reason", "behavior_identities"} or not record["reason"]
                or not record["behavior_identities"]
                or not set(record["behavior_identities"]) <= after | set(external) or identity in after):
            raise OracleFailure("A retired source assertion lacks actual behavior evidence.")
    mapping = {}
    for identity in sorted(before - set(retired)):
        if identity in renamed:
            target = renamed[identity]
        elif identity in after:
            target = identity
        else:
            name = identity.split("$", 1)[1]
            matches = [value for value in after if value.split("$", 1)[1] == name]
            if len(matches) != 1:
                raise OracleFailure(f"A missing or ambiguous test needs an explicit owner: {identity}")
            target = matches[0]
        if target not in after or target in mapping.values():
            raise OracleFailure("A migrated test is absent or substitutes another obligation.")
        mapping[identity] = target
    return mapping


def check_timing(previous: dict, current: dict, mapping: dict[str, str], retired: dict) -> None:
    timing_moves = {}
    for before, after in mapping.items():
        old_package, old_name = before.split("::", 1)[0], before.split("$", 1)[1]
        new_package, new_name = after.split("::", 1)[0], after.split("$", 1)[1]
        old_key, new_key = old_package + "::" + old_name, new_package + "::" + new_name
        if old_key in timing_moves and timing_moves[old_key] != new_key:
            raise OracleFailure("A timing identity has multiple destinations.")
        timing_moves[old_key] = new_key
    retired_keys = {identity.split("::", 1)[0] + "::" + identity.split("$", 1)[1]
        for identity in retired}
    for key, value in previous.items():
        if key in retired_keys:
            if key in current:
                raise OracleFailure(f"A retired source assertion retains a stale timing owner: {key}")
            continue
        target = timing_moves.get(key, key)
        if target not in current or canonical(current[target]) != canonical(value) or (target != key and key in current):
            raise OracleFailure(f"A measured timing value changed or lost its owner: {key}")


def preserve_cost_sources(evidence: Evidence) -> None:
    source = load(FIXTURES / "immutable_source.json")["files"]
    records = {}
    comparisons = []
    for name in source:
        if not name.startswith("crates/chelis-types/tests/"):
            continue
        original = support.run(("git", "show", f"{BASELINE}:{name}")).stdout
        actual = (ROOT / name).read_text()
        comparisons.append({"path": name, "baseline": original, "candidate": actual})
        records[name] = {"baseline": source[name], "candidate": digest(ROOT / name)}
    text = json.dumps(comparisons, ensure_ascii=False)
    if '"####' in text:
        raise OracleFailure("A cost source contains its Rust delimiter.")
    consumer_source = (FIXTURES / "cost_sources.rs").read_text()
    consumer_source += '\nconst SOURCES_JSON: &str = r####"' + text + '"####;\n'
    result = support.consumer(CRATE, consumer_source, dependencies=(
        'syn = { version = "2", features = ["full", "visit-mut"] }\n'
        'quote = "1"\nserde_json = "1"\n'))
    if result.stdout.splitlines().count("CHECKER_COST_SOURCE_CONTRACTS accepted") != 1:
        raise OracleFailure("Full-host cost syntax evidence is absent or duplicate.")
    evidence.save("cost-source-contracts.json", records)


def candidate_arguments(arguments: tuple[str, ...], baseline: dict, migrations: dict, *,
        targets: set[tuple[str, str]] | None = None) -> tuple[str, ...]:
    retired = set(migrations["retired"])
    package = arguments[arguments.index("-p") + 1]
    result = []
    index = 0
    while index < len(arguments):
        if arguments[index] == "--test":
            target = arguments[index + 1]
            identities = {identity for identity in baseline["selected"]
                if identity.split("$", 1)[0].split("::", 1)[1] == target}
            if identities and identities <= retired:
                index += 2
                continue
            if targets is not None and (package, target) not in targets:
                if not identities or (CRATE, target) not in targets:
                    raise OracleFailure(f"A mandatory test target lacks its exact execution owner: {package}::{target}")
                index += 2
                continue
        result.append(arguments[index])
        index += 1
    return tuple(result)


def host_controls(previous: dict, evidence: Evidence, external: set[str]) -> None:
    migrations = load(FIXTURES / "test_migrations.json")
    metadata = decode(support.run(("cargo", "metadata", "--locked", "--no-deps",
        "--format-version", "1")).stdout)
    targets = {(package["name"], target["name"]) for package in metadata["packages"]
        for target in package["targets"] if "test" in target["kind"]}
    current = {name: tests(candidate_arguments(arguments, previous["tests"][name], migrations, targets=targets), ROOT)
        for name, arguments in LANES.items()}
    current["portable-unit"] = tests(("-p", CRATE, "--all-targets", "--features",
        "serde,generalize-sweep-oracle"), ROOT)
    mapping = reconcile_inventories(previous["tests"], current, migrations, external=external)
    check_timing(previous["timing"], load(ROOT / "scripts/test_timing_baseline.json"),
        mapping, migrations["retired"])
    evidence.save("current-tests.json", current)
    evidence.save("cost-observations.json", {"baseline": previous["cost_observations"],
        "candidate": cost_observations(current)})
    evidence.save("test-migrations.json", {"identities": mapping, **migrations})
    preserve_cost_sources(evidence)


def prior_contracts(source: str) -> dict[str, str]:
    protected = {"REQUIRED", "SURF_CASES", "HOST_CASES", "EVIDENCE", "BASE_LANES",
        "SOURCE_MODULES", "FEATURES", "SCALES", "COST_OPERATIONS", "RETIRED_SOURCE_TEST"}
    contracts = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and (target.id in protected
                    or "BASELINE" in target.id):
                contracts[target.id] = ast.dump(node.value, include_attributes=False)
    if not any("BASELINE" in name for name in contracts):
        raise OracleFailure("A previous core oracle lacks an immutable baseline.")
    return contracts


def prior_core_obligations(evidence: Evidence) -> None:
    scripts = ("numeric_core_oracle.py", "type_unification_core_oracle.py",
        "dag_core_oracle.py", "effect_analysis_core_oracle.py")
    identities = {}
    for name in scripts:
        path = ROOT / "scripts" / name
        original = support.run(("git", "show", f"{BASELINE}:scripts/{name}")).stdout
        if prior_contracts(path.read_text()) != prior_contracts(original):
            raise OracleFailure(f"A previous core oracle changed its accepted baseline or obligations: {name}")
        identities[name] = {"candidate": digest(path),
            "baseline_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "contracts": prior_contracts(original)}
        support.run((sys.executable, str(path)))
    evidence.save("prior-core-obligations.json", identities)


def ci_ownership(evidence: Evidence) -> None:
    import ci_change_owned as ci
    config = ci.read_config(ROOT / ".config/ci-test-targets.toml")
    metadata = decode(support.run(("cargo", "metadata", "--locked", "--no-deps",
        "--format-version", "1")).stdout)
    tracked = set(support.run(("git", "ls-files", "-z")).stdout.split("\0")) - {""}
    ci.validate_config(config, metadata, tracked, lambda name: (ROOT / name).read_text())
    paths = ["scripts/checker_core_oracle.py", "scripts/test_checker_core_oracle.py"]
    paths.extend(str(path.relative_to(ROOT)) for path in sorted(FIXTURES.rglob("*")) if path.is_file())
    owners = {}
    for path in paths:
        matches = [rule for rule in config.path_rules if rule.matches(path)]
        if len(matches) != 1 or matches[0].disposition != "owner" or matches[0].owner is None:
            raise OracleFailure(f"A checker acceptance input lacks one exact CI owner: {path}")
        owner = matches[0].owner
        if not (ROOT / ".github/workflows" / owner.workflow).is_file():
            raise OracleFailure(f"A checker CI workflow is absent: {owner.workflow}")
        owners[path] = owner.as_dict()
    evidence.save("ci-ownership.json", {"configuration_digest": ci.config_digest(config), "owners": owners})


def policy_controls(evidence: Evidence) -> None:
    ci_ownership(evidence)
    environment = support._environment()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), environment.get("PYTHONPATH")) if value)
    for name in ("test_checker_core_oracle.py", "test_gate.py", "test_ci_change_owned.py",
            "test_check_configuration_closure.py", "test_no_std_core_support.py",
            "pipeline_core_dependency_guard.py", "pipeline_core_documentation_guard.py"):
        support.run((sys.executable, str(ROOT / "scripts" / name)), env=environment)
    support.run(("cargo", "check", "--locked", "-p", "chelis-cli", "-p", "chelis-compiler-api",
        "-p", "chelis-pipeline-core", "-p", "chelis-ir", "-p", "chelis-prove",
        "-p", "chelis-effects", "-p", "chelis-reef", "-p", "chelis-validate",
        "-p", "chelis-lsp", "-p", "chelis-python", "-p", "chelis-tide", "-p", "chelis-e2e",
        "--all-targets"))


def compiled_artifacts() -> dict:
    target = Path(support._environment()["CARGO_TARGET_DIR"])
    if not target.is_absolute():
        target = ROOT / target
    dependencies = target / TARGET / "debug/deps"
    libraries = list(dependencies.glob("libchelis_checker_core-*.rlib"))
    abi = list(dependencies.glob("libchelis_abi-*.rlib"))
    objects = list(dependencies.glob("chelis_no_std_consumer-*.o"))
    if not libraries or not abi or not objects:
        raise OracleFailure("A checker, descriptor, or allocator target artifact is absent.")
    return {str(path): digest(path) for path in sorted(set(libraries + abi + objects))}


def candidate_lanes(previous: dict, evidence: Evidence) -> None:
    if previous["fixtures"] != fixture_identity():
        raise OracleFailure("The fixed source comparison corpus changed.")
    portable_consumers(evidence)
    forbidden_graph_controls(evidence)
    external = authority_controls(evidence) | {"consumer::independent-core"}
    for kind in ("host", "diagnostics", "abi"):
        value = snapshot(ROOT, kind)
        evidence.save(f"candidate-{kind}.json", value)
        compare_snapshots(previous[kind], value, kind)
    host_controls(previous, evidence, external)
    prior_core_obligations(evidence)
    policy_controls(evidence)
    evidence.save("compiled-artifacts.json", compiled_artifacts())
    evidence.save("candidate-fixtures.json", {str(path.relative_to(FIXTURES)): digest(path)
        for path in sorted(FIXTURES.rglob("*")) if path.is_file()})
    if previous["fixtures"] != fixture_identity():
        raise OracleFailure("The fixed source comparison inputs changed during candidate acceptance.")


def source_identity() -> dict:
    files = support.run(("git", "ls-files", "-z")).stdout.split("\0")
    return {name: digest(ROOT / name) for name in files if name and (ROOT / name).is_file()}


def require_clean_head() -> str:
    if support.run(("git", "status", "--porcelain=v1", "--untracked-files=normal")).stdout.strip():
        raise OracleFailure("Complete acceptance needs a clean source commit.")
    return support.run(("git", "rev-parse", "HEAD")).stdout.strip()


def run_oracle(evidence: Evidence) -> None:
    head = require_clean_head()
    source = source_identity()
    evidence.save("provenance.json", {"head": head, "baseline": BASELINE, "source": source,
        "fixtures": fixture_identity(), "environment": {key: os.environ.get(key) for key in (
            "RUSTUP_TOOLCHAIN", "RUSTC", "RUSTFLAGS", "CARGO_TARGET_DIR", "CARGO_BUILD_JOBS",
            "CARGO_PROFILE_DEV_DEBUG", "CARGO_PROFILE_TEST_DEBUG", "LLVM_SYS_231_PREFIX",
            "SDKROOT", "CC", "CXX", "AR", "MACOSX_DEPLOYMENT_TARGET")}})
    previous = capture_baseline(evidence)
    candidate_lanes(previous, evidence)
    if require_clean_head() != head or source_identity() != source:
        raise OracleFailure("Candidate source changed during acceptance.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline", type=Path)
    options = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        print("The checker oracle requires Linux or macOS.", file=sys.stderr)
        return 1
    if sys.version_info[:2] != (3, 11) or Path.cwd().resolve() != ROOT:
        print("Use managed Python 3.11 from the repository root.", file=sys.stderr)
        return 1
    parent = ARTIFACTS / sys.platform
    parent.mkdir(parents=True, exist_ok=True)
    directory = options.capture_baseline or Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    evidence = Evidence(directory.resolve(), "baseline-capture" if options.capture_baseline else "implementation")
    support._execute = evidence.execute
    try:
        environment = support._environment()
        commands = [(environment.get("RUSTC", "rustc"), "-vV"),
            (environment.get("RUSTDOC", "rustdoc"), "-vV"), ("cargo", "--version"),
            ("cargo", "nextest", "--version"), ("git", "--version"), ("cc", "--version"),
            ("rustup", "target", "list", "--installed")]
        for command in commands:
            support.run(command, env=environment)
        if options.capture_baseline:
            capture_baseline(evidence)
        else:
            run_oracle(evidence)
        evidence.finish("passed")
    except (OracleFailure, OSError, ValueError, KeyError, TypeError, ET.ParseError, SyntaxError) as error:
        evidence.finish("failed")
        print(f"Checker core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        support._execute = evidence.original
    print(f"Checker {'baseline capture' if options.capture_baseline else 'core oracle'}: PASS ({sys.platform})")
    print(f"Evidence: {evidence.directory}")
    if options.capture_baseline:
        print("Baseline capture does not establish implementation acceptance.")
    print("Target compilation does not prove device execution or unrestricted stack capacity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
