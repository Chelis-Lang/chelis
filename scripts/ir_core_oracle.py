#!/usr/bin/env python3
"""Compare the complete portable IR engine with its immutable native source."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET

import checker_core_oracle as checker
import compiler_pipeline_oracle as pipeline
import dag_core_oracle as dag
import no_std_core_support as support
from no_std_core_support import ROOT, TARGET, OracleFailure

BASELINE = "23dc12afb7b0404b7f70a9fc2cacc3df7ed77cbe"
STORE = "97fc63833f45e8bd475c8046d125e1508e6a6f2f"
CRATE = "chelis-ir-core"
FIXTURES = ROOT / "scripts/fixtures/ir_core"
ARTIFACTS = ROOT / "target/agents/ir-core"
FORBIDDEN = (
    "chelis-ir", "chelis-types", "chelis-effects", "chelis-surf", "chelis-syntax-host",
    "chelis-compiler-api", "chelis-pipeline-core", "chelis-runtime", "chelis-prove",
    "chelis-stdlib", "chelis-cli", "chelis-reef", "chelis-backend-c", "chelis-backend-hip",
    "chelis-backend-metal", "stacker", "psm", "tokio", "rayon", "parking_lot", "getrandom", "pyo3",
)
PRIOR = ("numeric_core_oracle.py", "type_unification_core_oracle.py", "dag_core_oracle.py",
    "effect_analysis_core_oracle.py", "checker_core_oracle.py")
GROUPS = frozenset({"target-graphs", "independent-consumers", "immutable-semantics",
    "native-production", "complete-suites", "compiler-pipeline", "prior-cores",
    "negative-controls", "migration", "resources-caches-costs", "ci-ownership"})
POINTWISE = "eval_performance_pointwise_pair_witness"
PERFORMANCE = "eval_performance"
NATIVE_PACKAGES = ("chelis-backend-c", "chelis-backend-hip", "chelis-backend-metal",
    "chelis-effects", "chelis-compiler-api", "chelis-pipeline-core")
TIMING_PATHS = ("scripts/test_timing_baseline.json", ".config/ci-change-owned-durations.json")
LIMITS = {
    "target": TARGET, "physical_device_execution": False,
    "allocation": "The consumer supplies an allocator.",
    "stack": "The consumer supplies StackServices and adequate reservations.",
    "native_execution": "Native C output executes on this host only.",
    "scope": "The core is portable. Native admission and the full compiler need std.",
    "hosted_acceptance": "Linux and macOS each need this complete command at the exact source head.",
}
digest = checker.digest
load = checker.load
decode = checker.decode
canonical = checker.canonical


def require(condition: bool, message: str) -> None:
    if not condition:
        raise OracleFailure(message)


def fixture_identity() -> dict:
    return {str(path.relative_to(FIXTURES)): digest(path)
        for path in sorted(FIXTURES.rglob("*")) if path.is_file()}


class Evidence(checker.Evidence):
    """Retain real commands, tool hashes, consumer locks, and raw output."""

    def __init__(self, directory: Path, scope: str):
        super().__init__(directory, scope)
        self.script_identity = digest(Path(__file__))
        self.groups: set[str] = set()

    def execute(self, command, *, cwd=ROOT, env=None, live_stderr=False):
        selected_env = dict(env) if env is not None else support._environment()
        executable = shutil.which(str(command[0]), path=selected_env.get("PATH"))
        require(executable is not None, "A command executable is absent.")
        tool = Path(executable).resolve()
        before = {"path": str(tool), **digest(tool)}
        environment_tools = {}
        for key in ("RUSTC", "RUSTDOC", "CC", "CXX", "AR"):
            path = shutil.which(selected_env.get(key, ""), path=selected_env.get("PATH"))
            if path:
                resolved = Path(path).resolve()
                environment_tools[key] = {"path": str(resolved), **digest(resolved)}
        self.save(f"command-{len(self.receipts):05d}.inputs.json",
            {"command": list(map(str, command)), "tool": before, "environment_tools": environment_tools})
        result = super().execute(command, cwd=cwd, env=selected_env, live_stderr=live_stderr)
        receipt = self.receipts[-1]
        require(receipt["tool"] == before, "A command executable changed during execution.")
        receipt["environment"] = {key: selected_env.get(key) for key in (
            "RUSTUP_TOOLCHAIN", "RUSTC", "RUSTDOC", "RUSTFLAGS", "CARGO_TARGET_DIR",
            "CARGO_BUILD_JOBS", "SDKROOT", "CC", "CXX", "AR", "CHELIS_IR_CORPUS")}
        receipt["environment_tools"] = environment_tools
        inputs = self.directory / f"command-{receipt['identity']:05d}.inputs.json"
        receipt["files"]["inputs"] = {"path": inputs.name, **digest(inputs)}
        self.save("receipts.json", self.document("active"))
        return result

    def document(self, status: str) -> dict:
        return {"status": status, "scope": self.scope, "platform": sys.platform,
            "machine": platform.machine(), "python": sys.executable,
            "python_version": platform.python_version(), "script": self.script_identity,
            "invocation": sys.argv, "command_count": len(self.receipts), "receipts": self.receipts}

    def complete(self, group: str) -> None:
        require(group in GROUPS and group not in self.groups, "An acceptance group is unknown or duplicate.")
        self.groups.add(group)

    def finish(self, status: str) -> None:
        if status == "passed" and self.scope == "implementation":
            require(self.groups == GROUPS, "The complete acceptance group set differs.")
        self.save("receipts.json", self.document(status))
        if status == "passed":
            validate_receipts(self.directory)
            validate_artifacts(self.directory, self.scope)
        self.save("report.json", {"status": status, "scope": self.scope,
            "complete_implementation_acceptance": status == "passed" and self.scope == "implementation",
            "source": BASELINE, "accepted_store": STORE, "groups": sorted(self.groups),
            "limits": LIMITS})
        self.save("artifact-index.json", {str(path.relative_to(self.directory)): digest(path)
            for path in sorted(self.directory.rglob("*"))
            if path.is_file() and path.name != "artifact-index.json"})


def validate_receipts(directory: Path) -> list[dict]:
    document = load(directory / "receipts.json")
    required = {"status", "scope", "platform", "machine", "python", "python_version",
        "script", "invocation", "command_count", "receipts"}
    require(set(document) == required and document["scope"] in ("baseline-capture", "implementation")
        and document["script"] == digest(Path(__file__)) and document["invocation"],
        "The oracle source, scope, or invocation identity differs.")
    receipts = document["receipts"]
    require(document["status"] == "passed" and receipts
        and document["command_count"] == len(receipts), "Command evidence is absent or incomplete.")
    require([item["identity"] for item in receipts] == list(range(len(receipts))),
        "Command identities are missing or duplicate.")
    paths = set()
    for receipt in receipts:
        require(receipt["command"] and type(receipt["exit"]) is int,
            "A command or exit identity is absent.")
        require(type(receipt["elapsed_ns"]) is int and receipt["elapsed_ns"] >= 0,
            "A command duration is invalid.")
        require({"stdout", "stderr"} <= receipt["files"].keys(), "Raw command output is absent.")
        for tool in [receipt["tool"], *receipt.get("environment_tools", {}).values()]:
            require(Path(tool["path"]).is_absolute()
                and digest(Path(tool["path"])) == {key: tool[key] for key in ("bytes", "sha256")},
                "An executable tool identity differs.")
        for value in receipt["files"].values():
            relative = Path(value["path"])
            require(not relative.is_absolute() and ".." not in relative.parts and str(relative) not in paths,
                "An artifact path is unsafe or duplicate.")
            paths.add(str(relative))
            require(digest(directory / relative) == {key: value[key] for key in ("bytes", "sha256")},
                "A raw artifact size or digest differs.")
        inputs = load(directory / receipt["files"]["inputs"]["path"])
        require(inputs["command"] == receipt["command"] and inputs["tool"] == receipt["tool"]
            and inputs["environment_tools"] == receipt.get("environment_tools", {}),
            "A command differs from its pre-execution tool inputs.")
        if receipt["command"][:3] == ["cargo", "nextest", "run"]:
            require("junit" in receipt["files"], "A nextest command lacks its raw JUnit artifact.")
        if receipt["consumer"]:
            require({"Cargo.toml", "Cargo.lock"} <= receipt["files"].keys()
                and bool({"src/main.rs", "src/lib.rs"} & receipt["files"].keys()),
                "An independent consumer lacks its exact source, manifest, or lock.")
        if receipt["exit"]:
            failure = receipt.get("expected_failure", {})
            raw = "\n".join((directory / receipt["files"][stream]["path"]).read_text()
                for stream in ("stdout", "stderr"))
            require(failure.get("fragments") and all(text in raw for text in failure["fragments"]),
                "A failed command lacks its intended raw diagnostic.")
    return receipts


def validate_artifacts(directory: Path, scope: str) -> None:
    required = {"tool-inputs.json", "frozen-inputs.json", "baseline-source.json",
        "baseline.json", "historical-failures.json", "receipts.json"}
    if scope == "implementation":
        required.update({"provenance.json", "candidate-snapshot.json", "candidate-wire.json",
            "candidate-production.json", "runtime-seals.json", "current-tests.json", "test-migrations.json",
            "independent-consumers.json", "target-graphs.json", "negative-graphs.json",
            "authority-controls.json", "compiler-pipeline.json", "prior-core-obligations.json",
            "cost-observations.json", "cost-source-provenance.json", "timing-preservation.json",
            "ci-ownership.json", "compiled-artifacts.json"})
    require(all((directory / name).is_file() for name in required), "A mandatory evidence artifact is absent.")
    for tool in load(directory / "tool-inputs.json").values():
        require(digest(Path(tool["path"])) == {key: tool[key] for key in ("bytes", "sha256")},
            "A tool changed after its pre-execution identity capture.")
    previous = load(directory / "baseline.json")
    frozen = load(directory / "frozen-inputs.json")
    require(previous["revision"] == frozen["revision"] == BASELINE
        and previous["fixtures"] == frozen["fixtures"] == fixture_identity(),
        "Frozen source or fixture evidence differs.")
    validate_snapshot(previous["snapshot"])
    if scope == "implementation":
        compare_exact(previous["snapshot"], load(directory / "candidate-snapshot.json"), "retained semantic fields")
        compare_exact(previous["wire"], load(directory / "candidate-wire.json"), "retained binary wire fields")


def inventory_cases(metadata: dict) -> dict[str, bool]:
    result = {}
    for suite in metadata["rust-suites"].values():
        prefix = f"{suite['package-name']}::{suite['binary-name']}$"
        for name, case in suite["testcases"].items():
            require(case["kind"] == "test" and case["filter-match"]["status"] in ("matches", "mismatch"),
                "A test inventory record is invalid.")
            if case["filter-match"]["status"] != "matches":
                continue
            identity = prefix + name
            require(identity not in result, "A selected test identity is duplicate.")
            require(type(case["ignored"]) is bool, "A test ignore state is invalid.")
            result[identity] = case["ignored"]
    require(bool(result), "The selected test inventory is empty.")
    return result


def execution(stdout: str, selected: set[str], allowed_ignored: set[str]) -> set[str]:
    require(bool(selected) and not selected & allowed_ignored, "A test execution selection is empty or conflicts.")
    started, completed, ignored = Counter(), Counter(), Counter()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        event = decode(line)
        if event["type"] != "test":
            continue
        name, state = event["name"], event["event"]
        if name in allowed_ignored:
            require(state in ("started", "ignored"), "An unselected ignored case executed.")
            if state == "ignored":
                ignored[name] += 1
            continue
        require(name in selected, f"An unexpected test event appears: {name}")
        if state == "started":
            started[name] += 1
        elif state == "ok":
            completed[name] += 1
        else:
            raise OracleFailure(f"A mandatory test did not pass: {name}: {state}")
    require(set(started) == selected == set(completed)
        and all(count == 1 for count in (*started.values(), *completed.values(), *ignored.values())),
        "Selected and actual execution identities differ.")
    return set(completed)


def junit(raw: str, metadata: dict, selected: set[str], allowed_ignored: set[str]) -> dict:
    suites = {identity: f"{suite['package-name']}::{suite['binary-name']}"
        for identity, suite in metadata["rust-suites"].items()}
    observations = {}
    skipped = set()
    for suite in ET.fromstring(raw).iter("testsuite"):
        require(suite.attrib["name"] in suites, "A JUnit suite lacks its binary identity.")
        for case in suite.findall("testcase"):
            identity = suites[suite.attrib["name"]] + "$" + case.attrib["name"]
            require(identity not in observations and identity not in skipped, "A JUnit identity is duplicate.")
            if identity in allowed_ignored:
                require(case.find("skipped") is not None, "An ignored JUnit case lacks its skip state.")
                skipped.add(identity)
                continue
            require(identity in selected and "time" in case.attrib, "A JUnit identity or duration is absent.")
            require(not any(case.find(name) is not None for name in (
                "failure", "error", "skipped", "flakyFailure", "flakyError", "rerunFailure", "rerunError")),
                "A mandatory JUnit case failed, skipped, or retried.")
            observations[identity] = {"attributes": dict(case.attrib),
                "stdout": case.findtext("system-out", ""), "stderr": case.findtext("system-err", "")}
    require(set(observations) == selected, "JUnit and selected identities differ.")
    return {"raw_xml": raw, "observations": observations, "ignored": sorted(skipped)}


def environment(workspace: Path) -> dict:
    result = support._environment()
    if workspace != ROOT:
        result["CARGO_TARGET_DIR"] = str(workspace / "target")
    result["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"
    return result


def suite(workspace: Path, packages: tuple[str, ...], *, selection="ordinary", release=False) -> dict:
    args = tuple(value for package in packages for value in ("-p", package))
    args += ("--all-targets",)
    if "chelis-ir" in packages or CRATE in packages:
        args += ("--features", "lowering-trace")
    if release:
        args += ("--release",)
    if selection != "ordinary":
        args += ("-E", f"binary(={PERFORMANCE}) & " + (
            f"test(={POINTWISE})" if selection == "pointwise-release" else f"not(test(={POINTWISE}))"))
    with tempfile.TemporaryDirectory(prefix="ir-nextest-") as temporary:
        report = Path(temporary) / "results.xml"
        configuration = (workspace / ".config/nextest.toml").read_text()
        if "ir-core-acceptance" not in tomllib.loads(configuration)["profile"]:
            configuration += '\n[profile.ir-core-acceptance]\ninherits = "ci-full"\n'
        configuration += ('\n[profile.ir-core-acceptance.junit]\n'
            f'path = {json.dumps(str(report))}\nstore-success-output = true\nstore-failure-output = true\n')
        config = Path(temporary) / "nextest.toml"
        config.write_text(configuration)
        flags = ("--config-file", str(config), "--profile", "ir-core-acceptance", "--ignore-default-filter")
        listed = support.run(("cargo", "nextest", "list", "--locked", *args, *flags,
            "--run-ignored", "all", "--message-format", "json"), cwd=workspace, env=environment(workspace))
        metadata = decode(listed.stdout)
        cases = inventory_cases(metadata)
        manual = selection != "ordinary"
        selected = {name for name, ignored in cases.items() if ignored == manual}
        omitted = set(cases) - selected
        require(bool(selected), "A mandatory suite selection is empty.")
        ignore_args = ("--run-ignored", "only") if manual else ()
        run = support._execute(("cargo", "nextest", "run", "--locked", *args, *flags, *ignore_args,
            "--test-threads", "1", "--retries", "0", "--no-fail-fast",
            "--message-format", "libtest-json-plus"), cwd=workspace, env=environment(workspace))
        require(run.returncode == 0, "A mandatory complete nextest command failed.")
        return {"inventory": metadata, "selected": sorted(selected),
            "executed": sorted(execution(run.stdout, selected, omitted)),
            "configuration": configuration, "junit": junit(report.read_text(), metadata, selected, omitted)}


@contextmanager
def immutable_checkout(evidence: Evidence):
    # Later prior-core checks also need the candidate's baseline objects.
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{tree}}"), cwd=ROOT)
    if present.returncode:
        evidence.accept_negative([BASELINE], present.stderr)
        support.run(("git", "fetch", "--quiet", "--no-tags", "--depth=1",
            checker.REPOSITORY, BASELINE), cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="chelis-ir-baseline-") as temporary:
        workspace = Path(temporary) / "source"
        support.run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(workspace)))
        present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{tree}}"), cwd=workspace)
        if present.returncode:
            evidence.accept_negative([BASELINE], present.stderr)
            support.run(("git", "fetch", "--quiet", "--no-tags", "--depth=1",
                checker.REPOSITORY, BASELINE), cwd=workspace)
        support.run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=workspace)
        require(support.run(("git", "rev-parse", "HEAD"), cwd=workspace).stdout.strip() == BASELINE,
            "The immutable checkout revision differs.")
        files = load(FIXTURES / "immutable_source.json")
        require(files["revision"] == BASELINE, "The immutable source manifest revision differs.")
        actual = {name: digest(workspace / name) for name in files["files"] if (workspace / name).is_file()}
        require(actual == files["files"], "The immutable checkout content differs from its complete source manifest.")
        evidence.save("baseline-source.json", {"revision": BASELINE, "files": actual})
        yield workspace


def dependencies(workspace: Path, names: tuple[str, ...], *, defaults=True) -> str:
    return "".join(checker.dependency(name, workspace, defaults=defaults) for name in names)


def semantic_snapshot(workspace: Path, *, baseline: bool) -> dict:
    source = (FIXTURES / ("baseline_snapshot.rs" if baseline else "snapshot.rs")).read_text()
    names = ("chelis-dag-core", "chelis-deep", "chelis-effects", "chelis-types",
        "chelis-numeric-core", "chelis-syntax-host", "chelis-surf")
    extra = dependencies(workspace, names)
    if not baseline:
        extra += checker.dependency(CRATE, workspace)
    extra += 'serde_json = { version = "1", features = ["float_roundtrip"] }\nbincode = "=1.3.3"\n'
    env = environment(workspace)
    env["CHELIS_IR_CORPUS"] = str(FIXTURES / "corpus.json")
    result = support.consumer(workspace / "crates/chelis-ir", source,
        features=("lowering-trace",), default_features=True, dependencies=extra, env=env)
    value = dag.snapshot_record(result.stdout, "IR_SNAPSHOT")
    validate_snapshot(value)
    return value


def validate_snapshot(value: dict) -> None:
    schema = load(FIXTURES / "schema.json")
    require(schema["source"] == BASELINE and schema["accepted_store"] == STORE,
        "The frozen schema source identity differs.")
    require(set(value) == {"version", "cases", "contexts"} and value["version"] == schema["version"],
        "The semantic snapshot shape differs.")
    for field, identities, allowed in (("cases", schema["case_ids"], schema["record_fields"]),
            ("contexts", schema["context_ids"], schema["context_fields"])):
        records = value[field]
        require([record["id"] for record in records] == identities and len(identities) == len(set(identities)),
            "A semantic case is missing, duplicate, or out of order.")
        for record in records:
            expected = {"id", "check_error"} if "check_error" in record else set(allowed) - {"check_error"}
            require(set(record) == expected, "A semantic comparison field is absent or unexpected.")
    # No field normalization or round-trip comparison replaces the frozen bytes.


def compare_exact(previous, current, label: str) -> None:
    require(canonical(previous) == canonical(current), f"The complete {label} comparison differs.")


def preserve_history(evidence: Evidence) -> None:
    history = load(FIXTURES / "historical_failures.json")
    require(history["baseline"] == BASELINE and history["failures"], "Historical failure evidence is absent.")
    for record in history["failures"]:
        require(record["identity"] and record["outcome"] == "failed" and record["artifacts"],
            "A historical failure lacks its identity, outcome, or raw artifact.")
        for name, identity in record["artifacts"].items():
            require(digest(FIXTURES / name) == identity, "A retained historical failure differs.")
    evidence.save("historical-failures.json", history)


def production(workspace: Path, directory: Path, evidence: Evidence) -> dict:
    directory.mkdir()
    source = directory / "portable_graph.ch"
    source.write_text("def f(x: tensor[4, f32]) -> tensor[4, f32] = neg(x)\n"
        "out = f(to_tensor([1.0, -2.0, 3.5, 0.25], f32))\n")
    first = len(evidence.receipts)
    value = dag.production_snapshot(workspace, source, directory)
    for receipt in evidence.receipts[first:]:
        if receipt["exit"]:
            stderr = (evidence.directory / receipt["files"]["stderr"]["path"]).read_text()
            fragments = ["[05-OP-46]", "exp"] if "[05-OP-46]" in stderr else ["f64"] if "f64" in stderr else ["FP64"]
            require(all(fragment in stderr for fragment in fragments), "A native admission failure is unrelated.")
            receipt["expected_failure"] = {"fragments": fragments, "diagnostic": stderr}
    require(set(value) == set(load(FIXTURES / "schema.json")["production_artifacts"]),
        "Native artifact or admission evidence is incomplete.")
    return value


def runtime_seals(directory: Path) -> dict:
    result = {}
    for backend in ("c", "hip", "metal"):
        destination = directory / backend
        path = destination / dag.RECEIPT_FILE_NAME
        if not path.is_file():
            continue
        document = dag.read_receipt_document(path)
        archive = destination / document["archive"]
        result[backend] = dag.read_staging_receipt(path, archive, headers_dir=destination)
    require("c" in result, "The native runtime seal is absent.")
    return result


def capture_baseline(evidence: Evidence) -> dict:
    frozen = fixture_identity()
    evidence.save("frozen-inputs.json", {"revision": BASELINE, "fixtures": frozen})
    preserve_history(evidence)
    with immutable_checkout(evidence) as workspace:
        ordinary = suite(workspace, ("chelis-ir",))
        cases = inventory_cases(ordinary["inventory"])
        require(len(cases) == 1788 and len(ordinary["executed"]) == 1778,
            "The immutable complete IR inventory differs.")
        performance = suite(workspace, ("chelis-ir",), selection="performance-manual")
        pointwise = suite(workspace, ("chelis-ir",), selection="pointwise-release", release=True)
        require(len(performance["executed"]) == 3 and len(pointwise["executed"]) == 1,
            "The four full evaluation-performance witnesses differ.")
        manual = set(performance["executed"]) | set(pointwise["executed"])
        obligations = sorted(name for name, ignored in cases.items() if ignored and name not in manual)
        require(len(obligations) == 6, "The separate ignored obligation inventory differs.")
        support.run(("cargo", "test", "--locked", "-p", "chelis-ir", "--features", "lowering-trace", "--doc"),
            cwd=workspace, env=environment(workspace))
        native = suite(workspace, NATIVE_PACKAGES)
        native_ignored = sorted(identity for identity, ignored in inventory_cases(native["inventory"]).items() if ignored)
        value = {"revision": BASELINE, "fixtures": frozen,
            "tests": {"ordinary": ordinary, "performance-manual": performance,
                "pointwise-release": pointwise, "native-integration": native},
            "manual_obligations": obligations, "native_manual_obligations": native_ignored,
            "snapshot": semantic_snapshot(workspace, baseline=True),
            "wire": dag.wire_snapshot(workspace, baseline=False),
            "production": production(workspace, evidence.directory / "baseline-native", evidence),
            "runtime_seals": runtime_seals(evidence.directory / "baseline-native"),
            "timing": {name: load(workspace / name) for name in TIMING_PATHS},
            "cost_sources": {str(path.relative_to(workspace)): digest(path)
                for path in sorted((workspace / "crates/chelis-ir/tests").glob("*.rs"))
                if path.stem.endswith("cost") or path.stem in ("callable_snapshot_allocations", PERFORMANCE)},
            "scope": "baseline-capture", "complete_implementation_acceptance": False}
        evidence.save("baseline.json", value)
    require(fixture_identity() == frozen, "The frozen inputs changed during baseline capture.")
    return value


def inspect_graph(metadata: dict) -> dict:
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    packages = {package["id"]: package for package in metadata["packages"]}
    # Retain paths so a forbidden owner or feature has an exact route.
    pending = [(metadata["resolve"]["root"], [])]
    paths = {}
    while pending:
        identity, path = pending.pop()
        if identity in paths:
            continue
        path = [*path, packages[identity]["name"]]
        paths[identity] = path
        for edge in nodes[identity]["deps"]:
            if any(kind["kind"] is None for kind in edge["dep_kinds"]):
                if not any("proc-macro" in target["kind"] for target in packages[edge["pkg"]]["targets"]):
                    pending.append((edge["pkg"], path))
    try:
        names = support._normal_packages(metadata, FORBIDDEN)
    except OracleFailure as error:
        offending = [path for identity, path in paths.items()
            if packages[identity]["name"] in FORBIDDEN or "std" in nodes[identity]["features"]]
        raise OracleFailure(f"{error}. Dependency paths: {offending}") from error
    return {"packages": sorted(names), "paths": paths, "metadata": metadata}


def target_graphs(evidence: Evidence) -> None:
    manifest = tomllib.loads((ROOT / "crates" / CRATE / "Cargo.toml").read_text())
    features = manifest.get("features", {})
    require(not features.get("default") and "std" not in features and "serde" in features,
        "The portable feature contract differs.")
    source = (FIXTURES / "target.rs").read_text()
    extra = dependencies(ROOT, ("chelis-dag-core", "chelis-numeric-core", "chelis-stack-services"), defaults=False)
    graphs = {}
    for label, defaults, selected in (("default", True, ()), ("no-default", False, ()),
            ("serde", False, ("serde",))):
        with support._package(CRATE, source, executable=False, features=selected,
                default_features=defaults, dependencies=extra) as directory:
            metadata = decode(support.run(("cargo", "metadata", "--format-version", "1",
                "--filter-platform", TARGET), cwd=directory).stdout)
            graphs[label] = inspect_graph(metadata)
            built = support.run(("cargo", "build", "--locked", "--lib", "--target", TARGET,
                "--message-format", "json"), cwd=directory)
            artifacts = {}
            for line in built.stdout.splitlines():
                record = decode(line)
                if record.get("reason") == "compiler-artifact" and record["target"]["name"] == "chelis_ir_core":
                    artifacts.update({name: digest(Path(name)) for name in record["filenames"]})
            require(artifacts and any(name.endswith(".rlib") for name in artifacts),
                "The target command lacks its exact IR library artifact.")
            graphs[label]["artifacts"] = artifacts
    evidence.save("target-graphs.json", graphs)
    artifacts = {label: graph["artifacts"] for label, graph in graphs.items()}
    evidence.save("compiled-artifacts.json", artifacts)
    evidence.complete("target-graphs")


def independent_consumers(evidence: Evidence) -> None:
    extra = dependencies(ROOT, ("chelis-checker-core", "chelis-dag-core", "chelis-deep",
        "chelis-numeric-core", "chelis-stack-services", "chelis-unord"), defaults=False)
    extra += 'critical-section = { version = "1", features = ["std"] }\n'
    results = {}
    prefixes = {"evaluation": "IR_EVAL_CONSUMER", "lowering": "IR_LOWER_CONSUMER",
        "host": "IR_HOST_CONSUMER", "ownership": "IR_OWNERSHIP_CONSUMER", "resources": "IR_RESOURCE_CONSUMER"}
    for name, prefix in prefixes.items():
        source = (FIXTURES / f"{name}.rs").read_text()
        result = support.consumer(CRATE, source, features=("lowering-trace",), dependencies=extra)
        results[name] = dag.snapshot_record(result.stdout, prefix)
        require(bool(results[name]), "An independent consumer report is empty.")
    source = (FIXTURES / "serde.rs").read_text()
    result = support.consumer(CRATE, source, features=("serde",),
        dependencies=extra + 'bincode = "=1.3.3"\nserde_json = "1"\n')
    results["serde"] = dag.snapshot_record(result.stdout, "IR_SERDE_CONSUMER")
    evidence.save("independent-consumers.json", results)
    evidence.complete("independent-consumers")


def rejected_consumer(evidence: Evidence, control: dict) -> dict:
    names = () if control.get("portable_only") else (
        "chelis-ir", "chelis-types", "chelis-checker-core", "chelis-dag-core")
    with support._package(CRATE, control["source"], executable=False,
            features=control.get("features", ()), default_features=False,
            dependencies=dependencies(ROOT, names) + 'bincode = "=1.3.3"\n') as directory:
        result = support._execute(("cargo", "check", "--lib", "--message-format", "json"), cwd=directory)
        require(result.returncode != 0, "An authority misuse consumer compiled.")
        diagnostic = checker.negative_diagnostics(result.stdout, directory, control["source"], control["expected"])
        evidence.accept_negative(control["expected"], diagnostic)
        return {"source": control["source"], "expected": control["expected"], "diagnostic": diagnostic}


def negative_controls(evidence: Evidence) -> None:
    controls = load(FIXTURES / "authority.json")
    require(bool(controls), "The native authority controls are empty.")
    evidence.save("authority-controls.json", {name: rejected_consumer(evidence, control)
        for name, control in controls.items()})
    graphs = {}
    for name, extra, fragments in (
            ("native-ir", checker.dependency("chelis-ir", ROOT), ["forbidden target dependency", "chelis-ir"]),
            ("native-stack", checker.dependency("chelis-syntax-host", ROOT), ["forbidden target dependency", "chelis-syntax-host"]),
            ("backend", checker.dependency("chelis-backend-c", ROOT), ["forbidden target dependency", "chelis-backend-c"]),
            ("std-feature", 'serde = { version = "1", features = ["std"] }\n', ["target std feature", "serde"])):
        with support._package(CRATE, "#![no_std]\n", executable=False, features=(),
                default_features=False, dependencies=extra) as directory:
            metadata = decode(support.run(("cargo", "metadata", "--format-version", "1",
                "--filter-platform", TARGET), cwd=directory).stdout)
            try:
                inspect_graph(metadata)
            except OracleFailure as error:
                require(all(fragment in str(error) for fragment in fragments), "A graph control failed at an unrelated boundary.")
                graphs[name] = {"metadata": metadata, "boundary": str(error)}
            else:
                raise OracleFailure("A forbidden dependency graph passed.")
    evidence.save("negative-graphs.json", graphs)
    support.run((sys.executable, str(ROOT / "scripts/test_ir_core_oracle.py")))
    evidence.complete("negative-controls")


def reconcile(previous: dict, current: dict, migrations: dict, *, manual_before=(), manual_after=(), external=()) -> dict:
    before = set()
    after = set()
    for lanes, identities in ((previous, before), (current, after)):
        for lane in lanes.values():
            require(lane["selected"] == lane["executed"] and bool(lane["selected"]), "A test lane lacks complete execution.")
            require(not identities.intersection(lane["executed"]), "A test identity executed in duplicate lanes.")
            identities.update(lane["executed"])
    require(migrations["baseline"] == BASELINE, "The migration map baseline differs.")
    renamed, retired = migrations["renamed"], migrations["retired"]
    universe = before | set(manual_before) | set(external)
    require(not set(renamed).intersection(retired) and (set(renamed) | set(retired)) <= universe,
        "An exact test migration lacks its immutable source identity.")
    require(len(set(renamed.values())) == len(renamed), "Exact test migrations share a destination.")
    mapping = {}
    for identity in before - set(retired):
        destination = renamed.get(identity, identity)
        require(destination in after and destination not in mapping.values(),
            f"An immutable test lacks its exact executed owner: {identity}")
        mapping[identity] = destination
    for identity, record in retired.items():
        if identity not in before:
            continue
        require(record.get("reason") and record.get("behavior_identities")
            and set(record["behavior_identities"]) <= after | set(external) and identity not in after,
            "A retired source assertion lacks executed behavioral evidence.")
    expected_manual = {renamed.get(identity, identity) for identity in manual_before}
    require(expected_manual == set(manual_after), "A manual or hardware obligation lost its exact identity.")
    new = migrations.get("new_tests", [])
    require(len(new) == len(set(new)) and set(new) <= after
        and not set(new).intersection(mapping.values()), "New behavior identities are absent or duplicate.")
    return mapping


def timing_key(identity: str) -> str:
    owner, name = identity.split("$", 1)
    package, binary = owner.split("::", 1)
    return (package if binary == package.replace("-", "_") else owner) + "::" + name


def validate_timing(previous: dict, current: dict, moves: dict, *, durations=False, retired=()) -> None:
    if durations:
        compare_exact({key: value for key, value in previous.items() if key != "targets"},
            {key: value for key, value in current.items() if key != "targets"}, "duration source metadata")
    before = previous["targets"] if durations else previous
    after = current["targets"] if durations else current
    require(set(moves) <= before.keys() and len(set(moves.values())) == len(moves),
        "A timing move lacks its exact source or shares a destination.")
    retired = set(retired)
    for identity, value in before.items():
        if identity in retired:
            require(identity not in after, "A retired source assertion retains its old timing identity.")
            continue
        destination = moves.get(identity, identity)
        require(destination in after and canonical(after[destination]) == canonical(value)
            and (destination == identity or identity not in after),
            f"A timing value or duration sample changed: {identity}")


def measured_work(output: str) -> list[str]:
    markers = ("eval-performance ", "declarations examined at ", "callers of a shared helper:",
        "independent definitions:", "free-variable walk work,", "refinement scope entries at ",
        "record hoist nodes at ", "site lookup entries,", "terminal-name comparisons,",
        "classification bytes at ", "lowering bytes at ")
    records = []
    for line in output.splitlines():
        if not line.startswith(markers) and not re.match(r"\d+ scalar helpers allocated ", line):
            continue
        # Preserve workload scales and deterministic work. Native tests retain allocation bounds.
        line = re.sub(r"\b(?:wall|streamed|dense)=[0-9.e+-]+(?:ns|µs|ms|s)", "<time>", line)
        line = re.sub(r"\b(allocated|peak_heap_delta)=\d+", r"\1=<observed>", line)
        line = re.sub(r"(scalar helpers allocated )\d+( bytes)", r"\1<observed>\2", line)
        if "bytes at " in line:
            line = re.sub(r": \[[0-9, ]+\]$", ": <observed>", line)
        records.append(line)
    return records


def cost_observations(previous: dict, current: dict, mapping: dict) -> dict:
    before = {key: value for lane in previous.values() for key, value in lane["junit"]["observations"].items()}
    after = {key: value for lane in current.values() for key, value in lane["junit"]["observations"].items()}
    records = {}
    for identity, observation in before.items():
        binary = identity.split("::", 1)[1].split("$", 1)[0]
        if not (binary.endswith("cost") or binary in ("callable_snapshot_allocations", PERFORMANCE)):
            continue
        destination = mapping.get(identity, identity)
        require(destination in after, "A full-host cost case lost its executed owner.")
        old_output = observation["stdout"] + observation["stderr"]
        new_output = after[destination]["stdout"] + after[destination]["stderr"]
        require(not old_output.strip() or bool(new_output.strip()), "A full-host cost observation lost its measured output.")
        compare_exact(measured_work(old_output), measured_work(new_output), "full-host workload scales and measured work")
        records[identity] = {"destination": destination, "baseline": observation, "candidate": after[destination]}
    require(records and any(PERFORMANCE + "$" in identity for identity in records)
        and any("callable_snapshot_allocations$" in identity for identity in records),
        "Full-host cost or evaluation-performance evidence is absent.")
    return records


def current_suites(previous: dict, evidence: Evidence) -> None:
    current = {"ordinary": suite(ROOT, ("chelis-ir", CRATE)),
        "performance-manual": suite(ROOT, ("chelis-ir", CRATE), selection="performance-manual"),
        "pointwise-release": suite(ROOT, ("chelis-ir", CRATE), selection="pointwise-release", release=True),
        "native-integration": suite(ROOT, NATIVE_PACKAGES)}
    migrations = load(FIXTURES / "test_identity_migration.json")
    inventory = inventory_cases(current["ordinary"]["inventory"])
    performed = set(current["performance-manual"]["executed"]) | set(current["pointwise-release"]["executed"])
    obligations = sorted(identity for identity, ignored in inventory.items() if ignored and identity not in performed)
    native_ignored = sorted(identity for identity, ignored
        in inventory_cases(current["native-integration"]["inventory"]).items() if ignored)
    require(native_ignored == previous["native_manual_obligations"],
        "A native manual or hardware obligation changed without an exact identity map.")
    external = migrations.get("external_identities", [])
    mapping = reconcile(previous["tests"], current, migrations,
        manual_before=previous["manual_obligations"], manual_after=obligations, external=external)
    evidence.save("current-tests.json", current)
    evidence.save("test-migrations.json", {"mapping": mapping, "retired": migrations["retired"],
        "manual_obligations": obligations, "new_tests": migrations.get("new_tests", [])})
    for package in ("chelis-ir", CRATE):
        support.run(("cargo", "test", "--locked", "-p", package, "--features", "lowering-trace", "--doc"))
    timing_migrations = migrations["timing_migrations"]
    timing = {}
    for path, old in previous["timing"].items():
        new = load(ROOT / path)
        moves = timing_migrations["test_timing" if path.endswith("test_timing_baseline.json")
            else "change_owned_duration"]
        validate_timing(old, new, moves, durations=path.endswith("ci-change-owned-durations.json"),
            retired=[timing_key(identity) for identity in migrations["retired"]])
        timing[path] = {"baseline": old, "candidate": new, "migrations": moves}
    evidence.save("cost-source-provenance.json", {name: {"baseline": identity,
        "candidate": digest(ROOT / name)} for name, identity in previous["cost_sources"].items()})
    evidence.save("timing-preservation.json", timing)
    evidence.save("cost-observations.json", cost_observations(previous["tests"], current, mapping))
    evidence.complete("complete-suites")
    evidence.complete("migration")
    evidence.complete("resources-caches-costs")


def prior_cores(evidence: Evidence) -> None:
    identities = {}
    for name in PRIOR:
        path = ROOT / "scripts" / name
        original = support.run(("git", "show", f"{BASELINE}:scripts/{name}")).stdout
        contracts = checker.validate_prior_contracts(name, path.read_text(), original)
        identities[name] = {"candidate": digest(path), "baseline_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "accepted_contracts": contracts}
        support.run((sys.executable, str(path)))
    require(dag.GENERATED_SOURCE_BASELINE_REVISION == "ec322a9568c09cf73a4855860227aa9d86fc2d32",
        "The accepted DAG generated-source reference differs.")
    evidence.save("prior-core-obligations.json", identities)
    evidence.complete("prior-cores")


def pipeline_authority(evidence: Evidence) -> None:
    def runner(command, *, cwd, env, check):
        if tuple(command[:3]) != ("cargo", "nextest", "run"):
            return support._execute(command, cwd=cwd, env=env)
        with tempfile.TemporaryDirectory(prefix="ir-pipeline-nextest-") as temporary:
            report = Path(temporary) / "results.xml"
            config = Path(temporary) / "nextest.toml"
            configuration = (Path(cwd) / ".config/nextest.toml").read_text()
            configuration += ('\n[profile.ir-pipeline-acceptance.junit]\n'
                f'path = {json.dumps(str(report))}\n'
                'store-success-output = true\nstore-failure-output = true\n')
            config.write_text(configuration)
            return support._execute((*command, "--config-file", str(config),
                "--profile", "ir-pipeline-acceptance"), cwd=cwd, env=env)
    pipeline.run_oracle(runner=runner, environment=support._environment())
    evidence.save("compiler-pipeline.json", {"script": digest(ROOT / "scripts/compiler_pipeline_oracle.py"),
        "commands": [list(command) for command in pipeline.FOCUSED_COMMANDS]})
    evidence.complete("compiler-pipeline")


def ci_ownership(evidence: Evidence) -> None:
    import ci_change_owned as ci
    config = ci.read_config(ROOT / ".config/ci-test-targets.toml")
    metadata = decode(support.run(("cargo", "metadata", "--locked", "--no-deps", "--format-version", "1")).stdout)
    tracked = set(support.run(("git", "ls-files", "-z")).stdout.split("\0")) - {""}
    ci.validate_config(config, metadata, tracked, lambda name: (ROOT / name).read_text())
    paths = ["scripts/ir_core_oracle.py", "scripts/test_ir_core_oracle.py"]
    paths.extend(str(path.relative_to(ROOT)) for path in FIXTURES.rglob("*") if path.is_file())
    owners = {}
    for path in paths:
        matches = [rule for rule in config.path_rules if rule.matches(path)]
        require(len(matches) == 1 and matches[0].disposition == "owner" and matches[0].owner is not None,
            f"An IR acceptance input lacks one exact CI owner: {path}")
        owners[path] = matches[0].owner.as_dict()
    evidence.save("ci-ownership.json", {"configuration_digest": ci.config_digest(config), "owners": owners})
    evidence.complete("ci-ownership")


def tool_identities(evidence: Evidence) -> None:
    env = support._environment()
    tools = {}
    for name in ("cargo", "cargo-nextest", "rustup", env.get("RUSTC", "rustc"),
            env.get("RUSTDOC", "rustdoc"), "git", "cc", "ar", "llvm-ar"):
        found = shutil.which(name, path=env.get("PATH"))
        require(found is not None, f"A mandatory executable is absent: {name}")
        path = Path(found).resolve()
        tools[name] = {"path": str(path), **digest(path)}
    for name in ("cargo", "rustc", "rustdoc"):
        resolved = support.run(("rustup", "which", name), env=env).stdout.strip()
        path = Path(resolved)
        require(path.is_absolute() and path.is_file(), "A resolved Rust tool identity is absent.")
        tools["resolved-" + name] = {"path": str(path), **digest(path)}
    evidence.save("tool-inputs.json", tools)
    for command in ((env.get("RUSTC", "rustc"), "-vV"), (env.get("RUSTDOC", "rustdoc"), "-vV"),
            ("cargo", "--version"), ("cargo", "nextest", "--version"), ("git", "--version"),
            ("cc", "--version"), ("rustup", "target", "list", "--installed")):
        support.run(command, env=env)


def run_oracle(evidence: Evidence) -> None:
    head = checker.require_clean_head()
    source = checker.source_identity()
    evidence.save("provenance.json", {"head": head, "source": source, "baseline": BASELINE,
        "accepted_store": STORE, "fixtures": fixture_identity()})
    previous = capture_baseline(evidence)
    target_graphs(evidence)
    independent_consumers(evidence)
    candidate = semantic_snapshot(ROOT, baseline=False)
    compare_exact(previous["snapshot"], candidate, "semantic, context, trace, diagnostic, and storage")
    evidence.save("candidate-snapshot.json", candidate)
    wire = dag.wire_snapshot(ROOT, baseline=False)
    compare_exact(previous["wire"], wire, "wire and transformation")
    evidence.save("candidate-wire.json", wire)
    evidence.complete("immutable-semantics")
    native = production(ROOT, evidence.directory / "candidate-native", evidence)
    compare_exact(previous["production"], native, "generated source, admission, native output, and toolchain")
    evidence.save("candidate-production.json", native)
    seals = runtime_seals(evidence.directory / "candidate-native")
    evidence.save("runtime-seals.json", {"baseline": previous["runtime_seals"], "candidate": seals})
    # Archive digests prove each staging receipt. Generated source parity excludes archive bytes.
    compare_exact({key: {field: value for field, value in seal.items() if field != "archive_sha256"}
        for key, seal in previous["runtime_seals"].items()},
        {key: {field: value for field, value in seal.items() if field != "archive_sha256"}
        for key, seal in seals.items()}, "runtime seal contracts and headers")
    evidence.complete("native-production")
    negative_controls(evidence)
    current_suites(previous, evidence)
    pipeline_authority(evidence)
    prior_cores(evidence)
    ci_ownership(evidence)
    require(checker.require_clean_head() == head and checker.source_identity() == source,
        "The candidate source changed during complete acceptance.")
    require(previous["fixtures"] == fixture_identity(), "The frozen comparison inputs changed during acceptance.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline", type=Path,
        help="Capture immutable source evidence only. This does not establish implementation acceptance.")
    options = parser.parse_args()
    if sys.platform not in ("linux", "darwin") or sys.version_info[:2] != (3, 11) or Path.cwd().resolve() != ROOT:
        print("Use managed Python 3.11 from the repository root on Linux or macOS.", file=sys.stderr)
        return 1
    parent = ARTIFACTS / sys.platform
    parent.mkdir(parents=True, exist_ok=True)
    directory = options.capture_baseline or Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    evidence = Evidence(directory.resolve(), "baseline-capture" if options.capture_baseline else "implementation")
    support._execute = evidence.execute
    try:
        tool_identities(evidence)
        if options.capture_baseline:
            capture_baseline(evidence)
        else:
            run_oracle(evidence)
        evidence.finish("passed")
    except (RuntimeError, OSError, ValueError, KeyError, TypeError, ET.ParseError, SyntaxError) as error:
        evidence.save("failure.json", {"error": str(error), "type": type(error).__name__})
        evidence.finish("failed")
        print(f"IR core oracle: FAIL: {error}", file=sys.stderr)
        print(f"Evidence: {evidence.directory}", file=sys.stderr)
        return 1
    finally:
        support._execute = evidence.original
    print(f"IR {'baseline capture' if options.capture_baseline else 'core oracle'}: PASS ({sys.platform})")
    print(f"Evidence: {evidence.directory}")
    if options.capture_baseline:
        print("Baseline capture does not establish implementation acceptance.")
    for name, value in LIMITS.items():
        print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
