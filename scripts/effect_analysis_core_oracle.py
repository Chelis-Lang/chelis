#!/usr/bin/env python3
"""Compare the portable effect owner with the immutable host implementation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import struct
import tomllib

import no_std_core_support as support
from no_std_core_support import ROOT, TARGET, OracleFailure

BASELINE = "949d8606bca101b825ea49b0df4719da99458e09"
BASELINE_REPOSITORY = "https://github.com/Chelis-Lang/chelis.git"
CRATE = "chelis-effects-core"
RETIRED_SOURCE_TEST = (
    "chelis-effects::chelis_effects$tests::"
    "declared_effect_reader_has_no_local_optional_carrier_adapter")
FIXTURES = ROOT / "scripts/fixtures/effect_analysis_core"
ARTIFACTS = ROOT / "target/agents/effect-analysis-core"
FORBIDDEN = (
    "chelis-effects", "chelis-types", "chelis-surf", "chelis-syntax-host",
    "chelis-compiler-api", "chelis-pipeline-core", "chelis-runtime", "chelis-prove",
    "chelis-stdlib", "chelis-dag-core", "chelis-ir", "stacker", "psm", "tokio",
    "chelis-cli", "chelis-reef", "chelis-backend-c", "chelis-backend-hip",
    "chelis-backend-metal", "chelis-runtime-bundle", "chelis-std-bundle",
    "rayon", "parking_lot", "getrandom",
)
BASE_LANES = {
    "effects": ("-p", "chelis-effects", "--all-targets"),
    "checked-authority": ("-p", "chelis-types",
        "--test", "rt800_checked_program_construction",
        "--test", "issue_2110_local_ascription_obligations"),
    "host-policy": ("-p", "chelis-compiler-api", "--lib", "-E",
        "test(effect) | test(resource) | test(show_inferred) | test(prepared)"),
    "target-before-artifact": ("-p", "chelis-compiler-api", "--features", "emission-observer",
        "--test", "emission_observer", "--test", "resource_target_admission"),
    "prepared-reuse": ("-p", "chelis-compiler-api", "--test", "issue_3144_prepared_eval_reuse"),
    "semantic-pipeline": ("-p", "chelis-compiler-api", "--test", "pipeline_contract"),
    "vocabulary-guard": ("-p", "chelis-cli", "--test", "closed_vocabulary_architecture"),
}
REQUIRED = {
    "tests::issue_1205_effect_clone_work_is_linear",
    "let_chain_effect_checking_holds_linear_memory",
    "tests::effect_annotation_reconstruction_preserves_type_context",
    "reported_row_carries_io_through_an_imported_wrapper",
    "reported_row_carries_io_through_a_transitive_local_helper",
    "reported_rows_stay_empty_for_pure_defs_across_the_boundary",
    "rejected_resource_region_produces_no_emission_observation",
    "resource_requirements_survive_imports_and_context_serialization",
    "a_prepared_program_derives_its_facts_once_for_all_roots",
    "a_program_prepared_in_context_derives_its_facts_once_for_all_roots",
    "effect_only_reannotation_preserves_every_checked_context",
    "callable_owner_effect_reannotation_rejects_swapped_equal_span_owners",
    "callable_owner_effect_reannotation_rejects_moved_real_owner",
    "layered::artifact_outcome_tests::effect_rows_are_reported_only_on_request",
}


def digest(path: Path) -> dict:
    body = path.read_bytes()
    return {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def fixture_identity() -> dict:
    return {name: digest(FIXTURES / name) for name in ("corpus.json", "schema.json", "host.rs", "diagnostics.rs")}


class Evidence:
    """Retain actual command output and bind every receipt to its files."""
    def __init__(self, directory: Path, *, scope: str = "implementation"):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.scope = scope
        self.receipts: list[dict] = []
        self.original = support._execute

    def execute(self, command, *, cwd=ROOT, env=None, live_stderr=False):
        if live_stderr:
            raise OracleFailure("live stderr cannot omit required raw evidence")
        index = len(self.receipts)
        result = self.original(command, cwd=cwd, env=env, live_stderr=False)
        files = {}
        for stream in ("stdout", "stderr"):
            path = self.directory / f"command-{index:04d}.{stream}"
            path.write_text(getattr(result, stream) or "", encoding="utf-8")
            files[stream] = {"path": path.name, **digest(path)}
        if Path(cwd).name.startswith(("chelis-no-std-consumer-", "effect-target-")):
            lock = Path(cwd) / "Cargo.lock"
            if lock.is_file():
                path = self.directory / f"command-{index:04d}.Cargo.lock"
                path.write_bytes(lock.read_bytes())
                files["consumer_lock"] = {"path": path.name, **digest(path)}
        self.receipts.append({"identity": index, "command": list(map(str, command)),
            "cwd": str(cwd), "exit": result.returncode, "files": files})
        return result

    def save(self, name: str, value: dict) -> None:
        (self.directory / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def finish(self, status: str) -> None:
        self.save("receipts.json", {"status": status, "platform": sys.platform,
            "machine": platform.machine(), "scope": self.scope,
            "invocation": sys.argv, "python": sys.executable,
            "python_version": platform.python_version(),
            "command_count": len(self.receipts), "receipts": self.receipts})
        if status == "passed":
            validate_receipts(self.directory)
        self.save("artifact-index.json", {
            path.name: digest(path) for path in sorted(self.directory.iterdir())
            if path.is_file() and path.name != "artifact-index.json"})


def validate_receipts(directory: Path) -> list[dict]:
    try:
        manifest = json.loads((directory / "receipts.json").read_text())
        receipts = manifest["receipts"]
        if (not receipts or type(manifest["command_count"]) is not int
                or manifest["command_count"] != len(receipts)
                or any(type(item["identity"]) is not int for item in receipts)
                or [item["identity"] for item in receipts] != list(range(len(receipts)))):
            raise OracleFailure("missing or duplicate command receipt identity")
        paths = set()
        for item in receipts:
            if not item["command"] or type(item["exit"]) is not int:
                raise OracleFailure("invalid command receipt")
            required = {"stdout", "stderr"}
            if Path(item["cwd"]).name.startswith(("chelis-no-std-consumer-", "effect-target-")):
                required.add("consumer_lock")
            if set(item["files"]) != required:
                raise OracleFailure("missing command output or independent consumer lock receipt")
            for record in item["files"].values():
                path = directory / record["path"]
                if record["path"] in paths:
                    raise OracleFailure("duplicate raw command receipt")
                paths.add(record["path"])
                if path.parent != directory or digest(path) != {
                    "bytes": record["bytes"], "sha256": record["sha256"]}:
                    raise OracleFailure("raw command receipt differs")
        return receipts
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise OracleFailure(f"missing or invalid execution receipt: {error}") from error


def selected_tests(metadata: dict) -> set[str]:
    selected = support._selected_test_names(metadata)
    identities = []
    for suite in metadata["rust-suites"].values():
        for name, test in suite["testcases"].items():
            if test["filter-match"]["status"] == "matches":
                identities.append(f"{suite['package-name']}::{suite['binary-name']}${name}")
    if len(identities) != len(set(identities)):
        raise OracleFailure("duplicate selected execution identity")
    return selected


def check_execution(stdout: str, selected: set[str]) -> set[str]:
    completed = Counter()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if record["type"] != "test" or record["event"] == "started":
                continue
            name = record["name"]
            if record["event"] == "ignored" and name not in selected:
                continue
            if record["event"] != "ok":
                raise OracleFailure(f"required test did not pass: {name}")
            completed[name] += 1
        except (KeyError, TypeError, ValueError) as error:
            raise OracleFailure(f"invalid test execution record: {error}") from error
    if set(completed) != selected or any(count != 1 for count in completed.values()):
        raise OracleFailure("missing, duplicate, or unexpected execution identity")
    return set(completed)


def tests(arguments: tuple[str, ...], *, cwd: Path = ROOT) -> dict:
    environment = support._environment()
    environment["NEXTEST_EXPERIMENTAL_LIBTEST_JSON"] = "1"
    inventory = support.run(("cargo", "nextest", "list", "--locked", *arguments,
        "--run-ignored", "all", "--message-format", "json"), cwd=cwd, env=environment)
    selected = selected_tests(json.loads(inventory.stdout))
    result = support._execute(("cargo", "nextest", "run", "--locked", *arguments, "--no-fail-fast",
        "--message-format", "libtest-json-plus"), cwd=cwd, env=environment)
    if result.returncode != 0:
        raise OracleFailure("required nextest execution failed")
    executed = check_execution(result.stdout, selected)
    return {"selected": sorted(selected), "executed": sorted(executed)}


def checkout_baseline(destination: Path) -> Path:
    destination = destination.resolve()
    support.run(("git", "clone", "--quiet", "--shared", "--no-checkout", str(ROOT), str(destination)))
    present = support._execute(("git", "cat-file", "-e", f"{BASELINE}^{{commit}}"), cwd=destination)
    if present.returncode != 0:
        support.run(("git", "fetch", "--quiet", "--no-tags", "--depth=1",
                     BASELINE_REPOSITORY, BASELINE), cwd=destination)
    support.run(("git", "checkout", "--quiet", "--detach", BASELINE), cwd=destination)
    identity = support.run(("git", "rev-parse", "HEAD"), cwd=destination).stdout.strip()
    if identity != BASELINE:
        raise OracleFailure("immutable source identity differs")
    return destination


def dependency(crate: str, workspace: Path = ROOT, *, defaults: bool = False) -> str:
    return f'{crate} = {{ path = {json.dumps(str(workspace / "crates" / crate))}, default-features = {str(defaults).lower()} }}\n'


def host_snapshot(workspace: Path) -> dict:
    corpus = (FIXTURES / "corpus.json").read_text()
    source = (FIXTURES / "host.rs").read_text()
    if '"####' in corpus:
        raise OracleFailure("corpus contains its Rust delimiter")
    source += '\nconst CORPUS_JSON: &str = r####"' + corpus + '"####;\n'
    dependencies = 'bincode = "=1.3.3"\nserde_json = { version = "1", features = ["float_roundtrip"] }\n'
    for crate in ("chelis-types", "chelis-deep", "chelis-surf", "chelis-syntax-host"):
        dependencies += dependency(crate, workspace, defaults=True)
    result = support.consumer(workspace / "crates/chelis-effects", source,
        default_features=True, dependencies=dependencies)
    records = [line[len("EFFECT_HOST_SNAPSHOT "):] for line in result.stdout.splitlines()
        if line.startswith("EFFECT_HOST_SNAPSHOT ")]
    if len(records) != 1:
        raise OracleFailure("missing or duplicate semantic snapshot")
    snapshot = json.loads(records[0])
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot: dict) -> None:
    schema = json.loads((FIXTURES / "schema.json").read_text())
    corpus = json.loads((FIXTURES / "corpus.json").read_text())
    if set(snapshot) != set(schema["snapshot_fields"]):
        raise OracleFailure("semantic snapshot fields differ")
    if set(snapshot["cases"]) != {entry[0] for entry in corpus["cases"]}:
        raise OracleFailure("semantic case inventory differs")
    if set(snapshot["extensions"]) != {entry[0] for entry in corpus["extensions"]}:
        raise OracleFailure("context case inventory differs")
    def checked(value):
        if (not isinstance(value, dict) or set(value) != set(schema["checked_fields"])
                or not value["bytes"] or not value["payload"] or not value["tree"]):
            raise OracleFailure("checked owner or context evidence differs")
        if not 0 <= value["typed_nodes"] <= value["total_nodes"]:
            raise OracleFailure("checked statistics differ")
    def result(value, accepted):
        if value.get("status") != ("accepted" if accepted else "rejected"):
            raise OracleFailure("semantic disposition differs")
        if accepted:
            if set(value) != {"status", "input", "rows", "output", "targets"}:
                raise OracleFailure("successful effect result fields differ")
            checked(value["input"])
            checked(value["output"])
            if not isinstance(value["rows"], dict) or set(value["targets"]) != {"c", "hip", "metal"}:
                raise OracleFailure("rows or target outcomes are absent")
            for target in value["targets"].values():
                if target.get("status") == "accepted":
                    if set(target) != {"status"}:
                        raise OracleFailure("successful target fields differ")
                elif (set(target) != {"status", "errors"} or target["status"] != "rejected"
                        or not target["errors"]):
                    raise OracleFailure("target refusal evidence differs")
        elif value.get("stage") == "effects":
            if set(value) != {"status", "stage", "input", "rows", "errors"}:
                raise OracleFailure("effect rejection fields differ")
            checked(value["input"])
            if not value["errors"] or any(
                    set(item) != set(schema["effect_error_fields"]) or not item["message"]
                    for item in value["errors"]):
                raise OracleFailure("effect diagnostic fields differ")
        elif value.get("stage") == "types":
            report = value.get("report", {})
            if (set(value) != {"status", "stage", "report"}
                    or set(report) != {"typed_nodes", "total_nodes", "errors"}
                    or not report["errors"] or any(
                        set(item) != set(schema["type_error_fields"]) for item in report["errors"])):
                raise OracleFailure("type diagnostic fields differ")
        elif value.get("stage") != "admission" or set(value) != {"status", "stage", "error"} or not value["error"]:
            raise OracleFailure("admission diagnostic fields differ")
    checked(snapshot["library"])
    if set(snapshot["context"]) != {"bytes", "payload"} or not snapshot["context"]["bytes"]:
        raise OracleFailure("checked context evidence is absent")
    for name, _, accepted, _ in corpus["cases"]:
        result(snapshot["cases"][name], accepted)
    for name, accepted, _ in corpus["extensions"]:
        pair = snapshot["extensions"][name]
        if set(pair) != {"contextual", "monolithic"}:
            raise OracleFailure("context path evidence differs")
        result(pair["contextual"], accepted)
        result(pair["monolithic"], accepted)


def canonical_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def compare_snapshots(previous: dict, current: dict) -> None:
    validate_snapshot(previous)
    validate_snapshot(current)
    if canonical_json(previous) != canonical_json(current):
        raise OracleFailure("rows, annotations, owners, contexts, diagnostics, or host outcomes differ")


def diagnostic_snapshot(workspace: Path) -> dict:
    source = (FIXTURES / "diagnostics.rs").read_text()
    source += '\nconst CORPUS_JSON: &str = r####"' + (FIXTURES / "corpus.json").read_text() + '"####;\n'
    owner = CRATE if workspace == ROOT else "chelis-effects"
    dependencies = (
        f'effect-data = {{ package = {json.dumps(owner)}, '
        f'path = {json.dumps(str(workspace / "crates" / owner))} }}\n'
        'serde_json = { version = "1", features = ["float_roundtrip"] }\n')
    result = support.consumer(workspace / "crates/chelis-compiler-api", source,
        default_features=True, dependencies=dependencies)
    records = [line.removeprefix("EFFECT_DIAGNOSTIC_SNAPSHOT ") for line in result.stdout.splitlines()
        if line.startswith("EFFECT_DIAGNOSTIC_SNAPSHOT ")]
    if len(records) != 1:
        raise OracleFailure("missing or duplicate host diagnostic snapshot")
    snapshot = json.loads(records[0])
    validate_diagnostics(snapshot)
    return snapshot


def validate_diagnostics(snapshot: dict) -> None:
    corpus = json.loads((FIXTURES / "corpus.json").read_text())
    if (set(snapshot) != {"conversions", "outcomes"} or len(snapshot["conversions"]) != 16
            or set(snapshot["outcomes"]) != {entry[0] for entry in corpus["cases"]}):
        raise OracleFailure("host diagnostic snapshot inventory differs")
    identities = []
    fields = {"wire", "kind", "message", "severity_bits", "expected", "got",
        "suggestions", "span", "deep_path", "span_id", "unsupported_identity"}
    for record in snapshot["conversions"]:
        if set(record) != {"kind", "severity_bits", "diagnostic"} or set(record["diagnostic"]) != fields:
            raise OracleFailure("host diagnostic fields differ")
        identities.append((record["kind"], record["severity_bits"]))
    if len(identities) != len(set(identities)):
        raise OracleFailure("duplicate host diagnostic conversion")
    expected = {(kind, struct.unpack(">Q", struct.pack(">d", severity))[0])
        for kind in ("UnhandledEffect", "InvalidHandler", "BuildTargetMismatch", "TypeTotality")
        for severity in (0.0, 0.25, 0.8, 1.0)}
    if set(identities) != expected:
        raise OracleFailure("host diagnostic kinds or supplied severities differ")
    for outcome in snapshot["outcomes"].values():
        expected_fields = {
            "reported": {"status", "score", "components", "typed_nodes", "untyped_nodes",
                "total_nodes", "unresolved_names", "inferred_signatures", "diagnostics"},
            "rejected": {"status", "stage", "diagnostics", "transcript"},
        }
        if outcome.get("status") not in expected_fields or set(outcome) != expected_fields[outcome["status"]]:
            raise OracleFailure("compiler outcome fields differ")
        if "diagnostics" not in outcome or any(set(item) != fields for item in outcome["diagnostics"]):
            raise OracleFailure("compiler diagnostic details differ")


def capture_baseline(evidence: Evidence) -> dict:
    with tempfile.TemporaryDirectory(prefix="effect-base-") as temporary:
        workspace = checkout_baseline(Path(temporary) / "source")
        result = {"revision": BASELINE, "fixtures": fixture_identity(),
            "source": {name: digest(workspace / name) for name in (
                "Cargo.lock", "crates/chelis-effects/src/lib.rs", "crates/chelis-types/src/infer/checked.rs")},
            "timing": {key: value for key, value in json.loads(
                (workspace / "scripts/test_timing_baseline.json").read_text()).items()
                if key.startswith("chelis-effects::")},
            "snapshot": host_snapshot(workspace),
            "diagnostics": diagnostic_snapshot(workspace),
            "tests": {lane: tests(arguments, cwd=workspace) for lane, arguments in BASE_LANES.items()}}
        names = {identity.split("$", 1)[1] for lane in result["tests"].values() for identity in lane["executed"]}
        if not REQUIRED <= names:
            raise OracleFailure(f"required baseline tests are absent: {sorted(REQUIRED - names)}")
        evidence.save("baseline.json", result)
        return result


def portable_consumers(evidence: Evidence) -> None:
    manifest = tomllib.loads((ROOT / "crates" / CRATE / "Cargo.toml").read_text())
    features = manifest.get("features", {})
    if features.get("default", []) or "std" in features:
        raise OracleFailure("core defaults or std feature violate the portable contract")
    source = (FIXTURES / "core.rs").read_text()
    dependencies = "".join(dependency(crate) for crate in (
        "chelis-deep", "chelis-type-core", "chelis-stack-services"))
    graphs = {}
    for name, defaults, features in (
        ("default", True, ()), ("no-default", False, ()), ("serde", False, ("serde",)),
    ):
        graphs[name] = sorted(support.portable(CRATE, "#![no_std]\n" + source,
            default_features=defaults, features=features, forbidden=FORBIDDEN,
            dependencies=dependencies))
    evidence.save("target-graphs.json", graphs)
    support.consumer(CRATE, source + "\nfn main() { exercise(); }\n",
        dependencies=dependencies)
    with tempfile.TemporaryDirectory(prefix="effect-target-") as temporary:
        directory = Path(temporary)
        (directory / "src").mkdir()
        (directory / "Cargo.toml").write_text(
            '[package]\nname = "effect-target-consumer"\nversion = "0.0.0"\nedition = "2024"\n'
            '[workspace]\n[profile.dev]\npanic = "abort"\n[dependencies]\n'
            + dependency(CRATE) + dependencies)
        (directory / "src/main.rs").write_text("#![no_std]\n#![no_main]\n"
            + source + "\n" + (FIXTURES / "target.rs").read_text())
        support.run(("cargo", "rustc", "--bin", "effect-target-consumer",
            "--target", TARGET, "--", "--emit=obj"), cwd=directory)


def forbidden_graph_controls() -> None:
    for name, dependencies, expected in (
        ("hosted inference", dependency("chelis-types", defaults=True), "forbidden target dependency"),
        ("native stack", dependency("chelis-syntax-host", defaults=True), "forbidden target dependency"),
        ("DAG", dependency("chelis-dag-core"), "forbidden target dependency"),
        ("dependency std", 'serde = { version = "1", default-features = false, features = ["std"] }\n',
            "target std feature"),
    ):
        with support._package(CRATE, "#![no_std]\n", executable=False,
                features=(), default_features=False, dependencies=dependencies) as directory:
            result = support.run(("cargo", "metadata", "--format-version", "1",
                "--filter-platform", TARGET), cwd=directory)
            try:
                support._normal_packages(json.loads(result.stdout), FORBIDDEN)
            except OracleFailure as error:
                if expected not in str(error):
                    raise OracleFailure(f"{name} control failed for the wrong reason: {error}") from error
            else:
                raise OracleFailure(f"{name} dependency control did not reject")


def serde_controls() -> None:
    source = r'''
use chelis_type_core::{Effect, EffectSet};
fn main() {
    let mut row = EffectSet::new();
    row.insert(Effect::Io);
    let bytes = bincode::serialize(&row).unwrap();
    let restored: EffectSet = bincode::deserialize(&bytes).unwrap();
    assert_eq!(row, restored);
}
'''
    dependencies = dependency("chelis-type-core") + 'bincode = "=1.3.3"\n'
    support.consumer(CRATE, source, features=("serde",), dependencies=dependencies)
    support.compile_rejected(CRATE,
        'pub fn disabled() { let _ = bincode::serialize(&chelis_type_core::Effect::Io); }\n',
        dependencies=dependencies, expected=("Serialize", "Effect"))


def authority_controls() -> None:
    dependencies = dependency("chelis-types", defaults=True) + dependency("chelis-deep")
    support.compile_rejected(CRATE, r'''
pub fn forge(expressions: &[chelis_deep::Expr]) -> chelis_types::CheckedProgram {
    chelis_effects_core::analyze_effects(expressions, None).unwrap()
}
''', dependencies=dependencies, expected=("mismatched types", "CheckedProgram"))
    support.compile_rejected(CRATE,
        'use chelis_types::errors::ErrorWitness;\n'
        'pub fn forge() -> ErrorWitness { ErrorWitness(()) }\n',
        dependencies=dependencies, expected=("E0423", "private fields"))
    support.compile_rejected("chelis-types",
        'use chelis_types::DiagnosticCheckpoint;\n'
        'pub fn forge() { let _ = DiagnosticCheckpoint(0usize); }\n',
        features=("checkpoint-compile-probe",), expected=("E0423", "private fields"))
    for name in ("EffectError", "EffectErrorKind", "DefEffectRows"):
        support.compile_rejected("chelis-effects", f"use chelis_effects::{name};\n",
            default_features=True, expected=("E0603", name, "private"))
    for name in ("CheckedProgram", "ErrorWitness", "DiagnosticCheckpoint", "CheckedLibrary", "LoweredLibrary"):
        support.compile_rejected(CRATE, f"use chelis_effects_core::{name};\n",
            expected=("E0432", name))
    import check_pipeline_core_compile_fail as pipeline
    result = support._execute(pipeline.COMMAND)
    if result.returncode == 0:
        raise OracleFailure("pipeline authority misuse compiled")
    diagnostics = result.stdout + result.stderr
    if any(fragment not in diagnostics for fragment in pipeline.REQUIRED_DIAGNOSTICS):
        raise OracleFailure("pipeline authority compiler diagnostics differ")
    support.run((sys.executable, "scripts/check_checkpoint_compile_fail.py"))


def migrated_identity(identity: str, moves: dict) -> str:
    before = f"{moves['old_package']}::{moves['old_binary']}$"
    if identity.startswith(before) and identity[len(before):] in moves["same_test_names"]:
        return f"{moves['new_package']}::{moves['new_binary']}$" + identity[len(before):]
    return identity


def compare_inventories(previous: dict, current: dict, moves: dict) -> None:
    retirements = moves["retired_tests"]
    if set(retirements) - {RETIRED_SOURCE_TEST} or any(not reason for reason in retirements.values()):
        raise OracleFailure("unapproved test retirement")
    baseline = {identity for lane in previous.values() for identity in lane["executed"]}
    if not set(retirements) <= baseline:
        raise OracleFailure("retirement lacks original baseline execution")
    required = set()
    for lane in previous.values():
        if lane["selected"] != lane["executed"]:
            raise OracleFailure("baseline test receipt is incomplete")
        required.update(migrated_identity(identity, moves)
            for identity in lane["executed"] if identity not in retirements)
    actual = []
    for lane in current.values():
        if lane["selected"] != lane["executed"]:
            raise OracleFailure("current test receipt is incomplete")
        actual.extend(lane["executed"])
    if len(actual) != len(set(actual)):
        raise OracleFailure("duplicate current execution obligation")
    retired_names = {identity.split("$", 1)[1] for identity in retirements}
    if any(identity.split("$", 1)[1] in retired_names for identity in actual):
        raise OracleFailure("retired source-wording test remains executable")
    if not required <= set(actual):
        raise OracleFailure(f"frozen required identities are absent: {sorted(required - set(actual))}")
    old_prefix = f"{moves['old_package']}::{moves['old_binary']}$"
    for name in moves["same_test_names"]:
        if old_prefix + name in actual:
            raise OracleFailure(f"obsolete test identity remains: {name}")
        if migrated_identity(old_prefix + name, moves) not in actual:
            raise OracleFailure(f"moved test did not execute: {name}")


def check_timing(previous: dict, current: dict, moves: dict) -> None:
    for key, value in previous.items():
        name = key.removeprefix(moves["old_package"] + "::")
        moved = name in moves["same_test_names"]
        target = moves["new_package"] + "::" + name if moved else key
        if target not in current or current[target] != value or (moved and key in current):
            raise OracleFailure(f"timing ownership or measured baseline differs: {key}")


def host_controls(previous: dict, evidence: Evidence) -> None:
    current = {}
    for lane, arguments in BASE_LANES.items():
        current[lane] = tests(arguments)
    current["portable-unit"] = tests(("-p", CRATE, "--all-targets",
        "--features", "serde,test-instrumentation"))
    profile_identity = f"{CRATE}::chelis_effects_core$tests::work_profiles_are_explicit_and_isolated"
    if profile_identity not in current["portable-unit"]["executed"]:
        raise OracleFailure("explicit work-profile isolation test did not execute")
    moves = json.loads((FIXTURES / "moved_tests.json").read_text())
    compare_inventories(previous["tests"], current, moves)
    check_timing(previous["timing"], json.loads((ROOT / "scripts/test_timing_baseline.json").read_text()), moves)
    evidence.save("current-tests.json", current)
    evidence.save("test-migrations.json", {
        "identities": {identity: migrated_identity(identity, moves)
            for lane in previous["tests"].values() for identity in lane["executed"]
            if identity not in moves["retired_tests"]},
        "explicit_retirements": moves["retired_tests"]})
    support.run(("cargo", "check", "--locked", "-p", "chelis-cli", "-p", "chelis-compiler-api",
        "-p", "chelis-pipeline-core", "-p", "chelis-ir", "-p", "chelis-prove", "--all-targets"))


def dormant_semantic_control() -> None:
    support.consumer("chelis-compiler-api", r'''
use chelis_compiler_api::compiler::compile_for_execution_with_observer;
use chelis_compiler_api::schema::{CompileRequest, CompileTarget, SourceKind};
fn main() {
    let source = "def dormant(x: i64) -> i64 ! {} = debug(x)\n\
        def main(x: tensor[2,f32]) -> tensor[2,f32] = mul(x,x)\n";
    let mut observations = 0;
    let result = compile_for_execution_with_observer(CompileRequest {
        source_kind: SourceKind::Surf, source: source.into(),
        target: CompileTarget::C, entry_name: Some("main".into()),
    }, &mut |_| observations += 1);
    let error = result.map(|_| ()).expect_err("a dormant semantic defect must reject");
    assert_eq!(error.stage, "effects");
    assert!(!error.errors.is_empty());
    assert_eq!(observations, 0);
}
''', default_features=True, features=("emission-observer",))


def policy_controls() -> None:
    environment = support._environment()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(ROOT), environment.get("PYTHONPATH")) if value
    )
    for script in (
        "test_effect_analysis_core_oracle.py", "test_gate.py", "test_ci_change_owned.py",
        "test_check_configuration_closure.py", "test_pipeline_core_dependency_guard.py",
        "test_pipeline_core_documentation_guard.py", "test_loud_unsupported_phase2_oracle.py",
        "loud_unsupported_phase2_oracle.py",
    ):
        support.run((sys.executable, "scripts/" + script), env=environment)
    support.run((sys.executable, "scripts/pipeline_core_dependency_guard.py"), env=environment)
    support.run((sys.executable, "scripts/pipeline_core_documentation_guard.py"), env=environment)


def source_identity() -> dict:
    paths = {ROOT / name for name in ("Cargo.toml", "Cargo.lock", "Cargo.nix", "rust-toolchain.toml")}
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update(path for path in FIXTURES.iterdir() if path.is_file())
    paths.update((ROOT / ".github/workflows").glob("*.yml"))
    for crate in (ROOT / "crates").iterdir():
        if not crate.is_dir():
            continue
        if (crate / "Cargo.toml").is_file():
            paths.add(crate / "Cargo.toml")
        paths.update(crate.glob("*.rs"))
        paths.update((crate / "tests").rglob("Cargo.toml"))
        paths.update((crate / "tests").rglob("Cargo.lock"))
        for source in ("src", "tests"):
            paths.update((crate / source).rglob("*.rs"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths) if path.is_file()}


def compiled_artifacts() -> dict:
    target = Path(support._environment()["CARGO_TARGET_DIR"])
    if not target.is_absolute():
        target = ROOT / target
    dependencies = target / TARGET / "debug/deps"
    libraries = list(dependencies.glob("libchelis_effects_core-*.rlib"))
    objects = list(dependencies.glob("effect_target_consumer-*.o"))
    if not libraries or not objects:
        raise OracleFailure("target library or allocator consumer artifact is absent")
    paths = libraries + objects
    for name in ("chelis-no-std-consumer", "effect-target-consumer"):
        for directory in (target / "debug", target / TARGET / "debug"):
            if (directory / name).is_file():
                paths.append(directory / name)
    return {str(path): digest(path) for path in sorted(set(paths))}


def require_clean_head() -> None:
    result = support.run(("git", "status", "--porcelain=v1", "--untracked-files=normal"))
    if result.stdout.strip():
        raise OracleFailure("complete acceptance needs a clean, committed implementation head")


def run_oracle(evidence: Evidence) -> None:
    head = support.run(("git", "rev-parse", "HEAD")).stdout.strip()
    source = source_identity()
    evidence.save("provenance.json", {"head": head, "baseline": BASELINE,
        "source": source, "fixtures": {path.name: digest(path)
            for path in sorted(FIXTURES.iterdir()) if path.is_file()},
        "stack": {"effect_host_bytes": 64 * 1024 * 1024,
            "realizability_cost_bytes": 256 * 1024 * 1024},
        "environment": {key: os.environ.get(key) for key in (
            "RUSTUP_TOOLCHAIN", "RUSTC", "CARGO_BUILD_JOBS", "CARGO_PROFILE_DEV_DEBUG",
            "CARGO_PROFILE_TEST_DEBUG", "LLVM_SYS_231_PREFIX", "SDKROOT", "CARGO_TARGET_DIR")}})
    require_clean_head()
    previous = capture_baseline(evidence)
    portable_consumers(evidence)
    forbidden_graph_controls()
    serde_controls()
    authority_controls()
    current = host_snapshot(ROOT)
    evidence.save("current-snapshot.json", current)
    compare_snapshots(previous["snapshot"], current)
    diagnostics = diagnostic_snapshot(ROOT)
    evidence.save("current-diagnostics.json", diagnostics)
    if canonical_json(previous["diagnostics"]) != canonical_json(diagnostics):
        raise OracleFailure("host diagnostic conversion, details, or compiler outcome differs")
    host_controls(previous, evidence)
    dormant_semantic_control()
    policy_controls()
    evidence.save("compiled-artifacts.json", compiled_artifacts())
    if support.run(("git", "rev-parse", "HEAD")).stdout.strip() != head:
        raise OracleFailure("candidate head changed during acceptance")
    if source_identity() != source:
        raise OracleFailure("candidate source or manifest changed during acceptance")
    if fixture_identity() != previous["fixtures"]:
        raise OracleFailure("immutable corpus changed during acceptance")
    require_clean_head()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-baseline", type=Path)
    options = parser.parse_args()
    if sys.platform not in ("linux", "darwin"):
        print("effect core oracle requires Linux or macOS", file=sys.stderr)
        return 1
    directory = options.capture_baseline
    if directory is None:
        parent = ARTIFACTS / sys.platform
        parent.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
    evidence = Evidence(directory.resolve(),
        scope="baseline-capture" if options.capture_baseline else "implementation")
    support._execute = evidence.execute
    try:
        support.run(("rustc", "-vV"))
        support.run(("cargo", "--version"))
        support.run(("cargo", "nextest", "--version"))
        if options.capture_baseline:
            capture_baseline(evidence)
        else:
            run_oracle(evidence)
        evidence.finish("passed")
    except (OracleFailure, OSError, ValueError, KeyError, TypeError) as error:
        evidence.finish("failed")
        print(f"effect analysis core oracle: FAIL: {error}", file=sys.stderr)
        return 1
    finally:
        support._execute = evidence.original
    if options.capture_baseline:
        print(f"effect analysis baseline capture: PASS ({sys.platform})")
        print("Baseline capture does not establish implementation acceptance.")
    else:
        print(f"effect analysis core oracle: PASS ({sys.platform})")
    print(f"Evidence: {evidence.directory}")
    print("Target compilation does not prove device execution or unrestricted stack capacity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
