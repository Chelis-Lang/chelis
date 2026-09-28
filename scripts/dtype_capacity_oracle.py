#!/usr/bin/env python3
"""Issue the structural #1288 receipt consumed by the pre-Phase-4C framework.

This adapter has no saved-result input. It builds each exact current test
artifact and checks libtest's lifecycle JSON for every selected control. It
retains each nested process's framework output beside the framework receipt.
The binding selection deliberately requires the final zero-legacy gate; until
that gate exists and passes, this command fails without producing a receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from dtype_pre_phase4c_oracle import SourceIdentity, source_identity


PASS_LINE = "DTYPE CAPACITY ORACLE: PASS"
REPO_ROOT = Path(__file__).resolve().parents[1]


class CapacityOracleError(RuntimeError):
    """The capacity receipt is unavailable or does not describe actual execution."""


@dataclass(frozen=True)
class Group:
    name: str
    package: str
    binary: str
    selected: tuple[str, ...]
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    mutation: tuple[str, ...]


GROUPS = (
    Group(
        "primary",
        "chelis-cli",
        "capacity_census_tripwire",
        (
            "capacity_census_matches_public_surface",
            "migrated_primary_rows_have_exact_final_authority_and_no_transition_disposition",
            "primary_baseline_has_zero_legacy_rows",
            "duplicate_primary_descriptors_fail_before_map_collapse",
            "primary_metadata_and_manifest_mutations_fail",
            "every_primary_final_registration_binds_kind_identity_and_flags",
            "regeneration_preserves_final_authority_but_cannot_bless_an_unclassified_row",
            "maintainer_override_is_not_a_final_authority_class",
            "unflagged_row_with_issue_citation_still_requires_final_authority",
        ),
        ("capacity_census_matches_public_surface",),
        (
            "migrated_primary_rows_have_exact_final_authority_and_no_transition_disposition",
            "primary_baseline_has_zero_legacy_rows",
            "duplicate_primary_descriptors_fail_before_map_collapse",
            "every_primary_final_registration_binds_kind_identity_and_flags",
            "maintainer_override_is_not_a_final_authority_class",
            "unflagged_row_with_issue_citation_still_requires_final_authority",
        ),
        (
            "primary_metadata_and_manifest_mutations_fail",
            "regeneration_preserves_final_authority_but_cannot_bless_an_unclassified_row",
        ),
    ),
    Group(
        "stdlib",
        "chelis-cli",
        "capacity_census_tripwire",
        (
            "final_stdlib_registrations_are_exact_and_bijective_with_normative_registries",
            "an_exported_stdlib_def_without_a_signature_fails_loudly",
            "a_new_stdlib_numeric_def_requires_semantic_registration",
            "stdlib_closure_tests::import_failures_and_ambiguity_do_not_erase_numeric_capacity",
            "stdlib_closure_tests::unresolved_nominals_and_wrong_nominal_arguments_fail_closed",
            "stdlib_closure_tests::nested_alias_fields_and_precision_variables_are_capacity",
            "stdlib_closure_tests::forward_and_recursive_aliases_reach_a_finite_numeric_fixed_point",
            "stdlib_closure_tests::generic_summaries_substitute_used_parameter_positions",
            "stdlib_closure_tests::alias_body_changes_capacity_without_rewriting_the_authored_identity",
            "stdlib_closure_tests::stdlib_closure_preserves_all_final_registered_source_identities",
            "stdlib_closure_tests::symbolic_tensor_precision_requires_an_adt_operation_row",
            "stdlib_closure_tests::imported_symbolic_precision_reaches_adt_rows_without_tainting_boolean_instances",
        ),
        (
            "final_stdlib_registrations_are_exact_and_bijective_with_normative_registries",
            "stdlib_closure_tests::nested_alias_fields_and_precision_variables_are_capacity",
            "stdlib_closure_tests::forward_and_recursive_aliases_reach_a_finite_numeric_fixed_point",
            "stdlib_closure_tests::generic_summaries_substitute_used_parameter_positions",
            "stdlib_closure_tests::stdlib_closure_preserves_all_final_registered_source_identities",
            "stdlib_closure_tests::imported_symbolic_precision_reaches_adt_rows_without_tainting_boolean_instances",
        ),
        (
            "an_exported_stdlib_def_without_a_signature_fails_loudly",
            "stdlib_closure_tests::import_failures_and_ambiguity_do_not_erase_numeric_capacity",
            "stdlib_closure_tests::unresolved_nominals_and_wrong_nominal_arguments_fail_closed",
        ),
        (
            "a_new_stdlib_numeric_def_requires_semantic_registration",
            "stdlib_closure_tests::alias_body_changes_capacity_without_rewriting_the_authored_identity",
            "stdlib_closure_tests::symbolic_tensor_precision_requires_an_adt_operation_row",
        ),
    ),
    Group(
        "wire",
        "chelis-compiler-api",
        "capacity_census_wire",
        (
            "wire_schema_numeric_fields_match_the_reviewed_baseline",
            "adding_or_removing_a_public_serialized_f64_field_changes_the_census",
            "verified_wire_authority_cannot_be_replaced_by_a_descriptor_or_baseline",
        ),
        ("wire_schema_numeric_fields_match_the_reviewed_baseline",),
        ("verified_wire_authority_cannot_be_replaced_by_a_descriptor_or_baseline",),
        ("adding_or_removing_a_public_serialized_f64_field_changes_the_census",),
    ),
    Group(
        "bindings",
        "chelis-python",
        "capacity_census_bindings",
        (
            "binding_census_has_zero_legacy_rows",
            "registered_pyfunctions_match_the_reviewed_rustdoc_signatures",
            "final_binding_rows_cannot_regain_legacy_admission",
            "copied_missing_and_duplicate_binding_registrations_fail",
            "final_bindings_require_successful_current_exposure",
        ),
        ("binding_census_has_zero_legacy_rows",),
        ("final_binding_rows_cannot_regain_legacy_admission",),
        (
            "copied_missing_and_duplicate_binding_registrations_fail",
            "final_bindings_require_successful_current_exposure",
        ),
    ),
)


def validate_groups(groups: Sequence[Group] = GROUPS) -> None:
    if [group.name for group in groups] != ["primary", "stdlib", "wire", "bindings"]:
        raise CapacityOracleError("capacity receipt requires the exact four surface groups")
    for group in groups:
        selected = set(group.selected)
        if not selected or len(selected) != len(group.selected):
            raise CapacityOracleError(f"{group.name}: selected test identities are missing or duplicated")
        for obligation in (group.positive, group.negative, group.mutation):
            if not obligation or not set(obligation) <= selected:
                raise CapacityOracleError(f"{group.name}: incomplete test obligation selection")


def _source_path(root: Path, group: Group) -> Path:
    return root / "crates" / group.package / "tests" / f"{group.binary}.rs"


def _test_artifact(root: Path, target: Path, group: Group, output: str) -> Path:
    source = _source_path(root, group).resolve()
    try:
        records = [json.loads(line) for line in output.splitlines() if line]
    except json.JSONDecodeError as error:
        raise CapacityOracleError(f"{group.name}: Cargo emitted invalid artifact JSON") from error
    artifacts = []
    for record in records:
        if not isinstance(record, dict):
            raise CapacityOracleError(f"{group.name}: Cargo emitted a non-object artifact record")
        target_record = record.get("target")
        profile = record.get("profile")
        executable = record.get("executable")
        if (
            record.get("reason") == "compiler-artifact"
            and isinstance(target_record, dict)
            and target_record.get("name") == group.binary
            and target_record.get("kind") == ["test"]
            and target_record.get("src_path") == str(source)
            and isinstance(profile, dict)
            and profile.get("test") is True
            and isinstance(executable, str)
        ):
            artifacts.append(Path(executable))
    if len(artifacts) != 1:
        raise CapacityOracleError(f"{group.name}: Cargo did not emit one exact test artifact")
    artifact = artifacts[0].resolve()
    if not artifact.is_file() or not artifact.is_relative_to(target.resolve()):
        raise CapacityOracleError(f"{group.name}: Cargo emitted a missing or foreign test artifact")
    return artifact


def _selection(names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(names)
    if not names or len(names) != len(set(names)) or any(not isinstance(name, str) or not name for name in names):
        raise CapacityOracleError("libtest selection must be nonempty, exact, and unique")
    return names


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CapacityOracleError("libtest emitted duplicate JSON fields")
        result[key] = value
    return result


def _prepare_evidence_dir(root: Path, evidence: Path) -> Path:
    target = (root / "target").resolve()
    evidence = evidence.resolve()
    if not evidence.is_relative_to(target):
        raise CapacityOracleError("group evidence must be under the framework target directory")
    if evidence.exists():
        raise CapacityOracleError("group evidence directory already exists")
    evidence.mkdir(parents=True)
    return evidence


def _record_process(evidence: Path, name: str, command: Sequence[str], cwd: Path,
                    result: subprocess.CompletedProcess[str]) -> None:
    if not evidence.is_dir():
        raise CapacityOracleError("framework-owned group evidence directory is absent")
    files = {
        f"{name}.stdout": result.stdout,
        f"{name}.stderr": result.stderr,
        f"{name}.process.json": json.dumps({
            "argv": list(command),
            "cwd": str(cwd.resolve()),
            "returncode": result.returncode,
        }, sort_keys=True) + "\n",
    }
    if name == "cargo":
        files["cargo.artifacts.jsonl"] = result.stdout
    elif name == "libtest":
        files["libtest.lifecycle.jsonl"] = result.stdout
    try:
        for filename, contents in files.items():
            with (evidence / filename).open("x") as output:
                output.write(contents)
    except OSError as error:
        raise CapacityOracleError(f"could not retain {name} framework output") from error


def validate_libtest_events(expected: Sequence[str], events: Iterable[object]) -> tuple[str, ...]:
    expected = _selection(expected)
    expected_set = set(expected)
    started: set[str] = set()
    completed: set[str] = set()
    opened = closed = False
    for event in events:
        if not isinstance(event, dict) or closed:
            raise CapacityOracleError("libtest emitted an event outside the selected suite")
        kind, action = event.get("type"), event.get("event")
        if kind == "suite" and action == "started" and not opened:
            if type(event.get("test_count")) is not int or event["test_count"] != len(expected):
                raise CapacityOracleError("libtest selected suite count differs")
            opened = True
        elif kind == "test" and opened:
            name = event.get("name")
            if name not in expected_set:
                raise CapacityOracleError("libtest executed an unselected test")
            if action == "started" and name not in started:
                started.add(name)
            elif action == "stdout" and name in started and name not in completed:
                continue
            elif action == "ok" and name in started and name not in completed:
                completed.add(name)
            else:
                raise CapacityOracleError("libtest reported duplicate, ignored, or failed test outcome")
        elif kind == "suite" and action == "ok" and opened:
            if started != expected_set or completed != expected_set:
                raise CapacityOracleError("libtest selected cases did not all execute")
            totals = {"passed": len(expected), "failed": 0, "ignored": 0, "measured": 0}
            if any(type(event.get(key)) is not int or event[key] != value for key, value in totals.items()):
                raise CapacityOracleError("libtest final totals do not match execution")
            if type(event.get("filtered_out")) is not int or event["filtered_out"] < 0:
                raise CapacityOracleError("libtest omitted its excluded-test total")
            closed = True
        else:
            raise CapacityOracleError("unsupported or failed libtest lifecycle event")
    if not closed:
        raise CapacityOracleError("libtest did not finish the selected suite")
    return tuple(name for name in expected if name in completed)


def run_libtest(root: Path, binary: Path, selected: Sequence[str], evidence: Path) -> tuple[str, ...]:
    selected = _selection(selected)
    binary = binary.resolve()
    before = hashlib.sha256(binary.read_bytes()).hexdigest()
    command = (
        str(binary), "-Zunstable-options", "--format=json", "--exact",
        "--test-threads=1", *selected,
    )
    result = subprocess.run(
        command,
        cwd=root.resolve(),
        env={
            **os.environ,
            "RUSTC_BOOTSTRAP": "1",
            "PYO3_PYTHON": sys.executable,
            "VIRTUAL_ENV": sys.prefix,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    _record_process(evidence, "libtest", command, root, result)
    try:
        events = [json.loads(line, object_pairs_hook=_unique_fields) for line in result.stdout.splitlines()]
    except (CapacityOracleError, json.JSONDecodeError) as error:
        raise CapacityOracleError("libtest emitted invalid JSON execution") from error
    executed = validate_libtest_events(selected, events)
    if result.returncode or hashlib.sha256(binary.read_bytes()).hexdigest() != before:
        raise CapacityOracleError(f"libtest process failed or its executable changed: {result.stderr[-2000:]}")
    return executed


def execute_group(root: Path, target: Path, group: Group, evidence: Path) -> tuple[str, ...]:
    evidence = _prepare_evidence_dir(root, evidence)
    source = _source_path(root, group)
    if not source.is_file():
        raise CapacityOracleError(f"{group.name}: required current test source is absent")
    command = (
        "cargo", "test", "--locked", "--no-run", "--message-format=json",
        "-p", group.package, "--test", group.binary,
    )
    env = {
        **os.environ,
        "CARGO_TARGET_DIR": str(target),
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
        "RUSTC_BOOTSTRAP": "1",
        "PYO3_PYTHON": sys.executable,
        "VIRTUAL_ENV": sys.prefix,
    }
    result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, check=False)
    _record_process(evidence, "cargo", command, root, result)
    if result.returncode:
        raise CapacityOracleError(f"{group.name}: test artifact build failed: {result.stderr[-4000:]}")
    return run_libtest(root, _test_artifact(root, target, group, result.stdout), group.selected, evidence)


def _id(group: Group, test: str) -> str:
    return f"{group.name}::{group.package}/{group.binary}::{test}"


def receipt_packet(head: str, digest: str, run_id: str,
                   executions: Mapping[str, Sequence[str]]) -> dict:
    validate_groups()
    if set(executions) != {group.name for group in GROUPS}:
        raise CapacityOracleError("receipt omitted or added a capacity surface group")
    selected: list[str] = []
    obligations = {"positive": [], "negative": [], "mutation": []}
    for group in GROUPS:
        actual = tuple(executions[group.name])
        if actual != group.selected:
            raise CapacityOracleError(f"{group.name}: current execution differs from exact selection")
        selected.extend(_id(group, test) for test in group.selected)
        for name, tests in (("positive", group.positive), ("negative", group.negative),
                            ("mutation", group.mutation)):
            obligations[name].extend(_id(group, test) for test in tests)
    if len(set(selected)) != len(selected):
        raise CapacityOracleError("capacity test identity collision")
    return {
        "schema": 1,
        "run_id": run_id,
        "head": head,
        "source_digest": digest,
        "oracle": "capacity",
        "argv": [sys.executable, "scripts/dtype_capacity_oracle.py"],
        "selected": selected,
        "executed": [{"id": name, "outcome": "passed"} for name in selected],
        "obligations": obligations,
        "hosts": {},
        "devices": [],
    }


def _required_environment(root: Path) -> SourceIdentity:
    values = {name: os.environ.get(name) for name in (
        "CHELIS_ORACLE_HEAD", "CHELIS_ORACLE_SOURCE_DIGEST", "CHELIS_ORACLE_RUN_ID",
        "CHELIS_ORACLE_RECEIPT",
    )}
    if any(not value for value in values.values()):
        raise CapacityOracleError("capacity receipt requires framework-owned environment")
    actual = source_identity(root)
    if actual.head != values["CHELIS_ORACLE_HEAD"] or actual.digest != values["CHELIS_ORACLE_SOURCE_DIGEST"]:
        raise CapacityOracleError("capacity receipt environment is stale against current source")
    receipt = Path(values["CHELIS_ORACLE_RECEIPT"]).resolve()
    if receipt.exists() or not receipt.is_relative_to(root / "target"):
        raise CapacityOracleError("capacity receipt path must be new and under target/")
    return actual


def _receipt_evidence_dir(root: Path, receipt: Path) -> Path:
    evidence = receipt.parent / "capacity"
    if not evidence.is_relative_to((root / "target").resolve()):
        raise CapacityOracleError("framework evidence must be beside a target receipt")
    if evidence.exists():
        raise CapacityOracleError("framework evidence directory already exists for this receipt")
    return evidence


def main() -> int:
    try:
        validate_groups()
        identity = _required_environment(REPO_ROOT)
        # Bindings run first: the absent final-zero-legacy test is an explicit
        # integration blocker and must not be bypassed with other green legs.
        order = (GROUPS[-1], *GROUPS[:-1])
        target = REPO_ROOT / "target/dtype-capacity"
        receipt = Path(os.environ["CHELIS_ORACLE_RECEIPT"]).resolve()
        evidence = _receipt_evidence_dir(REPO_ROOT, receipt)
        executions = {
            group.name: execute_group(REPO_ROOT, target, group, evidence / group.name)
            for group in order
        }
        if source_identity(REPO_ROOT) != identity:
            raise CapacityOracleError("source changed while capacity controls executed")
        packet = receipt_packet(identity.head, identity.digest, os.environ["CHELIS_ORACLE_RUN_ID"], executions)
        Path(os.environ["CHELIS_ORACLE_RECEIPT"]).write_text(json.dumps(packet, sort_keys=True) + "\n")
        print(PASS_LINE)
        return 0
    except (CapacityOracleError, OSError, ValueError) as error:
        print(f"DTYPE CAPACITY ORACLE: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
