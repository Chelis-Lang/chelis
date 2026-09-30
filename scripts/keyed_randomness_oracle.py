#!/usr/bin/env python3
"""Finite #2413 keyed-randomness acceptance; see docs/keyed_randomness_oracle.md."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RETIRED = ("static_controls", "StaticControls", "execution_exclusion", "FixedControlPlan")


class OracleFailure(RuntimeError):
    """Required evidence is missing or unsuccessful."""


@dataclass(frozen=True)
class Suite:
    package: str
    target: str
    tests: tuple[str, ...]
    lanes: tuple[str, ...]

    @property
    def binary(self) -> str:
        return f"{self.package}::{self.target}"

    def identities(self) -> set[str]:
        return {f"{self.binary}::{name}" for name in self.tests}


# Reviewed finite corpus, not a discovery rule or a universal language claim.
# Removing/renaming one of these witnesses requires reviewing its replacement.
SUITES = (
    Suite("chelis-types", "random_key_reference", (
        "key_ref_worked_values_are_the_kernels_values",
    ), ("reference",)),
    Suite("chelis-types", "key_linearity", (
        "tensor_key_operations_preserve_shapes_and_affinity",
        "tensor_key_contracts_survive_checker_context_serialization",
        "deriving_keys_is_accepted_and_each_key_is_used_once",
        "reusing_a_key_after_a_key_operation_is_rejected",
        "reusing_a_key_after_a_draw_is_rejected",
        "a_key_is_never_copied_or_borrowed",
        "binding_a_key_to_another_name_moves_it",
        "a_closure_never_captures_a_key",
        "a_branch_consume_survives_the_join",
        "transform_calls_consume_key_arguments",
        "a_generic_reached_indirectly_is_never_instantiated_at_a_key",
        "keys_pass_through_concrete_parameters_and_data_types",
        "vmap_never_broadcasts_a_key",
    ), ("checker",)),
    Suite("chelis-types", "key_builtin_cases", (
        "every_builtin_case_answers_a_key_carrying_instantiation_as_declared",
    ), ("checker",)),
    Suite("chelis-types", "jit_par_passthrough", (
        "par_is_rejected_with_the_typed_issue_fence",
    ), ("par-refusal",)),
    Suite("chelis-ir", "key_operations_ir", (
        "the_key_chain_evaluates_to_the_reference_keys",
        "chained_draws_match_the_reference_at_every_float_dtype",
        "a_negative_runtime_split_count_traps_and_zero_is_empty",
        "dead_code_elimination_keeps_only_the_random_nodes_that_can_trap",
        "grad_keeps_a_discarded_key_sourced_draw_that_can_trap",
        "a_key_consumed_twice_is_rejected_for_every_consumer_kind",
        "rule_v3_admits_exclusive_arms_and_rejects_overlapping_ones",
        "a_key_constant_is_rejected_and_folding_keeps_derivations_symbolic",
        "a_key_takes_no_cotangent_and_grad_replays_the_forward_key",
        "vmap_maps_key_rows_and_splits_every_row",
        "vmap_of_vmap_of_a_draw_verifies_and_draws_each_row_with_its_key",
        "invalid_dropout_rates_trap_even_when_the_tensor_is_empty",
    ), ("dag", "verifier", "reference")),
    Suite("chelis-ir", "key_operand_random_ir", (
        "dropout_input_adjoint_replays_the_forward_mask_through_the_key",
        "uniform_bound_adjoints_match_the_05_op_8_transcription",
        "a_parameter_reaching_the_rate_through_adjoint_slots_is_rejected",
        "a_rate_reached_only_through_a_zero_cotangent_slot_differentiates",
        "the_verifier_rejects_a_replay_of_an_unconsumed_key",
    ), ("dag", "verifier", "reference")),
    Suite("chelis-ir", "key_operation_activations", (
        "a_split_arm_and_a_draw_arm_draw_only_the_selected_arms_bits",
        "a_key_derived_under_exclusive_sharing_is_used_only_under_that_activation",
        "an_inactive_runtime_split_reads_no_count",
        "a_branch_join_is_the_taken_arms_key",
    ), ("dag", "verifier")),
    Suite("chelis-compiler-api", "key_surface_lanes", (
        "every_key_source_draws_the_reference_bits_in_eval_and_c",
        "vmap_over_key_rows_is_the_stack_of_per_row_draws",
        "a_branch_draws_only_in_its_selected_arm",
        "keys_cross_host_functions_recursion_tuples_and_data_types",
        "keys_computed_inside_lowered_definitions_draw_the_reference_bits_in_eval_and_c",
        "grad_through_a_keyed_draw_replays_its_mask",
        "a_key_valued_branch_joins_to_the_taken_arms_key_in_eval_and_c",
        "a_vmapped_key_join_selects_each_rows_arm_in_eval_and_c",
    ), ("eval", "c", "reference")),
    # #2656 / #2707: deliberately required, never optional on a pre-merge tree.
    Suite("chelis-compiler-api", "key_tensor_forms", (
        "tensor_key_forms_match_the_independent_scalar_reference_in_eval_and_c",
        "tensor_key_split_checks_runtime_count_before_allocating",
        "tensor_fold_rejects_runtime_shape_mismatch_without_broadcasting",
    ), ("eval", "c", "reference")),
    Suite("chelis-compiler-api", "key_operations_c", (
        "the_key_chain_runs_to_the_reference_keys_in_c",
        "chained_draws_match_the_reference_in_c_at_every_float_dtype",
        "a_draw_and_a_fold_run_under_their_owners_activation_in_c_and_eval",
        "per_row_controls_activations_and_bound_adjoints_agree_in_c_and_eval",
        "vmap_of_vmap_of_a_draw_agrees_in_c_and_eval",
        "an_inactive_runtime_split_reads_no_count_in_c_as_in_eval",
        "a_negative_runtime_split_count_traps_in_c_as_in_eval",
    ), ("dag", "c-dag", "reference")),
    Suite("chelis-compiler-api", "key_operand_random_c", (
        "seeded_runtime_controls_match_the_spec_in_c_and_eval",
        "native_replay_and_bound_adjoints_match_eval",
    ), ("dag", "c-dag", "reference")),
    Suite("chelis-compiler-api", "dropout_fixed_stream_api", (
        "key_reference_matches_key_ref_py",
        "discarded_forward_draw_consumes_its_own_key_not_the_live_draws",
        "a_selected_activations_discarded_draws_still_validate_and_trap",
        "a_value_declaration_no_run_declaration_names_is_not_initialized",
        "concrete_runtime_rate_actual_draws_its_keys_mask",
        "generic_runtime_rate_draws_at_its_dtypes_width",
        "issue_2405_dropout_beneath_recursion_and_runtime_if_draws_its_keys",
        "issue_2405_unrelated_match_leaves_dropout_runnable",
        "uniform_draws_beneath_recursion_take_their_split_keys",
        "a_draw_in_an_unselected_arm_neither_validates_nor_draws_in_the_dag_evaluator",
        "nonunit_cotangent_matches_same_key_finite_differences",
    ), ("eval", "dag", "reference")),
    Suite("chelis-compiler-api", "wire_random_domains", (
        "a_computed_key_tensor_and_a_gradient_export_keep_one_consumer_per_key",
        "the_codec_admits_the_key_chain_and_rejects_every_malformed_key_form",
        "the_codec_admits_a_gated_key_operation_and_confines_its_keys",
        "a_replay_reads_only_its_forward_draws_key_under_that_draws_controls",
    ), ("wire", "verifier")),
    Suite("chelis-cli", "dropout_fixed_stream_cli", (
        "worked_values_match_key_ref_py",
        "executable_example_survives_format_check_and_exact_eval",
        "invalid_empty_rate_traps_in_eval_and_c_and_a_runtime_rate_builds",
        "a_keyed_dropout_under_a_runtime_arm_reads_only_its_own_key_in_eval_and_c",
    ), ("cli", "eval", "c", "examples", "reference")),
    Suite("chelis-cli", "issue_2463_key_dead_draw_traps", (
        "a_dead_invalid_draw_in_a_selected_def_traps_in_eval_and_c",
        "a_dead_value_declaration_reached_only_through_grad_traps_in_eval_and_c",
        "a_trap_in_a_discarded_dropout_input_traps_in_eval_and_c",
        "a_trap_in_a_discarded_uniform_template_traps_in_eval_and_c",
    ), ("cli", "eval", "c")),
    Suite("chelis-cli", "key_operations_in_branch_arms", (
        "grad_of_a_split_arm_beside_a_draw_arm_agrees_with_the_reference",
        "vmap_of_a_split_arm_beside_a_draw_arm_agrees_with_the_reference",
        "an_unselected_split_keys_does_not_trap_on_its_count",
        "a_selected_split_keys_still_traps_on_a_negative_count",
    ), ("cli", "eval", "c", "reference")),
    Suite("chelis-cli", "key_random_primitives_cli", (
        "uniform_like_draws_from_its_key_in_eval_and_c",
        "uniform_like_rejects_invalid_bounds_before_later_work_in_eval_and_c",
        "uniform_like_rejects_invalid_bounds_before_later_work_under_grad_in_eval_and_c",
        "uniform_like_valid_bounds_run_in_eval_c_and_grad",
    ), ("cli", "eval", "c", "reference")),
    Suite("chelis-cli", "jit_par_runtime_gap", (
        "eval_par_scalar_is_fenced_before_execution",
        "eval_par_tensor_is_fenced_before_execution",
        "build_c_par_tensor_is_fenced_before_emission",
    ), ("cli", "par-refusal")),
)


def expected_tests(suites: tuple[Suite, ...]) -> set[str]:
    identities = [identity for suite in suites for identity in suite.identities()]
    if not identities or len(identities) != len(set(identities)):
        raise OracleFailure("empty or duplicate acceptance identities")
    if any(not suite.tests or len(suite.tests) != len(set(suite.tests)) for suite in suites):
        raise OracleFailure("empty or duplicate suite test names")
    return set(identities)


def nextest_command(action: str, suites: tuple[Suite, ...], config: Path) -> list[str]:
    expected_tests(suites)
    packages = {suite.package for suite in suites}
    if len(packages) != 1:
        raise OracleFailure("nextest group must contain exactly one package")
    expression = " | ".join(
        f"(binary_id(={suite.binary}) & (" + " | ".join(f"test(={name})" for name in suite.tests) + "))"
        for suite in suites
    )
    package = next(iter(packages))
    command = ["cargo", "nextest", action, "--locked", "--config-file", str(config), "-p", package]
    if package == "chelis-compiler-api":
        # Native execution targets require the instrumented carried runtime.
        command.extend(["--features", "chelis-compiler-api/ownership-ledger"])
    for suite in suites:
        command.extend(["--test", suite.target])
    command.extend(["--profile", "ci-full", "--ignore-default-filter", "-E", expression])
    command.extend(["--message-format", "json"] if action == "list" else
                   ["--no-fail-fast", "--retries", "0", "--test-threads", "2"])
    return command


def isolated_nextest_config(root: Path, output: Path, target: Path) -> Path:
    source = (root / ".config/nextest.toml").read_text()
    settings = tomllib.loads(source)
    if "store" in settings or settings.get("profile", {}).get("ci-full", {}).get("junit", {}).get("path") != "junit.xml":
        raise OracleFailure("Nextest store/JUnit config changed; review oracle receipt routing")
    config = output / "nextest-isolated.toml"
    config.write_text(source + "\n[store]\ndir = " + json.dumps(str(target / "nextest")) + "\n")
    return config


def validate_listing(packet: dict, suites: tuple[Suite, ...], root: Path, target: Path) -> set[str]:
    expected = expected_tests(suites)
    try:
        if Path(packet["rust-build-meta"]["target-directory"]).resolve() != target.resolve():
            raise OracleFailure("nextest used a foreign target directory")
        binaries = packet["rust-suites"]
        if set(binaries) != {suite.binary for suite in suites}:
            raise OracleFailure("nextest binary census differs from required targets")
        selected = set()
        for suite in suites:
            row = binaries[suite.binary]
            if row["binary-id"] != suite.binary or row["package-name"] != suite.package or row["status"] != "listed":
                raise OracleFailure(f"wrong nextest binary identity: {suite.binary}")
            binary = Path(row["binary-path"]).resolve()
            if not binary.is_file() or not binary.is_relative_to(target.resolve()):
                raise OracleFailure(f"missing or foreign test binary: {suite.binary}")
            if Path(row["cwd"]).resolve() != (root / "crates" / suite.package).resolve():
                raise OracleFailure(f"foreign test checkout: {suite.binary}")
            for name, case in row["testcases"].items():
                status = case["filter-match"]["status"]
                if type(case["ignored"]) is not bool or status not in {"matches", "mismatch"}:
                    raise OracleFailure(f"malformed nextest case: {suite.binary}::{name}")
                if status == "matches":
                    if case["ignored"]:
                        raise OracleFailure(f"ignored required test: {suite.binary}::{name}")
                    selected.add(f"{suite.binary}::{name}")
        if selected != expected:
            raise OracleFailure(f"test selection differs: missing={sorted(expected - selected)}, extra={sorted(selected - expected)}")
        return selected
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise OracleFailure(f"malformed nextest listing: {error}") from error


def junit_outcomes(path: Path, expected: set[str]) -> dict[str, str]:
    try:
        document = ET.parse(path).getroot()
        if document.tag != "testsuites":
            raise OracleFailure("missing nextest testsuites report")
        observed = {}
        for case in document.iter("testcase"):
            if not case.get("classname") or not case.get("name"):
                raise OracleFailure("JUnit case lacks identity")
            identity = f"{case.get('classname')}::{case.get('name')}"
            if identity in observed or identity not in expected:
                raise OracleFailure(f"duplicate or unexpected execution: {identity}")
            status = "passed"
            if case.find("skipped") is not None:
                status = "skipped"
            if any(case.find(tag) is not None for tag in (
                "failure", "error", "rerunFailure", "rerunError", "flakyFailure", "flakyError",
            )):
                status = "failed"
            observed[identity] = status
        return observed
    except (OSError, ET.ParseError) as error:
        raise OracleFailure(f"missing or malformed JUnit: {error}") from error


def validate_junit(path: Path, expected: set[str]) -> set[str]:
    outcomes = junit_outcomes(path, expected)
    if set(outcomes) != expected or any(status != "passed" for status in outcomes.values()):
        raise OracleFailure(f"missing or nonpassing executions: {sorted(expected - {name for name, status in outcomes.items() if status == 'passed'})}")
    return set(outcomes)


def check_retirement(root: Path) -> None:
    pattern = re.compile(r"\b(?:" + "|".join(RETIRED) + r")\b")
    for package in ("chelis-ir", "chelis-compiler-api"):
        paths = sorted((root / "crates" / package / "src").rglob("*.rs"))
        if not paths:
            raise OracleFailure(f"missing production sources: {package}")
        for path in paths:
            match = pattern.search(path.read_text())
            if match:
                raise OracleFailure(f"retired dispatch marker {match[0]} in {path.relative_to(root)}")


def candidate_sha(root: Path) -> str:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, timeout=30, check=True)
        return result.stdout.strip()
    sha = git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or git("status", "--porcelain", "--untracked-files=normal"):
        raise OracleFailure("acceptance requires an unchanged, clean committed candidate")
    return sha


def run_command(argv: list[str], root: Path, environment: dict[str, str], output: Path,
                label: str, timeout: int) -> str:
    print(f"{label}: {' '.join(argv)}", flush=True)
    (output / f"{label}.command.json").write_text(json.dumps(argv) + "\n")
    stdout_path = output / f"{label}.stdout"
    stderr_path = output / f"{label}.stderr"
    timed_out = False
    # Files avoid an unbounded communicate() wait if a descendant retains a pipe.
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.Popen(argv, cwd=root, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    if timed_out or process.returncode:
        raise OracleFailure(f"{label}: {'timeout/interruption' if timed_out else f'exit {process.returncode}'}; see {output}")
    return stdout_path.read_text()


def execute(root: Path, output: Path, environment: dict[str, str], timeout: int,
            suites: tuple[Suite, ...] = SUITES) -> dict:
    receipt = {"schema": 1, "oracle": "keyed-randomness", "status": "fail", "candidate_sha": None,
               "started_at": datetime.now(timezone.utc).isoformat(), "groups": [],
               "required": sorted(expected_tests(suites)), "passed": [], "failed": [], "skipped": [], "unrun": [],
               "par": "typed rejection chelis#2503; no positive execution claim"}
    try:
        receipt["candidate_sha"] = candidate_sha(root)
        check_retirement(root)
        receipt["retirement"] = {"status": "pass", "forbidden_markers": list(RETIRED)}
        run_command(["cargo", "build", "--locked", "-p", "chelis-cli", "-p", "chelis-runtime"],
                    root, environment, output, "build", timeout)
        target = Path(environment["CARGO_TARGET_DIR"])
        config = isolated_nextest_config(root, output, target)
        junit = target / "nextest/ci-full/junit.xml"
        for package in dict.fromkeys(suite.package for suite in suites):
            group = tuple(suite for suite in suites if suite.package == package)
            listed = run_command(nextest_command("list", group, config), root, environment, output,
                                 f"{package}-list", timeout)
            selected = validate_listing(json.loads(listed), group, root, target)
            junit.unlink(missing_ok=True)
            run_error = None
            try:
                run_command(nextest_command("run", group, config), root, environment, output,
                            f"{package}-run", timeout)
            except OracleFailure as error:
                run_error = error
            outcomes = junit_outcomes(junit, selected)
            shutil.copyfile(junit, output / f"{package}.junit.xml")
            for status in ("passed", "failed", "skipped"):
                receipt[status].extend(sorted(name for name, result in outcomes.items() if result == status))
            complete = set(outcomes) == selected and all(status == "passed" for status in outcomes.values())
            receipt["groups"].append({"package": package, "status": "pass" if complete and run_error is None else "fail",
                                      "outcomes": outcomes,
                                      "lane_coverage": {suite.binary: list(suite.lanes) for suite in group}})
            if run_error is not None:
                raise run_error
            if not complete:
                raise OracleFailure(f"{package}: missing or nonpassing required executions")
        if candidate_sha(root) != receipt["candidate_sha"]:
            raise OracleFailure("candidate changed during acceptance")
        check_retirement(root)
        if set(receipt["passed"]) != set(receipt["required"]):
            raise OracleFailure("required tests were not all executed")
        receipt["status"] = "pass"
    except (OracleFailure, OSError, ValueError, subprocess.SubprocessError) as error:
        receipt["error"] = str(error)
    receipt["unrun"] = sorted(set(receipt["required"]) - set(receipt["passed"]) - set(receipt["failed"]) - set(receipt["skipped"]))
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=1800, help="seconds per build/list/run command (default: 1800)")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if sys.version_info[:2] != (3, 11) or Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        parser.error("use .venv/bin/python (uv-managed Python 3.11)")
    target = ROOT / "target/agents/2413-keyed-oracle"
    target.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(ROOT.resolve()):
        parser.error("oracle target must belong to this worktree")
    with (target / "oracle.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another keyed randomness oracle owns this worktree target")
        output = Path(tempfile.mkdtemp(prefix="run-", dir=target))
        environment = os.environ.copy()
        environment["CARGO_TARGET_DIR"] = str(target)
        environment["PYO3_PYTHON"] = sys.executable
        receipt = execute(ROOT, output, environment, args.timeout)
        verdict = receipt["status"].upper()
        print(f"KEYED RANDOMNESS ORACLE: {verdict}; receipt: {output / 'receipt.json'}")
        if receipt.get("error"):
            print(receipt["error"], file=sys.stderr)
        return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
