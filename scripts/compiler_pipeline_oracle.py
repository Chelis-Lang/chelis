#!/usr/bin/env python3
"""Authoritative oracle for canonical compiler-pipeline ownership and parity."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUSED_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("cargo", "nextest", "run", "-p", "chelis-pipeline-core"),
    ("cargo", "test", "-p", "chelis-pipeline-core", "--doc"),
    (
        sys.executable,
        "-m",
        "unittest",
        "scripts/test_pipeline_core_dependency_guard.py",
        "scripts/test_pipeline_core_documentation_guard.py",
        "scripts/test_pipeline_core_compile_fail.py",
    ),
    (sys.executable, "scripts/pipeline_core_dependency_guard.py"),
    (sys.executable, "scripts/pipeline_core_documentation_guard.py"),
    (sys.executable, "scripts/check_pipeline_core_compile_fail.py"),
    ("cargo", "nextest", "run", "-p", "chelis-reef", "--lib"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-reef",
        "--test",
        "pipeline_parity",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "--test",
        "type_analysis_outcome",
        "-E",
        "all()",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "-E",
        "test(analysis_uses_one_type_session)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "-E",
        "test(diagnostic_checkpoint_tests)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-types",
        "-E",
        "test(adt_deftype_and_construct) | test(adt_nullary_constructor) | test(in_vocabulary_tags_without_expression_disposition_are_rejected_loudly) | test(childless_block_is_malformed) | test(untagged_list_in_expression_position_is_rejected_loudly) | test(unknown_tag_outside_the_vocabulary_keeps_the_raw_string_loud_arm) | test(canonical_t_prim_with_an_extra_child_is_rejected_once) | test(independent_rhs_and_malformed_let_ascription_each_report_once) | test(malformed_canonical_primitive_reports_once_at_every_type_consumer) | test(malformed_nested_type_and_dimension_forms_report_once_without_drops) | test(prebound_failure_is_owned_by_its_exact_duplicate_name_declaration) | test(macro_expansion_attributes_to_call_site_module) | test(malformed_parameter_is_rejected_once_by_the_binder_owner) | test(cast_operand_and_target_failures_remain_two_independent_roots)",
    ),
    ("cargo", "nextest", "run", "-p", "chelis-macros"),
    (sys.executable, "scripts/check_checkpoint_compile_fail.py"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "pipeline_contract",
        "--test",
        "pipeline_facade_imports",
        "--test",
        "fragment_parity",
        "--test",
        "compiled_context",
        "--test",
        "redteam_typecheck_cache",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "-E",
        "test(source_arch)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "-E",
        "test(fragment::tests) | test(artifact_type_tests) | test(artifact_outcome_tests) | test(typed_deep_node_wire_bridge_preserves_the_complete_node_shape)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "deep_authoring",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-compiler-api",
        "--test",
        "invariant_decode",
    ),
    ("cargo", "test", "-p", "chelis-compiler-api", "--doc"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-prove",
        "-E",
        "test(f7_deep_fuzz_only_runs_the_fuzz_loop) | test(f6_deep_chelis_role_property_without_source_kind_is_error) | test(f6_deep_invalid_source_kind_is_error) | test(simplex_band_is_served_by_constructor_generation_not_starved) | test(typed_deep_clean_scalar_producer_still_passes) | test(typed_deep_nonfinite_scalar_producer_still_fails)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "issue_977_constraint_fuzz",
        "-E",
        "test(deep_constraint_sampler_matches_surf_evidence)",
    ),
    ("cargo", "nextest", "run", "-p", "chelis-validate"),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-deep",
        "-E",
        "test(typed_unknown_tag_warning) | test(typed_known_tag_missing_metadata_warning) | test(nested_bare_lists_stay_legal_after_858) | test(declaration_type_parameter_lists_use_syntax_roles) | test(resolve_body_less_function_is_malformed_not_panic) | test(splice_body_less_function_is_malformed_not_panic)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-e2e",
        "--test",
        "example_corpus_validate",
        "-E",
        "test(phase1f_negative_fixtures_fail_in_validator_and_compiler)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "opaque_check",
        "-E",
        "test(validate_deep_rejects_module_reopen) | test(validate_deep_rejects_stem_mangled_forges)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-lint",
        "-E",
        "test(opaque_domain_construction)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "style_gate",
        "-E",
        "test(validate_deep_fails_on_opaque_domain_construction) | test(validate_deep_fails_on_untyped_opaque_record_update) | test(validate_deep_still_rejects_crlf_after_directive_stripping) | test(validate_deep_still_rejects_missing_final_newline_after_directive_stripping)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "wrapped_black_scholes_fixture",
        "-E",
        "test(wrapped_fixture_includes_octant_emitted_spans) | test(wrapped_fixture_span_ids_match_sidecar)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "stdlib_typecheck_cache_oracle",
        "--test",
        "issue_207_check_exit_code_invariant",
        "--test",
        "check_exits_zero_with_errors_contract",
        "--ignore-default-filter",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "cli",
        "-E",
        "test(fmt_deep_rejects_a_bare_name_in_a_runtime_position)",
    ),
    (
        "cargo",
        "nextest",
        "run",
        "-p",
        "chelis-cli",
        "--test",
        "cli",
        "-E",
        "test(build_c_tensor_grad_local_wrapper_over_function_param_builds)",
    ),
    ("cargo", "nextest", "run", "-p", "chelis-e2e", "--test", "pipeline"),
    ("cargo", "check", "-p", "chelis-e2e", "--bin", "check_snippet"),
)


class OracleFailure(RuntimeError):
    """A canonical pipeline acceptance command failed."""


def command_text(command: Sequence[str]) -> str:
    return " ".join(command)


def cargo_bin_directory() -> Path | None:
    cargo = shutil.which("cargo")
    if cargo is not None:
        return Path(cargo).resolve().parent
    toolchains = Path.home() / ".rustup" / "toolchains"
    candidates = sorted(toolchains.glob("stable-*/bin/cargo"))
    return candidates[0].parent if candidates else None


def oracle_environment() -> dict[str, str]:
    environment = os.environ.copy()
    cargo_directory = cargo_bin_directory()
    path_entries = []
    if cargo_directory is not None:
        path_entries.append(str(cargo_directory))
    cargo_home_bin = Path.home() / ".cargo" / "bin"
    if cargo_home_bin.is_dir():
        path_entries.append(str(cargo_home_bin))
    if path_entries:
        environment["PATH"] = os.pathsep.join(
            [*path_entries, environment.get("PATH", "")]
        )
    environment.setdefault("PYO3_PYTHON", sys.executable)
    environment.setdefault(
        "CARGO_TARGET_DIR", str(REPO_ROOT / "target" / "agents" / "compiler-pipeline-oracle")
    )
    return environment


def run_oracle(
    commands: Sequence[Sequence[str]] = FOCUSED_COMMANDS,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: dict[str, str] | None = None,
) -> None:
    env = oracle_environment() if environment is None else environment
    for command in commands:
        print(f"+ {command_text(command)}", flush=True)
        completed = runner(command, cwd=REPO_ROOT, env=env, check=False)
        if completed.returncode != 0:
            raise OracleFailure(
                f"command failed with exit {completed.returncode}: {command_text(command)}"
            )
    print("compiler pipeline oracle: PASS", flush=True)


def main() -> int:
    from observed_cargo import observed_cargo_environment

    try:
        # This oracle also supports a rustup toolchain not yet on PATH.
        # Resolve that existing fallback before installing the observer.
        environment = oracle_environment()
        with observed_cargo_environment(
            environment, python=Path(environment["PYO3_PYTHON"]), required=True
        ) as environment:
            run_oracle(environment=environment)
    except (OracleFailure, ValueError, OSError) as error:
        print(f"compiler pipeline oracle: FAIL: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
