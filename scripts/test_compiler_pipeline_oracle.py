#!/usr/bin/env python3

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest


def load_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_module("compiler_pipeline_oracle", "compiler_pipeline_oracle.py")
checkpoint = load_module(
    "check_checkpoint_compile_fail", "check_checkpoint_compile_fail.py"
)


class CompilerPipelineOracleTests(unittest.TestCase):
    def test_commands_cover_every_required_consumer_and_guard(self) -> None:
        rendered = "\n".join(
            oracle.command_text(command) for command in oracle.FOCUSED_COMMANDS
        )
        for required in (
            "-p chelis-pipeline-core",
            "cargo test -p chelis-pipeline-core --doc",
            "scripts/test_pipeline_core_dependency_guard.py",
            "scripts/pipeline_core_dependency_guard.py",
            "scripts/test_pipeline_core_documentation_guard.py",
            "scripts/pipeline_core_documentation_guard.py",
            "scripts/test_pipeline_core_compile_fail.py",
            "scripts/check_pipeline_core_compile_fail.py",
            "-p chelis-reef --lib",
            "-p chelis-reef --test pipeline_parity",
            "pipeline_facade_imports",
            "-p chelis-types --test type_analysis_outcome",
            "analysis_uses_one_type_session",
            "diagnostic_checkpoint_tests",
            "scripts/check_checkpoint_compile_fail.py",
            "-p chelis-compiler-api --test pipeline_contract",
            "fragment_parity",
            "compiled_context",
            "redteam_typecheck_cache",
            "test(source_arch)",
            "test(fragment::tests) | test(artifact_type_tests) | test(artifact_outcome_tests)",
            "typed_deep_node_wire_bridge_preserves_the_complete_node_shape",
            "cargo test -p chelis-compiler-api --doc",
            "-p chelis-prove -E test(f7_deep_fuzz_only_runs_the_fuzz_loop)",
            "f6_deep_chelis_role_property_without_source_kind_is_error",
            "f6_deep_invalid_source_kind_is_error",
            "simplex_band_is_served_by_constructor_generation_not_starved",
            "typed_deep_clean_scalar_producer_still_passes",
            "typed_deep_nonfinite_scalar_producer_still_fails",
            "-p chelis-cli --test issue_977_constraint_fuzz",
            "deep_constraint_sampler_matches_surf_evidence",
            "-p chelis-validate",
            "-p chelis-cli --test opaque_check",
            "validate_deep_rejects_module_reopen",
            "validate_deep_rejects_stem_mangled_forges",
            "-p chelis-lint -E test(opaque_domain_construction)",
            "-p chelis-cli --test style_gate",
            "validate_deep_fails_on_opaque_domain_construction",
            "-p chelis-cli --test wrapped_black_scholes_fixture",
            "wrapped_fixture_includes_octant_emitted_spans",
            "wrapped_fixture_span_ids_match_sidecar",
            "-p chelis-cli --test stdlib_typecheck_cache_oracle",
            "issue_207_check_exit_code_invariant",
            "check_exits_zero_with_errors_contract",
            "fmt_deep_rejects_a_bare_name_in_a_runtime_position",
            "build_c_tensor_grad_local_wrapper_over_function_param_builds",
            "-p chelis-e2e --test pipeline",
            "cargo check -p chelis-e2e --bin check_snippet",
        ):
            self.assertIn(required, rendered)

    def test_failure_propagates_and_stops_later_commands(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **_kwargs):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 7)

        with self.assertRaisesRegex(oracle.OracleFailure, "exit 7"):
            oracle.run_oracle(
                (("cargo", "first"), ("cargo", "second")),
                runner=runner,
                environment={},
            )
        self.assertEqual(calls, [("cargo", "first")])

    def test_success_reports_the_authoritative_pass_marker(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **_kwargs):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            oracle.run_oracle(
                (("cargo", "one"), ("cargo", "two")),
                runner=runner,
                environment={},
            )
        self.assertEqual(calls, [("cargo", "one"), ("cargo", "two")])
        self.assertIn("compiler pipeline oracle: PASS", output.getvalue())

    def test_checkpoint_fixture_requires_the_exact_type_error(self) -> None:
        diagnostic = "mismatched types: expected `DiagnosticCheckpoint`, found `usize`"

        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 101, "", diagnostic)

        checkpoint.validate(runner=runner, environment={})

    def test_checkpoint_fixture_rejects_compile_success(self) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "", "")

        with self.assertRaisesRegex(
            checkpoint.CheckpointCompileFailure, "raw checkpoint offset compiled"
        ):
            checkpoint.validate(runner=runner, environment={})

    def test_checkpoint_fixture_rejects_an_unrelated_failure(self) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 101, "", "network failure")

        with self.assertRaisesRegex(
            checkpoint.CheckpointCompileFailure, "lacks the required text"
        ):
            checkpoint.validate(runner=runner, environment={})


if __name__ == "__main__":
    unittest.main()
