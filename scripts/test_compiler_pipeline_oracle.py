#!/usr/bin/env python3

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest


def load_oracle():
    path = Path(__file__).with_name("compiler_pipeline_oracle.py")
    spec = importlib.util.spec_from_file_location("compiler_pipeline_oracle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_oracle()


class CompilerPipelineOracleTests(unittest.TestCase):
    def test_commands_cover_every_required_consumer_and_guard(self) -> None:
        rendered = "\n".join(
            oracle.command_text(command) for command in oracle.FOCUSED_COMMANDS
        )
        for required in (
            "-p chelis-types --test type_analysis_outcome",
            "analysis_uses_one_type_session",
            "diagnostic_checkpoint_tests",
            "-p chelis-compiler-api --test pipeline_contract",
            "fragment_parity",
            "compiled_context",
            "redteam_typecheck_cache",
            "test(source_arch)",
            "test(fragment::tests) | test(artifact_type_tests) | test(artifact_outcome_tests)",
            "cargo test -p chelis-compiler-api --doc",
            "-p chelis-cli --test stdlib_typecheck_cache_oracle",
            "issue_207_check_exit_code_invariant",
            "check_exits_zero_with_errors_contract",
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


if __name__ == "__main__":
    unittest.main()
