#!/usr/bin/env python3
"""Unit tests for the chelis#1205 front-end performance oracle."""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import unittest


def load_oracle():
    path = Path(__file__).with_name("compiler_front_end_performance.py")
    spec = importlib.util.spec_from_file_location("compiler_front_end_performance", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


oracle = load_oracle()


class CompilerFrontEndPerformanceOracleTests(unittest.TestCase):
    def test_commands_cover_fixtures_structural_work_and_execution(self) -> None:
        rendered = "\n".join(
            oracle.command_text(command) for command in oracle.FOCUSED_COMMANDS
        )
        for required in (
            "scripts/test_front_end_performance_fixtures.py",
            "issue_1205_effect_clone_work_is_linear",
            "issue_1205_host_lowering_work_is_linear",
            "issue_1205_callable_scope_work_is_linear",
            "issue_1205_preflight_preserves_static_to_tensor_literals",
            "issue_1205_preflight_does_not_confuse_shadowed_operands_with_calls",
            "issue_1205_preflight_tracks_callable_shadowing_and_aliases",
            "cargo test -p chelis-cli --test issue_1205_front_end_performance",
            "--ignored --test-threads=1",
        ):
            self.assertIn(required, rendered)

    def test_failure_stops_later_commands(self) -> None:
        calls: list[tuple[str, ...]] = []

        def runner(command, **_kwargs):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 9)

        with self.assertRaisesRegex(oracle.OracleFailure, "exit 9"):
            oracle.run_oracle(
                (("cargo", "first"), ("cargo", "second")),
                runner=runner,
                environment={},
            )
        self.assertEqual(calls, [("cargo", "first")])

    def test_success_emits_the_exact_acceptance_marker(self) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0)

        output = io.StringIO()
        with redirect_stdout(output):
            oracle.run_oracle(
                (("cargo", "one"),), runner=runner, environment={}
            )
        self.assertTrue(
            output.getvalue().rstrip().endswith(
                "compiler front-end performance oracle: PASS"
            )
        )

    def test_oracle_disables_cargo_husky_mutation(self) -> None:
        seen = {}

        def runner(command, **kwargs):
            seen.update(kwargs["env"])
            return subprocess.CompletedProcess(command, 0)

        oracle.run_oracle((("cargo", "one"),), runner=runner, environment={})
        self.assertEqual(seen["CARGO_HUSKY_DONT_INSTALL_HOOKS"], "1")


if __name__ == "__main__":
    unittest.main()
