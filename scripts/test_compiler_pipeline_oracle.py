#!/usr/bin/env python3

import importlib.util
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
