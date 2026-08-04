#!/usr/bin/env python3

import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest


def load_checker():
    path = Path(__file__).with_name("check_pipeline_core_compile_fail.py")
    spec = importlib.util.spec_from_file_location("check_pipeline_core_compile_fail", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = load_checker()


class PipelineCoreCompileFailTests(unittest.TestCase):
    def test_required_artifact_errors_accept(self) -> None:
        diagnostic = """
error[E0599]: no method named `checked` found for enum `SemanticRejection`
error[E0308]: mismatched types
expected `NamedRoots`, found `ForwardNodeIndex`
"""

        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 101, "", diagnostic)

        checker.validate(runner=runner, environment={})

    def test_compile_success_rejects(self) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "", "")

        with self.assertRaisesRegex(
            checker.PipelineArtifactCompileFailure, "artifact misuse compiled"
        ):
            checker.validate(runner=runner, environment={})

    def test_unrelated_failure_rejects(self) -> None:
        def runner(command, **_kwargs):
            return subprocess.CompletedProcess(command, 101, "", "network failure")

        with self.assertRaisesRegex(
            checker.PipelineArtifactCompileFailure, "lacks the required text"
        ):
            checker.validate(runner=runner, environment={})


if __name__ == "__main__":
    unittest.main()
