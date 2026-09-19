#!/usr/bin/env python3
"""Unit and mutation tests for the chelis#1284 completion oracle."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_nonnumeric_lowering_oracle as oracle  # noqa: E402


class ManifestTests(unittest.TestCase):
    def test_manifest_names_exact_always_run_legs(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs("/chosen/python")],
            [
                "oracle self-tests and standing mutations",
                "direct IR lowering evaluation eager semantics and AD",
                "current WireDag identities",
                "compiled C exact semantics and ownership forms",
                "HIP structural kernels and target admission",
                "Metal typed target and emitter rejection",
            ],
        )

    def test_hardware_matrix_is_separate_from_the_always_run_oracle(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("scripts/hip_test.py", leg.argv, leg.name)
            self.assertNotIn("logical_comparison_where_gpu", leg.argv, leg.name)

    def test_self_test_uses_selected_python(self) -> None:
        self.assertEqual(
            oracle.oracle_legs("/chosen/python")[0].argv,
            ("/chosen/python", "scripts/test_dtype_nonnumeric_lowering_oracle.py"),
        )


class SourceMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tempdir.name)
        for contract in oracle.source_contracts():
            source = oracle.REPO_ROOT / contract.path
            target = self.repo / contract.path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def mutate(self, path: str, old: str, new: str) -> None:
        target = self.repo / path
        text = target.read_text()
        self.assertIn(old, text)
        target.write_text(text.replace(old, new, 1))

    def test_unmodified_contracts_pass(self) -> None:
        oracle.validate_source_contracts(self.repo)

    def test_where_storage_carrier_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/emit.rs",
            'Prim::F32 => "uint32_t",',
            'Prim::F32 => "float",',
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP stored-bit where"):
            oracle.validate_source_contracts(self.repo)

    def test_reduced_float_decode_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/emit.rs",
            '"chelis_f16_to_f32(a[idx_a])"',
            '"a[idx_a]"',
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP semantic comparison"):
            oracle.validate_source_contracts(self.repo)

    def test_metal_issue_authority_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-compiler-api/src/compiler.rs",
            "chelis_types::unimplemented_rejection!(\n                    1284,",
            "chelis_types::unimplemented_rejection!(\n                    9999,",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "Metal target disposition"):
            oracle.validate_source_contracts(self.repo)


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_first_failure(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(9, []),
        ]
        with self.assertRaisesRegex(SystemExit, "direct IR"):
            oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_from_repo_root(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, len(oracle.oracle_legs(sys.executable)))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
