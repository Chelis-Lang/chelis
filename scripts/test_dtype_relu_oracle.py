#!/usr/bin/env python3
"""Unit and mutation tests for the chelis#1313 completion oracle."""

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

import dtype_relu_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_every_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs("/chosen/python")],
            [
                "oracle self-tests and standing mutations",
                "sealed typed scalar/tensor semantics",
                "dedicated IR lowering evaluation and AD",
                "wire and target contracts",
                "compiled C exact bits and scalar selector",
                "HIP all-width structural kernels",
                "Metal admitted-width structural kernels",
                "numeric surface authority registrations",
                "generated rejection registry agreement",
            ],
        )

    def test_manifest_does_not_claim_hardware_execution(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("gpu_correctness", leg.argv, leg.name)


class SourceContractMutationTests(unittest.TestCase):
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

    def test_unmodified_contract_fixture_passes(self) -> None:
        oracle.validate_source_contracts(self.repo)

    def test_surrogate_lowering_mutation_fails(self) -> None:
        path = self.repo / "crates/chelis-ir/src/tier2.rs"
        source = path.read_text()
        path.write_text(
            source.replace(
                "add_synth(dag, RiscOp::Relu, vec![x], ty.clone(), parent_span)",
                "lower_max_elem(dag, x, zero, ty, parent_span)",
                1,
            )
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "dedicated lowering"):
            oracle.validate_source_contracts(self.repo)

    def test_gradient_alias_mutation_fails(self) -> None:
        path = self.repo / "crates/chelis-ir/src/grad.rs"
        source = path.read_text()
        path.write_text(
            source.replace(
                "let dx = dag.add_node(RiscOp::ReluAdjoint, vec![x, g]",
                "let dx = dag.add_node(RiscOp::MaxElem, vec![x, g]",
                1,
            )
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "dedicated adjoint"):
            oracle.validate_source_contracts(self.repo)

    def test_scalar_fmax_surrogate_mutation_fails(self) -> None:
        path = self.repo / "crates/chelis-backend-c/src/host_emit.rs"
        source = path.read_text()
        path.write_text(
            source.replace(
                '"    return x < {} ? {} : x;"',
                '"    return fmax(x, {});"',
                1,
            )
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "scalar stored-value"):
            oracle.validate_source_contracts(self.repo)

    def test_relu_registration_mutation_fails(self) -> None:
        path = self.repo / "crates/chelis-cli/tests/capacity_census_tripwire.rs"
        source = path.read_text()
        path.write_text(
            source.replace(
                "[compiler-risc-numeric] relu_adjoint(input: &tensor[D, p], cotangent:",
                "[compiler-risc-numeric] missing_relu_adjoint(input: &tensor[D, p], cotangent:",
            )
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "numeric authority"):
            oracle.validate_source_contracts(self.repo)

    def test_closed_issue_receipt_mutation_fails(self) -> None:
        path = self.repo / "crates/chelis-compiler-api/src/compiler.rs"
        path.write_text(path.read_text() + "\n// unimplemented chelis#1313\n")
        with self.assertRaisesRegex(oracle.OracleFailure, "stale closed-issue"):
            oracle.validate_source_contracts(self.repo)


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_first_failed_leg(self, run: mock.Mock) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "sealed typed"):
            oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, 2)

    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_uses_repository_root(self, run: mock.Mock) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle(sys.executable)
        self.assertEqual(run.call_count, len(oracle.oracle_legs(sys.executable)))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])


if __name__ == "__main__":
    unittest.main()
