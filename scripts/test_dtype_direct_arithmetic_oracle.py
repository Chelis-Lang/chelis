#!/usr/bin/env python3
"""Unit and mutation tests for the chelis#1306 completion oracle."""

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

import dtype_direct_arithmetic_oracle as oracle  # noqa: E402


class CommandManifestTests(unittest.TestCase):
    def test_manifest_names_each_acceptance_leg_once(self) -> None:
        self.assertEqual(
            [leg.name for leg in oracle.oracle_legs("/chosen/python")],
            [
                "oracle self-tests and standing mutations",
                "exact typed scalar and tensor kernels",
                "direct IR lowering, evaluation, and AD",
                "direct constant folding",
                "exact current WireDag decoder contract",
                "wire and target disposition contracts",
                "compiled C exact-value and trap behavior",
                "HIP structural contracts",
                "downstream exhaustive consumers",
            ],
        )

    def test_manifest_never_claims_ignored_hardware_execution(self) -> None:
        for leg in oracle.oracle_legs(sys.executable):
            self.assertNotIn("--ignored", leg.argv, leg.name)
            self.assertNotIn("scripts/hip_test.py", leg.argv, leg.name)

    def test_self_test_uses_the_oracle_interpreter(self) -> None:
        leg = oracle.oracle_legs("/chosen/python")[0]
        self.assertEqual(
            leg.argv,
            ("/chosen/python", "scripts/test_dtype_direct_arithmetic_oracle.py"),
        )

    def test_compiled_c_leg_runs_the_complete_direct_fixture_slice(self) -> None:
        leg = next(
            leg
            for leg in oracle.oracle_legs(sys.executable)
            if leg.name == "compiled C exact-value and trap behavior"
        )
        self.assertIn("exec_compile", leg.argv)
        self.assertEqual(leg.argv[-1], "direct_")

    def test_wire_decoder_leg_runs_the_review_regression_binary(self) -> None:
        leg = next(
            leg
            for leg in oracle.oracle_legs(sys.executable)
            if leg.name == "exact current WireDag decoder contract"
        )
        self.assertIn("--test", leg.argv)
        self.assertIn("wire_dag_v6_direct_arithmetic", leg.argv)
        self.assertNotIn("-E", leg.argv)

    def test_hip_leg_is_structural_only(self) -> None:
        leg = next(
            leg
            for leg in oracle.oracle_legs(sys.executable)
            if leg.name == "HIP structural contracts"
        )
        self.assertIn("codegen_structure", leg.argv)
        self.assertNotIn("gpu_correctness", leg.argv)


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

    def mutate(
        self, path: str, old: str, new: str, *, all_matches: bool = False
    ) -> None:
        target = self.repo / path
        source = target.read_text()
        self.assertIn(old, source, f"standing mutation precondition for {path}")
        count = -1 if all_matches else 1
        target.write_text(source.replace(old, new, count))

    def test_unmodified_contract_fixture_passes(self) -> None:
        oracle.validate_source_contracts(self.repo)

    def test_sub_surrogate_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-ir/src/tier2.rs",
            "add_synth(dag, RiscOp::Sub, vec![a, b], ty.clone(), parent_span)",
            "lower_add(dag, a, lower_neg(dag, b, ty, parent_span), ty, parent_span)",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "direct Sub lowering"):
            oracle.validate_source_contracts(self.repo)

    def test_min_surrogate_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-ir/src/tier2.rs",
            "add_synth(dag, RiscOp::MinElem, vec![a, b], ty.clone(), parent_span)",
            "add_synth(dag, RiscOp::MaxElem, vec![a, b], ty.clone(), parent_span)",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "direct MinElem lowering"):
            oracle.validate_source_contracts(self.repo)

    def test_c_checked_subtraction_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-c/src/emit.rs",
            "chelis_int_checked_sub",
            "chelis_int_checked_add",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "C checked subtraction"):
            oracle.validate_source_contracts(self.repo)

    def test_evaluator_canonical_nan_removal_fails(self) -> None:
        self.mutate(
            "crates/chelis-types/src/dtype_semantics.rs",
            "fn canonicalize_subtraction_f32(value: f32) -> f32 {",
            "fn preserve_subtraction_f32_nan(value: f32) -> f32 {",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "evaluator canonical subtraction NaNs"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_evaluator_canonical_nan_scope_broadening_fails(self) -> None:
        self.mutate(
            "crates/chelis-types/src/dtype_semantics.rs",
            "FloatBinOp::Sub => canonicalize_subtraction_f32(value),",
            "_ => canonicalize_subtraction_f32(value),",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "evaluator canonical subtraction NaNs"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_c_wide_nan_canonicalization_removal_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-c/src/emit.rs",
            'Prim::F32 => Some("chelis_f32_from_bits(UINT32_C(0x7fc00000))"),',
            "Prim::F32 => None,",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "C subtraction canonical NaNs"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_c_reduced_nan_canonicalization_removal_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-c/src/emit.rs",
            'Prim::F16 => "UINT16_C(0x7e00)",',
            'Prim::F16 => "UINT16_C(0xfe00)",',
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "C subtraction canonical NaNs"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_c_first_operand_extrema_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-c/src/emit.rs",
            'let comparison = if func.contains("max") { ">=" } else { "<=" };',
            'let comparison = if func.contains("max") { ">" } else { "<" };',
            all_matches=True,
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "C stored-bit extrema"):
            oracle.validate_source_contracts(self.repo)

    def test_c_bool_extrema_float_classification_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-c/src/emit.rs",
            "if ty.precision.is_integer() || matches!(ty.precision, Prim::Bool) {",
            "if ty.precision.is_integer() {",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "C Bool extrema avoid floating classification"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_hip_first_operand_extrema_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/kernels.rs",
            'let comparison = if is_max { ">=" } else { "<=" };',
            'let comparison = if is_max { ">" } else { "<" };',
            all_matches=True,
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP stored-bit extrema"):
            oracle.validate_source_contracts(self.repo)

    def test_hip_ties_away_rounding_restoration_fails(self) -> None:
        path = "crates/chelis-backend-hip/src/kernels.rs"
        target = self.repo / path
        target.write_text(
            target.read_text()
            + "\n// obsolete: chelis_u32 round_bit = 0x00008000u;\n"
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP narrow direct arithmetic"):
            oracle.validate_source_contracts(self.repo)

    def test_hip_bf16_nan_payload_restoration_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/kernels.rs",
            "return (chelis_u16)0x7fc0u;",
            "return (chelis_u16)((bits >> 16) | 0x0040u);",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP narrow direct arithmetic"):
            oracle.validate_source_contracts(self.repo)

    def test_hip_f16_nan_payload_restoration_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/kernels.rs",
            "return (chelis_u16)(mantissa == 0 ? ((sign >> 16) | 0x7c00u) : 0x7e00u);",
            "return (chelis_u16)((sign >> 16) | 0x7c00u | 0x0200u | (mantissa >> 13));",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HIP narrow direct arithmetic"):
            oracle.validate_source_contracts(self.repo)

    def test_hip_wide_nan_canonicalization_removal_fails(self) -> None:
        self.mutate(
            "crates/chelis-backend-hip/src/kernels.rs",
            "__int_as_float(0x7fc00000)",
            "a[idx_a] - b[idx_b]",
            all_matches=True,
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "HIP wide subtraction canonical NaNs"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_hip_reduced_float_matrix_admission_removal_fails(self) -> None:
        self.mutate(
            "spec/04-type-system.md",
            "[05-OP-40] `max_elem`/`min_elem` and their adjoints, [05-OP-41] `sub`",
            "[05-OP-43] `Relu`/`ReluAdjoint`",
            all_matches=True,
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "HIP reduced-float target matrix"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_target_authority_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-compiler-api/src/compiler.rs",
            "chelis_types::unimplemented_rejection!(\n                    2338,\n                    \"the Metal direct-subtraction/extrema",
            "chelis_types::unimplemented_rejection!(\n                    9999,\n                    \"the Metal direct-subtraction/extrema",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "target dispositions"):
            oracle.validate_source_contracts(self.repo)

    def test_retired_hip_rejection_restoration_fails(self) -> None:
        path = "crates/chelis-compiler-api/src/compiler.rs"
        target = self.repo / path
        target.write_text(
            target.read_text()
            + "\n// checked signed-integer subtraction needs an exact HIP overflow-trap channel\n"
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "target dispositions"):
            oracle.validate_source_contracts(self.repo)

    def test_wire_identity_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-compiler-api/src/schema.rs",
            "pub const WIRE_DAG_SCHEMA_VERSION: u32 = 17;",
            "pub const WIRE_DAG_SCHEMA_VERSION: u32 = 16;",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "current WireDag identities"):
            oracle.validate_source_contracts(self.repo)

    def test_surface_surrogate_narrative_mutation_fails(self) -> None:
        self.mutate(
            "docs/CHELIS_SURFACE.md",
            "complete `g` to the exact operand selected by [05-OP-40]",
            "(g*(x>=y), g*(x<y))",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current direct arithmetic surface narrative"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_required_surface_narrative_deletion_fails(self) -> None:
        self.mutate(
            "docs/CHELIS_SURFACE.md",
            "Tier-1 DAG:   add sub mul div floor_div trunc_div max_elem min_elem cmplt",
            "Tier-1 DAG:   add mul div floor_div trunc_div max_elem cmplt",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current direct arithmetic surface narrative"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_obsolete_surface_narrative_restoration_fails_independently(self) -> None:
        current = "Tier-1 DAG:   add sub mul div floor_div trunc_div max_elem min_elem cmplt"
        self.mutate(
            "docs/CHELIS_SURFACE.md",
            current,
            current + "\n| `sub` | `add(a, neg(b))` |",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current direct arithmetic surface narrative"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_builtin_direct_identity_grouping_mutation_fails(self) -> None:
        self.mutate(
            "crates/chelis-types/src/builtins.rs",
            'tensor_binop("add", &mut env, &mut vg);\n    tensor_binop("sub", &mut env, &mut vg);\n    tensor_binop("mul", &mut env, &mut vg);',
            'tensor_binop("add", &mut env, &mut vg);\n    tensor_binop("mul", &mut env, &mut vg);\n    tensor_binop("sub", &mut env, &mut vg);',
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current builtin direct arithmetic grouping"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_required_fused_extrema_narrative_deletion_fails(self) -> None:
        self.mutate(
            "crates/chelis-ir/tests/fusion_adversarial.rs",
            "ADV-10: MaxElem in fused chain preserves exact selected-operand semantics",
            "ADV-10: MaxElem in fused chain",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current fused direct-extrema narrative"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_obsolete_fused_extrema_narrative_restoration_fails(self) -> None:
        current = "ADV-10: MaxElem in fused chain preserves exact selected-operand semantics"
        self.mutate(
            "crates/chelis-ir/tests/fusion_adversarial.rs",
            current,
            current + "\n// ADV-10: MaxElem in fused chain uses fmaxf",
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "current fused direct-extrema narrative"
        ):
            oracle.validate_source_contracts(self.repo)

    def test_retired_timing_identity_restoration_fails(self) -> None:
        current = '"chelis-ir::tier2::tests::or_produces_max_elem":'
        self.mutate(
            "scripts/test_timing_baseline.json",
            current,
            '"chelis-ir::lower::tests::lower_sub_decomposes": 0.001,\n  ' + current,
        )
        with self.assertRaisesRegex(
            oracle.OracleFailure, "retired direct-arithmetic timing identities"
        ):
            oracle.validate_source_contracts(self.repo)


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle, "validate_source_contracts")
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_executes_every_leg_from_repository_root(
        self, run: mock.Mock, validate: mock.Mock
    ) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        oracle.run_oracle("/chosen/python")
        validate.assert_called_once_with(oracle.REPO_ROOT)
        self.assertEqual(run.call_count, len(oracle.oracle_legs("/chosen/python")))
        for call in run.call_args_list:
            self.assertEqual(call.kwargs["cwd"], oracle.REPO_ROOT)
            self.assertTrue(call.kwargs["check"])

    @mock.patch.object(oracle, "validate_source_contracts")
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_stops_at_first_failed_leg(
        self, run: mock.Mock, validate: mock.Mock
    ) -> None:
        run.side_effect = [
            subprocess.CompletedProcess([], 0),
            subprocess.CalledProcessError(7, []),
        ]
        with self.assertRaisesRegex(SystemExit, "typed scalar and tensor"):
            oracle.run_oracle("/chosen/python")
        validate.assert_called_once()
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
