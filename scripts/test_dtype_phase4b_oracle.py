#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_phase4b_oracle as oracle  # noqa: E402


CONTRACT_FILES = tuple(Path(relative) for relative in oracle.CONTRACT_FILES)


class ContractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        for relative in CONTRACT_FILES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, destination)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def replace(self, relative: Path, old: str, new: str) -> None:
        path = self.root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(old, text)
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def assert_contract_fails(self, message: str) -> None:
        with self.assertRaisesRegex(oracle.OracleError, message):
            oracle.validate_contract(self.root)

    def test_repository_contract_passes(self) -> None:
        oracle.validate_contract(REPO_ROOT)

    def test_missing_operation_atom_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-14]** `prod_reduce",
            "> **[05-OP-114]** `prod_reduce",
        )
        self.assert_contract_fails("OP-14")

    def test_cast_round_cannot_replace_the_saturating_rung(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-23]** `cast_saturate",
            "> **[05-OP-23]** `cast_round",
        )
        self.assert_contract_fails("cast_round")

    def test_to_string_requires_a_numbered_spec_atom(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-25]** `to_string",
            "> **[05-OP-125]** `to_string",
        )
        self.assert_contract_fails("OP-25")

    def test_to_string_domain_must_reject_unlisted_value_types(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Every other value type is a type error",
            "Every other value type is accepted",
        )
        self.assert_contract_fails("OP-25.*Every other value type")

    def test_to_string_raw_string_display_is_explicitly_non_injective(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "non-injective display form, not a serialization",
            "structure-preserving serialization",
        )
        self.assert_contract_fails("OP-25.*non-injective")

    def test_to_string_seed_cells_require_the_normative_atom(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Supported` exactly for [05-OP-25]'s scalar, tensor, and "
            "recursively admitted List domains",
            "Supported` by a future atom",
        )
        self.assert_contract_fails("to_string semantic authority")

    def test_extrema_tie_and_nan_adjoint_body_is_required(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "equal positive or negative infinities. The tie count `k` is",
            "equal finite values. The tie count `k` is",
        )
        self.assert_contract_fails("OP-12.*infinities")

    def test_additive_extrema_adjoint_contradiction_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> zero. Integer operands are forward-only and `grad` rejects them.",
            "> zero. Integer operands are forward-only and `grad` rejects them.\n"
            "> A backend MAY instead route the full non-NaN cotangent to only the "
            "last\n"
            "> element equal to the selected maximum.",
        )
        self.assert_contract_fails("frozen normative atom 05-OP-12")

    def test_legacy_bool_arithmetic_alias_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Logical operations do not alias arithmetic primitives.",
            "Logical operations do not alias arithmetic primitives. `and` is `mul`, "
            "`or` is `max_elem`, and `not` is `neg` on bool values.",
        )
        self.assert_contract_fails("frozen logical builtin contract")

    def test_product_tree_body_is_required(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`(p0 * p1) * (p2 * p3)`",
            "an implementation-defined combination",
        )
        self.assert_contract_fails("OP-14.*p0")

    def test_classification_reads_stored_width(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "It reads the finalized stored value\n> without conversion",
            "It widens the finalized value\n> before classification",
        )
        self.assert_contract_fails("OP-20.*stored value")

    def test_named_cast_ladder_forbids_a_rounding_cast(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "A rounding cast is not a separate\n> operation",
            "A rounding cast is a separate\n> operation",
        )
        self.assert_contract_fails("04-NUM-16.*rounding cast")

    def test_numeric_surface_key_drift_fails(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "(`BuiltinId`, `SurfaceClass`, operand `Prim`,",
            "(`BuiltinId`, `NumericSurface`, operand `Prim`,",
        )
        self.assert_contract_fails("NumericSurface")

    def test_missing_sibling_registry_key_fails(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "(`BuiltinId`, `SiblingDomain`, `SiblingCaseId`, `SemanticParams`)",
            "(`BuiltinId`, `SiblingDomain`, `SemanticParams`)",
        )
        self.assert_contract_fails("sibling registry key")

    def test_sibling_backend_product_is_required(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Every sibling `Supported` row expands across\nthe same exact backend set",
            "A sibling `Supported` row may omit backends from",
        )
        self.assert_contract_fails("sibling backend product")

    def test_to_string_scalar_case_is_required(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "`ToStringScalar`, `ToStringTensor`, and `ToStringList`",
            "`ToStringTensor` and `ToStringList`",
        )
        self.assert_contract_fails("to_string cases")

    def test_external_target_authority_is_required(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "(`ExternalCallableFamily`,\n`CanonicalCallableId`, "
            "`ExternalTargetContext`)",
            "(`ExternalCallableFamily`, `CanonicalCallableId`)",
        )
        self.assert_contract_fails("external target key")

    def test_additive_missing_row_default_fails(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "duplicate or missing expanded rows fail",
            "duplicate or missing expanded rows fail. The implementation MAY "
            "nevertheless treat an absent semantic row as `Supported` using its "
            "backend's default kernel",
        )
        self.assert_contract_fails("frozen capability schema")

    def test_effect_registry_covers_every_fixed_effect(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "`Random | Accum | Io | Test | Resource(ResourceId)`",
            "`Random | Accum | Io | Resource(ResourceId)`",
        )
        self.assert_contract_fails("frozen capability schema")

    def test_effect_registry_has_no_default_disposition(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "There is no\nmissing-row, wildcard, or default disposition.",
            "A missing effect row defaults to `Implemented`.",
        )
        self.assert_contract_fails("frozen capability schema")

    def test_reduction_rows_cannot_cite_the_tracking_hub(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #1281, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("reduction implementation owner")

    def test_logical_rows_cannot_cite_the_tracking_hub(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #1284, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("logical implementation owner")

    def test_product_rows_retain_their_concrete_owner(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #170, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("product implementation owner")

    def test_to_string_checker_narrowing_retains_its_concrete_owner(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "[#1282] owns aligning the pre-table",
            "[#729] owns aligning the pre-table",
        )
        self.assert_contract_fails("to_string checker owner")

    def test_to_string_gap_note_cites_both_implementation_owners(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Not fully implemented; see chelis#1282 and chelis#1059",
            "Not fully implemented; see chelis#1059",
        )
        self.assert_contract_fails("to_string implementation owners")

    def test_loud_unsupported_must_consume_external_target_key(self) -> None:
        self.replace(
            Path("spec/design/loud_unsupported.md"),
            "(ExternalCallableFamily, CanonicalCallableId, "
            "ExternalTargetContext)",
            "(ExternalCallableFamily, CanonicalCallableId)",
        )
        self.assert_contract_fails("loud external target key")

    def test_provenance_must_consume_stdlib_derivation(self) -> None:
        self.replace(
            Path("spec/design/spec_provenance.md"),
            "generated transitive dependency closure over the checked body",
            "manually recorded support claim",
        )
        self.assert_contract_fails("provenance stdlib derivation")

    def test_final_phase_oracle_must_nest_phase4b(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "It invokes the 4B, 4C, and 4D oracles",
            "It invokes the 4C and 4D oracles",
        )
        self.assert_contract_fails("dtype-plan final nesting")

    def test_roadmap_must_keep_concrete_reduction_owners(self) -> None:
        self.replace(
            Path("spec/design/remediation_roadmap.md"),
            "[#1281] owns the remaining reduction rows",
            "[#729] owns the remaining reduction rows",
        )
        self.assert_contract_fails("roadmap reduction owner")

    def test_additive_parent_absorption_clause_fails(self) -> None:
        self.replace(
            Path("spec/design/remediation_roadmap.md"),
            "([#170] owns the product-tree/backend rows; [#1281] owns the "
            "remaining reduction rows)",
            "([#170] owns the product-tree/backend rows; [#1281] owns the "
            "remaining reduction rows; the parent [#729] MAY silently absorb "
            "and close either child's work without a separate receipt)",
        )
        self.assert_contract_fails("frozen roadmap ownership")

    def test_status_must_keep_external_execution_authority(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "total external target-disposition registry",
            "semantic registry alone",
        )
        self.assert_contract_fails("status external target authority")


class RunnerTests(unittest.TestCase):
    @mock.patch.object(oracle.subprocess, "run")
    def test_runner_checks_generated_registry_from_repository_root(
        self, run: mock.Mock
    ) -> None:
        run.return_value = subprocess.CompletedProcess([], 0)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            oracle.run_oracle(sys.executable, REPO_ROOT)
        run.assert_called_once_with(
            (
                sys.executable,
                "scripts/generate_rejection_registries.py",
                "--check",
            ),
            cwd=REPO_ROOT,
            check=True,
        )
        self.assertEqual(output.getvalue().splitlines()[-1], oracle.PASS_LINE)

    @mock.patch.object(oracle.subprocess, "run")
    def test_registry_disagreement_fails_the_oracle(self, run: mock.Mock) -> None:
        run.side_effect = subprocess.CalledProcessError(1, [])
        with self.assertRaisesRegex(SystemExit, "rejection registry disagreement"):
            oracle.run_oracle(sys.executable, REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
