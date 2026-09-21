#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import json
import os
import re
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
import phase4b_change_report as change_report  # noqa: E402


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

    def assert_changed_contract_requires_acknowledgement(self, kind: str, identity: str) -> None:
        # Keep the original additive-prose mutations as enforcing review cues.
        # Required-clause removal controls continue to call validate_contract.
        def reader(root):
            return lambda relative: ((REPO_ROOT if relative == change_report.ORACLE else root) / relative).read_text(encoding="utf-8")
        before = change_report.snapshot(reader(REPO_ROOT))
        after = change_report.snapshot(reader(self.root))
        changes = change_report.compare_snapshots(before, after)
        self.assertIn((kind, identity), {(row["kind"], row["identity"]) for row in changes})
        errors = change_report.identity_acknowledgement_violations({"changes": changes}, [])
        self.assertTrue(any(change_report.acknowledgement_line(kind, identity) in error for error in errors), errors)

    def repository_atom(self, atom: str) -> str:
        text = (REPO_ROOT / "spec/05-risc-primitives.md").read_text(
            encoding="utf-8"
        )
        block = oracle.atom_blocks(text)[atom]
        registry = oracle.OP_MANIFEST_REGISTRY_FILES.get(atom)
        if registry is not None:
            block = (
                block
                + "\n"
                + (REPO_ROOT / registry).read_text(encoding="utf-8")
            )
        return oracle.normalize_atom_body(block)

    def test_repository_contract_passes(self) -> None:
        oracle.validate_contract(REPO_ROOT)

    def test_wire_binding_decisions_have_positive_and_negative_freeze_controls(self) -> None:
        cases = (
            ("spec/10-serialization.md", "Schema version 15 is explicitly\npresent", "wire v15 presence"),
            ("spec/10-serialization.md", "`schema_version: 3`", "execution v3 exactness"),
            ("spec/10-serialization.md", "f64: 16; f32: 8; f16: 4; bf16: 4", "wire IEEE bit widths"),
            ("spec/10-serialization.md", "No codec normalizes a NaN payload or a signed zero.", "wire bit preservation"),
            ("spec/10-serialization.md", "A raw source DTO is not an admitted executable AST.", "wire raw-source admission"),
            ("spec/10-serialization.md", "A reference is resolved only in its declared owner and namespace.", "wire reference scope"),
            ("spec/10-serialization.md", "Every requirement uses the exact\n`NonnegativeExtent` adapter over a nonnegative `int64`", "wire literal-witness requirement carrier"),
            ("spec/10-serialization.md", "`WireDagNode.shape_deps` contains exact u64 node\nreferences to strictly earlier nodes", "wire shape-dependency references"),
            ("spec/10-serialization.md", "`shape_deps`, `span_id` (explicitly null when absent), and `merged_spans` are\nmandatory fields", "wire mandatory invocation fields"),
            ("spec/10-serialization.md", "Bounds alone never establish transport authority.", "wire report numeric authority"),
            ("spec/04-type-system.md", "untyped_nodes = total_nodes - typed_nodes", "fitness counter consistency"),
            ("spec/11-ffi.md", "Dynamic Python object types do not establish nonnumeric capacity.", "binding dynamic capacity"),
            ("spec/11-ffi.md", "DLPack keywords are validated, never ignored.", "binding DLPack keyword admission"),
            ("spec/design/dtype_semantics.md", "No partial WireDag v9 is published.", "wire atomic cutover"),
        )
        for relative, required, label in cases:
            with self.subTest(label=label):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                self.assertTrue(required in original, f"missing decided contract: {label}")
                path.write_text(original.replace(required, "REMOVED CONTRACT", 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(label)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_python_shape_registry_is_exact_and_not_the_c_shape_atom(self) -> None:
        self.assertEqual(
            oracle.EXPECTED_OP_MANIFESTS["05-OP-45"],
            ("| full tensor shape | `chelis_python::NativeTensor::shape(self: &Self) -> Vec<i64>` |",),
        )
        self.replace(
            Path("spec/registry/python_tensor_metadata.md"),
            "-> Vec<i64>",
            "-> Vec<usize>",
        )
        self.assert_contract_fails("05-OP-45")

    # The whole-file digest tests these replaced now live in
    # FrozenContractChangeTests, which runs the same mutations against the
    # merge-base acknowledgement gate.

    def test_the_acknowledgement_gate_cannot_be_restated_as_a_digest(self) -> None:
        # The Phase 4 handoff region digest moved when the plan's oracle
        # description was rewritten. These three mutations are what defends the
        # new text, so it rests on required literals rather than only on a
        # re-hash.
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "The additive-contradiction leg is an acknowledgement, not a "
            "whole-file digest.",
            "The additive-contradiction leg is a whole-file digest.",
        )
        self.assert_contract_fails(
            "Phase 4B acknowledgement gate replaces whole-file digests"
        )

    def test_the_acknowledgement_cannot_become_a_blanket_declaration(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "requires each changed file to be named in the pull request body",
            "requires the pull request to declare that contract files changed",
        )
        self.assert_contract_fails(
            "Phase 4B acknowledgement is per changed file"
        )

    def test_a_stale_acknowledgement_cannot_be_made_advisory(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "An unacknowledged\nchange and an acknowledgement naming an "
            "unchanged file both fail\n`--require-acknowledgement`, which the "
            "dedicated `PR Contract Acknowledgements`\ncheck applies through "
            "`phase4b_change_report.py`. The full Phase 4B oracle runs\n"
            "independently in Docs.",
            "An unacknowledged change fails `--require-acknowledgement`; an "
            "acknowledgement naming an unchanged file is tolerated.",
        )
        self.assert_contract_fails(
            "Phase 4B acknowledgement enforcing mode"
        )

    def test_relu_device_completion_cannot_regress_to_issue_receipts(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "No backend cell cites [#1313] after it closes",
            "Every backend cell cites [#1313] after it closes",
        )
        self.assert_contract_fails("ReLU closed-issue receipt removal")

    def test_relu_child_oracle_cannot_drop_the_exact_success_line(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "DTYPE RELU ORACLE:\nPASS",
            "DTYPE RELU ORACLE: MAYBE",
        )
        self.assert_contract_fails("ReLU child oracle success line")

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

    def test_count_requires_a_numbered_spec_atom(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-29]** `count",
            "> **[05-OP-129]** `count",
        )
        self.assert_contract_fails("OP-29")

    def test_count_domain_and_result_dtype_are_frozen(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "admits exactly a `bool` tensor operand",
            "admits every numeric tensor operand",
        )
        self.assert_contract_fails("OP-29.*bool")

    def test_count_cannot_be_defined_as_cast_plus_sum(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "is a\n> dedicated reduction and is not a `cast` plus `sum` lowering",
            "lowers to `sum(cast(x, i64), axis)`",
        )
        self.assert_contract_fails("OP-29.*dedicated reduction")

    def test_every_phase4b_operation_atom_has_its_exact_heading(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        for number, heading in oracle.EXPECTED_PHASE4B_OP_HEADINGS.items():
            with self.subTest(number=number, heading=heading):
                original = path.read_text(encoding="utf-8")
                atom = f"05-OP-{number}"
                block = oracle.strict_atom_block(original, atom)
                first_line = block.splitlines()[0]
                self.assertIn(heading, first_line)
                path.write_text(
                    original.replace(first_line, first_line.replace(heading, "wrong_heading"), 1),
                    encoding="utf-8",
                )
                try:
                    self.assert_contract_fails(f"{atom}.*must begin")
                finally:
                    path.write_text(original, encoding="utf-8")

    # test_top_level_value_scope_is_frozen_in_both_owning_chapters moved to
    # FrozenContractChangeTests: neither clause sits inside a frozen region, so
    # the acknowledgement gate is the leg that catches those mutations.

    def test_exact_read_atom_freezes_json_and_csv_numeric_boundaries(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`csv_int` | `(List[Dict[string,string]], i64, string) -> i64`",
                "`csv_int` | `(List[Dict[string,f64]], i32, string) -> i32`",
                "OP-3.*csv_int",
            ),
            (
                "It never\n> truncates or rounds a float\n> into an integer",
                "It truncates float variants into i64",
                "OP-3.*never truncates",
            ),
            (
                "no JSON variant or default cell is fabricated",
                "a missing cell defaults to zero",
                "OP-3.*default cell",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_json_numeric_construction_never_implicitly_widens(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`JsonFloat(value)` accepts exactly f64 and\n> `JsonInt(value)` accepts exactly i64",
            "`JsonFloat(value)` accepts any float and widens it to f64",
        )
        self.assert_contract_fails("OP-4.*exactly f64")

    def test_numeric_serialization_preserves_exact_json_variants(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "emits a stored `JsonInt` i64\n> as its exact decimal digits",
                "emits every stored number through f64",
                "OP-5.*exact decimal digits",
            ),
            (
                "A non-finite `JsonFloat` is a loud serialization error",
                "A non-finite JsonFloat serializes as null",
                "OP-5.*non-finite",
            ),
            (
                "`to_csv` accepts only the\n> text-table type `List[Dict[string,string]]`",
                "to_csv infers numeric cell types",
                "OP-5.*text-table",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_cast_trunc_is_explicit_checked_and_structurally_rejected_by_ad(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "if that truncated integer is outside the\n> target range it traps `overflow`",
                "an out-of-range result wraps",
                "OP-6.*traps `overflow`",
            ),
            (
                "A **non-finite** source\n> (`NaN`, `±inf`) traps `Domain`",
                "a non-finite source becomes zero",
                "OP-6.*traps `Domain`",
            ),
            (
                "a gradient goal through it is a clean\n> error, never a silent zero",
                "its adjoint silently returns zero",
                "OP-6.*never a silent zero",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_round_to_all_float_widths_and_own_width_finalization_are_frozen(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| `f16` | exact decimal rounding of the exact binary value, one final "
                "rounding to f16 | `f16` |",
                "| `f16` | type error | n/a |",
                "f16",
            ),
            (
                "nearest to the EXACT binary value of `x`",
                "nearest to the value after widening `x` through f64",
                "EXACT binary value",
            ),
            (
                "finalized ONCE to the operand's own\n> storage width",
                "finalized to f64 for every operand dtype",
                "own storage width",
            ),
            (
                "| integer, bool, tensor | type error | n/a |",
                "| integer, bool, tensor | convert to f64 | `f64` |",
                "type error",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"OP-1.*{message}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_round_to_structurally_rejects_grad_instead_of_returning_zero(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "a differentiated graph containing it is structurally rejected\n"
            "> with `AdRejectionReason::PiecewiseConstant`, rather than receiving a "
            "silent\n> zero cotangent",
            "its derivative is defined as a silent zero cotangent",
        )
        self.assert_contract_fails("OP-1.*PiecewiseConstant")

    def test_integer_form_json_overflow_never_falls_back_to_jnum(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "An integer-form token outside i64 range SHALL ingest as\n"
            "> `JsonBigInt` carrying the token's exact decimal spelling; ingestion"
            " never\n"
            "> selects a lossy float image for an integer-form token.",
            "An integer-form token outside i64 range falls back to `JNum`.",
        )
        self.assert_contract_fails("OP-2.*JsonBigInt")

    def test_uniform_like_uses_one_common_float_dtype_and_own_width_fma(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "requires `low` and `high`\n> to have that same dtype `p`",
                "permits low and high to widen from any float dtype",
                "OP-8.*same dtype",
            ),
            (
                "For `p = f64`, the element is the one f64 fused multiply-add",
                "For p = f64, the element is computed by an f32 fused multiply-add",
                "OP-8.*f64.*fused multiply-add",
            ),
            (
                "For `p = f32`, it is the one f32 fused\n> multiply-add",
                "For p = f32, the element is computed by an f64 fused multiply-add",
                "OP-8.*f32.*fused multiply-add",
            ),
            (
                "For `p = f16` or `bf16`,\n"
                "> the stored bounds widen exactly to f32, their difference and the fused\n"
                "> multiply-add execute once in f32 using `round_f32(u)`, and the result narrows\n"
                "> exactly once to `p`",
                "For p = f16 or bf16, the result may round through f64 and f32",
                "OP-8.*bounds widen exactly",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_uniform_like_validates_total_bounds_before_random(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Both bounds must be finite and `low <=\n> high`",
                "Bounds may be non-finite or reversed",
                "OP-8.*bounds must be finite",
            ),
            (
                "At the selected arithmetic width, `high - low` must also be finite",
                "An infinite high-minus-low value is accepted",
                "OP-8.*must also be finite",
            ),
            (
                "These checks, including the equal-bound case, complete before the "
                "operation\n> consumes a Random call ordinal",
                "These checks occur after consuming Random",
                "OP-8.*before the operation consumes",
            ),
            (
                "There is no f32 public-bound signature, default bound,\n"
                "> or f64 intermediate",
                "A compatibility f32-bound signature remains available",
                "OP-8.*no f32 public-bound signature",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_dropout_has_same_dtype_random_and_saved_mask_contracts(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "requires `input: &tensor[D,p]` and a scalar `rate: p`",
                "requires an f32 rate regardless of the input dtype",
                "OP-37.*scalar `rate: p`",
            ),
            (
                "rate must be finite and satisfy `0 <= rate < 1`",
                "rate may be non-finite or greater than one",
                "OP-37.*0 <= rate < 1",
            ),
            (
                "The accepted call consumes exactly one\n"
                "> ordinal, including for an empty tensor or `rate = 0`",
                "Empty and zero-rate calls consume no ordinal",
                "OP-37.*exactly one ordinal",
            ),
            (
                "For f16 and bf16, `sub` exact-widens its stored operands to f32",
                "f16 and bf16 convert both operands through f64",
                "OP-37.*exact-widen",
            ),
            (
                "pathwise adjoint reuses the exact saved\n> mask",
                "pathwise adjoint resamples a fresh mask",
                "OP-37.*saved mask",
            ),
            (
                "no f32 public-rate signature, f64 funnel,\n"
                "> unscaled-dropout alias, or special `rate >= 1` default exists",
                "A legacy unscaled-dropout alias remains available",
                "OP-37.*no f32 public-rate",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_host_numeric_builtin_manifest_has_no_specialized_compatibility_identities(self) -> None:
        block = self.repository_atom("05-OP-38")
        for identity in (
            "`tensor_scan` | `(T,((T,i64)->T!E),i64)->tensor[n,..state_shape(T),element(T)]!E`",
            "`process_run` | `(string,List[string])->(i64,string,string)!{IO}`",
            "`test_assert_eq` | `(Q,Q,string)->unit!{Test}`",
        ):
            with self.subTest(identity=identity):
                self.assertIn(identity, block)

        self.replace(
            Path("spec/05-risc-primitives.md"),
            "No dtype-named, rank-named, evaluator-only, legacy, or compatibility\n"
            "> identity is part of this atom",
            "Legacy dtype-named and evaluator-only aliases remain available",
        )
        self.assert_contract_fails("OP-38.*No dtype-named")

    def test_window_reduction_atom_is_total_and_target_independent(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Sum, max, and min admit every active numeric\n"
                "> tensor dtype; mean admits every active float tensor dtype",
                "Every window reduction admits only f32 tensors",
                "OP-39.*every active numeric",
            ),
            (
                "per-dtype arithmetic\n"
                "> width, no-user-accumulator rule, canonical balanced tree",
                "all arithmetic widens through f64 and uses a backend-selected tree",
                "OP-39.*per-dtype arithmetic",
            ),
            (
                "first-order adjoint, overlap accumulation, and higher-order rule",
                "first-order adjoint only",
                "OP-39.*higher-order rule",
            ),
            (
                "No target-specific rank,\n"
                "> reducer, dtype, first-order-only, host-fallback, alias, or "
                "compatibility\n> identity belongs to this atom",
                "A host fallback and first-order-only target identity are permitted",
                "OP-39.*No target-specific rank",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_recursive_cotangents_and_control_flow_are_language_rules(self) -> None:
        path = self.root / "spec/06-transformations.md"
        mutations = (
            (
                "If `A = List[T]` and `dT` is defined, then `dA = List[dT]`",
                "List values are never differentiable",
                "recursive List cotangent",
            ),
            (
                "`match` differentiates the arm executed by the forward program",
                "match is rejected under grad",
                "match executed-arm adjoint",
            ),
            (
                "Recursive calls differentiate the finite recurrence actually executed "
                "by the\nforward program and reverse that recorded call trajectory",
                "Recursive calls are rejected by every target",
                "recursive-call adjoint",
            ),
            (
                "A missing host-ABI\ncarrier is a backend capability gap, not a "
                "language restriction",
                "A missing host ABI carrier narrows the language signature",
                "recursive AD target independence",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_optimizer_cannot_reintroduce_unsound_numeric_identities(self) -> None:
        path = self.root / "spec/06-transformations.md"
        mutations = (
            (
                "none of these\npatterns is unconditional",
                "all familiar algebraic patterns are unconditional",
                "optimizer proof obligation",
            ),
            (
                "Potentially effectful or trapping nodes are observable roots; purity alone "
                "does\nnot make a possible trap dead",
                "unused trapping nodes and duplicate effects may be removed",
                "optimizer observable roots",
            ),
            (
                "Fusion preserves every primitive's declared arithmetic width and stored-value\n"
                "finalization boundary",
                "Fusion may retain all intermediates at f64",
                "fusion finalization boundary",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_formal_grad_returns_only_the_balanced_recursive_gradient_payload(self) -> None:
        path = self.root / "spec/06-transformations.md"
        mutations = (
            (
                "upstream = balanced_sum(\n"
                "            exact_zero(cotangent_type(type_of(n))),",
                "upstream = left_fold_add(contributions[n])",
                "formal balanced cotangent accumulation",
            ),
            (
                "return pack_wrt_gradients(grads)",
                "return (outputs[0], tuple(grads))",
                "formal gradient-only result",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_rng_ordinals_are_consumed_once_only_after_validation(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Each entered random primitive consumes exactly one call ordinal",
                "A random primitive may consume an implementation-defined number "
                "of call ordinals",
                "05-RNG-1.*one call ordinal",
            ),
            (
                "validation\n> that precedes Random consumption consumes none",
                "validation failures may consume a Random ordinal",
                "05-RNG-1.*validation",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_sequence_padding_admits_bool_and_moves_exact_bits(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "admits every active tensor element dtype\n"
                "> `T` in spec/04 §1.1, including `bool`",
                "admits only active numeric primitive dtypes and rejects bool",
                "OP-9.*including `bool`",
            ),
            (
                "Every source and padding element is moved at its declared dtype `T` with\n"
                "> no arithmetic, widening, narrowing, or other rounding",
                "Every source and padding element converts through a common f64 dtype",
                "OP-9.*declared dtype",
            ),
            (
                "has the same dtype,\n"
                "> element-movement, float-domain differentiation, and "
                "no-public-accumulator\n> rules as [05-OP-9]",
                "retains the legacy numeric-only dtype domain",
                "OP-10.*same dtype",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_sequence_padding_float_adjoints_preserve_nested_list_shape(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "source cotangent preserves the outer and inner runtime List shapes "
                "exactly",
                "source cotangent flattens all source rows",
                "OP-9.*runtime List shapes",
            ),
            (
                "sum of `g[r, c]` over padded result cells in increasing row-major "
                "`(r, c)`\n> order",
                "pad cotangents accumulate in backend-selected order",
                "OP-9.*increasing row-major",
            ),
            (
                "canonical adjacent-pair balanced tree\n"
                "> with an exact positive-zero base leaf",
                "left fold with an omitted initializer",
                "OP-9.*positive-zero base leaf",
            ),
            (
                "For integer or bool `T`, the operation is\n> forward-only",
                "integer and bool inputs receive float cotangents",
                "OP-9.*forward-only",
            ),
            (
                "a source\n"
                "> element `(r, c)` receives `g[r, c]` when `c < width` and exact "
                "positive zero\n> otherwise",
                "truncated source elements are dropped from the cotangent",
                "OP-10.*positive zero otherwise",
            ),
            (
                "truncated source cells remain present in the nested List\n> cotangent",
                "truncated source cells are omitted",
                "OP-10.*remain present",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_checked_c_metadata_contract_cannot_rebuild_or_convert(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        for old, new, message in (
            ("excluding spare storage capacity", "including spare storage capacity", "OP-33.*excluding spare"),
            ("takes rank and every target extent as exact tagged\n> i64 scalars", "takes unclassified integer metadata", "OP-33.*exact tagged i64"),
            ("changes no metadata, ownership, or\n> payload", "may mutate the descriptor", "OP-33.*changes no metadata"),
            ("preserves every stored element bit", "converts elements through f32", "OP-33.*preserves every stored element bit"),
        ):
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_gather_adjoint_has_a_real_positive_zero_base_leaf(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "canonical balanced accumulation tree begins with an exact\n"
                "> positive-zero base leaf, not an omitted initializer",
                "canonical balanced accumulation tree begins with its first contribution",
                "OP-33.*positive-zero base leaf",
            ),
            (
                "positive-zero base leaf, not an omitted initializer",
                "positive-zero initializer that may be omitted",
                "OP-33.*omitted initializer",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_prints_exact_recursive_rendering_and_reports_io_failure(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "writes exactly `R` of its argument followed by one byte\n"
                "> `\\n` to standard output",
                "writes an implementation-defined debug rendering",
                "OP-32.*writes exactly",
            ),
            (
                "It adds no\n"
                "> label, prefix, extra space, truncation beyond the nested tensor "
                "rule, or\n> additional newline",
                "It may add labels and truncate containers",
                "OP-32.*adds no label",
            ),
            (
                "A short or failed write traps `IO`",
                "A short or failed write returns success",
                "OP-32.*traps `IO`",
            ),
            (
                "Successful return means every required byte was written",
                "Successful return may precede the final write",
                "OP-32.*every required byte",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_tensor_element_egress_is_rank_total_without_compatibility_aliases(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`chelis_tensor_elements` boxes every\n"
                "> element as its exact scalar in row-major order for every rank",
                "chelis_tensor_elements admits rank one only",
                "OP-33.*every rank",
            ),
            (
                "Rank zero\n> therefore returns a one-element list",
                "Rank zero traps Domain",
                "OP-33.*one-element list",
            ),
            (
                "There is no rank-specialized or recursively nested list-egress alias",
                "The old rank-one alias remains available",
                "OP-33.*no rank-specialized",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_scatter_indices_are_typed_bounded_and_validated_before_writes(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Scatter indices have any active signed-integer dtype",
                "Scatter indices are limited to i32 or i64",
                "OP-33.*any active signed-integer",
            ),
            (
                "are zero-based, and must lie in the\n> selected base-axis extent",
                "may be negative and wrap in the selected base-axis extent",
                "OP-33.*must lie in the selected base-axis extent",
            ),
            (
                "Any negative or out-of-range index traps `Domain`\n> before any write",
                "An invalid index traps `Domain` after preceding writes",
                "OP-33.*before any write",
            ),
            (
                "There is no public string\n"
                "> scatter mode and no i32/i64-only dispatch exception",
                "An i32/i64-only compatibility dispatch remains available",
                "OP-33.*no i32/i64-only",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_gather_accepts_every_active_signed_index_width(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`gather` admits an index tensor of any active signed-integer dtype",
            "`gather` admits only i32 and i64 index tensors",
        )
        self.assert_contract_fails("OP-33.*any active signed-integer")

    def test_sum_requires_a_complete_numbered_spec_atom(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-30]** `sum",
            "> **[05-OP-130]** `sum",
        )
        self.assert_contract_fails("OP-30")

    def test_zero_exception_external_families_require_numbered_atoms(self) -> None:
        expected = {
            31: "scalar_carrier",
            32: "shape_index",
            33: "runtime_tensor",
            34: "numeric_adt",
            35: "stdlib_numeric_def",
        }
        for number, name in expected.items():
            with self.subTest(number=number):
                path = self.root / "spec/05-risc-primitives.md"
                original = path.read_text(encoding="utf-8")
                self.assertIn(f"> **[05-OP-{number}]** `{name}", original)
                path.write_text(
                    original.replace(
                        f"> **[05-OP-{number}]** `{name}",
                        f"> **[05-OP-{number + 100}]** `{name}",
                        1,
                    ),
                    encoding="utf-8",
                )
                try:
                    self.assert_contract_fails(f"OP-{number}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_ordered_comparisons_require_a_numbered_spec_atom(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-36]** `comparison",
            "> **[05-OP-136]** `comparison",
        )
        self.assert_contract_fails("OP-36")

    def test_ordered_comparison_nan_lowerings_cannot_use_plain_negation(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| `gte(a, b)` | `not(lt)` | `and(not(nan), not(lt))` |",
                "| `gte(a, b)` | `not(lt)` | `not(lt)` |",
                "ordered gte lowering",
            ),
            (
                "| `lte(a, b)` | `not(gt)` | `and(not(nan), not(gt))` |",
                "| `lte(a, b)` | `not(gt)` | `not(gt)` |",
                "ordered lte lowering",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(operation=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_ordered_comparison_domains_and_nan_results_are_frozen(self) -> None:
        block = self.repository_atom("05-OP-36")
        for clause in (
            "exactly the seven language identities `cmplt`, `lt`, `eq`, `neq`, "
            "`gt`, `gte`, and `lte`",
            "five ordered identities `cmplt`, `lt`, `gt`, `gte`, and `lte`",
            "`eq` and `neq` additionally admit bool scalars and same-shaped bool tensors, "
            "string scalars, unit",
            "Functions and resource handles are not "
            "equality-comparable",
            "any NaN makes `cmplt`, `lt`, `eq`, `gt`, `gte`, and `lte` false and "
            "makes `neq` true",
            "may use [05-OP-20] plus [05-OP-26..28] without sending bool through "
            "arithmetic IR",
            "No identity has an alias, grandfathered path, deprecated spelling, or "
            "compatibility wrapper",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, block)

    def test_equality_is_broader_than_ordered_comparison_without_legacy_aliases(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`eq` and `neq` additionally admit bool\n"
                "> scalars and same-shaped bool tensors, string scalars, unit, and two "
                "`List`,\n> tuple, `Dict`, `Option`, or ADT values of one static type",
                "eq and neq admit active numeric values only",
                "OP-36.*additionally admit bool",
            ),
            (
                "Ordered comparison\n> of bool, string, or a structured value is a type error",
                "Ordered comparison admits bool and string values",
                "OP-36.*Ordered comparison",
            ),
            (
                "Dictionaries compare key/value sets\n"
                "> independent of insertion order",
                "Dictionaries compare insertion order",
                "OP-36.*independent of insertion order",
            ),
            (
                "No identity has an alias, grandfathered path, deprecated spelling, "
                "or\n> compatibility wrapper",
                "Legacy comparison aliases remain supported",
                "OP-36.*No identity has an alias",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_ordered_comparisons_remain_distinct_until_ad_is_registered(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "These seven operations remain distinct typed comparison\n"
            "> identities through AD and other semantic transforms",
            "These operations lower to logical nodes before AD and inherit their "
            "structural gradient rejection",
        )
        self.assert_contract_fails("OP-36.*distinct typed")

    def test_ad_completeness_lists_all_comparisons_and_rejects_count(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Each exact [05-OP-36] identity:\n"
                "`cmplt`, `lt`, `eq`, `neq`, `gt`, `gte`, and `lte`",
                "Only cmplt and lt",
                "comparison AD identity completeness",
            ),
            (
                "`cast_trunc`,\n"
                "`cast_saturate`, `cast_wrap`, `and`, `or`, `not`,\n"
                "`count`, `argmax_reduce`, and `argmin_reduce` likewise reject `grad`",
                "cast and logical operations reject grad; count returns zero",
                "count structural AD rejection",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_cmplt_and_lt_lowerings_cannot_be_deleted_or_weakened(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| `cmplt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |\n",
                "",
                "ordered cmplt lowering",
            ),
            (
                "| `lt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |\n",
                "",
                "ordered lt lowering",
            ),
            (
                "| `cmplt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |",
                "| `cmplt(a, b)` | `cmplt(a, b)` | `cmplt(a, b)` |",
                "ordered cmplt lowering",
            ),
            (
                "| `lt(a, b)` | `cmplt(a, b)` | "
                "`and(not(nan), cmplt(a, b))` |",
                "| `lt(a, b)` | `cmplt(a, b)` | `cmplt(a, b)` |",
                "ordered lt lowering",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_cmplt_and_lt_cannot_leave_the_closed_identity_or_nan_sets(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "seven\n> language identities `cmplt`, `lt`, `eq`",
                "five language identities `eq`",
                "seven language identities",
            ),
            (
                "any NaN makes `cmplt`, `lt`, `eq`, `gt`, `gte`, and `lte`\n"
                "> false",
                "any NaN makes `eq`, `gt`, `gte`, and `lte` false",
                "any NaN makes `cmplt`",
            ),
            (
                "No identity has an alias, grandfathered path, deprecated spelling, "
                "or\n> compatibility wrapper",
                "`lt` is a grandfathered alias of `cmplt`",
                "No identity has an alias",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"OP-36.*{message}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_sum_cannot_restore_bool_compatibility(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "active signed integer or\n> active float. `bool`, `string`, reserved "
            "dtype spellings, scalar, and all other operands are\n> type errors",
            "active signed integer, active float, or\n> `bool`; every other operand "
            "is a type error",
        )
        self.assert_contract_fails("OP-30")

    def test_scalar_carrier_requires_canonical_bits(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "unused high bits are zero",
            "unused high bits are ignored",
        )
        self.assert_contract_fails("OP-31.*unused high bits")

    def test_tensor_read_view_lifetime_ends_before_a_write_begins(self) -> None:
        mutations = (
            (
                Path("spec/05-risc-primitives.md"),
                "until that descriptor is passed to\n> "
                "`chelis_tensor_begin_write`, whichever comes first",
                "for as long as any descriptor owner remains live",
                "OP-31.*chelis_tensor_begin_write",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "A successful begin invalidates\n> every read view previously "
                "returned for that descriptor",
                "A successful begin preserves every prior read view",
                "OP-44.*successful begin invalidates",
            ),
            (
                Path("spec/design/compiled_value_ownership.md"),
                "A successful begin invalidates every previously returned read view; "
                "dereferencing\n  such a stale view violates the caller precondition",
                "A successful begin preserves every previously returned read view",
                "write-begin read-view invalidation",
            ),
        )
        for path, old, new, message in mutations:
            with self.subTest(message=message):
                contract = self.root / path
                original = contract.read_text(encoding="utf-8")
                self.assertIn(old, original)
                contract.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    contract.write_text(original, encoding="utf-8")

    def test_scalar_carrier_pins_exact_public_layouts(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-31"]
        for declaration in (
            "typedef uint8_t chelis_dtype;",
            "typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;",
            "enum { CHELIS_VALUE_UNIT = 0, CHELIS_VALUE_SCALAR = 1, CHELIS_VALUE_STRING = 2, CHELIS_VALUE_TENSOR = 3, CHELIS_VALUE_LIST = 4, CHELIS_VALUE_TUPLE = 5, CHELIS_VALUE_DICT = 6, CHELIS_VALUE_ADT = 7, CHELIS_VALUE_OPTION = 8, CHELIS_VALUE_MAPPED_FILE = 9 };",
            "typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;",
            "typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;",
            "typedef struct { const void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_read_view;",
            "typedef struct { void *data; int64_t count; chelis_dtype dtype; uint8_t reserved[7]; } chelis_write_view;",
            "typedef struct { chelis_value key; chelis_value value; } chelis_dict_entry;",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, block)

    def test_scalar_carrier_layout_mutation_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "typedef struct { const void *data; int64_t count; chelis_dtype dtype; "
            "uint8_t reserved[7]; } chelis_read_view;",
            "typedef struct { const void *data; int32_t count; chelis_dtype dtype; "
            "uint8_t reserved[3]; } chelis_read_view;",
        )
        self.assert_contract_fails("OP-31.*chelis_read_view")

    def test_compiled_owner_atoms_are_frozen(self) -> None:
        mutations = (
            (
                Path("spec/04-type-system.md"),
                "exactly one logical owner",
                "zero or more logical owners",
                "04-LIN-3.*logical owner",
            ),
            (
                Path("spec/04-type-system.md"),
                "externally supplied entry arguments\n> are borrowed",
                "externally supplied entry arguments\n> transfer ownership",
                "04-LIN-7.*entry arguments",
            ),
            (
                Path("spec/04-type-system.md"),
                "compiler SHALL create an ordinary copy",
                "compiler MAY consume the entry borrow directly",
                "04-LIN-7.*ordinary copy",
            ),
            (
                Path("spec/04-type-system.md"),
                "result owner may be the owner transferred\n> through an owned parameter",
                "result ownership is inferred from the returned address",
                "04-LIN-4.*transferred",
            ),
            (
                Path("spec/04-type-system.md"),
                "storage reclaimable\n> before a following tail call or loop back-edge",
                "reclaimable only after the function returns",
                "04-LIN-8.*tail call",
            ),
            (
                Path("spec/04-type-system.md"),
                "move may transfer the value to one explicit successor owner",
                "move may leave both source and successor owners live",
                "04-LIN-8.*successor owner",
            ),
        )
        for relative, old, new, message in mutations:
            with self.subTest(message=message):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_ffi_entry_borrow_cannot_become_an_owned_input(self) -> None:
        self.replace(
            Path("spec/11-ffi.md"),
            "compiled entry borrows every input runtime value",
            "compiled entry owns every input runtime value",
        )
        self.assert_contract_fails("FFI entry borrow")

    def test_linearity_model_cannot_restore_scope_end_drop(self) -> None:
        mutations = (
            (
                "For an unconsumed local owner, the compiler inserts `Drop` at the "
                "earliest\npost-dominating point after its last use",
                "The compiler inserts end-of-scope `Drop` operations for "
                "unconsumed local owners",
                "linearity last-use Drop placement",
            ),
            (
                "Lexical scope\nexit is the fallback only when no earlier valid "
                "terminal point can be proved",
                "Lexical scope exit is always the terminal point",
                "linearity scope-exit fallback",
            ),
        )
        path = self.root / "spec/04-type-system.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_primitive_marker_erasure_preserves_verified_ownership(self) -> None:
        mutations = (
            (
                "Source borrow syntax and primitive-DAG borrow markers\n"
                "are erased before backend emission",
                "Every ownership fact is erased before IR lowering",
                "primitive source-marker erasure",
            ),
            (
                "The resolved disposition of every use is not erased; ownership\n"
                "lowering first records explicit borrow, move, clone, and terminal "
                "`Drop` obligations\nin the verified ownership representation "
                "consumed by every backend",
                "Backends infer ownership from emitted addresses",
                "primitive verified ownership preservation",
            ),
        )
        path = self.root / "spec/05-risc-primitives.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_keeps_the_complete_c_authority_range(self) -> None:
        self.replace(
            Path("spec/11-ffi.md"),
            "governed by [05-OP-31..33]",
            "governed by [05-OP-31] and [05-OP-33]",
        )
        self.assert_contract_fails("FFI complete C authority range")

    def test_compiled_ownership_cannot_omit_option(self) -> None:
        mutations = (
            (
                "    Option,\n    MappedFile,",
                "    MappedFile,",
                "Option heap kind",
            ),
            (
                "| `Option<T>` where `T` has a target recursive-value representation "
                "| `Option` | opaque `chelis_option *` handle and "
                "`CHELIS_VALUE_OPTION` |",
                "| `Option<T>` | none | legacy by-value carrier |",
                "Option carrier mapping",
            ),
            (
                "Every target-representable `Option<T>`, including `Option` of a "
                "scalar, mapped\nresource, or another `Option`",
                "Only an `Option` whose child is already a heap handle",
                "recursive Option heap classification",
            ),
            (
                "balanced tensor/string/List/tuple/dictionary/ADT/Option/mapped-file "
                "ownership",
                "balanced tensor/string/List/tuple/dictionary/ADT/mapped-file ownership",
                "Option ownership fixtures",
            ),
            (
                "omit `ConcreteHostType::Option` or `CHELIS_VALUE_OPTION`",
                "omit an unrelated host variant",
                "Option omission mutation",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_rejects_recursive_function_values_exactly(
        self,
    ) -> None:
        mutations = (
            (
                "Each identity has\nexactly one disposition: structurally nonheap, "
                "target-rejected with an owning\ncapability issue, direct heap "
                "carrier, tagged heap payload, or private heap\nallocation",
                "First-class functions may be omitted from the ownership registry",
                "closed target-rejection disposition",
            ),
            (
                "A function\nstored in `Option`, `List`, tuple, dictionary, or ADT "
                "is a `FirstClassValue`,\nnot a contextual callback",
                "A function in an aggregate is treated as a contextual callback",
                "recursive function placement",
            ),
            (
                "`UnsupportedKind::HostAbi`, `Stage::Codegen(\"c\")`, and\n"
                "`Unimplemented { issue: #879 }` after the sealed ownership "
                "boundary certifies\nthe exact selected payload and before backend "
                "emission",
                "an empty scalar after the sealed ownership boundary",
                "recursive function target rejection",
            ),
            (
                "It is a target capability result, not a language type error, "
                "scalar\nsubstitution, empty value, or permission to omit the type "
                "from the registry",
                "It is a permanent language rejection",
                "function rejection semantics",
            ),
            (
                "admit `Option[function]` or another recursive function container "
                "without the\n  exact [#879] target rejection",
                "admit every recursive function container",
                "function-container omission mutation",
            ),
            (
                "[#909]/[#879]:** own shared first-class function representation "
                "and the\n  general C-host closure ABI",
                "[#1286]:** owns the general C-host closure ABI",
                "function-value external owners",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_cannot_omit_recursive_mapped_file(self) -> None:
        mutations = (
            (
                "| `MappedFile` resource | `MappedFile` | opaque "
                "`chelis_mapped_file *` handle and `CHELIS_VALUE_MAPPED_FILE` |",
                "| `MappedFile` resource | `MappedFile` | opaque "
                "`chelis_mapped_file *`; never a `chelis_value` |",
                "mapped-file carrier mapping",
            ),
            (
                "`CHELIS_VALUE_MAPPED_FILE` is the exact tagged representation "
                "when that handle\nis stored in `Option`, `List`, tuple, "
                "dictionary, or ADT",
                "The resource is never stored in a recursive aggregate",
                "recursive mapped-file representation",
            ),
            (
                "including `Option[MappedFile]`, nested resource aggregates",
                "excluding resource aggregates",
                "Option ownership fixtures",
            ),
            (
                "omit `CHELIS_VALUE_MAPPED_FILE` or its `Option[MappedFile]` "
                "fixture",
                "omit an unrelated resource fixture",
                "mapped-file omission mutation",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_cannot_omit_string_or_tensor_tags(self) -> None:
        mutations = (
            (
                "| `string` | `String` | fixed `chelis_string` wrapper with an "
                "opaque target and `CHELIS_VALUE_STRING` |",
                "| `string` | `String` | fixed `chelis_string` wrapper with an "
                "opaque target |",
                "string carrier mapping",
            ),
            (
                "| tensor value or internal tensor view | `Tensor` | opaque "
                "`chelis_tensor *` handle and `CHELIS_VALUE_TENSOR` |",
                "| tensor value or internal tensor view | `Tensor` | opaque "
                "`chelis_tensor *` handle |",
                "tensor carrier mapping",
            ),
            (
                "tensor storage is the sole private heap allocation with no public "
                "tag. Every\ndirectly carried public heap kind also has the table's "
                "exact tagged\nrepresentation for recursive aggregates",
                "Public heap kinds may omit tagged recursive representations",
                "public heap tag totality",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_metal_cannot_gain_unverified_reuse(self) -> None:
        mutations = (
            (
                "typed `MetalNeverReuse` plan whose input cannot carry "
                "`ReusableOwnedStorage`",
                "Metal plan that accepts `ReusableOwnedStorage`",
                "Metal typed no-reuse plan",
            ),
            (
                "Metal emission with distinct storage for every produced node and "
                "no input",
                "Metal emission may alias produced nodes and input storage",
                "Metal no-alias fixture",
            ),
            (
                "let the Metal plan accept `ReusableOwnedStorage`",
                "let an unrelated plan accept an unrelated token",
                "Metal no-reuse mutation",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_launch_gate_stays_narrower_than_class_closure(
        self,
    ) -> None:
        mutations = (
            (
                "scripts/compiled_value_ownership_oracle.py --phase launch",
                "scripts/compiled_value_ownership_oracle.py --phase complete "
                "--require-hip",
                "launch ownership oracle command",
            ),
            (
                "COMPILED VALUE OWNERSHIP LAUNCH SUBSET: PASS",
                "COMPILED VALUE OWNERSHIP ORACLE: PASS",
                "launch ownership oracle success line",
            ),
            (
                "The `complete --require-hip` invocation is the eventual [#1286] "
                "class-closure\noracle. It is deliberately stronger than the "
                "launch invocation",
                "The launch invocation closes the entire class",
                "launch and class-closure distinction",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                count = original.count(old)
                self.assertGreater(count, 0)
                replacement_count = count if message == "launch ownership oracle success line" else 1
                path.write_text(
                    original.replace(old, new, replacement_count), encoding="utf-8"
                )
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_keeps_external_issue_owners(self) -> None:
        mutations = (
            (
                "The top-level tuple missing-`main` observation is [#545], not an "
                "ownership-oracle row",
                "The top-level tuple missing-`main` observation joins this oracle",
                "top-level tuple external owner",
            ),
            (
                "Runtime-valued `with seed` remains [#735] syntax/semantics work; "
                "recursive-host operation support remains [#729]/[#730] capability "
                "work",
                "All secondary recursion observations join this oracle",
                "recursive support external owners",
            ),
            (
                "[#1172] owns the span-key cause that can over-broaden hints; Surf "
                "reachability is exposure evidence",
                "Every reachability observation joins this oracle",
                "reachability external owner",
            ),
        )
        path = self.root / "spec/design/compiled_value_ownership.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_requires_runtime_seal_supersession(self) -> None:
        self.replace(
            Path("spec/design/compiled_value_ownership.md"),
            "Phase 1 must\n  explicitly supersede its numbered-spec citations, "
            "`runtime_representation.md`\n  target, guards, and public-layout "
            "promise in the same atomic change",
            "Both carrier contracts may coexist during migration",
        )
        self.assert_contract_fails("runtime representation supersession")

    def test_implicit_linearity_distinguishes_current_and_successor_drop(self) -> None:
        mutations = (
            (
                "The verified `OwnershipProgram` makes `RiscOp::Copy` and "
                "`RiscOp::Drop` real\nownership operations",
                "The ownership program may treat copy and drop as emission no-ops",
                "current Drop implementation status",
            ),
            (
                "C and HIP emit the exact\ndescriptor release selected by the "
                "verified directive; Metal consumes the same\ndirective as a typed "
                "no-device-owner disposition",
                "Backends may reconstruct a terminal release after verification",
                "successor Drop release",
            ),
        )
        path = self.root / "spec/design/implicit_linearity.md"
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_roadmap_cannot_replace_launch_subset_with_full_class_gate(self) -> None:
        path = self.root / "spec/design/remediation_roadmap.md"
        mutations = (
            (
                "[#1286]'s verified compiled ownership and opaque unified-heap ABI "
                "through [#1362]'s C-lane `--phase launch` oracle",
                "[#1286]'s ownership through the full HIP oracle",
                "roadmap launch ownership gate",
            ),
            (
                "full [#1286] class closure, including HIP and non-launch children, "
                "remains tracker work and does not gate v0.19",
                "full class closure gates v0.19",
                "roadmap full ownership boundary",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_compiled_ownership_phase_one_cannot_skip_container_authority(self) -> None:
        self.replace(
            Path("spec/design/compiled_value_ownership.md"),
            "The exact ABI is [05-OP-31..33], [05-OP-44], and all four registries",
            "The exact ABI is [05-OP-31..33] and the three carrier registries",
        )
        self.assert_contract_fails("complete C ABI authority chain")

    def test_compiled_ownership_phase_row_maps_are_exact(self) -> None:
        path = Path("spec/design/compiled_value_ownership.md")
        mutations = (
            (
                "This phase promotes exactly twenty-three oracle rows: the five [#543]",
                "This phase promotes exactly twenty-two oracle rows: four [#543] rows",
                "Phase 1 exact ownership row map",
            ),
            (
                "This phase promotes exactly six oracle rows: the [#1346] fold row",
                "This phase promotes five oracle rows and leaves depth one unresolved",
                "Phase 2 exact ownership row map",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                contract = self.root / path
                original = contract.read_text(encoding="utf-8")
                self.assertIn(old, original)
                contract.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    contract.write_text(original, encoding="utf-8")

    def test_backend_cannot_accept_unverified_ownership(self) -> None:
        self.replace(
            Path("spec/design/compiled_value_ownership.md"),
            "No arrow after verification may accept the pre-verification form",
            "A backend may accept the pre-verification form as a fallback",
        )
        self.assert_contract_fails("verified backend boundary")

    def test_scalar_parse_nan_spelling_and_images_are_frozen(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "The only accepted NaN spelling is exactly\n> `NaN`",
                "Every implementation-defined NaN spelling is accepted",
            ),
            ("f16 `0x7e00`", "f16 `0x7e01`"),
            ("bf16 `0x7fc0`", "bf16 `0x7fc1`"),
            ("f32 `0x7fc00000`", "f32 `0x7fc00001`"),
            ("f64 `0x7ff8000000000000`", "f64 `0x7ff8000000000001`"),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("OP-31")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_scalar_parse_signed_decimal_grammar_is_exact(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Signed-decimal integer text is exactly an optional\n"
            "> `+` or `-` followed by one or more ASCII digits",
            "Signed-decimal integer text accepts the host parser grammar",
        )
        self.assert_contract_fails("OP-31.*Signed-decimal")

    def test_scalar_parse_finite_float_decimal_grammar_is_exact(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "and then an optional exponent\n> `[eE][+-]?[0-9]+`",
                "and then a host-defined exponent",
                "optional exponent",
            ),
            (
                "non-ASCII digit, hex form, or\n> suffix is malformed",
                "non-ASCII digits, hex forms, and suffixes are accepted",
                "hex form",
            ),
            (
                "Its exact decimal value rounds once at the requested\n> float width",
                "Its value converts through host double",
                "exact decimal value",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"OP-31.*{message}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_nan_text_exception_deliberately_loses_payload_only(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "NaN\n> text round-trips at the class level and deliberately loses payload bits",
            "NaN text preserves every payload bit",
        )
        self.assert_contract_fails("05-OBS-1.*payload")

    def test_dictionary_keys_admit_bool_and_signed_widths_but_reject_floats(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Dictionary keys are exactly `string`, `bool`, or a scalar of any active\n"
            "> signed-integer dtype",
            "Dictionary keys are any scalar dtype or string, including floats",
        )
        self.assert_contract_fails("OP-32.*Dictionary keys")

    def test_shape_index_atom_pins_exact_c_signatures(self) -> None:
        block = self.repository_atom("05-OP-32")
        for signature in (
            "chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)",
            "chelis_list *chelis_list_from_values(const chelis_value *items, int64_t len)",
            "chelis_dict *chelis_dict_insert(const chelis_dict *dict, chelis_value key, chelis_value value)",
            "chelis_list *chelis_mmap_read(const chelis_mapped_file *mapped, int64_t offset, int64_t len)",
        ):
            with self.subTest(signature=signature):
                self.assertIn(signature, block)

    def test_shape_index_signature_width_is_frozen(self) -> None:
        self.replace(
            Path("spec/registry/c_container_boundary.md"),
            "chelis_string chelis_string_slice(chelis_string value, int64_t start, int64_t len)",
            "chelis_string chelis_string_slice(chelis_string value, int32_t start, int32_t len)",
        )
        self.assert_contract_fails("OP-32.*chelis_string_slice")

    def test_string_slice_start_at_or_beyond_length_is_empty(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "A slice whose nonnegative start is at or beyond the scalar\n"
            "> length is empty",
            "A slice starting at the scalar length traps `Domain`",
        )
        self.assert_contract_fails("OP-32.*nonnegative start")

    def test_tensor_scan_cannot_restore_an_f64_accumulator_funnel(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`tensor_scan` accumulator and emitted elements remain at `T`",
            "`tensor_scan` stores every accumulator and element as f64",
        )
        self.assert_contract_fails("tensor_scan exact carrier")

    def test_runtime_tensor_sort_nan_order_is_frozen(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "NaNs follow all\n> non-NaNs",
            "NaNs precede every non-NaN",
        )
        self.assert_contract_fails("OP-33.*NaNs")

    def test_runtime_tensor_rank_has_no_rank_eight_limit(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "There is no rank-eight limit",
            "Rank is limited to `0..=8`",
        )
        self.assert_contract_fails("OP-31.*rank-eight")

    def test_negative_axis_normalization_is_exactly_one_step(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "a negative value first\n> normalizes by adding the rank exactly once",
                "a negative value normalizes modulo the rank",
                "OP-7",
            ),
            (
                "first\n> applies §2.3's one-step negative normalization",
                "first normalizes modulo the rank",
                "OP-33",
            ),
        )
        for old, new, atom in mutations:
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(atom)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_scatter_modes_are_separate_typed_identities(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`chelis_tensor_scatter_replace`",
            "`chelis_tensor_scatter` with a string mode",
        )
        self.assert_contract_fails("OP-33.*scatter_replace")

    def test_runtime_scatter_dtype_domains_are_distinct(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Replace admits every active dtype, including bool",
                "Replace admits only signed-integer and float dtypes",
            ),
            (
                "admits exactly active\n> signed-integer and float dtypes",
                "admits every active dtype including bool",
            ),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("OP-33")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_diagonal_admits_bool(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`diagonal` admits every active dtype including bool",
            "`diagonal` rejects bool",
        )
        self.assert_contract_fails("OP-33.*diagonal")

    def test_runtime_where_language_authority_is_frozen(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "This atom's selection rule also governs exactly the language builtin",
            "This atom governs only the public C callable and not the language builtin",
        )
        self.assert_contract_fails("OP-33.*language builtin")

    def test_runtime_where_signature_cannot_narrow_precision(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`(&tensor[D,bool], &tensor[D,p], &tensor[D,p]) -> tensor[D,p]`",
            "`(&tensor[D,bool], &tensor[D,f32], &tensor[D,f32]) -> tensor[D,f32]`",
        )
        self.assert_contract_fails("OP-33.*tensor")

    def test_numeric_adt_construction_stays_an_ordinary_exact_constructor(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "accepts every representable declared field tuple",
            "silently normalizes selected module-specific field tuples",
        )
        self.assert_contract_fails("OP-34.*representable")

    def test_numeric_adt_adjoint_preserves_the_executed_recursive_shape(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "ordinary constructor and the executed matching arm preserve the "
                "recursive\n> cotangent shape",
                "ADT constructors and matches structurally reject grad",
                "OP-34.*recursive cotangent shape",
            ),
            (
                "`JsonFloat(x)` followed by an executed `JsonFloat(y)` match routes\n"
                "> the cotangent of `y` to `x`",
                "JsonFloat match drops the field cotangent",
                "OP-34.*JsonFloat",
            ),
            (
                "integer-only ADTs naturally have only `unit`\n> field cotangents",
                "integer-only ADTs receive float cotangent fields",
                "OP-34.*unit",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_process_stdlib_identities_require_io_effects(self) -> None:
        self.replace(
            Path("spec/registry/stdlib_numeric_manifest.md"),
            "`process::run` | `(string,List[string])->(i64,string,string)!{IO}`",
            "`process::run` | `(string,List[string])->(i64,string,string)`",
        )
        self.assert_contract_fails("OP-35.*process::run")

    def test_host_operations_remain_language_legal_in_every_execution_mode(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "traps, and value\n> result in every language execution mode",
                "only in the evaluator execution mode",
                "05-HOST-1.*effects",
            ),
            (
                "it does not make the host operation illegal",
                "it makes the host operation illegal",
                "05-HOST-1.*does not make",
            ),
            (
                "are\n> legal host-runtime operations in every language execution mode",
                "are legal only under chelis eval",
                "05-HOST-2.*legal host-runtime operations",
            ),
            (
                "effect-boundary fact SHALL NOT be represented as a language-wide "
                "rejection,\n> inert stub, default value, or evaluator-only signature",
                "effect-boundary fact is a language-wide evaluator-only restriction",
                "05-HOST-2.*language-wide",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_test_assertions_are_generic_and_not_whole_module_rejections(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "assertion identities are generic; dtype-named or rank-named aliases "
                "do not\n> exist",
                "assertion identities retain dtype-named compatibility aliases",
                "05-HOST-3.*generic",
            ),
            (
                "SHALL NOT emit an inert assertion, default value,\n"
                "> compatibility helper, or whole-module rejection based on an "
                "unreachable\n> assertion",
                "may reject a whole module containing an unreachable assertion",
                "05-HOST-3.*whole-module rejection",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_linspace_has_no_nonpositive_count_compatibility_case(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "requires finite endpoints and\n> i64 `count >= 1`",
            "accepts nonpositive counts as a one-element result",
        )
        self.assert_contract_fails("OP-35.*count >= 1")

    def test_stdlib_manifest_has_exactly_eighty_four_unique_rows(self) -> None:
        registry = (
            REPO_ROOT / "spec/registry/stdlib_numeric_manifest.md"
        ).read_text(encoding="utf-8")
        rows = re.findall(r"^\| `([^`]+)` \|", registry, re.MULTILINE)
        self.assertEqual(len(rows), 84)
        self.assertEqual(len(set(rows)), 84)
        identities = set(rows)
        for identity in (
            "decimal::decimal_add",
            "io/json::json_bigint",
            "io/json::load_json",
            "time::date_lt",
            "tokenizer::load_tokenizer",
        ):
            with self.subTest(identity=identity):
                self.assertIn(identity, identities)

    def test_stdlib_manifest_duplicate_identity_fails(self) -> None:
        self.replace(
            Path("spec/registry/stdlib_numeric_manifest.md"),
            "| `index::take_list` |",
            "| `index::list_index` |",
        )
        self.assert_contract_fails("stdlib numeric manifest")

    def test_stdlib_manifest_uses_rank_polymorphic_sort_and_generic_scalar_equality(self) -> None:
        block = self.repository_atom("05-OP-35")
        self.assertIn(
            "`sort::sort` | `(&tensor[..r,p_numeric],i32)->"
            "(tensor[..r,p_numeric],tensor[..r,i64])`",
            block,
        )
        self.assertIn(
            "`test::assert_eq` | `(Q,Q,string)->unit!{Test}`",
            block,
        )
        for legacy_identity in ("sort::sort_1d", "sort::sort_2d", "test::assert_eq_int"):
            with self.subTest(legacy_identity=legacy_identity):
                self.assertNotIn(legacy_identity, block)

    def test_stdlib_json_parsing_inherits_op2_integer_overflow(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Numeric\n> tokens follow [05-OP-2]",
            "Integer-form tokens outside i64 fall back to `JsonFloat`",
        )
        self.assert_contract_fails("OP-35.*Numeric tokens")

    def test_normal_cdf_defines_infinities_nan_and_fixed_policy_tolerance(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`normal_cdf(+inf)` is exact `1p`, `normal_cdf(-inf)` is exact `0p`",
                "normal_cdf rejects both infinities",
                "OP-35.*normal_cdf\\(\\+inf\\)",
            ),
            (
                "a NaN\n> input returns [04-NUM-2]'s canonical NaN at `p_float`",
                "a NaN input traps Domain",
                "OP-35.*canonical NaN",
            ),
            (
                "The infinities have zero cotangent and NaN propagates the\n"
                "> canonical NaN cotangent; no non-finite input traps `Domain`",
                "All non-finite inputs trap Domain",
                "OP-35.*zero cotangent",
            ),
            (
                "`standard_contract_tolerance` is deliberately the fixed f32\n"
                "> tolerance policy of the named standard-contract property corpus",
                "standard_contract_tolerance narrows normal_cdf to f32",
                "OP-35.*fixed f32 tolerance",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_arange_is_same_signed_dtype_with_checked_exact_bounds(self) -> None:
        row_path = self.root / "spec/registry/stdlib_numeric_manifest.md"
        prose_path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| `tensor/construct::arange` | "
                "`(p_int,p_int)->tensor[n,p_int]` |",
                "| `tensor/construct::arange` | "
                "`(i32,i32)->tensor[n,i32]` |",
                "OP-35.*exact manifest",
            ),
            (
                "`arange(start,stop)` admits one active signed-integer dtype "
                "`p_int` for both\n> endpoints",
                "`arange(start,stop)` admits mixed integer endpoint dtypes",
                "OP-35.*both endpoints",
            ),
            (
                "returns the increasing half-open same-dtype sequence",
                "returns an i64 sequence for every endpoint dtype",
                "OP-35.*same-dtype sequence",
            ),
            (
                "Its\n> length and every step are checked in exact mathematical "
                "integers",
                "Its length and steps use unchecked host integer arithmetic",
                "OP-35.*exact mathematical integers",
            ),
            (
                "an\n> unrepresentable length or element traps `Overflow`",
                "an unrepresentable length or element wraps",
                "OP-35.*unrepresentable",
            ),
        )
        for old, new, message in mutations:
            path = row_path if old.startswith("| `") else prose_path
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_reduction_metadata_declarations_cannot_disappear(self) -> None:
        path = self.root / "spec/registry/c_tensor_runtime.md"
        original = path.read_text(encoding="utf-8")
        rows = [row for row in oracle.EXPECTED_OP_MANIFESTS["05-OP-33"] if "checked reduction" in row]
        self.assertEqual(len(rows), 8)
        for row in rows:
            with self.subTest(row=row):
                self.assertIn(row, original)
                path.write_text(original.replace(row + "\n", "", 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("05-OP-33.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_exact_op_manifest_row_deletion_fails(self) -> None:
        for atom, rows in oracle.EXPECTED_OP_MANIFESTS.items():
            relative = oracle.OP_MANIFEST_REGISTRY_FILES.get(
                atom, "spec/05-risc-primitives.md"
            )
            path = self.root / relative
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(rows[0], original)
                path.write_text(original.replace(rows[0] + "\n", "", 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"{atom}.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_exact_op_manifest_signature_or_shape_mutation_fails(self) -> None:
        mutations = {
            "05-OP-31": (
                "int64_t chelis_dtype_size(chelis_dtype dtype)",
                "int32_t chelis_dtype_size(chelis_dtype dtype)",
            ),
            "05-OP-32": (
                "int64_t chelis_string_len(chelis_string value)",
                "int32_t chelis_string_len(chelis_string value)",
            ),
            "05-OP-33": (
                "const int64_t *shape, chelis_dtype dtype)",
                "const int32_t *shape, chelis_dtype dtype)",
            ),
            "05-OP-34": (
                "JsonArray(List[Json]) | JsonObject(Dict[string,Json])",
                "JsonArray(List[Json])",
            ),
            "05-OP-44": (
                "void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar "
                "rank, const chelis_scalar *shape)",
                "void chelis_tensor_repurpose(chelis_tensor *tensor, chelis_scalar "
                "rank, chelis_scalar *shape)",
            ),
            "05-OP-35": ("(p_float)->p_float", "(f32)->f32"),
            "05-OP-38": (
                "(T,((T,i64)->T!E),i64)->tensor[n,..state_shape(T),element(T)]!E",
                "(f32,((f32,i64)->f32),i64)->tensor[n,f32]",
            ),
        }
        for atom, (old, new) in mutations.items():
            relative = oracle.OP_MANIFEST_REGISTRY_FILES.get(
                atom, "spec/05-risc-primitives.md"
            )
            path = self.root / relative
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"{atom}.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_exact_op_manifest_duplicate_label_or_identity_fails(self) -> None:
        for atom, rows in oracle.EXPECTED_OP_MANIFESTS.items():
            relative = oracle.OP_MANIFEST_REGISTRY_FILES.get(
                atom, "spec/05-risc-primitives.md"
            )
            path = self.root / relative
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(rows[0], original)
                mutated = (
                    original.replace(rows[1], rows[0], 1)
                    if len(rows) > 1
                    else original.replace(rows[0], rows[0] + "\n" + rows[0], 1)
                )
                path.write_text(mutated, encoding="utf-8")
                try:
                    self.assert_contract_fails(f"{atom}.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_assert_close_tensor_is_float_only_and_own_width(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-35"] + "\n" + (
            REPO_ROOT / "spec/registry/stdlib_numeric_manifest.md"
        ).read_text(encoding="utf-8")
        self.assertIn("&tensor[..r,p_float]", block)
        self.assertIn(
            "&tensor[..r,p_float],p_float,string)->unit!{Test}", block
        )
        self.assertIn("comparison executes at `p`'s [04-NUM-8] arithmetic width", block)
        self.assertIn("never converts either tensor\n> through f64", block)

    def test_stdlib_precision_variable_domains_are_exact(self) -> None:
        block = self.repository_atom("05-OP-35")
        for clause in (
            "`p` ranges over all active tensor element dtypes",
            "`p_numeric` over all active numeric dtypes",
            "`p_int` over all active signed integers",
            "`p_float` over all four active floats",
            "`Q` over one static type in [05-OP-36]'s scalar or recursive "
            "equality domain",
            "Every repeated variable denotes one common static type",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, block)

        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`p_numeric` over all active numeric dtypes",
                "`p_numeric` over active float dtypes only",
                "OP-35.*p_numeric",
            ),
            (
                "`p_int` over all active signed\n> integers",
                "`p_int` over i64 only",
                "OP-35.*p_int",
            ),
            (
                "`p_float` over all four active floats",
                "`p_float` over f32 and f64 only",
                "OP-35.*p_float",
            ),
            (
                "`Q` over one static type\n"
                "> in [05-OP-36]'s scalar or recursive equality domain",
                "`Q` over float scalars only",
                "OP-35.*`Q`",
            ),
            (
                "Every repeated variable denotes one\n> common static type",
                "Repeated variables may instantiate independently",
                "OP-35.*common static type",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_scalar_and_assertion_manifest_domains_cannot_narrow(self) -> None:
        path = self.root / "spec/registry/stdlib_numeric_manifest.md"
        mutations = (
            (
                "| `scalar::abs` | `(p_numeric)->p_numeric` |",
                "| `scalar::abs` | `(f32)->f32` |",
            ),
            (
                "| `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |",
                "| `scalar::max` | `(f32,f32)->f32` |",
            ),
            (
                "| `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |",
                "| `scalar::min` | `(f32,f32)->f32` |",
            ),
            (
                "| `test::assert_close` | "
                "`(p_float,p_float,p_float,string)->unit!{Test}` |",
                "| `test::assert_close` | "
                "`(f32,f32,f32,string)->unit!{Test}` |",
            ),
            (
                "| `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |",
                "| `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],f32,string)->unit!{Test}` |",
            ),
            (
                "| `test::assert_eq` | "
                "`(Q,Q,string)->unit!{Test}` |",
                "| `test::assert_eq` | `(f32,f32,string)->unit!{Test}` |",
            ),
        )
        for old, new in mutations:
            with self.subTest(row=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("OP-35.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_assertions_keep_common_dtypes_and_own_width_comparison(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "Test tolerances have the same active float dtype as the values",
                "Test tolerances are always f32 regardless of the value dtype",
                "OP-35.*same active float dtype",
            ),
            (
                "admits exactly one common active float dtype `p` and equal shapes",
                "admits independently typed float tensors with equal shapes",
                "OP-35.*one common active float dtype",
            ),
            (
                "stored\n> same-dtype tolerance converted exactly to that arithmetic width",
                "tolerance converted through f64 before comparison",
                "OP-35.*same-dtype tolerance",
            ),
            (
                "Scalar `assert_close` applies the same own-width rule\n"
                "> to any active float dtype",
                "Scalar `assert_close` converts every input through f64",
                "OP-35.*own-width rule",
            ),
            (
                "`assert_eq` uses [05-OP-36] equality for one\n"
                "> common scalar or recursively comparable `Q`",
                "Scalar assert_eq permits mixed or float-only values",
                "OP-35.*recursively comparable",
            ),
            (
                "Direct tensor arguments use\n"
                "> `assert_eq_tensor`, which requires equal shapes and one common "
                "active element\n> dtype",
                "Direct tensor arguments use scalar assertion compatibility aliases",
                "OP-35.*Direct tensor",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stack_and_reshape_operations_keep_their_exact_adjoints(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "The\n> float `squeeze` and `unsqueeze` adjoints are the reverse "
                "reshape graph",
                "Squeeze and unsqueeze structurally reject grad",
                "OP-35.*reverse reshape graph",
            ),
            (
                "The float `stack` adjoint slices the output cotangent along the "
                "inserted axis\n> in increasing input-list order",
                "Stack structurally rejects grad because no List cotangent carrier "
                "is implemented",
                "OP-35.*stack.*adjoint",
            ),
            (
                "returns a `List` of tensor cotangents with\n"
                "> exactly the input list's length, shapes, and dtype",
                "returns one tensor cotangent instead of a List",
                "OP-35.*returns a `List`",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_accumulator_precision_cannot_cross_numeric_kinds(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "An accumulator has the same numeric kind as its operands",
            "An accumulator may cross from integer operands to float",
        )
        self.assert_contract_fails("same-kind accumulator rule")

    def test_spec04_shape_axis_adds_rank_once_before_rejection(self) -> None:
        path = self.root / "spec/04-type-system.md"
        mutations = (
            (
                "Every literal or computed\naxis first adds the input rank exactly once when negative",
                "A negative axis normalizes modulo the input rank",
                "shape negative-axis normalization",
            ),
            (
                "A statically known normalized value outside `0..rank` is a type\nerror",
                "before normalization is a type error",
                "shape post-normalization rejection",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_runtime_extent_amendments_are_defended(self) -> None:
        mutations = (
            (
                Path("spec/04-type-system.md"),
                "access whose shape depends on the guarded extent",
                "access of the function",
                "runtime extent guard placement",
            ),
            (
                Path("spec/04-type-system.md"),
                "places guards by this rule",
                "may place guards anywhere",
                "runtime extent guard placement in every execution mode",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "`InputAxis(t, a)`",
                "`AxisRead(t, a)`",
                "folded tensor-axis extent carrier",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "`reshape` admits `Lit`, `Node`, `InputAxis`, and `Sym`",
                "`reshape` admits `Lit` and `Node`",
                "runtime extent owner admission",
            ),
            (
                Path("spec/04-type-system.md"),
                "and the dtype of the quantity that guard\n> finalizes",
                "and declared result dtype\n> ",
                "precondition guard finalized-quantity dtype",
            ),
            (
                Path("spec/04-type-system.md"),
                "`numeric trap: domain in <op> at i64`",
                "`numeric trap: domain in <op> at <prim>`",
                "runtime extent guard trap line",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "well formed only when the start\n  paired with it is `Lit(0)`",
                "well formed with any start",
                "ToEnd shrink end requires a zero start",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "`expand` sets the extent at `axis` and is well formed only "
                "when the operand's\nextent at `axis` is 1",
                "`expand` sets the extent at `axis` for any operand extent",
                "expand requires a unit source extent",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "A literal operand extent at\n`axis` other than 1 is a type error. A symbolic or runtime operand extent at\n`axis` other than 1 fails that claim's runtime extent guard and traps\n`Domain`, placed and rendered per `spec/04-type-system.md` §4.7 and\n[04-NUM-9].",
                "Any operand extent at\n`axis` is accepted.",
                "expand non-unit source extent is rejected or traps",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "| `expand` | `insert(sum(g, axis), axis, 1i64)`",
                "| `expand` | `sum(g, axis)` |",
                "expand adjoint restores the unit axis",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "| `insert` | `sum(g, axis)`",
                "| `insert` | `insert(g, axis, 1i64)` |",
                "insert adjoint collapses the inserted axis",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "A reduction axis, `expand`'s broadcast axis, and `insert`'s"
                "\n> new-axis position SHALL be",
                "A reduction axis SHALL be",
                "axis atom names both movement primitives",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "names the\n> dimension it creates, which is by construction not a dimension of the\n> operand; that name SHALL be statically resolvable in the same sense",
                "names any\n> dimension",
                "insert names a dimension absent from the operand",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "> `insert(g / divisor, axis, original_extent)` at the "
                "operand dtype.",
                "> `expand(g / divisor, original_shape, axis)` at the "
                "operand dtype.",
                "mean adjoint reinserts the reduced axis",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "[05-AXIS-1] governs the reduction, `expand`, and `insert`\n> family",
                "[05-AXIS-1] governs the static reduction/expand family",
                "C axis family names both movement primitives",
            ),
            (
                Path("spec/05-risc-primitives.md"),
                "| `insert` | `(&tensor[D,p], axis: i32, size: i64) -> "
                "tensor[D_plus,p]` | Insert a new dimension of width `size` "
                "at position `axis`, producing rank `rank(x) + 1`.",
                "| `insert` | unspecified |",
                "insert movement row",
            ),
            (
                Path("spec/04-type-system.md"),
                "Each operation has exactly one result shape. `expand` sets "
                "the extent at\n`axis` and leaves the rank unchanged; `insert` adds an axis of extent `size`\nat `axis` and produces rank `rank(x) + 1`. No result is deferred, no consumer\nselects between shapes, and no context supplies a default.",
                "A consumer selects between two candidate shapes, and an unconsumed result\ntakes a default at the freeze point.",
                "expand and insert each have one result shape",
            ),
            (
                Path("spec/04-type-system.md"),
                "`insert` admits `axis` in `0..=rank(x)`, so\n`axis == rank(x)` appends a trailing axis. An axis outside its operation's\nrange is a type error.",
                "`insert` admits any `axis`.",
                "insert axis range",
            ),
            (
                Path("spec/04-type-system.md"),
                "**Named-axis insert (`R+1`).** The inverse arithmetic "
                "direction: `insert`\nadds a *named* axis",
                "**Named-axis expand (`R+1`).** The inverse arithmetic "
                "direction: `expand`\nadds a *named* axis",
                "named-axis form belongs to insert",
            ),
            (
                Path("spec/06-transformations.md"),
                "over these) is not batched",
                "over these) is batched",
                "vmap runtime extent non-batching rule",
            ),
            (
                Path("spec/06-transformations.md"),
                "### 8.6 `batch_varying_extent` (vmap)",
                "### 8.6 `batch_shared_extent` (vmap)",
                "vmap batch-varying extent rejection",
            ),
        )
        for relative, old, new, message in mutations:
            with self.subTest(message=message):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_spec04_comparison_row_and_surfaces_are_closed(self) -> None:
        path = self.root / "spec/04-type-system.md"
        mutations = (
            (
                "Ordered comparison (`cmplt`, `lt`, `gt`, `gte`, `lte`)",
                "Ordered comparison (`cmplt`, `gt`, `gte`, `lte`)",
                "closed ordered-comparison precision row",
            ),
            (
                "Equality (`eq`, `neq`) | any active numeric dtype or bool",
                "Equality (`eq`, `neq`) | any active numeric dtype",
                "closed equality precision row",
            ),
            (
                "Every tensor comparison preserves the operand surface: two "
                "same-dtype tensors\n"
                "with identical dimensions return a bool tensor with those dimensions",
                "two tensors with arbitrary dimensions return a bool tensor",
                "comparison surface preservation",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_arithmetic_nan_finalization_bits_and_payload_loss_are_frozen(self) -> None:
        path = self.root / "spec/04-type-system.md"
        mutations = (
            ("f16 `0x7e00`", "f16 `0x7e01`"),
            ("bf16 `0x7fc0`", "bf16 `0x7fc1`"),
            ("f32\n> `0x7fc00000`", "f32 `0x7fc00001`"),
            ("f64 `0x7ff8000000000000`", "f64 `0x7ff8000000000001`"),
            (
                "Arithmetic preserves the NaN\n> class, not an input payload or sign",
                "Arithmetic preserves every NaN payload and sign",
            ),
            (
                "preserves NaN payload bits only when its governing operation atom\n"
                "> explicitly says it is bit-preserving",
                "always canonicalizes NaN payload bits",
            ),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("04-NUM-2")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_sum_result_rule_is_total_for_every_allowed_accumulator(self) -> None:
        text = (REPO_ROOT / "spec/04-type-system.md").read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", text)
        self.assertIn(
            "`sum_result(p, a) = p` exactly when `p` is `bf16` or `f16`; "
            "otherwise `sum_result(p, a) = a`",
            normalized,
        )
        self.assertIn("`bf16` | `f32`, `f64` | `bf16`", text)
        self.assertIn("`i32` | `i32`, `i64` | accumulator dtype `a`", text)

    def test_backend_neutral_contract_keeps_all_ten_active_primitives(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "one of the ten active primitives is well-typed",
            "one of the nine active primitives is well-typed",
        )
        self.assert_contract_fails("backend-neutral active primitive set")

    def test_sum_result_rule_cannot_drop_explicit_wider_accumulators(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "`sum_result(p, a) = p` exactly when `p` is `bf16` or `f16`;\n"
            "otherwise `sum_result(p, a) = a`",
            "`sum_result(p, a)` is implementation-defined for explicit accumulators",
        )
        self.assert_contract_fails("total sum result precision rule")

    def test_to_string_domain_rejects_non_list_composites_and_opaque_values(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Unit, tuples,\n> `Dict`, `Option`, ADTs, functions, resource handles, "
            "and\n> deferred values are type errors",
            "Unit and tuples are rendered recursively",
        )
        self.assert_contract_fails("OP-25.*Unit")

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
            "exact case enumerator is the closed set scalar, tensor, and List",
            "case enumerator also accepts unit",
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
            "> adjoint reverses that composition. Integer operands are forward-only and\n"
            "> `grad` rejects them.",
            "> adjoint reverses that composition. Integer operands are forward-only and\n"
            "> `grad` rejects them.\n"
            "> A backend MAY instead route the full non-NaN cotangent to only the "
            "last\n"
            "> element equal to the selected maximum.",
        )
        self.assert_changed_contract_requires_acknowledgement('atom', '05-OP-12')

    def test_plain_prose_after_extrema_atom_cannot_contradict_it(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> **[05-OP-13]**",
            "\nAn implementation MAY instead route the full non-NaN `max_reduce` "
            "cotangent to only the last element equal to the selected maximum.\n\n"
            "> **[05-OP-13]**",
        )
        self.assert_changed_contract_requires_acknowledgement('region', 'numeric primitive contracts')

    def test_plain_prose_before_multi_axis_contract_cannot_contradict_it(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "Multiple axes may be reduced in one call",
            "For multiple named-axis reductions, source spelling order controls "
            "evaluation order and therefore owns exact values, traps, NaN selection, "
            "and adjoints.\n\n"
            "Multiple axes may be reduced in one call",
        )
        self.assert_changed_contract_requires_acknowledgement('region', 'name-preserving rank polymorphism')

    def test_legacy_bool_arithmetic_alias_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Logical operations do not alias arithmetic primitives.",
            "Logical operations do not alias arithmetic primitives. `and` is `mul`, "
            "`or` is `max_elem`, and `not` is `neg` on bool values.",
        )
        self.assert_changed_contract_requires_acknowledgement('region', 'logical builtin contract')

    def test_product_tree_body_is_required(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "pairs adjacent values from left to right",
            "an implementation-defined combination",
        )
        self.assert_contract_fails("OP-14.*pairs adjacent")

    def test_product_tree_rounds_each_node_to_operand_storage(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-14"]
        self.assertIn(
            "finalized once to the operand storage dtype before it enters the next level",
            block,
        )

    def test_product_tree_cannot_round_only_at_the_root(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "is finalized once to the operand storage dtype before it enters the next level",
            "remains at the arithmetic width until one final root cast",
        )
        self.assert_contract_fails("OP-14.*operand storage dtype")

    def test_window_sum_uses_a_canonical_balanced_tree(self) -> None:
        text = (REPO_ROOT / "spec/05-risc-primitives.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "Window leaves are enumerated in row-major order and combined by the canonical adjacent-pair balanced tree",
            re.sub(r"\s+", " ", text),
        )
        self.assertIn(
            "overlapping windows are enumerated by increasing row-major output position",
            re.sub(r"\s+", " ", text),
        )

    def test_stdlib_numeric_closure_recurses_through_numeric_adts(self) -> None:
        self.assertIn(
            "A public signature is numeric when any reachable field of an admitted ADT",
            self.repository_atom("05-OP-34"),
        )

    def test_stdlib_numeric_closure_cannot_scan_direct_primitives_only(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "The reachability computation expands\n> nominal ADT definitions recursively "
            "to a fixed point",
            "The reachability computation examines only primitives written directly in "
            "the signature",
        )
        self.assert_contract_fails("OP-34")

    def test_stdlib_numeric_closure_includes_tensor_precision_variables(self) -> None:
        self.assertIn(
            "A precision type variable in a tensor element or linked scalar position is numeric",
            self.repository_atom("05-OP-34"),
        )
        self.assertIn(
            "`init/xavier::sample` is not a language operation and must not be exported",
            self.repository_atom("05-OP-35"),
        )

    def test_assert_close_tensor_cannot_restore_f64_comparison(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "It never converts either tensor\n> through f64",
            "It converts every tensor element through f64",
        )
        self.assert_contract_fails("OP-35.*never converts")

    def test_stdlib_decimal_contract_is_total(self) -> None:
        block = self.repository_atom("05-OP-35")
        for clause in (
            "The accepted decimal grammar is",
            "canonical zero is `Decimal { coefficient: 0i64, scale: 0i64 }`",
            "Addition, subtraction, multiplication, equality, and ordering use exact mathematical rationals",
            "`decimal_to_string` emits the unique canonical non-exponent form",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, block)

    def test_decimal_parse_normalizes_before_representation_checks(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "interpreted in exact arithmetic and normalized before either "
                "representation\n> check",
                "checked for i64 representation before normalization",
            ),
            (
                "removable trailing zeros do not cause `Overflow`",
                "removable trailing zeros may cause `Overflow`",
            ),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("OP-35")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_json_contract_is_total(self) -> None:
        block = self.repository_atom("05-OP-35")
        for clause in (
            "`try_parse_json` returns `None` exactly for invalid JSON text",
            "A leading U+FEFF byte-order mark is not RFC 8259 whitespace and is rejected",
            "Every finitely nested valid document is in the language",
            "there is no fixed semantic nesting depth such as 512",
            "Object serialization orders members by increasing Unicode scalar-value key sequence",
            "validate the complete document before opening or truncating the destination",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, block)

    def test_stdlib_json_rejects_bom_without_a_semantic_depth_cap(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "A leading U+FEFF\n"
                "> byte-order mark is not RFC 8259 whitespace and is rejected",
                "A leading byte-order mark is accepted as whitespace",
                "OP-35.*byte-order mark",
            ),
            (
                "Every finitely\n"
                "> nested valid document is in the language; there is no fixed semantic "
                "nesting\n> depth such as 512",
                "Documents deeper than 512 levels are language errors",
                "OP-35.*finitely nested",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_time_contract_is_total(self) -> None:
        block = self.repository_atom("05-OP-35")
        for clause in (
            "Years `0000` through `9999` use exactly four digits",
            "`day_of_week` fixes `1970-01-01` as Thursday",
            "date comparisons are lexicographic on `(year, month, day)`",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, block)

    def test_stdlib_assertion_and_process_contracts_are_total(self) -> None:
        normalized = self.repository_atom("05-OP-35")
        self.assertIn(
            "`assert_shape` requires its expected list to contain only nonnegative "
            "i64 extents and compares its length and every entry to the tensor's "
            "complete shape in axis order",
            normalized,
        )
        self.assertIn(
            "Every host-evaluator context carries an optional absolute, executable "
            "`chelis_executable` capability",
            normalized,
        )
        self.assertIn("`run_chelis` invokes exactly that path", normalized)

    def test_stdlib_random_graph_is_exact(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-35"]
        self.assertIn("`two_pi = round_p(2*pi)`", block)
        self.assertIn("`cos_term = cos(mul(two_pi, u2))`", block)
        self.assertIn("`mul(sub(mul(2p, u), 1p), bound)`", block)

    def test_stdlib_random_graphs_keep_pathwise_adjoints(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "every random\n"
                "> stdlib callable has the pathwise adjoint of its exact graph above",
                "random stdlib callables structurally reject grad",
                "OP-35.*pathwise adjoint",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_rng_stream_cannot_vary_by_compiler_version(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "compiler version and target do not vary this result",
            "compiler versions and targets may choose different streams",
        )
        self.assert_contract_fails("05-RNG-1.*compiler version")

    def test_random_rounded_high_endpoint_rule_is_frozen(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "rounded result equals the stored upper endpoint",
            "rounded result may equal the stored upper endpoint",
        )
        self.assert_contract_fails("OP-35.*upper endpoint")

    def test_xavier_computed_denominator_must_be_finite_positive(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "computed\n> denominator must be finite and strictly positive",
            "computed denominator may be non-finite or nonpositive",
        )
        self.assert_contract_fails("OP-35.*denominator")

    def test_stdlib_tensor_signatures_are_precision_generalized(self) -> None:
        block = self.repository_atom("05-OP-35")
        for signature in (
            "`contracts::normal_cdf` | `(p_float)->p_float`",
            "`init/random::normal_like` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}`",
            "`tensor/construct::linspace` | `(p_float,p_float,i64)->tensor[n,p_float]`",
            "`tensor/construct::stack` | `(List[tensor[..pre,..post,p]],i32)->tensor[..pre,rows,..post,p]`",
            "`test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}`",
            "`test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}`",
            "`test::assert_shape` | `(&tensor[..r,p],List[i64],string)->unit!{Test}`",
        ):
            with self.subTest(signature=signature):
                self.assertIn(signature, block)

    def test_stdlib_tensor_manifest_cannot_restore_fixed_current_ranks(self) -> None:
        path = self.root / "spec/registry/stdlib_numeric_manifest.md"
        mutations = (
            (
                "(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}",
                "(&tensor[n,p_float],p_float)->tensor[n,p_float]!{Random}",
            ),
            (
                "(&tensor[..pre,1,..post,p],i32)->tensor[..pre,..post,p]",
                "(&tensor[a,1,b,p],i32)->tensor[a,b,p]",
            ),
            (
                "(List[tensor[..pre,..post,p]],i32)->tensor[..pre,rows,..post,p]",
                "(List[tensor[d,p]],i32)->tensor[rows,d,p]",
            ),
            (
                "(&tensor[..pre,..post,p],i32)->tensor[..pre,1,..post,p]",
                "(&tensor[d,p],i32)->tensor[1,d,p]",
            ),
            (
                "(&tensor[..r,bool])->tensor[hits,i64]",
                "(&tensor[n,bool])->tensor[hits,i64]",
            ),
            (
                "| `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |",
                "| `test::assert_close_tensor` | "
                "`(&tensor[n,p_float],&tensor[n,p_float],p_float,string)->unit!{Test}` |",
            ),
            (
                "| `test::assert_eq_tensor` | "
                "`(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |",
                "| `test::assert_eq_tensor` | "
                "`(&tensor[n,p],&tensor[n,p],string)->unit!{Test}` |",
            ),
            (
                "(&tensor[..r,p],List[i64],string)->unit!{Test}",
                "(&tensor[n,p],i64,string)->unit!{Test}",
            ),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("05-OP-35.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_reshape_and_assert_shape_narratives_are_rank_total(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "all three are rank-polymorphic, bit-preserving reshape/concat"
                "\n> operations",
                "all three support only their current example ranks",
                "rank-polymorphic",
            ),
            (
                "compares its length and every\n> entry to the tensor's complete "
                "shape in axis order",
                "compares only the first extent",
                "complete shape",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"OP-35.*{message}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_linspace_count_one_adjoint_and_leaf_order_are_frozen(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`linspace` with count one, start receives the sole output cotangent "
                "and stop\n> receives exact zero",
                "`linspace` with count one splits the cotangent between both endpoints",
                "sole output cotangent",
            ),
            (
                "each enumerated by increasing\n> output index",
                "each enumerated in implementation-defined order",
                "increasing output index",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"OP-35.*{message}")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_stdlib_exact_domain_blanket_cannot_capture_decimal_or_calendar(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Every primitive-width intermediate in a graph whose contract names a dtype",
            "Every numeric intermediate in every stdlib operation",
        )
        self.assert_contract_fails("OP-35.*primitive-width")

    def test_date_difference_orientation_is_frozen(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`days_between(lhs,rhs) = ordinal(rhs) - ordinal(lhs)`",
            "`days_between(lhs,rhs) = ordinal(lhs) - ordinal(rhs)`",
        )
        self.assert_contract_fails("OP-35.*days_between")

    def test_duration_overflow_is_on_the_final_days_field(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "final normalized `days` field has no i64 representation",
            "any intermediate component total exceeds i64",
        )
        self.assert_contract_fails("OP-35.*final normalized")

    def test_assert_close_finite_pairs_use_inclusive_tolerance(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "is close exactly when that difference is less than or equal\n"
            "> to the converted tolerance",
            "is close under an implementation-defined comparison",
        )
        self.assert_contract_fails("OP-35.*less than or equal")

    def test_negative_year_canonical_digit_count_is_exact(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "A negative year uses `-` followed by exactly\n"
            "> `max(4, digits(|year|))` decimal digits, where `|year|` is the exact\n"
            "> mathematical magnitude rather than an i64 `abs`",
            "A negative year uses an implementation-defined number of digits",
        )
        self.assert_contract_fails("OP-35.*negative year")

    def test_tokenizer_merge_pairs_are_unique_across_ranks(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "no token\n> pair occurs at more than one merge rank",
            "a token pair may occur at multiple merge ranks",
        )
        self.assert_contract_fails("OP-35.*token pair")

    def test_tokenizer_merge_encode_decode_rules_are_frozen(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "repeatedly selects the lowest merge\n> rank and then the leftmost pair",
                "repeatedly selects an implementation-defined merge pair",
            ),
            (
                "`encode` maps each final token through `vocab`",
                "`encode` may assign an implementation-defined ID",
            ),
            (
                "`decode` maps each ID through the\n> inverse vocabulary",
                "`decode` may drop unknown IDs",
            ),
        )
        for old, new in mutations:
            with self.subTest(old=old):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails("OP-35")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_bool_counting_has_no_explicit_cast_compatibility_idiom(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "`and` / `or` / `not` are the\n  logical operations and `count` is the "
            "bool-tensor counting operation",
            "`and` / `or` / `not` are the logical operations; counting is the "
            "explicit-cast idiom",
        )
        # A required literal retains this clause without relying on Git history
        # or text digests; the independent copied-tree control below proves it.
        self.assert_contract_fails("bool counting operation")

    def test_num8_keeps_representation_distinct_from_width(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "Equal storage widths do not make two representations interchangeable",
            "Equal storage widths make two representations interchangeable",
        )
        self.assert_contract_fails("04-NUM-8.*representations interchangeable")

    def test_num11_preserves_device_and_binding_metadata_domains(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "A language binding or device descriptor SHALL preserve rank as i32\n"
            "> and each extent, stride, element count, and byte capacity as i64",
            "A language binding or device descriptor MAY narrow rank, extents,\n"
            "> strides, element counts, and byte capacities to an implementation width",
        )
        self.assert_contract_fails("04-NUM-11.*device descriptor")

    def test_shape_capacity_equivalence_never_saturates_into_equality(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "SHALL NOT wrap, saturate, truncate, or\n"
            "> substitute an overflow sentinel that can make unequal mathematical counts\n"
            "> equal",
            "MAY saturate both overflowing products to one sentinel and treat them\n"
            "> as equal",
        )
        self.assert_contract_fails("04-SHAPE-1.*overflow sentinel")

    def test_count_axes_remain_signature_checks_not_table_b_narrowing(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "axis values are not table axes",
            "each axis value becomes a target-specific Table B row",
        )
        self.assert_contract_fails("count axes stay in the signature rule")

    def test_count_child_oracle_receipt_is_required(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "DTYPE COUNT ORACLE: PASS",
            "DTYPE COUNT ORACLE: SKIPPED",
        )
        self.assert_contract_fails("Count child oracle success line")

    def test_runtime_axis_and_host_rows_keep_concrete_owners(self) -> None:
        path = self.root / "spec/design/capability_table.md"
        mutations = (
            (
                "runtime-axis `shape`, `ReduceWindow`, and `ReduceWindowGrad` "
                "([05-OP-7], [05-SHAPE-1], [05-OP-39], [#1298])",
                "runtime-axis operations ([#729])",
                "runtime-axis and window capability owner",
            ),
            (
                "no host fallback, permanent target rejection, zero adjoint, or "
                "signature narrowing is an implementation receipt",
                "a host fallback or narrowed signature counts as implemented",
                "runtime-axis target-independent signature",
            ),
            (
                "host numeric builtins `tensor_scan`, `process_run`, and generic "
                "assertions ([05-OP-38], [#1297])",
                "host numeric builtins ([#729])",
                "host numeric capability owner",
            ),
            (
                "Unit, tuple, `Dict`, `Option`, ADT, function, deferred, and "
                "resource cases are semantic `Rejected` rows",
                "Unit is a supported sibling case",
                "to_string rejected cases",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_effect_kind_has_no_source_compatibility_reexport(self) -> None:
        self.replace(
            Path("spec/design/loud_unsupported.md"),
            "consume `chelis_vocab::EffectKind` directly; no `chelis_types` re-export",
            "temporarily re-export `EffectKind` for source compatibility",
        )
        self.assert_contract_fails("EffectKind direct ownership")

    def test_wire_schema_has_no_versionless_compatibility_path(self) -> None:
        self.replace(
            Path("spec/10-serialization.md"),
            "There is no versionless default, legacy migration, additive-",
            "A missing version defaults to v1; legacy migration and additive-",
        )
        self.assert_contract_fails("wire versionless rejection")

    def test_wire_count_axes_are_canonical_and_fail_closed(self) -> None:
        path = self.root / "spec/10-serialization.md"
        mutations = (
            (
                "complete\nnon-empty vector of unique normalized original-axis positions "
                "in strictly\ndescending order",
                "possibly empty source-order vector with duplicate axes",
                "wire count canonical axes",
            ),
            (
                "decoder rejects an empty,\nduplicate, increasing, or out-of-range "
                "vector before IR construction",
                "decoder normalizes malformed axis vectors after IR construction",
                "wire count axis rejection",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_wire_pad_uses_only_the_exact_tagged_scalar_payload(self) -> None:
        path = self.root / "spec/10-serialization.md"
        mutations = (
            (
                "`WireRiscOp::Pad { fill: ScalarValue, ... }`",
                "`WireRiscOp::Pad { fill: f64, ... }`",
                "wire typed Pad variant",
            ),
            (
                "a raw JSON number, an untagged payload, a string-mode\n"
                "fill, or a mismatched dtype is a decode error before IR construction",
                "raw numbers and mismatched dtypes are converted during decode",
                "wire Pad payload rejection",
            ),
            (
                "No\nnumeric-fill migration or inferred fill dtype exists",
                "A v5 numeric fill migrates by inferring f64",
                "wire Pad no compatibility",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_phase_handoff_requires_zero_capacity_exceptions(self) -> None:
        self.replace(
            Path("spec/design/dtype_semantics.md"),
            "This is the executable requirement for zero capacity exceptions.",
            "the frozen legacy dispositions remain accepted",
        )
        self.assert_contract_fails("zero capacity exceptions")

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

    def test_atom_closure_declares_builtin_domains_before_phase4c(self) -> None:
        path = self.root / "spec/design/dtype_semantics.md"
        original = path.read_text(encoding="utf-8")
        old = (
            "[#1294] first introduces the closed builtin domain/case declaration "
            "types and\nattaches a non-empty exhaustive declaration to every "
            "`BuiltinDecl`"
        )
        self.assertIn(old, original)
        path.write_text(
            original.replace(
                old,
                "Phase 4C may infer missing builtin domains from implementation code",
                1,
            ),
            encoding="utf-8",
        )
        try:
            self.assert_contract_fails("pre-4C builtin domain declarations")
        finally:
            path.write_text(original, encoding="utf-8")

        self.replace(
            Path("spec/design/capability_table.md"),
            "The domain and case declarations\n  themselves are [#1294] prerequisite "
            "artifacts",
            "Phase 4C invents the domain and case declarations while populating rows",
        )
        self.assert_contract_fails("capability pre-4C builtin declarations")

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
        self.assert_changed_contract_requires_acknowledgement("region", "capability schema")

    def test_effect_registry_covers_every_fixed_effect(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "`Random | Accum | IO | Test | Resource(ResourceId)`",
            "`Random | Accum | IO | Resource(ResourceId)`",
        )
        self.assert_contract_fails("closed effect requirement domain")

    def test_effect_registry_has_no_default_disposition(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "There is no\nmissing-row, wildcard, or default disposition.",
            "A missing effect row defaults to `Implemented`.",
        )
        self.assert_contract_fails("effect no-default rule")

    def test_device_reduction_rows_cannot_cite_the_tracking_hub(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #2339, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("device reduction implementation owner")

    def test_logical_rows_cannot_cite_the_tracking_hub(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #2266, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("logical implementation owner")

    def test_product_rows_retain_their_concrete_owner(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "Unimplemented { issue: #1290, diagnostic_kind: UnsupportedFeature }",
            "Unimplemented { issue: #729, diagnostic_kind: UnsupportedFeature }",
        )
        self.assert_contract_fails("product implementation owner")

    def test_to_string_checker_narrowing_retains_its_concrete_owner(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "[#1282] owns checker/evaluator alignment",
            "[#729] owns checker/evaluator alignment",
        )
        self.assert_contract_fails("to_string checker owner")

    def test_loud_unsupported_must_consume_external_target_key(self) -> None:
        self.replace(
            Path("spec/design/loud_unsupported.md"),
            "(ExternalCallableFamily, CanonicalCallableId, "
            "ExternalTargetContext)",
            "(ExternalCallableFamily, CanonicalCallableId)",
        )
        self.assert_contract_fails("loud external target key")

    def test_unsupported_diagnostics_carry_semantic_or_capability_authority(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "A language-rejected case cites the normative atom that decides\n> it",
                "A language rejection cites its implementation issue",
                "05-UNS-5.*normative atom",
            ),
            (
                "A legal operation unavailable on the selected target identifies the\n"
                "> exact typed capability cell",
                "A target gap cites only the tracking issue",
                "05-UNS-5.*typed capability cell",
            ),
            (
                "it is not semantic\n> authority and does not alter legality",
                "issue metadata is semantic authority and may alter legality",
                "05-UNS-5.*not semantic authority",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

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

    def test_builtin_atom_closure_is_an_all_or_nothing_pre_4c_gate(self) -> None:
        path = self.root / "spec/design/dtype_semantics.md"
        mutations = (
            (
                "Before any Phase 4C key/cell type, authoring macro, or partial "
                "machine row may\nland",
                "Phase 4C key and cell types may land while semantic closure proceeds",
                "precedes every partial Phase 4C mechanism",
            ),
            (
                "discovers the union of every canonical Table-A IR/RISC operation "
                "identity and\nevery declared `BuiltinDecl` sibling domain/case and "
                "proves an exact\nbijection from every Table-A and sibling-builtin "
                "identity to one\nsemantically governing normative `[05-OP-N]` line",
                "samples common BuiltinDecl cases and reports their nearest prose",
                "total exact builtin-atom bijection",
            ),
            (
                "It authors every missing\natom, regenerates the rejection registry",
                "It records missing atoms as follow-up implementation work",
                "authors and regenerates",
            ),
            (
                "admits no count allowlist,\nunnumbered table/prose authority, issue "
                "citation, default, alias, age, or\ncompatibility exception",
                "admits a count allowlist and issue citations for older builtins",
                "zero authority exceptions",
            ),
            (
                "An open implementation issue can authorize only a\nlater Table-B "
                "`Unimplemented` receipt; it never satisfies semantic closure",
                "An open implementation issue may satisfy semantic closure",
                "Table-B-only",
            ),
            (
                "The oracle and its\nadversarial mutations must be green and merged "
                "before Phase 4C begins",
                "The oracle may become green after Phase 4C begins",
                "atom-closure merge gate",
            ),
            (
                "the [#1296] composite pre-4C oracle is green, merged, and\n"
                "wired to the normal gate",
                "the [#1296] composite pre-4C oracle is planned",
                "Phase 4C composite entry condition",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_composite_pre_4c_gate_cannot_skip_or_waive_a_prerequisite(self) -> None:
        path = self.root / "spec/design/dtype_semantics.md"
        mutations = (
            (
                "After the individual behavior, storage, census, and [#1294] "
                "atom-closure\n"
                "oracles land",
                "Before individual prerequisite oracles land",
                "follows all prerequisite oracles",
            ),
            (
                "fails for a missing, duplicate,\nskipped, stale, nonzero, or "
                "success-line-free leg",
                "may skip stale or unavailable legs",
                "runner totality",
            ),
            (
                "It is wired to the normal gate; prose coverage\n"
                "or a manual waiver is not an entry receipt",
                "A prose review or manual waiver is an entry receipt",
                "normal-gate wiring",
            ),
            (
                "It includes [#1294]'s exact builtin-atom closure",
                "It may omit [#1294]'s exact builtin-atom closure",
                "entry includes exact atom closure",
            ),
            (
                "[#1284], [#1287]-[#1298], and [#1306]",
                "[#1284], [#1287]-[#1298]",
                "includes every late prerequisite",
            ),
            (
                "chelis#1295's all-active-float random/rounding rules, chelis#1297's "
                "compiled\n"
                "host effects, chelis#1298's runtime-axis/window operations, and "
                "chelis#1306's\n"
                "direct subtraction/extrema identities have landed",
                "chelis#1295 and chelis#1298 may land after table population",
                "issue-owned behavior prerequisites",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_roadmap_must_keep_concrete_reduction_owners(self) -> None:
        self.replace(
            Path("spec/design/remediation_roadmap.md"),
            "[#1281] owns the remaining reduction rows",
            "[#729] owns the remaining reduction rows",
        )
        self.assert_contract_fails("roadmap reduction owner")

    def test_roadmap_keeps_the_runtime_representation_owner(self) -> None:
        self.replace(
            Path("spec/design/remediation_roadmap.md"),
            "[`runtime_representation.md`](runtime_representation.md) ([#893])",
            "[#893] has no design owner",
        )
        self.assert_contract_fails("runtime representation owner")

    def test_runtime_capacity_key_preserves_partial_division_domain(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "Division is a partial exact-integer operation, not rational arithmetic",
            "Division is normalized as unrestricted rational arithmetic",
        )
        self.assert_contract_fails("runtime capacity validity domain")

    def test_runtime_zero_product_preserves_nested_partial_division(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "`0 * (1 / n)` retains the partial quotient and remains distinct "
            "from zero\nunless `n != 0` and `n` divides 1 have both been proved",
            "`0 * (1 / n)` always collapses to zero",
        )
        self.assert_contract_fails("runtime zero-product validity domain")

    def test_runtime_capacity_key_is_carrier_independent(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "`CapacityKey` is deliberately carrier-independent",
            "`CapacityKey` is coupled to `DimExpr`",
        )
        self.assert_contract_fails("runtime capacity carrier independence")

    def test_runtime_guard_fact_stays_outside_capacity_predicates(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "The set is deliberately closed against equality learned only by "
            "passing a\n[#1277] runtime guard",
            "A passed runtime guard silently proves every later capacity reuse",
        )
        self.assert_contract_fails("runtime guard capacity boundary")

    def test_runtime_extent_plan_disclaims_capacity_equality(self) -> None:
        self.replace(
            Path("spec/design/runtime_extents.md"),
            "capacity and reuse equality\nover typed extent expressions\n"
            "([`runtime_representation.md`](runtime_representation.md), [#888])",
            "capacity and reuse equality are part of this plan",
        )
        self.assert_contract_fails("runtime extent capacity boundary")

    def test_runtime_phase0_debt_is_shrink_only(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "exact shrink-only transition-debt\nmanifest",
            "editable transition-debt\nmanifest",
        )
        self.assert_contract_fails("runtime Phase 0 shrink-only debt")

    def test_foreign_carrier_never_forms_rust_slices(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "A foreign carrier never constructs `TensorRef<T>`, `TensorMut<T>`, "
            "`&[T]`, or\n`&mut [T]`",
            "A foreign carrier may construct `TensorMut<T>` and `&mut [T]`",
        )
        self.assert_contract_fails("runtime foreign slice prohibition")

    def test_runtime_launch_gate_keeps_the_host_field_seal(self) -> None:
        self.replace(
            Path("spec/design/runtime_representation.md"),
            "the Phase 3 `--host` field-seal evidence",
            "no field-seal evidence",
        )
        self.assert_contract_fails("runtime launch-gate boundary")

    def test_additive_parent_absorption_clause_fails(self) -> None:
        self.replace(
            Path("spec/design/remediation_roadmap.md"),
            "([#1290] replaces noncanonical product/sum trees and is also part of "
            "[#170]; [#1281] owns the remaining reduction rows)",
            "([#1290] replaces noncanonical product/sum trees and is also part of "
            "[#170]; [#1281] owns the "
            "remaining reduction rows; the parent [#729] MAY silently absorb "
            "and close either child's work without a separate receipt)",
        )
        self.assert_changed_contract_requires_acknowledgement('region', 'roadmap ownership')

    def test_status_must_keep_external_execution_authority(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "total external target-disposition registry",
            "semantic registry alone",
        )
        self.assert_contract_fails("status external target authority")

    def test_surface_uses_exact_language_IO_spelling(self) -> None:
        path = Path("docs/CHELIS_SURFACE.md")
        original = (self.root / path).read_text(encoding="utf-8")
        mutated = original.replace("introduces `IO`", "introduces `Io`", 1).replace(
            "| `IO` | file ops", "| `Io` | file ops", 1
        )
        self.assertNotEqual(mutated, original)
        (self.root / path).write_text(mutated, encoding="utf-8")
        try:
            self.assert_contract_fails("surface IO spelling")
        finally:
            (self.root / path).write_text(original, encoding="utf-8")

    def test_status_execution_basis_is_the_reviewed_base(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "this revision is `main` at `4e200061`",
            "this revision is `main` at `1c52c05e`",
        )
        self.assert_contract_fails("status reviewed execution basis")

    def test_status_release_distance_matches_the_reviewed_base(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "`main` is eight commits ahead",
            "`main` is six commits ahead",
        )
        self.assert_contract_fails("status release distance")

    def test_status_issue_graph_counts_are_cross_checked(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "| #730 | 17 / 37 |",
            "| #730 | 16 / 37 |",
        )
        self.assert_contract_fails("sum to 54 open issues")

    def test_status_keeps_stdlib_alignment_owner(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "#1293 aligns all 83 recursively discovered stdlib numeric definitions",
            "the parent tracker implicitly owns stdlib alignment",
        )
        self.assert_contract_fails("status stdlib alignment owner")

    def test_recursive_runtime_printing_has_byte_exact_structural_grammar(self) -> None:
        path = Path("spec/05-risc-primitives.md")
        mutations = (
            (
                "A tuple\n> renders as `()` when it has no fields, `(R(v),)` when it has one",
                "A tuple uses an implementation-defined display",
                "OP-32.*A tuple renders",
            ),
            (
                "A dictionary renders entries in the canonical key order above as",
                "A dictionary renders in insertion order as",
                "OP-32.*A dictionary renders",
            ),
            (
                "An ADT renders its exact stored constructor-name bytes followed by `(`",
                "An ADT renders an implementation-defined debug name",
                "OP-32.*An ADT renders",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = (self.root / path).read_text(encoding="utf-8")
                self.assertIn(old, original)
                (self.root / path).write_text(
                    original.replace(old, new, 1), encoding="utf-8"
                )
                try:
                    self.assert_contract_fails(message)
                finally:
                    (self.root / path).write_text(original, encoding="utf-8")

    def test_gradient_contribution_order_uses_canonical_forward_ordinals(self) -> None:
        path = Path("spec/06-transformations.md")
        mutations = (
            (
                "canonical forward node ordinal, then by input-slot index",
                "the order returned by the current topological sort",
                "canonical consumer-edge order",
            ),
            (
                "independent of the work-list or topological-sort tie order",
                "may vary with the topological-sort tie order",
                "topological-sort independence",
            ),
            (
                "stable_topological_order(N, tie_break=canonical_forward_ordinal)",
                "topological_sort(N)",
                "stable formal traversal",
            ),
            (
                "values_sorted_by_key(contributions[n])",
                "values(contributions[n])",
                "key-sorted formal accumulation",
            ),
            (
                "contributions[n_j][(canonical_forward_ordinal(n_i), input_slot)] =\n"
                "    contribution_from_n_i",
                "adjoint[n_j] = Add(adjoint[n_j], contribution_from_n_i)",
                "keyed backward-traversal contribution queue",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                original = (self.root / path).read_text(encoding="utf-8")
                self.assertIn(old, original)
                (self.root / path).write_text(
                    original.replace(old, new, 1), encoding="utf-8"
                )
                try:
                    self.assert_contract_fails(message)
                finally:
                    (self.root / path).write_text(original, encoding="utf-8")

    def test_sparse_indices_use_every_active_signed_integer_width(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "`ScatterElements` SHALL take an index tensor of any active signed-integer dtype",
            "`ScatterElements` SHALL take only i32 or i64 indices",
        )
        self.assert_contract_fails("05-SPARSE-1.*active signed-integer")

    def test_effect_requirement_domain_uses_language_IO_spelling(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "`Random | Accum | IO | Test | Resource(ResourceId)`",
            "`Random | Accum | Io | Test | Resource(ResourceId)`",
        )
        self.assert_contract_fails("closed effect requirement domain")

    def test_count_uses_a_truthful_integer_reduction_AD_reason(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "AdRejectionReason::IntegerReductionOutput",
            "AdRejectionReason::IntegerIndexOutput",
        )
        self.assert_contract_fails("05-OP-29.*IntegerReductionOutput")

    def test_max_elem_ties_and_reference_pseudocode_select_the_first_operand(
        self,
    ) -> None:
        path = Path("spec/05-risc-primitives.md")
        original = (self.root / path).read_text(encoding="utf-8")
        mutations = (
            (
                "| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | "
                "Element-wise maximum | Whole `g` flows to the operand selected by "
                "[05-OP-40] |",
                "| `max_elem` | `(&tensor[D,p], &tensor[D,p]) -> tensor[D,p]` | "
                "Element-wise maximum | `(g * (x >= y), g * (x < y))` |",
                "max_elem table tie adjoint",
            ),
            (
                "On\n> floats, it returns the first NaN in operand order when either "
                "operand is NaN",
                "Float max/min use target-native NaN selection",
                "max_elem first-NaN selection",
            ),
            (
                "`max_elem` routes the whole cotangent to\n"
                "the first operand when the inputs are equal",
                "`max_elem` has a zero gradient when the inputs are equal",
                "max_elem first-operand tie adjoint",
            ),
            (
                "out[i] = select_max_first(a[i], b[i]);",
                "out[i] = a[i] > b[i] ? a[i] : b[i];",
                "max_elem first-operand reference selection",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                self.assertIn(old, original)
                (self.root / path).write_text(
                    original.replace(old, new, 1), encoding="utf-8"
                )
                try:
                    self.assert_contract_fails(message)
                finally:
                    (self.root / path).write_text(original, encoding="utf-8")

    def test_reference_pseudocode_cannot_claim_semantic_authority(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "illustrative\nimplementation shapes for the C backend; it is not a "
            "semantic oracle",
            "the direct reference implementation used as the C-backend semantic "
            "oracle",
        )
        self.assert_contract_fails("illustrative reference status")

    def test_extrema_have_one_exact_selection_atom(self) -> None:
        blocks = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )
        self.assertIn("05-OP-40", blocks)
        body = oracle.normalize_atom_body(blocks["05-OP-40"])
        for required in (
            "active signed-integer or float",
            "first NaN in operand order",
            "exact stored bits",
            "first operand on every equality",
            "signed-zero equality",
            "whole cotangent to the selected operand",
            "no accumulator",
            "never lowers through arithmetic negation",
        ):
            with self.subTest(required=required):
                self.assertIn(required, body)

    def test_sub_has_direct_checked_operation_semantics(self) -> None:
        blocks = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )
        self.assertIn("05-OP-41", blocks)
        body = oracle.normalize_atom_body(blocks["05-OP-41"])
        for required in (
            "exact mathematical difference",
            "traps `Overflow`",
            "direct checked subtraction",
            "never lowers through `neg`",
            "adjoint is `(g, neg(g))`",
            "no accumulator",
        ):
            with self.subTest(required=required):
                self.assertIn(required, body)

    def test_extrema_and_sub_cannot_regain_arithmetic_surrogate_lowerings(
        self,
    ) -> None:
        path = Path("spec/05-risc-primitives.md")
        original = (self.root / path).read_text(encoding="utf-8")
        mutations = (
            (
                "`sub` is a Tier 1 primitive governed by [05-OP-41]",
                "`sub(a, b)` lowers to `add(a, neg(b))`",
                "direct sub lowering",
            ),
            (
                "`min_elem` is a Tier 1 primitive governed by [05-OP-40]",
                "`min_elem(a, b)` lowers to `neg(max_elem(neg(a), neg(b)))`",
                "direct min_elem lowering",
            ),
        )
        for old, new, message in mutations:
            with self.subTest(message=message):
                self.assertIn(old, original)
                (self.root / path).write_text(
                    original.replace(old, new, 1), encoding="utf-8"
                )
                try:
                    self.assert_contract_fails(message)
                finally:
                    (self.root / path).write_text(original, encoding="utf-8")

    def test_captured_transformations_match_the_extrema_tie_rule(self) -> None:
        self.replace(
            Path("openspec/specs/transformations/spec.md"),
            "the selected operand receives the whole cotangent,\n  including the first "
            "operand on equality",
            "the gradient is zero at the non-differentiable point",
        )
        self.assert_contract_fails("captured extrema tie rule")

    def test_dtype_family_bound_production_is_a_closed_three_name_set(
        self,
    ) -> None:
        self.replace(
            Path("spec/02-surf-syntax.md"),
            "DtypeFamily   <- 'Float' / 'Int' / 'Numeric'",
            "DtypeFamily   <- TypeName",
        )
        self.assert_contract_fails("Surf dtype-family bound production")

    def test_declaration_binder_list_cannot_be_incomplete(self) -> None:
        self.replace(
            Path("spec/02-surf-syntax.md"),
            "A `sig`'s `[..]` clause is complete: every `t-var`, `d-var`, and\n"
            "`d-rank` name in the signature appears exactly once.",
            "A `sig`'s `[..]` clause is advisory: variables may be omitted.",
        )
        self.assert_contract_fails("Surf complete declaration binder list")

    def test_the_occurrence_rule_cannot_widen_past_bounded_binders(self) -> None:
        # [04-DTYPE-2] makes only a BOUNDED binder owe an occurrence, and the
        # checker agrees: `sig f[zz]: i32 -> i32` checks clean. Asserting it
        # for every listed name is normative prose broader than the decision.
        self.replace(
            Path("spec/02-surf-syntax.md"),
            "A listed name **that declares\na bound** must occur in the declared type.",
            "A listed name must occur in the declared type.",
        )
        self.assert_contract_fails("Surf occurrence rule is bounded-binder only")

    def test_one_declaration_cannot_have_two_binder_lists(self) -> None:
        self.replace(
            Path("spec/02-surf-syntax.md"),
            "One binder list owns each\ndeclaration: a standalone `sig` carries it, "
            "and a matching `def` must\nnot carry a second list.",
            "Both a standalone `sig` and its matching `def` may carry binder lists.",
        )
        self.assert_contract_fails("Surf single declaration binder-list owner")

    def test_the_two_bound_failures_cannot_claim_one_diagnostic_shape(
        self,
    ) -> None:
        # An instantiation outside the bound names one family and the
        # offending type; an empty intersection names two families and no
        # offending type. One clause covering both over-promises the second.
        self.replace(
            Path("spec/04-type-system.md"),
            "an empty intersection SHALL be a `PrecisionMismatch` naming both\n"
            "> families.",
            "an empty intersection SHALL name the required family and the\n"
            "> offending type.",
        )
        self.assert_contract_fails("empty intersection names both families")

    def test_deep_carries_dtype_bounds_as_a_defined_metadata_key(self) -> None:
        self.replace(
            Path("spec/03-deep-syntax.md"),
            "| `dtype_bounds` | metadata map | Dtype-family bounds on a "
            "`defsig`'s binders; see §2.2 |",
            "| `dtype_bounds` | string | Producer-specific bound provenance |",
        )
        self.assert_contract_fails("Deep dtype-family bound metadata key")

    def test_bounded_variables_unify_by_intersection_not_equality(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "Unifying two bounded variables SHALL\n> yield the intersection "
            "of their families.",
            "Unifying two bounded variables SHALL\n> require identical families.",
        )
        self.assert_contract_fails("dtype-family bound intersection")

    def test_an_unbounded_binder_is_not_narrowed_to_a_dtype(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "A binder that declares no bound\n> remains an unconstrained type "
            "variable admitting every type, not only a\n> dtype.",
            "A binder that declares no bound\n> ranges over every active dtype.",
        )
        self.assert_contract_fails(
            "unbounded binder stays a general type variable"
        )

    def test_deep_grammar_must_derive_a_map_valued_metadata_key(self) -> None:
        # The chapter's own PEG has to derive the Deep the language emits.
        # `chelis validate --deep` is the second implementation of exactly
        # this production, and it rejected every migrated stdlib module while
        # `MetaValue` had no nested-`Meta` alternative.
        self.replace(
            Path("spec/03-deep-syntax.md"),
            "MetaValue   \u2190 Meta / Node / Literal / Identifier / TypeName",
            "MetaValue   \u2190 Node / Literal / Identifier / TypeName",
        )
        self.assert_contract_fails(
            "Deep grammar derives a map-valued metadata key"
        )

    def test_deep_grammar_must_derive_the_declared_metadata_key_charset(
        self,
    ) -> None:
        # §1.1 declares `[A-Za-z_][A-Za-z0-9_]*`; the pre-#1417 §7 production
        # was `[a-z]+`, which derives neither `dtype_bounds` nor the four
        # underscored keys already shipping.
        self.replace(
            Path("spec/03-deep-syntax.md"),
            "MetaKey     \u2190 [A-Za-z_] [A-Za-z0-9_]*",
            "MetaKey     \u2190 [a-z]+",
        )
        self.assert_contract_fails(
            "Deep grammar derives the declared metadata key charset"
        )

    def test_stdlib_bound_obligation_cannot_cite_the_operation_class_table(
        self,
    ) -> None:
        # §5.4's rows are operation classes, not signatures. Citing it would
        # make `arange`'s `Int` bound optional and `assert_close`'s `Float`
        # bound wrong, contradicting [05-OP-35].
        self.replace(
            Path("spec/04-type-system.md"),
            "A public stdlib signature whose `[05-OP-35]` registry domain is "
            "exactly one of\nthese families declares that family as a bound.",
            "A public stdlib signature whose §5.4 row admits exactly one "
            "family declares\nthat family as a bound.",
        )
        self.assert_contract_fails(
            "stdlib bound obligation cites the registry domain"
        )

    def test_numeric_family_is_the_union_of_float_and_int(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "| `Numeric` | the union of `Float` and `Int` |",
            "| `Numeric` | every active float dtype of §1.1 |",
        )
        self.assert_contract_fails("dtype-family membership table")


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


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # These tiny repositories are deleted as soon as the command returns.
    # Automatic maintenance can detach and write .git/objects after that
    # boundary (chelis#1970). It has no role in the fixture's merge oracle.
    # Keep the setting invocation-local; never change developer Git config
    # or hide a real TemporaryDirectory cleanup failure.
    return subprocess.run(
        ("git", "-c", "maintenance.auto=false", "-C", str(root), *args),
        capture_output=True,
        text=True,
        check=False,
    )


def _git_ok(root: Path, *args: str) -> str:
    completed = _git(root, *args)
    if completed.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {root}: {completed.stderr}"
        )
    return completed.stdout


def _commit_all(root: Path, message: str) -> str:
    _git_ok(root, "add", "-A")
    _git_ok(
        root,
        "-c",
        "user.name=Frozen Contract Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--no-verify",
        "-q",
        "-m",
        message,
    )
    return _git_ok(root, "rev-parse", "HEAD").strip()


class GitFixtureLifetimeTests(unittest.TestCase):
    """A disposable fixture must not leave optional Git writers behind it."""

    def test_fixture_commits_do_not_launch_automatic_maintenance(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name) / "repo"
            root.mkdir()
            _git_ok(root, "init", "-q", "-b", "main")
            # Override any machine defaults and keep the raw control's
            # maintenance synchronous so the test itself owns every process.
            for key, value in (
                ("maintenance.auto", "true"),
                ("maintenance.autoDetach", "false"),
                ("gc.autoDetach", "false"),
                ("user.name", "Fixture Lifetime Test"),
                ("user.email", "test@example.invalid"),
                ("commit.gpgsign", "false"),
            ):
                _git_ok(root, "config", key, value)
            (root / "file").write_text("fixture\n", encoding="utf-8")

            def maintenance_children(trace: Path) -> list[dict]:
                events = [
                    json.loads(line)
                    for line in trace.read_text(encoding="utf-8").splitlines()
                ]
                self.assertTrue(any(event["event"] == "start" for event in events))
                return [
                    event for event in events
                    if event["event"] == "child_start"
                    and "maintenance" in event.get("argv", [])
                ]

            fixture_trace = Path(name) / "fixture.jsonl"
            with mock.patch.dict(os.environ, {"GIT_TRACE2_EVENT": str(fixture_trace)}):
                _commit_all(root, "fixture commit")

            # Negative control: the same real repository and enabled config
            # must expose maintenance if the fixture helper is bypassed.
            control_trace = Path(name) / "control.jsonl"
            with mock.patch.dict(os.environ, {"GIT_TRACE2_EVENT": str(control_trace)}):
                subprocess.run(
                    ("git", "-C", str(root), "commit", "--no-verify", "-q",
                     "--allow-empty", "-m", "unguarded control"),
                    check=True, capture_output=True, text=True,
                )
            self.assertTrue(maintenance_children(control_trace))
            self.assertEqual(maintenance_children(fixture_trace), [])


class AcknowledgementGrammarTests(unittest.TestCase):
    """The line grammar is exact, case-sensitive, and glob-free."""

    def parse(self, body: str) -> tuple[list[str], list[str]]:
        return oracle.parse_acknowledgements(body)

    def test_canonical_line_is_accepted(self) -> None:
        paths, errors = self.parse(
            "Some prose.\n"
            "Frozen-contract-change: spec/04-type-system.md\n"
            "Frozen-contract-change: AGENTS.md\n"
        )
        self.assertEqual(paths, ["spec/04-type-system.md", "AGENTS.md"])
        self.assertEqual(errors, [])

    def test_carriage_returns_from_a_github_body_are_tolerated(self) -> None:
        # The pull request body arrives from the GitHub event payload with
        # CRLF line endings; a lost acknowledgement here would read as an
        # unacknowledged change and block every PR that edits a contract file.
        paths, errors = self.parse(
            "Body.\r\nFrozen-contract-change: spec/11-ffi.md\r\nMore.\r\n"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(errors, [])

    def test_lowercase_key_is_a_malformed_line_not_a_silent_miss(self) -> None:
        paths, errors = self.parse("frozen-contract-change: spec/11-ffi.md\n")
        self.assertEqual(paths, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("malformed frozen contract acknowledgement", errors[0])

    def test_list_bullet_and_indentation_are_malformed_lines(self) -> None:
        for line in (
            "- Frozen-contract-change: spec/11-ffi.md",
            "  Frozen-contract-change: spec/11-ffi.md",
            "> Frozen-contract-change: spec/11-ffi.md",
            "* Frozen-contract-change: spec/11-ffi.md",
        ):
            with self.subTest(line=line):
                paths, errors = self.parse(line + "\n")
                self.assertEqual(paths, [])
                self.assertEqual(len(errors), 1)

    def test_missing_or_doubled_space_is_a_malformed_line(self) -> None:
        for line in (
            "Frozen-contract-change:spec/11-ffi.md",
            "Frozen-contract-change:  spec/11-ffi.md",
            "Frozen-contract-change: ",
            "Frozen-contract-change:",
        ):
            with self.subTest(line=line):
                paths, errors = self.parse(line + "\n")
                self.assertEqual(paths, [])
                self.assertEqual(len(errors), 1, errors)

    def test_trailing_content_after_the_path_is_a_malformed_line(self) -> None:
        paths, errors = self.parse(
            "Frozen-contract-change: spec/11-ffi.md (adds a sentence)\n"
        )
        self.assertEqual(paths, [])
        self.assertEqual(len(errors), 1)

    def test_globs_and_traversal_are_rejected_paths(self) -> None:
        for candidate in (
            "spec/*.md",
            "spec/0?-ffi.md",
            "spec/[01]1-ffi.md",
            "/spec/11-ffi.md",
            "spec/../spec/11-ffi.md",
            "./spec/11-ffi.md",
            "spec\\11-ffi.md",
            "spec//11-ffi.md",
        ):
            with self.subTest(candidate=candidate):
                paths, errors = self.parse(
                    f"Frozen-contract-change: {candidate}\n"
                )
                self.assertEqual(paths, [], candidate)
                self.assertEqual(len(errors), 1, candidate)

    def test_fenced_code_blocks_do_not_acknowledge(self) -> None:
        # A body has to be able to quote the grammar without acknowledging a
        # file, and a quoted example must not be mistaken for a real line.
        paths, errors = self.parse(
            "The grammar is:\n\n"
            "```\n"
            "Frozen-contract-change: spec/04-type-system.md\n"
            "frozen-contract-change: wrong case\n"
            "```\n\n"
            "Frozen-contract-change: spec/11-ffi.md\n"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(errors, [])

    def test_a_tilde_run_does_not_close_a_backtick_fence(self) -> None:
        # Round 1 F1. One boolean let any fence run close any other, so a line
        # that GitHub renders as code could still acknowledge a change.
        paths, errors = self.parse(
            "```\n"
            "~~~\n"
            "Frozen-contract-change: spec/04-type-system.md\n"
            "```\n"
        )
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])

    def test_a_short_run_does_not_close_a_longer_fence(self) -> None:
        # Round 1 F1. CommonMark requires the closing run to be at least as
        # long as the opening one, so an inner ``` stays inside an outer ````.
        paths, errors = self.parse(
            "````\n"
            "```\n"
            "Frozen-contract-change: spec/04-type-system.md\n"
            "```\n"
            "````\n"
        )
        self.assertEqual(paths, [])
        self.assertEqual(errors, [])

    def test_a_longer_run_closes_a_shorter_fence(self) -> None:
        paths, errors = self.parse(
            "```\n"
            "quoted\n"
            "````\n"
            "Frozen-contract-change: spec/11-ffi.md\n"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(errors, [])

    def test_a_tilde_fence_still_hides_its_contents(self) -> None:
        paths, errors = self.parse(
            "~~~\nFrozen-contract-change: spec/04-type-system.md\n~~~\n"
        )
        self.assertEqual((paths, errors), ([], []))

    def test_an_unclosed_fence_is_an_error_not_a_silent_swallow(self) -> None:
        # Round 1 F2. Without this the acknowledgement vanishes and the gate
        # tells the author to add a line the body already carries.
        paths, errors = self.parse(
            "```\nFrozen-contract-change: spec/11-ffi.md\n"
        )
        self.assertEqual(paths, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("unclosed", errors[0])

    def test_an_unclosed_fence_errors_even_beside_a_valid_line(self) -> None:
        # Round 2 RT2-5. Reporting the unclosed fence only when nothing parsed
        # survived the round-1 suite: an author who acknowledges one file and
        # then opens a fence loses every line after it with no diagnostic.
        paths, errors = self.parse(
            "Frozen-contract-change: spec/11-ffi.md\n"
            "```\n"
            "Frozen-contract-change: spec/10-serialization.md\n"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(len(errors), 1)
        self.assertIn("unclosed", errors[0])

    def test_trailing_whitespace_after_the_path_is_accepted(self) -> None:
        # Round 2 RT2-5. `raw.rstrip()` is the reason, and it was untested in
        # either direction, so replacing it with `rstrip("\n")` survived.
        paths, errors = self.parse(
            "Frozen-contract-change: spec/11-ffi.md   \t\n"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(errors, [])

    def test_lone_carriage_returns_still_separate_lines(self) -> None:
        # Round 2 RT2-5. Dropping the lone-CR normalization survived, because
        # every other test used LF or CRLF.
        paths, errors = self.parse(
            "Body.\rFrozen-contract-change: spec/11-ffi.md\rMore.\r"
        )
        self.assertEqual(paths, ["spec/11-ffi.md"])
        self.assertEqual(errors, [])

    def test_a_closed_fence_reports_no_unclosed_error(self) -> None:
        paths, errors = self.parse("```\nquoted\n```\n")
        self.assertEqual((paths, errors), ([], []))

    def test_a_mid_sentence_mention_acknowledges_nothing(self) -> None:
        paths, errors = self.parse(
            "Each change adds a `Frozen-contract-change: <path>` line.\n"
        )
        self.assertEqual((paths, errors), ([], []))


class FrozenContractChangeTests(unittest.TestCase):
    """The merge-base diff plus acknowledgement gate.

    Each test builds a real git repository so the check runs the same git
    plumbing it runs in CI.
    """

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        _git_ok(self.root, "init", "-q", "-b", "main")
        for relative in CONTRACT_FILES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO_ROOT / relative, destination)
        self.base = _commit_all(self.root, "baseline")
        _git_ok(self.root, "update-ref", "refs/remotes/origin/main", self.base)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def check(self, **kwargs: object) -> list[str]:
        parameters: dict[str, object] = {
            "root": self.root,
            "require_acknowledgement": True,
        }
        parameters.update(kwargs)
        return oracle.validate_frozen_contract_changes(**parameters)  # type: ignore[arg-type]

    def assert_fails(self, message: str, **kwargs: object) -> None:
        with self.assertRaisesRegex(oracle.OracleError, message):
            self.check(**kwargs)

    def append(self, relative: str, text: str) -> None:
        path = self.root / relative
        path.write_text(
            path.read_text(encoding="utf-8") + text, encoding="utf-8"
        )

    def test_unchanged_tree_passes(self) -> None:
        report = self.check()
        self.assertIn("0 of 31 contract files changed", report[0])

    def test_every_contract_file_is_watched(self) -> None:
        # The converted whole-file-digest test. Contradictory prose prepended
        # to any contract file must fail, and the failure must name the file.
        # The watched set is now all 30 CONTRACT_FILES, a superset of the 22
        # that carried a whole-file digest.
        self.assertEqual(len(CONTRACT_FILES), 31)
        for relative in oracle.CONTRACT_FILES:
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                path.write_text(
                    "An implementation MAY ignore the frozen Phase 4B "
                    "contract.\n\n" + original,
                    encoding="utf-8",
                )
                try:
                    self.assert_fails(
                        "unacknowledged frozen contract change: "
                        + re.escape(relative)
                    )
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_top_level_value_scope_is_frozen_in_both_owning_chapters(self) -> None:
        # [04-INF-4] and the spec/02 value-scope clause it qualifies move
        # together; weakening either half must trip the freeze. This arrived on
        # `main` asserting a whole-file digest mismatch. Neither clause is
        # inside a frozen region, so the acknowledgement gate is what catches
        # them now: same mutations, same guarantee.
        mutations = (
            (
                "spec/04-type-system.md",
                "body-type\n> metadata SHALL NOT make that later value visible",
                "body-type\n> metadata MAY make that later value visible",
            ),
            (
                "spec/04-type-system.md",
                "A reference to an\n> earlier value SHALL resolve",
                "A reference to an\n> earlier value MAY resolve",
            ),
            (
                "spec/02-surf-syntax.md",
                "a non-function value declared earlier in the enclosing module",
                "a non-function value declared anywhere in the enclosing module",
            ),
        )
        for relative, old, new in mutations:
            with self.subTest(message=f"{relative}: {old[:40]}"):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_fails(
                        "unacknowledged frozen contract change: "
                        + re.escape(relative)
                    )
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_transitive_top_level_initialization_frontier_is_frozen(self) -> None:
        mutations = (
            (
                "every non-function\n> top-level value in `V`'s eager reference set ([04-INF-7]) SHALL be available",
                "every non-function\n> top-level value in `V`'s eager reference set ([04-INF-7]) MAY be available",
            ),
            (
                "If `V` itself occurs in the set,\n> [04-INF-7]'s `CycleDetected` verdict takes precedence",
                "If `V` itself occurs in the set,\n> `UnboundVariable` MAY take precedence",
            ),
        )
        relative = "spec/04-type-system.md"
        for old, new in mutations:
            with self.subTest(message=old[:60]):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_fails(
                        "unacknowledged frozen contract change: "
                        + re.escape(relative)
                    )
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_spec06_additive_count_grad_contradiction_fails(self) -> None:
        self.append(
            "spec/06-transformations.md",
            "\nCount may return a silent zero cotangent when used under grad.\n",
        )
        self.assert_fails(
            "unacknowledged frozen contract change: spec/06-transformations.md"
        )

    def test_an_acknowledged_change_passes(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        report = self.check(acknowledgements=("spec/11-ffi.md",))
        self.assertIn("1 of 31 contract files changed", report[0])
        self.assertIn("  ok  Frozen-contract-change: spec/11-ffi.md", report)

    def test_a_body_line_acknowledges_the_change(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.check(body="Frozen-contract-change: spec/11-ffi.md\n")

    def test_acknowledging_one_file_does_not_cover_another(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.append("spec/10-serialization.md", "\nAn unreviewed sentence.\n")
        self.assert_fails(
            "unacknowledged frozen contract change: spec/10-serialization.md",
            body="Frozen-contract-change: spec/11-ffi.md\n",
        )

    def test_a_stale_acknowledgement_fails(self) -> None:
        self.assert_fails(
            "stale frozen contract acknowledgement for spec/11-ffi.md",
            body="Frozen-contract-change: spec/11-ffi.md\n",
        )

    def test_a_stale_acknowledgement_fails_beside_a_live_one(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.assert_fails(
            "stale frozen contract acknowledgement for spec/10-serialization.md",
            body=(
                "Frozen-contract-change: spec/11-ffi.md\n"
                "Frozen-contract-change: spec/10-serialization.md\n"
            ),
        )

    def test_a_wrong_case_path_is_not_the_contract_file(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.assert_fails(
            "names SPEC/11-ffi.md, which is not a frozen contract file",
            body="Frozen-contract-change: SPEC/11-ffi.md\n",
        )

    def test_a_glob_never_acknowledges_a_change(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.assert_fails(
            "malformed frozen contract acknowledgement path",
            body="Frozen-contract-change: spec/*.md\n",
        )

    def test_a_non_contract_file_cannot_be_acknowledged(self) -> None:
        self.assert_fails(
            "names README.md, which is not a frozen contract file",
            body="Frozen-contract-change: README.md\n",
        )

    def test_a_duplicate_acknowledgement_fails(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.assert_fails(
            "duplicate frozen contract acknowledgement for spec/11-ffi.md",
            body=(
                "Frozen-contract-change: spec/11-ffi.md\n"
                "Frozen-contract-change: spec/11-ffi.md\n"
            ),
        )

    def test_a_deleted_contract_file_is_a_change(self) -> None:
        (self.root / "spec/11-ffi.md").unlink()
        self.assert_fails(
            "unacknowledged frozen contract change: spec/11-ffi.md"
        )

    def test_a_whitespace_only_edit_is_a_change(self) -> None:
        self.append("spec/11-ffi.md", "\n")
        self.assert_fails(
            "unacknowledged frozen contract change: spec/11-ffi.md"
        )

    def test_a_change_reverted_in_the_working_tree_is_not_a_change(self) -> None:
        path = self.root / "spec/11-ffi.md"
        original = path.read_text(encoding="utf-8")
        path.write_text(original + "\nTemporary.\n", encoding="utf-8")
        path.write_text(original, encoding="utf-8")
        self.check()

    def test_a_committed_change_on_the_branch_is_still_a_change(self) -> None:
        # The comparison point is content, not the working tree's dirtiness:
        # committing the edit must not clear the acknowledgement requirement.
        _git_ok(self.root, "checkout", "-q", "-b", "topic")
        self.append("spec/11-ffi.md", "\nA committed sentence.\n")
        _commit_all(self.root, "edit the contract")
        self.assert_fails(
            "unacknowledged frozen contract change: spec/11-ffi.md"
        )

    def test_a_change_inherited_from_the_base_is_not_this_branch_s(self) -> None:
        # The merge base, not the base tip, is the comparison point. A contract
        # file that moved on `main` after this branch forked must not demand an
        # acknowledgement from a branch that never touched it.
        _git_ok(self.root, "checkout", "-q", "-b", "topic")
        self.append("spec/10-serialization.md", "\nThis branch's edit.\n")
        _commit_all(self.root, "branch edit")
        _git_ok(self.root, "checkout", "-q", "main")
        self.append("spec/11-ffi.md", "\nA later main-branch sentence.\n")
        moved = _commit_all(self.root, "main moves on")
        _git_ok(self.root, "update-ref", "refs/remotes/origin/main", moved)
        _git_ok(self.root, "checkout", "-q", "topic")
        report = self.check(
            acknowledgements=("spec/10-serialization.md",)
        )
        self.assertIn("1 of 31 contract files changed", report[0])

    def test_an_unreadable_baseline_blob_is_an_error_not_an_absence(self) -> None:
        # Round 1 F3. Reading a failed `git show` as "absent at the merge base"
        # would report a changed file as unchanged whenever the object store is
        # degraded. The tree listing and the blob read are now separate.
        merge_base = oracle.resolve_merge_base(self.root, "origin/main")
        present = oracle.contract_files_at(
            self.root, merge_base, oracle.CONTRACT_FILES
        )
        self.assertEqual(present, set(oracle.CONTRACT_FILES))

        real_git = oracle._git

        def failing_show(root: Path, *args: str):
            if args and args[0] == "show":
                return subprocess.CompletedProcess(
                    args, 128, b"", b"fatal: unable to read object"
                )
            return real_git(root, *args)

        with mock.patch.object(oracle, "_git", failing_show):
            with self.assertRaisesRegex(
                oracle.OracleError, r"cannot read .* at "
            ):
                oracle.changed_contract_files(
                    self.root, merge_base, oracle.CONTRACT_FILES
                )

    def test_an_unlistable_merge_base_tree_is_an_error(self) -> None:
        with self.assertRaisesRegex(
            oracle.OracleError, "cannot list frozen contract files"
        ):
            oracle.contract_files_at(
                self.root, "0" * 40, oracle.CONTRACT_FILES
            )

    def test_a_contract_file_absent_at_the_merge_base_is_a_change(self) -> None:
        # The other half of F3: a genuine absence must still read as a change,
        # not as an error.
        _git_ok(self.root, "checkout", "-q", "-b", "topic")
        new_path = self.root / "spec/11-ffi.md"
        merge_base = oracle.resolve_merge_base(self.root, "origin/main")
        present = oracle.contract_files_at(
            self.root, merge_base, ("spec/11-ffi.md", "docs/absent-probe.md")
        )
        self.assertEqual(present, {"spec/11-ffi.md"})
        self.assertTrue(new_path.exists())
        (self.root / "docs/absent-probe.md").write_text("new\n", encoding="utf-8")
        self.assertIn(
            "docs/absent-probe.md",
            oracle.changed_contract_files(
                self.root, merge_base, ("docs/absent-probe.md",)
            ),
        )

    def test_a_missing_merge_base_fails_loudly_in_strict_mode(self) -> None:
        _git_ok(self.root, "update-ref", "-d", "refs/remotes/origin/main")
        self.assert_fails("cannot determine the frozen contract merge base")

    def test_a_non_repository_root_fails_loudly_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as plain:
            self.assert_fails(
                "is not a git work tree", root=Path(plain)
            )

    def test_an_unreadable_tree_reports_and_exits_zero_in_advisory_mode(
        self,
    ) -> None:
        # Round 2 RT2-4. The unreadable-blob repair made `changed_contract_files`
        # raise, and advisory mode caught `OracleError` only around the merge
        # base, so a degraded object store aborted the run before the atom and
        # region digests. Advisory mode reports and continues.
        real_git = oracle._git

        def failing_listing(root: Path, *args: str):
            if args and args[0] == "ls-tree":
                return subprocess.CompletedProcess(
                    args, 128, b"", b"fatal: not a tree object"
                )
            return real_git(root, *args)

        with mock.patch.object(oracle, "_git", failing_listing):
            report = self.check(require_acknowledgement=False)
            self.assertTrue(
                any("cannot list frozen contract files" in line for line in report),
                report,
            )
            self.assertTrue(
                any("change detection skipped" in line for line in report), report
            )
            with self.assertRaisesRegex(
                oracle.OracleError, "cannot list frozen contract files"
            ):
                self.check(require_acknowledgement=True)

    def test_a_missing_merge_base_reports_and_exits_zero_in_advisory_mode(
        self,
    ) -> None:
        _git_ok(self.root, "update-ref", "-d", "refs/remotes/origin/main")
        report = self.check(require_acknowledgement=False)
        self.assertTrue(
            any("cannot determine" in line for line in report), report
        )
        self.assertTrue(
            any("change detection skipped" in line for line in report), report
        )

    def test_advisory_mode_reports_but_does_not_raise(self) -> None:
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        report = self.check(require_acknowledgement=False)
        self.assertTrue(
            any(
                "ISSUE unacknowledged frozen contract change: spec/11-ffi.md"
                in line
                for line in report
            ),
            report,
        )

    def test_an_explicit_base_ref_is_honoured(self) -> None:
        _git_ok(self.root, "checkout", "-q", "-b", "topic")
        self.append("spec/11-ffi.md", "\nA reviewed sentence.\n")
        self.check(base=self.base, acknowledgements=("spec/11-ffi.md",))


class FrozenRegionIndependenceTests(unittest.TestCase):
    """A required clause fails independently of Git and acknowledgement.

    Changed prose requires review acknowledgement; removed required literals
    additionally fail the semantic oracle even without comparison history.
    """

    def test_a_region_edit_fails_without_any_git_history(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for relative in CONTRACT_FILES:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO_ROOT / relative, destination)
            path = root / "spec/04-type-system.md"
            original = path.read_text(encoding="utf-8")
            old = (
                "`and` / `or` / `not` are the\n  logical operations and "
                "`count` is the bool-tensor counting operation"
            )
            self.assertIn(old, original)
            path.write_text(
                original.replace(
                    old,
                    "`and` / `or` / `not` are the logical operations; counting "
                    "is the explicit-cast idiom",
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                oracle.OracleError, "bool counting operation"
            ):
                oracle.validate_contract(root)


class MergeConflictFreedomTests(unittest.TestCase):
    """The property the acknowledgement gate exists to buy.

    Two pull requests that edit different frozen contract files, or disjoint
    sections of one, must not conflict in any tracked file other than the spec
    files they each edited. The control leg reproduces the whole-file digest
    table and shows the same pair of edits conflicting, so a clean merge in the
    main leg is evidence rather than an artifact of the harness.
    """

    def build(self, root: Path, with_digest_table: bool) -> None:
        (root / "spec").mkdir(parents=True, exist_ok=True)
        (root / "spec/a.md").write_text(
            "# A\n\n" + "".join(f"clause a{i}\n" for i in range(40)),
            encoding="utf-8",
        )
        (root / "spec/b.md").write_text(
            "# B\n\n" + "".join(f"clause b{i}\n" for i in range(40)),
            encoding="utf-8",
        )
        if with_digest_table:
            (root / "oracle.py").write_text(
                "FROZEN_FILE_DIGESTS = {\n"
                '    "spec/a.md": "' + "0" * 64 + '",\n'
                '    "spec/b.md": "' + "1" * 64 + '",\n'
                "}\n",
                encoding="utf-8",
            )

    def edit(self, root: Path, relative: str, line: str, replacement: str) -> None:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        self.assertIn(line, text)
        path.write_text(text.replace(line, replacement, 1), encoding="utf-8")

    def move_digest(self, root: Path, relative: str, digit: str) -> None:
        path = root / "oracle.py"
        text = path.read_text(encoding="utf-8")
        line = [entry for entry in text.splitlines() if relative in entry][0]
        path.write_text(
            text.replace(line, f'    "{relative}": "{digit * 64}",'),
            encoding="utf-8",
        )

    def merge_is_clean(self, root: Path) -> bool:
        merged = _git(root, "merge-tree", "--write-tree", "pr-one", "pr-two")
        self.assertIn(
            merged.returncode,
            (0, 1),
            f"git merge-tree errored: {merged.stderr}",
        )
        return merged.returncode == 0

    def scenario(
        self, with_digest_table: bool, same_file: bool
    ) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _git_ok(root, "init", "-q", "-b", "main")
            self.build(root, with_digest_table)
            _commit_all(root, "baseline")

            _git_ok(root, "checkout", "-q", "-b", "pr-one")
            self.edit(root, "spec/a.md", "clause a3\n", "clause a3 revised\n")
            if with_digest_table:
                self.move_digest(root, "spec/a.md", "a")
            _commit_all(root, "pr one")

            _git_ok(root, "checkout", "-q", "main")
            _git_ok(root, "checkout", "-q", "-b", "pr-two")
            if same_file:
                self.edit(
                    root, "spec/a.md", "clause a37\n", "clause a37 revised\n"
                )
                if with_digest_table:
                    self.move_digest(root, "spec/a.md", "b")
            else:
                self.edit(
                    root, "spec/b.md", "clause b3\n", "clause b3 revised\n"
                )
                if with_digest_table:
                    self.move_digest(root, "spec/b.md", "b")
            _commit_all(root, "pr two")

            clean = self.merge_is_clean(root)
            conflicts = _git(
                root, "merge-tree", "--write-tree", "pr-one", "pr-two"
            ).stdout
            return clean, conflicts

    def test_disjoint_contract_edits_merge_cleanly_without_the_digest_table(
        self,
    ) -> None:
        for same_file in (False, True):
            with self.subTest(same_file=same_file):
                clean, _ = self.scenario(
                    with_digest_table=False, same_file=same_file
                )
                self.assertTrue(
                    clean,
                    "edits to disjoint contract text must not conflict",
                )

    def test_the_digest_table_is_what_made_those_edits_conflict(self) -> None:
        # Control. Without this leg a green test above would prove only that
        # the harness cannot detect a conflict.
        for same_file in (False, True):
            with self.subTest(same_file=same_file):
                clean, conflicts = self.scenario(
                    with_digest_table=True, same_file=same_file
                )
                self.assertFalse(
                    clean,
                    "the whole-file digest table must reproduce the conflict "
                    "this change removes",
                )
                self.assertIn("oracle.py", conflicts)
                self.assertNotIn("spec/b.md", conflicts)


class OracleEntryPointTests(unittest.TestCase):
    """`main` runs the acknowledgement leg before it can print the pass line."""

    def test_strict_mode_fails_before_the_success_line(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            _git_ok(root, "init", "-q", "-b", "main")
            for relative in CONTRACT_FILES:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO_ROOT / relative, destination)
            base = _commit_all(root, "baseline")
            _git_ok(root, "update-ref", "refs/remotes/origin/main", base)
            path = root / "spec/11-ffi.md"
            path.write_text(
                path.read_text(encoding="utf-8") + "\nUnreviewed.\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                with self.assertRaisesRegex(
                    SystemExit, "unacknowledged frozen contract change"
                ):
                    oracle.main(["--require-acknowledgement"], root=root)
            self.assertNotIn(oracle.PASS_LINE, output.getvalue())

    def test_at_most_one_acknowledgement_source(self) -> None:
        parser = oracle.build_parser()
        args = parser.parse_args(
            ["--acknowledgements-file", "x", "--acknowledgements-env", "Y"]
        )
        with self.assertRaisesRegex(SystemExit, "at most one"):
            oracle.acknowledgement_body(args)

    def test_an_unset_acknowledgement_environment_variable_fails(self) -> None:
        parser = oracle.build_parser()
        args = parser.parse_args(
            ["--acknowledgements-env", "CHELIS_ACK_ABSENT_FOR_TEST"]
        )
        with mock.patch.dict(oracle.os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "is not set"):
                oracle.acknowledgement_body(args)

    def test_the_acknowledgement_environment_variable_is_read_verbatim(
        self,
    ) -> None:
        parser = oracle.build_parser()
        args = parser.parse_args(["--acknowledgements-env", "CHELIS_ACK_BODY"])
        body = "Frozen-contract-change: spec/11-ffi.md\n"
        with mock.patch.dict(oracle.os.environ, {"CHELIS_ACK_BODY": body}):
            self.assertEqual(oracle.acknowledgement_body(args), body)

    def test_an_acknowledgements_file_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "body.md"
            path.write_text(
                "Frozen-contract-change: spec/11-ffi.md\n", encoding="utf-8"
            )
            args = oracle.build_parser().parse_args(
                ["--acknowledgements-file", str(path)]
            )
            self.assertEqual(
                oracle.acknowledgement_body(args),
                "Frozen-contract-change: spec/11-ffi.md\n",
            )

    def test_a_missing_acknowledgements_file_fails(self) -> None:
        args = oracle.build_parser().parse_args(
            ["--acknowledgements-file", "/nonexistent/body.md"]
        )
        with self.assertRaisesRegex(SystemExit, "cannot read acknowledgements"):
            oracle.acknowledgement_body(args)

    def test_the_default_mode_does_not_require_acknowledgement(self) -> None:
        args = oracle.build_parser().parse_args([])
        self.assertFalse(args.require_acknowledgement)
        self.assertEqual(args.base, oracle.DEFAULT_BASE_REF)
        self.assertEqual(args.acknowledge, [])


if __name__ == "__main__":
    unittest.main()
