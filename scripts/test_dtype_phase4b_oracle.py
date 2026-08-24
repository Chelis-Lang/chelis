#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
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

    def repository_atom(self, atom: str) -> str:
        text = (REPO_ROOT / "spec/05-risc-primitives.md").read_text(
            encoding="utf-8"
        )
        return oracle.normalize_atom_body(oracle.atom_blocks(text)[atom])

    def test_repository_contract_passes(self) -> None:
        oracle.validate_contract(REPO_ROOT)

    def test_additive_prose_in_an_integrity_fingerprinted_file_fails(self) -> None:
        for relative_name in oracle.FROZEN_FILE_DIGESTS:
            relative = Path(relative_name)
            with self.subTest(relative=relative):
                path = self.root / relative
                original = path.read_text(encoding="utf-8")
                path.write_text(
                    "An implementation MAY ignore the frozen Phase 4B contract.\n\n"
                    + original,
                    encoding="utf-8",
                )
                try:
                    self.assert_contract_fails("frozen contract file")
                finally:
                    path.write_text(original, encoding="utf-8")

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
            "lowers to `sum(cast(x, int64), axis)`",
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

    def test_exact_read_atom_freezes_json_and_csv_numeric_boundaries(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "`csv_int` | `(List[Dict[string,string]], int64, string) -> int64`",
                "`csv_int` | `(List[Dict[string,f64]], int32, string) -> int32`",
                "OP-3.*csv_int",
            ),
            (
                "It never truncates or rounds a float\n> into an integer",
                "It truncates float variants into int64",
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
            "`JsonFloat(value)` accepts exactly f64 and\n> `JsonInt(value)` accepts exactly int64",
            "`JsonFloat(value)` accepts any float and widens it to f64",
        )
        self.assert_contract_fails("OP-4.*exactly f64")

    def test_numeric_serialization_preserves_exact_json_variants(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "emits a stored `JsonInt` int64\n> as its exact decimal digits",
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
                "| `f16` | type error | — |",
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
                "| integer, bool, tensor | type error | — |",
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
            "An integer-form token outside int64 range is a loud `Overflow`\n"
            "> error; punctuation never selects a lossy float fallback for an integer.",
            "An integer-form token outside int64 range falls back to `JNum`.",
        )
        self.assert_contract_fails("OP-2.*Overflow")

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
            "`tensor_scan` | `(T,((T,int64)->T!E),int64)->tensor[n,T]!E`",
            "`process_run` | `(string,List[string])->(int64,string,string)!{IO}`",
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
                "writes exactly [05-OP-25]'s complete recursive rendering\n"
                "> of its argument followed by one byte `\\n` to standard output",
                "writes an implementation-defined debug rendering",
                "OP-32.*complete recursive rendering",
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
                "Scatter indices are limited to int32 or int64",
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
                "> scatter mode and no int32/int64-only dispatch exception",
                "An int32/int64-only compatibility dispatch remains available",
                "OP-33.*no int32/int64-only",
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
            "`gather` admits only int32 and int64 index tensors",
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
                "Each exact [05-OP-36] identity —\n"
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

    def test_scalar_carrier_pins_exact_public_layouts(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-31"]
        for declaration in (
            "typedef uint8_t chelis_dtype;",
            "typedef struct { chelis_dtype dtype; uint8_t reserved[7]; uint64_t bits; } chelis_scalar;",
            "typedef struct { uint8_t is_some; uint8_t reserved[7]; chelis_scalar value; } chelis_option_scalar;",
            "typedef union { chelis_scalar scalar; void *handle; } chelis_value_payload;",
            "typedef struct { chelis_value_tag tag; uint8_t reserved[7]; chelis_value_payload payload; } chelis_value;",
            "typedef struct { void *data; const int64_t *shape; const int64_t *strides; int64_t size; int64_t byte_capacity; int32_t rank; chelis_dtype dtype; uint8_t owns_data; uint8_t reserved[2]; } chelis_tensor;",
            "typedef struct { chelis_value key; chelis_value value; } chelis_dict_entry;",
        ):
            with self.subTest(declaration=declaration):
                self.assertIn(declaration, block)

    def test_scalar_carrier_layout_mutation_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "const int64_t *shape; const int64_t *strides; int64_t size; "
            "int64_t byte_capacity; int32_t rank",
            "int64_t shape[8]; int64_t strides[8]; int64_t size; "
            "int64_t byte_capacity; int32_t rank",
        )
        self.assert_contract_fails("OP-31.*chelis_tensor")

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
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-32"]
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
            Path("spec/05-risc-primitives.md"),
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
            Path("spec/05-risc-primitives.md"),
            "`process::run` | `(string,List[string])->(int64,string,string)!{IO}`",
            "`process::run` | `(string,List[string])->(int64,string,string)`",
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
            "requires finite endpoints and\n> int64 `count >= 1`",
            "accepts nonpositive counts as a one-element result",
        )
        self.assert_contract_fails("OP-35.*count >= 1")

    def test_stdlib_manifest_has_exactly_eighty_three_unique_rows(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-35"]
        rows = re.findall(r"^> \| (\d+) \| `([^`]+)` \|", block, re.MULTILINE)
        self.assertEqual(len(rows), 83)
        self.assertEqual(len({number for number, _identity in rows}), 83)
        self.assertEqual(len({identity for _number, identity in rows}), 83)
        identities = {identity for _number, identity in rows}
        for identity in (
            "decimal::decimal_add",
            "io/json::load_json",
            "time::date_lt",
            "tokenizer::load_tokenizer",
        ):
            with self.subTest(identity=identity):
                self.assertIn(identity, identities)

    def test_stdlib_manifest_duplicate_identity_fails(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "| 21 | `index::take_list` |",
            "| 21 | `index::list_index` |",
        )
        self.assert_contract_fails("stdlib numeric manifest")

    def test_stdlib_manifest_uses_rank_polymorphic_sort_and_generic_scalar_equality(self) -> None:
        block = self.repository_atom("05-OP-35")
        self.assertIn(
            "`sort::sort` | `(&tensor[..r,p_numeric],int32)->"
            "(tensor[..r,p_numeric],tensor[..r,int64])`",
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
            "Integer-form tokens outside int64 fall back to `JsonFloat`",
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
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| 52 | `tensor/construct::arange` | "
                "`(p_int,p_int)->tensor[n,p_int]` |",
                "| 52 | `tensor/construct::arange` | "
                "`(int32,int32)->tensor[n,int32]` |",
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
                "returns an int64 sequence for every endpoint dtype",
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
            with self.subTest(message=message):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(message)
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_exact_op_manifest_row_deletion_fails(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        for atom, rows in oracle.EXPECTED_OP_MANIFESTS.items():
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
            "05-OP-35": ("(p_float)->p_float", "(f32)->f32"),
            "05-OP-38": (
                "(T,((T,int64)->T!E),int64)->tensor[n,T]!E",
                "(f32,((f32,int64)->f32),int64)->tensor[n,f32]",
            ),
        }
        path = self.root / "spec/05-risc-primitives.md"
        for atom, (old, new) in mutations.items():
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(old, original)
                path.write_text(original.replace(old, new, 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"{atom}.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_exact_op_manifest_duplicate_label_or_identity_fails(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        for atom, rows in oracle.EXPECTED_OP_MANIFESTS.items():
            with self.subTest(atom=atom):
                original = path.read_text(encoding="utf-8")
                self.assertIn(rows[1], original)
                path.write_text(original.replace(rows[1], rows[0], 1), encoding="utf-8")
                try:
                    self.assert_contract_fails(f"{atom}.*exact manifest")
                finally:
                    path.write_text(original, encoding="utf-8")

    def test_assert_close_tensor_is_float_only_and_own_width(self) -> None:
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-35"]
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
                "`p_int` over int64 only",
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
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "| 48 | `scalar::abs` | `(p_numeric)->p_numeric` |",
                "| 48 | `scalar::abs` | `(f32)->f32` |",
            ),
            (
                "| 49 | `scalar::max` | `(p_numeric,p_numeric)->p_numeric` |",
                "| 49 | `scalar::max` | `(f32,f32)->f32` |",
            ),
            (
                "| 50 | `scalar::min` | `(p_numeric,p_numeric)->p_numeric` |",
                "| 50 | `scalar::min` | `(f32,f32)->f32` |",
            ),
            (
                "| 58 | `test::assert_close` | "
                "`(p_float,p_float,p_float,string)->unit!{Test}` |",
                "| 58 | `test::assert_close` | "
                "`(f32,f32,f32,string)->unit!{Test}` |",
            ),
            (
                "| 59 | `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |",
                "| 59 | `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],f32,string)->unit!{Test}` |",
            ),
            (
                "| 60 | `test::assert_eq` | "
                "`(Q,Q,string)->unit!{Test}` |",
                "| 60 | `test::assert_eq` | `(f32,f32,string)->unit!{Test}` |",
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
        self.assertIn("`int32` | `int32`, `int64` | accumulator dtype `a`", text)

    def test_sum_result_rule_cannot_drop_explicit_wider_accumulators(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "`sum_result(p, a) = p` exactly when `p` is `bf16` or `f16`;\n"
            "otherwise `sum_result(p, a) = a`",
            "`sum_result(p, a)` is implementation-defined for explicit accumulators",
        )
        self.assert_contract_fails("total sum result precision rule")

    def test_to_string_domain_admits_recursive_values_but_rejects_opaque_values(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "Functions and resource\n> handles are type errors",
            "Functions and resource handles are rendered opaquely",
        )
        self.assert_contract_fails("OP-25.*Functions")

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
            "exact case enumerator covers unit, scalar, tensor, List, tuple, Dict, "
            "Option, and ADT values",
            "case enumerator covers only tensor and List values",
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
        self.assert_contract_fails("frozen normative atom 05-OP-12")

    def test_plain_prose_after_extrema_atom_cannot_contradict_it(self) -> None:
        self.replace(
            Path("spec/05-risc-primitives.md"),
            "> adjoint reverses that composition. Integer operands are forward-only and\n"
            "> `grad` rejects them.\n>\n"
            "> **[05-OP-13]**",
            "> adjoint reverses that composition. Integer operands are forward-only and\n"
            "> `grad` rejects them.\n\n"
            "An implementation MAY instead route the full non-NaN `max_reduce` "
            "cotangent to only the last element equal to the selected maximum.\n\n"
            "> **[05-OP-13]**",
        )
        self.assert_contract_fails("frozen numeric primitive contracts")

    def test_plain_prose_before_multi_axis_contract_cannot_contradict_it(self) -> None:
        self.replace(
            Path("spec/04-type-system.md"),
            "Multiple axes may be reduced in one call",
            "For multiple named-axis reductions, source spelling order controls "
            "evaluation order and therefore owns exact values, traps, NaN selection, "
            "and adjoints.\n\n"
            "Multiple axes may be reduced in one call",
        )
        self.assert_contract_fails("frozen name-preserving rank polymorphism")

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
                "checked for int64 representation before normalization",
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
            "int64 extents and compares its length and every entry to the tensor's "
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
        block = oracle.atom_blocks(
            (REPO_ROOT / "spec/05-risc-primitives.md").read_text(encoding="utf-8")
        )["05-OP-35"]
        for signature in (
            "`contracts::normal_cdf` | `(p_float)->p_float`",
            "`init/random::normal_like` | `(&tensor[..r,p_float],p_float,p_float)->tensor[..r,p_float]!{Random}`",
            "`tensor/construct::linspace` | `(p_float,p_float,int64)->tensor[n,p_float]`",
            "`tensor/construct::stack` | `(List[tensor[..pre,..post,p]],int32)->tensor[..pre,rows,..post,p]`",
            "`test::assert_close_tensor` | `(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}`",
            "`test::assert_eq_tensor` | `(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}`",
            "`test::assert_shape` | `(&tensor[..r,p],List[int64],string)->unit!{Test}`",
        ):
            with self.subTest(signature=signature):
                self.assertIn(signature, block)

    def test_stdlib_tensor_manifest_cannot_restore_fixed_current_ranks(self) -> None:
        path = self.root / "spec/05-risc-primitives.md"
        mutations = (
            (
                "(&tensor[..r,p_float],p_float)->tensor[..r,p_float]!{Random}",
                "(&tensor[n,p_float],p_float)->tensor[n,p_float]!{Random}",
            ),
            (
                "(&tensor[..pre,1,..post,p],int32)->tensor[..pre,..post,p]",
                "(&tensor[a,1,b,p],int32)->tensor[a,b,p]",
            ),
            (
                "(List[tensor[..pre,..post,p]],int32)->tensor[..pre,rows,..post,p]",
                "(List[tensor[d,p]],int32)->tensor[rows,d,p]",
            ),
            (
                "(&tensor[..pre,..post,p],int32)->tensor[..pre,1,..post,p]",
                "(&tensor[d,p],int32)->tensor[1,d,p]",
            ),
            (
                "(&tensor[..r,bool])->tensor[hits,int64]",
                "(&tensor[n,bool])->tensor[hits,int64]",
            ),
            (
                "| 59 | `test::assert_close_tensor` | "
                "`(&tensor[..r,p_float],&tensor[..r,p_float],p_float,string)->unit!{Test}` |",
                "| 59 | `test::assert_close_tensor` | "
                "`(&tensor[n,p_float],&tensor[n,p_float],p_float,string)->unit!{Test}` |",
            ),
            (
                "| 61 | `test::assert_eq_tensor` | "
                "`(&tensor[..r,p],&tensor[..r,p],string)->unit!{Test}` |",
                "| 61 | `test::assert_eq_tensor` | "
                "`(&tensor[n,p],&tensor[n,p],string)->unit!{Test}` |",
            ),
            (
                "(&tensor[..r,p],List[int64],string)->unit!{Test}",
                "(&tensor[n,p],int64,string)->unit!{Test}",
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
            "final normalized `days` field has no int64 representation",
            "any intermediate component total exceeds int64",
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
            "> mathematical magnitude rather than an int64 `abs`",
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
        self.assert_contract_fails("frozen contract file")

    def test_count_axes_remain_signature_checks_not_table_b_narrowing(self) -> None:
        self.replace(
            Path("spec/design/capability_table.md"),
            "axis values are not table axes",
            "each axis value becomes a target-specific Table B row",
        )
        self.assert_contract_fails("count axes stay in the signature rule")

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
                "No narrow case allowlist or unsupported-nested-carrier exception "
                "survives",
                "nested carriers may remain permanently unsupported",
                "recursive to_string full domain",
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
                "No v5\nnumeric-fill migration or inferred fill dtype exists",
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

    def test_agent_contract_cannot_restore_capacity_exceptions(self) -> None:
        self.replace(
            Path("AGENTS.md"),
            "No grandfather, permanent-disposition,",
            "legacy capacity rows retain their grandfather dispositions",
        )
        self.assert_contract_fails("agent zero-exception policy")

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
                "[#1284], and [#1287]-[#1298]",
                "[#1284], and [#1287]-[#1296]",
                "includes every late prerequisite",
            ),
            (
                "chelis#1295's all-active-float random/rounding rules, chelis#1297's "
                "compiled\n"
                "host effects, and chelis#1298's runtime-axis/window operations have "
                "landed",
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
        self.assert_contract_fails("frozen roadmap ownership")

    def test_status_must_keep_external_execution_authority(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "total external target-disposition registry",
            "semantic registry alone",
        )
        self.assert_contract_fails("status external target authority")

    def test_status_issue_graph_counts_are_cross_checked(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "| #730 | 17 / 37 |",
            "| #730 | 16 / 37 |",
        )
        self.assert_contract_fails("sum to 53 open issues")

    def test_status_keeps_stdlib_alignment_owner(self) -> None:
        self.replace(
            Path("docs/investigations/remediation_status_2026_08_04.md"),
            "#1293 aligns all 83 recursively discovered stdlib numeric definitions",
            "the parent tracker implicitly owns stdlib alignment",
        )
        self.assert_contract_fails("status stdlib alignment owner")


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
