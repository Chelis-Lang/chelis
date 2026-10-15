"""Unit and mutation tests for the chelis#1297 compiled host-effect oracle."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

if __package__:
    from . import dtype_compiled_host_effect_oracle as oracle
else:
    import dtype_compiled_host_effect_oracle as oracle


def contract_paths() -> set[str]:
    paths = {contract.path for contract in oracle.source_contracts()}
    paths.add("crates/chelis-types/src/builtins.rs")
    return paths


class MutatedTree:
    """A copy of every file the source contracts read, for one mutation."""

    def __enter__(self) -> Path:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        for relative in contract_paths():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(oracle.REPO_ROOT / relative, target)
        return root

    def __exit__(self, *exc_info: object) -> None:
        self.directory.cleanup()


def mutate(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    source = path.read_text()
    if old not in source:
        raise AssertionError(f"mutation anchor {old!r} missing from {relative}")
    path.write_text(source.replace(old, new, 1))


class OracleContractTests(unittest.TestCase):
    def test_current_tree_satisfies_every_contract(self):
        oracle.validate_source_contracts()

    def test_direct_and_package_entrypoints_execute_original_contracts(self):
        identities = (
            "OracleContractTests.test_current_tree_satisfies_every_contract",
            "OracleMutationTests.test_dtype_erasing_tensor_scan_fails",
        )
        entrypoints = (
            [sys.executable, "-B", "scripts/test_dtype_compiled_host_effect_oracle.py", *identities],
            [sys.executable, "-B", "-m", "unittest", *(
                f"scripts.test_dtype_compiled_host_effect_oracle.{identity}"
                for identity in identities
            )],
        )
        for argv in entrypoints:
            with self.subTest(argv=argv):
                result = subprocess.run(
                    argv, cwd=oracle.REPO_ROOT, capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("Ran 2 tests", result.stderr)
                self.assertIn("OK", result.stderr)

    def test_every_operation_family_names_both_lanes(self):
        names = {contract.name for contract in oracle.source_contracts()}
        for family in ["clock reads", "process_run", "round_to", "CSV", "assertions"]:
            for lane in ["one runtime definition", "evaluator calls the runtime", "compiled C calls the runtime"]:
                self.assertIn(f"{family}: {lane}", names)

    def test_acceptance_legs_run_parity_and_self_tests(self):
        argv = [" ".join(leg.argv) for leg in oracle.oracle_legs("python")]
        self.assertTrue(any("test_dtype_compiled_host_effect_oracle.py" in leg for leg in argv))
        self.assertTrue(any("--test compiled_host_effects" in leg for leg in argv))
        self.assertTrue(any("--test process_run_builtin" in leg for leg in argv))


class OracleMutationTests(unittest.TestCase):
    def assert_mutation_fails(self, relative: str, old: str, new: str) -> None:
        with MutatedTree() as root:
            mutate(root, relative, old, new)
            with self.assertRaises(oracle.OracleFailure):
                oracle.validate_source_contracts(root)

    def test_restored_evaluator_only_routing_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-ir/src/host.rs",
            "pub fn find_direct_builtin_call(",
            'pub const EVAL_ONLY_HOST_BUILTINS: &[&str] = &["process_run"];\n'
            "pub fn find_direct_builtin_call(",
        )

    def test_restored_whole_module_rejection_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-compiler-api/src/compiler.rs",
            "/// Closed target vocabulary for shared pre-codegen build gates.",
            'const HOST_ONLY_BUILTINS: &[&str] = &["tensor_scan", "test_assert"];\n'
            "/// Closed target vocabulary for shared pre-codegen build gates.",
        )

    def test_inert_assertion_fallback_fails(self):
        self.assert_mutation_fails(HOST_EMIT, '"test_assert" => {', '"test_assert_inert" => {')

    def test_default_clock_reading_fails(self):
        self.assert_mutation_fails(
            HOST_EMIT, "chelis_clock_wall_read()", "chelis_tuple_from_values(NULL, 0)"
        )

    def test_evaluator_private_copy_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-compiler-api/src/runtime/eval.rs",
            "use chelis_runtime::host_round::{FloatLayout, round_to_bits};",
            "use super::numeric_text::{FloatLayout, round_to_bits};",
        )

    def test_lossy_process_decoding_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-runtime/src/host_process.rs",
            "String::from_utf8(stdout)",
            "Ok::<String, ()>(String::from_utf8_lossy(&stdout).into_owned())",
        )

    def assert_scan_owner_mutation_fails(self, owner: str, old: str, new: str) -> None:
        with MutatedTree() as root:
            path = root / "crates/chelis-ir/src/host.rs"
            source = path.read_text()
            start, end = oracle.rust_owner_span(source, owner)
            body = source[start:end]
            self.assertEqual(body.count(old), 1, "one actual owner expression")
            path.write_text(source[:start] + body.replace(old, new, 1) + source[end:])
            with self.assertRaisesRegex(oracle.OracleFailure, "tensor_scan.*missing"):
                oracle.validate_source_contracts(root)

    def test_dtype_erasing_tensor_scan_fails(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_OWNER,
            'name: "to_tensor".into(),',
            'name: "to_tensor_f64".into(),',
        )

    def test_unrelated_generic_conversion_cannot_mask_scan_dtype_erasure(self):
        with MutatedTree() as root:
            path = root / "crates/chelis-ir/src/host.rs"
            source = path.read_text()
            start, end = oracle.rust_owner_span(source, oracle.TENSOR_SCAN_OWNER)
            self.assertIn('name: "to_tensor".into(),', source[:start])
            body = source[start:end].replace(
                'name: "to_tensor".into(),', 'name: "to_tensor_f64".into(),'
            )
            path.write_text(source[:start] + body + source[end:])
            with self.assertRaisesRegex(oracle.OracleFailure, 'tensor_scan.*to_tensor'):
                oracle.validate_source_contracts(root)

    def test_tensor_scan_stack_identity_must_belong_to_scan(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_OWNER,
            "name: TENSOR_SCAN_STACK.into(),",
            'name: "to_tensor".into(),',
        )

    def test_tensor_scan_state_identity_must_belong_to_checked_callback(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_CALLBACK_OWNER,
            "name: TENSOR_SCAN_STATE.into(),",
            'name: "copy".into(),',
        )

    def test_tensor_scan_state_dtype_is_derived_from_initial_state(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_OWNER,
            "let state_ty = host_expr_type(&init_expr);",
            "let state_ty = HostTypeTerm::Bool;",
        )

    def test_tensor_scan_callback_keeps_template_state_type(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_CALLBACK_OWNER,
            "HostExprKind::Var(template.to_string(), state_ty.clone())",
            "HostExprKind::Var(template.to_string(), HostTypeTerm::Bool)",
        )

    def test_required_scan_expressions_in_comments_cannot_supply_evidence(self):
        self.assert_scan_owner_mutation_fails(
            oracle.TENSOR_SCAN_OWNER,
            'name: "to_tensor".into(),',
            '// name: "to_tensor".into(),\n                name: "to_tensor_f64".into(),',
        )

    def test_scan_owner_absence_or_ambiguity_fails_closed(self):
        for owner in (oracle.TENSOR_SCAN_OWNER, oracle.TENSOR_SCAN_CALLBACK_OWNER):
            for replacement in ("missing_owner", owner + " {}\n" + owner):
                with self.subTest(owner=owner, replacement=replacement), MutatedTree() as root:
                    mutate(root, "crates/chelis-ir/src/host.rs", owner, replacement)
                    with self.assertRaisesRegex(oracle.OracleFailure, "owner.*exactly one"):
                        oracle.validate_source_contracts(root)

    def test_owner_without_body_cannot_capture_the_next_function(self):
        for owner in (oracle.TENSOR_SCAN_OWNER, oracle.TENSOR_SCAN_CALLBACK_OWNER):
            with self.subTest(owner=owner):
                source = owner + ";\nfn unrelated() { let value = 1; }"
                with self.assertRaisesRegex(oracle.OracleFailure, "owner.*missing body"):
                    oracle.rust_owner_span(source, owner)

    def test_comment_and_literal_owner_decoys_are_not_owners(self):
        with MutatedTree() as root:
            path = root / "crates/chelis-ir/src/host.rs"
            source = path.read_text()
            for owner in (oracle.TENSOR_SCAN_OWNER, oracle.TENSOR_SCAN_CALLBACK_OWNER):
                source = f'// {owner} {{}}\nconst DECOY: &str = r#"{owner} {{}}"#;\n' + source
            path.write_text(source)
            oracle.validate_source_contracts(root)

    def test_reordered_tensor_scan_bounds_check_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-ir/src/host.rs",
            'chelis_abi::failure::negative_length_prefix("tensor_scan")',
            '"tensor_scan length "',
        )

    def test_dtype_named_assertion_alias_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-types/src/builtins.rs",
            '    "process_run",\n',
            '    "process_run",\n    "test_assert_eq_f64",\n',
        )

    def test_restored_trap_abort_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-runtime/include/chelis_runtime.h", "    exit(1);\n}", "    abort();\n}"
        )

    def test_restored_compiled_target_rejection_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-compiler-api/src/compiler.rs",
            "/// Closed target vocabulary for shared pre-codegen build gates.",
            "// \"compiled targets (the host interpreter's eval/test lanes only)\"\n"
            "/// Closed target vocabulary for shared pre-codegen build gates.",
        )


HOST_EMIT = oracle.HOST_EMIT


if __name__ == "__main__":
    unittest.main()
