"""Unit and mutation tests for the chelis#1297 compiled host-effect oracle."""
from __future__ import annotations

from pathlib import Path
import shutil
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

    def test_dtype_erasing_tensor_scan_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-ir/src/host.rs",
            'name: "to_tensor".to_string(),',
            'name: "to_tensor_f64".to_string(),',
        )

    def test_reordered_tensor_scan_bounds_check_fails(self):
        self.assert_mutation_fails(
            "crates/chelis-ir/src/host.rs",
            "tensor_scan requires a non-negative length, got ",
            "tensor_scan length ",
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
