#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import loud_unsupported_phase2_oracle as oracle


def write_endpoint_fixture(root: Path, host_source: str) -> None:
    files = {
        "crates/chelis-ir/src/host.rs": host_source,
        "crates/chelis-compiler-api/src/compiler.rs": "fallible only\n",
        "crates/chelis-backend-c/src/emitted_expr.rs": "closed nodes\n",
        "crates/chelis-backend-c/src/host_emit.rs": (
            'require_same_abi_type(ty, &HostType::Unit, "unit expression")?;\n'
        ),
        "crates/chelis-backend-c/src/host_abi.rs": (
            "pub(crate) enum HostAbiType {}\n"
        ),
    }
    for relative, contents in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def valid_host_endpoint_source() -> str:
    return (
        "pub fn try_lower_compiled_program() {}\n"
        "fn lower_host_expr(\n"
        ") -> Result<HostExpr, crate::lower::LowerDiagnostic> { todo!() }\n"
        "fn lower_host_expr_kind(\n"
        ") -> Result<HostExpr, crate::lower::LowerDiagnostic> { todo!() }\n"
        "struct GenericAdtConstructor;\n"
        "struct GenericAdtField;\n"
        "struct InstantiatedAdtConstructor;\n"
        "enum AdtInstantiationError {}\n"
        "fn instantiate(\n"
        ") -> Result<InstantiatedAdtConstructor, AdtInstantiationError> { todo!() }\n"
    )


class LoudUnsupportedPhase2OracleTests(unittest.TestCase):
    def test_focused_commands_cover_every_phase2_layer(self) -> None:
        rendered = [oracle.command_text(command) for command in oracle.FOCUSED_COMMANDS]
        joined = "\n".join(rendered)
        for required in (
            "-p chelis-vocab",
            "runtime_dtype_generated_header",
            "runtime_dtype_invalid_ffi",
            "host_type_failure_states",
            "host_abi_tests",
            "emitted_expr::tests",
            "cargo test -p chelis-backend-c --doc",
            "pr799_host_resolution_result",
            "closed_vocabulary_architecture",
            "int_width_lane_matrix",
            "issue_734_tostring_placeholder",
        ):
            self.assertIn(required, joined)

    def test_effect_mutation_updates_owner_without_a_default_arm(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_effect_kind(source)
        self.assertNotEqual(mutated, source)
        self.assertIn("Phase2OracleMutation", mutated)
        self.assertIn('Self::Phase2OracleMutation => "phase2-oracle-mutation"', mutated)
        self.assertNotIn("_ =>", mutated.split("impl EffectKind", 1)[1].split("}", 1)[0])

    def test_dtype_mutation_updates_owner_without_a_default_arm(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_runtime_dtype(source)
        self.assertNotEqual(mutated, source)
        self.assertIn("Phase2OracleDType = 9", mutated)
        self.assertIn("pub const ALL: [Self; 10]", mutated)
        self.assertIn("9 => Ok(Self::Phase2OracleDType)", mutated)
        self.assertIn('Self::Phase2OracleDType => "phase2-oracle-dtype"', mutated)
        self.assertNotIn(
            "_ =>", mutated.split("impl RuntimeDType", 1)[1].split("\n}\n", 1)[0]
        )

    def test_dtype_mutation_refuses_a_drifted_owner_shape(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        drifted = source.replace("8 => Ok(Self::I16),", "8 => Ok(Self::I16), /* moved */")
        with self.assertRaisesRegex(oracle.OracleFailure, "RuntimeDType owner shape drifted"):
            oracle.mutate_runtime_dtype(drifted)

    def test_controlled_mutation_restores_original_bytes_on_failure(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "lib.rs"
            path.write_bytes(source)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_vocab_mutation(path):
                    mutated = path.read_text(encoding="utf-8")
                    self.assertIn("Phase2OracleMutation", mutated)
                    self.assertIn("Phase2OracleDType", mutated)
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), source)

    def test_endpoint_scan_is_green_on_the_repository(self) -> None:
        self.assertEqual(oracle.endpoint_violations(oracle.REPO_ROOT), [])

    def test_endpoint_scan_detects_a_reintroduced_infallible_wrapper(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            write_endpoint_fixture(
                root,
                valid_host_endpoint_source()
                + "pub fn lower_compiled_program() {}\n",
            )
            violations = oracle.endpoint_violations(root)
            self.assertTrue(
                any("pub fn lower_compiled_program" in violation for violation in violations),
                violations,
            )

    def test_endpoint_scan_detects_unit_and_raw_generic_field_escape_hatches(self) -> None:
        with tempfile.TemporaryDirectory() as raw_dir:
            root = Path(raw_dir)
            write_endpoint_fixture(
                root,
                valid_host_endpoint_source()
                + "fn lookup_adt_ctor() {}\n"
                + "let placeholder = HostExpr::new(HostExprKind::Unit);\n"
                + "_ => HostExpr::new(HostExprKind::Unit)\n",
            )
            violations = oracle.endpoint_violations(root)
            for forbidden in (
                "fn lookup_adt_ctor(",
                "let placeholder = HostExpr::new(HostExprKind::Unit)",
                "_ => HostExpr::new(HostExprKind::Unit)",
            ):
                self.assertTrue(
                    any(forbidden in violation for violation in violations),
                    violations,
                )


if __name__ == "__main__":
    unittest.main()
