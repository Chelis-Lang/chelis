#!/usr/bin/env python3

from pathlib import Path
import tempfile
import unittest

import loud_unsupported_phase2_oracle as oracle




class LoudUnsupportedPhase2OracleTests(unittest.TestCase):

    def test_vocab_mutation_needs_all_independent_consumer_failures(self) -> None:
        evidence = (
            "non-exhaustive patterns",
            "crates/chelis-surf/src/resugar.rs",
            "crates/chelis-checker-core/src/infer/expr.rs",
            "crates/chelis-effects-core/src/lib.rs",
            "crates/chelis-runtime/src/lib.rs",
        )
        output = "\n".join(evidence)
        oracle.validate_vocab_mutation_output(101, output)
        for missing in evidence:
            with self.subTest(missing=missing), self.assertRaisesRegex(
                oracle.OracleFailure, "required exhaustive consumer evidence",
            ):
                oracle.validate_vocab_mutation_output(
                    101, "\n".join(line for line in evidence if line != missing)
                )
        with self.assertRaisesRegex(oracle.OracleFailure, "compiled successfully"):
            oracle.validate_vocab_mutation_output(0, output)

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
        self.assertIn("Phase2OracleDType = 10", mutated)
        self.assertIn("pub const ALL: [Self; 11]", mutated)
        self.assertIn("10 => Ok(Self::Phase2OracleDType)", mutated)
        self.assertIn('Self::Phase2OracleDType => "phase2-oracle-dtype"', mutated)
        self.assertIn("Self::Phase2OracleDType => (Repr::Ieee754Binary32, Some(A::Ieee754Binary32))", mutated)
        self.assertNotIn(
            "_ =>", mutated.split("impl RuntimeDType", 1)[1].split("\n}\n", 1)[0]
        )

    def test_host_abi_mutation_updates_owner_without_arming_the_variant(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HOST_ABI_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_host_abi_type(source)
        self.assertNotEqual(mutated, source)
        self.assertIn("Phase2OracleAbi,", mutated)
        # The variant is added to the DECLARATION only: no consumer arm is
        # pre-wired, so every exhaustive match must go red under cargo check.
        self.assertEqual(mutated.count("Phase2OracleAbi"), 1)

    def test_host_abi_mutation_refuses_a_drifted_owner_shape(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HOST_ABI_SOURCE).read_text(encoding="utf-8")
        drifted = source.replace(
            "pub(crate) enum HostAbiType {",
            "pub(crate) enum RenamedHostAbiType {",
        )
        with self.assertRaisesRegex(oracle.OracleFailure, "HostAbiType owner shape drifted"):
            oracle.mutate_host_abi_type(drifted)

    def test_host_abi_mutation_restores_original_bytes_on_failure(self) -> None:
        source = (oracle.REPO_ROOT / oracle.HOST_ABI_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "host_abi.rs"
            path.write_bytes(source)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_host_abi_mutation(path):
                    mutated = path.read_text(encoding="utf-8")
                    self.assertIn("Phase2OracleAbi", mutated)
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), source)

    def test_dtype_mutation_refuses_a_drifted_owner_shape(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        drifted = source.replace("9 => Ok(Self::Key),", "9 => Ok(Self::Key), /* moved */")
        with self.assertRaisesRegex(oracle.OracleFailure, "RuntimeDType owner shape drifted"):
            oracle.mutate_runtime_dtype(drifted)

    def test_dtype_mutation_refuses_a_drifted_c_macro_owner_shape(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        drifted = source.replace(
            'Self::Key => "CHELIS_DTYPE_KEY",',
            'Self::Key => "CHELIS_DTYPE_KEY_DRIFTED",',
        )
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

    def test_numeric_dtype_admission_restores_original_bytes_on_failure(self) -> None:
        original = (oracle.REPO_ROOT / oracle.NUMERIC_PRIM_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "prim.rs"
            path.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_numeric_dtype_admission(path):
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), original)

    def test_numeric_dtype_admission_refuses_drift_without_source_loss(self) -> None:
        original = b"pub enum Unrelated {}\n"
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "prim.rs"
            path.write_bytes(original)
            with self.assertRaisesRegex(oracle.OracleFailure, "numeric dtype admission"):
                with oracle.temporary_numeric_dtype_admission(path):
                    self.fail("the probe accepted a different owner")
            self.assertEqual(path.read_bytes(), original)



if __name__ == "__main__":
    unittest.main()
