#!/usr/bin/env python3

from pathlib import Path
import re
import tempfile
import unittest

import diagnostic_kind_oracle as oracle


class DiagnosticKindOracleTests(unittest.TestCase):
    def test_focused_commands_execute_the_vocabulary_pipeline_and_doctests(self) -> None:
        rendered = [oracle.command_text(command) for command in oracle.FOCUSED_COMMANDS]
        joined = "\n".join(rendered)
        self.assertIn("-p chelis-vocab", joined)
        self.assertIn("diagnostic_kind_pipeline", joined)
        self.assertIn("-p chelis-compiler-api --doc", joined)

    def test_literal_mutation_is_outside_the_sealed_schema_module(self) -> None:
        source = (oracle.REPO_ROOT / oracle.MUTATION_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_diagnostic_literal(source)
        self.assertNotEqual(mutated, source)
        self.assertEqual(mutated.count("diagnostic_kind_oracle_literal"), 1)
        self.assertIn('kind: "unsupported_feature".to_owned()', mutated)

    def test_kind_write_mutation_targets_the_private_field(self) -> None:
        source = (oracle.REPO_ROOT / oracle.MUTATION_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_diagnostic_kind(source)
        self.assertNotEqual(mutated, source)
        self.assertEqual(mutated.count("diagnostic_kind_oracle_mutation"), 1)
        self.assertIn('diagnostic.kind = "compile_error".to_owned()', mutated)

    def test_wire_bridge_mutation_adds_a_working_producer_conversion(self) -> None:
        source = (oracle.REPO_ROOT / oracle.SCHEMA_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_wire_bridge(source)
        self.assertNotEqual(mutated, source)
        self.assertEqual(mutated.count("impl From<WireDiagnostic> for Diagnostic"), 1)
        self.assertIn("kind: wire.kind", mutated)

    def test_wire_bridge_mutation_runs_the_compiler_api_doctests(self) -> None:
        self.assertEqual(
            oracle.command_text(oracle.WIRE_BRIDGE_CHECK),
            "cargo test -p chelis-compiler-api --doc",
        )

    def test_vocabulary_mutation_adds_one_fully_rendered_owner_variant(self) -> None:
        source = (oracle.REPO_ROOT / oracle.VOCAB_SOURCE).read_text(encoding="utf-8")
        mutated = oracle.mutate_diagnostic_vocabulary(source)
        self.assertNotEqual(mutated, source)
        self.assertEqual(mutated.count("Phase3OracleKind"), 3)
        # Derived from the source rather than hard-coded. The property
        # under test is "the mutation adds exactly one variant and grows
        # `ALL` by one", which is independent of how many kinds the
        # vocabulary holds. A literal goes stale every time a kind is
        # added and then fails for a reason unrelated to the oracle.
        declared = re.search(r"pub const ALL: \[Self; (\d+)\]", source)
        self.assertIsNotNone(declared, "the vocabulary must declare `ALL`")
        assert declared is not None
        self.assertIn(
            f"pub const ALL: [Self; {int(declared.group(1)) + 1}]", mutated
        )
        self.assertIn(
            'Self::Phase3OracleKind => "phase3_oracle_kind"',
            mutated,
        )

    def test_vocabulary_mutation_checks_workspace_libraries(self) -> None:
        self.assertEqual(
            oracle.command_text(oracle.VOCABULARY_CHECK),
            "cargo check --workspace --lib",
        )

    def test_mutations_refuse_a_drifted_anchor(self) -> None:
        source = (oracle.REPO_ROOT / oracle.MUTATION_SOURCE).read_text(encoding="utf-8")
        drifted = source.replace("\n#[cfg(test)]", "\n// moved test module", 1)
        with self.assertRaisesRegex(oracle.OracleFailure, "mutation anchor"):
            oracle.mutate_diagnostic_literal(drifted)

    def test_temporary_mutation_restores_original_bytes_after_failure(self) -> None:
        original = (oracle.REPO_ROOT / oracle.MUTATION_SOURCE).read_bytes()
        with tempfile.TemporaryDirectory() as raw_dir:
            path = Path(raw_dir) / "context.rs"
            path.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                with oracle.temporary_mutation(path, oracle.mutate_diagnostic_kind):
                    self.assertIn(
                        "diagnostic_kind_oracle_mutation",
                        path.read_text(encoding="utf-8"),
                    )
                    raise RuntimeError("probe failed")
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
