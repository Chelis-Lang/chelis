"""Test-first contract for execution-issued CompilerJson binding authority."""

import unittest


class CompilerJsonAuthority(unittest.TestCase):
    def test_actual_registration_conversion_and_wire_execution_issue_authority(self):
        # The positive leg must run actual compiled adapters/PyO3 exposure and
        # the current private wire verifier. A synthetic receipt is not a
        # positive control. The Rust acceptance entry owns that heavy run.
        from capacity_census_compiler_json import VerifiedCompilerJsonBindings

        with self.assertRaises(TypeError):
            VerifiedCompilerJsonBindings()

    def test_each_adapter_binds_exact_root_and_direction(self):
        self.fail("spec/11 §1.1: exact four output roots and one eval map input; reject root/direction/slot swaps")

    def test_compiled_conversion_ownership_cannot_be_a_named_helper(self):
        self.fail("spec/11 §1.1: actual conversion trait owner/callee/payload; reject decoys, aliases to a wrong owner, missing or extra conversions")

    def test_eval_reader_ownership_requires_deserialization_into_the_tensor_map(self):
        self.fail("spec/10 §3.2: actual typed Deserialize ingress; compiler-api Serialize proof is insufficient")

    def test_compiler_json_authority_stays_inside_its_adapter_subtree(self):
        self.fail("C6: exact adapter succeeds; String/numeric siblings, wrapper lookalikes and extra fields reject")

    def test_stale_or_supplied_receipts_cannot_issue_binding_authority(self):
        self.fail("C6: current private execution witness required; stale source/artifacts and descriptor/baseline substitutions reject")

    def test_all_selected_native_cases_must_execute_without_skips(self):
        self.fail("C6: complete framework start/outcome evidence; missing/failed/ignored/skipped/zero-match selections reject")


if __name__ == "__main__":
    unittest.main()
