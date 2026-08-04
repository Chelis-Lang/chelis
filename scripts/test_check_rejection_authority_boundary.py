#!/usr/bin/env python3
"""Mutation controls for the rejection-authority privacy boundary."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "check_rejection_authority_boundary",
    HERE / "check_rejection_authority_boundary.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RejectionAuthorityBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = MODULE.SOURCE.read_text(encoding="utf-8")

    def test_production_source_has_one_validated_construction_boundary(self):
        self.assertEqual(MODULE.validate_source(self.source), [])

    def test_added_public_issue_constructor_is_rejected(self):
        mutated = self.source.replace(
            "impl IssueRef {",
            "impl IssueRef {\n    pub const fn redteam_unchecked(number: NonZeroU32) -> Self { Self(number) }",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("redteam_unchecked" in error for error in errors), errors)

    def test_public_generic_authority_constructor_is_rejected(self):
        mutated = self.source.replace(
            "    const fn unimplemented(",
            "    pub const fn unimplemented(",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("unimplemented" in error for error in errors), errors)

    def test_same_module_public_free_function_is_rejected(self):
        mutated = self.source.replace(
            "/// Sole downstream construction edge for a deliberate rejection.",
            "pub fn redteam_unchecked_authority() -> RejectionAuthority {\n"
            "    RejectionAuthority {\n"
            "        citation: RejectionCitation::Issue(IssueRef(NonZeroU32::MIN)),\n"
            "        hint: \"forged\",\n"
            "    }\n"
            "}\n\n"
            "/// Sole downstream construction edge for a deliberate rejection.",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(
            any("redteam_unchecked_authority" in error for error in errors), errors
        )

    def test_direct_public_builder_call_is_rejected_in_production(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            'let _ = chelis_types::unsupported::__build_deliberate_rejection('
            '"[04-TOT-2]", "hint");',
        )
        self.assertTrue(any("direct authority builder" in error for error in errors), errors)

    def test_public_builder_cannot_skip_registry_validation(self):
        mutated = self.source.replace(
            "let issue = match IssueRef::new(issue)",
            "let issue = match forged_issue(issue)",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("macro edge" in error for error in errors), errors)

    def test_production_authorities_do_not_use_response_only_atoms(self):
        self.assertEqual(MODULE.validate_production_usage(), [])

    def test_response_atom_mutation_is_rejected(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            'let _ = deliberate_rejection!("[05-UNS-1]", "generic");',
        )
        self.assertTrue(any("deciding atom" in error for error in errors), errors)

    def test_diagnostic_migration_issue_is_not_a_capability_owner(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            'let _ = unimplemented_rejection!(959, "unrelated capability");',
        )
        self.assertTrue(any("diagnostic migration" in error for error in errors), errors)

    def test_diagnostic_issue_with_token_spacing_is_still_rejected(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            'let _ = unimplemented_rejection /* gap */ ! (959, "unrelated");',
        )
        self.assertTrue(any("diagnostic migration" in error for error in errors), errors)

    def test_non_crates_custom_target_cannot_call_builder_directly(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            member = root / "tree-sitter-style"
            member.mkdir()
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["tree-sitter-style"]\nresolver = "3"\n'
            )
            (member / "Cargo.toml").write_text(
                "[package]\n"
                'name = "tree-sitter-style"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n\n'
                "[lib]\n"
                'path = "bindings/rust/lib.rs"\n'
            )
            source = member / "bindings/rust/lib.rs"
            source.parent.mkdir(parents=True)
            source.write_text(
                "let _ = chelis_types::unsupported::"
                '__build_unimplemented_rejection(729, "hidden");\n'
            )

            errors = MODULE.validate_production_usage(root)
            self.assertTrue(
                any("direct authority builder" in error for error in errors),
                errors,
            )

    def test_reachable_src_tests_module_cannot_call_builder_directly(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            member = root / "crates/example"
            member.mkdir(parents=True)
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = ["crates/example"]\nresolver = "3"\n'
            )
            (member / "Cargo.toml").write_text(
                "[package]\n"
                'name = "example"\n'
                'version = "0.0.0"\n'
                'edition = "2024"\n'
            )
            lib = member / "src/lib.rs"
            lib.parent.mkdir()
            lib.write_text("mod tests;\n")
            source = member / "src/tests/prod.rs"
            source.parent.mkdir()
            source.write_text(
                "let _ = chelis_types::unsupported::"
                '__build_unimplemented_rejection(729, "hidden");\n'
            )

            errors = MODULE.validate_production_usage(root)
            self.assertTrue(
                any("direct authority builder" in error for error in errors),
                errors,
            )


if __name__ == "__main__":
    unittest.main()
