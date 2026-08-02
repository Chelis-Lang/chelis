#!/usr/bin/env python3
"""Mutation controls for the rejection-authority privacy boundary."""

from __future__ import annotations

import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
