#!/usr/bin/env python3
"""Mutation controls for the rejection-authority privacy boundary."""

from __future__ import annotations

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from generate_rejection_registries import (
    ProductionSource,
    discover_production_sources,
)

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
        self.assertEqual(
            MODULE.validate_production_usage(
                sources=[
                    ProductionSource(
                        path=Path("crates/example/src/lib.rs"),
                        source=(
                            'deliberate_rejection!("[04-TOT-2]", "semantic");\n'
                            'unimplemented_rejection!(879, "implementation");\n'
                        ),
                    )
                ]
            ),
            [],
        )

    def test_main_uses_a_fresh_shared_production_graph(self):
        with mock.patch.object(
            MODULE, "discover_production_sources", return_value=[]
        ) as discover, redirect_stdout(io.StringIO()):
            self.assertEqual(MODULE.main(), 0)
        discover.assert_called_once_with(MODULE.ROOT)

    def test_boundary_consumes_the_shared_production_source_owner(self):
        self.assertIs(MODULE.discover_production_sources, discover_production_sources)
        self.assertFalse(hasattr(MODULE, "discover_authority_boundary_sources"))

    def test_path_reached_source_outside_target_directory_is_checked(self):
        errors = MODULE.validate_production_usage(
            sources=[
                ProductionSource(
                    path=Path("tests/support/outside.rs"),
                    source=(
                        "fn outside(issue: u32) { "
                        "chelis_types::unsupported::__build_unimplemented_rejection("
                        'issue, "dynamic"); }'
                    ),
                )
            ]
        )
        self.assertTrue(
            any("direct authority builder" in error for error in errors), errors
        )

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
