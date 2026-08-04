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

    def test_same_module_private_constructor_composition_is_rejected(self):
        mutated = self.source + (
            "\nfn redteam_private_authority() -> RejectionAuthority {\n"
            "    let issue = IssueRef::new(729).unwrap();\n"
            "    RejectionAuthority::unimplemented(issue, \"forged\").unwrap()\n"
            "}\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("private authority token" in error for error in errors), errors)

    def test_same_module_private_constructor_alias_is_rejected(self):
        mutated = self.source + (
            "\nfn redteam_private_alias() {\n"
            "    use self::IssueRef as HiddenIssue;\n"
            "    let _ = HiddenIssue::new(729);\n"
            "}\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("private authority token" in error for error in errors), errors)

    def test_same_module_associated_const_cannot_construct_with_self(self):
        mutated = self.source.replace(
            "impl IssueRef {",
            "impl IssueRef {\n"
            "    pub const REDTEAM_ROGUE: Self = Self(NonZeroU32::MIN);",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("associated const" in error for error in errors), errors)

    def test_private_scalar_tuple_field_cannot_be_made_public(self):
        mutated = self.source.replace(
            "pub struct IssueRef(NonZeroU32);",
            "pub struct IssueRef(pub NonZeroU32);",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("private authority layout" in error for error in errors), errors)

    def test_balanced_alias_and_public_trait_cannot_bypass_token_counts(self):
        mutated = self.source.replace(
            "enum RejectionCitation {\n    Atom(SpecAtomRef),\n    Issue(IssueRef),",
            "enum RejectionCitation {\n    Atom(SpecAtomRef),\n    Issue(RedteamIssueAlias),",
            1,
        ).replace(
            "        issue: IssueRef,",
            "        issue: RedteamIssueAlias,",
            1,
        )
        mutated += (
            "\ntype RedteamIssueAlias = IssueRef;\n"
            "pub trait RedteamForge {\n"
            "    fn redteam_forge(number: NonZeroU32) -> Self;\n"
            "}\n"
            "impl RedteamForge for IssueRef {\n"
            "    fn redteam_forge(number: NonZeroU32) -> Self { Self(number) }\n"
            "}\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("public item inventory" in error for error in errors), errors)

    def test_balanced_private_alias_standard_trait_impl_is_rejected(self):
        mutated = self.source.replace(
            "        issue: IssueRef,",
            "        issue: RedteamIssueAlias,",
            1,
        ).replace(
            "    authority: RejectionAuthority,",
            "    authority: RedteamAuthorityAlias,",
            1,
        )
        mutated += (
            "\ntype RedteamIssueAlias = IssueRef;\n"
            "type RedteamAuthorityAlias = RejectionAuthority;\n"
            "impl From<u32> for RedteamIssueAlias {\n"
            "    fn from(value: u32) -> Self {\n"
            "        Self(NonZeroU32::new(value).unwrap())\n"
            "    }\n"
            "}\n"
            "impl From<u32> for RedteamAuthorityAlias {\n"
            "    fn from(value: u32) -> Self {\n"
            "        Self::unimplemented(RedteamIssueAlias::from(value), \"forged\")"
            ".unwrap()\n"
            "    }\n"
            "}\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("authority alias" in error for error in errors), errors)

    def test_balanced_grouped_use_alias_trait_impl_is_rejected(self):
        mutated = self.source.replace(
            "    pub const fn issue(self) -> Option<IssueRef> {",
            "    pub const fn issue(self) -> Option<RedteamIssueAlias> {",
            1,
        ).replace(
            "    authority: RejectionAuthority,",
            "    authority: RedteamAuthorityAlias,",
            1,
        )
        mutated += (
            "\nuse self::{IssueRef as RedteamIssueAlias, "
            "RejectionAuthority as RedteamAuthorityAlias};\n"
            "impl From<u32> for RedteamIssueAlias {\n"
            "    fn from(value: u32) -> Self {\n"
            "        Self(NonZeroU32::new(value).unwrap())\n"
            "    }\n"
            "}\n"
            "impl From<RedteamIssueAlias> for RedteamAuthorityAlias {\n"
            "    fn from(issue: RedteamIssueAlias) -> Self {\n"
            "        Self::unimplemented(issue, \"forged\").unwrap()\n"
            "    }\n"
            "}\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("authority alias" in error for error in errors), errors)

    def test_unicode_public_method_cannot_evade_the_owner_inventory(self):
        mutated = self.source.replace(
            "impl IssueRef {",
            "impl IssueRef {\n"
            "    pub fn \u751f\u6210(number: NonZeroU32) -> Self { Self(number) }",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("non-ASCII Rust token" in error for error in errors), errors)

    def test_macro_cannot_generate_an_uninventoried_associated_const(self):
        mutated = self.source.replace(
            "impl IssueRef {",
            "impl IssueRef {\n    redteam_expose!();",
            1,
        )
        mutated = (
            "macro_rules! redteam_expose {\n"
            "    () => { pub const REDTEAM_ROGUE: Self = Self(NonZeroU32::MIN); };\n"
            "}\n"
            + mutated
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("macro inventory" in error for error in errors), errors)

    def test_protected_wrapper_cannot_gain_an_unreviewed_derive(self):
        mutated = self.source.replace(
            "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]\n"
            "pub struct IssueRef(NonZeroU32);",
            "#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, RedteamForge)]\n"
            "pub struct IssueRef(NonZeroU32);",
            1,
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("private authority layout" in error for error in errors), errors)

    def test_balanced_builder_alias_cannot_escape_the_owner(self):
        mutated = self.source.replace(
            "$crate::unsupported::__build_unimplemented_rejection($issue, $hint)",
            "$crate::unsupported::redteam_rogue($issue, $hint)",
            1,
        )
        mutated += (
            "\npub(crate) use self::__build_unimplemented_rejection "
            "as redteam_rogue;\n"
        )
        errors = MODULE.validate_source(mutated)
        self.assertTrue(any("authority alias" in error for error in errors), errors)

    def test_grouped_builder_alias_cannot_escape_the_owner(self):
        errors = MODULE.validate_usage_source(
            "crates/chelis-types/src/lib.rs",
            "use crate::unsupported::{"
            "__build_unimplemented_rejection as redteam_build};\n"
            "let _ = redteam_build(729, \"hidden\");\n",
        )
        self.assertTrue(any("authority alias" in error for error in errors), errors)

    def test_direct_public_builder_call_is_rejected_in_production(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            'let _ = chelis_types::unsupported::__build_deliberate_rejection('
            '"[04-TOT-2]", "hint");',
        )
        self.assertTrue(any("direct authority builder" in error for error in errors), errors)

    def test_crate_root_cannot_call_public_builder_directly(self):
        errors = MODULE.validate_usage_source(
            "crates/chelis-types/src/lib.rs",
            'let _ = crate::unsupported::__build_unimplemented_rejection(729, "hidden");',
        )
        self.assertTrue(any("direct authority builder" in error for error in errors), errors)

    def test_descendant_module_cannot_call_private_owner_constructors(self):
        errors = MODULE.validate_usage_source(
            "crates/chelis-types/src/unsupported/redteam.rs",
            "let issue = super::IssueRef::new(729).unwrap();\n"
            "let _ = super::RejectionAuthority::unimplemented(issue, \"hidden\");\n",
        )
        self.assertTrue(
            any("private authority constructor" in error for error in errors),
            errors,
        )

    def test_unicode_alias_cannot_evade_the_production_boundary(self):
        errors = MODULE.validate_usage_source(
            "crates/chelis-types/src/unsupported/redteam.rs",
            "type \u4f2a = super::IssueRef;\n"
            "let _ = \u4f2a::new(729).unwrap();\n",
        )
        self.assertTrue(any("non-ASCII Rust token" in error for error in errors), errors)

    def test_non_ascii_comments_and_literals_remain_allowed(self):
        errors = MODULE.validate_usage_source(
            "crates/example/src/lib.rs",
            "// Unicode prose: \u03bb\nconst MESSAGE: &str = \"\u4f60\u597d\";\n",
        )
        self.assertEqual(errors, [])

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
