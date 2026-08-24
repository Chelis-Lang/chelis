#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import dtype_builtin_atom_closure_oracle as oracle  # noqa: E402


SPEC = """\
# synthetic spec

> **[05-OP-1]** `add(left, right) -> result` computes at the operand width,
> returns the operand dtype, and has adjoint `(g, g)`. It has no accumulator.

> **[05-OP-2]** `to_string(value) -> result` renders recursively, rejects
> functions, is non-differentiable, and has no accumulator.
"""

GENERATED = """\
pub const SPEC_ATOMS: &[&str] = &[
    "[05-OP-1]",
    "[05-OP-2]",
];
"""


def numeric_identity() -> oracle.CapabilityIdentity:
    return oracle.CapabilityIdentity(
        builtin="add",
        domain=oracle.Domain.NUMERIC,
        case="TableA",
    )


def sibling_identity() -> oracle.CapabilityIdentity:
    return oracle.CapabilityIdentity(
        builtin="to_string",
        domain=oracle.Domain.BOUNDARY,
        case="ToStringRecursive",
    )


def registrations() -> tuple[oracle.SemanticRegistration, ...]:
    return (
        oracle.SemanticRegistration(numeric_identity(), "[05-OP-1]"),
        oracle.SemanticRegistration(sibling_identity(), "[05-OP-2]"),
    )


ESSENTIAL = {
    "[05-OP-1]": (
        "operand width",
        "operand dtype",
        "adjoint `(g, g)`",
        "no accumulator",
    ),
    "[05-OP-2]": (
        "renders recursively",
        "rejects functions",
        "non-differentiable",
        "no accumulator",
    ),
}

RISC_SOURCE = """\
pub enum RiscOp {
    Add,
    Copy,
}
pub enum RiscAtomIdentity {
    Add,
}
impl RiscAtomIdentity {
    pub const ALL: &[Self] = &[Self::Add];
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Add => "add",
        }
    }
}
impl RiscOp {
    pub const fn atom_disposition(&self) -> RiscAtomDisposition {
        use RiscAtomIdentity as Id;
        match self {
            Self::Add => Semantic(Id::Add),
            Self::Copy => Structural,
        }
    }
}
"""


class ClosureValidationTests(unittest.TestCase):
    def validate(
        self,
        *,
        discovered: tuple[oracle.CapabilityIdentity, ...] | None = None,
        registered: tuple[oracle.SemanticRegistration, ...] | None = None,
        spec: str = SPEC,
        generated: str = GENERATED,
        essential: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        oracle.validate_closure(
            discovered=(numeric_identity(), sibling_identity())
            if discovered is None
            else discovered,
            registrations=registrations() if registered is None else registered,
            spec=spec,
            generated_registry=generated,
            essential_semantics=ESSENTIAL if essential is None else essential,
        )

    def assert_fails(self, message: str, **kwargs: object) -> None:
        with self.assertRaisesRegex(oracle.OracleError, message):
            self.validate(**kwargs)

    def test_exact_synthetic_bijection_passes(self) -> None:
        self.validate()

    def test_deleted_registration_fails(self) -> None:
        self.assert_fails(
            "missing semantic registration.*to_string",
            registered=registrations()[:1],
        )

    def test_duplicate_registration_fails(self) -> None:
        self.assert_fails(
            "duplicate semantic registration.*add",
            registered=registrations() + (registrations()[0],),
        )

    def test_stale_registration_fails(self) -> None:
        stale = oracle.SemanticRegistration(
            oracle.CapabilityIdentity(
                builtin="renamed_add",
                domain=oracle.Domain.NUMERIC,
                case="TableA",
            ),
            "[05-OP-1]",
        )
        self.assert_fails(
            "stale semantic registration.*renamed_add",
            registered=(registrations()[0], registrations()[1], stale),
        )

    def test_renamed_discovery_identity_fails_without_exact_successor(self) -> None:
        renamed = oracle.CapabilityIdentity(
            builtin="plus",
            domain=oracle.Domain.NUMERIC,
            case="TableA",
        )
        self.assert_fails(
            "missing semantic registration.*plus",
            discovered=(renamed, sibling_identity()),
        )

    def test_cross_reference_does_not_define_an_atom(self) -> None:
        self.assert_fails(
            "does not exist as one normative definition line",
            spec="The language follows [05-OP-1].\n" + SPEC.replace(
                "> **[05-OP-1]**", "> **[05-OLD-1]**", 1
            ),
        )

    def test_duplicate_normative_atom_definition_fails(self) -> None:
        self.assert_fails(
            "must have exactly one normative definition line",
            spec=SPEC + "\n> **[05-OP-1]** `other` duplicate.\n",
        )

    def test_adjacent_blockquoted_atoms_have_disjoint_governing_names(self) -> None:
        contiguous = """\
> **[05-OP-1]** `add(left, right) -> result` owns add.
>
> **[05-OP-2]** `mul(left, right) -> result` owns mul.
"""
        blocks = oracle.atom_blocks(contiguous)
        self.assertNotIn("[05-OP-2]", blocks["[05-OP-1]"])
        self.assertEqual(oracle.governing_names(blocks["[05-OP-1]"]), frozenset({"add"}))
        self.assertEqual(oracle.governing_names(blocks["[05-OP-2]"]), frozenset({"mul"}))

    def test_only_first_table_cell_supplies_a_governing_identity(self) -> None:
        block = """\
> **[05-OP-1]** `carrier(value) -> result` owns exact callables.
>
> | callable | exact signature |
> |---|---|
> | scalar construction | `int64_t add(int64_t lhs, int64_t rhs)` |
> | `mul` | `(int64,int64)->int64` |
> | 3 | `stdlib::sub` | `(int64,int64)->int64` |
"""
        self.assertEqual(
            oracle.governing_names(block),
            frozenset({"carrier", "mul"}),
        )

    def test_unnumbered_prose_is_not_an_atom(self) -> None:
        wrong = (
            oracle.SemanticRegistration(numeric_identity(), "section 2.1"),
            registrations()[1],
        )
        self.assert_fails("must cite exact.*05-OP-N", registered=wrong)

    def test_issue_citation_is_not_an_atom(self) -> None:
        wrong = (
            oracle.SemanticRegistration(numeric_identity(), "#729"),
            registrations()[1],
        )
        self.assert_fails("must cite exact.*05-OP-N", registered=wrong)

    def test_default_or_fallback_registration_is_forbidden(self) -> None:
        for case in ("default", "*", "fallback"):
            with self.subTest(case=case):
                identity = oracle.CapabilityIdentity(
                    builtin="add",
                    domain=oracle.Domain.NUMERIC,
                    case=case,
                )
                wrong = (
                    oracle.SemanticRegistration(identity, "[05-OP-1]"),
                    registrations()[1],
                )
                self.assert_fails("forbidden default/fallback", registered=wrong)

    def test_compatibility_or_age_exception_is_forbidden(self) -> None:
        for case in ("LegacyAdd", "GrandfatheredAdd", "CompatAdd", "DeprecatedAdd"):
            with self.subTest(case=case):
                identity = oracle.CapabilityIdentity(
                    builtin="add",
                    domain=oracle.Domain.NUMERIC,
                    case=case,
                )
                wrong = (
                    oracle.SemanticRegistration(identity, "[05-OP-1]"),
                    registrations()[1],
                )
                self.assert_fails("compatibility/age exception", registered=wrong)

    def test_missing_generated_registry_membership_fails(self) -> None:
        self.assert_fails(
            "generated rejection registry is missing.*05-OP-2",
            generated=GENERATED.replace('    "[05-OP-2]",\n', ""),
        )

    def test_wrong_atom_mapping_fails_semantic_governance(self) -> None:
        wrong = (
            oracle.SemanticRegistration(numeric_identity(), "[05-OP-2]"),
            oracle.SemanticRegistration(sibling_identity(), "[05-OP-1]"),
        )
        self.assert_fails("does not name exact builtin.*add", registered=wrong)

    def test_semantic_mutation_fails(self) -> None:
        self.assert_fails(
            "missing essential semantic clause.*operand width",
            spec=SPEC.replace("operand width", "host f64 width", 1),
        )

    def test_registered_atom_without_semantic_mutation_contract_fails(self) -> None:
        sub = oracle.CapabilityIdentity("sub", oracle.Domain.NUMERIC, "TableA")
        self.assert_fails(
            "05-OP-3.*no essential-semantic mutation contract",
            discovered=(numeric_identity(), sibling_identity(), sub),
            registered=registrations()
            + (oracle.SemanticRegistration(sub, "[05-OP-3]"),),
            spec=SPEC + "\n> **[05-OP-3]** `sub(left, right) -> result` subtracts.\n",
            generated=GENERATED.replace("];", '    "[05-OP-3]",\n];'),
        )

    def test_atom_membership_is_explicit_op_1_through_39(self) -> None:
        expected = tuple(f"[05-OP-{number}]" for number in range(1, 40))
        oracle.validate_required_atom_numbers(
            "\n".join(f"> **{atom}** `op_{i}`" for i, atom in enumerate(expected, 1)),
            expected,
        )
        with self.assertRaisesRegex(oracle.OracleError, "missing required atom.*05-OP-23"):
            oracle.validate_required_atom_numbers(
                "\n".join(
                    f"> **{atom}** `op_{i}`"
                    for i, atom in enumerate(expected, 1)
                    if atom != "[05-OP-23]"
                ),
                expected,
            )


class SourceDiscoveryTests(unittest.TestCase):
    def test_builtin_declarations_are_discovered_without_a_count_allowlist(self) -> None:
        source = """\
pub const BUILTINS: &[BuiltinDecl] = &[
    BuiltinDecl {
        name: "add",
        inference: x,
    },
    BuiltinDecl {
        name: "to_string",
        inference: y,
    },
];
"""
        self.assertEqual(
            oracle.discover_builtin_names(source),
            ("add", "to_string"),
        )

    def test_duplicate_builtin_declaration_fails(self) -> None:
        source = """\
pub const BUILTINS: &[BuiltinDecl] = &[
    BuiltinDecl { name: "add", inference: x },
    BuiltinDecl { name: "add", inference: y },
];
"""
        with self.assertRaisesRegex(oracle.OracleError, "duplicate BuiltinDecl.*add"):
            oracle.discover_builtin_names(source)

    def test_risc_variants_are_discovered_without_payload_or_comment_noise(self) -> None:
        source = """\
pub enum RiscOp {
    Add,
    Sum { axis: usize, accumulator: Prim },
    /// A documented fake `NotAVariant`.
    ReduceWindow { reducer: Kind, window: Vec<usize> },
    Copy,
}
"""
        self.assertEqual(
            oracle.discover_risc_variants(source),
            ("Add", "Sum", "ReduceWindow", "Copy"),
        )

    def test_risc_identity_source_is_exact_and_closed(self) -> None:
        self.assertEqual(
            oracle.discover_risc_atom_identities(RISC_SOURCE),
            (numeric_identity(),),
        )

    def test_risc_identity_missing_from_all_fails(self) -> None:
        with self.assertRaisesRegex(oracle.OracleError, "RiscAtomIdentity.*ALL"):
            oracle.discover_risc_atom_identities(
                RISC_SOURCE.replace("&[Self::Add]", "&[]")
            )

    def test_risc_variant_without_disposition_fails(self) -> None:
        with self.assertRaisesRegex(oracle.OracleError, "RiscOp.*disposition.*Copy"):
            oracle.discover_risc_atom_identities(
                RISC_SOURCE.replace("            Self::Copy => Structural,\n", "")
            )

    def test_typed_builtin_capabilities_supply_the_domain_case_universe(self) -> None:
        source = """\
const NUMERIC_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl {
    domains: NUMERIC_DOMAIN,
    sibling_cases: &[],
};
const TO_STRING_CASES: &[BuiltinSiblingCaseDecl] = &[BuiltinSiblingCaseDecl {
    domain: BuiltinSemanticDomain::Boundary,
    case: BuiltinSiblingCaseId::ToStringRecursive,
}];
const TO_STRING_CAPABILITY: BuiltinCapabilityDecl = BuiltinCapabilityDecl {
    domains: BOUNDARY_DOMAIN,
    sibling_cases: TO_STRING_CASES,
};
pub const BUILTINS: &[BuiltinDecl] = &[
    BuiltinDecl { name: "add", capability: NUMERIC_CAPABILITY, inference: x },
    BuiltinDecl { name: "to_string", capability: TO_STRING_CAPABILITY, inference: y },
];
"""
        self.assertEqual(
            oracle.discover_builtin_capabilities(source),
            (numeric_identity(), sibling_identity()),
        )

    def test_builtin_without_typed_capability_field_fails(self) -> None:
        source = """\
pub const BUILTINS: &[BuiltinDecl] = &[
    BuiltinDecl { name: "new_op", inference: x },
];
"""
        with self.assertRaisesRegex(oracle.OracleError, "BuiltinDecl.*new_op.*capability"):
            oracle.discover_builtin_capabilities(source)


class CommandTests(unittest.TestCase):
    def test_success_line_is_last(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for relative, text in (
                (oracle.BUILTINS_REL, "pub const BUILTINS: &[BuiltinDecl] = &[];\n"),
                (oracle.RISC_REL, "pub enum RiscOp {}\n"),
                (
                    oracle.SPEC_REL,
                    "\n".join(
                        f"> **[05-OP-{number}]** `op_{number}` semantic"
                        for number in range(1, 40)
                    ),
                ),
                (
                    oracle.GENERATED_REL,
                    "\n".join(f'"[05-OP-{number}]",' for number in range(1, 40)),
                ),
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                oracle.main(root=root, essential={})
            self.assertEqual(output.getvalue().splitlines()[-1], oracle.PASS_LINE)


if __name__ == "__main__":
    unittest.main()
