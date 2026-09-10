"""Exact registration obligations of #1294; no name/prose inference."""
from __future__ import annotations

import unittest
from pathlib import Path

from scripts import builtin_atom_registry as registry
from scripts import builtin_atom_semantic_contracts as semantics


SPEC = """# Operations

The [builtin identity registry](registry/builtin_semantic_identities.md)
is incorporated by reference into each numbered operation atom named in its
Atom column.

> **[05-OP-45]** `add` uses the exact operand dtype.
> `add(left, right)` takes two same-shaped operands of an active arithmetic dtype.
> The result has the operand dtype; integer overflow traps.
> Its float adjoint is `(g, g)` and it has no accumulator.
"""
TABLE = """# Builtin semantic identities

| Identity | Atom |
|---|---|
| `Numeric:add:TableA` | [05-OP-45] |
"""


class RegistryTests(unittest.TestCase):
    def validate(self, table=TABLE, spec=SPEC, discovered=("Numeric:add:TableA",)):
        return registry.validate(discovered, table, spec, {"[05-OP-45]"})

    def test_exact_identity_passes(self):
        self.assertEqual(self.validate(), {"Numeric:add:TableA": "[05-OP-45]"})

    def test_missing_identity_fails(self):
        with self.assertRaisesRegex(registry.RegistryError, "missing"):
            self.validate(table=TABLE.split("| `Numeric:")[0])

    def test_duplicate_registration_fails_before_set_conversion(self):
        with self.assertRaisesRegex(registry.RegistryError, "duplicate"):
            self.validate(table=TABLE + "| `Numeric:add:TableA` | [05-OP-45] |\n")

    def test_stale_identity_fails(self):
        with self.assertRaisesRegex(registry.RegistryError, "stale"):
            self.validate(table=TABLE.replace(":add:", ":mul:"))

    def test_namespace_does_not_collapse(self):
        with self.assertRaisesRegex(registry.RegistryError, "identity|missing"):
            self.validate(table=TABLE.replace(":add:", ":stdlib::add:"))

    def test_wrong_existing_atom_fails_governance(self):
        other = registry.atom_blocks(SPEC)["[05-OP-45]"].replace(
            "[05-OP-45]", "[05-OP-46]"
        ).replace("`add", "`mul")
        with self.assertRaisesRegex(registry.RegistryError, "govern"):
            registry.validate(("Numeric:add:TableA",), TABLE.replace("45", "46"),
                              SPEC + "\n" + other, {"[05-OP-45]", "[05-OP-46]"})

    def test_cross_reference_is_not_definition(self):
        with self.assertRaisesRegex(registry.RegistryError, "definition"):
            self.validate(spec=SPEC.replace("> **[05-OP-45]**", "See [05-OP-45]"))

    def test_missing_generated_membership_fails(self):
        with self.assertRaisesRegex(registry.RegistryError, "generated"):
            registry.validate(("Numeric:add:TableA",), TABLE, SPEC, set())

    def test_non_op_and_issue_citations_fail(self):
        for citation in ("[05-OBS-1]", "chelis#1294", "spec/05", "[05-OP-999]"):
            with self.subTest(citation=citation), self.assertRaises(registry.RegistryError):
                self.validate(table=TABLE.replace("[05-OP-45]", citation))

    def test_discovery_duplicates_fail(self):
        with self.assertRaisesRegex(registry.RegistryError, "duplicate"):
            self.validate(discovered=("Numeric:add:TableA", "Numeric:add:TableA"))

    def test_empty_discovery_cannot_pass(self):
        with self.assertRaisesRegex(registry.RegistryError, "empty"):
            self.validate(discovered=())

    def test_registry_requires_normative_incorporation(self):
        with self.assertRaisesRegex(registry.RegistryError, "incorporat"):
            self.validate(spec=SPEC.replace("incorporated by reference", "mentioned"))

    def test_shared_incorporation_cannot_be_duplicated(self):
        with self.assertRaisesRegex(registry.RegistryError, "incorporat"):
            self.validate(spec=SPEC.split("> **")[0] + SPEC)

    def test_callable_cannot_be_replaced_by_an_ordinary_prose_mention(self):
        with self.assertRaisesRegex(registry.RegistryError, "govern"):
            self.validate(spec=SPEC.replace("`add`", "add").replace(
                "`add(left, right)`", "add(left, right)"
            ))

    def test_namespaced_callable_cannot_authorize_a_builtin(self):
        with self.assertRaisesRegex(registry.RegistryError, "govern"):
            self.validate(spec=SPEC.replace("`add", "`stdlib::add"))


class SemanticAuthorityTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.spec = (root / "spec/05-risc-primitives.md").read_text()
        self.rows = registry.parse_registry(
            (root / "spec/registry/builtin_semantic_identities.md").read_text()
        )

    def test_reference_to_callable_does_not_confer_its_authority(self):
        for identity, wrong_atom in (
            ("Numeric:div:TableA", "[05-OP-11]"),
            ("Boundary:to_string:ToStringScalar", "[05-OP-5]"),
            ("Numeric:relu:TableA", "[05-OP-40]"),
        ):
            with self.subTest(identity=identity), self.assertRaises(registry.RegistryError):
                semantics.validate_semantics({**self.rows, identity: wrong_atom}, self.spec)

    def test_new_ambiguous_reference_requires_a_discriminating_contract(self):
        changed = self.spec + "\n> **[05-OP-999]** This atom mentions `argmin_reduce`.\n"
        with self.assertRaisesRegex(registry.RegistryError, "ambiguous"):
            semantics.validate_semantics(self.rows, changed)

    def test_sparse_diagnostic_references_do_not_confer_builtin_authority(self):
        semantics.validate_semantics(self.rows, self.spec)
        for operation in ("gather", "scatter", "scatter_replace", "scatter_elements"):
            identity = next(name for name in self.rows if name.split(":")[1] == operation)
            wrong = {**self.rows, identity: "[05-OP-33]"}
            with self.subTest(operation=operation), self.assertRaisesRegex(
                registry.RegistryError, "govern"
            ):
                semantics.validate_semantics(wrong, self.spec)

    def test_actual_contract_does_not_require_field_labels(self):
        import re
        changed = re.sub(r"\b(?:Signature|Domain|Result|Failure|Adjoint|Accumulator):", "", self.spec)
        semantics.validate_semantics(self.rows, changed)

    def test_movement_trap_references_keep_the_builtin_authority(self):
        blocks = registry.atom_blocks(self.spec)
        for reference_atom in ("[05-OP-9]", "[05-OP-33]"):
            block = blocks[reference_atom]
            changed = self.spec.replace(block, block +
                "> Numeric failures use operation `shrink` or `stride`.\n")
            with self.subTest(reference_atom=reference_atom):
                semantics.validate_semantics(self.rows, changed)
                for operation in ("shrink", "stride"):
                    identity = next(name for name in self.rows
                                    if name.split(":")[1] == operation)
                    wrong = {identity: reference_atom,
                             **{name: atom for name, atom in self.rows.items()
                                if name != identity}}
                    with self.assertRaisesRegex(registry.RegistryError, "govern"):
                        semantics.validate_semantics(wrong, changed)

    def test_movement_ambiguity_requires_the_real_contract(self):
        blocks = registry.atom_blocks(self.spec)
        for operation in ("shrink", "stride", "permute"):
            identity = next(name for name in self.rows if name.split(":")[1] == operation)
            atom = self.rows[identity]
            clause = "`reshape(x,shape)`, `permute(x,axes)`, `expand(x,axis,size)`"
            block = blocks[atom]
            missing = "> " + semantics.normalized(block).replace(clause, "REMOVED") + "\n"
            changed = self.spec.replace(block, missing)
            changed += f"\n> **[05-OP-999]** Numeric failures use operation `{operation}`.\n"
            rows = {identity: atom, **self.rows}
            with self.subTest(operation=operation), self.assertRaisesRegex(
                registry.RegistryError, f"contract of {identity}"
            ):
                semantics.validate_semantics(rows, changed)


if __name__ == "__main__":
    unittest.main()
