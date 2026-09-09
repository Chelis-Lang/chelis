"""Exact registration obligations of #1294; no name/prose inference."""
from __future__ import annotations

import unittest

from scripts import builtin_atom_registry as registry


SPEC = """# Operations

> **[05-OP-45]** `add` uses the exact operand dtype.
> The [builtin identity registry](registry/builtin_semantic_identities.md)
> is incorporated by reference for this atom's exact identities.
> Signature: `add(left, right)` takes two same-shaped operands. Domain: active arithmetic dtypes.
> Result: operand dtype. Failure: checked integer overflow.
> Adjoint: `(g, g)` on floats. Accumulator: none.
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
        other = SPEC.replace("[05-OP-45]", "[05-OP-46]").replace("`add", "`mul")
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

    def test_semantic_fields_cannot_disappear(self):
        for field in ("Signature", "Domain", "Result", "Failure", "Adjoint", "Accumulator"):
            with self.subTest(field=field), self.assertRaises(registry.RegistryError):
                self.validate(spec=SPEC.replace(field + ":", "omitted:"))


if __name__ == "__main__":
    unittest.main()
