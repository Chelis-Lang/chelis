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
        return registry.validate(discovered, table, spec, {"[05-OP-45]"},
                                 callables=("Numeric:add:TableA",))

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
                              SPEC + "\n" + other, {"[05-OP-45]", "[05-OP-46]"},
                              callables=("Numeric:add:TableA",))

    def test_cross_reference_is_not_definition(self):
        with self.assertRaisesRegex(registry.RegistryError, "definition"):
            self.validate(spec=SPEC.replace("> **[05-OP-45]**", "See [05-OP-45]"))

    def test_missing_generated_membership_fails(self):
        with self.assertRaisesRegex(registry.RegistryError, "generated"):
            registry.validate(("Numeric:add:TableA",), TABLE, SPEC, set(),
                              callables=("Numeric:add:TableA",))

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


    def test_compiled_provenance_not_case_defines_callable_kind(self):
        for operation, public in (("MixedCase", True), ("lowercase", False)):
            identity = f"Numeric:{operation}:TableA"
            table = TABLE.replace("Numeric:add:TableA", identity)
            spec = SPEC.replace("add", operation)
            callables = (identity,) if public else ()
            with self.subTest(operation=operation):
                self.assertEqual(registry.validate(
                    (identity,), table, spec, {"[05-OP-45]"}, callables=callables
                ), {identity: "[05-OP-45]"})
                kind = "callable" if public else "internal primitive"
                with self.assertRaisesRegex(registry.RegistryError, kind):
                    registry.validate((identity,), table, SPEC, {"[05-OP-45]"},
                                      callables=callables)

    def test_callable_discovery_cannot_add_an_identity(self):
        with self.assertRaisesRegex(registry.RegistryError, "outside"):
            registry.validate(("Numeric:add:TableA",), TABLE, SPEC, {"[05-OP-45]"},
                              callables=("Numeric:Other:TableA",))


class SemanticAuthorityTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.spec = (root / "spec/05-risc-primitives.md").read_text()
        self.rows = registry.parse_registry(
            (root / "spec/registry/builtin_semantic_identities.md").read_text()
        )

    def test_every_registered_callable_is_named_by_its_governing_atom(self):
        # Membership of compiled discovery is independently checked by the full
        # oracle; this control checks the real table against the real chapter.
        table = "\n".join(f"| `{name}` | {atom} |" for name, atom in self.rows.items())
        registry.validate(self.rows, table, self.spec, set(registry.atom_blocks(self.spec)),
                          callables=self.rows.keys() - semantics.PRIMITIVE_CLAUSES.keys())

    def test_scan_identities_share_the_ordered_operation_contract(self):
        for identity in ("Numeric:Cumsum:TableA", "Numeric:cumsum:TableA"):
            with self.subTest(identity=identity):
                self.assertEqual(self.rows[identity], "[05-OP-53]")
                for wrong in ("[05-OP-33]", "[05-OP-30]"):
                    with self.assertRaises(registry.RegistryError):
                        semantics.validate_semantics({**self.rows, identity: wrong}, self.spec)

    def test_internal_scan_requires_its_direction_and_domain_contract(self):
        block = registry.atom_blocks(self.spec)["[05-OP-53]"]
        for clause in (
            "`Cumsum` is the internal directional scan identity",
            "Increasing traversal implements `cumsum`",
            "Decreasing traversal admits only float tensors",
            "reverses traversal direction",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, semantics.normalized(block))
                changed = self.spec.replace(block, "> " + semantics.normalized(block).replace(clause, "REMOVED") + "\n")
                with self.assertRaises(registry.RegistryError):
                    semantics.validate_semantics(self.rows, changed)

    def test_ordered_scan_contract_requires_each_decided_semantic_clause(self):
        semantics.validate_semantics(self.rows, self.spec)
        self.assertEqual(self.rows["Numeric:Cumsum:TableA"], "[05-OP-53]")
        block = registry.atom_blocks(self.spec)["[05-OP-53]"]
        normalized = semantics.normalized(block)
        for clause in (
            "`cumsum` admits signed-integer and float tensors",
            "`sum_result(p, default(p))` at the input shape",
            "the first input, converted exactly to §5.7.1's default sum-accumulator dtype, is the first prefix",
            "No base value enters the scan",
            "first prefix keeps the first input's stored value, a `-0` included",
            "Every prefix is finalized to the result dtype for output without narrowing the continuing accumulator",
            "Integer overflow is checked at each addition",
            "A zero-length axis returns an empty tensor",
            "float adjoint is the inclusive decreasing-axis scan of the cotangent at the same accumulator dtype under the same rule",
            "last cotangent is the last suffix",
            "each suffix is emitted at the operand dtype",
            "Integers are forward-only and bool is a type error",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, normalized)
                changed = self.spec.replace(block, "> " + normalized.replace(clause, "REMOVED") + "\n")
                self.assertNotEqual(changed, self.spec)
                with self.assertRaises(registry.RegistryError):
                    semantics.validate_semantics(self.rows, changed)

    def test_internal_primitive_names_do_not_create_public_callable_aliases(self):
        blocks = registry.atom_blocks(self.spec)
        for identity, clause in semantics.PRIMITIVE_CLAUSES.items():
            atom = self.rows[identity]
            block = blocks[atom]
            with self.subTest(identity=identity):
                self.assertIn(identity.split(":")[1], registry.named_operations(block))
                wrong_atom = next(other for other, text in blocks.items()
                                  if other != atom and clause not in semantics.normalized(text))
                changed = self.spec.replace(
                    blocks[wrong_atom],
                    blocks[wrong_atom] + f"> The internal `{identity.split(':')[1]}` identity exists.\n",
                )
                with self.assertRaisesRegex(registry.RegistryError, "govern"):
                    semantics.validate_semantics({**self.rows, identity: wrong_atom}, changed)

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

    def test_integer_remainder_boundary_contract_cannot_drift(self):
        for before, after in (
            ("Integer zero divisors trap DivZero", "Integer zero divisors trap Domain"),
            ("by -1 is zero", "by -1 traps Overflow"),
        ):
            changed = self.spec.replace(before, after)
            self.assertNotEqual(changed, self.spec)
            with self.subTest(clause=before), self.assertRaisesRegex(
                registry.RegistryError, "lost semantic obligation"
            ):
                semantics.validate_semantics(self.rows, changed)

    def test_window_diagnostic_references_do_not_confer_builtin_authority(self):
        semantics.validate_semantics(self.rows, self.spec)
        for operation in ("reduce_window_sum", "reduce_window_mean", "reduce_window_max", "reduce_window_min"):
            identity = next(name for name in self.rows if name.split(":")[1] == operation)
            with self.subTest(operation=operation), self.assertRaisesRegex(registry.RegistryError, "govern"):
                semantics.validate_semantics({**self.rows, identity: "[05-OP-33]"}, self.spec)

    def test_matmul_diagnostic_reference_does_not_confer_builtin_authority(self):
        semantics.validate_semantics(self.rows, self.spec)
        wrong = {**self.rows, "Numeric:matmul:TableA": "[05-OP-33]"}
        with self.assertRaisesRegex(registry.RegistryError, "govern"):
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
