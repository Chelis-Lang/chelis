"""Constructor scope controls for the native boundary's private adapters.

These tests establish lexical ownership only. The final factory must select
the actual governing conversion/validation methods and prove their behavior.
"""

import copy
import unittest

from capacity_census_native_flow import DefinitionIdentity, NativeFlowEvidenceError


def definition(index, name):
    return {
        "stable_crate_id": "fixture",
        "def_id": f"0:{index}",
        "def_path_hash": f"hash:{index}",
        "path": f"::{name}",
        "item_name": name,
        "crate": "fixture",
    }


def body(value, kind="AssocFn", ancestors=()):
    return {
        "definition": value,
        "kind": kind,
        "ancestors": list(ancestors),
        "substitutions": [],
        "open_type_or_const": False,
    }


class ConstructorScopes(unittest.TestCase):
    def setUp(self):
        self.carrier = definition(1, "Validated")
        self.root = definition(2, "validate")
        self.closure = definition(3, "closure")
        self.nested = definition(4, "nested_closure")
        self.root_body = body(self.root)
        self.closure_body = body(self.closure, "Closure", (self.root,))
        self.nested_body = body(self.nested, "Closure", (self.closure, self.root))
        self.raw = {
            "format": 3,
            "scope": "native-bindings",
            "errors": [],
            "bodies": [self.root_body, self.closure_body, self.nested_body],
            "aggregates": [self.aggregate(self.root_body)],
        }

    def aggregate(self, caller):
        return {"definition": self.carrier, "caller": copy.deepcopy(caller)}

    def check(self, raw=None, *, closures=False, roots=None):
        from capacity_census_native_construction import constructor_scope_ownership

        return constructor_scope_ownership(
            self.raw if raw is None else raw,
            DefinitionIdentity.from_record(self.carrier),
            roots or {DefinitionIdentity.from_record(self.root): closures},
        )

    def test_exact_root_and_explicit_closure_scopes(self):
        result = self.check()
        expected = frozenset({DefinitionIdentity.from_record(self.root)})
        self.assertEqual(result.allowed_owners, expected)
        self.assertEqual(result.required_owners, expected)
        self.raw["aggregates"] = [self.aggregate(self.nested_body)]
        result = self.check(closures=True)
        self.assertEqual(result.required_owners,
                         frozenset({DefinitionIdentity.from_record(self.nested)}))
        with self.assertRaisesRegex(NativeFlowEvidenceError, "outside"):
            self.check(closures=False)

    def test_same_named_helper_and_nested_function_are_outside(self):
        for kind, name in (("Fn", "validate"), ("Fn", "nested_function")):
            with self.subTest(kind=kind, name=name):
                raw = copy.deepcopy(self.raw)
                helper = body(definition(8, name), kind, (self.root,))
                raw["bodies"].append(helper)
                raw["aggregates"] = [self.aggregate(helper)]
                with self.assertRaisesRegex(NativeFlowEvidenceError, "outside"):
                    self.check(raw, closures=True)

    def test_closure_parent_chain_must_match_the_actual_body_graph(self):
        self.raw["aggregates"] = [self.aggregate(self.nested_body)]
        self.check(closures=True)
        for mutation in ("missing-parent", "changed-parent-chain", "cycle"):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(self.raw)
                if mutation == "missing-parent":
                    raw["bodies"].remove(raw["bodies"][1])
                elif mutation == "changed-parent-chain":
                    raw["bodies"][1]["ancestors"] = [definition(9, "unrelated")]
                else:
                    raw["bodies"][2]["ancestors"].insert(0, self.nested)
                    raw["aggregates"][0]["caller"] = copy.deepcopy(raw["bodies"][2])
                with self.assertRaises(NativeFlowEvidenceError):
                    self.check(raw, closures=True)

    def test_aggregate_caller_must_match_its_concrete_body(self):
        self.check()
        for mutation in ("missing", "kind", "ancestors", "open", "substitutions"):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(self.raw)
                caller = raw["aggregates"][0]["caller"]
                if mutation == "missing":
                    raw["bodies"] = []
                elif mutation == "kind":
                    caller["kind"] = "Closure"
                elif mutation == "ancestors":
                    caller["ancestors"] = [definition(9, "unrelated")]
                elif mutation == "open":
                    caller["open_type_or_const"] = True
                else:
                    caller["substitutions"] = [{"kind": "type", "type": "i64"}]
                with self.assertRaises(NativeFlowEvidenceError):
                    self.check(raw)

    def test_each_required_root_needs_an_observed_construction(self):
        self.check()
        raw = copy.deepcopy(self.raw)
        raw["aggregates"] = []
        with self.assertRaisesRegex(NativeFlowEvidenceError, "missing"):
            self.check(raw)
        other = definition(9, "other_validator")
        raw["bodies"].append(body(other))
        raw["aggregates"] = [self.aggregate(self.root_body)]
        roots = {DefinitionIdentity.from_record(self.root): False,
                 DefinitionIdentity.from_record(other): False}
        with self.assertRaisesRegex(NativeFlowEvidenceError, "missing"):
            self.check(raw, roots=roots)

    def test_conflicting_body_ancestry_and_open_construction_fail(self):
        self.check()
        for mutation in ("conflict", "open", "errors", "wrong-scope"):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(self.raw)
                if mutation == "conflict":
                    changed = copy.deepcopy(raw["bodies"][0])
                    changed["kind"] = "Fn"
                    raw["bodies"].append(changed)
                elif mutation == "open":
                    raw["bodies"][0]["open_type_or_const"] = True
                    raw["aggregates"][0]["caller"]["open_type_or_const"] = True
                elif mutation == "errors":
                    raw["errors"] = ["unresolved body"]
                else:
                    raw["scope"] = "compiler-json"
                with self.assertRaises(NativeFlowEvidenceError):
                    self.check(raw)
