"""Constructor scope controls for the native boundary's private adapters.

These tests establish direct construction ownership only. The final factory must select
the actual governing conversion/validation methods and prove their behavior.
"""

import copy
import tempfile
import unittest
from pathlib import Path

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
            "format": 4,
            "scope": "native-bindings",
            "errors": [],
            "bodies": [self.root_body, self.closure_body, self.nested_body],
            "aggregates": [self.aggregate(self.root_body)],
            "constructor_uses": [],
        }

    def aggregate(self, caller):
        return {"definition": self.carrier, "caller": copy.deepcopy(caller)}

    def check(self, raw=None, *, roots=None):
        from capacity_census_native_construction import constructor_scope_ownership

        return constructor_scope_ownership(
            self.raw if raw is None else raw,
            DefinitionIdentity.from_record(self.carrier),
            roots if roots is not None else {DefinitionIdentity.from_record(self.root)},
        )

    def test_exact_root_accepts_and_closure_construction_rejects(self):
        result = self.check()
        expected = frozenset({DefinitionIdentity.from_record(self.root)})
        self.assertEqual(result.allowed_owners, expected)
        self.assertEqual(result.required_owners, expected)
        self.raw["aggregates"] = [self.aggregate(self.nested_body)]
        with self.assertRaisesRegex(NativeFlowEvidenceError, "direct governing"):
            self.check()
        with self.assertRaisesRegex(NativeFlowEvidenceError, "policy"):
            self.check(roots={DefinitionIdentity.from_record(self.root): True})

    def test_same_named_helper_and_nested_function_are_outside(self):
        for kind, name in (("Fn", "validate"), ("Fn", "nested_function")):
            with self.subTest(kind=kind, name=name):
                raw = copy.deepcopy(self.raw)
                helper = body(definition(8, name), kind, (self.root,))
                raw["bodies"].append(helper)
                raw["aggregates"] = [self.aggregate(helper)]
                with self.assertRaisesRegex(NativeFlowEvidenceError, "outside"):
                    self.check(raw)

    def test_closure_parent_chain_must_match_the_actual_body_graph(self):
        self.raw["aggregates"] = [self.aggregate(self.nested_body)]
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
                    self.check(raw)

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
        roots = {DefinitionIdentity.from_record(self.root),
                 DefinitionIdentity.from_record(other)}
        with self.assertRaisesRegex(NativeFlowEvidenceError, "missing"):
            self.check(raw, roots=roots)

    def test_conflicting_body_ancestry_and_incomplete_evidence_fail(self):
        self.check()
        for mutation in ("conflict", "duplicate", "errors", "wrong-scope"):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(self.raw)
                if mutation == "conflict":
                    changed = copy.deepcopy(raw["bodies"][0])
                    changed["kind"] = "Fn"
                    raw["bodies"].append(changed)
                elif mutation == "duplicate":
                    raw["bodies"].append(copy.deepcopy(raw["bodies"][0]))
                elif mutation == "errors":
                    raw["errors"] = ["unresolved body"]
                else:
                    raw["scope"] = "compiler-json"
                with self.assertRaises(NativeFlowEvidenceError):
                    self.check(raw)

    def test_generic_templates_share_scope_without_proving_instantiation(self):
        self.check()
        self.raw["bodies"][0]["open_type_or_const"] = True
        self.raw["aggregates"][0]["caller"]["open_type_or_const"] = True
        self.check()
        # A caller still cannot assert a concrete instance absent from the
        # actual compiler body census. Lexical equality is not type authority.
        self.raw["aggregates"][0]["caller"]["open_type_or_const"] = False
        with self.assertRaisesRegex(NativeFlowEvidenceError, "actual body"):
            self.check()

    def test_constructor_use_census_cannot_be_omitted_or_downgraded(self):
        self.check()
        for mutation in ("missing", "not-list", "old-format"):
            raw = copy.deepcopy(self.raw)
            if mutation == "missing":
                del raw["constructor_uses"]
            elif mutation == "not-list":
                raw["constructor_uses"] = {}
            else:
                raw["format"] = 3
            with self.subTest(mutation=mutation), self.assertRaises(NativeFlowEvidenceError):
                self.check(raw)

    def test_empty_and_nonfunction_owner_policies_reject(self):
        with self.assertRaisesRegex(NativeFlowEvidenceError, "policy"):
            self.check(roots=set())
        self.raw["bodies"][0]["kind"] = "Closure"
        self.raw["aggregates"][0]["caller"]["kind"] = "Closure"
        with self.assertRaisesRegex(NativeFlowEvidenceError, "governing function"):
            self.check()


class CompiledConstructorScopes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_calls import build_driver

        cls.root = Path(__file__).resolve().parent.parent
        cls.scratch = tempfile.TemporaryDirectory(prefix="native-construction-",
                                                 dir=cls.root / "target")
        cls.driver = build_driver(
            cls.root, cls.root / "target/agents/native-bindings-driver/native-calls-tool"
        )

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def observe(self, source):
        from capacity_census_wire_calls import analyze_fixture

        return analyze_fixture(self.driver, Path(self.scratch.name), source, {},
                               scope="native-bindings").raw

    def policy(self, raw):
        # Select the actual trait implementation rather than an impl ordinal,
        # a helper's name, or the closure's display path.
        methods = [item for item in raw["bodies"]
                   if item["kind"] == "AssocFn"
                   and item["definition"]["item_name"] == "convert"
                   and item["implementation"]["trait"]["path"] == "::Convert"
                   and item["implementation"]["self_type"]["nominal"]["path"]
                   == "::Validated"]
        self.assertEqual(len(methods), 1)
        method = methods[0]
        return (
            DefinitionIdentity.from_record(method["implementation"]["self_type"]["nominal"]),
            {DefinitionIdentity.from_record(method["definition"])},
        )

    def test_actual_direct_conversion_and_unrelated_constructor(self):
        from capacity_census_native_construction import constructor_scope_ownership

        source = """
struct Validated(i64);
trait Convert { fn convert(value: i64) -> Self; }
impl Convert for Validated {
    fn convert(value: i64) -> Self {
        Validated(value)
    }
}
pub fn entry(value: i64) -> i64 { Validated::convert(value).0 }
"""
        raw = self.observe(source)
        result = constructor_scope_ownership(raw, *self.policy(raw))
        self.assertTrue(result.required_owners)
        kinds = {DefinitionIdentity.from_record(item["definition"]): item["kind"]
                 for item in raw["bodies"]}
        self.assertEqual({kinds[owner] for owner in result.required_owners}, {"AssocFn"})
        raw = self.observe(source + "fn convert(value: i64) -> Validated { Validated(value) }")
        with self.assertRaisesRegex(NativeFlowEvidenceError, "outside"):
            constructor_scope_ownership(raw, *self.policy(raw))

    def test_actual_raw_constructor_values_cannot_hide_beside_valid_aggregate(self):
        from capacity_census_native_construction import constructor_scope_ownership

        base = """
struct Validated(i64);
trait Convert { fn convert(value: i64) -> Self; }
impl Convert for Validated {
    fn convert(value: i64) -> Self { Validated(value) }
}
pub fn entry(value: i64) -> i64 { Validated::convert(value).0 }
"""
        raw = self.observe(base)
        self.assertTrue(constructor_scope_ownership(raw, *self.policy(raw)).required_owners)
        for extra in (
            "fn callback(v: Vec<i64>) -> Vec<Validated> { v.into_iter().map(Validated).collect() }",
            "fn callback() -> fn(i64) -> Validated { Validated }",
            "static CALLBACK: fn(i64) -> Validated = Validated;",
        ):
            with self.subTest(extra=extra):
                raw = self.observe(base + extra)
                self.assertTrue(raw["constructor_uses"])
                with self.assertRaisesRegex(NativeFlowEvidenceError, "constructor function value"):
                    constructor_scope_ownership(raw, *self.policy(raw))
        # Even the exact checked owner's body may not expose its raw constructor.
        inside = base.replace("Validated(value)",
                              "let _ctor: fn(i64) -> Self = Self; Validated(value)")
        raw = self.observe(inside)
        with self.assertRaisesRegex(NativeFlowEvidenceError, "constructor function value"):
            constructor_scope_ownership(raw, *self.policy(raw))

    def test_unrelated_constructor_is_observed_without_authority_and_mutations_reject(self):
        from capacity_census_native_construction import constructor_scope_ownership

        raw = self.observe("""
struct Validated(i64);
trait Convert { fn convert(value: i64) -> Self; }
impl Convert for Validated {
    fn convert(value: i64) -> Self { Validated(value) }
}
struct Other(i64);
fn callback() -> fn(i64) -> Other { Other }
pub fn entry(value: i64) -> i64 { Validated::convert(value).0 }
""")
        policy = self.policy(raw)
        constructor_scope_ownership(raw, *policy)
        self.assertTrue(raw["constructor_uses"])
        for mutation in ("duplicate", "caller", "carrier", "signature", "fields",
                         "position", "kind", "missing-field"):
            changed = copy.deepcopy(raw)
            row = changed["constructor_uses"][0]
            if mutation == "duplicate":
                changed["constructor_uses"].append(copy.deepcopy(row))
            elif mutation == "caller":
                row["caller"]["open_type_or_const"] = not row["caller"]["open_type_or_const"]
            elif mutation == "carrier":
                row["carrier"]["def_path_hash"] = "unrelated-carrier"
            elif mutation == "signature":
                row["formal_inputs"] = []
            elif mutation == "fields":
                row["fields"][0]["type"]["shape"] = {"tag": "primitive", "name": "f64"}
            elif mutation == "position":
                row["operand_index"] = -1
            elif mutation == "kind":
                row["kind"] = "Fn"
            else:
                del row["cast"]
            with self.subTest(mutation=mutation), self.assertRaises(NativeFlowEvidenceError):
                constructor_scope_ownership(changed, *policy)

    def test_actual_local_and_foreign_closure_construction_both_reject(self):
        from capacity_census_native_construction import constructor_scope_ownership

        for expression in ("(|| Validated(value))()",
                           "[value].into_iter().map(|v| Validated(v)).next().unwrap()"):
            with self.subTest(expression=expression):
                raw = self.observe("""
struct Validated(i64);
trait Convert { fn convert(value: i64) -> Self; }
impl Convert for Validated {
    fn convert(value: i64) -> Self { EXPR }
}
pub fn entry(value: i64) -> i64 { Validated::convert(value).0 }
""".replace("EXPR", expression))
                with self.assertRaisesRegex(NativeFlowEvidenceError, "direct governing"):
                    constructor_scope_ownership(raw, *self.policy(raw))

    def test_actual_nested_function_does_not_inherit_conversion_authority(self):
        from capacity_census_native_construction import constructor_scope_ownership

        raw = self.observe("""
struct Validated(i64);
trait Convert { fn convert(value: i64) -> Self; }
impl Convert for Validated {
    fn convert(value: i64) -> Self {
        fn helper(value: i64) -> Validated { Validated(value) }
        helper(value)
    }
}
pub fn entry(value: i64) -> i64 { Validated::convert(value).0 }
""")
        with self.assertRaisesRegex(NativeFlowEvidenceError, "outside"):
            constructor_scope_ownership(raw, *self.policy(raw))
