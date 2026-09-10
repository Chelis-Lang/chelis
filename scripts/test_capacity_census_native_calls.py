"""Compiler-derived collection controls for the native Python boundary.

These fixtures exercise the collector only.  They deliberately make no final
authority decision about a wrapper, constructor, or call path.
"""

import copy
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def named(record, name):
    return record["definition"].get("item_name") == name


def calls_named(evidence, name):
    return [
        call
        for call in evidence.raw["calls"]
        if call["callee"] is not None
        and call["callee"]["definition"].get("item_name") == name
    ]


def aggregates_named(evidence, name):
    return [
        aggregate
        for aggregate in evidence.raw["aggregates"]
        if aggregate["definition"].get("item_name") == name
    ]


def constructor_uses_named(evidence, name):
    return [
        use
        for use in evidence.raw["constructor_uses"]
        if use["carrier"].get("item_name") == name
    ]


class NativeEvidenceShapeControls(unittest.TestCase):
    def fixture(self):
        from capacity_census_wire_calls import COMPILER

        definition = {
            "def_id": "0:0",
            "crate": "native_fixture",
            "item_name": "native_fixture",
            "stable_crate_id": "0" * 16,
            "path": "::native_fixture",
            "def_path_hash": "DefPathHash(Fingerprint(0, 0))",
        }
        return {
            "format": 4,
            "scope": "native-bindings",
            "compiler": COMPILER,
            "crate": definition,
            "bodies": [],
            "calls": [],
            "aggregates": [],
            "constructor_uses": [],
            "flows": [],
            "errors": [],
            "rustc_command": [],
            "inputs": [],
        }

    def test_native_shape_is_distinct_and_exact(self):
        from capacity_census_wire_calls import read_evidence

        raw = self.fixture()
        evidence = read_evidence(raw)
        self.assertEqual(evidence.raw["scope"], "native-bindings")
        for mutation in ("scope", "format", "missing", "extra", "wrong-list"):
            changed = copy.deepcopy(raw)
            if mutation == "scope":
                changed["scope"] = "compiler-json"
            elif mutation == "format":
                changed["format"] = 2
            elif mutation == "missing":
                del changed["flows"]
            elif mutation == "extra":
                changed["authority"] = "TaggedTransport"
            else:
                changed["aggregates"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                read_evidence(changed)

    def test_constructor_census_is_mandatory_and_old_native_format_is_rejected(self):
        from capacity_census_wire_calls import read_evidence

        raw = self.fixture()
        self.assertEqual(read_evidence(raw).raw["constructor_uses"], [])
        for mutation in ("missing", "wrong-list", "old-format", "old-without-census"):
            changed = copy.deepcopy(raw)
            if mutation in {"old-format", "old-without-census"}:
                changed["format"] = 3
            if mutation in {"missing", "old-without-census"}:
                del changed["constructor_uses"]
            elif mutation == "wrong-list":
                changed["constructor_uses"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                read_evidence(changed)

    def test_native_collector_source_identity_is_complete_and_current(self):
        import hashlib

        from capacity_census_wire_calls import (
            NATIVE_COLLECTOR_SOURCES,
            _native_collector_inputs,
        )

        self.assertEqual(
            set(NATIVE_COLLECTOR_SOURCES),
            {
                "crates/chelis-compiler-api/examples/wire_calls/driver.rs",
                "scripts/capacity_census_wire_calls.py",
                "scripts/test_capacity_census_native_calls.py",
            },
        )
        inputs = _native_collector_inputs(ROOT)
        self.assertEqual(len(inputs), len(NATIVE_COLLECTOR_SOURCES))
        for item in inputs:
            path = Path(item["path"])
            self.assertEqual(
                item["sha256"],
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )


class NativeCompilerCollectionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_calls import build_driver

        cls.scratch = tempfile.TemporaryDirectory(
            prefix="native-calls-", dir=ROOT / "target"
        )
        cls.directory = Path(cls.scratch.name)
        cls.driver = build_driver(
            ROOT, ROOT / "target/agents/native-bindings-driver/native-calls-tool"
        )

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def observe(self, source):
        from capacity_census_wire_calls import analyze_fixture

        return analyze_fixture(
            self.driver,
            self.directory,
            source,
            {},
            scope="native-bindings",
        )

    def test_direct_aggregate_and_foreign_constructor_callback_are_distinct(self):
        # spec/11 §1.2 requires validation before wrapper construction. A raw
        # constructor exposed to a callback is an obligation, not validation.
        evidence = self.observe("""
pub struct Tensor(i64);
pub fn direct(value: i64) -> Tensor { Tensor(value) }
pub fn callback(values: Vec<i64>) -> Vec<Tensor> {
    values.into_iter().map(Tensor).collect()
}
""")
        self.assertEqual(evidence.raw["errors"], [])
        self.assertEqual(
            {row["caller"]["definition"]["item_name"]
             for row in aggregates_named(evidence, "Tensor")},
            {"direct"},
        )
        uses = constructor_uses_named(evidence, "Tensor")
        self.assertEqual(len(uses), 1)
        use = uses[0]
        self.assertEqual(use["caller"]["definition"]["item_name"], "callback")
        self.assertEqual(use["kind"], "Ctor(Struct, Fn)")
        self.assertNotEqual(use["definition"], use["carrier"])
        self.assertEqual(use["variant_definition"], use["carrier"])
        self.assertEqual(use["formal_result"]["nominal"], use["carrier"])
        self.assertEqual(use["formal_inputs"][0]["shape"],
                         {"tag": "primitive", "name": "i64"})
        self.assertEqual(use["fields"][0]["definition"]["item_name"], "0")
        self.assertEqual(use["fields"][0]["type"], use["formal_inputs"][0])
        self.assertEqual(use["context"], "terminator")
        self.assertIsNone(use["cast"])

    def test_tuple_enum_constructor_values_keep_exact_variant_fields(self):
        evidence = self.observe("""
pub enum Payload<T, const N: usize> { Left(T, [i64; N]), Right(T) }
pub fn left() -> fn(i64, [i64; 2]) -> Payload<i64, 2> {
    Payload::<i64, 2>::Left
}
pub fn right() -> fn(i64) -> Payload<i64, 2> { Payload::<i64, 2>::Right }
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = {row["caller"]["definition"]["item_name"]: row
                for row in constructor_uses_named(evidence, "Payload")}
        self.assertEqual(set(uses), {"left", "right"})
        left, right = uses["left"], uses["right"]
        self.assertEqual(left["kind"], "Ctor(Variant, Fn)")
        self.assertEqual(left["carrier"], right["carrier"])
        self.assertNotEqual(left["definition"], right["definition"])
        self.assertNotEqual(left["variant_definition"], right["variant_definition"])
        self.assertEqual(left["variant_definition"]["item_name"], "Left")
        self.assertEqual(right["variant_definition"]["item_name"], "Right")
        self.assertEqual([field["definition"]["item_name"] for field in left["fields"]],
                         ["0", "1"])
        self.assertEqual(len(right["fields"]), 1)
        self.assertNotEqual(left["fields"][0]["definition"],
                            right["fields"][0]["definition"])
        self.assertEqual([field["type"] for field in left["fields"]],
                         left["formal_inputs"])
        self.assertEqual(left["formal_inputs"][1]["shape"]["length"], 2)
        self.assertEqual([arg["kind"] for arg in left["arguments"]], ["type", "const"])

    def test_stored_returned_and_reified_constructor_values_are_collected(self):
        evidence = self.observe("""
pub struct Tensor(i64);
pub fn local(value: i64) -> Tensor {
    let constructor = Tensor;
    let copied = constructor;
    let pointer: fn(i64) -> Tensor = copied;
    pointer(value)
}
pub fn returned() -> impl Fn(i64) -> Tensor { Tensor }
pub fn stored() -> Vec<fn(i64) -> Tensor> { vec![Tensor] }
pub fn foreign_pointer(pointer: fn(i64) -> Tensor) -> fn(i64) -> Tensor { pointer }
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = constructor_uses_named(evidence, "Tensor")
        self.assertEqual({row["caller"]["definition"]["item_name"] for row in uses},
                         {"local", "returned", "stored"})
        local = [row for row in uses if row["caller"]["definition"]["item_name"] == "local"]
        self.assertTrue(any(row["operand"]["kind"] == "constant" for row in local))
        self.assertTrue(any(row["operand"]["kind"] in {"copy", "move"} for row in local))
        reified = [row for row in uses if row["cast"] is not None]
        self.assertTrue(reified)
        for row in reified:
            self.assertIn("ReifyFnPointer", row["cast"]["kind"])
            self.assertIn("fn(", row["cast"]["target"]["text"])
            self.assertEqual(row["context"], "statement")

    def test_promoted_constructor_values_are_collected(self):
        evidence = self.observe("""
pub struct Tensor(i64);
pub fn table() -> &'static [fn(i64) -> Tensor] { &[Tensor] }
pub fn borrowed() -> &'static fn(i64) -> Tensor {
    &(Tensor as fn(i64) -> Tensor)
}
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = constructor_uses_named(evidence, "Tensor")
        self.assertEqual(
            {row["caller"]["definition"]["item_name"] for row in uses},
            {"table", "borrowed"},
        )
        self.assertEqual(len(uses), 2)
        self.assertTrue(all(row["promoted"] is not None for row in uses))

    def test_constructor_occurrence_rows_have_an_exact_fail_closed_shape(self):
        from capacity_census_wire_calls import read_evidence

        evidence = self.observe("""
pub struct Tensor(i64);
pub fn callback() -> fn(i64) -> Tensor { Tensor }
""")
        self.assertEqual(evidence.raw["errors"], [])
        row = constructor_uses_named(evidence, "Tensor")[0]
        for key in row:
            changed = copy.deepcopy(evidence.raw)
            del changed["constructor_uses"][0][key]
            with self.subTest(mutation=f"missing-{key}"), self.assertRaises(ValueError):
                read_evidence(changed)
        for mutation in ("extra", "wrong-fields", "wrong-promoted", "wrong-cast"):
            changed = copy.deepcopy(evidence.raw)
            occurrence = changed["constructor_uses"][0]
            if mutation == "extra":
                occurrence["authority"] = "TaggedTransport"
            elif mutation == "wrong-fields":
                occurrence["fields"] = {}
            elif mutation == "wrong-promoted":
                occurrence["promoted"] = "0"
            else:
                occurrence["cast"] = {}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                read_evidence(changed)

    def test_const_and_static_constructor_pointer_origins_are_collected(self):
        evidence = self.observe("""
pub struct Tensor(i64);
pub const SAVED: fn(i64) -> Tensor = Tensor;
pub static STATIC: fn(i64) -> Tensor = Tensor;
impl Tensor { pub const CONSTRUCTOR: fn(i64) -> Self = Self; }
pub fn live(value: i64) -> Tensor { SAVED(value) }
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = constructor_uses_named(evidence, "Tensor")
        owners = {row["caller"]["definition"]["item_name"] for row in uses}
        self.assertTrue({"SAVED", "STATIC", "CONSTRUCTOR"} <= owners)
        for row in uses:
            if row["caller"]["definition"]["item_name"] in {"SAVED", "STATIC", "CONSTRUCTOR"}:
                self.assertIsNotNone(row["cast"])
                self.assertEqual(row["formal_result"]["nominal"], row["carrier"])

    def test_repeated_constructor_operands_keep_distinct_location_identities(self):
        evidence = self.observe("""
pub struct Tensor(i64);
fn sink<A, B>(_: A, _: B) {}
pub fn repeated() { sink(Tensor, Tensor) }
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = [row for row in constructor_uses_named(evidence, "Tensor")
                if row["caller"]["definition"]["item_name"] == "repeated"]
        self.assertEqual(len(uses), 2)
        first, second = uses
        self.assertEqual(first["definition"], second["definition"])
        self.assertEqual((first["block"], first["statement"]),
                         (second["block"], second["statement"]))
        self.assertNotEqual(first["operand_index"], second["operand_index"])
        self.assertTrue(all(row["context"] == "terminator" for row in uses))
        for row in uses:
            body = next(body for body in evidence.raw["bodies"]
                        if body["definition"] == row["caller"]["definition"])
            self.assertEqual(body["substitutions"], row["caller"]["substitutions"])
            self.assertEqual(body["ancestors"], row["caller"]["ancestors"])
            block = next(block for block in body["blocks"] if block["block"] == row["block"])
            self.assertEqual(row["statement"], block["statements"])

    def test_generic_constructor_values_retain_type_and_const_substitutions(self):
        evidence = self.observe("""
pub struct Capsule<T, const N: usize>(T, [i64; N]);
fn constructor<T, const N: usize>() -> fn(T, [i64; N]) -> Capsule<T, N> {
    Capsule::<T, N>
}
pub fn integer() -> fn(i64, [i64; 2]) -> Capsule<i64, 2> { constructor::<i64, 2>() }
pub fn floating() -> fn(f64, [i64; 3]) -> Capsule<f64, 3> { constructor::<f64, 3>() }
""")
        self.assertEqual(evidence.raw["errors"], [])
        uses = constructor_uses_named(evidence, "Capsule")
        self.assertTrue(any(row["caller"]["open_type_or_const"] for row in uses))
        closed = [row for row in uses if not row["caller"]["open_type_or_const"]]
        self.assertEqual(
            {(row["formal_inputs"][0]["shape"]["name"],
              row["formal_inputs"][1]["shape"]["length"]) for row in closed},
            {("i64", 2), ("f64", 3)},
        )
        for row in closed:
            self.assertEqual([arg["kind"] for arg in row["arguments"]], ["type", "const"])
            self.assertEqual([field["type"] for field in row["fields"]], row["formal_inputs"])
            self.assertFalse(row["formal_result"]["open"])

    def test_same_named_function_and_bare_pointer_are_not_raw_constructors(self):
        evidence = self.observe("""
pub struct Tensor(i64);
mod decoy { pub fn Tensor(value: i64) -> super::Tensor { super::Tensor(value) } }
impl Tensor { pub fn new(value: i64) -> Self { Self(value) } }
pub fn actual() -> fn(i64) -> Tensor { Tensor }
pub fn named() -> fn(i64) -> Tensor { decoy::Tensor }
pub fn method() -> fn(i64) -> Tensor { Tensor::new }
pub fn pointer(value: fn(i64) -> Tensor) -> fn(i64) -> Tensor { value }
""")
        self.assertEqual(evidence.raw["errors"], [])
        self.assertEqual(
            {row["caller"]["definition"]["item_name"]
             for row in constructor_uses_named(evidence, "Tensor")},
            {"actual"},
        )

    def test_unit_values_are_not_reported_as_callable_constructor_capabilities(self):
        # This census concerns FnDef constructor exposure. Unit/constant values
        # and arbitrary unsafe construction need their own consumer obligations.
        evidence = self.observe("""
pub struct Unit;
pub enum Value { Empty, Data(i64) }
pub fn unit() -> Unit { Unit }
pub fn empty() -> Value { Value::Empty }
pub fn data() -> fn(i64) -> Value { Value::Data }
""")
        self.assertEqual(evidence.raw["errors"], [])
        self.assertEqual(constructor_uses_named(evidence, "Unit"), [])
        self.assertEqual(
            {row["variant_definition"]["item_name"]
             for row in constructor_uses_named(evidence, "Value")},
            {"Data"},
        )

    def test_compiler_records_exact_impl_owner_and_ignores_same_named_helper(self):
        evidence = self.observe(
            """
struct ValidatedTensor(Vec<i64>);
trait Export { fn export(self) -> Vec<i64>; }
impl Export for ValidatedTensor {
    fn export(self) -> Vec<i64> { self.0 }
}
fn export(_: ValidatedTensor) -> Vec<f64> { Vec::new() }
pub fn live(value: ValidatedTensor) -> Vec<i64> { Export::export(value) }
"""
        )
        implementations = [
            body["implementation"]
            for body in evidence.raw["bodies"]
            if named(body, "export")
        ]
        self.assertEqual(len(implementations), 2)
        self.assertEqual(
            sum(impl is not None and impl["trait"] is not None for impl in implementations),
            1,
        )
        call = calls_named(evidence, "export")
        self.assertEqual(len(call), 1)
        self.assertEqual(call[0]["callee"]["definition"]["path"], "::Export::export")
        self.assertIn(
            "impl",
            call[0]["callee"]["resolved"]["definition"]["path"],
        )
        self.assertEqual(evidence.raw["format"], 4)
        self.assertEqual(evidence.raw["scope"], "native-bindings")
        self.assertIn("--crate-name=native_fixture", evidence.raw["rustc_command"])
        self.assertEqual(len(evidence.raw["inputs"]), 1)
        self.assertTrue(evidence.raw["inputs"][0]["hash"].startswith("sha256="))

    def test_direction_and_live_call_path_are_observable(self):
        forward = self.observe(
            """
struct CompiledInputs(Vec<i64>); struct CompiledTensorResults(Vec<i64>);
fn execute_checked(_: &CompiledInputs) -> CompiledTensorResults {
    CompiledTensorResults(Vec::new())
}
pub fn live(raw: Vec<i64>) -> CompiledTensorResults {
    let inputs = CompiledInputs(raw); execute_checked(&inputs)
}
fn unused(_: &CompiledInputs) -> CompiledTensorResults { CompiledTensorResults(Vec::new()) }
"""
        )
        call = calls_named(forward, "execute_checked")
        self.assertEqual(len(call), 1)
        self.assertEqual(call[0]["caller"]["definition"]["item_name"], "live")
        self.assertEqual(call[0]["arguments"][0]["type"]["shape"]["tag"], "reference")
        reverse = self.observe(
            """
struct CompiledInputs(Vec<i64>); struct CompiledTensorResults(Vec<i64>);
fn execute_checked(_: &CompiledTensorResults) -> CompiledInputs {
    CompiledInputs(Vec::new())
}
pub fn live(results: CompiledTensorResults) -> CompiledInputs {
    execute_checked(&results)
}
"""
        )
        reversed_call = calls_named(reverse, "execute_checked")[0]
        self.assertNotEqual(call[0]["formal_inputs"], reversed_call["formal_inputs"])
        self.assertNotEqual(call[0]["formal_result"], reversed_call["formal_result"])
        missing = self.observe(
            """
struct CompiledInputs(Vec<i64>); struct CompiledTensorResults(Vec<i64>);
fn execute_checked(_: &CompiledInputs) -> CompiledTensorResults {
    CompiledTensorResults(Vec::new())
}
pub fn live(_: CompiledInputs) -> CompiledTensorResults {
    CompiledTensorResults(Vec::new())
}
"""
        )
        self.assertEqual(calls_named(missing, "execute_checked"), [])

    def test_indirect_calls_keep_formal_types_without_inventing_a_callee(self):
        evidence = self.observe(
            """
pub struct Tensor(i64);
fn export(value: Tensor) -> i64 { value.0 }
pub fn direct(value: Tensor) -> i64 { export(value) }
pub fn indirect(value: Tensor, export: fn(Tensor) -> i64) -> i64 { export(value) }
"""
        )
        self.assertEqual(evidence.raw["errors"], [])
        calls = {
            call["caller"]["definition"]["item_name"]: call
            for call in evidence.raw["calls"]
        }
        direct, indirect = calls["direct"], calls["indirect"]
        self.assertIsNotNone(direct["callee"])
        self.assertEqual(indirect["kind"], "indirect")
        self.assertIsNone(indirect["callee"])
        self.assertEqual(indirect["formal_inputs"], direct["formal_inputs"])
        self.assertEqual(indirect["formal_result"], direct["formal_result"])
        self.assertEqual(indirect["formal_inputs"][0]["nominal"]["item_name"], "Tensor")
        self.assertEqual(
            indirect["formal_result"]["shape"], {"tag": "primitive", "name": "i64"}
        )

    def test_indirect_call_formals_distinguish_reversed_numeric_channels(self):
        evidence = self.observe(
            """
pub struct Tensor(i64);
pub fn export(value: Tensor, call: fn(Tensor) -> i64) -> i64 { call(value) }
pub fn import(value: i64, call: fn(i64) -> Tensor) -> Tensor { call(value) }
"""
        )
        self.assertEqual(evidence.raw["errors"], [])
        calls = {
            call["caller"]["definition"]["item_name"]: call
            for call in evidence.raw["calls"]
        }
        export, imported = calls["export"], calls["import"]
        self.assertIsNone(export["callee"])
        self.assertIsNone(imported["callee"])
        self.assertNotEqual(export["formal_inputs"], imported["formal_inputs"])
        self.assertNotEqual(export["formal_result"], imported["formal_result"])
        self.assertEqual(
            imported["formal_inputs"][0]["shape"], {"tag": "primitive", "name": "i64"}
        )
        self.assertEqual(imported["formal_result"]["nominal"]["item_name"], "Tensor")

    def test_conversion_closures_retain_exact_lexical_owners(self):
        evidence = self.observe(
            """
struct Tensor(i64); struct Results(Vec<i64>);
trait Export { fn export(self) -> Vec<Tensor>; }
impl Export for Results {
    fn export(self) -> Vec<Tensor> {
        self.0.into_iter().map(|value| Tensor(value)).collect()
    }
}
fn export(values: Vec<i64>) -> Vec<Tensor> {
    values.into_iter().map(|value| Tensor(value)).collect()
}
pub fn live(values: Vec<i64>) -> Vec<Tensor> { Results(values).export() }
"""
        )
        methods = [body for body in evidence.raw["bodies"] if named(body, "export")]
        owned = next(body for body in methods if body["implementation"] is not None)
        foreign = next(body for body in methods if body["implementation"] is None)
        constructors = aggregates_named(evidence, "Tensor")
        self.assertEqual(len(constructors), 2)
        parents = []
        for row in constructors:
            caller = row["caller"]
            self.assertEqual(caller["kind"], "Closure")
            self.assertIsNone(caller["implementation"])
            self.assertGreaterEqual(len(caller["ancestors"]), 2)
            parents.append(caller["ancestors"][0])
            body = next(body for body in evidence.raw["bodies"]
                        if body["definition"] == caller["definition"])
            self.assertEqual(body["kind"], caller["kind"])
            self.assertEqual(body["ancestors"], caller["ancestors"])
        self.assertCountEqual(parents, [owned["definition"], foreign["definition"]])
        self.assertNotEqual(owned["definition"], foreign["definition"])
        self.assertEqual(owned["kind"], "AssocFn")
        self.assertEqual(foreign["kind"], "Fn")

    def test_every_local_aggregate_constructor_and_relocation_is_visible(self):
        baseline = self.observe(
            """
struct ValidatedTensor(Vec<i64>);
impl ValidatedTensor { fn admit(v: Vec<i64>) -> Self { Self(v) } }
pub fn live(v: Vec<i64>) -> ValidatedTensor { ValidatedTensor::admit(v) }
"""
        )
        self.assertEqual(
            {row["caller"]["definition"]["item_name"] for row in aggregates_named(baseline, "ValidatedTensor")},
            {"admit"},
        )
        bypass = self.observe(
            """
struct ValidatedTensor(Vec<i64>);
impl ValidatedTensor { fn admit(v: Vec<i64>) -> Self { Self(v) } }
fn relocated(v: Vec<i64>) -> ValidatedTensor { ValidatedTensor(v) }
pub fn live(v: Vec<i64>) -> ValidatedTensor { ValidatedTensor::admit(v) }
"""
        )
        self.assertEqual(
            {row["caller"]["definition"]["item_name"] for row in aggregates_named(bypass, "ValidatedTensor")},
            {"admit", "relocated"},
        )

    def test_union_constructor_records_only_the_compiler_selected_field(self):
        evidence = self.observe(
            """
union NativeValue { integer: i64, floating: f64 }
pub fn integer(value: i64) -> NativeValue { NativeValue { integer: value } }
pub fn floating(value: f64) -> NativeValue { NativeValue { floating: value } }
"""
        )
        rows = aggregates_named(evidence, "NativeValue")
        self.assertEqual(len(rows), 2)
        selected = {
            row["caller"]["definition"]["item_name"]: [
                field["definition"]["item_name"] for field in row["fields"]
            ]
            for row in rows
        }
        self.assertEqual(
            selected,
            {"integer": ["integer"], "floating": ["floating"]},
        )
        self.assertEqual(evidence.raw["errors"], [])

    def test_aggregate_variant_identity_and_arguments_are_compiler_exact(self):
        evidence = self.observe(
            """
struct NativeValue(Vec<i64>);
enum Local<T> { Ok(T), Err }
pub fn core_ok(value: NativeValue) -> Result<NativeValue, &'static str> { Ok(value) }
pub fn core_err() -> Result<NativeValue, &'static str> { Err("bad") }
pub fn local_ok(value: NativeValue) -> Local<NativeValue> { Local::Ok(value) }
pub fn local_err() -> Local<NativeValue> { Local::Err }
"""
        )
        rows = [
            row
            for row in evidence.raw["aggregates"]
            if row["variant"] in {"Ok", "Err"}
        ]
        core = [row for row in rows if row["variant_definition"]["crate"] == "core"]
        local = [
            row for row in rows if row["variant_definition"]["crate"] == "native_fixture"
        ]
        self.assertEqual({row["variant"] for row in core}, {"Ok", "Err"})
        self.assertEqual({row["variant"] for row in local}, {"Ok", "Err"})
        self.assertTrue(
            all(
                {
                    "stable_crate_id",
                    "def_id",
                    "def_path_hash",
                }
                <= row["variant_definition"].keys()
                for row in rows
            )
        )
        self.assertNotEqual(
            next(row for row in core if row["variant"] == "Ok")["variant_definition"],
            next(row for row in local if row["variant"] == "Ok")["variant_definition"],
        )
        self.assertIn("NativeValue", json.dumps(core))

    def test_payload_alias_generic_nesting_and_lifetime_only_openness_are_recorded(self):
        integer = self.observe(
            """
type Extents = Vec<i64>;
struct ValidatedTensor<'a> { extents: &'a Extents }
impl<'a> ValidatedTensor<'a> { fn admit(extents: &'a Extents) -> Self { Self { extents } } }
pub fn live(extents: &Vec<i64>) -> ValidatedTensor<'_> { ValidatedTensor::admit(extents) }
"""
        )
        admit = next(body for body in integer.raw["bodies"] if named(body, "admit"))
        self.assertFalse(admit["open_type_or_const"])
        self.assertIn("i64", json.dumps(admit["formal_inputs"]))
        floating = self.observe(
            """
type Extents = Vec<f64>;
struct ValidatedTensor<'a> { extents: &'a Extents }
impl<'a> ValidatedTensor<'a> { fn admit(extents: &'a Extents) -> Self { Self { extents } } }
pub fn live(extents: &Vec<f64>) -> ValidatedTensor<'_> { ValidatedTensor::admit(extents) }
"""
        )
        other = next(body for body in floating.raw["bodies"] if named(body, "admit"))
        self.assertIn("f64", json.dumps(other["formal_inputs"]))
        self.assertNotEqual(admit["formal_inputs"], other["formal_inputs"])

    def test_type_generic_body_is_open_but_concrete_instance_is_retained(self):
        evidence = self.observe(
            """
struct Capsule<T>(T);
fn adopt<T>(value: T) -> Capsule<T> { Capsule(value) }
pub fn live(value: Vec<i64>) -> Capsule<Vec<i64>> { adopt(value) }
"""
        )
        bodies = [body for body in evidence.raw["bodies"] if named(body, "adopt")]
        self.assertTrue(any(body["open_type_or_const"] for body in bodies))
        self.assertTrue(any(not body["open_type_or_const"] for body in bodies))
        self.assertIn("i64", json.dumps(bodies))
        const = self.observe(
            """
struct Capsule<const N: usize>([i64; N]);
fn adopt<const N: usize>(value: [i64; N]) -> Capsule<N> { Capsule(value) }
pub fn live(value: [i64; 2]) -> Capsule<2> { adopt(value) }
"""
        )
        const_bodies = [body for body in const.raw["bodies"] if named(body, "adopt")]
        self.assertTrue(any(body["open_type_or_const"] for body in const_bodies))
        self.assertTrue(any(not body["open_type_or_const"] for body in const_bodies))
        closed = next(body for body in const_bodies if not body["open_type_or_const"])
        self.assertEqual(closed["formal_inputs"][0]["type"]["shape"]["length"], 2)
        self.assertEqual(closed["substitutions"][0]["kind"], "const")

    def test_impl_const_arguments_do_not_inherit_the_impls_generic_parameters(self):
        # PyO3 emits precisely this shape for PyFunctionArgument<'a, 'py, false>.
        # The anonymous bool body belongs syntactically to the generic impl,
        # but is not its associated method and has no inherited substitutions.
        for flag in ("false", "true"):
            evidence = self.observe("""
trait Argument<'a, const FLAG: bool> { fn get(self) -> i64; }
struct Model<T>(T);
impl<'a, T> Argument<'a, FLAG_VALUE> for &'a Model<T> {
    fn get(self) -> i64 { 7 }
}
pub fn live(value: &Model<i64>) -> i64 { Argument::get(value) }
""".replace("FLAG_VALUE", flag))
            self.assertEqual(evidence.raw["errors"], [])
            constants = [body for body in evidence.raw["bodies"]
                         if body["formal_result"]["shape"] == {"tag": "primitive", "name": "bool"}]
            self.assertTrue(constants)
            for body in constants:
                self.assertIsNone(body["implementation"])
                self.assertEqual(body["substitutions"], [])
            methods = [body for body in evidence.raw["bodies"] if named(body, "get")]
            self.assertTrue(any(body["open_type_or_const"] for body in methods))
            self.assertTrue(any(not body["open_type_or_const"] for body in methods))
            for body in methods:
                self.assertEqual(body["implementation"]["trait"]["path"], "::Argument")
                self.assertIn("Model", json.dumps(body["implementation"]["self_type"]))

    def test_indirect_call_and_branch_or_divergence_are_explicit_obligations(self):
        evidence = self.observe(
            """
struct ValidatedTensor(Vec<i64>); struct DLPackCapsule(Vec<i64>);
pub fn indirect(
    convert: fn(ValidatedTensor) -> DLPackCapsule,
    tensor: ValidatedTensor,
) -> DLPackCapsule { convert(tensor) }
fn reject() -> ! { panic!("rejected") }
pub fn branched(ok: bool, value: Vec<i64>) -> ValidatedTensor {
    if !ok { reject() }
    ValidatedTensor(value)
}
"""
        )
        indirect = [
            call
            for call in evidence.raw["calls"]
            if call["caller"]["definition"]["item_name"] == "indirect"
        ]
        self.assertEqual(len(indirect), 1)
        self.assertEqual(indirect[0]["kind"], "indirect")
        self.assertIsNone(indirect[0]["callee"])
        branched = next(body for body in evidence.raw["bodies"] if named(body, "branched"))
        self.assertTrue(any(len(block["successors"]) == 2 for block in branched["blocks"]))
        reject = calls_named(evidence, "reject")
        self.assertEqual(len(reject), 1)
        self.assertIsNone(reject[0]["target"])

    def test_indirect_calls_and_control_flow_successors_are_retained(self):
        evidence = self.observe(
            """
struct CompiledInputs(Vec<i64>);
fn run(value: &CompiledInputs) -> usize { value.0.len() }
pub fn live(ok: bool, value: &CompiledInputs, callback: fn(&CompiledInputs) -> usize) -> usize {
    if ok { callback(value) } else { run(value) }
}
"""
        )
        indirect = [call for call in evidence.raw["calls"] if call["kind"] == "indirect"]
        self.assertEqual(len(indirect), 1)
        self.assertIsNone(indirect[0]["callee"])
        live = next(body for body in evidence.raw["bodies"] if named(body, "live"))
        self.assertGreaterEqual(len(live["blocks"]), 3)
        self.assertTrue(any(len(block["successors"]) > 1 for block in live["blocks"]))
        self.assertTrue(any(call["target"] is not None for call in evidence.raw["calls"]))
        self.assertTrue(all("unwind" in call for call in evidence.raw["calls"]))

    def test_control_flow_and_assignment_kinds_are_exact_mir_variants(self):
        evidence = self.observe(
            """
struct CompiledInputs(Vec<i64>);
fn execute_checked(value: &CompiledInputs) -> usize { value.0.len() }
pub fn live(ok: bool, raw: Vec<i64>) -> usize {
    let admitted = CompiledInputs(raw);
    if ok { execute_checked(&admitted) } else { 0 }
}
"""
        )
        terminators = {
            block["terminator"]
            for body in evidence.raw["bodies"]
            for block in body["blocks"]
        }
        self.assertTrue({"SwitchInt", "Call", "Return"} <= terminators)
        self.assertTrue(
            terminators
            <= {
                "Goto",
                "SwitchInt",
                "UnwindResume",
                "UnwindTerminate",
                "Return",
                "Unreachable",
                "Drop",
                "Call",
                "TailCall",
                "Assert",
                "Yield",
                "CoroutineDrop",
                "FalseEdge",
                "FalseUnwind",
                "InlineAsm",
            }
        )
        rvalues = {flow["rvalue"] for flow in evidence.raw["flows"]}
        self.assertIn("call-result", rvalues)
        rvalues.remove("call-result")
        self.assertTrue({"Use", "Ref", "Aggregate"} <= rvalues)
        self.assertTrue(
            rvalues
            <= {
                "Use",
                "Repeat",
                "Ref",
                "ThreadLocalRef",
                "RawPtr",
                "Cast",
                "BinaryOp",
                "UnaryOp",
                "Discriminant",
                "Aggregate",
                "CopyForDeref",
                "WrapUnsafeBinder",
                "Reborrow",
            }
        )

    def test_assignment_places_join_constructor_call_and_return(self):
        evidence = self.observe(
            """
struct CompiledInputs(Vec<i64>); struct CompiledTensorResults(Vec<i64>);
impl CompiledInputs { fn admit(v: Vec<i64>) -> Self { Self(v) } }
fn execute_checked(_: &CompiledInputs) -> CompiledTensorResults { CompiledTensorResults(Vec::new()) }
fn adopt_outputs(v: CompiledTensorResults) -> Vec<i64> { v.0 }
pub fn live(raw: Vec<i64>) -> Vec<i64> {
    let admitted = CompiledInputs::admit(raw);
    let executed = execute_checked(&admitted);
    adopt_outputs(executed)
}
"""
        )
        live_calls = [
            call
            for call in evidence.raw["calls"]
            if call["caller"]["definition"]["item_name"] == "live"
        ]
        self.assertEqual(
            {call["callee"]["definition"]["item_name"] for call in live_calls},
            {"admit", "execute_checked", "adopt_outputs"},
        )
        destinations = {call["destination"]["id"] for call in live_calls}
        flow_destinations = {flow["destination"]["id"] for flow in evidence.raw["flows"]}
        self.assertTrue(destinations <= flow_destinations)
        self.assertTrue(any(flow["destination"]["local"] == 0 for flow in evidence.raw["flows"]))


if __name__ == "__main__":
    unittest.main()
