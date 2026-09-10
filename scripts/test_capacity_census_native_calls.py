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
            "format": 3,
            "scope": "native-bindings",
            "compiler": COMPILER,
            "crate": definition,
            "bodies": [],
            "calls": [],
            "aggregates": [],
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
        self.assertEqual(evidence.raw["format"], 3)
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
