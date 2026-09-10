"""Obligation controls for compiler-derived native adapter flow evidence.

These tests prove exact direct-call, CFG, place-flow, and constructor-owner
mechanisms. They do not treat the resulting report as numeric authority or as a
general MIR soundness proof.
"""

import copy
import tempfile
import unittest
from pathlib import Path

from capacity_census_native_flow import (
    AdapterFlowObligation,
    CallStage,
    ConstructorOwnership,
    DefinitionIdentity,
    native_flow_problems,
)


ROOT = Path(__file__).resolve().parent.parent


def definitions(value):
    if isinstance(value, dict):
        if {
            "stable_crate_id",
            "def_id",
            "def_path_hash",
        } <= value.keys():
            yield value
        for child in value.values():
            yield from definitions(child)
    elif isinstance(value, list):
        for child in value:
            yield from definitions(child)


def exact_definition(raw, path):
    found = {
        DefinitionIdentity.from_record(value)
        for value in definitions(raw)
        if value.get("path") == path
    }
    if len(found) != 1:
        raise AssertionError(f"expected one compiled identity for {path}: {found}")
    return found.pop()


def obligation(raw):
    entry = exact_definition(raw, "::entry")
    admit = exact_definition(raw, "::{impl#0}::admit")
    execute = exact_definition(raw, "::execute_checked")
    adopt = exact_definition(raw, "::{impl#1}::adopt_outputs")
    inputs = exact_definition(raw, "::Inputs")
    outputs = exact_definition(raw, "::RawOutputs")
    results = exact_definition(raw, "::Results")
    return AdapterFlowObligation(
        entry=entry,
        stages=(
            CallStage(admit, admit),
            CallStage(execute, execute),
            CallStage(adopt, adopt),
        ),
        constructors=(
            ConstructorOwnership(inputs, frozenset({admit}), frozenset({admit})),
            ConstructorOwnership(outputs, frozenset({execute}), frozenset({execute})),
            ConstructorOwnership(results, frozenset({adopt}), frozenset({adopt})),
        ),
        native_types=frozenset({inputs, outputs, results}),
    )


def result_obligation(raw):
    base = obligation(raw)
    variants = {
        DefinitionIdentity.from_record(row["variant_definition"])
        for row in raw["aggregates"]
        if row["variant"] == "Ok" and row["variant_definition"]["crate"] == "core"
    }
    if len(variants) != 1:
        raise AssertionError(f"expected one exact core Result::Ok identity: {variants}")
    return AdapterFlowObligation(
        entry=base.entry,
        stages=base.stages,
        constructors=base.constructors,
        native_types=base.native_types,
        success_variant=variants.pop(),
    )


def fallible_obligation(raw):
    base = result_obligation(raw)
    stage_identities = {(stage.declared, stage.resolved) for stage in base.stages}
    entry_calls = [
        call
        for call in raw["calls"]
        if DefinitionIdentity.from_record(call["caller"]["definition"]) == base.entry
        and call["callee"] is not None
    ]
    controls = []
    control_paths = []
    for call in entry_calls:
        declared_record = call["callee"]["definition"]
        resolved_record = call["callee"]["resolved"]["definition"]
        identity = (
            DefinitionIdentity.from_record(declared_record),
            DefinitionIdentity.from_record(resolved_record),
        )
        if identity in stage_identities:
            continue
        if declared_record["crate"] == "core":
            controls.append(CallStage(*identity))
            control_paths.append(declared_record["path"])
    self_describing = " ".join(control_paths)
    if "branch" not in self_describing or "from_residual" not in self_describing:
        raise AssertionError(f"missing exact core Result/Try controls: {control_paths}")
    return AdapterFlowObligation(
        entry=base.entry,
        stages=base.stages,
        constructors=base.constructors,
        native_types=base.native_types,
        success_variant=base.success_variant,
        control_calls=tuple(controls),
    )


POSITIVE = """
struct Inputs(Vec<i64>);
struct RawOutputs(Vec<i64>);
struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let admitted = Inputs::admit(value);
    let outputs = execute_checked(admitted);
    Results::adopt_outputs(outputs)
}
"""


RESULT_POSITIVE = POSITIVE.replace(
    "pub fn entry(value: Vec<i64>) -> Results {",
    "pub fn entry(fail: bool, value: Vec<i64>) -> Result<Results, &'static str> {",
).replace(
    "let admitted = Inputs::admit(value);",
    'if fail { return Err("failed") }\n    let admitted = Inputs::admit(value);',
).replace(
    "Results::adopt_outputs(outputs)\n}",
    "Ok(Results::adopt_outputs(outputs))\n}",
)


FALLIBLE = """
struct Inputs(Vec<i64>);
struct RawOutputs(Vec<i64>);
struct Results(Vec<i64>);
impl Inputs {
    fn admit(value: Vec<i64>) -> Result<Self, &'static str> {
        if value.is_empty() { Err("empty") } else { Ok(Self(value)) }
    }
}
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results {
    fn adopt_outputs(value: RawOutputs) -> Result<Self, &'static str> {
        if value.0.is_empty() { Err("empty") } else { Ok(Self(value.0)) }
    }
}
pub fn entry(value: Vec<i64>) -> Result<Results, &'static str> {
    let admitted = Inputs::admit(value)?;
    let outputs = execute_checked(admitted);
    Results::adopt_outputs(outputs)
}
"""


class NativeFlowShapeControls(unittest.TestCase):
    def identity(self, suffix="1"):
        return DefinitionIdentity.from_record(
            {
                "stable_crate_id": "a" * 16,
                "def_id": f"0:{suffix}",
                "def_path_hash": f"DefPathHash(Fingerprint(0, {suffix}))",
                "item_name": "diagnostic-only",
                "path": "::diagnostic_only",
            }
        )

    def test_definition_identity_uses_only_compiler_identity_fields(self):
        record = {
            "stable_crate_id": "a" * 16,
            "def_id": "0:1",
            "def_path_hash": "DefPathHash(Fingerprint(0, 1))",
            "item_name": "first",
            "path": "::first",
        }
        first = DefinitionIdentity.from_record(record)
        record["item_name"] = "second"
        record["path"] = "::second"
        self.assertEqual(first, DefinitionIdentity.from_record(record))
        record["def_path_hash"] = "DefPathHash(Fingerprint(0, 2))"
        self.assertNotEqual(first, DefinitionIdentity.from_record(record))

    def test_malformed_evidence_and_obligations_fail_closed(self):
        identity = self.identity()
        spec = AdapterFlowObligation(
            entry=identity,
            stages=(CallStage(identity, identity),),
            constructors=(),
            native_types=frozenset(),
        )
        self.assertIn("malformed native flow evidence", native_flow_problems([], spec)[0])
        with self.assertRaises(ValueError):
            ConstructorOwnership(identity, frozenset(), frozenset())
        with self.assertRaises(ValueError):
            AdapterFlowObligation(identity, (), (), frozenset())


class NativeFlowCompiledControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_wire_calls import build_driver

        cls.scratch = tempfile.TemporaryDirectory(
            prefix="native-flow-", dir=ROOT / "target"
        )
        cls.directory = Path(cls.scratch.name)
        cls.driver = build_driver(
            ROOT, ROOT / "target/agents/native-bindings-driver/native-flow-tool"
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
        ).raw

    def assert_problem(self, raw, expected):
        problems = native_flow_problems(raw, obligation(raw))
        self.assertTrue(
            any(expected in problem for problem in problems),
            f"missing {expected!r} in {problems}",
        )

    def test_exact_direct_chain_dominates_and_reaches_return(self):
        raw = self.observe(POSITIVE)
        self.assertEqual(native_flow_problems(raw, obligation(raw)), [])

    def test_flow_requires_constructor_use_evidence_and_rejects_guarded_callbacks(self):
        raw = self.observe(POSITIVE)
        expected = obligation(raw)
        self.assertEqual(native_flow_problems(raw, expected), [])
        del raw["constructor_uses"]
        self.assertTrue(any("constructor_uses" in p for p in native_flow_problems(raw, expected)))
        raw = self.observe(POSITIVE +
                           "fn callback() -> fn(Vec<i64>) -> Results { Results }")
        self.assert_problem(raw, "constructor function value")

    def test_constructor_promoted_identity_is_required_and_typed(self):
        raw = self.observe(
            POSITIVE + "fn callback() -> fn(Vec<i64>) -> Results { Results }"
        )
        expected = obligation(raw)
        self.assertTrue(all(row["promoted"] is None for row in raw["constructor_uses"]))
        changed = copy.deepcopy(raw)
        changed["constructor_uses"][0]["promoted"] = "0"
        problems = native_flow_problems(changed, expected)
        self.assertTrue(
            any("invalid native constructor use location" in problem for problem in problems),
            problems,
        )

    def test_manual_constructor_rule_cannot_authorize_closure_body(self):
        raw = self.observe(POSITIVE.replace("Self(value.0)", "(|| Self(value.0))()"))
        expected = obligation(raw)
        closure = next(DefinitionIdentity.from_record(a["caller"]["definition"])
                       for a in raw["aggregates"] if a["definition"]["path"] == "::Results")
        changed = AdapterFlowObligation(
            entry=expected.entry, stages=expected.stages,
            constructors=expected.constructors[:-1] + (
                ConstructorOwnership(expected.constructors[-1].carrier,
                                     frozenset({closure}), frozenset({closure})),),
            native_types=expected.native_types,
        )
        self.assertTrue(any("direct governing" in p for p in native_flow_problems(raw, changed)))

    def test_every_exact_result_success_must_be_stage_proven(self):
        raw = self.observe(RESULT_POSITIVE)
        self.assertEqual(native_flow_problems(raw, result_obligation(raw)), [])
        bypass = self.observe(
            RESULT_POSITIVE.replace(
                'if fail { return Err("failed") }',
                """
if fail {
        let unchecked = Results(Vec::new());
        return Ok(unchecked)
    }
""",
            )
        )
        problems = native_flow_problems(bypass, result_obligation(bypass))
        self.assertTrue(
            any("successful return bypasses final stage" in problem for problem in problems),
            problems,
        )
        lookalike = self.observe(
            RESULT_POSITIVE.replace(
                "Ok(Results::adopt_outputs(outputs))",
                """
let result = Results::adopt_outputs(outputs);
    let _lookalike = Local::Ok(&result);
    Ok(result)
""",
            )
            + """
enum Local<T> { Ok(T), Err }
"""
        )
        spec = result_obligation(lookalike)
        local_ok = next(
            DefinitionIdentity.from_record(row["variant_definition"])
            for row in lookalike["aggregates"]
            if row["variant"] == "Ok" and row["variant_definition"]["crate"] == "native_fixture"
        )
        self.assertNotEqual(spec.success_variant, local_ok)
        self.assertEqual(native_flow_problems(lookalike, spec), [])

    def test_exact_core_try_plumbing_preserves_the_fallible_adapter_chain(self):
        raw = self.observe(FALLIBLE)
        spec = fallible_obligation(raw)
        self.assertTrue(spec.control_calls)
        self.assertEqual(native_flow_problems(raw, spec), [])
        without_controls = AdapterFlowObligation(
            entry=spec.entry,
            stages=spec.stages,
            constructors=spec.constructors,
            native_types=spec.native_types,
            success_variant=spec.success_variant,
        )
        problems = native_flow_problems(raw, without_controls)
        self.assertTrue(
            any("unaccounted native-bearing call" in problem for problem in problems),
            problems,
        )

    def test_diagnostic_names_are_not_identity(self):
        raw = self.observe(POSITIVE)
        expected = obligation(raw)
        changed = copy.deepcopy(raw)
        for definition in definitions(changed):
            definition["item_name"] = "misleading"
            definition["path"] = "::misleading"
        self.assertEqual(native_flow_problems(changed, expected), [])
        changed = copy.deepcopy(raw)
        entry = next(
            body for body in changed["bodies"] if body["definition"]["path"] == "::entry"
        )
        entry["definition"]["def_path_hash"] = "DefPathHash(Fingerprint(0, 0))"
        problems = native_flow_problems(changed, expected)
        self.assertTrue(any("entry body" in problem for problem in problems), problems)

    def test_missing_and_reversed_stage_are_rejected(self):
        missing = self.observe(
            POSITIVE.replace(
                "let outputs = execute_checked(admitted);",
                "let outputs = RawOutputs(admitted.0);",
            )
        )
        self.assert_problem(missing, "missing exact stage")
        reversed_raw = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let early = execute_checked(Inputs(value));
    let admitted = Inputs::admit(early.0);
    let outputs = RawOutputs(admitted.0);
    Results::adopt_outputs(outputs)
}
"""
        )
        self.assert_problem(reversed_raw, "stage order")

    def test_bypassed_place_provenance_is_rejected(self):
        raw = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let bypass = Inputs(value);
    let admitted = Inputs::admit(Vec::new());
    let outputs = execute_checked(bypass);
    let _ = admitted;
    Results::adopt_outputs(outputs)
}
"""
        )
        self.assert_problem(raw, "place provenance")

    def test_indirect_and_same_named_lookalike_calls_are_rejected(self):
        indirect = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let callback: fn(Inputs) -> RawOutputs = execute_checked;
    let admitted = Inputs::admit(value);
    let outputs = callback(admitted);
    Results::adopt_outputs(outputs)
}
"""
        )
        self.assert_problem(indirect, "missing exact stage")
        lookalike = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
mod other {
    use super::*;
    pub(super) fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
}
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let admitted = Inputs::admit(value);
    let outputs = other::execute_checked(admitted);
    Results::adopt_outputs(outputs)
}
"""
        )
        spec = obligation(lookalike)
        real = exact_definition(lookalike, "::execute_checked")
        fake = exact_definition(lookalike, "::other::execute_checked")
        self.assertNotEqual(real, fake)
        self.assert_problem(lookalike, "missing exact stage")

    def test_private_unreachable_constructor_and_unused_validator_do_not_certify(self):
        raw = self.observe(
            POSITIVE
            + """
fn private_bypass(value: Vec<i64>) -> Results { Results(value) }
"""
        )
        self.assert_problem(raw, "constructor owner")
        unused = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
fn unused_validated(value: RawOutputs) -> Results { Results::adopt_outputs(value) }
fn unchecked(value: RawOutputs) -> Results { Results(value.0) }
pub fn entry(value: Vec<i64>) -> Results {
    let admitted = Inputs::admit(value);
    let outputs = execute_checked(admitted);
    unchecked(outputs)
}
"""
        )
        self.assert_problem(unused, "missing exact stage")
        self.assert_problem(unused, "constructor owner")

    def test_unaccounted_native_bearing_call_is_rejected(self):
        raw = self.observe(
            """
struct Inputs(Vec<i64>); struct RawOutputs(Vec<i64>); struct Results(Vec<i64>);
impl Inputs { fn admit(value: Vec<i64>) -> Self { Self(value) } }
fn execute_checked(value: Inputs) -> RawOutputs { RawOutputs(value.0) }
fn passthrough(value: RawOutputs) -> RawOutputs { value }
impl Results { fn adopt_outputs(value: RawOutputs) -> Self { Self(value.0) } }
pub fn entry(value: Vec<i64>) -> Results {
    let admitted = Inputs::admit(value);
    let outputs = execute_checked(admitted);
    let outputs = passthrough(outputs);
    Results::adopt_outputs(outputs)
}
"""
        )
        self.assert_problem(raw, "unaccounted native-bearing call")


if __name__ == "__main__":
    unittest.main()
