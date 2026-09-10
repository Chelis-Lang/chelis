"""Spec/11 section 1.3: each DLPack keyword reaches its validator unchanged.

These compiler fixtures prove only exact entry-to-call argument forwarding.
They neither establish a validator's semantics nor issue binding authority.
"""

import copy
import unittest

from capacity_census_native_arguments import ArgumentSlot, argument_forwarding_problems
from capacity_census_native_flow import CallStage
import test_capacity_census_native_flow as flows

SOURCE = """
fn validate(stream: Option<i64>, version: Option<i64>, copy: Option<bool>) -> bool {
    stream == version && copy.is_none()
}
pub fn entry(stream: Option<i64>, version: Option<i64>, copy: Option<bool>) -> bool {
    let forwarded = stream;
    validate(forwarded, version, copy)
}
"""


def obligation(raw):
    entry = flows.exact_definition(raw, "::entry")
    validate = flows.exact_definition(raw, "::validate")
    return entry, CallStage(validate, validate), (
        ArgumentSlot(local=1, argument=0),
        ArgumentSlot(local=2, argument=1),
        ArgumentSlot(local=3, argument=2),
    )


class ArgumentSlotControls(unittest.TestCase):
    def test_input_and_argument_positions_have_exact_integer_domains(self):
        self.assertEqual(ArgumentSlot(local=1, argument=0).argument, 0)
        for local in (0, -1, True, "1"):
            with self.subTest(local=local), self.assertRaises(ValueError):
                ArgumentSlot(local=local, argument=0)
        for argument in (-1, True, "0"):
            with self.subTest(argument=argument), self.assertRaises(ValueError):
                ArgumentSlot(local=1, argument=argument)


class ArgumentForwardingCompiledControls(unittest.TestCase):
    setUpClass = classmethod(flows.NativeFlowCompiledControls.setUpClass.__func__)
    tearDownClass = classmethod(flows.NativeFlowCompiledControls.tearDownClass.__func__)
    observe = flows.NativeFlowCompiledControls.observe

    def test_exact_copy_move_forwarding_preserves_each_keyword(self):
        for source in (SOURCE, SOURCE.replace("let forwarded = stream;", "let temporary = stream; let forwarded = temporary;")):
            raw = self.observe(source)
            self.assertEqual(argument_forwarding_problems(raw, *obligation(raw)), [])

    def test_default_swap_transformation_and_overwrite_do_not_forward_the_input(self):
        replacements = (
            ("validate(forwarded, version, copy)", "validate(None, version, copy)"),
            ("validate(forwarded, version, copy)", "validate(version, forwarded, copy)"),
            ("let forwarded = stream;", "let forwarded = stream.map(|value| value + 1);"),
            ("let forwarded = stream;", "let mut forwarded = stream; forwarded = Some(4);"),
            ("let forwarded = stream;", "let mut stream = stream; stream = None; let forwarded = stream;"),
        )
        for old, new in replacements:
            with self.subTest(replacement=new):
                raw = self.observe(SOURCE.replace(old, new))
                self.assertTrue(argument_forwarding_problems(raw, *obligation(raw)))

    def test_missing_ambiguous_or_indirect_validator_is_rejected(self):
        raw = self.observe(SOURCE)
        spec = obligation(raw)
        for mutation in ("missing", "duplicate", "indirect"):
            changed = copy.deepcopy(raw)
            call = next(c for c in changed["calls"] if flows.DefinitionIdentity.from_record(c["caller"]["definition"]) == spec[0])
            if mutation == "missing":
                changed["calls"].remove(call)
            elif mutation == "duplicate":
                changed["calls"].append(copy.deepcopy(call))
            else:
                call["kind"] = "indirect"
                call["callee"] = None
            with self.subTest(mutation=mutation):
                self.assertTrue(argument_forwarding_problems(changed, *spec))

    def test_closed_types_whole_places_unique_writes_and_order_are_required(self):
        raw = self.observe(SOURCE)
        spec = obligation(raw)
        for mutation in ("open", "type", "projection", "late", "duplicate-write", "cycle"):
            changed = copy.deepcopy(raw)
            call = next(c for c in changed["calls"] if flows.DefinitionIdentity.from_record(c["caller"]["definition"]) == spec[0])
            destination = call["arguments"][0]["place"]["local"]
            flow = next(f for f in changed["flows"] if flows.DefinitionIdentity.from_record(f["caller"]["definition"]) == spec[0] and f["destination"]["local"] == destination)
            if mutation == "open":
                call["arguments"][0]["place"]["type"]["open"] = True
            elif mutation == "type":
                call["formal_inputs"][0]["shape"] = {"tag": "primitive", "name": "i32"}
            elif mutation == "projection":
                flow["sources"][0]["projection"] = ["Field(0, i64)"]
            elif mutation == "late":
                flow["statement"] = call["statement"] + 1
            elif mutation == "duplicate-write":
                changed["flows"].append(copy.deepcopy(flow))
            else:
                flow["sources"][0] = copy.deepcopy(flow["destination"])
            with self.subTest(mutation=mutation):
                self.assertTrue(argument_forwarding_problems(changed, *spec))

    def test_absent_or_duplicate_slot_obligations_fail_closed(self):
        raw = self.observe(SOURCE)
        entry, stage, slots = obligation(raw)
        for changed in ((), slots + (slots[0],), (ArgumentSlot(local=9, argument=0),), (ArgumentSlot(local=1, argument=9),)):
            with self.subTest(slots=changed):
                self.assertTrue(argument_forwarding_problems(raw, entry, stage, changed))


if __name__ == "__main__":
    unittest.main()
