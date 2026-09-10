"""Spec/11 native observations must consume their exact validated receiver.

These are compiler-flow obligations, not authority for a Python registration.
"""

from dataclasses import replace
import unittest

from capacity_census_native_flow import (
    AdapterFlowObligation,
    CallStage,
    ConstructorOwnership,
    NativeReceiver,
    native_flow_problems,
)
import test_capacity_census_native_flow as flows


SOURCE = """
struct Tensor(i64);
struct Receiver { tensor: Tensor }
struct Request(i64);
struct Capsule(Request);
fn validated_tensor(value: i64) -> Tensor { Tensor(value) }
impl Request { fn validate(value: &Tensor) -> Self { Self(value.0) } }
fn export(request: Request) -> Capsule { Capsule(request) }
pub fn entry(receiver: &Receiver) -> Capsule {
    let request = Request::validate(&receiver.tensor);
    export(request)
}
"""


def receiver_obligation(raw):
    identity = lambda path: flows.exact_definition(raw, path)
    request = identity("::Request")
    capsule = identity("::Capsule")
    validate = identity("::{impl#0}::validate")
    export = identity("::export")
    receiver = identity("::Receiver")
    return AdapterFlowObligation(
        entry=identity("::entry"),
        stages=(CallStage(validate, validate), CallStage(export, export)),
        constructors=(
            ConstructorOwnership(request, frozenset({validate}), frozenset({validate})),
            ConstructorOwnership(capsule, frozenset({export}), frozenset({export})),
        ),
        native_types=frozenset({receiver, identity("::Tensor"), request, capsule}),
        receiver=NativeReceiver(local=1, carrier=receiver),
    )


class NativeReceiverShapeControls(unittest.TestCase):
    def test_receiver_obligation_requires_a_positive_local_and_native_identity(self):
        identity = flows.NativeFlowShapeControls().identity()
        for local in (0, -1, True, "1"):
            with self.subTest(local=local), self.assertRaises(ValueError):
                NativeReceiver(local=local, carrier=identity)
        with self.assertRaises(ValueError):
            AdapterFlowObligation(
                entry=identity,
                stages=(CallStage(identity, identity),),
                constructors=(),
                native_types=frozenset(),
                receiver=NativeReceiver(local=1, carrier=identity),
            )


class NativeReceiverCompiledControls(unittest.TestCase):
    # Reuse only the direct compiler harness; do not inherit its test cases.
    setUpClass = classmethod(flows.NativeFlowCompiledControls.setUpClass.__func__)
    tearDownClass = classmethod(flows.NativeFlowCompiledControls.tearDownClass.__func__)
    observe = flows.NativeFlowCompiledControls.observe

    def test_exact_receiver_field_reaches_validation_before_export(self):
        raw = self.observe(SOURCE)
        spec = receiver_obligation(raw)
        self.assertEqual(native_flow_problems(raw, spec), [])
        problems = native_flow_problems(raw, replace(spec, receiver=None))
        self.assertTrue(any("unaccounted native-bearing entry input" in p for p in problems), problems)

    def test_receiver_is_neither_mutable_owned_nor_a_same_named_foreign_type(self):
        for declaration in ("receiver: &mut Receiver", "receiver: Receiver"):
            raw = self.observe(SOURCE.replace("receiver: &Receiver", declaration))
            problems = native_flow_problems(raw, receiver_obligation(raw))
            self.assertTrue(any("exact shared receiver" in p for p in problems), problems)
        raw = self.observe(SOURCE)
        spec = receiver_obligation(raw)
        wrong = replace(spec.receiver.carrier, stable_crate_id="not-the-defining-crate")
        altered = replace(spec, native_types=spec.native_types | {wrong},
                          receiver=NativeReceiver(local=1, carrier=wrong))
        problems = native_flow_problems(raw, altered)
        self.assertTrue(any("exact shared receiver" in p for p in problems), problems)

    def test_unused_or_replaced_receiver_cannot_certify_another_tensor(self):
        raw = self.observe(SOURCE.replace(
            "let request = Request::validate(&receiver.tensor);",
            "let other = validated_tensor(7); let request = Request::validate(&other);",
        ))
        problems = native_flow_problems(raw, receiver_obligation(raw))
        self.assertTrue(any("receiver does not reach the first stage" in p for p in problems), problems)

    def test_additional_native_input_and_changed_slot_are_rejected(self):
        raw = self.observe(SOURCE.replace("receiver: &Receiver", "receiver: &Receiver, other: &Tensor"))
        problems = native_flow_problems(raw, receiver_obligation(raw))
        self.assertTrue(any("unaccounted native-bearing entry input" in p for p in problems), problems)
        raw = self.observe(SOURCE)
        spec = receiver_obligation(raw)
        problems = native_flow_problems(raw, replace(spec, receiver=replace(spec.receiver, local=2)))
        self.assertTrue(any("missing exact native receiver" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
