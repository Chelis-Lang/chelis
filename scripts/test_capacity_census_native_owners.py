"""Integrated native ownership obligations; no result from this suite is authority."""

import copy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import unittest

from capacity_census_graph import GraphError, RustdocGraph
from capacity_census_native_flow import DefinitionIdentity
from capacity_census_native_registration import CheckedNativeRegistrations


ROOT = Path(__file__).resolve().parent.parent
PUBLIC = (
    "chelis_python::CompiledModel::__call__",
    "chelis_python::NativeTensor::shape",
    "chelis_python::NativeTensor::__dlpack_device__",
    "chelis_python::NativeTensor::__dlpack__",
)


def _definition(value):
    return DefinitionIdentity.from_record(value)


def _entry(raw, name, self_path):
    rows = [
        body
        for body in raw["bodies"]
        if body.get("kind") == "AssocFn"
        and body.get("definition", {}).get("item_name") == name
        and body.get("implementation", {}).get("trait") is None
        and body["implementation"].get("self_type", {}).get("nominal", {}).get("crate")
        == "chelis_python"
        and body["implementation"]["self_type"]["nominal"].get("path") == self_path
    ]
    if len(rows) != 1:
        raise AssertionError((name, self_path, len(rows)))
    return rows[0]


class NativeOwnerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from capacity_census_native_registration import collect_native_registration
        from capacity_census_typed import generate_rustdoc_json
        from capacity_census_wire_calls import build_driver, collect_library

        requested = os.environ.get("CARGO_TARGET_DIR")
        cls.target = Path(requested).resolve() if requested else (
            ROOT / "target/agents/native-owner-integration"
        ).resolve()
        if not cls.target.is_relative_to(ROOT / "target"):
            raise RuntimeError("native owner integration target must belong to this worktree")
        cls.registrations = collect_native_registration(ROOT, cls.target)
        cls.documents = [
            generate_rustdoc_json(
                root=ROOT,
                package=package,
                crate_name=crate,
                target_dir=cls.target,
            )
            for package, crate in (
                ("chelis-python", "chelis_python"),
                ("chelis-abi", "chelis_abi"),
            )
        ]
        driver = build_driver(ROOT, cls.target / "native-owner-driver")
        cls.raw = collect_library(
            ROOT, cls.target, driver, scope="native-bindings"
        )["evidence"]

    def resolve(self, *, raw=None, documents=None, registrations=None):
        from capacity_census_native_owners import resolve_native_ownership

        return resolve_native_ownership(
            RustdocGraph(copy.deepcopy(self.documents if documents is None else documents)),
            self.raw if raw is None else raw,
            self.registrations if registrations is None else registrations,
        )

    def test_actual_registration_rustdoc_and_native4_form_one_bounded_report(self):
        before = hashlib.sha256(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report = self.resolve()
        after = hashlib.sha256(
            json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(after, before)
        self.assertEqual(tuple(row.public for row in report.entries), PUBLIC)
        self.assertEqual(len({row.implementation for row in report.entries}), 4)
        self.assertEqual(tuple(name for name, _ in report.private_fields), (
            "chelis_python::NativeTensor",
            "chelis_python::native_tensor::ValidatedTensor",
            "chelis_python::native_tensor::CompiledTensorResults",
            "chelis_python::dlpack::DLPackDevice",
            "chelis_python::dlpack::DLPackCapsule",
            "chelis_python::dlpack::DLPackRequest",
        ))
        self.assertEqual(report.export_abi, ("Legacy", "Versioned"))
        self.assertEqual(report.tensor_owner_choices, ("Cpu", "Gpu"))
        self.assertEqual(len(report.constructor_scopes), 12)
        self.assertNotIn("authority", report.__dataclass_fields__)
        dlpack = next(row for row in report.entries if row.public.endswith("::__dlpack__"))
        self.assertEqual(
            tuple((slot.local, slot.argument) for slot in dlpack.keyword_slots),
            ((2, 1), (3, 2), (4, 3), (5, 4)),
        )

    def test_foreign_same_named_impl_and_changed_receiver_are_rejected(self):
        foreign = copy.deepcopy(self.raw)
        body = _entry(foreign, "shape", "::NativeTensor")
        owner = body["implementation"]["self_type"]["nominal"]
        owner.update(crate="foreign_crate", stable_crate_id="f" * 16)
        with self.assertRaisesRegex(GraphError, "registered implementation"):
            self.resolve(raw=foreign)

        changed = copy.deepcopy(self.raw)
        body = _entry(changed, "shape", "::NativeTensor")
        body["formal_inputs"][0]["type"]["shape"]["mutable"] = True
        with self.assertRaisesRegex(GraphError, "receiver"):
            self.resolve(raw=changed)

    def test_raw_constructor_exposure_and_constructor_closure_are_rejected(self):
        exposed = copy.deepcopy(self.raw)
        use = copy.deepcopy(exposed["constructor_uses"][0])
        carrier = next(
            row["definition"]
            for row in exposed["aggregates"]
            if row["definition"].get("crate") == "chelis_python"
            and row["definition"].get("path") == "::NativeTensor"
        )
        use["operand_index"] += 1000
        use["carrier"] = copy.deepcopy(carrier)
        use["formal_result"]["nominal"] = copy.deepcopy(carrier)
        use["formal_result"]["shape"]["definition"] = copy.deepcopy(carrier)
        exposed["constructor_uses"].append(use)
        with self.assertRaisesRegex(GraphError, "constructor function value"):
            self.resolve(raw=exposed)

        closure = copy.deepcopy(self.raw)
        aggregate = next(
            row
            for row in closure["aggregates"]
            if row["definition"].get("crate") == "chelis_python"
            and row["definition"].get("path") == "::native_tensor::ValidatedTensorInner"
        )
        caller = _definition(aggregate["caller"]["definition"])
        for body in closure["bodies"]:
            if _definition(body["definition"]) == caller:
                body["kind"] = "Closure"
        for row in closure["aggregates"]:
            if _definition(row["caller"]["definition"]) == caller:
                row["caller"]["kind"] = "Closure"
        with self.assertRaisesRegex(GraphError, "governing method|constructor"):
            self.resolve(raw=closure)

    def test_missing_governing_body_and_changed_private_field_are_rejected(self):
        missing = copy.deepcopy(self.raw)
        missing["bodies"] = [
            body
            for body in missing["bodies"]
            if not (
                body["definition"].get("crate") == "chelis_python"
                and body["definition"].get("path") == "::gpu_input_tensor"
            )
        ]
        with self.assertRaisesRegex(GraphError, "governing function"):
            self.resolve(raw=missing)

        documents = copy.deepcopy(self.documents)
        graph = RustdocGraph(documents)
        crate, item = graph.locations["chelis_python::dlpack::DLPackRequest"]
        request = graph._item(crate, item)
        field = request["inner"]["struct"]["kind"]["plain"]["fields"][1]
        graph._item(crate, field)["name"] = "unchecked_abi"
        with self.assertRaisesRegex(GraphError, "private field"):
            self.resolve(documents=list(graph.documents.values()))

    def test_wrong_registered_implementation_and_unsealed_input_are_rejected(self):
        clone = object.__new__(CheckedNativeRegistrations)
        for field in fields(CheckedNativeRegistrations):
            object.__setattr__(clone, field.name, copy.deepcopy(getattr(self.registrations, field.name)))
        clone.packet["registrations"][0]["rust_name"] = "same_named_helper"
        from capacity_census_wire_adapters import canonical
        object.__setattr__(
            clone,
            "packet_sha256",
            hashlib.sha256(canonical(clone.packet).encode()).hexdigest(),
        )
        with self.assertRaisesRegex(GraphError, "registration"):
            self.resolve(registrations=clone)
        with self.assertRaisesRegex(GraphError, "checked native registrations"):
            self.resolve(registrations={})


if __name__ == "__main__":
    unittest.main()
