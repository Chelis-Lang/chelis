"""Publication ownership cannot be supplied by a name or an outer tag."""

import copy
import unittest

from capacity_census_graph import GraphError


def nominal(name, *arguments):
    crate, path = name.split("::", 1)
    return {
        "tag": "nominal",
        "definition": {"crate": crate, "path": "::" + path},
        "arguments": list(arguments),
    }


def primitive(name):
    return {"tag": "primitive", "name": name}


def definition(name, number, crate="chelis_compiler_api", item_name=None):
    return {
        "def_id": f"0:{number}",
        "def_path_hash": f"hash-{crate}-{number}",
        "stable_crate_id": "compiler" if crate == "chelis_compiler_api" else crate,
        "crate": crate,
        "path": name,
        "item_name": item_name,
    }


def check_report_call():
    root = definition("", 0)
    result = definition("::schema::CheckResult", 1)
    method = definition(
        "::check_report::<impl>::to_report_json", 2, item_name="to_report_json"
    )
    return root, {
        "caller": {
            "definition": method,
            "implementation": {"trait": None, "self_type": {"nominal": result}},
            "ancestors": [root],
            "substitutions": "[]",
        },
        "callee": definition("::ser::Serialize::serialize", 3, "serde_core"),
        "payloads": [{"shape": nominal("chelis_compiler_api::schema::CheckResult")}],
        "serializers": [],
        "local_callee": False,
    }



def codec_specialization():
    """Compiler-shaped fixture with one shared owner/payload type parameter."""
    owner = definition("::schema::Envelope", 20)
    carrier = definition("::schema::Carrier", 21)
    serializer = definition("::ser::Writer", 22, "encoding")
    trait = definition("::ser::Serialize", 23, "serde_core")

    def typed_nominal(declaration, *arguments):
        return {"tag": "nominal", "definition": declaration, "arguments": list(arguments)}

    parameter = {"tag": "parameter", "index": 0}
    payload = {"tag": "tuple", "elements": [parameter, parameter]}
    template = {
        "caller": {
            "definition": definition("::schema::codec", 24, item_name="serialize"),
            "ancestors": [definition("", 0)],
            "implementation": {
                "trait": trait,
                "self_type": {"nominal": owner, "shape": typed_nominal(owner, parameter)},
            },
            "substitutions": "display text is not authority",
        },
        "callee": definition("::ser::SerializeStruct::serialize_field", 25, "serde_core"),
        "source": {"span": "source.rs:7:1: 7:20", "expansion": "compiler expansion"},
        "payloads": [{"shape": payload}],
        "serializers": [{"shape": {"tag": "parameter", "index": 1}}],
        "local_callee": False,
    }
    concrete = copy.deepcopy(template)
    concrete["caller"]["implementation"]["self_type"]["shape"] = typed_nominal(
        owner, typed_nominal(carrier)
    )
    concrete["payloads"] = [{"shape": {
        "tag": "tuple", "elements": [typed_nominal(carrier), typed_nominal(carrier)]
    }}]
    concrete["serializers"] = [{"shape": typed_nominal(serializer)}]
    return template, concrete, trait, {
        "chelis_compiler_api::schema::Envelope": 1,
        "chelis_compiler_api::schema::Carrier": 0,
    }


class CodecSpecializations(unittest.TestCase):
    def owner(self, call, templates, trait, definitions):
        from capacity_census_wire_invocation_owners import codec_specialization_owner

        return codec_specialization_owner(call, templates, trait, definitions)

    def test_specializations_bind_owner_payload_and_serializer_shapes(self):
        template, call, trait, definitions = codec_specialization()
        self.assertEqual(
            self.owner(call, [template], trait, definitions),
            "wire-codec/chelis_compiler_api::schema::Envelope",
        )
        # The same derive span can own multiple field calls. Actual typed
        # obligations, rather than the span alone, select one template.
        other = copy.deepcopy(template)
        other["payloads"] = [{"shape": primitive("bool")}]
        self.assertEqual(
            self.owner(call, [other, template], trait, definitions),
            "wire-codec/chelis_compiler_api::schema::Envelope",
        )
        # Display spellings and formatter names do not grant or remove ownership.
        call["caller"]["substitutions"] = "unrelated pretty printer spelling"
        call["serializers"][0]["shape"]["definition"]["path"] = "::OtherWriter"
        self.owner(call, [template], trait, definitions)
        # A serializer-only call remains an obligation of its concrete wire owner.
        template["payloads"] = []
        call["payloads"] = []
        self.owner(call, [template], trait, definitions)

    def test_specializations_reject_missing_duplicate_or_replaced_templates(self):
        template, call, trait, definitions = codec_specialization()
        for templates in ([], [template, copy.deepcopy(template)]):
            with self.subTest(templates=len(templates)), self.assertRaises(GraphError):
                self.owner(call, templates, trait, definitions)
        for field in ("def_id", "def_path_hash", "stable_crate_id", "crate", "path", "item_name"):
            for location in ("caller", "callee", "owner", "trait"):
                changed = copy.deepcopy(template)
                targets = {
                    "caller": changed["caller"]["definition"],
                    "callee": changed["callee"],
                    "owner": changed["caller"]["implementation"]["self_type"]["shape"]["definition"],
                    "trait": changed["caller"]["implementation"]["trait"],
                }
                targets[location][field] = str(targets[location][field]) + "-copied"
                with self.subTest(location=location, field=field), self.assertRaises(GraphError):
                    self.owner(call, [changed], trait, definitions)
        for field in ("def_id", "def_path_hash", "stable_crate_id", "crate", "path", "item_name"):
            changed = copy.deepcopy(template)
            del changed["callee"][field]
            with self.subTest(missing=field), self.assertRaises(GraphError):
                self.owner(call, [changed], trait, definitions)
        for location in ("span", "expansion", "local_callee", "ancestry"):
            changed = copy.deepcopy(template)
            if location in ("span", "expansion"):
                changed["source"][location] += "-copied"
            elif location == "local_callee":
                changed[location] = True
            else:
                changed["caller"]["ancestors"][0]["def_path_hash"] += "-copied"
            with self.subTest(location=location), self.assertRaises(GraphError):
                self.owner(call, [changed], trait, definitions)

    def test_specializations_reject_rebound_parameters_and_unknown_shapes(self):
        template, call, trait, definitions = codec_specialization()
        for mutation in ("payload", "serializer", "repeated", "unknown", "open", "identity"):
            changed, current = copy.deepcopy(template), copy.deepcopy(call)
            if mutation == "payload":
                current["payloads"][0]["shape"]["elements"] = [primitive("bool")] * 2
            elif mutation == "serializer":
                changed["serializers"][0]["shape"]["index"] = 0
            elif mutation == "repeated":
                current["payloads"][0]["shape"]["elements"][1] = primitive("char")
            elif mutation == "unknown":
                current["serializers"][0]["shape"] = {"tag": "projection"}
            elif mutation == "open":
                current["serializers"][0]["shape"] = {"tag": "parameter", "index": 1}
            else:
                current["payloads"][0]["shape"]["elements"][0] = copy.deepcopy(
                    current["payloads"][0]["shape"]["elements"][0]
                )
                current["payloads"][0]["shape"]["elements"][0]["definition"]["def_path_hash"] += "-copied"
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                self.owner(current, [changed], trait, definitions)
        for malformed in ({"tag": "parameter", "index": True},
                          {"tag": "parameter", "index": -1},
                          {"tag": "parameter", "index": 0, "name": "T"},
                          {"tag": "future"}):
            changed = copy.deepcopy(template)
            changed["serializers"][0]["shape"] = malformed
            with self.subTest(malformed=malformed), self.assertRaises(GraphError):
                self.owner(call, [changed], trait, definitions)

    def test_specializations_cannot_admit_bare_numbers_or_binary_owners(self):
        from capacity_census_wire_publication import COMPILER_BINARY_OWNERS

        template, call, trait, definitions = codec_specialization()
        for payload in (primitive("f64"), primitive("u64")):
            changed, current = copy.deepcopy(template), copy.deepcopy(call)
            changed["payloads"] = current["payloads"] = [{"shape": payload}]
            with self.subTest(payload=payload), self.assertRaisesRegex(GraphError, "bare numeric"):
                self.owner(current, [changed], trait, definitions)
        for owner in COMPILER_BINARY_OWNERS:
            crate, path = owner.split("::", 1)
            binary = {"tag": "nominal", "definition": definition("::" + path, 40, crate), "arguments": []}
            for position in ("owner", "payload"):
                changed, current = copy.deepcopy(template), copy.deepcopy(call)
                if position == "owner":
                    for row in (changed, current):
                        row["caller"]["implementation"]["self_type"] = {
                            "nominal": binary["definition"], "shape": binary
                        }
                    changed["payloads"] = current["payloads"] = [{"shape": primitive("bool")}]
                else:
                    changed["payloads"] = current["payloads"] = [{"shape": binary}]
                with self.subTest(owner=owner, position=position), self.assertRaises(GraphError):
                    self.owner(current, [changed], trait, dict(definitions, **{owner: 0}))
        current = copy.deepcopy(call)
        current["caller"]["implementation"]["self_type"]["shape"]["arguments"] = [primitive("f64")]
        current["payloads"][0]["shape"]["elements"] = [primitive("f64")] * 2
        with self.assertRaisesRegex(GraphError, "bare numeric"):
            self.owner(current, [template], trait, definitions)

    def test_independent_publication_cannot_borrow_a_surrounding_codec(self):
        template, call, trait, definitions = codec_specialization()
        for change in ("source", "callee", "fully_concrete", "no_obligations"):
            current, templates = copy.deepcopy(call), [copy.deepcopy(template)]
            if change == "source":
                current["source"]["span"] = "source.rs:8:1: 8:20"
            elif change == "callee":
                current["callee"] = definition("::to_string", 41, "serde_json")
            elif change == "fully_concrete":
                # Even a fabricated identical site is not a generic obligation.
                templates = [copy.deepcopy(current)]
            else:
                current["payloads"] = current["serializers"] = []
                templates[0]["payloads"] = templates[0]["serializers"] = []
            with self.subTest(change=change), self.assertRaises(GraphError):
                self.owner(current, templates, trait, definitions)


class SerializeWithHelpers(unittest.TestCase):
    """A serde `with` field helper is owned by its exact enclosing derive."""

    PARENT = "::schema::execution::_#2::{impl#0}::serialize"
    OWNER = "chelis_compiler_api::schema::execution::TensorWire"

    def rows(self):
        def definition(path, name, hash_):
            return {"crate": "chelis_compiler_api", "path": path, "item_name": name,
                    "def_path_hash": hash_}

        derive = {
            "caller": {
                "definition": definition(self.PARENT, "serialize", "parent"),
                "implementation": {"self_type": {"nominal": definition(
                    "::schema::execution::TensorWire", "TensorWire", "owner")}},
                "ancestors": [],
            },
            "callee": {"crate": "serde_core", "path": "::ser::SerializeStruct::serialize_field"},
            "payloads": [{"text": "schema::execution::_::<impl Serialize for TensorWire>::serialize::__SerializeWith<'_>"}],
        }
        helper = {
            "caller": {
                "definition": definition(self.PARENT + "::{impl#0}::serialize", "serialize", "helper-fn"),
                "implementation": {"self_type": {"nominal": definition(
                    self.PARENT + "::__SerializeWith", "__SerializeWith", "helper")}},
                "ancestors": [definition(self.PARENT, "serialize", "parent")],
            },
            "callee": {"crate": "chelis_types",
                       "path": "::dtype_semantics::wire_codec::execution_storage::serialize"},
            "payloads": [],
        }
        return derive, helper

    def test_the_helper_is_owned_by_its_exact_derive(self):
        from capacity_census_wire_invocation_owners import serialize_with_helper_owner

        derive, helper = self.rows()
        self.assertEqual(
            serialize_with_helper_owner(helper, [derive, helper], {self.OWNER: 0}),
            self.OWNER,
        )

    def test_another_codec_parent_or_owner_is_rejected(self):
        from capacity_census_wire_invocation_owners import serialize_with_helper_owner

        for change in ("codec", "parent", "absent-owner", "no-payload", "two-owners"):
            derive, helper = self.rows()
            calls = [derive, helper]
            definitions = {self.OWNER: 0}
            if change == "codec":
                helper["callee"]["path"] = "::dtype_semantics::wire_codec::other::serialize"
            elif change == "parent":
                helper["caller"]["ancestors"][0]["def_path_hash"] = "elsewhere"
            elif change == "absent-owner":
                definitions = {}
            elif change == "no-payload":
                derive["payloads"] = []
            else:
                other = copy.deepcopy(derive)
                other["caller"]["implementation"]["self_type"]["nominal"]["path"] = "::schema::Other"
                calls.append(other)
                definitions["chelis_compiler_api::schema::Other"] = 0
            with self.subTest(change=change), self.assertRaises(GraphError):
                serialize_with_helper_owner(helper, calls, definitions)

    def test_an_ordinary_owner_is_not_a_helper(self):
        from capacity_census_wire_invocation_owners import serialize_with_helper_owner

        derive, _ = self.rows()
        self.assertIsNone(serialize_with_helper_owner(derive, [derive], {self.OWNER: 0}))


class InvocationOwnership(unittest.TestCase):
    def test_check_report_call_replays_exact_compiler_publisher_and_payload(self):
        from capacity_census_wire_invocation_owners import (
            check_report_call_owner,
            require_check_report_call,
        )

        root, call = check_report_call()
        self.assertEqual(
            check_report_call_owner(call, root),
            "check-report/chelis_compiler_api::schema::CheckResult",
        )
        self.assertIs(require_check_report_call([call], root), call)
        for calls in ([], [call, copy.deepcopy(call)]):
            with self.subTest(calls=calls), self.assertRaisesRegex(
                GraphError, "check-report publication edge"
            ):
                require_check_report_call(calls, root)
        for mutation in ("method", "receiver", "callee", "payload", "local"):
            changed = copy.deepcopy(call)
            if mutation == "method":
                changed["caller"]["definition"]["item_name"] = "format_report"
            elif mutation == "receiver":
                changed["caller"]["implementation"]["self_type"]["nominal"]["path"] = "::schema::WireCheckResult"
            elif mutation == "callee":
                changed["callee"]["path"] = "::ser::Serialize::serialize_copy"
            elif mutation == "payload":
                changed["payloads"][0]["shape"] = primitive("f64")
            else:
                changed["local_callee"] = True
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                check_report_call_owner(changed, root)

    def test_cache_load_and_save_bind_each_exact_owner_and_payload(self):
        from capacity_census_wire_invocation_owners import _local_binary_owner

        for module, name in (
            ("library_cache", "LibraryContext"),
            ("stdlib_cache", "StdLibContext"),
        ):
            for operation in ("load", "save"):
                row = {
                    "caller": {
                        "definition": {
                            "crate": "chelis_compiler_api",
                            "path": f"::{module}::load_or_build_{'library' if module == 'library_cache' else 'stdlib'}_context",
                        }
                    },
                    "callee": {
                        "crate": "chelis_compiler_api",
                        "path": f"::cache_envelope::{operation}",
                    },
                    "payloads": [
                        {"shape": nominal(f"chelis_compiler_api::{module}::{name}")}
                    ],
                }
                self.assertEqual(_local_binary_owner(row), "shared-cache-envelope")
                for change in ("caller", "callee", "payload"):
                    changed = copy.deepcopy(row)
                    if change == "payload":
                        changed["payloads"][0]["shape"] = primitive("f64")
                    elif change == "caller":
                        changed["caller"]["definition"]["path"] = (
                            "::unrelated::load_or_build_library_context"
                        )
                    else:
                        changed["callee"]["path"] = "::unrelated::" + operation
                    with (
                        self.subTest(module=module, operation=operation, change=change),
                        self.assertRaises(GraphError),
                    ):
                        _local_binary_owner(changed)

    def test_lowered_library_comparison_owns_only_exact_local_ascription_slice(self):
        from capacity_census_wire_invocation_owners import binary_call_owner

        caller = (
            "chelis_compiler_api::cache_envelope::"
            "lowered_library_payload_matches"
        )
        ascription = nominal(
            "chelis_types::infer::checked::CheckedLocalTensorAscription"
        )
        payload = {"tag": "slice", "element": ascription}
        self.assertEqual(
            binary_call_owner(caller, (payload,)),
            "native-lowered-payload-comparison",
        )

        for mutation, changed_caller, changed_payload in (
            ("caller", "other::lowered_library_payload_matches", payload),
            ("numeric", caller, primitive("f64")),
            (
                "vector",
                caller,
                nominal(
                    "alloc::vec::Vec",
                    ascription,
                    nominal("alloc::alloc::Global"),
                ),
            ),
            (
                "type",
                caller,
                {
                    "tag": "slice",
                    "element": nominal(
                        "chelis_types::infer::checked::LocalAscriptionId"
                    ),
                },
            ),
            (
                "crate",
                caller,
                {
                    "tag": "slice",
                    "element": nominal(
                        "lookalike::infer::checked::CheckedLocalTensorAscription"
                    ),
                },
            ),
        ):
            with self.subTest(mutation=mutation), self.assertRaises(GraphError):
                binary_call_owner(changed_caller, (changed_payload,))

    def test_known_carrier_and_container_do_not_authorize_arbitrary_numeric_payload(
        self,
    ):
        from capacity_census_wire_invocation_owners import require_wire_payload

        known = {"compiler::NumericScalar": 0, "compiler::Response": 1}
        carrier = nominal("compiler::NumericScalar")
        require_wire_payload(carrier, known)
        require_wire_payload(
            nominal("alloc::vec::Vec", carrier, nominal("alloc::alloc::Global")), known
        )
        require_wire_payload(nominal("compiler::Response", carrier), known)
        for payload in (
            primitive("f64"),
            primitive("u64"),
            nominal("arbitrary::Metadata"),
            nominal("compiler::Response", primitive("f64")),
            nominal(
                "alloc::vec::Vec", primitive("f64"), nominal("alloc::alloc::Global")
            ),
            {"tag": "unsupported", "display": "compiler::NumericScalar"},
        ):
            with self.subTest(payload=payload), self.assertRaises(GraphError):
                require_wire_payload(payload, known)

    def test_payload_shape_binds_width_container_and_defining_crate(self):
        from capacity_census_wire_invocation_owners import shape_key

        original = nominal("compiler::NumericScalar")
        copied = copy.deepcopy(original)
        self.assertEqual(shape_key(original), shape_key(copied))
        copied["definition"]["crate"] = "lookalike"
        self.assertNotEqual(shape_key(original), shape_key(copied))
        self.assertNotEqual(shape_key(primitive("i32")), shape_key(primitive("u32")))
        self.assertNotEqual(
            shape_key({"tag": "slice", "element": original}),
            shape_key({"tag": "array", "element": original, "length": 1}),
        )
        for malformed in (
            {},
            {"tag": "primitive", "name": "future_number"},
            {"tag": "array", "element": original, "length": True},
            {"tag": "parameter", "name": "T"},
        ):
            with self.subTest(malformed=malformed), self.assertRaises(GraphError):
                shape_key(malformed)

    def test_binary_owner_is_an_exact_use_contract_and_never_numeric_authority(self):
        from capacity_census_wire_invocation_owners import binary_call_owner

        owner = "chelis_compiler_api::cache_envelope::save"
        payload = nominal("chelis_compiler_api::stdlib_cache::StdLibContext")
        self.assertEqual(binary_call_owner(owner, (payload,)), "shared-cache-envelope")
        for caller, value in (
            ("other::save", payload),
            (owner, primitive("f64")),
            (owner, nominal("chelis_compiler_api::context::CompiledContext")),
            (owner, nominal("other::StdLibContext")),
        ):
            with (
                self.subTest(caller=caller, value=value),
                self.assertRaises(GraphError),
            ):
                binary_call_owner(caller, (value,))

    def test_invocation_receipt_cannot_be_constructed_from_a_saved_report(self):
        from capacity_census_wire_invocation_owners import VerifiedInvocations

        with self.assertRaisesRegex(TypeError, "actual compiled invocation"):
            VerifiedInvocations({"errors": [], "calls": []})


if __name__ == "__main__":
    unittest.main()
