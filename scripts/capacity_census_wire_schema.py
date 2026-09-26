"""Execution-backed Rust/Python codec graph for the C6 atomic wire cutover.

The receipt records independently checked fixed-number, tensor, report, JSON
version-envelope, source-coordinate, selected source materialization and DAG
admission cases. Complete publication discovery, other consumer entry points,
cache compatibility and live final authority remain
separate obligations.
"""

from __future__ import annotations

import json
import struct
import sys

from capacity_census_wire_adapters import CodecCase, _binary, canonical, codec_cases


def schema_cases(vocabulary: list[dict], order: tuple[str, ...]) -> list[CodecCase]:
    cases = []
    for carrier in (
        "UnitInterval",
        "SourceFloat",
        "SourceInteger",
        "NonnegativeCount",
        "NonnegativeExtent",
    ):
        floating = carrier in {"UnitInterval", "SourceFloat"}
        dtype = "f64" if floating else "int64"
        texts = (
            [
                "-0.0", "0.0", "5e-324", "2.2250738585072014e-308",
                "0.5", "1.0", "0.09891839475731867",
            ]
            if floating
            else ["0", "1", "9007199254740993", "9223372036854775807"]
        )
        if carrier == "SourceFloat":
            texts += [
                "-5e-324",
                "-1.0",
                "1.7976931348623157e308",
                "-1.7976931348623157e308",
                "-5.298400399058979e-69",
            ]
        elif carrier == "SourceInteger":
            texts += ["-1", "-9007199254740993", "-9223372036854775808"]
        for text in texts:
            value = float(text) if floating else int(text)
            element = struct.pack(">d", value).hex() if floating else value
            binary = struct.pack("<d" if floating else "<q", value).hex()
            expected = {
                "dtype": dtype,
                "elements": [element],
                "json": value,
                "binary": binary,
            }
            scalar = {"dtype": dtype, "bits" if floating else "value": element}
            for codec, input_value in (
                ("json", text),
                ("binary", binary),
                ("scalar", canonical(scalar)),
            ):
                cases.append(
                    CodecCase(
                        f"{carrier}/{codec}/{text}",
                        dtype,
                        carrier,
                        codec,
                        input_value,
                        expected,
                    )
                )
        bad_json = ["null", "true", '"1"', "{}", "[]", "1e999"]
        if not floating:
            bad_json += ["1.0", "1e0", "9223372036854775808", "-9223372036854775809"]
        if carrier == "UnitInterval":
            bad_json += ["-5e-324", "1.0000000000000002"]
        if carrier in {"NonnegativeCount", "NonnegativeExtent"}:
            bad_json += ["-1"]
        for index, text in enumerate(bad_json):
            cases.append(
                CodecCase(
                    f"{carrier}/json/reject-{index}", dtype, carrier, "json", text, None
                )
            )
        bad_scalars = [
            {"dtype": "bool", "value": True},
            {"dtype": "f32", "bits": "3f000000"},
            {"dtype": "int32", "value": 1},
        ]
        if floating:
            bad_scalars += [{"dtype": "int64", "value": 1}]
            bit_patterns = [
                "7ff0000000000000",
                "fff0000000000000",
                "7ff8000000000001",
                "7ff0000000000001",
            ]
            if carrier == "UnitInterval":
                bit_patterns += ["8000000000000001", "3ff0000000000001"]
            for pattern in bit_patterns:
                bad_scalars.append({"dtype": "f64", "bits": pattern})
                cases.append(
                    CodecCase(
                        f"{carrier}/binary/reject-{pattern}",
                        dtype,
                        carrier,
                        "binary",
                        bytes.fromhex(pattern)[::-1].hex(),
                        None,
                        "finite",
                    )
                )
        else:
            bad_scalars += [{"dtype": "f64", "bits": "3ff0000000000000"}]
            cases.append(
                CodecCase(
                    f"{carrier}/binary/truncated", dtype, carrier, "binary", "00", None
                )
            )
            if carrier in {"NonnegativeCount", "NonnegativeExtent"}:
                bad_scalars.append({"dtype": "int64", "value": -1})
                cases.append(
                    CodecCase(
                        f"{carrier}/binary/negative",
                        dtype,
                        carrier,
                        "binary",
                        "ffffffffffffffff",
                        None,
                        "nonnegative",
                    )
                )
        for index, scalar in enumerate(bad_scalars):
            cases.append(
                CodecCase(
                    f"{carrier}/scalar/reject-{index}",
                    dtype,
                    carrier,
                    "scalar",
                    canonical(scalar),
                    None,
                )
            )
    # The numeric-only subset must reject the otherwise valid canonical Bool
    # variant at both public decoding paths and direct scalar admission.
    for case in codec_cases(vocabulary, order):
        if case.carrier != "scalar" or case.expected is None:
            continue
        expected = case.expected if case.dtype != "bool" else None
        cases.append(
            CodecCase(
                "NumericScalar/" + case.identity,
                case.dtype,
                "NumericScalar",
                case.codec,
                case.input,
                expected,
                "numeric execution scalar" if expected is None else None,
            )
        )
        if case.codec == "json":
            cases.append(
                CodecCase(
                    "NumericScalar/scalar/" + case.identity,
                    case.dtype,
                    "NumericScalar",
                    "scalar",
                    case.input,
                    expected,
                    "numeric execution scalar" if expected is None else None,
                )
            )
    for dtype in vocabulary:
        name = dtype["name"]
        if dtype["kind"] == "key":
            # spec/10 section 3.2: a key tensor has no execution-value carrier
            # either, so every attempted tensor spelling of one is rejected.
            for label, storage in (
                ("no-literal-bits", {"dtype": name, "bits": ["0" * 16]}),
                ("no-literal-values", {"dtype": name, "values": [7]}),
            ):
                wire = {"shape": [1], "data": storage}
                for codec in ("json", "construct"):
                    cases.append(
                        CodecCase(
                            f"TensorValue/{codec}/{name}/{label}",
                            name,
                            "TensorValue",
                            codec,
                            canonical(wire),
                            None,
                        )
                    )
            continue
        value = (
            "8" + "0" * (dtype["width"] * 2 - 1)
            if dtype["kind"] == "float"
            else 2**53 + 1
            if name == "int64"
            else 1
            if dtype["kind"] == "integer"
            else True
        )
        for label, shape, elements, admitted in (
            ("scalar", [], [value], True),
            ("one", [1], [value], True),
            ("matrix", [1, 2], [value, value], True),
            ("empty", [0], [], True),
            ("late-zero", [2**63 - 1, 3, 0], [], True),
            ("count-mismatch", [2], [value], False),
            ("negative-extent", [-1], [], False),
            ("product-overflow", [2**63 - 1, 3], [value], False),
        ):
            storage = {
                "dtype": name,
                "bits" if dtype["kind"] == "float" else "values": elements,
            }
            wire = {"shape": shape, "data": storage}
            binary = (
                len(shape).to_bytes(8, "little")
                + b"".join(d.to_bytes(8, "little", signed=True) for d in shape)
                + _binary(order.index(name), dtype, elements, True)
            )
            expected = (
                {
                    "dtype": name,
                    "elements": elements,
                    "shape": shape,
                    "json": wire,
                    "binary": binary.hex(),
                }
                if admitted
                else None
            )
            for codec, input_value in (
                ("json", canonical(wire)),
                ("binary", binary.hex()),
                ("construct", canonical(wire)),
            ):
                cases.append(
                    CodecCase(
                        f"TensorValue/{codec}/{name}/{label}",
                        name,
                        "TensorValue",
                        codec,
                        input_value,
                        expected,
                        "tensor" if expected is None else None,
                    )
                )
    return cases


def diagnostic_cases():
    """Actual producer omission, exact report values, and consumer admission."""
    cases = []
    fields = sorted(
        (
            "kind",
            "message",
            "severity",
            "expected",
            "got",
            "suggestions",
            "span",
            "deep_path",
            "span_id",
        )
    )
    for value in (0.5, -0.0):
        wire = {
            "kind": "unsupported_feature",
            "message": "unsupported: a non-literal window list for `reduce_window_max` on the "
            "compiled-backend lowering of `reduce_window_*` (lowering); unimplemented "
            "chelis#1058: window and stride lists must be integer literals for the compiled "
            "lane today; a runtime-parameterized window previously lowered to a silent no-op; "
            "chelis#1058 owns compiled runtime-list support at source span `surf:82..85`",
            "severity": value,
            "expected": "literal window",
            "got": "runtime window",
            "suggestions": ["use literals"],
            "span": {"span": "range", "offset": 9007199254740993, "len": 0},
            "deep_path": {"def_qualified_name": "f", "path": "0.1"},
            "span_id": "projection-witness",
        }
        bits = struct.pack(">d", value).hex()
        for codec in ("construct", "json"):
            expected = {"wire": wire, "severity_bits": bits}
            if codec == "construct":
                expected.update(
                    internal_identity=True,
                    envelope_matches=True,
                    producer_schema_fields=fields,
                    consumer_schema_fields=fields,
                )
            cases.append(
                CodecCase(
                    f"Diagnostic/{codec}/{value}",
                    "report",
                    "Diagnostic",
                    codec,
                    canonical(value if codec == "construct" else wire),
                    expected,
                )
            )
    for value in (-0.1, 1.1):
        for codec in ("construct", "json"):
            request = (
                value
                if codec == "construct"
                else {
                    "kind": "unsupported_feature",
                    "message": "invalid",
                    "severity": value,
                }
            )
            cases.append(
                CodecCase(
                    f"Diagnostic/{codec}/reject-{value}",
                    "report",
                    "Diagnostic",
                    codec,
                    canonical(request),
                    None,
                    "finite f64 in [0,1]",
                )
            )
    cases.append(
        CodecCase(
            "Diagnostic/json/missing-severity",
            "report",
            "Diagnostic",
            "json",
            '{"kind":"unsupported_feature","message":"invalid"}',
            None,
            "missing field `severity`",
        )
    )
    return cases


def report_cases():
    cases = []

    def add(carrier, codec, name, value, expected, error=None):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/{name}",
                "report",
                carrier,
                codec,
                canonical(value),
                expected,
                error,
            )
        )

    for counts in (
        (0, 0, 0),
        (3, 2, 5),
        (9007199254740993, 0, 9007199254740993),
        (5, 0, 3),
        (3, 3, 5),
        (-1, 1, 0),
    ):
        value = {
            "score": 0.5,
            "components": {
                name: 1.0 for name in ("parse", "structure", "names", "types")
            },
            "typed_nodes": counts[0],
            "untyped_nodes": counts[1],
            "total_nodes": counts[2],
            "unresolved_names": [],
            "errors": [],
        }
        valid = (
            min(counts) >= 0
            and counts[0] <= counts[2]
            and counts[1] == counts[2] - counts[0]
        )
        expected = (
            {
                "score": 0.5,
                "components": value["components"],
                "counts": list(counts),
                "unresolved_names": [],
                "error_count": 0,
                "inferred_signatures": None,
            }
            if valid
            else None
        )
        for codec in ("json", "construct"):
            add(
                "CheckResult",
                codec,
                "counts-" + str(counts),
                value,
                expected,
                "inconsistent report counts"
                if not valid and min(counts) >= 0
                else None,
            )
    for indices in ((), (0,), (0, 1, 2), (1,), (0, 0), (0, 2), (18446744073709551615,)):
        values = [
            {
                "index": index,
                "name": "x",
                "written": False,
                "inferred_read_only": False,
                "checked_type": "unit",
                "display_type": "unit",
                "checked_type_structured": {"kind": "unit"},
                "display_type_structured": {"kind": "unit"},
            }
            for index in indices
        ]
        valid = indices == tuple(range(len(indices)))
        for codec in ("json", "construct"):
            add(
                "OrderedInferredParameters",
                codec,
                "indices-" + str(indices),
                values,
                values if valid else None,
                None if valid else "owning list position",
            )
    return cases


from dataclasses import asdict, dataclass, replace
import hashlib
import os
from pathlib import Path
import subprocess

from capacity_census_graph import Definition, Edge, GraphError, _attributes, _serde
from capacity_census_wire_operations import (
    OperationContract,
    operation_contracts,
    validate_numeric_operation_fields,
)
from capacity_census_wire_structural import (
    StructuralFieldContract,
    validate_source_carriers,
    validate_structural_execution,
    validate_structural_fields,
)
from capacity_census_wire_authority import LeafClassification, classification_plan
from capacity_census_wire_adapters import (
    VerifiedCodec,
    _CodecShapeGraph,
    _TYPES,
    _WIRE,
    _require,
    _target_lease,
    _verify_canonical_codec,
    check_observations,
    source_identity,
)

_SCHEMA = "chelis_compiler_api::schema::"
_NUMBER_SOURCE = "crates/chelis-compiler-api/src/schema/numbers.rs"
_EXECUTION_SOURCE = "crates/chelis-compiler-api/src/schema/execution.rs"
_FIXED = {
    _SCHEMA + "numbers::" + name: primitive
    for name, primitive in (
        ("UnitInterval", "f64"),
        ("SourceFloat", "f64"),
        ("SourceInteger", "i64"),
        ("NonnegativeCount", "i64"),
        ("NonnegativeExtent", "i64"),
    )
}
_NUMERIC = _SCHEMA + "execution::NumericScalar"
_TENSOR = _SCHEMA + "TensorValue"
_TENSOR_WIRE = _SCHEMA + "execution::TensorWire"
_ORDERED = _SCHEMA + "OrderedInferredParameters"
_REPORTS = {
    _SCHEMA + "CheckResult": "Diagnostic",
    _SCHEMA + "WireCheckResult": "WireDiagnostic",
}
_REPORT_WIRE = _SCHEMA + "reports::ReportWire"
_SCHEMA_SOURCE = "crates/chelis-compiler-api/src/schema.rs"
_REPORT_SOURCE = "crates/chelis-compiler-api/src/schema/reports.rs"
_ENVELOPE_SOURCE = "crates/chelis-compiler-api/src/schema/envelopes.rs"
_ENVELOPE = _SCHEMA + "envelopes::"
_ARTIFACT = _SCHEMA + "artifact::"
_ARTIFACT_SOURCE = "crates/chelis-compiler-api/src/schema/artifact.rs"
_ENVELOPES = {
    _SCHEMA + name
    for name in ("EvalResult", "WireDag", "WireApiEnvelope", "WireBatchResult")
}


class _SchemaShapeGraph(_CodecShapeGraph):
    """Shape inspection only; source/report role authority is not inferred."""

    def __init__(self, documents, vocabulary):
        super().__init__(documents, vocabulary)
        self._codec_overrides = {}

    def _codec(self, crate, inner):
        key = (crate, id(inner))
        return (
            self._codec_overrides[key]
            if key in self._codec_overrides
            else super()._codec(crate, inner)
        )

    def _schema_reference(self, identity, arguments=()):
        _require(identity in self.locations, "missing actual codec mirror " + identity)
        crate, item_id = self.locations[identity]
        self._definition(crate, item_id, identity)
        return ("reference", identity, arguments)

    def _diagnostic_codec(self, crate, item, identity, direction):
        """Bind both actual derive implementations to this exact report owner."""
        body = item["inner"].get("struct")
        _require(
            body is not None and not self._parameters(body) and not self._serde(item),
            "diagnostic declaration changed",
        )
        codec = self._serde_implementations(
            crate, body, set(), _SCHEMA_SOURCE, frozenset({direction})
        )
        schema = []
        for impl_id in body.get("impls", []):
            implementation = self._item(crate, impl_id)
            inner = implementation.get("inner", {}).get("impl", {})
            trait = inner.get("trait")
            if not trait:
                continue
            path = self.documents[crate]["paths"].get(str(trait["id"]), {}).get("path")
            is_schema = path == ["schemars", "JsonSchema"]
            is_serde = path in [
                [owner, module, name]
                for owner in ("serde", "serde_core")
                for module, name in (("ser", "Serialize"), ("de", "Deserialize"))
            ] or path in [["serde", "Serialize"], ["serde", "Deserialize"]]
            if not is_schema and not is_serde:
                continue
            receiver, arguments = self._nominal(crate, inner.get("for", {}))
            _require(
                receiver == identity
                and arguments
                in (None, {"angle_bracketed": {"args": [], "constraints": []}}),
                "diagnostic codec receiver changed",
            )
            methods = tuple(
                self._item(crate, method) for method in inner.get("items", [])
            )
            expected = (
                {"schema_name", "schema_id", "json_schema"}
                if is_schema
                else {direction.lower()}
            )
            _require(
                len(methods) == len(expected)
                and {method.get("name") for method in methods} == expected
                and all("function" in method.get("inner", {}) for method in methods),
                "diagnostic codec methods changed",
            )
            if is_schema:
                _require(
                    "#[automatically_derived]" in _attributes(implementation)
                    and not inner.get("blanket_impl")
                    and not inner.get("is_negative")
                    and not inner.get("is_synthetic")
                    and all(
                        (value.get("span") or {}).get("filename") == _SCHEMA_SOURCE
                        for value in (implementation, *methods)
                    ),
                    "diagnostic schema implementation provenance changed",
                )
                schema.append(
                    (
                        "schemars::JsonSchema",
                        _SCHEMA_SOURCE,
                        tuple(method["name"] for method in methods),
                    )
                )
        _require(
            len(schema) == 1, "missing or duplicate diagnostic schema implementation"
        )
        return (codec, tuple(schema))

    def _diagnostic_definition(self, crate, item_id, identity):
        item = self._item(crate, item_id)
        codec = self._diagnostic_codec(crate, item, identity, "Serialize")
        plain = item["inner"]["struct"]["kind"].get("plain")
        _require(
            plain is not None and not plain["has_stripped_fields"],
            "diagnostic fields are missing or stripped",
        )
        fields = [self._item(crate, field) for field in plain["fields"]]
        sidecars = [field for field in fields if field.get("name") == "unsupported"]
        _require(
            len(sidecars) == 1, "diagnostic requires its exact off-wire identity field"
        )
        sidecar = sidecars[0]
        _require(
            sidecar.get("visibility") == "crate"
            and (sidecar.get("span") or {}).get("filename") == _SCHEMA_SOURCE
            and set(_attributes(sidecar)) == {"#[serde(skip)]", "#[schemars(skip)]"},
            "diagnostic off-wire field visibility or omission changed",
        )
        ty = sidecar.get("inner", {}).get("struct_field", {})
        nominal_chain = []
        for expected in ("core::option::Option", "alloc::boxed::Box"):
            nominal, arguments = self._nominal(crate, ty)
            _require(
                nominal == expected
                and isinstance(arguments, dict)
                and set(arguments) == {"angle_bracketed"},
                "diagnostic off-wire container changed",
            )
            arguments = arguments["angle_bracketed"]
            _require(
                not arguments.get("constraints")
                and len(arguments.get("args", [])) == 1
                and set(arguments["args"][0]) == {"type"},
                "diagnostic off-wire container arguments changed",
            )
            nominal_chain.append(nominal)
            ty = arguments["args"][0]["type"]
        nominal, arguments = self._nominal(crate, ty)
        _require(
            nominal == "chelis_types::unsupported::Unsupported"
            and arguments
            in (None, {"angle_bracketed": {"args": [], "constraints": []}}),
            "diagnostic off-wire payload changed",
        )
        owner, payload_id, _ = self._resolve(crate, ty["resolved_path"]["id"])
        payload = self._item(owner, payload_id).get("inner", {}).get("struct")
        _require(
            payload is not None and not self._has_serde(owner, payload),
            "diagnostic off-wire payload requires its non-serde defining artifact",
        )
        nominal_chain.append(nominal)
        self.definitions[identity] = None
        edges = []
        members = []
        for field in fields:
            if field is sidecar:
                continue
            name = field.get("name")
            _require(
                name and name not in {member[0] for member in members},
                "missing or duplicate diagnostic field",
            )
            options = self._serde(field)
            resolved = self._type(crate, field["inner"]["struct_field"], {})
            self._validate_field_serde(options, resolved)
            edges.append(Edge(identity + "." + name, resolved, options))
            members.append((name, options))
        consumer_identity = _SCHEMA + "WireDiagnostic"
        self._schema_reference(consumer_identity)
        consumer = self.definitions[consumer_identity]
        consumer_owner, consumer_id = self.locations[consumer_identity]
        consumer_codec = self._diagnostic_codec(
            consumer_owner,
            self._item(consumer_owner, consumer_id),
            consumer_identity,
            "Deserialize",
        )
        _require(
            tuple((edge.path.rsplit(".", 1)[-1], edge.type) for edge in edges)
            == tuple(
                (edge.path.rsplit(".", 1)[-1], edge.type) for edge in consumer.edges
            ),
            "diagnostic producer and decoder fields disagree",
        )
        self.definitions[identity] = Definition(
            identity,
            "struct",
            (),
            (),
            (
                ("", (), ("plain", tuple(members))),
                (
                    "off-wire-derived-field",
                    "unsupported",
                    tuple(nominal_chain),
                    "crate",
                    ("serde(skip)", "schemars(skip)"),
                    codec,
                    consumer_codec,
                ),
            ),
            tuple(edges),
            "schema-codec:" + canonical(codec),
        )

    def _artifact_definition(self, crate, item_id, identity):
        item = self._item(crate, item_id)
        if identity == _ARTIFACT + "ArtifactAbiVersion":
            body = item["inner"].get("enum")
            _require(
                body is not None and not self._parameters(body),
                "artifact ABI must be a closed enum",
            )
            attrs = tuple(a for a in _attributes(item) if a.startswith("#[serde"))
            _require(
                attrs == ('#[serde(try_from = "u32", into = "u32")]',),
                "artifact ABI requires an exact uint32 discriminator codec",
            )
            _require(
                not body.get("has_stripped_variants") and len(body["variants"]) == 1,
                "artifact ABI closed version vocabulary changed",
            )
            variant = self._item(crate, body["variants"][0])
            _require(
                variant["name"] == "V2"
                and variant["inner"]["variant"]["kind"] == "plain"
                and not self._serde(variant),
                "artifact ABI requires the unit V2 variant",
            )
            codec = self._serde_implementations(crate, body, set(), _ARTIFACT_SOURCE)
            self.definitions[identity] = Definition(
                identity,
                "enum",
                (),
                attrs,
                (("V2", (), ("unit", ())),),
                (Edge(identity + ".$version", ("primitive", "u32"), ()),),
                "artifact-version-codec:" + codec,
            )
            return
        body = item["inner"].get("struct")
        _require(
            body is not None and not self._parameters(body),
            "artifact manifest must be a concrete struct",
        )
        self._codec_overrides[(crate, id(body))] = (
            "artifact-manifest-codec:"
            + self._serde_implementations(
                crate, body, {"Deserialize"}, _ARTIFACT_SOURCE
            )
        )
        super()._definition(crate, item_id, identity)
        native = self.definitions[identity]
        version = ("reference", _ARTIFACT + "ArtifactAbiVersion", ())
        string = ("atomic", "alloc::string::String")
        spec = ("reference", "chelis_compiler_api::compiler::ExecutionTensorSpec", ())
        expected = [
            ("abi_version", version),
            ("target", ("reference", _SCHEMA + "CompileTarget", ())),
            ("host_entry_name", string),
            ("device_entry_name", ("container", "core::option::Option", (string,))),
            ("inputs", ("container", "alloc::vec::Vec", (spec,))),
            ("outputs", ("container", "alloc::vec::Vec", (spec,))),
            ("symbolic_dims", ("container", "alloc::vec::Vec", (string,))),
            ("source_path", string),
            ("source_hash", string),
            ("runtime_sha256", string),
            ("library_sha256", string),
        ]

        def fields(definition):
            return [(e.path.rsplit(".", 1)[-1], e.type) for e in definition.edges]

        _require(
            not native.serde and fields(native) == expected,
            "artifact manifest fields changed",
        )
        helpers = []
        for name, expected_fields in (
            ("ArtifactAbiHeader", [("abi_version", version)]),
            ("CompiledArtifactManifestFields", expected),
        ):
            mirror = _ARTIFACT + name
            ty = self._schema_reference(mirror)
            owner, target = self.locations[mirror]
            helper = self._item(owner, target)
            self._private(helper, "::schema::artifact")
            self._serde_implementations(
                owner,
                helper["inner"]["struct"],
                set(),
                _ARTIFACT_SOURCE,
                frozenset({"Deserialize"}),
            )
            _require(
                not self.definitions[mirror].serde
                and fields(self.definitions[mirror]) == expected_fields,
                "artifact manifest header or metadata codec mirror changed",
            )
            helpers.append(Edge(identity + ".$" + name, ty, ()))
        self.definitions[identity] = replace(
            native, edges=native.edges + tuple(helpers)
        )

    def _report_definition(self, crate, item_id, identity):
        item = self._item(crate, item_id)
        body = item["inner"]["struct"]
        if identity == _ORDERED:
            self._codec_overrides[(crate, id(body))] = (
                "schema-codec:"
                + self._serde_implementations(
                    crate, body, {"Deserialize"}, _SCHEMA_SOURCE
                )
            )
            super()._definition(crate, item_id, identity)
            definition = self.definitions[identity]
            expected = (
                "container",
                "alloc::vec::Vec",
                (("reference", _SCHEMA + "WireInferredParameter", ()),),
            )
            _require(
                definition.serde == (("transparent", True),)
                and len(definition.edges) == 1
                and definition.edges[0].type == expected,
                "ordered parameter list carrier changed",
            )
            field = self._item(crate, body["kind"]["tuple"][0])
            self._private(field, "::schema")
            parameter = self.definitions[_SCHEMA + "WireInferredParameter"]
            _require(
                next(
                    (e.type for e in parameter.edges if e.path.endswith(".index")), None
                )
                == ("primitive", "u64"),
                "ordered parameter reference width changed",
            )
            return
        direction = "Serialize" if identity.endswith("::CheckResult") else "Deserialize"
        self._codec_overrides[(crate, id(body))] = (
            "schema-codec:"
            + self._serde_implementations(
                crate, body, {direction}, _REPORT_SOURCE, frozenset({direction})
            )
        )
        super()._definition(crate, item_id, identity)
        native = self.definitions[identity]
        diagnostic = ("reference", _SCHEMA + _REPORTS[identity], ())
        mirror_type = self._schema_reference(_REPORT_WIRE, (diagnostic,))
        mirror = self.definitions[_REPORT_WIRE]
        _require(
            mirror.parameters == ("D",)
            and mirror.codec == "serde-derived"
            and not mirror.serde,
            "report codec mirror generic/serde identity changed",
        )
        owner, target = self.locations[_REPORT_WIRE]
        mirror_item = self._item(owner, target)
        self._private(mirror_item, "::schema::reports")
        self._serde_implementations(
            owner, mirror_item["inner"]["struct"], set(), _REPORT_SOURCE
        )

        def substitute(value):
            if value == ("generic", "D"):
                return diagnostic
            return tuple(
                substitute(child) if isinstance(child, tuple) else child
                for child in value
            )

        projected = [
            (edge.path.rsplit(".", 1)[-1], substitute(edge.type))
            for edge in mirror.edges
        ]
        actual = [(edge.path.rsplit(".", 1)[-1], edge.type) for edge in native.edges]
        _require(
            projected == actual and not native.serde,
            "report public/private codec fields disagree",
        )
        expected_names = (
            "score",
            "components",
            "typed_nodes",
            "untyped_nodes",
            "total_nodes",
            "unresolved_names",
            "inferred_signatures",
            "errors",
        )
        _require(
            tuple(name for name, _ in actual) == expected_names,
            "report admitted field set changed",
        )
        for name, ty in actual:
            if name == "score" or name.endswith("_nodes"):
                required = "UnitInterval" if name == "score" else "NonnegativeCount"
                _require(
                    ty == ("reference", _SCHEMA + "numbers::" + required, ()),
                    "report fixed-number carrier changed",
                )
        self.definitions[identity] = replace(
            native, edges=native.edges + (Edge(identity + ".$wire", mirror_type, ()),)
        )

    def _metadata_extent_definition(self, crate, item_id, identity):
        super()._definition(crate, item_id, identity)
        definition = self.definitions[identity]
        extent = ("reference", _SCHEMA + "numbers::NonnegativeExtent", ())
        if identity == "chelis_compiler_api::compiler::ExecutionDim":
            expected = [
                (
                    "name",
                    (
                        "container",
                        "core::option::Option",
                        (("atomic", "alloc::string::String"),),
                    ),
                ),
                ("size", ("container", "core::option::Option", (extent,))),
            ]
            actual = [
                (edge.path.rsplit(".", 1)[-1], edge.type) for edge in definition.edges
            ]
            _require(
                actual == expected and definition.codec == "serde-derived",
                "execution metadata extent must retain its optional sealed int64 carrier",
            )
        else:
            _require(
                dict(definition.serde) == {"tag": "kind", "rename_all": "snake_case"}
                and tuple(v[0] for v in definition.layout)
                == ("Name", "Var", "Lit", "Wildcard", "Rank"),
                "inference dimension closed structural variants changed",
            )
            expected = {
                identity + "::Lit.size": extent,
                identity + "::Var.id": ("primitive", "u32"),
                identity + "::Rank.id": ("primitive", "u32"),
            }
            for path, ty in expected.items():
                _require(
                    next((e.type for e in definition.edges if e.path == path), None)
                    == ty,
                    "inference dimension extent/namespace carrier changed",
                )

    def _result_reference_definition(self, crate, item_id, identity):
        item = self._item(crate, item_id)
        body = item["inner"]["struct"]
        self._codec_overrides[(crate, id(body))] = (
            "result-reference-codec:"
            + self._serde_implementations(
                crate, body, {"Serialize", "Deserialize"}, _ENVELOPE_SOURCE
            )
        )
        super()._definition(crate, item_id, identity)
        native = self.definitions[identity]
        named = (
            "container",
            "alloc::collections::btree::map::BTreeMap",
            (("atomic", "alloc::string::String"), ("primitive", "u64")),
        )
        expected = [("dag", ("reference", _SCHEMA + "WireDag", ()))]
        lower = identity == _SCHEMA + "LowerResult"
        expected += (
            [("named_roots", named)]
            if lower
            else [
                ("output_node", ("primitive", "u64")),
                ("grad_nodes_by_name", named),
                ("forward_nodes_by_name", named),
            ]
        )

        def fields(definition):
            return [
                (
                    edge.path.rsplit(".", 1)[-1],
                    edge.type[-1]
                    if edge.type[0] == "borrow" and not edge.type[1]
                    else edge.type,
                )
                for edge in definition.edges
            ]

        _require(
            not native.serde and fields(native) == expected,
            "result reference public carrier changed",
        )
        mirrors = []
        for suffix, direction in (("", "Deserialize"), ("Ref", "Serialize")):
            mirror = _ENVELOPE + ("LowerFields" if lower else "GradFields") + suffix
            ty = self._schema_reference(mirror)
            owner, target = self.locations[mirror]
            helper = self._item(owner, target)
            self._private(helper, "::schema::envelopes")
            self._serde_implementations(
                owner,
                helper["inner"]["struct"],
                set(),
                _ENVELOPE_SOURCE,
                frozenset({direction}),
            )
            _require(
                not self.definitions[mirror].serde
                and fields(self.definitions[mirror]) == expected,
                "result reference private codec mirror changed",
            )
            mirrors.append(Edge(identity + ".$" + direction, ty, ()))
        self.definitions[identity] = replace(
            native, edges=native.edges + tuple(mirrors)
        )

    def _location_definition(self, crate, item_id, identity):
        super()._definition(crate, item_id, identity)
        definition = self.definitions[identity]
        item = self._item(crate, item_id)
        body = item["inner"][definition.kind]
        self._serde_implementations(crate, body, set(), _SCHEMA_SOURCE)
        diagnostic = identity.endswith("::DiagnosticSpan")
        expected_serde = {"deny_unknown_fields": True}
        if diagnostic:
            expected_serde.update(tag="span", rename_all="snake_case")
            expected = {
                identity + "::Point.offset",
                identity + "::Range.offset",
                identity + "::Range.len",
            }
            _require(
                tuple(sorted(v[0] for v in definition.layout)) == ("Point", "Range"),
                "source location requires distinct measured range and point",
            )
        else:
            expected = {identity + ".offset", identity + ".len"}
        _require(
            dict(definition.serde) == expected_serde and not definition.parameters,
            "source coordinate serde discriminator/strictness changed",
        )
        _require(
            {edge.path for edge in definition.edges} == expected
            and all(
                edge.type == ("primitive", "u64") and not edge.serde
                for edge in definition.edges
            ),
            "source coordinates require exact UTF-8 offset/extent uint64 fields",
        )

    def _envelope_definition(self, crate, item_id, identity):
        item = self._item(crate, item_id)
        body = item["inner"].get("struct", item["inner"].get("enum"))
        both = identity in {_SCHEMA + "EvalResult", _SCHEMA + "WireDag"}
        required = frozenset({"Serialize", "Deserialize"} if both else {"Deserialize"})
        source = _SCHEMA_SOURCE if identity == _SCHEMA + "WireDag" else _ENVELOPE_SOURCE
        self._codec_overrides[(crate, id(body))] = (
            "json-envelope-codec:"
            + self._serde_implementations(crate, body, set(required), source, required)
        )
        super()._definition(crate, item_id, identity)
        native = self.definitions[identity]
        _require(not native.serde, "custom JSON envelope acquired serde attributes")
        helpers = []
        if both:
            dag = identity == _SCHEMA + "WireDag"
            stem = _SCHEMA + "WireDagFields" if dag else _ENVELOPE + "ExecutionFields"
            helpers += [
                ("decode", self._schema_reference(stem)),
                ("encode", self._schema_reference(stem + "Ref")),
                ("header", self._schema_reference(_ENVELOPE + "VersionHeader")),
            ]
            expected = [
                ("schema_version", ("primitive", "u32")),
                (
                    "nodes" if dag else "roots",
                    (
                        "container",
                        "alloc::vec::Vec",
                        (
                            (
                                "reference",
                                _SCHEMA + ("WireDagNode" if dag else "EvaluatedRoot"),
                                (),
                            ),
                        ),
                    ),
                ),
            ]
            if dag:
                expected.append(
                    ("roots", ("container", "alloc::vec::Vec", (("primitive", "u64"),)))
                )
            else:
                expected += [
                    ("manifest", ("reference", _SCHEMA + "RootManifestResult", ())),
                    (
                        "transcript",
                        (
                            "container",
                            "alloc::vec::Vec",
                            (("atomic", "alloc::string::String"),),
                        ),
                    ),
                ]

            def fields(definition):
                def unborrow(ty):
                    if ty[0] != "borrow":
                        return ty
                    _require(not ty[1], "mutable encoder mirror reference")
                    inner = ty[-1]
                    return (
                        ("container", "alloc::vec::Vec", (inner[1],))
                        if inner[0] == "slice"
                        else inner
                    )

                return [
                    (e.path.rsplit(".", 1)[-1], unborrow(e.type))
                    for e in definition.edges
                ]

            _require(
                fields(native) == expected, "versioned envelope public fields changed"
            )
            for mirror in (stem, stem + "Ref"):
                owner, target = self.locations[mirror]
                helper = self._item(owner, target)
                self._private(helper, "::schema" if dag else "::schema::envelopes")
                direction = (
                    {"Serialize"}
                    if mirror.endswith("Ref")
                    else ({"Serialize", "Deserialize"} if dag else {"Deserialize"})
                )
                self._serde_implementations(
                    owner,
                    helper["inner"]["struct"],
                    set(),
                    source,
                    frozenset(direction),
                )
                _require(
                    fields(self.definitions[mirror]) == expected,
                    "versioned envelope mirror fields changed",
                )
            header = self.definitions[_ENVELOPE + "VersionHeader"]
            _require(
                len(header.edges) == 1
                and header.edges[0].path.endswith(".schema_version")
                and header.edges[0].type
                == ("container", "core::option::Option", (("primitive", "u32"),))
                and not header.serde
                and not header.edges[0].serde,
                "version header must retain exact explicit uint32 discriminator",
            )
        elif identity == _SCHEMA + "WireApiEnvelope":
            helpers += [
                ("header", self._schema_reference(_ENVELOPE + "ResponseHeader")),
                (
                    "producer",
                    self._schema_reference(
                        _SCHEMA + "ApiEnvelope", (("generic", "T"),)
                    ),
                ),
            ]
            expected = (
                (
                    "Success",
                    ("reference", _SCHEMA + "WireApiSuccess", (("generic", "T"),)),
                ),
                ("Failure", ("reference", _SCHEMA + "WireApiFailure", ())),
            )
            _require(
                native.parameters == ("T",)
                and tuple(
                    (v[0], e.type)
                    for v, e in zip(native.layout, native.edges, strict=True)
                )
                == expected,
                "response envelope closed dispatch/payload contract changed",
            )
            header = self.definitions[_ENVELOPE + "ResponseHeader"]
            _require(
                len(header.edges) == 1
                and header.edges[0].path.endswith(".ok")
                and header.edges[0].type == ("primitive", "bool")
                and not header.serde,
                "response header requires actual boolean discriminator",
            )
        else:
            helpers += [
                ("header", self._schema_reference(_ENVELOPE + "BatchHeader")),
                ("producer", self._schema_reference(_SCHEMA + "BatchResult")),
            ]
            kind = self.definitions[_ENVELOPE + "BatchKind"]
            names = (
                "Parse",
                "Desugar",
                "Check",
                "Lower",
                "Compile",
                "Eval",
                "Grad",
                "Validate",
                "Decompile",
            )
            _require(
                tuple(v[0] for v in native.layout) == names
                and tuple(v[0] for v in kind.layout) == names
                and not kind.edges
                and kind.serde == (("rename_all", "snake_case"),),
                "batch envelope closed dispatch vocabulary changed",
            )
            for name, edge in zip(names, native.edges, strict=True):
                result = "WireCheckResult" if name == "Check" else name + "Result"
                _require(
                    edge.type
                    == (
                        "reference",
                        _SCHEMA + "WireApiEnvelope",
                        (("reference", _SCHEMA + result, ()),),
                    ),
                    "batch envelope selected payload type changed",
                )
        self.definitions[identity] = replace(
            native,
            layout=native.layout + (("actual-json-codec-paths", tuple(helpers)),),
            edges=native.edges
            + tuple(Edge(identity + ".$" + name, ty, ()) for name, ty in helpers),
        )

    def _serde(self, item):
        # These built-in derive options affect presence/shape, not numeric
        # authority. Every field remains reachable, including omitted fields.
        options = _serde(
            item,
            flags={"deny_unknown_fields", "default", "transparent", "untagged", "skip"},
            values={
                "tag",
                "content",
                "rename",
                "rename_all",
                "skip_serializing_if",
                "default",
                "deserialize_with",
            },
        )
        if "deserialize_with" in dict(options):
            location = self.locations.get(_SCHEMA + "WireDagNode")
            fields = ()
            if location is not None:
                owner = self._item(*location)
                fields = (
                    owner.get("inner", {}).get("struct", {}).get("kind", {})
                    .get("plain", {}).get("fields", ())
                )
            _require(
                options == (("deserialize_with", "require_explicit_span"),)
                and item.get("name") == "span_id"
                and location is not None
                and any(self._item(location[0], field) is item for field in fields),
                "required span decoder belongs only to the exact WireDagNode field",
            )
        return options

    def _validate_field_serde(self, options, ty):
        options = dict(options)
        if "deserialize_with" in options:
            _require(
                ty == (
                    "container", "core::option::Option",
                    (("atomic", "alloc::string::String"),),
                ),
                "required span decoder requires optional text without numeric payload",
            )
        default = options.get("default")
        if isinstance(default, str):
            _require(
                default == "default_true" and ty == ("primitive", "bool"),
                "unsupported or type-mismatched serde default function",
            )
        if options.get("skip"):
            _require(
                ty
                == (
                    "container",
                    "core::option::Option",
                    (("atomic", "alloc::string::String"),),
                ),
                "only the supported nonnumeric optional text may be skipped",
            )
        predicate = options.get("skip_serializing_if")
        if predicate is None:
            return
        supported = (
            (
                predicate == "Option::is_none"
                and ty[:2] == ("container", "core::option::Option")
            )
            or (
                predicate == "Vec::is_empty"
                and ty[:2] == ("container", "alloc::vec::Vec")
            )
            or (
                predicate == "BTreeMap::is_empty"
                and ty[:2] == ("container", "alloc::collections::btree::map::BTreeMap")
            )
            or (
                predicate == "<[String]>::is_empty"
                and ty[0] == "borrow"
                and not ty[1]
                and ty[-1] == ("slice", ("atomic", "alloc::string::String"))
            )
        )
        _require(supported, "unsupported or type-mismatched serde omission predicate")

    def _plain_fields(self, crate, item):
        body = item["inner"].get("struct")
        _require(
            body is not None and not self._parameters(body),
            "unsupported schema carrier struct",
        )
        plain = body["kind"].get("plain")
        _require(
            plain is not None and not plain["has_stripped_fields"],
            "missing complete schema fields",
        )
        result = []
        for field_id in plain["fields"]:
            field = self._item(crate, field_id)
            _require(not _serde(field), "unexpected custom schema field attributes")
            ty = self._type(crate, field["inner"]["struct_field"], {})
            result.append((field["name"], ty))
        return tuple(result)

    def _peel_alias(self, ty):
        seen = set()
        while ty[0] == "reference":
            definition = self.definitions[ty[1]]
            _require(definition is not None, "unresolved alias during schema admission")
            if definition.kind != "type_alias":
                break
            _require(
                ty[1] not in seen
                and not definition.parameters
                and len(definition.edges) == 1,
                "unsupported schema carrier alias",
            )
            seen.add(ty[1])
            ty = definition.edges[0].type
        return ty

    def _definition(self, crate, item_id, identity):
        if identity in self.definitions:
            return
        if identity == _SCHEMA + "Diagnostic":
            return self._diagnostic_definition(crate, item_id, identity)
        if identity in {
            _ARTIFACT + "ArtifactAbiVersion",
            _ARTIFACT + "CompiledArtifactManifest",
        }:
            return self._artifact_definition(crate, item_id, identity)
        if identity in {
            "chelis_compiler_api::compiler::ExecutionDim",
            _SCHEMA + "WireInferredDim",
        }:
            return self._metadata_extent_definition(crate, item_id, identity)
        if identity in {_SCHEMA + "LowerResult", _SCHEMA + "GradResult"}:
            return self._result_reference_definition(crate, item_id, identity)
        if identity in {_SCHEMA + "Span", _SCHEMA + "DiagnosticSpan"}:
            return self._location_definition(crate, item_id, identity)
        if identity in _ENVELOPES:
            return self._envelope_definition(crate, item_id, identity)
        if identity in _REPORTS or identity == _ORDERED:
            return self._report_definition(crate, item_id, identity)
        if identity not in _FIXED and identity not in {_NUMERIC, _TENSOR, _TENSOR_WIRE}:
            return super()._definition(crate, item_id, identity)
        item = self._item(crate, item_id)
        body = item["inner"].get("struct")
        _require(
            body is not None and not self._parameters(body),
            "schema codec carrier must be an exact nongeneric struct",
        )
        if identity in _FIXED or identity == _NUMERIC:
            ids = body["kind"].get("tuple")
            _require(
                isinstance(ids, list) and len(ids) == 1,
                "fixed schema codec must wrap one sealed scalar",
            )
            field = self._item(crate, ids[0])
            self._private(
                field,
                "::schema::numbers" if identity in _FIXED else "::schema::execution",
            )
            _require(not _serde(field), "fixed schema codec field attributes changed")
            ty = self._type(crate, field["inner"]["struct_field"], {})
            _require(
                self._peel_alias(ty) == ("reference", _TYPES + "ScalarValue", ()),
                "fixed schema codec must use the canonical sealed scalar",
            )
            if identity in _FIXED:
                _require(not _serde(item), "fixed JSON-number codec attributes changed")
                codec = self._serde_implementations(
                    crate, body, {"Serialize", "Deserialize"}, _NUMBER_SOURCE
                )
                edges = (
                    Edge(identity + ".$number", ("primitive", _FIXED[identity]), ()),
                )
                attrs = ()
            else:
                attrs = tuple(a for a in _attributes(item) if a.startswith("#[serde"))
                _require(
                    attrs
                    == ('#[serde(try_from = "ScalarValue", into = "ScalarValue")]',),
                    "numeric-only scalar conversion contract changed",
                )
                codec = self._serde_implementations(
                    crate, body, set(), _EXECUTION_SOURCE
                )
                edges = (Edge(identity + ".$numeric", ty, ()),)
            self.definitions[identity] = Definition(
                identity,
                "struct",
                (),
                attrs,
                (("private-sealed-scalar", ty),),
                edges,
                "schema-codec:" + codec,
            )
            return
        if identity == _TENSOR_WIRE:
            self._private(item, "::schema::execution")
            super()._definition(crate, item_id, identity)
            definition = self.definitions[identity]
            expected = (
                ("shape", ("container", "alloc::vec::Vec", (("primitive", "i64"),))),
                ("data", ("reference", _TYPES + "TensorStorage", ())),
            )
            _require(
                self._plain_fields(crate, item) == expected,
                "tensor wire shape or storage carrier changed",
            )
            _require(
                definition.serde == (("deny_unknown_fields", True),),
                "tensor wire strict shape contract changed",
            )
            self._serde_implementations(crate, body, set(), _EXECUTION_SOURCE)
            return
        _require(not _serde(item), "tensor codec attributes changed")
        native = self._plain_fields(crate, item)
        normalized = tuple((name, self._peel_alias(ty)) for name, ty in native)
        expected = (
            ("shape", ("container", "alloc::vec::Vec", (("primitive", "i64"),))),
            ("data", ("reference", _TYPES + "TensorStorage", ())),
        )
        _require(normalized == expected, "tensor public/native carrier fields changed")
        codec = self._serde_implementations(
            crate, body, {"Serialize", "Deserialize"}, _EXECUTION_SOURCE
        )
        _require(
            _TENSOR_WIRE in self.locations,
            "missing actual private tensor wire definition",
        )
        owner, target = self.locations[_TENSOR_WIRE]
        ty = self._type(owner, {"resolved_path": {"id": target, "args": None}}, {})
        self.definitions[identity] = Definition(
            identity,
            "struct",
            (),
            (),
            (("native", native),),
            (Edge(identity + ".$wire", ty, ()),),
            "schema-codec:" + codec,
        )

    def schema_graph(self):
        return self.publication_graph().graph

    def publication_graph(self):
        from capacity_census_wire_publication import (
            COMPILER_BINARY_OWNERS,
            discover_published_graph,
        )
        from capacity_census_wire_fixed_roles import (
            compiler_fixed_field_contracts,
            validate_fixed_number_fields,
        )

        publication = discover_published_graph(
            self, "chelis_compiler_api", binary_owners=COMPILER_BINARY_OWNERS
        )
        validate_fixed_number_fields(
            publication.graph, compiler_fixed_field_contracts()
        )
        validate_structural_fields(publication.graph)
        validate_source_carriers(publication.graph)
        return publication


@dataclass(frozen=True, init=False, slots=True)
class VerifiedSchemaCodecs:
    canonical: VerifiedCodec
    document: str
    imported_documents: tuple[str, ...]
    outcomes: tuple
    graph_identity: str
    publication_identity: str
    candidates: tuple
    declarations: tuple
    operations: tuple[OperationContract, ...]
    structural: tuple[StructuralFieldContract, ...]
    structural_executions: tuple[str, ...]
    materialization_executions: tuple[str, ...]
    local_declarations: str
    consumer_executions: tuple
    classifications: tuple[LeafClassification, ...]
    binary_sha256: str

    def __new__(cls, *args, **kwargs):
        raise TypeError("schema codec evidence requires actual execution")

    def validate(self, documents):
        self.canonical.validate(documents)
        _require(
            self.canonical.profile == "compiler-api",
            "schema evidence requires the actual API feature profile",
        )
        matching = [
            d
            for d in documents
            if d.get("index", {}).get(str(d.get("root")), {}).get("name")
            == "chelis_compiler_api"
        ]
        _require(
            len(matching) == 1 and canonical(matching[0]) == self.document,
            "schema codec evidence does not bind this artifact",
        )
        expected = {self.canonical.document, self.document, *self.imported_documents}
        _require(
            len(documents) == len(expected)
            and {canonical(d) for d in documents} == expected,
            "schema codec evidence does not bind the complete defining artifact set",
        )

    def execution_report(self):
        return {
            "scope": "Rust numeric DTO/report/envelope codecs, source coordinates, declared source materialization routes and selected DAG admissions",
            "remaining": [
                "publication-call ownership",
                "other producer/consumer entry points",
                "cache compatibility",
                "live authority activation",
            ],
            "graph_identity": self.graph_identity,
            "publication_identity": self.publication_identity,
            "serialization_candidates": [asdict(c) for c in self.candidates],
            "serialization_definitions": [asdict(d) for d in self.declarations],
            "numeric_operations": [asdict(c) for c in self.operations],
            "structural_roles": [asdict(c) for c in self.structural],
            "structural_executions": list(self.structural_executions),
            "materialization_executions": list(self.materialization_executions),
            "local_declarations": json.loads(self.local_declarations),
            "consumer_executions": [asdict(row) for row in self.consumer_executions],
            "classification_plan": [asdict(c) for c in self.classifications],
            "canonical": self.canonical.execution_report(),
            "binary_sha256": self.binary_sha256,
            "cases": [asdict(row) for row in self.outcomes],
        }


class SchemaWireGraph(_SchemaShapeGraph):
    def __init__(self, documents, receipt: VerifiedSchemaCodecs):
        _require(
            type(receipt) is VerifiedSchemaCodecs,
            "verified schema codec receipt required",
        )
        documents = tuple(documents)
        receipt.validate(documents)
        super().__init__(documents, json.loads(receipt.canonical.vocabulary))
        from capacity_census_wire_publication import require_shared_binding_protocols

        require_shared_binding_protocols(self)
        self.receipt = receipt

    def _discover(self, roots):
        self.receipt.validate(tuple(self.documents.values()))
        return super()._discover(roots)

    def publication_graph(self):
        publication = super().publication_graph()
        _require(
            publication.identity == self.receipt.publication_identity
            and publication.graph.identity == self.receipt.graph_identity,
            "publication graph differs from verified schema evidence",
        )
        operations = validate_numeric_operation_fields(
            publication.graph,
            operation_contracts(),
            (
                Path(self.receipt.canonical.root) / "spec/05-risc-primitives.md"
            ).read_text(),
        )
        _require(
            operations == self.receipt.operations,
            "numeric operation registrations differ from verified schema evidence",
        )
        _require(
            validate_structural_fields(publication.graph) == self.receipt.structural,
            "structural role contracts differ from verified schema evidence",
        )
        canonical_shape = _CodecShapeGraph(
            tuple(self.documents.values()),
            json.loads(self.receipt.canonical.vocabulary),
        ).canonical_graph()
        _require(
            canonical_shape.identity == self.receipt.canonical.graph_identity,
            "canonical stored carrier differs from executed schema evidence",
        )
        _require(
            classification_plan(
                publication.graph, canonical_shape, operations, self.receipt.structural
            )
            == self.receipt.classifications,
            "wire classification plan differs from executed schema evidence",
        )
        return publication


def verify_schema_codecs(root: Path, target: Path) -> VerifiedSchemaCodecs:
    root, target = root.resolve(), target.resolve()
    _require(
        target.is_relative_to(root / "target"),
        "schema target must belong to this worktree",
    )
    with _target_lease(target):
        codec = _verify_canonical_codec(root, target, api_profile=True)
        environment = {
            **os.environ,
            "CARGO_BUILD_JOBS": "1",
            "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
            "RUSTC_BOOTSTRAP": "1",
            "CARGO_TARGET_DIR": str(target),
            "PYO3_PYTHON": sys.executable,
            "VIRTUAL_ENV": sys.prefix,
        }
        for package in ("chelis-compiler-api", "chelis-pipeline-core", "chelis-python"):
            command = [
                "cargo",
                "rustdoc",
                "--locked",
                "-p",
                package,
                "--lib",
                "--",
                "--output-format",
                "json",
                "-Z",
                "unstable-options",
                "--document-private-items",
            ]
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            _require(
                result.returncode == 0,
                "schema artifact build failed: " + result.stderr[-8000:],
            )
        document = json.loads((target / "doc/chelis_compiler_api.json").read_text())
        imported_documents = [
            json.loads((target / f"doc/{name}.json").read_text())
            for name in ("chelis_pipeline_core", "chelis_python")
        ]
        documents = [json.loads(codec.document), document, *imported_documents]
        vocabulary = json.loads(codec.vocabulary)
        graph = _SchemaShapeGraph(documents, vocabulary)
        from capacity_census_wire_publication import require_shared_binding_protocols

        require_shared_binding_protocols(graph)
        publication = graph.publication_graph()
        discovered = publication.graph
        operations = validate_numeric_operation_fields(
            discovered,
            operation_contracts(),
            (root / "spec/05-risc-primitives.md").read_text(),
        )
        canonical_shape = _CodecShapeGraph(documents, vocabulary).canonical_graph()
        _require(
            canonical_shape.identity == codec.graph_identity,
            "canonical stored carrier differs from executed schema evidence",
        )
        structural = validate_structural_fields(discovered)
        classifications = classification_plan(
            discovered, canonical_shape, operations, structural
        )
        from capacity_census_wire_local import (
            build_probe,
            expand_library,
            verify_expanded_library,
        )

        local_probe = build_probe(root, target)
        local_probe_hash = hashlib.sha256(local_probe.read_bytes()).hexdigest()
        expanded = expand_library(root, target, package="chelis-compiler-api")
        local_closure = verify_expanded_library(
            expanded, local_probe, root=root, target=target
        )
        _require(
            hashlib.sha256(local_probe.read_bytes()).hexdigest() == local_probe_hash,
            "local declaration probe binary changed during execution",
        )
        local_declarations = canonical(
            {
                **asdict(local_closure),
                "serde_package": expanded.serde_package,
                "schema_package": expanded.schema_package,
                "probe_sha256": local_probe_hash,
                "command": expanded.command,
            }
        )
        from capacity_census_wire_envelopes import (
            envelope_cases,
            dag_cases,
            result_reference_cases,
            metadata_reference_cases,
        )
        from capacity_census_wire_sources import source_cases
        from capacity_census_wire_artifact import artifact_cases
        from capacity_census_wire_materialization import (
            materialization_cases,
            validate_materialization_execution,
        )

        cases = (
            schema_cases(vocabulary, graph.orders[_WIRE + "BinaryScalarWire"])
            + diagnostic_cases()
            + report_cases()
            + envelope_cases()
            + dag_cases()
            + result_reference_cases()
            + metadata_reference_cases()
            + source_cases()
            + artifact_cases()
            + materialization_cases()
        )
        binary = target / "debug/examples/wire_schema_probe"
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        result = subprocess.run(
            [str(binary)],
            cwd=root,
            input="".join(canonical(case.request()) + "\n" for case in cases),
            capture_output=True,
            text=True,
            check=False,
        )
        _require(
            result.returncode == 0,
            "actual schema probe failed: " + result.stderr[-4000:],
        )
        try:
            rows = [json.loads(line) for line in result.stdout.splitlines()]
        except ValueError as error:
            raise GraphError("invalid schema observation output") from error
        _require(
            bool(rows) and rows[0] == {"schema_probe": 1},
            "missing actual schema probe header",
        )
        outcomes = check_observations(cases, rows[1:])
        structural_executions = validate_structural_execution(cases, outcomes)
        materialization_executions = validate_materialization_execution(cases, outcomes)
        _require(
            hashlib.sha256(binary.read_bytes()).hexdigest() == binary_hash,
            "schema probe binary changed during execution",
        )
        from capacity_census_wire_runner import build_and_run_rust_test

        consumer_executions = (
            build_and_run_rust_test(
                root,
                target,
                "chelis-python",
                "execution_wire_facade",
                (
                    "actual_facade_and_native_preserve_exact_values_and_reject_malformed_carriers",
                ),
            ),
            build_and_run_rust_test(
                root,
                target,
                "chelis-compiler-api",
                "wire_reports",
                (
                    "report_document_producer_and_consumer_preserve_typed_values",
                    "report_document_producer_rejects_inconsistent_counts",
                ),
            ),
        )
        _require(
            source_identity(root) == codec.source_sha256,
            "schema source changed during execution",
        )
        witness = object.__new__(VerifiedSchemaCodecs)
        for key, value in dict(
            canonical=codec,
            document=canonical(document),
            imported_documents=tuple(canonical(d) for d in imported_documents),
            outcomes=outcomes,
            graph_identity=discovered.identity,
            publication_identity=publication.identity,
            candidates=publication.candidates,
            declarations=publication.declarations,
            operations=operations,
            structural=structural,
            structural_executions=structural_executions,
            materialization_executions=materialization_executions,
            local_declarations=local_declarations,
            consumer_executions=consumer_executions,
            classifications=classifications,
            binary_sha256=binary_hash,
        ).items():
            object.__setattr__(witness, key, value)
        return witness
