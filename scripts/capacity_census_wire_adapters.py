"""Execution-backed adapters for the exact stored-value wire codec.

This module is infrastructure for the atomic C6 migration. It does not register
live census rows or supply numeric-operation or source-provenance authority.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from capacity_census_graph import GraphError


CARGO_CACHE_TAG_SIGNATURE = "Signature: 8a477f597d28d172789f06886806bc55\n"
CARGO_CACHE_TAG = "".join(
    (
        CARGO_CACHE_TAG_SIGNATURE,
        "# This file is a cache directory tag created by cargo.\n",
        "# For information about cache directory tags, see https://bford.info/cachedir/\n",
    )
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CodecCase:
    identity: str
    dtype: str
    carrier: str
    codec: str
    input: str
    expected: dict | None
    rejection_contains: str | None = None

    def request(self) -> dict:
        return {
            "id": self.identity,
            "carrier": self.carrier,
            "codec": self.codec,
            "input": self.input,
        }


@dataclass(frozen=True)
class ExecutionOutcome:
    identity: str
    selected: bool
    executed: bool
    outcome: str
    observation_sha256: str


def check_observations(
    cases: list[CodecCase], observations: list[dict]
) -> tuple[ExecutionOutcome, ...]:
    selected = {case.identity: case for case in cases}
    if not selected or len(selected) != len(cases):
        raise GraphError("empty or duplicate codec case selection")
    executed = {}
    for row in observations:
        identity = row.get("id")
        if identity not in selected or identity in executed or "observation" not in row:
            raise GraphError("unexpected, duplicate or unexecuted codec case")
        case = selected[identity]
        if row["observation"] != case.expected:
            raise GraphError(f"wrong codec observation for {identity}")
        if case.expected is None and not row.get("decode_error"):
            raise GraphError(f"missing actual decode rejection for {identity}")
        if case.rejection_contains and case.rejection_contains not in row.get(
            "decode_error", ""
        ):
            raise GraphError(f"wrong decode rejection reason for {identity}")
        if case.expected is not None and "decode_error" in row:
            raise GraphError(f"unexpected decode error for {identity}")
        executed[identity] = ExecutionOutcome(
            identity,
            True,
            True,
            "passed",
            hashlib.sha256(canonical(row).encode()).hexdigest(),
        )
    if set(executed) != set(selected):
        raise GraphError("selected codec cases were not all executed")
    return tuple(executed[case.identity] for case in cases)


def _binary(index: int, dtype: dict, elements: list, storage: bool) -> bytes:
    result = index.to_bytes(4, "little")
    if storage:
        result += len(elements).to_bytes(8, "little")
    for element in elements:
        # Float bits and key digits are both length-prefixed strings.
        if dtype["kind"] in {"float", "key"}:
            text = element.encode("ascii")
            result += len(text).to_bytes(8, "little") + text
        elif dtype["kind"] == "integer":
            result += element.to_bytes(dtype["width"], "little", signed=True)
        else:
            result += bytes([int(element)])
    return result


def _carried(vocabulary: list[dict]) -> tuple[str, ...]:
    """The runtime dtypes a scalar or graph literal carries: every one but the
    random key, which spec/10 section 3.2 gives no scalar object and no graph
    literal."""
    return tuple(d["name"] for d in vocabulary if d["kind"] != "key")


def _stored(vocabulary: list[dict]) -> tuple[str, ...]:
    """The storage mirrors' variants: the carried dtypes, then the key storage
    object that only a tensor execution value admits (spec/10 section 3.2).
    The key is appended last so every carried dtype keeps the scalar ordinal."""
    return _carried(vocabulary) + tuple(
        d["name"] for d in vocabulary if d["kind"] == "key"
    )


def codec_cases(
    vocabulary: list[dict], order: tuple[str, ...] | None = None
) -> list[CodecCase]:
    """Independent JSON/bincode cases; binary ordinals come from actual rustdoc."""
    names = tuple(d["name"] for d in vocabulary)
    carried = _carried(vocabulary)
    order = carried if order is None else order
    if (
        not carried
        or len(set(names)) != len(names)
        or set(order) != set(carried)
        or len(order) != len(carried)
    ):
        raise GraphError("codec vocabulary/order is empty, duplicated or incomplete")
    cases = []
    for dtype in vocabulary:
        name, width, kind = dtype["name"], dtype["width"], dtype["kind"]
        if kind == "key":
            # A key is stored (runtime dtype `key`, one 64-bit word) but has no
            # scalar object and no graph literal (spec/10 section 3.2). The
            # scalar mirrors have no key variant, so a key spelling is rejected
            # by its JSON tag and at the first bincode ordinal past the carried
            # variants. The storage mirrors do hold the key storage object,
            # appended at that same ordinal, for tensor execution values only;
            # the graph codec `TensorStorage` decodes it and then refuses it.
            if name != "key" or width != 8:
                raise GraphError("unsupported random key runtime dtype")
            word = (7).to_bytes(8, "little")
            digits = "0" * 16
            refusal = "a random key has no literal carrier"
            for carrier in ("scalar", "storage"):
                storage = carrier == "storage"
                field = "values" if storage else "value"
                for label, payload in (
                    ("no-literal-bits", digits),
                    ("no-literal-value", 7),
                ):
                    member = "bits" if label == "no-literal-bits" else field
                    bad = {"dtype": name, member: [payload] if storage else payload}
                    cases.append(
                        CodecCase(
                            f"{carrier}/json/{name}/{label}",
                            name,
                            carrier,
                            "json",
                            canonical(bad),
                            None,
                            "unknown variant"
                            if not storage
                            else refusal
                            if member == "bits"
                            else "unknown field",
                        )
                    )
                binary = len(order).to_bytes(4, "little")
                if storage:
                    # A well-formed key storage object, so the refusal is the
                    # graph codec's own and not a malformed payload.
                    text = digits.encode("ascii")
                    binary += (1).to_bytes(8, "little")
                    binary += len(text).to_bytes(8, "little") + text
                else:
                    binary += word
                cases.append(
                    CodecCase(
                        f"{carrier}/binary/{name}/no-literal-ordinal",
                        name,
                        carrier,
                        "binary",
                        binary.hex(),
                        None,
                        refusal if storage else "variant index",
                    )
                )
            continue
        if kind not in {"float", "integer", "bool"} or width not in {1, 2, 4, 8}:
            raise GraphError("unsupported runtime dtype representation")
        if kind == "float":
            # Own-width bit patterns: both zeros, subnormal, both infinities,
            # distinct quiet/signaling NaNs, finite fraction and maximal bits.
            exponent = {"f64": 11, "f32": 8, "f16": 5, "bf16": 8}.get(name)
            if exponent is None or width < 2:
                raise GraphError("unsupported floating runtime dtype")
            sign = 1 << (width * 8 - 1)
            infinity = ((1 << exponent) - 1) << (width * 8 - 1 - exponent)
            raw = [
                0,
                sign,
                1,
                infinity,
                sign | infinity,
                infinity | 1,
                infinity | (1 << (width * 8 - 2 - exponent)),
                sign | infinity | 3,
                infinity - 1,
                sign - 1,
            ]
            values = [f"{v:0{width * 2}x}" for v in raw]
        elif kind == "integer":
            bound = 1 << (width * 8 - 1)
            values = [-bound, -1, 0, 1, bound - 1]
            if width == 8:
                values += [-(2**53 + 1), 2**53 + 1]
        else:
            if name != "bool" or width != 1:
                raise GraphError("unsupported nonnumeric runtime dtype")
            values = [False, True]
        index = order.index(name)
        for carrier in ("scalar", "storage"):
            storage = carrier == "storage"
            field = "bits" if kind == "float" else "values" if storage else "value"
            groups = (
                [(f"value-{i}", [v]) for i, v in enumerate(values)]
                if not storage
                else [("values", values), ("empty", [])]
            )
            if storage and name in {"f16", "bf16"}:
                groups.append(("all-bits", [f"{v:04x}" for v in range(65536)]))
            for label, elements in groups:
                wire = {"dtype": name, field: elements if storage else elements[0]}
                binary = _binary(index, dtype, elements, storage).hex()
                expected = {
                    "dtype": name,
                    "elements": elements,
                    "json": wire,
                    "binary": binary,
                }
                for codec, value in (("json", canonical(wire)), ("binary", binary)):
                    cases.append(
                        CodecCase(
                            f"{carrier}/{codec}/{name}/{label}",
                            name,
                            carrier,
                            codec,
                            value,
                            expected,
                        )
                    )
            wire = {"dtype": name, field: values[:1] if storage else values[0]}
            invalid = [
                ("missing-tag", {field: wire[field]}),
                ("unknown-tag", {**wire, "dtype": "unknown"}),
                ("missing-payload", {"dtype": name}),
                ("extra-field", {**wire, "unexpected": 7}),
                (
                    "wrong-member",
                    {
                        "dtype": name,
                        "value" if field != "value" else "bits": wire[field],
                    },
                ),
            ]
            if kind == "float":
                bad_values = [
                    "0" * (2 * width - 1),
                    "0" * (2 * width + 1),
                    "F" * (2 * width),
                    "g" * (2 * width),
                    "0x" + "0" * (2 * width - 2),
                    0.5,
                ]
            elif kind == "integer":
                bad_values = [
                    -(1 << (width * 8 - 1)) - 1,
                    1 << (width * 8 - 1),
                    1.5,
                    "1",
                    True,
                ]
            else:
                bad_values = [0, 1, "true"]
            for i, bad in enumerate(bad_values):
                invalid.append(
                    (
                        f"bad-payload-{i}",
                        {"dtype": name, field: [bad] if storage else bad},
                    )
                )
                if kind == "float" and isinstance(bad, str):
                    raw = _binary(index, dtype, [bad], storage).hex()
                    cases.append(
                        CodecCase(
                            f"{carrier}/binary/{name}/bad-grammar-{i}",
                            name,
                            carrier,
                            "binary",
                            raw,
                            None,
                            "lowercase hexadecimal digits",
                        )
                    )
            for label, bad in invalid:
                cases.append(
                    CodecCase(
                        f"{carrier}/json/{name}/{label}",
                        name,
                        carrier,
                        "json",
                        canonical(bad),
                        None,
                        "missing field `dtype`"
                        if label == "missing-tag"
                        else "unknown variant"
                        if label == "unknown-tag"
                        else "unknown field"
                        if label in {"extra-field", "wrong-member"}
                        else None,
                    )
                )
            duplicate = canonical(wire)[:-1] + ',"dtype":' + json.dumps(name) + "}"
            cases.append(
                CodecCase(
                    f"{carrier}/json/{name}/duplicate-tag",
                    name,
                    carrier,
                    "json",
                    duplicate,
                    None,
                    "duplicate field `dtype`",
                )
            )
            good = _binary(index, dtype, values[:1], storage)
            for label, bad in (
                ("truncated", good[:-1]),
                ("unknown-tag", (2**32 - 1).to_bytes(4, "little") + good[4:]),
            ):
                cases.append(
                    CodecCase(
                        f"{carrier}/binary/{name}/{label}",
                        name,
                        carrier,
                        "binary",
                        bad.hex(),
                        None,
                    )
                )
            if kind == "bool":
                cases.append(
                    CodecCase(
                        f"{carrier}/binary/{name}/invalid-bool",
                        name,
                        carrier,
                        "binary",
                        (good[:-1] + b"\x02").hex(),
                        None,
                    )
                )
    return cases


def source_identity(root: Path) -> str:
    """Read compiler inputs directly; git index flags cannot hide byte changes."""
    paths = set(root.glob("crates/**/*.rs")) | set(root.glob("crates/**/Cargo.toml"))
    paths |= set(root.glob("scripts/capacity_census*.py"))
    paths |= set(root.glob("scripts/test_capacity_census*.py"))
    paths |= set(root.glob("scripts/fixtures/capacity_graph/**/*.*"))
    paths |= set(root.glob("bindings/python/**/*.py"))
    paths |= set(root.glob("tests/support/**/*.rs"))
    paths |= set(root.glob("tests/conformance/hull/**/*.py"))
    # Bytecode caches are products of test execution, not authored inputs.
    paths = {path for path in paths if "__pycache__" not in path.parts and path.is_file()}
    paths |= {
        root / name
        for name in (
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            "crates/chelis-types/examples/wire_codec_probe.rs",
            "spec/02-surf-syntax.md",
            "spec/03-deep-syntax.md",
            "spec/04-type-system.md",
            "spec/05-risc-primitives.md",
            "spec/10-serialization.md",
            "spec/11-ffi.md",
        )
    }
    digest = hashlib.sha256()
    for path in sorted(paths):
        if not path.is_file():
            raise GraphError(f"missing codec source input {path}")
        relative = path.relative_to(root).as_posix().encode()
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(content).to_bytes(8, "big") + content)
    return digest.hexdigest()


from capacity_census_graph import (
    Definition,
    Edge,
    Leaf,
    RustdocGraph,
    _attributes,
    _serde,
)

_TYPES = "chelis_types::dtype_semantics::"
_WIRE = _TYPES + "wire_codec::"
_HEX = _WIRE + "HexBits"
# spec/10 section 3.2's key digits `h` and the scalar key carrier. Neither is a
# number: the digits are a private String, and the carrier's native payload is
# the opaque `RandomKey`, so both add no numeric leaf.
_KEY_HEX = _WIRE + "KeyHex"
_KEY_BITS = _WIRE + "KeyBits"
_MIRRORS = tuple(
    _WIRE + name
    for name in ("ScalarWire", "BinaryScalarWire", "StorageWire", "BinaryStorageWire")
)
_CARRIERS = {
    _TYPES + "ScalarValue": ("bits", "Bits", "ScalarWire"),
    _TYPES + "TensorStorage": ("buf", "Buf", "StorageWire"),
}
_CODEC_SOURCE = "crates/chelis-types/src/dtype_semantics/wire_codec.rs"


def _require(condition: bool, message: str):
    if not condition:
        raise GraphError(message)


class _CodecShapeGraph(RustdocGraph):
    """Preliminary shape inspection; this class cannot mint an authority witness."""

    def __init__(self, documents, vocabulary):
        super().__init__(documents)
        self.vocabulary = {d["name"]: d for d in vocabulary}
        _require(len(self.vocabulary) == len(vocabulary), "duplicate runtime dtype")
        self.orders = {}

    def _serde_implementations(
        self,
        crate,
        inner,
        custom,
        source=_CODEC_SOURCE,
        required=frozenset({"Serialize", "Deserialize"}),
    ):
        found = {}
        for impl_id in inner.get("impls", []):
            item = self._item(crate, impl_id)
            body = item.get("inner", {}).get("impl", {})
            trait = body.get("trait")
            if not trait:
                continue
            path = (
                self.documents[crate]["paths"].get(str(trait["id"]), {}).get("path", [])
            )
            actual = "::".join(path)
            names = {
                f"{owner}::{module}::{name}": name
                for owner in ("serde", "serde_core")
                for module, name in (("ser", "Serialize"), ("de", "Deserialize"))
            }
            names.update(
                {"serde::Serialize": "Serialize", "serde::Deserialize": "Deserialize"}
            )
            name = names.get(actual)
            if name is None:
                continue
            _require(name not in found, "multiple actual serde implementations")
            _require(
                not body.get("blanket_impl")
                and not body.get("is_negative")
                and not body.get("is_synthetic"),
                "unsupported serde implementation",
            )
            _require(
                not _serde(item), "unexpected serde options on codec implementation"
            )
            derived = "#[automatically_derived]" in _attributes(item)
            _require(derived != (name in custom), "serde implementation mode changed")
            span = item.get("span")
            _require(
                bool(span) and span.get("filename") == source,
                "codec implementation source identity changed",
            )
            _require(bool(body.get("items")), "missing serde implementation method")
            methods = tuple(
                (self._item(crate, i).get("name"), self._item(crate, i).get("span"))
                for i in body["items"]
            )
            _require(
                all(method_span and method_span.get("filename") == source
                    for _, method_span in methods),
                "codec method source identity changed",
            )
            # The current execution witness retains the complete rustdoc
            # document (and source/artifact hashes), including every span.
            # A reviewed structural identity must survive comment-only moves.
            found[name] = (actual, derived, source, tuple(method for method, _ in methods))
        _require(
            set(found) == required,
            "missing actual serde encoder or decoder",
        )
        return canonical(found)

    @staticmethod
    def _private(item, module):
        visibility = item.get("visibility")
        _require(
            isinstance(visibility, dict)
            and visibility.get("restricted", {}).get("path") == module,
            "canonical carrier field is not module-private",
        )

    def _nominal(self, crate, ty):
        _require(set(ty) == {"resolved_path"}, "expected exact nominal carrier")
        body = ty["resolved_path"]
        record = self.documents[crate]["paths"].get(str(body["id"]))
        _require(bool(record), "unresolved canonical carrier type")
        return "::".join(record["path"]), body.get("args")

    def _native_layout(self, crate, item, storage):
        body = item["inner"].get("enum")
        _require(
            body is not None
            and not body.get("has_stripped_variants")
            and not self._parameters(body),
            "unsupported native storage enum",
        )
        self._private(item, "::dtype_semantics")
        _require(not _serde(item), "native storage gained serde options")
        result = []
        for variant_id in body["variants"]:
            variant = self._item(crate, variant_id)
            _require(not _serde(variant), "native storage variant gained serde options")
            fields = variant["inner"]["variant"]["kind"].get("tuple")
            _require(
                isinstance(fields, list) and len(fields) == 1,
                "native storage payload changed",
            )
            field = self._item(crate, fields[0])
            _require(not _serde(field), "native storage field gained serde options")
            ty = field["inner"]["struct_field"]
            if storage:
                nominal, args = self._nominal(crate, ty)
                _require(
                    nominal == "alloc::vec::Vec"
                    and args is not None
                    and not args["angle_bracketed"]["constraints"]
                    and len(args["angle_bracketed"]["args"]) == 1,
                    "native storage is not an exact Vec payload",
                )
                ty = args["angle_bracketed"]["args"][0].get("type")
                _require(ty is not None, "native storage element is unresolved")
            if set(ty) == {"primitive"}:
                primitive = ty["primitive"]
            else:
                nominal, args = self._nominal(crate, ty)
                # The opaque `RandomKey` is admitted only as the key dtype's
                # payload; the vocabulary comparison below enforces that.
                _require(
                    args is None
                    and nominal
                    in {
                        "half::binary16::f16",
                        "half::bfloat::bf16",
                        _TYPES + "RandomKey",
                    },
                    "unsupported native numeric adapter",
                )
                primitive = nominal.rsplit("::", 1)[-1]
            result.append((variant["name"], primitive))
        expected = {}
        for dtype in self.vocabulary.values():
            name = dtype["name"]
            variant = (
                "I" + name[3:]
                if dtype["kind"] == "integer"
                else name[0].upper() + name[1:]
            )
            primitive = (
                "i" + name[3:]
                if dtype["kind"] == "integer"
                else "u8"
                if storage and name == "bool"
                # The structurally non-numeric key carries the opaque key.
                else "RandomKey"
                if dtype["kind"] == "key"
                else name
            )
            expected[variant] = primitive
        _require(
            len(result) == len(expected) and dict(result) == expected,
            "native dtype/width vocabulary differs from executed runtime vocabulary",
        )
        return tuple(result)

    def _definition(self, crate, item_id, identity):
        if identity in self.definitions:
            return
        if (
            identity not in _CARRIERS
            and identity not in {_HEX, _KEY_HEX, _KEY_BITS}
            and identity not in _MIRRORS
        ):
            return super()._definition(crate, item_id, identity)
        item = self._item(crate, item_id)
        if identity == _KEY_HEX:
            # Transparent over a private String with a strict custom decoder,
            # like HexBits but with no width parameter: the width is the key's.
            body = item["inner"].get("struct")
            _require(
                body is not None and not self._parameters(body),
                "KeyHex must be an exact nongeneric struct",
            )
            self._private(item, "::dtype_semantics::wire_codec")
            _require(
                [a for a in _attributes(item) if a.startswith("#[serde")]
                == ["#[serde(transparent)]"],
                "KeyHex transparency changed",
            )
            ids = body["kind"].get("tuple")
            _require(
                isinstance(ids, list) and len(ids) == 1,
                "KeyHex payload layout changed",
            )
            field = self._item(crate, ids[0])
            self._private(field, "::dtype_semantics::wire_codec")
            _require(not _serde(field), "KeyHex payload gained serde options")
            ty = self._type(crate, field["inner"]["struct_field"], {})
            _require(
                ty == ("atomic", "alloc::string::String"),
                "KeyHex payload must be an exact String",
            )
            codec = self._serde_implementations(crate, body, {"Deserialize"})
            self.definitions[identity] = Definition(
                identity,
                "struct",
                (),
                (("transparent", True),),
                (("private-string",),),
                (Edge(identity + ".$0", ty, ()),),
                "custom-shape:" + codec,
            )
            return
        if identity == _KEY_BITS:
            # The scalar key carrier of `{"type":"key","bits":h}`: its native
            # payload is the opaque key and both directions of its codec are
            # the KeyHex digits, never an integer or IEEE bits.
            body = item["inner"].get("struct")
            _require(
                body is not None and not self._parameters(body) and not _serde(item),
                "KeyBits carrier layout changed",
            )
            ids = body["kind"].get("tuple")
            _require(
                isinstance(ids, list) and len(ids) == 1,
                "KeyBits must have one private key payload",
            )
            field = self._item(crate, ids[0])
            self._private(field, "::dtype_semantics::wire_codec")
            _require(not _serde(field), "KeyBits payload gained serde options")
            nominal, args = self._nominal(crate, field["inner"]["struct_field"])
            _require(
                nominal == _TYPES + "RandomKey" and args is None,
                "KeyBits native payload must be the opaque RandomKey",
            )
            codec = self._serde_implementations(
                crate, body, {"Serialize", "Deserialize"}
            )
            _require(_KEY_HEX in self.locations, "missing private key digit codec")
            target_crate, target_id = self.locations[_KEY_HEX]
            self.definitions[identity] = None
            ty = self._type(
                target_crate, {"resolved_path": {"id": target_id, "args": None}}, {}
            )
            self.definitions[identity] = Definition(
                identity,
                "struct",
                (),
                (),
                (("private-native", nominal),),
                (Edge(identity + ".$bits", ty, ()),),
                "custom-shape:" + codec,
            )
            return
        if identity == _HEX:
            body = item["inner"].get("struct")
            _require(
                body is not None
                and self._parameters(body) == ("DIGITS",)
                and self._parameter_kinds(body) == ("usize",),
                "HexBits const contract changed",
            )
            self._private(item, "::dtype_semantics::wire_codec")
            _require(
                [a for a in _attributes(item) if a.startswith("#[serde")]
                == ["#[serde(transparent)]"],
                "HexBits transparency changed",
            )
            ids = body["kind"].get("tuple")
            _require(
                isinstance(ids, list) and len(ids) == 1,
                "HexBits payload layout changed",
            )
            field = self._item(crate, ids[0])
            self._private(field, "::dtype_semantics::wire_codec")
            _require(not _serde(field), "HexBits payload gained serde options")
            ty = self._type(crate, field["inner"]["struct_field"], {})
            _require(
                ty == ("atomic", "alloc::string::String"),
                "HexBits payload must be an exact String",
            )
            codec = self._serde_implementations(crate, body, {"Deserialize"})
            self.definitions[identity] = Definition(
                identity,
                "struct",
                ("DIGITS",),
                (("transparent", True),),
                (("private-string",),),
                (Edge(identity + ".$0", ty, ()),),
                "custom-shape:" + codec,
                ("usize",),
            )
            return
        if identity in _CARRIERS:
            field_name, native_name, mirror = _CARRIERS[identity]
            body = item["inner"].get("struct")
            _require(
                body is not None and not self._parameters(body) and not _serde(item),
                "canonical stored carrier layout changed",
            )
            plain = body["kind"].get("plain")
            _require(
                plain is not None
                and not plain["has_stripped_fields"]
                and len(plain["fields"]) == 1,
                "canonical stored carrier must have one private payload",
            )
            field = self._item(crate, plain["fields"][0])
            self._private(field, "::dtype_semantics")
            _require(
                field["name"] == field_name and not _serde(field),
                "canonical stored carrier payload changed",
            )
            nominal, args = self._nominal(crate, field["inner"]["struct_field"])
            _require(
                nominal == _TYPES + native_name and args is None,
                "canonical carrier native payload changed",
            )
            native = self._item(*self.locations[nominal])
            native_layout = self._native_layout(crate, native, mirror == "StorageWire")
            codec = self._serde_implementations(
                crate, body, {"Serialize", "Deserialize"}
            )
            self.definitions[identity] = None
            edges = []
            for label, name in (("json", mirror), ("binary", "Binary" + mirror)):
                target = _WIRE + name
                _require(
                    target in self.locations, "missing private wire codec definition"
                )
                target_crate, target_id = self.locations[target]
                ty = self._type(
                    target_crate, {"resolved_path": {"id": target_id, "args": None}}, {}
                )
                edges.append(Edge(identity + ".$" + label, ty, ()))
            self.definitions[identity] = Definition(
                identity,
                "struct",
                (),
                (),
                (("private-native", nominal, native_layout),),
                tuple(edges),
                "custom-shape:" + codec,
            )
            return
        self._private(item, "::dtype_semantics::wire_codec")
        super()._definition(crate, item_id, identity)
        definition = self.definitions[identity]
        _require(
            definition is not None
            and definition.kind == "enum"
            and not definition.parameters,
            "unsupported wire mirror",
        )
        binary = identity.rsplit("::", 1)[-1].startswith("Binary")
        storage = identity.endswith("StorageWire")
        _require(
            definition.serde
            == (() if binary else (("deny_unknown_fields", True), ("tag", "dtype"))),
            "wire dtype discriminator contract changed",
        )
        codec = self._serde_implementations(crate, item["inner"]["enum"], set())
        edges = []
        order = []
        for variant, attrs, layout in definition.layout:
            name = (
                "int" + variant[1:]
                if variant.startswith("I")
                else variant[0].lower() + variant[1:]
            )
            _require(
                name in self.vocabulary, "wire variant lacks executed runtime dtype"
            )
            dtype = self.vocabulary[name]
            # A scalar key is its own execution value, never a scalar object.
            _require(
                storage or dtype["kind"] != "key",
                "a random key has no wire literal carrier",
            )
            _require(
                attrs == (() if binary else (("rename", name),)),
                "wire variant dtype name changed",
            )
            if dtype["kind"] == "key":
                # The key storage object `{"dtype":"key","bits":[h,...]}`: its
                # payload is KeyHex digits, which carry no numeric authority.
                _require(
                    layout == ("struct", (("bits", ()),)),
                    "key storage payload fields changed",
                )
                edge = next(
                    (
                        e
                        for e in definition.edges
                        if e.path == identity + "::" + variant + ".bits"
                    ),
                    None,
                )
                _require(
                    edge is not None
                    and not edge.serde
                    and edge.type
                    == ("container", "alloc::vec::Vec", (("reference", _KEY_HEX, ()),)),
                    "key storage payload must be KeyHex digits",
                )
                edges.append(edge)
                order.append(name)
                continue
            field = (
                "bits" if dtype["kind"] == "float" else "values" if storage else "value"
            )
            _require(
                layout == ("struct", ((field, ()),)), "wire payload fields changed"
            )
            edge = next(
                (
                    e
                    for e in definition.edges
                    if e.path == identity + "::" + variant + "." + field
                ),
                None,
            )
            _require(edge is not None and not edge.serde, "missing exact wire payload")
            if dtype["kind"] == "float":
                payload = ("reference", _HEX, (("const", str(dtype["width"] * 2)),))
                primitive = name
            else:
                primitive = (
                    "i" + str(dtype["width"] * 8)
                    if dtype["kind"] == "integer"
                    else "bool"
                )
                payload = ("primitive", primitive)
            expected = (
                ("container", "alloc::vec::Vec", (payload,)) if storage else payload
            )
            _require(
                edge.type == expected, "wire payload width/carrier disagrees with dtype"
            )
            edges.append(
                replace(edge, type=("encoded-numeric", primitive, edge.type))
                if dtype["kind"] == "float"
                else edge
            )
            order.append(name)
        vocabulary = list(self.vocabulary.values())
        carried = _stored(vocabulary) if storage else _carried(vocabulary)
        _require(
            len(order) == len(carried) and set(order) == set(carried),
            "wire mirror dtype vocabulary is incomplete",
        )
        _require(
            order[len(_carried(vocabulary)) :] == list(carried[len(_carried(vocabulary)) :]),
            "the key storage variant must follow every carried dtype",
        )
        self.orders[identity] = tuple(order)
        self.definitions[identity] = replace(
            definition, edges=tuple(edges), codec="derived-shape:" + codec
        )

    def _evaluate(self, expr, origin, summaries):
        if expr[0] == "encoded-numeric":
            # The enclosing exact dtype/width relationship is checked above.
            # HexBits by itself has no dtype and receives no numeric authority.
            return {Leaf(origin, expr[1])}
        return super()._evaluate(expr, origin, summaries)

    def canonical_graph(self):
        roots = []
        for identity in sorted(_CARRIERS):
            _require(identity in self.locations, "missing canonical stored carrier")
            crate, item_id = self.locations[identity]
            roots.append(
                (crate, identity, {"resolved_path": {"id": item_id, "args": None}})
            )
        result = self._discover(roots)
        scalar = [self.orders.get(_WIRE + n) for n in ("ScalarWire", "BinaryScalarWire")]
        storage = [
            self.orders.get(_WIRE + n) for n in ("StorageWire", "BinaryStorageWire")
        ]
        keys = tuple(
            d["name"] for d in self.vocabulary.values() if d["kind"] == "key"
        )
        _require(
            len(self.orders) == 4
            and scalar[0] is not None
            and all(order == scalar[0] for order in scalar)
            and all(order == scalar[0] + keys for order in storage),
            "JSON/positional dtype orders disagree",
        )
        return result


@dataclass(frozen=True, init=False, slots=True)
class VerifiedCodec:
    """Only the fixed executable probe may construct a current-source receipt."""

    root: str
    source_sha256: str
    document: str
    vocabulary: str
    graph_identity: str
    outcomes: tuple[ExecutionOutcome, ...]
    binary_sha256: str
    profile: str

    def __new__(cls, *args, **kwargs):
        raise TypeError(
            "use verify_canonical_codec; a descriptor is not execution evidence"
        )

    def execution_report(self) -> dict:
        return {
            "source_sha256": self.source_sha256,
            "graph_identity": self.graph_identity,
            "binary_sha256": self.binary_sha256,
            "profile": self.profile,
            "cases": [asdict(row) for row in self.outcomes],
        }

    def validate(self, documents):
        _require(
            source_identity(Path(self.root)) == self.source_sha256,
            "codec execution is stale against actual source bytes",
        )
        matching = [
            d
            for d in documents
            if d.get("index", {}).get(str(d.get("root")), {}).get("name")
            == "chelis_types"
        ]
        _require(
            len(matching) == 1 and canonical(matching[0]) == self.document,
            "codec execution does not bind this defining artifact",
        )


class CanonicalWireGraph(_CodecShapeGraph):
    """Graph traversal with verified JSON/positional canonical carrier adapters.

    Each float mirror edge exposes the actual declared numeric width despite its
    bit-string representation. A bare String or a separately named HexBits type
    receives no numeric authority. Other custom codecs still fail closed.
    """

    def __init__(self, documents, receipt: VerifiedCodec):
        _require(type(receipt) is VerifiedCodec, "a verified codec receipt is required")
        documents = tuple(documents)
        receipt.validate(documents)
        super().__init__(documents, json.loads(receipt.vocabulary))
        self.receipt = receipt

    def _discover(self, roots):
        self.receipt.validate(self.documents.values())
        return super()._discover(roots)


@contextmanager
def _target_lease(target: Path):
    target.mkdir(parents=True, exist_ok=True)
    with (target / ".wire-codec-verifier.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise GraphError(
                "codec target is already owned by another verifier sequence"
            ) from error
        try:
            marker = target / "CACHEDIR.TAG"
            if marker.exists():
                if not marker.read_text().startswith(CARGO_CACHE_TAG_SIGNATURE):
                    raise GraphError("codec target has invalid CACHEDIR.TAG")
            else:
                marker.write_text(CARGO_CACHE_TAG)
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def verify_canonical_codec(root: Path, target: Path) -> VerifiedCodec:
    root, target = root.resolve(), target.resolve()
    _require(
        target.is_relative_to(root / "target"),
        "codec target must belong to this worktree",
    )
    with _target_lease(target):
        return _verify_canonical_codec(root, target)


def _verify_canonical_codec(
    root: Path, target: Path, *, api_profile: bool = False
) -> VerifiedCodec:
    """Build actual code/artifacts and independently check every selected case.

    Callers must acquire their normal heavyweight-command slot before invoking
    this function. The target must be owned by this worktree. No saved receipt,
    caller-supplied program, or expected-PASS descriptor can bypass execution.
    """
    root, target = root.resolve(), target.resolve()
    _require(
        target.is_relative_to(root / "target"),
        "codec target must belong to this worktree",
    )
    before = source_identity(root)
    environment = {
        **os.environ,
        "CARGO_BUILD_JOBS": "1",
        "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
        "RUSTC_BOOTSTRAP": "1",
        "CARGO_TARGET_DIR": str(target),
    }
    extra_packages = ["-p", "chelis-compiler-api"] if api_profile else []
    commands = [
        [
            "cargo",
            "clean",
            "-p",
            "chelis-types",
            "-p",
            "chelis-vocab",
            *extra_packages,
            "--target-dir",
            str(target),
        ],
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "chelis-types",
            *extra_packages,
            *(["--examples"] if api_profile else ["--example", "wire_codec_probe"]),
        ],
        [
            "cargo",
            "rustdoc",
            "--locked",
            "-p",
            "chelis-types",
            "--lib",
            "--",
            "--output-format",
            "json",
            "-Z",
            "unstable-options",
            "--document-private-items",
        ],
    ]
    for command in commands:
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
            "codec artifact build failed: " + result.stderr[-8000:],
        )
    document = json.loads((target / "doc/chelis_types.json").read_text())
    binary = target / "debug/examples/wire_codec_probe"
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()

    def execute(requests):
        result = subprocess.run(
            [str(binary)],
            cwd=root,
            input="".join(canonical(request) + "\n" for request in requests),
            capture_output=True,
            text=True,
            check=False,
        )
        _require(
            result.returncode == 0,
            "actual codec probe failed: " + result.stderr[-4000:],
        )
        try:
            rows = [json.loads(line) for line in result.stdout.splitlines()]
        except ValueError as error:
            raise GraphError("invalid actual codec observation output") from error
        _require(
            bool(rows) and set(rows[0]) == {"vocabulary"},
            "missing executed runtime vocabulary",
        )
        return rows

    vocabulary_rows = execute([])
    _require(len(vocabulary_rows) == 1, "unexpected runtime vocabulary observations")
    vocabulary = vocabulary_rows[0]["vocabulary"]
    graph = _CodecShapeGraph([document], vocabulary)
    discovered = graph.canonical_graph()
    cases = codec_cases(vocabulary, graph.orders[_WIRE + "BinaryScalarWire"])
    observed = execute([case.request() for case in cases])
    _require(
        observed[0] == vocabulary_rows[0], "runtime vocabulary changed during execution"
    )
    outcomes = check_observations(cases, observed[1:])
    _require(
        hashlib.sha256(binary.read_bytes()).hexdigest() == binary_hash,
        "codec probe binary changed during execution",
    )
    _require(
        source_identity(root) == before,
        "codec source changed during artifact generation/execution",
    )
    receipt = object.__new__(VerifiedCodec)
    for key, value in dict(
        root=str(root),
        source_sha256=before,
        document=canonical(document),
        vocabulary=canonical(vocabulary),
        graph_identity=discovered.identity,
        outcomes=outcomes,
        binary_sha256=binary_hash,
        profile="compiler-api" if api_profile else "types-default",
    ).items():
        object.__setattr__(receipt, key, value)
    return receipt
