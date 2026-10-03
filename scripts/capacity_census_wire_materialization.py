"""Independent spec/10 §3.3 source transport and admission observations.

The cases name existing public text, RawExpr stamping, and extension-data
boundaries. WireLiteral/WireDeepExpr decoding remains transport: there is no
production DTO-to-AST ingress. Expected trees come from the written grammar,
and scalar results from integer arithmetic/IEEE bits, never probe snapshots.
This is a bounded corpus, not a census of every source consumer or annotation.
"""

from __future__ import annotations

import struct

from capacity_census_wire_adapters import CodecCase, canonical


def _float(value):
    return {"f64_bits": struct.pack(">d", value).hex()}


def _atom(kind, value):
    return {"kind": {"type": "atom", "atom": {"type": kind, "value": value}}}


def _node(tag, *children, **metadata):
    entries = [{"key": key, "value": value} for key, value in sorted(metadata.items())]
    return {
        "kind": {
            "type": "list",
            "elements": [
                _atom("symbol", tag),
                {"kind": {"type": "map", "entries": entries}},
                *children,
            ],
        }
    }


def _definition(expression, **metadata):
    return [_node("def", _atom("symbol", "value"), expression, **metadata)]


def _literal(kind, value, suffix=None):
    dto = {"kind": ("typed_" if suffix else "") + kind, "value": value}
    if suffix:
        dto["suffix"] = suffix
    return dto


def _surf_literal(dto):
    return [
        {
            "kind": "let_def",
            "name": "value",
            "ty": None,
            "value": {"kind": "lit", "literal": dto},
        }
    ]


def source_parameter_cases():
    """spec/10 §3.3: lexical int64 selectors/axes precede source admission."""
    cases = []

    def add(carrier, codec, name, value, expected, error=None):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/parameter-{name}",
                "source-materialization",
                carrier,
                codec,
                value,
                expected,
                error,
            )
        )

    def expression(kind, field, value):
        return {
            "kind": kind,
            "expr": {"kind": "var", "name": "x", "span": {"offset": 0, "len": 1}},
            field: value,
            "span": {"offset": 0, "len": 20},
        }

    for kind, field in [("tuple_get", "index"), ("vmap", "axis")]:
        for value in [-(2**63), -1, 0, 9007199254740993, 2**63 - 1] + (
            [None] if kind == "vmap" else []
        ):
            expected = {
                "kind": kind,
                "expr": {"kind": "var", "name": "x"},
                field: value,
            }
            add(
                "WireSurfExpr",
                "json",
                f"{field}-{value}",
                canonical(expression(kind, field, value)),
                {"syntax": expected},
            )
        for index, value in enumerate(
            [1.0, 2**64 - 1, "1", True, {"dtype": "int64", "value": 1}]
            + ([None] if kind == "tuple_get" else [])
        ):
            add(
                "WireSurfExpr",
                "json",
                f"invalid-{field}-{index}",
                canonical(expression(kind, field, value)),
                None,
            )

    for value in [0, 1, 9007199254740993]:
        source = f"value = x.{value}"
        selector = {
            "kind": "tuple_get",
            "expr": {"kind": "var", "name": "x"},
            "index": value,
        }
        surf = [{"kind": "let_def", "name": "value", "ty": None, "value": selector}]
        deep = _definition(
            _node(
                "tuple-get",
                _node("var", _atom("symbol", "x"), span=_atom("str", "surf:8..9")),
                _node(
                    "lit",
                    _atom("int", value),
                    type=_node("t-prim", _atom("symbol", "i32")),
                ),
                span=_atom("str", f"surf:8..{len(source)}"),
            ),
            span=_atom("str", f"surf:0..{len(source)}"),
        )
        add("SourceProgram", "parse-surf", f"tuple-{value}", source, {"syntax": surf})
        add("SourceProgram", "desugar-surf", f"tuple-{value}", source, {"syntax": deep})
    for value in [None, 1, 9007199254740993]:
        source = "value = vmap(f" + ("" if value is None else f", axis={value}") + ")"
        mapper = {"kind": "vmap", "expr": {"kind": "var", "name": "f"}, "axis": value}
        surf = [{"kind": "let_def", "name": "value", "ty": None, "value": mapper}]
        deep = _definition(
            _node(
                "vmap",
                _node("var", _atom("symbol", "f"), span=_atom("str", "surf:13..14")),
                _node(
                    "lit",
                    _atom("int", 0 if value is None else value),
                    type=_node("t-prim", _atom("symbol", "i32")),
                ),
                span=_atom("str", f"surf:8..{len(source)}"),
            ),
            span=_atom("str", f"surf:0..{len(source)}"),
        )
        add("SourceProgram", "parse-surf", f"axis-{value}", source, {"syntax": surf})
        add("SourceProgram", "desugar-surf", f"axis-{value}", source, {"syntax": deep})
    for name, source in [
        ("tuple-negative", "value = x.-1"),
        ("tuple-fractional", "value = x.1.5"),
        ("axis-negative", "value = vmap(f, axis=-1)"),
        ("axis-fractional", "value = vmap(f, axis=1.5)"),
    ]:
        add("SourceProgram", "parse-surf", name, source, None)
    for value in [0, 2, 9007199254740993]:
        add(
            "SourceProgram",
            "check-surf",
            f"tuple-bound-{value}",
            f"value = (7, 8).{value}",
            {"admitted": True} if value == 0 else None,
            None if value == 0 else "index",
        )
    function = "def f(x: tensor[3, f32]) -> tensor[3, f32] = x\n"
    for value in [None, 1, 2, 9007199254740993]:
        source = (
            function
            + "value = vmap(f"
            + ("" if value is None else f", axis={value}")
            + ")"
        )
        valid = value is None or value == 1
        add(
            "SourceProgram",
            "check-surf",
            f"axis-bound-{value}",
            source,
            {"admitted": True} if valid else None,
            None if valid else "axis",
        )
    return cases


def materialization_cases():
    cases = []

    def add(carrier, codec, name, value, expected, error=None):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/{name}",
                "source-materialization",
                carrier,
                codec,
                value,
                expected,
                error,
            )
        )

    def dto(carrier, name, value, expected):
        add(carrier, "json", name, canonical(value), {"syntax": expected})

    # Exact lexical carriers precede target-width rounding, including a raw
    # suffix spelling which only the later source boundary may authorize.
    for value in [-(2**63), -9007199254740993, 0, 9007199254740993, 2**63 - 1]:
        atom = {"type": "int", "value": value}
        dto("WireDeepAtom", str(value), atom, atom)
        for suffix in [None, "i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"]:
            literal = _literal("int", value, suffix)
            dto("WireLiteral", f"integer-{value}-{suffix}", literal, literal)
    for value in [
        0.0,
        -0.0,
        5e-324,
        2.2250738585072014e-308,
        1.0000000000000002,
        1.7976931348623157e308,
    ]:
        bits = _float(value)
        dto(
            "WireDeepAtom",
            bits["f64_bits"],
            {"type": "float", "value": value},
            {"type": "float", "value": bits},
        )
        for suffix in [None, "f16", "bf16", "f32", "f64"]:
            dto(
                "WireLiteral",
                f"float-{bits['f64_bits']}-{suffix}",
                _literal("float", value, suffix),
                _literal("float", bits, suffix),
            )
    for suffix in ["future", "i8"]:
        dto(
            "WireLiteral",
            f"unadmitted-{suffix}",
            _literal("float", 1.0, suffix),
            _literal("float", _float(1.0), suffix),
        )
    for carrier, discriminator in [("WireDeepAtom", "type"), ("WireLiteral", "kind")]:
        for kind, values in [
            ("int", ["null", '"1"', "1.0", "9223372036854775808"]),
            ("float", ["null", '"NaN"', "1e999"]),
        ]:
            variants = (kind, "typed_" + kind) if carrier == "WireLiteral" else (kind,)
            for variant in variants:
                suffix = ',"suffix":"i64"' if kind == "int" else ',"suffix":"f64"'
                if variant == kind:
                    suffix = ""
                for index, value in enumerate(values):
                    add(
                        carrier,
                        "json",
                        f"invalid-{variant}-{index}",
                        f'{{"{discriminator}":"{variant}","value":{value}{suffix}}}',
                        None,
                    )
        add(
            carrier,
            "json",
            "unknown-discriminator",
            f'{{"{discriminator}":"future","value":1}}',
            None,
            "unknown variant",
        )

    # These DTOs preserve future/malformed syntax, yet cannot authorize it.
    unknown = _definition(_node("future_form", _atom("int", 1)))
    bad_type = _definition(_node("lit", _atom("int", 1), type=_atom("bool", False)))
    opaque = {"kind": {"type": "extension_data", "syntax": "(unclosed"}}
    for name, expression in [
        ("unknown-form", unknown[0]),
        ("bad-live-type", bad_type[0]),
        ("unparsed-extension", opaque),
    ]:
        dto("WireDeepExpr", name, expression, expression)
    add(
        "WireDeepExpr",
        "json",
        "unknown-wire-tag",
        '{"kind":{"type":"future"}}',
        None,
        "unknown variant",
    )
    add(
        "WireDeepExpr",
        "json",
        "nested-fractional-integer",
        canonical(_node("lit", _atom("int", 1.5))),
        None,
    )

    integer_types = {suffix: suffix for suffix in ["i8", "i16", "i32", "i64"]}
    float_types = {suffix: suffix for suffix in ["f16", "bf16", "f32", "f64"]}
    for suffix, prim in (integer_types | float_types).items():
        # Large source values must remain exact even when this target dtype
        # will reject their range or round them at a subsequent semantic stage.
        value = 9007199254740993
        metadata = {"type": _node("t-prim", _atom("symbol", prim))}
        if suffix in float_types:
            metadata["literal_source"] = _atom("symbol", "integer")
        expected = _definition(_node("lit", _atom("int", value), **metadata))
        source = f"(def {{}} value (lit {{}} {value}{suffix}))"
        add(
            "SourceProgram",
            "parse-deep",
            f"integer-{suffix}",
            source,
            {"syntax": expected},
        )
    for suffix in float_types:
        value = 1.0000000000000002
        source = f"value = {value}{suffix}"
        literal = _literal("float", _float(value), suffix)
        add(
            "SourceProgram",
            "parse-surf",
            f"float-{suffix}",
            source,
            {"syntax": _surf_literal(literal)},
        )
        deep = _definition(
            _node(
                "lit",
                _atom("float", _float(value)),
                type=_node("t-prim", _atom("symbol", suffix)),
            )
        )
        add(
            "SourceProgram",
            "parse-deep",
            f"float-{suffix}",
            f"(def {{}} value (lit {{}} {value}{suffix}))",
            {"syntax": deep},
        )
    for suffix in integer_types:
        add(
            "SourceProgram",
            "parse-surf",
            f"integer-{suffix}",
            f"value = 7{suffix}",
            {"syntax": _surf_literal(_literal("int", 7, suffix))},
        )

    # Desugaring is a source producer with its own API. Its derived span IDs
    # are preserved as opaque strings; their spelling confers no authority.
    desugared = _definition(
        _node(
            "lit",
            _atom("int", 9007199254740993),
            type=_node("t-prim", _atom("symbol", "i64")),
            span=_atom("str", "surf:8..27"),
        ),
        span=_atom("str", "surf:0..27"),
    )
    add(
        "SourceProgram",
        "desugar-surf",
        "int64-wide",
        "value = 9007199254740993i64",
        {"syntax": desugared},
    )

    for name, token, scalar in [
        (
            "int64-wide",
            "9007199254740993i64",
            {"dtype": "int64", "value": 9007199254740993},
        ),
        (
            "int64-negative-wide",
            "-9007199254740993i64",
            {"dtype": "int64", "value": -9007199254740993},
        ),
        ("f64-negative-zero", "-0.0f64", {"dtype": "f64", "bits": "8000000000000000"}),
        (
            "f64-lexical-bit",
            "1.0000000000000002f64",
            {"dtype": "f64", "bits": "3ff0000000000001"},
        ),
        ("f32-integer-origin", "16777217f32", {"dtype": "f32", "bits": "4b800000"}),
        (
            "f64-finite-max",
            "1.7976931348623157e308f64",
            {"dtype": "f64", "bits": "7fefffffffffffff"},
        ),
        (
            "f64-finite-min",
            "-1.7976931348623157e308f64",
            {"dtype": "f64", "bits": "ffefffffffffffff"},
        ),
        # At 2**62, f32 spacing is 2**39. This exact integer is one
        # above the midpoint, so direct rounding chooses the upper float.
        # A prior f64 conversion loses that one and ties to the lower float.
        (
            "f32-no-double-round",
            f"{2**62 + 2**38 + 1}f32",
            {"dtype": "f32", "bits": "5e800001"},
        ),
    ]:
        deep_source = f"(def {{}} value (lit {{}} {token}))"
        for codec in ["check-deep", "eval-deep"]:
            expected = (
                {"admitted": True}
                if codec.startswith("check")
                else {
                    "roots": [
                        {"name": "value", "value": {"type": "scalar", "value": scalar}}
                    ]
                }
            )
            add("SourceProgram", codec, name, deep_source, expected)
        if not name.startswith("f32-"):
            for codec in ["check-surf", "eval-surf"]:
                expected = (
                    {"admitted": True}
                    if codec.startswith("check")
                    else {
                        "roots": [
                            {
                                "name": "value",
                                "value": {"type": "scalar", "value": scalar},
                            }
                        ]
                    }
                )
                add("SourceProgram", codec, name, f"value = {token}", expected)

    for token in ["1.0i8", "1.0future", "1e999f64"]:
        for codec in ["parse-surf", "desugar-surf", "check-surf", "eval-surf"]:
            add("SourceProgram", codec, f"invalid-{token}", f"value = {token}", None)
        for codec in ["parse-deep", "check-deep", "eval-deep"]:
            add(
                "SourceProgram",
                codec,
                f"invalid-{token}",
                f"(def {{}} value (lit {{}} {token}))",
                None,
            )
    for token in ["1e999", "-1e999", "-1e999f32"]:
        for codec in [
            "parse-deep",
            "check-deep",
            "eval-deep",
            "validate-deep",
            "decompile-deep",
        ]:
            add(
                "SourceProgram",
                codec,
                f"overflow-{token}",
                f"(def {{}} value (lit {{}} {token}))",
                None,
                "invalid number",
            )
    for value in [-0.0, 1.7976931348623157e308]:
        deep = _definition(
            _node(
                "lit",
                _atom("float", _float(value)),
                type=_node("t-prim", _atom("symbol", "f64")),
            )
        )
        add(
            "SourceProgram",
            "parse-deep",
            f"finite-bits-{value}",
            f"(def {{}} value (lit {{}} {value}f64))",
            {"syntax": deep},
        )

    for name, body, reason in [
        ("unmarked-integer-float", "(lit {type: (t-prim {} f64)} 1)", "literal"),
        (
            "integer-marker-integer-type",
            "(lit {type: (t-prim {} i64), literal_source: integer} 1)",
            "literal_source",
        ),
        (
            "integer-marker-float-atom",
            "(lit {type: (t-prim {} f64), literal_source: integer} 1.0)",
            "literal_source",
        ),
        ("out-of-range-integer", "(lit {} 128i8)", "i8"),
    ]:
        for codec in ["check-deep", "eval-deep"]:
            add("SourceProgram", codec, name, f"(def {{}} value {body})", None, reason)

    # Defined live keys must re-enter expression/metadata admission; a raw DTO
    # or a well-formed enclosing declaration cannot bless malformed descendants.
    for key in ["property_seed", "property_samples", "property_tolerance"]:
        good = f"(def {{{key}: (lit {{}} 1)}} value (lit {{}} 7))"
        expected = _definition(
            _node("lit", _atom("int", 7)), **{key: _node("lit", _atom("int", 1))}
        )
        add("SourceProgram", "parse-deep", f"live-{key}", good, {"syntax": expected})
        add("SourceProgram", "check-deep", f"live-{key}", good, {"admitted": True})
        for name, payload, reason in [
            ("bare", "bare_name", key),
            ("nested-shape", "(lit {span: 1} 1)", "span"),
            ("unknown-tag", "(future_form {} 1)", "future_form"),
            (
                "nested-duplicate",
                "(lit {type: (t-prim {} i32), type: (t-prim {} i32)} 1)",
                "type",
            ),
        ]:
            for codec in ["parse-deep", "check-deep", "eval-deep"]:
                preserved = name == "unknown-tag" and codec == "parse-deep"
                parsed = _definition(
                    _node("lit", _atom("int", 7)),
                    **{key: _node("future_form", _atom("int", 1))},
                )
                add(
                    "SourceProgram",
                    codec,
                    f"{key}-{name}",
                    f"(def {{{key}: {payload}}} value (lit {{}} 7))",
                    {"syntax": parsed} if preserved else None,
                    None if preserved else reason,
                )

    # Structured annotation containers retain live children too.
    preconditions = _definition(
        _node("lit", _atom("int", 7)),
        property_preconditions=_node("tuple", _node("lit", _atom("bool", True))),
    )
    invariant = [
        _node(
            "deftype",
            _atom("symbol", "T"),
            {"kind": {"type": "list", "elements": []}},
            _node("variant", _atom("symbol", "T")),
            opaque=_atom("bool", True),
            invariant=_node(
                "fn",
                _node("params", _atom("symbol", "x")),
                _node("lit", _atom("bool", True)),
            ),
        )
    ]
    for name, template, expected in [
        (
            "property_preconditions",
            "(def {property_preconditions: (tuple {} BODY)} value (lit {} 7))",
            preconditions,
        ),
        (
            "invariant",
            "(deftype {opaque: true, invariant: (fn {} (params {} x) BODY)} T () (variant {} T))",
            invariant,
        ),
    ]:
        add(
            "SourceProgram",
            "parse-deep",
            f"live-{name}",
            template.replace("BODY", "(lit {} true)"),
            {"syntax": expected},
        )
        for label, body, reason in [
            ("bare", "bare_name", name),
            ("unknown", "(future_form {} true)", "future_form"),
            ("shape", "(lit {span: 1} true)", "span"),
        ]:
            for codec in ["parse-deep", "check-deep", "eval-deep"]:
                preserved = label == "unknown" and codec == "parse-deep"
                parsed = (
                    _definition(
                        _node("lit", _atom("int", 7)),
                        property_preconditions=_node(
                            "tuple", _node("future_form", _atom("bool", True))
                        ),
                    )
                    if name == "property_preconditions"
                    else [
                        _node(
                            "deftype",
                            _atom("symbol", "T"),
                            {"kind": {"type": "list", "elements": []}},
                            _node("variant", _atom("symbol", "T")),
                            opaque=_atom("bool", True),
                            invariant=_node(
                                "fn",
                                _node("params", _atom("symbol", "x")),
                                _node("future_form", _atom("bool", True)),
                            ),
                        )
                    ]
                )
                add(
                    "SourceProgram",
                    codec,
                    f"{name}-{label}",
                    template.replace("BODY", body),
                    {"syntax": parsed} if preserved else None,
                    None if preserved else reason,
                )

    for name, source, reason in [
        ("bare-runtime", "(def {} value bare_name)", "bare_name"),
        ("invalid-live-type", "(def {type: false} value (lit {} 7))", "type"),
        (
            "duplicate-type",
            "(def {type: (t-prim {} i32), type: (t-prim {} i32)} value (lit {} 7))",
            "type",
        ),
        (
            "duplicate-extension",
            "(def {custom: 1, custom: 2} value (lit {} 7))",
            "custom",
        ),
        (
            "duplicate-span-extension",
            "(def {span_future: 1, span_future: 2} value (lit {} 7))",
            "span_future",
        ),
        ("wrong-placement", '(def {} value (var {surf_path: "X"} x))', "surf_path"),
        (
            "wrong-live-placement",
            "(def {} value (var {property_seed: (lit {} 1)} x))",
            "property_seed",
        ),
        ("unknown-surf-key", "(def {surf_future: 1} value (lit {} 7))", "surf_future"),
    ]:
        for codec in [
            "parse-deep",
            "check-deep",
            "validate-deep",
            "decompile-deep",
            "eval-deep",
        ]:
            add("SourceProgram", codec, name, source, None, reason)
    good = "(def {} value (lit {} 7))"
    add("SourceProgram", "validate-deep", "valid-program", good, {"admitted": True})
    add(
        "SourceProgram",
        "decompile-deep",
        "valid-program",
        good,
        {"surf": "value = 7\n"},
    )
    unknown_source = "(def {} value (future_form {} 1))"
    add(
        "SourceProgram",
        "parse-deep",
        "preserved-unknown",
        unknown_source,
        {"syntax": unknown},
    )
    for codec in ["check-deep", "validate-deep", "decompile-deep", "eval-deep"]:
        add(
            "SourceProgram",
            codec,
            "unknown-not-executable",
            unknown_source,
            None,
            "future_form",
        )

    for payload, valid, reason in [
        ("(lit {} 7)", True, None),
        ("(lit {type: (t-prim {} i64)} 9007199254740993)", True, None),
        ("bare_name", False, "property_seed"),
        ("(lit {type: false} 1)", False, "type"),
        ("(lit {type: (t-prim {} i32), type: (t-prim {} i32)} 1)", False, "type"),
        ('(var {surf_path: "X"} x)', False, "surf_path"),
        ("(future_form {} 1)", False, "future_form"),
    ]:
        source = f"(def {{source: (macro_name {payload})}} value (lit {{}} 1))"
        add("SourceHistory", "transport", payload, source, {"data": payload})
        add(
            "SourceHistory",
            "live-annotation",
            payload,
            source,
            {"admitted": True} if valid else None,
            reason,
        )
    add(
        "SourceHistory",
        "transport",
        "malformed-history-owner",
        "(def {source: false} value (lit {} 1))",
        None,
        "source",
    )

    for data in [
        "(lit {} 7)",
        "(future {type: false, type: 1} bare)",
        "(9007199254740993 -0.0)",
    ]:
        add("ExtensionData", "data", data, data, {"data": data})
        add("ExtensionData", "runtime", data, data, None, "extension")
    add("ExtensionData", "data", "unbalanced", "(unclosed", None)
    for codec, source in [("file", good), ("runtime", "(lit {} 7)")]:
        add(
            "RawSourceAdmission",
            codec,
            "valid",
            source,
            {"forms": 1, "deep": source + "\n"},
        )
        add(
            "RawSourceAdmission",
            codec,
            "bare-name",
            "bare_name",
            None,
            "bare identifier" if codec == "file" else "bare_name",
        )
    return cases + source_parameter_cases()


def source_field_evidence():
    """Each decided source field owes transport and normal source admission."""
    schema = "chelis_compiler_api::schema::"
    deep_integer = (
        "SourceProgram/eval-deep/int64-wide",
        "SourceProgram/eval-deep/out-of-range-integer",
    )
    deep_float = (
        "SourceProgram/eval-deep/f64-negative-zero",
        "SourceProgram/eval-deep/invalid-1e999f64",
    )
    surf_integer = (
        "SourceProgram/eval-surf/int64-wide",
        "SourceProgram/eval-surf/invalid-1.0i8",
    )
    surf_float = (
        "SourceProgram/eval-surf/f64-negative-zero",
        "SourceProgram/eval-surf/invalid-1e999f64",
    )
    rows = {
        "WireDeepAtom::Int.value": (
            ("WireDeepAtom/json/9007199254740993", "WireDeepAtom/json/invalid-int-2"),
            deep_integer,
        ),
        "WireDeepAtom::Float.value": (
            ("WireDeepAtom/json/8000000000000000", "WireDeepAtom/json/invalid-float-2"),
            deep_float,
        ),
        "WireLiteral::Int.value": (
            (
                "WireLiteral/json/integer-9007199254740993-None",
                "WireLiteral/json/invalid-int-2",
            ),
            surf_integer,
        ),
        "WireLiteral::TypedInt.value": (
            (
                "WireLiteral/json/integer-9007199254740993-i64",
                "WireLiteral/json/invalid-typed_int-2",
            ),
            surf_integer,
            (
                "SourceProgram/eval-deep/f32-no-double-round",
                "SourceProgram/eval-deep/integer-marker-float-atom",
            ),
        ),
        "WireLiteral::Float.value": (
            (
                "WireLiteral/json/float-8000000000000000-None",
                "WireLiteral/json/invalid-float-2",
            ),
            surf_float,
        ),
        "WireLiteral::TypedFloat.value": (
            (
                "WireLiteral/json/float-8000000000000000-f64",
                "WireLiteral/json/invalid-typed_float-2",
            ),
            surf_float,
            (
                "WireLiteral/json/unadmitted-future",
                "SourceProgram/check-surf/invalid-1.0future",
            ),
        ),
        "WireSurfExpr::TupleGet.index": (
            (
                "WireSurfExpr/json/parameter-index-9007199254740993",
                "WireSurfExpr/json/parameter-invalid-index-0",
            ),
            (
                "SourceProgram/check-surf/parameter-tuple-bound-0",
                "SourceProgram/check-surf/parameter-tuple-bound-9007199254740993",
            ),
        ),
        "WireSurfExpr::Vmap.axis": (
            (
                "WireSurfExpr/json/parameter-axis-None",
                "WireSurfExpr/json/parameter-invalid-axis-0",
            ),
            (
                "SourceProgram/check-surf/parameter-axis-bound-1",
                "SourceProgram/check-surf/parameter-axis-bound-2",
            ),
            (
                "SourceProgram/check-surf/parameter-axis-bound-None",
                "SourceProgram/check-surf/parameter-axis-bound-9007199254740993",
            ),
        ),
    }
    return {schema + field: pairs for field, pairs in rows.items()}


def validate_materialization_execution(cases, outcomes):
    """Bind field and live-source obligations to actual selected observations."""
    from capacity_census_graph import GraphError
    from capacity_census_wire_fixed_roles import compiler_fixed_field_contracts

    fields = source_field_evidence()
    expected = {
        c.field
        for c in compiler_fixed_field_contracts()
        if c.role in {"source-integer", "source-float"}
    }
    if set(fields) != expected:
        raise GraphError(
            "materialization execution fields differ from decided source roles"
        )
    pairs = [pair for field_pairs in fields.values() for pair in field_pairs]
    for key in ("property_seed", "property_samples", "property_tolerance"):
        for problem in ("bare", "nested-shape", "nested-duplicate", "unknown-tag"):
            pairs.append(
                (
                    f"SourceProgram/check-deep/live-{key}",
                    f"SourceProgram/check-deep/{key}-{problem}",
                )
            )
    for codec in ("file", "runtime"):
        pairs.append(
            (
                f"RawSourceAdmission/{codec}/valid",
                f"RawSourceAdmission/{codec}/bare-name",
            )
        )
    pairs.extend(
        (
            (
                "SourceHistory/live-annotation/(lit {} 7)",
                "SourceHistory/live-annotation/(lit {type: false} 1)",
            ),
            (
                "SourceHistory/live-annotation/(lit {} 7)",
                "SourceHistory/live-annotation/(future_form {} 1)",
            ),
            ("ExtensionData/data/(lit {} 7)", "ExtensionData/runtime/(lit {} 7)"),
        )
    )
    selected = {c.identity: c for c in cases}
    executed = {o.identity: o for o in outcomes}
    if len(selected) != len(cases) or len(executed) != len(outcomes):
        raise GraphError("duplicate materialization execution identity")
    required = set()
    for accepted, rejected in pairs:
        for identity, positive in ((accepted, True), (rejected, False)):
            case, outcome = selected.get(identity), executed.get(identity)
            if (
                case is None
                or outcome is None
                or not outcome.selected
                or not outcome.executed
                or outcome.outcome != "passed"
                or len(outcome.observation_sha256) != 64
            ):
                raise GraphError(
                    f"missing or failed materialization execution {identity}"
                )
            if (case.expected is not None) != positive:
                raise GraphError(
                    f"materialization execution polarity changed for {identity}"
                )
            required.add(identity)
    return tuple(sorted(required))
