//! spec/10 §3.3: lexical source numbers and executable-source admission.

pub use chelis_compiler_api::schema;
#[path = "../src/source_wire.rs"]
mod source_wire;

use chelis_compiler_api::compiler;
use chelis_deep::{Atom, Expr, Span};
use chelis_surf::ast::{Literal, LiteralSuffix};
use schema::{DesugarRequest, ParseRequest, SourceKind, WireDeepAtom, WireLiteral};
use serde::{
    Deserialize,
    de::{
        IntoDeserializer, Visitor,
        value::{Error, MapDeserializer},
    },
};
use serde_json::Value;

fn roundtrip<T: serde::Serialize + serde::de::DeserializeOwned>(value: &T) -> Value {
    let text = serde_json::to_string(value).unwrap();
    let decoded: T = serde_json::from_str(&text).unwrap();
    assert_eq!(serde_json::to_string(&decoded).unwrap(), text);
    serde_json::from_str(&text).unwrap()
}

#[test]
fn native_source_producers_preserve_finite_lexical_bits_and_suffixes() {
    for value in [
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        1.0000000000000002,
    ] {
        let plain = source_wire::wire_literal(&Literal::Float(value)).unwrap();
        assert_eq!(
            roundtrip(&plain)["value"].as_f64().unwrap().to_bits(),
            value.to_bits()
        );
        for suffix in [
            LiteralSuffix::F16,
            LiteralSuffix::Bf16,
            LiteralSuffix::F32,
            LiteralSuffix::F64,
        ] {
            let typed = source_wire::wire_literal(&Literal::TypedFloat(value, suffix)).unwrap();
            let json = roundtrip(&typed);
            assert_eq!(json["value"].as_f64().unwrap().to_bits(), value.to_bits());
            assert_eq!(json["suffix"], suffix.as_str());
        }
        let deep =
            source_wire::wire_deep_expr(&Expr::Atom(Atom::Float(value), Span::new(4, 7))).unwrap();
        assert_eq!(
            roundtrip(&deep)["kind"]["atom"]["value"]
                .as_f64()
                .unwrap()
                .to_bits(),
            value.to_bits()
        );
    }
}

#[test]
fn native_source_producers_reject_nonfinite_literals_including_nested_deep() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for literal in [
            Literal::Float(value),
            Literal::TypedFloat(value, LiteralSuffix::F16),
            Literal::TypedFloat(value, LiteralSuffix::Bf16),
            Literal::TypedFloat(value, LiteralSuffix::F32),
            Literal::TypedFloat(value, LiteralSuffix::F64),
        ] {
            let error = source_wire::wire_literal(&literal).expect_err("nonfinite native literal");
            assert!(error.contains("finite"), "{error}");
        }
        let atom = Expr::Atom(Atom::Float(value), Span::new(4, 7));
        for expr in [atom.clone(), Expr::BareList(vec![atom], Span::new(0, 12))] {
            let error = source_wire::wire_deep_expr(&expr).expect_err("nonfinite native Deep atom");
            assert!(error.contains("finite"), "{error}");
        }
    }
}

#[test]
fn source_integer_producers_keep_all_int64_bits_before_dtype_admission() {
    for value in [
        i64::MIN,
        -9_007_199_254_740_993,
        9_007_199_254_740_993,
        i64::MAX,
    ] {
        let literal = source_wire::wire_literal(&Literal::Int(value)).unwrap();
        assert_eq!(roundtrip(&literal)["value"].as_i64(), Some(value));
        for suffix in [
            LiteralSuffix::I8,
            LiteralSuffix::I16,
            LiteralSuffix::I32,
            LiteralSuffix::I64,
            LiteralSuffix::F16,
            LiteralSuffix::Bf16,
            LiteralSuffix::F32,
            LiteralSuffix::F64,
        ] {
            let literal = source_wire::wire_literal(&Literal::TypedInt(value, suffix)).unwrap();
            let json = roundtrip(&literal);
            assert_eq!(json["value"].as_i64(), Some(value));
            assert_eq!(json["suffix"], suffix.as_str());
        }
        let deep =
            source_wire::wire_deep_expr(&Expr::Atom(Atom::Int(value), Span::new(0, 20))).unwrap();
        assert_eq!(
            roundtrip(&deep)["kind"]["atom"]["value"].as_i64(),
            Some(value)
        );
    }
}

// Unlike JSON text, another serde data source can actually supply NaN/inf.
// Feed the real enum deserializers so rejection is proved at their carrier.
enum Input<'a> {
    Text(&'a str),
    Float(f64),
}
impl<'de> serde::Deserializer<'de> for Input<'de> {
    type Error = Error;
    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        match self {
            Self::Text(value) => visitor.visit_borrowed_str(value),
            Self::Float(value) => visitor.visit_f64(value),
        }
    }
    serde::forward_to_deserialize_any! { bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map struct enum identifier ignored_any }
}
impl<'de> IntoDeserializer<'de, Error> for Input<'de> {
    type Deserializer = Self;
    fn into_deserializer(self) -> Self {
        self
    }
}

fn literal_from_serde(value: f64, typed: bool) -> Result<WireLiteral, Error> {
    let mut fields = vec![
        (
            "kind",
            Input::Text(if typed { "typed_float" } else { "float" }),
        ),
        ("value", Input::Float(value)),
    ];
    if typed {
        fields.push(("suffix", Input::Text("f64")));
    }
    WireLiteral::deserialize(MapDeserializer::new(fields.into_iter()))
}
fn atom_from_serde(value: f64) -> Result<WireDeepAtom, Error> {
    WireDeepAtom::deserialize(MapDeserializer::new(
        [
            ("type", Input::Text("float")),
            ("value", Input::Float(value)),
        ]
        .into_iter(),
    ))
}

#[test]
fn every_float_dto_rejects_nonfinite_serde_input_but_accepts_signed_zero() {
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for typed in [false, true] {
            let error = literal_from_serde(value, typed).expect_err("nonfinite DTO literal");
            assert!(error.to_string().contains("finite"), "{error}");
        }
        let error = atom_from_serde(value).expect_err("nonfinite DTO atom");
        assert!(error.to_string().contains("finite"), "{error}");
    }
    for value in [0.0_f64, -0.0, 1.0] {
        for typed in [false, true] {
            assert_eq!(
                roundtrip(&literal_from_serde(value, typed).unwrap())["value"]
                    .as_f64()
                    .unwrap()
                    .to_bits(),
                value.to_bits()
            );
        }
        assert_eq!(
            roundtrip(&atom_from_serde(value).unwrap())["value"]
                .as_f64()
                .unwrap()
                .to_bits(),
            value.to_bits()
        );
    }
}

#[test]
fn dto_numeric_grammar_rejects_null_strings_fractional_integers_and_overflow() {
    for bad in [
        "null",
        "\"1\"",
        "1.0",
        "1e0",
        "1.5",
        "9223372036854775808",
        "-9223372036854775809",
    ] {
        for (kind, suffix) in [("int", ""), ("typed_int", ",\"suffix\":\"i64\"")] {
            assert!(
                serde_json::from_str::<WireLiteral>(&format!(
                    "{{\"kind\":\"{kind}\",\"value\":{bad}{suffix}}}"
                ))
                .is_err(),
                "{kind}: {bad}"
            );
        }
        assert!(
            serde_json::from_str::<WireDeepAtom>(&format!("{{\"type\":\"int\",\"value\":{bad}}}"))
                .is_err(),
            "{bad}"
        );
    }
    for bad in ["null", "\"NaN\"", "1e999"] {
        for (kind, suffix) in [("float", ""), ("typed_float", ",\"suffix\":\"f64\"")] {
            assert!(
                serde_json::from_str::<WireLiteral>(&format!(
                    "{{\"kind\":\"{kind}\",\"value\":{bad}{suffix}}}"
                ))
                .is_err(),
                "{kind}: {bad}"
            );
        }
        assert!(
            serde_json::from_str::<WireDeepAtom>(&format!(
                "{{\"type\":\"float\",\"value\":{bad}}}"
            ))
            .is_err()
        );
    }
}

#[test]
fn public_source_paths_preserve_integer_origin_and_closed_suffix_admission() {
    let integer = 9_007_199_254_740_993_i64;
    for suffix in ["i8", "i16", "i32", "i64", "f16", "bf16", "f32", "f64"] {
        let source = format!("(def {{}} value (lit {{}} {integer}{suffix}))");
        let parsed = compiler::parse(ParseRequest {
            source_kind: SourceKind::Deep,
            source,
        })
        .unwrap();
        let json = serde_json::to_string(&parsed).unwrap();
        assert!(json.contains(&integer.to_string()), "{suffix}: {json}");
        assert!(!json.contains("9007199254740992"), "{suffix}: {json}");
        if suffix.contains('f') {
            assert!(json.contains("literal_source"), "{suffix}: {json}");
        }
    }
    for suffix in ["f16", "bf16", "f32", "f64"] {
        let source = format!("value = 1.0000000000000002{suffix}");
        let parsed = compiler::parse(ParseRequest {
            source_kind: SourceKind::Surf,
            source: source.clone(),
        })
        .unwrap();
        assert!(
            serde_json::to_string(&parsed)
                .unwrap()
                .contains("1.0000000000000002")
        );
        let desugared = compiler::desugar(DesugarRequest { source }).unwrap();
        assert!(
            serde_json::to_string(&desugared.deep_ast)
                .unwrap()
                .contains("1.0000000000000002")
        );
    }
    for (kind, source) in [
        (SourceKind::Surf, "value = 1.0i8"),
        (SourceKind::Surf, "value = 1.0future"),
        (SourceKind::Surf, "value = 1e999f64"),
        (SourceKind::Deep, "(def {} value (lit {} 1.0i8))"),
        (SourceKind::Deep, "(def {} value (lit {} 1e999f64))"),
    ] {
        let error = compiler::parse(ParseRequest {
            source_kind: kind,
            source: source.into(),
        })
        .expect_err(source);
        if kind == SourceKind::Deep && source.contains("1e999") {
            assert_eq!(error.stage, "parse");
            assert!(error.errors.iter().any(|diagnostic| {
                diagnostic.kind() == chelis_vocab::DiagnosticKind::DeepParseError
                    && diagnostic.message.contains("invalid number")
            }));
        }
    }
}

#[test]
fn raw_dto_and_preserved_unknown_forms_do_not_establish_source_admission() {
    let raw: WireLiteral =
        serde_json::from_str("{\"kind\":\"typed_float\",\"value\":1.0,\"suffix\":\"future\"}")
            .unwrap();
    assert_eq!(roundtrip(&raw)["suffix"], "future");
    assert!(
        compiler::parse(ParseRequest {
            source_kind: SourceKind::Surf,
            source: "value = 1.0future".into()
        })
        .is_err()
    );
    let valid = "(def {type: (t-prim {} i32), doc: \"keep\"} value (lit {} 1))";
    assert!(
        compiler::parse(ParseRequest {
            source_kind: SourceKind::Deep,
            source: valid.into()
        })
        .is_ok()
    );
    for invalid in [
        "(def {type: false} value (lit {} 1))",
        "(def {type: (t-prim {} i32), type: (t-prim {} i32)} value (lit {} 1))",
    ] {
        assert!(
            compiler::parse(ParseRequest {
                source_kind: SourceKind::Deep,
                source: invalid.into()
            })
            .is_err(),
            "{invalid}"
        );
    }
    let source = "(def {} value (future-form {sentinel_meta: \"keep\"} (lit {} 1)))";
    let parsed = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: source.into(),
    })
    .unwrap();
    let json = serde_json::to_string(&parsed).unwrap();
    assert!(
        json.contains("future-form") && json.contains("sentinel_meta"),
        "{json}"
    );
    assert!(
        compiler::decompile(schema::DecompileRequest {
            source: source.into()
        })
        .is_err()
    );
}

#[test]
fn typed_deep_wire_bridge_preserves_complete_node_shape() {
    let exprs =
        chelis_deep::parse_and_stamp_file("(def {doc: \"test\"} root (var {} value))").unwrap();
    let wire = source_wire::wire_deep_expr(&exprs[0]).unwrap();
    let json = roundtrip(&wire);
    let elements = json["kind"]["elements"].as_array().unwrap();
    assert_eq!(elements.len(), 4);
    assert_eq!(elements[0]["kind"]["atom"]["value"], "def");
    assert_eq!(elements[1]["kind"]["entries"][0]["key"], "doc");
    assert_eq!(
        elements[1]["kind"]["entries"][0]["value"]["kind"]["atom"]["value"],
        "test"
    );
    assert_eq!(elements[2]["kind"]["atom"]["value"], "root");
    assert_eq!(
        elements[3]["kind"]["elements"][2]["kind"]["atom"]["value"],
        "value"
    );
}
