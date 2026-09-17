//! chelis#730 Phase 2 contract tests for the HostType failure-state split.
//!
//! The value-level tests lock the logical and inference vocabulary. The
//! source-boundary tests lock the completed migration away from the anonymous
//! `Unknown` sentinel. C-host ABI parity lives in the backend crate's
//! `host_abi_tests` module.

use std::fs;
use std::path::Path;

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, Metadata};
use chelis_deep::parser::parse_str;
use chelis_ir::{
    ConcreteHostType, HostInferenceVar, HostPrecisionTerm, HostShapeSlot, HostShapeTerm,
    HostTensorTypeTerm, HostTypeDecodeError, HostTypeResolutionError, HostTypeTerm,
    decode_host_type, decode_host_type_metadata,
};
use chelis_types::types::Prim;

#[test]
fn every_active_narrow_scalar_remains_a_known_logical_type() {
    for precision in [Prim::F16, Prim::Bf16, Prim::Int8, Prim::Int16] {
        let resolved = HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision))
            .into_concrete()
            .expect("an active scalar precision is a known logical type");
        assert_eq!(resolved, ConcreteHostType::Scalar(precision));
    }
}

#[test]
fn named_polymorphic_states_preserve_their_kind_and_identity() {
    let type_err = HostTypeTerm::TypeVariable("element".into())
        .into_concrete()
        .expect_err("an unspecialized type variable is not concrete");
    assert_eq!(
        type_err,
        HostTypeResolutionError::UnresolvedTypeVariable {
            name: "element".into()
        }
    );

    let precision_err = HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
        precision: HostPrecisionTerm::Variable("p".into()),
        shape: HostShapeTerm::Concrete(Vec::new()),
    })
    .into_concrete()
    .expect_err("an unspecialized precision variable is not concrete");
    assert_eq!(
        precision_err,
        HostTypeResolutionError::UnresolvedPrecisionVariable { name: "p".into() }
    );

    let rank_err = HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
        precision: HostPrecisionTerm::Concrete(Prim::F32),
        shape: HostShapeTerm::Polymorphic(vec![HostShapeSlot::RankVariable("rank".into())]),
    })
    .into_concrete()
    .expect_err("an unspecialized rank variable is not concrete");
    assert_eq!(
        rank_err,
        HostTypeResolutionError::UnresolvedRankVariable {
            name: "rank".into()
        }
    );
}

#[test]
fn empty_collection_inference_does_not_choose_an_element_type() {
    let element = HostInferenceVar(17);
    let err = HostTypeTerm::List(Box::new(HostTypeTerm::InferenceVariable(element)))
        .into_concrete()
        .expect_err("an underconstrained empty list must stay underconstrained");
    assert_eq!(
        err,
        HostTypeResolutionError::UnresolvedInferenceVariable { variable: element }
    );
}

#[test]
fn divergence_has_no_value_representation() {
    assert_eq!(
        HostTypeTerm::Never.into_concrete(),
        Err(HostTypeResolutionError::NeverReachedValueBoundary)
    );
}

#[test]
fn manually_invalid_polymorphic_shape_fails_instead_of_panicking_or_defaulting() {
    let invalid = HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
        precision: HostPrecisionTerm::Concrete(Prim::F32),
        shape: HostShapeTerm::Polymorphic(Vec::new()),
    });
    assert_eq!(
        invalid.into_concrete(),
        Err(HostTypeResolutionError::InvalidPolymorphicShape)
    );
}

#[test]
fn missing_and_malformed_metadata_are_distinct_decode_failures() {
    assert_ne!(
        HostTypeDecodeError::MissingTypeMetadata,
        HostTypeDecodeError::MalformedTypeSyntax {
            detail: "type metadata was not a Deep type node".into()
        }
    );
    assert_ne!(
        HostTypeDecodeError::MalformedTypeSyntax {
            detail: "missing t-prim child".into()
        },
        HostTypeDecodeError::UnknownPrimitive {
            name: "float24".into()
        }
    );
}

fn parse_one(source: &str) -> chelis_deep::ast::Expr {
    parse_str(source)
        .expect("valid Deep fixture")
        .into_iter()
        .next()
        .expect("one Deep fixture")
}

#[test]
fn raw_decoder_preserves_exact_primitives_and_polymorphic_names() {
    for precision in [
        Prim::F32,
        Prim::F64,
        Prim::F16,
        Prim::Bf16,
        Prim::Int8,
        Prim::Int16,
        Prim::Int32,
        Prim::Int64,
        Prim::Bool,
        Prim::String,
    ] {
        let expr = parse_one(&format!("(t-prim {{}} {})", precision.name()));
        assert_eq!(
            decode_host_type(&expr),
            Ok(HostTypeTerm::Scalar(HostPrecisionTerm::Concrete(precision)))
        );
    }

    let tensor =
        parse_one("(t-tensor {} (d-rank {} pre) (d-name {} seq) (d-rank {} post) (t-var {} p))");
    assert_eq!(
        decode_host_type(&tensor),
        Ok(HostTypeTerm::PolymorphicTensor(HostTensorTypeTerm {
            precision: HostPrecisionTerm::Variable("p".into()),
            shape: HostShapeTerm::Polymorphic(vec![
                HostShapeSlot::RankVariable("pre".into()),
                HostShapeSlot::Dim(chelis_ir::DimInfo::Named("seq".into(), None)),
                HostShapeSlot::RankVariable("post".into()),
            ]),
        }))
    );
}

fn legacy_node(expr: &Expr) -> Expr {
    let Expr::Node(node, span) = expr else {
        panic!("test fixture must be a decoded node");
    };
    Expr::List(node.to_list(*span), *span)
}

#[test]
fn raw_decoder_matches_successor_and_legacy_carriers() {
    for source in [
        "(t-prim {} i64)",
        "(t-ref {} (t-prim {} f32))",
        "(t-tensor {} (d-lit {} 4) (t-var {} p))",
        "(t-adt {} Option (t-prim {} bool))",
        "(t-tuple {} (t-prim {} i32) (t-unit {}))",
        "(t-fn {} (t-prim {} i32) (t-prim {} bool))",
    ] {
        let successor = parse_one(source);
        let legacy = legacy_node(&successor);
        assert_eq!(
            decode_host_type(&legacy),
            decode_host_type(&successor),
            "{source}"
        );
    }
}

#[test]
fn raw_decoder_fails_closed_for_every_non_type_carrier_class() {
    let span = Span::new(0, 0);
    let malformed = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(chelis_deep::DeepTag::TPrim), span),
                Expr::Atom(Atom::Name("not-metadata".into()), span),
                Expr::Atom(Atom::Name("i64".into()), span),
            ],
        },
        span,
    );
    let rejected = [
        Expr::BareList(vec![], span),
        Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "future-type".into(),
            meta: Metadata::default(),
            children: vec![],
            span,
        })),
        Expr::Atom(Atom::Name("i64".into()), span),
        Expr::Map(Metadata::default(), span),
        malformed,
    ];

    for expr in rejected {
        assert!(
            matches!(
                decode_host_type(&expr),
                Err(HostTypeDecodeError::MalformedTypeSyntax { .. })
            ),
            "{expr:?}"
        );
    }
}

#[test]
fn raw_decoder_rejects_malformed_and_unknown_syntax_without_a_term() {
    // The parser now rejects a zero-child `t-prim` before it can reach the
    // decoder. Construct the controlled legacy mutation directly so this
    // remains a decoder failure-state test rather than a parser test.
    let span = Span::new(0, 0);
    let malformed = Expr::List(
        List {
            elements: vec![
                Expr::Atom(Atom::Tag(chelis_deep::DeepTag::TPrim), span),
                Expr::Map(Metadata::default(), span),
            ],
        },
        span,
    );
    assert!(matches!(
        decode_host_type(&malformed),
        Err(HostTypeDecodeError::MalformedTypeSyntax { .. })
    ));

    let unknown = parse_one("(t-prim {} float24)");
    assert_eq!(
        decode_host_type(&unknown),
        Err(HostTypeDecodeError::UnknownPrimitive {
            name: "float24".into()
        })
    );

    let absent = parse_one("(lit {} 1)");
    assert_eq!(
        decode_host_type_metadata(&absent),
        Err(HostTypeDecodeError::MissingTypeMetadata)
    );

    assert!(parse_str("(lit {type: nope} 1)").is_err());
    // Invalid type payloads cannot enter an AST annotation, including via serde.
    assert!(
        chelis_deep::annotations::TypeSyntax::try_new(Expr::Atom(Atom::Name("nope".into()), span),)
            .is_err()
    );
    let malformed =
        serde_json::json!({"entries": [["type", Expr::Atom(Atom::Name("nope".into()), span)]]});
    assert!(serde_json::from_value::<Metadata>(malformed).is_err());
}

fn production_legacy_unknown_lines(source: &str) -> Vec<(usize, &str)> {
    source
        .lines()
        .enumerate()
        .filter_map(|(index, line)| {
            let code = line.split("//").next().unwrap_or_default();
            code.contains("HostType::Unknown")
                .then_some((index + 1, line))
        })
        .collect()
}

fn host_sources() -> (String, String) {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let host = fs::read_to_string(crate_dir.join("src/host.rs")).expect("read host lowerer");
    let backend = fs::read_to_string(
        crate_dir
            .parent()
            .expect("crates directory")
            .join("chelis-backend-c/src/host_emit.rs"),
    )
    .expect("read C host emitter");
    (host, backend)
}

#[test]
fn legacy_unknown_producers_and_consumers_are_gone() {
    let (host, backend) = host_sources();
    let host_hits = production_legacy_unknown_lines(&host);
    let backend_hits = production_legacy_unknown_lines(&backend);
    assert!(
        host_hits.is_empty(),
        "the host lowerer still carries the legacy anonymous state; every producer and \
         consumer must migrate to HostTypeTerm/decode errors/resolution errors: \
         {host_hits:?}"
    );
    assert!(
        backend_hits.is_empty(),
        "the C emitter still interprets the legacy anonymous state: {backend_hits:?}"
    );
}

#[test]
fn raw_host_type_decoding_is_result_typed() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let state = fs::read_to_string(crate_dir.join("src/host_type_state.rs"))
        .expect("read typed host-state module");
    assert!(
        state.contains("Result<HostTypeTerm, HostTypeDecodeError>"),
        "raw Deep host-type decoding must return a typed Result; defining the error \
         vocabulary without using it at the boundary is insufficient"
    );
    let definition = ["fn stamped_", "parts"].concat();
    let call = ["stamped_", "parts("].concat();
    assert!(
        !state.contains(&definition) && !state.contains(&call),
        "E5b requires every host-type read to disposition ExprCarrier directly"
    );
}

#[test]
fn c_emitter_has_no_infallible_or_default_host_type_conversion() {
    let (_, backend) = host_sources();
    assert!(
        backend.contains("fn c_type(ty: &HostAbiType)"),
        "C declarations must consume the backend-owned resolved ABI vocabulary"
    );
    assert!(
        !backend.contains("HostType::Unknown => \"void*\""),
        "an unresolved logical type must never become a plausible ABI pointer"
    );
    assert!(
        !backend.contains("_ => \"chelis_value_from_int64(0)\""),
        "boxing an unsupported host type must return Unsupported, not a boxed zero"
    );
    assert!(
        !backend.contains("_ => \"0\".to_string()"),
        "unboxing an unsupported host type must return Unsupported, not zero"
    );
}
