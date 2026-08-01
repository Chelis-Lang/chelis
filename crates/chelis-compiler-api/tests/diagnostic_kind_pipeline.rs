use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{ParseRequest, SourceKind, WireDiagnostic};
use chelis_vocab::DiagnosticKind;

#[test]
fn producer_kind_is_typed_and_serializes_with_the_stable_wire_spelling() {
    let error = compiler::parse(ParseRequest {
        source_kind: SourceKind::Surf,
        source: "(".to_string(),
    })
    .expect_err("malformed Surf must be rejected");
    let diagnostic = error.errors.first().expect("one parse diagnostic");

    assert_eq!(diagnostic.kind(), DiagnosticKind::SurfParseError);
    let json = serde_json::to_value(diagnostic).expect("serialize producer diagnostic");
    assert_eq!(json["kind"], DiagnosticKind::SurfParseError.as_str());
}

#[test]
fn wire_diagnostic_is_a_read_only_consumer_shape() {
    let json = r#"{
        "kind":"unsupported_feature",
        "message":"unsupported: example",
        "severity":1.0,
        "suggestions":[]
    }"#;
    let diagnostic: WireDiagnostic = serde_json::from_str(json).expect("wire diagnostic");

    assert_eq!(diagnostic.kind, "unsupported_feature");
    assert_eq!(
        DiagnosticKind::decode(&diagnostic.kind),
        Ok(DiagnosticKind::UnsupportedFeature)
    );
}
