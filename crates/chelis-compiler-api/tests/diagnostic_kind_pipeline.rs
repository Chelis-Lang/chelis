use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{
    ApiEnvelope, BatchRequest, CheckRequest, ParseRequest, SourceKind, WireApiEnvelope,
    WireBatchResult, WireBatchResultEnvelope, WireDiagnostic,
};
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

#[test]
fn wire_api_envelope_decodes_success_and_complete_failure_shapes() {
    let success: WireApiEnvelope<serde_json::Value> =
        serde_json::from_str(r#"{"ok":true,"result":{"value":42}}"#).expect("success envelope");
    match success {
        WireApiEnvelope::Success(success) => {
            assert!(success.ok);
            assert_eq!(success.result["value"], 42);
        }
        WireApiEnvelope::Failure(_) => panic!("expected success envelope"),
    }

    let producer: ApiEnvelope<serde_json::Value> = ApiEnvelope::invalid_request("bad request");
    let producer_json = serde_json::to_string(&producer).expect("serialize producer envelope");
    let failure: WireApiEnvelope<serde_json::Value> =
        serde_json::from_str(&producer_json).expect("failure envelope");
    match failure {
        WireApiEnvelope::Failure(failure) => {
            assert!(!failure.ok);
            assert_eq!(failure.stage, "http");
            assert_eq!(failure.errors[0].kind, "invalid_request");
        }
        WireApiEnvelope::Success(_) => panic!("expected failure envelope"),
    }
}

#[test]
fn wire_batch_envelope_decodes_nested_diagnostic_failures() {
    let producer = compiler::batch(vec![BatchRequest::Check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: "(".to_string(),
    })]);
    let producer_json = serde_json::to_string(&producer).expect("serialize batch producer");
    let batch: WireBatchResultEnvelope =
        serde_json::from_str(&producer_json).expect("batch envelope");

    match &batch.results[0] {
        WireBatchResult::Check(WireApiEnvelope::Failure(failure)) => {
            assert_eq!(failure.stage, "parse");
            assert_eq!(failure.errors[0].kind, "surf_parse_error");
        }
        other => panic!("expected a check failure, got {other:?}"),
    }
}
