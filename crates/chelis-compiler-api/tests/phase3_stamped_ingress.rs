//! Compiler API ingress regressions for chelis#731 Phase 3.
//!
//! Every public Deep text boundary must consume the role-stamped carrier.
//! A bare name in a RuntimeExpr slot is therefore an ingress error, not a
//! legacy tree that reaches a downstream checker or authoring helper.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{
    CheckRequest, DeepCallGraphRequest, DeepOutlineRequest, DeepReferencesRequest, ParseRequest,
    SourceKind,
};

const VALID_MODULE: &str =
    "(module {} phase3.ingress (def {} f (lit {type: (t-prim {} int64)} 1)))";
const BARE_NAME_BODY_MODULE: &str = "(module {} phase3.ingress (def {} f unwrapped_name))";

fn assert_stamp_error(error: compiler::CompilerError, stage: &str) {
    assert_eq!(error.stage, stage);
    assert_eq!(error.errors.len(), 1);
    assert_eq!(error.errors[0].kind, "deep_stamp_error");
    assert!(error.errors[0].message.contains("bare name"));
    assert!(error.errors[0].span.is_some());
}

#[test]
fn generic_parse_and_check_use_stamped_deep_ingress() {
    let parse_error = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("Deep parse must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(parse_error, "parse");

    let check_error = compiler::check(CheckRequest {
        source_kind: SourceKind::Deep,
        source: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("Deep check must share the stamped ingress boundary");
    assert_stamp_error(check_error, "parse");

    compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: VALID_MODULE.to_string(),
    })
    .expect("well-formed Deep still parses");
}

#[test]
fn generic_parse_wire_preserves_unknown_form_head_and_metadata() {
    let parsed = compiler::parse(ParseRequest {
        source_kind: SourceKind::Deep,
        source: r#"(def {} f
  (future-form {sentinel_meta: "keep-me"}
    (lit {} 1)))"#
            .to_string(),
    })
    .expect("unknown forms remain available for downstream diagnostics");

    let wire = serde_json::to_string(&parsed.deep_ast.expect("Deep AST must be present"))
        .expect("wire AST must serialize");
    assert!(
        wire.contains("future-form"),
        "unknown head was lost: {wire}"
    );
    assert!(
        wire.contains("sentinel_meta"),
        "metadata key was lost: {wire}"
    );
    assert!(wire.contains("keep-me"), "metadata value was lost: {wire}");
}

#[test]
fn read_only_authoring_ingresses_use_the_stamp_gate() {
    let outline_error = compiler::deep_outline(DeepOutlineRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("outline must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(outline_error, "deep-outline");

    let references_error = compiler::deep_references(DeepReferencesRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
        symbol: "f".to_string(),
    })
    .expect_err("references must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(references_error, "deep-references");

    let graph_error = compiler::deep_call_graph(DeepCallGraphRequest {
        module: BARE_NAME_BODY_MODULE.to_string(),
    })
    .expect_err("call graph must reject a bare RuntimeExpr name at stamping");
    assert_stamp_error(graph_error, "deep-call-graph");

    compiler::deep_outline(DeepOutlineRequest {
        module: VALID_MODULE.to_string(),
    })
    .expect("well-formed stamped module still outlines");
    compiler::deep_references(DeepReferencesRequest {
        module: VALID_MODULE.to_string(),
        symbol: "f".to_string(),
    })
    .expect("well-formed stamped module still supports references");
    compiler::deep_call_graph(DeepCallGraphRequest {
        module: VALID_MODULE.to_string(),
    })
    .expect("well-formed stamped module still supports call graph");
}
