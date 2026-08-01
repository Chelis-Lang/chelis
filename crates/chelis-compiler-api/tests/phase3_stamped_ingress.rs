//! Compiler API ingress regressions for chelis#731 Phase 3.
//!
//! Every public Deep text boundary must consume the role-stamped carrier.
//! A bare name in a RuntimeExpr slot is therefore an ingress error, not a
//! legacy tree that reaches a downstream checker or authoring helper.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{CheckRequest, ParseRequest, SourceKind};

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
