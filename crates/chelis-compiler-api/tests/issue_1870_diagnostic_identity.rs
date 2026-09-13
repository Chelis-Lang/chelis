use chelis_compiler_api::{
    compiler::compile,
    schema::{CompileRequest, CompileTarget, SourceKind},
};
use chelis_types::unsupported::{RejectionAuthorityKind, Stage, UnsupportedKind};

#[test]
fn c_softmax_exposes_the_production_unsupported_identity() {
    let source = include_str!("../../../examples/annotated_concat_softmax.ch");
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect_err("C host emission must reject softmax");
    let diagnostic = error.errors.first().expect("one diagnostic");
    let identity = diagnostic
        .unsupported_identity()
        .expect("unsupported diagnostics retain their typed identity");
    assert_eq!(identity.brand, "unsupported:");
    assert_eq!(identity.kind.as_str(), "unsupported_feature");
    assert_eq!(
        identity.payload.what,
        UnsupportedKind::Builtin("softmax".into())
    );
    assert_eq!(identity.payload.context, "`chelis build` host emission");
    assert_eq!(identity.payload.stage, Stage::Codegen("c"));
    assert_eq!(
        identity.payload.disposition,
        RejectionAuthorityKind::Deliberate
    );
    assert_eq!(
        identity.payload.atom.map(|atom| atom.as_str()),
        Some("[04-TOT-2]")
    );
    assert_eq!(identity.payload.tracking_issue, None);
    assert_eq!(identity.payload.span, None);
    assert_eq!(
        identity.payload.supported_alternative.as_deref(),
        Some("run this program with `chelis eval`")
    );
}

#[test]
fn lowering_rejection_retains_stage_span_and_tracking_metadata() {
    let source = "def f(x: tensor[6, f32], w: int64, s: int64) -> tensor[5, f32] = \
                  reduce_window_max(x, [w], [s])\n\
                  out = f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0]), 2i64, 1i64)\n";
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect_err("compiled lowering must reject a runtime window list");
    let diagnostic = error.errors.first().expect("one diagnostic");
    let identity = diagnostic
        .unsupported_identity()
        .expect("lowering must preserve the typed unsupported value");
    assert_eq!(identity.kind.as_str(), "unsupported_feature");
    assert_eq!(identity.payload.stage, Stage::Lowering);
    assert_eq!(
        identity.payload.disposition,
        RejectionAuthorityKind::Unimplemented
    );
    assert_eq!(identity.payload.atom, None);
    assert_eq!(
        identity.payload.tracking_issue.map(|issue| issue.number()),
        Some(1058)
    );
    let span = identity.payload.span.expect("source-associated rejection");
    assert_eq!(span.span_id.as_deref(), Some("surf:86..89"));
    assert_eq!(
        identity.payload.supported_alternative.as_deref(),
        Some("use integer literal window and stride lists")
    );
}
