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
    let source = "def f(x: tensor[6, f32], w: i64, s: i64) -> tensor[5, f32] = \
                  reduce_window_max(x, [w], [s])\n\
                  out = f(to_tensor([1.0, 5.0, 2.0, 8.0, 3.0, 9.0], f32), 2i64, 1i64)\n";
    for target in [CompileTarget::C, CompileTarget::Hip] {
        let error = compile(CompileRequest {
            source_kind: SourceKind::Surf,
            source: source.into(),
            target,
            entry_name: None,
        })
        .expect_err("compiled lowering must reject a runtime window list");
        assert_eq!(error.stage, "lower");
        let diagnostic = error.errors.first().expect("one diagnostic");
        assert_eq!(
            diagnostic.message,
            "unsupported: a non-literal window list for `reduce_window_max` on the \
             compiled-backend lowering of `reduce_window_*` (lowering); unimplemented \
             chelis#1058: window and stride lists must be integer literals for the compiled \
             lane today; a runtime-parameterized window previously lowered to a silent no-op; \
             chelis#1058 owns compiled runtime-list support at source span `surf:82..85`"
        );
        let identity = diagnostic
            .unsupported_identity()
            .expect("lowering must preserve the typed unsupported value");
        assert_eq!(identity.kind.as_str(), "unsupported_feature");
        let wire = serde_json::to_value(diagnostic).expect("diagnostic wire value");
        assert_eq!(wire["kind"], "unsupported_feature");
        assert_eq!(wire["span_id"], "surf:82..85");
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
        assert_eq!(span.span_id.as_deref(), Some("surf:82..85"));
        assert_eq!(
            identity.payload.supported_alternative.as_deref(),
            Some("use integer literal window and stride lists")
        );
    }
}

#[test]
fn par_exposes_its_rendered_supported_alternative_as_typed_data() {
    let source = "def g() -> f32 = par { 1.0f32; 2.0f32 }\nout = g()\n";
    let error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect_err("every target refuses `par`");
    let diagnostic = error.errors.first().expect("one diagnostic");
    let identity = diagnostic
        .unsupported_identity()
        .expect("the production `par` rejection retains typed identity");
    assert_eq!(
        identity.payload.what,
        UnsupportedKind::Construct("`par` expression".into())
    );
    assert_eq!(
        identity.payload.supported_alternative.as_deref(),
        Some("use `do { ... }` when sequential evaluation is intended")
    );
}
