//! chelis#2388: `par` is rejected before any execution lane can erase an
//! effectful non-final child.

use chelis_compiler_api::{
    compiler::{check, compile, eval},
    schema::{CheckRequest, CompileRequest, CompileTarget, EvalRequest, SourceKind},
};
use chelis_types::unsupported::{RejectionAuthorityKind, Stage, UnsupportedKind};
use chelis_vocab::DiagnosticKind;

const SURF_PAR: &str = "def loss(x: f32) -> f32 = par { fail(\"par boom\"); add(x, 1.0f32) }\n\
                         out = loss(3.0f32)\n";

const DEEP_PAR: &str = "(def {} out\n\
  (par {}\n\
    (lit {type: (t-prim {} f32)} 1.0)\n\
    (lit {type: (t-prim {} f32)} 2.0)))\n";

const SEQUENTIAL_CONTROL: &str = "def loss(x: f32) -> f32 = do { add(x, 1.0f32); add(x, 2.0f32) }\n\
     out = loss(3.0f32)\n";

fn assert_par_fence(diagnostic: &chelis_compiler_api::schema::Diagnostic) {
    assert_eq!(diagnostic.kind(), DiagnosticKind::UnsupportedFeature);
    assert!(
        diagnostic.message.contains("not fully implemented"),
        "{}",
        diagnostic.message
    );
    assert!(
        diagnostic.message.contains("unimplemented chelis#2388"),
        "{}",
        diagnostic.message
    );

    let identity = diagnostic
        .unsupported_identity()
        .expect("the checker fence must retain its structured unsupported identity");
    assert_eq!(
        identity.payload.what,
        UnsupportedKind::Construct("`par` expression".into())
    );
    assert_eq!(identity.payload.stage, Stage::Checker);
    assert_eq!(
        identity.payload.disposition,
        RejectionAuthorityKind::Unimplemented
    );
    assert_eq!(identity.payload.atom, None);
    assert_eq!(
        identity.payload.tracking_issue.map(|issue| issue.number()),
        Some(2388)
    );
    assert_eq!(
        identity.payload.supported_alternative.as_deref(),
        Some("use `do { ... }` when sequential evaluation is intended")
    );
    assert!(
        identity.payload.span.is_some(),
        "the par source span is known"
    );
}

#[test]
fn surf_and_deep_par_are_rejected_by_check() {
    for (source_kind, source) in [(SourceKind::Surf, SURF_PAR), (SourceKind::Deep, DEEP_PAR)] {
        let result = check(CheckRequest {
            source_kind,
            source: source.into(),
        })
        .expect("check reports typed diagnostics in its result");
        let [diagnostic] = result.errors.as_slice() else {
            panic!("expected one par fence diagnostic, got {:?}", result.errors);
        };
        assert_par_fence(diagnostic);
        assert!(result.score.get() < 1.0);
    }
}

#[test]
fn eval_and_c_build_stop_at_the_same_par_fence() {
    let eval_error = eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: SURF_PAR.into(),
        bindings: Default::default(),
    })
    .expect_err("eval must not reach the incomplete par implementation");
    assert_eq!(eval_error.stage, "check");
    let [eval_diagnostic] = eval_error.errors.as_slice() else {
        panic!("expected one eval fence diagnostic: {eval_error:?}");
    };
    assert_par_fence(eval_diagnostic);

    let build_error = compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: SURF_PAR.into(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect_err("C build must not reach the incomplete par implementation");
    assert_eq!(build_error.stage, "check");
    let [build_diagnostic] = build_error.errors.as_slice() else {
        panic!("expected one build fence diagnostic: {build_error:?}");
    };
    assert_par_fence(build_diagnostic);
}

#[test]
fn ordinary_sequential_do_remains_supported() {
    let checked = check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: SEQUENTIAL_CONTROL.into(),
    })
    .expect("check sequential control");
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    assert_eq!(checked.score.get(), 1.0);

    eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: SEQUENTIAL_CONTROL.into(),
        bindings: Default::default(),
    })
    .expect("eval sequential control");

    compile(CompileRequest {
        source_kind: SourceKind::Surf,
        source: SEQUENTIAL_CONTROL.into(),
        target: CompileTarget::C,
        entry_name: None,
    })
    .expect("build sequential control");
}
