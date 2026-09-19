//! chelis#1955 / chelis#1473: selector-resolution failures remain typed at
//! preparation and become stable `desugar`-stage compiler diagnostics.

use chelis_compiler_api::{
    compiler,
    pipeline::{PreparationError, prepare_source, prepare_surf_decls_with_context},
    schema::{CheckRequest, DesugarRequest, SourceKind},
};
use chelis_surf::desugar::DesugarError;

const BAD_SELECTOR: &str =
    "def pair(x: f32, w: f32) -> f32 = mul(x, w)\nout = grad(pair, wrt=typo)\n";

#[test]
fn preparation_retains_the_typed_desugar_error() {
    let error = prepare_source(SourceKind::Surf, BAD_SELECTOR, None)
        .expect_err("unknown selector must reject during preparation");
    assert!(matches!(
        error,
        PreparationError::SurfDesugar(DesugarError::UnknownGradParameter {
            parameter,
            ..
        }) if parameter == "typo"
    ));
}

#[test]
fn linked_callable_context_resolves_qualified_aliases_before_layered_checking() {
    let library_decls =
        chelis_surf::parser::parse_str("def pkg__pair__pair(x: f32, w: f32) -> f32 = mul(x, w)\n")
            .expect("library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");
    let entry =
        chelis_surf::parser::parse_str("alias = pkg__pair__pair\nselected = grad(alias, wrt=w)\n")
            .expect("entry parses");

    assert!(prepare_surf_decls_with_context(&entry, &library, None).is_ok());
    assert!(matches!(
        chelis_compiler_api::pipeline::prepare_surf_decls(&entry, None),
        Err(PreparationError::SurfDesugar(
            DesugarError::UnresolvedGradTarget { .. }
        ))
    ));
}

#[test]
fn contradictory_deep_selector_is_rejected_without_rewriting() {
    let source = "(def {} pair (fn {} (params {} x w) (app {} (var {} mul) (var {} x) (var {} w))))\n\
                  (def {} selected (grad {wrt: (var {} w)} (var {} pair) \
                    (lit {type: (t-prim {} i32)} 0)))";
    let error = prepare_source(SourceKind::Deep, source, None)
        .expect_err("contradictory Deep selector must reject");
    assert!(matches!(error, PreparationError::DeepSelector(_)));
}

#[test]
fn compiler_api_desugar_and_check_report_the_same_stage_and_kind() {
    let desugar = compiler::desugar(DesugarRequest {
        source: BAD_SELECTOR.to_string(),
    })
    .expect_err("desugar must reject");
    let check = compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: BAD_SELECTOR.to_string(),
    })
    .expect_err("check must reject");

    for error in [desugar, check] {
        assert_eq!(error.stage, "desugar", "{error:?}");
        assert_eq!(error.errors.len(), 1, "{error:?}");
        assert_eq!(
            error.errors[0].kind(),
            chelis_vocab::DiagnosticKind::TypeMismatch,
            "{error:?}"
        );
        assert!(
            error.errors[0]
                .message
                .contains("unknown `grad` parameter `typo`"),
            "{error:?}"
        );
    }
}
