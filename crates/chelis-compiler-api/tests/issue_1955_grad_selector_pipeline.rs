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
fn linked_branch_joins_require_exact_callable_origin() {
    let library_decls = chelis_surf::parser::parse_str(
        "def pkg__pair__left(x: f32, w: f32) -> f32 = mul(x, w)\n\
         def pkg__pair__right(x: f32, w: f32) -> f32 = add(x, w)\n",
    )
    .expect("library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");

    let distinct = chelis_surf::parser::parse_str(
        "def probe(flag: bool) -> unit = {\n\
           chosen = if flag then pkg__pair__left else pkg__pair__right\n\
           selected = grad(chosen, wrt=w)\n\
           drop(selected)\n\
         }\n",
    )
    .expect("distinct entry parses");
    assert!(matches!(
        prepare_surf_decls_with_context(&distinct, &library, None),
        Err(PreparationError::SurfDesugar(
            DesugarError::UnresolvedGradTarget { .. }
        ))
    ));

    let same = chelis_surf::parser::parse_str(
        "def probe(flag: bool) -> unit = {\n\
           left = pkg__pair__left\n\
           right = left\n\
           chosen = if flag then left else right\n\
           selected = grad(chosen, wrt=w)\n\
           drop(selected)\n\
         }\n",
    )
    .expect("same-origin entry parses");
    assert!(prepare_surf_decls_with_context(&same, &library, None).is_ok());
}

#[test]
fn layered_context_preserves_same_origin_structural_joins() {
    let library_decls = chelis_surf::parser::parse_str(
        "def pkg__pair__left(x: f32, w: f32) -> f32 = mul(x, w)\n\
         def pkg__pair__right(x: f32, w: f32) -> f32 = add(x, w)\n\
         left_alias = pkg__pair__left\n\
         same_if = if true then pkg__pair__left else left_alias\n\
         same_match = match true with {\n\
           | true => pkg__pair__left\n\
           | false => left_alias\n\
         }\n\
         same_tuple = (if true then (pkg__pair__left,) else (left_alias,)).0\n\
         same_block = if true then {\n\
           local = pkg__pair__left\n\
           local\n\
         } else {\n\
           local = left_alias\n\
           local\n\
         }\n\
         distinct_if = if true then pkg__pair__left else pkg__pair__right\n\
         distinct_match = match true with {\n\
           | true => pkg__pair__left\n\
           | false => pkg__pair__right\n\
         }\n\
         distinct_tuple = (if true then (pkg__pair__left,) else (pkg__pair__right,)).0\n\
         distinct_block = if true then {\n\
           local = pkg__pair__left\n\
           local\n\
         } else {\n\
           local = pkg__pair__right\n\
           local\n\
         }\n",
    )
    .expect("layered library parses");
    let library =
        chelis_surf::desugar::desugar_program(&library_decls).expect("layered library desugars");

    let same = chelis_surf::parser::parse_str(
        "selected = (\n\
           grad(same_if, wrt=w),\n\
           grad(same_match, wrt=w),\n\
           grad(same_tuple, wrt=w),\n\
           grad(same_block, wrt=w)\n\
         )\n",
    )
    .expect("same-origin entry parses");
    assert!(prepare_surf_decls_with_context(&same, &library, None).is_ok());

    for name in [
        "distinct_if",
        "distinct_match",
        "distinct_tuple",
        "distinct_block",
    ] {
        let entry = chelis_surf::parser::parse_str(&format!("selected = grad({name}, wrt=w)\n"))
            .expect("distinct-origin entry parses");
        assert!(
            matches!(
                prepare_surf_decls_with_context(&entry, &library, None),
                Err(PreparationError::SurfDesugar(
                    DesugarError::UnresolvedGradTarget { .. }
                ))
            ),
            "{name}"
        );
    }
}

#[test]
fn linked_context_preseeds_forward_function_origins_before_aliases() {
    let library_decls = chelis_surf::parser::parse_str(
        "alias = pkg__pair__pair\n\
         def pkg__pair__pair(x: f32, w: f32) -> f32 = mul(x, w)\n",
    )
    .expect("forward-alias library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");
    let entry = chelis_surf::parser::parse_str("selected = grad(alias, wrt=w)\n")
        .expect("forward-alias entry parses");
    assert!(prepare_surf_decls_with_context(&entry, &library, None).is_ok());

    let distinct_decls = chelis_surf::parser::parse_str(
        "chosen = if true then pkg__pair__left else pkg__pair__right\n\
         def pkg__pair__left(x: f32, w: f32) -> f32 = mul(x, w)\n\
         def pkg__pair__right(x: f32, w: f32) -> f32 = add(x, w)\n",
    )
    .expect("distinct forward-origin library parses");
    let distinct_library =
        chelis_surf::desugar::desugar_program(&distinct_decls).expect("library desugars");
    let distinct_entry = chelis_surf::parser::parse_str("selected = grad(chosen, wrt=w)\n")
        .expect("distinct entry parses");
    assert!(matches!(
        prepare_surf_decls_with_context(&distinct_entry, &distinct_library, None),
        Err(PreparationError::SurfDesugar(
            DesugarError::UnresolvedGradTarget { .. }
        ))
    ));
}

#[test]
fn linked_context_projects_match_bound_callable_origins() {
    let library_decls = chelis_surf::parser::parse_str(
        "def pkg__pair__left(x: f32, w: f32) -> f32 = mul(x, w)\n\
         def pkg__pair__right(x: f32, w: f32) -> f32 = add(x, w)\n\
         matched = match (pkg__pair__left,) with {\n\
           | (chosen,) => chosen\n\
         }\n\
         distinct = match (if true then (pkg__pair__left,) else (pkg__pair__right,)) with {\n\
           | (chosen,) => chosen\n\
         }\n\
         def chosen(x: f32, w: f32) -> f32 = mul(x, w)\n\
         dynamic = match candidates with {\n\
           | (chosen,) => chosen\n\
         }\n",
    )
    .expect("match-bound library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");

    let matching = chelis_surf::parser::parse_str("selected = grad(matched, wrt=w)\n")
        .expect("matching entry parses");
    assert!(prepare_surf_decls_with_context(&matching, &library, None).is_ok());

    for name in ["distinct", "dynamic"] {
        let entry = chelis_surf::parser::parse_str(&format!("selected = grad({name}, wrt=w)\n"))
            .expect("negative entry parses");
        assert!(
            matches!(
                prepare_surf_decls_with_context(&entry, &library, None),
                Err(PreparationError::SurfDesugar(
                    DesugarError::UnresolvedGradTarget { .. }
                ))
            ),
            "{name}"
        );
    }
}

#[test]
fn linked_context_projects_constructor_and_record_payload_origins() {
    let library_decls = chelis_surf::parser::parse_str(
        "type FnBox = | FnBox(f32 -> f32 -> f32)\n\
         type Outer = | Outer(FnBox)\n\
         type FnRecord = | FnRecord { callback: f32 -> f32 -> f32 }\n\
         type OuterRecord = | OuterRecord { payload: FnRecord }\n\
         def pkg__pair__pair(x: f32, w: f32) -> f32 = mul(x, w)\n\
         alias = pkg__pair__pair\n\
         boxed = FnBox(alias)\n\
         nested_boxed = Outer(boxed)\n\
         recorded = FnRecord { callback: alias }\n\
         nested_recorded = OuterRecord { payload: recorded }\n\
         ctor_match = match boxed with {\n\
           | FnBox(chosen) => chosen\n\
         }\n\
         nested_ctor_match = match nested_boxed with {\n\
           | Outer(FnBox(chosen)) => chosen\n\
         }\n\
         record_match = match recorded with {\n\
           | FnRecord { callback: chosen } => chosen\n\
         }\n\
         nested_record_match = match nested_recorded with {\n\
           | OuterRecord { payload: FnRecord { callback: chosen } } => chosen\n\
         }\n",
    )
    .expect("aggregate-payload library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");
    let entry = chelis_surf::parser::parse_str(
        "selected = (\n\
           grad(ctor_match, wrt=w),\n\
           grad(nested_ctor_match, wrt=w),\n\
           grad(record_match, wrt=w),\n\
           grad(nested_record_match, wrt=w)\n\
         )\n",
    )
    .expect("aggregate-payload entry parses");
    assert!(prepare_surf_decls_with_context(&entry, &library, None).is_ok());
}

#[test]
fn linked_context_rejects_distinct_or_dynamic_constructor_and_record_payloads() {
    let library_decls = chelis_surf::parser::parse_str(
        "def pkg__pair__left(x: f32, w: f32) -> f32 = mul(x, w)\n\
         def pkg__pair__right(x: f32, w: f32) -> f32 = add(x, w)\n\
         distinct_ctor = match \
           (if true then FnBox(pkg__pair__left) else FnBox(pkg__pair__right)) with {\n\
           | FnBox(chosen) => chosen\n\
         }\n\
         dynamic_ctor = match candidates with {\n\
           | FnBox(chosen) => chosen\n\
         }\n\
         distinct_record = match \
           (if true then FnRecord { callback: pkg__pair__left } else FnRecord { callback: pkg__pair__right }) with {\n\
           | FnRecord { callback: chosen } => chosen\n\
         }\n\
         dynamic_record = match candidates with {\n\
           | FnRecord { callback: chosen } => chosen\n\
         }\n",
    )
    .expect("negative aggregate-payload library parses");
    let library = chelis_surf::desugar::desugar_program(&library_decls).expect("library desugars");

    for name in [
        "distinct_ctor",
        "dynamic_ctor",
        "distinct_record",
        "dynamic_record",
    ] {
        let entry = chelis_surf::parser::parse_str(&format!("selected = grad({name}, wrt=w)\n"))
            .expect("negative entry parses");
        assert!(
            matches!(
                prepare_surf_decls_with_context(&entry, &library, None),
                Err(PreparationError::SurfDesugar(
                    DesugarError::UnresolvedGradTarget { .. }
                ))
            ),
            "{name}"
        );
    }
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
