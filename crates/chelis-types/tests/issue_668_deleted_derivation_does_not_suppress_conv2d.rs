//! chelis#668 / chelis#1537: a let-bound movement result must not hide
//! [05-OP-51]'s static convolution range checks. PP9 also removes the former
//! backend-only symbolic-metadata rejection and applies the surviving semantic
//! checks at both public checker ingresses.

use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("Deep fixture")
}

fn messages(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect();
    out.sort();
    out
}

fn ir_diagnostics(source: &str) -> Vec<String> {
    match check_ir_program(
        &deep(source),
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn typed_diagnostics(source: &str) -> Vec<String> {
    match check_typed_program(
        &deep(source),
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn assert_conv_still_validates(source: &str, needle: &str, label: &str) {
    let diagnostics = ir_diagnostics(source);
    assert!(
        diagnostics.iter().any(|m| m.contains(needle)),
        "{label}: a let-bound movement operand must not switch the `conv` \
         validator off; expected a diagnostic containing {needle:?}, got \
         {diagnostics:?}"
    );
}

const CONV2D_STRIDE_ZERO_DIRECT: &str = include_str!(
    "fixtures/core/tests/issue_668_deleted_derivation_does_not_suppress_conv2d/CONV2D_STRIDE_ZERO_DIRECT_1.deep"
);

/// `stride = 0` is undefined: the output spatial-dim formula divides by it.
/// Routing the input through each of the three operations whose derivation
/// chelis#668 deleted must not stop the guard firing.
///
/// REGRESSION. At `948ed5736` all three scored 1 with no errors, and the first
/// then panicked in codegen instead of failing to compile.
///
#[test]
fn an_invalid_stride_is_refused_through_each_deleted_derivation() {
    for (binding, input, label) in [
        (
            r#"(app {span: "surf:11..32"}
      (var {span: "surf:11..17"} expand)
      (var {span: "surf:18..19"} x)
      (lit {span: "surf:21..25", surf_literal_style: "explicit", type: (t-prim {} i32)} 0)
      (lit {span: "surf:27..31", type: (t-prim {} i64)} 2))"#,
            "(t-tensor {} (d-lit {} 1) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))",
            "let-bound expand",
        ),
        (
            r#"(app {span: "surf:11..32"}
      (var {span: "surf:11..17"} insert)
      (var {span: "surf:18..19"} x)
      (lit {span: "surf:21..25", surf_literal_style: "explicit", type: (t-prim {} i32)} 0)
      (lit {span: "surf:27..31", type: (t-prim {} i64)} 2))"#,
            "(t-tensor {} (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))",
            "let-bound insert",
        ),
        (
            r#"(app {span: "surf:11..44"}
      (var {span: "surf:11..17"} stride)
      (var {span: "surf:18..19"} x)
      (lit {span: "surf:21..25", type: (t-prim {} i64)} 1)
      (lit {span: "surf:27..31", type: (t-prim {} i64)} 1)
      (lit {span: "surf:33..37", type: (t-prim {} i64)} 1)
      (lit {span: "surf:39..43", type: (t-prim {} i64)} 1))"#,
            "(t-tensor {} (d-lit {} 2) (d-lit {} 3) (d-lit {} 8) (d-lit {} 8) (t-prim {} f32))",
            "let-bound stride",
        ),
    ] {
        let source = include_str!("fixtures/core/tests/issue_668_deleted_derivation_does_not_suppress_conv2d/movement_template.deep").replace("@input@", input).replace("@binding@", binding);
        assert_conv_still_validates(&source, "requires a positive stride", label);
    }
}

/// The control for the row above: the same invalid stride with no intervening
/// binding reaches the stride check itself and names it.
///
/// DISPOSITION LOCK, green in all three states. It is what proves the rows
/// above measure the SUPPRESSION rather than conv validation in general.
#[test]
fn an_invalid_stride_without_a_let_binding_names_the_stride() {
    assert_conv_still_validates(
        CONV2D_STRIDE_ZERO_DIRECT,
        "requires a positive stride, got 0",
        "conv with stride 0 and no let-binding",
    );
}

/// PP9 relocates symbolic-metadata capability refusal to chelis#730.
#[test]
fn a_symbolic_spatial_dimension_through_a_let_bound_expand_is_checker_legal() {
    let diagnostics = ir_diagnostics(include_str!(
        "fixtures/core/tests/issue_668_deleted_derivation_does_not_suppress_conv2d/a_symbolic_spatial_dimension_through_a_let_bound_expand_is_checker_legal_1.deep"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// A well-typed `conv` behind a let-bound `expand`.
#[test]
fn a_well_typed_conv_through_a_let_bound_expand_is_accepted() {
    let diagnostics = ir_diagnostics(include_str!(
        "fixtures/core/tests/issue_668_deleted_derivation_does_not_suppress_conv2d/a_well_typed_conv_through_a_let_bound_expand_is_accepted_1.deep"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// Symbolic conv metadata remains legal through a chain.
#[test]
fn a_symbolic_conv_chain_is_checker_legal() {
    let chained = ir_diagnostics(include_str!(
        "fixtures/core/tests/issue_668_deleted_derivation_does_not_suppress_conv2d/a_symbolic_conv_chain_is_checker_legal_1.deep"
    ));
    assert!(chained.is_empty(), "{chained:?}");
}

/// PP9 applies the surviving validator rule at both checker ingresses.
#[test]
fn the_shared_semantic_validator_runs_on_both_checker_ingresses() {
    assert!(
        ir_diagnostics(CONV2D_STRIDE_ZERO_DIRECT)
            .iter()
            .any(|m| m.contains("requires a positive stride, got 0")),
        "the IR ingress runs the validator"
    );
    assert!(
        typed_diagnostics(CONV2D_STRIDE_ZERO_DIRECT)
            .iter()
            .any(|m| m.contains("requires a positive stride, got 0")),
        "the typed ingress runs the same semantic validator"
    );
}
