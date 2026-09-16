//! chelis#668 / chelis#1537: a let-bound movement result must not hide
//! [05-OP-51]'s static convolution range checks. PP9 also removes the former
//! backend-only symbolic-metadata rejection and applies the surviving semantic
//! checks at both public checker ingresses.

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn deep(source: &str) -> Vec<chelis_deep::Expr> {
    desugar_program(&parse_surf(source).unwrap_or_else(|error| {
        panic!("fixture must parse as Surf: {error:?}\n{source}");
    }))
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
    match check_ir_program(&deep(source)) {
        Ok(_) => Vec::new(),
        Err(result) => messages(&result.errors),
    }
}

fn typed_diagnostics(source: &str) -> Vec<String> {
    match check_typed_program(&deep(source)) {
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

const CONV2D_STRIDE_ZERO_DIRECT: &str = "module Repro.Conv2dStrideZeroDirect\n\
     def f(x: tensor[2, 3, 8, 8, f32], k: tensor[4, 3, 3, 3, f32]) = \
     conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])\n";

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
            "y = expand(x, 0i32, 2i64)",
            "tensor[1, 3, 8, 8, f32]",
            "let-bound expand",
        ),
        (
            "y = insert(x, 0i32, 2i64)",
            "tensor[3, 8, 8, f32]",
            "let-bound insert",
        ),
        (
            "y = stride(x, 1i64, 1i64, 1i64, 1i64)",
            "tensor[2, 3, 8, 8, f32]",
            "let-bound stride",
        ),
    ] {
        let source = format!(
            "module Repro.Conv2dStrideZero\n\
             def f(x: {input}, k: tensor[4, 3, 3, 3, f32]) = {{\n\
               {binding}\n\
               conv(&y, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])\n\
             }}\n"
        );
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
    let diagnostics = ir_diagnostics(
        "module Repro.Conv2dSymbolicSpatial\n\
         def f(x: tensor[1, 3, h, 8, f32], k: tensor[4, 3, 3, 3, f32]) = {\n\
           y = expand(x, 0i32, 2i64)\n\
           conv(&y, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
    );
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// A well-typed `conv` behind a let-bound `expand`.
#[test]
fn a_well_typed_conv_through_a_let_bound_expand_is_accepted() {
    let diagnostics = ir_diagnostics(
        "module Repro.Conv2dWellTyped\n\
         def f(\n\
           x: tensor[1, 3, 8, 8, f32],\n\
           k: tensor[4, 3, 3, 3, f32],\n\
         ) -> tensor[2, 4, 6, 6, f32] = {\n\
           y = expand(x, 0i32, 2i64)\n\
           conv(&y, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
    );
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

/// Symbolic conv metadata remains legal through a chain.
#[test]
fn a_symbolic_conv_chain_is_checker_legal() {
    let chained = ir_diagnostics(
        "module Repro.Conv2dCascade\n\
         def f(\n\
           x: tensor[2, 3, h, 8, f32],\n\
           k1: tensor[4, 3, 3, 3, f32],\n\
           k2: tensor[4, 4, 3, 3, f32],\n\
         ) = {\n\
           y = conv(&x, &k1, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
           conv(&y, &k2, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n\
         }\n",
    );
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
