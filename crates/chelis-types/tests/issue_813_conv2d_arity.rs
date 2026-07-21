//! Chelis-Lang/chelis#813 -- a one-argument `conv2d` call emitted its
//! inference-layer arity diagnostic and then panicked in the symbolic
//! validator while unconditionally indexing the missing kernel argument.
//!
//! Spec authority: spec/04-type-system.md §10 [04-TOT-3]. Every malformed
//! built-in application must return a below-perfect fitness report with a
//! diagnostic; checker panics are forbidden.

use std::panic::{AssertUnwindSafe, catch_unwind};

use chelis_deep::Expr;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::check_ir_fitness;
use chelis_types::errors::CheckErrorKind;

fn surf_to_deep(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).expect("Surf fixture must parse");
    chelis_macros::expand_program(
        &desugar_program(&decls),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("Surf fixture must expand")
    .into_exprs()
}

fn conv2d_program(arity: usize) -> String {
    let args = ["x", "k", "1", "0", "0", "0"][..arity].join(", ");
    format!(
        "def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) \
         -> tensor[1, 8, 6, 6, f32] = conv2d({args})\n"
    )
}

#[test]
fn conv2d_arity_zero_through_six_is_total_and_exact() {
    for arity in 0..=6 {
        let deep = surf_to_deep(&conv2d_program(arity));
        let outcome = catch_unwind(AssertUnwindSafe(|| check_ir_fitness(&deep)));
        let report = outcome.unwrap_or_else(|_| panic!("conv2d arity {arity} must not panic"));

        if arity == 4 {
            assert_eq!(report.score, 1.0, "canonical arity must score perfectly");
            assert!(
                report.errors.is_empty(),
                "canonical arity must be diagnostic-free: {:?}",
                report.errors
            );
            continue;
        }

        assert!(
            report.score < 1.0,
            "conv2d arity {arity} must score below perfect"
        );
        assert_eq!(
            report.errors.len(),
            1,
            "conv2d arity {arity} must produce one owning diagnostic without spray: {:?}",
            report.errors
        );
        assert!(
            matches!(report.errors[0].kind, CheckErrorKind::ArityMismatch),
            "conv2d arity {arity} must report ArityMismatch, got {:?}",
            report.errors[0]
        );
    }
}
