//! Chelis-Lang/chelis#813 -- a one-argument `conv` call emitted its
//! inference-layer arity diagnostic and then panicked in the symbolic
//! validator while unconditionally indexing the missing kernel argument.
//!
//! Spec authority: spec/04-type-system.md §10 [04-TOT-3]. Every malformed
//! built-in application must return a below-perfect fitness report with a
//! diagnostic; checker panics are forbidden.

use std::panic::{AssertUnwindSafe, catch_unwind};

use chelis_deep::Expr;
use chelis_deep::parser::parse_str as parse_deep_lenient;

use chelis_types::errors::CheckErrorKind;
use chelis_types::{BUILTIN_NAMES, check_ir_fitness};

fn core_program(source: &str) -> Vec<Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("Deep fixture")
}

fn conv_program(arity: usize) -> String {
    let args = [
        r#"(var {span: "surf:96..97"} x)"#,
        r#"(var {span: "surf:99..100"} k)"#,
        r#"(app {span: "surf:102..113"}
        (var {} Cons)
        (lit {span: "surf:103..107", type: (t-prim {} i64)} 1)
        (app {}
          (var {} Cons)
          (lit {span: "surf:108..112", type: (t-prim {} i64)} 1)
          (var {} Nil)))"#,
        r#"(app {span: "surf:115..140"}
        (var {} Cons)
        (tuple {span: "surf:116..127"}
          (lit {span: "surf:117..121", type: (t-prim {} i64)} 0)
          (lit {span: "surf:122..126", type: (t-prim {} i64)} 0))
        (app {}
          (var {} Cons)
          (tuple {span: "surf:128..139"}
            (lit {span: "surf:129..133", type: (t-prim {} i64)} 0)
            (lit {span: "surf:134..138", type: (t-prim {} i64)} 0))
          (var {} Nil)))"#,
        r#"(lit {span: "surf:142..143", type: (t-prim {} i32)} 0)"#,
        r#"(lit {span: "surf:145..146", type: (t-prim {} i32)} 0)"#,
    ][..arity]
        .join(" ");
    include_str!("fixtures/core/tests/issue_813_conv2d_arity/conv_arity_template.deep")
        .replace("@args@", &args)
}

#[test]
fn conv_arity_zero_through_six_is_total_and_exact() {
    for arity in 0..=6 {
        let deep = core_program(&conv_program(arity));
        let outcome = catch_unwind(AssertUnwindSafe(|| check_ir_fitness(&deep)));
        let report = outcome.unwrap_or_else(|_| panic!("conv arity {arity} must not panic"));

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
            "conv arity {arity} must score below perfect"
        );
        assert_eq!(
            report.errors.len(),
            1,
            "conv arity {arity} must produce one owning diagnostic without spray: {:?}",
            report.errors
        );
        assert!(
            matches!(report.errors[0].kind, CheckErrorKind::ArityMismatch),
            "conv arity {arity} must report ArityMismatch, got {:?}",
            report.errors[0]
        );
    }
}

#[test]
fn every_builtin_rejects_zero_and_oversized_arity_without_panicking() {
    for name in BUILTIN_NAMES {
        for arity in [0, 12] {
            let args = std::iter::repeat_n("(lit {type: (t-prim {} f32)} 1.0)", arity)
                .collect::<Vec<_>>()
                .join(" ");
            let source = format!("(def {{}} bad (app {{}} (var {{}} {name}) {args}))");
            let deep = parse_deep_lenient(&source).expect("generated Deep fixture parses");
            let outcome = catch_unwind(AssertUnwindSafe(|| check_ir_fitness(&deep)));
            let report = outcome
                .unwrap_or_else(|_| panic!("builtin `{name}` with arity {arity} must not panic"));
            assert!(
                report.score < 1.0 && !report.errors.is_empty(),
                "builtin `{name}` with arity {arity} must reject loudly: {report:?}"
            );
        }
    }
}
