//! chelis#2140: direct `index` applications require the exact i64 index type.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut messages = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    messages.sort();
    messages
}

fn whole_program(source: &str) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let parsed = parse_str(source).expect("valid Surf");
    let desugared = desugar_program(&parsed).expect("Surf fixture must desugar");
    let expanded: Vec<Expr> = expand_program(&desugared, &ExpansionOptions::default())
        .expect("macro expansion")
        .into_exprs();
    [
        (
            "typed",
            check_typed_program(&desugared)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
        (
            "IR",
            check_ir_program(&expanded)
                .map(|_| ())
                .map_err(|report| report.errors),
        ),
    ]
}

fn assert_accepted(label: &str, source: &str) {
    for (entry, result) in whole_program(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{entry}] over-rejected:\n{}\nsource:\n{source}",
                rendered(&errors).join("\n")
            );
        }
    }
}

fn assert_index_type_rejected(label: &str, source: &str, index_type: &str) {
    let expected = format!("index expects i64 index, got {index_type}");
    for (entry, result) in whole_program(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{entry}] unexpectedly accepted:\n{source}"),
        };
        assert!(
            errors.iter().any(|error| {
                matches!(error.kind, CheckErrorKind::TypeMismatch)
                    && error.message.contains(&expected)
            }),
            "{label} [{entry}] expected operation-owned diagnostic {expected:?}:\n{}",
            rendered(&errors).join("\n")
        );
    }
}

#[test]
fn direct_index_accepts_exact_i64_at_both_checker_ingresses() {
    assert_accepted("exact i64 index", "out: i64 = index([1i64], 0i64)\n");
}

#[test]
fn direct_index_rejects_every_other_integer_width_at_both_checker_ingresses() {
    // Mutation-equivalent negative control: restoring Prim::is_integer() in
    // the direct route makes every one of these witnesses incorrectly pass.
    for (index_type, literal) in [("i8", "0i8"), ("i16", "0i16"), ("i32", "0i32")] {
        let source = format!("out: i64 = index([1i64], {literal})\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}

#[test]
fn direct_index_rejects_active_non_integer_primitives_at_both_checker_ingresses() {
    // Reserved f8e4m3 is rejected earlier by its UnsupportedTensorPrecision
    // owner and therefore must not be forced through this operation-owned guard.
    for index_type in ["f16", "bf16", "f32", "f64", "bool", "string"] {
        let source = format!("def pick(xs: List[i64], i: {index_type}) -> i64 = index(xs, i)\n");
        assert_index_type_rejected(index_type, &source, index_type);
    }
}
