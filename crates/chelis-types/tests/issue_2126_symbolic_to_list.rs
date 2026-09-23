//! chelis#2126: `to_list(tensor[n, p])` establishes `List[p]` even while
//! the authored tensor precision remains polymorphic.

use chelis_deep::Expr;

use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn rendered(errors: &[CheckError]) -> String {
    let mut messages = errors
        .iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect::<Vec<_>>();
    messages.sort();
    messages.join("\n")
}

fn both_ingresses(source: (&str, &str)) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let desugared: Vec<Expr> = serde_json::from_str(source.0).expect("legacy fixture");
    let expanded: Vec<Expr> = serde_json::from_str(source.1).expect("expanded fixture");
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

fn assert_accepts(label: &str, source: (&str, &str)) {
    for (ingress, result) in both_ingresses(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{ingress}] unexpectedly rejected:\n{}\nsource:\n{source:?}",
                rendered(&errors)
            );
        }
    }
}

fn assert_rejects(label: &str, source: (&str, &str), expected: &str) {
    for (ingress, result) in both_ingresses(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{ingress}] unexpectedly accepted:\n{source:?}"),
        };
        assert!(
            errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{label} [{ingress}] expected TypeMismatch:\n{}",
            rendered(&errors)
        );
        assert!(
            errors.iter().any(|error| error.message.contains(expected)),
            "{label} [{ingress}] did not contain {expected:?}:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn symbolic_to_list_preserves_the_list_constructor_and_leaf_type() {
    for (label, source) in [
        (
            "i16 leaf",
            (
                include_str!(
                    "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_preserves_the_list_constructor_and_leaf_type_0_legacy.json"
                ),
                include_str!(
                    "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_preserves_the_list_constructor_and_leaf_type_0_expanded.json"
                ),
            ),
        ),
        (
            "bool leaf",
            (
                include_str!(
                    "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_preserves_the_list_constructor_and_leaf_type_1_legacy.json"
                ),
                include_str!(
                    "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_preserves_the_list_constructor_and_leaf_type_1_expanded.json"
                ),
            ),
        ),
    ] {
        assert_accepts(label, source);
    }
}

#[test]
fn explicit_list_annotation_remains_a_positive_control() {
    assert_accepts(
        "explicit List[p]",
        (
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/explicit_list_annotation_remains_a_positive_control_2_legacy.json"
            ),
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/explicit_list_annotation_remains_a_positive_control_2_expanded.json"
            ),
        ),
    );
}

#[test]
fn symbolic_to_list_cannot_narrow_a_rigid_dtype_binder_to_string() {
    assert_rejects(
        "string result",
        (
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_cannot_narrow_a_rigid_dtype_binder_to_string_3_legacy.json"
            ),
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/symbolic_to_list_cannot_narrow_a_rigid_dtype_binder_to_string_3_expanded.json"
            ),
        ),
        "[04-INF-6]",
    );
}

#[test]
fn a_genuinely_unknown_collection_constructor_still_rejects() {
    assert_rejects(
        "unknown index operand",
        (
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/a_genuinely_unknown_collection_constructor_still_rejects_4_legacy.json"
            ),
            include_str!(
                "fixtures/core/tests/issue_2126_symbolic_to_list/a_genuinely_unknown_collection_constructor_still_rejects_4_expanded.json"
            ),
        ),
        "unresolved `index` shape obligation",
    );
}
