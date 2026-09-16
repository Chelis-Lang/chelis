//! chelis#2126: `to_list(tensor[n, p])` establishes `List[p]` even while
//! the authored tensor precision remains polymorphic.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
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

fn both_ingresses(source: &str) -> [(&'static str, Result<(), Vec<CheckError>>); 2] {
    let parsed = parse_str(source).expect("valid Surf");
    let desugared = desugar_program(&parsed);
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

fn assert_accepts(label: &str, source: &str) {
    for (ingress, result) in both_ingresses(source) {
        if let Err(errors) = result {
            panic!(
                "{label} [{ingress}] unexpectedly rejected:\n{}\nsource:\n{source}",
                rendered(&errors)
            );
        }
    }
}

fn assert_rejects(label: &str, source: &str, expected: &str) {
    for (ingress, result) in both_ingresses(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{ingress}] unexpectedly accepted:\n{source}"),
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
            "int16 leaf",
            "def first[n, p](values: tensor[n, p]) -> p = \
             index(to_list(values), 0i64)\n\
             out: int16 = first(to_tensor([cast(321, int16), cast(7, int16)]))\n",
        ),
        (
            "bool leaf",
            "def first[n, p](values: tensor[n, p]) -> p = \
             index(to_list(values), 0i64)\n\
             out: bool = first(to_tensor([true, false]))\n",
        ),
    ] {
        assert_accepts(label, source);
    }
}

#[test]
fn explicit_list_annotation_remains_a_positive_control() {
    assert_accepts(
        "explicit List[p]",
        "def first[n, p](values: tensor[n, p]) -> p = {\n\
         xs: List[p] = to_list(values)\n\
         index(xs, 0i64)\n\
         }\n\
         out: int16 = first(to_tensor([cast(321, int16), cast(7, int16)]))\n",
    );
}

#[test]
fn symbolic_to_list_cannot_narrow_a_rigid_dtype_binder_to_string() {
    assert_rejects(
        "string result",
        "def bad[n, p](values: tensor[n, p]) -> List[string] = to_list(values)\n",
        "[04-INF-6]",
    );
}

#[test]
fn a_genuinely_unknown_collection_constructor_still_rejects() {
    assert_rejects(
        "unknown index operand",
        "def at(xs, i: int64) = index(xs, i)\n",
        "unresolved `index` shape obligation",
    );
}
