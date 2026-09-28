//! [05-OP-57]: `to_tensor` of a `List` whose leaf is a type binder bounded by
//! `Float`, `Int` or `Numeric` has a known result. The bound makes the leaf a
//! scalar tensor-element dtype, so the nesting depth is the rank and the
//! binder is the result precision, as `to_list` maps `tensor[n, p]` back to
//! `List[p]` (chelis#2126).
//!
//! The checker used to publish the builtin scheme's opaque result for such a
//! leaf. A consumer of that result, a `reshape` for example, then stayed
//! suspended past its declaration boundary, and a declared result of the
//! wrong rank was accepted. chelis-std's two-layer random initialisers
//! (#2413) reach this shape: their pure layers rebuild a tensor with
//! `to_tensor` over a `List[p]`.

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

fn assert_rejects(label: &str, source: &str, kind: fn(&CheckErrorKind) -> bool, expected: &str) {
    for (ingress, result) in both_ingresses(source) {
        let errors = match result {
            Err(errors) => errors,
            Ok(()) => panic!("{label} [{ingress}] unexpectedly accepted:\n{source}"),
        };
        assert!(
            errors
                .iter()
                .any(|error| kind(&error.kind) && error.message.contains(expected)),
            "{label} [{ingress}] expected a diagnostic of the named kind containing {expected:?}:\n{}",
            rendered(&errors)
        );
    }
}

#[test]
fn a_reshape_of_to_tensor_over_a_bounded_binder_leaf_is_decided() {
    for (label, source) in [
        (
            "Float leaf",
            "def flatten_back[p: Float](xs: List[p]) -> List[p] = \
             to_list(reshape(to_tensor(xs), [cast(len(xs), i64)]))\n\
             narrow = flatten_back([1.0f32, 2.0f32])\n\
             wide = flatten_back([1.0f64, 2.0f64])\n",
        ),
        (
            "Int leaf",
            "def flatten_back[p: Int](xs: List[p]) -> List[p] = \
             to_list(reshape(to_tensor(xs), [cast(len(xs), i64)]))\n\
             out = flatten_back([1i64, 2i64])\n",
        ),
        (
            "Numeric leaf",
            "def flatten_back[p: Numeric](xs: List[p]) -> List[p] = \
             to_list(reshape(to_tensor(xs), [cast(len(xs), i64)]))\n\
             out = flatten_back([1i32, 2i32])\n",
        ),
    ] {
        assert_accepts(label, source);
    }
}

#[test]
fn the_nesting_depth_of_a_bounded_leaf_is_the_rank() {
    assert_accepts(
        "rank two",
        "def lift[p: Numeric](xs: List[List[p]]) -> tensor[2, 2, p] = to_tensor(xs)\n\
         out = lift([[1.0f32, 2.0f32], [3.0f32, 4.0f32]])\n",
    );
    assert_rejects(
        "rank one declared rank two",
        "def lift[p: Float](xs: List[p]) -> tensor[2, 2, p] = to_tensor(xs)\n",
        |kind| matches!(kind, CheckErrorKind::DimensionMismatch),
        "doesn't match declared signature",
    );
    assert_rejects(
        "rank two declared rank one",
        "def lift[p: Numeric](xs: List[List[p]]) -> tensor[4, p] = to_tensor(xs)\n",
        |kind| matches!(kind, CheckErrorKind::DimensionMismatch),
        "doesn't match declared signature",
    );
}

#[test]
fn an_unbounded_binder_leaf_stays_an_explicit_obligation() {
    // Without a dtype-family bound the leaf may itself be a List, so the
    // rank is not known and the suspended reshape still rejects.
    assert_rejects(
        "unbounded leaf",
        "def flatten_back[a](xs: List[a]) -> List[a] = \
         to_list(reshape(to_tensor(xs), [cast(len(xs), i64)]))\n",
        |kind| matches!(kind, CheckErrorKind::TypeMismatch),
        "unresolved `reshape` shape obligation",
    );
}
