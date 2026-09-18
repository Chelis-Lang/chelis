//! chelis#1618: multi-index `grad` reads the tuple through the total carrier.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

const PAIR2: &str = "
    (defsig {} pair2
      (t-fn {} (t-prim {} f32) (t-prim {} f32) (t-prim {} f32)))
    (def {} pair2
      (fn {} (params {} x y) (var {} x)))";

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("fixture must stamp")
}

fn diagnostics(
    result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>,
) -> Vec<String> {
    result
        .err()
        .map(|report| {
            report
                .errors
                .into_iter()
                .map(|CheckError { kind, message, .. }| format!("[{kind:?}] {message}"))
                .collect()
        })
        .unwrap_or_default()
}

fn ingress_diagnostics(source: &str) -> Vec<String> {
    let program = stamped(source);
    let typed = diagnostics(check_typed_program(&program));
    let ir = diagnostics(check_ir_program(&program));
    assert_eq!(
        typed, ir,
        "stamped and normalizing ingresses must report the same defects"
    );
    typed
}

#[test]
fn stamped_tuple_wrt_is_accepted() {
    let diagnostics = ingress_diagnostics(&format!(
        "{PAIR2}
         (def {{}} gradient
           (grad {{}} (var {{}} pair2) (tuple {{}} 0 1)))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn stamped_tuple_wrt_rejects_a_non_integer_element_with_the_element_reason() {
    let diagnostics = ingress_diagnostics(&format!(
        "{PAIR2}
         (def {{}} gradient
           (grad {{}} (var {{}} pair2)
             (tuple {{}} 0 (var {{}} zz))))"
    ));
    assert!(
        diagnostics.iter().any(|message| {
            message.contains("[TypeMismatch]")
                && message.contains("tuple must contain integer parameter indices")
        }),
        "{diagnostics:?}"
    );
}

#[test]
fn stamped_tuple_wrt_rejects_a_negative_element_with_the_value_reason() {
    let diagnostics = ingress_diagnostics(&format!(
        "{PAIR2}
         (def {{}} gradient
           (grad {{}} (var {{}} pair2) (tuple {{}} 0 -1)))"
    ));
    assert!(
        diagnostics.iter().any(|message| {
            message.contains("[DimensionMismatch]")
                && message.contains("index must be non-negative, got -1")
        }),
        "{diagnostics:?}"
    );
}

#[test]
fn stamped_tuple_wrt_rejects_an_empty_explicit_target() {
    let diagnostics = ingress_diagnostics(&format!(
        "{PAIR2}
         (def {{}} gradient
           (grad {{}} (var {{}} pair2) (tuple {{}})))"
    ));
    assert!(
        diagnostics.iter().any(|message| {
            message.contains("[MalformedForm]")
                && message.contains("grad `wrt` tuple must contain at least one parameter index")
        }),
        "{diagnostics:?}"
    );
}
