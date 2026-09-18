//! chelis#1125 PP7/E5e: structural function-tail readers preserve ingress parity.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn stamped(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("fixture must stamp")
}

fn messages(errors: Vec<CheckError>) -> Vec<String> {
    errors
        .into_iter()
        .map(|error| format!("[{:?}] {}", error.kind, error.message))
        .collect()
}

fn ingress_messages(source: &str) -> Vec<String> {
    let program = stamped(source);
    let typed = check_typed_program(&program)
        .err()
        .map(|report| messages(report.errors))
        .unwrap_or_default();
    let ir = check_ir_program(&program)
        .err()
        .map(|report| messages(report.errors))
        .unwrap_or_default();
    assert_eq!(typed, ir, "checker ingresses must report the same defects");
    typed
}

const CHOICE: &str = "
    (deftype {} Choice () (variant {} Left) (variant {} Right))";

#[test]
fn guarded_match_tail_accepts_the_same_owned_return_on_both_ingresses() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (n)
           (t-fn {{}} (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32)))
             (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice value)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (app {{}} (var {{}} eq)
                   (lit {{type: (t-prim {{}} i32)}} 1)
                   (lit {{type: (t-prim {{}} i32)}} 1))
                 (var {{}} value))
               (arm {{}} (pat-ctor {{}} Right) () (var {{}} value)))))"
    ));
    assert!(diagnostics.is_empty(), "{diagnostics:?}");
}

#[test]
fn guarded_match_tail_rejects_a_different_return_on_both_ingresses() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose (n)
           (t-fn {{}} (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32)))
             (t-ref {{}} (t-tensor {{}} (d-var {{}} n) (t-prim {{}} i32)))
             (t-tensor {{}} (d-var {{}} n) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice expected wrong)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (app {{}} (var {{}} eq)
                   (lit {{type: (t-prim {{}} i32)}} 1)
                   (lit {{type: (t-prim {{}} i32)}} 1))
                 (var {{}} expected))
               (arm {{}} (pat-ctor {{}} Right) () (var {{}} wrong)))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn same_local_alias_spelling_does_not_merge_distinct_borrowed_parameters() {
    let diagnostics = ingress_messages(&format!(
        "{CHOICE}
         (defsig {{}} choose
           (t-fn {{}}
             (t-adt {{}} Choice)
             (t-ref {{}} (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32)))
             (t-ref {{}} (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32)))
             (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} f32))))
         (def {{}} choose
           (fn {{}} (params {{}} choice left right)
             (match {{}} (var {{}} choice)
               (arm {{}} (pat-ctor {{}} Left)
                 (lit {{type: (t-prim {{}} bool)}} true)
                 (let {{}} (bind {{}} result (var {{}} left)) (var {{}} result)))
               (arm {{}} (pat-ctor {{}} Right) ()
                 (let {{}} (bind {{}} result (var {{}} right)) (var {{}} result))))))"
    ));
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}

#[test]
fn locally_constructed_borrow_is_not_relaxed_into_an_owned_return() {
    let diagnostics = ingress_messages(
        "(defsig {} local
           (t-fn {}
             (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))
             (t-tensor {} (d-lit {} 3) (t-prim {} f32))))
         (def {} local
           (fn {} (params {} input)
             (let {}
               (bind {}
                 owned (copy {} (var {} input))
                 borrowed (borrow {} (var {} owned)))
               (var {} borrowed))))",
    );
    assert!(
        diagnostics
            .iter()
            .any(|message| message.contains("doesn't match declared signature")),
        "{diagnostics:?}"
    );
}
