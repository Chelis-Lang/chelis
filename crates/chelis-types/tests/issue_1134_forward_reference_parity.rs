//! chelis#1134 / [04-INF-4]: top-level declaration scope and inference order
//! are identical at the stamped typed and serialized-IR checker ingresses.

use chelis_deep::parse_and_stamp;
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};

type Diagnostics = Vec<(String, String)>;

fn deep_program(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp(source).expect("Deep fixture must parse and stamp")
}

fn surf_program(source: &str) -> Vec<chelis_deep::Expr> {
    let declarations = parse_surf(source).expect("Surf fixture must parse");
    desugar_program(&declarations)
}

fn diagnostics(program: &[chelis_deep::Expr]) -> (Diagnostics, Diagnostics) {
    let summarize = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| {
                (
                    error.kind.diagnostic_name().to_string(),
                    error.message.clone(),
                )
            })
            .collect(),
    };
    (
        summarize(check_ir_program(program)),
        summarize(check_typed_program(program)),
    )
}

fn assert_accepts_at_both_ingresses(program: &[chelis_deep::Expr], label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(ir.is_empty(), "{label}: expected acceptance, got {ir:#?}");
}

fn assert_rejects_identically(program: &[chelis_deep::Expr], expected_kind: &str, label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(
        ir.iter().any(|(kind, _)| kind == expected_kind),
        "{label}: expected {expected_kind}, got {ir:#?}"
    );
}

#[test]
fn defsig_less_forward_value_reference_is_legal_at_both_ingresses() {
    let program = deep_program(
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} int32)} 7))\n",
    );
    assert_accepts_at_both_ingresses(&program, "defsig-less forward value");
}

#[test]
fn backward_value_reference_remains_legal_at_both_ingresses() {
    let program = deep_program(
        "(def {} base (lit {type: (t-prim {} int32)} 7))\n\n\
         (def {} use_base (var {} base))\n",
    );
    assert_accepts_at_both_ingresses(&program, "backward value");
}

#[test]
fn bare_forward_generic_helper_is_inferred_before_all_callers() {
    let program = surf_program(
        r#"
def use_int(x: int32) -> int32 = identity(x)
def use_float(x: f32) -> f32 = identity(x)
def identity(x) = x
"#,
    );
    assert_accepts_at_both_ingresses(&program, "bare generic forward helper");
}

#[test]
fn module_wrapped_forward_generic_helper_has_the_same_verdict() {
    let program = surf_program(
        r#"
module ForwardParity
def use_int(x: int32) -> int32 = identity(x)
def use_float(x: f32) -> f32 = identity(x)
def identity(x) = x
"#,
    );
    assert_accepts_at_both_ingresses(&program, "module generic forward helper");
}

#[test]
fn missing_top_level_name_rejects_identically() {
    let program = deep_program("(def {} use_missing (var {} missing))\n");
    assert_rejects_identically(&program, "UnboundVariable", "missing top-level name");
}

#[test]
fn local_let_forward_reference_remains_sequential() {
    let program = deep_program(
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} int32)} 1))\n\
             (var {} x)))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "local let forward reference");
}

#[test]
fn eager_top_level_value_cycle_rejects_as_cycle_at_both_ingresses() {
    let program = deep_program(
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
    );
    assert_rejects_identically(&program, "CycleDetected", "eager value cycle");
}
