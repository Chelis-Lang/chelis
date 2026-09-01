//! chelis#1134 / [04-INF-4]: top-level eager-value scope is identical at the
//! stamped typed and serialized-IR checker ingresses.

use chelis_deep::parse_and_stamp;
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{TypeEnv, check_ir_program, check_ir_with_context, check_typed_program};

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
fn defsig_less_forward_value_reference_rejects_at_both_ingresses() {
    let program = deep_program(
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} int32)} 7))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "defsig-less forward value");
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
fn ascribed_external_input_self_reference_is_legal_at_both_ingresses() {
    let program = surf_program("x = (x : tensor[4, f32])\n");
    assert_accepts_at_both_ingresses(&program, "external input self-reference");
}

#[test]
fn declaration_typed_self_reference_is_an_external_input_at_both_ingresses() {
    let program = surf_program("x: tensor[4, f32] = x\n");
    assert_accepts_at_both_ingresses(&program, "declaration-typed external input");
}

#[test]
fn bare_self_reference_is_not_an_external_input_at_either_ingress() {
    let program = surf_program("x = x\n");
    assert_rejects_identically(&program, "CycleDetected", "bare self-reference");
}

#[test]
fn type_stamped_deep_self_reference_is_an_explicit_external_input() {
    let program = deep_program("(def {} x (var {type: (t-prim {} int32)} x))\n");
    assert_accepts_at_both_ingresses(&program, "typed Deep external input");
}

#[test]
fn a_later_external_input_is_not_visible_to_an_earlier_declaration() {
    let program = surf_program(
        r#"
module ExternalScope
def capture() -> int32 = x
x = (x : int32)
"#,
    );
    assert_rejects_identically(&program, "UnboundVariable", "later external input");
}

#[test]
fn context_check_cannot_prebind_a_later_external_input_globally() {
    let program = surf_program(
        r#"
module ExternalContext
def capture() -> int32 = x
x = (x : int32)
"#,
    );
    let result = check_ir_with_context(&TypeEnv::empty(), &program)
        .expect_err("later external input must remain unavailable in context checks");
    assert!(
        result.errors.iter().any(|error| matches!(
            error.kind,
            chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
        ) && error.message.contains("x")),
        "expected x to remain unbound: {result:#?}"
    );
}

#[test]
fn sequential_let_shadow_does_not_create_a_top_level_dependency() {
    let program = surf_program(
        r#"
module ShadowParity
def helper() = {
  result = 1
  copied = result
  copied
}
result = helper()
"#,
    );
    assert_accepts_at_both_ingresses(&program, "sequential let shadow");
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

#[test]
fn diagnostics_remain_in_source_order_when_a_value_mentions_a_later_value() {
    let program = deep_program(
        "(def {} first\n\
           (let {}\n\
             (bind {} dependency (var {} later))\n\
             (app {} (var {} missing_first))))\n\n\
         (def {} later (app {} (var {} missing_later)))\n",
    );
    let (ir, typed) = diagnostics(&program);
    assert_eq!(ir, typed, "ingress diagnostics diverged");
    let messages = ir
        .iter()
        .map(|(_, message)| message.as_str())
        .collect::<Vec<_>>();
    let first = messages
        .iter()
        .position(|message| message.contains("missing_first"))
        .expect("missing_first diagnostic");
    let later = messages
        .iter()
        .position(|message| message.contains("missing_later"))
        .expect("missing_later diagnostic");
    assert!(
        first < later,
        "diagnostics must retain source order: {ir:#?}"
    );
}
