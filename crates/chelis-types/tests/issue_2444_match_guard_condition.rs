//! Issue #2444: a `match` arm guard and an `if` condition are one obligation.
//!
//! Both positions require a `bool`. `infer_if` discharged that by unifying the
//! condition with `bool`, while the guard check applied the substitution and
//! required the result to ALREADY be `bool`. An `eq` over a dtype-binder
//! operand is still a pending variable when its consumer runs (it could be a
//! `bool` scalar or a `bool` tensor until the operand's shape is known), so
//! `| v if eq(v, cast(0, p)) =>` was rejected as `match arm guard must be
//! bool, got ?350` while `if eq(x, cast(0, p)) then ...` checked clean.
//!
//! Each case below writes one condition expression twice, as a guard and as
//! an `if` condition, and asserts that the two positions return the same
//! verdict on both checker ingresses (chelis#1107). The accepted rows are the
//! issue's; the rejected rows are the negative parity that keeps "the same
//! unification" from meaning "no check": a condition that cannot be `bool` is
//! still rejected in both positions.
//!
//! What is claimed is the conditions listed here. It is not a claim about
//! every expression a condition position can hold.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_types::errors::CheckError;
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_surf(source).unwrap_or_else(|e| panic!("surf must parse: {source}\n{e:?}"));
    desugar_program(&decls).expect("Surf fixture must desugar")
}

fn expanded(source: &str) -> Vec<Expr> {
    expand_program(&desugared(source), &ExpansionOptions::default())
        .expect("macro expand")
        .into_exprs()
}

fn rendered(errors: &[CheckError]) -> Vec<String> {
    let mut out: Vec<String> = errors
        .iter()
        .map(|e| format!("[{:?}] {}", e.kind, e.message))
        .collect();
    out.sort();
    out
}

/// Every diagnostic for `source`, having first asserted that the stamped
/// ingress and the normalizing ingress agree on it.
fn agreed_diagnostics(source: &str) -> Vec<String> {
    let typed = match check_typed_program(&desugared(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    let ir = match check_ir_program(&expanded(source)) {
        Ok(_) => Vec::new(),
        Err(result) => rendered(&result.errors),
    };
    assert_eq!(
        typed, ir,
        "both checker ingresses must return the same diagnostics for:\n{source}"
    );
    typed
}

/// `(binder list, parameter type, condition)`: the condition is written over
/// the parameter `x` so the guard and the `if` spell it identically.
const ACCEPTED: &[(&str, &str, &str)] = &[
    // The issue's rows: a dtype-generic `eq` whose result is still pending.
    ("[p: Numeric]", "p", "eq(x, cast(0, p))"),
    ("[p: Float]", "p", "eq(x, cast(0.0, p))"),
    ("[p: Int]", "p", "gt(x, cast(1, p))"),
    ("[p: Numeric]", "p", "eq(cast(x, f64), 0.5f64)"),
    ("[p: Int]", "p", "eq(cast(x, i64), 300i64)"),
    // The concrete control, which both positions accepted before the fix.
    ("", "f32", "eq(x, 0.0)"),
];

const REJECTED: &[(&str, &str, &str)] = &[
    // A bounded binder is never `bool`, so the unification fails.
    ("[p: Numeric]", "p", "x"),
    ("[p: Int]", "p", "cast(1, p)"),
    // A concrete non-`bool` condition.
    ("", "i32", "add(x, 1)"),
    ("", "f32", "x"),
];

fn guard_program(binders: &str, param: &str, condition: &str) -> String {
    format!(
        "def decide{binders}(x: {param}) -> bool =\n  match x with {{\n    | v if {condition} => true\n    | _ => false\n  }}\n"
    )
}

fn if_program(binders: &str, param: &str, condition: &str) -> String {
    format!("def decide{binders}(x: {param}) -> bool = if {condition} then true else false\n")
}

/// REGRESSION TEST. Every issue row is rejected as `match arm guard must be
/// bool, got ?N` before the fix; the `if` spelling of the same condition
/// checks clean before and after.
#[test]
fn a_pending_condition_checks_as_a_guard_exactly_as_it_does_under_if() {
    for (binders, param, condition) in ACCEPTED {
        let guard = agreed_diagnostics(&guard_program(binders, param, condition));
        let under_if = agreed_diagnostics(&if_program(binders, param, condition));
        assert!(
            under_if.is_empty(),
            "`if {condition}` under `{binders}` must check clean, got {under_if:?}"
        );
        assert!(
            guard.is_empty(),
            "the guard `if {condition}` under `{binders}` must check clean like the \
             `if` condition does, got {guard:?}"
        );
    }
}

/// REGRESSION TEST. The issue's third spelling: the condition is bound first
/// and the guard names the binding, so the guard's type is a variable that
/// only the guard's own obligation can settle.
#[test]
fn a_guard_naming_a_pending_binding_checks_clean() {
    let diagnostics = agreed_diagnostics(
        "def decide[p: Numeric](x: p) -> bool = {\n  z = eq(x, cast(0, p))\n  match x with {\n    | _ if z => true\n    | _ => false\n  }\n}\n",
    );
    assert!(diagnostics.is_empty(), "got {diagnostics:?}");
}

/// Negative parity. A condition that cannot be `bool` is rejected in both
/// positions, each with its own position named.
#[test]
fn a_condition_that_cannot_be_bool_is_rejected_in_both_positions() {
    for (binders, param, condition) in REJECTED {
        let guard = agreed_diagnostics(&guard_program(binders, param, condition));
        let under_if = agreed_diagnostics(&if_program(binders, param, condition));
        assert!(
            guard
                .iter()
                .any(|message| message.contains("match arm guard must be bool")),
            "the guard `if {condition}` under `{binders}` must be rejected, got {guard:?}"
        );
        assert!(
            under_if
                .iter()
                .any(|message| message.contains("if condition must be bool")),
            "`if {condition}` under `{binders}` must be rejected, got {under_if:?}"
        );
    }
}
