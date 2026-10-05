//! Patterns must be admissible at their scrutinee type at both checker ingresses.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep: Vec<Expr> =
        desugar_program(&parse_str(source).expect("Surf parses")).expect("Surf desugars");
    let expanded = expand_program(&deep, &ExpansionOptions::default())
        .expect("macro expansion")
        .into_exprs();
    let typed = check_typed_program(&deep)
        .err()
        .map_or_else(Vec::new, |r| r.errors);
    let ir = check_ir_program(&expanded)
        .err()
        .map_or_else(Vec::new, |r| r.errors);
    let render = |errors: &[CheckError]| {
        errors
            .iter()
            .map(|e| format!("{:?}: {}", e.kind, e.message))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        render(&typed),
        render(&ir),
        "checker ingress parity: {source}"
    );
    typed
}

#[test]
fn foreign_nominal_constructor_rejects_even_with_a_fallback() {
    let source = "type T = | A(i32)\n\
                  type U = | C(i32)\n\
                  def f(u: U) -> i32 = match u with { | A(a) => a | C(c) => c }\n";
    let errors = diagnostics(source);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
        "foreign nominal constructor must be rejected: {errors:?}"
    );
}

#[test]
fn foreign_record_constructor_rejects_even_with_a_fallback() {
    let source = "type T = | A { value: i32 }\n\
                  type U = | C { value: i32 }\n\
                  def f(u: U) -> i32 = match u with { | A { value } => value | C { value } => value }\n";
    let errors = diagnostics(source);
    assert!(
        errors
            .iter()
            .any(|e| matches!(e.kind, CheckErrorKind::TypeMismatch)),
        "foreign record constructor must be rejected: {errors:?}"
    );
}

#[test]
fn tuple_pattern_requires_a_tuple_of_exact_arity() {
    for (source, label) in [
        (
            "def f(n: i32) -> i32 = match n with { | (a, b) => a | _ => 0 }\n",
            "scalar",
        ),
        (
            "def f(t: (i32, i32)) -> i32 = match t with { | (a, b, c) => a | _ => 0 }\n",
            "arity",
        ),
        (
            "def f(t: ((i32, i32), i32)) -> i32 = match t with { | ((a, b, c), d) => a | _ => 0 }\n",
            "nested arity",
        ),
    ] {
        let errors = diagnostics(source);
        assert!(
            errors.iter().any(|e| matches!(
                e.kind,
                CheckErrorKind::TypeMismatch | CheckErrorKind::ArityMismatch
            )),
            "{label} tuple mismatch must be rejected: {errors:?}"
        );
    }
}

#[test]
fn valid_patterns_preserve_field_and_tuple_element_types() {
    for source in [
        "type T = | A(i32) | B\n\
         def f(t: T) -> i32 = match t with { | A(a) => add(a, 1) | B => 0 }\n",
        "def f(t: (i32, i32)) -> i32 = match t with { | (a, b) => add(a, b) }\n",
        "def f() -> i32 = (fn (q) -> match q with { | (a, b) => add(a, b) })((2i32, 3i32))\n",
    ] {
        let errors = diagnostics(source);
        assert!(
            errors.is_empty(),
            "valid pattern must check: {source}: {errors:?}"
        );
    }
}
