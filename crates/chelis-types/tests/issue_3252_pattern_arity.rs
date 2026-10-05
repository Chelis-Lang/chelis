//! Constructor patterns have the variant's exact field count at both checker ingresses.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    desugar_program(&parse_str(source).expect("Surf parses")).expect("Surf desugars")
}

fn errors(source: &str) -> Vec<CheckError> {
    let deep = desugared(source);
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

const PREFIX: &str = "type T = | A(i32, i32) | B\n";

#[test]
fn wrong_constructor_field_counts_are_type_errors() {
    for (pattern, expected, supplied) in [("A(x)", 2, 1), ("A(x, y, z)", 2, 3), ("B(x)", 0, 1)] {
        let source =
            format!("{PREFIX}def f(t: T) -> i32 = match t with {{ | {pattern} => 1 | _ => 0 }}\n");
        let diagnostics = errors(&source);
        assert!(
            diagnostics.iter().any(|e| {
                matches!(e.kind, CheckErrorKind::ArityMismatch)
                    && e.message.contains(pattern.split('(').next().unwrap())
                    && e.message.contains(&expected.to_string())
                    && e.message.contains(&supplied.to_string())
            }),
            "{pattern} must report its constructor and both counts: {diagnostics:?}"
        );
    }
}

#[test]
fn nested_constructor_pattern_has_the_same_arity_rule() {
    let source = format!(
        "{PREFIX}type Outer = | Wrap(T)\n\
         def f(t: Outer) -> i32 = match t with {{ | Wrap(A(x)) => x | _ => 0 }}\n"
    );
    let diagnostics = errors(&source);
    assert!(
        diagnostics.iter().any(|e| {
            matches!(e.kind, CheckErrorKind::ArityMismatch)
                && e.message.contains("A")
                && e.message.contains('2')
                && e.message.contains('1')
        }),
        "nested A pattern must be rejected: {diagnostics:?}"
    );
}

#[test]
fn extra_subpattern_is_still_checked() {
    let source =
        format!("{PREFIX}def f(t: T) -> i32 = match t with {{ | A(x, y, B(z)) => 1 | _ => 0 }}\n");
    let diagnostics = errors(&source);
    assert!(
        diagnostics.iter().any(|e| e.message.contains("'A'"))
            && diagnostics.iter().any(|e| e.message.contains("'B'")),
        "both the outer mismatch and extra nested pattern must be checked: {diagnostics:?}"
    );
}

#[test]
fn exact_constructor_field_counts_check_clean() {
    let source = format!(
        "{PREFIX}def f(t: T) -> i32 = match t with {{ | A(x, y) => add(x, y) | B => 0 }}\n"
    );
    assert!(
        errors(&source).is_empty(),
        "valid constructor patterns must check"
    );
}
