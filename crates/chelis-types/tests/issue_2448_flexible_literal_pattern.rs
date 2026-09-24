//! [04-PAT-1] remains an obligation when a pattern first sees a flexible
//! scrutinee. [04-INF-1] keeps the enclosing lambda monomorphic until use.

use chelis_deep::Expr;
use chelis_macros::{ExpansionOptions, expand_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_types::errors::{CheckError, CheckErrorKind};
use chelis_types::{check_ir_program, check_typed_program};

fn desugared(source: &str) -> Vec<Expr> {
    let decls = parse_str(source).unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    desugar_program(&decls).expect("desugar fixture")
}

fn diagnostics(source: &str) -> Vec<CheckError> {
    let deep = desugared(source);
    let typed = check_typed_program(&deep)
        .err()
        .map_or_else(Vec::new, |result| result.errors);
    let expanded = expand_program(&deep, &ExpansionOptions::default())
        .expect("expand fixture")
        .into_exprs();
    let ir = check_ir_program(&expanded)
        .err()
        .map_or_else(Vec::new, |result| result.errors);
    let messages = |errors: &[CheckError]| {
        errors
            .iter()
            .map(|error| {
                (
                    format!("{:?}: {}", error.kind, error.message),
                    error.suggestions.clone(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(
        messages(&typed),
        messages(&ir),
        "checker ingress parity: {source}"
    );
    typed
}

fn sole_pattern_error(source: &str, fragments: &[&str]) {
    let errors = diagnostics(source);
    let [error] = errors.as_slice() else {
        panic!("expected one [04-PAT-1] error, got {errors:?} for {source}");
    };
    assert!(matches!(error.kind, CheckErrorKind::TypeMismatch));
    assert!(error.message.contains("[04-PAT-1]"), "{error:?}");
    for fragment in fragments {
        assert!(
            error.message.contains(fragment),
            "missing {fragment:?}: {error:?}"
        );
    }
}

#[test]
fn first_application_decides_family_range_and_nonprimitive_patterns() {
    for (source, fragments) in [
        (
            "def h(x: f32) -> bool = {\n  k = fn (y) -> match y with { | 0 => true | _ => false }\n  k(x)\n}\n",
            vec!["integer literal pattern `0`", "`f32`"],
        ),
        (
            "def h(x: i8) -> bool = {\n  k = fn (y) -> match y with { | 300 => true | _ => false }\n  k(x)\n}\n",
            vec!["integer literal pattern `300`", "`i8`", "[-128, 127]"],
        ),
        (
            "def h(x: f16) -> bool = {\n  k = fn (y) -> match y with { | 70000.0 => true | _ => false }\n  k(x)\n}\n",
            vec!["float literal pattern `70000.0`", "`f16`", "infinity"],
        ),
        (
            "def h(x: (i32, f32)) -> bool = {\n  k = fn (y) -> match y with { | 0 => true | _ => false }\n  k(x)\n}\n",
            vec!["integer literal pattern `0`", "primitive scrutinee"],
        ),
        (
            "def h(x: (i32, f32)) -> bool = {\n  k = fn (y) -> match y with { | (_, 0) => true | _ => false }\n  k(x)\n}\n",
            vec!["integer literal pattern `0`", "`f32`"],
        ),
    ] {
        sole_pattern_error(source, &fragments);
    }
}

#[test]
fn unresolved_pattern_obligation_rejects_at_own_declaration_boundary() {
    for source in [
        "def h() = {\n  k = fn (y) -> match y with { | 0 => true | _ => false }\n  k\n}\n",
        "def pick(y) -> bool = match y with { | 0 => true | _ => false }\nout = pick(0i32)\n",
    ] {
        sole_pattern_error(source, &["literal pattern", "declaration boundary"]);
    }
}

#[test]
fn an_authored_binder_reached_after_pattern_inference_is_checked() {
    sole_pattern_error(
        "def h[p: Int](x: p) -> bool = {\n  k = fn (y) -> match y with { | 300 => true | _ => false }\n  k(x)\n}\n",
        &["`p: Int`", "`i8`"],
    );
}

#[test]
fn admissible_first_use_and_unconstrained_lambda_still_check() {
    for source in [
        "def h(x: i64) -> bool = {\n  k = fn (y) -> match y with { | 300 => true | _ => false }\n  k(x)\n}\n",
        "def h(x: f32) -> bool = {\n  k = fn (y) -> match y with { | 0.0 => true | _ => false }\n  k(x)\n}\n",
        "def h(x: f32) -> bool = {\n  k = fn (y) -> match y with { | 70000.0 => true | _ => false }\n  k(x)\n}\n",
        "def h() -> (i32, bool) = {\n  k = fn (y) -> y\n  (k(1), k(true))\n}\n",
    ] {
        assert!(diagnostics(source).is_empty(), "must check: {source}");
    }
}

#[test]
fn a_pattern_lambda_stays_monomorphic_across_multiple_uses() {
    // [04-INF-1]: even two widths that individually admit `| 0 =>` do not
    // re-instantiate the lambda. Its first application fixes one monotype.
    for second_call in ["k(0i64)", "k(0.0f32)"] {
        let source = format!(
            "def h() = {{\n  k = fn (y) -> match y with {{ | 0 => true | _ => false }}\n  a = k(0i32)\n  b = {second_call}\n  (a, b)\n}}\n"
        );
        assert!(
            !diagnostics(&source).is_empty(),
            "[04-INF-1] binds the lambda at its first use: {source}"
        );
    }
    let same_dtype = "def h() = {\n  k = fn (y) -> match y with { | 0 => true | _ => false }\n  a = k(0i32)\n  b = k(1i32)\n  (a, b)\n}\n";
    assert!(
        diagnostics(same_dtype).is_empty(),
        "reusing the same monomorphic type must check"
    );
}

#[test]
fn float_pattern_diagnostics_use_short_exponents() {
    for (pattern, expected) in [("3.4e38", "3.4e38"), ("1e-7", "1e-7")] {
        sole_pattern_error(
            &format!("def f(x: i32) -> i32 = match x with {{ | {pattern} => 1 | _ => 0 }}\n"),
            &[&format!("floating-point literal pattern `{expected}`")],
        );
    }
    sole_pattern_error(
        "def f(x: i32) -> i32 = match x with { | 1.0 => 1 | _ => 0 }\n",
        &["floating-point literal pattern `1.0`"],
    );
}
