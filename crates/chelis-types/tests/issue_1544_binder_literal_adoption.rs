//! P10b binder-literal adoption at both checker ingresses (#1544/#1553).

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_surf::{desugar::desugar_program, parser::parse_str};
use chelis_types::{check_ir_program, check_typed_program};

fn surf(source: &str) -> Vec<Expr> {
    desugar_program(&parse_str(source).expect("Surf"))
}

fn deep(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("Deep")
}

fn diagnostics(program: &[Expr]) -> (Vec<String>, Vec<String>) {
    let render = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => vec![],
        Err(result) => result
            .errors
            .iter()
            .map(|error| format!("[{:?}] {}", error.kind, error.message))
            .collect(),
    };
    (
        render(check_ir_program(program)),
        render(check_typed_program(program)),
    )
}

fn assert_both(program: &[Expr], expected: Option<&str>, label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: checker ingresses diverged");
    match expected {
        None => assert!(ir.is_empty(), "{label}: {ir:#?}"),
        Some(citation) => assert!(
            ir.iter().any(|error| error.contains(citation)),
            "{label}: expected {citation}, got {ir:#?}"
        ),
    }
}

fn binder_program(bounds: &str, literal_meta: &str, literal: &str) -> Vec<Expr> {
    deep(&format!(
        "(defsig {{dtype_bounds: {{p: {bounds}}}}} scale (t-fn {{}} (t-var {{}} p) (t-var {{}} p)))\n\
         (def {{}} scale (fn {{}} (params {{}} (x {{type: (t-var {{}} p)}}))\n\
           (cast {{}} (lit {{{literal_meta}type: (t-var {{}} p)}} {literal}) (t-var {{}} p))))\n"
    ))
}

fn forged_local_neg_program() -> Vec<Expr> {
    deep(
        "(defsig {dtype_bounds: {p: float}} scale (t-fn {} (t-fn {} (t-var {} p) (t-var {} p)) (t-var {} p) (t-var {} p)))\n\
         (def {} scale (fn {} (params {} (neg {type: (t-fn {} (t-var {} p) (t-var {} p))}) (x {type: (t-var {} p)}))\n\
           (cast {} (app {} (var {} neg) (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)) (t-var {} p))))\n",
    )
}

#[test]
fn bounded_binder_casts_accept_matching_literals() {
    for (family, literal) in [
        ("Float", "0.1"),
        ("Int", "1"),
        ("Float", "1"),
        ("Numeric", "0.1"),
        ("Numeric", "1"),
        ("Float", "-0.1"),
        ("Float", "-0.1f64"),
    ] {
        let source = format!("def f[p: {family}](x: p) -> p = cast({literal}, p)\n");
        assert_both(&surf(&source), None, &source);
    }
    assert_both(
        &surf("sig f[p: Float]: p -> p\ndef f(x) = cast(0.1, p)\n"),
        None,
        "standalone signature",
    );
}

#[test]
fn unbounded_binders_reject_every_literal_polarity_and_kind() {
    for literal in ["0.1", "-0.1", "0.1f64", "-0.1f64", "true", "\"text\"", "()"] {
        let source = format!("def f[p](x: p) -> p = cast({literal}, p)\n");
        assert_both(&surf(&source), Some("[04-DTYPE-1]"), literal);
    }
    assert_both(
        &surf("sig f: p -> p\ndef f(x) = cast(0.1, p)\n"),
        Some("[04-DTYPE-1]"),
        "standalone signature",
    );
}

#[test]
fn only_the_exact_adopting_relation_can_stamp_a_binder_literal() {
    assert_both(
        &surf("def f[p: Float](x: p) -> p = (0.1 : p)\n"),
        Some("[04-INF-6]"),
        "ascription",
    );
    for (label, marker) in [
        ("explicit", "surf_literal_style: \"explicit\", "),
        ("malformed", "surf_literal_style: 1, "),
        ("missing", ""),
    ] {
        assert_both(
            &binder_program("float", marker, "0.1"),
            Some("[04-INF-6]"),
            label,
        );
    }
    assert_both(
        &binder_program("int", "surf_literal_style: \"unsuffixed\", ", "1.9"),
        Some("[04-INF-6]"),
        "float under Int",
    );
}

#[test]
fn adoption_does_not_cross_binders_or_computed_operands() {
    for (label, source) in [
        (
            "wrong binder",
            "(defsig {dtype_bounds: {p: float, q: float}} f (t-fn {} (t-var {} p) (t-var {} q) (t-var {} q)))\n(def {} f (fn {} (params {} (x {type: (t-var {} p)}) (y {type: (t-var {} q)})) (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1) (t-var {} q))))\n",
        ),
        (
            "computed operand",
            "(defsig {dtype_bounds: {p: float}} f (t-fn {} (t-var {} p) (t-var {} p)))\n(def {} f (fn {} (params {} (x {type: (t-var {} p)})) (cast {} (app {} (var {} add) (var {} x) (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)) (t-var {} p))))\n",
        ),
    ] {
        assert_both(&deep(source), Some("[04-INF-6]"), label);
    }
    assert_both(
        &deep(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1) (t-var {} p))))\n",
        ),
        Some("[04-DTYPE-1]"),
        "undeclared binder",
    );
    assert_both(
        &forged_local_neg_program(),
        Some("[04-INF-6]"),
        "a local callable named neg is not unary syntax",
    );
}

#[test]
fn adopted_integer_literals_fit_every_family_member() {
    for (family, literal, expected) in [
        ("Int", "127", None),
        ("Int", "-128", None),
        ("Numeric", "127", None),
        ("Numeric", "-128", None),
        ("Int", "128", Some("§5.6")),
        ("Int", "-129", Some("§5.6")),
        ("Numeric", "128", Some("§5.6")),
        ("Numeric", "-129", Some("§5.6")),
        ("Float", "10000000000.0", None),
    ] {
        let source = format!("def f[p: {family}](x: p) -> p = cast({literal}, p)\n");
        assert_both(&surf(&source), expected, &source);
    }
}
