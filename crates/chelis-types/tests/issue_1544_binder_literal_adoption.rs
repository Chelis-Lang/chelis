//! chelis#1544: P10b / spec/04 §5.6 binder-literal adoption at both checker
//! ingresses, with [04-INF-6] rigidity outside the exact cast parent.

use chelis_deep::{Expr, parse_and_stamp_file};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};

fn surf(source: &str) -> Vec<Expr> {
    let declarations = parse_surf(source).expect("Surf fixture must parse");
    desugar_program(&declarations)
}

fn deep(source: &str) -> Vec<Expr> {
    parse_and_stamp_file(source).expect("Deep fixture must parse and stamp")
}

fn diagnostics(program: &[Expr]) -> (Vec<String>, Vec<String>) {
    let render = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
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

fn assert_both_accept(program: &[Expr], label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: checker ingresses diverged");
    assert!(ir.is_empty(), "{label}: expected acceptance, got {ir:#?}");
}

fn assert_both_reject_with(program: &[Expr], citation: &str, label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: checker ingresses diverged");
    assert!(
        ir.iter().any(|diagnostic| diagnostic.contains(citation)),
        "{label}: expected a {citation} rejection, got {ir:#?}"
    );
}

#[test]
fn bounded_binder_cast_adopts_matching_numeric_literals_at_both_ingresses() {
    for (label, source) in [
        (
            "float under Float",
            "def scale[p: Float](x: p) -> p = cast(0.1, p)\n",
        ),
        (
            "integer under Int",
            "def addk[p: Int](x: p) -> p = cast(1, p)\n",
        ),
        (
            "integer under Float",
            "def addk[p: Float](x: p) -> p = cast(1, p)\n",
        ),
        (
            "float under Numeric",
            "def scale[p: Numeric](x: p) -> p = cast(0.1, p)\n",
        ),
        (
            "integer under Numeric",
            "def addk[p: Numeric](x: p) -> p = cast(1, p)\n",
        ),
        (
            "standalone signature binder",
            "sig scale[p: Float]: p -> p\ndef scale(x) = cast(0.1, p)\n",
        ),
        (
            "negative float under Float",
            "def scale[p: Float](x: p) -> p = cast(-0.1, p)\n",
        ),
        (
            "negative suffixed float under Float",
            "def scale[p: Float](x: p) -> p = cast(-0.1f64, p)\n",
        ),
    ] {
        assert_both_accept(&surf(source), label);
    }
}

#[test]
fn unbounded_binder_cannot_name_a_cast_dtype() {
    for (label, source) in [
        (
            "positive unsuffixed",
            "def scale[p](x: p) -> p = cast(0.1, p)\n",
        ),
        (
            "negative unsuffixed",
            "def scale[p](x: p) -> p = cast(-0.1, p)\n",
        ),
        (
            "positive suffixed",
            "def scale[p](x: p) -> p = cast(0.1f64, p)\n",
        ),
        (
            "negative suffixed",
            "def scale[p](x: p) -> p = cast(-0.1f64, p)\n",
        ),
        (
            "direct boolean",
            "def identity[p](x: p) -> p = cast(true, p)\n",
        ),
        (
            "direct string",
            "def identity[p](x: p) -> p = cast(\"text\", p)\n",
        ),
        ("direct unit", "def identity[p](x: p) -> p = cast((), p)\n"),
        (
            "standalone signature",
            "sig scale: p -> p\ndef scale(x) = cast(0.1, p)\n",
        ),
    ] {
        assert_both_reject_with(&surf(source), "[04-DTYPE-1]", label);
    }
}

#[test]
fn direct_literal_ascription_to_a_rigid_binder_is_rejected() {
    assert_both_reject_with(
        &surf("def scale[p: Float](x: p) -> p = (0.1 : p)\n"),
        "[04-INF-6]",
        "direct binder literal ascription",
    );
}

#[test]
fn a_literal_stamp_must_match_the_immediately_adopting_cast_binder() {
    assert_both_reject_with(
        &deep(
            "(defsig {dtype_bounds: {p: float, q: float}} scale (t-fn {} (t-var {} p) (t-var {} q) (t-var {} q)))\n\
             (def {} scale (fn {} (params {} (x {type: (t-var {} p)}) (y {type: (t-var {} q)}))\n\
               (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1) (t-var {} q))))\n",
        ),
        "[04-INF-6]",
        "wrong literal binder",
    );
}

#[test]
fn adoption_does_not_leak_into_a_computed_cast_operand() {
    assert_both_reject_with(
        &deep(
            "(defsig {dtype_bounds: {p: float}} scale (t-fn {} (t-var {} p) (t-var {} p)))\n\
             (def {} scale (fn {} (params {} (x {type: (t-var {} p)}))\n\
               (cast {} (app {} (var {} add) (var {} x)\n\
                 (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)) (t-var {} p))))\n",
        ),
        "[04-INF-6]",
        "computed cast operand",
    );
}

#[test]
fn an_undeclared_binder_cannot_author_literal_adoption() {
    assert_both_reject_with(
        &deep(
            "(defsig {} scale (t-fn {} (t-prim {} f32) (t-prim {} f32)))\n\
             (def {} scale (fn {} (params {} (x {type: (t-prim {} f32)}))\n\
               (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 0.1)\n\
                 (t-var {} p))))\n",
        ),
        "[04-DTYPE-1]",
        "undeclared literal binder",
    );
}

#[test]
fn a_float_literal_cannot_adopt_an_int_family_binder() {
    assert_both_reject_with(
        &deep(
            "(defsig {dtype_bounds: {p: int}} trunc_to (t-fn {} (t-var {} p) (t-var {} p)))\n\
             (def {} trunc_to (fn {} (params {} (x {type: (t-var {} p)}))\n\
               (cast {} (lit {surf_literal_style: \"unsuffixed\", type: (t-var {} p)} 1.9)\n\
                 (t-var {} p))))\n",
        ),
        "[04-INF-6]",
        "float literal under Int binder",
    );
}
