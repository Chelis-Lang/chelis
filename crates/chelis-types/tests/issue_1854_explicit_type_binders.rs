//! chelis#1854 acceptance at both checker ingresses.

use chelis_deep::{Expr, parser::parse_str as parse_deep};
use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};
use chelis_types::{check_ir_program, check_typed_program};

const REJECTED_DTYPES: [(&str, Option<&str>); 7] = [
    ("float32", Some("f32")),
    ("fp32", Some("f32")),
    ("int33", Some("i32")),
    ("double", Some("f64")),
    ("half", Some("f16")),
    ("f8e4m3", None),
    ("f8e5m2", None),
];

fn surf(source: &str) -> Vec<Expr> {
    desugar_program(&parse_surf(source).expect("Surf fixture must parse"))
}

#[derive(Debug, PartialEq, Eq)]
struct Diagnostic {
    kind: String,
    message: String,
    suggestions: Vec<String>,
    severity_bits: u64,
    expected: Option<String>,
    got: Option<String>,
    span_offset: Option<usize>,
    span_id: Option<String>,
}

fn diagnostics(
    result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>,
) -> Vec<Diagnostic> {
    result.err().map_or_else(Vec::new, |report| {
        report
            .errors
            .into_iter()
            .map(|error| Diagnostic {
                kind: error.kind.diagnostic_name().to_string(),
                message: error.message,
                suggestions: error.suggestions,
                severity_bits: error.severity.to_bits(),
                expected: error.expected,
                got: error.got,
                span_offset: error.span_offset,
                span_id: error.span_id,
            })
            .collect()
    })
}

fn assert_both_ingresses_reject_once(program: &[Expr], name: &str, nearest: Option<&str>) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must report the same ordered diagnostics for `{name}`"
    );
    assert_eq!(
        ir.len(),
        1,
        "`{name}` must have one owning diagnostic: {ir:?}"
    );
    assert!(
        ir[0].message.contains(&format!("`{name}`")),
        "the diagnostic must name the unknown spelling: {ir:?}"
    );
    if let Some(nearest) = nearest {
        assert!(
            ir[0]
                .suggestions
                .iter()
                .any(|suggestion| suggestion.contains(&format!("`{nearest}`"))),
            "the diagnostic for `{name}` must suggest nearest dtype `{nearest}`: {ir:?}"
        );
    }
}

#[test]
fn unknown_surf_scalar_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in REJECTED_DTYPES {
        assert_both_ingresses_reject_once(
            &surf(&format!("def ident(x: {name}) = 0.0f32")),
            name,
            nearest,
        );
    }
}

#[test]
fn unknown_surf_tensor_precision_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in REJECTED_DTYPES {
        assert_both_ingresses_reject_once(
            &surf(&format!(
                "sig ident: tensor[2, {name}] -> f32\ndef ident(x) = 0.0f32"
            )),
            name,
            nearest,
        );
    }
}

#[test]
fn undeclared_surf_dimension_and_rank_variables_reject_at_both_ingresses() {
    for (source, name, needle) in [
        (
            "sig shaped: tensor[n, f32] -> f32\ndef shaped(x) = 0.0f32",
            "n",
            "undeclared dimension variable",
        ),
        (
            "sig shaped: tensor[..r, f32] -> f32\ndef shaped(x) = 0.0f32",
            "r",
            "undeclared rank variable",
        ),
    ] {
        let program = surf(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "Surf ingress parity for undeclared `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong Surf undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_surf_binders_and_active_primitives_check_clean() {
    for source in [
        "def ident[a](x: a) -> a = x",
        "sig ident[a]: a -> a\ndef ident(x) = x",
        "def constant[a]() -> i32 = 1i32",
        "sig constant[a]: i32 -> i32\ndef constant(x) = x",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = x",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\ndef ident(x) = x",
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def ident(x: tensor[n, p]) -> tensor[n, p] = x",
        "sig ident[r]: tensor[..r, f32] -> tensor[..r, f32]\ndef ident(x) = x",
        "def ident(x: f32) -> f32 = x",
    ] {
        let program = surf(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!("IR ingress must accept `{source}`: {:?}", report.errors)
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!("typed ingress must accept `{source}`: {:?}", report.errors)
        });
    }
}

#[test]
fn explicit_deep_t_var_nodes_remain_valid_binders() {
    let program = parse_deep(
        "(defsig {} ident (float32)
           (t-fn {} (t-var {} float32) (t-var {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-var {} float32)})) (var {} x)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("an authored Deep `t-var` is explicit");
    check_typed_program(&program).expect("typed ingress must agree");
}

#[test]
fn active_primitive_names_cannot_be_rebound_by_deep_t_var_nodes() {
    for name in ["f32", "i64", "bool"] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident ({name})
               (t-fn {{}} (t-var {{}} {name}) (t-prim {{}} f32)))
             (def {{}} ident (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "active-primitive rejection parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains("active primitive")
                && ir[0].message.contains(&format!("`{name}`")),
            "wrong active-primitive binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_deep_unknown_primitives_reject_like_desugared_surf() {
    for (name, nearest) in REJECTED_DTYPES {
        let program = parse_deep(&format!(
            "(defsig {{}} ident
               (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} f32)))
             (def {{}} ident
               (fn {{}} (params {{}} x) (lit {{type: (t-prim {{}} f32)}} 0.0)))"
        ))
        .expect("Deep fixture must parse");
        assert_both_ingresses_reject_once(&program, name, nearest);
    }
}

#[test]
fn undeclared_deep_type_dimension_and_rank_variables_reject() {
    for (name, ty, needle) in [
        (
            "a",
            "(t-fn {} (t-var {} a) (t-prim {} f32))",
            "undeclared type variable",
        ),
        (
            "n",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {ty})
             (def {{}} f (fn {{}} (params {{}} x) (lit {{type: (t-prim {{}} f32)}} 0.0)))"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "ingress parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn empty_deep_binder_list_is_rejected_at_both_checker_ingresses() {
    let program = parse_deep(
        "(defsig {} f () (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 0.0)))",
    )
    .expect("structural fixture parses before semantic validation");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(typed, ir, "empty-list rejection must have ingress parity");
    assert_eq!(ir.len(), 1, "empty binder list must have one owner: {ir:?}");
    assert!(
        ir[0].message.contains("empty list must be omitted"),
        "wrong empty-list diagnostic: {ir:?}"
    );
}

#[test]
fn malformed_and_duplicate_deep_binder_lists_reject_at_both_ingresses() {
    for (binders, needle) in [
        ("(a a)", "duplicate `defsig` binder `a`"),
        (
            "(a 1)",
            "expected a nonempty structural list of distinct symbol names",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} (t-fn {{}} (t-var {{}} a) (t-var {{}} a)))
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("structural fixture parses before checker validation");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "malformed-list ingress parity for `{binders}`");
        assert_eq!(ir.len(), 1, "`{binders}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle),
            "wrong malformed-list diagnostic for `{binders}`: {ir:?}"
        );
    }
}

#[test]
fn declared_deep_type_dimension_and_rank_variables_check_clean() {
    for (binders, ty) in [
        ("(a)", "(t-fn {} (t-var {} a) (t-var {} a))"),
        (
            "(n)",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) \
                      (t-tensor {} (d-var {} n) (t-prim {} f32)))",
        ),
        (
            "(r)",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) \
                      (t-tensor {} (d-rank {} r) (t-prim {} f32)))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} {ty})
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        check_ir_program(&program).expect("declared Deep binder must check");
        check_typed_program(&program).expect("typed ingress must agree");
    }
}

#[test]
fn declared_but_unused_deep_binder_is_a_vacuous_quantifier() {
    let program = parse_deep(
        "(defsig {} constant (a) (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("IR ingress must accept an unused unbounded binder");
    check_typed_program(&program).expect("typed ingress must accept the same binder policy");
}

#[test]
fn declared_but_unused_bounded_deep_binder_rejects_at_both_ingresses() {
    let program = parse_deep(
        "(defsig {dtype_bounds: {a: numeric}} constant (a)
           (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "bounded-unused rejection must have ingress parity"
    );
    assert_eq!(
        ir.len(),
        1,
        "bounded-unused binder must report once: {ir:?}"
    );
    assert!(
        ir[0].message.contains("does not occur") && ir[0].message.contains("`a`"),
        "wrong bounded-unused diagnostic: {ir:?}"
    );
}
