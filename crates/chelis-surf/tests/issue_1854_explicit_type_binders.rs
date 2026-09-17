//! chelis#1854: Surf type variables are introduced only by a declaration's
//! explicit `[..]` binder list. Unknown lowercase scalar and tensor-precision
//! names must remain `t-prim` so the shared Deep type resolver can reject them.

use chelis_deep::{parser::parse_str as parse_deep, printer::print_canonical_flat};
use chelis_surf::{
    desugar::desugar_program,
    parser::parse_str,
    resugar::{ResugarError, resugar_program},
};

const UNKNOWN_DTYPES: [&str; 7] = [
    "float32", "fp32", "int33", "double", "half", "f8e4m3", "f8e5m2",
];

fn deep(source: &str) -> String {
    let declarations = parse_str(source).expect("Surf fixture must parse");
    print_canonical_flat(&desugar_program(&declarations))
}

#[test]
fn unlisted_scalar_names_lower_to_unknown_primitives() {
    for name in UNKNOWN_DTYPES {
        for source in [
            format!("def ident(x: {name}) -> {name} = x"),
            format!("sig ident: {name} -> {name}\ndef ident(x) = x"),
        ] {
            let rendered = deep(&source);
            assert!(
                rendered.contains(&format!("(t-prim {{}} {name})")),
                "`{name}` must reach primitive resolution: {rendered}"
            );
            assert!(
                !rendered.contains(&format!("(t-var {{}} {name})")),
                "`{name}` must not become an implicit type binder: {rendered}"
            );
        }
    }
}

#[test]
fn unlisted_tensor_precision_names_lower_to_unknown_primitives() {
    let rendered = deep("sig ident: tensor[n, p] -> tensor[n, p]\ndef ident(x) = x");
    assert!(
        rendered.contains("(d-var {} n)"),
        "the dimension remains structurally a variable use: {rendered}"
    );
    assert!(
        rendered.contains("(t-prim {} p)") && !rendered.contains("(t-var {} p)"),
        "an unlisted precision name must not become a type binder: {rendered}"
    );
    assert!(
        rendered.contains("(defsig {} ident (t-fn "),
        "the monomorphic Deep form must omit a binder-list child: {rendered}"
    );
}

#[test]
fn declared_binders_lower_to_variable_forms_in_both_signature_forms() {
    for (source, name) in [
        ("def ident[a](x: a) -> a = x", "a"),
        ("sig ident[a]: a -> a\ndef ident(x) = x", "a"),
        ("def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = x", "p"),
        (
            "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\ndef ident(x) = x",
            "p",
        ),
    ] {
        let rendered = deep(source);
        assert!(
            rendered.contains(&format!("(t-var {{}} {name})")),
            "a declared binder must lower to `t-var`: {source}\n{rendered}"
        );
        assert!(
            !rendered.contains(&format!("(t-prim {{}} {name})")),
            "a declared binder must not lower to `t-prim`: {source}\n{rendered}"
        );
    }
}

#[test]
fn matching_def_annotations_reuse_the_standalone_signature_binders() {
    let rendered = deep(
        "sig ident[n, p]: tensor[n, p] -> tensor[n, p]\n\
         def ident(x: tensor[n, p]) -> tensor[n, p] = x",
    );
    assert!(
        rendered.contains("(defsig {} ident (n p) ")
            && rendered.matches("(d-var {} n)").count() >= 2
            && rendered.matches("(t-var {} p)").count() >= 2
            && !rendered.contains("(t-prim {} p)"),
        "the owning sig binder scope must cover matching-def annotations: {rendered}"
    );
}

#[test]
fn matching_def_cannot_declare_a_second_binder_list() {
    let error = parse_str(
        "sig ident[a]: a -> a\n\
         def ident[a](x: a) -> a = x",
    )
    .expect_err("one declaration has one binder owner");
    assert!(
        error
            .to_string()
            .contains("a declaration's `defsig` owns its binders"),
        "wrong duplicate-owner diagnostic: {error}"
    );
}

#[test]
fn duplicate_surf_binder_names_are_rejected_before_desugaring() {
    let error = parse_str("sig ident[a, a]: a -> a\ndef ident(x) = x")
        .expect_err("a declaration cannot bind one name twice");
    assert!(
        error.to_string().contains("duplicate binder `a`"),
        "wrong duplicate-binder diagnostic: {error}"
    );
}

#[test]
fn dimension_and_rank_variables_require_declared_binders() {
    let unlisted = deep("sig shaped: tensor[n, f32] -> tensor[..r, f32]\ndef shaped(x) = x");
    assert!(
        unlisted.contains("(defsig {} shaped (t-fn ")
            && unlisted.contains("(d-var {} n)")
            && unlisted.contains("(d-rank {} r)"),
        "unlisted variables must remain uses outside an empty binder list: {unlisted}"
    );

    let listed = deep("sig shaped[n, r]: tensor[n, f32] -> tensor[..r, f32]\ndef shaped(x) = x");
    assert!(
        listed.contains("(defsig {} shaped (n r) ")
            && listed.contains("(d-var {} n)")
            && listed.contains("(d-rank {} r)"),
        "listed dimension and rank binders must be carried explicitly: {listed}"
    );
}

#[test]
fn active_primitives_are_never_rebound_by_a_binder_list() {
    let rendered = deep("def ident[f32](x: f32) -> f32 = x");
    assert!(
        rendered.contains("(t-prim {} f32)") && !rendered.contains("(t-var {} f32)"),
        "the active primitive vocabulary must outrank binder spelling: {rendered}"
    );
}

#[test]
fn active_primitive_t_vars_have_no_resugar_fallback() {
    let program = parse_deep(
        "(defsig {} ident (f32)
           (t-fn {} (t-var {} f32) (t-var {} f32)))
         (def {} ident (fn {} (params {} x) (var {} x)))",
    )
    .expect("structural Deep fixture must parse");
    let error = resugar_program(&program).expect_err("active primitives cannot be Surf binders");
    assert_eq!(
        error,
        ResugarError::InvalidSurfaceIdentifier {
            name: "f32".to_string(),
            role: "type variable",
        }
    );
}

#[test]
fn defsig_resugar_arity_reports_the_two_or_three_child_contract() {
    for (source, actual) in [
        ("(defsig {} ident)", 1),
        (
            "(defsig {} ident (a) (t-fn {} (t-var {} a) (t-var {} a)) extra)",
            4,
        ),
    ] {
        let program = parse_deep(source).expect("malformed-arity fixture must parse structurally");
        let error = resugar_program(&program).expect_err("invalid defsig arity must not resugar");
        assert_eq!(
            error,
            ResugarError::AlternativeArity {
                tag: "defsig",
                first: 2,
                second: 3,
                actual,
            }
        );
        assert!(
            error.to_string().contains("expects 2 or 3 children"),
            "the diagnostic must state the complete contract: {error}"
        );
    }
}

#[test]
fn declared_but_unused_unbounded_binders_are_preserved() {
    for source in [
        "def constant[a]() -> i32 = 1i32",
        "sig constant[a]: () -> i32\ndef constant() = 1i32",
    ] {
        let rendered = deep(source);
        assert!(
            rendered.contains("(defsig {} constant (a) "),
            "a vacuous explicit quantifier must remain structural: {rendered}"
        );
        let program = parse_deep(&rendered).expect("desugared Deep must parse");
        let recovered = resugar_program(&program).expect("unused unbounded binder must resugar");
        let redeep = print_canonical_flat(&desugar_program(&recovered));
        assert!(
            redeep.contains("(defsig {} constant (a) "),
            "round-trip must preserve the declared binder: {redeep}"
        );
    }
}
