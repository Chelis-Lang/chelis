//! chelis#1854: Surf type variables are introduced only by a declaration's
//! explicit `[..]` binder list. Unknown lowercase scalar and tensor-precision
//! names must remain `t-prim` so the shared Deep type resolver can reject them.

use chelis_deep::{
    Atom as DeepAtom, DeepTag, Expr as DeepExpr, List as DeepList, Metadata, Span,
    parser::parse_str as parse_deep, printer::print_canonical_flat,
};
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

fn malformed_defsig(children: Vec<DeepExpr>) -> DeepExpr {
    let span = Span::new(0, 0);
    let mut elements = vec![
        DeepExpr::Atom(DeepAtom::Tag(DeepTag::Defsig), span),
        DeepExpr::Map(Metadata::default(), span),
    ];
    elements.extend(children);
    DeepExpr::List(DeepList { elements }, span)
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
fn declaration_body_annotations_reuse_bare_type_binders_without_broadening_tensors() {
    for source in [
        "def ident[p](x: p) -> p = { y: p = x\n y }",
        "def ident[p](x: p) -> p = { apply = fn(y: p) -> y\n apply(x) }",
        "def ident[p](x: p) -> p = (x: p)",
        "def ident[p](xs: List[p]) -> List[p] = { ys: List[p] = xs\n ys }",
        "sig ident[p]: p -> p\ndef ident(x) = { y: p = x\n y }",
    ] {
        let rendered = deep(source);
        assert!(
            rendered.contains("(t-var {} p)") && !rendered.contains("(t-prim {} p)"),
            "an enclosing declaration binder must cover ordinary body type positions: \
             {source}\n{rendered}"
        );
    }

    for source in [
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         { y: tensor[n, p] = x\n y }",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         { apply = fn(y: tensor[n, p]) -> y\n apply(x) }",
        "def ident[n, p](x: tensor[n, p]) -> tensor[n, p] = \
         (x: tensor[n, p])",
    ] {
        let rendered = deep(source);
        assert!(
            rendered.contains("(t-prim {} p)"),
            "chelis#1904 remains the owner of body-local tensor precision scope: \
             {source}\n{rendered}"
        );
    }
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
            .contains("a declaration's `defsig` owns its binder list"),
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
fn forbidden_dtype_vocabulary_is_rejected_at_the_surf_binder_list() {
    for (name, source) in [
        ("f32", "def shaped[f32](x: tensor[f32, i32]) -> i32 = 0i32"),
        (
            "f8e4m3",
            "def shaped[f8e4m3](x: tensor[f8e4m3, i32]) -> i32 = 0i32",
        ),
        (
            "i32",
            "def shaped[i32](x: tensor[..i32, f32]) -> i32 = 0i32",
        ),
        (
            "f8e5m2",
            "def shaped[f8e5m2](x: tensor[..f8e5m2, f32]) -> i32 = 0i32",
        ),
        ("bool", "def constant[bool]() -> i32 = 1i32"),
        ("u8", "def constant[u8]() -> i32 = 1i32"),
        ("int32", "def constant[int32]() -> i32 = 1i32"),
        ("complex64", "def constant[complex64]() -> i32 = 1i32"),
    ] {
        let error = parse_str(source).expect_err("dtype vocabulary cannot be rebound");
        assert!(
            error.to_string().contains(&format!("`{name}`"))
                && error.to_string().contains("cannot be a declaration binder"),
            "wrong forbidden-binder diagnostic for `{name}`: {error}"
        );
    }
}

#[test]
fn arbitrary_non_dtype_names_remain_legal_surf_binders() {
    for source in [
        "def shaped[float32](x: tensor[float32, f32]) -> tensor[float32, f32] = x",
        "def shaped[float32](x: tensor[..float32, f32]) -> tensor[..float32, f32] = x",
        "def constant[float32]() -> i32 = 1i32",
    ] {
        let rendered = deep(source);
        assert!(
            rendered.contains("(defsig {}")
                && rendered.contains("(float32)")
                && (rendered.contains("(d-var {} float32)")
                    || rendered.contains("(d-rank {} float32)")
                    || rendered.contains("(t-prim {} i32)")),
            "an arbitrary intentional binder must remain legal: {source}\n{rendered}"
        );
    }
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
    let span = Span::new(0, 0);
    let name = || DeepExpr::Atom(DeepAtom::Name("ident".to_string()), span);
    let binders = || {
        DeepExpr::BareList(
            vec![DeepExpr::Atom(DeepAtom::Name("a".to_string()), span)],
            span,
        )
    };
    let ty = parse_deep("(t-fn {} (t-var {} a) (t-var {} a))")
        .expect("valid type fixture")
        .remove(0);
    let extra = parse_deep("(t-prim {} f32)")
        .expect("valid extra fixture")
        .remove(0);
    for (program, actual) in [
        (vec![malformed_defsig(vec![name()])], 1),
        (
            vec![malformed_defsig(vec![
                name(),
                binders(),
                ty.clone(),
                extra.clone(),
            ])],
            4,
        ),
    ] {
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
fn deep_defsig_ingress_reports_the_declared_arity_before_child_decoding() {
    for (source, actual) in [
        ("(defsig {} ident)", 1),
        ("(defsig {} ident (a) (t-var {} a) (t-prim {} f32))", 4),
    ] {
        let error = parse_deep(source).expect_err("invalid defsig arity must reject");
        let rendered = error.to_string();
        assert!(
            rendered.contains(&format!(
                "wrong child count for `defsig`: expected Range(2, 3), got {actual}"
            )),
            "arity must preempt child decoding for `{source}`: {rendered}"
        );
        assert!(
            !rendered.contains("undecodable type head"),
            "a child-role diagnostic must not mask the defsig arity: {rendered}"
        );
    }
}

#[test]
fn declared_but_unused_unbounded_binders_are_preserved() {
    for source in [
        "def constant[a]() -> i32 = 1i32",
        "sig constant[a]: i32 -> i32\ndef constant(x) = x",
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
