//! Acceptance oracle for chelis#1030, the AST-backed resugaring foundation.
//!
//! These tests are derived from the active Surf and Deep contracts:
//! `spec/02-surf-syntax.md` gives `cast(e, p)` and `par { e1; e2 }` as
//! the Surf spellings, while `spec/03-deep-syntax.md` defines the matching
//! `cast` and `par` nodes.  Resugaring must preserve those node classes and
//! malformed Deep must fail rather than turn into a placeholder program.

use chelis_deep::ast::{Atom as DeepAtom, Metadata, UnknownFormData};
use chelis_deep::parser::parse_str as parse_deep;
use chelis_deep::printer::print_canonical;
use chelis_deep::{DeepTag, Expr as DeepExpr, Span};
use chelis_surf::decompile::{decompile_program, try_decompile_program};
use chelis_surf::desugar::desugar_program;
use chelis_surf::format::format_expression;
use chelis_surf::parser::parse_str as parse_surf;
use chelis_surf::resugar::resugar_expression;

fn parse_one_deep(source: &str) -> DeepExpr {
    let mut exprs = parse_deep(source).expect("Deep fixture parses");
    assert_eq!(exprs.len(), 1, "fixture must contain one Deep expression");
    exprs.remove(0)
}

fn redesugar_expression(source: &str) -> String {
    let decls = parse_surf(&format!("def result() = {source}"))
        .unwrap_or_else(|error| panic!("resugared Surf did not parse: {error}\n{source}"));
    print_canonical(&desugar_program(&decls).expect("Surf fixture must desugar"))
}

#[test]
fn cast_resugars_through_the_canonical_surf_ast_printer() {
    let deep = parse_one_deep("(cast {} (lit {} 1.0) (t-prim {} f32))");

    let surf_ast = resugar_expression(&deep).expect("valid Deep cast resugars");
    let surf = format_expression(&surf_ast);

    // A cast's literal operand keeps its suffix (spec/03 §6.3.1).
    assert_eq!(surf, "cast(1.0f32, f32)");
    let redesugared = redesugar_expression(&surf);
    assert!(
        redesugared.contains("(cast {"),
        "cast must re-desugar as a cast node:\n{redesugared}"
    );
    assert!(
        !redesugared.contains("(var {} as)"),
        "cast must not reparse as an application of `as`:\n{redesugared}"
    );
}

#[test]
fn cast_operand_resugars_structurally_instead_of_as_a_string_fragment() {
    let deep =
        parse_one_deep("(cast {} (app {} (var {} add) (var {} x) (lit {} 1.0)) (t-prim {} f32))");

    let surf_ast = resugar_expression(&deep).expect("cast over an application resugars");

    assert_eq!(format_expression(&surf_ast), "cast((x + 1.0), f32)");
}

#[test]
fn cast_preserves_literal_dtype_metadata_with_a_surface_suffix() {
    let deep = parse_one_deep("(cast {} (lit {type: (t-prim {} f64)} 1.25) (t-prim {} f32))");

    let surf_ast = resugar_expression(&deep).expect("typed literal resugars");
    let surf = format_expression(&surf_ast);

    assert_eq!(surf, "cast(1.25f64, f32)");
    let redesugared = redesugar_expression(&surf);
    assert!(
        redesugared.contains("type: (t-prim {} f64)"),
        "literal precision must survive the round trip:\n{redesugared}"
    );
}

#[test]
fn cast_preserves_default_literal_dtype_metadata_with_a_surface_suffix() {
    let deep = parse_one_deep("(cast {} (lit {type: (t-prim {} f32)} 1.25) (t-prim {} f64))");

    let surf_ast = resugar_expression(&deep).expect("typed f32 literal resugars");
    let surf = format_expression(&surf_ast);

    assert_eq!(surf, "cast(1.25f32, f64)");
    let redesugared = redesugar_expression(&surf);
    assert!(
        redesugared.contains("type: (t-prim {} f32)"),
        "default literal precision must survive the round trip:\n{redesugared}"
    );
}

#[test]
fn supported_expression_nodes_preserve_semantic_type_metadata() {
    for (deep_source, expected) in [
        ("(var {type: (t-prim {} f64)} x)", "(x : f64)"),
        (
            "(app {type: (t-prim {} f64)} (var {} f) (var {} x))",
            "(f(x) : f64)",
        ),
    ] {
        let deep = parse_one_deep(deep_source);
        let surf = format_expression(
            &resugar_expression(&deep).expect("typed supported expression resugars"),
        );

        assert_eq!(surf, expected);
        let redesugared = redesugar_expression(&surf);
        assert!(
            redesugared.contains("type: (t-prim {} f64)"),
            "semantic expression type must survive the round trip:\n{redesugared}"
        );
    }
}

#[test]
fn semantic_type_metadata_is_preserved_or_fails_explicitly() {
    let deep = parse_one_deep("(var {type: (t-adt {} Option (t-prim {} f64))} x)");

    match resugar_expression(&deep) {
        Ok(surface) => {
            let surf = format_expression(&surface);
            assert_eq!(surf, "(x : Option[f64])");
            let redesugared = redesugar_expression(&surf);
            assert!(
                redesugared.contains("type: (t-adt {} Option (t-prim {} f64))"),
                "supported semantic metadata must survive the round trip: {redesugared}"
            );
        }
        Err(error) => assert!(
            error
                .to_string()
                .contains("`t-adt` is outside the tested resugaring foundation"),
            "unsupported semantic metadata must identify the unimplemented type form: {error}"
        ),
    }
}

#[test]
fn non_numeric_literal_type_metadata_resugars_when_compatible() {
    for (deep_source, expected) in [
        ("(lit {type: (t-prim {} bool)} true)", "true"),
        ("(lit {type: (t-prim {} string)} \"text\")", "\"text\""),
        ("(lit {type: (t-unit {})} ())", "()"),
    ] {
        let deep = parse_one_deep(deep_source);
        let surf = format_expression(
            &resugar_expression(&deep).expect("compatible non-numeric literal resugars"),
        );

        assert_eq!(surf, expected);
        redesugar_expression(&surf);
    }
}

#[test]
fn incompatible_literal_type_metadata_is_rejected() {
    let deep = parse_one_deep("(lit {type: (t-prim {} bool)} 1)");

    let error = resugar_expression(&deep).expect_err("integer is not a bool literal");

    let diagnostic = error.to_string();
    assert!(
        diagnostic.contains("canonical atom/primitive pairing"),
        "diagnostic should identify the literal/type mismatch: {error}"
    );
}

#[test]
fn int64_minimum_resugars_to_parseable_typed_surf() {
    let deep = parse_one_deep("(lit {type: (t-prim {} i64)} -9223372036854775808)");

    let surf = format_expression(&resugar_expression(&deep).expect("i64 minimum resugars"));

    assert_eq!(surf, "-9223372036854775808i64");
    let redesugared = redesugar_expression(&surf);
    assert!(
        redesugared.contains("type: (t-prim {} i64)"),
        "the typed minimum must survive the round trip:\n{redesugared}"
    );
}

#[test]
fn names_that_canonical_surf_cannot_parse_are_rejected() {
    for deep_source in ["(var {} if)", "(var {} bad-name)", "(var {} Upper.lower)"] {
        let deep = parse_one_deep(deep_source);

        let error = resugar_expression(&deep)
            .expect_err("a successful resugar must never print an invalid Surf name");

        assert!(
            error.to_string().contains("not a valid Surf"),
            "diagnostic should identify the invalid name: {error}"
        );
    }
}

#[test]
fn par_resugars_through_the_canonical_surf_ast_printer() {
    let deep = parse_one_deep("(par {} (lit {} 1) (lit {} 2))");

    let surf_ast = resugar_expression(&deep).expect("valid Deep par resugars");
    let surf = format_expression(&surf_ast);

    assert_eq!(surf, "par { 1; 2 }");
    let redesugared = redesugar_expression(&surf);
    assert!(
        redesugared.contains("(par {"),
        "par must re-desugar as a par node:\n{redesugared}"
    );
    assert_eq!(
        redesugared.matches("(lit {").count(),
        2,
        "both par tasks must survive the round trip:\n{redesugared}"
    );
}

#[test]
fn public_decompiler_uses_the_shared_ast_spelling_for_cast_and_par() {
    // Suffixed operands keep their cast nodes (spec/04 §5.6).
    let source = "def result() = par { cast(1.0f64, f32); cast(2.0f64, f32) }";
    let decls = parse_surf(source).expect("canonical Surf fixture parses");
    let deep = desugar_program(&decls).expect("Surf fixture must desugar");

    let emitted = decompile_program(&deep);

    parse_surf(&emitted)
        .unwrap_or_else(|error| panic!("decompiler emitted invalid Surf: {error}\n{emitted}"));
    assert!(
        emitted.contains("par {"),
        "decompiler must use the Surf par form:\n{emitted}"
    );
    assert_eq!(
        emitted.matches("cast(").count(),
        2,
        "both casts must use the canonical call-like form:\n{emitted}"
    );
    assert!(
        !emitted.contains(" as "),
        "decompiler must not invent the rejected cast alias:\n{emitted}"
    );
}

#[test]
fn malformed_cast_is_rejected_before_resugaring_node_construction() {
    let error = parse_deep("(cast {} (lit {} 1.0))")
        .expect_err("a one-child cast cannot become a stamped Deep node");

    assert!(
        error.to_string().contains("wrong child count for `cast`")
            && error.to_string().contains("expected Range(2, 3), got 1"),
        "diagnostic should identify the violated Deep shape: {error}"
    );
}

#[test]
fn public_decompiler_propagates_malformed_foundation_nodes() {
    let span = Span::new(0, 0);
    let malformed = DeepExpr::UnknownForm(Box::new(UnknownFormData {
        head: "future-form".to_string(),
        meta: Metadata::default(),
        children: vec![DeepExpr::Atom(DeepAtom::Name("value".to_string()), span)],
        span,
    }));

    let error = try_decompile_program(std::slice::from_ref(&malformed))
        .expect_err("fallible public boundary must propagate an unknown Deep form");
    assert!(error.to_string().contains("unknown form `future-form`"));

    let displayed = decompile_program(std::slice::from_ref(&malformed));
    assert!(
        displayed.starts_with("-- Deep resugaring error:"),
        "display wrapper must expose the error: {displayed}"
    );
    assert!(
        !displayed.contains("= ()"),
        "malformed Deep must never become a placeholder program: {displayed}"
    );
}

#[test]
fn fallible_program_boundary_rejects_invalid_declaration_without_expression_fallback() {
    let malformed = parse_one_deep("(def {} result (var {} if))");

    let error = try_decompile_program(std::slice::from_ref(&malformed))
        .expect_err("fallible public boundary must not return unparsable Surf");

    assert!(
        error.to_string().contains("valid Surf value identifier"),
        "diagnostic must preserve the invalid declaration's actual cause: {error}"
    );
}

#[test]
fn constructed_non_finite_deep_float_fails_loudly() {
    let span = Span::new(0, 0);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let deep = DeepExpr::node(
            DeepTag::Lit,
            Metadata::default(),
            vec![DeepExpr::Atom(DeepAtom::Float(value), span)],
            span,
        );

        let error = resugar_expression(&deep)
            .expect_err("non-finite Deep floats have no canonical Surf literal");

        assert!(
            error.to_string().contains("finite float literal"),
            "diagnostic must identify the unrepresentable float: {error}"
        );
    }
}

#[test]
fn empty_par_is_an_error_not_an_empty_surface_block() {
    let error = parse_deep("(par {})").expect_err("an empty par cannot become a stamped node");

    assert!(
        error.to_string().contains("wrong child count for `par`")
            && error.to_string().contains("expected AtLeast(1), got 0"),
        "diagnostic should identify the violated Deep shape: {error}"
    );
}

#[test]
fn structural_bare_lists_fail_closed_in_expression_position() {
    let span = Span::new(0, 0);
    let bare = DeepExpr::BareList(
        vec![DeepExpr::Atom(DeepAtom::Name("value".to_string()), span)],
        span,
    );

    let error = resugar_expression(&bare)
        .expect_err("a structural list cannot be reinterpreted as a runtime expression");
    assert!(
        error.to_string().contains("structural bare list"),
        "diagnostic must preserve the stamped structural class: {error}"
    );
}

#[test]
fn non_finite_deep_literals_are_explicitly_unrepresentable_in_surf() {
    // Textual overflow is rejected at ingress; exercise the resugar boundary
    // with programmatic Deep, including a literal in pattern position.
    let span = Span::new(0, 0);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let pattern = DeepExpr::node(
            DeepTag::PatLit,
            Metadata::default(),
            vec![DeepExpr::Atom(DeepAtom::Float(value), span)],
            span,
        );
        let arm = DeepExpr::node(
            DeepTag::Arm,
            Metadata::default(),
            vec![
                pattern,
                DeepExpr::BareList(vec![], span),
                parse_one_deep("(lit {} 1)"),
            ],
            span,
        );
        let deep = DeepExpr::node(
            DeepTag::Match,
            Metadata::default(),
            vec![parse_one_deep("(var {} x)"), arm],
            span,
        );
        let error = resugar_expression(&deep)
            .expect_err("non-finite Deep patterns have no canonical Surf literal");
        assert!(error.to_string().contains("non-finite float"), "{error}");
    }
}

#[test]
fn overflowing_deep_text_is_rejected_before_resugaring() {
    for source in [
        "(lit {} 1e400)",
        "(match {} (var {} x) (arm {} (pat-lit {} 1e400) () (lit {} 1)))",
    ] {
        let error = parse_deep(source).expect_err("overflowing source number must reject");
        assert!(
            error.to_string().contains("invalid number") && error.to_string().contains("1e400"),
            "diagnostic must identify source-number overflow: {error}"
        );
    }
}

#[test]
fn declaration_tags_fail_when_requested_as_expressions() {
    let deep = parse_one_deep("(def {} f (var {} x))");

    let error = resugar_expression(&deep).expect_err("def is not an expression");

    assert!(
        error
            .to_string()
            .contains("`def` does not have a Surf representation in expression position"),
        "contextual tags must fail explicitly: {error}"
    );
}

#[test]
fn gt_application_resugars_to_the_greater_than_operator_in_order() {
    // chelis#1180: `a > b` desugars to `(app {} (var {} gt) a' b')`, so the
    // resugarer must print that application back as the operator form with
    // the operand order preserved, closing the bidirectional laws for `>`.
    let deep = parse_one_deep("(app {} (var {} gt) (var {} a) (var {} b))");

    let surf_ast = resugar_expression(&deep).expect("gt application resugars");
    let surf = format_expression(&surf_ast);

    assert_eq!(surf, "(a > b)");
    let redesugared = redesugar_expression(&surf);
    let gt_offset = redesugared
        .find("(var {} gt)")
        .expect("`a > b` must re-desugar through the gt built-in");
    let a_offset = redesugared.find("} a)").expect("left operand present");
    let b_offset = redesugared.find("} b)").expect("right operand present");
    assert!(
        gt_offset < a_offset && a_offset < b_offset,
        "gt application must keep the authored operand order:\n{redesugared}"
    );
    assert!(
        !redesugared.contains("cmplt"),
        "`a > b` must not re-desugar through operand-swapped cmplt:\n{redesugared}"
    );
}

#[test]
fn swapped_cmplt_still_resugars_faithfully_as_less_than() {
    // Hand-written Deep may still spell `cmplt(b, a)`; the resugarer keeps
    // printing that faithfully as `b < a` rather than guessing `a > b`.
    let deep = parse_one_deep("(app {} (var {} cmplt) (var {} b) (var {} a))");

    let surf_ast = resugar_expression(&deep).expect("cmplt application resugars");

    assert_eq!(format_expression(&surf_ast), "(b < a)");
}
