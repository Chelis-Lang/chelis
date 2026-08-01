//! Acceptance oracle for chelis#1030, the AST-backed resugaring foundation.
//!
//! These tests are derived from the active Surf and Deep contracts:
//! `spec/02-surf-syntax.md` gives `cast(e, p)` and `par { e1; e2 }` as
//! the Surf spellings, while `spec/03-deep-syntax.md` defines the matching
//! `cast` and `par` nodes.  Resugaring must preserve those node classes and
//! malformed Deep must fail rather than turn into a placeholder program.

use chelis_deep::Expr as DeepExpr;
use chelis_deep::parser::parse_str as parse_deep;
use chelis_deep::printer::print_canonical;
use chelis_surf::decompile::decompile_program;
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
    print_canonical(&desugar_program(&decls))
}

#[test]
fn cast_resugars_through_the_canonical_surf_ast_printer() {
    let deep = parse_one_deep("(cast {} (lit {} 1.0) (t-prim {} f32))");

    let surf_ast = resugar_expression(&deep).expect("valid Deep cast resugars");
    let surf = format_expression(&surf_ast);

    assert_eq!(surf, "cast(1.0, f32)");
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

    assert_eq!(format_expression(&surf_ast), "cast(add(x, 1.0), f32)");
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
    let source = "def result() = par { cast(1.0, f32); cast(2.0, f32) }";
    let decls = parse_surf(source).expect("canonical Surf fixture parses");
    let deep = desugar_program(&decls);

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
fn malformed_cast_is_an_error_not_a_placeholder() {
    let malformed = parse_one_deep("(cast {} (lit {} 1.0))");

    let error = resugar_expression(&malformed).expect_err("one-child cast is malformed");

    assert!(
        error
            .to_string()
            .contains("`cast` expects exactly 2 children"),
        "diagnostic should identify the violated Deep shape: {error}"
    );
}

#[test]
fn empty_par_is_an_error_not_an_empty_surface_block() {
    let malformed = parse_one_deep("(par {})");

    let error = resugar_expression(&malformed).expect_err("Deep par requires a task");

    assert!(
        error.to_string().contains("`par` expects at least 1 child"),
        "diagnostic should identify the violated Deep shape: {error}"
    );
}

#[test]
fn unimplemented_tags_fail_instead_of_falling_back_to_call_syntax() {
    let deep = parse_one_deep("(fn {} (params {} x) (var {} x))");

    let error = resugar_expression(&deep).expect_err("fn is outside the foundation slice");

    assert!(
        error
            .to_string()
            .contains("`fn` is outside the tested resugaring foundation"),
        "unsupported tags must fail explicitly: {error}"
    );
}
