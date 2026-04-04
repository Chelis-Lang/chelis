//! Integration tests: Surf parse → desugar → Deep print → Deep parse roundtrip.

use chelis_deep::parser::parse_str as deep_parse;
use chelis_deep::printer::print_canonical;
use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str as surf_parse;

/// Parse Surf source, desugar to Deep, print Deep, re-parse Deep.
/// Verifies the generated Deep is valid and round-trips through the Deep parser.
fn roundtrip(surf_source: &str) {
    // 1. Parse Surf
    let decls = surf_parse(surf_source).unwrap_or_else(|e| {
        panic!("Surf parse failed for:\n{surf_source}\nError: {e}");
    });

    // 2. Desugar to Deep
    let deep_exprs = desugar_program(&decls);
    assert!(
        !deep_exprs.is_empty(),
        "Desugarer produced no output for:\n{surf_source}"
    );

    // 3. Print Deep
    let deep_text = print_canonical(&deep_exprs);
    assert!(
        !deep_text.is_empty(),
        "Deep printer produced empty output for:\n{surf_source}"
    );

    // 4. Re-parse Deep — this is the key check
    let reparsed = deep_parse(&deep_text).unwrap_or_else(|e| {
        panic!(
            "Deep re-parse failed.\nOriginal Surf:\n{surf_source}\nDeep output:\n{deep_text}\nError: {e}"
        );
    });

    // 5. Re-print and compare (Deep round-trip stability)
    let reprinted = print_canonical(&reparsed);
    assert_eq!(
        deep_text, reprinted,
        "Deep round-trip mismatch.\nFirst print:\n{deep_text}\nSecond print:\n{reprinted}"
    );
}

// === Fixture roundtrips ===

#[test]
fn roundtrip_simple_def() {
    roundtrip(include_str!("fixtures/simple_def.ch"));
}

#[test]
fn roundtrip_untyped_def() {
    roundtrip(include_str!("fixtures/untyped_def.ch"));
}

#[test]
fn roundtrip_operators() {
    roundtrip(include_str!("fixtures/operators.ch"));
}

#[test]
fn roundtrip_adt() {
    roundtrip(include_str!("fixtures/adt.ch"));
}

#[test]
fn roundtrip_let_expr() {
    roundtrip(include_str!("fixtures/let_expr.ch"));
}

#[test]
fn roundtrip_lambda() {
    roundtrip(include_str!("fixtures/lambda.ch"));
}

#[test]
fn roundtrip_pipe() {
    roundtrip(include_str!("fixtures/pipe.ch"));
}

#[test]
fn roundtrip_match() {
    roundtrip(include_str!("fixtures/match_expr.ch"));
}

// === Specific output shape checks ===

#[test]
fn typed_def_emits_defsig() {
    let decls = surf_parse("def f(x: f32): f32 = x").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(text.contains("(defsig {} f"), "Missing defsig in:\n{text}");
    assert!(text.contains("(def {} f"), "Missing def in:\n{text}");
}

#[test]
fn type_var_not_prim() {
    let decls = surf_parse("type Option a = | Some a | None").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(t-var {} a)"),
        "Expected t-var for type param 'a', got:\n{text}"
    );
}

#[test]
fn annotation_in_metadata() {
    let decls = surf_parse("def f(x) = x : f32").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("{type: (t-prim {} f32)}"),
        "Expected type annotation in metadata, got:\n{text}"
    );
}

#[test]
fn pat_lit_raw_value() {
    let decls =
        surf_parse("type B = | T | F\ndef f(x: f32): f32 = match x with { | 0 -> 1 | _ -> x }")
            .unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        text.contains("(pat-lit {} 0)"),
        "Expected raw value in pat-lit, got:\n{text}"
    );
}

#[test]
fn no_legacy_tags() {
    // Verify no legacy tags appear in desugared output
    let decls = surf_parse("def f(x: f32): f32 = x + 1").unwrap();
    let deep = desugar_program(&decls);
    let text = print_canonical(&deep);
    assert!(
        !text.contains("(sig "),
        "Legacy 'sig' tag found in:\n{text}"
    );
    assert!(
        !text.contains("(apply "),
        "Legacy 'apply' tag found in:\n{text}"
    );
    assert!(
        !text.contains("(: "),
        "Legacy ':' annotation tag found in:\n{text}"
    );
}
