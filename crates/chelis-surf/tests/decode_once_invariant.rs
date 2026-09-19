//! PERMANENT decode-once invariant, desugar side (chelis#731 Phase 3):
//! the desugarer and macro expander are typed producers, so their output
//! never carries a closed-vocabulary tag as a raw element-0 string. A
//! violation here means a producer bypassed the typed constructors and
//! its nodes would silently miss every typed dispatch arm. The parser
//! side of the same invariant lives in chelis-deep's validate tests; the
//! macro expander splices parsed (already-stamped) subtrees and builds
//! through the same typed constructors, covered by its own suite.

use chelis_deep::validate::find_raw_vocabulary_tag;

fn desugared(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = chelis_surf::parser::parse_str(source).expect("surf parse");
    chelis_surf::desugar::desugar_program(&decls).expect("Surf fixture must desugar")
}

#[test]
fn desugared_trees_carry_no_raw_vocabulary_tag_strings() {
    let source = "module M.Main\n\
        sig f[n] : tensor[n, f32] -> tensor[n, f32] ! { Random }\n\
        def f(x: tensor[n, f32]) -> tensor[n, f32] = with seed(42i64) { relu(x) }\n\
        def g(c: bool, x: f32) -> f32 = if c then x else neg(x)\n\
        def h(t: (f32, f32)) -> f32 = t.0\n\
        out = print(g(true, 1.5))\n";
    let exprs = desugared(source);
    assert_eq!(
        find_raw_vocabulary_tag(&exprs),
        None,
        "desugar/macro output must stamp every vocabulary tag"
    );
}

#[test]
fn desugared_match_and_adt_trees_carry_no_raw_vocabulary_tag_strings() {
    let source = "module M.Main\n\
        type Shape = | Circle { radius: f32 } | Square { side: f32 }\n\
        def area(s: Shape) -> f32 = match s with {\n\
          | Circle(r) => mul(r, r)\n\
          | Square(d) => mul(d, d)\n\
        }\n";
    let exprs = desugared(source);
    assert_eq!(
        find_raw_vocabulary_tag(&exprs),
        None,
        "pattern/ADT desugar output must stamp every vocabulary tag"
    );
}
